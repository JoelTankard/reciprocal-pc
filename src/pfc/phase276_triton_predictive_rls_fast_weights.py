from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


RANK = tl.constexpr(32)


@triton.jit
def _predictive_rls_reads_kernel(
    keys,
    errors,
    reads,
    key_stride_batch: tl.constexpr,
    key_stride_sequence: tl.constexpr,
    key_stride_rank: tl.constexpr,
    read_stride_batch: tl.constexpr,
    read_stride_sequence: tl.constexpr,
    read_stride_scale: tl.constexpr,
    read_stride_rank: tl.constexpr,
    SEQUENCE: tl.constexpr,
    FAST_FORGETTING: tl.constexpr,
):
    program = tl.program_id(0)
    scale = program & 1
    batch = program >> 1
    channels = tl.arange(0, RANK)
    rows = channels[:, None]
    columns = channels[None, :]
    memory = tl.zeros((RANK, RANK), tl.float32)
    inverse_covariance = tl.where(rows == columns, 1.0, 0.0).to(tl.float32)
    forgetting = tl.where(
        scale == 0,
        FAST_FORGETTING,
        libdevice.sqrt(FAST_FORGETTING),
    )

    for step in tl.range(0, SEQUENCE, num_stages=1):
        if step > 0:
            key = tl.load(
                keys
                + batch * key_stride_batch
                + (step - 1) * key_stride_sequence
                + channels * key_stride_rank
            ).to(tl.float32)
            value = tl.load(
                errors
                + batch * key_stride_batch
                + step * key_stride_sequence
                + channels * key_stride_rank
            ).to(tl.float32)
            projected_key = tl.sum(inverse_covariance * key[None, :], axis=1)
            denominator = forgetting + tl.sum(key * projected_key, axis=0)
            gain = projected_key / denominator
            predicted = tl.sum(key[:, None] * memory, axis=0)
            residual = value - predicted
            memory = memory + gain[:, None] * residual[None, :]
            inverse_covariance = (
                inverse_covariance - gain[:, None] * projected_key[None, :]
            ) / forgetting
        query = tl.load(
            keys
            + batch * key_stride_batch
            + step * key_stride_sequence
            + channels * key_stride_rank
        ).to(tl.float32)
        read = tl.sum(query[:, None] * memory, axis=0)
        tl.store(
            reads
            + batch * read_stride_batch
            + step * read_stride_sequence
            + scale * read_stride_scale
            + channels * read_stride_rank,
            read,
        )


def triton_predictive_rls_reads_scales(keys: Tensor, errors: Tensor) -> Tensor:
    if keys.shape != errors.shape or keys.ndim != 3 or keys.shape[-1] != 32:
        raise ValueError("Phase276 keys/errors must be [batch, sequence, 32]")
    if keys.dtype != torch.float32 or errors.dtype != torch.float32:
        raise ValueError("Phase276 keys/errors must be float32")
    if not keys.is_cuda or not errors.is_cuda:
        raise ValueError("Phase276 RLS kernel requires CUDA")
    if not keys.is_contiguous() or not errors.is_contiguous():
        raise ValueError("Phase276 RLS kernel inputs must be contiguous")
    batch, sequence, _ = keys.shape
    reads = torch.empty(
        batch,
        sequence,
        2,
        32,
        device=keys.device,
        dtype=torch.float32,
    )
    _predictive_rls_reads_kernel[(batch * 2,)](
        keys,
        errors,
        reads,
        key_stride_batch=keys.stride(0),
        key_stride_sequence=keys.stride(1),
        key_stride_rank=keys.stride(2),
        read_stride_batch=reads.stride(0),
        read_stride_sequence=reads.stride(1),
        read_stride_scale=reads.stride(2),
        read_stride_rank=reads.stride(3),
        SEQUENCE=sequence,
        FAST_FORGETTING=0.99,
        num_warps=8,
    )
    return reads


def triton_predictive_rls_reads(keys: Tensor, errors: Tensor) -> Tensor:
    return triton_predictive_rls_reads_scales(keys, errors).mean(dim=2)


@triton.jit
def _predictive_rls_stateful_kernel(
    keys,
    errors,
    initial_memory,
    initial_inverse,
    previous_key,
    token_ids,
    codebook,
    previous_expected_code,
    reads,
    final_memory,
    final_inverse,
    update_norms,
    post_write_reads,
    key_stride_batch: tl.constexpr,
    key_stride_sequence: tl.constexpr,
    key_stride_rank: tl.constexpr,
    previous_key_stride_batch: tl.constexpr,
    token_stride_batch: tl.constexpr,
    code_stride_vocab: tl.constexpr,
    previous_expected_stride_batch: tl.constexpr,
    state_stride_batch: tl.constexpr,
    state_stride_scale: tl.constexpr,
    state_stride_row: tl.constexpr,
    state_stride_column: tl.constexpr,
    read_stride_batch: tl.constexpr,
    read_stride_sequence: tl.constexpr,
    read_stride_scale: tl.constexpr,
    read_stride_rank: tl.constexpr,
    update_stride_batch: tl.constexpr,
    update_stride_sequence: tl.constexpr,
    update_stride_scale: tl.constexpr,
    post_stride_batch: tl.constexpr,
    post_stride_sequence: tl.constexpr,
    post_stride_scale: tl.constexpr,
    post_stride_rank: tl.constexpr,
    SEQUENCE: tl.constexpr,
    FAST_FORGETTING: tl.constexpr,
    HAS_PREVIOUS: tl.constexpr,
    DECODE_TOKEN_ERROR: tl.constexpr,
    WRITE_DIAGNOSTICS: tl.constexpr,
):
    program = tl.program_id(0)
    scale = program & 1
    batch = program >> 1
    channels = tl.arange(0, RANK)
    rows = channels[:, None]
    columns = channels[None, :]
    state_offsets = (
        batch * state_stride_batch
        + scale * state_stride_scale
        + rows * state_stride_row
        + columns * state_stride_column
    )
    if HAS_PREVIOUS:
        memory = tl.load(initial_memory + state_offsets).to(tl.float32)
        inverse_covariance = tl.load(initial_inverse + state_offsets).to(tl.float32)
    else:
        memory = tl.zeros((RANK, RANK), tl.float32)
        inverse_covariance = tl.where(rows == columns, 1.0, 0.0).to(tl.float32)
    forgetting = tl.where(
        scale == 0,
        FAST_FORGETTING,
        libdevice.sqrt(FAST_FORGETTING),
    )

    for step in tl.range(0, SEQUENCE, num_stages=1):
        update_norm = 0.0
        post_write_read = tl.zeros((RANK,), tl.float32)
        if HAS_PREVIOUS or step > 0:
            if step == 0:
                key = tl.load(
                    previous_key
                    + batch * previous_key_stride_batch
                    + channels * key_stride_rank
                ).to(tl.float32)
            else:
                key = tl.load(
                    keys
                    + batch * key_stride_batch
                    + (step - 1) * key_stride_sequence
                    + channels * key_stride_rank
                ).to(tl.float32)
            if DECODE_TOKEN_ERROR:
                token = tl.load(token_ids + batch * token_stride_batch)
                value = tl.load(
                    codebook + token * code_stride_vocab + channels
                ).to(tl.float32) - tl.load(
                    previous_expected_code
                    + batch * previous_expected_stride_batch
                    + channels
                ).to(tl.float32)
            else:
                value = tl.load(
                    errors
                    + batch * key_stride_batch
                    + step * key_stride_sequence
                    + channels * key_stride_rank
                ).to(tl.float32)
            projected_key = tl.sum(inverse_covariance * key[None, :], axis=1)
            denominator = forgetting + tl.sum(key * projected_key, axis=0)
            gain = projected_key / denominator
            predicted = tl.sum(key[:, None] * memory, axis=0)
            residual = value - predicted
            memory = memory + gain[:, None] * residual[None, :]
            inverse_covariance = (
                inverse_covariance - gain[:, None] * projected_key[None, :]
            ) / forgetting
            update_norm = tl.sqrt(tl.sum(gain * gain, axis=0)) * tl.sqrt(
                tl.sum(residual * residual, axis=0)
            )
            post_write_read = tl.sum(key[:, None] * memory, axis=0)
        if WRITE_DIAGNOSTICS:
            tl.store(
                update_norms
                + batch * update_stride_batch
                + step * update_stride_sequence
                + scale * update_stride_scale,
                update_norm,
            )
            tl.store(
                post_write_reads
                + batch * post_stride_batch
                + step * post_stride_sequence
                + scale * post_stride_scale
                + channels * post_stride_rank,
                post_write_read,
            )
        query = tl.load(
            keys
            + batch * key_stride_batch
            + step * key_stride_sequence
            + channels * key_stride_rank
        ).to(tl.float32)
        read = tl.sum(query[:, None] * memory, axis=0)
        tl.store(
            reads
            + batch * read_stride_batch
            + step * read_stride_sequence
            + scale * read_stride_scale
            + channels * read_stride_rank,
            read,
        )
    tl.store(final_memory + state_offsets, memory)
    tl.store(final_inverse + state_offsets, inverse_covariance)


def _triton_predictive_rls_stateful_scales(
    keys: Tensor,
    errors: Tensor,
    initial_memory: Tensor,
    initial_inverse_covariance: Tensor,
    previous_key: Tensor,
    *,
    has_previous: bool,
    write_diagnostics: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if keys.shape != errors.shape or keys.ndim != 3 or keys.shape[-1] != 32:
        raise ValueError("Phase276 stateful keys/errors must be [batch, sequence, 32]")
    batch, sequence, _ = keys.shape
    expected_state_shape = (batch, 2, 32, 32)
    if initial_memory.shape != expected_state_shape:
        raise ValueError("Phase276 stateful memory shape differs")
    if initial_inverse_covariance.shape != expected_state_shape:
        raise ValueError("Phase276 stateful inverse shape differs")
    if previous_key.shape != (batch, 32):
        raise ValueError("Phase276 stateful previous key shape differs")
    tensors = (
        keys,
        errors,
        initial_memory,
        initial_inverse_covariance,
        previous_key,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("Phase276 stateful tensors require CUDA")
    if keys.dtype != torch.float32 or errors.dtype != torch.float32:
        raise ValueError("Phase276 stateful keys/errors must be float32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Phase276 stateful tensors must be contiguous")
    reads = torch.empty(
        batch, sequence, 2, 32, device=keys.device, dtype=torch.float32
    )
    final_memory = torch.empty_like(initial_memory)
    final_inverse = torch.empty_like(initial_inverse_covariance)
    if write_diagnostics:
        update_norms = torch.empty(
            batch,
            sequence,
            2,
            device=keys.device,
            dtype=torch.float32,
        )
        post_write_reads = torch.empty_like(reads)
    else:
        update_norms = torch.empty(1, device=keys.device, dtype=torch.float32)
        post_write_reads = update_norms
    _predictive_rls_stateful_kernel[(batch * 2,)](
        keys,
        errors,
        initial_memory,
        initial_inverse_covariance,
        previous_key,
        errors,
        errors,
        errors,
        reads,
        final_memory,
        final_inverse,
        update_norms,
        post_write_reads,
        key_stride_batch=keys.stride(0),
        key_stride_sequence=keys.stride(1),
        key_stride_rank=keys.stride(2),
        previous_key_stride_batch=previous_key.stride(0),
        token_stride_batch=0,
        code_stride_vocab=0,
        previous_expected_stride_batch=0,
        state_stride_batch=initial_memory.stride(0),
        state_stride_scale=initial_memory.stride(1),
        state_stride_row=initial_memory.stride(2),
        state_stride_column=initial_memory.stride(3),
        read_stride_batch=reads.stride(0),
        read_stride_sequence=reads.stride(1),
        read_stride_scale=reads.stride(2),
        read_stride_rank=reads.stride(3),
        update_stride_batch=(
            update_norms.stride(0) if write_diagnostics else 0
        ),
        update_stride_sequence=(
            update_norms.stride(1) if write_diagnostics else 0
        ),
        update_stride_scale=(
            update_norms.stride(2) if write_diagnostics else 0
        ),
        post_stride_batch=(
            post_write_reads.stride(0) if write_diagnostics else 0
        ),
        post_stride_sequence=(
            post_write_reads.stride(1) if write_diagnostics else 0
        ),
        post_stride_scale=(
            post_write_reads.stride(2) if write_diagnostics else 0
        ),
        post_stride_rank=(
            post_write_reads.stride(3) if write_diagnostics else 0
        ),
        SEQUENCE=sequence,
        FAST_FORGETTING=0.99,
        HAS_PREVIOUS=has_previous,
        DECODE_TOKEN_ERROR=False,
        WRITE_DIAGNOSTICS=write_diagnostics,
        num_warps=8,
    )
    return (
        reads,
        final_memory,
        final_inverse,
        update_norms,
        post_write_reads,
    )


def triton_predictive_rls_stateful(
    keys: Tensor,
    errors: Tensor,
    initial_memory: Tensor,
    initial_inverse_covariance: Tensor,
    previous_key: Tensor,
    *,
    has_previous: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    reads, final_memory, final_inverse, _, _ = (
        _triton_predictive_rls_stateful_scales(
            keys,
            errors,
            initial_memory,
            initial_inverse_covariance,
            previous_key,
            has_previous=has_previous,
        )
    )
    return reads.mean(dim=2), final_memory, final_inverse


def triton_predictive_rls_stateful_scales_with_diagnostics(
    keys: Tensor,
    errors: Tensor,
    initial_memory: Tensor,
    initial_inverse_covariance: Tensor,
    previous_key: Tensor,
    *,
    has_previous: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _triton_predictive_rls_stateful_scales(
        keys,
        errors,
        initial_memory,
        initial_inverse_covariance,
        previous_key,
        has_previous=has_previous,
        write_diagnostics=True,
    )


def triton_predictive_rls_stateful_scales(
    keys: Tensor,
    errors: Tensor,
    initial_memory: Tensor,
    initial_inverse_covariance: Tensor,
    previous_key: Tensor,
    *,
    has_previous: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    reads, final_memory, final_inverse, _, _ = (
        _triton_predictive_rls_stateful_scales(
            keys,
            errors,
            initial_memory,
            initial_inverse_covariance,
            previous_key,
            has_previous=has_previous,
        )
    )
    return reads, final_memory, final_inverse


@triton.jit
def _predictive_rls_decode_key_kernel(
    features,
    basis,
    keys,
    feature_stride_batch: tl.constexpr,
    basis_stride_width: tl.constexpr,
    key_stride_batch: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    batch = tl.program_id(0)
    width_offsets = tl.arange(0, BLOCK_WIDTH)
    channels = tl.arange(0, RANK)
    width_mask = width_offsets < WIDTH
    feature = tl.load(
        features + batch * feature_stride_batch + width_offsets,
        mask=width_mask,
        other=0.0,
    ).to(tl.float32)
    basis_values = tl.load(
        basis
        + width_offsets[:, None] * basis_stride_width
        + channels[None, :],
        mask=width_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    projected = tl.sum(feature[:, None] * basis_values, axis=0)
    norm = tl.sqrt(tl.sum(projected * projected, axis=0))
    normalized = projected / tl.maximum(norm, 1.0e-12)
    tl.store(keys + batch * key_stride_batch + channels, normalized)


def triton_predictive_rls_decode_key(features: Tensor, basis: Tensor) -> Tensor:
    if features.ndim != 2 or basis.ndim != 2:
        raise ValueError("Phase276 decode features/basis must be matrices")
    batch, width = features.shape
    if basis.shape != (width, 32):
        raise ValueError("Phase276 decode basis shape differs")
    if not features.is_cuda or not basis.is_cuda:
        raise ValueError("Phase276 decode key tensors require CUDA")
    if features.dtype != torch.float16 or basis.dtype != torch.float16:
        raise ValueError("Phase276 decode features/basis must be float16")
    if not features.is_contiguous() or not basis.is_contiguous():
        raise ValueError("Phase276 decode features/basis must be contiguous")
    keys = torch.empty(batch, 32, device=features.device, dtype=torch.float32)
    block_width = triton.next_power_of_2(width)
    _predictive_rls_decode_key_kernel[(batch,)](
        features,
        basis,
        keys,
        feature_stride_batch=features.stride(0),
        basis_stride_width=basis.stride(0),
        key_stride_batch=keys.stride(0),
        WIDTH=width,
        BLOCK_WIDTH=block_width,
        num_warps=8,
    )
    return keys


@triton.jit
def _token_conditioned_predictive_rls_decode_key_kernel(
    features,
    basis,
    gate_projection,
    keys,
    centered_gate,
    feature_stride_batch: tl.constexpr,
    basis_stride_width: tl.constexpr,
    gate_projection_stride_width: tl.constexpr,
    key_stride_batch: tl.constexpr,
    gate_stride_batch: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    batch = tl.program_id(0)
    width_offsets = tl.arange(0, BLOCK_WIDTH)
    channels = tl.arange(0, RANK)
    width_mask = width_offsets < WIDTH
    feature = tl.load(
        features + batch * feature_stride_batch + width_offsets,
        mask=width_mask,
        other=0.0,
    ).to(tl.float32)
    basis_values = tl.load(
        basis
        + width_offsets[:, None] * basis_stride_width
        + channels[None, :],
        mask=width_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    gate_projection_values = tl.load(
        gate_projection
        + width_offsets[:, None] * gate_projection_stride_width
        + channels[None, :],
        mask=width_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    projected = tl.sum(feature[:, None] * basis_values, axis=0)
    norm = tl.sqrt(tl.sum(projected * projected, axis=0))
    normalized = projected / tl.maximum(norm, 1.0e-12)
    preactivation = tl.sum(
        feature[:, None] * gate_projection_values,
        axis=0,
    )
    gate = 2.0 * tl.sigmoid(preactivation) - 1.0
    tl.store(keys + batch * key_stride_batch + channels, normalized)
    tl.store(centered_gate + batch * gate_stride_batch + channels, gate)


def triton_token_conditioned_predictive_rls_decode_key(
    features: Tensor,
    basis: Tensor,
    gate_projection: Tensor,
) -> tuple[Tensor, Tensor]:
    if features.ndim != 2 or basis.ndim != 2 or gate_projection.ndim != 2:
        raise ValueError("Phase494 decode feature/projection tensors must be matrices")
    batch, width = features.shape
    if basis.shape != (width, 32) or gate_projection.shape != (width, 32):
        raise ValueError("Phase494 decode projection shape differs")
    tensors = (features, basis, gate_projection)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("Phase494 decode key/gate tensors require CUDA")
    if any(tensor.dtype != torch.float16 for tensor in tensors):
        raise ValueError("Phase494 decode key/gate tensors must be float16")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Phase494 decode key/gate tensors must be contiguous")
    keys = torch.empty(batch, 32, device=features.device, dtype=torch.float32)
    centered_gate = torch.empty_like(keys)
    block_width = triton.next_power_of_2(width)
    _token_conditioned_predictive_rls_decode_key_kernel[(batch,)](
        features,
        basis,
        gate_projection,
        keys,
        centered_gate,
        feature_stride_batch=features.stride(0),
        basis_stride_width=basis.stride(0),
        gate_projection_stride_width=gate_projection.stride(0),
        key_stride_batch=keys.stride(0),
        gate_stride_batch=centered_gate.stride(0),
        WIDTH=width,
        BLOCK_WIDTH=block_width,
        num_warps=8,
    )
    return keys, centered_gate


def _triton_predictive_rls_stateful_decode_scales(
    keys: Tensor,
    token_ids: Tensor,
    codebook: Tensor,
    previous_expected_code: Tensor,
    initial_memory: Tensor,
    initial_inverse_covariance: Tensor,
    previous_key: Tensor,
    *,
    has_previous: bool,
    write_diagnostics: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if keys.ndim != 3 or keys.shape[1:] != (1, 32):
        raise ValueError("Phase276 decode keys must be [batch, 1, 32]")
    batch = keys.shape[0]
    if token_ids.shape != (batch,):
        raise ValueError("Phase276 decode token shape differs")
    if codebook.ndim != 2 or codebook.shape[1] != 32:
        raise ValueError("Phase276 decode codebook shape differs")
    if previous_expected_code.shape != (batch, 32):
        raise ValueError("Phase276 previous expected code shape differs")
    expected_state_shape = (batch, 2, 32, 32)
    if initial_memory.shape != expected_state_shape:
        raise ValueError("Phase276 decode memory shape differs")
    if initial_inverse_covariance.shape != expected_state_shape:
        raise ValueError("Phase276 decode inverse shape differs")
    if previous_key.shape != (batch, 32):
        raise ValueError("Phase276 decode previous key shape differs")
    tensors = (
        keys,
        token_ids,
        codebook,
        previous_expected_code,
        initial_memory,
        initial_inverse_covariance,
        previous_key,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("Phase276 decode state tensors require CUDA")
    if keys.dtype != torch.float32:
        raise ValueError("Phase276 decode keys must be float32")
    if codebook.dtype != torch.float16:
        raise ValueError("Phase276 decode codebook must be float16")
    if previous_expected_code.dtype != torch.float32:
        raise ValueError("Phase276 previous expected code must be float32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Phase276 decode state tensors must be contiguous")
    reads = torch.empty(batch, 1, 2, 32, device=keys.device, dtype=torch.float32)
    final_memory = torch.empty_like(initial_memory)
    final_inverse = torch.empty_like(initial_inverse_covariance)
    if write_diagnostics:
        update_norms = torch.empty(
            batch, 1, 2, device=keys.device, dtype=torch.float32
        )
        post_write_reads = torch.empty_like(reads)
    else:
        update_norms = torch.empty(1, device=keys.device, dtype=torch.float32)
        post_write_reads = update_norms
    _predictive_rls_stateful_kernel[(batch * 2,)](
        keys,
        keys,
        initial_memory,
        initial_inverse_covariance,
        previous_key,
        token_ids,
        codebook,
        previous_expected_code,
        reads,
        final_memory,
        final_inverse,
        update_norms,
        post_write_reads,
        key_stride_batch=keys.stride(0),
        key_stride_sequence=keys.stride(1),
        key_stride_rank=keys.stride(2),
        previous_key_stride_batch=previous_key.stride(0),
        token_stride_batch=token_ids.stride(0),
        code_stride_vocab=codebook.stride(0),
        previous_expected_stride_batch=previous_expected_code.stride(0),
        state_stride_batch=initial_memory.stride(0),
        state_stride_scale=initial_memory.stride(1),
        state_stride_row=initial_memory.stride(2),
        state_stride_column=initial_memory.stride(3),
        read_stride_batch=reads.stride(0),
        read_stride_sequence=reads.stride(1),
        read_stride_scale=reads.stride(2),
        read_stride_rank=reads.stride(3),
        update_stride_batch=(
            update_norms.stride(0) if write_diagnostics else 0
        ),
        update_stride_sequence=(
            update_norms.stride(1) if write_diagnostics else 0
        ),
        update_stride_scale=(
            update_norms.stride(2) if write_diagnostics else 0
        ),
        post_stride_batch=(
            post_write_reads.stride(0) if write_diagnostics else 0
        ),
        post_stride_sequence=(
            post_write_reads.stride(1) if write_diagnostics else 0
        ),
        post_stride_scale=(
            post_write_reads.stride(2) if write_diagnostics else 0
        ),
        post_stride_rank=(
            post_write_reads.stride(3) if write_diagnostics else 0
        ),
        SEQUENCE=1,
        FAST_FORGETTING=0.99,
        HAS_PREVIOUS=has_previous,
        DECODE_TOKEN_ERROR=True,
        WRITE_DIAGNOSTICS=write_diagnostics,
        num_warps=8,
    )
    return (
        reads,
        final_memory,
        final_inverse,
        update_norms,
        post_write_reads,
    )


def triton_predictive_rls_stateful_decode_scales(
    keys: Tensor,
    token_ids: Tensor,
    codebook: Tensor,
    previous_expected_code: Tensor,
    initial_memory: Tensor,
    initial_inverse_covariance: Tensor,
    previous_key: Tensor,
    *,
    has_previous: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    reads, final_memory, final_inverse, _, _ = (
        _triton_predictive_rls_stateful_decode_scales(
            keys,
            token_ids,
            codebook,
            previous_expected_code,
            initial_memory,
            initial_inverse_covariance,
            previous_key,
            has_previous=has_previous,
        )
    )
    return reads, final_memory, final_inverse


def triton_predictive_rls_stateful_decode_scales_with_diagnostics(
    keys: Tensor,
    token_ids: Tensor,
    codebook: Tensor,
    previous_expected_code: Tensor,
    initial_memory: Tensor,
    initial_inverse_covariance: Tensor,
    previous_key: Tensor,
    *,
    has_previous: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _triton_predictive_rls_stateful_decode_scales(
        keys,
        token_ids,
        codebook,
        previous_expected_code,
        initial_memory,
        initial_inverse_covariance,
        previous_key,
        has_previous=has_previous,
        write_diagnostics=True,
    )


@triton.jit
def _predictive_expected_code_kernel(
    logits,
    codebook,
    expected_code,
    logit_stride_batch: tl.constexpr,
    code_stride_vocab: tl.constexpr,
    expected_stride_batch: tl.constexpr,
    VOCAB: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
    BLOCK_RANK: tl.constexpr,
):
    batch = tl.program_id(0)
    rank_block = tl.program_id(1)
    vocabulary_offsets = tl.arange(0, BLOCK_VOCAB)
    channels = rank_block * BLOCK_RANK + tl.arange(0, BLOCK_RANK)
    running_max = -float("inf")
    running_denominator = 0.0
    running_projection = tl.zeros((BLOCK_RANK,), dtype=tl.float32)
    for start in tl.range(0, VOCAB, BLOCK_VOCAB, num_stages=1):
        vocabulary = start + vocabulary_offsets
        vocabulary_mask = vocabulary < VOCAB
        values = tl.load(
            logits + batch * logit_stride_batch + vocabulary,
            mask=vocabulary_mask,
            other=-float("inf"),
        ).to(tl.float32)
        block_max = tl.max(values, axis=0)
        next_max = tl.maximum(running_max, block_max)
        previous_weight = tl.exp(running_max - next_max)
        weights = tl.exp(values - next_max)
        codes = tl.load(
            codebook
            + vocabulary[:, None] * code_stride_vocab
            + channels[None, :],
            mask=vocabulary_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        running_projection = (
            running_projection * previous_weight
            + tl.sum(weights[:, None] * codes, axis=0)
        )
        running_denominator = (
            running_denominator * previous_weight + tl.sum(weights, axis=0)
        )
        running_max = next_max
    tl.store(
        expected_code + batch * expected_stride_batch + channels,
        running_projection / running_denominator,
    )


def triton_predictive_expected_code(logits: Tensor, codebook: Tensor) -> Tensor:
    if logits.ndim != 2 or codebook.ndim != 2:
        raise ValueError("Phase276 expected logits/codebook must be matrices")
    batch, vocab = logits.shape
    if codebook.shape != (vocab, 32):
        raise ValueError("Phase276 expected codebook shape differs")
    if not logits.is_cuda or not codebook.is_cuda:
        raise ValueError("Phase276 expected tensors require CUDA")
    if logits.dtype != torch.float16 or codebook.dtype != torch.float16:
        raise ValueError("Phase276 expected logits/codebook must be float16")
    if not logits.is_contiguous() or not codebook.is_contiguous():
        raise ValueError("Phase276 expected tensors must be contiguous")
    expected = torch.empty(batch, 32, device=logits.device, dtype=torch.float32)
    block_vocab = 256
    block_rank = 4
    _predictive_expected_code_kernel[(batch, triton.cdiv(32, block_rank))](
        logits,
        codebook,
        expected,
        logit_stride_batch=logits.stride(0),
        code_stride_vocab=codebook.stride(0),
        expected_stride_batch=expected.stride(0),
        VOCAB=vocab,
        BLOCK_VOCAB=block_vocab,
        BLOCK_RANK=block_rank,
        num_warps=4,
    )
    return expected


@triton.jit
def _predictive_rls_compose_kernel(
    base_logits,
    scale_reads,
    codebook,
    corrected_logits,
    base_stride_batch: tl.constexpr,
    read_stride_batch: tl.constexpr,
    read_stride_scale: tl.constexpr,
    code_stride_vocab: tl.constexpr,
    output_stride_batch: tl.constexpr,
    VOCAB: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
):
    batch = tl.program_id(0)
    vocabulary = tl.program_id(1) * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
    channels = tl.arange(0, RANK)
    mask = vocabulary < VOCAB
    fast_read = tl.load(
        scale_reads + batch * read_stride_batch + channels
    ).to(tl.float32)
    slow_read = tl.load(
        scale_reads
        + batch * read_stride_batch
        + read_stride_scale
        + channels
    ).to(tl.float32)
    mean_read = 0.5 * (fast_read + slow_read)
    codes = tl.load(
        codebook
        + vocabulary[:, None] * code_stride_vocab
        + channels[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    bias = tl.sum(codes * mean_read[None, :], axis=1)
    base = tl.load(
        base_logits + batch * base_stride_batch + vocabulary,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        corrected_logits + batch * output_stride_batch + vocabulary,
        base + bias,
        mask=mask,
    )


def triton_predictive_rls_compose(
    base_logits: Tensor,
    scale_reads: Tensor,
    codebook: Tensor,
) -> Tensor:
    if base_logits.ndim != 2 or codebook.ndim != 2:
        raise ValueError("Phase276 compose logits/codebook must be matrices")
    batch, vocab = base_logits.shape
    if scale_reads.shape != (batch, 2, 32):
        raise ValueError("Phase276 compose reads must be [batch, 2, 32]")
    if codebook.shape != (vocab, 32):
        raise ValueError("Phase276 compose codebook shape differs")
    tensors = (base_logits, scale_reads, codebook)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("Phase276 compose tensors require CUDA")
    if base_logits.dtype != torch.float16 or codebook.dtype != torch.float16:
        raise ValueError("Phase276 compose logits/codebook must be float16")
    if scale_reads.dtype != torch.float32:
        raise ValueError("Phase276 compose reads must be float32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Phase276 compose tensors must be contiguous")
    output = torch.empty(batch, vocab, device=base_logits.device, dtype=torch.float32)
    block_vocab = 128
    _predictive_rls_compose_kernel[(batch, triton.cdiv(vocab, block_vocab))](
        base_logits,
        scale_reads,
        codebook,
        output,
        base_stride_batch=base_logits.stride(0),
        read_stride_batch=scale_reads.stride(0),
        read_stride_scale=scale_reads.stride(1),
        code_stride_vocab=codebook.stride(0),
        output_stride_batch=output.stride(0),
        VOCAB=vocab,
        BLOCK_VOCAB=block_vocab,
        num_warps=4,
    )
    return output


@triton.jit
def _token_conditioned_predictive_rls_compose_kernel(
    base_logits,
    scale_reads,
    centered_gate,
    codebook,
    corrected_logits,
    base_stride_batch: tl.constexpr,
    read_stride_batch: tl.constexpr,
    read_stride_scale: tl.constexpr,
    gate_stride_batch: tl.constexpr,
    code_stride_vocab: tl.constexpr,
    output_stride_batch: tl.constexpr,
    VOCAB: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
    RMS_EPS: tl.constexpr,
):
    batch = tl.program_id(0)
    vocabulary = tl.program_id(1) * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
    channels = tl.arange(0, RANK)
    mask = vocabulary < VOCAB
    fast_read = tl.load(
        scale_reads + batch * read_stride_batch + channels
    ).to(tl.float32)
    slow_read = tl.load(
        scale_reads
        + batch * read_stride_batch
        + read_stride_scale
        + channels
    ).to(tl.float32)
    mean_read = 0.5 * (fast_read + slow_read)
    read_rms = tl.sqrt(tl.sum(mean_read * mean_read, axis=0) / RANK + RMS_EPS)
    gate = tl.load(
        centered_gate + batch * gate_stride_batch + channels
    ).to(tl.float32)
    candidate_read = mean_read + gate * (mean_read / read_rms)
    codes = tl.load(
        codebook
        + vocabulary[:, None] * code_stride_vocab
        + channels[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    bias = tl.sum(codes * candidate_read[None, :], axis=1)
    base = tl.load(
        base_logits + batch * base_stride_batch + vocabulary,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        corrected_logits + batch * output_stride_batch + vocabulary,
        base + bias,
        mask=mask,
    )


def triton_token_conditioned_predictive_rls_compose(
    base_logits: Tensor,
    scale_reads: Tensor,
    centered_gate: Tensor,
    codebook: Tensor,
    *,
    rms_eps: float,
) -> Tensor:
    if base_logits.ndim != 2 or codebook.ndim != 2:
        raise ValueError("Phase494 compose logits/codebook must be matrices")
    batch, vocab = base_logits.shape
    if scale_reads.shape != (batch, 2, 32):
        raise ValueError("Phase494 compose reads must be [batch,2,32]")
    if centered_gate.shape != (batch, 32):
        raise ValueError("Phase494 compose gate must be [batch,32]")
    if codebook.shape != (vocab, 32):
        raise ValueError("Phase494 compose codebook shape differs")
    if rms_eps <= 0:
        raise ValueError("Phase494 compose rms_eps must be positive")
    tensors = (base_logits, scale_reads, centered_gate, codebook)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("Phase494 compose tensors require CUDA")
    if base_logits.dtype != torch.float16 or codebook.dtype != torch.float16:
        raise ValueError("Phase494 compose logits/codebook must be float16")
    if scale_reads.dtype != torch.float32 or centered_gate.dtype != torch.float32:
        raise ValueError("Phase494 compose reads/gate must be float32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Phase494 compose tensors must be contiguous")
    output = torch.empty(batch, vocab, device=base_logits.device, dtype=torch.float32)
    block_vocab = 128
    _token_conditioned_predictive_rls_compose_kernel[
        (batch, triton.cdiv(vocab, block_vocab))
    ](
        base_logits,
        scale_reads,
        centered_gate,
        codebook,
        output,
        base_stride_batch=base_logits.stride(0),
        read_stride_batch=scale_reads.stride(0),
        read_stride_scale=scale_reads.stride(1),
        gate_stride_batch=centered_gate.stride(0),
        code_stride_vocab=codebook.stride(0),
        output_stride_batch=output.stride(0),
        VOCAB=vocab,
        BLOCK_VOCAB=block_vocab,
        RMS_EPS=rms_eps,
        num_warps=4,
    )
    return output


def triton_predictive_rls_decode(
    base_logits: Tensor,
    features: Tensor,
    basis: Tensor,
    token_ids: Tensor,
    codebook: Tensor,
    previous_expected_code: Tensor,
    initial_memory: Tensor,
    initial_inverse_covariance: Tensor,
    previous_key: Tensor,
    *,
    has_previous: bool,
    previous_expected_ready: torch.cuda.Event | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batch, width = features.shape
    vocab = base_logits.shape[1]
    expected_shapes = (
        base_logits.shape == (batch, vocab),
        basis.shape == (width, 32),
        token_ids.shape == (batch,),
        codebook.shape == (vocab, 32),
        previous_expected_code.shape == (batch, 32),
        initial_memory.shape == (batch, 2, 32, 32),
        initial_inverse_covariance.shape == (batch, 2, 32, 32),
        previous_key.shape == (batch, 32),
    )
    tensors = (
        base_logits,
        features,
        basis,
        token_ids,
        codebook,
        previous_expected_code,
        initial_memory,
        initial_inverse_covariance,
        previous_key,
    )
    if not all(expected_shapes):
        raise ValueError("Phase276 fused decode shape differs")
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Phase276 fused decode requires contiguous CUDA tensors")
    if (
        base_logits.dtype != torch.float16
        or features.dtype != torch.float16
        or basis.dtype != torch.float16
        or codebook.dtype != torch.float16
        or previous_expected_code.dtype != torch.float32
        or initial_memory.dtype != torch.float32
        or initial_inverse_covariance.dtype != torch.float32
        or previous_key.dtype != torch.float32
    ):
        raise ValueError("Phase276 fused decode dtype differs")

    keys = torch.empty(batch, 32, device=features.device, dtype=torch.float32)
    block_width = triton.next_power_of_2(width)
    _predictive_rls_decode_key_kernel[(batch,)](
        features,
        basis,
        keys,
        feature_stride_batch=features.stride(0),
        basis_stride_width=basis.stride(0),
        key_stride_batch=keys.stride(0),
        WIDTH=width,
        BLOCK_WIDTH=block_width,
        num_warps=8,
    )
    if previous_expected_ready is not None:
        stream = torch.cuda.current_stream(previous_expected_code.device)
        stream.wait_event(previous_expected_ready)
        previous_expected_code.record_stream(stream)

    sequence_keys = keys.unsqueeze(1)
    reads = torch.empty(
        batch, 1, 2, 32, device=features.device, dtype=torch.float32
    )
    final_memory = torch.empty_like(initial_memory)
    final_inverse = torch.empty_like(initial_inverse_covariance)
    unused_diagnostics = torch.empty(
        1, device=features.device, dtype=torch.float32
    )
    _predictive_rls_stateful_kernel[(batch * 2,)](
        sequence_keys,
        sequence_keys,
        initial_memory,
        initial_inverse_covariance,
        previous_key,
        token_ids,
        codebook,
        previous_expected_code,
        reads,
        final_memory,
        final_inverse,
        unused_diagnostics,
        unused_diagnostics,
        key_stride_batch=sequence_keys.stride(0),
        key_stride_sequence=sequence_keys.stride(1),
        key_stride_rank=sequence_keys.stride(2),
        previous_key_stride_batch=previous_key.stride(0),
        token_stride_batch=token_ids.stride(0),
        code_stride_vocab=codebook.stride(0),
        previous_expected_stride_batch=previous_expected_code.stride(0),
        state_stride_batch=initial_memory.stride(0),
        state_stride_scale=initial_memory.stride(1),
        state_stride_row=initial_memory.stride(2),
        state_stride_column=initial_memory.stride(3),
        read_stride_batch=reads.stride(0),
        read_stride_sequence=reads.stride(1),
        read_stride_scale=reads.stride(2),
        read_stride_rank=reads.stride(3),
        update_stride_batch=0,
        update_stride_sequence=0,
        update_stride_scale=0,
        post_stride_batch=0,
        post_stride_sequence=0,
        post_stride_scale=0,
        post_stride_rank=0,
        SEQUENCE=1,
        FAST_FORGETTING=0.99,
        HAS_PREVIOUS=has_previous,
        DECODE_TOKEN_ERROR=True,
        WRITE_DIAGNOSTICS=False,
        num_warps=8,
    )
    corrected = torch.empty(
        batch, vocab, device=base_logits.device, dtype=torch.float32
    )
    block_vocab = 128
    scale_reads = reads[:, 0]
    _predictive_rls_compose_kernel[(batch, triton.cdiv(vocab, block_vocab))](
        base_logits,
        scale_reads,
        codebook,
        corrected,
        base_stride_batch=base_logits.stride(0),
        read_stride_batch=scale_reads.stride(0),
        read_stride_scale=scale_reads.stride(1),
        code_stride_vocab=codebook.stride(0),
        output_stride_batch=corrected.stride(0),
        VOCAB=vocab,
        BLOCK_VOCAB=block_vocab,
        num_warps=4,
    )
    return corrected, final_memory, final_inverse, keys


def triton_token_conditioned_predictive_rls_decode(
    base_logits: Tensor,
    features: Tensor,
    basis: Tensor,
    gate_projection: Tensor,
    token_ids: Tensor,
    codebook: Tensor,
    previous_expected_code: Tensor,
    initial_memory: Tensor,
    initial_inverse_covariance: Tensor,
    previous_key: Tensor,
    *,
    has_previous: bool,
    previous_expected_ready: torch.cuda.Event | None,
    rms_eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch, width = features.shape
    vocab = base_logits.shape[1]
    expected_shapes = (
        base_logits.shape == (batch, vocab),
        basis.shape == (width, 32),
        gate_projection.shape == (width, 32),
        token_ids.shape == (batch,),
        codebook.shape == (vocab, 32),
        previous_expected_code.shape == (batch, 32),
        initial_memory.shape == (batch, 2, 32, 32),
        initial_inverse_covariance.shape == (batch, 2, 32, 32),
        previous_key.shape == (batch, 32),
    )
    tensors = (
        base_logits,
        features,
        basis,
        gate_projection,
        token_ids,
        codebook,
        previous_expected_code,
        initial_memory,
        initial_inverse_covariance,
        previous_key,
    )
    if not all(expected_shapes):
        raise ValueError("Phase494 fused decode shape differs")
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Phase494 fused decode requires contiguous CUDA tensors")
    if (
        base_logits.dtype != torch.float16
        or features.dtype != torch.float16
        or basis.dtype != torch.float16
        or gate_projection.dtype != torch.float16
        or codebook.dtype != torch.float16
        or previous_expected_code.dtype != torch.float32
        or initial_memory.dtype != torch.float32
        or initial_inverse_covariance.dtype != torch.float32
        or previous_key.dtype != torch.float32
    ):
        raise ValueError("Phase494 fused decode dtype differs")
    if rms_eps <= 0:
        raise ValueError("Phase494 fused decode rms_eps must be positive")

    keys, centered_gate = triton_token_conditioned_predictive_rls_decode_key(
        features,
        basis,
        gate_projection,
    )
    if previous_expected_ready is not None:
        stream = torch.cuda.current_stream(previous_expected_code.device)
        stream.wait_event(previous_expected_ready)
        previous_expected_code.record_stream(stream)

    sequence_keys = keys.unsqueeze(1)
    reads = torch.empty(
        batch, 1, 2, 32, device=features.device, dtype=torch.float32
    )
    final_memory = torch.empty_like(initial_memory)
    final_inverse = torch.empty_like(initial_inverse_covariance)
    unused_diagnostics = torch.empty(
        1, device=features.device, dtype=torch.float32
    )
    _predictive_rls_stateful_kernel[(batch * 2,)](
        sequence_keys,
        sequence_keys,
        initial_memory,
        initial_inverse_covariance,
        previous_key,
        token_ids,
        codebook,
        previous_expected_code,
        reads,
        final_memory,
        final_inverse,
        unused_diagnostics,
        unused_diagnostics,
        key_stride_batch=sequence_keys.stride(0),
        key_stride_sequence=sequence_keys.stride(1),
        key_stride_rank=sequence_keys.stride(2),
        previous_key_stride_batch=previous_key.stride(0),
        token_stride_batch=token_ids.stride(0),
        code_stride_vocab=codebook.stride(0),
        previous_expected_stride_batch=previous_expected_code.stride(0),
        state_stride_batch=initial_memory.stride(0),
        state_stride_scale=initial_memory.stride(1),
        state_stride_row=initial_memory.stride(2),
        state_stride_column=initial_memory.stride(3),
        read_stride_batch=reads.stride(0),
        read_stride_sequence=reads.stride(1),
        read_stride_scale=reads.stride(2),
        read_stride_rank=reads.stride(3),
        update_stride_batch=0,
        update_stride_sequence=0,
        update_stride_scale=0,
        post_stride_batch=0,
        post_stride_sequence=0,
        post_stride_scale=0,
        post_stride_rank=0,
        SEQUENCE=1,
        FAST_FORGETTING=0.99,
        HAS_PREVIOUS=has_previous,
        DECODE_TOKEN_ERROR=True,
        WRITE_DIAGNOSTICS=False,
        num_warps=8,
    )
    corrected = triton_token_conditioned_predictive_rls_compose(
        base_logits,
        reads[:, 0],
        centered_gate,
        codebook,
        rms_eps=rms_eps,
    )
    return corrected, final_memory, final_inverse, keys, centered_gate
