from __future__ import annotations

import torch
from torch import Tensor

from pfc.phase274_predictive_error_fast_weights import (
    PREDICTIVE_FAST_DECAY,
    PREDICTIVE_SLOW_DECAY,
)
from pfc.phase275_predictive_delta_fast_weights import (
    predictive_error_keys_and_values,
)


def _rls_update(
    memory: Tensor,
    inverse_covariance: Tensor,
    key: Tensor,
    value: Tensor,
    forgetting: float,
) -> tuple[Tensor, Tensor, Tensor]:
    projected_key = torch.einsum("brs,bs->br", inverse_covariance, key)
    denominator = forgetting + (key * projected_key).sum(dim=-1)
    gain = projected_key / denominator.unsqueeze(-1)
    residual = value - torch.einsum("br,brs->bs", key, memory)
    memory = memory + gain.unsqueeze(-1) * residual.unsqueeze(-2)
    inverse_covariance = (
        inverse_covariance
        - gain.unsqueeze(-1) * projected_key.unsqueeze(-2)
    ) / forgetting
    return memory, inverse_covariance, denominator


def predictive_rls_reads_reference(
    keys: Tensor,
    errors: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    if keys.shape != errors.shape or keys.ndim != 3:
        raise ValueError("keys and errors must share [batch, sequence, rank]")
    batch, sequence, rank = keys.shape
    fast_memory = torch.zeros(
        batch, rank, rank, device=keys.device, dtype=torch.float32
    )
    slow_memory = torch.zeros_like(fast_memory)
    identity = torch.eye(rank, device=keys.device, dtype=torch.float32)
    fast_inverse = identity.unsqueeze(0).expand(batch, -1, -1).clone()
    slow_inverse = fast_inverse.clone()
    minimum_denominator = keys.new_full((), float("inf"))
    reads = []
    scale_reads = []
    for position in range(sequence):
        if position > 0:
            key = keys[:, position - 1]
            value = errors[:, position]
            fast_memory, fast_inverse, fast_denominator = _rls_update(
                fast_memory,
                fast_inverse,
                key,
                value,
                PREDICTIVE_FAST_DECAY,
            )
            slow_memory, slow_inverse, slow_denominator = _rls_update(
                slow_memory,
                slow_inverse,
                key,
                value,
                PREDICTIVE_SLOW_DECAY,
            )
            minimum_denominator = torch.minimum(
                minimum_denominator,
                torch.minimum(fast_denominator.amin(), slow_denominator.amin()),
            )
        query = keys[:, position]
        fast_read = torch.einsum("br,brs->bs", query, fast_memory)
        slow_read = torch.einsum("br,brs->bs", query, slow_memory)
        scale_reads.append(torch.stack((fast_read, slow_read), dim=1))
        reads.append(0.5 * (fast_read + slow_read))
    return torch.stack(reads, dim=1), {
        "scale_reads": torch.stack(scale_reads, dim=1),
        "final_fast_memory": fast_memory,
        "final_slow_memory": slow_memory,
        "final_fast_inverse_covariance": fast_inverse,
        "final_slow_inverse_covariance": slow_inverse,
        "minimum_gain_denominator": minimum_denominator,
    }


def predictive_rls_reads_cuda(keys: Tensor, errors: Tensor) -> Tensor:
    if not keys.is_cuda or not errors.is_cuda:
        raise ValueError("Phase276 Triton RLS fast weights require CUDA")
    from pfc.phase276_triton_predictive_rls_fast_weights import (
        triton_predictive_rls_reads,
    )

    return triton_predictive_rls_reads(keys.contiguous(), errors.contiguous())


def predictive_rls_reads_cuda_scales(keys: Tensor, errors: Tensor) -> Tensor:
    if not keys.is_cuda or not errors.is_cuda:
        raise ValueError("Phase276 Triton RLS fast weights require CUDA")
    from pfc.phase276_triton_predictive_rls_fast_weights import (
        triton_predictive_rls_reads_scales,
    )

    return triton_predictive_rls_reads_scales(
        keys.contiguous(),
        errors.contiguous(),
    )


def predictive_rls_stateful_reference(
    keys: Tensor,
    errors: Tensor,
    initial_memory: Tensor,
    initial_inverse_covariance: Tensor,
    previous_key: Tensor,
    *,
    has_previous: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    if keys.shape != errors.shape or keys.ndim != 3 or keys.shape[-1] != 32:
        raise ValueError("Phase276 stateful keys/errors must be [batch, sequence, 32]")
    batch, sequence, rank = keys.shape
    expected_state_shape = (batch, 2, rank, rank)
    if initial_memory.shape != expected_state_shape:
        raise ValueError("Phase276 initial memory shape differs")
    if initial_inverse_covariance.shape != expected_state_shape:
        raise ValueError("Phase276 initial inverse covariance shape differs")
    if previous_key.shape != (batch, rank):
        raise ValueError("Phase276 previous key shape differs")
    memories = initial_memory.float().clone()
    inverses = initial_inverse_covariance.float().clone()
    forgetting = (PREDICTIVE_FAST_DECAY, PREDICTIVE_SLOW_DECAY)
    reads = []
    for position in range(sequence):
        if has_previous or position > 0:
            key = previous_key.float() if position == 0 else keys[:, position - 1]
            value = errors[:, position]
            for scale, decay in enumerate(forgetting):
                memory, inverse, _ = _rls_update(
                    memories[:, scale],
                    inverses[:, scale],
                    key,
                    value,
                    decay,
                )
                memories[:, scale] = memory
                inverses[:, scale] = inverse
        query = keys[:, position]
        scale_reads = torch.einsum("br,bsrt->bst", query, memories)
        reads.append(scale_reads.mean(dim=1))
    return (
        torch.stack(reads, dim=1),
        memories.to(initial_memory.dtype),
        inverses.to(initial_inverse_covariance.dtype),
    )


def predictive_rls_stateful_cuda(
    keys: Tensor,
    errors: Tensor,
    initial_memory: Tensor,
    initial_inverse_covariance: Tensor,
    previous_key: Tensor,
    *,
    has_previous: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    if not keys.is_cuda:
        raise ValueError("Phase276 stateful Triton RLS requires CUDA")
    from pfc.phase276_triton_predictive_rls_fast_weights import (
        triton_predictive_rls_stateful,
    )

    return triton_predictive_rls_stateful(
        keys.contiguous(),
        errors.contiguous(),
        initial_memory.contiguous(),
        initial_inverse_covariance.contiguous(),
        previous_key.contiguous(),
        has_previous=has_previous,
    )


def predictive_rls_stateful_cuda_scales(
    keys: Tensor,
    errors: Tensor,
    initial_memory: Tensor,
    initial_inverse_covariance: Tensor,
    previous_key: Tensor,
    *,
    has_previous: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    if not keys.is_cuda:
        raise ValueError("Phase276 stateful Triton RLS requires CUDA")
    from pfc.phase276_triton_predictive_rls_fast_weights import (
        triton_predictive_rls_stateful_scales,
    )

    return triton_predictive_rls_stateful_scales(
        keys.contiguous(),
        errors.contiguous(),
        initial_memory.contiguous(),
        initial_inverse_covariance.contiguous(),
        previous_key.contiguous(),
        has_previous=has_previous,
    )


def predictive_rls_stateful_cuda_scales_with_diagnostics(
    keys: Tensor,
    errors: Tensor,
    initial_memory: Tensor,
    initial_inverse_covariance: Tensor,
    previous_key: Tensor,
    *,
    has_previous: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if not keys.is_cuda:
        raise ValueError("Phase276 diagnostic Triton RLS requires CUDA")
    from pfc.phase276_triton_predictive_rls_fast_weights import (
        triton_predictive_rls_stateful_scales_with_diagnostics,
    )

    return triton_predictive_rls_stateful_scales_with_diagnostics(
        keys.contiguous(),
        errors.contiguous(),
        initial_memory.contiguous(),
        initial_inverse_covariance.contiguous(),
        previous_key.contiguous(),
        has_previous=has_previous,
    )


def predictive_rls_decode_key_cuda(features: Tensor, basis: Tensor) -> Tensor:
    if not features.is_cuda:
        raise ValueError("Phase276 Triton decode key requires CUDA")
    from pfc.phase276_triton_predictive_rls_fast_weights import (
        triton_predictive_rls_decode_key,
    )

    return triton_predictive_rls_decode_key(
        features.contiguous(), basis.contiguous()
    )


def predictive_rls_stateful_decode_scales_cuda(
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
    if not keys.is_cuda:
        raise ValueError("Phase276 stateful Triton decode RLS requires CUDA")
    from pfc.phase276_triton_predictive_rls_fast_weights import (
        triton_predictive_rls_stateful_decode_scales,
    )

    return triton_predictive_rls_stateful_decode_scales(
        keys.contiguous(),
        token_ids.contiguous(),
        codebook.contiguous(),
        previous_expected_code.contiguous(),
        initial_memory.contiguous(),
        initial_inverse_covariance.contiguous(),
        previous_key.contiguous(),
        has_previous=has_previous,
    )


def predictive_rls_stateful_decode_scales_cuda_with_diagnostics(
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
    if not keys.is_cuda:
        raise ValueError("Phase276 diagnostic Triton decode requires CUDA")
    from pfc.phase276_triton_predictive_rls_fast_weights import (
        triton_predictive_rls_stateful_decode_scales_with_diagnostics,
    )

    return triton_predictive_rls_stateful_decode_scales_with_diagnostics(
        keys.contiguous(),
        token_ids.contiguous(),
        codebook.contiguous(),
        previous_expected_code.contiguous(),
        initial_memory.contiguous(),
        initial_inverse_covariance.contiguous(),
        previous_key.contiguous(),
        has_previous=has_previous,
    )


def predictive_rls_compose_cuda(
    base_logits: Tensor,
    scale_reads: Tensor,
    codebook: Tensor,
) -> Tensor:
    if not base_logits.is_cuda:
        raise ValueError("Phase276 Triton vocabulary compose requires CUDA")
    from pfc.phase276_triton_predictive_rls_fast_weights import (
        triton_predictive_rls_compose,
    )

    return triton_predictive_rls_compose(
        base_logits.contiguous(),
        scale_reads.contiguous(),
        codebook.contiguous(),
    )


def predictive_rls_expected_code_cuda(
    base_logits: Tensor,
    codebook: Tensor,
) -> Tensor:
    if not base_logits.is_cuda:
        raise ValueError("Phase276 Triton expected code requires CUDA")
    from pfc.phase276_triton_predictive_rls_fast_weights import (
        triton_predictive_expected_code,
    )

    return triton_predictive_expected_code(
        base_logits.contiguous(), codebook.contiguous()
    )


def predictive_rls_decode_cuda(
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
    if not base_logits.is_cuda:
        raise ValueError("Phase276 fused decode requires CUDA")
    from pfc.phase276_triton_predictive_rls_fast_weights import (
        triton_predictive_rls_decode,
    )

    return triton_predictive_rls_decode(
        base_logits,
        features,
        basis,
        token_ids.contiguous(),
        codebook,
        previous_expected_code,
        initial_memory,
        initial_inverse_covariance,
        previous_key,
        has_previous=has_previous,
        previous_expected_ready=previous_expected_ready,
    )


def token_conditioned_predictive_rls_decode_cuda(
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
    if not base_logits.is_cuda:
        raise ValueError("Phase494 fused decode requires CUDA")
    from pfc.phase276_triton_predictive_rls_fast_weights import (
        triton_token_conditioned_predictive_rls_decode,
    )

    return triton_token_conditioned_predictive_rls_decode(
        base_logits,
        features,
        basis,
        gate_projection,
        token_ids.contiguous(),
        codebook,
        previous_expected_code,
        initial_memory,
        initial_inverse_covariance,
        previous_key,
        has_previous=has_previous,
        previous_expected_ready=previous_expected_ready,
        rms_eps=rms_eps,
    )


def predictive_rls_fast_weight_logits(
    base_logits: Tensor,
    prediction_features: Tensor,
    token_ids: Tensor,
    basis: Tensor,
    codebook: Tensor,
    *,
    force_reference: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    keys, errors = predictive_error_keys_and_values(
        base_logits,
        prediction_features,
        token_ids,
        basis,
        codebook,
    )
    if keys.is_cuda and not force_reference:
        read = predictive_rls_reads_cuda(keys, errors)
        reference_diagnostics = {
            "final_fast_memory": keys.new_empty(0),
            "final_slow_memory": keys.new_empty(0),
            "final_fast_inverse_covariance": keys.new_empty(0),
            "final_slow_inverse_covariance": keys.new_empty(0),
            "minimum_gain_denominator": keys.new_empty(0),
        }
    else:
        read, reference_diagnostics = predictive_rls_reads_reference(keys, errors)
    vocabulary_bias = read @ codebook.detach().float().T
    return base_logits.float() + vocabulary_bias, {
        "keys": keys,
        "errors": errors,
        "predictive_error_read": read,
        "vocabulary_bias": vocabulary_bias,
        **reference_diagnostics,
    }


def token_conditioned_predictive_rls_fast_weight_logits(
    base_logits: Tensor,
    prediction_features: Tensor,
    token_ids: Tensor,
    basis: Tensor,
    codebook: Tensor,
    gate_projection: Tensor,
    *,
    rms_eps: float,
    force_reference: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    from pfc.phase494_token_conditioned_predictive_read_gate import (
        token_conditioned_predictive_read,
    )

    keys, errors = predictive_error_keys_and_values(
        base_logits,
        prediction_features,
        token_ids,
        basis,
        codebook,
    )
    if keys.is_cuda and not force_reference:
        parent_read = predictive_rls_reads_cuda(keys, errors)
        reference_diagnostics = {
            "final_fast_memory": keys.new_empty(0),
            "final_slow_memory": keys.new_empty(0),
            "final_fast_inverse_covariance": keys.new_empty(0),
            "final_slow_inverse_covariance": keys.new_empty(0),
            "minimum_gain_denominator": keys.new_empty(0),
        }
    else:
        parent_read, reference_diagnostics = predictive_rls_reads_reference(
            keys,
            errors,
        )
    candidate_read, gate_diagnostics = token_conditioned_predictive_read(
        prediction_features,
        parent_read,
        gate_projection,
        rms_eps,
    )
    vocabulary_bias = candidate_read @ codebook.detach().float().T
    return base_logits.float() + vocabulary_bias, {
        "keys": keys,
        "errors": errors,
        "vocabulary_bias": vocabulary_bias,
        **gate_diagnostics,
        **reference_diagnostics,
    }
