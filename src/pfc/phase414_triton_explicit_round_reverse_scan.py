from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl


_profile_counters_enabled = False
_profile_scan_calls = 0
_profile_stage_launches = 0


def reset_scan_profile_counters(*, enabled: bool) -> None:
    global _profile_counters_enabled
    global _profile_scan_calls
    global _profile_stage_launches
    _profile_counters_enabled = enabled
    _profile_scan_calls = 0
    _profile_stage_launches = 0


def scan_profile_counters() -> tuple[int, int]:
    return _profile_scan_calls, _profile_stage_launches


@triton.jit
def _float_multiply(left, right):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _float_add(left, right):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _pair_scan_stage_kernel(
    matrix,
    write,
    output_matrix,
    output_write,
    total_units,
    offset,
    SEQUENCE: tl.constexpr,
    PAIRS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    units = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = units < total_units
    sequence_position = (units // PAIRS) % SEQUENCE
    has_previous = sequence_position >= offset
    previous_units = units - offset * PAIRS
    matrix_base = units * 4
    previous_matrix_base = previous_units * 4
    write_base = units * 2
    previous_write_base = previous_units * 2
    previous_mask = mask & has_previous

    l00 = tl.load(
        matrix + previous_matrix_base,
        mask=previous_mask,
        other=0.0,
    ).to(tl.float32)
    l01 = tl.load(
        matrix + previous_matrix_base + 1,
        mask=previous_mask,
        other=0.0,
    ).to(tl.float32)
    l10 = tl.load(
        matrix + previous_matrix_base + 2,
        mask=previous_mask,
        other=0.0,
    ).to(tl.float32)
    l11 = tl.load(
        matrix + previous_matrix_base + 3,
        mask=previous_mask,
        other=0.0,
    ).to(tl.float32)
    r00 = tl.load(matrix + matrix_base, mask=mask, other=0.0).to(tl.float32)
    r01 = tl.load(matrix + matrix_base + 1, mask=mask, other=0.0).to(
        tl.float32
    )
    r10 = tl.load(matrix + matrix_base + 2, mask=mask, other=0.0).to(
        tl.float32
    )
    r11 = tl.load(matrix + matrix_base + 3, mask=mask, other=0.0).to(
        tl.float32
    )
    m00 = _float_add(_float_multiply(l00, r00), _float_multiply(l01, r10))
    m01 = _float_add(_float_multiply(l00, r01), _float_multiply(l01, r11))
    m10 = _float_add(_float_multiply(l10, r00), _float_multiply(l11, r10))
    m11 = _float_add(_float_multiply(l10, r01), _float_multiply(l11, r11))

    v0 = tl.load(
        write + previous_write_base,
        mask=previous_mask,
        other=0.0,
    ).to(tl.float32)
    v1 = tl.load(
        write + previous_write_base + 1,
        mask=previous_mask,
        other=0.0,
    ).to(tl.float32)
    current_w0 = tl.load(write + write_base, mask=mask, other=0.0).to(
        tl.float32
    )
    current_w1 = tl.load(write + write_base + 1, mask=mask, other=0.0).to(
        tl.float32
    )
    composed_w0 = _float_add(
        _float_add(_float_multiply(v0, r00), _float_multiply(v1, r10)),
        current_w0,
    )
    composed_w1 = _float_add(
        _float_add(_float_multiply(v0, r01), _float_multiply(v1, r11)),
        current_w1,
    )

    tl.store(
        output_matrix + matrix_base,
        tl.where(has_previous, m00, r00),
        mask=mask,
    )
    tl.store(
        output_matrix + matrix_base + 1,
        tl.where(has_previous, m01, r01),
        mask=mask,
    )
    tl.store(
        output_matrix + matrix_base + 2,
        tl.where(has_previous, m10, r10),
        mask=mask,
    )
    tl.store(
        output_matrix + matrix_base + 3,
        tl.where(has_previous, m11, r11),
        mask=mask,
    )
    tl.store(
        output_write + write_base,
        tl.where(has_previous, composed_w0, current_w0),
        mask=mask,
    )
    tl.store(
        output_write + write_base + 1,
        tl.where(has_previous, composed_w1, current_w1),
        mask=mask,
    )


@triton.jit
def _pair_scan_final_kernel(
    prefix_matrix,
    prefix_write,
    initial,
    output,
    total_values,
    SEQUENCE: tl.constexpr,
    PAIRS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    indices = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < total_values
    unit = indices // 2
    column = indices % 2
    pair = unit % PAIRS
    group = unit // (SEQUENCE * PAIRS)
    initial_base = (group * PAIRS + pair) * 2
    matrix_base = unit * 4
    initial_0 = tl.load(
        initial + initial_base,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    initial_1 = tl.load(
        initial + initial_base + 1,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    matrix_0 = tl.load(
        prefix_matrix + matrix_base + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    matrix_1 = tl.load(
        prefix_matrix + matrix_base + 2 + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    write_value = tl.load(
        prefix_write + indices,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    projected = _float_add(
        _float_multiply(initial_0, matrix_0),
        _float_multiply(initial_1, matrix_1),
    )
    tl.store(
        output + indices,
        _float_add(projected, write_value),
        mask=mask,
    )


def pair_affine_scan_explicit_round(
    matrix: Tensor,
    write: Tensor,
    initial: Tensor,
) -> Tensor:
    global _profile_scan_calls
    global _profile_stage_launches
    matrix = matrix.contiguous()
    write = write.contiguous()
    initial = initial.contiguous()
    if (
        not matrix.is_cuda
        or not write.is_cuda
        or not initial.is_cuda
        or matrix.dtype != torch.float32
        or write.dtype != torch.float32
        or initial.dtype != torch.float32
    ):
        raise ValueError("Phase414 scan requires contiguous CUDA FP32 tensors")
    if matrix.ndim != 5 or matrix.shape[-2:] != (2, 2):
        raise ValueError("matrix must have shape [group, sequence, pair, 2, 2]")
    if write.shape != matrix.shape[:-1]:
        raise ValueError("write must have shape [group, sequence, pair, 2]")
    if initial.shape != (matrix.shape[0], matrix.shape[2], 2):
        raise ValueError("initial must have shape [group, pair, 2]")

    _, sequence, pairs = matrix.shape[:3]
    if _profile_counters_enabled:
        _profile_scan_calls += 1
    matrix_buffers = (torch.empty_like(matrix), torch.empty_like(matrix))
    write_buffers = (torch.empty_like(write), torch.empty_like(write))
    source_matrix = matrix
    source_write = write
    total_units = write.numel() // 2
    block_size = 256
    grid = (triton.cdiv(total_units, block_size),)
    offset = 1
    stage = 0
    while offset < sequence:
        output_matrix = matrix_buffers[stage & 1]
        output_write = write_buffers[stage & 1]
        _pair_scan_stage_kernel[grid](
            source_matrix,
            source_write,
            output_matrix,
            output_write,
            total_units,
            offset,
            SEQUENCE=sequence,
            PAIRS=pairs,
            BLOCK_SIZE=block_size,
        )
        source_matrix = output_matrix
        source_write = output_write
        offset *= 2
        stage += 1

    output = torch.empty_like(write)
    final_grid = (triton.cdiv(output.numel(), block_size),)
    _pair_scan_final_kernel[final_grid](
        source_matrix,
        source_write,
        initial,
        output,
        output.numel(),
        SEQUENCE=sequence,
        PAIRS=pairs,
        BLOCK_SIZE=block_size,
    )
    if _profile_counters_enabled:
        _profile_stage_launches += stage + 1
    return output
