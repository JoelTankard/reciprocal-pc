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
def _constant_decay_scan_stage_kernel(
    source,
    output,
    total_values,
    decay_power,
    offset,
    SEQUENCE: tl.constexpr,
    INNER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    indices = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < total_values
    sequence_positions = (indices // INNER) % SEQUENCE
    current = tl.load(source + indices, mask=mask, other=0.0).to(tl.float32)
    has_previous = sequence_positions >= offset
    previous = tl.load(
        source + indices - offset * INNER,
        mask=mask & has_previous,
        other=0.0,
    ).to(tl.float32)
    scaled = _float_multiply(previous, decay_power)
    composed = _float_add(current, scaled)
    result = tl.where(has_previous, composed, current)
    tl.store(output + indices, result, mask=mask)


def triton_constant_decay_prefix(write: Tensor, decay: float) -> Tensor:
    global _profile_scan_calls
    global _profile_stage_launches
    if (
        not write.is_cuda
        or write.dtype != torch.float32
        or not write.is_contiguous()
        or write.ndim < 3
    ):
        raise ValueError("Phase345 scan kernel requires contiguous CUDA FP32")
    sequence = write.shape[1]
    if sequence <= 1:
        return write
    if _profile_counters_enabled:
        _profile_scan_calls += 1
    inner = write.numel() // (write.shape[0] * sequence)
    buffers = (torch.empty_like(write), torch.empty_like(write))
    source = write
    offset = 1
    decay_power = decay
    stage = 0
    block_size = 256
    grid = (triton.cdiv(write.numel(), block_size),)
    while offset < sequence:
        output = buffers[stage & 1]
        _constant_decay_scan_stage_kernel[grid](
            source,
            output,
            write.numel(),
            decay_power,
            offset,
            SEQUENCE=sequence,
            INNER=inner,
            BLOCK_SIZE=block_size,
        )
        source = output
        offset *= 2
        decay_power *= decay_power
        stage += 1
    if _profile_counters_enabled:
        _profile_stage_launches += stage
    return source
