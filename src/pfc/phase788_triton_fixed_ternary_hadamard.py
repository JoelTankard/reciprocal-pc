from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl


_WIDTH = 512
_STAGES = 9
_INV_SQRT_WIDTH = 0.044194173824159216
_launches = 0


@triton.jit
def _phase788_fixed_ternary_hadamard_kernel(
    values,
    signs,
    output,
    value_stride_row: tl.constexpr,
    value_stride_channel: tl.constexpr,
    output_stride_row: tl.constexpr,
    output_stride_channel: tl.constexpr,
    WIDTH: tl.constexpr,
    STAGES: tl.constexpr,
    INV_SQRT_WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    channels = tl.arange(0, WIDTH)
    current = tl.load(
        values + row * value_stride_row + channels * value_stride_channel
    ).to(tl.float32)
    current *= tl.load(signs + channels).to(tl.float32)

    for stage in tl.static_range(0, STAGES):
        stride = 1 << stage
        partner = tl.gather(current, channels ^ stride, axis=0)
        current = tl.where(
            (channels & stride) == 0,
            current + partner,
            partner - current,
        )

    current *= INV_SQRT_WIDTH
    prediction = 2.0 * tl.sigmoid(2.0 * current) - 1.0
    tl.store(
        output + row * output_stride_row + channels * output_stride_channel,
        prediction,
    )


def fused_fixed_ternary_hadamard_lexical_prediction(
    values: Tensor,
    signs: Tensor,
) -> Tensor:
    """Apply the fixed width-512 signed Hadamard and tanh in one CUDA launch."""

    if not torch.is_inference_mode_enabled():
        raise ValueError("fused fixed lexical Hadamard requires inference mode")
    if values.ndim < 1 or values.shape[-1] != _WIDTH:
        raise ValueError("fused fixed lexical Hadamard requires width 512")
    if signs.shape != (_WIDTH,) or signs.dtype != torch.int8:
        raise ValueError("fused fixed lexical Hadamard signs differ")
    if not values.is_cuda or not signs.is_cuda:
        raise ValueError("fused fixed lexical Hadamard requires CUDA")
    if values.dtype != torch.float32:
        raise ValueError("fused fixed lexical Hadamard requires FP32")
    if values.device != signs.device:
        raise ValueError("fused fixed lexical Hadamard tensors differ in device")
    if not values.is_contiguous() or not signs.is_contiguous():
        raise ValueError("fused fixed lexical Hadamard tensors must be contiguous")

    flat_values = values.view(-1, _WIDTH)
    output = torch.empty_like(flat_values)
    _phase788_fixed_ternary_hadamard_kernel[(flat_values.shape[0],)](
        flat_values,
        signs,
        output,
        flat_values.stride(0),
        flat_values.stride(1),
        output.stride(0),
        output.stride(1),
        WIDTH=_WIDTH,
        STAGES=_STAGES,
        INV_SQRT_WIDTH=_INV_SQRT_WIDTH,
        num_warps=8,
    )
    global _launches
    _launches += 1
    return output.view_as(values)


def fused_fixed_ternary_hadamard_launches() -> int:
    return _launches


__all__ = [
    "fused_fixed_ternary_hadamard_launches",
    "fused_fixed_ternary_hadamard_lexical_prediction",
]
