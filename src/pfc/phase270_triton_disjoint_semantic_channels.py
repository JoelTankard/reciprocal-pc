from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl


VALUE_WIDTH = tl.constexpr(8)


@triton.jit
def _half_multiply(left, right):
    return tl.inline_asm_elementwise(
        "mul.rn.f16x2 $0, $1, $2;",
        "=r,r,r",
        [left, right],
        dtype=tl.float16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _half_add(left, right):
    return tl.inline_asm_elementwise(
        "add.rn.f16x2 $0, $1, $2;",
        "=r,r,r",
        [left, right],
        dtype=tl.float16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _compose_disjoint_semantic_channels_kernel(
    parent_read,
    fast_encoded_read,
    slow_encoded_read,
    fast_active,
    slow_active,
    channel_decoders,
    parent_stride_batch: tl.constexpr,
    parent_stride_width: tl.constexpr,
    fast_stride_batch: tl.constexpr,
    fast_stride_value: tl.constexpr,
    slow_stride_batch: tl.constexpr,
    slow_stride_value: tl.constexpr,
    fast_active_stride_batch: tl.constexpr,
    slow_active_stride_batch: tl.constexpr,
    decoder_stride_group: tl.constexpr,
    decoder_stride_value: tl.constexpr,
    decoder_stride_width: tl.constexpr,
    WIDTH: tl.constexpr,
    HALF_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    channels = tl.arange(0, BLOCK_SIZE)
    width_mask = channels < WIDTH
    half_channels = channels & (HALF_WIDTH - 1)
    parent_offsets = (
        batch * parent_stride_batch + channels * parent_stride_width
    )
    parent = tl.load(
        parent_read + parent_offsets,
        mask=width_mask,
        other=0.0,
    ).to(tl.float16)
    values = tl.arange(0, VALUE_WIDTH)
    fast_encoded = tl.load(
        fast_encoded_read
        + batch * fast_stride_batch
        + values * fast_stride_value,
    ).to(tl.float16)
    slow_encoded = tl.load(
        slow_encoded_read
        + batch * slow_stride_batch
        + values * slow_stride_value,
    ).to(tl.float16)
    decoder_offsets = (
        values[:, None] * decoder_stride_value
        + half_channels[None, :] * decoder_stride_width
    )
    fast_decoder = tl.load(
        channel_decoders + decoder_offsets,
        mask=width_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    slow_decoder = tl.load(
        channel_decoders + decoder_stride_group + decoder_offsets,
        mask=width_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    fast = tl.sum(
        fast_encoded[:, None].to(tl.float32) * fast_decoder.to(tl.float32),
        axis=0,
    ).to(tl.float16)
    slow = tl.sum(
        slow_encoded[:, None].to(tl.float32) * slow_decoder.to(tl.float32),
        axis=0,
    ).to(tl.float16)
    fast_on = tl.load(
        fast_active + batch * fast_active_stride_batch,
    ).to(tl.float16)
    slow_on = tl.load(
        slow_active + batch * slow_active_stride_batch,
    ).to(tl.float16)
    lower_half = channels < HALF_WIDTH
    semantic = tl.where(lower_half, fast, slow)
    active = tl.where(lower_half, fast_on, slow_on)
    numerator = _half_multiply(
        parent,
        tl.full((BLOCK_SIZE,), 4.0, tl.float16),
    )
    numerator = _half_add(
        numerator,
        _half_multiply(semantic, active),
    )
    denominator = (4.0 + active).to(tl.float16)
    result = (numerator / denominator).to(tl.float16)
    tl.store(parent_read + parent_offsets, result, mask=width_mask)


def triton_compose_disjoint_semantic_channels_in_place(
    parent_read: Tensor,
    fast_encoded_read: Tensor,
    slow_encoded_read: Tensor,
    fast_active: Tensor,
    slow_active: Tensor,
    channel_decoders: Tensor,
) -> None:
    if parent_read.dtype != torch.float16 or parent_read.ndim != 2:
        raise ValueError("Phase270 parent read must be rank-two FP16")
    batch, width = parent_read.shape
    if width <= 0 or width % 2:
        raise ValueError("Phase270 parent read width must be positive and even")
    half_width = width // 2
    if half_width & (half_width - 1):
        raise ValueError("Phase270 half width must be a power of two")
    if fast_encoded_read.shape != (batch, VALUE_WIDTH) or slow_encoded_read.shape != (
        batch,
        VALUE_WIDTH,
    ):
        raise ValueError("Phase270 encoded semantic read shape differs")
    if (
        fast_encoded_read.dtype != torch.float16
        or slow_encoded_read.dtype != torch.float16
    ):
        raise ValueError("Phase270 encoded semantic reads must be FP16")
    if fast_active.shape != (batch, 1) or slow_active.shape != (batch, 1):
        raise ValueError("Phase270 semantic active shape differs")
    if channel_decoders.shape != (2, VALUE_WIDTH, half_width):
        raise ValueError("Phase270 semantic channel decoder shape differs")
    if channel_decoders.dtype != torch.float16:
        raise ValueError("Phase270 semantic channel decoders must be FP16")
    block_size = triton.next_power_of_2(width)
    _compose_disjoint_semantic_channels_kernel[(batch,)](
        parent_read,
        fast_encoded_read,
        slow_encoded_read,
        fast_active,
        slow_active,
        channel_decoders,
        parent_stride_batch=parent_read.stride(0),
        parent_stride_width=parent_read.stride(1),
        fast_stride_batch=fast_encoded_read.stride(0),
        fast_stride_value=fast_encoded_read.stride(1),
        slow_stride_batch=slow_encoded_read.stride(0),
        slow_stride_value=slow_encoded_read.stride(1),
        fast_active_stride_batch=fast_active.stride(0),
        slow_active_stride_batch=slow_active.stride(0),
        decoder_stride_group=channel_decoders.stride(0),
        decoder_stride_value=channel_decoders.stride(1),
        decoder_stride_width=channel_decoders.stride(2),
        WIDTH=width,
        HALF_WIDTH=half_width,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
