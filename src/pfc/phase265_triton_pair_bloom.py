from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl


GENERATIONS = tl.constexpr(2)
PROBES = tl.constexpr(4)
BYTES_PER_PROBE = tl.constexpr(1024)
BITS_PER_PROBE = tl.constexpr(8192)


@triton.jit
def _pair_recurrence_bloom_step_kernel(
    cache,
    previous_token_ids,
    token_ids,
    seen,
    PREVIOUS_TOKEN_STRIDE: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    ACTIVE_GENERATION: tl.constexpr,
):
    batch = tl.program_id(0)
    probes = tl.arange(0, PROBES)
    previous_token = tl.load(
        previous_token_ids + batch * PREVIOUS_TOKEN_STRIDE
    ).to(tl.int64)
    token = tl.load(token_ids + batch * TOKEN_STRIDE).to(tl.int64)
    pair_id = previous_token * VOCAB_SIZE + token + 1
    multiplier_zero = tl.full((PROBES,), 2_246_822_519, tl.int64)
    multiplier_one = tl.full((PROBES,), 3_266_489_917, tl.int64)
    multiplier_two = tl.full((PROBES,), 668_265_263, tl.int64)
    multiplier_three = tl.full((PROBES,), 1_274_126_177, tl.int64)
    multipliers = tl.where(
        probes == 0,
        multiplier_zero,
        tl.where(
            probes == 1,
            multiplier_one,
            tl.where(probes == 2, multiplier_two, multiplier_three),
        ),
    )
    offsets = tl.where(
        probes == 0,
        1_654_911_287,
        tl.where(
            probes == 1,
            2_045_814_391,
            tl.where(probes == 2, 739_632_901, 281_243_693),
        ),
    ).to(tl.int64)
    mixed = pair_id * multipliers + offsets
    mixed = mixed ^ (mixed >> 16)
    addresses = mixed & (BITS_PER_PROBE - 1)
    byte_addresses = addresses >> 3
    bit_masks = (1 << (addresses & 7)).to(tl.uint8)
    batch_offset = batch * GENERATIONS * PROBES * BYTES_PER_PROBE

    generation_zero_offsets = (
        batch_offset + probes * BYTES_PER_PROBE + byte_addresses
    )
    generation_one_offsets = (
        batch_offset
        + PROBES * BYTES_PER_PROBE
        + probes * BYTES_PER_PROBE
        + byte_addresses
    )
    generation_zero_values = tl.load(cache + generation_zero_offsets).to(tl.uint8)
    generation_one_values = tl.load(cache + generation_one_offsets).to(tl.uint8)
    generation_zero_hit = tl.sum(
        ((generation_zero_values & bit_masks) != 0).to(tl.int32), axis=0
    ) == PROBES
    generation_one_hit = tl.sum(
        ((generation_one_values & bit_masks) != 0).to(tl.int32), axis=0
    ) == PROBES
    tl.store(seen + batch, generation_zero_hit | generation_one_hit)

    active_offsets = (
        batch_offset
        + ACTIVE_GENERATION * PROBES * BYTES_PER_PROBE
        + probes * BYTES_PER_PROBE
        + byte_addresses
    )
    active_values = tl.load(cache + active_offsets).to(tl.uint8)
    tl.store(cache + active_offsets, active_values | bit_masks)


def triton_pair_recurrence_bloom_step(
    cache: Tensor,
    previous_token_ids: Tensor,
    token_ids: Tensor,
    *,
    vocab_size: int,
    active_generation: int,
) -> tuple[Tensor, Tensor]:
    if cache.dtype != torch.uint8 or cache.shape[-3:] != (
        GENERATIONS,
        PROBES,
        BYTES_PER_PROBE,
    ):
        raise ValueError("Phase265 pair Bloom cache differs")
    if not cache.is_contiguous():
        raise ValueError("Phase265 pair Bloom cache must be contiguous")
    batch = cache.shape[0]
    if previous_token_ids.shape != (batch,) or token_ids.shape != (batch,):
        raise ValueError("Phase265 pair Bloom token shape differs")
    if previous_token_ids.dtype != torch.long or token_ids.dtype != torch.long:
        raise ValueError("Phase265 pair Bloom tokens must be int64")
    if active_generation not in (0, 1):
        raise ValueError("Phase265 active Bloom generation differs")
    updated = cache.clone()
    seen = torch.empty(batch, device=cache.device, dtype=torch.bool)
    _pair_recurrence_bloom_step_kernel[(batch,)](
        updated,
        previous_token_ids,
        token_ids,
        seen,
        PREVIOUS_TOKEN_STRIDE=previous_token_ids.stride(0),
        TOKEN_STRIDE=token_ids.stride(0),
        VOCAB_SIZE=vocab_size,
        ACTIVE_GENERATION=active_generation,
        num_warps=1,
    )
    return seen, updated
