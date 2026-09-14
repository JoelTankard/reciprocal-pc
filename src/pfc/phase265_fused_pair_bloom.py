from __future__ import annotations

import torch
from torch import Tensor


def fused_pair_recurrence_bloom_step(
    cache: Tensor,
    previous_token_ids: Tensor,
    token_ids: Tensor,
    *,
    vocab_size: int,
    active_generation: int,
) -> tuple[Tensor, Tensor]:
    if not cache.is_cuda:
        raise ValueError("Phase265 fused pair Bloom requires CUDA")
    from pfc.phase265_triton_pair_bloom import triton_pair_recurrence_bloom_step

    with torch.autograd.profiler.record_function(
        "phase265::fused_pair_recurrence_bloom_step"
    ):
        return triton_pair_recurrence_bloom_step(
            cache,
            previous_token_ids,
            token_ids,
            vocab_size=vocab_size,
            active_generation=active_generation,
        )
