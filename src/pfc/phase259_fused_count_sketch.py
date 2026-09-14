from __future__ import annotations

import torch
from torch import Tensor


def fused_indexed_count_sketch_step(
    memory: Tensor,
    normalizer: Tensor,
    last_update: Tensor,
    encoded_source: Tensor,
    prior_token_ids: Tensor,
    query_token_ids: Tensor,
    indexed_addresses: Tensor,
    indexed_signs: Tensor,
    *,
    next_token_count: int | Tensor,
    write_enabled: bool,
    fast_decay: float,
    rms_eps: float,
) -> Tensor:
    if not memory.is_cuda:
        raise ValueError("Phase259 fused CountSketch requires CUDA")
    from pfc.phase259_triton_count_sketch import triton_count_sketch_step

    with torch.autograd.profiler.record_function(
        "phase259::fused_indexed_count_sketch"
    ):
        return triton_count_sketch_step(
            memory,
            normalizer,
            last_update,
            encoded_source,
            prior_token_ids,
            query_token_ids,
            indexed_addresses,
            indexed_signs,
            next_token_count=next_token_count,
            write_enabled=write_enabled,
            fast_decay=fast_decay,
            rms_eps=rms_eps,
        )


def fused_dual_order_count_sketch_step(
    memory: Tensor,
    normalizer: Tensor,
    last_update: Tensor,
    encoded_source: Tensor,
    previous_prior_token_ids: Tensor,
    prior_token_ids: Tensor,
    query_token_ids: Tensor,
    indexed_addresses: Tensor,
    indexed_signs: Tensor,
    hash_multipliers: Tensor,
    hash_offsets: Tensor,
    sign_multipliers: Tensor,
    sign_offsets: Tensor,
    *,
    vocab_size: int,
    next_token_count: int | Tensor,
    write_enabled: bool,
    fast_decay: float,
    rms_eps: float,
    zero_anchored_pair_consensus: bool = False,
) -> Tensor:
    if not memory.is_cuda:
        raise ValueError("Phase262 fused dual-order CountSketch requires CUDA")
    from pfc.phase259_triton_count_sketch import triton_count_sketch_step

    with torch.autograd.profiler.record_function(
        "phase262::fused_dual_order_count_sketch"
    ):
        return triton_count_sketch_step(
            memory,
            normalizer,
            last_update,
            encoded_source,
            prior_token_ids,
            query_token_ids,
            indexed_addresses,
            indexed_signs,
            previous_prior_token_ids=previous_prior_token_ids,
            hash_multipliers=hash_multipliers,
            hash_offsets=hash_offsets,
            sign_multipliers=sign_multipliers,
            sign_offsets=sign_offsets,
            vocab_size=vocab_size,
            next_token_count=next_token_count,
            write_enabled=write_enabled,
            fast_decay=fast_decay,
            rms_eps=rms_eps,
            zero_anchored_pair_consensus=zero_anchored_pair_consensus,
        )
