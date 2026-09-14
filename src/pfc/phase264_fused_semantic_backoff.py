from __future__ import annotations

import torch
from torch import Tensor


def fused_semantic_context_codes(
    context: Tensor,
    source: Tensor,
    hash_mask: Tensor,
    *,
    context_decay: float,
    zero_pad_to_power_of_two: bool = False,
) -> Tensor:
    if not context.is_cuda:
        raise ValueError("Phase264 fused semantic context requires CUDA")
    from pfc.phase264_triton_semantic_backoff import triton_semantic_context_codes

    with torch.autograd.profiler.record_function(
        "phase264::fused_semantic_context_codes"
    ):
        return triton_semantic_context_codes(
            context,
            source,
            hash_mask,
            context_decay=context_decay,
            zero_pad_to_power_of_two=zero_pad_to_power_of_two,
        )


def fused_semantic_memory_step(
    memory: Tensor,
    normalizer: Tensor,
    last_update: Tensor,
    encoded_source: Tensor,
    pending_codes: Tensor,
    query_codes: Tensor,
    *,
    next_token_count: int | Tensor,
    write_enabled: bool,
    fast_decay: float,
    rms_eps: float,
) -> tuple[Tensor, Tensor]:
    if not memory.is_cuda:
        raise ValueError("Phase264 fused semantic memory requires CUDA")
    from pfc.phase264_triton_semantic_backoff import triton_semantic_memory_step

    with torch.autograd.profiler.record_function(
        "phase264::fused_semantic_memory_step"
    ):
        return triton_semantic_memory_step(
            memory,
            normalizer,
            last_update,
            encoded_source,
            pending_codes,
            query_codes,
            next_token_count=next_token_count,
            write_enabled=write_enabled,
            fast_decay=fast_decay,
            rms_eps=rms_eps,
        )


def fused_compose_semantic_read_in_place(
    parent_read: Tensor,
    fast_encoded_read: Tensor,
    slow_encoded_read: Tensor,
    fast_active: Tensor,
    slow_active: Tensor,
    value_decoder: Tensor,
) -> Tensor:
    if not parent_read.is_cuda:
        raise ValueError("Phase264 fused semantic composition requires CUDA")
    from pfc.phase264_triton_semantic_backoff import (
        triton_compose_semantic_read_in_place,
    )

    with torch.autograd.profiler.record_function(
        "phase264::fused_semantic_composition"
    ):
        triton_compose_semantic_read_in_place(
            parent_read,
            fast_encoded_read,
            slow_encoded_read,
            fast_active,
            slow_active,
            value_decoder,
        )
    return parent_read
