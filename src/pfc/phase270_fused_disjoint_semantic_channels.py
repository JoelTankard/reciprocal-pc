from __future__ import annotations

import torch
from torch import Tensor


def fused_compose_disjoint_semantic_channels_in_place(
    parent_read: Tensor,
    fast_encoded_read: Tensor,
    slow_encoded_read: Tensor,
    fast_active: Tensor,
    slow_active: Tensor,
    channel_decoders: Tensor,
) -> Tensor:
    if not parent_read.is_cuda:
        raise ValueError("Phase270 fused semantic composition requires CUDA")
    from pfc.phase270_triton_disjoint_semantic_channels import (
        triton_compose_disjoint_semantic_channels_in_place,
    )

    with torch.autograd.profiler.record_function(
        "phase270::fused_disjoint_semantic_composition"
    ):
        triton_compose_disjoint_semantic_channels_in_place(
            parent_read,
            fast_encoded_read,
            slow_encoded_read,
            fast_active,
            slow_active,
            channel_decoders,
        )
    return parent_read
