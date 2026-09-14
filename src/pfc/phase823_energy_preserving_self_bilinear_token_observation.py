from __future__ import annotations

import math

import torch
from torch import Tensor


SELF_BILINEAR_CHANNEL_OFFSET = 257
SELF_BILINEAR_WIDTH = 512


def _observation_terms(rows: Tensor, rms_eps: float) -> tuple[Tensor, ...]:
    if rows.ndim < 1 or rows.shape[-1] != SELF_BILINEAR_WIDTH:
        raise ValueError("self-bilinear token rows require width 512")
    if not torch.is_floating_point(rows):
        raise TypeError("self-bilinear token rows must be floating point")
    if rms_eps <= 0:
        raise ValueError("rms_eps must be positive")

    raw_rms = rows.square().mean(dim=-1, keepdim=True).sqrt()
    safe_raw_rms = raw_rms.clamp_min(rms_eps)
    semantic = rows / safe_raw_rms
    product = semantic * torch.roll(
        semantic,
        shifts=SELF_BILINEAR_CHANNEL_OFFSET,
        dims=-1,
    )
    centered = product - product.mean(dim=-1, keepdim=True)
    projection = (centered * semantic).mean(dim=-1, keepdim=True)
    orthogonal = centered - projection * semantic
    orthogonal_rms = orthogonal.square().mean(dim=-1, keepdim=True).sqrt()
    safe_orthogonal_rms = orthogonal_rms.clamp_min(rms_eps)
    bilinear = orthogonal / safe_orthogonal_rms
    active = (raw_rms > rms_eps) & (orthogonal_rms > rms_eps)
    return (
        raw_rms,
        safe_raw_rms,
        semantic,
        centered,
        projection,
        orthogonal_rms,
        safe_orthogonal_rms,
        bilinear,
        active,
    )


def energy_preserving_self_bilinear_token_observation(
    rows: Tensor,
    *,
    rms_eps: float,
) -> Tensor:
    """Compose one learned token row with its orthogonal self-product view."""

    (
        raw_rms,
        _,
        semantic,
        _,
        _,
        _,
        _,
        bilinear,
        active,
    ) = _observation_terms(rows, rms_eps)
    observation = raw_rms * (semantic + bilinear) / math.sqrt(2.0)
    return torch.where(active, observation, rows)


def energy_preserving_self_bilinear_token_observation_vjp(
    rows: Tensor,
    output_gradient: Tensor,
    *,
    rms_eps: float,
) -> Tensor:
    """Apply the exact local VJP of the self-bilinear token observation."""

    if rows.shape != output_gradient.shape:
        raise ValueError("rows and output gradient must have matching shapes")
    if rows.dtype != output_gradient.dtype or rows.device != output_gradient.device:
        raise TypeError("rows and output gradient must share dtype and device")
    (
        raw_rms,
        safe_raw_rms,
        semantic,
        centered,
        projection,
        _,
        safe_orthogonal_rms,
        bilinear,
        active,
    ) = _observation_terms(rows, rms_eps)
    width = rows.shape[-1]
    inverse_sqrt_two = 1.0 / math.sqrt(2.0)

    composition = (semantic + bilinear) * inverse_sqrt_two
    raw_rms_gradient = (output_gradient * composition).sum(
        dim=-1,
        keepdim=True,
    )
    semantic_gradient = raw_rms * output_gradient * inverse_sqrt_two
    bilinear_gradient = raw_rms * output_gradient * inverse_sqrt_two

    orthogonal_gradient = (
        bilinear_gradient
        - bilinear
        * (bilinear_gradient * bilinear).mean(dim=-1, keepdim=True)
    ) / safe_orthogonal_rms
    projection_gradient = -(orthogonal_gradient * semantic).sum(
        dim=-1,
        keepdim=True,
    )
    centered_gradient = (
        orthogonal_gradient + projection_gradient * semantic / width
    )
    semantic_gradient = (
        semantic_gradient
        - projection * orthogonal_gradient
        + projection_gradient * centered / width
    )

    product_gradient = centered_gradient - centered_gradient.mean(
        dim=-1,
        keepdim=True,
    )
    semantic_gradient = (
        semantic_gradient
        + product_gradient
        * torch.roll(
            semantic,
            shifts=SELF_BILINEAR_CHANNEL_OFFSET,
            dims=-1,
        )
        + torch.roll(
            product_gradient * semantic,
            shifts=-SELF_BILINEAR_CHANNEL_OFFSET,
            dims=-1,
        )
    )

    raw_rms_gradient = raw_rms_gradient - (
        semantic_gradient * semantic
    ).sum(dim=-1, keepdim=True) / safe_raw_rms
    input_gradient = (
        semantic_gradient / safe_raw_rms
        + raw_rms_gradient * semantic / width
    )
    return torch.where(active, input_gradient, output_gradient)


__all__ = [
    "SELF_BILINEAR_CHANNEL_OFFSET",
    "SELF_BILINEAR_WIDTH",
    "energy_preserving_self_bilinear_token_observation",
    "energy_preserving_self_bilinear_token_observation_vjp",
]
