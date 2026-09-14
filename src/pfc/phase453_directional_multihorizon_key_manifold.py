from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def project_multihorizon_key_gradient_to_tangent_(
    parameter: Tensor,
    gradient: Tensor,
) -> dict[str, Tensor]:
    """Project one complete multihorizon key update without host synchronization."""

    if parameter.shape != gradient.shape:
        raise ValueError("parameter and gradient shapes differ")
    if parameter.device != gradient.device:
        raise ValueError("parameter and gradient devices differ")
    weights = parameter.detach().float()
    ordinary = gradient.detach().float()
    weight_square_sum = weights.square().sum()
    ordinary_dot = (weights * ordinary).sum()
    ordinary_norm = ordinary.square().sum().sqrt()
    weight_norm = weight_square_sum.sqrt()
    projected = ordinary - weights * (ordinary_dot / weight_square_sum)
    projected = projected - weights * (
        (weights * projected).sum() / weight_square_sum
    )
    projected_dot = (weights.double() * projected.double()).sum().float()
    projected_norm = projected.square().sum().sqrt()
    gradient.copy_(projected.to(gradient.dtype))
    tiny = torch.finfo(torch.float32).tiny
    return {
        "ordinary_gradient_rms": ordinary.square().mean().sqrt(),
        "tangent_gradient_rms": projected.square().mean().sqrt(),
        "ordinary_radial_cosine_abs": ordinary_dot.abs()
        / (weight_norm * ordinary_norm).clamp_min(tiny),
        "tangent_radial_cosine_abs": projected_dot.abs()
        / (weight_norm * projected_norm).clamp_min(tiny),
        "tangent_dot_abs": projected_dot.abs(),
    }


@torch.no_grad()
def restore_multihorizon_key_global_rms_(
    parameter: Tensor,
    target_rms: float,
) -> dict[str, Tensor]:
    """Restore global RMS using only device operations in the static update."""

    before_rms = parameter.detach().float().square().mean().sqrt()
    target = torch.full_like(before_rms, target_rms)
    scale = target / before_rms
    parameter.mul_(scale.to(parameter.dtype))
    after_rms = parameter.detach().float().square().mean().sqrt()
    return {
        "pre_renormalization_rms": before_rms,
        "post_renormalization_rms": after_rms,
        "target_rms": target,
        "renormalization_scale": scale,
        "post_rms_abs_error": (after_rms - target).abs(),
    }
