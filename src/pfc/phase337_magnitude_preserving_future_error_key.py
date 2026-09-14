from __future__ import annotations

import math

import torch
from torch import Tensor


def global_rms(value: Tensor) -> Tensor:
    if value.numel() == 0:
        raise ValueError("global RMS requires a nonempty tensor")
    return value.detach().float().square().mean().sqrt()


@torch.no_grad()
def project_gradient_to_global_rms_tangent_(
    parameter: Tensor,
    gradient: Tensor,
) -> dict[str, Tensor]:
    if parameter.shape != gradient.shape:
        raise ValueError("parameter and gradient shapes differ")
    if parameter.device != gradient.device:
        raise ValueError("parameter and gradient devices differ")
    weights = parameter.detach().float()
    ordinary = gradient.detach().float()
    weight_square_sum = weights.square().sum()
    if not bool(torch.isfinite(weight_square_sum)) or float(weight_square_sum) <= 0:
        raise ValueError("fixed-RMS parameter norm must be positive and finite")
    ordinary_dot = (weights * ordinary).sum()
    ordinary_norm = ordinary.square().sum().sqrt()
    weight_norm = weight_square_sum.sqrt()
    projected = ordinary - weights * (ordinary_dot / weight_square_sum)
    projected_dot = (weights * projected).sum()
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
def renormalize_to_global_rms_(
    parameter: Tensor,
    target_rms: float,
) -> dict[str, Tensor]:
    if not math.isfinite(target_rms) or target_rms <= 0:
        raise ValueError("target global RMS must be positive and finite")
    before_rms = global_rms(parameter)
    if not bool(torch.isfinite(before_rms)) or float(before_rms) <= 0:
        raise ValueError("current global RMS must be positive and finite")
    scale = before_rms.new_tensor(target_rms) / before_rms
    parameter.mul_(scale.to(parameter.dtype))
    after_rms = global_rms(parameter)
    target = after_rms.new_tensor(target_rms)
    return {
        "pre_renormalization_rms": before_rms,
        "post_renormalization_rms": after_rms,
        "target_rms": target,
        "renormalization_scale": scale,
        "post_rms_abs_error": (after_rms - target).abs(),
    }
