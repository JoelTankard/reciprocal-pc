from __future__ import annotations

import torch
from torch import Tensor


def held_query_displacement_corrections(
    current_gradients: dict[str, Tensor],
    previous_displacements: dict[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    if current_gradients.keys() != previous_displacements.keys():
        raise ValueError("Phase297 gradient and displacement names differ")
    if not current_gradients:
        raise ValueError("Phase297 requires at least one deployed gradient")

    device = next(iter(current_gradients.values())).device
    corrections: dict[str, Tensor] = {}
    active_count = torch.zeros((), device=device, dtype=torch.float32)
    positive_harm = torch.zeros((), device=device, dtype=torch.float32)
    post_dot_max_abs = torch.zeros((), device=device, dtype=torch.float32)
    correction_square_sum = torch.zeros((), device=device, dtype=torch.float32)
    displacement_square_sum = torch.zeros((), device=device, dtype=torch.float32)
    gradient_square_sum = torch.zeros((), device=device, dtype=torch.float32)
    correction_norm_ratio_max = torch.zeros((), device=device, dtype=torch.float32)
    value_count = 0

    for name, gradient in current_gradients.items():
        displacement = previous_displacements[name]
        if gradient.shape != displacement.shape:
            raise ValueError(f"Phase297 shape differs for {name}")
        if gradient.device != displacement.device:
            raise ValueError(f"Phase297 device differs for {name}")
        float_gradient = gradient.detach().float()
        float_displacement = displacement.detach().float()
        directional_derivative = (float_gradient * float_displacement).sum()
        gradient_norm_square = float_gradient.square().sum()
        active = (directional_derivative > 0) & (gradient_norm_square > 0)
        scale = torch.where(
            active,
            -directional_derivative / gradient_norm_square.clamp_min(
                torch.finfo(torch.float32).tiny
            ),
            torch.zeros_like(directional_derivative),
        )
        correction = scale * float_gradient
        corrections[name] = correction

        post_dot = (
            float_gradient * (float_displacement + correction)
        ).sum()
        correction_norm = correction.square().sum().sqrt()
        displacement_norm = float_displacement.square().sum().sqrt()
        active_count.add_(active.float())
        positive_harm.add_(torch.where(active, directional_derivative, 0.0))
        post_dot_max_abs.copy_(
            torch.maximum(
                post_dot_max_abs,
                torch.where(active, post_dot.abs(), 0.0),
            )
        )
        correction_square_sum.add_(correction.square().sum())
        displacement_square_sum.add_(float_displacement.square().sum())
        gradient_square_sum.add_(float_gradient.square().sum())
        correction_norm_ratio_max.copy_(
            torch.maximum(
                correction_norm_ratio_max,
                correction_norm / displacement_norm.clamp_min(
                    torch.finfo(torch.float32).tiny
                ),
            )
        )
        value_count += gradient.numel()

    tensor_count = len(current_gradients)
    return corrections, {
        "eligible_tensor_count": active_count.new_full((), tensor_count),
        "active_tensor_count": active_count,
        "active_tensor_fraction": active_count / float(tensor_count),
        "positive_harm_sum": positive_harm,
        "post_projection_dot_max_abs": post_dot_max_abs,
        "rollback_rms": (correction_square_sum / float(value_count)).sqrt(),
        "previous_displacement_rms": (
            displacement_square_sum / float(value_count)
        ).sqrt(),
        "current_gradient_rms": (gradient_square_sum / float(value_count)).sqrt(),
        "correction_norm_ratio_max": correction_norm_ratio_max,
    }
