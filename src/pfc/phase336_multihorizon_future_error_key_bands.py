from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


FUTURE_ERROR_KEY_HORIZONS = (1, 4, 16, 64)
FUTURE_ERROR_KEY_BANDS = tuple(
    tuple(range(offset, 32, len(FUTURE_ERROR_KEY_HORIZONS)))
    for offset in range(len(FUTURE_ERROR_KEY_HORIZONS))
)


def _validated_layout(
    horizons: tuple[int, ...],
    code_width: int,
) -> tuple[tuple[int, ...], ...]:
    if len(horizons) != len(FUTURE_ERROR_KEY_BANDS):
        raise ValueError("Phase336 requires exactly four future horizons")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("Phase336 future horizons must be positive")
    bands = tuple(
        tuple(range(offset, code_width, len(horizons)))
        for offset in range(len(horizons))
    )
    if any(len(band) != 8 for band in bands):
        raise ValueError("Phase336 requires four eight-channel key bands")
    covered = [channel for band in bands for channel in band]
    if sorted(covered) != list(range(code_width)):
        raise ValueError("Phase336 key bands must cover each channel exactly once")
    return bands


def multihorizon_future_error_key_band_gradient(
    prediction_features: Tensor,
    future_error_codes: Tensor,
    projection: Tensor,
    *,
    horizons: tuple[int, ...] = FUTURE_ERROR_KEY_HORIZONS,
) -> tuple[Tensor, dict[str, Tensor]]:
    if prediction_features.ndim != 3 or future_error_codes.ndim != 3:
        raise ValueError("Phase336 features and errors must be sequences")
    if prediction_features.shape[:2] != future_error_codes.shape[:2]:
        raise ValueError("Phase336 feature and error sequences differ")
    if projection.shape != (
        prediction_features.shape[-1],
        future_error_codes.shape[-1],
    ):
        raise ValueError("Phase336 projection shape differs")
    if prediction_features.device != future_error_codes.device:
        raise ValueError("Phase336 feature and error devices differ")
    if projection.device != prediction_features.device:
        raise ValueError("Phase336 projection device differs")

    code_width = future_error_codes.shape[-1]
    bands = _validated_layout(horizons, code_width)
    sequence = prediction_features.shape[1]
    if max(horizons) >= sequence:
        raise ValueError("Phase336 horizon must be shorter than the sequence")
    if len(set(horizons)) == 1:
        from pfc.phase298_intrinsic_future_error_key import (
            intrinsic_future_error_key_gradient,
        )

        horizon = horizons[0]
        return intrinsic_future_error_key_gradient(
            prediction_features[:, :-horizon],
            future_error_codes[:, horizon:],
            projection,
        )

    features = prediction_features.detach().float()
    targets = future_error_codes.detach().float()
    weights = projection.detach().float()
    gradient = torch.zeros_like(weights)
    objective = torch.zeros((), device=features.device)
    prediction_square_sum = torch.zeros((), device=features.device)
    target_square_sum = torch.zeros((), device=features.device)
    residual_square_sum = torch.zeros((), device=features.device)
    scalar_count = 0
    cosine_sum = torch.zeros((), device=features.device)
    sample_count_total = 0
    diagnostics: dict[str, Tensor] = {}

    band_stride = len(horizons)
    for horizon, band in zip(horizons, bands, strict=True):
        band_offset = band[0]
        band_features = features[:, :-horizon]
        band_targets = targets[:, horizon:, band_offset::band_stride]
        band_weights = weights[:, band_offset::band_stride]
        prediction = band_features @ band_weights
        residual = prediction - band_targets
        sample_count = band_features.shape[0] * band_features.shape[1]
        flat_features = band_features.reshape(-1, band_features.shape[-1])
        flat_residual = residual.reshape(-1, residual.shape[-1])
        band_gradient = flat_features.T @ flat_residual / sample_count
        gradient[:, band_offset::band_stride].copy_(band_gradient)

        band_objective = 0.5 * residual.square().sum(dim=-1).mean()
        band_cosine = F.cosine_similarity(
            prediction,
            band_targets,
            dim=-1,
            eps=1e-12,
        ).mean()
        objective.add_(band_objective)
        prediction_square_sum.add_(prediction.square().sum())
        target_square_sum.add_(band_targets.square().sum())
        residual_square_sum.add_(residual.square().sum())
        scalar_count += prediction.numel()
        cosine_sum.add_(band_cosine * sample_count)
        sample_count_total += sample_count
        prefix = f"multihorizon_future_error_h{horizon}"
        diagnostics.update(
            {
                f"{prefix}_eligible_samples": torch.full(
                    (),
                    sample_count,
                    device=features.device,
                    dtype=torch.int64,
                ),
                f"{prefix}_objective": band_objective,
                f"{prefix}_prediction_rms": prediction.square().mean().sqrt(),
                f"{prefix}_target_rms": band_targets.square().mean().sqrt(),
                f"{prefix}_residual_rms": residual.square().mean().sqrt(),
                f"{prefix}_cosine": band_cosine,
                f"{prefix}_gradient_rms": band_gradient.square().mean().sqrt(),
            }
        )

    diagnostics.update(
        {
            "intrinsic_future_error_key_objective": objective,
            "intrinsic_future_error_key_prediction_rms": (
                prediction_square_sum / scalar_count
            ).sqrt(),
            "intrinsic_future_error_key_target_rms": (
                target_square_sum / scalar_count
            ).sqrt(),
            "intrinsic_future_error_key_residual_rms": (
                residual_square_sum / scalar_count
            ).sqrt(),
            "intrinsic_future_error_key_cosine": cosine_sum / sample_count_total,
            "intrinsic_future_error_key_gradient_rms": (
                gradient.square().mean().sqrt()
            ),
        }
    )
    return gradient.to(projection.dtype), diagnostics
