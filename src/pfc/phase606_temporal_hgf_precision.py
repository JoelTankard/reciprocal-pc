from __future__ import annotations

import math

import torch
from torch import Tensor


TONIC_LOG_VOLATILITY = -10.0
PROCESS_VARIANCE = math.exp(TONIC_LOG_VOLATILITY)


def _statistics_dtype(reciprocal: Tensor) -> torch.dtype:
    return torch.float64 if reciprocal.dtype == torch.float64 else torch.float32


def _validate_sequence_inputs(
    reciprocal: Tensor,
    initial_posterior_precision: Tensor,
    sequence_length: int,
    sensory_predicted_precision: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if reciprocal.ndim != 3:
        raise ValueError("reciprocal must have [layer,lower,parent] shape")
    layers, lower_width, parent_width = reciprocal.shape
    if min(layers, lower_width, parent_width, sequence_length) < 1:
        raise ValueError("temporal HGF dimensions must be positive")
    if lower_width != parent_width:
        raise ValueError("temporal HGF requires equal state widths")
    if not reciprocal.is_floating_point():
        raise ValueError("reciprocal must use floating-point values")
    if initial_posterior_precision.ndim != 3:
        raise ValueError("initial posterior must have [layer,batch,width] shape")
    if initial_posterior_precision.shape[0] != layers:
        raise ValueError("initial posterior layer count differs")
    if initial_posterior_precision.shape[2] != parent_width:
        raise ValueError("initial posterior width differs")
    batch = initial_posterior_precision.shape[1]
    if batch < 1:
        raise ValueError("temporal HGF batch must be positive")
    dtype = _statistics_dtype(reciprocal)
    initial = initial_posterior_precision.detach().to(
        device=reciprocal.device,
        dtype=dtype,
    )
    sensory = (
        torch.ones(
            batch,
            sequence_length,
            lower_width,
            device=reciprocal.device,
            dtype=dtype,
        )
        if sensory_predicted_precision is None
        else sensory_predicted_precision.detach().to(
            device=reciprocal.device,
            dtype=dtype,
        )
    )
    if sensory.shape != (batch, sequence_length, lower_width):
        raise ValueError("sensory predicted precision shape differs")
    return initial, sensory


def _reference_precision_scan(
    evidence: Tensor,
    initial_posterior: Tensor,
    reset_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    posterior_steps = []
    predicted_steps = []
    previous = initial_posterior
    for position, current_evidence in enumerate(evidence.unbind(dim=1)):
        if reset_mask is not None:
            previous = torch.where(
                reset_mask[:, position].unsqueeze(-1),
                torch.ones_like(previous),
                previous,
            )
        current_predicted = previous / (1.0 + PROCESS_VARIANCE * previous)
        current_posterior = current_predicted + current_evidence
        predicted_steps.append(current_predicted)
        posterior_steps.append(current_posterior)
        previous = current_posterior
    return torch.stack(predicted_steps, dim=1), torch.stack(posterior_steps, dim=1)


def temporal_hgf_precision_scan(
    evidence: Tensor,
    initial_posterior: Tensor,
    reset_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if reset_mask is not None and reset_mask.shape != evidence.shape[:2]:
        raise ValueError("temporal HGF reset mask shape differs")
    if (
        evidence.is_cuda
        and evidence.dtype == torch.float32
        and initial_posterior.dtype == torch.float32
    ):
        from pfc.phase606_triton_temporal_hgf_precision import (
            triton_temporal_hgf_precision_scan,
        )

        return triton_temporal_hgf_precision_scan(
            evidence,
            initial_posterior,
            reset_mask,
        )
    return _reference_precision_scan(evidence, initial_posterior, reset_mask)


def _squared_coupling_evidence(child: Tensor, squared_coupling: Tensor) -> Tensor:
    if child.is_cuda and child.dtype == torch.float32:
        from pfc.phase606_triton_temporal_hgf_precision import (
            triton_squared_coupling_evidence,
        )

        return triton_squared_coupling_evidence(child, squared_coupling)
    return torch.matmul(child, squared_coupling)


@torch.no_grad()
def temporal_hgf_precision_sequence(
    reciprocal: Tensor,
    initial_posterior_precision: Tensor,
    *,
    sequence_length: int,
    sensory_predicted_precision: Tensor | None = None,
    reset_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """Run causal HGF Eq. 7-8 precision carry without differentiating statistics."""

    initial, sensory = _validate_sequence_inputs(
        reciprocal,
        initial_posterior_precision,
        sequence_length,
        sensory_predicted_precision,
    )
    squared = reciprocal.detach().to(initial.dtype).square()
    child_layers = []
    predicted_layers = []
    evidence_layers = []
    posterior_layers = []
    child = sensory
    for layer in range(reciprocal.shape[0]):
        evidence = _squared_coupling_evidence(child, squared[layer])
        predicted, posterior = temporal_hgf_precision_scan(
            evidence,
            initial[layer],
            reset_mask,
        )
        child_layers.append(child)
        predicted_layers.append(predicted)
        evidence_layers.append(evidence)
        posterior_layers.append(posterior)
        child = predicted
    child_stack = torch.stack(child_layers)
    predicted_stack = torch.stack(predicted_layers)
    evidence_stack = torch.stack(evidence_layers)
    posterior_stack = torch.stack(posterior_layers)
    final_posterior = posterior_stack[:, :, -1]
    samples = initial.shape[1] * sequence_length
    child_flat = child_stack.reshape(reciprocal.shape[0], samples, reciprocal.shape[1])
    posterior_flat = posterior_stack.reshape(
        reciprocal.shape[0], samples, reciprocal.shape[2]
    )
    effective_energy = torch.bmm(child_flat.square(), squared) / (
        posterior_flat.square()
    )
    diagnostics = {
        "predicted_precision_by_layer": predicted_stack.mean(dim=(1, 2, 3)),
        "child_predicted_precision_by_layer": child_stack.mean(dim=(1, 2, 3)),
        "lower_precision_by_layer": child_stack.mean(dim=(1, 2, 3)),
        "evidence_precision_by_layer": evidence_stack.mean(dim=(1, 2, 3)),
        "posterior_precision_by_layer": posterior_stack.mean(dim=(1, 2, 3)),
        "posterior_precision_min": posterior_stack.min(),
        "posterior_precision_max": posterior_stack.max(),
        "effective_reciprocal_rms_by_layer": (
            effective_energy.mean(dim=(1, 2)) / reciprocal.shape[1]
        ).sqrt(),
    }
    return (
        child_stack,
        predicted_stack,
        evidence_stack,
        posterior_stack,
        final_posterior,
        diagnostics,
    )


def temporal_hgf_bottom_up(
    lower_errors: Tensor,
    reciprocal: Tensor,
    child_predicted_precision: Tensor,
    posterior_precision: Tensor,
) -> Tensor:
    if lower_errors.shape != child_predicted_precision.shape:
        raise ValueError("temporal HGF child precision shape differs from errors")
    if lower_errors.shape != posterior_precision.shape:
        raise ValueError("temporal HGF posterior precision shape differs from errors")
    if lower_errors.ndim < 3 or lower_errors.shape[0] != reciprocal.shape[0]:
        raise ValueError("temporal HGF errors must start with layer dimension")
    layers = lower_errors.shape[0]
    width = lower_errors.shape[-1]
    if reciprocal.shape != (layers, width, width):
        raise ValueError("temporal HGF reciprocal shape differs from errors")
    samples = lower_errors.numel() // (layers * width)
    errors = lower_errors.reshape(layers, samples, width)
    child = child_predicted_precision.to(lower_errors.dtype).reshape(
        layers, samples, width
    )
    posterior = posterior_precision.to(lower_errors.dtype).reshape(
        layers, samples, width
    )
    result = torch.bmm(errors * child, reciprocal) / posterior
    return result.reshape_as(lower_errors)


def temporal_hgf_reverse_message(
    target_direction: Tensor,
    reciprocal_reverse: Tensor,
    child_predicted_precision: Tensor,
    posterior_precision: Tensor,
) -> Tensor:
    if target_direction.shape != child_predicted_precision.shape:
        raise ValueError("temporal HGF reverse child precision shape differs")
    if target_direction.shape != posterior_precision.shape:
        raise ValueError("temporal HGF reverse posterior precision shape differs")
    if target_direction.ndim < 3:
        raise ValueError("temporal HGF reverse direction must start with layer")
    layers = target_direction.shape[0]
    width = target_direction.shape[-1]
    if reciprocal_reverse.shape != (layers, width, width):
        raise ValueError("temporal HGF reverse map shape differs")
    samples = target_direction.numel() // (layers * width)
    direction = target_direction.reshape(layers, samples, width)
    child = child_predicted_precision.to(target_direction.dtype).reshape(
        layers, samples, width
    )
    posterior = posterior_precision.to(target_direction.dtype).reshape(
        layers, samples, width
    )
    result = torch.bmm(direction / posterior, reciprocal_reverse) * child
    return result.reshape_as(target_direction)


__all__ = [
    "PROCESS_VARIANCE",
    "TONIC_LOG_VOLATILITY",
    "temporal_hgf_bottom_up",
    "temporal_hgf_precision_scan",
    "temporal_hgf_precision_sequence",
    "temporal_hgf_reverse_message",
]
