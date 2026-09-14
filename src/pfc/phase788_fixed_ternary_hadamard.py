from __future__ import annotations

import math

import torch
from torch import Tensor


def _validate(values: Tensor, signs: Tensor) -> None:
    if values.ndim < 1:
        raise ValueError("fixed lexical Hadamard values require a feature axis")
    width = values.shape[-1]
    if width <= 0 or width & (width - 1):
        raise ValueError("fixed lexical Hadamard width must be a power of two")
    if signs.shape != (width,):
        raise ValueError("fixed lexical Hadamard signs must match width")
    if signs.device != values.device:
        raise ValueError("fixed lexical Hadamard signs must share device")
    if signs.dtype != torch.int8:
        raise ValueError("fixed lexical Hadamard signs must use int8 storage")


def validate_fixed_ternary_hadamard_signs(signs: Tensor) -> None:
    if signs.ndim != 1 or signs.dtype != torch.int8:
        raise ValueError("fixed lexical Hadamard signs must be one-dimensional int8")
    if not bool(torch.all((signs == -1) | (signs == 1)).item()):
        raise ValueError("fixed lexical Hadamard signs must be plus or minus one")


def fixed_ternary_hadamard_lexical_predictor(
    values: Tensor,
    signs: Tensor,
) -> Tensor:
    """Apply Q(x) = H(x * s) without materializing Q."""

    _validate(values, signs)
    from pfc.native_pc_lm import _normalized_hadamard

    return _normalized_hadamard(values * signs.to(values.dtype))


def fixed_ternary_hadamard_lexical_inverse(
    direction: Tensor,
    signs: Tensor,
) -> Tensor:
    """Apply Q^-1(y) = H(y) * s through fixed, parameter-free algebra."""

    _validate(direction, signs)
    from pfc.native_pc_lm import _normalized_hadamard

    return _normalized_hadamard(direction) * signs.to(direction.dtype)


def materialize_fixed_ternary_hadamard(
    signs: Tensor,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Materialize Q for bounded mechanics oracles only."""

    width = signs.numel()
    identity = torch.eye(width, device=signs.device, dtype=dtype)
    return fixed_ternary_hadamard_lexical_predictor(identity, signs)


def fixed_ternary_hadamard_scale(width: int) -> float:
    if width <= 0 or width & (width - 1):
        raise ValueError("fixed lexical Hadamard width must be a power of two")
    return 1.0 / math.sqrt(width)


__all__ = [
    "fixed_ternary_hadamard_lexical_inverse",
    "fixed_ternary_hadamard_lexical_predictor",
    "fixed_ternary_hadamard_scale",
    "materialize_fixed_ternary_hadamard",
    "validate_fixed_ternary_hadamard_signs",
]
