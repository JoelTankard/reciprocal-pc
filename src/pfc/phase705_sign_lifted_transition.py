from __future__ import annotations

import torch
from torch import Tensor


def lift_sign_channels(value: Tensor) -> Tensor:
    """Lift one signed feature axis into positive-then-negative magnitudes."""

    return torch.cat((torch.relu(value), torch.relu(-value)), dim=-1)


def split_sign_lifted_matrix(matrix: Tensor) -> tuple[Tensor, Tensor]:
    if matrix.ndim < 2 or matrix.shape[-2] % 2:
        raise ValueError("sign-lifted matrix needs an even input feature axis")
    midpoint = matrix.shape[-2] // 2
    return matrix[..., :midpoint, :], matrix[..., midpoint:, :]


def initialize_sign_lifted_forward(signed: Tensor) -> Tensor:
    """Return [W, -W] along the input feature axis."""

    if signed.ndim < 2:
        raise ValueError("sign-lifted initialization requires a matrix")
    return torch.cat((signed, -signed), dim=-2)


def sign_lifted_projection(inputs: Tensor, matrix: Tensor) -> Tensor:
    if matrix.shape[-2] != 2 * inputs.shape[-1]:
        raise ValueError("sign-lifted projection feature width differs")
    return torch.matmul(lift_sign_channels(inputs), matrix)


def sign_lifted_tangent_projection(
    inputs: Tensor,
    input_tangent: Tensor,
    matrix: Tensor,
) -> Tensor:
    """Apply the fixed-owner JVP of the sign-lifted projection."""

    if inputs.shape != input_tangent.shape:
        raise ValueError("sign-lifted tangent input shapes differ")
    lifted_tangent = torch.cat(
        (
            (inputs >= 0).to(input_tangent.dtype) * input_tangent,
            -(inputs < 0).to(input_tangent.dtype) * input_tangent,
        ),
        dim=-1,
    )
    if matrix.shape[-2] != lifted_tangent.shape[-1]:
        raise ValueError("sign-lifted tangent projection width differs")
    return torch.matmul(lifted_tangent, matrix)


def masked_reciprocal_message(
    hidden: Tensor,
    output_direction: Tensor,
    positive_feedback: Tensor,
    negative_feedback: Tensor,
) -> Tensor:
    """Collapse independent source-sign messages with fixed zero ownership."""

    if positive_feedback.shape != negative_feedback.shape:
        raise ValueError("sign-lifted reciprocal map shapes differ")
    if output_direction.shape[-1] != positive_feedback.shape[-2]:
        raise ValueError("sign-lifted reciprocal output width differs")
    if hidden.shape[-1] != positive_feedback.shape[-1]:
        raise ValueError("sign-lifted reciprocal hidden width differs")
    positive_message = torch.matmul(output_direction, positive_feedback)
    negative_message = torch.matmul(output_direction, negative_feedback)
    positive_owner = hidden >= 0
    negative_owner = hidden < 0
    return (
        positive_owner.to(positive_message.dtype) * positive_message
        - negative_owner.to(negative_message.dtype) * negative_message
    )


def local_sign_lifted_forward_gradient(
    hidden: Tensor,
    output_direction: Tensor,
) -> Tensor:
    if hidden.ndim < 2 or output_direction.ndim != hidden.ndim:
        raise ValueError("sign-lifted forward gradient ranks differ")
    if hidden.shape[:-1] != output_direction.shape[:-1] or hidden.shape[-2] == 0:
        raise ValueError("sign-lifted forward sample counts differ")
    return -(
        lift_sign_channels(hidden).transpose(-1, -2) @ output_direction
    ) / hidden.shape[-2]


def local_sign_lifted_reciprocal_gradients(
    output_direction: Tensor,
    hidden: Tensor,
) -> tuple[Tensor, Tensor]:
    if hidden.ndim < 2 or output_direction.ndim != hidden.ndim:
        raise ValueError("sign-lifted reciprocal gradient ranks differ")
    if hidden.shape[:-1] != output_direction.shape[:-1] or hidden.shape[-2] == 0:
        raise ValueError("sign-lifted reciprocal sample counts differ")
    positive = torch.relu(hidden)
    negative = torch.relu(-hidden)
    samples = hidden.shape[-2]
    return (
        -(output_direction.transpose(-1, -2) @ positive) / samples,
        -(output_direction.transpose(-1, -2) @ negative) / samples,
    )
