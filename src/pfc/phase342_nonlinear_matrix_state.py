from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


NONLINEAR_MATRIX_HEADS = 16
NONLINEAR_MATRIX_KEY_WIDTH = 16
NONLINEAR_MATRIX_VALUE_WIDTH = 16
NONLINEAR_MATRIX_MODEL_WIDTH = (
    NONLINEAR_MATRIX_HEADS * NONLINEAR_MATRIX_VALUE_WIDTH
)
NONLINEAR_MATRIX_PROJECTION_WIDTH = (
    2 * NONLINEAR_MATRIX_KEY_WIDTH + NONLINEAR_MATRIX_HEADS
)
NONLINEAR_MATRIX_STATE_VALUES = (
    NONLINEAR_MATRIX_HEADS
    * NONLINEAR_MATRIX_KEY_WIDTH
    * NONLINEAR_MATRIX_VALUE_WIDTH
)


def _validate_inputs(
    source: Tensor,
    initial_state: Tensor,
    input_projection: Tensor,
    transition: Tensor,
    forget_bias: Tensor,
    residual_weight: Tensor,
    output_norm: Tensor,
    output_scale: Tensor,
) -> tuple[Tensor, bool]:
    squeezed = source.ndim == 2
    if squeezed:
        source = source.unsqueeze(1)
    if source.ndim != 3 or source.shape[-1] != NONLINEAR_MATRIX_MODEL_WIDTH:
        raise ValueError("nonlinear matrix source must have shape [batch, sequence, 256]")
    batch = source.shape[0]
    expected_state = (
        batch,
        NONLINEAR_MATRIX_HEADS,
        NONLINEAR_MATRIX_KEY_WIDTH,
        NONLINEAR_MATRIX_VALUE_WIDTH,
    )
    if initial_state.shape != expected_state:
        raise ValueError("nonlinear matrix initial state has wrong shape")
    expected_shapes = (
        (
            input_projection,
            (
                NONLINEAR_MATRIX_MODEL_WIDTH,
                NONLINEAR_MATRIX_PROJECTION_WIDTH,
            ),
        ),
        (
            transition,
            (
                NONLINEAR_MATRIX_HEADS,
                NONLINEAR_MATRIX_VALUE_WIDTH,
                NONLINEAR_MATRIX_VALUE_WIDTH,
            ),
        ),
        ((forget_bias, (NONLINEAR_MATRIX_HEADS,))),
        ((residual_weight, (NONLINEAR_MATRIX_MODEL_WIDTH,))),
        ((output_norm, (NONLINEAR_MATRIX_MODEL_WIDTH,))),
        ((output_scale, (NONLINEAR_MATRIX_MODEL_WIDTH,))),
    )
    if any(tensor.shape != shape for tensor, shape in expected_shapes):
        raise ValueError("nonlinear matrix parameter has wrong shape")
    return source, squeezed


def eager_nonlinear_matrix_recurrence(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    transition: Tensor,
    forget: Tensor,
    initial_state: Tensor,
) -> tuple[Tensor, Tensor]:
    """Reference M2RNN recurrence. Sequence axis is explicit and causal."""

    state = initial_state
    reads = []
    for index in range(query.shape[1]):
        current_key = key[:, index]
        current_value = value[:, index]
        candidate = torch.tanh(
            torch.matmul(state, transition)
            + current_key[:, None, :, None] * current_value[:, :, None, :]
        )
        retention = forget[:, index, :, None, None]
        state = retention * state + (1.0 - retention) * candidate
        reads.append(
            torch.einsum(
                "bk,bnkv->bnv",
                query[:, index],
                state,
            )
        )
    if not reads:
        empty = value.new_empty(
            value.shape[0],
            0,
            NONLINEAR_MATRIX_HEADS,
            NONLINEAR_MATRIX_VALUE_WIDTH,
        )
        return empty, state
    return torch.stack(reads, dim=1), state


def nonlinear_matrix_state_sequence(
    source: Tensor,
    initial_state: Tensor,
    input_projection: Tensor,
    transition: Tensor,
    forget_bias: Tensor,
    residual_weight: Tensor,
    output_norm: Tensor,
    output_scale: Tensor,
    *,
    rms_eps: float,
) -> tuple[Tensor, Tensor]:
    """Return the zero-gated parent injection and final fixed-size matrix state."""

    source, squeezed = _validate_inputs(
        source,
        initial_state,
        input_projection,
        transition,
        forget_bias,
        residual_weight,
        output_norm,
        output_scale,
    )
    projected = torch.matmul(source, input_projection)
    query, key, forget_pre = projected.split(
        (
            NONLINEAR_MATRIX_KEY_WIDTH,
            NONLINEAR_MATRIX_KEY_WIDTH,
            NONLINEAR_MATRIX_HEADS,
        ),
        dim=-1,
    )
    query = F.silu(query)
    key = F.silu(key)
    value = F.silu(
        source.reshape(
            source.shape[0],
            source.shape[1],
            NONLINEAR_MATRIX_HEADS,
            NONLINEAR_MATRIX_VALUE_WIDTH,
        )
    )
    forget = torch.sigmoid(-(forget_pre + forget_bias))
    if source.is_cuda:
        from pfc.phase342_triton_nonlinear_matrix_state import (
            triton_nonlinear_matrix_recurrence,
        )

        read, final_state = triton_nonlinear_matrix_recurrence(
            *(
                tensor.to(torch.float16)
                for tensor in (
                    query,
                    key,
                    value,
                    transition,
                    forget,
                    initial_state,
                )
            )
        )
        read = read.to(source.dtype)
        final_state = final_state.to(initial_state.dtype)
    else:
        read, final_state = eager_nonlinear_matrix_recurrence(
            query,
            key,
            value,
            transition,
            forget,
            initial_state,
        )
    residual = read + value * residual_weight.view(
        NONLINEAR_MATRIX_HEADS,
        NONLINEAR_MATRIX_VALUE_WIDTH,
    )
    gated = residual * F.silu(
        source.reshape(
            source.shape[0],
            source.shape[1],
            NONLINEAR_MATRIX_HEADS,
            NONLINEAR_MATRIX_VALUE_WIDTH,
        )
    )
    flattened = gated.flatten(start_dim=-2)
    normalized = (
        flattened
        * torch.rsqrt(flattened.square().mean(dim=-1, keepdim=True) + rms_eps)
        * output_norm
    )
    injection = normalized * output_scale
    if squeezed:
        injection = injection.squeeze(1)
    return injection, final_state
