# Copyright (c) 2026, Mayank Mishra
# SPDX-License-Identifier: Apache-2.0
#
# Modified for ReciprocalPC Phase342: fixed 16x16 heads, shared query/key
# heads, explicit initial/final state gradients, and no kernel autotuning.

from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl


_HEADS = 16
_KEY_WIDTH = 16
_VALUE_WIDTH = 16


@triton.jit
def _tanh(value):
    return 2.0 / (1.0 + tl.exp(-2.0 * value)) - 1.0


@triton.jit
def _phase342_forward_kernel(
    query,
    key,
    value,
    transition,
    forget,
    initial_state,
    state_sequence,
    final_state,
    output,
    SEQUENCE,
    HEADS: tl.constexpr,
    KEY_WIDTH: tl.constexpr,
    VALUE_WIDTH: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    rows = tl.arange(0, KEY_WIDTH)
    columns = tl.arange(0, VALUE_WIDTH)

    transition_offsets = (
        head * VALUE_WIDTH * VALUE_WIDTH
        + columns[:, None] * VALUE_WIDTH
        + columns[None, :]
    )
    weight = tl.load(transition + transition_offsets)
    state_offsets = (
        batch * HEADS * KEY_WIDTH * VALUE_WIDTH
        + head * KEY_WIDTH * VALUE_WIDTH
        + rows[:, None] * VALUE_WIDTH
        + columns[None, :]
    )
    state = tl.load(initial_state + state_offsets)

    for sequence_index in range(0, SEQUENCE):
        shared_base = (batch * SEQUENCE + sequence_index) * KEY_WIDTH
        query_value = tl.load(query + shared_base + rows)
        key_value = tl.load(key + shared_base + rows)
        head_base = (
            (batch * SEQUENCE + sequence_index) * HEADS + head
        )
        value_value = tl.load(
            value + head_base * VALUE_WIDTH + columns
        )
        retention = tl.load(forget + head_base)

        transformed = tl.dot(state, weight, out_dtype=tl.float32)
        candidate = _tanh(
            transformed
            + key_value[:, None].to(tl.float32)
            * value_value[None, :].to(tl.float32)
        )
        state = (
            retention.to(tl.float32) * state.to(tl.float32)
            + (1.0 - retention.to(tl.float32)) * candidate
        )
        sequence_state_offsets = (
            (
                (batch * SEQUENCE + sequence_index) * HEADS
                + head
            )
            * KEY_WIDTH
            * VALUE_WIDTH
            + rows[:, None] * VALUE_WIDTH
            + columns[None, :]
        )
        tl.store(state_sequence + sequence_state_offsets, state)
        read = tl.sum(
            query_value[:, None].to(tl.float32) * state,
            axis=0,
        )
        tl.store(
            output + head_base * VALUE_WIDTH + columns,
            read,
        )
        state = state.to(tl.float16)

    tl.store(final_state + state_offsets, state)


@triton.jit
def _phase342_backward_kernel(
    query,
    key,
    value,
    transition,
    forget,
    initial_state,
    state_sequence,
    output_gradient,
    final_state_gradient,
    query_gradient,
    key_gradient,
    value_gradient,
    transition_gradient,
    forget_gradient,
    initial_state_gradient,
    SEQUENCE,
    HEADS: tl.constexpr,
    KEY_WIDTH: tl.constexpr,
    VALUE_WIDTH: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    rows = tl.arange(0, KEY_WIDTH)
    columns = tl.arange(0, VALUE_WIDTH)

    transition_offsets = (
        head * VALUE_WIDTH * VALUE_WIDTH
        + columns[:, None] * VALUE_WIDTH
        + columns[None, :]
    )
    weight = tl.load(transition + transition_offsets)
    initial_offsets = (
        batch * HEADS * KEY_WIDTH * VALUE_WIDTH
        + head * KEY_WIDTH * VALUE_WIDTH
        + rows[:, None] * VALUE_WIDTH
        + columns[None, :]
    )
    state_gradient = tl.load(
        final_state_gradient + initial_offsets
    ).to(tl.float32)
    weight_gradient = tl.zeros(
        (VALUE_WIDTH, VALUE_WIDTH),
        dtype=tl.float32,
    )

    for sequence_index in range(SEQUENCE - 1, -1, -1):
        head_base = (
            (batch * SEQUENCE + sequence_index) * HEADS + head
        )
        sequence_state_offsets = (
            head_base * KEY_WIDTH * VALUE_WIDTH
            + rows[:, None] * VALUE_WIDTH
            + columns[None, :]
        )
        state = tl.load(
            state_sequence + sequence_state_offsets
        ).to(tl.float32)
        if sequence_index == 0:
            previous_state = tl.load(
                initial_state + initial_offsets
            ).to(tl.float32)
        else:
            previous_state = tl.load(
                state_sequence
                + sequence_state_offsets
                - HEADS * KEY_WIDTH * VALUE_WIDTH
            ).to(tl.float32)

        shared_base = (batch * SEQUENCE + sequence_index) * KEY_WIDTH
        query_value = tl.load(query + shared_base + rows)
        key_value = tl.load(key + shared_base + rows)
        value_value = tl.load(
            value + head_base * VALUE_WIDTH + columns
        )
        retention = tl.load(forget + head_base).to(tl.float32)
        output_delta = tl.load(
            output_gradient + head_base * VALUE_WIDTH + columns
        ).to(tl.float32)

        query_delta = tl.sum(
            state * output_delta[None, :],
            axis=1,
        )
        tl.atomic_add(
            query_gradient + shared_base + rows,
            query_delta,
        )
        total_state_delta = (
            query_value[:, None].to(tl.float32)
            * output_delta[None, :]
            + state_gradient
        )

        transformed = tl.dot(
            previous_state.to(tl.float16),
            weight,
            out_dtype=tl.float32,
        )
        candidate = _tanh(
            transformed
            + key_value[:, None].to(tl.float32)
            * value_value[None, :].to(tl.float32)
        )
        forget_delta = tl.sum(
            total_state_delta * (previous_state - candidate)
        )
        tl.store(forget_gradient + head_base, forget_delta)

        candidate_delta = (
            (1.0 - retention)
            * total_state_delta
            * (1.0 - candidate * candidate)
        )
        state_gradient = (
            retention * total_state_delta
            + tl.dot(
                candidate_delta.to(tl.float16),
                tl.trans(weight),
                out_dtype=tl.float32,
            )
        )
        weight_gradient += tl.dot(
            tl.trans(previous_state.to(tl.float16)),
            candidate_delta.to(tl.float16),
            out_dtype=tl.float32,
        )
        key_delta = tl.sum(
            candidate_delta * value_value[None, :].to(tl.float32),
            axis=1,
        )
        value_delta = tl.sum(
            candidate_delta * key_value[:, None].to(tl.float32),
            axis=0,
        )
        tl.atomic_add(
            key_gradient + shared_base + rows,
            key_delta,
        )
        tl.store(
            value_gradient + head_base * VALUE_WIDTH + columns,
            value_delta,
        )

    tl.store(initial_state_gradient + initial_offsets, state_gradient)
    tl.atomic_add(
        transition_gradient + transition_offsets,
        weight_gradient,
    )


def _validate(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    transition: Tensor,
    forget: Tensor,
    initial_state: Tensor,
) -> tuple[int, int]:
    if query.ndim != 3:
        raise ValueError("Phase342 query must have [batch, sequence, 16]")
    batch, sequence, key_width = query.shape
    expected = (
        (key, (batch, sequence, _KEY_WIDTH)),
        (value, (batch, sequence, _HEADS, _VALUE_WIDTH)),
        (transition, (_HEADS, _VALUE_WIDTH, _VALUE_WIDTH)),
        (forget, (batch, sequence, _HEADS)),
        (initial_state, (batch, _HEADS, _KEY_WIDTH, _VALUE_WIDTH)),
    )
    if key_width != _KEY_WIDTH or any(
        tensor.shape != shape for tensor, shape in expected
    ):
        raise ValueError("Phase342 recurrence input shape differs")
    tensors = (query, key, value, transition, forget, initial_state)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("Phase342 Triton recurrence requires CUDA")
    if any(tensor.dtype != torch.float16 for tensor in tensors):
        raise ValueError("Phase342 Triton recurrence requires float16")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("Phase342 Triton recurrence inputs must share one device")
    if sequence <= 0:
        raise ValueError("Phase342 Triton recurrence requires a nonempty sequence")
    return batch, sequence


def _forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    transition: Tensor,
    forget: Tensor,
    initial_state: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    batch, sequence = _validate(
        query,
        key,
        value,
        transition,
        forget,
        initial_state,
    )
    output = torch.empty(
        batch,
        sequence,
        _HEADS,
        _VALUE_WIDTH,
        device=query.device,
        dtype=query.dtype,
    )
    state_sequence = torch.empty(
        batch,
        sequence,
        _HEADS,
        _KEY_WIDTH,
        _VALUE_WIDTH,
        device=query.device,
        dtype=query.dtype,
    )
    final_state = torch.empty_like(initial_state)
    _phase342_forward_kernel[(batch, _HEADS)](
        query,
        key,
        value,
        transition,
        forget,
        initial_state,
        state_sequence,
        final_state,
        output,
        sequence,
        HEADS=_HEADS,
        KEY_WIDTH=_KEY_WIDTH,
        VALUE_WIDTH=_VALUE_WIDTH,
        num_warps=4,
        num_stages=1,
    )
    return output, final_state, state_sequence


class _Phase342NonlinearMatrixFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        transition: Tensor,
        forget: Tensor,
        initial_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        inputs = tuple(
            tensor.contiguous()
            for tensor in (
                query,
                key,
                value,
                transition,
                forget,
                initial_state,
            )
        )
        output, final_state, state_sequence = _forward(*inputs)
        ctx.save_for_backward(*inputs, state_sequence)
        return output, final_state

    @staticmethod
    def backward(
        ctx,
        output_gradient: Tensor | None,
        final_state_gradient: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        (
            query,
            key,
            value,
            transition,
            forget,
            initial_state,
            state_sequence,
        ) = ctx.saved_tensors
        if output_gradient is None:
            output_gradient = torch.zeros_like(value)
        if final_state_gradient is None:
            final_state_gradient = torch.zeros_like(initial_state)
        output_gradient = output_gradient.contiguous()
        final_state_gradient = final_state_gradient.contiguous()
        query_gradient = torch.zeros_like(query, dtype=torch.float32)
        key_gradient = torch.zeros_like(key, dtype=torch.float32)
        value_gradient = torch.empty_like(value)
        transition_gradient = torch.zeros_like(transition, dtype=torch.float32)
        forget_gradient = torch.empty_like(forget)
        initial_state_gradient = torch.empty_like(initial_state)
        batch, sequence, _ = query.shape
        _phase342_backward_kernel[(batch, _HEADS)](
            query,
            key,
            value,
            transition,
            forget,
            initial_state,
            state_sequence,
            output_gradient,
            final_state_gradient,
            query_gradient,
            key_gradient,
            value_gradient,
            transition_gradient,
            forget_gradient,
            initial_state_gradient,
            sequence,
            HEADS=_HEADS,
            KEY_WIDTH=_KEY_WIDTH,
            VALUE_WIDTH=_VALUE_WIDTH,
            num_warps=4,
            num_stages=1,
        )
        return (
            query_gradient.to(query.dtype),
            key_gradient.to(key.dtype),
            value_gradient,
            transition_gradient.to(transition.dtype),
            forget_gradient,
            initial_state_gradient,
        )


def triton_nonlinear_matrix_recurrence(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    transition: Tensor,
    forget: Tensor,
    initial_state: Tensor,
) -> tuple[Tensor, Tensor]:
    _validate(query, key, value, transition, forget, initial_state)
    return _Phase342NonlinearMatrixFunction.apply(
        query,
        key,
        value,
        transition,
        forget,
        initial_state,
    )
