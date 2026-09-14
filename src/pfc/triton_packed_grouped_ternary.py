from __future__ import annotations

import math

import torch
from torch import Tensor
import triton
import triton.language as tl

from pfc.packed_grouped_ternary import (
    PackedGroupedTernaryMatrix,
    _validate_matmul_selection,
)


@triton.jit
def _packed_grouped_ternary_matmul_kernel(
    inputs,
    codes,
    scales,
    output,
    walsh_token_ids,
    INPUT_MATRIX_STRIDE: tl.constexpr,
    INPUT_ROW_STRIDE: tl.constexpr,
    INPUT_FEATURE_STRIDE: tl.constexpr,
    INPUT_FEATURES: tl.constexpr,
    FULL_INPUT_FEATURES: tl.constexpr,
    FULL_OUTPUT_FEATURES: tl.constexpr,
    INPUT_OFFSET: tl.constexpr,
    OUTPUT_OFFSET: tl.constexpr,
    SELECTED_OUTPUTS: tl.constexpr,
    ROWS: tl.constexpr,
    MATRIX_OFFSET: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_INPUTS: tl.constexpr,
    BLOCK_OUTPUTS: tl.constexpr,
    ABSOLUTE_STATES: tl.constexpr,
    WALSH_INPUT: tl.constexpr,
    WALSH_OUTPUT: tl.constexpr,
):
    output_offsets = tl.program_id(0) * BLOCK_OUTPUTS + tl.arange(0, BLOCK_OUTPUTS)
    output_mask = output_offsets < SELECTED_OUTPUTS
    row = tl.program_id(1)
    local_matrix = tl.program_id(2)
    matrix = MATRIX_OFFSET + local_matrix
    accumulator = tl.zeros((BLOCK_OUTPUTS,), dtype=tl.float32)
    matrix_weight_base = matrix * FULL_INPUT_FEATURES * FULL_OUTPUT_FEATURES
    input_row_base = local_matrix * INPUT_MATRIX_STRIDE + row * INPUT_ROW_STRIDE
    if WALSH_INPUT or WALSH_OUTPUT:
        token_id = tl.load(walsh_token_ids).to(tl.int64)
        route = ((token_id * 2654435761) & 0xFFFFFFFF) >> 30

    for input_start in tl.range(0, INPUT_FEATURES, BLOCK_INPUTS, num_stages=1):
        input_offsets = input_start + tl.arange(0, BLOCK_INPUTS)
        input_mask = input_offsets < INPUT_FEATURES
        input_values = tl.load(
            inputs + input_row_base + input_offsets * INPUT_FEATURE_STRIDE,
            mask=input_mask,
            other=0.0,
        ).to(tl.float32)
        if WALSH_INPUT:
            input_coordinates = INPUT_OFFSET + input_offsets
            masked = route & input_coordinates
            parity = (masked & 1) ^ (masked >> 1)
            input_values *= 1.0 - 2.0 * parity.to(tl.float32)
        weight_offsets = (
            matrix_weight_base
            + (INPUT_OFFSET + input_offsets[:, None]) * FULL_OUTPUT_FEATURES
            + OUTPUT_OFFSET
            + output_offsets[None, :]
        )
        byte_offsets = weight_offsets // 4
        shifts = (weight_offsets % 4) * 2
        packed_values = tl.load(
            codes + byte_offsets,
            mask=input_mask[:, None] & output_mask[None, :],
            other=0x55,
        )
        states = ((packed_values >> shifts) & 0b11).to(tl.int8) - 1
        if ABSOLUTE_STATES:
            states = tl.abs(states)
        scale_values = tl.load(
            scales + weight_offsets // GROUP_SIZE,
            mask=input_mask[:, None] & output_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        products = input_values[:, None] * states.to(tl.float32) * scale_values
        accumulator += tl.sum(products, axis=0)

    if WALSH_OUTPUT:
        output_coordinates = OUTPUT_OFFSET + output_offsets
        masked = route & output_coordinates
        parity = (masked & 1) ^ (masked >> 1)
        accumulator *= 1.0 - 2.0 * parity.to(tl.float32)

    output_base = (local_matrix * ROWS + row) * SELECTED_OUTPUTS
    tl.store(
        output + output_base + output_offsets,
        accumulator,
        mask=output_mask,
    )


def _regular_row_layout(inputs: Tensor, leading_rank: int) -> tuple[int, int]:
    row_shape = inputs.shape[leading_rank:-1]
    rows = math.prod(row_shape)
    row_stride = inputs.stride(-2) if row_shape else 0
    expected = row_stride
    for dimension in range(inputs.ndim - 2, leading_rank - 1, -1):
        if inputs.stride(dimension) != expected:
            raise ValueError("packed grouped ternary input rows are not regular")
        expected *= inputs.shape[dimension]
    return rows, row_stride


def triton_packed_grouped_ternary_matmul(
    inputs: Tensor,
    packed: PackedGroupedTernaryMatrix,
    *,
    matrix_index: int | None,
    input_offset: int,
    output_offset: int,
    output_features: int,
    absolute_states: bool = False,
    walsh_token_ids: Tensor | None = None,
    walsh_input: bool = False,
    walsh_output: bool = False,
) -> Tensor:
    _validate_matmul_selection(
        inputs,
        packed,
        matrix_index=matrix_index,
        input_offset=input_offset,
        output_offset=output_offset,
        output_features=output_features,
    )
    if not inputs.is_cuda or not packed.codes.is_cuda or not packed.scales.is_cuda:
        raise ValueError("packed grouped ternary Triton path requires CUDA")
    if inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("packed grouped ternary input dtype differs")
    if packed.scales.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("packed grouped ternary scale dtype differs")
    if not (inputs.device == packed.codes.device == packed.scales.device):
        raise ValueError("packed grouped ternary CUDA devices differ")
    if inputs.stride(-1) <= 0:
        raise ValueError("packed grouped ternary feature stride differs")
    walsh_enabled = bool(walsh_input or walsh_output)
    if walsh_enabled:
        if walsh_token_ids is None:
            raise ValueError("Walsh-bound packed matmul requires token ids")
        if walsh_token_ids.numel() != 1:
            raise ValueError("Walsh-bound packed matmul requires one token id")
        if walsh_token_ids.device != inputs.device:
            raise ValueError("Walsh-bound packed matmul token device differs")
        if walsh_token_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("Walsh-bound packed matmul token dtype differs")
    elif walsh_token_ids is not None:
        raise ValueError("Walsh token ids supplied without a bound coordinate")

    if matrix_index is None:
        leading_rank = len(packed.leading_shape)
        if leading_rank != 1:
            raise ValueError("packed grouped ternary Triton leading rank differs")
        local_matrices = packed.matrix_count
        matrix_offset = 0
        input_matrix_stride = inputs.stride(0)
    else:
        leading_rank = 0
        local_matrices = 1
        matrix_offset = matrix_index
        input_matrix_stride = 0
    rows, input_row_stride = _regular_row_layout(inputs, leading_rank)
    output = torch.empty(
        (*inputs.shape[:-1], output_features),
        device=inputs.device,
        dtype=inputs.dtype,
    )
    grid = (triton.cdiv(output_features, 32), rows, local_matrices)
    _packed_grouped_ternary_matmul_kernel[grid](
        inputs,
        packed.codes,
        packed.scales,
        output,
        walsh_token_ids if walsh_token_ids is not None else inputs,
        INPUT_MATRIX_STRIDE=input_matrix_stride,
        INPUT_ROW_STRIDE=input_row_stride,
        INPUT_FEATURE_STRIDE=inputs.stride(-1),
        INPUT_FEATURES=inputs.shape[-1],
        FULL_INPUT_FEATURES=packed.input_features,
        FULL_OUTPUT_FEATURES=packed.output_features,
        INPUT_OFFSET=input_offset,
        OUTPUT_OFFSET=output_offset,
        SELECTED_OUTPUTS=output_features,
        ROWS=rows,
        MATRIX_OFFSET=matrix_offset,
        GROUP_SIZE=packed.group_size,
        BLOCK_INPUTS=32,
        BLOCK_OUTPUTS=32,
        ABSOLUTE_STATES=absolute_states,
        WALSH_INPUT=walsh_input,
        WALSH_OUTPUT=walsh_output,
        num_warps=4,
    )
    return output
