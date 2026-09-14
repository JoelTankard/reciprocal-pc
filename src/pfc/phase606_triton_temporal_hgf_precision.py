from __future__ import annotations

import math

import torch
from torch import Tensor
import triton
import triton.language as tl


PROCESS_VARIANCE = math.exp(-10.0)


@triton.jit
def _squared_coupling_evidence_kernel(
    child,
    squared_coupling,
    evidence,
    rows,
    width: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_PARENTS: tl.constexpr,
    BLOCK_LOWER: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    parent_offsets = (
        tl.program_id(1) * BLOCK_PARENTS + tl.arange(0, BLOCK_PARENTS)
    )
    accumulator = tl.zeros((BLOCK_ROWS, BLOCK_PARENTS), dtype=tl.float32)
    for lower_start in tl.range(0, width, BLOCK_LOWER):
        lower_offsets = lower_start + tl.arange(0, BLOCK_LOWER)
        child_values = tl.load(
            child + row_offsets[:, None] * width + lower_offsets[None, :],
            mask=(row_offsets[:, None] < rows) & (lower_offsets[None, :] < width),
            other=0.0,
        ).to(tl.float32)
        coupling_values = tl.load(
            squared_coupling
            + lower_offsets[:, None] * width
            + parent_offsets[None, :],
            mask=(lower_offsets[:, None] < width)
            & (parent_offsets[None, :] < width),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.dot(
            child_values,
            coupling_values,
            input_precision="ieee",
        )
    tl.store(
        evidence + row_offsets[:, None] * width + parent_offsets[None, :],
        accumulator,
        mask=(row_offsets[:, None] < rows) & (parent_offsets[None, :] < width),
    )


@triton.jit
def _temporal_hgf_precision_scan_kernel(
    evidence,
    initial_posterior,
    reset_mask,
    predicted,
    posterior,
    evidence_stride_batch: tl.constexpr,
    evidence_stride_sequence: tl.constexpr,
    evidence_stride_width: tl.constexpr,
    initial_stride_batch: tl.constexpr,
    initial_stride_width: tl.constexpr,
    output_stride_batch: tl.constexpr,
    output_stride_sequence: tl.constexpr,
    output_stride_width: tl.constexpr,
    sequence,
    width: tl.constexpr,
    process_variance: tl.constexpr,
    HAS_RESET: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    batch = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
    mask = channels < width
    previous = tl.load(
        initial_posterior
        + batch * initial_stride_batch
        + channels * initial_stride_width,
        mask=mask,
        other=1.0,
    ).to(tl.float32)
    position = 0
    while position < sequence:
        if HAS_RESET:
            reset = tl.load(reset_mask + batch * sequence + position)
            previous = tl.where(reset, 1.0, previous)
        current_predicted = previous / (1.0 + process_variance * previous)
        current_evidence = tl.load(
            evidence
            + batch * evidence_stride_batch
            + position * evidence_stride_sequence
            + channels * evidence_stride_width,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        current_posterior = current_predicted + current_evidence
        output_offset = (
            batch * output_stride_batch
            + position * output_stride_sequence
            + channels * output_stride_width
        )
        tl.store(predicted + output_offset, current_predicted, mask=mask)
        tl.store(posterior + output_offset, current_posterior, mask=mask)
        previous = current_posterior
        position += 1


def triton_temporal_hgf_precision_scan(
    evidence: Tensor,
    initial_posterior: Tensor,
    reset_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if evidence.ndim != 3:
        raise ValueError("Phase606 evidence must have [batch,sequence,width] shape")
    batch, sequence, width = evidence.shape
    if min(batch, sequence, width) < 1:
        raise ValueError("Phase606 evidence dimensions must be positive")
    if initial_posterior.shape != (batch, width):
        raise ValueError("Phase606 initial posterior shape differs")
    if any(not tensor.is_cuda for tensor in (evidence, initial_posterior)):
        raise ValueError("Phase606 scan requires CUDA tensors")
    if any(tensor.dtype != torch.float32 for tensor in (evidence, initial_posterior)):
        raise ValueError("Phase606 scan requires FP32 tensors")
    if reset_mask is not None:
        if reset_mask.shape != (batch, sequence):
            raise ValueError("Phase638 reset mask shape differs")
        if not reset_mask.is_cuda or reset_mask.dtype != torch.bool:
            raise ValueError("Phase638 reset mask must be CUDA bool")
    evidence = evidence.contiguous()
    initial_posterior = initial_posterior.contiguous()
    has_reset = reset_mask is not None
    reset_mask = (
        reset_mask.contiguous()
        if reset_mask is not None
        else torch.empty(1, device=evidence.device, dtype=torch.bool)
    )
    predicted = torch.empty_like(evidence)
    posterior = torch.empty_like(evidence)
    block_width = 32
    _temporal_hgf_precision_scan_kernel[
        (batch, triton.cdiv(width, block_width))
    ](
        evidence,
        initial_posterior,
        reset_mask,
        predicted,
        posterior,
        evidence_stride_batch=evidence.stride(0),
        evidence_stride_sequence=evidence.stride(1),
        evidence_stride_width=evidence.stride(2),
        initial_stride_batch=initial_posterior.stride(0),
        initial_stride_width=initial_posterior.stride(1),
        output_stride_batch=predicted.stride(0),
        output_stride_sequence=predicted.stride(1),
        output_stride_width=predicted.stride(2),
        sequence=sequence,
        width=width,
        process_variance=PROCESS_VARIANCE,
        HAS_RESET=has_reset,
        BLOCK_WIDTH=block_width,
        num_warps=1,
    )
    return predicted, posterior


def triton_squared_coupling_evidence(
    child: Tensor,
    squared_coupling: Tensor,
) -> Tensor:
    if child.ndim != 3:
        raise ValueError("Phase606 child precision must be [batch,sequence,width]")
    batch, sequence, width = child.shape
    if squared_coupling.shape != (width, width):
        raise ValueError("Phase606 squared coupling shape differs")
    if any(not tensor.is_cuda for tensor in (child, squared_coupling)):
        raise ValueError("Phase606 evidence kernel requires CUDA tensors")
    if any(tensor.dtype != torch.float32 for tensor in (child, squared_coupling)):
        raise ValueError("Phase606 evidence kernel requires FP32 tensors")
    child = child.contiguous()
    squared_coupling = squared_coupling.contiguous()
    evidence = torch.empty_like(child)
    rows = batch * sequence
    block_rows = 16
    block_parents = 32
    block_lower = 32
    _squared_coupling_evidence_kernel[
        (triton.cdiv(rows, block_rows), triton.cdiv(width, block_parents))
    ](
        child,
        squared_coupling,
        evidence,
        rows,
        width=width,
        BLOCK_ROWS=block_rows,
        BLOCK_PARENTS=block_parents,
        BLOCK_LOWER=block_lower,
        num_warps=4,
    )
    return evidence


__all__ = [
    "triton_squared_coupling_evidence",
    "triton_temporal_hgf_precision_scan",
]
