from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


RANK = tl.constexpr(32)


@triton.jit
def _predictive_delta_reads_kernel(
    keys,
    errors,
    reads,
    key_stride_batch: tl.constexpr,
    key_stride_sequence: tl.constexpr,
    key_stride_rank: tl.constexpr,
    read_stride_batch: tl.constexpr,
    read_stride_sequence: tl.constexpr,
    read_stride_scale: tl.constexpr,
    read_stride_rank: tl.constexpr,
    SEQUENCE: tl.constexpr,
    FAST_DECAY: tl.constexpr,
):
    program = tl.program_id(0)
    scale = program & 1
    batch = program >> 1
    channels = tl.arange(0, RANK)
    matrix = tl.zeros((RANK, RANK), tl.float32)
    decay = tl.where(scale == 0, FAST_DECAY, libdevice.sqrt(FAST_DECAY))

    for step in tl.range(0, SEQUENCE, num_stages=1):
        if step > 0:
            key = tl.load(
                keys
                + batch * key_stride_batch
                + (step - 1) * key_stride_sequence
                + channels * key_stride_rank
            ).to(tl.float32)
            value = tl.load(
                errors
                + batch * key_stride_batch
                + step * key_stride_sequence
                + channels * key_stride_rank
            ).to(tl.float32)
            matrix = decay * matrix
            predicted = tl.sum(key[:, None] * matrix, axis=0)
            residual = value - predicted
            matrix = matrix + key[:, None] * residual[None, :]
        query = tl.load(
            keys
            + batch * key_stride_batch
            + step * key_stride_sequence
            + channels * key_stride_rank
        ).to(tl.float32)
        read = tl.sum(query[:, None] * matrix, axis=0)
        tl.store(
            reads
            + batch * read_stride_batch
            + step * read_stride_sequence
            + scale * read_stride_scale
            + channels * read_stride_rank,
            read,
        )


def triton_predictive_delta_reads(keys: Tensor, errors: Tensor) -> Tensor:
    if keys.shape != errors.shape or keys.ndim != 3 or keys.shape[-1] != 32:
        raise ValueError("Phase275 keys/errors must be [batch, sequence, 32]")
    if keys.dtype != torch.float32 or errors.dtype != torch.float32:
        raise ValueError("Phase275 keys/errors must be float32")
    if not keys.is_cuda or not errors.is_cuda:
        raise ValueError("Phase275 delta kernel requires CUDA")
    if not keys.is_contiguous() or not errors.is_contiguous():
        raise ValueError("Phase275 delta kernel inputs must be contiguous")
    batch, sequence, _ = keys.shape
    reads = torch.empty(
        batch,
        sequence,
        2,
        32,
        device=keys.device,
        dtype=torch.float32,
    )
    _predictive_delta_reads_kernel[(batch * 2,)](
        keys,
        errors,
        reads,
        key_stride_batch=keys.stride(0),
        key_stride_sequence=keys.stride(1),
        key_stride_rank=keys.stride(2),
        read_stride_batch=reads.stride(0),
        read_stride_sequence=reads.stride(1),
        read_stride_scale=reads.stride(2),
        read_stride_rank=reads.stride(3),
        SEQUENCE=sequence,
        FAST_DECAY=0.99,
        num_warps=8,
    )
    return reads.mean(dim=2)
