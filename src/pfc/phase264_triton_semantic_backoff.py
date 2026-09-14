from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


TABLE_COUNT = tl.constexpr(6)
TABLE_BLOCK = tl.constexpr(8)
MEDIAN_TABLES = tl.constexpr(3)
BUCKET_COUNT = tl.constexpr(64)
VALUE_WIDTH = tl.constexpr(8)
SMALLEST_POSITIVE_HALF = tl.constexpr(5.960464477539063e-8)


@triton.jit
def _half_multiply(left, right):
    return tl.inline_asm_elementwise(
        "mul.rn.f16x2 $0, $1, $2;",
        "=r,r,r",
        [left, right],
        dtype=tl.float16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _half_add(left, right):
    return tl.inline_asm_elementwise(
        "add.rn.f16x2 $0, $1, $2;",
        "=r,r,r",
        [left, right],
        dtype=tl.float16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _half_subtract(left, right):
    return tl.inline_asm_elementwise(
        "sub.rn.f16x2 $0, $1, $2;",
        "=r,r,r",
        [left, right],
        dtype=tl.float16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _hadamard_score(masked_context, channels, row: tl.constexpr):
    parity = channels & row
    parity = parity ^ (parity >> 16)
    parity = parity ^ (parity >> 8)
    parity = parity ^ (parity >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    signs = tl.where((parity & 1) == 0, 1.0, -1.0)
    return tl.sum(masked_context.to(tl.float32) * signs, axis=0)


@triton.jit
def _semantic_context_code_kernel(
    context,
    source,
    hash_mask,
    output_codes,
    CONTEXT_STRIDE_BATCH: tl.constexpr,
    CONTEXT_STRIDE_WIDTH: tl.constexpr,
    SOURCE_STRIDE_BATCH: tl.constexpr,
    SOURCE_STRIDE_WIDTH: tl.constexpr,
    CODE_STRIDE_BATCH: tl.constexpr,
    CODE_STRIDE_TABLE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CONTEXT_DECAY: tl.constexpr,
):
    batch = tl.program_id(0)
    channels = tl.arange(0, BLOCK_SIZE)
    width_mask = channels < WIDTH
    context_offsets = (
        batch * CONTEXT_STRIDE_BATCH + channels * CONTEXT_STRIDE_WIDTH
    )
    source_offsets = batch * SOURCE_STRIDE_BATCH + channels * SOURCE_STRIDE_WIDTH
    previous = tl.load(context + context_offsets, mask=width_mask, other=0.0).to(
        tl.float16
    )
    current_source = tl.load(
        source + source_offsets,
        mask=width_mask,
        other=0.0,
    ).to(tl.float16)
    decayed = (previous.to(tl.float32) * CONTEXT_DECAY).to(tl.float16)
    current = _half_add(decayed, current_source)
    tl.store(context + context_offsets, current, mask=width_mask)
    mask_values = tl.load(hash_mask + channels, mask=width_mask, other=0).to(
        tl.float16
    )
    masked_context = _half_multiply(current, mask_values)

    for table in range(TABLE_COUNT):
        code = 0
        for bit in range(6):
            score = _hadamard_score(
                masked_context,
                channels,
                table * 6 + bit,
            )
            code += tl.where(score > 0.0, 1 << bit, 0)
        sign_score = _hadamard_score(
            masked_context,
            channels,
            36 + table,
        )
        code += tl.where(sign_score > 0.0, 64, 0)
        tl.store(
            output_codes
            + batch * CODE_STRIDE_BATCH
            + table * CODE_STRIDE_TABLE,
            code,
        )


@triton.jit
def _semantic_memory_step_kernel(
    memory,
    normalizer,
    last_update,
    encoded_source,
    pending_codes,
    query_codes,
    output_groups,
    output_active,
    memory_stride_batch: tl.constexpr,
    memory_stride_rank: tl.constexpr,
    memory_stride_value: tl.constexpr,
    normalizer_stride_batch: tl.constexpr,
    normalizer_stride_rank: tl.constexpr,
    last_update_stride_batch: tl.constexpr,
    last_update_stride_rank: tl.constexpr,
    source_stride_batch: tl.constexpr,
    source_stride_value: tl.constexpr,
    code_stride_batch: tl.constexpr,
    code_stride_table: tl.constexpr,
    output_stride_batch: tl.constexpr,
    output_stride_group: tl.constexpr,
    output_stride_value: tl.constexpr,
    active_stride_batch: tl.constexpr,
    active_stride_group: tl.constexpr,
    next_token_count,
    WRITE_ENABLED: tl.constexpr,
    DYNAMIC_TOKEN_COUNT: tl.constexpr,
    FAST_DECAY: tl.constexpr,
    RMS_EPS: tl.constexpr,
):
    batch = tl.program_id(0)
    tables = tl.arange(0, TABLE_BLOCK)
    channels = tl.arange(0, VALUE_WIDTH)
    table_mask = tables < TABLE_COUNT
    matrix_mask = table_mask[:, None]
    current_token_count = (
        tl.load(next_token_count)
        if DYNAMIC_TOKEN_COUNT
        else next_token_count
    )

    pending_code = tl.load(
        pending_codes
        + batch * code_stride_batch
        + tables * code_stride_table,
        mask=table_mask,
        other=0,
    ).to(tl.int32)
    query_code = tl.load(
        query_codes + batch * code_stride_batch + tables * code_stride_table,
        mask=table_mask,
        other=0,
    ).to(tl.int32)
    pending_addresses = (pending_code & 63) + tables * BUCKET_COUNT
    query_addresses = (query_code & 63) + tables * BUCKET_COUNT
    pending_signs = tl.where((pending_code & 64) != 0, 1.0, -1.0).to(
        tl.float16
    )
    query_signs = tl.where((query_code & 64) != 0, 1.0, -1.0).to(
        tl.float16
    )
    decay = tl.where(
        tables < MEDIAN_TABLES,
        FAST_DECAY,
        libdevice.sqrt(FAST_DECAY),
    ).to(tl.float16)
    write_scale = _half_subtract(
        tl.full((TABLE_BLOCK,), 1.0, tl.float16),
        decay,
    )

    if WRITE_ENABLED:
        pending_normalizer_offsets = (
            batch * normalizer_stride_batch
            + pending_addresses * normalizer_stride_rank
        )
        pending_last_update_offsets = (
            batch * last_update_stride_batch
            + pending_addresses * last_update_stride_rank
        )
        elapsed = current_token_count - tl.load(
            last_update + pending_last_update_offsets,
            mask=table_mask,
            other=current_token_count,
        )
        write_decay = libdevice.pow(
            decay.to(tl.float32),
            elapsed.to(tl.float32),
        ).to(tl.float16)
        memory_offsets = (
            batch * memory_stride_batch
            + pending_addresses[:, None] * memory_stride_rank
            + channels[None, :] * memory_stride_value
        )
        source_value = tl.load(
            encoded_source
            + batch * source_stride_batch
            + channels * source_stride_value,
        ).to(tl.float16)
        old_memory = tl.load(memory + memory_offsets, mask=matrix_mask).to(
            tl.float16
        )
        updated_memory = _half_add(
            _half_multiply(write_decay[:, None], old_memory),
            _half_multiply(
                _half_multiply(pending_signs[:, None], source_value[None, :]),
                write_scale[:, None],
            ),
        )
        tl.store(memory + memory_offsets, updated_memory, mask=matrix_mask)
        old_normalizer = tl.load(
            normalizer + pending_normalizer_offsets,
            mask=table_mask,
            other=0.0,
        ).to(tl.float16)
        updated_normalizer = _half_add(
            _half_multiply(write_decay, old_normalizer),
            write_scale,
        )
        tl.store(
            normalizer + pending_normalizer_offsets,
            updated_normalizer,
            mask=table_mask,
        )
        tl.store(
            last_update + pending_last_update_offsets,
            current_token_count,
            mask=table_mask,
        )
        tl.debug_barrier()

    query_normalizer_offsets = (
        batch * normalizer_stride_batch
        + query_addresses * normalizer_stride_rank
    )
    query_last_update_offsets = (
        batch * last_update_stride_batch
        + query_addresses * last_update_stride_rank
    )
    query_elapsed = current_token_count - tl.load(
        last_update + query_last_update_offsets,
        mask=table_mask,
        other=current_token_count,
    )
    query_decay = libdevice.pow(
        decay.to(tl.float32),
        query_elapsed.to(tl.float32),
    ).to(tl.float16)
    query_memory_offsets = (
        batch * memory_stride_batch
        + query_addresses[:, None] * memory_stride_rank
        + channels[None, :] * memory_stride_value
    )
    query_memory = tl.load(
        memory + query_memory_offsets,
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float16)
    numerator = _half_multiply(query_memory, query_decay[:, None])
    numerator = _half_multiply(numerator, query_signs[:, None])
    query_normalizer = tl.load(
        normalizer + query_normalizer_offsets,
        mask=table_mask,
        other=0.0,
    ).to(tl.float16)
    raw_denominator = _half_multiply(query_normalizer, query_decay)
    denominator_floor = tl.maximum(
        _half_multiply(
            tl.full((TABLE_BLOCK,), RMS_EPS, tl.float16),
            write_scale,
        ),
        SMALLEST_POSITIVE_HALF,
    ).to(tl.float16)
    denominator = tl.maximum(raw_denominator, denominator_floor).to(tl.float16)
    table_reads = (numerator / denominator[:, None]).to(tl.float16)

    values = table_reads.to(tl.float32)
    positive_infinity = 1.0e20
    negative_infinity = -1.0e20
    for group in range(2):
        group_start = group * MEDIAN_TABLES
        group_table_mask = (
            (tables >= group_start) & (tables < group_start + MEDIAN_TABLES)
        )
        group_mask = group_table_mask[:, None]
        group_sum = tl.sum(tl.where(group_mask, values, 0.0), axis=0)
        group_min = tl.min(
            tl.where(group_mask, values, positive_infinity),
            axis=0,
        )
        group_max = tl.max(
            tl.where(group_mask, values, negative_infinity),
            axis=0,
        )
        group_read = (group_sum - group_min - group_max).to(tl.float16)
        tl.store(
            output_groups
            + batch * output_stride_batch
            + group * output_stride_group
            + channels * output_stride_value,
            group_read,
        )
        active_count = tl.sum(
            tl.where(group_table_mask & (raw_denominator > 0.0), 1, 0),
            axis=0,
        )
        tl.store(
            output_active
            + batch * active_stride_batch
            + group * active_stride_group,
            active_count >= 2,
        )


@triton.jit
def _compose_semantic_read_kernel(
    parent_read,
    fast_encoded_read,
    slow_encoded_read,
    fast_active,
    slow_active,
    value_decoder,
    parent_stride_batch: tl.constexpr,
    parent_stride_width: tl.constexpr,
    fast_stride_batch: tl.constexpr,
    fast_stride_value: tl.constexpr,
    slow_stride_batch: tl.constexpr,
    slow_stride_value: tl.constexpr,
    active_stride_batch: tl.constexpr,
    decoder_stride_value: tl.constexpr,
    decoder_stride_width: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    channels = tl.arange(0, BLOCK_SIZE)
    width_mask = channels < WIDTH
    parent_offsets = (
        batch * parent_stride_batch + channels * parent_stride_width
    )
    parent = tl.load(
        parent_read + parent_offsets,
        mask=width_mask,
        other=0.0,
    ).to(tl.float16)
    values = tl.arange(0, VALUE_WIDTH)
    fast_encoded = tl.load(
        fast_encoded_read
        + batch * fast_stride_batch
        + values * fast_stride_value,
    ).to(tl.float16)
    slow_encoded = tl.load(
        slow_encoded_read
        + batch * slow_stride_batch
        + values * slow_stride_value,
    ).to(tl.float16)
    decoder = tl.load(
        value_decoder
        + values[:, None] * decoder_stride_value
        + channels[None, :] * decoder_stride_width,
        mask=width_mask[None, :],
        other=0.0,
    ).to(tl.float16)
    fast = tl.sum(
        fast_encoded[:, None].to(tl.float32) * decoder.to(tl.float32),
        axis=0,
    ).to(tl.float16)
    slow = tl.sum(
        slow_encoded[:, None].to(tl.float32) * decoder.to(tl.float32),
        axis=0,
    ).to(tl.float16)
    fast_on = tl.load(
        fast_active + batch * active_stride_batch,
    ).to(tl.float16)
    slow_on = tl.load(
        slow_active + batch * active_stride_batch,
    ).to(tl.float16)
    numerator = _half_multiply(
        parent,
        tl.full((BLOCK_SIZE,), 4.0, tl.float16),
    )
    numerator = _half_add(
        numerator,
        _half_multiply(fast, fast_on),
    )
    numerator = _half_add(
        numerator,
        _half_multiply(slow, slow_on),
    )
    denominator = (4.0 + fast_on + slow_on).to(tl.float16)
    result = (numerator / denominator).to(tl.float16)
    tl.store(parent_read + parent_offsets, result, mask=width_mask)


def triton_semantic_context_codes(
    context: Tensor,
    source: Tensor,
    hash_mask: Tensor,
    *,
    context_decay: float,
    zero_pad_to_power_of_two: bool = False,
) -> Tensor:
    if context.dtype != torch.float16 or source.dtype != torch.float16:
        raise ValueError("Phase264 fused semantic context requires FP16")
    if context.shape != source.shape or context.ndim != 2:
        raise ValueError("Phase264 semantic context/source shape differs")
    width = context.shape[1]
    non_power_of_two = bool(width & (width - 1))
    if zero_pad_to_power_of_two:
        if width <= 0 or width > 1024 or not non_power_of_two:
            raise ValueError(
                "Phase264 padded semantic width must be non-power-of-two at most 1024"
            )
    elif width <= 0 or non_power_of_two or width > 256:
        raise ValueError(
            "Phase264 semantic width must be power-of-two at most 256"
        )
    if hash_mask.shape != (width,) or hash_mask.dtype != torch.int8:
        raise ValueError("Phase264 semantic hash mask differs")
    codes = torch.empty(
        context.shape[0],
        TABLE_COUNT,
        device=context.device,
        dtype=torch.uint8,
    )
    block_size = triton.next_power_of_2(width)
    _semantic_context_code_kernel[(context.shape[0],)](
        context,
        source,
        hash_mask,
        codes,
        CONTEXT_STRIDE_BATCH=context.stride(0),
        CONTEXT_STRIDE_WIDTH=context.stride(1),
        SOURCE_STRIDE_BATCH=source.stride(0),
        SOURCE_STRIDE_WIDTH=source.stride(1),
        CODE_STRIDE_BATCH=codes.stride(0),
        CODE_STRIDE_TABLE=codes.stride(1),
        WIDTH=width,
        BLOCK_SIZE=block_size,
        CONTEXT_DECAY=context_decay,
        num_warps=4,
    )
    return codes


def triton_semantic_memory_step(
    memory: Tensor,
    normalizer: Tensor,
    last_update: Tensor,
    encoded_source: Tensor,
    pending_codes: Tensor,
    query_codes: Tensor,
    *,
    next_token_count: int | Tensor,
    write_enabled: bool,
    fast_decay: float,
    rms_eps: float,
) -> tuple[Tensor, Tensor]:
    expected_rank = TABLE_COUNT * BUCKET_COUNT
    if memory.dtype != torch.float16 or memory.shape[0] != 1:
        raise ValueError("Phase264 semantic memory must be one FP16 layer")
    if memory.ndim != 4 or memory.shape[2:] != (expected_rank, VALUE_WIDTH):
        raise ValueError("Phase264 semantic memory shape differs")
    if normalizer.shape != memory.shape[:3] or normalizer.dtype != torch.float16:
        raise ValueError("Phase264 semantic normalizer differs")
    if last_update.shape != normalizer.shape or last_update.dtype != torch.int32:
        raise ValueError("Phase264 semantic timestamp differs")
    dynamic_token_count = isinstance(next_token_count, Tensor)
    if dynamic_token_count and (
        next_token_count.shape != (1,)
        or next_token_count.dtype != torch.int64
        or next_token_count.device != memory.device
        or not next_token_count.is_contiguous()
    ):
        raise ValueError("Phase389 dynamic semantic token count differs")
    batch = memory.shape[1]
    if encoded_source.shape != (batch, VALUE_WIDTH):
        raise ValueError("Phase264 encoded semantic source differs")
    if encoded_source.dtype != torch.float16:
        raise ValueError("Phase264 encoded semantic source must be FP16")
    if pending_codes.shape != (batch, TABLE_COUNT) or query_codes.shape != (
        batch,
        TABLE_COUNT,
    ):
        raise ValueError("Phase264 semantic code shape differs")
    if pending_codes.dtype != torch.uint8 or query_codes.dtype != torch.uint8:
        raise ValueError("Phase264 semantic codes must be uint8")
    groups = torch.empty(
        batch,
        2,
        VALUE_WIDTH,
        device=memory.device,
        dtype=memory.dtype,
    )
    active = torch.empty(batch, 2, device=memory.device, dtype=torch.uint8)
    _semantic_memory_step_kernel[(batch,)](
        memory[0],
        normalizer[0],
        last_update[0],
        encoded_source,
        pending_codes,
        query_codes,
        groups,
        active,
        memory_stride_batch=memory.stride(1),
        memory_stride_rank=memory.stride(2),
        memory_stride_value=memory.stride(3),
        normalizer_stride_batch=normalizer.stride(1),
        normalizer_stride_rank=normalizer.stride(2),
        last_update_stride_batch=last_update.stride(1),
        last_update_stride_rank=last_update.stride(2),
        source_stride_batch=encoded_source.stride(0),
        source_stride_value=encoded_source.stride(1),
        code_stride_batch=pending_codes.stride(0),
        code_stride_table=pending_codes.stride(1),
        output_stride_batch=groups.stride(0),
        output_stride_group=groups.stride(1),
        output_stride_value=groups.stride(2),
        active_stride_batch=active.stride(0),
        active_stride_group=active.stride(1),
        next_token_count=next_token_count,
        WRITE_ENABLED=write_enabled,
        DYNAMIC_TOKEN_COUNT=dynamic_token_count,
        FAST_DECAY=fast_decay,
        RMS_EPS=rms_eps,
        num_warps=4,
    )
    return groups, active


def triton_compose_semantic_read_in_place(
    parent_read: Tensor,
    fast_encoded_read: Tensor,
    slow_encoded_read: Tensor,
    fast_active: Tensor,
    slow_active: Tensor,
    value_decoder: Tensor,
) -> None:
    if parent_read.dtype != torch.float16 or parent_read.ndim != 2:
        raise ValueError("Phase264 parent read must be rank-two FP16")
    batch, width = parent_read.shape
    if fast_encoded_read.shape != (batch, VALUE_WIDTH) or slow_encoded_read.shape != (
        batch,
        VALUE_WIDTH,
    ):
        raise ValueError("Phase264 encoded semantic read shape differs")
    if (
        fast_encoded_read.dtype != torch.float16
        or slow_encoded_read.dtype != torch.float16
    ):
        raise ValueError("Phase264 encoded semantic reads must be FP16")
    if fast_active.shape != (batch, 1) or slow_active.shape != (batch, 1):
        raise ValueError("Phase264 semantic active shape differs")
    if value_decoder.shape != (VALUE_WIDTH, width):
        raise ValueError("Phase264 semantic value decoder shape differs")
    if value_decoder.dtype != torch.float16:
        raise ValueError("Phase264 semantic value decoder must be FP16")
    block_size = triton.next_power_of_2(width)
    _compose_semantic_read_kernel[(batch,)](
        parent_read,
        fast_encoded_read,
        slow_encoded_read,
        fast_active,
        slow_active,
        value_decoder,
        parent_stride_batch=parent_read.stride(0),
        parent_stride_width=parent_read.stride(1),
        fast_stride_batch=fast_encoded_read.stride(0),
        fast_stride_value=fast_encoded_read.stride(1),
        slow_stride_batch=slow_encoded_read.stride(0),
        slow_stride_value=slow_encoded_read.stride(1),
        active_stride_batch=fast_active.stride(0),
        decoder_stride_value=value_decoder.stride(0),
        decoder_stride_width=value_decoder.stride(1),
        WIDTH=width,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
