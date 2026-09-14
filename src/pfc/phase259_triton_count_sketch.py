from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


BASE_TABLE_COUNT = 6
MEDIAN_TABLE_COUNT = 3
TABLES = tl.constexpr(BASE_TABLE_COUNT)
MEDIAN_TABLES = tl.constexpr(MEDIAN_TABLE_COUNT)
VALUE_WIDTH = tl.constexpr(32)
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
def _fused_count_sketch_kernel(
    memory,
    normalizer,
    last_update,
    encoded_source,
    previous_prior_token_ids,
    prior_token_ids,
    query_token_ids,
    indexed_addresses,
    indexed_signs,
    hash_multipliers,
    hash_offsets,
    sign_multipliers,
    sign_offsets,
    output,
    next_token_count,
    BATCH_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    MEMORY_STRIDE_LAYER: tl.constexpr,
    MEMORY_STRIDE_BATCH: tl.constexpr,
    MEMORY_STRIDE_RANK: tl.constexpr,
    MEMORY_STRIDE_VALUE: tl.constexpr,
    NORMALIZER_STRIDE_LAYER: tl.constexpr,
    NORMALIZER_STRIDE_BATCH: tl.constexpr,
    NORMALIZER_STRIDE_RANK: tl.constexpr,
    LAST_UPDATE_STRIDE_LAYER: tl.constexpr,
    LAST_UPDATE_STRIDE_BATCH: tl.constexpr,
    LAST_UPDATE_STRIDE_RANK: tl.constexpr,
    SOURCE_STRIDE_LAYER: tl.constexpr,
    SOURCE_STRIDE_BATCH: tl.constexpr,
    SOURCE_STRIDE_VALUE: tl.constexpr,
    PREVIOUS_PRIOR_TOKEN_STRIDE: tl.constexpr,
    PRIOR_TOKEN_STRIDE: tl.constexpr,
    QUERY_TOKEN_STRIDE: tl.constexpr,
    ADDRESS_STRIDE_TOKEN: tl.constexpr,
    ADDRESS_STRIDE_TABLE: tl.constexpr,
    SIGN_STRIDE_TOKEN: tl.constexpr,
    SIGN_STRIDE_TABLE: tl.constexpr,
    TABLE_COUNT: tl.constexpr,
    TABLE_BLOCK: tl.constexpr,
    READ_GROUPS: tl.constexpr,
    DUAL_ORDER: tl.constexpr,
    ZERO_ANCHORED_PAIR_CONSENSUS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BUCKET_COUNT: tl.constexpr,
    WRITE_ENABLED: tl.constexpr,
    DYNAMIC_TOKEN_COUNT: tl.constexpr,
    FAST_DECAY: tl.constexpr,
    RMS_EPS: tl.constexpr,
):
    layer_batch = tl.program_id(0)
    batch = layer_batch % BATCH_SIZE
    tables = tl.arange(0, TABLE_BLOCK)
    channels = tl.arange(0, VALUE_WIDTH)
    table_mask = tables < TABLE_COUNT
    matrix_mask = table_mask[:, None]
    current_token_count = (
        tl.load(next_token_count)
        if DYNAMIC_TOKEN_COUNT
        else next_token_count
    )

    base_tables = tables % TABLES
    previous_prior_token = tl.load(
        previous_prior_token_ids + batch * PREVIOUS_PRIOR_TOKEN_STRIDE
    )
    prior_token = tl.load(prior_token_ids + batch * PRIOR_TOKEN_STRIDE)
    query_token = tl.load(query_token_ids + batch * QUERY_TOKEN_STRIDE)
    prior_addresses = tl.load(
        indexed_addresses
        + prior_token * ADDRESS_STRIDE_TOKEN
        + base_tables * ADDRESS_STRIDE_TABLE,
        mask=table_mask,
        other=0,
    )
    query_addresses = tl.load(
        indexed_addresses
        + query_token * ADDRESS_STRIDE_TOKEN
        + base_tables * ADDRESS_STRIDE_TABLE,
        mask=table_mask,
        other=0,
    )
    prior_signs = tl.load(
        indexed_signs
        + prior_token * SIGN_STRIDE_TOKEN
        + base_tables * SIGN_STRIDE_TABLE,
        mask=table_mask,
        other=0.0,
    ).to(tl.float16)
    query_signs = tl.load(
        indexed_signs
        + query_token * SIGN_STRIDE_TOKEN
        + base_tables * SIGN_STRIDE_TABLE,
        mask=table_mask,
        other=0.0,
    ).to(tl.float16)
    if DUAL_ORDER:
        hash_multiplier = tl.load(hash_multipliers + base_tables, mask=table_mask)
        hash_offset = tl.load(hash_offsets + base_tables, mask=table_mask)
        sign_multiplier = tl.load(sign_multipliers + base_tables, mask=table_mask)
        sign_offset = tl.load(sign_offsets + base_tables, mask=table_mask)
        prior_pair_key = previous_prior_token * VOCAB_SIZE + prior_token
        query_pair_key = prior_token * VOCAB_SIZE + query_token
        prior_pair_hash = prior_pair_key * hash_multiplier + hash_offset
        prior_pair_hash = prior_pair_hash ^ (prior_pair_hash >> 16)
        query_pair_hash = query_pair_key * hash_multiplier + hash_offset
        query_pair_hash = query_pair_hash ^ (query_pair_hash >> 16)
        pair_table_offsets = tables * BUCKET_COUNT
        prior_pair_addresses = prior_pair_hash % BUCKET_COUNT + pair_table_offsets
        query_pair_addresses = query_pair_hash % BUCKET_COUNT + pair_table_offsets
        prior_sign_hash = prior_pair_key * sign_multiplier + sign_offset
        prior_sign_hash = prior_sign_hash ^ (prior_sign_hash >> 13)
        query_sign_hash = query_pair_key * sign_multiplier + sign_offset
        query_sign_hash = query_sign_hash ^ (query_sign_hash >> 13)
        prior_pair_signs = tl.where(prior_sign_hash % 2 == 0, 1.0, -1.0)
        query_pair_signs = tl.where(query_sign_hash % 2 == 0, 1.0, -1.0)
        pair_mask = tables >= TABLES
        prior_addresses = tl.where(pair_mask, prior_pair_addresses, prior_addresses)
        query_addresses = tl.where(pair_mask, query_pair_addresses, query_addresses)
        prior_signs = tl.where(pair_mask, prior_pair_signs, prior_signs).to(
            tl.float16
        )
        query_signs = tl.where(pair_mask, query_pair_signs, query_signs).to(
            tl.float16
        )
    decay = tl.where(
        base_tables < MEDIAN_TABLES,
        FAST_DECAY,
        libdevice.sqrt(FAST_DECAY),
    ).to(tl.float16)
    write_scale = _half_subtract(
        tl.full((TABLE_BLOCK,), 1.0, tl.float16),
        decay,
    )

    if WRITE_ENABLED:
        layer = layer_batch // BATCH_SIZE
        prior_normalizer_offsets = (
            layer * NORMALIZER_STRIDE_LAYER
            + batch * NORMALIZER_STRIDE_BATCH
            + prior_addresses * NORMALIZER_STRIDE_RANK
        )
        prior_last_update_offsets = (
            layer * LAST_UPDATE_STRIDE_LAYER
            + batch * LAST_UPDATE_STRIDE_BATCH
            + prior_addresses * LAST_UPDATE_STRIDE_RANK
        )
        elapsed = current_token_count - tl.load(
            last_update + prior_last_update_offsets,
            mask=table_mask,
            other=current_token_count,
        )
        write_decay = libdevice.pow(
            decay.to(tl.float32),
            elapsed.to(tl.float32),
        ).to(tl.float16)
        memory_offsets = (
            layer * MEMORY_STRIDE_LAYER
            + batch * MEMORY_STRIDE_BATCH
            + prior_addresses[:, None] * MEMORY_STRIDE_RANK
            + channels[None, :] * MEMORY_STRIDE_VALUE
        )
        source = tl.load(
            encoded_source
            + layer * SOURCE_STRIDE_LAYER
            + batch * SOURCE_STRIDE_BATCH
            + channels * SOURCE_STRIDE_VALUE,
        ).to(tl.float16)
        old_memory = tl.load(memory + memory_offsets, mask=matrix_mask).to(
            tl.float16
        )
        decayed_memory = _half_multiply(write_decay[:, None], old_memory)
        signed_source = _half_multiply(
            prior_signs[:, None],
            source[None, :],
        )
        write = _half_multiply(signed_source, write_scale[:, None])
        updated_memory = _half_add(decayed_memory, write)
        tl.store(
            memory + memory_offsets,
            updated_memory,
            mask=matrix_mask,
        )
        old_normalizer = tl.load(
            normalizer + prior_normalizer_offsets,
            mask=table_mask,
            other=0.0,
        ).to(tl.float16)
        updated_normalizer = _half_add(
            _half_multiply(write_decay, old_normalizer),
            write_scale,
        )
        tl.store(
            normalizer + prior_normalizer_offsets,
            updated_normalizer,
            mask=table_mask,
        )
        tl.store(
            last_update + prior_last_update_offsets,
            current_token_count,
            mask=table_mask,
        )
        tl.debug_barrier()

    layer = layer_batch // BATCH_SIZE
    query_normalizer_offsets = (
        layer * NORMALIZER_STRIDE_LAYER
        + batch * NORMALIZER_STRIDE_BATCH
        + query_addresses * NORMALIZER_STRIDE_RANK
    )
    query_last_update_offsets = (
        layer * LAST_UPDATE_STRIDE_LAYER
        + batch * LAST_UPDATE_STRIDE_BATCH
        + query_addresses * LAST_UPDATE_STRIDE_RANK
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
        layer * MEMORY_STRIDE_LAYER
        + batch * MEMORY_STRIDE_BATCH
        + query_addresses[:, None] * MEMORY_STRIDE_RANK
        + channels[None, :] * MEMORY_STRIDE_VALUE
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
    denominator = _half_multiply(query_normalizer, query_decay)
    denominator_floor = tl.maximum(
        _half_multiply(
            tl.full((TABLE_BLOCK,), RMS_EPS, tl.float16),
            write_scale,
        ),
        SMALLEST_POSITIVE_HALF,
    ).to(tl.float16)
    denominator = tl.maximum(denominator, denominator_floor).to(tl.float16)
    table_reads = (numerator / denominator[:, None]).to(tl.float16)

    positive_infinity = 1.0e20
    negative_infinity = -1.0e20
    values = table_reads.to(tl.float32)
    output_offsets = layer_batch * (READ_GROUPS * VALUE_WIDTH) + channels
    for group in range(READ_GROUPS):
        group_start = group * MEDIAN_TABLES
        group_mask = (
            (tables >= group_start) & (tables < group_start + MEDIAN_TABLES)
        )[:, None]
        group_sum = tl.sum(tl.where(group_mask, values, 0.0), axis=0)
        group_min = tl.min(
            tl.where(group_mask, values, positive_infinity),
            axis=0,
        )
        group_max = tl.max(
            tl.where(group_mask, values, negative_infinity),
            axis=0,
        )
        if ZERO_ANCHORED_PAIR_CONSENSUS and group >= 2:
            group_min = tl.minimum(group_min, 0.0)
            group_max = tl.maximum(group_max, 0.0)
            group_read = (0.5 * (group_sum - group_min - group_max)).to(
                tl.float16
            )
        else:
            group_read = (group_sum - group_min - group_max).to(tl.float16)
        tl.store(output + output_offsets + group * VALUE_WIDTH, group_read)


def triton_count_sketch_step(
    memory: Tensor,
    normalizer: Tensor,
    last_update: Tensor,
    encoded_source: Tensor,
    prior_token_ids: Tensor,
    query_token_ids: Tensor,
    indexed_addresses: Tensor,
    indexed_signs: Tensor,
    previous_prior_token_ids: Tensor | None = None,
    hash_multipliers: Tensor | None = None,
    hash_offsets: Tensor | None = None,
    sign_multipliers: Tensor | None = None,
    sign_offsets: Tensor | None = None,
    vocab_size: int | None = None,
    *,
    next_token_count: int | Tensor,
    write_enabled: bool,
    fast_decay: float,
    rms_eps: float,
    zero_anchored_pair_consensus: bool = False,
) -> Tensor:
    if memory.dtype != torch.float16:
        raise ValueError("Phase259 memory must be FP16")
    if memory.ndim != 4 or memory.shape[-1] != VALUE_WIDTH:
        raise ValueError("Phase259 memory must have value width 32")
    if normalizer.shape != memory.shape[:3] or normalizer.dtype != torch.float16:
        raise ValueError("Phase259 normalizer shape or dtype differs")
    if last_update.shape != normalizer.shape or last_update.dtype != torch.int32:
        raise ValueError("Phase259 timestamp shape or dtype differs")
    dynamic_token_count = isinstance(next_token_count, Tensor)
    if dynamic_token_count and (
        next_token_count.shape != (1,)
        or next_token_count.dtype != torch.int64
        or next_token_count.device != memory.device
        or not next_token_count.is_contiguous()
    ):
        raise ValueError("Phase389 dynamic CountSketch token count differs")
    if encoded_source.shape != memory.shape[:2] + (VALUE_WIDTH,):
        raise ValueError("Phase259 encoded source shape differs")
    dual_order = previous_prior_token_ids is not None
    if (
        prior_token_ids.shape != (memory.shape[1],)
        or query_token_ids.shape != (memory.shape[1],)
    ):
        raise ValueError("Phase259 token shape differs")
    if dual_order and previous_prior_token_ids.shape != (memory.shape[1],):
        raise ValueError("Phase262 previous-prior token shape differs")
    if zero_anchored_pair_consensus and not dual_order:
        raise ValueError("zero-anchored pair consensus requires dual-order memory")
    if indexed_addresses.shape != indexed_signs.shape:
        raise ValueError("Phase259 address/sign shape differs")
    if indexed_addresses.shape[1] != BASE_TABLE_COUNT:
        raise ValueError("Phase259 requires six address tables")
    if indexed_addresses.dtype != torch.int64:
        raise ValueError("Phase259 addresses must be int64")
    if indexed_signs.dtype != torch.float16:
        raise ValueError("Phase259 signs must be FP16")
    hash_tensors = (
        hash_multipliers,
        hash_offsets,
        sign_multipliers,
        sign_offsets,
    )
    if dual_order:
        if any(value is None for value in hash_tensors):
            raise ValueError("Phase262 dual-order hash constants are missing")
        if any(
            value.shape != (BASE_TABLE_COUNT,)
            for value in hash_tensors
            if value is not None
        ):
            raise ValueError("Phase262 dual-order hash constant shape differs")
        if any(value.dtype != torch.int64 for value in hash_tensors if value is not None):
            raise ValueError("Phase262 dual-order hash constants must be int64")
        if vocab_size is None or vocab_size <= 1:
            raise ValueError("Phase262 dual-order vocabulary size is invalid")
    table_count = (
        2 * BASE_TABLE_COUNT if dual_order else BASE_TABLE_COUNT
    )
    read_groups = table_count // MEDIAN_TABLE_COUNT
    table_block = 16 if dual_order else 8
    if memory.shape[2] % table_count:
        raise ValueError("CountSketch rank must divide evenly across tables")
    previous_prior_token_ids = (
        previous_prior_token_ids if dual_order else prior_token_ids
    )
    hash_multipliers = hash_multipliers if dual_order else indexed_addresses
    hash_offsets = hash_offsets if dual_order else indexed_addresses
    sign_multipliers = sign_multipliers if dual_order else indexed_addresses
    sign_offsets = sign_offsets if dual_order else indexed_addresses
    output = torch.empty(
        memory.shape[0],
        memory.shape[1],
        read_groups * VALUE_WIDTH,
        device=memory.device,
        dtype=memory.dtype,
    )
    _fused_count_sketch_kernel[(memory.shape[0] * memory.shape[1],)](
        memory,
        normalizer,
        last_update,
        encoded_source,
        previous_prior_token_ids,
        prior_token_ids,
        query_token_ids,
        indexed_addresses,
        indexed_signs,
        hash_multipliers,
        hash_offsets,
        sign_multipliers,
        sign_offsets,
        output,
        next_token_count,
        BATCH_SIZE=memory.shape[1],
        RANK=memory.shape[2],
        MEMORY_STRIDE_LAYER=memory.stride(0),
        MEMORY_STRIDE_BATCH=memory.stride(1),
        MEMORY_STRIDE_RANK=memory.stride(2),
        MEMORY_STRIDE_VALUE=memory.stride(3),
        NORMALIZER_STRIDE_LAYER=normalizer.stride(0),
        NORMALIZER_STRIDE_BATCH=normalizer.stride(1),
        NORMALIZER_STRIDE_RANK=normalizer.stride(2),
        LAST_UPDATE_STRIDE_LAYER=last_update.stride(0),
        LAST_UPDATE_STRIDE_BATCH=last_update.stride(1),
        LAST_UPDATE_STRIDE_RANK=last_update.stride(2),
        SOURCE_STRIDE_LAYER=encoded_source.stride(0),
        SOURCE_STRIDE_BATCH=encoded_source.stride(1),
        SOURCE_STRIDE_VALUE=encoded_source.stride(2),
        PREVIOUS_PRIOR_TOKEN_STRIDE=previous_prior_token_ids.stride(0),
        PRIOR_TOKEN_STRIDE=prior_token_ids.stride(0),
        QUERY_TOKEN_STRIDE=query_token_ids.stride(0),
        ADDRESS_STRIDE_TOKEN=indexed_addresses.stride(0),
        ADDRESS_STRIDE_TABLE=indexed_addresses.stride(1),
        SIGN_STRIDE_TOKEN=indexed_signs.stride(0),
        SIGN_STRIDE_TABLE=indexed_signs.stride(1),
        TABLE_COUNT=table_count,
        TABLE_BLOCK=table_block,
        READ_GROUPS=read_groups,
        DUAL_ORDER=dual_order,
        ZERO_ANCHORED_PAIR_CONSENSUS=zero_anchored_pair_consensus,
        VOCAB_SIZE=vocab_size or 1,
        BUCKET_COUNT=memory.shape[2] // table_count,
        WRITE_ENABLED=write_enabled,
        DYNAMIC_TOKEN_COUNT=dynamic_token_count,
        FAST_DECAY=fast_decay,
        RMS_EPS=rms_eps,
        num_warps=4,
    )
    return output
