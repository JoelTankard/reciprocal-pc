from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, TYPE_CHECKING

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pfc.phase259_fused_count_sketch import (
    fused_dual_order_count_sketch_step,
    fused_indexed_count_sketch_step,
)
from pfc.phase264_fused_semantic_backoff import (
    fused_compose_semantic_read_in_place,
    fused_semantic_context_codes,
    fused_semantic_memory_step,
)
from pfc.phase265_fused_pair_bloom import fused_pair_recurrence_bloom_step
from pfc.phase270_fused_disjoint_semantic_channels import (
    fused_compose_disjoint_semantic_channels_in_place,
)
from pfc.phase342_nonlinear_matrix_state import (
    NONLINEAR_MATRIX_HEADS,
    NONLINEAR_MATRIX_KEY_WIDTH,
    NONLINEAR_MATRIX_MODEL_WIDTH,
    NONLINEAR_MATRIX_PROJECTION_WIDTH,
    NONLINEAR_MATRIX_STATE_VALUES,
    NONLINEAR_MATRIX_VALUE_WIDTH,
)
from pfc.phase606_temporal_hgf_precision import (
    temporal_hgf_bottom_up,
    temporal_hgf_precision_sequence,
)


SELECTIVE_TEMPORAL_PHASE_LIMIT = math.pi / 4.0
MULTISCALE_COUNT_SKETCH_TABLES = 6
MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES = 3
MULTISCALE_COUNT_SKETCH_HASH_MULTIPLIERS = (
    2_654_435_761,
    2_246_822_519,
    3_266_489_917,
    668_265_263,
    374_761_393,
    1_274_126_177,
)
MULTISCALE_COUNT_SKETCH_HASH_OFFSETS = (
    1_013_904_223,
    1_654_911_287,
    2_045_814_391,
    739_632_901,
    1_931_847_217,
    281_243_693,
)
MULTISCALE_COUNT_SKETCH_SIGN_MULTIPLIERS = (
    1_591_733_291,
    1_107_351_759,
    2_973_764_127,
    855_927_361,
    1_378_736_443,
    2_147_483_629,
)
MULTISCALE_COUNT_SKETCH_SIGN_OFFSETS = (
    911_382_323,
    1_729_473_811,
    467_889_221,
    2_033_948_019,
    1_211_431_047,
    653_287_291,
)
DUAL_ORDER_COUNT_SKETCH_ADDRESSING = (
    "dual_order_indexed_lazy_multiscale_count_sketch"
)
ZERO_ANCHORED_PAIR_CONSENSUS_ADDRESSING = (
    "zero_anchored_pair_consensus_dual_order_indexed_lazy_multiscale_count_sketch"
)
CAUSAL_SEMANTIC_BACKOFF_COUNT_SKETCH_ADDRESSING = (
    "causal_semantic_backoff_dual_order_indexed_lazy_multiscale_count_sketch"
)
PAIR_MISS_SEMANTIC_BACKOFF_COUNT_SKETCH_ADDRESSING = (
    "pair_miss_semantic_backoff_dual_order_indexed_lazy_multiscale_count_sketch"
)
PAIR_MISS_DISJOINT_TIMESCALE_SEMANTIC_CHANNELS_ADDRESSING = (
    "pair_miss_disjoint_timescale_semantic_channels_dual_order_indexed_lazy_"
    "multiscale_count_sketch"
)
SEMANTIC_COUNT_SKETCH_TABLES = 6
SEMANTIC_COUNT_SKETCH_BUCKETS = 64
SEMANTIC_COUNT_SKETCH_VALUE_WIDTH = 8
SEMANTIC_COUNT_SKETCH_HASH_BITS = 6
PAIR_RECURRENCE_BLOOM_GENERATIONS = 2
PAIR_RECURRENCE_BLOOM_PROBES = 4
PAIR_RECURRENCE_BLOOM_BITS = 8192
PAIR_RECURRENCE_BLOOM_BYTES = PAIR_RECURRENCE_BLOOM_BITS // 8
PAIR_RECURRENCE_BLOOM_GENERATION_SPAN = 2048
PAIR_RECURRENCE_BLOOM_HASH_MULTIPLIERS = (
    2_246_822_519,
    3_266_489_917,
    668_265_263,
    1_274_126_177,
)
PAIR_RECURRENCE_BLOOM_HASH_OFFSETS = (
    1_654_911_287,
    2_045_814_391,
    739_632_901,
    281_243_693,
)
DEPTH_STRATIFIED_TEMPORAL_GATE_BIASES = (
    1.2586915494460322,
    2.501908776536572,
    3.6843577592399015,
    4.8481214707894,
)


def _uses_fp16_cuda_fused_inference(value: Tensor) -> bool:
    return bool(
        value.is_cuda
        and value.dtype == torch.float16
        and not torch.is_grad_enabled()
    )


DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS = frozenset(
    {
        DUAL_ORDER_COUNT_SKETCH_ADDRESSING,
        ZERO_ANCHORED_PAIR_CONSENSUS_ADDRESSING,
        CAUSAL_SEMANTIC_BACKOFF_COUNT_SKETCH_ADDRESSING,
        PAIR_MISS_SEMANTIC_BACKOFF_COUNT_SKETCH_ADDRESSING,
        PAIR_MISS_DISJOINT_TIMESCALE_SEMANTIC_CHANNELS_ADDRESSING,
    }
)
MULTISCALE_COUNT_SKETCH_ADDRESSINGS = frozenset(
    {
        "multiscale_count_sketch",
        "lazy_multiscale_count_sketch",
        "indexed_lazy_multiscale_count_sketch",
        *DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS,
    }
)
INDEXED_COUNT_SKETCH_ADDRESSINGS = frozenset(
    {
        "indexed_lazy_multiscale_count_sketch",
        *DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS,
    }
)
LAZY_COUNT_SKETCH_ADDRESSINGS = frozenset(
    {
        "lazy_multiscale_count_sketch",
        *INDEXED_COUNT_SKETCH_ADDRESSINGS,
    }
)


def _rms_norm(value: Tensor, weight: Tensor, eps: float) -> Tensor:
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps) * weight


_semantic_hadamard_profile_stage_hook: Callable[[str, bool], None] | None = None


def set_semantic_hadamard_profile_stage_hook(
    hook: Callable[[str, bool], None] | None,
) -> None:
    global _semantic_hadamard_profile_stage_hook
    _semantic_hadamard_profile_stage_hook = hook


def _mark_semantic_hadamard_profile_stage(name: str, opening: bool) -> None:
    hook = _semantic_hadamard_profile_stage_hook
    if hook is not None:
        hook(name, opening)


def _normalized_hadamard(value: Tensor) -> Tensor:
    """Apply an orthonormal Sylvester Hadamard transform on the last axis."""

    width = value.shape[-1]
    if width <= 0 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    transformed = value
    block = 1
    while block < width:
        grouped = transformed.reshape(*value.shape[:-1], -1, 2, block)
        first, second = grouped.unbind(dim=-2)
        transformed = torch.cat((first + second, first - second), dim=-1).reshape(
            value.shape
        )
        block *= 2
    return transformed * (1.0 / math.sqrt(width))


def _bent_hadamard_sign_schedule(depth: int, width: int) -> Tensor:
    """Build cumulative signs with flat-spectrum adjacent differences."""

    if depth <= 0:
        raise ValueError("bent Hadamard schedule requires positive depth")
    if width <= 0 or width & (width - 1):
        raise ValueError("bent Hadamard schedule requires power-of-two width")
    bit_count = width.bit_length() - 1
    if bit_count == 0 or bit_count % 2:
        raise ValueError("bent Hadamard schedule requires an even positive log2 width")

    channels = torch.arange(width, dtype=torch.int64)
    bits = ((channels[:, None] >> torch.arange(bit_count)) & 1).bool()
    cumulative = torch.ones(width, dtype=torch.float32)
    signs = [cumulative.clone()]
    rotating_bits = list(range(1, bit_count))
    for transition in range(depth - 1):
        if transition and transition % (bit_count - 1) == 0:
            rotating_bits = list(range(1, bit_count))
        permutation = [0, *rotating_bits]
        quadratic = torch.zeros(width, dtype=torch.bool)
        for index in range(bit_count // 2):
            quadratic.logical_xor_(
                bits[:, permutation[index]]
                & bits[:, permutation[bit_count - 1 - index]]
            )
        cumulative = cumulative * torch.where(quadratic, -1.0, 1.0)
        signs.append(cumulative.clone())
        rotating_bits = [rotating_bits[-1], *rotating_bits[:-1]]
    return torch.stack(signs)


def _fixed_orthogonal_depth_readout(states: Tensor, signs: Tensor) -> Tensor:
    """Combine layer states through fixed, distinct orthogonal bases."""

    if states.ndim < 2:
        raise ValueError("depth readout states require layer and feature axes")
    depth, width = states.shape[0], states.shape[-1]
    if signs.shape != (depth, width):
        raise ValueError("depth readout signs must have [depth,width]")
    sign_shape = (depth,) + (1,) * (states.ndim - 2) + (width,)
    return _normalized_hadamard(
        states * signs.view(sign_shape).to(states.dtype)
    ).sum(dim=0)


def _fixed_orthogonal_depth_readout_inverse(
    direction: Tensor,
    signs: Tensor,
) -> Tensor:
    """Map one readout direction into every fixed layer basis."""

    if direction.ndim < 1:
        raise ValueError("depth readout direction requires a feature axis")
    depth, width = signs.shape
    if direction.shape[-1] != width:
        raise ValueError("depth readout direction width differs")
    expanded = direction.unsqueeze(0).expand(depth, *direction.shape)
    sign_shape = (depth,) + (1,) * (direction.ndim - 1) + (width,)
    return _normalized_hadamard(expanded) * signs.view(sign_shape).to(
        direction.dtype
    )


def _affine_scan(decay: Tensor, write: Tensor, initial: Tensor) -> Tensor:
    """Parallel prefix scan for h[t] = decay[t] * h[t-1] + write[t]."""

    if decay.shape != write.shape or decay.ndim != 3:
        raise ValueError("decay and write must share shape [batch, sequence, width]")
    if initial.shape != (decay.shape[0], decay.shape[2]):
        raise ValueError("initial state must have shape [batch, width]")
    prefix_decay = decay
    prefix_write = write
    offset = 1
    while offset < decay.shape[1]:
        right_decay = prefix_decay[:, offset:]
        composed_decay = right_decay * prefix_decay[:, :-offset]
        composed_write = (
            prefix_write[:, offset:] + right_decay * prefix_write[:, :-offset]
        )
        prefix_decay = torch.cat((prefix_decay[:, :offset], composed_decay), dim=1)
        prefix_write = torch.cat((prefix_write[:, :offset], composed_write), dim=1)
        offset *= 2
    return prefix_decay * initial.unsqueeze(1) + prefix_write


def _constant_decay_scan(
    write: Tensor,
    initial: Tensor,
    decay: float | Tensor,
    *,
    fused: bool = False,
    reset_mask: Tensor | None = None,
) -> Tensor:
    """Parallel scan for state[t] = decay * state[t-1] + write[t]."""

    if write.ndim < 3:
        raise ValueError("write must have shape [batch, sequence, ...]")
    if initial.shape != write.shape[:1] + write.shape[2:]:
        raise ValueError("initial must match write without sequence dimension")
    if isinstance(decay, Tensor):
        if (
            decay.ndim != 0
            or decay.device != write.device
            or decay.dtype != torch.float64
        ):
            raise ValueError(
                "tensor decay must be a float64 scalar on the write device"
            )
        if fused:
            raise ValueError("fused scan requires a Python decay scalar")
        decay_tensor = decay.to(dtype=write.dtype)
    else:
        if not 0 <= decay < 1:
            raise ValueError("decay must be in [0, 1)")
        decay_tensor = torch.full(
            (),
            decay,
            device=write.device,
            dtype=write.dtype,
        )
    if reset_mask is not None:
        if reset_mask.shape != write.shape[:2]:
            raise ValueError("reset mask must have [batch, sequence] shape")
        segment_ids = reset_mask.to(torch.long).cumsum(dim=1)
        prefix = write
        offset = 1
        decay_power: float | Tensor = decay
        while offset < write.shape[1]:
            same_segment = segment_ids[:, offset:].eq(
                segment_ids[:, :-offset]
            )
            same_segment = same_segment.reshape(
                same_segment.shape + (1,) * (write.ndim - 2)
            )
            composed = (
                prefix[:, offset:]
                + decay_power * prefix[:, :-offset]
            )
            prefix = torch.cat(
                (
                    prefix[:, :offset],
                    torch.where(
                        same_segment,
                        composed,
                        prefix[:, offset:],
                    ),
                ),
                dim=1,
            )
            offset *= 2
            decay_power = decay_power * decay_power
        initial_power = torch.pow(
            decay_tensor,
            torch.arange(
                1,
                write.shape[1] + 1,
                device=write.device,
                dtype=write.dtype,
            ),
        )
        power_shape = (1, write.shape[1]) + (1,) * (write.ndim - 2)
        initial_contribution = initial_power.view(power_shape) * initial.unsqueeze(1)
        before_first_reset = segment_ids.eq(0).reshape(
            segment_ids.shape + (1,) * (write.ndim - 2)
        )
        return prefix + torch.where(
            before_first_reset,
            initial_contribution,
            torch.zeros_like(initial_contribution),
        )
    if (
        fused
        and write.is_cuda
        and write.dtype == torch.float32
        and write.is_contiguous()
        and initial.is_contiguous()
        and not torch.is_grad_enabled()
    ):
        from pfc.phase345_triton_constant_decay_scan import (
            triton_constant_decay_prefix,
        )

        prefix = triton_constant_decay_prefix(write, decay)
    elif isinstance(decay, Tensor):
        prefix = write
        offset = 1
        decay_power = decay
        while offset < write.shape[1]:
            composed = prefix[:, offset:] + decay_power * prefix[:, :-offset]
            prefix = torch.cat((prefix[:, :offset], composed), dim=1)
            offset *= 2
            decay_power = decay_power * decay_power
    else:
        prefix = write
        offset = 1
        decay_power = decay
        while offset < write.shape[1]:
            composed = prefix[:, offset:] + decay_power * prefix[:, :-offset]
            prefix = torch.cat((prefix[:, :offset], composed), dim=1)
            offset *= 2
            decay_power *= decay_power
    initial_power = torch.pow(
        decay_tensor,
        torch.arange(
            1,
            write.shape[1] + 1,
            device=write.device,
            dtype=write.dtype,
        ),
    )
    power_shape = (1, write.shape[1]) + (1,) * (write.ndim - 2)
    return prefix + initial_power.view(power_shape) * initial.unsqueeze(1)


def _matrix_memory_affine_scan(
    matrix: Tensor,
    write: Tensor,
    initial: Tensor,
) -> Tensor:
    """Parallel scan for S[t] = A[t] S[t-1] + B[t]."""

    if matrix.ndim != 5 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must have shape [batch, sequence, head, key, key]")
    if write.ndim != 5 or write.shape[:-1] != matrix.shape[:-1]:
        raise ValueError("write must have shape [batch, sequence, head, key, value]")
    if initial.shape != write.shape[:1] + write.shape[2:]:
        raise ValueError("initial must match write without sequence dimension")
    prefix_matrix = matrix
    prefix_write = write
    offset = 1
    while offset < matrix.shape[1]:
        earlier_matrix = prefix_matrix[:, :-offset]
        later_matrix = prefix_matrix[:, offset:]
        composed_matrix = torch.matmul(later_matrix, earlier_matrix)
        composed_write = (
            torch.matmul(later_matrix, prefix_write[:, :-offset])
            + prefix_write[:, offset:]
        )
        prefix_matrix = torch.cat(
            (prefix_matrix[:, :offset], composed_matrix),
            dim=1,
        )
        prefix_write = torch.cat(
            (prefix_write[:, :offset], composed_write),
            dim=1,
        )
        offset *= 2
    return torch.matmul(prefix_matrix, initial.unsqueeze(1)) + prefix_write


def _pair_recurrence_matrix(decay: Tensor, rotation: Tensor) -> Tensor:
    """Build row-vector matrices for decay * rotate(previous) per pair."""

    if decay.shape[-1] != 2 or rotation.shape[-1] != 2:
        raise ValueError("paired decay and rotation must end in two channels")
    cosine = rotation[..., 0]
    sine = rotation[..., 1]
    decay_x = decay[..., 0]
    decay_y = decay[..., 1]
    return torch.stack(
        (
            cosine * decay_x,
            sine * decay_y,
            -sine * decay_x,
            cosine * decay_y,
        ),
        dim=-1,
    ).reshape(*decay.shape[:-1], 2, 2)


def _pair_matrix_multiply(left: Tensor, right: Tensor) -> Tensor:
    """Explicit 2x2 product. Fuses better than tiny batched GEMMs."""

    if left.shape != right.shape or left.shape[-2:] != (2, 2):
        raise ValueError("paired matrices must share shape ending in 2x2")
    left_00 = left[..., 0, 0]
    left_01 = left[..., 0, 1]
    left_10 = left[..., 1, 0]
    left_11 = left[..., 1, 1]
    right_00 = right[..., 0, 0]
    right_01 = right[..., 0, 1]
    right_10 = right[..., 1, 0]
    right_11 = right[..., 1, 1]
    return torch.stack(
        (
            left_00 * right_00 + left_01 * right_10,
            left_00 * right_01 + left_01 * right_11,
            left_10 * right_00 + left_11 * right_10,
            left_10 * right_01 + left_11 * right_11,
        ),
        dim=-1,
    ).reshape_as(left)


def _pair_vector_matrix_multiply(vector: Tensor, matrix: Tensor) -> Tensor:
    """Explicit row-vector by 2x2 matrix product."""

    if vector.shape != matrix.shape[:-1] or vector.shape[-1] != 2:
        raise ValueError("paired vector and matrix shapes do not align")
    return torch.stack(
        (
            vector[..., 0] * matrix[..., 0, 0] + vector[..., 1] * matrix[..., 1, 0],
            vector[..., 0] * matrix[..., 0, 1] + vector[..., 1] * matrix[..., 1, 1],
        ),
        dim=-1,
    )


def _pair_affine_scan(
    matrix: Tensor,
    write: Tensor,
    initial: Tensor,
) -> Tensor:
    """Parallel scan for paired row states h[t] = h[t-1] A[t] + b[t]."""

    if matrix.ndim != 5 or matrix.shape[-2:] != (2, 2):
        raise ValueError("matrix must have shape [group, sequence, pair, 2, 2]")
    if write.shape != matrix.shape[:-1]:
        raise ValueError("write must have shape [group, sequence, pair, 2]")
    if initial.shape != (matrix.shape[0], matrix.shape[2], 2):
        raise ValueError("initial must have shape [group, pair, 2]")
    prefix_matrix = matrix
    prefix_write = write
    offset = 1
    while offset < matrix.shape[1]:
        earlier_matrix = prefix_matrix[:, :-offset]
        later_matrix = prefix_matrix[:, offset:]
        composed_matrix = _pair_matrix_multiply(
            earlier_matrix,
            later_matrix,
        )
        composed_write = (
            _pair_vector_matrix_multiply(
                prefix_write[:, :-offset],
                later_matrix,
            )
            + prefix_write[:, offset:]
        )
        prefix_matrix = torch.cat(
            (prefix_matrix[:, :offset], composed_matrix),
            dim=1,
        )
        prefix_write = torch.cat(
            (prefix_write[:, :offset], composed_write),
            dim=1,
        )
        offset *= 2
    return (
        _pair_vector_matrix_multiply(
            initial[:, None].expand_as(prefix_write),
            prefix_matrix,
        )
        + prefix_write
    )


def _group4_recurrence_matrix(decay: Tensor, rotation: Tensor) -> Tensor:
    """Build 4x4 block-diagonal pair recurrences for grouped delta mixing."""

    if decay.shape[-1] != 4 or rotation.shape[-2:] != (2, 2):
        raise ValueError("grouped decay and rotation must end in 4 and 2x2")
    pair_matrix = _pair_recurrence_matrix(
        decay.reshape(*decay.shape[:-1], 2, 2),
        rotation,
    )
    zeros = torch.zeros_like(pair_matrix[..., 0, :, :])
    upper = torch.cat((pair_matrix[..., 0, :, :], zeros), dim=-1)
    lower = torch.cat((zeros, pair_matrix[..., 1, :, :]), dim=-1)
    return torch.cat((upper, lower), dim=-2)


def _group4_affine_scan(
    matrix: Tensor,
    write: Tensor,
    initial: Tensor,
) -> Tensor:
    """Parallel scan for grouped row states h[t] = h[t-1] A[t] + b[t]."""

    if matrix.ndim != 5 or matrix.shape[-2:] != (4, 4):
        raise ValueError("matrix must have shape [group, sequence, block, 4, 4]")
    if write.shape != matrix.shape[:-1]:
        raise ValueError("write must have shape [group, sequence, block, 4]")
    if initial.shape != (matrix.shape[0], matrix.shape[2], 4):
        raise ValueError("initial must have shape [group, block, 4]")
    prefix_matrix = matrix
    prefix_write = write
    offset = 1
    while offset < matrix.shape[1]:
        earlier_matrix = prefix_matrix[:, :-offset]
        later_matrix = prefix_matrix[:, offset:]
        composed_matrix = torch.matmul(earlier_matrix, later_matrix)
        composed_write = (
            torch.matmul(
                prefix_write[:, :-offset].unsqueeze(-2),
                later_matrix,
            ).squeeze(-2)
            + prefix_write[:, offset:]
        )
        prefix_matrix = torch.cat(
            (prefix_matrix[:, :offset], composed_matrix),
            dim=1,
        )
        prefix_write = torch.cat(
            (prefix_write[:, :offset], composed_write),
            dim=1,
        )
        offset *= 2
    return (
        torch.matmul(
            initial[:, None].expand_as(prefix_write).unsqueeze(-2),
            prefix_matrix,
        ).squeeze(-2)
        + prefix_write
    )


@dataclass(frozen=True)
class NativePCConfig:
    """WebGPU-oriented recurrent predictive-coding language-model config."""

    vocab_size: int = 4096
    bos_token_id: int = 0
    width: int = 256
    depth: int = 4
    ffn_width: int = 1888
    settling_rounds: int = 1
    shared_embedding_decoder: bool = False
    metric_consistent_shared_endpoint: bool = False
    row_energy_preserving_input_compander: bool = False
    energy_preserving_token_fingerprint_observation: bool = False
    self_dual_hadamard_token_observation_coupling: bool = False
    energy_preserving_self_bilinear_token_observation: bool = False
    projected_self_bilinear_token_observation: bool = False
    self_gated_isometric_sign_token_observation: bool = False
    kronecker_input_embedding: bool = False
    kronecker_input_codec_sha256: str = ""
    bpe_ancestry_input_embedding: bool = False
    bpe_ancestry_input_artifact_sha256: str = ""
    bpe_ancestry_fused_execution: bool = False
    split_ternary_lexical_endpoint: bool = False
    split_ternary_lexical_dense_width: int = 0
    state_step: float = 0.5
    top_down_scale: float = 0.25
    loop_aware_settling_scaling: bool = False
    deeploop_postnorm_settling: bool = False
    exact_diagonal_hgf_posterior_sweep: bool = False
    predicted_precision_hgf_posterior_sweep: bool = False
    temporal_hgf_precision_carry: bool = False
    bf16_training_fp32_recurrent_statistics: bool = False
    intrinsic_ternary_reciprocal_ffn: bool = False
    lifted_antisymmetric_reciprocal_ffn: bool = False
    context_bound_ternary_parameter_superposition: bool = False
    fused_context_bound_ternary_decode: bool = False
    ternary_tpc_rtrl_elementwise_transition: bool = False
    ordered_swiglu_rglru_carry: bool = False
    lexical_coordinate_transition_carrier: bool = False
    post_coinitialized_lexical_coordinate_transition_carrier: bool = False
    fixed_identity_lexical_coordinate_transition_core: bool = False
    causal_level_innovation_lifting: bool = False
    signed_shunting_ternary_transition: bool = False
    token_gated_diagonal_handoff: bool = False
    causal_ternary_fir_input_state: bool = False
    causal_ternary_fir_independent_clip: bool = False
    causal_ternary_fir_fp32_deployment_scales: bool = False
    lagged_latent_error_settling_carrier: bool = False
    mean_free_learned_recurrent_write: bool = False
    ternary_dale_quadrant_transition: bool = False
    sign_lifted_dual_ternary_transition: bool = False
    source_separated_transition_compartments: bool = False
    active_interleaved_source_ownership: bool = False
    contractive_ternary_reciprocal_butterfly_ffn: bool = False
    parallel_full_state_reciprocal_proposal: bool = False
    fused_parallel_reciprocal_butterfly_decode: bool = False
    bos_boundary_state_reset: bool = False
    bos_boundary_exact_segmented_training: bool = False
    depth_stratified_temporal_hierarchy: bool = False
    deterministic_ternary_ffn_experts: bool = False
    shared_features_routed_ternary_ffn_output: bool = False
    depth_wide_ternary_lexical_endpoint: bool = False
    feature_ready_depth_wide_lexical_residual: bool = False
    cozero_reciprocal_lexical_adapters: bool = False
    swiglu_wide_lexical_cohead: bool = False
    context_routed_ternary_lexical_residual: bool = False
    bounded_multiplicative_lexical_modulator: bool = False
    signed_square_ternary_lexical_lift: bool = False
    cross_channel_quadratic_ternary_lexical_lift: bool = False
    single_endpoint_depth_innovation_predictor: bool = False
    energy_conserving_lexical_group_contrast: bool = False
    causal_anti_hebbian_lexical_butterfly: bool = False
    fused_causal_anti_hebbian_lexical_butterfly_execution: bool = False
    shared_base_ternary_delta_mixture: bool = False
    pairwise_competitive_swiglu: bool = False
    bilinear_hebbian_ffn: bool = False
    deterministic_ternary_ffn_residual_experts: bool = False
    selected_rank_one_ternary_residuals: bool = False
    selected_diagonal_ternary_residuals: bool = False
    selected_diagonal_swiglu_residuals: bool = False
    routed_ternary_ffn_channel_allocation: bool = False
    zero_mean_contrastive_ffn_channel_allocation: bool = False
    self_gated_ternary_hidden_allocation: bool = False
    selected_diagonal_decode_gather: bool = False
    selected_only_residual_expert_decode: bool = False
    selected_only_residual_expert_sequence: bool = False
    cublas_routed_residual_expert_prefill: bool = False
    mandatory_full_bandwidth_bottom_clamp: bool = False
    mandatory_ternary_state_carrier: bool = False
    residual_deep_state_recirculation: bool = False
    fused_residual_recirculation_decode: bool = False
    ternary_vector_recirculation_affordance: bool = False
    causal_diagonal_adaptive_recirculation: bool = False
    visit_framed_settling: bool = False
    untied_settling_visits: bool = False
    visit_residual_adapters: bool = False
    visit_low_rank_adapter_rank: int = 0
    nonlinear_visit_low_rank_adapter: bool = False
    dual_stream_visit_adapter_rank: int = 0
    persistent_transition_adapter_rank: int = 0
    selective_temporal_phase_rank: int = 0
    decoupled_transition_write_gate: bool = False
    pairwise_delta_erase_rank: int = 0
    pairwise_delta_group_width: int = 2
    reversible_recurrent_coupling: bool = False
    affine_reversible_transition: bool = False
    depth_tied_reciprocal_transition_core: bool = False
    balanced_two_of_four_reciprocal_transition: bool = False
    softsign_reciprocal_transition: bool = False
    alternating_triangular_coupling: bool = False
    contractive_reversible_retention: bool = False
    representably_contractive_retention: bool = False
    pair_norm_bounded_convex_recurrence: bool = False
    hippo_legs_order2_temporal_coordinates: bool = False
    quadrature_preserving_recurrent_state: bool = False
    local_quadrature_write_credit: bool = False
    bounded_manifold_depth_bus: bool = False
    fused_manifold_depth_bus_inference: bool = False
    bounded_depth_attention_residual: bool = False
    stable_latent_expert_branch: bool = False
    fused_constant_decay_scan: bool = False
    nonlinear_matrix_state_lane: bool = False
    fused_nonlinear_matrix_decode: bool = False
    decoupled_bounded_memory_emission: bool = False
    coefficient_normalized_emission: bool = False
    compact_active_triangular_coupling: bool = False
    butterfly_channel_diffusion: bool = False
    layer_rotating_bent_butterfly_diffusion: bool = False
    residual_stream: bool = False
    residual_stream_scale_start: float = 1.15
    residual_stream_scale_end: float = 1.05
    sensory_reinjection_start: float = 0.20
    sensory_reinjection_end: float = 0.05
    learn_residual_stream_scales: bool = False
    parallel_readout: bool = False
    parallel_readout_init: str = "bottom"
    projected_readout: bool = False
    hybrid_projected_readout: bool = False
    fixed_orthogonal_depth_readout: bool = False
    fixed_ternary_hadamard_lexical_predictor: bool = False
    fixed_hadamard_top8_lexical_reranker: bool = False
    shared_nonlinear_candidate_compatibility: bool = False
    state_conditioned_confidence_readout: bool = False
    predictive_future_error_key_rank: int = 0
    token_conditioned_predictive_read_gate: bool = False
    bounded_surprise_exact_cache: bool = False
    horizon_isolated_predictive_rls: bool = False
    stability_limited_query_aware_predictive_rls: bool = False
    self_protecting_predictive_rls: bool = False
    continuous_regret_weighted_rls: bool = False
    ldlt_information_predictive_rls: bool = False
    dominant_aware_log_covariance_predictive_rls: bool = False
    temporal_rotation_min_period: float = 0.0
    temporal_rotation_max_period: float = 2048.0
    associative_memory_rank: int = 0
    associative_memory_layers: int = 0
    associative_memory_value_width: int = 0
    associative_memory_addressing: str = "dense"
    padded_semantic_hadamard_hash: bool = False
    fused_indexed_count_sketch_inference: bool = False
    fused_sparse_count_sketch_prefill: bool = False
    composable_sparse_count_sketch_prefill: bool = False
    explicit_round_forward_pair_scan: bool = False
    exact_whole_prefill_workspace: bool = False
    exact_packed_reversible_projection_prefill: bool = False
    associative_memory_decay: float = 0.95
    scale_invariant_associative_state: bool = False
    associative_memory_scale: float = 0.0
    associative_memory_context_scale: float | None = None
    associative_memory_context_gate_count: float = 0.0
    associative_memory_context_buckets: int = 1
    associative_memory_latent_bits: int = 0
    associative_memory_codebook_size: int = 0
    associative_memory_codebook_rate: float = 0.0
    block_delta_memory_heads: int = 0
    block_delta_memory_key_width: int = 0
    block_delta_memory_projection_rank: int = 0
    gdn2_mixer_layers: int = 0
    gdn2_mixer_heads: int = 0
    gdn2_mixer_key_width: int = 0
    factorization_memory_layers: int = 0
    factorization_memory_slots: int = 0
    factorization_memory_value_width: int = 0
    bounded_attention_layers: int = 0
    bounded_attention_window: int = 0
    bounded_attention_width: int = 0
    bounded_attention_rotary: bool = False
    bounded_attention_utility_gate: bool = False
    bounded_attention_causal_semantic_utility_rank: int = 0
    bounded_attention_channel_preserving_semantic_utility_gate: bool = False
    swiglu_ffn: bool = False
    bounded_recurrent_feature_context: bool = False
    identity_preserving_context_modulation: bool = False
    attention_conditioned_feature_write: bool = False
    rms_eps: float = 1e-6


@dataclass(frozen=True)
class NativePCState:
    """Bounded recurrent state. Every tensor is context-length independent."""

    layers: Tensor
    token_count: int = 0
    token_count_cuda: Tensor | None = None
    manifold_depth_bus: Tensor | None = None
    nonlinear_matrix_state: Tensor | None = None
    recurrent_layers: Tensor | None = None
    feature_context: Tensor | None = None
    associative_memory: Tensor | None = None
    associative_normalizer: Tensor | None = None
    associative_keys: Tensor | None = None
    associative_last_update: Tensor | None = None
    associative_context_tokens: Tensor | None = None
    associative_previous_context_tokens: Tensor | None = None
    associative_semantic_memory: Tensor | None = None
    associative_semantic_normalizer: Tensor | None = None
    associative_semantic_last_update: Tensor | None = None
    associative_semantic_context: Tensor | None = None
    associative_semantic_codes: Tensor | None = None
    associative_pair_recurrence_bloom: Tensor | None = None
    persistent_transition_adapter: Tensor | None = None
    gdn2_memory: Tensor | None = None
    factorization_memory: Tensor | None = None
    bounded_attention_keys: Tensor | None = None
    bounded_attention_values: Tensor | None = None
    hgf_posterior_precision: Tensor | None = None
    causal_fir_input_history: Tensor | None = None
    causal_fir_input_valid_count: Tensor | None = None
    latent_prediction_error: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return self.layers.shape[1]

    @property
    def byte_count(self) -> int:
        tensors = (
            self.layers,
            self.token_count_cuda,
            self.manifold_depth_bus,
            self.nonlinear_matrix_state,
            self.recurrent_layers,
            self.feature_context,
            self.associative_memory,
            self.associative_normalizer,
            self.associative_keys,
            self.associative_last_update,
            self.associative_context_tokens,
            self.associative_previous_context_tokens,
            self.associative_semantic_memory,
            self.associative_semantic_normalizer,
            self.associative_semantic_last_update,
            self.associative_semantic_context,
            self.associative_semantic_codes,
            self.associative_pair_recurrence_bloom,
            self.persistent_transition_adapter,
            self.gdn2_memory,
            self.factorization_memory,
            self.bounded_attention_keys,
            self.bounded_attention_values,
            self.hgf_posterior_precision,
            self.causal_fir_input_history,
            self.causal_fir_input_valid_count,
            self.latent_prediction_error,
        )
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
            if tensor is not None
        )


class NativeReciprocalPCLanguageModel(nn.Module):
    """Bounded-state ReciprocalPC language model.

    Every token clamps the bottom observation. Each settling round performs one
    layer-parallel Jacobi update from a frozen state snapshot. Generative and
    reciprocal matrices remain independent; inference never reads a forward
    weight through its transpose. The default architecture is attention-free;
    experimental hybrids may add a fixed-width local attention cache.
    """

    def __init__(self, config: NativePCConfig) -> None:
        super().__init__()
        if config.vocab_size <= 1:
            raise ValueError("vocab_size must exceed one")
        if not 0 <= config.bos_token_id < config.vocab_size:
            raise ValueError("bos_token_id must be inside vocabulary")
        if min(config.width, config.depth, config.ffn_width) <= 0:
            raise ValueError("width, depth, and ffn_width must be positive")
        if config.kronecker_input_embedding and config.bpe_ancestry_input_embedding:
            raise ValueError("structured input mechanisms are mutually exclusive")
        if config.kronecker_input_embedding:
            codec_sha256 = config.kronecker_input_codec_sha256
            if (
                not config.shared_embedding_decoder
                or config.split_ternary_lexical_endpoint
                or len(codec_sha256) != 64
                or any(character not in "0123456789abcdef" for character in codec_sha256)
            ):
                raise ValueError(
                    "Kronecker input requires one ordinary shared output head and "
                    "a lowercase codec SHA-256"
                )
        elif config.kronecker_input_codec_sha256:
            raise ValueError("Kronecker codec identity requires Kronecker input")
        if config.bpe_ancestry_input_embedding:
            artifact_sha256 = config.bpe_ancestry_input_artifact_sha256
            if (
                not config.shared_embedding_decoder
                or config.split_ternary_lexical_endpoint
                or config.vocab_size != 32_768
                or len(artifact_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in artifact_sha256
                )
            ):
                raise ValueError(
                    "BPE ancestry input requires the ordinary shared output head, "
                    "the frozen vocabulary, and a lowercase artifact SHA-256"
                )
        elif config.bpe_ancestry_input_artifact_sha256:
            raise ValueError("BPE ancestry artifact identity requires ancestry input")
        if (
            config.bpe_ancestry_fused_execution
            and not config.bpe_ancestry_input_embedding
        ):
            raise ValueError("fused BPE ancestry execution requires ancestry input")
        if config.metric_consistent_shared_endpoint and (
            not config.shared_embedding_decoder
            or config.kronecker_input_embedding
            or config.bpe_ancestry_input_embedding
            or config.split_ternary_lexical_endpoint
            or any(
                (
                    config.depth_wide_ternary_lexical_endpoint,
                    config.feature_ready_depth_wide_lexical_residual,
                    config.cozero_reciprocal_lexical_adapters,
                    config.swiglu_wide_lexical_cohead,
                    config.context_routed_ternary_lexical_residual,
                    config.bounded_multiplicative_lexical_modulator,
                    config.signed_square_ternary_lexical_lift,
                    config.cross_channel_quadratic_ternary_lexical_lift,
                    config.shared_base_ternary_delta_mixture,
                    config.state_conditioned_confidence_readout,
                )
            )
        ):
            raise ValueError(
                "metric-consistent scoring requires the sole ordinary shared "
                "lexical endpoint"
            )
        if config.row_energy_preserving_input_compander and (
            not config.shared_embedding_decoder
            or not config.fixed_ternary_hadamard_lexical_predictor
            or not config.temporal_hgf_precision_carry
            or config.kronecker_input_embedding
            or config.bpe_ancestry_input_embedding
            or config.split_ternary_lexical_endpoint
            or config.metric_consistent_shared_endpoint
            or config.energy_preserving_self_bilinear_token_observation
            or config.projected_self_bilinear_token_observation
            or config.self_gated_isometric_sign_token_observation
        ):
            raise ValueError(
                "row-energy-preserving input companding requires the Phase802 "
                "shared fixed-Hadamard input and excludes other input or shared "
                "endpoint variants"
            )
        if config.energy_preserving_token_fingerprint_observation and (
            config.vocab_size != 32_768
            or config.width != 512
            or not config.shared_embedding_decoder
            or not config.fixed_ternary_hadamard_lexical_predictor
            or not config.temporal_hgf_precision_carry
            or config.row_energy_preserving_input_compander
            or config.energy_preserving_self_bilinear_token_observation
            or config.projected_self_bilinear_token_observation
            or config.self_gated_isometric_sign_token_observation
            or config.kronecker_input_embedding
            or config.bpe_ancestry_input_embedding
            or config.split_ternary_lexical_endpoint
            or config.metric_consistent_shared_endpoint
        ):
            raise ValueError(
                "energy-preserving token fingerprints require the exact Phase802 "
                "width-512 shared fixed-Hadamard architecture and exclude other "
                "input or shared-endpoint variants"
            )
        if config.self_dual_hadamard_token_observation_coupling and (
            config.vocab_size != 32_768
            or config.width != 512
            or config.depth != 2
            or config.ffn_width != 1040
            or not config.shared_embedding_decoder
            or not config.fixed_ternary_hadamard_lexical_predictor
            or not config.temporal_hgf_precision_carry
            or config.row_energy_preserving_input_compander
            or config.energy_preserving_token_fingerprint_observation
            or config.energy_preserving_self_bilinear_token_observation
            or config.projected_self_bilinear_token_observation
            or config.self_gated_isometric_sign_token_observation
            or config.kronecker_input_embedding
            or config.bpe_ancestry_input_embedding
            or config.split_ternary_lexical_endpoint
            or config.metric_consistent_shared_endpoint
            or config.fixed_hadamard_top8_lexical_reranker
            or config.shared_nonlinear_candidate_compatibility
        ):
            raise ValueError(
                "self-dual Hadamard token coupling requires the exact Phase802 "
                "width-512 shared fixed-Hadamard architecture and excludes other "
                "input or lexical-endpoint variants"
            )
        if config.energy_preserving_self_bilinear_token_observation and (
            config.vocab_size != 32_768
            or config.width != 512
            or config.depth != 2
            or config.ffn_width != 1040
            or not config.shared_embedding_decoder
            or not config.fixed_ternary_hadamard_lexical_predictor
            or not config.temporal_hgf_precision_carry
            or config.row_energy_preserving_input_compander
            or config.energy_preserving_token_fingerprint_observation
            or config.self_dual_hadamard_token_observation_coupling
            or config.projected_self_bilinear_token_observation
            or config.self_gated_isometric_sign_token_observation
            or config.kronecker_input_embedding
            or config.bpe_ancestry_input_embedding
            or config.split_ternary_lexical_endpoint
            or config.metric_consistent_shared_endpoint
            or config.fixed_hadamard_top8_lexical_reranker
            or config.shared_nonlinear_candidate_compatibility
        ):
            raise ValueError(
                "self-bilinear token observation requires the exact Phase802 "
                "width-512 shared fixed-Hadamard architecture and excludes other "
                "input or lexical-endpoint variants"
            )
        if config.projected_self_bilinear_token_observation and (
            config.vocab_size != 32_768
            or config.width != 512
            or config.depth != 2
            or config.ffn_width != 1040
            or not config.shared_embedding_decoder
            or not config.fixed_ternary_hadamard_lexical_predictor
            or not config.temporal_hgf_precision_carry
            or config.row_energy_preserving_input_compander
            or config.energy_preserving_token_fingerprint_observation
            or config.self_dual_hadamard_token_observation_coupling
            or config.energy_preserving_self_bilinear_token_observation
            or config.self_gated_isometric_sign_token_observation
            or config.kronecker_input_embedding
            or config.bpe_ancestry_input_embedding
            or config.split_ternary_lexical_endpoint
            or config.metric_consistent_shared_endpoint
            or config.fixed_hadamard_top8_lexical_reranker
            or config.shared_nonlinear_candidate_compatibility
            or config.self_gated_ternary_hidden_allocation
        ):
            raise ValueError(
                "projected self-bilinear observation requires the exact "
                "Phase802 width-512 shared fixed-Hadamard architecture and "
                "excludes other input, hidden-allocation, or endpoint variants"
            )
        if config.self_gated_isometric_sign_token_observation and (
            config.vocab_size != 32_768
            or config.width != 512
            or config.depth != 2
            or config.ffn_width != 1040
            or not config.shared_embedding_decoder
            or not config.fixed_ternary_hadamard_lexical_predictor
            or not config.temporal_hgf_precision_carry
            or config.row_energy_preserving_input_compander
            or config.energy_preserving_token_fingerprint_observation
            or config.self_dual_hadamard_token_observation_coupling
            or config.energy_preserving_self_bilinear_token_observation
            or config.projected_self_bilinear_token_observation
            or config.kronecker_input_embedding
            or config.bpe_ancestry_input_embedding
            or config.split_ternary_lexical_endpoint
            or config.metric_consistent_shared_endpoint
            or config.fixed_hadamard_top8_lexical_reranker
            or config.shared_nonlinear_candidate_compatibility
        ):
            raise ValueError(
                "self-gated sign token observation requires the exact Phase802 "
                "width-512 shared fixed-Hadamard architecture and excludes other "
                "input or lexical-endpoint variants"
            )
        if config.self_gated_ternary_hidden_allocation and (
            config.vocab_size != 32_768
            or config.width != 512
            or config.depth != 2
            or config.ffn_width != 1040
            or not config.shared_embedding_decoder
            or not config.fixed_ternary_hadamard_lexical_predictor
            or not config.temporal_hgf_precision_carry
            or not config.swiglu_ffn
            or config.row_energy_preserving_input_compander
            or config.energy_preserving_token_fingerprint_observation
            or config.self_dual_hadamard_token_observation_coupling
            or config.energy_preserving_self_bilinear_token_observation
            or config.projected_self_bilinear_token_observation
            or config.self_gated_isometric_sign_token_observation
            or config.kronecker_input_embedding
            or config.bpe_ancestry_input_embedding
            or config.split_ternary_lexical_endpoint
            or config.metric_consistent_shared_endpoint
            or config.fixed_hadamard_top8_lexical_reranker
            or config.shared_nonlinear_candidate_compatibility
            or config.context_bound_ternary_parameter_superposition
            or config.shared_features_routed_ternary_ffn_output
            or config.routed_ternary_ffn_channel_allocation
            or config.zero_mean_contrastive_ffn_channel_allocation
            or config.deterministic_ternary_ffn_experts
            or config.deterministic_ternary_ffn_residual_experts
            or config.selected_rank_one_ternary_residuals
            or config.selected_diagonal_ternary_residuals
            or config.selected_diagonal_swiglu_residuals
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.intrinsic_ternary_reciprocal_ffn
            or config.lifted_antisymmetric_reciprocal_ffn
            or config.ternary_dale_quadrant_transition
            or config.sign_lifted_dual_ternary_transition
        ):
            raise ValueError(
                "self-gated ternary hidden allocation requires the exact "
                "Phase802 shared fixed-Hadamard SwiGLU architecture and "
                "excludes other input, transition, or endpoint variants"
            )
        if config.intrinsic_ternary_reciprocal_ffn and (
            config.swiglu_ffn
            or config.ffn_width != config.width
            or config.compact_active_triangular_coupling
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.ternary_tpc_rtrl_elementwise_transition
        ):
            raise ValueError(
                "intrinsic reciprocal FFN requires a square non-SwiGLU transition"
            )
        if config.lifted_antisymmetric_reciprocal_ffn and (
            not config.swiglu_ffn
            or not config.reversible_recurrent_coupling
            or config.alternating_triangular_coupling
            or config.compact_active_triangular_coupling
            or config.intrinsic_ternary_reciprocal_ffn
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.ternary_tpc_rtrl_elementwise_transition
            or config.ternary_dale_quadrant_transition
            or config.sign_lifted_dual_ternary_transition
            or config.signed_shunting_ternary_transition
            or config.shared_features_routed_ternary_ffn_output
            or config.routed_ternary_ffn_channel_allocation
            or config.deterministic_ternary_ffn_experts
            or config.deterministic_ternary_ffn_residual_experts
            or config.selected_rank_one_ternary_residuals
            or config.selected_diagonal_ternary_residuals
            or config.selected_diagonal_swiglu_residuals
        ):
            raise ValueError(
                "lifted antisymmetric reciprocal FFN requires the complete "
                "ordinary Phase650 SwiGLU transition"
            )
        if config.depth_tied_reciprocal_transition_core and (
            config.depth < 2
            or not config.swiglu_ffn
            or not config.reversible_recurrent_coupling
            or not config.decoupled_transition_write_gate
            or not config.temporal_hgf_precision_carry
            or config.alternating_triangular_coupling
            or config.compact_active_triangular_coupling
            or config.intrinsic_ternary_reciprocal_ffn
            or config.lifted_antisymmetric_reciprocal_ffn
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.ternary_tpc_rtrl_elementwise_transition
            or config.ternary_dale_quadrant_transition
            or config.sign_lifted_dual_ternary_transition
            or config.signed_shunting_ternary_transition
            or config.shared_features_routed_ternary_ffn_output
            or config.routed_ternary_ffn_channel_allocation
            or config.deterministic_ternary_ffn_experts
            or config.deterministic_ternary_ffn_residual_experts
            or config.selected_rank_one_ternary_residuals
            or config.selected_diagonal_ternary_residuals
            or config.selected_diagonal_swiglu_residuals
            or config.source_separated_transition_compartments
            or config.lexical_coordinate_transition_carrier
            or config.post_coinitialized_lexical_coordinate_transition_carrier
            or config.fixed_identity_lexical_coordinate_transition_core
            or config.context_bound_ternary_parameter_superposition
            or config.balanced_two_of_four_reciprocal_transition
            or config.softsign_reciprocal_transition
        ):
            raise ValueError(
                "depth-tied transition ownership requires the ordinary "
                "multi-depth Phase650 temporal-precision SwiGLU transition"
            )
        if config.balanced_two_of_four_reciprocal_transition and (
            config.depth != 2
            or config.width % 4 != 0
            or config.ffn_width % 4 != 0
            or not config.swiglu_ffn
            or not config.reversible_recurrent_coupling
            or not config.decoupled_transition_write_gate
            or not config.temporal_hgf_precision_carry
            or config.depth_tied_reciprocal_transition_core
            or config.alternating_triangular_coupling
            or config.compact_active_triangular_coupling
            or config.intrinsic_ternary_reciprocal_ffn
            or config.lifted_antisymmetric_reciprocal_ffn
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.ternary_tpc_rtrl_elementwise_transition
            or config.ternary_dale_quadrant_transition
            or config.sign_lifted_dual_ternary_transition
            or config.signed_shunting_ternary_transition
            or config.shared_features_routed_ternary_ffn_output
            or config.routed_ternary_ffn_channel_allocation
            or config.deterministic_ternary_ffn_experts
            or config.deterministic_ternary_ffn_residual_experts
            or config.selected_rank_one_ternary_residuals
            or config.selected_diagonal_ternary_residuals
            or config.selected_diagonal_swiglu_residuals
            or config.source_separated_transition_compartments
            or config.lexical_coordinate_transition_carrier
            or config.post_coinitialized_lexical_coordinate_transition_carrier
            or config.fixed_identity_lexical_coordinate_transition_core
            or config.context_bound_ternary_parameter_superposition
            or config.softsign_reciprocal_transition
        ):
            raise ValueError(
                "balanced two-of-four transition requires the exact independent "
                "Phase650 temporal-precision SwiGLU transition"
            )
        if config.softsign_reciprocal_transition and (
            config.depth < 2
            or not config.swiglu_ffn
            or not config.reversible_recurrent_coupling
            or not config.decoupled_transition_write_gate
            or not config.temporal_hgf_precision_carry
            or config.depth_tied_reciprocal_transition_core
            or config.balanced_two_of_four_reciprocal_transition
            or config.alternating_triangular_coupling
            or config.compact_active_triangular_coupling
            or config.intrinsic_ternary_reciprocal_ffn
            or config.lifted_antisymmetric_reciprocal_ffn
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.ternary_tpc_rtrl_elementwise_transition
            or config.ternary_dale_quadrant_transition
            or config.sign_lifted_dual_ternary_transition
            or config.signed_shunting_ternary_transition
            or config.shared_features_routed_ternary_ffn_output
            or config.routed_ternary_ffn_channel_allocation
            or config.deterministic_ternary_ffn_experts
            or config.deterministic_ternary_ffn_residual_experts
            or config.selected_rank_one_ternary_residuals
            or config.selected_diagonal_ternary_residuals
            or config.selected_diagonal_swiglu_residuals
            or config.source_separated_transition_compartments
            or config.lexical_coordinate_transition_carrier
            or config.post_coinitialized_lexical_coordinate_transition_carrier
            or config.fixed_identity_lexical_coordinate_transition_core
            or config.context_bound_ternary_parameter_superposition
        ):
            raise ValueError(
                "softsign reciprocal transition requires the exact independent "
                "Phase650 temporal-precision SwiGLU transition"
            )
        if config.context_bound_ternary_parameter_superposition and (
            not config.swiglu_ffn
            or not config.reversible_recurrent_coupling
            or config.width % 4 != 0
            or config.ffn_width % 4 != 0
            or config.alternating_triangular_coupling
            or config.compact_active_triangular_coupling
            or config.intrinsic_ternary_reciprocal_ffn
            or config.lifted_antisymmetric_reciprocal_ffn
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.ternary_tpc_rtrl_elementwise_transition
            or config.ternary_dale_quadrant_transition
            or config.sign_lifted_dual_ternary_transition
            or config.signed_shunting_ternary_transition
            or config.shared_features_routed_ternary_ffn_output
            or config.routed_ternary_ffn_channel_allocation
            or config.deterministic_ternary_ffn_experts
            or config.deterministic_ternary_ffn_residual_experts
            or config.selected_rank_one_ternary_residuals
            or config.selected_diagonal_ternary_residuals
            or config.selected_diagonal_swiglu_residuals
            or config.source_separated_transition_compartments
        ):
            raise ValueError(
                "context-bound ternary superposition requires the complete "
                "ordinary Phase650 SwiGLU transition and four-way widths"
            )
        if (
            config.fused_context_bound_ternary_decode
            and not config.context_bound_ternary_parameter_superposition
        ):
            raise ValueError(
                "fused context-bound decode requires context superposition"
            )
        masked_lexical_coordinate_carrier = bool(
            config.lexical_coordinate_transition_carrier
            or config.post_coinitialized_lexical_coordinate_transition_carrier
        )
        lexical_coordinate_carrier = bool(
            masked_lexical_coordinate_carrier
            or config.fixed_identity_lexical_coordinate_transition_core
        )
        if sum(
            (
                config.lexical_coordinate_transition_carrier,
                config.post_coinitialized_lexical_coordinate_transition_carrier,
                config.fixed_identity_lexical_coordinate_transition_core,
            )
        ) > 1:
            raise ValueError("lexical-coordinate carrier modes are exclusive")
        if lexical_coordinate_carrier:
            if config.ffn_width <= config.width:
                raise ValueError(
                    "lexical-coordinate transition requires an FFN tail"
                )
            if not config.swiglu_ffn or not config.reversible_recurrent_coupling:
                raise ValueError(
                    "lexical-coordinate transition requires reversible SwiGLU"
                )
            if any(
                (
                    config.contractive_ternary_reciprocal_butterfly_ffn,
                    config.ternary_tpc_rtrl_elementwise_transition,
                    config.ternary_dale_quadrant_transition,
                    config.sign_lifted_dual_ternary_transition,
                    config.signed_shunting_ternary_transition,
                    config.shared_features_routed_ternary_ffn_output,
                    config.routed_ternary_ffn_channel_allocation,
                    config.deterministic_ternary_ffn_experts,
                    config.deterministic_ternary_ffn_residual_experts,
                    config.selected_rank_one_ternary_residuals,
                    config.selected_diagonal_ternary_residuals,
                    config.selected_diagonal_swiglu_residuals,
                )
            ):
                raise ValueError(
                    "lexical-coordinate transition excludes alternate FFN outputs"
                )
        if config.contractive_ternary_reciprocal_butterfly_ffn and (
            config.width & (config.width - 1)
            or config.compact_active_triangular_coupling
        ):
            raise ValueError(
                "contractive reciprocal butterfly requires power-of-two full width"
            )
        if config.contractive_ternary_reciprocal_butterfly_ffn and any(
            (
                config.deterministic_ternary_ffn_experts,
                config.deterministic_ternary_ffn_residual_experts,
                config.shared_features_routed_ternary_ffn_output,
                config.routed_ternary_ffn_channel_allocation,
                config.selected_rank_one_ternary_residuals,
                config.selected_diagonal_ternary_residuals,
                config.selected_diagonal_swiglu_residuals,
            )
        ):
            raise ValueError(
                "contractive reciprocal butterfly excludes FFN specialists"
            )
        if config.parallel_full_state_reciprocal_proposal and not (
            config.contractive_ternary_reciprocal_butterfly_ffn
            and config.reversible_recurrent_coupling
            and not config.alternating_triangular_coupling
            and config.contractive_reversible_retention
            and config.pair_norm_bounded_convex_recurrence
        ):
            raise ValueError(
                "parallel reciprocal proposal requires the full contractive "
                "butterfly two-write graph"
            )
        if config.ordered_swiglu_rglru_carry and (
            not config.swiglu_ffn
            or not config.reversible_recurrent_coupling
            or config.ternary_tpc_rtrl_elementwise_transition
            or config.alternating_triangular_coupling
            or config.compact_active_triangular_coupling
            or not config.pair_norm_bounded_convex_recurrence
            or config.quadrature_preserving_recurrent_state
            or config.decoupled_bounded_memory_emission
            or config.coefficient_normalized_emission
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.parallel_full_state_reciprocal_proposal
            or config.causal_level_innovation_lifting
            or config.butterfly_channel_diffusion
            or config.bos_boundary_state_reset
        ):
            raise ValueError(
                "ordered SwiGLU RG-LRU carry requires the exact two-write "
                "Phase650 bounded-convex topology"
            )
        if config.causal_level_innovation_lifting and (
            not config.reversible_recurrent_coupling
            or config.alternating_triangular_coupling
            or config.decoupled_bounded_memory_emission
            or config.butterfly_channel_diffusion
            or config.bounded_manifold_depth_bus
            or config.bounded_depth_attention_residual
            or config.swiglu_wide_lexical_cohead
        ):
            raise ValueError(
                "causal level-innovation lifting requires the ordinary full "
                "two-write Phase650 transition"
            )
        if config.signed_shunting_ternary_transition and (
            not config.swiglu_ffn
            or not config.reversible_recurrent_coupling
            or config.alternating_triangular_coupling
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.compact_active_triangular_coupling
            or config.parallel_full_state_reciprocal_proposal
            or config.causal_level_innovation_lifting
            or config.source_separated_transition_compartments
            or config.active_interleaved_source_ownership
            or config.deterministic_ternary_ffn_experts
            or config.deterministic_ternary_ffn_residual_experts
            or config.shared_features_routed_ternary_ffn_output
            or config.routed_ternary_ffn_channel_allocation
            or config.selected_rank_one_ternary_residuals
            or config.selected_diagonal_ternary_residuals
            or config.selected_diagonal_swiglu_residuals
        ):
            raise ValueError(
                "signed shunting requires the ordinary two-write Phase650 "
                "SwiGLU transition"
            )
        if config.ternary_dale_quadrant_transition and (
            not config.swiglu_ffn
            or not config.reversible_recurrent_coupling
            or config.alternating_triangular_coupling
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.compact_active_triangular_coupling
            or config.parallel_full_state_reciprocal_proposal
            or config.causal_level_innovation_lifting
            or config.signed_shunting_ternary_transition
            or config.token_gated_diagonal_handoff
            or config.source_separated_transition_compartments
            or config.active_interleaved_source_ownership
            or config.deterministic_ternary_ffn_experts
            or config.deterministic_ternary_ffn_residual_experts
            or config.shared_features_routed_ternary_ffn_output
            or config.routed_ternary_ffn_channel_allocation
            or config.selected_rank_one_ternary_residuals
            or config.selected_diagonal_ternary_residuals
            or config.selected_diagonal_swiglu_residuals
        ):
            raise ValueError(
                "ternary Dale transition requires the ordinary two-write "
                "Phase650 SwiGLU transition"
            )
        if config.sign_lifted_dual_ternary_transition and (
            not config.swiglu_ffn
            or not config.reversible_recurrent_coupling
            or config.alternating_triangular_coupling
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.compact_active_triangular_coupling
            or config.parallel_full_state_reciprocal_proposal
            or config.causal_level_innovation_lifting
            or config.signed_shunting_ternary_transition
            or config.token_gated_diagonal_handoff
            or config.ternary_dale_quadrant_transition
            or config.source_separated_transition_compartments
            or config.active_interleaved_source_ownership
            or config.deterministic_ternary_ffn_experts
            or config.deterministic_ternary_ffn_residual_experts
            or config.shared_features_routed_ternary_ffn_output
            or config.routed_ternary_ffn_channel_allocation
            or config.selected_rank_one_ternary_residuals
            or config.selected_diagonal_ternary_residuals
            or config.selected_diagonal_swiglu_residuals
        ):
            raise ValueError(
                "sign-lifted dual transition requires the ordinary two-write "
                "Phase650 SwiGLU transition"
            )
        if config.token_gated_diagonal_handoff and (
            config.depth != 2
            or not config.swiglu_ffn
            or not config.reversible_recurrent_coupling
            or config.alternating_triangular_coupling
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.compact_active_triangular_coupling
            or config.parallel_full_state_reciprocal_proposal
            or config.causal_level_innovation_lifting
            or config.signed_shunting_ternary_transition
            or config.source_separated_transition_compartments
            or config.active_interleaved_source_ownership
            or config.bounded_manifold_depth_bus
            or config.bounded_depth_attention_residual
            or config.bounded_recurrent_feature_context
        ):
            raise ValueError(
                "token-gated handoff requires the ordinary depth-2 Phase650 "
                "SwiGLU transition"
            )
        if config.causal_ternary_fir_input_state and (
            not config.swiglu_ffn
            or not config.reversible_recurrent_coupling
            or config.alternating_triangular_coupling
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.compact_active_triangular_coupling
            or config.parallel_full_state_reciprocal_proposal
            or config.causal_level_innovation_lifting
            or config.signed_shunting_ternary_transition
            or config.token_gated_diagonal_handoff
            or config.ternary_dale_quadrant_transition
            or config.sign_lifted_dual_ternary_transition
            or config.source_separated_transition_compartments
            or config.active_interleaved_source_ownership
            or config.bounded_manifold_depth_bus
            or config.bounded_depth_attention_residual
            or config.bounded_recurrent_feature_context
            or config.bos_boundary_state_reset
        ):
            raise ValueError(
                "causal ternary FIR input state requires the ordinary Phase650 "
                "SwiGLU transition without boundary resets"
            )
        if (
            config.causal_ternary_fir_independent_clip
            or config.causal_ternary_fir_fp32_deployment_scales
        ) and not config.causal_ternary_fir_input_state:
            raise ValueError(
                "causal FIR integration repairs require causal FIR input state"
            )
        if config.lagged_latent_error_settling_carrier and (
            not config.swiglu_ffn
            or not config.reversible_recurrent_coupling
            or config.alternating_triangular_coupling
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.compact_active_triangular_coupling
            or config.parallel_full_state_reciprocal_proposal
            or config.causal_level_innovation_lifting
            or config.signed_shunting_ternary_transition
            or config.token_gated_diagonal_handoff
            or config.causal_ternary_fir_input_state
            or config.ternary_dale_quadrant_transition
            or config.sign_lifted_dual_ternary_transition
            or config.source_separated_transition_compartments
            or config.active_interleaved_source_ownership
            or config.bounded_manifold_depth_bus
            or config.bounded_depth_attention_residual
            or config.bounded_recurrent_feature_context
            or config.identity_preserving_context_modulation
            or config.mandatory_full_bandwidth_bottom_clamp
            or config.residual_deep_state_recirculation
        ):
            raise ValueError(
                "lagged latent-error settling carry requires the ordinary "
                "Phase650 SwiGLU transition and settling path"
            )
        if config.mean_free_learned_recurrent_write and (
            not config.swiglu_ffn
            or not config.reversible_recurrent_coupling
            or config.alternating_triangular_coupling
            or not config.contractive_reversible_retention
            or not config.representably_contractive_retention
            or not config.pair_norm_bounded_convex_recurrence
            or config.decoupled_bounded_memory_emission
            or config.compact_active_triangular_coupling
            or config.parallel_full_state_reciprocal_proposal
            or config.causal_level_innovation_lifting
            or config.signed_shunting_ternary_transition
            or config.token_gated_diagonal_handoff
            or config.causal_ternary_fir_input_state
            or config.ternary_dale_quadrant_transition
            or config.sign_lifted_dual_ternary_transition
            or config.source_separated_transition_compartments
            or config.active_interleaved_source_ownership
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.bounded_manifold_depth_bus
            or config.bounded_depth_attention_residual
            or config.bounded_recurrent_feature_context
            or config.bos_boundary_state_reset
        ):
            raise ValueError(
                "mean-free learned write requires the ordinary invertible "
                "Phase650 SwiGLU transition"
            )
        if config.fused_parallel_reciprocal_butterfly_decode and not (
            config.parallel_full_state_reciprocal_proposal
            and config.contractive_ternary_reciprocal_butterfly_ffn
            and config.width == 512
            and config.depth == 2
        ):
            raise ValueError(
                "fused reciprocal decode requires the Phase658 parallel butterfly"
            )
        if config.deterministic_ternary_ffn_experts and not config.swiglu_ffn:
            raise ValueError(
                "deterministic ternary experts require SwiGLU"
            )
        if config.pairwise_competitive_swiglu and (
            not config.swiglu_ffn or config.ffn_width % 2
        ):
            raise ValueError(
                "pairwise competitive SwiGLU requires SwiGLU and even FFN width"
            )
        if config.bilinear_hebbian_ffn and (
            not config.swiglu_ffn
            or config.pairwise_competitive_swiglu
            or config.deterministic_ternary_ffn_experts
            or config.deterministic_ternary_ffn_residual_experts
            or config.shared_features_routed_ternary_ffn_output
            or config.routed_ternary_ffn_channel_allocation
            or config.selected_rank_one_ternary_residuals
            or config.selected_diagonal_ternary_residuals
            or config.selected_diagonal_swiglu_residuals
            or config.swiglu_wide_lexical_cohead
            or config.source_separated_transition_compartments
            or config.intrinsic_ternary_reciprocal_ffn
            or config.contractive_ternary_reciprocal_butterfly_ffn
        ):
            raise ValueError(
                "bilinear Hebbian FFN requires the ordinary Phase650 SwiGLU path"
            )
        if (
            config.deterministic_ternary_ffn_residual_experts
            and not config.swiglu_ffn
        ):
            raise ValueError(
                "deterministic ternary residual experts require SwiGLU"
            )
        if (
            config.selected_rank_one_ternary_residuals
            and not config.swiglu_ffn
        ):
            raise ValueError("rank-one ternary residuals require SwiGLU")
        if (
            config.selected_diagonal_ternary_residuals
            and not config.swiglu_ffn
        ):
            raise ValueError("diagonal ternary residuals require SwiGLU")
        if (
            config.selected_diagonal_swiglu_residuals
            and not config.swiglu_ffn
        ):
            raise ValueError("diagonal SwiGLU residuals require SwiGLU")
        if (
            config.routed_ternary_ffn_channel_allocation
            and not config.swiglu_ffn
        ):
            raise ValueError("routed FFN channel allocation requires SwiGLU")
        if (
            config.shared_features_routed_ternary_ffn_output
            and not config.swiglu_ffn
        ):
            raise ValueError("routed ternary FFN output requires SwiGLU")
        if config.swiglu_wide_lexical_cohead and not (
            config.swiglu_ffn
            and config.reversible_recurrent_coupling
            and not config.alternating_triangular_coupling
            and not config.contractive_ternary_reciprocal_butterfly_ffn
            and not config.parallel_full_state_reciprocal_proposal
            and not config.decoupled_bounded_memory_emission
        ):
            raise ValueError(
                "SwiGLU lexical co-head requires the ordinary ordered "
                "two-write SwiGLU coupling"
            )
        if config.context_routed_ternary_lexical_residual and any(
            (
                config.depth_wide_ternary_lexical_endpoint,
                config.feature_ready_depth_wide_lexical_residual,
                config.cozero_reciprocal_lexical_adapters,
                config.swiglu_wide_lexical_cohead,
                config.signed_square_ternary_lexical_lift,
                config.cross_channel_quadratic_ternary_lexical_lift,
                config.state_conditioned_confidence_readout,
            )
        ):
            raise ValueError(
                "context-routed lexical residual excludes other lexical heads "
                "and confidence scaling"
            )
        if config.source_separated_transition_compartments and (
            not config.swiglu_ffn
            or not config.reversible_recurrent_coupling
            or config.alternating_triangular_coupling
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.compact_active_triangular_coupling
            or config.parallel_full_state_reciprocal_proposal
            or config.deterministic_ternary_ffn_experts
            or config.deterministic_ternary_ffn_residual_experts
            or config.shared_features_routed_ternary_ffn_output
            or config.routed_ternary_ffn_channel_allocation
            or config.selected_rank_one_ternary_residuals
            or config.selected_diagonal_ternary_residuals
            or config.selected_diagonal_swiglu_residuals
        ):
            raise ValueError(
                "source-separated transition compartments require the ordinary "
                "two-write reversible SwiGLU transition"
            )
        if config.active_interleaved_source_ownership and (
            config.source_separated_transition_compartments
            or not config.swiglu_ffn
            or not config.reversible_recurrent_coupling
            or config.alternating_triangular_coupling
            or config.contractive_ternary_reciprocal_butterfly_ffn
            or config.compact_active_triangular_coupling
            or config.parallel_full_state_reciprocal_proposal
            or config.deterministic_ternary_ffn_experts
            or config.deterministic_ternary_ffn_residual_experts
            or config.shared_features_routed_ternary_ffn_output
            or config.routed_ternary_ffn_channel_allocation
            or config.selected_rank_one_ternary_residuals
            or config.selected_diagonal_ternary_residuals
            or config.selected_diagonal_swiglu_residuals
        ):
            raise ValueError(
                "active interleaved source ownership requires the single "
                "ordinary two-write reversible SwiGLU transition"
            )
        if config.bounded_multiplicative_lexical_modulator and (
            not config.shared_embedding_decoder
            or any(
                (
                    config.depth_wide_ternary_lexical_endpoint,
                    config.feature_ready_depth_wide_lexical_residual,
                    config.cozero_reciprocal_lexical_adapters,
                    config.swiglu_wide_lexical_cohead,
                    config.context_routed_ternary_lexical_residual,
                    config.signed_square_ternary_lexical_lift,
                    config.cross_channel_quadratic_ternary_lexical_lift,
                    config.state_conditioned_confidence_readout,
                )
            )
        ):
            raise ValueError(
                "bounded multiplicative lexical modulation requires the "
                "single shared lexical endpoint"
            )
        if config.signed_square_ternary_lexical_lift and (
            not config.shared_embedding_decoder
            or config.split_ternary_lexical_endpoint
            or any(
                (
                    config.depth_wide_ternary_lexical_endpoint,
                    config.feature_ready_depth_wide_lexical_residual,
                    config.cozero_reciprocal_lexical_adapters,
                    config.swiglu_wide_lexical_cohead,
                    config.context_routed_ternary_lexical_residual,
                    config.bounded_multiplicative_lexical_modulator,
                    config.cross_channel_quadratic_ternary_lexical_lift,
                    config.state_conditioned_confidence_readout,
                )
            )
        ):
            raise ValueError(
                "signed-square lexical lift requires the single shared "
                "lexical endpoint"
            )
        if config.cross_channel_quadratic_ternary_lexical_lift and (
            config.width != 512
            or config.depth != 2
            or config.ffn_width != 1040
            or not config.shared_embedding_decoder
            or config.split_ternary_lexical_endpoint
            or any(
                (
                    config.depth_wide_ternary_lexical_endpoint,
                    config.feature_ready_depth_wide_lexical_residual,
                    config.cozero_reciprocal_lexical_adapters,
                    config.swiglu_wide_lexical_cohead,
                    config.context_routed_ternary_lexical_residual,
                    config.bounded_multiplicative_lexical_modulator,
                    config.signed_square_ternary_lexical_lift,
                    config.shared_base_ternary_delta_mixture,
                    config.state_conditioned_confidence_readout,
                )
            )
        ):
            raise ValueError(
                "cross-channel quadratic lexical lift requires the exact "
                "width-512 depth-2 Phase650 shared lexical endpoint"
            )
        if config.single_endpoint_depth_innovation_predictor and (
            config.width != 512
            or config.depth != 2
            or config.ffn_width != 1040
            or not config.shared_embedding_decoder
            or not config.projected_readout
            or not config.hybrid_projected_readout
            or config.split_ternary_lexical_endpoint
            or config.metric_consistent_shared_endpoint
            or any(
                (
                    config.depth_wide_ternary_lexical_endpoint,
                    config.feature_ready_depth_wide_lexical_residual,
                    config.cozero_reciprocal_lexical_adapters,
                    config.swiglu_wide_lexical_cohead,
                    config.context_routed_ternary_lexical_residual,
                    config.bounded_multiplicative_lexical_modulator,
                    config.signed_square_ternary_lexical_lift,
                    config.cross_channel_quadratic_ternary_lexical_lift,
                    config.energy_conserving_lexical_group_contrast,
                    config.causal_anti_hebbian_lexical_butterfly,
                    config.shared_base_ternary_delta_mixture,
                    config.state_conditioned_confidence_readout,
                )
            )
        ):
            raise ValueError(
                "single-endpoint depth innovation requires the exact "
                "width-512 depth-2 Phase650 shared lexical endpoint"
            )
        if config.energy_conserving_lexical_group_contrast and (
            config.width != 512
            or config.depth != 2
            or config.ffn_width != 1040
            or not config.shared_embedding_decoder
            or config.split_ternary_lexical_endpoint
            or any(
                (
                    config.depth_wide_ternary_lexical_endpoint,
                    config.feature_ready_depth_wide_lexical_residual,
                    config.cozero_reciprocal_lexical_adapters,
                    config.swiglu_wide_lexical_cohead,
                    config.context_routed_ternary_lexical_residual,
                    config.bounded_multiplicative_lexical_modulator,
                    config.signed_square_ternary_lexical_lift,
                    config.cross_channel_quadratic_ternary_lexical_lift,
                    config.shared_base_ternary_delta_mixture,
                    config.state_conditioned_confidence_readout,
                )
            )
        ):
            raise ValueError(
                "energy-conserving lexical group contrast requires the exact "
                "width-512 depth-2 Phase650 physical groups and shared endpoint"
            )
        if config.causal_anti_hebbian_lexical_butterfly and (
            config.width != 512
            or config.depth != 2
            or config.ffn_width != 1040
            or not config.shared_embedding_decoder
            or config.split_ternary_lexical_endpoint
            or config.metric_consistent_shared_endpoint
            or any(
                (
                    config.depth_wide_ternary_lexical_endpoint,
                    config.feature_ready_depth_wide_lexical_residual,
                    config.cozero_reciprocal_lexical_adapters,
                    config.swiglu_wide_lexical_cohead,
                    config.context_routed_ternary_lexical_residual,
                    config.bounded_multiplicative_lexical_modulator,
                    config.signed_square_ternary_lexical_lift,
                    config.cross_channel_quadratic_ternary_lexical_lift,
                    config.energy_conserving_lexical_group_contrast,
                    config.shared_base_ternary_delta_mixture,
                    config.state_conditioned_confidence_readout,
                )
            )
        ):
            raise ValueError(
                "causal anti-Hebbian lexical butterfly requires the exact "
                "width-512 depth-2 Phase650 shared endpoint"
            )
        if (
            config.fused_causal_anti_hebbian_lexical_butterfly_execution
            and not config.causal_anti_hebbian_lexical_butterfly
        ):
            raise ValueError(
                "fused anti-Hebbian lexical execution requires Phase742"
            )
        if config.shared_base_ternary_delta_mixture and (
            not config.shared_embedding_decoder
            or any(
                (
                    config.depth_wide_ternary_lexical_endpoint,
                    config.feature_ready_depth_wide_lexical_residual,
                    config.cozero_reciprocal_lexical_adapters,
                    config.swiglu_wide_lexical_cohead,
                    config.context_routed_ternary_lexical_residual,
                    config.bounded_multiplicative_lexical_modulator,
                    config.signed_square_ternary_lexical_lift,
                    config.cross_channel_quadratic_ternary_lexical_lift,
                    config.state_conditioned_confidence_readout,
                )
            )
        ):
            raise ValueError(
                "shared-base ternary-delta mixture requires the single "
                "shared lexical endpoint"
            )
        if (
            config.zero_mean_contrastive_ffn_channel_allocation
            and not config.routed_ternary_ffn_channel_allocation
        ):
            raise ValueError(
                "zero-mean contrastive allocation requires routed allocation"
            )
        if (
            config.selected_diagonal_decode_gather
            and not (
                config.selected_diagonal_ternary_residuals
                or config.selected_diagonal_swiglu_residuals
            )
        ):
            raise ValueError(
                "selected diagonal decode gather requires diagonal residuals"
            )
        if (
            (
                config.selected_rank_one_ternary_residuals
                or config.selected_diagonal_ternary_residuals
                or config.selected_diagonal_swiglu_residuals
                or config.routed_ternary_ffn_channel_allocation
                or config.self_gated_ternary_hidden_allocation
                or config.shared_features_routed_ternary_ffn_output
            )
            and (
                config.deterministic_ternary_ffn_experts
                or config.deterministic_ternary_ffn_residual_experts
            )
        ):
            raise ValueError("vector and FFN expert banks are mutually exclusive")
        if sum(
            int(enabled)
            for enabled in (
                config.selected_rank_one_ternary_residuals,
                config.selected_diagonal_ternary_residuals,
                config.selected_diagonal_swiglu_residuals,
                config.routed_ternary_ffn_channel_allocation,
                config.self_gated_ternary_hidden_allocation,
                config.shared_features_routed_ternary_ffn_output,
            )
        ) > 1:
            raise ValueError("routed vector specialists are mutually exclusive")
        if (
            config.selected_only_residual_expert_decode
            and not config.deterministic_ternary_ffn_residual_experts
        ):
            raise ValueError(
                "selected-only residual decode requires residual experts"
            )
        if (
            config.selected_only_residual_expert_sequence
            and not config.selected_only_residual_expert_decode
        ):
            raise ValueError(
                "selected-only residual sequence requires selected decode"
            )
        if (
            config.cublas_routed_residual_expert_prefill
            and not config.selected_only_residual_expert_decode
        ):
            raise ValueError(
                "cuBLAS routed residual prefill requires selected decode"
            )
        if (
            config.cublas_routed_residual_expert_prefill
            and config.selected_only_residual_expert_sequence
        ):
            raise ValueError(
                "cuBLAS and Triton routed residual prefill are mutually exclusive"
            )
        if (
            config.deterministic_ternary_ffn_experts
            and config.deterministic_ternary_ffn_residual_experts
        ):
            raise ValueError("complete and residual FFN experts are mutually exclusive")
        if (
            config.deterministic_ternary_ffn_residual_experts
            and config.ffn_width % 4
        ):
            raise ValueError("residual FFN expert width must divide into four")
        if (
            (
                config.deterministic_ternary_ffn_experts
                or config.deterministic_ternary_ffn_residual_experts
            )
            and (
                not config.reversible_recurrent_coupling
                or config.alternating_triangular_coupling
            )
        ):
            raise ValueError(
                "deterministic ternary experts require two-write reversible coupling"
            )
        if config.predictive_future_error_key_rank not in (0, 32):
            raise ValueError("predictive future-error key rank must be 0 or 32")
        if config.predictive_future_error_key_rank > config.width:
            raise ValueError("predictive future-error key rank exceeds width")
        if (
            config.token_conditioned_predictive_read_gate
            and config.predictive_future_error_key_rank != 32
        ):
            raise ValueError(
                "token-conditioned predictive read gate requires a rank-32 "
                "future-error key"
            )
        if config.token_conditioned_predictive_read_gate and (
            config.bounded_surprise_exact_cache
            or config.horizon_isolated_predictive_rls
            or config.stability_limited_query_aware_predictive_rls
            or config.self_protecting_predictive_rls
            or config.continuous_regret_weighted_rls
        ):
            raise ValueError(
                "token-conditioned predictive read gate requires the parent "
                "predictive RLS path"
            )
        if (
            config.bounded_surprise_exact_cache
            and config.predictive_future_error_key_rank != 32
        ):
            raise ValueError(
                "bounded surprise exact cache requires a rank-32 future-error key"
            )
        if (
            config.horizon_isolated_predictive_rls
            and config.predictive_future_error_key_rank != 32
        ):
            raise ValueError(
                "horizon-isolated predictive RLS requires a rank-32 future-error key"
            )
        if (
            config.stability_limited_query_aware_predictive_rls
            and config.predictive_future_error_key_rank != 32
        ):
            raise ValueError(
                "query-aware predictive RLS requires a rank-32 future-error key"
            )
        if (
            config.self_protecting_predictive_rls
            and config.predictive_future_error_key_rank != 32
        ):
            raise ValueError(
                "self-protecting predictive RLS requires a rank-32 future-error key"
            )
        if (
            config.continuous_regret_weighted_rls
            and config.predictive_future_error_key_rank != 32
        ):
            raise ValueError(
                "continuous regret-weighted RLS requires a rank-32 future-error key"
            )
        if (
            config.ldlt_information_predictive_rls
            and config.predictive_future_error_key_rank != 32
        ):
            raise ValueError(
                "LDLT information predictive RLS requires a rank-32 future-error key"
            )
        if (
            config.dominant_aware_log_covariance_predictive_rls
            and config.predictive_future_error_key_rank != 32
        ):
            raise ValueError(
                "dominant-aware log-covariance predictive RLS requires a "
                "rank-32 future-error key"
            )
        if config.ldlt_information_predictive_rls and (
            config.token_conditioned_predictive_read_gate
            or config.bounded_surprise_exact_cache
            or config.horizon_isolated_predictive_rls
            or config.stability_limited_query_aware_predictive_rls
            or config.self_protecting_predictive_rls
            or config.continuous_regret_weighted_rls
            or config.dominant_aware_log_covariance_predictive_rls
        ):
            raise ValueError(
                "LDLT information predictive RLS excludes other predictive-memory "
                "variants"
            )
        if config.dominant_aware_log_covariance_predictive_rls and (
            config.token_conditioned_predictive_read_gate
            or config.bounded_surprise_exact_cache
            or config.horizon_isolated_predictive_rls
            or config.stability_limited_query_aware_predictive_rls
            or config.self_protecting_predictive_rls
            or config.continuous_regret_weighted_rls
            or config.ldlt_information_predictive_rls
        ):
            raise ValueError(
                "dominant-aware log-covariance predictive RLS excludes other "
                "predictive-memory variants"
            )
        if (
            config.horizon_isolated_predictive_rls
            and config.bounded_surprise_exact_cache
        ):
            raise ValueError(
                "horizon-isolated predictive RLS excludes bounded surprise cache"
            )
        if config.stability_limited_query_aware_predictive_rls and (
            config.bounded_surprise_exact_cache
            or config.horizon_isolated_predictive_rls
            or config.self_protecting_predictive_rls
            or config.continuous_regret_weighted_rls
        ):
            raise ValueError(
                "query-aware predictive RLS excludes rejected predictive-memory "
                "candidates"
            )
        if config.self_protecting_predictive_rls and (
            config.bounded_surprise_exact_cache
            or config.horizon_isolated_predictive_rls
            or config.stability_limited_query_aware_predictive_rls
            or config.continuous_regret_weighted_rls
        ):
            raise ValueError(
                "self-protecting predictive RLS excludes rejected "
                "predictive-memory candidates"
            )
        if config.continuous_regret_weighted_rls and (
            config.bounded_surprise_exact_cache
            or config.horizon_isolated_predictive_rls
            or config.stability_limited_query_aware_predictive_rls
            or config.self_protecting_predictive_rls
        ):
            raise ValueError(
                "continuous regret-weighted RLS excludes rejected "
                "predictive-memory candidates"
            )
        if config.settling_rounds <= 0:
            raise ValueError("settling_rounds must be positive")
        hgf_sweep_count = sum(
            (
                config.exact_diagonal_hgf_posterior_sweep,
                config.predicted_precision_hgf_posterior_sweep,
                config.temporal_hgf_precision_carry,
            )
        )
        if hgf_sweep_count > 1:
            raise ValueError("HGF posterior sweep variants are mutually exclusive")
        if (
            config.bos_boundary_exact_segmented_training
            and not config.bos_boundary_state_reset
        ):
            raise ValueError(
                "exact segmented training requires BOS boundary state reset"
            )
        if config.depth_stratified_temporal_hierarchy and (
            not config.temporal_hgf_precision_carry
            or config.depth != len(DEPTH_STRATIFIED_TEMPORAL_GATE_BIASES)
            or not config.decoupled_transition_write_gate
        ):
            raise ValueError(
                "depth-stratified temporal hierarchy requires the four-layer "
                "temporal-precision parent with decoupled write gates"
            )
        if config.mandatory_full_bandwidth_bottom_clamp and (
            config.exact_diagonal_hgf_posterior_sweep
            or config.predicted_precision_hgf_posterior_sweep
            or config.temporal_hgf_precision_carry
        ):
            raise ValueError("full-bandwidth bottom clamp excludes HGF sweeps")
        if (
            config.mandatory_ternary_state_carrier
            and not config.mandatory_full_bandwidth_bottom_clamp
        ):
            raise ValueError(
                "ternary state carrier requires the mandatory full-bandwidth clamp"
            )
        if (
            config.residual_deep_state_recirculation
            and config.mandatory_full_bandwidth_bottom_clamp
        ):
            raise ValueError(
                "residual deep-state recirculation excludes mandatory state replacement"
            )
        if config.fused_residual_recirculation_decode and not (
            config.residual_deep_state_recirculation
            and config.width == 512
            and config.depth == 2
        ):
            raise ValueError(
                "fused residual recirculation requires the Phase716 shape"
            )
        if config.ternary_vector_recirculation_affordance and not (
            config.residual_deep_state_recirculation
            and config.fused_residual_recirculation_decode
            and config.width == 512
            and config.depth == 2
        ):
            raise ValueError(
                "ternary vector recirculation requires the exact Phase718 shape"
            )
        if config.causal_diagonal_adaptive_recirculation and not (
            config.residual_deep_state_recirculation
            and config.fused_residual_recirculation_decode
            and not config.ternary_vector_recirculation_affordance
            and config.width == 512
            and config.depth == 2
        ):
            raise ValueError(
                "causal diagonal adaptive recirculation requires the Phase722 shape"
            )
        if (
            (
                config.exact_diagonal_hgf_posterior_sweep
                or config.predicted_precision_hgf_posterior_sweep
                or config.temporal_hgf_precision_carry
            )
            and config.settling_rounds != 1
        ):
            raise ValueError("HGF posterior sweep requires one round")
        if (
            config.deeploop_postnorm_settling
            and config.settling_rounds < 2
            and not (
                config.exact_diagonal_hgf_posterior_sweep
                or config.predicted_precision_hgf_posterior_sweep
                or config.temporal_hgf_precision_carry
            )
        ):
            raise ValueError("DeepLoop post-norm settling requires repeated visits")
        if config.deeploop_postnorm_settling and not config.loop_aware_settling_scaling:
            raise ValueError("DeepLoop post-norm settling requires loop-aware settling")
        if config.visit_framed_settling and config.settling_rounds < 2:
            raise ValueError("visit-framed settling requires at least two rounds")
        if config.untied_settling_visits and config.settling_rounds < 2:
            raise ValueError("untied settling visits require at least two rounds")
        if config.visit_residual_adapters and config.settling_rounds < 2:
            raise ValueError("visit residual adapters require at least two rounds")
        if config.visit_low_rank_adapter_rank < 0:
            raise ValueError("visit low-rank adapter rank must be nonnegative")
        if config.visit_low_rank_adapter_rank > config.width:
            raise ValueError("visit low-rank adapter rank cannot exceed width")
        if config.visit_low_rank_adapter_rank > 0 and config.settling_rounds < 2:
            raise ValueError("visit low-rank adapters require at least two rounds")
        if (
            config.nonlinear_visit_low_rank_adapter
            and config.visit_low_rank_adapter_rank == 0
        ):
            raise ValueError("nonlinear visit adapter requires positive rank")
        if not 0 <= config.dual_stream_visit_adapter_rank <= config.width:
            raise ValueError("dual-stream visit adapter rank must be in [0, width]")
        if config.dual_stream_visit_adapter_rank > 0 and config.settling_rounds < 2:
            raise ValueError("dual-stream visit adapter requires at least two rounds")
        if config.dual_stream_visit_adapter_rank > 0 and (
            config.visit_residual_adapters or config.visit_low_rank_adapter_rank > 0
        ):
            raise ValueError(
                "dual-stream and existing visit adapters are mutually exclusive"
            )
        if not 0 <= config.persistent_transition_adapter_rank <= config.width:
            raise ValueError("persistent transition adapter rank must be in [0, width]")
        if not 0 <= config.selective_temporal_phase_rank <= config.width:
            raise ValueError("selective temporal phase rank must be in [0, width]")
        if not 0 <= config.pairwise_delta_erase_rank <= config.width:
            raise ValueError("pairwise delta erase rank must be in [0, width]")
        if config.untied_settling_visits and (
            config.visit_residual_adapters
            or config.visit_low_rank_adapter_rank > 0
            or config.dual_stream_visit_adapter_rank > 0
        ):
            raise ValueError(
                "untied settling visits and residual adapters are mutually exclusive"
            )
        if not 0 < config.state_step <= 1:
            raise ValueError("state_step must be in (0, 1]")
        if config.top_down_scale < 0:
            raise ValueError("top_down_scale must be nonnegative")
        if (
            min(
                config.residual_stream_scale_start,
                config.residual_stream_scale_end,
            )
            <= 0
        ):
            raise ValueError("residual stream scales must be positive")
        if (
            min(
                config.sensory_reinjection_start,
                config.sensory_reinjection_end,
            )
            < 0
        ):
            raise ValueError("sensory reinjection scales must be nonnegative")
        if config.learn_residual_stream_scales and not config.residual_stream:
            raise ValueError("learned residual scales require residual stream")
        if config.parallel_readout_init not in {"bottom", "uniform"}:
            raise ValueError("parallel readout init must be bottom or uniform")
        if config.parallel_readout and config.projected_readout:
            raise ValueError("parallel and projected readout are mutually exclusive")
        if config.fixed_orthogonal_depth_readout and (
            config.parallel_readout or config.projected_readout
        ):
            raise ValueError(
                "fixed orthogonal depth readout excludes learned readouts"
            )
        if config.fixed_orthogonal_depth_readout and (
            config.width <= 0
            or config.width & (config.width - 1)
            or (config.width.bit_length() - 1) % 2
        ):
            raise ValueError(
                "fixed orthogonal depth readout requires an even-log2 "
                "power-of-two width"
            )
        if config.fixed_ternary_hadamard_lexical_predictor and (
            config.width <= 0 or config.width & (config.width - 1)
        ):
            raise ValueError(
                "fixed ternary Hadamard lexical predictor requires a "
                "power-of-two width"
            )
        if config.fixed_hadamard_top8_lexical_reranker and (
            not config.fixed_ternary_hadamard_lexical_predictor
            or config.vocab_size < 8
            or config.width % 128 != 0
        ):
            raise ValueError(
                "fixed-Hadamard top-eight reranker requires the fixed "
                "Hadamard predictor, at least eight tokens, and width divisible "
                "by 128"
            )
        if config.shared_nonlinear_candidate_compatibility and (
            not config.fixed_ternary_hadamard_lexical_predictor
            or not config.shared_embedding_decoder
            or config.split_ternary_lexical_dense_width != 0
            or config.vocab_size < 8
            or config.width % 128 != 0
            or config.fixed_hadamard_top8_lexical_reranker
        ):
            raise ValueError(
                "shared nonlinear candidate compatibility requires the fixed "
                "Hadamard predictor, unsplit shared endpoint, at least eight "
                "tokens, width divisible by 128, and no token-table reranker"
            )
        if config.fixed_ternary_hadamard_lexical_predictor and any(
            (
                config.depth_wide_ternary_lexical_endpoint,
                config.feature_ready_depth_wide_lexical_residual,
                config.cozero_reciprocal_lexical_adapters,
                config.single_endpoint_depth_innovation_predictor,
                config.energy_conserving_lexical_group_contrast,
                config.causal_anti_hebbian_lexical_butterfly,
                config.lexical_coordinate_transition_carrier,
                config.post_coinitialized_lexical_coordinate_transition_carrier,
                config.fixed_identity_lexical_coordinate_transition_core,
                config.swiglu_wide_lexical_cohead,
                config.context_routed_ternary_lexical_residual,
                config.bounded_multiplicative_lexical_modulator,
                config.signed_square_ternary_lexical_lift,
                config.cross_channel_quadratic_ternary_lexical_lift,
                config.shared_base_ternary_delta_mixture,
            )
        ):
            raise ValueError(
                "fixed ternary Hadamard lexical predictor excludes other "
                "lexical endpoint variants"
            )
        if sum(
            (
                config.depth_wide_ternary_lexical_endpoint,
                config.feature_ready_depth_wide_lexical_residual,
                config.cozero_reciprocal_lexical_adapters,
            )
        ) > 1:
            raise ValueError("depth-wide lexical endpoint modes are mutually exclusive")
        if config.split_ternary_lexical_endpoint and (
            not config.shared_embedding_decoder
            or not 0 < config.split_ternary_lexical_dense_width < config.width
        ):
            raise ValueError(
                "split ternary lexical endpoint requires a shared endpoint and "
                "dense width strictly inside model width"
            )
        if (
            not config.split_ternary_lexical_endpoint
            and config.split_ternary_lexical_dense_width != 0
        ):
            raise ValueError(
                "split ternary lexical dense width requires the split endpoint"
            )
        if config.split_ternary_lexical_endpoint and any(
            (
                config.depth_wide_ternary_lexical_endpoint,
                config.feature_ready_depth_wide_lexical_residual,
                config.cozero_reciprocal_lexical_adapters,
            )
        ):
            raise ValueError("split and depth-wide lexical endpoints are exclusive")
        if (
            config.depth_wide_ternary_lexical_endpoint
            or config.feature_ready_depth_wide_lexical_residual
            or config.cozero_reciprocal_lexical_adapters
        ) and (
            config.depth != 4
            or config.parallel_readout
            or not config.projected_readout
            or not config.hybrid_projected_readout
            or config.fixed_orthogonal_depth_readout
            or config.state_conditioned_confidence_readout
        ):
            raise ValueError(
                "depth-wide lexical endpoint requires the four-depth "
                "Phase606 hybrid readout"
            )
        if config.hybrid_projected_readout and not config.projected_readout:
            raise ValueError("hybrid projected readout requires projected readout")
        if config.state_conditioned_confidence_readout and config.width % 16:
            raise ValueError(
                "state-conditioned confidence readout requires width divisible by 16"
            )
        if config.temporal_rotation_min_period < 0:
            raise ValueError("temporal rotation minimum period must be nonnegative")
        if config.temporal_rotation_max_period <= 0:
            raise ValueError("temporal rotation maximum period must be positive")
        if (
            config.temporal_rotation_min_period > 0
            and config.temporal_rotation_max_period
            < config.temporal_rotation_min_period
        ):
            raise ValueError("temporal rotation periods must be ordered")
        if config.temporal_rotation_min_period > 0 and config.width % 2:
            raise ValueError("temporal rotation requires even width")
        if (
            config.selective_temporal_phase_rank > 0
            and config.temporal_rotation_min_period == 0
        ):
            raise ValueError("selective temporal phase requires temporal rotation")
        if (
            config.pairwise_delta_erase_rank > 0
            and config.temporal_rotation_min_period == 0
        ):
            raise ValueError("pairwise delta erase requires temporal rotation")
        if config.pairwise_delta_erase_rank > 0 and config.width % 2:
            raise ValueError("pairwise delta erase requires even width")
        if config.pairwise_delta_group_width not in (2, 4):
            raise ValueError("pairwise delta group width must be 2 or 4")
        if config.width % config.pairwise_delta_group_width:
            raise ValueError("pairwise delta group width must divide model width")
        if (
            config.pairwise_delta_group_width != 2
            and config.pairwise_delta_erase_rank == 0
        ):
            raise ValueError("grouped delta transition requires delta erase")
        if config.reversible_recurrent_coupling:
            if config.width % 4:
                raise ValueError(
                    "reversible recurrent coupling requires width divisible by four"
                )
            if not config.decoupled_transition_write_gate:
                raise ValueError(
                    "reversible recurrent coupling requires decoupled write gates"
                )
            if config.pairwise_delta_erase_rank > 0:
                raise ValueError(
                    "reversible recurrent coupling replaces pairwise delta erase"
                )
            if config.temporal_rotation_min_period == 0:
                raise ValueError(
                    "reversible recurrent coupling requires temporal rotation"
                )
            if config.selective_temporal_phase_rank > 0:
                raise ValueError(
                    "reversible recurrent coupling excludes selective phase"
                )
            if (
                config.persistent_transition_adapter_rank > 0
                or config.gdn2_mixer_layers > 0
                or config.factorization_memory_layers > 0
                or config.bounded_recurrent_feature_context
                or config.identity_preserving_context_modulation
                or config.attention_conditioned_feature_write
            ):
                raise ValueError(
                    "reversible recurrent coupling supports the attention-free "
                    "coupling transition graph only"
                )
            if config.bounded_attention_layers > 0 and (
                config.alternating_triangular_coupling
                or config.compact_active_triangular_coupling
                or config.butterfly_channel_diffusion
            ):
                raise ValueError(
                    "reversible bounded attention requires the full two-write layout"
                )
        if config.affine_reversible_transition and (
            not config.reversible_recurrent_coupling
            or config.alternating_triangular_coupling
            or config.compact_active_triangular_coupling
            or config.ordered_swiglu_rglru_carry
            or config.parallel_full_state_reciprocal_proposal
        ):
            raise ValueError(
                "affine reversible transition requires the ordinary full two-write "
                "Phase650 coupling"
            )
        if (
            config.alternating_triangular_coupling
            and not config.reversible_recurrent_coupling
        ):
            raise ValueError(
                "alternating triangular coupling requires reversible coupling"
            )
        if (
            config.contractive_reversible_retention
            and not config.reversible_recurrent_coupling
        ):
            raise ValueError(
                "contractive reversible retention requires reversible coupling"
            )
        if (
            config.representably_contractive_retention
            and not config.contractive_reversible_retention
        ):
            raise ValueError(
                "representably contractive retention requires contractive retention"
            )
        if config.pair_norm_bounded_convex_recurrence and (
            not config.reversible_recurrent_coupling
            or not config.contractive_reversible_retention
            or not config.representably_contractive_retention
            or config.alternating_triangular_coupling
        ):
            raise ValueError(
                "pair-norm bounded recurrence requires representably contractive "
                "full two-write reversible coupling"
            )
        if config.hippo_legs_order2_temporal_coordinates and (
            not config.pair_norm_bounded_convex_recurrence
            or config.ordered_swiglu_rglru_carry
            or config.affine_reversible_transition
            or config.alternating_triangular_coupling
            or config.quadrature_preserving_recurrent_state
            or config.decoupled_bounded_memory_emission
            or config.coefficient_normalized_emission
            or config.bos_boundary_state_reset
        ):
            raise ValueError(
                "order-2 HiPPO-LegS requires the ordinary coupled pair-norm "
                "transition without competing recurrence or BOS reset"
            )
        if (
            config.quadrature_preserving_recurrent_state
            and (
                not config.pair_norm_bounded_convex_recurrence
                or config.decoupled_bounded_memory_emission
            )
        ):
            raise ValueError(
                "quadrature-preserving recurrent state requires coupled "
                "pair-norm bounded recurrence"
            )
        if (
            config.local_quadrature_write_credit
            and not config.quadrature_preserving_recurrent_state
        ):
            raise ValueError(
                "local quadrature write credit requires quadrature-preserving "
                "recurrent state"
            )
        if config.bounded_manifold_depth_bus and (
            not config.reversible_recurrent_coupling
            or not config.contractive_reversible_retention
            or not config.representably_contractive_retention
            or not config.pair_norm_bounded_convex_recurrence
            or config.quadrature_preserving_recurrent_state
            or config.decoupled_bounded_memory_emission
        ):
            raise ValueError(
                "bounded manifold depth bus requires the coupled Phase298 "
                "pair-norm reversible transition"
            )
        if (
            config.fused_manifold_depth_bus_inference
            and not config.bounded_manifold_depth_bus
        ):
            raise ValueError(
                "fused manifold depth-bus inference requires the bounded "
                "manifold depth bus"
            )
        if (
            config.bounded_depth_attention_residual
            and config.bounded_manifold_depth_bus
        ):
            raise ValueError(
                "bounded depth attention and manifold depth bus are exclusive"
            )
        if config.bounded_depth_attention_residual and config.depth < 2:
            raise ValueError("bounded depth attention requires depth at least two")
        if (
            config.nonlinear_matrix_state_lane
            and config.width != NONLINEAR_MATRIX_MODEL_WIDTH
        ):
            raise ValueError("nonlinear matrix state lane requires width 256")
        if (
            config.fused_nonlinear_matrix_decode
            and not config.nonlinear_matrix_state_lane
        ):
            raise ValueError(
                "fused nonlinear matrix decode requires nonlinear matrix state"
            )
        if (
            config.decoupled_bounded_memory_emission
            and not config.pair_norm_bounded_convex_recurrence
        ):
            raise ValueError(
                "decoupled bounded memory emission requires pair-norm bounded "
                "recurrence"
            )
        if (
            config.coefficient_normalized_emission
            and not config.decoupled_bounded_memory_emission
        ):
            raise ValueError(
                "coefficient-normalized emission requires decoupled bounded "
                "memory emission"
            )
        if (
            config.alternating_triangular_coupling
            and not config.contractive_reversible_retention
        ):
            raise ValueError(
                "alternating triangular coupling requires contractive retention"
            )
        if config.alternating_triangular_coupling and config.depth % 2:
            raise ValueError(
                "alternating triangular coupling requires even model depth"
            )
        if (
            config.compact_active_triangular_coupling
            and not config.alternating_triangular_coupling
        ):
            raise ValueError(
                "compact active triangular coupling requires alternating coupling"
            )
        if (
            config.butterfly_channel_diffusion
            and not config.compact_active_triangular_coupling
        ):
            raise ValueError(
                "butterfly channel diffusion requires compact active coupling"
            )
        if config.butterfly_channel_diffusion and config.width & (config.width - 1):
            raise ValueError("butterfly channel diffusion requires power-of-two width")
        if (
            config.layer_rotating_bent_butterfly_diffusion
            and not config.butterfly_channel_diffusion
        ):
            raise ValueError(
                "layer-rotating bent butterfly requires butterfly diffusion"
            )
        if config.layer_rotating_bent_butterfly_diffusion and (
            (config.width.bit_length() - 1) % 2
        ):
            raise ValueError("layer-rotating bent butterfly requires even log2 width")
        if config.associative_memory_rank < 0:
            raise ValueError("associative memory rank must be nonnegative")
        if not 0 <= config.associative_memory_value_width <= config.width:
            raise ValueError("associative memory value width must be in [0, width]")
        if (
            config.associative_memory_value_width > 0
            and config.associative_memory_rank == 0
        ):
            raise ValueError("associative memory value width requires positive rank")
        if (
            config.associative_memory_addressing == "dense"
            and config.associative_memory_rank > config.width
        ):
            raise ValueError("dense associative memory rank cannot exceed width")
        if not 0 <= config.associative_memory_layers <= config.depth:
            raise ValueError("associative memory layers must be in [0, depth]")
        if config.associative_memory_addressing not in {
            "dense",
            "block_delta",
            "token_hash",
            "multiscale_count_sketch",
            "lazy_multiscale_count_sketch",
            "indexed_lazy_multiscale_count_sketch",
            *DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS,
            "token_bigram_hash",
            "token_plus_bigram_hash",
            "token_plus_context_bucket",
            "token_plus_latent_hash",
            "token_plus_latent_codebook",
        }:
            raise ValueError(
                "associative memory addressing must be dense, block_delta, token_hash, "
                "multiscale_count_sketch, lazy_multiscale_count_sketch, "
                "indexed_lazy_multiscale_count_sketch, token_bigram_hash, "
                "token_plus_bigram_hash, or "
                "token_plus_context_bucket, token_plus_latent_hash, or "
                "token_plus_latent_codebook"
            )
        block_delta_enabled = config.associative_memory_addressing == "block_delta"
        block_delta_shape = (
            config.block_delta_memory_heads,
            config.block_delta_memory_key_width,
            config.block_delta_memory_projection_rank,
        )
        if block_delta_enabled:
            if min(block_delta_shape) <= 0:
                raise ValueError("block delta memory dimensions must be positive")
            if config.associative_memory_layers == 0:
                raise ValueError("block delta memory requires explicit layers")
            if (
                config.associative_memory_rank
                != config.block_delta_memory_heads * config.block_delta_memory_key_width
            ):
                raise ValueError("block delta rank must equal heads * key width")
            if (
                config.width
                != config.block_delta_memory_heads
                * config.associative_memory_value_width
            ):
                raise ValueError("model width must equal heads * memory value width")
        elif any(block_delta_shape):
            raise ValueError("block delta dimensions require block_delta addressing")
        gdn2_shape = (
            config.gdn2_mixer_layers,
            config.gdn2_mixer_heads,
            config.gdn2_mixer_key_width,
        )
        if config.gdn2_mixer_layers > 0:
            if min(gdn2_shape) <= 0:
                raise ValueError("GDN2 mixer dimensions must be positive")
            if config.gdn2_mixer_layers > config.depth:
                raise ValueError("GDN2 mixer layers cannot exceed model depth")
            if config.width % config.gdn2_mixer_heads != 0:
                raise ValueError("model width must divide evenly across GDN2 heads")
        elif any(gdn2_shape):
            raise ValueError("GDN2 heads and key width require mixer layers")
        factorization_shape = (
            config.factorization_memory_layers,
            config.factorization_memory_slots,
            config.factorization_memory_value_width,
        )
        if config.factorization_memory_layers > 0:
            if min(factorization_shape) <= 0:
                raise ValueError("factorization memory dimensions must be positive")
            if config.factorization_memory_layers > config.depth:
                raise ValueError(
                    "factorization memory layers cannot exceed model depth"
                )
            if config.factorization_memory_value_width > config.width:
                raise ValueError(
                    "factorization memory value width cannot exceed model width"
                )
        elif any(factorization_shape):
            raise ValueError(
                "factorization memory slots and value width require active layers"
            )
        bounded_attention_shape = (
            config.bounded_attention_layers,
            config.bounded_attention_window,
            config.bounded_attention_width,
        )
        if config.bounded_attention_layers > 0:
            if min(bounded_attention_shape) <= 0:
                raise ValueError("bounded attention dimensions must be positive")
            if config.bounded_attention_layers > config.depth:
                raise ValueError("bounded attention layers cannot exceed model depth")
            if config.bounded_attention_width > config.width:
                raise ValueError("bounded attention width cannot exceed model width")
        elif any(bounded_attention_shape):
            raise ValueError("bounded attention window and width require active layers")
        if config.bounded_attention_rotary:
            if config.bounded_attention_layers == 0:
                raise ValueError("rotary bounded attention requires active layers")
            if config.bounded_attention_width % 2:
                raise ValueError("rotary bounded attention width must be even")
        if (
            config.bounded_attention_utility_gate
            and config.bounded_attention_layers == 0
        ):
            raise ValueError("bounded attention utility gate requires active layers")
        semantic_utility_rank = config.bounded_attention_causal_semantic_utility_rank
        if semantic_utility_rank < 0:
            raise ValueError("causal semantic utility rank must be nonnegative")
        if semantic_utility_rank > 0:
            if config.bounded_attention_layers == 0:
                raise ValueError(
                    "causal semantic utility gate requires active attention layers"
                )
            if semantic_utility_rank > config.bounded_attention_width:
                raise ValueError("causal semantic utility rank exceeds attention width")
            if config.bounded_attention_utility_gate:
                raise ValueError("recall utility gate variants are mutually exclusive")
            if config.bounded_attention_channel_preserving_semantic_utility_gate:
                if semantic_utility_rank != config.bounded_attention_width:
                    raise ValueError(
                        "channel-preserving semantic rank must equal attention width"
                    )
                if config.width % semantic_utility_rank:
                    raise ValueError(
                        "channel-preserving semantic rank must divide model width"
                    )
        elif config.bounded_attention_channel_preserving_semantic_utility_gate:
            raise ValueError(
                "channel-preserving semantic gate requires positive semantic rank"
            )
        if (
            config.associative_memory_addressing
            in {
                "token_hash",
                "multiscale_count_sketch",
                "lazy_multiscale_count_sketch",
                "indexed_lazy_multiscale_count_sketch",
                *DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS,
                "token_bigram_hash",
                "token_plus_bigram_hash",
                "token_plus_context_bucket",
                "token_plus_latent_hash",
                "token_plus_latent_codebook",
            }
            and config.associative_memory_layers == 0
        ):
            raise ValueError("token addressing requires explicit memory layers")
        if config.associative_memory_addressing in {
            "multiscale_count_sketch",
            "lazy_multiscale_count_sketch",
            "indexed_lazy_multiscale_count_sketch",
        }:
            bucket_count = math.isqrt(config.vocab_size - 1) + 1
            expected_rank = MULTISCALE_COUNT_SKETCH_TABLES * bucket_count
            if config.associative_memory_rank != expected_rank:
                raise ValueError(
                "multiscale CountSketch rank must equal six times ceil(sqrt(vocab))"
            )
        if config.associative_memory_addressing in DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS:
            bucket_count = math.isqrt(config.vocab_size - 1) + 1
            expected_rank = 2 * MULTISCALE_COUNT_SKETCH_TABLES * bucket_count
            if config.associative_memory_rank != expected_rank:
                raise ValueError(
                    "dual-order CountSketch rank must equal twelve times "
                    "ceil(sqrt(vocab))"
                )
        if (
            config.associative_memory_addressing
            in {
                CAUSAL_SEMANTIC_BACKOFF_COUNT_SKETCH_ADDRESSING,
                PAIR_MISS_SEMANTIC_BACKOFF_COUNT_SKETCH_ADDRESSING,
                PAIR_MISS_DISJOINT_TIMESCALE_SEMANTIC_CHANNELS_ADDRESSING,
            }
        ):
            if config.associative_memory_layers != 1:
                raise ValueError("semantic CountSketch backoff requires one memory layer")
            if config.width < 64 or (
                config.width & (config.width - 1)
                and not config.padded_semantic_hadamard_hash
            ):
                raise ValueError(
                    "semantic CountSketch backoff requires width at least 64 and "
                    "power-of-two width unless padded semantic Hadamard hashing is enabled"
                )
        elif config.padded_semantic_hadamard_hash:
            raise ValueError(
                "padded semantic Hadamard hashing requires semantic CountSketch backoff"
            )
        if config.fused_indexed_count_sketch_inference:
            if config.associative_memory_addressing not in (
                INDEXED_COUNT_SKETCH_ADDRESSINGS
            ):
                raise ValueError(
                    "fused CountSketch inference requires indexed lazy addressing"
                )
            if not config.scale_invariant_associative_state:
                raise ValueError(
                    "fused CountSketch inference requires scale-invariant state"
                )
        if config.fused_sparse_count_sketch_prefill:
            if (
                config.associative_memory_addressing
                != PAIR_MISS_SEMANTIC_BACKOFF_COUNT_SKETCH_ADDRESSING
            ):
                raise ValueError(
                    "sparse CountSketch prefill requires Phase298 pair-miss "
                    "semantic-backoff addressing"
                )
            if not config.scale_invariant_associative_state:
                raise ValueError(
                    "sparse CountSketch prefill requires scale-invariant state"
                )
        if (
            config.composable_sparse_count_sketch_prefill
            and not config.fused_sparse_count_sketch_prefill
        ):
            raise ValueError(
                "composable sparse prefill requires fused sparse prefill"
            )
        if (
            config.explicit_round_forward_pair_scan
            and not config.reversible_recurrent_coupling
        ):
            raise ValueError(
                "explicit-round forward scan requires reversible coupling"
            )
        if config.exact_whole_prefill_workspace:
            if (
                config.associative_memory_addressing
                != PAIR_MISS_SEMANTIC_BACKOFF_COUNT_SKETCH_ADDRESSING
                or not config.reversible_recurrent_coupling
                or not config.residual_stream
            ):
                raise ValueError(
                    "exact whole-prefill workspace requires Phase298 semantic "
                    "backoff, reversible coupling, and residual stream"
                )
        if config.exact_packed_reversible_projection_prefill and (
            not config.reversible_recurrent_coupling
            or config.compact_active_triangular_coupling
        ):
            raise ValueError(
                "packed prefill projection requires noncompact reversible "
                "coupling with a transition write gate"
            )
        if (
            config.associative_memory_addressing
            in {
                "token_plus_bigram_hash",
                "token_plus_context_bucket",
                "token_plus_latent_hash",
                "token_plus_latent_codebook",
            }
            and config.associative_memory_layers < 2
        ):
            raise ValueError(
                "factorized context addressing requires at least two memory layers"
            )
        if config.associative_memory_context_buckets <= 0:
            raise ValueError("associative memory context buckets must be positive")
        if not 0 <= config.associative_memory_latent_bits <= 30:
            raise ValueError("associative memory latent bits must be in [0, 30]")
        if (
            config.associative_memory_addressing == "token_plus_latent_hash"
            and config.associative_memory_latent_bits == 0
        ):
            raise ValueError("latent hash addressing requires positive latent bits")
        if config.associative_memory_codebook_size < 0:
            raise ValueError("associative memory codebook size must be nonnegative")
        if not 0 <= config.associative_memory_codebook_rate <= 1:
            raise ValueError("associative memory codebook rate must be in [0, 1]")
        if config.associative_memory_addressing == "token_plus_latent_codebook":
            if config.associative_memory_codebook_size <= 0:
                raise ValueError("latent codebook addressing requires positive size")
            if config.associative_memory_codebook_rate <= 0:
                raise ValueError("latent codebook addressing requires positive rate")
            if config.associative_memory_codebook_size > config.associative_memory_rank:
                raise ValueError("latent codebook size cannot exceed memory rank")
        if (
            config.associative_memory_addressing == "token_plus_context_bucket"
            and config.associative_memory_rank
            < config.vocab_size * config.associative_memory_context_buckets
        ):
            raise ValueError(
                "context-bucket rank must cover vocab_size * context_buckets"
            )
        if not 0 <= config.associative_memory_decay < 1:
            raise ValueError("associative memory decay must be in [0, 1)")
        if config.scale_invariant_associative_state:
            if config.associative_memory_rank <= 0:
                raise ValueError(
                    "scale-invariant associative state requires positive rank"
                )
            if config.associative_memory_addressing == "block_delta":
                raise ValueError(
                    "scale-invariant associative state excludes block-delta memory"
                )
        if config.associative_memory_scale < 0:
            raise ValueError("associative memory scale must be nonnegative")
        if (
            config.associative_memory_context_scale is not None
            and config.associative_memory_context_scale < 0
        ):
            raise ValueError("associative memory context scale must be nonnegative")
        if (
            config.associative_memory_context_scale is not None
            and config.associative_memory_addressing
            not in {
                "token_plus_bigram_hash",
                "token_plus_context_bucket",
                "token_plus_latent_hash",
                "token_plus_latent_codebook",
            }
        ):
            raise ValueError(
                "associative memory context scale requires factorized context"
            )
        if config.associative_memory_context_gate_count < 0:
            raise ValueError(
                "associative memory context gate count must be nonnegative"
            )
        if (
            config.associative_memory_context_gate_count > 0
            and config.associative_memory_addressing
            not in {
                "token_plus_bigram_hash",
                "token_plus_context_bucket",
                "token_plus_latent_hash",
                "token_plus_latent_codebook",
            }
        ):
            raise ValueError(
                "associative memory context gate requires factorized context"
            )
        if config.associative_memory_scale > 0 and config.associative_memory_rank == 0:
            raise ValueError("associative memory scale requires positive rank")
        if config.rms_eps <= 0:
            raise ValueError("rms_eps must be positive")
        if (
            config.bounded_recurrent_feature_context
            and config.identity_preserving_context_modulation
        ):
            raise ValueError("recurrent feature-context modes are mutually exclusive")
        if (
            config.attention_conditioned_feature_write
            and config.bounded_attention_layers == 0
        ):
            raise ValueError(
                "attention-conditioned feature write requires bounded attention"
            )
        if config.attention_conditioned_feature_write and (
            config.bounded_recurrent_feature_context
            or config.identity_preserving_context_modulation
        ):
            raise ValueError(
                "attention and recurrent feature-context modes are mutually exclusive"
            )

        self.config = config
        if config.energy_preserving_token_fingerprint_observation:
            from pfc.phase819_energy_preserving_token_fingerprint_observation import (
                build_packed_token_fingerprints,
            )

            packed_token_fingerprints = build_packed_token_fingerprints(
                config.vocab_size,
                config.width,
            )
        else:
            packed_token_fingerprints = None
        self.register_buffer(
            "energy_preserving_token_fingerprint_codes",
            packed_token_fingerprints,
            persistent=True,
        )
        depth = config.depth
        transition_owner_count = (
            1 if config.depth_tied_reciprocal_transition_core else depth
        )
        width = config.width
        ffn_width = config.ffn_width
        ffn_input_width = (
            width
            if config.contractive_ternary_reciprocal_butterfly_ffn
            else ffn_width * (2 if config.swiglu_ffn else 1)
        )
        transition_width = (
            width // 2 if config.compact_active_triangular_coupling else width
        )

        associative_read_scales = torch.full(
            (depth,),
            config.associative_memory_scale,
            dtype=torch.float32,
        )
        if (
            config.associative_memory_addressing
            in {
                "token_plus_bigram_hash",
                "token_plus_context_bucket",
                "token_plus_latent_hash",
                "token_plus_latent_codebook",
            }
            and config.associative_memory_context_scale is not None
        ):
            associative_read_scales[1 : config.associative_memory_layers] = (
                config.associative_memory_context_scale
            )
        self.register_buffer(
            "associative_read_scales",
            associative_read_scales,
            persistent=False,
        )
        self.register_buffer(
            "butterfly_layer_signs",
            _bent_hadamard_sign_schedule(depth, width)
            if config.layer_rotating_bent_butterfly_diffusion
            else None,
            persistent=False,
        )
        if config.associative_memory_addressing in INDEXED_COUNT_SKETCH_ADDRESSINGS:
            bucket_count = config.associative_memory_rank // (
                MULTISCALE_COUNT_SKETCH_TABLES
                * (
                    2
                    if config.associative_memory_addressing
                    in DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS
                    else 1
                )
            )
            tokens = torch.arange(config.vocab_size, dtype=torch.long).unsqueeze(-1)
            multipliers = torch.tensor(
                MULTISCALE_COUNT_SKETCH_HASH_MULTIPLIERS,
                dtype=torch.long,
            )
            offsets = torch.tensor(
                MULTISCALE_COUNT_SKETCH_HASH_OFFSETS,
                dtype=torch.long,
            )
            mixed = tokens * multipliers + offsets
            mixed = torch.bitwise_xor(mixed, torch.bitwise_right_shift(mixed, 16))
            local_addresses = mixed.remainder(bucket_count)
            table_offsets = torch.arange(
                MULTISCALE_COUNT_SKETCH_TABLES,
                dtype=torch.long,
            ) * bucket_count
            indexed_addresses = local_addresses + table_offsets

            sign_multipliers = torch.tensor(
                MULTISCALE_COUNT_SKETCH_SIGN_MULTIPLIERS,
                dtype=torch.long,
            )
            sign_offsets = torch.tensor(
                MULTISCALE_COUNT_SKETCH_SIGN_OFFSETS,
                dtype=torch.long,
            )
            sign_hash = tokens * sign_multipliers + sign_offsets
            sign_hash = torch.bitwise_xor(
                sign_hash,
                torch.bitwise_right_shift(sign_hash, 13),
            )
            indexed_signs = torch.where(
                sign_hash.remainder(2) == 0,
                torch.ones_like(sign_hash, dtype=torch.float32),
                -torch.ones_like(sign_hash, dtype=torch.float32),
            )
        else:
            indexed_addresses = None
            indexed_signs = None
        self.register_buffer(
            "indexed_count_sketch_addresses",
            indexed_addresses,
            persistent=False,
        )
        self.register_buffer(
            "indexed_count_sketch_signs",
            indexed_signs,
            persistent=False,
        )
        dual_order_hash_constants = (
            config.associative_memory_addressing
            in DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS
        )
        count_sketch_hash_constants = (
            config.associative_memory_addressing
            in MULTISCALE_COUNT_SKETCH_ADDRESSINGS
        )
        for name, values in (
            ("count_sketch_hash_multipliers", MULTISCALE_COUNT_SKETCH_HASH_MULTIPLIERS),
            ("count_sketch_hash_offsets", MULTISCALE_COUNT_SKETCH_HASH_OFFSETS),
            ("count_sketch_sign_multipliers", MULTISCALE_COUNT_SKETCH_SIGN_MULTIPLIERS),
            ("count_sketch_sign_offsets", MULTISCALE_COUNT_SKETCH_SIGN_OFFSETS),
        ):
            self.register_buffer(
                name,
                torch.tensor(values, dtype=torch.long)
                if count_sketch_hash_constants
                else None,
                persistent=False,
            )
        semantic_count_sketch = (
            config.associative_memory_addressing
            in {
                CAUSAL_SEMANTIC_BACKOFF_COUNT_SKETCH_ADDRESSING,
                PAIR_MISS_SEMANTIC_BACKOFF_COUNT_SKETCH_ADDRESSING,
                PAIR_MISS_DISJOINT_TIMESCALE_SEMANTIC_CHANNELS_ADDRESSING,
            }
        )
        fast_decay = config.associative_memory_decay
        slow_decay = math.sqrt(fast_decay)
        multiscale_decays = (
            torch.tensor(
                [fast_decay] * MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES
                + [slow_decay] * MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES,
                dtype=torch.float32,
            ).repeat(2 if dual_order_hash_constants else 1)
            if config.associative_memory_addressing
            in MULTISCALE_COUNT_SKETCH_ADDRESSINGS
            else None
        )
        self.register_buffer(
            "count_sketch_table_decays",
            multiscale_decays,
            persistent=False,
        )
        self.register_buffer(
            "count_sketch_scan_decays",
            torch.tensor((fast_decay, slow_decay), dtype=torch.float64)
            if multiscale_decays is not None
            else None,
            persistent=False,
        )
        self.register_buffer(
            "semantic_count_sketch_table_decays",
            torch.tensor(
                (fast_decay, fast_decay, fast_decay, slow_decay, slow_decay, slow_decay),
                dtype=torch.float32,
            )
            if semantic_count_sketch
            else None,
            persistent=False,
        )
        self.register_buffer(
            "semantic_count_sketch_bit_weights",
            torch.tensor((1, 2, 4, 8, 16, 32), dtype=torch.int64)
            if semantic_count_sketch
            else None,
            persistent=False,
        )

        embedding_width = (
            config.split_ternary_lexical_dense_width
            if config.split_ternary_lexical_endpoint
            else width
        )
        self.embedding = nn.Embedding(config.vocab_size, embedding_width)
        self.ternary_embedding = (
            nn.Embedding(config.vocab_size, width - embedding_width)
            if config.split_ternary_lexical_endpoint
            else None
        )
        if config.kronecker_input_embedding:
            self.kronecker_input_projection = nn.Parameter(
                torch.empty(KRONECKER_CODEC_DIMENSION, width)
            )
        else:
            self.register_parameter("kronecker_input_projection", None)
        if config.bpe_ancestry_input_embedding:
            self.bpe_ancestry_input_table = nn.Parameter(
                torch.empty(config.vocab_size, width)
            )
        else:
            self.register_parameter("bpe_ancestry_input_table", None)
        self.register_buffer(
            "_kronecker_byte_buffer",
            torch.empty(
                0,
                KRONECKER_POSITION_COUNT,
                dtype=torch.uint8,
            )
            if config.kronecker_input_embedding
            else None,
            persistent=False,
        )
        self.register_buffer(
            "_kronecker_length_buffer",
            torch.empty(0, dtype=torch.int16)
            if config.kronecker_input_embedding
            else None,
            persistent=False,
        )
        self.register_buffer(
            "_kronecker_projection_row_sum",
            torch.empty(width, dtype=torch.float32)
            if config.kronecker_input_embedding
            else None,
            persistent=False,
        )
        self.register_buffer(
            "_bpe_ancestry_indices",
            torch.empty(0, ANCESTRY_MAX_NODES, dtype=torch.int32)
            if config.bpe_ancestry_input_embedding
            else None,
            persistent=False,
        )
        self.register_buffer(
            "_bpe_ancestry_counts",
            torch.empty(0, dtype=torch.int16)
            if config.bpe_ancestry_input_embedding
            else None,
            persistent=False,
        )
        self.register_buffer(
            "_bpe_ancestry_inverse_sqrt_counts",
            torch.empty(0, dtype=torch.float32)
            if config.bpe_ancestry_input_embedding
            else None,
            persistent=False,
        )
        self.register_buffer(
            "_metric_shared_row_norm_bias",
            torch.empty(config.vocab_size, dtype=torch.float32)
            if config.metric_consistent_shared_endpoint
            else None,
            persistent=False,
        )
        self._kronecker_codec_sha256: str | None = None
        self._bpe_ancestry_artifact_sha256: str | None = None
        self._bpe_ancestry_buffer_sha256: str | None = None
        self._bpe_ancestry_parents_cpu: Tensor | None = None
        if config.shared_embedding_decoder:
            self.register_parameter("decoder", None)
        else:
            self.decoder = nn.Parameter(torch.empty(width, config.vocab_size))
        if config.fixed_ternary_hadamard_lexical_predictor:
            self.token_predictor = nn.Parameter(
                torch.empty(0),
                requires_grad=False,
            )
            self.register_buffer(
                "fixed_ternary_hadamard_lexical_signs",
                torch.empty(width, dtype=torch.int8),
            )
        else:
            self.token_predictor = nn.Parameter(torch.empty(width, width))
            self.register_buffer(
                "fixed_ternary_hadamard_lexical_signs",
                None,
            )
        if config.fixed_hadamard_top8_lexical_reranker:
            self.top8_lexical_reranker_codes = nn.Parameter(
                torch.zeros(config.vocab_size, width)
            )
        else:
            self.register_parameter("top8_lexical_reranker_codes", None)
        if config.shared_nonlinear_candidate_compatibility:
            self.shared_candidate_compatibility_vector = nn.Parameter(
                torch.zeros(1, width)
            )
        else:
            self.register_parameter(
                "shared_candidate_compatibility_vector",
                None,
            )
        if config.self_dual_hadamard_token_observation_coupling:
            self.token_observation_coupling_diagonals = nn.Parameter(
                torch.zeros(2, width // 2)
            )
        else:
            self.register_parameter(
                "token_observation_coupling_diagonals",
                None,
            )
        if config.projected_self_bilinear_token_observation:
            self.projected_self_bilinear_observation_matrix = nn.Parameter(
                torch.zeros(width, width)
            )
        else:
            self.register_parameter(
                "projected_self_bilinear_observation_matrix",
                None,
            )
        if config.single_endpoint_depth_innovation_predictor:
            self.depth_innovation_predictor = nn.Parameter(
                torch.zeros(width, width)
            )
        else:
            self.register_parameter("depth_innovation_predictor", None)
        if config.causal_anti_hebbian_lexical_butterfly:
            from pfc.phase742_anti_hebbian_lexical_butterfly import (
                LEXICAL_BUTTERFLY_PAIRS,
                LEXICAL_BUTTERFLY_STAGES,
            )

            self.anti_hebbian_lexical_butterfly_coefficients = nn.Parameter(
                torch.zeros(LEXICAL_BUTTERFLY_STAGES, LEXICAL_BUTTERFLY_PAIRS)
            )
        else:
            self.register_parameter(
                "anti_hebbian_lexical_butterfly_coefficients",
                None,
            )
        if config.depth_wide_ternary_lexical_endpoint:
            self.depth_lexical_predictors = nn.Parameter(
                torch.zeros(config.depth - 1, width, width)
            )
        else:
            self.register_parameter("depth_lexical_predictors", None)
        if (
            config.depth_wide_ternary_lexical_endpoint
            or config.feature_ready_depth_wide_lexical_residual
            or config.cozero_reciprocal_lexical_adapters
        ):
            self.depth_lexical_decoders = nn.Parameter(
                torch.empty(config.depth - 1, width, config.vocab_size)
            )
        else:
            self.register_parameter("depth_lexical_decoders", None)
        if config.swiglu_wide_lexical_cohead:
            self.swiglu_lexical_decoder = nn.Parameter(
                torch.zeros(config.ffn_width, config.vocab_size)
            )
        else:
            self.register_parameter("swiglu_lexical_decoder", None)
        if config.context_routed_ternary_lexical_residual:
            self.context_routed_lexical_residual_decoders = nn.Parameter(
                torch.zeros(4, width, config.vocab_size)
            )
        else:
            self.register_parameter(
                "context_routed_lexical_residual_decoders",
                None,
            )
        if config.bounded_multiplicative_lexical_modulator:
            self.bounded_multiplicative_lexical_decoder = nn.Parameter(
                torch.zeros(width, config.vocab_size)
            )
        else:
            self.register_parameter(
                "bounded_multiplicative_lexical_decoder",
                None,
            )
        if config.signed_square_ternary_lexical_lift:
            self.signed_square_lexical_decoder = nn.Parameter(
                torch.zeros(width, config.vocab_size)
            )
        else:
            self.register_parameter(
                "signed_square_lexical_decoder",
                None,
            )
        if config.cross_channel_quadratic_ternary_lexical_lift:
            self.cross_channel_quadratic_lexical_decoder = nn.Parameter(
                torch.zeros(width, config.vocab_size)
            )
        else:
            self.register_parameter(
                "cross_channel_quadratic_lexical_decoder",
                None,
            )
        if config.shared_base_ternary_delta_mixture:
            self.shared_base_ternary_delta_decoder = nn.Parameter(
                torch.zeros(width, config.vocab_size)
            )
            gate_indices = torch.arange(width)
            gate_signs = torch.where(
                gate_indices.remainder(2) == 0,
                1.0,
                -1.0,
            )
        else:
            self.register_parameter("shared_base_ternary_delta_decoder", None)
            gate_signs = None
        self.register_buffer(
            "shared_base_delta_gate_signs",
            gate_signs,
            persistent=False,
        )
        if config.mandatory_full_bandwidth_bottom_clamp:
            self.full_bandwidth_value = nn.Parameter(torch.empty(width, width))
            self.full_bandwidth_gate = nn.Parameter(torch.empty(width, width))
        else:
            self.register_parameter("full_bandwidth_value", None)
            self.register_parameter("full_bandwidth_gate", None)
        if config.token_gated_diagonal_handoff:
            self.diagonal_handoff_token_gate = nn.Parameter(
                torch.zeros(1, width, width)
            )
        else:
            self.register_parameter("diagonal_handoff_token_gate", None)
        if config.affine_reversible_transition:
            self.affine_reversible_scale = nn.Parameter(
                torch.empty(
                    transition_owner_count,
                    width // 2,
                    width // 2,
                )
            )
        else:
            self.register_parameter("affine_reversible_scale", None)
        if config.ternary_vector_recirculation_affordance:
            self.ternary_recirculation_selector = nn.Parameter(
                torch.zeros(1, width)
            )
        else:
            self.register_parameter("ternary_recirculation_selector", None)
        self._ternary_vector_recirculation_active = False
        adaptive_controller_names = (
            "adaptive_recirculation_alpha_source",
            "adaptive_recirculation_alpha_destination",
            "adaptive_recirculation_beta_source",
            "adaptive_recirculation_beta_destination",
        )
        for name in adaptive_controller_names:
            if config.causal_diagonal_adaptive_recirculation:
                setattr(self, name, nn.Parameter(torch.zeros(1, width)))
            else:
                self.register_parameter(name, None)
        if config.causal_ternary_fir_input_state:
            self.causal_ternary_fir_input_taps = nn.Parameter(
                torch.zeros(config.depth, CAUSAL_TERNARY_FIR_TAPS, width)
            )
        else:
            self.register_parameter("causal_ternary_fir_input_taps", None)
        if config.lagged_latent_error_settling_carrier:
            self.latent_error_settling_gain = nn.Parameter(
                torch.zeros(config.depth, width)
            )
        else:
            self.register_parameter("latent_error_settling_gain", None)
        if config.predictive_future_error_key_rank > 0:
            self.predictive_future_error_key_projection = nn.Parameter(
                torch.empty(width, config.predictive_future_error_key_rank)
            )
        else:
            self.register_parameter("predictive_future_error_key_projection", None)
        if config.token_conditioned_predictive_read_gate:
            self.predictive_read_gate_projection = nn.Parameter(
                torch.zeros(width, 32)
            )
        else:
            self.register_parameter("predictive_read_gate_projection", None)
        if config.bounded_surprise_exact_cache:
            self.predictive_exception_cache_gate = nn.Parameter(torch.empty(32))
        else:
            self.register_parameter("predictive_exception_cache_gate", None)
        if config.bounded_manifold_depth_bus:
            self.manifold_depth_bus_controller = nn.Parameter(
                torch.empty(depth, 2 * width, 5)
            )
            self.manifold_depth_bus_bias = nn.Parameter(torch.zeros(depth, 5))
        else:
            self.register_parameter("manifold_depth_bus_controller", None)
            self.register_parameter("manifold_depth_bus_bias", None)
        if config.bounded_depth_attention_residual:
            self.depth_attention_queries = nn.Parameter(
                torch.empty(depth - 1, width)
            )
            self.depth_attention_residual_logits = nn.Parameter(
                torch.zeros(depth - 1)
            )
        else:
            self.register_parameter("depth_attention_queries", None)
            self.register_parameter("depth_attention_residual_logits", None)
        if config.nonlinear_matrix_state_lane:
            self.nonlinear_matrix_input_projection = nn.Parameter(
                torch.empty(width, NONLINEAR_MATRIX_PROJECTION_WIDTH)
            )
            self.nonlinear_matrix_transition = nn.Parameter(
                torch.empty(
                    NONLINEAR_MATRIX_HEADS,
                    NONLINEAR_MATRIX_VALUE_WIDTH,
                    NONLINEAR_MATRIX_VALUE_WIDTH,
                )
            )
            self.nonlinear_matrix_forget_bias = nn.Parameter(
                torch.empty(NONLINEAR_MATRIX_HEADS)
            )
            self.nonlinear_matrix_residual_weight = nn.Parameter(torch.empty(width))
            self.nonlinear_matrix_output_norm = nn.Parameter(torch.empty(width))
            self.nonlinear_matrix_output_scale = nn.Parameter(torch.empty(width))
        else:
            self.register_parameter("nonlinear_matrix_input_projection", None)
            self.register_parameter("nonlinear_matrix_transition", None)
            self.register_parameter("nonlinear_matrix_forget_bias", None)
            self.register_parameter("nonlinear_matrix_residual_weight", None)
            self.register_parameter("nonlinear_matrix_output_norm", None)
            self.register_parameter("nonlinear_matrix_output_scale", None)

        # Stacked same-shaped layer weights. Layout is input x output, matching
        # matmul reads required by WGSL. No runtime transposes. Temporal state
        # uses an associative diagonal recurrence; `temporal` maps the current
        # lower-layer observation into its candidate write.
        if config.ternary_tpc_rtrl_elementwise_transition:
            if transition_width != width:
                raise ValueError("Phase714 RG-LRU requires full-width transition state")
            self.register_parameter("temporal", None)
            self.register_parameter("gate", None)
            self.rglru_recurrent_gate_weight = nn.Parameter(
                torch.empty(depth, width, width)
            )
            self.rglru_input_gate_weight = nn.Parameter(
                torch.empty(depth, width, width)
            )
            self.rglru_recurrent_gate_bias = nn.Parameter(torch.zeros(depth, width))
            self.rglru_input_gate_bias = nn.Parameter(torch.zeros(depth, width))
            self.rglru_recurrent_parameter = nn.Parameter(torch.empty(depth, width))
        else:
            self.temporal = nn.Parameter(torch.empty(depth, transition_width, width))
            self.gate = nn.Parameter(torch.empty(depth, transition_width, width))
            self.register_parameter("rglru_recurrent_gate_weight", None)
            self.register_parameter("rglru_input_gate_weight", None)
            self.register_parameter("rglru_recurrent_gate_bias", None)
            self.register_parameter("rglru_input_gate_bias", None)
            self.register_parameter("rglru_recurrent_parameter", None)
        if config.ordered_swiglu_rglru_carry:
            half = width // 2
            ordered_shape = (depth, 2, half, half)
            ordered_bias_shape = (depth, 2, half)
            self.ordered_rglru_recurrent_gate_weight = nn.Parameter(
                torch.empty(ordered_shape)
            )
            self.ordered_rglru_input_gate_weight = nn.Parameter(
                torch.empty(ordered_shape)
            )
            self.ordered_rglru_recurrent_gate_bias = nn.Parameter(
                torch.zeros(ordered_bias_shape)
            )
            self.ordered_rglru_input_gate_bias = nn.Parameter(
                torch.zeros(ordered_bias_shape)
            )
            self.ordered_rglru_recurrent_parameter = nn.Parameter(
                torch.empty(ordered_bias_shape)
            )
        else:
            self.register_parameter("ordered_rglru_recurrent_gate_weight", None)
            self.register_parameter("ordered_rglru_input_gate_weight", None)
            self.register_parameter("ordered_rglru_recurrent_gate_bias", None)
            self.register_parameter("ordered_rglru_input_gate_bias", None)
            self.register_parameter("ordered_rglru_recurrent_parameter", None)
        self.generative = nn.Parameter(torch.empty(depth, width, width))
        self.reciprocal = nn.Parameter(torch.empty(depth, width, width))
        self.candidate = nn.Parameter(torch.empty(depth, width, width))
        if config.untied_settling_visits:
            self.later_candidates = nn.Parameter(
                torch.empty(config.settling_rounds - 1, depth, width, width)
            )
        else:
            self.register_parameter("later_candidates", None)
        if config.visit_residual_adapters:
            self.later_candidate_adapters = nn.Parameter(
                torch.zeros(config.settling_rounds - 1, depth, width)
            )
        else:
            self.register_parameter("later_candidate_adapters", None)
        if config.visit_low_rank_adapter_rank > 0:
            rank = config.visit_low_rank_adapter_rank
            self.later_candidate_adapter_down = nn.Parameter(
                torch.empty(config.settling_rounds - 1, depth, width, rank)
            )
            self.later_candidate_adapter_up = nn.Parameter(
                torch.zeros(config.settling_rounds - 1, depth, rank, width)
            )
        else:
            self.register_parameter("later_candidate_adapter_down", None)
            self.register_parameter("later_candidate_adapter_up", None)
        if config.dual_stream_visit_adapter_rank > 0:
            rank = config.dual_stream_visit_adapter_rank
            visits = config.settling_rounds - 1
            self.later_dual_stream_adapter_down = nn.Parameter(
                torch.empty(visits, depth, 4, width, rank)
            )
            self.later_dual_stream_adapter_up = nn.Parameter(
                torch.zeros(visits, depth, 2, rank, width)
            )
        else:
            self.register_parameter("later_dual_stream_adapter_down", None)
            self.register_parameter("later_dual_stream_adapter_up", None)
        if config.persistent_transition_adapter_rank > 0:
            rank = config.persistent_transition_adapter_rank
            self.persistent_transition_adapter_down = nn.Parameter(
                torch.empty(depth, width, rank)
            )
            self.persistent_transition_adapter_up = nn.Parameter(
                torch.zeros(depth, rank, width)
            )
            self.persistent_transition_adapter_decay_logit = nn.Parameter(
                torch.empty(depth, rank)
            )
        else:
            self.register_parameter("persistent_transition_adapter_down", None)
            self.register_parameter("persistent_transition_adapter_up", None)
            self.register_parameter(
                "persistent_transition_adapter_decay_logit",
                None,
            )
        if config.decoupled_transition_write_gate:
            self.transition_write_gate = nn.Parameter(
                torch.empty(depth, transition_width, width)
            )
            self.transition_write_gate_bias = nn.Parameter(
                torch.empty(depth, 1, transition_width)
            )
        else:
            self.register_parameter("transition_write_gate", None)
            self.register_parameter("transition_write_gate_bias", None)
        if config.pairwise_delta_erase_rank > 0:
            rank = config.pairwise_delta_erase_rank
            self.pairwise_delta_down = nn.Parameter(torch.empty(depth, width, rank))
            self.pairwise_delta_key_up = nn.Parameter(torch.empty(depth, rank, width))
            self.pairwise_delta_erase_up = nn.Parameter(torch.zeros(depth, rank, width))
            self.pairwise_delta_erase_bias = nn.Parameter(torch.zeros(depth, 1, width))
            self.pairwise_delta_strength = nn.Parameter(torch.zeros(depth))
        else:
            self.register_parameter("pairwise_delta_down", None)
            self.register_parameter("pairwise_delta_key_up", None)
            self.register_parameter("pairwise_delta_erase_up", None)
            self.register_parameter("pairwise_delta_erase_bias", None)
            self.register_parameter("pairwise_delta_strength", None)
        if config.associative_memory_addressing == "block_delta":
            memory_layers = self._associative_layer_count()
            projection_rank = config.block_delta_memory_projection_rank
            key_channels = config.associative_memory_rank
            self.block_delta_down = nn.Parameter(
                torch.empty(memory_layers, width, projection_rank)
            )
            self.block_delta_query_up = nn.Parameter(
                torch.empty(memory_layers, projection_rank, key_channels)
            )
            self.block_delta_key_up = nn.Parameter(
                torch.empty(memory_layers, projection_rank, key_channels)
            )
            self.block_delta_erase_up = nn.Parameter(
                torch.zeros(memory_layers, projection_rank, key_channels)
            )
            self.block_delta_decay_up = nn.Parameter(
                torch.zeros(memory_layers, projection_rank, key_channels)
            )
            self.block_delta_write_up = nn.Parameter(
                torch.zeros(memory_layers, projection_rank, width)
            )
            self.block_delta_erase_bias = nn.Parameter(
                torch.zeros(memory_layers, 1, key_channels)
            )
            decay = max(config.associative_memory_decay, config.rms_eps)
            decay_logit = math.log(decay / (1.0 - decay))
            self.block_delta_decay_bias = nn.Parameter(
                torch.full((memory_layers, 1, key_channels), decay_logit)
            )
            self.block_delta_write_bias = nn.Parameter(
                torch.zeros(memory_layers, 1, width)
            )
        else:
            self.register_parameter("block_delta_down", None)
            self.register_parameter("block_delta_query_up", None)
            self.register_parameter("block_delta_key_up", None)
            self.register_parameter("block_delta_erase_up", None)
            self.register_parameter("block_delta_decay_up", None)
            self.register_parameter("block_delta_write_up", None)
            self.register_parameter("block_delta_erase_bias", None)
            self.register_parameter("block_delta_decay_bias", None)
            self.register_parameter("block_delta_write_bias", None)
        if config.gdn2_mixer_layers > 0:
            mixer_layers = config.gdn2_mixer_layers
            heads = config.gdn2_mixer_heads
            key_channels = heads * config.gdn2_mixer_key_width
            value_width = width // heads
            self.gdn2_query = nn.Parameter(
                torch.empty(mixer_layers, width, key_channels)
            )
            self.gdn2_key = nn.Parameter(torch.empty(mixer_layers, width, key_channels))
            self.gdn2_value = nn.Parameter(torch.empty(mixer_layers, width, width))
            self.gdn2_decay_down = nn.Parameter(
                torch.empty(mixer_layers, width, value_width)
            )
            self.gdn2_decay_up = nn.Parameter(
                torch.empty(mixer_layers, value_width, key_channels)
            )
            self.gdn2_erase = nn.Parameter(
                torch.empty(mixer_layers, width, key_channels)
            )
            self.gdn2_write = nn.Parameter(torch.empty(mixer_layers, width, width))
            self.gdn2_decay_log_rate = nn.Parameter(torch.empty(mixer_layers, heads))
            self.gdn2_decay_bias = nn.Parameter(torch.empty(mixer_layers, key_channels))
            self.gdn2_output_gate_down = nn.Parameter(
                torch.empty(mixer_layers, width, value_width)
            )
            self.gdn2_output_gate_up = nn.Parameter(
                torch.empty(mixer_layers, value_width, width)
            )
            self.gdn2_output_gate_bias = nn.Parameter(
                torch.zeros(mixer_layers, 1, width)
            )
            self.gdn2_output_norm = nn.Parameter(
                torch.ones(mixer_layers, 1, 1, value_width)
            )
            self.gdn2_output = nn.Parameter(torch.zeros(mixer_layers, width, width))
        else:
            self.register_parameter("gdn2_query", None)
            self.register_parameter("gdn2_key", None)
            self.register_parameter("gdn2_value", None)
            self.register_parameter("gdn2_decay_down", None)
            self.register_parameter("gdn2_decay_up", None)
            self.register_parameter("gdn2_erase", None)
            self.register_parameter("gdn2_write", None)
            self.register_parameter("gdn2_decay_log_rate", None)
            self.register_parameter("gdn2_decay_bias", None)
            self.register_parameter("gdn2_output_gate_down", None)
            self.register_parameter("gdn2_output_gate_up", None)
            self.register_parameter("gdn2_output_gate_bias", None)
            self.register_parameter("gdn2_output_norm", None)
            self.register_parameter("gdn2_output", None)
        if config.factorization_memory_layers > 0:
            memory_layers = config.factorization_memory_layers
            slots = config.factorization_memory_slots
            memory_width = config.factorization_memory_value_width
            self.factorization_affinity = nn.Parameter(
                torch.empty(memory_layers, width, slots)
            )
            self.factorization_update = nn.Parameter(torch.empty(memory_layers, width))
            self.factorization_update_bias = nn.Parameter(torch.zeros(memory_layers))
            self.factorization_merge = nn.Parameter(torch.empty(memory_layers, width))
            self.factorization_merge_bias = nn.Parameter(torch.zeros(memory_layers))
            self.factorization_value_encoder = nn.Parameter(
                torch.empty(memory_layers, width, memory_width)
            )
            self.factorization_value_decoder = nn.Parameter(
                torch.zeros(memory_layers, memory_width, width)
            )
        else:
            self.register_parameter("factorization_affinity", None)
            self.register_parameter("factorization_update", None)
            self.register_parameter("factorization_update_bias", None)
            self.register_parameter("factorization_merge", None)
            self.register_parameter("factorization_merge_bias", None)
            self.register_parameter("factorization_value_encoder", None)
            self.register_parameter("factorization_value_decoder", None)
        if config.bounded_attention_layers > 0:
            attention_layers = config.bounded_attention_layers
            attention_width = config.bounded_attention_width
            self.bounded_attention_query = nn.Parameter(
                torch.empty(attention_layers, width, attention_width)
            )
            self.bounded_attention_key = nn.Parameter(
                torch.empty(attention_layers, width, attention_width)
            )
            self.bounded_attention_value = nn.Parameter(
                torch.empty(attention_layers, width, attention_width)
            )
            self.bounded_attention_output = nn.Parameter(
                torch.zeros(attention_layers, attention_width, width)
            )
        else:
            self.register_parameter("bounded_attention_query", None)
            self.register_parameter("bounded_attention_key", None)
            self.register_parameter("bounded_attention_value", None)
            self.register_parameter("bounded_attention_output", None)
        if config.bounded_attention_utility_gate:
            self.bounded_attention_utility_weight = nn.Parameter(
                torch.zeros(config.bounded_attention_layers, 4)
            )
            self.bounded_attention_utility_bias = nn.Parameter(
                torch.zeros(config.bounded_attention_layers)
            )
        else:
            self.register_parameter("bounded_attention_utility_weight", None)
            self.register_parameter("bounded_attention_utility_bias", None)
        if semantic_utility_rank > 0:
            semantic_shape = (
                config.bounded_attention_layers,
                width,
                semantic_utility_rank,
            )
            self.bounded_attention_semantic_utility_source = nn.Parameter(
                torch.empty(semantic_shape)
            )
            self.bounded_attention_semantic_utility_read = nn.Parameter(
                torch.zeros(semantic_shape)
            )
            semantic_bias_shape = (
                (config.bounded_attention_layers, semantic_utility_rank)
                if config.bounded_attention_channel_preserving_semantic_utility_gate
                else (config.bounded_attention_layers,)
            )
            self.bounded_attention_semantic_utility_bias = nn.Parameter(
                torch.zeros(semantic_bias_shape)
            )
        else:
            self.register_parameter("bounded_attention_semantic_utility_source", None)
            self.register_parameter("bounded_attention_semantic_utility_read", None)
            self.register_parameter("bounded_attention_semantic_utility_bias", None)
        self._bounded_attention_causal_exploration_step: int | None = None
        if config.ternary_tpc_rtrl_elementwise_transition:
            self.register_parameter("ffn_in", None)
            self.register_parameter("ffn_out", None)
        elif config.contractive_ternary_reciprocal_butterfly_ffn:
            self.ffn_in = nn.Parameter(
                torch.empty(depth, width.bit_length() - 1, width // 2, 2)
            )
            self.ffn_out = self.ffn_in
        elif config.intrinsic_ternary_reciprocal_ffn:
            self.ffn_in = nn.Parameter(
                torch.empty(depth, transition_width, ffn_input_width)
            )
            self.ffn_out = self.ffn_in
        else:
            self.ffn_in = nn.Parameter(
                torch.empty(
                    transition_owner_count,
                    transition_width,
                    ffn_input_width,
                )
            )
            self.ffn_out = nn.Parameter(
                torch.empty(
                    transition_owner_count,
                    (
                        ffn_width - width
                        if lexical_coordinate_carrier
                        else 2 * ffn_width
                        if config.sign_lifted_dual_ternary_transition
                        else ffn_width
                    ),
                    (
                        width
                        if lexical_coordinate_carrier
                        else transition_width
                    ),
                )
            )
        self.register_buffer(
            "lifted_antisymmetric_ffn_reverse",
            torch.empty(depth, transition_width, ffn_width)
            if config.lifted_antisymmetric_reciprocal_ffn
            else None,
        )
        if masked_lexical_coordinate_carrier:
            self.lexical_transition_participation = nn.Parameter(
                torch.zeros(depth, width)
            )
        else:
            self.register_parameter("lexical_transition_participation", None)
        if config.ternary_dale_quadrant_transition:
            self.ffn_out_dale_other = nn.Parameter(
                torch.empty(3, depth, ffn_width, transition_width)
            )
        else:
            self.register_parameter("ffn_out_dale_other", None)
        if config.source_separated_transition_compartments:
            self.consequence_ffn_in = nn.Parameter(
                torch.empty(depth, transition_width, ffn_input_width)
            )
            self.consequence_ffn_out = nn.Parameter(
                torch.zeros(depth, ffn_width, transition_width)
            )
        else:
            self.register_parameter("consequence_ffn_in", None)
            self.register_parameter("consequence_ffn_out", None)
        if config.shared_features_routed_ternary_ffn_output:
            self.ffn_routed_out = nn.Parameter(
                torch.empty(3, depth, ffn_width, transition_width)
            )
        else:
            self.register_parameter("ffn_routed_out", None)
        if config.deterministic_ternary_ffn_experts:
            extra_experts = 3
            self.ffn_expert_in = nn.Parameter(
                torch.empty(
                    extra_experts,
                    depth,
                    transition_width,
                    ffn_input_width,
                )
            )
            self.ffn_expert_out = nn.Parameter(
                torch.empty(
                    extra_experts,
                    depth,
                    ffn_width,
                    transition_width,
                )
            )
        else:
            self.register_parameter("ffn_expert_in", None)
            self.register_parameter("ffn_expert_out", None)
        if config.deterministic_ternary_ffn_residual_experts:
            residual_width = ffn_width // 4
            residual_input_width = 2 * residual_width
            self.ffn_residual_expert_in = nn.Parameter(
                torch.empty(4, depth, transition_width, residual_input_width)
            )
            self.ffn_residual_expert_out = nn.Parameter(
                torch.empty(4, depth, residual_width, transition_width)
            )
        else:
            self.register_parameter("ffn_residual_expert_in", None)
            self.register_parameter("ffn_residual_expert_out", None)
        if config.selected_rank_one_ternary_residuals:
            rank_one_shape = (4, depth, transition_width)
            self.rank_one_residual_source = nn.Parameter(torch.empty(rank_one_shape))
            self.rank_one_residual_output = nn.Parameter(torch.empty(rank_one_shape))
        else:
            self.register_parameter("rank_one_residual_source", None)
            self.register_parameter("rank_one_residual_output", None)
        if config.selected_diagonal_ternary_residuals:
            self.diagonal_residual = nn.Parameter(
                torch.empty(4, depth, transition_width)
            )
        else:
            self.register_parameter("diagonal_residual", None)
        if config.selected_diagonal_swiglu_residuals:
            diagonal_swiglu_shape = (4, depth, transition_width)
            self.diagonal_swiglu_gate = nn.Parameter(
                torch.empty(diagonal_swiglu_shape)
            )
            self.diagonal_swiglu_value = nn.Parameter(
                torch.empty(diagonal_swiglu_shape)
            )
            self.diagonal_swiglu_output = nn.Parameter(
                torch.empty(diagonal_swiglu_shape)
            )
        else:
            self.register_parameter("diagonal_swiglu_gate", None)
            self.register_parameter("diagonal_swiglu_value", None)
            self.register_parameter("diagonal_swiglu_output", None)
        if config.routed_ternary_ffn_channel_allocation:
            self.ffn_channel_allocation = nn.Parameter(
                torch.empty(4, depth, ffn_width)
            )
        else:
            self.register_parameter("ffn_channel_allocation", None)
        if config.self_gated_ternary_hidden_allocation:
            self.self_gated_hidden_allocation = nn.Parameter(
                torch.empty(transition_owner_count, ffn_width)
            )
        else:
            self.register_parameter("self_gated_hidden_allocation", None)
        if config.stable_latent_expert_branch:
            from pfc.phase493_stable_latent_experts import (
                STABLE_LATENT_EXPERTS,
                STABLE_LATENT_EXPERT_HIDDEN,
                STABLE_LATENT_WIDTH,
            )

            self.stable_latent_down = nn.Parameter(
                torch.empty(width, STABLE_LATENT_WIDTH)
            )
            self.stable_latent_router = nn.Parameter(
                torch.empty(width, STABLE_LATENT_EXPERTS)
            )
            expert_shape = (
                STABLE_LATENT_EXPERTS,
                STABLE_LATENT_WIDTH,
                STABLE_LATENT_EXPERT_HIDDEN,
            )
            self.stable_latent_expert_gate = nn.Parameter(torch.empty(expert_shape))
            self.stable_latent_expert_up = nn.Parameter(torch.empty(expert_shape))
            self.stable_latent_expert_down = nn.Parameter(torch.empty(expert_shape))
            self.stable_latent_norm = nn.Parameter(torch.ones(STABLE_LATENT_WIDTH))
            self.stable_latent_output = nn.Parameter(
                torch.zeros(STABLE_LATENT_WIDTH, width)
            )
            self.register_buffer(
                "stable_latent_router_bias",
                torch.zeros(STABLE_LATENT_EXPERTS, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.register_parameter("stable_latent_down", None)
            self.register_parameter("stable_latent_router", None)
            self.register_parameter("stable_latent_expert_gate", None)
            self.register_parameter("stable_latent_expert_up", None)
            self.register_parameter("stable_latent_expert_down", None)
            self.register_parameter("stable_latent_norm", None)
            self.register_parameter("stable_latent_output", None)
            self.register_buffer(
                "stable_latent_router_bias",
                None,
                persistent=True,
            )
        self._stable_latent_last_router_scores: Tensor | None = None
        self._stable_latent_last_route_ids: Tensor | None = None

        self.state_norm = nn.Parameter(torch.ones(depth, 1, transition_width))
        self.drive_norm = nn.Parameter(torch.ones(depth, 1, width))
        self.output_norm = nn.Parameter(torch.ones(width))
        if (
            config.depth_wide_ternary_lexical_endpoint
            or config.feature_ready_depth_wide_lexical_residual
            or config.cozero_reciprocal_lexical_adapters
        ):
            self.depth_lexical_output_norms = nn.Parameter(
                torch.ones(config.depth - 1, width)
            )
        else:
            self.register_parameter("depth_lexical_output_norms", None)
        if config.state_conditioned_confidence_readout:
            self.confidence_readout_weights = nn.Parameter(torch.zeros(16))
            self.confidence_readout_bias = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("confidence_readout_weights", None)
            self.register_parameter("confidence_readout_bias", None)
        if config.ternary_tpc_rtrl_elementwise_transition:
            self.register_parameter("gate_bias", None)
            self.register_parameter("ffn_in_bias", None)
            self.register_parameter("ffn_out_bias", None)
        elif config.depth_stratified_temporal_hierarchy:
            depth_bias = torch.tensor(
                DEPTH_STRATIFIED_TEMPORAL_GATE_BIASES,
                dtype=torch.float32,
            ).reshape(depth, 1, 1)
            gate_bias = depth_bias.expand(depth, 1, transition_width).clone()
        else:
            gate_bias = torch.full((depth, 1, transition_width), 1.0)
        if not config.ternary_tpc_rtrl_elementwise_transition:
            self.gate_bias = nn.Parameter(gate_bias)
            self.ffn_in_bias = nn.Parameter(torch.zeros(depth, 1, ffn_input_width))
        if config.source_separated_transition_compartments:
            self.consequence_ffn_in_bias = nn.Parameter(
                torch.zeros(depth, 1, ffn_input_width)
            )
            self.consequence_ffn_out_bias = nn.Parameter(
                torch.zeros(depth, 1, transition_width)
            )
        else:
            self.register_parameter("consequence_ffn_in_bias", None)
            self.register_parameter("consequence_ffn_out_bias", None)
        if config.deterministic_ternary_ffn_experts:
            self.ffn_expert_in_bias = nn.Parameter(
                torch.zeros(3, depth, 1, ffn_input_width)
            )
        else:
            self.register_parameter("ffn_expert_in_bias", None)
        if config.deterministic_ternary_ffn_residual_experts:
            self.ffn_residual_expert_in_bias = nn.Parameter(
                torch.zeros(4, depth, 1, 2 * (ffn_width // 4))
            )
        else:
            self.register_parameter("ffn_residual_expert_in_bias", None)
        if not config.ternary_tpc_rtrl_elementwise_transition:
            self.ffn_out_bias = nn.Parameter(torch.zeros(depth, 1, transition_width))
        if config.learn_residual_stream_scales:
            self.residual_stream_scales = nn.Parameter(torch.empty(depth, 1, 1))
        else:
            self.register_parameter("residual_stream_scales", None)
        if config.parallel_readout:
            self.readout_scales = nn.Parameter(torch.empty(depth))
        else:
            self.register_parameter("readout_scales", None)
        if config.projected_readout:
            self.readout_projection = nn.Parameter(torch.empty(depth, width, width))
        else:
            self.register_parameter("readout_projection", None)
        if config.fixed_orthogonal_depth_readout:
            self.register_buffer(
                "fixed_orthogonal_depth_readout_signs",
                _bent_hadamard_sign_schedule(depth, width),
                persistent=True,
            )
        else:
            self.register_buffer(
                "fixed_orthogonal_depth_readout_signs",
                None,
                persistent=True,
            )
        if config.temporal_rotation_min_period > 0:
            periods = torch.logspace(
                math.log10(config.temporal_rotation_min_period),
                math.log10(config.temporal_rotation_max_period),
                width // 2,
            )
            angles = 2.0 * math.pi / periods
            rotation = torch.stack((angles.cos(), angles.sin()), dim=-1)
            self.register_buffer(
                "temporal_rotation",
                rotation.unsqueeze(0).expand(depth, -1, -1).clone(),
                persistent=False,
            )
        else:
            self.register_buffer("temporal_rotation", None, persistent=False)
        if config.selective_temporal_phase_rank > 0:
            rank = config.selective_temporal_phase_rank
            self.temporal_phase_down = nn.Parameter(torch.empty(depth, width, rank))
            self.temporal_phase_up = nn.Parameter(torch.zeros(depth, rank, width // 2))
        else:
            self.register_parameter("temporal_phase_down", None)
            self.register_parameter("temporal_phase_up", None)
        if config.visit_framed_settling:
            channels = torch.arange(width, dtype=torch.int64)
            pattern_count = max(1, math.ceil(math.log2(width)))
            visit_frames = torch.ones(
                config.settling_rounds,
                depth,
                width,
                dtype=torch.float32,
            )
            for visit in range(1, config.settling_rounds):
                for layer in range(depth):
                    pattern = ((visit - 1) * depth + layer) % pattern_count
                    stride = 1 << pattern
                    visit_frames[visit, layer] = torch.where(
                        (channels // stride).remainder(2) == 0,
                        1.0,
                        -1.0,
                    )
            self.register_buffer(
                "settling_visit_frames",
                visit_frames,
                persistent=False,
            )
        else:
            self.register_buffer(
                "settling_visit_frames",
                None,
                persistent=False,
            )
        if (
            config.associative_memory_rank > 0
            and config.associative_memory_addressing == "dense"
        ):
            row = torch.arange(width, dtype=torch.float32).view(width, 1)
            column = torch.arange(
                1,
                config.associative_memory_rank + 1,
                dtype=torch.float32,
            ).view(1, config.associative_memory_rank)
            projection = math.sqrt(2.0 / width) * torch.cos(
                math.pi * (row + 0.5) * column / width
            )
            self.register_buffer(
                "associative_key_projection",
                projection.unsqueeze(0)
                .expand(self._associative_layer_count(), -1, -1)
                .clone(),
            )
        else:
            self.register_buffer(
                "associative_key_projection",
                None,
                persistent=False,
            )
        if config.associative_memory_addressing == "token_plus_latent_hash":
            row = torch.arange(width, dtype=torch.float32).view(width, 1)
            column = torch.arange(
                1,
                config.associative_memory_latent_bits + 1,
                dtype=torch.float32,
            ).view(1, config.associative_memory_latent_bits)
            latent_projection = math.sqrt(2.0 / width) * torch.cos(
                math.pi * (row + 0.5) * column / width
            )
            self.register_buffer(
                "associative_latent_projection",
                latent_projection.unsqueeze(0)
                .expand(self._associative_layer_count(), -1, -1)
                .clone(),
                persistent=False,
            )
        else:
            self.register_buffer(
                "associative_latent_projection",
                None,
                persistent=False,
            )
        if config.associative_memory_addressing == "token_plus_latent_codebook":
            source_channel = torch.arange(
                width,
                dtype=torch.float32,
            ).view(width, 1)
            code_channel = torch.arange(
                1,
                config.associative_memory_codebook_size + 1,
                dtype=torch.float32,
            ).view(1, config.associative_memory_codebook_size)
            codebook = math.sqrt(2.0 / width) * torch.cos(
                math.pi * (source_channel + 0.5) * code_channel / width
            )
            codebook = F.normalize(codebook, dim=0)
            self.register_buffer(
                "associative_latent_codebook",
                codebook.unsqueeze(0)
                .expand(self._associative_layer_count() - 1, -1, -1)
                .clone(),
            )
            self.register_buffer(
                "associative_latent_codebook_counts",
                torch.zeros(
                    self._associative_layer_count() - 1,
                    config.associative_memory_codebook_size,
                ),
            )
        else:
            self.register_buffer(
                "associative_latent_codebook",
                None,
                persistent=False,
            )
            self.register_buffer(
                "associative_latent_codebook_counts",
                None,
                persistent=False,
            )
        if self._uses_semantic_count_sketch_backoff():
            channels = torch.arange(width, dtype=torch.int64)
            mixed = (channels + 1) * 2_654_435_761 + 1_013_904_223
            mixed = torch.bitwise_xor(mixed, torch.bitwise_right_shift(mixed, 16))
            semantic_hash_mask = torch.where(
                mixed.remainder(2) == 0,
                torch.ones_like(mixed, dtype=torch.int8),
                -torch.ones_like(mixed, dtype=torch.int8),
            )
            self.register_buffer(
                "associative_semantic_hash_mask",
                semantic_hash_mask,
                persistent=False,
            )

            source_channel = torch.arange(width, dtype=torch.float32).view(width, 1)
            memory_channel = torch.arange(
                SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
                dtype=torch.float32,
            ).view(1, SEMANTIC_COUNT_SKETCH_VALUE_WIDTH)
            basis_scale = torch.full(
                (1, SEMANTIC_COUNT_SKETCH_VALUE_WIDTH),
                math.sqrt(2.0 / width),
                dtype=torch.float32,
            )
            basis_scale[:, 0] = math.sqrt(1.0 / width)
            semantic_encoder = basis_scale * torch.cos(
                math.pi * (source_channel + 0.5) * memory_channel / width
            )
            decoder_source = torch.arange(
                SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
                dtype=torch.float32,
            ).view(SEMANTIC_COUNT_SKETCH_VALUE_WIDTH, 1)
            decoder_target = torch.arange(width, dtype=torch.float32).view(1, width)
            decoder_scale = basis_scale.reshape(
                SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
                1,
            ) * math.sqrt(width / SEMANTIC_COUNT_SKETCH_VALUE_WIDTH)
            semantic_decoder = decoder_scale * torch.cos(
                math.pi * (decoder_target + 0.5) * decoder_source / width
            )
            self.register_buffer(
                "associative_semantic_value_encoder",
                semantic_encoder,
                persistent=False,
            )
            self.register_buffer(
                "associative_semantic_value_decoder",
                semantic_decoder,
                persistent=False,
            )
            if self._uses_disjoint_timescale_semantic_channels():
                half_width = width // 2
                channel_source = torch.arange(
                    SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
                    dtype=torch.float32,
                ).view(SEMANTIC_COUNT_SKETCH_VALUE_WIDTH, 1)
                channel_target = torch.arange(
                    half_width,
                    dtype=torch.float32,
                ).view(1, half_width)
                channel_basis_scale = torch.full(
                    (SEMANTIC_COUNT_SKETCH_VALUE_WIDTH, 1),
                    math.sqrt(2.0 / half_width),
                    dtype=torch.float32,
                )
                channel_basis_scale[0] = math.sqrt(1.0 / half_width)
                channel_decoder = (
                    channel_basis_scale
                    * math.sqrt(width / SEMANTIC_COUNT_SKETCH_VALUE_WIDTH)
                    * torch.cos(
                        math.pi
                        * (channel_target + 0.5)
                        * channel_source
                        / half_width
                    )
                )
                self.register_buffer(
                    "associative_semantic_channel_decoders",
                    channel_decoder.unsqueeze(0).expand(2, -1, -1).clone(),
                    persistent=False,
                )
            else:
                self.register_buffer(
                    "associative_semantic_channel_decoders",
                    None,
                    persistent=False,
                )
        else:
            self.register_buffer(
                "associative_semantic_hash_mask",
                None,
                persistent=False,
            )
            self.register_buffer(
                "associative_semantic_value_encoder",
                None,
                persistent=False,
            )
            self.register_buffer(
                "associative_semantic_value_decoder",
                None,
                persistent=False,
            )
            self.register_buffer(
                "associative_semantic_channel_decoders",
                None,
                persistent=False,
            )
        value_width = self._associative_value_width()
        if (
            config.associative_memory_rank > 0
            and value_width < width
            and config.associative_memory_addressing != "block_delta"
        ):
            source_channel = torch.arange(
                width,
                dtype=torch.float32,
            ).view(width, 1)
            memory_channel = torch.arange(
                value_width,
                dtype=torch.float32,
            ).view(1, value_width)
            basis_scale = torch.full(
                (1, value_width),
                math.sqrt(2.0 / width),
                dtype=torch.float32,
            )
            basis_scale[:, 0] = math.sqrt(1.0 / width)
            encoder = basis_scale * torch.cos(
                math.pi * (source_channel + 0.5) * memory_channel / width
            )
            decoder_source = torch.arange(
                value_width,
                dtype=torch.float32,
            ).view(value_width, 1)
            decoder_target = torch.arange(
                width,
                dtype=torch.float32,
            ).view(1, width)
            decoder_scale = basis_scale.reshape(value_width, 1) * math.sqrt(
                width / value_width
            )
            decoder = decoder_scale * torch.cos(
                math.pi * (decoder_target + 0.5) * decoder_source / width
            )
            if config.associative_memory_addressing in MULTISCALE_COUNT_SKETCH_ADDRESSINGS:
                read_groups = (
                    4
                    if config.associative_memory_addressing
                    in DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS
                    else 2
                )
                decoder = torch.cat((decoder,) * read_groups, dim=0) / read_groups
            self.register_buffer("associative_value_encoder", encoder)
            self.register_buffer("associative_value_decoder", decoder)
        else:
            self.register_buffer(
                "associative_value_encoder",
                None,
                persistent=False,
            )
            self.register_buffer(
                "associative_value_decoder",
                None,
                persistent=False,
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        width = self.config.width
        nn.init.normal_(self.embedding.weight, mean=0.0, std=width**-0.5)
        if self.ternary_embedding is not None:
            nn.init.normal_(
                self.ternary_embedding.weight,
                mean=0.0,
                std=width**-0.5,
            )
        if self.decoder is not None:
            nn.init.normal_(self.decoder, mean=0.0, std=width**-0.5)
        if self.config.fixed_ternary_hadamard_lexical_predictor:
            if self.fixed_ternary_hadamard_lexical_signs is None:
                raise RuntimeError("fixed lexical Hadamard signs are missing")
            matched_predictor_draw = self.output_norm.new_empty(width, width)
            nn.init.normal_(
                matched_predictor_draw,
                mean=0.0,
                std=width**-0.5,
            )
            self.fixed_ternary_hadamard_lexical_signs.copy_(
                torch.where(
                    matched_predictor_draw[0] >= 0,
                    torch.ones_like(matched_predictor_draw[0]),
                    -torch.ones_like(matched_predictor_draw[0]),
                ).to(torch.int8)
            )
        else:
            nn.init.normal_(self.token_predictor, mean=0.0, std=width**-0.5)
        if self.top8_lexical_reranker_codes is not None:
            nn.init.zeros_(self.top8_lexical_reranker_codes)
        if self.shared_candidate_compatibility_vector is not None:
            nn.init.zeros_(self.shared_candidate_compatibility_vector)
        if self.token_observation_coupling_diagonals is not None:
            nn.init.zeros_(self.token_observation_coupling_diagonals)
        if self.projected_self_bilinear_observation_matrix is not None:
            nn.init.zeros_(self.projected_self_bilinear_observation_matrix)
        if self.depth_innovation_predictor is not None:
            nn.init.zeros_(self.depth_innovation_predictor)
        if self.anti_hebbian_lexical_butterfly_coefficients is not None:
            nn.init.zeros_(self.anti_hebbian_lexical_butterfly_coefficients)
        if self.depth_lexical_predictors is not None:
            nn.init.zeros_(self.depth_lexical_predictors)
        if self.depth_lexical_decoders is not None:
            if (
                self.config.feature_ready_depth_wide_lexical_residual
                or self.config.cozero_reciprocal_lexical_adapters
            ):
                nn.init.zeros_(self.depth_lexical_decoders)
            else:
                nn.init.normal_(
                    self.depth_lexical_decoders,
                    mean=0.0,
                    std=width**-0.5,
                )
        if self.swiglu_lexical_decoder is not None:
            nn.init.zeros_(self.swiglu_lexical_decoder)
        if self.context_routed_lexical_residual_decoders is not None:
            nn.init.zeros_(self.context_routed_lexical_residual_decoders)
        if self.bounded_multiplicative_lexical_decoder is not None:
            nn.init.zeros_(self.bounded_multiplicative_lexical_decoder)
        if self.signed_square_lexical_decoder is not None:
            nn.init.zeros_(self.signed_square_lexical_decoder)
        if self.cross_channel_quadratic_lexical_decoder is not None:
            nn.init.zeros_(self.cross_channel_quadratic_lexical_decoder)
        if self.shared_base_ternary_delta_decoder is not None:
            nn.init.zeros_(self.shared_base_ternary_delta_decoder)
        if self.diagonal_handoff_token_gate is not None:
            nn.init.zeros_(self.diagonal_handoff_token_gate)
        if self.ternary_recirculation_selector is not None:
            nn.init.zeros_(self.ternary_recirculation_selector)
        for controller in self.adaptive_recirculation_controllers(
            require_enabled=False
        ):
            nn.init.zeros_(controller)
        if self.causal_ternary_fir_input_taps is not None:
            nn.init.zeros_(self.causal_ternary_fir_input_taps)
        if self.latent_error_settling_gain is not None:
            nn.init.zeros_(self.latent_error_settling_gain)
        if self.full_bandwidth_value is not None:
            nn.init.orthogonal_(self.full_bandwidth_value)
        if self.full_bandwidth_gate is not None:
            nn.init.orthogonal_(self.full_bandwidth_gate)
        if self.predictive_future_error_key_projection is not None:
            nn.init.zeros_(self.predictive_future_error_key_projection)
        if self.predictive_read_gate_projection is not None:
            nn.init.zeros_(self.predictive_read_gate_projection)
        if self.predictive_exception_cache_gate is not None:
            nn.init.zeros_(self.predictive_exception_cache_gate)
        if self.config.ternary_tpc_rtrl_elementwise_transition:
            assert self.rglru_recurrent_parameter is not None
            generator = torch.Generator(device="cpu")
            generator.manual_seed(539)
            with torch.no_grad():
                self.rglru_recurrent_parameter.copy_(
                    initialize_recurrent_parameter(
                        self.config.depth * width,
                        generator=generator,
                        device="cpu",
                        dtype=self.rglru_recurrent_parameter.dtype,
                    )
                    .reshape(self.config.depth, width)
                    .to(self.rglru_recurrent_parameter.device)
                )
        if self.config.ordered_swiglu_rglru_carry:
            assert self.ordered_rglru_recurrent_parameter is not None
            generator = torch.Generator(device="cpu")
            generator.manual_seed(539)
            with torch.no_grad():
                self.ordered_rglru_recurrent_parameter.copy_(
                    initialize_recurrent_parameter(
                        self.config.depth * 2 * (width // 2),
                        generator=generator,
                        device="cpu",
                        dtype=self.ordered_rglru_recurrent_parameter.dtype,
                    )
                    .reshape(self.config.depth, 2, width // 2)
                    .to(self.ordered_rglru_recurrent_parameter.device)
                )
        for layer in range(self.config.depth):
            if self.config.ternary_tpc_rtrl_elementwise_transition:
                assert self.rglru_recurrent_gate_weight is not None
                assert self.rglru_input_gate_weight is not None
                assert self.rglru_recurrent_parameter is not None
                nn.init.normal_(
                    self.rglru_recurrent_gate_weight[layer],
                    mean=0.0,
                    std=width**-0.5,
                )
                nn.init.normal_(
                    self.rglru_input_gate_weight[layer],
                    mean=0.0,
                    std=width**-0.5,
                )
            else:
                assert self.temporal is not None
                assert self.gate is not None
                assert self.ffn_in is not None
                assert self.ffn_out is not None
                nn.init.normal_(self.temporal[layer], mean=0.0, std=width**-0.5)
            nn.init.normal_(self.generative[layer], mean=0.0, std=width**-0.5)
            nn.init.normal_(self.reciprocal[layer], mean=0.0, std=width**-0.5)
            nn.init.eye_(self.candidate[layer])
            if self.config.deeploop_postnorm_settling:
                with torch.no_grad():
                    self.candidate[layer].mul_(self.deeploop_initialization_gain())
            if not self.config.ternary_tpc_rtrl_elementwise_transition:
                nn.init.zeros_(self.gate[layer])
                if self.transition_write_gate is not None:
                    with torch.no_grad():
                        self.transition_write_gate[layer].copy_(-self.gate[layer])
                owner = self.transition_owner_index(layer)
                if owner == layer:
                    nn.init.kaiming_uniform_(self.ffn_in[owner], a=5**0.5)
                    if self.affine_reversible_scale is not None:
                        with torch.random.fork_rng(devices=[]):
                            nn.init.kaiming_uniform_(
                                self.affine_reversible_scale[owner],
                                a=5**0.5,
                            )
                if self.config.intrinsic_ternary_reciprocal_ffn:
                    with torch.no_grad():
                        upper = torch.triu(self.ffn_in[owner])
                        self.ffn_in[owner].copy_(
                            upper
                            + torch.triu(upper, diagonal=1).transpose(0, 1)
                        )
                elif not self.config.contractive_ternary_reciprocal_butterfly_ffn:
                    if owner == layer:
                        nn.init.zeros_(self.ffn_out[owner])
                    if self.ffn_out_dale_other is not None:
                        nn.init.zeros_(self.ffn_out_dale_other[:, layer])
                if self.lexical_transition_participation is not None:
                    if (
                        self.config.post_coinitialized_lexical_coordinate_transition_carrier
                    ):
                        nn.init.ones_(self.lexical_transition_participation[layer])
                    else:
                        nn.init.zeros_(self.lexical_transition_participation[layer])
            if self.config.ordered_swiglu_rglru_carry:
                assert self.ordered_rglru_recurrent_gate_weight is not None
                assert self.ordered_rglru_input_gate_weight is not None
                for branch in range(2):
                    nn.init.normal_(
                        self.ordered_rglru_recurrent_gate_weight[layer, branch],
                        mean=0.0,
                        std=(width // 2) ** -0.5,
                    )
                    nn.init.normal_(
                        self.ordered_rglru_input_gate_weight[layer, branch],
                        mean=0.0,
                        std=(width // 2) ** -0.5,
                    )
            if self.consequence_ffn_in is not None:
                assert self.consequence_ffn_out is not None
                assert self.consequence_ffn_in_bias is not None
                assert self.consequence_ffn_out_bias is not None
                with torch.no_grad():
                    self.consequence_ffn_in[layer].copy_(self.ffn_in[self.transition_owner_index(layer)])
                    self.consequence_ffn_in_bias[layer].copy_(
                        self.ffn_in_bias[layer]
                    )
                nn.init.zeros_(self.consequence_ffn_out[layer])
                nn.init.zeros_(self.consequence_ffn_out_bias[layer])
            if self.ffn_routed_out is not None:
                nn.init.zeros_(self.ffn_routed_out[:, layer])
            if self.ffn_expert_in is not None:
                if self.ffn_expert_out is None:
                    raise RuntimeError("deterministic FFN expert output is missing")
                for expert in range(self.ffn_expert_in.shape[0]):
                    nn.init.kaiming_uniform_(
                        self.ffn_expert_in[expert, layer],
                        a=5**0.5,
                    )
                    nn.init.zeros_(self.ffn_expert_out[expert, layer])
            if self.ffn_residual_expert_in is not None:
                if self.ffn_residual_expert_out is None:
                    raise RuntimeError("residual FFN expert output is missing")
                for expert in range(self.ffn_residual_expert_in.shape[0]):
                    nn.init.kaiming_uniform_(
                        self.ffn_residual_expert_in[expert, layer],
                        a=5**0.5,
                    )
                    nn.init.zeros_(self.ffn_residual_expert_out[expert, layer])
            if self.rank_one_residual_source is not None:
                if self.rank_one_residual_output is None:
                    raise RuntimeError("rank-one residual output is missing")
                for route in range(self.rank_one_residual_source.shape[0]):
                    nn.init.normal_(
                        self.rank_one_residual_source[route, layer],
                        mean=0.0,
                        std=width**-0.5,
                    )
                    nn.init.zeros_(self.rank_one_residual_output[route, layer])
            if self.diagonal_residual is not None:
                nn.init.zeros_(self.diagonal_residual[:, layer])
            if self.diagonal_swiglu_gate is not None:
                if (
                    self.diagonal_swiglu_value is None
                    or self.diagonal_swiglu_output is None
                ):
                    raise RuntimeError("diagonal SwiGLU vectors are incomplete")
                nn.init.normal_(
                    self.diagonal_swiglu_gate[:, layer],
                    mean=0.0,
                    std=width**-0.5,
                )
                nn.init.normal_(
                    self.diagonal_swiglu_value[:, layer],
                    mean=0.0,
                    std=width**-0.5,
                )
                nn.init.zeros_(self.diagonal_swiglu_output[:, layer])
            if self.ffn_channel_allocation is not None:
                nn.init.zeros_(self.ffn_channel_allocation[:, layer])
        if self.self_gated_hidden_allocation is not None:
            nn.init.zeros_(self.self_gated_hidden_allocation)
        if self.transition_write_gate_bias is not None:
            with torch.no_grad():
                self.transition_write_gate_bias.copy_(-self.gate_bias)
        if self.pairwise_delta_down is not None:
            if self.pairwise_delta_key_up is None:
                raise RuntimeError("pairwise delta key projection is missing")
            for layer in range(self.config.depth):
                nn.init.orthogonal_(self.pairwise_delta_down[layer])
                nn.init.normal_(
                    self.pairwise_delta_key_up[layer],
                    mean=0.0,
                    std=self.config.pairwise_delta_erase_rank**-0.5,
                )
        if self.block_delta_down is not None:
            if self.block_delta_query_up is None or self.block_delta_key_up is None:
                raise RuntimeError("block delta content projections are missing")
            for layer in range(self._associative_layer_count()):
                nn.init.orthogonal_(self.block_delta_down[layer])
                nn.init.normal_(
                    self.block_delta_query_up[layer],
                    mean=0.0,
                    std=self.config.block_delta_memory_projection_rank**-0.5,
                )
                nn.init.normal_(
                    self.block_delta_key_up[layer],
                    mean=0.0,
                    std=self.config.block_delta_memory_projection_rank**-0.5,
                )
        if self.gdn2_query is not None:
            gdn2_parameters = (
                self.gdn2_query,
                self.gdn2_key,
                self.gdn2_value,
                self.gdn2_decay_down,
                self.gdn2_decay_up,
                self.gdn2_erase,
                self.gdn2_write,
                self.gdn2_output_gate_down,
                self.gdn2_output_gate_up,
            )
            if any(parameter is None for parameter in gdn2_parameters):
                raise RuntimeError("GDN2 mixer projections are incomplete")
            for parameter in gdn2_parameters:
                assert parameter is not None
                for layer in range(self.config.gdn2_mixer_layers):
                    nn.init.xavier_uniform_(parameter[layer], gain=2**-2.5)
            if self.gdn2_decay_log_rate is None or self.gdn2_decay_bias is None:
                raise RuntimeError("GDN2 mixer decay parameters are missing")
            with torch.no_grad():
                self.gdn2_decay_log_rate.uniform_(1.0, 16.0).log_()
                dt = (
                    torch.empty_like(self.gdn2_decay_bias)
                    .uniform_(math.log(0.001), math.log(0.1))
                    .exp_()
                    .clamp_min_(1e-4)
                )
                self.gdn2_decay_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        if self.factorization_affinity is not None:
            factorization_parameters = (
                self.factorization_update,
                self.factorization_merge,
                self.factorization_value_encoder,
            )
            if any(parameter is None for parameter in factorization_parameters):
                raise RuntimeError("factorization memory parameters are incomplete")
            nn.init.normal_(
                self.factorization_affinity,
                mean=0.0,
                std=width**-0.5,
            )
            assert self.factorization_update is not None
            assert self.factorization_merge is not None
            assert self.factorization_value_encoder is not None
            nn.init.normal_(self.factorization_update, mean=0.0, std=width**-0.5)
            nn.init.normal_(self.factorization_merge, mean=0.0, std=width**-0.5)
            nn.init.normal_(
                self.factorization_value_encoder,
                mean=0.0,
                std=width**-0.5,
            )
            if (
                self.factorization_update_bias is None
                or self.factorization_merge_bias is None
                or self.factorization_value_decoder is None
            ):
                raise RuntimeError("factorization memory output parameters are missing")
            nn.init.zeros_(self.factorization_update_bias)
            nn.init.zeros_(self.factorization_merge_bias)
            nn.init.zeros_(self.factorization_value_decoder)
        if self.bounded_attention_query is not None:
            attention_parameters = (
                self.bounded_attention_key,
                self.bounded_attention_value,
                self.bounded_attention_output,
            )
            if any(parameter is None for parameter in attention_parameters):
                raise RuntimeError("bounded attention parameters are incomplete")
            nn.init.normal_(
                self.bounded_attention_query,
                mean=0.0,
                std=width**-0.5,
            )
            assert self.bounded_attention_key is not None
            assert self.bounded_attention_value is not None
            assert self.bounded_attention_output is not None
            nn.init.normal_(
                self.bounded_attention_key,
                mean=0.0,
                std=width**-0.5,
            )
            nn.init.normal_(
                self.bounded_attention_value,
                mean=0.0,
                std=width**-0.5,
            )
            nn.init.zeros_(self.bounded_attention_output)
        if self.bounded_attention_semantic_utility_source is not None:
            nn.init.normal_(
                self.bounded_attention_semantic_utility_source,
                mean=0.0,
                std=width**-0.5,
            )
            assert self.bounded_attention_semantic_utility_read is not None
            assert self.bounded_attention_semantic_utility_bias is not None
            nn.init.zeros_(self.bounded_attention_semantic_utility_read)
            nn.init.zeros_(self.bounded_attention_semantic_utility_bias)
        if self.later_candidates is not None:
            for visit in range(self.later_candidates.shape[0]):
                for layer in range(self.config.depth):
                    nn.init.eye_(self.later_candidates[visit, layer])
        if self.later_candidate_adapter_down is not None:
            for visit in range(self.later_candidate_adapter_down.shape[0]):
                for layer in range(self.config.depth):
                    nn.init.orthogonal_(self.later_candidate_adapter_down[visit, layer])
        if self.later_dual_stream_adapter_down is not None:
            for visit in range(self.later_dual_stream_adapter_down.shape[0]):
                for layer in range(self.config.depth):
                    for pathway in range(4):
                        nn.init.orthogonal_(
                            self.later_dual_stream_adapter_down[
                                visit,
                                layer,
                                pathway,
                            ]
                        )
                    with torch.no_grad():
                        down = self.later_dual_stream_adapter_down[visit, layer]
                        down.abs_()
                        down[0].mul_(1.5)
                        down[2].mul_(1.5)
                        down[1].mul_(0.5)
                        down[3].mul_(0.5)
        if self.persistent_transition_adapter_down is not None:
            for layer in range(self.config.depth):
                nn.init.orthogonal_(self.persistent_transition_adapter_down[layer])
            if self.persistent_transition_adapter_decay_logit is None:
                raise RuntimeError("persistent transition decay is missing")
            with torch.no_grad():
                self.persistent_transition_adapter_decay_logit.fill_(
                    math.log(0.99 / 0.01)
                )
        if self.temporal_phase_down is not None:
            for layer in range(self.config.depth):
                nn.init.orthogonal_(self.temporal_phase_down[layer])
        if self.residual_stream_scales is not None:
            with torch.no_grad():
                self.residual_stream_scales.copy_(
                    torch.linspace(
                        self.config.residual_stream_scale_start,
                        self.config.residual_stream_scale_end,
                        self.config.depth,
                        device=self.residual_stream_scales.device,
                        dtype=self.residual_stream_scales.dtype,
                    ).view(self.config.depth, 1, 1)
                )
        if self.readout_scales is not None:
            with torch.no_grad():
                if self.config.parallel_readout_init == "bottom":
                    self.readout_scales.zero_()
                    self.readout_scales[0] = 1.0
                else:
                    self.readout_scales.fill_(1.0 / self.config.depth)
        if self.readout_projection is not None:
            with torch.no_grad():
                self.readout_projection.zero_()
                if not self.config.hybrid_projected_readout:
                    self.readout_projection[0].copy_(
                        torch.eye(
                            self.config.width,
                            device=self.readout_projection.device,
                            dtype=self.readout_projection.dtype,
                        )
                    )
        if self.confidence_readout_weights is not None:
            with torch.no_grad():
                self.confidence_readout_weights.zero_()
                assert self.confidence_readout_bias is not None
                self.confidence_readout_bias.zero_()
        if self.manifold_depth_bus_controller is not None:
            nn.init.zeros_(self.manifold_depth_bus_controller)
            controller_std = 0.01 / math.sqrt(2 * width)
            for output in (0, 1, 4):
                nn.init.normal_(
                    self.manifold_depth_bus_controller[..., output],
                    mean=0.0,
                    std=controller_std,
                )
            if self.manifold_depth_bus_bias is None:
                raise RuntimeError("bounded manifold depth-bus bias is missing")
            nn.init.zeros_(self.manifold_depth_bus_bias)
        if self.depth_attention_queries is not None:
            depth_attention_generator = torch.Generator(
                device=self.depth_attention_queries.device
            ).manual_seed(
                (torch.initial_seed() ^ 0x4B334450) & ((1 << 63) - 1)
            )
            nn.init.normal_(
                self.depth_attention_queries,
                mean=0.0,
                std=width**-0.5,
                generator=depth_attention_generator,
            )
            if self.depth_attention_residual_logits is None:
                raise RuntimeError("depth-attention residual logits are missing")
            nn.init.zeros_(self.depth_attention_residual_logits)
        if self.stable_latent_down is not None:
            from pfc.phase493_stable_latent_experts import STABLE_LATENT_WIDTH

            stable_latent_generator = torch.Generator(
                device=self.stable_latent_down.device
            ).manual_seed(
                (torch.initial_seed() ^ 0x4B334C4D) & ((1 << 63) - 1)
            )
            nn.init.normal_(
                self.stable_latent_down,
                mean=0.0,
                std=width**-0.5,
                generator=stable_latent_generator,
            )
            assert self.stable_latent_router is not None
            nn.init.normal_(
                self.stable_latent_router,
                mean=0.0,
                std=width**-0.5,
                generator=stable_latent_generator,
            )
            for parameter in (
                self.stable_latent_expert_gate,
                self.stable_latent_expert_up,
                self.stable_latent_expert_down,
            ):
                if parameter is None:
                    raise RuntimeError("stable latent expert parameters are incomplete")
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=STABLE_LATENT_WIDTH**-0.5,
                    generator=stable_latent_generator,
                )
            if (
                self.stable_latent_norm is None
                or self.stable_latent_output is None
                or self.stable_latent_router_bias is None
            ):
                raise RuntimeError("stable latent expert outputs are incomplete")
            nn.init.ones_(self.stable_latent_norm)
            nn.init.zeros_(self.stable_latent_output)
            self.stable_latent_router_bias.zero_()
        if self.nonlinear_matrix_input_projection is not None:
            nn.init.normal_(
                self.nonlinear_matrix_input_projection[
                    :, : 2 * NONLINEAR_MATRIX_KEY_WIDTH
                ],
                mean=0.0,
                std=width**-0.5,
            )
            nn.init.zeros_(
                self.nonlinear_matrix_input_projection[
                    :, 2 * NONLINEAR_MATRIX_KEY_WIDTH :
                ]
            )
            if (
                self.nonlinear_matrix_transition is None
                or self.nonlinear_matrix_forget_bias is None
                or self.nonlinear_matrix_residual_weight is None
                or self.nonlinear_matrix_output_norm is None
                or self.nonlinear_matrix_output_scale is None
            ):
                raise RuntimeError("nonlinear matrix parameters are incomplete")
            with torch.no_grad():
                self.nonlinear_matrix_transition.copy_(
                    torch.eye(
                        NONLINEAR_MATRIX_VALUE_WIDTH,
                        device=self.nonlinear_matrix_transition.device,
                        dtype=self.nonlinear_matrix_transition.dtype,
                    ).expand_as(self.nonlinear_matrix_transition)
                )
                periods = torch.logspace(
                    math.log10(4.0),
                    math.log10(2048.0),
                    NONLINEAR_MATRIX_HEADS,
                    device=self.nonlinear_matrix_forget_bias.device,
                    dtype=self.nonlinear_matrix_forget_bias.dtype,
                )
                retention = torch.exp(-1.0 / periods)
                self.nonlinear_matrix_forget_bias.copy_(
                    torch.log((1.0 - retention) / retention)
                )
                self.nonlinear_matrix_residual_weight.fill_(1.0)
                self.nonlinear_matrix_output_norm.fill_(1.0)
                self.nonlinear_matrix_output_scale.zero_()
        if self.kronecker_input_projection is not None:
            generator = torch.Generator(
                device=self.kronecker_input_projection.device
            ).manual_seed(539)
            nn.init.normal_(
                self.kronecker_input_projection,
                mean=0.0,
                std=KRONECKER_CODEC_DIMENSION**-0.5,
                generator=generator,
            )
            self.refresh_kronecker_projection_cache()
        if self.bpe_ancestry_input_table is not None:
            nn.init.zeros_(self.bpe_ancestry_input_table)
        self.refresh_metric_shared_endpoint_cache()
        self.refresh_lifted_antisymmetric_reciprocal_cache()

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embedding.weight.dtype

    @property
    def transition_owner_count(self) -> int:
        return (
            1
            if self.config.depth_tied_reciprocal_transition_core
            else self.config.depth
        )

    def transition_owner_index(self, layer: int) -> int:
        if not 0 <= layer < self.config.depth:
            raise ValueError("transition layer is out of range")
        return 0 if self.config.depth_tied_reciprocal_transition_core else layer

    def decoder_matrix(self) -> Tensor:
        if self.decoder is not None:
            return self.decoder
        if not self.config.shared_embedding_decoder:
            raise RuntimeError("decoder parameter is missing")
        if self.ternary_embedding is not None:
            return torch.cat(
                (
                    self.embedding.weight.transpose(0, 1),
                    self.ternary_embedding.weight.transpose(0, 1),
                ),
                dim=0,
            )
        return self.embedding.weight.transpose(0, 1)

    @property
    def kronecker_codec_sha256(self) -> str | None:
        return self._kronecker_codec_sha256

    @property
    def bpe_ancestry_artifact_sha256(self) -> str | None:
        return self._bpe_ancestry_artifact_sha256

    @property
    def bpe_ancestry_buffer_sha256(self) -> str | None:
        return self._bpe_ancestry_buffer_sha256

    @torch.no_grad()
    def install_kronecker_input_codec(
        self,
        byte_buffer: Tensor,
        length_buffer: Tensor,
    ) -> None:
        if not self.config.kronecker_input_embedding:
            raise RuntimeError("Kronecker input is disabled")
        validate_kronecker_buffers(
            byte_buffer,
            length_buffer,
            vocab_size=self.config.vocab_size,
        )
        actual_sha256 = kronecker_buffers_sha256(byte_buffer, length_buffer)
        if actual_sha256 != self.config.kronecker_input_codec_sha256:
            raise ValueError("Kronecker codec SHA-256 differs")
        self._kronecker_byte_buffer = byte_buffer.detach().to(
            device=self.device,
            dtype=torch.uint8,
        )
        self._kronecker_length_buffer = length_buffer.detach().to(
            device=self.device,
            dtype=torch.int16,
        )
        self._kronecker_codec_sha256 = actual_sha256

    @torch.no_grad()
    def install_bpe_ancestry_input(
        self,
        ancestry: CanonicalBPEAncestry,
    ) -> None:
        if not self.config.bpe_ancestry_input_embedding:
            raise RuntimeError("BPE ancestry input is disabled")
        if ancestry.artifact_sha256 != self.config.bpe_ancestry_input_artifact_sha256:
            raise ValueError("BPE ancestry artifact SHA-256 differs")
        validate_ancestry_buffers(
            ancestry.indices,
            ancestry.counts,
            ancestry.inverse_sqrt_counts,
            vocab_size=self.config.vocab_size,
        )
        self._bpe_ancestry_indices = ancestry.indices.detach().to(
            device=self.device,
            dtype=torch.int32,
        ).contiguous()
        self._bpe_ancestry_counts = ancestry.counts.detach().to(
            device=self.device,
            dtype=torch.int16,
        ).contiguous()
        self._bpe_ancestry_inverse_sqrt_counts = (
            ancestry.inverse_sqrt_counts.detach().to(
                device=self.device,
                dtype=torch.float32,
            ).contiguous()
        )
        self._bpe_ancestry_artifact_sha256 = ancestry.artifact_sha256
        self._bpe_ancestry_buffer_sha256 = ancestry.buffer_sha256
        self._bpe_ancestry_parents_cpu = ancestry.parents.detach().to(
            device="cpu",
            dtype=torch.int32,
        ).contiguous()

    @torch.no_grad()
    def initialize_bpe_ancestry_input_from_embedding(self) -> None:
        table = self.bpe_ancestry_input_table
        parents = self._bpe_ancestry_parents_cpu
        counts = self._bpe_ancestry_counts
        if not self.config.bpe_ancestry_input_embedding:
            if table is not None:
                raise RuntimeError("disabled BPE ancestry input has a table")
            return
        if (
            table is None
            or parents is None
            or counts is None
            or counts.shape != (self.config.vocab_size,)
            or self._bpe_ancestry_artifact_sha256
            != self.config.bpe_ancestry_input_artifact_sha256
        ):
            raise RuntimeError("install BPE ancestry before initialization")
        table.copy_(
            mobius_innovations(
                self.embedding.weight.detach(),
                parents.to(device=self.device),
                counts,
            )
        )
        self._bpe_ancestry_parents_cpu = None

    @torch.no_grad()
    def refresh_kronecker_projection_cache(self) -> None:
        projection = self.kronecker_input_projection
        row_sum = self._kronecker_projection_row_sum
        if projection is None:
            return
        if row_sum is None or row_sum.shape != (self.config.width,):
            raise RuntimeError("Kronecker projection row-sum cache is missing")
        row_sum.copy_(projection.detach().float().sum(dim=0))

    @torch.no_grad()
    def refresh_metric_shared_endpoint_cache(self) -> None:
        bias = self._metric_shared_row_norm_bias
        if not self.config.metric_consistent_shared_endpoint:
            if bias is not None:
                raise RuntimeError("disabled metric shared endpoint has a bias cache")
            return
        if bias is None or bias.shape != (self.config.vocab_size,):
            raise RuntimeError("metric shared endpoint bias cache is missing")
        from pfc.phase734_metric_shared_endpoint import metric_shared_row_norm_bias

        bias.copy_(metric_shared_row_norm_bias(self.embedding.weight.detach().float()))

    @torch.no_grad()
    def refresh_lifted_antisymmetric_reciprocal_cache(self) -> None:
        reverse = self.lifted_antisymmetric_ffn_reverse
        if not self.config.lifted_antisymmetric_reciprocal_ffn:
            if reverse is not None:
                raise RuntimeError(
                    "disabled lifted antisymmetric transition has a reverse block"
                )
            return
        if self.ffn_out is None or reverse is None:
            raise RuntimeError("lifted antisymmetric transition blocks are missing")
        expected = (
            self.config.depth,
            self.ffn_out.shape[-1],
            self.ffn_out.shape[-2],
        )
        if tuple(reverse.shape) != expected:
            raise RuntimeError("lifted antisymmetric reverse-block shape differs")
        from pfc.phase735_lifted_antisymmetric_transition import (
            refresh_antisymmetric_reverse_,
        )

        refresh_antisymmetric_reverse_(self.ffn_out, reverse)

    def embed_tokens(self, token_ids: Tensor) -> Tensor:
        if self.config.bpe_ancestry_input_embedding:
            if (
                self.bpe_ancestry_input_table is None
                or self._bpe_ancestry_indices is None
                or self._bpe_ancestry_counts is None
                or self._bpe_ancestry_inverse_sqrt_counts is None
                or self._bpe_ancestry_artifact_sha256
                != self.config.bpe_ancestry_input_artifact_sha256
            ):
                raise RuntimeError("install the pinned BPE ancestry before use")
            ancestry_sum = ancestry_fixed_order_sum
            if self.config.bpe_ancestry_fused_execution:
                from pfc.phase750_fused_bpe_ancestry_input import (
                    fused_ancestry_fixed_order_sum,
                )

                ancestry_sum = fused_ancestry_fixed_order_sum
            return ancestry_sum(
                token_ids,
                self.bpe_ancestry_input_table,
                self._bpe_ancestry_indices,
                self._bpe_ancestry_counts,
                self._bpe_ancestry_inverse_sqrt_counts,
            )
        if self.config.kronecker_input_embedding:
            if (
                self.kronecker_input_projection is None
                or self._kronecker_byte_buffer is None
                or self._kronecker_length_buffer is None
                or self._kronecker_projection_row_sum is None
                or self._kronecker_codec_sha256
                != self.config.kronecker_input_codec_sha256
            ):
                raise RuntimeError("install the pinned Kronecker codec before use")
            return sparse_kronecker_project(
                token_ids,
                self._kronecker_byte_buffer,
                self._kronecker_length_buffer,
                self.kronecker_input_projection,
                self._kronecker_projection_row_sum,
            )
        dense = self.embedding(token_ids)
        if self.config.row_energy_preserving_input_compander:
            from pfc.phase818_row_energy_preserving_input_compander import (
                row_energy_preserving_input_compander,
            )

            dense = row_energy_preserving_input_compander(
                dense,
                rms_eps=self.config.rms_eps,
            )
        if self.config.energy_preserving_token_fingerprint_observation:
            from pfc.phase819_energy_preserving_token_fingerprint_observation import (
                energy_preserving_token_fingerprint_observation,
                unpack_token_fingerprints,
            )

            packed_codes = self.energy_preserving_token_fingerprint_codes
            if packed_codes is None:
                raise RuntimeError("packed token fingerprints are missing")
            fingerprints = unpack_token_fingerprints(
                packed_codes,
                token_ids,
                dtype=dense.dtype,
            )
            dense = energy_preserving_token_fingerprint_observation(
                dense,
                fingerprints,
                rms_eps=self.config.rms_eps,
            )
        if self.config.self_dual_hadamard_token_observation_coupling:
            from pfc.phase822_self_dual_hadamard_token_observation_coupling import (
                deployment_token_observation_coupling_diagonals,
                self_dual_hadamard_token_observation_coupling,
            )

            packed_runtime = getattr(
                self,
                "_phase822_packed_token_observation_coupling",
                None,
            )
            diagonals = (
                packed_runtime.diagonals(dtype=dense.dtype)
                if packed_runtime is not None
                else deployment_token_observation_coupling_diagonals(
                    self.token_observation_coupling_diagonals
                )
            )
            signs = self.fixed_ternary_hadamard_lexical_signs
            if diagonals is None or signs is None:
                raise RuntimeError("self-dual token observation coupling is incomplete")
            dense = self_dual_hadamard_token_observation_coupling(
                dense,
                diagonals,
                signs,
            )
        if self.config.energy_preserving_self_bilinear_token_observation:
            from pfc.phase823_energy_preserving_self_bilinear_token_observation import (
                energy_preserving_self_bilinear_token_observation,
            )

            dense = energy_preserving_self_bilinear_token_observation(
                dense,
                rms_eps=self.config.rms_eps,
            )
        if self.config.projected_self_bilinear_token_observation:
            from pfc.phase827_projected_self_bilinear_observation import (
                projected_self_bilinear_token_observation,
            )

            packed_runtime = getattr(
                self,
                "_phase827_packed_projected_self_bilinear_observation",
                None,
            )
            if packed_runtime is not None:
                dense = packed_runtime.observe(
                    dense,
                    rms_eps=self.config.rms_eps,
                )
            else:
                projector = self.projected_self_bilinear_observation_matrix
                if projector is None:
                    raise RuntimeError(
                        "projected self-bilinear observation is incomplete"
                    )
                dense = projected_self_bilinear_token_observation(
                    dense,
                    projector,
                    rms_eps=self.config.rms_eps,
                ).output
        if self.config.self_gated_isometric_sign_token_observation:
            from pfc.phase824_self_gated_isometric_sign_coupling import (
                self_gated_isometric_sign_token_observation,
            )

            dense = self_gated_isometric_sign_token_observation(dense)
        if self.ternary_embedding is None:
            return dense
        return torch.cat((dense, self.ternary_embedding(token_ids)), dim=-1)

    def _token_gated_handoff(self, observation: Tensor, source: Tensor):
        if not self.config.token_gated_diagonal_handoff:
            raise RuntimeError("token-gated handoff is disabled")
        packed = getattr(self, "_packed_grouped_ternary_handoff_gate", None)
        if packed is not None:
            logits = packed.matmul(observation, matrix_index=0)
        else:
            if self.diagonal_handoff_token_gate is None:
                raise RuntimeError("token-gated handoff matrix is missing")
            logits = torch.matmul(observation, self.diagonal_handoff_token_gate[0])
        from pfc.phase702_token_gated_handoff import token_gated_handoff

        return token_gated_handoff(source, logits)

    def _causal_ternary_fir_taps(self, dtype: torch.dtype) -> Tensor:
        if not self.config.causal_ternary_fir_input_state:
            raise RuntimeError("causal FIR input state is disabled")
        packed = getattr(self, "_packed_causal_ternary_fir_input_taps", None)
        if packed is not None:
            from pfc.packed_grouped_ternary import dense_weight_from_grouped_ternary

            return dense_weight_from_grouped_ternary(packed, dtype=dtype)
        if self.causal_ternary_fir_input_taps is None:
            raise RuntimeError("causal FIR input taps are missing")
        return self.causal_ternary_fir_input_taps

    def _latent_error_settling_gain(self, dtype: torch.dtype) -> Tensor:
        if not self.config.lagged_latent_error_settling_carrier:
            raise RuntimeError("lagged latent-error settling carrier is disabled")
        packed = getattr(
            self,
            "_packed_lagged_latent_error_settling_gain",
            None,
        )
        if packed is not None:
            from pfc.packed_grouped_ternary import dense_weight_from_grouped_ternary

            return dense_weight_from_grouped_ternary(
                packed,
                dtype=dtype,
            ).squeeze(-2)
        if self.latent_error_settling_gain is None:
            raise RuntimeError("latent-error settling gain is missing")
        return self.latent_error_settling_gain

    def decode_features(self, features: Tensor) -> Tensor:
        if self.config.shared_embedding_decoder:
            if self.ternary_embedding is not None:
                dense_width = self.config.split_ternary_lexical_dense_width
                return F.linear(
                    features[..., :dense_width],
                    self.embedding.weight,
                ) + F.linear(
                    features[..., dense_width:],
                    self.ternary_embedding.weight,
                )
            if self.config.metric_consistent_shared_endpoint:
                if self._metric_shared_row_norm_bias is None:
                    raise RuntimeError("metric shared endpoint bias cache is missing")
                from pfc.phase734_metric_shared_endpoint import metric_shared_logits

                return metric_shared_logits(
                    features,
                    self.embedding.weight,
                    self._metric_shared_row_norm_bias,
                )
            return F.linear(features, self.embedding.weight)
        if self.decoder is None:
            raise RuntimeError("decoder parameter is missing")
        return features @ self.decoder

    def _token_prediction_preactivation(self, normalized: Tensor) -> Tensor:
        if not self.config.fixed_ternary_hadamard_lexical_predictor:
            return normalized @ self.token_predictor
        signs = self.fixed_ternary_hadamard_lexical_signs
        if signs is None:
            raise RuntimeError("fixed lexical Hadamard signs are missing")
        from pfc.phase788_fixed_ternary_hadamard import (
            fixed_ternary_hadamard_lexical_predictor,
        )

        return fixed_ternary_hadamard_lexical_predictor(normalized, signs)

    def _token_prediction_features(self, normalized: Tensor) -> Tensor:
        if not self.config.fixed_ternary_hadamard_lexical_predictor:
            return torch.tanh(normalized @ self.token_predictor)
        signs = self.fixed_ternary_hadamard_lexical_signs
        if signs is None:
            raise RuntimeError("fixed lexical Hadamard signs are missing")
        if (
            torch.is_inference_mode_enabled()
            and normalized.is_cuda
            and normalized.dtype == torch.float32
            and normalized.is_contiguous()
            and signs.is_contiguous()
        ):
            from pfc.phase788_triton_fixed_ternary_hadamard import (
                fused_fixed_ternary_hadamard_lexical_prediction,
            )

            return fused_fixed_ternary_hadamard_lexical_prediction(
                normalized,
                signs,
            )
        return torch.tanh(self._token_prediction_preactivation(normalized))

    def decode_swiglu_lexical_feature(self, feature: Tensor) -> Tensor:
        if self.swiglu_lexical_decoder is None:
            raise RuntimeError("SwiGLU lexical co-head is disabled")
        if feature.shape[-1] != self.config.ffn_width:
            raise ValueError("SwiGLU lexical feature width differs")
        return feature @ self.swiglu_lexical_decoder

    def decode_context_routed_lexical_residual(
        self,
        features: Tensor,
        token_ids: Tensor,
    ) -> Tensor:
        if self.context_routed_lexical_residual_decoders is None:
            raise RuntimeError("context-routed lexical residual is disabled")
        from pfc.phase674_context_routed_lexical_residual import (
            routed_lexical_logits,
        )

        return routed_lexical_logits(
            features,
            self.context_routed_lexical_residual_decoders,
            token_ids,
        )

    def decode_bounded_multiplicative_lexical_modulator(
        self,
        features: Tensor,
    ) -> Tensor:
        decoder = self.bounded_multiplicative_lexical_decoder
        if decoder is None:
            raise RuntimeError("bounded multiplicative lexical modulator is disabled")
        if features.shape[-1] != self.config.width:
            raise ValueError("multiplicative lexical feature width differs")
        return features @ decoder

    def decode_signed_square_lexical_lift(self, features: Tensor) -> Tensor:
        decoder = self.signed_square_lexical_decoder
        if decoder is None:
            raise RuntimeError("signed-square lexical lift is disabled")
        if features.shape[-1] != self.config.width:
            raise ValueError("signed-square lexical feature width differs")
        from pfc.phase715_signed_square_ternary_lexical_lift import (
            signed_square_features,
        )

        return signed_square_features(features) @ decoder

    def decode_cross_channel_quadratic_lexical_lift(
        self,
        features: Tensor,
    ) -> Tensor:
        decoder = self.cross_channel_quadratic_lexical_decoder
        if decoder is None:
            raise RuntimeError("cross-channel quadratic lexical lift is disabled")
        if features.shape[-1] != self.config.width:
            raise ValueError("cross-channel quadratic feature width differs")
        from pfc.phase768_cross_channel_quadratic_lexical_lift import (
            cross_channel_quadratic_features,
        )

        return cross_channel_quadratic_features(features) @ decoder

    def decode_shared_base_ternary_delta(self, features: Tensor) -> Tensor:
        decoder = self.shared_base_ternary_delta_decoder
        if decoder is None:
            raise RuntimeError("shared-base ternary-delta mixture is disabled")
        if features.shape[-1] != self.config.width:
            raise ValueError("ternary-delta lexical feature width differs")
        return features @ decoder

    def deployment_parameter_count(self) -> int:
        codebook_values = (
            self.associative_latent_codebook.numel()
            if self.associative_latent_codebook is not None
            else 0
        )
        released_top8_values = (
            self.config.vocab_size * self.config.width
            if self.config.fixed_hadamard_top8_lexical_reranker
            and self.top8_lexical_reranker_codes is None
            else 0
        )
        released_shared_candidate_values = (
            self.config.width
            if self.config.shared_nonlinear_candidate_compatibility
            and self.shared_candidate_compatibility_vector is None
            else 0
        )
        released_token_observation_coupling_values = (
            self.config.width
            if self.config.self_dual_hadamard_token_observation_coupling
            and self.token_observation_coupling_diagonals is None
            else 0
        )
        released_projected_self_bilinear_values = (
            self.config.width * self.config.width
            if self.config.projected_self_bilinear_token_observation
            and self.projected_self_bilinear_observation_matrix is None
            else 0
        )
        return (
            sum(parameter.numel() for parameter in self.parameters())
            + codebook_values
            + released_top8_values
            + released_shared_candidate_values
            + released_token_observation_coupling_values
            + released_projected_self_bilinear_values
        )

    @torch.no_grad()
    def refresh_ternary_vector_recirculation_active(self) -> None:
        selector = self.ternary_recirculation_selector
        self._ternary_vector_recirculation_active = bool(
            selector is not None and torch.count_nonzero(selector).item() > 0
        )

    def adaptive_recirculation_controllers(
        self,
        *,
        require_enabled: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[()]:
        controllers = (
            self.adaptive_recirculation_alpha_source,
            self.adaptive_recirculation_alpha_destination,
            self.adaptive_recirculation_beta_source,
            self.adaptive_recirculation_beta_destination,
        )
        if all(controller is None for controller in controllers):
            if require_enabled:
                raise RuntimeError("adaptive recirculation controller is disabled")
            return ()
        if any(controller is None for controller in controllers):
            raise RuntimeError("adaptive recirculation controller is incomplete")
        return controllers  # type: ignore[return-value]

    def _associative_layer_count(self) -> int:
        if self.config.associative_memory_rank == 0:
            return 0
        return self.config.associative_memory_layers or self.config.depth

    def _uses_semantic_count_sketch_backoff(self) -> bool:
        return (
            self.config.associative_memory_addressing
            in {
                CAUSAL_SEMANTIC_BACKOFF_COUNT_SKETCH_ADDRESSING,
                PAIR_MISS_SEMANTIC_BACKOFF_COUNT_SKETCH_ADDRESSING,
                PAIR_MISS_DISJOINT_TIMESCALE_SEMANTIC_CHANNELS_ADDRESSING,
            }
        )

    def _uses_pair_miss_semantic_backoff(self) -> bool:
        return (
            self.config.associative_memory_addressing
            in {
                PAIR_MISS_SEMANTIC_BACKOFF_COUNT_SKETCH_ADDRESSING,
                PAIR_MISS_DISJOINT_TIMESCALE_SEMANTIC_CHANNELS_ADDRESSING,
            }
        )

    def _uses_disjoint_timescale_semantic_channels(self) -> bool:
        return (
            self.config.associative_memory_addressing
            == PAIR_MISS_DISJOINT_TIMESCALE_SEMANTIC_CHANNELS_ADDRESSING
        )

    def _associative_value_width(self) -> int:
        return self.config.associative_memory_value_width or self.config.width

    def _encode_associative_values(self, source: Tensor) -> Tensor:
        if self.associative_value_encoder is None:
            return source
        return source @ self.associative_value_encoder

    def _decode_associative_values(self, values: Tensor) -> Tensor:
        if self.associative_value_decoder is None:
            return values
        return values @ self.associative_value_decoder

    def _encode_semantic_values(self, source: Tensor) -> Tensor:
        if self.associative_semantic_value_encoder is None:
            raise RuntimeError("semantic CountSketch value encoder is missing")
        return source @ self.associative_semantic_value_encoder

    def _decode_semantic_values(self, values: Tensor) -> Tensor:
        if self.associative_semantic_value_decoder is None:
            raise RuntimeError("semantic CountSketch value decoder is missing")
        return values @ self.associative_semantic_value_decoder

    def _decode_semantic_channel_values(
        self,
        values: Tensor,
        group_index: int,
    ) -> Tensor:
        decoders = self.associative_semantic_channel_decoders
        if decoders is None:
            raise RuntimeError("disjoint semantic channel decoders are missing")
        if group_index not in (0, 1):
            raise ValueError("semantic channel group must be fast or slow")
        decoded = values @ decoders[group_index]
        zeros = torch.zeros_like(decoded)
        return (
            torch.cat((decoded, zeros), dim=-1)
            if group_index == 0
            else torch.cat((zeros, decoded), dim=-1)
        )

    def _semantic_context_decay(self) -> float:
        return math.sqrt(self.config.associative_memory_decay)

    def _semantic_context_codes(self, context: Tensor) -> Tensor:
        if not self._uses_semantic_count_sketch_backoff():
            raise RuntimeError("semantic CountSketch backoff is disabled")
        if context.shape[-1] != self.config.width:
            raise ValueError("semantic context must end in model width")
        if self.associative_semantic_hash_mask is None:
            raise RuntimeError("semantic CountSketch hash mask is missing")
        padded_profile = self.config.padded_semantic_hadamard_hash and bool(
            self.config.width & (self.config.width - 1)
        )
        if padded_profile:
            _mark_semantic_hadamard_profile_stage(
                "semantic_context_mask_and_zero_pad", True
            )
        masked = context * self.associative_semantic_hash_mask.to(context.dtype)
        if padded_profile:
            workspace_width = 1 << (self.config.width - 1).bit_length()
            masked = F.pad(masked, (0, workspace_width - self.config.width))
            _mark_semantic_hadamard_profile_stage(
                "semantic_context_mask_and_zero_pad", False
            )
            _mark_semantic_hadamard_profile_stage(
                "normalized_1024_point_hadamard", True
            )
        transformed = _normalized_hadamard(masked)
        if padded_profile:
            _mark_semantic_hadamard_profile_stage(
                "normalized_1024_point_hadamard", False
            )
            _mark_semantic_hadamard_profile_stage(
                "semantic_address_and_sign_extraction", True
            )
        address_bits = transformed[..., :36].reshape(
            *context.shape[:-1],
            SEMANTIC_COUNT_SKETCH_TABLES,
            SEMANTIC_COUNT_SKETCH_HASH_BITS,
        ) > 0
        bit_weights = self.semantic_count_sketch_bit_weights
        if bit_weights is None:
            raise RuntimeError("semantic CountSketch bit weights are missing")
        addresses = (address_bits.to(torch.int64) * bit_weights).sum(dim=-1)
        positive_sign = transformed[..., 36:42] > 0
        codes = (addresses | (positive_sign.to(torch.int64) << 6)).to(torch.uint8)
        if padded_profile:
            _mark_semantic_hadamard_profile_stage(
                "semantic_address_and_sign_extraction", False
            )
        return codes

    @staticmethod
    def _unpack_semantic_context_codes(codes: Tensor) -> tuple[Tensor, Tensor]:
        if codes.shape[-1] != SEMANTIC_COUNT_SKETCH_TABLES:
            raise ValueError("semantic context codes must contain six tables")
        raw = codes.to(torch.int64)
        addresses = torch.bitwise_and(raw, SEMANTIC_COUNT_SKETCH_BUCKETS - 1)
        signs = torch.where(
            torch.bitwise_and(raw, 64) != 0,
            torch.ones_like(raw),
            -torch.ones_like(raw),
        )
        return addresses, signs

    def _pair_recurrence_bloom_addresses(
        self,
        previous_token_ids: Tensor,
        token_ids: Tensor,
    ) -> Tensor:
        if previous_token_ids.shape != token_ids.shape:
            raise ValueError("pair recurrence token shapes must match")
        pair_ids = (
            previous_token_ids.to(torch.int64) * self.config.vocab_size
            + token_ids.to(torch.int64)
            + 1
        )
        addresses = []
        for multiplier, offset in zip(
            PAIR_RECURRENCE_BLOOM_HASH_MULTIPLIERS,
            PAIR_RECURRENCE_BLOOM_HASH_OFFSETS,
            strict=True,
        ):
            mixed = pair_ids * multiplier + offset
            mixed = torch.bitwise_xor(mixed, torch.bitwise_right_shift(mixed, 16))
            addresses.append(
                torch.bitwise_and(mixed, PAIR_RECURRENCE_BLOOM_BITS - 1)
            )
        return torch.stack(addresses, dim=-1)

    @staticmethod
    def _unpack_pair_recurrence_bloom(cache: Tensor) -> Tensor:
        if cache.shape[-3:] != (
            PAIR_RECURRENCE_BLOOM_GENERATIONS,
            PAIR_RECURRENCE_BLOOM_PROBES,
            PAIR_RECURRENCE_BLOOM_BYTES,
        ):
            raise ValueError("invalid pair recurrence Bloom shape")
        if cache.dtype != torch.uint8:
            raise ValueError("pair recurrence Bloom must use uint8")
        bit_weights = torch.bitwise_left_shift(
            torch.ones(8, device=cache.device, dtype=torch.uint8),
            torch.arange(8, device=cache.device, dtype=torch.uint8),
        )
        return (
            torch.bitwise_and(cache.unsqueeze(-1), bit_weights) != 0
        ).reshape(*cache.shape[:-1], PAIR_RECURRENCE_BLOOM_BITS)

    @staticmethod
    def _pack_pair_recurrence_bloom(bits: Tensor) -> Tensor:
        if bits.shape[-3:] != (
            PAIR_RECURRENCE_BLOOM_GENERATIONS,
            PAIR_RECURRENCE_BLOOM_PROBES,
            PAIR_RECURRENCE_BLOOM_BITS,
        ):
            raise ValueError("invalid unpacked pair recurrence Bloom shape")
        bit_weights = torch.bitwise_left_shift(
            torch.ones(8, device=bits.device, dtype=torch.int16),
            torch.arange(8, device=bits.device, dtype=torch.int16),
        )
        grouped = bits.reshape(
            *bits.shape[:-1],
            PAIR_RECURRENCE_BLOOM_BYTES,
            8,
        )
        return (grouped.to(torch.int16) * bit_weights).sum(dim=-1).to(torch.uint8)

    def _pair_recurrence_bloom_step(
        self,
        previous_token_ids: Tensor,
        token_ids: Tensor,
        initial_cache: Tensor,
        token_count: int,
    ) -> tuple[Tensor, Tensor]:
        if not self._uses_pair_miss_semantic_backoff():
            raise RuntimeError("pair-miss semantic backoff is disabled")
        if token_count < 0:
            raise ValueError("token count must be nonnegative")
        batch = token_ids.shape[0]
        if initial_cache.shape != (
            batch,
            PAIR_RECURRENCE_BLOOM_GENERATIONS,
            PAIR_RECURRENCE_BLOOM_PROBES,
            PAIR_RECURRENCE_BLOOM_BYTES,
        ):
            raise ValueError("invalid pair recurrence Bloom step state")
        cache = initial_cache
        active_generation = (
            token_count // PAIR_RECURRENCE_BLOOM_GENERATION_SPAN
        ) % PAIR_RECURRENCE_BLOOM_GENERATIONS
        if token_count > 0 and (
            token_count % PAIR_RECURRENCE_BLOOM_GENERATION_SPAN == 0
        ):
            cache = cache.clone()
            cache[:, active_generation].zero_()
        if cache.is_cuda and self.config.fused_indexed_count_sketch_inference:
            return fused_pair_recurrence_bloom_step(
                cache,
                previous_token_ids,
                token_ids,
                vocab_size=self.config.vocab_size,
                active_generation=active_generation,
            )
        addresses = self._pair_recurrence_bloom_addresses(
            previous_token_ids, token_ids
        )
        byte_addresses = torch.bitwise_right_shift(addresses, 3)
        bit_masks = torch.bitwise_left_shift(
            torch.ones_like(addresses, dtype=torch.uint8),
            torch.bitwise_and(addresses, 7).to(torch.uint8),
        )
        generation_hits = []
        for generation in range(PAIR_RECURRENCE_BLOOM_GENERATIONS):
            probe_hits = []
            for probe in range(PAIR_RECURRENCE_BLOOM_PROBES):
                value = torch.gather(
                    cache[:, generation, probe],
                    1,
                    byte_addresses[:, probe : probe + 1],
                ).squeeze(1)
                probe_hits.append(
                    torch.bitwise_and(value, bit_masks[:, probe]) != 0
                )
            generation_hits.append(torch.stack(probe_hits, dim=-1).all(dim=-1))
        seen = torch.stack(generation_hits, dim=-1).any(dim=-1)
        updated = cache.clone()
        for probe in range(PAIR_RECURRENCE_BLOOM_PROBES):
            byte_address = byte_addresses[:, probe : probe + 1]
            current = torch.gather(
                updated[:, active_generation, probe], 1, byte_address
            )
            updated[:, active_generation, probe].scatter_(
                1,
                byte_address,
                torch.bitwise_or(current, bit_masks[:, probe : probe + 1]),
            )
        return seen, updated

    def _pair_recurrence_bloom_sequence(
        self,
        token_ids: Tensor,
        initial_context_token: Tensor,
        initial_cache: Tensor,
        token_count: int,
        reset_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not self._uses_pair_miss_semantic_backoff():
            raise RuntimeError("pair-miss semantic backoff is disabled")
        if token_ids.ndim != 2 or initial_context_token.shape != token_ids.shape[:1]:
            raise ValueError("invalid pair recurrence Bloom sequence tokens")
        if reset_mask is not None:
            if reset_mask.shape != token_ids.shape:
                raise ValueError("pair recurrence reset mask shape differs")
            return self._pair_recurrence_bloom_sequence_with_resets(
                token_ids,
                initial_context_token,
                initial_cache,
                token_count,
                reset_mask,
            )
        context_tokens = torch.cat(
            (initial_context_token.unsqueeze(1), token_ids[:, :-1]), dim=1
        )
        addresses = self._pair_recurrence_bloom_addresses(
            context_tokens, token_ids
        )
        cache = initial_cache
        seen_chunks = []
        offset = 0
        while offset < token_ids.shape[1]:
            absolute_start = token_count + offset
            active_generation = (
                absolute_start // PAIR_RECURRENCE_BLOOM_GENERATION_SPAN
            ) % PAIR_RECURRENCE_BLOOM_GENERATIONS
            if absolute_start > 0 and (
                absolute_start % PAIR_RECURRENCE_BLOOM_GENERATION_SPAN == 0
            ):
                cache = cache.clone()
                cache[:, active_generation].zero_()
            until_boundary = PAIR_RECURRENCE_BLOOM_GENERATION_SPAN - (
                absolute_start % PAIR_RECURRENCE_BLOOM_GENERATION_SPAN
            )
            stop = min(token_ids.shape[1], offset + until_boundary)
            chunk_addresses = addresses[:, offset:stop]
            unpacked = self._unpack_pair_recurrence_bloom(cache)
            causal_prior = torch.ones(
                stop - offset,
                stop - offset,
                device=token_ids.device,
                dtype=torch.bool,
            ).tril(diagonal=-1)
            generation_probe_hits: list[list[Tensor]] = [
                [] for _ in range(PAIR_RECURRENCE_BLOOM_GENERATIONS)
            ]
            for probe in range(PAIR_RECURRENCE_BLOOM_PROBES):
                probe_addresses = chunk_addresses[:, :, probe]
                prior_hit = (
                    (
                        probe_addresses.unsqueeze(2)
                        == probe_addresses.unsqueeze(1)
                    )
                    & causal_prior
                ).any(dim=-1)
                for generation in range(PAIR_RECURRENCE_BLOOM_GENERATIONS):
                    initial_hit = torch.gather(
                        unpacked[:, generation, probe],
                        1,
                        probe_addresses,
                    )
                    generation_probe_hits[generation].append(
                        initial_hit | prior_hit
                        if generation == active_generation
                        else initial_hit
                    )
                unpacked[:, active_generation, probe].scatter_(
                    1,
                    probe_addresses,
                    True,
                )
            seen_chunks.append(
                torch.stack(
                    [
                        torch.stack(probe_hits, dim=-1).all(dim=-1)
                        for probe_hits in generation_probe_hits
                    ],
                    dim=-1,
                ).any(dim=-1)
            )
            cache = self._pack_pair_recurrence_bloom(unpacked)
            offset = stop
        return torch.cat(seen_chunks, dim=1), cache

    def _pair_recurrence_bloom_sequence_with_resets(
        self,
        token_ids: Tensor,
        initial_context_token: Tensor,
        initial_cache: Tensor,
        token_count: int,
        reset_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, sequence = token_ids.shape
        bos_context = torch.full_like(token_ids, self.config.bos_token_id)
        context_tokens = torch.cat(
            (initial_context_token.unsqueeze(1), token_ids[:, :-1]),
            dim=1,
        )
        context_tokens = torch.where(reset_mask, bos_context, context_tokens)
        addresses = self._pair_recurrence_bloom_addresses(
            context_tokens,
            token_ids,
        )
        segment_ids = reset_mask.to(torch.long).cumsum(dim=1)
        causal = torch.ones(
            sequence,
            sequence,
            device=token_ids.device,
            dtype=torch.bool,
        ).tril(diagonal=-1)
        same_segment = segment_ids.unsqueeze(2).eq(segment_ids.unsqueeze(1))
        unpacked = self._unpack_pair_recurrence_bloom(initial_cache)
        before_first_reset = segment_ids.eq(0)
        local_positions = torch.arange(
            sequence,
            device=token_ids.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch, -1)
        boundary_positions = torch.where(
            reset_mask,
            local_positions,
            torch.full_like(local_positions, -1),
        ).cummax(dim=1).values
        ages = torch.where(
            boundary_positions.ge(0),
            local_positions - boundary_positions,
            local_positions + token_count,
        )
        active_generations = torch.div(
            ages,
            PAIR_RECURRENCE_BLOOM_GENERATION_SPAN,
            rounding_mode="floor",
        ).remainder(PAIR_RECURRENCE_BLOOM_GENERATIONS)
        generation_hits = []
        for generation in range(PAIR_RECURRENCE_BLOOM_GENERATIONS):
            probe_hits = []
            for probe in range(PAIR_RECURRENCE_BLOOM_PROBES):
                probe_addresses = addresses[:, :, probe]
                initial_hit = torch.gather(
                    unpacked[:, generation, probe],
                    1,
                    probe_addresses,
                ) & before_first_reset
                prior_hit = (
                    probe_addresses.unsqueeze(2).eq(
                        probe_addresses.unsqueeze(1)
                    )
                    & causal
                    & same_segment
                    & active_generations.unsqueeze(1).eq(generation)
                ).any(dim=-1)
                probe_hits.append(initial_hit | prior_hit)
            generation_hits.append(torch.stack(probe_hits, dim=-1).all(dim=-1))
        seen = torch.stack(generation_hits, dim=-1).any(dim=-1)

        has_reset = reset_mask.any(dim=1, keepdim=True)
        last_segment = segment_ids.eq(segment_ids[:, -1:])
        retained = torch.where(has_reset, last_segment, torch.ones_like(last_segment))
        final_bits = torch.where(
            has_reset.reshape(batch, 1, 1, 1),
            torch.zeros_like(unpacked),
            unpacked,
        )
        for generation in range(PAIR_RECURRENCE_BLOOM_GENERATIONS):
            generation_mask = retained & active_generations.eq(generation)
            for probe in range(PAIR_RECURRENCE_BLOOM_PROBES):
                additions = torch.zeros(
                    batch,
                    PAIR_RECURRENCE_BLOOM_BITS,
                    device=token_ids.device,
                    dtype=torch.int16,
                )
                additions.scatter_reduce_(
                    1,
                    addresses[:, :, probe],
                    generation_mask.to(torch.int16),
                    reduce="amax",
                    include_self=True,
                )
                final_bits[:, generation, probe] |= additions.bool()
        return seen, self._pack_pair_recurrence_bloom(final_bits)

    def _scale_associative_reads(self, reads: Tensor) -> Tensor:
        if reads.shape[0] != self.config.depth:
            raise ValueError("associative reads must start with layer dimension")
        scale_shape = (self.config.depth,) + (1,) * (reads.ndim - 1)
        scales = self.associative_read_scales.to(dtype=reads.dtype)
        return reads * scales.view(scale_shape)

    def _block_delta_factors(
        self,
        source: Tensor,
        layer: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.block_delta_down is None:
            raise RuntimeError("block delta memory is disabled")
        projections = (
            self.block_delta_query_up,
            self.block_delta_key_up,
            self.block_delta_erase_up,
            self.block_delta_decay_up,
            self.block_delta_write_up,
        )
        if any(projection is None for projection in projections):
            raise RuntimeError("block delta projections are missing")
        if (
            self.block_delta_erase_bias is None
            or self.block_delta_decay_bias is None
            or self.block_delta_write_bias is None
        ):
            raise RuntimeError("block delta biases are missing")
        heads = self.config.block_delta_memory_heads
        key_width = self.config.block_delta_memory_key_width
        value_width = self._associative_value_width()
        hidden = torch.tanh(source @ self.block_delta_down[layer])
        query_raw = (hidden @ self.block_delta_query_up[layer]).reshape(
            *source.shape[:-1], heads, key_width
        )
        key_raw = (hidden @ self.block_delta_key_up[layer]).reshape(
            *source.shape[:-1], heads, key_width
        )
        query = F.normalize(query_raw, dim=-1, eps=self.config.rms_eps)
        key = F.normalize(key_raw, dim=-1, eps=self.config.rms_eps)
        erase = torch.sigmoid(
            hidden @ self.block_delta_erase_up[layer]
            + self.block_delta_erase_bias[layer]
        ).reshape(*source.shape[:-1], heads, key_width)
        decay = torch.sigmoid(
            hidden @ self.block_delta_decay_up[layer]
            + self.block_delta_decay_bias[layer]
        ).reshape(*source.shape[:-1], heads, key_width)
        write = torch.sigmoid(
            hidden @ self.block_delta_write_up[layer]
            + self.block_delta_write_bias[layer]
        ).reshape(*source.shape[:-1], heads, value_width)
        return query, key, erase, write, decay

    def _block_delta_sequence(
        self,
        source: Tensor,
        initial_memory: Tensor,
        layer: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        heads = self.config.block_delta_memory_heads
        key_width = self.config.block_delta_memory_key_width
        value_width = self._associative_value_width()
        query, key, erase, write_gate, decay = self._block_delta_factors(
            source,
            layer,
        )
        value = source.reshape(*source.shape[:-1], heads, value_width)
        transition = torch.diag_embed(decay) - key.unsqueeze(-1) * (
            erase * key * decay
        ).unsqueeze(-2)
        write = key.unsqueeze(-1) * (write_gate * value).unsqueeze(-2)
        memory = _matrix_memory_affine_scan(
            transition,
            write,
            initial_memory.reshape(source.shape[0], heads, key_width, value_width),
        )
        read = (query.unsqueeze(-1) * memory).sum(dim=-2).reshape_as(source)
        flat_memory = memory[:, -1].reshape(
            source.shape[0],
            self.config.associative_memory_rank,
            value_width,
        )
        zero_normalizer = torch.zeros(
            source.shape[0],
            self.config.associative_memory_rank,
            device=source.device,
            dtype=source.dtype,
        )
        return read, flat_memory, zero_normalizer, key[:, -1].flatten(-2)

    def _gdn2_sequence(
        self,
        source: Tensor,
        initial_memory: Tensor,
        layer: int,
    ) -> tuple[Tensor, Tensor]:
        """Apply the paper's decoupled erase/write matrix recurrence."""

        parameters = (
            self.gdn2_query,
            self.gdn2_key,
            self.gdn2_value,
            self.gdn2_decay_down,
            self.gdn2_decay_up,
            self.gdn2_erase,
            self.gdn2_write,
            self.gdn2_decay_log_rate,
            self.gdn2_decay_bias,
            self.gdn2_output_gate_down,
            self.gdn2_output_gate_up,
            self.gdn2_output_gate_bias,
            self.gdn2_output_norm,
            self.gdn2_output,
        )
        if any(parameter is None for parameter in parameters):
            raise RuntimeError("GDN2 mixer is disabled or incomplete")
        if not 0 <= layer < self.config.gdn2_mixer_layers:
            raise ValueError("GDN2 mixer layer is out of range")
        if source.ndim != 3 or source.shape[-1] != self.config.width:
            raise ValueError("GDN2 source must have [batch, sequence, width]")

        heads = self.config.gdn2_mixer_heads
        key_width = self.config.gdn2_mixer_key_width
        value_width = self.config.width // heads
        expected_memory = (source.shape[0], heads, key_width, value_width)
        if initial_memory.shape != expected_memory:
            raise ValueError("GDN2 initial memory shape differs")

        assert self.gdn2_query is not None
        assert self.gdn2_key is not None
        assert self.gdn2_value is not None
        assert self.gdn2_decay_down is not None
        assert self.gdn2_decay_up is not None
        assert self.gdn2_erase is not None
        assert self.gdn2_write is not None
        assert self.gdn2_decay_log_rate is not None
        assert self.gdn2_decay_bias is not None
        assert self.gdn2_output_gate_down is not None
        assert self.gdn2_output_gate_up is not None
        assert self.gdn2_output_gate_bias is not None
        assert self.gdn2_output_norm is not None
        assert self.gdn2_output is not None

        query = F.silu(source @ self.gdn2_query[layer]).reshape(
            *source.shape[:-1], heads, key_width
        )
        key = F.silu(source @ self.gdn2_key[layer]).reshape(
            *source.shape[:-1], heads, key_width
        )
        query = F.normalize(query, dim=-1, eps=self.config.rms_eps)
        key = F.normalize(key, dim=-1, eps=self.config.rms_eps)
        value = F.silu(source @ self.gdn2_value[layer]).reshape(
            *source.shape[:-1], heads, value_width
        )
        erase = torch.sigmoid(source @ self.gdn2_erase[layer]).reshape_as(key)
        write_gate = torch.sigmoid(source @ self.gdn2_write[layer]).reshape_as(value)

        decay_pre = (
            F.silu(source @ self.gdn2_decay_down[layer]) @ self.gdn2_decay_up[layer]
        )
        rates = (
            self.gdn2_decay_log_rate[layer].float().exp().repeat_interleave(key_width)
        )
        log_decay = -rates * F.softplus(
            decay_pre.float() + self.gdn2_decay_bias[layer].float()
        )
        decay = (
            log_decay.exp()
            .to(source.dtype)
            .reshape(*source.shape[:-1], heads, key_width)
        )

        transition = torch.diag_embed(decay) - key.unsqueeze(-1) * (
            erase * key * decay
        ).unsqueeze(-2)
        write = key.unsqueeze(-1) * (write_gate * value).unsqueeze(-2)
        memory = _matrix_memory_affine_scan(transition, write, initial_memory)
        read = (query.unsqueeze(-1) * memory).sum(dim=-2)

        output_gate = F.silu(
            F.silu(source @ self.gdn2_output_gate_down[layer])
            @ self.gdn2_output_gate_up[layer]
            + self.gdn2_output_gate_bias[layer]
        ).reshape_as(read)
        normalized_read = read * (
            read.float().square().mean(dim=-1, keepdim=True) + self.config.rms_eps
        ).rsqrt().to(read.dtype)
        normalized_read = normalized_read * self.gdn2_output_norm[layer]
        output = (normalized_read * output_gate).reshape_as(source)
        output = output @ self.gdn2_output[layer]
        return output, memory[:, -1]

    def _factorization_start_layer(self) -> int:
        return self.config.depth - self.config.factorization_memory_layers

    def _factorization_memory_factors(
        self,
        source: Tensor,
        memory_layer: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        parameters = (
            self.factorization_affinity,
            self.factorization_update,
            self.factorization_update_bias,
            self.factorization_merge,
            self.factorization_merge_bias,
            self.factorization_value_encoder,
        )
        if any(parameter is None for parameter in parameters):
            raise RuntimeError("factorization memory is disabled or incomplete")
        if not 0 <= memory_layer < self.config.factorization_memory_layers:
            raise ValueError("factorization memory layer is out of range")
        if source.shape[-1] != self.config.width:
            raise ValueError("factorization memory source must end in model width")

        assert self.factorization_affinity is not None
        assert self.factorization_update is not None
        assert self.factorization_update_bias is not None
        assert self.factorization_merge is not None
        assert self.factorization_merge_bias is not None
        assert self.factorization_value_encoder is not None
        normalized = source * torch.rsqrt(
            source.square().mean(dim=-1, keepdim=True) + self.config.rms_eps
        )
        affinity = torch.softmax(
            normalized @ self.factorization_affinity[memory_layer],
            dim=-1,
        )
        update_rate = torch.sigmoid(
            normalized @ self.factorization_update[memory_layer]
            + self.factorization_update_bias[memory_layer]
        ).unsqueeze(-1)
        merge_rate = torch.sigmoid(
            normalized @ self.factorization_merge[memory_layer]
            + self.factorization_merge_bias[memory_layer]
        ).unsqueeze(-1)
        value = normalized @ self.factorization_value_encoder[memory_layer]
        return update_rate * affinity, merge_rate * affinity, value

    def _factorization_memory_step(
        self,
        source: Tensor,
        initial_memory: Tensor,
        memory_layer: int,
    ) -> tuple[Tensor, Tensor]:
        if source.ndim != 2:
            raise ValueError("factorization memory step source must be [batch, width]")
        expected_memory = (
            source.shape[0],
            self.config.factorization_memory_slots,
            self.config.factorization_memory_value_width,
        )
        if initial_memory.shape != expected_memory:
            raise ValueError("factorization memory step state differs")
        update, merge, value = self._factorization_memory_factors(
            source,
            memory_layer,
        )
        memory = (1.0 - update).unsqueeze(-1) * initial_memory + update.unsqueeze(
            -1
        ) * value.unsqueeze(-2)
        normalized_memory = memory * torch.rsqrt(
            memory.square().mean(dim=-1, keepdim=True) + self.config.rms_eps
        )
        merged = (merge.unsqueeze(-1) * normalized_memory).sum(dim=-2)
        if self.factorization_value_decoder is None:
            raise RuntimeError("factorization memory decoder is missing")
        return merged @ self.factorization_value_decoder[memory_layer], memory

    def _factorization_memory_sequence(
        self,
        source: Tensor,
        initial_memory: Tensor,
        memory_layer: int,
    ) -> tuple[Tensor, Tensor]:
        if source.ndim != 3:
            raise ValueError(
                "factorization memory sequence source must be [batch, sequence, width]"
            )
        batch, sequence, _ = source.shape
        slots = self.config.factorization_memory_slots
        memory_width = self.config.factorization_memory_value_width
        expected_memory = (batch, slots, memory_width)
        if initial_memory.shape != expected_memory:
            raise ValueError("factorization memory sequence state differs")
        update, merge, value = self._factorization_memory_factors(
            source,
            memory_layer,
        )
        decay = (
            (1.0 - update)
            .unsqueeze(-1)
            .expand(
                batch,
                sequence,
                slots,
                memory_width,
            )
        )
        write = update.unsqueeze(-1) * value.unsqueeze(-2)
        memories = _affine_scan(
            decay.reshape(batch, sequence, slots * memory_width),
            write.reshape(batch, sequence, slots * memory_width),
            initial_memory.reshape(batch, slots * memory_width),
        ).reshape(batch, sequence, slots, memory_width)
        normalized_memory = memories * torch.rsqrt(
            memories.square().mean(dim=-1, keepdim=True) + self.config.rms_eps
        )
        merged = (merge.unsqueeze(-1) * normalized_memory).sum(dim=-2)
        if self.factorization_value_decoder is None:
            raise RuntimeError("factorization memory decoder is missing")
        read = merged @ self.factorization_value_decoder[memory_layer]
        return read, memories[:, -1]

    def _bounded_attention_start_layer(self) -> int:
        return self.config.depth - self.config.bounded_attention_layers

    def _rotate_bounded_attention(
        self,
        value: Tensor,
        prior_token_count: int,
    ) -> Tensor:
        if value.ndim != 3:
            raise ValueError("rotary attention input must be [batch, sequence, width]")
        width = value.shape[-1]
        if width % 2:
            raise ValueError("rotary attention width must be even")
        positions = torch.arange(
            prior_token_count,
            prior_token_count + value.shape[1],
            device=value.device,
            dtype=torch.float32,
        )
        channels = torch.arange(0, width, 2, device=value.device, dtype=torch.float32)
        inverse_frequency = torch.pow(10_000.0, -channels / width)
        phase = torch.outer(positions, inverse_frequency)
        cosine = phase.cos().to(dtype=value.dtype).unsqueeze(0)
        sine = phase.sin().to(dtype=value.dtype).unsqueeze(0)
        even = value[..., 0::2]
        odd = value[..., 1::2]
        return torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine),
            dim=-1,
        ).flatten(-2)

    def _bounded_attention_sequence(
        self,
        source: Tensor,
        initial_keys: Tensor,
        initial_values: Tensor,
        attention_layer: int,
        prior_token_count: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if source.ndim != 3:
            raise ValueError(
                "bounded attention source must be [batch, sequence, width]"
            )
        if not 0 <= attention_layer < self.config.bounded_attention_layers:
            raise ValueError("bounded attention layer is out of range")
        batch, sequence, width = source.shape
        window = self.config.bounded_attention_window
        attention_width = self.config.bounded_attention_width
        expected_cache = (batch, window, attention_width)
        if (
            initial_keys.shape != expected_cache
            or initial_values.shape != expected_cache
        ):
            raise ValueError("bounded attention cache shape differs")
        parameters = (
            self.bounded_attention_query,
            self.bounded_attention_key,
            self.bounded_attention_value,
            self.bounded_attention_output,
        )
        if any(parameter is None for parameter in parameters):
            raise RuntimeError("bounded attention is disabled or incomplete")
        assert self.bounded_attention_query is not None
        assert self.bounded_attention_key is not None
        assert self.bounded_attention_value is not None
        assert self.bounded_attention_output is not None

        normalized = source * torch.rsqrt(
            source.square().mean(dim=-1, keepdim=True) + self.config.rms_eps
        )
        query = normalized @ self.bounded_attention_query[attention_layer]
        key = normalized @ self.bounded_attention_key[attention_layer]
        value = normalized @ self.bounded_attention_value[attention_layer]
        if self.config.bounded_attention_rotary:
            query = self._rotate_bounded_attention(query, prior_token_count)
            key = self._rotate_bounded_attention(key, prior_token_count)
        all_keys = torch.cat((initial_keys, key), dim=1)
        all_values = torch.cat((initial_values, value), dim=1)
        key_windows = all_keys.unfold(1, window, 1)[:, 1:].permute(0, 1, 3, 2)
        value_windows = all_values.unfold(1, window, 1)[:, 1:].permute(0, 1, 3, 2)
        scores = torch.einsum("bth,btwh->btw", query, key_windows)
        scores = scores * (attention_width**-0.5)

        prior_valid = min(max(prior_token_count, 0), window)
        valid_counts = (
            torch.arange(1, sequence + 1, device=source.device) + prior_valid
        ).clamp_max(window)
        positions = torch.arange(window, device=source.device)
        valid = positions.unsqueeze(0) >= (window - valid_counts).unsqueeze(1)
        scores = scores.masked_fill(~valid.unsqueeze(0), -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        merged = torch.einsum("btw,btwh->bth", weights, value_windows)
        read = merged @ self.bounded_attention_output[attention_layer]
        read, _, _ = self._bounded_attention_utility(
            source,
            read,
            weights,
            attention_layer,
        )
        return read, all_keys[:, -window:], all_values[:, -window:]

    def _bounded_attention_utility(
        self,
        source: Tensor,
        ungated_read: Tensor,
        weights: Tensor,
        attention_layer: int,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        semantic_rank = self.config.bounded_attention_causal_semantic_utility_rank
        if not self.config.bounded_attention_utility_gate and semantic_rank == 0:
            gate = torch.ones_like(ungated_read[..., :1])
            return ungated_read, None, gate
        if source.shape != ungated_read.shape:
            raise ValueError("utility source and recall read shapes differ")
        if weights.shape[:2] != source.shape[:2]:
            raise ValueError("utility attention weights shape differs")
        if semantic_rank > 0:
            if (
                self.bounded_attention_semantic_utility_source is None
                or self.bounded_attention_semantic_utility_read is None
                or self.bounded_attention_semantic_utility_bias is None
            ):
                raise RuntimeError("causal semantic utility parameters are missing")
            source_normalized = source * torch.rsqrt(
                source.square().mean(dim=-1, keepdim=True) + self.config.rms_eps
            )
            read_normalized = ungated_read * torch.rsqrt(
                ungated_read.square().mean(dim=-1, keepdim=True)
                + self.config.rms_eps
            )
            source_semantic = (
                source_normalized
                @ self.bounded_attention_semantic_utility_source[attention_layer]
            )
            read_semantic = (
                read_normalized
                @ self.bounded_attention_semantic_utility_read[attention_layer]
            )
            features = torch.cat(
                (
                    source_normalized,
                    read_normalized,
                    source_semantic,
                    read_semantic,
                ),
                dim=-1,
            ).detach()
            if self.config.bounded_attention_channel_preserving_semantic_utility_gate:
                logit = (
                    source_semantic * read_semantic
                    + self.bounded_attention_semantic_utility_bias[attention_layer]
                )
                activation = torch.tanh(logit)
                channel_gate = 1.0 + 0.5 * (
                    activation - activation.mean(dim=-1, keepdim=True)
                )
                gate = channel_gate.repeat(
                    1,
                    1,
                    ungated_read.shape[-1] // semantic_rank,
                )
                return ungated_read * gate, features, gate
            logit = (
                (source_semantic * read_semantic).sum(dim=-1)
                * semantic_rank**-0.5
                + self.bounded_attention_semantic_utility_bias[attention_layer]
            )
            exploration = self._bounded_attention_causal_utility_exploration(
                source.shape[0],
                source.shape[1],
                source.device,
                source.dtype,
            )
            logit = logit + exploration
            gate = (2.0 * torch.sigmoid(logit)).unsqueeze(-1)
            return ungated_read * gate, features, gate
        if (
            self.bounded_attention_utility_weight is None
            or self.bounded_attention_utility_bias is None
        ):
            raise RuntimeError("bounded attention utility parameters are missing")
        source_rms = torch.sqrt(source.square().mean(dim=-1) + self.config.rms_eps)
        read_rms = torch.sqrt(ungated_read.square().mean(dim=-1) + self.config.rms_eps)
        cosine = (source * ungated_read).mean(dim=-1) / (source_rms * read_rms)
        read_ratio = read_rms / (source_rms + read_rms + self.config.rms_eps)
        features = torch.stack(
            (
                weights.amax(dim=-1),
                weights.square().sum(dim=-1),
                cosine.clamp(-1.0, 1.0),
                read_ratio,
            ),
            dim=-1,
        ).detach()
        logit = (
            features @ self.bounded_attention_utility_weight[attention_layer]
            + self.bounded_attention_utility_bias[attention_layer]
        )
        gate = (2.0 * torch.sigmoid(logit)).unsqueeze(-1)
        return ungated_read * gate, features, gate

    def _bounded_attention_causal_utility_exploration(
        self,
        batch: int,
        sequence: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        rank = self.config.bounded_attention_causal_semantic_utility_rank
        if (
            rank == 0
            or not self.training
            or self._bounded_attention_causal_exploration_step is None
        ):
            return torch.zeros(batch, sequence, device=device, dtype=dtype)
        block = self.config.bounded_attention_window
        episode_width = 2 * block
        if sequence % episode_width:
            raise ValueError(
                "causal semantic utility requires complete support/query episodes"
            )
        positions = torch.arange(sequence, device=device)
        episodes = torch.div(positions, episode_width, rounding_mode="floor")
        support = positions.remainder(episode_width) < block
        batch_indices = torch.arange(batch, device=device).unsqueeze(1)
        parity = (
            batch_indices
            + episodes.unsqueeze(0)
            + self._bounded_attention_causal_exploration_step
        ).remainder(2)
        signs = parity.to(dtype).mul_(2.0).sub_(1.0)
        return signs * support.to(dtype).unsqueeze(0)

    def settling_state_step(self) -> float:
        if not self.config.loop_aware_settling_scaling:
            return self.config.state_step
        return self.config.state_step / math.sqrt(self.config.settling_rounds)

    def deeploop_skip_scale(self) -> float:
        """DeepLoop alpha for one Jacobi block revisited across settle rounds."""

        return math.sqrt(2.0 * self.config.settling_rounds)

    def deeploop_initialization_gain(self) -> float:
        """DeepLoop beta for one Jacobi block revisited across settle rounds."""

        return (8.0 * self.config.settling_rounds) ** -0.5

    def candidate_for_visit(self, visit: int) -> Tensor:
        if not 0 <= visit < self.config.settling_rounds:
            raise ValueError("settling visit is out of range")
        if visit == 0 or self.later_candidates is None:
            return self.candidate
        return self.later_candidates[visit - 1]

    def candidate_adapter_for_visit(self, visit: int) -> Tensor | None:
        if not 0 <= visit < self.config.settling_rounds:
            raise ValueError("settling visit is out of range")
        if visit == 0 or self.later_candidate_adapters is None:
            return None
        return self.later_candidate_adapters[visit - 1]

    def candidate_low_rank_adapter_for_visit(
        self,
        visit: int,
    ) -> tuple[Tensor, Tensor] | None:
        if not 0 <= visit < self.config.settling_rounds:
            raise ValueError("settling visit is out of range")
        if visit == 0 or self.later_candidate_adapter_down is None:
            return None
        if self.later_candidate_adapter_up is None:
            raise RuntimeError("visit low-rank adapter output factor is missing")
        return (
            self.later_candidate_adapter_down[visit - 1],
            self.later_candidate_adapter_up[visit - 1],
        )

    def candidate_low_rank_hidden_for_visit(
        self,
        value: Tensor,
        visit: int,
    ) -> tuple[Tensor, Tensor, Tensor] | None:
        adapter = self.candidate_low_rank_adapter_for_visit(visit)
        if adapter is None:
            return None
        down, up = adapter
        hidden_pre = torch.bmm(value, down)
        hidden = (
            torch.tanh(hidden_pre)
            if self.config.nonlinear_visit_low_rank_adapter
            else hidden_pre
        )
        return hidden_pre, hidden, up

    def candidate_dual_stream_hidden_for_visit(
        self,
        value: Tensor,
        visit: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | None:
        if not 0 <= visit < self.config.settling_rounds:
            raise ValueError("settling visit is out of range")
        if visit == 0 or self.later_dual_stream_adapter_down is None:
            return None
        if self.later_dual_stream_adapter_up is None:
            raise RuntimeError("dual-stream visit adapter output factors are missing")
        down = self.later_dual_stream_adapter_down[visit - 1]
        up = self.later_dual_stream_adapter_up[visit - 1]
        positive_source = torch.relu(value)
        negative_source = torch.relu(-value)
        positive_hidden = torch.tanh(
            torch.bmm(positive_source, down[:, 0])
            - torch.bmm(negative_source, down[:, 1])
        )
        negative_hidden = torch.tanh(
            torch.bmm(negative_source, down[:, 2])
            - torch.bmm(positive_source, down[:, 3])
        )
        return (
            positive_source,
            negative_source,
            positive_hidden,
            negative_hidden,
            up[:, 0],
            up[:, 1],
        )

    def candidate_pre_for_visit(self, value: Tensor, visit: int) -> Tensor:
        candidate_pre = torch.bmm(value, self.candidate_for_visit(visit))
        adapter = self.candidate_adapter_for_visit(visit)
        if adapter is not None:
            candidate_pre = candidate_pre + value * adapter.unsqueeze(1)
        low_rank_hidden = self.candidate_low_rank_hidden_for_visit(value, visit)
        if low_rank_hidden is not None:
            _hidden_pre, hidden, up = low_rank_hidden
            candidate_pre = candidate_pre + torch.bmm(hidden, up)
        dual_stream_hidden = self.candidate_dual_stream_hidden_for_visit(value, visit)
        if dual_stream_hidden is not None:
            (
                _positive_source,
                _negative_source,
                positive_hidden,
                negative_hidden,
                positive_up,
                negative_up,
            ) = dual_stream_hidden
            candidate_pre = (
                candidate_pre
                + torch.bmm(positive_hidden, positive_up)
                - torch.bmm(negative_hidden, negative_up)
            )
        return candidate_pre

    def activate_settling_candidate(self, candidate_pre: Tensor) -> Tensor:
        if self.config.softsign_reciprocal_transition:
            return candidate_pre / (1.0 + candidate_pre.abs())
        return torch.tanh(candidate_pre)

    @torch.no_grad()
    def set_associative_memory_context_scale(self, scale: float) -> None:
        if self.config.associative_memory_addressing not in {
            "token_plus_bigram_hash",
            "token_plus_context_bucket",
            "token_plus_latent_hash",
            "token_plus_latent_codebook",
        }:
            raise RuntimeError("context scale requires factorized context")
        if scale < 0:
            raise ValueError("context scale must be nonnegative")
        self.associative_read_scales[1 : self._associative_layer_count()].fill_(scale)

    @torch.no_grad()
    def update_associative_latent_codebook(self, raw_states: Tensor) -> None:
        if self.config.associative_memory_addressing != "token_plus_latent_codebook":
            return
        if self.associative_latent_codebook is None:
            raise RuntimeError("latent associative codebook is missing")
        if self.associative_latent_codebook_counts is None:
            raise RuntimeError("latent associative codebook counts are missing")
        if raw_states.ndim != 4 or raw_states.shape[0] != self.config.depth:
            raise ValueError("raw states must have [layer, batch, sequence, width]")
        batch, sequence, width = raw_states.shape[1:]
        code_count = self.config.associative_memory_codebook_size
        rate = self.config.associative_memory_codebook_rate
        for layer in range(1, self._associative_layer_count()):
            source = torch.cat(
                (
                    torch.zeros(
                        batch,
                        1,
                        width,
                        device=raw_states.device,
                        dtype=raw_states.dtype,
                    ),
                    raw_states[layer - 1, :, :-1],
                ),
                dim=1,
            ).reshape(batch * sequence, width)
            normalized = F.normalize(source, dim=-1)
            assignments = (
                normalized @ self.associative_latent_codebook[layer - 1]
            ).argmax(dim=-1)
            counts = torch.zeros(
                code_count,
                device=raw_states.device,
                dtype=raw_states.dtype,
            )
            counts.scatter_add_(
                0,
                assignments,
                torch.ones_like(assignments, dtype=raw_states.dtype),
            )
            sums = torch.zeros(
                width,
                code_count,
                device=raw_states.device,
                dtype=raw_states.dtype,
            )
            sums.scatter_add_(
                1,
                assignments.view(1, -1).expand(width, -1),
                normalized.transpose(0, 1),
            )
            means = sums / counts.clamp_min(1).unsqueeze(0)
            active = counts > 0
            blended = F.normalize(
                (1.0 - rate) * self.associative_latent_codebook[layer - 1]
                + rate * means,
                dim=0,
            )
            codebook = self.associative_latent_codebook[layer - 1]
            codebook.copy_(torch.where(active.unsqueeze(0), blended, codebook))
            self.associative_latent_codebook_counts[layer - 1].add_(counts)

    def training_feedback_parameter_count(self) -> int:
        return 0

    @staticmethod
    def global_backprops() -> int:
        return 0

    @staticmethod
    def forward_weight_transpose_reads() -> int:
        return 0

    def _uses_recurrent_feature_context(self) -> bool:
        return bool(
            self.config.bounded_recurrent_feature_context
            or self.config.identity_preserving_context_modulation
        )

    def empty_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> NativePCState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        state_device = device or self.device
        state_dtype = dtype or self.dtype
        recurrent_statistics_dtype = (
            torch.float32
            if self.config.bf16_training_fp32_recurrent_statistics
            else state_dtype
        )
        layers = torch.zeros(
            self.config.depth,
            batch_size,
            self.config.width,
            device=state_device,
            dtype=state_dtype,
        )
        manifold_depth_bus = (
            torch.zeros(
                self.config.depth,
                batch_size,
                2,
                self.config.width,
                device=state_device,
                dtype=state_dtype,
            )
            if self.config.bounded_manifold_depth_bus
            else None
        )
        nonlinear_matrix_state = (
            torch.zeros(
                batch_size,
                NONLINEAR_MATRIX_HEADS,
                NONLINEAR_MATRIX_KEY_WIDTH,
                NONLINEAR_MATRIX_VALUE_WIDTH,
                device=state_device,
                dtype=state_dtype,
            )
            if self.config.nonlinear_matrix_state_lane
            else None
        )
        recurrent_layers = (
            torch.zeros_like(layers)
            if (
                self.config.decoupled_bounded_memory_emission
                or self.config.causal_level_innovation_lifting
            )
            else None
        )
        feature_context = (
            torch.zeros_like(layers) if self._uses_recurrent_feature_context() else None
        )
        rank = self.config.associative_memory_rank
        memory_layers = self._associative_layer_count()
        value_width = self._associative_value_width()
        persistent_transition_adapter = (
            torch.zeros(
                self.config.depth,
                batch_size,
                self.config.persistent_transition_adapter_rank,
                device=state_device,
                dtype=state_dtype,
            )
            if self.config.persistent_transition_adapter_rank > 0
            else None
        )
        gdn2_memory = (
            torch.zeros(
                self.config.gdn2_mixer_layers,
                batch_size,
                self.config.gdn2_mixer_heads,
                self.config.gdn2_mixer_key_width,
                self.config.width // self.config.gdn2_mixer_heads,
                device=state_device,
                dtype=state_dtype,
            )
            if self.config.gdn2_mixer_layers > 0
            else None
        )
        factorization_memory = (
            torch.zeros(
                self.config.factorization_memory_layers,
                batch_size,
                self.config.factorization_memory_slots,
                self.config.factorization_memory_value_width,
                device=state_device,
                dtype=state_dtype,
            )
            if self.config.factorization_memory_layers > 0
            else None
        )
        bounded_attention_keys = (
            torch.zeros(
                self.config.bounded_attention_layers,
                batch_size,
                self.config.bounded_attention_window,
                self.config.bounded_attention_width,
                device=state_device,
                dtype=state_dtype,
            )
            if self.config.bounded_attention_layers > 0
            else None
        )
        bounded_attention_values = (
            torch.zeros_like(bounded_attention_keys)
            if bounded_attention_keys is not None
            else None
        )
        hgf_posterior_precision = (
            torch.ones(
                self.config.depth,
                batch_size,
                self.config.width,
                device=state_device,
                dtype=torch.float32,
            )
            if self.config.temporal_hgf_precision_carry
            else None
        )
        causal_fir_input_history = (
            torch.zeros(
                self.config.depth,
                CAUSAL_TERNARY_FIR_TAPS,
                batch_size,
                self.config.width,
                device=state_device,
                dtype=state_dtype,
            )
            if self.config.causal_ternary_fir_input_state
            else None
        )
        causal_fir_input_valid_count = (
            torch.zeros(
                self.config.depth,
                batch_size,
                device=state_device,
                dtype=torch.int32,
            )
            if self.config.causal_ternary_fir_input_state
            else None
        )
        latent_prediction_error = (
            torch.zeros(
                self.config.depth,
                batch_size,
                self.config.width,
                device=state_device,
                dtype=torch.float16,
            )
            if self.config.lagged_latent_error_settling_carrier
            else None
        )
        semantic_memory = (
            torch.zeros(
                1,
                batch_size,
                SEMANTIC_COUNT_SKETCH_TABLES * SEMANTIC_COUNT_SKETCH_BUCKETS,
                SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
                device=state_device,
                dtype=recurrent_statistics_dtype,
            )
            if self._uses_semantic_count_sketch_backoff()
            else None
        )
        semantic_normalizer = (
            torch.zeros(
                1,
                batch_size,
                SEMANTIC_COUNT_SKETCH_TABLES * SEMANTIC_COUNT_SKETCH_BUCKETS,
                device=state_device,
                dtype=recurrent_statistics_dtype,
            )
            if self._uses_semantic_count_sketch_backoff()
            else None
        )
        semantic_last_update = (
            torch.zeros(
                1,
                batch_size,
                SEMANTIC_COUNT_SKETCH_TABLES * SEMANTIC_COUNT_SKETCH_BUCKETS,
                device=state_device,
                dtype=torch.int32,
            )
            if self._uses_semantic_count_sketch_backoff()
            else None
        )
        semantic_context = (
            torch.zeros(
                batch_size,
                self.config.width,
                device=state_device,
                dtype=recurrent_statistics_dtype,
            )
            if self._uses_semantic_count_sketch_backoff()
            else None
        )
        semantic_codes = (
            torch.zeros(
                batch_size,
                SEMANTIC_COUNT_SKETCH_TABLES,
                device=state_device,
                dtype=torch.uint8,
            )
            if self._uses_semantic_count_sketch_backoff()
            else None
        )
        pair_recurrence_bloom = (
            torch.zeros(
                batch_size,
                PAIR_RECURRENCE_BLOOM_GENERATIONS,
                PAIR_RECURRENCE_BLOOM_PROBES,
                PAIR_RECURRENCE_BLOOM_BYTES,
                device=state_device,
                dtype=torch.uint8,
            )
            if self._uses_pair_miss_semantic_backoff()
            else None
        )
        if rank == 0:
            return NativePCState(
                layers=layers,
                manifold_depth_bus=manifold_depth_bus,
                nonlinear_matrix_state=nonlinear_matrix_state,
                recurrent_layers=recurrent_layers,
                feature_context=feature_context,
                associative_semantic_memory=semantic_memory,
                associative_semantic_normalizer=semantic_normalizer,
                associative_semantic_last_update=semantic_last_update,
                associative_semantic_context=semantic_context,
                associative_semantic_codes=semantic_codes,
                associative_pair_recurrence_bloom=pair_recurrence_bloom,
                persistent_transition_adapter=persistent_transition_adapter,
                gdn2_memory=gdn2_memory,
                factorization_memory=factorization_memory,
                bounded_attention_keys=bounded_attention_keys,
                bounded_attention_values=bounded_attention_values,
                hgf_posterior_precision=hgf_posterior_precision,
                causal_fir_input_history=causal_fir_input_history,
                causal_fir_input_valid_count=causal_fir_input_valid_count,
                latent_prediction_error=latent_prediction_error,
            )
        return NativePCState(
            layers=layers,
            manifold_depth_bus=manifold_depth_bus,
            nonlinear_matrix_state=nonlinear_matrix_state,
            recurrent_layers=recurrent_layers,
            feature_context=feature_context,
            associative_memory=torch.zeros(
                memory_layers,
                batch_size,
                rank,
                value_width,
                device=state_device,
                dtype=recurrent_statistics_dtype,
            ),
            associative_normalizer=torch.zeros(
                memory_layers,
                batch_size,
                rank,
                device=state_device,
                dtype=recurrent_statistics_dtype,
            ),
            associative_keys=(
                None
                if self.config.associative_memory_addressing
                in INDEXED_COUNT_SKETCH_ADDRESSINGS
                else torch.zeros(
                    memory_layers,
                    batch_size,
                    rank,
                    device=state_device,
                    dtype=recurrent_statistics_dtype,
                )
            ),
            associative_last_update=(
                torch.zeros(
                    memory_layers,
                    batch_size,
                    rank,
                    device=state_device,
                    dtype=torch.int32,
                )
                if self.config.associative_memory_addressing
                in LAZY_COUNT_SKETCH_ADDRESSINGS
                else None
            ),
            associative_context_tokens=(
                torch.full(
                    (batch_size,),
                    self.config.bos_token_id,
                    device=state_device,
                    dtype=torch.long,
                )
                if self.config.associative_memory_addressing
                in {
                    "token_bigram_hash",
                    "token_plus_bigram_hash",
                    "token_plus_context_bucket",
                    "indexed_lazy_multiscale_count_sketch",
                    *DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS,
                }
                else None
            ),
            associative_previous_context_tokens=(
                torch.full(
                    (batch_size,),
                    self.config.bos_token_id,
                    device=state_device,
                    dtype=torch.long,
                )
                if self.config.associative_memory_addressing
                in DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS
                else None
            ),
            associative_semantic_memory=semantic_memory,
            associative_semantic_normalizer=semantic_normalizer,
            associative_semantic_last_update=semantic_last_update,
            associative_semantic_context=semantic_context,
            associative_semantic_codes=semantic_codes,
            associative_pair_recurrence_bloom=pair_recurrence_bloom,
            persistent_transition_adapter=persistent_transition_adapter,
            gdn2_memory=gdn2_memory,
            factorization_memory=factorization_memory,
            bounded_attention_keys=bounded_attention_keys,
            bounded_attention_values=bounded_attention_values,
            hgf_posterior_precision=hgf_posterior_precision,
            causal_fir_input_history=causal_fir_input_history,
            causal_fir_input_valid_count=causal_fir_input_valid_count,
            latent_prediction_error=latent_prediction_error,
        )

    def inference_state_bytes(
        self,
        batch_size: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> int:
        element_size = torch.empty((), dtype=dtype or self.dtype).element_size()
        rank = self.config.associative_memory_rank
        memory_layers = self._associative_layer_count()
        value_width = self._associative_value_width()
        layer_values = self.config.depth * self.config.width
        manifold_depth_bus_values = (
            2 * layer_values if self.config.bounded_manifold_depth_bus else 0
        )
        nonlinear_matrix_state_values = (
            NONLINEAR_MATRIX_STATE_VALUES
            if self.config.nonlinear_matrix_state_lane
            else 0
        )
        recurrent_layer_values = (
            layer_values
            if (
                self.config.decoupled_bounded_memory_emission
                or self.config.causal_level_innovation_lifting
            )
            else 0
        )
        feature_context_values = (
            layer_values if self._uses_recurrent_feature_context() else 0
        )
        causal_fir_values = (
            CAUSAL_TERNARY_FIR_TAPS * layer_values
            if self.config.causal_ternary_fir_input_state
            else 0
        )
        causal_fir_count_bytes = (
            batch_size
            * self.config.depth
            * torch.empty((), dtype=torch.int32).element_size()
            if self.config.causal_ternary_fir_input_state
            else 0
        )
        latent_prediction_error_bytes = (
            batch_size
            * layer_values
            * torch.empty((), dtype=torch.float16).element_size()
            if self.config.lagged_latent_error_settling_carrier
            else 0
        )
        persistent_values = (
            self.config.depth * self.config.persistent_transition_adapter_rank
        )
        gdn2_values = (
            self.config.gdn2_mixer_layers
            * self.config.gdn2_mixer_heads
            * self.config.gdn2_mixer_key_width
            * (self.config.width // self.config.gdn2_mixer_heads)
            if self.config.gdn2_mixer_layers > 0
            else 0
        )
        factorization_values = (
            self.config.factorization_memory_layers
            * self.config.factorization_memory_slots
            * self.config.factorization_memory_value_width
        )
        bounded_attention_values = (
            2
            * self.config.bounded_attention_layers
            * self.config.bounded_attention_window
            * self.config.bounded_attention_width
        )
        memory_values = memory_layers * (
            rank * value_width
            + rank
            + (
                0
                if self.config.associative_memory_addressing
                in INDEXED_COUNT_SKETCH_ADDRESSINGS
                else rank
            )
        )
        context_bytes = (
            batch_size * torch.empty((), dtype=torch.long).element_size()
            if self.config.associative_memory_addressing
            in {
                "token_bigram_hash",
                "token_plus_bigram_hash",
                "token_plus_context_bucket",
                "indexed_lazy_multiscale_count_sketch",
                *DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS,
            }
            else 0
        )
        previous_context_bytes = (
            batch_size * torch.empty((), dtype=torch.long).element_size()
            if self.config.associative_memory_addressing
            in DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS
            else 0
        )
        lazy_timestamp_bytes = (
            batch_size
            * memory_layers
            * rank
            * torch.empty((), dtype=torch.int32).element_size()
            if self.config.associative_memory_addressing
            in LAZY_COUNT_SKETCH_ADDRESSINGS
            else 0
        )
        semantic_values = (
            SEMANTIC_COUNT_SKETCH_TABLES
            * SEMANTIC_COUNT_SKETCH_BUCKETS
            * (SEMANTIC_COUNT_SKETCH_VALUE_WIDTH + 1)
            + self.config.width
            if self._uses_semantic_count_sketch_backoff()
            else 0
        )
        semantic_timestamp_bytes = (
            batch_size
            * SEMANTIC_COUNT_SKETCH_TABLES
            * SEMANTIC_COUNT_SKETCH_BUCKETS
            * torch.empty((), dtype=torch.int32).element_size()
            if self._uses_semantic_count_sketch_backoff()
            else 0
        )
        semantic_code_bytes = (
            batch_size * SEMANTIC_COUNT_SKETCH_TABLES
            if self._uses_semantic_count_sketch_backoff()
            else 0
        )
        pair_recurrence_bloom_bytes = (
            batch_size
            * PAIR_RECURRENCE_BLOOM_GENERATIONS
            * PAIR_RECURRENCE_BLOOM_PROBES
            * PAIR_RECURRENCE_BLOOM_BYTES
            if self._uses_pair_miss_semantic_backoff()
            else 0
        )
        return (
            batch_size
            * (
                layer_values
                + manifold_depth_bus_values
                + nonlinear_matrix_state_values
                + recurrent_layer_values
                + feature_context_values
                + causal_fir_values
                + persistent_values
                + memory_values
                + gdn2_values
                + factorization_values
                + bounded_attention_values
                + semantic_values
            )
            * element_size
            + context_bytes
            + previous_context_bytes
            + lazy_timestamp_bytes
            + semantic_timestamp_bytes
            + semantic_code_bytes
            + pair_recurrence_bloom_bytes
            + causal_fir_count_bytes
            + latent_prediction_error_bytes
        )

    def estimated_flops_per_token(self, batch_size: int = 1) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        depth = self.config.depth
        width = self.config.width
        ffn_width = self.config.ffn_width
        transition_macs = depth * 2 * width * width
        causal_fir_macs = (
            depth * CAUSAL_TERNARY_FIR_TAPS * width
            if self.config.causal_ternary_fir_input_state
            else 0
        )
        latent_error_macs = (
            depth * width * width
            if self.config.lagged_latent_error_settling_carrier
            else 0
        )
        latent_error_elementwise_flops = (
            depth * 8 * width
            if self.config.lagged_latent_error_settling_carrier
            else 0
        )
        decoupled_write_macs = (
            depth * width * width if self.transition_write_gate is not None else 0
        )
        manifold_depth_bus_macs = (
            depth * 2 * width * 5
            if self.config.bounded_manifold_depth_bus
            else 0
        )
        depth_attention_residual_flops = (
            sum(
                7 * source_count * width
                + 4 * source_count
                + 3 * width
                + 1
                for source_count in range(2, depth + 1)
            )
            if self.config.bounded_depth_attention_residual
            else 0
        )
        stable_latent_expert_macs = (
            width * 32
            + width * 8
            + 2 * 3 * 32 * 32
            + 32 * width
            if self.config.stable_latent_expert_branch
            else 0
        )
        stable_latent_expert_elementwise_flops = (
            1_258 if self.config.stable_latent_expert_branch else 0
        )
        predictive_read_gate_macs = (
            width * 32
            if self.config.token_conditioned_predictive_read_gate
            else 0
        )
        predictive_read_gate_elementwise_flops = (
            14 * 32
            if self.config.token_conditioned_predictive_read_gate
            else 0
        )
        projected_observation_macs = (
            width * width
            if self.config.projected_self_bilinear_token_observation
            else 0
        )
        projected_observation_elementwise_flops = (
            24 * width
            if self.config.projected_self_bilinear_token_observation
            else 0
        )
        nonlinear_matrix_state_macs = (
            self.config.width * NONLINEAR_MATRIX_PROJECTION_WIDTH
            + NONLINEAR_MATRIX_HEADS
            * NONLINEAR_MATRIX_KEY_WIDTH
            * NONLINEAR_MATRIX_VALUE_WIDTH
            * NONLINEAR_MATRIX_VALUE_WIDTH
            + NONLINEAR_MATRIX_STATE_VALUES
            if self.config.nonlinear_matrix_state_lane
            else 0
        )
        nonlinear_matrix_state_flops = (
            14 * NONLINEAR_MATRIX_STATE_VALUES
            + 12 * self.config.width
            + 8 * NONLINEAR_MATRIX_HEADS
            if self.config.nonlinear_matrix_state_lane
            else 0
        )
        persistent_transition_macs = (
            depth * 2 * width * self.config.persistent_transition_adapter_rank
        )
        selective_temporal_phase_macs = (
            depth * self.config.selective_temporal_phase_rank * (width + width // 2)
        )
        pairwise_delta_erase_macs = (
            depth * self.config.pairwise_delta_erase_rank * (width + 2 * width)
        )
        block_delta_macs = 0
        if self.block_delta_down is not None:
            memory_layers = self._associative_layer_count()
            projection_rank = self.config.block_delta_memory_projection_rank
            key_channels = self.config.associative_memory_rank
            heads = self.config.block_delta_memory_heads
            key_width = self.config.block_delta_memory_key_width
            value_width = self._associative_value_width()
            block_delta_macs = memory_layers * (
                width * projection_rank
                + projection_rank * (4 * key_channels + width)
                + heads * (2 * key_width * key_width + 3 * key_width * value_width)
            )
        gdn2_macs = 0
        if self.gdn2_query is not None:
            mixer_layers = self.config.gdn2_mixer_layers
            heads = self.config.gdn2_mixer_heads
            key_width = self.config.gdn2_mixer_key_width
            key_channels = heads * key_width
            value_width = width // heads
            gdn2_macs = mixer_layers * (
                3 * width * key_channels
                + 3 * width * width
                + 3 * width * value_width
                + value_width * key_channels
                + heads * (2 * key_width * key_width + 3 * key_width * value_width)
            )
        factorization_macs = self.config.factorization_memory_layers * (
            width * self.config.factorization_memory_slots
            + 2 * width
            + 2 * width * self.config.factorization_memory_value_width
        )
        factorization_elementwise_flops = self.config.factorization_memory_layers * (
            8
            * self.config.factorization_memory_slots
            * self.config.factorization_memory_value_width
            + 5 * self.config.factorization_memory_slots
            + 4
        )
        bounded_attention_macs = self.config.bounded_attention_layers * (
            4 * width * self.config.bounded_attention_width
            + 2
            * self.config.bounded_attention_window
            * self.config.bounded_attention_width
        )
        bounded_attention_elementwise_flops = self.config.bounded_attention_layers * (
            5 * self.config.bounded_attention_window
            + (
                6 * self.config.bounded_attention_width
                if self.config.bounded_attention_rotary
                else 0
            )
        )
        semantic_utility_rank = (
            self.config.bounded_attention_causal_semantic_utility_rank
        )
        bounded_attention_semantic_utility_macs = (
            self.config.bounded_attention_layers
            * 2
            * width
            * semantic_utility_rank
        )
        bounded_attention_utility_flops = 0
        if self.config.bounded_attention_utility_gate:
            bounded_attention_utility_flops = self.config.bounded_attention_layers * (
                6 * width + 3 * self.config.bounded_attention_window + 20
            )
        elif self.config.bounded_attention_channel_preserving_semantic_utility_gate:
            bounded_attention_utility_flops = self.config.bounded_attention_layers * (
                6 * width + 11 * semantic_utility_rank + 12
            )
        elif semantic_utility_rank > 0:
            bounded_attention_utility_flops = self.config.bounded_attention_layers * (
                6 * width + 3 * semantic_utility_rank + 12
            )
        ffn_macs = depth * (3 if self.config.swiglu_ffn else 2) * width * ffn_width
        ffn_elementwise_flops = depth * ffn_width if self.config.swiglu_ffn else 0
        if self.config.deterministic_ternary_ffn_experts:
            ffn_macs *= 4
            ffn_elementwise_flops *= 4
        elif self.config.shared_features_routed_ternary_ffn_output:
            ffn_macs += 3 * depth * width * ffn_width
        elif self.config.deterministic_ternary_ffn_residual_experts:
            ffn_macs *= 2
            ffn_elementwise_flops *= 2
        if self.config.alternating_triangular_coupling:
            transition_macs //= 2
            decoupled_write_macs //= 2
            ffn_macs //= 2
            ffn_elementwise_flops //= 2
        butterfly_flops = (
            4
            * depth
            * width
            * (
                int(math.log2(width))
                + 1
                + int(self.config.layer_rotating_bent_butterfly_diffusion)
            )
            if self.config.butterfly_channel_diffusion
            else 0
        )
        settling_macs = self.config.settling_rounds * depth * 3 * width * width
        visit_adapter_macs = (
            (self.config.settling_rounds - 1)
            * depth
            * 2
            * width
            * self.config.visit_low_rank_adapter_rank
        )
        dual_stream_visit_adapter_macs = (
            (self.config.settling_rounds - 1)
            * depth
            * 6
            * width
            * self.config.dual_stream_visit_adapter_rank
        )
        temporal_rotation_flops = (
            depth * 3 * width if self.temporal_rotation is not None else 0
        )
        decoupled_emission_flops = (
            depth * width
            if self.config.decoupled_bounded_memory_emission
            else 0
        )
        coefficient_normalization_flops = (
            2 * depth * width
            if self.config.coefficient_normalized_emission
            else 0
        )
        quadrature_composition_flops = (
            2 * depth * width
            if self.config.quadrature_preserving_recurrent_state
            else 0
        )
        causal_lifting_flops = (
            4 * depth * width
            if self.config.causal_level_innovation_lifting
            else 0
        )
        manifold_depth_bus_flops = (
            depth * (24 * width + 20)
            if self.config.bounded_manifold_depth_bus
            else 0
        )
        feature_context_flops = 0
        if self._uses_recurrent_feature_context():
            feature_context_flops = (
                depth
                * width
                * (
                    (12 if self.config.bounded_recurrent_feature_context else 9)
                    + (3 if self.temporal_rotation is not None else 0)
                )
            )
        readout_macs = (
            depth * width * width if self.readout_projection is not None else 0
        )
        orthogonal_depth_readout_flops = (
            depth * width * (int(math.log2(width)) + 1)
            + (depth - 1) * width
            if self.config.fixed_orthogonal_depth_readout
            else 0
        )
        confidence_flops = (
            2 * width + 2 * 16 + self.config.vocab_size + 3
            if self.confidence_readout_weights is not None
            else 0
        )
        memory_layers = self._associative_layer_count()
        value_width = self._associative_value_width()
        compression_macs = 0
        if self.associative_value_encoder is not None:
            decoder_value_width = value_width * (
                2
                if self.config.associative_memory_addressing
                in MULTISCALE_COUNT_SKETCH_ADDRESSINGS
                else 1
            )
            if (
                self.config.associative_memory_addressing
                in DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS
            ):
                decoder_value_width = 4 * value_width
            compression_macs = width * (value_width + decoder_value_width)
        associative_macs = (
            memory_layers
            * (
                (
                    width * self.config.associative_memory_rank
                    if self.config.associative_memory_addressing == "dense"
                    else 0
                )
                + 3 * self.config.associative_memory_rank * value_width
                + compression_macs
            )
            + (
                (memory_layers - 1) * width * self.config.associative_memory_latent_bits
                if self.config.associative_memory_addressing == "token_plus_latent_hash"
                else 0
            )
            + (
                (memory_layers - 1)
                * width
                * self.config.associative_memory_codebook_size
                if self.config.associative_memory_addressing
                == "token_plus_latent_codebook"
                else 0
            )
            if self.config.associative_memory_rank > 0
            else 0
        )
        semantic_context_flops = 0
        if self._uses_semantic_count_sketch_backoff():
            semantic_rank = (
                SEMANTIC_COUNT_SKETCH_TABLES * SEMANTIC_COUNT_SKETCH_BUCKETS
            )
            semantic_decoder_width = (
                width
                if self._uses_disjoint_timescale_semantic_channels()
                else 2 * width
            )
            associative_macs += (
                3 * semantic_rank * SEMANTIC_COUNT_SKETCH_VALUE_WIDTH
                + width * SEMANTIC_COUNT_SKETCH_VALUE_WIDTH
                + semantic_decoder_width * SEMANTIC_COUNT_SKETCH_VALUE_WIDTH
            )
            semantic_workspace_width = (
                1 << (width - 1).bit_length()
                if self.config.padded_semantic_hadamard_hash
                else width
            )
            semantic_context_flops = semantic_workspace_width * (
                int(math.log2(semantic_workspace_width)) + 7
            )
        pair_recurrence_flops = (
            64 if self._uses_pair_miss_semantic_backoff() else 0
        )
        if self.block_delta_down is not None:
            associative_macs = block_delta_macs
        output_macs = (
            0
            if self.config.fixed_ternary_hadamard_lexical_predictor
            else width * width
        ) + width * self.config.vocab_size
        lexical_hadamard_flops = (
            width * (int(math.log2(width)) + 1)
            if self.config.fixed_ternary_hadamard_lexical_predictor
            else 0
        )
        if self.context_routed_lexical_residual_decoders is not None:
            output_macs += width * self.config.vocab_size
        if self.bounded_multiplicative_lexical_decoder is not None:
            output_macs += width * self.config.vocab_size
        if self.signed_square_lexical_decoder is not None:
            output_macs += width * self.config.vocab_size
        if self.cross_channel_quadratic_lexical_decoder is not None:
            output_macs += width * self.config.vocab_size
        if self.depth_innovation_predictor is not None:
            output_macs += width * width
        if self.swiglu_lexical_decoder is not None:
            output_macs += ffn_width * self.config.vocab_size
        if self.config.depth_wide_ternary_lexical_endpoint:
            output_macs += (self.config.depth - 1) * (
                width * width + width * self.config.vocab_size
            )
        elif (
            self.config.feature_ready_depth_wide_lexical_residual
            or self.config.cozero_reciprocal_lexical_adapters
        ):
            output_macs += (
                (self.config.depth - 1) * width * self.config.vocab_size
            )
        matrix_flops = (
            2
            * batch_size
            * (
                transition_macs
                + causal_fir_macs
                + latent_error_macs
                + decoupled_write_macs
                + manifold_depth_bus_macs
                + stable_latent_expert_macs
                + predictive_read_gate_macs
                + projected_observation_macs
                + nonlinear_matrix_state_macs
                + persistent_transition_macs
                + selective_temporal_phase_macs
                + pairwise_delta_erase_macs
                + gdn2_macs
                + factorization_macs
                + bounded_attention_macs
                + bounded_attention_semantic_utility_macs
                + ffn_macs
                + settling_macs
                + visit_adapter_macs
                + dual_stream_visit_adapter_macs
                + readout_macs
                + associative_macs
                + output_macs
            )
        )
        return matrix_flops + batch_size * (
            temporal_rotation_flops
            + decoupled_emission_flops
            + coefficient_normalization_flops
            + quadrature_composition_flops
            + causal_lifting_flops
            + manifold_depth_bus_flops
            + depth_attention_residual_flops
            + stable_latent_expert_elementwise_flops
            + predictive_read_gate_elementwise_flops
            + projected_observation_elementwise_flops
            + nonlinear_matrix_state_flops
            + feature_context_flops
            + factorization_elementwise_flops
            + bounded_attention_elementwise_flops
            + bounded_attention_utility_flops
            + ffn_elementwise_flops
            + butterfly_flops
            + confidence_flops
            + semantic_context_flops
            + pair_recurrence_flops
            + orthogonal_depth_readout_flops
            + lexical_hadamard_flops
            + latent_error_elementwise_flops
        )

    def _stable_latent_expert_residual(
        self,
        source: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        from pfc.phase493_stable_latent_experts import (
            STABLE_LATENT_EXPERT_HIDDEN,
            STABLE_LATENT_GATE_BETA,
            STABLE_LATENT_TOP_K,
            STABLE_LATENT_UP_BETA,
            STABLE_LATENT_WIDTH,
            bounded_situ,
            bounded_tanh_branch,
            selected_mixture_weights,
            stable_topk_routes,
        )

        parameters = (
            self.stable_latent_down,
            self.stable_latent_router,
            self.stable_latent_expert_gate,
            self.stable_latent_expert_up,
            self.stable_latent_expert_down,
            self.stable_latent_norm,
            self.stable_latent_output,
        )
        if any(parameter is None for parameter in parameters):
            raise RuntimeError("stable latent expert branch is disabled")
        if self.stable_latent_router_bias is None:
            raise RuntimeError("stable latent router bias is missing")
        if source.shape[-1] != self.config.width:
            raise ValueError("stable latent expert source width differs")
        (
            latent_down,
            router,
            expert_gate,
            expert_up,
            expert_down,
            latent_norm,
            latent_output,
        ) = parameters
        assert latent_down is not None
        assert router is not None
        assert expert_gate is not None
        assert expert_up is not None
        assert expert_down is not None
        assert latent_norm is not None
        assert latent_output is not None

        latent = source @ latent_down
        raw_scores = torch.sigmoid(source @ router)
        route_ids = stable_topk_routes(
            raw_scores,
            self.stable_latent_router_bias,
        )
        weights = selected_mixture_weights(raw_scores, route_ids)
        flat_latent = latent.reshape(-1, STABLE_LATENT_WIDTH)
        flat_routes = route_ids.reshape(-1, STABLE_LATENT_TOP_K)
        active_latent = (
            flat_latent.unsqueeze(1)
            .expand(-1, STABLE_LATENT_TOP_K, -1)
            .reshape(-1, STABLE_LATENT_WIDTH)
        )
        active_routes = flat_routes.reshape(-1)
        gate_pre = torch.bmm(
            active_latent.unsqueeze(1),
            expert_gate[active_routes],
        ).reshape(-1, STABLE_LATENT_EXPERT_HIDDEN)
        up_pre = torch.bmm(
            active_latent.unsqueeze(1),
            expert_up[active_routes],
        ).reshape(-1, STABLE_LATENT_EXPERT_HIDDEN)
        hidden = bounded_situ(
            gate_pre,
            STABLE_LATENT_GATE_BETA,
        ) * bounded_tanh_branch(
            up_pre,
            STABLE_LATENT_UP_BETA,
        )
        expert_values = torch.bmm(
            hidden.unsqueeze(1),
            expert_down[active_routes],
        ).reshape(
            *source.shape[:-1],
            STABLE_LATENT_TOP_K,
            STABLE_LATENT_WIDTH,
        )
        aggregate = (weights.unsqueeze(-1) * expert_values).sum(dim=-2)
        normalized = (
            aggregate
            * torch.rsqrt(
                aggregate.square().mean(dim=-1, keepdim=True)
                + self.config.rms_eps
            )
            * latent_norm
        )
        residual = normalized @ latent_output
        if torch.is_grad_enabled():
            self._stable_latent_last_router_scores = raw_scores.detach()
            self._stable_latent_last_route_ids = route_ids.detach()
        return residual, route_ids, weights

    @torch.no_grad()
    def commit_stable_latent_router_bias(self) -> tuple[Tensor, Tensor] | None:
        from pfc.phase493_stable_latent_experts import (
            STABLE_LATENT_EXPERTS,
            next_quantile_router_bias,
        )

        if self.stable_latent_router_bias is None:
            return None
        scores = self._stable_latent_last_router_scores
        routes = self._stable_latent_last_route_ids
        if scores is None or routes is None:
            raise RuntimeError("stable latent routing diagnostics are missing")
        route_counts = torch.bincount(
            routes.reshape(-1),
            minlength=STABLE_LATENT_EXPERTS,
        )
        next_bias = next_quantile_router_bias(
            scores,
            self.stable_latent_router_bias,
        )
        self.stable_latent_router_bias.copy_(
            next_bias.to(
                device=self.stable_latent_router_bias.device,
                dtype=torch.float32,
            )
        )
        self._stable_latent_last_router_scores = None
        self._stable_latent_last_route_ids = None
        return route_counts, next_bias

    def _depth_attention_residual(
        self,
        sources: Tensor,
        *,
        layer: int,
    ) -> tuple[Tensor, Tensor]:
        if (
            self.depth_attention_queries is None
            or self.depth_attention_residual_logits is None
        ):
            raise RuntimeError("bounded depth attention is disabled")
        if not 1 <= layer < self.config.depth:
            raise ValueError("depth-attention destination layer is out of range")
        if (
            sources.shape[0] != layer + 1
            or sources.shape[-1] != self.config.width
        ):
            raise ValueError("depth-attention source bank has invalid shape")
        normalized = sources * torch.rsqrt(
            sources.square().mean(dim=-1, keepdim=True)
            + self.config.rms_eps
        )
        query_shape = (
            1,
            *(1 for _ in range(sources.ndim - 2)),
            self.config.width,
        )
        scores = (
            normalized
            * self.depth_attention_queries[layer - 1].view(query_shape)
        ).sum(dim=-1)
        weights = torch.softmax(scores, dim=0)
        selected = (weights.unsqueeze(-1) * sources).sum(dim=0)
        parent = sources[-1]
        coefficient = (
            2.0
            * torch.sigmoid(self.depth_attention_residual_logits[layer - 1])
            - 1.0
        )
        return parent + coefficient * (selected - parent), weights

    def _depth_attention_step_inputs(
        self,
        observation: Tensor,
        prior_layers: Tensor,
    ) -> Tensor:
        if (
            prior_layers.shape[0] != self.config.depth
            or prior_layers.shape[1:] != observation.shape
        ):
            raise ValueError("depth-attention prior layers have invalid shape")
        rows = [observation]
        for layer in range(1, self.config.depth):
            sources = torch.cat(
                (observation.unsqueeze(0), prior_layers[:layer]),
                dim=0,
            )
            composed, _ = self._depth_attention_residual(
                sources,
                layer=layer,
            )
            rows.append(composed)
        return torch.stack(rows)

    def _manifold_depth_bus_maps(
        self,
        lower_streams: Tensor,
        *,
        layer: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if (
            self.manifold_depth_bus_controller is None
            or self.manifold_depth_bus_bias is None
        ):
            raise RuntimeError("bounded manifold depth bus is disabled")
        if lower_streams.shape[-2:] != (2, self.config.width):
            raise ValueError("bounded manifold lower streams have invalid shape")
        flattened = lower_streams.flatten(-2)
        normalized = flattened * torch.rsqrt(
            flattened.square().mean(dim=-1, keepdim=True)
            + self.config.rms_eps
        )
        if layer is None:
            if (
                normalized.ndim != 3
                or normalized.shape[0] != self.config.depth
            ):
                raise ValueError(
                    "layer-parallel manifold streams must have "
                    "[depth, batch, 2, width]"
                )
            logits = (
                torch.bmm(normalized, self.manifold_depth_bus_controller)
                + self.manifold_depth_bus_bias.unsqueeze(1)
            )
        else:
            if not 0 <= layer < self.config.depth:
                raise ValueError("bounded manifold layer is out of range")
            logits = (
                normalized @ self.manifold_depth_bus_controller[layer]
                + self.manifold_depth_bus_bias[layer]
            )
        pre = torch.softmax(logits[..., :2], dim=-1)
        post_logits = torch.stack(
            (
                logits[..., 2] + logits[..., 3],
                logits[..., 2] - logits[..., 3],
            ),
            dim=-1,
        )
        post = 2.0 * torch.sigmoid(post_logits)
        exchange = torch.sigmoid(logits[..., 4])
        core_input = lower_streams[..., 0, :] + pre[..., 1, None] * (
            lower_streams[..., 1, :] - lower_streams[..., 0, :]
        )
        return core_input, pre, post, exchange

    @staticmethod
    def _manifold_depth_bus_output(
        lower_streams: Tensor,
        core_input: Tensor,
        core_output: Tensor,
        post: Tensor,
        exchange: Tensor,
    ) -> Tensor:
        first = lower_streams[..., 0, :]
        second = lower_streams[..., 1, :]
        mixed_first = first + exchange[..., None] * (second - first)
        mixed_second = second + exchange[..., None] * (first - second)
        mixed = torch.stack((mixed_first, mixed_second), dim=-2)
        residual = core_output - core_input
        return (
            core_output.unsqueeze(-2)
            + (mixed - core_input.unsqueeze(-2))
            + (post - 1.0).unsqueeze(-1) * residual.unsqueeze(-2)
        )

    def _nonlinear_matrix_lane(
        self,
        normalized_source: Tensor,
        initial_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        parameters = (
            self.nonlinear_matrix_input_projection,
            self.nonlinear_matrix_transition,
            self.nonlinear_matrix_forget_bias,
            self.nonlinear_matrix_residual_weight,
            self.nonlinear_matrix_output_norm,
            self.nonlinear_matrix_output_scale,
        )
        if any(parameter is None for parameter in parameters):
            raise RuntimeError("nonlinear matrix parameters are missing")
        (
            input_projection,
            transition,
            forget_bias,
            residual_weight,
            output_norm,
            output_scale,
        ) = parameters
        if (
            self.config.fused_nonlinear_matrix_decode
            and normalized_source.ndim == 2
            and normalized_source.is_cuda
            and normalized_source.dtype == torch.float16
            and initial_state.dtype == torch.float16
            and torch.is_inference_mode_enabled()
        ):
            from pfc.phase388_fused_matrix_decode import (
                fused_nonlinear_matrix_decode,
            )

            return fused_nonlinear_matrix_decode(
                normalized_source,
                initial_state,
                input_projection,
                transition,
                forget_bias,
                residual_weight,
                output_norm,
                output_scale,
                rms_eps=self.config.rms_eps,
            )
        return nonlinear_matrix_state_sequence(
            normalized_source,
            initial_state,
            input_projection,
            transition,
            forget_bias,
            residual_weight,
            output_norm,
            output_scale,
            rms_eps=self.config.rms_eps,
        )

    def _ffn_hidden(self, projected: Tensor) -> Tensor:
        if not self.config.swiglu_ffn:
            return F.silu(projected)
        if self.config.bilinear_hebbian_ffn:
            from pfc.phase783_bilinear_hebbian_transition import (
                bilinear_hebbian_terms,
            )

            return bilinear_hebbian_terms(projected)[0]
        gate, value = projected.chunk(2, dim=-1)
        hidden = F.silu(gate) * value
        if self.config.pairwise_competitive_swiglu:
            from pfc.phase684_pairwise_competitive_swiglu import (
                pairwise_competitive_swiglu,
            )

            hidden = pairwise_competitive_swiglu(hidden)
        return hidden

    def _ffn_matmul(self, source: Tensor, weight: Tensor) -> Tensor:
        if (
            not self.config.intrinsic_ternary_reciprocal_ffn
            or not source.is_cuda
            or source.dtype != torch.float32
            or weight.dtype != torch.float32
        ):
            return torch.matmul(source, weight)
        from pfc.phase655_triton_intrinsic_reciprocal import (
            triton_intrinsic_reciprocal_matmul,
        )

        if weight.ndim == 2:
            return triton_intrinsic_reciprocal_matmul(source, weight)
        if (
            weight.ndim == 3
            and source.ndim >= 3
            and source.shape[0] == weight.shape[0]
        ):
            return torch.stack(
                tuple(
                    triton_intrinsic_reciprocal_matmul(source[layer], weight[layer])
                    for layer in range(weight.shape[0])
                )
            )
        raise ValueError("intrinsic reciprocal batched matrix layout differs")

    def _mandatory_full_bandwidth_observation(
        self,
        observation: Tensor,
        top_states: Tensor,
        initial_top: Tensor,
        *,
        initial_active: bool | Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not self.config.mandatory_full_bandwidth_bottom_clamp:
            return observation, {}
        if self.full_bandwidth_value is None or self.full_bandwidth_gate is None:
            raise RuntimeError("full-bandwidth maps are missing")
        if observation.ndim != 3 or top_states.shape != observation.shape:
            raise ValueError("full-bandwidth sequence tensors must share [B,T,D]")
        if initial_top.shape != observation[:, 0].shape:
            raise ValueError("full-bandwidth initial top state shape differs")
        previous_top = torch.cat(
            (initial_top.unsqueeze(1), top_states[:, :-1]),
            dim=1,
        )
        normalized_top = _rms_norm(
            previous_top,
            self.state_norm[-1, 0],
            self.config.rms_eps,
        )
        normalized_token = _rms_norm(
            observation,
            self.state_norm[0, 0],
            self.config.rms_eps,
        )
        state_code = None
        if self.config.mandatory_ternary_state_carrier:
            fused, value, gate, state_code = mandatory_ternary_state_fusion(
                normalized_top,
                normalized_token,
                self.full_bandwidth_value,
                self.full_bandwidth_gate,
            )
            value_source = state_code
        else:
            fused, value, gate = mandatory_full_bandwidth_fusion(
                normalized_top,
                normalized_token,
                self.full_bandwidth_value,
                self.full_bandwidth_gate,
            )
            value_source = normalized_top
        active = torch.ones(
            (*observation.shape[:-1], 1),
            device=observation.device,
            dtype=torch.bool,
        )
        if isinstance(initial_active, Tensor):
            if initial_active.numel() != 1:
                raise ValueError("full-bandwidth initial activity must be scalar")
            first_active = initial_active.to(device=observation.device, dtype=torch.bool)
            first_active = first_active.reshape(1, 1, 1).expand(
                observation.shape[0], 1, 1
            )
        else:
            first_active = torch.full(
                (observation.shape[0], 1, 1),
                bool(initial_active),
                device=observation.device,
                dtype=torch.bool,
            )
        active = torch.cat((first_active, active[:, 1:]), dim=1)
        result = torch.where(active, fused, observation)
        cache = {
            "previous_top": previous_top,
            "normalized_previous_top": normalized_top,
            "value_source": value_source,
            "normalized_token": normalized_token,
            "value": value,
            "gate": gate,
            "active": active,
        }
        if state_code is not None:
            cache["state_code"] = state_code
        return result, cache

    def _residual_deep_state_recirculation_observation(
        self,
        observation: Tensor,
        top_states: Tensor,
        initial_top: Tensor,
        *,
        initial_active: bool | Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not self.config.residual_deep_state_recirculation:
            return observation, {}
        if observation.ndim != 3 or top_states.shape != observation.shape:
            raise ValueError("recirculation sequence tensors must share [B,T,D]")
        if initial_top.shape != observation[:, 0].shape:
            raise ValueError("recirculation initial top state shape differs")
        previous_top = torch.cat(
            (initial_top.unsqueeze(1), top_states[:, :-1]),
            dim=1,
        )
        active = torch.ones(
            (*observation.shape[:-1], 1),
            device=observation.device,
            dtype=torch.bool,
        )
        if isinstance(initial_active, Tensor):
            if initial_active.numel() != 1:
                raise ValueError("recirculation initial activity must be scalar")
            first_active = initial_active.to(
                device=observation.device,
                dtype=torch.bool,
            ).reshape(1, 1, 1).expand(observation.shape[0], 1, 1)
        else:
            first_active = torch.full(
                (observation.shape[0], 1, 1),
                bool(initial_active),
                device=observation.device,
                dtype=torch.bool,
            )
        active = torch.cat((first_active, active[:, 1:]), dim=1)
        selector = getattr(self, "ternary_recirculation_selector", None)
        selector_active = bool(
            selector is not None
            and getattr(self, "_ternary_vector_recirculation_active", False)
        )
        adaptive_enabled = bool(
            getattr(self.config, "causal_diagonal_adaptive_recirculation", False)
        )
        adaptive_controllers = (
            self.adaptive_recirculation_controllers()
            if adaptive_enabled
            else ()
        )
        if (
            self.config.fused_residual_recirculation_decode
            and torch.is_inference_mode_enabled()
            and observation.is_cuda
            and observation.dtype == torch.float32
            and observation.shape[1] == 1
            and observation.is_contiguous()
            and previous_top.is_contiguous()
            and active.is_contiguous()
        ):
            if adaptive_enabled:
                from pfc.phase722_triton_adaptive_recirculation_decode import (
                    fused_causal_diagonal_adaptive_recirculation_decode,
                )

                result = fused_causal_diagonal_adaptive_recirculation_decode(
                    observation,
                    previous_top,
                    active,
                    adaptive_controllers,
                )
            elif selector_active:
                from pfc.phase721_triton_vector_recirculation_decode import (
                    fused_ternary_vector_recirculation_decode,
                )

                result = fused_ternary_vector_recirculation_decode(
                    observation,
                    previous_top,
                    active,
                    selector,
                )
            else:
                from pfc.phase717_triton_residual_recirculation_decode import (
                    fused_residual_recirculation_decode,
                )

                result = fused_residual_recirculation_decode(
                    observation,
                    previous_top,
                    active,
                )
            return result, {
                "destination": observation,
                "previous_top": previous_top,
                "active": active,
                "fused_decode": torch.ones(
                    (), device=observation.device, dtype=torch.bool
                ),
            }
        if selector_active:
            if selector is None:
                raise RuntimeError("ternary recirculation selector is missing")
            return ternary_vector_recirculation(
                observation,
                previous_top,
                active,
                ternary_recirculation_codes(selector),
            )
        if adaptive_enabled:
            return causal_diagonal_adaptive_recirculation(
                observation,
                previous_top,
                active,
                adaptive_recirculation_codes(*adaptive_controllers),
            )
        return residual_deep_state_recirculation(
            observation,
            previous_top,
            active,
        )

    def _state_conditioned_settling_observation(
        self,
        observation: Tensor,
        top_states: Tensor,
        initial_top: Tensor,
        *,
        initial_active: bool | Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.config.mandatory_full_bandwidth_bottom_clamp:
            return self._mandatory_full_bandwidth_observation(
                observation,
                top_states,
                initial_top,
                initial_active=initial_active,
            )
        return self._residual_deep_state_recirculation_observation(
            observation,
            top_states,
            initial_top,
            initial_active=initial_active,
        )

    def _validate_state(self, state: NativePCState, batch_size: int) -> None:
        if state.batch_size != batch_size:
            raise ValueError("state and token batch sizes must match")
        if state.token_count_cuda is not None and (
            state.token_count_cuda.shape != (1,)
            or state.token_count_cuda.dtype != torch.int64
            or state.token_count_cuda.device != state.layers.device
        ):
            raise ValueError("invalid CUDA graph token-count state")
        expected_layers = (self.config.depth, batch_size, self.config.width)
        if state.layers.shape != expected_layers:
            raise ValueError("invalid state shape")
        if self.config.temporal_hgf_precision_carry:
            if state.hgf_posterior_precision is None:
                raise ValueError("temporal HGF posterior precision is missing")
            if state.hgf_posterior_precision.shape != expected_layers:
                raise ValueError("invalid temporal HGF posterior precision shape")
            if state.hgf_posterior_precision.dtype != torch.float32:
                raise ValueError("temporal HGF posterior precision must be FP32")
            if state.hgf_posterior_precision.device != state.layers.device:
                raise ValueError("temporal HGF precision device differs from state")
        elif state.hgf_posterior_precision is not None:
            raise ValueError("disabled temporal HGF carry must have no precision state")
        if self.config.causal_ternary_fir_input_state:
            expected_fir_history = (
                self.config.depth,
                CAUSAL_TERNARY_FIR_TAPS,
                batch_size,
                self.config.width,
            )
            if state.causal_fir_input_history is None:
                raise ValueError("causal FIR input history is missing")
            if state.causal_fir_input_history.shape != expected_fir_history:
                raise ValueError("invalid causal FIR input-history shape")
            if state.causal_fir_input_history.dtype != state.layers.dtype:
                raise ValueError("causal FIR input history dtype differs from state")
            if state.causal_fir_input_history.device != state.layers.device:
                raise ValueError("causal FIR input history device differs from state")
            if state.causal_fir_input_valid_count is None:
                raise ValueError("causal FIR input valid count is missing")
            if state.causal_fir_input_valid_count.shape != (
                self.config.depth,
                batch_size,
            ):
                raise ValueError("invalid causal FIR input valid-count shape")
            if state.causal_fir_input_valid_count.dtype != torch.int32:
                raise ValueError("causal FIR input valid count must be int32")
            if state.causal_fir_input_valid_count.device != state.layers.device:
                raise ValueError("causal FIR input valid-count device differs")
        elif (
            state.causal_fir_input_history is not None
            or state.causal_fir_input_valid_count is not None
        ):
            raise ValueError("disabled causal FIR input state must be absent")
        if self.config.lagged_latent_error_settling_carrier:
            if state.latent_prediction_error is None:
                raise ValueError("latent prediction-error state is missing")
            if state.latent_prediction_error.shape != expected_layers:
                raise ValueError("invalid latent prediction-error state shape")
            if state.latent_prediction_error.dtype != torch.float16:
                raise ValueError("latent prediction-error state must be FP16")
            if state.latent_prediction_error.device != state.layers.device:
                raise ValueError("latent prediction-error state device differs")
        elif state.latent_prediction_error is not None:
            raise ValueError("disabled latent prediction-error state must be absent")
        if self.config.bounded_manifold_depth_bus:
            expected_depth_bus = (
                self.config.depth,
                batch_size,
                2,
                self.config.width,
            )
            if state.manifold_depth_bus is None:
                raise ValueError("bounded manifold depth-bus state is missing")
            if state.manifold_depth_bus.shape != expected_depth_bus:
                raise ValueError("invalid bounded manifold depth-bus state shape")
        elif state.manifold_depth_bus is not None:
            raise ValueError(
                "disabled bounded manifold depth bus must have no state"
            )
        expected_nonlinear_matrix_state = (
            batch_size,
            NONLINEAR_MATRIX_HEADS,
            NONLINEAR_MATRIX_KEY_WIDTH,
            NONLINEAR_MATRIX_VALUE_WIDTH,
        )
        if self.config.nonlinear_matrix_state_lane:
            if state.nonlinear_matrix_state is None:
                raise ValueError("nonlinear matrix state is missing")
            if state.nonlinear_matrix_state.shape != expected_nonlinear_matrix_state:
                raise ValueError("invalid nonlinear matrix state shape")
        elif state.nonlinear_matrix_state is not None:
            raise ValueError("disabled nonlinear matrix lane must have no state")
        if (
            self.config.decoupled_bounded_memory_emission
            or self.config.causal_level_innovation_lifting
        ):
            if state.recurrent_layers is None:
                raise ValueError("separate recurrent layer state is missing")
            if state.recurrent_layers.shape != expected_layers:
                raise ValueError("invalid separate recurrent layer state shape")
        elif state.recurrent_layers is not None:
            raise ValueError("disabled separate recurrent state must be absent")
        if self._uses_recurrent_feature_context():
            if state.feature_context is None:
                raise ValueError("bounded recurrent feature context is missing")
            if state.feature_context.shape != expected_layers:
                raise ValueError("invalid bounded recurrent feature context shape")
        elif state.feature_context is not None:
            raise ValueError(
                "disabled bounded recurrent feature context must have no state"
            )
        persistent_rank = self.config.persistent_transition_adapter_rank
        if persistent_rank == 0:
            if state.persistent_transition_adapter is not None:
                raise ValueError(
                    "disabled persistent transition adapter must have no state"
                )
        elif state.persistent_transition_adapter is None:
            raise ValueError("persistent transition adapter state is missing")
        elif state.persistent_transition_adapter.shape != (
            self.config.depth,
            batch_size,
            persistent_rank,
        ):
            raise ValueError("invalid persistent transition adapter state shape")
        if self.config.gdn2_mixer_layers == 0:
            if state.gdn2_memory is not None:
                raise ValueError("disabled GDN2 mixer must have no state")
        elif state.gdn2_memory is None:
            raise ValueError("GDN2 mixer state is missing")
        elif state.gdn2_memory.shape != (
            self.config.gdn2_mixer_layers,
            batch_size,
            self.config.gdn2_mixer_heads,
            self.config.gdn2_mixer_key_width,
            self.config.width // self.config.gdn2_mixer_heads,
        ):
            raise ValueError("invalid GDN2 mixer state shape")
        if self.config.factorization_memory_layers == 0:
            if state.factorization_memory is not None:
                raise ValueError("disabled factorization memory must have no state")
        elif state.factorization_memory is None:
            raise ValueError("factorization memory state is missing")
        elif state.factorization_memory.shape != (
            self.config.factorization_memory_layers,
            batch_size,
            self.config.factorization_memory_slots,
            self.config.factorization_memory_value_width,
        ):
            raise ValueError("invalid factorization memory state shape")
        bounded_attention_state = (
            state.bounded_attention_keys,
            state.bounded_attention_values,
        )
        if self.config.bounded_attention_layers == 0:
            if any(tensor is not None for tensor in bounded_attention_state):
                raise ValueError("disabled bounded attention must have no state")
        elif any(tensor is None for tensor in bounded_attention_state):
            raise ValueError("bounded attention state is missing")
        else:
            expected_bounded_attention = (
                self.config.bounded_attention_layers,
                batch_size,
                self.config.bounded_attention_window,
                self.config.bounded_attention_width,
            )
            if any(
                tensor is None or tensor.shape != expected_bounded_attention
                for tensor in bounded_attention_state
            ):
                raise ValueError("invalid bounded attention state shape")
        rank = self.config.associative_memory_rank
        memory_layers = self._associative_layer_count()
        value_width = self._associative_value_width()
        memory_tensors = (
            state.associative_memory,
            state.associative_normalizer,
            state.associative_keys,
        )
        if rank == 0:
            if any(tensor is not None for tensor in memory_tensors):
                raise ValueError("disabled associative memory must have no state")
            if state.associative_last_update is not None:
                raise ValueError("disabled associative memory must have no timestamps")
            return
        if state.associative_memory is None:
            raise ValueError("associative memory state is missing")
        if state.associative_normalizer is None:
            raise ValueError("associative normalizer state is missing")
        indexed_lazy = (
            self.config.associative_memory_addressing
            in INDEXED_COUNT_SKETCH_ADDRESSINGS
        )
        if indexed_lazy:
            if state.associative_keys is not None:
                raise ValueError("indexed lazy memory must not store dense keys")
        elif state.associative_keys is None:
            raise ValueError("associative key state is missing")
        if state.associative_memory.shape != (
            memory_layers,
            batch_size,
            rank,
            value_width,
        ):
            raise ValueError("invalid associative memory shape")
        if state.associative_normalizer.shape != (
            memory_layers,
            batch_size,
            rank,
        ):
            raise ValueError("invalid associative normalizer shape")
        if state.associative_keys is not None and state.associative_keys.shape != (
            memory_layers,
            batch_size,
            rank,
        ):
            raise ValueError("invalid associative key shape")
        if self.config.associative_memory_addressing in LAZY_COUNT_SKETCH_ADDRESSINGS:
            if state.associative_last_update is None:
                raise ValueError("lazy associative timestamps are missing")
            if state.associative_last_update.shape != (
                memory_layers,
                batch_size,
                rank,
            ):
                raise ValueError("invalid lazy associative timestamp shape")
            if state.associative_last_update.dtype != torch.int32:
                raise ValueError("lazy associative timestamps must use int32")
        elif state.associative_last_update is not None:
            raise ValueError("eager associative memory must have no timestamps")
        if self.config.associative_memory_addressing in {
            "token_bigram_hash",
            "token_plus_bigram_hash",
            "token_plus_context_bucket",
            "indexed_lazy_multiscale_count_sketch",
            *DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS,
        }:
            if state.associative_context_tokens is None:
                raise ValueError("associative context token state is missing")
            if state.associative_context_tokens.shape != (batch_size,):
                raise ValueError("invalid associative context token shape")
        elif state.associative_context_tokens is not None:
            raise ValueError("context-free associative memory has context state")
        if self.config.associative_memory_addressing in DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS:
            if state.associative_previous_context_tokens is None:
                raise ValueError("dual-order previous context token state is missing")
            if state.associative_previous_context_tokens.shape != (batch_size,):
                raise ValueError("invalid dual-order previous context token shape")
        elif state.associative_previous_context_tokens is not None:
            raise ValueError("single-order associative memory has second context state")
        semantic_state = (
            state.associative_semantic_memory,
            state.associative_semantic_normalizer,
            state.associative_semantic_last_update,
            state.associative_semantic_context,
            state.associative_semantic_codes,
        )
        if not self._uses_semantic_count_sketch_backoff():
            if any(tensor is not None for tensor in semantic_state):
                raise ValueError("disabled semantic CountSketch must have no state")
        elif any(tensor is None for tensor in semantic_state):
            raise ValueError("semantic CountSketch state is missing")
        else:
            semantic_rank = (
                SEMANTIC_COUNT_SKETCH_TABLES * SEMANTIC_COUNT_SKETCH_BUCKETS
            )
            if state.associative_semantic_memory is None or (
                state.associative_semantic_memory.shape
                != (
                    1,
                    batch_size,
                    semantic_rank,
                    SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
                )
            ):
                raise ValueError("invalid semantic CountSketch memory shape")
            if state.associative_semantic_normalizer is None or (
                state.associative_semantic_normalizer.shape
                != (1, batch_size, semantic_rank)
            ):
                raise ValueError("invalid semantic CountSketch normalizer shape")
            if state.associative_semantic_last_update is None or (
                state.associative_semantic_last_update.shape
                != (1, batch_size, semantic_rank)
            ):
                raise ValueError("invalid semantic CountSketch timestamp shape")
            if state.associative_semantic_last_update.dtype != torch.int32:
                raise ValueError("semantic CountSketch timestamps must use int32")
            if state.associative_semantic_context is None or (
                state.associative_semantic_context.shape
                != (batch_size, self.config.width)
            ):
                raise ValueError("invalid semantic context shape")
            if state.associative_semantic_codes is None or (
                state.associative_semantic_codes.shape
                != (batch_size, SEMANTIC_COUNT_SKETCH_TABLES)
            ):
                raise ValueError("invalid semantic context code shape")
            if state.associative_semantic_codes.dtype != torch.uint8:
                raise ValueError("semantic context codes must use uint8")
        pair_recurrence_bloom = state.associative_pair_recurrence_bloom
        if self._uses_pair_miss_semantic_backoff():
            if pair_recurrence_bloom is None or pair_recurrence_bloom.shape != (
                batch_size,
                PAIR_RECURRENCE_BLOOM_GENERATIONS,
                PAIR_RECURRENCE_BLOOM_PROBES,
                PAIR_RECURRENCE_BLOOM_BYTES,
            ):
                raise ValueError("invalid pair recurrence Bloom state shape")
            if pair_recurrence_bloom.dtype != torch.uint8:
                raise ValueError("pair recurrence Bloom state must use uint8")
        elif pair_recurrence_bloom is not None:
            raise ValueError("disabled pair recurrence Bloom must have no state")

    def _associative_keys(
        self,
        source: Tensor,
        layer: int | None = None,
        token_ids: Tensor | None = None,
        context_token_ids: Tensor | None = None,
    ) -> Tensor:
        if self.config.associative_memory_rank == 0:
            raise RuntimeError("associative memory is disabled")
        if source.shape[-1] != self.config.width:
            raise ValueError("associative source must end in model width")
        if (
            self.config.associative_memory_addressing
            in MULTISCALE_COUNT_SKETCH_ADDRESSINGS
        ):
            if token_ids is None:
                raise ValueError("multiscale CountSketch addressing requires token ids")
            keys = self._multiscale_count_sketch_keys(
                token_ids,
                source.dtype,
                context_token_ids,
            )
            if layer is None:
                keys = keys.unsqueeze(0).expand(
                    self._associative_layer_count(),
                    *keys.shape,
                )
            return keys
        if self.config.associative_memory_addressing in {
            "token_hash",
            "token_bigram_hash",
            "token_plus_bigram_hash",
            "token_plus_context_bucket",
            "token_plus_latent_hash",
            "token_plus_latent_codebook",
        }:
            if token_ids is None:
                raise ValueError("token addressing requires token ids")
            if layer is None:
                addresses = torch.stack(
                    tuple(
                        self._associative_addresses(
                            token_ids,
                            context_token_ids,
                            memory_layer,
                            source[memory_layer],
                        )
                        for memory_layer in range(self._associative_layer_count())
                    )
                )
            else:
                addresses = self._associative_addresses(
                    token_ids,
                    context_token_ids,
                    layer,
                    source,
                )
            keys = F.one_hot(
                addresses,
                num_classes=self.config.associative_memory_rank,
            ).to(dtype=source.dtype)
            return keys
        if self.associative_key_projection is None:
            raise RuntimeError("dense associative key projection is missing")
        if layer is None:
            if source.shape[0] != self._associative_layer_count():
                raise ValueError("stacked associative source must start with layer")
            original_shape = source.shape
            memory_layers = self._associative_layer_count()
            flat = source.reshape(memory_layers, -1, self.config.width)
            scores = torch.bmm(flat, self.associative_key_projection).reshape(
                *original_shape[:-1],
                self.config.associative_memory_rank,
            )
        else:
            scores = source @ self.associative_key_projection[layer]
        weights = scores.square().square() + self.config.rms_eps
        return weights / weights.sum(dim=-1, keepdim=True)

    def _multiscale_count_sketch_bucket_count(self) -> int:
        if (
            self.config.associative_memory_addressing
            not in MULTISCALE_COUNT_SKETCH_ADDRESSINGS
        ):
            raise RuntimeError("multiscale CountSketch memory is disabled")
        return self.config.associative_memory_rank // self._count_sketch_table_count()

    def _count_sketch_table_count(self) -> int:
        return MULTISCALE_COUNT_SKETCH_TABLES * (
            2
            if self.config.associative_memory_addressing
            in DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS
            else 1
        )

    def _count_sketch_read_group_count(self) -> int:
        return self._count_sketch_table_count() // MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES

    def _reduce_count_sketch_read_group(
        self,
        table_reads: Tensor,
        group_index: int,
    ) -> Tensor:
        if table_reads.shape[-2] != MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES:
            raise ValueError("CountSketch read group must contain three tables")
        if not 0 <= group_index < self._count_sketch_read_group_count():
            raise ValueError("CountSketch read group index is out of range")
        if (
            self.config.associative_memory_addressing
            == ZERO_ANCHORED_PAIR_CONSENSUS_ADDRESSING
            and group_index >= 2
        ):
            anchored = torch.cat(
                (table_reads, torch.zeros_like(table_reads[..., :1, :])),
                dim=-2,
            )
            ordered = anchored.sort(dim=-2, stable=True).values
            return (ordered[..., 1, :] + ordered[..., 2, :]) * 0.5
        return table_reads.median(dim=-2).values

    def _count_sketch_hash_addresses_and_signs(
        self,
        keys: Tensor,
        *,
        table_offset: int,
    ) -> tuple[Tensor, Tensor]:
        bucket_count = self._multiscale_count_sketch_bucket_count()
        key = keys.to(torch.long).unsqueeze(-1)
        multipliers = self.count_sketch_hash_multipliers
        offsets = self.count_sketch_hash_offsets
        if multipliers is None or offsets is None:
            raise RuntimeError("CountSketch address constants are missing")
        mixed = key * multipliers + offsets
        mixed = torch.bitwise_xor(mixed, torch.bitwise_right_shift(mixed, 16))
        local_addresses = mixed.remainder(bucket_count)
        table_offsets = (
            torch.arange(
                MULTISCALE_COUNT_SKETCH_TABLES,
                device=keys.device,
                dtype=torch.long,
            )
            + table_offset
        ) * bucket_count
        addresses = local_addresses + table_offsets

        sign_multipliers = self.count_sketch_sign_multipliers
        sign_offsets = self.count_sketch_sign_offsets
        if sign_multipliers is None or sign_offsets is None:
            raise RuntimeError("CountSketch sign constants are missing")
        sign_hash = key * sign_multipliers + sign_offsets
        sign_hash = torch.bitwise_xor(
            sign_hash,
            torch.bitwise_right_shift(sign_hash, 13),
        )
        signs = torch.where(
            sign_hash.remainder(2) == 0,
            torch.ones_like(sign_hash),
            -torch.ones_like(sign_hash),
        )
        return addresses, signs

    def _multiscale_count_sketch_addresses_and_signs(
        self,
        token_ids: Tensor,
        context_token_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        token_addresses, token_signs = self._count_sketch_hash_addresses_and_signs(
            token_ids,
            table_offset=0,
        )
        if self.config.associative_memory_addressing not in DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS:
            return token_addresses, token_signs
        if context_token_ids is None or context_token_ids.shape != token_ids.shape:
            raise ValueError("dual-order CountSketch requires matching context tokens")
        pair_keys = context_token_ids.to(torch.long) * self.config.vocab_size + token_ids
        pair_addresses, pair_signs = self._count_sketch_hash_addresses_and_signs(
            pair_keys,
            table_offset=MULTISCALE_COUNT_SKETCH_TABLES,
        )
        return (
            torch.cat((token_addresses, pair_addresses), dim=-1),
            torch.cat((token_signs, pair_signs), dim=-1),
        )

    def _multiscale_count_sketch_keys(
        self,
        token_ids: Tensor,
        dtype: torch.dtype,
        context_token_ids: Tensor | None = None,
    ) -> Tensor:
        addresses, signs = self._multiscale_count_sketch_addresses_and_signs(
            token_ids,
            context_token_ids,
        )
        keys = torch.zeros(
            *token_ids.shape,
            self.config.associative_memory_rank,
            device=token_ids.device,
            dtype=dtype,
        )
        return keys.scatter(-1, addresses, signs.to(dtype))

    def _indexed_multiscale_count_sketch_addresses_and_signs(
        self,
        token_ids: Tensor,
        context_token_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if (
            self.indexed_count_sketch_addresses is None
            or self.indexed_count_sketch_signs is None
        ):
            raise RuntimeError("indexed multiscale CountSketch is disabled")
        token_addresses = self.indexed_count_sketch_addresses[token_ids]
        token_signs = self.indexed_count_sketch_signs[token_ids]
        if self.config.associative_memory_addressing not in DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS:
            return token_addresses, token_signs
        if context_token_ids is None or context_token_ids.shape != token_ids.shape:
            raise ValueError("dual-order indexed sketch requires matching context tokens")
        pair_keys = context_token_ids.to(torch.long) * self.config.vocab_size + token_ids
        pair_addresses, pair_signs = self._count_sketch_hash_addresses_and_signs(
            pair_keys,
            table_offset=MULTISCALE_COUNT_SKETCH_TABLES,
        )
        return (
            torch.cat((token_addresses, pair_addresses), dim=-1),
            torch.cat((token_signs, pair_signs.to(token_signs.dtype)), dim=-1),
        )

    def _multiscale_count_sketch_decays(self, reference: Tensor) -> Tensor:
        decays = self.count_sketch_table_decays
        if decays is None:
            raise RuntimeError("multiscale CountSketch decays are missing")
        return decays.to(dtype=reference.dtype)

    def _count_sketch_scan_decay(self, group_start: int) -> Tensor:
        decays = self.count_sketch_scan_decays
        if decays is None:
            raise RuntimeError("CountSketch scan decays are missing")
        group = (group_start // MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES) % 2
        return decays[group]

    def _multiscale_count_sketch_sequence(
        self,
        source: Tensor,
        initial_memory: Tensor,
        initial_normalizer: Tensor,
        initial_key: Tensor,
        token_ids: Tensor,
        initial_context_token: Tensor | None = None,
        reset_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, sequence, _ = source.shape
        statistics_dtype = initial_memory.dtype
        if (
            initial_normalizer.dtype != statistics_dtype
            or initial_key.dtype != statistics_dtype
        ):
            raise ValueError("CountSketch statistics dtypes differ")
        table_count = self._count_sketch_table_count()
        bucket_count = self._multiscale_count_sketch_bucket_count()
        value_width = self._associative_value_width()
        context_token_ids = None
        if self.config.associative_memory_addressing in DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS:
            if initial_context_token is None:
                raise ValueError("dual-order sequence requires an initial context token")
            context_token_ids = torch.cat(
                (initial_context_token.unsqueeze(1), token_ids[:, :-1]),
                dim=1,
            )
            if reset_mask is not None:
                context_token_ids = torch.where(
                    reset_mask,
                    torch.full_like(context_token_ids, self.config.bos_token_id),
                    context_token_ids,
                )
        addresses, signs = self._multiscale_count_sketch_addresses_and_signs(
            token_ids,
            context_token_ids,
        )
        table_offsets = (
            torch.arange(
                table_count,
                device=token_ids.device,
                dtype=torch.long,
            )
            * bucket_count
        )
        local_addresses = addresses - table_offsets.view(1, 1, table_count)

        initial_key_tables = initial_key.reshape(batch, table_count, bucket_count)
        initial_addresses = initial_key_tables.abs().argmax(dim=-1)
        initial_signs = torch.gather(
            initial_key_tables,
            2,
            initial_addresses.unsqueeze(-1),
        ).squeeze(-1)
        initial_strength = initial_key_tables.abs().sum(dim=-1)
        pending_addresses = torch.cat(
            (initial_addresses.unsqueeze(1), local_addresses[:, :-1]),
            dim=1,
        )
        pending_signs = torch.cat(
            (initial_signs.unsqueeze(1), signs[:, :-1].to(statistics_dtype)),
            dim=1,
        )

        decays = self._multiscale_count_sketch_decays(initial_memory)
        write_scales = (
            1.0 - decays
            if self.config.scale_invariant_associative_state
            else torch.ones_like(decays)
        )
        pending_strength = torch.cat(
            (
                initial_strength.unsqueeze(1),
                torch.ones(
                    batch,
                    sequence - 1,
                    table_count,
                    device=source.device,
                    dtype=statistics_dtype,
                ),
            ),
            dim=1,
        )
        pending_strength = pending_strength * write_scales.view(1, 1, table_count)
        if reset_mask is not None:
            pending_strength = pending_strength.masked_fill(
                reset_mask.unsqueeze(-1),
                0,
            )

        encoded_source = self._encode_associative_values(source).to(
            statistics_dtype
        )
        initial_memory_tables = initial_memory.reshape(
            batch, table_count, bucket_count, value_width
        )
        initial_normalizer_tables = initial_normalizer.reshape(
            batch, table_count, bucket_count
        )
        if self.config.fused_sparse_count_sketch_prefill:
            if (
                not source.is_cuda
                or source.dtype not in (torch.float16, torch.float32)
                or torch.is_grad_enabled()
            ):
                raise ValueError(
                    "sparse CountSketch prefill requires no-grad CUDA FP16 or FP32"
                )
            if self.config.composable_sparse_count_sketch_prefill:
                from pfc.phase386_composable_sparse_prefill import (
                    composable_sparse_ordered_count_sketch_prefill,
                )

                sparse_prefill = composable_sparse_ordered_count_sketch_prefill
            else:
                from pfc.phase382_sparse_count_sketch_prefill import (
                    sparse_ordered_count_sketch_prefill,
                )

                sparse_prefill = sparse_ordered_count_sketch_prefill

            numerator, denominator, final_memory, final_normalizer = (
                sparse_prefill(
                    encoded_source,
                    initial_memory_tables,
                    initial_normalizer_tables,
                    pending_addresses,
                    pending_signs,
                    pending_strength,
                    local_addresses,
                    signs,
                    self.count_sketch_scan_decays,
                )
            )
            denominator_floor = self._associative_denominator_floor(
                source,
                write_scales.view(1, 1, table_count, 1),
            )
            table_reads = numerator / denominator.unsqueeze(-1).clamp_min(
                denominator_floor
            )
            group_reads = tuple(
                self._reduce_count_sketch_read_group(
                    table_reads[:, :, start : start + MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES],
                    start // MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES,
                )
                for start in range(
                    0,
                    table_count,
                    MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES,
                )
            )
            read = self._decode_associative_values(
                torch.cat(group_reads, dim=-1)
            ).to(source.dtype)
            final_key = self._multiscale_count_sketch_keys(
                token_ids[:, -1],
                statistics_dtype,
                context_token_ids[:, -1] if context_token_ids is not None else None,
            )
            return (
                read,
                final_memory.reshape(
                    batch,
                    table_count * bucket_count,
                    value_width,
                ),
                final_normalizer.reshape(
                    batch,
                    table_count * bucket_count,
                ),
                final_key,
            )
        signed_writes = torch.zeros(
            batch,
            sequence,
            table_count,
            bucket_count,
            value_width,
            device=source.device,
            dtype=statistics_dtype,
        )
        signed_writes = signed_writes.scatter(
            3,
            pending_addresses.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, 1, value_width
            ),
            encoded_source.unsqueeze(2).unsqueeze(3)
            * pending_signs.unsqueeze(-1).unsqueeze(-1)
            * pending_strength.unsqueeze(-1).unsqueeze(-1),
        )
        normalizer_writes = torch.zeros(
            batch,
            sequence,
            table_count,
            bucket_count,
            device=source.device,
            dtype=statistics_dtype,
        )
        normalizer_writes = normalizer_writes.scatter(
            3,
            pending_addresses.unsqueeze(-1),
            pending_strength.unsqueeze(-1),
        )
        def scan_group(
            writes: Tensor,
            initial: Tensor,
            decay: Tensor,
        ) -> Tensor:
            group_tables = writes.shape[2]
            flat_writes = writes.permute(0, 2, 1, *range(3, writes.ndim)).reshape(
                batch * group_tables,
                sequence,
                *writes.shape[3:],
            )
            flat_initial = initial.reshape(
                batch * group_tables,
                *initial.shape[2:],
            )
            flat_reset = (
                reset_mask.unsqueeze(1)
                .expand(batch, group_tables, sequence)
                .reshape(batch * group_tables, sequence)
                if reset_mask is not None
                else None
            )
            scanned = _constant_decay_scan(
                flat_writes,
                flat_initial,
                decay,
                fused=self.config.fused_constant_decay_scan,
                reset_mask=flat_reset,
            )
            return scanned.reshape(
                batch,
                group_tables,
                sequence,
                *writes.shape[3:],
            ).transpose(1, 2)

        group_width = MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES
        group_starts = range(0, table_count, group_width)
        memory_states = torch.cat(
            tuple(
                scan_group(
                    signed_writes[:, :, start : start + group_width],
                    initial_memory_tables[:, start : start + group_width],
                    self._count_sketch_scan_decay(start),
                )
                for start in group_starts
            ),
            dim=2,
        )
        normalizer_states = torch.cat(
            tuple(
                scan_group(
                    normalizer_writes[:, :, start : start + group_width],
                    initial_normalizer_tables[:, start : start + group_width],
                    self._count_sketch_scan_decay(start),
                )
                for start in range(0, table_count, group_width)
            ),
            dim=2,
        )

        numerator = torch.gather(
            memory_states,
            3,
            local_addresses.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, 1, value_width
            ),
        ).squeeze(3)
        numerator = numerator * signs.to(statistics_dtype).unsqueeze(-1)
        denominator = torch.gather(
            normalizer_states,
            3,
            local_addresses.unsqueeze(-1),
        )
        denominator_floor = self._associative_denominator_floor(
            initial_memory,
            write_scales.view(1, 1, table_count, 1),
        )
        table_reads = numerator / denominator.clamp_min(denominator_floor)
        group_reads = tuple(
            self._reduce_count_sketch_read_group(
                table_reads[:, :, start : start + group_width],
                start // group_width,
            )
            for start in range(0, table_count, group_width)
        )
        read = self._decode_associative_values(
            torch.cat(group_reads, dim=-1)
        ).to(source.dtype)
        final_key = self._multiscale_count_sketch_keys(
            token_ids[:, -1],
            statistics_dtype,
            context_token_ids[:, -1] if context_token_ids is not None else None,
        )
        return (
            read,
            memory_states[:, -1].reshape(
                batch, table_count * bucket_count, value_width
            ),
            normalizer_states[:, -1].reshape(batch, table_count * bucket_count),
            final_key,
        )

    def _semantic_count_sketch_decays(self, reference: Tensor) -> Tensor:
        decays = self.semantic_count_sketch_table_decays
        if decays is None:
            raise RuntimeError("semantic CountSketch decays are missing")
        return decays.to(dtype=reference.dtype)

    def _semantic_count_sketch_group_reads(
        self,
        table_reads: Tensor,
        query_normalizers: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        if table_reads.shape[-2:] != (
            SEMANTIC_COUNT_SKETCH_TABLES,
            SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
        ):
            raise ValueError("invalid semantic CountSketch table-read shape")
        if query_normalizers.shape != table_reads.shape[:-1]:
            raise ValueError("semantic normalizers must match table reads")
        decoded_groups: list[Tensor] = []
        active_groups: list[Tensor] = []
        for group_index, start in enumerate(
            (0, MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES)
        ):
            stop = start + MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES
            encoded = table_reads[..., start:stop, :].median(dim=-2).values
            decoded_groups.append(
                self._decode_semantic_channel_values(encoded, group_index)
                if self._uses_disjoint_timescale_semantic_channels()
                else self._decode_semantic_values(encoded)
            )
            active_groups.append(
                (query_normalizers[..., start:stop] > 0)
                .sum(dim=-1, keepdim=True)
                .ge(2)
            )
        return (
            (decoded_groups[0], decoded_groups[1]),
            (active_groups[0], active_groups[1]),
        )

    @staticmethod
    def _compose_semantic_count_sketch_read(
        parent_read: Tensor,
        semantic_reads: tuple[Tensor, Tensor],
        semantic_active: tuple[Tensor, Tensor],
    ) -> Tensor:
        if any(read.shape != parent_read.shape for read in semantic_reads):
            raise ValueError("semantic and exact reads must share shape")
        active = tuple(mask.to(parent_read.dtype) for mask in semantic_active)
        numerator = 4.0 * parent_read
        denominator = parent_read.new_full(parent_read.shape[:-1] + (1,), 4.0)
        for read, mask in zip(semantic_reads, active, strict=True):
            numerator = numerator + mask * read
            denominator = denominator + mask
        return numerator / denominator

    @staticmethod
    def _compose_disjoint_timescale_semantic_channel_read(
        parent_read: Tensor,
        semantic_reads: tuple[Tensor, Tensor],
        semantic_active: tuple[Tensor, Tensor],
    ) -> Tensor:
        if any(read.shape != parent_read.shape for read in semantic_reads):
            raise ValueError("semantic and exact reads must share shape")
        if parent_read.shape[-1] % 2:
            raise ValueError("disjoint semantic channels require even width")
        half = parent_read.shape[-1] // 2
        active = tuple(mask.to(parent_read.dtype) for mask in semantic_active)
        fast = (
            4.0 * parent_read[..., :half]
            + active[0] * semantic_reads[0][..., :half]
        ) / (4.0 + active[0])
        slow = (
            4.0 * parent_read[..., half:]
            + active[1] * semantic_reads[1][..., half:]
        ) / (4.0 + active[1])
        return torch.cat((fast, slow), dim=-1)

    def _semantic_count_sketch_sequence(
        self,
        source: Tensor,
        initial_memory: Tensor,
        initial_normalizer: Tensor,
        initial_context: Tensor,
        initial_codes: Tensor,
        *,
        initial_write_enabled: bool,
        reset_mask: Tensor | None = None,
    ) -> tuple[
        tuple[Tensor, Tensor],
        tuple[Tensor, Tensor],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        if not self._uses_semantic_count_sketch_backoff():
            raise RuntimeError("semantic CountSketch backoff is disabled")
        batch, sequence, width = source.shape
        statistics_dtype = initial_memory.dtype
        if (
            initial_normalizer.dtype != statistics_dtype
            or initial_context.dtype != statistics_dtype
        ):
            raise ValueError("semantic CountSketch statistics dtypes differ")
        if width != self.config.width:
            raise ValueError("semantic source width differs")
        semantic_rank = SEMANTIC_COUNT_SKETCH_TABLES * SEMANTIC_COUNT_SKETCH_BUCKETS
        if initial_memory.shape != (
            batch,
            semantic_rank,
            SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
        ):
            raise ValueError("invalid initial semantic memory shape")
        if initial_normalizer.shape != (batch, semantic_rank):
            raise ValueError("invalid initial semantic normalizer shape")
        if initial_context.shape != (batch, width):
            raise ValueError("invalid initial semantic context shape")
        if initial_codes.shape != (batch, SEMANTIC_COUNT_SKETCH_TABLES):
            raise ValueError("invalid initial semantic codes shape")

        context_states = _constant_decay_scan(
            source.detach().to(statistics_dtype),
            initial_context.detach(),
            self._count_sketch_scan_decay(
                MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES
            ),
            fused=self.config.fused_constant_decay_scan,
            reset_mask=reset_mask,
        )
        query_codes = self._semantic_context_codes(context_states)
        query_addresses, query_signs = self._unpack_semantic_context_codes(
            query_codes
        )
        initial_addresses, initial_signs = self._unpack_semantic_context_codes(
            initial_codes
        )
        pending_addresses = torch.cat(
            (initial_addresses.unsqueeze(1), query_addresses[:, :-1]),
            dim=1,
        )
        pending_signs = torch.cat(
            (initial_signs.unsqueeze(1), query_signs[:, :-1]),
            dim=1,
        ).to(statistics_dtype)

        decays = self._semantic_count_sketch_decays(initial_memory)
        write_scales = 1.0 - decays
        first_strength = initial_memory.new_full(
            (batch, 1, SEMANTIC_COUNT_SKETCH_TABLES),
            float(initial_write_enabled),
        )
        pending_strength = torch.cat(
            (
                first_strength,
                torch.ones(
                    batch,
                    sequence - 1,
                    SEMANTIC_COUNT_SKETCH_TABLES,
                    device=source.device,
                    dtype=statistics_dtype,
                ),
            ),
            dim=1,
        ) * write_scales.view(1, 1, SEMANTIC_COUNT_SKETCH_TABLES)
        if reset_mask is not None:
            pending_strength = pending_strength.masked_fill(
                reset_mask.unsqueeze(-1),
                0,
            )
        encoded_source = self._encode_semantic_values(source).to(
            statistics_dtype
        )
        if self.config.exact_whole_prefill_workspace:
            from pfc.phase444_exact_whole_prefill_workspace import (
                dense_semantic_prefix_explicit_round,
            )

            initial_memory_tables = initial_memory.reshape(
                batch,
                SEMANTIC_COUNT_SKETCH_TABLES,
                SEMANTIC_COUNT_SKETCH_BUCKETS,
                SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
            ).contiguous()
            initial_normalizer_tables = initial_normalizer.reshape(
                batch,
                SEMANTIC_COUNT_SKETCH_TABLES,
                SEMANTIC_COUNT_SKETCH_BUCKETS,
            ).contiguous()
            scan_decays = self.count_sketch_scan_decays
            if scan_decays is None:
                raise RuntimeError("semantic CountSketch scan decays are missing")
            (
                numerator,
                query_normalizers,
                final_memory,
                final_normalizer,
            ) = dense_semantic_prefix_explicit_round(
                encoded_source.contiguous(),
                initial_memory_tables,
                initial_normalizer_tables,
                pending_addresses.contiguous(),
                pending_signs.contiguous(),
                pending_strength.contiguous(),
                query_addresses.contiguous(),
                query_signs.to(statistics_dtype).contiguous(),
                scan_decays.contiguous(),
            )
            denominator_floor = self._associative_denominator_floor(
                initial_memory,
                write_scales.view(1, 1, SEMANTIC_COUNT_SKETCH_TABLES, 1),
            )
            table_reads = numerator / query_normalizers.unsqueeze(-1).clamp_min(
                denominator_floor
            )
            semantic_reads, semantic_active = self._semantic_count_sketch_group_reads(
                table_reads,
                query_normalizers,
            )
            semantic_reads = tuple(
                read.to(source.dtype) for read in semantic_reads
            )
            return (
                semantic_reads,
                semantic_active,
                final_memory.reshape(
                    batch,
                    semantic_rank,
                    SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
                ),
                final_normalizer.reshape(batch, semantic_rank),
                context_states[:, -1].clone(),
                query_codes[:, -1].clone(),
            )
        signed_writes = torch.zeros(
            batch,
            sequence,
            SEMANTIC_COUNT_SKETCH_TABLES,
            SEMANTIC_COUNT_SKETCH_BUCKETS,
            SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
            device=source.device,
            dtype=statistics_dtype,
        ).scatter(
            3,
            pending_addresses.unsqueeze(-1).unsqueeze(-1).expand(
                -1,
                -1,
                -1,
                1,
                SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
            ),
            encoded_source.unsqueeze(2).unsqueeze(3)
            * pending_signs.unsqueeze(-1).unsqueeze(-1)
            * pending_strength.unsqueeze(-1).unsqueeze(-1),
        )
        normalizer_writes = torch.zeros(
            batch,
            sequence,
            SEMANTIC_COUNT_SKETCH_TABLES,
            SEMANTIC_COUNT_SKETCH_BUCKETS,
            device=source.device,
            dtype=statistics_dtype,
        ).scatter(
            3,
            pending_addresses.unsqueeze(-1),
            pending_strength.unsqueeze(-1),
        )
        initial_memory_tables = initial_memory.reshape(
            batch,
            SEMANTIC_COUNT_SKETCH_TABLES,
            SEMANTIC_COUNT_SKETCH_BUCKETS,
            SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
        )
        initial_normalizer_tables = initial_normalizer.reshape(
            batch,
            SEMANTIC_COUNT_SKETCH_TABLES,
            SEMANTIC_COUNT_SKETCH_BUCKETS,
        )

        def scan_group(writes: Tensor, initial: Tensor, decay: Tensor) -> Tensor:
            group_tables = writes.shape[2]
            flat_writes = writes.permute(
                0,
                2,
                1,
                *range(3, writes.ndim),
            ).reshape(batch * group_tables, sequence, *writes.shape[3:])
            flat_initial = initial.reshape(
                batch * group_tables,
                *initial.shape[2:],
            )
            flat_reset = (
                reset_mask.unsqueeze(1)
                .expand(batch, group_tables, sequence)
                .reshape(batch * group_tables, sequence)
                if reset_mask is not None
                else None
            )
            scanned = _constant_decay_scan(
                flat_writes,
                flat_initial,
                decay,
                fused=self.config.fused_constant_decay_scan,
                reset_mask=flat_reset,
            )
            return scanned.reshape(
                batch,
                group_tables,
                sequence,
                *writes.shape[3:],
            ).transpose(1, 2)

        group_width = MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES
        memory_states = torch.cat(
            tuple(
                scan_group(
                    signed_writes[:, :, start : start + group_width],
                    initial_memory_tables[:, start : start + group_width],
                    self._count_sketch_scan_decay(start),
                )
                for start in (0, group_width)
            ),
            dim=2,
        )
        normalizer_states = torch.cat(
            tuple(
                scan_group(
                    normalizer_writes[:, :, start : start + group_width],
                    initial_normalizer_tables[:, start : start + group_width],
                    self._count_sketch_scan_decay(start),
                )
                for start in (0, group_width)
            ),
            dim=2,
        )
        numerator = torch.gather(
            memory_states,
            3,
            query_addresses.unsqueeze(-1).unsqueeze(-1).expand(
                -1,
                -1,
                -1,
                1,
                SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
            ),
        ).squeeze(3)
        numerator = numerator * query_signs.to(statistics_dtype).unsqueeze(-1)
        query_normalizers = torch.gather(
            normalizer_states,
            3,
            query_addresses.unsqueeze(-1),
        ).squeeze(-1)
        denominator_floor = self._associative_denominator_floor(
            initial_memory,
            write_scales.view(1, 1, SEMANTIC_COUNT_SKETCH_TABLES, 1),
        )
        table_reads = numerator / query_normalizers.unsqueeze(-1).clamp_min(
            denominator_floor
        )
        semantic_reads, semantic_active = self._semantic_count_sketch_group_reads(
            table_reads,
            query_normalizers,
        )
        semantic_reads = tuple(read.to(source.dtype) for read in semantic_reads)
        return (
            semantic_reads,
            semantic_active,
            memory_states[:, -1].reshape(
                batch,
                semantic_rank,
                SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
            ).clone(),
            normalizer_states[:, -1].reshape(batch, semantic_rank).clone(),
            context_states[:, -1].clone(),
            query_codes[:, -1].clone(),
        )

    def _semantic_backoff_associative_sequence(
        self,
        source: Tensor,
        initial_memory: Tensor,
        initial_normalizer: Tensor,
        initial_key: Tensor,
        token_ids: Tensor,
        initial_context_token: Tensor,
        initial_semantic_memory: Tensor,
        initial_semantic_normalizer: Tensor,
        initial_semantic_context: Tensor,
        initial_semantic_codes: Tensor,
        *,
        initial_write_enabled: bool,
        semantic_enabled: Tensor | None = None,
        reset_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        parent_read, memory, normalizer, key = (
            self._multiscale_count_sketch_sequence(
                source,
                initial_memory,
                initial_normalizer,
                initial_key,
                token_ids,
                initial_context_token,
                reset_mask,
            )
        )
        (
            semantic_reads,
            semantic_active,
            semantic_memory,
            semantic_normalizer,
            semantic_context,
            semantic_codes,
        ) = self._semantic_count_sketch_sequence(
            source,
            initial_semantic_memory,
            initial_semantic_normalizer,
            initial_semantic_context,
            initial_semantic_codes,
            initial_write_enabled=initial_write_enabled,
            reset_mask=reset_mask,
        )
        if semantic_enabled is not None:
            if semantic_enabled.shape != source.shape[:2] + (1,):
                raise ValueError("semantic enabled mask must be [batch, sequence, 1]")
            semantic_active = tuple(
                active & semantic_enabled for active in semantic_active
            )
        read = (
            self._compose_disjoint_timescale_semantic_channel_read(
                parent_read,
                semantic_reads,
                semantic_active,
            )
            if self._uses_disjoint_timescale_semantic_channels()
            else self._compose_semantic_count_sketch_read(
                parent_read,
                semantic_reads,
                semantic_active,
            )
        )
        return (
            read,
            memory,
            normalizer,
            key,
            semantic_memory,
            semantic_normalizer,
            semantic_context,
            semantic_codes,
        )

    def _associative_addresses(
        self,
        token_ids: Tensor,
        context_token_ids: Tensor | None,
        layer: int,
        source: Tensor | None = None,
    ) -> Tensor:
        addressing = self.config.associative_memory_addressing
        if addressing not in {
            "token_hash",
            "token_bigram_hash",
            "token_plus_bigram_hash",
            "token_plus_context_bucket",
            "token_plus_latent_hash",
            "token_plus_latent_codebook",
        }:
            raise RuntimeError("associative addresses require token addressing")
        if not 0 <= layer < self._associative_layer_count():
            raise ValueError("associative memory layer is out of range")
        if addressing == "token_hash" or (
            addressing in {"token_plus_latent_hash", "token_plus_latent_codebook"}
            and layer == 0
        ):
            addresses = token_ids
        elif addressing == "token_plus_latent_hash":
            if source is None:
                raise ValueError("latent hash addressing requires source state")
            if self.associative_latent_projection is None:
                raise RuntimeError("latent associative projection is missing")
            scores = source @ self.associative_latent_projection[layer]
            bit_weights = torch.bitwise_left_shift(
                torch.ones(
                    self.config.associative_memory_latent_bits,
                    device=source.device,
                    dtype=torch.long,
                ),
                torch.arange(
                    self.config.associative_memory_latent_bits,
                    device=source.device,
                    dtype=torch.long,
                ),
            )
            addresses = ((scores > 0).to(torch.long) * bit_weights).sum(dim=-1)
        elif addressing == "token_plus_latent_codebook":
            if source is None:
                raise ValueError("latent codebook addressing requires source state")
            if self.associative_latent_codebook is None:
                raise RuntimeError("latent associative codebook is missing")
            normalized = F.normalize(source, dim=-1)
            addresses = (
                normalized @ self.associative_latent_codebook[layer - 1]
            ).argmax(dim=-1)
        else:
            if context_token_ids is None:
                raise ValueError(
                    "contextual token addressing requires context token ids"
                )
            if addressing == "token_plus_context_bucket":
                base_addresses = (
                    token_ids * self.config.associative_memory_context_buckets
                )
                context_addresses = base_addresses + (
                    context_token_ids * 2_654_435_761
                ).remainder(self.config.associative_memory_context_buckets)
            else:
                base_addresses = token_ids
                context_addresses = (
                    context_token_ids * 2_654_435_761 + token_ids * 2_246_822_519
                )
            addresses = (
                base_addresses
                if addressing
                in {
                    "token_plus_bigram_hash",
                    "token_plus_context_bucket",
                }
                and layer == 0
                else context_addresses
            )
        return addresses.remainder(self.config.associative_memory_rank)

    def _token_associative_sequence(
        self,
        source: Tensor,
        initial_memory: Tensor,
        initial_normalizer: Tensor,
        initial_key: Tensor,
        layer: int,
        token_ids: Tensor,
        context_token_ids: Tensor | None,
        reset_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Exact token-address recurrence without sequence-by-rank tensors."""

        addresses = self._associative_addresses(
            token_ids,
            context_token_ids,
            layer,
            source,
        )
        batch, sequence = addresses.shape
        encoded_source = self._encode_associative_values(source)
        initial_address = initial_key.argmax(dim=-1)
        pending_addresses = torch.cat(
            (initial_address.unsqueeze(1), addresses[:, :-1]),
            dim=1,
        )
        pending_strength = torch.cat(
            (
                initial_key.sum(dim=-1, keepdim=True),
                torch.ones(
                    batch,
                    sequence - 1,
                    device=source.device,
                    dtype=source.dtype,
                ),
            ),
            dim=1,
        )
        write_scale = self._associative_state_write_scale(source)
        pending_strength = pending_strength * write_scale
        if reset_mask is not None:
            pending_strength = pending_strength.masked_fill(reset_mask, 0)
        time = torch.arange(sequence, device=source.device)
        lag = time.unsqueeze(1) - time.unsqueeze(0)
        causal = lag >= 0
        decay = torch.as_tensor(
            self.config.associative_memory_decay,
            device=source.device,
            dtype=source.dtype,
        )
        decay_weights = decay.pow(lag.clamp_min(0)).masked_fill(~causal, 0)
        address_match = addresses.unsqueeze(2) == pending_addresses.unsqueeze(1)
        same_segment = None
        before_first_reset = None
        if reset_mask is not None:
            segment_ids = reset_mask.to(torch.long).cumsum(dim=1)
            same_segment = segment_ids.unsqueeze(2).eq(segment_ids.unsqueeze(1))
            before_first_reset = segment_ids.eq(0)
        read_weights = (
            address_match.to(source.dtype)
            * decay_weights.unsqueeze(0)
            * pending_strength.unsqueeze(1)
        )
        if same_segment is not None:
            read_weights = read_weights * same_segment.to(source.dtype)
        numerator = torch.bmm(read_weights, encoded_source)
        initial_memory_at_address = torch.gather(
            initial_memory,
            1,
            addresses.unsqueeze(-1).expand(
                batch,
                sequence,
                initial_memory.shape[-1],
            ),
        )
        initial_power = decay.pow(time + 1).view(1, sequence, 1)
        if before_first_reset is not None:
            initial_power = initial_power * before_first_reset.unsqueeze(-1)
        numerator = numerator + initial_power * initial_memory_at_address
        initial_normalizer_at_address = torch.gather(
            initial_normalizer,
            1,
            addresses,
        ).unsqueeze(-1)
        denominator_floor = self._associative_denominator_floor(
            source,
            write_scale,
        )
        denominator = (
            read_weights.sum(dim=-1, keepdim=True)
            + initial_power * initial_normalizer_at_address
        ).clamp_min(denominator_floor)
        read = self._decode_associative_values(numerator / denominator)

        final_power = decay.pow(sequence - 1 - time) * pending_strength
        if reset_mask is not None:
            final_segment = reset_mask.to(torch.long).cumsum(dim=1)
            final_power = final_power * final_segment.eq(
                final_segment[:, -1:]
            ).to(final_power.dtype)
            initial_retained = final_segment[:, -1].eq(0).reshape(batch, 1, 1)
            memory = (
                initial_memory
                * decay.pow(sequence)
                * initial_retained.to(initial_memory.dtype)
            )
        else:
            memory = initial_memory * decay.pow(sequence)
        memory = memory.scatter_add(
            1,
            pending_addresses.unsqueeze(-1).expand_as(encoded_source),
            encoded_source * final_power.unsqueeze(-1),
        )
        normalizer = initial_normalizer * decay.pow(sequence)
        if reset_mask is not None:
            normalizer = normalizer * initial_retained.squeeze(-1).to(
                initial_normalizer.dtype
            )
        normalizer = normalizer.scatter_add(
            1,
            pending_addresses,
            final_power,
        )
        final_key = F.one_hot(
            addresses[:, -1],
            num_classes=self.config.associative_memory_rank,
        ).to(source.dtype)
        if (
            self.config.associative_memory_addressing
            in {
                "token_plus_bigram_hash",
                "token_plus_context_bucket",
                "token_plus_latent_hash",
                "token_plus_latent_codebook",
            }
            and layer > 0
            and self.config.associative_memory_context_gate_count > 0
        ):
            read = (
                read
                * denominator
                / (
                    denominator
                    + self.config.associative_memory_context_gate_count * write_scale
                )
            )
        return read, memory, normalizer, final_key

    def _associative_state_write_scale(self, reference: Tensor) -> Tensor:
        scale = (
            1.0 - self.config.associative_memory_decay
            if self.config.scale_invariant_associative_state
            else 1.0
        )
        return reference.new_full((), scale)

    def _associative_denominator_floor(
        self,
        reference: Tensor,
        write_scale: Tensor,
    ) -> Tensor:
        requested = reference.new_full((), self.config.rms_eps) * write_scale
        zero = reference.new_full((), 0.0)
        smallest_positive = torch.nextafter(zero, reference.new_full((), 1.0))
        return torch.maximum(requested, smallest_positive)

    def _multiscale_count_sketch_step(
        self,
        source: Tensor,
        state: NativePCState,
        token_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if state.associative_memory is None:
            raise RuntimeError("associative memory state is missing")
        if state.associative_normalizer is None or state.associative_keys is None:
            raise RuntimeError("associative address state is missing")
        memory_layers = self._associative_layer_count()
        memory_source = source[:memory_layers]
        statistics_dtype = state.associative_memory.dtype
        statistics_source = memory_source.to(statistics_dtype)
        encoded_source = self._encode_associative_values(statistics_source)
        keys = self._associative_keys(statistics_source, token_ids=token_ids)
        table_count = MULTISCALE_COUNT_SKETCH_TABLES
        bucket_count = self._multiscale_count_sketch_bucket_count()
        value_width = self._associative_value_width()
        table_decays = self._multiscale_count_sketch_decays(
            state.associative_memory
        )
        rank_decays = table_decays.repeat_interleave(bucket_count)
        rank_write_scales = (
            1.0 - rank_decays
            if self.config.scale_invariant_associative_state
            else torch.ones_like(rank_decays)
        )
        memory = (
            rank_decays.view(1, 1, -1, 1) * state.associative_memory
            + state.associative_keys.unsqueeze(-1)
            * encoded_source.unsqueeze(-2)
            * rank_write_scales.view(1, 1, -1, 1)
        )
        normalizer = (
            rank_decays.view(1, 1, -1) * state.associative_normalizer
            + state.associative_keys.abs()
            * rank_write_scales.view(1, 1, -1)
        )

        addresses, signs = self._multiscale_count_sketch_addresses_and_signs(token_ids)
        gather_addresses = addresses.unsqueeze(0).expand(memory_layers, -1, -1)
        numerator = torch.gather(
            memory,
            2,
            gather_addresses.unsqueeze(-1).expand(-1, -1, -1, value_width),
        )
        numerator = numerator * signs.to(statistics_dtype).unsqueeze(0).unsqueeze(-1)
        table_write_scales = (
            1.0 - table_decays
            if self.config.scale_invariant_associative_state
            else torch.ones_like(table_decays)
        )
        denominator = torch.gather(normalizer, 2, gather_addresses).unsqueeze(-1)
        denominator = denominator.clamp_min(
            self._associative_denominator_floor(
                state.associative_memory,
                table_write_scales.view(1, 1, table_count, 1),
            )
        )
        table_reads = numerator / denominator
        fast_read = table_reads[
            :, :, :MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES
        ].median(dim=2).values
        slow_read = table_reads[
            :, :, MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES:
        ].median(dim=2).values
        read = self._decode_associative_values(
            torch.cat((fast_read, slow_read), dim=-1)
        ).to(source.dtype)
        if memory_layers < self.config.depth:
            read = torch.cat(
                (
                    read,
                    torch.zeros(
                        self.config.depth - memory_layers,
                        *read.shape[1:],
                        device=read.device,
                        dtype=read.dtype,
                    ),
                ),
                dim=0,
            )
        return read, memory, normalizer, keys

    def _materialize_lazy_multiscale_count_sketch_state(
        self,
        state: NativePCState,
    ) -> tuple[Tensor, Tensor]:
        if state.associative_memory is None or state.associative_normalizer is None:
            raise RuntimeError("lazy associative memory state is missing")
        if state.associative_last_update is None:
            raise RuntimeError("lazy associative timestamps are missing")
        bucket_count = self._multiscale_count_sketch_bucket_count()
        rank_decays = self._multiscale_count_sketch_decays(
            state.associative_memory
        ).repeat_interleave(bucket_count)
        elapsed = (
            state.token_count - state.associative_last_update.to(torch.int64)
        ).clamp_min(0)
        decay = rank_decays.view(1, 1, -1).pow(
            elapsed.to(state.associative_memory.dtype)
        )
        return (
            state.associative_memory * decay.unsqueeze(-1),
            state.associative_normalizer * decay,
        )

    def _materialize_lazy_semantic_count_sketch_state(
        self,
        state: NativePCState,
    ) -> tuple[Tensor, Tensor]:
        if (
            state.associative_semantic_memory is None
            or state.associative_semantic_normalizer is None
            or state.associative_semantic_last_update is None
        ):
            raise RuntimeError("semantic CountSketch state is missing")
        rank_decays = self._semantic_count_sketch_decays(
            state.associative_semantic_memory
        ).repeat_interleave(SEMANTIC_COUNT_SKETCH_BUCKETS)
        elapsed = (
            state.token_count - state.associative_semantic_last_update.to(torch.int64)
        ).clamp_min(0)
        decay = rank_decays.view(1, 1, -1).pow(
            elapsed.to(state.associative_semantic_memory.dtype)
        )
        return (
            state.associative_semantic_memory * decay.unsqueeze(-1),
            state.associative_semantic_normalizer * decay,
        )

    def _lazy_semantic_count_sketch_step(
        self,
        source: Tensor,
        state: NativePCState,
        encoded_associative_source: Tensor | None = None,
        *,
        next_token_count: int | Tensor,
    ) -> tuple[
        tuple[Tensor, Tensor],
        tuple[Tensor, Tensor],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        if (
            state.associative_semantic_memory is None
            or state.associative_semantic_normalizer is None
            or state.associative_semantic_last_update is None
            or state.associative_semantic_context is None
            or state.associative_semantic_codes is None
        ):
            raise RuntimeError("semantic CountSketch state is missing")
        if (
            self.config.fused_indexed_count_sketch_inference
            and _uses_fp16_cuda_fused_inference(source)
        ):
            memory_source = source[0]
            if self.associative_semantic_hash_mask is None:
                raise RuntimeError("semantic CountSketch hash mask is missing")
            query_codes = fused_semantic_context_codes(
                state.associative_semantic_context,
                memory_source,
                self.associative_semantic_hash_mask,
                context_decay=self._semantic_context_decay(),
                zero_pad_to_power_of_two=(
                    self.config.padded_semantic_hadamard_hash
                    and bool(self.config.width & (self.config.width - 1))
                ),
            )
            if encoded_associative_source is None or (
                encoded_associative_source.shape[:2]
                != (1, memory_source.shape[0])
                or encoded_associative_source.shape[-1]
                < SEMANTIC_COUNT_SKETCH_VALUE_WIDTH
            ):
                raise RuntimeError("fused semantic source encoding is missing")
            encoded_source = (
                encoded_associative_source[
                    0, :, :SEMANTIC_COUNT_SKETCH_VALUE_WIDTH
                ]
                if state.token_count > 0
                else memory_source.new_empty(
                    memory_source.shape[0],
                    SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
                )
            )
            encoded_groups, active = fused_semantic_memory_step(
                state.associative_semantic_memory,
                state.associative_semantic_normalizer,
                state.associative_semantic_last_update,
                encoded_source,
                state.associative_semantic_codes,
                query_codes,
                next_token_count=next_token_count,
                write_enabled=state.token_count > 0,
                fast_decay=self.config.associative_memory_decay,
                rms_eps=self.config.rms_eps,
            )
            return (
                (encoded_groups[:, 0], encoded_groups[:, 1]),
                (active[:, 0:1], active[:, 1:2]),
                state.associative_semantic_memory,
                state.associative_semantic_normalizer,
                state.associative_semantic_last_update,
                state.associative_semantic_context,
                query_codes,
            )
        memory_source = source[0]
        statistics_dtype = state.associative_semantic_memory.dtype
        statistics_source = memory_source.to(statistics_dtype)
        context = (
            self._semantic_context_decay()
            * state.associative_semantic_context.detach()
            + statistics_source.detach()
        )
        query_codes = self._semantic_context_codes(context)
        pending_addresses, pending_signs = self._unpack_semantic_context_codes(
            state.associative_semantic_codes
        )
        query_addresses, query_signs = self._unpack_semantic_context_codes(query_codes)
        table_offsets = (
            torch.arange(
                SEMANTIC_COUNT_SKETCH_TABLES,
                device=source.device,
                dtype=torch.long,
            )
            * SEMANTIC_COUNT_SKETCH_BUCKETS
        )
        pending_addresses = pending_addresses + table_offsets
        query_addresses = query_addresses + table_offsets
        decays = self._semantic_count_sketch_decays(
            state.associative_semantic_memory
        )
        write_scales = 1.0 - decays
        functional = torch.is_grad_enabled()
        memory = (
            state.associative_semantic_memory.clone()
            if functional
            else state.associative_semantic_memory
        )
        normalizer = (
            state.associative_semantic_normalizer.clone()
            if functional
            else state.associative_semantic_normalizer
        )
        last_update = (
            state.associative_semantic_last_update.clone()
            if functional
            else state.associative_semantic_last_update
        )
        next_token_count = state.token_count + 1
        if state.token_count > 0:
            gathered_memory = torch.gather(
                memory[0],
                1,
                pending_addresses.unsqueeze(-1).expand(
                    -1,
                    -1,
                    SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
                ),
            )
            gathered_normalizer = torch.gather(
                normalizer[0],
                1,
                pending_addresses,
            )
            gathered_last_update = torch.gather(
                last_update[0],
                1,
                pending_addresses,
            )
            elapsed = next_token_count - gathered_last_update.to(torch.int64)
            write_decay = decays.view(1, -1).pow(
                elapsed.to(statistics_dtype)
            )
            encoded_source = self._encode_semantic_values(
                statistics_source
            )
            updated_memory = (
                write_decay.unsqueeze(-1) * gathered_memory
                + pending_signs.to(statistics_dtype).unsqueeze(-1)
                * encoded_source.unsqueeze(1)
                * write_scales.view(1, -1, 1)
            )
            updated_normalizer = (
                write_decay * gathered_normalizer + write_scales.view(1, -1)
            )
            memory[0].scatter_(
                1,
                pending_addresses.unsqueeze(-1).expand(
                    -1,
                    -1,
                    SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
                ),
                updated_memory,
            )
            normalizer[0].scatter_(1, pending_addresses, updated_normalizer)
            last_update[0].scatter_(
                1,
                pending_addresses,
                torch.full_like(pending_addresses, next_token_count, dtype=torch.int32),
            )

        numerator = torch.gather(
            memory[0],
            1,
            query_addresses.unsqueeze(-1).expand(
                -1,
                -1,
                SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
            ),
        )
        query_normalizers = torch.gather(normalizer[0], 1, query_addresses)
        query_last_update = torch.gather(last_update[0], 1, query_addresses)
        query_elapsed = next_token_count - query_last_update.to(torch.int64)
        query_decay = decays.view(1, -1).pow(
            query_elapsed.to(statistics_dtype)
        )
        numerator = (
            numerator
            * query_decay.unsqueeze(-1)
            * query_signs.to(statistics_dtype).unsqueeze(-1)
        )
        query_normalizers = query_normalizers * query_decay
        denominator_floor = self._associative_denominator_floor(
            state.associative_semantic_memory,
            write_scales.view(1, SEMANTIC_COUNT_SKETCH_TABLES, 1),
        )
        table_reads = numerator / query_normalizers.unsqueeze(-1).clamp_min(
            denominator_floor
        )
        semantic_reads, semantic_active = self._semantic_count_sketch_group_reads(
            table_reads,
            query_normalizers,
        )
        semantic_reads = tuple(read.to(source.dtype) for read in semantic_reads)
        return (
            semantic_reads,
            semantic_active,
            memory,
            normalizer,
            last_update,
            context,
            query_codes,
        )

    def _lazy_multiscale_count_sketch_step(
        self,
        source: Tensor,
        state: NativePCState,
        token_ids: Tensor,
        *,
        next_token_count: int | Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor, Tensor]:
        if state.associative_memory is None:
            raise RuntimeError("associative memory state is missing")
        if state.associative_normalizer is None:
            raise RuntimeError("associative normalizer state is missing")
        if state.associative_last_update is None:
            raise RuntimeError("lazy associative timestamps are missing")
        memory_layers = self._associative_layer_count()
        memory_source = source[:memory_layers]
        statistics_dtype = state.associative_memory.dtype
        statistics_source = memory_source.to(statistics_dtype)
        encoded_source = self._encode_associative_values(statistics_source)
        indexed = (
            self.config.associative_memory_addressing
            in INDEXED_COUNT_SKETCH_ADDRESSINGS
        )
        dual_order = (
            self.config.associative_memory_addressing
            in DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS
        )
        if indexed:
            if state.associative_context_tokens is None:
                raise RuntimeError("indexed lazy prior token state is missing")
            if dual_order and state.associative_previous_context_tokens is None:
                raise RuntimeError("dual-order second prior token state is missing")
            keys = None
        else:
            if state.associative_keys is None:
                raise RuntimeError("associative key state is missing")
            keys = self._associative_keys(statistics_source, token_ids=token_ids)
        if (
            self.config.fused_indexed_count_sketch_inference
            and indexed
            and _uses_fp16_cuda_fused_inference(source)
        ):
            assert state.associative_context_tokens is not None
            if (
                self.indexed_count_sketch_addresses is None
                or self.indexed_count_sketch_signs is None
            ):
                raise RuntimeError("indexed CountSketch buffers are missing")
            if dual_order:
                assert state.associative_previous_context_tokens is not None
                hash_constants = (
                    self.count_sketch_hash_multipliers,
                    self.count_sketch_hash_offsets,
                    self.count_sketch_sign_multipliers,
                    self.count_sketch_sign_offsets,
                )
                if any(value is None for value in hash_constants):
                    raise RuntimeError("dual-order CountSketch hashes are missing")
                fused_read = fused_dual_order_count_sketch_step(
                    state.associative_memory,
                    state.associative_normalizer,
                    state.associative_last_update,
                    encoded_source,
                    state.associative_previous_context_tokens,
                    state.associative_context_tokens,
                    token_ids,
                    self.indexed_count_sketch_addresses,
                    self.indexed_count_sketch_signs,
                    *hash_constants,
                    vocab_size=self.config.vocab_size,
                    next_token_count=next_token_count,
                    write_enabled=state.token_count > 0,
                    fast_decay=self.config.associative_memory_decay,
                    rms_eps=self.config.rms_eps,
                    zero_anchored_pair_consensus=(
                        self.config.associative_memory_addressing
                        == ZERO_ANCHORED_PAIR_CONSENSUS_ADDRESSING
                    ),
                )
            else:
                fused_read = fused_indexed_count_sketch_step(
                    state.associative_memory,
                    state.associative_normalizer,
                    state.associative_last_update,
                    encoded_source,
                    state.associative_context_tokens,
                    token_ids,
                    self.indexed_count_sketch_addresses,
                    self.indexed_count_sketch_signs,
                    next_token_count=next_token_count,
                    write_enabled=state.token_count > 0,
                    fast_decay=self.config.associative_memory_decay,
                    rms_eps=self.config.rms_eps,
                )
            read = self._decode_associative_values(fused_read)
            if memory_layers < self.config.depth:
                read = torch.cat(
                    (
                        read,
                        torch.zeros(
                            self.config.depth - memory_layers,
                            *read.shape[1:],
                            device=read.device,
                            dtype=read.dtype,
                        ),
                    ),
                    dim=0,
                )
            return (
                read,
                state.associative_memory,
                state.associative_normalizer,
                None,
                state.associative_last_update,
                encoded_source,
            )
        table_count = self._count_sketch_table_count()
        bucket_count = self._multiscale_count_sketch_bucket_count()
        value_width = self._associative_value_width()
        table_decays = self._multiscale_count_sketch_decays(
            state.associative_memory
        )
        table_write_scales = (
            1.0 - table_decays
            if self.config.scale_invariant_associative_state
            else torch.ones_like(table_decays)
        )
        functional = torch.is_grad_enabled()
        memory = (
            state.associative_memory.clone()
            if functional
            else state.associative_memory
        )
        normalizer = (
            state.associative_normalizer.clone()
            if functional
            else state.associative_normalizer
        )
        last_update = (
            state.associative_last_update.clone()
            if functional
            else state.associative_last_update
        )
        next_token_count = state.token_count + 1

        if state.token_count > 0:
            if indexed:
                assert state.associative_context_tokens is not None
                pending_addresses, pending_signs = (
                    self._indexed_multiscale_count_sketch_addresses_and_signs(
                        state.associative_context_tokens,
                        state.associative_previous_context_tokens,
                    )
                )
                pending_addresses = pending_addresses.unsqueeze(0).expand(
                    memory_layers, -1, -1
                )
                pending_signs = pending_signs.unsqueeze(0).expand(
                    memory_layers, -1, -1
                )
                pending_strength = torch.ones_like(pending_signs)
            else:
                assert state.associative_keys is not None
                key_tables = state.associative_keys.reshape(
                    memory_layers,
                    token_ids.shape[0],
                    table_count,
                    bucket_count,
                )
                pending_local_addresses = key_tables.abs().argmax(dim=-1)
                pending_signs = torch.gather(
                    key_tables,
                    3,
                    pending_local_addresses.unsqueeze(-1),
                ).squeeze(-1)
                pending_strength = key_tables.abs().sum(dim=-1)
                table_offsets = (
                    torch.arange(
                        table_count,
                        device=token_ids.device,
                        dtype=torch.long,
                    )
                    * bucket_count
                )
                pending_addresses = pending_local_addresses + table_offsets.view(
                    1, 1, table_count
                )
            gathered_memory = torch.gather(
                memory,
                2,
                pending_addresses.unsqueeze(-1).expand(
                    -1, -1, -1, value_width
                ),
            )
            gathered_normalizer = torch.gather(
                normalizer,
                2,
                pending_addresses,
            )
            gathered_last_update = torch.gather(
                last_update,
                2,
                pending_addresses,
            )
            elapsed = next_token_count - gathered_last_update.to(torch.int64)
            decay = table_decays.view(1, 1, table_count).pow(
                elapsed.to(statistics_dtype)
            )
            write_scale = table_write_scales.view(1, 1, table_count)
            updated_memory = (
                decay.unsqueeze(-1) * gathered_memory
                + pending_signs.unsqueeze(-1)
                * encoded_source.unsqueeze(2)
                * pending_strength.unsqueeze(-1)
                * write_scale.unsqueeze(-1)
            )
            updated_normalizer = (
                decay * gathered_normalizer
                + pending_strength * write_scale
            )
            memory.scatter_(
                2,
                pending_addresses.unsqueeze(-1).expand(
                    -1, -1, -1, value_width
                ),
                updated_memory,
            )
            normalizer.scatter_(2, pending_addresses, updated_normalizer)
            last_update.scatter_(
                2,
                pending_addresses,
                next_token_count,
            )

        if indexed:
            addresses, signs = (
                self._indexed_multiscale_count_sketch_addresses_and_signs(
                    token_ids,
                    state.associative_context_tokens,
                )
            )
        else:
            addresses, signs = self._multiscale_count_sketch_addresses_and_signs(
                token_ids
            )
        gather_addresses = addresses.unsqueeze(0).expand(memory_layers, -1, -1)
        numerator = torch.gather(
            memory,
            2,
            gather_addresses.unsqueeze(-1).expand(-1, -1, -1, value_width),
        )
        denominator = torch.gather(normalizer, 2, gather_addresses).unsqueeze(-1)
        query_last_update = torch.gather(last_update, 2, gather_addresses)
        query_elapsed = next_token_count - query_last_update.to(torch.int64)
        query_decay = table_decays.view(1, 1, table_count).pow(
            query_elapsed.to(statistics_dtype)
        )
        numerator = (
            numerator
            * query_decay.unsqueeze(-1)
            * signs.to(statistics_dtype).unsqueeze(0).unsqueeze(-1)
        )
        denominator = denominator * query_decay.unsqueeze(-1)
        denominator = denominator.clamp_min(
            self._associative_denominator_floor(
                state.associative_memory,
                table_write_scales.view(1, 1, table_count, 1),
            )
        )
        table_reads = numerator / denominator
        group_width = MULTISCALE_COUNT_SKETCH_MEDIAN_TABLES
        group_reads = tuple(
            self._reduce_count_sketch_read_group(
                table_reads[:, :, start : start + group_width],
                start // group_width,
            )
            for start in range(0, table_count, group_width)
        )
        read = self._decode_associative_values(
            torch.cat(group_reads, dim=-1)
        ).to(source.dtype)
        if memory_layers < self.config.depth:
            read = torch.cat(
                (
                    read,
                    torch.zeros(
                        self.config.depth - memory_layers,
                        *read.shape[1:],
                        device=read.device,
                        dtype=read.dtype,
                    ),
                ),
                dim=0,
            )
        return read, memory, normalizer, keys, last_update, encoded_source

    def _associative_step(
        self,
        source: Tensor,
        state: NativePCState,
        token_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if state.associative_memory is None:
            raise RuntimeError("associative memory state is missing")
        if state.associative_normalizer is None or state.associative_keys is None:
            raise RuntimeError("associative address state is missing")
        if self.config.associative_memory_addressing == "multiscale_count_sketch":
            return self._multiscale_count_sketch_step(source, state, token_ids)
        memory_layers = self._associative_layer_count()
        memory_source = source[:memory_layers]
        if self.config.associative_memory_addressing == "block_delta":
            reads = []
            memories = []
            normalizers = []
            keys = []
            for layer in range(memory_layers):
                read, memory, normalizer, key = self._block_delta_sequence(
                    memory_source[layer].unsqueeze(1),
                    state.associative_memory[layer],
                    layer,
                )
                reads.append(read[:, 0])
                memories.append(memory)
                normalizers.append(normalizer)
                keys.append(key)
            read_stack = torch.stack(reads)
            if memory_layers < self.config.depth:
                read_stack = torch.cat(
                    (
                        read_stack,
                        torch.zeros(
                            self.config.depth - memory_layers,
                            *read_stack.shape[1:],
                            device=read_stack.device,
                            dtype=read_stack.dtype,
                        ),
                    ),
                    dim=0,
                )
            return (
                read_stack,
                torch.stack(memories),
                torch.stack(normalizers),
                torch.stack(keys),
            )
        encoded_source = self._encode_associative_values(memory_source)
        keys = self._associative_keys(
            memory_source,
            token_ids=token_ids,
            context_token_ids=state.associative_context_tokens,
        )
        write_scale = self._associative_state_write_scale(source)
        memory = (
            self.config.associative_memory_decay * state.associative_memory
            + state.associative_keys.unsqueeze(-1)
            * encoded_source.unsqueeze(-2)
            * write_scale
        )
        normalizer = (
            self.config.associative_memory_decay * state.associative_normalizer
            + state.associative_keys * write_scale
        )
        numerator = torch.matmul(keys.unsqueeze(-2), memory).squeeze(-2)
        denominator_floor = self._associative_denominator_floor(
            source,
            write_scale,
        )
        denominator = (
            (keys * normalizer).sum(dim=-1, keepdim=True).clamp_min(denominator_floor)
        )
        read = self._decode_associative_values(numerator / denominator)
        if (
            self.config.associative_memory_addressing
            in {
                "token_plus_bigram_hash",
                "token_plus_context_bucket",
                "token_plus_latent_hash",
                "token_plus_latent_codebook",
            }
            and self.config.associative_memory_context_gate_count > 0
        ):
            context_confidence = denominator / (
                denominator
                + self.config.associative_memory_context_gate_count * write_scale
            )
            confidence = torch.cat(
                (torch.ones_like(context_confidence[:1]), context_confidence[1:]),
                dim=0,
            )
            read = read * confidence
        if memory_layers < self.config.depth:
            read = torch.cat(
                (
                    read,
                    torch.zeros(
                        self.config.depth - memory_layers,
                        *read.shape[1:],
                        device=read.device,
                        dtype=read.dtype,
                    ),
                ),
                dim=0,
            )
        return read, memory, normalizer, keys

    def _associative_sequence(
        self,
        source: Tensor,
        initial_memory: Tensor,
        initial_normalizer: Tensor,
        initial_key: Tensor,
        layer: int,
        token_ids: Tensor | None = None,
        initial_context_token: Tensor | None = None,
        reset_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.config.associative_memory_addressing == "block_delta":
            return self._block_delta_sequence(source, initial_memory, layer)
        if (
            self.config.associative_memory_addressing
            in MULTISCALE_COUNT_SKETCH_ADDRESSINGS
        ):
            if token_ids is None:
                raise ValueError("multiscale CountSketch sequence requires token ids")
            return self._multiscale_count_sketch_sequence(
                source,
                initial_memory,
                initial_normalizer,
                initial_key,
                token_ids,
                initial_context_token,
                reset_mask,
            )
        context_token_ids = None
        if self.config.associative_memory_addressing in {
            "token_bigram_hash",
            "token_plus_bigram_hash",
            "token_plus_context_bucket",
        }:
            if token_ids is None or initial_context_token is None:
                raise ValueError(
                    "contextual sequence requires tokens and initial context"
                )
            context_token_ids = torch.cat(
                (initial_context_token.unsqueeze(1), token_ids[:, :-1]),
                dim=1,
            )
            if reset_mask is not None:
                context_token_ids = torch.where(
                    reset_mask,
                    torch.full_like(context_token_ids, self.config.bos_token_id),
                    context_token_ids,
                )
        if self.config.associative_memory_addressing in {
            "token_hash",
            "token_bigram_hash",
            "token_plus_bigram_hash",
            "token_plus_context_bucket",
            "token_plus_latent_hash",
            "token_plus_latent_codebook",
        }:
            if token_ids is None:
                raise ValueError("token associative sequence requires token ids")
            return self._token_associative_sequence(
                source,
                initial_memory,
                initial_normalizer,
                initial_key,
                layer,
                token_ids,
                context_token_ids,
                reset_mask,
            )
        keys = self._associative_keys(
            source,
            layer,
            token_ids,
            context_token_ids,
        )
        encoded_source = self._encode_associative_values(source)
        pending_keys = torch.cat((initial_key.unsqueeze(1), keys[:, :-1]), dim=1)
        write_scale = self._associative_state_write_scale(source)
        memory = _constant_decay_scan(
            pending_keys.unsqueeze(-1) * encoded_source.unsqueeze(-2) * write_scale,
            initial_memory,
            self.config.associative_memory_decay,
            fused=self.config.fused_constant_decay_scan,
            reset_mask=reset_mask,
        )
        normalizer = _constant_decay_scan(
            pending_keys * write_scale,
            initial_normalizer,
            self.config.associative_memory_decay,
            fused=self.config.fused_constant_decay_scan,
            reset_mask=reset_mask,
        )
        numerator = torch.matmul(keys.unsqueeze(-2), memory).squeeze(-2)
        denominator_floor = self._associative_denominator_floor(
            source,
            write_scale,
        )
        denominator = (
            (keys * normalizer).sum(dim=-1, keepdim=True).clamp_min(denominator_floor)
        )
        read = self._decode_associative_values(numerator / denominator)
        if (
            self.config.associative_memory_addressing
            in {"token_plus_bigram_hash", "token_plus_context_bucket"}
            and layer > 0
            and self.config.associative_memory_context_gate_count > 0
        ):
            read = (
                read
                * denominator
                / (
                    denominator
                    + self.config.associative_memory_context_gate_count * write_scale
                )
            )
        return read, memory[:, -1], normalizer[:, -1], keys[:, -1]

    def _associative_reads_for_lower(
        self,
        lower: Tensor,
        token_ids: Tensor | None = None,
    ) -> Tensor:
        if (
            not self.config.bos_boundary_state_reset
            or token_ids is None
        ):
            return self._associative_reads_for_lower_contiguous(lower, token_ids)
        if self.training and not self.config.bos_boundary_exact_segmented_training:
            return self._associative_reads_for_lower_contiguous(
                lower,
                token_ids,
                apply_boundary_resets=True,
            )
        segments = shared_bos_segments(token_ids, self.config.bos_token_id)
        if len(segments) == 1 and not segments[0][2]:
            return self._associative_reads_for_lower_contiguous(
                lower,
                token_ids,
                apply_boundary_resets=False,
            )
        return torch.cat(
            [
                self._associative_reads_for_lower_contiguous(
                    lower[:, :, start:end],
                    token_ids[:, start:end],
                    apply_boundary_resets=False,
                )
                for start, end, _reset in segments
            ],
            dim=2,
        )

    def _associative_reads_for_lower_contiguous(
        self,
        lower: Tensor,
        token_ids: Tensor | None = None,
        *,
        apply_boundary_resets: bool = True,
    ) -> Tensor:
        """Rebuild training-batch reads while treating fast weights as local state."""

        if self.config.associative_memory_rank == 0:
            return torch.zeros_like(lower)
        if lower.ndim != 4 or lower.shape[0] != self.config.depth:
            raise ValueError("lower must have [layer, batch, sequence, width]")
        batch = lower.shape[1]
        rank = self.config.associative_memory_rank
        value_width = self._associative_value_width()
        statistics_dtype = (
            torch.float32
            if self.config.bf16_training_fp32_recurrent_statistics
            else lower.dtype
        )
        reset_mask = (
            token_ids.eq(self.config.bos_token_id)
            if (
                self.config.bos_boundary_state_reset
                and token_ids is not None
                and apply_boundary_resets
            )
            else None
        )
        reads = []
        for layer in range(self._associative_layer_count()):
            initial_memory = torch.zeros(
                batch,
                rank,
                value_width,
                device=lower.device,
                dtype=statistics_dtype,
            )
            initial_normalizer = torch.zeros(
                batch,
                rank,
                device=lower.device,
                dtype=statistics_dtype,
            )
            initial_key = torch.zeros_like(initial_normalizer)
            initial_context_token = (
                torch.full(
                    (batch,),
                    self.config.bos_token_id,
                    device=lower.device,
                    dtype=torch.long,
                )
                if self.config.associative_memory_addressing
                in {
                    "token_bigram_hash",
                    "token_plus_bigram_hash",
                    "token_plus_context_bucket",
                    *DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS,
                }
                else None
            )
            if self._uses_semantic_count_sketch_backoff():
                if token_ids is None or initial_context_token is None:
                    raise ValueError("semantic CountSketch requires token ids")
                semantic_rank = (
                    SEMANTIC_COUNT_SKETCH_TABLES * SEMANTIC_COUNT_SKETCH_BUCKETS
                )
                semantic_enabled = None
                if self._uses_pair_miss_semantic_backoff():
                    pair_seen, _ = self._pair_recurrence_bloom_sequence(
                        token_ids,
                        initial_context_token,
                        torch.zeros(
                            batch,
                            PAIR_RECURRENCE_BLOOM_GENERATIONS,
                            PAIR_RECURRENCE_BLOOM_PROBES,
                            PAIR_RECURRENCE_BLOOM_BYTES,
                            device=lower.device,
                            dtype=torch.uint8,
                        ),
                        0,
                        reset_mask,
                    )
                    semantic_enabled = (~pair_seen).unsqueeze(-1)
                read, *_ = self._semantic_backoff_associative_sequence(
                    lower[layer],
                    initial_memory,
                    initial_normalizer,
                    initial_key,
                    token_ids,
                    initial_context_token,
                    torch.zeros(
                        batch,
                        semantic_rank,
                        SEMANTIC_COUNT_SKETCH_VALUE_WIDTH,
                        device=lower.device,
                        dtype=statistics_dtype,
                    ),
                    torch.zeros(
                        batch,
                        semantic_rank,
                        device=lower.device,
                        dtype=statistics_dtype,
                    ),
                    torch.zeros(
                        batch,
                        self.config.width,
                        device=lower.device,
                        dtype=statistics_dtype,
                    ),
                    torch.zeros(
                        batch,
                        SEMANTIC_COUNT_SKETCH_TABLES,
                        device=lower.device,
                        dtype=torch.uint8,
                    ),
                    initial_write_enabled=False,
                    semantic_enabled=semantic_enabled,
                    reset_mask=reset_mask,
                )
            else:
                read, _, _, _ = self._associative_sequence(
                    lower[layer],
                    initial_memory,
                    initial_normalizer,
                    initial_key,
                    layer,
                    token_ids,
                    initial_context_token,
                    reset_mask,
                )
            reads.append(read)
        if len(reads) < self.config.depth:
            reads.extend(
                torch.zeros_like(lower[layer])
                for layer in range(len(reads), self.config.depth)
            )
        return torch.stack(reads)

    def _temporal_rotation_for_normalized(
        self,
        normalized: Tensor,
        layer: int | None = None,
    ) -> Tensor | None:
        """Return fixed or token-conditioned paired temporal rotations."""

        if self.temporal_rotation is None:
            return None
        pairs = self.config.width // 2
        if layer is None:
            if (
                normalized.shape[0] != self.config.depth
                or normalized.shape[-1] != self.config.width
            ):
                raise ValueError("normalized temporal source must be layer-leading")
            middle_shape = normalized.shape[1:-1]
            base = self.temporal_rotation.view(
                self.config.depth,
                *(1,) * len(middle_shape),
                pairs,
                2,
            )
            if self.temporal_phase_down is None:
                return base.expand(
                    self.config.depth,
                    *middle_shape,
                    pairs,
                    2,
                )
            if self.temporal_phase_up is None:
                raise RuntimeError("selective temporal phase output is missing")
            flat = normalized.reshape(
                self.config.depth,
                -1,
                self.config.width,
            )
            hidden = torch.tanh(torch.bmm(flat, self.temporal_phase_down))
            phase = SELECTIVE_TEMPORAL_PHASE_LIMIT * torch.tanh(
                torch.bmm(hidden, self.temporal_phase_up)
            )
            phase = phase.reshape(
                self.config.depth,
                *middle_shape,
                pairs,
            )
        else:
            if not 0 <= layer < self.config.depth:
                raise ValueError("temporal phase layer is out of range")
            if normalized.shape[-1] != self.config.width:
                raise ValueError("normalized temporal source width mismatch")
            middle_shape = normalized.shape[:-1]
            base = self.temporal_rotation[layer].view(
                *(1,) * len(middle_shape),
                pairs,
                2,
            )
            if self.temporal_phase_down is None:
                return base.expand(*middle_shape, pairs, 2)
            if self.temporal_phase_up is None:
                raise RuntimeError("selective temporal phase output is missing")
            hidden = torch.tanh(normalized @ self.temporal_phase_down[layer])
            phase = SELECTIVE_TEMPORAL_PHASE_LIMIT * torch.tanh(
                hidden @ self.temporal_phase_up[layer]
            )
        phase_cosine = phase.cos()
        phase_sine = phase.sin()
        base_cosine = base[..., 0]
        base_sine = base[..., 1]
        return torch.stack(
            (
                base_cosine * phase_cosine - base_sine * phase_sine,
                base_sine * phase_cosine + base_cosine * phase_sine,
            ),
            dim=-1,
        )

    def _rotate_temporal_states(
        self,
        value: Tensor,
        rotation: Tensor | None = None,
    ) -> Tensor:
        """Apply row-vector rotations to layer-leading state pairs."""

        if self.temporal_rotation is None:
            return value
        if value.shape[0] != self.config.depth or value.shape[-1] != self.config.width:
            raise ValueError("temporal state must be layer-leading with model width")
        paired = value.reshape(
            *value.shape[:-1],
            self.config.width // 2,
            2,
        )
        if rotation is None:
            rotation = self.temporal_rotation.view(
                self.config.depth,
                *(1,) * (value.ndim - 2),
                self.config.width // 2,
                2,
            )
        elif rotation.shape != paired.shape[:-1] + (2,):
            raise ValueError("temporal rotation does not match state shape")
        cosine = rotation[..., 0]
        sine = rotation[..., 1]
        rotated = torch.stack(
            (
                paired[..., 0] * cosine - paired[..., 1] * sine,
                paired[..., 0] * sine + paired[..., 1] * cosine,
            ),
            dim=-1,
        )
        return rotated.reshape_as(value)

    def _pairwise_delta_erase_factors(
        self,
        normalized: Tensor,
        layer: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | None:
        """Build content keys and gated erase vectors for each state group."""

        if self.pairwise_delta_down is None:
            return None
        if (
            self.pairwise_delta_key_up is None
            or self.pairwise_delta_erase_up is None
            or self.pairwise_delta_erase_bias is None
            or self.pairwise_delta_strength is None
        ):
            raise RuntimeError("pairwise delta erase parameters are incomplete")
        group_width = self.config.pairwise_delta_group_width
        groups = self.config.width // group_width
        if layer is None:
            if normalized.shape[0] != self.config.depth:
                raise ValueError("pairwise delta source must be layer-leading")
            flat = normalized.reshape(
                self.config.depth,
                -1,
                self.config.width,
            )
            hidden = torch.tanh(torch.bmm(flat, self.pairwise_delta_down))
            key_raw = torch.bmm(hidden, self.pairwise_delta_key_up)
            erase = torch.sigmoid(
                torch.bmm(hidden, self.pairwise_delta_erase_up)
                + self.pairwise_delta_erase_bias
            )
            middle_shape = normalized.shape[1:-1]
            hidden = hidden.reshape(
                self.config.depth,
                *middle_shape,
                self.config.pairwise_delta_erase_rank,
            )
            key_raw = key_raw.reshape(
                self.config.depth,
                *middle_shape,
                groups,
                group_width,
            )
            erase = erase.reshape(
                self.config.depth,
                *middle_shape,
                groups,
                group_width,
            )
            strength = torch.tanh(self.pairwise_delta_strength).view(
                self.config.depth,
                *(1,) * (len(middle_shape) + 3),
            )
        else:
            if not 0 <= layer < self.config.depth:
                raise ValueError("pairwise delta layer is out of range")
            hidden = torch.tanh(normalized @ self.pairwise_delta_down[layer])
            key_raw = (hidden @ self.pairwise_delta_key_up[layer]).reshape(
                *normalized.shape[:-1],
                groups,
                group_width,
            )
            erase = torch.sigmoid(
                hidden @ self.pairwise_delta_erase_up[layer]
                + self.pairwise_delta_erase_bias[layer, 0]
            ).reshape(*normalized.shape[:-1], groups, group_width)
            strength = torch.tanh(self.pairwise_delta_strength[layer])
        key = F.normalize(key_raw, dim=-1, eps=self.config.rms_eps)
        return hidden, key_raw, key, erase, erase * key, strength

    def _pair_transition_matrix(
        self,
        normalized: Tensor,
        paired_decay: Tensor,
        rotation: Tensor,
        layer: int | None = None,
    ) -> Tensor:
        """Build base rotating retention plus content-directed rank-one erase."""

        matrix = _pair_recurrence_matrix(paired_decay, rotation)
        factors = self._pairwise_delta_erase_factors(normalized, layer)
        if factors is None:
            return matrix
        _, _, key, _, erase_vector, strength = factors
        correction = erase_vector.unsqueeze(-1) * key.unsqueeze(-2)
        return matrix - strength * correction

    def _group4_transition_matrix(
        self,
        normalized: Tensor,
        grouped_decay: Tensor,
        rotation: Tensor,
        layer: int | None = None,
    ) -> Tensor:
        """Build pair-rotation base plus rank-one mixing across four channels."""

        matrix = _group4_recurrence_matrix(grouped_decay, rotation)
        factors = self._pairwise_delta_erase_factors(normalized, layer)
        if factors is None:
            raise RuntimeError("grouped delta transition factors are missing")
        _, _, key, _, erase_vector, strength = factors
        correction = erase_vector.unsqueeze(-1) * key.unsqueeze(-2)
        return matrix - strength * correction

    def _layer_schedule(
        self,
        start: float,
        end: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return torch.linspace(
            start,
            end,
            self.config.depth,
            device=device,
            dtype=dtype,
        ).view(self.config.depth, 1, 1)

    def _residual_stream_scale(self, layer: int) -> float | Tensor:
        if self.residual_stream_scales is not None:
            return self.residual_stream_scales[layer, 0, 0]
        if self.config.depth == 1:
            return self.config.residual_stream_scale_start
        fraction = layer / float(self.config.depth - 1)
        return (
            (1.0 - fraction) * self.config.residual_stream_scale_start
            + fraction * self.config.residual_stream_scale_end
        )

    def _sensory_reinjection_scale(self, layer: int) -> float:
        if self.config.depth == 1:
            return self.config.sensory_reinjection_start
        fraction = layer / float(self.config.depth - 1)
        return (
            (1.0 - fraction) * self.config.sensory_reinjection_start
            + fraction * self.config.sensory_reinjection_end
        )

    def _persistent_transition_adapter_step(
        self,
        normalized: Tensor,
        initial: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.persistent_transition_adapter_down is None:
            raise RuntimeError("persistent transition adapter is disabled")
        if self.persistent_transition_adapter_up is None:
            raise RuntimeError("persistent transition adapter output is missing")
        if self.persistent_transition_adapter_decay_logit is None:
            raise RuntimeError("persistent transition adapter decay is missing")
        rank = self.config.persistent_transition_adapter_rank
        if normalized.ndim != 3 or normalized.shape[:2] != initial.shape[:2]:
            raise ValueError("persistent transition step dimensions do not align")
        if initial.shape[-1] != rank:
            raise ValueError("persistent transition state rank does not match")
        write = torch.tanh(
            torch.bmm(normalized, self.persistent_transition_adapter_down)
        )
        decay = torch.sigmoid(self.persistent_transition_adapter_decay_logit).unsqueeze(
            1
        )
        state = decay * initial + (1.0 - decay) * write
        return state, torch.bmm(state, self.persistent_transition_adapter_up)

    def _persistent_transition_adapter_sequence(
        self,
        normalized: Tensor,
        initial: Tensor,
        layer: int,
    ) -> tuple[Tensor, Tensor]:
        if self.persistent_transition_adapter_down is None:
            raise RuntimeError("persistent transition adapter is disabled")
        if self.persistent_transition_adapter_up is None:
            raise RuntimeError("persistent transition adapter output is missing")
        if self.persistent_transition_adapter_decay_logit is None:
            raise RuntimeError("persistent transition adapter decay is missing")
        rank = self.config.persistent_transition_adapter_rank
        if normalized.ndim != 3 or initial.shape != (normalized.shape[0], rank):
            raise ValueError("persistent transition sequence state does not align")
        write = torch.tanh(normalized @ self.persistent_transition_adapter_down[layer])
        decay = (
            torch.sigmoid(self.persistent_transition_adapter_decay_logit[layer])
            .view(1, 1, rank)
            .expand_as(write)
        )
        states = _affine_scan(
            decay,
            (1.0 - decay) * write,
            initial,
        )
        return (
            states,
            states @ self.persistent_transition_adapter_up[layer],
        )

    def _project_ternary_dale_ffn_out(
        self,
        hidden: Tensor,
        *,
        layer: int | None,
        target_slice: slice,
        packed_ffn: object | None,
        branch: int,
    ) -> Tensor:
        if not self.config.ternary_dale_quadrant_transition:
            raise RuntimeError("ternary Dale transition is disabled")
        if self.ffn_out_dale_other is None:
            raise RuntimeError("ternary Dale companion banks are missing")
        if packed_ffn is not None:
            return packed_ffn.project_out(  # type: ignore[no-any-return,union-attr]
                hidden,
                layer=layer,
                branch=branch,
            )
        from pfc.phase704_ternary_dale_transition import (
            join_primary_and_other_banks,
            project_dale_banks,
            project_dale_banks_bmm,
        )

        banks = join_primary_and_other_banks(
            self.ffn_out,
            self.ffn_out_dale_other,
        )
        if layer is None:
            banks = banks[:, :, :, target_slice]
            if hidden.shape[0] != self.config.depth:
                raise ValueError("layer-leading Dale hidden state differs")
            original_shape = hidden.shape
            flattened = hidden.reshape(self.config.depth, -1, hidden.shape[-1])
            projected = project_dale_banks_bmm(flattened, banks)
            return projected.reshape(*original_shape[:-1], projected.shape[-1])
        return project_dale_banks(hidden, banks[:, layer, :, target_slice])

    def _project_sign_lifted_dual_ffn_out(
        self,
        hidden: Tensor,
        *,
        layer: int | None,
        target_slice: slice,
        packed_ffn: object | None,
        branch: int,
    ) -> Tensor:
        if not self.config.sign_lifted_dual_ternary_transition:
            raise RuntimeError("sign-lifted dual transition is disabled")
        if packed_ffn is not None:
            return packed_ffn.project_out(  # type: ignore[no-any-return,union-attr]
                hidden,
                layer=layer,
                branch=branch,
            )
        from pfc.phase705_sign_lifted_transition import sign_lifted_projection

        matrix = (
            self.ffn_out[:, :, target_slice]
            if layer is None
            else self.ffn_out[self.transition_owner_index(layer), :, target_slice]
        )
        return sign_lifted_projection(hidden, matrix)

    def _affine_reversible_scale_terms(
        self,
        normalized_condition: Tensor,
        *,
        layer: int | None,
    ):
        if not self.config.affine_reversible_transition:
            raise RuntimeError("affine reversible transition is disabled")
        packed = getattr(self, "_packed_grouped_ternary_affine_scale", None)
        if packed is not None:
            matrix_index = (
                0
                if packed.matrix_count == 1
                else layer
            )
            preactivation = packed.matmul(
                normalized_condition,
                matrix_index=matrix_index,
            )
        else:
            if self.affine_reversible_scale is None:
                raise RuntimeError("affine reversible scale matrix is missing")
            matrix = (
                self.affine_reversible_scale
                if layer is None
                else self.affine_reversible_scale[
                    self.transition_owner_index(layer)
                ]
            )
            preactivation = torch.matmul(normalized_condition, matrix)
        from pfc.phase757_affine_reversible_transition import affine_scale_terms

        return affine_scale_terms(preactivation)

    def _affine_reversible_scale(
        self,
        condition: Tensor,
        *,
        branch: int,
        layer: int | None,
    ):
        if branch not in (0, 1):
            raise ValueError("affine reversible branch must be zero or one")
        half = self.config.width // 2
        condition_slice = slice(half, self.config.width) if branch == 0 else slice(0, half)
        norm = (
            self.state_norm[:, :, condition_slice]
            if layer is None
            else self.state_norm[layer, 0, condition_slice]
        )
        normalized = _rms_norm(condition, norm, self.config.rms_eps)
        return self._affine_reversible_scale_terms(
            normalized,
            layer=layer,
        )

    def _reversible_coupling_terms(
        self,
        condition: Tensor,
        branch: int,
        associative_read: Tensor | None,
        layer: int | None = None,
        bounded_attention_read: Tensor | None = None,
        *,
        token_ids: Tensor | None = None,
        candidate_residual: Tensor | None = None,
        exact_prefill_handoff: bool = False,
        packed_input_projection: Tensor | None = None,
        residual_route_plan: CublasResidualRoutePlan | None = None,
        return_hidden: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Return retention and write terms for one additive coupling branch."""

        if not self.config.reversible_recurrent_coupling:
            raise RuntimeError("reversible recurrent coupling is disabled")
        if branch not in (0, 1):
            raise ValueError("reversible coupling branch must be zero or one")
        if self.transition_write_gate is None:
            raise RuntimeError("reversible coupling write gate is missing")
        if self.transition_write_gate_bias is None:
            raise RuntimeError("reversible coupling write-gate bias is missing")
        width = self.config.width
        half = width // 2
        condition_slice = slice(half, width) if branch == 0 else slice(0, half)
        target_slice = slice(0, half) if branch == 0 else slice(half, width)
        if condition.shape[-1] != half:
            raise ValueError("reversible coupling condition has wrong width")

        compact = self.config.compact_active_triangular_coupling
        packed_ffn = getattr(self, "_packed_grouped_ternary_ffn", None)
        if packed_ffn is not None and compact:
            raise RuntimeError("packed grouped FFN does not support compact coupling")
        if layer is None:
            norm = (
                self.state_norm if compact else self.state_norm[:, :, condition_slice]
            )
            temporal = (
                self.temporal if compact else self.temporal[:, condition_slice, :]
            )
            gate = self.gate if compact else self.gate[:, condition_slice, :]
            write_gate = (
                self.transition_write_gate
                if compact
                else self.transition_write_gate[:, condition_slice, :]
            )
            ffn_in = (
                None
                if (
                    packed_ffn is not None
                    or self.config.contractive_ternary_reciprocal_butterfly_ffn
                )
                else self.ffn_in
                if compact
                else self.ffn_in[:, condition_slice, :]
            )
            ffn_out = (
                None
                if (
                    packed_ffn is not None
                    or self.config.contractive_ternary_reciprocal_butterfly_ffn
                    or self.config.ternary_dale_quadrant_transition
                    or self.config.sign_lifted_dual_ternary_transition
                )
                else self.ffn_out
                if compact
                else self.ffn_out[:, :, target_slice]
            )
            gate_bias = (
                self.gate_bias if compact else self.gate_bias[:, :, target_slice]
            )
            write_bias = (
                self.transition_write_gate_bias
                if compact
                else self.transition_write_gate_bias[:, :, target_slice]
            )
            ffn_in_bias = self.ffn_in_bias
            ffn_out_bias = (
                self.ffn_out_bias if compact else self.ffn_out_bias[:, :, target_slice]
            )
        else:
            if not 0 <= layer < self.config.depth:
                raise ValueError("reversible coupling layer is out of range")
            norm = (
                self.state_norm[layer, 0]
                if compact
                else self.state_norm[layer, 0, condition_slice]
            )
            temporal = (
                self.temporal[layer]
                if compact
                else self.temporal[layer, condition_slice, :]
            )
            gate = self.gate[layer] if compact else self.gate[layer, condition_slice, :]
            write_gate = (
                self.transition_write_gate[layer]
                if compact
                else self.transition_write_gate[layer, condition_slice, :]
            )
            ffn_in = (
                None
                if (
                    packed_ffn is not None
                    or self.config.contractive_ternary_reciprocal_butterfly_ffn
                )
                else self.ffn_in[self.transition_owner_index(layer)]
                if compact
                else self.ffn_in[self.transition_owner_index(layer), condition_slice, :]
            )
            ffn_out = (
                None
                if (
                    packed_ffn is not None
                    or self.config.contractive_ternary_reciprocal_butterfly_ffn
                    or self.config.ternary_dale_quadrant_transition
                    or self.config.sign_lifted_dual_ternary_transition
                )
                else self.ffn_out[self.transition_owner_index(layer)]
                if compact
                else self.ffn_out[self.transition_owner_index(layer), :, target_slice]
            )
            gate_bias = (
                self.gate_bias[layer, 0]
                if compact
                else self.gate_bias[layer, 0, target_slice]
            )
            write_bias = (
                self.transition_write_gate_bias[layer, 0]
                if compact
                else self.transition_write_gate_bias[layer, 0, target_slice]
            )
            ffn_in_bias = self.ffn_in_bias[layer, 0]
            ffn_out_bias = (
                self.ffn_out_bias[layer, 0]
                if compact
                else self.ffn_out_bias[layer, 0, target_slice]
            )

        normalized = _rms_norm(condition, norm, self.config.rms_eps)
        fused_context_bound_decode = bool(
            self.config.fused_context_bound_ternary_decode
            and packed_ffn is not None
            and token_ids is not None
            and token_ids.numel() == 1
            and torch.is_inference_mode_enabled()
            and condition.is_cuda
        )
        ffn_normalized = normalized
        if (
            self.config.context_bound_ternary_parameter_superposition
            and not fused_context_bound_decode
        ):
            if token_ids is None:
                raise ValueError("context-bound FFN requires token ids")
            from pfc.phase736_context_bound_ternary_superposition import (
                bind_walsh_context,
            )

            ffn_normalized = bind_walsh_context(
                normalized,
                token_ids,
                coordinate_offset=condition_slice.start or 0,
            )
        fused_butterfly_decode = (
            self.config.fused_parallel_reciprocal_butterfly_decode
            and layer is None
            and torch.is_inference_mode_enabled()
            and condition.is_cuda
        )
        if self.config.contractive_ternary_reciprocal_butterfly_ffn:
            if packed_ffn is not None or packed_input_projection is not None:
                raise ValueError("butterfly FFN excludes packed dense projections")
            temporal_projection = torch.matmul(normalized, temporal)
            gate_projection = torch.matmul(normalized, gate)
            write_projection = torch.matmul(normalized, write_gate)
            coefficients = self.ffn_in if layer is None else self.ffn_in[self.transition_owner_index(layer)]
            if fused_butterfly_decode:
                from pfc.phase659_triton_reciprocal_butterfly_decode import (
                    fused_reciprocal_butterfly_decode,
                )

                ffn_output = fused_reciprocal_butterfly_decode(
                    normalized,
                    coefficients,
                    ffn_in_bias,
                    branch=branch,
                )
            else:
                padded = normalized.new_zeros(*normalized.shape[:-1], width)
                padded[..., condition_slice] = normalized
                ffn_projection = butterfly_apply(padded, coefficients)
        elif packed_ffn is not None:
            if packed_input_projection is not None:
                raise ValueError("packed grouped FFN conflicts with packed prefill input")
            temporal_projection = torch.matmul(normalized, temporal)
            gate_projection = torch.matmul(normalized, gate)
            write_projection = torch.matmul(normalized, write_gate)
            ffn_projection = packed_ffn.project_in(
                ffn_normalized,
                layer=layer,
                branch=branch,
                walsh_token_ids=token_ids if fused_context_bound_decode else None,
            )
        elif packed_input_projection is None:
            assert ffn_in is not None
            temporal_projection = torch.matmul(normalized, temporal)
            gate_projection = torch.matmul(normalized, gate)
            write_projection = torch.matmul(normalized, write_gate)
            ffn_projection = self._ffn_matmul(ffn_normalized, ffn_in)
        else:
            if compact or layer is None:
                raise ValueError(
                    "packed projection requires one noncompact prefill layer"
                )
            from pfc.phase448_exact_packed_prefill_projection import (
                packed_reversible_branch_projection,
            )

            (
                temporal_projection,
                gate_projection,
                write_projection,
                ffn_projection,
            ) = packed_reversible_branch_projection(
                normalized,
                packed_input_projection[condition_slice],
                width=width,
                ffn_input_width=ffn_in.shape[-1],
                ffn_normalized=(
                    ffn_normalized
                    if self.config.context_bound_ternary_parameter_superposition
                    else None
                ),
            )
        temporal_primary, temporal_modulation = temporal_projection.chunk(
            2,
            dim=-1,
        )
        gate_primary, gate_modulation = gate_projection.chunk(2, dim=-1)
        write_primary, memory_primary = write_projection.chunk(2, dim=-1)
        if self.config.contractive_ternary_reciprocal_butterfly_ffn:
            if not fused_butterfly_decode:
                hidden = torch.tanh(ffn_projection + ffn_in_bias)
                ffn_output = butterfly_apply(
                    hidden,
                    coefficients,
                    reverse=True,
                )[..., target_slice]
        else:
            hidden = self._ffn_hidden(ffn_projection + ffn_in_bias)
            if (
                self.config.context_bound_ternary_parameter_superposition
                and not fused_context_bound_decode
            ):
                assert token_ids is not None
                from pfc.phase736_context_bound_ternary_superposition import (
                    bind_walsh_context,
                )

                hidden = bind_walsh_context(hidden, token_ids)
        consequence_hidden: Tensor | None = None
        if self.consequence_ffn_in is not None:
            if (
                self.consequence_ffn_out is None
                or self.consequence_ffn_in_bias is None
                or self.consequence_ffn_out_bias is None
            ):
                raise RuntimeError("consequence FFN compartment is incomplete")
            consequence_in = (
                self.consequence_ffn_in[:, condition_slice, :]
                if layer is None
                else self.consequence_ffn_in[layer, condition_slice, :]
            )
            consequence_in_bias = (
                self.consequence_ffn_in_bias
                if layer is None
                else self.consequence_ffn_in_bias[layer, 0]
            )
            consequence_hidden = self._ffn_hidden(
                torch.matmul(normalized, consequence_in) + consequence_in_bias
            )
        if self.config.contractive_ternary_reciprocal_butterfly_ffn:
            pass
        elif self.config.shared_features_routed_ternary_ffn_output:
            if token_ids is None:
                raise ValueError("routed FFN output requires token ids")
            if self.ffn_routed_out is None:
                raise RuntimeError("routed FFN output bank is missing")
            extra_out = (
                self.ffn_routed_out[:, :, :, target_slice]
                if layer is None
                else self.ffn_routed_out[:, layer, :, target_slice]
            )
            if packed_ffn is None:
                assert ffn_out is not None
                base_out = ffn_out
            else:
                base_out = (
                    self.ffn_out[:, :, target_slice]
                    if layer is None
                    else self.ffn_out[self.transition_owner_index(layer), :, target_slice]
                )
            from pfc.phase637_shared_features_routed_output import (
                routed_output_bank,
                selected_decode_output,
            )

            if token_ids.numel() == 1:
                ffn_output = selected_decode_output(
                    hidden,
                    base_out,
                    extra_out,
                    token_ids,
                )
            else:
                ffn_output = routed_output_bank(
                    hidden,
                    base_out,
                    extra_out,
                    token_ids,
                )
        elif self.config.self_gated_ternary_hidden_allocation:
            if token_ids is None:
                raise ValueError("self-gated hidden allocation requires token ids")
            if self.self_gated_hidden_allocation is None:
                raise RuntimeError("self-gated hidden allocation is missing")
            from pfc.phase825_self_gated_ternary_hidden_allocation import (
                self_gated_ternary_hidden_allocation,
            )

            allocation = (
                self.self_gated_hidden_allocation
                if layer is None
                else self.self_gated_hidden_allocation[
                    self.transition_owner_index(layer)
                ]
            )
            hidden, _selected_allocation, _controller_signs = (
                self_gated_ternary_hidden_allocation(
                    hidden,
                    self.embedding(token_ids),
                    allocation,
                )
            )
            if packed_ffn is None:
                assert ffn_out is not None
                ffn_output = self._ffn_matmul(hidden, ffn_out)
            else:
                ffn_output = packed_ffn.project_out(
                    hidden,
                    layer=layer,
                    branch=branch,
                )
        elif self.config.routed_ternary_ffn_channel_allocation:
            if token_ids is None:
                raise ValueError("routed FFN channel allocation requires token ids")
            if self.ffn_channel_allocation is None:
                raise RuntimeError("routed FFN channel allocation is missing")
            allocation_vectors = (
                self.ffn_channel_allocation
                if layer is None
                else self.ffn_channel_allocation[:, layer]
            )
            if self.config.zero_mean_contrastive_ffn_channel_allocation:
                from pfc.phase636_zero_mean_contrastive_ffn_allocation import (
                    allocate_contrastive_ffn_channels,
                )

                hidden, _selected_allocation = allocate_contrastive_ffn_channels(
                    hidden,
                    token_ids,
                    allocation_vectors,
                )
            else:
                from pfc.phase635_routed_ffn_channel_allocation import (
                    allocate_ffn_channels,
                )

                hidden, _selected_allocation = allocate_ffn_channels(
                    hidden,
                    token_ids,
                    allocation_vectors,
                )
            if packed_ffn is None:
                assert ffn_out is not None
                ffn_output = self._ffn_matmul(hidden, ffn_out)
            else:
                ffn_output = packed_ffn.project_out(
                    hidden,
                    layer=layer,
                    branch=branch,
                )
        elif self.config.selected_diagonal_swiglu_residuals:
            if token_ids is None:
                raise ValueError("diagonal SwiGLU residuals require token ids")
            if (
                self.diagonal_swiglu_gate is None
                or self.diagonal_swiglu_value is None
                or self.diagonal_swiglu_output is None
            ):
                raise RuntimeError("diagonal SwiGLU residual vectors are missing")
            gate_vectors = (
                self.diagonal_swiglu_gate[:, :, condition_slice]
                if layer is None
                else self.diagonal_swiglu_gate[:, layer, condition_slice]
            )
            value_vectors = (
                self.diagonal_swiglu_value[:, :, condition_slice]
                if layer is None
                else self.diagonal_swiglu_value[:, layer, condition_slice]
            )
            output_vectors = (
                self.diagonal_swiglu_output[:, :, target_slice]
                if layer is None
                else self.diagonal_swiglu_output[:, layer, target_slice]
            )
            from pfc.phase634_diagonal_swiglu_residuals import (
                selected_diagonal_swiglu_residual,
            )

            residual_addition = selected_diagonal_swiglu_residual(
                normalized,
                token_ids,
                gate_vectors,
                value_vectors,
                output_vectors,
            )
            if packed_ffn is None:
                assert ffn_out is not None
                base_output = self._ffn_matmul(hidden, ffn_out)
            else:
                base_output = packed_ffn.project_out(
                    hidden,
                    layer=layer,
                    branch=branch,
                )
            ffn_output = base_output + residual_addition
        elif self.config.selected_diagonal_ternary_residuals:
            if token_ids is None:
                raise ValueError("diagonal ternary residuals require token ids")
            if self.diagonal_residual is None:
                raise RuntimeError("diagonal residual vectors are missing")
            vectors = (
                self.diagonal_residual[:, :, target_slice]
                if layer is None
                else self.diagonal_residual[:, layer, target_slice]
            )
            if self.config.selected_diagonal_decode_gather:
                from pfc.phase633_exact_decode_gather import (
                    selected_diagonal_residual,
                )
            else:
                from pfc.phase632_diagonal_ternary_residuals import (
                    selected_diagonal_residual,
                )

            residual_addition = selected_diagonal_residual(
                normalized,
                token_ids,
                vectors,
            )
            if packed_ffn is None:
                assert ffn_out is not None
                base_output = self._ffn_matmul(hidden, ffn_out)
            else:
                base_output = packed_ffn.project_out(
                    hidden,
                    layer=layer,
                    branch=branch,
                )
            ffn_output = base_output + residual_addition
        elif self.config.selected_rank_one_ternary_residuals:
            if token_ids is None:
                raise ValueError("rank-one ternary residuals require token ids")
            if (
                self.rank_one_residual_source is None
                or self.rank_one_residual_output is None
            ):
                raise RuntimeError("rank-one residual vectors are missing")
            source_vectors = (
                self.rank_one_residual_source[:, :, condition_slice]
                if layer is None
                else self.rank_one_residual_source[:, layer, condition_slice]
            )
            output_vectors = (
                self.rank_one_residual_output[:, :, target_slice]
                if layer is None
                else self.rank_one_residual_output[:, layer, target_slice]
            )
            from pfc.phase631_rank_one_ternary_residuals import (
                selected_rank_one_residual,
            )

            residual_addition = selected_rank_one_residual(
                normalized,
                token_ids,
                source_vectors,
                output_vectors,
            )
            if packed_ffn is None:
                assert ffn_out is not None
                base_output = self._ffn_matmul(hidden, ffn_out)
            else:
                base_output = packed_ffn.project_out(
                    hidden,
                    layer=layer,
                    branch=branch,
                )
            ffn_output = base_output + residual_addition
        elif self.config.deterministic_ternary_ffn_residual_experts:
            if token_ids is None:
                raise ValueError("deterministic residual FFN experts require token ids")
            if (
                self.ffn_residual_expert_in is None
                or self.ffn_residual_expert_out is None
                or self.ffn_residual_expert_in_bias is None
            ):
                raise RuntimeError("deterministic residual FFN expert bank is missing")
            residual_in = (
                self.ffn_residual_expert_in[:, :, condition_slice, :]
                if layer is None
                else self.ffn_residual_expert_in[:, layer, condition_slice, :]
            )
            residual_out = (
                self.ffn_residual_expert_out[:, :, :, target_slice]
                if layer is None
                else self.ffn_residual_expert_out[:, layer, :, target_slice]
            )
            residual_bias = (
                self.ffn_residual_expert_in_bias[:, :, 0]
                if layer is None
                else self.ffn_residual_expert_in_bias[:, layer, 0]
            )
            selected_only_decode = bool(
                self.config.selected_only_residual_expert_decode
                and layer is None
                and normalized.shape[1] == 1
            )
            selected_only_sequence = bool(
                self.config.selected_only_residual_expert_sequence
                and layer is not None
                and not self.training
                and normalized.is_cuda
            )
            cublas_routed_prefill = bool(
                self.config.cublas_routed_residual_expert_prefill
                and layer is not None
                and residual_route_plan is not None
            )
            if cublas_routed_prefill:
                from pfc.phase629_cublas_routed_residual_prefill import (
                    cublas_routed_residual_prefill,
                )

                residual_addition = cublas_routed_residual_prefill(
                    normalized,
                    residual_in,
                    residual_bias,
                    residual_out,
                    residual_route_plan,
                )
            elif selected_only_sequence:
                from pfc.phase628_triton_routed_residual_sequence import (
                    triton_routed_residual_sequence,
                )

                residual_addition = triton_routed_residual_sequence(
                    normalized,
                    token_ids,
                    residual_in,
                    residual_bias,
                    residual_out,
                )
            elif selected_only_decode:
                from pfc.phase627_selected_only_residual_decode import (
                    select_single_decode_expert_parameters,
                )

                selected_in, selected_bias, selected_out = (
                    select_single_decode_expert_parameters(
                        residual_in,
                        residual_bias,
                        residual_out,
                        token_ids,
                    )
                )
                residual_projection = torch.matmul(normalized, selected_in)
                residual_projection = residual_projection + selected_bias[:, None, :]
                residual_hidden = self._ffn_hidden(residual_projection)
                residual_output = torch.matmul(residual_hidden, selected_out)
                residual_addition = residual_output
            else:
                residual_projection = torch.stack(
                    tuple(
                        torch.matmul(normalized, residual_in[expert])
                        for expert in range(residual_in.shape[0])
                    ),
                    dim=0,
                )
                if layer is None:
                    residual_projection = (
                        residual_projection + residual_bias[:, :, None, :]
                    )
                else:
                    residual_projection = (
                        residual_projection + residual_bias[:, None, None, :]
                    )
                residual_hidden = self._ffn_hidden(residual_projection)
                residual_output = torch.stack(
                    tuple(
                        torch.matmul(residual_hidden[expert], residual_out[expert])
                        for expert in range(residual_out.shape[0])
                    ),
                    dim=0,
                )
                from pfc.phase626_shared_ternary_residual_experts import (
                    select_routed_expert,
                )

                residual_addition = select_routed_expert(
                    residual_output,
                    token_ids,
                )
            if packed_ffn is None:
                assert ffn_out is not None
                base_output = self._ffn_matmul(hidden, ffn_out)
            else:
                base_output = packed_ffn.project_out(
                    hidden,
                    layer=layer,
                    branch=branch,
                )
            ffn_output = base_output + residual_addition
        elif self.config.deterministic_ternary_ffn_experts:
            if token_ids is None:
                raise ValueError("deterministic FFN experts require token ids")
            if (
                self.ffn_expert_in is None
                or self.ffn_expert_out is None
                or self.ffn_expert_in_bias is None
            ):
                raise RuntimeError("deterministic FFN expert bank is missing")
            extra_in = (
                self.ffn_expert_in[:, :, condition_slice, :]
                if layer is None
                else self.ffn_expert_in[:, layer, condition_slice, :]
            )
            extra_out = (
                self.ffn_expert_out[:, :, :, target_slice]
                if layer is None
                else self.ffn_expert_out[:, layer, :, target_slice]
            )
            extra_bias = (
                self.ffn_expert_in_bias[:, :, 0]
                if layer is None
                else self.ffn_expert_in_bias[:, layer, 0]
            )
            if layer is None:
                extra_projection = torch.stack(
                    tuple(
                        torch.matmul(normalized, extra_in[expert])
                        for expert in range(extra_in.shape[0])
                    ),
                    dim=0,
                )
                extra_hidden = self._ffn_hidden(
                    extra_projection + extra_bias.unsqueeze(2)
                )
                extra_output = torch.stack(
                    tuple(
                        torch.matmul(extra_hidden[expert], extra_out[expert])
                        for expert in range(extra_out.shape[0])
                    ),
                    dim=0,
                )
            else:
                extra_projection = torch.stack(
                    tuple(
                        torch.matmul(normalized, extra_in[expert])
                        for expert in range(extra_in.shape[0])
                    ),
                    dim=0,
                )
                extra_hidden = self._ffn_hidden(
                    extra_projection + extra_bias[:, None, None, :]
                )
                extra_output = torch.stack(
                    tuple(
                        torch.matmul(extra_hidden[expert], extra_out[expert])
                        for expert in range(extra_out.shape[0])
                    ),
                    dim=0,
                )
            if packed_ffn is None:
                assert ffn_out is not None
                base_output = self._ffn_matmul(hidden, ffn_out)
            else:
                base_output = packed_ffn.project_out(
                    hidden,
                    layer=layer,
                    branch=branch,
                )
            from pfc.phase625_deterministic_ternary_experts import (
                select_routed_expert,
            )

            ffn_output = select_routed_expert(
                torch.cat((base_output.unsqueeze(0), extra_output), dim=0),
                token_ids,
            )
        elif self.config.sign_lifted_dual_ternary_transition:
            ffn_output = self._project_sign_lifted_dual_ffn_out(
                hidden,
                layer=layer,
                target_slice=target_slice,
                packed_ffn=packed_ffn,
                branch=branch,
            )
        elif self.config.ternary_dale_quadrant_transition:
            ffn_output = self._project_ternary_dale_ffn_out(
                hidden,
                layer=layer,
                target_slice=target_slice,
                packed_ffn=packed_ffn,
                branch=branch,
            )
        elif self.config.signed_shunting_ternary_transition:
            if packed_ffn is None:
                assert ffn_out is not None
                from pfc.phase701_signed_shunting_transition import (
                    signed_shunting_projection,
                )

                ffn_output = signed_shunting_projection(hidden, ffn_out).voltage
            else:
                signed_current = packed_ffn.project_out(
                    hidden,
                    layer=layer,
                    branch=branch,
                )
                conductance = 1.0 + packed_ffn.project_out_abs(
                    hidden.abs(),
                    layer=layer,
                    branch=branch,
                )
                ffn_output = signed_current / conductance
        elif self.config.fixed_identity_lexical_coordinate_transition_core:
            from pfc.phase728_fixed_identity_lexical_core import (
                fixed_identity_transition_output,
            )

            tail_carrier = self.ffn_out if layer is None else self.ffn_out[self.transition_owner_index(layer)]
            ffn_output = fixed_identity_transition_output(
                hidden,
                self.token_predictor,
                tail_carrier,
                target_slice,
            )
        elif (
            self.config.lexical_coordinate_transition_carrier
            or self.config.post_coinitialized_lexical_coordinate_transition_carrier
        ):
            if self.lexical_transition_participation is None:
                raise RuntimeError("lexical transition participation is missing")
            from pfc.phase726_lexical_coordinate_transition import (
                lexical_coordinate_transition_output,
            )

            participation = (
                self.lexical_transition_participation
                if layer is None
                else self.lexical_transition_participation[layer]
            )
            tail_carrier = self.ffn_out if layer is None else self.ffn_out[self.transition_owner_index(layer)]
            ffn_output = lexical_coordinate_transition_output(
                hidden,
                participation,
                self.token_predictor,
                tail_carrier,
                target_slice,
            )
        elif packed_ffn is None:
            assert ffn_out is not None
            ffn_output = self._ffn_matmul(hidden, ffn_out)
        else:
            ffn_output = packed_ffn.project_out(
                hidden,
                layer=layer,
                branch=branch,
                walsh_token_ids=token_ids if fused_context_bound_decode else None,
            )
        if (
            self.config.context_bound_ternary_parameter_superposition
            and not fused_context_bound_decode
        ):
            assert token_ids is not None
            from pfc.phase736_context_bound_ternary_superposition import (
                bind_walsh_context,
            )

            ffn_output = bind_walsh_context(
                ffn_output,
                token_ids,
                coordinate_offset=target_slice.start or 0,
            )
        candidate_pre = temporal_primary + ffn_output + ffn_out_bias
        if consequence_hidden is not None:
            assert self.consequence_ffn_out is not None
            assert self.consequence_ffn_out_bias is not None
            consequence_out = (
                self.consequence_ffn_out[:, :, target_slice]
                if layer is None
                else self.consequence_ffn_out[layer, :, target_slice]
            )
            consequence_out_bias = (
                self.consequence_ffn_out_bias[:, :, target_slice]
                if layer is None
                else self.consequence_ffn_out_bias[layer, 0, target_slice]
            )
            candidate_pre = candidate_pre + torch.matmul(
                consequence_hidden,
                consequence_out,
            ) + consequence_out_bias
        if candidate_residual is not None:
            if candidate_residual.shape != condition.shape:
                raise ValueError(
                    "reversible coupling candidate residual shape differs"
                )
            candidate_pre = candidate_pre + candidate_residual
        if bounded_attention_read is not None:
            if bounded_attention_read.shape != condition.shape:
                raise ValueError("reversible coupling attention read shape differs")
            candidate_pre = candidate_pre + bounded_attention_read
        candidate_tanh = torch.tanh(candidate_pre)
        temporal_modulation_sigmoid = torch.sigmoid(temporal_modulation)
        if self.config.contractive_reversible_retention:
            retention_pre = gate_primary + gate_bias + gate_modulation
            if self.config.pair_norm_bounded_convex_recurrence:
                retention_pair = torch.sigmoid(
                    retention_pre.reshape(*retention_pre.shape[:-1], -1, 2).mean(
                        dim=-1,
                        keepdim=True,
                    )
                )
                retention = retention_pair.expand(
                    *retention_pair.shape[:-1],
                    2,
                ).reshape_as(retention_pre)
            else:
                retention = torch.sigmoid(retention_pre)
            retention = self._cap_representable_retention(retention)
        else:
            retention = torch.sigmoid(gate_primary + gate_bias) * (
                0.5 + torch.sigmoid(gate_modulation)
            )
        primary_write_sigmoid = torch.sigmoid(write_primary + write_bias)
        if associative_read is not None:
            if associative_read.shape != condition.shape:
                raise ValueError("reversible coupling memory read shape differs")
        if exact_prefill_handoff:
            from pfc.phase444_exact_whole_prefill_workspace import (
                reversible_write_handoff_explicit_round,
            )

            write = reversible_write_handoff_explicit_round(
                candidate_tanh.contiguous(),
                temporal_modulation_sigmoid.contiguous(),
                primary_write_sigmoid.contiguous(),
                torch.sigmoid(memory_primary + write_bias).contiguous()
                if associative_read is not None
                else None,
                associative_read,
            )
        else:
            candidate = candidate_tanh * (0.5 + temporal_modulation_sigmoid)
            write = primary_write_sigmoid * candidate
            if associative_read is not None:
                write = write + torch.sigmoid(memory_primary + write_bias) * (
                    associative_read
                )
        if return_hidden:
            return retention, write, hidden
        return retention, write

    def _cap_representable_retention(self, retention: Tensor) -> Tensor:
        if not self.config.representably_contractive_retention:
            return retention
        one = retention.new_full((), 1.0)
        upper = torch.nextafter(one, retention.new_full((), 0.0))
        return torch.minimum(retention, upper)

    def _bounded_recurrence_proposal_coefficient(
        self,
        retention: Tensor,
    ) -> Tensor:
        if self.config.quadrature_preserving_recurrent_state:
            return torch.sqrt(torch.clamp_min(1.0 - retention.square(), 0.0))
        return 1.0 - retention

    def _effective_learned_recurrent_write(self, write: Tensor) -> Tensor:
        if not self.config.mean_free_learned_recurrent_write:
            return write
        centered = write - write.mean(dim=-1, keepdim=True)
        return centered - centered.mean(dim=-1, keepdim=True)

    def _butterfly_signs(self, value: Tensor, layer: int | None) -> Tensor:
        signs = self.butterfly_layer_signs
        if signs is None:
            raise RuntimeError("layer-rotating butterfly signs are missing")
        if layer is None:
            if value.shape[0] != self.config.depth:
                raise ValueError("layer-parallel butterfly depth differs")
            return signs[:, None, :]
        if not 0 <= layer < self.config.depth:
            raise ValueError("butterfly layer is out of range")
        return signs[layer]

    def _butterfly_to_basis(self, value: Tensor, layer: int | None) -> Tensor:
        if self.config.layer_rotating_bent_butterfly_diffusion:
            value = value * self._butterfly_signs(value, layer)
        return _normalized_hadamard(value)

    def _butterfly_from_basis(self, value: Tensor, layer: int | None) -> Tensor:
        value = _normalized_hadamard(value)
        if self.config.layer_rotating_bent_butterfly_diffusion:
            value = value * self._butterfly_signs(value, layer)
        return value

    def _ordered_rglru_step(
        self,
        proposal: Tensor,
        previous: Tensor,
        *,
        layer: int | None,
        branch: int,
    ) -> Tensor:
        if not self.config.ordered_swiglu_rglru_carry:
            raise RuntimeError("ordered RG-LRU carry is disabled")
        if branch not in (0, 1):
            raise ValueError("ordered RG-LRU branch must be zero or one")
        if any(
            parameter is None
            for parameter in (
                self.ordered_rglru_recurrent_gate_weight,
                self.ordered_rglru_input_gate_weight,
                self.ordered_rglru_recurrent_gate_bias,
                self.ordered_rglru_input_gate_bias,
                self.ordered_rglru_recurrent_parameter,
            )
        ):
            raise RuntimeError("ordered RG-LRU parameters are incomplete")
        assert self.ordered_rglru_recurrent_gate_weight is not None
        assert self.ordered_rglru_input_gate_weight is not None
        assert self.ordered_rglru_recurrent_gate_bias is not None
        assert self.ordered_rglru_input_gate_bias is not None
        assert self.ordered_rglru_recurrent_parameter is not None
        if layer is None:
            if proposal.shape[0] != self.config.depth:
                raise ValueError("ordered RG-LRU depth axis differs")
            return torch.stack(
                tuple(
                    self._ordered_rglru_step(
                        proposal[index],
                        previous[index],
                        layer=index,
                        branch=branch,
                    )
                    for index in range(self.config.depth)
                ),
                dim=0,
            )
        return rglru_step(
            proposal,
            previous,
            self.ordered_rglru_recurrent_gate_weight[layer, branch],
            self.ordered_rglru_recurrent_gate_bias[layer, branch],
            self.ordered_rglru_input_gate_weight[layer, branch],
            self.ordered_rglru_input_gate_bias[layer, branch],
            self.ordered_rglru_recurrent_parameter[layer, branch],
        ).state

    def _ordered_rglru_sequence(
        self,
        proposal: Tensor,
        initial: Tensor,
        *,
        layer: int,
        branch: int,
    ) -> Tensor:
        if not self.config.ordered_swiglu_rglru_carry:
            raise RuntimeError("ordered RG-LRU carry is disabled")
        if any(
            parameter is None
            for parameter in (
                self.ordered_rglru_recurrent_gate_weight,
                self.ordered_rglru_input_gate_weight,
                self.ordered_rglru_recurrent_gate_bias,
                self.ordered_rglru_input_gate_bias,
                self.ordered_rglru_recurrent_parameter,
            )
        ):
            raise RuntimeError("ordered RG-LRU parameters are incomplete")
        assert self.ordered_rglru_recurrent_gate_weight is not None
        assert self.ordered_rglru_input_gate_weight is not None
        assert self.ordered_rglru_recurrent_gate_bias is not None
        assert self.ordered_rglru_input_gate_bias is not None
        assert self.ordered_rglru_recurrent_parameter is not None
        parameters = (
            self.ordered_rglru_recurrent_gate_weight[layer, branch],
            self.ordered_rglru_recurrent_gate_bias[layer, branch],
            self.ordered_rglru_input_gate_weight[layer, branch],
            self.ordered_rglru_input_gate_bias[layer, branch],
            self.ordered_rglru_recurrent_parameter[layer, branch],
        )
        if proposal.is_cuda and proposal.dtype == torch.float32:
            from pfc.phase714_triton_rglru import rglru_sequence_cuda

            states, _ = rglru_sequence_cuda(
                proposal.contiguous(),
                initial.contiguous(),
                *parameters,
            )
        else:
            states, _ = rglru_sequence_reference(proposal, initial, *parameters)
        return states

    def _reversible_coupling_layers(
        self,
        lower: Tensor,
        previous: Tensor,
        observation: Tensor,
        associative_read: Tensor | None,
        bounded_attention_read: Tensor | None = None,
        *,
        token_ids: Tensor | None = None,
        temporal_age: int | Tensor = 1,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Decode one token through layer-parallel reversible coupling blocks."""

        source = lower
        if self.config.residual_stream:
            if self.residual_stream_scales is None:
                residual_scale = self._layer_schedule(
                    self.config.residual_stream_scale_start,
                    self.config.residual_stream_scale_end,
                    device=lower.device,
                    dtype=lower.dtype,
                )
            else:
                residual_scale = self.residual_stream_scales
            sensory_scale = self._layer_schedule(
                self.config.sensory_reinjection_start,
                self.config.sensory_reinjection_end,
                device=lower.device,
                dtype=lower.dtype,
            )
            source = residual_scale * lower + sensory_scale * observation.unsqueeze(0)
        stable_residual = None
        if self.config.stable_latent_expert_branch:
            if self.state_norm.shape[-1] != self.config.width:
                raise RuntimeError(
                    "stable latent experts require full-width reversible norm"
                )
            normalized_source = _rms_norm(
                source[0],
                self.state_norm[0, 0],
                self.config.rms_eps,
            )
            stable_residual, _, _ = self._stable_latent_expert_residual(
                normalized_source
            )
            stable_residual = torch.cat(
                (
                    stable_residual.unsqueeze(0),
                    torch.zeros_like(source[1:]),
                ),
                dim=0,
            )
        scaled_read = (
            self._scale_associative_reads(associative_read)
            if associative_read is not None
            else None
        )
        butterfly = self.config.butterfly_channel_diffusion
        if butterfly:
            source = self._butterfly_to_basis(source, None)
            previous = self._butterfly_to_basis(previous, None)
            if scaled_read is not None:
                scaled_read = self._butterfly_to_basis(scaled_read, None)
        half = self.config.width // 2
        first, second = source.split(half, dim=-1)
        carried_previous = (
            previous
            if (
                self.config.ordered_swiglu_rglru_carry
                or self.config.hippo_legs_order2_temporal_coordinates
            )
            else self._rotate_temporal_states(previous)
        )
        previous_first, previous_second = carried_previous.split(half, dim=-1)
        first_read = scaled_read[..., :half] if scaled_read is not None else None
        second_read = scaled_read[..., half:] if scaled_read is not None else None
        first_attention_read = (
            bounded_attention_read[..., :half]
            if bounded_attention_read is not None
            else None
        )
        second_attention_read = (
            bounded_attention_read[..., half:]
            if bounded_attention_read is not None
            else None
        )

        if self.config.alternating_triangular_coupling:
            even = (
                torch.arange(self.config.depth, device=source.device) % 2 == 0
            ).view(self.config.depth, 1, 1)
            condition = torch.where(even, second, first)
            target = torch.where(even, first, second)
            previous_target = torch.where(
                even,
                previous_first,
                previous_second,
            )
            if scaled_read is None:
                target_read = None
            else:
                assert first_read is not None and second_read is not None
                target_read = torch.where(even, first_read, second_read)

            compact = self.config.compact_active_triangular_coupling
            norm = (
                self.state_norm
                if compact
                else torch.where(
                    even,
                    self.state_norm[:, :, half:],
                    self.state_norm[:, :, :half],
                )
            )
            temporal = (
                self.temporal
                if compact
                else torch.where(
                    even,
                    self.temporal[:, half:, :],
                    self.temporal[:, :half, :],
                )
            )
            gate = (
                self.gate
                if compact
                else torch.where(
                    even,
                    self.gate[:, half:, :],
                    self.gate[:, :half, :],
                )
            )
            if self.transition_write_gate is None:
                raise RuntimeError("reversible coupling write gate is missing")
            write_gate = (
                self.transition_write_gate
                if compact
                else torch.where(
                    even,
                    self.transition_write_gate[:, half:, :],
                    self.transition_write_gate[:, :half, :],
                )
            )
            ffn_in = (
                self.ffn_in
                if compact
                else torch.where(
                    even,
                    self.ffn_in[:, half:, :],
                    self.ffn_in[:, :half, :],
                )
            )
            ffn_out = None
            if not (
                self.config.lexical_coordinate_transition_carrier
                or self.config.post_coinitialized_lexical_coordinate_transition_carrier
                or self.config.fixed_identity_lexical_coordinate_transition_core
            ):
                ffn_out = (
                    self.ffn_out
                    if compact
                    else torch.where(
                        even,
                        self.ffn_out[:, :, :half],
                        self.ffn_out[:, :, half:],
                    )
                )
            gate_bias = (
                self.gate_bias
                if compact
                else torch.where(
                    even,
                    self.gate_bias[:, :, :half],
                    self.gate_bias[:, :, half:],
                )
            )
            if self.transition_write_gate_bias is None:
                raise RuntimeError("reversible coupling write-gate bias is missing")
            write_bias = (
                self.transition_write_gate_bias
                if compact
                else torch.where(
                    even,
                    self.transition_write_gate_bias[:, :, :half],
                    self.transition_write_gate_bias[:, :, half:],
                )
            )
            ffn_out_bias = (
                self.ffn_out_bias
                if compact
                else torch.where(
                    even,
                    self.ffn_out_bias[:, :, :half],
                    self.ffn_out_bias[:, :, half:],
                )
            )

            normalized = _rms_norm(condition, norm, self.config.rms_eps)
            temporal_primary, temporal_modulation = torch.bmm(
                normalized,
                temporal,
            ).chunk(2, dim=-1)
            gate_primary, gate_modulation = torch.bmm(
                normalized,
                gate,
            ).chunk(2, dim=-1)
            write_primary, memory_primary = torch.bmm(
                normalized,
                write_gate,
            ).chunk(2, dim=-1)
            hidden = self._ffn_hidden(
                self._ffn_matmul(normalized, ffn_in) + self.ffn_in_bias
            )
            if self.config.fixed_identity_lexical_coordinate_transition_core:
                from pfc.phase728_fixed_identity_lexical_core import (
                    fixed_identity_transition_output,
                )

                ffn_output = torch.stack(
                    tuple(
                        fixed_identity_transition_output(
                            hidden[layer],
                            self.token_predictor,
                            self.ffn_out[self.transition_owner_index(layer)],
                            slice(0, half)
                            if layer % 2 == 0
                            else slice(half, self.config.width),
                        )
                        for layer in range(self.config.depth)
                    ),
                    dim=0,
                )
            elif (
                self.config.lexical_coordinate_transition_carrier
                or self.config.post_coinitialized_lexical_coordinate_transition_carrier
            ):
                if self.lexical_transition_participation is None:
                    raise RuntimeError(
                        "lexical transition participation is missing"
                    )
                from pfc.phase726_lexical_coordinate_transition import (
                    lexical_coordinate_transition_output,
                )

                ffn_output = torch.stack(
                    tuple(
                        lexical_coordinate_transition_output(
                            hidden[layer],
                            self.lexical_transition_participation[layer],
                            self.token_predictor,
                            self.ffn_out[self.transition_owner_index(layer)],
                            slice(0, half)
                            if layer % 2 == 0
                            else slice(half, self.config.width),
                        )
                        for layer in range(self.config.depth)
                    ),
                    dim=0,
                )
            else:
                assert ffn_out is not None
                ffn_output = self._ffn_matmul(hidden, ffn_out)
            candidate = torch.tanh(
                temporal_primary
                + ffn_output
                + ffn_out_bias
                + (
                    torch.where(
                        even,
                        stable_residual[..., :half],
                        stable_residual[..., half:],
                    )
                    if stable_residual is not None
                    else 0.0
                )
            ) * (0.5 + torch.sigmoid(temporal_modulation))
            retention = torch.sigmoid(gate_primary + gate_bias + gate_modulation)
            retention = self._cap_representable_retention(retention)
            write = torch.sigmoid(write_primary + write_bias) * candidate
            if target_read is not None:
                write = write + torch.sigmoid(memory_primary + write_bias) * target_read
            target_output = target + retention * previous_target + write
            first_output = torch.where(even, target_output, first)
            second_output = torch.where(even, second, target_output)
            output = torch.cat((first_output, second_output), dim=-1)
            return self._butterfly_from_basis(output, None) if butterfly else output

        first_retention, first_write = self._reversible_coupling_terms(
            second,
            0,
            first_read,
            token_ids=token_ids,
            bounded_attention_read=first_attention_read,
            candidate_residual=(
                stable_residual[..., :half]
                if stable_residual is not None
                else None
            ),
        )
        if self.config.ordered_swiglu_rglru_carry:
            first_write = self._effective_learned_recurrent_write(first_write)
            first_proposal = torch.tanh(first + first_write)
            first_output = self._ordered_rglru_step(
                first_proposal,
                previous_first,
                layer=None,
                branch=0,
            )
            first_recurrent = first_output
            first_memory = first_output
        elif self.config.pair_norm_bounded_convex_recurrence:
            first_write = self._effective_learned_recurrent_write(first_write)
            if self.config.affine_reversible_transition:
                from pfc.phase757_affine_reversible_transition import (
                    affine_injection,
                )

                first_scale = self._affine_reversible_scale(
                    second,
                    branch=0,
                    layer=None,
                ).scale
                first_injection = affine_injection(
                    first,
                    first_write,
                    first_scale,
                )
            else:
                first_injection = first + first_write
            first_proposal = torch.tanh(first_injection)
            if self.config.hippo_legs_order2_temporal_coordinates:
                first_pairs = first_injection.reshape(
                    *first_injection.shape[:-1], half // 2, 2
                )
                first_terms = hippo_legs_order2_terms(first_pairs, temporal_age)
                first_recurrent = (
                    torch.matmul(
                        previous_first.reshape(
                            *previous_first.shape[:-1], half // 2, 2
                        ).unsqueeze(-2),
                        first_terms.matrix,
                    )
                    .squeeze(-2)
                    .reshape_as(previous_first)
                )
                first_memory = hippo_legs_order2_step(
                    first_pairs,
                    previous_first.reshape(
                        *previous_first.shape[:-1], half // 2, 2
                    ),
                    temporal_age,
                )
                first_memory = first_memory.reshape_as(previous_first)
            else:
                first_recurrent = first_retention * previous_first
                first_proposal_coefficient = (
                    self._bounded_recurrence_proposal_coefficient(first_retention)
                )
                first_memory = (
                    first_recurrent
                    + first_proposal_coefficient * first_proposal
                )
            first_output = (
                first_recurrent + first_proposal
                if self.config.decoupled_bounded_memory_emission
                else first_memory
            )
            if self.config.coefficient_normalized_emission:
                first_output = first_output / (1.0 + first_retention)
        else:
            first_recurrent = first_retention * previous_first
            if self.config.affine_reversible_transition:
                from pfc.phase757_affine_reversible_transition import (
                    affine_injection,
                )

                first_scale = self._affine_reversible_scale(
                    second,
                    branch=0,
                    layer=None,
                ).scale
                first_output = first_recurrent + affine_injection(
                    first,
                    first_write,
                    first_scale,
                )
            else:
                first_output = first + first_recurrent + first_write
        second_condition = (
            first
            if self.config.parallel_full_state_reciprocal_proposal
            else first_output
        )
        second_terms = self._reversible_coupling_terms(
            second_condition,
            1,
            second_read,
            token_ids=token_ids,
            bounded_attention_read=second_attention_read,
            candidate_residual=(
                stable_residual[..., half:]
                if stable_residual is not None
                else None
            ),
            return_hidden=self.config.swiglu_wide_lexical_cohead,
        )
        swiglu_lexical_feature = None
        if self.config.swiglu_wide_lexical_cohead:
            second_retention, second_write, all_second_hidden = second_terms
            swiglu_lexical_feature = all_second_hidden[-1]
        else:
            second_retention, second_write = second_terms
        if self.config.ordered_swiglu_rglru_carry:
            second_write = self._effective_learned_recurrent_write(second_write)
            second_proposal = torch.tanh(second + second_write)
            second_output = self._ordered_rglru_step(
                second_proposal,
                previous_second,
                layer=None,
                branch=1,
            )
            second_recurrent = second_output
            second_memory = second_output
        elif self.config.pair_norm_bounded_convex_recurrence:
            second_write = self._effective_learned_recurrent_write(second_write)
            if self.config.affine_reversible_transition:
                from pfc.phase757_affine_reversible_transition import (
                    affine_injection,
                )

                second_scale = self._affine_reversible_scale(
                    second_condition,
                    branch=1,
                    layer=None,
                ).scale
                second_injection = affine_injection(
                    second,
                    second_write,
                    second_scale,
                )
            else:
                second_injection = second + second_write
            second_proposal = torch.tanh(second_injection)
            if self.config.hippo_legs_order2_temporal_coordinates:
                second_pairs = second_injection.reshape(
                    *second_injection.shape[:-1], half // 2, 2
                )
                second_terms = hippo_legs_order2_terms(second_pairs, temporal_age)
                second_recurrent = (
                    torch.matmul(
                        previous_second.reshape(
                            *previous_second.shape[:-1], half // 2, 2
                        ).unsqueeze(-2),
                        second_terms.matrix,
                    )
                    .squeeze(-2)
                    .reshape_as(previous_second)
                )
                second_memory = hippo_legs_order2_step(
                    second_pairs,
                    previous_second.reshape(
                        *previous_second.shape[:-1], half // 2, 2
                    ),
                    temporal_age,
                )
                second_memory = second_memory.reshape_as(previous_second)
            else:
                second_recurrent = second_retention * previous_second
                second_proposal_coefficient = (
                    self._bounded_recurrence_proposal_coefficient(second_retention)
                )
                second_memory = (
                    second_recurrent
                    + second_proposal_coefficient * second_proposal
                )
            second_output = (
                second_recurrent + second_proposal
                if self.config.decoupled_bounded_memory_emission
                else second_memory
            )
            if self.config.coefficient_normalized_emission:
                second_output = second_output / (1.0 + second_retention)
        else:
            second_recurrent = second_retention * previous_second
            if self.config.affine_reversible_transition:
                from pfc.phase757_affine_reversible_transition import (
                    affine_injection,
                )

                second_scale = self._affine_reversible_scale(
                    second_condition,
                    branch=1,
                    layer=None,
                ).scale
                second_output = second_recurrent + affine_injection(
                    second,
                    second_write,
                    second_scale,
                )
            else:
                second_output = second + second_recurrent + second_write
        output = torch.cat((first_output, second_output), dim=-1)
        if self.config.causal_level_innovation_lifting:
            return output, torch.cat((first_recurrent, second_recurrent), dim=-1)
        if swiglu_lexical_feature is not None:
            return output, swiglu_lexical_feature
        if not self.config.decoupled_bounded_memory_emission:
            return output
        return output, torch.cat((first_memory, second_memory), dim=-1)

    def _forward_pair_affine_scan(
        self,
        matrix: Tensor,
        write: Tensor,
        initial: Tensor,
    ) -> Tensor:
        if self.config.explicit_round_forward_pair_scan:
            from pfc.phase440_exact_forward_pair_scan import (
                forward_pair_affine_scan_explicit_round,
            )

            return forward_pair_affine_scan_explicit_round(
                matrix,
                write,
                initial,
            )
        return _pair_affine_scan(matrix, write, initial)

    def _reversible_coupling_sequence(
        self,
        lower: Tensor,
        initial: Tensor,
        layer: int,
        observation: Tensor,
        associative_read: Tensor | None,
        bounded_attention_read: Tensor | None = None,
        packed_input_projection: Tensor | None = None,
        residual_route_plan: CublasResidualRoutePlan | None = None,
        *,
        token_ids: Tensor | None = None,
        temporal_start_count: int = 0,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Prefill one reversible block with two half-width associative scans."""

        source = lower
        if self.config.residual_stream:
            source = (
                self._residual_stream_scale(layer) * lower
                + self._sensory_reinjection_scale(layer) * observation
            )
        stable_residual = None
        if layer == 0 and self.config.stable_latent_expert_branch:
            if self.state_norm.shape[-1] != self.config.width:
                raise RuntimeError(
                    "stable latent experts require full-width reversible norm"
                )
            normalized_source = _rms_norm(
                source,
                self.state_norm[layer, 0],
                self.config.rms_eps,
            )
            stable_residual, _, _ = self._stable_latent_expert_residual(
                normalized_source
            )
        scaled_read = (
            self.associative_read_scales[layer] * associative_read
            if associative_read is not None
            else None
        )
        butterfly = self.config.butterfly_channel_diffusion
        if butterfly:
            source = self._butterfly_to_basis(source, layer)
            initial = self._butterfly_to_basis(initial, layer)
            if scaled_read is not None:
                scaled_read = self._butterfly_to_basis(scaled_read, layer)
        half = self.config.width // 2
        pair_count = half // 2
        first, second = source.split(half, dim=-1)
        first_read = scaled_read[..., :half] if scaled_read is not None else None
        second_read = scaled_read[..., half:] if scaled_read is not None else None
        first_attention_read = (
            bounded_attention_read[..., :half]
            if bounded_attention_read is not None
            else None
        )
        second_attention_read = (
            bounded_attention_read[..., half:]
            if bounded_attention_read is not None
            else None
        )
        rotation = self._temporal_rotation_for_normalized(source, layer)
        if rotation is None:
            raise RuntimeError("reversible coupling temporal rotation is missing")
        first_rotation = rotation[..., :pair_count, :]
        second_rotation = rotation[..., pair_count:, :]
        boundary_matrix_mask = (
            token_ids.eq(self.config.bos_token_id)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            if self.config.bos_boundary_state_reset and token_ids is not None
            else None
        )

        if self.config.alternating_triangular_coupling:
            branch = layer % 2
            if branch == 0:
                condition = second
                target = first
                target_read = first_read
                target_rotation = first_rotation
                target_initial = initial[:, :half]
            else:
                condition = first
                target = second
                target_read = second_read
                target_rotation = second_rotation
                target_initial = initial[:, half:]
            retention, write = self._reversible_coupling_terms(
                condition,
                branch,
                target_read,
                layer,
                candidate_residual=(
                    (
                        stable_residual[..., :half]
                        if branch == 0
                        else stable_residual[..., half:]
                    )
                    if stable_residual is not None
                    else None
                ),
                exact_prefill_handoff=self.config.exact_whole_prefill_workspace,
                packed_input_projection=packed_input_projection,
                residual_route_plan=residual_route_plan,
            )
            matrix = _pair_recurrence_matrix(
                retention.reshape(*retention.shape[:-1], pair_count, 2),
                target_rotation,
            )
            if boundary_matrix_mask is not None:
                matrix = matrix.masked_fill(boundary_matrix_mask, 0)
            target_output = self._forward_pair_affine_scan(
                matrix,
                (target + write).reshape(*target.shape[:-1], pair_count, 2),
                target_initial.reshape(initial.shape[0], pair_count, 2),
            ).reshape_as(target)
            output = (
                torch.cat((target_output, second), dim=-1)
                if branch == 0
                else torch.cat((first, target_output), dim=-1)
            )
            return self._butterfly_from_basis(output, layer) if butterfly else output

        first_retention, first_write = self._reversible_coupling_terms(
            second,
            0,
            first_read,
            layer,
            token_ids=token_ids,
            bounded_attention_read=first_attention_read,
            candidate_residual=(
                stable_residual[..., :half]
                if stable_residual is not None
                else None
            ),
            exact_prefill_handoff=self.config.exact_whole_prefill_workspace,
            packed_input_projection=packed_input_projection,
            residual_route_plan=residual_route_plan,
        )
        first_write = self._effective_learned_recurrent_write(first_write)
        if self.config.affine_reversible_transition:
            from pfc.phase757_affine_reversible_transition import affine_injection

            first_scale = self._affine_reversible_scale(
                second,
                branch=0,
                layer=layer,
            ).scale
            first_scan_write = affine_injection(first, first_write, first_scale)
        else:
            first_scan_write = first + first_write
        if self.config.ordered_swiglu_rglru_carry:
            first_proposal = torch.tanh(first_scan_write)
            first_memory = self._ordered_rglru_sequence(
                first_proposal,
                initial[:, :half],
                layer=layer,
                branch=0,
            )
            first_output = first_memory
            first_recurrent = first_memory
        else:
            if self.config.hippo_legs_order2_temporal_coordinates:
                first_pairs = first_scan_write.reshape(
                    *first_scan_write.shape[:-1], pair_count, 2
                )
                first_ages = (
                    torch.arange(
                        first_pairs.shape[1],
                        device=first_pairs.device,
                        dtype=first_pairs.dtype,
                    )
                    + temporal_start_count
                    + 1
                ).view(1, first_pairs.shape[1], 1)
                first_terms = hippo_legs_order2_terms(first_pairs, first_ages)
                first_matrix = first_terms.matrix
                first_scan_write = first_terms.write
            else:
                first_matrix = _pair_recurrence_matrix(
                    first_retention.reshape(
                        *first_retention.shape[:-1], pair_count, 2
                    ),
                    first_rotation,
                )
            if boundary_matrix_mask is not None:
                first_matrix = first_matrix.masked_fill(boundary_matrix_mask, 0)
            if (
                self.config.pair_norm_bounded_convex_recurrence
                and not self.config.hippo_legs_order2_temporal_coordinates
            ):
                first_proposal = torch.tanh(first_scan_write)
                first_scan_write = (
                    self._bounded_recurrence_proposal_coefficient(first_retention)
                    * first_proposal
                )
            if self.config.hippo_legs_order2_temporal_coordinates:
                first_memory = hippo_legs_order2_sequence(
                    first_pairs,
                    initial[:, :half].reshape(initial.shape[0], pair_count, 2),
                    temporal_start_count,
                ).reshape_as(first)
            else:
                first_memory = self._forward_pair_affine_scan(
                    first_matrix,
                    first_scan_write.reshape(*first.shape[:-1], pair_count, 2),
                    initial[:, :half].reshape(initial.shape[0], pair_count, 2),
                ).reshape_as(first)
            first_output = first_memory
            if self.config.decoupled_bounded_memory_emission:
                first_output = first_memory + first_retention * first_proposal
            if self.config.coefficient_normalized_emission:
                first_output = first_output / (1.0 + first_retention)
            first_previous = torch.cat(
                (initial[:, :half].unsqueeze(1), first_memory[:, :-1]),
                dim=1,
            )
            first_recurrent = (
                torch.matmul(
                    first_previous.reshape(
                        *first_previous.shape[:-1], pair_count, 2
                    ).unsqueeze(-2),
                    first_matrix,
                )
                .squeeze(-2)
                .reshape_as(first_memory)
            )

        second_condition = (
            first
            if self.config.parallel_full_state_reciprocal_proposal
            else first_output
        )
        second_terms = self._reversible_coupling_terms(
            second_condition,
            1,
            second_read,
            layer,
            token_ids=token_ids,
            bounded_attention_read=second_attention_read,
            candidate_residual=(
                stable_residual[..., half:]
                if stable_residual is not None
                else None
            ),
            exact_prefill_handoff=self.config.exact_whole_prefill_workspace,
            packed_input_projection=packed_input_projection,
            residual_route_plan=residual_route_plan,
            return_hidden=(
                self.config.swiglu_wide_lexical_cohead
                and layer == self.config.depth - 1
            ),
        )
        swiglu_lexical_feature = None
        if (
            self.config.swiglu_wide_lexical_cohead
            and layer == self.config.depth - 1
        ):
            second_retention, second_write, swiglu_lexical_feature = second_terms
        else:
            second_retention, second_write = second_terms
        second_write = self._effective_learned_recurrent_write(second_write)
        if self.config.affine_reversible_transition:
            from pfc.phase757_affine_reversible_transition import affine_injection

            second_scale = self._affine_reversible_scale(
                second_condition,
                branch=1,
                layer=layer,
            ).scale
            second_scan_write = affine_injection(
                second,
                second_write,
                second_scale,
            )
        else:
            second_scan_write = second + second_write
        if self.config.ordered_swiglu_rglru_carry:
            second_proposal = torch.tanh(second_scan_write)
            second_memory = self._ordered_rglru_sequence(
                second_proposal,
                initial[:, half:],
                layer=layer,
                branch=1,
            )
            second_output = second_memory
            second_recurrent = second_memory
        else:
            if self.config.hippo_legs_order2_temporal_coordinates:
                second_pairs = second_scan_write.reshape(
                    *second_scan_write.shape[:-1], pair_count, 2
                )
                second_ages = (
                    torch.arange(
                        second_pairs.shape[1],
                        device=second_pairs.device,
                        dtype=second_pairs.dtype,
                    )
                    + temporal_start_count
                    + 1
                ).view(1, second_pairs.shape[1], 1)
                second_terms = hippo_legs_order2_terms(second_pairs, second_ages)
                second_matrix = second_terms.matrix
                second_scan_write = second_terms.write
            else:
                second_matrix = _pair_recurrence_matrix(
                    second_retention.reshape(
                        *second_retention.shape[:-1],
                        pair_count,
                        2,
                    ),
                    second_rotation,
                )
            if boundary_matrix_mask is not None:
                second_matrix = second_matrix.masked_fill(boundary_matrix_mask, 0)
            if (
                self.config.pair_norm_bounded_convex_recurrence
                and not self.config.hippo_legs_order2_temporal_coordinates
            ):
                second_proposal = torch.tanh(second_scan_write)
                second_scan_write = (
                    self._bounded_recurrence_proposal_coefficient(second_retention)
                    * second_proposal
                )
            if self.config.hippo_legs_order2_temporal_coordinates:
                second_memory = hippo_legs_order2_sequence(
                    second_pairs,
                    initial[:, half:].reshape(initial.shape[0], pair_count, 2),
                    temporal_start_count,
                ).reshape_as(second)
            else:
                second_memory = self._forward_pair_affine_scan(
                    second_matrix,
                    second_scan_write.reshape(*second.shape[:-1], pair_count, 2),
                    initial[:, half:].reshape(initial.shape[0], pair_count, 2),
                ).reshape_as(second)
            second_output = second_memory
            if self.config.decoupled_bounded_memory_emission:
                second_output = second_memory + second_retention * second_proposal
            if self.config.coefficient_normalized_emission:
                second_output = second_output / (1.0 + second_retention)
            second_previous = torch.cat(
                (initial[:, half:].unsqueeze(1), second_memory[:, :-1]),
                dim=1,
            )
            second_recurrent = (
                torch.matmul(
                    second_previous.reshape(
                        *second_previous.shape[:-1], pair_count, 2
                    ).unsqueeze(-2),
                    second_matrix,
                )
                .squeeze(-2)
                .reshape_as(second_memory)
            )
        output = torch.cat((first_output, second_output), dim=-1)
        transition_observer = getattr(
            self,
            "_reversible_transition_stage_observer",
            None,
        )
        if transition_observer is not None:
            transition_observer(
                layer=layer,
                source=source.detach(),
                midpoint=torch.cat((first_output, second), dim=-1).detach(),
                final=output.detach(),
            )
        if self.config.causal_level_innovation_lifting:
            return output, torch.cat((first_recurrent, second_recurrent), dim=-1)
        if swiglu_lexical_feature is not None:
            return output, swiglu_lexical_feature
        if not self.config.decoupled_bounded_memory_emission:
            return output
        return output, torch.cat((first_memory, second_memory), dim=-1)

    def reversible_coupling_inverse_sequence(
        self,
        raw: Tensor,
        initial: Tensor,
        layer: int,
        observation: Tensor,
        associative_read: Tensor | None = None,
        bounded_attention_read: Tensor | None = None,
    ) -> Tensor:
        """Recover a coupling block input using forward-oriented local functions."""

        if not self.config.reversible_recurrent_coupling:
            raise RuntimeError("reversible recurrent coupling is disabled")
        if self.config.hippo_legs_order2_temporal_coordinates:
            raise RuntimeError("inverse reconstruction is undefined for HiPPO-LegS")
        if self.config.decoupled_bounded_memory_emission:
            raise RuntimeError(
                "inverse reconstruction is undefined for decoupled emitted state"
            )
        if raw.ndim != 3 or raw.shape[-1] != self.config.width:
            raise ValueError("reversible coupling raw sequence shape differs")
        if initial.shape != (raw.shape[0], self.config.width):
            raise ValueError("reversible coupling initial state shape differs")
        if observation.shape != raw.shape:
            raise ValueError("reversible coupling observation shape differs")
        scaled_read = (
            self.associative_read_scales[layer] * associative_read
            if associative_read is not None
            else None
        )
        butterfly = self.config.butterfly_channel_diffusion
        if butterfly:
            raw = self._butterfly_to_basis(raw, layer)
            initial = self._butterfly_to_basis(initial, layer)
            if scaled_read is not None:
                scaled_read = self._butterfly_to_basis(scaled_read, layer)
        half = self.config.width // 2
        pair_count = half // 2
        first_output, second_output = raw.split(half, dim=-1)
        first_read = scaled_read[..., :half] if scaled_read is not None else None
        second_read = scaled_read[..., half:] if scaled_read is not None else None
        first_attention_read = (
            bounded_attention_read[..., :half]
            if bounded_attention_read is not None
            else None
        )
        second_attention_read = (
            bounded_attention_read[..., half:]
            if bounded_attention_read is not None
            else None
        )
        rotation = self._temporal_rotation_for_normalized(raw, layer)
        if rotation is None:
            raise RuntimeError("reversible coupling temporal rotation is missing")

        if self.config.alternating_triangular_coupling:
            branch = layer % 2
            if branch == 0:
                condition = second_output
                target_output = first_output
                target_read = first_read
                target_initial = initial[:, :half]
                target_rotation = rotation[..., :pair_count, :]
            else:
                condition = first_output
                target_output = second_output
                target_read = second_read
                target_initial = initial[:, half:]
                target_rotation = rotation[..., pair_count:, :]
            previous_target = torch.cat(
                (target_initial.unsqueeze(1), target_output[:, :-1]),
                dim=1,
            )
            retention, write = self._reversible_coupling_terms(
                condition,
                branch,
                target_read,
                layer,
            )
            matrix = _pair_recurrence_matrix(
                retention.reshape(*retention.shape[:-1], pair_count, 2),
                target_rotation,
            )
            recurrent = (
                torch.matmul(
                    previous_target.reshape(
                        *previous_target.shape[:-1],
                        pair_count,
                        2,
                    ).unsqueeze(-2),
                    matrix,
                )
                .squeeze(-2)
                .reshape_as(target_output)
            )
            target = target_output - recurrent - write
            source = (
                torch.cat((target, condition), dim=-1)
                if branch == 0
                else torch.cat((condition, target), dim=-1)
            )
            if butterfly:
                source = self._butterfly_from_basis(source, layer)
            if not self.config.residual_stream:
                return source
            return (
                source - self._sensory_reinjection_scale(layer) * observation
            ) / self._residual_stream_scale(layer)

        previous_second = torch.cat(
            (initial[:, half:].unsqueeze(1), second_output[:, :-1]),
            dim=1,
        )
        second_retention, second_write = self._reversible_coupling_terms(
            first_output,
            1,
            second_read,
            layer,
            bounded_attention_read=second_attention_read,
        )
        second_write = self._effective_learned_recurrent_write(second_write)
        second_matrix = _pair_recurrence_matrix(
            second_retention.reshape(
                *second_retention.shape[:-1],
                pair_count,
                2,
            ),
            rotation[..., pair_count:, :],
        )
        second_recurrent = (
            torch.matmul(
                previous_second.reshape(
                    *previous_second.shape[:-1],
                    pair_count,
                    2,
                ).unsqueeze(-2),
                second_matrix,
            )
            .squeeze(-2)
            .reshape_as(second_output)
        )
        if self.config.pair_norm_bounded_convex_recurrence:
            second_proposal_coefficient = (
                self._bounded_recurrence_proposal_coefficient(second_retention)
            )
            second_proposal = (second_output - second_recurrent) / (
                second_proposal_coefficient
            )
            second_limit = torch.nextafter(
                second_proposal.new_full((), 1.0),
                second_proposal.new_full((), 0.0),
            )
            second_injection = torch.atanh(
                second_proposal.clamp(min=-second_limit, max=second_limit)
            )
            if self.config.affine_reversible_transition:
                from pfc.phase757_affine_reversible_transition import affine_inverse

                second_scale = self._affine_reversible_scale(
                    first_output,
                    branch=1,
                    layer=layer,
                ).scale
                second = affine_inverse(
                    second_injection,
                    second_write,
                    second_scale,
                )
            else:
                second = second_injection - second_write
        elif self.config.affine_reversible_transition:
            from pfc.phase757_affine_reversible_transition import affine_inverse

            second_scale = self._affine_reversible_scale(
                first_output,
                branch=1,
                layer=layer,
            ).scale
            second = affine_inverse(
                second_output - second_recurrent,
                second_write,
                second_scale,
            )
        else:
            second = second_output - second_recurrent - second_write

        previous_first = torch.cat(
            (initial[:, :half].unsqueeze(1), first_output[:, :-1]),
            dim=1,
        )
        first_retention, first_write = self._reversible_coupling_terms(
            second,
            0,
            first_read,
            layer,
            bounded_attention_read=first_attention_read,
        )
        first_write = self._effective_learned_recurrent_write(first_write)
        first_matrix = _pair_recurrence_matrix(
            first_retention.reshape(*first_retention.shape[:-1], pair_count, 2),
            rotation[..., :pair_count, :],
        )
        first_recurrent = (
            torch.matmul(
                previous_first.reshape(
                    *previous_first.shape[:-1],
                    pair_count,
                    2,
                ).unsqueeze(-2),
                first_matrix,
            )
            .squeeze(-2)
            .reshape_as(first_output)
        )
        if self.config.pair_norm_bounded_convex_recurrence:
            first_proposal_coefficient = (
                self._bounded_recurrence_proposal_coefficient(first_retention)
            )
            first_proposal = (first_output - first_recurrent) / (
                first_proposal_coefficient
            )
            first_limit = torch.nextafter(
                first_proposal.new_full((), 1.0),
                first_proposal.new_full((), 0.0),
            )
            first_injection = torch.atanh(
                first_proposal.clamp(min=-first_limit, max=first_limit)
            )
            if self.config.affine_reversible_transition:
                from pfc.phase757_affine_reversible_transition import affine_inverse

                first_scale = self._affine_reversible_scale(
                    second,
                    branch=0,
                    layer=layer,
                ).scale
                first = affine_inverse(
                    first_injection,
                    first_write,
                    first_scale,
                )
            else:
                first = first_injection - first_write
        elif self.config.affine_reversible_transition:
            from pfc.phase757_affine_reversible_transition import affine_inverse

            first_scale = self._affine_reversible_scale(
                second,
                branch=0,
                layer=layer,
            ).scale
            first = affine_inverse(
                first_output - first_recurrent,
                first_write,
                first_scale,
            )
        else:
            first = first_output - first_recurrent - first_write
        source = torch.cat((first, second), dim=-1)
        if not self.config.residual_stream:
            return source
        return (
            source - self._sensory_reinjection_scale(layer) * observation
        ) / self._residual_stream_scale(layer)

    def _transition_layers(
        self,
        lower: Tensor,
        previous: Tensor,
        observation: Tensor,
        associative_read: Tensor | None = None,
        persistent_adapter_initial: Tensor | None = None,
        gdn2_read: Tensor | None = None,
        factorization_read: Tensor | None = None,
        bounded_attention_read: Tensor | None = None,
        *,
        token_ids: Tensor | None = None,
        temporal_age: int | Tensor = 1,
        feature_context_initial: Tensor | None = None,
        return_persistent_adapter_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, Tensor]:
        if self.config.ternary_tpc_rtrl_elementwise_transition:
            if any(
                value is not None
                for value in (
                    persistent_adapter_initial,
                    gdn2_read,
                    factorization_read,
                    bounded_attention_read,
                    feature_context_initial,
                )
            ):
                raise ValueError("Phase714 RG-LRU does not accept transition side state")
            assert self.rglru_recurrent_gate_weight is not None
            assert self.rglru_input_gate_weight is not None
            assert self.rglru_recurrent_gate_bias is not None
            assert self.rglru_input_gate_bias is not None
            assert self.rglru_recurrent_parameter is not None
            states = []
            for layer in range(self.config.depth):
                source = lower[layer]
                if self.config.residual_stream:
                    source = (
                        self._residual_stream_scale(layer) * source
                        + self._sensory_reinjection_scale(layer) * observation
                    )
                normalized = _rms_norm(
                    source,
                    self.state_norm[layer, 0],
                    self.config.rms_eps,
                )
                states.append(
                    rglru_step(
                        normalized,
                        previous[layer],
                        self.rglru_recurrent_gate_weight[layer],
                        self.rglru_recurrent_gate_bias[layer],
                        self.rglru_input_gate_weight[layer],
                        self.rglru_input_gate_bias[layer],
                        self.rglru_recurrent_parameter[layer],
                    ).state
                )
            raw = torch.stack(states, dim=0)
            return (raw, None) if return_persistent_adapter_state else raw
        if self.config.reversible_recurrent_coupling:
            if any(
                value is not None
                for value in (
                    persistent_adapter_initial,
                    gdn2_read,
                    factorization_read,
                    feature_context_initial,
                )
            ):
                raise ValueError("unsupported reversible coupling side state")
            reversible_result = self._reversible_coupling_layers(
                lower,
                previous,
                observation,
                associative_read,
                bounded_attention_read,
                token_ids=token_ids,
                temporal_age=temporal_age,
            )
            if self.config.causal_level_innovation_lifting:
                if not isinstance(reversible_result, tuple):
                    raise RuntimeError("causal lifting recurrence result is missing")
                recurrent_raw, recurrent_prediction = reversible_result
                emitted = causal_level_innovation_lift(
                    recurrent_raw,
                    recurrent_prediction,
                )
                return (
                    (emitted, None, recurrent_raw)
                    if return_persistent_adapter_state
                    else emitted
                )
            if self.config.swiglu_wide_lexical_cohead:
                if not isinstance(reversible_result, tuple):
                    raise RuntimeError("SwiGLU lexical feature result is missing")
                raw, swiglu_lexical_feature = reversible_result
                return (
                    (raw, None, swiglu_lexical_feature)
                    if return_persistent_adapter_state
                    else reversible_result
                )
            if self.config.decoupled_bounded_memory_emission:
                if not isinstance(reversible_result, tuple):
                    raise RuntimeError("decoupled recurrent memory result is missing")
                raw, recurrent_memory = reversible_result
                return (
                    (raw, None, recurrent_memory)
                    if return_persistent_adapter_state
                    else raw
                )
            if isinstance(reversible_result, tuple):
                raise RuntimeError("unexpected decoupled recurrent memory result")
            return (
                (reversible_result, None)
                if return_persistent_adapter_state
                else reversible_result
            )
        source = lower
        if self.config.residual_stream:
            if self.residual_stream_scales is None:
                residual_scale = self._layer_schedule(
                    self.config.residual_stream_scale_start,
                    self.config.residual_stream_scale_end,
                    device=lower.device,
                    dtype=lower.dtype,
                )
            else:
                residual_scale = self.residual_stream_scales
            sensory_scale = self._layer_schedule(
                self.config.sensory_reinjection_start,
                self.config.sensory_reinjection_end,
                device=lower.device,
                dtype=lower.dtype,
            )
            source = residual_scale * lower + sensory_scale * observation.unsqueeze(0)
        normalized = _rms_norm(
            source,
            self.state_norm,
            self.config.rms_eps,
        )
        temporal_rotation = self._temporal_rotation_for_normalized(normalized)
        decay = torch.sigmoid(torch.bmm(normalized, self.gate) + self.gate_bias)
        write_gate = (
            torch.sigmoid(
                torch.bmm(normalized, self.transition_write_gate)
                + self.transition_write_gate_bias
            )
            if self.transition_write_gate is not None
            and self.transition_write_gate_bias is not None
            else 1.0 - decay
        )
        delta_factors = None
        if self.pairwise_delta_down is not None:
            delta_factors = self._pairwise_delta_erase_factors(normalized)
            if delta_factors is None:
                raise RuntimeError("delta transition factors are missing")
        feature_context_state = None
        feature_normalized = normalized
        if self._uses_recurrent_feature_context():
            if feature_context_initial is None:
                raise ValueError("bounded recurrent feature context is missing")
            context_gate = write_gate.mean(dim=-1, keepdim=True)
            rotated_context = self._rotate_temporal_states(
                feature_context_initial,
                temporal_rotation,
            )
            feature_context_state = (
                1.0 - context_gate
            ) * rotated_context + context_gate * torch.tanh(normalized)
            if self.config.bounded_recurrent_feature_context:
                feature_normalized = _rms_norm(
                    source + torch.tanh(feature_context_initial),
                    self.state_norm,
                    self.config.rms_eps,
                )
            else:
                feature_normalized = normalized * (
                    1.0 + torch.tanh(feature_context_initial)
                )
        if self.config.attention_conditioned_feature_write:
            if bounded_attention_read is None:
                raise ValueError("attention-conditioned feature read is missing")
            feature_normalized = feature_normalized + bounded_attention_read
        hidden = self._ffn_hidden(
            self._ffn_matmul(feature_normalized, self.ffn_in) + self.ffn_in_bias
        )
        candidate_pre = (
            torch.bmm(feature_normalized, self.temporal)
            + self._ffn_matmul(hidden, self.ffn_out)
            + self.ffn_out_bias
        )
        if self.config.stable_latent_expert_branch:
            stable_residual, _, _ = self._stable_latent_expert_residual(
                feature_normalized[0]
            )
            candidate_pre = torch.cat(
                (
                    candidate_pre[:1] + stable_residual.unsqueeze(0),
                    candidate_pre[1:],
                ),
                dim=0,
            )
        if gdn2_read is not None:
            candidate_pre = candidate_pre + gdn2_read
        if factorization_read is not None:
            candidate_pre = candidate_pre + factorization_read
        if (
            bounded_attention_read is not None
            and not self.config.attention_conditioned_feature_write
        ):
            candidate_pre = candidate_pre + bounded_attention_read
        persistent_adapter_state = None
        if self.persistent_transition_adapter_down is not None:
            if persistent_adapter_initial is None:
                persistent_adapter_initial = torch.zeros(
                    self.config.depth,
                    lower.shape[1],
                    self.config.persistent_transition_adapter_rank,
                    device=lower.device,
                    dtype=lower.dtype,
                )
            persistent_adapter_state, persistent_residual = (
                self._persistent_transition_adapter_step(
                    normalized,
                    persistent_adapter_initial,
                )
            )
            candidate_pre = candidate_pre + persistent_residual
        transformed = torch.tanh(candidate_pre)
        candidate = source + transformed if self.config.residual_stream else transformed
        if associative_read is not None:
            candidate = candidate + self._scale_associative_reads(associative_read)
        write = write_gate * candidate
        if self.pairwise_delta_down is not None:
            if temporal_rotation is None:
                raise RuntimeError("pairwise delta erase requires temporal rotation")
            group_width = self.config.pairwise_delta_group_width
            groups = self.config.width // group_width
            grouped_previous = previous.reshape(
                *previous.shape[:-1], groups, group_width
            )
            assert delta_factors is not None
            _, _, key, _, erase_vector, strength = delta_factors
            base = decay * self._rotate_temporal_states(
                previous,
                temporal_rotation,
            )
            memory_read = (grouped_previous * erase_vector).sum(dim=-1, keepdim=True)
            correction = (strength.squeeze(-1) * memory_read * key).reshape_as(previous)
            raw = base - correction + write
        else:
            recurrent_previous = self._rotate_temporal_states(
                previous,
                temporal_rotation,
            )
            raw = decay * recurrent_previous + write
        if return_persistent_adapter_state:
            if feature_context_state is not None:
                return raw, persistent_adapter_state, feature_context_state
            return raw, persistent_adapter_state
        return raw

    def _transition_sequence(
        self,
        lower: Tensor,
        initial: Tensor,
        layer: int,
        observation: Tensor,
        associative_read: Tensor | None = None,
        persistent_adapter_initial: Tensor | None = None,
        gdn2_read: Tensor | None = None,
        factorization_read: Tensor | None = None,
        bounded_attention_read: Tensor | None = None,
        *,
        token_ids: Tensor | None = None,
        temporal_start_count: int = 0,
        feature_context_initial: Tensor | None = None,
        return_persistent_adapter_state: bool = False,
        packed_input_projection: Tensor | None = None,
        residual_route_plan: CublasResidualRoutePlan | None = None,
    ) -> Tensor | tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, Tensor]:
        if self.config.ternary_tpc_rtrl_elementwise_transition:
            if any(
                value is not None
                for value in (
                    persistent_adapter_initial,
                    gdn2_read,
                    factorization_read,
                    bounded_attention_read,
                    feature_context_initial,
                )
            ):
                raise ValueError("Phase714 RG-LRU does not accept transition side state")
            assert self.rglru_recurrent_gate_weight is not None
            assert self.rglru_input_gate_weight is not None
            assert self.rglru_recurrent_gate_bias is not None
            assert self.rglru_input_gate_bias is not None
            assert self.rglru_recurrent_parameter is not None
            source = lower
            if self.config.residual_stream:
                source = (
                    self._residual_stream_scale(layer) * lower
                    + self._sensory_reinjection_scale(layer) * observation
                )
            normalized = _rms_norm(
                source,
                self.state_norm[layer, 0],
                self.config.rms_eps,
            )
            if normalized.is_cuda and normalized.dtype == torch.float32:
                from pfc.phase714_triton_rglru import rglru_sequence_cuda

                states, _ = rglru_sequence_cuda(
                    normalized.contiguous(),
                    initial.contiguous(),
                    self.rglru_recurrent_gate_weight[layer],
                    self.rglru_recurrent_gate_bias[layer],
                    self.rglru_input_gate_weight[layer],
                    self.rglru_input_gate_bias[layer],
                    self.rglru_recurrent_parameter[layer],
                )
            else:
                states, _ = rglru_sequence_reference(
                    normalized,
                    initial,
                    self.rglru_recurrent_gate_weight[layer],
                    self.rglru_recurrent_gate_bias[layer],
                    self.rglru_input_gate_weight[layer],
                    self.rglru_input_gate_bias[layer],
                    self.rglru_recurrent_parameter[layer],
                )
            return (states, None) if return_persistent_adapter_state else states
        if self.config.reversible_recurrent_coupling:
            if any(
                value is not None
                for value in (
                    persistent_adapter_initial,
                    gdn2_read,
                    factorization_read,
                    feature_context_initial,
                )
            ):
                raise ValueError("unsupported reversible coupling side state")
            reversible_result = self._reversible_coupling_sequence(
                lower,
                initial,
                layer,
                observation,
                associative_read,
                bounded_attention_read,
                packed_input_projection,
                token_ids=token_ids,
                temporal_start_count=temporal_start_count,
                residual_route_plan=residual_route_plan,
            )
            if self.config.causal_level_innovation_lifting:
                if not isinstance(reversible_result, tuple):
                    raise RuntimeError("causal lifting recurrence sequence is missing")
                recurrent_raw, recurrent_prediction = reversible_result
                emitted = causal_level_innovation_lift(
                    recurrent_raw,
                    recurrent_prediction,
                )
                return (
                    (emitted, None, recurrent_raw)
                    if return_persistent_adapter_state
                    else emitted
                )
            if (
                self.config.swiglu_wide_lexical_cohead
                and layer == self.config.depth - 1
            ):
                if not isinstance(reversible_result, tuple):
                    raise RuntimeError("SwiGLU lexical feature sequence is missing")
                raw, swiglu_lexical_feature = reversible_result
                return (
                    (raw, None, swiglu_lexical_feature)
                    if return_persistent_adapter_state
                    else reversible_result
                )
            if self.config.decoupled_bounded_memory_emission:
                if not isinstance(reversible_result, tuple):
                    raise RuntimeError("decoupled recurrent memory result is missing")
                raw, recurrent_memory = reversible_result
                return (
                    (raw, None, recurrent_memory)
                    if return_persistent_adapter_state
                    else raw
                )
            if isinstance(reversible_result, tuple):
                raise RuntimeError("unexpected decoupled recurrent memory result")
            return (
                (reversible_result, None)
                if return_persistent_adapter_state
                else reversible_result
            )
        source = lower
        if self.config.residual_stream:
            source = (
                self._residual_stream_scale(layer) * lower
                + self._sensory_reinjection_scale(layer) * observation
            )
        normalized = _rms_norm(
            source,
            self.state_norm[layer, 0],
            self.config.rms_eps,
        )
        temporal_rotation = self._temporal_rotation_for_normalized(
            normalized,
            layer,
        )
        decay = torch.sigmoid(normalized @ self.gate[layer] + self.gate_bias[layer, 0])
        boundary_decay_mask = (
            token_ids.eq(self.config.bos_token_id).unsqueeze(-1)
            if self.config.bos_boundary_state_reset and token_ids is not None
            else None
        )
        write_gate = (
            torch.sigmoid(
                normalized @ self.transition_write_gate[layer]
                + self.transition_write_gate_bias[layer, 0]
            )
            if self.transition_write_gate is not None
            and self.transition_write_gate_bias is not None
            else 1.0 - decay
        )
        transition_matrix = None
        if temporal_rotation is not None:
            batch, sequence, width = decay.shape
            if (
                self.pairwise_delta_down is not None
                and self.config.pairwise_delta_group_width == 4
            ):
                groups = width // 4
                transition_matrix = self._group4_transition_matrix(
                    normalized,
                    decay.reshape(batch, sequence, groups, 4),
                    temporal_rotation.reshape(batch, sequence, groups, 2, 2),
                    layer,
                )
            else:
                transition_matrix = self._pair_transition_matrix(
                    normalized,
                    decay.reshape(batch, sequence, width // 2, 2),
                    temporal_rotation,
                    layer,
                )
        feature_context_states = None
        feature_normalized = normalized
        if self._uses_recurrent_feature_context():
            if feature_context_initial is None:
                raise ValueError("bounded recurrent feature context is missing")
            context_gate = write_gate.mean(dim=-1, keepdim=True)
            context_decay = (1.0 - context_gate).expand_as(normalized)
            context_write = context_gate * torch.tanh(normalized)
            if temporal_rotation is None:
                if boundary_decay_mask is not None:
                    context_decay = context_decay.masked_fill(
                        boundary_decay_mask,
                        0,
                    )
                feature_context_states = _affine_scan(
                    context_decay,
                    context_write,
                    feature_context_initial,
                )
            else:
                pairs = width // 2
                context_matrix = _pair_recurrence_matrix(
                    context_decay.reshape(batch, sequence, pairs, 2),
                    temporal_rotation,
                )
                if boundary_decay_mask is not None:
                    context_matrix = context_matrix.masked_fill(
                        boundary_decay_mask.unsqueeze(-1).unsqueeze(-1),
                        0,
                    )
                feature_context_states = _pair_affine_scan(
                    context_matrix,
                    context_write.reshape(batch, sequence, pairs, 2),
                    feature_context_initial.reshape(batch, pairs, 2),
                ).reshape(batch, sequence, width)
            previous_context = torch.cat(
                (
                    feature_context_initial.unsqueeze(1),
                    feature_context_states[:, :-1],
                ),
                dim=1,
            )
            if self.config.bounded_recurrent_feature_context:
                feature_normalized = _rms_norm(
                    source + torch.tanh(previous_context),
                    self.state_norm[layer, 0],
                    self.config.rms_eps,
                )
            else:
                feature_normalized = normalized * (1.0 + torch.tanh(previous_context))
        if (
            self.config.attention_conditioned_feature_write
            and layer >= self._bounded_attention_start_layer()
        ):
            if bounded_attention_read is None:
                raise ValueError("attention-conditioned feature read is missing")
            feature_normalized = feature_normalized + bounded_attention_read
        hidden = self._ffn_hidden(
            self._ffn_matmul(feature_normalized, self.ffn_in[self.transition_owner_index(layer)])
            + self.ffn_in_bias[layer, 0]
        )
        candidate_pre = (
            feature_normalized @ self.temporal[layer]
            + self._ffn_matmul(hidden, self.ffn_out[self.transition_owner_index(layer)])
            + self.ffn_out_bias[layer, 0]
        )
        if layer == 0 and self.config.stable_latent_expert_branch:
            stable_residual, _, _ = self._stable_latent_expert_residual(
                feature_normalized
            )
            candidate_pre = candidate_pre + stable_residual
        if gdn2_read is not None:
            candidate_pre = candidate_pre + gdn2_read
        if factorization_read is not None:
            candidate_pre = candidate_pre + factorization_read
        if (
            bounded_attention_read is not None
            and not self.config.attention_conditioned_feature_write
        ):
            candidate_pre = candidate_pre + bounded_attention_read
        persistent_adapter_states = None
        if self.persistent_transition_adapter_down is not None:
            if persistent_adapter_initial is None:
                persistent_adapter_initial = torch.zeros(
                    lower.shape[0],
                    self.config.persistent_transition_adapter_rank,
                    device=lower.device,
                    dtype=lower.dtype,
                )
            persistent_adapter_states, persistent_residual = (
                self._persistent_transition_adapter_sequence(
                    normalized,
                    persistent_adapter_initial,
                    layer,
                )
            )
            candidate_pre = candidate_pre + persistent_residual
        transformed = torch.tanh(candidate_pre)
        candidate = source + transformed if self.config.residual_stream else transformed
        if associative_read is not None:
            candidate = (
                candidate + self.associative_read_scales[layer] * associative_read
            )
        write = write_gate * candidate
        if temporal_rotation is None:
            if boundary_decay_mask is not None:
                decay = decay.masked_fill(boundary_decay_mask, 0)
            raw = _affine_scan(decay, write, initial)
        else:
            if (
                self.pairwise_delta_down is not None
                and self.config.pairwise_delta_group_width == 4
            ):
                assert transition_matrix is not None
                if boundary_decay_mask is not None:
                    transition_matrix = transition_matrix.masked_fill(
                        boundary_decay_mask.unsqueeze(-1).unsqueeze(-1),
                        0,
                    )
                groups = width // 4
                raw = _group4_affine_scan(
                    transition_matrix,
                    write.reshape(batch, sequence, groups, 4),
                    initial.reshape(batch, groups, 4),
                ).reshape(batch, sequence, width)
            else:
                assert transition_matrix is not None
                if boundary_decay_mask is not None:
                    transition_matrix = transition_matrix.masked_fill(
                        boundary_decay_mask.unsqueeze(-1).unsqueeze(-1),
                        0,
                    )
                paired_write = write.reshape(batch, sequence, width // 2, 2)
                paired_initial = initial.reshape(batch, width // 2, 2)
                raw = _pair_affine_scan(
                    transition_matrix,
                    paired_write,
                    paired_initial,
                ).reshape(batch, sequence, width)
        if return_persistent_adapter_state:
            if feature_context_states is not None:
                return (
                    raw,
                    persistent_adapter_states[:, -1]
                    if persistent_adapter_states is not None
                    else None,
                    feature_context_states[:, -1],
                )
            final_persistent_state = (
                persistent_adapter_states[:, -1]
                if persistent_adapter_states is not None
                else None
            )
            return raw, final_persistent_state
        return raw

    def _settling_round(
        self,
        observation: Tensor,
        state_old: Tensor,
        temporal_drive: Tensor,
        visit: int = 0,
        *,
        generative: Tensor | None = None,
        reciprocal: Tensor | None = None,
        hgf_child_predicted_precision: Tensor | None = None,
        hgf_posterior_precision: Tensor | None = None,
    ) -> Tensor:
        generative_weight = self.generative if generative is None else generative
        reciprocal_weight = self.reciprocal if reciprocal is None else reciprocal
        if generative_weight.shape != self.generative.shape:
            raise ValueError("settling generative override shape differs")
        if reciprocal_weight.shape != self.reciprocal.shape:
            raise ValueError("settling reciprocal override shape differs")
        lower_states = torch.cat((observation.unsqueeze(0), state_old[:-1]), dim=0)
        lower_predictions = torch.bmm(state_old, generative_weight)
        lower_errors = lower_states - lower_predictions
        if self.config.exact_diagonal_hgf_posterior_sweep:
            effective_reciprocal, _, _, _ = diagonal_hgf_effective_reciprocal(
                reciprocal_weight
            )
            bottom_up_corrections = torch.bmm(
                lower_errors,
                effective_reciprocal,
            )
        elif self.config.predicted_precision_hgf_posterior_sweep:
            effective_reciprocal, _, _, _ = (
                predicted_precision_hgf_effective_reciprocal(
                    reciprocal_weight
                )
            )
            bottom_up_corrections = torch.bmm(
                lower_errors,
                effective_reciprocal,
            )
        elif self.config.temporal_hgf_precision_carry:
            if (
                hgf_child_predicted_precision is None
                or hgf_posterior_precision is None
            ):
                raise ValueError("temporal HGF settling precision is missing")
            bottom_up_corrections = temporal_hgf_bottom_up(
                lower_errors,
                reciprocal_weight,
                hgf_child_predicted_precision,
                hgf_posterior_precision,
            )
        else:
            bottom_up_corrections = torch.bmm(lower_errors, reciprocal_weight)

        top_predictions = torch.cat(
            (lower_predictions[1:], torch.zeros_like(lower_predictions[-1:])),
            dim=0,
        )
        target = (
            temporal_drive
            + bottom_up_corrections
            + self.config.top_down_scale * (top_predictions - state_old)
        )
        if not self.config.deeploop_postnorm_settling:
            target = _rms_norm(target, self.drive_norm, self.config.rms_eps)
        if self.settling_visit_frames is None:
            candidate = self.activate_settling_candidate(
                self.candidate_pre_for_visit(target, visit)
            )
        else:
            frame = self.settling_visit_frames[visit].view(
                self.config.depth,
                1,
                self.config.width,
            )
            candidate = (
                self.activate_settling_candidate(
                    self.candidate_pre_for_visit(target * frame, visit)
                )
                * frame
            )
        if self.config.deeploop_postnorm_settling:
            return _rms_norm(
                self.deeploop_skip_scale() * state_old + candidate,
                self.drive_norm,
                self.config.rms_eps,
            )
        state_step = self.settling_state_step()
        return state_old + state_step * (candidate - state_old)

    def _settle(
        self,
        observation: Tensor,
        raw_states: Tensor,
        *,
        hgf_child_predicted_precision: Tensor | None = None,
        hgf_posterior_precision: Tensor | None = None,
    ) -> Tensor:
        settled = raw_states
        for visit in range(self.config.settling_rounds):
            settled = self._settling_round(
                observation,
                settled,
                raw_states,
                visit,
                hgf_child_predicted_precision=hgf_child_predicted_precision,
                hgf_posterior_precision=hgf_posterior_precision,
            )
        return settled

    def _readout_state(self, settled_states: Tensor) -> Tensor:
        if settled_states.shape[0] != self.config.depth:
            raise ValueError("settled states must start with layer dimension")
        if self.config.fixed_orthogonal_depth_readout:
            if self.fixed_orthogonal_depth_readout_signs is None:
                raise RuntimeError("fixed depth readout signs are missing")
            return _fixed_orthogonal_depth_readout(
                settled_states,
                self.fixed_orthogonal_depth_readout_signs,
            )
        if self.readout_projection is not None:
            if self.config.hybrid_projected_readout:
                if self.config.depth < 2:
                    return settled_states[0]
                original_shape = settled_states.shape[1:]
                flat_states = settled_states[1:].reshape(
                    self.config.depth - 1,
                    -1,
                    self.config.width,
                )
                projected = (
                    torch.bmm(
                        flat_states,
                        self.readout_projection[1:],
                    )
                    .sum(dim=0)
                    .reshape(original_shape)
                )
                return settled_states[0] + projected
            original_shape = settled_states.shape[1:]
            flat_states = settled_states.reshape(
                self.config.depth,
                -1,
                self.config.width,
            )
            return (
                torch.bmm(
                    flat_states,
                    self.readout_projection,
                )
                .sum(dim=0)
                .reshape(original_shape)
            )
        if self.readout_scales is None:
            return settled_states[0]
        scale_shape = (self.config.depth,) + (1,) * (settled_states.ndim - 1)
        return (settled_states * self.readout_scales.view(scale_shape)).sum(dim=0)

    def _depth_innovation_readout_components(
        self,
        settled_states: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (
            not self.config.single_endpoint_depth_innovation_predictor
            or self.readout_projection is None
            or not self.config.hybrid_projected_readout
            or settled_states.shape[0] != 2
        ):
            raise RuntimeError("single-endpoint depth innovation is disabled")
        upper_contribution = settled_states[1] @ self.readout_projection[1]
        return settled_states[0] + upper_contribution, upper_contribution

    def _prediction_features_and_logits(
        self,
        settled_states: Tensor,
        swiglu_lexical_feature: Tensor | None = None,
        token_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        confidence_scale = None
        if (
            self.config.depth_wide_ternary_lexical_endpoint
            or self.config.feature_ready_depth_wide_lexical_residual
            or self.config.cozero_reciprocal_lexical_adapters
        ):
            predictions, logits = self._depth_wide_prediction_features_and_logits(
                settled_states
            )
            prediction = predictions[0]
        elif self.config.single_endpoint_depth_innovation_predictor:
            readout_state, upper_contribution = (
                self._depth_innovation_readout_components(settled_states)
            )
            normalized = _rms_norm(
                readout_state,
                self.output_norm,
                self.config.rms_eps,
            )
            innovation_normalized = _rms_norm(
                upper_contribution,
                self.output_norm,
                self.config.rms_eps,
            )
            packed_runtime = getattr(
                self,
                "_phase778_packed_depth_innovation_predictor",
                None,
            )
            if (
                packed_runtime is not None
                and torch.is_inference_mode_enabled()
                and normalized.is_cuda
            ):
                preactivation = packed_runtime.matmul(
                    normalized,
                    self.token_predictor,
                    innovation_normalized,
                )
            else:
                predictor = self.depth_innovation_predictor
                if predictor is None:
                    raise RuntimeError("depth innovation predictor is missing")
                preactivation = (
                    normalized @ self.token_predictor
                    + innovation_normalized @ predictor
                )
            prediction = torch.tanh(preactivation)
            logits = self.decode_features(prediction)
        else:
            readout_state = self._readout_state(settled_states)
            normalized = _rms_norm(
                readout_state,
                self.output_norm,
                self.config.rms_eps,
            )
            prediction = self._token_prediction_features(normalized)
            if self.config.energy_conserving_lexical_group_contrast:
                from pfc.phase741_lexical_group_contrast import (
                    energy_conserving_lexical_group_contrast,
                )

                prediction = energy_conserving_lexical_group_contrast(
                    prediction,
                    rms_eps=self.config.rms_eps,
                )
            if self.config.causal_anti_hebbian_lexical_butterfly:
                coefficients = self.anti_hebbian_lexical_butterfly_coefficients
                if coefficients is None:
                    raise RuntimeError(
                        "causal anti-Hebbian lexical butterfly coefficients are missing"
                    )
                fused_execution = (
                    self.config.fused_causal_anti_hebbian_lexical_butterfly_execution
                    and torch.is_inference_mode_enabled()
                    and prediction.is_cuda
                    and prediction.dtype == torch.float32
                    and coefficients.dtype == torch.float32
                    and prediction.is_contiguous()
                    and coefficients.is_contiguous()
                )
                if fused_execution:
                    from pfc.phase743_triton_anti_hebbian_lexical_butterfly import (
                        fused_anti_hebbian_lexical_butterfly_inference,
                    )

                    prediction = fused_anti_hebbian_lexical_butterfly_inference(
                        prediction,
                        coefficients,
                    )
                else:
                    from pfc.phase742_anti_hebbian_lexical_butterfly import (
                        anti_hebbian_lexical_butterfly,
                    )

                    prediction = anti_hebbian_lexical_butterfly(
                        prediction,
                        coefficients,
                    )
            logits = self.decode_features(prediction)
            if self.confidence_readout_weights is not None:
                confidence_features = self._confidence_features(normalized)
                confidence_scale = self._confidence_scale(confidence_features)
        if self.shared_base_ternary_delta_decoder is not None:
            if self.shared_base_delta_gate_signs is None:
                raise RuntimeError("shared-base mixture gate signs are missing")
            logits = shared_base_delta_mixture(
                prediction,
                logits,
                self.decode_shared_base_ternary_delta(prediction),
                gate_signs=self.shared_base_delta_gate_signs,
            )[0]
        if self.swiglu_lexical_decoder is not None:
            if swiglu_lexical_feature is None:
                raise ValueError("SwiGLU lexical feature is required")
            logits = logits + self.decode_swiglu_lexical_feature(
                swiglu_lexical_feature
            )
        elif swiglu_lexical_feature is not None:
            raise ValueError("SwiGLU lexical feature supplied to disabled co-head")
        if self.context_routed_lexical_residual_decoders is not None:
            if token_ids is None:
                raise ValueError("context-routed lexical residual requires token ids")
            logits = logits + self.decode_context_routed_lexical_residual(
                prediction,
                token_ids,
            )
        elif token_ids is not None and self.config.context_routed_ternary_lexical_residual:
            raise RuntimeError("context-routed lexical residual decoder is missing")
        if self.bounded_multiplicative_lexical_decoder is not None:
            from pfc.phase676_bounded_multiplicative_lexical_modulator import (
                bounded_multiplicative_logits,
            )

            auxiliary_logits = (
                self.decode_bounded_multiplicative_lexical_modulator(prediction)
            )
            logits, _, _ = bounded_multiplicative_logits(
                logits,
                auxiliary_logits,
            )
        if self.signed_square_lexical_decoder is not None:
            logits = logits + self.decode_signed_square_lexical_lift(prediction)
        if self.cross_channel_quadratic_lexical_decoder is not None:
            logits = logits + self.decode_cross_channel_quadratic_lexical_lift(
                prediction
            )
        if confidence_scale is not None:
            logits = logits * confidence_scale
        if self.config.fixed_hadamard_top8_lexical_reranker:
            packed_runtime = getattr(
                self,
                "_phase804_packed_top8_lexical_reranker",
                None,
            )
            if packed_runtime is not None:
                logits = packed_runtime.apply(logits, prediction)[0]
            else:
                lexical_codes = self.top8_lexical_reranker_codes
                if lexical_codes is None:
                    raise RuntimeError("Phase804 lexical reranker table is missing")
                from pfc.phase804_fixed_hadamard_top8_lexical_reranker import (
                    apply_top8_lexical_reranker,
                )

                logits = apply_top8_lexical_reranker(
                    logits,
                    prediction,
                    lexical_codes,
                )[0]
        if self.config.shared_nonlinear_candidate_compatibility:
            packed_runtime = getattr(
                self,
                "_phase811_packed_shared_candidate_comparator",
                None,
            )
            comparator = (
                packed_runtime.vector(dtype=prediction.dtype)
                if packed_runtime is not None
                else self.shared_candidate_compatibility_vector
            )
            signs = self.fixed_ternary_hadamard_lexical_signs
            if comparator is None or signs is None:
                raise RuntimeError(
                    "shared nonlinear candidate compatibility is incomplete"
                )
            from pfc.phase811_shared_nonlinear_candidate_compatibility import (
                apply_shared_nonlinear_candidate_compatibility,
            )

            logits = apply_shared_nonlinear_candidate_compatibility(
                logits,
                prediction,
                self.embedding.weight,
                comparator,
                signs,
                rms_eps=self.config.rms_eps,
            )[0]
        return prediction, logits

    def _depth_wide_prediction_features_and_logits(
        self,
        settled_states: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (
            not (
                self.config.depth_wide_ternary_lexical_endpoint
                or self.config.feature_ready_depth_wide_lexical_residual
                or self.config.cozero_reciprocal_lexical_adapters
            )
            or self.depth_lexical_decoders is None
            or self.depth_lexical_output_norms is None
        ):
            raise RuntimeError("depth-wide lexical endpoint is disabled")
        if settled_states.shape[0] != self.config.depth:
            raise ValueError("settled states must start with layer dimension")

        parent_readout = self._readout_state(settled_states)
        parent_normalized = _rms_norm(
            parent_readout,
            self.output_norm,
            self.config.rms_eps,
        )
        parent_prediction = torch.tanh(
            parent_normalized @ self.token_predictor
        )
        extra_norm_shape = (
            (self.config.depth - 1,)
            + (1,) * (settled_states.ndim - 2)
            + (self.config.width,)
        )
        extra_normalized = _rms_norm(
            settled_states[1:],
            self.depth_lexical_output_norms.view(extra_norm_shape),
            self.config.rms_eps,
        )
        if self.config.depth_wide_ternary_lexical_endpoint:
            if self.depth_lexical_predictors is None:
                raise RuntimeError("depth-wide lexical predictors are missing")
            extra_prediction = torch.tanh(
                torch.einsum(
                    "d...w,dwk->d...k",
                    extra_normalized,
                    self.depth_lexical_predictors,
                )
            )
        else:
            if self.depth_lexical_predictors is not None:
                raise RuntimeError("feature-ready lexical residual has predictors")
            extra_prediction = torch.tanh(extra_normalized)
        logits = self.decode_features(parent_prediction) + torch.einsum(
            "d...w,dwv->...v",
            extra_prediction,
            self.depth_lexical_decoders,
        )
        return torch.cat(
            (parent_prediction.unsqueeze(0), extra_prediction),
            dim=0,
        ), logits

    def _logits(
        self,
        settled_states: Tensor,
        swiglu_lexical_feature: Tensor | None = None,
        token_ids: Tensor | None = None,
    ) -> Tensor:
        _, logits = self._prediction_features_and_logits(
            settled_states,
            swiglu_lexical_feature,
            token_ids,
        )
        return logits

    def _confidence_features(self, normalized_readout: Tensor) -> Tensor:
        if self.confidence_readout_weights is None:
            raise RuntimeError("state-conditioned confidence readout is disabled")
        group_width = self.config.width // self.confidence_readout_weights.numel()
        return (
            normalized_readout.unflatten(
                -1,
                (self.confidence_readout_weights.numel(), group_width),
            )
            .square()
            .mean(dim=-1)
        )

    def _confidence_scale(self, confidence_features: Tensor) -> Tensor:
        if (
            self.confidence_readout_weights is None
            or self.confidence_readout_bias is None
        ):
            raise RuntimeError("state-conditioned confidence readout is disabled")
        raw_confidence = (
            confidence_features @ self.confidence_readout_weights
            + self.confidence_readout_bias
        )
        return torch.exp(torch.tanh(raw_confidence)).unsqueeze(-1)

    def step_with_prediction_features(
        self,
        token_ids: Tensor,
        state: NativePCState | None = None,
        *,
        return_swiglu_lexical_feature: bool = False,
    ) -> (
        tuple[Tensor, Tensor, NativePCState]
        | tuple[Tensor, Tensor, NativePCState, Tensor]
    ):
        if token_ids.ndim != 1:
            raise ValueError("token_ids must have shape [batch]")
        if (
            return_swiglu_lexical_feature
            and not self.config.swiglu_wide_lexical_cohead
        ):
            raise ValueError("SwiGLU lexical feature return requested while disabled")
        if self.config.bos_boundary_state_reset:
            boundary = token_ids.eq(self.config.bos_token_id)
            if bool(boundary.any().item()):
                if not bool(boundary.all().item()):
                    raise ValueError("BOS boundary positions must match across batch")
                state = self.empty_state(
                    token_ids.shape[0],
                    device=token_ids.device,
                )
        if state is None:
            state = self.empty_state(token_ids.shape[0], device=token_ids.device)
        self._validate_state(state, token_ids.shape[0])
        if (
            state.token_count_cuda is not None
            and self.config.associative_memory_rank > 0
            and not self.config.fused_indexed_count_sketch_inference
        ):
            raise ValueError(
                "CUDA graph token count requires fused indexed CountSketch"
            )
        next_token_count: int | Tensor = (
            state.token_count_cuda + 1
            if state.token_count_cuda is not None
            else state.token_count + 1
        )

        observation = self.embed_tokens(token_ids)
        manifold_lower_streams = None
        manifold_post = None
        manifold_exchange = None
        manifold_compact_maps = None
        fused_manifold_step = False
        if self.config.bounded_manifold_depth_bus:
            if state.manifold_depth_bus is None:
                raise RuntimeError("bounded manifold depth-bus state is missing")
            observation_streams = observation.unsqueeze(0).unsqueeze(2).expand(
                1,
                observation.shape[0],
                2,
                observation.shape[1],
            )
            manifold_lower_streams = torch.cat(
                (observation_streams, state.manifold_depth_bus[:-1]),
                dim=0,
            )
            fused_manifold_step = (
                self.config.fused_manifold_depth_bus_inference
                and manifold_lower_streams.is_cuda
                and manifold_lower_streams.dtype == torch.float16
                and not torch.is_grad_enabled()
            )
            if fused_manifold_step:
                if (
                    self.manifold_depth_bus_controller is None
                    or self.manifold_depth_bus_bias is None
                ):
                    raise RuntimeError("bounded manifold controller is missing")
                lower, manifold_compact_maps = fused_manifold_depth_bus_maps(
                    manifold_lower_streams,
                    self.manifold_depth_bus_controller,
                    self.manifold_depth_bus_bias,
                    rms_eps=self.config.rms_eps,
                )
            else:
                lower, _, manifold_post, manifold_exchange = (
                    self._manifold_depth_bus_maps(manifold_lower_streams)
                )
        else:
            lower = (
                self._depth_attention_step_inputs(observation, state.layers)
                if self.config.bounded_depth_attention_residual
                else torch.cat(
                    (observation.unsqueeze(0), state.layers[:-1]),
                    dim=0,
                )
            )
        if self.config.token_gated_diagonal_handoff:
            gated_handoff = self._token_gated_handoff(observation, lower[1])
            lower = torch.cat(
                (lower[:1], gated_handoff.source.unsqueeze(0)),
                dim=0,
            )
        nonlinear_matrix_state = None
        if self.config.nonlinear_matrix_state_lane:
            if state.nonlinear_matrix_state is None:
                raise RuntimeError("nonlinear matrix state is missing")
            normalized_matrix_source = _rms_norm(
                lower[0],
                self.state_norm[0, 0],
                self.config.rms_eps,
            )
            matrix_injection, nonlinear_matrix_state = self._nonlinear_matrix_lane(
                normalized_matrix_source,
                state.nonlinear_matrix_state,
            )
            lower = torch.cat(
                ((lower[0] + matrix_injection).unsqueeze(0), lower[1:]),
                dim=0,
            )
        causal_fir_input_history = None
        causal_fir_input_valid_count = None
        if self.config.causal_ternary_fir_input_state:
            if (
                state.causal_fir_input_history is None
                or state.causal_fir_input_valid_count is None
            ):
                raise RuntimeError("causal FIR input state is incomplete")
            causal_fir_taps = self._causal_ternary_fir_taps(lower.dtype)
            (
                lower,
                causal_fir_input_history,
                causal_fir_input_valid_count,
                _,
                _,
            ) = causal_ternary_fir_step(
                lower,
                state.causal_fir_input_history,
                state.causal_fir_input_valid_count,
                causal_fir_taps,
            )
        associative_read: Tensor | None = None
        memory = None
        normalizer = None
        keys = None
        last_update = None
        associative_encoded_source = None
        semantic_memory = None
        semantic_normalizer = None
        semantic_last_update = None
        semantic_context = None
        semantic_codes = None
        pair_recurrence_bloom = state.associative_pair_recurrence_bloom
        semantic_enabled_step = None
        if self._uses_pair_miss_semantic_backoff():
            if (
                state.associative_context_tokens is None
                or pair_recurrence_bloom is None
            ):
                raise RuntimeError("pair recurrence Bloom state is missing")
            pair_seen, pair_recurrence_bloom = self._pair_recurrence_bloom_step(
                state.associative_context_tokens,
                token_ids,
                pair_recurrence_bloom,
                state.token_count,
            )
            semantic_enabled_step = (~pair_seen).unsqueeze(-1)
        if self.config.associative_memory_rank > 0:
            if (
                self.config.associative_memory_addressing
                in LAZY_COUNT_SKETCH_ADDRESSINGS
            ):
                (
                    associative_read,
                    memory,
                    normalizer,
                    keys,
                    last_update,
                    associative_encoded_source,
                ) = self._lazy_multiscale_count_sketch_step(
                    lower,
                    state,
                    token_ids,
                    next_token_count=next_token_count,
                )
                if self._uses_semantic_count_sketch_backoff():
                    (
                        semantic_reads,
                        semantic_active,
                        semantic_memory,
                        semantic_normalizer,
                        semantic_last_update,
                        semantic_context,
                        semantic_codes,
                    ) = self._lazy_semantic_count_sketch_step(
                        lower,
                        state,
                        associative_encoded_source,
                        next_token_count=next_token_count,
                    )
                    if semantic_enabled_step is not None:
                        semantic_active = tuple(
                            active & semantic_enabled_step
                            for active in semantic_active
                        )
                    if self._uses_disjoint_timescale_semantic_channels():
                        if (
                            self.config.fused_indexed_count_sketch_inference
                            and _uses_fp16_cuda_fused_inference(lower)
                            and semantic_reads[0].shape[-1]
                            == SEMANTIC_COUNT_SKETCH_VALUE_WIDTH
                        ):
                            if self.associative_semantic_channel_decoders is None:
                                raise RuntimeError(
                                    "disjoint semantic channel decoders are missing"
                                )
                            fused_compose_disjoint_semantic_channels_in_place(
                                associative_read[0],
                                semantic_reads[0],
                                semantic_reads[1],
                                semantic_active[0],
                                semantic_active[1],
                                self.associative_semantic_channel_decoders,
                            )
                        else:
                            if semantic_reads[0].shape[-1] == (
                                SEMANTIC_COUNT_SKETCH_VALUE_WIDTH
                            ):
                                semantic_reads = (
                                    self._decode_semantic_channel_values(
                                        semantic_reads[0], 0
                                    ),
                                    self._decode_semantic_channel_values(
                                        semantic_reads[1], 1
                                    ),
                                )
                            combined = (
                                self._compose_disjoint_timescale_semantic_channel_read(
                                    associative_read[0],
                                    semantic_reads,
                                    semantic_active,
                                )
                            )
                            associative_read = torch.cat(
                                (combined.unsqueeze(0), associative_read[1:]),
                                dim=0,
                            )
                    elif (
                        self.config.fused_indexed_count_sketch_inference
                        and _uses_fp16_cuda_fused_inference(lower)
                    ):
                        fused_compose_semantic_read_in_place(
                            associative_read[0],
                            semantic_reads[0],
                            semantic_reads[1],
                            semantic_active[0],
                            semantic_active[1],
                            self.associative_semantic_value_decoder,
                        )
                    else:
                        combined = self._compose_semantic_count_sketch_read(
                            associative_read[0],
                            semantic_reads,
                            semantic_active,
                        )
                        associative_read = torch.cat(
                            (combined.unsqueeze(0), associative_read[1:]),
                            dim=0,
                        )
            else:
                associative_read, memory, normalizer, keys = self._associative_step(
                    lower,
                    state,
                    token_ids,
                )
        gdn2_read = None
        gdn2_memory = None
        if self.config.gdn2_mixer_layers > 0:
            if state.gdn2_memory is None:
                raise RuntimeError("GDN2 mixer state is missing")
            normalized_lower = _rms_norm(
                lower,
                self.state_norm,
                self.config.rms_eps,
            )
            gdn2_reads = []
            gdn2_memories = []
            for layer in range(self.config.gdn2_mixer_layers):
                layer_read, layer_memory = self._gdn2_sequence(
                    normalized_lower[layer].unsqueeze(1),
                    state.gdn2_memory[layer],
                    layer,
                )
                gdn2_reads.append(layer_read[:, 0])
                gdn2_memories.append(layer_memory)
            gdn2_read = torch.cat(
                (
                    torch.stack(gdn2_reads),
                    torch.zeros_like(lower[self.config.gdn2_mixer_layers :]),
                ),
                dim=0,
            )
            gdn2_memory = torch.stack(gdn2_memories)
        factorization_read = None
        factorization_memory = None
        if self.config.factorization_memory_layers > 0:
            if state.factorization_memory is None:
                raise RuntimeError("factorization memory state is missing")
            start_layer = self._factorization_start_layer()
            factorization_reads = []
            factorization_memories = []
            for layer in range(start_layer, self.config.depth):
                memory_layer = layer - start_layer
                layer_read, layer_memory = self._factorization_memory_step(
                    lower[layer],
                    state.factorization_memory[memory_layer],
                    memory_layer,
                )
                factorization_reads.append(layer_read)
                factorization_memories.append(layer_memory)
            factorization_read = torch.cat(
                (
                    torch.zeros_like(lower[:start_layer]),
                    torch.stack(factorization_reads),
                ),
                dim=0,
            )
            factorization_memory = torch.stack(factorization_memories)
        bounded_attention_read = None
        bounded_attention_keys = None
        bounded_attention_values = None
        if self.config.bounded_attention_layers > 0:
            if (
                state.bounded_attention_keys is None
                or state.bounded_attention_values is None
            ):
                raise RuntimeError("bounded attention state is missing")
            start_layer = self._bounded_attention_start_layer()
            bounded_reads = []
            bounded_keys = []
            bounded_values = []
            for layer in range(start_layer, self.config.depth):
                attention_layer = layer - start_layer
                layer_read, layer_keys, layer_values = self._bounded_attention_sequence(
                    lower[layer].unsqueeze(1),
                    state.bounded_attention_keys[attention_layer],
                    state.bounded_attention_values[attention_layer],
                    attention_layer,
                    state.token_count,
                )
                bounded_reads.append(layer_read[:, 0])
                bounded_keys.append(layer_keys)
                bounded_values.append(layer_values)
            bounded_attention_read = torch.cat(
                (
                    torch.zeros_like(lower[:start_layer]),
                    torch.stack(bounded_reads),
                ),
                dim=0,
            )
            bounded_attention_keys = torch.stack(bounded_keys)
            bounded_attention_values = torch.stack(bounded_values)
        transition_result = self._transition_layers(
            lower,
            (
                state.recurrent_layers
                if (
                    self.config.decoupled_bounded_memory_emission
                    or self.config.causal_level_innovation_lifting
                )
                else state.layers
            ),
            observation,
            associative_read,
            state.persistent_transition_adapter,
            gdn2_read,
            factorization_read,
            bounded_attention_read,
            token_ids=token_ids,
            temporal_age=next_token_count,
            feature_context_initial=state.feature_context,
            return_persistent_adapter_state=True,
        )
        if not isinstance(transition_result, tuple):
            raise RuntimeError("transition state result is missing")
        recurrent_layers = None
        feature_context = None
        swiglu_lexical_feature = None
        if self.config.swiglu_wide_lexical_cohead:
            raw_states, persistent_transition_adapter, swiglu_lexical_feature = (
                transition_result
            )
        elif (
            self.config.decoupled_bounded_memory_emission
            or self.config.causal_level_innovation_lifting
        ):
            raw_states, persistent_transition_adapter, recurrent_layers = (
                transition_result
            )
        elif self._uses_recurrent_feature_context():
            raw_states, persistent_transition_adapter, feature_context = (
                transition_result
            )
        else:
            raw_states, persistent_transition_adapter = transition_result
        manifold_depth_bus = None
        emitted_raw_states = raw_states
        if manifold_lower_streams is not None:
            if fused_manifold_step:
                if manifold_compact_maps is None:
                    raise RuntimeError("fused bounded manifold maps are missing")
                (
                    manifold_depth_bus,
                    emitted_raw_states,
                ) = fused_manifold_depth_bus_output(
                    manifold_lower_streams,
                    lower,
                    raw_states,
                    manifold_compact_maps,
                )
            else:
                if manifold_post is None or manifold_exchange is None:
                    raise RuntimeError("bounded manifold maps are missing")
                manifold_depth_bus = self._manifold_depth_bus_output(
                    manifold_lower_streams,
                    lower,
                    raw_states,
                    manifold_post,
                    manifold_exchange,
                )
                emitted_raw_states = manifold_depth_bus.mean(dim=2)
        latent_prediction_error = None
        lagged_prediction_error = None
        if self.config.lagged_latent_error_settling_carrier:
            if state.latent_prediction_error is None:
                raise RuntimeError("latent prediction-error state is missing")
            latent_prediction_error = stored_latent_prediction_error(
                observation,
                raw_states,
                self.generative,
            )
            lagged_prediction_error = lagged_latent_error_step(
                state.latent_prediction_error,
                token_ids,
                self.config.bos_token_id,
            )
            emitted_raw_states = apply_lagged_latent_error(
                raw_states,
                lagged_prediction_error,
                self._latent_error_settling_gain(raw_states.dtype),
            )
        settling_observation, _ = self._state_conditioned_settling_observation(
            observation.unsqueeze(1),
            raw_states[-1].unsqueeze(1),
            state.layers[-1],
            initial_active=(
                state.token_count_cuda > 0
                if state.token_count_cuda is not None
                else state.token_count > 0
            ),
        )
        hgf_child_predicted_precision = None
        hgf_posterior_precision = None
        next_hgf_posterior_precision = None
        if self.config.temporal_hgf_precision_carry:
            if state.hgf_posterior_precision is None:
                raise RuntimeError("temporal HGF posterior precision is missing")
            (
                hgf_child_sequence,
                _hgf_predicted_sequence,
                _hgf_evidence_sequence,
                hgf_posterior_sequence,
                next_hgf_posterior_precision,
                _hgf_diagnostics,
            ) = temporal_hgf_precision_sequence(
                self.reciprocal,
                state.hgf_posterior_precision,
                sequence_length=1,
            )
            hgf_child_predicted_precision = hgf_child_sequence[:, :, 0]
            hgf_posterior_precision = hgf_posterior_sequence[:, :, 0]
        settled = self._settle(
            settling_observation[:, 0],
            emitted_raw_states,
            hgf_child_predicted_precision=hgf_child_predicted_precision,
            hgf_posterior_precision=hgf_posterior_precision,
        )
        prediction_features, logits = self._prediction_features_and_logits(
            settled,
            swiglu_lexical_feature,
            token_ids,
        )
        next_state = NativePCState(
            layers=raw_states,
            token_count=state.token_count + 1,
            token_count_cuda=(
                next_token_count
                if isinstance(next_token_count, Tensor)
                else None
            ),
            manifold_depth_bus=manifold_depth_bus,
            nonlinear_matrix_state=nonlinear_matrix_state,
            recurrent_layers=recurrent_layers,
            feature_context=feature_context,
            associative_memory=memory,
            associative_normalizer=normalizer,
            associative_keys=keys,
            associative_last_update=last_update,
            associative_context_tokens=(
                token_ids
                if self.config.associative_memory_addressing
                in {
                    "token_bigram_hash",
                    "token_plus_bigram_hash",
                    "token_plus_context_bucket",
                    "indexed_lazy_multiscale_count_sketch",
                    *DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS,
                }
                else None
            ),
            associative_previous_context_tokens=(
                state.associative_context_tokens
                if self.config.associative_memory_addressing
                in DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS
                else None
            ),
            associative_semantic_memory=semantic_memory,
            associative_semantic_normalizer=semantic_normalizer,
            associative_semantic_last_update=semantic_last_update,
            associative_semantic_context=semantic_context,
            associative_semantic_codes=semantic_codes,
            associative_pair_recurrence_bloom=pair_recurrence_bloom,
            persistent_transition_adapter=persistent_transition_adapter,
            gdn2_memory=gdn2_memory,
            factorization_memory=factorization_memory,
            bounded_attention_keys=bounded_attention_keys,
            bounded_attention_values=bounded_attention_values,
            hgf_posterior_precision=next_hgf_posterior_precision,
            causal_fir_input_history=causal_fir_input_history,
            causal_fir_input_valid_count=causal_fir_input_valid_count,
            latent_prediction_error=latent_prediction_error,
        )
        if return_swiglu_lexical_feature:
            if swiglu_lexical_feature is None:
                raise RuntimeError("SwiGLU lexical step feature was not retained")
            return logits, prediction_features, next_state, swiglu_lexical_feature
        return logits, prediction_features, next_state

    def step(
        self,
        token_ids: Tensor,
        state: NativePCState | None = None,
    ) -> tuple[Tensor, NativePCState]:
        logits, _, next_state = self.step_with_prediction_features(token_ids, state)
        return logits, next_state

    def _prefill_states(
        self,
        token_ids: Tensor,
        state: NativePCState | None,
        *,
        return_associative_reads: bool = False,
        return_swiglu_lexical_features: bool = False,
        return_recurrent_sequences: bool = False,
        return_lagged_latent_errors: bool = False,
    ) -> tuple[object, ...]:
        if (
            return_swiglu_lexical_features
            and not self.config.swiglu_wide_lexical_cohead
        ):
            raise ValueError("SwiGLU lexical feature return requested while disabled")
        if (
            return_recurrent_sequences
            and not (
                self.config.causal_level_innovation_lifting
                or self.config.lagged_latent_error_settling_carrier
            )
        ):
            raise ValueError("causal recurrent sequence return requested while disabled")
        if (
            return_lagged_latent_errors
            and not self.config.lagged_latent_error_settling_carrier
        ):
            raise ValueError("lagged latent-error return requested while disabled")
        if not self.config.bos_boundary_state_reset:
            return self._prefill_states_contiguous(
                token_ids,
                state,
                return_associative_reads=return_associative_reads,
                return_swiglu_lexical_features=return_swiglu_lexical_features,
                return_recurrent_sequences=return_recurrent_sequences,
                return_lagged_latent_errors=return_lagged_latent_errors,
            )
        if self.training and not self.config.bos_boundary_exact_segmented_training:
            return self._prefill_states_contiguous(
                token_ids,
                state,
                return_associative_reads=return_associative_reads,
                return_swiglu_lexical_features=return_swiglu_lexical_features,
                return_recurrent_sequences=return_recurrent_sequences,
                return_lagged_latent_errors=return_lagged_latent_errors,
                apply_boundary_resets=True,
            )
        segments = shared_bos_segments(token_ids, self.config.bos_token_id)
        if len(segments) == 1 and not segments[0][2]:
            return self._prefill_states_contiguous(
                token_ids,
                state,
                return_associative_reads=return_associative_reads,
                return_swiglu_lexical_features=return_swiglu_lexical_features,
                return_recurrent_sequences=return_recurrent_sequences,
                return_lagged_latent_errors=return_lagged_latent_errors,
                apply_boundary_resets=False,
            )
        observation_parts = []
        raw_parts = []
        settled_parts = []
        swiglu_lexical_parts = []
        associative_parts = []
        recurrent_parts = []
        latent_error_parts = []
        current = state
        for start, end, reset in segments:
            if reset:
                current = None
            result = self._prefill_states_contiguous(
                token_ids[:, start:end],
                current,
                return_associative_reads=return_associative_reads,
                return_swiglu_lexical_features=return_swiglu_lexical_features,
                return_recurrent_sequences=return_recurrent_sequences,
                return_lagged_latent_errors=return_lagged_latent_errors,
                apply_boundary_resets=False,
            )
            observation_parts.append(result[0])
            raw_parts.append(result[1])
            settled_parts.append(result[2])
            current = result[3]
            if return_swiglu_lexical_features:
                swiglu_lexical_parts.append(result[4])
            if return_associative_reads:
                associative_parts.append(
                    result[5] if return_swiglu_lexical_features else result[4]
                )
            if return_recurrent_sequences:
                recurrent_index = 4 + int(return_swiglu_lexical_features) + int(
                    return_associative_reads
                )
                recurrent_parts.append(result[recurrent_index])
            if return_lagged_latent_errors:
                latent_error_index = (
                    4
                    + int(return_swiglu_lexical_features)
                    + int(return_associative_reads)
                    + int(return_recurrent_sequences)
                )
                latent_error_parts.append(result[latent_error_index])
        if current is None:
            raise RuntimeError("BOS-segmented prefill produced no state")
        combined = (
            torch.cat(observation_parts, dim=1),
            torch.cat(raw_parts, dim=2),
            torch.cat(settled_parts, dim=2),
            current,
        )
        if return_swiglu_lexical_features:
            combined = (*combined, torch.cat(swiglu_lexical_parts, dim=1))
        if return_associative_reads:
            combined = (*combined, torch.cat(associative_parts, dim=2))
        if return_recurrent_sequences:
            combined = (*combined, torch.cat(recurrent_parts, dim=2))
        if return_lagged_latent_errors:
            combined = (
                *combined,
                (
                    torch.cat([part[0] for part in latent_error_parts], dim=2),
                    torch.cat([part[1] for part in latent_error_parts], dim=2),
                ),
            )
        return combined

    def _prefill_states_contiguous(
        self,
        token_ids: Tensor,
        state: NativePCState | None,
        *,
        return_associative_reads: bool = False,
        return_swiglu_lexical_features: bool = False,
        return_recurrent_sequences: bool = False,
        return_lagged_latent_errors: bool = False,
        apply_boundary_resets: bool = True,
    ) -> tuple[object, ...]:
        if (
            return_swiglu_lexical_features
            and not self.config.swiglu_wide_lexical_cohead
        ):
            raise ValueError("SwiGLU lexical feature return requested while disabled")
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] <= 0:
            raise ValueError("sequence must contain at least one token")
        if state is None:
            state = self.empty_state(token_ids.shape[0], device=token_ids.device)
        self._validate_state(state, token_ids.shape[0])
        if state.token_count_cuda is not None:
            raise ValueError("CUDA graph token-count state supports step only")
        boundary_reset_mask = (
            token_ids.eq(self.config.bos_token_id)
            if self.config.bos_boundary_state_reset and apply_boundary_resets
            else None
        )

        residual_route_plan = None
        if (
            self.config.cublas_routed_residual_expert_prefill
            and not self.training
            and token_ids.is_cuda
        ):
            from pfc.phase629_cublas_routed_residual_prefill import (
                build_cublas_residual_route_plan,
            )

            residual_route_plan = build_cublas_residual_route_plan(token_ids)

        initial_associative_memory = state.associative_memory
        initial_associative_normalizer = state.associative_normalizer
        if (
            self.config.associative_memory_addressing
            in LAZY_COUNT_SKETCH_ADDRESSINGS
        ):
            (
                initial_associative_memory,
                initial_associative_normalizer,
            ) = self._materialize_lazy_multiscale_count_sketch_state(state)
        initial_semantic_memory = state.associative_semantic_memory
        initial_semantic_normalizer = state.associative_semantic_normalizer
        if self._uses_semantic_count_sketch_backoff():
            (
                initial_semantic_memory,
                initial_semantic_normalizer,
            ) = self._materialize_lazy_semantic_count_sketch_state(state)
        final_pair_recurrence_bloom = state.associative_pair_recurrence_bloom
        semantic_enabled_sequence = None
        if self._uses_pair_miss_semantic_backoff():
            if (
                state.associative_context_tokens is None
                or final_pair_recurrence_bloom is None
            ):
                raise RuntimeError("pair recurrence Bloom state is missing")
            pair_seen, final_pair_recurrence_bloom = (
                self._pair_recurrence_bloom_sequence(
                    token_ids,
                    state.associative_context_tokens,
                    final_pair_recurrence_bloom,
                    state.token_count,
                    boundary_reset_mask,
                )
            )
            semantic_enabled_sequence = (~pair_seen).unsqueeze(-1)
        indexed_initial_key = None
        if (
            self.config.associative_memory_addressing
            in INDEXED_COUNT_SKETCH_ADDRESSINGS
        ):
            if initial_associative_memory is None:
                raise RuntimeError("indexed associative memory state is missing")
            statistics_dtype = initial_associative_memory.dtype
            if state.associative_context_tokens is None:
                raise RuntimeError("indexed lazy prior token state is missing")
            if (
                self.config.associative_memory_addressing
                in DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS
                and state.associative_previous_context_tokens is None
            ):
                raise RuntimeError("dual-order second prior token state is missing")
            indexed_initial_key = (
                self._multiscale_count_sketch_keys(
                    state.associative_context_tokens,
                    statistics_dtype,
                    state.associative_previous_context_tokens,
                )
                if state.token_count > 0
                else torch.zeros(
                    token_ids.shape[0],
                    self.config.associative_memory_rank,
                    device=token_ids.device,
                    dtype=statistics_dtype,
                )
            )

        observation = self.embed_tokens(token_ids)
        lower = observation
        depth_attention_sources = (
            [observation]
            if self.config.bounded_depth_attention_residual
            else None
        )
        manifold_lower_streams = (
            observation.unsqueeze(2).expand(
                observation.shape[0],
                observation.shape[1],
                2,
                observation.shape[2],
            )
            if self.config.bounded_manifold_depth_bus
            else None
        )
        raw_sequences = []
        core_sequences = []
        manifold_depth_bus_sequences = []
        final_recurrent_layers = []
        retained_recurrent_sequences = []
        final_feature_context = []
        swiglu_lexical_features = None
        final_memory = []
        final_normalizer = []
        final_keys = []
        final_semantic_memory = None
        final_semantic_normalizer = None
        final_semantic_context = None
        final_semantic_codes = None
        final_persistent_transition_adapter = []
        final_gdn2_memory = []
        final_factorization_memory = []
        final_bounded_attention_keys = []
        final_bounded_attention_values = []
        final_causal_fir_input_history = []
        final_causal_fir_input_valid_count = []
        retained_associative_reads = []
        nonlinear_matrix_state = None
        packed_reversible_projection = None
        causal_fir_taps = (
            self._causal_ternary_fir_taps(lower.dtype)
            if self.config.causal_ternary_fir_input_state
            else None
        )
        if self.config.exact_packed_reversible_projection_prefill:
            if self.transition_write_gate is None:
                raise RuntimeError("packed prefill write-gate projection is missing")
            from pfc.phase448_exact_packed_prefill_projection import (
                pack_reversible_prefill_projections,
            )

            packed_reversible_projection = pack_reversible_prefill_projections(
                self.temporal,
                self.gate,
                self.transition_write_gate,
                self.ffn_in,
            )
        for layer in range(self.config.depth):
            manifold_post = None
            manifold_exchange = None
            if manifold_lower_streams is not None:
                lower, _, manifold_post, manifold_exchange = (
                    self._manifold_depth_bus_maps(
                        manifold_lower_streams,
                        layer=layer,
                    )
                )
            if depth_attention_sources is not None and layer > 0:
                lower, _ = self._depth_attention_residual(
                    torch.stack(depth_attention_sources),
                    layer=layer,
                )
            if layer == 1 and self.config.token_gated_diagonal_handoff:
                lower = self._token_gated_handoff(observation, lower).source
            if layer == 0 and self.config.nonlinear_matrix_state_lane:
                if state.nonlinear_matrix_state is None:
                    raise RuntimeError("nonlinear matrix state is missing")
                normalized_matrix_source = _rms_norm(
                    lower,
                    self.state_norm[0, 0],
                    self.config.rms_eps,
                )
                matrix_injection, nonlinear_matrix_state = (
                    self._nonlinear_matrix_lane(
                        normalized_matrix_source,
                        state.nonlinear_matrix_state,
                    )
                )
                lower = lower + matrix_injection
            if self.config.causal_ternary_fir_input_state:
                if (
                    causal_fir_taps is None
                    or state.causal_fir_input_history is None
                    or state.causal_fir_input_valid_count is None
                ):
                    raise RuntimeError("causal FIR input state is incomplete")
                (
                    adjusted_lower,
                    layer_fir_history,
                    layer_fir_valid_count,
                    _,
                    _,
                ) = causal_ternary_fir_sequence(
                    lower.unsqueeze(0),
                    state.causal_fir_input_history[layer : layer + 1],
                    state.causal_fir_input_valid_count[layer : layer + 1],
                    causal_fir_taps[layer : layer + 1],
                )
                lower = adjusted_lower[0]
                final_causal_fir_input_history.append(layer_fir_history[0])
                final_causal_fir_input_valid_count.append(
                    layer_fir_valid_count[0]
                )
            associative_read = None
            if layer < self._associative_layer_count():
                if initial_associative_memory is None:
                    raise RuntimeError("associative memory state is missing")
                if initial_associative_normalizer is None:
                    raise RuntimeError("associative normalizer state is missing")
                if state.associative_keys is None and indexed_initial_key is None:
                    raise RuntimeError("associative key state is missing")
                initial_key = (
                    indexed_initial_key
                    if indexed_initial_key is not None
                    else state.associative_keys[layer]
                )
                if self._uses_semantic_count_sketch_backoff():
                    if (
                        initial_semantic_memory is None
                        or initial_semantic_normalizer is None
                        or state.associative_semantic_context is None
                        or state.associative_semantic_codes is None
                        or state.associative_context_tokens is None
                    ):
                        raise RuntimeError("semantic CountSketch state is missing")
                    (
                        associative_read,
                        layer_memory,
                        layer_normalizer,
                        layer_key,
                        final_semantic_memory,
                        final_semantic_normalizer,
                        final_semantic_context,
                        final_semantic_codes,
                    ) = self._semantic_backoff_associative_sequence(
                        lower,
                        initial_associative_memory[layer],
                        initial_associative_normalizer[layer],
                        initial_key,
                        token_ids,
                        state.associative_context_tokens,
                        initial_semantic_memory[layer],
                        initial_semantic_normalizer[layer],
                        state.associative_semantic_context,
                        state.associative_semantic_codes,
                        initial_write_enabled=state.token_count > 0,
                        semantic_enabled=semantic_enabled_sequence,
                        reset_mask=boundary_reset_mask,
                    )
                else:
                    (
                        associative_read,
                        layer_memory,
                        layer_normalizer,
                        layer_key,
                    ) = self._associative_sequence(
                        lower,
                        initial_associative_memory[layer],
                        initial_associative_normalizer[layer],
                        initial_key,
                        layer,
                        token_ids,
                        state.associative_context_tokens,
                        boundary_reset_mask,
                    )
                final_memory.append(layer_memory)
                final_normalizer.append(layer_normalizer)
                final_keys.append(layer_key)
            if return_associative_reads:
                retained_associative_reads.append(
                    associative_read
                    if associative_read is not None
                    else torch.zeros_like(lower)
                )
            gdn2_read = None
            if layer < self.config.gdn2_mixer_layers:
                if state.gdn2_memory is None:
                    raise RuntimeError("GDN2 mixer state is missing")
                normalized_lower = _rms_norm(
                    lower,
                    self.state_norm[layer, 0],
                    self.config.rms_eps,
                )
                gdn2_read, layer_gdn2_memory = self._gdn2_sequence(
                    normalized_lower,
                    state.gdn2_memory[layer],
                    layer,
                )
                final_gdn2_memory.append(layer_gdn2_memory)
            factorization_read = None
            start_layer = self._factorization_start_layer()
            if layer >= start_layer:
                if state.factorization_memory is None:
                    raise RuntimeError("factorization memory state is missing")
                memory_layer = layer - start_layer
                (
                    factorization_read,
                    layer_factorization_memory,
                ) = self._factorization_memory_sequence(
                    lower,
                    state.factorization_memory[memory_layer],
                    memory_layer,
                )
                final_factorization_memory.append(layer_factorization_memory)
            bounded_attention_read = None
            attention_start_layer = self._bounded_attention_start_layer()
            if layer >= attention_start_layer:
                if (
                    state.bounded_attention_keys is None
                    or state.bounded_attention_values is None
                ):
                    raise RuntimeError("bounded attention state is missing")
                attention_layer = layer - attention_start_layer
                (
                    bounded_attention_read,
                    layer_bounded_keys,
                    layer_bounded_values,
                ) = self._bounded_attention_sequence(
                    lower,
                    state.bounded_attention_keys[attention_layer],
                    state.bounded_attention_values[attention_layer],
                    attention_layer,
                    state.token_count,
                )
                final_bounded_attention_keys.append(layer_bounded_keys)
                final_bounded_attention_values.append(layer_bounded_values)
            feature_context_arguments = (
                {
                    "feature_context_initial": state.feature_context[layer]
                    if state.feature_context is not None
                    else None
                }
                if self._uses_recurrent_feature_context()
                else {}
            )
            transition_result = self._transition_sequence(
                lower,
                (
                    state.recurrent_layers[layer]
                    if (
                        self.config.decoupled_bounded_memory_emission
                        or self.config.causal_level_innovation_lifting
                    )
                    else state.layers[layer]
                ),
                layer,
                observation,
                associative_read,
                (
                    state.persistent_transition_adapter[layer]
                    if state.persistent_transition_adapter is not None
                    else None
                ),
                gdn2_read,
                factorization_read,
                bounded_attention_read,
                token_ids=token_ids,
                temporal_start_count=state.token_count,
                residual_route_plan=residual_route_plan,
                return_persistent_adapter_state=True,
                packed_input_projection=(
                    packed_reversible_projection[layer]
                    if packed_reversible_projection is not None
                    else None
                ),
                **feature_context_arguments,
            )
            if not isinstance(transition_result, tuple):
                raise RuntimeError("transition sequence state result is missing")
            if (
                self.config.swiglu_wide_lexical_cohead
                and layer == self.config.depth - 1
            ):
                raw, persistent_transition_adapter, swiglu_lexical_features = (
                    transition_result
                )
            elif (
                self.config.decoupled_bounded_memory_emission
                or self.config.causal_level_innovation_lifting
            ):
                raw, persistent_transition_adapter, recurrent_memory = (
                    transition_result
                )
                final_recurrent_layers.append(recurrent_memory[:, -1])
                if return_recurrent_sequences:
                    retained_recurrent_sequences.append(recurrent_memory)
            elif self._uses_recurrent_feature_context():
                raw, persistent_transition_adapter, layer_feature_context = (
                    transition_result
                )
                final_feature_context.append(layer_feature_context)
            else:
                raw, persistent_transition_adapter = transition_result
            core_sequences.append(raw)
            emitted_raw = raw
            if manifold_lower_streams is not None:
                if manifold_post is None or manifold_exchange is None:
                    raise RuntimeError("bounded manifold maps are missing")
                manifold_outputs = self._manifold_depth_bus_output(
                    manifold_lower_streams,
                    lower,
                    raw,
                    manifold_post,
                    manifold_exchange,
                )
                manifold_depth_bus_sequences.append(manifold_outputs)
                emitted_raw = manifold_outputs.mean(dim=2)
            raw_sequences.append(emitted_raw)
            if persistent_transition_adapter is not None:
                final_persistent_transition_adapter.append(
                    persistent_transition_adapter
                )
            if manifold_lower_streams is None:
                shifted_raw = torch.cat(
                    (state.layers[layer].unsqueeze(1), raw[:, :-1]),
                    dim=1,
                )
                if boundary_reset_mask is not None:
                    shifted_raw = shifted_raw.masked_fill(
                        boundary_reset_mask.unsqueeze(-1),
                        0,
                    )
                if depth_attention_sources is not None:
                    depth_attention_sources.append(shifted_raw)
                lower = shifted_raw
            else:
                if state.manifold_depth_bus is None:
                    raise RuntimeError("bounded manifold depth-bus state is missing")
                manifold_lower_streams = torch.cat(
                    (
                        state.manifold_depth_bus[layer].unsqueeze(1),
                        manifold_outputs[:, :-1],
                    ),
                    dim=1,
                )
        raw_stack = torch.stack(raw_sequences, dim=0)
        core_stack = torch.stack(core_sequences, dim=0)

        latent_prediction_errors = None
        lagged_prediction_errors = None
        settling_raw_stack = raw_stack
        if self.config.lagged_latent_error_settling_carrier:
            if state.latent_prediction_error is None:
                raise RuntimeError("latent prediction-error state is missing")
            latent_prediction_errors = stored_latent_prediction_error(
                observation,
                core_stack,
                self.generative,
            )
            lagged_prediction_errors = lagged_latent_error_sequence(
                latent_prediction_errors,
                state.latent_prediction_error,
                token_ids,
                self.config.bos_token_id,
            )
            settling_raw_stack = apply_lagged_latent_error(
                raw_stack,
                lagged_prediction_errors,
                self._latent_error_settling_gain(raw_stack.dtype),
            )

        depth, batch, sequence, width = raw_stack.shape
        flat_raw = settling_raw_stack.reshape(depth, batch * sequence, width)
        settling_observation, _ = self._state_conditioned_settling_observation(
            observation,
            core_stack[-1],
            state.layers[-1],
            initial_active=state.token_count > 0,
        )
        flat_observation = settling_observation.reshape(batch * sequence, width)
        flat_hgf_child_predicted_precision = None
        flat_hgf_posterior_precision = None
        next_hgf_posterior_precision = None
        if self.config.temporal_hgf_precision_carry:
            if state.hgf_posterior_precision is None:
                raise RuntimeError("temporal HGF posterior precision is missing")
            precision_sequence = (
                temporal_hgf_precision_sequence_with_bos_reset(
                    self.reciprocal,
                    state.hgf_posterior_precision,
                    token_ids,
                    self.config.bos_token_id,
                )
                if boundary_reset_mask is not None
                else temporal_hgf_precision_sequence(
                    self.reciprocal,
                    state.hgf_posterior_precision,
                    sequence_length=sequence,
                )
            )
            (
                hgf_child_sequence,
                _hgf_predicted_sequence,
                _hgf_evidence_sequence,
                hgf_posterior_sequence,
                next_hgf_posterior_precision,
                _hgf_diagnostics,
            ) = precision_sequence
            flat_hgf_child_predicted_precision = hgf_child_sequence.reshape(
                depth, batch * sequence, width
            )
            flat_hgf_posterior_precision = hgf_posterior_sequence.reshape(
                depth, batch * sequence, width
            )
        flat_settled = self._settle(
            flat_observation,
            flat_raw,
            hgf_child_predicted_precision=flat_hgf_child_predicted_precision,
            hgf_posterior_precision=flat_hgf_posterior_precision,
        )
        settled = flat_settled.reshape(depth, batch, sequence, width)
        final_state = NativePCState(
            layers=core_stack[:, :, -1],
            token_count=state.token_count + token_ids.shape[1],
            manifold_depth_bus=(
                torch.stack(manifold_depth_bus_sequences, dim=0)[:, :, -1]
                if manifold_depth_bus_sequences
                else None
            ),
            nonlinear_matrix_state=nonlinear_matrix_state,
            recurrent_layers=(
                torch.stack(final_recurrent_layers)
                if final_recurrent_layers
                else None
            ),
            feature_context=(
                torch.stack(final_feature_context) if final_feature_context else None
            ),
            associative_memory=(torch.stack(final_memory) if final_memory else None),
            associative_normalizer=(
                torch.stack(final_normalizer) if final_normalizer else None
            ),
            associative_keys=(
                None
                if self.config.associative_memory_addressing
                in INDEXED_COUNT_SKETCH_ADDRESSINGS
                else torch.stack(final_keys)
                if final_keys
                else None
            ),
            associative_last_update=(
                torch.full_like(
                    state.associative_last_update,
                    state.token_count + token_ids.shape[1],
                )
                if state.associative_last_update is not None
                else None
            ),
            associative_context_tokens=(
                token_ids[:, -1]
                if self.config.associative_memory_addressing
                in {
                    "token_bigram_hash",
                    "token_plus_bigram_hash",
                    "token_plus_context_bucket",
                    "indexed_lazy_multiscale_count_sketch",
                    *DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS,
                }
                else None
            ),
            associative_previous_context_tokens=(
                (
                    token_ids[:, -2]
                    if token_ids.shape[1] > 1
                    else state.associative_context_tokens
                )
                if self.config.associative_memory_addressing
                in DUAL_ORDER_COUNT_SKETCH_ADDRESSINGS
                else None
            ),
            associative_semantic_memory=(
                final_semantic_memory.unsqueeze(0)
                if final_semantic_memory is not None
                else None
            ),
            associative_semantic_normalizer=(
                final_semantic_normalizer.unsqueeze(0)
                if final_semantic_normalizer is not None
                else None
            ),
            associative_semantic_last_update=(
                torch.full_like(
                    state.associative_semantic_last_update,
                    state.token_count + token_ids.shape[1],
                )
                if state.associative_semantic_last_update is not None
                else None
            ),
            associative_semantic_context=final_semantic_context,
            associative_semantic_codes=final_semantic_codes,
            associative_pair_recurrence_bloom=final_pair_recurrence_bloom,
            persistent_transition_adapter=(
                torch.stack(final_persistent_transition_adapter)
                if final_persistent_transition_adapter
                else None
            ),
            gdn2_memory=(torch.stack(final_gdn2_memory) if final_gdn2_memory else None),
            factorization_memory=(
                torch.stack(final_factorization_memory)
                if final_factorization_memory
                else None
            ),
            bounded_attention_keys=(
                torch.stack(final_bounded_attention_keys)
                if final_bounded_attention_keys
                else None
            ),
            bounded_attention_values=(
                torch.stack(final_bounded_attention_values)
                if final_bounded_attention_values
                else None
            ),
            hgf_posterior_precision=next_hgf_posterior_precision,
            causal_fir_input_history=(
                torch.stack(final_causal_fir_input_history)
                if final_causal_fir_input_history
                else None
            ),
            causal_fir_input_valid_count=(
                torch.stack(final_causal_fir_input_valid_count)
                if final_causal_fir_input_valid_count
                else None
            ),
            latent_prediction_error=(
                latent_prediction_errors[:, :, -1]
                if latent_prediction_errors is not None
                else None
            ),
        )
        result: tuple[object, ...] = (
            observation,
            settling_raw_stack,
            settled,
            final_state,
        )
        if return_swiglu_lexical_features:
            if swiglu_lexical_features is None:
                raise RuntimeError("SwiGLU lexical feature sequence was not retained")
            result = (*result, swiglu_lexical_features)
        if return_associative_reads:
            result = (*result, torch.stack(retained_associative_reads))
        if return_recurrent_sequences:
            if self.config.lagged_latent_error_settling_carrier:
                result = (*result, core_stack)
            else:
                if len(retained_recurrent_sequences) != self.config.depth:
                    raise RuntimeError("causal recurrent sequences are incomplete")
                result = (*result, torch.stack(retained_recurrent_sequences))
        if return_lagged_latent_errors:
            if (
                latent_prediction_errors is None
                or lagged_prediction_errors is None
            ):
                raise RuntimeError("lagged latent-error sequences are missing")
            result = (
                *result,
                (latent_prediction_errors, lagged_prediction_errors),
            )
        return result

    def prefill(
        self,
        token_ids: Tensor,
        state: NativePCState | None = None,
    ) -> tuple[Tensor, NativePCState]:
        result = self._prefill_states(
            token_ids,
            state,
            return_swiglu_lexical_features=self.config.swiglu_wide_lexical_cohead,
        )
        settled = result[2]
        final_state = result[3]
        swiglu_lexical_feature = (
            result[4][:, -1]
            if self.config.swiglu_wide_lexical_cohead
            else None
        )
        return (
            self._logits(
                settled[:, :, -1],
                swiglu_lexical_feature,
                token_ids[:, -1],
            ),
            final_state,
        )

    def prefill_sequential(
        self,
        token_ids: Tensor,
        state: NativePCState | None = None,
    ) -> tuple[Tensor, NativePCState]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] <= 0:
            raise ValueError("sequence must contain at least one token")
        current = state
        logits: Tensor | None = None
        for token in token_ids.unbind(dim=1):
            logits, current = self.step(token, current)
        assert logits is not None and current is not None
        return logits, current

    def forward(
        self,
        token_ids: Tensor,
        state: NativePCState | None = None,
    ) -> Tensor:
        result = self._prefill_states(
            token_ids,
            state,
            return_swiglu_lexical_features=self.config.swiglu_wide_lexical_cohead,
        )
        settled = result[2]
        swiglu_lexical_features = (
            result[4] if self.config.swiglu_wide_lexical_cohead else None
        )
        return self._logits(settled, swiglu_lexical_features, token_ids)
