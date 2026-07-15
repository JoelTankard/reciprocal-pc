from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pfc.bounded_cp_her import BoundedCPHERCache
from pfc.bounded_cp_her import BoundedCPHERHierarchy
from pfc.bounded_cp_her import BoundedCPHERLocalResult


@dataclass(frozen=True)
class MessagePassingPCConfig:
    """Capacity-scaled predictive-coding LM with recurrent causal messages."""

    vocab_size: int
    bos_token_id: int
    seq_len: int = 1024
    hidden_dim: int = 320
    num_layers: int = 4
    num_heads: int = 4
    inference_steps: int = 4
    inference_lr: float = 0.12
    inference_lr_final: float = 0.0
    inference_lr_schedule_start_step: int = 0
    inference_lr_schedule_end_step: int = 0
    inference_optimizer: str = "gradient"
    inference_momentum: float = 0.9
    token_feedback_scale: float = 0.35
    prediction_precision: float = 1.0
    spiking_precision_scale: float = 0.0
    generative_precision: float = 0.25
    state_delta_rms_cap: float = 0.20
    attention_temperature: float = 1.0
    fast_attention_window: int = 0
    fast_attention_layers: int = 0
    route_teacher_temperature: float = 0.25
    route_update_mode: str = "cosine_teacher"
    route_lr_scale: float = 0.25
    feedback_lr_scale: float = 0.25
    generative_lr_scale: float = 0.25
    position_scale: float = 0.10
    decoder_supervision_mode: str = "all_layers"
    decoder_state_mode: str = "raw"
    readout_layer: int = 0
    readout_objective_share: float = 0.0
    winner_margin: float = 0.0
    winner_margin_scale: float = 0.0
    block_state_mode: str = "squashed_state"
    block_parameterization: str = "standard"
    residual_scale: float = 0.25
    self_init_gain: float = 0.02
    message_init_gain: float = 0.02
    ffn_dim: int = 0
    ffn_init_gain: float = 0.4
    ffn_lr_scale: float = 1.0
    ffn_learning_source: str = "phase"
    token_value_embeddings: bool = False
    token_value_scale: float = 1.0
    token_value_lr_scale: float = 0.5
    causal_encoder_decoder: bool = False
    causal_encoder_depth: int = 0
    cross_attention_lr_scale: float = 1.0
    cross_attention_gate_lr_scale: float = 1.0
    cross_attention_output_init_gain: float = 0.0
    sensory_reinjection_start: float = 0.0
    sensory_reinjection_end: float = 0.0
    sensory_reinjection_mode: str = "post_block"
    residual_stream_scale_start: float = 1.0
    residual_stream_scale_end: float = 1.0
    learn_residual_stream_scales: bool = False
    residual_stream_scale_mode: str = "layer_scalar"
    residual_stream_lr_scale: float = 1.0
    residual_stream_lr_scale_final: float = 0.0
    residual_stream_lr_schedule_start_step: int = 0
    residual_stream_lr_schedule_end_step: int = 0
    token_feedback_mode: str = "direct_layer"
    inference_mode: str = "heuristic"
    meta_pc_feedback_mode: str = "exact_vjp"
    meta_pc_reciprocal_scale: float = 0.0
    meta_pc_reciprocal_learning_rate: float = 0.0
    meta_pc_reciprocal_noise_std: float = 0.01
    meta_pc_reciprocal_update_before_settle: bool = False
    meta_pc_state_gated_reciprocal: bool = False
    meta_pc_variance_normalize_weights: bool = False
    meta_pc_relinearize_inference: bool = False
    meta_pc_cross_memory_skip_credit: bool = False
    meta_pc_cross_memory_skip_credit_scale: float = 1.0
    meta_pc_output_feedback_mode: str = "decoder_transpose"
    meta_pc_output_feedback_lr_scale: float = 1.0
    meta_pc_output_feedback_rank: int = 0
    meta_pc_output_feedback_values: bool = False
    bidirectional_free_settling: bool = False
    bidirectional_deployment_settling: bool = False
    recurrent_shared_steps: int = 1
    recurrent_step_scale: float = 1.0
    learning_phase_source: str = "settled"
    forward_update_top_layers: int = 0
    free_phase_amortization_scale: float = 0.0
    free_local_token_credit_scale: float = 0.0
    free_local_token_embedding_credit_scale: float = 0.0
    decoder_learning_source: str = "settled"
    learned_feedback_proposal_scale: float = 0.0
    residual_error_transport_mode: str = "energy_vjp"
    residual_error_suffix_decay: float = 1.0
    residual_error_target_layer: str = "top"
    residual_error_proxy_memory_scale: float = 0.0
    residual_error_proxy_memory_decay: float = 0.0
    residual_error_upper_proxy_state_scale: float = 0.0
    residual_error_upper_proxy_state_decay: float = 0.0
    residual_error_upper_content_state_scale: float = 0.0
    residual_error_upper_content_state_decay: float = 0.0
    residual_error_upper_content_write_period: int = 0
    residual_error_upper_content_write_offset: int = 0
    residual_error_upper_content_read_period: int = 0
    residual_error_upper_content_read_offset: int = 0
    residual_error_upper_content_read_delay: int = 1
    residual_error_upper_content_bind_current_scale: float = 0.0
    residual_error_upper_content_bind_product_scale: float = 0.0
    residual_error_upper_content_bind_current_delay: int = 0
    residual_phase_transport_lr_scale: float = 1.0
    residual_phase_transport_refine_passes: int = 0
    difference_target_inverse_noise: float = 0.05
    difference_target_inverse_damping: float = 0.0
    difference_target_path_mode: str = "adjacent"
    hierarchical_error_base_depth: int = 0
    hierarchical_error_binding_mode: str = "sign"
    hierarchical_error_prediction_mode: str = "gated"
    hierarchical_error_target_mode: str = "all_tokens"
    hierarchical_error_outcome_code_scale: float = 1.0
    hierarchical_error_outcome_readout_scale: float = 0.0
    hierarchical_error_top_down_scale: float = 1.0
    hierarchical_error_context_credit_scale: float = 1.0
    hierarchical_error_token_credit_scale: float = 0.0
    hierarchical_error_context_mode: str = "instant"
    hierarchical_error_gate_lr_scale: float = 1.0
    hierarchical_error_initial_write_rate: float = 0.5
    hierarchical_error_memory_scale: float = 1.0
    hierarchical_error_learning_start_step: int = 0
    hierarchical_error_level_start_interval: int = 0
    hierarchical_error_recursive_cp_rank: int = 0
    hierarchical_error_recursive_cp_upper_lr_scale: float = 1.0
    hierarchical_error_recursive_cp_component_norm_cap: float = 0.0
    hierarchical_error_recursive_cp_delayed_consolidation: bool = False
    hierarchical_error_recursive_cp_consolidation_rate: float = 0.1
    hierarchical_error_recursive_cp_min_improvement: float = 0.0
    state_prediction_separation: bool = False
    prediction_stream_window: int = 0
    sps_prediction_input_scale: float = 0.0
    sps_layer_token_credit_scale: float = 0.0
    sps_future_state_credit_scale: float = 0.0
    sps_future_observation_credit_scale: float = 0.0
    sps_node_residual_hierarchy: bool = False
    hierarchical_error_outer_product_contract: bool = False
    energy_backtracking_steps: int = 0
    matrix_optimizer: str = "adam"
    optimizer_warmup_steps: int = 0
    muon_lr_scale: float = 1.0
    muon_momentum: float = 0.95
    muon_momentum_final: float = 0.95
    muon_momentum_warmup_steps: int = 0
    muon_beta2: float = 0.9
    muon_ns_steps: int = 5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_eps: float = 1e-8
    gradient_rms_cap: float = 1.0
    update_rms_cap: float = 0.01


@dataclass
class _CrossAttentionCache:
    query_source: Tensor
    memory_input: Tensor
    memory: Tensor
    query_projection: Tensor
    key_projection: Tensor
    query_pre_norm: Tensor
    key_pre_norm: Tensor
    query: Tensor
    key: Tensor
    value: Tensor
    attention_mask: Tensor
    weights: Tensor
    message: Tensor
    output: Tensor
    gate: Tensor
    memory_is_lower_state: bool


@dataclass
class _AttentionCache:
    source_input: Tensor
    source_input_scale: float | Tensor
    direct_input_scale: float | Tensor
    source: Tensor
    query_projection: Tensor
    key_projection: Tensor
    query_pre_norm: Tensor
    key_pre_norm: Tensor
    query: Tensor
    key: Tensor
    value: Tensor
    attention_mask: Tensor
    weights: Tensor
    message: Tensor
    attention_prediction: Tensor
    attention_prediction_derivative: Tensor
    ffn_source: Tensor | None
    ffn_pre_activation: Tensor | None
    ffn_activation: Tensor | None
    prediction: Tensor
    prediction_derivative: Tensor
    token_ids: Tensor | None
    cross_attention: _CrossAttentionCache | None


@dataclass
class _InverseAttentionCache:
    source: Tensor
    query: Tensor
    key: Tensor
    value: Tensor
    attention_mask: Tensor
    weights: Tensor
    message: Tensor
    prediction: Tensor


@dataclass
class _HierarchicalContextCache:
    candidates: list[Tensor]
    memories: list[Tensor]
    contexts: list[Tensor]
    previous_contexts: list[Tensor]
    write_gates: list[Tensor]
    write_probabilities: list[Tensor]
    eligibility_features: list[Tensor]
    eligibility_biases: list[Tensor]


@dataclass
class _NodeResidualHierarchyCache:
    candidates: list[Tensor]
    pre_gate_predictions: list[Tensor]
    predictions: list[Tensor]
    memories: list[Tensor]
    previous_memories: list[Tensor]
    write_gates: list[Tensor]
    write_probabilities: list[Tensor]
    gate_features: list[Tensor]


def _rms_norm(value: Tensor, eps: float = 1e-6) -> Tensor:
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)


def _rms_norm_vjp(value: Tensor, output_signal: Tensor, eps: float = 1e-6) -> Tensor:
    inverse_rms = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    projected = (output_signal * value).mean(dim=-1, keepdim=True)
    return output_signal * inverse_rms - value * inverse_rms.pow(3) * projected


def _rms(value: Tensor) -> Tensor:
    if value.numel() == 0:
        return value.new_zeros(())
    return value.float().square().mean().sqrt()


def _cap_token_rms(value: Tensor, cap: float) -> Tensor:
    if cap <= 0:
        return value
    mean_square = value.float().square().mean(dim=-1, keepdim=True)
    rms = mean_square.clamp_min(cap * cap).sqrt()
    return value * (cap / rms).to(value.dtype)


def _cap_token_rms_vjp(
    value: Tensor,
    output_signal: Tensor,
    cap: float,
) -> Tensor:
    if cap <= 0:
        return output_signal
    value_float = value.float()
    signal_float = output_signal.float()
    mean_square = value_float.square().mean(dim=-1, keepdim=True)
    active = mean_square > cap * cap
    rms = mean_square.clamp_min(cap * cap).sqrt()
    scale = cap / rms
    radial = (
        value_float * signal_float
    ).mean(dim=-1, keepdim=True) / mean_square.clamp_min(cap * cap)
    capped_signal = scale * (signal_float - value_float * radial)
    return torch.where(active, capped_signal, signal_float).to(
        output_signal.dtype
    )


def _sinusoidal_positions(length: int, width: int) -> Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10_000.0) / width)
    )
    encoding = torch.zeros(length, width, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * frequency)
    if width > 1:
        encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
    return encoding


def _rotary_positions(
    length: int,
    head_dim: int,
    *,
    base: float = 100_000.0,
) -> tuple[Tensor, Tensor]:
    channel = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inverse_frequency = torch.pow(base, -channel / head_dim)
    frequency = torch.outer(torch.arange(length, dtype=torch.float32), inverse_frequency)
    return (
        frequency.cos().unsqueeze(0).unsqueeze(0),
        frequency.sin().unsqueeze(0).unsqueeze(0),
    )


def _apply_rotary(value: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    half = value.shape[-1] // 2
    first, second = value[..., :half], value[..., half:]
    return torch.cat(
        (
            first * cosine + second * sine,
            first * -sine + second * cosine,
        ),
        dim=-1,
    )


def _apply_rotary_vjp(signal: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    half = signal.shape[-1] // 2
    first, second = signal[..., :half], signal[..., half:]
    return torch.cat(
        (
            first * cosine - second * sine,
            first * sine + second * cosine,
        ),
        dim=-1,
    )


_POLAR_EXPRESS_COEFFICIENTS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def _polar_orthogonalize(value: Tensor, steps: int) -> Tensor:
    compute = value.bfloat16() if value.is_cuda else value.float()
    target_row_norm = compute.float().norm() / math.sqrt(compute.shape[-2])
    row_norm = compute.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    compute = compute * (target_row_norm / row_norm).to(compute.dtype)
    compute = compute / (compute.norm() * 1.01 + 1e-6)
    if compute.shape[-2] > compute.shape[-1]:
        for a, b, c in _POLAR_EXPRESS_COEFFICIENTS[:steps]:
            gram = compute.mT @ compute
            correction = b * gram + c * (gram @ gram)
            compute = a * compute + compute @ correction
    else:
        for a, b, c in _POLAR_EXPRESS_COEFFICIENTS[:steps]:
            gram = compute @ compute.mT
            correction = b * gram + c * (gram @ gram)
            compute = a * compute + correction @ compute
    direction = compute.to(value.dtype)
    target_norm = math.sqrt(min(direction.shape[-2], direction.shape[-1]))
    return direction * (target_norm / direction.float().norm().clamp_min(1e-6)).to(
        direction.dtype
    )


class MessagePassingPCLanguageModel(nn.Module):
    """Layered PC network trained with explicit local errors, never parameter backprop."""

    def __init__(self, config: MessagePassingPCConfig) -> None:
        super().__init__()
        if config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if config.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if config.num_heads <= 0 or config.hidden_dim % config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if config.inference_steps <= 0:
            raise ValueError("inference_steps must be positive")
        if config.inference_lr <= 0:
            raise ValueError("inference_lr must be positive")
        if config.inference_lr_final < 0:
            raise ValueError("inference_lr_final must be nonnegative")
        if (
            config.inference_lr_schedule_start_step < 0
            or config.inference_lr_schedule_end_step < 0
        ):
            raise ValueError("inference LR schedule steps must be nonnegative")
        if (
            config.inference_lr_final > 0
            and config.inference_lr_schedule_end_step
            <= config.inference_lr_schedule_start_step
        ):
            raise ValueError("inference LR schedule end must exceed start")
        if config.inference_optimizer not in {
            "gradient",
            "heavy_ball",
            "conjugate_gradient",
        }:
            raise ValueError(
                "inference_optimizer must be gradient, heavy_ball, or "
                "conjugate_gradient"
            )
        if not 0 <= config.inference_momentum < 1:
            raise ValueError("inference_momentum must be in [0, 1)")
        if config.spiking_precision_scale < 0:
            raise ValueError("spiking_precision_scale must be nonnegative")
        if (
            config.spiking_precision_scale > 0
            and config.inference_optimizer != "gradient"
        ):
            raise ValueError("spiking precision requires gradient inference")
        if config.attention_temperature <= 0 or config.route_teacher_temperature <= 0:
            raise ValueError("attention temperatures must be positive")
        if config.fast_attention_window < 0 or config.fast_attention_layers < 0:
            raise ValueError("fast attention settings must be nonnegative")
        if (config.fast_attention_window == 0) != (
            config.fast_attention_layers == 0
        ):
            raise ValueError(
                "fast attention window and layer count must both be zero or positive"
            )
        if config.fast_attention_layers > config.num_layers:
            raise ValueError("fast attention layers cannot exceed num_layers")
        if config.fast_attention_window > 0 and config.state_prediction_separation:
            raise ValueError(
                "fast attention windows do not support state_prediction_separation"
            )
        if config.route_update_mode not in {
            "cosine_teacher",
            "local_energy",
            "token_error_teacher",
        }:
            raise ValueError(
                "route_update_mode must be cosine_teacher, local_energy, "
                "or token_error_teacher"
            )
        if config.decoder_supervision_mode not in {"all_layers", "top_only"}:
            raise ValueError("decoder_supervision_mode must be all_layers or top_only")
        if config.decoder_state_mode not in {"raw", "rms_norm"}:
            raise ValueError("decoder_state_mode must be raw or rms_norm")
        if config.readout_layer < 0 or config.readout_layer > config.num_layers:
            raise ValueError("readout_layer must be between 0 and num_layers")
        if not 0.0 <= config.readout_objective_share <= 1.0:
            raise ValueError("readout_objective_share must be in [0, 1]")
        if config.winner_margin < 0 or config.winner_margin_scale < 0:
            raise ValueError("winner margin settings must be nonnegative")
        if config.block_state_mode not in {
            "squashed_state",
            "identity_residual",
            "nanochat_residual",
        }:
            raise ValueError(
                "block_state_mode must be squashed_state, identity_residual, "
                "or nanochat_residual"
            )
        if (
            config.block_state_mode == "nanochat_residual"
            and config.hidden_dim // config.num_heads % 2 != 0
        ):
            raise ValueError("nanochat_residual requires an even head dimension")
        if config.block_parameterization not in {"standard", "depth_mup"}:
            raise ValueError(
                "block_parameterization must be standard or depth_mup"
            )
        if (
            config.block_parameterization == "depth_mup"
            and config.block_state_mode != "identity_residual"
        ):
            raise ValueError(
                "depth_mup block parameterization requires identity_residual"
            )
        if config.block_parameterization == "depth_mup" and config.ffn_dim > 0:
            raise ValueError(
                "depth_mup block parameterization does not yet support FFNs"
            )
        if config.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if config.self_init_gain < 0 or config.message_init_gain < 0:
            raise ValueError("block initialization gains must be nonnegative")
        if config.ffn_dim < 0:
            raise ValueError("ffn_dim must be nonnegative")
        if config.ffn_init_gain < 0:
            raise ValueError("ffn_init_gain must be nonnegative")
        if config.ffn_lr_scale < 0:
            raise ValueError("ffn_lr_scale must be nonnegative")
        if config.ffn_learning_source not in {"phase", "free_to_settled"}:
            raise ValueError(
                "ffn_learning_source must be phase or free_to_settled"
            )
        if config.token_value_scale < 0 or config.token_value_lr_scale < 0:
            raise ValueError("token value scales must be nonnegative")
        if (
            config.cross_attention_lr_scale < 0
            or config.cross_attention_gate_lr_scale < 0
        ):
            raise ValueError("cross-attention LR scales must be nonnegative")
        if config.cross_attention_output_init_gain < 0:
            raise ValueError(
                "cross_attention_output_init_gain must be nonnegative"
            )
        if config.causal_encoder_depth < 0:
            raise ValueError("causal_encoder_depth must be nonnegative")
        if not config.causal_encoder_decoder and config.causal_encoder_depth != 0:
            raise ValueError(
                "causal_encoder_depth requires causal encoder-decoder"
            )
        if (
            config.causal_encoder_decoder
            and config.causal_encoder_depth >= config.num_layers
        ):
            raise ValueError(
                "causal_encoder_depth must be smaller than num_layers"
            )
        if config.causal_encoder_decoder and (
            config.block_state_mode != "nanochat_residual"
            or config.inference_mode != "meta_pc"
            or config.state_prediction_separation
        ):
            raise ValueError(
                "causal encoder-decoder currently requires nanochat_residual "
                "Meta-PC without state prediction separation"
            )
        if config.meta_pc_cross_memory_skip_credit and (
            not config.causal_encoder_decoder
            or config.causal_encoder_depth <= 0
            or config.causal_encoder_depth + 1 >= config.num_layers
            or config.meta_pc_feedback_mode != "exact_vjp"
        ):
            raise ValueError(
                "meta_pc_cross_memory_skip_credit requires exact-vjp causal "
                "encoder-decoder with a contextual encoder and at least two "
                "decoder layers"
            )
        if config.meta_pc_cross_memory_skip_credit_scale < 0:
            raise ValueError(
                "meta_pc_cross_memory_skip_credit_scale must be nonnegative"
            )
        if (
            config.meta_pc_relinearize_inference
            and config.inference_mode != "meta_pc"
        ):
            raise ValueError(
                "meta_pc_relinearize_inference requires meta_pc inference"
            )
        if (
            config.sensory_reinjection_start < 0
            or config.sensory_reinjection_end < 0
        ):
            raise ValueError("sensory reinjection scales must be nonnegative")
        if config.sensory_reinjection_mode not in {"post_block", "pre_block"}:
            raise ValueError(
                "sensory_reinjection_mode must be post_block or pre_block"
            )
        if (
            config.residual_stream_scale_start <= 0
            or config.residual_stream_scale_end <= 0
        ):
            raise ValueError("residual stream scales must be positive")
        if (
            config.residual_stream_lr_scale < 0
            or config.residual_stream_lr_scale_final < 0
        ):
            raise ValueError("residual stream LR scales must be nonnegative")
        if (
            config.residual_stream_lr_schedule_start_step < 0
            or config.residual_stream_lr_schedule_end_step < 0
        ):
            raise ValueError("residual stream LR schedule steps must be nonnegative")
        if (
            config.residual_stream_lr_scale_final > 0
            and config.residual_stream_lr_schedule_end_step
            <= config.residual_stream_lr_schedule_start_step
        ):
            raise ValueError(
                "residual stream LR schedule end must exceed start"
            )
        if config.residual_stream_scale_mode not in {
            "layer_scalar",
            "channel",
        }:
            raise ValueError(
                "residual_stream_scale_mode must be layer_scalar or channel"
            )
        if (
            config.residual_stream_scale_mode == "channel"
            and not config.learn_residual_stream_scales
        ):
            raise ValueError(
                "channel residual stream scales must be learned"
            )
        if config.learn_residual_stream_scales and (
            config.block_state_mode != "nanochat_residual"
            or config.inference_mode != "meta_pc"
        ):
            raise ValueError(
                "learned residual stream scales require nanochat Meta-PC"
            )
        if config.token_value_embeddings and (
            config.block_state_mode != "nanochat_residual"
            or config.inference_mode != "meta_pc"
            or config.state_prediction_separation
        ):
            raise ValueError(
                "token value embeddings currently require nanochat_residual "
                "Meta-PC without state prediction separation"
            )
        if (
            config.sensory_reinjection_start > 0
            or config.sensory_reinjection_end > 0
        ) and (
            config.block_state_mode != "nanochat_residual"
            or config.inference_mode != "meta_pc"
            or config.state_prediction_separation
        ):
            raise ValueError(
                "sensory reinjection currently requires nanochat_residual "
                "Meta-PC without state prediction separation"
            )
        if config.token_feedback_mode not in {"direct_layer", "top_propagated"}:
            raise ValueError("token_feedback_mode must be direct_layer or top_propagated")
        if config.inference_mode not in {
            "heuristic",
            "pc_energy",
            "meta_pc",
            "bidirectional_pc",
            "residual_error_pc",
            "her_error_hierarchy",
            "difference_target",
            "recurrent_local",
            "layerwise_local",
        }:
            raise ValueError(
                "inference_mode must be heuristic, pc_energy, "
                "meta_pc, bidirectional_pc, residual_error_pc, "
                "her_error_hierarchy, difference_target, recurrent_local, "
                "or layerwise_local"
            )
        if config.meta_pc_feedback_mode not in {
            "exact_vjp",
            "reciprocal_attention",
            "learned_reciprocal",
            "stage_reciprocal",
        }:
            raise ValueError(
                "meta_pc_feedback_mode must be exact_vjp, "
                "reciprocal_attention, learned_reciprocal, or "
                "stage_reciprocal"
            )
        if config.meta_pc_reciprocal_scale < 0:
            raise ValueError("meta_pc_reciprocal_scale must be nonnegative")
        if not 0 <= config.meta_pc_reciprocal_learning_rate <= 1:
            raise ValueError(
                "meta_pc_reciprocal_learning_rate must be in [0, 1]"
            )
        if config.meta_pc_reciprocal_noise_std <= 0:
            raise ValueError("meta_pc_reciprocal_noise_std must be positive")
        if (
            config.meta_pc_feedback_mode in {
                "learned_reciprocal",
                "stage_reciprocal",
            }
            and config.meta_pc_reciprocal_learning_rate <= 0
        ):
            raise ValueError(
                "learned reciprocal feedback requires a positive learning rate"
            )
        if (
            config.meta_pc_reciprocal_update_before_settle
            and config.meta_pc_feedback_mode not in {
                "learned_reciprocal",
                "stage_reciprocal",
            }
        ):
            raise ValueError(
                "pre-settle reciprocal update requires learned reciprocal feedback"
            )
        if (
            config.meta_pc_state_gated_reciprocal
            and config.meta_pc_feedback_mode != "learned_reciprocal"
        ):
            raise ValueError(
                "state-gated reciprocal feedback requires learned_reciprocal feedback"
            )
        if (
            config.meta_pc_feedback_mode == "stage_reciprocal"
            and config.block_state_mode != "nanochat_residual"
        ):
            raise ValueError(
                "stage_reciprocal feedback requires nanochat_residual blocks"
            )
        if (
            config.meta_pc_feedback_mode == "stage_reciprocal"
            and config.causal_encoder_decoder
        ):
            raise ValueError(
                "stage_reciprocal feedback does not support cross-attention"
            )
        if config.meta_pc_output_feedback_mode not in {
            "decoder_transpose",
            "fixed_random",
            "kp_learned",
            "kp_factorized",
        }:
            raise ValueError(
                "meta_pc_output_feedback_mode must be decoder_transpose, "
                "fixed_random, kp_learned, or kp_factorized"
            )
        if config.meta_pc_output_feedback_lr_scale < 0:
            raise ValueError(
                "meta_pc_output_feedback_lr_scale must be nonnegative"
            )
        if (
            config.meta_pc_output_feedback_mode != "decoder_transpose"
            and config.inference_mode
            not in {"meta_pc", "residual_error_pc"}
        ):
            raise ValueError(
                "independent output feedback requires meta_pc or "
                "residual_error_pc mode"
            )
        if config.meta_pc_output_feedback_mode == "kp_factorized":
            if not 0 < config.meta_pc_output_feedback_rank <= config.hidden_dim:
                raise ValueError(
                    "factorized Meta-PC output feedback rank must be in "
                    "[1, hidden_dim]"
                )
        elif config.meta_pc_output_feedback_rank != 0:
            raise ValueError(
                "meta_pc_output_feedback_rank requires kp_factorized mode"
            )
        if config.meta_pc_output_feedback_values and (
            config.meta_pc_output_feedback_mode != "kp_learned"
            or config.token_value_embeddings
        ):
            raise ValueError(
                "shared output-feedback values require kp_learned feedback "
                "without separate token value embeddings"
            )
        if config.recurrent_shared_steps < 1:
            raise ValueError("recurrent_shared_steps must be positive")
        if not 0 < config.recurrent_step_scale <= 1:
            raise ValueError("recurrent_step_scale must be in (0, 1]")
        if (
            config.inference_mode == "recurrent_local"
            and config.num_layers != 1
        ):
            raise ValueError("recurrent_local requires exactly one block")
        if config.inference_mode in {"recurrent_local", "layerwise_local"} and (
            config.block_state_mode != "nanochat_residual"
            or config.decoder_supervision_mode != "top_only"
            or config.readout_layer not in {0, config.num_layers}
            or config.state_prediction_separation
        ):
            raise ValueError(
                "local outcome modes require nanochat_residual blocks, "
                "top_only supervision, top readout, and no state separation"
            )
        if (
            config.inference_mode == "difference_target"
            and config.decoder_supervision_mode != "top_only"
        ):
            raise ValueError(
                "difference_target requires top_only decoder supervision"
            )
        if (
            config.inference_mode == "difference_target"
            and config.readout_layer not in {0, config.num_layers}
        ):
            raise ValueError("difference_target requires the top readout layer")
        if config.inference_mode == "meta_pc" and (
            config.decoder_supervision_mode != "top_only"
            or config.readout_layer not in {0, config.num_layers}
            or config.state_prediction_separation
        ):
            raise ValueError(
                "meta_pc requires top_only decoder supervision, top readout, "
                "and no state separation"
            )
        if config.inference_mode == "meta_pc" and (
            config.learning_phase_source != "settled"
            or config.forward_update_top_layers != config.num_layers
        ):
            raise ValueError(
                "meta_pc requires settled learning and Forward Update on "
                "every layer"
            )
        if config.inference_mode in {
            "bidirectional_pc",
            "residual_error_pc",
        } and (
            config.block_state_mode != "identity_residual"
            or config.decoder_supervision_mode != "top_only"
            or config.readout_layer not in {0, config.num_layers}
            or config.state_prediction_separation
        ):
            raise ValueError(
                "bidirectional and residual-error PC require identity_residual "
                "blocks, top_only supervision, top readout, and no state "
                "separation"
            )
        if config.bidirectional_deployment_settling and (
            config.inference_mode != "bidirectional_pc"
            or not config.bidirectional_free_settling
        ):
            raise ValueError(
                "bidirectional deployment settling requires bidirectional_pc "
                "and free settling"
            )
        if (
            config.inference_mode == "residual_error_pc"
            and config.hierarchical_error_base_depth != 0
        ):
            raise ValueError("residual_error_pc requires base depth 0")
        if config.learning_phase_source not in {"settled", "free", "dual"}:
            raise ValueError("learning_phase_source must be settled, free, or dual")
        if not 0 <= config.forward_update_top_layers <= config.num_layers:
            raise ValueError(
                "forward_update_top_layers must be between 0 and num_layers"
            )
        if (
            config.forward_update_top_layers > 0
            and config.learning_phase_source != "settled"
        ):
            raise ValueError(
                "forward_update_top_layers requires settled learning_phase_source"
            )
        if config.free_phase_amortization_scale < 0:
            raise ValueError("free_phase_amortization_scale must be nonnegative")
        if config.free_local_token_credit_scale < 0:
            raise ValueError("free_local_token_credit_scale must be nonnegative")
        if config.free_local_token_embedding_credit_scale < 0:
            raise ValueError(
                "free_local_token_embedding_credit_scale must be nonnegative"
            )
        if (
            config.free_phase_amortization_scale > 0
            and config.learning_phase_source != "settled"
        ):
            raise ValueError(
                "free_phase_amortization_scale requires settled learning_phase_source"
            )
        if config.decoder_learning_source not in {"settled", "free"}:
            raise ValueError("decoder_learning_source must be settled or free")
        if config.learned_feedback_proposal_scale < 0:
            raise ValueError("learned_feedback_proposal_scale must be nonnegative")
        if config.residual_error_transport_mode not in {
            "energy_vjp",
            "learned_only",
        }:
            raise ValueError(
                "residual_error_transport_mode must be energy_vjp or learned_only"
            )
        if config.residual_error_suffix_decay < 0:
            raise ValueError("residual_error_suffix_decay must be nonnegative")
        if config.residual_error_target_layer not in {"top", "base"}:
            raise ValueError("residual_error_target_layer must be top or base")
        if config.residual_error_proxy_memory_scale < 0:
            raise ValueError(
                "residual_error_proxy_memory_scale must be nonnegative"
            )
        if not 0 <= config.residual_error_proxy_memory_decay <= 1:
            raise ValueError(
                "residual_error_proxy_memory_decay must be in [0, 1]"
            )
        if config.residual_error_upper_proxy_state_scale < 0:
            raise ValueError(
                "residual_error_upper_proxy_state_scale must be nonnegative"
            )
        if not 0 <= config.residual_error_upper_proxy_state_decay <= 1:
            raise ValueError(
                "residual_error_upper_proxy_state_decay must be in [0, 1]"
            )
        if config.residual_error_upper_content_state_scale < 0:
            raise ValueError(
                "residual_error_upper_content_state_scale must be nonnegative"
            )
        if not 0 <= config.residual_error_upper_content_state_decay <= 1:
            raise ValueError(
                "residual_error_upper_content_state_decay must be in [0, 1]"
            )
        if config.residual_error_upper_content_write_period < 0:
            raise ValueError(
                "residual_error_upper_content_write_period must be nonnegative"
            )
        if config.residual_error_upper_content_read_period < 0:
            raise ValueError(
                "residual_error_upper_content_read_period must be nonnegative"
            )
        if config.residual_error_upper_content_read_delay < 1:
            raise ValueError(
                "residual_error_upper_content_read_delay must be positive"
            )
        if config.residual_error_upper_content_bind_current_scale < 0:
            raise ValueError(
                "residual_error_upper_content_bind_current_scale must be nonnegative"
            )
        if config.residual_error_upper_content_bind_product_scale < 0:
            raise ValueError(
                "residual_error_upper_content_bind_product_scale must be nonnegative"
            )
        if config.residual_error_upper_content_bind_current_delay < 0:
            raise ValueError(
                "residual_error_upper_content_bind_current_delay must be nonnegative"
            )
        if config.residual_error_upper_content_write_period == 0:
            if config.residual_error_upper_content_write_offset != 0:
                raise ValueError(
                    "residual_error_upper_content_write_offset requires write_period"
                )
        elif not (
            0
            <= config.residual_error_upper_content_write_offset
            < config.residual_error_upper_content_write_period
        ):
            raise ValueError(
                "residual_error_upper_content_write_offset must be in [0, write_period)"
            )
        if config.residual_error_upper_content_read_period == 0:
            if config.residual_error_upper_content_read_offset != 0:
                raise ValueError(
                    "residual_error_upper_content_read_offset requires read_period"
                )
        elif not (
            0
            <= config.residual_error_upper_content_read_offset
            < config.residual_error_upper_content_read_period
        ):
            raise ValueError(
                "residual_error_upper_content_read_offset must be in [0, read_period)"
            )
        if config.residual_phase_transport_lr_scale < 0:
            raise ValueError(
                "residual_phase_transport_lr_scale must be nonnegative"
            )
        if config.residual_phase_transport_refine_passes < 0:
            raise ValueError(
                "residual_phase_transport_refine_passes must be nonnegative"
            )
        if config.difference_target_inverse_noise <= 0:
            raise ValueError(
                "difference_target_inverse_noise must be positive"
            )
        if config.difference_target_inverse_damping < 0:
            raise ValueError(
                "difference_target_inverse_damping must be nonnegative"
            )
        if config.difference_target_path_mode not in {"adjacent", "direct"}:
            raise ValueError(
                "difference_target_path_mode must be adjacent or direct"
            )
        if (
            config.hierarchical_error_base_depth < 0
            or config.hierarchical_error_base_depth >= config.num_layers
        ):
            raise ValueError(
                "hierarchical_error_base_depth must be between 0 and "
                "num_layers - 1"
            )
        if config.hierarchical_error_binding_mode not in {
            "factorized",
            "sign",
            "identity",
            "unitary",
        }:
            raise ValueError(
                "hierarchical_error_binding_mode must be factorized, sign, "
                "identity, or unitary"
            )
        if config.hierarchical_error_prediction_mode not in {
            "gated",
            "direct",
            "direct_transport",
            "direct_phase_transport",
            "recursive_cp",
        }:
            raise ValueError(
                "hierarchical_error_prediction_mode must be gated, direct, "
                "direct_transport, direct_phase_transport, or recursive_cp"
            )
        direct_error_predictor = (
            config.inference_mode == "her_error_hierarchy"
            and config.hierarchical_error_binding_mode == "factorized"
        ) or config.inference_mode == "residual_error_pc"
        if (
            config.hierarchical_error_prediction_mode
            in {"direct", "direct_transport", "direct_phase_transport"}
            and not direct_error_predictor
        ):
            raise ValueError(
                "direct error prediction requires factorized "
                "her_error_hierarchy or residual_error_pc"
            )
        if (
            config.hierarchical_error_prediction_mode
            in {"direct_transport", "direct_phase_transport"}
            and config.inference_mode != "residual_error_pc"
        ):
            raise ValueError(
                "direct transport error prediction requires residual_error_pc"
            )
        if config.residual_error_transport_mode == "learned_only" and (
            config.inference_mode != "residual_error_pc"
            or config.hierarchical_error_prediction_mode
            not in {"direct_transport", "direct_phase_transport"}
            or config.learned_feedback_proposal_scale <= 0
        ):
            raise ValueError(
                "learned_only residual error transport requires "
                "residual_error_pc direct transport with positive learned "
                "feedback proposal scale"
            )
        if config.hierarchical_error_recursive_cp_rank < 0:
            raise ValueError(
                "hierarchical_error_recursive_cp_rank must be nonnegative"
            )
        if config.hierarchical_error_recursive_cp_upper_lr_scale < 0:
            raise ValueError(
                "hierarchical_error_recursive_cp_upper_lr_scale must be "
                "nonnegative"
            )
        if config.hierarchical_error_recursive_cp_component_norm_cap < 0:
            raise ValueError(
                "hierarchical_error_recursive_cp_component_norm_cap must be "
                "nonnegative"
            )
        if not 0 < config.hierarchical_error_recursive_cp_consolidation_rate <= 1:
            raise ValueError(
                "hierarchical_error_recursive_cp_consolidation_rate must be "
                "in (0, 1]"
            )
        if config.hierarchical_error_recursive_cp_min_improvement < 0:
            raise ValueError(
                "hierarchical_error_recursive_cp_min_improvement must be "
                "nonnegative"
            )
        recursive_cp = config.hierarchical_error_recursive_cp_rank > 0
        if recursive_cp != (
            config.hierarchical_error_prediction_mode == "recursive_cp"
        ):
            raise ValueError(
                "recursive_cp prediction mode requires a positive recursive "
                "CP rank, and recursive CP rank requires that mode"
            )
        if (
            config.hierarchical_error_recursive_cp_component_norm_cap > 0
            and not recursive_cp
        ):
            raise ValueError(
                "recursive CP component norm cap requires recursive_cp mode"
            )
        if config.hierarchical_error_recursive_cp_delayed_consolidation and (
            not recursive_cp
            or config.hierarchical_error_recursive_cp_component_norm_cap <= 0
        ):
            raise ValueError(
                "delayed recursive CP consolidation requires recursive_cp mode "
                "and a positive component norm cap"
            )
        if recursive_cp and (
            config.inference_mode != "her_error_hierarchy"
            or config.hierarchical_error_binding_mode != "factorized"
            or config.hierarchical_error_target_mode != "fixed_outcome"
            or config.hierarchical_error_outer_product_contract
            or config.sps_node_residual_hierarchy
            or config.state_prediction_separation
            or config.hierarchical_error_token_credit_scale > 0
            or config.hierarchical_error_outcome_readout_scale > 0
            or not math.isclose(
                config.hierarchical_error_top_down_scale,
                1.0,
            )
        ):
            raise ValueError(
                "recursive CP HER requires factorized fixed-outcome "
                "hierarchy, unit top-down scale, and no SPS, outer-product, "
                "token-credit, or outcome-readout side path"
            )
        if config.hierarchical_error_target_mode not in {
            "all_tokens",
            "fixed_outcome",
            "mistakes",
        }:
            raise ValueError(
                "hierarchical_error_target_mode must be all_tokens, "
                "fixed_outcome, or mistakes"
            )
        if config.hierarchical_error_outcome_code_scale < 0:
            raise ValueError(
                "hierarchical_error_outcome_code_scale must be nonnegative"
            )
        if config.hierarchical_error_outcome_readout_scale < 0:
            raise ValueError(
                "hierarchical_error_outcome_readout_scale must be nonnegative"
            )
        if config.hierarchical_error_outcome_readout_scale > 0 and (
            config.inference_mode != "her_error_hierarchy"
            or config.hierarchical_error_binding_mode != "factorized"
            or config.hierarchical_error_target_mode != "fixed_outcome"
        ):
            raise ValueError(
                "hierarchical_error_outcome_readout_scale requires "
                "factorized fixed-outcome her_error_hierarchy"
            )
        if config.hierarchical_error_top_down_scale < 0:
            raise ValueError(
                "hierarchical_error_top_down_scale must be nonnegative"
            )
        if config.hierarchical_error_context_credit_scale < 0:
            raise ValueError(
                "hierarchical_error_context_credit_scale must be nonnegative"
            )
        if config.hierarchical_error_token_credit_scale < 0:
            raise ValueError(
                "hierarchical_error_token_credit_scale must be nonnegative"
            )
        if config.hierarchical_error_token_credit_scale > 0 and (
            config.inference_mode != "her_error_hierarchy"
            or config.hierarchical_error_binding_mode != "factorized"
        ):
            raise ValueError(
                "hierarchical_error_token_credit_scale requires factorized "
                "her_error_hierarchy"
            )
        if config.hierarchical_error_context_mode not in {
            "instant",
            "persistent",
            "persistent_hard",
        }:
            raise ValueError(
                "hierarchical_error_context_mode must be instant, persistent, "
                "or persistent_hard"
            )
        if config.hierarchical_error_gate_lr_scale < 0:
            raise ValueError(
                "hierarchical_error_gate_lr_scale must be nonnegative"
            )
        if not 0 < config.hierarchical_error_initial_write_rate < 1:
            raise ValueError(
                "hierarchical_error_initial_write_rate must be in (0, 1)"
            )
        if not 0 <= config.hierarchical_error_memory_scale <= 1:
            raise ValueError(
                "hierarchical_error_memory_scale must be in [0, 1]"
            )
        if config.hierarchical_error_learning_start_step < 0:
            raise ValueError(
                "hierarchical_error_learning_start_step must be nonnegative"
            )
        if config.hierarchical_error_level_start_interval < 0:
            raise ValueError(
                "hierarchical_error_level_start_interval must be nonnegative"
            )
        if config.prediction_stream_window < 0:
            raise ValueError("prediction_stream_window must be nonnegative")
        if config.sps_prediction_input_scale < 0:
            raise ValueError("sps_prediction_input_scale must be nonnegative")
        if config.sps_layer_token_credit_scale < 0:
            raise ValueError("sps_layer_token_credit_scale must be nonnegative")
        if config.sps_future_state_credit_scale < 0:
            raise ValueError("sps_future_state_credit_scale must be nonnegative")
        if config.sps_future_observation_credit_scale < 0:
            raise ValueError(
                "sps_future_observation_credit_scale must be nonnegative"
            )
        if config.state_prediction_separation and (
            config.inference_mode != "her_error_hierarchy"
            or config.hierarchical_error_binding_mode != "factorized"
        ):
            raise ValueError(
                "state_prediction_separation requires factorized "
                "her_error_hierarchy"
            )
        if not config.state_prediction_separation and (
            config.sps_prediction_input_scale > 0
            or config.sps_layer_token_credit_scale > 0
            or config.sps_future_state_credit_scale > 0
            or config.sps_future_observation_credit_scale > 0
            or config.sps_node_residual_hierarchy
        ):
            raise ValueError("SPS credit and input scales require state separation")
        if config.sps_node_residual_hierarchy and (
            config.inference_mode != "her_error_hierarchy"
            or config.hierarchical_error_binding_mode != "factorized"
        ):
            raise ValueError(
                "sps_node_residual_hierarchy requires factorized "
                "her_error_hierarchy"
            )
        if (
            config.sps_node_residual_hierarchy
            and config.hierarchical_error_base_depth < 1
        ):
            raise ValueError(
                "sps_node_residual_hierarchy requires base depth >= 1"
            )
        if config.sps_node_residual_hierarchy and not math.isclose(
            config.hierarchical_error_top_down_scale,
            1.0,
        ):
            raise ValueError(
                "sps_node_residual_hierarchy requires top-down scale 1"
            )
        if config.sps_node_residual_hierarchy and (
            config.sps_layer_token_credit_scale > 0
            or config.sps_future_state_credit_scale > 0
            or config.sps_future_observation_credit_scale > 0
            or config.hierarchical_error_token_credit_scale > 0
        ):
            raise ValueError(
                "sps_node_residual_hierarchy replaces auxiliary SPS and HER "
                "token credit"
            )
        if config.hierarchical_error_outer_product_contract and (
            config.inference_mode != "her_error_hierarchy"
            or config.hierarchical_error_binding_mode != "factorized"
        ):
            raise ValueError(
                "hierarchical_error_outer_product_contract requires "
                "factorized her_error_hierarchy"
            )
        if (
            config.hierarchical_error_outer_product_contract
            and config.hierarchical_error_prediction_mode != "gated"
        ):
            raise ValueError(
                "hierarchical_error_outer_product_contract requires gated "
                "prediction mode"
            )
        if config.hierarchical_error_outer_product_contract and (
            config.num_layers - config.hierarchical_error_base_depth != 1
        ):
            raise ValueError(
                "hierarchical_error_outer_product_contract requires exactly "
                "one hierarchical error level"
            )
        if config.hierarchical_error_outer_product_contract and (
            config.sps_node_residual_hierarchy
            or config.hierarchical_error_token_credit_scale > 0
            or config.hierarchical_error_target_mode != "all_tokens"
            or not math.isclose(config.hierarchical_error_top_down_scale, 1.0)
        ):
            raise ValueError(
                "hierarchical_error_outer_product_contract requires all-token "
                "targets, unit top-down scale, and no node or token-credit path"
            )
        if (
            config.hierarchical_error_context_mode != "instant"
            and config.inference_mode not in {
                "her_error_hierarchy",
                "residual_error_pc",
            }
        ):
            raise ValueError(
                "persistent hierarchical context requires her_error_hierarchy or "
                "residual_error_pc"
            )
        if (
            config.inference_optimizer == "conjugate_gradient"
            and config.learned_feedback_proposal_scale > 0
        ):
            raise ValueError(
                "conjugate-gradient inference requires exact energy gradients"
            )
        if config.energy_backtracking_steps < 0:
            raise ValueError("energy_backtracking_steps must be nonnegative")
        if (
            config.inference_optimizer != "gradient"
            and config.energy_backtracking_steps <= 0
        ):
            raise ValueError(
                "accelerated inference requires energy backtracking"
            )
        if config.matrix_optimizer not in {"adam", "muon"}:
            raise ValueError("matrix_optimizer must be adam or muon")
        if config.optimizer_warmup_steps < 0:
            raise ValueError("optimizer_warmup_steps must be nonnegative")
        if config.muon_lr_scale < 0:
            raise ValueError("muon_lr_scale must be nonnegative")
        if (
            not 0 <= config.muon_momentum < 1
            or not 0 <= config.muon_momentum_final < 1
            or not 0 <= config.muon_beta2 < 1
        ):
            raise ValueError("Muon momentum and beta2 must be in [0, 1)")
        if config.muon_momentum_warmup_steps < 0:
            raise ValueError("muon_momentum_warmup_steps must be nonnegative")
        if not 1 <= config.muon_ns_steps <= len(_POLAR_EXPRESS_COEFFICIENTS):
            raise ValueError("muon_ns_steps is outside supported coefficient range")
        if not 0 <= config.adam_beta1 < 1 or not 0 <= config.adam_beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        self.config = config
        width = config.hidden_dim

        self.embedding = nn.Embedding(config.vocab_size, width)
        self.prediction_token = (
            nn.Parameter(torch.empty(width))
            if config.state_prediction_separation
            else None
        )
        self.decoder_weight = nn.Parameter(torch.empty(config.vocab_size, width))
        self.decoder_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.feedback_embedding = nn.Parameter(torch.empty(config.vocab_size, width))
        if config.meta_pc_output_feedback_mode == "kp_factorized":
            rank = config.meta_pc_output_feedback_rank
            self.feedback_token_factors = nn.Parameter(
                torch.empty(config.vocab_size, rank)
            )
            self.feedback_state_factors = nn.Parameter(torch.empty(rank, width))
        else:
            self.register_parameter("feedback_token_factors", None)
            self.register_parameter("feedback_state_factors", None)
        if config.learn_residual_stream_scales:
            residual_scale_shape = (
                (config.num_layers, config.hidden_dim)
                if config.residual_stream_scale_mode == "channel"
                else (config.num_layers,)
            )
            self.residual_stream_scales = nn.Parameter(
                torch.empty(residual_scale_shape)
            )
        else:
            self.register_parameter("residual_stream_scales", None)

        self.query_weights = nn.ParameterList([nn.Parameter(torch.empty(width, width)) for _ in range(config.num_layers)])
        self.key_weights = nn.ParameterList([nn.Parameter(torch.empty(width, width)) for _ in range(config.num_layers)])
        self.value_weights = nn.ParameterList([nn.Parameter(torch.empty(width, width)) for _ in range(config.num_layers)])
        self.self_weights = nn.ParameterList([nn.Parameter(torch.empty(width, width)) for _ in range(config.num_layers)])
        self.message_weights = nn.ParameterList([nn.Parameter(torch.empty(width, width)) for _ in range(config.num_layers)])
        self.cross_query_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(width, width)) for _ in range(config.num_layers)]
            if config.causal_encoder_decoder
            else []
        )
        self.cross_key_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(width, width)) for _ in range(config.num_layers)]
            if config.causal_encoder_decoder
            else []
        )
        self.cross_value_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(width, width)) for _ in range(config.num_layers)]
            if config.causal_encoder_decoder
            else []
        )
        self.cross_message_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(width, width)) for _ in range(config.num_layers)]
            if config.causal_encoder_decoder
            else []
        )
        self.cross_attention_gates = (
            nn.Parameter(torch.zeros(config.num_layers))
            if config.causal_encoder_decoder
            else None
        )
        self.ffn_up_weights = nn.ParameterList(
            [
                nn.Parameter(torch.empty(config.ffn_dim, width))
                for _ in range(config.num_layers)
            ]
            if config.ffn_dim > 0
            else []
        )
        self.ffn_down_weights = nn.ParameterList(
            [
                nn.Parameter(torch.empty(width, config.ffn_dim))
                for _ in range(config.num_layers)
            ]
            if config.ffn_dim > 0
            else []
        )
        value_layer_parity = (config.num_layers - 1) % 2
        self.token_value_tables = nn.ParameterDict(
            {
                str(layer): nn.Parameter(
                    torch.empty(config.vocab_size, width)
                )
                for layer in range(config.num_layers)
                if config.token_value_embeddings
                and layer % 2 == value_layer_parity
            }
        )
        self.inverse_query_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(width, width)) for _ in range(config.num_layers)]
            if config.inference_mode == "bidirectional_pc"
            or (
                config.inference_mode == "residual_error_pc"
                and config.hierarchical_error_prediction_mode != "direct"
            )
            else []
        )
        self.inverse_key_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(width, width)) for _ in range(config.num_layers)]
            if config.inference_mode == "bidirectional_pc"
            or (
                config.inference_mode == "residual_error_pc"
                and config.hierarchical_error_prediction_mode != "direct"
            )
            else []
        )
        self.generative_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(width, width)) for _ in range(config.num_layers)]
        )
        self.feedback_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(width, width)) for _ in range(config.num_layers)]
        )
        uses_stage_reciprocal = config.meta_pc_feedback_mode == "stage_reciprocal"
        self.feedback_query_weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(width, width)) for _ in range(config.num_layers)]
            if uses_stage_reciprocal
            else []
        )
        self.feedback_key_weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(width, width)) for _ in range(config.num_layers)]
            if uses_stage_reciprocal
            else []
        )
        self.feedback_value_weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(width, width)) for _ in range(config.num_layers)]
            if uses_stage_reciprocal
            else []
        )
        self.feedback_message_weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(width, width)) for _ in range(config.num_layers)]
            if uses_stage_reciprocal
            else []
        )
        self.feedback_ffn_up_weights = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(width, config.ffn_dim))
                for _ in range(config.num_layers)
            ]
            if uses_stage_reciprocal and config.ffn_dim > 0
            else []
        )
        self.feedback_ffn_down_weights = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(config.ffn_dim, width))
                for _ in range(config.num_layers)
            ]
            if uses_stage_reciprocal and config.ffn_dim > 0
            else []
        )
        self.feedback_state_weights = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(width, width))
                for _ in range(config.num_layers)
            ]
            if config.meta_pc_state_gated_reciprocal
            else []
        )
        self.state_biases = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(config.num_layers)])
        error_levels = config.num_layers - config.hierarchical_error_base_depth
        self.hierarchical_gate_weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(width)) for _ in range(error_levels)]
            if config.hierarchical_error_context_mode != "instant"
            else []
        )
        self.hierarchical_gate_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(())) for _ in range(error_levels)]
            if config.hierarchical_error_context_mode != "instant"
            else []
        )
        self.recursive_cp_her = (
            BoundedCPHERHierarchy(
                width=width,
                depth=error_levels,
                rank=config.hierarchical_error_recursive_cp_rank,
            )
            if config.hierarchical_error_recursive_cp_rank > 0
            else None
        )
        self.recursive_cp_her_fast = (
            copy.deepcopy(self.recursive_cp_her)
            if config.hierarchical_error_recursive_cp_delayed_consolidation
            else None
        )

        self.register_buffer("position_encoding", _sinusoidal_positions(config.seq_len, width), persistent=False)
        rotary_cosine, rotary_sine = _rotary_positions(config.seq_len, self.head_dim)
        self.register_buffer("rotary_cosine", rotary_cosine, persistent=False)
        self.register_buffer("rotary_sine", rotary_sine, persistent=False)
        self.register_buffer("local_adam_step", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "recursive_cp_consolidation_attempts",
            torch.zeros((), dtype=torch.long),
            persistent=config.hierarchical_error_recursive_cp_delayed_consolidation,
        )
        self.register_buffer(
            "recursive_cp_consolidation_accepts",
            torch.zeros((), dtype=torch.long),
            persistent=config.hierarchical_error_recursive_cp_delayed_consolidation,
        )
        self.reset_parameters()

        self._adam_buffer_names: dict[str, tuple[str, str]] = {}
        self._muon_buffer_names: dict[str, tuple[str, str]] = {}
        learned = list(self.named_parameters())
        for index, (name, parameter) in enumerate(learned):
            parameter.requires_grad_(False)
            if name.startswith(
                (
                    "feedback_query_weights.",
                    "feedback_key_weights.",
                    "feedback_value_weights.",
                    "feedback_message_weights.",
                    "feedback_ffn_up_weights.",
                    "feedback_ffn_down_weights.",
                )
            ):
                continue
            if self._uses_muon(name, parameter):
                momentum_name = f"local_muon_momentum_{index}"
                variance_name = f"local_muon_variance_{index}"
                variance_shape = (
                    (parameter.shape[-2], 1)
                    if parameter.shape[-2] >= parameter.shape[-1]
                    else (1, parameter.shape[-1])
                )
                self.register_buffer(momentum_name, torch.zeros_like(parameter))
                self.register_buffer(
                    variance_name,
                    torch.zeros(
                        variance_shape,
                        dtype=parameter.dtype,
                        device=parameter.device,
                    ),
                )
                self._muon_buffer_names[name] = (momentum_name, variance_name)
            else:
                mean_name = f"local_adam_mean_{index}"
                variance_name = f"local_adam_variance_{index}"
                self.register_buffer(mean_name, torch.zeros_like(parameter))
                self.register_buffer(variance_name, torch.zeros_like(parameter))
                self._adam_buffer_names[name] = (mean_name, variance_name)

        self._last_update_diagnostics: dict[str, float] = {}
        self._stage_probe_bases: dict[
            tuple[torch.device, torch.dtype, int], Tensor
        ] = {}

    def reset_parameters(self) -> None:
        width = self.config.hidden_dim
        nanochat_residual = self.config.block_state_mode == "nanochat_residual"
        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.8 if nanochat_residual else 0.02,
        )
        if self.prediction_token is not None:
            nn.init.normal_(
                self.prediction_token,
                mean=0.0,
                std=0.8 if nanochat_residual else 0.02,
            )
        with torch.no_grad():
            if nanochat_residual:
                nn.init.normal_(self.decoder_weight, mean=0.0, std=0.001)
            else:
                self.decoder_weight.copy_(self.embedding.weight)
            if self.config.meta_pc_output_feedback_mode == "decoder_transpose":
                self.feedback_embedding.copy_(self.decoder_weight)
            else:
                # Training-only feedback must not perturb deployment-weight RNG.
                with torch.random.fork_rng(devices=[]):
                    nn.init.normal_(
                        self.feedback_embedding,
                        mean=0.0,
                        std=0.001 if nanochat_residual else 0.02,
                    )
                    if (
                        self.feedback_token_factors is not None
                        and self.feedback_state_factors is not None
                    ):
                        target_codebook_std = (
                            0.001 if nanochat_residual else 0.02
                        )
                        balanced_factor_std = math.sqrt(
                            target_codebook_std
                            / math.sqrt(self.config.meta_pc_output_feedback_rank)
                        )
                        nn.init.normal_(
                            self.feedback_token_factors,
                            mean=0.0,
                            std=balanced_factor_std,
                        )
                        nn.init.normal_(
                            self.feedback_state_factors,
                            mean=0.0,
                            std=balanced_factor_std,
                        )
            if self.residual_stream_scales is not None:
                residual_schedule = torch.linspace(
                    self.config.residual_stream_scale_start,
                    self.config.residual_stream_scale_end,
                    self.config.num_layers,
                    device=self.residual_stream_scales.device,
                    dtype=self.residual_stream_scales.dtype,
                )
                if self.residual_stream_scales.ndim == 2:
                    residual_schedule = residual_schedule[:, None].expand_as(
                        self.residual_stream_scales
                    )
                self.residual_stream_scales.copy_(residual_schedule)
        for layer in range(self.config.num_layers):
            if nanochat_residual:
                bound = math.sqrt(3.0 / width)
                nn.init.uniform_(self.query_weights[layer], -bound, bound)
                nn.init.uniform_(self.key_weights[layer], -bound, bound)
                nn.init.uniform_(self.value_weights[layer], -bound, bound)
                nn.init.zeros_(self.self_weights[layer])
                nn.init.zeros_(self.message_weights[layer])
            else:
                nn.init.orthogonal_(self.query_weights[layer])
                nn.init.orthogonal_(self.key_weights[layer])
                nn.init.orthogonal_(self.value_weights[layer])
                nn.init.normal_(
                    self.self_weights[layer],
                    mean=0.0,
                    std=(
                        self.config.self_init_gain
                        if self.config.block_parameterization == "depth_mup"
                        else self.config.self_init_gain / math.sqrt(width)
                    ),
                )
                nn.init.normal_(
                    self.message_weights[layer],
                    mean=0.0,
                    std=(
                        self.config.message_init_gain
                        if self.config.block_parameterization == "depth_mup"
                        else self.config.message_init_gain / math.sqrt(width)
                    ),
                )
            if self.config.causal_encoder_decoder:
                cross_bound = math.sqrt(3.0 / width)
                nn.init.uniform_(
                    self.cross_query_weights[layer],
                    -cross_bound,
                    cross_bound,
                )
                nn.init.uniform_(
                    self.cross_key_weights[layer],
                    -cross_bound,
                    cross_bound,
                )
                nn.init.uniform_(
                    self.cross_value_weights[layer],
                    -cross_bound,
                    cross_bound,
                )
                if self.config.cross_attention_output_init_gain > 0:
                    nn.init.normal_(
                        self.cross_message_weights[layer],
                        mean=0.0,
                        std=(
                            self.config.cross_attention_output_init_gain
                            / math.sqrt(width)
                        ),
                    )
                else:
                    nn.init.zeros_(self.cross_message_weights[layer])
            with torch.no_grad():
                if self.config.inference_mode == "bidirectional_pc":
                    nn.init.orthogonal_(self.inverse_query_weights[layer])
                    nn.init.orthogonal_(self.inverse_key_weights[layer])
                    self.generative_weights[layer].copy_(torch.eye(width))
                elif self.config.inference_mode == "residual_error_pc":
                    if (
                        self.config.hierarchical_error_prediction_mode
                        != "direct"
                    ):
                        nn.init.orthogonal_(self.inverse_query_weights[layer])
                        nn.init.orthogonal_(self.inverse_key_weights[layer])
                    if self.config.hierarchical_error_prediction_mode in {
                        "direct",
                        "direct_transport",
                        "direct_phase_transport",
                    }:
                        nn.init.normal_(
                            self.generative_weights[layer],
                            mean=0.0,
                            std=(
                                self.config.self_init_gain
                                / math.sqrt(width)
                            ),
                        )
                    else:
                        self.generative_weights[layer].copy_(torch.eye(width))
                elif self.config.inference_mode == "her_error_hierarchy":
                    self.generative_weights[layer].zero_()
                elif self.config.inference_mode == "difference_target":
                    bound = math.sqrt(3.0 / width)
                    nn.init.uniform_(
                        self.generative_weights[layer],
                        -bound,
                        bound,
                    )
                else:
                    self.generative_weights[layer].copy_(torch.eye(width))
                if self.config.inference_mode == "bidirectional_pc":
                    self.feedback_weights[layer].zero_()
                elif self.config.inference_mode == "residual_error_pc":
                    if self.config.hierarchical_error_prediction_mode == "direct":
                        self.feedback_weights[layer].zero_()
                    else:
                        nn.init.normal_(
                            self.feedback_weights[layer],
                            mean=0.0,
                            std=self.config.message_init_gain / math.sqrt(width),
                        )
                elif (
                    self.config.inference_mode == "her_error_hierarchy"
                    and self.config.hierarchical_error_binding_mode
                    == "factorized"
                    and not self.config.hierarchical_error_outer_product_contract
                ) or self.config.inference_mode == "difference_target":
                    self.feedback_weights[layer].zero_()
                else:
                    self.feedback_weights[layer].copy_(torch.eye(width))
                for weights in (
                    self.feedback_query_weights,
                    self.feedback_key_weights,
                    self.feedback_value_weights,
                    self.feedback_message_weights,
                    self.feedback_ffn_up_weights,
                    self.feedback_ffn_down_weights,
                ):
                    if weights:
                        weights[layer].zero_()
                if self.feedback_state_weights:
                    self.feedback_state_weights[layer].zero_()
            with torch.no_grad():
                self.state_biases[layer].zero_()
        if self.cross_attention_gates is not None:
            with torch.no_grad():
                self.cross_attention_gates.zero_()
        if self.config.ffn_dim > 0:
            bound = (
                math.sqrt(3.0)
                * self.config.ffn_init_gain
                / math.sqrt(width)
            )
            for layer in range(self.config.num_layers):
                nn.init.uniform_(self.ffn_up_weights[layer], -bound, bound)
                nn.init.zeros_(self.ffn_down_weights[layer])
        token_value_bound = math.sqrt(3.0 / width)
        for table in self.token_value_tables.values():
            nn.init.uniform_(table, -token_value_bound, token_value_bound)
        for offset, (weight, bias) in enumerate(
            zip(self.hierarchical_gate_weights, self.hierarchical_gate_biases)
        ):
            write_rate = (
                self.config.hierarchical_error_initial_write_rate
                ** (offset + 1)
            )
            with torch.no_grad():
                weight.zero_()
                bias.fill_(math.log(write_rate / (1.0 - write_rate)))
        if self.recursive_cp_her is not None:
            self.recursive_cp_her.reset_parameters()
        if self.recursive_cp_her_fast is not None:
            if self.recursive_cp_her is None:
                raise RuntimeError("fast recursive CP hierarchy requires deployed hierarchy")
            self.recursive_cp_her_fast.load_state_dict(
                self.recursive_cp_her.state_dict()
            )
        self.recursive_cp_consolidation_attempts.zero_()
        self.recursive_cp_consolidation_accepts.zero_()

    def _uses_muon(self, name: str, parameter: Tensor) -> bool:
        if (
            self.config.inference_mode in {
                "her_error_hierarchy",
                "difference_target",
            }
            and name.startswith(("generative_weights.", "feedback_weights."))
        ):
            return False
        return (
            self.config.matrix_optimizer == "muon"
            and parameter.ndim == 2
            and name.startswith(
                (
                    "query_weights.",
                    "key_weights.",
                    "value_weights.",
                    "self_weights.",
                    "message_weights.",
                    "cross_query_weights.",
                    "cross_key_weights.",
                    "cross_value_weights.",
                    "cross_message_weights.",
                    "ffn_up_weights.",
                    "ffn_down_weights.",
                    "inverse_query_weights.",
                    "inverse_key_weights.",
                    "generative_weights.",
                    "feedback_weights.",
                    "feedback_query_weights.",
                    "feedback_key_weights.",
                    "feedback_value_weights.",
                    "feedback_message_weights.",
                    "feedback_ffn_up_weights.",
                    "feedback_ffn_down_weights.",
                )
            )
        )

    @property
    def head_dim(self) -> int:
        return self.config.hidden_dim // self.config.num_heads

    def learned_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def persistent_slot_count(self) -> int:
        return sum(value.numel() for value in self.state_dict().values())

    def contextual_parameter_count(self) -> int:
        block_parameters = (
            list(self.query_weights)
            + list(self.key_weights)
            + list(self.value_weights)
            + list(self.self_weights)
            + list(self.message_weights)
            + list(self.cross_query_weights)
            + list(self.cross_key_weights)
            + list(self.cross_value_weights)
            + list(self.cross_message_weights)
            + list(self.ffn_up_weights)
            + list(self.ffn_down_weights)
            + list(self.token_value_tables.values())
            + list(self.inverse_query_weights)
            + list(self.inverse_key_weights)
            + list(self.generative_weights)
            + list(self.feedback_weights)
            + list(self.feedback_query_weights)
            + list(self.feedback_key_weights)
            + list(self.feedback_value_weights)
            + list(self.feedback_message_weights)
            + list(self.feedback_ffn_up_weights)
            + list(self.feedback_ffn_down_weights)
            + list(self.feedback_state_weights)
            + list(self.state_biases)
            + list(self.hierarchical_gate_weights)
            + list(self.hierarchical_gate_biases)
        )
        if self.feedback_token_factors is not None:
            block_parameters.append(self.feedback_token_factors)
        if self.feedback_state_factors is not None:
            block_parameters.append(self.feedback_state_factors)
        if self.residual_stream_scales is not None:
            block_parameters.append(self.residual_stream_scales)
        if self.cross_attention_gates is not None:
            block_parameters.append(self.cross_attention_gates)
        if self.recursive_cp_her is not None:
            block_parameters.extend(self.recursive_cp_her.parameters())
        if self.recursive_cp_her_fast is not None:
            block_parameters.extend(self.recursive_cp_her_fast.parameters())
        return self.feedback_embedding.numel() + sum(parameter.numel() for parameter in block_parameters)

    @torch.no_grad()
    def reset_local_optimizer_state(self, scope: str) -> dict[str, int]:
        if scope not in {"route", "muon", "context", "recursive_cp", "all"}:
            raise ValueError(
                "optimizer reset scope must be route, muon, context, "
                "recursive_cp, or all"
            )
        context_names = {
            name
            for name, _ in self.named_parameters()
            if name not in {"decoder_weight", "decoder_bias", "embedding.weight"}
        }
        route_names = {
            name
            for name, _ in self.named_parameters()
            if name.startswith(
                (
                    "query_weights.",
                    "key_weights.",
                    "cross_query_weights.",
                    "cross_key_weights.",
                    "inverse_query_weights.",
                    "inverse_key_weights.",
                )
            )
        }
        recursive_cp_names = {
            name
            for name, _ in self.named_parameters()
            if name.startswith(("recursive_cp_her.", "recursive_cp_her_fast."))
        }
        reset_counts = {"adam": 0, "muon": 0}
        for name, (mean_name, variance_name) in self._adam_buffer_names.items():
            if (
                scope == "all"
                or (scope == "context" and name in context_names)
                or (scope == "route" and name in route_names)
                or (scope == "recursive_cp" and name in recursive_cp_names)
            ):
                getattr(self, mean_name).zero_()
                getattr(self, variance_name).zero_()
                reset_counts["adam"] += 1
        for name, (momentum_name, variance_name) in self._muon_buffer_names.items():
            if (
                scope in {"muon", "all"}
                or (scope == "context" and name in context_names)
                or (scope == "route" and name in route_names)
                or (scope == "recursive_cp" and name in recursive_cp_names)
            ):
                getattr(self, momentum_name).zero_()
                getattr(self, variance_name).zero_()
                reset_counts["muon"] += 1
        if scope == "all":
            self.local_adam_step.zero_()
        return reset_counts

    @torch.no_grad()
    def enforce_recursive_cp_component_bound(self, *, include_deployed: bool = True):
        cap = self.config.hierarchical_error_recursive_cp_component_norm_cap
        if self.recursive_cp_her is None or cap <= 0:
            return None
        deployed_stats = None
        if include_deployed:
            deployed_stats = self.recursive_cp_her.canonicalize_(
                maximum_component_norm=cap,
            )
        if self.recursive_cp_her_fast is None:
            return deployed_stats
        return self.recursive_cp_her_fast.canonicalize_(
            maximum_component_norm=cap,
        )

    @torch.no_grad()
    def initialize_recursive_cp_fast_from_deployed(self) -> None:
        if self.recursive_cp_her is None or self.recursive_cp_her_fast is None:
            raise ValueError("delayed recursive CP consolidation is not enabled")
        self.recursive_cp_her_fast.load_state_dict(self.recursive_cp_her.state_dict())
        for name, buffer_names in self._adam_buffer_names.items():
            if name.startswith("recursive_cp_her_fast."):
                for buffer_name in buffer_names:
                    getattr(self, buffer_name).zero_()
        self.recursive_cp_consolidation_attempts.zero_()
        self.recursive_cp_consolidation_accepts.zero_()

    def recursive_cp_delayed_checkpoint_keys(self) -> set[str]:
        if self.recursive_cp_her_fast is None:
            return set()
        keys = {
            "recursive_cp_consolidation_attempts",
            "recursive_cp_consolidation_accepts",
        }
        keys.update(
            f"recursive_cp_her_fast.{name}"
            for name, _ in self.recursive_cp_her_fast.named_parameters()
        )
        for name, buffer_names in self._adam_buffer_names.items():
            if name.startswith("recursive_cp_her_fast."):
                keys.update(buffer_names)
        return keys

    @torch.no_grad()
    def _consolidate_recursive_cp_if_better(
        self,
        memories: list[Tensor],
        target: Tensor,
        deployed_result: BoundedCPHERLocalResult,
        fast_result: BoundedCPHERLocalResult,
    ) -> dict[str, float]:
        if self.recursive_cp_her is None or self.recursive_cp_her_fast is None:
            raise ValueError("delayed recursive CP consolidation is not enabled")
        deployed_score = float(
            self.recursive_cp_her.relative_error_score(deployed_result).item()
        )
        fast_score = float(
            self.recursive_cp_her_fast.relative_error_score(fast_result).item()
        )
        candidate_score = deployed_score
        accepted = False
        self.recursive_cp_consolidation_attempts.add_(1)
        minimum = self.config.hierarchical_error_recursive_cp_min_improvement
        if fast_score + minimum < deployed_score:
            deployed_parameters = tuple(self.recursive_cp_her.parameters())
            fast_parameters = tuple(self.recursive_cp_her_fast.parameters())
            backup = [parameter.clone() for parameter in deployed_parameters]
            rate = self.config.hierarchical_error_recursive_cp_consolidation_rate
            for deployed, fast in zip(deployed_parameters, fast_parameters):
                deployed.lerp_(fast, rate)
            cap = self.config.hierarchical_error_recursive_cp_component_norm_cap
            self.recursive_cp_her.canonicalize_(maximum_component_norm=cap)
            candidate_result = self.recursive_cp_her.local_gradients(
                memories,
                target,
            )
            candidate_score = float(
                self.recursive_cp_her.relative_error_score(candidate_result).item()
            )
            accepted = candidate_score + minimum < deployed_score
            if accepted:
                self.recursive_cp_consolidation_accepts.add_(1)
            else:
                for deployed, original in zip(deployed_parameters, backup):
                    deployed.copy_(original)
        attempts = int(self.recursive_cp_consolidation_attempts.item())
        accepts = int(self.recursive_cp_consolidation_accepts.item())
        return {
            "deployed_score": deployed_score,
            "fast_score": fast_score,
            "candidate_score": candidate_score,
            "accepted": float(accepted),
            "attempts": float(attempts),
            "accepts": float(accepts),
            "acceptance_rate": accepts / max(attempts, 1),
        }

    @torch.no_grad()
    def reset_hierarchical_error_predictors(self) -> dict[str, int]:
        if self.config.inference_mode != "her_error_hierarchy":
            raise ValueError(
                "hierarchical error predictor reset requires her_error_hierarchy"
            )
        reset_names: set[str] = set()
        if self.recursive_cp_her is not None:
            self.recursive_cp_her.reset_parameters()
            reset_names.update(
                f"recursive_cp_her.{name}"
                for name, _ in self.recursive_cp_her.named_parameters()
            )
            if self.recursive_cp_her_fast is not None:
                self.recursive_cp_her_fast.load_state_dict(
                    self.recursive_cp_her.state_dict()
                )
                reset_names.update(
                    f"recursive_cp_her_fast.{name}"
                    for name, _ in self.recursive_cp_her_fast.named_parameters()
                )
                self.recursive_cp_consolidation_attempts.zero_()
                self.recursive_cp_consolidation_accepts.zero_()
        else:
            for layer in range(
                self.config.hierarchical_error_base_depth,
                self.config.num_layers,
            ):
                self.generative_weights[layer].zero_()
                reset_names.add(f"generative_weights.{layer}")
                self.feedback_weights[layer].zero_()
                reset_names.add(f"feedback_weights.{layer}")
        reset_buffers = 0
        for name in reset_names:
            buffers = self._adam_buffer_names.get(name)
            if buffers is None:
                continue
            for buffer_name in buffers:
                getattr(self, buffer_name).zero_()
            reset_buffers += 1
        return {
            "parameters": len(reset_names),
            "optimizer_buffers": reset_buffers,
        }

    def _hierarchical_error_level_start_step(self, offset: int) -> int:
        return (
            self.config.hierarchical_error_learning_start_step
            + offset * self.config.hierarchical_error_level_start_interval
        )

    def _hierarchical_error_level_is_active(
        self,
        layer: int,
        current_step: int,
    ) -> bool:
        offset = layer - self.config.hierarchical_error_base_depth
        return current_step >= self._hierarchical_error_level_start_step(offset)

    def active_parameter_count(self) -> int:
        inactive_block = 0
        if self.config.block_state_mode == "nanochat_residual":
            inactive_block = sum(
                parameter.numel()
                for parameter in (*self.self_weights, *self.state_biases)
            )
        if (
            self.config.causal_encoder_decoder
            and self.config.causal_encoder_depth > 0
        ):
            inactive_block += sum(
                parameter.numel()
                for parameter in (
                    *self.cross_query_weights[: self.config.causal_encoder_depth],
                    *self.cross_key_weights[: self.config.causal_encoder_depth],
                    *self.cross_value_weights[: self.config.causal_encoder_depth],
                    *self.cross_message_weights[: self.config.causal_encoder_depth],
                )
            )
            inactive_block += self.config.causal_encoder_depth
        if self.config.inference_mode == "her_error_hierarchy":
            inactive = self.feedback_embedding.numel() + inactive_block
            if self.recursive_cp_her is not None:
                inactive += sum(
                    weight.numel()
                    for weight in (
                        *self.generative_weights,
                        *self.feedback_weights,
                    )
                )
                if self.config.route_lr_scale <= 0:
                    inactive += sum(
                        weight.numel()
                        for weight in (*self.query_weights, *self.key_weights)
                    )
                if self.recursive_cp_her_fast is not None:
                    inactive += sum(
                        parameter.numel()
                        for parameter in self.recursive_cp_her_fast.parameters()
                    )
                return self.learned_parameter_count() - inactive
            if self.config.sps_node_residual_hierarchy:
                inactive += sum(
                    weight.numel()
                    for weight in (
                        *self.generative_weights,
                        *self.feedback_weights,
                    )
                )
                if self.config.route_lr_scale <= 0:
                    inactive += sum(
                        weight.numel()
                        for weight in (*self.query_weights, *self.key_weights)
                    )
                return self.learned_parameter_count() - inactive
            if self.config.hierarchical_error_binding_mode != "factorized":
                inactive += sum(
                    weight.numel() for weight in self.feedback_weights
                )
            elif self.config.hierarchical_error_prediction_mode == "direct":
                inactive += sum(
                    weight.numel() for weight in self.feedback_weights
                )
            inactive += sum(
                self.generative_weights[layer].numel()
                for layer in range(self.config.hierarchical_error_base_depth)
            )
            if (
                self.config.hierarchical_error_binding_mode == "factorized"
                and self.config.hierarchical_error_prediction_mode == "gated"
            ):
                inactive += sum(
                    self.feedback_weights[layer].numel()
                    for layer in range(
                        self.config.hierarchical_error_base_depth
                    )
                )
            if self.config.route_lr_scale <= 0:
                inactive += sum(
                    weight.numel()
                    for weight in (*self.query_weights, *self.key_weights)
                )
            return self.learned_parameter_count() - inactive
        if self.config.inference_mode == "difference_target":
            inactive = self.feedback_embedding.numel() + inactive_block
            if self.config.route_lr_scale <= 0:
                inactive += sum(
                    weight.numel()
                    for weight in (*self.query_weights, *self.key_weights)
                )
            return self.learned_parameter_count() - inactive
        if self.config.inference_mode in {
            "recurrent_local",
            "layerwise_local",
            "meta_pc",
        }:
            inactive = inactive_block
            if (
                self.config.inference_mode != "meta_pc"
                or self.config.meta_pc_output_feedback_mode
                in {"decoder_transpose", "kp_factorized"}
            ):
                inactive += self.feedback_embedding.numel()
            inactive += sum(
                weight.numel() for weight in self.generative_weights
            )
            if (
                self.config.inference_mode == "meta_pc"
                and self.config.meta_pc_feedback_mode == "learned_reciprocal"
            ):
                # Block 0 has no lower latent displacement to transport into.
                inactive += self.feedback_weights[0].numel()
                if self.feedback_state_weights:
                    inactive += self.feedback_state_weights[0].numel()
            else:
                inactive += sum(
                    weight.numel() for weight in self.feedback_weights
                )
            if self.config.route_lr_scale <= 0:
                inactive += sum(
                    weight.numel()
                    for weight in (*self.query_weights, *self.key_weights)
                )
            return self.learned_parameter_count() - inactive
        if self.config.inference_mode in {
            "bidirectional_pc",
            "residual_error_pc",
        }:
            inactive = inactive_block
            if (
                self.config.inference_mode != "residual_error_pc"
                or self.config.meta_pc_output_feedback_mode
                in {"decoder_transpose", "fixed_random", "kp_factorized"}
            ):
                inactive += self.feedback_embedding.numel()
            if (
                self.config.inference_mode == "residual_error_pc"
                and self.config.hierarchical_error_prediction_mode == "direct"
            ):
                inactive += sum(
                    weight.numel() for weight in self.feedback_weights
                )
            if self.config.route_lr_scale <= 0:
                inactive += sum(
                    weight.numel()
                    for weight in (
                        *self.query_weights,
                        *self.key_weights,
                        *self.inverse_query_weights,
                        *self.inverse_key_weights,
                    )
                )
            return self.learned_parameter_count() - inactive
        if self.config.inference_mode != "pc_energy":
            inactive = inactive_block
            if self.config.route_lr_scale <= 0:
                inactive += sum(
                    weight.numel()
                    for weight in (*self.query_weights, *self.key_weights)
                )
            return self.learned_parameter_count() - inactive
        inactive = inactive_block + sum(
            weight.numel() for weight in self.generative_weights
        )
        if self.config.route_lr_scale <= 0:
            inactive += sum(
                weight.numel()
                for weight in (*self.query_weights, *self.key_weights)
            )
        if self.config.learned_feedback_proposal_scale <= 0:
            inactive += self.feedback_embedding.numel()
            inactive += sum(weight.numel() for weight in self.feedback_weights)
        return self.learned_parameter_count() - inactive

    def training_feedback_parameter_count(self) -> int:
        if self.config.inference_mode not in {"meta_pc", "residual_error_pc"}:
            return 0
        training_only = 0
        if not self.config.meta_pc_output_feedback_values:
            if self.config.meta_pc_output_feedback_mode in {
                "fixed_random",
                "kp_learned",
            }:
                training_only += self.feedback_embedding.numel()
            elif self.config.meta_pc_output_feedback_mode == "kp_factorized":
                if self.feedback_token_factors is None or self.feedback_state_factors is None:
                    raise RuntimeError("factorized output feedback is not initialized")
                training_only += (
                    self.feedback_token_factors.numel()
                    + self.feedback_state_factors.numel()
                )
        if self.config.meta_pc_feedback_mode == "learned_reciprocal":
            training_only += sum(
                weight.numel() for weight in self.feedback_weights[1:]
            )
            training_only += sum(
                weight.numel() for weight in self.feedback_state_weights[1:]
            )
        if self.config.meta_pc_feedback_mode == "stage_reciprocal":
            training_only += sum(
                weight.numel()
                for weights in (
                    self.feedback_query_weights,
                    self.feedback_key_weights,
                    self.feedback_value_weights,
                    self.feedback_message_weights,
                    self.feedback_ffn_up_weights,
                    self.feedback_ffn_down_weights,
                )
                for weight in weights
            )
        return training_only

    def deployment_parameter_count(self) -> int:
        return self.active_parameter_count() - self.training_feedback_parameter_count()

    def get_device(self) -> torch.device:
        return self.embedding.weight.device

    def _position(self, length: int, like: Tensor) -> Tensor:
        source_length = (
            (length + 1) // 2
            if self.config.state_prediction_separation
            else length
        )
        if source_length <= self.position_encoding.shape[0]:
            position = self.position_encoding[:source_length]
        else:
            position = _sinusoidal_positions(
                source_length,
                self.config.hidden_dim,
            )
        if self.config.state_prediction_separation:
            position = position.repeat_interleave(2, dim=0)[:length]
        return position.to(device=like.device, dtype=like.dtype)

    def _rotary(self, length: int, like: Tensor) -> tuple[Tensor, Tensor]:
        source_length = (
            (length + 1) // 2
            if self.config.state_prediction_separation
            else length
        )
        if source_length <= self.rotary_cosine.shape[2]:
            cosine = self.rotary_cosine[:, :, :source_length]
            sine = self.rotary_sine[:, :, :source_length]
        else:
            cosine, sine = _rotary_positions(source_length, self.head_dim)
        if self.config.state_prediction_separation:
            cosine = cosine.repeat_interleave(2, dim=2)[..., :length, :]
            sine = sine.repeat_interleave(2, dim=2)[..., :length, :]
        return (
            cosine.to(device=like.device, dtype=like.dtype),
            sine.to(device=like.device, dtype=like.dtype),
        )

    def _inference_lr(self, step: int) -> float:
        final = self.config.inference_lr_final
        start = self.config.inference_lr_schedule_start_step
        end = self.config.inference_lr_schedule_end_step
        if final <= 0 or step <= start:
            return self.config.inference_lr
        if step >= end:
            return final
        progress = (step - start) / float(end - start)
        return self.config.inference_lr + progress * (final - self.config.inference_lr)

    def _optimizer_lr_multiplier(self, step: int) -> float:
        warmup_steps = self.config.optimizer_warmup_steps
        if warmup_steps <= 0:
            return 1.0
        effective_step = step if step > 0 else int(self.local_adam_step.item()) + 1
        return min(max(effective_step / float(warmup_steps), 0.0), 1.0)

    def _residual_stream_lr_scale(self, step: int) -> float:
        initial = self.config.residual_stream_lr_scale
        final = self.config.residual_stream_lr_scale_final
        if final <= 0:
            return initial
        effective_step = step if step > 0 else int(self.local_adam_step.item()) + 1
        start = self.config.residual_stream_lr_schedule_start_step
        end = self.config.residual_stream_lr_schedule_end_step
        if effective_step <= start:
            return initial
        if effective_step >= end:
            return final
        progress = (effective_step - start) / float(end - start)
        return initial + progress * (final - initial)

    def _muon_momentum(self, step: int) -> float:
        warmup_steps = self.config.muon_momentum_warmup_steps
        if warmup_steps <= 0:
            return self.config.muon_momentum
        effective_step = step if step > 0 else int(self.local_adam_step.item()) + 1
        progress = min(max((effective_step - 1) / float(warmup_steps), 0.0), 1.0)
        return self.config.muon_momentum + progress * (
            self.config.muon_momentum_final - self.config.muon_momentum
        )

    def _readout_depth(self) -> int:
        if self.config.inference_mode == "recurrent_local":
            return self.config.recurrent_shared_steps
        return self.config.num_layers if self.config.readout_layer == 0 else self.config.readout_layer

    def _split_heads(self, value: Tensor) -> Tensor:
        batch, tokens, _ = value.shape
        return value.view(batch, tokens, self.config.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, value: Tensor) -> Tensor:
        batch, _, tokens, _ = value.shape
        return value.transpose(1, 2).contiguous().view(batch, tokens, self.config.hidden_dim)

    def _prediction_states(self, states: Tensor) -> Tensor:
        if not self.config.state_prediction_separation:
            return states
        return states[:, 1::2]

    def _prediction_only(self, states: Tensor) -> Tensor:
        if not self.config.state_prediction_separation:
            return states
        prediction = torch.zeros_like(states)
        prediction[:, 1::2] = states[:, 1::2]
        return prediction

    def _persistent_state_only(self, states: Tensor) -> Tensor:
        if not self.config.state_prediction_separation:
            return states
        persistent = torch.zeros_like(states)
        persistent[:, 0::2] = states[:, 0::2]
        return persistent

    def _state_context(self, states: Tensor) -> Tensor:
        if not self.config.state_prediction_separation:
            return states
        return states[:, 0::2].repeat_interleave(2, dim=1)

    def _state_context_vjp(self, signal: Tensor) -> Tensor:
        if not self.config.state_prediction_separation:
            return signal
        routed = torch.zeros_like(signal)
        routed[:, 0::2] = signal[:, 0::2] + signal[:, 1::2]
        return routed

    def _expand_prediction_signal(
        self,
        signal: Tensor,
        template: Tensor,
    ) -> Tensor:
        if not self.config.state_prediction_separation:
            return signal
        expanded = torch.zeros_like(template)
        expanded[:, 1::2] = signal
        return expanded

    def _causal_mask(
        self,
        tokens: int,
        device: torch.device,
        *,
        layer: int | None = None,
    ) -> Tensor:
        if not self.config.state_prediction_separation:
            mask = torch.ones(
                tokens,
                tokens,
                dtype=torch.bool,
                device=device,
            ).triu(diagonal=1)
            if (
                layer is not None
                and layer < self.config.fast_attention_layers
            ):
                position = torch.arange(tokens, device=device)
                distance = position.unsqueeze(1) - position.unsqueeze(0)
                mask |= distance >= self.config.fast_attention_window
            return mask
        if tokens % 2 != 0:
            raise ValueError("separated state/prediction sequence must be paired")
        slot = torch.arange(tokens, device=device)
        position = torch.div(slot, 2, rounding_mode="floor")
        is_prediction = slot.remainder(2).bool()
        query_position = position.unsqueeze(1)
        key_position = position.unsqueeze(0)
        key_is_prediction = is_prediction.unsqueeze(0)
        query_is_prediction = is_prediction.unsqueeze(1)
        persistent_input = (~key_is_prediction) & (
            key_position <= query_position
        )
        distance = query_position - key_position
        window = self.config.prediction_stream_window
        recent_prediction_for_state = (
            key_is_prediction
            & (~query_is_prediction)
            & (distance >= 1)
            & (distance <= window)
        )
        recent_prediction_for_prediction = (
            key_is_prediction
            & query_is_prediction
            & (distance >= 0)
            & (distance <= window)
        )
        allowed = (
            persistent_input
            | recent_prediction_for_state
            | recent_prediction_for_prediction
        )
        return ~allowed

    def _compose_prediction(
        self,
        lower_state: Tensor,
        residual: Tensor,
        bias: Tensor,
        *,
        branch_scale: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        if self.config.block_state_mode == "squashed_state":
            prediction = torch.tanh(lower_state + residual + bias)
            return prediction, 1.0 - prediction.square()
        if self.config.block_state_mode == "nanochat_residual":
            return (
                lower_state + branch_scale * residual,
                torch.full_like(residual, branch_scale),
            )
        residual_activation = torch.tanh(residual + bias)
        prediction = lower_state + self.config.residual_scale * residual_activation
        derivative = self.config.residual_scale * (1.0 - residual_activation.square())
        return prediction, derivative

    def _block_weight_scale(self) -> float:
        if self.config.block_parameterization == "depth_mup":
            return 1.0 / math.sqrt(
                self.config.hidden_dim * self.config.num_layers
            )
        return 1.0

    def _sensory_reinjection_scale(self, layer: int) -> float:
        if self.config.num_layers == 1:
            return self.config.sensory_reinjection_start
        fraction = layer / float(self.config.num_layers - 1)
        return (
            (1.0 - fraction) * self.config.sensory_reinjection_start
            + fraction * self.config.sensory_reinjection_end
        )

    def _residual_stream_scale(self, layer: int) -> float | Tensor:
        if self.residual_stream_scales is not None:
            return self.residual_stream_scales[layer]
        if self.config.num_layers == 1:
            return self.config.residual_stream_scale_start
        fraction = layer / float(self.config.num_layers - 1)
        return (
            (1.0 - fraction) * self.config.residual_stream_scale_start
            + fraction * self.config.residual_stream_scale_end
        )

    def _residual_stream_scale_mean(self, layer: int) -> float:
        scale = self._residual_stream_scale(layer)
        if isinstance(scale, Tensor):
            return float(scale.float().mean().item())
        return float(scale)

    def _causal_encoder_memory(
        self,
        states: list[Tensor],
        layer: int,
    ) -> Tensor | None:
        if (
            not self.config.causal_encoder_decoder
            or layer < self.config.causal_encoder_depth
        ):
            return None
        return states[self.config.causal_encoder_depth]

    def _apply_inference_delta(
        self,
        state: Tensor,
        delta: Tensor,
        rate: float,
    ) -> Tensor:
        updated = state + rate * delta
        if self.config.block_state_mode == "squashed_state":
            return updated.clamp(-3.0, 3.0)
        return updated

    def _block(
        self,
        layer: int,
        lower_state: Tensor,
        *,
        branch_scale: float = 1.0,
        token_ids: Tensor | None = None,
        sensory_state: Tensor | None = None,
        encoder_memory: Tensor | None = None,
    ) -> _AttentionCache:
        direct_input_scale = 1.0
        source_input_scale = 1.0
        source_input = lower_state
        direct_input = lower_state
        sensory_scale = self._sensory_reinjection_scale(layer)
        if self.config.block_state_mode == "nanochat_residual":
            direct_input_scale = self._residual_stream_scale(layer)
            direct_input = direct_input_scale * lower_state
            if self.config.sensory_reinjection_mode == "pre_block":
                source_input_scale = direct_input_scale
                source_input = direct_input
                if sensory_scale > 0:
                    if sensory_state is None:
                        raise ValueError(
                            "sensory state is required by sensory reinjection"
                        )
                    source_input = source_input + sensory_scale * sensory_state
                direct_input = source_input
        source = _rms_norm(source_input)
        query_projection = self._split_heads(
            F.linear(source, self.query_weights[layer])
        )
        key_projection = self._split_heads(
            F.linear(source, self.key_weights[layer])
        )
        query_pre_norm = query_projection
        key_pre_norm = key_projection
        query = query_projection
        key = key_projection
        if self.config.block_state_mode == "nanochat_residual":
            cosine, sine = self._rotary(lower_state.shape[1], query_projection)
            query_pre_norm = _apply_rotary(query_projection, cosine, sine)
            key_pre_norm = _apply_rotary(key_projection, cosine, sine)
            query = 1.2 * _rms_norm(query_pre_norm)
            key = 1.2 * _rms_norm(key_pre_norm)
        value_projection = F.linear(source, self.value_weights[layer])
        token_value_key = str(layer)
        if token_value_key in self.token_value_tables:
            if token_ids is None:
                raise ValueError("token IDs are required by token value embeddings")
            value_projection = value_projection + (
                self.config.token_value_scale
                * self.token_value_tables[token_value_key][token_ids]
            )
        elif (
            self.config.meta_pc_output_feedback_values
            and (
                layer % 2 == (self.config.num_layers - 1) % 2
                or layer == self.config.num_layers - 1
            )
        ):
            if token_ids is None:
                raise ValueError("token IDs are required by shared feedback values")
            value_projection = value_projection + (
                self.config.token_value_scale
                * F.normalize(
                    self.feedback_embedding[token_ids],
                    dim=-1,
                    eps=1e-8,
                )
            )
        value = self._split_heads(value_projection)
        score_scale = 1.0 / (math.sqrt(self.head_dim) * self.config.attention_temperature)
        scores = torch.matmul(query, key.transpose(-2, -1)) * score_scale
        attention_mask = self._causal_mask(
            lower_state.shape[1],
            lower_state.device,
            layer=layer,
        )
        scores.masked_fill_(attention_mask, -torch.inf)
        weights = F.softmax(scores, dim=-1)
        message = self._merge_heads(torch.matmul(weights, value))
        cross_attention: _CrossAttentionCache | None = None
        cross_residual: Tensor | None = None
        if (
            self.config.causal_encoder_decoder
            and layer >= self.config.causal_encoder_depth
        ):
            memory_input = (
                sensory_state if encoder_memory is None else encoder_memory
            )
            if memory_input is None:
                raise ValueError(
                    "encoder memory is required by causal encoder-decoder"
                )
            if self.cross_attention_gates is None:
                raise RuntimeError("cross-attention gates are not initialized")
            memory = _rms_norm(memory_input)
            cross_query_projection = self._split_heads(
                F.linear(source, self.cross_query_weights[layer])
            )
            cross_key_projection = self._split_heads(
                F.linear(memory, self.cross_key_weights[layer])
            )
            cross_query_pre_norm = cross_query_projection
            cross_key_pre_norm = cross_key_projection
            cosine, sine = self._rotary(
                lower_state.shape[1],
                cross_query_projection,
            )
            cross_query_pre_norm = _apply_rotary(
                cross_query_projection,
                cosine,
                sine,
            )
            cross_key_pre_norm = _apply_rotary(
                cross_key_projection,
                cosine,
                sine,
            )
            cross_query = 1.2 * _rms_norm(cross_query_pre_norm)
            cross_key = 1.2 * _rms_norm(cross_key_pre_norm)
            cross_value = self._split_heads(
                F.linear(memory, self.cross_value_weights[layer])
            )
            cross_scores = torch.matmul(
                cross_query,
                cross_key.transpose(-2, -1),
            ) * score_scale
            cross_mask = self._causal_mask(
                lower_state.shape[1],
                lower_state.device,
                layer=layer,
            )
            cross_scores.masked_fill_(cross_mask, -torch.inf)
            cross_weights = F.softmax(cross_scores, dim=-1)
            cross_message = self._merge_heads(
                torch.matmul(cross_weights, cross_value)
            )
            cross_output = F.linear(
                cross_message,
                self.cross_message_weights[layer],
            )
            cross_gate = torch.sigmoid(self.cross_attention_gates[layer])
            cross_residual = cross_gate * cross_output
            cross_attention = _CrossAttentionCache(
                query_source=source,
                memory_input=memory_input,
                memory=memory,
                query_projection=cross_query_projection,
                key_projection=cross_key_projection,
                query_pre_norm=cross_query_pre_norm,
                key_pre_norm=cross_key_pre_norm,
                query=cross_query,
                key=cross_key,
                value=cross_value,
                attention_mask=cross_mask,
                weights=cross_weights,
                message=cross_message,
                output=cross_output,
                gate=cross_gate,
                memory_is_lower_state=memory_input is lower_state,
            )
        if self.config.block_state_mode == "nanochat_residual":
            residual = F.linear(message, self.message_weights[layer])
            if cross_residual is not None:
                residual = residual + cross_residual
        else:
            residual = self._block_weight_scale() * (
                F.linear(source, self.self_weights[layer])
                + F.linear(message, self.message_weights[layer])
            ) / math.sqrt(2.0)
        attention_prediction, attention_prediction_derivative = self._compose_prediction(
            direct_input,
            residual,
            self.state_biases[layer],
            branch_scale=branch_scale,
        )
        ffn_source: Tensor | None = None
        ffn_pre_activation: Tensor | None = None
        ffn_activation: Tensor | None = None
        if self.config.ffn_dim > 0:
            ffn_source = _rms_norm(attention_prediction)
            ffn_pre_activation = F.linear(
                ffn_source,
                self.ffn_up_weights[layer],
            )
            ffn_activation = F.relu(ffn_pre_activation).square()
            ffn_residual = F.linear(
                ffn_activation,
                self.ffn_down_weights[layer],
            )
            if self.config.block_state_mode == "nanochat_residual":
                prediction = attention_prediction + branch_scale * ffn_residual
                prediction_derivative = torch.full_like(
                    ffn_residual,
                    branch_scale,
                )
            else:
                ffn_residual_activation = torch.tanh(ffn_residual)
                prediction = attention_prediction + (
                    self.config.residual_scale * ffn_residual_activation
                )
                prediction_derivative = self.config.residual_scale * (
                    1.0 - ffn_residual_activation.square()
                )
        else:
            prediction = attention_prediction
            prediction_derivative = attention_prediction_derivative
        if sensory_scale > 0 and self.config.sensory_reinjection_mode == "post_block":
            if sensory_state is None:
                raise ValueError("sensory state is required by sensory reinjection")
            prediction = prediction + sensory_scale * sensory_state
        return _AttentionCache(
            source_input=source_input,
            source_input_scale=source_input_scale,
            direct_input_scale=direct_input_scale,
            source=source,
            query_projection=query_projection,
            key_projection=key_projection,
            query_pre_norm=query_pre_norm,
            key_pre_norm=key_pre_norm,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            weights=weights,
            message=message,
            attention_prediction=attention_prediction,
            attention_prediction_derivative=attention_prediction_derivative,
            ffn_source=ffn_source,
            ffn_pre_activation=ffn_pre_activation,
            ffn_activation=ffn_activation,
            prediction=prediction,
            prediction_derivative=prediction_derivative,
            token_ids=token_ids,
            cross_attention=cross_attention,
        )

    def _inverse_block(
        self,
        layer: int,
        upper_state: Tensor,
    ) -> _InverseAttentionCache:
        """Learned causal inverse used only for local target propagation."""
        source = _rms_norm(upper_state)
        query_weights = (
            self.inverse_query_weights[layer]
            if self.config.inference_mode in {
                "bidirectional_pc",
                "residual_error_pc",
            }
            else self.query_weights[layer]
        )
        key_weights = (
            self.inverse_key_weights[layer]
            if self.config.inference_mode in {
                "bidirectional_pc",
                "residual_error_pc",
            }
            else self.key_weights[layer]
        )
        query = self._split_heads(
            F.linear(source, query_weights)
        )
        key = self._split_heads(
            F.linear(source, key_weights)
        )
        if self.config.block_state_mode == "nanochat_residual":
            cosine, sine = self._rotary(upper_state.shape[1], query)
            query = 1.2 * _rms_norm(
                _apply_rotary(query, cosine, sine)
            )
            key = 1.2 * _rms_norm(
                _apply_rotary(key, cosine, sine)
            )
        value = self._split_heads(
            F.linear(source, self.generative_weights[layer])
        )
        score_scale = 1.0 / (
            math.sqrt(self.head_dim) * self.config.attention_temperature
        )
        scores = torch.matmul(query, key.transpose(-2, -1)) * score_scale
        attention_mask = self._causal_mask(
            upper_state.shape[1],
            upper_state.device,
            layer=layer,
        )
        scores.masked_fill_(attention_mask, -torch.inf)
        weights = F.softmax(scores, dim=-1)
        message = self._merge_heads(torch.matmul(weights, value))
        prediction = upper_state + F.linear(
            message,
            self.feedback_weights[layer],
        )
        return _InverseAttentionCache(
            source=source,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            weights=weights,
            message=message,
            prediction=prediction,
        )

    def _inverse_parameter_gradients(
        self,
        layer: int,
        cache: _InverseAttentionCache,
        output_signal: Tensor,
        *,
        token_count: float,
    ) -> dict[str, Tensor]:
        flat_signal = output_signal.reshape(-1, self.config.hidden_dim)
        flat_message = cache.message.reshape(-1, self.config.hidden_dim)
        message_signal = F.linear(
            output_signal,
            self.feedback_weights[layer].t(),
        )
        value_signal = torch.matmul(
            cache.weights.transpose(-2, -1),
            self._split_heads(message_signal),
        )
        flat_value_signal = self._merge_heads(value_signal).reshape(
            -1,
            self.config.hidden_dim,
        )
        flat_source = cache.source.reshape(-1, self.config.hidden_dim)
        return {
            f"feedback_weights.{layer}": (
                flat_signal.t() @ flat_message
            ) / token_count,
            f"generative_weights.{layer}": (
                flat_value_signal.t() @ flat_source
            ) / token_count,
        }

    def _inverse_input_signal(
        self,
        layer: int,
        upper_state: Tensor,
        cache: _InverseAttentionCache,
        output_signal: Tensor,
    ) -> Tensor:
        """Exact local VJP through one top-down attention prediction."""
        if self.config.block_state_mode != "identity_residual":
            raise ValueError(
                "inverse input VJP currently requires identity_residual blocks"
            )
        message_signal = F.linear(
            output_signal,
            self.feedback_weights[layer].t(),
        )
        head_message_signal = self._split_heads(message_signal)
        weight_signal = torch.matmul(
            head_message_signal,
            cache.value.transpose(-2, -1),
        )
        centered_weight_signal = weight_signal - (
            cache.weights * weight_signal
        ).sum(dim=-1, keepdim=True)
        score_signal = cache.weights * centered_weight_signal
        score_signal.masked_fill_(cache.attention_mask, 0.0)
        score_scale = 1.0 / (
            math.sqrt(self.head_dim) * self.config.attention_temperature
        )
        query_signal = torch.matmul(score_signal, cache.key) * score_scale
        key_signal = (
            torch.matmul(score_signal.transpose(-2, -1), cache.query)
            * score_scale
        )
        value_signal = torch.matmul(
            cache.weights.transpose(-2, -1),
            head_message_signal,
        )
        query_weights = (
            self.inverse_query_weights[layer]
            if self.config.inference_mode in {
                "bidirectional_pc",
                "residual_error_pc",
            }
            else self.query_weights[layer]
        )
        key_weights = (
            self.inverse_key_weights[layer]
            if self.config.inference_mode in {
                "bidirectional_pc",
                "residual_error_pc",
            }
            else self.key_weights[layer]
        )
        source_signal = F.linear(
            self._merge_heads(query_signal), query_weights.t()
        )
        source_signal.add_(
            F.linear(self._merge_heads(key_signal), key_weights.t())
        )
        source_signal.add_(
            F.linear(
                self._merge_heads(value_signal),
                self.generative_weights[layer].t(),
            )
        )
        return output_signal + _rms_norm_vjp(upper_state, source_signal)

    def _bidirectional_inverse_parameter_gradients(
        self,
        layer: int,
        cache: _InverseAttentionCache,
        output_signal: Tensor,
        *,
        token_count: float,
    ) -> dict[str, Tensor]:
        """Local gradients for one generative attention edge."""
        message_signal = F.linear(
            output_signal,
            self.feedback_weights[layer].t(),
        )
        head_message_signal = self._split_heads(message_signal)
        weight_signal = torch.matmul(
            head_message_signal,
            cache.value.transpose(-2, -1),
        )
        centered_weight_signal = weight_signal - (
            cache.weights * weight_signal
        ).sum(dim=-1, keepdim=True)
        score_signal = cache.weights * centered_weight_signal
        score_signal.masked_fill_(cache.attention_mask, 0.0)
        score_scale = 1.0 / (
            math.sqrt(self.head_dim) * self.config.attention_temperature
        )
        query_signal = torch.matmul(score_signal, cache.key) * score_scale
        key_signal = (
            torch.matmul(score_signal.transpose(-2, -1), cache.query)
            * score_scale
        )
        value_signal = torch.matmul(
            cache.weights.transpose(-2, -1),
            head_message_signal,
        )
        flat_source = cache.source.reshape(-1, self.config.hidden_dim)
        return {
            f"inverse_query_weights.{layer}": (
                self._merge_heads(query_signal).reshape(
                    -1, self.config.hidden_dim
                ).t()
                @ flat_source
            ) / token_count,
            f"inverse_key_weights.{layer}": (
                self._merge_heads(key_signal).reshape(
                    -1, self.config.hidden_dim
                ).t()
                @ flat_source
            ) / token_count,
            f"generative_weights.{layer}": (
                self._merge_heads(value_signal).reshape(
                    -1, self.config.hidden_dim
                ).t()
                @ flat_source
            ) / token_count,
            f"feedback_weights.{layer}": (
                output_signal.reshape(-1, self.config.hidden_dim).t()
                @ cache.message.reshape(-1, self.config.hidden_dim)
            ) / token_count,
        }

    def _block_internal_signals(
        self,
        layer: int,
        cache: _AttentionCache,
        output_signal: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if self.config.ffn_dim <= 0:
            attention_residual_signal = (
                output_signal * cache.attention_prediction_derivative
            )
            lower_direct_signal = (
                attention_residual_signal
                if self.config.block_state_mode == "squashed_state"
                else output_signal
            )
            return attention_residual_signal, lower_direct_signal, None, None

        if (
            cache.ffn_source is None
            or cache.ffn_pre_activation is None
            or cache.ffn_activation is None
        ):
            raise RuntimeError("FFN cache is incomplete")
        ffn_residual_signal = output_signal * cache.prediction_derivative
        ffn_direct_signal = output_signal
        ffn_activation_signal = F.linear(
            ffn_residual_signal,
            self.ffn_down_weights[layer].t(),
        )
        ffn_pre_activation_signal = ffn_activation_signal * (
            2.0 * F.relu(cache.ffn_pre_activation)
        )
        ffn_source_signal = F.linear(
            ffn_pre_activation_signal,
            self.ffn_up_weights[layer].t(),
        )
        attention_output_signal = ffn_direct_signal + _rms_norm_vjp(
            cache.attention_prediction,
            ffn_source_signal,
        )
        attention_residual_signal = (
            attention_output_signal * cache.attention_prediction_derivative
        )
        lower_direct_signal = (
            attention_residual_signal
            if self.config.block_state_mode == "squashed_state"
            else attention_output_signal
        )
        return (
            attention_residual_signal,
            lower_direct_signal,
            ffn_residual_signal,
            ffn_pre_activation_signal,
        )

    def _attention_local_gradients(
        self,
        cache: _AttentionCache | _CrossAttentionCache,
        head_message_gradient: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        weight_gradient = torch.matmul(head_message_gradient, cache.value.transpose(-2, -1))
        centered_weight_gradient = weight_gradient - (
            cache.weights * weight_gradient
        ).sum(dim=-1, keepdim=True)
        score_gradient = cache.weights * centered_weight_gradient
        score_gradient.masked_fill_(
            cache.attention_mask,
            0.0,
        )
        score_scale = 1.0 / (
            math.sqrt(self.head_dim) * self.config.attention_temperature
        )
        query_gradient = torch.matmul(score_gradient, cache.key) * score_scale
        key_gradient = torch.matmul(score_gradient.transpose(-2, -1), cache.query) * score_scale
        value_gradient = torch.matmul(cache.weights.transpose(-2, -1), head_message_gradient)
        return query_gradient, key_gradient, value_gradient, score_gradient

    def _route_projection_signals(
        self,
        cache: _AttentionCache | _CrossAttentionCache,
        query_signal: Tensor,
        key_signal: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.config.block_state_mode != "nanochat_residual":
            return query_signal, key_signal
        query_pre_norm_signal = _rms_norm_vjp(
            cache.query_pre_norm,
            1.2 * query_signal,
        )
        key_pre_norm_signal = _rms_norm_vjp(
            cache.key_pre_norm,
            1.2 * key_signal,
        )
        cosine, sine = self._rotary(
            cache.query_projection.shape[2],
            cache.query_projection,
        )
        return (
            _apply_rotary_vjp(query_pre_norm_signal, cosine, sine),
            _apply_rotary_vjp(key_pre_norm_signal, cosine, sine),
        )

    def _cosine_route_teacher_gradients(
        self,
        cache: _AttentionCache,
        head_message_gradient: Tensor,
        *,
        causal_mask: Tensor,
        weight_by_credit: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        credit_strength = head_message_gradient.float().square().mean(
            dim=-1,
            keepdim=True,
        ).sqrt()
        desired_message = F.normalize(-head_message_gradient, dim=-1, eps=1e-6)
        normalized_value = F.normalize(cache.value, dim=-1, eps=1e-6)
        teacher_scores = torch.matmul(
            desired_message,
            normalized_value.transpose(-2, -1),
        )
        teacher_scores.div_(self.config.route_teacher_temperature)
        teacher_scores.masked_fill_(causal_mask, -torch.inf)
        teacher_weights = F.softmax(teacher_scores, dim=-1)
        route_gradient = cache.weights - teacher_weights
        route_tv_by_query = 0.5 * route_gradient.abs().sum(dim=-1)
        if weight_by_credit:
            relative_credit = credit_strength / credit_strength.mean().clamp_min(1e-8)
            relative_credit.clamp_(max=4.0)
            route_gradient = route_gradient * relative_credit.to(route_gradient.dtype)
            route_tv = (
                route_tv_by_query * relative_credit.squeeze(-1)
            ).sum() / relative_credit.sum().clamp_min(1.0)
        else:
            route_tv = route_tv_by_query.mean()
        route_score_scale = 1.0 / (
            math.sqrt(self.head_dim) * self.config.attention_temperature
        )
        query_gradient = torch.matmul(route_gradient, cache.key) * route_score_scale
        key_gradient = (
            torch.matmul(route_gradient.transpose(-2, -1), cache.query)
            * route_score_scale
        )
        return query_gradient, key_gradient, route_tv

    def _block_input_signal(
        self,
        layer: int,
        lower_state: Tensor,
        cache: _AttentionCache,
        output_signal: Tensor,
    ) -> Tensor:
        attention_residual_signal, lower_direct_signal, _, _ = (
            self._block_internal_signals(
                layer,
                cache,
                output_signal,
            )
        )
        if self.config.block_state_mode == "nanochat_residual":
            source_signal = torch.zeros_like(lower_state)
            message_signal = F.linear(
                attention_residual_signal,
                self.message_weights[layer].t(),
            )
        else:
            scale = self._block_weight_scale() / math.sqrt(2.0)
            source_signal = scale * F.linear(
                attention_residual_signal,
                self.self_weights[layer].t(),
            )
            message_signal = scale * F.linear(
                attention_residual_signal,
                self.message_weights[layer].t(),
            )
        head_message_signal = self._split_heads(message_signal)
        query_signal, key_signal, value_signal, _ = self._attention_local_gradients(
            cache,
            head_message_signal,
        )
        query_signal, key_signal = self._route_projection_signals(
            cache,
            query_signal,
            key_signal,
        )
        source_signal = source_signal + F.linear(
            self._merge_heads(query_signal),
            self.query_weights[layer].t(),
        )
        source_signal = source_signal + F.linear(
            self._merge_heads(key_signal),
            self.key_weights[layer].t(),
        )
        source_signal = source_signal + F.linear(
            self._merge_heads(value_signal),
            self.value_weights[layer].t(),
        )
        memory_input_signal = torch.zeros_like(lower_state)
        if cache.cross_attention is not None:
            cross = cache.cross_attention
            cross_message_signal = F.linear(
                cross.gate * attention_residual_signal,
                self.cross_message_weights[layer].t(),
            )
            (
                cross_query_signal,
                cross_key_signal,
                cross_value_signal,
                _,
            ) = self._attention_local_gradients(
                cross,
                self._split_heads(cross_message_signal),
            )
            cross_query_signal, cross_key_signal = (
                self._route_projection_signals(
                    cross,
                    cross_query_signal,
                    cross_key_signal,
                )
            )
            source_signal = source_signal + F.linear(
                self._merge_heads(cross_query_signal),
                self.cross_query_weights[layer].t(),
            )
            if cross.memory_is_lower_state:
                memory_input_signal = self._cross_memory_input_signal(
                    layer,
                    cross,
                    cross_key_signal,
                    cross_value_signal,
                )
        return memory_input_signal + (
            cache.direct_input_scale * lower_direct_signal
            + cache.source_input_scale
            * _rms_norm_vjp(cache.source_input, source_signal)
        )

    def _cross_memory_input_signal(
        self,
        layer: int,
        cross: _CrossAttentionCache,
        key_signal: Tensor,
        value_signal: Tensor,
    ) -> Tensor:
        memory_signal = F.linear(
            self._merge_heads(key_signal),
            self.cross_key_weights[layer].t(),
        )
        memory_signal = memory_signal + F.linear(
            self._merge_heads(value_signal),
            self.cross_value_weights[layer].t(),
        )
        return _rms_norm_vjp(cross.memory_input, memory_signal)

    def _block_cross_memory_signal(
        self,
        layer: int,
        cache: _AttentionCache,
        output_signal: Tensor,
    ) -> Tensor:
        cross = cache.cross_attention
        if cross is None:
            return torch.zeros_like(output_signal)
        attention_residual_signal, _, _, _ = self._block_internal_signals(
            layer,
            cache,
            output_signal,
        )
        cross_message_signal = F.linear(
            cross.gate * attention_residual_signal,
            self.cross_message_weights[layer].t(),
        )
        _, key_signal, value_signal, _ = self._attention_local_gradients(
            cross,
            self._split_heads(cross_message_signal),
        )
        _, key_signal = self._route_projection_signals(
            cross,
            torch.zeros_like(key_signal),
            key_signal,
        )
        return self._cross_memory_input_signal(
            layer,
            cross,
            key_signal,
            value_signal,
        )

    def _forward_states(
        self,
        x: Tensor,
        *,
        with_cache: bool,
        layer_count: int | None = None,
    ) -> tuple[list[Tensor], list[_AttentionCache]]:
        token_state = self.embedding(x)
        if self.config.state_prediction_separation:
            if self.prediction_token is None:
                raise RuntimeError("prediction token is not initialized")
            prediction_state = self.prediction_token.view(1, 1, -1) + (
                self.config.sps_prediction_input_scale * token_state
            )
            token_state = torch.stack(
                (token_state, prediction_state),
                dim=2,
            ).reshape(
                token_state.shape[0],
                token_state.shape[1] * 2,
                token_state.shape[2],
            )
        if self.config.block_state_mode == "nanochat_residual":
            state = _rms_norm(token_state)
        else:
            position = self._position(token_state.shape[1], token_state).unsqueeze(0)
            state = torch.tanh(token_state + self.config.position_scale * position)
        states = [state]
        caches: list[_AttentionCache] = []
        active_layer_count = (
            (
                self.config.recurrent_shared_steps
                if self.config.inference_mode == "recurrent_local"
                else self.config.num_layers
            )
            if layer_count is None
            else layer_count
        )
        for layer in range(active_layer_count):
            block_layer = (
                0
                if self.config.inference_mode == "recurrent_local"
                else layer
            )
            refinement_scale = (
                self.config.recurrent_step_scale
                if self.config.inference_mode == "recurrent_local" and layer > 0
                else 1.0
            )
            cache = self._block(
                block_layer,
                states[-1],
                branch_scale=refinement_scale,
                token_ids=x,
                sensory_state=states[0],
                encoder_memory=self._causal_encoder_memory(
                    states,
                    block_layer,
                ),
            )
            states.append(cache.prediction)
            if with_cache:
                caches.append(cache)
        return states, caches

    def _embedding_state_vjp(
        self,
        x: Tensor,
        free_state: Tensor,
        output_signal: Tensor,
    ) -> Tensor:
        embedding_source = self.embedding(x)
        if not self.config.state_prediction_separation:
            if self.config.block_state_mode == "nanochat_residual":
                return _rms_norm_vjp(embedding_source, output_signal)
            return output_signal * (1.0 - free_state.square())

        state_signal = output_signal[:, 0::2]
        if self.config.block_state_mode == "nanochat_residual":
            embedding_signal = _rms_norm_vjp(embedding_source, state_signal)
        else:
            embedding_signal = state_signal * (
                1.0 - free_state[:, 0::2].square()
            )
        prediction_input_scale = self.config.sps_prediction_input_scale
        if prediction_input_scale == 0:
            return embedding_signal
        prediction_signal = output_signal[:, 1::2]
        if self.config.block_state_mode == "nanochat_residual":
            if self.prediction_token is None:
                raise RuntimeError("prediction token is not initialized")
            prediction_source = self.prediction_token.view(1, 1, -1) + (
                prediction_input_scale * embedding_source
            )
            prediction_signal = _rms_norm_vjp(
                prediction_source,
                prediction_signal,
            )
        else:
            prediction_signal = prediction_signal * (
                1.0 - free_state[:, 1::2].square()
            )
        return embedding_signal + prediction_input_scale * prediction_signal

    def _prediction_token_state_vjp(
        self,
        x: Tensor,
        free_state: Tensor,
        output_signal: Tensor,
    ) -> Tensor:
        if not self.config.state_prediction_separation:
            return free_state.new_zeros(self.config.hidden_dim)
        if self.prediction_token is None:
            raise RuntimeError("prediction token is not initialized")
        prediction_signal = output_signal[:, 1::2]
        if self.config.block_state_mode == "nanochat_residual":
            prediction_source = self.prediction_token.view(1, 1, -1) + (
                self.config.sps_prediction_input_scale * self.embedding(x)
            )
            prediction_signal = _rms_norm_vjp(
                prediction_source,
                prediction_signal,
            )
        else:
            prediction_signal = prediction_signal * (
                1.0 - free_state[:, 1::2].square()
            )
        return prediction_signal.sum(dim=(0, 1))

    @staticmethod
    def _causal_affine_scan(retain: Tensor, innovation: Tensor) -> Tensor:
        """Parallel inclusive scan for h[t] = retain[t] * h[t-1] + innovation[t]."""
        if innovation.shape[1] == 0:
            return innovation
        transition = retain
        value = innovation
        offset = 1
        while offset < innovation.shape[1]:
            next_transition = transition.clone()
            next_value = value.clone()
            next_transition[:, offset:] = (
                transition[:, offset:] * transition[:, :-offset]
            )
            next_value[:, offset:] = value[:, offset:] + (
                transition[:, offset:] * value[:, :-offset]
            )
            transition = next_transition
            value = next_value
            offset *= 2
        return value

    def _hierarchical_error_context_cache(
        self,
        states: list[Tensor],
        *,
        candidates_override: list[Tensor] | None = None,
    ) -> _HierarchicalContextCache:
        if self.config.hierarchical_error_context_mode == "instant":
            raise RuntimeError("persistent hierarchical context is not enabled")
        base_depth = self.config.hierarchical_error_base_depth
        candidates: list[Tensor] = []
        memories: list[Tensor] = []
        contexts: list[Tensor] = []
        previous_contexts: list[Tensor] = []
        write_gates: list[Tensor] = []
        write_probabilities: list[Tensor] = []
        eligibility_features: list[Tensor] = []
        eligibility_biases: list[Tensor] = []
        feature_scale = 1.0 / math.sqrt(self.config.hidden_dim)
        context_candidates = (
            [
                states[layer + 1]
                for layer in range(base_depth, self.config.num_layers)
            ]
            if candidates_override is None
            else candidates_override
        )
        if len(context_candidates) != len(self.hierarchical_gate_weights):
            raise ValueError("hierarchical context candidate count is incorrect")
        for offset, candidate in enumerate(context_candidates):
            temporal_candidate = (
                candidate[:, 0::2]
                if self.config.state_prediction_separation
                else candidate
            )
            gate_feature = _rms_norm(temporal_candidate) * feature_scale
            gate_score = (
                gate_feature * self.hierarchical_gate_weights[offset]
            ).sum(dim=-1, keepdim=True)
            gate_score = gate_score + self.hierarchical_gate_biases[offset]
            learned_probability = torch.sigmoid(gate_score)
            write_probability = torch.cat(
                (
                    torch.ones_like(learned_probability[:, :1]),
                    learned_probability[:, 1:],
                ),
                dim=1,
            )
            if self.config.hierarchical_error_context_mode == "persistent_hard":
                positions = torch.arange(
                    temporal_candidate.shape[1],
                    device=candidate.device,
                    dtype=torch.float32,
                )
                threshold = torch.remainder(
                    (positions + 1.0) * 0.6180339887498949
                    + (offset + 1.0) * 0.4142135623730950,
                    1.0,
                ).view(1, -1, 1)
                write_gate = (write_probability.float() >= threshold).to(
                    candidate.dtype
                )
                write_gate[:, :1] = 1.0
            else:
                write_gate = write_probability
            retain = 1.0 - write_gate
            memory = self._causal_affine_scan(
                retain,
                write_gate * temporal_candidate,
            )
            if self.config.hierarchical_error_memory_scale == 0:
                context = temporal_candidate
            elif self.config.hierarchical_error_memory_scale == 1:
                context = memory
            else:
                context = temporal_candidate + (
                    self.config.hierarchical_error_memory_scale
                    * (memory - temporal_candidate)
                )
            previous_context = torch.cat(
                (torch.zeros_like(memory[:, :1]), memory[:, :-1]),
                dim=1,
            )
            eligibility_feature = self._causal_affine_scan(
                retain,
                gate_feature,
            )
            eligibility_bias = self._causal_affine_scan(
                retain,
                torch.ones_like(write_gate),
            )
            if self.config.state_prediction_separation:
                temporal_candidate = temporal_candidate.repeat_interleave(
                    2,
                    dim=1,
                )
                memory = memory.repeat_interleave(2, dim=1)
                context = context.repeat_interleave(2, dim=1)
                previous_context = previous_context.repeat_interleave(2, dim=1)
                write_gate = write_gate.repeat_interleave(2, dim=1)
                write_probability = write_probability.repeat_interleave(
                    2,
                    dim=1,
                )
                eligibility_feature = eligibility_feature.repeat_interleave(
                    2,
                    dim=1,
                )
                eligibility_bias = eligibility_bias.repeat_interleave(
                    2,
                    dim=1,
                )
            candidates.append(temporal_candidate)
            memories.append(memory)
            contexts.append(context)
            previous_contexts.append(previous_context)
            write_gates.append(write_gate)
            write_probabilities.append(write_probability)
            eligibility_features.append(eligibility_feature)
            eligibility_biases.append(eligibility_bias)
        return _HierarchicalContextCache(
            candidates=candidates,
            memories=memories,
            contexts=contexts,
            previous_contexts=previous_contexts,
            write_gates=write_gates,
            write_probabilities=write_probabilities,
            eligibility_features=eligibility_features,
            eligibility_biases=eligibility_biases,
        )

    @staticmethod
    def _hierarchical_error_binding_key(context: Tensor) -> Tensor:
        """Distributed, exactly reversible replacement for HER's outer product."""
        return torch.where(context >= 0, 1.0, -1.0).to(context.dtype)

    @staticmethod
    def _hierarchical_error_unitary_key(context: Tensor) -> Tensor:
        """Return unit-modulus Fourier keys for reversible holographic binding."""
        spectrum = torch.fft.rfft(context.float(), dim=-1)
        magnitude = spectrum.abs()
        return torch.where(
            magnitude > 1e-8,
            spectrum / magnitude.clamp_min(1e-8),
            torch.ones_like(spectrum),
        )

    def _hierarchical_error_bind(self, context: Tensor, error: Tensor) -> Tensor:
        if self.config.hierarchical_error_binding_mode == "identity":
            return error
        if self.config.hierarchical_error_binding_mode == "unitary":
            key = self._hierarchical_error_unitary_key(context)
            bound = torch.fft.irfft(
                key * torch.fft.rfft(error.float(), dim=-1),
                n=error.shape[-1],
                dim=-1,
            )
            return bound.to(error.dtype)
        return self._hierarchical_error_binding_key(context) * error

    def _hierarchical_error_unbind(self, context: Tensor, code: Tensor) -> Tensor:
        if self.config.hierarchical_error_binding_mode == "identity":
            return code
        if self.config.hierarchical_error_binding_mode == "unitary":
            key = self._hierarchical_error_unitary_key(context)
            unbound = torch.fft.irfft(
                key.conj() * torch.fft.rfft(code.float(), dim=-1),
                n=code.shape[-1],
                dim=-1,
            )
            return unbound.to(code.dtype)
        return self._hierarchical_error_binding_key(context) * code

    def _hierarchical_error_pass(
        self,
        states: list[Tensor],
        *,
        contexts: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor], Tensor]:
        if self.config.hierarchical_error_binding_mode == "factorized":
            return self._hierarchical_error_factorized_pass(states)[:4]
        base_depth = self.config.hierarchical_error_base_depth
        error_layers = range(base_depth, self.config.num_layers)
        if contexts is None:
            if self.config.hierarchical_error_context_mode != "instant":
                contexts = self._hierarchical_error_context_cache(states).contexts
            else:
                contexts = [states[layer + 1] for layer in error_layers]
        predictions = [
            F.linear(
                _rms_norm(contexts[offset]),
                self.generative_weights[layer],
            )
            for offset, layer in enumerate(error_layers)
        ]
        modulated_predictions = [prediction.clone() for prediction in predictions]
        scale = self.config.hierarchical_error_top_down_scale
        for offset in range(len(predictions) - 2, -1, -1):
            higher_correction = self._hierarchical_error_unbind(
                contexts[offset],
                modulated_predictions[offset + 1],
            )
            modulated_predictions[offset].add_(higher_correction, alpha=scale)
        base_correction = self._hierarchical_error_unbind(
            states[base_depth],
            modulated_predictions[0],
        )
        base_correction = _cap_token_rms(
            base_correction,
            self.config.state_delta_rms_cap,
        )
        modulated_state = states[base_depth] + scale * base_correction
        return (
            modulated_state,
            predictions,
            modulated_predictions,
            base_correction,
        )

    def _node_residual_hierarchy_cache(
        self,
        states: list[Tensor],
    ) -> _NodeResidualHierarchyCache:
        """Build causal working memories of adjacent-node residual predictions."""
        base_depth = self.config.hierarchical_error_base_depth
        candidates: list[Tensor] = []
        pre_gate_predictions: list[Tensor] = []
        predictions: list[Tensor] = []
        memories: list[Tensor] = []
        previous_memories: list[Tensor] = []
        write_gates: list[Tensor] = []
        write_probabilities: list[Tensor] = []
        gate_features: list[Tensor] = []
        feature_scale = 1.0 / math.sqrt(self.config.hidden_dim)

        for offset, layer in enumerate(range(base_depth, self.config.num_layers)):
            candidate = (
                states[layer + 1][:, 1::2] - states[layer][:, 1::2]
            )
            context = states[layer][:, 0::2]
            feature = _rms_norm(context) * feature_scale
            if self.config.hierarchical_error_context_mode == "instant":
                write_probability = torch.ones_like(candidate[..., :1])
                write_gate = write_probability
                memory = candidate
                pre_gate_prediction = candidate
                prediction = pre_gate_prediction
            else:
                gate_score = (
                    feature * self.hierarchical_gate_weights[offset]
                ).sum(dim=-1, keepdim=True)
                gate_score = gate_score + self.hierarchical_gate_biases[offset]
                learned_probability = torch.sigmoid(gate_score)
                write_probability = torch.cat(
                    (
                        torch.ones_like(learned_probability[:, :1]),
                        learned_probability[:, 1:],
                    ),
                    dim=1,
                )
                if (
                    self.config.hierarchical_error_context_mode
                    == "persistent_hard"
                ):
                    positions = torch.arange(
                        candidate.shape[1],
                        device=candidate.device,
                        dtype=torch.float32,
                    )
                    threshold = torch.remainder(
                        (positions + 1.0) * 0.6180339887498949
                        + (offset + 1.0) * 0.4142135623730950,
                        1.0,
                    ).view(1, -1, 1)
                    write_gate = (
                        write_probability.float() >= threshold
                    ).to(candidate.dtype)
                    write_gate[:, :1] = 1.0
                else:
                    write_gate = write_probability
                memory = self._causal_affine_scan(
                    1.0 - write_gate,
                    write_gate * candidate,
                )
                memory_scale = self.config.hierarchical_error_memory_scale
                pre_gate_prediction = candidate + memory_scale * (
                    memory - candidate
                )
                prediction = write_probability * pre_gate_prediction
            previous_memory = torch.cat(
                (torch.zeros_like(memory[:, :1]), memory[:, :-1]),
                dim=1,
            )
            candidates.append(candidate)
            pre_gate_predictions.append(pre_gate_prediction)
            predictions.append(prediction)
            memories.append(memory)
            previous_memories.append(previous_memory)
            write_gates.append(write_gate)
            write_probabilities.append(write_probability)
            gate_features.append(feature)

        return _NodeResidualHierarchyCache(
            candidates=candidates,
            pre_gate_predictions=pre_gate_predictions,
            predictions=predictions,
            memories=memories,
            previous_memories=previous_memories,
            write_gates=write_gates,
            write_probabilities=write_probabilities,
            gate_features=gate_features,
        )

    def _node_residual_memory_vjp(
        self,
        cache: _NodeResidualHierarchyCache,
        level: int,
        output_signal: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Route a level-local signal through its causal residual memory."""
        if self.config.hierarchical_error_context_mode == "instant":
            return output_signal, torch.zeros_like(output_signal[..., :1])
        memory_scale = self.config.hierarchical_error_memory_scale
        if memory_scale == 0:
            return output_signal, torch.zeros_like(output_signal[..., :1])

        write_gate = cache.write_gates[level]
        write_probability = cache.write_probabilities[level]
        retain = 1.0 - write_gate
        pre_gate_signal = write_probability * output_signal
        memory_signal = memory_scale * pre_gate_signal
        reverse_transition = torch.zeros_like(retain)
        if retain.shape[1] > 1:
            reverse_transition[:, 1:] = retain.flip(1)[:, :-1]
        total_memory_signal = self._causal_affine_scan(
            reverse_transition,
            memory_signal.flip(1),
        ).flip(1)
        candidate_signal = (
            (1.0 - memory_scale) * pre_gate_signal
            + write_gate * total_memory_signal
        )
        write_gate_signal = (
            total_memory_signal
            * (
                cache.candidates[level]
                - cache.previous_memories[level]
            )
        ).sum(dim=-1, keepdim=True)
        memory_gate_score_signal = (
            write_gate_signal
            * write_probability
            * (1.0 - write_probability)
        )
        precision_gate_score_signal = (
            output_signal * cache.pre_gate_predictions[level]
        ).sum(dim=-1, keepdim=True) * write_probability * (
            1.0 - write_probability
        )
        gate_score_signal = (
            memory_gate_score_signal + precision_gate_score_signal
        )
        return candidate_signal, gate_score_signal

    def _node_residual_hierarchy_pass(
        self,
        states: list[Tensor],
    ) -> tuple[
        Tensor,
        list[Tensor],
        list[Tensor],
        Tensor,
        _NodeResidualHierarchyCache,
    ]:
        cache = self._node_residual_hierarchy_cache(states)
        base_depth = self.config.hierarchical_error_base_depth
        predictions = [
            self._expand_prediction_signal(prediction, states[base_depth])
            for prediction in cache.predictions
        ]
        correction = torch.zeros_like(states[base_depth])
        cumulative_predictions = [torch.empty(0)] * len(predictions)
        for level in range(len(predictions) - 1, -1, -1):
            correction = predictions[level] + correction
            cumulative_predictions[level] = correction.clone()
        deployed = states[base_depth] + correction
        return (
            deployed,
            predictions,
            cumulative_predictions,
            correction,
            cache,
        )

    def _hierarchical_error_factorized_pass(
        self,
        states: list[Tensor],
        *,
        contexts: list[Tensor] | None = None,
    ) -> tuple[
        Tensor,
        list[Tensor],
        list[Tensor],
        Tensor,
        list[Tensor],
        list[Tensor],
        list[Tensor],
    ]:
        """Predict residual errors through context-keyed low-rank skip paths."""
        if self.recursive_cp_her is not None:
            raw_memories, normalized_memories, cp_cache = (
                self._hierarchical_error_recursive_cp_cache(
                    states,
                    contexts=contexts,
                )
            )
            del raw_memories, normalized_memories
            base_state = states[self.config.hierarchical_error_base_depth]
            shape = base_state.shape
            predictions = [
                contribution.reshape(shape)
                for contribution in self.recursive_cp_her.base_contributions(
                    cp_cache
                )
            ]
            cumulative = torch.zeros_like(predictions[0])
            modulated_predictions = [torch.empty(0)] * len(predictions)
            for offset in range(len(predictions) - 1, -1, -1):
                cumulative = predictions[offset] + cumulative
                modulated_predictions[offset] = cumulative
            base_correction = _cap_token_rms(
                cp_cache.correction.reshape(shape),
                self.config.state_delta_rms_cap,
            )
            deployed = base_state + base_correction
            zero_keys = [torch.zeros_like(prediction) for prediction in predictions]
            unit_gates = [
                prediction.new_ones(
                    (*prediction.shape[:-1], self.config.num_heads, 1)
                )
                for prediction in predictions
            ]
            return (
                deployed,
                predictions,
                modulated_predictions,
                base_correction,
                zero_keys,
                predictions,
                unit_gates,
            )
        if self.config.sps_node_residual_hierarchy:
            (
                deployed,
                predictions,
                cumulative_predictions,
                correction,
                node_cache,
            ) = self._node_residual_hierarchy_pass(states)
            zero_keys = [torch.zeros_like(value) for value in predictions]
            candidate_values = [
                self._expand_prediction_signal(candidate, states[layer + 1])
                for candidate, layer in zip(
                    node_cache.candidates,
                    range(
                        self.config.hierarchical_error_base_depth,
                        self.config.num_layers,
                    ),
                )
            ]
            return (
                deployed,
                predictions,
                cumulative_predictions,
                correction,
                zero_keys,
                candidate_values,
                node_cache.write_gates,
            )
        base_depth = self.config.hierarchical_error_base_depth
        predictions: list[Tensor] = []
        keys: list[Tensor] = []
        values: list[Tensor] = []
        gates: list[Tensor] = []
        score_scale = 1.0 / math.sqrt(self.head_dim)
        error_layers = range(base_depth, self.config.num_layers)
        if contexts is None:
            if self.config.hierarchical_error_context_mode != "instant":
                contexts = self._hierarchical_error_context_cache(
                    states,
                    candidates_override=[
                        self._state_context(states[layer])
                        for layer in error_layers
                    ],
                ).contexts
            else:
                contexts = [
                    self._state_context(states[layer])
                    for layer in error_layers
                ]
        for offset, layer in enumerate(error_layers):
            source = _rms_norm(
                self._prediction_only(states[layer + 1])
            )
            value = F.linear(source, self.generative_weights[layer])
            if self.config.hierarchical_error_prediction_mode == "direct":
                key = torch.zeros_like(value)
                gate = value.new_ones(
                    (*value.shape[:-1], self.config.num_heads, 1)
                )
                prediction = value
            else:
                key = F.linear(source, self.feedback_weights[layer])
            if self.config.hierarchical_error_outer_product_contract:
                normalized_context = _rms_norm(contexts[offset])
                gate = (normalized_context * key).mean(
                    dim=-1,
                    keepdim=True,
                )
                prediction = gate * value
            elif self.config.hierarchical_error_prediction_mode == "gated":
                context_heads = self._split_heads(_rms_norm(contexts[offset]))
                key_heads = self._split_heads(key)
                gate = 2.0 * torch.sigmoid(
                    (context_heads * key_heads).sum(dim=-1, keepdim=True)
                    * score_scale
                )
                prediction = self._merge_heads(
                    gate * self._split_heads(value)
                )
            predictions.append(prediction)
            keys.append(key)
            values.append(value)
            gates.append(gate)

        cumulative = torch.zeros_like(predictions[0])
        modulated_predictions = [torch.empty(0)] * len(predictions)
        for offset in range(len(predictions) - 1, -1, -1):
            cumulative = predictions[offset] + cumulative
            modulated_predictions[offset] = cumulative
        base_correction = _cap_token_rms(
            modulated_predictions[0],
            self.config.state_delta_rms_cap,
        )
        modulated_state = states[base_depth] + (
            self.config.hierarchical_error_top_down_scale * base_correction
        )
        return (
            modulated_state,
            predictions,
            modulated_predictions,
            base_correction,
            keys,
            values,
            gates,
        )

    def _hierarchical_error_recursive_cp_cache(
        self,
        states: list[Tensor],
        *,
        contexts: list[Tensor] | None,
    ) -> tuple[list[Tensor], list[Tensor], BoundedCPHERCache]:
        if self.recursive_cp_her is None:
            raise RuntimeError("recursive CP HER is disabled")
        base_depth = self.config.hierarchical_error_base_depth
        if contexts is None:
            raw_memories = [
                self._state_context(states[layer])
                for layer in range(base_depth, self.config.num_layers)
            ]
        else:
            raw_memories = list(contexts)
        normalized_memories = [_rms_norm(memory) for memory in raw_memories]
        flat_memories = [
            memory.reshape(-1, self.config.hidden_dim)
            for memory in normalized_memories
        ]
        return (
            raw_memories,
            normalized_memories,
            self.recursive_cp_her.predict(flat_memories),
        )

    @staticmethod
    def _hierarchical_error_residual_handoff(
        target: Tensor,
        predictions: list[Tensor],
    ) -> tuple[list[Tensor], Tensor]:
        outcomes: list[Tensor] = []
        residual = target
        for prediction in predictions:
            outcomes.append(residual)
            residual = residual - prediction
        return outcomes, residual

    def _hierarchical_error_target(
        self,
        state: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        target, probabilities = self._decoder_negative_state_gradient(state, y)
        active_fraction = state.new_ones(())
        if self.config.hierarchical_error_target_mode == "fixed_outcome":
            target = F.embedding(
                y,
                _rms_norm(self.feedback_embedding),
            ) * self.config.hierarchical_error_outcome_code_scale
            if self.config.state_prediction_separation:
                target = self._expand_prediction_signal(target, state)
            return target, probabilities, active_fraction
        if self.config.hierarchical_error_target_mode == "mistakes":
            active = probabilities.argmax(dim=-1).ne(y)
            active_signal = active.unsqueeze(-1)
            if self.config.state_prediction_separation:
                active_signal = self._expand_prediction_signal(
                    active_signal,
                    target,
                )
            target = target * active_signal
            active_fraction = active.float().mean()
        return target, probabilities, active_fraction

    def _hierarchical_error_factorized_signals(
        self,
        *,
        context_state: Tensor,
        prediction: Tensor,
        key: Tensor,
        value: Tensor,
        gate: Tensor,
        target: Tensor,
        extra_prediction_gradient: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.config.hierarchical_error_prediction_mode == "direct":
            predictor_error = prediction - target
            value_signal = predictor_error
            if extra_prediction_gradient is not None:
                value_signal = value_signal + extra_prediction_gradient
            return (
                predictor_error,
                value_signal,
                torch.zeros_like(key),
                torch.zeros_like(context_state),
                torch.zeros_like(context_state),
            )
        if self.config.hierarchical_error_outer_product_contract:
            predictor_error = prediction - target
            output_gradient = predictor_error
            if extra_prediction_gradient is not None:
                output_gradient = output_gradient + extra_prediction_gradient
            normalized_context = _rms_norm(context_state)
            value_square = value.square().mean(dim=-1, keepdim=True)
            value_target = (value * target).mean(dim=-1, keepdim=True)
            key_square = key.square().mean(dim=-1, keepdim=True)
            key_context = (key * normalized_context).mean(
                dim=-1,
                keepdim=True,
            )
            key_signal = (
                key * value_square - normalized_context * value_target
            )
            value_signal = value * key_square - target * key_context
            output_score_signal = (output_gradient * value).mean(
                dim=-1,
                keepdim=True,
            )
            normalized_context_gradient = key * output_score_signal
            context_gradient = _rms_norm_vjp(
                context_state,
                normalized_context_gradient,
            )
            return (
                predictor_error,
                value_signal,
                key_signal,
                normalized_context_gradient,
                context_gradient,
            )
        predictor_error = prediction - target
        prediction_gradient = predictor_error
        if extra_prediction_gradient is not None:
            prediction_gradient = (
                prediction_gradient + extra_prediction_gradient
            )
        error_heads = self._split_heads(prediction_gradient)
        value_heads = self._split_heads(value)
        value_signal_heads = error_heads * gate
        gate_signal = (error_heads * value_heads).sum(
            dim=-1,
            keepdim=True,
        )
        score_signal = gate_signal * gate * (1.0 - 0.5 * gate)
        scale = 1.0 / math.sqrt(self.head_dim)
        context_heads = self._split_heads(_rms_norm(context_state))
        key_heads = self._split_heads(key)
        key_signal_heads = score_signal * context_heads * scale
        context_signal_heads = score_signal * key_heads * scale
        value_signal = self._merge_heads(value_signal_heads)
        key_signal = self._merge_heads(key_signal_heads)
        normalized_context_gradient = self._merge_heads(context_signal_heads)
        context_gradient = _rms_norm_vjp(
            context_state,
            normalized_context_gradient,
        )
        return (
            predictor_error,
            value_signal,
            key_signal,
            normalized_context_gradient,
            context_gradient,
        )

    def _settle_sps_node_residual_hierarchy(
        self,
        free_states: list[Tensor],
        y: Tensor,
        *,
        inference_lr: float,
    ) -> tuple[list[Tensor], dict[str, float]]:
        """Settle each node against the unpredicted error from the level below."""
        states = [state.clone() for state in free_states]
        initial_deployed = self._node_residual_hierarchy_pass(states)[0]
        initial_target, _, _ = self._hierarchical_error_target(
            initial_deployed,
            y,
        )
        initial_energy = self._decoder_output_energy(initial_deployed, y)
        prediction_error_rms = states[0].new_zeros(())
        outcome_rms = states[0].new_zeros(())
        predictor_rms = states[0].new_zeros(())
        predictor_error_rms = states[0].new_zeros(())
        prediction_target_cosine = states[0].new_zeros(())
        candidate_credit_rms = states[0].new_zeros(())
        final_residual_rms = states[0].new_zeros(())
        gate_score_signal_rms = states[0].new_zeros(())
        gate_write_rate = states[0].new_zeros(())
        gate_write_probability = states[0].new_zeros(())
        memory_change_rms = states[0].new_zeros(())
        base_depth = self.config.hierarchical_error_base_depth
        level_count = self.config.num_layers - base_depth

        for _ in range(self.config.inference_steps):
            deployed, _, _, _, node_cache = (
                self._node_residual_hierarchy_pass(states)
            )
            target, _, _ = self._hierarchical_error_target(deployed, y)
            deltas = [
                torch.zeros_like(states[layer + 1])
                for layer in range(self.config.num_layers)
            ]
            block_caches = [
                self._block(layer, states[layer])
                for layer in range(self.config.num_layers)
            ]
            for layer, block_cache in enumerate(block_caches):
                block_error = states[layer + 1] - block_cache.prediction
                deltas[layer].add_(
                    block_error,
                    alpha=-self.config.prediction_precision,
                )
                if layer > 0:
                    deltas[layer - 1].add_(
                        self._block_input_signal(
                            layer,
                            states[layer],
                            block_cache,
                            block_error,
                        ),
                        alpha=self.config.prediction_precision,
                    )
                prediction_error_rms = (
                    prediction_error_rms + _rms(block_error)
                )

            deltas[base_depth - 1].add_(target)
            remaining = self._prediction_states(target)
            for level, layer in enumerate(
                range(base_depth, self.config.num_layers)
            ):
                prediction = node_cache.predictions[level]
                predictor_error = prediction - remaining
                candidate_credit, gate_score_signal = (
                    self._node_residual_memory_vjp(
                        node_cache,
                        level,
                        -predictor_error,
                    )
                )
                expanded_credit = self._expand_prediction_signal(
                    candidate_credit,
                    states[layer + 1],
                )
                deltas[layer].add_(
                    expanded_credit,
                    alpha=self.config.hierarchical_error_context_credit_scale,
                )
                outcome_rms = outcome_rms + _rms(remaining)
                predictor_rms = predictor_rms + _rms(prediction)
                predictor_error_rms = (
                    predictor_error_rms + _rms(predictor_error)
                )
                prediction_target_cosine = (
                    prediction_target_cosine
                    + F.cosine_similarity(
                        prediction.float().reshape(
                            -1,
                            self.config.hidden_dim,
                        ),
                        remaining.float().reshape(
                            -1,
                            self.config.hidden_dim,
                        ),
                        dim=-1,
                        eps=1e-8,
                    ).mean()
                )
                candidate_credit_rms = (
                    candidate_credit_rms + _rms(candidate_credit)
                )
                gate_score_signal_rms = (
                    gate_score_signal_rms + _rms(gate_score_signal)
                )
                measured_gate = (
                    node_cache.write_gates[level][:, 1:]
                    if node_cache.write_gates[level].shape[1] > 1
                    else node_cache.write_gates[level]
                )
                measured_probability = (
                    node_cache.write_probabilities[level][:, 1:]
                    if node_cache.write_probabilities[level].shape[1] > 1
                    else node_cache.write_probabilities[level]
                )
                gate_write_rate = gate_write_rate + measured_gate.mean()
                gate_write_probability = (
                    gate_write_probability + measured_probability.mean()
                )
                memory_change_rms = memory_change_rms + _rms(
                    node_cache.memories[level]
                    - node_cache.previous_memories[level]
                )
                remaining = remaining - prediction
            final_residual_rms = final_residual_rms + _rms(remaining)

            for layer, delta in enumerate(deltas):
                states[layer + 1] = self._apply_inference_delta(
                    states[layer + 1],
                    _cap_token_rms(delta, self.config.state_delta_rms_cap),
                    inference_lr,
                )

        final_deployed, _, _, final_correction, _ = (
            self._node_residual_hierarchy_pass(states)
        )
        final_energy = self._decoder_output_energy(final_deployed, y)
        layer_steps = float(
            self.config.inference_steps * self.config.num_layers
        )
        level_steps = float(self.config.inference_steps * level_count)
        state_stream_delta_rms = torch.stack(
            [
                _rms(states[index][:, 0::2] - free_states[index][:, 0::2])
                for index in range(1, len(states))
            ]
        ).mean()
        prediction_stream_delta_rms = torch.stack(
            [
                _rms(states[index][:, 1::2] - free_states[index][:, 1::2])
                for index in range(1, len(states))
            ]
        ).mean()
        top_down_target_cosine = F.cosine_similarity(
            final_correction[:, 1::2].float().reshape(
                -1,
                self.config.hidden_dim,
            ),
            initial_target[:, 1::2].float().reshape(
                -1,
                self.config.hidden_dim,
            ),
            dim=-1,
            eps=1e-8,
        ).mean()
        diagnostics = {
            "update_diag_message_pc_feedback_rms": float(
                (candidate_credit_rms / level_steps).item()
            ),
            "update_diag_message_pc_prediction_error_rms": float(
                (prediction_error_rms / layer_steps).item()
            ),
            "update_diag_message_pc_phase_delta_rms": float(
                torch.stack(
                    [
                        _rms(states[index] - free_states[index])
                        for index in range(1, len(states))
                    ]
                ).mean().item()
            ),
            "update_diag_message_pc_her_settle_residual_rms": float(
                (predictor_error_rms / level_steps).item()
            ),
            "update_diag_message_pc_energy_before": float(initial_energy.item()),
            "update_diag_message_pc_energy_after": float(final_energy.item()),
            "update_diag_message_pc_energy_reduction": float(
                (initial_energy - final_energy).item()
            ),
            "update_diag_message_pc_effective_inference_lr": inference_lr,
            "update_diag_message_pc_her_settle_steps": float(
                self.config.inference_steps
            ),
            "update_diag_message_pc_her_outcome_rms": float(
                (outcome_rms / level_steps).item()
            ),
            "update_diag_message_pc_her_prediction_rms": float(
                (predictor_rms / level_steps).item()
            ),
            "update_diag_message_pc_her_prediction_target_cosine": float(
                (prediction_target_cosine / level_steps).item()
            ),
            "update_diag_message_pc_her_predictor_error_rms": float(
                (predictor_error_rms / level_steps).item()
            ),
            "update_diag_message_pc_her_final_residual_rms": float(
                (
                    final_residual_rms
                    / max(float(self.config.inference_steps), 1.0)
                ).item()
            ),
            "update_diag_message_pc_her_top_down_correction_rms": float(
                _rms(final_correction).item()
            ),
            "update_diag_message_pc_her_top_down_target_cosine": float(
                top_down_target_cosine.item()
            ),
            "update_diag_message_pc_her_gate_write_rate": float(
                (gate_write_rate / level_steps).item()
            ),
            "update_diag_message_pc_her_gate_write_probability": float(
                (gate_write_probability / level_steps).item()
            ),
            "update_diag_message_pc_her_gate_score_gradient_rms": float(
                (gate_score_signal_rms / level_steps).item()
            ),
            "update_diag_message_pc_her_gate_context_change_rms": float(
                (memory_change_rms / level_steps).item()
            ),
            "update_diag_message_pc_sps_state_stream_delta_rms": float(
                state_stream_delta_rms.item()
            ),
            "update_diag_message_pc_sps_prediction_stream_delta_rms": float(
                prediction_stream_delta_rms.item()
            ),
            "update_diag_message_pc_sps_future_state_credit_rms": 0.0,
            "update_diag_message_pc_sps_layer_token_credit_rms": 0.0,
            "update_diag_message_pc_sps_future_observation_credit_rms": 0.0,
            "update_diag_message_pc_sps_future_observation_cosine": 0.0,
            "update_diag_message_pc_uses_sps_node_residual_hierarchy": 1.0,
            "update_diag_message_pc_sps_node_candidate_credit_rms": float(
                (candidate_credit_rms / level_steps).item()
            ),
        }
        return states, diagnostics

    def _settle_hierarchical_factorized(
        self,
        free_states: list[Tensor],
        y: Tensor,
        *,
        inference_lr: float,
    ) -> tuple[list[Tensor], dict[str, float]]:
        if self.config.sps_node_residual_hierarchy:
            return self._settle_sps_node_residual_hierarchy(
                free_states,
                y,
                inference_lr=inference_lr,
            )
        states = [state.clone() for state in free_states]
        initial_deployed = self._hierarchical_error_factorized_pass(states)[0]
        initial_energy = self._decoder_output_energy(initial_deployed, y)
        feedback_rms = states[0].new_zeros(())
        prediction_rms = states[0].new_zeros(())
        residual_rms = states[0].new_zeros(())
        layer_token_credit_rms = states[0].new_zeros(())
        layer_token_credit_count = 0
        future_state_credit_rms = states[0].new_zeros(())
        future_state_credit_count = 0
        future_observation_credit_rms = states[0].new_zeros(())
        future_observation_cosine = states[0].new_zeros(())
        future_observation_credit_count = 0

        for _ in range(self.config.inference_steps):
            base_depth = self.config.hierarchical_error_base_depth
            error_layers = range(base_depth, self.config.num_layers)
            persistent_context_cache = None
            if self.config.hierarchical_error_context_mode != "instant":
                persistent_context_cache = (
                    self._hierarchical_error_context_cache(
                        states,
                        candidates_override=[
                            self._state_context(states[layer])
                            for layer in error_layers
                        ],
                    )
                )
            (
                deployed,
                predictions,
                _,
                _,
                keys,
                values,
                gates,
            ) = self._hierarchical_error_factorized_pass(
                states,
                contexts=(
                    persistent_context_cache.contexts
                    if persistent_context_cache is not None
                    else None
                ),
            )
            target, _, _ = self._hierarchical_error_target(deployed, y)
            deltas = [
                torch.zeros_like(states[layer + 1])
                for layer in range(self.config.num_layers)
            ]

            block_caches = [
                self._block(layer, states[layer])
                for layer in range(self.config.num_layers)
            ]
            block_errors = [
                states[layer + 1] - block_caches[layer].prediction
                for layer in range(self.config.num_layers)
            ]
            for layer, block_error in enumerate(block_errors):
                deltas[layer].add_(
                    block_error,
                    alpha=-self.config.prediction_precision,
                )
                if layer > 0:
                    deltas[layer - 1].add_(
                        self._block_input_signal(
                            layer,
                            states[layer],
                            block_caches[layer],
                            block_error,
                        ),
                        alpha=self.config.prediction_precision,
                    )
                prediction_rms = prediction_rms + _rms(block_error)

            observation_credit_scale = (
                self.config.sps_future_observation_credit_scale
            )
            if (
                self.config.state_prediction_separation
                and observation_credit_scale > 0
            ):
                for layer in range(self.config.num_layers):
                    prediction_state = states[layer + 1][:, 1::2][:, :-1]
                    observed_future_state = free_states[layer + 1][
                        :, 0::2
                    ][:, 1:]
                    temporal_residual = observed_future_state - prediction_state
                    temporal_credit = torch.zeros_like(states[layer + 1])
                    temporal_credit[:, 1:-1:2] = temporal_residual
                    deltas[layer].add_(
                        temporal_credit,
                        alpha=observation_credit_scale,
                    )
                    future_observation_credit_rms = (
                        future_observation_credit_rms + _rms(temporal_residual)
                    )
                    future_observation_cosine = (
                        future_observation_cosine
                        + F.cosine_similarity(
                            prediction_state.float().reshape(
                                -1,
                                self.config.hidden_dim,
                            ),
                            observed_future_state.float().reshape(
                                -1,
                                self.config.hidden_dim,
                            ),
                            dim=-1,
                            eps=1e-8,
                        ).mean()
                    )
                    future_observation_credit_count += 1

            layer_credit_scale = self.config.sps_layer_token_credit_scale
            future_credit_scale = self.config.sps_future_state_credit_scale
            if self.config.state_prediction_separation and (
                layer_credit_scale > 0 or future_credit_scale > 0
            ):
                for layer in range(1, self.config.num_layers):
                    upper_negative_gradient, _ = (
                        self._decoder_negative_state_gradient(
                            states[layer + 1],
                            y,
                        )
                    )
                    if layer_credit_scale > 0:
                        deltas[layer].add_(
                            upper_negative_gradient,
                            alpha=layer_credit_scale,
                        )
                        layer_token_credit_rms = (
                            layer_token_credit_rms
                            + _rms(upper_negative_gradient[:, 1::2])
                        )
                        layer_token_credit_count += 1
                    if future_credit_scale > 0:
                        lower_negative_gradient = self._persistent_state_only(
                            self._block_input_signal(
                                layer,
                                states[layer],
                                block_caches[layer],
                                upper_negative_gradient,
                            )
                        )
                        deltas[layer - 1].add_(
                            lower_negative_gradient,
                            alpha=future_credit_scale,
                        )
                        future_state_credit_rms = (
                            future_state_credit_rms
                            + _rms(lower_negative_gradient[:, 0::2])
                        )
                        future_state_credit_count += 1

            if self.recursive_cp_her is not None:
                raw_memories, normalized_memories, _ = (
                    self._hierarchical_error_recursive_cp_cache(
                        states,
                        contexts=(
                            persistent_context_cache.contexts
                            if persistent_context_cache is not None
                            else None
                        ),
                    )
                )
                recursive_result = self.recursive_cp_her.local_gradients(
                    [
                        memory.reshape(-1, self.config.hidden_dim)
                        for memory in normalized_memories
                    ],
                    target.reshape(-1, self.config.hidden_dim),
                )
                contributions = [
                    contribution.reshape_as(target)
                    for contribution in self.recursive_cp_her.base_contributions(
                        recursive_result.cache
                    )
                ]
                residual = target - sum(
                    contributions,
                    torch.zeros_like(target),
                )
                for offset, layer in enumerate(error_layers):
                    context_gradient = _rms_norm_vjp(
                        raw_memories[offset],
                        recursive_result.memory_gradients[offset].reshape_as(
                            raw_memories[offset]
                        ),
                    )
                    if persistent_context_cache is not None:
                        write_gate = (
                            persistent_context_cache.write_gates[offset]
                        )
                        memory_scale = (
                            self.config.hierarchical_error_memory_scale
                        )
                        context_gradient = context_gradient * (
                            (1.0 - memory_scale)
                            + memory_scale * write_gate
                        )
                    if layer > 0:
                        deltas[layer - 1].add_(
                            context_gradient,
                            alpha=(
                                -self.config.hierarchical_error_context_credit_scale
                            ),
                        )
                    feedback_rms = feedback_rms + _rms(context_gradient)
                    residual_rms = residual_rms + _rms(
                        contributions[offset] - target
                    )
            else:
                outcomes, residual = self._hierarchical_error_residual_handoff(
                    target,
                    predictions,
                )
            for offset, layer in enumerate(
                range(0) if self.recursive_cp_her is not None else error_layers
            ):
                (
                    predictor_error,
                    value_signal,
                    key_signal,
                    _,
                    context_gradient,
                ) = self._hierarchical_error_factorized_signals(
                    context_state=(
                        persistent_context_cache.contexts[offset]
                        if persistent_context_cache is not None
                        else self._state_context(states[layer])
                    ),
                    prediction=predictions[offset],
                    key=keys[offset],
                    value=values[offset],
                    gate=gates[offset],
                    target=outcomes[offset],
                )
                context_gradient = self._state_context_vjp(
                    context_gradient
                )
                normalized_source_gradient = F.linear(
                    value_signal,
                    self.generative_weights[layer].t(),
                )
                if self.config.hierarchical_error_prediction_mode == "gated":
                    normalized_source_gradient = (
                        normalized_source_gradient
                        + F.linear(
                            key_signal,
                            self.feedback_weights[layer].t(),
                        )
                    )
                source_gradient = _rms_norm_vjp(
                    states[layer + 1],
                    normalized_source_gradient,
                )
                deltas[layer].add_(
                    source_gradient,
                    alpha=-self.config.hierarchical_error_context_credit_scale,
                )
                if layer > 0:
                    if persistent_context_cache is not None:
                        write_gate = (
                            persistent_context_cache.write_gates[offset]
                        )
                        memory_scale = (
                            self.config.hierarchical_error_memory_scale
                        )
                        context_gradient = context_gradient * (
                            (1.0 - memory_scale)
                            + memory_scale * write_gate
                        )
                    deltas[layer - 1].add_(
                        context_gradient,
                        alpha=-self.config.hierarchical_error_context_credit_scale,
                    )
                feedback_rms = feedback_rms + _rms(source_gradient)
                residual_rms = residual_rms + _rms(predictor_error)

            if base_depth > 0:
                deltas[base_depth - 1].add_(target)
            for layer, delta in enumerate(deltas):
                states[layer + 1] = self._apply_inference_delta(
                    states[layer + 1],
                    _cap_token_rms(delta, self.config.state_delta_rms_cap),
                    inference_lr,
                )

        final_deployed = self._hierarchical_error_factorized_pass(states)[0]
        final_energy = self._decoder_output_energy(final_deployed, y)
        layer_steps = float(
            self.config.inference_steps * self.config.num_layers
        )
        error_steps = float(
            self.config.inference_steps
            * (self.config.num_layers - self.config.hierarchical_error_base_depth)
        )
        state_stream_delta_rms = states[0].new_zeros(())
        prediction_stream_delta_rms = states[0].new_zeros(())
        if self.config.state_prediction_separation:
            state_stream_delta_rms = torch.stack(
                [
                    _rms(states[index][:, 0::2] - free_states[index][:, 0::2])
                    for index in range(1, len(states))
                ]
            ).mean()
            prediction_stream_delta_rms = torch.stack(
                [
                    _rms(states[index][:, 1::2] - free_states[index][:, 1::2])
                    for index in range(1, len(states))
                ]
            ).mean()
        diagnostics = {
            "update_diag_message_pc_feedback_rms": float(
                (feedback_rms / error_steps).item()
            ),
            "update_diag_message_pc_prediction_error_rms": float(
                (prediction_rms / layer_steps).item()
            ),
            "update_diag_message_pc_phase_delta_rms": float(
                torch.stack(
                    [
                        _rms(states[index] - free_states[index])
                        for index in range(1, len(states))
                    ]
                ).mean().item()
            ),
            "update_diag_message_pc_her_settle_residual_rms": float(
                (residual_rms / error_steps).item()
            ),
            "update_diag_message_pc_energy_before": float(initial_energy.item()),
            "update_diag_message_pc_energy_after": float(final_energy.item()),
            "update_diag_message_pc_energy_reduction": float(
                (initial_energy - final_energy).item()
            ),
            "update_diag_message_pc_effective_inference_lr": inference_lr,
            "update_diag_message_pc_her_settle_steps": float(
                self.config.inference_steps
            ),
            "update_diag_message_pc_sps_state_stream_delta_rms": float(
                state_stream_delta_rms.item()
            ),
            "update_diag_message_pc_sps_prediction_stream_delta_rms": float(
                prediction_stream_delta_rms.item()
            ),
            "update_diag_message_pc_sps_future_state_credit_rms": float(
                (
                    future_state_credit_rms
                    / max(float(future_state_credit_count), 1.0)
                ).item()
            ),
            "update_diag_message_pc_sps_layer_token_credit_rms": float(
                (
                    layer_token_credit_rms
                    / max(float(layer_token_credit_count), 1.0)
                ).item()
            ),
            "update_diag_message_pc_sps_future_observation_credit_rms": float(
                (
                    future_observation_credit_rms
                    / max(float(future_observation_credit_count), 1.0)
                ).item()
            ),
            "update_diag_message_pc_sps_future_observation_cosine": float(
                (
                    future_observation_cosine
                    / max(float(future_observation_credit_count), 1.0)
                ).item()
            ),
        }
        return states, diagnostics

    def forward_states(self, x: Tensor, **_: object) -> Tensor:
        if self.config.inference_mode == "her_error_hierarchy":
            states, _ = self._forward_states(x, with_cache=False)
            modulated_state, _, _, _ = self._hierarchical_error_pass(states)
            return modulated_state
        states, _ = self._forward_states(
            x,
            with_cache=False,
            layer_count=self._readout_depth(),
        )
        if self.config.bidirectional_deployment_settling:
            states, _ = self._settle_bidirectional_pc(
                states,
                None,
                inference_lr=self._inference_lr(
                    int(self.local_adam_step.item())
                ),
                phase="deployment",
            )
        return states[-1]

    def output_features(self, states: Tensor, _: Tensor) -> Tensor:
        return self._decoder_features(states)

    def _decoder_features(self, states: Tensor) -> Tensor:
        states = self._prediction_states(states)
        if self.config.decoder_state_mode == "rms_norm":
            return _rms_norm(states)
        return states

    def _decoder_logits(self, states: Tensor) -> Tensor:
        return F.linear(
            self._decoder_features(states),
            self.decoder_weight,
            self.decoder_bias,
        )

    def _hierarchical_error_outcome_logits(self, correction: Tensor) -> Tensor:
        scale = self.config.hierarchical_error_outcome_readout_scale
        if scale <= 0:
            prediction = self._prediction_states(correction)
            return prediction.new_zeros(
                (*prediction.shape[:-1], self.config.vocab_size)
            )
        return (
            scale
            * F.linear(
                self._prediction_states(correction),
                _rms_norm(self.feedback_embedding),
            )
            / math.sqrt(self.config.hidden_dim)
        )

    def _hierarchical_error_outcome_gradient(
        self,
        probabilities: Tensor,
        y: Tensor,
    ) -> Tensor:
        codebook = _rms_norm(self.feedback_embedding)
        return (
            self.config.hierarchical_error_outcome_readout_scale
            * (probabilities @ codebook - F.embedding(y, codebook))
            / math.sqrt(self.config.hidden_dim)
        )

    def _winner_margin_terms(
        self,
        logits: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        target_logits = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        wrong_logits = logits.clone()
        wrong_logits.scatter_(-1, y.unsqueeze(-1), -torch.inf)
        top_wrong_logits, top_wrong_ids = wrong_logits.max(dim=-1)
        violations = self.config.winner_margin - target_logits + top_wrong_logits
        active = violations > 0
        return violations.clamp_min(0.0), top_wrong_ids, active

    def _decoder_output_energy(self, state: Tensor, y: Tensor) -> Tensor:
        logits = self._decoder_logits(state)
        energy = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            y.reshape(-1),
        )
        if self.config.winner_margin_scale > 0:
            violations, _, _ = self._winner_margin_terms(logits, y)
            energy = energy + self.config.winner_margin_scale * violations.mean()
        return energy

    def _decoder_negative_state_gradient(
        self,
        state: Tensor,
        y: Tensor,
        *,
        extra_logits: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        prediction_state = self._prediction_states(state)
        logits = self._decoder_logits(state)
        if extra_logits is not None:
            logits = logits + extra_logits
        probabilities = F.softmax(logits, dim=-1)
        feature_feedback = self.decoder_weight[y] - probabilities @ self.decoder_weight
        if self.config.winner_margin_scale > 0:
            _, top_wrong_ids, active = self._winner_margin_terms(logits, y)
            margin_feedback = self.decoder_weight[y] - self.decoder_weight[top_wrong_ids]
            feature_feedback = feature_feedback + self.config.winner_margin_scale * (
                active.unsqueeze(-1) * margin_feedback
            )
        if self.config.decoder_state_mode == "rms_norm":
            feature_feedback = _rms_norm_vjp(
                prediction_state,
                feature_feedback,
            )
        return (
            self._expand_prediction_signal(feature_feedback, state),
            probabilities,
        )

    def _meta_pc_top_feedback(
        self,
        state: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.config.meta_pc_output_feedback_mode == "decoder_transpose":
            return self._decoder_negative_state_gradient(state, y)

        prediction_state = self._prediction_states(state)
        logits = self._decoder_logits(state)
        probabilities = F.softmax(logits, dim=-1)
        feature_feedback = self._independent_output_feedback(probabilities, y)
        if self.config.winner_margin_scale > 0:
            _, top_wrong_ids, active = self._winner_margin_terms(logits, y)
            margin_feedback = self._independent_output_feedback_rows(
                y,
            ) - self._independent_output_feedback_rows(top_wrong_ids)
            feature_feedback = feature_feedback + self.config.winner_margin_scale * (
                active.unsqueeze(-1) * margin_feedback
            )
        if self.config.decoder_state_mode == "rms_norm":
            feature_feedback = _rms_norm_vjp(
                prediction_state,
                feature_feedback,
            )
        return (
            self._expand_prediction_signal(feature_feedback, state),
            probabilities,
        )

    def _independent_output_feedback_rows(self, token_ids: Tensor) -> Tensor:
        if self.config.meta_pc_output_feedback_mode == "kp_factorized":
            if (
                self.feedback_token_factors is None
                or self.feedback_state_factors is None
            ):
                raise RuntimeError("factorized output feedback is not initialized")
            return self.feedback_token_factors[token_ids] @ self.feedback_state_factors
        return self.feedback_embedding[token_ids]

    def _independent_output_feedback(
        self,
        probabilities: Tensor,
        y: Tensor,
    ) -> Tensor:
        if self.config.meta_pc_output_feedback_mode == "kp_factorized":
            if (
                self.feedback_token_factors is None
                or self.feedback_state_factors is None
            ):
                raise RuntimeError("factorized output feedback is not initialized")
            factor_error = (
                self.feedback_token_factors[y]
                - probabilities @ self.feedback_token_factors
            )
            return factor_error @ self.feedback_state_factors
        return self.feedback_embedding[y] - probabilities @ self.feedback_embedding

    def _difference_target_states(
        self,
        free_states: list[Tensor],
        y: Tensor,
        *,
        target_step: float,
    ) -> tuple[list[Tensor], list[Tensor]]:
        top_feedback, _ = self._decoder_negative_state_gradient(
            free_states[-1],
            y,
        )
        top_delta = _cap_token_rms(
            target_step * top_feedback,
            self.config.state_delta_rms_cap,
        )
        targets = list(free_states)
        target_deltas = [torch.zeros_like(state) for state in free_states]
        target_deltas[-1] = top_delta
        targets[-1] = free_states[-1] + top_delta
        if self.config.difference_target_path_mode == "direct":
            for layer in range(self.config.num_layers):
                target_inverse = self._inverse_block(
                    layer,
                    targets[-1],
                ).prediction
                free_inverse = self._inverse_block(
                    layer,
                    free_states[-1],
                ).prediction
                delta = _cap_token_rms(
                    target_inverse - free_inverse,
                    self.config.state_delta_rms_cap,
                )
                target_deltas[layer] = delta
                targets[layer] = free_states[layer] + delta
            return targets, target_deltas
        for layer in range(self.config.num_layers - 1, -1, -1):
            target_inverse = self._inverse_block(
                layer,
                targets[layer + 1],
            ).prediction
            free_inverse = self._inverse_block(
                layer,
                free_states[layer + 1],
            ).prediction
            delta = _cap_token_rms(
                target_inverse - free_inverse,
                self.config.state_delta_rms_cap,
            )
            target_deltas[layer] = delta
            targets[layer] = free_states[layer] + delta
        return targets, target_deltas

    def _forward_from_state(self, start_layer: int, state: Tensor) -> Tensor:
        for layer in range(start_layer, self.config.num_layers):
            state = self._block(layer, state).prediction
        return state

    def _difference_inverse_gradients(
        self,
        layer: int,
        free_lower: Tensor,
        free_upper: Tensor,
        *,
        token_count: float,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Tensor]:
        noise = self.config.difference_target_inverse_noise * torch.randn_like(
            free_lower
        )
        noisy_lower = free_lower + noise
        if self.config.difference_target_path_mode == "direct":
            noisy_upper = self._forward_from_state(layer, noisy_lower)
        else:
            noisy_upper = self._block(layer, noisy_lower).prediction
        noisy_inverse = self._inverse_block(layer, noisy_upper)
        free_inverse = self._inverse_block(layer, free_upper)
        predicted_delta = noisy_inverse.prediction - free_inverse.prediction
        reconstruction_error = predicted_delta - noise
        gradients = self._inverse_parameter_gradients(
            layer,
            noisy_inverse,
            reconstruction_error,
            token_count=token_count,
        )
        free_gradients = self._inverse_parameter_gradients(
            layer,
            free_inverse,
            -reconstruction_error,
            token_count=token_count,
        )
        for name, gradient in free_gradients.items():
            gradients[name].add_(gradient)
        damping_response = free_lower.new_zeros(())
        if self.config.difference_target_inverse_damping > 0:
            upper_noise = (
                self.config.difference_target_inverse_noise
                * torch.randn_like(free_upper)
            )
            damped_inverse = self._inverse_block(
                layer,
                free_upper + upper_noise,
            )
            damping_delta = damped_inverse.prediction - free_inverse.prediction
            damping_signal = (
                self.config.difference_target_inverse_damping
                * damping_delta
            )
            damped_gradients = self._inverse_parameter_gradients(
                layer,
                damped_inverse,
                damping_signal,
                token_count=token_count,
            )
            free_damping_gradients = self._inverse_parameter_gradients(
                layer,
                free_inverse,
                -damping_signal,
                token_count=token_count,
            )
            for name in gradients:
                gradients[name].add_(damped_gradients[name])
                gradients[name].add_(free_damping_gradients[name])
            damping_response = _rms(damping_delta)
        alignment = F.cosine_similarity(
            predicted_delta.float().reshape(-1, self.config.hidden_dim),
            noise.float().reshape(-1, self.config.hidden_dim),
            dim=-1,
        ).mean()
        return (
            gradients,
            _rms(reconstruction_error),
            _rms(predicted_delta),
            alignment,
            damping_response,
        )

    def _winner_margin_decoder_gradients(
        self,
        decoder_features: Tensor,
        y: Tensor,
        logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        weight_gradient = torch.zeros_like(self.decoder_weight)
        bias_gradient = torch.zeros_like(self.decoder_bias)
        if self.config.winner_margin_scale <= 0:
            return weight_gradient, bias_gradient, logits.new_zeros(())
        _, top_wrong_ids, active = self._winner_margin_terms(logits, y)
        flat_features = decoder_features.reshape(-1, decoder_features.shape[-1])
        flat_targets = y.reshape(-1)
        flat_wrong = top_wrong_ids.reshape(-1)
        flat_active = active.reshape(-1)
        active_features = flat_features * flat_active.unsqueeze(-1)
        weight_gradient.index_add_(0, flat_wrong, active_features)
        weight_gradient.index_add_(0, flat_targets, -active_features)
        active_values = flat_active.to(bias_gradient.dtype)
        bias_gradient.index_add_(0, flat_wrong, active_values)
        bias_gradient.index_add_(0, flat_targets, -active_values)
        scale = self.config.winner_margin_scale / float(max(flat_targets.numel(), 1))
        return (
            weight_gradient * scale,
            bias_gradient * scale,
            flat_active.float().mean(),
        )

    def output_logits(self, states: Tensor, _: Tensor, **__: object) -> Tensor:
        return self._decoder_logits(states)

    def forward(self, x: Tensor, context_ids: Tensor | None = None) -> Tensor:
        del context_ids
        if self.config.inference_mode == "her_error_hierarchy":
            states, _ = self._forward_states(x, with_cache=False)
            modulated_state, _, _, correction = (
                self._hierarchical_error_pass(states)
            )
            return self._decoder_logits(
                modulated_state
            ) + self._hierarchical_error_outcome_logits(correction)
        return self._decoder_logits(self.forward_states(x))

    def forward_last_logits(self, x: Tensor, context_ids: Tensor | None = None) -> Tensor:
        return self.forward(x, context_ids=context_ids)[:, -1:, :]

    def _token_feedback(self, state: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        logits = self._decoder_logits(state)
        probabilities = F.softmax(logits, dim=-1)
        feedback = self.feedback_embedding[y] - probabilities @ self.feedback_embedding
        return feedback, probabilities

    def _feedback_path(
        self,
        states: list[Tensor],
        y: Tensor,
    ) -> tuple[list[Tensor], list[Tensor]]:
        sources: list[Tensor] = [states[0].new_empty(0) for _ in range(self.config.num_layers)]
        outputs: list[Tensor] = [states[0].new_empty(0) for _ in range(self.config.num_layers)]
        if self.config.token_feedback_mode == "direct_layer":
            for layer in range(self.config.num_layers):
                source, _ = self._token_feedback(states[layer + 1], y)
                sources[layer] = source
                outputs[layer] = F.linear(source, self.feedback_weights[layer])
            return sources, outputs

        signal, _ = self._token_feedback(states[-1], y)
        for layer in range(self.config.num_layers - 1, -1, -1):
            sources[layer] = signal
            signal = F.linear(signal, self.feedback_weights[layer])
            outputs[layer] = signal
        return sources, outputs

    def _generative_prediction(self, layer: int, upper_state: Tensor) -> Tensor:
        return torch.tanh(F.linear(_rms_norm(upper_state), self.generative_weights[layer]))

    def _decoder_state_weights(self) -> list[float]:
        if self.config.decoder_supervision_mode == "top_only":
            return [0.0] * (self.config.num_layers - 1) + [1.0]
        if self.config.readout_objective_share > 0 and self.config.num_layers > 1:
            readout_index = self._readout_depth() - 1
            other_share = (1.0 - self.config.readout_objective_share) / (
                self.config.num_layers - 1
            )
            weights = [other_share] * self.config.num_layers
            weights[readout_index] = self.config.readout_objective_share
            return weights
        normalizer = self.config.num_layers * (self.config.num_layers + 1) / 2.0
        return [
            (layer + 1) / normalizer
            for layer in range(self.config.num_layers)
        ]

    def _settle_heuristic(
        self,
        free_states: list[Tensor],
        y: Tensor,
        *,
        inference_lr: float,
    ) -> tuple[list[Tensor], dict[str, float]]:
        states = [state.clone() for state in free_states]
        feedback_rms = states[0].new_zeros(())
        prediction_rms = states[0].new_zeros(())
        for _ in range(self.config.inference_steps):
            block_caches = [self._block(layer, states[layer]) for layer in range(self.config.num_layers)]
            _, learned_feedbacks = self._feedback_path(states, y)
            deltas: list[Tensor] = []
            for layer, cache in enumerate(block_caches):
                current = states[layer + 1]
                learned_feedback = learned_feedbacks[layer]
                layer_gain = (layer + 1) / self.config.num_layers
                delta = self.config.token_feedback_scale * layer_gain * learned_feedback
                delta = delta + self.config.prediction_precision * (cache.prediction - current)

                own_lower_prediction = self._generative_prediction(layer, current)
                own_lower_error = states[layer] - own_lower_prediction
                own_upper_drive = F.linear(
                    own_lower_error * (1.0 - own_lower_prediction.square()),
                    self.generative_weights[layer].t(),
                )
                delta = delta + self.config.generative_precision * own_upper_drive
                if layer + 1 < self.config.num_layers:
                    higher_prediction = self._generative_prediction(layer + 1, states[layer + 2])
                    delta = delta + self.config.generative_precision * (higher_prediction - current)

                delta = _cap_token_rms(delta, self.config.state_delta_rms_cap)
                deltas.append(delta)
                feedback_rms = feedback_rms + _rms(learned_feedback)
                prediction_rms = prediction_rms + _rms(cache.prediction - current)
            for layer, delta in enumerate(deltas):
                states[layer + 1] = self._apply_inference_delta(
                    states[layer + 1],
                    delta,
                    inference_lr,
                )

        normalizer = float(self.config.inference_steps * self.config.num_layers)
        diagnostics = {
            "update_diag_message_pc_feedback_rms": float((feedback_rms / normalizer).item()),
            "update_diag_message_pc_prediction_error_rms": float((prediction_rms / normalizer).item()),
            "update_diag_message_pc_phase_delta_rms": float(
                torch.stack([_rms(states[index] - free_states[index]) for index in range(1, len(states))]).mean().item()
            ),
        }
        return states, diagnostics

    def _inference_precision_multipliers(
        self,
        inference_round: int,
        inference_lr: float,
    ) -> tuple[list[float] | None, list[float] | None, float]:
        scale = self.config.spiking_precision_scale
        if scale <= 0:
            return None, None, 1.0
        exact_spike = max(1.0, 1.0 / max(inference_lr, 1e-8))
        spike = 1.0 + scale * (exact_spike - 1.0)
        prediction = [1.0] * self.config.num_layers
        output = [1.0] * self.config.num_layers
        if inference_round == 0:
            for layer, weight in enumerate(self._decoder_state_weights()):
                if weight > 0:
                    output[layer] = spike
        elif inference_round <= self.config.num_layers:
            prediction[self.config.num_layers - inference_round] = spike
        return prediction, output, spike

    def _bidirectional_pc_energy(
        self,
        states: list[Tensor],
        y: Tensor | None,
    ) -> Tensor:
        reconstruction_energy = states[0].new_zeros(())
        for layer in range(self.config.num_layers):
            prediction = self._inverse_block(
                layer, states[layer + 1]
            ).prediction
            error = prediction - states[layer]
            reconstruction_energy = reconstruction_energy + (
                0.5 * error.square().sum(dim=-1).mean()
            )
        energy = self.config.prediction_precision * reconstruction_energy
        if y is not None:
            energy = energy + (
                self.config.token_feedback_scale
                * self._decoder_output_energy(states[-1], y)
            )
        return energy

    def _bidirectional_pc_state_deltas(
        self,
        states: list[Tensor],
        y: Tensor | None,
    ) -> tuple[list[Tensor], Tensor]:
        caches = [
            self._inverse_block(layer, states[layer + 1])
            for layer in range(self.config.num_layers)
        ]
        errors = [
            caches[layer].prediction - states[layer]
            for layer in range(self.config.num_layers)
        ]
        deltas: list[Tensor] = []
        for state_index in range(1, self.config.num_layers + 1):
            delta = -self.config.prediction_precision * (
                self._inverse_input_signal(
                    state_index - 1,
                    states[state_index],
                    caches[state_index - 1],
                    errors[state_index - 1],
                )
            )
            if state_index < self.config.num_layers:
                delta = delta + (
                    self.config.prediction_precision * errors[state_index]
                )
            if y is not None and state_index == self.config.num_layers:
                token_delta, _ = self._decoder_negative_state_gradient(
                    states[state_index], y
                )
                delta = delta + self.config.token_feedback_scale * token_delta
            deltas.append(
                _cap_token_rms(delta, self.config.state_delta_rms_cap)
            )
        return deltas, torch.stack([_rms(error) for error in errors]).mean()

    def _settle_bidirectional_pc(
        self,
        initial_states: list[Tensor],
        y: Tensor | None,
        *,
        inference_lr: float,
        phase: str,
    ) -> tuple[list[Tensor], dict[str, float]]:
        states = [state.clone() for state in initial_states]
        initial_energy = self._bidirectional_pc_energy(states, y)
        current_energy = initial_energy
        prediction_error_rms = states[0].new_zeros(())
        accepted_lr_total = 0.0
        accepted_steps = 0
        backtracking_rejections = 0
        for _ in range(self.config.inference_steps):
            deltas, round_prediction_error_rms = (
                self._bidirectional_pc_state_deltas(states, y)
            )
            prediction_error_rms.add_(round_prediction_error_rms)
            candidate_lr = inference_lr
            trial_count = max(self.config.energy_backtracking_steps, 0) + 1
            accepted = False
            for _trial in range(trial_count):
                candidate_states = [states[0]] + [
                    self._apply_inference_delta(
                        states[layer + 1], delta, candidate_lr
                    )
                    for layer, delta in enumerate(deltas)
                ]
                candidate_energy = self._bidirectional_pc_energy(
                    candidate_states, y
                )
                if (
                    self.config.energy_backtracking_steps <= 0
                    or bool(candidate_energy <= current_energy)
                ):
                    states = candidate_states
                    current_energy = candidate_energy
                    accepted_lr_total += candidate_lr
                    accepted_steps += 1
                    accepted = True
                    break
                candidate_lr *= 0.5
                backtracking_rejections += 1
            if not accepted:
                break
        final_energy = self._bidirectional_pc_energy(states, y)
        prefix = f"update_diag_message_pc_bidirectional_{phase}"
        diagnostics = {
            f"{prefix}_energy_before": float(initial_energy.item()),
            f"{prefix}_energy_after": float(final_energy.item()),
            f"{prefix}_energy_reduction": float(
                (initial_energy - final_energy).item()
            ),
            f"{prefix}_prediction_error_rms": float(
                (prediction_error_rms / self.config.inference_steps).item()
            ),
            f"{prefix}_accepted_lr": (
                accepted_lr_total / max(accepted_steps, 1)
            ),
            f"{prefix}_backtracking_rejections": float(
                backtracking_rejections
            ),
        }
        for state_index in range(1, len(states)):
            diagnostics[
                f"{prefix}_delta_rms_layer_{state_index}"
            ] = float(
                _rms(states[state_index] - initial_states[state_index]).item()
            )
        return states, diagnostics

    def _residual_transport_block(
        self,
        layer: int,
        state: Tensor,
    ) -> _InverseAttentionCache:
        """Independent causal feedback path; no forward-weight transport."""
        source = _rms_norm(state)
        query = self._split_heads(
            F.linear(source, self.inverse_query_weights[layer])
        )
        key = self._split_heads(
            F.linear(source, self.inverse_key_weights[layer])
        )
        value = self._split_heads(source)
        score_scale = 1.0 / (
            math.sqrt(self.head_dim) * self.config.attention_temperature
        )
        scores = torch.matmul(query, key.transpose(-2, -1)) * score_scale
        attention_mask = self._causal_mask(
            state.shape[1],
            state.device,
            layer=layer,
        )
        scores.masked_fill_(attention_mask, -torch.inf)
        weights = F.softmax(scores, dim=-1)
        message = self._merge_heads(torch.matmul(weights, value))
        prediction = state + F.linear(
            message,
            self.feedback_weights[layer],
        )
        return _InverseAttentionCache(
            source=source,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            weights=weights,
            message=message,
            prediction=prediction,
        )

    def _residual_transport_parameter_gradients(
        self,
        layer: int,
        cache: _InverseAttentionCache,
        output_signal: Tensor,
        *,
        token_count: float,
    ) -> dict[str, Tensor]:
        message_signal = F.linear(
            output_signal,
            self.feedback_weights[layer].t(),
        )
        head_message_signal = self._split_heads(message_signal)
        weight_signal = torch.matmul(
            head_message_signal,
            cache.value.transpose(-2, -1),
        )
        centered_weight_signal = weight_signal - (
            cache.weights * weight_signal
        ).sum(dim=-1, keepdim=True)
        score_signal = cache.weights * centered_weight_signal
        score_signal.masked_fill_(cache.attention_mask, 0.0)
        score_scale = 1.0 / (
            math.sqrt(self.head_dim) * self.config.attention_temperature
        )
        query_signal = torch.matmul(score_signal, cache.key) * score_scale
        key_signal = (
            torch.matmul(score_signal.transpose(-2, -1), cache.query)
            * score_scale
        )
        flat_source = cache.source.reshape(-1, self.config.hidden_dim)
        return {
            f"inverse_query_weights.{layer}": (
                self._merge_heads(query_signal)
                .reshape(-1, self.config.hidden_dim)
                .t()
                @ flat_source
            )
            / token_count,
            f"inverse_key_weights.{layer}": (
                self._merge_heads(key_signal)
                .reshape(-1, self.config.hidden_dim)
                .t()
                @ flat_source
            )
            / token_count,
            f"feedback_weights.{layer}": (
                output_signal.reshape(-1, self.config.hidden_dim).t()
                @ cache.message.reshape(-1, self.config.hidden_dim)
            )
            / token_count,
        }

    def _residual_transport_input_signal(
        self,
        layer: int,
        state: Tensor,
        cache: _InverseAttentionCache,
        output_signal: Tensor,
    ) -> Tensor:
        message_signal = F.linear(
            output_signal,
            self.feedback_weights[layer].t(),
        )
        head_message_signal = self._split_heads(message_signal)
        weight_signal = torch.matmul(
            head_message_signal,
            cache.value.transpose(-2, -1),
        )
        centered_weight_signal = weight_signal - (
            cache.weights * weight_signal
        ).sum(dim=-1, keepdim=True)
        score_signal = cache.weights * centered_weight_signal
        score_signal.masked_fill_(cache.attention_mask, 0.0)
        score_scale = 1.0 / (
            math.sqrt(self.head_dim) * self.config.attention_temperature
        )
        query_signal = torch.matmul(score_signal, cache.key) * score_scale
        key_signal = (
            torch.matmul(score_signal.transpose(-2, -1), cache.query)
            * score_scale
        )
        value_signal = torch.matmul(
            cache.weights.transpose(-2, -1),
            head_message_signal,
        )
        source_signal = F.linear(
            self._merge_heads(query_signal),
            self.inverse_query_weights[layer].t(),
        )
        source_signal.add_(
            F.linear(
                self._merge_heads(key_signal),
                self.inverse_key_weights[layer].t(),
            )
        )
        source_signal.add_(self._merge_heads(value_signal))
        return output_signal + _rms_norm_vjp(state, source_signal)

    def _residual_phase_transport_block(
        self,
        layer: int,
        state: Tensor,
        error_signal: Tensor,
    ) -> _InverseAttentionCache:
        """Route later residual activity to earlier state nodes."""
        source = _rms_norm(state)
        query = self._split_heads(
            F.linear(source, self.inverse_query_weights[layer])
        )
        key = self._split_heads(
            F.linear(source, self.inverse_key_weights[layer])
        )
        value = self._split_heads(error_signal)
        score_scale = 1.0 / (
            math.sqrt(self.head_dim) * self.config.attention_temperature
        )
        scores = torch.matmul(query, key.transpose(-2, -1)) * score_scale
        attention_mask = self._causal_mask(
            state.shape[1],
            state.device,
            layer=layer,
        )
        scores.masked_fill_(attention_mask, -torch.inf)
        weights = F.softmax(scores, dim=-1)
        message = self._merge_heads(
            torch.matmul(weights.transpose(-2, -1), value)
        )
        proposal = F.linear(message, self.feedback_weights[layer])
        return _InverseAttentionCache(
            source=source,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            weights=weights,
            message=message,
            prediction=proposal,
        )

    def _residual_phase_transport_parameter_gradients(
        self,
        layer: int,
        cache: _InverseAttentionCache,
        output_signal: Tensor,
        *,
        token_count: float,
    ) -> dict[str, Tensor]:
        message_signal = F.linear(
            output_signal,
            self.feedback_weights[layer].t(),
        )
        head_message_signal = self._split_heads(message_signal)
        weight_signal = torch.matmul(
            cache.value,
            head_message_signal.transpose(-2, -1),
        )
        centered_weight_signal = weight_signal - (
            cache.weights * weight_signal
        ).sum(dim=-1, keepdim=True)
        score_signal = cache.weights * centered_weight_signal
        score_signal.masked_fill_(cache.attention_mask, 0.0)
        score_scale = 1.0 / (
            math.sqrt(self.head_dim) * self.config.attention_temperature
        )
        query_signal = torch.matmul(score_signal, cache.key) * score_scale
        key_signal = (
            torch.matmul(score_signal.transpose(-2, -1), cache.query)
            * score_scale
        )
        flat_source = cache.source.reshape(-1, self.config.hidden_dim)
        return {
            f"inverse_query_weights.{layer}": (
                self._merge_heads(query_signal)
                .reshape(-1, self.config.hidden_dim)
                .t()
                @ flat_source
            )
            / token_count,
            f"inverse_key_weights.{layer}": (
                self._merge_heads(key_signal)
                .reshape(-1, self.config.hidden_dim)
                .t()
                @ flat_source
            )
            / token_count,
            f"feedback_weights.{layer}": (
                output_signal.reshape(-1, self.config.hidden_dim).t()
                @ cache.message.reshape(-1, self.config.hidden_dim)
            )
            / token_count,
        }

    def _residual_error_prediction(
        self,
        layer: int,
        state: Tensor,
    ) -> tuple[_InverseAttentionCache | None, Tensor]:
        """Predict an error vector, never an adjacent representation state."""
        if self.config.hierarchical_error_prediction_mode in {
            "direct",
            "direct_transport",
            "direct_phase_transport",
        }:
            return None, F.linear(
                _rms_norm(state),
                self.generative_weights[layer],
            )
        cache = self._inverse_block(layer, state)
        return cache, cache.prediction - state

    def _residual_error_token_delta(
        self,
        y: Tensor,
        probabilities: Tensor,
        decoder_token_delta: Tensor,
    ) -> Tensor:
        if self.config.hierarchical_error_target_mode == "all_tokens":
            return decoder_token_delta
        if self.config.hierarchical_error_target_mode == "fixed_outcome":
            codebook = _rms_norm(self.feedback_embedding)
            observed = F.embedding(y, codebook)
            expected = probabilities @ codebook
            return (
                self.config.hierarchical_error_outcome_code_scale
                * (observed - expected)
                / math.sqrt(self.config.hidden_dim)
            )
        mistakes = probabilities.argmax(dim=-1) != y
        return mistakes.unsqueeze(-1) * decoder_token_delta

    def _residual_error_proxy_memory(self, token_delta: Tensor) -> Tensor:
        """Causal memory of previous token-error targets."""
        scale = self.config.residual_error_proxy_memory_scale
        if scale == 0 or token_delta.shape[1] == 0:
            return torch.zeros_like(token_delta)
        retained = self._causal_affine_scan(
            token_delta.new_full(
                token_delta.shape[:2] + (1,),
                self.config.residual_error_proxy_memory_decay,
            ),
            token_delta,
        )
        previous = torch.cat(
            (torch.zeros_like(retained[:, :1]), retained[:, :-1]),
            dim=1,
        )
        return scale * previous

    def _causal_previous_ema(
        self,
        value: Tensor,
        *,
        decay: float,
        scale: float,
    ) -> Tensor:
        if scale == 0 or value.shape[1] == 0:
            return torch.zeros_like(value)
        retained = self._causal_affine_scan(
            value.new_full(value.shape[:2] + (1,), decay),
            value,
        )
        previous = torch.cat(
            (torch.zeros_like(retained[:, :1]), retained[:, :-1]),
            dim=1,
        )
        return scale * previous

    def _residual_error_upper_proxy_state_delta(
        self,
        token_delta: Tensor,
    ) -> Tensor:
        return self._causal_previous_ema(
            token_delta,
            decay=self.config.residual_error_upper_proxy_state_decay,
            scale=self.config.residual_error_upper_proxy_state_scale,
        )

    def _residual_error_period_mask(
        self,
        x: Tensor,
        *,
        period: int,
        offset: int,
    ) -> Tensor:
        if period == 0:
            return x.new_ones(x.shape + (1,), dtype=self.embedding.weight.dtype)
        positions = torch.arange(x.shape[1], device=x.device).view(1, -1, 1)
        return (positions.remainder(period) == offset).to(
            dtype=self.embedding.weight.dtype
        )

    def _causal_delayed_event_ema(
        self,
        value: Tensor,
        write_mask: Tensor,
        *,
        decay: float,
        scale: float,
        delay: int,
    ) -> Tensor:
        if scale == 0 or value.shape[1] == 0:
            return torch.zeros_like(value)
        state = torch.zeros_like(value[:, 0])
        retained: list[Tensor] = []
        for token_index in range(value.shape[1]):
            write = write_mask[:, token_index]
            candidate = value[:, token_index] + decay * state
            state = torch.where(write.bool(), candidate, state)
            retained.append(state)
        retained_value = torch.stack(retained, dim=1)
        if delay >= value.shape[1]:
            shifted = torch.zeros_like(retained_value)
        else:
            shifted = torch.cat(
                (
                    torch.zeros_like(retained_value[:, :delay]),
                    retained_value[:, :-delay],
                ),
                dim=1,
        )
        return scale * shifted

    def _delayed_tokens(self, value: Tensor, *, delay: int) -> Tensor:
        if delay == 0 or value.shape[1] == 0:
            return value
        if delay >= value.shape[1]:
            return torch.zeros_like(value)
        return torch.cat(
            (torch.zeros_like(value[:, :delay]), value[:, :-delay]),
            dim=1,
        )

    def _residual_error_upper_content_state_delta(
        self,
        x: Tensor,
    ) -> Tensor:
        """Causal upper-state memory of previous sensory content."""
        scale = self.config.residual_error_upper_content_state_scale
        if scale == 0 or x.shape[1] == 0:
            shape = x.shape + (self.config.hidden_dim,)
            return self.embedding.weight.new_zeros(shape)
        content = _rms_norm(self.embedding(x))
        write_mask = self._residual_error_period_mask(
            x,
            period=self.config.residual_error_upper_content_write_period,
            offset=self.config.residual_error_upper_content_write_offset,
        )
        read_mask = self._residual_error_period_mask(
            x,
            period=self.config.residual_error_upper_content_read_period,
            offset=self.config.residual_error_upper_content_read_offset,
        )
        raw_memory = self._causal_delayed_event_ema(
            content,
            decay=self.config.residual_error_upper_content_state_decay,
            scale=1.0,
            delay=self.config.residual_error_upper_content_read_delay,
            write_mask=write_mask,
        )
        current_content = self._delayed_tokens(
            content,
            delay=self.config.residual_error_upper_content_bind_current_delay,
        )
        bound_content = raw_memory
        if self.config.residual_error_upper_content_bind_current_scale > 0:
            bound_content = bound_content + (
                self.config.residual_error_upper_content_bind_current_scale
                * current_content
            )
        if self.config.residual_error_upper_content_bind_product_scale > 0:
            bound_content = bound_content + (
                self.config.residual_error_upper_content_bind_product_scale
                * _rms_norm(raw_memory * current_content)
            )
        return scale * bound_content * read_mask

    def _residual_error_target_state(
        self,
        reference_state: Tensor,
        token_delta: Tensor,
    ) -> tuple[Tensor, Tensor]:
        proxy_memory = self._residual_error_proxy_memory(token_delta)
        target_delta = self.config.token_feedback_scale * (
            token_delta + proxy_memory
        )
        return reference_state + target_delta, proxy_memory

    def _residual_error_exact_input_signal(
        self,
        layer: int,
        state: Tensor,
        cache: _InverseAttentionCache | None,
        output_signal: Tensor,
    ) -> Tensor:
        if cache is None:
            if self.config.residual_error_transport_mode == "learned_only":
                return torch.zeros_like(output_signal)
            source_signal = F.linear(
                output_signal,
                self.generative_weights[layer].t(),
            )
            return _rms_norm_vjp(state, source_signal)
        return (
            self._inverse_input_signal(
                layer,
                state,
                cache,
                output_signal,
            )
            - output_signal
        )

    def _residual_error_learned_input_signal(
        self,
        layer: int,
        state: Tensor,
        cache: _InverseAttentionCache | None,
        output_signal: Tensor,
    ) -> Tensor:
        if cache is not None or self.config.learned_feedback_proposal_scale <= 0:
            return torch.zeros_like(output_signal)
        if self.config.hierarchical_error_prediction_mode == "direct_transport":
            transport_cache = self._residual_transport_block(layer, state)
            return (
                self._residual_transport_input_signal(
                    layer,
                    state,
                    transport_cache,
                    output_signal,
                )
                - output_signal
            )
        if (
            self.config.hierarchical_error_prediction_mode
            == "direct_phase_transport"
        ):
            transport_cache = self._residual_phase_transport_block(
                layer,
                state,
                output_signal,
            )
            return transport_cache.prediction
        return torch.zeros_like(output_signal)

    def _residual_error_input_signal(
        self,
        layer: int,
        state: Tensor,
        cache: _InverseAttentionCache | None,
        output_signal: Tensor,
    ) -> Tensor:
        signal = self._residual_error_exact_input_signal(
            layer,
            state,
            cache,
            output_signal,
        )
        if self.config.learned_feedback_proposal_scale > 0:
            signal = signal + (
                self.config.learned_feedback_proposal_scale
                * self._residual_error_learned_input_signal(
                    layer,
                    state,
                    cache,
                    output_signal,
                )
            )
        return signal

    def _residual_error_pc_terms(
        self,
        states: list[Tensor],
        target_state: Tensor,
        *,
        context_offsets: list[Tensor] | None = None,
        target_state_index: int | None = None,
    ) -> tuple[list[_InverseAttentionCache | None], list[Tensor], list[Tensor]]:
        if target_state_index is None:
            target_state_index = self.config.num_layers
        if not 1 <= target_state_index <= self.config.num_layers:
            raise ValueError("target_state_index must refer to a latent layer")
        caches: list[_InverseAttentionCache | None] = []
        predictions: list[Tensor] = []
        residuals = [target_state - states[target_state_index]]
        for layer in range(self.config.num_layers):
            predictor_state = states[layer + 1]
            if context_offsets is not None:
                predictor_state = predictor_state + context_offsets[layer]
            cache, prediction = self._residual_error_prediction(
                layer,
                predictor_state,
            )
            caches.append(cache)
            predictions.append(prediction)
            residuals.append(residuals[-1] - prediction)
        return caches, predictions, residuals

    def _residual_error_pc_energy(
        self,
        states: list[Tensor],
        free_states: list[Tensor],
        target_state: Tensor,
        *,
        context_offsets: list[Tensor] | None = None,
        target_state_index: int | None = None,
        upper_proxy_target_state: Tensor | None = None,
    ) -> Tensor:
        _, _, residuals = self._residual_error_pc_terms(
            states,
            target_state,
            context_offsets=context_offsets,
            target_state_index=target_state_index,
        )
        residual_energy = torch.stack(
            [
                0.5 * residual.square().sum(dim=-1).mean()
                for residual in residuals
            ]
        ).sum()
        state_anchor_energy = torch.stack(
            [
                0.5
                * (state - free_state).square().sum(dim=-1).mean()
                for state, free_state in zip(states[1:], free_states[1:])
            ]
        ).sum()
        upper_proxy_energy = states[0].new_zeros(())
        if (
            upper_proxy_target_state is not None
            and target_state_index != self.config.num_layers
        ):
            upper_proxy_energy = (
                0.5
                * (states[-1] - upper_proxy_target_state)
                .square()
                .sum(dim=-1)
                .mean()
            )
        return (
            self.config.prediction_precision * residual_energy
            + self.config.generative_precision
            * (state_anchor_energy + upper_proxy_energy)
        )

    def _residual_error_suffix_signals(
        self,
        residuals: list[Tensor],
    ) -> list[Tensor]:
        suffix_residual = torch.zeros_like(residuals[0])
        signals = [torch.empty(0, device=residuals[0].device)] * (
            self.config.num_layers
        )
        suffix_decay = self.config.residual_error_suffix_decay
        for layer in range(self.config.num_layers - 1, -1, -1):
            suffix_residual = residuals[layer + 1] + (
                suffix_decay * suffix_residual
            )
            signals[layer] = suffix_residual.clone()
        return signals

    def _residual_error_pc_state_deltas(
        self,
        states: list[Tensor],
        free_states: list[Tensor],
        target_state: Tensor,
        *,
        context_offsets: list[Tensor] | None = None,
        target_state_index: int | None = None,
        upper_proxy_target_state: Tensor | None = None,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        if target_state_index is None:
            target_state_index = self.config.num_layers
        caches, predictions, residuals = self._residual_error_pc_terms(
            states,
            target_state,
            context_offsets=context_offsets,
            target_state_index=target_state_index,
        )
        prediction_signals = self._residual_error_suffix_signals(residuals)

        direct_target_signal = torch.stack(residuals).sum(dim=0)
        deltas: list[Tensor] = []
        for layer in range(self.config.num_layers):
            predictor_state = states[layer + 1]
            if context_offsets is not None:
                predictor_state = predictor_state + context_offsets[layer]
            delta = self.config.prediction_precision * (
                self._residual_error_input_signal(
                    layer,
                    predictor_state,
                    caches[layer],
                    prediction_signals[layer],
                )
            )
            if layer + 1 == target_state_index:
                delta = delta + (
                    self.config.prediction_precision * direct_target_signal
                )
            if (
                upper_proxy_target_state is not None
                and layer + 1 == self.config.num_layers
                and target_state_index != self.config.num_layers
            ):
                delta = delta - self.config.generative_precision * (
                    states[layer + 1] - upper_proxy_target_state
                )
            delta = delta - self.config.generative_precision * (
                states[layer + 1] - free_states[layer + 1]
            )
            deltas.append(
                _cap_token_rms(delta, self.config.state_delta_rms_cap)
            )
        return deltas, predictions, residuals

    def _residual_error_pc_predictor_parameter_gradients(
        self,
        states: list[Tensor],
        target_state: Tensor,
        *,
        token_count: float,
        context_offsets: list[Tensor] | None = None,
        target_state_index: int | None = None,
    ) -> dict[str, Tensor]:
        caches, _, residuals = self._residual_error_pc_terms(
            states,
            target_state,
            context_offsets=context_offsets,
            target_state_index=target_state_index,
        )
        output_signals = [
            -self.config.prediction_precision * signal
            for signal in self._residual_error_suffix_signals(residuals)
        ]
        gradients: dict[str, Tensor] = {}
        for layer, (cache, output_signal) in enumerate(
            zip(caches, output_signals)
        ):
            if cache is None:
                predictor_state = states[layer + 1]
                if context_offsets is not None:
                    predictor_state = predictor_state + context_offsets[layer]
                gradients[f"generative_weights.{layer}"] = (
                    output_signal.reshape(-1, self.config.hidden_dim).t()
                    @ _rms_norm(predictor_state).reshape(
                        -1,
                        self.config.hidden_dim,
                    )
                ) / token_count
            else:
                gradients.update(
                    self._bidirectional_inverse_parameter_gradients(
                        layer,
                        cache,
                        output_signal,
                        token_count=token_count,
                    )
                )
        return gradients

    def _residual_phase_transport_training_gradients(
        self,
        free_states: list[Tensor],
        settled_states: list[Tensor],
        target_state: Tensor,
        *,
        accepted_lr: float,
        token_count: float,
        context_offsets: list[Tensor] | None = None,
        target_state_index: int | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor | list[Tensor]]]:
        _, _, free_residuals = self._residual_error_pc_terms(
            free_states,
            target_state,
            context_offsets=context_offsets,
            target_state_index=target_state_index,
        )
        transport_signals = self._residual_error_suffix_signals(
            free_residuals
        )
        phase_scale = max(
            self.config.inference_steps * accepted_lr,
            1e-6,
        )
        gradients: dict[str, Tensor] = {}
        fit_rms = free_states[0].new_zeros(())
        target_rms = free_states[0].new_zeros(())
        cosine = free_states[0].new_zeros(())
        fit_by_layer = [
            free_states[0].new_zeros(())
            for _ in range(self.config.num_layers)
        ]
        target_by_layer = [
            free_states[0].new_zeros(())
            for _ in range(self.config.num_layers)
        ]
        cosine_by_layer = [
            free_states[0].new_zeros(())
            for _ in range(self.config.num_layers)
        ]
        for layer, transport_signal in enumerate(transport_signals):
            transport_cache = self._residual_phase_transport_block(
                layer,
                free_states[layer + 1],
                transport_signal,
            )
            transport_target = (
                settled_states[layer + 1] - free_states[layer + 1]
            ) / phase_scale
            transport_error = transport_cache.prediction - transport_target
            layer_fit_rms = _rms(transport_error)
            layer_target_rms = _rms(transport_target)
            layer_cosine = F.cosine_similarity(
                transport_cache.prediction.reshape(
                    -1,
                    self.config.hidden_dim,
                ),
                transport_target.reshape(
                    -1,
                    self.config.hidden_dim,
                ),
                dim=-1,
            ).mean()
            fit_rms.add_(layer_fit_rms)
            target_rms.add_(layer_target_rms)
            cosine.add_(layer_cosine)
            fit_by_layer[layer] = layer_fit_rms
            target_by_layer[layer] = layer_target_rms
            cosine_by_layer[layer] = layer_cosine
            transport_gradients = (
                self._residual_phase_transport_parameter_gradients(
                    layer,
                    transport_cache,
                    transport_error,
                    token_count=token_count,
                )
            )
            for name, gradient in transport_gradients.items():
                self._add_gradient(
                    gradients,
                    name,
                    (
                        gradient
                        * self.config.residual_phase_transport_lr_scale
                    ),
                )
        return gradients, {
            "fit_rms": fit_rms,
            "target_rms": target_rms,
            "cosine": cosine,
            "fit_by_layer": fit_by_layer,
            "target_by_layer": target_by_layer,
            "cosine_by_layer": cosine_by_layer,
        }

    def _settle_residual_error_pc(
        self,
        free_states: list[Tensor],
        target_state: Tensor,
        *,
        inference_lr: float,
        context_offsets: list[Tensor] | None = None,
        target_state_index: int | None = None,
        upper_proxy_target_state: Tensor | None = None,
    ) -> tuple[list[Tensor], dict[str, float]]:
        if target_state_index is None:
            target_state_index = self.config.num_layers
        states = [state.clone() for state in free_states]
        initial_energy = self._residual_error_pc_energy(
            states,
            free_states,
            target_state,
            context_offsets=context_offsets,
            target_state_index=target_state_index,
            upper_proxy_target_state=upper_proxy_target_state,
        )
        current_energy = initial_energy
        accepted_lr_total = 0.0
        accepted_steps = 0
        backtracking_rejections = 0
        for _ in range(self.config.inference_steps):
            deltas, _, _ = self._residual_error_pc_state_deltas(
                states,
                free_states,
                target_state,
                context_offsets=context_offsets,
                target_state_index=target_state_index,
                upper_proxy_target_state=upper_proxy_target_state,
            )
            candidate_lr = inference_lr
            trial_count = max(self.config.energy_backtracking_steps, 0) + 1
            accepted = False
            for _trial in range(trial_count):
                candidate_states = [states[0]] + [
                    self._apply_inference_delta(
                        states[layer + 1],
                        delta,
                        candidate_lr,
                    )
                    for layer, delta in enumerate(deltas)
                ]
                candidate_energy = self._residual_error_pc_energy(
                    candidate_states,
                    free_states,
                    target_state,
                    context_offsets=context_offsets,
                    target_state_index=target_state_index,
                    upper_proxy_target_state=upper_proxy_target_state,
                )
                if (
                    self.config.energy_backtracking_steps <= 0
                    or bool(candidate_energy <= current_energy)
                ):
                    states = candidate_states
                    current_energy = candidate_energy
                    accepted_lr_total += candidate_lr
                    accepted_steps += 1
                    accepted = True
                    break
                candidate_lr *= 0.5
                backtracking_rejections += 1
            if not accepted:
                break

        final_energy = self._residual_error_pc_energy(
            states,
            free_states,
            target_state,
            context_offsets=context_offsets,
            target_state_index=target_state_index,
            upper_proxy_target_state=upper_proxy_target_state,
        )
        _, predictions, residuals = self._residual_error_pc_terms(
            states,
            target_state,
            context_offsets=context_offsets,
            target_state_index=target_state_index,
        )
        prediction_signals = self._residual_error_suffix_signals(residuals)
        diagnostics = {
            "update_diag_message_pc_residual_error_energy_before": float(
                initial_energy.item()
            ),
            "update_diag_message_pc_residual_error_energy_after": float(
                final_energy.item()
            ),
            "update_diag_message_pc_residual_error_energy_reduction": float(
                (initial_energy - final_energy).item()
            ),
            "update_diag_message_pc_residual_error_base_rms": float(
                _rms(residuals[0]).item()
            ),
            "update_diag_message_pc_residual_error_final_rms": float(
                _rms(residuals[-1]).item()
            ),
            "update_diag_message_pc_residual_error_prediction_rms": float(
                torch.stack([_rms(prediction) for prediction in predictions])
                .mean()
                .item()
            ),
            "update_diag_message_pc_residual_error_accepted_lr": (
                accepted_lr_total / max(accepted_steps, 1)
            ),
            "update_diag_message_pc_residual_error_accepted_steps": float(
                accepted_steps
            ),
            "update_diag_message_pc_residual_error_monotone_energy": float(
                final_energy <= initial_energy
            ),
            "update_diag_message_pc_residual_error_backtracking_rejections": float(
                backtracking_rejections
            ),
            "update_diag_message_pc_residual_error_suffix_decay": (
                self.config.residual_error_suffix_decay
            ),
            "update_diag_message_pc_residual_error_target_layer_index": float(
                target_state_index
            ),
            "update_diag_message_pc_residual_error_upper_proxy_target_rms": (
                float(
                    _rms(upper_proxy_target_state - free_states[-1]).item()
                )
                if upper_proxy_target_state is not None
                else 0.0
            ),
            "update_diag_message_pc_residual_error_upper_proxy_state_delta_rms": (
                float(_rms(states[-1] - free_states[-1]).item())
                if upper_proxy_target_state is not None
                else 0.0
            ),
        }
        for layer, (prediction, residual) in enumerate(
            zip(predictions, residuals[1:])
        ):
            predictor_state = states[layer + 1]
            if context_offsets is not None:
                predictor_state = predictor_state + context_offsets[layer]
            exact_input_signal = self._residual_error_exact_input_signal(
                layer,
                predictor_state,
                None,
                prediction_signals[layer],
            )
            learned_input_signal = self._residual_error_learned_input_signal(
                layer,
                predictor_state,
                None,
                prediction_signals[layer],
            )
            exact_signal_rms = _rms(exact_input_signal)
            learned_signal_rms = _rms(learned_input_signal)
            diagnostics[
                f"update_diag_message_pc_residual_error_prediction_rms_layer_{layer + 1}"
            ] = float(_rms(prediction).item())
            diagnostics[
                f"update_diag_message_pc_residual_error_residual_rms_layer_{layer + 1}"
            ] = float(_rms(residual).item())
            diagnostics[
                f"update_diag_message_pc_residual_error_state_delta_rms_layer_{layer + 1}"
            ] = float(
                _rms(states[layer + 1] - free_states[layer + 1]).item()
            )
            diagnostics[
                f"update_diag_message_pc_residual_error_exact_transport_signal_rms_layer_{layer + 1}"
            ] = float(exact_signal_rms.item())
            diagnostics[
                f"update_diag_message_pc_residual_error_learned_transport_signal_rms_layer_{layer + 1}"
            ] = float(learned_signal_rms.item())
            diagnostics[
                f"update_diag_message_pc_residual_error_learned_transport_signal_ratio_layer_{layer + 1}"
            ] = float(
                (
                    learned_signal_rms
                    / torch.clamp(
                        exact_signal_rms + learned_signal_rms,
                        min=1e-12,
                    )
                ).item()
            )
        return states, diagnostics

    def _pc_energy(
        self,
        states: list[Tensor],
        y: Tensor,
        *,
        prediction_multipliers: list[float] | None = None,
        output_multipliers: list[float] | None = None,
    ) -> Tensor:
        prediction_energy = states[0].new_zeros(())
        for layer in range(self.config.num_layers):
            prediction = self._block(layer, states[layer]).prediction
            error = states[layer + 1] - prediction
            multiplier = (
                1.0
                if prediction_multipliers is None
                else prediction_multipliers[layer]
            )
            prediction_energy = prediction_energy + (
                0.5 * multiplier * error.square().sum(dim=-1).mean()
            )
        output_energy = states[0].new_zeros(())
        for layer, (state, layer_weight) in enumerate(
            zip(states[1:], self._decoder_state_weights())
        ):
            if layer_weight == 0.0:
                continue
            multiplier = (
                1.0
                if output_multipliers is None
                else output_multipliers[layer]
            )
            output_energy = output_energy + (
                layer_weight
                * multiplier
                * self._decoder_output_energy(state, y)
            )
        return (
            self.config.prediction_precision * prediction_energy
            + self.config.token_feedback_scale * output_energy
        )

    def _pc_energy_state_deltas(
        self,
        states: list[Tensor],
        y: Tensor,
        *,
        prediction_multipliers: list[float] | None = None,
        output_multipliers: list[float] | None = None,
    ) -> tuple[list[Tensor], Tensor, Tensor]:
        block_caches = [
            self._block(layer, states[layer])
            for layer in range(self.config.num_layers)
        ]
        prediction_errors = [
            states[layer + 1] - block_caches[layer].prediction
            for layer in range(self.config.num_layers)
        ]
        output_feedbacks: list[Tensor | None] = []
        for state, layer_weight in zip(states[1:], self._decoder_state_weights()):
            if layer_weight == 0.0:
                output_feedbacks.append(None)
                continue
            output_feedback, _ = self._decoder_negative_state_gradient(state, y)
            output_feedbacks.append(layer_weight * output_feedback)
        learned_feedbacks: list[Tensor] | None = None
        if self.config.learned_feedback_proposal_scale > 0:
            _, learned_feedbacks = self._feedback_path(states, y)

        deltas: list[Tensor] = []
        feedback_rms = states[0].new_zeros(())
        prediction_rms = states[0].new_zeros(())
        for layer in range(self.config.num_layers):
            own_multiplier = (
                1.0
                if prediction_multipliers is None
                else prediction_multipliers[layer]
            )
            delta = (
                -self.config.prediction_precision
                * own_multiplier
                * prediction_errors[layer]
            )
            if layer + 1 < self.config.num_layers:
                higher_multiplier = (
                    1.0
                    if prediction_multipliers is None
                    else prediction_multipliers[layer + 1]
                )
                delta = delta + (
                    self.config.prediction_precision
                    * higher_multiplier
                    * self._block_input_signal(
                        layer + 1,
                        states[layer + 1],
                        block_caches[layer + 1],
                        prediction_errors[layer + 1],
                    )
                )
            output_feedback = output_feedbacks[layer]
            if output_feedback is not None:
                output_multiplier = (
                    1.0
                    if output_multipliers is None
                    else output_multipliers[layer]
                )
                delta = delta + (
                    self.config.token_feedback_scale
                    * output_multiplier
                    * output_feedback
                )
                feedback_rms = feedback_rms + _rms(output_feedback)
            if learned_feedbacks is not None:
                learned_feedback = learned_feedbacks[layer]
                delta = delta + (
                    self.config.token_feedback_scale
                    * self.config.learned_feedback_proposal_scale
                    * learned_feedback
                )
                feedback_rms = feedback_rms + _rms(learned_feedback)
            deltas.append(_cap_token_rms(delta, self.config.state_delta_rms_cap))
            prediction_rms = prediction_rms + _rms(prediction_errors[layer])
        return deltas, feedback_rms, prediction_rms

    def _settle_pc_energy(
        self,
        free_states: list[Tensor],
        y: Tensor,
        *,
        inference_lr: float,
    ) -> tuple[list[Tensor], dict[str, float]]:
        states = [state.clone() for state in free_states]
        initial_energy = self._pc_energy(states, y)
        current_energy = initial_energy
        feedback_rms = states[0].new_zeros(())
        prediction_rms = states[0].new_zeros(())
        accepted_rate_total = 0.0
        accepted_steps = 0
        backtracking_rejections = 0
        acceleration_restarts = 0
        acceleration_beta_total = 0.0
        acceleration_beta_count = 0
        previous_deltas: list[Tensor] | None = None
        previous_directions: list[Tensor] | None = None

        def state_dot(left: list[Tensor], right: list[Tensor]) -> Tensor:
            return torch.stack(
                [
                    (left_value.float() * right_value.float()).sum()
                    for left_value, right_value in zip(left, right)
                ]
            ).sum()

        spike_multiplier = 1.0
        for inference_round in range(self.config.inference_steps):
            (
                prediction_multipliers,
                output_multipliers,
                round_spike_multiplier,
            ) = self._inference_precision_multipliers(
                inference_round,
                inference_lr,
            )
            spike_multiplier = max(spike_multiplier, round_spike_multiplier)
            if prediction_multipliers is not None:
                current_energy = self._pc_energy(
                    states,
                    y,
                    prediction_multipliers=prediction_multipliers,
                    output_multipliers=output_multipliers,
                )
            deltas, step_feedback_rms, step_prediction_rms = (
                self._pc_energy_state_deltas(
                    states,
                    y,
                    prediction_multipliers=prediction_multipliers,
                    output_multipliers=output_multipliers,
                )
            )
            feedback_rms = feedback_rms + step_feedback_rms
            prediction_rms = prediction_rms + step_prediction_rms
            directions = deltas
            acceleration_beta = 0.0
            if (
                self.config.inference_optimizer == "heavy_ball"
                and previous_directions is not None
            ):
                acceleration_beta = self.config.inference_momentum
                directions = [
                    delta + acceleration_beta * previous_direction
                    for delta, previous_direction in zip(
                        deltas,
                        previous_directions,
                    )
                ]
            elif (
                self.config.inference_optimizer == "conjugate_gradient"
                and previous_deltas is not None
                and previous_directions is not None
            ):
                numerator = state_dot(
                    deltas,
                    [
                        delta - previous_delta
                        for delta, previous_delta in zip(deltas, previous_deltas)
                    ],
                )
                denominator = state_dot(previous_deltas, previous_deltas)
                acceleration_beta = max(
                    0.0,
                    float((numerator / denominator.clamp_min(1e-30)).item()),
                )
                directions = [
                    delta + acceleration_beta * previous_direction
                    for delta, previous_direction in zip(
                        deltas,
                        previous_directions,
                    )
                ]
            if directions is not deltas:
                acceleration_beta_total += acceleration_beta
                acceleration_beta_count += 1
                if float(state_dot(directions, deltas).item()) <= 0.0:
                    directions = deltas
                    acceleration_restarts += 1
            if self.config.energy_backtracking_steps <= 0:
                for layer, direction in enumerate(directions):
                    states[layer + 1] = self._apply_inference_delta(
                        states[layer + 1],
                        direction,
                        inference_lr,
                    )
                accepted_rate_total += inference_lr
                accepted_steps += 1
                continue

            accepted = False
            accepted_directions = directions
            direction_candidates = [directions]
            if directions is not deltas:
                direction_candidates.append(deltas)
            for candidate_index, candidate_directions in enumerate(
                direction_candidates
            ):
                candidate_rate = inference_lr
                for _trial in range(self.config.energy_backtracking_steps + 1):
                    candidate_states = [states[0]] + [
                        self._apply_inference_delta(
                            states[layer + 1],
                            direction,
                            candidate_rate,
                        )
                        for layer, direction in enumerate(candidate_directions)
                    ]
                    candidate_energy = self._pc_energy(
                        candidate_states,
                        y,
                        prediction_multipliers=prediction_multipliers,
                        output_multipliers=output_multipliers,
                    )
                    if bool(candidate_energy <= current_energy):
                        states = candidate_states
                        current_energy = candidate_energy
                        accepted_rate_total += candidate_rate
                        accepted_steps += 1
                        accepted_directions = candidate_directions
                        accepted = True
                        if candidate_index > 0:
                            acceleration_restarts += 1
                        break
                    backtracking_rejections += 1
                    candidate_rate *= 0.5
                if accepted:
                    break
            if not accepted:
                break
            if self.config.inference_optimizer == "heavy_ball":
                previous_directions = [
                    direction.clone() for direction in accepted_directions
                ]
            elif self.config.inference_optimizer == "conjugate_gradient":
                previous_deltas = [delta.clone() for delta in deltas]
                previous_directions = [
                    direction.clone() for direction in accepted_directions
                ]

        if self.config.spiking_precision_scale > 0:
            final_energy = self._pc_energy(states, y)
        else:
            final_energy = (
                current_energy
                if self.config.energy_backtracking_steps > 0
                else self._pc_energy(states, y)
            )
        normalizer = float(self.config.inference_steps * self.config.num_layers)
        diagnostics = {
            "update_diag_message_pc_feedback_rms": float(
                (feedback_rms / self.config.inference_steps).item()
            ),
            "update_diag_message_pc_prediction_error_rms": float(
                (prediction_rms / normalizer).item()
            ),
            "update_diag_message_pc_phase_delta_rms": float(
                torch.stack(
                    [
                        _rms(states[index] - free_states[index])
                        for index in range(1, len(states))
                    ]
                ).mean().item()
            ),
            "update_diag_message_pc_energy_before": float(initial_energy.item()),
            "update_diag_message_pc_energy_after": float(final_energy.item()),
            "update_diag_message_pc_energy_reduction": float(
                (initial_energy - final_energy).item()
            ),
            "update_diag_message_pc_inference_accepted_lr": (
                accepted_rate_total / max(accepted_steps, 1)
            ),
            "update_diag_message_pc_inference_total_accepted_lr": (
                accepted_rate_total
            ),
            "update_diag_message_pc_inference_backtracking_rejections": float(
                backtracking_rejections
            ),
            "update_diag_message_pc_inference_acceleration_restarts": float(
                acceleration_restarts
            ),
            "update_diag_message_pc_inference_acceleration_beta": (
                acceleration_beta_total / max(acceleration_beta_count, 1)
            ),
            "update_diag_message_pc_spiking_precision_scale": (
                self.config.spiking_precision_scale
            ),
            "update_diag_message_pc_spiking_precision_multiplier": (
                spike_multiplier
            ),
        }
        for layer in range(self.config.num_layers):
            diagnostics[
                f"update_diag_message_pc_phase_delta_rms_layer_{layer + 1}"
            ] = float(_rms(states[layer + 1] - free_states[layer + 1]).item())
        return states, diagnostics

    def _meta_pc_residuals(
        self,
        feedforward_states: list[Tensor],
        block_caches: list[_AttentionCache],
        displacements: list[Tensor],
    ) -> list[Tensor]:
        residuals: list[Tensor] = []
        for upper_offset in range(1, self.config.num_layers):
            desired_lower = self._meta_pc_feedback_signal(
                upper_offset,
                feedforward_states[upper_offset],
                block_caches[upper_offset],
                displacements[upper_offset],
            )
            desired_lower = _cap_token_rms(
                desired_lower,
                self.config.state_delta_rms_cap,
            )
            residuals.append(
                displacements[upper_offset - 1] - desired_lower
            )
        return residuals

    def _meta_pc_feedback_signal(
        self,
        layer: int,
        lower_state: Tensor,
        cache: _AttentionCache,
        upper_displacement: Tensor,
    ) -> Tensor:
        if self.config.meta_pc_feedback_mode == "exact_vjp":
            return self._block_input_signal(
                layer,
                lower_state,
                cache,
                upper_displacement,
            )

        if self.config.meta_pc_feedback_mode == "stage_reciprocal":
            return self._stage_reciprocal_block_input_signal(
                layer,
                lower_state,
                cache,
                upper_displacement,
            )

        if self.config.meta_pc_feedback_mode == "learned_reciprocal":
            learned_signal = F.linear(
                upper_displacement,
                self.feedback_weights[layer],
            )
            if self.config.meta_pc_state_gated_reciprocal:
                learned_signal = learned_signal + torch.tanh(lower_state) * F.linear(
                    upper_displacement,
                    self.feedback_state_weights[layer],
                )
            if self.config.meta_pc_reciprocal_scale <= 0:
                return learned_signal
            head_signal = self._split_heads(upper_displacement)
            routed_signal = self._merge_heads(
                torch.matmul(cache.weights.transpose(-2, -1), head_signal)
            )
            return learned_signal + (
                self.config.meta_pc_reciprocal_scale
                * F.linear(routed_signal, self.feedback_weights[layer])
            )

        # Error returns over reciprocal attention edges. This preserves the
        # residual identity path without a forward-weight transpose/Jacobian.
        head_signal = self._split_heads(upper_displacement)
        reciprocal_signal = self._merge_heads(
            torch.matmul(cache.weights.transpose(-2, -1), head_signal)
        )
        reciprocal_signal = _rms_norm_vjp(lower_state, reciprocal_signal)
        return upper_displacement + (
            self.config.meta_pc_reciprocal_scale * reciprocal_signal
        )

    def _stage_reciprocal_internal_signals(
        self,
        layer: int,
        cache: _AttentionCache,
        output_signal: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if self.config.ffn_dim <= 0:
            attention_residual_signal = (
                output_signal * cache.attention_prediction_derivative
            )
            return attention_residual_signal, output_signal, None, None
        if (
            cache.ffn_pre_activation is None
            or cache.ffn_activation is None
        ):
            raise RuntimeError("FFN cache is incomplete")
        ffn_residual_signal = output_signal * cache.prediction_derivative
        ffn_activation_signal = F.linear(
            ffn_residual_signal,
            self.feedback_ffn_down_weights[layer],
        )
        ffn_pre_activation_signal = ffn_activation_signal * (
            2.0 * F.relu(cache.ffn_pre_activation)
        )
        ffn_source_signal = F.linear(
            ffn_pre_activation_signal,
            self.feedback_ffn_up_weights[layer],
        )
        attention_output_signal = output_signal + _rms_norm_vjp(
            cache.attention_prediction,
            ffn_source_signal,
        )
        return (
            attention_output_signal * cache.attention_prediction_derivative,
            attention_output_signal,
            ffn_residual_signal,
            ffn_pre_activation_signal,
        )

    def _stage_reciprocal_block_input_signal(
        self,
        layer: int,
        lower_state: Tensor,
        cache: _AttentionCache,
        output_signal: Tensor,
    ) -> Tensor:
        """Return block credit through independently learned local mirrors."""

        attention_residual_signal, lower_direct_signal, _, _ = (
            self._stage_reciprocal_internal_signals(
                layer,
                cache,
                output_signal,
            )
        )
        message_signal = F.linear(
            attention_residual_signal,
            self.feedback_message_weights[layer],
        )
        query_signal, key_signal, value_signal, _ = (
            self._attention_local_gradients(
                cache,
                self._split_heads(message_signal),
            )
        )
        query_signal, key_signal = self._route_projection_signals(
            cache,
            query_signal,
            key_signal,
        )
        source_signal = F.linear(
            self._merge_heads(query_signal),
            self.feedback_query_weights[layer],
        )
        source_signal = source_signal + F.linear(
            self._merge_heads(key_signal),
            self.feedback_key_weights[layer],
        )
        source_signal = source_signal + F.linear(
            self._merge_heads(value_signal),
            self.feedback_value_weights[layer],
        )
        return (
            cache.direct_input_scale * lower_direct_signal
            + cache.source_input_scale
            * _rms_norm_vjp(cache.source_input, source_signal)
        )

    @torch.no_grad()
    def _stage_orthogonal_probe_noise(
        self,
        input_state: Tensor,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        """Build orthogonal local perturbations when a batch spans the stage."""

        input_dim = input_state.shape[-1]
        sample_count = input_state.numel() // input_dim
        order = 1 << max(input_dim - 1, 0).bit_length()
        if order > sample_count:
            noise = torch.empty_like(input_state).reshape(-1, input_dim)
            noise.bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0)
            return noise

        cache_key = (input_state.device, input_state.dtype, order)
        basis = self._stage_probe_bases.get(cache_key)
        if basis is None:
            basis = torch.ones(
                1,
                1,
                device=input_state.device,
                dtype=input_state.dtype,
            )
            while basis.shape[0] < order:
                top = torch.cat((basis, basis), dim=1)
                bottom = torch.cat((basis, -basis), dim=1)
                basis = torch.cat((top, bottom), dim=0)
            self._stage_probe_bases[cache_key] = basis
        noise = basis[:, :input_dim].clone()
        row_sign = torch.empty(
            order,
            1,
            device=input_state.device,
            dtype=input_state.dtype,
        )
        row_sign.bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0)
        column_sign = torch.empty(
            1,
            input_dim,
            device=input_state.device,
            dtype=input_state.dtype,
        )
        column_sign.bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0)
        return noise * row_sign * column_sign

    @torch.no_grad()
    def _linear_stage_mirror_target(
        self,
        input_state: Tensor,
        forward_weight: Tensor,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        """Estimate one linear-stage transpose from local perturbation response."""

        noise = self._stage_orthogonal_probe_noise(
            input_state,
            generator=generator,
        )
        probe_count = noise.shape[0]
        noise_std = self.config.meta_pc_reciprocal_noise_std
        perturbation = noise_std * noise
        response = F.linear(perturbation, forward_weight) / noise_std
        flat_noise = noise.float()
        flat_response = response.reshape(-1, response.shape[-1]).float()
        sample_count = float(max(probe_count, 1))
        return torch.nan_to_num(
            (flat_noise.t() @ flat_response) / sample_count
        )

    @torch.no_grad()
    def _update_stage_reciprocal_feedback(
        self,
        feedforward_states: list[Tensor],
        token_ids: Tensor,
        *,
        current_step: int,
    ) -> dict[str, float]:
        learning_rate = self.config.meta_pc_reciprocal_learning_rate
        update_rms: list[Tensor] = []
        target_cosines: list[Tensor] = []
        stage_update_rms: dict[str, list[Tensor]] = {}
        for layer in range(self.config.num_layers):
            lower_state = feedforward_states[layer]
            cache = self._block(
                layer,
                lower_state,
                token_ids=token_ids,
                sensory_state=feedforward_states[0],
                encoder_memory=self._causal_encoder_memory(
                    feedforward_states,
                    layer,
                ),
            )
            generator = torch.Generator(device=lower_state.device)
            generator.manual_seed(
                1_000_003 * max(current_step, 0) + 97_409 * layer + 193
            )
            stages: list[tuple[str, Tensor, Tensor, Tensor]] = [
                (
                    "query",
                    self.feedback_query_weights[layer],
                    cache.source,
                    self.query_weights[layer],
                ),
                (
                    "key",
                    self.feedback_key_weights[layer],
                    cache.source,
                    self.key_weights[layer],
                ),
                (
                    "value",
                    self.feedback_value_weights[layer],
                    cache.source,
                    self.value_weights[layer],
                ),
                (
                    "message",
                    self.feedback_message_weights[layer],
                    cache.message,
                    self.message_weights[layer],
                ),
            ]
            if self.config.ffn_dim > 0:
                if cache.ffn_source is None or cache.ffn_activation is None:
                    raise RuntimeError("FFN cache is incomplete")
                stages.extend(
                    (
                        (
                            "ffn_up",
                            self.feedback_ffn_up_weights[layer],
                            cache.ffn_source,
                            self.ffn_up_weights[layer],
                        ),
                        (
                            "ffn_down",
                            self.feedback_ffn_down_weights[layer],
                            cache.ffn_activation,
                            self.ffn_down_weights[layer],
                        ),
                    )
                )
            for name, mirror, stage_input, forward_weight in stages:
                target = self._linear_stage_mirror_target(
                    stage_input,
                    forward_weight,
                    generator=generator,
                ).to(mirror.dtype)
                target_cosines.append(
                    F.cosine_similarity(
                        mirror.float().reshape(1, -1),
                        target.float().reshape(1, -1),
                        eps=1e-12,
                    ).squeeze(0)
                )
                delta = learning_rate * (target - mirror)
                mirror.add_(delta)
                measured_rms = _rms(delta)
                update_rms.append(measured_rms)
                stage_update_rms.setdefault(name, []).append(measured_rms)

        if not update_rms:
            return {}
        diagnostics = {
            "update_diag_message_pc_meta_reciprocal_feedback_update_rms": float(
                torch.stack(update_rms).mean().item()
            ),
            "update_diag_message_pc_meta_reciprocal_feedback_target_cosine": float(
                torch.stack(target_cosines).mean().item()
            ),
            "update_diag_message_pc_meta_reciprocal_feedback_layers": float(
                self.config.num_layers
            ),
            "update_diag_message_pc_meta_stage_reciprocal_mirrors": float(
                len(update_rms)
            ),
        }
        for name, values in stage_update_rms.items():
            diagnostics[
                f"update_diag_message_pc_meta_stage_reciprocal_{name}_update_rms"
            ] = float(torch.stack(values).mean().item())
        return diagnostics

    @torch.no_grad()
    def _update_meta_pc_reciprocal_feedback(
        self,
        feedforward_states: list[Tensor],
        token_ids: Tensor,
        *,
        current_step: int,
    ) -> dict[str, float]:
        """Track local block Jacobians with perturbation-driven Hebbian mirrors."""

        if self.config.meta_pc_feedback_mode == "stage_reciprocal":
            return self._update_stage_reciprocal_feedback(
                feedforward_states,
                token_ids,
                current_step=current_step,
            )
        if self.config.meta_pc_feedback_mode != "learned_reciprocal":
            return {}
        learning_rate = self.config.meta_pc_reciprocal_learning_rate
        noise_std = self.config.meta_pc_reciprocal_noise_std
        update_rms: list[Tensor] = []
        state_update_rms: list[Tensor] = []
        target_cosines: list[Tensor] = []
        prediction_cosines: list[Tensor] = []
        width = self.config.hidden_dim
        token_count = float(max(token_ids.numel(), 1))
        for layer in range(1, self.config.num_layers):
            lower_state = feedforward_states[layer]
            generator = torch.Generator(device=lower_state.device)
            generator.manual_seed(
                1_000_003 * max(current_step, 0) + 97_409 * layer + 17
            )
            noise = torch.empty_like(lower_state)
            noise.bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0)
            perturbed_lower_state = lower_state + noise_std * noise
            encoder_memory = self._causal_encoder_memory(
                feedforward_states,
                layer,
            )
            if encoder_memory is lower_state:
                encoder_memory = perturbed_lower_state
            perturbed_cache = self._block(
                layer,
                perturbed_lower_state,
                token_ids=token_ids,
                sensory_state=feedforward_states[0],
                encoder_memory=encoder_memory,
            )
            response = (
                perturbed_cache.prediction - feedforward_states[layer + 1]
            ) / noise_std
            flat_noise = noise.reshape(-1, width).float()
            flat_response = response.reshape(-1, width).float()
            local_jacobian_transpose = (
                flat_noise.t() @ flat_response
            ) / token_count
            local_jacobian_transpose = torch.nan_to_num(
                local_jacobian_transpose
            ).to(self.feedback_weights[layer].dtype)
            feedback_weight = self.feedback_weights[layer]
            target_cosines.append(
                F.cosine_similarity(
                    feedback_weight.float().reshape(1, -1),
                    local_jacobian_transpose.float().reshape(1, -1),
                ).squeeze(0)
            )
            if self.config.meta_pc_state_gated_reciprocal:
                state_weight = self.feedback_state_weights[layer]
                flat_state_noise = (
                    noise * torch.tanh(lower_state)
                ).reshape(-1, width).float()
                predicted_response = (
                    flat_noise @ feedback_weight.float()
                    + flat_state_noise @ state_weight.float()
                )
                response_error = predicted_response - flat_response
                delta = -learning_rate * (
                    flat_noise.t() @ response_error
                ) / token_count
                state_delta = -learning_rate * (
                    flat_state_noise.t() @ response_error
                ) / token_count
                prediction_cosines.append(
                    F.cosine_similarity(
                        predicted_response.reshape(1, -1),
                        flat_response.reshape(1, -1),
                        dim=-1,
                        eps=1e-12,
                    ).squeeze(0)
                )
            else:
                delta = learning_rate * (
                    local_jacobian_transpose - feedback_weight
                )
                state_delta = None
            if self.config.update_rms_cap > 0:
                delta_rms = _rms(delta)
                if float(delta_rms.item()) > self.config.update_rms_cap:
                    delta.mul_(
                        self.config.update_rms_cap
                        / float(delta_rms.item())
                    )
                if state_delta is not None:
                    state_delta_rms = _rms(state_delta)
                    if float(state_delta_rms.item()) > self.config.update_rms_cap:
                        state_delta.mul_(
                            self.config.update_rms_cap
                            / float(state_delta_rms.item())
                        )
            feedback_weight.add_(delta)
            update_rms.append(_rms(delta))
            if state_delta is not None:
                state_weight.add_(state_delta.to(state_weight.dtype))
                state_update_rms.append(_rms(state_delta))

        if not update_rms:
            return {}
        diagnostics = {
            "update_diag_message_pc_meta_reciprocal_feedback_update_rms": float(
                torch.stack(update_rms).mean().item()
            ),
            "update_diag_message_pc_meta_reciprocal_feedback_target_cosine": float(
                torch.stack(target_cosines).mean().item()
            ),
            "update_diag_message_pc_meta_reciprocal_feedback_layers": float(
                len(update_rms)
            ),
        }
        if state_update_rms:
            diagnostics[
                "update_diag_message_pc_meta_state_gated_feedback_update_rms"
            ] = float(torch.stack(state_update_rms).mean().item())
            diagnostics[
                "update_diag_message_pc_meta_state_gated_response_cosine"
            ] = float(torch.stack(prediction_cosines).mean().item())
        return diagnostics

    def _settle_meta_pc(
        self,
        feedforward_states: list[Tensor],
        y: Tensor,
        *,
        inference_lr: float,
        token_ids: Tensor | None = None,
    ) -> tuple[list[Tensor], dict[str, float]]:
        """Infer frozen-prediction error targets using Meta-PC local dynamics."""
        def build_caches(states: list[Tensor]) -> list[_AttentionCache]:
            return [
                self._block(
                    layer,
                    states[layer],
                    token_ids=token_ids,
                    sensory_state=states[0],
                    encoder_memory=self._causal_encoder_memory(
                        states,
                        layer,
                    ),
                )
                for layer in range(self.config.num_layers)
            ]

        initial_block_caches = build_caches(feedforward_states)
        block_caches = initial_block_caches
        feedback_states = feedforward_states
        top_feedback, _ = self._meta_pc_top_feedback(
            feedforward_states[-1],
            y,
        )
        initial_top_feedback = top_feedback
        top_displacement = _cap_token_rms(
            self.config.token_feedback_scale * top_feedback,
            self.config.state_delta_rms_cap,
        )
        displacements = [
            torch.zeros_like(feedforward_states[layer + 1])
            for layer in range(self.config.num_layers)
        ]
        displacements[-1] = top_displacement
        initial_residuals = self._meta_pc_residuals(
            feedforward_states,
            block_caches,
            displacements,
        )
        initial_energy = feedforward_states[0].new_zeros((), dtype=torch.float32)
        if initial_residuals:
            initial_energy = 0.5 * torch.stack(
                [
                    residual.float().square().mean()
                    for residual in initial_residuals
                ]
            ).sum()

        cross_memory_skip_signal = torch.zeros_like(
            feedforward_states[max(self.config.causal_encoder_depth, 1)]
        )
        prediction_geometry_drift = feedforward_states[0].new_zeros(())
        top_feedback_cosine = feedforward_states[0].new_ones(())
        cycle_energies: list[Tensor] = []
        for inference_cycle in range(self.config.inference_steps):
            if (
                self.config.meta_pc_relinearize_inference
                and inference_cycle > 0
            ):
                linearization_states = [feedforward_states[0]] + [
                    feedforward_states[layer + 1] + displacements[layer]
                    for layer in range(self.config.num_layers)
                ]
                feedback_states = linearization_states
                block_caches = build_caches(linearization_states)
                prediction_geometry_drift = torch.stack(
                    [
                        _rms(cache.prediction - initial_cache.prediction)
                        for cache, initial_cache in zip(
                            block_caches,
                            initial_block_caches,
                        )
                    ]
                ).mean()
                current_top_feedback, _ = self._meta_pc_top_feedback(
                    linearization_states[-1],
                    y,
                )
                top_feedback_cosine = F.cosine_similarity(
                    initial_top_feedback.float().reshape(1, -1),
                    current_top_feedback.float().reshape(1, -1),
                    dim=-1,
                ).squeeze(0)
                displacements[-1] = _cap_token_rms(
                    displacements[-1]
                    + inference_lr
                    * self.config.token_feedback_scale
                    * current_top_feedback,
                    self.config.state_delta_rms_cap,
                )
            if self.config.meta_pc_cross_memory_skip_credit:
                cross_memory_skip_signal.zero_()
                for decoder_layer in range(
                    self.config.causal_encoder_depth + 1,
                    self.config.num_layers,
                ):
                    cross_memory_skip_signal.add_(
                        self.config.meta_pc_cross_memory_skip_credit_scale
                        * self._block_cross_memory_signal(
                            decoder_layer,
                            block_caches[decoder_layer],
                            displacements[decoder_layer],
                        )
                    )
                cross_memory_skip_signal = _cap_token_rms(
                    cross_memory_skip_signal,
                    self.config.state_delta_rms_cap,
                )
            # Top-down blocked sweep: each level sees newest superior error.
            for upper_offset in range(
                self.config.num_layers - 1,
                0,
                -1,
            ):
                desired_lower = self._meta_pc_feedback_signal(
                    upper_offset,
                    feedback_states[upper_offset],
                    block_caches[upper_offset],
                    displacements[upper_offset],
                )
                desired_lower = _cap_token_rms(
                    desired_lower,
                    self.config.state_delta_rms_cap,
                )
                if (
                    self.config.meta_pc_cross_memory_skip_credit
                    and upper_offset == self.config.causal_encoder_depth
                ):
                    desired_lower = _cap_token_rms(
                        desired_lower + cross_memory_skip_signal,
                        self.config.state_delta_rms_cap,
                    )
                residual = displacements[upper_offset - 1] - desired_lower
                displacements[upper_offset - 1] = _cap_token_rms(
                    displacements[upper_offset - 1]
                    - inference_lr * residual,
                    self.config.state_delta_rms_cap,
                )
            cycle_residuals = self._meta_pc_residuals(
                feedback_states,
                block_caches,
                displacements,
            )
            cycle_energy = feedforward_states[0].new_zeros(
                (),
                dtype=torch.float32,
            )
            if cycle_residuals:
                cycle_energy = 0.5 * torch.stack(
                    [
                        residual.float().square().mean()
                        for residual in cycle_residuals
                    ]
                ).sum()
            cycle_energies.append(cycle_energy)

        final_residuals = self._meta_pc_residuals(
            feedback_states,
            block_caches,
            displacements,
        )
        final_energy = feedforward_states[0].new_zeros((), dtype=torch.float32)
        if final_residuals:
            final_energy = 0.5 * torch.stack(
                [residual.float().square().mean() for residual in final_residuals]
            ).sum()
        states = [feedforward_states[0]] + [
            feedforward_states[layer + 1] + displacements[layer]
            for layer in range(self.config.num_layers)
        ]
        displacement_rms = torch.stack(
            [_rms(displacement) for displacement in displacements]
        )
        diagnostics = {
            "update_diag_message_pc_meta_energy_before": float(
                initial_energy.item()
            ),
            "update_diag_message_pc_meta_energy_after": float(
                final_energy.item()
            ),
            "update_diag_message_pc_meta_energy_reduction": float(
                (initial_energy - final_energy).item()
            ),
            "update_diag_message_pc_meta_top_displacement_rms": float(
                displacement_rms[-1].item()
            ),
            "update_diag_message_pc_meta_bottom_displacement_rms": float(
                displacement_rms[0].item()
            ),
            "update_diag_message_pc_meta_mean_displacement_rms": float(
                displacement_rms.mean().item()
            ),
            "update_diag_message_pc_meta_uses_frozen_predictions": float(
                not self.config.meta_pc_relinearize_inference
            ),
            "update_diag_message_pc_meta_uses_relinearized_inference": float(
                self.config.meta_pc_relinearize_inference
            ),
            "update_diag_message_pc_meta_prediction_geometry_drift_rms": float(
                prediction_geometry_drift.item()
            ),
            "update_diag_message_pc_meta_top_feedback_cosine": float(
                top_feedback_cosine.item()
            ),
            "update_diag_message_pc_meta_uses_blocked_sweep": 1.0,
            "update_diag_message_pc_meta_uses_exact_vjp": float(
                self.config.meta_pc_feedback_mode == "exact_vjp"
            ),
            "update_diag_message_pc_meta_uses_reciprocal_attention": float(
                self.config.meta_pc_feedback_mode == "reciprocal_attention"
            ),
            "update_diag_message_pc_meta_uses_learned_reciprocal": float(
                self.config.meta_pc_feedback_mode == "learned_reciprocal"
            ),
            "update_diag_message_pc_meta_uses_stage_reciprocal": float(
                self.config.meta_pc_feedback_mode == "stage_reciprocal"
            ),
            "update_diag_message_pc_meta_uses_cross_memory_skip_credit": float(
                self.config.meta_pc_cross_memory_skip_credit
            ),
            "update_diag_message_pc_meta_cross_memory_skip_credit_rms": float(
                _rms(cross_memory_skip_signal).item()
            ),
            "update_diag_message_pc_meta_cross_memory_skip_credit_scale": (
                self.config.meta_pc_cross_memory_skip_credit_scale
            ),
            "update_diag_message_pc_meta_reciprocal_scale": (
                self.config.meta_pc_reciprocal_scale
            ),
            "update_diag_message_pc_meta_reciprocal_learning_rate": (
                self.config.meta_pc_reciprocal_learning_rate
            ),
            "update_diag_message_pc_meta_reciprocal_noise_std": (
                self.config.meta_pc_reciprocal_noise_std
            ),
            "update_diag_message_pc_meta_uses_pre_settle_reciprocal_update": float(
                self.config.meta_pc_reciprocal_update_before_settle
            ),
            "update_diag_message_pc_meta_uses_state_gated_reciprocal": float(
                self.config.meta_pc_state_gated_reciprocal
            ),
            "update_diag_message_pc_meta_uses_decoder_output_feedback": float(
                self.config.meta_pc_output_feedback_mode
                == "decoder_transpose"
            ),
            "update_diag_message_pc_meta_uses_fixed_output_feedback": float(
                self.config.meta_pc_output_feedback_mode == "fixed_random"
            ),
            "update_diag_message_pc_meta_uses_kp_output_feedback": float(
                self.config.meta_pc_output_feedback_mode == "kp_learned"
            ),
            "update_diag_message_pc_meta_uses_factorized_output_feedback": float(
                self.config.meta_pc_output_feedback_mode == "kp_factorized"
            ),
        }
        for layer, displacement in enumerate(displacements):
            diagnostics[
                f"update_diag_message_pc_phase_delta_rms_layer_{layer + 1}"
            ] = float(_rms(displacement).item())
        for cycle, energy in enumerate(cycle_energies, start=1):
            diagnostics[
                f"update_diag_message_pc_meta_energy_cycle_{cycle}"
            ] = float(energy.item())
        for layer, residual in enumerate(final_residuals, start=1):
            diagnostics[
                f"update_diag_message_pc_meta_residual_rms_layer_{layer}"
            ] = float(_rms(residual).item())
        return states, diagnostics

    def _settle(
        self,
        free_states: list[Tensor],
        y: Tensor,
        *,
        inference_lr: float | None = None,
        token_ids: Tensor | None = None,
    ) -> tuple[list[Tensor], dict[str, float]]:
        effective_lr = self.config.inference_lr if inference_lr is None else inference_lr
        if self.config.inference_mode == "pc_energy":
            return self._settle_pc_energy(free_states, y, inference_lr=effective_lr)
        if self.config.inference_mode == "meta_pc":
            return self._settle_meta_pc(
                free_states,
                y,
                inference_lr=effective_lr,
                token_ids=token_ids,
            )
        return self._settle_heuristic(free_states, y, inference_lr=effective_lr)

    @staticmethod
    def _cross_entropy_gradient(state: Tensor, y: Tensor, probabilities: Tensor) -> tuple[Tensor, Tensor]:
        flat_state = state.reshape(-1, state.shape[-1])
        flat_y = y.reshape(-1)
        flat_probabilities = probabilities.reshape(-1, probabilities.shape[-1])
        count = float(max(flat_y.numel(), 1))
        weight_gradient = flat_probabilities.t() @ flat_state
        target_sum = torch.zeros_like(weight_gradient)
        target_sum.index_add_(0, flat_y, flat_state)
        weight_gradient = (weight_gradient - target_sum) / count
        bias_gradient = flat_probabilities.sum(dim=0)
        target_count = torch.zeros_like(bias_gradient)
        target_count.index_add_(0, flat_y, torch.ones_like(flat_y, dtype=bias_gradient.dtype))
        return weight_gradient, (bias_gradient - target_count) / count

    @staticmethod
    def _feedback_embedding_gradient(
        probabilities: Tensor,
        y: Tensor,
        source_gradient: Tensor,
    ) -> Tensor:
        flat_probabilities = probabilities.reshape(-1, probabilities.shape[-1])
        flat_source_gradient = source_gradient.reshape(-1, source_gradient.shape[-1])
        flat_y = y.reshape(-1)
        count = float(max(flat_y.numel(), 1))
        target_sum = torch.zeros(
            flat_probabilities.shape[-1],
            flat_source_gradient.shape[-1],
            device=flat_source_gradient.device,
            dtype=flat_source_gradient.dtype,
        )
        target_sum.index_add_(0, flat_y, flat_source_gradient)
        return (target_sum - flat_probabilities.t() @ flat_source_gradient) / count

    def _add_gradient(self, gradients: dict[str, Tensor], name: str, value: Tensor) -> None:
        value = value.to(dtype=dict(self.named_parameters())[name].dtype)
        if name in gradients:
            gradients[name].add_(value)
        else:
            gradients[name] = value

    def _adam_update(
        self,
        name: str,
        parameter: Tensor,
        gradient: Tensor,
        *,
        lr: float,
        weight_decay: float,
    ) -> float:
        if lr <= 0:
            return 0.0
        gradient = torch.nan_to_num(gradient)
        gradient_rms = _rms(gradient)
        if self.config.gradient_rms_cap > 0 and float(gradient_rms.item()) > self.config.gradient_rms_cap:
            gradient = gradient * (self.config.gradient_rms_cap / float(gradient_rms.item()))
        mean_name, variance_name = self._adam_buffer_names[name]
        mean = getattr(self, mean_name)
        variance = getattr(self, variance_name)
        mean.mul_(self.config.adam_beta1).add_(gradient, alpha=1.0 - self.config.adam_beta1)
        variance.mul_(self.config.adam_beta2).addcmul_(gradient, gradient, value=1.0 - self.config.adam_beta2)
        step = int(self.local_adam_step.item())
        mean_hat = mean / (1.0 - self.config.adam_beta1**step)
        variance_hat = variance / (1.0 - self.config.adam_beta2**step)
        update = mean_hat / (variance_hat.sqrt() + self.config.adam_eps)
        delta = update.mul(-lr)
        if weight_decay > 0:
            delta.add_(parameter, alpha=-lr * weight_decay)
        delta_rms = _rms(delta)
        if self.config.update_rms_cap > 0 and float(delta_rms.item()) > self.config.update_rms_cap:
            delta.mul_(self.config.update_rms_cap / float(delta_rms.item()))
            delta_rms = delta.new_tensor(self.config.update_rms_cap)
        parameter.add_(delta)
        if name.startswith(
            (
                "query_weights.",
                "key_weights.",
                "inverse_query_weights.",
                "inverse_key_weights.",
            )
        ):
            target_rms = 1.0 / math.sqrt(self.config.hidden_dim)
            parameter.mul_(target_rms / _rms(parameter).clamp_min(1e-12).to(parameter.dtype))
        return float(delta_rms.item())

    def _muon_update(
        self,
        name: str,
        parameter: Tensor,
        gradient: Tensor,
        *,
        lr: float,
        weight_decay: float,
        momentum: float | None = None,
    ) -> float:
        if lr <= 0:
            return 0.0
        gradient = torch.nan_to_num(gradient)
        gradient_rms = _rms(gradient)
        if (
            self.config.gradient_rms_cap > 0
            and float(gradient_rms.item()) > self.config.gradient_rms_cap
        ):
            gradient = gradient * (
                self.config.gradient_rms_cap / float(gradient_rms.item())
            )
        momentum_name, variance_name = self._muon_buffer_names[name]
        momentum_buffer = getattr(self, momentum_name)
        variance_buffer = getattr(self, variance_name)
        effective_momentum = self.config.muon_momentum if momentum is None else momentum
        momentum_buffer.lerp_(gradient, 1.0 - effective_momentum)
        direction = torch.lerp(
            gradient,
            momentum_buffer,
            effective_momentum,
        )
        direction = _polar_orthogonalize(direction, self.config.muon_ns_steps)

        reduction_dim = -1 if parameter.shape[-2] >= parameter.shape[-1] else -2
        variance_mean = direction.float().square().mean(
            dim=reduction_dim,
            keepdim=True,
        )
        reduced_size = direction.shape[reduction_dim]
        variance_norm = (variance_mean.sum() * reduced_size).sqrt()
        variance_buffer.lerp_(
            variance_mean.to(variance_buffer.dtype),
            1.0 - self.config.muon_beta2,
        )
        variance_scale = variance_buffer.clamp_min(1e-10).rsqrt()
        scaled_norm = (
            variance_mean
            * reduced_size
            * variance_scale.float().square()
        ).sum().sqrt()
        direction = direction * (
            variance_scale * (variance_norm / scaled_norm.clamp_min(1e-10))
        ).to(direction.dtype)

        shape_lr = lr * math.sqrt(
            max(1.0, parameter.shape[-2] / parameter.shape[-1])
        )
        delta = direction.mul(-shape_lr)
        if weight_decay > 0:
            cautious_mask = (direction * parameter) >= 0
            delta.add_(
                parameter * cautious_mask,
                alpha=-shape_lr * weight_decay,
            )
        delta_rms = _rms(delta)
        if (
            self.config.update_rms_cap > 0
            and float(delta_rms.item()) > self.config.update_rms_cap
        ):
            delta.mul_(self.config.update_rms_cap / float(delta_rms.item()))
            delta_rms = delta.new_tensor(self.config.update_rms_cap)
        parameter.add_(delta)
        if name.startswith(
            (
                "query_weights.",
                "key_weights.",
                "inverse_query_weights.",
                "inverse_key_weights.",
            )
        ):
            target_rms = 1.0 / math.sqrt(self.config.hidden_dim)
            parameter.mul_(
                target_rms / _rms(parameter).clamp_min(1e-12).to(parameter.dtype)
            )
        return float(delta_rms.item())

    def _meta_pc_variance_normalize_forward_weights(self) -> float:
        """Apply Meta-PCN's sample-variance spectral proxy to block weights."""
        scales: list[float] = []
        forward_weights = (
            *self.query_weights,
            *self.key_weights,
            *self.value_weights,
            *self.message_weights,
            *self.cross_query_weights,
            *self.cross_key_weights,
            *self.cross_value_weights,
            *self.cross_message_weights,
            *self.ffn_up_weights,
            *self.ffn_down_weights,
        )
        for weight in forward_weights:
            if weight.ndim != 2:
                continue
            standard_deviation = weight.float().std(unbiased=False)
            spectral_proxy = (
                math.sqrt(weight.shape[0]) + math.sqrt(weight.shape[1])
            ) * standard_deviation
            scale = float(spectral_proxy.item())
            if scale <= 1e-12:
                continue
            weight.div_(spectral_proxy.to(weight.dtype))
            scales.append(scale)
        return sum(scales) / max(len(scales), 1)

    def _phase_block_gradients(
        self,
        layer: int,
        lower_state: Tensor,
        target_state: Tensor,
        *,
        route_lower_state: Tensor | None,
        causal_mask: Tensor,
        token_count: float,
        branch_scale: float = 1.0,
        token_ids: Tensor | None = None,
        sensory_state: Tensor | None = None,
        encoder_memory: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Tensor]:
        scale = (
            1.0
            if self.config.block_state_mode == "nanochat_residual"
            else self._block_weight_scale() / math.sqrt(2.0)
        )
        cache = self._block(
            layer,
            lower_state,
            branch_scale=branch_scale,
            token_ids=token_ids,
            sensory_state=sensory_state,
            encoder_memory=encoder_memory,
        )
        prediction_error = cache.prediction - target_state
        if self.config.meta_pc_feedback_mode == "stage_reciprocal":
            (
                attention_residual_gradient,
                lower_direct_gradient,
                ffn_residual_gradient,
                ffn_pre_activation_gradient,
            ) = self._stage_reciprocal_internal_signals(
                layer,
                cache,
                prediction_error,
            )
        else:
            (
                attention_residual_gradient,
                lower_direct_gradient,
                ffn_residual_gradient,
                ffn_pre_activation_gradient,
            ) = self._block_internal_signals(
                layer,
                cache,
                prediction_error,
            )
        flat_attention_residual_gradient = attention_residual_gradient.reshape(
            -1,
            self.config.hidden_dim,
        )
        flat_source = cache.source.reshape(-1, self.config.hidden_dim)
        flat_message = cache.message.reshape(-1, self.config.hidden_dim)
        phase_gradients = {
            f"message_weights.{layer}": (
                scale
                * (flat_attention_residual_gradient.t() @ flat_message)
                / token_count
            ),
        }
        if self.config.block_state_mode != "nanochat_residual":
            phase_gradients[f"self_weights.{layer}"] = (
                scale
                * (flat_attention_residual_gradient.t() @ flat_source)
                / token_count
            )
            phase_gradients[f"state_biases.{layer}"] = (
                flat_attention_residual_gradient.mean(dim=0)
            )
        if self.config.ffn_dim > 0:
            if (
                cache.ffn_source is None
                or cache.ffn_activation is None
                or ffn_residual_gradient is None
                or ffn_pre_activation_gradient is None
            ):
                raise RuntimeError("FFN gradient cache is incomplete")
            flat_ffn_residual_gradient = ffn_residual_gradient.reshape(
                -1,
                self.config.hidden_dim,
            )
            flat_ffn_activation = cache.ffn_activation.reshape(
                -1,
                self.config.ffn_dim,
            )
            flat_ffn_pre_activation_gradient = (
                ffn_pre_activation_gradient.reshape(
                    -1,
                    self.config.ffn_dim,
                )
            )
            flat_ffn_source = cache.ffn_source.reshape(
                -1,
                self.config.hidden_dim,
            )
            phase_gradients[f"ffn_down_weights.{layer}"] = (
                flat_ffn_residual_gradient.t() @ flat_ffn_activation
            ) / token_count
            phase_gradients[f"ffn_up_weights.{layer}"] = (
                flat_ffn_pre_activation_gradient.t() @ flat_ffn_source
            ) / token_count

        message_feedback = (
            self.feedback_message_weights[layer]
            if self.config.meta_pc_feedback_mode == "stage_reciprocal"
            else self.message_weights[layer].t()
        )
        message_gradient = scale * F.linear(
            attention_residual_gradient,
            message_feedback,
        )
        head_message_gradient = self._split_heads(message_gradient)
        (
            local_query_gradient,
            local_key_gradient,
            value_gradient,
            local_score_gradient,
        ) = self._attention_local_gradients(cache, head_message_gradient)
        if self.residual_stream_scales is not None:
            local_query_source, local_key_source = (
                self._route_projection_signals(
                    cache,
                    local_query_gradient,
                    local_key_gradient,
                )
            )
            source_input_gradient = lower_direct_gradient
            if self.config.sensory_reinjection_mode == "pre_block":
                query_feedback = (
                    self.feedback_query_weights[layer]
                    if self.config.meta_pc_feedback_mode == "stage_reciprocal"
                    else self.query_weights[layer].t()
                )
                key_feedback = (
                    self.feedback_key_weights[layer]
                    if self.config.meta_pc_feedback_mode == "stage_reciprocal"
                    else self.key_weights[layer].t()
                )
                value_feedback = (
                    self.feedback_value_weights[layer]
                    if self.config.meta_pc_feedback_mode == "stage_reciprocal"
                    else self.value_weights[layer].t()
                )
                source_signal = F.linear(
                    self._merge_heads(local_query_source),
                    query_feedback,
                )
                source_signal.add_(
                    F.linear(
                        self._merge_heads(local_key_source),
                        key_feedback,
                    )
                )
                source_signal.add_(
                    F.linear(
                        self._merge_heads(value_gradient),
                        value_feedback,
                    )
                )
                source_input_gradient = source_input_gradient + _rms_norm_vjp(
                    cache.source_input,
                    source_signal,
                )
            scale_gradient = torch.zeros_like(self.residual_stream_scales)
            channel_gradient = (
                source_input_gradient * lower_state
            ).sum(dim=(0, 1)) / token_count
            if self.residual_stream_scales.ndim == 1:
                scale_gradient[layer] = channel_gradient.sum()
            else:
                scale_gradient[layer] = channel_gradient
            phase_gradients["residual_stream_scales"] = scale_gradient
        flat_value_gradient = self._merge_heads(value_gradient).reshape(
            -1,
            self.config.hidden_dim,
        )
        phase_gradients[f"value_weights.{layer}"] = (
            flat_value_gradient.t() @ flat_source
        ) / token_count
        token_value_key = str(layer)
        if token_value_key in self.token_value_tables:
            if cache.token_ids is None:
                raise RuntimeError("token value cache is missing token IDs")
            token_value_gradient = torch.zeros_like(
                self.token_value_tables[token_value_key]
            )
            token_value_gradient.index_add_(
                0,
                cache.token_ids.reshape(-1),
                (
                    self.config.token_value_scale
                    * flat_value_gradient
                ),
            )
            phase_gradients[f"token_value_tables.{token_value_key}"] = (
                token_value_gradient / token_count
            )

        route_tv = lower_state.new_zeros(())
        route_source = flat_source
        route_gradient_cache = cache
        route_score_gradient = local_score_gradient
        if self.config.route_update_mode == "token_error_teacher":
            if route_lower_state is None:
                raise ValueError(
                    "token_error_teacher requires the free lower-layer state"
                )
            route_cache = self._block(
                layer,
                route_lower_state,
                branch_scale=branch_scale,
                token_ids=token_ids,
                sensory_state=sensory_state,
                encoder_memory=encoder_memory,
            )
            route_prediction_error = route_cache.prediction - target_state
            if self.config.meta_pc_feedback_mode == "stage_reciprocal":
                route_prediction_gradient, _, _, _ = (
                    self._stage_reciprocal_internal_signals(
                        layer,
                        route_cache,
                        route_prediction_error,
                    )
                )
                route_message_feedback = self.feedback_message_weights[layer]
            else:
                route_prediction_gradient, _, _, _ = (
                    self._block_internal_signals(
                        layer,
                        route_cache,
                        route_prediction_error,
                    )
                )
                route_message_feedback = self.message_weights[layer].t()
            route_message_gradient = scale * F.linear(
                route_prediction_gradient,
                route_message_feedback,
            )
            route_head_message_gradient = self._split_heads(
                route_message_gradient
            )
            _, _, _, route_score_gradient = self._attention_local_gradients(
                route_cache,
                route_head_message_gradient,
            )
            query_gradient, key_gradient, route_tv = (
                self._cosine_route_teacher_gradients(
                    route_cache,
                    route_head_message_gradient,
                    causal_mask=causal_mask,
                    weight_by_credit=True,
                )
            )
            route_source = route_cache.source.reshape(
                -1,
                self.config.hidden_dim,
            )
            route_gradient_cache = route_cache
        elif (
            self.config.inference_mode == "heuristic"
            and self.config.route_update_mode == "cosine_teacher"
        ):
            query_gradient, key_gradient, route_tv = (
                self._cosine_route_teacher_gradients(
                    cache,
                    head_message_gradient,
                    causal_mask=causal_mask,
                    weight_by_credit=False,
                )
            )
        else:
            query_gradient = local_query_gradient
            key_gradient = local_key_gradient
        query_gradient, key_gradient = self._route_projection_signals(
            route_gradient_cache,
            query_gradient,
            key_gradient,
        )
        flat_query_gradient = self._merge_heads(query_gradient).reshape(
            -1,
            self.config.hidden_dim,
        )
        flat_key_gradient = self._merge_heads(key_gradient).reshape(
            -1,
            self.config.hidden_dim,
        )
        phase_gradients[f"query_weights.{layer}"] = (
            flat_query_gradient.t() @ route_source
        ) / token_count
        phase_gradients[f"key_weights.{layer}"] = (
            flat_key_gradient.t() @ route_source
        ) / token_count

        if cache.cross_attention is not None:
            cross = cache.cross_attention
            cross_output_gradient = cross.gate * attention_residual_gradient
            flat_cross_output_gradient = cross_output_gradient.reshape(
                -1,
                self.config.hidden_dim,
            )
            phase_gradients[f"cross_message_weights.{layer}"] = (
                flat_cross_output_gradient.t()
                @ cross.message.reshape(-1, self.config.hidden_dim)
            ) / token_count
            if self.cross_attention_gates is None:
                raise RuntimeError("cross-attention gates are not initialized")
            gate_gradient = torch.zeros_like(self.cross_attention_gates)
            gate_gradient[layer] = (
                attention_residual_gradient * cross.output
            ).sum() * cross.gate * (1.0 - cross.gate) / token_count
            phase_gradients["cross_attention_gates"] = gate_gradient

            cross_message_gradient = F.linear(
                cross_output_gradient,
                self.cross_message_weights[layer].t(),
            )
            (
                cross_query_gradient,
                cross_key_gradient,
                cross_value_gradient,
                _,
            ) = self._attention_local_gradients(
                cross,
                self._split_heads(cross_message_gradient),
            )
            cross_query_gradient, cross_key_gradient = (
                self._route_projection_signals(
                    cross,
                    cross_query_gradient,
                    cross_key_gradient,
                )
            )
            flat_cross_query_gradient = self._merge_heads(
                cross_query_gradient
            ).reshape(-1, self.config.hidden_dim)
            flat_cross_key_gradient = self._merge_heads(
                cross_key_gradient
            ).reshape(-1, self.config.hidden_dim)
            flat_cross_value_gradient = self._merge_heads(
                cross_value_gradient
            ).reshape(-1, self.config.hidden_dim)
            phase_gradients[f"cross_query_weights.{layer}"] = (
                flat_cross_query_gradient.t()
                @ cross.query_source.reshape(-1, self.config.hidden_dim)
            ) / token_count
            phase_gradients[f"cross_key_weights.{layer}"] = (
                flat_cross_key_gradient.t()
                @ cross.memory.reshape(-1, self.config.hidden_dim)
            ) / token_count
            phase_gradients[f"cross_value_weights.{layer}"] = (
                flat_cross_value_gradient.t()
                @ cross.memory.reshape(-1, self.config.hidden_dim)
            ) / token_count

            if (
                token_ids is not None
                and sensory_state is not None
                and self.config.causal_encoder_depth == 0
            ):
                cross_memory_signal = F.linear(
                    self._merge_heads(cross_key_gradient),
                    self.cross_key_weights[layer].t(),
                )
                cross_memory_signal = cross_memory_signal + F.linear(
                    self._merge_heads(cross_value_gradient),
                    self.cross_value_weights[layer].t(),
                )
                sensory_signal = _rms_norm_vjp(
                    cross.memory_input,
                    cross_memory_signal,
                )
                embedding_state_gradient = self._embedding_state_vjp(
                    token_ids,
                    sensory_state,
                    sensory_signal,
                )
                embedding_gradient = torch.zeros_like(self.embedding.weight)
                embedding_gradient.index_add_(
                    0,
                    token_ids.reshape(-1),
                    embedding_state_gradient.reshape(
                        -1,
                        self.config.hidden_dim,
                    ),
                )
                phase_gradients["embedding.weight"] = (
                    embedding_gradient / token_count
                )

        if self.config.inference_mode == "heuristic":
            generative_prediction = self._generative_prediction(layer, target_state)
            generative_gradient = (generative_prediction - lower_state) * (
                1.0 - generative_prediction.square()
            )
            phase_gradients[f"generative_weights.{layer}"] = (
                generative_gradient.reshape(-1, self.config.hidden_dim).t()
                @ _rms_norm(target_state).reshape(-1, self.config.hidden_dim)
            ) / token_count

        attention_entropy = -(
            cache.weights.clamp_min(1e-12)
            * cache.weights.clamp_min(1e-12).log()
        ).sum(dim=-1).mean()
        return (
            phase_gradients,
            attention_entropy,
            route_tv,
            _rms(route_score_gradient),
            _rms(cache.prediction - target_state),
        )

    def _hierarchical_error_local_update(
        self,
        x: Tensor,
        y: Tensor,
        *,
        decoder_lr: float,
        state_lr: float,
        embed_lr: float,
        predictor_state_lr: float | None,
        weight_decay: float,
        current_step: int,
        return_sequence_loss: bool,
        return_token_loss: bool,
        return_final_state: bool,
    ) -> float | tuple[float, Tensor] | tuple[float, Tensor, Tensor] | tuple[float, Tensor, Tensor, Tensor]:
        free_states, _ = self._forward_states(x, with_cache=False)
        base_depth = self.config.hierarchical_error_base_depth
        factorized_keys: list[Tensor] = []
        factorized_values: list[Tensor] = []
        factorized_gates: list[Tensor] = []
        persistent_context_cache: _HierarchicalContextCache | None = None
        node_residual_cache: _NodeResidualHierarchyCache | None = None
        if self.config.hierarchical_error_binding_mode == "factorized":
            if (
                self.config.hierarchical_error_context_mode != "instant"
                and not self.config.sps_node_residual_hierarchy
            ):
                persistent_context_cache = (
                    self._hierarchical_error_context_cache(
                        free_states,
                        candidates_override=[
                            self._state_context(free_states[layer])
                            for layer in range(
                                base_depth,
                                self.config.num_layers,
                            )
                        ],
                    )
                )
            (
                deployed_state,
                error_predictions,
                modulated_error_predictions,
                top_down_correction,
                factorized_keys,
                factorized_values,
                factorized_gates,
            ) = self._hierarchical_error_factorized_pass(
                free_states,
                contexts=(
                    persistent_context_cache.contexts
                    if persistent_context_cache is not None
                    else None
                ),
            )
            if self.config.sps_node_residual_hierarchy:
                node_residual_cache = self._node_residual_hierarchy_cache(
                    free_states
                )
        else:
            if self.config.hierarchical_error_context_mode != "instant":
                persistent_context_cache = (
                    self._hierarchical_error_context_cache(free_states)
                )
            (
                deployed_state,
                error_predictions,
                modulated_error_predictions,
                top_down_correction,
            ) = self._hierarchical_error_pass(
                free_states,
                contexts=(
                    persistent_context_cache.contexts
                    if persistent_context_cache is not None
                    else None
                ),
            )
        settled_states = free_states
        settle_diagnostics: dict[str, float] = {}
        if self.config.hierarchical_error_binding_mode == "factorized":
            settled_states, settle_diagnostics = (
                self._settle_hierarchical_factorized(
                    free_states,
                    y,
                    inference_lr=self._inference_lr(current_step),
                )
            )
        outcome_readout_logits = self._hierarchical_error_outcome_logits(
            top_down_correction
        )
        deployment_logits = (
            self._decoder_logits(deployed_state) + outcome_readout_logits
        )
        token_loss = F.cross_entropy(
            deployment_logits.reshape(-1, self.config.vocab_size),
            y.reshape(-1),
            reduction="none",
        ).view_as(y)
        loss = float(token_loss.mean().item())
        gradients: dict[str, Tensor] = {}
        token_count = float(max(x.numel(), 1))

        decoder_features = self._decoder_features(deployed_state)
        probabilities = F.softmax(deployment_logits, dim=-1)
        decoder_weight_gradient, decoder_bias_gradient = (
            self._cross_entropy_gradient(
                decoder_features,
                y,
                probabilities,
            )
        )
        (
            margin_weight_gradient,
            margin_bias_gradient,
            winner_margin_active,
        ) = self._winner_margin_decoder_gradients(
            decoder_features,
            y,
            deployment_logits,
        )
        decoder_weight_gradient.add_(margin_weight_gradient)
        decoder_bias_gradient.add_(margin_bias_gradient)
        self._add_gradient(gradients, "decoder_weight", decoder_weight_gradient)
        self._add_gradient(gradients, "decoder_bias", decoder_bias_gradient)

        (
            bottom_up_residual,
            _,
            target_active_fraction,
        ) = self._hierarchical_error_target(
            (
                deployed_state
                if self.config.sps_node_residual_hierarchy
                else free_states[base_depth]
            ),
            y,
        )
        deployed_negative_gradient, _ = self._decoder_negative_state_gradient(
            deployed_state,
            y,
            extra_logits=outcome_readout_logits,
        )
        token_credit_prediction_gradient: Tensor | None = None
        token_credit_gradient_rms = free_states[0].new_zeros(())
        if self.config.hierarchical_error_token_credit_scale > 0:
            token_credit_prediction_gradient = (
                self.config.hierarchical_error_token_credit_scale
                * _cap_token_rms_vjp(
                    modulated_error_predictions[0],
                    (
                        -self.config.hierarchical_error_top_down_scale
                        * deployed_negative_gradient
                    ),
                    self.config.state_delta_rms_cap,
                )
            )
            token_credit_gradient_rms = _rms(
                token_credit_prediction_gradient
            )
        outcome_readout_prediction_gradient: Tensor | None = None
        outcome_readout_gradient_rms = free_states[0].new_zeros(())
        if self.config.hierarchical_error_outcome_readout_scale > 0:
            code_gradient = self._hierarchical_error_outcome_gradient(
                probabilities,
                y,
            )
            code_gradient = self._expand_prediction_signal(
                code_gradient,
                modulated_error_predictions[0],
            )
            outcome_readout_prediction_gradient = _cap_token_rms_vjp(
                modulated_error_predictions[0],
                code_gradient,
                self.config.state_delta_rms_cap,
            )
            outcome_readout_gradient_rms = _rms(
                outcome_readout_prediction_gradient
            )
        extra_prediction_gradient = token_credit_prediction_gradient
        if outcome_readout_prediction_gradient is not None:
            extra_prediction_gradient = (
                outcome_readout_prediction_gradient
                if extra_prediction_gradient is None
                else extra_prediction_gradient
                + outcome_readout_prediction_gradient
            )
        base_error = bottom_up_residual
        base_error_rms = _rms(bottom_up_residual)
        causal_mask = self._causal_mask(
            free_states[0].shape[1],
            x.device,
        )
        attention_entropy = free_states[0].new_zeros(())
        route_tv = free_states[0].new_zeros(())
        route_credit_rms = free_states[0].new_zeros(())
        context_gradient_rms = free_states[0].new_zeros(())
        outcome_rms = free_states[0].new_zeros(())
        predictor_error_rms = free_states[0].new_zeros(())
        prediction_rms = free_states[0].new_zeros(())
        prediction_target_cosine = free_states[0].new_zeros(())
        base_local_credit_rms = free_states[0].new_zeros(())
        gate_write_rate = free_states[0].new_zeros(())
        gate_write_probability = free_states[0].new_zeros(())
        gate_score_gradient_rms = free_states[0].new_zeros(())
        gate_eligibility_rms = free_states[0].new_zeros(())
        gate_context_change_rms = free_states[0].new_zeros(())
        node_residual_level_diagnostics: list[dict[str, float]] = []

        if self.config.hierarchical_error_binding_mode == "factorized":
            for layer in range(self.config.num_layers):
                (
                    block_gradients,
                    layer_attention_entropy,
                    layer_route_tv,
                    layer_route_credit_rms,
                    _,
                ) = self._phase_block_gradients(
                    layer,
                    free_states[layer],
                    settled_states[layer + 1],
                    route_lower_state=free_states[layer],
                    causal_mask=causal_mask,
                    token_count=token_count,
                )
                for name, block_gradient in block_gradients.items():
                    self._add_gradient(gradients, name, block_gradient)
                attention_entropy = attention_entropy + layer_attention_entropy
                route_tv = route_tv + layer_route_tv
                route_credit_rms = (
                    route_credit_rms + layer_route_credit_rms
                )

        for layer in (
            range(0)
            if self.config.hierarchical_error_binding_mode == "factorized"
            else range(base_depth)
        ):
            base_negative_gradient, _ = self._decoder_negative_state_gradient(
                free_states[layer + 1],
                y,
            )
            base_target = free_states[layer + 1] + base_negative_gradient
            (
                block_gradients,
                layer_attention_entropy,
                layer_route_tv,
                layer_route_credit_rms,
                _,
            ) = self._phase_block_gradients(
                layer,
                free_states[layer],
                base_target,
                route_lower_state=free_states[layer],
                causal_mask=causal_mask,
                token_count=token_count,
            )
            for name, block_gradient in block_gradients.items():
                self._add_gradient(gradients, name, block_gradient)
            attention_entropy = attention_entropy + layer_attention_entropy
            route_tv = route_tv + layer_route_tv
            route_credit_rms = route_credit_rms + layer_route_credit_rms
            base_local_credit_rms = (
                base_local_credit_rms + _rms(base_negative_gradient)
            )

        if self.config.sps_node_residual_hierarchy:
            if node_residual_cache is None:
                raise RuntimeError("node residual hierarchy cache is missing")
            remaining = self._prediction_states(bottom_up_residual)
            for level in range(self.config.num_layers - base_depth):
                prediction = node_residual_cache.predictions[level]
                predictor_error = prediction - remaining
                target_energy = remaining.float().square().sum().item()
                prediction_energy = prediction.float().square().sum().item()
                error_energy = predictor_error.float().square().sum().item()
                prediction_dot = (
                    prediction.float() * remaining.float()
                ).sum().item()
                node_residual_level_diagnostics.append(
                    {
                        "target_rms": float(_rms(remaining).item()),
                        "prediction_rms": float(_rms(prediction).item()),
                        "predictor_error_rms": float(
                            _rms(predictor_error).item()
                        ),
                        "prediction_target_cosine": float(
                            F.cosine_similarity(
                                prediction.float().reshape(
                                    -1,
                                    self.config.hidden_dim,
                                ),
                                remaining.float().reshape(
                                    -1,
                                    self.config.hidden_dim,
                                ),
                                dim=-1,
                                eps=1e-8,
                            )
                            .mean()
                            .item()
                        ),
                        "residual_energy_explained_fraction": (
                            1.0 - error_energy / max(target_energy, 1e-12)
                        ),
                        "least_squares_prediction_scale": (
                            prediction_dot / max(prediction_energy, 1e-12)
                        ),
                    }
                )
                candidate_gradient, gate_score_gradient = (
                    self._node_residual_memory_vjp(
                        node_residual_cache,
                        level,
                        predictor_error,
                    )
                )
                if self.config.hierarchical_error_context_mode != "instant":
                    self._add_gradient(
                        gradients,
                        f"hierarchical_gate_weights.{level}",
                        (
                            gate_score_gradient
                            * node_residual_cache.gate_features[level]
                        ).sum(dim=(0, 1))
                        / token_count,
                    )
                    self._add_gradient(
                        gradients,
                        f"hierarchical_gate_biases.{level}",
                        gate_score_gradient.sum() / token_count,
                    )
                measured_gate = (
                    node_residual_cache.write_gates[level][:, 1:]
                    if node_residual_cache.write_gates[level].shape[1] > 1
                    else node_residual_cache.write_gates[level]
                )
                measured_probability = (
                    node_residual_cache.write_probabilities[level][:, 1:]
                    if node_residual_cache.write_probabilities[level].shape[1]
                    > 1
                    else node_residual_cache.write_probabilities[level]
                )
                context_gradient_rms = (
                    context_gradient_rms + _rms(candidate_gradient)
                )
                outcome_rms = outcome_rms + _rms(remaining)
                predictor_error_rms = (
                    predictor_error_rms + _rms(predictor_error)
                )
                prediction_rms = prediction_rms + _rms(prediction)
                prediction_target_cosine = (
                    prediction_target_cosine
                    + F.cosine_similarity(
                        prediction.float().reshape(
                            -1,
                            self.config.hidden_dim,
                        ),
                        remaining.float().reshape(
                            -1,
                            self.config.hidden_dim,
                        ),
                        dim=-1,
                        eps=1e-8,
                    ).mean()
                )
                gate_write_rate = gate_write_rate + measured_gate.mean()
                gate_write_probability = (
                    gate_write_probability + measured_probability.mean()
                )
                gate_score_gradient_rms = (
                    gate_score_gradient_rms + _rms(gate_score_gradient)
                )
                gate_eligibility_rms = (
                    gate_eligibility_rms
                    + _rms(node_residual_cache.gate_features[level])
                )
                gate_context_change_rms = (
                    gate_context_change_rms
                    + _rms(
                        node_residual_cache.memories[level]
                        - node_residual_cache.previous_memories[level]
                    )
                )
                remaining = remaining - prediction
            bottom_up_residual = self._expand_prediction_signal(
                remaining,
                bottom_up_residual,
            )

        recursive_cp_final_residual: Tensor | None = None
        recursive_cp_consolidation = {
            "deployed_score": 0.0,
            "fast_score": 0.0,
            "candidate_score": 0.0,
            "accepted": 0.0,
            "attempts": 0.0,
            "accepts": 0.0,
            "acceptance_rate": 0.0,
        }
        recursive_cp_level_diagnostics: list[dict[str, float]] = []
        if self.recursive_cp_her is not None:
            raw_memories, normalized_memories, _ = (
                self._hierarchical_error_recursive_cp_cache(
                    free_states,
                    contexts=(
                        persistent_context_cache.contexts
                        if persistent_context_cache is not None
                        else None
                    ),
                )
            )
            flattened_memories = [
                memory.reshape(-1, self.config.hidden_dim)
                for memory in normalized_memories
            ]
            flattened_target = bottom_up_residual.reshape(
                -1,
                self.config.hidden_dim,
            )
            deployed_recursive_result = self.recursive_cp_her.local_gradients(
                flattened_memories,
                flattened_target,
            )
            learning_recursive_result = deployed_recursive_result
            recursive_parameter_prefix = "recursive_cp_her"
            if self.recursive_cp_her_fast is not None:
                fast_recursive_result = self.recursive_cp_her_fast.local_gradients(
                    flattened_memories,
                    flattened_target,
                )
                recursive_cp_consolidation = (
                    self._consolidate_recursive_cp_if_better(
                        flattened_memories,
                        flattened_target,
                        deployed_recursive_result,
                        fast_recursive_result,
                    )
                )
                learning_recursive_result = fast_recursive_result
                recursive_parameter_prefix = "recursive_cp_her_fast"
            for name, gradient in learning_recursive_result.gradients.items():
                if gradient.numel():
                    self._add_gradient(
                        gradients,
                        f"{recursive_parameter_prefix}.{name}",
                        gradient,
                    )
            contributions = [
                contribution.reshape_as(bottom_up_residual)
                for contribution in self.recursive_cp_her.base_contributions(
                    deployed_recursive_result.cache
                )
            ]
            recursive_target = bottom_up_residual
            recursive_target_rms = _rms(recursive_target)
            recursive_vector_target = bottom_up_residual
            for offset, (raw_memory, memory_gradient, contribution) in enumerate(
                zip(
                    raw_memories,
                    deployed_recursive_result.memory_gradients,
                    contributions,
                )
            ):
                vector_error = contribution - recursive_vector_target
                vector_target_energy = (
                    recursive_vector_target.float().square().sum().item()
                )
                vector_error_energy = vector_error.float().square().sum().item()
                vector_prediction_energy = (
                    contribution.float().square().sum().item()
                )
                vector_dot = (
                    contribution.float()
                    * recursive_vector_target.float()
                ).sum().item()
                tensor_outcome_energy = (
                    self.recursive_cp_her.squared_norm(
                        deployed_recursive_result.outcomes[offset]
                    )
                    .mean()
                    .item()
                )
                tensor_prediction_energy = (
                    self.recursive_cp_her.squared_norm(
                        deployed_recursive_result.cache.modulated[offset]
                    )
                    .mean()
                    .item()
                )
                tensor_error_energy = (
                    self.recursive_cp_her.squared_norm(
                        deployed_recursive_result.errors[offset]
                    )
                    .mean()
                    .item()
                )
                tensor_denominator = (
                    tensor_outcome_energy * tensor_prediction_energy
                ) ** 0.5
                recursive_cp_level_diagnostics.append(
                    {
                        "target_rms": float(_rms(recursive_vector_target).item()),
                        "prediction_rms": float(_rms(contribution).item()),
                        "predictor_error_rms": float(_rms(vector_error).item()),
                        "prediction_target_cosine": float(
                            F.cosine_similarity(
                                contribution.float().reshape(
                                    -1,
                                    self.config.hidden_dim,
                                ),
                                recursive_vector_target.float().reshape(
                                    -1,
                                    self.config.hidden_dim,
                                ),
                                dim=-1,
                                eps=1e-8,
                            )
                            .mean()
                            .item()
                        ),
                        "residual_energy_explained_fraction": (
                            1.0
                            - vector_error_energy
                            / max(vector_target_energy, 1e-12)
                        ),
                        "least_squares_prediction_scale": (
                            vector_dot / max(vector_prediction_energy, 1e-12)
                        ),
                        "tensor_outcome_energy": tensor_outcome_energy,
                        "tensor_prediction_energy": tensor_prediction_energy,
                        "tensor_error_energy": tensor_error_energy,
                        "tensor_residual_energy_explained_fraction": (
                            1.0
                            - tensor_error_energy
                            / max(tensor_outcome_energy, 1e-12)
                        ),
                        "tensor_prediction_target_cosine": (
                            0.0
                            if tensor_denominator == 0.0
                            else (
                                tensor_prediction_energy
                                + tensor_outcome_energy
                                - tensor_error_energy
                            )
                            / (2.0 * tensor_denominator)
                        ),
                    }
                )
                recursive_vector_target = recursive_vector_target - contribution
                context_gradient = _rms_norm_vjp(
                    raw_memory,
                    memory_gradient.reshape_as(raw_memory),
                )
                context_gradient.mul_(
                    self.config.hierarchical_error_context_credit_scale
                )
                context_gradient_rms = context_gradient_rms + _rms(
                    context_gradient
                )
                outcome_rms = outcome_rms + recursive_target_rms
                prediction_rms = prediction_rms + _rms(contribution)
                predictor_error_rms = predictor_error_rms + _rms(
                    contribution - recursive_target
                )
                prediction_target_cosine = (
                    prediction_target_cosine
                    + F.cosine_similarity(
                        contribution.float().reshape(
                            -1,
                            self.config.hidden_dim,
                        ),
                        recursive_target.float().reshape(
                            -1,
                            self.config.hidden_dim,
                        ),
                        dim=-1,
                        eps=1e-8,
                    ).mean()
                )
                if persistent_context_cache is not None:
                    candidate = persistent_context_cache.candidates[offset]
                    previous_context = (
                        persistent_context_cache.previous_contexts[offset]
                    )
                    write_gate = persistent_context_cache.write_gates[offset]
                    write_probability = (
                        persistent_context_cache.write_probabilities[offset]
                    )
                    memory_scale = self.config.hierarchical_error_memory_scale
                    gate_signal = memory_scale * (
                        context_gradient * (candidate - previous_context)
                    ).sum(dim=-1, keepdim=True)
                    gate_score_gradient = (
                        gate_signal
                        * write_probability
                        * (1.0 - write_probability)
                    )
                    eligibility_feature = (
                        persistent_context_cache.eligibility_features[offset]
                    )
                    eligibility_bias = (
                        persistent_context_cache.eligibility_biases[offset]
                    )
                    self._add_gradient(
                        gradients,
                        f"hierarchical_gate_weights.{offset}",
                        (
                            gate_score_gradient * eligibility_feature
                        ).sum(dim=(0, 1))
                        / token_count,
                    )
                    self._add_gradient(
                        gradients,
                        f"hierarchical_gate_biases.{offset}",
                        (gate_score_gradient * eligibility_bias).sum()
                        / token_count,
                    )
                    measured_gate = (
                        write_gate[:, 1:]
                        if write_gate.shape[1] > 1
                        else write_gate
                    )
                    measured_probability = (
                        write_probability[:, 1:]
                        if write_probability.shape[1] > 1
                        else write_probability
                    )
                    gate_write_rate = gate_write_rate + measured_gate.mean()
                    gate_write_probability = (
                        gate_write_probability + measured_probability.mean()
                    )
                    gate_score_gradient_rms = (
                        gate_score_gradient_rms + _rms(gate_score_gradient)
                    )
                    gate_eligibility_rms = (
                        gate_eligibility_rms + _rms(eligibility_feature)
                    )
                    gate_context_change_rms = (
                        gate_context_change_rms
                        + _rms(
                            persistent_context_cache.memories[offset]
                            - previous_context
                        )
                    )
            recursive_prediction = sum(
                contributions,
                torch.zeros_like(bottom_up_residual),
            )
            recursive_cp_final_residual = (
                bottom_up_residual - recursive_prediction
            )

        error_layers = (
            range(0)
            if self.config.sps_node_residual_hierarchy
            or self.recursive_cp_her is not None
            else range(base_depth, self.config.num_layers)
        )
        factorized_outcomes: list[Tensor] | None = None
        factorized_final_residual: Tensor | None = None
        if (
            self.config.hierarchical_error_binding_mode == "factorized"
            and self.recursive_cp_her is None
        ):
            (
                factorized_outcomes,
                factorized_final_residual,
            ) = self._hierarchical_error_residual_handoff(
                bottom_up_residual,
                error_predictions,
            )
        for offset, layer in enumerate(error_layers):
            source_state = (
                self._prediction_only(free_states[layer + 1])
                if self.config.hierarchical_error_binding_mode == "factorized"
                else (
                    persistent_context_cache.contexts[offset]
                    if persistent_context_cache is not None
                    else free_states[layer + 1]
                )
            )
            normalized_source = _rms_norm(source_state)
            flat_source = normalized_source.reshape(-1, self.config.hidden_dim)
            factorized_context_gradient: Tensor | None = None
            if self.config.hierarchical_error_binding_mode == "factorized":
                if factorized_outcomes is None:
                    raise RuntimeError("factorized residual handoff is missing")
                outcome = factorized_outcomes[offset]
                (
                    predictor_error,
                    value_signal,
                    key_signal,
                    _,
                    factorized_context_gradient,
                ) = self._hierarchical_error_factorized_signals(
                    context_state=(
                        persistent_context_cache.contexts[offset]
                        if persistent_context_cache is not None
                        else self._state_context(free_states[layer])
                    ),
                    prediction=error_predictions[offset],
                    key=factorized_keys[offset],
                    value=factorized_values[offset],
                    gate=factorized_gates[offset],
                    target=outcome,
                    extra_prediction_gradient=(
                        extra_prediction_gradient
                    ),
                )
                factorized_context_gradient = self._state_context_vjp(
                    factorized_context_gradient
                )
                self._add_gradient(
                    gradients,
                    f"generative_weights.{layer}",
                    (
                        value_signal.reshape(-1, self.config.hidden_dim).t()
                        @ flat_source
                    )
                    / token_count,
                )
                normalized_source_gradient = F.linear(
                    value_signal,
                    self.generative_weights[layer].t(),
                )
                if self.config.hierarchical_error_prediction_mode == "gated":
                    self._add_gradient(
                        gradients,
                        f"feedback_weights.{layer}",
                        (
                            key_signal.reshape(-1, self.config.hidden_dim).t()
                            @ flat_source
                        )
                        / token_count,
                    )
                    normalized_source_gradient = (
                        normalized_source_gradient
                        + F.linear(
                            key_signal,
                            self.feedback_weights[layer].t(),
                        )
                    )
            else:
                binding_context = (
                    free_states[base_depth]
                    if persistent_context_cache is not None and offset == 0
                    else (
                        persistent_context_cache.contexts[offset - 1]
                        if persistent_context_cache is not None
                        else free_states[layer]
                    )
                )
                outcome = self._hierarchical_error_bind(
                    binding_context,
                    bottom_up_residual,
                )
                predictor_error = (
                    modulated_error_predictions[offset] - outcome
                )
                flat_error = predictor_error.reshape(
                    -1,
                    self.config.hidden_dim,
                )
                self._add_gradient(
                    gradients,
                    f"generative_weights.{layer}",
                    (flat_error.t() @ flat_source) / token_count,
                )
                normalized_source_gradient = F.linear(
                    predictor_error,
                    self.generative_weights[layer].t(),
                )
            context_gradient = _rms_norm_vjp(
                source_state,
                normalized_source_gradient,
            )
            context_gradient.mul_(
                self.config.hierarchical_error_context_credit_scale
            )
            if (
                self.config.hierarchical_error_binding_mode == "factorized"
                and persistent_context_cache is not None
                and factorized_context_gradient is not None
            ):
                gate_context_gradient = factorized_context_gradient * (
                    self.config.hierarchical_error_context_credit_scale
                )
                candidate = persistent_context_cache.candidates[offset]
                previous_context = (
                    persistent_context_cache.previous_contexts[offset]
                )
                write_gate = persistent_context_cache.write_gates[offset]
                write_probability = (
                    persistent_context_cache.write_probabilities[offset]
                )
                memory_scale = self.config.hierarchical_error_memory_scale
                gate_signal = memory_scale * (
                    gate_context_gradient * (candidate - previous_context)
                ).sum(dim=-1, keepdim=True)
                gate_score_gradient = (
                    gate_signal
                    * write_probability
                    * (1.0 - write_probability)
                )
                eligibility_feature = (
                    persistent_context_cache.eligibility_features[offset]
                )
                eligibility_bias = (
                    persistent_context_cache.eligibility_biases[offset]
                )
                self._add_gradient(
                    gradients,
                    f"hierarchical_gate_weights.{offset}",
                    (
                        gate_score_gradient * eligibility_feature
                    ).sum(dim=(0, 1))
                    / token_count,
                )
                self._add_gradient(
                    gradients,
                    f"hierarchical_gate_biases.{offset}",
                    (gate_score_gradient * eligibility_bias).sum()
                    / token_count,
                )
                measured_gate = (
                    write_gate[:, 1:]
                    if write_gate.shape[1] > 1
                    else write_gate
                )
                measured_probability = (
                    write_probability[:, 1:]
                    if write_probability.shape[1] > 1
                    else write_probability
                )
                gate_write_rate = gate_write_rate + measured_gate.mean()
                gate_write_probability = (
                    gate_write_probability + measured_probability.mean()
                )
                gate_score_gradient_rms = (
                    gate_score_gradient_rms + _rms(gate_score_gradient)
                )
                gate_eligibility_rms = (
                    gate_eligibility_rms + _rms(eligibility_feature)
                )
                gate_context_change_rms = (
                    gate_context_change_rms
                    + _rms(
                        persistent_context_cache.memories[offset]
                        - previous_context
                    )
                )
            if self.config.hierarchical_error_binding_mode != "factorized":
                if persistent_context_cache is not None:
                    candidate = persistent_context_cache.candidates[offset]
                    previous_context = (
                        persistent_context_cache.previous_contexts[offset]
                    )
                    write_gate = persistent_context_cache.write_gates[offset]
                    write_probability = (
                        persistent_context_cache.write_probabilities[offset]
                    )
                    memory_scale = self.config.hierarchical_error_memory_scale
                    gate_signal = memory_scale * (
                        context_gradient * (candidate - previous_context)
                    ).sum(dim=-1, keepdim=True)
                    gate_score_gradient = (
                        gate_signal
                        * write_probability
                        * (1.0 - write_probability)
                    )
                    eligibility_feature = (
                        persistent_context_cache.eligibility_features[offset]
                    )
                    eligibility_bias = (
                        persistent_context_cache.eligibility_biases[offset]
                    )
                    self._add_gradient(
                        gradients,
                        f"hierarchical_gate_weights.{offset}",
                        (
                            gate_score_gradient * eligibility_feature
                        ).sum(dim=(0, 1))
                        / token_count,
                    )
                    self._add_gradient(
                        gradients,
                        f"hierarchical_gate_biases.{offset}",
                        (
                            gate_score_gradient * eligibility_bias
                        ).sum()
                        / token_count,
                    )
                    candidate_credit = (
                        (1.0 - memory_scale) + memory_scale * write_gate
                    )
                    candidate_gradient = candidate_credit * context_gradient
                    context_target = candidate - candidate_gradient
                    measured_gate = (
                        write_gate[:, 1:]
                        if write_gate.shape[1] > 1
                        else write_gate
                    )
                    measured_probability = (
                        write_probability[:, 1:]
                        if write_probability.shape[1] > 1
                        else write_probability
                    )
                    gate_write_rate = gate_write_rate + measured_gate.mean()
                    gate_write_probability = (
                        gate_write_probability + measured_probability.mean()
                    )
                    gate_score_gradient_rms = (
                        gate_score_gradient_rms + _rms(gate_score_gradient)
                    )
                    gate_eligibility_rms = (
                        gate_eligibility_rms + _rms(eligibility_feature)
                    )
                    gate_context_change_rms = (
                        gate_context_change_rms
                        + _rms(
                            persistent_context_cache.memories[offset]
                            - previous_context
                        )
                    )
                else:
                    context_target = free_states[layer + 1] - context_gradient
                (
                    block_gradients,
                    layer_attention_entropy,
                    layer_route_tv,
                    layer_route_credit_rms,
                    _,
                ) = self._phase_block_gradients(
                    layer,
                    free_states[layer],
                    context_target,
                    route_lower_state=free_states[layer],
                    causal_mask=causal_mask,
                    token_count=token_count,
                )
                for name, block_gradient in block_gradients.items():
                    self._add_gradient(gradients, name, block_gradient)

                attention_entropy = attention_entropy + layer_attention_entropy
                route_tv = route_tv + layer_route_tv
                route_credit_rms = route_credit_rms + layer_route_credit_rms
            context_gradient_rms = context_gradient_rms + _rms(context_gradient)
            outcome_rms = outcome_rms + _rms(outcome)
            predictor_error_rms = predictor_error_rms + _rms(predictor_error)
            prediction_rms = prediction_rms + _rms(error_predictions[offset])
            prediction_target_cosine = prediction_target_cosine + (
                F.cosine_similarity(
                    error_predictions[offset].float().reshape(
                        -1,
                        self.config.hidden_dim,
                    ),
                    outcome.float().reshape(-1, self.config.hidden_dim),
                    dim=-1,
                    eps=1e-8,
                ).mean()
            )

            bottom_up_residual = outcome - error_predictions[offset]

        if factorized_final_residual is not None:
            bottom_up_residual = factorized_final_residual
        if recursive_cp_final_residual is not None:
            bottom_up_residual = recursive_cp_final_residual

        initial_state_gradient = -deployed_negative_gradient
        if self.config.state_prediction_separation:
            first_cache = self._block(0, free_states[0])
            first_prediction_error = (
                first_cache.prediction - settled_states[1]
            )
            initial_state_gradient = self._block_input_signal(
                0,
                free_states[0],
                first_cache,
                first_prediction_error,
            )
        input_gradient = self._embedding_state_vjp(
            x,
            free_states[0],
            initial_state_gradient,
        )
        embedding_gradient = torch.zeros_like(self.embedding.weight)
        embedding_gradient.index_add_(
            0,
            x.reshape(-1),
            input_gradient.reshape(-1, self.config.hidden_dim),
        )
        embedding_gradient.div_(token_count)
        self._add_gradient(gradients, "embedding.weight", embedding_gradient)
        if self.config.state_prediction_separation:
            prediction_token_gradient = self._prediction_token_state_vjp(
                x,
                free_states[0],
                initial_state_gradient,
            ) / token_count
            self._add_gradient(
                gradients,
                "prediction_token",
                prediction_token_gradient,
            )

        optimizer_lr_multiplier = self._optimizer_lr_multiplier(current_step)
        effective_muon_momentum = self._muon_momentum(current_step)
        self.local_adam_step.add_(1)
        parameters = dict(self.named_parameters())
        update_rms_by_group = {
            "decoder": 0.0,
            "context": 0.0,
            "embedding": 0.0,
        }
        update_count_by_group = {
            "decoder": 0,
            "context": 0,
            "embedding": 0,
        }
        contextual_update_rms = {
            "route": 0.0,
            "generative": 0.0,
            "ffn": 0.0,
            "gate": 0.0,
        }
        contextual_update_count = {
            "route": 0,
            "generative": 0,
            "ffn": 0,
            "gate": 0,
        }
        for name, gradient in gradients.items():
            component = ""
            if name in {"decoder_weight", "decoder_bias"}:
                lr = decoder_lr * optimizer_lr_multiplier
                group = "decoder"
            elif name in {"embedding.weight", "prediction_token"}:
                lr = embed_lr * optimizer_lr_multiplier
                group = "embedding"
            else:
                lr = state_lr * optimizer_lr_multiplier
                group = "context"
                if name.startswith(("query_weights.", "key_weights.")):
                    lr *= self.config.route_lr_scale
                    component = "route"
                elif name.startswith(
                    ("generative_weights.", "feedback_weights.")
                ):
                    layer = int(name.rsplit(".", 1)[1])
                    if not self._hierarchical_error_level_is_active(
                        layer,
                        current_step,
                    ):
                        continue
                    if predictor_state_lr is not None:
                        lr = predictor_state_lr * optimizer_lr_multiplier
                    lr *= self.config.generative_lr_scale
                    component = "generative"
                elif name.startswith(
                    ("recursive_cp_her.", "recursive_cp_her_fast.")
                ):
                    offset = int(name.rsplit(".", 1)[1])
                    layer = base_depth + offset
                    if not self._hierarchical_error_level_is_active(
                        layer,
                        current_step,
                    ):
                        continue
                    if predictor_state_lr is not None:
                        lr = predictor_state_lr * optimizer_lr_multiplier
                    lr *= self.config.generative_lr_scale
                    if offset > 0:
                        lr *= (
                            self.config.hierarchical_error_recursive_cp_upper_lr_scale
                        )
                    component = "generative"
                elif name.startswith(("ffn_up_weights.", "ffn_down_weights.")):
                    lr *= self.config.ffn_lr_scale
                    component = "ffn"
                elif name.startswith("hierarchical_gate_"):
                    lr *= self.config.hierarchical_error_gate_lr_scale
                    component = "gate"
            if lr == 0:
                continue
            if name in self._muon_buffer_names:
                delta_rms = self._muon_update(
                    name,
                    parameters[name],
                    gradient,
                    lr=lr * self.config.muon_lr_scale,
                    weight_decay=weight_decay,
                    momentum=effective_muon_momentum,
                )
            else:
                delta_rms = self._adam_update(
                    name,
                    parameters[name],
                    gradient,
                    lr=lr,
                    weight_decay=weight_decay,
                )
            update_rms_by_group[group] += delta_rms
            update_count_by_group[group] += 1
            if group == "context" and component:
                contextual_update_rms[component] += delta_rms
                contextual_update_count[component] += 1

        recursive_cp_scale_stats = self.enforce_recursive_cp_component_bound(
            include_deployed=self.recursive_cp_her_fast is None,
        )
        layer_count = float(self.config.num_layers)
        error_layer_count = float(self.config.num_layers - base_depth)
        diagnostics = {
            "update_diag_message_pc_feedback_rms": 0.0,
            "update_diag_message_pc_prediction_error_rms": float(
                (predictor_error_rms / error_layer_count).item()
            ),
            "update_diag_message_pc_phase_delta_rms": 0.0,
            "update_diag_message_pc_attention_entropy": float(
                (attention_entropy / layer_count).item()
            ),
            "update_diag_message_pc_route_tv": float(
                (route_tv / layer_count).item()
            ),
            "update_diag_message_pc_route_credit_rms": float(
                (route_credit_rms / layer_count).item()
            ),
            "update_diag_message_pc_uses_coherent_energy": 0.0,
            "update_diag_message_pc_uses_her_error_hierarchy": 1.0,
            "update_diag_message_pc_uses_state_prediction_separation": float(
                self.config.state_prediction_separation
            ),
            "update_diag_message_pc_prediction_stream_window": float(
                self.config.prediction_stream_window
            ),
            "update_diag_message_pc_sps_prediction_input_scale": (
                self.config.sps_prediction_input_scale
            ),
            "update_diag_message_pc_sps_layer_token_credit_scale": (
                self.config.sps_layer_token_credit_scale
            ),
            "update_diag_message_pc_sps_future_state_credit_scale": (
                self.config.sps_future_state_credit_scale
            ),
            "update_diag_message_pc_sps_future_observation_credit_scale": (
                self.config.sps_future_observation_credit_scale
            ),
            "update_diag_message_pc_uses_sps_node_residual_hierarchy": float(
                self.config.sps_node_residual_hierarchy
            ),
            "update_diag_message_pc_uses_her_outer_product_contract": float(
                self.config.hierarchical_error_outer_product_contract
            ),
            "update_diag_message_pc_uses_her_recursive_cp": float(
                self.recursive_cp_her is not None
            ),
            "update_diag_message_pc_her_recursive_cp_rank": float(
                self.config.hierarchical_error_recursive_cp_rank
            ),
            "update_diag_message_pc_her_recursive_cp_upper_lr_scale": (
                self.config.hierarchical_error_recursive_cp_upper_lr_scale
            ),
            "update_diag_message_pc_her_recursive_cp_component_norm_cap": (
                self.config.hierarchical_error_recursive_cp_component_norm_cap
            ),
            "update_diag_message_pc_her_recursive_cp_uses_delayed_consolidation": float(
                self.recursive_cp_her_fast is not None
            ),
            "update_diag_message_pc_her_recursive_cp_consolidation_rate": (
                self.config.hierarchical_error_recursive_cp_consolidation_rate
            ),
            "update_diag_message_pc_her_recursive_cp_min_improvement": (
                self.config.hierarchical_error_recursive_cp_min_improvement
            ),
            "update_diag_message_pc_her_recursive_cp_deployed_score": (
                recursive_cp_consolidation["deployed_score"]
            ),
            "update_diag_message_pc_her_recursive_cp_fast_score": (
                recursive_cp_consolidation["fast_score"]
            ),
            "update_diag_message_pc_her_recursive_cp_candidate_score": (
                recursive_cp_consolidation["candidate_score"]
            ),
            "update_diag_message_pc_her_recursive_cp_consolidation_accepted": (
                recursive_cp_consolidation["accepted"]
            ),
            "update_diag_message_pc_her_recursive_cp_consolidation_attempts": (
                recursive_cp_consolidation["attempts"]
            ),
            "update_diag_message_pc_her_recursive_cp_consolidation_accepts": (
                recursive_cp_consolidation["accepts"]
            ),
            "update_diag_message_pc_her_recursive_cp_consolidation_acceptance_rate": (
                recursive_cp_consolidation["acceptance_rate"]
            ),
            "update_diag_message_pc_her_recursive_cp_clipped_fraction": (
                0.0
                if recursive_cp_scale_stats is None
                else recursive_cp_scale_stats.clipped_count
                / max(recursive_cp_scale_stats.component_count, 1)
            ),
            "update_diag_message_pc_her_recursive_cp_max_norm_before_clip": (
                0.0
                if recursive_cp_scale_stats is None
                else recursive_cp_scale_stats.maximum_norm_before_clip
            ),
            "update_diag_message_pc_her_recursive_cp_max_norm_after_clip": (
                0.0
                if recursive_cp_scale_stats is None
                else recursive_cp_scale_stats.maximum_norm_after_clip
            ),
            "update_diag_message_pc_her_top_down_scale": (
                self.config.hierarchical_error_top_down_scale
            ),
            "update_diag_message_pc_her_outcome_code_scale": (
                self.config.hierarchical_error_outcome_code_scale
            ),
            "update_diag_message_pc_her_outcome_readout_scale": (
                self.config.hierarchical_error_outcome_readout_scale
            ),
            "update_diag_message_pc_her_outcome_readout_gradient_rms": float(
                outcome_readout_gradient_rms.item()
            ),
            "update_diag_message_pc_her_base_depth": float(base_depth),
            "update_diag_message_pc_her_error_levels": error_layer_count,
            "update_diag_message_pc_her_active_error_levels": float(
                sum(
                    self._hierarchical_error_level_is_active(layer, current_step)
                    for layer in range(base_depth, self.config.num_layers)
                )
            ),
            "update_diag_message_pc_her_uses_direct_prediction": float(
                self.config.hierarchical_error_prediction_mode == "direct"
            ),
            "update_diag_message_pc_her_learning_start_step": float(
                self.config.hierarchical_error_learning_start_step
            ),
            "update_diag_message_pc_her_level_start_interval": float(
                self.config.hierarchical_error_level_start_interval
            ),
            "update_diag_message_pc_her_context_credit_scale": (
                self.config.hierarchical_error_context_credit_scale
            ),
            "update_diag_message_pc_her_token_credit_scale": (
                self.config.hierarchical_error_token_credit_scale
            ),
            "update_diag_message_pc_her_token_credit_gradient_rms": float(
                token_credit_gradient_rms.item()
            ),
            "update_diag_message_pc_her_target_active_fraction": float(
                target_active_fraction.item()
            ),
            "update_diag_message_pc_her_uses_free_to_settled": float(
                self.config.hierarchical_error_binding_mode == "factorized"
            ),
            "update_diag_message_pc_her_base_error_rms": float(
                base_error_rms.item()
            ),
            "update_diag_message_pc_her_outcome_rms": float(
                (outcome_rms / error_layer_count).item()
            ),
            "update_diag_message_pc_her_prediction_rms": float(
                (prediction_rms / error_layer_count).item()
            ),
            "update_diag_message_pc_her_prediction_target_cosine": float(
                (prediction_target_cosine / error_layer_count).item()
            ),
            "update_diag_message_pc_her_predictor_error_rms": float(
                (predictor_error_rms / error_layer_count).item()
            ),
            "update_diag_message_pc_her_base_local_credit_rms": float(
                (
                    base_local_credit_rms / max(float(base_depth), 1.0)
                ).item()
            ),
            "update_diag_message_pc_her_final_residual_rms": float(
                _rms(bottom_up_residual).item()
            ),
            "update_diag_message_pc_her_top_down_correction_rms": float(
                _rms(top_down_correction).item()
            ),
            "update_diag_message_pc_her_top_down_target_cosine": float(
                F.cosine_similarity(
                    top_down_correction.float().reshape(
                        -1,
                        self.config.hidden_dim,
                    ),
                    base_error.float().reshape(-1, self.config.hidden_dim),
                    dim=-1,
                    eps=1e-8,
                ).mean().item()
            ),
            "update_diag_message_pc_her_context_gradient_rms": float(
                (context_gradient_rms / error_layer_count).item()
            ),
            "update_diag_message_pc_her_uses_persistent_context": float(
                persistent_context_cache is not None
                or (
                    self.config.sps_node_residual_hierarchy
                    and self.config.hierarchical_error_context_mode
                    != "instant"
                )
            ),
            "update_diag_message_pc_her_memory_scale": (
                self.config.hierarchical_error_memory_scale
            ),
            "update_diag_message_pc_her_gate_write_rate": float(
                (gate_write_rate / error_layer_count).item()
            ),
            "update_diag_message_pc_her_gate_write_probability": float(
                (gate_write_probability / error_layer_count).item()
            ),
            "update_diag_message_pc_her_gate_score_gradient_rms": float(
                (gate_score_gradient_rms / error_layer_count).item()
            ),
            "update_diag_message_pc_her_gate_eligibility_rms": float(
                (gate_eligibility_rms / error_layer_count).item()
            ),
            "update_diag_message_pc_her_gate_context_change_rms": float(
                (gate_context_change_rms / error_layer_count).item()
            ),
            "update_diag_message_pc_local_prediction_rms": float(
                (predictor_error_rms / error_layer_count).item()
            ),
            "update_diag_message_pc_decoder_update_rms": (
                update_rms_by_group["decoder"]
                / max(update_count_by_group["decoder"], 1)
            ),
            "update_diag_message_pc_context_update_rms": (
                update_rms_by_group["context"]
                / max(update_count_by_group["context"], 1)
            ),
            "update_diag_message_pc_route_update_rms": (
                contextual_update_rms["route"]
                / max(contextual_update_count["route"], 1)
            ),
            "update_diag_message_pc_feedback_update_rms": 0.0,
            "update_diag_message_pc_generative_update_rms": (
                contextual_update_rms["generative"]
                / max(contextual_update_count["generative"], 1)
            ),
            "update_diag_message_pc_ffn_update_rms": (
                contextual_update_rms["ffn"]
                / max(contextual_update_count["ffn"], 1)
            ),
            "update_diag_message_pc_her_gate_update_rms": (
                contextual_update_rms["gate"]
                / max(contextual_update_count["gate"], 1)
            ),
            "update_diag_message_pc_embedding_update_rms": (
                update_rms_by_group["embedding"]
                / max(update_count_by_group["embedding"], 1)
            ),
            "update_diag_message_pc_winner_margin_active_fraction": float(
                winner_margin_active.item()
            ),
            "update_diag_message_pc_feedback_decoder_cosine": float(
                F.cosine_similarity(
                    self.feedback_embedding.float().reshape(1, -1),
                    self.decoder_weight.float().reshape(1, -1),
                ).item()
            ),
            "update_diag_message_pc_query_rms": float(
                torch.stack([_rms(weight) for weight in self.query_weights])
                .mean()
                .item()
            ),
            "update_diag_message_pc_key_rms": float(
                torch.stack([_rms(weight) for weight in self.key_weights])
                .mean()
                .item()
            ),
            "message_pc_learned_params": float(self.learned_parameter_count()),
            "message_pc_context_params": float(self.contextual_parameter_count()),
            "message_pc_persistent_slots": float(self.persistent_slot_count()),
            "message_pc_active_learned_params": float(
                self.active_parameter_count()
            ),
            "update_diag_message_pc_uses_local_muon": float(
                self.config.matrix_optimizer == "muon"
            ),
            "update_diag_message_pc_optimizer_lr_multiplier": (
                optimizer_lr_multiplier
            ),
            "update_diag_message_pc_effective_muon_momentum": (
                effective_muon_momentum
            ),
        }
        if self.recursive_cp_her is not None:
            for offset, (inputs, contexts, values) in enumerate(
                zip(
                    self.recursive_cp_her.input_factors,
                    self.recursive_cp_her.context_factors,
                    self.recursive_cp_her.value_factors,
                )
            ):
                prefix = (
                    "update_diag_message_pc_her_recursive_cp_level_"
                    f"{base_depth + offset}"
                )
                diagnostics[f"{prefix}_input_factor_rms"] = float(
                    _rms(inputs).item()
                )
                diagnostics[f"{prefix}_context_factor_rms"] = (
                    float(_rms(contexts).item()) if contexts.numel() else 0.0
                )
                diagnostics[f"{prefix}_value_factor_rms"] = float(
                    _rms(values).item()
                )
                diagnostics[f"{prefix}_value_norm_max"] = float(
                    values.norm(dim=-1).max().item()
                )
                if offset < len(recursive_cp_level_diagnostics):
                    for name, value in recursive_cp_level_diagnostics[
                        offset
                    ].items():
                        diagnostics[f"{prefix}_{name}"] = value
        if self.recursive_cp_her_fast is not None:
            for offset, values in enumerate(
                self.recursive_cp_her_fast.value_factors
            ):
                prefix = (
                    "update_diag_message_pc_her_recursive_cp_fast_level_"
                    f"{base_depth + offset}"
                )
                diagnostics[f"{prefix}_value_factor_rms"] = float(
                    _rms(values).item()
                )
                diagnostics[f"{prefix}_value_norm_max"] = float(
                    values.norm(dim=-1).max().item()
                )
        if self.config.sps_node_residual_hierarchy:
            for offset, values in enumerate(node_residual_level_diagnostics):
                prefix = (
                    "update_diag_message_pc_sps_node_level_"
                    f"{base_depth + offset}"
                )
                for name, value in values.items():
                    diagnostics[f"{prefix}_{name}"] = value
        diagnostics.update(settle_diagnostics)
        self._last_update_diagnostics = diagnostics

        sequence_loss = token_loss.mean(dim=1).detach()
        final_state = deployed_state[:, -1].detach()
        if return_sequence_loss and return_token_loss and return_final_state:
            return loss, sequence_loss, token_loss.detach(), final_state
        if return_sequence_loss and return_token_loss:
            return loss, sequence_loss, token_loss.detach()
        if return_sequence_loss and return_final_state:
            return loss, sequence_loss, final_state
        if return_sequence_loss:
            return loss, sequence_loss
        if return_final_state:
            return loss, final_state
        return loss

    def _recurrent_local_update(
        self,
        x: Tensor,
        y: Tensor,
        *,
        decoder_lr: float,
        state_lr: float,
        embed_lr: float,
        weight_decay: float,
        current_step: int,
        return_sequence_loss: bool,
        return_token_loss: bool,
        return_final_state: bool,
    ) -> float | tuple[float, Tensor] | tuple[float, Tensor, Tensor] | tuple[float, Tensor, Tensor, Tensor]:
        states, caches = self._forward_states(x, with_cache=True)
        shared_recurrence = self.config.inference_mode == "recurrent_local"
        step_count = (
            self.config.recurrent_shared_steps
            if shared_recurrence
            else self.config.num_layers
        )
        if len(caches) != step_count or len(states) != step_count + 1:
            raise RuntimeError("local outcome-state cache is incomplete")

        gradients: dict[str, Tensor] = {}
        token_count = float(max(x.numel(), 1))
        step_weight = 1.0 / float(step_count)
        effective_target_step = self._inference_lr(current_step)
        causal_mask = self._causal_mask(x.shape[1], x.device)
        step_token_losses: list[Tensor] = []
        attention_entropy = states[0].new_zeros(())
        route_tv = states[0].new_zeros(())
        route_credit_rms = states[0].new_zeros(())
        local_prediction_rms = states[0].new_zeros(())
        margin_active = states[0].new_zeros(())
        embedding_input_gradient: Tensor | None = None

        for recurrent_step, (state, cache) in enumerate(
            zip(states[1:], caches)
        ):
            block_layer = 0 if shared_recurrence else recurrent_step
            logits = self._decoder_logits(state)
            token_loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                y.reshape(-1),
                reduction="none",
            ).view_as(y)
            step_token_losses.append(token_loss)
            probabilities = F.softmax(logits, dim=-1)
            decoder_features = self._decoder_features(state)
            decoder_weight_gradient, decoder_bias_gradient = (
                self._cross_entropy_gradient(
                    decoder_features,
                    y,
                    probabilities,
                )
            )
            (
                margin_weight_gradient,
                margin_bias_gradient,
                step_margin_active,
            ) = self._winner_margin_decoder_gradients(
                decoder_features,
                y,
                logits,
            )
            decoder_weight_gradient.add_(margin_weight_gradient)
            decoder_bias_gradient.add_(margin_bias_gradient)
            decoder_step_weight = (
                step_weight
                if shared_recurrence
                else float(recurrent_step == step_count - 1)
            )
            if decoder_step_weight > 0:
                self._add_gradient(
                    gradients,
                    "decoder_weight",
                    decoder_step_weight * decoder_weight_gradient,
                )
                self._add_gradient(
                    gradients,
                    "decoder_bias",
                    decoder_step_weight * decoder_bias_gradient,
                )
                margin_active.add_(
                    step_margin_active,
                    alpha=decoder_step_weight,
                )

            descent_direction, _ = self._decoder_negative_state_gradient(
                state,
                y,
            )
            target_delta = _cap_token_rms(
                effective_target_step * descent_direction,
                self.config.state_delta_rms_cap,
            )
            target_state = state + target_delta
            (
                phase_gradients,
                step_attention_entropy,
                step_route_tv,
                step_route_credit_rms,
                step_prediction_rms,
            ) = self._phase_block_gradients(
                block_layer,
                states[recurrent_step],
                target_state,
                route_lower_state=states[recurrent_step],
                causal_mask=causal_mask,
                token_count=token_count,
                branch_scale=(
                    1.0
                    if not shared_recurrence or recurrent_step == 0
                    else self.config.recurrent_step_scale
                ),
            )
            phase_step_weight = step_weight if shared_recurrence else 1.0
            for name, gradient in phase_gradients.items():
                self._add_gradient(
                    gradients,
                    name,
                    phase_step_weight * gradient,
                )
            attention_entropy.add_(step_attention_entropy, alpha=step_weight)
            route_tv.add_(step_route_tv, alpha=step_weight)
            route_credit_rms.add_(step_route_credit_rms, alpha=step_weight)
            local_prediction_rms.add_(step_prediction_rms, alpha=step_weight)

            if recurrent_step == 0:
                embedding_input_gradient = (
                    phase_step_weight
                    * self._block_input_signal(
                        block_layer,
                        states[0],
                        cache,
                        -target_delta,
                    )
                )

        if embedding_input_gradient is None:
            raise RuntimeError("local outcome embedding signal is missing")
        embedding_state_gradient = self._embedding_state_vjp(
            x,
            states[0],
            embedding_input_gradient,
        )
        embedding_gradient = torch.zeros_like(self.embedding.weight)
        embedding_gradient.index_add_(
            0,
            x.reshape(-1),
            embedding_state_gradient.reshape(-1, self.config.hidden_dim),
        )
        embedding_gradient.div_(token_count)
        self._add_gradient(
            gradients,
            "embedding.weight",
            embedding_gradient,
        )

        optimizer_lr_multiplier = self._optimizer_lr_multiplier(current_step)
        effective_muon_momentum = self._muon_momentum(current_step)
        self.local_adam_step.add_(1)
        parameters = dict(self.named_parameters())
        update_rms_by_group = {
            "decoder": 0.0,
            "context": 0.0,
            "embedding": 0.0,
        }
        update_count_by_group = {
            "decoder": 0,
            "context": 0,
            "embedding": 0,
        }
        contextual_update_rms = {
            "route": 0.0,
            "ffn": 0.0,
        }
        contextual_update_count = {
            "route": 0,
            "ffn": 0,
        }
        for name, gradient in gradients.items():
            component = ""
            if name in {"decoder_weight", "decoder_bias"}:
                lr = decoder_lr * optimizer_lr_multiplier
                group = "decoder"
            elif name == "embedding.weight":
                lr = embed_lr * optimizer_lr_multiplier
                group = "embedding"
            elif name.startswith("token_value_tables."):
                lr = (
                    embed_lr
                    * self.config.token_value_lr_scale
                    * optimizer_lr_multiplier
                )
                group = "embedding"
            elif (
                name == "feedback_embedding"
                and self.config.inference_mode == "meta_pc"
                and self.config.meta_pc_output_feedback_mode == "kp_learned"
            ):
                lr = (
                    decoder_lr
                    * self.config.meta_pc_output_feedback_lr_scale
                    * optimizer_lr_multiplier
                )
                group = "decoder"
            else:
                lr = state_lr * optimizer_lr_multiplier
                group = "context"
                if name.startswith(("query_weights.", "key_weights.")):
                    lr *= self.config.route_lr_scale
                    component = "route"
                elif name.startswith(("ffn_up_weights.", "ffn_down_weights.")):
                    lr *= self.config.ffn_lr_scale
                    component = "ffn"
            if name in self._muon_buffer_names:
                delta_rms = self._muon_update(
                    name,
                    parameters[name],
                    gradient,
                    lr=lr * self.config.muon_lr_scale,
                    weight_decay=weight_decay,
                    momentum=effective_muon_momentum,
                )
            else:
                delta_rms = self._adam_update(
                    name,
                    parameters[name],
                    gradient,
                    lr=lr,
                    weight_decay=weight_decay,
                )
            update_rms_by_group[group] += delta_rms
            update_count_by_group[group] += 1
            if group == "context" and component:
                contextual_update_rms[component] += delta_rms
                contextual_update_count[component] += 1

        first_loss = step_token_losses[0].mean()
        final_token_loss = step_token_losses[-1]
        final_loss = final_token_loss.mean()
        self._last_update_diagnostics = {
            "update_diag_message_pc_uses_recurrent_local": float(
                shared_recurrence
            ),
            "update_diag_message_pc_uses_layerwise_local": float(
                not shared_recurrence
            ),
            "update_diag_message_pc_recurrent_shared_steps": float(
                step_count
            ),
            "update_diag_message_pc_recurrent_step_scale": float(
                self.config.recurrent_step_scale
            ),
            "update_diag_message_pc_effective_inference_lr": (
                effective_target_step
            ),
            "update_diag_message_pc_recurrent_first_loss": float(
                first_loss.item()
            ),
            "update_diag_message_pc_recurrent_final_loss": float(
                final_loss.item()
            ),
            "update_diag_message_pc_recurrent_loss_improvement": float(
                (first_loss - final_loss).item()
            ),
            "update_diag_message_pc_recurrent_state_change_rms": float(
                _rms(states[-1] - states[0]).item()
            ),
            "update_diag_message_pc_attention_entropy": float(
                attention_entropy.item()
            ),
            "update_diag_message_pc_route_tv": float(route_tv.item()),
            "update_diag_message_pc_route_credit_rms": float(
                route_credit_rms.item()
            ),
            "update_diag_message_pc_local_prediction_rms": float(
                local_prediction_rms.item()
            ),
            "update_diag_message_pc_decoder_update_rms": (
                update_rms_by_group["decoder"]
                / max(update_count_by_group["decoder"], 1)
            ),
            "update_diag_message_pc_context_update_rms": (
                update_rms_by_group["context"]
                / max(update_count_by_group["context"], 1)
            ),
            "update_diag_message_pc_route_update_rms": (
                contextual_update_rms["route"]
                / max(contextual_update_count["route"], 1)
            ),
            "update_diag_message_pc_ffn_update_rms": (
                contextual_update_rms["ffn"]
                / max(contextual_update_count["ffn"], 1)
            ),
            "update_diag_message_pc_embedding_update_rms": (
                update_rms_by_group["embedding"]
                / max(update_count_by_group["embedding"], 1)
            ),
            "update_diag_message_pc_winner_margin_active_fraction": float(
                margin_active.item()
            ),
            "message_pc_learned_params": float(self.learned_parameter_count()),
            "message_pc_context_params": float(self.contextual_parameter_count()),
            "message_pc_persistent_slots": float(self.persistent_slot_count()),
            "message_pc_active_learned_params": float(
                self.active_parameter_count()
            ),
            "update_diag_message_pc_uses_local_muon": float(
                self.config.matrix_optimizer == "muon"
            ),
            "update_diag_message_pc_optimizer_lr_multiplier": (
                optimizer_lr_multiplier
            ),
            "update_diag_message_pc_effective_muon_momentum": (
                effective_muon_momentum
            ),
        }

        loss = float(final_loss.item())
        sequence_loss = final_token_loss.mean(dim=1).detach()
        final_state = states[-1][:, -1].detach()
        if return_sequence_loss and return_token_loss and return_final_state:
            return loss, sequence_loss, final_token_loss.detach(), final_state
        if return_sequence_loss and return_token_loss:
            return loss, sequence_loss, final_token_loss.detach()
        if return_sequence_loss and return_final_state:
            return loss, sequence_loss, final_state
        if return_sequence_loss:
            return loss, sequence_loss
        if return_final_state:
            return loss, final_state
        return loss

    def _difference_target_local_update(
        self,
        x: Tensor,
        y: Tensor,
        *,
        decoder_lr: float,
        state_lr: float,
        embed_lr: float,
        weight_decay: float,
        current_step: int,
        return_sequence_loss: bool,
        return_token_loss: bool,
        return_final_state: bool,
    ) -> float | tuple[float, Tensor] | tuple[float, Tensor, Tensor] | tuple[float, Tensor, Tensor, Tensor]:
        free_states, _ = self._forward_states(x, with_cache=False)
        deployment_logits = self._decoder_logits(free_states[-1])
        token_loss = F.cross_entropy(
            deployment_logits.reshape(-1, self.config.vocab_size),
            y.reshape(-1),
            reduction="none",
        ).view_as(y)
        loss = float(token_loss.mean().item())
        effective_target_step = self._inference_lr(current_step)
        target_states, target_deltas = self._difference_target_states(
            free_states,
            y,
            target_step=effective_target_step,
        )
        gradients: dict[str, Tensor] = {}
        token_count = float(max(x.numel(), 1))

        decoder_features = self._decoder_features(free_states[-1])
        probabilities = F.softmax(deployment_logits, dim=-1)
        decoder_weight_gradient, decoder_bias_gradient = (
            self._cross_entropy_gradient(
                decoder_features,
                y,
                probabilities,
            )
        )
        margin_weight_gradient, margin_bias_gradient, margin_active = (
            self._winner_margin_decoder_gradients(
                decoder_features,
                y,
                deployment_logits,
            )
        )
        decoder_weight_gradient.add_(margin_weight_gradient)
        decoder_bias_gradient.add_(margin_bias_gradient)
        self._add_gradient(
            gradients,
            "decoder_weight",
            decoder_weight_gradient,
        )
        self._add_gradient(
            gradients,
            "decoder_bias",
            decoder_bias_gradient,
        )

        attention_entropy = free_states[0].new_zeros(())
        route_tv = free_states[0].new_zeros(())
        route_credit_rms = free_states[0].new_zeros(())
        prediction_error_rms = free_states[0].new_zeros(())
        inverse_error_rms = free_states[0].new_zeros(())
        inverse_prediction_rms = free_states[0].new_zeros(())
        inverse_alignment = free_states[0].new_zeros(())
        inverse_damping_response_rms = free_states[0].new_zeros(())
        causal_mask = self._causal_mask(x.shape[1], x.device)
        for layer in range(self.config.num_layers):
            (
                phase_gradients,
                phase_attention_entropy,
                phase_route_tv,
                phase_route_credit_rms,
                phase_prediction_error_rms,
            ) = self._phase_block_gradients(
                layer,
                free_states[layer],
                target_states[layer + 1],
                route_lower_state=free_states[layer],
                causal_mask=causal_mask,
                token_count=token_count,
            )
            for name, gradient in phase_gradients.items():
                self._add_gradient(gradients, name, gradient)
            attention_entropy.add_(phase_attention_entropy)
            route_tv.add_(phase_route_tv)
            route_credit_rms.add_(phase_route_credit_rms)
            prediction_error_rms.add_(phase_prediction_error_rms)

            (
                inverse_gradients,
                layer_inverse_error,
                layer_inverse_prediction,
                layer_inverse_alignment,
                layer_inverse_damping_response,
            ) = self._difference_inverse_gradients(
                layer,
                free_states[layer],
                (
                    free_states[-1]
                    if self.config.difference_target_path_mode == "direct"
                    else free_states[layer + 1]
                ),
                token_count=token_count,
            )
            for name, gradient in inverse_gradients.items():
                self._add_gradient(gradients, name, gradient)
            inverse_error_rms.add_(layer_inverse_error)
            inverse_prediction_rms.add_(layer_inverse_prediction)
            inverse_alignment.add_(layer_inverse_alignment)
            inverse_damping_response_rms.add_(
                layer_inverse_damping_response
            )

        embedding_input_gradient = self._embedding_state_vjp(
            x,
            free_states[0],
            free_states[0] - target_states[0],
        )
        embedding_gradient = torch.zeros_like(self.embedding.weight)
        embedding_gradient.index_add_(
            0,
            x.reshape(-1),
            embedding_input_gradient.reshape(-1, self.config.hidden_dim),
        )
        embedding_gradient.div_(token_count)
        self._add_gradient(
            gradients,
            "embedding.weight",
            embedding_gradient,
        )

        optimizer_lr_multiplier = self._optimizer_lr_multiplier(current_step)
        effective_muon_momentum = self._muon_momentum(current_step)
        self.local_adam_step.add_(1)
        parameters = dict(self.named_parameters())
        update_rms_by_group = {
            "decoder": 0.0,
            "context": 0.0,
            "embedding": 0.0,
        }
        update_count_by_group = {
            "decoder": 0,
            "context": 0,
            "embedding": 0,
        }
        contextual_update_rms = {
            "route": 0.0,
            "feedback": 0.0,
            "generative": 0.0,
            "ffn": 0.0,
            "scale": 0.0,
        }
        contextual_update_count = {
            "route": 0,
            "feedback": 0,
            "generative": 0,
            "ffn": 0,
            "scale": 0,
        }
        for name, gradient in gradients.items():
            component = ""
            if name in {"decoder_weight", "decoder_bias"}:
                lr = decoder_lr * optimizer_lr_multiplier
                group = "decoder"
            elif name == "embedding.weight":
                lr = embed_lr * optimizer_lr_multiplier
                group = "embedding"
            else:
                lr = state_lr * optimizer_lr_multiplier
                group = "context"
                if name.startswith(("query_weights.", "key_weights.")):
                    lr *= self.config.route_lr_scale
                    component = "route"
                elif name.startswith("feedback_weights."):
                    lr *= self.config.feedback_lr_scale
                    component = "feedback"
                elif name.startswith("generative_weights."):
                    lr *= self.config.generative_lr_scale
                    component = "generative"
                elif name.startswith(("ffn_up_weights.", "ffn_down_weights.")):
                    lr *= self.config.ffn_lr_scale
                    component = "ffn"
            if name in self._muon_buffer_names:
                delta_rms = self._muon_update(
                    name,
                    parameters[name],
                    gradient,
                    lr=lr * self.config.muon_lr_scale,
                    weight_decay=weight_decay,
                    momentum=effective_muon_momentum,
                )
            else:
                delta_rms = self._adam_update(
                    name,
                    parameters[name],
                    gradient,
                    lr=lr,
                    weight_decay=weight_decay,
                )
            update_rms_by_group[group] += delta_rms
            update_count_by_group[group] += 1
            if group == "context" and component:
                contextual_update_rms[component] += delta_rms
                contextual_update_count[component] += 1

        layer_count = float(self.config.num_layers)
        target_delta_rms = torch.stack(
            [_rms(delta) for delta in target_deltas]
        )
        self._last_update_diagnostics = {
            "update_diag_message_pc_uses_difference_target": 1.0,
            "update_diag_message_pc_difference_target_uses_direct_path": float(
                self.config.difference_target_path_mode == "direct"
            ),
            "update_diag_message_pc_effective_inference_lr": (
                effective_target_step
            ),
            "update_diag_message_pc_difference_target_top_rms": float(
                target_delta_rms[-1].item()
            ),
            "update_diag_message_pc_difference_target_bottom_rms": float(
                target_delta_rms[0].item()
            ),
            "update_diag_message_pc_difference_target_mean_rms": float(
                target_delta_rms.mean().item()
            ),
            "update_diag_message_pc_inverse_noise_rms": (
                self.config.difference_target_inverse_noise
            ),
            "update_diag_message_pc_inverse_damping": (
                self.config.difference_target_inverse_damping
            ),
            "update_diag_message_pc_inverse_reconstruction_rms": float(
                (inverse_error_rms / layer_count).item()
            ),
            "update_diag_message_pc_inverse_prediction_rms": float(
                (inverse_prediction_rms / layer_count).item()
            ),
            "update_diag_message_pc_inverse_gain": float(
                (
                    inverse_prediction_rms
                    / layer_count
                    / self.config.difference_target_inverse_noise
                ).item()
            ),
            "update_diag_message_pc_inverse_alignment": float(
                (inverse_alignment / layer_count).item()
            ),
            "update_diag_message_pc_inverse_damping_response_rms": float(
                (inverse_damping_response_rms / layer_count).item()
            ),
            "update_diag_message_pc_attention_entropy": float(
                (attention_entropy / layer_count).item()
            ),
            "update_diag_message_pc_route_tv": float(
                (route_tv / layer_count).item()
            ),
            "update_diag_message_pc_route_credit_rms": float(
                (route_credit_rms / layer_count).item()
            ),
            "update_diag_message_pc_local_prediction_rms": float(
                (prediction_error_rms / layer_count).item()
            ),
            "update_diag_message_pc_decoder_update_rms": (
                update_rms_by_group["decoder"]
                / max(update_count_by_group["decoder"], 1)
            ),
            "update_diag_message_pc_context_update_rms": (
                update_rms_by_group["context"]
                / max(update_count_by_group["context"], 1)
            ),
            "update_diag_message_pc_route_update_rms": (
                contextual_update_rms["route"]
                / max(contextual_update_count["route"], 1)
            ),
            "update_diag_message_pc_feedback_update_rms": (
                contextual_update_rms["feedback"]
                / max(contextual_update_count["feedback"], 1)
            ),
            "update_diag_message_pc_generative_update_rms": (
                contextual_update_rms["generative"]
                / max(contextual_update_count["generative"], 1)
            ),
            "update_diag_message_pc_ffn_update_rms": (
                contextual_update_rms["ffn"]
                / max(contextual_update_count["ffn"], 1)
            ),
            "update_diag_message_pc_embedding_update_rms": (
                update_rms_by_group["embedding"]
                / max(update_count_by_group["embedding"], 1)
            ),
            "update_diag_message_pc_winner_margin_active_fraction": float(
                margin_active.item()
            ),
            "update_diag_message_pc_query_rms": float(
                torch.stack(
                    [_rms(weight) for weight in self.query_weights]
                ).mean().item()
            ),
            "update_diag_message_pc_key_rms": float(
                torch.stack(
                    [_rms(weight) for weight in self.key_weights]
                ).mean().item()
            ),
            "message_pc_learned_params": float(self.learned_parameter_count()),
            "message_pc_context_params": float(self.contextual_parameter_count()),
            "message_pc_persistent_slots": float(self.persistent_slot_count()),
            "message_pc_active_learned_params": float(
                self.active_parameter_count()
            ),
            "update_diag_message_pc_uses_local_muon": float(
                self.config.matrix_optimizer == "muon"
            ),
            "update_diag_message_pc_optimizer_lr_multiplier": (
                optimizer_lr_multiplier
            ),
            "update_diag_message_pc_effective_muon_momentum": (
                effective_muon_momentum
            ),
        }

        sequence_loss = token_loss.mean(dim=1).detach()
        final_state = free_states[-1][:, -1].detach()
        if return_sequence_loss and return_token_loss and return_final_state:
            return loss, sequence_loss, token_loss.detach(), final_state
        if return_sequence_loss and return_token_loss:
            return loss, sequence_loss, token_loss.detach()
        if return_sequence_loss and return_final_state:
            return loss, sequence_loss, final_state
        if return_sequence_loss:
            return loss, sequence_loss
        if return_final_state:
            return loss, final_state
        return loss

    @torch.no_grad()
    def _residual_error_pc_local_update(
        self,
        x: Tensor,
        y: Tensor,
        *,
        decoder_lr: float,
        state_lr: float,
        embed_lr: float,
        weight_decay: float,
        current_step: int,
        return_sequence_loss: bool,
        return_token_loss: bool,
        return_final_state: bool,
    ) -> float | tuple[float, Tensor] | tuple[float, Tensor, Tensor] | tuple[float, Tensor, Tensor, Tensor]:
        free_states, _ = self._forward_states(x, with_cache=False)
        persistent_context_cache: _HierarchicalContextCache | None = None
        context_offsets: list[Tensor] | None = None
        if self.config.hierarchical_error_context_mode != "instant":
            persistent_context_cache = self._hierarchical_error_context_cache(
                free_states
            )
            context_offsets = [
                context - free_states[layer + 1]
                for layer, context in enumerate(
                    persistent_context_cache.contexts
                )
            ]
        deployment_state = free_states[-1]
        deployment_logits = self._decoder_logits(deployment_state)
        token_loss = F.cross_entropy(
            deployment_logits.reshape(-1, self.config.vocab_size),
            y.reshape(-1),
            reduction="none",
        ).view_as(y)
        loss = float(token_loss.mean().item())
        decoder_token_delta, probabilities = self._meta_pc_top_feedback(
            deployment_state,
            y,
        )
        token_delta = self._residual_error_token_delta(
            y,
            probabilities,
            decoder_token_delta,
        )
        target_state_index = (
            1
            if self.config.residual_error_target_layer == "base"
            else self.config.num_layers
        )
        target_reference_state = free_states[target_state_index]
        target_state, proxy_memory = self._residual_error_target_state(
            target_reference_state,
            token_delta,
        )
        upper_proxy_delta = self._residual_error_upper_proxy_state_delta(
            token_delta
        )
        upper_content_delta = self._residual_error_upper_content_state_delta(x)
        upper_state_delta = upper_proxy_delta + upper_content_delta
        upper_proxy_target_state = (
            free_states[-1] + upper_state_delta
            if (
                self.config.residual_error_upper_proxy_state_scale > 0
                or self.config.residual_error_upper_content_state_scale > 0
            )
            else None
        )
        token_count = float(max(x.numel(), 1))
        optimizer_lr_multiplier = self._optimizer_lr_multiplier(current_step)
        effective_muon_momentum = self._muon_momentum(current_step)
        self.local_adam_step.add_(1)
        predictor_prewrite_update_rms = 0.0
        predictor_prewrite_update_count = 0
        if self.config.hierarchical_error_prediction_mode in {
            "direct",
            "direct_transport",
            "direct_phase_transport",
        }:
            predictor_prewrite_gradients = (
                self._residual_error_pc_predictor_parameter_gradients(
                    free_states,
                    target_state,
                    token_count=token_count,
                    context_offsets=context_offsets,
                    target_state_index=target_state_index,
                )
            )
            parameters = dict(self.named_parameters())
            predictor_lr = (
                state_lr
                * optimizer_lr_multiplier
                * self.config.generative_lr_scale
            )
            for name, gradient in predictor_prewrite_gradients.items():
                if name in self._muon_buffer_names:
                    update_rms = self._muon_update(
                        name,
                        parameters[name],
                        gradient,
                        lr=predictor_lr * self.config.muon_lr_scale,
                        weight_decay=weight_decay,
                        momentum=effective_muon_momentum,
                    )
                else:
                    update_rms = self._adam_update(
                        name,
                        parameters[name],
                        gradient,
                        lr=predictor_lr,
                        weight_decay=weight_decay,
                    )
                predictor_prewrite_update_rms += update_rms
                predictor_prewrite_update_count += 1
        effective_inference_lr = self._inference_lr(current_step)
        settled_states, diagnostics = self._settle_residual_error_pc(
            free_states,
            target_state,
            inference_lr=effective_inference_lr,
            context_offsets=context_offsets,
            target_state_index=target_state_index,
            upper_proxy_target_state=upper_proxy_target_state,
        )
        phase_refine_update_rms = 0.0
        phase_refine_update_count = 0
        phase_refine_passes = 0
        if (
            self.config.hierarchical_error_prediction_mode
            == "direct_phase_transport"
            and self.config.residual_phase_transport_refine_passes > 0
        ):
            parameters = dict(self.named_parameters())
            for _ in range(self.config.residual_phase_transport_refine_passes):
                phase_gradients, _ = (
                    self._residual_phase_transport_training_gradients(
                        free_states,
                        settled_states,
                    target_state,
                    accepted_lr=diagnostics[
                        "update_diag_message_pc_residual_error_accepted_lr"
                        ],
                        token_count=token_count,
                        context_offsets=context_offsets,
                        target_state_index=target_state_index,
                    )
                )
                for name, gradient in phase_gradients.items():
                    lr = state_lr * optimizer_lr_multiplier
                    if name.startswith(
                        ("inverse_query_weights.", "inverse_key_weights.")
                    ):
                        lr *= self.config.route_lr_scale
                    elif name.startswith("feedback_weights."):
                        lr *= self.config.feedback_lr_scale
                    if name in self._muon_buffer_names:
                        update_rms = self._muon_update(
                            name,
                            parameters[name],
                            gradient,
                            lr=lr * self.config.muon_lr_scale,
                            weight_decay=weight_decay,
                            momentum=effective_muon_momentum,
                        )
                    else:
                        update_rms = self._adam_update(
                            name,
                            parameters[name],
                            gradient,
                            lr=lr,
                            weight_decay=weight_decay,
                        )
                    phase_refine_update_rms += update_rms
                    phase_refine_update_count += 1
                phase_refine_passes += 1
                settled_states, diagnostics = self._settle_residual_error_pc(
                    free_states,
                    target_state,
                    inference_lr=effective_inference_lr,
                    context_offsets=context_offsets,
                    target_state_index=target_state_index,
                    upper_proxy_target_state=upper_proxy_target_state,
                )
        diagnostics[
            "update_diag_message_pc_residual_error_token_target_delta_rms"
        ] = float(_rms(target_state - target_reference_state).item())
        diagnostics[
            "update_diag_message_pc_residual_error_proxy_memory_rms"
        ] = float(_rms(proxy_memory).item())
        diagnostics[
            "update_diag_message_pc_residual_error_proxy_memory_scale"
        ] = self.config.residual_error_proxy_memory_scale
        diagnostics[
            "update_diag_message_pc_residual_error_proxy_memory_decay"
        ] = self.config.residual_error_proxy_memory_decay
        diagnostics[
            "update_diag_message_pc_residual_error_upper_proxy_state_scale"
        ] = self.config.residual_error_upper_proxy_state_scale
        diagnostics[
            "update_diag_message_pc_residual_error_upper_proxy_state_decay"
        ] = self.config.residual_error_upper_proxy_state_decay
        diagnostics[
            "update_diag_message_pc_residual_error_upper_proxy_state_input_rms"
        ] = float(_rms(upper_proxy_delta).item())
        diagnostics[
            "update_diag_message_pc_residual_error_upper_content_state_scale"
        ] = self.config.residual_error_upper_content_state_scale
        diagnostics[
            "update_diag_message_pc_residual_error_upper_content_state_decay"
        ] = self.config.residual_error_upper_content_state_decay
        diagnostics[
            "update_diag_message_pc_residual_error_upper_content_write_period"
        ] = float(self.config.residual_error_upper_content_write_period)
        diagnostics[
            "update_diag_message_pc_residual_error_upper_content_write_offset"
        ] = float(self.config.residual_error_upper_content_write_offset)
        diagnostics[
            "update_diag_message_pc_residual_error_upper_content_read_period"
        ] = float(self.config.residual_error_upper_content_read_period)
        diagnostics[
            "update_diag_message_pc_residual_error_upper_content_read_offset"
        ] = float(self.config.residual_error_upper_content_read_offset)
        diagnostics[
            "update_diag_message_pc_residual_error_upper_content_read_delay"
        ] = float(self.config.residual_error_upper_content_read_delay)
        diagnostics[
            "update_diag_message_pc_residual_error_upper_content_bind_current_scale"
        ] = float(
            self.config.residual_error_upper_content_bind_current_scale
        )
        diagnostics[
            "update_diag_message_pc_residual_error_upper_content_bind_product_scale"
        ] = float(
            self.config.residual_error_upper_content_bind_product_scale
        )
        diagnostics[
            "update_diag_message_pc_residual_error_upper_content_bind_current_delay"
        ] = float(
            self.config.residual_error_upper_content_bind_current_delay
        )
        diagnostics[
            "update_diag_message_pc_residual_error_upper_content_state_input_rms"
        ] = float(_rms(upper_content_delta).item())
        diagnostics[
            "update_diag_message_pc_residual_error_upper_state_input_rms"
        ] = float(_rms(upper_state_delta).item())
        diagnostics[
            "update_diag_message_pc_residual_phase_transport_refine_passes"
        ] = float(phase_refine_passes)
        diagnostics[
            "update_diag_message_pc_residual_phase_transport_refine_update_rms"
        ] = (
            phase_refine_update_rms / max(phase_refine_update_count, 1)
        )
        diagnostics["update_diag_message_pc_effective_inference_lr"] = (
            effective_inference_lr
        )
        diagnostics["update_diag_message_pc_uses_residual_error_nodes"] = 1.0
        diagnostics[
            "update_diag_message_pc_residual_error_uses_fixed_outcome"
        ] = float(
            self.config.hierarchical_error_target_mode == "fixed_outcome"
        )
        diagnostics[
            "update_diag_message_pc_residual_error_uses_decoder_output_feedback"
        ] = float(
            self.config.meta_pc_output_feedback_mode == "decoder_transpose"
        )
        diagnostics[
            "update_diag_message_pc_residual_error_uses_kp_output_feedback"
        ] = float(self.config.meta_pc_output_feedback_mode == "kp_learned")
        diagnostics[
            "update_diag_message_pc_residual_error_uses_factorized_output_feedback"
        ] = float(self.config.meta_pc_output_feedback_mode == "kp_factorized")
        diagnostics[
            "update_diag_message_pc_residual_error_uses_learned_only_transport"
        ] = float(self.config.residual_error_transport_mode == "learned_only")
        diagnostics[
            "update_diag_message_pc_residual_error_uses_persistent_context"
        ] = float(self.config.hierarchical_error_context_mode != "instant")
        diagnostics[
            "update_diag_message_pc_residual_error_uses_hard_persistent_context"
        ] = float(
            self.config.hierarchical_error_context_mode == "persistent_hard"
        )
        diagnostics[
            "update_diag_message_pc_residual_error_context_memory_scale"
        ] = self.config.hierarchical_error_memory_scale

        gradients: dict[str, Tensor] = {}
        decoder_features = self._decoder_features(deployment_state)
        decoder_weight_gradient, decoder_bias_gradient = (
            self._cross_entropy_gradient(
                decoder_features,
                y,
                probabilities,
            )
        )
        (
            margin_weight_gradient,
            margin_bias_gradient,
            winner_margin_active,
        ) = self._winner_margin_decoder_gradients(
            decoder_features,
            y,
            deployment_logits,
        )
        decoder_weight_gradient.add_(margin_weight_gradient)
        decoder_bias_gradient.add_(margin_bias_gradient)
        self._add_gradient(
            gradients,
            "decoder_weight",
            decoder_weight_gradient,
        )
        self._add_gradient(
            gradients,
            "decoder_bias",
            decoder_bias_gradient,
        )
        if self.config.meta_pc_output_feedback_mode == "kp_learned":
            self._add_gradient(
                gradients,
                "feedback_embedding",
                decoder_weight_gradient.clone(),
            )
        elif self.config.meta_pc_output_feedback_mode == "kp_factorized":
            if (
                self.feedback_token_factors is None
                or self.feedback_state_factors is None
            ):
                raise RuntimeError("factorized output feedback is not initialized")
            self._add_gradient(
                gradients,
                "feedback_token_factors",
                decoder_weight_gradient @ self.feedback_state_factors.t(),
            )
            self._add_gradient(
                gradients,
                "feedback_state_factors",
                self.feedback_token_factors.t() @ decoder_weight_gradient,
            )

        causal_mask = self._causal_mask(x.shape[1], x.device)
        attention_entropy = free_states[0].new_zeros(())
        route_tv = free_states[0].new_zeros(())
        route_credit_rms = free_states[0].new_zeros(())
        recognition_error_rms = free_states[0].new_zeros(())
        first_recognition_input_gradient: Tensor | None = None
        for layer in range(self.config.num_layers):
            (
                phase_gradients,
                layer_attention_entropy,
                layer_route_tv,
                layer_route_credit_rms,
                layer_prediction_error_rms,
            ) = self._phase_block_gradients(
                layer,
                settled_states[layer],
                settled_states[layer + 1],
                route_lower_state=settled_states[layer],
                causal_mask=causal_mask,
                token_count=token_count,
            )
            for name, gradient in phase_gradients.items():
                self._add_gradient(gradients, name, gradient)
            attention_entropy.add_(layer_attention_entropy)
            route_tv.add_(layer_route_tv)
            route_credit_rms.add_(layer_route_credit_rms)
            recognition_error_rms.add_(layer_prediction_error_rms)

            if layer == 0:
                recognition_cache = self._block(
                    layer,
                    settled_states[layer],
                )
                recognition_error = (
                    recognition_cache.prediction
                    - settled_states[layer + 1]
                )
                first_recognition_input_gradient = self._block_input_signal(
                    layer,
                    settled_states[layer],
                    recognition_cache,
                    recognition_error,
                )

        transport_reconstruction_rms = free_states[0].new_zeros(())
        if (
            self.config.hierarchical_error_prediction_mode
            == "direct_transport"
        ):
            for layer in range(self.config.num_layers):
                transport_cache = self._residual_transport_block(
                    layer,
                    free_states[layer + 1],
                )
                transport_error = (
                    transport_cache.prediction - free_states[layer]
                )
                transport_reconstruction_rms.add_(_rms(transport_error))
                transport_gradients = (
                    self._residual_transport_parameter_gradients(
                        layer,
                        transport_cache,
                        transport_error,
                        token_count=token_count,
                    )
                )
                for name, gradient in transport_gradients.items():
                    self._add_gradient(gradients, name, gradient)

        free_predictor_caches, free_predictions, free_residuals = (
            self._residual_error_pc_terms(
                free_states,
                target_state,
                context_offsets=context_offsets,
                target_state_index=target_state_index,
            )
        )
        phase_transport_fit_rms = free_states[0].new_zeros(())
        phase_transport_target_rms = free_states[0].new_zeros(())
        phase_transport_cosine = free_states[0].new_zeros(())
        phase_transport_fit_rms_by_layer = [
            free_states[0].new_zeros(())
            for _ in range(self.config.num_layers)
        ]
        phase_transport_target_rms_by_layer = [
            free_states[0].new_zeros(())
            for _ in range(self.config.num_layers)
        ]
        phase_transport_cosine_by_layer = [
            free_states[0].new_zeros(())
            for _ in range(self.config.num_layers)
        ]
        if (
            self.config.hierarchical_error_prediction_mode
            == "direct_phase_transport"
        ):
            accepted_lr = diagnostics[
                "update_diag_message_pc_residual_error_accepted_lr"
            ]
            phase_gradients, phase_stats = (
                self._residual_phase_transport_training_gradients(
                    free_states,
                    settled_states,
                    target_state,
                    accepted_lr=accepted_lr,
                    token_count=token_count,
                    context_offsets=context_offsets,
                    target_state_index=target_state_index,
                )
            )
            phase_transport_fit_rms = phase_stats["fit_rms"]  # type: ignore[assignment]
            phase_transport_target_rms = phase_stats["target_rms"]  # type: ignore[assignment]
            phase_transport_cosine = phase_stats["cosine"]  # type: ignore[assignment]
            phase_transport_fit_rms_by_layer = phase_stats["fit_by_layer"]  # type: ignore[assignment]
            phase_transport_target_rms_by_layer = phase_stats["target_by_layer"]  # type: ignore[assignment]
            phase_transport_cosine_by_layer = phase_stats["cosine_by_layer"]  # type: ignore[assignment]
            for name, gradient in phase_gradients.items():
                self._add_gradient(gradients, name, gradient)
        if self.config.hierarchical_error_prediction_mode not in {
            "direct",
            "direct_transport",
            "direct_phase_transport",
        }:
            predictor_gradients = (
                self._residual_error_pc_predictor_parameter_gradients(
                    free_states,
                    target_state,
                    token_count=token_count,
                    context_offsets=context_offsets,
                    target_state_index=target_state_index,
                )
            )
            for name, gradient in predictor_gradients.items():
                self._add_gradient(gradients, name, gradient)

        gate_write_rate = free_states[0].new_zeros(())
        gate_write_probability = free_states[0].new_zeros(())
        gate_gradient_rms = free_states[0].new_zeros(())
        if persistent_context_cache is not None:
            output_signals = [
                -self.config.prediction_precision * signal
                for signal in self._residual_error_suffix_signals(
                    free_residuals
                )
            ]
            for layer, (cache, output_signal) in enumerate(
                zip(free_predictor_caches, output_signals)
            ):
                context = persistent_context_cache.contexts[layer]
                context_gradient = self._residual_error_input_signal(
                    layer,
                    context,
                    cache,
                    output_signal,
                )
                candidate = persistent_context_cache.candidates[layer]
                previous_context = (
                    persistent_context_cache.previous_contexts[layer]
                )
                write_gate = persistent_context_cache.write_gates[layer]
                write_probability = (
                    persistent_context_cache.write_probabilities[layer]
                )
                gate_signal = self.config.hierarchical_error_memory_scale * (
                    context_gradient * (candidate - previous_context)
                ).sum(dim=-1, keepdim=True)
                gate_score_gradient = gate_signal * write_probability * (
                    1.0 - write_probability
                )
                eligibility_feature = (
                    persistent_context_cache.eligibility_features[layer]
                )
                eligibility_bias = (
                    persistent_context_cache.eligibility_biases[layer]
                )
                self._add_gradient(
                    gradients,
                    f"hierarchical_gate_weights.{layer}",
                    (
                        gate_score_gradient * eligibility_feature
                    ).sum(dim=(0, 1))
                    / token_count,
                )
                self._add_gradient(
                    gradients,
                    f"hierarchical_gate_biases.{layer}",
                    (gate_score_gradient * eligibility_bias).sum()
                    / token_count,
                )
                measured_gate = (
                    write_gate[:, 1:]
                    if write_gate.shape[1] > 1
                    else write_gate
                )
                measured_probability = (
                    write_probability[:, 1:]
                    if write_probability.shape[1] > 1
                    else write_probability
                )
                gate_write_rate.add_(measured_gate.mean())
                gate_write_probability.add_(measured_probability.mean())
                gate_gradient_rms.add_(_rms(gate_score_gradient))
            diagnostics[
                "update_diag_message_pc_residual_error_gate_write_rate"
            ] = float((gate_write_rate / self.config.num_layers).item())
            diagnostics[
                "update_diag_message_pc_residual_error_gate_write_probability"
            ] = float(
                (gate_write_probability / self.config.num_layers).item()
            )
            diagnostics[
                "update_diag_message_pc_residual_error_gate_gradient_rms"
            ] = float((gate_gradient_rms / self.config.num_layers).item())
        for layer in range(self.config.num_layers):
            incoming = free_residuals[layer]
            residual = free_residuals[layer + 1]
            prediction = free_predictions[layer]
            incoming_energy = incoming.square().mean().clamp_min(1e-12)
            prediction_cosine = F.cosine_similarity(
                prediction.reshape(-1, self.config.hidden_dim),
                incoming.reshape(-1, self.config.hidden_dim),
                dim=-1,
            ).mean()
            diagnostics[
                f"update_diag_message_pc_residual_error_free_incoming_rms_layer_{layer + 1}"
            ] = float(_rms(incoming).item())
            diagnostics[
                f"update_diag_message_pc_residual_error_free_prediction_rms_layer_{layer + 1}"
            ] = float(_rms(prediction).item())
            diagnostics[
                f"update_diag_message_pc_residual_error_free_residual_rms_layer_{layer + 1}"
            ] = float(_rms(residual).item())
            diagnostics[
                f"update_diag_message_pc_residual_error_free_prediction_cosine_layer_{layer + 1}"
            ] = float(prediction_cosine.item())
            diagnostics[
                f"update_diag_message_pc_residual_error_free_explained_energy_layer_{layer + 1}"
            ] = float(
                (1.0 - residual.square().mean() / incoming_energy).item()
            )

        if first_recognition_input_gradient is None:
            raise RuntimeError("residual-error embedding signal is missing")
        embedding_input_gradient = self._embedding_state_vjp(
            x,
            settled_states[0],
            first_recognition_input_gradient,
        )
        embedding_gradient = torch.zeros_like(self.embedding.weight)
        embedding_gradient.index_add_(
            0,
            x.reshape(-1),
            embedding_input_gradient.reshape(-1, self.config.hidden_dim),
        )
        embedding_gradient.div_(token_count)
        self._add_gradient(
            gradients,
            "embedding.weight",
            embedding_gradient,
        )

        parameters = dict(self.named_parameters())
        update_rms_by_group = {
            "decoder": 0.0,
            "context": 0.0,
            "embedding": 0.0,
        }
        update_count_by_group = {
            "decoder": 0,
            "context": 0,
            "embedding": 0,
        }
        contextual_update_rms = {
            "route": 0.0,
            "feedback": 0.0,
            "generative": predictor_prewrite_update_rms,
            "ffn": 0.0,
            "gate": 0.0,
        }
        contextual_update_count = {
            "route": 0,
            "feedback": 0,
            "generative": predictor_prewrite_update_count,
            "ffn": 0,
            "gate": 0,
        }
        for name, gradient in gradients.items():
            component = ""
            if name in {"decoder_weight", "decoder_bias"}:
                lr = decoder_lr * optimizer_lr_multiplier
                group = "decoder"
            elif name == "embedding.weight":
                lr = embed_lr * optimizer_lr_multiplier
                group = "embedding"
            elif name in {
                "feedback_embedding",
                "feedback_token_factors",
                "feedback_state_factors",
            }:
                lr = (
                    decoder_lr
                    * self.config.meta_pc_output_feedback_lr_scale
                    * optimizer_lr_multiplier
                )
                group = "decoder"
            else:
                lr = state_lr * optimizer_lr_multiplier
                group = "context"
                if name.startswith(
                    (
                        "query_weights.",
                        "key_weights.",
                        "inverse_query_weights.",
                        "inverse_key_weights.",
                    )
                ):
                    lr *= self.config.route_lr_scale
                    component = "route"
                elif name.startswith("feedback_weights."):
                    lr *= self.config.feedback_lr_scale
                    component = "feedback"
                elif name.startswith("generative_weights."):
                    lr *= self.config.generative_lr_scale
                    component = "generative"
                elif name.startswith(("ffn_up_weights.", "ffn_down_weights.")):
                    lr *= self.config.ffn_lr_scale
                    component = "ffn"
                elif name.startswith("hierarchical_gate_"):
                    lr *= self.config.hierarchical_error_gate_lr_scale
                    component = "gate"
            if name in self._muon_buffer_names:
                delta_rms = self._muon_update(
                    name,
                    parameters[name],
                    gradient,
                    lr=lr * self.config.muon_lr_scale,
                    weight_decay=weight_decay,
                    momentum=effective_muon_momentum,
                )
            else:
                delta_rms = self._adam_update(
                    name,
                    parameters[name],
                    gradient,
                    lr=lr,
                    weight_decay=weight_decay,
                )
            update_rms_by_group[group] += delta_rms
            update_count_by_group[group] += 1
            if group == "context" and component:
                contextual_update_rms[component] += delta_rms
                contextual_update_count[component] += 1

        layer_count = float(self.config.num_layers)
        diagnostics.update(
            {
                "update_diag_message_pc_attention_entropy": float(
                    (attention_entropy / layer_count).item()
                ),
                "update_diag_message_pc_route_tv": float(
                    (route_tv / layer_count).item()
                ),
                "update_diag_message_pc_route_credit_rms": float(
                    (route_credit_rms / layer_count).item()
                ),
                "update_diag_message_pc_local_prediction_rms": float(
                    (recognition_error_rms / layer_count).item()
                ),
                "update_diag_message_pc_residual_transport_reconstruction_rms": float(
                    (transport_reconstruction_rms / layer_count).item()
                ),
                "update_diag_message_pc_residual_phase_transport_fit_rms": float(
                    (phase_transport_fit_rms / layer_count).item()
                ),
                "update_diag_message_pc_residual_phase_transport_target_rms": float(
                    (phase_transport_target_rms / layer_count).item()
                ),
                "update_diag_message_pc_residual_phase_transport_cosine": float(
                    (phase_transport_cosine / layer_count).item()
                ),
                "update_diag_message_pc_residual_phase_transport_lr_scale": (
                    self.config.residual_phase_transport_lr_scale
                ),
                "update_diag_message_pc_decoder_update_rms": (
                    update_rms_by_group["decoder"]
                    / max(update_count_by_group["decoder"], 1)
                ),
                "update_diag_message_pc_context_update_rms": (
                    update_rms_by_group["context"]
                    / max(update_count_by_group["context"], 1)
                ),
                "update_diag_message_pc_route_update_rms": (
                    contextual_update_rms["route"]
                    / max(contextual_update_count["route"], 1)
                ),
                "update_diag_message_pc_feedback_update_rms": (
                    contextual_update_rms["feedback"]
                    / max(contextual_update_count["feedback"], 1)
                ),
                "update_diag_message_pc_generative_update_rms": (
                    contextual_update_rms["generative"]
                    / max(contextual_update_count["generative"], 1)
                ),
                "update_diag_message_pc_ffn_update_rms": (
                    contextual_update_rms["ffn"]
                    / max(contextual_update_count["ffn"], 1)
                ),
                "update_diag_message_pc_residual_error_gate_update_rms": (
                    contextual_update_rms["gate"]
                    / max(contextual_update_count["gate"], 1)
                ),
                "update_diag_message_pc_embedding_update_rms": (
                    update_rms_by_group["embedding"]
                    / max(update_count_by_group["embedding"], 1)
                ),
                "update_diag_message_pc_winner_margin_active_fraction": float(
                    winner_margin_active.item()
                ),
                "message_pc_learned_params": float(
                    self.learned_parameter_count()
                ),
                "message_pc_context_params": float(
                    self.contextual_parameter_count()
                ),
                "message_pc_persistent_slots": float(
                    self.persistent_slot_count()
                ),
                "message_pc_active_learned_params": float(
                    self.active_parameter_count()
                ),
                "message_pc_deployment_params": float(
                    self.deployment_parameter_count()
                ),
                "message_pc_training_feedback_params": float(
                    self.training_feedback_parameter_count()
                ),
                "update_diag_message_pc_uses_local_muon": float(
                    self.config.matrix_optimizer == "muon"
                ),
                "update_diag_message_pc_optimizer_lr_multiplier": (
                    optimizer_lr_multiplier
                ),
                "update_diag_message_pc_effective_muon_momentum": (
                    effective_muon_momentum
                ),
            }
        )
        for layer in range(self.config.num_layers):
            diagnostics.update(
                {
                    f"update_diag_message_pc_residual_phase_transport_fit_rms_layer_{layer}": float(
                        phase_transport_fit_rms_by_layer[layer].item()
                    ),
                    f"update_diag_message_pc_residual_phase_transport_target_rms_layer_{layer}": float(
                        phase_transport_target_rms_by_layer[layer].item()
                    ),
                    f"update_diag_message_pc_residual_phase_transport_cosine_layer_{layer}": float(
                        phase_transport_cosine_by_layer[layer].item()
                    ),
                }
            )
        self._last_update_diagnostics = diagnostics

        sequence_loss = token_loss.mean(dim=1).detach()
        final_state = deployment_state[:, -1].detach()
        if return_sequence_loss and return_token_loss and return_final_state:
            return loss, sequence_loss, token_loss.detach(), final_state
        if return_sequence_loss and return_token_loss:
            return loss, sequence_loss, token_loss.detach()
        if return_sequence_loss and return_final_state:
            return loss, sequence_loss, final_state
        if return_sequence_loss:
            return loss, sequence_loss
        if return_final_state:
            return loss, final_state
        return loss

    @torch.no_grad()
    def _bidirectional_pc_local_update(
        self,
        x: Tensor,
        y: Tensor,
        *,
        decoder_lr: float,
        state_lr: float,
        embed_lr: float,
        weight_decay: float,
        current_step: int,
        return_sequence_loss: bool,
        return_token_loss: bool,
        return_final_state: bool,
    ) -> float | tuple[float, Tensor] | tuple[float, Tensor, Tensor] | tuple[float, Tensor, Tensor, Tensor]:
        feedforward_states, _ = self._forward_states(x, with_cache=False)
        effective_inference_lr = self._inference_lr(current_step)
        if self.config.bidirectional_free_settling:
            free_states, diagnostics = self._settle_bidirectional_pc(
                feedforward_states,
                None,
                inference_lr=effective_inference_lr,
                phase="free",
            )
        else:
            free_states = [state.clone() for state in feedforward_states]
            free_energy = self._bidirectional_pc_energy(free_states, None)
            prefix = "update_diag_message_pc_bidirectional_free"
            diagnostics = {
                f"{prefix}_energy_before": float(free_energy.item()),
                f"{prefix}_energy_after": float(free_energy.item()),
                f"{prefix}_energy_reduction": 0.0,
                f"{prefix}_prediction_error_rms": 0.0,
                f"{prefix}_accepted_lr": 0.0,
                f"{prefix}_backtracking_rejections": 0.0,
            }
            for state_index in range(1, len(free_states)):
                diagnostics[
                    f"{prefix}_delta_rms_layer_{state_index}"
                ] = 0.0
        deployment_state = (
            free_states[-1]
            if self.config.bidirectional_deployment_settling
            else feedforward_states[-1]
        )
        deployment_logits = self._decoder_logits(deployment_state)
        token_loss = F.cross_entropy(
            deployment_logits.reshape(-1, self.config.vocab_size),
            y.reshape(-1),
            reduction="none",
        ).view_as(y)
        loss = float(token_loss.mean().item())
        clamped_states, clamped_diagnostics = self._settle_bidirectional_pc(
            free_states,
            y,
            inference_lr=effective_inference_lr,
            phase="clamped",
        )
        diagnostics.update(clamped_diagnostics)
        diagnostics["update_diag_message_pc_effective_inference_lr"] = (
            effective_inference_lr
        )
        diagnostics["update_diag_message_pc_uses_bidirectional_energy"] = 1.0
        diagnostics[
            "update_diag_message_pc_bidirectional_uses_free_settling"
        ] = float(self.config.bidirectional_free_settling)
        diagnostics[
            "update_diag_message_pc_bidirectional_deployment_uses_free_settling"
        ] = float(self.config.bidirectional_deployment_settling)

        gradients: dict[str, Tensor] = {}
        token_count = float(max(x.numel(), 1))
        probabilities = F.softmax(deployment_logits, dim=-1)
        decoder_features = self._decoder_features(deployment_state)
        decoder_weight_gradient, decoder_bias_gradient = (
            self._cross_entropy_gradient(
                decoder_features,
                y,
                probabilities,
            )
        )
        (
            margin_weight_gradient,
            margin_bias_gradient,
            winner_margin_active,
        ) = self._winner_margin_decoder_gradients(
            decoder_features,
            y,
            deployment_logits,
        )
        decoder_weight_gradient.add_(margin_weight_gradient)
        decoder_bias_gradient.add_(margin_bias_gradient)
        self._add_gradient(
            gradients, "decoder_weight", decoder_weight_gradient
        )
        self._add_gradient(
            gradients, "decoder_bias", decoder_bias_gradient
        )

        attention_entropy = feedforward_states[0].new_zeros(())
        route_tv = feedforward_states[0].new_zeros(())
        route_credit_rms = feedforward_states[0].new_zeros(())
        recognition_error_rms = feedforward_states[0].new_zeros(())
        generative_error_rms = feedforward_states[0].new_zeros(())
        first_recognition_input_gradient: Tensor | None = None
        forward_update_layers = 0
        for layer in range(self.config.num_layers):
            uses_forward_update = (
                self.config.forward_update_top_layers > 0
                and layer
                >= self.config.num_layers
                - self.config.forward_update_top_layers
            )
            recognition_lower_state = (
                feedforward_states[layer]
                if uses_forward_update
                else clamped_states[layer]
            )
            if uses_forward_update:
                forward_update_layers += 1
            (
                phase_gradients,
                layer_attention_entropy,
                layer_route_tv,
                layer_route_credit_rms,
                layer_prediction_error_rms,
            ) = self._phase_block_gradients(
                layer,
                recognition_lower_state,
                clamped_states[layer + 1],
                route_lower_state=feedforward_states[layer],
                causal_mask=self._causal_mask(
                    x.shape[1],
                    x.device,
                    layer=layer,
                ),
                token_count=token_count,
            )
            for name, gradient in phase_gradients.items():
                self._add_gradient(gradients, name, gradient)
            attention_entropy.add_(layer_attention_entropy)
            route_tv.add_(layer_route_tv)
            route_credit_rms.add_(layer_route_credit_rms)
            recognition_error_rms.add_(layer_prediction_error_rms)

            recognition_cache = self._block(
                layer, recognition_lower_state
            )
            recognition_error = (
                recognition_cache.prediction - clamped_states[layer + 1]
            )
            if layer == 0:
                first_recognition_input_gradient = self._block_input_signal(
                    layer,
                    recognition_lower_state,
                    recognition_cache,
                    recognition_error,
                )

            clamped_inverse_cache = self._inverse_block(
                layer, clamped_states[layer + 1]
            )
            clamped_inverse_error = (
                clamped_inverse_cache.prediction - clamped_states[layer]
            )
            generative_error_rms.add_(_rms(clamped_inverse_error))
            clamped_inverse_gradients = (
                self._bidirectional_inverse_parameter_gradients(
                    layer,
                    clamped_inverse_cache,
                    clamped_inverse_error,
                    token_count=token_count,
                )
            )
            for name, clamped_gradient in clamped_inverse_gradients.items():
                self._add_gradient(
                    gradients,
                    name,
                    self.config.prediction_precision * clamped_gradient,
                )
        diagnostics[
            "update_diag_message_pc_bidirectional_forward_update_layers"
        ] = float(forward_update_layers)

        if first_recognition_input_gradient is None:
            raise RuntimeError("bidirectional embedding signal is missing")
        embedding_input_gradient = self._embedding_state_vjp(
            x,
            clamped_states[0],
            first_recognition_input_gradient,
        )
        embedding_gradient = torch.zeros_like(self.embedding.weight)
        embedding_gradient.index_add_(
            0,
            x.reshape(-1),
            embedding_input_gradient.reshape(-1, self.config.hidden_dim),
        )
        embedding_gradient.div_(token_count)
        self._add_gradient(
            gradients,
            "embedding.weight",
            embedding_gradient,
        )

        optimizer_lr_multiplier = self._optimizer_lr_multiplier(current_step)
        effective_muon_momentum = self._muon_momentum(current_step)
        self.local_adam_step.add_(1)
        parameters = dict(self.named_parameters())
        update_rms_by_group = {
            "decoder": 0.0,
            "context": 0.0,
            "embedding": 0.0,
        }
        update_count_by_group = {
            "decoder": 0,
            "context": 0,
            "embedding": 0,
        }
        contextual_update_rms = {
            "route": 0.0,
            "feedback": 0.0,
            "generative": 0.0,
            "ffn": 0.0,
            "scale": 0.0,
        }
        contextual_update_count = {
            "route": 0,
            "feedback": 0,
            "generative": 0,
            "ffn": 0,
            "scale": 0,
        }
        for name, gradient in gradients.items():
            component = ""
            if name in {"decoder_weight", "decoder_bias"}:
                lr = decoder_lr * optimizer_lr_multiplier
                group = "decoder"
            elif name == "embedding.weight":
                lr = embed_lr * optimizer_lr_multiplier
                group = "embedding"
            else:
                lr = state_lr * optimizer_lr_multiplier
                group = "context"
                if name.startswith(
                    (
                        "query_weights.",
                        "key_weights.",
                        "inverse_query_weights.",
                        "inverse_key_weights.",
                    )
                ):
                    lr *= self.config.route_lr_scale
                    component = "route"
                elif name.startswith("feedback_weights."):
                    lr *= self.config.feedback_lr_scale
                    component = "feedback"
                elif name.startswith("generative_weights."):
                    lr *= self.config.generative_lr_scale
                    component = "generative"
                elif name.startswith(("ffn_up_weights.", "ffn_down_weights.")):
                    lr *= self.config.ffn_lr_scale
                    component = "ffn"
            if name in self._muon_buffer_names:
                delta_rms = self._muon_update(
                    name,
                    parameters[name],
                    gradient,
                    lr=lr * self.config.muon_lr_scale,
                    weight_decay=weight_decay,
                    momentum=effective_muon_momentum,
                )
            else:
                delta_rms = self._adam_update(
                    name,
                    parameters[name],
                    gradient,
                    lr=lr,
                    weight_decay=weight_decay,
                )
            update_rms_by_group[group] += delta_rms
            update_count_by_group[group] += 1
            if group == "context" and component:
                contextual_update_rms[component] += delta_rms
                contextual_update_count[component] += 1

        layer_count = float(self.config.num_layers)
        diagnostics.update(
            {
                "update_diag_message_pc_attention_entropy": float(
                    (attention_entropy / layer_count).item()
                ),
                "update_diag_message_pc_route_tv": float(
                    (route_tv / layer_count).item()
                ),
                "update_diag_message_pc_route_credit_rms": float(
                    (route_credit_rms / layer_count).item()
                ),
                "update_diag_message_pc_local_prediction_rms": float(
                    (recognition_error_rms / layer_count).item()
                ),
                "update_diag_message_pc_bidirectional_generative_error_rms": float(
                    (generative_error_rms / layer_count).item()
                ),
                "update_diag_message_pc_decoder_update_rms": (
                    update_rms_by_group["decoder"]
                    / max(update_count_by_group["decoder"], 1)
                ),
                "update_diag_message_pc_context_update_rms": (
                    update_rms_by_group["context"]
                    / max(update_count_by_group["context"], 1)
                ),
                "update_diag_message_pc_route_update_rms": (
                    contextual_update_rms["route"]
                    / max(contextual_update_count["route"], 1)
                ),
                "update_diag_message_pc_feedback_update_rms": (
                    contextual_update_rms["feedback"]
                    / max(contextual_update_count["feedback"], 1)
                ),
                "update_diag_message_pc_generative_update_rms": (
                    contextual_update_rms["generative"]
                    / max(contextual_update_count["generative"], 1)
                ),
                "update_diag_message_pc_ffn_update_rms": (
                    contextual_update_rms["ffn"]
                    / max(contextual_update_count["ffn"], 1)
                ),
                "update_diag_message_pc_embedding_update_rms": (
                    update_rms_by_group["embedding"]
                    / max(update_count_by_group["embedding"], 1)
                ),
                "update_diag_message_pc_winner_margin_active_fraction": float(
                    winner_margin_active.item()
                ),
                "message_pc_learned_params": float(
                    self.learned_parameter_count()
                ),
                "message_pc_context_params": float(
                    self.contextual_parameter_count()
                ),
                "message_pc_persistent_slots": float(
                    self.persistent_slot_count()
                ),
                "message_pc_active_learned_params": float(
                    self.active_parameter_count()
                ),
                "update_diag_message_pc_uses_local_muon": float(
                    self.config.matrix_optimizer == "muon"
                ),
                "update_diag_message_pc_optimizer_lr_multiplier": (
                    optimizer_lr_multiplier
                ),
                "update_diag_message_pc_effective_muon_momentum": (
                    effective_muon_momentum
                ),
            }
        )
        self._last_update_diagnostics = diagnostics

        sequence_loss = token_loss.mean(dim=1).detach()
        final_state = deployment_state[:, -1].detach()
        if return_sequence_loss and return_token_loss and return_final_state:
            return loss, sequence_loss, token_loss.detach(), final_state
        if return_sequence_loss and return_token_loss:
            return loss, sequence_loss, token_loss.detach()
        if return_sequence_loss and return_final_state:
            return loss, sequence_loss, final_state
        if return_sequence_loss:
            return loss, sequence_loss
        if return_final_state:
            return loss, final_state
        return loss

    @torch.no_grad()
    def local_update(
        self,
        x: Tensor,
        y: Tensor,
        *,
        decoder_lr: float,
        state_lr: float,
        embed_lr: float,
        weight_decay: float = 0.0,
        current_step: int = 0,
        predictor_state_lr: float | None = None,
        return_sequence_loss: bool = False,
        return_token_loss: bool = False,
        return_final_state: bool = False,
        **_: object,
    ) -> float | tuple[float, Tensor] | tuple[float, Tensor, Tensor] | tuple[float, Tensor, Tensor, Tensor]:
        if self.config.inference_mode == "bidirectional_pc":
            return self._bidirectional_pc_local_update(
                x,
                y,
                decoder_lr=decoder_lr,
                state_lr=state_lr,
                embed_lr=embed_lr,
                weight_decay=weight_decay,
                current_step=current_step,
                return_sequence_loss=return_sequence_loss,
                return_token_loss=return_token_loss,
                return_final_state=return_final_state,
            )
        if self.config.inference_mode == "residual_error_pc":
            return self._residual_error_pc_local_update(
                x,
                y,
                decoder_lr=decoder_lr,
                state_lr=state_lr,
                embed_lr=embed_lr,
                weight_decay=weight_decay,
                current_step=current_step,
                return_sequence_loss=return_sequence_loss,
                return_token_loss=return_token_loss,
                return_final_state=return_final_state,
            )
        if self.config.inference_mode == "her_error_hierarchy":
            return self._hierarchical_error_local_update(
                x,
                y,
                decoder_lr=decoder_lr,
                state_lr=state_lr,
                embed_lr=embed_lr,
                predictor_state_lr=predictor_state_lr,
                weight_decay=weight_decay,
                current_step=current_step,
                return_sequence_loss=return_sequence_loss,
                return_token_loss=return_token_loss,
                return_final_state=return_final_state,
            )
        if self.config.inference_mode == "difference_target":
            return self._difference_target_local_update(
                x,
                y,
                decoder_lr=decoder_lr,
                state_lr=state_lr,
                embed_lr=embed_lr,
                weight_decay=weight_decay,
                current_step=current_step,
                return_sequence_loss=return_sequence_loss,
                return_token_loss=return_token_loss,
                return_final_state=return_final_state,
            )
        if self.config.inference_mode in {
            "recurrent_local",
            "layerwise_local",
        }:
            return self._recurrent_local_update(
                x,
                y,
                decoder_lr=decoder_lr,
                state_lr=state_lr,
                embed_lr=embed_lr,
                weight_decay=weight_decay,
                current_step=current_step,
                return_sequence_loss=return_sequence_loss,
                return_token_loss=return_token_loss,
                return_final_state=return_final_state,
            )
        free_states, _ = self._forward_states(x, with_cache=False)
        deployment_logits = self._decoder_logits(free_states[self._readout_depth()])
        token_loss = F.cross_entropy(
            deployment_logits.reshape(-1, self.config.vocab_size),
            y.reshape(-1),
            reduction="none",
        ).view_as(y)
        loss = float(token_loss.mean().item())
        effective_inference_lr = self._inference_lr(current_step)
        reciprocal_diagnostics: dict[str, float] = {}
        if (
            self.config.inference_mode == "meta_pc"
            and self.config.meta_pc_reciprocal_update_before_settle
        ):
            reciprocal_diagnostics = self._update_meta_pc_reciprocal_feedback(
                free_states,
                x,
                current_step=current_step,
            )
        settled_states, diagnostics = self._settle(
            free_states,
            y,
            inference_lr=effective_inference_lr,
            token_ids=x,
        )
        diagnostics["update_diag_message_pc_effective_inference_lr"] = (
            effective_inference_lr
        )
        if (
            self.config.inference_mode == "meta_pc"
            and not self.config.meta_pc_reciprocal_update_before_settle
        ):
            reciprocal_diagnostics = self._update_meta_pc_reciprocal_feedback(
                    free_states,
                    x,
                    current_step=current_step,
            )
        diagnostics.update(reciprocal_diagnostics)
        gradients: dict[str, Tensor] = {}
        free_feedback_sources: list[Tensor] = []
        free_feedback_predictions: list[Tensor] = []
        uses_learned_feedback = (
            self.config.inference_mode == "heuristic"
            or self.config.learned_feedback_proposal_scale > 0
        )
        if uses_learned_feedback:
            free_feedback_sources, free_feedback_predictions = self._feedback_path(
                free_states,
                y,
            )

        decoder_weight_gradient = torch.zeros_like(self.decoder_weight)
        decoder_bias_gradient = torch.zeros_like(self.decoder_bias)
        winner_margin_active = free_states[0].new_zeros(())
        layer_weight_total = 0.0
        if (
            self.config.inference_mode == "pc_energy"
            and self.config.decoder_learning_source == "settled"
        ):
            available_decoder_states = settled_states[1:]
        else:
            available_decoder_states = free_states[1:]
        decoder_states = [
            (state, layer_weight)
            for state, layer_weight in zip(
                available_decoder_states,
                self._decoder_state_weights(),
            )
            if layer_weight > 0.0
        ]
        for state, layer_weight in decoder_states:
            decoder_features = self._decoder_features(state)
            logits = F.linear(decoder_features, self.decoder_weight, self.decoder_bias)
            probabilities = F.softmax(logits, dim=-1)
            weight_gradient, bias_gradient = self._cross_entropy_gradient(
                decoder_features,
                y,
                probabilities,
            )
            margin_weight_gradient, margin_bias_gradient, margin_active = (
                self._winner_margin_decoder_gradients(
                    decoder_features,
                    y,
                    logits,
                )
            )
            weight_gradient.add_(margin_weight_gradient)
            bias_gradient.add_(margin_bias_gradient)
            decoder_weight_gradient.add_(weight_gradient, alpha=layer_weight)
            decoder_bias_gradient.add_(bias_gradient, alpha=layer_weight)
            winner_margin_active.add_(margin_active, alpha=layer_weight)
            layer_weight_total += layer_weight
        decoder_weight_gradient.div_(layer_weight_total)
        decoder_bias_gradient.div_(layer_weight_total)
        self._add_gradient(gradients, "decoder_weight", decoder_weight_gradient)
        self._add_gradient(gradients, "decoder_bias", decoder_bias_gradient)
        if (
            self.config.inference_mode == "meta_pc"
            and self.config.meta_pc_output_feedback_mode == "kp_learned"
        ):
            # KP feedback synapses receive the same local hidden/output-error
            # association as the decoder, without reading decoder weights.
            self._add_gradient(
                gradients,
                "feedback_embedding",
                decoder_weight_gradient.clone(),
            )
        elif (
            self.config.inference_mode == "meta_pc"
            and self.config.meta_pc_output_feedback_mode == "kp_factorized"
        ):
            if (
                self.feedback_token_factors is None
                or self.feedback_state_factors is None
            ):
                raise RuntimeError("factorized output feedback is not initialized")
            # Chain the same local hidden/output-error association through
            # B = A C. Neither factor reads the decoder matrix.
            self._add_gradient(
                gradients,
                "feedback_token_factors",
                decoder_weight_gradient @ self.feedback_state_factors.t(),
            )
            self._add_gradient(
                gradients,
                "feedback_state_factors",
                self.feedback_token_factors.t() @ decoder_weight_gradient,
            )

        feedback_embedding_gradient = torch.zeros_like(self.feedback_embedding)
        attention_entropy = free_states[0].new_zeros(())
        route_tv = free_states[0].new_zeros(())
        route_credit_rms = free_states[0].new_zeros(())
        prediction_error_rms = free_states[0].new_zeros(())
        amortization_gradient_rms = free_states[0].new_zeros(())
        amortization_gradient_count = 0
        free_local_token_credit_rms = free_states[0].new_zeros(())
        free_local_token_credit_gradient_rms = free_states[0].new_zeros(())
        free_local_token_credit_gradient_count = 0
        free_local_token_embedding_input_gradient = torch.zeros_like(
            free_states[0]
        )
        free_local_token_embedding_credit_rms = free_states[0].new_zeros(())
        causal_mask = self._causal_mask(x.shape[1], x.device)
        token_count = float(max(x.numel(), 1))

        for layer in range(self.config.num_layers):
            target_state = settled_states[layer + 1]
            uses_forward_update = (
                self.config.forward_update_top_layers > 0
                and layer
                >= self.config.num_layers
                - self.config.forward_update_top_layers
            )
            if self.config.learning_phase_source == "free":
                phase_sources = ((free_states[layer], 1.0),)
            elif self.config.learning_phase_source == "dual" and layer > 0:
                phase_sources = (
                    (settled_states[layer], 0.5),
                    (free_states[layer], 0.5),
                )
            elif uses_forward_update:
                phase_sources = ((free_states[layer], 1.0),)
            else:
                phase_sources = ((settled_states[layer], 1.0),)
            for lower_state, phase_weight in phase_sources:
                (
                    phase_gradients,
                    phase_attention_entropy,
                    phase_route_tv,
                    phase_route_credit_rms,
                    phase_prediction_error_rms,
                ) = self._phase_block_gradients(
                    layer,
                    lower_state,
                    target_state,
                    route_lower_state=free_states[layer],
                    causal_mask=causal_mask,
                    token_count=token_count,
                    token_ids=x,
                    sensory_state=free_states[0],
                    encoder_memory=self._causal_encoder_memory(
                        free_states,
                        layer,
                    ),
                )
                for name, phase_gradient in phase_gradients.items():
                    if (
                        self.config.ffn_learning_source == "free_to_settled"
                        and name.startswith(
                            ("ffn_up_weights.", "ffn_down_weights.")
                        )
                    ):
                        continue
                    self._add_gradient(
                        gradients,
                        name,
                        phase_weight * phase_gradient,
                    )
                attention_entropy = (
                    attention_entropy + phase_weight * phase_attention_entropy
                )
                route_tv = route_tv + phase_weight * phase_route_tv
                route_credit_rms = (
                    route_credit_rms + phase_weight * phase_route_credit_rms
                )
                prediction_error_rms = (
                    prediction_error_rms
                    + phase_weight * phase_prediction_error_rms
                )

            if (
                self.config.ffn_dim > 0
                and self.config.ffn_learning_source == "free_to_settled"
            ):
                ffn_amortization_gradients, _, _, _, _ = (
                    self._phase_block_gradients(
                        layer,
                        free_states[layer],
                        target_state,
                        route_lower_state=free_states[layer],
                        causal_mask=causal_mask,
                        token_count=token_count,
                        token_ids=x,
                        sensory_state=free_states[0],
                        encoder_memory=self._causal_encoder_memory(
                            free_states,
                            layer,
                        ),
                    )
                )
                for name, gradient in ffn_amortization_gradients.items():
                    if name.startswith(
                        ("ffn_up_weights.", "ffn_down_weights.")
                    ):
                        self._add_gradient(gradients, name, gradient)

            if layer > 0 and self.config.free_phase_amortization_scale > 0:
                amortization_gradients, _, _, _, _ = self._phase_block_gradients(
                    layer,
                    free_states[layer],
                    target_state,
                    route_lower_state=free_states[layer],
                    causal_mask=causal_mask,
                    token_count=token_count,
                    token_ids=x,
                    sensory_state=free_states[0],
                    encoder_memory=self._causal_encoder_memory(
                        free_states,
                        layer,
                    ),
                )
                for name, amortization_gradient in amortization_gradients.items():
                    scaled_gradient = (
                        self.config.free_phase_amortization_scale
                        * amortization_gradient
                    )
                    self._add_gradient(gradients, name, scaled_gradient)
                    amortization_gradient_rms = (
                        amortization_gradient_rms + _rms(scaled_gradient)
                    )
                    amortization_gradient_count += 1

            if self.config.free_local_token_credit_scale > 0 or (
                layer == 0
                and self.config.free_local_token_embedding_credit_scale > 0
            ):
                negative_state_gradient, _ = (
                    self._decoder_negative_state_gradient(
                        free_states[layer + 1],
                        y,
                    )
                )
                if self.config.free_local_token_credit_scale > 0:
                    free_local_target = (
                        free_states[layer + 1] + negative_state_gradient
                    )
                    local_token_gradients, _, _, _, _ = (
                        self._phase_block_gradients(
                            layer,
                            free_states[layer],
                            free_local_target,
                            route_lower_state=free_states[layer],
                            causal_mask=causal_mask,
                            token_count=token_count,
                            token_ids=x,
                            sensory_state=free_states[0],
                            encoder_memory=self._causal_encoder_memory(
                                free_states,
                                layer,
                            ),
                        )
                    )
                    for name, local_token_gradient in local_token_gradients.items():
                        scaled_gradient = (
                            self.config.free_local_token_credit_scale
                            * local_token_gradient
                        )
                        self._add_gradient(gradients, name, scaled_gradient)
                        free_local_token_credit_gradient_rms = (
                            free_local_token_credit_gradient_rms
                            + _rms(scaled_gradient)
                        )
                        free_local_token_credit_gradient_count += 1
                    free_local_token_credit_rms = (
                        free_local_token_credit_rms
                        + _rms(negative_state_gradient)
                    )
                if (
                    layer == 0
                    and self.config.free_local_token_embedding_credit_scale > 0
                ):
                    first_cache = self._block(
                        0,
                        free_states[0],
                        token_ids=x,
                        sensory_state=free_states[0],
                        encoder_memory=self._causal_encoder_memory(
                            free_states,
                            0,
                        ),
                    )
                    embedding_input_gradient = self._embedding_state_vjp(
                        x,
                        free_states[0],
                        self._block_input_signal(
                            0,
                            free_states[0],
                            first_cache,
                            -negative_state_gradient,
                        ),
                    )
                    embedding_input_gradient.mul_(
                        self.config.free_local_token_embedding_credit_scale
                    )
                    free_local_token_embedding_input_gradient.add_(
                        embedding_input_gradient
                    )
                    free_local_token_embedding_credit_rms = _rms(
                        embedding_input_gradient
                    )

            if uses_learned_feedback:
                feedback_source = free_feedback_sources[layer]
                predicted_correction = free_feedback_predictions[layer]
                desired_correction = settled_states[layer + 1] - free_states[layer + 1]
                if self.config.inference_mode == "pc_energy":
                    total_accepted_rate = max(
                        diagnostics.get(
                            "update_diag_message_pc_inference_total_accepted_lr",
                            effective_inference_lr,
                        ),
                        1e-8,
                    )
                    desired_correction = desired_correction / (
                        total_accepted_rate
                        * max(self.config.token_feedback_scale, 1e-8)
                    )
                correction_gradient = predicted_correction - desired_correction
                flat_feedback_source = feedback_source.reshape(-1, self.config.hidden_dim)
                flat_correction_gradient = correction_gradient.reshape(
                    -1,
                    self.config.hidden_dim,
                )
                self._add_gradient(
                    gradients,
                    f"feedback_weights.{layer}",
                    (flat_correction_gradient.t() @ flat_feedback_source) / token_count,
                )
        if uses_learned_feedback:
            feedback_embedding_gradient.copy_(self.feedback_embedding - self.decoder_weight)
            self._add_gradient(gradients, "feedback_embedding", feedback_embedding_gradient)
        if self.config.inference_mode == "heuristic":
            first_lower_prediction = self._generative_prediction(0, settled_states[1])
            input_gradient = self._embedding_state_vjp(
                x,
                free_states[0],
                free_states[0] - first_lower_prediction,
            )
        else:
            first_cache = self._block(
                0,
                settled_states[0],
                token_ids=x,
                sensory_state=free_states[0],
                encoder_memory=self._causal_encoder_memory(
                    free_states,
                    0,
                ),
            )
            first_error = settled_states[1] - first_cache.prediction
            if self.config.meta_pc_feedback_mode == "stage_reciprocal":
                input_signal = self._stage_reciprocal_block_input_signal(
                    0,
                    settled_states[0],
                    first_cache,
                    first_error,
                )
            else:
                input_signal = self._block_input_signal(
                    0,
                    settled_states[0],
                    first_cache,
                    first_error,
                )
            input_gradient = self._embedding_state_vjp(
                x,
                free_states[0],
                -input_signal,
            )
        input_gradient.add_(free_local_token_embedding_input_gradient)
        embedding_gradient = torch.zeros_like(self.embedding.weight)
        embedding_gradient.index_add_(0, x.reshape(-1), input_gradient.reshape(-1, self.config.hidden_dim))
        embedding_gradient.div_(token_count)
        self._add_gradient(gradients, "embedding.weight", embedding_gradient)

        decoder_output_feedback = (
            self.decoder_weight[y] - probabilities @ self.decoder_weight
        )
        independent_output_feedback = self._independent_output_feedback(
            probabilities,
            y,
        )
        output_feedback_cosine = F.cosine_similarity(
            independent_output_feedback.float().reshape(
                -1,
                self.config.hidden_dim,
            ),
            decoder_output_feedback.float().reshape(
                -1,
                self.config.hidden_dim,
            ),
            dim=-1,
            eps=1e-8,
        ).mean()
        output_feedback_sign_agreement = (
            independent_output_feedback.sign()
            == decoder_output_feedback.sign()
        ).float().mean()

        optimizer_lr_multiplier = self._optimizer_lr_multiplier(current_step)
        effective_residual_stream_lr_scale = self._residual_stream_lr_scale(
            current_step
        )
        effective_muon_momentum = self._muon_momentum(current_step)
        self.local_adam_step.add_(1)
        parameters = dict(self.named_parameters())
        update_rms_by_group = {"decoder": 0.0, "context": 0.0, "embedding": 0.0}
        update_count_by_group = {"decoder": 0, "context": 0, "embedding": 0}
        contextual_update_rms = {
            "route": 0.0,
            "cross_attention": 0.0,
            "feedback": 0.0,
            "generative": 0.0,
            "ffn": 0.0,
            "scale": 0.0,
        }
        contextual_update_count = {
            "route": 0,
            "cross_attention": 0,
            "feedback": 0,
            "generative": 0,
            "ffn": 0,
            "scale": 0,
        }
        for name, gradient in gradients.items():
            component = ""
            if name in {"decoder_weight", "decoder_bias"}:
                lr = decoder_lr * optimizer_lr_multiplier
                group = "decoder"
            elif name == "embedding.weight":
                lr = embed_lr * optimizer_lr_multiplier
                group = "embedding"
            elif name.startswith("token_value_tables."):
                lr = (
                    embed_lr
                    * self.config.token_value_lr_scale
                    * optimizer_lr_multiplier
                )
                group = "embedding"
            elif (
                name
                in {
                    "feedback_embedding",
                    "feedback_token_factors",
                    "feedback_state_factors",
                }
                and self.config.inference_mode == "meta_pc"
                and self.config.meta_pc_output_feedback_mode
                in {"kp_learned", "kp_factorized"}
            ):
                lr = (
                    decoder_lr
                    * self.config.meta_pc_output_feedback_lr_scale
                    * optimizer_lr_multiplier
                )
                group = "decoder"
            elif name == "residual_stream_scales":
                lr = (
                    state_lr
                    * effective_residual_stream_lr_scale
                    * optimizer_lr_multiplier
                )
                group = "context"
                component = "scale"
            elif name == "cross_attention_gates":
                lr = (
                    state_lr
                    * self.config.cross_attention_gate_lr_scale
                    * optimizer_lr_multiplier
                )
                group = "context"
                component = "cross_attention"
            else:
                lr = state_lr * optimizer_lr_multiplier
                group = "context"
                if name.startswith(
                    (
                        "cross_query_weights.",
                        "cross_key_weights.",
                        "cross_value_weights.",
                        "cross_message_weights.",
                    )
                ):
                    lr *= self.config.cross_attention_lr_scale
                    if name.startswith(
                        ("cross_query_weights.", "cross_key_weights.")
                    ):
                        lr *= self.config.route_lr_scale
                    component = "cross_attention"
                elif name.startswith(("query_weights.", "key_weights.")):
                    lr *= self.config.route_lr_scale
                    component = "route"
                elif name == "feedback_embedding" or name.startswith("feedback_weights."):
                    lr *= self.config.feedback_lr_scale
                    component = "feedback"
                elif name.startswith("generative_weights."):
                    lr *= self.config.generative_lr_scale
                    component = "generative"
                elif name.startswith(("ffn_up_weights.", "ffn_down_weights.")):
                    lr *= self.config.ffn_lr_scale
                    component = "ffn"
            if name in self._muon_buffer_names:
                delta_rms = self._muon_update(
                    name,
                    parameters[name],
                    gradient,
                    lr=lr * self.config.muon_lr_scale,
                    weight_decay=weight_decay,
                    momentum=effective_muon_momentum,
                )
            else:
                delta_rms = self._adam_update(
                    name,
                    parameters[name],
                    gradient,
                    lr=lr,
                    weight_decay=weight_decay,
                )
            update_rms_by_group[group] += delta_rms
            update_count_by_group[group] += 1
            if group == "context" and component:
                contextual_update_rms[component] += delta_rms
                contextual_update_count[component] += 1

        if self.config.meta_pc_output_feedback_mode == "kp_factorized":
            feedback_decoder_cosine = output_feedback_cosine
        else:
            feedback_decoder_cosine = F.cosine_similarity(
                self.feedback_embedding.float().reshape(1, -1),
                self.decoder_weight.float().reshape(1, -1),
            ).squeeze(0)
        layer_count = float(self.config.num_layers)
        diagnostics.update(
            {
                "update_diag_message_pc_attention_entropy": float((attention_entropy / layer_count).item()),
                "update_diag_message_pc_route_tv": float((route_tv / layer_count).item()),
                "update_diag_message_pc_route_credit_rms": float(
                    (route_credit_rms / layer_count).item()
                ),
                "update_diag_message_pc_route_uses_local_energy": float(
                    self.config.route_update_mode == "local_energy"
                    or (
                        self.config.inference_mode == "pc_energy"
                        and self.config.route_update_mode
                        != "token_error_teacher"
                    )
                ),
                "update_diag_message_pc_route_uses_token_error_teacher": float(
                    self.config.route_update_mode == "token_error_teacher"
                ),
                "update_diag_message_pc_uses_coherent_energy": float(
                    self.config.inference_mode == "pc_energy"
                ),
                "update_diag_message_pc_meta_variance_normalized_weights": float(
                    self.config.meta_pc_variance_normalize_weights
                ),
                "update_diag_message_pc_output_feedback_cosine": float(
                    output_feedback_cosine.item()
                ),
                "update_diag_message_pc_output_feedback_sign_agreement": float(
                    output_feedback_sign_agreement.item()
                ),
                "update_diag_message_pc_meta_variance_normalization_mean_scale": (
                    self._meta_pc_variance_normalize_forward_weights()
                    if self.config.inference_mode == "meta_pc"
                    and self.config.meta_pc_variance_normalize_weights
                    else 1.0
                ),
                "update_diag_message_pc_learning_uses_free_source": float(
                    self.config.learning_phase_source == "free"
                ),
                "update_diag_message_pc_learning_uses_dual_source": float(
                    self.config.learning_phase_source == "dual"
                ),
                "update_diag_message_pc_forward_update_top_layers": float(
                    self.config.forward_update_top_layers
                ),
                "update_diag_message_pc_free_phase_amortization_scale": (
                    self.config.free_phase_amortization_scale
                ),
                "update_diag_message_pc_free_phase_amortization_gradient_rms": (
                    float(
                        (
                            amortization_gradient_rms
                            / max(amortization_gradient_count, 1)
                        ).item()
                    )
                ),
                "update_diag_message_pc_free_local_token_credit_scale": (
                    self.config.free_local_token_credit_scale
                ),
                "update_diag_message_pc_free_local_token_credit_rms": float(
                    (free_local_token_credit_rms / layer_count).item()
                ),
                "update_diag_message_pc_free_local_token_credit_gradient_rms": (
                    float(
                        (
                            free_local_token_credit_gradient_rms
                            / max(free_local_token_credit_gradient_count, 1)
                        ).item()
                    )
                ),
                "update_diag_message_pc_free_local_token_embedding_credit_scale": (
                    self.config.free_local_token_embedding_credit_scale
                ),
                "update_diag_message_pc_free_local_token_embedding_credit_rms": (
                    float(free_local_token_embedding_credit_rms.item())
                ),
                "update_diag_message_pc_decoder_uses_free_source": float(
                    self.config.decoder_learning_source == "free"
                ),
                "update_diag_message_pc_learned_feedback_proposal_rms": (
                    float(
                        torch.stack(
                            [_rms(value) for value in free_feedback_predictions]
                        ).mean().item()
                    )
                    if free_feedback_predictions
                    else 0.0
                ),
                "update_diag_message_pc_local_prediction_rms": float((prediction_error_rms / layer_count).item()),
                "update_diag_message_pc_decoder_update_rms": update_rms_by_group["decoder"]
                / max(update_count_by_group["decoder"], 1),
                "update_diag_message_pc_context_update_rms": update_rms_by_group["context"]
                / max(update_count_by_group["context"], 1),
                "update_diag_message_pc_route_update_rms": contextual_update_rms["route"]
                / max(contextual_update_count["route"], 1),
                "update_diag_message_pc_cross_attention_update_rms": (
                    contextual_update_rms["cross_attention"]
                    / max(contextual_update_count["cross_attention"], 1)
                ),
                "update_diag_message_pc_feedback_update_rms": contextual_update_rms["feedback"]
                / max(contextual_update_count["feedback"], 1),
                "update_diag_message_pc_generative_update_rms": contextual_update_rms["generative"]
                / max(contextual_update_count["generative"], 1),
                "update_diag_message_pc_ffn_update_rms": contextual_update_rms["ffn"]
                / max(contextual_update_count["ffn"], 1),
                "update_diag_message_pc_residual_stream_scale_update_rms": contextual_update_rms["scale"]
                / max(contextual_update_count["scale"], 1),
                "update_diag_message_pc_effective_residual_stream_lr_scale": (
                    effective_residual_stream_lr_scale
                ),
                "update_diag_message_pc_ffn_uses_free_to_settled": float(
                    self.config.ffn_learning_source == "free_to_settled"
                ),
                "update_diag_message_pc_embedding_update_rms": update_rms_by_group["embedding"]
                / max(update_count_by_group["embedding"], 1),
                "update_diag_message_pc_winner_margin_active_fraction": float(
                    (winner_margin_active / layer_weight_total).item()
                ),
                "update_diag_message_pc_feedback_decoder_cosine": float(
                    feedback_decoder_cosine.item()
                ),
                "update_diag_message_pc_query_rms": float(
                    torch.stack([_rms(weight) for weight in self.query_weights]).mean().item()
                ),
                "update_diag_message_pc_key_rms": float(
                    torch.stack([_rms(weight) for weight in self.key_weights]).mean().item()
                ),
                "message_pc_learned_params": float(self.learned_parameter_count()),
                "message_pc_context_params": float(self.contextual_parameter_count()),
                "message_pc_persistent_slots": float(self.persistent_slot_count()),
                "message_pc_active_learned_params": float(
                    self.active_parameter_count()
                ),
                "update_diag_message_pc_uses_local_muon": float(
                    self.config.matrix_optimizer == "muon"
                ),
                "update_diag_message_pc_optimizer_lr_multiplier": (
                    optimizer_lr_multiplier
                ),
                "update_diag_message_pc_effective_muon_momentum": (
                    effective_muon_momentum
                ),
            }
        )
        self._last_update_diagnostics = diagnostics

        sequence_loss = token_loss.mean(dim=1).detach()
        final_state = free_states[-1][:, -1].detach()
        if return_sequence_loss and return_token_loss and return_final_state:
            return loss, sequence_loss, token_loss.detach(), final_state
        if return_sequence_loss and return_token_loss:
            return loss, sequence_loss, token_loss.detach()
        if return_sequence_loss and return_final_state:
            return loss, sequence_loss, final_state
        if return_sequence_loss:
            return loss, sequence_loss
        if return_final_state:
            return loss, final_state
        return loss

    def prune_edges(self, fraction: float, *, min_fan_in: int = 1) -> int:
        del fraction, min_fan_in
        return 0

    def connection_stats(self) -> dict[str, float]:
        return {
            "active_edges": 0.0,
            "mean_fan_in": float(self.config.hidden_dim),
            "message_pc_learned_params": float(self.learned_parameter_count()),
            "message_pc_active_learned_params": float(self.active_parameter_count()),
            "message_pc_deployment_params": float(
                self.deployment_parameter_count()
            ),
            "message_pc_training_feedback_params": float(
                self.training_feedback_parameter_count()
            ),
            "message_pc_context_params": float(self.contextual_parameter_count()),
            "message_pc_persistent_slots": float(self.persistent_slot_count()),
            "message_pc_inference_steps": float(self.config.inference_steps),
            "message_pc_recurrent_shared_steps": float(
                self.config.recurrent_shared_steps
            ),
            "message_pc_recurrent_step_scale": float(
                self.config.recurrent_step_scale
            ),
            "message_pc_uses_recurrent_local": float(
                self.config.inference_mode == "recurrent_local"
            ),
            "message_pc_uses_layerwise_local": float(
                self.config.inference_mode == "layerwise_local"
            ),
            "message_pc_inference_uses_heavy_ball": float(
                self.config.inference_optimizer == "heavy_ball"
            ),
            "message_pc_inference_uses_conjugate_gradient": float(
                self.config.inference_optimizer == "conjugate_gradient"
            ),
            "message_pc_inference_momentum": float(
                self.config.inference_momentum
            ),
            "message_pc_spiking_precision_scale": float(
                self.config.spiking_precision_scale
            ),
            "message_pc_uses_depth_mup": float(
                self.config.block_parameterization == "depth_mup"
            ),
            "message_pc_block_weight_scale": self._block_weight_scale(),
            "message_pc_fast_attention_window": float(
                self.config.fast_attention_window
            ),
            "message_pc_fast_attention_layers": float(
                self.config.fast_attention_layers
            ),
            "message_pc_ffn_dim": float(self.config.ffn_dim),
            "message_pc_ffn_lr_scale": float(self.config.ffn_lr_scale),
            "message_pc_ffn_uses_free_to_settled": float(
                self.config.ffn_learning_source == "free_to_settled"
            ),
            "message_pc_uses_token_value_embeddings": float(
                self.config.token_value_embeddings
            ),
            "message_pc_token_value_params": float(
                sum(table.numel() for table in self.token_value_tables.values())
            ),
            "message_pc_token_value_scale": self.config.token_value_scale,
            "message_pc_token_value_lr_scale": (
                self.config.token_value_lr_scale
            ),
            "message_pc_uses_causal_encoder_decoder": float(
                self.config.causal_encoder_decoder
            ),
            "message_pc_uses_meta_cross_memory_skip_credit": float(
                self.config.meta_pc_cross_memory_skip_credit
            ),
            "message_pc_meta_cross_memory_skip_credit_scale": (
                self.config.meta_pc_cross_memory_skip_credit_scale
            ),
            "message_pc_causal_encoder_depth": float(
                self.config.causal_encoder_depth
            ),
            "message_pc_cross_attention_params": float(
                sum(
                    parameter.numel()
                    for parameter in (
                        *self.cross_query_weights[
                            self.config.causal_encoder_depth :
                        ],
                        *self.cross_key_weights[
                            self.config.causal_encoder_depth :
                        ],
                        *self.cross_value_weights[
                            self.config.causal_encoder_depth :
                        ],
                        *self.cross_message_weights[
                            self.config.causal_encoder_depth :
                        ],
                    )
                )
                + (
                    self.cross_attention_gates[
                        self.config.causal_encoder_depth :
                    ].numel()
                    if self.cross_attention_gates is not None
                    else 0
                )
            ),
            "message_pc_cross_attention_lr_scale": (
                self.config.cross_attention_lr_scale
            ),
            "message_pc_cross_attention_gate_lr_scale": (
                self.config.cross_attention_gate_lr_scale
            ),
            "message_pc_cross_attention_output_init_gain": (
                self.config.cross_attention_output_init_gain
            ),
            "message_pc_cross_attention_gate_mean": (
                float(
                    torch.sigmoid(
                        self.cross_attention_gates[
                            self.config.causal_encoder_depth :
                        ]
                    )
                    .float()
                    .mean()
                    .item()
                )
                if self.cross_attention_gates is not None
                else 0.0
            ),
            "message_pc_sensory_reinjection_start": (
                self.config.sensory_reinjection_start
            ),
            "message_pc_sensory_reinjection_end": (
                self.config.sensory_reinjection_end
            ),
            "message_pc_sensory_reinjection_uses_pre_block": float(
                self.config.sensory_reinjection_mode == "pre_block"
            ),
            "message_pc_residual_stream_scale_start": (
                self.config.residual_stream_scale_start
            ),
            "message_pc_residual_stream_scale_end": (
                self.config.residual_stream_scale_end
            ),
            "message_pc_learns_residual_stream_scales": float(
                self.residual_stream_scales is not None
            ),
            "message_pc_residual_stream_uses_channel_precision": float(
                self.config.residual_stream_scale_mode == "channel"
            ),
            "message_pc_residual_stream_lr_scale": (
                self.config.residual_stream_lr_scale
            ),
            "message_pc_residual_stream_lr_scale_final": (
                self.config.residual_stream_lr_scale_final
            ),
            "message_pc_residual_stream_lr_schedule_start_step": float(
                self.config.residual_stream_lr_schedule_start_step
            ),
            "message_pc_residual_stream_lr_schedule_end_step": float(
                self.config.residual_stream_lr_schedule_end_step
            ),
            "message_pc_residual_stream_actual_first": (
                self._residual_stream_scale_mean(0)
            ),
            "message_pc_residual_stream_actual_last": (
                self._residual_stream_scale_mean(self.config.num_layers - 1)
            ),
            "message_pc_residual_stream_actual_std": (
                float(self.residual_stream_scales.float().std(unbiased=False).item())
                if self.residual_stream_scales is not None
                else 0.0
            ),
            "message_pc_meta_uses_decoder_output_feedback": float(
                self.config.meta_pc_output_feedback_mode
                == "decoder_transpose"
            ),
            "message_pc_meta_uses_fixed_output_feedback": float(
                self.config.meta_pc_output_feedback_mode == "fixed_random"
            ),
            "message_pc_meta_uses_kp_output_feedback": float(
                self.config.meta_pc_output_feedback_mode == "kp_learned"
            ),
            "message_pc_meta_uses_factorized_output_feedback": float(
                self.config.meta_pc_output_feedback_mode == "kp_factorized"
            ),
            "message_pc_meta_output_feedback_rank": float(
                self.config.meta_pc_output_feedback_rank
            ),
            "message_pc_meta_output_feedback_factor_params": float(
                (self.feedback_token_factors.numel() if self.feedback_token_factors is not None else 0)
                + (self.feedback_state_factors.numel() if self.feedback_state_factors is not None else 0)
            ),
            "message_pc_meta_uses_output_feedback_values": float(
                self.config.meta_pc_output_feedback_values
            ),
            "message_pc_meta_uses_exact_vjp": float(
                self.config.inference_mode == "meta_pc"
                and self.config.meta_pc_feedback_mode == "exact_vjp"
            ),
            "message_pc_meta_uses_learned_reciprocal": float(
                self.config.inference_mode == "meta_pc"
                and self.config.meta_pc_feedback_mode == "learned_reciprocal"
            ),
            "message_pc_meta_uses_stage_reciprocal": float(
                self.config.inference_mode == "meta_pc"
                and self.config.meta_pc_feedback_mode == "stage_reciprocal"
            ),
            "message_pc_meta_reciprocal_learning_rate": (
                self.config.meta_pc_reciprocal_learning_rate
            ),
            "message_pc_meta_reciprocal_noise_std": (
                self.config.meta_pc_reciprocal_noise_std
            ),
            "message_pc_meta_uses_pre_settle_reciprocal_update": float(
                self.config.meta_pc_reciprocal_update_before_settle
            ),
            "message_pc_meta_uses_state_gated_reciprocal": float(
                self.config.meta_pc_state_gated_reciprocal
            ),
            "message_pc_meta_state_gated_reciprocal_params": float(
                sum(
                    weight.numel()
                    for weight in self.feedback_state_weights[1:]
                )
            ),
            "message_pc_meta_stage_reciprocal_params": float(
                sum(
                    weight.numel()
                    for weights in (
                        self.feedback_query_weights,
                        self.feedback_key_weights,
                        self.feedback_value_weights,
                        self.feedback_message_weights,
                        self.feedback_ffn_up_weights,
                        self.feedback_ffn_down_weights,
                    )
                    for weight in weights
                )
            ),
            "message_pc_meta_uses_relinearized_inference": float(
                self.config.meta_pc_relinearize_inference
            ),
            "message_pc_meta_reciprocal_params": float(
                sum(
                    weight.numel()
                    for weight in self.feedback_weights[1:]
                )
                if self.config.meta_pc_feedback_mode == "learned_reciprocal"
                else 0
            ),
            "message_pc_meta_output_feedback_lr_scale": (
                self.config.meta_pc_output_feedback_lr_scale
            ),
            "message_pc_readout_layer": float(self._readout_depth()),
            "message_pc_readout_objective_share": float(
                self.config.readout_objective_share
            ),
            "message_pc_learning_uses_free_source": float(
                self.config.learning_phase_source == "free"
            ),
            "message_pc_learning_uses_dual_source": float(
                self.config.learning_phase_source == "dual"
            ),
            "message_pc_forward_update_top_layers": float(
                self.config.forward_update_top_layers
            ),
            "message_pc_free_phase_amortization_scale": float(
                self.config.free_phase_amortization_scale
            ),
            "message_pc_free_local_token_credit_scale": float(
                self.config.free_local_token_credit_scale
            ),
            "message_pc_free_local_token_embedding_credit_scale": float(
                self.config.free_local_token_embedding_credit_scale
            ),
            "message_pc_decoder_uses_free_source": float(
                self.config.decoder_learning_source == "free"
            ),
            "message_pc_learned_feedback_proposal_scale": float(
                self.config.learned_feedback_proposal_scale
            ),
            "message_pc_residual_error_uses_learned_only_transport": float(
                self.config.residual_error_transport_mode == "learned_only"
            ),
            "message_pc_residual_error_target_layer": (
                self.config.residual_error_target_layer
            ),
            "message_pc_residual_error_proxy_memory_scale": float(
                self.config.residual_error_proxy_memory_scale
            ),
            "message_pc_residual_error_proxy_memory_decay": float(
                self.config.residual_error_proxy_memory_decay
            ),
            "message_pc_residual_error_upper_proxy_state_scale": float(
                self.config.residual_error_upper_proxy_state_scale
            ),
            "message_pc_residual_error_upper_proxy_state_decay": float(
                self.config.residual_error_upper_proxy_state_decay
            ),
            "message_pc_residual_error_upper_content_state_scale": float(
                self.config.residual_error_upper_content_state_scale
            ),
            "message_pc_residual_error_upper_content_state_decay": float(
                self.config.residual_error_upper_content_state_decay
            ),
            "message_pc_residual_error_upper_content_write_period": float(
                self.config.residual_error_upper_content_write_period
            ),
            "message_pc_residual_error_upper_content_write_offset": float(
                self.config.residual_error_upper_content_write_offset
            ),
            "message_pc_residual_error_upper_content_read_period": float(
                self.config.residual_error_upper_content_read_period
            ),
            "message_pc_residual_error_upper_content_read_offset": float(
                self.config.residual_error_upper_content_read_offset
            ),
            "message_pc_residual_error_upper_content_read_delay": float(
                self.config.residual_error_upper_content_read_delay
            ),
            "message_pc_residual_error_upper_content_bind_current_scale": float(
                self.config.residual_error_upper_content_bind_current_scale
            ),
            "message_pc_residual_error_upper_content_bind_product_scale": float(
                self.config.residual_error_upper_content_bind_product_scale
            ),
            "message_pc_residual_error_upper_content_bind_current_delay": float(
                self.config.residual_error_upper_content_bind_current_delay
            ),
            "message_pc_hierarchical_error_context_mode": (
                self.config.hierarchical_error_context_mode
            ),
            "message_pc_hierarchical_error_memory_scale": float(
                self.config.hierarchical_error_memory_scale
            ),
            "message_pc_uses_difference_target": float(
                self.config.inference_mode == "difference_target"
            ),
            "message_pc_uses_bidirectional_pc": float(
                self.config.inference_mode == "bidirectional_pc"
            ),
            "message_pc_bidirectional_uses_free_settled_deployment": float(
                self.config.bidirectional_deployment_settling
            ),
            "message_pc_uses_residual_error_pc": float(
                self.config.inference_mode == "residual_error_pc"
            ),
            "message_pc_difference_target_inverse_noise": float(
                self.config.difference_target_inverse_noise
            ),
            "message_pc_difference_target_inverse_damping": float(
                self.config.difference_target_inverse_damping
            ),
            "message_pc_difference_target_uses_direct_path": float(
                self.config.difference_target_path_mode == "direct"
            ),
        }
