from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from pfc.native_pc_lm import NativePCConfig, NativeReciprocalPCLanguageModel
from pfc.native_pc_training import NativePCTrainer, NativePCTrainingConfig
from pfc.ternary_local_optimizer import (
    GROUPED_ABSMEAN_ALGORITHM,
    TernaryLocalOptimizer,
)


ARCHITECTURE_PHASE = 827
DEFAULT_SEED = 539

# Public release names. Historical checkpoint keys and internal stage_reciprocal
# identifiers remain unchanged.
ReciprocalPC = NativeReciprocalPCLanguageModel
ReciprocalPCConfig = NativePCConfig
ReciprocalPCTrainer = NativePCTrainer
ReciprocalPCTrainingConfig = NativePCTrainingConfig


def model_config() -> ReciprocalPCConfig:
    """Return the exact accepted Phase827 forward configuration."""

    return replace(
        ReciprocalPCConfig(),
        associative_memory_addressing=(
            "pair_miss_semantic_backoff_dual_order_indexed_lazy_multiscale_count_sketch"
        ),
        associative_memory_decay=0.99,
        associative_memory_layers=1,
        associative_memory_rank=2184,
        associative_memory_scale=11.5,
        associative_memory_value_width=32,
        bos_token_id=32759,
        contractive_reversible_retention=True,
        decoupled_transition_write_gate=True,
        deeploop_postnorm_settling=True,
        depth=2,
        ffn_width=1040,
        fixed_ternary_hadamard_lexical_predictor=True,
        fused_indexed_count_sketch_inference=True,
        hybrid_projected_readout=True,
        learn_residual_stream_scales=True,
        loop_aware_settling_scaling=True,
        pair_norm_bounded_convex_recurrence=True,
        predictive_future_error_key_rank=32,
        projected_readout=True,
        projected_self_bilinear_token_observation=True,
        representably_contractive_retention=True,
        residual_stream=True,
        reversible_recurrent_coupling=True,
        scale_invariant_associative_state=True,
        shared_embedding_decoder=True,
        swiglu_ffn=True,
        temporal_hgf_precision_carry=True,
        temporal_rotation_min_period=4.0,
        vocab_size=32768,
        width=512,
    )


def training_config() -> ReciprocalPCTrainingConfig:
    """Return the exact accepted Phase827 strict-local training configuration."""

    return replace(
        ReciprocalPCTrainingConfig(),
        active_gradient_frontier=True,
        calibrated_settling_mirrors=True,
        calibrated_transition_credit=True,
        charged_held_query_displacement_projection=True,
        delayed_consequence_selective_credit=True,
        direct_ordered_branch_credit=True,
        direct_ordered_exact_source_accumulation=True,
        direct_ordered_memory_source_credit=True,
        direct_ordered_spatiotemporal_credit=True,
        direct_ordered_temporal_credit=True,
        directional_multihorizon_future_error_key_manifold=True,
        grouped_selective_ternary_training=True,
        intrinsic_future_error_key_projection=True,
        magnitude_preserving_future_error_key_projection=True,
        optimizer_name=GROUPED_ABSMEAN_ALGORITHM,
        ordered_reversible_branch_credit=True,
        predictive_rls_likelihood=True,
        single_round_reciprocal_credit=True,
        swiglu_local_credit=True,
        temporal_prediction_scale=0.0,
        time_indexed_multihorizon_future_error_key_bands=True,
    )


def create_reciprocal_pc(
    *,
    device: torch.device | str = "cpu",
    seed: int = DEFAULT_SEED,
) -> tuple[ReciprocalPC, ReciprocalPCTrainer]:
    """Construct the accepted model and its strict-local trainer."""

    target = torch.device(device)
    devices = [target.index or 0] if target.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if target.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = ReciprocalPC(model_config()).to(target)
        trainer = ReciprocalPCTrainer(model, training_config())
    _validate_release_identity(model, trainer)
    return model, trainer


def load_reciprocal_pc_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
) -> tuple[ReciprocalPC, ReciprocalPCTrainer, dict[str, Any]]:
    """Load a trusted Phase827-format training checkpoint."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("ReciprocalPC checkpoint must contain a mapping")
    required = {"model_config", "trainer_config", "model", "trainer", "step"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"ReciprocalPC checkpoint is missing: {', '.join(missing)}")
    if payload.get("lifetime_global_backprops") != 0:
        raise ValueError("checkpoint does not preserve zero global backpropagation")
    if payload.get("lifetime_forward_weight_transpose_reads") != 0:
        raise ValueError("checkpoint contains forward-weight transpose reads")
    if payload["model_config"] != asdict(model_config()):
        raise ValueError("checkpoint model configuration differs from Phase827")
    if payload["trainer_config"] != asdict(training_config()):
        raise ValueError("checkpoint trainer configuration differs from Phase827")

    target = torch.device(map_location)
    model = ReciprocalPC(ReciprocalPCConfig(**payload["model_config"])).to(target)
    trainer = ReciprocalPCTrainer(
        model,
        ReciprocalPCTrainingConfig(**payload["trainer_config"]),
    )
    model.load_state_dict(payload["model"])
    trainer.load_state_dict(payload["trainer"])
    if trainer.update_count != int(payload["step"]):
        raise ValueError("checkpoint step and trainer update count differ")
    _validate_release_identity(model, trainer)
    return model, trainer, payload


def _validate_release_identity(
    model: ReciprocalPC,
    trainer: ReciprocalPCTrainer,
) -> None:
    if asdict(model.config) != asdict(model_config()):
        raise ValueError("ReciprocalPC model configuration differs from Phase827")
    if asdict(trainer.config) != asdict(training_config()):
        raise ValueError("ReciprocalPC trainer configuration differs from Phase827")
    if not isinstance(trainer.optimizer, TernaryLocalOptimizer):
        raise TypeError("ReciprocalPC requires the grouped ternary optimizer")
    projector = model.projected_self_bilinear_observation_matrix
    if projector is None or tuple(projector.shape) != (512, 512):
        raise ValueError("ReciprocalPC projected observation matrix differs")
    if model.token_predictor.numel() != 0 or trainer.readout_feedback.numel() != 0:
        raise ValueError("ReciprocalPC tied lexical endpoint differs")
