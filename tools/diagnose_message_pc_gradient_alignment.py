from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from pfc.llm_data import NanochatBestfitBatcher, build_text_tokenizer
from pfc.message_passing_pc_lm import MessagePassingPCLanguageModel
from tools.diagnose_pc_context_use import load_model


@dataclass(frozen=True)
class SettlingVariant:
    label: str
    inference_steps: int
    token_feedback_scale: float
    learned_feedback_proposal_scale: float
    inference_lr: float
    state_delta_rms_cap: float
    inference_optimizer: str
    inference_momentum: float


def parse_settling_variant(
    value: str,
    model: MessagePassingPCLanguageModel,
) -> SettlingVariant:
    fields = value.split(":")
    if not 4 <= len(fields) <= 8:
        raise ValueError(
            "settling variant must be "
            "label:steps:token_feedback:learned_feedback"
            "[:inference_lr[:state_cap[:optimizer[:momentum]]]]"
        )
    label = fields[0].strip()
    variant = SettlingVariant(
        label=label,
        inference_steps=int(fields[1]),
        token_feedback_scale=float(fields[2]),
        learned_feedback_proposal_scale=float(fields[3]),
        inference_lr=(
            float(fields[4]) if len(fields) >= 5 else model.config.inference_lr
        ),
        state_delta_rms_cap=(
            float(fields[5])
            if len(fields) >= 6
            else model.config.state_delta_rms_cap
        ),
        inference_optimizer=(
            fields[6] if len(fields) >= 7 else model.config.inference_optimizer
        ),
        inference_momentum=(
            float(fields[7])
            if len(fields) >= 8
            else model.config.inference_momentum
        ),
    )
    if not variant.label:
        raise ValueError("settling variant label must be nonempty")
    if variant.inference_steps <= 0:
        raise ValueError("settling variant steps must be positive")
    if variant.token_feedback_scale <= 0:
        raise ValueError("settling variant token feedback must be positive")
    if variant.learned_feedback_proposal_scale < 0:
        raise ValueError("settling variant learned feedback must be nonnegative")
    if variant.inference_lr <= 0:
        raise ValueError("settling variant inference LR must be positive")
    if variant.state_delta_rms_cap < 0:
        raise ValueError("settling variant state cap must be nonnegative")
    if variant.inference_optimizer not in {
        "gradient",
        "heavy_ball",
        "conjugate_gradient",
    }:
        raise ValueError("settling variant inference optimizer is unknown")
    if not 0 <= variant.inference_momentum < 1:
        raise ValueError("settling variant inference momentum must be in [0, 1)")
    if (
        variant.inference_optimizer == "conjugate_gradient"
        and variant.learned_feedback_proposal_scale > 0
    ):
        raise ValueError("conjugate-gradient variant requires exact energy gradients")
    return variant


def parameter_group(name: str) -> str:
    if name.startswith("decoder_"):
        return "decoder"
    if name == "embedding.weight":
        return "embedding"
    if name.startswith(("query_weights.", "key_weights.")):
        return "attention_route"
    if name.startswith("value_weights."):
        return "attention_value"
    if name.startswith(("self_weights.", "message_weights.")):
        return "recurrent_block"
    if name.startswith("ffn_"):
        return "ffn"
    if name.startswith("state_biases."):
        return "state_bias"
    if name == "feedback_embedding" or name.startswith("feedback_weights."):
        return "feedback"
    if name.startswith("generative_weights."):
        return "generative"
    return "other"


def vector_alignment(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    *,
    candidate_is_update: bool,
) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    for name in sorted(reference.keys() & candidate.keys()):
        oracle = reference[name].detach().float()
        measured = candidate[name].detach().float()
        if candidate_is_update:
            measured = -measured
        group = parameter_group(name)
        for key in (group, "all"):
            values = totals.setdefault(
                key,
                {
                    "dot": 0.0,
                    "reference_sq": 0.0,
                    "candidate_sq": 0.0,
                    "coordinate_agreement": 0.0,
                    "coordinates": 0.0,
                    "tensors": 0.0,
                },
            )
            values["dot"] += float((oracle * measured).sum().item())
            values["reference_sq"] += float(oracle.square().sum().item())
            values["candidate_sq"] += float(measured.square().sum().item())
            nonzero = (oracle != 0) & (measured != 0)
            values["coordinate_agreement"] += float(
                ((oracle * measured > 0) & nonzero).sum().item()
            )
            values["coordinates"] += float(nonzero.sum().item())
            values["tensors"] += 1.0

    result: dict[str, dict[str, float]] = {}
    for group, values in totals.items():
        denominator = math.sqrt(
            values["reference_sq"] * values["candidate_sq"]
        )
        result[group] = {
            "cosine": values["dot"] / max(denominator, 1e-30),
            "reference_rms": math.sqrt(
                values["reference_sq"] / max(values["coordinates"], 1.0)
            ),
            "candidate_rms": math.sqrt(
                values["candidate_sq"] / max(values["coordinates"], 1.0)
            ),
            "candidate_to_reference_norm": math.sqrt(
                values["candidate_sq"] / max(values["reference_sq"], 1e-30)
            ),
            "coordinate_sign_agreement": values["coordinate_agreement"]
            / max(values["coordinates"], 1.0),
            "coordinates": values["coordinates"],
            "tensors": values["tensors"],
        }
    return result


def tensor_alignment(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    *,
    candidate_is_update: bool,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name in sorted(reference.keys() & candidate.keys()):
        oracle = reference[name].detach().float()
        measured = candidate[name].detach().float()
        if candidate_is_update:
            measured = -measured
        oracle_norm = float(oracle.norm().item())
        measured_norm = float(measured.norm().item())
        cosine = float(
            (oracle * measured).sum().item()
            / max(oracle_norm * measured_norm, 1e-30)
        )
        rows.append(
            {
                "name": name,
                "group": parameter_group(name),
                "cosine": cosine,
                "reference_norm": oracle_norm,
                "candidate_norm": measured_norm,
                "candidate_to_reference_norm": measured_norm
                / max(oracle_norm, 1e-30),
            }
        )
    return rows


def capture_local_gradients(
    model: MessagePassingPCLanguageModel,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    decoder_lr: float,
    state_lr: float,
    embed_lr: float,
    current_step: int,
) -> dict[str, torch.Tensor]:
    gradients, _ = capture_local_gradient_probe(
        model,
        x,
        y,
        decoder_lr=decoder_lr,
        state_lr=state_lr,
        embed_lr=embed_lr,
        current_step=current_step,
    )
    return gradients


def capture_local_gradient_probe(
    model: MessagePassingPCLanguageModel,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    decoder_lr: float,
    state_lr: float,
    embed_lr: float,
    current_step: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    captured: dict[str, torch.Tensor] = {}
    settled_states: list[torch.Tensor] = []
    original_settle = model._settle

    def capture(
        _: MessagePassingPCLanguageModel,
        name: str,
        parameter: torch.Tensor,
        gradient: torch.Tensor,
        **__: object,
    ) -> float:
        del parameter
        captured[name] = gradient.detach().clone()
        return 0.0

    def capture_settle(
        _: MessagePassingPCLanguageModel,
        free_states: list[torch.Tensor],
        targets: torch.Tensor,
        **kwargs: object,
    ) -> tuple[list[torch.Tensor], dict[str, float]]:
        states, diagnostics = original_settle(
            free_states,
            targets,
            **kwargs,  # type: ignore[arg-type]
        )
        settled_states.extend(state.detach().clone() for state in states)
        return states, diagnostics

    model._adam_update = MethodType(capture, model)  # type: ignore[method-assign]
    model._muon_update = MethodType(capture, model)  # type: ignore[method-assign]
    model._settle = MethodType(capture_settle, model)  # type: ignore[method-assign]
    model.local_update(
        x,
        y,
        decoder_lr=decoder_lr,
        state_lr=state_lr,
        embed_lr=embed_lr,
        weight_decay=0.0,
        current_step=current_step,
    )
    equilibrium: dict[str, object] = {}
    if settled_states and model.config.inference_mode == "pc_energy":
        original_config = model.config
        model.config = replace(
            model.config,
            learned_feedback_proposal_scale=0.0,
            state_delta_rms_cap=0.0,
        )
        energy_deltas, _, _ = model._pc_energy_state_deltas(settled_states, y)
        model.config = original_config
        layer_rms = [
            float(delta.float().square().mean().sqrt().item())
            for delta in energy_deltas
        ]
        equilibrium = {
            "energy_state_gradient_rms": sum(layer_rms) / len(layer_rms),
            "energy_state_gradient_max_rms": max(layer_rms),
            "energy_state_gradient_rms_by_layer": layer_rms,
        }
    return captured, equilibrium


@torch.no_grad()
def free_phase_block_gradients(
    model: MessagePassingPCLanguageModel,
    x: torch.Tensor,
) -> dict[str, torch.Tensor]:
    free_states, _ = model._forward_states(x, with_cache=False)
    causal_mask = model._causal_mask(x.shape[1], x.device)
    token_count = float(max(x.numel(), 1))
    gradients: dict[str, torch.Tensor] = {}
    for layer in range(model.config.num_layers):
        phase_gradients, _, _, _, _ = model._phase_block_gradients(
            layer,
            free_states[layer],
            free_states[layer + 1],
            route_lower_state=free_states[layer],
            causal_mask=causal_mask,
            token_count=token_count,
            token_ids=x,
            sensory_state=free_states[0],
        )
        gradients.update(
            {
                name: gradient.detach().clone()
                for name, gradient in phase_gradients.items()
            }
        )
    return gradients


@torch.no_grad()
def free_local_ce_block_gradients(
    model: MessagePassingPCLanguageModel,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    layers: tuple[int, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Measure direct free-state token credit through one local block at a time."""

    free_states, _ = model._forward_states(x, with_cache=False)
    causal_mask = model._causal_mask(x.shape[1], x.device)
    token_count = float(max(x.numel(), 1))
    selected_layers = (
        tuple(range(model.config.num_layers)) if layers is None else layers
    )
    gradients: dict[str, torch.Tensor] = {}
    for layer in selected_layers:
        if not 0 <= layer < model.config.num_layers:
            raise ValueError(f"local CE layer outside model depth: {layer}")
        negative_state_gradient, _ = model._decoder_negative_state_gradient(
            free_states[layer + 1],
            y,
        )
        local_target = free_states[layer + 1] + negative_state_gradient
        phase_gradients, _, _, _, _ = model._phase_block_gradients(
            layer,
            free_states[layer],
            local_target,
            route_lower_state=free_states[layer],
            causal_mask=causal_mask,
            token_count=token_count,
            token_ids=x,
            sensory_state=free_states[0],
        )
        gradients.update(
            {
                name: gradient.detach().clone()
                for name, gradient in phase_gradients.items()
            }
        )
    return gradients


@torch.no_grad()
def free_local_ce_embedding_gradient(
    model: MessagePassingPCLanguageModel,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Measure first-block token CE credit to its local input embedding."""

    free_states, _ = model._forward_states(x, with_cache=False)
    first_cache = model._block(
        0,
        free_states[0],
        token_ids=x,
        sensory_state=free_states[0],
    )
    negative_state_gradient, _ = model._decoder_negative_state_gradient(
        free_states[1],
        y,
    )
    output_gradient = -negative_state_gradient
    input_gradient = model._block_input_signal(
        0,
        free_states[0],
        first_cache,
        output_gradient,
    ) * (1.0 - free_states[0].square())
    embedding_gradient = torch.zeros_like(model.embedding.weight)
    embedding_gradient.index_add_(
        0,
        x.reshape(-1),
        input_gradient.reshape(-1, model.config.hidden_dim),
    )
    embedding_gradient.div_(float(max(x.numel(), 1)))
    return {"embedding.weight": embedding_gradient}


def settling_diagnostics(
    model: MessagePassingPCLanguageModel,
) -> dict[str, float]:
    names = (
        "update_diag_message_pc_effective_inference_lr",
        "update_diag_message_pc_phase_delta_rms",
        "update_diag_message_pc_prediction_error_rms",
        "update_diag_message_pc_feedback_rms",
        "update_diag_message_pc_energy_before",
        "update_diag_message_pc_energy_after",
        "update_diag_message_pc_energy_reduction",
        "update_diag_message_pc_inference_accepted_lr",
        "update_diag_message_pc_inference_total_accepted_lr",
        "update_diag_message_pc_inference_backtracking_rejections",
        "update_diag_message_pc_inference_acceleration_restarts",
        "update_diag_message_pc_inference_acceleration_beta",
    )
    return {
        name.removeprefix("update_diag_message_pc_"): float(
            model._last_update_diagnostics[name]
        )
        for name in names
        if name in model._last_update_diagnostics
    }


def oracle_gradients(
    model: MessagePassingPCLanguageModel,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[float, dict[str, torch.Tensor]]:
    parameters = dict(model.named_parameters())
    for parameter in parameters.values():
        parameter.requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, model.config.vocab_size),
        y.reshape(-1),
    )
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters.values()),
        allow_unused=True,
    )
    return float(loss.item()), {
        name: gradient.detach()
        for (name, _), gradient in zip(parameters.items(), gradients)
        if gradient is not None
    }


@torch.no_grad()
def deployment_loss(
    model: MessagePassingPCLanguageModel,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    logits = model(x)
    return float(
        F.cross_entropy(
            logits.reshape(-1, model.config.vocab_size),
            y.reshape(-1),
        ).item()
    )


def first_order_loss_delta_by_group(
    oracle: dict[str, torch.Tensor],
    updates: dict[str, torch.Tensor],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in oracle.keys() & updates.keys():
        group = parameter_group(name)
        contribution = float(
            (oracle[name].float() * updates[name].float()).sum().item()
        )
        result[group] = result.get(group, 0.0) + contribution
        result["all"] = result.get("all", 0.0) + contribution
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare explicit Message-PC local updates with a diagnostic-only "
            "next-token cross-entropy gradient oracle."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("runs/pc_modal_lm/tokenizer_4096"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--decoder-lr-scale", type=float, default=1.0)
    parser.add_argument("--state-lr-scale", type=float, default=1.0)
    parser.add_argument("--embed-lr-scale", type=float, default=14.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--settling-variant",
        action="append",
        default=[],
        metavar=(
            "LABEL:STEPS:TOKEN_FEEDBACK:LEARNED_FEEDBACK"
            "[:INFERENCE_LR[:STATE_CAP[:OPTIMIZER[:MOMENTUM]]]]"
        ),
        help=(
            "repeatable diagnostic-only settling override; compares each local "
            "gradient with the same deployment CE oracle"
        ),
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.seq_len <= 0:
        raise ValueError("batch size and sequence length must be positive")
    if min(
        args.learning_rate,
        args.decoder_lr_scale,
        args.state_lr_scale,
        args.embed_lr_scale,
        args.weight_decay,
    ) < 0:
        raise ValueError("learning rates and weight decay must be nonnegative")

    device = torch.device(args.device)
    loaded, checkpoint_meta = load_model(args.checkpoint, device)
    if not isinstance(loaded, MessagePassingPCLanguageModel):
        raise ValueError("checkpoint must be a Message-PC model")
    checkpoint_step = int(
        checkpoint_meta.get("step", checkpoint_meta.get("steps", 0))
    )
    tokenizer = build_text_tokenizer("nanochat", args.tokenizer_dir)
    batcher = NanochatBestfitBatcher(
        tokenizer_dir=args.tokenizer_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
        tokenizer=tokenizer,
        split="val",
    )
    x, y, _ = batcher.next_with_context()
    decoder_lr = args.learning_rate * args.decoder_lr_scale
    state_lr = args.learning_rate * args.state_lr_scale
    embed_lr = args.learning_rate * args.embed_lr_scale
    current_step = checkpoint_step + 1

    local_model = copy.deepcopy(loaded)
    local_gradients, local_equilibrium_diagnostics = capture_local_gradient_probe(
        local_model,
        x,
        y,
        decoder_lr=decoder_lr,
        state_lr=state_lr,
        embed_lr=embed_lr,
        current_step=current_step,
    )
    local_settling_diagnostics = settling_diagnostics(local_model)
    oracle_model = copy.deepcopy(loaded)
    oracle_loss, oracle = oracle_gradients(oracle_model, x, y)
    del oracle_model

    free_gradients = free_phase_block_gradients(loaded, x)
    free_gradient_max_abs = max(
        (float(gradient.abs().max().item()) for gradient in free_gradients.values()),
        default=0.0,
    )
    free_local_ce_sweep: list[dict[str, object]] = []
    all_layer_local_ce_gradients: dict[str, torch.Tensor] | None = None
    local_ce_variants = [
        ("all_layers", None),
        *[
            (f"layer_{layer}", (layer,))
            for layer in range(loaded.config.num_layers)
        ],
    ]
    for label, layers in local_ce_variants:
        direct_gradients = free_local_ce_block_gradients(
            loaded,
            x,
            y,
            layers=layers,
        )
        if layers is None:
            all_layer_local_ce_gradients = direct_gradients
        free_local_ce_sweep.append(
            {
                "label": label,
                "layers": (
                    list(range(loaded.config.num_layers))
                    if layers is None
                    else list(layers)
                ),
                "alignment": vector_alignment(
                    oracle,
                    direct_gradients,
                    candidate_is_update=False,
                ),
                "tensor_alignment": tensor_alignment(
                    oracle,
                    direct_gradients,
                    candidate_is_update=False,
                ),
            }
        )
    if all_layer_local_ce_gradients is None:
        raise RuntimeError("all-layer local CE diagnostic was not measured")
    direct_embedding_gradient = free_local_ce_embedding_gradient(loaded, x, y)
    free_local_ce_embedding_alignment = vector_alignment(
        oracle,
        direct_embedding_gradient,
        candidate_is_update=False,
    )
    free_local_ce_embedding_tensor_alignment = tensor_alignment(
        oracle,
        direct_embedding_gradient,
        candidate_is_update=False,
    )
    free_local_ce_embedding_blend_sweep: list[dict[str, object]] = []
    for scale in (0.03, 0.1, 0.25, 0.5, 1.0):
        blended = {
            name: gradient.detach().clone()
            for name, gradient in local_gradients.items()
        }
        blended["embedding.weight"].add_(
            direct_embedding_gradient["embedding.weight"],
            alpha=scale,
        )
        free_local_ce_embedding_blend_sweep.append(
            {
                "scale": scale,
                "alignment": vector_alignment(
                    oracle,
                    blended,
                    candidate_is_update=False,
                ),
            }
        )
    free_local_ce_blend_sweep: list[dict[str, object]] = []
    for scale in (0.03, 0.1, 0.25, 0.5, 1.0):
        blended = {
            name: gradient.detach().clone()
            for name, gradient in local_gradients.items()
        }
        for name, direct_gradient in all_layer_local_ce_gradients.items():
            if name in blended:
                blended[name].add_(direct_gradient, alpha=scale)
            else:
                blended[name] = scale * direct_gradient
        free_local_ce_blend_sweep.append(
            {
                "scale": scale,
                "alignment": vector_alignment(
                    oracle,
                    blended,
                    candidate_is_update=False,
                ),
                "tensor_alignment": tensor_alignment(
                    oracle,
                    blended,
                    candidate_is_update=False,
                ),
            }
        )
    settling_sweep: list[dict[str, object]] = []
    seen_variant_labels: set[str] = set()
    for value in args.settling_variant:
        variant = parse_settling_variant(value, loaded)
        if variant.label in seen_variant_labels:
            raise ValueError(f"duplicate settling variant label: {variant.label}")
        seen_variant_labels.add(variant.label)
        variant_model = copy.deepcopy(loaded)
        variant_model.config = replace(
            variant_model.config,
            inference_steps=variant.inference_steps,
            token_feedback_scale=variant.token_feedback_scale,
            learned_feedback_proposal_scale=(
                variant.learned_feedback_proposal_scale
            ),
            inference_lr=variant.inference_lr,
            state_delta_rms_cap=variant.state_delta_rms_cap,
            inference_optimizer=variant.inference_optimizer,
            inference_momentum=variant.inference_momentum,
        )
        variant_gradients, variant_equilibrium = capture_local_gradient_probe(
            variant_model,
            x,
            y,
            decoder_lr=decoder_lr,
            state_lr=state_lr,
            embed_lr=embed_lr,
            current_step=current_step,
        )
        settling_sweep.append(
            {
                "label": variant.label,
                "inference_steps": variant.inference_steps,
                "token_feedback_scale": variant.token_feedback_scale,
                "learned_feedback_proposal_scale": (
                    variant.learned_feedback_proposal_scale
                ),
                "inference_lr": variant.inference_lr,
                "state_delta_rms_cap": variant.state_delta_rms_cap,
                "inference_optimizer": variant.inference_optimizer,
                "inference_momentum": variant.inference_momentum,
                "alignment": vector_alignment(
                    oracle,
                    variant_gradients,
                    candidate_is_update=False,
                ),
                "tensor_alignment": tensor_alignment(
                    oracle,
                    variant_gradients,
                    candidate_is_update=False,
                ),
                "settling": settling_diagnostics(variant_model),
                "equilibrium": variant_equilibrium,
            }
        )
        del variant_model

    update_model = copy.deepcopy(loaded)
    before = {
        name: parameter.detach().clone()
        for name, parameter in update_model.named_parameters()
        if name in oracle
    }
    loss_before = deployment_loss(update_model, x, y)
    update_model.local_update(
        x,
        y,
        decoder_lr=decoder_lr,
        state_lr=state_lr,
        embed_lr=embed_lr,
        weight_decay=args.weight_decay,
        current_step=current_step,
    )
    loss_after = deployment_loss(update_model, x, y)
    updates = {
        name: parameter.detach() - before[name]
        for name, parameter in update_model.named_parameters()
        if name in before
    }

    first_order_delta_by_group = first_order_loss_delta_by_group(oracle, updates)

    optimizer_reset_sweep: list[dict[str, object]] = []
    for reset_scope in ("muon", "context", "all"):
        reset_model = copy.deepcopy(loaded)
        reset_counts = reset_model.reset_local_optimizer_state(reset_scope)
        reset_before = {
            name: parameter.detach().clone()
            for name, parameter in reset_model.named_parameters()
            if name in oracle
        }
        reset_loss_before = deployment_loss(reset_model, x, y)
        reset_model.local_update(
            x,
            y,
            decoder_lr=decoder_lr,
            state_lr=state_lr,
            embed_lr=embed_lr,
            weight_decay=args.weight_decay,
            current_step=current_step,
        )
        reset_loss_after = deployment_loss(reset_model, x, y)
        reset_updates = {
            name: parameter.detach() - reset_before[name]
            for name, parameter in reset_model.named_parameters()
            if name in reset_before
        }
        optimizer_reset_sweep.append(
            {
                "scope": reset_scope,
                "reset_counts": reset_counts,
                "deployment_loss_before": reset_loss_before,
                "deployment_loss_after": reset_loss_after,
                "deployment_loss_delta": reset_loss_after - reset_loss_before,
                "actual_update_descent_alignment": vector_alignment(
                    oracle,
                    reset_updates,
                    candidate_is_update=True,
                ),
                "first_order_loss_delta_by_group": (
                    first_order_loss_delta_by_group(oracle, reset_updates)
                ),
            }
        )
        del reset_model

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "tokens": int(x.numel()),
        "diagnostic_uses_parameter_autograd": True,
        "training_uses_parameter_autograd": False,
        "oracle_loss": oracle_loss,
        "free_phase_block_gradient_max_abs": free_gradient_max_abs,
        "free_local_ce_alignment_sweep": free_local_ce_sweep,
        "free_local_ce_blend_sweep": free_local_ce_blend_sweep,
        "free_local_ce_embedding_alignment": free_local_ce_embedding_alignment,
        "free_local_ce_embedding_tensor_alignment": (
            free_local_ce_embedding_tensor_alignment
        ),
        "free_local_ce_embedding_blend_sweep": (
            free_local_ce_embedding_blend_sweep
        ),
        "deployment_loss_before": loss_before,
        "deployment_loss_after_one_local_update": loss_after,
        "deployment_loss_delta": loss_after - loss_before,
        "raw_local_gradient_alignment": vector_alignment(
            oracle,
            local_gradients,
            candidate_is_update=False,
        ),
        "local_settling_diagnostics": local_settling_diagnostics,
        "local_equilibrium_diagnostics": local_equilibrium_diagnostics,
        "settling_alignment_sweep": settling_sweep,
        "actual_update_descent_alignment": vector_alignment(
            oracle,
            updates,
            candidate_is_update=True,
        ),
        "optimizer_reset_sweep": optimizer_reset_sweep,
        "first_order_loss_delta_by_group": first_order_delta_by_group,
        "raw_tensor_alignment": tensor_alignment(
            oracle,
            local_gradients,
            candidate_is_update=False,
        ),
        "actual_update_tensor_alignment": tensor_alignment(
            oracle,
            updates,
            candidate_is_update=True,
        ),
        "oracle_parameter_tensors": len(oracle),
        "local_parameter_tensors": len(local_gradients),
        "aligned_parameter_tensors": len(oracle.keys() & local_gradients.keys()),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
