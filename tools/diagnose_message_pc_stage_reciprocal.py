from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F

from pfc.llm_data import NanochatBestfitBatcher, build_text_tokenizer
from pfc.message_passing_pc_lm import MessagePassingPCLanguageModel
from tools.diagnose_message_pc_reciprocal_feedback import (
    add_alignment,
    empty_alignment,
    finish_alignment,
    signal_alignment,
)
from tools.diagnose_pc_context_use import load_model


_STAGE_FEEDBACK_PREFIXES = (
    "feedback_query_weights.",
    "feedback_key_weights.",
    "feedback_value_weights.",
    "feedback_message_weights.",
    "feedback_ffn_up_weights.",
    "feedback_ffn_down_weights.",
)


@torch.no_grad()
def build_stage_model(
    source: MessagePassingPCLanguageModel,
    *,
    learning_rate: float,
) -> MessagePassingPCLanguageModel:
    """Clone deployment and output-feedback weights into a stage mirror model."""

    config = replace(
        source.config,
        meta_pc_feedback_mode="stage_reciprocal",
        meta_pc_reciprocal_learning_rate=learning_rate,
        meta_pc_state_gated_reciprocal=False,
        meta_pc_reciprocal_update_before_settle=True,
    )
    model = MessagePassingPCLanguageModel(config).to(source.get_device())
    source_parameters = dict(source.named_parameters())
    for name, target in model.named_parameters():
        if name.startswith(_STAGE_FEEDBACK_PREFIXES):
            continue
        candidate = source_parameters.get(name)
        if candidate is not None and candidate.shape == target.shape:
            target.copy_(candidate)
    model.eval()
    return model


def credit_mask(y: Tensor, *, bos_token_id: int, skip_first: int) -> Tensor:
    mask = y != bos_token_id
    if skip_first > 0:
        mask = mask.clone()
        mask[:, :skip_first] = False
    return mask


def matrix_alignment(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    reference = reference.float()
    candidate = candidate.float()
    return {
        "cosine": float(
            F.cosine_similarity(
                reference.reshape(1, -1),
                candidate.reshape(1, -1),
                dim=-1,
                eps=1e-12,
            ).item()
        ),
        "norm_ratio": float(
            (candidate.norm() / reference.norm().clamp_min(1e-12)).item()
        ),
        "relative_error": float(
            (
                (candidate - reference).norm()
                / reference.norm().clamp_min(1e-12)
            ).item()
        ),
    }


def stage_mirror_geometry(
    model: MessagePassingPCLanguageModel,
) -> dict[str, object]:
    forward_groups = {
        "query": model.query_weights,
        "key": model.key_weights,
        "value": model.value_weights,
        "message": model.message_weights,
        "ffn_up": model.ffn_up_weights,
        "ffn_down": model.ffn_down_weights,
    }
    mirror_groups = {
        "query": model.feedback_query_weights,
        "key": model.feedback_key_weights,
        "value": model.feedback_value_weights,
        "message": model.feedback_message_weights,
        "ffn_up": model.feedback_ffn_up_weights,
        "ffn_down": model.feedback_ffn_down_weights,
    }
    result: dict[str, object] = {}
    for layer in range(model.config.num_layers):
        result[str(layer)] = {
            name: matrix_alignment(
                forward_groups[name][layer].t(),
                mirrors[layer],
            )
            for name, mirrors in mirror_groups.items()
            if mirrors
        }
    return result


@torch.no_grad()
def evaluate_feedback(
    model: MessagePassingPCLanguageModel,
    batches: list[tuple[Tensor, Tensor]],
    *,
    bos_token_id: int,
    skip_first: int,
) -> dict[str, object]:
    layers = list(range(1, model.config.num_layers))
    lower_layers = list(range(model.config.num_layers - 1))
    adjacent = empty_alignment()
    composed = empty_alignment()
    adjacent_by_layer = {layer: empty_alignment() for layer in layers}
    composed_by_layer = {layer: empty_alignment() for layer in lower_layers}

    for x, y in batches:
        states, caches = model._forward_states(x, with_cache=True)
        top_feedback, _ = model._meta_pc_top_feedback(states[-1], y)
        mask = credit_mask(
            y,
            bos_token_id=bos_token_id,
            skip_first=skip_first,
        )
        exact_upper = top_feedback
        exact_by_lower_layer: dict[int, Tensor] = {}
        for layer in reversed(layers):
            exact_lower = model._block_input_signal(
                layer,
                states[layer],
                caches[layer],
                exact_upper,
            )
            candidate_lower = model._meta_pc_feedback_signal(
                layer,
                states[layer],
                caches[layer],
                exact_upper,
            )
            values = signal_alignment(exact_lower, candidate_lower, mask)
            add_alignment(adjacent, values)
            add_alignment(adjacent_by_layer[layer], values)
            exact_by_lower_layer[layer - 1] = exact_lower
            exact_upper = exact_lower

        candidate_upper = top_feedback
        for layer in reversed(layers):
            candidate_lower = model._meta_pc_feedback_signal(
                layer,
                states[layer],
                caches[layer],
                candidate_upper,
            )
            lower_layer = layer - 1
            values = signal_alignment(
                exact_by_lower_layer[lower_layer],
                candidate_lower,
                mask,
            )
            add_alignment(composed, values)
            add_alignment(composed_by_layer[lower_layer], values)
            candidate_upper = candidate_lower

    return {
        "adjacent": finish_alignment(adjacent),
        "composed": finish_alignment(composed),
        "adjacent_by_block": {
            str(layer): finish_alignment(adjacent_by_layer[layer])
            for layer in layers
        },
        "composed_by_lower_layer": {
            str(layer): finish_alignment(composed_by_layer[layer])
            for layer in lower_layers
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit local stage mirrors without backprop and measure held-out "
            "Meta-PC credit against exact block VJPs."
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
    parser.add_argument("--calibration-steps", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.25],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(
        args.batch_size,
        args.seq_len,
        args.calibration_steps,
        args.eval_steps,
    ) <= 0:
        raise ValueError("batch, sequence, and step counts must be positive")
    if args.skip_first < 0 or args.skip_first >= args.seq_len:
        raise ValueError("skip-first must be in [0, seq-len)")
    if any(not 0 < rate <= 1 for rate in args.learning_rates):
        raise ValueError("learning rates must be in (0, 1]")

    device = torch.device(args.device)
    loaded, checkpoint_meta = load_model(args.checkpoint, device)
    if not isinstance(loaded, MessagePassingPCLanguageModel):
        raise ValueError("checkpoint must contain a message-PC model")
    if loaded.config.inference_mode != "meta_pc":
        raise ValueError("checkpoint must use Meta-PC")
    if loaded.config.block_state_mode != "nanochat_residual":
        raise ValueError("stage feedback requires nanochat-residual blocks")
    if loaded.config.causal_encoder_decoder:
        raise ValueError("stage feedback does not support cross-attention")

    tokenizer = build_text_tokenizer("nanochat", args.tokenizer_dir)
    bos_token_id = tokenizer.get_bos_token_id()
    batcher = NanochatBestfitBatcher(
        tokenizer_dir=args.tokenizer_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
        tokenizer=tokenizer,
        split="val",
    )
    calibration_batches = [
        batcher.next_with_context()[0]
        for _ in range(args.calibration_steps)
    ]
    eval_batches = [
        batcher.next_with_context()[:2]
        for _ in range(args.eval_steps)
    ]

    configured = evaluate_feedback(
        loaded,
        eval_batches,
        bos_token_id=bos_token_id,
        skip_first=args.skip_first,
    )
    stage_results: dict[str, object] = {}
    for rate in args.learning_rates:
        stage = build_stage_model(loaded, learning_rate=rate)
        calibration_diagnostics: dict[str, float] = {}
        for calibration_step, x in enumerate(calibration_batches, start=1):
            states, _ = stage._forward_states(x, with_cache=False)
            calibration_diagnostics = stage._update_stage_reciprocal_feedback(
                states,
                x,
                current_step=calibration_step,
            )
        stage_results[f"{rate:.6g}"] = {
            "learning_rate": rate,
            "calibration_final": calibration_diagnostics,
            "mirror_geometry": stage_mirror_geometry(stage),
            "credit": evaluate_feedback(
                stage,
                eval_batches,
                bos_token_id=bos_token_id,
                skip_first=args.skip_first,
            ),
        }
        del stage

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_meta.get(
            "step",
            checkpoint_meta.get("steps", 0),
        ),
        "checkpoint_feedback_mode": loaded.config.meta_pc_feedback_mode,
        "output_feedback_mode": loaded.config.meta_pc_output_feedback_mode,
        "calibration_steps": args.calibration_steps,
        "eval_steps": args.eval_steps,
        "skip_first_attention_sink_positions": args.skip_first,
        "global_backprops": 0,
        "candidate_uses_forward_weight_transpose": 0,
        "mirror_estimator": "local_orthogonal_hadamard_linear_stage_response",
        "configured_credit": configured,
        "stage_reciprocal": stage_results,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
