from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F

from pfc.llm_data import NanochatBestfitBatcher, build_text_tokenizer
from pfc.message_passing_pc_lm import MessagePassingPCLanguageModel
from tools.diagnose_pc_context_use import load_model


def empty_alignment() -> dict[str, float]:
    return {
        "tokens": 0.0,
        "cosine_sum": 0.0,
        "sign_agreement_sum": 0.0,
        "norm_ratio_sum": 0.0,
    }


def signal_alignment(
    reference: Tensor,
    candidate: Tensor,
    mask: Tensor,
) -> dict[str, float]:
    width = reference.shape[-1]
    reference_rows = reference.reshape(-1, width)[mask.reshape(-1)].float()
    candidate_rows = candidate.reshape(-1, width)[mask.reshape(-1)].float()
    reference_norm = reference_rows.norm(dim=-1)
    candidate_norm = candidate_rows.norm(dim=-1)
    return {
        "tokens": float(reference_rows.shape[0]),
        "cosine_sum": float(
            F.cosine_similarity(
                reference_rows,
                candidate_rows,
                dim=-1,
                eps=1e-8,
            ).sum().item()
        ),
        "sign_agreement_sum": float(
            (
                reference_rows.sign() == candidate_rows.sign()
            ).float().mean(dim=-1).sum().item()
        ),
        "norm_ratio_sum": float(
            (
                candidate_norm / reference_norm.clamp_min(1e-8)
            ).sum().item()
        ),
    }


def add_alignment(
    totals: dict[str, float],
    values: dict[str, float],
) -> None:
    for key, value in values.items():
        totals[key] += value


def finish_alignment(totals: dict[str, float]) -> dict[str, float]:
    count = max(totals["tokens"], 1.0)
    return {
        "evaluated_tokens": totals["tokens"],
        "cosine": totals["cosine_sum"] / count,
        "sign_agreement": totals["sign_agreement_sum"] / count,
        "norm_ratio": totals["norm_ratio_sum"] / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure learned reciprocal Meta-PC credit against exact local VJPs."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("runs/pc_modal_lm/tokenizer_4096"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.eval_steps) <= 0:
        raise ValueError("batch size, sequence length, and eval steps must be positive")

    device = torch.device(args.device)
    loaded, checkpoint_meta = load_model(args.checkpoint, device)
    if not isinstance(loaded, MessagePassingPCLanguageModel):
        raise ValueError("checkpoint must contain a message-PC model")
    model = loaded
    if (
        model.config.inference_mode != "meta_pc"
        or model.config.meta_pc_feedback_mode != "learned_reciprocal"
    ):
        raise ValueError("checkpoint must use learned-reciprocal Meta-PC")

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
    layers = range(1, model.config.num_layers)
    learned_by_layer = {layer: empty_alignment() for layer in layers}
    identity_by_layer = {layer: empty_alignment() for layer in layers}
    learned_total = empty_alignment()
    identity_total = empty_alignment()

    with torch.no_grad():
        for _ in range(args.eval_steps):
            x, y, _ = batcher.next_with_context()
            states, caches = model._forward_states(x, with_cache=True)
            exact_upper, _ = model._meta_pc_top_feedback(states[-1], y)
            mask = y != bos_token_id
            for layer in range(model.config.num_layers - 1, 0, -1):
                exact_lower = model._block_input_signal(
                    layer,
                    states[layer],
                    caches[layer],
                    exact_upper,
                )
                learned_lower = model._meta_pc_feedback_signal(
                    layer,
                    states[layer],
                    caches[layer],
                    exact_upper,
                )
                learned_values = signal_alignment(
                    exact_lower,
                    learned_lower,
                    mask,
                )
                identity_values = signal_alignment(
                    exact_lower,
                    exact_upper,
                    mask,
                )
                add_alignment(learned_by_layer[layer], learned_values)
                add_alignment(identity_by_layer[layer], identity_values)
                add_alignment(learned_total, learned_values)
                add_alignment(identity_total, identity_values)
                exact_upper = exact_lower

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_meta.get(
            "step",
            checkpoint_meta.get("steps", 0),
        ),
        "eval_steps": args.eval_steps,
        "feedback_mode": model.config.meta_pc_feedback_mode,
        "output_feedback_mode": model.config.meta_pc_output_feedback_mode,
        "global_backprops": 0,
        "uses_forward_weight_transpose_for_candidate": 0,
        "learned_alignment": finish_alignment(learned_total),
        "identity_alignment": finish_alignment(identity_total),
        "learned_alignment_by_layer": {
            str(layer): finish_alignment(learned_by_layer[layer])
            for layer in layers
        },
        "identity_alignment_by_layer": {
            str(layer): finish_alignment(identity_by_layer[layer])
            for layer in layers
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
