from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F

try:
    from diagnose_pc_context_use import finish_metrics, load_model, metric_totals
except ModuleNotFoundError:
    from tools.diagnose_pc_context_use import finish_metrics, load_model, metric_totals
from pfc.llm_data import NanochatBestfitBatcher, build_text_tokenizer
from pfc.message_passing_pc_lm import MessagePassingPCLanguageModel


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shift_previous_error(value: Tensor) -> Tensor:
    shifted = torch.zeros_like(value)
    shifted[:, 1:] = value[:, :-1]
    return shifted


def observed_next_tokens(x: Tensor, bos_token_id: int) -> Tensor:
    observed = torch.full_like(x, bos_token_id)
    observed[:, :-1] = x[:, 1:]
    return observed


def causal_error_carry_logits(
    model: MessagePassingPCLanguageModel,
    x: Tensor,
    *,
    scale: float,
) -> tuple[Tensor, list[float]]:
    if scale < 0:
        raise ValueError("carry scale must be nonnegative")
    if model.config.token_feedback_mode != "direct_layer":
        raise ValueError("causal error carry currently requires direct_layer feedback")
    states, _ = model._forward_states(x, with_cache=False, layer_count=0)
    state = states[0]
    observed = observed_next_tokens(x, model.config.bos_token_id)
    correction_rms: list[float] = []
    for layer in range(model.config.num_layers):
        state = model._block(layer, state).prediction
        feedback_source, _ = model._token_feedback(state, observed)
        correction = F.linear(feedback_source, model.feedback_weights[layer])
        shifted = shift_previous_error(correction)
        correction_rms.append(float(shifted.float().square().mean().sqrt().item()))
        state = model._apply_inference_delta(state, shifted, scale)
    return model._decoder_logits(state), correction_rms


def top_only_error_carry_logits(
    model: MessagePassingPCLanguageModel,
    x: Tensor,
    *,
    scale: float,
) -> tuple[Tensor, float]:
    if scale < 0:
        raise ValueError("carry scale must be nonnegative")
    if model.config.token_feedback_mode != "direct_layer":
        raise ValueError("causal error carry currently requires direct_layer feedback")
    states, _ = model._forward_states(x, with_cache=False)
    observed = observed_next_tokens(x, model.config.bos_token_id)
    feedback_source, _ = model._token_feedback(states[-1], observed)
    correction = F.linear(feedback_source, model.feedback_weights[-1])
    shifted = shift_previous_error(correction)
    corrected = model._apply_inference_delta(states[-1], shifted, scale)
    return (
        model._decoder_logits(corrected),
        float(shifted.float().square().mean().sqrt().item()),
    )


def parse_scales(raw: str) -> list[float]:
    scales = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not scales or any(scale < 0 for scale in scales):
        raise ValueError("scales must contain nonnegative comma-separated values")
    return list(dict.fromkeys(scales))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate causal carry of observed previous-token PC error."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("runs/pc_modal_lm/tokenizer_4096"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--scales", default="0,0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.eval_steps) <= 0:
        raise ValueError("batch size, sequence length, and eval steps must be positive")
    scales = parse_scales(args.scales)

    device = torch.device(args.device)
    loaded_model, checkpoint_meta = load_model(args.checkpoint, device)
    if not isinstance(loaded_model, MessagePassingPCLanguageModel):
        raise ValueError("checkpoint must contain a message-PC model")
    model = loaded_model
    tokenizer = build_text_tokenizer("nanochat", args.tokenizer_dir)
    batcher = NanochatBestfitBatcher(
        tokenizer_dir=args.tokenizer_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
        tokenizer=tokenizer,
        split="val",
    )
    byte_lengths = tokenizer.token_byte_lengths(device)
    mode_names = [
        *(f"all_layers_scale_{scale:g}" for scale in scales),
        *(f"top_only_scale_{scale:g}" for scale in scales),
    ]
    totals = {
        name: {"correct": 0.0, "tokens": 0.0, "nats": 0.0, "bytes": 0.0}
        for name in mode_names
    }
    all_layer_correction_rms = {
        scale: torch.zeros(model.config.num_layers, dtype=torch.float64)
        for scale in scales
    }
    top_correction_rms = {scale: 0.0 for scale in scales}
    shift_alignment_correct = 0
    shift_alignment_total = 0
    zero_scale_max_abs_diff = 0.0

    with torch.no_grad():
        for _ in range(args.eval_steps):
            x, y, _ = batcher.next_with_context()
            observed = observed_next_tokens(x, model.config.bos_token_id)
            shift_alignment_correct += int((observed[:, :-1] == y[:, :-1]).sum().item())
            shift_alignment_total += int(y[:, :-1].numel())
            baseline = model(x)
            for scale in scales:
                all_logits, layer_rms = causal_error_carry_logits(
                    model,
                    x,
                    scale=scale,
                )
                top_logits, top_rms = top_only_error_carry_logits(
                    model,
                    x,
                    scale=scale,
                )
                if scale == 0:
                    zero_scale_max_abs_diff = max(
                        zero_scale_max_abs_diff,
                        float((all_logits - baseline).abs().max().item()),
                        float((top_logits - baseline).abs().max().item()),
                    )
                all_layer_correction_rms[scale].add_(torch.tensor(layer_rms))
                top_correction_rms[scale] += top_rms
                for name, logits in (
                    (f"all_layers_scale_{scale:g}", all_logits),
                    (f"top_only_scale_{scale:g}", top_logits),
                ):
                    batch_totals = metric_totals(
                        logits,
                        y,
                        bos_token_id=tokenizer.get_bos_token_id(),
                        byte_lengths=byte_lengths,
                    )
                    for key, value in batch_totals.items():
                        totals[name][key] += value

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_meta.get(
            "step",
            checkpoint_meta.get("steps", 0),
        ),
        "eval_steps": args.eval_steps,
        "evaluated_tokens": totals[mode_names[0]]["tokens"],
        "tokenizer_sha256": file_sha256(args.tokenizer_dir / "tokenizer.pkl"),
        "climbmix_loader": "nanochat-bestfit",
        "shift_alignment_fraction": shift_alignment_correct
        / max(shift_alignment_total, 1),
        "zero_scale_max_abs_diff": zero_scale_max_abs_diff,
        "modes": {name: finish_metrics(values) for name, values in totals.items()},
        "mean_shifted_correction_rms": {
            f"all_layers_scale_{scale:g}": (
                all_layer_correction_rms[scale] / args.eval_steps
            ).tolist()
            for scale in scales
        },
        "mean_top_shifted_correction_rms": {
            f"top_only_scale_{scale:g}": top_correction_rms[scale] / args.eval_steps
            for scale in scales
        },
        "uses_only_observed_input_tokens": True,
        "global_backprops": 0,
        "uses_ngram": 0,
        "uses_q_table": 0,
        "uses_transformer": 0,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
