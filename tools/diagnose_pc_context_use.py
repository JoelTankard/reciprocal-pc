from __future__ import annotations

import argparse
import json
import math
from dataclasses import MISSING, fields
from pathlib import Path

import torch
import torch.nn.functional as F

from pfc.llm_data import NanochatBestfitBatcher, TokenBatcher, build_text_tokenizer, list_climbmix_shards
from pfc.message_passing_pc_lm import MessagePassingPCConfig, MessagePassingPCLanguageModel
from pfc.pc_modal_lm import (
    PCModalLMConfig,
    SinglePCModalLanguageModel,
    augment_position_context,
)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[SinglePCModalLanguageModel | MessagePassingPCLanguageModel, dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict) or "model" not in payload or "meta" not in payload:
        raise ValueError("checkpoint must contain model and meta")
    meta = payload["meta"]
    if not isinstance(meta, dict):
        raise ValueError("checkpoint metadata must be a dictionary")
    model_name = meta.get("model")
    if model_name == "pc_modal_lm":
        config_type = PCModalLMConfig
        model_type = SinglePCModalLanguageModel
    elif model_name == "message_passing_pc_lm":
        config_type = MessagePassingPCConfig
        model_type = MessagePassingPCLanguageModel
    else:
        raise ValueError("checkpoint is not a supported PC language model")
    config_values: dict[str, object] = {}
    missing: set[str] = set()
    for field in fields(config_type):
        if field.name in meta:
            config_values[field.name] = meta[field.name]
        elif field.default is not MISSING:
            config_values[field.name] = field.default
        elif field.default_factory is not MISSING:
            config_values[field.name] = field.default_factory()
        else:
            missing.add(field.name)
    if missing:
        raise ValueError(f"checkpoint metadata missing config fields: {sorted(missing)}")
    config = config_type(**config_values)
    model = model_type(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    checkpoint_meta = dict(meta)
    for key in ("step", "steps"):
        if key in payload:
            checkpoint_meta[key] = payload[key]
    return model, checkpoint_meta


def free_states_and_logits(
    model: SinglePCModalLanguageModel | MessagePassingPCLanguageModel,
    x: torch.Tensor,
    context_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, MessagePassingPCLanguageModel):
        states = model.forward_states(x)
        return states, model.output_logits(states, x)
    states = model.forward_states(x, context_ids=context_ids)
    states = model.infer_predictive_coding_states(x, states)
    states = model.infer_temporal_pc_states(x, states, context_ids=context_ids)
    states = model.infer_self_settle_states(x, states, context_ids=context_ids)
    return states, model.output_logits(states, x)


def isolated_token_states_and_logits(
    model: SinglePCModalLanguageModel | MessagePassingPCLanguageModel,
    x: torch.Tensor,
    *,
    context_ids: torch.Tensor | None,
    token_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_x = x.reshape(-1)
    flat_context_ids = None if context_ids is None else context_ids.reshape(-1)
    state_parts: list[torch.Tensor] = []
    logit_parts: list[torch.Tensor] = []
    for start in range(0, flat_x.numel(), token_batch_size):
        token_x = flat_x[start : start + token_batch_size].unsqueeze(1)
        token_context_ids = (
            None
            if flat_context_ids is None
            else flat_context_ids[start : start + token_batch_size].unsqueeze(1)
        )
        states, logits = free_states_and_logits(model, token_x, token_context_ids)
        state_parts.append(states[:, 0])
        logit_parts.append(logits[:, 0])
    state_width = (
        model.config.hidden_dim
        if isinstance(model, MessagePassingPCLanguageModel)
        else model.config.node_count
    )
    states = torch.cat(state_parts).view(*x.shape, state_width)
    logits = torch.cat(logit_parts).view(*x.shape, model.config.vocab_size)
    return states, logits


def metric_totals(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    bos_token_id: int,
    byte_lengths: torch.Tensor,
) -> dict[str, float]:
    mask = y != bos_token_id
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1),
        reduction="none",
    )
    lengths = byte_lengths[y.reshape(-1)]
    predictions = logits.argmax(dim=-1)
    return {
        "correct": float((predictions[mask] == y[mask]).sum().item()),
        "tokens": float(mask.sum().item()),
        "nats": float((losses * (lengths > 0)).sum().item()),
        "bytes": float(lengths.sum().item()),
    }


def finish_metrics(totals: dict[str, float]) -> dict[str, float]:
    return {
        "accuracy": totals["correct"] / max(totals["tokens"], 1.0),
        "val_bpb": totals["nats"] / max(math.log(2.0) * totals["bytes"], 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure how much a PC checkpoint uses causal context beyond the current token."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/base_data_climbmix"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("runs/pc_modal_lm/tokenizer_4096"))
    parser.add_argument("--tokenizer", choices=("nanochat", "byte"), default="nanochat")
    parser.add_argument(
        "--climbmix-loader",
        choices=("stream", "nanochat-bestfit"),
        default="nanochat-bestfit",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--token-batch-size", type=int, default=256)
    parser.add_argument("--position-context-count", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.seq_len <= 0 or args.eval_steps <= 0 or args.token_batch_size <= 0:
        raise ValueError("batch size, sequence length, eval steps, and token batch size must be positive")

    device = torch.device(args.device)
    model, checkpoint_meta = load_model(args.checkpoint, device)
    tokenizer = build_text_tokenizer(args.tokenizer, args.tokenizer_dir)
    position_context_count = (
        int(checkpoint_meta.get("position_context_count", 0))
        if args.position_context_count is None
        else args.position_context_count
    )
    if position_context_count < 0:
        raise ValueError("position_context_count must be nonnegative")
    if args.climbmix_loader == "nanochat-bestfit":
        if args.tokenizer != "nanochat":
            raise ValueError("nanochat-bestfit requires the nanochat tokenizer")
        batcher: TokenBatcher | NanochatBestfitBatcher = NanochatBestfitBatcher(
            tokenizer_dir=args.tokenizer_dir,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
            tokenizer=tokenizer,
            split="val",
        )
    else:
        _, val_path = list_climbmix_shards(args.data_dir)
        batcher = TokenBatcher([val_path], args.batch_size, args.seq_len, device, tokenizer)
    byte_lengths = tokenizer.token_byte_lengths(device)

    full_totals = {"correct": 0.0, "tokens": 0.0, "nats": 0.0, "bytes": 0.0}
    isolated_totals = dict(full_totals)
    context_changed = 0.0
    context_rescued = 0.0
    context_harmed = 0.0
    compared_tokens = 0.0
    state_delta_sq = 0.0
    state_sq = 0.0
    state_elements = 0
    state_cosine_sum = 0.0
    state_rows = 0
    logit_delta_sq = 0.0
    logit_elements = 0

    with torch.no_grad():
        for _ in range(args.eval_steps):
            x, y, context_ids = batcher.next_with_context()
            context_ids = augment_position_context(context_ids, position_context_count)
            full_states, full_logits = free_states_and_logits(model, x, context_ids)
            isolated_states, isolated_logits = isolated_token_states_and_logits(
                model,
                x,
                context_ids=context_ids,
                token_batch_size=args.token_batch_size,
            )
            for target, source in (
                (full_totals, metric_totals(full_logits, y, bos_token_id=tokenizer.get_bos_token_id(), byte_lengths=byte_lengths)),
                (
                    isolated_totals,
                    metric_totals(
                        isolated_logits,
                        y,
                        bos_token_id=tokenizer.get_bos_token_id(),
                        byte_lengths=byte_lengths,
                    ),
                ),
            ):
                for key, value in source.items():
                    target[key] += value

            mask = y != tokenizer.get_bos_token_id()
            full_predictions = full_logits.argmax(dim=-1)
            isolated_predictions = isolated_logits.argmax(dim=-1)
            full_correct = full_predictions == y
            isolated_correct = isolated_predictions == y
            context_changed += float((full_predictions[mask] != isolated_predictions[mask]).sum().item())
            context_rescued += float((full_correct[mask] & ~isolated_correct[mask]).sum().item())
            context_harmed += float((~full_correct[mask] & isolated_correct[mask]).sum().item())
            compared_tokens += float(mask.sum().item())

            state_delta_sq += float((full_states - isolated_states).pow(2).sum().item())
            state_sq += float(full_states.pow(2).sum().item())
            state_elements += full_states.numel()
            state_cosine_sum += float(
                F.cosine_similarity(full_states, isolated_states, dim=-1).sum().item()
            )
            state_rows += full_states.shape[0] * full_states.shape[1]
            logit_delta_sq += float((full_logits - isolated_logits).pow(2).sum().item())
            logit_elements += full_logits.numel()

    full = finish_metrics(full_totals)
    isolated = finish_metrics(isolated_totals)
    state_delta_rms = math.sqrt(state_delta_sq / max(state_elements, 1))
    full_state_rms = math.sqrt(state_sq / max(state_elements, 1))
    result: dict[str, float | str] = {
        "checkpoint": str(args.checkpoint),
        "climbmix_loader": args.climbmix_loader,
        "position_context_count": float(position_context_count),
        "eval_steps": float(args.eval_steps),
        "evaluated_tokens": compared_tokens,
        "full_accuracy": full["accuracy"],
        "full_val_bpb": full["val_bpb"],
        "isolated_token_accuracy": isolated["accuracy"],
        "isolated_token_val_bpb": isolated["val_bpb"],
        "context_accuracy_gain": full["accuracy"] - isolated["accuracy"],
        "context_bpb_gain": isolated["val_bpb"] - full["val_bpb"],
        "context_argmax_change_rate": context_changed / max(compared_tokens, 1.0),
        "context_rescue_rate": context_rescued / max(compared_tokens, 1.0),
        "context_harm_rate": context_harmed / max(compared_tokens, 1.0),
        "state_context_delta_rms": state_delta_rms,
        "state_context_relative_rms": state_delta_rms / max(full_state_rms, 1e-12),
        "state_full_isolated_cosine": state_cosine_sum / max(state_rows, 1),
        "logit_context_delta_rms": math.sqrt(logit_delta_sq / max(logit_elements, 1)),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
