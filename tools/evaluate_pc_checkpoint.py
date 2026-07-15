from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

if __package__:
    from tools.diagnose_pc_context_use import (
        finish_metrics,
        free_states_and_logits,
        load_model,
        metric_totals,
    )
else:
    from diagnose_pc_context_use import (
        finish_metrics,
        free_states_and_logits,
        load_model,
        metric_totals,
    )
from pfc.llm_data import NanochatBestfitBatcher, build_text_tokenizer
from pfc.message_passing_pc_lm import MessagePassingPCLanguageModel


def effective_message_readout_layer(
    model: object,
    override: int,
) -> tuple[int | None, str | None]:
    if not isinstance(model, MessagePassingPCLanguageModel):
        return None, None
    if override > 0:
        return override, "cli"
    configured = model.config.readout_layer
    if configured > 0:
        return configured, "checkpoint"
    return model.config.num_layers, "checkpoint-top"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a PC checkpoint on the exact nanochat BOS best-fit validation stream."
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
    parser.add_argument("--timing-steps", type=int, default=100)
    parser.add_argument("--timing-warmup-steps", type=int, default=10)
    parser.add_argument(
        "--message-readout-layer",
        type=int,
        default=0,
        help="0 uses configured/top readout; 1..L selects that message-PC layer.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.eval_steps, args.timing_steps) <= 0:
        raise ValueError("batch size, sequence length, and eval steps must be positive")
    if args.timing_warmup_steps < 0:
        raise ValueError("timing_warmup_steps must be nonnegative")

    device = torch.device(args.device)
    model, checkpoint_meta = load_model(args.checkpoint, device)
    if args.message_readout_layer < 0:
        raise ValueError("message_readout_layer must be nonnegative")
    if not isinstance(model, MessagePassingPCLanguageModel) and args.message_readout_layer > 0:
        raise ValueError("message_readout_layer requires a message-PC checkpoint")
    if (
        isinstance(model, MessagePassingPCLanguageModel)
        and args.message_readout_layer > model.config.num_layers
    ):
        raise ValueError("message_readout_layer exceeds checkpoint depth")
    effective_readout_layer, readout_source = effective_message_readout_layer(
        model,
        args.message_readout_layer,
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
    byte_lengths = tokenizer.token_byte_lengths(device)
    totals = {"correct": 0.0, "tokens": 0.0, "nats": 0.0, "bytes": 0.0}

    def logits_for(
        x: torch.Tensor,
        context_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(model, MessagePassingPCLanguageModel) and args.message_readout_layer > 0:
            states, _ = model._forward_states(x, with_cache=False)
            return model._decoder_logits(states[args.message_readout_layer])
        _, logits = free_states_and_logits(model, x, context_ids)
        return logits

    with torch.no_grad():
        for _ in range(args.eval_steps):
            x, y, context_ids = batcher.next_with_context()
            logits = logits_for(x, context_ids)
            batch_totals = metric_totals(
                logits,
                y,
                bos_token_id=tokenizer.get_bos_token_id(),
                byte_lengths=byte_lengths,
            )
            for key, value in batch_totals.items():
                totals[key] += value

        timing_x, _, timing_context_ids = batcher.next_with_context()
        for _ in range(args.timing_warmup_steps):
            logits_for(timing_x, timing_context_ids)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(args.timing_steps):
            logits_for(timing_x, timing_context_ids)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_ms = 1000.0 * (time.perf_counter() - started) / args.timing_steps

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "model": checkpoint_meta.get("model", "unknown"),
        "checkpoint_step": checkpoint_meta.get("step", checkpoint_meta.get("steps", 0)),
        "message_readout_layer": effective_readout_layer,
        "message_readout_override": args.message_readout_layer,
        "message_readout_source": readout_source,
        "eval_steps": args.eval_steps,
        "evaluated_tokens": totals["tokens"],
        "inference_ms_per_batch": inference_ms,
        "timing_steps": args.timing_steps,
        "tokenizer": "nanochat-bpe",
        "climbmix_loader": "nanochat-bestfit",
        **finish_metrics(totals),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
