#!/usr/bin/env python3
"""Run nanochat base training with dense PFC-compatible validation rows."""

from __future__ import annotations

import json
import math
import os
import runpy
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _int_env(name: str) -> int:
    return int(_required_env(name))


RESULTS_PATH = Path(_required_env("PFC_PROGRESS_RESULTS"))
RUN_TAG = _required_env("PFC_PROGRESS_RUN_TAG")
CHECKPOINT_DIR = _required_env("PFC_PROGRESS_CHECKPOINT_DIR")
EVAL_EVERY = _int_env("PFC_PROGRESS_EVAL_EVERY")
TOTAL_STEPS = _int_env("PFC_PROGRESS_TOTAL_STEPS")
TOTAL_BATCH_SIZE = _int_env("PFC_PROGRESS_TOTAL_BATCH_SIZE")
SEQ_LEN = _int_env("PFC_PROGRESS_SEQ_LEN")

_evaluation_index = 0


@torch.no_grad()
def evaluate_bpb_and_log_accuracy(model, batches, steps, token_bytes):
    """Match nanochat BPB evaluation and add exact token accuracy."""
    global _evaluation_index

    device = model.get_device()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    correct = torch.tensor(0, dtype=torch.int64, device=device)
    total = torch.tensor(0, dtype=torch.int64, device=device)
    batch_iter = iter(batches)

    for _ in range(steps):
        x, y = next(batch_iter)
        logits = model(x)
        flat_y = y.reshape(-1)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            flat_y,
            ignore_index=-1,
            reduction="none",
        )
        valid = flat_y >= 0
        safe_y = torch.where(valid, flat_y, torch.zeros_like(flat_y))
        num_bytes = torch.where(
            valid,
            token_bytes[safe_y],
            torch.zeros_like(flat_y, dtype=token_bytes.dtype),
        )
        metric_mask = num_bytes > 0
        total_nats += (loss * metric_mask).sum()
        total_bytes += num_bytes.sum()
        predictions = logits.argmax(dim=-1).reshape(-1)
        correct += ((predictions == flat_y) & metric_mask).sum()
        total += metric_mask.sum()

    if dist.is_initialized() and dist.get_world_size() > 1:
        for value in (total_nats, total_bytes, correct, total):
            dist.all_reduce(value, op=dist.ReduceOp.SUM)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    byte_count = int(total_bytes.item())
    token_count = int(total.item())
    bpb = (
        float(total_nats.item()) / (math.log(2.0) * byte_count)
        if byte_count
        else float("inf")
    )
    accuracy = float(correct.item()) / token_count if token_count else 0.0
    step = _evaluation_index * EVAL_EVERY
    _evaluation_index += 1

    if step > 0 and (not dist.is_initialized() or dist.get_rank() == 0):
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        training_tokens_seen = float(step * TOTAL_BATCH_SIZE)
        row = {
            "accuracy": accuracy,
            "baseline_data": "karpathy/climbmix-400b-shuffle",
            "baseline_model": "nanochat_official_gpt",
            "data_mode": "climbmix",
            "device": str(device),
            "event": "train_progress",
            "global_backprops": float(step),
            "inference_ms_per_batch": elapsed * 1000.0 / max(int(steps), 1),
            "model": "nanochat_gpt_baseline",
            "model_path": str(
                Path(CHECKPOINT_DIR) / f"model_{step:06d}.pt"
            ) if step % 1000 == 0 else "",
            "parameter_slots": float(parameter_count),
            "params": float(parameter_count),
            "run_tag": RUN_TAG,
            "schedule_horizon_steps": float(TOTAL_STEPS),
            "seq_len": float(SEQ_LEN),
            "step": float(step),
            "steps": float(TOTAL_STEPS),
            "tokenizer": "nanochat-bpe",
            "total_batch_size": float(TOTAL_BATCH_SIZE),
            "train_ms_per_batch": 0.0,
            "training_rule": "nanochat-backprop-muon-adamw",
            "training_tokens_per_param": (
                training_tokens_seen / float(parameter_count)
                if parameter_count
                else 0.0
            ),
            "training_tokens_seen": training_tokens_seen,
            "uses_ngram": 0.0,
            "uses_q_table": 0.0,
            "uses_transformer": 1.0,
            "val_bpb": bpb,
            "vocab_size": float(token_bytes.numel()),
        }
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print("PFC_PROGRESS " + json.dumps(row, sort_keys=True), flush=True)

    return bpb


def main() -> None:
    import nanochat.loss_eval

    nanochat.loss_eval.evaluate_bpb = evaluate_bpb_and_log_accuracy
    runpy.run_module("scripts.base_train", run_name="__main__")


if __name__ == "__main__":
    main()
