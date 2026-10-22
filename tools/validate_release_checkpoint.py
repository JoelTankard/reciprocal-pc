from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from pfc import (
    GOLDEN_CHECKPOINT_SHA256,
    AdditiveRLSBranch,
    create_golden,
    create_reciprocal_pc,
    golden_local_update,
    load_golden_checkpoint,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_logits(args) -> dict:
    actual_sha256 = sha256(args.checkpoint)
    if actual_sha256 != args.expected_sha256:
        raise ValueError(f"checkpoint SHA-256 differs: {actual_sha256} != {args.expected_sha256}")
    model, payload = load_golden_checkpoint(args.checkpoint, map_location=args.device)
    # Independently reconstruct the exported native/branch tensors rather than
    # treating two identical loader calls as proof that all weights were used.
    restored = create_golden(device=args.device)
    restored.base_model.load_state_dict(payload["model"], strict=True)
    restored.load_branch_state_dict(payload["branch"])
    restored.base_model.refresh_ternary_vector_recirculation_active()
    restored.base_model.refresh_kronecker_projection_cache()
    restored.base_model.refresh_metric_shared_endpoint_cache()
    restored.base_model.refresh_lifted_antisymmetric_reciprocal_cache()
    restored.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 23_930_435:
        raise ValueError("Golden parameter count differs")
    checks = []
    with torch.inference_mode():
        for length in args.lengths:
            tokens = torch.arange(length, dtype=torch.long, device=args.device)[None] % 32768
            tokens[0, 0] = 32759
            expected = model(tokens)["logits"]
            actual = restored(tokens)["logits"]
            if expected.shape != (1, length, 32768) or not bool(torch.isfinite(expected).all()):
                raise ValueError(f"nonfinite or malformed Golden logits at length {length}")
            if not torch.equal(expected, actual):
                raise ValueError(f"checkpoint reload changed Golden logits at length {length}")
            checks.append({"length": length, "logits_shape": list(actual.shape), "finite": True, "reload_logits_exact": True})
            del expected, actual
    return {
        "checkpoint_sha256": actual_sha256,
        "schema": payload["schema"],
        "inference_only": True,
        "parameters": parameter_count,
        "gain": float(model.gain.detach()),
        "logits": checks,
    }


def training_integration(args) -> dict:
    # Never attach a trainer to the interpolated release. This is a disposable,
    # fresh unaveraged native trajectory with the measured branch integration.
    native, trainer = create_reciprocal_pc(device=args.device)
    branch = AdditiveRLSBranch(native)
    with torch.no_grad():
        branch.gain.fill_(0.057484474033117294)
    optimizer = torch.optim.AdamW([branch.gain], lr=0.002, weight_decay=0.0)
    original = native._prefill_states
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    native._prefill_states = counted
    tokens = torch.arange(args.training_length, dtype=torch.long, device=args.device)[None] % 32768
    tokens[0, 0] = 32759
    targets = (tokens + 1) % 32768
    result = golden_local_update(branch, trainer, optimizer, tokens, targets)
    if calls != 1 or native._prefill_states is not counted or trainer.update_count != 1:
        raise ValueError("shared-prefill native execution/update count differs")
    metrics = result["native"]
    if int(metrics["global_backprops"]) != 0 or int(metrics["forward_weight_transpose_reads"]) != 0:
        raise ValueError("shared-prefill update lost strict-local credit")
    return {
        "fresh_unaveraged_weights": True,
        "length": args.training_length,
        "actual_native_prefill_executions": calls,
        "native_updates": trainer.update_count,
        "global_backprops": int(metrics["global_backprops"]),
        "forward_weight_transpose_reads": int(metrics["forward_weight_transpose_reads"]),
        "gain_loss": float(result["gain"]["loss"]),
        "gain_gradient": float(result["gain"]["gradient"]),
        "updated_gain": float(branch.gain.detach()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check actual Golden inference reload at short/full context and optional fresh shared-prefill training integration.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-sha256", default=GOLDEN_CHECKPOINT_SHA256)
    parser.add_argument("--lengths", nargs="+", type=int, default=[64, 2048])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--training-update", action="store_true")
    parser.add_argument("--training-length", type=int, choices=[128, 2048], default=128)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    result = {"device": args.device, "inference": checkpoint_logits(args)}
    gc.collect()
    if args.training_update:
        result["shared_prefill_training"] = training_integration(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
