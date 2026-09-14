from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from pfc import load_reciprocal_pc_checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and smoke-test a Phase827 ReciprocalPC checkpoint."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()

    actual_sha256 = sha256(args.checkpoint)
    if args.expected_sha256 and actual_sha256 != args.expected_sha256:
        raise ValueError(
            f"checkpoint SHA-256 differs: {actual_sha256} != {args.expected_sha256}"
        )

    model, trainer, payload = load_reciprocal_pc_checkpoint(
        args.checkpoint,
        map_location=args.device,
    )
    model.eval()
    tokens = torch.tensor([[32759, 1, 2, 3]], device=args.device)
    with torch.no_grad():
        logits = model(tokens)
    if logits.shape != (1, 4, 32768) or not torch.isfinite(logits).all():
        raise ValueError("checkpoint inference smoke test failed")

    print(
        {
            "checkpoint_sha256": actual_sha256,
            "device": args.device,
            "step": payload.get("step"),
            "update_count": trainer.update_count,
            "global_backprops": payload.get("lifetime_global_backprops"),
            "forward_weight_transpose_reads": payload.get(
                "lifetime_forward_weight_transpose_reads"
            ),
            "logits_shape": list(logits.shape),
            "finite": True,
        }
    )


if __name__ == "__main__":
    main()
