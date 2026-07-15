#!/usr/bin/env python3
"""Fill final training timing into nanochat progress JSONL rows atomically."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    args = parser.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    total_seconds = float(meta["loop_state"]["total_training_time"])
    final_step = int(meta["step"])
    train_ms_per_batch = total_seconds * 1000.0 / final_step

    rows = []
    for line in args.results.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["train_ms_per_batch"] = train_ms_per_batch
        rows.append(json.dumps(row, sort_keys=True))

    temporary = args.results.with_suffix(args.results.suffix + ".tmp")
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    os.replace(temporary, args.results)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "total_training_time": total_seconds,
                "train_ms_per_batch": train_ms_per_batch,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
