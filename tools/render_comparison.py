#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CURVES = (
    ("GPT-50k", ROOT / "data/curves/gpt-50k.jsonl", "#2389d7"),
    (
        "Blue exact-credit PC",
        ROOT / "data/curves/blue-exact-credit-pc-50k.jsonl",
        "#f2a900",
    ),
    ("ReciprocalPC (no backprop)", ROOT / "data/curves/reciprocalpc-50k.jsonl", "#22a65a"),
)
OUTPUT = ROOT / "data/graphs/three-models-50k.png"


def read_rows(path: Path) -> list[dict[str, float]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    figure, axes = plt.subplots(2, 1, figsize=(15, 9.5), sharex=True)
    figure.patch.set_facecolor("white")
    for label, path, color in CURVES:
        rows = read_rows(path)
        steps = [row["step"] for row in rows]
        axes[0].plot(
            steps,
            [row["val_bpb"] for row in rows],
            color=color,
            label=label,
            linewidth=2.2,
        )
        axes[1].plot(
            steps,
            [row["accuracy"] for row in rows],
            color=color,
            label=label,
            linewidth=2.2,
        )

    axes[0].set_title("ReciprocalPC vs Exact-Credit Oracle and GPT", loc="left", fontsize=17, weight="bold")
    axes[0].set_ylabel("validation BPB (lower is better)")
    axes[1].set_ylabel("accuracy (higher is better)")
    axes[1].set_xlabel("step")
    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(True, alpha=0.24)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=120, facecolor="white", transparent=False)
    print(OUTPUT)


if __name__ == "__main__":
    main()
