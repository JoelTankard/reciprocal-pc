from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CURVE_DIR = REPO_ROOT / "data" / "curves"

GPT_CURVE = CURVE_DIR / "gpt-50k.jsonl"
BLUE_CURVE = CURVE_DIR / "blue-exact-credit-pc-50k.jsonl"
RECIPROCAL_PC_CURVE = CURVE_DIR / "reciprocalpc-50k.jsonl"


def load_curve(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
