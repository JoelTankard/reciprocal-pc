from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from pfc.native_pc_lm import NativePCState
from pfc.native_pc_lm import NativeReciprocalPCLanguageModel


PrefillResult = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    NativePCState,
]
PrefillCallable = Callable[[torch.Tensor, NativePCState | None], PrefillResult]


def install_compiled_prefill(
    model: NativeReciprocalPCLanguageModel,
) -> PrefillCallable:
    """Install the frozen Phase381 Inductor boundary on one model instance."""
    eager_prefill: PrefillCallable = model._prefill_states
    compiled_prefill: PrefillCallable = torch.compile(
        eager_prefill,
        backend="inductor",
        mode="default",
        dynamic=False,
        fullgraph=False,
    )
    object.__setattr__(model, "_prefill_states", compiled_prefill)
    return compiled_prefill


def compiler_counter_snapshot() -> dict[str, Any]:
    graph_breaks = dict(torch._dynamo.utils.counters.get("graph_break", {}))
    stats = dict(torch._dynamo.utils.counters.get("stats", {}))
    return {
        "unique_graphs": int(stats.get("unique_graphs", 0)),
        "graph_break_count": sum(int(value) for value in graph_breaks.values()),
        "graph_breaks": {str(key): int(value) for key, value in graph_breaks.items()},
        "stats": {str(key): int(value) for key, value in stats.items()},
    }
