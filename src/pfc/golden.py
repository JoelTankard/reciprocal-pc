"""Golden additive-RLS architecture and model-only interpolated inference loading."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .native_pc_lm import NativeReciprocalPCLanguageModel
from .reciprocal_pc import DEFAULT_SEED, model_config
from .rls_branch import AdditiveRLSBranch


Golden = AdditiveRLSBranch
INFERENCE_SCHEMA = "golden-drift-inference-only.v1"
GOLDEN_CHECKPOINT_URL = (
    "https://github.com/JoelTankard/reciprocal-pc/releases/download/"
    "golden-20261022/golden-inference.pt"
)
GOLDEN_CHECKPOINT_SHA256 = "31932202917638a4d4f5464ecd340f296de1949bf58fa5dcbf89fdbfcbbe2e85"


def create_golden(
    *, device: torch.device | str = "cpu", seed: int = DEFAULT_SEED
) -> Golden:
    """Construct the untrained Golden architecture without creating a trainer.

    Use ``load_golden_checkpoint`` for the promoted measured weights. Forward
    and prefill return dictionaries with full-sequence logits and functional
    RLSBranchState; step consumes one observed token per batch lane.
    """
    target = torch.device(device)
    devices = [target.index or 0] if target.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if target.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        native = NativeReciprocalPCLanguageModel(model_config()).to(target)
        return Golden(native)


def _refresh_native_caches(model: NativeReciprocalPCLanguageModel) -> None:
    # Same refresh sequence as the frozen native model-only export path. Derive
    # caches from restored source tensors without running trainer quantization.
    model.refresh_ternary_vector_recirculation_active()
    model.refresh_kronecker_projection_cache()
    model.refresh_metric_shared_endpoint_cache()
    model.refresh_lifted_antisymmetric_reciprocal_cache()


def load_golden_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
) -> tuple[Golden, dict[str, Any]]:
    """Load an inference-only Golden export on CPU or CUDA, never a trainer.

    This accepts the exact exported payload, not a native training checkpoint.
    The returned model is in eval mode. Its interpolated weights are not a
    resumable trajectory, and ``golden_local_update`` rejects it explicitly.
    Download/hash verification is the caller's responsibility.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"schema", "model_config", "model", "branch", "transformation"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Golden inference checkpoint must contain exactly schema/model_config/model/branch/transformation")
    if payload["schema"] != INFERENCE_SCHEMA:
        raise ValueError("unsupported Golden inference checkpoint schema")
    if payload["model_config"] != asdict(model_config()):
        raise ValueError("Golden checkpoint model configuration differs from the accepted native architecture")
    transformation = payload["transformation"]
    if not isinstance(transformation, dict) or transformation.get("inference_only") is not True:
        raise ValueError("Golden transformation must declare inference_only=True")
    if not isinstance(payload["model"], dict) or not isinstance(payload["branch"], dict):
        raise ValueError("Golden model and branch states must be mappings")
    model = create_golden()
    expected = model.base_model.state_dict()
    if set(payload["model"]) != set(expected):
        raise ValueError("Golden native checkpoint tensor names differ")
    for name, destination in expected.items():
        source = payload["model"][name]
        if not isinstance(source, torch.Tensor) or source.shape != destination.shape:
            raise ValueError(f"invalid native checkpoint tensor shape: {name}")
        if source.dtype != destination.dtype or source.layout != torch.strided:
            raise ValueError(f"invalid native checkpoint tensor dtype/layout: {name}")
        if source.is_floating_point() and not bool(torch.isfinite(source).all().item()):
            raise ValueError(f"nonfinite native checkpoint tensor: {name}")
    model.base_model.load_state_dict(payload["model"], strict=True)
    model.load_branch_state_dict(payload["branch"])
    model.to(torch.device(map_location))
    _refresh_native_caches(model.base_model)
    model._golden_inference_only = True
    return model.eval(), payload
