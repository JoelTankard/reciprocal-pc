from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
from torch import Tensor

from pfc.native_pc_training import CAPTURE_HOST_METRIC_FIELDS
from pfc.native_pc_training import _scoped_deterministic_index_add_
from pfc.native_pc_training import NativePCTrainer
from pfc.ternary_local_optimizer import GROUPED_ABSMEAN_ALGORITHM
from pfc.ternary_local_optimizer import TernaryLocalOptimizer


HELD_QUERY_CAPTURE_ROUTE = "held_query"
DORMANT_FIXED_HADAMARD_BASE_LIKELIHOOD_CAPTURE_ROUTE = (
    "dormant_fixed_hadamard_base_likelihood"
)
_DORMANT_FIXED_HADAMARD_RETIRED_FLAGS = (
    "predictive_rls_likelihood",
    "delayed_consequence_selective_credit",
    "charged_held_query_displacement_projection",
    "intrinsic_future_error_key_projection",
    "time_indexed_multihorizon_future_error_key_bands",
    "magnitude_preserving_future_error_key_projection",
    "directional_multihorizon_future_error_key_manifold",
)


def _is_dormant_fixed_hadamard_base_likelihood_capture(
    trainer: NativePCTrainer,
) -> bool:
    config = trainer.config
    model = trainer.model
    key = model.predictive_future_error_key_projection
    basis = getattr(trainer, "_predictive_rls_basis", None)
    codebook = getattr(trainer, "_predictive_rls_codebook", None)
    previous_displacements = getattr(
        trainer,
        "_charged_held_query_previous_displacements",
        None,
    )
    displacement_initialized = getattr(
        trainer,
        "_charged_held_query_displacement_initialized",
        None,
    )
    return bool(
        not config.exact_vjp_credit
        and model.config.fixed_ternary_hadamard_lexical_predictor
        and model.config.predictive_future_error_key_rank == 32
        and config.grouped_selective_ternary_training
        and config.optimizer_name == GROUPED_ABSMEAN_ALGORITHM
        and all(
            not bool(getattr(config, name))
            for name in _DORMANT_FIXED_HADAMARD_RETIRED_FLAGS
        )
        and isinstance(key, Tensor)
        and key.shape == (model.config.width, 32)
        and isinstance(basis, Tensor)
        and basis.numel() == 0
        and isinstance(codebook, Tensor)
        and codebook.numel() == 0
        and isinstance(previous_displacements, dict)
        and not previous_displacements
        and isinstance(displacement_initialized, Tensor)
        and displacement_initialized.ndim == 0
        and not bool(displacement_initialized.item())
    )


def require_exact_split_capture_route(trainer: NativePCTrainer) -> str:
    if trainer.config.charged_held_query_displacement_projection:
        if int(trainer._charged_held_query_displacement_initialized.item()) != 1:
            raise ValueError(
                "whole-update capture requires initialized held-query state"
            )
        return HELD_QUERY_CAPTURE_ROUTE
    if _is_dormant_fixed_hadamard_base_likelihood_capture(trainer):
        return DORMANT_FIXED_HADAMARD_BASE_LIKELIHOOD_CAPTURE_ROUTE
    raise ValueError(
        "whole-update capture requires held-query credit or the exact dormant "
        "fixed-Hadamard base-likelihood route"
    )


def _prepare_capture_control_flow(trainer: NativePCTrainer) -> None:
    require_exact_split_capture_route(trainer)
    # This legacy flag also selects graph-safe nonfinite accounting.
    trainer._static_capture_held_query_initialized = True
    if trainer.config.predictive_rls_likelihood:
        trainer._defer_predictive_rls_geometry_refresh = True
    trainer._defer_host_metric_scalar_extraction = True


def _clear_capture_control_flow(trainer: NativePCTrainer) -> None:
    if hasattr(trainer, "_static_capture_held_query_initialized"):
        del trainer._static_capture_held_query_initialized
    if hasattr(trainer, "_defer_predictive_rls_geometry_refresh"):
        del trainer._defer_predictive_rls_geometry_refresh
    if hasattr(trainer, "_defer_host_metric_scalar_extraction"):
        del trainer._defer_host_metric_scalar_extraction


@torch.no_grad()
def _refresh_predictive_rls_geometry_after_replay(
    trainer: NativePCTrainer,
) -> bool:
    if not trainer.config.predictive_rls_likelihood:
        return False
    trainer.refresh_predictive_rls_geometry()
    return True


def finalize_graph_metrics(
    metrics: dict[str, Tensor | int],
) -> dict[str, Tensor | int]:
    finalized = dict(metrics)
    for name in CAPTURE_HOST_METRIC_FIELDS:
        value = metrics.get(name)
        if not isinstance(value, Tensor) or value.ndim != 0:
            raise RuntimeError(f"graph metric {name} is not a scalar tensor")
        finalized[name] = int(value.item())
    return finalized


def deferred_graph_metrics(
    metrics: dict[str, Tensor | int],
) -> dict[str, Tensor | int]:
    """Expose live graph metrics without synchronizing scalar counters."""

    deferred = dict(metrics)
    for name in CAPTURE_HOST_METRIC_FIELDS:
        value = metrics.get(name)
        if not isinstance(value, Tensor) or value.ndim != 0:
            raise RuntimeError(f"graph metric {name} is not a scalar tensor")
    return deferred


def _prepare_adamw_capture(
    trainer: NativePCTrainer,
    device: torch.device,
) -> list[bool]:
    optimizer = trainer.optimizer
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("whole-update capture requires AdamW")
    prior = [bool(group.get("capturable", False)) for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        if group.get("fused") not in (None, False):
            raise ValueError("whole-update capture forbids fused AdamW")
        group["capturable"] = True
    for state in optimizer.state.values():
        step = state.get("step")
        if isinstance(step, Tensor) and step.device != device:
            state["step"] = step.to(device=device)
    return prior


def _restore_adamw_metadata(
    trainer: NativePCTrainer,
    capturable: list[bool],
) -> None:
    if len(capturable) != len(trainer.optimizer.param_groups):
        raise RuntimeError("AdamW group count changed during capture")
    for group, value in zip(trainer.optimizer.param_groups, capturable):
        group["capturable"] = value


def _require_eager_split_optimizer(
    trainer: NativePCTrainer,
) -> torch.optim.Optimizer:
    optimizer = trainer.optimizer
    if isinstance(optimizer, torch.optim.AdamW):
        for group in optimizer.param_groups:
            if bool(group.get("capturable", False)):
                raise ValueError(
                    "split-update capture requires noncapturable AdamW"
                )
            if group.get("fused") not in (None, False):
                raise ValueError("split-update capture forbids fused AdamW")
    elif isinstance(optimizer, TernaryLocalOptimizer):
        algorithms = {str(group["algorithm"]) for group in optimizer.param_groups}
        unsupported = algorithms - {GROUPED_ABSMEAN_ALGORITHM, "adamw"}
        if unsupported:
            raise ValueError(
                "split-update capture does not support ternary routes: "
                + ", ".join(sorted(unsupported))
            )
        for state in optimizer.state.values():
            for name, value in state.items():
                if not isinstance(value, (Tensor, bool, int, float, str, type(None))):
                    raise TypeError(
                        "split-update ternary optimizer state is not flat: " + name
                    )
    else:
        raise TypeError(
            "split-update capture requires AdamW or grouped ternary optimizer"
        )
    return optimizer


def _require_eager_adamw(trainer: NativePCTrainer) -> torch.optim.AdamW:
    """Retain the Phase368 AdamW-only validation interface."""

    optimizer = trainer.optimizer
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("split-update capture requires AdamW")
    _require_eager_split_optimizer(trainer)
    return optimizer


@dataclass(frozen=True)
class _OptimizerBoundaryMetadata:
    state_layout: tuple[tuple[int, tuple[str, ...]], ...]
    scalar_state_values: tuple[
        tuple[dict[str, Any], str, bool | int | float | str | None], ...
    ]
    cpu_rng_state: Tensor
    cuda_rng_state: Tensor | None


def _cpu_clone_capture_state(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {name: _cpu_clone_capture_state(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_clone_capture_state(item) for item in value)
    if isinstance(value, list):
        return [_cpu_clone_capture_state(item) for item in value]
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    raise TypeError(f"unsupported capture-state value: {type(value).__name__}")


@torch.no_grad()
def _restore_capture_state_(live: Any, saved: Any, *, path: str) -> None:
    if isinstance(saved, Tensor):
        if not isinstance(live, Tensor) or live.shape != saved.shape:
            raise RuntimeError(f"capture-state tensor layout changed: {path}")
        live.copy_(saved.to(device=live.device, dtype=live.dtype))
        return
    if isinstance(saved, dict):
        if not isinstance(live, dict) or live.keys() != saved.keys():
            raise RuntimeError(f"capture-state mapping changed: {path}")
        for name in saved:
            _restore_capture_state_(
                live[name], saved[name], path=f"{path}.{name}"
            )
        return
    if isinstance(saved, (tuple, list)):
        if not isinstance(live, type(saved)) or len(live) != len(saved):
            raise RuntimeError(f"capture-state sequence changed: {path}")
        for index, (live_item, saved_item) in enumerate(zip(live, saved, strict=True)):
            _restore_capture_state_(
                live_item, saved_item, path=f"{path}[{index}]"
            )
        return
    if live != saved:
        raise RuntimeError(f"capture changed non-tensor state: {path}")


def _clone_persistent_capture_boundary(
    trainer: NativePCTrainer,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    trainer_state = dict(trainer.state_dict())
    trainer_state.pop("optimizer")
    trainer_state.pop("update_count")
    return (
        _cpu_clone_capture_state(trainer.model.state_dict()),
        _cpu_clone_capture_state(trainer_state),
        int(trainer.update_count),
    )


@torch.no_grad()
def _restore_persistent_capture_boundary(
    trainer: NativePCTrainer,
    snapshot: tuple[dict[str, Any], dict[str, Any], int],
) -> None:
    model_state, trainer_state, update_count = snapshot
    live_trainer_state = dict(trainer.state_dict())
    live_trainer_state.pop("optimizer")
    live_trainer_state.pop("update_count")
    _restore_capture_state_(
        trainer.model.state_dict(), model_state, path="model"
    )
    _restore_capture_state_(live_trainer_state, trainer_state, path="trainer")
    trainer._update_count.fill_(update_count)


def _optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.nn.Parameter, ...]:
    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return tuple(parameters)


def _clone_optimizer_boundary(
    optimizer: torch.optim.Optimizer,
) -> tuple[
    tuple[tuple[torch.nn.Parameter, Tensor], ...],
    tuple[tuple[Tensor, Tensor], ...],
    tuple[tuple[torch.nn.Parameter, Tensor | None], ...],
    _OptimizerBoundaryMetadata,
]:
    parameters = _optimizer_parameters(optimizer)
    parameter_values = tuple(
        (parameter, parameter.detach().clone()) for parameter in parameters
    )
    state_values = tuple(
        (value, value.detach().clone())
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, Tensor)
    )
    gradient_references = tuple(
        (parameter, parameter.grad) for parameter in parameters
    )
    state_layout = tuple(
        (id(parameter), tuple(sorted(state)))
        for parameter, state in optimizer.state.items()
    )
    scalar_state_values = tuple(
        (state, name, value)
        for state in optimizer.state.values()
        for name, value in state.items()
        if isinstance(value, (bool, int, float, str, type(None)))
    )
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("split-update optimizer parameters must share one device")
    device = next(iter(devices))
    cpu_rng_state = torch.random.get_rng_state().clone()
    cuda_rng_state = (
        torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    )
    metadata = _OptimizerBoundaryMetadata(
        state_layout=state_layout,
        scalar_state_values=scalar_state_values,
        cpu_rng_state=cpu_rng_state,
        cuda_rng_state=cuda_rng_state,
    )
    return parameter_values, state_values, gradient_references, metadata


@torch.no_grad()
def _restore_optimizer_boundary(
    optimizer: torch.optim.Optimizer,
    snapshot: tuple[
        tuple[tuple[torch.nn.Parameter, Tensor], ...],
        tuple[tuple[Tensor, Tensor], ...],
        tuple[tuple[torch.nn.Parameter, Tensor | None], ...],
        _OptimizerBoundaryMetadata,
    ],
) -> None:
    parameter_values, state_values, gradient_references, metadata = snapshot
    current_layout = tuple(
        (id(parameter), tuple(sorted(state)))
        for parameter, state in optimizer.state.items()
    )
    if current_layout != metadata.state_layout:
        raise RuntimeError("optimizer state layout changed during split capture")
    for destination, source in parameter_values:
        destination.copy_(source)
    for destination, source in state_values:
        destination.copy_(source)
    for state, name, value in metadata.scalar_state_values:
        state[name] = value
    for parameter, gradient in gradient_references:
        parameter.grad = gradient
    torch.random.set_rng_state(metadata.cpu_rng_state)
    if metadata.cuda_rng_state is not None:
        device = parameter_values[0][0].device
        torch.cuda.set_rng_state(metadata.cuda_rng_state, device=device)


def _is_bound_optimizer_step(
    action: object,
    optimizer: torch.optim.Optimizer,
) -> bool:
    return (
        callable(action)
        and getattr(action, "__self__", None) is optimizer
        and getattr(action, "__name__", None) == "step"
    )


class StaticWholeUpdateCUDAGraph:
    """Own stable storage for one fixed-shape strict-local update graph."""

    def __init__(
        self,
        trainer: NativePCTrainer,
        example_tokens: Tensor,
        example_targets: Tensor,
    ) -> None:
        if (
            example_tokens.device.type != "cuda"
            or example_targets.device != example_tokens.device
        ):
            raise ValueError("whole-update capture requires one CUDA device")
        if (
            example_tokens.shape != example_targets.shape
            or example_tokens.ndim != 2
        ):
            raise ValueError("whole-update capture inputs must share [batch, sequence]")
        self.trainer = trainer
        self.device = example_tokens.device
        self.static_tokens = torch.empty_like(example_tokens)
        self.static_targets = torch.empty_like(example_targets)
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream(device=self.device)
        self.metrics: dict[str, Tensor | int]
        self.spectral_refresh_count = 0

        trainer.ensure_forward_jacobian_probe_buffer(example_tokens)

        torch.cuda.synchronize(self.device)
        reserved_before = torch.cuda.memory_reserved(self.device)
        capture_started = time.perf_counter()
        prior_capturable = _prepare_adamw_capture(trainer, self.device)
        _prepare_capture_control_flow(trainer)
        current_stream = torch.cuda.current_stream(self.device)
        self.capture_stream.wait_stream(current_stream)
        try:
            with torch.cuda.graph(
                self.graph,
                stream=self.capture_stream,
                capture_error_mode="global",
            ):
                self.metrics = trainer.local_update(
                    self.static_tokens,
                    self.static_targets,
                )
        finally:
            _clear_capture_control_flow(trainer)
            _restore_adamw_metadata(trainer, prior_capturable)
        current_stream.wait_stream(self.capture_stream)
        torch.cuda.synchronize(self.device)
        self.capture_seconds = time.perf_counter() - capture_started
        self.graph_pool_reserved_bytes = max(
            0,
            torch.cuda.memory_reserved(self.device) - reserved_before,
        )
        self.static_input_bytes = (
            self.static_tokens.numel() * self.static_tokens.element_size()
            + self.static_targets.numel() * self.static_targets.element_size()
        )
        self.static_output_bytes = sum(
            value.numel() * value.element_size()
            for value in self.metrics.values()
            if isinstance(value, Tensor)
        )
        self.graph_owned_host_metric_ids = tuple(
            id(self.metrics[name]) for name in CAPTURE_HOST_METRIC_FIELDS
        )

    @torch.no_grad()
    def replay(
        self,
        tokens: Tensor,
        targets: Tensor,
        *,
        materialize_host_metrics: bool = True,
    ) -> dict[str, Tensor | int]:
        if (
            tokens.shape != self.static_tokens.shape
            or targets.shape != self.static_targets.shape
            or tokens.dtype != self.static_tokens.dtype
            or targets.dtype != self.static_targets.dtype
            or tokens.device != self.device
            or targets.device != self.device
        ):
            raise ValueError("whole-update replay input contract changed")
        self.trainer.refresh_forward_jacobian_probes(tokens)
        self.static_tokens.copy_(tokens)
        self.static_targets.copy_(targets)
        self.graph.replay()
        if _refresh_predictive_rls_geometry_after_replay(self.trainer):
            self.spectral_refresh_count += 1
        return (
            finalize_graph_metrics(self.metrics)
            if materialize_host_metrics
            else deferred_graph_metrics(self.metrics)
        )


class StaticSplitUpdateCUDAGraph:
    """Capture strict-local work around one unchanged eager optimizer step."""

    def __init__(
        self,
        trainer: NativePCTrainer,
        example_tokens: Tensor,
        example_targets: Tensor,
    ) -> None:
        if (
            example_tokens.device.type != "cuda"
            or example_targets.device != example_tokens.device
        ):
            raise ValueError("split-update capture requires one CUDA device")
        if (
            example_tokens.shape != example_targets.shape
            or example_tokens.ndim != 2
        ):
            raise ValueError("split-update capture inputs must share [batch, sequence]")
        self.trainer = trainer
        self.optimizer = _require_eager_split_optimizer(trainer)
        self.device = example_tokens.device
        self.static_tokens = torch.empty_like(example_tokens)
        self.static_targets = torch.empty_like(example_targets)
        self.pre_optimizer_graph = torch.cuda.CUDAGraph()
        self.post_optimizer_graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream(device=self.device)
        self.metrics: dict[str, Tensor | int]
        self.spectral_refresh_count = 0
        self.pre_replay_count = 0
        self.eager_optimizer_step_count = 0
        self.post_replay_count = 0

        trainer.ensure_forward_jacobian_probe_buffer(example_tokens)

        torch.cuda.synchronize(self.device)
        reserved_before = torch.cuda.memory_reserved(self.device)
        capture_started = time.perf_counter()
        boundary_snapshot = _clone_optimizer_boundary(self.optimizer)
        persistent_snapshot = _clone_persistent_capture_boundary(trainer)
        _prepare_capture_control_flow(trainer)
        current_stream = torch.cuda.current_stream(self.device)
        self.capture_stream.wait_stream(current_stream)
        segments = trainer._local_update_segments(
            self.static_tokens,
            self.static_targets,
        )
        try:
            with torch.no_grad():
                with torch.cuda.graph(
                    self.pre_optimizer_graph,
                    stream=self.capture_stream,
                    capture_error_mode="global",
                ):
                    optimizer_action = next(segments)
            if not _is_bound_optimizer_step(optimizer_action, self.optimizer):
                raise RuntimeError(
                    "split-update capture reached a different optimizer action"
                )
            self.optimizer_action = optimizer_action
            self.static_gradients = tuple(
                (parameter, parameter.grad)
                for parameter in _optimizer_parameters(self.optimizer)
                if parameter.grad is not None
            )
            frame = segments.gi_frame
            if frame is None:
                raise RuntimeError("split-update generator frame ended before post graph")
            self.graph_owned_pre_context = dict(frame.f_locals)

            current_stream.wait_stream(self.capture_stream)
            self.optimizer_action()
            self.capture_stream.wait_stream(current_stream)
            with torch.no_grad():
                with torch.cuda.graph(
                    self.post_optimizer_graph,
                    stream=self.capture_stream,
                    capture_error_mode="global",
                ):
                    try:
                        next(segments)
                    except StopIteration as completed:
                        self.metrics = completed.value
                    else:
                        raise RuntimeError("split-update generator yielded more than once")
            if not isinstance(self.metrics, dict):
                raise RuntimeError("split-update post graph did not return metrics")
        finally:
            current_stream.wait_stream(self.capture_stream)
            torch.cuda.synchronize(self.device)
            _restore_persistent_capture_boundary(trainer, persistent_snapshot)
            _restore_optimizer_boundary(self.optimizer, boundary_snapshot)
            _clear_capture_control_flow(trainer)
        torch.cuda.synchronize(self.device)
        self.capture_seconds = time.perf_counter() - capture_started
        self.graph_pool_reserved_bytes = max(
            0,
            torch.cuda.memory_reserved(self.device) - reserved_before,
        )
        self.static_input_bytes = (
            self.static_tokens.numel() * self.static_tokens.element_size()
            + self.static_targets.numel() * self.static_targets.element_size()
        )
        self.static_gradient_bytes = sum(
            gradient.numel() * gradient.element_size()
            for _, gradient in self.static_gradients
        )
        self.static_output_bytes = sum(
            value.numel() * value.element_size()
            for value in self.metrics.values()
            if isinstance(value, Tensor)
        )
        self.graph_owned_host_metric_ids = tuple(
            id(self.metrics[name]) for name in CAPTURE_HOST_METRIC_FIELDS
        )

    def _install_static_gradients(self) -> None:
        static_by_parameter = {
            id(parameter): gradient for parameter, gradient in self.static_gradients
        }
        for parameter in _optimizer_parameters(self.optimizer):
            parameter.grad = static_by_parameter.get(id(parameter))

    @torch.no_grad()
    def replay(
        self,
        tokens: Tensor,
        targets: Tensor,
    ) -> dict[str, Tensor | int]:
        if (
            tokens.shape != self.static_tokens.shape
            or targets.shape != self.static_targets.shape
            or tokens.dtype != self.static_tokens.dtype
            or targets.dtype != self.static_targets.dtype
            or tokens.device != self.device
            or targets.device != self.device
        ):
            raise ValueError("split-update replay input contract changed")
        self.trainer.refresh_forward_jacobian_probes(tokens)
        self.static_tokens.copy_(tokens)
        self.static_targets.copy_(targets)
        self._install_static_gradients()
        self.pre_optimizer_graph.replay()
        self.pre_replay_count += 1
        if self.trainer.config.nonfinite_local_credit_quarantine:
            self.trainer._materialize_captured_nonfinite_local_credit_events()
        self.optimizer_action()
        if self.trainer.config.nonfinite_local_credit_quarantine:
            self.trainer._require_finite_optimizer_state_after_quarantine()
        self.eager_optimizer_step_count += 1
        self.post_optimizer_graph.replay()
        self.post_replay_count += 1
        if _refresh_predictive_rls_geometry_after_replay(self.trainer):
            self.spectral_refresh_count += 1
        return finalize_graph_metrics(self.metrics)


@torch.no_grad()
def _eager_embedding_gradient_reduction(
    embedding: Tensor,
    token_ids: Tensor,
    observation_direction: Tensor,
    *,
    sample_count: int,
    width: int,
) -> Tensor:
    if sample_count != token_ids.numel() or sample_count <= 0:
        raise ValueError("embedding-gradient sample count differs from token count")
    if observation_direction.numel() != sample_count * width:
        raise ValueError("embedding-gradient observation direction shape differs")
    if embedding.ndim != 2 or embedding.shape[1] != width:
        raise ValueError("embedding-gradient parameter shape differs")
    if (
        token_ids.device != embedding.device
        or observation_direction.device != embedding.device
    ):
        raise ValueError("embedding-gradient tensors must share one device")
    gradient = torch.zeros_like(embedding)
    gradient.index_add_(
        0,
        token_ids.reshape(-1),
        -observation_direction.reshape(sample_count, width) / sample_count,
    )
    return gradient


class StaticEmbeddingGradientSplitUpdateCUDAGraph(StaticSplitUpdateCUDAGraph):
    """Replay captured local credit with the embedding reduction left eager."""

    def __init__(
        self,
        trainer: NativePCTrainer,
        example_tokens: Tensor,
        example_targets: Tensor,
    ) -> None:
        if not trainer.config.calibrated_transition_credit:
            raise ValueError(
                "eager embedding-gradient split requires calibrated transition credit"
            )
        super().__init__(trainer, example_tokens, example_targets)
        self.eager_embedding_gradient_reduction_count = 0

    @torch.no_grad()
    def _rebuild_embedding_gradient_eager(self) -> None:
        token_ids = self.graph_owned_pre_context.get("token_ids")
        observation_direction = self.graph_owned_pre_context.get(
            "observation_direction"
        )
        sample_count = self.graph_owned_pre_context.get("sample_count")
        if not isinstance(token_ids, Tensor):
            raise RuntimeError("captured token ids are missing")
        if not isinstance(observation_direction, Tensor):
            raise RuntimeError("captured observation direction is missing")
        if not isinstance(sample_count, int):
            raise RuntimeError("captured embedding-gradient sample count is missing")
        embedding = self.trainer.model.embedding.weight
        embedding.grad = _eager_embedding_gradient_reduction(
            embedding,
            token_ids,
            observation_direction,
            sample_count=sample_count,
            width=self.trainer.model.config.width,
        )
        self.eager_embedding_gradient_reduction_count += 1

    @torch.no_grad()
    def replay(
        self,
        tokens: Tensor,
        targets: Tensor,
    ) -> dict[str, Tensor | int]:
        if (
            tokens.shape != self.static_tokens.shape
            or targets.shape != self.static_targets.shape
            or tokens.dtype != self.static_tokens.dtype
            or targets.dtype != self.static_targets.dtype
            or tokens.device != self.device
            or targets.device != self.device
        ):
            raise ValueError("split-update replay input contract changed")
        self.trainer.refresh_forward_jacobian_probes(tokens)
        self.static_tokens.copy_(tokens)
        self.static_targets.copy_(targets)
        self._install_static_gradients()
        self.pre_optimizer_graph.replay()
        self.pre_replay_count += 1
        if self.trainer.config.nonfinite_local_credit_quarantine:
            self.trainer._materialize_captured_nonfinite_local_credit_events()
        self._rebuild_embedding_gradient_eager()
        if self.trainer.config.nonfinite_local_credit_quarantine:
            self.trainer._quarantine_nonfinite_local_credit(
                stage="pre_optimizer_commit",
                merge_existing_stage=True,
            )
        self.optimizer_action()
        if self.trainer.config.nonfinite_local_credit_quarantine:
            self.trainer._require_finite_optimizer_state_after_quarantine()
        self.eager_optimizer_step_count += 1
        self.post_optimizer_graph.replay()
        self.post_replay_count += 1
        if _refresh_predictive_rls_geometry_after_replay(self.trainer):
            self.spectral_refresh_count += 1
        return finalize_graph_metrics(self.metrics)


@torch.no_grad()
def _eager_embedding_gradient_source_reduction(
    embedding: Tensor,
    token_ids: Tensor,
    embedding_gradient_source: Tensor,
    *,
    sample_count: int,
    width: int,
) -> Tensor:
    if sample_count != token_ids.numel() or sample_count <= 0:
        raise ValueError("embedding-gradient sample count differs from token count")
    if embedding_gradient_source.shape != (sample_count, width):
        raise ValueError("exact embedding-gradient source shape differs")
    if embedding.ndim != 2 or embedding.shape[1] != width:
        raise ValueError("embedding-gradient parameter shape differs")
    if (
        token_ids.device != embedding.device
        or embedding_gradient_source.device != embedding.device
    ):
        raise ValueError("embedding-gradient tensors must share one device")
    gradient = torch.zeros_like(embedding)
    _scoped_deterministic_index_add_(
        gradient,
        0,
        token_ids.reshape(-1),
        embedding_gradient_source,
    )
    return gradient


class StaticExactEmbeddingGradientSplitUpdateCUDAGraph(StaticSplitUpdateCUDAGraph):
    """Replay captured local credit with its exact embedding source left eager."""

    def __init__(
        self,
        trainer: NativePCTrainer,
        example_tokens: Tensor,
        example_targets: Tensor,
    ) -> None:
        if not trainer.config.calibrated_transition_credit:
            raise ValueError(
                "exact embedding-gradient split requires calibrated transition credit"
            )
        super().__init__(trainer, example_tokens, example_targets)
        self.eager_embedding_gradient_reduction_count = 0

    @torch.no_grad()
    def _rebuild_embedding_gradient_eager(self) -> None:
        token_ids = self.graph_owned_pre_context.get("token_ids")
        source = self.graph_owned_pre_context.get("embedding_gradient_source")
        sample_count = self.graph_owned_pre_context.get("sample_count")
        if not isinstance(token_ids, Tensor):
            raise RuntimeError("captured token ids are missing")
        if not isinstance(source, Tensor):
            raise RuntimeError("captured exact embedding-gradient source is missing")
        if not isinstance(sample_count, int):
            raise RuntimeError("captured embedding-gradient sample count is missing")
        embedding = self.trainer.model.embedding.weight
        embedding.grad = _eager_embedding_gradient_source_reduction(
            embedding,
            token_ids,
            source,
            sample_count=sample_count,
            width=self.trainer.model.config.width,
        )
        self.eager_embedding_gradient_reduction_count += 1

    @torch.no_grad()
    def replay(
        self,
        tokens: Tensor,
        targets: Tensor,
    ) -> dict[str, Tensor | int]:
        if (
            tokens.shape != self.static_tokens.shape
            or targets.shape != self.static_targets.shape
            or tokens.dtype != self.static_tokens.dtype
            or targets.dtype != self.static_targets.dtype
            or tokens.device != self.device
            or targets.device != self.device
        ):
            raise ValueError("split-update replay input contract changed")
        self.static_tokens.copy_(tokens)
        self.static_targets.copy_(targets)
        self._install_static_gradients()
        self.pre_optimizer_graph.replay()
        self.pre_replay_count += 1
        self._rebuild_embedding_gradient_eager()
        self.optimizer_action()
        self.eager_optimizer_step_count += 1
        self.post_optimizer_graph.replay()
        self.post_replay_count += 1
        if _refresh_predictive_rls_geometry_after_replay(self.trainer):
            self.spectral_refresh_count += 1
        return finalize_graph_metrics(self.metrics)


@torch.no_grad()
def _eager_post_clip_embedding_gradient_reconstruction(
    embedding: Tensor,
    token_ids: Tensor,
    embedding_gradient_source: Tensor,
    gradient_total_norm: Tensor,
    *,
    sample_count: int,
    width: int,
    max_norm: float,
    embedding_gradient_base: Tensor | None = None,
) -> Tensor:
    if gradient_total_norm.ndim != 0:
        raise ValueError("embedding-gradient total norm must be scalar")
    if gradient_total_norm.device != embedding.device:
        raise ValueError("embedding-gradient total norm device differs")
    if embedding_gradient_base is None:
        gradient = torch.zeros_like(embedding)
    else:
        if embedding_gradient_base.shape != embedding.shape:
            raise ValueError("embedding-gradient base shape differs")
        if embedding_gradient_base.device != embedding.device:
            raise ValueError("embedding-gradient base device differs")
        gradient = embedding_gradient_base.clone()
    common_dtype = torch.promote_types(
        gradient.dtype,
        embedding_gradient_source.dtype,
    )
    gradient = gradient.to(common_dtype)
    _scoped_deterministic_index_add_(
        gradient,
        0,
        token_ids.reshape(-1),
        embedding_gradient_source.to(common_dtype),
    )
    clip_coefficient = float(max_norm) / (gradient_total_norm + 1e-6)
    clip_coefficient_clamped = torch.clamp(clip_coefficient, max=1.0)
    torch._foreach_mul_(
        [gradient],
        clip_coefficient_clamped.to(gradient.device),
    )
    return gradient


@torch.no_grad()
def _eager_post_clip_kronecker_projection_gradient_reconstruction(
    trainer: NativePCTrainer,
    token_ids: Tensor,
    embedding_gradient_source: Tensor,
    gradient_total_norm: Tensor,
) -> Tensor:
    model = trainer.model
    projection = model.kronecker_input_projection
    byte_buffer = model._kronecker_byte_buffer
    length_buffer = model._kronecker_length_buffer
    if (
        projection is None
        or byte_buffer is None
        or length_buffer is None
        or model.kronecker_codec_sha256
        != model.config.kronecker_input_codec_sha256
    ):
        raise RuntimeError("Kronecker projection reconstruction state is missing")
    if embedding_gradient_source.shape != (
        token_ids.numel(),
        model.config.width,
    ):
        raise ValueError("Kronecker projection reconstruction source differs")
    dense_base, sparse_indices, sparse_source = (
        kronecker_projection_gradient_components(
            token_ids,
            byte_buffer,
            length_buffer,
            embedding_gradient_source,
        )
    )
    gradient = dense_base.unsqueeze(0).expand_as(projection).clone()
    _scoped_deterministic_index_add_(
        gradient,
        0,
        sparse_indices,
        sparse_source,
    )
    clip_coefficient = trainer.config.grad_clip / (gradient_total_norm + 1e-6)
    gradient.mul_(torch.clamp(clip_coefficient, max=1.0))
    return gradient


@torch.no_grad()
def _eager_post_clip_bpe_ancestry_gradient_reconstruction(
    trainer: NativePCTrainer,
    token_ids: Tensor,
    embedding_gradient_source: Tensor,
    gradient_total_norm: Tensor,
) -> Tensor:
    model = trainer.model
    table = model.bpe_ancestry_input_table
    indices = model._bpe_ancestry_indices
    counts = model._bpe_ancestry_counts
    inverse_sqrt_counts = model._bpe_ancestry_inverse_sqrt_counts
    if (
        table is None
        or indices is None
        or counts is None
        or inverse_sqrt_counts is None
        or model.bpe_ancestry_artifact_sha256
        != model.config.bpe_ancestry_input_artifact_sha256
    ):
        raise RuntimeError("BPE ancestry reconstruction state is missing")
    if embedding_gradient_source.shape != (
        token_ids.numel(),
        model.config.width,
    ):
        raise ValueError("BPE ancestry reconstruction source differs")
    if getattr(model.config, "bpe_ancestry_fused_execution", False):
        from pfc.phase750_fused_bpe_ancestry_input import fused_ancestry_gradient

        gradient = fused_ancestry_gradient(
            token_ids,
            embedding_gradient_source,
            table,
            indices,
            counts,
            inverse_sqrt_counts,
        )
    else:
        ancestry_indices, ancestry_source = ancestry_gradient_components(
            token_ids,
            embedding_gradient_source,
            indices,
            counts,
            inverse_sqrt_counts,
        )
        gradient = torch.zeros_like(table)
        _scoped_deterministic_index_add_(
            gradient,
            0,
            ancestry_indices,
            ancestry_source,
        )
    clip_coefficient = trainer.config.grad_clip / (gradient_total_norm + 1e-6)
    gradient.mul_(torch.clamp(clip_coefficient, max=1.0))
    return gradient


class StaticPostClipEmbeddingGradientSplitUpdateCUDAGraph(
    StaticSplitUpdateCUDAGraph
):
    """Replay captured local credit with exact post-clip embedding reconstruction."""

    def __init__(
        self,
        trainer: NativePCTrainer,
        example_tokens: Tensor,
        example_targets: Tensor,
    ) -> None:
        if not trainer.config.calibrated_transition_credit:
            raise ValueError(
                "post-clip embedding split requires calibrated transition credit"
            )
        if trainer.config.grad_clip <= 0:
            raise ValueError("post-clip embedding split requires active gradient clip")
        super().__init__(trainer, example_tokens, example_targets)
        self.eager_embedding_gradient_reduction_count = 0

    @torch.no_grad()
    def _rebuild_embedding_gradient_eager(self) -> None:
        context = self.graph_owned_pre_context
        token_ids = context.get("token_ids")
        source = context.get("embedding_gradient_source")
        total_norm = context.get("gradient_total_norm")
        sample_count = context.get("sample_count")
        base = (
            context.get("embedding_output_association")
            if self.trainer.model.config.shared_embedding_decoder
            else None
        )
        if not isinstance(token_ids, Tensor):
            raise RuntimeError("captured token ids are missing")
        if not isinstance(source, Tensor):
            raise RuntimeError("captured exact embedding-gradient source is missing")
        if not isinstance(total_norm, Tensor):
            raise RuntimeError("captured gradient total norm is missing")
        if not isinstance(sample_count, int):
            raise RuntimeError("captured embedding-gradient sample count is missing")
        if self.trainer.model.config.shared_embedding_decoder and not isinstance(
            base, Tensor
        ):
            raise RuntimeError("captured shared endpoint gradient base is missing")
        model = self.trainer.model
        embedding = model.embedding.weight
        ternary_embedding = getattr(model, "ternary_embedding", None)
        if getattr(model.config, "bpe_ancestry_input_embedding", False):
            if not isinstance(base, Tensor):
                raise RuntimeError("BPE ancestry output-head gradient base is missing")
            embedding.grad = _eager_post_clip_embedding_gradient_reconstruction(
                embedding,
                token_ids,
                torch.zeros_like(source),
                total_norm,
                sample_count=sample_count,
                width=model.config.width,
                max_norm=self.trainer.config.grad_clip,
                embedding_gradient_base=base,
            )
            model.bpe_ancestry_input_table.grad = (
                _eager_post_clip_bpe_ancestry_gradient_reconstruction(
                    self.trainer,
                    token_ids,
                    source,
                    total_norm,
                )
            )
        elif getattr(model.config, "kronecker_input_embedding", False):
            if not isinstance(base, Tensor):
                raise RuntimeError("Kronecker output-head gradient base is missing")
            embedding.grad = _eager_post_clip_embedding_gradient_reconstruction(
                embedding,
                token_ids,
                torch.zeros_like(source),
                total_norm,
                sample_count=sample_count,
                width=model.config.width,
                max_norm=self.trainer.config.grad_clip,
                embedding_gradient_base=base,
            )
            model.kronecker_input_projection.grad = (
                _eager_post_clip_kronecker_projection_gradient_reconstruction(
                    self.trainer,
                    token_ids,
                    source,
                    total_norm,
                )
            )
        elif ternary_embedding is None:
            embedding.grad = _eager_post_clip_embedding_gradient_reconstruction(
                embedding,
                token_ids,
                source,
                total_norm,
                sample_count=sample_count,
                width=model.config.width,
                max_norm=self.trainer.config.grad_clip,
                embedding_gradient_base=base,
            )
        else:
            dense_width = model.config.split_ternary_lexical_dense_width
            dense_base = base[:, :dense_width] if base is not None else None
            residual_base = base[:, dense_width:] if base is not None else None
            embedding.grad = _eager_post_clip_embedding_gradient_reconstruction(
                embedding,
                token_ids,
                source[:, :dense_width],
                total_norm,
                sample_count=sample_count,
                width=dense_width,
                max_norm=self.trainer.config.grad_clip,
                embedding_gradient_base=dense_base,
            )
            ternary_embedding.weight.grad = (
                _eager_post_clip_embedding_gradient_reconstruction(
                    ternary_embedding.weight,
                    token_ids,
                    source[:, dense_width:],
                    total_norm,
                    sample_count=sample_count,
                    width=model.config.width - dense_width,
                    max_norm=self.trainer.config.grad_clip,
                    embedding_gradient_base=residual_base,
                )
            )
        self.eager_embedding_gradient_reduction_count += 1

    @torch.no_grad()
    def replay(
        self,
        tokens: Tensor,
        targets: Tensor,
        *,
        materialize_host_metrics: bool = True,
    ) -> dict[str, Tensor | int]:
        if (
            tokens.shape != self.static_tokens.shape
            or targets.shape != self.static_targets.shape
            or tokens.dtype != self.static_tokens.dtype
            or targets.dtype != self.static_targets.dtype
            or tokens.device != self.device
            or targets.device != self.device
        ):
            raise ValueError("split-update replay input contract changed")
        self.static_tokens.copy_(tokens)
        self.static_targets.copy_(targets)
        self._install_static_gradients()
        self.pre_optimizer_graph.replay()
        self.pre_replay_count += 1
        if self.trainer.config.nonfinite_local_credit_quarantine:
            self.trainer._materialize_captured_nonfinite_local_credit_events()
        self._rebuild_embedding_gradient_eager()
        if self.trainer.config.nonfinite_local_credit_quarantine:
            self.trainer._quarantine_nonfinite_local_credit(
                stage="pre_optimizer_commit",
                merge_existing_stage=True,
            )
        self.optimizer_action()
        if self.trainer.config.nonfinite_local_credit_quarantine:
            self.trainer._require_finite_optimizer_state_after_quarantine()
        self.eager_optimizer_step_count += 1
        self.post_optimizer_graph.replay()
        self.post_replay_count += 1
        if _refresh_predictive_rls_geometry_after_replay(self.trainer):
            self.spectral_refresh_count += 1
        return (
            finalize_graph_metrics(self.metrics)
            if materialize_host_metrics
            else deferred_graph_metrics(self.metrics)
        )


def capture_resource_row(graph: StaticWholeUpdateCUDAGraph) -> dict[str, Any]:
    return {
        "capture_seconds": graph.capture_seconds,
        "graph_pool_reserved_bytes": graph.graph_pool_reserved_bytes,
        "static_input_bytes": graph.static_input_bytes,
        "static_output_bytes": graph.static_output_bytes,
        "eager_spectral_refresh_after_replay": bool(
            graph.trainer.config.predictive_rls_likelihood
        ),
        "spectral_refresh_count": graph.spectral_refresh_count,
        "post_replay_host_metric_fields": len(CAPTURE_HOST_METRIC_FIELDS),
        "post_replay_host_metric_bytes": sum(
            graph.metrics[name].numel() * graph.metrics[name].element_size()
            for name in CAPTURE_HOST_METRIC_FIELDS
        ),
    }


def split_capture_resource_row(
    graph: StaticSplitUpdateCUDAGraph,
) -> dict[str, Any]:
    return {
        "capture_seconds": graph.capture_seconds,
        "graph_pool_reserved_bytes": graph.graph_pool_reserved_bytes,
        "static_input_bytes": graph.static_input_bytes,
        "static_gradient_bytes": graph.static_gradient_bytes,
        "static_output_bytes": graph.static_output_bytes,
        "pre_optimizer_replays": graph.pre_replay_count,
        "eager_optimizer_steps": graph.eager_optimizer_step_count,
        "post_optimizer_replays": graph.post_replay_count,
        "eager_spectral_refresh_after_replay": bool(
            graph.trainer.config.predictive_rls_likelihood
        ),
        "spectral_refresh_count": graph.spectral_refresh_count,
        "post_replay_host_metric_fields": len(CAPTURE_HOST_METRIC_FIELDS),
        "post_replay_host_metric_bytes": sum(
            graph.metrics[name].numel() * graph.metrics[name].element_size()
            for name in CAPTURE_HOST_METRIC_FIELDS
        ),
    }
