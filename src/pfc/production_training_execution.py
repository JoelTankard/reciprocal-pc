from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

import torch
from torch import Tensor

from pfc.native_pc_training import CAPTURE_HOST_METRIC_FIELDS
from pfc.native_pc_training import NativePCTrainer
from pfc.prefill_execution import compiler_counter_snapshot
from pfc.prefill_execution import install_compiled_prefill
from pfc.whole_update_execution import (
    finalize_graph_metrics,
    require_exact_split_capture_route,
    StaticPostClipEmbeddingGradientSplitUpdateCUDAGraph,
)


def configure_exact_production_training_execution(
    trainer: NativePCTrainer,
) -> None:
    """Install the accepted exact production workspace configuration."""

    model = trainer.model
    object.__setattr__(
        model,
        "config",
        replace(
            model.config,
            fused_sparse_count_sketch_prefill=True,
            composable_sparse_count_sketch_prefill=True,
            explicit_round_forward_pair_scan=True,
            exact_whole_prefill_workspace=True,
        ),
    )
    object.__setattr__(
        trainer,
        "config",
        replace(
            trainer.config,
            reuse_prefill_associative_reads=True,
            sparse_ordered_memory_source_credit=False,
            exact_ordered_credit_workspace=True,
            exact_table_major_memory_workspace=False,
            exact_bounded_memory_workspace=True,
        ),
    )
    object.__setattr__(
        model,
        "_transition_sequence",
        torch.compiler.disable(model._transition_sequence),
    )
    object.__setattr__(
        model,
        "_semantic_backoff_associative_sequence",
        torch.compiler.disable(model._semantic_backoff_associative_sequence),
    )


class ProductionTrainingExecutor:
    """Lazy exact CUDA execution for one fixed-shape ReciprocalPC trainer."""

    def __init__(self, trainer: NativePCTrainer) -> None:
        if trainer.model.device.type != "cuda":
            raise ValueError("production training execution requires CUDA")
        if trainer.model.dtype != torch.float32:
            raise ValueError("production training execution requires float32")
        if trainer.config.intrinsic_key_aligned_predictive_codebook:
            raise ValueError(
                "production training execution forbids rejected aligned codebook"
            )
        if trainer.config.exact_vjp_credit:
            raise ValueError(
                "production training execution requires strict local credit"
            )
        if not trainer.config.reuse_prefill_associative_reads:
            raise ValueError(
                "production training execution requires retained prefill reads"
            )
        if not trainer.config.charged_held_query_displacement_projection:
            raise ValueError(
                "production training execution requires Phase298 held-query credit"
            )
        self.trainer = trainer
        self.device = trainer.model.device
        self.compiled_prefill: Any | None = None
        self.graph: StaticPostClipEmbeddingGradientSplitUpdateCUDAGraph | None = (
            None
        )
        self.input_signature: tuple[
            tuple[int, ...],
            torch.dtype,
            torch.device,
        ] | None = None
        self.compile_warmup_seconds = 0.0
        self.graph_capture_seconds = 0.0
        self.compile_warmup_count = 0
        self.compiled_prefill_result_count = 0
        self.graph_construction_count = 0
        self.eager_bootstrap_update_count = 0
        self.graph_replay_update_count = 0
        self.compiler: dict[str, Any] = {}

    @staticmethod
    def _signature(tokens: Tensor) -> tuple[
        tuple[int, ...],
        torch.dtype,
        torch.device,
    ]:
        return tuple(tokens.shape), tokens.dtype, tokens.device

    def _validate_inputs(self, tokens: Tensor, targets: Tensor) -> None:
        if (
            tokens.ndim != 2
            or targets.shape != tokens.shape
            or tokens.dtype != torch.int64
            or targets.dtype != torch.int64
            or tokens.device != self.device
            or targets.device != self.device
        ):
            raise ValueError(
                "production execution inputs must be CUDA int64 [batch, sequence]"
            )
        signature = self._signature(tokens)
        if self.input_signature is not None and signature != self.input_signature:
            raise ValueError("production execution input signature changed")

    @torch.no_grad()
    def prepare(self, tokens: Tensor, targets: Tensor) -> None:
        """Warm compilation or construct the graph outside update timing."""

        self._validate_inputs(tokens, targets)
        if self.input_signature is None:
            self.input_signature = self._signature(tokens)
        if self.compiled_prefill is None:
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            started = time.perf_counter()
            self.compiled_prefill = install_compiled_prefill(self.trainer.model)
            warmup = self.compiled_prefill(
                tokens,
                None,
                return_associative_reads=True,
            )
            if not all(
                bool(torch.isfinite(value).all())
                for value in warmup
                if isinstance(value, Tensor)
            ):
                raise RuntimeError("compiled prefill warmup is non-finite")
            torch.cuda.synchronize(self.device)
            self.compile_warmup_seconds = time.perf_counter() - started
            self.compile_warmup_count = 1
            self.compiled_prefill_result_count = len(warmup)
            self.compiler = compiler_counter_snapshot()
            del warmup
        if self.graph is not None:
            return
        held_query_initialized = int(
            self.trainer._charged_held_query_displacement_initialized.item()
        )
        if held_query_initialized == 0:
            return
        self.graph = StaticPostClipEmbeddingGradientSplitUpdateCUDAGraph(
            self.trainer,
            tokens,
            targets,
        )
        self.graph_capture_seconds = self.graph.capture_seconds
        self.graph_construction_count = 1

    @torch.no_grad()
    def update(
        self,
        tokens: Tensor,
        targets: Tensor,
    ) -> dict[str, Tensor | int]:
        """Execute one retained update after prepare."""

        self._validate_inputs(tokens, targets)
        if self.compiled_prefill is None:
            raise RuntimeError("production execution requires prepare before update")
        if self.graph is not None:
            self.graph_replay_update_count += 1
            return self.graph.replay(tokens, targets)
        if self.eager_bootstrap_update_count:
            raise RuntimeError(
                "production execution requires graph preparation after bootstrap"
            )
        metrics = self.trainer.local_update(tokens, targets)
        self.eager_bootstrap_update_count = 1
        return metrics

    def execution_counters(self) -> dict[str, int | float | dict[str, Any]]:
        graph = self.graph
        return {
            "compile_warmup_count": self.compile_warmup_count,
            "compile_warmup_seconds": self.compile_warmup_seconds,
            "compiled_prefill_result_count": (
                self.compiled_prefill_result_count
            ),
            "graph_construction_count": self.graph_construction_count,
            "graph_capture_seconds": self.graph_capture_seconds,
            "eager_bootstrap_updates": self.eager_bootstrap_update_count,
            "graph_replay_updates": self.graph_replay_update_count,
            "pre_optimizer_graph_replays": (
                0 if graph is None else graph.pre_replay_count
            ),
            "eager_embedding_gradient_reconstructions": (
                0
                if graph is None
                else graph.eager_embedding_gradient_reduction_count
            ),
            "eager_optimizer_steps": (
                0 if graph is None else graph.eager_optimizer_step_count
            ),
            "post_optimizer_graph_replays": (
                0 if graph is None else graph.post_replay_count
            ),
            "geometry_refreshes": (
                0 if graph is None else graph.spectral_refresh_count
            ),
            "compiler": dict(self.compiler),
        }


class ExactSplitTrainingExecutor:
    """Exact split graph for one unchanged fixed-shape trainer."""

    def __init__(self, trainer: NativePCTrainer) -> None:
        if trainer.model.device.type != "cuda":
            raise ValueError("exact split execution requires CUDA")
        if trainer.model.dtype == torch.bfloat16:
            if not trainer.config.bf16_stochastic_training_state:
                raise ValueError(
                    "exact split BF16 execution requires stochastic BF16 state"
                )
        elif trainer.model.dtype != torch.float32:
            raise ValueError("exact split execution requires float32 or BF16")
        if trainer.config.exact_vjp_credit:
            raise ValueError("exact split execution requires strict local credit")
        self.capture_route = require_exact_split_capture_route(trainer)
        self.trainer = trainer
        self.device = trainer.model.device
        self.graph: StaticPostClipEmbeddingGradientSplitUpdateCUDAGraph | None = None
        self.input_signature: tuple[
            tuple[int, ...], torch.dtype, torch.device
        ] | None = None
        self.graph_capture_seconds = 0.0

    @staticmethod
    def _signature(tokens: Tensor) -> tuple[
        tuple[int, ...], torch.dtype, torch.device
    ]:
        return tuple(tokens.shape), tokens.dtype, tokens.device

    def _validate(self, tokens: Tensor, targets: Tensor) -> None:
        if (
            tokens.ndim != 2
            or targets.shape != tokens.shape
            or tokens.dtype != torch.int64
            or targets.dtype != torch.int64
            or tokens.device != self.device
            or targets.device != self.device
        ):
            raise ValueError("exact split inputs must be CUDA int64 [batch, sequence]")
        signature = self._signature(tokens)
        if self.input_signature is not None and signature != self.input_signature:
            raise ValueError("exact split input signature changed")

    @torch.no_grad()
    def prepare(self, tokens: Tensor, targets: Tensor) -> None:
        self._validate(tokens, targets)
        if self.input_signature is None:
            self.input_signature = self._signature(tokens)
        if self.graph is not None:
            return
        current_route = require_exact_split_capture_route(self.trainer)
        if current_route != self.capture_route:
            raise ValueError("exact split capture route changed before prepare")
        self.graph = StaticPostClipEmbeddingGradientSplitUpdateCUDAGraph(
            self.trainer, tokens, targets
        )
        self.graph_capture_seconds = self.graph.capture_seconds

    @torch.no_grad()
    def update(self, tokens: Tensor, targets: Tensor) -> dict[str, Tensor | int]:
        self._validate(tokens, targets)
        if self.graph is None:
            raise RuntimeError("exact split execution requires prepare")
        return self.graph.replay(tokens, targets)

    @torch.no_grad()
    def update_deferred(
        self,
        tokens: Tensor,
        targets: Tensor,
    ) -> dict[str, Tensor | int]:
        """Replay one update while leaving host-only counters on CUDA."""

        self._validate(tokens, targets)
        if self.graph is None:
            raise RuntimeError("deferred execution requires prepared split graph")
        return self.graph.replay(
            tokens,
            targets,
            materialize_host_metrics=False,
        )

    @staticmethod
    def materialize_metrics(
        metrics: dict[str, Tensor | int],
    ) -> dict[str, Tensor | int]:
        """Materialize a deferred metric view before retention or host use."""

        host_fields = tuple(metrics.get(name) for name in CAPTURE_HOST_METRIC_FIELDS)
        if all(isinstance(value, int) for value in host_fields):
            return dict(metrics)
        if not all(isinstance(value, Tensor) for value in host_fields):
            raise RuntimeError("graph host metrics mix tensors and integers")
        return finalize_graph_metrics(metrics)

    def execution_counters(self) -> dict[str, int | float | str]:
        graph = self.graph
        return {
            "capture_route": self.capture_route,
            "graph_construction_count": int(graph is not None),
            "graph_capture_seconds": self.graph_capture_seconds,
            "graph_replay_updates": 0 if graph is None else graph.pre_replay_count,
            "pre_optimizer_graph_replays": 0 if graph is None else graph.pre_replay_count,
            "eager_embedding_gradient_reconstructions": (
                0 if graph is None else graph.eager_embedding_gradient_reduction_count
            ),
            "eager_optimizer_steps": (
                0 if graph is None else graph.eager_optimizer_step_count
            ),
            "post_optimizer_graph_replays": (
                0 if graph is None else graph.post_replay_count
            ),
            "geometry_refreshes": 0 if graph is None else graph.spectral_refresh_count,
        }
