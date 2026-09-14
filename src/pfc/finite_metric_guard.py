from __future__ import annotations

import torch
from torch import Tensor


def _floating_binding_signature(
    metrics: dict[str, Tensor | int],
) -> tuple[
    tuple[
        str,
        int,
        int,
        tuple[int, ...],
        tuple[int, ...],
        torch.dtype,
        torch.device,
    ],
    ...,
]:
    return tuple(
        (
            name,
            id(value),
            value.data_ptr(),
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.device,
        )
        for name, value in metrics.items()
        if isinstance(value, Tensor) and value.is_floating_point()
    )


def require_aggregate_strict_finite(metrics: dict[str, Tensor | int]) -> None:
    """Validate strict counters and all floating metrics with one normal-path sync."""

    if int(metrics.get("global_backprops", -1)) != 0:
        raise ValueError("global-backprop counter differs")
    if int(metrics.get("forward_weight_transpose_reads", -1)) != 0:
        raise ValueError("transpose-read counter differs")

    floating = [
        (name, value)
        for name, value in metrics.items()
        if isinstance(value, Tensor) and value.is_floating_point()
    ]
    if not floating:
        return
    device = floating[0][1].device
    if any(value.device != device for _, value in floating):
        raise ValueError("floating metrics span multiple devices")
    finite = torch.stack(
        [torch.isfinite(value).all() for _name, value in floating]
    ).all()
    if bool(finite.item()):
        return

    for name, value in floating:
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"metric is nonfinite: {name}")
    raise RuntimeError("aggregate finite guard failed without an offending metric")


def require_foreach_norm_strict_finite(metrics: dict[str, Tensor | int]) -> None:
    """Validate all floating metrics through one non-mutating foreach reduction."""

    if int(metrics.get("global_backprops", -1)) != 0:
        raise ValueError("global-backprop counter differs")
    if int(metrics.get("forward_weight_transpose_reads", -1)) != 0:
        raise ValueError("transpose-read counter differs")

    floating = [
        (name, value)
        for name, value in metrics.items()
        if isinstance(value, Tensor) and value.is_floating_point()
    ]
    if not floating:
        return
    device = floating[0][1].device
    if any(value.device != device for _, value in floating):
        raise ValueError("floating metrics span multiple devices")

    nonempty = [(name, value) for name, value in floating if value.numel()]
    if not nonempty:
        return
    norms = torch._foreach_norm(
        [value for _name, value in nonempty],
        ord=float("inf"),
    )
    finite = torch.stack([torch.isfinite(norm) for norm in norms]).all()
    if bool(finite.item()):
        return

    for name, value in nonempty:
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"metric is nonfinite: {name}")
    raise RuntimeError("foreach finite guard failed without an offending metric")


class CapturedForeachNormFiniteGuard:
    """Replay a full-metric finite check bound to stable CUDA tensor storage."""

    def __init__(
        self,
        metrics: dict[str, Tensor | int],
        *,
        materialized_host_fields: tuple[str, ...] = (),
    ) -> None:
        floating = [
            (name, value)
            for name, value in metrics.items()
            if isinstance(value, Tensor) and value.is_floating_point()
        ]
        if not floating:
            raise ValueError("captured finite guard requires floating metrics")
        self.device = floating[0][1].device
        if self.device.type != "cuda":
            raise ValueError("captured finite guard requires CUDA metrics")
        if any(value.device != self.device for _, value in floating):
            raise ValueError("floating metrics span multiple devices")
        self._floating = tuple(floating)
        self._binding = _floating_binding_signature(metrics)
        self._materialized_host_fields = frozenset(materialized_host_fields)
        unknown_host_fields = self._materialized_host_fields - metrics.keys()
        if unknown_host_fields:
            raise ValueError(
                "captured finite guard host fields are missing: "
                + ", ".join(sorted(unknown_host_fields))
            )
        self._runtime_binding = tuple(
            item
            for item in self._binding
            if item[0] not in self._materialized_host_fields
        )
        self._nonempty = tuple(
            (name, value) for name, value in floating if value.numel()
        )
        if not self._nonempty:
            raise ValueError("captured finite guard requires nonempty metrics")
        inputs = [value for _name, value in self._nonempty]

        torch.cuda.synchronize(self.device)
        reserved_before = torch.cuda.memory_reserved(self.device)
        self.capture_stream = torch.cuda.Stream(device=self.device)
        current_stream = torch.cuda.current_stream(self.device)
        self.capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.capture_stream):
            for _ in range(3):
                warm_norms = torch._foreach_norm(inputs, ord=float("inf"))
                warm_finite = torch.stack(
                    [torch.isfinite(norm) for norm in warm_norms]
                ).all()
        current_stream.wait_stream(self.capture_stream)
        torch.cuda.synchronize(self.device)
        del warm_norms, warm_finite

        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream.wait_stream(current_stream)
        with torch.cuda.graph(
            self.graph,
            stream=self.capture_stream,
            capture_error_mode="global",
        ):
            self._norms = torch._foreach_norm(inputs, ord=float("inf"))
            self._finite = torch.stack(
                [torch.isfinite(norm) for norm in self._norms]
            ).all()
        current_stream.wait_stream(self.capture_stream)
        torch.cuda.synchronize(self.device)
        self.graph.replay()
        torch.cuda.synchronize(self.device)
        if not bool(self._finite.item()):
            raise ValueError("captured finite guard inputs start nonfinite")
        self.graph_pool_reserved_bytes = max(
            0,
            torch.cuda.memory_reserved(self.device) - reserved_before,
        )
        self.capture_count = 1
        self.construction_validation_replays = 1
        self.graph_replay_count = 0

    @property
    def floating_metric_count(self) -> int:
        return len(self._floating)

    @property
    def binding_signature(self) -> tuple:
        return self._binding

    def binding_matches(self, metrics: dict[str, Tensor | int]) -> bool:
        return _floating_binding_signature(metrics) == self._runtime_binding

    def binding_differences(
        self,
        metrics: dict[str, Tensor | int],
    ) -> tuple[str, ...]:
        current = _floating_binding_signature(metrics)
        differences = []
        for index in range(max(len(self._runtime_binding), len(current))):
            expected = (
                self._runtime_binding[index]
                if index < len(self._runtime_binding)
                else None
            )
            actual = current[index] if index < len(current) else None
            if expected != actual:
                differences.append(
                    f"index={index} expected={expected!r} actual={actual!r}"
                )
        return tuple(differences)

    def require(self, metrics: dict[str, Tensor | int]) -> None:
        if int(metrics.get("global_backprops", -1)) != 0:
            raise ValueError("global-backprop counter differs")
        if int(metrics.get("forward_weight_transpose_reads", -1)) != 0:
            raise ValueError("transpose-read counter differs")
        if not self.binding_matches(metrics):
            first = self.binding_differences(metrics)[0]
            raise ValueError(f"captured finite metric binding changed: {first}")

        self.graph.replay()
        self.graph_replay_count += 1
        if bool(self._finite.item()):
            return

        for name, value in self._nonempty:
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"metric is nonfinite: {name}")
        raise RuntimeError("captured finite guard failed without an offending metric")
