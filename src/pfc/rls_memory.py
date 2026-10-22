"""Functional, float32 recursive-least-squares associative memory.

Each independent batch lane holds S[B, V, K] and P[B, K, K], with
S = 0 and P = I / ridge initially. Reads are S @ query. A write uses
p = P @ key, d = 1 + key.T @ p, then updates
S' = S + (value - S @ key) @ (p / d).T and P' = P - p @ p.T / d.
There is no forgetting factor, learned projection, readout, or growing history.
The persistent tensor payload is 4 * B * (V * K + K * K) bytes.

This is an independent additive branch, not a replacement for native recurrent,
lexical, count-sketch, or predictive fast memory. It owns no neural projections
or decoders. Device placement permits sequential CPU reference execution or
CUDA tensor execution. The fused sequence executor in rls_sequence preserves
causal ordering and carries final S/P plus the previous key across chunks.
The batch axis represents independent streams, never successive time steps.
This reference performs host-visible validation and is not a fused CUDA kernel.

STRICT CAUSALITY CONTRACT: predict a target using read(state, query) BEFORE
writing that target. Only after the prediction has been scored may integration
call write(state, key, target_value). Never predict that same target from the
returned, target-written state. This primitive cannot enforce target provenance;
integration owns query/key construction and ordering.

All operations disable gradient recording and reject gradient-bearing tensors.
Only float32 is supported: this is a reference local-learning implementation,
not a mixed-precision path; matrix products disable ambient autocast. Inputs must
be finite; invalid arithmetic raises instead of clipping or repairing state.

State containers are frozen, but PyTorch tensors themselves remain mutable.
Callers must treat their contents as read-only. read() returns a fresh tensor;
write() returns fresh S and P tensors with no alias to any input, leaving the
supplied state unchanged. Keeping old states is optional and is not required by
this implementation. Persistent memory size is independent of the write count.
"""

from dataclasses import dataclass
import math
from numbers import Real

import torch


@dataclass(frozen=True)
class RLSState:
    """Read-only-by-contract tensor state with explicit leading batch axes."""

    S: torch.Tensor  # [batch, value_dim, key_dim]
    P: torch.Tensor  # [batch, key_dim, key_dim]


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.layout != torch.strided or tensor.dtype != torch.float32:
        raise TypeError(f"{name} must be a dense float32 tensor")
    if tensor.requires_grad:
        raise ValueError(f"{name} must not require gradients")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains nonfinite values")


def _validate_state(state: RLSState) -> tuple[int, int, int]:
    if not isinstance(state, RLSState):
        raise TypeError("state must be an RLSState")
    _finite_tensor("state.S", state.S)
    _finite_tensor("state.P", state.P)
    if state.S.ndim != 3 or state.P.ndim != 3:
        raise ValueError("state must have S[B,V,K] and P[B,K,K]")
    batch, value_dim, key_dim = state.S.shape
    if min(batch, value_dim, key_dim) <= 0:
        raise ValueError("state dimensions must all be positive")
    if state.P.shape != (batch, key_dim, key_dim):
        raise ValueError("state.P shape must match state.S batch and key dimensions")
    if state.S.device != state.P.device:
        raise ValueError("state.S and state.P must be on the same device")
    return batch, key_dim, value_dim


def _validate_vector(
    name: str, tensor: torch.Tensor, shape: tuple[int, int], device: torch.device
) -> None:
    _finite_tensor(name, tensor)
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on state device {device}")


@torch.no_grad()
def initial_state(
    batch: int,
    key_dim: int,
    value_dim: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    *,
    ridge: float = 0.25,
) -> RLSState:
    """Allocate independent zero maps and I/ridge covariances for each lane.

    ``device=None`` uses PyTorch's default device; pass a device explicitly for
    reproducible placement. ``dtype`` must be torch.float32. Ridge must be a
    finite positive real whose inverse is representable in float32.
    """
    _positive_int("batch", batch)
    _positive_int("key_dim", key_dim)
    _positive_int("value_dim", value_dim)
    if dtype != torch.float32:
        raise TypeError("RLS reference state supports only torch.float32")
    if isinstance(ridge, bool) or not isinstance(ridge, Real):
        raise TypeError("ridge must be a finite positive real")
    ridge = float(ridge)
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    inverse_ridge = 1.0 / ridge
    finfo = torch.finfo(torch.float32)
    if not math.isfinite(inverse_ridge) or not (
        finfo.tiny * finfo.eps <= inverse_ridge <= finfo.max
    ):
        raise ValueError("inverse ridge must be a finite positive float32 value")
    S = torch.zeros((batch, value_dim, key_dim), device=device, dtype=dtype)
    P = (torch.eye(key_dim, device=device, dtype=dtype) * inverse_ridge).unsqueeze(0)
    # clone materializes independent batch lanes rather than an aliased expand.
    P = P.expand(batch, key_dim, key_dim).clone()
    return RLSState(S=S, P=P)


@torch.no_grad()
def read(state: RLSState, query: torch.Tensor) -> torch.Tensor:
    """Return S @ query as [B,V]; query must be [B,K], without broadcasting.

    This call never updates memory. Query must contain only information already
    available at prediction time, not the target being predicted.
    """
    batch, key_dim, _ = _validate_state(state)
    _validate_vector("query", query, (batch, key_dim), state.S.device)
    with torch.autocast(device_type=state.S.device.type, enabled=False):
        prediction = torch.bmm(state.S, query.unsqueeze(-1)).squeeze(-1)
    _finite_tensor("prediction", prediction)
    return prediction


@torch.no_grad()
def write(state: RLSState, key: torch.Tensor, value: torch.Tensor) -> RLSState:
    """Return one RLS update per lane, without mutating or aliasing inputs.

    key is [B,K], value is [B,V]. Score the old state's prediction before using
    this operation with the corresponding target value. Failure in any lane
    rejects the entire update and leaves every input tensor unchanged.
    """
    batch, key_dim, value_dim = _validate_state(state)
    _validate_vector("key", key, (batch, key_dim), state.S.device)
    _validate_vector("value", value, (batch, value_dim), state.S.device)
    with torch.autocast(device_type=state.S.device.type, enabled=False):
        p = torch.bmm(state.P, key.unsqueeze(-1)).squeeze(-1)
        _finite_tensor("P @ key", p)
        quadratic = (key * p).sum(dim=-1, keepdim=True)
        denominator = 1.0 + quadratic
        if not bool(torch.isfinite(denominator).all()) or not bool((denominator > 0).all()):
            raise ValueError("RLS denominator must be finite and strictly positive")
        if not bool((quadratic >= 0).all()):
            raise ValueError("RLS key.T @ P @ key must be nonnegative")
        residual = value - torch.bmm(state.S, key.unsqueeze(-1)).squeeze(-1)
        gain = p / denominator
        S = state.S + residual.unsqueeze(-1) * gain.unsqueeze(-2)
        P = state.P - p.unsqueeze(-1) * gain.unsqueeze(-2)
        _finite_tensor("updated S", S)
        _finite_tensor("updated P", P)
    return RLSState(S=S, P=P)


def state_bytes(state: RLSState) -> int:
    """Return persistent tensor payload bytes, excluding Python/allocator overhead.

    Temporary arithmetic tensors and any old states retained by the caller are
    not included. The primitive itself retains only the returned S and P.
    """
    _validate_state(state)
    return state.S.numel() * state.S.element_size() + state.P.numel() * state.P.element_size()
