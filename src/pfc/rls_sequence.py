"""Causal full-embedding RLS reads with a CPU reference and CUDA scan.

At position t, write (previous_key, values[:, t]) if a previous key exists,
then read keys[:, t]. Values are *observed* embeddings, never unseen targets.
A fresh stream's first read is zero. Only final S/P and the last key cross a
chunk boundary; the caller carries has_previous=True after a nonempty chunk.
There is no autograd path, forgetting, per-step inverse, or saved state history.

Inputs are dense float32 tensors with B >= 1, T >= 1, K=32 and V=512, on one
CPU or CUDA device. Strided inputs are supported without contiguous copies.
CUDA requires Triton and launches eight independent 64-value CTAs per batch
lane, each scanning time sequentially with four warps. Each CTA keeps a
64x32 S tile and a replicated 32x32 P in float32; only tile zero stores final
P and last_key. Covariance work is duplicated eight times to bound register
pressure instead of holding a whole 512x32 S in one CTA. Small batches may
underutilize the GPU; long scans, reductions for strict arithmetic validation,
register spills and device watchdog limits are resource hazards to measure.
T is a compile-time loop bound, not an unrolled history allocation. Compilation
specializes by T, strides and state/previous-key presence. Launch uses no TF32
matrix products and disables FMA contraction for reference-like arithmetic.

CUDA validates all input elements and intermediate/final arithmetic on device,
then performs one host-visible status reduction after the complete scan. Any
nonfinite input/arithmetic, nonpositive denominator or negative quadratic
raises; invalid values are never clipped, repaired or exposed as a result.
CPU calls the existing rls_memory.read/write reference, including its strict
per-operation validation. Input tensors are never mutated. Returned reads use
4*B*T*512 bytes; final S/P use 4*B*(512*32+32*32), last_key 4*B*32,
and CUDA status scratch 4*B*8 bytes, independent of T except for reads.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor

from . import rls_memory

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
    triton = None
    tl = None


_KEY_DIM = 32
_VALUE_DIM = 512
_VALUE_TILE = 64


def _check_tensor(name: str, tensor: Tensor, shape: tuple[int, ...], device: torch.device) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.layout != torch.strided or tensor.dtype != torch.float32:
        raise TypeError(f"{name} must be a dense float32 tensor")
    if tensor.requires_grad:
        raise ValueError(f"{name} must not require gradients")
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}")


def _inverse_ridge(ridge: float) -> float:
    if isinstance(ridge, bool) or not isinstance(ridge, Real):
        raise TypeError("ridge must be a finite positive real")
    ridge = float(ridge)
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    inverse = 1.0 / ridge
    finfo = torch.finfo(torch.float32)
    if not math.isfinite(inverse) or not finfo.tiny * finfo.eps <= inverse <= finfo.max:
        raise ValueError("inverse ridge must be a finite positive float32 value")
    return inverse


if triton is not None:

    @triton.jit
    def _all_finite(value):
        # Comparison is false for NaN as well as either infinity.
        if len(value.shape) == 0:
            return tl.abs(value) <= 3.4028234663852886e38
        else:
            return tl.sum((~(tl.abs(value) <= 3.4028234663852886e38)).to(tl.int32)) == 0

    @triton.jit
    def _sequence_reads_kernel(
        keys,
        values,
        initial_s,
        initial_p,
        previous_key,
        reads,
        final_s,
        final_p,
        last_key,
        statuses,
        KEY_STRIDES: tl.constexpr,
        VALUE_STRIDES: tl.constexpr,
        S_STRIDES: tl.constexpr,
        P_STRIDES: tl.constexpr,
        PREVIOUS_STRIDES: tl.constexpr,
        SEQUENCE: tl.constexpr,
        INVERSE_RIDGE: tl.constexpr,
        HAS_STATE: tl.constexpr,
        HAS_PREVIOUS: tl.constexpr,
        PREVIOUS_PROVIDED: tl.constexpr,
        VALUE_TILE: tl.constexpr,
    ):
        batch = tl.program_id(0)
        tile = tl.program_id(1)
        k = tl.arange(0, 32)
        v = tile * VALUE_TILE + tl.arange(0, VALUE_TILE)
        status = tl.full((), 0, tl.int32)
        if HAS_STATE:
            memory = tl.load(
                initial_s + batch * S_STRIDES[0]
                + v[:, None] * S_STRIDES[1] + k[None, :] * S_STRIDES[2]
            ).to(tl.float32)
            inverse = tl.load(
                initial_p + batch * P_STRIDES[0]
                + k[:, None] * P_STRIDES[1] + k[None, :] * P_STRIDES[2]
            ).to(tl.float32)
            status = status | tl.where(_all_finite(memory) & _all_finite(inverse), 0, 1)
        else:
            memory = tl.full((VALUE_TILE, 32), 0.0, tl.float32)
            inverse = tl.where(k[:, None] == k[None, :], INVERSE_RIDGE, 0.0).to(tl.float32)
        if PREVIOUS_PROVIDED:
            prior = tl.load(
                previous_key + batch * PREVIOUS_STRIDES[0] + k * PREVIOUS_STRIDES[1]
            ).to(tl.float32)
            status = status | tl.where(_all_finite(prior), 0, 1)
        else:
            prior = tl.full((32,), 0.0, tl.float32)

        for step in tl.range(0, SEQUENCE, num_stages=1):
            query = tl.load(
                keys + batch * KEY_STRIDES[0] + step * KEY_STRIDES[1] + k * KEY_STRIDES[2]
            ).to(tl.float32)
            value = tl.load(
                values + batch * VALUE_STRIDES[0]
                + step * VALUE_STRIDES[1] + v * VALUE_STRIDES[2]
            ).to(tl.float32)
            # Validate even the first observed value, which has no write on a fresh stream.
            status = status | tl.where(_all_finite(query) & _all_finite(value), 0, 1)
            if HAS_PREVIOUS or step > 0:
                projected = tl.sum(inverse * prior[None, :], axis=1)
                quadratic = tl.sum(prior * projected, axis=0)
                denominator = 1.0 + quadratic
                status = status | tl.where(
                    _all_finite(denominator) & (denominator > 0.0), 0, 2
                )
                status = status | tl.where(quadratic >= 0.0, 0, 4)
                gain = projected / denominator
                predicted = tl.sum(memory * prior[None, :], axis=1)
                residual = value - predicted
                memory = memory + residual[:, None] * gain[None, :]
                inverse = inverse - projected[:, None] * gain[None, :]
                valid_update = (
                    _all_finite(projected) & _all_finite(gain)
                    & _all_finite(predicted) & _all_finite(residual)
                    & _all_finite(memory) & _all_finite(inverse)
                )
                status = status | tl.where(valid_update, 0, 8)
            read = tl.sum(memory * query[None, :], axis=1)
            status = status | tl.where(_all_finite(read), 0, 8)
            tl.store(reads + (batch * SEQUENCE + step) * 512 + v, read)
            prior = query

        tl.store(final_s + batch * 512 * 32 + v[:, None] * 32 + k[None, :], memory)
        if tile == 0:
            tl.store(final_p + batch * 32 * 32 + k[:, None] * 32 + k[None, :], inverse)
            tl.store(last_key + batch * 32 + k, prior)
        tl.store(statuses + batch * (512 // VALUE_TILE) + tile, status)


def _cpu_sequence_reads(
    keys: Tensor,
    values: Tensor,
    state: rls_memory.RLSState | None,
    previous_key: Tensor | None,
    has_previous: bool,
    ridge: float,
) -> tuple[Tensor, rls_memory.RLSState, Tensor]:
    # The primitive checks used vectors; check all inputs here as well, including
    # an unused first value or an explicitly provided but inactive previous key.
    inputs = [keys, values]
    if state is not None:
        inputs.extend((state.S, state.P))
    if previous_key is not None:
        inputs.append(previous_key)
    if not bool(torch.stack([torch.isfinite(tensor).all() for tensor in inputs]).all()):
        raise ValueError("RLS sequence inputs must be finite")
    batch, sequence, _ = keys.shape
    current = state if state is not None else rls_memory.initial_state(
        batch, _KEY_DIM, _VALUE_DIM, device=keys.device, ridge=ridge
    )
    reads = torch.empty((batch, sequence, _VALUE_DIM), dtype=torch.float32, device=keys.device)
    prior = previous_key
    for step in range(sequence):
        if has_previous or step > 0:
            current = rls_memory.write(current, prior, values[:, step])
        reads[:, step] = rls_memory.read(current, keys[:, step])
        prior = keys[:, step]
    if current is state:
        # A one-position chunk without a previous key performs no write, but
        # still returns independently owned state just like the CUDA path.
        current = rls_memory.RLSState(S=current.S.clone(), P=current.P.clone())
    return reads, current, keys[:, -1].clone()


@torch.no_grad()
def sequence_reads(
    keys: Tensor,
    values: Tensor,
    *,
    state: rls_memory.RLSState | None = None,
    previous_key: Tensor | None = None,
    has_previous: bool = False,
    ridge: float = 0.25,
) -> tuple[Tensor, rls_memory.RLSState, Tensor]:
    """Return (reads[B,T,512], final RLSState, last_key[B,32]).

    Initial state and previous-key presence are independent: an existing state
    can be read without an immediate write, or an initial empty state can accept
    a known previous key. has_previous=True requires previous_key. Ridge only
    initializes a missing state. Pass returned state/last_key and
    has_previous=True to continue a stream with identical causal ordering.
    No inputs may require gradients; detach native features/embeddings first.
    """
    if not isinstance(keys, Tensor):
        raise TypeError("keys must be a torch.Tensor")
    if keys.ndim != 3 or keys.shape[-1] != _KEY_DIM:
        raise ValueError("keys must have shape [batch, sequence, 32]")
    batch, sequence, _ = keys.shape
    if batch <= 0 or sequence <= 0:
        raise ValueError("batch and sequence dimensions must be positive")
    if keys.device.type not in ("cpu", "cuda"):
        raise ValueError("RLS sequences support only CPU or CUDA")
    _check_tensor("keys", keys, (batch, sequence, _KEY_DIM), keys.device)
    _check_tensor("values", values, (batch, sequence, _VALUE_DIM), keys.device)
    if not isinstance(has_previous, bool):
        raise TypeError("has_previous must be a bool")
    inverse_ridge = _inverse_ridge(ridge)
    if state is not None:
        if not isinstance(state, rls_memory.RLSState):
            raise TypeError("state must be an rls_memory.RLSState")
        _check_tensor("state.S", state.S, (batch, _VALUE_DIM, _KEY_DIM), keys.device)
        _check_tensor("state.P", state.P, (batch, _KEY_DIM, _KEY_DIM), keys.device)
    if previous_key is not None:
        _check_tensor("previous_key", previous_key, (batch, _KEY_DIM), keys.device)
    elif has_previous:
        raise ValueError("has_previous=True requires previous_key")

    with torch.autocast(device_type=keys.device.type, enabled=False):
        if keys.device.type == "cpu":
            return _cpu_sequence_reads(keys, values, state, previous_key, has_previous, ridge)
        if triton is None:
            raise RuntimeError("CUDA RLS sequence execution requires Triton")
        reads = torch.empty((batch, sequence, _VALUE_DIM), dtype=torch.float32, device=keys.device)
        final_s = torch.empty((batch, _VALUE_DIM, _KEY_DIM), dtype=torch.float32, device=keys.device)
        final_p = torch.empty((batch, _KEY_DIM, _KEY_DIM), dtype=torch.float32, device=keys.device)
        last_key = torch.empty((batch, _KEY_DIM), dtype=torch.float32, device=keys.device)
        statuses = torch.empty((batch, _VALUE_DIM // _VALUE_TILE), dtype=torch.int32, device=keys.device)
        # Unused pointers are erased by constexpr branches, not read as state.
        initial_s = keys if state is None else state.S
        initial_p = keys if state is None else state.P
        prior = keys if previous_key is None else previous_key
        with torch.cuda.device(keys.device):
            _sequence_reads_kernel[(batch, _VALUE_DIM // _VALUE_TILE)](
                keys, values, initial_s, initial_p, prior,
                reads, final_s, final_p, last_key, statuses,
                KEY_STRIDES=keys.stride(),
                VALUE_STRIDES=values.stride(),
                S_STRIDES=initial_s.stride(),
                P_STRIDES=initial_p.stride(),
                PREVIOUS_STRIDES=prior.stride(),
                SEQUENCE=sequence,
                INVERSE_RIDGE=inverse_ridge,
                HAS_STATE=state is not None,
                HAS_PREVIOUS=has_previous,
                PREVIOUS_PROVIDED=previous_key is not None,
                VALUE_TILE=_VALUE_TILE,
                num_warps=4,
                enable_fp_fusion=False,
            )
        if bool(statuses.any().item()):
            raise ValueError(
                "RLS CUDA scan encountered nonfinite input/arithmetic, a nonfinite or "
                "nonpositive denominator, or a negative key.T @ P @ key"
            )
        return reads, rls_memory.RLSState(S=final_s, P=final_p), last_key
