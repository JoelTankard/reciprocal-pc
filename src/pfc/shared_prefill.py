"""Reuse one target-independent native prefill in a Golden additive-RLS update.

Use only around ``branch(tokens)``, ``branch.assign_gain_gradient(...)`` and
``trainer.local_update(tokens, targets)``, in that order, before stepping the gain
optimizer. The first request runs the original native method with the trainer's
return flags; the branch receives its original four entries and the trainer
receives the complete original tuple. No weights, losses, credit or guards are
replaced. Exactly one completed native update is required.

Only the accepted Phase827 configuration is supported (with either existing
value of reuse_prefill_associative_reads). That flag merely retains already
computed reads, zero-fills inactive layers, and stacks them; it does not change
forward arithmetic or consume RNG. The other three return flags are false.
Causal attention exploration is disabled in this configuration. Its model state
must be None at entry; the native trainer still performs its own setup/reset.

Mutation checks use PyTorch version counters, identity and storage metadata, not
copies or tensor comparisons. Inference-mode tensors and untracked writes via
.data, raw storage, external kernels or concurrent threads are not supported.
The caller must not perform such writes while this context is active. This is a
single-thread, eager, one-update scope, not a persistent or concurrent cache.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
import math
from threading import get_ident
from typing import Iterator

import torch
from torch import Tensor


__all__ = ["shared_prefill", "golden_local_update"]


class _UnsupportedPrefill(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _UnsupportedPrefill(f"shared_prefill: {message}")


def _tensor_stamp(tensor: Tensor) -> tuple:
    _require(type(tensor) in (Tensor, torch.nn.Parameter), "tensor subclasses are unsupported")
    _require(tensor.layout == torch.strided, "only strided tensors are supported")
    try:
        version = tensor._version
    except RuntimeError as error:
        raise _UnsupportedPrefill(
            "shared_prefill: inference tensors lack mutation version counters"
        ) from error
    return (
        id(tensor), version, tensor.data_ptr(), tuple(tensor.shape),
        tuple(tensor.stride()), tensor.storage_offset(), tensor.dtype,
        tensor.device, tensor.requires_grad,
    )


def _tree_stamp(value: object) -> tuple:
    if isinstance(value, Tensor):
        return ("tensor", _tensor_stamp(value))
    if value is None or type(value) in (bool, int, float, str):
        return (type(value), value)
    if isinstance(value, tuple):
        return ("tuple", id(value), tuple(_tree_stamp(item) for item in value))
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value), id(value),
            tuple((field.name, _tree_stamp(getattr(value, field.name))) for field in fields(value)),
        )
    raise _UnsupportedPrefill(f"shared_prefill: unsupported prefill state {type(value).__name__}")


def _config_stamp(config: object) -> tuple:
    return tuple((field.name, getattr(config, field.name)) for field in fields(config))


def _model_stamp(model: torch.nn.Module) -> tuple:
    return (
        tuple((name, _tensor_stamp(value)) for name, value in model.named_parameters()),
        tuple((name, _tensor_stamp(value)) for name, value in model.named_buffers()),
        tuple((name, id(module), module.training) for name, module in model.named_modules()),
    )


def _execution_stamp(device: torch.device) -> tuple:
    return (
        torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
        torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type),
        torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None,
    )


@contextmanager
def shared_prefill(model: torch.nn.Module, trainer: object, tokens: Tensor) -> Iterator[None]:
    """Share the actual native prefill between branch and trainer, or raise.

    Existing instance wrappers (for example an execution counter) are called for
    the one real prefill and restored exactly, including on exceptional exits.
    Non-None recurrent input states, changed tensors/configuration/execution
    modes, nested scopes, additional requests and incomplete updates are errors.
    Errors do not roll back an update: a caller must discard a failed update just
    as it would after any native training exception.
    """
    from .native_pc_lm import NativePCState, NativeReciprocalPCLanguageModel
    from .native_pc_training import NativePCTrainer
    from .reciprocal_pc import model_config, training_config

    _require(type(model) is NativeReciprocalPCLanguageModel, "unsupported model implementation")
    _require(type(trainer) is NativePCTrainer, "unsupported trainer implementation")
    _require(trainer.model is model, "trainer must own this model")
    _require(model.config == model_config(), "unsupported model configuration")
    reuse_reads = trainer.config.reuse_prefill_associative_reads
    _require(type(reuse_reads) is bool, "associative-read return flag must be boolean")
    _require(
        trainer.config == replace(training_config(), reuse_prefill_associative_reads=reuse_reads),
        "unsupported trainer configuration",
    )
    _require(model.training, "model must be in training mode")
    _require(not torch.is_inference_mode_enabled(), "inference mode is unsupported")
    _require(isinstance(tokens, Tensor), "tokens must be a tensor")
    _require(
        tokens.ndim == 2 and tokens.shape[0] > 0 and tokens.shape[1] >= 2,
        "tokens must have nonempty [batch, time >= 2] shape",
    )
    _require(tokens.dtype == torch.long and tokens.device == model.device,
             "tokens must be int64 on the model device")
    _require(model._bounded_attention_causal_exploration_step is None,
             "causal exploration state must be None before the branch")

    original = model._prefill_states
    _require(callable(original), "native prefill must be callable")
    _require(not getattr(original, "_shared_prefill_context", False), "nested scopes are unsupported")
    had_instance_method = "_prefill_states" in model.__dict__
    instance_method = model.__dict__.get("_prefill_states")
    owner_thread = get_ident()
    token_stamp = _tensor_stamp(tokens)
    model_state = _model_stamp(model)
    model_config_object = model.config
    trainer_config_object = trainer.config
    model_configuration = _config_stamp(model.config)
    trainer_configuration = _config_stamp(trainer.config)
    update_counter = trainer._update_count
    update_stamp = _tensor_stamp(update_counter)
    calls = 0
    failed = False
    cached = None
    cached_stamp = None
    execution = None
    trainer_flags = (reuse_reads, False, False, False)

    def check_shared_state() -> None:
        _require(get_ident() == owner_thread, "cross-thread requests are unsupported")
        _require(model.__dict__.get("_prefill_states") is prefill, "prefill method changed inside scope")
        _require(trainer.model is model, "trainer model changed inside scope")
        _require(
            model.config is model_config_object and trainer.config is trainer_config_object
            and _config_stamp(model.config) == model_configuration
            and _config_stamp(trainer.config) == trainer_configuration,
            "configuration changed inside scope",
        )
        _require(_tensor_stamp(tokens) == token_stamp, "tokens mutated inside scope")
        _require(_model_stamp(model) == model_state, "model tensors or training modes changed before reuse")
        _require(trainer._update_count is update_counter and _tensor_stamp(update_counter) == update_stamp,
                 "native update state changed before reuse")
        _require(model._bounded_attention_causal_exploration_step is None,
                 "causal exploration state changed before reuse")
        _require(not torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
                 "prefill requests must run under native no_grad, not inference mode")

    def prefill(
        token_ids: Tensor,
        state: object,
        *,
        return_associative_reads: bool = False,
        return_swiglu_lexical_features: bool = False,
        return_recurrent_sequences: bool = False,
        return_lagged_latent_errors: bool = False,
    ) -> tuple:
        nonlocal calls, failed, cached, cached_stamp, execution
        try:
            check_shared_state()
            _require(token_ids is tokens, "both requests must use the identical tokens tensor")
            _require(state is None, "recurrent input state is unsupported")
            flags = (
                return_associative_reads, return_swiglu_lexical_features,
                return_recurrent_sequences, return_lagged_latent_errors,
            )
            _require(all(type(flag) is bool for flag in flags), "return flags must be boolean")
            if calls == 0:
                _require(flags == (False, False, False, False), "first request must be the branch prefill")
                execution = _execution_stamp(tokens.device)
                calls = -1  # Reject reentrant use while the native method is running.
                cached = original(
                    token_ids, None,
                    return_associative_reads=reuse_reads,
                    return_swiglu_lexical_features=False,
                    return_recurrent_sequences=False,
                    return_lagged_latent_errors=False,
                )
                _require(type(cached) is tuple and len(cached) == 4 + int(reuse_reads),
                         "native prefill returned an unsupported tuple")
                _require(all(isinstance(value, Tensor) for value in cached[:3]),
                         "native prefill tensors are missing")
                _require(type(cached[3]) is NativePCState, "native final state is missing")
                if reuse_reads:
                    _require(isinstance(cached[4], Tensor), "native associative reads are missing")
                check_shared_state()
                _require(_execution_stamp(tokens.device) == execution, "native prefill changed execution mode")
                cached_stamp = _tree_stamp(cached)
                calls = 1
                return cached[:4]
            _require(calls == 1 and cached is not None, "expected exactly branch then trainer prefill")
            _require(flags == trainer_flags, "second request must use the trainer's exact return flags")
            _require(_execution_stamp(tokens.device) == execution, "execution mode changed before reuse")
            _require(_tree_stamp(cached) == cached_stamp, "cached tensors or final state mutated before reuse")
            result = cached
            cached = cached_stamp = None
            calls = 2
            return result
        except BaseException:
            failed = True
            raise

    prefill._shared_prefill_context = True
    model._prefill_states = prefill
    try:
        yield
        _require(not failed, "a failed prefill request was caught inside scope")
        _require(calls == 2, "scope must consume branch and trainer prefill exactly once")
        _require(get_ident() == owner_thread, "scope exited on a different thread")
        _require(model.__dict__.get("_prefill_states") is prefill, "prefill method changed inside scope")
        _require(trainer.model is model and model.config is model_config_object
                 and trainer.config is trainer_config_object
                 and _config_stamp(model.config) == model_configuration
                 and _config_stamp(trainer.config) == trainer_configuration,
                 "configuration changed inside scope")
        _require(_tensor_stamp(tokens) == token_stamp, "tokens mutated during native update")
        _require(trainer._update_count is update_counter
                 and update_counter._version == update_stamp[1] + 1,
                 "scope must complete exactly one native update")
        _require(model._bounded_attention_causal_exploration_step is None,
                 "native trainer did not reset causal exploration state")
    finally:
        if had_instance_method:
            model._prefill_states = instance_method
        else:
            delattr(model, "_prefill_states")
        cached = cached_stamp = model_state = execution = None


def golden_local_update(
    branch,
    trainer,
    gain_optimizer: torch.optim.Optimizer,
    tokens: Tensor,
    targets: Tensor,
) -> dict:
    """Perform one unaveraged strict-local native and output-local gain update.

    The gain optimizer must own only ``branch.gain``. Native metrics and finite
    credit guards are preserved; a failed update is not rolled back. Promoted
    inference-only weights must never be used as a resumable training state.
    """
    from .finite_metric_guard import require_aggregate_strict_finite
    from .rls_branch import AdditiveRLSBranch

    if not isinstance(branch, AdditiveRLSBranch) or trainer.model is not branch.base_model:
        raise ValueError("trainer must own the branch's unchanged native model")
    if getattr(branch, "_golden_inference_only", False):
        raise ValueError("Golden inference-only weights cannot resume training")
    parameters = [parameter for group in gain_optimizer.param_groups for parameter in group["params"]]
    if len(parameters) != 1 or parameters[0] is not branch.gain:
        raise ValueError("gain optimizer must own only branch.gain")
    gain_optimizer.zero_grad(set_to_none=True)
    with shared_prefill(branch.base_model, trainer, tokens):
        output = branch(tokens)
        objective = branch.assign_gain_gradient(output, targets)
        del output
        metrics = trainer.local_update(tokens, targets)
        require_aggregate_strict_finite(metrics)
        for name in ("nonfinite_local_credit_events_this_update", "nonfinite_local_credit_event_count"):
            if name not in metrics or int(metrics[name]) != 0:
                raise FloatingPointError(f"nonfinite native credit guard failed: {name}")
        if any(isinstance(value, float) and not math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("nonfinite host metric")
    gain_optimizer.step()
    if not bool(torch.isfinite(branch.gain).all().item()):
        raise FloatingPointError("nonfinite gain after optimizer step")
    for name, value in gain_optimizer.state.get(branch.gain, {}).items():
        if isinstance(value, Tensor) and not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"nonfinite gain optimizer state: {name}")
    return {"native": metrics, "gain": objective}
