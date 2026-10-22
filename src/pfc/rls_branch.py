"""Frozen-address, full-embedding additive memory for the unchanged native model.

Construct ``branch = AdditiveRLSBranch(base_model)``; ``branch.base_model is
base_model``. Native training continues through the original trainer. Optimize
only ``[branch.gain]`` after ``branch.assign_gain_gradient(output, targets)``;
never pass ``branch.parameters()`` to the gain optimizer. All branch execution
and scalar credit are explicitly outside autograd.

``forward``/``prefill`` consume [B,T] observed token IDs; ``step`` consumes [B].
Their dictionaries contain candidate ``logits``, unchanged ``native_logits``,
unscaled ``branch_logits``, and a functional ``state``. Pass that state into the
next call for chunking; omit it to start independent contexts. No targets or
masks enter the forward path. Persistent branch tensors total 69,760 bytes per
batch lane (float32 S, P and previous key), independent of context length.

Save the native model/trainer normally and save ``branch.branch_state_dict()``
plus the external gain optimizer state separately. Reconstruct around that same
native model and call ``load_branch_state_dict`` to restore only gain/projection
in place. Standard full ``state_dict`` / ``load_state_dict`` (assign=False) also
work, including native weights, without replacing the base_model object.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .rls_memory import RLSState, state_bytes as rls_state_bytes
from .rls_sequence import sequence_reads


@dataclass(frozen=True)
class RLSBranchState:
    """Caller-owned state; calls never modify its branch tensors in place."""

    native_state: Any = None
    rls_state: RLSState | None = None
    previous_key: Tensor | None = None
    has_previous: bool = False


class AdditiveRLSBranch(nn.Module):
    """One learned scalar and one independently seeded fixed projection."""

    def __init__(
        self,
        base_model: nn.Module,
        key_dim: int = 32,
        ridge: float = 0.25,
        projection_seed: int = 20261021,
    ) -> None:
        super().__init__()
        if key_dim != 32:
            raise ValueError("the sequence implementation requires key_dim=32")
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        embedding = base_model.embedding.weight
        if embedding.ndim != 2 or embedding.shape[1] != 512:
            raise ValueError("the accepted native model must have 512D embeddings")
        self.base_model = base_model
        self.ridge = float(ridge)
        # CPU-local generation leaves both native CPU and device RNG untouched.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(projection_seed)
        random_basis = torch.randn(512, key_dim, generator=generator, dtype=torch.float32)
        projection, _ = torch.linalg.qr(random_basis, mode="reduced")
        self.register_buffer("projection", projection.to(device=embedding.device))
        self.gain = nn.Parameter(torch.zeros((), device=embedding.device, dtype=torch.float32))

    @torch.no_grad()
    def _readout(
        self,
        tokens: Tensor,
        features: Tensor,
        native_logits: Tensor,
        native_state: Any,
        state: RLSBranchState,
    ) -> dict[str, Any]:
        # Autocast must not reduce precision in the float32 RLS addresses/values.
        with torch.autocast(device_type=features.device.type, enabled=False):
            projected = features.detach().float() @ self.projection.float()
            norms = torch.linalg.vector_norm(projected, dim=-1, keepdim=True)
            if not bool((torch.isfinite(norms) & (norms > 0)).all().item()):
                raise FloatingPointError("nonfinite or zero projected native feature norm")
            keys = projected / norms
            values = self.base_model.embedding(tokens).detach().float()
            reads, final_rls, last_key = sequence_reads(
                keys,
                values,
                state=state.rls_state,
                previous_key=state.previous_key,
                has_previous=state.has_previous,
                ridge=self.ridge,
            )
            branch_logits = self.base_model.decode_features(
                reads.to(dtype=self.base_model.embedding.weight.dtype)
            )
            logits = native_logits + self.gain * branch_logits
        finite = (
            torch.isfinite(native_logits).all()
            & torch.isfinite(branch_logits).all()
            & torch.isfinite(logits).all()
            & torch.isfinite(self.gain).all()
        )
        if not bool(finite.item()):
            raise FloatingPointError("nonfinite native/additive logits or gain")
        return {
            "logits": logits,
            "native_logits": native_logits,
            "branch_logits": branch_logits,
            "state": RLSBranchState(native_state, final_rls, last_key, True),
        }

    @torch.no_grad()
    def forward(
        self, tokens: Tensor, state: RLSBranchState | None = None
    ) -> dict[str, Any]:
        """Return [B,T,V] logits and next state, with no target-dependent work."""
        return self.prefill(tokens, state)

    @torch.no_grad()
    def prefill(
        self, tokens: Tensor, state: RLSBranchState | None = None
    ) -> dict[str, Any]:
        if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("tokens must have nonempty shape [batch,time]")
        if state is None:
            state = RLSBranchState()
        if not isinstance(state, RLSBranchState):
            raise TypeError("state must be RLSBranchState or None")
        _, _, settled, native_state = self.base_model._prefill_states(
            tokens, state.native_state
        )
        features, native_logits = self.base_model._prediction_features_and_logits(
            settled, token_ids=tokens
        )
        return self._readout(tokens, features, native_logits, native_state, state)

    @torch.no_grad()
    def step(
        self, tokens: Tensor, state: RLSBranchState | None = None
    ) -> dict[str, Any]:
        """Return [B,V] logits and next state for one observed token per lane."""
        if tokens.ndim != 1 or tokens.shape[0] == 0:
            raise ValueError("tokens must have nonempty shape [batch]")
        if state is None:
            state = RLSBranchState()
        if not isinstance(state, RLSBranchState):
            raise TypeError("state must be RLSBranchState or None")
        native_logits, features, native_state = self.base_model.step_with_prediction_features(
            tokens, state.native_state
        )
        output = self._readout(
            tokens.unsqueeze(1),
            features.unsqueeze(1),
            native_logits.unsqueeze(1),
            native_state,
            state,
        )
        for name in ("logits", "native_logits", "branch_logits"):
            output[name] = output[name].squeeze(1)
        return output

    @torch.no_grad()
    def assign_gain_gradient(
        self,
        output: Mapping[str, Any],
        targets: Tensor,
        *,
        valid_mask: Tensor | None = None,
        ignore_index: int = -100,
    ) -> dict[str, Any]:
        """Assign d(mean valid CE)/d(gain), replacing rather than accumulating.

        Credit is mean_valid(sum_v softmax(total)_v * branch_v - branch_target).
        ``total`` is the actual saved candidate output, not a native-only softmax
        or a recomputation with a potentially changed gain. The mean excludes
        ignore_index and false mask entries identically for loss and gradient.
        Return loss/gradient scalar tensors and an integer valid_count. There is
        no optimizer step, native gradient mutation, backward pass or VJP here.
        """
        logits = output["logits"].detach()
        branch_logits = output["branch_logits"].detach()
        if logits.shape != branch_logits.shape or logits.shape[:-1] != targets.shape:
            raise ValueError("logits/branch_logits and targets must have matching token axes")
        if targets.dtype != torch.long or targets.device != logits.device:
            raise ValueError("targets must be int64 on the logits device")
        valid = targets.ne(ignore_index)
        if valid_mask is not None:
            if (
                valid_mask.shape != targets.shape
                or valid_mask.dtype != torch.bool
                or valid_mask.device != targets.device
            ):
                raise ValueError("valid_mask must be boolean with the targets shape/device")
            valid = valid & valid_mask
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            raise ValueError("the gain objective requires at least one valid target")
        safe_targets = targets.masked_fill(~valid, 0).unsqueeze(-1)
        if bool(((safe_targets < 0) | (safe_targets >= logits.shape[-1])).any().item()):
            raise ValueError("valid targets are outside the output vocabulary")
        with torch.autocast(device_type=logits.device.type, enabled=False):
            log_probabilities = torch.log_softmax(logits.float(), dim=-1)
            loss = -log_probabilities.gather(-1, safe_targets).squeeze(-1)[valid].mean()
            branch_target = branch_logits.gather(-1, safe_targets).squeeze(-1).float()
            # Reuse the softmax workspace; never modify either saved logit tensor.
            expected_branch = log_probabilities.exp_().mul_(branch_logits.float()).sum(dim=-1)
            gradient = (expected_branch - branch_target)[valid].mean()
        if not bool((torch.isfinite(loss) & torch.isfinite(gradient)).item()):
            raise FloatingPointError("nonfinite gain objective or output-local credit")
        self.gain.grad = gradient.to(device=self.gain.device, dtype=self.gain.dtype).clone()
        return {"loss": loss, "gradient": gradient, "valid_count": valid_count}

    @torch.no_grad()
    def branch_state_dict(self) -> dict[str, Tensor]:
        """Snapshot only new slow state, excluding native weights and fast state."""
        return {"gain": self.gain.detach().clone(), "projection": self.projection.detach().clone()}

    @torch.no_grad()
    def load_branch_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore new slow state in place, retaining model/parameter identities."""
        if set(state) != {"gain", "projection"}:
            raise ValueError("branch checkpoint must contain exactly gain and projection")
        for name, destination in (("gain", self.gain), ("projection", self.projection)):
            source = state[name]
            if not isinstance(source, Tensor) or source.shape != destination.shape:
                raise ValueError(f"invalid {name} checkpoint tensor shape")
            if source.dtype != torch.float32 or not bool(torch.isfinite(source).all().item()):
                raise ValueError(f"{name} checkpoint must be finite float32")
        self.gain.copy_(state["gain"])
        self.projection.copy_(state["projection"])

    def slow_state_bytes(self) -> dict[str, int]:
        """New persistent tensor payload only; excludes native and optimizer state."""
        return {
            "gain": self.gain.numel() * self.gain.element_size(),
            "projection": self.projection.numel() * self.projection.element_size(),
        }

    @staticmethod
    def state_bytes(state: RLSBranchState) -> int:
        """Added fast-state payload; excludes native state/Python metadata."""
        total = 0 if state.rls_state is None else rls_state_bytes(state.rls_state)
        if state.previous_key is not None:
            total += state.previous_key.numel() * state.previous_key.element_size()
        return total
