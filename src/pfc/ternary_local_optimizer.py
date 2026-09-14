from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import torch
from torch import Tensor



TERNARY_STATES = (-1, 0, 1)
DST_TRANSITION_FACTOR = 3.0
TWN_THRESHOLD_FACTOR = 0.75
MANTISSA_PLANE_COUNT = 6
MANTISSA_MIN_EXPONENT = -8
MANTISSA_MAX_EXPONENT = 8
GROUPED_ABSMEAN_ALGORITHM = "ternary_qat_absmean_g128"
GROUPED_ABSMEAN_SIZE = 128
SYMMETRIC_INT8_ALGORITHM = "int8_qat_symmetric_per_token"
MIXED_LOW_BIT_ENDPOINT_OPTIMIZER = "ternary_int8_endpoint_qat"
DIRECT_TERNARY_ALGORITHMS = ("ternary_dst", "ternary_mantissa6")
TERNARY_ALGORITHMS = (
    *DIRECT_TERNARY_ALGORITHMS,
    "ternary_qat_carrier",
    GROUPED_ABSMEAN_ALGORITHM,
)
LOW_BIT_ALGORITHMS = (*TERNARY_ALGORITHMS, SYMMETRIC_INT8_ALGORITHM)


def pack_ternary_states(states: Tensor) -> tuple[Tensor, int]:
    """Pack four {-1, 0, +1} states into each byte."""

    integer_states = states.detach().to(dtype=torch.int8).reshape(-1)
    if not torch.equal(integer_states, states.detach().reshape(-1)):
        raise ValueError("ternary pack input contains non-integral values")
    if bool(((integer_states < -1) | (integer_states > 1)).any()):
        raise ValueError("ternary pack input lies outside {-1, 0, +1}")
    value_count = integer_states.numel()
    padding = (-value_count) % 4
    codes = integer_states.to(torch.uint8).add_(1)
    if padding:
        codes = torch.cat(
            (
                codes,
                torch.ones(padding, device=codes.device, dtype=torch.uint8),
            )
        )
    codes = codes.reshape(-1, 4)
    packed = (
        codes[:, 0]
        | torch.bitwise_left_shift(codes[:, 1], 2)
        | torch.bitwise_left_shift(codes[:, 2], 4)
        | torch.bitwise_left_shift(codes[:, 3], 6)
    )
    return packed, value_count


def unpack_ternary_states(
    packed: Tensor,
    value_count: int,
    *,
    shape: tuple[int, ...] | torch.Size | None = None,
) -> Tensor:
    """Restore exact {-1, 0, +1} states from two-bit codes."""

    if packed.dtype != torch.uint8 or packed.ndim != 1:
        raise ValueError("packed ternary values must be a flat uint8 tensor")
    if value_count < 0 or value_count > 4 * packed.numel():
        raise ValueError("packed ternary value count is invalid")
    shifts = torch.tensor((0, 2, 4, 6), device=packed.device, dtype=torch.uint8)
    codes = torch.bitwise_and(
        torch.bitwise_right_shift(packed.unsqueeze(1), shifts),
        0b11,
    ).reshape(-1)[:value_count]
    if bool((codes == 0b11).any()):
        raise ValueError("packed ternary stream contains reserved code 3")
    states = codes.to(torch.int8).sub_(1)
    if shape is not None:
        expected = math.prod(shape)
        if expected != value_count:
            raise ValueError("requested ternary shape has wrong value count")
        states = states.reshape(shape)
    return states


def _broadcast_scale(parameter: Tensor, scale: Tensor) -> Tensor:
    if scale.ndim != parameter.ndim:
        raise ValueError("ternary scale rank differs from parameter rank")
    for parameter_size, scale_size in zip(parameter.shape, scale.shape, strict=True):
        if scale_size not in (1, parameter_size):
            raise ValueError("ternary scale shape cannot broadcast to parameter")
    return scale


@torch.no_grad()
def project_parameter_to_ternary_(
    parameter: Tensor,
    *,
    reduce_dim: int,
) -> Tensor:
    """Project initialization to scaled ternary states without a shadow copy."""

    if parameter.ndim < 2:
        raise ValueError("ternary projection requires a matrix-like parameter")
    reduce_dim = reduce_dim % parameter.ndim
    absolute = parameter.detach().float().abs()
    base_scale = absolute.mean(dim=reduce_dim, keepdim=True)
    assigned = absolute >= (0.5 * base_scale)
    assigned = assigned & (base_scale > 0)
    assigned_count = assigned.sum(dim=reduce_dim, keepdim=True)
    assigned_sum = (absolute * assigned).sum(dim=reduce_dim, keepdim=True)
    fan_in = parameter.shape[reduce_dim]
    fallback = torch.full_like(base_scale, fan_in**-0.5)
    scale = torch.where(
        assigned_count > 0,
        assigned_sum / assigned_count.clamp_min(1),
        fallback,
    ).clamp_min(torch.finfo(torch.float32).tiny)
    states = torch.where(
        assigned,
        parameter.detach().float().sign(),
        torch.zeros((), device=parameter.device, dtype=torch.float32),
    )
    parameter.copy_((states * scale).to(parameter.dtype))
    return scale


def _initial_ternary_scale(parameter: Tensor, *, reduce_dim: int) -> Tensor:
    reduce_dim %= parameter.ndim
    absolute = parameter.detach().float().abs()
    base_scale = absolute.mean(dim=reduce_dim, keepdim=True)
    assigned = absolute >= (0.5 * base_scale)
    assigned = assigned & (base_scale > 0)
    assigned_count = assigned.sum(dim=reduce_dim, keepdim=True)
    assigned_sum = (absolute * assigned).sum(dim=reduce_dim, keepdim=True)
    fallback = torch.full_like(base_scale, parameter.shape[reduce_dim] ** -0.5)
    return torch.where(
        assigned_count > 0,
        assigned_sum / assigned_count.clamp_min(1),
        fallback,
    ).clamp_min(torch.finfo(torch.float32).tiny)


def _initial_qat_carrier_and_scale(
    parameter: Tensor,
    *,
    reduce_dim: int,
) -> tuple[Tensor, Tensor]:
    """Build Bonsai-style TWN codes and exact row-L2 scales."""

    reduce_dim %= parameter.ndim
    dense = parameter.detach().float()
    absolute = dense.abs()
    threshold = TWN_THRESHOLD_FACTOR * absolute.mean(
        dim=reduce_dim,
        keepdim=True,
    )
    assigned = absolute > threshold
    states = torch.where(assigned, dense.sign(), torch.zeros_like(dense))

    missing = states.abs().sum(dim=reduce_dim, keepdim=True) == 0
    if bool(missing.any()):
        fallback_index = absolute.argmax(dim=reduce_dim, keepdim=True)
        fallback_sign = dense.gather(reduce_dim, fallback_index).sign()
        fallback_sign = torch.where(
            fallback_sign == 0,
            torch.ones_like(fallback_sign),
            fallback_sign,
        )
        fallback = torch.zeros_like(dense).scatter(
            reduce_dim,
            fallback_index,
            fallback_sign,
        )
        states = torch.where(missing.expand_as(states), fallback, states)

    scale = torch.linalg.vector_norm(
        dense,
        dim=reduce_dim,
        keepdim=True,
    )
    return states, scale


def _reconstruct_normalized_ternary(
    states: Tensor,
    scale: Tensor,
    *,
    reduce_dim: int,
) -> Tensor:
    reduce_dim %= states.ndim
    states_float = states.float()
    norm = torch.linalg.vector_norm(
        states_float,
        dim=reduce_dim,
        keepdim=True,
    ).clamp_min(1.0)
    return (states_float / norm) * _broadcast_scale(states, scale)


def grouped_absmean_quantize(
    carrier: Tensor,
    *,
    group_size: int = GROUPED_ABSMEAN_SIZE,
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize contiguous carrier groups with non-learned absmean scales."""

    if carrier.ndim < 2:
        raise ValueError("grouped ternary QAT requires a matrix-like carrier")
    if group_size <= 0:
        raise ValueError("ternary group size must be positive")
    flat = carrier.detach().float().reshape(-1)
    value_count = flat.numel()
    group_count = (value_count + group_size - 1) // group_size
    padding = group_count * group_size - value_count
    if padding:
        flat = torch.cat((flat, flat.new_zeros(padding)))
    groups = flat.reshape(group_count, group_size)
    counts = torch.full(
        (group_count, 1),
        group_size,
        device=carrier.device,
        dtype=torch.float32,
    )
    if padding:
        counts[-1, 0] = group_size - padding
    scales = (groups.abs().sum(dim=1, keepdim=True) / counts).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    states = torch.round((groups / scales).clamp_(-1.0, 1.0)).to(torch.int8)
    reconstruction = states.float() * scales
    return (
        states.reshape(-1)[:value_count].reshape_as(carrier),
        scales.reshape(-1),
        reconstruction.reshape(-1)[:value_count].reshape_as(carrier),
    )


def grouped_absmean_stochastic_quantize(
    carrier: Tensor,
    *,
    group_size: int = GROUPED_ABSMEAN_SIZE,
    uniform: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample ternary codes whose reconstruction mean is the clipped carrier."""

    if carrier.ndim < 2:
        raise ValueError("grouped ternary QAT requires a matrix-like carrier")
    if group_size <= 0:
        raise ValueError("ternary group size must be positive")
    flat = carrier.detach().float().reshape(-1)
    value_count = flat.numel()
    group_count = (value_count + group_size - 1) // group_size
    padding = group_count * group_size - value_count
    if padding:
        flat = torch.cat((flat, flat.new_zeros(padding)))
    groups = flat.reshape(group_count, group_size)
    counts = torch.full(
        (group_count, 1),
        group_size,
        device=carrier.device,
        dtype=torch.float32,
    )
    if padding:
        counts[-1, 0] = group_size - padding
    scales = (groups.abs().sum(dim=1, keepdim=True) / counts).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    normalized = (groups / scales).clamp_(-1.0, 1.0)
    if uniform is None:
        uniform = torch.rand(
            normalized.shape,
            device=normalized.device,
            dtype=torch.float32,
            generator=generator,
        )
    elif (
        uniform.shape != normalized.shape
        or uniform.device != normalized.device
        or not uniform.is_floating_point()
    ):
        raise ValueError("grouped stochastic ternary uniform layout differs")
    states = (
        normalized.sign()
        * (uniform.to(torch.float32) < normalized.abs()).to(torch.float32)
    ).to(torch.int8)
    reconstruction = states.float() * scales
    return (
        states.reshape(-1)[:value_count].reshape_as(carrier),
        scales.reshape(-1),
        reconstruction.reshape(-1)[:value_count].reshape_as(carrier),
    )


def _reconstruct_grouped_absmean_states(
    states: Tensor,
    scales: Tensor,
    *,
    group_size: int,
) -> Tensor:
    if group_size <= 0:
        raise ValueError("ternary group size must be positive")
    expanded = scales.reshape(-1).repeat_interleave(group_size)
    if expanded.numel() < states.numel():
        raise ValueError("grouped ternary scale count is too small")
    return (
        states.reshape(-1).float() * expanded[: states.numel()]
    ).reshape_as(states)


def _grouped_absmean_quantize_for_group(
    carrier: Tensor,
    group: dict[str, Any],
) -> tuple[Tensor, Tensor, Tensor]:
    if bool(group.get("grouped_symmetric_last_two_dims", False)):
        if carrier.ndim not in (2, 3) or carrier.shape[-2] != carrier.shape[-1]:
            raise ValueError("symmetric grouped ternary QAT requires square matrices")
        matrices = carrier.unsqueeze(0) if carrier.ndim == 2 else carrier
        if not torch.equal(matrices, matrices.transpose(-1, -2)):
            raise RuntimeError("symmetric grouped ternary carrier lost exact symmetry")
        width = matrices.shape[-1]
        upper = torch.triu_indices(width, width, device=carrier.device)
        states = torch.zeros_like(matrices, dtype=torch.int8)
        reconstruction = torch.zeros_like(matrices, dtype=torch.float32)
        scales: list[Tensor] = []
        for layer in range(matrices.shape[0]):
            values = matrices[layer, upper[0], upper[1]].unsqueeze(0)
            upper_states, layer_scales, upper_reconstruction = (
                grouped_absmean_quantize(
                    values,
                    group_size=int(group.get("group_size", GROUPED_ABSMEAN_SIZE)),
                )
            )
            upper_states = upper_states[0]
            upper_reconstruction = upper_reconstruction[0]
            states[layer, upper[0], upper[1]] = upper_states
            reconstruction[layer, upper[0], upper[1]] = upper_reconstruction
            states[layer, upper[1], upper[0]] = upper_states
            reconstruction[layer, upper[1], upper[0]] = upper_reconstruction
            scales.append(layer_scales)
        if carrier.ndim == 2:
            return states[0], torch.cat(scales), reconstruction[0]
        return states, torch.cat(scales), reconstruction
    if bool(group.get("grouped_pair_contract", False)):
        return pair_ternary_quantize(carrier)
    balanced_role = group.get("balanced_two_of_four_role")
    if balanced_role is not None:
        states, scales, reconstruction = grouped_absmean_quantize(
            carrier,
            group_size=int(group.get("group_size", GROUPED_ABSMEAN_SIZE)),
        )
        return (
            states,
            scales * BALANCED_TWO_OF_FOUR_COMPENSATION,
            reconstruction * BALANCED_TWO_OF_FOUR_COMPENSATION,
        )
    independent_slices = int(group.get("grouped_independent_leading_slices", 1))
    group_size = int(group.get("group_size", GROUPED_ABSMEAN_SIZE))
    if independent_slices == 1:
        return grouped_absmean_quantize(carrier, group_size=group_size)
    if carrier.shape[0] != independent_slices:
        raise ValueError("independent grouped slices differ from leading dimension")
    def quantize_slice(value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if value.ndim != 1:
            return grouped_absmean_quantize(value, group_size=group_size)
        states, scales, reconstruction = grouped_absmean_quantize(
            value.unsqueeze(0),
            group_size=group_size,
        )
        return states.squeeze(0), scales, reconstruction.squeeze(0)

    quantized = tuple(
        quantize_slice(carrier[index]) for index in range(independent_slices)
    )
    return (
        torch.stack(tuple(row[0] for row in quantized), dim=0),
        torch.cat(tuple(row[1] for row in quantized), dim=0),
        torch.stack(tuple(row[2] for row in quantized), dim=0),
    )


def symmetric_int8_quantize(
    carrier: Tensor,
    *,
    reduce_dim: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize each token vector with one symmetric signed-INT8 scale."""

    if carrier.ndim < 2:
        raise ValueError("symmetric INT8 QAT requires a matrix-like carrier")
    reduce_dim %= carrier.ndim
    dense = carrier.detach().float()
    maximum = dense.abs().amax(dim=reduce_dim, keepdim=True)
    scale = torch.where(
        maximum > 0,
        maximum / 127.0,
        torch.ones_like(maximum),
    )
    codes = torch.round((dense / scale).clamp_(-127.0, 127.0)).to(torch.int8)
    reconstruction = codes.float() * scale
    return codes, scale, reconstruction


def _adaptive_mantissa_scale(
    target: Tensor,
    base_scale: Tensor,
    *,
    reduce_dim: int,
) -> tuple[Tensor, Tensor]:
    representable_units = sum(
        3.0**-plane for plane in range(MANTISSA_PLANE_COUNT)
    )
    maximum = target.detach().float().abs().amax(
        dim=reduce_dim % target.ndim,
        keepdim=True,
    )
    ratio = maximum / (base_scale * representable_units)
    exponent = torch.where(
        maximum > 0,
        torch.ceil(
            torch.log(ratio.clamp_min(torch.finfo(torch.float32).tiny))
            / math.log(3.0)
        ),
        torch.zeros_like(maximum),
    ).clamp_(MANTISSA_MIN_EXPONENT, MANTISSA_MAX_EXPONENT)
    exponent = exponent.to(torch.int8)
    effective_scale = base_scale * torch.pow(
        torch.tensor(3.0, device=base_scale.device, dtype=torch.float32),
        exponent.float(),
    )
    return effective_scale, exponent


def _quantize_mantissa(
    target: Tensor,
    base_scale: Tensor,
    *,
    reduce_dim: int,
    uniform: Tensor | None = None,
) -> tuple[tuple[Tensor, ...], Tensor, Tensor]:
    target_float = target.detach().float()
    effective_scale, exponent = _adaptive_mantissa_scale(
        target_float,
        base_scale,
        reduce_dim=reduce_dim,
    )
    reconstruction = torch.zeros_like(target_float)
    digits: list[Tensor] = []
    for plane in range(MANTISSA_PLANE_COUNT):
        scale = effective_scale / (3**plane)
        quotient = (target_float - reconstruction) / scale
        if plane + 1 < MANTISSA_PLANE_COUNT or uniform is None:
            digit = torch.round(quotient).clamp_(-1, 1).to(torch.int8)
        else:
            probability = quotient.abs().clamp_(0.0, 1.0)
            digit = (
                (uniform < probability).to(torch.int8)
                * quotient.sign().to(torch.int8)
            )
        reconstruction.add_(digit.float() * scale)
        digits.append(digit)

    # Stochastic targets can have multiple exact exponent/digit encodings.
    # Re-encode the deployed value greedily so equal weights have equal state.
    canonical_scale, canonical_exponent = _adaptive_mantissa_scale(
        reconstruction,
        base_scale,
        reduce_dim=reduce_dim,
    )
    canonical_reconstruction = torch.zeros_like(reconstruction)
    canonical_digits: list[Tensor] = []
    for plane in range(MANTISSA_PLANE_COUNT):
        scale = canonical_scale / (3**plane)
        digit = torch.round(
            (reconstruction - canonical_reconstruction) / scale
        ).clamp_(-1, 1).to(torch.int8)
        canonical_reconstruction.add_(digit.float() * scale)
        canonical_digits.append(digit)
    delta = (canonical_reconstruction - reconstruction).abs()
    tolerance = canonical_scale / (3 ** (MANTISSA_PLANE_COUNT - 1))
    if bool((delta > tolerance).any()):
        raise RuntimeError(
            "ternary mantissa canonicalization exceeded one finest-plane unit"
        )
    return tuple(canonical_digits), canonical_exponent, canonical_reconstruction


class TernaryLocalOptimizer(torch.optim.Optimizer):
    """Local ternary optimizers plus AdamW for declared dense exceptions."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        if weight_decay < 0:
            raise ValueError("weight decay must be nonnegative")
        if not 0 <= adam_betas[0] < 1 or not 0 <= adam_betas[1] < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        if adam_eps <= 0:
            raise ValueError("Adam epsilon must be positive")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "algorithm": "ternary_dst",
            "scale_reduce_dim": -2,
            "betas": adam_betas,
            "eps": adam_eps,
        }
        super().__init__(params, defaults)
        self._transition_randomness_groups: dict[
            int,
            tuple[int, bool, tuple[int, ...]],
        ] = {}
        self._transition_request_groups: dict[
            int,
            tuple[int, bool, tuple[int, ...]],
        ] = {}
        self._transition_request_group_sizes: dict[int, int] = {}
        self._balanced_two_of_four_masks: dict[int, Tensor] = {}
        self._stochastic_rounding_generator: torch.Generator | None = None
        self._stochastic_rounding_rng_owner: Tensor | None = None
        for group in self.param_groups:
            algorithm = group["algorithm"]
            bf16_stochastic_update = bool(
                group.get("bf16_stochastic_update", False)
            )
            if bf16_stochastic_update and bool(
                group.get("unbiased_stochastic_rounding", False)
            ):
                raise ValueError(
                    "BF16 weight rounding and stochastic ternary codes are mutually exclusive"
                )
            if bf16_stochastic_update and algorithm not in (
                GROUPED_ABSMEAN_ALGORITHM,
                "adamw",
            ):
                raise ValueError(
                    "BF16 stochastic training state requires grouped QAT or AdamW"
                )
            if algorithm not in (*LOW_BIT_ALGORITHMS, "adamw"):
                raise ValueError(f"unsupported ternary parameter route: {algorithm}")
            if bool(group.get("codes_only_deployment", False)) and (
                algorithm != GROUPED_ABSMEAN_ALGORITHM
            ):
                raise ValueError(
                    "codes-only deployment requires grouped absmean ternary QAT"
                )
            if algorithm in LOW_BIT_ALGORITHMS:
                for parameter in group["params"]:
                    if parameter.ndim < 2:
                        raise ValueError("low-bit groups require matrix parameters")
                    state = self.state[parameter]
                    state["step"] = 0
                    state["transition_count"] = torch.zeros(
                        (), device=parameter.device, dtype=torch.int64
                    )
                    state["last_transition_count"] = torch.zeros(
                        (), device=parameter.device, dtype=torch.int64
                    )
                    if algorithm == "ternary_dst":
                        state["scale"] = project_parameter_to_ternary_(
                            parameter,
                            reduce_dim=int(group["scale_reduce_dim"]),
                        )
                    elif algorithm == "ternary_mantissa6":
                        base_scale = _initial_ternary_scale(
                            parameter,
                            reduce_dim=int(group["scale_reduce_dim"]),
                        )
                        digits, exponent, reconstruction = _quantize_mantissa(
                            parameter,
                            base_scale,
                            reduce_dim=int(group["scale_reduce_dim"]),
                        )
                        state["base_scale"] = base_scale
                        state["row_exponent"] = exponent
                        state["packed_planes"] = tuple(
                            pack_ternary_states(digit)[0] for digit in digits
                        )
                        with torch.no_grad():
                            parameter.copy_(reconstruction.to(parameter.dtype))
                    elif algorithm == GROUPED_ABSMEAN_ALGORITHM:
                        carrier = parameter.detach().float().clone()
                        balanced_role = group.get("balanced_two_of_four_role")
                        if balanced_role is not None:
                            mask = balanced_two_of_four_mask(
                                carrier,
                                role=int(balanced_role),
                            )
                            carrier.mul_(mask)
                            self._balanced_two_of_four_masks[id(parameter)] = mask
                        states, scale, reconstruction = (
                            _grouped_absmean_quantize_for_group(carrier, group)
                        )
                        state["carrier"] = carrier
                        state["scale"] = scale
                        state["exp_avg"] = torch.zeros_like(carrier)
                        state["exp_avg_sq"] = torch.zeros_like(carrier)
                        state["packed_states"] = pack_ternary_states(states)[0]
                        with torch.no_grad():
                            parameter.copy_(reconstruction.to(parameter.dtype))
                    elif algorithm == SYMMETRIC_INT8_ALGORITHM:
                        carrier = parameter.detach().float().clone()
                        codes, scale, reconstruction = symmetric_int8_quantize(
                            carrier,
                            reduce_dim=int(group["scale_reduce_dim"]),
                        )
                        state["carrier"] = carrier
                        state["scale"] = scale
                        state["exp_avg"] = torch.zeros_like(carrier)
                        state["exp_avg_sq"] = torch.zeros_like(carrier)
                        state["codes"] = codes
                        with torch.no_grad():
                            parameter.copy_(reconstruction.to(parameter.dtype))
                    else:
                        reduce_dim = int(group["scale_reduce_dim"])
                        carrier, scale = _initial_qat_carrier_and_scale(
                            parameter,
                            reduce_dim=reduce_dim,
                        )
                        states = carrier.round().clamp_(-1, 1).to(torch.int8)
                        state["carrier"] = carrier
                        state["scale"] = scale
                        state["exp_avg"] = torch.zeros_like(carrier)
                        state["exp_avg_sq"] = torch.zeros_like(carrier)
                        state["scale_exp_avg"] = torch.zeros_like(scale)
                        state["scale_exp_avg_sq"] = torch.zeros_like(scale)
                        state["packed_states"] = pack_ternary_states(states)[0]
                        with torch.no_grad():
                            parameter.copy_(
                                _reconstruct_normalized_ternary(
                                    states,
                                    scale,
                                    reduce_dim=reduce_dim,
                                ).to(parameter.dtype)
                            )
            if bf16_stochastic_update:
                for parameter in group["params"]:
                    parameter.data = parameter.detach().to(torch.bfloat16)
                    parameter.grad_dtype = None
                    state = self.state[parameter]
                    if algorithm == GROUPED_ABSMEAN_ALGORITHM:
                        for name in ("carrier", "exp_avg", "exp_avg_sq"):
                            state[name] = state[name].to(torch.bfloat16)
                        states, scale, reconstruction = (
                            _grouped_absmean_quantize_for_group(
                                state["carrier"],
                                group,
                            )
                        )
                        state["scale"] = scale
                        state["packed_states"] = pack_ternary_states(states)[0]
                        parameter.data.copy_(reconstruction.to(torch.bfloat16))
        stochastic_parameters = [
            parameter
            for group in self.param_groups
            if bool(group.get("unbiased_stochastic_rounding", False))
            or bool(group.get("bf16_stochastic_update", False))
            for parameter in group["params"]
        ]
        if stochastic_parameters:
            devices = {parameter.device for parameter in stochastic_parameters}
            seeds = {
                int(group.get("stochastic_rounding_seed", torch.initial_seed()))
                for group in self.param_groups
                if bool(group.get("unbiased_stochastic_rounding", False))
                or bool(group.get("bf16_stochastic_update", False))
            }
            if len(devices) != 1 or len(seeds) != 1:
                raise ValueError(
                    "stochastic grouped rounding requires one device and seed"
                )
            owner = stochastic_parameters[0]
            generator = torch.Generator(device=owner.device)
            generator.manual_seed(next(iter(seeds)))
            self._stochastic_rounding_generator = generator
            self._stochastic_rounding_rng_owner = owner
            self.state[owner]["stochastic_rounding_rng_state"] = (
                generator.get_state().cpu()
            )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        saved_counters: dict[tuple[int, str], int] = {}
        saved_groups = state_dict.get("param_groups", ())
        if len(saved_groups) != len(self.param_groups):
            raise RuntimeError("restored low-bit parameter groups differ")
        for saved_group, current_group in zip(
            saved_groups,
            self.param_groups,
            strict=True,
        ):
            saved_parameters = saved_group.get("params", ())
            current_parameters = current_group.get("params", ())
            if len(saved_parameters) != len(current_parameters):
                raise RuntimeError("restored low-bit parameter layout differs")
            for saved_parameter, current_parameter in zip(
                saved_parameters,
                current_parameters,
                strict=True,
            ):
                saved_state = state_dict.get("state", {}).get(saved_parameter, {})
                for name in ("transition_count", "last_transition_count"):
                    counter = saved_state.get(name)
                    if isinstance(counter, Tensor) and counter.numel() == 1:
                        saved_counters[(id(current_parameter), name)] = int(
                            counter.item()
                        )
        super().load_state_dict(state_dict)
        self._balanced_two_of_four_masks = {}
        for group in self.param_groups:
            bf16_stochastic_update = bool(
                group.get("bf16_stochastic_update", False)
            )
            balanced_role = group.get("balanced_two_of_four_role")
            if balanced_role is not None:
                for parameter in group["params"]:
                    mask = balanced_two_of_four_mask(
                        parameter,
                        role=int(balanced_role),
                    )
                    self._balanced_two_of_four_masks[id(parameter)] = mask
                    state = self.state[parameter]
                    for name in ("carrier", "exp_avg", "exp_avg_sq"):
                        value = state.get(name)
                        if isinstance(value, Tensor) and bool(
                            value.masked_select(~mask).any()
                        ):
                            raise RuntimeError(
                                f"restored balanced two-of-four {name} "
                                "has inactive values"
                            )
            if bf16_stochastic_update:
                for parameter in group["params"]:
                    if parameter.dtype != torch.bfloat16:
                        raise RuntimeError(
                            "restored BF16 optimizer parameter has wrong dtype"
                        )
                    state = self.state[parameter]
                    if group["algorithm"] == GROUPED_ABSMEAN_ALGORITHM:
                        for name in ("carrier", "exp_avg", "exp_avg_sq"):
                            value = state.get(name)
                            if not isinstance(value, Tensor):
                                raise RuntimeError(
                                    f"restored BF16 grouped {name} is missing"
                                )
                            state[name] = value.to(torch.bfloat16)
                        restored_scale = state.get("scale")
                        if not isinstance(restored_scale, Tensor):
                            raise RuntimeError(
                                "restored BF16 grouped scale is missing"
                            )
                        _, exact_scale, _ = _grouped_absmean_quantize_for_group(
                            state["carrier"],
                            group,
                        )
                        state["scale"] = exact_scale
                    elif state:
                        for name in ("exp_avg", "exp_avg_sq"):
                            value = state.get(name)
                            if isinstance(value, Tensor):
                                state[name] = value.to(torch.bfloat16)
        for parameter in self.low_bit_parameters():
            state = self.state[parameter]
            for name in ("transition_count", "last_transition_count"):
                counter = saved_counters.get((id(parameter), name))
                if counter is None:
                    raise RuntimeError(f"restored low-bit {name} is invalid")
                state[name] = torch.tensor(
                    counter,
                    device=parameter.device,
                    dtype=torch.int64,
                )
        for parameter in self.ternary_parameters():
            state = self.state[parameter]
            algorithm = self.algorithm_for(parameter)
            if algorithm in ("ternary_qat_carrier", GROUPED_ABSMEAN_ALGORITHM):
                packed = state.get("packed_states")
                if not isinstance(packed, Tensor):
                    raise RuntimeError("restored QAT packed ternary state is missing")
                state["packed_states"] = packed.to(
                    device=parameter.device,
                    dtype=torch.uint8,
                ).reshape(-1)
            elif algorithm == "ternary_mantissa6":
                packed_planes = state.get("packed_planes")
                if not isinstance(packed_planes, (tuple, list)):
                    raise RuntimeError("restored packed ternary planes are missing")
                state["packed_planes"] = tuple(
                    packed.to(device=parameter.device, dtype=torch.uint8).reshape(-1)
                    for packed in packed_planes
                )
        for parameter in self.int8_parameters():
            state = self.state[parameter]
            codes = state.get("codes")
            if not isinstance(codes, Tensor):
                raise RuntimeError("restored INT8 QAT codes are missing")
            state["codes"] = codes.to(
                device=parameter.device,
                dtype=torch.int8,
            ).reshape(parameter.shape)
        owner = self._stochastic_rounding_rng_owner
        generator = self._stochastic_rounding_generator
        if owner is not None:
            if generator is None:
                raise RuntimeError("stochastic grouped RNG generator is missing")
            rng_state = self.state[owner].get("stochastic_rounding_rng_state")
            if not isinstance(rng_state, Tensor):
                raise RuntimeError("restored stochastic grouped RNG state is missing")
            rng_state = rng_state.detach().cpu().to(torch.uint8).reshape(-1)
            self.state[owner]["stochastic_rounding_rng_state"] = rng_state
            generator.set_state(rng_state)

    def algorithm_for(self, parameter: Tensor) -> str:
        return str(self._group_for(parameter)["algorithm"])

    def _group_for(self, parameter: Tensor) -> dict[str, Any]:
        for group in self.param_groups:
            if any(candidate is parameter for candidate in group["params"]):
                return group
        raise KeyError("parameter is not owned by this optimizer")

    def scale_for(self, parameter: Tensor) -> Tensor:
        if self.algorithm_for(parameter) not in (
            "ternary_dst",
            "ternary_mantissa6",
            "ternary_qat_carrier",
            GROUPED_ABSMEAN_ALGORITHM,
            SYMMETRIC_INT8_ALGORITHM,
        ):
            raise ValueError("dense parameter has no low-bit scale")
        state = self.state[parameter]
        scale = state.get(
            "base_scale"
            if self.algorithm_for(parameter) == "ternary_mantissa6"
            else "scale"
        )
        if not isinstance(scale, Tensor):
            raise RuntimeError("low-bit scale is missing")
        return scale

    def states_for(self, parameter: Tensor, *, verify: bool = True) -> Tensor:
        algorithm = self.algorithm_for(parameter)
        if algorithm not in (
            "ternary_dst",
            "ternary_qat_carrier",
            GROUPED_ABSMEAN_ALGORITHM,
        ):
            raise ValueError("single-plane states require a ternary or QAT route")
        if algorithm == "ternary_dst":
            scale = _broadcast_scale(parameter, self.scale_for(parameter))
            states = (parameter.detach().float() / scale).round().to(torch.int8)
            reconstructed = states.to(torch.float32) * scale
        elif algorithm == "ternary_qat_carrier":
            carrier = self.carrier_for(parameter)
            states = carrier.round().clamp_(-1, 1).to(torch.int8)
            group = self._group_for(parameter)
            reconstructed = _reconstruct_normalized_ternary(
                states,
                self.scale_for(parameter),
                reduce_dim=int(group["scale_reduce_dim"]),
            )
        else:
            group = self._group_for(parameter)
            if bool(group.get("unbiased_stochastic_rounding", False)):
                packed = self.state[parameter].get("packed_states")
                if not isinstance(packed, Tensor):
                    raise RuntimeError("QAT packed ternary state is missing")
                states = unpack_ternary_states(
                    packed,
                    parameter.numel(),
                    shape=parameter.shape,
                )
                _, expected_scale, _ = grouped_absmean_quantize(
                    self.carrier_for(parameter),
                    group_size=int(
                        group.get("group_size", GROUPED_ABSMEAN_SIZE)
                    ),
                )
                reconstructed = _reconstruct_grouped_absmean_states(
                    states,
                    expected_scale,
                    group_size=int(
                        group.get("group_size", GROUPED_ABSMEAN_SIZE)
                    ),
                )
            else:
                states, expected_scale, reconstructed = (
                    _grouped_absmean_quantize_for_group(
                        self.carrier_for(parameter),
                        group,
                    )
                )
            if verify and not torch.equal(expected_scale, self.scale_for(parameter)):
                raise RuntimeError("grouped ternary scale differs from carrier absmean")
        if verify:
            if bool(((states < -1) | (states > 1)).any()):
                raise RuntimeError("parameter state lies outside ternary space")
            if not torch.equal(reconstructed.to(parameter.dtype), parameter.detach()):
                raise RuntimeError("parameter is not exactly scale times ternary state")
            if algorithm in ("ternary_qat_carrier", GROUPED_ABSMEAN_ALGORITHM):
                packed = self.state[parameter].get("packed_states")
                if not isinstance(packed, Tensor):
                    raise RuntimeError("QAT packed ternary state is missing")
                restored = unpack_ternary_states(
                    packed,
                    parameter.numel(),
                    shape=parameter.shape,
                )
                if not torch.equal(restored, states):
                    raise RuntimeError("QAT packed ternary state differs")
        return states

    def carrier_for(self, parameter: Tensor) -> Tensor:
        if self.algorithm_for(parameter) not in (
            "ternary_qat_carrier",
            GROUPED_ABSMEAN_ALGORITHM,
            SYMMETRIC_INT8_ALGORITHM,
        ):
            raise ValueError("continuous carrier requires a low-bit QAT route")
        carrier = self.state[parameter].get("carrier")
        if not isinstance(carrier, Tensor):
            raise RuntimeError("QAT low-bit carrier is missing")
        return carrier

    @torch.no_grad()
    def reset_grouped_carrier_(self, parameter: Tensor) -> None:
        """Initialize one grouped QAT route from its current dense weight."""

        if self.algorithm_for(parameter) != GROUPED_ABSMEAN_ALGORITHM:
            raise ValueError("grouped carrier reset requires grouped absmean QAT")
        state = self.state[parameter]
        carrier = parameter.detach().float().clone()
        mask = self._balanced_two_of_four_masks.get(id(parameter))
        if mask is not None:
            carrier.mul_(mask)
        states, scale, reconstruction = _grouped_absmean_quantize_for_group(
            carrier,
            self._group_for(parameter),
        )
        state["step"] = 0
        state["transition_count"].zero_()
        state["last_transition_count"].zero_()
        state["carrier"] = carrier
        state["scale"] = scale
        state["exp_avg"] = torch.zeros_like(carrier)
        state["exp_avg_sq"] = torch.zeros_like(carrier)
        state["packed_states"] = pack_ternary_states(states)[0]
        if bool(self._group_for(parameter).get("bf16_stochastic_update", False)):
            for name in ("carrier", "exp_avg", "exp_avg_sq"):
                state[name] = state[name].to(torch.bfloat16)
            states, scale, reconstruction = _grouped_absmean_quantize_for_group(
                state["carrier"],
                self._group_for(parameter),
            )
            state["scale"] = scale
            state["packed_states"] = pack_ternary_states(states)[0]
        parameter.copy_(reconstruction.to(parameter.dtype))

    @torch.no_grad()
    def replace_grouped_carrier_(self, parameter: Tensor, carrier: Tensor) -> None:
        """Replace one grouped QAT carrier and reset its untouched step-zero state."""

        if self.algorithm_for(parameter) != GROUPED_ABSMEAN_ALGORITHM:
            raise ValueError("grouped carrier replacement requires grouped absmean QAT")
        if carrier.shape != parameter.shape or not bool(torch.isfinite(carrier).all()):
            raise ValueError("replacement grouped carrier differs")
        group = self._group_for(parameter)
        if bool(group.get("nonnegative_parameter", False)) and bool((carrier < 0).any()):
            raise ValueError("nonnegative grouped carrier replacement is negative")
        state = self.state[parameter]
        replacement = carrier.detach().float().clone()
        mask = self._balanced_two_of_four_masks.get(id(parameter))
        if mask is not None:
            replacement.mul_(mask)
        states, scale, reconstruction = _grouped_absmean_quantize_for_group(
            replacement,
            group,
        )
        state["step"] = 0
        state["transition_count"].zero_()
        state["last_transition_count"].zero_()
        state["carrier"].copy_(replacement.to(state["carrier"].dtype))
        state["exp_avg"].zero_()
        state["exp_avg_sq"].zero_()
        if bool(group.get("bf16_stochastic_update", False)):
            states, scale, reconstruction = _grouped_absmean_quantize_for_group(
                state["carrier"],
                group,
            )
        state["scale"].copy_(scale)
        state["packed_states"].copy_(pack_ternary_states(states)[0])
        parameter.copy_(reconstruction.to(parameter.dtype))

    @torch.no_grad()
    def reset_symmetric_int8_carrier_(self, parameter: Tensor) -> None:
        """Initialize one INT8 QAT route from its current dense weight."""

        if self.algorithm_for(parameter) != SYMMETRIC_INT8_ALGORITHM:
            raise ValueError("INT8 carrier reset requires symmetric INT8 QAT")
        state = self.state[parameter]
        carrier = parameter.detach().float().clone()
        codes, scale, reconstruction = symmetric_int8_quantize(
            carrier,
            reduce_dim=int(self._group_for(parameter)["scale_reduce_dim"]),
        )
        state["step"] = 0
        state["transition_count"].zero_()
        state["last_transition_count"].zero_()
        state["carrier"] = carrier
        state["scale"] = scale
        state["exp_avg"] = torch.zeros_like(carrier)
        state["exp_avg_sq"] = torch.zeros_like(carrier)
        state["codes"] = codes
        parameter.copy_(reconstruction.to(parameter.dtype))

    def reconstruct_qat(self, parameter: Tensor) -> Tensor:
        algorithm = self.algorithm_for(parameter)
        if algorithm not in (
            "ternary_qat_carrier",
            GROUPED_ABSMEAN_ALGORITHM,
            SYMMETRIC_INT8_ALGORITHM,
        ):
            raise ValueError("QAT reconstruction requires a low-bit QAT route")
        if algorithm == GROUPED_ABSMEAN_ALGORITHM:
            return _grouped_absmean_quantize_for_group(
                self.carrier_for(parameter),
                self._group_for(parameter),
            )[2]
        group = self._group_for(parameter)
        if algorithm == SYMMETRIC_INT8_ALGORITHM:
            return symmetric_int8_quantize(
                self.carrier_for(parameter),
                reduce_dim=int(group["scale_reduce_dim"]),
            )[2]
        return _reconstruct_normalized_ternary(
            self.states_for(parameter, verify=False),
            self.scale_for(parameter),
            reduce_dim=int(group["scale_reduce_dim"]),
        )

    def ternary_parameters(self) -> tuple[Tensor, ...]:
        return tuple(
            parameter
            for group in self.param_groups
            if group["algorithm"] in TERNARY_ALGORITHMS
            for parameter in group["params"]
        )

    def int8_parameters(self) -> tuple[Tensor, ...]:
        return tuple(
            parameter
            for group in self.param_groups
            if group["algorithm"] == SYMMETRIC_INT8_ALGORITHM
            for parameter in group["params"]
        )

    def low_bit_parameters(self) -> tuple[Tensor, ...]:
        return (*self.ternary_parameters(), *self.int8_parameters())

    def codes_for(self, parameter: Tensor, *, verify: bool = True) -> Tensor:
        if self.algorithm_for(parameter) != SYMMETRIC_INT8_ALGORITHM:
            raise ValueError("INT8 codes require symmetric INT8 QAT")
        codes = self.state[parameter].get("codes")
        if not isinstance(codes, Tensor) or codes.dtype != torch.int8:
            raise RuntimeError("INT8 QAT codes are missing")
        if verify:
            expected_codes, expected_scale, reconstruction = symmetric_int8_quantize(
                self.carrier_for(parameter),
                reduce_dim=int(self._group_for(parameter)["scale_reduce_dim"]),
            )
            if not torch.equal(codes, expected_codes):
                raise RuntimeError("INT8 codes differ from carrier quantization")
            if not torch.equal(self.scale_for(parameter), expected_scale):
                raise RuntimeError("INT8 scale differs from carrier max-abs")
            if not torch.equal(reconstruction.to(parameter.dtype), parameter.detach()):
                raise RuntimeError("parameter differs from INT8 reconstruction")
        return codes

    def mantissa_digits_for(
        self,
        parameter: Tensor,
        *,
        verify: bool = True,
    ) -> tuple[Tensor, ...]:
        if self.algorithm_for(parameter) != "ternary_mantissa6":
            raise ValueError("mantissa digits require ternary_mantissa6")
        packed_planes = self.state[parameter].get("packed_planes")
        if not isinstance(packed_planes, tuple) or len(packed_planes) != (
            MANTISSA_PLANE_COUNT
        ):
            raise RuntimeError("packed ternary mantissa planes are missing")
        digits = tuple(
            unpack_ternary_states(
                packed,
                parameter.numel(),
                shape=parameter.shape,
            )
            for packed in packed_planes
        )
        if verify:
            reconstructed = self.reconstruct_mantissa(parameter, digits=digits)
            if not torch.equal(reconstructed.to(parameter.dtype), parameter.detach()):
                raise RuntimeError("parameter differs from ternary mantissa state")
        return digits

    def row_exponent_for(self, parameter: Tensor) -> Tensor:
        if self.algorithm_for(parameter) != "ternary_mantissa6":
            raise ValueError("row exponent requires ternary_mantissa6")
        exponent = self.state[parameter].get("row_exponent")
        if not isinstance(exponent, Tensor) or exponent.dtype != torch.int8:
            raise RuntimeError("ternary row exponent is missing")
        return exponent

    def reconstruct_mantissa(
        self,
        parameter: Tensor,
        *,
        digits: tuple[Tensor, ...] | None = None,
    ) -> Tensor:
        if digits is None:
            digits = self.mantissa_digits_for(parameter, verify=False)
        if len(digits) != MANTISSA_PLANE_COUNT:
            raise ValueError("ternary mantissa plane count differs")
        base_scale = _broadcast_scale(parameter, self.scale_for(parameter))
        exponent = self.row_exponent_for(parameter)
        effective_scale = base_scale * torch.pow(
            torch.tensor(3.0, device=parameter.device, dtype=torch.float32),
            exponent.float(),
        )
        reconstruction = torch.zeros_like(parameter, dtype=torch.float32)
        for plane, digit in enumerate(digits):
            reconstruction.add_(digit.float() * (effective_scale / (3**plane)))
        return reconstruction

    def _store_mantissa_state_(
        self,
        parameter: Tensor,
        *,
        digits: tuple[Tensor, ...],
        exponent: Tensor,
    ) -> None:
        if len(digits) != MANTISSA_PLANE_COUNT:
            raise ValueError("ternary mantissa plane count differs")
        expected_exponent_shape = self.scale_for(parameter).shape
        if exponent.shape != expected_exponent_shape or exponent.dtype != torch.int8:
            raise ValueError("ternary row exponent layout differs")
        self.state[parameter]["row_exponent"] = exponent.detach().clone()
        self.state[parameter]["packed_planes"] = tuple(
            pack_ternary_states(digit)[0] for digit in digits
        )

    def tie_transition_randomness(
        self,
        groups: Iterable[tuple[tuple[Tensor, bool], ...]],
    ) -> None:
        """Share stochastic draws across transpose-aligned reciprocal states."""

        assignments: dict[int, tuple[int, bool, tuple[int, ...]]] = {}
        for group_index, members in enumerate(groups):
            if len(members) < 2 or members[0][1]:
                raise ValueError("tied ternary group needs one canonical first member")
            canonical_shape = tuple(members[0][0].shape)
            canonical_algorithm = self.algorithm_for(members[0][0])
            for parameter, transposed in members:
                if self.algorithm_for(parameter) != canonical_algorithm or (
                    canonical_algorithm
                    not in ("ternary_dst", "ternary_mantissa6")
                ):
                    raise ValueError("tied randomness requires ternary parameters")
                expected_shape = (
                    (*canonical_shape[:-2], canonical_shape[-1], canonical_shape[-2])
                    if transposed
                    else canonical_shape
                )
                if tuple(parameter.shape) != expected_shape:
                    raise ValueError("tied ternary parameter shape differs")
                if id(parameter) in assignments:
                    raise ValueError("ternary parameter belongs to multiple tied groups")
                assignments[id(parameter)] = (
                    group_index,
                    transposed,
                    canonical_shape,
                )
        self._transition_randomness_groups = assignments

    @torch.no_grad()
    def align_reciprocal_states_(
        self,
        groups: Iterable[tuple[tuple[Tensor, bool], ...]],
    ) -> None:
        """Copy one canonical ternary state and scale into reciprocal storage."""

        for members in groups:
            if len(members) < 2 or members[0][1]:
                raise ValueError("aligned ternary group needs one canonical first member")
            canonical = members[0][0]
            canonical_shape = tuple(canonical.shape)
            canonical_algorithm = self.algorithm_for(canonical)
            canonical_scale = self.scale_for(canonical)
            canonical_states = (
                self.states_for(canonical)
                if canonical_algorithm == "ternary_dst"
                else None
            )
            canonical_digits = (
                self.mantissa_digits_for(canonical)
                if canonical_algorithm == "ternary_mantissa6"
                else None
            )
            canonical_exponent = (
                self.row_exponent_for(canonical)
                if canonical_algorithm == "ternary_mantissa6"
                else None
            )
            for parameter, transposed in members[1:]:
                if self.algorithm_for(parameter) != canonical_algorithm:
                    raise ValueError("aligned ternary algorithm differs")
                expected_shape = (
                    (*canonical_shape[:-2], canonical_shape[-1], canonical_shape[-2])
                    if transposed
                    else canonical_shape
                )
                if tuple(parameter.shape) != expected_shape:
                    raise ValueError("aligned ternary parameter shape differs")
                target_scale = (
                    canonical_scale.transpose(-1, -2)
                    if transposed
                    else canonical_scale
                )
                scale = self.scale_for(parameter)
                if scale.shape != target_scale.shape:
                    raise ValueError("aligned ternary scale shape differs")
                scale.copy_(target_scale)
                if canonical_algorithm == "ternary_dst":
                    assert canonical_states is not None
                    target_states = (
                        canonical_states.transpose(-1, -2)
                        if transposed
                        else canonical_states
                    )
                    parameter.copy_(
                        (
                            target_states.to(torch.float32)
                            * _broadcast_scale(parameter, scale)
                        ).to(parameter.dtype)
                    )
                else:
                    assert canonical_digits is not None
                    assert canonical_exponent is not None
                    target_digits = tuple(
                        digit.transpose(-1, -2) if transposed else digit
                        for digit in canonical_digits
                    )
                    target_exponent = (
                        canonical_exponent.transpose(-1, -2)
                        if transposed
                        else canonical_exponent
                    )
                    self._store_mantissa_state_(
                        parameter,
                        digits=target_digits,
                        exponent=target_exponent,
                    )
                    parameter.copy_(
                        self.reconstruct_mantissa(
                            parameter,
                            digits=target_digits,
                        ).to(parameter.dtype)
                    )

    @torch.no_grad()
    def project_reciprocal_mantissa_states_(
        self,
        groups: Iterable[tuple[tuple[Tensor, bool], ...]],
    ) -> None:
        """Project bidirectional local updates onto one reciprocal state."""

        for members in groups:
            if len(members) < 2 or members[0][1]:
                raise ValueError("reciprocal projection needs one canonical member")
            canonical = members[0][0]
            canonical_shape = tuple(canonical.shape)
            if self.algorithm_for(canonical) != "ternary_mantissa6":
                raise ValueError("reciprocal projection requires mantissa states")
            snapshots: list[tuple[Tensor, Tensor]] = []
            target = torch.zeros_like(canonical, dtype=torch.float32)
            for parameter, transposed in members:
                if self.algorithm_for(parameter) != "ternary_mantissa6":
                    raise ValueError("reciprocal projection algorithm differs")
                expected_shape = (
                    (*canonical_shape[:-2], canonical_shape[-1], canonical_shape[-2])
                    if transposed
                    else canonical_shape
                )
                if tuple(parameter.shape) != expected_shape:
                    raise ValueError("reciprocal projection shape differs")
                snapshots.append((parameter, parameter.detach().clone()))
                oriented = parameter.transpose(-1, -2) if transposed else parameter
                target.add_(oriented.detach().float())
            target.div_(len(members))
            canonical.copy_(target.to(canonical.dtype))
            self.snap_parameters_((canonical,))
            self.align_reciprocal_states_((members,))
            for parameter, before in snapshots:
                changed = (parameter.detach() != before).sum(dtype=torch.int64)
                state = self.state[parameter]
                state["transition_count"].add_(changed)
                state["last_transition_count"].add_(changed)

    def tie_transition_requests(
        self,
        groups: Iterable[tuple[tuple[Tensor, bool], ...]],
    ) -> None:
        """Average strict-local requests for deployed reciprocal pairs."""

        assignments: dict[int, tuple[int, bool, tuple[int, ...]]] = {}
        group_sizes: dict[int, int] = {}
        for group_index, members in enumerate(groups):
            if len(members) != 2 or members[0][1]:
                raise ValueError("tied ternary request needs one reciprocal pair")
            canonical_shape = tuple(members[0][0].shape)
            canonical_algorithm = self.algorithm_for(members[0][0])
            group_sizes[group_index] = len(members)
            for parameter, transposed in members:
                if self.algorithm_for(parameter) != canonical_algorithm or (
                    canonical_algorithm
                    not in ("ternary_dst", "ternary_mantissa6")
                ):
                    raise ValueError("tied request requires ternary parameters")
                expected_shape = (
                    (*canonical_shape[:-2], canonical_shape[-1], canonical_shape[-2])
                    if transposed
                    else canonical_shape
                )
                if tuple(parameter.shape) != expected_shape:
                    raise ValueError("tied ternary request shape differs")
                if id(parameter) in assignments:
                    raise ValueError("ternary parameter belongs to multiple request groups")
                assignments[id(parameter)] = (
                    group_index,
                    transposed,
                    canonical_shape,
                )
        self._transition_request_groups = assignments
        self._transition_request_group_sizes = group_sizes

    @torch.no_grad()
    def fold_parameter_delta_into_gradient_(
        self,
        parameter: Tensor,
        delta: Tensor,
    ) -> None:
        """Add one current-step weight displacement to a ternary transition request."""

        group = self._group_for(parameter)
        if group["algorithm"] not in LOW_BIT_ALGORITHMS:
            raise ValueError("only low-bit parameters accept a folded displacement")
        if parameter.grad is None:
            raise RuntimeError("ternary displacement requires a current local gradient")
        if delta.shape != parameter.shape:
            raise ValueError("ternary displacement shape differs from parameter")
        parameter.grad.add_(
            delta.to(device=parameter.device, dtype=parameter.grad.dtype),
            alpha=-1.0 / float(group["lr"]),
        )

    def tensor_state_bytes(self) -> int:
        def tensor_bytes(value: Any) -> int:
            if isinstance(value, Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, (tuple, list)):
                return sum(tensor_bytes(item) for item in value)
            if isinstance(value, dict):
                return sum(tensor_bytes(item) for item in value.values())
            return 0

        return sum(
            tensor_bytes(value)
            for state in self.state.values()
            for value in state.values()
        )

    def ternary_scale_bytes(self) -> int:
        return sum(
            self.scale_for(parameter).numel()
            * self.scale_for(parameter).element_size()
            for parameter in self.ternary_parameters()
        )

    def semantic_packed_bytes(self) -> int:
        total = 0
        for parameter in self.low_bit_parameters():
            algorithm = self.algorithm_for(parameter)
            if algorithm == SYMMETRIC_INT8_ALGORITHM:
                total += parameter.numel()
            elif algorithm in (
                "ternary_dst",
                "ternary_qat_carrier",
                GROUPED_ABSMEAN_ALGORITHM,
            ):
                group = self._group_for(parameter)
                if bool(group.get("grouped_symmetric_last_two_dims", False)):
                    width = parameter.shape[-1]
                    matrix_count = parameter.numel() // (width * width)
                    deployed_values = matrix_count * width * (width + 1) // 2
                    total += (deployed_values + 3) // 4
                else:
                    total += (parameter.numel() + 3) // 4
            else:
                packed_planes = self.state[parameter]["packed_planes"]
                total += sum(packed.numel() for packed in packed_planes)
                total += self.row_exponent_for(parameter).numel()
            group = self._group_for(parameter)
            if not (
                bool(group.get("grouped_pair_contract", False))
                or bool(group.get("codes_only_deployment", False))
            ):
                scale_bytes = (
                    2
                    if algorithm
                    in (GROUPED_ABSMEAN_ALGORITHM, SYMMETRIC_INT8_ALGORITHM)
                    else self.scale_for(parameter).element_size()
                )
                total += self.scale_for(parameter).numel() * scale_bytes
        return total

    @torch.no_grad()
    def snap_parameters_(self, parameters: Iterable[Tensor]) -> None:
        """Project externally refreshed mirrors onto their fixed ternary scales."""

        for parameter in parameters:
            algorithm = self.algorithm_for(parameter)
            if algorithm not in DIRECT_TERNARY_ALGORITHMS:
                raise ValueError("cannot snap a dense optimizer parameter")
            if algorithm == "ternary_dst":
                scale = _broadcast_scale(parameter, self.scale_for(parameter))
                states = (parameter.detach().float() / scale).round().clamp_(-1, 1)
                parameter.copy_((states * scale).to(parameter.dtype))
            else:
                group = self._group_for(parameter)
                digits, exponent, reconstruction = _quantize_mantissa(
                    parameter,
                    self.scale_for(parameter),
                    reduce_dim=int(group["scale_reduce_dim"]),
                )
                self._store_mantissa_state_(
                    parameter,
                    digits=digits,
                    exponent=exponent,
                )
                parameter.copy_(reconstruction.to(parameter.dtype))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if not bool(group.get("bf16_stochastic_update", False)):
                continue
            for parameter in group["params"]:
                if (
                    parameter.grad is not None
                    and parameter.grad.dtype != parameter.dtype
                ):
                    parameter.grad = parameter.grad.to(parameter.dtype)
        request_sums: dict[int, Tensor] = {}
        request_counts: dict[int, int] = {}
        for group in self.param_groups:
            if group["algorithm"] not in DIRECT_TERNARY_ALGORITHMS:
                continue
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                assignment = self._transition_request_groups.get(id(parameter))
                if assignment is None:
                    continue
                group_index, transposed, _canonical_shape = assignment
                request = self._requested_discrete_delta(parameter, group)
                if transposed:
                    request = request.transpose(-1, -2)
                if group_index in request_sums:
                    request_sums[group_index].add_(request)
                else:
                    request_sums[group_index] = request.clone()
                request_counts[group_index] = request_counts.get(group_index, 0) + 1
        request_values_by_group: dict[int, Tensor] = {}
        for group_index, request_sum in request_sums.items():
            expected = self._transition_request_group_sizes[group_index]
            if request_counts[group_index] != expected:
                raise RuntimeError("tied ternary request has a missing gradient")
            request_values_by_group[group_index] = request_sum.div_(expected)

        random_values_by_group: dict[int, Tensor] = {}
        stochastic_generator = self._stochastic_rounding_generator
        stochastic_owner = self._stochastic_rounding_rng_owner
        if stochastic_owner is not None:
            if stochastic_generator is None:
                raise RuntimeError("stochastic grouped RNG generator is missing")
            rng_state = self.state[stochastic_owner].get(
                "stochastic_rounding_rng_state"
            )
            if not isinstance(rng_state, Tensor):
                raise RuntimeError("stochastic grouped RNG state is missing")
            stochastic_generator.set_state(rng_state.cpu())
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("ternary optimizer does not support sparse gradients")
                if group["algorithm"] in DIRECT_TERNARY_ALGORITHMS:
                    request_assignment = self._transition_request_groups.get(
                        id(parameter)
                    )
                    requested: Tensor | None = None
                    if request_assignment is not None:
                        request_group_index, request_transposed, _shape = (
                            request_assignment
                        )
                        requested = request_values_by_group[request_group_index]
                        if request_transposed:
                            requested = requested.transpose(-1, -2)
                    randomness = self._transition_randomness_groups.get(id(parameter))
                    uniform: Tensor | None = None
                    if randomness is not None:
                        group_index, transposed, canonical_shape = randomness
                        if group_index not in random_values_by_group:
                            random_values_by_group[group_index] = torch.rand(
                                canonical_shape,
                                device=parameter.device,
                                dtype=torch.float32,
                            )
                        uniform = random_values_by_group[group_index]
                        if transposed:
                            uniform = uniform.transpose(-1, -2)
                    if group["algorithm"] == "ternary_dst":
                        self._step_ternary(
                            parameter,
                            group,
                            uniform=uniform,
                            requested=requested,
                        )
                    else:
                        self._step_mantissa(
                            parameter,
                            group,
                            uniform=uniform,
                            requested=requested,
                        )
                elif group["algorithm"] == "ternary_qat_carrier":
                    self._step_qat_carrier(parameter, group)
                elif group["algorithm"] == GROUPED_ABSMEAN_ALGORITHM:
                    self._step_grouped_absmean_qat(
                        parameter,
                        group,
                        generator=stochastic_generator,
                    )
                elif group["algorithm"] == SYMMETRIC_INT8_ALGORITHM:
                    self._step_symmetric_int8_qat(parameter, group)
                else:
                    self._step_adamw(
                        parameter,
                        group,
                        generator=stochastic_generator,
                    )
        if stochastic_owner is not None:
            assert stochastic_generator is not None
            rng_state = self.state[stochastic_owner].get(
                "stochastic_rounding_rng_state"
            )
            if not isinstance(rng_state, Tensor):
                raise RuntimeError("stochastic grouped RNG state is missing")
            rng_state.copy_(stochastic_generator.get_state().cpu())
        return loss

    def _step_ternary(
        self,
        parameter: Tensor,
        group: dict[str, Any],
        *,
        uniform: Tensor | None,
        requested: Tensor | None,
    ) -> None:
        state = self.state[parameter]
        scale = _broadcast_scale(parameter, self.scale_for(parameter))
        current = self.states_for(parameter).to(torch.float32)
        if requested is None:
            requested = self._requested_state_delta(parameter, group)
        elif requested.shape != parameter.shape:
            raise RuntimeError("tied ternary request layout differs")
        whole = torch.trunc(requested)
        remainder = requested - whole
        probability = torch.tanh(DST_TRANSITION_FACTOR * remainder.abs())
        if uniform is None:
            uniform = torch.rand_like(probability)
        elif uniform.shape != probability.shape or uniform.device != probability.device:
            raise RuntimeError("tied ternary randomness layout differs")
        stochastic = (uniform < probability).to(torch.float32) * remainder.sign()
        updated = (current + whole + stochastic).clamp_(-1.0, 1.0)
        transition_count = (updated != current).sum(dtype=torch.int64)
        parameter.copy_((updated * scale).to(parameter.dtype))
        state["step"] += 1
        state["transition_count"].add_(transition_count)
        state["last_transition_count"].copy_(transition_count)

    def _requested_state_delta(
        self,
        parameter: Tensor,
        group: dict[str, Any],
    ) -> Tensor:
        scale = _broadcast_scale(parameter, self.scale_for(parameter))
        gradient = parameter.grad.detach().float()
        if group["weight_decay"]:
            gradient = gradient + group["weight_decay"] * parameter.detach().float()
        return -group["lr"] * gradient / scale

    def _requested_weight_delta(
        self,
        parameter: Tensor,
        group: dict[str, Any],
    ) -> Tensor:
        gradient = parameter.grad.detach().float()
        if group["weight_decay"]:
            gradient = gradient + group["weight_decay"] * parameter.detach().float()
        return -group["lr"] * gradient

    def _requested_discrete_delta(
        self,
        parameter: Tensor,
        group: dict[str, Any],
    ) -> Tensor:
        if group["algorithm"] == "ternary_dst":
            return self._requested_state_delta(parameter, group)
        if group["algorithm"] == "ternary_mantissa6":
            return self._requested_weight_delta(parameter, group)
        raise ValueError("requested discrete delta requires a ternary algorithm")

    def _step_mantissa(
        self,
        parameter: Tensor,
        group: dict[str, Any],
        *,
        uniform: Tensor | None,
        requested: Tensor | None,
    ) -> None:
        state = self.state[parameter]
        if requested is None:
            requested = self._requested_weight_delta(parameter, group)
        elif requested.shape != parameter.shape:
            raise RuntimeError("tied ternary request layout differs")
        if uniform is None:
            uniform = torch.rand_like(parameter, dtype=torch.float32)
        elif uniform.shape != parameter.shape or uniform.device != parameter.device:
            raise RuntimeError("tied ternary randomness layout differs")
        target = parameter.detach().float() + requested
        digits, exponent, reconstruction = _quantize_mantissa(
            target,
            self.scale_for(parameter),
            reduce_dim=int(group["scale_reduce_dim"]),
            uniform=uniform,
        )
        transition_count = (
            reconstruction != parameter.detach().float()
        ).sum(dtype=torch.int64)
        self._store_mantissa_state_(
            parameter,
            digits=digits,
            exponent=exponent,
        )
        parameter.copy_(reconstruction.to(parameter.dtype))
        state["step"] += 1
        state["transition_count"].add_(transition_count)
        state["last_transition_count"].copy_(transition_count)

    def _step_qat_carrier(
        self,
        parameter: Tensor,
        group: dict[str, Any],
    ) -> None:
        state = self.state[parameter]
        reduce_dim = int(group["scale_reduce_dim"]) % parameter.ndim
        states_before = self.states_for(parameter).to(torch.float32)
        norm = torch.linalg.vector_norm(
            states_before,
            dim=reduce_dim,
            keepdim=True,
        ).clamp_min(1.0)
        normalized = states_before / norm
        scale = self.scale_for(parameter)
        gradient = parameter.grad.detach().float()
        scale_gradient = (gradient * normalized).sum(
            dim=reduce_dim,
            keepdim=True,
        )
        tangent_gradient = gradient - normalized * scale_gradient
        carrier_gradient = (
            _broadcast_scale(parameter, scale) / norm
        ) * tangent_gradient

        state["step"] += 1
        self._cautious_adamw_update_(
            self.carrier_for(parameter),
            carrier_gradient,
            exp_avg=state["exp_avg"],
            exp_avg_sq=state["exp_avg_sq"],
            step=state["step"],
            lr=float(group["lr"]),
            weight_decay=float(group["weight_decay"]),
            betas=group["betas"],
            eps=float(group["eps"]),
        )
        self._cautious_adamw_update_(
            scale,
            scale_gradient,
            exp_avg=state["scale_exp_avg"],
            exp_avg_sq=state["scale_exp_avg_sq"],
            step=state["step"],
            lr=float(group["lr"]),
            weight_decay=float(group.get("scale_weight_decay", 0.0)),
            betas=group["betas"],
            eps=float(group["eps"]),
        )

        states_after = self.carrier_for(parameter).round().clamp_(-1, 1).to(
            torch.int8
        )
        transition_count = (
            states_after != states_before.to(torch.int8)
        ).sum(dtype=torch.int64)
        state["packed_states"] = pack_ternary_states(states_after)[0]
        parameter.copy_(
            _reconstruct_normalized_ternary(
                states_after,
                scale,
                reduce_dim=reduce_dim,
            ).to(parameter.dtype)
        )
        state["transition_count"].add_(transition_count)
        state["last_transition_count"].copy_(transition_count)

    def _step_grouped_absmean_qat(
        self,
        parameter: Tensor,
        group: dict[str, Any],
        *,
        generator: torch.Generator | None,
    ) -> None:
        state = self.state[parameter]
        states_before = self.states_for(parameter).to(torch.int8)
        carrier = self.carrier_for(parameter)
        gradient = parameter.grad.detach().float()
        mask = self._balanced_two_of_four_masks.get(id(parameter))
        if mask is not None:
            gradient = gradient * mask
            carrier.mul_(mask)

        state["step"] += 1
        beta1, beta2 = group["betas"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        bf16_stochastic_update = bool(
            group.get("bf16_stochastic_update", False)
        )
        if bf16_stochastic_update:
            next_exp_avg = exp_avg.float().mul(beta1).add_(
                gradient,
                alpha=1.0 - beta1,
            )
            next_exp_avg_sq = exp_avg_sq.float().mul(beta2).addcmul_(
                gradient,
                gradient,
                value=1.0 - beta2,
            )
            if mask is not None:
                next_exp_avg.mul_(mask)
                next_exp_avg_sq.mul_(mask)
            exp_avg.copy_(next_exp_avg.to(torch.bfloat16))
            exp_avg_sq.copy_(next_exp_avg_sq.to(torch.bfloat16))
            moment = exp_avg.float()
            moment_sq = exp_avg_sq.float()
        else:
            if mask is not None:
                exp_avg.mul_(mask)
                exp_avg_sq.mul_(mask)
            exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                gradient,
                gradient,
                value=1.0 - beta2,
            )
            moment = exp_avg
            moment_sq = exp_avg_sq
        denominator = (moment_sq / (1.0 - beta2 ** state["step"])).sqrt_()
        denominator.add_(float(group["eps"]))
        direction = moment / (1.0 - beta1 ** state["step"])
        if bf16_stochastic_update:
            target = carrier.float().mul_(
                1.0 - float(group["lr"]) * float(group["weight_decay"])
            )
            target.addcdiv_(direction, denominator, value=-float(group["lr"]))
            if mask is not None:
                target.mul_(mask)
            if bool(group.get("nonnegative_parameter", False)):
                target.clamp_min_(0.0)
            rounded = stochastic_round_float32_to_bfloat16(
                target,
                generator=generator,
            )
            self._record_bf16_rounding_statistics_(
                state,
                current=carrier,
                target=target,
                rounded=rounded,
            )
            carrier.copy_(rounded)
        else:
            carrier.mul_(
                1.0 - float(group["lr"]) * float(group["weight_decay"])
            )
            carrier.addcdiv_(direction, denominator, value=-float(group["lr"]))
        if mask is not None:
            carrier.mul_(mask)
            exp_avg.mul_(mask)
            exp_avg_sq.mul_(mask)
        if (
            not bf16_stochastic_update
            and bool(group.get("nonnegative_parameter", False))
        ):
            carrier.clamp_min_(0.0)

        if bool(group.get("unbiased_stochastic_rounding", False)):
            if any(
                (
                    bool(group.get("grouped_symmetric_last_two_dims", False)),
                    bool(group.get("grouped_pair_contract", False)),
                    group.get("balanced_two_of_four_role") is not None,
                    int(group.get("grouped_independent_leading_slices", 1)) != 1,
                )
            ):
                raise RuntimeError(
                    "stochastic grouped rounding reached a structured group"
                )
            states_after, scale, reconstruction = (
                grouped_absmean_stochastic_quantize(
                    carrier,
                    group_size=int(
                        group.get("group_size", GROUPED_ABSMEAN_SIZE)
                    ),
                    generator=generator,
                )
            )
        else:
            states_after, scale, reconstruction = (
                _grouped_absmean_quantize_for_group(
                    carrier,
                    group,
                )
            )
        transition_count = (states_after != states_before).sum(dtype=torch.int64)
        state["scale"].copy_(scale)
        state["packed_states"].copy_(pack_ternary_states(states_after)[0])
        parameter.copy_(reconstruction.to(parameter.dtype))
        state["transition_count"].add_(transition_count)
        state["last_transition_count"].copy_(transition_count)

    def _step_symmetric_int8_qat(
        self,
        parameter: Tensor,
        group: dict[str, Any],
    ) -> None:
        state = self.state[parameter]
        codes_before = self.codes_for(parameter).clone()
        carrier = self.carrier_for(parameter)
        gradient = parameter.grad.detach().float()

        state["step"] += 1
        beta1, beta2 = group["betas"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(
            gradient,
            gradient,
            value=1.0 - beta2,
        )
        denominator = (exp_avg_sq / (1.0 - beta2 ** state["step"])).sqrt_()
        denominator.add_(float(group["eps"]))
        direction = exp_avg / (1.0 - beta1 ** state["step"])
        carrier.mul_(1.0 - float(group["lr"]) * float(group["weight_decay"]))
        carrier.addcdiv_(direction, denominator, value=-float(group["lr"]))

        codes_after, scale, reconstruction = symmetric_int8_quantize(
            carrier,
            reduce_dim=int(group["scale_reduce_dim"]),
        )
        transition_count = (codes_after != codes_before).sum(dtype=torch.int64)
        state["scale"] = scale
        state["codes"] = codes_after
        parameter.copy_(reconstruction.to(parameter.dtype))
        state["transition_count"].add_(transition_count)
        state["last_transition_count"].copy_(transition_count)

    @staticmethod
    def _cautious_adamw_update_(
        value: Tensor,
        gradient: Tensor,
        *,
        exp_avg: Tensor,
        exp_avg_sq: Tensor,
        step: int,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
    ) -> None:
        beta1, beta2 = betas
        exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(
            gradient,
            gradient,
            value=1.0 - beta2,
        )
        denominator = (exp_avg_sq / (1.0 - beta2**step)).sqrt().add_(eps)
        direction = exp_avg / (1.0 - beta1**step)
        mask = (exp_avg * gradient > 0).to(direction.dtype)
        direction = direction * (mask / mask.mean().clamp_min(1e-3))
        value.mul_(1.0 - lr * weight_decay)
        value.addcdiv_(direction, denominator, value=-lr)

    @staticmethod
    def _record_bf16_rounding_statistics_(
        state: dict[str, Any],
        *,
        current: Tensor,
        target: Tensor,
        rounded: Tensor,
    ) -> None:
        nearest = target.to(torch.bfloat16)
        target_changed = target != current.float()
        nearest_no_change = target_changed & (nearest == current)
        rescued_motion = nearest_no_change & (rounded != current)
        state["last_bf16_target_change_count"] = target_changed.sum(
            dtype=torch.int64
        )
        state["last_bf16_nearest_no_change_count"] = nearest_no_change.sum(
            dtype=torch.int64
        )
        state["last_bf16_rescued_motion_count"] = rescued_motion.sum(
            dtype=torch.int64
        )
        for last_name, total_name in (
            (
                "last_bf16_target_change_count",
                "bf16_target_change_count",
            ),
            (
                "last_bf16_nearest_no_change_count",
                "bf16_nearest_no_change_count",
            ),
            (
                "last_bf16_rescued_motion_count",
                "bf16_rescued_motion_count",
            ),
        ):
            if total_name not in state:
                state[total_name] = torch.zeros(
                    (),
                    device=current.device,
                    dtype=torch.int64,
                )
            state[total_name].add_(state[last_name])

    def _step_adamw(
        self,
        parameter: Tensor,
        group: dict[str, Any],
        *,
        generator: torch.Generator | None,
    ) -> None:
        gradient = parameter.grad.detach().float()
        state = self.state[parameter]
        bf16_stochastic_update = bool(
            group.get("bf16_stochastic_update", False)
        )
        if "exp_avg" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(
                parameter,
                dtype=(
                    torch.bfloat16
                    if bf16_stochastic_update
                    else torch.float32
                ),
                memory_format=torch.preserve_format,
            )
            state["exp_avg_sq"] = torch.zeros_like(
                parameter,
                dtype=(
                    torch.bfloat16
                    if bf16_stochastic_update
                    else torch.float32
                ),
                memory_format=torch.preserve_format,
            )
        state["step"] += 1
        beta1, beta2 = group["betas"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        if bf16_stochastic_update:
            next_exp_avg = exp_avg.float().mul(beta1).add_(
                gradient,
                alpha=1.0 - beta1,
            )
            next_exp_avg_sq = exp_avg_sq.float().mul(beta2).addcmul_(
                gradient,
                gradient,
                value=1.0 - beta2,
            )
            exp_avg.copy_(next_exp_avg.to(torch.bfloat16))
            exp_avg_sq.copy_(next_exp_avg_sq.to(torch.bfloat16))
            moment = exp_avg.float()
            moment_sq = exp_avg_sq.float()
        else:
            exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                gradient,
                gradient,
                value=1.0 - beta2,
            )
            moment = exp_avg
            moment_sq = exp_avg_sq
        step = state["step"]
        update = (moment / (1.0 - beta1**step)) / (
            (moment_sq / (1.0 - beta2**step)).sqrt() + group["eps"]
        )
        if bf16_stochastic_update:
            target = parameter.float().mul_(
                1.0 - group["lr"] * group["weight_decay"]
            )
            target.add_(update, alpha=-group["lr"])
            if bool(group.get("nonnegative_parameter", False)):
                target.clamp_min_(0.0)
            rounded = stochastic_round_float32_to_bfloat16(
                target,
                generator=generator,
            )
            self._record_bf16_rounding_statistics_(
                state,
                current=parameter,
                target=target,
                rounded=rounded,
            )
            parameter.copy_(rounded)
        else:
            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
            parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
            if bool(group.get("nonnegative_parameter", False)):
                parameter.clamp_min_(0.0)
