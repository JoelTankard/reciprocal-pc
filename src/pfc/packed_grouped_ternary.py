from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PackedGroupedTernaryMatrix(nn.Module):
    """Row-major two-bit ternary weights with one scale per fixed group."""

    def __init__(
        self,
        codes: Tensor,
        scales: Tensor,
        *,
        input_features: int,
        output_features: int,
        leading_shape: tuple[int, ...],
        group_size: int,
    ) -> None:
        super().__init__()
        self.register_buffer("codes", codes.contiguous(), persistent=True)
        self.register_buffer("scales", scales.contiguous(), persistent=True)
        self.input_features = input_features
        self.output_features = output_features
        self.leading_shape = leading_shape
        self.group_size = group_size
        self._validate()

    @property
    def matrix_count(self) -> int:
        return math.prod(self.leading_shape)

    @property
    def weight_count(self) -> int:
        return self.matrix_count * self.input_features * self.output_features

    @property
    def storage_bytes(self) -> int:
        return (
            self.codes.numel() * self.codes.element_size()
            + self.scales.numel() * self.scales.element_size()
        )

    def _validate(self) -> None:
        if self.codes.dtype != torch.uint8 or self.codes.ndim != 1:
            raise ValueError("packed grouped ternary codes must be flat uint8")
        if not self.scales.is_floating_point() or self.scales.ndim != 1:
            raise ValueError("packed grouped ternary scales must be flat floating point")
        if self.codes.device != self.scales.device:
            raise ValueError("packed grouped ternary tensor devices differ")
        if self.input_features <= 0 or self.output_features <= 0:
            raise ValueError("packed grouped ternary dimensions must be positive")
        if not self.leading_shape or any(size <= 0 for size in self.leading_shape):
            raise ValueError("packed grouped ternary leading shape differs")
        if self.group_size <= 0:
            raise ValueError("packed grouped ternary group size must be positive")
        if self.codes.numel() != (self.weight_count + 3) // 4:
            raise ValueError("packed grouped ternary code count differs")
        if self.scales.numel() != (
            self.weight_count + self.group_size - 1
        ) // self.group_size:
            raise ValueError("packed grouped ternary scale count differs")
        if not bool(torch.isfinite(self.scales).all()):
            raise ValueError("packed grouped ternary scales must be finite")
        if bool((self.scales <= 0).any()):
            raise ValueError("packed grouped ternary scales must be positive")

    def matmul(
        self,
        inputs: Tensor,
        *,
        matrix_index: int | None = None,
        input_offset: int = 0,
        output_offset: int = 0,
        output_features: int | None = None,
        absolute_states: bool = False,
        walsh_token_ids: Tensor | None = None,
        walsh_input: bool = False,
        walsh_output: bool = False,
    ) -> Tensor:
        selected_outputs = output_features or self.output_features - output_offset
        if inputs.is_cuda:
            from pfc.triton_packed_grouped_ternary import (
                triton_packed_grouped_ternary_matmul,
            )

            return triton_packed_grouped_ternary_matmul(
                inputs,
                self,
                matrix_index=matrix_index,
                input_offset=input_offset,
                output_offset=output_offset,
                output_features=selected_outputs,
                absolute_states=absolute_states,
                walsh_token_ids=walsh_token_ids,
                walsh_input=walsh_input,
                walsh_output=walsh_output,
            )
        if walsh_token_ids is not None or walsh_input or walsh_output:
            raise ValueError("Walsh-bound packed matmul requires CUDA")
        return packed_grouped_ternary_matmul_reference(
            inputs,
            self,
            matrix_index=matrix_index,
            input_offset=input_offset,
            output_offset=output_offset,
            output_features=selected_outputs,
            absolute_states=absolute_states,
        )


def _validated_states(states: Tensor) -> Tensor:
    if states.ndim < 3:
        raise ValueError("grouped ternary matrix needs a leading matrix dimension")
    integer = states.detach().to(dtype=torch.int8)
    if not torch.equal(integer.to(dtype=states.dtype), states.detach()):
        raise ValueError("grouped ternary states contain non-integral values")
    if bool(((integer < -1) | (integer > 1)).any()):
        raise ValueError("grouped ternary states lie outside {-1, 0, +1}")
    return integer


def pack_grouped_ternary_matrix(
    states: Tensor,
    scales: Tensor,
    *,
    group_size: int = 128,
    scale_dtype: torch.dtype = torch.float16,
) -> PackedGroupedTernaryMatrix:
    integer = _validated_states(states)
    weight_count = integer.numel()
    expected_scales = (weight_count + group_size - 1) // group_size
    if scales.ndim != 1 or scales.numel() != expected_scales:
        raise ValueError("grouped ternary source scale count differs")
    if not scales.is_floating_point() or not bool(torch.isfinite(scales).all()):
        raise ValueError("grouped ternary source scales differ")
    if bool((scales <= 0).any()):
        raise ValueError("grouped ternary source scales must be positive")

    flat = integer.reshape(-1).to(dtype=torch.uint8).add_(1)
    padding = (-weight_count) % 4
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding), value=1)
    codes = flat.reshape(-1, 4)
    packed = (
        codes[:, 0]
        | torch.bitwise_left_shift(codes[:, 1], 2)
        | torch.bitwise_left_shift(codes[:, 2], 4)
        | torch.bitwise_left_shift(codes[:, 3], 6)
    )
    return PackedGroupedTernaryMatrix(
        packed,
        scales.detach().to(device=states.device, dtype=scale_dtype),
        input_features=int(states.shape[-2]),
        output_features=int(states.shape[-1]),
        leading_shape=tuple(int(size) for size in states.shape[:-2]),
        group_size=group_size,
    )


def unpack_grouped_ternary_states(packed: PackedGroupedTernaryMatrix) -> Tensor:
    shifts = torch.tensor(
        (0, 2, 4, 6),
        device=packed.codes.device,
        dtype=torch.uint8,
    )
    codes = torch.bitwise_and(
        torch.bitwise_right_shift(packed.codes.unsqueeze(-1), shifts),
        0b11,
    ).reshape(-1)[: packed.weight_count]
    if bool((codes == 0b11).any()):
        raise ValueError("packed grouped ternary stream contains reserved code 3")
    return codes.to(dtype=torch.int8).sub_(1).reshape(
        *packed.leading_shape,
        packed.input_features,
        packed.output_features,
    )


def dense_weight_from_grouped_ternary(
    packed: PackedGroupedTernaryMatrix,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    states = unpack_grouped_ternary_states(packed).reshape(-1).to(dtype=dtype)
    scale_per_weight = packed.scales.repeat_interleave(packed.group_size)[
        : packed.weight_count
    ].to(dtype=dtype)
    return (states * scale_per_weight).reshape(
        *packed.leading_shape,
        packed.input_features,
        packed.output_features,
    )


def _validate_matmul_selection(
    inputs: Tensor,
    packed: PackedGroupedTernaryMatrix,
    *,
    matrix_index: int | None,
    input_offset: int,
    output_offset: int,
    output_features: int,
) -> None:
    if inputs.ndim < 2:
        raise ValueError("packed grouped ternary input rank differs")
    if matrix_index is None:
        leading_rank = len(packed.leading_shape)
        if tuple(inputs.shape[:leading_rank]) != packed.leading_shape:
            raise ValueError("packed grouped ternary input leading shape differs")
    elif not 0 <= matrix_index < packed.matrix_count:
        raise ValueError("packed grouped ternary matrix index differs")
    if input_offset < 0 or input_offset + inputs.shape[-1] > packed.input_features:
        raise ValueError("packed grouped ternary input selection differs")
    if output_features <= 0 or output_offset < 0 or (
        output_offset + output_features > packed.output_features
    ):
        raise ValueError("packed grouped ternary output selection differs")


def packed_grouped_ternary_matmul_reference(
    inputs: Tensor,
    packed: PackedGroupedTernaryMatrix,
    *,
    matrix_index: int | None,
    input_offset: int,
    output_offset: int,
    output_features: int,
    absolute_states: bool = False,
) -> Tensor:
    _validate_matmul_selection(
        inputs,
        packed,
        matrix_index=matrix_index,
        input_offset=input_offset,
        output_offset=output_offset,
        output_features=output_features,
    )
    weight = dense_weight_from_grouped_ternary(packed, dtype=inputs.dtype)
    if absolute_states:
        weight = weight.abs()
    if matrix_index is not None:
        weight = weight.reshape(
            packed.matrix_count,
            packed.input_features,
            packed.output_features,
        )[matrix_index]
    weight = weight[
        ...,
        input_offset : input_offset + inputs.shape[-1],
        output_offset : output_offset + output_features,
    ]
    return torch.matmul(inputs, weight)


class PackedGroupedTernaryFFN(nn.Module):
    def __init__(
        self,
        ffn_in: PackedGroupedTernaryMatrix,
        ffn_out: PackedGroupedTernaryMatrix,
        ffn_out_dale_other: PackedGroupedTernaryMatrix | None = None,
        *,
        sign_lifted_dual_transition: bool = False,
    ) -> None:
        super().__init__()
        self.ffn_in = ffn_in
        self.ffn_out = ffn_out
        self.ffn_out_dale_other = ffn_out_dale_other
        self.sign_lifted_dual_transition = sign_lifted_dual_transition
        if ffn_in.leading_shape != ffn_out.leading_shape:
            raise ValueError("packed FFN layer counts differ")
        expected_output_input = (
            ffn_in.output_features
            if sign_lifted_dual_transition
            else ffn_in.output_features // 2
        )
        if ffn_out.input_features != expected_output_input:
            raise ValueError("packed SwiGLU FFN dimensions differ")
        if sign_lifted_dual_transition and ffn_out_dale_other is not None:
            raise ValueError("sign-lifted and Dale packed modes are exclusive")
        if ffn_out_dale_other is not None and (
            ffn_out_dale_other.leading_shape != (3, *ffn_out.leading_shape)
            or ffn_out_dale_other.input_features != ffn_out.input_features
            or ffn_out_dale_other.output_features != ffn_out.output_features
            or ffn_out_dale_other.group_size != ffn_out.group_size
        ):
            raise ValueError("packed Dale FFN companion dimensions differ")

    @property
    def storage_bytes(self) -> int:
        return (
            self.ffn_in.storage_bytes
            + self.ffn_out.storage_bytes
            + (
                self.ffn_out_dale_other.storage_bytes
                if self.ffn_out_dale_other is not None
                else 0
            )
        )

    @staticmethod
    def _owner_index(
        packed: PackedGroupedTernaryMatrix,
        layer: int | None,
    ) -> int | None:
        if packed.matrix_count == 1:
            return 0
        return layer

    def project_in(
        self,
        inputs: Tensor,
        *,
        layer: int | None,
        branch: int,
        walsh_token_ids: Tensor | None = None,
    ) -> Tensor:
        half = self.ffn_in.input_features // 2
        return self.ffn_in.matmul(
            inputs,
            matrix_index=self._owner_index(self.ffn_in, layer),
            input_offset=half if branch == 0 else 0,
            walsh_token_ids=walsh_token_ids,
            walsh_input=walsh_token_ids is not None,
        )

    def project_out(
        self,
        inputs: Tensor,
        *,
        layer: int | None,
        branch: int,
        walsh_token_ids: Tensor | None = None,
    ) -> Tensor:
        half = self.ffn_out.output_features // 2
        output_offset = 0 if branch == 0 else half
        if self.ffn_out_dale_other is None:
            if self.sign_lifted_dual_transition:
                from pfc.phase705_sign_lifted_transition import lift_sign_channels

                inputs = lift_sign_channels(inputs)
            return self.ffn_out.matmul(
                inputs,
                matrix_index=self._owner_index(self.ffn_out, layer),
                output_offset=output_offset,
                output_features=half,
                walsh_token_ids=walsh_token_ids,
                walsh_input=walsh_token_ids is not None,
                walsh_output=walsh_token_ids is not None,
            )
        positive = torch.relu(inputs)
        negative = torch.relu(-inputs)
        primary = self.ffn_out.matmul(
            positive,
            matrix_index=layer,
            output_offset=output_offset,
            output_features=half,
        )
        if layer is None:
            companions = self.ffn_out_dale_other.matmul(
                torch.stack((positive, negative, negative), dim=0),
                matrix_index=None,
                output_offset=output_offset,
                output_features=half,
            )
        else:
            depth = self.ffn_out.leading_shape[0]
            companions = torch.stack(
                tuple(
                    self.ffn_out_dale_other.matmul(
                        positive if bank == 0 else negative,
                        matrix_index=bank * depth + layer,
                        output_offset=output_offset,
                        output_features=half,
                    )
                    for bank in range(3)
                ),
                dim=0,
            )
        return primary + companions[1] - companions[0] - companions[2]

    def project_out_abs(
        self,
        inputs: Tensor,
        *,
        layer: int | None,
        branch: int,
    ) -> Tensor:
        half = self.ffn_out.output_features // 2
        return self.ffn_out.matmul(
            inputs,
            matrix_index=self._owner_index(self.ffn_out, layer),
            output_offset=0 if branch == 0 else half,
            output_features=half,
            absolute_states=True,
        )


def install_packed_grouped_ternary_ffn(
    model: nn.Module,
    runtime: PackedGroupedTernaryFFN,
    *,
    release_dense: bool,
) -> None:
    config = model.config
    if not config.reversible_recurrent_coupling:
        raise ValueError("packed grouped FFN requires reversible coupling")
    if config.compact_active_triangular_coupling:
        raise ValueError("packed grouped FFN does not support compact coupling")
    if config.alternating_triangular_coupling:
        raise ValueError("packed grouped FFN does not support alternating coupling")
    if config.exact_packed_reversible_projection_prefill:
        raise ValueError("packed grouped FFN conflicts with packed prefill projection")
    transition_owner_count = (
        1 if config.depth_tied_reciprocal_transition_core else config.depth
    )
    expected_in = (transition_owner_count, config.width, 2 * config.ffn_width)
    expected_out = (
        transition_owner_count,
        2 * config.ffn_width
        if config.sign_lifted_dual_ternary_transition
        else config.ffn_width,
        config.width,
    )
    actual_in = (
        *runtime.ffn_in.leading_shape,
        runtime.ffn_in.input_features,
        runtime.ffn_in.output_features,
    )
    actual_out = (
        *runtime.ffn_out.leading_shape,
        runtime.ffn_out.input_features,
        runtime.ffn_out.output_features,
    )
    if actual_in != expected_in or actual_out != expected_out:
        raise ValueError("packed grouped FFN model dimensions differ")
    if config.ternary_dale_quadrant_transition != (
        runtime.ffn_out_dale_other is not None
    ):
        raise ValueError("packed grouped FFN Dale mode differs")
    if config.sign_lifted_dual_ternary_transition != (
        runtime.sign_lifted_dual_transition
    ):
        raise ValueError("packed grouped FFN sign-lifted mode differs")
    if hasattr(model, "_packed_grouped_ternary_ffn"):
        raise ValueError("packed grouped FFN is already installed")
    model.add_module("_packed_grouped_ternary_ffn", runtime)
    if release_dense:
        model.register_parameter("ffn_in", None)
        model.register_parameter("ffn_out", None)
        if config.ternary_dale_quadrant_transition:
            model.register_parameter("ffn_out_dale_other", None)


def install_packed_grouped_ternary_handoff_gate(
    model: nn.Module,
    runtime: PackedGroupedTernaryMatrix,
    *,
    release_dense: bool,
) -> None:
    config = model.config
    if not config.token_gated_diagonal_handoff:
        raise ValueError("packed handoff gate requires token-gated handoff")
    expected = (1, config.width, config.width)
    actual = (
        *runtime.leading_shape,
        runtime.input_features,
        runtime.output_features,
    )
    if actual != expected:
        raise ValueError("packed handoff gate dimensions differ")
    if hasattr(model, "_packed_grouped_ternary_handoff_gate"):
        raise ValueError("packed handoff gate is already installed")
    model.add_module("_packed_grouped_ternary_handoff_gate", runtime)
    if release_dense:
        model.register_parameter("diagonal_handoff_token_gate", None)


def install_packed_grouped_ternary_affine_scale(
    model: nn.Module,
    runtime: PackedGroupedTernaryMatrix,
    *,
    release_dense: bool,
) -> None:
    config = model.config
    if not config.affine_reversible_transition:
        raise ValueError("packed affine scale requires affine reversible transition")
    owner_count = 1 if config.depth_tied_reciprocal_transition_core else config.depth
    half = config.width // 2
    expected = (owner_count, half, half)
    actual = (
        *runtime.leading_shape,
        runtime.input_features,
        runtime.output_features,
    )
    if actual != expected:
        raise ValueError("packed affine scale dimensions differ")
    if hasattr(model, "_packed_grouped_ternary_affine_scale"):
        raise ValueError("packed affine scale is already installed")
    model.add_module("_packed_grouped_ternary_affine_scale", runtime)
    if release_dense:
        model.register_parameter("affine_reversible_scale", None)


def install_packed_causal_ternary_fir_input_taps(
    model: nn.Module,
    runtime: PackedGroupedTernaryMatrix,
    *,
    release_dense: bool,
) -> None:
    config = model.config
    if not config.causal_ternary_fir_input_state:
        raise ValueError("packed causal FIR taps require causal FIR input state")
    expected = (config.depth, 4, config.width)
    actual = (
        *runtime.leading_shape,
        runtime.input_features,
        runtime.output_features,
    )
    if actual != expected:
        raise ValueError("packed causal FIR tap dimensions differ")
    if (
        config.causal_ternary_fir_fp32_deployment_scales
        and runtime.scales.dtype != torch.float32
    ):
        raise ValueError("packed causal FIR deployment scales must remain FP32")
    if hasattr(model, "_packed_causal_ternary_fir_input_taps"):
        raise ValueError("packed causal FIR taps are already installed")
    model.add_module("_packed_causal_ternary_fir_input_taps", runtime)
    if release_dense:
        model.register_parameter("causal_ternary_fir_input_taps", None)


def install_packed_lagged_latent_error_settling_gain(
    model: nn.Module,
    runtime: PackedGroupedTernaryMatrix,
    *,
    release_dense: bool,
) -> None:
    config = model.config
    if not config.lagged_latent_error_settling_carrier:
        raise ValueError("packed latent-error gain requires settling carrier")
    expected = (config.depth, 1, config.width)
    actual = (
        *runtime.leading_shape,
        runtime.input_features,
        runtime.output_features,
    )
    if actual != expected:
        raise ValueError("packed latent-error settling gain dimensions differ")
    if runtime.scales.dtype != torch.float32:
        raise ValueError("packed latent-error deployment scales must remain FP32")
    if hasattr(model, "_packed_lagged_latent_error_settling_gain"):
        raise ValueError("packed latent-error settling gain is already installed")
    model.add_module("_packed_lagged_latent_error_settling_gain", runtime)
    if release_dense:
        model.register_parameter("latent_error_settling_gain", None)


def model_tensor_storage_bytes(model: nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in (*model.parameters(), *model.buffers())
    )
