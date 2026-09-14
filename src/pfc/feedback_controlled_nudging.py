from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pfc.native_pc_lm import (
    NativeReciprocalPCLanguageModel,
    _affine_scan,
    _group4_affine_scan,
    _pair_affine_scan,
    _rms_norm,
)


def _rms_normalize(value: Tensor, eps: float) -> Tensor:
    inverse_rms = torch.rsqrt(
        value.float().square().mean(dim=-1, keepdim=True) + eps
    )
    return value * inverse_rms.to(value.dtype)


def _cap_rms(value: Tensor, maximum: float, eps: float) -> Tensor:
    squared_rms = value.float().square().mean(dim=-1, keepdim=True)
    exact_rms = squared_rms.sqrt()
    safe_rms = squared_rms.clamp_min(eps**2).sqrt()
    rms = exact_rms.detach() + safe_rms - safe_rms.detach()
    scale = torch.clamp(maximum / (rms + eps), max=1.0)
    return value * scale.to(value.dtype)


@dataclass(frozen=True)
class ActivityFeedbackControllerConfig:
    depth: int
    width: int
    rank: int = 64
    seed: int = 152
    output_rms_cap: float = 1.0
    eps: float = 1e-8

    @property
    def learned_value_count(self) -> int:
        projection_values = 5 * self.depth * self.width * self.rank
        bias_values = self.depth * self.rank
        return projection_values + bias_values


class ActivityFeedbackController(nn.Module):
    """Phase151 tokenwise signal reused as an activity-control drive."""

    def __init__(self, config: ActivityFeedbackControllerConfig) -> None:
        super().__init__()
        if min(config.depth, config.width, config.rank) <= 0:
            raise ValueError("feedback dimensions must be positive")
        if config.output_rms_cap <= 0 or config.eps <= 0:
            raise ValueError("feedback bounds must be positive")
        self.config = config
        down_shape = (config.depth, config.width, config.rank)
        up_shape = (config.depth, config.rank, config.width)
        self.task_down = nn.Parameter(torch.empty(down_shape))
        self.error_down = nn.Parameter(torch.empty(down_shape))
        self.raw_down = nn.Parameter(torch.empty(down_shape))
        self.displacement_down = nn.Parameter(torch.empty(down_shape))
        self.output_up = nn.Parameter(torch.empty(up_shape))
        self.state_bias = nn.Parameter(torch.zeros(config.depth, config.rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
        input_std = self.config.width**-0.5
        output_std = 0.01 * self.config.rank**-0.5
        with torch.no_grad():
            for parameter in (
                self.task_down,
                self.error_down,
                self.raw_down,
                self.displacement_down,
            ):
                values = torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=torch.float32,
                )
                parameter.copy_(values.mul_(input_std))
            values = torch.randn(
                self.output_up.shape,
                generator=generator,
                dtype=torch.float32,
            )
            self.output_up.copy_(values.mul_(output_std))
            self.state_bias.zero_()

    def load_phase151_state_free_(self, state: Mapping[str, Tensor]) -> None:
        expected = set(self.state_dict())
        available = {name: value for name, value in state.items() if name in expected}
        if set(available) != expected:
            missing = sorted(expected - set(available))
            raise ValueError(f"Phase151 state-free warm start is missing: {missing}")
        self.load_state_dict(available)

    def forward(
        self,
        base_direction: Tensor,
        raw_state: Tensor,
        settled_state: Tensor,
    ) -> Tensor:
        if base_direction.ndim != 4:
            raise ValueError(
                "feedback inputs must have [layer, batch, sequence, width]"
            )
        if base_direction.shape != raw_state.shape:
            raise ValueError("base direction and raw state shapes differ")
        if base_direction.shape != settled_state.shape:
            raise ValueError("base direction and settled state shapes differ")
        if (
            base_direction.shape[0] != self.config.depth
            or base_direction.shape[-1] != self.config.width
        ):
            raise ValueError("feedback depth or width differs")

        normalized_raw = _rms_normalize(raw_state, self.config.eps)
        normalized_displacement = _rms_normalize(
            settled_state - raw_state,
            self.config.eps,
        )
        task_code = torch.einsum(
            "lbsw,lwr->lbsr",
            base_direction,
            self.task_down,
        )
        state_drive = torch.tanh(
            torch.einsum(
                "lbsw,lwr->lbsr",
                base_direction,
                self.error_down,
            )
            + torch.einsum(
                "lbsw,lwr->lbsr",
                normalized_raw,
                self.raw_down,
            )
            + torch.einsum(
                "lbsw,lwr->lbsr",
                normalized_displacement,
                self.displacement_down,
            )
            + self.state_bias[:, None, None]
        )
        correction = torch.einsum(
            "lbsr,lrw->lbsw",
            task_code * (1.0 + torch.tanh(state_drive)),
            self.output_up,
        )
        return _cap_rms(
            base_direction + correction,
            self.config.output_rms_cap,
            self.config.eps,
        )


@dataclass(frozen=True)
class SharedDirectionMagnitudeControllerConfig:
    depth: int
    width: int
    hidden_width: int = 16
    seed: int = 160
    eps: float = 1e-8

    @property
    def learned_value_count(self) -> int:
        feature_count = 13
        return (
            feature_count * self.hidden_width
            + self.hidden_width
            + self.hidden_width * self.hidden_width
            + self.hidden_width
            + 2 * (self.hidden_width + 1)
        )


class SharedDirectionMagnitudeController(nn.Module):
    """Shape-independent local credit rule with separate output heads."""

    feature_count = 13

    def __init__(
        self,
        config: SharedDirectionMagnitudeControllerConfig,
    ) -> None:
        super().__init__()
        if min(config.depth, config.width, config.hidden_width) <= 0:
            raise ValueError("shared controller dimensions must be positive")
        if config.eps <= 0:
            raise ValueError("shared controller epsilon must be positive")
        self.config = config
        self.input_layer = nn.Linear(self.feature_count, config.hidden_width)
        self.hidden_layer = nn.Linear(config.hidden_width, config.hidden_width)
        self.direction_head = nn.Linear(config.hidden_width, 1)
        self.magnitude_head = nn.Linear(config.hidden_width, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
        with torch.no_grad():
            for layer in (self.input_layer, self.hidden_layer):
                values = torch.randn(
                    layer.weight.shape,
                    generator=generator,
                    dtype=torch.float32,
                )
                layer.weight.copy_(values.mul_(layer.in_features**-0.5))
                layer.bias.zero_()
            self.direction_head.weight.zero_()
            self.direction_head.bias.zero_()
            self.magnitude_head.weight.zero_()
            self.magnitude_head.bias.zero_()

    def _features(
        self,
        base_direction: Tensor,
        raw_state: Tensor,
        settled_state: Tensor,
    ) -> Tensor:
        normalized_base = _rms_normalize(base_direction, self.config.eps)
        normalized_raw = _rms_normalize(raw_state, self.config.eps)
        normalized_displacement = _rms_normalize(
            settled_state - raw_state,
            self.config.eps,
        )
        coordinate = torch.stack(
            (
                normalized_base,
                normalized_raw,
                normalized_displacement,
                normalized_base * normalized_raw,
                normalized_base * normalized_displacement,
                normalized_raw * normalized_displacement,
            ),
            dim=-1,
        )
        pooled = coordinate.mean(dim=-2, keepdim=True).expand_as(coordinate)
        depth = base_direction.shape[0]
        layer_coordinate = (
            2.0
            * (
                torch.arange(
                    depth,
                    device=base_direction.device,
                    dtype=base_direction.dtype,
                )
                + 0.5
            )
            / depth
            - 1.0
        ).view(depth, 1, 1, 1, 1)
        layer_coordinate = layer_coordinate.expand(*coordinate.shape[:-1], 1)
        return torch.cat((coordinate, pooled, layer_coordinate), dim=-1)

    def forward_components(
        self,
        base_direction: Tensor,
        raw_state: Tensor,
        settled_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if base_direction.ndim != 4:
            raise ValueError(
                "shared controller inputs must have [layer, batch, sequence, width]"
            )
        if base_direction.shape != raw_state.shape:
            raise ValueError("shared controller base and raw shapes differ")
        if base_direction.shape != settled_state.shape:
            raise ValueError("shared controller base and settled shapes differ")
        hidden = F.silu(self.input_layer(self._features(
            base_direction,
            raw_state,
            settled_state,
        )))
        hidden = F.silu(self.hidden_layer(hidden))
        base_rms = (
            base_direction.float().square().mean(dim=-1, keepdim=True)
            + self.config.eps
        ).sqrt().to(base_direction.dtype)
        direction = base_direction + base_rms * torch.tanh(
            self.direction_head(hidden).squeeze(-1)
        )
        pooled_hidden = hidden.mean(dim=-2)
        magnitude = F.softplus(
            self.magnitude_head(pooled_hidden).squeeze(-1)
        ) / math.log(2.0)
        return direction, magnitude.unsqueeze(-1)

    def forward(
        self,
        base_direction: Tensor,
        raw_state: Tensor,
        settled_state: Tensor,
    ) -> Tensor:
        direction, magnitude = self.forward_components(
            base_direction,
            raw_state,
            settled_state,
        )
        return direction * magnitude


@dataclass(frozen=True)
class SharedSlotDirectionMagnitudeControllerConfig:
    depth: int
    width: int
    hidden_width: int = 16
    slots: int = 8
    seed: int = 161
    eps: float = 1e-8

    @property
    def learned_value_count(self) -> int:
        feature_count = 13
        trunk = (
            feature_count * self.hidden_width
            + self.hidden_width
            + self.hidden_width * self.hidden_width
            + self.hidden_width
        )
        slot_router = (
            self.slots * self.hidden_width
            + self.hidden_width * self.hidden_width
            + self.hidden_width
        )
        heads = 2 * self.hidden_width + 1 + self.hidden_width + 1
        return trunk + slot_router + heads


class SharedSlotDirectionMagnitudeController(
    SharedDirectionMagnitudeController
):
    """Shared inducing slots route credit across layers and channels."""

    def __init__(
        self,
        config: SharedSlotDirectionMagnitudeControllerConfig,
    ) -> None:
        if config.slots <= 0:
            raise ValueError("shared slot count must be positive")
        super().__init__(
            SharedDirectionMagnitudeControllerConfig(
                depth=config.depth,
                width=config.width,
                hidden_width=config.hidden_width,
                seed=config.seed,
                eps=config.eps,
            )
        )
        self.config = config
        self.slot_queries = nn.Parameter(
            torch.empty(config.slots, config.hidden_width)
        )
        self.value_layer = nn.Linear(config.hidden_width, config.hidden_width)
        self.direction_head = nn.Linear(2 * config.hidden_width, 1)
        self.reset_slot_parameters()

    def reset_slot_parameters(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed + 1)
        with torch.no_grad():
            slot_values = torch.randn(
                self.slot_queries.shape,
                generator=generator,
                dtype=torch.float32,
            )
            self.slot_queries.copy_(
                slot_values.mul_(self.config.hidden_width**-0.5)
            )
            value_weights = torch.randn(
                self.value_layer.weight.shape,
                generator=generator,
                dtype=torch.float32,
            )
            self.value_layer.weight.copy_(
                value_weights.mul_(self.config.hidden_width**-0.5)
            )
            self.value_layer.bias.zero_()
            self.direction_head.weight.zero_()
            self.direction_head.bias.zero_()

    def _slot_mix(self, hidden: Tensor) -> Tensor:
        depth, batch, sequence, width, hidden_width = hidden.shape
        scores = torch.einsum(
            "dbswh,kh->dbswk",
            hidden,
            self.slot_queries,
        ) / math.sqrt(hidden_width)
        flat_scores = scores.permute(1, 2, 0, 3, 4).reshape(
            batch,
            sequence,
            depth * width,
            self.config.slots,
        )
        flat_values = self.value_layer(hidden).permute(1, 2, 0, 3, 4).reshape(
            batch,
            sequence,
            depth * width,
            hidden_width,
        )
        slot_values = torch.einsum(
            "bsnk,bsnh->bskh",
            torch.softmax(flat_scores, dim=2),
            flat_values,
        )
        mixed = torch.einsum(
            "bsnk,bskh->bsnh",
            torch.softmax(flat_scores, dim=-1),
            slot_values,
        )
        return mixed.reshape(
            batch,
            sequence,
            depth,
            width,
            hidden_width,
        ).permute(2, 0, 1, 3, 4)

    def forward_components(
        self,
        base_direction: Tensor,
        raw_state: Tensor,
        settled_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if base_direction.ndim != 4:
            raise ValueError(
                "shared slot inputs must have [layer, batch, sequence, width]"
            )
        if base_direction.shape != raw_state.shape:
            raise ValueError("shared slot base and raw shapes differ")
        if base_direction.shape != settled_state.shape:
            raise ValueError("shared slot base and settled shapes differ")
        hidden = F.silu(self.input_layer(self._features(
            base_direction,
            raw_state,
            settled_state,
        )))
        hidden = F.silu(self.hidden_layer(hidden))
        mixed = self._slot_mix(hidden)
        base_rms = (
            base_direction.float().square().mean(dim=-1, keepdim=True)
            + self.config.eps
        ).sqrt().to(base_direction.dtype)
        direction = base_direction + base_rms * torch.tanh(
            self.direction_head(torch.cat((hidden, mixed), dim=-1)).squeeze(-1)
        )
        magnitude = F.softplus(
            self.magnitude_head(hidden.mean(dim=-2)).squeeze(-1)
        ) / math.log(2.0)
        return direction, magnitude.unsqueeze(-1)


@dataclass(frozen=True)
class DualStreamRecurrentErrorBusControllerConfig:
    depth: int
    width: int
    hidden_width: int = 16
    slots: int = 8
    seed: int = 162
    eps: float = 1e-8

    @property
    def learned_value_count(self) -> int:
        hidden = self.hidden_width
        magnitude_trunk_and_head = (
            13 * hidden
            + hidden
            + hidden * hidden
            + hidden
            + hidden
            + 1
        )
        streams_and_bus = (
            2 * hidden
            + hidden
            + 4 * hidden
            + hidden
            + self.slots * hidden
            + 2 * (hidden * hidden + hidden)
            + self.slots * hidden
        )
        direction_head = 3 * hidden + 1
        return magnitude_trunk_and_head + streams_and_bus + direction_head


class DualStreamRecurrentErrorBusController(
    SharedDirectionMagnitudeController
):
    """Routes a separate error stream through a reverse-depth slot bus."""

    def __init__(
        self,
        config: DualStreamRecurrentErrorBusControllerConfig,
    ) -> None:
        if config.slots <= 0:
            raise ValueError("recurrent error bus slot count must be positive")
        super().__init__(
            SharedDirectionMagnitudeControllerConfig(
                depth=config.depth,
                width=config.width,
                hidden_width=config.hidden_width,
                seed=config.seed,
                eps=config.eps,
            )
        )
        self.config = config
        hidden = config.hidden_width
        self.error_encoder = nn.Linear(2, hidden)
        self.activity_encoder = nn.Linear(4, hidden)
        self.slot_queries = nn.Parameter(torch.empty(config.slots, hidden))
        self.error_value_layer = nn.Linear(hidden, hidden)
        self.activity_query_layer = nn.Linear(hidden, hidden)
        self.retention_logits = nn.Parameter(torch.empty(config.slots, hidden))
        self.direction_head = nn.Linear(3 * hidden, 1)
        self.reset_bus_parameters()

    def reset_bus_parameters(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed + 1)
        with torch.no_grad():
            for layer in (
                self.error_encoder,
                self.activity_encoder,
                self.error_value_layer,
                self.activity_query_layer,
            ):
                values = torch.randn(
                    layer.weight.shape,
                    generator=generator,
                    dtype=torch.float32,
                )
                layer.weight.copy_(values.mul_(layer.in_features**-0.5))
                layer.bias.zero_()
            slot_values = torch.randn(
                self.slot_queries.shape,
                generator=generator,
                dtype=torch.float32,
            )
            self.slot_queries.copy_(
                slot_values.mul_(self.config.hidden_width**-0.5)
            )
            self.retention_logits.zero_()
            self.direction_head.weight.zero_()
            self.direction_head.bias.zero_()

    def _layer_coordinate(self, reference: Tensor) -> Tensor:
        depth = reference.shape[0]
        coordinate = (
            2.0
            * (
                torch.arange(
                    depth,
                    device=reference.device,
                    dtype=reference.dtype,
                )
                + 0.5
            )
            / depth
            - 1.0
        ).view(depth, 1, 1, 1)
        return coordinate.expand_as(reference)

    def _stream_hidden(
        self,
        base_direction: Tensor,
        raw_state: Tensor,
        settled_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        normalized_base = _rms_normalize(base_direction, self.config.eps)
        normalized_raw = _rms_normalize(raw_state, self.config.eps)
        normalized_displacement = _rms_normalize(
            settled_state - raw_state,
            self.config.eps,
        )
        layer_coordinate = self._layer_coordinate(base_direction)
        error_hidden = F.silu(
            self.error_encoder(
                torch.stack((normalized_base, layer_coordinate), dim=-1)
            )
        )
        activity_hidden = F.silu(
            self.activity_encoder(
                torch.stack(
                    (
                        normalized_raw,
                        normalized_displacement,
                        normalized_raw * normalized_displacement,
                        layer_coordinate,
                    ),
                    dim=-1,
                )
            )
        )
        return error_hidden, activity_hidden

    def _recurrent_error_messages(
        self,
        error_hidden: Tensor,
        activity_hidden: Tensor,
    ) -> Tensor:
        depth, batch, sequence, _, hidden = error_hidden.shape
        bus = error_hidden.new_zeros(
            batch,
            sequence,
            self.config.slots,
            hidden,
        )
        retention = torch.sigmoid(self.retention_logits).view(
            1,
            1,
            self.config.slots,
            hidden,
        )
        messages: list[Tensor | None] = [None] * depth
        scale = math.sqrt(hidden)
        for layer_index in range(depth - 1, -1, -1):
            layer_error = error_hidden[layer_index]
            assignment = torch.softmax(
                torch.einsum(
                    "bswh,kh->bswk",
                    layer_error,
                    self.slot_queries,
                )
                / scale,
                dim=2,
            )
            layer_message = torch.einsum(
                "bswk,bswh->bskh",
                assignment,
                self.error_value_layer(layer_error),
            )
            bus = retention * bus + (1.0 - retention) * layer_message
            activity_query = self.activity_query_layer(
                activity_hidden[layer_index]
            )
            read_weights = torch.softmax(
                torch.einsum(
                    "bswh,bskh->bswk",
                    activity_query,
                    bus,
                )
                / scale,
                dim=-1,
            )
            messages[layer_index] = torch.einsum(
                "bswk,bskh->bswh",
                read_weights,
                bus,
            )
        return torch.stack(
            [message for message in messages if message is not None],
            dim=0,
        )

    def forward_components(
        self,
        base_direction: Tensor,
        raw_state: Tensor,
        settled_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if base_direction.ndim != 4:
            raise ValueError(
                "recurrent error bus inputs must have "
                "[layer, batch, sequence, width]"
            )
        if base_direction.shape != raw_state.shape:
            raise ValueError("recurrent error bus base and raw shapes differ")
        if base_direction.shape != settled_state.shape:
            raise ValueError(
                "recurrent error bus base and settled shapes differ"
            )
        magnitude_hidden = F.silu(
            self.input_layer(
                self._features(base_direction, raw_state, settled_state)
            )
        )
        magnitude_hidden = F.silu(self.hidden_layer(magnitude_hidden))
        error_hidden, activity_hidden = self._stream_hidden(
            base_direction,
            raw_state,
            settled_state,
        )
        error_message = self._recurrent_error_messages(
            error_hidden,
            activity_hidden,
        )
        base_rms = (
            base_direction.float().square().mean(dim=-1, keepdim=True)
            + self.config.eps
        ).sqrt().to(base_direction.dtype)
        direction = base_direction + base_rms * torch.tanh(
            self.direction_head(
                torch.cat(
                    (error_hidden, activity_hidden, error_message),
                    dim=-1,
                )
            ).squeeze(-1)
        )
        magnitude = F.softplus(
            self.magnitude_head(magnitude_hidden.mean(dim=-2)).squeeze(-1)
        ) / math.log(2.0)
        return direction, magnitude.unsqueeze(-1)


@dataclass(frozen=True)
class CenteredDualSignRecurrentErrorBusControllerConfig:
    depth: int
    width: int
    hidden_width: int = 16
    slots: int = 8
    seed: int = 163
    eps: float = 1e-8

    @property
    def learned_value_count(self) -> int:
        hidden = self.hidden_width
        magnitude_trunk_and_head = (
            13 * hidden
            + hidden
            + hidden * hidden
            + hidden
            + hidden
            + 1
        )
        activity_stream = 4 * hidden + hidden + hidden * hidden + hidden
        signed_error_buses = (
            2 * (2 * hidden + hidden)
            + 2 * self.slots * hidden
            + 2 * (hidden * hidden + hidden)
            + 2 * self.slots * hidden
        )
        direction_head = 5 * hidden + 1
        return (
            magnitude_trunk_and_head
            + activity_stream
            + signed_error_buses
            + direction_head
        )


class CenteredDualSignRecurrentErrorBusController(
    DualStreamRecurrentErrorBusController
):
    """Centers task error and routes positive and negative buses separately."""

    def __init__(
        self,
        config: CenteredDualSignRecurrentErrorBusControllerConfig,
    ) -> None:
        super().__init__(
            DualStreamRecurrentErrorBusControllerConfig(
                depth=config.depth,
                width=config.width,
                hidden_width=config.hidden_width,
                slots=config.slots,
                seed=config.seed,
                eps=config.eps,
            )
        )
        self.config = config
        hidden = config.hidden_width
        del self.error_encoder
        self.positive_error_encoder = nn.Linear(2, hidden)
        self.negative_error_encoder = nn.Linear(2, hidden)
        self.slot_queries = nn.Parameter(
            torch.empty(2, config.slots, hidden)
        )
        self.error_value_layers = nn.ModuleList(
            (nn.Linear(hidden, hidden), nn.Linear(hidden, hidden))
        )
        del self.error_value_layer
        self.retention_logits = nn.Parameter(
            torch.empty(2, config.slots, hidden)
        )
        self.direction_head = nn.Linear(5 * hidden, 1)
        self.reset_signed_bus_parameters()

    def reset_signed_bus_parameters(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed + 1)
        with torch.no_grad():
            for layer in (
                self.activity_encoder,
                self.activity_query_layer,
                self.positive_error_encoder,
                self.negative_error_encoder,
                *self.error_value_layers,
            ):
                values = torch.randn(
                    layer.weight.shape,
                    generator=generator,
                    dtype=torch.float32,
                )
                layer.weight.copy_(values.mul_(layer.in_features**-0.5))
                layer.bias.zero_()
            slot_values = torch.randn(
                self.slot_queries.shape,
                generator=generator,
                dtype=torch.float32,
            )
            self.slot_queries.copy_(
                slot_values.mul_(self.config.hidden_width**-0.5)
            )
            self.retention_logits.zero_()
            self.direction_head.weight.zero_()
            self.direction_head.bias.zero_()

    def _centered_signed_error(
        self,
        base_direction: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        normalized = _rms_normalize(base_direction, self.config.eps)
        centered = normalized - normalized.mean(
            dim=(1, 2),
            keepdim=True,
        )
        centered = _rms_normalize(centered, self.config.eps)
        centered = centered - centered.mean(dim=(1, 2), keepdim=True)
        return centered, F.relu(centered), F.relu(-centered)

    def _stream_hidden(
        self,
        base_direction: Tensor,
        raw_state: Tensor,
        settled_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        _, positive_error, negative_error = self._centered_signed_error(
            base_direction
        )
        layer_coordinate = self._layer_coordinate(base_direction)
        positive_hidden = F.silu(
            self.positive_error_encoder(
                torch.stack((positive_error, layer_coordinate), dim=-1)
            )
        )
        negative_hidden = F.silu(
            self.negative_error_encoder(
                torch.stack((negative_error, layer_coordinate), dim=-1)
            )
        )
        normalized_raw = _rms_normalize(raw_state, self.config.eps)
        normalized_displacement = _rms_normalize(
            settled_state - raw_state,
            self.config.eps,
        )
        activity_hidden = F.silu(
            self.activity_encoder(
                torch.stack(
                    (
                        normalized_raw,
                        normalized_displacement,
                        normalized_raw * normalized_displacement,
                        layer_coordinate,
                    ),
                    dim=-1,
                )
            )
        )
        return torch.stack((positive_hidden, negative_hidden)), activity_hidden

    def _recurrent_error_messages(
        self,
        error_hidden: Tensor,
        activity_hidden: Tensor,
    ) -> Tensor:
        signs, depth, batch, sequence, _, hidden = error_hidden.shape
        if signs != 2:
            raise ValueError("dual-sign error hidden state must have two streams")
        bus = error_hidden.new_zeros(
            signs,
            batch,
            sequence,
            self.config.slots,
            hidden,
        )
        retention = torch.sigmoid(self.retention_logits).view(
            signs,
            1,
            1,
            self.config.slots,
            hidden,
        )
        messages: list[Tensor | None] = [None] * depth
        scale = math.sqrt(hidden)
        for layer_index in range(depth - 1, -1, -1):
            layer_error = error_hidden[:, layer_index]
            assignment = torch.softmax(
                torch.einsum(
                    "cbswh,ckh->cbswk",
                    layer_error,
                    self.slot_queries,
                )
                / scale,
                dim=3,
            )
            projected_error = torch.stack(
                (
                    self.error_value_layers[0](layer_error[0]),
                    self.error_value_layers[1](layer_error[1]),
                )
            )
            layer_message = torch.einsum(
                "cbswk,cbswh->cbskh",
                assignment,
                projected_error,
            )
            bus = retention * bus + (1.0 - retention) * layer_message
            activity_query = self.activity_query_layer(
                activity_hidden[layer_index]
            )
            read_weights = torch.softmax(
                torch.einsum(
                    "bswh,cbskh->cbswk",
                    activity_query,
                    bus,
                )
                / scale,
                dim=-1,
            )
            messages[layer_index] = torch.einsum(
                "cbswk,cbskh->cbswh",
                read_weights,
                bus,
            )
        return torch.stack(
            [message for message in messages if message is not None],
            dim=1,
        )

    def forward_components(
        self,
        base_direction: Tensor,
        raw_state: Tensor,
        settled_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if base_direction.ndim != 4:
            raise ValueError(
                "centered dual-sign inputs must have "
                "[layer, batch, sequence, width]"
            )
        if base_direction.shape != raw_state.shape:
            raise ValueError("centered dual-sign base and raw shapes differ")
        if base_direction.shape != settled_state.shape:
            raise ValueError("centered dual-sign base and settled shapes differ")
        magnitude_hidden = F.silu(
            self.input_layer(
                self._features(base_direction, raw_state, settled_state)
            )
        )
        magnitude_hidden = F.silu(self.hidden_layer(magnitude_hidden))
        signed_error_hidden, activity_hidden = self._stream_hidden(
            base_direction,
            raw_state,
            settled_state,
        )
        signed_messages = self._recurrent_error_messages(
            signed_error_hidden,
            activity_hidden,
        )
        base_rms = (
            base_direction.float().square().mean(dim=-1, keepdim=True)
            + self.config.eps
        ).sqrt().to(base_direction.dtype)
        direction_features = torch.cat(
            (
                signed_error_hidden[0],
                signed_error_hidden[1],
                activity_hidden,
                signed_messages[0],
                signed_messages[1],
            ),
            dim=-1,
        )
        direction = base_direction + base_rms * torch.tanh(
            self.direction_head(direction_features).squeeze(-1)
        )
        magnitude = F.softplus(
            self.magnitude_head(magnitude_hidden.mean(dim=-2)).squeeze(-1)
        ) / math.log(2.0)
        return direction, magnitude.unsqueeze(-1)


@dataclass(frozen=True)
class ControlledRawTrajectory:
    raw: Tensor
    displacement: Tensor
    temporal_carry: Tensor


def feedback_controlled_raw_trajectory(
    model: NativeReciprocalPCLanguageModel,
    *,
    lower: Tensor,
    observation: Tensor,
    free_raw: Tensor,
    layer: int,
    teaching_signal: Tensor,
    nudge_scale: float = 0.05,
) -> ControlledRawTrajectory:
    """Run task drive through the frozen forward recurrent operator."""

    if lower.shape != free_raw.shape or teaching_signal.shape != free_raw.shape:
        raise ValueError("controlled trajectory tensors must share shape")
    if lower.ndim != 3:
        raise ValueError("controlled trajectory must have [batch, sequence, width]")
    if observation.shape != lower.shape:
        raise ValueError("controlled observation shape differs")
    if not 0 <= layer < model.config.depth:
        raise ValueError("controlled trajectory layer is out of range")
    if nudge_scale <= 0:
        raise ValueError("controlled nudge scale must be positive")

    batch, sequence, width = lower.shape
    with torch.no_grad():
        source = lower
        if model.config.residual_stream:
            source = (
                model._residual_stream_scale(layer) * lower
                + model._sensory_reinjection_scale(layer) * observation
            )
        normalized = _rms_norm(
            source,
            model.state_norm[layer, 0],
            model.config.rms_eps,
        )
        temporal_rotation = model._temporal_rotation_for_normalized(
            normalized,
            layer,
        )
        decay = torch.sigmoid(
            normalized @ model.gate[layer] + model.gate_bias[layer, 0]
        )
        if temporal_rotation is None:
            operator: Tensor | None = None
        elif (
            model.pairwise_delta_down is not None
            and model.config.pairwise_delta_group_width == 4
        ):
            groups = width // 4
            operator = model._group4_transition_matrix(
                normalized,
                decay.reshape(batch, sequence, groups, 4),
                temporal_rotation.reshape(batch, sequence, groups, 2, 2),
                layer,
            ).detach()
        else:
            operator = model._pair_transition_matrix(
                normalized,
                decay.reshape(batch, sequence, width // 2, 2),
                temporal_rotation,
                layer,
            ).detach()
        decay = decay.detach()

    direct = nudge_scale * teaching_signal
    if temporal_rotation is None:
        displacement = _affine_scan(
            decay,
            direct,
            torch.zeros(batch, width, device=lower.device, dtype=lower.dtype),
        )
    elif model.config.pairwise_delta_group_width == 4:
        assert operator is not None
        groups = width // 4
        displacement = _group4_affine_scan(
            operator,
            direct.reshape(batch, sequence, groups, 4),
            torch.zeros(batch, groups, 4, device=lower.device, dtype=lower.dtype),
        ).reshape_as(free_raw)
    else:
        assert operator is not None
        pairs = width // 2
        displacement = _pair_affine_scan(
            operator,
            direct.reshape(batch, sequence, pairs, 2),
            torch.zeros(batch, pairs, 2, device=lower.device, dtype=lower.dtype),
        ).reshape_as(free_raw)
    return ControlledRawTrajectory(
        raw=free_raw + displacement,
        displacement=displacement,
        temporal_carry=displacement - direct,
    )
