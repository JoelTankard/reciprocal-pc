from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pfc.packed_grouped_ternary import (
    PackedGroupedTernaryMatrix,
    pack_grouped_ternary_matrix,
)
from pfc.phase823_energy_preserving_self_bilinear_token_observation import (
    _observation_terms,
)


PROJECTED_SELF_BILINEAR_GROUP_SIZE = 128
PROJECTED_SELF_BILINEAR_WIDTH = 512


@dataclass(frozen=True)
class ProjectedSelfBilinearTerms:
    raw_rms: Tensor
    source: Tensor
    projected: Tensor
    activation: Tensor
    output: Tensor


@dataclass(frozen=True)
class ProjectedSelfBilinearVJP:
    input_gradient: Tensor
    projector_gradient: Tensor


def projected_self_bilinear_source(
    rows: Tensor,
    *,
    rms_eps: float,
) -> tuple[Tensor, Tensor]:
    """Return detached Phase823 RMS and orthogonal self-product source."""

    (
        raw_rms,
        _,
        _,
        _,
        _,
        _,
        _,
        bilinear,
        active,
    ) = _observation_terms(rows, rms_eps)
    source = torch.where(active, bilinear, torch.zeros_like(bilinear))
    return raw_rms.detach(), source.detach()


def _validate_projector(rows: Tensor, projector: Tensor) -> None:
    if projector.shape != (
        PROJECTED_SELF_BILINEAR_WIDTH,
        PROJECTED_SELF_BILINEAR_WIDTH,
    ):
        raise ValueError("projected self-bilinear matrix shape differs")
    if projector.device != rows.device or projector.dtype != rows.dtype:
        raise TypeError("projected self-bilinear tensors must match rows")


def projected_self_bilinear_token_observation(
    rows: Tensor,
    projector: Tensor,
    *,
    rms_eps: float,
) -> ProjectedSelfBilinearTerms:
    """Inject a bounded zero-start residual from the detached Phase823 source."""

    _validate_projector(rows, projector)
    raw_rms, source = projected_self_bilinear_source(rows, rms_eps=rms_eps)
    projected = torch.matmul(source, projector) / math.sqrt(
        PROJECTED_SELF_BILINEAR_WIDTH
    )
    activation = torch.tanh(projected)
    output = rows + raw_rms * activation
    return ProjectedSelfBilinearTerms(
        raw_rms=raw_rms,
        source=source,
        projected=projected,
        activation=activation,
        output=output,
    )


def projected_self_bilinear_token_observation_vjp(
    rows: Tensor,
    projector: Tensor,
    output_gradient: Tensor,
    *,
    rms_eps: float,
) -> ProjectedSelfBilinearVJP:
    """Apply direct local projector credit while preserving raw embedding credit."""

    if rows.shape != output_gradient.shape:
        raise ValueError("projected self-bilinear gradient shape differs")
    if rows.device != output_gradient.device or rows.dtype != output_gradient.dtype:
        raise TypeError("projected self-bilinear gradient must match rows")
    terms = projected_self_bilinear_token_observation(
        rows,
        projector,
        rms_eps=rms_eps,
    )
    projected_gradient = (
        terms.raw_rms
        * output_gradient
        * (1.0 - terms.activation.square())
    )
    source = terms.source.reshape(-1, PROJECTED_SELF_BILINEAR_WIDTH)
    projected_gradient = projected_gradient.reshape(
        -1,
        PROJECTED_SELF_BILINEAR_WIDTH,
    )
    projector_gradient = torch.matmul(
        source.transpose(0, 1),
        projected_gradient,
    ) / math.sqrt(PROJECTED_SELF_BILINEAR_WIDTH)
    return ProjectedSelfBilinearVJP(
        input_gradient=output_gradient,
        projector_gradient=projector_gradient,
    )


class PackedProjectedSelfBilinearObservation(nn.Module):
    def __init__(self, packed: PackedGroupedTernaryMatrix) -> None:
        super().__init__()
        if (
            packed.leading_shape != (1,)
            or packed.input_features != PROJECTED_SELF_BILINEAR_WIDTH
            or packed.output_features != PROJECTED_SELF_BILINEAR_WIDTH
            or packed.group_size != PROJECTED_SELF_BILINEAR_GROUP_SIZE
            or packed.scales.dtype != torch.float16
        ):
            raise ValueError("packed projected self-bilinear geometry differs")
        self.packed = packed

    @property
    def storage_bytes(self) -> int:
        return self.packed.storage_bytes

    def project(self, source: Tensor) -> Tensor:
        return self.packed.matmul(source, matrix_index=0)

    def observe(self, rows: Tensor, *, rms_eps: float) -> Tensor:
        raw_rms, source = projected_self_bilinear_source(rows, rms_eps=rms_eps)
        projected = self.project(source) / math.sqrt(
            PROJECTED_SELF_BILINEAR_WIDTH
        )
        return rows + raw_rms * torch.tanh(projected)


def pack_projected_self_bilinear_observation(
    states: Tensor,
    scales: Tensor,
) -> PackedProjectedSelfBilinearObservation:
    if states.shape != (
        PROJECTED_SELF_BILINEAR_WIDTH,
        PROJECTED_SELF_BILINEAR_WIDTH,
    ):
        raise ValueError("projected self-bilinear source states differ")
    packed = pack_grouped_ternary_matrix(
        states.unsqueeze(0),
        scales.clamp_min(torch.finfo(torch.float16).tiny),
        group_size=PROJECTED_SELF_BILINEAR_GROUP_SIZE,
        scale_dtype=torch.float16,
    )
    return PackedProjectedSelfBilinearObservation(packed)


def install_packed_projected_self_bilinear_observation(
    model: nn.Module,
    runtime: PackedProjectedSelfBilinearObservation,
    *,
    release_dense: bool,
) -> None:
    if not model.config.projected_self_bilinear_token_observation:
        raise ValueError("packed projected observation requires enabled architecture")
    parameter = getattr(
        model,
        "projected_self_bilinear_observation_matrix",
        None,
    )
    if parameter is None:
        raise ValueError("dense projected self-bilinear matrix is missing")
    if hasattr(model, "_phase827_packed_projected_self_bilinear_observation"):
        raise ValueError("packed projected observation is already installed")
    model.add_module(
        "_phase827_packed_projected_self_bilinear_observation",
        runtime,
    )
    if release_dense:
        model.register_parameter(
            "projected_self_bilinear_observation_matrix",
            None,
        )


__all__ = [
    "PROJECTED_SELF_BILINEAR_GROUP_SIZE",
    "PROJECTED_SELF_BILINEAR_WIDTH",
    "PackedProjectedSelfBilinearObservation",
    "ProjectedSelfBilinearTerms",
    "ProjectedSelfBilinearVJP",
    "install_packed_projected_self_bilinear_observation",
    "pack_projected_self_bilinear_observation",
    "projected_self_bilinear_source",
    "projected_self_bilinear_token_observation",
    "projected_self_bilinear_token_observation_vjp",
]
