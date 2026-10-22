from __future__ import annotations

import torch
from torch import Tensor

from pfc.phase414_triton_explicit_round_reverse_scan import (
    pair_affine_scan_explicit_round,
)


def reverse_pair_matrix_credit_explicit_round(
    matrix: Tensor,
    direction: Tensor,
) -> Tensor:
    """Exact reverse credit using the Phase414 explicit-round scan."""

    if matrix.ndim != 6 or matrix.shape[-2:] != (2, 2):
        raise ValueError(
            "matrix must have shape [layer, batch, sequence, pair, 2, 2]"
        )
    layers, batch, sequence, pairs = matrix.shape[:4]
    width = pairs * 2
    if direction.shape != (layers, batch, sequence, width):
        raise ValueError("direction shape does not match paired transition matrix")
    grouped_matrix = matrix.reshape(
        layers * batch,
        sequence,
        pairs,
        2,
        2,
    )
    paired_direction = direction.reshape(
        layers * batch,
        sequence,
        pairs,
        2,
    )
    future_reverse_matrix = torch.cat(
        (
            grouped_matrix[:, 1:].transpose(-1, -2),
            torch.zeros_like(grouped_matrix[:, :1]),
        ),
        dim=1,
    )
    reverse_credit = pair_affine_scan_explicit_round(
        future_reverse_matrix.flip(1),
        paired_direction.flip(1),
        torch.zeros(
            layers * batch,
            pairs,
            2,
            device=direction.device,
            dtype=direction.dtype,
        ),
    )
    return reverse_credit.reshape(
        layers,
        batch,
        sequence,
        width,
    ).flip(2)
