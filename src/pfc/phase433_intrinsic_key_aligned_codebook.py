from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


@torch.no_grad()
def build_intrinsic_key_aligned_predictive_codebook(
    decoder: Tensor,
    learned_key_basis: Tensor,
) -> Tensor:
    """Encode vocabulary errors in the current learned predictive-key basis."""
    if decoder.ndim != 2:
        raise ValueError("decoder must be a matrix")
    if learned_key_basis.ndim != 2:
        raise ValueError("learned key basis must be a matrix")
    if decoder.shape[0] != learned_key_basis.shape[0]:
        raise ValueError("decoder and learned key widths differ")
    if learned_key_basis.shape[1] <= 0:
        raise ValueError("learned key basis rank must be positive")
    if decoder.device != learned_key_basis.device:
        raise ValueError("decoder and learned key basis devices differ")

    source = decoder.detach().to(dtype=torch.float64)
    basis = learned_key_basis.detach().to(dtype=torch.float64)
    codebook = F.normalize(source.T @ basis, dim=-1, eps=1e-12)
    return codebook.to(dtype=decoder.dtype)
