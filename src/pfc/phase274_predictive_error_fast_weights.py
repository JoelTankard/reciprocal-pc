from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F

from pfc.native_pc_lm import _constant_decay_scan


PREDICTIVE_FAST_WEIGHT_RANK = 32
PREDICTIVE_FAST_DECAY = 0.99
PREDICTIVE_SLOW_DECAY = math.sqrt(PREDICTIVE_FAST_DECAY)


def build_predictive_error_geometry(
    decoder: Tensor,
    rank: int = PREDICTIVE_FAST_WEIGHT_RANK,
) -> tuple[Tensor, Tensor]:
    """Build deterministic learned context basis and unit output codes."""
    if decoder.ndim != 2:
        raise ValueError("decoder must be a matrix")
    if not 0 < rank <= min(decoder.shape):
        raise ValueError("predictive geometry rank is invalid")
    source = decoder.detach().to(device="cpu", dtype=torch.float64)
    basis, _, _ = torch.linalg.svd(source, full_matrices=False)
    basis = basis[:, :rank]
    columns = torch.arange(rank)
    anchors = basis.abs().argmax(dim=0)
    signs = torch.sign(basis[anchors, columns])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    basis = basis * signs
    codebook = F.normalize(source.T @ basis, dim=-1, eps=1e-12)
    return (
        basis.to(device=decoder.device, dtype=decoder.dtype),
        codebook.to(device=decoder.device, dtype=decoder.dtype),
    )


@torch.no_grad()
def build_predictive_error_geometry_covariance(
    decoder: Tensor,
    rank: int = PREDICTIVE_FAST_WEIGHT_RANK,
) -> tuple[Tensor, Tensor]:
    """Build the same geometry without copying a CUDA decoder to the CPU."""
    if decoder.ndim != 2:
        raise ValueError("decoder must be a matrix")
    if not 0 < rank <= min(decoder.shape):
        raise ValueError("predictive geometry rank is invalid")
    source = decoder.detach().to(dtype=torch.float64)
    covariance = source @ source.T
    _, vectors = torch.linalg.eigh(covariance)
    basis = vectors[:, -rank:].flip(-1)
    columns = torch.arange(rank, device=basis.device)
    anchors = basis.abs().argmax(dim=0)
    signs = torch.sign(basis[anchors, columns])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    basis = basis * signs
    codebook = F.normalize(source.T @ basis, dim=-1, eps=1e-12)
    return basis.to(decoder.dtype), codebook.to(decoder.dtype)


def _debiased_fast_matrices(
    associations: Tensor,
    decay: float,
) -> Tensor:
    batch, sequence, rank, output_rank = associations.shape
    if rank != output_rank:
        raise ValueError("fast-weight association must be square")
    decay_tensor = associations.new_full((), decay, dtype=torch.float64)
    moments = _constant_decay_scan(
        (1.0 - decay) * associations,
        torch.zeros(
            batch,
            rank,
            rank,
            device=associations.device,
            dtype=associations.dtype,
        ),
        decay_tensor,
    )
    steps = torch.arange(sequence, device=associations.device, dtype=torch.float32)
    normalizer = 1.0 - torch.pow(decay_tensor, steps)
    return torch.where(
        normalizer.view(1, sequence, 1, 1) > 0,
        moments
        / normalizer.clamp_min(torch.finfo(moments.dtype).tiny).view(
            1, sequence, 1, 1
        ),
        torch.zeros_like(moments),
    )


def predictive_error_fast_weight_logits(
    base_logits: Tensor,
    prediction_features: Tensor,
    token_ids: Tensor,
    basis: Tensor,
    codebook: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    if base_logits.shape[:2] != token_ids.shape:
        raise ValueError("logits and token ids differ")
    if prediction_features.shape[:2] != token_ids.shape:
        raise ValueError("features and token ids differ")
    if prediction_features.shape[-1] != basis.shape[0]:
        raise ValueError("feature and predictive basis widths differ")
    if codebook.shape != (base_logits.shape[-1], basis.shape[1]):
        raise ValueError("codebook shape differs from logits and basis")
    batch, sequence = token_ids.shape
    rank = basis.shape[1]
    keys = F.normalize(
        prediction_features.detach().float() @ basis.detach().float(),
        dim=-1,
        eps=1e-12,
    )
    codes = codebook.detach().float()
    probabilities = torch.softmax(base_logits.detach().float(), dim=-1)
    expected_codes = probabilities @ codes
    errors = torch.zeros(
        batch,
        sequence,
        rank,
        device=base_logits.device,
        dtype=torch.float32,
    )
    if sequence > 1:
        errors[:, 1:] = codes[token_ids[:, 1:]] - expected_codes[:, :-1]
    associations = torch.zeros(
        batch,
        sequence,
        rank,
        rank,
        device=base_logits.device,
        dtype=torch.float32,
    )
    if sequence > 1:
        associations[:, 1:] = (
            keys[:, :-1].unsqueeze(-1) * errors[:, 1:].unsqueeze(-2)
        )
    fast_matrices = _debiased_fast_matrices(
        associations, PREDICTIVE_FAST_DECAY
    )
    slow_matrices = _debiased_fast_matrices(
        associations, PREDICTIVE_SLOW_DECAY
    )
    fast_read = torch.einsum("btr,btrs->bts", keys, fast_matrices)
    slow_read = torch.einsum("btr,btrs->bts", keys, slow_matrices)
    predictive_error_read = 0.5 * (fast_read + slow_read)
    vocabulary_bias = predictive_error_read @ codes.T
    return base_logits.float() + vocabulary_bias, {
        "keys": keys,
        "expected_codes": expected_codes,
        "errors": errors,
        "fast_matrices": fast_matrices,
        "slow_matrices": slow_matrices,
        "predictive_error_read": predictive_error_read,
        "vocabulary_bias": vocabulary_bias,
    }


def scalar_predictive_error_fast_weight_logits(
    base_logits: Tensor,
    prediction_features: Tensor,
    token_ids: Tensor,
    basis: Tensor,
    codebook: Tensor,
) -> Tensor:
    batch, sequence, _ = base_logits.shape
    rank = basis.shape[1]
    keys = F.normalize(
        prediction_features.detach().float() @ basis.detach().float(),
        dim=-1,
        eps=1e-12,
    )
    codes = codebook.detach().float()
    probabilities = torch.softmax(base_logits.detach().float(), dim=-1)
    expected_codes = probabilities @ codes
    fast = torch.zeros(batch, rank, rank, device=base_logits.device)
    slow = torch.zeros_like(fast)
    fast_normalizer = torch.zeros(batch, 1, 1, device=base_logits.device)
    slow_normalizer = torch.zeros_like(fast_normalizer)
    rows = []
    for position in range(sequence):
        if position > 0:
            error = codes[token_ids[:, position]] - expected_codes[:, position - 1]
            association = keys[:, position - 1].unsqueeze(-1) * error.unsqueeze(-2)
            fast = (
                PREDICTIVE_FAST_DECAY * fast
                + (1.0 - PREDICTIVE_FAST_DECAY) * association
            )
            slow = (
                PREDICTIVE_SLOW_DECAY * slow
                + (1.0 - PREDICTIVE_SLOW_DECAY) * association
            )
            fast_normalizer = (
                PREDICTIVE_FAST_DECAY * fast_normalizer
                + (1.0 - PREDICTIVE_FAST_DECAY)
            )
            slow_normalizer = (
                PREDICTIVE_SLOW_DECAY * slow_normalizer
                + (1.0 - PREDICTIVE_SLOW_DECAY)
            )
        fast_read = torch.einsum(
            "br,brs->bs", keys[:, position], fast / fast_normalizer.clamp_min(1e-30)
        )
        slow_read = torch.einsum(
            "br,brs->bs", keys[:, position], slow / slow_normalizer.clamp_min(1e-30)
        )
        read = 0.5 * (fast_read + slow_read)
        rows.append(base_logits[:, position].float() + read @ codes.T)
    return torch.stack(rows, dim=1)
