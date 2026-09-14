from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from pfc.phase274_predictive_error_fast_weights import (
    PREDICTIVE_FAST_DECAY,
    PREDICTIVE_SLOW_DECAY,
)


def predictive_error_keys_and_values(
    base_logits: Tensor,
    prediction_features: Tensor,
    token_ids: Tensor,
    basis: Tensor,
    codebook: Tensor,
) -> tuple[Tensor, Tensor]:
    if base_logits.shape[:2] != token_ids.shape:
        raise ValueError("logits and token ids differ")
    if prediction_features.shape[:2] != token_ids.shape:
        raise ValueError("features and token ids differ")
    if prediction_features.shape[-1] != basis.shape[0]:
        raise ValueError("feature and predictive basis widths differ")
    if codebook.shape != (base_logits.shape[-1], basis.shape[1]):
        raise ValueError("codebook shape differs from logits and basis")
    keys = F.normalize(
        prediction_features.detach().float() @ basis.detach().float(),
        dim=-1,
        eps=1e-12,
    )
    codes = codebook.detach().float()
    expected_codes = torch.softmax(base_logits.detach().float(), dim=-1) @ codes
    errors = torch.zeros_like(keys)
    if token_ids.shape[1] > 1:
        errors[:, 1:] = codes[token_ids[:, 1:]] - expected_codes[:, :-1]
    return keys, errors


def predictive_delta_reads_reference(
    keys: Tensor,
    errors: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if keys.shape != errors.shape or keys.ndim != 3:
        raise ValueError("keys and errors must share [batch, sequence, rank]")
    batch, sequence, rank = keys.shape
    fast = torch.zeros(batch, rank, rank, device=keys.device, dtype=torch.float32)
    slow = torch.zeros_like(fast)
    reads = []
    for position in range(sequence):
        if position > 0:
            key = keys[:, position - 1]
            value = errors[:, position]
            decayed_fast = PREDICTIVE_FAST_DECAY * fast
            decayed_slow = PREDICTIVE_SLOW_DECAY * slow
            fast_residual = value - torch.einsum("br,brs->bs", key, decayed_fast)
            slow_residual = value - torch.einsum("br,brs->bs", key, decayed_slow)
            fast = decayed_fast + key.unsqueeze(-1) * fast_residual.unsqueeze(-2)
            slow = decayed_slow + key.unsqueeze(-1) * slow_residual.unsqueeze(-2)
        fast_read = torch.einsum("br,brs->bs", keys[:, position], fast)
        slow_read = torch.einsum("br,brs->bs", keys[:, position], slow)
        reads.append(0.5 * (fast_read + slow_read))
    return torch.stack(reads, dim=1), fast, slow


def predictive_delta_reads_cuda(keys: Tensor, errors: Tensor) -> Tensor:
    if not keys.is_cuda or not errors.is_cuda:
        raise ValueError("Phase275 Triton delta fast weights require CUDA")
    from pfc.phase275_triton_predictive_delta_fast_weights import (
        triton_predictive_delta_reads,
    )

    return triton_predictive_delta_reads(keys.contiguous(), errors.contiguous())


def predictive_delta_fast_weight_logits(
    base_logits: Tensor,
    prediction_features: Tensor,
    token_ids: Tensor,
    basis: Tensor,
    codebook: Tensor,
    *,
    force_reference: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    keys, errors = predictive_error_keys_and_values(
        base_logits,
        prediction_features,
        token_ids,
        basis,
        codebook,
    )
    if keys.is_cuda and not force_reference:
        read = predictive_delta_reads_cuda(keys, errors)
        final_fast = keys.new_empty(0)
        final_slow = keys.new_empty(0)
    else:
        read, final_fast, final_slow = predictive_delta_reads_reference(keys, errors)
    vocabulary_bias = read @ codebook.detach().float().T
    return base_logits.float() + vocabulary_bias, {
        "keys": keys,
        "errors": errors,
        "predictive_error_read": read,
        "vocabulary_bias": vocabulary_bias,
        "final_fast_matrix": final_fast,
        "final_slow_matrix": final_slow,
    }
