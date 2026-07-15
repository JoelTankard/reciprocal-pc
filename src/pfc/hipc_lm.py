from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pfc.llm_data import (
    VOCAB_SIZE,
    BOS_TOKEN_ID,
    TokenBatcher,
    build_text_tokenizer,
    download_climbmix_shards,
    list_climbmix_shards,
    train_nanochat_bpe_tokenizer,
)
from pfc.resource_guard import PeakMemoryTracker, ResourceGuard


@dataclass(frozen=True)
class HiPCLMConfig:
    vocab_size: int = VOCAB_SIZE
    bos_token_id: int = BOS_TOKEN_ID
    hidden_dim: int = 128
    feature_groups: int = 8
    top_k: int = 2
    concept_dim: int = 64
    recurrent_stride: int = 1
    low_cell_mode: str = "additive"
    low_cell_residual_weight: float = 0.5
    concept_memory_slots: int = 0
    concept_memory_top_k: int = 4
    concept_bias_weight: float = 0.0
    concept_adapter_weight: float = 0.0
    concept_update_rate: float = 0.05
    concept_novelty_threshold: float = 0.15
    phrase_stride: int = 0
    doc_stride: int = 0
    hierarchy_adapter_weight: float = 0.0
    high_vocab_weight: float = 0.0
    high_vocab_min_confidence: float = 0.0
    low_memory_slots: int = 0
    low_memory_top_k: int = 4
    low_memory_weight: float = 0.0
    low_state_memory_weight: float = 0.0
    low_feature_prior_weight: float = 0.0
    low_latent_prediction_weight: float = 0.0
    low_latent_contrastive_weight: float = 0.0
    low_latent_contrastive_temperature: float = 0.1
    low_memory_update_rate: float = 0.05
    low_memory_novelty_threshold: float = 0.15
    low_memory_read_temperature: float = 16.0
    low_memory_train_read_scale: bool = False
    low_memory_min_surprise: float = 0.0
    low_memory_cache_slots: int = 0
    sequence_memory_slots: int = 0
    sequence_memory_top_k: int = 4
    sequence_memory_weight: float = 0.0
    sequence_memory_read_temperature: float = 32.0
    sequence_memory_min_surprise: float = 0.0
    memory_decay: float = 1.0
    memory_decay_interval: int = 0
    memory_decay_min_count: float = 0.01


class SparseFeatureBank(nn.Module):
    def __init__(self, hidden_dim: int, feature_groups: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_groups = feature_groups
        self.norm_weight = nn.Parameter(torch.ones(feature_groups, hidden_dim))
        self.norm_bias = nn.Parameter(torch.zeros(feature_groups, hidden_dim))
        self.w1 = nn.Parameter(torch.empty(feature_groups, hidden_dim, hidden_dim * 2))
        self.b1 = nn.Parameter(torch.empty(feature_groups, hidden_dim * 2))
        self.w2 = nn.Parameter(torch.empty(feature_groups, hidden_dim * 2, hidden_dim))
        self.b2 = nn.Parameter(torch.empty(feature_groups, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for group_id in range(self.feature_groups):
            nn.init.kaiming_uniform_(self.w1[group_id], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w2[group_id], a=math.sqrt(5))
        fan_in1 = self.hidden_dim
        bound1 = 1 / math.sqrt(fan_in1)
        nn.init.uniform_(self.b1, -bound1, bound1)
        fan_in2 = self.hidden_dim * 2
        bound2 = 1 / math.sqrt(fan_in2)
        nn.init.uniform_(self.b2, -bound2, bound2)

    def forward(self, state: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
        mean = state.mean(dim=-1, keepdim=True)
        centered = state - mean
        inv_std = torch.rsqrt(centered.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        normed = centered * inv_std
        group_state = normed.unsqueeze(1) * self.norm_weight.unsqueeze(0) + self.norm_bias.unsqueeze(0)
        hidden = F.gelu(torch.einsum("bgi,gih->bgh", group_state, self.w1) + self.b1.unsqueeze(0))
        outputs = torch.einsum("bgh,ghd->bgd", hidden, self.w2) + self.b2.unsqueeze(0)
        gather_index = indices.unsqueeze(-1).expand(-1, -1, state.shape[-1])
        selected = outputs.gather(dim=1, index=gather_index)
        return (selected * weights.unsqueeze(-1)).sum(dim=1)


class HiPCLanguageModel(nn.Module):
    """No-attention high/low predictive-coding language model."""

    def __init__(self, config: HiPCLMConfig) -> None:
        super().__init__()
        if config.top_k > config.feature_groups:
            raise ValueError("top_k must be <= feature_groups")
        if config.recurrent_stride <= 0:
            raise ValueError("recurrent_stride must be positive")
        if config.low_cell_mode not in {"additive", "predictive"}:
            raise ValueError("low_cell_mode must be additive or predictive")
        if config.low_cell_residual_weight < 0:
            raise ValueError("low_cell_residual_weight must be nonnegative")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.low_router = nn.Linear(config.hidden_dim, config.feature_groups)
        self.low_features = SparseFeatureBank(config.hidden_dim, config.feature_groups)
        self.low_predictor = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.low_cell_norm = nn.LayerNorm(config.hidden_dim)
        self.low_token_norm = nn.LayerNorm(config.hidden_dim)
        self.low_cell_state = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.low_cell_token = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.low_cell_gate = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.low_feature_predictor = (
            nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            if config.low_feature_prior_weight > 0
            else None
        )
        self.low_latent_predictor = (
            nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            if config.low_latent_prediction_weight > 0 or config.low_latent_contrastive_weight > 0
            else None
        )
        self.high_encoder = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.concept_dim),
            nn.GELU(),
        )
        self.high_router = nn.Linear(config.concept_dim, config.feature_groups)
        self.high_features = SparseFeatureBank(config.concept_dim, config.feature_groups)
        self.high_predictor = nn.Linear(config.concept_dim, config.concept_dim)
        self.high_to_low = nn.Linear(config.concept_dim, config.hidden_dim)
        if config.phrase_stride < 0 or config.doc_stride < 0:
            raise ValueError("phrase_stride and doc_stride must be nonnegative")
        if config.hierarchy_adapter_weight < 0:
            raise ValueError("hierarchy_adapter_weight must be nonnegative")
        if config.high_vocab_weight < 0:
            raise ValueError("high_vocab_weight must be nonnegative")
        if not 0 <= config.high_vocab_min_confidence < 1:
            raise ValueError("high_vocab_min_confidence must be in [0, 1)")
        self.hierarchy_enabled = config.hierarchy_adapter_weight > 0 and (config.phrase_stride > 0 or config.doc_stride > 0)
        if self.hierarchy_enabled:
            self.phrase_encoder = nn.Sequential(
                nn.LayerNorm(config.concept_dim),
                nn.Linear(config.concept_dim, config.concept_dim),
                nn.GELU(),
            )
            self.doc_encoder = nn.Sequential(
                nn.LayerNorm(config.concept_dim),
                nn.Linear(config.concept_dim, config.concept_dim),
                nn.GELU(),
            )
            self.phrase_predictor = nn.Linear(config.concept_dim, config.concept_dim)
            self.doc_predictor = nn.Linear(config.concept_dim, config.concept_dim)
            self.phrase_to_low = nn.Linear(config.concept_dim, config.hidden_dim)
            self.doc_to_low = nn.Linear(config.concept_dim, config.hidden_dim)
        else:
            self.phrase_encoder = None
            self.doc_encoder = None
            self.phrase_predictor = None
            self.doc_predictor = None
            self.phrase_to_low = None
            self.doc_to_low = None
        self.high_decoder = (
            nn.Sequential(nn.LayerNorm(config.concept_dim), nn.Linear(config.concept_dim, config.vocab_size))
            if config.high_vocab_weight > 0
            else None
        )
        self.decoder = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.vocab_size))
        self.low_memory_read_log_scale = nn.Parameter(torch.tensor(math.log(config.low_memory_read_temperature)))
        if config.low_memory_slots < 0:
            raise ValueError("low_memory_slots must be nonnegative")
        if config.low_state_memory_weight < 0:
            raise ValueError("low_state_memory_weight must be nonnegative")
        if config.low_feature_prior_weight < 0:
            raise ValueError("low_feature_prior_weight must be nonnegative")
        if config.low_latent_prediction_weight < 0:
            raise ValueError("low_latent_prediction_weight must be nonnegative")
        if config.low_latent_contrastive_weight < 0:
            raise ValueError("low_latent_contrastive_weight must be nonnegative")
        if config.low_latent_contrastive_temperature <= 0:
            raise ValueError("low_latent_contrastive_temperature must be positive")
        if config.low_memory_top_k <= 0:
            raise ValueError("low_memory_top_k must be positive")
        if config.low_memory_read_temperature <= 0:
            raise ValueError("low_memory_read_temperature must be positive")
        if config.low_memory_min_surprise < 0:
            raise ValueError("low_memory_min_surprise must be nonnegative")
        if config.low_memory_cache_slots < 0:
            raise ValueError("low_memory_cache_slots must be nonnegative")
        if config.sequence_memory_slots < 0:
            raise ValueError("sequence_memory_slots must be nonnegative")
        if config.sequence_memory_top_k <= 0:
            raise ValueError("sequence_memory_top_k must be positive")
        if config.sequence_memory_weight < 0:
            raise ValueError("sequence_memory_weight must be nonnegative")
        if config.sequence_memory_read_temperature <= 0:
            raise ValueError("sequence_memory_read_temperature must be positive")
        if config.sequence_memory_min_surprise < 0:
            raise ValueError("sequence_memory_min_surprise must be nonnegative")
        low_memory_enabled = config.low_memory_slots > 0 and (
            config.low_memory_weight > 0 or config.low_state_memory_weight > 0
        )
        if low_memory_enabled:
            self.register_buffer("low_memory_keys", F.normalize(torch.randn(config.low_memory_slots, config.hidden_dim), dim=-1))
            self.register_buffer("low_memory_values", torch.zeros(config.low_memory_slots, config.vocab_size))
            state_values = (
                torch.zeros(config.low_memory_slots, config.hidden_dim)
                if config.low_state_memory_weight > 0
                else torch.empty(0, config.hidden_dim)
            )
            self.register_buffer("low_memory_state_values", state_values)
            self.register_buffer("low_memory_counts", torch.zeros(config.low_memory_slots))
            self.register_buffer("low_memory_scores", torch.zeros(config.low_memory_slots))
        else:
            self.register_buffer("low_memory_keys", torch.empty(0, config.hidden_dim))
            self.register_buffer("low_memory_values", torch.empty(0, config.vocab_size))
            self.register_buffer("low_memory_state_values", torch.empty(0, config.hidden_dim))
            self.register_buffer("low_memory_counts", torch.empty(0))
            self.register_buffer("low_memory_scores", torch.empty(0))
        self.register_buffer("low_memory_keys_cache", torch.empty(0, config.hidden_dim))
        self.register_buffer("low_memory_values_cache", torch.empty(0, config.vocab_size))
        self.register_buffer("low_memory_state_values_cache", torch.empty(0, config.hidden_dim))
        self.register_buffer("low_memory_counts_cache", torch.empty(0))
        if config.concept_memory_slots < 0:
            raise ValueError("concept_memory_slots must be nonnegative")
        if config.concept_memory_top_k <= 0:
            raise ValueError("concept_memory_top_k must be positive")
        if config.concept_memory_slots > 0 and (config.concept_bias_weight > 0 or config.concept_adapter_weight > 0):
            self.register_buffer("concept_keys", F.normalize(torch.randn(config.concept_memory_slots, config.concept_dim), dim=-1))
            self.register_buffer("concept_values", torch.zeros(config.concept_memory_slots, config.hidden_dim))
            self.register_buffer("concept_counts", torch.zeros(config.concept_memory_slots))
        else:
            self.register_buffer("concept_keys", torch.empty(0, config.concept_dim))
            self.register_buffer("concept_values", torch.empty(0, config.hidden_dim))
            self.register_buffer("concept_counts", torch.empty(0))
        self.register_buffer("concept_keys_cache", torch.empty(0, config.concept_dim))
        self.register_buffer("concept_values_cache", torch.empty(0, config.hidden_dim))

    def _route(self, state: Tensor, router: nn.Linear) -> tuple[Tensor, Tensor]:
        logits = router(state)
        values, indices = torch.topk(logits, k=self.config.top_k, dim=-1)
        return indices, F.softmax(values, dim=-1)

    def correction_step(self, low_state: Tensor, high_state: Tensor) -> tuple[Tensor, Tensor]:
        high_input = self.high_encoder(low_state.detach())
        hi, hw = self._route(high_state + high_input, self.high_router)
        high_proposed = high_state + self.high_features(high_state + high_input, hi, hw)
        high_predicted = self.high_predictor(high_state)
        high_state = high_proposed - 0.1 * (high_proposed - high_predicted.detach())
        concept_target = self.high_to_low(high_state.detach())
        li, lw = self._route(low_state + concept_target, self.low_router)
        low_proposed = low_state + self.low_features(low_state + concept_target, li, lw)
        low_predicted = self.low_predictor(low_state)
        low_state = low_proposed - 0.1 * (low_proposed - low_predicted.detach())
        return low_state, high_state

    def low_token_step(self, low_state: Tensor, token_state: Tensor) -> Tensor:
        if self.config.low_cell_mode == "predictive":
            state_norm = self.low_cell_norm(low_state)
            token_norm = self.low_token_norm(token_state)
            candidate = torch.tanh(self.low_cell_state(state_norm) + self.low_cell_token(token_norm))
            gate = torch.sigmoid(self.low_cell_gate(torch.cat([state_norm, token_norm], dim=-1)))
            mixed = low_state + gate * (token_state + candidate - low_state)
            return mixed + self.config.low_cell_residual_weight * torch.tanh(self.low_predictor(mixed))
        return low_state + token_state + 0.1 * torch.tanh(self.low_predictor(low_state))

    def forward_states_single_correction(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        base_state = self.embedding(torch.full((x.shape[0],), self.config.bos_token_id, dtype=torch.long, device=x.device))
        high_state = self.high_encoder(base_state)
        token_states = self.embedding(x)
        first_low_state = self.low_token_step(base_state, token_states[:, 0])
        corrected_low_state, high_state = self.correction_step(first_low_state, high_state)
        low_states = [corrected_low_state]
        low_state = corrected_low_state
        for token_state in token_states[:, 1:].unbind(dim=1):
            low_state = self.low_token_step(low_state.detach(), token_state)
            low_states.append(low_state)
        high_states = high_state.unsqueeze(1).repeat(1, x.shape[1], 1)
        return torch.stack(low_states, dim=1), high_states, self.high_to_low(high_state)

    def forward_states(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.config.recurrent_stride >= x.shape[1]:
            return self.forward_states_single_correction(x)
        low_state = self.embedding(torch.full((x.shape[0],), self.config.bos_token_id, dtype=torch.long, device=x.device))
        high_state = self.high_encoder(low_state)
        low_states = []
        high_states = []
        for step_idx, token in enumerate(x.unbind(dim=1)):
            low_state = low_state.detach()
            high_state = high_state.detach()
            token_state = self.embedding(token)
            low_state = self.low_token_step(low_state, token_state)
            if step_idx % self.config.recurrent_stride != 0:
                low_states.append(low_state)
                high_states.append(high_state)
                continue
            low_state, high_state = self.correction_step(low_state, high_state)
            low_states.append(low_state)
            high_states.append(high_state)
        return torch.stack(low_states, dim=1), torch.stack(high_states, dim=1), self.high_to_low(high_state)

    def forward_last_logits(self, x: Tensor) -> Tensor:
        if self.config.sequence_memory_weight > 0 and self.config.sequence_memory_slots > 0:
            return self.forward(x)[:, -1:, :]
        if self.config.recurrent_stride >= x.shape[1]:
            base_state = self.embedding(torch.full((x.shape[0],), self.config.bos_token_id, dtype=torch.long, device=x.device))
            high_state = self.high_encoder(base_state)
            token_states = self.embedding(x)
            low_state = self.low_token_step(base_state, token_states[:, 0])
            low_state, high_state = self.correction_step(low_state, high_state)
            if x.shape[1] > 1:
                for token_state in token_states[:, 1:].unbind(dim=1):
                    low_state = self.low_token_step(low_state.detach(), token_state)
        else:
            low_state = self.embedding(torch.full((x.shape[0],), self.config.bos_token_id, dtype=torch.long, device=x.device))
            high_state = self.high_encoder(low_state)
            for step_idx, token in enumerate(x.unbind(dim=1)):
                low_state = self.low_token_step(low_state.detach(), self.embedding(token))
                high_state = high_state.detach()
                if step_idx % self.config.recurrent_stride == 0:
                    low_state, high_state = self.correction_step(low_state, high_state)
        states = low_state.unsqueeze(1)
        high_states = high_state.unsqueeze(1)
        states = self.apply_hierarchy_adapter(states, high_states)
        states = self.apply_concept_adapter(states, high_states)
        return self.combine_logits(states, high_states, x[:, -1:])

    def forward(self, x: Tensor) -> Tensor:
        states, high_states, _ = self.forward_states(x)
        states = self.apply_hierarchy_adapter(states, high_states)
        states = self.apply_concept_adapter(states, high_states)
        return self.combine_logits(states, high_states, x)

    def combine_logits(self, states: Tensor, high_states: Tensor, x: Tensor) -> Tensor:
        adapted_states = self.apply_low_memory_state_adapter(states)
        adapted_states = self.apply_low_feature_prior(adapted_states)
        logits = self.decoder(adapted_states)
        logits = logits + self.low_prototype_logits(states)
        if self.config.high_vocab_weight > 0 and self.high_decoder is not None:
            gate = self.high_vocab_gate(high_states)
            logits = logits + self.config.high_vocab_weight * gate * self.high_decoder(high_states)
        logits = logits + self.sequence_memory_logits(states, x, logits)
        return logits

    def sequence_memory_logits(self, states: Tensor, x: Tensor, base_logits: Tensor) -> Tensor:
        if (
            self.config.sequence_memory_weight <= 0
            or self.config.sequence_memory_slots <= 0
            or states.shape[1] <= 1
            or x.shape[1] != states.shape[1]
        ):
            return torch.zeros_like(base_logits)
        norm_states = F.normalize(states, dim=-1)
        with torch.no_grad():
            probs = F.softmax(base_logits.detach(), dim=-1)
            successor_probs = probs[:, :-1].gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
            surprises = (1.0 - successor_probs).clamp_min(0.0)
        batch_size, seq_len = x.shape
        key_count = seq_len - 1
        max_slots = min(self.config.sequence_memory_slots, key_count)
        read_k = min(self.config.sequence_memory_top_k, max_slots)
        if read_k <= 0:
            return torch.zeros_like(base_logits)
        key_positions = torch.arange(key_count, device=x.device)
        query_positions = torch.arange(seq_len, device=x.device)
        causal = key_positions.unsqueeze(0) < query_positions.unsqueeze(1)
        valid_scores = surprises.unsqueeze(1).expand(-1, seq_len, -1)
        valid = causal.unsqueeze(0) & (valid_scores >= self.config.sequence_memory_min_surprise)
        slot_scores = valid_scores.masked_fill(~valid, -1.0)
        slot_scores, slot_indices = torch.topk(slot_scores, k=max_slots, dim=-1)
        slot_valid = slot_scores >= self.config.sequence_memory_min_surprise

        sims = torch.bmm(norm_states, norm_states[:, :-1].transpose(1, 2))
        slot_sims = sims.gather(2, slot_indices).masked_fill(~slot_valid, -1e9)
        read_values, read_indices = torch.topk(slot_sims, k=read_k, dim=-1)
        read_valid = read_values > -1e8
        read_values = read_values.masked_fill(~read_valid, -1e9)
        read_weights = F.softmax(read_values * self.config.sequence_memory_read_temperature, dim=-1).masked_fill(~read_valid, 0.0)
        confidence = read_values.masked_fill(~read_valid, 0.0).max(dim=-1).values.clamp_min(0.0).unsqueeze(-1)
        token_table = x[:, 1:].unsqueeze(1).expand(batch_size, seq_len, key_count)
        slot_tokens = token_table.gather(2, slot_indices)
        read_tokens = slot_tokens.gather(2, read_indices)
        output = torch.zeros_like(base_logits)
        contribution = self.config.sequence_memory_weight * confidence * read_weights
        output.scatter_add_(2, read_tokens, contribution.to(output.dtype))
        return output

    def apply_low_feature_prior(self, states: Tensor) -> Tensor:
        if self.config.low_feature_prior_weight <= 0 or self.low_feature_predictor is None:
            return states
        return states + self.config.low_feature_prior_weight * torch.tanh(self.low_feature_predictor(states))

    def low_memory_read_scale(self) -> Tensor | float:
        if self.config.low_memory_train_read_scale:
            return self.low_memory_read_log_scale.exp().clamp(1.0, 256.0)
        return self.config.low_memory_read_temperature

    def apply_low_memory_state_adapter(self, states: Tensor) -> Tensor:
        if (
            self.low_memory_state_values.numel() == 0
            or self.config.low_state_memory_weight <= 0
            or self.low_memory_counts.numel() == 0
        ):
            return states
        flat_states = F.normalize(states.reshape(-1, states.shape[-1]), dim=-1)
        if self.low_memory_keys_cache.numel() > 0 and self.low_memory_keys_cache.device == states.device:
            keys = self.low_memory_keys_cache
            values_table = self.low_memory_state_values_cache
            counts = self.low_memory_counts_cache
        else:
            keys = self.low_memory_keys.to(states.device)
            values_table = self.low_memory_state_values
            counts = self.low_memory_counts.to(states.device)
        active = counts > 0
        if not bool(active.any()):
            return states
        keys = keys[active]
        if values_table.device.type == "cpu":
            values_table = values_table[active.to("cpu")]
        else:
            values_table = values_table[active]
        sims = flat_states @ keys.t()
        top_values, top_indices = torch.topk(sims, k=min(self.config.low_memory_top_k, keys.shape[0]), dim=-1)
        read_scale = self.low_memory_read_scale()
        weights = F.softmax(top_values * read_scale, dim=-1)
        if values_table.device.type == "cpu":
            values = values_table.index_select(0, top_indices.reshape(-1).to("cpu")).to(states.device)
        else:
            values = values_table.index_select(0, top_indices.reshape(-1))
        values = values.view(top_indices.shape[0], top_indices.shape[1], self.config.hidden_dim)
        residual = (values * weights.unsqueeze(-1)).sum(dim=1)
        confidence = top_values.max(dim=-1).values.clamp_min(0.0).unsqueeze(-1)
        adapter = self.config.low_state_memory_weight * confidence * residual
        return states + adapter.view(*states.shape)

    def low_prototype_logits(self, states: Tensor) -> Tensor:
        if self.low_memory_values.numel() == 0 or self.config.low_memory_weight <= 0:
            return torch.zeros(*states.shape[:2], self.config.vocab_size, device=states.device, dtype=states.dtype)
        flat_states = F.normalize(states.reshape(-1, states.shape[-1]), dim=-1)
        if self.low_memory_keys_cache.numel() > 0 and self.low_memory_keys_cache.device == states.device:
            keys = self.low_memory_keys_cache
            values_table = self.low_memory_values_cache
            counts = self.low_memory_counts_cache
        else:
            keys = self.low_memory_keys.to(states.device)
            values_table = self.low_memory_values
            counts = self.low_memory_counts.to(states.device)
        active = counts > 0
        if not bool(active.any()):
            return torch.zeros(*states.shape[:2], self.config.vocab_size, device=states.device, dtype=states.dtype)
        keys = keys[active]
        if values_table.device.type == "cpu":
            values_table = values_table[active.to("cpu")]
        else:
            values_table = values_table[active]
        sims = flat_states @ keys.t()
        top_values, top_indices = torch.topk(sims, k=min(self.config.low_memory_top_k, keys.shape[0]), dim=-1)
        read_scale = self.low_memory_read_scale()
        weights = F.softmax(top_values * read_scale, dim=-1)
        if values_table.device.type == "cpu":
            values = values_table.index_select(0, top_indices.reshape(-1).to("cpu")).to(states.device)
        else:
            values = values_table.index_select(0, top_indices.reshape(-1))
        values = values.view(top_indices.shape[0], top_indices.shape[1], self.config.vocab_size)
        residual = (values * weights.unsqueeze(-1)).sum(dim=1)
        confidence = top_values.max(dim=-1).values.clamp_min(0.0).unsqueeze(-1)
        logits = self.config.low_memory_weight * confidence * residual
        return logits.view(*states.shape[:2], self.config.vocab_size)

    def high_vocab_gate(self, high_states: Tensor) -> Tensor:
        if self.config.high_vocab_min_confidence <= 0 or self.concept_keys.numel() == 0:
            return torch.ones(*high_states.shape[:2], 1, device=high_states.device, dtype=high_states.dtype)
        flat_states = F.normalize(high_states.reshape(-1, high_states.shape[-1]), dim=-1)
        if self.concept_keys_cache.numel() > 0 and self.concept_keys_cache.device == high_states.device:
            keys = self.concept_keys_cache
        else:
            keys = self.concept_keys.to(device=high_states.device, dtype=torch.float32)
        confidence = (flat_states @ keys.t()).max(dim=-1).values.clamp_min(0.0)
        threshold = self.config.high_vocab_min_confidence
        gate = ((confidence - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
        return gate.view(*high_states.shape[:2], 1).to(dtype=high_states.dtype)

    def pooled_high_states(self, high_states: Tensor, stride: int, encoder: Optional[nn.Module]) -> Tensor:
        if stride <= 0 or encoder is None:
            return high_states.new_zeros(high_states.shape)
        segments = []
        for start in range(0, high_states.shape[1], stride):
            chunk = high_states[:, start : start + stride]
            state = encoder(chunk.mean(dim=1))
            segments.append(state.unsqueeze(1).expand(-1, chunk.shape[1], -1))
        return torch.cat(segments, dim=1)

    def apply_hierarchy_adapter(self, states: Tensor, high_states: Tensor) -> Tensor:
        if not self.hierarchy_enabled:
            return states
        adapter = torch.zeros_like(states)
        if self.config.phrase_stride > 0 and self.phrase_to_low is not None:
            phrase_states = self.pooled_high_states(high_states, self.config.phrase_stride, self.phrase_encoder)
            adapter = adapter + self.phrase_to_low(phrase_states)
        if self.config.doc_stride > 0 and self.doc_to_low is not None:
            doc_states = self.pooled_high_states(high_states, self.config.doc_stride, self.doc_encoder)
            adapter = adapter + self.doc_to_low(doc_states)
        return states + self.config.hierarchy_adapter_weight * adapter

    def hierarchy_transition_loss(self, high_states: Tensor) -> Tensor:
        loss = high_states.new_tensor(0.0)
        if not self.hierarchy_enabled:
            return loss
        if self.config.phrase_stride > 0 and self.phrase_predictor is not None:
            phrase_states = self.pooled_high_states(high_states.detach(), self.config.phrase_stride, self.phrase_encoder)
            if phrase_states.shape[1] > 1:
                loss = loss + F.mse_loss(self.phrase_predictor(phrase_states[:, :-1]), phrase_states[:, 1:])
        if self.config.doc_stride > 0 and self.doc_predictor is not None:
            doc_states = self.pooled_high_states(high_states.detach(), self.config.doc_stride, self.doc_encoder)
            if doc_states.shape[1] > 1:
                loss = loss + F.mse_loss(self.doc_predictor(doc_states[:, :-1]), doc_states[:, 1:])
        return loss

    def get_device(self) -> torch.device:
        return next(self.parameters()).device

    def offload_external_memory(self) -> None:
        if self.low_memory_values.numel() > 0:
            self.low_memory_keys = self.low_memory_keys.to("cpu")
            self.low_memory_values = self.low_memory_values.to("cpu")
            self.low_memory_state_values = self.low_memory_state_values.to("cpu")
            self.low_memory_counts = self.low_memory_counts.to("cpu")
            self.low_memory_scores = self.low_memory_scores.to("cpu")
            self.low_memory_keys_cache = torch.empty(0, self.config.hidden_dim, device=self.get_device())
            self.low_memory_values_cache = torch.empty(0, self.config.vocab_size, device=self.get_device())
            self.low_memory_state_values_cache = torch.empty(0, self.config.hidden_dim, device=self.get_device())
            self.low_memory_counts_cache = torch.empty(0, device=self.get_device())
        if self.concept_values.numel() > 0:
            self.concept_keys = self.concept_keys.to("cpu")
            self.concept_values = self.concept_values.to("cpu")
            self.concept_counts = self.concept_counts.to("cpu")
            self.concept_keys_cache = torch.empty(0, self.config.concept_dim, device=self.get_device())
            self.concept_values_cache = torch.empty(0, self.config.hidden_dim, device=self.get_device())
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def refresh_low_memory_cache(self) -> None:
        if self.low_memory_values.numel() == 0:
            self.low_memory_keys_cache = torch.empty(0, self.config.hidden_dim, device=self.get_device())
            self.low_memory_values_cache = torch.empty(0, self.config.vocab_size, device=self.get_device())
            self.low_memory_state_values_cache = torch.empty(0, self.config.hidden_dim, device=self.get_device())
            self.low_memory_counts_cache = torch.empty(0, device=self.get_device())
            return
        device = self.get_device()
        active = self.low_memory_counts > 0
        if self.config.low_memory_cache_slots > 0 and bool(active.any()):
            active_scores = self.low_memory_scores.masked_fill(~active, -1.0)
            slots = torch.topk(active_scores, k=min(self.config.low_memory_cache_slots, int(active.sum().item()))).indices
            self.low_memory_keys_cache = self.low_memory_keys.index_select(0, slots).to(device=device, dtype=torch.float32)
            self.low_memory_values_cache = self.low_memory_values.index_select(0, slots).to(device=device, dtype=torch.float32)
            self.low_memory_state_values_cache = (
                self.low_memory_state_values.index_select(0, slots).to(device=device, dtype=torch.float32)
                if self.low_memory_state_values.numel() > 0
                else torch.empty(0, self.config.hidden_dim, device=device)
            )
            self.low_memory_counts_cache = self.low_memory_counts.index_select(0, slots).to(device=device, dtype=torch.float32)
            return
        self.low_memory_keys_cache = self.low_memory_keys.to(device=device, dtype=torch.float32)
        self.low_memory_values_cache = self.low_memory_values.to(device=device, dtype=torch.float32)
        self.low_memory_state_values_cache = (
            self.low_memory_state_values.to(device=device, dtype=torch.float32)
            if self.low_memory_state_values.numel() > 0
            else torch.empty(0, self.config.hidden_dim, device=device)
        )
        self.low_memory_counts_cache = self.low_memory_counts.to(device=device, dtype=torch.float32)

    def refresh_concept_adapter_cache(self) -> None:
        if self.concept_values.numel() == 0:
            self.concept_keys_cache = torch.empty(0, self.config.concept_dim, device=self.get_device())
            self.concept_values_cache = torch.empty(0, self.config.hidden_dim, device=self.get_device())
            return
        device = self.get_device()
        self.concept_keys_cache = self.concept_keys.to(device=device, dtype=torch.float32)
        self.concept_values_cache = self.concept_values.to(device=device, dtype=torch.float32)

    def apply_concept_adapter(self, states: Tensor, high_states: Tensor) -> Tensor:
        weight = self.config.concept_adapter_weight or self.config.concept_bias_weight
        if self.concept_values.numel() == 0 or weight <= 0:
            return states
        if high_states.shape[1] > 1 and torch.equal(high_states[:, :1], high_states[:, -1:]):
            adapter = self.concept_adapter_for_high_state(high_states[:, 0])
            return states + adapter.unsqueeze(1)
        flat_states = F.normalize(high_states.reshape(-1, high_states.shape[-1]), dim=-1)
        adapter = self.retrieve_concept_adapter(flat_states, high_states.device)
        return states + adapter.view(*states.shape)

    def concept_adapter_for_high_state(self, high_state: Tensor) -> Tensor:
        weight = self.config.concept_adapter_weight or self.config.concept_bias_weight
        if self.concept_values.numel() == 0 or weight <= 0:
            return torch.zeros(high_state.shape[0], self.config.hidden_dim, device=high_state.device)
        flat_states = F.normalize(high_state, dim=-1)
        return self.retrieve_concept_adapter(flat_states, high_state.device)

    def retrieve_concept_adapter(self, flat_states: Tensor, device: torch.device) -> Tensor:
        weight = self.config.concept_adapter_weight or self.config.concept_bias_weight
        if self.concept_keys_cache.numel() > 0 and self.concept_keys_cache.device == device:
            keys = self.concept_keys_cache
            values_table = self.concept_values_cache
        else:
            keys = self.concept_keys.to(device)
            values_table = self.concept_values
        sims = flat_states @ keys.t()
        top_values, top_indices = torch.topk(sims, k=min(self.config.concept_memory_top_k, keys.shape[0]), dim=-1)
        weights = F.softmax(top_values, dim=-1)
        if values_table.device.type == "cpu":
            values = values_table.index_select(0, top_indices.reshape(-1).to("cpu")).to(device)
        else:
            values = values_table.index_select(0, top_indices.reshape(-1))
        values = values.view(top_indices.shape[0], top_indices.shape[1], self.config.hidden_dim)
        mixed_values = (values * weights.unsqueeze(-1)).sum(dim=1)
        confidence = top_values.max(dim=-1).values.clamp_min(0.0).unsqueeze(-1)
        return weight * confidence * mixed_values

    def update_low_memory(
        self,
        low_states: Tensor,
        y: Tensor,
        logits: Tensor,
        state_corrections: Tensor | None = None,
    ) -> None:
        if self.low_memory_values.numel() == 0:
            return
        if self.low_memory_keys.device.type != "cpu":
            self.low_memory_keys = self.low_memory_keys.to("cpu")
            self.low_memory_values = self.low_memory_values.to("cpu")
            self.low_memory_state_values = self.low_memory_state_values.to("cpu")
            self.low_memory_counts = self.low_memory_counts.to("cpu")
            self.low_memory_scores = self.low_memory_scores.to("cpu")
        self.low_memory_keys_cache = torch.empty(0, self.config.hidden_dim, device=self.get_device())
        self.low_memory_values_cache = torch.empty(0, self.config.vocab_size, device=self.get_device())
        self.low_memory_state_values_cache = torch.empty(0, self.config.hidden_dim, device=self.get_device())
        self.low_memory_counts_cache = torch.empty(0, device=self.get_device())
        raw_states = low_states.detach().reshape(-1, low_states.shape[-1]).to("cpu")
        states = F.normalize(raw_states, dim=-1)
        targets = y.detach().reshape(-1).to("cpu")
        if state_corrections is None:
            target_embeddings = self.embedding(targets.to(self.get_device())).detach().to("cpu")
            hidden_residuals = target_embeddings - raw_states
        else:
            hidden_residuals = state_corrections.detach().reshape(-1, state_corrections.shape[-1]).to("cpu")
        probs = F.softmax(logits.detach().reshape(-1, logits.shape[-1]).to("cpu"), dim=-1)
        residuals = -probs
        residuals[torch.arange(targets.shape[0]), targets] += 1.0
        valid = targets != self.config.bos_token_id
        rate = self.config.low_memory_update_rate
        surprises = residuals.norm(dim=-1) * rate
        for state, residual, hidden_residual, surprise in zip(
            states[valid],
            residuals[valid],
            hidden_residuals[valid],
            surprises[valid],
        ):
            surprise_value = float(surprise.item())
            if surprise_value < self.config.low_memory_min_surprise:
                continue
            sims = self.low_memory_keys @ state
            best = int(sims.argmax().item())
            similar = float(sims[best].item()) >= self.config.low_memory_novelty_threshold and self.low_memory_counts[best] > 0
            if similar:
                if surprise_value <= float(self.low_memory_scores[best].item()):
                    continue
            else:
                empty = (self.low_memory_counts <= 0).nonzero(as_tuple=False)
                if empty.numel() > 0:
                    best = int(empty[0].item())
                else:
                    weakest = int(self.low_memory_scores.argmin().item())
                    if surprise_value <= float(self.low_memory_scores[weakest].item()):
                        continue
                    best = weakest
            self.low_memory_keys[best] = state
            self.low_memory_values[best] = residual
            if self.low_memory_state_values.numel() > 0:
                self.low_memory_state_values[best] = hidden_residual
            self.low_memory_counts[best] += 1.0
            self.low_memory_scores[best] = surprise_value

    def update_concept_memory(self, high_states: Tensor, low_states: Tensor) -> None:
        if self.concept_values.numel() == 0:
            return
        if self.concept_keys.device.type != "cpu":
            self.concept_keys = self.concept_keys.to("cpu")
            self.concept_values = self.concept_values.to("cpu")
            self.concept_counts = self.concept_counts.to("cpu")
        self.concept_keys_cache = torch.empty(0, self.config.concept_dim, device=self.get_device())
        self.concept_values_cache = torch.empty(0, self.config.hidden_dim, device=self.get_device())
        states = F.normalize(high_states.detach().mean(dim=1).to("cpu"), dim=-1)
        low_values = low_states.detach().mean(dim=1).to("cpu")
        for state, low_value in zip(states, low_values):
            sims = self.concept_keys @ state
            best = int(sims.argmax().item())
            if float(sims[best].item()) < self.config.concept_novelty_threshold:
                empty = (self.concept_counts <= 0).nonzero(as_tuple=False)
                best = int(empty[0].item()) if empty.numel() > 0 else int(self.concept_counts.argmin().item())
                self.concept_keys[best] = state
                self.concept_values[best].zero_()
                self.concept_counts[best] = 0.0
            rate = self.config.concept_update_rate
            self.concept_keys[best] = F.normalize((1.0 - rate) * self.concept_keys[best] + rate * state, dim=0)
            self.concept_values[best] = (1.0 - rate) * self.concept_values[best] + rate * low_value
            self.concept_counts[best] += 1.0

    def apply_memory_decay(self) -> None:
        decay = self.config.memory_decay
        if decay <= 0 or decay >= 1.0:
            return
        min_count = self.config.memory_decay_min_count
        if self.low_memory_counts.numel() > 0:
            if self.low_memory_counts.device.type != "cpu":
                self.low_memory_counts = self.low_memory_counts.to("cpu")
                self.low_memory_values = self.low_memory_values.to("cpu")
                self.low_memory_state_values = self.low_memory_state_values.to("cpu")
                self.low_memory_keys = self.low_memory_keys.to("cpu")
                self.low_memory_scores = self.low_memory_scores.to("cpu")
            self.low_memory_counts.mul_(decay)
            self.low_memory_scores.mul_(decay)
            stale = self.low_memory_counts < min_count
            if bool(stale.any()):
                self.low_memory_counts[stale] = 0.0
                self.low_memory_values[stale] = 0.0
                if self.low_memory_state_values.numel() > 0:
                    self.low_memory_state_values[stale] = 0.0
                self.low_memory_scores[stale] = 0.0
        if self.concept_counts.numel() > 0:
            if self.concept_counts.device.type != "cpu":
                self.concept_counts = self.concept_counts.to("cpu")
                self.concept_values = self.concept_values.to("cpu")
                self.concept_keys = self.concept_keys.to("cpu")
            self.concept_counts.mul_(decay)
            stale = self.concept_counts < min_count
            if bool(stale.any()):
                self.concept_counts[stale] = 0.0
                self.concept_values[stale] = 0.0


@dataclass(frozen=True)
class TinyGPTConfig:
    vocab_size: int = VOCAB_SIZE
    bos_token_id: int = BOS_TOKEN_ID
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    seq_len: int = 64


class TinyGPTLanguageModel(nn.Module):
    """Small causal Transformer baseline for same-data comparison."""

    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position = nn.Embedding(config.seq_len, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        positions = torch.arange(x.shape[1], device=x.device)
        state = self.embedding(x) + self.position(positions).unsqueeze(0)
        mask = torch.triu(torch.full((x.shape[1], x.shape[1]), float("-inf"), device=x.device), diagonal=1)
        state = self.blocks(state, mask=mask)
        return self.head(self.norm(state))

    def get_device(self) -> torch.device:
        return next(self.parameters()).device


def count_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def should_store_checkpoint_tensor_as_fp16(name: str, value: Tensor) -> bool:
    if not value.is_floating_point():
        return False
    if "count" in name or "score" in name or name.endswith("row_totals"):
        return False
    return True


def checkpoint_payload(
    model: nn.Module,
    model_meta: dict[str, int | str],
) -> dict[str, object]:
    state = dict(model.state_dict())
    state.pop("low_memory_keys_cache", None)
    state.pop("low_memory_values_cache", None)
    state.pop("low_memory_state_values_cache", None)
    state.pop("low_memory_counts_cache", None)
    state.pop("concept_keys_cache", None)
    state.pop("concept_values_cache", None)
    state = {
        name: value.to(torch.float16) if should_store_checkpoint_tensor_as_fp16(name, value) else value
        for name, value in state.items()
    }
    return {"config": model_meta, "checkpoint_float_dtype": "float16", "model": state}


def inflate_checkpoint_state(raw_state: dict[str, Tensor]) -> dict[str, Tensor]:
    return dict(raw_state)


def load_checkpoint_model(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, object]]:
    payload = torch.load(path, map_location="cpu")
    meta = dict(payload["config"])
    model_name = str(meta.pop("model"))
    if model_name == "hipc_lm":
        fields = HiPCLMConfig.__dataclass_fields__
        config = HiPCLMConfig(**{key: value for key, value in meta.items() if key in fields})
        extras = {key: value for key, value in meta.items() if key not in fields}
        model: nn.Module = HiPCLanguageModel(config).to(device)
    elif model_name == "tiny_gpt_baseline":
        fields = TinyGPTConfig.__dataclass_fields__
        config = TinyGPTConfig(**{key: value for key, value in meta.items() if key in fields})
        extras = {key: value for key, value in meta.items() if key not in fields}
        model = TinyGPTLanguageModel(config).to(device)
    else:
        raise ValueError(f"unknown checkpoint model: {model_name}")
    state = inflate_checkpoint_state(payload["model"])
    model.load_state_dict(state, strict=False)
    if isinstance(model, HiPCLanguageModel):
        model.offload_external_memory()
        model.refresh_low_memory_cache()
        model.refresh_concept_adapter_cache()
    return model, {"model": model_name, **asdict(config), **extras}


def apply_loaded_memory_overrides(model: nn.Module, model_meta: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    if not isinstance(model, HiPCLanguageModel):
        return model_meta
    applied: dict[str, object] = {}
    if args.load_low_memory_cache_slots is not None:
        applied["low_memory_cache_slots"] = int(args.load_low_memory_cache_slots)
    sequence_memory_slots = int(getattr(args, "sequence_memory_slots", model.config.sequence_memory_slots))
    sequence_memory_weight = float(getattr(args, "sequence_memory_weight", model.config.sequence_memory_weight))
    sequence_memory_top_k = int(getattr(args, "sequence_memory_top_k", model.config.sequence_memory_top_k))
    sequence_memory_read_temperature = float(
        getattr(args, "sequence_memory_read_temperature", model.config.sequence_memory_read_temperature)
    )
    sequence_memory_min_surprise = float(getattr(args, "sequence_memory_min_surprise", model.config.sequence_memory_min_surprise))
    if sequence_memory_slots > 0:
        applied["sequence_memory_slots"] = sequence_memory_slots
    if sequence_memory_weight > 0:
        applied["sequence_memory_weight"] = sequence_memory_weight
    if sequence_memory_top_k != model.config.sequence_memory_top_k:
        applied["sequence_memory_top_k"] = sequence_memory_top_k
    if sequence_memory_read_temperature != model.config.sequence_memory_read_temperature:
        applied["sequence_memory_read_temperature"] = sequence_memory_read_temperature
    if sequence_memory_min_surprise != model.config.sequence_memory_min_surprise:
        applied["sequence_memory_min_surprise"] = sequence_memory_min_surprise
    if not applied:
        return model_meta
    model.config = replace(model.config, **applied)
    model.refresh_low_memory_cache()
    return {**model_meta, **applied}


def apply_local_update(
    params: list[nn.Parameter],
    loss: Tensor,
    *,
    lr: float,
    weight_decay: float,
    grad_clip: float,
) -> None:
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    pairs = [(param, grad) for param, grad in zip(params, grads) if grad is not None]
    if not pairs:
        return
    scale = 1.0
    if grad_clip > 0:
        norm = torch.sqrt(sum(grad.detach().pow(2).sum() for _, grad in pairs))
        scale = min(1.0, grad_clip / float(norm.item() + 1e-12))
    with torch.no_grad():
        for param, grad in pairs:
            if weight_decay:
                param.mul_(1.0 - lr * weight_decay)
            param.add_(grad, alpha=-lr * scale)


def local_train_step(
    model: HiPCLanguageModel,
    x: Tensor,
    y: Tensor,
    *,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    decoder_lr_scale: float = 1.0,
    state_lr_scale: float = 1.0,
    train_logit_temperature: float = 1.0,
) -> float:
    if train_logit_temperature <= 0:
        raise ValueError("train_logit_temperature must be positive")
    states, high_states, concept_target = model.forward_states(x)
    adapted_states = model.apply_hierarchy_adapter(states, high_states)
    adapted_states = model.apply_concept_adapter(adapted_states, high_states)
    memory_states = adapted_states.detach()
    if model.config.low_state_memory_weight > 0:
        memory_states = memory_states.requires_grad_(True)
    logits = model.combine_logits(memory_states, high_states.detach(), x)
    train_logits = logits / train_logit_temperature
    loss = F.cross_entropy(train_logits.reshape(-1, model.config.vocab_size), y.reshape(-1))
    state_corrections = None
    if memory_states.requires_grad:
        state_grad = torch.autograd.grad(loss, memory_states, retain_graph=True)[0]
        state_corrections = -state_grad.detach()
    decoder_params = list(model.decoder.parameters())
    if model.high_decoder is not None:
        decoder_params.extend(model.high_decoder.parameters())
    apply_local_update(decoder_params, loss, lr=lr * decoder_lr_scale, weight_decay=weight_decay, grad_clip=grad_clip)
    model.update_low_memory(states, y, logits, state_corrections=state_corrections)
    model.update_concept_memory(high_states, states)

    adapted_states = model.apply_hierarchy_adapter(states, high_states)
    adapted_states = model.apply_concept_adapter(adapted_states, high_states)
    local_logits = model.combine_logits(adapted_states, high_states, x)
    train_local_logits = local_logits / train_logit_temperature
    local_loss = F.cross_entropy(train_local_logits.reshape(-1, model.config.vocab_size), y.reshape(-1))
    transition_loss = F.mse_loss(model.low_predictor(states[:, :-1].detach()), states[:, 1:].detach())
    if high_states.shape[1] > 1:
        transition_loss = transition_loss + F.mse_loss(
            model.high_predictor(high_states[:, :-1].detach()),
            high_states[:, 1:].detach(),
        )
    feature_loss = states.new_tensor(0.0)
    if model.config.low_feature_prior_weight > 0 and model.low_feature_predictor is not None:
        target_features = model.embedding(y).detach()
        feature_loss = F.mse_loss(model.low_feature_predictor(states.detach()), target_features)
    latent_loss = states.new_tensor(0.0)
    if model.config.low_latent_prediction_weight > 0 and model.low_latent_predictor is not None:
        target_features = model.embedding(y).detach()
        latent_loss = F.mse_loss(model.low_latent_predictor(states), target_features)
    contrastive_loss = states.new_tensor(0.0)
    if model.config.low_latent_contrastive_weight > 0 and model.low_latent_predictor is not None:
        predicted = F.normalize(model.low_latent_predictor(states).reshape(-1, model.config.hidden_dim), dim=-1)
        target_features = F.normalize(model.embedding(y).detach().reshape(-1, model.config.hidden_dim), dim=-1)
        contrastive_logits = predicted @ target_features.t()
        contrastive_logits = contrastive_logits / model.config.low_latent_contrastive_temperature
        contrastive_targets = torch.arange(contrastive_logits.shape[0], device=contrastive_logits.device)
        contrastive_loss = F.cross_entropy(contrastive_logits, contrastive_targets)
    concept_loss = F.mse_loss(states[:, -1], concept_target.detach())
    local_loss = (
        local_loss
        + 0.1 * transition_loss
        + 0.05 * concept_loss
        + 0.1 * feature_loss
        + model.config.low_latent_prediction_weight * latent_loss
        + model.config.low_latent_contrastive_weight * contrastive_loss
        + 0.05 * model.hierarchy_transition_loss(high_states)
    )
    local_params = [
        *model.embedding.parameters(),
        *model.low_router.parameters(),
        *model.low_features.parameters(),
        *model.low_predictor.parameters(),
        *model.low_cell_norm.parameters(),
        *model.low_token_norm.parameters(),
        *model.low_cell_state.parameters(),
        *model.low_cell_token.parameters(),
        *model.low_cell_gate.parameters(),
        *model.high_encoder.parameters(),
        *model.high_router.parameters(),
        *model.high_features.parameters(),
        *model.high_predictor.parameters(),
        *model.high_to_low.parameters(),
    ]
    if model.config.low_memory_train_read_scale:
        local_params.append(model.low_memory_read_log_scale)
    if model.low_feature_predictor is not None:
        local_params.extend(model.low_feature_predictor.parameters())
    if model.low_latent_predictor is not None:
        local_params.extend(model.low_latent_predictor.parameters())
    if model.hierarchy_enabled:
        hierarchy_modules = [
            model.phrase_encoder,
            model.doc_encoder,
            model.phrase_predictor,
            model.doc_predictor,
            model.phrase_to_low,
            model.doc_to_low,
        ]
        for module in hierarchy_modules:
            if module is not None:
                local_params.extend(module.parameters())
    if model.high_decoder is not None:
        local_params.extend(model.high_decoder.parameters())
    apply_local_update(local_params, local_loss, lr=lr * state_lr_scale, weight_decay=weight_decay, grad_clip=grad_clip)
    return float(loss.detach().item())


def backprop_train_step(
    model: TinyGPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    y: Tensor,
    *,
    grad_clip: float,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), y.reshape(-1))
    loss.backward()
    if grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach().item())


@torch.no_grad()
def evaluate(model: nn.Module, batcher: TokenBatcher, *, steps: int, logit_temperature: float = 1.0) -> dict[str, float]:
    if logit_temperature <= 0:
        raise ValueError("logit_temperature must be positive")
    total = 0
    correct = 0
    total_nats = 0.0
    total_bytes = 0.0
    top2_correct = 0
    top5_correct = 0
    top10_correct = 0
    rank_sum = 0.0
    target_margin_sum = 0.0
    target_prob_sum = 0.0
    top_prob_sum = 0.0
    entropy_sum = 0.0
    top2_gap_sum = 0.0
    byte_lengths = batcher.tokenizer.token_byte_lengths(model.get_device())  # type: ignore[attr-defined]
    for _ in range(steps):
        x, y = batcher.next()
        logits = model(x)
        scaled_logits = logits / logit_temperature
        loss = F.cross_entropy(scaled_logits.reshape(-1, scaled_logits.shape[-1]), y.reshape(-1), reduction="none")
        pred = logits.argmax(dim=-1)
        mask = y != batcher.tokenizer.get_bos_token_id()
        correct += int((pred[mask] == y[mask]).sum().item())
        masked_total = int(mask.sum().item())
        total += masked_total
        if masked_total > 0:
            flat_logits = logits[mask]
            flat_scaled_logits = scaled_logits[mask]
            flat_targets = y[mask]
            target_logits = flat_logits.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
            top_k = min(10, flat_logits.shape[-1])
            top_values, top_indices = flat_logits.topk(k=top_k, dim=-1)
            top2_correct += int((top_indices[:, : min(2, top_k)] == flat_targets.unsqueeze(1)).any(dim=1).sum().item())
            top5_correct += int((top_indices[:, : min(5, top_k)] == flat_targets.unsqueeze(1)).any(dim=1).sum().item())
            top10_correct += int((top_indices == flat_targets.unsqueeze(1)).any(dim=1).sum().item())
            rank_sum += float((flat_logits > target_logits.unsqueeze(1)).sum(dim=1).add(1).float().sum().item())
            if flat_logits.shape[-1] > 1:
                target_mask = F.one_hot(flat_targets, num_classes=flat_logits.shape[-1]).bool()
                best_wrong = flat_logits.masked_fill(target_mask, -torch.inf).amax(dim=1)
                target_margin_sum += float((target_logits - best_wrong).sum().item())
                top2_gap_sum += float((top_values[:, 0] - top_values[:, 1]).sum().item())
            log_probs = F.log_softmax(flat_scaled_logits, dim=-1)
            probs = log_probs.exp()
            target_prob_sum += float(probs.gather(1, flat_targets.unsqueeze(1)).sum().item())
            top_prob_sum += float(probs.amax(dim=1).sum().item())
            entropy_sum += float((-(probs * log_probs).sum(dim=1)).sum().item())
        lengths = byte_lengths[y.reshape(-1)]
        total_nats += float((loss * (lengths > 0)).sum().item())
        total_bytes += float(lengths.sum().item())
    return {
        "accuracy": correct / max(total, 1),
        "val_bpb": total_nats / max(math.log(2) * total_bytes, 1e-12),
        "top2_accuracy": top2_correct / max(total, 1),
        "top5_accuracy": top5_correct / max(total, 1),
        "top10_accuracy": top10_correct / max(total, 1),
        "mean_correct_rank": rank_sum / max(total, 1),
        "mean_target_logit_margin": target_margin_sum / max(total, 1),
        "mean_target_prob": target_prob_sum / max(total, 1),
        "mean_top_prob": top_prob_sum / max(total, 1),
        "mean_entropy_nats": entropy_sum / max(total, 1),
        "mean_top2_logit_gap": top2_gap_sum / max(total, 1),
    }


def infer_speed(model: nn.Module, x: Tensor, *, steps: int) -> float:
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            _ = model(x)
    return (time.perf_counter() - start) * 1000.0 / max(steps, 1)


def next_token_logits(model: nn.Module, x: Tensor) -> Tensor:
    forward_last = getattr(model, "forward_last_logits", None)
    if forward_last is not None:
        return forward_last(x)
    return model(x)[:, -1:, :]


def infer_next_token_speed(model: nn.Module, x: Tensor, *, steps: int) -> float:
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            _ = next_token_logits(model, x)
    return (time.perf_counter() - start) * 1000.0 / max(steps, 1)


@torch.no_grad()
def generate_text(
    model: nn.Module,
    tokenizer: object,
    prompt: str,
    *,
    max_new_tokens: int,
    seq_len: int,
    temperature: float,
    top_k: int,
) -> str:
    encode = getattr(tokenizer, "encode")
    decode = getattr(tokenizer, "decode")
    ids = list(encode(prompt))
    device = model.get_device()  # type: ignore[attr-defined]
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-seq_len:]], dtype=torch.long, device=device)
        logits = next_token_logits(model, x)[:, -1, :]
        if temperature <= 0:
            next_id = int(logits.argmax(dim=-1).item())
        else:
            logits = logits / temperature
            if top_k > 0:
                values, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
                probs = F.softmax(values, dim=-1)
                next_id = int(indices.gather(1, torch.multinomial(probs, num_samples=1)).item())
            else:
                probs = F.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
        ids.append(next_id)
    return str(decode(ids))


@torch.no_grad()
def score_text(model: nn.Module, tokenizer: object, text: str, *, seq_len: int, logit_temperature: float = 1.0) -> dict[str, float]:
    if logit_temperature <= 0:
        raise ValueError("logit_temperature must be positive")
    encode = getattr(tokenizer, "encode")
    token_byte_lengths = getattr(tokenizer, "token_byte_lengths")
    ids = list(encode(text))
    if len(ids) < 2:
        return {"score_tokens": 0.0, "score_accuracy": 0.0, "score_nll": 0.0, "score_bpb": 0.0}
    device = model.get_device()  # type: ignore[attr-defined]
    x = torch.tensor([ids[:-1][-seq_len:]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:][-seq_len:]], dtype=torch.long, device=device)
    logits = model(x)
    scaled_logits = logits / logit_temperature
    losses = F.cross_entropy(scaled_logits.reshape(-1, scaled_logits.shape[-1]), y.reshape(-1), reduction="none")
    pred = logits.argmax(dim=-1)
    mask = y != getattr(tokenizer, "get_bos_token_id")()
    byte_lengths = token_byte_lengths(device)[y.reshape(-1)]
    token_count = int(mask.sum().item())
    correct = int((pred[mask] == y[mask]).sum().item())
    total_nats = float((losses * (byte_lengths > 0)).sum().item())
    total_bytes = float(byte_lengths.sum().item())
    return {
        "score_tokens": float(token_count),
        "score_accuracy": correct / max(token_count, 1),
        "score_nll": float(losses.mean().item()),
        "score_bpb": total_nats / max(math.log(2) * total_bytes, 1e-12),
    }


def choose_device(raw: str) -> torch.device:
    if raw:
        return torch.device(raw)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(args: argparse.Namespace, device: torch.device, *, vocab_size: int, bos_token_id: int) -> tuple[nn.Module, dict[str, int | str]]:
    if args.model == "hipc":
        config = HiPCLMConfig(
            vocab_size=vocab_size,
            hidden_dim=args.hidden_dim,
            feature_groups=args.feature_groups,
            top_k=args.top_k,
            concept_dim=args.concept_dim,
            bos_token_id=bos_token_id,
            recurrent_stride=args.recurrent_stride,
            low_cell_mode=args.low_cell_mode,
            low_cell_residual_weight=args.low_cell_residual_weight,
            concept_memory_slots=args.concept_memory_slots,
            concept_memory_top_k=args.concept_memory_top_k,
            concept_bias_weight=args.concept_bias_weight,
            concept_adapter_weight=args.concept_adapter_weight,
            concept_update_rate=args.concept_update_rate,
            concept_novelty_threshold=args.concept_novelty_threshold,
            phrase_stride=args.phrase_stride,
            doc_stride=args.doc_stride,
            hierarchy_adapter_weight=args.hierarchy_adapter_weight,
            high_vocab_weight=args.high_vocab_weight,
            high_vocab_min_confidence=args.high_vocab_min_confidence,
            low_memory_slots=args.low_memory_slots,
            low_memory_top_k=args.low_memory_top_k,
            low_memory_weight=args.low_memory_weight,
            low_state_memory_weight=args.low_state_memory_weight,
            low_feature_prior_weight=args.low_feature_prior_weight,
            low_latent_prediction_weight=args.low_latent_prediction_weight,
            low_latent_contrastive_weight=args.low_latent_contrastive_weight,
            low_latent_contrastive_temperature=args.low_latent_contrastive_temperature,
            low_memory_update_rate=args.low_memory_update_rate,
            low_memory_novelty_threshold=args.low_memory_novelty_threshold,
            low_memory_read_temperature=args.low_memory_read_temperature,
            low_memory_train_read_scale=args.low_memory_train_read_scale,
            low_memory_min_surprise=args.low_memory_min_surprise,
            low_memory_cache_slots=args.low_memory_cache_slots,
            sequence_memory_slots=args.sequence_memory_slots,
            sequence_memory_top_k=args.sequence_memory_top_k,
            sequence_memory_weight=args.sequence_memory_weight,
            sequence_memory_read_temperature=args.sequence_memory_read_temperature,
            sequence_memory_min_surprise=args.sequence_memory_min_surprise,
            memory_decay=args.memory_decay,
            memory_decay_interval=args.memory_decay_interval,
            memory_decay_min_count=args.memory_decay_min_count,
        )
        model = HiPCLanguageModel(config).to(device)
        model.offload_external_memory()
        return model, {"model": "hipc_lm", **asdict(config)}
    if args.model == "tiny-gpt":
        config = TinyGPTConfig(
            vocab_size=vocab_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.gpt_layers,
            num_heads=args.gpt_heads,
            seq_len=args.seq_len,
            bos_token_id=bos_token_id,
        )
        return TinyGPTLanguageModel(config).to(device), {"model": "tiny_gpt_baseline", **asdict(config)}
    raise ValueError(f"unknown model: {args.model}")


def train(args: argparse.Namespace) -> dict[str, float | str]:
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    data_dir = Path(args.data_dir)
    if args.download_shards > 0:
        download_climbmix_shards(data_dir, train_shards=args.download_shards, workers=args.download_workers)
    train_paths, val_path = list_climbmix_shards(data_dir)
    train_paths = train_paths[: args.train_shards]
    if args.train_tokenizer:
        train_nanochat_bpe_tokenizer(
            train_paths,
            args.tokenizer_dir,
            vocab_size=args.tokenizer_vocab_size,
            max_chars=args.tokenizer_max_chars,
            doc_cap=args.tokenizer_doc_cap,
            mix_shards=args.mix_train_shards,
        )
    tokenizer = build_text_tokenizer(args.tokenizer, args.tokenizer_dir)
    guard = ResourceGuard(args.max_system_memory_percent, args.max_system_load_percent, args.max_gpu_memory_percent)
    guard.check(label="hipc-lm-start")

    if args.load_checkpoint:
        model, model_meta = load_checkpoint_model(args.load_checkpoint, device)
        model_meta = apply_loaded_memory_overrides(model, model_meta, args)
    else:
        model, model_meta = build_model(
            args,
            device,
            vocab_size=tokenizer.get_vocab_size(),
            bos_token_id=tokenizer.get_bos_token_id(),
        )
    train_batcher = TokenBatcher(train_paths, args.batch_size, args.seq_len, device, tokenizer, shuffle_docs=args.mix_train_shards)
    val_batcher = TokenBatcher([val_path], args.batch_size, args.seq_len, device, tokenizer)
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.model == "tiny-gpt"
        else None
    )
    train_peak_cpu = train_peak_gpu = 0.0
    start = time.perf_counter()
    last_loss = 0.0
    with PeakMemoryTracker() as tracker:
        for step in range(1, args.steps + 1):
            x, y = train_batcher.next()
            if isinstance(model, HiPCLanguageModel):
                last_loss = local_train_step(
                    model,
                    x,
                    y,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    grad_clip=args.grad_clip,
                    decoder_lr_scale=args.decoder_lr_scale,
                    state_lr_scale=args.state_lr_scale,
                    train_logit_temperature=args.train_logit_temperature,
                )
                if args.memory_decay_interval > 0 and step % args.memory_decay_interval == 0:
                    model.apply_memory_decay()
            else:
                if optimizer is None:
                    raise RuntimeError("baseline optimizer missing")
                last_loss = backprop_train_step(model, optimizer, x, y, grad_clip=args.grad_clip)  # type: ignore[arg-type]
            if args.log_interval > 0 and not args.quiet and step % args.log_interval == 0:
                elapsed_s = time.perf_counter() - start
                ms_per_batch = elapsed_s * 1000.0 / step
                progress_metrics = (
                    evaluate(model, val_batcher, steps=args.progress_eval_steps, logit_temperature=args.eval_logit_temperature)
                    if args.progress_eval_steps > 0
                    else {}
                )
                print(
                    json.dumps(
                        {
                            "event": "train_progress",
                            "step": step,
                            "steps": args.steps,
                            "train_loss": last_loss,
                            "train_ms_per_batch": ms_per_batch,
                            **progress_metrics,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if args.resource_check_interval > 0 and step % args.resource_check_interval == 0:
                guard.check(label=f"hipc-lm-step-{step}")
        train_peak_cpu = tracker.peak_cpu_memory_gb
        train_peak_gpu = tracker.peak_gpu_memory_gb
    train_s = time.perf_counter() - start
    if isinstance(model, HiPCLanguageModel):
        model.offload_external_memory()
        model.refresh_low_memory_cache()
        model.refresh_concept_adapter_cache()
    eval_metrics = evaluate(model, val_batcher, steps=args.eval_steps, logit_temperature=args.eval_logit_temperature)
    sample_x, _ = val_batcher.next()
    with PeakMemoryTracker() as tracker:
        infer_ms = infer_speed(model, sample_x, steps=args.infer_steps)
        infer_peak_cpu = tracker.peak_cpu_memory_gb
        infer_peak_gpu = tracker.peak_gpu_memory_gb
    with PeakMemoryTracker() as tracker:
        next_infer_ms = infer_next_token_speed(model, sample_x, steps=args.infer_steps)
        next_infer_peak_cpu = tracker.peak_cpu_memory_gb
        next_infer_peak_gpu = tracker.peak_gpu_memory_gb
    sample_text = ""
    if args.sample_prompt:
        sample_text = generate_text(
            model,
            tokenizer,
            args.sample_prompt,
            max_new_tokens=args.sample_tokens,
            seq_len=args.seq_len,
            temperature=args.sample_temperature,
            top_k=args.sample_top_k,
        )
    prompt_scores = (
        score_text(model, tokenizer, args.score_text, seq_len=args.seq_len, logit_temperature=args.eval_logit_temperature)
        if args.score_text
        else {}
    )
    prior_steps = float(args.prior_steps) if args.prior_steps is not None else float(model_meta.get("trained_steps", 0.0))
    total_steps = prior_steps + float(args.steps)
    model_meta = {**model_meta, "trained_steps": total_steps}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{model_meta['model']}_{args.run_tag}.pt"
    torch.save(
        checkpoint_payload(model, model_meta),
        model_path,
    )
    is_hipc = isinstance(model, HiPCLanguageModel)
    result: dict[str, float | str] = {
        "model": str(model_meta["model"]),
        "baseline_data": "karpathy/climbmix-400b-shuffle",
        "tokenizer": tokenizer.name,
        "vocab_size": float(tokenizer.get_vocab_size()),
        "device": str(device),
        "global_backprops": 0.0 if is_hipc else float(args.steps),
        "training_rule": "local-pc" if is_hipc else "backprop",
        "params": float(count_params(model)),
        "model_file_size_mb": model_path.stat().st_size / (1024 * 1024),
        "steps": total_steps,
        "run_steps": float(args.steps),
        "prior_steps": prior_steps,
        "lr": float(args.lr),
        "decoder_lr_scale": float(args.decoder_lr_scale),
        "state_lr_scale": float(args.state_lr_scale),
        "train_logit_temperature": float(args.train_logit_temperature),
        "seq_len": float(args.seq_len),
        "seed": float(args.seed),
        "eval_logit_temperature": float(args.eval_logit_temperature),
        "recurrent_stride": float(model_meta.get("recurrent_stride", args.recurrent_stride)),
        "low_cell_mode": str(model_meta.get("low_cell_mode", args.low_cell_mode)),
        "low_cell_residual_weight": float(model_meta.get("low_cell_residual_weight", args.low_cell_residual_weight)),
        "concept_memory_slots": float(model_meta.get("concept_memory_slots", args.concept_memory_slots)),
        "concept_memory_top_k": float(model_meta.get("concept_memory_top_k", args.concept_memory_top_k)),
        "concept_bias_weight": float(model_meta.get("concept_bias_weight", args.concept_bias_weight)),
        "concept_adapter_weight": float(model_meta.get("concept_adapter_weight", args.concept_adapter_weight)),
        "concept_update_rate": float(model_meta.get("concept_update_rate", args.concept_update_rate)),
        "concept_novelty_threshold": float(model_meta.get("concept_novelty_threshold", args.concept_novelty_threshold)),
        "phrase_stride": float(model_meta.get("phrase_stride", args.phrase_stride)),
        "doc_stride": float(model_meta.get("doc_stride", args.doc_stride)),
        "hierarchy_adapter_weight": float(model_meta.get("hierarchy_adapter_weight", args.hierarchy_adapter_weight)),
        "high_vocab_weight": float(model_meta.get("high_vocab_weight", args.high_vocab_weight)),
        "high_vocab_min_confidence": float(model_meta.get("high_vocab_min_confidence", args.high_vocab_min_confidence)),
        "low_memory_slots": float(model_meta.get("low_memory_slots", args.low_memory_slots)),
        "low_memory_top_k": float(model_meta.get("low_memory_top_k", args.low_memory_top_k)),
        "low_memory_weight": float(model_meta.get("low_memory_weight", args.low_memory_weight)),
        "low_state_memory_weight": float(model_meta.get("low_state_memory_weight", args.low_state_memory_weight)),
        "low_feature_prior_weight": float(model_meta.get("low_feature_prior_weight", args.low_feature_prior_weight)),
        "low_latent_prediction_weight": float(model_meta.get("low_latent_prediction_weight", args.low_latent_prediction_weight)),
        "low_latent_contrastive_weight": float(model_meta.get("low_latent_contrastive_weight", args.low_latent_contrastive_weight)),
        "low_latent_contrastive_temperature": float(
            model_meta.get("low_latent_contrastive_temperature", args.low_latent_contrastive_temperature)
        ),
        "low_memory_update_rate": float(model_meta.get("low_memory_update_rate", args.low_memory_update_rate)),
        "low_memory_novelty_threshold": float(model_meta.get("low_memory_novelty_threshold", args.low_memory_novelty_threshold)),
        "low_memory_read_temperature": float(model_meta.get("low_memory_read_temperature", args.low_memory_read_temperature)),
        "low_memory_train_read_scale": float(model_meta.get("low_memory_train_read_scale", args.low_memory_train_read_scale)),
        "low_memory_min_surprise": float(model_meta.get("low_memory_min_surprise", args.low_memory_min_surprise)),
        "low_memory_cache_slots": float(model_meta.get("low_memory_cache_slots", args.low_memory_cache_slots)),
        "sequence_memory_slots": float(model_meta.get("sequence_memory_slots", args.sequence_memory_slots)),
        "sequence_memory_top_k": float(model_meta.get("sequence_memory_top_k", args.sequence_memory_top_k)),
        "sequence_memory_weight": float(model_meta.get("sequence_memory_weight", args.sequence_memory_weight)),
        "sequence_memory_read_temperature": float(
            model_meta.get("sequence_memory_read_temperature", args.sequence_memory_read_temperature)
        ),
        "sequence_memory_min_surprise": float(model_meta.get("sequence_memory_min_surprise", args.sequence_memory_min_surprise)),
        "memory_decay": float(model_meta.get("memory_decay", args.memory_decay)),
        "memory_decay_interval": float(model_meta.get("memory_decay_interval", args.memory_decay_interval)),
        "memory_decay_min_count": float(model_meta.get("memory_decay_min_count", args.memory_decay_min_count)),
        "accuracy": eval_metrics["accuracy"],
        "val_bpb": eval_metrics["val_bpb"],
        "train_loss": last_loss,
        "train_ms_per_batch": train_s * 1000.0 / args.steps if args.steps > 0 else 0.0,
        "peak_train_cpu_memory_gb": train_peak_cpu,
        "peak_train_gpu_memory_gb": train_peak_gpu,
        "inference_ms_per_batch": infer_ms,
        "peak_inference_cpu_memory_gb": infer_peak_cpu,
        "peak_inference_gpu_memory_gb": infer_peak_gpu,
        "next_token_inference_ms_per_batch": next_infer_ms,
        "peak_next_token_inference_cpu_memory_gb": next_infer_peak_cpu,
        "peak_next_token_inference_gpu_memory_gb": next_infer_peak_gpu,
        "model_path": str(model_path),
    }
    if sample_text:
        result["sample_text"] = sample_text
    result.update(prompt_scores)
    args.results_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.results_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
    if not args.quiet:
        print(json.dumps(result, sort_keys=True))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny HiPC LM and same-data tiny GPT baseline")
    parser.add_argument("--model", choices=["hipc", "tiny-gpt"], default="hipc")
    parser.add_argument("--tokenizer", choices=["byte", "nanochat"], default="byte")
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("runs/hipc_lm/tokenizer"))
    parser.add_argument("--train-tokenizer", action="store_true")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=4096)
    parser.add_argument("--tokenizer-max-chars", type=int, default=2_000_000)
    parser.add_argument("--tokenizer-doc-cap", type=int, default=10_000)
    parser.add_argument("--data-dir", type=Path, default=Path("data/base_data_climbmix"))
    parser.add_argument("--download-shards", type=int, default=0)
    parser.add_argument("--download-workers", type=int, default=1)
    parser.add_argument("--train-shards", type=int, default=1)
    parser.add_argument("--mix-train-shards", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hipc_lm"))
    parser.add_argument("--run-tag", type=str, default="smoke")
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument("--load-low-memory-cache-slots", type=int, default=None)
    parser.add_argument("--results-jsonl", type=Path, default=Path("runs/hipc_lm/results.jsonl"))
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prior-steps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=2)
    parser.add_argument("--eval-logit-temperature", type=float, default=1.0)
    parser.add_argument("--progress-eval-steps", type=int, default=0)
    parser.add_argument("--infer-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--concept-dim", type=int, default=64)
    parser.add_argument("--feature-groups", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--recurrent-stride", type=int, default=1)
    parser.add_argument("--low-cell-mode", choices=["additive", "predictive"], default="additive")
    parser.add_argument("--low-cell-residual-weight", type=float, default=0.5)
    parser.add_argument("--concept-memory-slots", type=int, default=0)
    parser.add_argument("--concept-memory-top-k", type=int, default=4)
    parser.add_argument("--concept-bias-weight", type=float, default=0.0)
    parser.add_argument("--concept-adapter-weight", type=float, default=0.0)
    parser.add_argument("--concept-update-rate", type=float, default=0.05)
    parser.add_argument("--concept-novelty-threshold", type=float, default=0.15)
    parser.add_argument("--phrase-stride", type=int, default=0)
    parser.add_argument("--doc-stride", type=int, default=0)
    parser.add_argument("--hierarchy-adapter-weight", type=float, default=0.0)
    parser.add_argument("--high-vocab-weight", type=float, default=0.0)
    parser.add_argument("--high-vocab-min-confidence", type=float, default=0.0)
    parser.add_argument("--low-memory-slots", type=int, default=0)
    parser.add_argument("--low-memory-top-k", type=int, default=4)
    parser.add_argument("--low-memory-weight", type=float, default=0.0)
    parser.add_argument("--low-state-memory-weight", type=float, default=0.0)
    parser.add_argument("--low-feature-prior-weight", type=float, default=0.0)
    parser.add_argument("--low-latent-prediction-weight", type=float, default=0.0)
    parser.add_argument("--low-latent-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--low-latent-contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--low-memory-update-rate", type=float, default=0.05)
    parser.add_argument("--low-memory-novelty-threshold", type=float, default=0.15)
    parser.add_argument("--low-memory-read-temperature", type=float, default=16.0)
    parser.add_argument("--low-memory-train-read-scale", action="store_true")
    parser.add_argument("--low-memory-min-surprise", type=float, default=0.0)
    parser.add_argument("--low-memory-cache-slots", type=int, default=0)
    parser.add_argument("--sequence-memory-slots", type=int, default=0)
    parser.add_argument("--sequence-memory-top-k", type=int, default=4)
    parser.add_argument("--sequence-memory-weight", type=float, default=0.0)
    parser.add_argument("--sequence-memory-read-temperature", type=float, default=32.0)
    parser.add_argument("--sequence-memory-min-surprise", type=float, default=0.0)
    parser.add_argument("--memory-decay", type=float, default=1.0)
    parser.add_argument("--memory-decay-interval", type=int, default=0)
    parser.add_argument("--memory-decay-min-count", type=float, default=0.01)
    parser.add_argument("--gpt-layers", type=int, default=2)
    parser.add_argument("--gpt-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decoder-lr-scale", type=float, default=1.0)
    parser.add_argument("--state-lr-scale", type=float, default=1.0)
    parser.add_argument("--train-logit-temperature", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--resource-check-interval", type=int, default=10)
    parser.add_argument("--max-system-memory-percent", type=float, default=90.0)
    parser.add_argument("--max-system-load-percent", type=float, default=90.0)
    parser.add_argument("--max-gpu-memory-percent", type=float, default=90.0)
    parser.add_argument("--sample-prompt", type=str, default="")
    parser.add_argument("--sample-tokens", type=int, default=64)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--sample-top-k", type=int, default=40)
    parser.add_argument("--score-text", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
