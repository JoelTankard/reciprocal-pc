from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F

from pfc.llm_data import NanochatBestfitBatcher, build_text_tokenizer
from pfc.message_passing_pc_lm import (
    MessagePassingPCLanguageModel,
    _rms_norm_vjp,
)
from tools.diagnose_message_pc_reciprocal_feedback import (
    add_alignment,
    empty_alignment,
    finish_alignment,
    signal_alignment,
)
from tools.diagnose_pc_context_use import load_model


def estimate_jacobian_transpose(
    noise: Tensor,
    response: Tensor,
) -> Tensor:
    """Estimate J^T from isotropic perturbations and local responses."""

    width = noise.shape[-1]
    flat_noise = noise.reshape(-1, width).float()
    flat_response = response.reshape(-1, width).float()
    return flat_noise.t() @ flat_response / float(max(flat_noise.shape[0], 1))


def estimate_routed_jacobian_transpose(
    noise: Tensor,
    routed_noise: Tensor,
    response: Tensor,
    self_jacobian_transpose: Tensor,
) -> Tensor:
    """Fit an off-token feedback map after removing same-token response."""

    width = noise.shape[-1]
    flat_noise = noise.reshape(-1, width).float()
    flat_routed_noise = routed_noise.reshape(-1, width).float()
    flat_response = response.reshape(-1, width).float()
    residual = flat_response - flat_noise @ self_jacobian_transpose.float()
    numerator = flat_routed_noise.t() @ residual
    denominator = flat_routed_noise.square().sum(dim=0).clamp_min(1e-12)
    return numerator / denominator.unsqueeze(-1)


def fit_state_gated_jacobian_transpose(
    noise: Tensor,
    lower_state: Tensor,
    response: Tensor,
    *,
    ridge: float = 1e-3,
) -> tuple[Tensor, Tensor]:
    """Fit response = noise F + (noise * state) G by local regression."""

    width = noise.shape[-1]
    flat_noise = noise.reshape(-1, width).float()
    flat_state = torch.tanh(lower_state).reshape(-1, width).float()
    flat_response = response.reshape(-1, width).float()
    features = torch.cat((flat_noise, flat_noise * flat_state), dim=-1)
    gram = features.t() @ features
    ridge_scale = ridge * gram.diagonal().mean().clamp_min(1e-12)
    gram.diagonal().add_(ridge_scale)
    solution = torch.linalg.solve(gram, features.t() @ flat_response)
    return solution[:width], solution[width:]


def fit_two_feature_jacobian_transpose(
    first_feature: Tensor,
    second_feature: Tensor,
    response: Tensor,
    *,
    ridge: float = 1e-3,
) -> tuple[Tensor, Tensor]:
    """Fit response from two locally observable perturbation features."""

    width = first_feature.shape[-1]
    first = first_feature.reshape(-1, width).float()
    second = second_feature.reshape(-1, width).float()
    target = response.reshape(-1, width).float()
    features = torch.cat((first, second), dim=-1)
    gram = features.t() @ features
    ridge_scale = ridge * gram.diagonal().mean().clamp_min(1e-12)
    gram.diagonal().add_(ridge_scale)
    solution = torch.linalg.solve(gram, features.t() @ target)
    return solution[:width], solution[width:]


def estimate_clustered_jacobian_transpose(
    noise: Tensor,
    lower_state: Tensor,
    response: Tensor,
    projection: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fit one perturbation-response map per fixed local-state region."""

    width = noise.shape[-1]
    cluster_count = projection.shape[-1]
    flat_noise = noise.reshape(-1, width).float()
    flat_state = lower_state.reshape(-1, width).float()
    flat_response = response.reshape(-1, width).float()
    cluster = (flat_state @ projection.float()).argmax(dim=-1)
    maps = flat_noise.new_zeros(cluster_count, width, width)
    counts = flat_noise.new_zeros(cluster_count)
    for index in range(cluster_count):
        selected = cluster == index
        count = selected.sum()
        if int(count.item()) == 0:
            continue
        maps[index] = (
            flat_noise[selected].t() @ flat_response[selected]
        ) / count.float()
        counts[index] = count
    return maps, counts


def apply_clustered_feedback(
    signal: Tensor,
    lower_state: Tensor,
    maps: Tensor,
    projection: Tensor,
) -> Tensor:
    width = signal.shape[-1]
    cluster = (
        lower_state.reshape(-1, width).float() @ projection.float()
    ).argmax(dim=-1)
    flat_signal = signal.reshape(-1, width)
    output = torch.empty_like(flat_signal)
    for index in range(maps.shape[0]):
        selected = cluster == index
        if bool(selected.any()):
            output[selected] = F.linear(flat_signal[selected], maps[index])
    return output.view_as(signal)


def matrix_cosine(left: Tensor, right: Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.float().reshape(1, -1),
            right.float().reshape(1, -1),
            dim=-1,
            eps=1e-12,
        ).item()
    )


def matrix_geometry(matrix: Tensor, *, top_k: int = 8) -> dict[str, object]:
    singular_values = torch.linalg.svdvals(matrix.float())
    frobenius_squared = matrix.float().square().sum()
    spectral_squared = singular_values[0].square().clamp_min(1e-30)
    return {
        "frobenius_norm": float(frobenius_squared.sqrt().item()),
        "spectral_norm": float(singular_values[0].item()),
        "stable_rank": float((frobenius_squared / spectral_squared).item()),
        "top_singular_values": [
            float(value)
            for value in singular_values[: max(top_k, 1)].tolist()
        ],
    }


@torch.no_grad()
def estimate_block_jacobian_transpose(
    model: MessagePassingPCLanguageModel,
    states: list[Tensor],
    token_ids: Tensor,
    layer: int,
    *,
    probes: int,
    epsilon: float,
    seed: int,
    cluster_projection: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    lower_state = states[layer]
    baseline_cache = model._block(
        layer,
        lower_state,
        token_ids=token_ids,
        sensory_state=states[0],
        encoder_memory=model._causal_encoder_memory(states, layer),
    )
    estimate = torch.zeros(
        model.config.hidden_dim,
        model.config.hidden_dim,
        device=lower_state.device,
        dtype=torch.float32,
    )
    probe_values: list[tuple[Tensor, Tensor, Tensor]] = []
    route_weights = baseline_cache.weights.clone()
    route_weights.diagonal(dim1=-2, dim2=-1).zero_()
    for probe in range(probes):
        generator = torch.Generator(device=lower_state.device)
        generator.manual_seed(seed + 104_729 * probe + 7_919 * layer)
        noise = torch.empty_like(lower_state)
        noise.bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0)
        perturbed_lower = lower_state + epsilon * noise
        encoder_memory = model._causal_encoder_memory(states, layer)
        if encoder_memory is lower_state:
            encoder_memory = perturbed_lower
        perturbed = model._block(
            layer,
            perturbed_lower,
            token_ids=token_ids,
            sensory_state=states[0],
            encoder_memory=encoder_memory,
        ).prediction
        response = (perturbed - baseline_cache.prediction) / epsilon
        routed_noise = model._merge_heads(
            torch.matmul(route_weights, model._split_heads(noise))
        )
        estimate.add_(estimate_jacobian_transpose(noise, response))
        probe_values.append((noise, routed_noise, response))
    estimate.div_(float(probes))
    route_estimate = torch.stack(
        [
            estimate_routed_jacobian_transpose(
                noise,
                routed_noise,
                response,
                estimate,
            )
            for noise, routed_noise, response in probe_values
        ]
    ).mean(dim=0)
    gated_self, gated_state = fit_state_gated_jacobian_transpose(
        torch.stack([values[0] for values in probe_values]),
        lower_state.unsqueeze(0).expand(probes, *lower_state.shape),
        torch.stack([values[2] for values in probe_values]),
    )
    stacked_noise = torch.stack([values[0] for values in probe_values])
    expanded_source_input = baseline_cache.source_input.unsqueeze(0).expand(
        probes,
        *baseline_cache.source_input.shape,
    )
    normalized_noise = baseline_cache.source_input_scale * _rms_norm_vjp(
        expanded_source_input,
        stacked_noise,
    )
    structural_direct, structural_normalized = (
        fit_two_feature_jacobian_transpose(
            stacked_noise,
            normalized_noise,
            torch.stack([values[2] for values in probe_values]),
        )
    )
    clustered_maps, clustered_counts = estimate_clustered_jacobian_transpose(
        stacked_noise,
        lower_state.unsqueeze(0).expand(probes, *lower_state.shape),
        torch.stack([values[2] for values in probe_values]),
        cluster_projection,
    )
    return (
        estimate,
        route_estimate,
        gated_self,
        gated_state,
        structural_direct,
        structural_normalized,
        clustered_maps,
        clustered_counts,
    )


def reciprocal_attention_route(
    model: MessagePassingPCLanguageModel,
    signal: Tensor,
    weights: Tensor,
) -> Tensor:
    off_diagonal = weights.clone()
    off_diagonal.diagonal(dim1=-2, dim2=-1).zero_()
    return model._merge_heads(
        torch.matmul(
            off_diagonal.transpose(-2, -1),
            model._split_heads(signal),
        )
    )


@torch.no_grad()
def estimate_top_jacobian_transpose(
    model: MessagePassingPCLanguageModel,
    states: list[Tensor],
    token_ids: Tensor,
    lower_layer: int,
    *,
    probes: int,
    epsilon: float,
    seed: int,
) -> Tensor:
    """Estimate direct top-state response to one lower block output."""

    lower_state = states[lower_layer + 1]
    estimate = torch.zeros(
        model.config.hidden_dim,
        model.config.hidden_dim,
        device=lower_state.device,
        dtype=torch.float32,
    )
    for probe in range(probes):
        generator = torch.Generator(device=lower_state.device)
        generator.manual_seed(seed + 130_363 * probe + 12_289 * lower_layer)
        noise = torch.empty_like(lower_state)
        noise.bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0)
        propagated_states = list(states[: lower_layer + 1])
        propagated_states.append(lower_state + epsilon * noise)
        for block_layer in range(lower_layer + 1, model.config.num_layers):
            propagated_states.append(
                model._block(
                    block_layer,
                    propagated_states[-1],
                    token_ids=token_ids,
                    sensory_state=states[0],
                    encoder_memory=model._causal_encoder_memory(
                        propagated_states,
                        block_layer,
                    ),
                ).prediction
            )
        response = (propagated_states[-1] - states[-1]) / epsilon
        estimate.add_(estimate_jacobian_transpose(noise, response))
    return estimate / float(probes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure average local Jacobian geometry and static-feedback fit "
            "for a Meta-PC checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("runs/pc_modal_lm/tokenizer_4096"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--calibration-steps", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=4)
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(
        args.batch_size,
        args.seq_len,
        args.calibration_steps,
        args.eval_steps,
        args.probes,
    ) <= 0:
        raise ValueError("batch, sequence, step, and probe counts must be positive")
    if args.epsilon <= 0:
        raise ValueError("epsilon must be positive")

    device = torch.device(args.device)
    loaded, checkpoint_meta = load_model(args.checkpoint, device)
    if not isinstance(loaded, MessagePassingPCLanguageModel):
        raise ValueError("checkpoint must contain a message-PC model")
    model = loaded
    if model.config.inference_mode != "meta_pc":
        raise ValueError("checkpoint must use Meta-PC")

    tokenizer = build_text_tokenizer("nanochat", args.tokenizer_dir)
    bos_token_id = tokenizer.get_bos_token_id()
    batcher = NanochatBestfitBatcher(
        tokenizer_dir=args.tokenizer_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
        tokenizer=tokenizer,
        split="val",
    )
    layers = list(range(1, model.config.num_layers))
    cluster_count = 8
    cluster_generator = torch.Generator(device=device)
    cluster_generator.manual_seed(8_675_309)
    cluster_projection = torch.randn(
        model.config.hidden_dim,
        cluster_count,
        generator=cluster_generator,
        device=device,
    ) / math.sqrt(model.config.hidden_dim)
    batch_matrices: dict[int, list[Tensor]] = {layer: [] for layer in layers}
    batch_route_matrices: dict[int, list[Tensor]] = {
        layer: [] for layer in layers
    }
    batch_gated_self_matrices: dict[int, list[Tensor]] = {
        layer: [] for layer in layers
    }
    batch_gated_state_matrices: dict[int, list[Tensor]] = {
        layer: [] for layer in layers
    }
    batch_structural_direct_matrices: dict[int, list[Tensor]] = {
        layer: [] for layer in layers
    }
    batch_structural_normalized_matrices: dict[int, list[Tensor]] = {
        layer: [] for layer in layers
    }
    cluster_matrix_sums = {
        layer: torch.zeros(
            cluster_count,
            model.config.hidden_dim,
            model.config.hidden_dim,
            device=device,
        )
        for layer in layers
    }
    cluster_counts = {
        layer: torch.zeros(cluster_count, device=device)
        for layer in layers
    }
    direct_layers = list(range(model.config.num_layers - 1))
    batch_direct_matrices: dict[int, list[Tensor]] = {
        layer: [] for layer in direct_layers
    }
    with torch.no_grad():
        for calibration_step in range(args.calibration_steps):
            x, _, _ = batcher.next_with_context()
            states, _ = model._forward_states(x, with_cache=False)
            for layer in layers:
                (
                    matrix,
                    route_matrix,
                    gated_self_matrix,
                    gated_state_matrix,
                    structural_direct_matrix,
                    structural_normalized_matrix,
                    clustered_matrices,
                    clustered_count,
                ) = estimate_block_jacobian_transpose(
                    model,
                    states,
                    x,
                    layer,
                    probes=args.probes,
                    epsilon=args.epsilon,
                    seed=1_000_003 * calibration_step + 17,
                    cluster_projection=cluster_projection,
                )
                batch_matrices[layer].append(matrix)
                batch_route_matrices[layer].append(route_matrix)
                batch_gated_self_matrices[layer].append(gated_self_matrix)
                batch_gated_state_matrices[layer].append(gated_state_matrix)
                batch_structural_direct_matrices[layer].append(
                    structural_direct_matrix
                )
                batch_structural_normalized_matrices[layer].append(
                    structural_normalized_matrix
                )
                cluster_matrix_sums[layer].add_(
                    clustered_matrices * clustered_count[:, None, None]
                )
                cluster_counts[layer].add_(clustered_count)
            for layer in direct_layers:
                batch_direct_matrices[layer].append(
                    estimate_top_jacobian_transpose(
                        model,
                        states,
                        x,
                        layer,
                        probes=args.probes,
                        epsilon=args.epsilon,
                        seed=2_000_003 * calibration_step + 29,
                    )
                )

    mean_matrices = {
        layer: torch.stack(batch_matrices[layer]).mean(dim=0)
        for layer in layers
    }
    mean_route_matrices = {
        layer: torch.stack(batch_route_matrices[layer]).mean(dim=0)
        for layer in layers
    }
    mean_direct_matrices = {
        layer: torch.stack(batch_direct_matrices[layer]).mean(dim=0)
        for layer in direct_layers
    }
    mean_gated_self_matrices = {
        layer: torch.stack(batch_gated_self_matrices[layer]).mean(dim=0)
        for layer in layers
    }
    mean_gated_state_matrices = {
        layer: torch.stack(batch_gated_state_matrices[layer]).mean(dim=0)
        for layer in layers
    }
    mean_structural_direct_matrices = {
        layer: torch.stack(
            batch_structural_direct_matrices[layer]
        ).mean(dim=0)
        for layer in layers
    }
    mean_structural_normalized_matrices = {
        layer: torch.stack(
            batch_structural_normalized_matrices[layer]
        ).mean(dim=0)
        for layer in layers
    }
    mean_clustered_matrices = {
        layer: cluster_matrix_sums[layer]
        / cluster_counts[layer][:, None, None].clamp_min(1.0)
        for layer in layers
    }
    average_alignment_by_layer = {
        layer: empty_alignment() for layer in layers
    }
    learned_alignment_by_layer = {
        layer: empty_alignment() for layer in layers
    }
    routed_alignment_by_layer = {
        layer: empty_alignment() for layer in layers
    }
    average_alignment = empty_alignment()
    learned_alignment = empty_alignment()
    routed_alignment = empty_alignment()
    direct_alignment_by_layer = {
        layer: empty_alignment() for layer in direct_layers
    }
    composed_alignment_by_layer = {
        layer: empty_alignment() for layer in direct_layers
    }
    direct_alignment = empty_alignment()
    composed_alignment = empty_alignment()
    gated_alignment_by_layer = {
        layer: empty_alignment() for layer in layers
    }
    gated_composed_alignment_by_layer = {
        layer: empty_alignment() for layer in direct_layers
    }
    gated_alignment = empty_alignment()
    gated_composed_alignment = empty_alignment()
    structural_alignment_by_layer = {
        layer: empty_alignment() for layer in layers
    }
    structural_composed_alignment_by_layer = {
        layer: empty_alignment() for layer in direct_layers
    }
    structural_alignment = empty_alignment()
    structural_composed_alignment = empty_alignment()
    clustered_alignment_by_layer = {
        layer: empty_alignment() for layer in layers
    }
    clustered_composed_alignment_by_layer = {
        layer: empty_alignment() for layer in direct_layers
    }
    clustered_alignment = empty_alignment()
    clustered_composed_alignment = empty_alignment()
    with torch.no_grad():
        for _ in range(args.eval_steps):
            x, y, _ = batcher.next_with_context()
            states, caches = model._forward_states(x, with_cache=True)
            top_feedback, _ = model._meta_pc_top_feedback(states[-1], y)
            exact_upper = top_feedback
            mask = y != bos_token_id
            exact_by_layer: dict[int, Tensor] = {}
            for layer in reversed(layers):
                exact_lower = model._block_input_signal(
                    layer,
                    states[layer],
                    caches[layer],
                    exact_upper,
                )
                average_lower = F.linear(exact_upper, mean_matrices[layer])
                routed_signal = reciprocal_attention_route(
                    model,
                    exact_upper,
                    caches[layer].weights,
                )
                routed_lower = average_lower + F.linear(
                    routed_signal,
                    mean_route_matrices[layer],
                )
                gated_lower = F.linear(
                    exact_upper,
                    mean_gated_self_matrices[layer],
                ) + torch.tanh(states[layer]) * F.linear(
                    exact_upper,
                    mean_gated_state_matrices[layer],
                )
                structural_lower = F.linear(
                    exact_upper,
                    mean_structural_direct_matrices[layer],
                ) + caches[layer].source_input_scale * _rms_norm_vjp(
                    caches[layer].source_input,
                    F.linear(
                        exact_upper,
                        mean_structural_normalized_matrices[layer],
                    ),
                )
                clustered_lower = apply_clustered_feedback(
                    exact_upper,
                    states[layer],
                    mean_clustered_matrices[layer],
                    cluster_projection,
                )
                learned_lower = model._meta_pc_feedback_signal(
                    layer,
                    states[layer],
                    caches[layer],
                    exact_upper,
                )
                average_values = signal_alignment(
                    exact_lower,
                    average_lower,
                    mask,
                )
                learned_values = signal_alignment(
                    exact_lower,
                    learned_lower,
                    mask,
                )
                routed_values = signal_alignment(
                    exact_lower,
                    routed_lower,
                    mask,
                )
                gated_values = signal_alignment(
                    exact_lower,
                    gated_lower,
                    mask,
                )
                structural_values = signal_alignment(
                    exact_lower,
                    structural_lower,
                    mask,
                )
                clustered_values = signal_alignment(
                    exact_lower,
                    clustered_lower,
                    mask,
                )
                add_alignment(average_alignment_by_layer[layer], average_values)
                add_alignment(learned_alignment_by_layer[layer], learned_values)
                add_alignment(routed_alignment_by_layer[layer], routed_values)
                add_alignment(gated_alignment_by_layer[layer], gated_values)
                add_alignment(
                    structural_alignment_by_layer[layer],
                    structural_values,
                )
                add_alignment(
                    clustered_alignment_by_layer[layer],
                    clustered_values,
                )
                add_alignment(average_alignment, average_values)
                add_alignment(learned_alignment, learned_values)
                add_alignment(routed_alignment, routed_values)
                add_alignment(gated_alignment, gated_values)
                add_alignment(structural_alignment, structural_values)
                add_alignment(clustered_alignment, clustered_values)
                exact_by_layer[layer - 1] = exact_lower
                exact_upper = exact_lower

            configured_upper = top_feedback
            gated_upper = top_feedback
            structural_upper = top_feedback
            clustered_upper = top_feedback
            for layer in reversed(layers):
                configured_lower = model._meta_pc_feedback_signal(
                    layer,
                    states[layer],
                    caches[layer],
                    configured_upper,
                )
                target_layer = layer - 1
                composed_values = signal_alignment(
                    exact_by_layer[target_layer],
                    configured_lower,
                    mask,
                )
                add_alignment(
                    composed_alignment_by_layer[target_layer],
                    composed_values,
                )
                add_alignment(composed_alignment, composed_values)
                configured_upper = configured_lower
                gated_lower = F.linear(
                    gated_upper,
                    mean_gated_self_matrices[layer],
                ) + torch.tanh(states[layer]) * F.linear(
                    gated_upper,
                    mean_gated_state_matrices[layer],
                )
                gated_composed_values = signal_alignment(
                    exact_by_layer[target_layer],
                    gated_lower,
                    mask,
                )
                add_alignment(
                    gated_composed_alignment_by_layer[target_layer],
                    gated_composed_values,
                )
                add_alignment(
                    gated_composed_alignment,
                    gated_composed_values,
                )
                gated_upper = gated_lower
                structural_lower = F.linear(
                    structural_upper,
                    mean_structural_direct_matrices[layer],
                ) + caches[layer].source_input_scale * _rms_norm_vjp(
                    caches[layer].source_input,
                    F.linear(
                        structural_upper,
                        mean_structural_normalized_matrices[layer],
                    ),
                )
                structural_composed_values = signal_alignment(
                    exact_by_layer[target_layer],
                    structural_lower,
                    mask,
                )
                add_alignment(
                    structural_composed_alignment_by_layer[target_layer],
                    structural_composed_values,
                )
                add_alignment(
                    structural_composed_alignment,
                    structural_composed_values,
                )
                structural_upper = structural_lower
                clustered_lower = apply_clustered_feedback(
                    clustered_upper,
                    states[layer],
                    mean_clustered_matrices[layer],
                    cluster_projection,
                )
                clustered_composed_values = signal_alignment(
                    exact_by_layer[target_layer],
                    clustered_lower,
                    mask,
                )
                add_alignment(
                    clustered_composed_alignment_by_layer[target_layer],
                    clustered_composed_values,
                )
                add_alignment(
                    clustered_composed_alignment,
                    clustered_composed_values,
                )
                clustered_upper = clustered_lower

            for layer in direct_layers:
                direct_lower = F.linear(
                    top_feedback,
                    mean_direct_matrices[layer],
                )
                direct_values = signal_alignment(
                    exact_by_layer[layer],
                    direct_lower,
                    mask,
                )
                add_alignment(direct_alignment_by_layer[layer], direct_values)
                add_alignment(direct_alignment, direct_values)

    layer_results: dict[str, object] = {}
    for layer in layers:
        mean_matrix = mean_matrices[layer]
        feedback_matrix = model.feedback_weights[layer]
        drift = [
            matrix_cosine(matrix, mean_matrix)
            for matrix in batch_matrices[layer]
        ]
        layer_results[str(layer)] = {
            "geometry": matrix_geometry(mean_matrix),
            "route_geometry": matrix_geometry(mean_route_matrices[layer]),
            "gated_state_geometry": matrix_geometry(
                mean_gated_state_matrices[layer]
            ),
            "structural_normalized_geometry": matrix_geometry(
                mean_structural_normalized_matrices[layer]
            ),
            "cluster_counts": [
                int(value) for value in cluster_counts[layer].tolist()
            ],
            "batch_to_mean_cosine_mean": sum(drift) / len(drift),
            "batch_to_mean_cosine_min": min(drift),
            "feedback_matrix_to_mean_cosine": matrix_cosine(
                feedback_matrix,
                mean_matrix,
            ),
            "average_jacobian_credit": finish_alignment(
                average_alignment_by_layer[layer]
            ),
            "routed_average_jacobian_credit": finish_alignment(
                routed_alignment_by_layer[layer]
            ),
            "state_gated_average_credit": finish_alignment(
                gated_alignment_by_layer[layer]
            ),
            "structural_average_credit": finish_alignment(
                structural_alignment_by_layer[layer]
            ),
            "clustered_average_credit": finish_alignment(
                clustered_alignment_by_layer[layer]
            ),
            "configured_feedback_credit": finish_alignment(
                learned_alignment_by_layer[layer]
            ),
        }

    direct_results = {
        str(layer): {
            "geometry": matrix_geometry(mean_direct_matrices[layer]),
            "batch_to_mean_cosine_mean": sum(
                matrix_cosine(matrix, mean_direct_matrices[layer])
                for matrix in batch_direct_matrices[layer]
            )
            / len(batch_direct_matrices[layer]),
            "direct_average_credit": finish_alignment(
                direct_alignment_by_layer[layer]
            ),
            "configured_composed_credit": finish_alignment(
                composed_alignment_by_layer[layer]
            ),
            "state_gated_composed_credit": finish_alignment(
                gated_composed_alignment_by_layer[layer]
            ),
            "structural_composed_credit": finish_alignment(
                structural_composed_alignment_by_layer[layer]
            ),
            "clustered_composed_credit": finish_alignment(
                clustered_composed_alignment_by_layer[layer]
            ),
        }
        for layer in direct_layers
    }

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_meta.get(
            "step",
            checkpoint_meta.get("steps", 0),
        ),
        "calibration_steps": args.calibration_steps,
        "eval_steps": args.eval_steps,
        "probes": args.probes,
        "epsilon": args.epsilon,
        "feedback_mode": model.config.meta_pc_feedback_mode,
        "output_feedback_mode": model.config.meta_pc_output_feedback_mode,
        "global_backprops": 0,
        "estimator": "local_rademacher_perturbation_response",
        "average_jacobian_credit": finish_alignment(average_alignment),
        "routed_average_jacobian_credit": finish_alignment(routed_alignment),
        "state_gated_average_credit": finish_alignment(gated_alignment),
        "state_gated_composed_credit": finish_alignment(
            gated_composed_alignment
        ),
        "structural_average_credit": finish_alignment(structural_alignment),
        "structural_composed_credit": finish_alignment(
            structural_composed_alignment
        ),
        "cluster_count": cluster_count,
        "clustered_average_credit": finish_alignment(clustered_alignment),
        "clustered_composed_credit": finish_alignment(
            clustered_composed_alignment
        ),
        "configured_feedback_credit": finish_alignment(learned_alignment),
        "configured_composed_credit": finish_alignment(composed_alignment),
        "direct_average_credit": finish_alignment(direct_alignment),
        "direct_layers": direct_results,
        "layers": layer_results,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
