import pytest
import torch
import torch.nn.functional as F

from pfc.message_passing_pc_lm import (
    MessagePassingPCConfig,
    MessagePassingPCLanguageModel,
)
from tools.diagnose_message_pc_jacobian_geometry import (
    apply_clustered_feedback,
    estimate_clustered_jacobian_transpose,
    estimate_jacobian_transpose,
    estimate_routed_jacobian_transpose,
    fit_state_gated_jacobian_transpose,
    fit_two_feature_jacobian_transpose,
    matrix_cosine,
    matrix_geometry,
)
from tools.diagnose_message_pc_stage_reciprocal import (
    build_stage_model,
    credit_mask,
    matrix_alignment,
)


def test_estimate_jacobian_transpose_recovers_linear_map() -> None:
    torch.manual_seed(1)
    width = 8
    rows = 32_768
    forward_weight = torch.randn(width, width)
    noise = torch.empty(rows, width).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    response = noise @ forward_weight.t()

    estimate = estimate_jacobian_transpose(noise, response)

    assert torch.allclose(estimate, forward_weight.t(), atol=0.035, rtol=0.04)


def test_matrix_geometry_reports_identity_stable_rank() -> None:
    identity = torch.eye(8)

    geometry = matrix_geometry(identity, top_k=3)

    assert geometry["stable_rank"] == 8.0
    assert geometry["top_singular_values"] == [1.0, 1.0, 1.0]
    assert matrix_cosine(identity, identity) == pytest.approx(1.0)


def test_estimate_routed_jacobian_recovers_residual_map() -> None:
    torch.manual_seed(2)
    width = 8
    rows = 32_768
    self_map = torch.randn(width, width)
    route_map = torch.randn(width, width)
    noise = torch.randn(rows, width)
    routed_noise = torch.randn(rows, width)
    response = noise @ self_map + routed_noise @ route_map

    estimate = estimate_routed_jacobian_transpose(
        noise,
        routed_noise,
        response,
        self_map,
    )

    assert torch.allclose(estimate, route_map, atol=0.04, rtol=0.04)


def test_fit_state_gated_jacobian_recovers_bilinear_map() -> None:
    torch.manual_seed(3)
    width = 8
    rows = 32_768
    self_map = torch.randn(width, width)
    state_map = torch.randn(width, width)
    noise = torch.randn(rows, width)
    lower_state = torch.randn(rows, width)
    response = noise @ self_map + (
        noise * torch.tanh(lower_state)
    ) @ state_map

    fitted_self, fitted_state = fit_state_gated_jacobian_transpose(
        noise,
        lower_state,
        response,
    )

    assert torch.allclose(fitted_self, self_map, atol=0.01, rtol=0.01)
    assert torch.allclose(fitted_state, state_map, atol=0.01, rtol=0.01)


def test_fit_two_feature_jacobian_recovers_both_maps() -> None:
    torch.manual_seed(4)
    width = 8
    rows = 32_768
    first_map = torch.randn(width, width)
    second_map = torch.randn(width, width)
    first = torch.randn(rows, width)
    second = torch.randn(rows, width)
    response = first @ first_map + second @ second_map

    fitted_first, fitted_second = fit_two_feature_jacobian_transpose(
        first,
        second,
        response,
    )

    assert torch.allclose(fitted_first, first_map, atol=0.01, rtol=0.01)
    assert torch.allclose(fitted_second, second_map, atol=0.01, rtol=0.01)


def test_clustered_jacobian_recovers_state_regions() -> None:
    torch.manual_seed(5)
    width = 8
    rows = 65_536
    projection = torch.eye(width, 2)
    lower_state = torch.randn(rows, width)
    lower_state[:, :2] = 0.0
    lower_state[: rows // 2, 0] = 1.0
    lower_state[rows // 2 :, 1] = 1.0
    maps = torch.randn(2, width, width)
    noise = torch.randn(rows, width)
    cluster = (lower_state @ projection).argmax(dim=-1)
    response = torch.empty_like(noise)
    for index in range(2):
        selected = cluster == index
        response[selected] = noise[selected] @ maps[index]

    fitted, counts = estimate_clustered_jacobian_transpose(
        noise,
        lower_state,
        response,
        projection,
    )
    signal = torch.randn(rows, width)
    actual = apply_clustered_feedback(
        signal,
        lower_state,
        fitted,
        projection,
    )
    expected = torch.empty_like(signal)
    for index in range(2):
        selected = cluster == index
        expected[selected] = F.linear(signal[selected], maps[index])

    assert counts.tolist() == [rows // 2, rows // 2]
    assert torch.allclose(fitted, maps, atol=0.04, rtol=0.04)
    assert torch.allclose(actual, expected, atol=0.25, rtol=0.08)


def test_stage_credit_mask_excludes_sink_and_bos_positions() -> None:
    y = torch.tensor([[1, 2, 3, 31, 5], [6, 7, 8, 9, 10]])

    mask = credit_mask(y, bos_token_id=31, skip_first=2)

    assert mask.tolist() == [
        [False, False, True, False, True],
        [False, False, True, True, True],
    ]


def test_build_stage_model_preserves_forward_without_copying_stage_mirrors() -> None:
    torch.manual_seed(6)
    source = MessagePassingPCLanguageModel(
        MessagePassingPCConfig(
            vocab_size=32,
            bos_token_id=31,
            seq_len=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            inference_steps=1,
            route_update_mode="local_energy",
            block_state_mode="nanochat_residual",
            ffn_dim=48,
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            meta_pc_feedback_mode="reciprocal_attention",
            meta_pc_output_feedback_mode="kp_learned",
        )
    )
    stage = build_stage_model(source, learning_rate=0.1)
    x = torch.randint(0, source.config.vocab_size, (2, 8))

    source_states, _ = source._forward_states(x, with_cache=False)
    stage_states, _ = stage._forward_states(x, with_cache=False)

    assert stage.config.meta_pc_feedback_mode == "stage_reciprocal"
    assert stage.config.meta_pc_reciprocal_update_before_settle
    assert torch.equal(stage.feedback_embedding, source.feedback_embedding)
    assert torch.equal(stage_states[-1], source_states[-1])
    assert torch.count_nonzero(stage.feedback_query_weights[1]) == 0


def test_stage_matrix_alignment_reports_exact_match() -> None:
    matrix = torch.randn(8, 5)

    alignment = matrix_alignment(matrix, matrix.clone())

    assert alignment["cosine"] == pytest.approx(1.0)
    assert alignment["norm_ratio"] == pytest.approx(1.0)
    assert alignment["relative_error"] == 0.0
