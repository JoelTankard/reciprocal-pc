from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from pfc.message_passing_pc_lm import (
    MessagePassingPCConfig,
    MessagePassingPCLanguageModel,
    _cap_token_rms,
    _cap_token_rms_vjp,
    _rms_norm,
    _rms_norm_vjp,
)
from pfc.pc_modal_lm import build_model, count_parameter_slots, count_params, parse_args
from tools.diagnose_pc_context_use import load_model
from tools.diagnose_message_pc_causal_error_carry import (
    causal_error_carry_logits,
    observed_next_tokens,
)
from tools.diagnose_message_pc_gradient_alignment import (
    capture_local_gradients,
    free_phase_block_gradients,
    free_local_ce_block_gradients,
    free_local_ce_embedding_gradient,
    oracle_gradients,
    parse_settling_variant,
    vector_alignment,
)
from tools.evaluate_pc_checkpoint import effective_message_readout_layer


def small_config(**overrides: object) -> MessagePassingPCConfig:
    values: dict[str, object] = {
        "vocab_size": 32,
        "bos_token_id": 31,
        "seq_len": 8,
        "hidden_dim": 32,
        "num_layers": 2,
        "num_heads": 4,
        "inference_steps": 2,
        "route_update_mode": "local_energy",
    }
    values.update(overrides)
    return MessagePassingPCConfig(**values)  # type: ignore[arg-type]


def test_message_pc_validates_head_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MessagePassingPCLanguageModel(small_config(hidden_dim=30))


def test_message_pc_token_rms_cap_vjp_matches_local_autograd() -> None:
    torch.manual_seed(63)
    value = torch.randn(2, 3, 32)
    value[0].mul_(0.01)
    value.requires_grad_()
    output_signal = torch.randn_like(value)
    objective = (_cap_token_rms(value, 0.2) * output_signal).sum()
    (expected,) = torch.autograd.grad(objective, (value,))

    actual = _cap_token_rms_vjp(value.detach(), output_signal, 0.2)

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)

    with pytest.raises(ValueError, match="route_update_mode"):
        MessagePassingPCLanguageModel(small_config(route_update_mode="unknown"))
    with pytest.raises(ValueError, match="decoder_supervision_mode"):
        MessagePassingPCLanguageModel(small_config(decoder_supervision_mode="unknown"))
    with pytest.raises(ValueError, match="decoder_state_mode"):
        MessagePassingPCLanguageModel(small_config(decoder_state_mode="unknown"))
    with pytest.raises(ValueError, match="readout_layer"):
        MessagePassingPCLanguageModel(small_config(readout_layer=3))
    with pytest.raises(ValueError, match="readout_objective_share"):
        MessagePassingPCLanguageModel(small_config(readout_objective_share=1.1))
    with pytest.raises(ValueError, match="winner margin"):
        MessagePassingPCLanguageModel(small_config(winner_margin_scale=-0.1))
    with pytest.raises(ValueError, match="block_state_mode"):
        MessagePassingPCLanguageModel(small_config(block_state_mode="unknown"))
    with pytest.raises(ValueError, match="even head dimension"):
        MessagePassingPCLanguageModel(
            small_config(
                hidden_dim=36,
                block_state_mode="nanochat_residual",
            )
        )
    with pytest.raises(ValueError, match="block_parameterization"):
        MessagePassingPCLanguageModel(
            small_config(block_parameterization="unknown")
        )
    with pytest.raises(ValueError, match="identity_residual"):
        MessagePassingPCLanguageModel(
            small_config(block_parameterization="depth_mup")
        )
    with pytest.raises(ValueError, match="does not yet support FFNs"):
        MessagePassingPCLanguageModel(
            small_config(
                block_state_mode="identity_residual",
                block_parameterization="depth_mup",
                ffn_dim=48,
            )
        )
    with pytest.raises(ValueError, match="residual_scale"):
        MessagePassingPCLanguageModel(small_config(residual_scale=0.0))
    with pytest.raises(ValueError, match="initialization gains"):
        MessagePassingPCLanguageModel(small_config(message_init_gain=-0.1))
    with pytest.raises(ValueError, match="ffn_dim"):
        MessagePassingPCLanguageModel(small_config(ffn_dim=-1))
    with pytest.raises(ValueError, match="ffn_init_gain"):
        MessagePassingPCLanguageModel(small_config(ffn_init_gain=-0.1))
    with pytest.raises(ValueError, match="ffn_lr_scale"):
        MessagePassingPCLanguageModel(small_config(ffn_lr_scale=-0.1))
    with pytest.raises(ValueError, match="ffn_learning_source"):
        MessagePassingPCLanguageModel(
            small_config(ffn_learning_source="unknown")
        )
    with pytest.raises(ValueError, match="token_feedback_mode"):
        MessagePassingPCLanguageModel(small_config(token_feedback_mode="unknown"))
    with pytest.raises(ValueError, match="inference_mode"):
        MessagePassingPCLanguageModel(small_config(inference_mode="unknown"))
    with pytest.raises(ValueError, match="meta_pc_feedback_mode"):
        MessagePassingPCLanguageModel(
            small_config(meta_pc_feedback_mode="unknown")
        )
    with pytest.raises(ValueError, match="meta_pc_reciprocal_scale"):
        MessagePassingPCLanguageModel(
            small_config(meta_pc_reciprocal_scale=-0.1)
        )
    with pytest.raises(ValueError, match="reciprocal_learning_rate"):
        MessagePassingPCLanguageModel(
            small_config(meta_pc_reciprocal_learning_rate=1.1)
        )
    with pytest.raises(ValueError, match="reciprocal_noise_std"):
        MessagePassingPCLanguageModel(
            small_config(meta_pc_reciprocal_noise_std=0.0)
        )
    with pytest.raises(ValueError, match="positive learning rate"):
        MessagePassingPCLanguageModel(
            small_config(
                inference_mode="meta_pc",
                decoder_supervision_mode="top_only",
                forward_update_top_layers=2,
                meta_pc_feedback_mode="learned_reciprocal",
            )
        )
    with pytest.raises(ValueError, match="pre-settle reciprocal update"):
        MessagePassingPCLanguageModel(
            small_config(meta_pc_reciprocal_update_before_settle=True)
        )
    with pytest.raises(ValueError, match="state-gated reciprocal feedback"):
        MessagePassingPCLanguageModel(
            small_config(meta_pc_state_gated_reciprocal=True)
        )
    with pytest.raises(ValueError, match="positive learning rate"):
        MessagePassingPCLanguageModel(
            small_config(meta_pc_feedback_mode="stage_reciprocal")
        )
    with pytest.raises(ValueError, match="nanochat_residual"):
        MessagePassingPCLanguageModel(
            small_config(
                meta_pc_feedback_mode="stage_reciprocal",
                meta_pc_reciprocal_learning_rate=0.1,
            )
        )
    with pytest.raises(ValueError, match="meta_pc requires top_only"):
        MessagePassingPCLanguageModel(
            small_config(
                inference_mode="meta_pc",
                forward_update_top_layers=2,
            )
        )
    with pytest.raises(ValueError, match="Forward Update on every layer"):
        MessagePassingPCLanguageModel(
            small_config(
                inference_mode="meta_pc",
                decoder_supervision_mode="top_only",
                forward_update_top_layers=1,
            )
        )
    with pytest.raises(ValueError, match="recurrent_shared_steps"):
        MessagePassingPCLanguageModel(
            small_config(recurrent_shared_steps=0)
        )
    with pytest.raises(ValueError, match="recurrent_step_scale"):
        MessagePassingPCLanguageModel(
            small_config(recurrent_step_scale=0.0)
        )
    with pytest.raises(ValueError, match="recurrent_step_scale"):
        MessagePassingPCLanguageModel(
            small_config(recurrent_step_scale=1.1)
        )
    with pytest.raises(ValueError, match="exactly one block"):
        MessagePassingPCLanguageModel(
            small_config(inference_mode="recurrent_local")
        )
    with pytest.raises(ValueError, match="local outcome modes"):
        MessagePassingPCLanguageModel(
            small_config(inference_mode="layerwise_local")
        )
    with pytest.raises(ValueError, match="top_only"):
        MessagePassingPCLanguageModel(
            small_config(inference_mode="difference_target")
        )
    with pytest.raises(ValueError, match="inverse_noise"):
        MessagePassingPCLanguageModel(
            small_config(difference_target_inverse_noise=0.0)
        )
    with pytest.raises(ValueError, match="inverse_damping"):
        MessagePassingPCLanguageModel(
            small_config(difference_target_inverse_damping=-0.1)
        )
    with pytest.raises(ValueError, match="path_mode"):
        MessagePassingPCLanguageModel(
            small_config(difference_target_path_mode="unknown")
        )
    with pytest.raises(ValueError, match="learning_phase_source"):
        MessagePassingPCLanguageModel(small_config(learning_phase_source="unknown"))
    with pytest.raises(ValueError, match="free_phase_amortization_scale"):
        MessagePassingPCLanguageModel(
            small_config(free_phase_amortization_scale=-0.1)
        )
    with pytest.raises(ValueError, match="free_local_token_credit_scale"):
        MessagePassingPCLanguageModel(
            small_config(free_local_token_credit_scale=-0.1)
        )
    with pytest.raises(
        ValueError,
        match="free_local_token_embedding_credit_scale",
    ):
        MessagePassingPCLanguageModel(
            small_config(free_local_token_embedding_credit_scale=-0.1)
        )
    with pytest.raises(ValueError, match="requires settled"):
        MessagePassingPCLanguageModel(
            small_config(
                learning_phase_source="free",
                free_phase_amortization_scale=0.1,
            )
        )
    with pytest.raises(ValueError, match="decoder_learning_source"):
        MessagePassingPCLanguageModel(small_config(decoder_learning_source="unknown"))
    with pytest.raises(ValueError, match="learned_feedback_proposal_scale"):
        MessagePassingPCLanguageModel(
            small_config(learned_feedback_proposal_scale=-0.1)
        )
    with pytest.raises(ValueError, match="residual_error_transport_mode"):
        MessagePassingPCLanguageModel(
            small_config(residual_error_transport_mode="unknown")
        )
    with pytest.raises(ValueError, match="learned_only"):
        MessagePassingPCLanguageModel(
            small_config(residual_error_transport_mode="learned_only")
        )
    with pytest.raises(ValueError, match="residual_error_suffix_decay"):
        MessagePassingPCLanguageModel(
            small_config(residual_error_suffix_decay=-0.1)
        )
    with pytest.raises(ValueError, match="residual_error_target_layer"):
        MessagePassingPCLanguageModel(
            small_config(residual_error_target_layer="unknown")
        )
    with pytest.raises(ValueError, match="proxy_memory_scale"):
        MessagePassingPCLanguageModel(
            small_config(residual_error_proxy_memory_scale=-0.1)
        )
    with pytest.raises(ValueError, match="proxy_memory_decay"):
        MessagePassingPCLanguageModel(
            small_config(residual_error_proxy_memory_decay=1.1)
        )
    with pytest.raises(ValueError, match="upper_proxy_state_scale"):
        MessagePassingPCLanguageModel(
            small_config(residual_error_upper_proxy_state_scale=-0.1)
        )
    with pytest.raises(ValueError, match="upper_proxy_state_decay"):
        MessagePassingPCLanguageModel(
            small_config(residual_error_upper_proxy_state_decay=1.1)
        )
    with pytest.raises(ValueError, match="upper_content_state_scale"):
        MessagePassingPCLanguageModel(
            small_config(residual_error_upper_content_state_scale=-0.1)
        )
    with pytest.raises(ValueError, match="upper_content_state_decay"):
        MessagePassingPCLanguageModel(
            small_config(residual_error_upper_content_state_decay=1.1)
        )
    with pytest.raises(ValueError, match="upper_content_write_period"):
        MessagePassingPCLanguageModel(
            small_config(residual_error_upper_content_write_period=-1)
        )
    with pytest.raises(ValueError, match="upper_content_read_period"):
        MessagePassingPCLanguageModel(
            small_config(residual_error_upper_content_read_period=-1)
        )
    with pytest.raises(ValueError, match="upper_content_read_delay"):
        MessagePassingPCLanguageModel(
            small_config(residual_error_upper_content_read_delay=0)
        )
    with pytest.raises(ValueError, match="upper_content_write_offset"):
        MessagePassingPCLanguageModel(
            small_config(
                residual_error_upper_content_write_period=3,
                residual_error_upper_content_write_offset=3,
            )
        )
    with pytest.raises(ValueError, match="upper_content_read_offset"):
        MessagePassingPCLanguageModel(
            small_config(
                residual_error_upper_content_read_period=3,
                residual_error_upper_content_read_offset=3,
            )
        )
    with pytest.raises(ValueError, match="residual_phase_transport_lr_scale"):
        MessagePassingPCLanguageModel(
            small_config(residual_phase_transport_lr_scale=-0.1)
        )
    with pytest.raises(ValueError, match="residual_phase_transport_refine_passes"):
        MessagePassingPCLanguageModel(
            small_config(residual_phase_transport_refine_passes=-1)
        )
    with pytest.raises(ValueError, match="both be zero or positive"):
        MessagePassingPCLanguageModel(
            small_config(fast_attention_window=4)
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        MessagePassingPCLanguageModel(
            small_config(
                fast_attention_window=4,
                fast_attention_layers=3,
            )
        )
    with pytest.raises(ValueError, match="deployment settling requires"):
        MessagePassingPCLanguageModel(
            bidirectional_config(
                bidirectional_deployment_settling=True,
            )
        )
    with pytest.raises(ValueError, match="hierarchical_error_top_down_scale"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_top_down_scale=-0.1)
        )
    with pytest.raises(
        ValueError,
        match="hierarchical_error_context_credit_scale",
    ):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_context_credit_scale=-0.1)
        )
    with pytest.raises(
        ValueError,
        match="hierarchical_error_token_credit_scale",
    ):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_token_credit_scale=-0.1)
        )
    with pytest.raises(
        ValueError,
        match="requires factorized her_error_hierarchy",
    ):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_token_credit_scale=0.1)
        )
    with pytest.raises(ValueError, match="hierarchical_error_base_depth"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_base_depth=-1)
        )
    with pytest.raises(ValueError, match="hierarchical_error_base_depth"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_base_depth=2)
        )
    with pytest.raises(ValueError, match="hierarchical_error_binding_mode"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_binding_mode="unknown")
        )
    with pytest.raises(ValueError, match="prediction_mode"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_prediction_mode="unknown")
        )
    with pytest.raises(ValueError, match="requires factorized"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_prediction_mode="direct")
        )
    with pytest.raises(ValueError, match="positive recursive CP rank"):
        MessagePassingPCLanguageModel(
            small_config(
                inference_mode="her_error_hierarchy",
                hierarchical_error_binding_mode="factorized",
                hierarchical_error_prediction_mode="recursive_cp",
            )
        )
    with pytest.raises(ValueError, match="rank requires that mode"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_recursive_cp_rank=4)
        )
    with pytest.raises(ValueError, match="fixed-outcome"):
        MessagePassingPCLanguageModel(
            small_config(
                inference_mode="her_error_hierarchy",
                hierarchical_error_binding_mode="factorized",
                hierarchical_error_prediction_mode="recursive_cp",
                hierarchical_error_recursive_cp_rank=4,
            )
        )
    with pytest.raises(ValueError, match="recursive_cp_rank"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_recursive_cp_rank=-1)
        )
    with pytest.raises(ValueError, match="upper_lr_scale"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_recursive_cp_upper_lr_scale=-1.0)
        )
    with pytest.raises(ValueError, match="component_norm_cap"):
        MessagePassingPCLanguageModel(
            small_config(
                hierarchical_error_recursive_cp_component_norm_cap=-1.0
            )
        )
    with pytest.raises(ValueError, match="component norm cap requires"):
        MessagePassingPCLanguageModel(
            small_config(
                hierarchical_error_recursive_cp_component_norm_cap=0.1
            )
        )
    with pytest.raises(ValueError, match="hierarchical_error_target_mode"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_target_mode="unknown")
        )
    with pytest.raises(ValueError, match="outcome_code_scale"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_outcome_code_scale=-0.1)
        )
    with pytest.raises(ValueError, match="outcome_readout_scale"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_outcome_readout_scale=-0.1)
        )
    with pytest.raises(ValueError, match="fixed-outcome"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_outcome_readout_scale=1.0)
        )
    with pytest.raises(ValueError, match="hierarchical_error_context_mode"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_context_mode="unknown")
        )
    with pytest.raises(ValueError, match="hierarchical_error_gate_lr_scale"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_gate_lr_scale=-0.1)
        )
    with pytest.raises(ValueError, match="hierarchical_error_initial_write_rate"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_initial_write_rate=1.0)
        )
    with pytest.raises(ValueError, match="hierarchical_error_memory_scale"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_memory_scale=1.1)
        )
    with pytest.raises(ValueError, match="learning_start_step"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_learning_start_step=-1)
        )
    with pytest.raises(ValueError, match="level_start_interval"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_level_start_interval=-1)
        )
    with pytest.raises(ValueError, match="prediction_stream_window"):
        MessagePassingPCLanguageModel(
            small_config(prediction_stream_window=-1)
        )
    with pytest.raises(ValueError, match="sps_prediction_input_scale"):
        MessagePassingPCLanguageModel(
            small_config(sps_prediction_input_scale=-0.1)
        )
    with pytest.raises(ValueError, match="sps_future_state_credit_scale"):
        MessagePassingPCLanguageModel(
            small_config(sps_future_state_credit_scale=-0.1)
        )
    with pytest.raises(ValueError, match="sps_layer_token_credit_scale"):
        MessagePassingPCLanguageModel(
            small_config(sps_layer_token_credit_scale=-0.1)
        )
    with pytest.raises(ValueError, match="sps_future_observation_credit_scale"):
        MessagePassingPCLanguageModel(
            small_config(sps_future_observation_credit_scale=-0.1)
        )
    with pytest.raises(ValueError, match="require state separation"):
        MessagePassingPCLanguageModel(
            small_config(sps_future_state_credit_scale=1.0)
        )
    with pytest.raises(
        ValueError,
        match="state_prediction_separation requires factorized",
    ):
        MessagePassingPCLanguageModel(
            small_config(state_prediction_separation=True)
        )
    with pytest.raises(ValueError, match="requires her_error_hierarchy"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_context_mode="persistent")
        )
    with pytest.raises(ValueError, match="energy_backtracking_steps"):
        MessagePassingPCLanguageModel(small_config(energy_backtracking_steps=-1))
    with pytest.raises(ValueError, match="matrix_optimizer"):
        MessagePassingPCLanguageModel(small_config(matrix_optimizer="unknown"))
    with pytest.raises(ValueError, match="muon_lr_scale"):
        MessagePassingPCLanguageModel(small_config(muon_lr_scale=-1.0))
    with pytest.raises(ValueError, match="optimizer_warmup_steps"):
        MessagePassingPCLanguageModel(small_config(optimizer_warmup_steps=-1))
    with pytest.raises(ValueError, match="Muon momentum"):
        MessagePassingPCLanguageModel(small_config(muon_momentum=1.0))
    with pytest.raises(ValueError, match="Muon momentum"):
        MessagePassingPCLanguageModel(small_config(muon_momentum_final=1.0))
    with pytest.raises(ValueError, match="muon_momentum_warmup_steps"):
        MessagePassingPCLanguageModel(small_config(muon_momentum_warmup_steps=-1))
    with pytest.raises(ValueError, match="muon_ns_steps"):
        MessagePassingPCLanguageModel(small_config(muon_ns_steps=0))
    with pytest.raises(ValueError, match="schedule end"):
        MessagePassingPCLanguageModel(
            small_config(
                inference_lr_final=0.01,
                inference_lr_schedule_start_step=100,
                inference_lr_schedule_end_step=100,
            )
        )
    with pytest.raises(ValueError, match="inference_optimizer"):
        MessagePassingPCLanguageModel(
            small_config(inference_optimizer="unknown")
        )
    with pytest.raises(ValueError, match="inference_momentum"):
        MessagePassingPCLanguageModel(
            small_config(inference_momentum=1.0)
        )
    with pytest.raises(ValueError, match="spiking_precision_scale"):
        MessagePassingPCLanguageModel(
            small_config(spiking_precision_scale=-0.1)
        )
    with pytest.raises(ValueError, match="spiking precision requires gradient"):
        MessagePassingPCLanguageModel(
            small_config(
                inference_optimizer="heavy_ball",
                energy_backtracking_steps=4,
                spiking_precision_scale=1.0,
            )
        )
    with pytest.raises(ValueError, match="forward_update_top_layers"):
        MessagePassingPCLanguageModel(
            small_config(forward_update_top_layers=3)
        )
    with pytest.raises(ValueError, match="requires settled"):
        MessagePassingPCLanguageModel(
            small_config(
                learning_phase_source="free",
                forward_update_top_layers=1,
            )
        )
    with pytest.raises(ValueError, match="requires energy backtracking"):
        MessagePassingPCLanguageModel(
            small_config(inference_optimizer="heavy_ball")
        )
    with pytest.raises(ValueError, match="exact energy gradients"):
        MessagePassingPCLanguageModel(
            small_config(
                inference_optimizer="conjugate_gradient",
                learned_feedback_proposal_scale=1.0,
                energy_backtracking_steps=4,
            )
        )


def test_recursive_cp_launchers_use_fixed_outcome_target() -> None:
    stale_launchers: list[str] = []
    for script in Path("tools").glob("*.ps1"):
        text = script.read_text()
        if "recursive_cp" not in text:
            continue
        if (
            '-HierarchicalErrorTargetMode "all_tokens"' in text
            or "--message-pc-hierarchical-error-target-mode all_tokens" in text
        ):
            stale_launchers.append(str(script))

    assert stale_launchers == []


def test_recursive_cp_resume_launchers_reset_her_predictors() -> None:
    stale_resumes: list[str] = []
    for script in Path("tools").glob("resume_*recursive_cp*.ps1"):
        text = script.read_text()
        if "recursive_cp" in text and "-ResumeResetHerPredictors" not in text:
            stale_resumes.append(str(script))

    assert stale_resumes == []


@pytest.mark.parametrize(
    "block_state_mode",
    ["squashed_state", "identity_residual", "nanochat_residual"],
)
@pytest.mark.parametrize("ffn_dim", [0, 48])
def test_message_pc_forward_is_causal(block_state_mode: str, ffn_dim: int) -> None:
    torch.manual_seed(1)
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode=block_state_mode,
            ffn_dim=ffn_dim,
        )
    )
    original = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    changed = original.clone()
    changed[:, 5:] = torch.tensor([[20, 21, 22]])

    original_logits = model(original)
    changed_logits = model(changed)

    assert torch.equal(original_logits[:, :5], changed_logits[:, :5])
    assert not torch.equal(original_logits[:, 5:], changed_logits[:, 5:])


def test_message_pc_her_binding_is_reversible_and_top_down_is_recursive() -> None:
    torch.manual_seed(41)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_top_down_scale=0.5,
            state_delta_rms_cap=0.0,
        )
    )
    context = torch.randn(2, 8, 32)
    error = torch.randn_like(context)
    assert torch.equal(
        model._hierarchical_error_unbind(
            context,
            model._hierarchical_error_bind(context, error),
        ),
        error,
    )

    for weight in model.generative_weights:
        weight.normal_(mean=0.0, std=0.01)
    x = torch.randint(0, 31, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)
    deployed, predictions, modulated, correction = (
        model._hierarchical_error_pass(states)
    )
    expected_top = predictions[1]
    expected_low = predictions[0] + 0.5 * model._hierarchical_error_unbind(
        states[1],
        expected_top,
    )
    expected_correction = model._hierarchical_error_unbind(
        states[0],
        expected_low,
    )

    assert torch.allclose(modulated[1], expected_top)
    assert torch.allclose(modulated[0], expected_low)
    assert torch.allclose(correction, expected_correction)
    assert torch.allclose(deployed, states[0] + 0.5 * expected_correction)


def test_message_pc_her_identity_binding_passes_residual_unchanged() -> None:
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="identity",
        )
    )
    context = torch.randn(2, 8, 32)
    error = torch.randn_like(context)

    assert torch.equal(model._hierarchical_error_bind(context, error), error)
    assert torch.equal(model._hierarchical_error_unbind(context, error), error)


def test_message_pc_her_unitary_binding_is_reversible_and_contextual() -> None:
    torch.manual_seed(49)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="unitary",
        )
    )
    context = torch.randn(2, 8, 32)
    other_context = torch.roll(context, shifts=1, dims=-1)
    error = torch.randn_like(context)

    bound = model._hierarchical_error_bind(context, error)
    recovered = model._hierarchical_error_unbind(context, bound)
    wrongly_recovered = model._hierarchical_error_unbind(other_context, bound)

    assert torch.allclose(recovered, error, atol=2e-6, rtol=2e-6)
    assert not torch.allclose(wrongly_recovered, error, atol=1e-3, rtol=1e-3)


def test_message_pc_her_factorized_paths_skip_to_base_and_sum() -> None:
    torch.manual_seed(50)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_top_down_scale=0.5,
            state_delta_rms_cap=0.0,
        )
    )
    assert all(
        torch.count_nonzero(weight).item() == 0
        for weight in model.feedback_weights
    )
    for weight in model.generative_weights:
        weight.normal_(mean=0.0, std=0.01)
    x = torch.randint(0, 31, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)

    deployed, predictions, modulated, correction = (
        model._hierarchical_error_pass(states)
    )
    expected = torch.stack(predictions).sum(dim=0)

    assert torch.allclose(modulated[0], expected)
    assert torch.allclose(correction, expected)
    assert torch.allclose(deployed, states[0] + 0.5 * expected)


def test_message_pc_her_outer_product_contract_validates_scope() -> None:
    with pytest.raises(ValueError, match="requires factorized"):
        MessagePassingPCLanguageModel(
            small_config(hierarchical_error_outer_product_contract=True)
        )
    with pytest.raises(ValueError, match="exactly one"):
        MessagePassingPCLanguageModel(
            small_config(
                inference_mode="her_error_hierarchy",
                hierarchical_error_binding_mode="factorized",
                hierarchical_error_outer_product_contract=True,
            )
        )
    with pytest.raises(ValueError, match="unit top-down scale"):
        MessagePassingPCLanguageModel(
            small_config(
                inference_mode="her_error_hierarchy",
                hierarchical_error_binding_mode="factorized",
                hierarchical_error_base_depth=1,
                hierarchical_error_top_down_scale=0.5,
                hierarchical_error_outer_product_contract=True,
            )
        )


def test_message_pc_her_outer_product_contract_modulates_lower_prediction() -> None:
    torch.manual_seed(70)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_base_depth=1,
            hierarchical_error_outer_product_contract=True,
            state_delta_rms_cap=0.0,
        )
    )
    assert torch.equal(model.feedback_weights[1], torch.eye(32))
    assert torch.count_nonzero(model.generative_weights[1]) == 0
    model.generative_weights[1].copy_(torch.eye(32))
    base = torch.randn(2, 8, 32)
    upper = torch.randn(2, 8, 32)
    states = [torch.zeros_like(base), base, upper]

    deployed, predictions, _, correction, keys, values, scores = (
        model._hierarchical_error_factorized_pass(states)
    )
    normalized_base = base * torch.rsqrt(
        base.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    normalized_upper = upper * torch.rsqrt(
        upper.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    expected_score = (normalized_base * normalized_upper).mean(
        dim=-1,
        keepdim=True,
    )
    expected = expected_score * normalized_upper

    assert scores[0].shape == (2, 8, 1)
    assert torch.allclose(keys[0], normalized_upper)
    assert torch.allclose(values[0], normalized_upper)
    assert torch.allclose(predictions[0], expected)
    assert torch.allclose(correction, expected)
    assert torch.allclose(deployed, base + expected)


def test_message_pc_her_outer_product_signals_match_local_autograd() -> None:
    torch.manual_seed(71)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_base_depth=1,
            hierarchical_error_outer_product_contract=True,
        )
    )
    context = torch.randn(2, 4, 32, requires_grad=True)
    key = torch.randn(2, 4, 32, requires_grad=True)
    value = torch.randn(2, 4, 32, requires_grad=True)
    target = torch.randn(2, 4, 32)
    normalized_context = context * torch.rsqrt(
        context.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    prediction = (
        (normalized_context * key).mean(dim=-1, keepdim=True) * value
    )
    predicted_matrix = key.unsqueeze(-1) * value.unsqueeze(-2)
    target_matrix = normalized_context.detach().unsqueeze(-1) * target.unsqueeze(-2)
    matrix_objective = (
        0.5
        * predicted_matrix.sub(target_matrix).square().sum()
        / model.config.hidden_dim
    )
    expected_key, expected_value = torch.autograd.grad(
        matrix_objective,
        (key, value),
        retain_graph=True,
    )
    output_objective = 0.5 * (prediction - target).square().sum()
    (expected_context,) = torch.autograd.grad(output_objective, (context,))

    (
        predictor_error,
        value_signal,
        key_signal,
        _,
        context_gradient,
    ) = model._hierarchical_error_factorized_signals(
        context_state=context.detach(),
        prediction=prediction.detach(),
        key=key.detach(),
        value=value.detach(),
        gate=(
            normalized_context.detach() * key.detach()
        ).mean(dim=-1, keepdim=True),
        target=target,
    )

    assert torch.allclose(predictor_error, prediction.detach() - target)
    assert torch.allclose(key_signal, expected_key, atol=2e-6, rtol=2e-6)
    assert torch.allclose(value_signal, expected_value, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        context_gradient,
        expected_context,
        atol=2e-6,
        rtol=2e-6,
    )


def test_message_pc_her_outer_product_contract_learns_without_backprop() -> None:
    torch.manual_seed(72)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_base_depth=1,
            hierarchical_error_outer_product_contract=True,
            route_lr_scale=0.0,
            matrix_optimizer="adam",
        )
    )
    value_before = model.generative_weights[1].clone()
    key_before = model.feedback_weights[1].clone()
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31

    for _ in range(2):
        model.local_update(
            x,
            y,
            decoder_lr=0.0,
            state_lr=1e-3,
            embed_lr=0.0,
            weight_decay=0.0,
        )

    assert not torch.equal(model.generative_weights[1], value_before)
    assert not torch.equal(model.feedback_weights[1], key_before)
    assert all(parameter.grad is None for parameter in model.parameters())
    diagnostics = model._last_update_diagnostics
    assert diagnostics[
        "update_diag_message_pc_uses_her_outer_product_contract"
    ] == 1.0
    assert diagnostics["update_diag_message_pc_generative_update_rms"] > 0.0


def test_message_pc_sps_attention_mask_separates_persistent_and_ephemeral() -> None:
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            state_prediction_separation=True,
            prediction_stream_window=0,
        )
    )

    allowed = ~model._causal_mask(6, torch.device("cpu"))

    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(allowed, expected)


def test_message_pc_sps_prediction_window_is_bounded() -> None:
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            state_prediction_separation=True,
            prediction_stream_window=1,
        )
    )

    allowed = ~model._causal_mask(6, torch.device("cpu"))

    assert allowed[2, 1]
    assert allowed[3, 1]
    assert allowed[4, 3]
    assert allowed[5, 3]
    assert not allowed[4, 1]
    assert not allowed[5, 1]


def test_message_pc_sps_keeps_prediction_state_ephemeral() -> None:
    torch.manual_seed(64)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_base_depth=1,
            state_prediction_separation=True,
            prediction_stream_window=0,
            block_state_mode="identity_residual",
        )
    )
    x = torch.randint(0, 31, (2, 8))
    before, _ = model._forward_states(x, with_cache=False)
    with torch.no_grad():
        assert model.prediction_token is not None
        model.prediction_token.add_(3.0 * torch.randn_like(model.prediction_token))
    after, _ = model._forward_states(x, with_cache=False)

    assert before[0].shape == (2, 16, 32)
    assert model(x).shape == (2, 8, 32)
    for before_layer, after_layer in zip(before, after):
        assert torch.allclose(
            before_layer[:, 0::2],
            after_layer[:, 0::2],
            atol=1e-6,
            rtol=1e-6,
        )
    assert not torch.allclose(before[-1][:, 1::2], after[-1][:, 1::2])


def test_message_pc_sps_prediction_input_bridge_preserves_stream_roles() -> None:
    torch.manual_seed(66)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            state_prediction_separation=True,
            sps_prediction_input_scale=1.0,
            block_state_mode="identity_residual",
        )
    )
    x = torch.randint(0, 31, (2, 8))
    changed = x.clone()
    changed[:, 3] = (changed[:, 3] + 1) % 31

    states, _ = model._forward_states(x, with_cache=False)
    changed_states, _ = model._forward_states(changed, with_cache=False)

    assert torch.equal(
        states[0][:, 0::2][:, :3],
        changed_states[0][:, 0::2][:, :3],
    )
    assert not torch.equal(
        states[0][:, 1::2][:, 3],
        changed_states[0][:, 1::2][:, 3],
    )
    assert model._causal_mask(16, torch.device("cpu"))[6, 7]


def test_message_pc_sps_node_residual_hierarchy_validates_contract() -> None:
    with pytest.raises(ValueError, match="require state separation"):
        MessagePassingPCLanguageModel(
            small_config(sps_node_residual_hierarchy=True)
        )
    common = {
        "inference_mode": "her_error_hierarchy",
        "hierarchical_error_binding_mode": "factorized",
        "state_prediction_separation": True,
        "sps_node_residual_hierarchy": True,
    }
    with pytest.raises(ValueError, match="base depth"):
        MessagePassingPCLanguageModel(small_config(**common))
    with pytest.raises(ValueError, match="top-down scale 1"):
        MessagePassingPCLanguageModel(
            small_config(
                **common,
                hierarchical_error_base_depth=1,
                hierarchical_error_top_down_scale=0.5,
            )
        )
    with pytest.raises(ValueError, match="replaces auxiliary"):
        MessagePassingPCLanguageModel(
            small_config(
                **common,
                hierarchical_error_base_depth=1,
                sps_layer_token_credit_scale=1.0,
            )
        )


def test_message_pc_sps_node_residuals_telescope_to_top_prediction() -> None:
    torch.manual_seed(73)
    model = MessagePassingPCLanguageModel(
        small_config(
            num_layers=3,
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_base_depth=1,
            hierarchical_error_context_mode="instant",
            state_prediction_separation=True,
            sps_prediction_input_scale=1.0,
            sps_node_residual_hierarchy=True,
            block_state_mode="identity_residual",
        )
    )
    x = torch.randint(0, 31, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)
    deployed, predictions, _, correction, _ = (
        model._node_residual_hierarchy_pass(states)
    )

    assert torch.allclose(deployed[:, 1::2], states[-1][:, 1::2])
    assert torch.equal(deployed[:, 0::2], states[1][:, 0::2])
    assert torch.allclose(
        correction[:, 1::2],
        states[-1][:, 1::2] - states[1][:, 1::2],
    )
    assert all(torch.count_nonzero(value[:, 0::2]) == 0 for value in predictions)

    before = model(x)
    with torch.no_grad():
        for weight in (*model.generative_weights, *model.feedback_weights):
            weight.normal_(mean=0.0, std=10.0)
    assert torch.equal(before, model(x))


def test_message_pc_sps_node_residual_levels_reduce_constructed_residual() -> None:
    torch.manual_seed(76)
    model = MessagePassingPCLanguageModel(
        small_config(
            num_layers=4,
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_base_depth=1,
            hierarchical_error_context_mode="instant",
            state_prediction_separation=True,
            sps_prediction_input_scale=1.0,
            sps_node_residual_hierarchy=True,
            block_state_mode="identity_residual",
        )
    )
    states = [torch.zeros(1, 8, 32) for _ in range(5)]
    increments = []
    for index, scale in enumerate((3.0, 2.0, 1.0)):
        increment = torch.zeros(1, 4, 32)
        increment[..., index] = scale
        increments.append(increment)
    cumulative = torch.zeros_like(increments[0])
    states[1][:, 1::2] = cumulative
    for offset, increment in enumerate(increments):
        cumulative = cumulative + increment
        states[offset + 2][:, 1::2] = cumulative

    _, predictions, _, correction, _ = model._node_residual_hierarchy_pass(states)
    remaining = cumulative.clone()
    previous_energy = remaining.square().sum()
    for prediction, increment in zip(predictions, increments):
        assert torch.allclose(prediction[:, 1::2], increment)
        remaining = remaining - prediction[:, 1::2]
        energy = remaining.square().sum()
        assert energy < previous_energy
        previous_energy = energy

    assert torch.allclose(correction[:, 1::2], cumulative)
    assert previous_energy.item() == pytest.approx(0.0, abs=1e-5)


def test_message_pc_sps_node_residual_memory_vjp_matches_autograd() -> None:
    torch.manual_seed(74)
    model = MessagePassingPCLanguageModel(
        small_config(
            num_layers=3,
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_base_depth=1,
            hierarchical_error_context_mode="persistent",
            hierarchical_error_memory_scale=0.6,
            state_prediction_separation=True,
            sps_prediction_input_scale=1.0,
            sps_node_residual_hierarchy=True,
            block_state_mode="identity_residual",
        )
    )
    x = torch.randint(0, 31, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)
    states = [state.detach().clone() for state in states]
    states[2].requires_grad_()
    model.hierarchical_gate_weights[0].requires_grad_()
    cache = model._node_residual_hierarchy_cache(states)
    output_signal = torch.randn_like(cache.predictions[0])
    objective = (cache.predictions[0] * output_signal).sum()
    expected, expected_gate = torch.autograd.grad(
        objective,
        (states[2], model.hierarchical_gate_weights[0]),
    )

    candidate_signal, gate_score_signal = model._node_residual_memory_vjp(
        cache,
        0,
        output_signal,
    )

    assert torch.allclose(
        candidate_signal,
        expected[:, 1::2],
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.count_nonzero(expected[:, 0::2]) == 0
    actual_gate = (
        gate_score_signal * cache.gate_features[0]
    ).sum(dim=(0, 1))
    assert torch.allclose(actual_gate, expected_gate, atol=2e-5, rtol=2e-5)
    model.hierarchical_gate_weights[0].requires_grad_(False)


def test_message_pc_sps_node_residual_hierarchy_learns_without_side_heads() -> None:
    torch.manual_seed(75)
    model = MessagePassingPCLanguageModel(
        small_config(
            num_layers=3,
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_base_depth=1,
            hierarchical_error_context_mode="persistent",
            hierarchical_error_memory_scale=0.2,
            state_prediction_separation=True,
            sps_prediction_input_scale=1.0,
            sps_node_residual_hierarchy=True,
            block_state_mode="identity_residual",
            matrix_optimizer="adam",
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    side_before = [
        weight.clone()
        for weight in (*model.generative_weights, *model.feedback_weights)
    ]
    gates_before = [weight.clone() for weight in model.hierarchical_gate_weights]

    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        weight_decay=0.0,
    )

    side_after = [*model.generative_weights, *model.feedback_weights]
    assert all(
        torch.equal(before, after)
        for before, after in zip(side_before, side_after)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(gates_before, model.hierarchical_gate_weights)
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    diagnostics = model._last_update_diagnostics
    assert diagnostics[
        "update_diag_message_pc_uses_sps_node_residual_hierarchy"
    ] == 1.0
    assert diagnostics["update_diag_message_pc_sps_node_candidate_credit_rms"] > 0
    assert diagnostics["update_diag_message_pc_her_gate_update_rms"] > 0
    assert diagnostics["update_diag_message_pc_energy_reduction"] > 0
    for level in range(1, 3):
        prefix = f"update_diag_message_pc_sps_node_level_{level}"
        assert diagnostics[f"{prefix}_target_rms"] > 0
        assert diagnostics[f"{prefix}_prediction_rms"] > 0
        assert math.isfinite(diagnostics[f"{prefix}_prediction_target_cosine"])
        assert math.isfinite(
            diagnostics[f"{prefix}_least_squares_prediction_scale"]
        )


def test_message_pc_sps_future_credit_routes_through_persistent_slots() -> None:
    torch.manual_seed(67)
    base = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_base_depth=1,
            state_prediction_separation=True,
            prediction_stream_window=0,
            block_state_mode="identity_residual",
        )
    )
    future = copy.deepcopy(base)
    future.config = replace(future.config, sps_future_state_credit_scale=1.0)
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    free_states, _ = base._forward_states(x, with_cache=False)

    base_settled, base_diag = base._settle_hierarchical_factorized(
        free_states,
        y,
        inference_lr=0.05,
    )
    future_settled, future_diag = future._settle_hierarchical_factorized(
        free_states,
        y,
        inference_lr=0.05,
    )

    assert base_diag["update_diag_message_pc_sps_future_state_credit_rms"] == 0
    assert future_diag["update_diag_message_pc_sps_future_state_credit_rms"] > 0
    assert not torch.allclose(
        base_settled[1][:, 0::2],
        future_settled[1][:, 0::2],
    )
    assert torch.equal(
        future._persistent_state_only(free_states[1])[:, 1::2],
        torch.zeros_like(free_states[1][:, 1::2]),
    )


def test_message_pc_sps_layer_token_credit_settles_prediction_nodes() -> None:
    torch.manual_seed(70)
    base = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_base_depth=1,
            state_prediction_separation=True,
            prediction_stream_window=0,
            block_state_mode="identity_residual",
        )
    )
    layered = copy.deepcopy(base)
    layered.config = replace(layered.config, sps_layer_token_credit_scale=1.0)
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    free_states, _ = base._forward_states(x, with_cache=False)

    base_settled, base_diag = base._settle_hierarchical_factorized(
        free_states,
        y,
        inference_lr=0.05,
    )
    layered_settled, layered_diag = layered._settle_hierarchical_factorized(
        free_states,
        y,
        inference_lr=0.05,
    )

    assert base_diag["update_diag_message_pc_sps_layer_token_credit_rms"] == 0
    assert layered_diag["update_diag_message_pc_sps_layer_token_credit_rms"] > 0
    assert not torch.allclose(
        base_settled[-1][:, 1::2],
        layered_settled[-1][:, 1::2],
    )


def test_message_pc_sps_future_observation_is_train_only_and_causal() -> None:
    torch.manual_seed(72)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_base_depth=1,
            state_prediction_separation=True,
            prediction_stream_window=0,
            sps_prediction_input_scale=1.0,
            sps_future_observation_credit_scale=1.0,
            block_state_mode="identity_residual",
        )
    )
    x = torch.randint(0, 31, (2, 8))
    changed = x.clone()
    changed[:, 3] = (changed[:, 3] + 1) % 31
    y = (x + 1) % 31
    free_states, _ = model._forward_states(x, with_cache=False)
    changed_free_states, _ = model._forward_states(changed, with_cache=False)

    assert torch.allclose(
        free_states[1][:, 5],
        changed_free_states[1][:, 5],
        atol=1e-6,
        rtol=1e-6,
    )
    settled, diagnostics = model._settle_hierarchical_factorized(
        free_states,
        y,
        inference_lr=0.05,
    )
    changed_settled, _ = model._settle_hierarchical_factorized(
        changed_free_states,
        y,
        inference_lr=0.05,
    )

    assert diagnostics[
        "update_diag_message_pc_sps_future_observation_credit_rms"
    ] > 0
    assert not torch.allclose(
        settled[1][:, 5],
        changed_settled[1][:, 5],
    )
    assert torch.allclose(
        model(x)[:, 2],
        model(changed)[:, 2],
        atol=1e-6,
        rtol=1e-6,
    )


def test_message_pc_sps_future_credit_is_temporal_and_causal() -> None:
    torch.manual_seed(68)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            state_prediction_separation=True,
            prediction_stream_window=0,
            block_state_mode="identity_residual",
        )
    )
    x = torch.randint(0, 31, (1, 8))
    states, _ = model._forward_states(x, with_cache=False)
    cache = model._block(1, states[1])
    output_signal = torch.zeros_like(states[2])
    output_signal[:, 5] = torch.randn_like(output_signal[:, 5])

    future_credit = model._persistent_state_only(
        model._block_input_signal(1, states[1], cache, output_signal)
    )

    assert torch.count_nonzero(future_credit[:, 0:6:2]) > 0
    assert torch.count_nonzero(future_credit[:, 6::2]) == 0
    assert torch.count_nonzero(future_credit[:, 1::2]) == 0


@pytest.mark.parametrize("block_state_mode", ["identity_residual", "nanochat_residual"])
def test_message_pc_sps_prediction_input_embedding_vjp_matches_finite_difference(
    block_state_mode: str,
) -> None:
    torch.manual_seed(69)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            state_prediction_separation=True,
            sps_prediction_input_scale=0.7,
            block_state_mode=block_state_mode,
        )
    ).double()
    x = torch.tensor([[1, 2, 3]])
    free_state = model._forward_states(x, with_cache=False)[0][0]
    output_signal = torch.randn_like(free_state)
    analytic = model._embedding_state_vjp(x, free_state, output_signal)

    def objective() -> float:
        state = model._forward_states(x, with_cache=False)[0][0]
        return float((state * output_signal).sum().item())

    epsilon = 1e-6
    with torch.no_grad():
        model.embedding.weight[2, 7] += epsilon
    plus = objective()
    with torch.no_grad():
        model.embedding.weight[2, 7] -= 2.0 * epsilon
    minus = objective()
    with torch.no_grad():
        model.embedding.weight[2, 7] += epsilon

    numerical = (plus - minus) / (2.0 * epsilon)
    assert analytic[0, 1, 7].item() == pytest.approx(
        numerical,
        rel=3e-4,
        abs=1e-8,
    )


def test_message_pc_sps_routes_local_credit_into_both_stream_inputs() -> None:
    torch.manual_seed(65)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_base_depth=1,
            hierarchical_error_context_mode="persistent_hard",
            hierarchical_error_memory_scale=0.1,
            state_prediction_separation=True,
            prediction_stream_window=0,
            sps_prediction_input_scale=1.0,
            sps_layer_token_credit_scale=1.0,
            sps_future_state_credit_scale=1.0,
            sps_future_observation_credit_scale=1.0,
            block_state_mode="identity_residual",
            matrix_optimizer="adam",
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    embedding_before = model.embedding.weight.clone()
    assert model.prediction_token is not None
    prediction_token_before = model.prediction_token.clone()

    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=0.0,
        embed_lr=1e-3,
        weight_decay=0.0,
    )

    assert not torch.equal(model.embedding.weight, embedding_before)
    assert not torch.equal(model.prediction_token, prediction_token_before)
    diagnostics = model._last_update_diagnostics
    assert diagnostics[
        "update_diag_message_pc_uses_state_prediction_separation"
    ] == 1.0
    assert diagnostics["update_diag_message_pc_sps_state_stream_delta_rms"] > 0
    assert diagnostics[
        "update_diag_message_pc_sps_prediction_stream_delta_rms"
    ] > 0
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_her_factorized_path_accepts_persistent_gate_context() -> None:
    torch.manual_seed(59)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_context_mode="persistent_hard",
            hierarchical_error_memory_scale=1.0,
        )
    )
    for weight in (*model.generative_weights, *model.feedback_weights):
        weight.normal_(mean=0.0, std=0.01)
    x = torch.randint(0, 31, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)

    persistent = model._hierarchical_error_factorized_pass(states)[1]
    instant = model._hierarchical_error_factorized_pass(
        states,
        contexts=[states[layer] for layer in range(model.config.num_layers)],
    )[1]

    assert torch.allclose(persistent[0][:, 0], instant[0][:, 0])
    assert any(
        not torch.allclose(persistent[level][:, 1:], instant[level][:, 1:])
        for level in range(model.config.num_layers)
    )


def test_message_pc_her_factorized_persistent_gate_learns_locally() -> None:
    torch.manual_seed(60)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_context_mode="persistent_hard",
            hierarchical_error_memory_scale=0.1,
            route_lr_scale=0.0,
            matrix_optimizer="adam",
        )
    )
    for weight in (*model.generative_weights, *model.feedback_weights):
        weight.normal_(mean=0.0, std=0.01)
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    gate_before = [weight.clone() for weight in model.hierarchical_gate_weights]

    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=1e-3,
        embed_lr=0.0,
        weight_decay=0.0,
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(gate_before, model.hierarchical_gate_weights)
    )
    diagnostics = model._last_update_diagnostics
    assert diagnostics["update_diag_message_pc_her_uses_persistent_context"] == 1.0
    assert diagnostics["update_diag_message_pc_her_gate_update_rms"] > 0.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_her_factorized_local_rule_updates_value_and_key() -> None:
    torch.manual_seed(51)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_base_depth=1,
            hierarchical_error_context_credit_scale=0.0,
            route_lr_scale=0.0,
            matrix_optimizer="adam",
        )
    )
    model.generative_weights[1].normal_(mean=0.0, std=0.01)
    value_before = model.generative_weights[1].clone()
    key_before = model.feedback_weights[1].clone()
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31

    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=1e-4,
        embed_lr=0.0,
        weight_decay=0.0,
    )

    assert not torch.equal(model.generative_weights[1], value_before)
    assert not torch.equal(model.feedback_weights[1], key_before)
    diagnostics = model._last_update_diagnostics
    assert diagnostics["update_diag_message_pc_her_settle_steps"] == float(
        model.config.inference_steps
    )
    assert diagnostics["update_diag_message_pc_phase_delta_rms"] > 0.0
    assert math.isfinite(diagnostics["update_diag_message_pc_energy_before"])
    assert math.isfinite(diagnostics["update_diag_message_pc_energy_after"])
    assert model.active_parameter_count() > MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="identity",
            hierarchical_error_base_depth=1,
            route_lr_scale=0.0,
        )
    ).active_parameter_count()


def test_message_pc_her_factorized_token_credit_matches_local_autograd() -> None:
    torch.manual_seed(61)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
        )
    )
    context = torch.randn(2, 4, 32, requires_grad=True)
    key = torch.randn(2, 4, 32, requires_grad=True)
    value = torch.randn(2, 4, 32, requires_grad=True)
    target = torch.randn(2, 4, 32)
    extra_prediction_gradient = torch.randn(2, 4, 32) * 0.2
    normalized_context = context * torch.rsqrt(
        context.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    context_heads = model._split_heads(normalized_context)
    key_heads = model._split_heads(key)
    gate = 2.0 * torch.sigmoid(
        (context_heads * key_heads).sum(dim=-1, keepdim=True)
        / math.sqrt(model.head_dim)
    )
    prediction = model._merge_heads(gate * model._split_heads(value))
    objective = 0.5 * (prediction - target).square().sum()
    objective = objective + (
        prediction * extra_prediction_gradient
    ).sum()
    expected_value, expected_key, expected_context = torch.autograd.grad(
        objective,
        (value, key, context),
    )

    (
        predictor_error,
        value_signal,
        key_signal,
        _,
        context_gradient,
    ) = model._hierarchical_error_factorized_signals(
        context_state=context.detach(),
        prediction=prediction.detach(),
        key=key.detach(),
        value=value.detach(),
        gate=gate.detach(),
        target=target,
        extra_prediction_gradient=extra_prediction_gradient,
    )

    assert torch.allclose(predictor_error, prediction.detach() - target)
    assert torch.allclose(value_signal, expected_value, atol=2e-6, rtol=2e-6)
    assert torch.allclose(key_signal, expected_key, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        context_gradient,
        expected_context,
        atol=2e-6,
        rtol=2e-6,
    )


def test_message_pc_her_factorized_token_credit_is_local_and_reported() -> None:
    torch.manual_seed(62)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_context_mode="persistent_hard",
            hierarchical_error_memory_scale=0.1,
            hierarchical_error_token_credit_scale=0.25,
            matrix_optimizer="adam",
        )
    )
    for weight in (*model.generative_weights, *model.feedback_weights):
        weight.normal_(mean=0.0, std=0.01)
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31

    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=1e-3,
        embed_lr=0.0,
        weight_decay=0.0,
    )

    diagnostics = model._last_update_diagnostics
    assert diagnostics["update_diag_message_pc_her_token_credit_scale"] == 0.25
    assert diagnostics["update_diag_message_pc_her_token_credit_gradient_rms"] > 0
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_her_mistake_target_masks_correct_tokens() -> None:
    torch.manual_seed(52)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_target_mode="mistakes",
            hierarchical_error_context_credit_scale=0.0,
            route_lr_scale=0.0,
            matrix_optimizer="adam",
        )
    )
    model.decoder_weight.zero_()
    model.decoder_bias.zero_()
    model.decoder_bias[3] = 100.0
    x = torch.randint(0, 31, (2, 8))
    y = torch.full_like(x, 3)

    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=1e-4,
        embed_lr=0.0,
        weight_decay=0.0,
    )

    assert (
        model._last_update_diagnostics[
            "update_diag_message_pc_her_target_active_fraction"
        ]
        == 0.0
    )


def test_message_pc_her_fixed_outcome_target_uses_stable_codebook() -> None:
    torch.manual_seed(73)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_target_mode="fixed_outcome",
            hierarchical_error_outcome_code_scale=0.2,
        )
    )
    state = torch.randn(2, 8, 32)
    y = torch.randint(0, 31, (2, 8))
    target, _, _ = model._hierarchical_error_target(state, y)
    normalized_codebook = model.feedback_embedding * torch.rsqrt(
        model.feedback_embedding.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    expected = F.embedding(y, normalized_codebook) * 0.2
    model.decoder_weight.normal_(mean=0.0, std=3.0)
    changed_decoder_target, _, _ = model._hierarchical_error_target(state, y)

    assert torch.allclose(target, expected)
    assert torch.equal(target, changed_decoder_target)
    assert target.float().square().mean().sqrt().item() == pytest.approx(
        0.2,
        abs=5e-4,
    )


def test_message_pc_her_fixed_outcome_learns_without_moving_codebook() -> None:
    torch.manual_seed(74)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_base_depth=1,
            hierarchical_error_target_mode="fixed_outcome",
            hierarchical_error_outcome_code_scale=0.2,
            hierarchical_error_outcome_readout_scale=2.0,
            route_lr_scale=0.0,
            matrix_optimizer="adam",
        )
    )
    codebook_before = model.feedback_embedding.clone()
    predictor_before = model.generative_weights[1].clone()
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31

    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        weight_decay=0.0,
    )

    assert torch.equal(model.feedback_embedding, codebook_before)
    assert not torch.equal(model.generative_weights[1], predictor_before)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model._last_update_diagnostics[
        "update_diag_message_pc_her_outcome_code_scale"
    ] == pytest.approx(0.2)
    assert model._last_update_diagnostics[
        "update_diag_message_pc_her_outcome_readout_gradient_rms"
    ] > 0


def test_message_pc_her_fixed_outcome_readout_gradient_matches_autograd() -> None:
    torch.manual_seed(75)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_target_mode="fixed_outcome",
            hierarchical_error_outcome_code_scale=0.2,
            hierarchical_error_outcome_readout_scale=3.0,
        )
    )
    correction = torch.randn(2, 8, 32, requires_grad=True)
    base_logits = torch.randn(2, 8, 32)
    y = torch.randint(0, 31, (2, 8))
    logits = base_logits + model._hierarchical_error_outcome_logits(correction)
    objective = F.cross_entropy(
        logits.reshape(-1, 32),
        y.reshape(-1),
        reduction="sum",
    )
    (expected,) = torch.autograd.grad(objective, (correction,))
    probabilities = F.softmax(logits.detach(), dim=-1)
    actual = model._hierarchical_error_outcome_gradient(probabilities, y)

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_message_pc_her_fixed_outcome_readout_is_deployed() -> None:
    torch.manual_seed(76)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_target_mode="fixed_outcome",
            hierarchical_error_outcome_code_scale=0.2,
            hierarchical_error_outcome_readout_scale=2.0,
            state_delta_rms_cap=0.0,
        )
    )
    model.generative_weights[0].normal_(mean=0.0, std=0.01)
    x = torch.randint(0, 31, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)
    deployed, _, _, correction = model._hierarchical_error_pass(states)
    expected = model._decoder_logits(
        deployed
    ) + model._hierarchical_error_outcome_logits(correction)

    assert torch.allclose(model(x), expected)


def test_message_pc_her_base_depth_keeps_lower_predictor_and_specializes_upper() -> None:
    torch.manual_seed(45)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_base_depth=1,
            hierarchical_error_top_down_scale=0.5,
            state_delta_rms_cap=0.0,
        )
    )
    model.generative_weights[0].normal_(mean=0.0, std=10.0)
    model.generative_weights[1].normal_(mean=0.0, std=0.01)
    x = torch.randint(0, 31, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)
    deployed, predictions, modulated, correction = (
        model._hierarchical_error_pass(states)
    )
    normalized = states[2] * torch.rsqrt(
        states[2].square().mean(dim=-1, keepdim=True) + 1e-6
    )
    expected_prediction = F.linear(normalized, model.generative_weights[1])
    expected_correction = model._hierarchical_error_unbind(
        states[1],
        expected_prediction,
    )

    assert len(predictions) == 1
    assert torch.allclose(predictions[0], expected_prediction, atol=1e-5)
    assert torch.equal(modulated[0], predictions[0])
    assert torch.allclose(correction, expected_correction, atol=1e-5)
    assert torch.allclose(deployed, states[1] + 0.5 * expected_correction)


@pytest.mark.parametrize("context_mode", ["instant", "persistent_hard"])
def test_message_pc_her_forward_is_causal(context_mode: str) -> None:
    torch.manual_seed(42)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_context_mode=context_mode,
        )
    )
    for weight in model.generative_weights:
        weight.normal_(mean=0.0, std=0.01)
    original = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    changed = original.clone()
    changed[:, 5:] = torch.tensor([[20, 21, 22]])

    original_logits = model(original)
    changed_logits = model(changed)

    assert torch.equal(original_logits[:, :5], changed_logits[:, :5])
    assert not torch.equal(original_logits[:, 5:], changed_logits[:, 5:])


def test_message_pc_causal_affine_scan_matches_scalar_recurrence() -> None:
    torch.manual_seed(53)
    retain = torch.rand(2, 8, 1)
    innovation = torch.randn(2, 8, 5)
    expected_steps = []
    previous = torch.zeros(2, 5)
    for token in range(8):
        previous = retain[:, token] * previous + innovation[:, token]
        expected_steps.append(previous)
    expected = torch.stack(expected_steps, dim=1)

    actual = MessagePassingPCLanguageModel._causal_affine_scan(
        retain,
        innovation,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_message_pc_her_persistent_context_uses_hierarchical_timescales() -> None:
    torch.manual_seed(54)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_context_mode="persistent",
        )
    )
    x = torch.randint(0, 31, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)

    cache = model._hierarchical_error_context_cache(states)

    assert torch.equal(cache.contexts[0][:, 0], cache.candidates[0][:, 0])
    assert torch.equal(cache.contexts[1][:, 0], cache.candidates[1][:, 0])
    assert cache.write_gates[0][:, 1:].mean() == pytest.approx(0.5)
    assert cache.write_gates[1][:, 1:].mean() == pytest.approx(0.25)
    for level in range(2):
        expected = (
            (1.0 - cache.write_gates[level][:, 1:])
            * cache.contexts[level][:, :-1]
            + cache.write_gates[level][:, 1:]
            * cache.candidates[level][:, 1:]
        )
        assert torch.allclose(
            cache.contexts[level][:, 1:],
            expected,
            atol=1e-6,
            rtol=1e-6,
        )


def test_message_pc_her_hard_gate_maintains_or_replaces_whole_context() -> None:
    torch.manual_seed(57)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_context_mode="persistent_hard",
        )
    )
    x = torch.randint(0, 31, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)

    cache = model._hierarchical_error_context_cache(states)

    for level in range(2):
        assert torch.all(
            (cache.write_gates[level] == 0)
            | (cache.write_gates[level] == 1)
        )
        expected = torch.where(
            cache.write_gates[level].bool(),
            cache.candidates[level],
            cache.previous_contexts[level],
        )
        assert torch.equal(cache.contexts[level], expected)
    assert cache.write_probabilities[0][:, 1:].mean() == pytest.approx(0.5)
    assert cache.write_probabilities[1][:, 1:].mean() == pytest.approx(0.25)


def test_message_pc_her_zero_memory_scale_preserves_current_context() -> None:
    torch.manual_seed(58)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_context_mode="persistent_hard",
            hierarchical_error_memory_scale=0.0,
        )
    )
    x = torch.randint(0, 31, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)

    cache = model._hierarchical_error_context_cache(states)

    assert torch.equal(cache.contexts[0], cache.candidates[0])
    assert torch.equal(cache.contexts[1], cache.candidates[1])


def test_message_pc_her_persistent_context_drives_predictions_and_top_down() -> None:
    torch.manual_seed(55)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_context_mode="persistent",
            hierarchical_error_top_down_scale=0.5,
            state_delta_rms_cap=0.0,
        )
    )
    for weight in model.generative_weights:
        weight.normal_(mean=0.0, std=0.01)
    x = torch.randint(0, 31, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)
    cache = model._hierarchical_error_context_cache(states)

    deployed, predictions, modulated, correction = (
        model._hierarchical_error_pass(states, contexts=cache.contexts)
    )
    expected_predictions = [
        F.linear(
            cache.contexts[level]
            * torch.rsqrt(
                cache.contexts[level].square().mean(dim=-1, keepdim=True)
                + 1e-6
            ),
            model.generative_weights[level],
        )
        for level in range(2)
    ]
    expected_low = expected_predictions[0] + 0.5 * (
        model._hierarchical_error_unbind(
            cache.contexts[0],
            expected_predictions[1],
        )
    )
    expected_correction = model._hierarchical_error_unbind(
        states[0],
        expected_low,
    )

    assert torch.allclose(predictions[0], expected_predictions[0])
    assert torch.allclose(predictions[1], expected_predictions[1])
    assert torch.allclose(modulated[0], expected_low)
    assert torch.allclose(correction, expected_correction)
    assert torch.allclose(deployed, states[0] + 0.5 * expected_correction)


def test_message_pc_her_direct_predictor_has_no_key_gate() -> None:
    torch.manual_seed(64)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_prediction_mode="direct",
            state_delta_rms_cap=0.0,
        )
    )
    for weight in model.generative_weights:
        weight.normal_(mean=0.0, std=0.01)
    for weight in model.feedback_weights:
        weight.normal_(mean=0.0, std=10.0)
    x = torch.randint(0, 31, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)

    _, predictions, _, _, keys, values, gates = (
        model._hierarchical_error_factorized_pass(states)
    )

    assert all(torch.equal(prediction, value) for prediction, value in zip(predictions, values))
    assert all(torch.count_nonzero(key) == 0 for key in keys)
    assert all(torch.equal(gate, torch.ones_like(gate)) for gate in gates)


def test_message_pc_her_residual_handoff_is_exact_at_every_level() -> None:
    target = torch.randn(2, 4, 32)
    predictions = [torch.randn_like(target) for _ in range(3)]

    outcomes, final_residual = (
        MessagePassingPCLanguageModel._hierarchical_error_residual_handoff(
            target,
            predictions,
        )
    )

    assert torch.equal(outcomes[0], target)
    assert torch.equal(outcomes[1], target - predictions[0])
    assert torch.equal(
        outcomes[2],
        target - predictions[0] - predictions[1],
    )
    assert torch.equal(
        final_residual,
        target - predictions[0] - predictions[1] - predictions[2],
    )


def test_message_pc_her_direct_learning_starts_one_level_at_a_time() -> None:
    torch.manual_seed(65)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_prediction_mode="direct",
            hierarchical_error_learning_start_step=10,
            hierarchical_error_level_start_interval=5,
            hierarchical_error_context_credit_scale=0.0,
            route_lr_scale=0.0,
            matrix_optimizer="adam",
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31

    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=1e-3,
        embed_lr=0.0,
        current_step=9,
    )
    assert all(torch.count_nonzero(weight) == 0 for weight in model.generative_weights)

    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=1e-3,
        embed_lr=0.0,
        current_step=10,
    )
    assert torch.count_nonzero(model.generative_weights[0]) > 0
    assert torch.count_nonzero(model.generative_weights[1]) == 0
    assert model._last_update_diagnostics[
        "update_diag_message_pc_her_active_error_levels"
    ] == 1.0

    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=1e-3,
        embed_lr=0.0,
        current_step=15,
    )
    assert torch.count_nonzero(model.generative_weights[1]) > 0
    assert model._last_update_diagnostics[
        "update_diag_message_pc_her_active_error_levels"
    ] == 2.0


def test_message_pc_her_predictor_only_lr_freezes_base_parameters() -> None:
    torch.manual_seed(66)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_prediction_mode="direct",
            hierarchical_error_context_credit_scale=0.0,
            route_lr_scale=0.0,
            matrix_optimizer="adam",
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    before = {name: parameter.clone() for name, parameter in model.named_parameters()}

    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=0.0,
        embed_lr=0.0,
        predictor_state_lr=1e-3,
        current_step=1,
    )

    changed = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter, before[name])
    }
    assert changed
    assert all(name.startswith("generative_weights.") for name in changed)


def test_message_pc_her_predictor_reset_preserves_lower_base_weights() -> None:
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_prediction_mode="direct",
            hierarchical_error_base_depth=1,
        )
    )
    for weight in (*model.generative_weights, *model.feedback_weights):
        weight.fill_(1.0)

    counts = model.reset_hierarchical_error_predictors()

    assert torch.count_nonzero(model.generative_weights[0]) > 0
    assert torch.count_nonzero(model.feedback_weights[0]) > 0
    assert torch.count_nonzero(model.generative_weights[1]) == 0
    assert torch.count_nonzero(model.feedback_weights[1]) == 0
    assert counts == {"parameters": 2, "optimizer_buffers": 2}


def test_message_pc_recursive_cp_uses_bounded_paper_handoff() -> None:
    torch.manual_seed(67)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_prediction_mode="recursive_cp",
            hierarchical_error_target_mode="fixed_outcome",
            hierarchical_error_recursive_cp_rank=4,
            state_delta_rms_cap=0.0,
        )
    )
    assert model.recursive_cp_her is not None
    for values in model.recursive_cp_her.value_factors:
        values.normal_(mean=0.0, std=0.01)
    x = torch.randint(0, 31, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)

    deployed, predictions, _, correction, keys, _, gates = (
        model._hierarchical_error_factorized_pass(states)
    )
    _, _, cache = model._hierarchical_error_recursive_cp_cache(
        states,
        contexts=None,
    )
    expected = cache.correction.reshape_as(correction)

    assert torch.allclose(correction, expected)
    assert torch.allclose(deployed, states[0] + expected)
    assert torch.allclose(sum(predictions), expected, atol=2e-6, rtol=2e-6)
    assert all(torch.count_nonzero(key) == 0 for key in keys)
    assert all(torch.equal(gate, torch.ones_like(gate)) for gate in gates)
    assert cache.modulated[0].rank == 8
    assert model.recursive_cp_her.parameter_count == 4 * 32 * (2 + 3)


def test_message_pc_recursive_cp_predictor_only_update_freezes_old_heads() -> None:
    torch.manual_seed(68)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_prediction_mode="recursive_cp",
            hierarchical_error_target_mode="fixed_outcome",
            hierarchical_error_recursive_cp_rank=4,
            hierarchical_error_recursive_cp_upper_lr_scale=2.0,
            hierarchical_error_context_mode="persistent_hard",
            generative_lr_scale=1.0,
            route_lr_scale=0.0,
            matrix_optimizer="adam",
        )
    )
    assert model.recursive_cp_her is not None
    before = {
        name: parameter.clone()
        for name, parameter in model.named_parameters()
    }
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31

    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=0.0,
        predictor_state_lr=1e-3,
        embed_lr=0.0,
        current_step=1,
    )

    changed = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter, before[name])
    }
    assert changed
    assert all(name.startswith("recursive_cp_her.") for name in changed)
    assert all(parameter.grad is None for parameter in model.parameters())
    diagnostics = model._last_update_diagnostics
    assert diagnostics["update_diag_message_pc_uses_her_recursive_cp"] == 1.0
    assert diagnostics["update_diag_message_pc_her_recursive_cp_rank"] == 4.0
    assert diagnostics[
        "update_diag_message_pc_her_recursive_cp_upper_lr_scale"
    ] == 2.0


def test_message_pc_recursive_cp_accepts_stable_fixed_outcome_target() -> None:
    torch.manual_seed(90)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_prediction_mode="recursive_cp",
            hierarchical_error_target_mode="fixed_outcome",
            hierarchical_error_outcome_code_scale=0.2,
            hierarchical_error_recursive_cp_rank=4,
            hierarchical_error_context_mode="persistent_hard",
            matrix_optimizer="adam",
            route_lr_scale=0.0,
        )
    )
    assert model.recursive_cp_her is not None
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    state_before = {
        name: parameter.clone()
        for name, parameter in model.named_parameters()
    }

    target, _, _ = model._hierarchical_error_target(
        torch.randn(2, 8, 32),
        y,
    )
    model.decoder_weight.normal_(mean=0.0, std=3.0)
    changed_decoder_target, _, _ = model._hierarchical_error_target(
        torch.randn(2, 8, 32),
        y,
    )
    expected = F.embedding(y, _rms_norm(model.feedback_embedding)) * 0.2

    assert torch.allclose(target, expected)
    assert torch.allclose(changed_decoder_target, expected)

    model.load_state_dict(state_before, strict=False)
    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=0.0,
        predictor_state_lr=1e-3,
        embed_lr=0.0,
        current_step=1,
    )
    changed = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter, state_before[name])
    }
    assert changed
    assert all(name.startswith("recursive_cp_her.") for name in changed)
    diagnostics = model._last_update_diagnostics
    assert diagnostics["update_diag_message_pc_uses_her_recursive_cp"] == 1.0
    assert diagnostics["update_diag_message_pc_her_outcome_code_scale"] == 0.2
    assert diagnostics["update_diag_message_pc_her_outcome_rms"] > 0.0


def test_message_pc_recursive_cp_learns_stable_fixed_outcome_on_repeated_batch() -> None:
    torch.manual_seed(123)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_prediction_mode="recursive_cp",
            hierarchical_error_target_mode="fixed_outcome",
            hierarchical_error_outcome_code_scale=0.2,
            hierarchical_error_recursive_cp_rank=8,
            hierarchical_error_recursive_cp_component_norm_cap=0.5,
            hierarchical_error_context_mode="persistent_hard",
            matrix_optimizer="adam",
            route_lr_scale=0.0,
        )
    )
    x = torch.randint(0, 31, (4, 8))
    y = (x + 1) % 31

    first_error = 0.0
    last_error = 0.0
    last_cosine = 0.0
    for step in range(1, 11):
        model.local_update(
            x,
            y,
            decoder_lr=0.0,
            state_lr=0.0,
            predictor_state_lr=1e-3,
            embed_lr=0.0,
            current_step=step,
        )
        diagnostics = model._last_update_diagnostics
        if step == 1:
            first_error = diagnostics[
                "update_diag_message_pc_her_predictor_error_rms"
            ]
        if step == 10:
            last_error = diagnostics[
                "update_diag_message_pc_her_predictor_error_rms"
            ]
            last_cosine = diagnostics[
                "update_diag_message_pc_her_prediction_target_cosine"
            ]

    assert last_error < first_error - 0.002
    assert last_cosine > 0.25
    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_recursive_cp_fixed_outcome_audits_early_and_mature_levels() -> None:
    torch.manual_seed(321)
    model = MessagePassingPCLanguageModel(
        small_config(
            num_layers=4,
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_prediction_mode="recursive_cp",
            hierarchical_error_target_mode="fixed_outcome",
            hierarchical_error_outcome_code_scale=0.2,
            hierarchical_error_recursive_cp_rank=8,
            hierarchical_error_recursive_cp_component_norm_cap=0.5,
            hierarchical_error_context_mode="instant",
            matrix_optimizer="adam",
            route_lr_scale=0.0,
        )
    )
    x = torch.arange(0, 31).repeat(3)[:64].reshape(8, 8)
    y = (x + 1) % 31

    def snapshot() -> list[dict[str, float]]:
        diagnostics = model._last_update_diagnostics
        rows: list[dict[str, float]] = []
        for level in range(4):
            prefix = f"update_diag_message_pc_her_recursive_cp_level_{level}"
            rows.append(
                {
                    "target_rms": diagnostics[f"{prefix}_target_rms"],
                    "prediction_rms": diagnostics[f"{prefix}_prediction_rms"],
                    "predictor_error_rms": diagnostics[
                        f"{prefix}_predictor_error_rms"
                    ],
                    "prediction_target_cosine": diagnostics[
                        f"{prefix}_prediction_target_cosine"
                    ],
                    "residual_energy_explained_fraction": diagnostics[
                        f"{prefix}_residual_energy_explained_fraction"
                    ],
                    "least_squares_prediction_scale": diagnostics[
                        f"{prefix}_least_squares_prediction_scale"
                    ],
                    "tensor_outcome_energy": diagnostics[
                        f"{prefix}_tensor_outcome_energy"
                    ],
                    "tensor_prediction_energy": diagnostics[
                        f"{prefix}_tensor_prediction_energy"
                    ],
                    "tensor_error_energy": diagnostics[
                        f"{prefix}_tensor_error_energy"
                    ],
                    "tensor_residual_energy_explained_fraction": diagnostics[
                        f"{prefix}_tensor_residual_energy_explained_fraction"
                    ],
                    "tensor_prediction_target_cosine": diagnostics[
                        f"{prefix}_tensor_prediction_target_cosine"
                    ],
                }
            )
        return rows

    snapshots: dict[int, list[dict[str, float]]] = {}
    for step in range(1, 21):
        model.local_update(
            x,
            y,
            decoder_lr=0.0,
            state_lr=0.0,
            predictor_state_lr=1e-3,
            embed_lr=0.0,
            current_step=step,
        )
        if step in {1, 20}:
            snapshots[step] = snapshot()

    early = snapshots[1]
    mature = snapshots[20]
    for early_row, mature_row in zip(early, mature):
        assert all(math.isfinite(value) for value in mature_row.values())
        assert early_row["target_rms"] > 0.0
        assert early_row["prediction_rms"] == 0.0
        assert early_row["tensor_prediction_energy"] == 0.0
        assert mature_row["prediction_rms"] > 0.0
        assert mature_row["tensor_prediction_energy"] > 0.0
        assert mature_row["predictor_error_rms"] < early_row["predictor_error_rms"]
        assert mature_row["residual_energy_explained_fraction"] > 0.0
        assert mature_row["tensor_outcome_energy"] > 0.0
        assert mature_row["tensor_error_energy"] > 0.0

    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_recursive_cp_fixed_outcome_improves_heldout_batch() -> None:
    torch.manual_seed(42)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_prediction_mode="recursive_cp",
            hierarchical_error_target_mode="fixed_outcome",
            hierarchical_error_outcome_code_scale=0.2,
            hierarchical_error_recursive_cp_rank=16,
            hierarchical_error_recursive_cp_component_norm_cap=0.2,
            hierarchical_error_context_mode="instant",
            matrix_optimizer="adam",
            route_lr_scale=0.0,
        )
    )
    train_x = torch.arange(0, 31).repeat(3)[:64].reshape(8, 8)
    train_y = (train_x + 1) % 31
    val_x = train_x.flip(1)
    val_y = (val_x + 1) % 31

    def measure_validation() -> tuple[float, float]:
        model.local_update(
            val_x,
            val_y,
            decoder_lr=0.0,
            state_lr=0.0,
            predictor_state_lr=0.0,
            embed_lr=0.0,
            current_step=999,
        )
        diagnostics = model._last_update_diagnostics
        return (
            diagnostics["update_diag_message_pc_her_predictor_error_rms"],
            diagnostics["update_diag_message_pc_her_prediction_target_cosine"],
        )

    before_error, before_cosine = measure_validation()
    for step in range(1, 31):
        model.local_update(
            train_x,
            train_y,
            decoder_lr=0.0,
            state_lr=0.0,
            predictor_state_lr=2e-4,
            embed_lr=0.0,
            current_step=step,
        )
    after_error, after_cosine = measure_validation()

    assert after_error < before_error - 0.001
    assert after_cosine > before_cosine + 0.15
    assert all(parameter.grad is None for parameter in model.parameters())


def delayed_recursive_cp_config(**overrides: object) -> MessagePassingPCConfig:
    values: dict[str, object] = {
        "inference_mode": "her_error_hierarchy",
        "hierarchical_error_binding_mode": "factorized",
        "hierarchical_error_prediction_mode": "recursive_cp",
        "hierarchical_error_target_mode": "fixed_outcome",
        "hierarchical_error_recursive_cp_rank": 4,
        "hierarchical_error_recursive_cp_component_norm_cap": 0.1,
        "hierarchical_error_recursive_cp_delayed_consolidation": True,
        "matrix_optimizer": "adam",
    }
    values.update(overrides)
    return small_config(**values)


def test_message_pc_recursive_cp_delayed_consolidation_validates_contract() -> None:
    with pytest.raises(ValueError, match="delayed recursive CP consolidation"):
        MessagePassingPCLanguageModel(
            small_config(
                hierarchical_error_recursive_cp_delayed_consolidation=True,
            )
        )
    with pytest.raises(ValueError, match="consolidation_rate"):
        MessagePassingPCLanguageModel(
            delayed_recursive_cp_config(
                hierarchical_error_recursive_cp_consolidation_rate=0.0,
            )
        )
    with pytest.raises(ValueError, match="min_improvement"):
        MessagePassingPCLanguageModel(
            delayed_recursive_cp_config(
                hierarchical_error_recursive_cp_min_improvement=-0.1,
            )
        )


def test_message_pc_recursive_cp_delayed_consolidation_accepts_local_improvement() -> None:
    torch.manual_seed(70)
    model = MessagePassingPCLanguageModel(
        delayed_recursive_cp_config(hierarchical_error_base_depth=1)
    )
    assert model.recursive_cp_her is not None
    assert model.recursive_cp_her_fast is not None
    memories = [torch.randn(16, 32)]
    for values in model.recursive_cp_her_fast.value_factors:
        values.normal_(mean=0.0, std=0.001)
    target = model.recursive_cp_her_fast.predict(memories).correction
    deployed_result = model.recursive_cp_her.local_gradients(memories, target)
    fast_result = model.recursive_cp_her_fast.local_gradients(memories, target)

    diagnostics = model._consolidate_recursive_cp_if_better(
        memories,
        target,
        deployed_result,
        fast_result,
    )

    assert diagnostics["fast_score"] == pytest.approx(0.0, abs=1e-6)
    assert diagnostics["candidate_score"] < diagnostics["deployed_score"]
    assert diagnostics["accepted"] == 1.0
    assert diagnostics["attempts"] == 1.0
    assert diagnostics["accepts"] == 1.0
    assert any(
        torch.count_nonzero(values)
        for values in model.recursive_cp_her.value_factors
    )


def test_message_pc_recursive_cp_delayed_consolidation_rejects_regression() -> None:
    torch.manual_seed(71)
    model = MessagePassingPCLanguageModel(
        delayed_recursive_cp_config(hierarchical_error_base_depth=1)
    )
    assert model.recursive_cp_her is not None
    assert model.recursive_cp_her_fast is not None
    memories = [torch.randn(16, 32)]
    for values in model.recursive_cp_her_fast.value_factors:
        values.normal_(mean=0.0, std=0.001)
    target = -model.recursive_cp_her_fast.predict(memories).correction
    deployed_result = model.recursive_cp_her.local_gradients(memories, target)
    fast_result = model.recursive_cp_her_fast.local_gradients(memories, target)
    deployed_before = copy.deepcopy(model.recursive_cp_her.state_dict())

    diagnostics = model._consolidate_recursive_cp_if_better(
        memories,
        target,
        deployed_result,
        fast_result,
    )

    assert diagnostics["fast_score"] > diagnostics["deployed_score"]
    assert diagnostics["accepted"] == 0.0
    assert diagnostics["attempts"] == 1.0
    assert diagnostics["accepts"] == 0.0
    for name, value in model.recursive_cp_her.state_dict().items():
        assert torch.equal(value, deployed_before[name])


def test_message_pc_recursive_cp_delayed_update_trains_fast_trace_only() -> None:
    torch.manual_seed(72)
    model = MessagePassingPCLanguageModel(delayed_recursive_cp_config())
    assert model.recursive_cp_her is not None
    assert model.recursive_cp_her_fast is not None
    model.enforce_recursive_cp_component_bound()
    deployed_before = copy.deepcopy(model.recursive_cp_her.state_dict())
    fast_before = copy.deepcopy(model.recursive_cp_her_fast.state_dict())
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31

    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=0.0,
        predictor_state_lr=1e-3,
        embed_lr=0.0,
        current_step=1,
    )

    for name, value in model.recursive_cp_her.state_dict().items():
        assert torch.equal(value, deployed_before[name])
    assert any(
        not torch.equal(value, fast_before[name])
        for name, value in model.recursive_cp_her_fast.state_dict().items()
    )
    diagnostics = model._last_update_diagnostics
    assert diagnostics[
        "update_diag_message_pc_her_recursive_cp_uses_delayed_consolidation"
    ] == 1.0
    assert diagnostics[
        "update_diag_message_pc_her_recursive_cp_consolidation_accepted"
    ] == 0.0
    assert diagnostics[
        "update_diag_message_pc_her_recursive_cp_consolidation_attempts"
    ] == 1.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_recursive_cp_delayed_checkpoint_migrates_from_single_copy() -> None:
    torch.manual_seed(73)
    source = MessagePassingPCLanguageModel(
        replace(
            delayed_recursive_cp_config(),
            hierarchical_error_recursive_cp_delayed_consolidation=False,
        )
    )
    resumed = MessagePassingPCLanguageModel(delayed_recursive_cp_config())

    incompatible = resumed.load_state_dict(source.state_dict(), strict=False)

    assert set(incompatible.missing_keys) == resumed.recursive_cp_delayed_checkpoint_keys()
    assert incompatible.unexpected_keys == []
    resumed.initialize_recursive_cp_fast_from_deployed()
    assert resumed.recursive_cp_her is not None
    assert resumed.recursive_cp_her_fast is not None
    for deployed, fast in zip(
        resumed.recursive_cp_her.parameters(),
        resumed.recursive_cp_her_fast.parameters(),
    ):
        assert torch.equal(deployed, fast)


def test_message_pc_recursive_cp_component_bound_applies_after_local_update() -> None:
    torch.manual_seed(69)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_prediction_mode="recursive_cp",
            hierarchical_error_target_mode="fixed_outcome",
            hierarchical_error_recursive_cp_rank=4,
            hierarchical_error_recursive_cp_component_norm_cap=0.1,
            matrix_optimizer="adam",
        )
    )
    assert model.recursive_cp_her is not None
    for values in model.recursive_cp_her.value_factors:
        values.fill_(2.0)
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31

    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=0.0,
        predictor_state_lr=0.0,
        embed_lr=0.0,
        current_step=1,
    )

    for inputs, contexts, values in zip(
        model.recursive_cp_her.input_factors,
        model.recursive_cp_her.context_factors,
        model.recursive_cp_her.value_factors,
    ):
        assert torch.allclose(inputs.norm(dim=-1), torch.ones(4), atol=2e-6)
        if contexts.numel():
            assert torch.allclose(
                contexts.norm(dim=-1),
                torch.ones_like(contexts[..., 0]),
                atol=2e-6,
            )
        assert torch.all(values.norm(dim=-1) <= 0.1000002)
    diagnostics = model._last_update_diagnostics
    assert diagnostics[
        "update_diag_message_pc_her_recursive_cp_component_norm_cap"
    ] == pytest.approx(0.1)
    assert diagnostics[
        "update_diag_message_pc_her_recursive_cp_clipped_fraction"
    ] == pytest.approx(1.0)
    assert diagnostics[
        "update_diag_message_pc_her_recursive_cp_max_norm_after_clip"
    ] == pytest.approx(0.1, abs=2e-7)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_recursive_cp_reset_clears_only_predictor_and_optimizer() -> None:
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_prediction_mode="recursive_cp",
            hierarchical_error_target_mode="fixed_outcome",
            hierarchical_error_recursive_cp_rank=4,
            matrix_optimizer="adam",
        )
    )
    assert model.recursive_cp_her is not None
    base_before = model.decoder_weight.clone()
    for parameter in model.recursive_cp_her.parameters():
        parameter.fill_(1.0)
    for name, buffers in model._adam_buffer_names.items():
        if name.startswith("recursive_cp_her."):
            for buffer in buffers:
                getattr(model, buffer).fill_(1.0)

    counts = model.reset_hierarchical_error_predictors()

    assert torch.equal(model.decoder_weight, base_before)
    assert all(
        torch.count_nonzero(values) == 0
        for values in model.recursive_cp_her.value_factors
    )
    assert counts == {"parameters": 6, "optimizer_buffers": 6}
    for name, buffers in model._adam_buffer_names.items():
        if name.startswith("recursive_cp_her."):
            assert all(torch.count_nonzero(getattr(model, buffer)) == 0 for buffer in buffers)


@pytest.mark.parametrize("context_mode", ["persistent", "persistent_hard"])
def test_message_pc_her_persistent_gate_learns_locally_without_autograd(
    context_mode: str,
) -> None:
    torch.manual_seed(56)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_context_mode=context_mode,
            route_lr_scale=0.0,
            matrix_optimizer="adam",
        )
    )
    for weight in model.generative_weights:
        weight.normal_(mean=0.0, std=0.01)
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    gate_before = [weight.clone() for weight in model.hierarchical_gate_weights]

    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=1e-3,
        embed_lr=0.0,
        weight_decay=0.0,
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(gate_before, model.hierarchical_gate_weights)
    )
    diagnostics = model._last_update_diagnostics
    assert diagnostics["update_diag_message_pc_her_uses_persistent_context"] == 1.0
    assert diagnostics["update_diag_message_pc_her_gate_update_rms"] > 0.0
    assert diagnostics["update_diag_message_pc_her_gate_eligibility_rms"] > 0.0
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_causal_error_carry_matches_zero_scale_and_stays_causal() -> None:
    torch.manual_seed(31)
    model = MessagePassingPCLanguageModel(
        small_config(
            token_feedback_mode="direct_layer",
            block_state_mode="identity_residual",
        )
    )
    original = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    changed = original.clone()
    changed[:, 5:] = torch.tensor([[20, 21, 22]])

    baseline = model(original)
    zero_logits, _ = causal_error_carry_logits(model, original, scale=0.0)
    original_logits, _ = causal_error_carry_logits(model, original, scale=0.25)
    changed_logits, _ = causal_error_carry_logits(model, changed, scale=0.25)

    assert torch.equal(zero_logits, baseline)
    assert torch.equal(original_logits[:, :5], changed_logits[:, :5])
    assert not torch.equal(original_logits[:, 5:], changed_logits[:, 5:])
    observed = observed_next_tokens(original, model.config.bos_token_id)
    assert torch.equal(observed[:, :-1], original[:, 1:])
    assert observed[0, -1].item() == model.config.bos_token_id
    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_early_readout_skips_deeper_blocks() -> None:
    torch.manual_seed(19)
    model = MessagePassingPCLanguageModel(
        small_config(readout_layer=1, decoder_state_mode="rms_norm")
    )
    x = torch.randint(0, 31, (2, 8))
    full_states, _ = model._forward_states(x, with_cache=False)
    expected = model._decoder_logits(full_states[1])

    before = model(x)
    model.self_weights[1].normal_(mean=0.0, std=3.0)
    model.message_weights[1].normal_(mean=0.0, std=3.0)
    after = model(x)

    assert torch.equal(before, expected)
    assert torch.equal(after, expected)


def test_message_pc_readout_objective_share_keeps_all_layers_supervised() -> None:
    early = MessagePassingPCLanguageModel(
        small_config(
            num_layers=4,
            readout_layer=1,
            readout_objective_share=0.7,
        )
    )
    top = MessagePassingPCLanguageModel(
        small_config(
            num_layers=4,
            readout_layer=0,
            readout_objective_share=0.7,
        )
    )

    assert early._decoder_state_weights() == pytest.approx([0.7, 0.1, 0.1, 0.1])
    assert top._decoder_state_weights() == pytest.approx([0.1, 0.1, 0.1, 0.7])
    assert sum(early._decoder_state_weights()) == pytest.approx(1.0)


def test_message_pc_evaluator_reports_effective_readout() -> None:
    selected = MessagePassingPCLanguageModel(small_config(readout_layer=1))
    top = MessagePassingPCLanguageModel(small_config(readout_layer=0))

    assert effective_message_readout_layer(selected, 0) == (1, "checkpoint")
    assert effective_message_readout_layer(selected, 2) == (2, "cli")
    assert effective_message_readout_layer(top, 0) == (2, "checkpoint-top")


def test_message_pc_local_update_uses_no_parameter_gradients() -> None:
    torch.manual_seed(2)
    model = MessagePassingPCLanguageModel(small_config())
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    loss = model.local_update(x, y, decoder_lr=1e-3, state_lr=1e-3, embed_lr=1e-3)

    assert isinstance(loss, float)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not torch.equal(model.decoder_weight, before["decoder_weight"])
    assert not torch.equal(model.query_weights[0], before["query_weights.0"])
    assert not torch.equal(model.feedback_weights[0], before["feedback_weights.0"])
    assert int(model.local_adam_step.item()) == 1
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())


def test_message_pc_her_update_uses_only_local_error_paths() -> None:
    torch.manual_seed(43)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            route_lr_scale=1.0,
            generative_lr_scale=1.0,
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    loss = model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
    )

    assert isinstance(loss, float)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not torch.equal(model.decoder_weight, before["decoder_weight"])
    assert not torch.equal(
        model.generative_weights[0],
        before["generative_weights.0"],
    )
    assert torch.equal(
        model.feedback_weights[0],
        before["feedback_weights.0"],
    )
    diagnostics = model._last_update_diagnostics
    assert diagnostics["update_diag_message_pc_uses_her_error_hierarchy"] == 1.0
    assert diagnostics["update_diag_message_pc_her_base_error_rms"] > 0
    assert diagnostics["update_diag_message_pc_her_predictor_error_rms"] > 0
    assert -1.0 <= diagnostics[
        "update_diag_message_pc_her_prediction_target_cosine"
    ] <= 1.0
    assert -1.0 <= diagnostics[
        "update_diag_message_pc_her_top_down_target_cosine"
    ] <= 1.0
    assert diagnostics["update_diag_message_pc_generative_update_rms"] > 0
    assert diagnostics["update_diag_message_pc_feedback_update_rms"] == 0.0
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())


def test_message_pc_her_base_depth_trains_base_and_upper_paths_separately() -> None:
    torch.manual_seed(46)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_base_depth=1,
            route_lr_scale=1.0,
            generative_lr_scale=1.0,
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
    )

    assert not torch.equal(model.self_weights[0], before["self_weights.0"])
    assert torch.equal(
        model.generative_weights[0],
        before["generative_weights.0"],
    )
    assert not torch.equal(
        model.generative_weights[1],
        before["generative_weights.1"],
    )
    diagnostics = model._last_update_diagnostics
    assert diagnostics["update_diag_message_pc_her_base_depth"] == 1.0
    assert diagnostics["update_diag_message_pc_her_error_levels"] == 1.0
    assert diagnostics["update_diag_message_pc_her_base_local_credit_rms"] > 0


def test_message_pc_her_predictor_update_descends_declared_local_error() -> None:
    torch.manual_seed(48)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_base_depth=1,
            hierarchical_error_binding_mode="identity",
            hierarchical_error_context_credit_scale=0.0,
            route_lr_scale=0.0,
            generative_lr_scale=1.0,
            matrix_optimizer="adam",
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31

    def predictor_error() -> float:
        states, _ = model._forward_states(x, with_cache=False)
        _, predictions, modulated, _ = model._hierarchical_error_pass(states)
        residual, _ = model._decoder_negative_state_gradient(states[1], y)
        losses = []
        for offset, layer in enumerate(range(1, model.config.num_layers)):
            outcome = model._hierarchical_error_bind(states[layer], residual)
            losses.append(0.5 * (modulated[offset] - outcome).square().sum())
            residual = outcome - predictions[offset]
        return float(torch.stack(losses).sum().item())

    before = predictor_error()
    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=1e-5,
        embed_lr=0.0,
        weight_decay=0.0,
    )
    after = predictor_error()

    assert after < before


def test_message_pc_gradient_alignment_oracle_matches_free_decoder_rule() -> None:
    torch.manual_seed(47)
    model = MessagePassingPCLanguageModel(
        small_config(
            decoder_supervision_mode="top_only",
            decoder_learning_source="free",
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    _, oracle = oracle_gradients(copy.deepcopy(model), x, y)
    local = capture_local_gradients(
        copy.deepcopy(model),
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=1,
    )

    raw = vector_alignment(
        oracle,
        local,
        candidate_is_update=False,
    )
    synthetic_update = {name: -gradient for name, gradient in oracle.items()}
    update = vector_alignment(
        oracle,
        synthetic_update,
        candidate_is_update=True,
    )

    assert raw["decoder"]["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert update["all"]["cosine"] == pytest.approx(1.0, abs=1e-6)


def test_message_pc_free_phase_has_zero_local_block_gradient() -> None:
    torch.manual_seed(53)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            route_update_mode="local_energy",
        )
    )
    x = torch.randint(0, 31, (2, 8))

    gradients = free_phase_block_gradients(model, x)

    assert gradients
    assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients.values())


def bidirectional_config(**overrides: object) -> MessagePassingPCConfig:
    values: dict[str, object] = {
        "inference_mode": "bidirectional_pc",
        "block_state_mode": "identity_residual",
        "decoder_supervision_mode": "top_only",
        "readout_layer": 0,
        "state_delta_rms_cap": 0.0,
        "energy_backtracking_steps": 8,
    }
    values.update(overrides)
    return small_config(**values)


def test_bidirectional_pc_inverse_input_signal_matches_autograd() -> None:
    torch.manual_seed(79)
    model = MessagePassingPCLanguageModel(bidirectional_config())
    model.feedback_weights[0].normal_(std=0.1)
    upper = torch.randn(2, 5, model.config.hidden_dim, requires_grad=True)
    output_signal = torch.randn_like(upper)

    prediction = model._inverse_block(0, upper).prediction
    (expected,) = torch.autograd.grad(
        (prediction * output_signal).sum(),
        (upper,),
    )
    detached_upper = upper.detach()
    cache = model._inverse_block(0, detached_upper)
    actual = model._inverse_input_signal(
        0,
        detached_upper,
        cache,
        output_signal,
    )

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_bidirectional_pc_inverse_parameter_gradients_match_autograd() -> None:
    torch.manual_seed(83)
    model = MessagePassingPCLanguageModel(bidirectional_config())
    model.feedback_weights[0].normal_(std=0.1)
    parameter_names = (
        "inverse_query_weights.0",
        "inverse_key_weights.0",
        "generative_weights.0",
        "feedback_weights.0",
    )
    parameters = dict(model.named_parameters())
    for name in parameter_names:
        parameters[name].requires_grad_(True)
    upper = torch.randn(2, 5, model.config.hidden_dim)
    output_signal = torch.randn_like(upper)
    token_count = float(upper.shape[0] * upper.shape[1])

    cache = model._inverse_block(0, upper)
    expected_values = torch.autograd.grad(
        (cache.prediction * output_signal).sum() / token_count,
        tuple(parameters[name] for name in parameter_names),
    )
    actual = model._bidirectional_inverse_parameter_gradients(
        0,
        cache,
        output_signal,
        token_count=token_count,
    )

    for name, expected in zip(parameter_names, expected_values):
        assert torch.allclose(actual[name], expected, atol=2e-6, rtol=2e-5)


def test_bidirectional_pc_state_deltas_match_energy_oracle() -> None:
    torch.manual_seed(89)
    model = MessagePassingPCLanguageModel(bidirectional_config())
    for feedback_weight in model.feedback_weights:
        feedback_weight.normal_(std=0.1)
    x = torch.randint(0, 31, (2, 6))
    y = (x + 1) % 31
    forward_states, _ = model._forward_states(x, with_cache=False)
    states = [forward_states[0]] + [
        (state + 0.05 * torch.randn_like(state)).requires_grad_()
        for state in forward_states[1:]
    ]

    energy = model._bidirectional_pc_energy(states, y)
    expected_gradients = torch.autograd.grad(energy, tuple(states[1:]))
    actual_deltas, _ = model._bidirectional_pc_state_deltas(states, y)
    token_count = float(x.numel())

    for actual, expected in zip(actual_deltas, expected_gradients):
        assert torch.allclose(
            actual,
            -token_count * expected,
            atol=3e-6,
            rtol=3e-5,
        )


def test_bidirectional_pc_settling_decreases_energy() -> None:
    torch.manual_seed(97)
    model = MessagePassingPCLanguageModel(
        bidirectional_config(inference_steps=6, inference_lr=0.5)
    )
    for feedback_weight in model.feedback_weights:
        feedback_weight.normal_(std=0.1)
    x = torch.randint(0, 31, (2, 5))
    y = (x + 1) % 31
    states, _ = model._forward_states(x, with_cache=False)
    before = model._bidirectional_pc_energy(states, y)

    settled, diagnostics = model._settle_bidirectional_pc(
        states,
        y,
        inference_lr=0.5,
        phase="test",
    )
    after = model._bidirectional_pc_energy(settled, y)

    assert after < before
    assert diagnostics[
        "update_diag_message_pc_bidirectional_test_energy_reduction"
    ] > 0


def test_bidirectional_pc_top_clamp_reaches_every_latent_layer() -> None:
    torch.manual_seed(101)
    model = MessagePassingPCLanguageModel(
        bidirectional_config(
            num_layers=4,
            inference_steps=6,
            inference_lr=0.1,
        )
    )
    x = torch.randint(0, 31, (2, 5))
    y = (x + 1) % 31
    states, _ = model._forward_states(x, with_cache=False)
    free_states, _ = model._settle_bidirectional_pc(
        states,
        None,
        inference_lr=0.1,
        phase="free_test",
    )
    clamped_states, _ = model._settle_bidirectional_pc(
        free_states,
        y,
        inference_lr=0.1,
        phase="clamped_test",
    )

    assert all(
        torch.count_nonzero(clamped - free) > 0
        for clamped, free in zip(clamped_states[1:], free_states[1:])
    )


def test_bidirectional_pc_local_update_uses_no_parameter_autograd() -> None:
    torch.manual_seed(103)
    model = MessagePassingPCLanguageModel(
        bidirectional_config(inference_steps=6, inference_lr=0.1)
    )
    x = torch.randint(0, 31, (2, 5))
    y = (x + 1) % 31
    before = {
        name: parameter.clone()
        for name, parameter in model.named_parameters()
    }

    for step in range(1, 3):
        model.local_update(
            x,
            y,
            decoder_lr=1e-3,
            state_lr=1e-3,
            embed_lr=1e-3,
            current_step=step,
        )

    after = dict(model.named_parameters())
    assert not torch.equal(before["decoder_weight"], after["decoder_weight"])
    assert not torch.equal(before["message_weights.0"], after["message_weights.0"])
    assert not torch.equal(before["feedback_weights.0"], after["feedback_weights.0"])
    assert not torch.equal(
        before["inverse_query_weights.0"],
        after["inverse_query_weights.0"],
    )
    assert model._last_update_diagnostics[
        "update_diag_message_pc_bidirectional_uses_free_settling"
    ] == 0.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_bidirectional_pc_deployment_reads_free_settled_state() -> None:
    torch.manual_seed(104)
    model = MessagePassingPCLanguageModel(
        bidirectional_config(
            bidirectional_free_settling=True,
            bidirectional_deployment_settling=True,
            inference_steps=4,
            inference_lr=0.1,
        )
    )
    for feedback_weight in model.feedback_weights:
        feedback_weight.normal_(std=0.1)
    x = torch.randint(0, 31, (2, 5))
    feedforward_states, _ = model._forward_states(x, with_cache=False)
    settled_states, _ = model._settle_bidirectional_pc(
        feedforward_states,
        None,
        inference_lr=0.1,
        phase="expected_deployment",
    )

    actual = model(x)
    expected = model._decoder_logits(settled_states[-1])

    assert torch.allclose(actual, expected)
    assert not torch.equal(
        actual,
        model._decoder_logits(feedforward_states[-1]),
    )


def test_bidirectional_pc_local_loss_uses_free_settled_state() -> None:
    torch.manual_seed(106)
    model = MessagePassingPCLanguageModel(
        bidirectional_config(
            bidirectional_free_settling=True,
            bidirectional_deployment_settling=True,
            inference_steps=4,
            inference_lr=0.1,
        )
    )
    for feedback_weight in model.feedback_weights:
        feedback_weight.normal_(std=0.1)
    x = torch.randint(0, 31, (2, 5))
    y = (x + 1) % 31
    expected = F.cross_entropy(
        model(x).reshape(-1, model.config.vocab_size),
        y.reshape(-1),
    )

    actual = model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=0,
    )

    assert actual == pytest.approx(float(expected.item()))
    assert model._last_update_diagnostics[
        "update_diag_message_pc_bidirectional_deployment_uses_free_settling"
    ] == 1.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_bidirectional_pc_forward_update_trains_from_initial_lower_state() -> None:
    torch.manual_seed(109)
    model = MessagePassingPCLanguageModel(
        bidirectional_config(
            inference_steps=4,
            inference_lr=0.1,
            forward_update_top_layers=2,
        )
    )
    x = torch.randint(0, 31, (2, 5))
    y = (x + 1) % 31
    before = [weight.clone() for weight in model.message_weights]

    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=1,
    )

    assert all(
        not torch.equal(old, new)
        for old, new in zip(before, model.message_weights)
    )
    assert model._last_update_diagnostics[
        "update_diag_message_pc_bidirectional_forward_update_layers"
    ] == 2.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_fast_attention_window_only_limits_configured_lower_layers() -> None:
    model = MessagePassingPCLanguageModel(
        small_config(
            fast_attention_window=2,
            fast_attention_layers=1,
        )
    )

    lower_mask = model._causal_mask(
        6,
        torch.device("cpu"),
        layer=0,
    )
    upper_mask = model._causal_mask(
        6,
        torch.device("cpu"),
        layer=1,
    )

    assert not lower_mask[5, 5]
    assert not lower_mask[5, 4]
    assert lower_mask[5, 3]
    assert not upper_mask[5, 0]
    assert upper_mask[0, 1]


def test_fast_attention_hides_distant_token_only_from_lower_state() -> None:
    torch.manual_seed(105)
    model = MessagePassingPCLanguageModel(
        small_config(
            fast_attention_window=2,
            fast_attention_layers=1,
        )
    )
    x = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = x.clone()
    changed[:, 0] = 7

    states, _ = model._forward_states(x, with_cache=False)
    changed_states, _ = model._forward_states(changed, with_cache=False)

    assert torch.equal(states[1][:, 4], changed_states[1][:, 4])
    assert not torch.equal(states[2][:, 4], changed_states[2][:, 4])


def test_fast_attention_window_limits_phase_transport_lower_layer() -> None:
    torch.manual_seed(156)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            fast_attention_window=2,
            fast_attention_layers=1,
            hierarchical_error_prediction_mode="direct_phase_transport",
            learned_feedback_proposal_scale=1.0,
        )
    )
    state = torch.randn(2, 6, model.config.hidden_dim)
    error_signal = torch.randn_like(state)

    lower_cache = model._residual_phase_transport_block(
        0,
        state,
        error_signal,
    )
    upper_cache = model._residual_phase_transport_block(
        1,
        state,
        error_signal,
    )

    assert lower_cache.attention_mask[5, 3]
    assert not lower_cache.attention_mask[5, 4]
    assert not upper_cache.attention_mask[5, 3]


def test_fast_slow_forward_update_language_gate_records_depth_failure() -> None:
    root = Path(
        "runs/pc_modal_lm/nanochat_exact_seq1024/"
        "fast_slow_forward_update_100_20260715"
    )
    graph_path = root / "graph.jsonl"
    if not graph_path.exists():
        pytest.skip("fast/slow language gate artifact is not synced")
    rows = [
        json.loads(line)
        for line in graph_path.read_text().splitlines()
        if line.strip()
    ]
    by_tag = {row["run_tag"]: row for row in rows}
    fast_slow = by_tag["Fast/slow PC"]
    all_full = by_tag["All-full PC"]
    gpt = by_tag["GPT measured"]

    assert fast_slow["val_bpb"] > all_full["val_bpb"]
    assert fast_slow["accuracy"] < all_full["accuracy"]
    assert fast_slow["val_bpb"] > gpt["val_bpb"]
    assert fast_slow["accuracy"] < gpt["accuracy"]
    for raw_path in (root / "candidate.jsonl", root / "control.jsonl"):
        raw_rows = [
            json.loads(line)
            for line in raw_path.read_text().splitlines()
            if line.strip()
        ]
        assert raw_rows
        assert all(row["global_backprops"] == 0.0 for row in raw_rows)


def test_learned_only_residual_transport_depth_pair_records_depth_failure() -> None:
    root = Path(
        "runs/pc_modal_lm/nanochat_exact_seq1024/"
        "residual_error_learned_only_depth_pair_100_20260715"
    )
    paths = {
        "l1": root / "l1_d448" / "train.log",
        "l2": root / "l2_d384" / "train.log",
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("learned-only residual depth pair artifacts are not synced")

    rows = {}
    for name, path in paths.items():
        raw = path.read_bytes()
        text = raw.decode("utf-16") if b"\x00" in raw[:40] else raw.decode()
        parsed = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip().startswith("{")
        ]
        assert parsed
        rows[name] = parsed[-1]

    l1 = rows["l1"]
    l2 = rows["l2"]
    assert l1["message_pc_active_learned_params"] < l2[
        "message_pc_active_learned_params"
    ]
    assert l2["message_pc_active_learned_params"] < 7_400_000
    assert l1["message_pc_meta_uses_exact_vjp"] == 0.0
    assert l2["message_pc_meta_uses_exact_vjp"] == 0.0
    assert l1["message_pc_meta_uses_decoder_output_feedback"] == 0.0
    assert l2["message_pc_meta_uses_decoder_output_feedback"] == 0.0
    assert l1["message_pc_residual_error_uses_learned_only_transport"] == 1.0
    assert l2["message_pc_residual_error_uses_learned_only_transport"] == 1.0
    assert l1["global_backprops"] == 0.0
    assert l2["global_backprops"] == 0.0
    assert l2["val_bpb"] > l1["val_bpb"]
    assert l2["accuracy"] < l1["accuracy"]


def test_learned_only_phase_transport_depth_pair_records_mixed_depth_signal() -> None:
    root = Path(
        "runs/pc_modal_lm/nanochat_exact_seq1024/"
        "residual_error_learned_only_phase_depth_pair_100_20260715"
    )
    paths = {
        "l1": root / "l1_d448" / "train.log",
        "l2": root / "l2_d384" / "train.log",
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("learned-only phase depth pair artifacts are not synced")

    rows = {}
    for name, path in paths.items():
        raw = path.read_bytes()
        text = raw.decode("utf-16") if b"\x00" in raw[:40] else raw.decode()
        parsed = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip().startswith("{")
        ]
        progress = [row for row in parsed if row.get("event") == "train_progress"]
        assert progress
        rows[name] = progress[-1]

    l1 = rows["l1"]
    l2 = rows["l2"]
    assert l1["message_pc_active_learned_params"] < l2[
        "message_pc_active_learned_params"
    ]
    assert l2["message_pc_active_learned_params"] < 7_400_000
    for row in (l1, l2):
        assert row["global_backprops"] == 0.0
        assert row["message_pc_meta_uses_exact_vjp"] == 0.0
        assert row["message_pc_meta_uses_decoder_output_feedback"] == 0.0
        assert row["message_pc_meta_uses_kp_output_feedback"] == 1.0
        assert row["message_pc_residual_error_uses_learned_only_transport"] == 1.0
        assert row["update_diag_message_pc_residual_error_monotone_energy"] == 1.0
        assert row["update_diag_message_pc_residual_error_accepted_steps"] == 6.0
    assert l2["val_bpb"] < l1["val_bpb"]
    assert l2["accuracy"] < l1["accuracy"]
    assert (
        l2["update_diag_message_pc_residual_phase_transport_cosine"]
        > l1["update_diag_message_pc_residual_phase_transport_cosine"]
    )


def test_learned_only_phase_transport_diag_records_per_layer_credit() -> None:
    path = Path(
        "runs/pc_modal_lm/nanochat_exact_seq1024/"
        "residual_error_learned_only_phase_transport_diag_100_20260715/"
        "l2_d384/results.jsonl"
    )
    if not path.exists():
        pytest.skip("learned-only phase transport diagnostic artifact is not synced")

    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip().startswith("{")
    ]
    progress = [row for row in rows if row.get("event") == "train_progress"]
    assert progress
    row = progress[-1]

    assert row["global_backprops"] == 0.0
    assert row["message_pc_meta_uses_exact_vjp"] == 0.0
    assert row["message_pc_meta_uses_decoder_output_feedback"] == 0.0
    assert row["message_pc_meta_uses_kp_output_feedback"] == 1.0
    assert row["message_pc_residual_error_uses_learned_only_transport"] == 1.0
    for layer in (1, 2):
        assert row[
            f"update_diag_message_pc_residual_error_exact_transport_signal_rms_layer_{layer}"
        ] == 0.0
        assert row[
            f"update_diag_message_pc_residual_error_learned_transport_signal_rms_layer_{layer}"
        ] > 0.0
        assert row[
            f"update_diag_message_pc_residual_error_learned_transport_signal_ratio_layer_{layer}"
        ] == 1.0


def test_phase_transport_refine_improves_l2_bpb_and_upper_signal() -> None:
    root = Path("runs/pc_modal_lm/nanochat_exact_seq1024")
    baseline_path = (
        root
        / "residual_error_learned_only_phase_transport_diag_100_20260715"
        / "l2_d384"
        / "results.jsonl"
    )
    refine_path = (
        root
        / "residual_error_learned_only_phase_transport_refine1_100_20260715"
        / "l2_d384"
        / "results.jsonl"
    )
    if not baseline_path.exists() or not refine_path.exists():
        pytest.skip("phase transport refine artifacts are not synced")

    def latest_progress(path: Path) -> dict[str, object]:
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip().startswith("{")
        ]
        progress = [row for row in rows if row.get("event") == "train_progress"]
        assert progress
        return progress[-1]

    baseline = latest_progress(baseline_path)
    refine = latest_progress(refine_path)
    assert refine["global_backprops"] == 0.0
    assert refine["update_diag_message_pc_residual_phase_transport_refine_passes"] == 1.0
    assert refine["val_bpb"] < baseline["val_bpb"]
    assert (
        refine[
            "update_diag_message_pc_residual_phase_transport_cosine_layer_1"
        ]
        > baseline[
            "update_diag_message_pc_residual_phase_transport_cosine_layer_1"
        ]
    )
    assert (
        refine[
            "update_diag_message_pc_residual_error_learned_transport_signal_rms_layer_2"
        ]
        > baseline[
            "update_diag_message_pc_residual_error_learned_transport_signal_rms_layer_2"
        ]
    )


def test_phase_transport_refine_depth_pair_records_mixed_signal() -> None:
    root = (
        Path("runs/pc_modal_lm/nanochat_exact_seq1024")
        / "residual_error_learned_only_phase_transport_refine1_100_20260715"
    )
    paths = {
        "l1": root / "l1_d448" / "results.jsonl",
        "l2": root / "l2_d384" / "results.jsonl",
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("phase transport refine depth pair artifacts are not synced")

    def latest_progress(path: Path) -> dict[str, object]:
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip().startswith("{")
        ]
        progress = [row for row in rows if row.get("event") == "train_progress"]
        assert progress
        return progress[-1]

    l1 = latest_progress(paths["l1"])
    l2 = latest_progress(paths["l2"])
    assert l1["message_pc_active_learned_params"] < l2[
        "message_pc_active_learned_params"
    ]
    assert l2["message_pc_active_learned_params"] < 7_400_000
    for row in (l1, l2):
        assert row["global_backprops"] == 0.0
        assert row["message_pc_residual_error_uses_learned_only_transport"] == 1.0
        assert row[
            "update_diag_message_pc_residual_phase_transport_refine_passes"
        ] == 1.0
        assert row[
            "update_diag_message_pc_residual_error_exact_transport_signal_rms_layer_1"
        ] == 0.0
    assert l2[
        "update_diag_message_pc_residual_error_exact_transport_signal_rms_layer_2"
    ] == 0.0
    assert l2["val_bpb"] < l1["val_bpb"]
    assert l2["accuracy"] < l1["accuracy"]


def test_phase_transport_refine2_rejected_despite_upper_cosine_gain() -> None:
    root = Path("runs/pc_modal_lm/nanochat_exact_seq1024")
    refine1_path = (
        root
        / "residual_error_learned_only_phase_transport_refine1_100_20260715"
        / "l2_d384"
        / "results.jsonl"
    )
    refine2_path = (
        root
        / "residual_error_learned_only_phase_transport_refine2_100_20260715"
        / "l2_d384"
        / "results.jsonl"
    )
    if not refine1_path.exists() or not refine2_path.exists():
        pytest.skip("phase transport refine2 artifacts are not synced")

    def latest_progress(path: Path) -> dict[str, object]:
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip().startswith("{")
        ]
        progress = [row for row in rows if row.get("event") == "train_progress"]
        assert progress
        return progress[-1]

    refine1 = latest_progress(refine1_path)
    refine2 = latest_progress(refine2_path)
    assert refine2[
        "update_diag_message_pc_residual_phase_transport_refine_passes"
    ] == 2.0
    assert refine2["global_backprops"] == 0.0
    assert refine2["val_bpb"] > refine1["val_bpb"]
    assert (
        refine2[
            "update_diag_message_pc_residual_phase_transport_cosine_layer_1"
        ]
        > refine1[
            "update_diag_message_pc_residual_phase_transport_cosine_layer_1"
        ]
    )


def test_phase_transport_refine_target_modes_record_bpb_accuracy_tradeoff() -> None:
    root = Path("runs/pc_modal_lm/nanochat_exact_seq1024")
    paths = {
        "all_tokens": (
            root
            / "residual_error_learned_only_phase_transport_refine1_100_20260715"
            / "l2_d384"
            / "results.jsonl"
        ),
        "mistakes": (
            root
            / "residual_error_learned_only_phase_transport_refine1_mistakes_100_20260715"
            / "l2_d384"
            / "results.jsonl"
        ),
        "fixed_outcome": (
            root
            / "residual_error_learned_only_phase_transport_refine1_fixed_outcome_100_20260715"
            / "l2_d384"
            / "results.jsonl"
        ),
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("phase transport target-mode artifacts are not synced")

    def latest_progress(path: Path) -> dict[str, object]:
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip().startswith("{")
        ]
        progress = [row for row in rows if row.get("event") == "train_progress"]
        assert progress
        return progress[-1]

    rows = {name: latest_progress(path) for name, path in paths.items()}
    for row in rows.values():
        assert row["global_backprops"] == 0.0
        assert row[
            "update_diag_message_pc_residual_phase_transport_refine_passes"
        ] == 1.0
        assert row[
            "update_diag_message_pc_residual_error_exact_transport_signal_rms_layer_1"
        ] == 0.0
        assert row[
            "update_diag_message_pc_residual_error_exact_transport_signal_rms_layer_2"
        ] == 0.0

    assert rows["mistakes"]["val_bpb"] < rows["all_tokens"]["val_bpb"]
    assert rows["mistakes"]["accuracy"] < rows["all_tokens"]["accuracy"]
    assert rows["fixed_outcome"]["accuracy"] > rows["all_tokens"]["accuracy"]
    assert rows["fixed_outcome"]["val_bpb"] > rows["all_tokens"]["val_bpb"]


def test_phase_transport_refine_token_feedback_sweep_finds_stronger_target() -> None:
    root = Path("runs/pc_modal_lm/nanochat_exact_seq1024")
    paths = {
        "tf1": (
            root
            / "residual_error_learned_only_phase_transport_refine1_100_20260715"
            / "l2_d384"
            / "results.jsonl"
        ),
        "tf15": (
            root
            / "residual_error_learned_only_phase_transport_refine1_tf_sweep_100_20260715"
            / "tf15_l2_d384"
            / "results.jsonl"
        ),
        "tf20": (
            root
            / "residual_error_learned_only_phase_transport_refine1_tf20_100_20260715"
            / "l2_d384"
            / "results.jsonl"
        ),
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("phase transport token-feedback artifacts are not synced")

    def latest_progress(path: Path) -> dict[str, object]:
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip().startswith("{")
        ]
        progress = [row for row in rows if row.get("event") == "train_progress"]
        assert progress
        return progress[-1]

    rows = {name: latest_progress(path) for name, path in paths.items()}
    for row in rows.values():
        assert row["global_backprops"] == 0.0
        assert row[
            "update_diag_message_pc_residual_phase_transport_refine_passes"
        ] == 1.0
    assert rows["tf15"]["val_bpb"] < rows["tf1"]["val_bpb"]
    assert rows["tf15"]["accuracy"] > rows["tf1"]["accuracy"]
    assert rows["tf15"][
        "update_diag_message_pc_residual_error_learned_transport_signal_rms_layer_2"
    ] > rows["tf1"][
        "update_diag_message_pc_residual_error_learned_transport_signal_rms_layer_2"
    ]
    assert rows["tf20"]["accuracy"] > rows["tf15"]["accuracy"]
    assert rows["tf20"]["val_bpb"] > rows["tf15"]["val_bpb"]


def test_phase_transport_refine_tf15_depth_pair_records_bpb_only_depth_gain() -> None:
    root = Path("runs/pc_modal_lm/nanochat_exact_seq1024")
    paths = {
        "l1": (
            root
            / "residual_error_learned_only_phase_transport_refine1_tf15_100_20260715"
            / "l1_d448"
            / "results.jsonl"
        ),
        "l2": (
            root
            / "residual_error_learned_only_phase_transport_refine1_tf_sweep_100_20260715"
            / "tf15_l2_d384"
            / "results.jsonl"
        ),
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("phase transport tf1.5 depth artifacts are not synced")

    def latest_progress(path: Path) -> dict[str, object]:
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip().startswith("{")
        ]
        progress = [row for row in rows if row.get("event") == "train_progress"]
        assert progress
        return progress[-1]

    l1 = latest_progress(paths["l1"])
    l2 = latest_progress(paths["l2"])
    assert l1["message_pc_active_learned_params"] < l2[
        "message_pc_active_learned_params"
    ]
    for row in (l1, l2):
        assert row["global_backprops"] == 0.0
        assert row[
            "update_diag_message_pc_residual_phase_transport_refine_passes"
        ] == 1.0
        assert row[
            "update_diag_message_pc_residual_error_exact_transport_signal_rms_layer_1"
        ] == 0.0
    assert l2[
        "update_diag_message_pc_residual_error_exact_transport_signal_rms_layer_2"
    ] == 0.0
    assert l2["val_bpb"] < l1["val_bpb"]
    assert l2["accuracy"] < l1["accuracy"]


def test_phase_transport_refine_tf15_depth_pair_1k_records_depth_failure() -> None:
    root = (
        Path("runs/pc_modal_lm/nanochat_exact_seq1024")
        / "residual_error_learned_only_phase_transport_refine1_tf15_depth_pair_1k_20260715"
    )
    paths = {
        "l1": root / "l1_d448" / "results.jsonl",
        "l2": root / "l2_d384" / "results.jsonl",
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("phase transport tf1.5 1k artifacts are not synced")

    def latest_progress(path: Path) -> dict[str, object]:
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip().startswith("{")
        ]
        progress = [row for row in rows if row.get("event") == "train_progress"]
        assert progress
        assert progress[-1]["step"] == 1000.0
        return progress[-1]

    l1 = latest_progress(paths["l1"])
    l2 = latest_progress(paths["l2"])
    assert l1["message_pc_active_learned_params"] < l2[
        "message_pc_active_learned_params"
    ]
    for row in (l1, l2):
        assert row["global_backprops"] == 0.0
        assert row[
            "update_diag_message_pc_residual_phase_transport_refine_passes"
        ] == 1.0
    assert l2["val_bpb"] > l1["val_bpb"]
    assert l2["accuracy"] < l1["accuracy"]
    assert (
        l2["update_diag_message_pc_residual_phase_transport_cosine_layer_1"]
        > 0.4
    )


def test_phase_transport_refine_tf15_suffix_decay_rejected_at_1k() -> None:
    def latest_progress(path: Path) -> dict[str, object]:
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip().startswith("{")
        ]
        progress = [row for row in rows if row.get("event") == "train_progress"]
        assert progress
        return progress[-1]

    base = (
        Path("runs/pc_modal_lm/nanochat_exact_seq1024")
        / "residual_error_learned_only_phase_transport_refine1_tf15_suffix_decay_100_20260715"
    )
    paths = {
        "sd050": base / "sd050_l2_d384" / "results.jsonl",
        "sd025": base / "sd025_l2_d384" / "results.jsonl",
        "parent": Path("runs/pc_modal_lm/nanochat_exact_seq1024")
        / "residual_error_learned_only_phase_transport_refine1_tf_sweep_100_20260715"
        / "tf15_l2_d384"
        / "results.jsonl",
        "full": Path("runs/pc_modal_lm/nanochat_exact_seq1024")
        / "residual_error_learned_only_phase_transport_refine1_tf15_sd050_l2_1k_20260715"
        / "l2_d384"
        / "results.jsonl",
        "full_parent": Path("runs/pc_modal_lm/nanochat_exact_seq1024")
        / "residual_error_learned_only_phase_transport_refine1_tf15_depth_pair_1k_20260715"
        / "l2_d384"
        / "results.jsonl",
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("phase transport suffix-decay artifacts are not synced")

    sd050 = latest_progress(paths["sd050"])
    sd025 = latest_progress(paths["sd025"])
    parent = latest_progress(paths["parent"])
    full = latest_progress(paths["full"])
    full_parent = latest_progress(paths["full_parent"])

    assert sd050["update_diag_message_pc_residual_error_suffix_decay"] == 0.5
    assert sd050["val_bpb"] < parent["val_bpb"]
    assert sd050["accuracy"] > parent["accuracy"]
    assert sd025["update_diag_message_pc_residual_error_suffix_decay"] == 0.25
    assert sd025["val_bpb"] > sd050["val_bpb"]
    assert full["step"] == 1000.0
    assert full["update_diag_message_pc_residual_error_suffix_decay"] == 0.5
    assert full["val_bpb"] > full_parent["val_bpb"]
    assert full["accuracy"] < full_parent["accuracy"]


def test_phase_transport_refine_tf15_fast_window_rejected_at_100() -> None:
    def latest_progress(path: Path) -> dict[str, object]:
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip().startswith("{")
        ]
        progress = [row for row in rows if row.get("event") == "train_progress"]
        assert progress
        return progress[-1]

    root = (
        Path("runs/pc_modal_lm/nanochat_exact_seq1024")
        / "residual_error_learned_only_phase_transport_refine1_tf15_fast_window_100_20260715"
    )
    paths = {
        "fw064": root / "fw064_l2_d384" / "results.jsonl",
        "fw128": root / "fw128_l2_d384" / "results.jsonl",
        "fw256": root / "fw256_l2_d384" / "results.jsonl",
        "parent": Path("runs/pc_modal_lm/nanochat_exact_seq1024")
        / "residual_error_learned_only_phase_transport_refine1_tf_sweep_100_20260715"
        / "tf15_l2_d384"
        / "results.jsonl",
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("phase transport fast-window artifacts are not synced")

    parent = latest_progress(paths["parent"])
    rows = {name: latest_progress(path) for name, path in paths.items()}
    for name, window in (("fw064", 64.0), ("fw128", 128.0), ("fw256", 256.0)):
        row = rows[name]
        assert row["message_pc_fast_attention_window"] == window
        assert row["message_pc_fast_attention_layers"] == 1.0
        assert row["val_bpb"] > parent["val_bpb"]
        assert row["accuracy"] < parent["accuracy"]
    assert rows["fw256"]["val_bpb"] < rows["fw128"]["val_bpb"]
    assert rows["fw128"]["val_bpb"] < rows["fw064"]["val_bpb"]


def test_phase_transport_refine_tf15_persistent_context_rejected_at_100() -> None:
    def latest_progress(path: Path) -> dict[str, object]:
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip().startswith("{")
        ]
        progress = [row for row in rows if row.get("event") == "train_progress"]
        assert progress
        return progress[-1]

    root = (
        Path("runs/pc_modal_lm/nanochat_exact_seq1024")
        / "residual_error_learned_only_phase_transport_refine1_tf15_persistent_context_100_20260715"
    )
    paths = {
        "soft_ms050": root / "soft_ms050_l2_d384" / "results.jsonl",
        "soft_ms100": root / "soft_ms100_l2_d384" / "results.jsonl",
        "hard_ms050": root / "hard_ms050_l2_d384" / "results.jsonl",
        "hard_ms100": root / "hard_ms100_l2_d384" / "results.jsonl",
        "parent": Path("runs/pc_modal_lm/nanochat_exact_seq1024")
        / "residual_error_learned_only_phase_transport_refine1_tf_sweep_100_20260715"
        / "tf15_l2_d384"
        / "results.jsonl",
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("phase transport persistent-context artifacts are not synced")

    parent = latest_progress(paths["parent"])
    rows = {name: latest_progress(path) for name, path in paths.items()}
    expected = {
        "soft_ms050": "pctx_soft_ms050",
        "soft_ms100": "pctx_soft_ms100",
        "hard_ms050": "pctx_hard_ms050",
        "hard_ms100": "pctx_hard_ms100",
    }
    for name, run_tag_fragment in expected.items():
        row = rows[name]
        assert run_tag_fragment in row["run_tag"]
        assert row["val_bpb"] > parent["val_bpb"]
        assert row["update_diag_message_pc_context_update_rms"] > 0
        assert (
            row["update_diag_message_pc_residual_phase_transport_cosine_layer_1"]
            < parent["update_diag_message_pc_residual_phase_transport_cosine_layer_1"]
        )
    assert rows["hard_ms100"]["accuracy"] > parent["accuracy"]
    assert rows["soft_ms100"]["accuracy"] > parent["accuracy"]


def residual_error_config(**overrides: object) -> MessagePassingPCConfig:
    values: dict[str, object] = {
        "inference_mode": "residual_error_pc",
        "block_state_mode": "identity_residual",
        "decoder_supervision_mode": "top_only",
        "readout_layer": 0,
        "state_delta_rms_cap": 0.0,
        "energy_backtracking_steps": 8,
        "route_update_mode": "local_energy",
    }
    values.update(overrides)
    return small_config(**values)


def test_residual_error_suffix_decay_damps_deeper_residuals() -> None:
    model = MessagePassingPCLanguageModel(
        residual_error_config(num_layers=3, residual_error_suffix_decay=0.5)
    )
    residuals = [
        torch.full((1, 1, model.config.hidden_dim), float(index))
        for index in range(4)
    ]

    signals = model._residual_error_suffix_signals(residuals)

    assert torch.allclose(
        signals[2],
        residuals[3],
    )
    assert torch.allclose(
        signals[1],
        residuals[2] + 0.5 * residuals[3],
    )
    assert torch.allclose(
        signals[0],
        residuals[1] + 0.5 * residuals[2] + 0.25 * residuals[3],
    )


def test_residual_error_pc_predicts_error_not_adjacent_state() -> None:
    torch.manual_seed(107)
    model = MessagePassingPCLanguageModel(residual_error_config())
    model.feedback_weights[0].zero_()
    state = torch.randn(2, 5, model.config.hidden_dim)

    _, prediction = model._residual_error_prediction(0, state)

    assert torch.count_nonzero(prediction) == 0


def test_residual_error_pc_stats_do_not_claim_meta_exact_vjp() -> None:
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            hierarchical_error_prediction_mode="direct_transport",
            learned_feedback_proposal_scale=1.0,
            residual_error_transport_mode="learned_only",
            meta_pc_output_feedback_mode="kp_learned",
        )
    )

    stats = model.connection_stats()

    assert stats["message_pc_uses_residual_error_pc"] == 1.0
    assert stats["message_pc_meta_uses_exact_vjp"] == 0.0
    assert stats["message_pc_meta_uses_kp_output_feedback"] == 1.0
    assert stats["message_pc_residual_error_uses_learned_only_transport"] == 1.0


def test_residual_error_pc_direct_predictor_starts_as_small_local_map() -> None:
    model = MessagePassingPCLanguageModel(
        residual_error_config(hierarchical_error_prediction_mode="direct")
    )
    state = torch.randn(2, 5, model.config.hidden_dim)

    cache, prediction = model._residual_error_prediction(0, state)

    assert cache is None
    assert 0 < prediction.square().mean().sqrt() < 0.1
    assert len(model.inverse_query_weights) == 0
    assert len(model.inverse_key_weights) == 0
    assert torch.count_nonzero(model.feedback_weights[0]) == 0


def test_residual_error_pc_direct_local_math_matches_energy_oracle() -> None:
    torch.manual_seed(108)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            num_layers=2,
            hierarchical_error_prediction_mode="direct",
        )
    )
    for weight in model.generative_weights:
        weight.normal_(std=0.05)
    x = torch.randint(0, 31, (2, 5))
    free_states, _ = model._forward_states(x, with_cache=False)
    states = [free_states[0]] + [
        (state + 0.05 * torch.randn_like(state)).requires_grad_()
        for state in free_states[1:]
    ]
    target = free_states[-1] + 0.1 * torch.randn_like(free_states[-1])
    parameters = tuple(model.generative_weights)
    for parameter in parameters:
        parameter.requires_grad_(True)

    energy = model._residual_error_pc_energy(states, free_states, target)
    expected = torch.autograd.grad(
        energy,
        (*states[1:], *parameters),
    )
    expected_state = expected[: model.config.num_layers]
    expected_parameters = expected[model.config.num_layers :]
    actual_state, _, _ = model._residual_error_pc_state_deltas(
        states,
        free_states,
        target,
    )
    actual_parameters = model._residual_error_pc_predictor_parameter_gradients(
        states,
        target,
        token_count=float(x.numel()),
    )
    token_count = float(x.numel())

    for actual, expected in zip(actual_state, expected_state):
        assert torch.allclose(
            actual,
            -token_count * expected,
            atol=4e-6,
            rtol=4e-5,
        )
    for layer, expected in enumerate(expected_parameters):
        assert torch.allclose(
            actual_parameters[f"generative_weights.{layer}"],
            expected,
            atol=4e-6,
            rtol=4e-5,
        )


def test_residual_error_pc_direct_transport_local_vjps_match_oracle() -> None:
    torch.manual_seed(110)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            hierarchical_error_prediction_mode="direct_transport",
            learned_feedback_proposal_scale=1.0,
        )
    )
    state = torch.randn(
        2,
        5,
        model.config.hidden_dim,
        requires_grad=True,
    )
    target = torch.randn_like(state)
    parameter_names = (
        "inverse_query_weights.0",
        "inverse_key_weights.0",
        "feedback_weights.0",
    )
    parameters = dict(model.named_parameters())
    for name in parameter_names:
        parameters[name].requires_grad_(True)
    cache = model._residual_transport_block(0, state)
    energy = 0.5 * (
        cache.prediction - target
    ).square().sum(dim=-1).mean()
    expected = torch.autograd.grad(
        energy,
        (state, *(parameters[name] for name in parameter_names)),
    )
    output_signal = cache.prediction - target
    actual_input = model._residual_transport_input_signal(
        0,
        state,
        cache,
        output_signal,
    )
    actual_parameters = model._residual_transport_parameter_gradients(
        0,
        cache,
        output_signal,
        token_count=float(state.shape[0] * state.shape[1]),
    )
    token_count = float(state.shape[0] * state.shape[1])

    assert torch.allclose(
        actual_input,
        token_count * expected[0],
        atol=4e-6,
        rtol=4e-5,
    )
    for name, expected_gradient in zip(parameter_names, expected[1:]):
        assert torch.allclose(
            actual_parameters[name],
            expected_gradient,
            atol=4e-6,
            rtol=4e-5,
        )


def test_residual_error_pc_direct_transport_routes_future_error_backward() -> None:
    torch.manual_seed(112)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            hierarchical_error_prediction_mode="direct_transport",
            learned_feedback_proposal_scale=1.0,
        )
    )
    state = torch.randn(1, 6, model.config.hidden_dim)
    output_signal = torch.zeros_like(state)
    output_signal[:, -1] = torch.randn_like(output_signal[:, -1])
    cache = model._residual_transport_block(0, state)

    transported = model._residual_transport_input_signal(
        0,
        state,
        cache,
        output_signal,
    ) - output_signal

    assert torch.count_nonzero(transported[:, :-1]) > 0
    assert len(model.inverse_query_weights) == model.config.num_layers
    assert len(model.inverse_key_weights) == model.config.num_layers


def test_residual_error_pc_learned_only_transport_skips_predictor_transpose() -> None:
    torch.manual_seed(154)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            hierarchical_error_prediction_mode="direct_transport",
            learned_feedback_proposal_scale=0.5,
            residual_error_transport_mode="learned_only",
        )
    )
    state = torch.randn(2, 5, model.config.hidden_dim)
    output_signal = torch.randn_like(state)
    for weight in (
        model.inverse_query_weights[0],
        model.inverse_key_weights[0],
        model.feedback_weights[0],
    ):
        weight.normal_(std=0.05)
    with torch.no_grad():
        model.generative_weights[0].normal_(std=0.05)
    cache_a = model._residual_transport_block(0, state)
    expected = 0.5 * (
        model._residual_transport_input_signal(
            0,
            state,
            cache_a,
            output_signal,
        )
        - output_signal
    )
    actual_a = model._residual_error_input_signal(
        0,
        state,
        None,
        output_signal,
    )
    with torch.no_grad():
        model.generative_weights[0].normal_(std=1.0)
    cache_b = model._residual_transport_block(0, state)
    actual_b = model._residual_error_input_signal(
        0,
        state,
        None,
        output_signal,
    )

    assert torch.allclose(actual_a, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_b, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cache_b.prediction, cache_a.prediction)


def test_residual_error_pc_phase_transport_local_gradients_match_oracle() -> None:
    torch.manual_seed(114)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            hierarchical_error_prediction_mode="direct_phase_transport",
            learned_feedback_proposal_scale=1.0,
        )
    )
    state = torch.randn(2, 5, model.config.hidden_dim)
    error_signal = torch.randn_like(state)
    target = torch.randn_like(state)
    parameter_names = (
        "inverse_query_weights.0",
        "inverse_key_weights.0",
        "feedback_weights.0",
    )
    parameters = dict(model.named_parameters())
    for name in parameter_names:
        parameters[name].requires_grad_(True)
    cache = model._residual_phase_transport_block(0, state, error_signal)
    energy = 0.5 * (
        cache.prediction - target
    ).square().sum(dim=-1).mean()
    expected = torch.autograd.grad(
        energy,
        tuple(parameters[name] for name in parameter_names),
    )
    actual = model._residual_phase_transport_parameter_gradients(
        0,
        cache,
        cache.prediction - target,
        token_count=float(state.shape[0] * state.shape[1]),
    )

    for name, expected_gradient in zip(parameter_names, expected):
        assert torch.allclose(
            actual[name],
            expected_gradient,
            atol=4e-6,
            rtol=4e-5,
        )


def test_residual_error_pc_phase_transport_routes_future_activity() -> None:
    torch.manual_seed(116)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            hierarchical_error_prediction_mode="direct_phase_transport",
            learned_feedback_proposal_scale=1.0,
        )
    )
    state = torch.randn(1, 6, model.config.hidden_dim)
    error_signal = torch.zeros_like(state)
    error_signal[:, -1] = torch.randn_like(error_signal[:, -1])

    cache = model._residual_phase_transport_block(0, state, error_signal)

    assert torch.count_nonzero(cache.prediction[:, :-1]) > 0


def test_residual_error_pc_learned_only_phase_transport_skips_predictor_transpose() -> None:
    torch.manual_seed(155)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            hierarchical_error_prediction_mode="direct_phase_transport",
            learned_feedback_proposal_scale=0.25,
            residual_error_transport_mode="learned_only",
        )
    )
    state = torch.randn(2, 5, model.config.hidden_dim)
    output_signal = torch.randn_like(state)
    for weight in (
        model.inverse_query_weights[0],
        model.inverse_key_weights[0],
        model.feedback_weights[0],
    ):
        weight.normal_(std=0.05)
    expected = 0.25 * model._residual_phase_transport_block(
        0,
        state,
        output_signal,
    ).prediction
    with torch.no_grad():
        model.generative_weights[0].normal_(std=0.05)
    actual_a = model._residual_error_input_signal(
        0,
        state,
        None,
        output_signal,
    )
    with torch.no_grad():
        model.generative_weights[0].normal_(std=1.0)
    actual_b = model._residual_error_input_signal(
        0,
        state,
        None,
        output_signal,
    )

    assert torch.allclose(actual_a, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_b, expected, atol=1e-6, rtol=1e-6)


def test_residual_error_pc_passes_unexplained_residual_upward() -> None:
    torch.manual_seed(109)
    model = MessagePassingPCLanguageModel(residual_error_config())
    x = torch.randint(0, 31, (2, 5))
    states, _ = model._forward_states(x, with_cache=False)
    target = states[-1] + 0.1 * torch.randn_like(states[-1])

    _, predictions, residuals = model._residual_error_pc_terms(
        states,
        target,
    )

    assert torch.allclose(residuals[0], target - states[-1])
    for prediction, incoming, outgoing in zip(
        predictions,
        residuals,
        residuals[1:],
    ):
        assert torch.allclose(outgoing, incoming - prediction)


def test_residual_error_pc_state_deltas_match_energy_oracle() -> None:
    torch.manual_seed(113)
    model = MessagePassingPCLanguageModel(residual_error_config())
    for feedback_weight in model.feedback_weights:
        feedback_weight.normal_(std=0.1)
    x = torch.randint(0, 31, (2, 5))
    free_states, _ = model._forward_states(x, with_cache=False)
    states = [free_states[0]] + [
        (state + 0.05 * torch.randn_like(state)).requires_grad_()
        for state in free_states[1:]
    ]
    target = free_states[-1] + 0.1 * torch.randn_like(free_states[-1])

    energy = model._residual_error_pc_energy(
        states,
        free_states,
        target,
    )
    expected_gradients = torch.autograd.grad(energy, tuple(states[1:]))
    actual_deltas, _, _ = model._residual_error_pc_state_deltas(
        states,
        free_states,
        target,
    )
    token_count = float(x.numel())

    for actual, expected in zip(actual_deltas, expected_gradients):
        assert torch.allclose(
            actual,
            -token_count * expected,
            atol=4e-6,
            rtol=4e-5,
        )


def test_residual_error_pc_predictor_gradients_match_energy_oracle() -> None:
    torch.manual_seed(127)
    model = MessagePassingPCLanguageModel(residual_error_config(num_layers=2))
    for feedback_weight in model.feedback_weights:
        feedback_weight.normal_(std=0.1)
    x = torch.randint(0, 31, (2, 5))
    states, _ = model._forward_states(x, with_cache=False)
    target = states[-1].detach() + 0.1 * torch.randn_like(states[-1])
    parameter_names = tuple(
        name
        for layer in range(model.config.num_layers)
        for name in (
            f"inverse_query_weights.{layer}",
            f"inverse_key_weights.{layer}",
            f"generative_weights.{layer}",
            f"feedback_weights.{layer}",
        )
    )
    parameters = dict(model.named_parameters())
    for name in parameter_names:
        parameters[name].requires_grad_(True)

    energy = model._residual_error_pc_energy(
        states,
        states,
        target,
    )
    expected_values = torch.autograd.grad(
        energy,
        tuple(parameters[name] for name in parameter_names),
    )
    actual = model._residual_error_pc_predictor_parameter_gradients(
        states,
        target,
        token_count=float(x.numel()),
    )

    for name, expected in zip(parameter_names, expected_values):
        assert torch.allclose(
            actual[name],
            expected,
            atol=4e-6,
            rtol=4e-5,
        )


def test_residual_error_pc_settling_decreases_energy() -> None:
    torch.manual_seed(139)
    model = MessagePassingPCLanguageModel(
        residual_error_config(inference_steps=8, inference_lr=0.5)
    )
    for feedback_weight in model.feedback_weights:
        feedback_weight.normal_(std=0.1)
    x = torch.randint(0, 31, (2, 5))
    free_states, _ = model._forward_states(x, with_cache=False)
    target = free_states[-1] + 0.1 * torch.randn_like(free_states[-1])
    before = model._residual_error_pc_energy(
        free_states,
        free_states,
        target,
    )

    settled, diagnostics = model._settle_residual_error_pc(
        free_states,
        target,
        inference_lr=0.5,
    )
    after = model._residual_error_pc_energy(
        settled,
        free_states,
        target,
    )

    assert after < before
    assert diagnostics[
        "update_diag_message_pc_residual_error_energy_reduction"
    ] > 0
    assert diagnostics[
        "update_diag_message_pc_residual_error_monotone_energy"
    ] == 1.0


def test_residual_error_pc_learned_only_transport_settling_lowers_declared_energy() -> None:
    torch.manual_seed(160)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            num_layers=2,
            inference_steps=4,
            inference_lr=0.1,
            hierarchical_error_prediction_mode="direct_transport",
            learned_feedback_proposal_scale=1.0,
            residual_error_transport_mode="learned_only",
        )
    )
    for weight in (
        *model.inverse_query_weights,
        *model.inverse_key_weights,
        *model.feedback_weights,
    ):
        weight.normal_(std=0.05)
    x = torch.randint(0, 31, (2, 5))
    free_states, _ = model._forward_states(x, with_cache=False)
    target = free_states[-1] + 0.1 * torch.randn_like(free_states[-1])
    before = model._residual_error_pc_energy(
        free_states,
        free_states,
        target,
    )

    settled, diagnostics = model._settle_residual_error_pc(
        free_states,
        target,
        inference_lr=0.1,
    )
    after = model._residual_error_pc_energy(
        settled,
        free_states,
        target,
    )

    assert after < before
    assert diagnostics[
        "update_diag_message_pc_residual_error_accepted_steps"
    ] == 4.0
    assert diagnostics[
        "update_diag_message_pc_residual_error_monotone_energy"
    ] == 1.0
    for layer in range(1, model.config.num_layers + 1):
        assert diagnostics[
            f"update_diag_message_pc_residual_error_exact_transport_signal_rms_layer_{layer}"
        ] == 0.0
        assert diagnostics[
            f"update_diag_message_pc_residual_error_learned_transport_signal_rms_layer_{layer}"
        ] > 0.0
        assert diagnostics[
            f"update_diag_message_pc_residual_error_learned_transport_signal_ratio_layer_{layer}"
        ] == 1.0


def test_residual_error_pc_token_target_moves_every_latent_layer() -> None:
    torch.manual_seed(131)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            num_layers=4,
            inference_steps=8,
            inference_lr=0.2,
        )
    )
    for feedback_weight in model.feedback_weights:
        feedback_weight.normal_(std=0.1)
    x = torch.randint(0, 31, (2, 5))
    free_states, _ = model._forward_states(x, with_cache=False)
    target = free_states[-1] + 0.1 * torch.randn_like(free_states[-1])

    settled, _ = model._settle_residual_error_pc(
        free_states,
        target,
        inference_lr=0.2,
    )

    assert all(
        torch.count_nonzero(settled_state - free_state) > 0
        for settled_state, free_state in zip(
            settled[1:],
            free_states[1:],
        )
    )


def test_residual_error_pc_base_target_anchors_lower_residual() -> None:
    torch.manual_seed(191)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            num_layers=3,
            residual_error_target_layer="base",
            inference_steps=1,
            inference_lr=0.2,
            state_delta_rms_cap=1.0,
        )
    )
    x = torch.randint(0, 31, (2, 5))
    free_states, _ = model._forward_states(x, with_cache=False)
    target = free_states[1] + 0.1 * torch.randn_like(free_states[1])

    settled, diagnostics = model._settle_residual_error_pc(
        free_states,
        target,
        inference_lr=0.2,
        target_state_index=1,
    )

    assert diagnostics["update_diag_message_pc_residual_error_target_layer_index"] == 1.0
    assert torch.count_nonzero(settled[1] - free_states[1]) > 0


def test_residual_error_proxy_memory_is_causal_previous_error_target() -> None:
    torch.manual_seed(193)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            residual_error_proxy_memory_scale=0.5,
            residual_error_proxy_memory_decay=0.25,
        )
    )
    token_delta = torch.randn(2, 5, model.config.hidden_dim)

    memory = model._residual_error_proxy_memory(token_delta)
    changed_future = token_delta.clone()
    changed_future[:, 3].add_(10.0)
    changed_memory = model._residual_error_proxy_memory(changed_future)

    assert torch.count_nonzero(memory[:, 0]) == 0
    assert torch.allclose(
        memory[:, 1],
        0.5 * token_delta[:, 0],
    )
    assert torch.allclose(
        memory[:, 2],
        0.5 * (token_delta[:, 1] + 0.25 * token_delta[:, 0]),
    )
    assert torch.allclose(memory[:, :4], changed_memory[:, :4])
    assert not torch.allclose(memory[:, 4], changed_memory[:, 4])


def test_residual_error_upper_proxy_state_is_causal_previous_error_target() -> None:
    torch.manual_seed(195)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            residual_error_upper_proxy_state_scale=0.25,
            residual_error_upper_proxy_state_decay=0.5,
        )
    )
    token_delta = torch.randn(2, 5, model.config.hidden_dim)

    upper = model._residual_error_upper_proxy_state_delta(token_delta)
    changed_future = token_delta.clone()
    changed_future[:, 3].sub_(10.0)
    changed_upper = model._residual_error_upper_proxy_state_delta(changed_future)

    assert torch.count_nonzero(upper[:, 0]) == 0
    assert torch.allclose(upper[:, 1], 0.25 * token_delta[:, 0])
    assert torch.allclose(
        upper[:, 2],
        0.25 * (token_delta[:, 1] + 0.5 * token_delta[:, 0]),
    )
    assert torch.allclose(upper[:, :4], changed_upper[:, :4])
    assert not torch.allclose(upper[:, 4], changed_upper[:, 4])


def test_residual_error_upper_content_state_is_causal_previous_input() -> None:
    torch.manual_seed(196)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            residual_error_upper_content_state_scale=0.25,
            residual_error_upper_content_state_decay=0.5,
        )
    )
    x = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    content = _rms_norm(model.embedding(x))

    upper = model._residual_error_upper_content_state_delta(x)
    changed_future = x.clone()
    changed_future[:, 3] = 7
    changed_upper = model._residual_error_upper_content_state_delta(
        changed_future
    )

    assert torch.count_nonzero(upper[:, 0]) == 0
    assert torch.allclose(upper[:, 1], 0.25 * content[:, 0])
    assert torch.allclose(
        upper[:, 2],
        0.25 * (content[:, 1] + 0.5 * content[:, 0]),
    )
    assert torch.allclose(upper[:, :4], changed_upper[:, :4])
    assert not torch.allclose(upper[:, 4], changed_upper[:, 4])


def test_residual_error_upper_content_state_can_select_delayed_operand() -> None:
    torch.manual_seed(197)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            residual_error_upper_content_state_scale=0.5,
            residual_error_upper_content_state_decay=0.0,
            residual_error_upper_content_write_period=3,
            residual_error_upper_content_write_offset=0,
            residual_error_upper_content_read_period=3,
            residual_error_upper_content_read_offset=2,
            residual_error_upper_content_read_delay=3,
        )
    )
    x = torch.tensor([[1, 2, 7, 3, 4, 7, 5, 6, 7]])
    content = _rms_norm(model.embedding(x))

    upper = model._residual_error_upper_content_state_delta(x)
    changed_current_left = x.clone()
    changed_current_left[:, 3] = 7
    changed_upper = model._residual_error_upper_content_state_delta(
        changed_current_left
    )

    assert torch.count_nonzero(upper[:, :5]) == 0
    assert torch.allclose(upper[:, 5], 0.5 * content[:, 0])
    assert torch.allclose(upper[:, 8], 0.5 * content[:, 3])
    assert torch.allclose(upper[:, 5], changed_upper[:, 5])
    assert not torch.allclose(upper[:, 8], changed_upper[:, 8])
    assert torch.count_nonzero(upper[:, 6:8]) == 0


def test_residual_error_upper_proxy_state_anchors_top_state() -> None:
    torch.manual_seed(196)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            num_layers=2,
            residual_error_target_layer="base",
            residual_error_upper_proxy_state_scale=0.5,
            residual_error_upper_proxy_state_decay=0.5,
            inference_steps=1,
            inference_lr=0.2,
            state_delta_rms_cap=1.0,
        )
    )
    x = torch.randint(0, 31, (2, 6))
    free_states, _ = model._forward_states(x, with_cache=False)
    token_delta = torch.randn_like(free_states[1])
    target_state = free_states[1] + 0.1 * token_delta
    upper_delta = model._residual_error_upper_proxy_state_delta(token_delta)
    upper_target = free_states[-1] + upper_delta

    settled, diagnostics = model._settle_residual_error_pc(
        free_states,
        target_state,
        inference_lr=0.2,
        target_state_index=1,
        upper_proxy_target_state=upper_target,
    )

    assert diagnostics[
        "update_diag_message_pc_residual_error_upper_proxy_target_rms"
    ] > 0.0
    assert diagnostics[
        "update_diag_message_pc_residual_error_upper_proxy_state_delta_rms"
    ] > 0.0
    assert torch.count_nonzero(settled[-1] - free_states[-1]) > 0


def test_residual_error_local_update_reports_proxy_memory_target() -> None:
    torch.manual_seed(197)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            residual_error_target_layer="base",
            residual_error_proxy_memory_scale=0.5,
            residual_error_proxy_memory_decay=0.25,
            inference_steps=1,
        )
    )
    x = torch.randint(0, 31, (2, 5))
    y = (x + 1) % 31

    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
    )

    diagnostics = model._last_update_diagnostics
    assert diagnostics[
        "update_diag_message_pc_residual_error_proxy_memory_scale"
    ] == 0.5
    assert diagnostics[
        "update_diag_message_pc_residual_error_proxy_memory_decay"
    ] == 0.25
    assert diagnostics[
        "update_diag_message_pc_residual_error_proxy_memory_rms"
    ] > 0.0


def test_residual_error_local_update_reports_upper_proxy_state_target() -> None:
    torch.manual_seed(198)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            residual_error_target_layer="base",
            residual_error_upper_proxy_state_scale=0.5,
            residual_error_upper_proxy_state_decay=0.5,
            inference_steps=1,
        )
    )
    x = torch.randint(0, 31, (2, 5))
    y = (x + 1) % 31

    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
    )

    diagnostics = model._last_update_diagnostics
    assert diagnostics[
        "update_diag_message_pc_residual_error_upper_proxy_state_scale"
    ] == 0.5
    assert diagnostics[
        "update_diag_message_pc_residual_error_upper_proxy_state_decay"
    ] == 0.5
    assert diagnostics[
        "update_diag_message_pc_residual_error_upper_proxy_state_input_rms"
    ] > 0.0


def test_residual_error_local_update_reports_upper_content_state_target() -> None:
    torch.manual_seed(199)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            residual_error_target_layer="base",
            residual_error_upper_content_state_scale=0.5,
            residual_error_upper_content_state_decay=0.5,
            residual_error_upper_content_write_period=3,
            residual_error_upper_content_write_offset=0,
            residual_error_upper_content_read_period=3,
            residual_error_upper_content_read_offset=2,
            residual_error_upper_content_read_delay=3,
            inference_steps=1,
        )
    )
    x = torch.randint(0, 31, (2, 6))
    y = (x + 1) % 31

    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
    )

    diagnostics = model._last_update_diagnostics
    assert diagnostics[
        "update_diag_message_pc_residual_error_upper_content_state_scale"
    ] == 0.5
    assert diagnostics[
        "update_diag_message_pc_residual_error_upper_content_state_decay"
    ] == 0.5
    assert diagnostics[
        "update_diag_message_pc_residual_error_upper_content_write_period"
    ] == 3.0
    assert diagnostics[
        "update_diag_message_pc_residual_error_upper_content_write_offset"
    ] == 0.0
    assert diagnostics[
        "update_diag_message_pc_residual_error_upper_content_read_period"
    ] == 3.0
    assert diagnostics[
        "update_diag_message_pc_residual_error_upper_content_read_offset"
    ] == 2.0
    assert diagnostics[
        "update_diag_message_pc_residual_error_upper_content_read_delay"
    ] == 3.0
    assert diagnostics[
        "update_diag_message_pc_residual_error_upper_content_state_input_rms"
    ] > 0.0
    assert diagnostics[
        "update_diag_message_pc_residual_error_upper_state_input_rms"
    ] > 0.0


def test_residual_error_pc_local_update_uses_no_parameter_autograd() -> None:
    torch.manual_seed(137)
    model = MessagePassingPCLanguageModel(
        residual_error_config(inference_steps=6, inference_lr=0.1)
    )
    x = torch.randint(0, 31, (2, 5))
    y = (x + 1) % 31
    before = {
        name: parameter.clone()
        for name, parameter in model.named_parameters()
    }

    for step in range(1, 3):
        model.local_update(
            x,
            y,
            decoder_lr=1e-3,
            state_lr=1e-3,
            embed_lr=1e-3,
            current_step=step,
        )

    after = dict(model.named_parameters())
    assert not torch.equal(before["decoder_weight"], after["decoder_weight"])
    assert not torch.equal(
        before["message_weights.0"],
        after["message_weights.0"],
    )
    assert not torch.equal(
        before["feedback_weights.0"],
        after["feedback_weights.0"],
    )
    assert not torch.equal(
        before["inverse_query_weights.0"],
        after["inverse_query_weights.0"],
    )
    assert model._last_update_diagnostics[
        "update_diag_message_pc_uses_residual_error_nodes"
    ] == 1.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_residual_error_pc_direct_local_update_learns_predictor() -> None:
    torch.manual_seed(147)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            inference_steps=4,
            inference_lr=0.1,
            hierarchical_error_prediction_mode="direct",
        )
    )
    x = torch.randint(0, 31, (2, 5))
    y = (x + 1) % 31
    before = model.generative_weights[0].clone()

    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=1,
    )

    assert not torch.equal(before, model.generative_weights[0])
    assert model._last_update_diagnostics[
        "update_diag_message_pc_residual_error_free_prediction_rms_layer_1"
    ] > 0.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_residual_error_pc_direct_transport_learns_local_feedback_path() -> None:
    torch.manual_seed(148)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            inference_steps=4,
            inference_lr=0.1,
            hierarchical_error_prediction_mode="direct_transport",
            learned_feedback_proposal_scale=1.0,
        )
    )
    x = torch.randint(0, 31, (2, 6))
    y = (x + 1) % 31
    before = {
        name: parameter.clone()
        for name, parameter in model.named_parameters()
    }

    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=1,
    )

    after = dict(model.named_parameters())
    for name in (
        "inverse_query_weights.0",
        "inverse_key_weights.0",
        "feedback_weights.0",
        "generative_weights.0",
    ):
        assert not torch.equal(before[name], after[name])
    assert model._last_update_diagnostics[
        "update_diag_message_pc_residual_transport_reconstruction_rms"
    ] > 0
    assert model._last_update_diagnostics[
        "update_diag_message_pc_residual_error_energy_reduction"
    ] > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_residual_error_pc_phase_transport_learns_accepted_state_change() -> None:
    torch.manual_seed(150)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            inference_steps=4,
            inference_lr=0.1,
            hierarchical_error_prediction_mode="direct_phase_transport",
            learned_feedback_proposal_scale=0.25,
            residual_error_suffix_decay=0.5,
        )
    )
    x = torch.randint(0, 31, (2, 6))
    y = (x + 1) % 31
    before = {
        name: parameter.clone()
        for name, parameter in model.named_parameters()
    }

    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=1,
    )

    after = dict(model.named_parameters())
    for name in (
        "inverse_query_weights.0",
        "inverse_key_weights.0",
        "feedback_weights.0",
        "generative_weights.0",
    ):
        assert not torch.equal(before[name], after[name])
    assert model._last_update_diagnostics[
        "update_diag_message_pc_residual_phase_transport_target_rms"
    ] > 0
    assert model._last_update_diagnostics[
        "update_diag_message_pc_residual_phase_transport_lr_scale"
    ] == 1.0
    assert model._last_update_diagnostics[
        "update_diag_message_pc_residual_error_suffix_decay"
    ] == 0.5
    for layer in range(model.config.num_layers):
        assert model._last_update_diagnostics[
            f"update_diag_message_pc_residual_phase_transport_target_rms_layer_{layer}"
        ] > 0
        assert model._last_update_diagnostics[
            f"update_diag_message_pc_residual_phase_transport_fit_rms_layer_{layer}"
        ] > 0
        assert math.isfinite(
            model._last_update_diagnostics[
                f"update_diag_message_pc_residual_phase_transport_cosine_layer_{layer}"
            ]
        )
    assert model._last_update_diagnostics[
        "update_diag_message_pc_residual_error_energy_reduction"
    ] > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_residual_error_pc_phase_transport_refine_resettles_same_step() -> None:
    torch.manual_seed(151)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            inference_steps=4,
            inference_lr=0.1,
            hierarchical_error_prediction_mode="direct_phase_transport",
            learned_feedback_proposal_scale=0.25,
            residual_phase_transport_refine_passes=1,
        )
    )
    x = torch.randint(0, 31, (2, 6))
    y = (x + 1) % 31

    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=1,
    )

    diagnostics = model._last_update_diagnostics
    assert diagnostics[
        "update_diag_message_pc_residual_phase_transport_refine_passes"
    ] == 1.0
    assert diagnostics[
        "update_diag_message_pc_residual_phase_transport_refine_update_rms"
    ] > 0.0
    assert diagnostics[
        "update_diag_message_pc_residual_error_monotone_energy"
    ] == 1.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_residual_error_pc_learns_local_persistent_context_gates() -> None:
    torch.manual_seed(149)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            inference_steps=4,
            inference_lr=0.1,
            hierarchical_error_context_mode="persistent_hard",
            hierarchical_error_memory_scale=1.0,
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    before = [weight.clone() for weight in model.hierarchical_gate_weights]

    for step in range(1, 3):
        model.local_update(
            x,
            y,
            decoder_lr=1e-3,
            state_lr=1e-3,
            embed_lr=1e-3,
            current_step=step,
        )

    assert all(
        not torch.equal(old, new)
        for old, new in zip(before, model.hierarchical_gate_weights)
    )
    assert model._last_update_diagnostics[
        "update_diag_message_pc_residual_error_gate_gradient_rms"
    ] > 0
    assert model._last_update_diagnostics[
        "update_diag_message_pc_residual_error_uses_persistent_context"
    ] == 1.0
    assert model._last_update_diagnostics[
        "update_diag_message_pc_residual_error_uses_hard_persistent_context"
    ] == 1.0
    assert model._last_update_diagnostics[
        "update_diag_message_pc_residual_error_context_memory_scale"
    ] == 1.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_residual_error_pc_fixed_outcome_basis_stays_frozen() -> None:
    torch.manual_seed(151)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            hierarchical_error_target_mode="fixed_outcome",
            hierarchical_error_outcome_code_scale=0.5,
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    fixed_basis = model.feedback_embedding.clone()

    for step in range(1, 3):
        model.local_update(
            x,
            y,
            decoder_lr=1e-3,
            state_lr=1e-3,
            embed_lr=1e-3,
            current_step=step,
        )

    assert torch.equal(model.feedback_embedding, fixed_basis)
    assert model._last_update_diagnostics[
        "update_diag_message_pc_residual_error_uses_fixed_outcome"
    ] == 1.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_residual_error_pc_all_token_target_can_use_independent_output_feedback() -> None:
    torch.manual_seed(152)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            hierarchical_error_target_mode="all_tokens",
            meta_pc_output_feedback_mode="fixed_random",
        )
    )
    model.decoder_weight.zero_()
    model.decoder_bias.zero_()
    x = torch.randint(0, 31, (2, 6))
    y = (x + 1) % 31

    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=0.0,
        embed_lr=0.0,
        weight_decay=0.0,
        current_step=1,
    )

    diagnostics = model._last_update_diagnostics
    assert diagnostics[
        "update_diag_message_pc_residual_error_token_target_delta_rms"
    ] > 0.0
    assert diagnostics[
        "update_diag_message_pc_residual_error_uses_decoder_output_feedback"
    ] == 0.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_residual_error_pc_kp_output_feedback_uses_local_decoder_association() -> None:
    torch.manual_seed(153)
    model = MessagePassingPCLanguageModel(
        residual_error_config(
            hierarchical_error_target_mode="all_tokens",
            meta_pc_output_feedback_mode="kp_learned",
            meta_pc_output_feedback_lr_scale=1.0,
        )
    )
    x = torch.randint(0, 31, (2, 6))
    y = (x + 1) % 31
    decoder_before = model.decoder_weight.clone()
    feedback_before = model.feedback_embedding.clone()

    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=0.0,
        embed_lr=0.0,
        weight_decay=0.0,
        current_step=1,
    )

    decoder_delta = model.decoder_weight - decoder_before
    feedback_delta = model.feedback_embedding - feedback_before
    assert torch.allclose(feedback_delta, decoder_delta, atol=1e-8, rtol=1e-6)
    assert model._last_update_diagnostics[
        "update_diag_message_pc_residual_error_uses_kp_output_feedback"
    ] == 1.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_free_local_ce_matches_last_block_oracle() -> None:
    torch.manual_seed(59)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            route_update_mode="local_energy",
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    _, oracle = oracle_gradients(copy.deepcopy(model), x, y)

    local = free_local_ce_block_gradients(model, x, y, layers=(1,))

    assert local
    for name, gradient in local.items():
        oracle_gradient = oracle[name]
        assert F.cosine_similarity(
            gradient.float().reshape(1, -1),
            oracle_gradient.float().reshape(1, -1),
        ).item() == pytest.approx(1.0, abs=2e-5)
        assert gradient.norm().item() == pytest.approx(
            oracle_gradient.norm().item(),
            rel=2e-5,
            abs=2e-7,
        )


def test_message_pc_free_local_ce_embedding_matches_one_block_oracle() -> None:
    torch.manual_seed(67)
    model = MessagePassingPCLanguageModel(
        small_config(
            num_layers=1,
            inference_mode="pc_energy",
            route_update_mode="local_energy",
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    _, oracle = oracle_gradients(copy.deepcopy(model), x, y)

    local = free_local_ce_embedding_gradient(model, x, y)["embedding.weight"]
    expected = oracle["embedding.weight"]

    assert F.cosine_similarity(
        local.float().reshape(1, -1),
        expected.float().reshape(1, -1),
    ).item() == pytest.approx(1.0, abs=2e-5)
    assert local.norm().item() == pytest.approx(
        expected.norm().item(),
        rel=2e-5,
        abs=2e-7,
    )


def test_message_pc_free_local_token_credit_adds_expected_gradients() -> None:
    torch.manual_seed(61)
    base = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            route_update_mode="local_energy",
            learned_feedback_proposal_scale=0.0,
        )
    )
    augmented = copy.deepcopy(base)
    augmented.config = replace(
        augmented.config,
        free_local_token_credit_scale=0.25,
    )
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    expected = free_local_ce_block_gradients(base, x, y)

    base_gradients = capture_local_gradients(
        copy.deepcopy(base),
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=1,
    )
    augmented_gradients = capture_local_gradients(
        augmented,
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=1,
    )

    for name, expected_gradient in expected.items():
        observed_delta = augmented_gradients[name] - base_gradients[name]
        assert torch.allclose(
            observed_delta,
            0.25 * expected_gradient,
            atol=2e-7,
            rtol=2e-5,
        )


def test_message_pc_free_local_token_embedding_credit_adds_expected_gradient() -> None:
    torch.manual_seed(71)
    base = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            route_update_mode="local_energy",
            learned_feedback_proposal_scale=0.0,
        )
    )
    augmented = copy.deepcopy(base)
    augmented.config = replace(
        augmented.config,
        free_local_token_embedding_credit_scale=0.25,
    )
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31
    expected = free_local_ce_embedding_gradient(base, x, y)["embedding.weight"]

    base_gradients = capture_local_gradients(
        copy.deepcopy(base),
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=1,
    )
    augmented_gradients = capture_local_gradients(
        augmented,
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=1,
    )

    observed_delta = (
        augmented_gradients["embedding.weight"]
        - base_gradients["embedding.weight"]
    )
    assert torch.allclose(
        observed_delta,
        0.25 * expected,
        atol=2e-7,
        rtol=2e-5,
    )


def test_message_pc_parses_settling_alignment_variant() -> None:
    model = MessagePassingPCLanguageModel(small_config())

    inherited = parse_settling_variant("i16_b03:16:0.3:0", model)
    explicit = parse_settling_variant("i32_b1:32:1:1:0.02:0", model)
    accelerated = parse_settling_variant(
        "i16_cg:16:0.1:0:0.02:0:conjugate_gradient:0.7",
        model,
    )

    assert inherited.inference_steps == 16
    assert inherited.inference_lr == model.config.inference_lr
    assert inherited.state_delta_rms_cap == model.config.state_delta_rms_cap
    assert explicit.inference_steps == 32
    assert explicit.inference_lr == pytest.approx(0.02)
    assert explicit.state_delta_rms_cap == 0.0
    assert accelerated.inference_optimizer == "conjugate_gradient"
    assert accelerated.inference_momentum == pytest.approx(0.7)
    with pytest.raises(ValueError, match="settling variant"):
        parse_settling_variant("bad:4:1", model)


def test_message_pc_ffn_updates_with_explicit_local_errors_only() -> None:
    torch.manual_seed(43)
    model = MessagePassingPCLanguageModel(
        small_config(
            ffn_dim=48,
            matrix_optimizer="adam",
        )
    )
    model.ffn_down_weights[0].normal_(mean=0.0, std=0.01)
    before_up = model.ffn_up_weights[0].clone()
    before_down = model.ffn_down_weights[0].clone()
    x = torch.randint(0, 31, (2, 8))
    y = (x + 1) % 31

    model.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=1e-3,
        embed_lr=0.0,
    )

    assert not torch.equal(model.ffn_up_weights[0], before_up)
    assert not torch.equal(model.ffn_down_weights[0], before_down)
    assert model._last_update_diagnostics[
        "update_diag_message_pc_ffn_update_rms"
    ] > 0.0
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_disabled_ffn_preserves_initialization_and_forward_exactly() -> None:
    config = small_config(ffn_dim=0)
    torch.manual_seed(44)
    implicit = MessagePassingPCLanguageModel(small_config())
    torch.manual_seed(44)
    explicit = MessagePassingPCLanguageModel(config)
    x = torch.randint(0, 31, (2, 8))

    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    for name, value in implicit.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name]), name
    assert torch.equal(implicit(x), explicit(x))


@pytest.mark.parametrize(
    "block_state_mode",
    ["squashed_state", "identity_residual", "nanochat_residual"],
)
def test_message_pc_zero_output_ffn_preserves_common_seed_and_forward(
    block_state_mode: str,
) -> None:
    torch.manual_seed(45)
    control = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode=block_state_mode,
            ffn_dim=0,
        )
    )
    torch.manual_seed(45)
    expanded = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode=block_state_mode,
            ffn_dim=48,
        )
    )
    expanded_parameters = dict(expanded.named_parameters())
    x = torch.randint(0, 31, (2, 8))

    for name, value in control.named_parameters():
        assert torch.equal(value, expanded_parameters[name]), name
    assert all(torch.count_nonzero(weight) == 0 for weight in expanded.ffn_down_weights)
    assert torch.equal(control(x), expanded(x))


def test_message_pc_ffn_can_isolate_free_to_settled_amortization() -> None:
    torch.manual_seed(46)
    phase = MessagePassingPCLanguageModel(
        small_config(
            ffn_dim=48,
            ffn_lr_scale=0.2,
            ffn_learning_source="phase",
            inference_mode="pc_energy",
            block_state_mode="identity_residual",
            energy_backtracking_steps=4,
            learned_feedback_proposal_scale=1.0,
            route_lr_scale=0.0,
        )
    )
    amortized = MessagePassingPCLanguageModel(
        small_config(
            ffn_dim=48,
            ffn_lr_scale=0.2,
            ffn_learning_source="free_to_settled",
            inference_mode="pc_energy",
            block_state_mode="identity_residual",
            energy_backtracking_steps=4,
            learned_feedback_proposal_scale=1.0,
            route_lr_scale=0.0,
        )
    )
    amortized.load_state_dict(copy.deepcopy(phase.state_dict()))
    x = torch.randint(0, 31, (2, 8))
    y = (x + 7) % 31

    phase.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=1e-3,
        embed_lr=0.0,
    )
    amortized.local_update(
        x,
        y,
        decoder_lr=0.0,
        state_lr=1e-3,
        embed_lr=0.0,
    )

    assert torch.equal(phase.self_weights[1], amortized.self_weights[1])
    assert torch.equal(
        phase.ffn_down_weights[0],
        amortized.ffn_down_weights[0],
    )
    assert not torch.equal(
        phase.ffn_down_weights[1],
        amortized.ffn_down_weights[1],
    )
    assert amortized._last_update_diagnostics[
        "update_diag_message_pc_ffn_uses_free_to_settled"
    ] == 1.0
    assert all(parameter.grad is None for parameter in amortized.parameters())


def test_message_pc_free_phase_source_bridges_deployment_state_locally() -> None:
    torch.manual_seed(23)
    settled = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            learning_phase_source="settled",
            block_state_mode="identity_residual",
            decoder_state_mode="rms_norm",
            readout_layer=1,
            inference_lr=0.03,
            energy_backtracking_steps=8,
            token_feedback_scale=3.0,
            message_init_gain=1.0,
        )
    )
    free = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            learning_phase_source="free",
            block_state_mode="identity_residual",
            decoder_state_mode="rms_norm",
            readout_layer=1,
            inference_lr=0.03,
            energy_backtracking_steps=8,
            token_feedback_scale=3.0,
            message_init_gain=1.0,
        )
    )
    free.load_state_dict(copy.deepcopy(settled.state_dict()))
    dual = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            learning_phase_source="dual",
            block_state_mode="identity_residual",
            decoder_state_mode="rms_norm",
            readout_layer=1,
            inference_lr=0.03,
            energy_backtracking_steps=8,
            token_feedback_scale=3.0,
            message_init_gain=1.0,
        )
    )
    dual.load_state_dict(copy.deepcopy(settled.state_dict()))
    x = torch.randint(0, 31, (2, 8))
    y = (x + 3) % 31

    settled.local_update(x, y, decoder_lr=0.0, state_lr=1e-3, embed_lr=0.0)
    free.local_update(x, y, decoder_lr=0.0, state_lr=1e-3, embed_lr=0.0)
    dual.local_update(x, y, decoder_lr=0.0, state_lr=1e-3, embed_lr=0.0)

    assert torch.equal(settled.self_weights[0], free.self_weights[0])
    assert torch.equal(settled.self_weights[0], dual.self_weights[0])
    assert not torch.equal(settled.self_weights[1], free.self_weights[1])
    assert not torch.equal(settled.self_weights[1], dual.self_weights[1])
    assert not torch.equal(free.self_weights[1], dual.self_weights[1])
    assert settled._last_update_diagnostics[
        "update_diag_message_pc_learning_uses_free_source"
    ] == 0.0
    assert free._last_update_diagnostics[
        "update_diag_message_pc_learning_uses_free_source"
    ] == 1.0
    assert dual._last_update_diagnostics[
        "update_diag_message_pc_learning_uses_dual_source"
    ] == 1.0
    assert all(parameter.grad is None for parameter in free.parameters())
    assert all(parameter.grad is None for parameter in dual.parameters())


def test_message_pc_forward_update_changes_only_selected_top_blocks() -> None:
    torch.manual_seed(31)
    control = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            learning_phase_source="settled",
            block_state_mode="identity_residual",
            decoder_state_mode="rms_norm",
            inference_lr=0.03,
            energy_backtracking_steps=8,
            token_feedback_scale=3.0,
            message_init_gain=1.0,
            route_lr_scale=0.0,
            generative_lr_scale=0.0,
        )
    )
    forward = MessagePassingPCLanguageModel(
        replace(control.config, forward_update_top_layers=1)
    )
    forward.load_state_dict(copy.deepcopy(control.state_dict()))
    x = torch.randint(0, 31, (2, 8))
    y = (x + 4) % 31

    control.local_update(x, y, decoder_lr=0.0, state_lr=1e-3, embed_lr=0.0)
    forward.local_update(x, y, decoder_lr=0.0, state_lr=1e-3, embed_lr=0.0)

    assert torch.equal(control.self_weights[0], forward.self_weights[0])
    assert not torch.equal(control.self_weights[1], forward.self_weights[1])
    assert forward._last_update_diagnostics[
        "update_diag_message_pc_forward_update_top_layers"
    ] == 1.0
    assert all(parameter.grad is None for parameter in forward.parameters())


def test_message_pc_adds_small_free_phase_amortization_only_to_deeper_blocks() -> None:
    torch.manual_seed(29)
    control = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            learning_phase_source="settled",
            block_state_mode="identity_residual",
            decoder_state_mode="rms_norm",
            inference_lr=0.03,
            energy_backtracking_steps=8,
            token_feedback_scale=3.0,
            message_init_gain=1.0,
            route_lr_scale=0.0,
            generative_lr_scale=0.0,
        )
    )
    amortized = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            learning_phase_source="settled",
            free_phase_amortization_scale=0.1,
            block_state_mode="identity_residual",
            decoder_state_mode="rms_norm",
            inference_lr=0.03,
            energy_backtracking_steps=8,
            token_feedback_scale=3.0,
            message_init_gain=1.0,
            route_lr_scale=0.0,
            generative_lr_scale=0.0,
        )
    )
    amortized.load_state_dict(copy.deepcopy(control.state_dict()))
    x = torch.randint(0, 31, (2, 8))
    y = (x + 5) % 31

    control.local_update(x, y, decoder_lr=0.0, state_lr=1e-3, embed_lr=0.0)
    amortized.local_update(x, y, decoder_lr=0.0, state_lr=1e-3, embed_lr=0.0)

    assert torch.equal(control.self_weights[0], amortized.self_weights[0])
    assert not torch.equal(control.self_weights[1], amortized.self_weights[1])
    assert amortized._last_update_diagnostics[
        "update_diag_message_pc_free_phase_amortization_scale"
    ] == pytest.approx(0.1)
    assert amortized._last_update_diagnostics[
        "update_diag_message_pc_free_phase_amortization_gradient_rms"
    ] > 0.0
    assert all(parameter.grad is None for parameter in amortized.parameters())


def test_message_pc_free_decoder_source_trains_deployed_state() -> None:
    torch.manual_seed(25)
    settled = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            decoder_learning_source="settled",
            decoder_supervision_mode="top_only",
            block_state_mode="identity_residual",
            decoder_state_mode="rms_norm",
            inference_lr=0.03,
            energy_backtracking_steps=8,
            token_feedback_scale=3.0,
            message_init_gain=1.0,
        )
    )
    free = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            decoder_learning_source="free",
            decoder_supervision_mode="top_only",
            block_state_mode="identity_residual",
            decoder_state_mode="rms_norm",
            inference_lr=0.03,
            energy_backtracking_steps=8,
            token_feedback_scale=3.0,
            message_init_gain=1.0,
        )
    )
    free.load_state_dict(copy.deepcopy(settled.state_dict()))
    x = torch.randint(0, 31, (2, 8))
    y = (x + 7) % 31

    settled.local_update(x, y, decoder_lr=1e-3, state_lr=0.0, embed_lr=0.0)
    free.local_update(x, y, decoder_lr=1e-3, state_lr=0.0, embed_lr=0.0)

    assert not torch.equal(settled.decoder_weight, free.decoder_weight)
    assert settled._last_update_diagnostics[
        "update_diag_message_pc_decoder_uses_free_source"
    ] == 0.0
    assert free._last_update_diagnostics[
        "update_diag_message_pc_decoder_uses_free_source"
    ] == 1.0
    assert all(parameter.grad is None for parameter in free.parameters())


def test_message_pc_learned_feedback_proposal_descends_and_learns_locally() -> None:
    torch.manual_seed(24)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            learned_feedback_proposal_scale=1.0,
            decoder_supervision_mode="top_only",
            block_state_mode="identity_residual",
            residual_scale=0.25,
            decoder_state_mode="rms_norm",
            inference_lr=0.03,
            energy_backtracking_steps=8,
            token_feedback_scale=3.0,
            feedback_lr_scale=1.0,
        )
    )
    without_proposal = MessagePassingPCLanguageModel(
        small_config(inference_mode="pc_energy")
    )
    x = torch.randint(0, 31, (2, 8))
    y = (x + 5) % 31
    before = model.feedback_weights[0].clone()

    model.local_update(x, y, decoder_lr=1e-3, state_lr=1e-3, embed_lr=1e-3)

    assert model._last_update_diagnostics[
        "update_diag_message_pc_energy_reduction"
    ] >= 0.0
    assert model._last_update_diagnostics[
        "update_diag_message_pc_learned_feedback_proposal_rms"
    ] > 0.0
    assert not torch.equal(model.feedback_weights[0], before)
    assert model.active_parameter_count() > without_proposal.active_parameter_count()
    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_block_initialization_gains_are_explicit() -> None:
    torch.manual_seed(20)
    model = MessagePassingPCLanguageModel(
        small_config(self_init_gain=0.25, message_init_gain=1.0)
    )
    expected_self_rms = 0.25 / model.config.hidden_dim**0.5
    expected_message_rms = 1.0 / model.config.hidden_dim**0.5

    assert model.self_weights[0].square().mean().sqrt().item() == pytest.approx(
        expected_self_rms,
        rel=0.08,
    )
    assert model.message_weights[0].square().mean().sqrt().item() == pytest.approx(
        expected_message_rms,
        rel=0.08,
    )


def test_message_pc_inference_lr_schedule_interpolates_by_training_step() -> None:
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_lr=0.03,
            inference_lr_final=0.012,
            inference_lr_schedule_start_step=100,
            inference_lr_schedule_end_step=400,
        )
    )

    assert model._inference_lr(1) == pytest.approx(0.03)
    assert model._inference_lr(100) == pytest.approx(0.03)
    assert model._inference_lr(250) == pytest.approx(0.021)
    assert model._inference_lr(400) == pytest.approx(0.012)
    assert model._inference_lr(900) == pytest.approx(0.012)

    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=250,
    )
    assert model._last_update_diagnostics[
        "update_diag_message_pc_effective_inference_lr"
    ] == pytest.approx(0.021)


def test_message_pc_optimizer_schedules_match_nanochat_step_indexing() -> None:
    model = MessagePassingPCLanguageModel(
        small_config(
            matrix_optimizer="muon",
            optimizer_warmup_steps=40,
            muon_momentum=0.85,
            muon_momentum_final=0.97,
            muon_momentum_warmup_steps=400,
        )
    )

    assert model._optimizer_lr_multiplier(1) == pytest.approx(1 / 40)
    assert model._optimizer_lr_multiplier(20) == pytest.approx(0.5)
    assert model._optimizer_lr_multiplier(40) == pytest.approx(1.0)
    assert model._optimizer_lr_multiplier(100) == pytest.approx(1.0)
    assert model._muon_momentum(1) == pytest.approx(0.85)
    assert model._muon_momentum(201) == pytest.approx(0.91)
    assert model._muon_momentum(401) == pytest.approx(0.97)

    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=20,
    )
    assert model._last_update_diagnostics[
        "update_diag_message_pc_optimizer_lr_multiplier"
    ] == pytest.approx(0.5)
    assert model._last_update_diagnostics[
        "update_diag_message_pc_effective_muon_momentum"
    ] == pytest.approx(0.85 + 19 / 400 * 0.12)


def test_message_pc_depth_mup_uses_raw_weights_and_explicit_scale() -> None:
    standard_config = small_config(
        block_state_mode="identity_residual",
        self_init_gain=0.5,
        message_init_gain=1.0,
    )
    torch.manual_seed(58)
    standard = MessagePassingPCLanguageModel(standard_config)
    torch.manual_seed(58)
    depth_mup = MessagePassingPCLanguageModel(
        replace(standard_config, block_parameterization="depth_mup")
    )

    width_scale = math.sqrt(standard_config.hidden_dim)
    assert torch.allclose(
        depth_mup.self_weights[0],
        standard.self_weights[0] * width_scale,
    )
    assert torch.allclose(
        depth_mup.message_weights[0],
        standard.message_weights[0] * width_scale,
    )
    assert depth_mup._block_weight_scale() == pytest.approx(
        1.0 / math.sqrt(
            standard_config.hidden_dim * standard_config.num_layers
        )
    )
    assert standard._block_weight_scale() == 1.0


def test_message_pc_depth_mup_local_block_gradients_match_autograd() -> None:
    torch.manual_seed(59)
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="identity_residual",
            block_parameterization="depth_mup",
            residual_scale=1.0,
            self_init_gain=1.0,
            message_init_gain=1.0,
        )
    ).double()
    lower_state = torch.randn(
        1,
        4,
        model.config.hidden_dim,
        dtype=torch.float64,
    )
    target_state = torch.randn_like(lower_state)
    names = (
        "self_weights.0",
        "message_weights.0",
        "query_weights.0",
        "key_weights.0",
        "value_weights.0",
    )
    parameters = dict(model.named_parameters())
    for name in names:
        parameters[name].requires_grad_(True)

    prediction = model._block(0, lower_state).prediction
    objective = 0.5 * (prediction - target_state).square().sum() / 4.0
    oracle = torch.autograd.grad(
        objective,
        [parameters[name] for name in names],
    )
    local, _, _, _, _ = model._phase_block_gradients(
        0,
        lower_state,
        target_state,
        route_lower_state=lower_state,
        causal_mask=model._causal_mask(4, lower_state.device),
        token_count=4.0,
    )

    for name, oracle_gradient in zip(names, oracle):
        assert torch.allclose(
            local[name],
            oracle_gradient,
            rtol=2e-8,
            atol=2e-9,
        )


def test_message_pc_nanochat_residual_initializes_as_identity() -> None:
    torch.manual_seed(60)
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            ffn_dim=48,
        )
    )
    x = torch.randint(0, 31, (2, 8))

    states, caches = model._forward_states(x, with_cache=True)
    embedded = model.embedding(x)
    expected_input = embedded * torch.rsqrt(
        embedded.square().mean(dim=-1, keepdim=True) + 1e-6
    )

    assert torch.equal(states[0], expected_input)
    assert all(torch.equal(state, states[0]) for state in states[1:])
    assert all(torch.count_nonzero(weight) == 0 for weight in model.self_weights)
    assert all(torch.count_nonzero(weight) == 0 for weight in model.message_weights)
    assert all(torch.count_nonzero(weight) == 0 for weight in model.ffn_down_weights)
    assert not torch.equal(model.decoder_weight, model.embedding.weight)
    assert torch.allclose(
        caches[0].query.square().mean(dim=-1).sqrt(),
        torch.full_like(caches[0].query[..., 0], 1.2),
        rtol=2e-4,
        atol=2e-4,
    )


def test_message_pc_nanochat_residual_local_block_gradients_match_autograd() -> None:
    torch.manual_seed(61)
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            ffn_dim=48,
            route_update_mode="local_energy",
        )
    ).double()
    model.message_weights[0].normal_(mean=0.0, std=0.05)
    model.ffn_down_weights[0].normal_(mean=0.0, std=0.03)
    lower_state = torch.randn(
        1,
        4,
        model.config.hidden_dim,
        dtype=torch.float64,
    )
    target_state = torch.randn_like(lower_state)
    names = (
        "message_weights.0",
        "query_weights.0",
        "key_weights.0",
        "value_weights.0",
        "ffn_up_weights.0",
        "ffn_down_weights.0",
    )
    parameters = dict(model.named_parameters())
    for name in names:
        parameters[name].requires_grad_(True)

    prediction = model._block(0, lower_state).prediction
    objective = 0.5 * (prediction - target_state).square().sum() / 4.0
    oracle = torch.autograd.grad(
        objective,
        [parameters[name] for name in names],
    )
    local, _, _, _, _ = model._phase_block_gradients(
        0,
        lower_state,
        target_state,
        route_lower_state=lower_state,
        causal_mask=model._causal_mask(4, lower_state.device),
        token_count=4.0,
    )

    assert "self_weights.0" not in local
    assert "state_biases.0" not in local
    for name, oracle_gradient in zip(names, oracle):
        assert torch.allclose(
            local[name],
            oracle_gradient,
            rtol=3e-7,
            atol=3e-9,
        ), name


def test_message_pc_recurrent_local_reuses_one_block_without_autograd() -> None:
    torch.manual_seed(64)
    model = MessagePassingPCLanguageModel(
        small_config(
            num_layers=1,
            inference_mode="recurrent_local",
            recurrent_shared_steps=3,
            recurrent_step_scale=0.25,
            inference_lr=1.0,
            decoder_supervision_mode="top_only",
            decoder_state_mode="rms_norm",
            block_state_mode="nanochat_residual",
            ffn_dim=48,
            route_update_mode="local_energy",
        )
    )
    with torch.no_grad():
        model.message_weights[0].normal_(mean=0.0, std=0.02)
        model.ffn_down_weights[0].normal_(mean=0.0, std=0.01)
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))

    states, caches = model._forward_states(x, with_cache=True)
    manual_state = states[0]
    for recurrent_step in range(3):
        manual_state = model._block(
            0,
            manual_state,
            branch_scale=1.0 if recurrent_step == 0 else 0.25,
        ).prediction

    assert len(states) == 4
    assert len(caches) == 3
    assert torch.allclose(states[-1], manual_state)
    assert torch.allclose(model.forward_states(x), manual_state)

    before = model.message_weights[0].clone()
    loss = model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=1,
    )

    assert math.isfinite(loss)
    assert not torch.equal(model.message_weights[0], before)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model._last_update_diagnostics[
        "update_diag_message_pc_uses_recurrent_local"
    ] == 1.0
    assert model._last_update_diagnostics[
        "update_diag_message_pc_recurrent_shared_steps"
    ] == 3.0
    assert model._last_update_diagnostics[
        "update_diag_message_pc_recurrent_step_scale"
    ] == 0.25
    assert model.active_parameter_count() < model.learned_parameter_count()


def test_message_pc_recurrent_local_gradient_matches_detached_cycle_oracle() -> None:
    torch.manual_seed(65)
    model = MessagePassingPCLanguageModel(
        small_config(
            num_layers=1,
            inference_mode="recurrent_local",
            recurrent_shared_steps=3,
            recurrent_step_scale=0.25,
            inference_lr=1.0,
            state_delta_rms_cap=0.0,
            decoder_supervision_mode="top_only",
            decoder_state_mode="rms_norm",
            block_state_mode="nanochat_residual",
            ffn_dim=48,
            route_update_mode="local_energy",
        )
    ).double()
    with torch.no_grad():
        model.message_weights[0].normal_(mean=0.0, std=0.04)
        model.ffn_down_weights[0].normal_(mean=0.0, std=0.03)
    x = torch.randint(0, 31, (1, 4))
    y = torch.randint(0, 31, (1, 4))
    names = (
        "message_weights.0",
        "query_weights.0",
        "key_weights.0",
        "value_weights.0",
        "ffn_up_weights.0",
        "ffn_down_weights.0",
    )
    parameters = dict(model.named_parameters())
    for name in names:
        parameters[name].requires_grad_(True)

    initial_state = model._forward_states(
        x,
        with_cache=False,
        layer_count=0,
    )[0][0].detach()
    lower_states: list[torch.Tensor] = []
    output_states: list[torch.Tensor] = []
    objectives: list[torch.Tensor] = []
    state = initial_state
    for recurrent_step in range(3):
        lower_state = state.detach()
        output_state = model._block(
            0,
            lower_state,
            branch_scale=1.0 if recurrent_step == 0 else 0.25,
        ).prediction
        lower_states.append(lower_state)
        output_states.append(output_state)
        objectives.append(
            F.cross_entropy(
                model._decoder_logits(output_state).reshape(-1, 32),
                y.reshape(-1),
            )
        )
        state = output_state
    oracle = torch.autograd.grad(
        torch.stack(objectives).mean(),
        [parameters[name] for name in names],
    )

    local = {name: torch.zeros_like(parameters[name]) for name in names}
    for recurrent_step, (lower_state, output_state) in enumerate(
        zip(lower_states, output_states)
    ):
        with torch.no_grad():
            descent, _ = model._decoder_negative_state_gradient(
                output_state.detach(),
                y,
            )
            target_state = output_state.detach() + descent
        cycle_gradients, _, _, _, _ = model._phase_block_gradients(
            0,
            lower_state,
            target_state,
            route_lower_state=lower_state,
            causal_mask=model._causal_mask(4, lower_state.device),
            token_count=4.0,
            branch_scale=1.0 if recurrent_step == 0 else 0.25,
        )
        for name in names:
            local[name].add_(cycle_gradients[name], alpha=1.0 / 3.0)

    for name, oracle_gradient in zip(names, oracle):
        assert torch.allclose(
            local[name],
            oracle_gradient,
            rtol=4e-7,
            atol=4e-9,
        ), name


def test_message_pc_layerwise_local_trains_unique_blocks_without_autograd() -> None:
    torch.manual_seed(66)
    model = MessagePassingPCLanguageModel(
        small_config(
            num_layers=3,
            inference_mode="layerwise_local",
            inference_lr=1.0,
            decoder_supervision_mode="top_only",
            decoder_state_mode="rms_norm",
            block_state_mode="nanochat_residual",
            ffn_dim=48,
            route_update_mode="local_energy",
        )
    )
    with torch.no_grad():
        for layer in range(3):
            model.message_weights[layer].normal_(mean=0.0, std=0.02)
            model.ffn_down_weights[layer].normal_(mean=0.0, std=0.01)
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))

    states, caches = model._forward_states(x, with_cache=True)
    manual_state = states[0]
    for layer in range(3):
        manual_state = model._block(layer, manual_state).prediction

    assert len(states) == 4
    assert len(caches) == 3
    assert model._readout_depth() == 3
    assert torch.allclose(states[-1], manual_state)

    before = [weight.clone() for weight in model.message_weights]
    loss = model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=1,
    )

    assert math.isfinite(loss)
    assert all(
        not torch.equal(weight, original)
        for weight, original in zip(model.message_weights, before)
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model._last_update_diagnostics[
        "update_diag_message_pc_uses_recurrent_local"
    ] == 0.0
    assert model._last_update_diagnostics[
        "update_diag_message_pc_uses_layerwise_local"
    ] == 1.0


@pytest.mark.parametrize("path_mode", ["adjacent", "direct"])
def test_message_pc_difference_target_identity_inverse_preserves_delta(
    path_mode: str,
) -> None:
    torch.manual_seed(62)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="difference_target",
            decoder_supervision_mode="top_only",
            block_state_mode="nanochat_residual",
            ffn_dim=48,
            state_delta_rms_cap=0.0,
            difference_target_path_mode=path_mode,
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    free_states, _ = model._forward_states(x, with_cache=False)

    targets, deltas = model._difference_target_states(
        free_states,
        y,
        target_step=0.7,
    )

    assert all(torch.count_nonzero(weight) == 0 for weight in model.feedback_weights)
    for delta in deltas[:-1]:
        assert torch.allclose(delta, deltas[-1], atol=2e-7, rtol=2e-5)
    for free_state, target, delta in zip(free_states, targets, deltas):
        assert torch.allclose(target, free_state + delta)


def test_message_pc_direct_difference_targets_skip_intermediate_states() -> None:
    torch.manual_seed(67)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="difference_target",
            decoder_supervision_mode="top_only",
            block_state_mode="nanochat_residual",
            difference_target_path_mode="direct",
            state_delta_rms_cap=0.0,
        )
    )
    for weight in model.feedback_weights:
        weight.normal_(mean=0.0, std=0.03)
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    free_states, _ = model._forward_states(x, with_cache=False)
    changed_states = list(free_states)
    changed_states[1] = changed_states[1] + torch.randn_like(
        changed_states[1]
    )

    _, original_deltas = model._difference_target_states(
        free_states,
        y,
        target_step=0.7,
    )
    _, changed_deltas = model._difference_target_states(
        changed_states,
        y,
        target_step=0.7,
    )

    for original, changed in zip(original_deltas, changed_deltas):
        assert torch.equal(original, changed)


def test_message_pc_difference_inverse_gradients_match_autograd() -> None:
    torch.manual_seed(63)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="difference_target",
            decoder_supervision_mode="top_only",
            block_state_mode="nanochat_residual",
        )
    ).double()
    model.feedback_weights[0].normal_(mean=0.0, std=0.03)
    upper_state = torch.randn(
        1,
        4,
        model.config.hidden_dim,
        dtype=torch.float64,
    )
    target_state = torch.randn_like(upper_state)
    names = ("generative_weights.0", "feedback_weights.0")
    parameters = dict(model.named_parameters())
    for name in names:
        parameters[name].requires_grad_(True)

    cache = model._inverse_block(0, upper_state)
    objective = 0.5 * (cache.prediction - target_state).square().sum() / 4.0
    oracle = torch.autograd.grad(
        objective,
        [parameters[name] for name in names],
    )
    local = model._inverse_parameter_gradients(
        0,
        cache,
        cache.prediction - target_state,
        token_count=4.0,
    )

    for name, oracle_gradient in zip(names, oracle):
        assert torch.allclose(
            local[name],
            oracle_gradient,
            rtol=3e-7,
            atol=3e-9,
        ), name


def test_message_pc_difference_inverse_damping_penalizes_feedback_gain() -> None:
    torch.manual_seed(65)
    control = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="difference_target",
            decoder_supervision_mode="top_only",
            block_state_mode="nanochat_residual",
            difference_target_inverse_damping=0.0,
        )
    )
    damped = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="difference_target",
            decoder_supervision_mode="top_only",
            block_state_mode="nanochat_residual",
            difference_target_inverse_damping=1.0,
        )
    )
    damped.load_state_dict(copy.deepcopy(control.state_dict()))
    lower_state = torch.randn(2, 8, control.config.hidden_dim)
    upper_state = control._block(0, lower_state).prediction

    torch.manual_seed(66)
    control_gradients, _, _, _, control_response = (
        control._difference_inverse_gradients(
            0,
            lower_state,
            upper_state,
            token_count=16.0,
        )
    )
    torch.manual_seed(66)
    damped_gradients, _, _, _, damped_response = (
        damped._difference_inverse_gradients(
            0,
            lower_state,
            upper_state,
            token_count=16.0,
        )
    )

    control_norm = control_gradients["feedback_weights.0"].norm()
    damped_norm = damped_gradients["feedback_weights.0"].norm()
    assert control_response == 0
    assert damped_response > 0
    assert damped_norm > 100 * control_norm


def test_message_pc_direct_inverse_reconstruction_uses_full_forward_path() -> None:
    torch.manual_seed(68)
    adjacent = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="difference_target",
            decoder_supervision_mode="top_only",
            block_state_mode="nanochat_residual",
            difference_target_path_mode="adjacent",
        )
    )
    direct = MessagePassingPCLanguageModel(
        replace(
            adjacent.config,
            difference_target_path_mode="direct",
        )
    )
    direct.load_state_dict(copy.deepcopy(adjacent.state_dict()))
    adjacent.message_weights[1].normal_(mean=0.0, std=0.05)
    direct.message_weights[1].copy_(adjacent.message_weights[1])
    lower_state = torch.randn(2, 8, adjacent.config.hidden_dim)
    adjacent_upper = adjacent._block(0, lower_state).prediction
    direct_top = direct._forward_from_state(0, lower_state)

    torch.manual_seed(69)
    _, adjacent_error, _, _, _ = adjacent._difference_inverse_gradients(
        0,
        lower_state,
        adjacent_upper,
        token_count=16.0,
    )
    torch.manual_seed(69)
    _, direct_error, _, _, _ = direct._difference_inverse_gradients(
        0,
        lower_state,
        direct_top,
        token_count=16.0,
    )

    assert adjacent_error < 1e-6
    assert direct_error > 100 * adjacent_error


@pytest.mark.parametrize("path_mode", ["adjacent", "direct"])
def test_message_pc_difference_target_local_update_is_manual_and_complete(
    path_mode: str,
) -> None:
    torch.manual_seed(64)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="difference_target",
            decoder_supervision_mode="top_only",
            block_state_mode="nanochat_residual",
            ffn_dim=48,
            route_lr_scale=1.0,
            feedback_lr_scale=1.0,
            generative_lr_scale=1.0,
            matrix_optimizer="adam",
            inference_lr=1.0,
            difference_target_path_mode=path_mode,
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    before = {
        name: parameter.clone()
        for name, parameter in model.named_parameters()
    }

    for step in range(2):
        loss = model.local_update(
            x,
            y,
            decoder_lr=1e-3,
            state_lr=1e-3,
            embed_lr=1e-3,
            current_step=step,
        )

    assert isinstance(loss, float)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not torch.equal(model.decoder_weight, before["decoder_weight"])
    assert not torch.equal(model.embedding.weight, before["embedding.weight"])
    assert not torch.equal(model.message_weights[0], before["message_weights.0"])
    assert not torch.equal(model.feedback_weights[0], before["feedback_weights.0"])
    diagnostics = model._last_update_diagnostics
    assert diagnostics["update_diag_message_pc_uses_difference_target"] == 1.0
    assert diagnostics[
        "update_diag_message_pc_difference_target_uses_direct_path"
    ] == float(path_mode == "direct")
    assert diagnostics["update_diag_message_pc_inverse_reconstruction_rms"] > 0.0
    assert math.isfinite(diagnostics["update_diag_message_pc_inverse_alignment"])


@pytest.mark.parametrize("block_state_mode", ["squashed_state", "identity_residual"])
def test_message_pc_local_attention_gradient_matches_finite_difference(
    block_state_mode: str,
) -> None:
    torch.manual_seed(21)
    model = MessagePassingPCLanguageModel(
        small_config(
            attention_temperature=0.7,
            route_update_mode="local_energy",
            block_state_mode=block_state_mode,
            residual_scale=0.4,
        )
    ).double()
    model.message_weights[0].normal_(mean=0.0, std=0.2)
    lower_state = torch.randn(1, 4, model.config.hidden_dim, dtype=torch.float64) * 0.2
    target_state = torch.randn_like(lower_state) * 0.2
    cache = model._block(0, lower_state)
    prediction_gradient = (cache.prediction - target_state) * cache.prediction_derivative
    message_gradient = F.linear(
        prediction_gradient,
        model.message_weights[0].t(),
    ) / 2.0**0.5
    head_message_gradient = model._split_heads(message_gradient)
    query_gradient, key_gradient, value_gradient, _ = model._attention_local_gradients(
        cache,
        head_message_gradient,
    )

    def energy(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> float:
        score_scale = 1.0 / (
            model.head_dim**0.5 * model.config.attention_temperature
        )
        scores = torch.matmul(query, key.transpose(-2, -1)) * score_scale
        scores.masked_fill_(model._causal_mask(4, scores.device), -torch.inf)
        weights = F.softmax(scores, dim=-1)
        message = model._merge_heads(torch.matmul(weights, value))
        residual = (
            F.linear(cache.source, model.self_weights[0])
            + F.linear(message, model.message_weights[0])
        ) / 2.0**0.5
        prediction, _ = model._compose_prediction(
            lower_state,
            residual,
            model.state_biases[0],
        )
        return float((0.5 * (prediction - target_state).square().sum()).item())

    epsilon = 1e-6
    checks = (
        (cache.query, query_gradient, (0, 0, 2, 1)),
        (cache.key, key_gradient, (0, 0, 1, 2)),
        (cache.value, value_gradient, (0, 0, 1, 3)),
    )
    base = [cache.query.clone(), cache.key.clone(), cache.value.clone()]
    for tensor_index, (tensor, gradient, index) in enumerate(checks):
        plus = [value.clone() for value in base]
        minus = [value.clone() for value in base]
        plus[tensor_index][index] += epsilon
        minus[tensor_index][index] -= epsilon
        numerical = (energy(*plus) - energy(*minus)) / (2.0 * epsilon)
        assert gradient[index].item() == pytest.approx(numerical, rel=2e-4, abs=1e-8)


def test_message_pc_token_error_teacher_routes_from_free_correction() -> None:
    torch.manual_seed(38)
    model = MessagePassingPCLanguageModel(
        small_config(route_update_mode="token_error_teacher")
    )
    route_lower_state = torch.randn(1, 4, model.config.hidden_dim) * 0.2
    phase_lower_a = torch.randn_like(route_lower_state) * 0.2
    phase_lower_b = torch.randn_like(route_lower_state) * 0.2
    free_prediction = model._block(0, route_lower_state).prediction
    causal_mask = model._causal_mask(4, route_lower_state.device)

    zero_gradients, _, zero_tv, zero_credit, _ = model._phase_block_gradients(
        0,
        phase_lower_a,
        free_prediction,
        route_lower_state=route_lower_state,
        causal_mask=causal_mask,
        token_count=4.0,
    )

    assert torch.count_nonzero(zero_gradients["query_weights.0"]) == 0
    assert torch.count_nonzero(zero_gradients["key_weights.0"]) == 0
    assert zero_tv == 0
    assert zero_credit == 0

    target_state = free_prediction + 0.1 * torch.randn_like(free_prediction)
    gradients_a, _, route_tv_a, route_credit_a, _ = (
        model._phase_block_gradients(
            0,
            phase_lower_a,
            target_state,
            route_lower_state=route_lower_state,
            causal_mask=causal_mask,
            token_count=4.0,
        )
    )
    gradients_b, _, route_tv_b, route_credit_b, _ = (
        model._phase_block_gradients(
            0,
            phase_lower_b,
            target_state,
            route_lower_state=route_lower_state,
            causal_mask=causal_mask,
            token_count=4.0,
        )
    )

    assert torch.equal(
        gradients_a["query_weights.0"],
        gradients_b["query_weights.0"],
    )
    assert torch.equal(
        gradients_a["key_weights.0"],
        gradients_b["key_weights.0"],
    )
    assert not torch.equal(
        gradients_a["self_weights.0"],
        gradients_b["self_weights.0"],
    )
    assert torch.count_nonzero(gradients_a["query_weights.0"]) > 0
    assert route_tv_a > 0
    assert route_tv_a == route_tv_b
    assert route_credit_a > 0
    assert route_credit_a == route_credit_b


@pytest.mark.parametrize(
    "block_state_mode",
    ["squashed_state", "identity_residual", "nanochat_residual"],
)
@pytest.mark.parametrize("ffn_dim", [0, 48])
def test_message_pc_block_input_signal_matches_finite_difference(
    block_state_mode: str,
    ffn_dim: int,
) -> None:
    torch.manual_seed(26)
    model = MessagePassingPCLanguageModel(
        small_config(
            attention_temperature=0.8,
            block_state_mode=block_state_mode,
            residual_scale=0.4,
            residual_stream_scale_start=(
                1.13 if block_state_mode == "nanochat_residual" else 1.0
            ),
            residual_stream_scale_end=(
                1.13 if block_state_mode == "nanochat_residual" else 1.0
            ),
            ffn_dim=ffn_dim,
        )
    ).double()
    model.self_weights[0].normal_(mean=0.0, std=0.15)
    model.message_weights[0].normal_(mean=0.0, std=0.15)
    if ffn_dim > 0:
        model.ffn_down_weights[0].normal_(mean=0.0, std=0.05)
    lower_state = torch.randn(1, 4, model.config.hidden_dim, dtype=torch.float64) * 0.2
    output_signal = torch.randn_like(lower_state) * 0.2
    cache = model._block(0, lower_state)
    analytic = model._block_input_signal(0, lower_state, cache, output_signal)

    def objective(value: torch.Tensor) -> float:
        return float((model._block(0, value).prediction * output_signal).sum().item())

    epsilon = 1e-6
    for index in ((0, 0, 0), (0, 2, 7), (0, 3, 19)):
        plus = lower_state.clone()
        minus = lower_state.clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical = (objective(plus) - objective(minus)) / (2.0 * epsilon)
        assert analytic[index].item() == pytest.approx(numerical, rel=3e-4, abs=1e-8)


@pytest.mark.parametrize(
    "block_state_mode",
    ["squashed_state", "identity_residual", "nanochat_residual"],
)
def test_message_pc_ffn_local_gradients_match_finite_difference(
    block_state_mode: str,
) -> None:
    torch.manual_seed(42)
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode=block_state_mode,
            residual_scale=0.4,
            ffn_dim=48,
            route_update_mode="local_energy",
        )
    ).double()
    model.ffn_down_weights[0].normal_(mean=0.0, std=0.05)
    lower_state = torch.randn(1, 4, model.config.hidden_dim, dtype=torch.float64) * 0.2
    target_state = torch.randn_like(lower_state) * 0.2
    gradients, _, _, _, _ = model._phase_block_gradients(
        0,
        lower_state,
        target_state,
        route_lower_state=lower_state,
        causal_mask=model._causal_mask(4, lower_state.device),
        token_count=4.0,
    )

    def objective() -> float:
        prediction = model._block(0, lower_state).prediction
        return float((0.5 * (prediction - target_state).square().sum() / 4.0).item())

    epsilon = 1e-6
    checks = (
        ("ffn_up_weights.0", model.ffn_up_weights[0], (7, 11)),
        ("ffn_down_weights.0", model.ffn_down_weights[0], (13, 7)),
    )
    for name, parameter, index in checks:
        original = parameter[index].item()
        parameter[index] = original + epsilon
        plus = objective()
        parameter[index] = original - epsilon
        minus = objective()
        parameter[index] = original
        numerical = (plus - minus) / (2.0 * epsilon)
        assert gradients[name][index].item() == pytest.approx(
            numerical,
            rel=4e-4,
            abs=1e-8,
        )


@pytest.mark.parametrize("decoder_state_mode", ["raw", "rms_norm"])
def test_message_pc_decoder_state_feedback_matches_finite_difference(
    decoder_state_mode: str,
) -> None:
    torch.manual_seed(31)
    model = MessagePassingPCLanguageModel(
        small_config(
            decoder_state_mode=decoder_state_mode,
            winner_margin=2.5,
            winner_margin_scale=0.75,
        )
    ).double()
    state = torch.randn(1, 3, model.config.hidden_dim, dtype=torch.float64) * 0.4
    y = torch.randint(0, model.config.vocab_size, (1, 3))
    negative_gradient, _ = model._decoder_negative_state_gradient(state, y)

    def objective(value: torch.Tensor) -> float:
        logits = model._decoder_logits(value)
        ce = F.cross_entropy(
            logits.reshape(-1, model.config.vocab_size),
            y.reshape(-1),
            reduction="sum",
        )
        violations, _, _ = model._winner_margin_terms(logits, y)
        return float((ce + model.config.winner_margin_scale * violations.sum()).item())

    epsilon = 1e-6
    for index in ((0, 0, 0), (0, 1, 7), (0, 2, 19)):
        plus = state.clone()
        minus = state.clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical = (objective(plus) - objective(minus)) / (2.0 * epsilon)
        assert negative_gradient[index].item() == pytest.approx(
            -numerical,
            rel=3e-4,
            abs=1e-8,
        )


def test_message_pc_rms_decoder_is_state_scale_invariant() -> None:
    torch.manual_seed(32)
    model = MessagePassingPCLanguageModel(
        small_config(decoder_state_mode="rms_norm")
    )
    state = torch.randn(2, 4, model.config.hidden_dim)

    assert torch.allclose(
        model._decoder_logits(state),
        model._decoder_logits(3.0 * state),
        atol=2e-6,
        rtol=2e-6,
    )


def test_message_pc_winner_margin_decoder_gradient_matches_finite_difference() -> None:
    torch.manual_seed(33)
    model = MessagePassingPCLanguageModel(
        small_config(winner_margin=2.5, winner_margin_scale=0.75)
    ).double()
    features = torch.randn(1, 3, model.config.hidden_dim, dtype=torch.float64) * 0.3
    y = torch.tensor([[2, 5, 7]])
    logits = F.linear(features, model.decoder_weight, model.decoder_bias)
    weight_gradient, bias_gradient, active_fraction = (
        model._winner_margin_decoder_gradients(features, y, logits)
    )
    _, wrong_ids, _ = model._winner_margin_terms(logits, y)

    def objective(weight: torch.Tensor, bias: torch.Tensor) -> float:
        candidate_logits = F.linear(features, weight, bias)
        violations, _, _ = model._winner_margin_terms(candidate_logits, y)
        return float((model.config.winner_margin_scale * violations.mean()).item())

    epsilon = 1e-6
    checks = (
        ("weight", (int(y[0, 0]), 3)),
        ("weight", (int(wrong_ids[0, 0]), 3)),
        ("bias", (int(y[0, 1]),)),
        ("bias", (int(wrong_ids[0, 1]),)),
    )
    for kind, index in checks:
        plus_weight = model.decoder_weight.clone()
        minus_weight = model.decoder_weight.clone()
        plus_bias = model.decoder_bias.clone()
        minus_bias = model.decoder_bias.clone()
        if kind == "weight":
            plus_weight[index] += epsilon
            minus_weight[index] -= epsilon
            analytic = weight_gradient[index].item()
        else:
            plus_bias[index] += epsilon
            minus_bias[index] -= epsilon
            analytic = bias_gradient[index].item()
        numerical = (
            objective(plus_weight, plus_bias) - objective(minus_weight, minus_bias)
        ) / (2.0 * epsilon)
        assert analytic == pytest.approx(numerical, rel=3e-4, abs=1e-8)
    assert active_fraction == pytest.approx(1.0)


def test_message_pc_coherent_inference_reduces_single_energy() -> None:
    torch.manual_seed(27)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            route_update_mode="local_energy",
            decoder_supervision_mode="top_only",
            inference_steps=4,
            inference_lr=0.01,
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    free_states, _ = model._forward_states(x, with_cache=False)

    settled_states, diagnostics = model._settle(free_states, y)

    assert model._pc_energy(settled_states, y) < model._pc_energy(free_states, y)
    assert diagnostics["update_diag_message_pc_energy_reduction"] > 0
    assert diagnostics["update_diag_message_pc_phase_delta_rms"] > 0


def test_message_pc_meta_inference_reduces_frozen_error_objective() -> None:
    torch.manual_seed(68)
    model = MessagePassingPCLanguageModel(
        small_config(
            num_layers=3,
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=3,
            inference_steps=8,
            inference_lr=0.25,
            token_feedback_scale=1.0,
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    free_states, _ = model._forward_states(x, with_cache=False)

    settled_states, diagnostics = model._settle(free_states, y)

    assert diagnostics["update_diag_message_pc_meta_energy_reduction"] > 0
    assert diagnostics["update_diag_message_pc_meta_energy_after"] < diagnostics[
        "update_diag_message_pc_meta_energy_before"
    ]
    assert diagnostics[
        "update_diag_message_pc_meta_uses_frozen_predictions"
    ] == pytest.approx(1.0)
    assert torch.equal(settled_states[0], free_states[0])
    assert all(
        not torch.equal(settled, free)
        for settled, free in zip(settled_states[1:], free_states[1:])
    )


def test_message_pc_relinearized_one_cycle_matches_frozen_meta_pc() -> None:
    torch.manual_seed(95)
    config = small_config(
        num_layers=3,
        inference_mode="meta_pc",
        decoder_supervision_mode="top_only",
        forward_update_top_layers=3,
        inference_steps=1,
        inference_lr=0.25,
        token_feedback_scale=1.0,
    )
    frozen = MessagePassingPCLanguageModel(config)
    dynamic = MessagePassingPCLanguageModel(
        replace(config, meta_pc_relinearize_inference=True)
    )
    dynamic.load_state_dict(frozen.state_dict())
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    free_states, _ = frozen._forward_states(x, with_cache=False)

    with torch.no_grad():
        frozen_states, frozen_diagnostics = frozen._settle_meta_pc(
            free_states,
            y,
            inference_lr=0.25,
            token_ids=x,
        )
        dynamic_states, dynamic_diagnostics = dynamic._settle_meta_pc(
            free_states,
            y,
            inference_lr=0.25,
            token_ids=x,
        )

    assert all(
        torch.equal(frozen_state, dynamic_state)
        for frozen_state, dynamic_state in zip(frozen_states, dynamic_states)
    )
    assert dynamic_diagnostics[
        "update_diag_message_pc_meta_prediction_geometry_drift_rms"
    ] == 0.0
    assert dynamic_diagnostics[
        "update_diag_message_pc_meta_energy_cycle_1"
    ] == frozen_diagnostics[
        "update_diag_message_pc_meta_energy_cycle_1"
    ]


def test_message_pc_relinearized_cycles_refresh_feedback_geometry() -> None:
    torch.manual_seed(96)
    config = small_config(
        num_layers=3,
        inference_mode="meta_pc",
        decoder_supervision_mode="top_only",
        decoder_state_mode="rms_norm",
        forward_update_top_layers=3,
        inference_steps=3,
        inference_lr=0.25,
        token_feedback_scale=1.0,
    )
    frozen = MessagePassingPCLanguageModel(config)
    dynamic = MessagePassingPCLanguageModel(
        replace(config, meta_pc_relinearize_inference=True)
    )
    dynamic.load_state_dict(frozen.state_dict())
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    free_states, _ = frozen._forward_states(x, with_cache=False)

    with torch.no_grad():
        frozen_states, _ = frozen._settle_meta_pc(
            free_states,
            y,
            inference_lr=0.25,
            token_ids=x,
        )
        dynamic_states, diagnostics = dynamic._settle_meta_pc(
            free_states,
            y,
            inference_lr=0.25,
            token_ids=x,
        )

    assert diagnostics[
        "update_diag_message_pc_meta_uses_relinearized_inference"
    ] == 1.0
    assert diagnostics[
        "update_diag_message_pc_meta_uses_frozen_predictions"
    ] == 0.0
    assert diagnostics[
        "update_diag_message_pc_meta_prediction_geometry_drift_rms"
    ] > 0.0
    assert diagnostics[
        "update_diag_message_pc_meta_top_feedback_cosine"
    ] < 1.0
    assert "update_diag_message_pc_meta_energy_cycle_3" in diagnostics
    assert any(
        not torch.equal(frozen_state, dynamic_state)
        for frozen_state, dynamic_state in zip(
            frozen_states[1:],
            dynamic_states[1:],
        )
    )


def test_message_pc_meta_blocked_sweep_uses_new_superior_error() -> None:
    torch.manual_seed(69)
    rate = 0.2
    model = MessagePassingPCLanguageModel(
        small_config(
            num_layers=3,
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=3,
            inference_steps=1,
            inference_lr=rate,
            token_feedback_scale=1.0,
            state_delta_rms_cap=100.0,
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    free_states, caches = model._forward_states(x, with_cache=True)
    top_feedback, _ = model._decoder_negative_state_gradient(
        free_states[-1],
        y,
    )
    top_displacement = model.config.token_feedback_scale * top_feedback
    expected_middle = rate * model._block_input_signal(
        2,
        free_states[2],
        caches[2],
        top_displacement,
    )
    expected_bottom = rate * model._block_input_signal(
        1,
        free_states[1],
        caches[1],
        expected_middle,
    )

    settled_states, _ = model._settle(free_states, y)

    assert torch.allclose(
        settled_states[2] - free_states[2],
        expected_middle,
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(
        settled_states[1] - free_states[1],
        expected_bottom,
        atol=2e-6,
        rtol=2e-6,
    )


def test_message_pc_meta_reciprocal_feedback_uses_connected_nodes_without_vjp() -> None:
    torch.manual_seed(71)
    scale = 0.3
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            meta_pc_feedback_mode="reciprocal_attention",
            meta_pc_reciprocal_scale=scale,
        )
    )
    lower_state = torch.randn(2, 8, model.config.hidden_dim)
    cache = model._block(1, lower_state)
    upper_displacement = torch.randn_like(lower_state)
    expected_message = model._merge_heads(
        torch.matmul(
            cache.weights.transpose(-2, -1),
            model._split_heads(upper_displacement),
        )
    )
    expected = upper_displacement + scale * _rms_norm_vjp(
        lower_state,
        expected_message,
    )

    actual = model._meta_pc_feedback_signal(
        1,
        lower_state,
        cache,
        upper_displacement,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_message_pc_meta_learned_reciprocal_feedback_uses_independent_weights() -> None:
    torch.manual_seed(73)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            meta_pc_feedback_mode="learned_reciprocal",
            meta_pc_reciprocal_learning_rate=0.1,
        )
    )
    lower_state = torch.randn(2, 8, model.config.hidden_dim)
    cache = model._block(1, lower_state)
    upper_displacement = torch.randn_like(lower_state)
    model.feedback_weights[1].copy_(2.0 * torch.eye(model.config.hidden_dim))

    actual = model._meta_pc_feedback_signal(
        1,
        lower_state,
        cache,
        upper_displacement,
    )

    assert torch.allclose(actual, 2.0 * upper_displacement)


def test_message_pc_meta_reciprocal_mirroring_is_local_and_deterministic() -> None:
    torch.manual_seed(74)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            meta_pc_feedback_mode="learned_reciprocal",
            meta_pc_reciprocal_learning_rate=0.05,
        )
    )
    replay = copy.deepcopy(model)
    x = torch.randint(0, model.config.vocab_size, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)
    replay_states, _ = replay._forward_states(x, with_cache=False)
    block_zero_before = model.feedback_weights[0].clone()
    block_one_before = model.feedback_weights[1].clone()

    diagnostics = model._update_meta_pc_reciprocal_feedback(
        states,
        x,
        current_step=17,
    )
    replay._update_meta_pc_reciprocal_feedback(
        replay_states,
        x,
        current_step=17,
    )

    assert torch.equal(model.feedback_weights[0], block_zero_before)
    assert not torch.equal(model.feedback_weights[1], block_one_before)
    assert torch.equal(model.feedback_weights[1], replay.feedback_weights[1])
    assert diagnostics[
        "update_diag_message_pc_meta_reciprocal_feedback_update_rms"
    ] > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_output_feedback_init_does_not_change_deployment_model() -> None:
    torch.manual_seed(75)
    exact = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            meta_pc_feedback_mode="exact_vjp",
            meta_pc_output_feedback_mode="decoder_transpose",
        )
    )
    torch.manual_seed(75)
    strict = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            meta_pc_feedback_mode="learned_reciprocal",
            meta_pc_reciprocal_learning_rate=0.01,
            meta_pc_output_feedback_mode="kp_learned",
        )
    )
    exact_parameters = dict(exact.named_parameters())
    strict_parameters = dict(strict.named_parameters())
    for name in exact_parameters.keys() & strict_parameters.keys():
        if name == "feedback_embedding":
            continue
        assert torch.equal(exact_parameters[name], strict_parameters[name]), name
    x = torch.randint(0, exact.config.vocab_size, (2, 8))

    assert torch.equal(exact(x), strict(x))
    assert not torch.equal(strict.feedback_embedding, strict.decoder_weight)


def test_message_pc_state_gated_reciprocal_depends_on_local_state() -> None:
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            meta_pc_feedback_mode="learned_reciprocal",
            meta_pc_reciprocal_learning_rate=0.05,
            meta_pc_state_gated_reciprocal=True,
        )
    )
    with torch.no_grad():
        model.feedback_weights[1].zero_()
        model.feedback_state_weights[1].copy_(
            torch.eye(model.config.hidden_dim)
        )
    x = torch.randint(0, model.config.vocab_size, (2, 8))
    states, caches = model._forward_states(x, with_cache=True)
    upper = torch.ones_like(states[2])

    positive = model._meta_pc_feedback_signal(
        1,
        torch.ones_like(states[1]),
        caches[1],
        upper,
    )
    negative = model._meta_pc_feedback_signal(
        1,
        -torch.ones_like(states[1]),
        caches[1],
        upper,
    )

    assert torch.all(positive > 0)
    assert torch.all(negative < 0)


def test_message_pc_state_gated_reciprocal_learns_from_local_response() -> None:
    torch.manual_seed(76)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            meta_pc_feedback_mode="learned_reciprocal",
            meta_pc_reciprocal_learning_rate=0.05,
            meta_pc_state_gated_reciprocal=True,
        )
    )
    x = torch.randint(0, model.config.vocab_size, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)
    before = model.feedback_state_weights[1].clone()

    diagnostics = model._update_meta_pc_reciprocal_feedback(
        states,
        x,
        current_step=3,
    )

    assert not torch.equal(model.feedback_state_weights[1], before)
    assert diagnostics[
        "update_diag_message_pc_meta_state_gated_feedback_update_rms"
    ] > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_state_gated_feedback_is_training_only_capacity() -> None:
    common = {
        "inference_mode": "meta_pc",
        "decoder_supervision_mode": "top_only",
        "forward_update_top_layers": 2,
        "meta_pc_feedback_mode": "learned_reciprocal",
        "meta_pc_reciprocal_learning_rate": 0.05,
        "meta_pc_output_feedback_mode": "kp_learned",
    }
    plain = MessagePassingPCLanguageModel(small_config(**common))
    gated = MessagePassingPCLanguageModel(
        small_config(**common, meta_pc_state_gated_reciprocal=True)
    )
    added = (
        (gated.config.num_layers - 1) * gated.config.hidden_dim**2
    )

    assert gated.deployment_parameter_count() == plain.deployment_parameter_count()
    assert (
        gated.training_feedback_parameter_count()
        - plain.training_feedback_parameter_count()
    ) == added


@pytest.mark.parametrize(
    ("update_before_settle", "expected"),
    [(False, ["settle", "update"]), (True, ["update", "settle"])],
)
def test_message_pc_meta_reciprocal_update_order_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
    update_before_settle: bool,
    expected: list[str],
) -> None:
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            meta_pc_feedback_mode="learned_reciprocal",
            meta_pc_reciprocal_learning_rate=0.05,
            meta_pc_reciprocal_update_before_settle=update_before_settle,
        )
    )
    events: list[str] = []
    original_settle = model._settle
    original_update = model._update_meta_pc_reciprocal_feedback

    def settle(*args: object, **kwargs: object) -> object:
        events.append("settle")
        return original_settle(*args, **kwargs)

    def update(*args: object, **kwargs: object) -> object:
        events.append("update")
        return original_update(*args, **kwargs)

    monkeypatch.setattr(model, "_settle", settle)
    monkeypatch.setattr(model, "_update_meta_pc_reciprocal_feedback", update)
    x = torch.randint(0, model.config.vocab_size, (2, 8))
    y = torch.randint(0, model.config.vocab_size, (2, 8))

    model.local_update(
        x,
        y,
        decoder_lr=0.001,
        state_lr=0.001,
        embed_lr=0.001,
        weight_decay=0.0,
        current_step=1,
    )

    assert events == expected


def test_message_pc_learned_reciprocal_counts_only_used_feedback_paths() -> None:
    identity = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            meta_pc_feedback_mode="reciprocal_attention",
            meta_pc_output_feedback_mode="kp_learned",
        )
    )
    learned = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            meta_pc_feedback_mode="learned_reciprocal",
            meta_pc_reciprocal_learning_rate=0.05,
            meta_pc_output_feedback_mode="kp_learned",
        )
    )

    assert learned.active_parameter_count() - identity.active_parameter_count() == (
        (learned.config.num_layers - 1) * learned.config.hidden_dim**2
    )
    assert learned.training_feedback_parameter_count() == (
        learned.feedback_embedding.numel()
        + (learned.config.num_layers - 1) * learned.config.hidden_dim**2
    )
    assert learned.deployment_parameter_count() == identity.deployment_parameter_count()


def test_message_pc_stage_reciprocal_matches_exact_vjp_with_exact_mirrors() -> None:
    torch.manual_seed(77)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            block_state_mode="nanochat_residual",
            ffn_dim=48,
            token_value_embeddings=True,
            sensory_reinjection_start=0.2,
            sensory_reinjection_end=0.05,
            sensory_reinjection_mode="pre_block",
            residual_stream_scale_start=1.15,
            residual_stream_scale_end=1.05,
            learn_residual_stream_scales=True,
            meta_pc_feedback_mode="stage_reciprocal",
            meta_pc_reciprocal_learning_rate=0.1,
        )
    )
    with torch.no_grad():
        model.message_weights[1].normal_(std=0.05)
        model.ffn_down_weights[1].normal_(std=0.05)
        model.feedback_query_weights[1].copy_(model.query_weights[1].t())
        model.feedback_key_weights[1].copy_(model.key_weights[1].t())
        model.feedback_value_weights[1].copy_(model.value_weights[1].t())
        model.feedback_message_weights[1].copy_(model.message_weights[1].t())
        model.feedback_ffn_up_weights[1].copy_(model.ffn_up_weights[1].t())
        model.feedback_ffn_down_weights[1].copy_(model.ffn_down_weights[1].t())
    x = torch.randint(0, model.config.vocab_size, (2, 8))
    states, caches = model._forward_states(x, with_cache=True)
    output_signal = torch.randn_like(states[2])

    expected = model._block_input_signal(
        1,
        states[1],
        caches[1],
        output_signal,
    )
    actual = model._stage_reciprocal_block_input_signal(
        1,
        states[1],
        caches[1],
        output_signal,
    )

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_message_pc_stage_reciprocal_learns_from_local_stage_responses() -> None:
    torch.manual_seed(78)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            block_state_mode="nanochat_residual",
            ffn_dim=48,
            meta_pc_feedback_mode="stage_reciprocal",
            meta_pc_reciprocal_learning_rate=0.1,
        )
    )
    with torch.no_grad():
        model.message_weights[1].normal_(std=0.05)
        model.ffn_down_weights[1].normal_(std=0.05)
    x = torch.randint(0, model.config.vocab_size, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)

    diagnostics = model._update_meta_pc_reciprocal_feedback(
        states,
        x,
        current_step=3,
    )

    assert torch.count_nonzero(model.feedback_query_weights[1]) > 0
    assert torch.count_nonzero(model.feedback_message_weights[1]) > 0
    assert torch.count_nonzero(model.feedback_ffn_up_weights[1]) > 0
    assert torch.count_nonzero(model.feedback_ffn_down_weights[1]) > 0
    assert diagnostics[
        "update_diag_message_pc_meta_stage_reciprocal_mirrors"
    ] == 12.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_stage_orthogonal_probes_recover_linear_reciprocal() -> None:
    torch.manual_seed(79)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            block_state_mode="nanochat_residual",
            meta_pc_feedback_mode="stage_reciprocal",
            meta_pc_reciprocal_learning_rate=1.0,
        )
    )
    input_state = torch.randn(2, 32, 32)
    forward_weight = torch.randn(48, 32)
    generator = torch.Generator().manual_seed(81)

    noise = model._stage_orthogonal_probe_noise(
        input_state,
        generator=generator,
    )
    target = model._linear_stage_mirror_target(
        input_state,
        forward_weight,
        generator=torch.Generator().manual_seed(81),
    )

    assert noise.shape == (32, 32)
    assert torch.allclose(noise.t() @ noise / 32.0, torch.eye(32))
    assert torch.allclose(target, forward_weight.t(), atol=2e-5, rtol=2e-5)


def test_message_pc_stage_update_recovers_all_reciprocals_in_one_phase() -> None:
    torch.manual_seed(82)
    model = MessagePassingPCLanguageModel(
        small_config(
            seq_len=32,
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            block_state_mode="nanochat_residual",
            ffn_dim=48,
            meta_pc_feedback_mode="stage_reciprocal",
            meta_pc_reciprocal_learning_rate=1.0,
        )
    )
    x = torch.randint(0, model.config.vocab_size, (2, 32))
    states, _ = model._forward_states(x, with_cache=False)

    model._update_stage_reciprocal_feedback(states, x, current_step=1)

    pairs = (
        (model.feedback_query_weights[1], model.query_weights[1]),
        (model.feedback_key_weights[1], model.key_weights[1]),
        (model.feedback_value_weights[1], model.value_weights[1]),
        (model.feedback_message_weights[1], model.message_weights[1]),
        (model.feedback_ffn_up_weights[1], model.ffn_up_weights[1]),
        (model.feedback_ffn_down_weights[1], model.ffn_down_weights[1]),
    )
    for mirror, forward in pairs:
        assert torch.allclose(mirror, forward.t(), atol=2e-5, rtol=2e-5)


def test_message_pc_stage_reciprocal_is_training_only_capacity() -> None:
    common = {
        "inference_mode": "meta_pc",
        "decoder_supervision_mode": "top_only",
        "forward_update_top_layers": 2,
        "block_state_mode": "nanochat_residual",
        "ffn_dim": 48,
        "meta_pc_output_feedback_mode": "kp_learned",
    }
    control = MessagePassingPCLanguageModel(
        small_config(**common, meta_pc_feedback_mode="reciprocal_attention")
    )
    stage = MessagePassingPCLanguageModel(
        small_config(
            **common,
            meta_pc_feedback_mode="stage_reciprocal",
            meta_pc_reciprocal_learning_rate=0.1,
        )
    )
    per_layer = (
        4 * stage.config.hidden_dim**2
        + 2 * stage.config.hidden_dim * stage.config.ffn_dim
    )

    assert stage.deployment_parameter_count() == control.deployment_parameter_count()
    assert (
        stage.training_feedback_parameter_count()
        - control.training_feedback_parameter_count()
    ) == stage.config.num_layers * per_layer
    assert stage.connection_stats()[
        "message_pc_meta_stage_reciprocal_params"
    ] == stage.config.num_layers * per_layer
    assert (
        stage.persistent_slot_count() - control.persistent_slot_count()
    ) == stage.config.num_layers * per_layer
    assert not any(
        name.startswith(
            (
                "feedback_query_weights.",
                "feedback_key_weights.",
                "feedback_value_weights.",
                "feedback_message_weights.",
                "feedback_ffn_up_weights.",
                "feedback_ffn_down_weights.",
            )
        )
        for name in (*stage._adam_buffer_names, *stage._muon_buffer_names)
    )


def test_message_pc_stage_reciprocal_state_dict_preserves_mirrors() -> None:
    config = small_config(
        inference_mode="meta_pc",
        decoder_supervision_mode="top_only",
        forward_update_top_layers=2,
        block_state_mode="nanochat_residual",
        ffn_dim=48,
        meta_pc_feedback_mode="stage_reciprocal",
        meta_pc_reciprocal_learning_rate=1.0,
    )
    model = MessagePassingPCLanguageModel(config)
    model.feedback_query_weights[1].normal_()
    model.feedback_ffn_down_weights[1].normal_()
    reloaded = MessagePassingPCLanguageModel(config)

    reloaded.load_state_dict(model.state_dict())

    assert torch.equal(
        reloaded.feedback_query_weights[1],
        model.feedback_query_weights[1],
    )
    assert torch.equal(
        reloaded.feedback_ffn_down_weights[1],
        model.feedback_ffn_down_weights[1],
    )


def test_message_pc_stage_reciprocal_preserves_deployment_rng() -> None:
    common = {
        "inference_mode": "meta_pc",
        "decoder_supervision_mode": "top_only",
        "forward_update_top_layers": 2,
        "block_state_mode": "nanochat_residual",
        "ffn_dim": 48,
        "meta_pc_output_feedback_mode": "kp_learned",
    }
    torch.manual_seed(83)
    exact = MessagePassingPCLanguageModel(
        small_config(**common, meta_pc_feedback_mode="exact_vjp")
    )
    torch.manual_seed(83)
    stage = MessagePassingPCLanguageModel(
        small_config(
            **common,
            meta_pc_feedback_mode="stage_reciprocal",
            meta_pc_reciprocal_learning_rate=1.0,
        )
    )
    stage_parameters = dict(stage.named_parameters())
    for name, parameter in exact.named_parameters():
        if name.startswith("feedback_weights."):
            continue
        assert torch.equal(stage_parameters[name], parameter), name
    x = torch.randint(0, exact.config.vocab_size, (2, 8))

    exact_states, _ = exact._forward_states(x, with_cache=False)
    stage_states, _ = stage._forward_states(x, with_cache=False)

    assert torch.equal(stage_states[-1], exact_states[-1])


def test_message_pc_stage_reciprocal_matches_exact_local_update() -> None:
    common = {
        "seq_len": 32,
        "inference_mode": "meta_pc",
        "decoder_supervision_mode": "top_only",
        "forward_update_top_layers": 2,
        "block_state_mode": "nanochat_residual",
        "ffn_dim": 48,
        "token_value_embeddings": True,
        "sensory_reinjection_start": 0.2,
        "sensory_reinjection_end": 0.05,
        "sensory_reinjection_mode": "pre_block",
        "residual_stream_scale_start": 1.15,
        "residual_stream_scale_end": 1.05,
        "learn_residual_stream_scales": True,
        "meta_pc_output_feedback_mode": "kp_learned",
        "matrix_optimizer": "adam",
    }
    torch.manual_seed(84)
    exact = MessagePassingPCLanguageModel(
        small_config(**common, meta_pc_feedback_mode="exact_vjp")
    )
    torch.manual_seed(84)
    stage = MessagePassingPCLanguageModel(
        small_config(
            **common,
            meta_pc_feedback_mode="stage_reciprocal",
            meta_pc_reciprocal_learning_rate=1.0,
            meta_pc_reciprocal_update_before_settle=True,
        )
    )
    generator = torch.Generator().manual_seed(85)
    x = torch.randint(0, exact.config.vocab_size, (2, 32), generator=generator)
    y = torch.randint(0, exact.config.vocab_size, (2, 32), generator=generator)

    exact.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=1,
    )
    stage.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=1,
    )

    stage_parameters = dict(stage.named_parameters())
    for name, parameter in exact.named_parameters():
        if name.startswith(("feedback_weights.", "generative_weights.")):
            continue
        assert torch.allclose(
            stage_parameters[name],
            parameter,
            atol=2e-6,
            rtol=2e-6,
        ), name


def test_message_pc_meta_variance_normalization_sets_unit_spectral_proxy() -> None:
    torch.manual_seed(72)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
        )
    )
    model.query_weights[0].normal_(mean=0.0, std=0.2)
    model.value_weights[1].normal_(mean=0.0, std=0.05)
    model.message_weights[0].zero_()
    model.message_weights[1].normal_(mean=0.0, std=0.01)

    mean_scale = model._meta_pc_variance_normalize_forward_weights()

    assert mean_scale > 0
    assert torch.count_nonzero(model.message_weights[0]) == 0
    for weight in (
        *model.query_weights,
        *model.key_weights,
        *model.value_weights,
        model.message_weights[1],
    ):
        proxy = (
            weight.shape[0] ** 0.5 + weight.shape[1] ** 0.5
        ) * weight.float().std(unbiased=False)
        assert proxy.item() == pytest.approx(1.0, rel=1e-5, abs=1e-6)


def test_message_pc_token_value_gradient_matches_finite_difference() -> None:
    torch.manual_seed(73)
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            token_value_embeddings=True,
            token_value_scale=1.2,
            sensory_reinjection_start=0.2,
            sensory_reinjection_end=0.05,
        )
    ).double()
    x = torch.tensor([[3, 5, 7, 3, 11, 13, 17, 19]])
    states, _ = model._forward_states(x, with_cache=False)
    target = states[2] + 0.1 * torch.randn_like(states[2])
    gradients, *_ = model._phase_block_gradients(
        1,
        states[1],
        target,
        route_lower_state=states[1],
        causal_mask=model._causal_mask(x.shape[1], x.device),
        token_count=float(x.numel()),
        token_ids=x,
        sensory_state=states[0],
    )
    table = model.token_value_tables["1"]
    token_id = 3
    channel = 4
    epsilon = 1e-5

    def objective() -> float:
        cache = model._block(
            1,
            states[1],
            token_ids=x,
            sensory_state=states[0],
        )
        return float(
            (cache.prediction - target).square().sum().item()
            / (2.0 * x.numel())
        )

    original = float(table[token_id, channel].item())
    table[token_id, channel] = original + epsilon
    positive = objective()
    table[token_id, channel] = original - epsilon
    negative = objective()
    table[token_id, channel] = original
    numerical = (positive - negative) / (2.0 * epsilon)

    assert gradients["token_value_tables.1"][token_id, channel].item() == pytest.approx(
        numerical,
        rel=2e-4,
        abs=2e-6,
    )


def test_message_pc_sensory_reinjection_persists_input_across_depth() -> None:
    torch.manual_seed(74)
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            sensory_reinjection_start=0.2,
            sensory_reinjection_end=0.05,
        )
    )
    for weight in (*model.message_weights, *model.self_weights):
        weight.zero_()
    x = torch.randint(0, 31, (2, 8))

    states, _ = model._forward_states(x, with_cache=False)

    assert torch.allclose(states[1], 1.2 * states[0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(states[2], 1.25 * states[0], atol=1e-6, rtol=1e-6)


def test_message_pc_residual_stream_schedule_scales_depth() -> None:
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            residual_stream_scale_start=1.15,
            residual_stream_scale_end=1.05,
        )
    )
    for weight in (*model.message_weights, *model.self_weights):
        weight.zero_()
    x = torch.randint(0, 31, (2, 8))

    states, _ = model._forward_states(x, with_cache=False)

    assert torch.allclose(states[1], 1.15 * states[0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(states[2], 1.05 * states[1], atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="residual stream scales"):
        MessagePassingPCLanguageModel(
            small_config(residual_stream_scale_start=0.0)
        )
    with pytest.raises(ValueError, match="residual stream LR scales"):
        MessagePassingPCLanguageModel(
            small_config(residual_stream_lr_scale=-0.1)
        )
    with pytest.raises(ValueError, match="schedule end"):
        MessagePassingPCLanguageModel(
            small_config(
                residual_stream_lr_scale_final=2.0,
                residual_stream_lr_schedule_start_step=100,
                residual_stream_lr_schedule_end_step=100,
            )
        )
    with pytest.raises(ValueError, match="learned residual stream scales"):
        MessagePassingPCLanguageModel(
            small_config(learn_residual_stream_scales=True)
        )
    with pytest.raises(ValueError, match="residual_stream_scale_mode"):
        MessagePassingPCLanguageModel(
            small_config(residual_stream_scale_mode="unknown")
        )
    with pytest.raises(ValueError, match="must be learned"):
        MessagePassingPCLanguageModel(
            small_config(residual_stream_scale_mode="channel")
        )


def test_message_pc_learned_residual_stream_scale_gradient_matches_finite_difference() -> None:
    torch.manual_seed(81)
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            ffn_dim=48,
            sensory_reinjection_start=0.2,
            sensory_reinjection_end=0.05,
            sensory_reinjection_mode="pre_block",
            residual_stream_scale_start=1.15,
            residual_stream_scale_end=1.05,
            learn_residual_stream_scales=True,
        )
    ).double()
    x = torch.tensor([[3, 5, 7, 11, 13, 17, 19, 23]])
    states, _ = model._forward_states(x, with_cache=False)
    target = states[1] + 0.1 * torch.randn_like(states[1])
    gradients, *_ = model._phase_block_gradients(
        0,
        states[0],
        target,
        route_lower_state=states[0],
        causal_mask=model._causal_mask(x.shape[1], x.device),
        token_count=float(x.numel()),
        token_ids=x,
        sensory_state=states[0],
    )

    def objective() -> float:
        prediction = model._block(
            0,
            states[0],
            token_ids=x,
            sensory_state=states[0],
        ).prediction
        return float(
            (prediction - target).square().sum().item()
            / (2.0 * x.numel())
        )

    assert model.residual_stream_scales is not None
    original = float(model.residual_stream_scales[0].item())
    epsilon = 1e-5
    model.residual_stream_scales[0] = original + epsilon
    positive = objective()
    model.residual_stream_scales[0] = original - epsilon
    negative = objective()
    model.residual_stream_scales[0] = original
    numerical = (positive - negative) / (2.0 * epsilon)

    assert gradients["residual_stream_scales"][0].item() == pytest.approx(
        numerical,
        rel=3e-4,
        abs=2e-7,
    )
    assert gradients["residual_stream_scales"][1].item() == 0.0


def test_message_pc_local_update_learns_residual_stream_scales_without_grad() -> None:
    torch.manual_seed(82)
    fixed = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            ffn_dim=48,
            sensory_reinjection_start=0.2,
            sensory_reinjection_end=0.05,
            sensory_reinjection_mode="pre_block",
            residual_stream_scale_start=1.15,
            residual_stream_scale_end=1.05,
        )
    )
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            ffn_dim=48,
            sensory_reinjection_start=0.2,
            sensory_reinjection_end=0.05,
            sensory_reinjection_mode="pre_block",
            residual_stream_scale_start=1.15,
            residual_stream_scale_end=1.05,
            learn_residual_stream_scales=True,
        )
    )
    assert model.residual_stream_scales is not None
    assert torch.allclose(
        model.residual_stream_scales,
        torch.tensor([1.15, 1.05]),
    )
    assert model.active_parameter_count() == fixed.active_parameter_count() + 2
    before = model.residual_stream_scales.clone()
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))

    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
    )

    assert not torch.equal(model.residual_stream_scales, before)
    assert model.residual_stream_scales.grad is None
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.connection_stats()[
        "message_pc_learns_residual_stream_scales"
    ] == 1.0


def test_message_pc_channel_residual_precision_gradient_matches_finite_difference() -> None:
    torch.manual_seed(83)
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            ffn_dim=48,
            sensory_reinjection_start=0.2,
            sensory_reinjection_end=0.05,
            sensory_reinjection_mode="pre_block",
            residual_stream_scale_start=1.15,
            residual_stream_scale_end=1.05,
            learn_residual_stream_scales=True,
            residual_stream_scale_mode="channel",
        )
    ).double()
    assert model.residual_stream_scales is not None
    assert model.residual_stream_scales.shape == (2, 32)
    assert torch.allclose(
        model.residual_stream_scales[0],
        torch.full((32,), 1.15, dtype=torch.float64),
    )
    assert torch.allclose(
        model.residual_stream_scales[1],
        torch.full((32,), 1.05, dtype=torch.float64),
    )
    x = torch.tensor([[3, 5, 7, 11, 13, 17, 19, 23]])
    states, _ = model._forward_states(x, with_cache=False)
    target = states[1] + 0.1 * torch.randn_like(states[1])
    gradients, *_ = model._phase_block_gradients(
        0,
        states[0],
        target,
        route_lower_state=states[0],
        causal_mask=model._causal_mask(x.shape[1], x.device),
        token_count=float(x.numel()),
        token_ids=x,
        sensory_state=states[0],
    )

    def objective() -> float:
        prediction = model._block(
            0,
            states[0],
            token_ids=x,
            sensory_state=states[0],
        ).prediction
        return float(
            (prediction - target).square().sum().item()
            / (2.0 * x.numel())
        )

    channel = 7
    original = float(model.residual_stream_scales[0, channel].item())
    epsilon = 1e-5
    model.residual_stream_scales[0, channel] = original + epsilon
    positive = objective()
    model.residual_stream_scales[0, channel] = original - epsilon
    negative = objective()
    model.residual_stream_scales[0, channel] = original
    numerical = (positive - negative) / (2.0 * epsilon)

    assert gradients["residual_stream_scales"][0, channel].item() == pytest.approx(
        numerical,
        rel=3e-4,
        abs=2e-7,
    )
    assert torch.count_nonzero(
        gradients["residual_stream_scales"][1]
    ) == 0
    fixed = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            ffn_dim=48,
            sensory_reinjection_start=0.2,
            sensory_reinjection_end=0.05,
            sensory_reinjection_mode="pre_block",
            residual_stream_scale_start=1.15,
            residual_stream_scale_end=1.05,
        )
    )
    assert model.active_parameter_count() == fixed.active_parameter_count() + 64
    assert model.connection_stats()[
        "message_pc_residual_stream_uses_channel_precision"
    ] == 1.0


def test_message_pc_residual_precision_lr_schedule_changes_after_reversal() -> None:
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            learn_residual_stream_scales=True,
            residual_stream_lr_scale=1.0,
            residual_stream_lr_scale_final=3.0,
            residual_stream_lr_schedule_start_step=100,
            residual_stream_lr_schedule_end_step=500,
        )
    )

    assert model._residual_stream_lr_scale(50) == pytest.approx(1.0)
    assert model._residual_stream_lr_scale(100) == pytest.approx(1.0)
    assert model._residual_stream_lr_scale(300) == pytest.approx(2.0)
    assert model._residual_stream_lr_scale(500) == pytest.approx(3.0)
    assert model._residual_stream_lr_scale(700) == pytest.approx(3.0)


def test_message_pc_pre_block_sensory_input_and_vjp_match() -> None:
    torch.manual_seed(80)
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            sensory_reinjection_start=0.2,
            sensory_reinjection_end=0.2,
            sensory_reinjection_mode="pre_block",
            residual_stream_scale_start=1.15,
            residual_stream_scale_end=1.15,
        )
    ).double()
    model.message_weights[0].normal_(mean=0.0, std=0.15)
    lower_state = torch.randn(1, 4, 32, dtype=torch.float64) * 0.2
    sensory_state = torch.randn_like(lower_state) * 0.2
    output_signal = torch.randn_like(lower_state) * 0.2
    cache = model._block(0, lower_state, sensory_state=sensory_state)
    expected_input = 1.15 * lower_state + 0.2 * sensory_state

    assert torch.allclose(cache.source_input, expected_input)
    assert torch.allclose(cache.source, F.normalize(expected_input, dim=-1) * math.sqrt(32), atol=2e-4, rtol=2e-4)
    analytic = model._block_input_signal(0, lower_state, cache, output_signal)

    def objective(value: torch.Tensor) -> float:
        prediction = model._block(
            0,
            value,
            sensory_state=sensory_state,
        ).prediction
        return float((prediction * output_signal).sum().item())

    epsilon = 1e-6
    for index in ((0, 0, 0), (0, 2, 7), (0, 3, 19)):
        plus = lower_state.clone()
        minus = lower_state.clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical = (objective(plus) - objective(minus)) / (2.0 * epsilon)
        assert analytic[index].item() == pytest.approx(
            numerical,
            rel=3e-4,
            abs=1e-8,
        )

    with pytest.raises(ValueError, match="sensory_reinjection_mode"):
        MessagePassingPCLanguageModel(
            small_config(sensory_reinjection_mode="unknown")
        )


def test_message_pc_matched_token_value_capacity() -> None:
    model = MessagePassingPCLanguageModel(
        MessagePassingPCConfig(
            vocab_size=4096,
            bos_token_id=4095,
            seq_len=1024,
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            inference_steps=1,
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            decoder_state_mode="rms_norm",
            forward_update_top_layers=4,
            ffn_dim=1022,
            token_value_embeddings=True,
        )
    )

    assert model.active_parameter_count() == 7_340_032
    assert sum(table.numel() for table in model.token_value_tables.values()) == 2_097_152
    stats = model.connection_stats()
    assert stats["message_pc_uses_token_value_embeddings"] == 1.0
    assert stats["message_pc_token_value_params"] == 2_097_152.0


def test_message_pc_causal_encoder_decoder_has_no_future_leak() -> None:
    torch.manual_seed(89)
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            causal_encoder_decoder=True,
            causal_encoder_depth=1,
        )
    )
    for weight in model.cross_message_weights:
        weight.normal_(mean=0.0, std=0.08)
    prefix = torch.tensor([[3, 5, 7, 11]])
    first = torch.cat((prefix, torch.tensor([[13, 17, 19, 23]])), dim=1)
    second = torch.cat((prefix, torch.tensor([[2, 4, 6, 8]])), dim=1)

    first_stack, first_caches = model._forward_states(
        first,
        with_cache=True,
    )
    second_states = model.forward_states(second)

    assert first_caches[0].cross_attention is None
    assert first_caches[1].cross_attention is not None
    assert torch.equal(
        first_caches[1].cross_attention.memory_input,
        first_stack[1],
    )
    assert torch.equal(first_stack[-1][:, :4], second_states[:, :4])
    assert not torch.equal(first_stack[-1][:, 4:], second_states[:, 4:])


def test_message_pc_causal_encoder_decoder_gradients_match_autograd() -> None:
    torch.manual_seed(90)
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            causal_encoder_decoder=True,
        )
    ).double()
    model.cross_message_weights[1].normal_(mean=0.0, std=0.05)
    lower_state = torch.randn(1, 4, 32, dtype=torch.float64)
    sensory_state = torch.randn_like(lower_state)
    target_state = torch.randn_like(lower_state)
    names = (
        "cross_query_weights.1",
        "cross_key_weights.1",
        "cross_value_weights.1",
        "cross_message_weights.1",
        "cross_attention_gates",
    )
    parameters = dict(model.named_parameters())
    for name in names:
        parameters[name].requires_grad_(True)

    prediction = model._block(
        1,
        lower_state,
        sensory_state=sensory_state,
    ).prediction
    objective = 0.5 * (prediction - target_state).square().sum() / 4.0
    oracle = torch.autograd.grad(
        objective,
        [parameters[name] for name in names],
    )
    local, *_ = model._phase_block_gradients(
        1,
        lower_state,
        target_state,
        route_lower_state=lower_state,
        causal_mask=model._causal_mask(4, lower_state.device),
        token_count=4.0,
        sensory_state=sensory_state,
    )

    for name, oracle_gradient in zip(names, oracle):
        assert torch.allclose(
            local[name],
            oracle_gradient,
            rtol=5e-7,
            atol=5e-9,
        ), name


@pytest.mark.parametrize("memory_is_lower_state", [False, True])
def test_message_pc_causal_encoder_decoder_input_vjp_matches_autograd(
    memory_is_lower_state: bool,
) -> None:
    torch.manual_seed(91)
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            causal_encoder_decoder=True,
        )
    ).double()
    layer = 0 if memory_is_lower_state else 1
    model.message_weights[layer].normal_(mean=0.0, std=0.04)
    model.cross_message_weights[layer].normal_(mean=0.0, std=0.04)
    lower_state = torch.randn(
        1,
        4,
        32,
        dtype=torch.float64,
        requires_grad=True,
    )
    sensory_state = (
        lower_state
        if memory_is_lower_state
        else torch.randn_like(lower_state).detach()
    )
    output_signal = torch.randn_like(lower_state)
    cache = model._block(
        layer,
        lower_state,
        sensory_state=sensory_state,
    )
    (oracle,) = torch.autograd.grad(
        (cache.prediction * output_signal).sum(),
        (lower_state,),
    )

    local = model._block_input_signal(
        layer,
        lower_state.detach(),
        cache,
        output_signal,
    )

    assert torch.allclose(local, oracle, rtol=8e-7, atol=8e-9)


def test_message_pc_cross_memory_skip_signal_matches_autograd() -> None:
    torch.manual_seed(93)
    model = MessagePassingPCLanguageModel(
        small_config(
            num_layers=3,
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=3,
            causal_encoder_decoder=True,
            causal_encoder_depth=1,
        )
    ).double()
    model.cross_message_weights[2].normal_(mean=0.0, std=0.04)
    lower_state = torch.randn(1, 4, 32, dtype=torch.float64)
    encoder_memory = torch.randn_like(lower_state, requires_grad=True)
    output_signal = torch.randn_like(lower_state)
    cache = model._block(
        2,
        lower_state,
        sensory_state=torch.randn_like(lower_state),
        encoder_memory=encoder_memory,
    )
    (oracle,) = torch.autograd.grad(
        (cache.prediction * output_signal).sum(),
        (encoder_memory,),
    )

    local = model._block_cross_memory_signal(2, cache, output_signal)

    assert torch.allclose(local, oracle, rtol=8e-7, atol=8e-9)


def test_message_pc_cross_memory_skip_credit_reaches_encoder_state() -> None:
    torch.manual_seed(94)
    base_config = small_config(
        num_layers=3,
        block_state_mode="nanochat_residual",
        inference_mode="meta_pc",
        decoder_supervision_mode="top_only",
        decoder_state_mode="rms_norm",
        forward_update_top_layers=3,
        causal_encoder_decoder=True,
        causal_encoder_depth=1,
        cross_attention_output_init_gain=0.1,
    )
    baseline = MessagePassingPCLanguageModel(base_config)
    skip_model = MessagePassingPCLanguageModel(
        replace(base_config, meta_pc_cross_memory_skip_credit=True)
    )
    skip_model.load_state_dict(baseline.state_dict())
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    free_states, _ = baseline._forward_states(x, with_cache=False)

    with torch.no_grad():
        baseline_states, _ = baseline._settle_meta_pc(
            free_states,
            y,
            inference_lr=0.12,
            token_ids=x,
        )
        skip_states, diagnostics = skip_model._settle_meta_pc(
            free_states,
            y,
            inference_lr=0.12,
            token_ids=x,
        )

    assert diagnostics[
        "update_diag_message_pc_meta_uses_cross_memory_skip_credit"
    ] == 1.0
    assert diagnostics[
        "update_diag_message_pc_meta_cross_memory_skip_credit_rms"
    ] > 0.0
    assert not torch.equal(skip_states[1], baseline_states[1])


def test_message_pc_causal_encoder_decoder_updates_without_parameter_grad() -> None:
    torch.manual_seed(92)
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            decoder_state_mode="rms_norm",
            forward_update_top_layers=2,
            causal_encoder_decoder=True,
        )
    )
    for weight in model.cross_message_weights:
        weight.normal_(mean=0.0, std=0.04)
    before_query = model.cross_query_weights[0].clone()
    before_gate = model.cross_attention_gates.clone()
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))

    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
    )

    assert not torch.equal(model.cross_query_weights[0], before_query)
    assert not torch.equal(model.cross_attention_gates, before_gate)
    assert model.cross_attention_gates.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.connection_stats()[
        "message_pc_uses_causal_encoder_decoder"
    ] == 1.0


def test_message_pc_causal_encoder_decoder_matches_gpt_parameter_budget() -> None:
    model = MessagePassingPCLanguageModel(
        MessagePassingPCConfig(
            vocab_size=4096,
            bos_token_id=4095,
            seq_len=1024,
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            inference_steps=1,
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            decoder_state_mode="rms_norm",
            forward_update_top_layers=4,
            ffn_dim=766,
            token_value_embeddings=True,
            causal_encoder_decoder=True,
            causal_encoder_depth=2,
            learn_residual_stream_scales=True,
        )
    )

    assert model.deployment_parameter_count() == 7_340_038
    assert model.connection_stats()[
        "message_pc_cross_attention_params"
    ] == 524_290.0


def test_message_pc_validates_causal_encoder_decoder_contract() -> None:
    with pytest.raises(ValueError, match="causal encoder-decoder"):
        MessagePassingPCLanguageModel(
            small_config(causal_encoder_decoder=True)
        )
    with pytest.raises(ValueError, match="cross-attention LR scales"):
        MessagePassingPCLanguageModel(
            small_config(cross_attention_lr_scale=-0.1)
        )
    with pytest.raises(ValueError, match="output_init_gain"):
        MessagePassingPCLanguageModel(
            small_config(cross_attention_output_init_gain=-0.1)
        )
    with pytest.raises(ValueError, match="requires causal encoder-decoder"):
        MessagePassingPCLanguageModel(
            small_config(causal_encoder_depth=1)
        )
    with pytest.raises(ValueError, match="smaller than num_layers"):
        MessagePassingPCLanguageModel(
            small_config(
                block_state_mode="nanochat_residual",
                inference_mode="meta_pc",
                decoder_supervision_mode="top_only",
                forward_update_top_layers=2,
                causal_encoder_decoder=True,
                causal_encoder_depth=2,
            )
        )
    with pytest.raises(ValueError, match="cross_memory_skip_credit"):
        MessagePassingPCLanguageModel(
            small_config(meta_pc_cross_memory_skip_credit=True)
        )
    with pytest.raises(ValueError, match="skip_credit_scale"):
        MessagePassingPCLanguageModel(
            small_config(meta_pc_cross_memory_skip_credit_scale=-0.1)
        )


def test_message_pc_validates_independent_meta_output_feedback() -> None:
    with pytest.raises(ValueError, match="meta_pc_output_feedback_mode"):
        MessagePassingPCLanguageModel(
            small_config(meta_pc_output_feedback_mode="unknown")
        )
    with pytest.raises(ValueError, match="meta_pc_output_feedback_lr_scale"):
        MessagePassingPCLanguageModel(
            small_config(meta_pc_output_feedback_lr_scale=-0.1)
        )
    with pytest.raises(ValueError, match="requires meta_pc"):
        MessagePassingPCLanguageModel(
            small_config(meta_pc_output_feedback_mode="fixed_random")
        )
    with pytest.raises(ValueError, match="relinearize_inference"):
        MessagePassingPCLanguageModel(
            small_config(meta_pc_relinearize_inference=True)
        )
    with pytest.raises(ValueError, match="rank must be"):
        MessagePassingPCLanguageModel(
            small_config(
                inference_mode="meta_pc",
                decoder_supervision_mode="top_only",
                forward_update_top_layers=2,
                meta_pc_output_feedback_mode="kp_factorized",
            )
        )
    with pytest.raises(ValueError, match="requires kp_factorized"):
        MessagePassingPCLanguageModel(
            small_config(meta_pc_output_feedback_rank=8)
        )
    with pytest.raises(ValueError, match="shared output-feedback values"):
        MessagePassingPCLanguageModel(
            small_config(meta_pc_output_feedback_values=True)
        )


def test_message_pc_independent_output_feedback_does_not_copy_decoder() -> None:
    torch.manual_seed(75)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            meta_pc_output_feedback_mode="fixed_random",
        )
    )

    assert not torch.equal(model.feedback_embedding, model.decoder_weight)


def test_message_pc_shared_output_feedback_supplies_normalized_values() -> None:
    torch.manual_seed(81)
    model = MessagePassingPCLanguageModel(
        small_config(
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            meta_pc_output_feedback_mode="kp_learned",
            meta_pc_output_feedback_values=True,
            token_value_scale=1.25,
        )
    )
    model.value_weights[1].zero_()
    lower_state = torch.randn(2, 8, 32)
    token_ids = torch.randint(0, 31, (2, 8))

    cache = model._block(1, lower_state, token_ids=token_ids)
    actual = model._merge_heads(cache.value)
    expected = 1.25 * F.normalize(
        model.feedback_embedding[token_ids],
        dim=-1,
        eps=1e-8,
    )

    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_message_pc_independent_top_feedback_matches_local_codebook_vjp() -> None:
    torch.manual_seed(76)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            decoder_state_mode="rms_norm",
            forward_update_top_layers=2,
            meta_pc_output_feedback_mode="fixed_random",
        )
    )
    state = torch.randn(2, 8, 32)
    y = torch.randint(0, 31, (2, 8))

    actual, probabilities = model._meta_pc_top_feedback(state, y)
    prediction_state = model._prediction_states(state)
    expected = model.feedback_embedding[y] - probabilities @ model.feedback_embedding
    expected = _rms_norm_vjp(prediction_state, expected)
    expected = model._expand_prediction_signal(expected, state)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_message_pc_factorized_top_feedback_matches_materialized_codebook() -> None:
    torch.manual_seed(78)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            decoder_state_mode="rms_norm",
            forward_update_top_layers=2,
            meta_pc_output_feedback_mode="kp_factorized",
            meta_pc_output_feedback_rank=8,
        )
    )
    state = torch.randn(2, 8, 32)
    y = torch.randint(0, 31, (2, 8))
    assert model.feedback_token_factors is not None
    assert model.feedback_state_factors is not None
    codebook = model.feedback_token_factors @ model.feedback_state_factors
    assert model.feedback_token_factors.std().item() == pytest.approx(
        model.feedback_state_factors.std().item(),
        rel=0.2,
    )
    assert codebook.std().item() == pytest.approx(0.02, rel=0.35)

    actual, probabilities = model._meta_pc_top_feedback(state, y)
    expected = codebook[y] - probabilities @ codebook
    expected = _rms_norm_vjp(model._prediction_states(state), expected)
    expected = model._expand_prediction_signal(expected, state)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("mode", "feedback_changes"),
    (("fixed_random", False), ("kp_learned", True)),
)
def test_message_pc_kp_output_feedback_uses_local_decoder_association(
    mode: str,
    feedback_changes: bool,
) -> None:
    torch.manual_seed(77)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            meta_pc_output_feedback_mode=mode,
            meta_pc_output_feedback_lr_scale=1.0,
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    decoder_before = model.decoder_weight.clone()
    feedback_before = model.feedback_embedding.clone()

    model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=0.0,
        embed_lr=0.0,
        weight_decay=0.0,
    )

    decoder_delta = model.decoder_weight - decoder_before
    feedback_delta = model.feedback_embedding - feedback_before
    if feedback_changes:
        assert torch.allclose(feedback_delta, decoder_delta, atol=1e-8, rtol=1e-6)
    else:
        assert torch.count_nonzero(feedback_delta) == 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_matched_kp_output_feedback_capacity() -> None:
    model = MessagePassingPCLanguageModel(
        MessagePassingPCConfig(
            vocab_size=4096,
            bos_token_id=4095,
            seq_len=1024,
            hidden_dim=320,
            num_layers=4,
            num_heads=4,
            inference_steps=1,
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            decoder_state_mode="rms_norm",
            forward_update_top_layers=4,
            ffn_dim=688,
            meta_pc_output_feedback_mode="kp_learned",
            meta_pc_output_feedback_values=True,
        )
    )

    assert model.active_parameter_count() == 7_335_936
    assert model.connection_stats()[
        "message_pc_meta_uses_kp_output_feedback"
    ] == 1.0
    assert model.connection_stats()[
        "message_pc_meta_uses_output_feedback_values"
    ] == 1.0


def test_message_pc_factorized_feedback_gradients_chain_local_association() -> None:
    torch.manual_seed(79)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            meta_pc_output_feedback_mode="kp_factorized",
            meta_pc_output_feedback_rank=8,
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    assert model.feedback_token_factors is not None
    assert model.feedback_state_factors is not None
    token_factors = model.feedback_token_factors.clone()
    state_factors = model.feedback_state_factors.clone()

    gradients = capture_local_gradients(
        model,
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
        current_step=1,
    )

    decoder_gradient = gradients["decoder_weight"]
    assert torch.allclose(
        gradients["feedback_token_factors"],
        decoder_gradient @ state_factors.t(),
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.allclose(
        gradients["feedback_state_factors"],
        token_factors.t() @ decoder_gradient,
        atol=1e-7,
        rtol=1e-6,
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_message_pc_matched_factorized_output_feedback_capacity() -> None:
    model = MessagePassingPCLanguageModel(
        MessagePassingPCConfig(
            vocab_size=4096,
            bos_token_id=4095,
            seq_len=1024,
            hidden_dim=320,
            num_layers=4,
            num_heads=4,
            inference_steps=1,
            block_state_mode="nanochat_residual",
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            decoder_state_mode="rms_norm",
            forward_update_top_layers=4,
            ffn_dim=1090,
            meta_pc_output_feedback_mode="kp_factorized",
            meta_pc_output_feedback_rank=64,
        )
    )

    assert model.active_parameter_count() == 7_336_960
    stats = model.connection_stats()
    assert stats["message_pc_meta_uses_factorized_output_feedback"] == 1.0
    assert stats["message_pc_meta_output_feedback_rank"] == 64.0
    assert stats["message_pc_meta_output_feedback_factor_params"] == 282_624.0


def test_message_pc_meta_local_update_changes_forward_blocks_without_grad() -> None:
    torch.manual_seed(70)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="meta_pc",
            decoder_supervision_mode="top_only",
            forward_update_top_layers=2,
            inference_steps=4,
            inference_lr=0.2,
            token_feedback_scale=1.0,
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    before = [weight.clone() for weight in model.message_weights]

    loss = model.local_update(
        x,
        y,
        decoder_lr=1e-3,
        state_lr=1e-3,
        embed_lr=1e-3,
    )

    assert math.isfinite(loss)
    assert all(
        not torch.equal(weight, old)
        for weight, old in zip(model.message_weights, before)
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model._last_update_diagnostics[
        "update_diag_message_pc_meta_uses_frozen_predictions"
    ] == pytest.approx(1.0)


def test_message_pc_spiking_precision_tracks_error_wavefront() -> None:
    model = MessagePassingPCLanguageModel(
        small_config(
            num_layers=4,
            inference_steps=4,
            inference_lr=0.05,
            inference_mode="pc_energy",
            decoder_supervision_mode="top_only",
            spiking_precision_scale=1.0,
        )
    )

    prediction, output, multiplier = model._inference_precision_multipliers(
        0,
        0.05,
    )
    assert multiplier == pytest.approx(20.0)
    assert prediction == [1.0, 1.0, 1.0, 1.0]
    assert output == [1.0, 1.0, 1.0, 20.0]

    for inference_round, error_layer in ((1, 3), (2, 2), (3, 1), (4, 0)):
        prediction, output, _ = model._inference_precision_multipliers(
            inference_round,
            0.05,
        )
        assert prediction is not None
        assert output == [1.0, 1.0, 1.0, 1.0]
        assert prediction[error_layer] == pytest.approx(20.0)
        assert sum(value > 1.0 for value in prediction) == 1


def test_message_pc_spiking_deltas_match_weighted_energy_gradient() -> None:
    torch.manual_seed(56)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            decoder_supervision_mode="top_only",
            decoder_state_mode="rms_norm",
            block_state_mode="identity_residual",
            residual_scale=1.0,
            message_init_gain=1.0,
            state_delta_rms_cap=0.0,
            spiking_precision_scale=1.0,
        )
    ).double()
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    with torch.no_grad():
        free_states, _ = model._forward_states(x, with_cache=False)
    states = [free_states[0]] + [
        state.detach().requires_grad_(True) for state in free_states[1:]
    ]
    prediction, output, _ = model._inference_precision_multipliers(1, 0.05)

    energy = model._pc_energy(
        states,
        y,
        prediction_multipliers=prediction,
        output_multipliers=output,
    )
    oracle_gradients = torch.autograd.grad(energy, states[1:])
    local_deltas, _, _ = model._pc_energy_state_deltas(
        states,
        y,
        prediction_multipliers=prediction,
        output_multipliers=output,
    )

    for local_delta, oracle_gradient in zip(local_deltas, oracle_gradients):
        assert torch.allclose(
            local_delta,
            -oracle_gradient * x.numel(),
            rtol=2e-8,
            atol=2e-9,
        )


def test_message_pc_spiking_precision_amplifies_deep_wavefront() -> None:
    torch.manual_seed(57)
    uniform = MessagePassingPCLanguageModel(
        small_config(
            num_layers=4,
            inference_steps=4,
            inference_lr=0.01,
            inference_mode="pc_energy",
            decoder_supervision_mode="top_only",
            token_feedback_scale=0.1,
            state_delta_rms_cap=0.0,
        )
    )
    spiking = MessagePassingPCLanguageModel(
        replace(uniform.config, spiking_precision_scale=1.0)
    )
    spiking.load_state_dict(copy.deepcopy(uniform.state_dict()))
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    free_states, _ = uniform._forward_states(x, with_cache=False)

    uniform_states, uniform_diagnostics = uniform._settle(free_states, y)
    spiking_states, spiking_diagnostics = spiking._settle(free_states, y)
    uniform_bottom = float(
        (uniform_states[1] - free_states[1]).float().square().mean().sqrt().item()
    )
    spiking_bottom = float(
        (spiking_states[1] - free_states[1]).float().square().mean().sqrt().item()
    )

    assert spiking_bottom > 10.0 * uniform_bottom
    assert spiking_diagnostics[
        "update_diag_message_pc_spiking_precision_multiplier"
    ] == pytest.approx(100.0)
    assert uniform_diagnostics[
        "update_diag_message_pc_spiking_precision_multiplier"
    ] == pytest.approx(1.0)


@pytest.mark.parametrize("block_parameterization", ["standard", "depth_mup"])
def test_message_pc_state_deltas_match_declared_energy_gradient(
    block_parameterization: str,
) -> None:
    torch.manual_seed(36)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            decoder_supervision_mode="all_layers",
            decoder_state_mode="rms_norm",
            block_state_mode="identity_residual",
            block_parameterization=block_parameterization,
            residual_scale=1.0,
            message_init_gain=1.0,
            state_delta_rms_cap=0.0,
        )
    ).double()
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    with torch.no_grad():
        free_states, _ = model._forward_states(x, with_cache=False)
    differentiable_states = [free_states[0]] + [
        state.detach().requires_grad_(True) for state in free_states[1:]
    ]

    energy = model._pc_energy(differentiable_states, y)
    oracle_gradients = torch.autograd.grad(energy, differentiable_states[1:])
    local_deltas, _, _ = model._pc_energy_state_deltas(
        differentiable_states,
        y,
    )

    token_count = x.numel()
    for local_delta, oracle_gradient in zip(local_deltas, oracle_gradients):
        assert torch.allclose(
            local_delta,
            -oracle_gradient * token_count,
            rtol=2e-8,
            atol=2e-9,
        )


def test_message_pc_coherent_all_layer_objective_injects_direct_token_error() -> None:
    torch.manual_seed(29)
    all_layers = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            decoder_supervision_mode="all_layers",
            inference_steps=1,
            inference_lr=0.01,
        )
    )
    top_only = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            decoder_supervision_mode="top_only",
            inference_steps=1,
            inference_lr=0.01,
        )
    )
    top_only.load_state_dict(copy.deepcopy(all_layers.state_dict()))
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    free_states, _ = all_layers._forward_states(x, with_cache=False)

    all_settled, _ = all_layers._settle(free_states, y)
    top_settled, _ = top_only._settle(free_states, y)

    assert not torch.equal(all_settled[1], free_states[1])
    assert torch.equal(top_settled[1], free_states[1])
    assert all_layers._pc_energy(all_settled, y) < all_layers._pc_energy(free_states, y)


def test_message_pc_coherent_backtracking_guarantees_energy_descent() -> None:
    torch.manual_seed(30)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            decoder_supervision_mode="all_layers",
            block_state_mode="identity_residual",
            residual_scale=1.0,
            message_init_gain=1.0,
            inference_steps=4,
            inference_lr=10.0,
            state_delta_rms_cap=0.0,
            energy_backtracking_steps=16,
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    free_states, _ = model._forward_states(x, with_cache=False)

    settled_states, diagnostics = model._settle(free_states, y)

    assert model._pc_energy(settled_states, y) <= model._pc_energy(free_states, y)
    assert diagnostics["update_diag_message_pc_energy_reduction"] >= 0
    assert diagnostics["update_diag_message_pc_inference_backtracking_rejections"] > 0
    assert diagnostics["update_diag_message_pc_inference_accepted_lr"] < 10.0


@pytest.mark.parametrize(
    "inference_optimizer",
    ["heavy_ball", "conjugate_gradient"],
)
def test_message_pc_accelerated_inference_preserves_energy_descent(
    inference_optimizer: str,
) -> None:
    torch.manual_seed(54)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            inference_optimizer=inference_optimizer,
            decoder_supervision_mode="top_only",
            block_state_mode="identity_residual",
            inference_steps=12,
            inference_lr=0.02,
            token_feedback_scale=0.1,
            learned_feedback_proposal_scale=0.0,
            energy_backtracking_steps=8,
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    free_states, _ = model._forward_states(x, with_cache=False)

    settled_states, diagnostics = model._settle(free_states, y)

    assert model._pc_energy(settled_states, y) <= model._pc_energy(free_states, y)
    assert diagnostics["update_diag_message_pc_energy_reduction"] > 0
    assert diagnostics[
        "update_diag_message_pc_inference_acceleration_beta"
    ] >= 0


def test_message_pc_identity_inference_does_not_project_free_states() -> None:
    torch.manual_seed(37)
    model = MessagePassingPCLanguageModel(
        small_config(
            num_layers=4,
            inference_mode="pc_energy",
            decoder_supervision_mode="all_layers",
            block_state_mode="identity_residual",
            residual_scale=1.0,
            message_init_gain=1.0,
            inference_steps=4,
            inference_lr=0.001,
            energy_backtracking_steps=16,
        )
    )
    for bias in model.state_biases:
        bias.fill_(10.0)
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    free_states, _ = model._forward_states(x, with_cache=False)
    assert float(free_states[-1].max().item()) > 3.0

    settled_states, diagnostics = model._settle(free_states, y)

    assert diagnostics["update_diag_message_pc_energy_reduction"] > 0
    assert diagnostics["update_diag_message_pc_inference_accepted_lr"] > 0
    assert float(settled_states[-1].max().item()) > 3.0


def test_message_pc_coherent_update_uses_no_inactive_feedback_parameters() -> None:
    torch.manual_seed(28)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="pc_energy",
            route_update_mode="local_energy",
            decoder_supervision_mode="top_only",
            inference_steps=4,
            inference_lr=0.03,
        )
    )
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    inactive_names = {
        name
        for name, _ in model.named_parameters()
        if name == "feedback_embedding"
        or name.startswith(("feedback_weights.", "generative_weights."))
    }
    before = {
        name: parameter.clone()
        for name, parameter in model.named_parameters()
        if name in inactive_names
    }

    model.local_update(x, y, decoder_lr=1e-3, state_lr=1e-3, embed_lr=1e-3)

    for name, value in before.items():
        assert torch.equal(value, dict(model.named_parameters())[name]), name
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model._last_update_diagnostics["update_diag_message_pc_uses_coherent_energy"] == 1.0


def test_message_pc_component_scales_control_adam_step_size() -> None:
    torch.manual_seed(22)
    low = MessagePassingPCLanguageModel(
        small_config(
            route_lr_scale=0.1,
            feedback_lr_scale=0.1,
            generative_lr_scale=0.1,
        )
    )
    high = MessagePassingPCLanguageModel(
        small_config(
            route_lr_scale=0.2,
            feedback_lr_scale=0.2,
            generative_lr_scale=0.2,
        )
    )
    high.load_state_dict(copy.deepcopy(low.state_dict()))
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))

    low.local_update(x, y, decoder_lr=1e-3, state_lr=1e-3, embed_lr=1e-3)
    high.local_update(x, y, decoder_lr=1e-3, state_lr=1e-3, embed_lr=1e-3)

    for key in (
        "update_diag_message_pc_route_update_rms",
        "update_diag_message_pc_feedback_update_rms",
        "update_diag_message_pc_generative_update_rms",
    ):
        assert high._last_update_diagnostics[key] == pytest.approx(
            2.0 * low._last_update_diagnostics[key],
            rel=2e-3,
        )


def test_message_pc_top_only_decoder_changes_only_deployed_head_credit() -> None:
    torch.manual_seed(24)
    all_layers = MessagePassingPCLanguageModel(
        small_config(decoder_supervision_mode="all_layers")
    )
    top_only = MessagePassingPCLanguageModel(
        small_config(decoder_supervision_mode="top_only")
    )
    top_only.load_state_dict(copy.deepcopy(all_layers.state_dict()))
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))

    all_layers.local_update(x, y, decoder_lr=1e-3, state_lr=0.0, embed_lr=0.0)
    top_only.local_update(x, y, decoder_lr=1e-3, state_lr=0.0, embed_lr=0.0)

    assert not torch.equal(all_layers.decoder_weight, top_only.decoder_weight)
    for name, parameter in all_layers.named_parameters():
        if name not in {"decoder_weight", "decoder_bias"}:
            assert torch.equal(parameter, dict(top_only.named_parameters())[name]), name


def test_message_pc_top_feedback_propagates_through_learned_chain() -> None:
    torch.manual_seed(25)
    model = MessagePassingPCLanguageModel(
        small_config(token_feedback_mode="top_propagated")
    )
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    states, _ = model._forward_states(x, with_cache=False)

    sources, outputs = model._feedback_path(states, y)

    top_source, _ = model._token_feedback(states[-1], y)
    assert torch.equal(sources[-1], top_source)
    for layer in range(model.config.num_layers - 1, -1, -1):
        assert torch.equal(
            outputs[layer],
            F.linear(sources[layer], model.feedback_weights[layer]),
        )
        if layer > 0:
            assert torch.equal(sources[layer - 1], outputs[layer])


def test_message_pc_zero_component_scales_freeze_selected_paths() -> None:
    torch.manual_seed(23)
    model = MessagePassingPCLanguageModel(
        small_config(route_lr_scale=0.0, feedback_lr_scale=0.0, generative_lr_scale=0.0)
    )
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    frozen_names = {
        name
        for name, _ in model.named_parameters()
        if name.startswith(("query_weights.", "key_weights.", "feedback_weights.", "generative_weights."))
        or name == "feedback_embedding"
    }
    before = {
        name: parameter.clone()
        for name, parameter in model.named_parameters()
        if name in frozen_names
    }

    model.local_update(x, y, decoder_lr=1e-3, state_lr=1e-3, embed_lr=1e-3)

    for name, value in before.items():
        assert torch.equal(value, dict(model.named_parameters())[name]), name
    assert model._last_update_diagnostics["update_diag_message_pc_route_update_rms"] == 0.0
    assert model._last_update_diagnostics["update_diag_message_pc_feedback_update_rms"] == 0.0
    assert model._last_update_diagnostics["update_diag_message_pc_generative_update_rms"] == 0.0


@pytest.mark.parametrize("block_state_mode", ["squashed_state", "identity_residual"])
def test_message_pc_fixed_batch_learning_reduces_free_loss(block_state_mode: str) -> None:
    torch.manual_seed(3)
    model = MessagePassingPCLanguageModel(small_config(block_state_mode=block_state_mode))
    x = torch.randint(0, 31, (4, 8))
    y = (x + 1) % 31
    before = F.cross_entropy(model(x).reshape(-1, 32), y.reshape(-1)).item()

    for _ in range(20):
        model.local_update(x, y, decoder_lr=1e-3, state_lr=1e-3, embed_lr=1e-3)

    after = F.cross_entropy(model(x).reshape(-1, 32), y.reshape(-1)).item()
    assert after < before


def test_message_pc_her_fixed_batch_learning_reduces_deployed_loss() -> None:
    torch.manual_seed(44)
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            route_lr_scale=1.0,
            generative_lr_scale=1.0,
        )
    )
    x = torch.randint(0, 31, (4, 8))
    y = (x + 1) % 31
    before = F.cross_entropy(model(x).reshape(-1, 32), y.reshape(-1)).item()

    for _ in range(40):
        model.local_update(
            x,
            y,
            decoder_lr=1e-3,
            state_lr=1e-3,
            embed_lr=1e-3,
        )

    after = F.cross_entropy(model(x).reshape(-1, 32), y.reshape(-1)).item()
    assert after < before


def test_message_pc_checkpoint_restores_local_adam_exactly() -> None:
    torch.manual_seed(4)
    first = MessagePassingPCLanguageModel(small_config())
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    first.local_update(x, y, decoder_lr=1e-3, state_lr=1e-3, embed_lr=1e-3)
    checkpoint = copy.deepcopy(first.state_dict())

    resumed = MessagePassingPCLanguageModel(small_config())
    resumed.load_state_dict(checkpoint)
    first.local_update(x, y, decoder_lr=1e-3, state_lr=1e-3, embed_lr=1e-3)
    resumed.local_update(x, y, decoder_lr=1e-3, state_lr=1e-3, embed_lr=1e-3)

    assert first.state_dict().keys() == resumed.state_dict().keys()
    for name, value in first.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name


def test_message_pc_local_muon_update_is_finite_and_descent_aligned() -> None:
    torch.manual_seed(34)
    model = MessagePassingPCLanguageModel(
        small_config(
            matrix_optimizer="muon",
            muon_momentum=0.0,
            muon_beta2=0.0,
        )
    )
    name = "self_weights.0"
    parameter = dict(model.named_parameters())[name]
    parameter.zero_()
    gradient = torch.randn_like(parameter)

    delta_rms = model._muon_update(
        name,
        parameter,
        gradient,
        lr=0.02,
        weight_decay=0.0,
    )

    applied_descent = -parameter
    assert delta_rms > 0
    assert torch.isfinite(parameter).all()
    assert F.cosine_similarity(
        applied_descent.flatten(),
        gradient.flatten(),
        dim=0,
    ) > 0.5
    assert name in model._muon_buffer_names
    assert name not in model._adam_buffer_names


def test_message_pc_checkpoint_restores_local_muon_exactly() -> None:
    torch.manual_seed(35)
    first = MessagePassingPCLanguageModel(
        small_config(matrix_optimizer="muon")
    )
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))
    first.local_update(x, y, decoder_lr=1e-3, state_lr=1e-3, embed_lr=1e-3)
    checkpoint = copy.deepcopy(first.state_dict())

    resumed = MessagePassingPCLanguageModel(
        small_config(matrix_optimizer="muon")
    )
    resumed.load_state_dict(checkpoint)
    first.local_update(x, y, decoder_lr=1e-3, state_lr=1e-3, embed_lr=1e-3)
    resumed.local_update(x, y, decoder_lr=1e-3, state_lr=1e-3, embed_lr=1e-3)

    assert first._last_update_diagnostics[
        "update_diag_message_pc_uses_local_muon"
    ] == 1.0
    for name, value in first.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name


def test_message_pc_optimizer_state_reset_scopes() -> None:
    model = MessagePassingPCLanguageModel(small_config(matrix_optimizer="muon"))
    for mean_name, variance_name in model._adam_buffer_names.values():
        getattr(model, mean_name).fill_(1.0)
        getattr(model, variance_name).fill_(2.0)
    for momentum_name, variance_name in model._muon_buffer_names.values():
        getattr(model, momentum_name).fill_(3.0)
        getattr(model, variance_name).fill_(4.0)
    model.local_adam_step.fill_(17)

    muon_counts = model.reset_local_optimizer_state("muon")
    assert muon_counts == {"adam": 0, "muon": len(model._muon_buffer_names)}
    assert int(model.local_adam_step.item()) == 17
    assert all(
        torch.count_nonzero(getattr(model, buffer_name)) == 0
        for names in model._muon_buffer_names.values()
        for buffer_name in names
    )
    assert any(
        torch.count_nonzero(getattr(model, buffer_name)) > 0
        for names in model._adam_buffer_names.values()
        for buffer_name in names
    )

    context_counts = model.reset_local_optimizer_state("context")
    context_adam_names = {
        name
        for name in model._adam_buffer_names
        if name not in {"decoder_weight", "decoder_bias", "embedding.weight"}
    }
    assert context_counts["adam"] == len(context_adam_names)
    assert context_counts["muon"] == len(model._muon_buffer_names)
    decoder_mean_name, _ = model._adam_buffer_names["decoder_weight"]
    assert torch.count_nonzero(getattr(model, decoder_mean_name)) > 0
    assert int(model.local_adam_step.item()) == 17

    all_counts = model.reset_local_optimizer_state("all")
    assert all_counts == {
        "adam": len(model._adam_buffer_names),
        "muon": len(model._muon_buffer_names),
    }
    assert int(model.local_adam_step.item()) == 0
    assert all(
        torch.count_nonzero(getattr(model, buffer_name)) == 0
        for names in model._adam_buffer_names.values()
        for buffer_name in names
    )


def test_message_pc_optimizer_state_reset_rejects_unknown_scope() -> None:
    model = MessagePassingPCLanguageModel(small_config())

    with pytest.raises(ValueError, match="optimizer reset scope"):
        model.reset_local_optimizer_state("decoder")


def test_message_pc_optimizer_state_reset_can_target_routes() -> None:
    model = MessagePassingPCLanguageModel(small_config(matrix_optimizer="muon"))
    for momentum_name, variance_name in model._muon_buffer_names.values():
        getattr(model, momentum_name).fill_(1.0)
        getattr(model, variance_name).fill_(2.0)

    counts = model.reset_local_optimizer_state("route")

    route_names = {
        name
        for name in model._muon_buffer_names
        if name.startswith(("query_weights.", "key_weights."))
    }
    assert counts == {"adam": 0, "muon": len(route_names)}
    for name, buffer_names in model._muon_buffer_names.items():
        nonzero = sum(
            int(torch.count_nonzero(getattr(model, buffer_name)).item())
            for buffer_name in buffer_names
        )
        assert (nonzero == 0) == (name in route_names)


def test_message_pc_optimizer_state_reset_can_target_recursive_cp() -> None:
    model = MessagePassingPCLanguageModel(
        small_config(
            inference_mode="her_error_hierarchy",
            hierarchical_error_binding_mode="factorized",
            hierarchical_error_prediction_mode="recursive_cp",
            hierarchical_error_target_mode="fixed_outcome",
            hierarchical_error_recursive_cp_rank=4,
            matrix_optimizer="adam",
        )
    )
    recursive_names = {
        name
        for name in model._adam_buffer_names
        if name.startswith("recursive_cp_her.")
    }
    for names in model._adam_buffer_names.values():
        for buffer_name in names:
            getattr(model, buffer_name).fill_(1.0)

    counts = model.reset_local_optimizer_state("recursive_cp")

    assert counts == {"adam": len(recursive_names), "muon": 0}
    for name, names in model._adam_buffer_names.items():
        expected_nonzero = name not in recursive_names
        for buffer_name in names:
            assert bool(torch.count_nonzero(getattr(model, buffer_name))) == (
                expected_nonzero
            )


def test_message_pc_old_checkpoint_metadata_uses_compatible_route_default(tmp_path) -> None:
    config = small_config(route_update_mode="local_energy")
    model = MessagePassingPCLanguageModel(config)
    meta = {"model": "message_passing_pc_lm", **asdict(config)}
    meta.pop("route_update_mode")
    checkpoint = tmp_path / "old_message_pc.pt"
    torch.save({"model": model.state_dict(), "meta": meta, "step": 73}, checkpoint)

    loaded, loaded_meta = load_model(checkpoint, torch.device("cpu"))

    assert isinstance(loaded, MessagePassingPCLanguageModel)
    assert loaded.config.route_update_mode == "cosine_teacher"
    assert loaded.config.inference_mode == "heuristic"
    assert loaded_meta["step"] == 73


def test_message_pc_homeostasis_preserves_feedback_and_route_signal() -> None:
    torch.manual_seed(5)
    model = MessagePassingPCLanguageModel(small_config())
    model.feedback_embedding.zero_()
    model.query_weights[0].mul_(0.1)
    x = torch.randint(0, 31, (2, 8))
    y = torch.randint(0, 31, (2, 8))

    model.local_update(x, y, decoder_lr=1e-3, state_lr=1e-3, embed_lr=1e-3)

    feedback_cosine = F.cosine_similarity(
        model.feedback_embedding.flatten(),
        model.decoder_weight.flatten(),
        dim=0,
    )
    expected_route_rms = 1.0 / model.config.hidden_dim**0.5
    assert feedback_cosine > 0.7
    assert model.query_weights[0].square().mean().sqrt().item() == pytest.approx(
        expected_route_rms,
        rel=1e-5,
    )


def test_message_pc_capacity_counts_separate_learned_and_persistent_slots() -> None:
    model = MessagePassingPCLanguageModel(
        MessagePassingPCConfig(vocab_size=4096, bos_token_id=4095, seq_len=1024)
    )

    assert model.learned_parameter_count() == 6_804_736
    assert model.contextual_parameter_count() == 4_179_200
    assert model.persistent_slot_count() == 20_414_209
    assert model.active_parameter_count() == 6_804_736
    coherent = MessagePassingPCLanguageModel(
        MessagePassingPCConfig(
            vocab_size=4096,
            bos_token_id=4095,
            seq_len=1024,
            inference_mode="pc_energy",
        )
    )
    assert coherent.active_parameter_count() == 4_674_816
    frozen_routes = MessagePassingPCLanguageModel(
        MessagePassingPCConfig(
            vocab_size=4096,
            bos_token_id=4095,
            seq_len=1024,
            inference_mode="pc_energy",
            learned_feedback_proposal_scale=1.0,
            route_lr_scale=0.0,
        )
    )
    assert frozen_routes.active_parameter_count() == 5_575_936
    her = MessagePassingPCLanguageModel(
        MessagePassingPCConfig(
            vocab_size=4096,
            bos_token_id=4095,
            seq_len=1024,
            inference_mode="her_error_hierarchy",
        )
    )
    assert her.learned_parameter_count() == 6_804_736
    assert her.active_parameter_count() == 5_084_416
    her_base_one = MessagePassingPCLanguageModel(
        MessagePassingPCConfig(
            vocab_size=4096,
            bos_token_id=4095,
            seq_len=1024,
            inference_mode="her_error_hierarchy",
            hierarchical_error_base_depth=1,
        )
    )
    assert her_base_one.active_parameter_count() == 4_982_016
    assert count_params(model) == 6_804_736
    assert count_parameter_slots(model) == 20_414_209

    d320_ffn208 = MessagePassingPCLanguageModel(
        MessagePassingPCConfig(
            vocab_size=4096,
            bos_token_id=4095,
            seq_len=1024,
            hidden_dim=320,
            ffn_dim=208,
            inference_mode="pc_energy",
            learned_feedback_proposal_scale=1.0,
            route_lr_scale=0.0,
        )
    )
    d288_ffn624 = MessagePassingPCLanguageModel(
        MessagePassingPCConfig(
            vocab_size=4096,
            bos_token_id=4095,
            seq_len=1024,
            hidden_dim=288,
            ffn_dim=624,
            inference_mode="pc_energy",
            learned_feedback_proposal_scale=1.0,
            route_lr_scale=0.0,
        )
    )
    d256_ffn1024 = MessagePassingPCLanguageModel(
        MessagePassingPCConfig(
            vocab_size=4096,
            bos_token_id=4095,
            seq_len=1024,
            hidden_dim=256,
            ffn_dim=1024,
            inference_mode="pc_energy",
            learned_feedback_proposal_scale=1.0,
            route_lr_scale=0.0,
        )
    )
    assert d320_ffn208.learned_parameter_count() == 7_337_216
    assert d288_ffn624.learned_parameter_count() == 7_304_320
    assert d256_ffn1024.learned_parameter_count() == 7_083_008
    assert d320_ffn208.learned_parameter_count() < 7_340_162
    assert d288_ffn624.learned_parameter_count() < 7_340_162
    assert d256_ffn1024.learned_parameter_count() < 7_340_162


def test_pc_modal_runner_builds_message_pc_model() -> None:
    args = parse_args(
        [
            "--model",
            "pc-message",
            "--seq-len",
            "8",
            "--message-pc-hidden-dim",
            "32",
            "--message-pc-layers",
            "2",
            "--message-pc-heads",
            "4",
            "--message-pc-route-update-mode",
            "token_error_teacher",
            "--message-pc-decoder-supervision-mode",
            "top_only",
            "--message-pc-decoder-state-mode",
            "rms_norm",
            "--message-pc-readout-layer",
            "1",
            "--message-pc-readout-objective-share",
            "0.7",
            "--message-pc-winner-margin",
            "2.5",
            "--message-pc-winner-margin-scale",
            "0.75",
            "--message-pc-block-state-mode",
            "identity_residual",
            "--message-pc-residual-scale",
            "0.4",
            "--message-pc-self-init-gain",
            "0.25",
            "--message-pc-message-init-gain",
            "1.0",
            "--message-pc-ffn-dim",
            "48",
            "--message-pc-ffn-init-gain",
            "0.3",
            "--message-pc-ffn-lr-scale",
            "0.2",
            "--message-pc-ffn-learning-source",
            "free_to_settled",
            "--message-pc-token-feedback-mode",
            "top_propagated",
            "--message-pc-inference-mode",
            "pc_energy",
            "--message-pc-inference-optimizer",
            "heavy_ball",
            "--message-pc-inference-momentum",
            "0.8",
            "--message-pc-learning-phase-source",
            "free",
            "--message-pc-decoder-learning-source",
            "free",
            "--message-pc-learned-feedback-proposal-scale",
            "0.75",
            "--message-pc-residual-error-suffix-decay",
            "0.5",
            "--message-pc-residual-error-target-layer",
            "base",
            "--message-pc-residual-error-proxy-memory-scale",
            "0.35",
            "--message-pc-residual-error-proxy-memory-decay",
            "0.8",
            "--message-pc-residual-error-upper-proxy-state-scale",
            "0.45",
            "--message-pc-residual-error-upper-proxy-state-decay",
            "0.7",
            "--message-pc-residual-error-upper-content-state-scale",
            "0.55",
            "--message-pc-residual-error-upper-content-state-decay",
            "0.6",
            "--message-pc-residual-error-upper-content-write-period",
            "3",
            "--message-pc-residual-error-upper-content-write-offset",
            "0",
            "--message-pc-residual-error-upper-content-read-period",
            "3",
            "--message-pc-residual-error-upper-content-read-offset",
            "2",
            "--message-pc-residual-error-upper-content-read-delay",
            "3",
            "--message-pc-hierarchical-error-top-down-scale",
            "0.6",
            "--message-pc-hierarchical-error-base-depth",
            "1",
            "--message-pc-hierarchical-error-binding-mode",
            "identity",
            "--message-pc-hierarchical-error-target-mode",
            "mistakes",
            "--message-pc-hierarchical-error-context-credit-scale",
            "0.4",
            "--message-pc-inference-lr-final",
            "0.012",
            "--message-pc-inference-lr-schedule-start-step",
            "100",
            "--message-pc-inference-lr-schedule-end-step",
            "400",
            "--message-pc-energy-backtracking-steps",
            "6",
            "--message-pc-matrix-optimizer",
            "muon",
            "--message-pc-optimizer-warmup-steps",
            "40",
            "--message-pc-muon-momentum",
            "0.9",
            "--message-pc-muon-momentum-final",
            "0.97",
            "--message-pc-muon-momentum-warmup-steps",
            "400",
            "--message-pc-muon-lr-scale",
            "4.0",
            "--message-pc-muon-beta2",
            "0.8",
            "--message-pc-muon-ns-steps",
            "4",
        ]
    )

    model, meta = build_model(args, torch.device("cpu"), vocab_size=32, bos_token_id=31)

    assert isinstance(model, MessagePassingPCLanguageModel)
    assert meta["model"] == "message_passing_pc_lm"
    assert meta["hidden_dim"] == 32
    assert meta["route_update_mode"] == "token_error_teacher"
    assert meta["decoder_supervision_mode"] == "top_only"
    assert meta["decoder_state_mode"] == "rms_norm"
    assert meta["readout_layer"] == 1
    assert meta["readout_objective_share"] == pytest.approx(0.7)
    assert meta["winner_margin"] == pytest.approx(2.5)
    assert meta["winner_margin_scale"] == pytest.approx(0.75)
    assert meta["block_state_mode"] == "identity_residual"
    assert meta["residual_scale"] == pytest.approx(0.4)
    assert meta["self_init_gain"] == pytest.approx(0.25)
    assert meta["message_init_gain"] == pytest.approx(1.0)
    assert meta["ffn_dim"] == 48
    assert meta["ffn_init_gain"] == pytest.approx(0.3)
    assert meta["ffn_lr_scale"] == pytest.approx(0.2)
    assert meta["ffn_learning_source"] == "free_to_settled"
    assert meta["token_feedback_mode"] == "top_propagated"
    assert meta["inference_mode"] == "pc_energy"
    assert meta["inference_optimizer"] == "heavy_ball"
    assert meta["inference_momentum"] == pytest.approx(0.8)
    assert meta["learning_phase_source"] == "free"
    assert meta["free_phase_amortization_scale"] == 0.0
    assert meta["decoder_learning_source"] == "free"
    assert meta["learned_feedback_proposal_scale"] == pytest.approx(0.75)
    assert meta["residual_error_suffix_decay"] == pytest.approx(0.5)
    assert meta["residual_error_target_layer"] == "base"
    assert meta["residual_error_proxy_memory_scale"] == pytest.approx(0.35)
    assert meta["residual_error_proxy_memory_decay"] == pytest.approx(0.8)
    assert meta["residual_error_upper_proxy_state_scale"] == pytest.approx(0.45)
    assert meta["residual_error_upper_proxy_state_decay"] == pytest.approx(0.7)
    assert meta["residual_error_upper_content_state_scale"] == pytest.approx(0.55)
    assert meta["residual_error_upper_content_state_decay"] == pytest.approx(0.6)
    assert meta["residual_error_upper_content_write_period"] == 3
    assert meta["residual_error_upper_content_write_offset"] == 0
    assert meta["residual_error_upper_content_read_period"] == 3
    assert meta["residual_error_upper_content_read_offset"] == 2
    assert meta["residual_error_upper_content_read_delay"] == 3
    assert meta["hierarchical_error_top_down_scale"] == pytest.approx(0.6)
    assert meta["hierarchical_error_base_depth"] == 1
    assert meta["hierarchical_error_binding_mode"] == "identity"
    assert meta["hierarchical_error_target_mode"] == "mistakes"
    assert meta["hierarchical_error_context_credit_scale"] == pytest.approx(0.4)
    assert meta["inference_lr_final"] == pytest.approx(0.012)
    assert meta["inference_lr_schedule_start_step"] == 100
    assert meta["inference_lr_schedule_end_step"] == 400
    assert meta["energy_backtracking_steps"] == 6
    assert meta["matrix_optimizer"] == "muon"
    assert meta["optimizer_warmup_steps"] == 40
    assert meta["muon_lr_scale"] == pytest.approx(4.0)
    assert meta["muon_momentum"] == pytest.approx(0.9)
    assert meta["muon_momentum_final"] == pytest.approx(0.97)
    assert meta["muon_momentum_warmup_steps"] == 400
    assert meta["muon_beta2"] == pytest.approx(0.8)
    assert meta["muon_ns_steps"] == 4


def test_pc_modal_runner_builds_recurrent_local_message_pc() -> None:
    args = parse_args(
        [
            "--model",
            "pc-message",
            "--seq-len",
            "8",
            "--message-pc-hidden-dim",
            "32",
            "--message-pc-layers",
            "1",
            "--message-pc-heads",
            "4",
            "--message-pc-inference-mode",
            "recurrent_local",
            "--message-pc-recurrent-shared-steps",
            "3",
            "--message-pc-recurrent-step-scale",
            "0.25",
            "--message-pc-inference-lr",
            "1.0",
            "--message-pc-decoder-supervision-mode",
            "top_only",
            "--message-pc-decoder-state-mode",
            "rms_norm",
            "--message-pc-block-state-mode",
            "nanochat_residual",
            "--message-pc-ffn-dim",
            "48",
            "--message-pc-route-update-mode",
            "local_energy",
        ]
    )

    model, meta = build_model(
        args,
        torch.device("cpu"),
        vocab_size=32,
        bos_token_id=31,
    )

    assert isinstance(model, MessagePassingPCLanguageModel)
    assert meta["inference_mode"] == "recurrent_local"
    assert meta["recurrent_shared_steps"] == 3
    assert meta["recurrent_step_scale"] == pytest.approx(0.25)
    assert model._readout_depth() == 3


def test_pc_modal_runner_builds_fast_slow_bidirectional_deployment() -> None:
    args = parse_args(
        [
            "--model",
            "pc-message",
            "--seq-len",
            "8",
            "--message-pc-hidden-dim",
            "32",
            "--message-pc-layers",
            "2",
            "--message-pc-heads",
            "4",
            "--message-pc-inference-mode",
            "bidirectional_pc",
            "--message-pc-bidirectional-free-settling",
            "--message-pc-bidirectional-deployment-settling",
            "--message-pc-fast-attention-window",
            "2",
            "--message-pc-fast-attention-layers",
            "1",
            "--message-pc-decoder-supervision-mode",
            "top_only",
            "--message-pc-block-state-mode",
            "identity_residual",
        ]
    )

    model, meta = build_model(
        args,
        torch.device("cpu"),
        vocab_size=32,
        bos_token_id=31,
    )

    assert isinstance(model, MessagePassingPCLanguageModel)
    assert model.config.bidirectional_free_settling
    assert model.config.bidirectional_deployment_settling
    assert meta["fast_attention_window"] == 2
    assert meta["fast_attention_layers"] == 1


def test_message_pc_recurrent_local_matched_budget_stays_below_gpt() -> None:
    model = MessagePassingPCLanguageModel(
        MessagePassingPCConfig(
            vocab_size=4096,
            bos_token_id=4095,
            seq_len=1024,
            hidden_dim=512,
            num_layers=1,
            num_heads=8,
            inference_mode="recurrent_local",
            recurrent_shared_steps=4,
            decoder_supervision_mode="top_only",
            decoder_state_mode="rms_norm",
            block_state_mode="nanochat_residual",
            ffn_dim=2000,
            route_update_mode="local_energy",
        )
    )

    assert model.active_parameter_count() < 7_340_162


def test_message_pc_layerwise_local_matched_budget_stays_below_gpt() -> None:
    model = MessagePassingPCLanguageModel(
        MessagePassingPCConfig(
            vocab_size=4096,
            bos_token_id=4095,
            seq_len=1024,
            hidden_dim=312,
            num_layers=4,
            num_heads=6,
            inference_mode="layerwise_local",
            decoder_supervision_mode="top_only",
            decoder_state_mode="rms_norm",
            block_state_mode="nanochat_residual",
            ffn_dim=1248,
            route_update_mode="local_energy",
        )
    )

    assert model.active_parameter_count() == 7_232_512
    assert model.active_parameter_count() < 7_340_162


def test_pc_modal_runner_builds_her_error_hierarchy() -> None:
    args = parse_args(
        [
            "--model",
            "pc-message",
            "--seq-len",
            "8",
            "--message-pc-hidden-dim",
            "32",
            "--message-pc-layers",
            "2",
            "--message-pc-heads",
            "4",
            "--message-pc-inference-mode",
            "her_error_hierarchy",
            "--message-pc-hierarchical-error-binding-mode",
            "factorized",
            "--message-pc-hierarchical-error-target-mode",
            "fixed_outcome",
            "--message-pc-hierarchical-error-outcome-code-scale",
            "0.2",
            "--message-pc-hierarchical-error-outcome-readout-scale",
            "3.0",
            "--message-pc-hierarchical-error-context-mode",
            "persistent_hard",
            "--message-pc-hierarchical-error-gate-lr-scale",
            "2.5",
            "--message-pc-hierarchical-error-initial-write-rate",
            "0.9",
            "--message-pc-hierarchical-error-memory-scale",
            "0.2",
            "--message-pc-hierarchical-error-token-credit-scale",
            "0.3",
            "--message-pc-state-prediction-separation",
            "--message-pc-prediction-stream-window",
            "16",
            "--message-pc-sps-prediction-input-scale",
            "0.75",
            "--message-pc-sps-future-state-credit-scale",
            "1.25",
            "--message-pc-sps-layer-token-credit-scale",
            "0.5",
            "--message-pc-sps-future-observation-credit-scale",
            "0.6",
        ]
    )

    model, meta = build_model(
        args,
        torch.device("cpu"),
        vocab_size=32,
        bos_token_id=31,
    )

    assert isinstance(model, MessagePassingPCLanguageModel)
    assert meta["inference_mode"] == "her_error_hierarchy"
    assert meta["hierarchical_error_binding_mode"] == "factorized"
    assert meta["hierarchical_error_target_mode"] == "fixed_outcome"
    assert meta["hierarchical_error_outcome_code_scale"] == pytest.approx(0.2)
    assert meta["hierarchical_error_outcome_readout_scale"] == pytest.approx(3.0)
    assert meta["hierarchical_error_context_mode"] == "persistent_hard"
    assert meta["hierarchical_error_gate_lr_scale"] == pytest.approx(2.5)
    assert meta["hierarchical_error_initial_write_rate"] == pytest.approx(0.9)
    assert meta["hierarchical_error_memory_scale"] == pytest.approx(0.2)
    assert meta["hierarchical_error_token_credit_scale"] == pytest.approx(0.3)
    assert meta["state_prediction_separation"] is True
    assert meta["prediction_stream_window"] == 16
    assert meta["sps_prediction_input_scale"] == pytest.approx(0.75)
    assert meta["sps_layer_token_credit_scale"] == pytest.approx(0.5)
    assert meta["sps_future_state_credit_scale"] == pytest.approx(1.25)
    assert meta["sps_future_observation_credit_scale"] == pytest.approx(0.6)
    assert model.prediction_token is not None
    assert len(model.hierarchical_gate_weights) == 2
    assert all(
        torch.count_nonzero(weight) == 0
        for weight in model.generative_weights
    )


def test_pc_modal_runner_wires_recursive_cp_her() -> None:
    args = parse_args(
        [
            "--model",
            "pc-message",
            "--seq-len",
            "8",
            "--message-pc-hidden-dim",
            "32",
            "--message-pc-layers",
            "3",
            "--message-pc-heads",
            "4",
            "--message-pc-inference-mode",
            "her_error_hierarchy",
            "--message-pc-hierarchical-error-binding-mode",
            "factorized",
            "--message-pc-hierarchical-error-prediction-mode",
            "recursive_cp",
            "--message-pc-hierarchical-error-target-mode",
            "fixed_outcome",
            "--message-pc-hierarchical-error-recursive-cp-rank",
            "8",
            "--message-pc-hierarchical-error-recursive-cp-upper-lr-scale",
            "2",
            "--message-pc-hierarchical-error-recursive-cp-component-norm-cap",
            "0.1",
            "--message-pc-hierarchical-error-recursive-cp-delayed-consolidation",
            "--message-pc-hierarchical-error-recursive-cp-consolidation-rate",
            "0.2",
            "--message-pc-hierarchical-error-recursive-cp-min-improvement",
            "0.0001",
            "--message-pc-hierarchical-error-context-mode",
            "persistent_hard",
        ]
    )

    model, meta = build_model(
        args,
        torch.device("cpu"),
        vocab_size=32,
        bos_token_id=31,
    )

    assert isinstance(model, MessagePassingPCLanguageModel)
    assert model.recursive_cp_her is not None
    assert model.recursive_cp_her.rank == 8
    assert meta["hierarchical_error_prediction_mode"] == "recursive_cp"
    assert meta["hierarchical_error_target_mode"] == "fixed_outcome"
    assert meta["hierarchical_error_recursive_cp_rank"] == 8
    assert meta["hierarchical_error_recursive_cp_upper_lr_scale"] == 2.0
    assert meta[
        "hierarchical_error_recursive_cp_component_norm_cap"
    ] == pytest.approx(0.1)
    assert model.recursive_cp_her_fast is not None
    assert meta["hierarchical_error_recursive_cp_delayed_consolidation"] is True
    assert meta[
        "hierarchical_error_recursive_cp_consolidation_rate"
    ] == pytest.approx(0.2)
    assert meta[
        "hierarchical_error_recursive_cp_min_improvement"
    ] == pytest.approx(0.0001)


def test_pc_modal_runner_wires_sps_node_residual_hierarchy() -> None:
    args = parse_args(
        [
            "--model",
            "pc-message",
            "--seq-len",
            "8",
            "--message-pc-hidden-dim",
            "32",
            "--message-pc-layers",
            "3",
            "--message-pc-heads",
            "4",
            "--message-pc-inference-mode",
            "her_error_hierarchy",
            "--message-pc-hierarchical-error-binding-mode",
            "factorized",
            "--message-pc-hierarchical-error-base-depth",
            "1",
            "--message-pc-hierarchical-error-context-mode",
            "persistent",
            "--message-pc-state-prediction-separation",
            "--message-pc-sps-node-residual-hierarchy",
        ]
    )

    model, meta = build_model(
        args,
        torch.device("cpu"),
        vocab_size=32,
        bos_token_id=31,
    )

    assert isinstance(model, MessagePassingPCLanguageModel)
    assert meta["sps_node_residual_hierarchy"] is True
    side_parameter_count = sum(
        value.numel()
        for value in (*model.generative_weights, *model.feedback_weights)
    )
    assert (
        model.learned_parameter_count() - model.active_parameter_count()
        >= side_parameter_count
    )


def test_pc_modal_runner_wires_her_outer_product_contract() -> None:
    args = parse_args(
        [
            "--model",
            "pc-message",
            "--seq-len",
            "8",
            "--message-pc-hidden-dim",
            "32",
            "--message-pc-layers",
            "2",
            "--message-pc-heads",
            "4",
            "--message-pc-inference-mode",
            "her_error_hierarchy",
            "--message-pc-hierarchical-error-binding-mode",
            "factorized",
            "--message-pc-hierarchical-error-base-depth",
            "1",
            "--message-pc-hierarchical-error-outer-product-contract",
        ]
    )

    model, meta = build_model(
        args,
        torch.device("cpu"),
        vocab_size=32,
        bos_token_id=31,
    )

    assert isinstance(model, MessagePassingPCLanguageModel)
    assert meta["hierarchical_error_outer_product_contract"] is True
    assert torch.equal(model.feedback_weights[1], torch.eye(32))


def test_pc_modal_runner_wires_free_phase_amortization_scale() -> None:
    args = parse_args(
        [
            "--model",
            "pc-message",
            "--seq-len",
            "8",
            "--message-pc-hidden-dim",
            "32",
            "--message-pc-layers",
            "2",
            "--message-pc-heads",
            "4",
            "--message-pc-learning-phase-source",
            "settled",
            "--message-pc-free-phase-amortization-scale",
            "0.125",
        ]
    )

    model, meta = build_model(
        args,
        torch.device("cpu"),
        vocab_size=32,
        bos_token_id=31,
    )

    assert isinstance(model, MessagePassingPCLanguageModel)
    assert model.config.free_phase_amortization_scale == pytest.approx(0.125)
    assert meta["free_phase_amortization_scale"] == pytest.approx(0.125)


def test_pc_modal_runner_wires_spiking_precision_and_forward_update() -> None:
    args = parse_args(
        [
            "--model",
            "pc-message",
            "--seq-len",
            "8",
            "--message-pc-hidden-dim",
            "32",
            "--message-pc-layers",
            "2",
            "--message-pc-heads",
            "4",
            "--message-pc-block-state-mode",
            "identity_residual",
            "--message-pc-block-parameterization",
            "depth_mup",
            "--message-pc-spiking-precision-scale",
            "0.25",
            "--message-pc-forward-update-top-layers",
            "1",
        ]
    )

    model, meta = build_model(
        args,
        torch.device("cpu"),
        vocab_size=32,
        bos_token_id=31,
    )

    assert isinstance(model, MessagePassingPCLanguageModel)
    assert model.config.spiking_precision_scale == pytest.approx(0.25)
    assert meta["spiking_precision_scale"] == pytest.approx(0.25)
    assert model.config.block_parameterization == "depth_mup"
    assert meta["block_parameterization"] == "depth_mup"
    assert model.config.forward_update_top_layers == 1
    assert meta["forward_update_top_layers"] == 1


def test_pc_modal_runner_builds_meta_pc() -> None:
    args = parse_args(
        [
            "--model",
            "pc-message",
            "--seq-len",
            "8",
            "--message-pc-hidden-dim",
            "32",
            "--message-pc-layers",
            "2",
            "--message-pc-heads",
            "4",
            "--message-pc-inference-mode",
            "meta_pc",
            "--message-pc-decoder-supervision-mode",
            "top_only",
            "--message-pc-forward-update-top-layers",
            "2",
            "--message-pc-inference-steps",
            "20",
            "--message-pc-inference-lr",
            "0.2",
            "--message-pc-meta-feedback-mode",
            "reciprocal_attention",
            "--message-pc-meta-reciprocal-scale",
            "0.3",
            "--message-pc-meta-reciprocal-learning-rate",
            "0.02",
            "--message-pc-meta-reciprocal-noise-std",
            "0.04",
            "--message-pc-meta-variance-normalize-weights",
            "--message-pc-meta-relinearize-inference",
            "--message-pc-meta-output-feedback-mode",
            "kp_factorized",
            "--message-pc-meta-output-feedback-lr-scale",
            "0.75",
            "--message-pc-meta-output-feedback-rank",
            "8",
            "--message-pc-block-state-mode",
            "nanochat_residual",
            "--message-pc-token-value-embeddings",
            "--message-pc-token-value-scale",
            "1.25",
            "--message-pc-token-value-lr-scale",
            "0.4",
            "--message-pc-causal-encoder-decoder",
            "--message-pc-causal-encoder-depth",
            "1",
            "--message-pc-cross-attention-lr-scale",
            "1.3",
            "--message-pc-cross-attention-gate-lr-scale",
            "0.7",
            "--message-pc-cross-attention-output-init-gain",
            "0.1",
            "--message-pc-sensory-reinjection-start",
            "0.2",
            "--message-pc-sensory-reinjection-end",
            "0.05",
            "--message-pc-sensory-reinjection-mode",
            "pre_block",
            "--message-pc-residual-stream-scale-start",
            "1.15",
            "--message-pc-residual-stream-scale-end",
            "1.05",
            "--message-pc-learn-residual-stream-scales",
            "--message-pc-residual-stream-scale-mode",
            "channel",
            "--message-pc-residual-stream-lr-scale",
            "0.6",
            "--message-pc-residual-stream-lr-scale-final",
            "1.8",
            "--message-pc-residual-stream-lr-schedule-start-step",
            "100",
            "--message-pc-residual-stream-lr-schedule-end-step",
            "500",
        ]
    )

    model, meta = build_model(
        args,
        torch.device("cpu"),
        vocab_size=32,
        bos_token_id=31,
    )

    assert isinstance(model, MessagePassingPCLanguageModel)
    assert model.config.inference_mode == "meta_pc"
    assert model.config.forward_update_top_layers == 2
    assert model.config.inference_steps == 20
    assert model.config.meta_pc_feedback_mode == "reciprocal_attention"
    assert model.config.meta_pc_reciprocal_scale == pytest.approx(0.3)
    assert model.config.meta_pc_reciprocal_learning_rate == pytest.approx(0.02)
    assert model.config.meta_pc_reciprocal_noise_std == pytest.approx(0.04)
    assert model.config.meta_pc_variance_normalize_weights is True
    assert model.config.meta_pc_relinearize_inference is True
    assert model.config.meta_pc_output_feedback_mode == "kp_factorized"
    assert model.config.meta_pc_output_feedback_lr_scale == pytest.approx(0.75)
    assert model.config.meta_pc_output_feedback_rank == 8
    assert model.config.token_value_embeddings is True
    assert model.config.token_value_scale == pytest.approx(1.25)
    assert model.config.token_value_lr_scale == pytest.approx(0.4)
    assert model.config.causal_encoder_decoder is True
    assert model.config.causal_encoder_depth == 1
    assert model.config.cross_attention_lr_scale == pytest.approx(1.3)
    assert model.config.cross_attention_gate_lr_scale == pytest.approx(0.7)
    assert model.config.cross_attention_output_init_gain == pytest.approx(0.1)
    assert model.config.sensory_reinjection_start == pytest.approx(0.2)
    assert model.config.sensory_reinjection_end == pytest.approx(0.05)
    assert model.config.sensory_reinjection_mode == "pre_block"
    assert model.config.residual_stream_scale_start == pytest.approx(1.15)
    assert model.config.residual_stream_scale_end == pytest.approx(1.05)
    assert model.config.learn_residual_stream_scales is True
    assert model.config.residual_stream_scale_mode == "channel"
    assert model.config.residual_stream_lr_scale == pytest.approx(0.6)
    assert model.config.residual_stream_lr_scale_final == pytest.approx(1.8)
    assert model.config.residual_stream_lr_schedule_start_step == 100
    assert model.config.residual_stream_lr_schedule_end_step == 500
    assert meta["inference_mode"] == "meta_pc"


def test_pc_modal_runner_wires_pre_settle_reciprocal_update() -> None:
    args = parse_args(
        [
            "--model",
            "pc-message",
            "--seq-len",
            "8",
            "--message-pc-hidden-dim",
            "32",
            "--message-pc-layers",
            "2",
            "--message-pc-heads",
            "4",
            "--message-pc-inference-mode",
            "meta_pc",
            "--message-pc-decoder-supervision-mode",
            "top_only",
            "--message-pc-forward-update-top-layers",
            "2",
            "--message-pc-meta-feedback-mode",
            "learned_reciprocal",
            "--message-pc-meta-reciprocal-learning-rate",
            "0.2",
            "--message-pc-meta-reciprocal-update-before-settle",
            "--message-pc-meta-state-gated-reciprocal",
        ]
    )

    model, _ = build_model(
        args,
        torch.device("cpu"),
        vocab_size=32,
        bos_token_id=31,
    )

    assert isinstance(model, MessagePassingPCLanguageModel)
    assert model.config.meta_pc_reciprocal_update_before_settle is True
    assert model.config.meta_pc_state_gated_reciprocal is True
    assert model.connection_stats()[
        "message_pc_meta_uses_pre_settle_reciprocal_update"
    ] == 1.0


def test_pc_modal_runner_wires_stage_reciprocal_feedback() -> None:
    args = parse_args(
        [
            "--model",
            "pc-message",
            "--seq-len",
            "8",
            "--message-pc-hidden-dim",
            "32",
            "--message-pc-layers",
            "2",
            "--message-pc-heads",
            "4",
            "--message-pc-ffn-dim",
            "48",
            "--message-pc-block-state-mode",
            "nanochat_residual",
            "--message-pc-inference-mode",
            "meta_pc",
            "--message-pc-decoder-supervision-mode",
            "top_only",
            "--message-pc-forward-update-top-layers",
            "2",
            "--message-pc-meta-feedback-mode",
            "stage_reciprocal",
            "--message-pc-meta-reciprocal-learning-rate",
            "0.1",
        ]
    )

    model, _ = build_model(
        args,
        torch.device("cpu"),
        vocab_size=32,
        bos_token_id=31,
    )

    assert isinstance(model, MessagePassingPCLanguageModel)
    assert model.config.meta_pc_feedback_mode == "stage_reciprocal"
    assert model.connection_stats()[
        "message_pc_meta_uses_stage_reciprocal"
    ] == 1.0


def test_pc_modal_runner_wires_cross_memory_skip_credit() -> None:
    args = parse_args(
        [
            "--model",
            "pc-message",
            "--seq-len",
            "8",
            "--message-pc-hidden-dim",
            "32",
            "--message-pc-layers",
            "3",
            "--message-pc-heads",
            "4",
            "--message-pc-inference-mode",
            "meta_pc",
            "--message-pc-decoder-supervision-mode",
            "top_only",
            "--message-pc-forward-update-top-layers",
            "3",
            "--message-pc-block-state-mode",
            "nanochat_residual",
            "--message-pc-causal-encoder-decoder",
            "--message-pc-causal-encoder-depth",
            "1",
            "--message-pc-meta-cross-memory-skip-credit",
            "--message-pc-meta-cross-memory-skip-credit-scale",
            "32",
        ]
    )

    model, _ = build_model(
        args,
        torch.device("cpu"),
        vocab_size=32,
        bos_token_id=31,
    )

    assert isinstance(model, MessagePassingPCLanguageModel)
    assert model.config.meta_pc_cross_memory_skip_credit is True
    assert model.config.meta_pc_cross_memory_skip_credit_scale == 32.0


def test_pc_modal_runner_wires_shared_output_feedback_values() -> None:
    args = parse_args(
        [
            "--model",
            "pc-message",
            "--seq-len",
            "8",
            "--message-pc-hidden-dim",
            "32",
            "--message-pc-layers",
            "2",
            "--message-pc-heads",
            "4",
            "--message-pc-inference-mode",
            "meta_pc",
            "--message-pc-decoder-supervision-mode",
            "top_only",
            "--message-pc-forward-update-top-layers",
            "2",
            "--message-pc-block-state-mode",
            "nanochat_residual",
            "--message-pc-meta-output-feedback-mode",
            "kp_learned",
            "--message-pc-meta-output-feedback-values",
        ]
    )

    model, _ = build_model(
        args,
        torch.device("cpu"),
        vocab_size=32,
        bos_token_id=31,
    )

    assert isinstance(model, MessagePassingPCLanguageModel)
    assert model.config.meta_pc_output_feedback_values is True


def test_pc_modal_runner_wires_free_local_token_credit_scale() -> None:
    args = parse_args(
        [
            "--model",
            "pc-message",
            "--seq-len",
            "8",
            "--message-pc-hidden-dim",
            "32",
            "--message-pc-layers",
            "2",
            "--message-pc-heads",
            "4",
            "--message-pc-free-local-token-credit-scale",
            "0.5",
        ]
    )

    model, meta = build_model(
        args,
        torch.device("cpu"),
        vocab_size=32,
        bos_token_id=31,
    )

    assert isinstance(model, MessagePassingPCLanguageModel)
    assert model.config.free_local_token_credit_scale == pytest.approx(0.5)
    assert meta["free_local_token_credit_scale"] == pytest.approx(0.5)


def test_pc_modal_runner_wires_free_local_token_embedding_credit_scale() -> None:
    args = parse_args(
        [
            "--model",
            "pc-message",
            "--seq-len",
            "8",
            "--message-pc-hidden-dim",
            "32",
            "--message-pc-layers",
            "2",
            "--message-pc-heads",
            "4",
            "--message-pc-free-local-token-embedding-credit-scale",
            "0.25",
        ]
    )

    model, meta = build_model(
        args,
        torch.device("cpu"),
        vocab_size=32,
        bos_token_id=31,
    )

    assert isinstance(model, MessagePassingPCLanguageModel)
    assert model.config.free_local_token_embedding_credit_scale == pytest.approx(
        0.25
    )
    assert meta["free_local_token_embedding_credit_scale"] == pytest.approx(0.25)
