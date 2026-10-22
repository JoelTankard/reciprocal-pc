from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from pfc import (
    AdditiveRLSBranch,
    Golden,
    ReciprocalPCTrainer,
    create_golden,
    create_reciprocal_pc,
    golden_local_update,
    load_golden_checkpoint,
    load_reciprocal_pc_checkpoint,
    model_config,
    shared_prefill,
)
from pfc.golden import INFERENCE_SCHEMA
from pfc.rls_memory import initial_state, read, write
from pfc.rls_sequence import sequence_reads


@pytest.fixture(scope="module")
def golden():
    model = create_golden()
    with torch.no_grad():
        model.gain.fill_(0.057484474033117294)
    return model.eval()


def inference_payload(model):
    return {
        "schema": INFERENCE_SCHEMA,
        "model_config": asdict(model_config()),
        "model": model.base_model.state_dict(),
        "branch": model.branch_state_dict(),
        "transformation": {"inference_only": True, "alpha": 0.2, "later_step": 50000, "anchor_step": 49000},
    }


def assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            assert_tree_equal(actual[name], expected[name])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_tree_equal(left, right)
    else:
        assert actual == expected


def test_golden_factory_and_loader_never_construct_trainer(golden, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("inference created a trainer")

    monkeypatch.setattr(ReciprocalPCTrainer, "__init__", forbidden)
    constructed = create_golden()
    assert isinstance(constructed, Golden)
    assert sum(parameter.numel() for parameter in constructed.parameters()) == 23_930_435
    del constructed
    path = tmp_path / "golden.pt"
    torch.save(inference_payload(golden), path)
    restored, payload = load_golden_checkpoint(path)
    tokens = torch.tensor([[32759, 11, 23, 47, 11, 23]])
    assert payload["schema"] == INFERENCE_SCHEMA
    assert not restored.training
    assert restored._golden_inference_only
    assert torch.equal(restored(tokens)["logits"], golden(tokens)["logits"])
    assert_tree_equal(restored.base_model.state_dict(), payload["model"])
    assert_tree_equal(restored.branch_state_dict(), payload["branch"])


@pytest.mark.parametrize("mutation", ["config", "schema", "extra", "projection", "gain", "nonfinite"])
def test_loader_rejects_invalid_export(golden, tmp_path, mutation):
    payload = inference_payload(golden)
    if mutation == "config":
        payload["model_config"]["width"] = 256
    elif mutation == "schema":
        payload["schema"] = "rpc-additive-rls-full-state.v1"
    elif mutation == "extra":
        payload["trainer"] = {}
    elif mutation == "projection":
        payload["branch"]["projection"] = torch.zeros(32, 512)
    elif mutation == "gain":
        payload["branch"]["gain"] = torch.zeros(1)
    else:
        payload["branch"]["gain"].fill_(float("nan"))
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_golden_checkpoint(path)


def test_inference_checkpoint_cannot_be_restored_as_training(golden, tmp_path):
    path = tmp_path / "inference.pt"
    torch.save(inference_payload(golden), path)
    with pytest.raises(ValueError, match="inference-only.*cannot resume"):
        load_reciprocal_pc_checkpoint(path)


def test_rls_write_matches_independent_ridge_solution_without_mutating_state():
    keys = torch.tensor([[1.0, 0.0], [0.5, 1.0], [-0.25, 0.75]])
    values = torch.tensor([[2.0, -1.0, 0.5], [0.0, 3.0, 1.0], [1.0, 0.5, -2.0]])
    state = initial_state(1, 2, 3)
    initial_s, initial_p = state.S.clone(), state.P.clone()
    untouched = state
    for key, value in zip(keys, values):
        state = write(state, key[None], value[None])
    inverse = torch.linalg.inv(0.25 * torch.eye(2) + keys.T @ keys)
    expected = values.T @ keys @ inverse
    torch.testing.assert_close(state.S[0], expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state.P[0], inverse, rtol=1e-5, atol=1e-6)
    query = torch.tensor([[0.4, -0.3]])
    torch.testing.assert_close(read(state, query), query @ expected.T)
    assert torch.equal(untouched.S, initial_s)
    assert torch.equal(untouched.P, initial_p)


def test_sequence_writes_observed_value_for_previous_key_not_future_target():
    keys = torch.zeros(1, 4, 32)
    keys[0, :, 0] = 1
    values = torch.arange(4 * 512, dtype=torch.float32).reshape(1, 4, 512) / 512
    reads, state, last_key = sequence_reads(keys, values)
    assert torch.count_nonzero(reads[:, 0]) == 0
    # With the same unit key, the first completed write is v1 / (ridge + 1).
    torch.testing.assert_close(reads[:, 1], values[:, 1] / 1.25)
    torch.testing.assert_close(reads[:, 2], (values[:, 1] + values[:, 2]) / 2.25)
    changed = values.clone()
    changed[:, 3] += 100
    changed_reads, _, _ = sequence_reads(keys, changed)
    assert torch.equal(changed_reads[:, :3], reads[:, :3])
    first, prefix_state, previous = sequence_reads(keys[:, :2], values[:, :2])
    saved_s, saved_p, saved_key = prefix_state.S.clone(), prefix_state.P.clone(), previous.clone()
    rest, final, final_key = sequence_reads(keys[:, 2:], values[:, 2:], state=prefix_state, previous_key=previous, has_previous=True)
    assert torch.equal(torch.cat((first, rest), dim=1), reads)
    assert torch.equal(final.S, state.S) and torch.equal(final.P, state.P)
    assert torch.equal(final_key, last_key)
    assert torch.equal(prefix_state.S, saved_s) and torch.equal(prefix_state.P, saved_p)
    assert torch.equal(previous, saved_key)


def test_golden_full_sequence_chunk_and_step_are_causal(golden):
    tokens = torch.tensor([[32759, 11, 23, 47, 11, 23, 47, 91]])
    whole = golden(tokens)
    prefix = golden(tokens[:, :4])
    saved_s = prefix["state"].rls_state.S.clone()
    tail = golden(tokens[:, 4:], prefix["state"])
    chunked = torch.cat((prefix["logits"], tail["logits"]), dim=1)
    torch.testing.assert_close(chunked, whole["logits"], rtol=3e-4, atol=5e-5)
    assert torch.equal(prefix["state"].rls_state.S, saved_s)
    current, logits = None, []
    for token in tokens.unbind(1):
        output = golden.step(token, current)
        current = output["state"]
        logits.append(output["logits"])
    torch.testing.assert_close(torch.stack(logits, 1), whole["logits"], rtol=3e-4, atol=5e-5)
    changed = tokens.clone()
    changed[:, 4:] = torch.tensor([5, 6, 7, 8])
    torch.testing.assert_close(golden(changed)["logits"][:, :4], whole["logits"][:, :4], rtol=3e-4, atol=5e-5)
    assert torch.count_nonzero(whole["branch_logits"][:, 0]) == 0
    assert torch.count_nonzero(whole["branch_logits"][:, 1:]) > 0
    assert golden.state_bytes(current) == 69_760
    assert torch.equal(golden(tokens)["logits"], whole["logits"])
    assert all(parameter.grad is None for parameter in golden.base_model.parameters())


def test_gain_credit_matches_actual_candidate_cross_entropy_gradient(golden):
    tokens = torch.tensor([[32759, 11, 23, 47]])
    targets = torch.tensor([[11, 23, -100, 91]])
    output = golden(tokens)
    objective = golden.assign_gain_gradient(output, targets)
    gain = golden.gain.detach().clone().requires_grad_(True)
    expected_loss = F.cross_entropy((output["native_logits"] + gain * output["branch_logits"]).reshape(-1, 32768), targets.flatten())
    expected_loss.backward()
    torch.testing.assert_close(objective["loss"], expected_loss.detach())
    torch.testing.assert_close(objective["gradient"], gain.grad, rtol=1e-5, atol=1e-6)
    assert objective["valid_count"] == 3
    assert all(parameter.grad is None for parameter in golden.base_model.parameters())
    golden.gain.grad = None


def test_shared_update_matches_unshared_actual_native_and_gain_state(monkeypatch):
    native, trainer = create_reciprocal_pc()
    reference_native, reference_trainer = create_reciprocal_pc()
    branch, reference = AdditiveRLSBranch(native), AdditiveRLSBranch(reference_native)
    with torch.no_grad():
        branch.gain.fill_(0.057484474033117294)
        reference.gain.copy_(branch.gain)
    optimizer = torch.optim.AdamW([branch.gain], lr=0.002, weight_decay=0.0)
    reference_optimizer = torch.optim.AdamW([reference.gain], lr=0.002, weight_decay=0.0)
    tokens = torch.arange(128, dtype=torch.long)[None]
    tokens[0, 0] = 32759
    targets = (tokens + 1) % 32768
    original = native._prefill_states
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(native, "_prefill_states", counted)
    rng = torch.get_rng_state()
    reference_optimizer.zero_grad(set_to_none=True)
    output = reference(tokens)
    expected_credit = reference.assign_gain_gradient(output, targets)
    del output
    expected_metrics = reference_trainer.local_update(tokens, targets)
    reference_optimizer.step()
    torch.set_rng_state(rng)
    actual = golden_local_update(branch, trainer, optimizer, tokens, targets)
    assert len(calls) == 1
    assert native._prefill_states is counted
    assert trainer.update_count == reference_trainer.update_count == 1
    assert_tree_equal(actual["native"], expected_metrics)
    assert_tree_equal(actual["gain"], expected_credit)
    assert_tree_equal(native.state_dict(), reference_native.state_dict())
    assert_tree_equal(trainer.state_dict(), reference_trainer.state_dict())
    assert_tree_equal(branch.branch_state_dict(), reference.branch_state_dict())
    assert_tree_equal(optimizer.state_dict(), reference_optimizer.state_dict())
    with pytest.raises(ValueError, match="only branch.gain"):
        golden_local_update(branch, trainer, torch.optim.AdamW(branch.parameters()), tokens, targets)
    branch._golden_inference_only = True
    with pytest.raises(ValueError, match="inference-only"):
        golden_local_update(branch, trainer, optimizer, tokens, targets)


def test_shared_prefill_restores_original_method_after_exception():
    native, trainer = create_reciprocal_pc()
    tokens = torch.tensor([[32759, 11]])
    assert "_prefill_states" not in native.__dict__
    with pytest.raises(RuntimeError, match="body failure"):
        with shared_prefill(native, trainer, tokens):
            raise RuntimeError("body failure")
    assert "_prefill_states" not in native.__dict__


def test_nonfinite_gain_fails_before_native_update_and_restores_prefill():
    native, trainer = create_reciprocal_pc()
    branch = AdditiveRLSBranch(native)
    with torch.no_grad():
        branch.gain.fill_(float("inf"))
    optimizer = torch.optim.AdamW([branch.gain], lr=0.002)
    tokens = torch.arange(128, dtype=torch.long)[None]
    tokens[0, 0] = 32759
    with pytest.raises(FloatingPointError, match="nonfinite native/additive"):
        golden_local_update(branch, trainer, optimizer, tokens, (tokens + 1) % 32768)
    assert trainer.update_count == 0
    assert not optimizer.state
    assert "_prefill_states" not in native.__dict__
