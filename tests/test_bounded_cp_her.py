from __future__ import annotations

import pytest
import torch

from pfc.bounded_cp_her import BoundedCPHERHierarchy


def _materialize_components(components) -> torch.Tensor:
    rows = []
    for token in range(components.coefficients.shape[0]):
        tensor = None
        for rank in range(components.rank):
            term = components.coefficients[token, rank]
            for factor in components.factors:
                term = torch.einsum(
                    "...,i->...i",
                    term,
                    factor[token, rank],
                )
            tensor = term if tensor is None else tensor + term
        rows.append(tensor)
    return torch.stack(rows)


def _materialize_association(model: BoundedCPHERHierarchy, level: int) -> torch.Tensor:
    terms = []
    for rank in range(model.rank):
        term = model.input_factors[level][rank]
        for mode in range(level):
            term = torch.einsum(
                "...i,j->...ij",
                term,
                model.context_factors[level][rank, mode],
            )
        term = torch.einsum(
            "...i,j->...ij",
            term,
            model.value_factors[level][rank],
        )
        terms.append(term)
    return torch.stack(terms).sum(dim=0)


def test_bounded_cp_recursive_top_down_matches_dense_tensors() -> None:
    torch.manual_seed(1)
    model = BoundedCPHERHierarchy(width=3, depth=3, rank=2)
    for values in model.value_factors:
        values.normal_()
    memories = [torch.randn(4, 3) for _ in range(3)]

    cache = model.predict(memories)
    dense = [_materialize_association(model, level) for level in range(3)]
    top = torch.einsum("ni,ijkl->njkl", memories[2], dense[2])
    middle_weight = dense[1].unsqueeze(0) + top
    middle = torch.einsum("ni,nijk->njk", memories[1], middle_weight)
    base_weight = dense[0].unsqueeze(0) + middle
    expected = torch.einsum("ni,nij->nj", memories[0], base_weight)

    assert torch.allclose(cache.correction, expected, atol=2e-5, rtol=2e-5)
    assert cache.modulated[0].rank == model.depth * model.rank


def test_bounded_cp_local_vjps_match_autograd_oracle() -> None:
    torch.manual_seed(2)
    model = BoundedCPHERHierarchy(width=3, depth=2, rank=2)
    for values in model.value_factors:
        values.normal_()
    memories = [torch.randn(5, 3) for _ in range(2)]
    target = torch.randn(5, 3)
    result = model.local_gradients(memories, target)

    oracle = BoundedCPHERHierarchy(width=3, depth=2, rank=2)
    oracle.load_state_dict(model.state_dict())
    for parameter in oracle.parameters():
        parameter.requires_grad_(True)
    oracle_memories = [memory.clone().requires_grad_(True) for memory in memories]
    oracle_cache = oracle.predict(oracle_memories)
    for level in range(model.depth):
        # Outcomes remain fixed local targets. Oracle differentiates only the
        # current association and its superior dynamic-weight contraction.
        local_error = _materialize_components(result.errors[level]).detach()
        objective = (
            _materialize_components(oracle_cache.modulated[level])
            * local_error
        ).sum() / target.shape[0]
        parameters = [
            oracle.input_factors[level],
            oracle.value_factors[level],
            oracle_memories[level],
        ]
        names = [
            f"input_factors.{level}",
            f"value_factors.{level}",
        ]
        if oracle.context_factors[level].numel():
            parameters.insert(1, oracle.context_factors[level])
            names.insert(1, f"context_factors.{level}")
        expected = torch.autograd.grad(objective, parameters, retain_graph=True)
        for name, gradient in zip(names, expected[:-1]):
            assert torch.allclose(
                result.gradients[name],
                gradient,
                atol=2e-5,
                rtol=2e-5,
            )
        assert torch.allclose(
            result.memory_gradients[level],
            expected[-1] * target.shape[0],
            atol=2e-5,
            rtol=2e-5,
        )


def test_bounded_cp_updates_locally_without_parameter_gradients() -> None:
    torch.manual_seed(3)
    model = BoundedCPHERHierarchy(width=8, depth=3, rank=4)
    memories = [torch.randn(16, 8) for _ in range(3)]
    target = torch.randn(16, 8)
    before = {
        name: parameter.clone()
        for name, parameter in model.named_parameters()
    }

    result = model.local_update(
        memories,
        target,
        learning_rates=(0.1, 0.2, 0.2),
    )

    assert model.parameter_count == 4 * 8 * (2 + 3 + 4)
    assert result.cache.modulated[0].rank <= model.maximum_component_count
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in model.named_parameters()
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_bounded_cp_local_updates_reduce_fixed_target_error() -> None:
    torch.manual_seed(6)
    model = BoundedCPHERHierarchy(width=4, depth=2, rank=8)
    memories = [torch.randn(32, 4) for _ in range(2)]
    target = torch.tanh(memories[0] @ torch.randn(4, 4)) * 0.2

    before = BoundedCPHERHierarchy.relative_error_score(
        model.local_gradients(memories, target)
    )
    for _ in range(200):
        model.local_update(
            memories,
            target,
            learning_rates=(0.02, 0.02),
        )
        model.canonicalize_(maximum_component_norm=1.0)
    after = BoundedCPHERHierarchy.relative_error_score(
        model.local_gradients(memories, target)
    )

    assert after < before * 0.7
    assert all(parameter.grad is None for parameter in model.parameters())


def test_bounded_cp_canonicalization_preserves_association_and_fixes_gauge() -> None:
    torch.manual_seed(4)
    canonical = BoundedCPHERHierarchy(width=5, depth=3, rank=3)
    for values in canonical.value_factors:
        values.normal_()
    rescaled = BoundedCPHERHierarchy(width=5, depth=3, rank=3)
    rescaled.load_state_dict(canonical.state_dict())
    with torch.no_grad():
        for level in range(rescaled.depth):
            rescaled.input_factors[level].mul_(7.0)
            rescaled.value_factors[level].div_(7.0)
            for mode in range(level):
                rescaled.context_factors[level][:, mode, :].mul_(3.0)
                rescaled.value_factors[level].div_(3.0)
    before = [
        _materialize_association(canonical, level)
        for level in range(canonical.depth)
    ]
    assert all(
        torch.allclose(
            _materialize_association(rescaled, level),
            before[level],
            atol=2e-5,
            rtol=2e-5,
        )
        for level in range(canonical.depth)
    )

    canonical.canonicalize_(maximum_component_norm=1e6)
    rescaled.canonicalize_(maximum_component_norm=1e6)

    for level in range(canonical.depth):
        assert torch.allclose(
            _materialize_association(canonical, level),
            before[level],
            atol=2e-5,
            rtol=2e-5,
        )
        assert torch.allclose(
            canonical.input_factors[level],
            rescaled.input_factors[level],
            atol=2e-6,
            rtol=2e-6,
        )
        assert torch.allclose(
            canonical.context_factors[level],
            rescaled.context_factors[level],
            atol=2e-6,
            rtol=2e-6,
        )
        assert torch.allclose(
            canonical.value_factors[level],
            rescaled.value_factors[level],
            atol=2e-5,
            rtol=2e-5,
        )
        assert torch.allclose(
            canonical.input_factors[level].norm(dim=-1),
            torch.ones(canonical.rank),
            atol=2e-6,
        )
        if level:
            assert torch.allclose(
                canonical.context_factors[level].norm(dim=-1),
                torch.ones(canonical.rank, level),
                atol=2e-6,
            )


def test_bounded_cp_canonicalization_clips_full_component_norm() -> None:
    torch.manual_seed(5)
    model = BoundedCPHERHierarchy(width=6, depth=3, rank=4)
    for values in model.value_factors:
        values.fill_(2.0)

    stats = model.canonicalize_(maximum_component_norm=0.1)

    assert stats.component_count == 12
    assert stats.clipped_count == 12
    assert stats.maximum_norm_before_clip > 0.1
    assert stats.maximum_norm_after_clip == pytest.approx(0.1, abs=2e-7)
    for values in model.value_factors:
        assert torch.all(values.norm(dim=-1) <= 0.1000002)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_bounded_cp_tensor_energy_matches_dense_materialization() -> None:
    torch.manual_seed(6)
    model = BoundedCPHERHierarchy(width=3, depth=3, rank=2)
    for values in model.value_factors:
        values.normal_()
    memories = [torch.randn(4, 3) for _ in range(3)]
    target = torch.randn(4, 3)

    result = model.local_gradients(memories, target)
    expected_ratios = []
    for outcome, error in zip(result.outcomes, result.errors):
        dense_outcome = _materialize_components(outcome)
        dense_error = _materialize_components(error)
        outcome_energy = dense_outcome.flatten(1).square().sum(dim=1)
        error_energy = dense_error.flatten(1).square().sum(dim=1)
        assert torch.allclose(
            model.squared_norm(outcome),
            outcome_energy,
            atol=3e-5,
            rtol=3e-5,
        )
        assert torch.allclose(
            model.squared_norm(error),
            error_energy,
            atol=3e-5,
            rtol=3e-5,
        )
        expected_ratios.append((error_energy / outcome_energy).mean())
    assert torch.allclose(
        model.relative_error_score(result),
        torch.stack(expected_ratios).mean(),
        atol=3e-5,
        rtol=3e-5,
    )
