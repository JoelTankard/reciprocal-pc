from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn


@dataclass(frozen=True)
class BatchedCPComponents:
    coefficients: Tensor
    factors: tuple[Tensor, ...]

    @property
    def rank(self) -> int:
        return self.coefficients.shape[1]

    @property
    def order(self) -> int:
        return len(self.factors)


@dataclass(frozen=True)
class BoundedCPHERCache:
    memories: tuple[Tensor, ...]
    unmodulated: tuple[BatchedCPComponents, ...]
    modulated: tuple[BatchedCPComponents, ...]
    correction: Tensor


@dataclass(frozen=True)
class BoundedCPHERLocalResult:
    cache: BoundedCPHERCache
    outcomes: tuple[BatchedCPComponents, ...]
    errors: tuple[BatchedCPComponents, ...]
    gradients: dict[str, Tensor]
    memory_gradients: tuple[Tensor, ...]


@dataclass(frozen=True)
class BoundedCPHERCanonicalizationStats:
    component_count: int
    clipped_count: int
    maximum_norm_before_clip: float
    maximum_norm_after_clip: float


def _scale(
    components: BatchedCPComponents,
    value: float,
) -> BatchedCPComponents:
    return BatchedCPComponents(
        coefficients=components.coefficients * value,
        factors=components.factors,
    )


def _concatenate(
    left: BatchedCPComponents,
    right: BatchedCPComponents,
) -> BatchedCPComponents:
    if left.order != right.order:
        raise ValueError("CP component orders must match")
    return BatchedCPComponents(
        coefficients=torch.cat((left.coefficients, right.coefficients), dim=1),
        factors=tuple(
            torch.cat((left_factor, right_factor), dim=1)
            for left_factor, right_factor in zip(left.factors, right.factors)
        ),
    )


def _prepend(
    components: BatchedCPComponents,
    factor: Tensor,
) -> BatchedCPComponents:
    expanded = factor.unsqueeze(1).expand(-1, components.rank, -1)
    return BatchedCPComponents(
        coefficients=components.coefficients,
        factors=(expanded, *components.factors),
    )


def _contract_first(
    components: BatchedCPComponents,
    vector: Tensor,
) -> BatchedCPComponents:
    coefficient = components.coefficients * torch.einsum(
        "nqd,nd->nq",
        components.factors[0],
        vector,
    )
    return BatchedCPComponents(
        coefficients=coefficient,
        factors=components.factors[1:],
    )


def _to_vector(components: BatchedCPComponents) -> Tensor:
    if components.order != 1:
        raise ValueError("vector conversion requires first-order components")
    return torch.einsum(
        "nq,nqd->nd",
        components.coefficients,
        components.factors[0],
    )


def _squared_norm(components: BatchedCPComponents) -> Tensor:
    gram = (
        components.coefficients.unsqueeze(2)
        * components.coefficients.unsqueeze(1)
    )
    for factor in components.factors:
        gram = gram * torch.einsum("nqd,nrd->nqr", factor, factor)
    return gram.sum(dim=(1, 2)).clamp_min(0.0)


class BoundedCPHERHierarchy(nn.Module):
    """Bounded CP approximation to recursively expanding HER tensors."""

    def __init__(self, *, width: int, depth: int, rank: int) -> None:
        super().__init__()
        if min(width, depth, rank) <= 0:
            raise ValueError("width, depth, and rank must be positive")
        self.width = width
        self.depth = depth
        self.rank = rank
        self.input_factors = nn.ParameterList(
            [nn.Parameter(torch.empty(rank, width)) for _ in range(depth)]
        )
        self.context_factors = nn.ParameterList(
            [
                nn.Parameter(torch.empty(rank, level, width))
                for level in range(depth)
            ]
        )
        self.value_factors = nn.ParameterList(
            [nn.Parameter(torch.zeros(rank, width)) for _ in range(depth)]
        )
        self.reset_parameters()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def reset_parameters(self) -> None:
        scale = self.width**-0.5
        with torch.no_grad():
            for inputs, contexts, values in zip(
                self.input_factors,
                self.context_factors,
                self.value_factors,
            ):
                nn.init.normal_(inputs, mean=0.0, std=scale)
                if contexts.numel():
                    nn.init.normal_(contexts, mean=0.0, std=scale)
                values.zero_()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def maximum_component_count(self) -> int:
        return self.depth * self.rank

    @torch.no_grad()
    def canonicalize_(
        self,
        *,
        maximum_component_norm: float,
        eps: float = 1e-12,
    ) -> BoundedCPHERCanonicalizationStats:
        """Fix CP scale gauge and bound each rank-one association tensor."""
        if maximum_component_norm <= 0:
            raise ValueError("maximum_component_norm must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        clipped_count = 0
        maximum_before = 0.0
        maximum_after = 0.0
        for inputs, contexts, values in zip(
            self.input_factors,
            self.context_factors,
            self.value_factors,
        ):
            input_norms = inputs.norm(dim=-1, keepdim=True)
            inputs.div_(input_norms.clamp_min(eps))
            transferred_scale = input_norms.squeeze(-1)
            for mode in range(contexts.shape[1]):
                context = contexts[:, mode, :]
                context_norms = context.norm(dim=-1, keepdim=True)
                context.div_(context_norms.clamp_min(eps))
                transferred_scale.mul_(context_norms.squeeze(-1))
            values.mul_(transferred_scale.unsqueeze(-1))
            component_norms = values.norm(dim=-1, keepdim=True)
            maximum_before = max(
                maximum_before,
                float(component_norms.max().item()),
            )
            scales = (maximum_component_norm / component_norms.clamp_min(eps)).clamp(
                max=1.0
            )
            clipped_count += int((scales < 1.0).sum().item())
            values.mul_(scales)
            maximum_after = max(
                maximum_after,
                float(values.norm(dim=-1).max().item()),
            )
        return BoundedCPHERCanonicalizationStats(
            component_count=self.depth * self.rank,
            clipped_count=clipped_count,
            maximum_norm_before_clip=maximum_before,
            maximum_norm_after_clip=maximum_after,
        )

    def _validate_memories(self, memories: list[Tensor] | tuple[Tensor, ...]) -> None:
        if len(memories) != self.depth:
            raise ValueError("memory count must equal hierarchy depth")
        expected = memories[0].shape
        if len(expected) != 2 or expected[1] != self.width:
            raise ValueError("memories must have shape [tokens, width]")
        if any(memory.shape != expected for memory in memories):
            raise ValueError("all memories must have equal shape")

    def _association(self, level: int, memory: Tensor) -> BatchedCPComponents:
        coefficients = memory @ self.input_factors[level].t()
        token_count = memory.shape[0]
        factors = tuple(
            self.context_factors[level][:, mode, :]
            .unsqueeze(0)
            .expand(token_count, -1, -1)
            for mode in range(level)
        )
        value = self.value_factors[level].unsqueeze(0).expand(
            token_count,
            -1,
            -1,
        )
        return BatchedCPComponents(
            coefficients=coefficients,
            factors=(*factors, value),
        )

    def predict(
        self,
        memories: list[Tensor] | tuple[Tensor, ...],
    ) -> BoundedCPHERCache:
        self._validate_memories(memories)
        memory_tuple = tuple(memories)
        unmodulated = tuple(
            self._association(level, memory_tuple[level])
            for level in range(self.depth)
        )
        modulated = list(unmodulated)
        for level in range(self.depth - 2, -1, -1):
            modulated[level] = _concatenate(
                modulated[level],
                _contract_first(modulated[level + 1], memory_tuple[level]),
            )
        correction = _to_vector(modulated[0])
        return BoundedCPHERCache(
            memories=memory_tuple,
            unmodulated=unmodulated,
            modulated=tuple(modulated),
            correction=correction,
        )

    def base_contributions(self, cache: BoundedCPHERCache) -> tuple[Tensor, ...]:
        contributions: list[Tensor] = []
        for level, components in enumerate(cache.unmodulated):
            contracted = components
            for lower_level in range(level - 1, -1, -1):
                contracted = _contract_first(
                    contracted,
                    cache.memories[lower_level],
                )
            contributions.append(_to_vector(contracted))
        return tuple(contributions)

    @staticmethod
    def squared_norm(components: BatchedCPComponents) -> Tensor:
        return _squared_norm(components)

    @staticmethod
    def relative_error_score(
        result: BoundedCPHERLocalResult,
        *,
        eps: float = 1e-12,
    ) -> Tensor:
        if eps <= 0:
            raise ValueError("eps must be positive")
        level_scores = [
            (
                _squared_norm(error)
                / _squared_norm(outcome).clamp_min(eps)
            ).mean()
            for outcome, error in zip(result.outcomes, result.errors)
        ]
        return torch.stack(level_scores).mean()

    def _outcomes(
        self,
        cache: BoundedCPHERCache,
        target: Tensor,
    ) -> tuple[BatchedCPComponents, ...]:
        if target.shape != cache.memories[0].shape:
            raise ValueError("target must match flattened memory shape")
        outcomes = [
            BatchedCPComponents(
                coefficients=target.new_ones((target.shape[0], 1)),
                factors=(target.unsqueeze(1),),
            )
        ]
        for level in range(self.depth - 1):
            residual = _concatenate(
                outcomes[level],
                _scale(cache.unmodulated[level], -1.0),
            )
            outcomes.append(_prepend(residual, cache.memories[level]))
        return tuple(outcomes)

    @staticmethod
    def _association_gradient(
        *,
        memory: Tensor,
        input_factors: Tensor,
        context_factors: Tensor,
        value_factors: Tensor,
        error: BatchedCPComponents,
    ) -> tuple[Tensor, Tensor, Tensor]:
        level = context_factors.shape[1]
        source_coefficients = memory @ input_factors.t()
        own_factors = [
            *[context_factors[:, mode, :] for mode in range(level)],
            value_factors,
        ]
        dots = [
            torch.einsum("rd,nqd->nrq", factor, error_factor)
            for factor, error_factor in zip(own_factors, error.factors)
        ]
        products = torch.stack(dots).prod(dim=0)
        weighted_products = products * error.coefficients.unsqueeze(1)
        input_scalars = weighted_products.sum(dim=-1)
        token_count = max(memory.shape[0], 1)
        input_gradient = torch.einsum(
            "nr,nd->rd",
            input_scalars,
            memory,
        ) / token_count
        factor_gradients: list[Tensor] = []
        for mode, error_factor in enumerate(error.factors):
            product_except = torch.ones_like(products)
            for other_mode, dot in enumerate(dots):
                if other_mode != mode:
                    product_except = product_except * dot
            weights = (
                source_coefficients.unsqueeze(-1)
                * product_except
                * error.coefficients.unsqueeze(1)
            )
            factor_gradients.append(
                torch.einsum("nrq,nqd->rd", weights, error_factor)
                / token_count
            )
        if level:
            context_gradient = torch.stack(factor_gradients[:-1], dim=1)
        else:
            context_gradient = torch.zeros_like(context_factors)
        return input_gradient, context_gradient, factor_gradients[-1]

    @staticmethod
    def _memory_gradient(
        *,
        input_factors: Tensor,
        context_factors: Tensor,
        value_factors: Tensor,
        error: BatchedCPComponents,
        superior: BatchedCPComponents | None,
    ) -> Tensor:
        level = context_factors.shape[1]
        own_factors = [
            *[context_factors[:, mode, :] for mode in range(level)],
            value_factors,
        ]
        own_products = torch.stack(
            [
                torch.einsum("rd,nqd->nrq", factor, error_factor)
                for factor, error_factor in zip(own_factors, error.factors)
            ]
        ).prod(dim=0)
        own_projections = (
            own_products * error.coefficients.unsqueeze(1)
        ).sum(dim=-1)
        gradient = own_projections @ input_factors
        if superior is not None:
            superior_products = error.coefficients.new_ones(
                (
                    error.coefficients.shape[0],
                    superior.rank,
                    error.rank,
                )
            )
            for superior_factor, error_factor in zip(
                superior.factors[1:],
                error.factors,
            ):
                superior_products = superior_products * torch.einsum(
                    "nrd,nqd->nrq",
                    superior_factor,
                    error_factor,
                )
            superior_projections = superior.coefficients * (
                superior_products * error.coefficients.unsqueeze(1)
            ).sum(dim=-1)
            gradient = gradient + torch.einsum(
                "nr,nrd->nd",
                superior_projections,
                superior.factors[0],
            )
        return gradient

    def local_gradients(
        self,
        memories: list[Tensor] | tuple[Tensor, ...],
        target: Tensor,
    ) -> BoundedCPHERLocalResult:
        cache = self.predict(memories)
        outcomes = self._outcomes(cache, target)
        errors = tuple(
            _concatenate(modulated, _scale(outcome, -1.0))
            for modulated, outcome in zip(cache.modulated, outcomes)
        )
        gradients: dict[str, Tensor] = {}
        memory_gradients: list[Tensor] = []
        for level, error in enumerate(errors):
            input_gradient, context_gradient, value_gradient = (
                self._association_gradient(
                    memory=cache.memories[level],
                    input_factors=self.input_factors[level],
                    context_factors=self.context_factors[level],
                    value_factors=self.value_factors[level],
                    error=error,
                )
            )
            gradients[f"input_factors.{level}"] = input_gradient
            gradients[f"context_factors.{level}"] = context_gradient
            gradients[f"value_factors.{level}"] = value_gradient
            memory_gradients.append(
                self._memory_gradient(
                    input_factors=self.input_factors[level],
                    context_factors=self.context_factors[level],
                    value_factors=self.value_factors[level],
                    error=error,
                    superior=(
                        cache.modulated[level + 1]
                        if level + 1 < self.depth
                        else None
                    ),
                )
            )
        return BoundedCPHERLocalResult(
            cache=cache,
            outcomes=outcomes,
            errors=errors,
            gradients=gradients,
            memory_gradients=tuple(memory_gradients),
        )

    @torch.no_grad()
    def local_update(
        self,
        memories: list[Tensor] | tuple[Tensor, ...],
        target: Tensor,
        *,
        learning_rates: tuple[float, ...],
    ) -> BoundedCPHERLocalResult:
        if len(learning_rates) != self.depth:
            raise ValueError("learning_rates must cover every hierarchy level")
        result = self.local_gradients(memories, target)
        parameters = dict(self.named_parameters())
        for name, gradient in result.gradients.items():
            level = int(name.rsplit(".", 1)[1])
            parameters[name].add_(gradient, alpha=-learning_rates[level])
        return result
