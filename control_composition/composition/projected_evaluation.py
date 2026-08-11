"""Strategy input and result identities for non-promotable projections."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.component_action import ComponentAction

from control_composition.information import CompositionInformationPlan
from control_composition.residual_anchor_lifecycle import (
    ResidualAnchorLifecycleSnapshot,
)

from .context import CompositionContextSnapshot
from .core import CompositionSpec
from .formal_strategy import (
    CompositionStrategyDecision,
    FormalCompositionStrategySpec,
)
from .projected_snapshot import ProjectedCompositionInputSnapshot


PROJECTED_COMPOSITION_STRATEGY_INPUT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.projected_composition_strategy_input.development"
)
PROJECTED_COMPOSITION_STRATEGY_INPUT_HASH_DOMAIN_V2_DEVELOPMENT = (
    "dexteleop.projected_composition_strategy_input.v2.development"
)
PROJECTED_COMPOSITION_EVALUATION_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.projected_composition_evaluation.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class ProjectedCompositionStrategyInput:
    """Structural strategy view whose identity can never enter promotion."""

    runtime_session_id: str
    control_domain_id: str
    clock_domain_id: str
    strategy_spec: FormalCompositionStrategySpec
    composition_spec: CompositionSpec
    input_snapshot: ProjectedCompositionInputSnapshot
    context_snapshot: CompositionContextSnapshot | None
    information_plan: CompositionInformationPlan | None
    residual_anchor_lifecycle_snapshot: (
        ResidualAnchorLifecycleSnapshot | None
    ) = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        session = _required(self.runtime_session_id, name="runtime_session_id")
        domain = _required(self.control_domain_id, name="control_domain_id")
        clock = _required(self.clock_domain_id, name="clock_domain_id")
        if not isinstance(self.strategy_spec, FormalCompositionStrategySpec):
            raise TypeError("strategy_spec must be FormalCompositionStrategySpec.")
        if not isinstance(self.composition_spec, CompositionSpec):
            raise TypeError("composition_spec must be CompositionSpec.")
        if not isinstance(
            self.input_snapshot,
            ProjectedCompositionInputSnapshot,
        ):
            raise TypeError(
                "input_snapshot must be ProjectedCompositionInputSnapshot."
            )
        if (
            self.strategy_spec.composition_spec_hash
            != self.composition_spec.semantic_content_hash
            or self.input_snapshot.composition_spec_hash
            != self.composition_spec.semantic_content_hash
        ):
            raise ValueError("Projected strategy contracts disagree.")
        if (
            self.input_snapshot.control_domain_id != domain
            or self.input_snapshot.target_clock_domain_id != clock
        ):
            raise ValueError("Projected strategy input belongs to another receiver.")
        channels = self.strategy_spec.context_channels
        context_id = None
        context_hash = None
        if channels:
            if self.information_plan is None or self.context_snapshot is None:
                raise ValueError("Projected strategy context is incomplete.")
            if (
                self.information_plan.context_channels != channels
                or self.information_plan.required_context_channels
                != self.strategy_spec.required_context_channels
            ):
                raise ValueError("Projected strategy information plan mismatch.")
        if self.context_snapshot is not None:
            if not isinstance(self.context_snapshot, CompositionContextSnapshot):
                raise TypeError("context_snapshot must be CompositionContextSnapshot.")
            if (
                self.context_snapshot.control_domain_id != domain
                or self.context_snapshot.cutoff_at
                != self.input_snapshot.cutoff_at
                or self.context_snapshot.cutoff_admission_sequence
                != self.input_snapshot.cutoff_admission_sequence
                or self.context_snapshot.target_effective_at
                != self.input_snapshot.target_effective_at
                or self.context_snapshot.required_channels != channels
            ):
                raise ValueError("Projected action/context snapshots differ.")
            context_id = self.context_snapshot.context_snapshot_id
            context_hash = self.context_snapshot.semantic_content_hash
        information_hash = (
            None
            if self.information_plan is None
            else self.information_plan.semantic_content_hash
        )
        lifecycle = self.residual_anchor_lifecycle_snapshot
        required_lifecycle_hash = (
            self.strategy_spec.required_residual_anchor_lifecycle_spec_hash
        )
        lifecycle_id = None
        lifecycle_hash = None
        if required_lifecycle_hash is not None:
            if not isinstance(lifecycle, ResidualAnchorLifecycleSnapshot):
                raise ValueError(
                    "Projected strategy requires residual anchor lifecycle."
                )
            if (
                lifecycle.lifecycle_spec_hash != required_lifecycle_hash
                or lifecycle.control_domain_id != domain
                or lifecycle.cutoff_at != self.input_snapshot.cutoff_at
                or lifecycle.cutoff_admission_sequence
                != self.input_snapshot.cutoff_admission_sequence
                or lifecycle.target_effective_at
                != self.input_snapshot.target_effective_at
            ):
                raise ValueError(
                    "Projected residual anchor lifecycle uses another cut."
                )
            lifecycle_id = lifecycle.lifecycle_snapshot_id
            lifecycle_hash = lifecycle.semantic_content_hash
        elif lifecycle is not None:
            raise ValueError(
                "Projected strategy did not declare residual anchor lifecycle."
            )
        payload = {
            "runtime_session_id": session,
            "control_domain_id": domain,
            "clock_domain_id": clock,
            "strategy_spec_hash": self.strategy_spec.semantic_content_hash,
            "composition_spec_hash": self.composition_spec.semantic_content_hash,
            "projected_input_snapshot_id": (
                self.input_snapshot.projected_input_snapshot_id
            ),
            "projected_input_snapshot_hash": (
                self.input_snapshot.semantic_content_hash
            ),
            "context_snapshot_id": context_id,
            "context_snapshot_hash": context_hash,
            "information_plan_hash": information_hash,
            "promotable": False,
        }
        hash_domain = PROJECTED_COMPOSITION_STRATEGY_INPUT_HASH_DOMAIN_DEVELOPMENT
        if lifecycle_id is not None:
            payload["residual_anchor_lifecycle_snapshot_id"] = lifecycle_id
            payload["residual_anchor_lifecycle_snapshot_hash"] = lifecycle_hash
            hash_domain = (
                PROJECTED_COMPOSITION_STRATEGY_INPUT_HASH_DOMAIN_V2_DEVELOPMENT
            )
        object.__setattr__(self, "runtime_session_id", session)
        object.__setattr__(self, "control_domain_id", domain)
        object.__setattr__(self, "clock_domain_id", clock)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=hash_domain,
            ),
        )

    @property
    def projected_strategy_input_id(self) -> str:
        return f"projected-composition-strategy-input:{self.semantic_content_hash}"

    @property
    def promotable(self) -> bool:
        return False

    @property
    def strategy_control_binding_id(self) -> None:
        return None

    @property
    def strategy_control_binding_hash(self) -> None:
        return None


class ProjectedCompositionStrategy(Protocol):
    @property
    def spec(self) -> FormalCompositionStrategySpec: ...

    def decide_projected(
        self,
        strategy_input: ProjectedCompositionStrategyInput,
    ) -> CompositionStrategyDecision: ...


@dataclass(frozen=True)
class ProjectedCompositionEvaluation:
    """Conditional composition result with no promotion surface."""

    projection_basis_hash: str
    projected_strategy_input_id: str
    projected_strategy_input_hash: str
    target_effective_at: float
    target_clock_domain_id: str
    output_status: str
    reason: str
    composed_candidate: ComponentAction | None
    projected_input_ids: tuple[str, ...]
    projected_input_hashes: tuple[str, ...]
    context_ids: tuple[str, ...]
    composition_supported_until: float | None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        basis_hash = _required(
            self.projection_basis_hash,
            name="projection_basis_hash",
        )
        strategy_input_id = _required(
            self.projected_strategy_input_id,
            name="projected_strategy_input_id",
        )
        strategy_input_hash = _required(
            self.projected_strategy_input_hash,
            name="projected_strategy_input_hash",
        )
        target = _finite(
            self.target_effective_at,
            name="target_effective_at",
        )
        clock = _required(
            self.target_clock_domain_id,
            name="target_clock_domain_id",
        )
        status = _required(self.output_status, name="output_status")
        if status not in {"ready", "no_action", "invalid"}:
            raise ValueError("Unsupported projected evaluation status.")
        reason = _required(self.reason, name="reason")
        if (status == "ready") != (self.composed_candidate is not None):
            raise ValueError("Only a ready projected evaluation may carry action.")
        input_ids = tuple(
            _required(value, name="projected_input_id")
            for value in self.projected_input_ids
        )
        input_hashes = tuple(
            _required(value, name="projected_input_hash")
            for value in self.projected_input_hashes
        )
        if len(input_ids) != len(input_hashes):
            raise ValueError("Projected input ids/hashes must be aligned.")
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("Projected input ids must be unique.")
        context_ids = tuple(
            _required(value, name="context_id") for value in self.context_ids
        )
        supported_until = self.composition_supported_until
        if supported_until is not None:
            supported_until = _finite(
                supported_until,
                name="composition_supported_until",
            )
            if status != "ready" or supported_until <= target:
                raise ValueError("Projected support must extend beyond a ready target.")
        elif status == "ready":
            raise ValueError("Ready projected evaluation requires support boundary.")
        payload = {
            "projection_basis_hash": basis_hash,
            "projected_strategy_input_id": strategy_input_id,
            "projected_strategy_input_hash": strategy_input_hash,
            "target_effective_at": target,
            "target_clock_domain_id": clock,
            "output_status": status,
            "reason": reason,
            "composed_candidate_hash": (
                None
                if self.composed_candidate is None
                else self.composed_candidate.semantic_content_hash
            ),
            "projected_input_ids": input_ids,
            "projected_input_hashes": input_hashes,
            "context_ids": context_ids,
            "composition_supported_until": supported_until,
            "authoritative": False,
            "promotable": False,
        }
        object.__setattr__(self, "projection_basis_hash", basis_hash)
        object.__setattr__(self, "projected_strategy_input_id", strategy_input_id)
        object.__setattr__(self, "projected_strategy_input_hash", strategy_input_hash)
        object.__setattr__(self, "target_effective_at", target)
        object.__setattr__(self, "target_clock_domain_id", clock)
        object.__setattr__(self, "output_status", status)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "projected_input_ids", input_ids)
        object.__setattr__(self, "projected_input_hashes", input_hashes)
        object.__setattr__(self, "context_ids", context_ids)
        object.__setattr__(
            self,
            "composition_supported_until",
            supported_until,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=PROJECTED_COMPOSITION_EVALUATION_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def projected_evaluation_id(self) -> str:
        return f"projected-composition-evaluation:{self.semantic_content_hash}"

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def promotable(self) -> bool:
        return False


def evaluate_projected_composition_strategy(
    strategy: ProjectedCompositionStrategy,
    strategy_input: ProjectedCompositionStrategyInput,
    *,
    support_limits: tuple[float, ...] = (),
) -> ProjectedCompositionEvaluation:
    """Run an explicitly projection-compatible pure strategy."""

    if not isinstance(strategy_input, ProjectedCompositionStrategyInput):
        raise TypeError(
            "strategy_input must be ProjectedCompositionStrategyInput."
        )
    decide_projected = getattr(strategy, "decide_projected", None)
    if not callable(decide_projected):
        raise TypeError(
            "Strategy does not declare non-authoritative projected evaluation."
        )
    if getattr(strategy, "spec", None) != strategy_input.strategy_spec:
        raise ValueError("Projected strategy spec mismatch.")
    decision = decide_projected(strategy_input)
    if not isinstance(decision, CompositionStrategyDecision):
        raise TypeError("Projected strategy must return CompositionStrategyDecision.")
    available_context_ids = (
        set()
        if strategy_input.context_snapshot is None
        else set(
            strategy_input.context_snapshot.selected_context_id_by_channel.values()
        )
    )
    if not set(decision.used_context_ids).issubset(available_context_ids):
        raise ValueError("Projected decision references context outside the cut.")
    snapshot = strategy_input.input_snapshot
    missing_required = tuple(
        role
        for role in strategy_input.composition_spec.required_roles
        if role not in snapshot.selected_input_by_role
    )
    if decision.output_status == "ready" and missing_required:
        raise ValueError("Projected strategy returned ready with missing roles.")
    if decision.composed_candidate is not None and (
        decision.composed_candidate.action_schema.semantic_content_hash
        != snapshot.action_schema_hash
    ):
        raise ValueError("Projected strategy returned another action schema.")
    ordered_inputs = tuple(
        snapshot.selected_input_by_role[role]
        for role in strategy_input.composition_spec.input_roles
        if role in snapshot.selected_input_by_role
    )
    limits = [value.effective_until for value in ordered_inputs]
    limits.extend(
        _finite(value, name="support_limit") for value in support_limits
    )
    supported_until = (
        min(limits) if decision.output_status == "ready" and limits else None
    )
    if (
        decision.output_status == "ready"
        and (supported_until is None or supported_until <= snapshot.target_effective_at)
    ):
        decision = CompositionStrategyDecision(
            output_status="no_action",
            reason="projected_support_expired",
            composed_candidate=None,
            used_context_ids=decision.used_context_ids,
        )
        supported_until = None
    return ProjectedCompositionEvaluation(
        projection_basis_hash=snapshot.projection_basis_hash,
        projected_strategy_input_id=(
            strategy_input.projected_strategy_input_id
        ),
        projected_strategy_input_hash=strategy_input.semantic_content_hash,
        target_effective_at=snapshot.target_effective_at,
        target_clock_domain_id=snapshot.target_clock_domain_id,
        output_status=decision.output_status,
        reason=decision.reason,
        composed_candidate=decision.composed_candidate,
        projected_input_ids=tuple(value.projected_input_id for value in ordered_inputs),
        projected_input_hashes=tuple(
            value.semantic_content_hash for value in ordered_inputs
        ),
        context_ids=decision.used_context_ids,
        composition_supported_until=supported_until,
    )


__all__ = [
    "PROJECTED_COMPOSITION_EVALUATION_HASH_DOMAIN_DEVELOPMENT",
    "PROJECTED_COMPOSITION_STRATEGY_INPUT_HASH_DOMAIN_DEVELOPMENT",
    "ProjectedCompositionEvaluation",
    "ProjectedCompositionStrategy",
    "ProjectedCompositionStrategyInput",
    "evaluate_projected_composition_strategy",
]
