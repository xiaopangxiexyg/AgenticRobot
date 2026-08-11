"""Coherent conditional corrected-horizon previews from one frozen cut."""

from __future__ import annotations

from dataclasses import dataclass, field

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.component_action import ComponentAction

from control_composition.composition import (
    CompositionContextSnapshot,
    ProjectedCompositionEvaluation,
    ProjectedCompositionInputSnapshot,
    ProjectedCompositionStrategyInput,
    build_composition_context_snapshot,
    build_projected_composition_input_snapshot,
    evaluate_projected_composition_strategy,
)
from control_composition.eligibility import (
    CompositionSourceEligibilitySnapshot,
    build_composition_source_eligibility_snapshot,
)
from control_composition.residual_anchor_lifecycle import (
    ResidualAnchorLifecycleSnapshot,
    build_residual_anchor_lifecycle_snapshot,
)

from .preview import (
    TrajectoryIntentPreviewPoint,
    TrajectoryIntentPreviewSnapshot,
    build_trajectory_intent_preview,
)
from .projection import (
    FrozenTrajectoryCompositionProjection,
    TrajectoryProjectedInputSet,
    project_trajectory_composition_inputs,
)


TRAJECTORY_CORRECTED_PREVIEW_POINT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.trajectory_corrected_preview_point.development"
)
TRAJECTORY_CORRECTED_PREVIEW_SNAPSHOT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.trajectory_corrected_preview_snapshot.development"
)
TRAJECTORY_CORRECTED_PREVIEW_STATUSES_DEVELOPMENT = frozenset(
    {"ready", "unavailable", "invalid"}
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _context_block_reason(
    projection: FrozenTrajectoryCompositionProjection,
    *,
    input_snapshot: ProjectedCompositionInputSnapshot,
    context_snapshot: CompositionContextSnapshot | None,
) -> str | None:
    if context_snapshot is None:
        return None
    plan = projection.information_plan
    missing_required_channels = tuple(
        channel
        for channel in context_snapshot.missing_channels
        if plan is None or channel in plan.required_context_channels
    )
    if missing_required_channels:
        return "missing_context:" + ",".join(missing_required_channels)
    if plan is None:
        return None
    mismatches: list[str] = []
    for requirement in plan.context_requirements:
        if requirement.binding_mode != "selected_input_ancestor":
            continue
        if requirement.channel not in context_snapshot.selected_context_id_by_channel:
            continue
        context_id = context_snapshot.selected_context_id_by_channel[
            requirement.channel
        ]
        roles = requirement.bound_input_roles or tuple(
            input_snapshot.selected_input_by_role
        )
        mismatches.extend(
            f"{requirement.channel}@{role}"
            for role in roles
            if role not in input_snapshot.selected_input_by_role
            or context_id
            not in input_snapshot.selected_input_by_role[role].context_ids
        )
    if mismatches:
        return "context_dependency_mismatch:" + ",".join(sorted(mismatches))
    age_clock_mismatch = tuple(
        requirement.channel
        for requirement in plan.context_requirements
        if (
            requirement.maximum_age_sec is not None
            and requirement.channel in context_snapshot.selected_context_by_channel
            and context_snapshot.selected_context_by_channel[
                requirement.channel
            ].produced_clock_domain_id
            != input_snapshot.target_clock_domain_id
        )
    )
    if age_clock_mismatch:
        return "context_age_clock_mismatch:" + ",".join(age_clock_mismatch)
    stale = tuple(
        requirement.channel
        for requirement in plan.context_requirements
        if (
            requirement.maximum_age_sec is not None
            and requirement.channel in context_snapshot.selected_context_by_channel
            and input_snapshot.target_effective_at
            - context_snapshot.selected_context_by_channel[
                requirement.channel
            ].produced_at
            >= requirement.maximum_age_sec
        )
    )
    return None if not stale else "stale_context:" + ",".join(stale)


def _support_limits(
    projection: FrozenTrajectoryCompositionProjection,
    *,
    input_snapshot: ProjectedCompositionInputSnapshot,
    context_snapshot: CompositionContextSnapshot | None,
    source_eligibility_snapshot: CompositionSourceEligibilitySnapshot | None,
) -> tuple[float, ...]:
    limits: list[float] = []
    if context_snapshot is not None:
        limits.extend(
            value.effective_until
            for value in context_snapshot.selected_context_by_channel.values()
        )
        if projection.information_plan is not None:
            for requirement in projection.information_plan.context_requirements:
                if requirement.maximum_age_sec is None:
                    continue
                record = context_snapshot.selected_context_by_channel.get(
                    requirement.channel
                )
                if record is not None:
                    limits.append(
                        record.produced_at + requirement.maximum_age_sec
                    )
    if source_eligibility_snapshot is not None:
        limits.extend(
            source_eligibility_snapshot.selected_state_by_role[
                role
            ].lease_expires_at
            for role in input_snapshot.selected_input_by_role
            if role in source_eligibility_snapshot.selected_state_by_role
        )
    return tuple(limits)


@dataclass(frozen=True)
class TrajectoryCorrectedPreviewPoint:
    """One nominal keyframe and its conditional composition projection."""

    projection_basis_hash: str
    intent_point: TrajectoryIntentPreviewPoint
    projected_inputs: TrajectoryProjectedInputSet
    output_status: str
    reason: str
    input_snapshot: ProjectedCompositionInputSnapshot | None = None
    context_snapshot: CompositionContextSnapshot | None = None
    source_eligibility_snapshot: CompositionSourceEligibilitySnapshot | None = None
    residual_anchor_lifecycle_snapshot: (
        ResidualAnchorLifecycleSnapshot | None
    ) = None
    projected_evaluation: ProjectedCompositionEvaluation | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        basis_hash = _required(
            self.projection_basis_hash,
            name="projection_basis_hash",
        )
        if not isinstance(self.intent_point, TrajectoryIntentPreviewPoint):
            raise TypeError("intent_point must be TrajectoryIntentPreviewPoint.")
        if not isinstance(self.projected_inputs, TrajectoryProjectedInputSet):
            raise TypeError("projected_inputs must be TrajectoryProjectedInputSet.")
        if self.projected_inputs.projection_basis_hash != basis_hash:
            raise ValueError("Corrected preview point mixes projection bases.")
        if self.projected_inputs.preparation.target_effective_at != (
            self.intent_point.target_effective_at
        ):
            raise ValueError("Corrected preview target mismatch.")
        status = _required(self.output_status, name="output_status")
        if status not in TRAJECTORY_CORRECTED_PREVIEW_STATUSES_DEVELOPMENT:
            raise ValueError("Unsupported corrected preview status.")
        reason = _required(self.reason, name="reason")
        if self.input_snapshot is not None:
            if not isinstance(
                self.input_snapshot,
                ProjectedCompositionInputSnapshot,
            ):
                raise TypeError("input_snapshot must be projected or None.")
            if (
                self.input_snapshot.projection_basis_hash != basis_hash
                or self.input_snapshot.target_effective_at
                != self.intent_point.target_effective_at
            ):
                raise ValueError("Corrected input snapshot mismatch.")
        if self.context_snapshot is not None and not isinstance(
            self.context_snapshot,
            CompositionContextSnapshot,
        ):
            raise TypeError("context_snapshot must be typed or None.")
        if self.source_eligibility_snapshot is not None and not isinstance(
            self.source_eligibility_snapshot,
            CompositionSourceEligibilitySnapshot,
        ):
            raise TypeError("source_eligibility_snapshot must be typed or None.")
        if self.residual_anchor_lifecycle_snapshot is not None:
            lifecycle = self.residual_anchor_lifecycle_snapshot
            if not isinstance(lifecycle, ResidualAnchorLifecycleSnapshot):
                raise TypeError(
                    "residual_anchor_lifecycle_snapshot must be typed or None."
                )
            if (
                lifecycle.target_effective_at
                != self.intent_point.target_effective_at
            ):
                raise ValueError("Corrected lifecycle target mismatch.")
            if self.input_snapshot is not None and (
                lifecycle.control_domain_id
                != self.input_snapshot.control_domain_id
                or lifecycle.cutoff_at != self.input_snapshot.cutoff_at
                or lifecycle.cutoff_admission_sequence
                != self.input_snapshot.cutoff_admission_sequence
            ):
                raise ValueError("Corrected lifecycle/input cuts differ.")
        evaluation = self.projected_evaluation
        if evaluation is not None:
            if not isinstance(evaluation, ProjectedCompositionEvaluation):
                raise TypeError("projected_evaluation must be typed or None.")
            if (
                evaluation.projection_basis_hash != basis_hash
                or evaluation.target_effective_at
                != self.intent_point.target_effective_at
            ):
                raise ValueError("Corrected evaluation mismatch.")
        corrected = self.corrected_action
        if status == "ready" and corrected is None:
            raise ValueError("Ready corrected preview requires an action.")
        if status != "ready" and corrected is not None:
            raise ValueError("Only a ready corrected preview may carry action.")
        payload = {
            "projection_basis_hash": basis_hash,
            "intent_point_hash": self.intent_point.semantic_content_hash,
            "projected_input_set_hash": self.projected_inputs.semantic_content_hash,
            "output_status": status,
            "reason": reason,
            "projected_input_snapshot_hash": (
                None
                if self.input_snapshot is None
                else self.input_snapshot.semantic_content_hash
            ),
            "context_snapshot_hash": (
                None
                if self.context_snapshot is None
                else self.context_snapshot.semantic_content_hash
            ),
            "source_eligibility_snapshot_hash": (
                None
                if self.source_eligibility_snapshot is None
                else self.source_eligibility_snapshot.semantic_content_hash
            ),
            "residual_anchor_lifecycle_snapshot_hash": (
                None
                if self.residual_anchor_lifecycle_snapshot is None
                else self.residual_anchor_lifecycle_snapshot.semantic_content_hash
            ),
            "projected_evaluation_hash": (
                None if evaluation is None else evaluation.semantic_content_hash
            ),
            "corrected_action_hash": (
                None if corrected is None else corrected.semantic_content_hash
            ),
            "authoritative": False,
            "promotable": False,
            "execution_guaranteed": False,
        }
        object.__setattr__(self, "projection_basis_hash", basis_hash)
        object.__setattr__(self, "output_status", status)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    TRAJECTORY_CORRECTED_PREVIEW_POINT_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def point_id(self) -> str:
        return f"trajectory-corrected-preview-point:{self.semantic_content_hash}"

    @property
    def corrected_action(self) -> ComponentAction | None:
        evaluation = self.projected_evaluation
        if evaluation is None or evaluation.output_status != "ready":
            return None
        return evaluation.composed_candidate

    @property
    def promotable(self) -> bool:
        return False

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def execution_guaranteed(self) -> bool:
        return False


@dataclass(frozen=True)
class TrajectoryCorrectedPreviewSnapshot:
    """All trajectory keyframes evaluated from exactly one frozen basis."""

    projection: FrozenTrajectoryCompositionProjection
    intent_preview: TrajectoryIntentPreviewSnapshot
    points: tuple[TrajectoryCorrectedPreviewPoint, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.projection,
            FrozenTrajectoryCompositionProjection,
        ):
            raise TypeError("projection must be frozen and typed.")
        if not isinstance(self.intent_preview, TrajectoryIntentPreviewSnapshot):
            raise TypeError("intent_preview must be typed.")
        active = self.projection.state.active_trajectory
        if active is None or self.intent_preview.trajectory.semantic_content_hash != (
            active.semantic_content_hash
        ):
            raise ValueError("Intent preview and frozen active trajectory differ.")
        points = tuple(self.points)
        if len(points) != len(self.intent_preview.points):
            raise ValueError("Corrected horizon must cover every intent point.")
        for corrected, intent in zip(points, self.intent_preview.points):
            if not isinstance(corrected, TrajectoryCorrectedPreviewPoint):
                raise TypeError("points must contain corrected preview points.")
            if (
                corrected.projection_basis_hash
                != self.projection.semantic_content_hash
                or corrected.intent_point.semantic_content_hash
                != intent.semantic_content_hash
            ):
                raise ValueError("Corrected horizon point order/basis mismatch.")
        payload = {
            "projection_id": self.projection.projection_id,
            "projection_hash": self.projection.semantic_content_hash,
            "intent_preview_hash": self.intent_preview.semantic_content_hash,
            "point_hashes": tuple(value.semantic_content_hash for value in points),
            "authoritative": False,
            "promotable": False,
        }
        object.__setattr__(self, "points", points)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    TRAJECTORY_CORRECTED_PREVIEW_SNAPSHOT_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def preview_id(self) -> str:
        return f"trajectory-corrected-preview:{self.semantic_content_hash}"

    @property
    def promotable(self) -> bool:
        return False

    @property
    def authoritative(self) -> bool:
        return False


def _unavailable_point(
    projection: FrozenTrajectoryCompositionProjection,
    intent_point: TrajectoryIntentPreviewPoint,
    projected_inputs: TrajectoryProjectedInputSet,
    *,
    reason: str,
    invalid: bool = False,
    input_snapshot: ProjectedCompositionInputSnapshot | None = None,
    context_snapshot: CompositionContextSnapshot | None = None,
    source_eligibility_snapshot: CompositionSourceEligibilitySnapshot | None = None,
    residual_anchor_lifecycle_snapshot: (
        ResidualAnchorLifecycleSnapshot | None
    ) = None,
    evaluation: ProjectedCompositionEvaluation | None = None,
) -> TrajectoryCorrectedPreviewPoint:
    return TrajectoryCorrectedPreviewPoint(
        projection_basis_hash=projection.semantic_content_hash,
        intent_point=intent_point,
        projected_inputs=projected_inputs,
        output_status="invalid" if invalid else "unavailable",
        reason=reason,
        input_snapshot=input_snapshot,
        context_snapshot=context_snapshot,
        source_eligibility_snapshot=source_eligibility_snapshot,
        residual_anchor_lifecycle_snapshot=(
            residual_anchor_lifecycle_snapshot
        ),
        projected_evaluation=evaluation,
    )


def build_trajectory_corrected_preview(
    projection: FrozenTrajectoryCompositionProjection,
) -> TrajectoryCorrectedPreviewSnapshot:
    """Evaluate the whole horizon without touching any runtime/store owner."""

    if not isinstance(projection, FrozenTrajectoryCompositionProjection):
        raise TypeError("projection must be FrozenTrajectoryCompositionProjection.")
    active = projection.state.active_trajectory
    assert active is not None
    intent_preview = build_trajectory_intent_preview(
        active,
        generated_at=projection.control_cut.cutoff_at,
        generated_clock_domain_id=projection.control_cut.clock_domain_id,
    )
    points: list[TrajectoryCorrectedPreviewPoint] = []
    activation_binding = projection.state.residual_session_activation_binding
    activation_scope = (
        None if activation_binding is None else activation_binding.activation_scope
    )
    for intent_point in intent_preview.points:
        projected_inputs = project_trajectory_composition_inputs(
            projection,
            target_effective_at=intent_point.target_effective_at,
        )
        if projected_inputs.output_status != "ready":
            points.append(
                _unavailable_point(
                    projection,
                    intent_point,
                    projected_inputs,
                    reason=projected_inputs.reason,
                    invalid=projected_inputs.output_status == "invalid",
                )
            )
            continue

        eligibility_snapshot = None
        if projection.source_eligibility_projection is not None:
            assert projection.source_eligibility_spec is not None
            eligibility_snapshot = build_composition_source_eligibility_snapshot(
                projection.source_eligibility_projection,
                producer_epochs=projection.producer_epochs,
                spec=projection.source_eligibility_spec,
                composition_spec=projection.composition_spec,
                target_effective_at=intent_point.target_effective_at,
            )
        lifecycle_snapshot = None
        if projection.residual_anchor_lifecycle_projection is not None:
            assert projection.residual_anchor_lifecycle_spec is not None
            assert projection.residual_anchor_lifecycle_scope_id is not None
            assert activation_scope is not None
            lifecycle_snapshot = build_residual_anchor_lifecycle_snapshot(
                projection.residual_anchor_lifecycle_projection,
                producer_epochs=projection.producer_epochs,
                spec=projection.residual_anchor_lifecycle_spec,
                activation_scope=activation_scope,
                lifecycle_scope_id=(
                    projection.residual_anchor_lifecycle_scope_id
                ),
                target_effective_at=intent_point.target_effective_at,
            )
        try:
            input_snapshot = build_projected_composition_input_snapshot(
                projected_inputs.inputs,
                projection_basis_hash=projection.semantic_content_hash,
                control_cut=projection.control_cut,
                producer_epochs=projection.producer_epochs,
                spec=projection.composition_spec,
                target_effective_at=intent_point.target_effective_at,
                activation_scope=activation_scope,
                source_eligibility_snapshot=eligibility_snapshot,
                temporal_selection_spec=projection.temporal_selection_spec,
            )
        except (TypeError, ValueError) as exc:
            points.append(
                _unavailable_point(
                    projection,
                    intent_point,
                    projected_inputs,
                    reason=f"projected_input_snapshot_unavailable:{exc}",
                    residual_anchor_lifecycle_snapshot=lifecycle_snapshot,
                )
            )
            continue

        context_snapshot = None
        channels = projection.strategy.spec.context_channels
        if channels:
            assert projection.context_projection is not None
            context_snapshot = build_composition_context_snapshot(
                projection.context_projection,
                producer_epochs=projection.producer_epochs,
                target_effective_at=intent_point.target_effective_at,
                required_channels=channels,
            )
        blocked_reason = _context_block_reason(
            projection,
            input_snapshot=input_snapshot,
            context_snapshot=context_snapshot,
        )
        if blocked_reason is not None:
            points.append(
                _unavailable_point(
                    projection,
                    intent_point,
                    projected_inputs,
                    reason=blocked_reason,
                    input_snapshot=input_snapshot,
                    context_snapshot=context_snapshot,
                    source_eligibility_snapshot=eligibility_snapshot,
                    residual_anchor_lifecycle_snapshot=lifecycle_snapshot,
                )
            )
            continue

        strategy_input = ProjectedCompositionStrategyInput(
            runtime_session_id=projection.control_cut.runtime_session_id,
            control_domain_id=projection.control_cut.control_domain_id,
            clock_domain_id=projection.control_cut.clock_domain_id,
            strategy_spec=projection.strategy.spec,
            composition_spec=projection.composition_spec,
            input_snapshot=input_snapshot,
            context_snapshot=context_snapshot,
            information_plan=projection.information_plan,
            residual_anchor_lifecycle_snapshot=lifecycle_snapshot,
        )
        try:
            evaluation = evaluate_projected_composition_strategy(
                projection.strategy,
                strategy_input,
                support_limits=_support_limits(
                    projection,
                    input_snapshot=input_snapshot,
                    context_snapshot=context_snapshot,
                    source_eligibility_snapshot=eligibility_snapshot,
                ),
            )
        except (TypeError, ValueError) as exc:
            points.append(
                _unavailable_point(
                    projection,
                    intent_point,
                    projected_inputs,
                    reason=f"projected_strategy_invalid:{exc}",
                    invalid=True,
                    input_snapshot=input_snapshot,
                    context_snapshot=context_snapshot,
                    source_eligibility_snapshot=eligibility_snapshot,
                    residual_anchor_lifecycle_snapshot=lifecycle_snapshot,
                )
            )
            continue
        if evaluation.output_status != "ready":
            points.append(
                _unavailable_point(
                    projection,
                    intent_point,
                    projected_inputs,
                    reason=f"projected_strategy:{evaluation.reason}",
                    invalid=evaluation.output_status == "invalid",
                    input_snapshot=input_snapshot,
                    context_snapshot=context_snapshot,
                    source_eligibility_snapshot=eligibility_snapshot,
                    residual_anchor_lifecycle_snapshot=lifecycle_snapshot,
                    evaluation=evaluation,
                )
            )
            continue
        points.append(
            TrajectoryCorrectedPreviewPoint(
                projection_basis_hash=projection.semantic_content_hash,
                intent_point=intent_point,
                projected_inputs=projected_inputs,
                output_status="ready",
                reason="conditional_composition_ready",
                input_snapshot=input_snapshot,
                context_snapshot=context_snapshot,
                source_eligibility_snapshot=eligibility_snapshot,
                residual_anchor_lifecycle_snapshot=lifecycle_snapshot,
                projected_evaluation=evaluation,
            )
        )
    return TrajectoryCorrectedPreviewSnapshot(
        projection=projection,
        intent_preview=intent_preview,
        points=tuple(points),
    )


__all__ = [
    "TRAJECTORY_CORRECTED_PREVIEW_POINT_HASH_DOMAIN_DEVELOPMENT",
    "TRAJECTORY_CORRECTED_PREVIEW_SNAPSHOT_HASH_DOMAIN_DEVELOPMENT",
    "TRAJECTORY_CORRECTED_PREVIEW_STATUSES_DEVELOPMENT",
    "TrajectoryCorrectedPreviewPoint",
    "TrajectoryCorrectedPreviewSnapshot",
    "build_trajectory_corrected_preview",
]
