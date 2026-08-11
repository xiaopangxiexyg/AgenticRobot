"""Evaluate exact trajectory targets without proposal publication."""

from __future__ import annotations

from robot_motion.action.component_action import ComponentAction

from ..composition import CompositionInputAdmissionReceipt, CompositionSpec
from ..observability import SourceDifferenceSpec
from ..runtime import CompositeProposalSubmission, ControlCompositionRuntime
from .contracts import (
    TRAJECTORY_EDITING_MODE_PASS_THROUGH,
    TrajectoryEditingState,
    TrajectoryTargetPreparation,
)
from .input_preparation import prepare_trajectory_composition_inputs
from .shadow_contracts import (
    TrajectoryCompositionContributorStamp,
    TrajectoryCompositionShadowResult,
)
from .warp import ResidualTemporalWarpSpec, TrajectoryResidualTemporalSample


def _state_preview_fields(
    state: TrajectoryEditingState,
    *,
    target_effective_at: float,
) -> dict[str, object]:
    base = state.active_base_binding
    shadow = state.shadow_trajectory
    shadow_keyframe = (
        None if shadow is None else shadow.keyframe_at(target_effective_at)
    )
    return {
        "active_base_binding_id": (
            None if base is None else base.base_binding_id
        ),
        "active_base_binding_hash": (
            None if base is None else base.semantic_content_hash
        ),
        "shadow_trajectory_id": (
            None if shadow is None else shadow.trajectory_id
        ),
        "shadow_trajectory_hash": (
            None if shadow is None else shadow.semantic_content_hash
        ),
        "shadow_keyframe_hash": (
            None
            if shadow_keyframe is None
            else shadow_keyframe.semantic_content_hash
        ),
        "shadow_action": (
            None if shadow_keyframe is None else shadow_keyframe.action
        ),
    }


def _result(
    state: TrajectoryEditingState,
    preparation: TrajectoryTargetPreparation,
    *,
    output_status: str,
    reason: str,
    residual_sample: TrajectoryResidualTemporalSample | None = None,
    nominal_input_admission: CompositionInputAdmissionReceipt | None = None,
    residual_input_admission: CompositionInputAdmissionReceipt | None = None,
    composition_submission: CompositeProposalSubmission | None = None,
) -> TrajectoryCompositionShadowResult:
    return TrajectoryCompositionShadowResult(
        editing_state_hash=state.semantic_content_hash,
        interaction_mode=state.interaction_mode,
        base_lifecycle_status=state.base_lifecycle_status,
        preparation=preparation,
        output_status=output_status,
        reason=reason,
        residual_sample=residual_sample,
        nominal_input_admission=nominal_input_admission,
        residual_input_admission=residual_input_admission,
        composition_submission=composition_submission,
        **_state_preview_fields(
            state,
            target_effective_at=preparation.target_effective_at,
        ),
    )


def evaluate_trajectory_composition_shadow(
    runtime: ControlCompositionRuntime,
    *,
    state: TrajectoryEditingState,
    target_effective_at: float,
    composition_spec: CompositionSpec,
    nominal_role: str,
    residual_role: str,
    nominal_stamp: TrajectoryCompositionContributorStamp,
    residual_stamp: TrajectoryCompositionContributorStamp | None = None,
    residual_action: ComponentAction | None = None,
    residual_fact_id: str | None = None,
    residual_fact_hash: str | None = None,
    residual_anchor_receipt_id: str | None = None,
    residual_anchor_receipt_hash: str | None = None,
    residual_effective_until: float | None = None,
    residual_warp_spec: ResidualTemporalWarpSpec | None = None,
    required_context_channels: tuple[str, ...] = (),
    source_difference_spec: SourceDifferenceSpec | None = None,
    cutoff_at: float | None = None,
    residual_anchor_lifecycle_scope_id: str | None = None,
) -> TrajectoryCompositionShadowResult:
    """Evaluate one exact target and stop before proposal publication."""

    inputs = prepare_trajectory_composition_inputs(
        runtime,
        state=state,
        target_effective_at=target_effective_at,
        composition_spec=composition_spec,
        nominal_role=nominal_role,
        residual_role=residual_role,
        nominal_stamp=nominal_stamp,
        residual_stamp=residual_stamp,
        residual_action=residual_action,
        residual_fact_id=residual_fact_id,
        residual_fact_hash=residual_fact_hash,
        residual_anchor_receipt_id=residual_anchor_receipt_id,
        residual_anchor_receipt_hash=residual_anchor_receipt_hash,
        residual_effective_until=residual_effective_until,
        residual_warp_spec=residual_warp_spec,
        admission_mode="shadow",
    )
    if (
        inputs.output_status != "ready"
        or state.interaction_mode == TRAJECTORY_EDITING_MODE_PASS_THROUGH
    ):
        return _result(
            state,
            inputs.preparation,
            output_status=inputs.output_status,
            reason=inputs.reason,
            residual_sample=inputs.residual_sample,
            nominal_input_admission=inputs.nominal_input_admission,
            residual_input_admission=inputs.residual_input_admission,
        )

    submission = runtime.evaluate_composition(
        spec=composition_spec,
        target_effective_at=inputs.preparation.target_effective_at,
        required_context_channels=tuple(required_context_channels),
        cutoff_at=cutoff_at,
        source_difference_spec=source_difference_spec,
        activation_scope=(
            state.residual_session_activation_binding.activation_scope
        ),
        residual_anchor_lifecycle_scope_id=(
            residual_anchor_lifecycle_scope_id
        ),
    )
    evaluation = submission.composition_evaluation
    return _result(
        state,
        inputs.preparation,
        output_status=evaluation.output_status,
        reason=f"composition_evaluation:{evaluation.reason}",
        residual_sample=inputs.residual_sample,
        nominal_input_admission=inputs.nominal_input_admission,
        residual_input_admission=inputs.residual_input_admission,
        composition_submission=submission,
    )


__all__ = ["evaluate_trajectory_composition_shadow"]
