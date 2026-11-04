"""Deterministic reducer for latching one nominal trajectory revision."""

from __future__ import annotations

from dataclasses import replace

from robot_motion.action import ComponentActionTrajectory
from robot_motion.runtime.control_admission import ControlCut

from ..activation import CompositionActivationScope
from ..residual_anchor import ResidualAnchorReceipt, ResidualAnchorRequest

from ._validation import _finite, _nonnegative_int, _required
from .contracts import (
    TRAJECTORY_EDITING_MODE_PASS_THROUGH,
    TRAJECTORY_EDITING_MODE_RESIDUAL_ACTIVE,
    TRAJECTORY_TARGET_STATUS_ALREADY_EMITTED,
    TRAJECTORY_TARGET_STATUS_NO_ACTION,
    TRAJECTORY_TARGET_STATUS_READY,
    TrajectoryEditingState,
    TrajectoryTargetPreparation,
)
from .lifecycle import (
    TRAJECTORY_BASE_LIFECYCLE_EXHAUSTED,
    TRAJECTORY_BASE_LIFECYCLE_STABLE,
    TRAJECTORY_INTERACTION_MODE_PASS_THROUGH,
    NominalBaseBinding,
    ResidualSessionActivationBinding,
    validate_trajectory_interaction_base_state,
)
from .anchor_lifecycle import (
    RESIDUAL_REBASE_CAPTURE_FIRST_FRESH_AFTER_BARRIER,
    ActiveResidualAnchorBinding,
    ResidualRebaseRequest,
    build_successor_active_residual_anchor_binding,
)


def empty_trajectory_editing_state(*, reducer_id: str) -> TrajectoryEditingState:
    return TrajectoryEditingState(
        reducer_id=_required(reducer_id, name="reducer_id"),
        revision=0,
        interaction_mode=TRAJECTORY_EDITING_MODE_PASS_THROUGH,
        base_lifecycle_status=TRAJECTORY_BASE_LIFECYCLE_STABLE,
    )


def _validate_trajectory_compatibility(
    active: ComponentActionTrajectory | None,
    candidate: ComponentActionTrajectory,
) -> None:
    if active is None:
        return
    if candidate.clock_domain_id != active.clock_domain_id:
        raise ValueError("Nominal trajectories use another clock domain.")
    if candidate.action_schema_hash != active.action_schema_hash:
        raise ValueError("Nominal trajectories use another action schema.")
    if candidate.component_coverage != active.component_coverage:
        raise ValueError("Nominal trajectories change component coverage.")


def offer_nominal_trajectory(
    state: TrajectoryEditingState,
    trajectory: ComponentActionTrajectory,
    *,
    offer_sequence: int,
) -> TrajectoryEditingState:
    """Admit a receiver-ordered immutable trajectory offer.

    Pass-through tracks the latest offer.  During a residual session the
    active trajectory stays latched and only one latest shadow candidate is
    retained.  No producer plan is mutated or committed here.
    """

    if not isinstance(state, TrajectoryEditingState):
        raise TypeError("state must be a TrajectoryEditingState.")
    if not isinstance(trajectory, ComponentActionTrajectory):
        raise TypeError("trajectory must be a ComponentActionTrajectory.")
    sequence = _nonnegative_int(offer_sequence, name="offer_sequence")
    maximum = state.maximum_offer_sequence
    if maximum is not None and sequence <= maximum:
        raise ValueError("Trajectory offer sequence must strictly increase.")
    _validate_trajectory_compatibility(state.active_trajectory, trajectory)

    if state.interaction_mode == TRAJECTORY_EDITING_MODE_PASS_THROUGH:
        return replace(
            state,
            revision=state.revision + 1,
            active_trajectory=trajectory,
            active_offer_sequence=sequence,
            shadow_trajectory=None,
            shadow_offer_sequence=None,
        )
    if state.interaction_mode != TRAJECTORY_EDITING_MODE_PASS_THROUGH:
        return replace(
            state,
            revision=state.revision + 1,
            shadow_trajectory=trajectory,
            shadow_offer_sequence=sequence,
        )
    raise ValueError("Unsupported trajectory editing mode.")


def activate_residual_session(
    state: TrajectoryEditingState,
    *,
    residual_session_id: str,
    activation_scope: CompositionActivationScope,
    control_cut: ControlCut,
    residual_anchor_receipt: ResidualAnchorReceipt,
    first_editable_target_at: float,
) -> TrajectoryEditingState:
    """Atomically latch one cut-selected initial base and anchor receipt."""

    if not isinstance(state, TrajectoryEditingState):
        raise TypeError("state must be a TrajectoryEditingState.")
    if state.interaction_mode != TRAJECTORY_EDITING_MODE_PASS_THROUGH:
        raise ValueError("A residual session is already active.")
    if state.active_trajectory is None:
        raise ValueError("Cannot activate residual editing without a base.")
    assert state.active_offer_sequence is not None
    session_id = _required(
        residual_session_id,
        name="residual_session_id",
    )
    boundary = _finite(
        first_editable_target_at,
        name="first_editable_target_at",
    )
    base_binding = NominalBaseBinding(
        residual_session_id=session_id,
        binding_revision=0,
        trajectory=state.active_trajectory,
        active_offer_sequence=state.active_offer_sequence,
        bound_for_target_at=boundary,
    )
    activation_binding = ResidualSessionActivationBinding(
        residual_session_id=session_id,
        activation_scope=activation_scope,
        control_cut=control_cut,
        residual_anchor_receipt=residual_anchor_receipt,
        initial_base_binding=base_binding,
        initial_publication_finalized_target_at=(
            state.last_emitted_target_at
        ),
        initial_published_action_hash=state.last_emitted_action_hash,
        initial_emission_receipt_id=state.last_emission_receipt_id,
        initial_emission_receipt_hash=state.last_emission_receipt_hash,
    )
    anchor_binding = ActiveResidualAnchorBinding.initial(
        residual_session_id=session_id,
        residual_anchor_receipt=residual_anchor_receipt,
        effective_for_target_at=boundary,
    )
    return replace(
        state,
        revision=state.revision + 1,
        interaction_mode=TRAJECTORY_EDITING_MODE_RESIDUAL_ACTIVE,
        base_lifecycle_status=TRAJECTORY_BASE_LIFECYCLE_STABLE,
        residual_session_activation_binding=activation_binding,
        active_base_binding=base_binding,
        active_residual_anchor_binding=anchor_binding,
        pending_residual_rebase=None,
        shadow_trajectory=None,
        shadow_offer_sequence=None,
    )


def request_residual_rebase(
    state: TrajectoryEditingState,
    *,
    anchor_request: ResidualAnchorRequest,
    effective_not_before_target_at: float,
    capture_deadline: float,
    trigger_kind: str,
    trigger_fact_id: str,
    trigger_fact_hash: str,
    capture_mode: str = RESIDUAL_REBASE_CAPTURE_FIRST_FRESH_AFTER_BARRIER,
    trigger_source_action_id: str | None = None,
    trigger_source_action_hash: str | None = None,
) -> TrajectoryEditingState:
    """Fence the old anchor from one exact future target boundary onward."""

    if not isinstance(state, TrajectoryEditingState):
        raise TypeError("state must be a TrajectoryEditingState.")
    if state.interaction_mode != TRAJECTORY_EDITING_MODE_RESIDUAL_ACTIVE:
        raise ValueError("Residual rebase requires a residual-active session.")
    active_anchor = state.active_residual_anchor_binding
    active_trajectory = state.active_trajectory
    if active_anchor is None or active_trajectory is None:
        raise ValueError("Residual-active state has no active anchor or base.")
    request = ResidualRebaseRequest.for_active_binding(
        active_anchor,
        anchor_request=anchor_request,
        effective_not_before_target_at=effective_not_before_target_at,
        capture_deadline=capture_deadline,
        trigger_kind=trigger_kind,
        trigger_fact_id=trigger_fact_id,
        trigger_fact_hash=trigger_fact_hash,
        capture_mode=capture_mode,
        trigger_source_action_id=trigger_source_action_id,
        trigger_source_action_hash=trigger_source_action_hash,
    )
    boundary = request.effective_not_before_target_at
    if active_trajectory.keyframe_at(boundary) is None:
        raise ValueError("Residual rebase boundary must be a trajectory keyframe.")
    if state.last_emitted_target_at is not None and boundary <= (
        state.last_emitted_target_at
    ):
        raise ValueError(
            "Residual rebase boundary must follow the publication barrier."
        )
    pending = state.pending_residual_rebase
    if pending is not None:
        if pending.semantic_content_hash == request.semantic_content_hash:
            return state
        raise ValueError("Another residual rebase request is already pending.")
    return replace(
        state,
        revision=state.revision + 1,
        pending_residual_rebase=request,
    )


def complete_residual_rebase(
    state: TrajectoryEditingState,
    request: ResidualRebaseRequest,
    receipt: ResidualAnchorReceipt,
    *,
    effective_for_target_at: float,
) -> TrajectoryEditingState:
    """Atomically activate one fresh anchor receipt at an uncommitted target."""

    if not isinstance(state, TrajectoryEditingState):
        raise TypeError("state must be a TrajectoryEditingState.")
    if not isinstance(request, ResidualRebaseRequest):
        raise TypeError("request must be ResidualRebaseRequest.")
    if not isinstance(receipt, ResidualAnchorReceipt):
        raise TypeError("receipt must be ResidualAnchorReceipt.")
    active_anchor = state.active_residual_anchor_binding
    active_trajectory = state.active_trajectory
    if active_anchor is None or active_trajectory is None:
        raise ValueError("No residual-active anchor is available for rebase.")
    pending = state.pending_residual_rebase
    if pending is None:
        if (
            active_anchor.rebase_request_hash == request.semantic_content_hash
            and active_anchor.residual_anchor_receipt.semantic_content_hash
            == receipt.semantic_content_hash
            and active_anchor.effective_for_target_at
            == _finite(
                effective_for_target_at,
                name="effective_for_target_at",
            )
        ):
            return state
        raise ValueError("No matching residual rebase request is pending.")
    if pending.semantic_content_hash != request.semantic_content_hash:
        raise ValueError("Residual rebase completion uses another request.")
    effective_at = _finite(
        effective_for_target_at,
        name="effective_for_target_at",
    )
    if active_trajectory.keyframe_at(effective_at) is None:
        raise ValueError(
            "Successor residual anchor boundary must be a trajectory keyframe."
        )
    if state.last_emitted_target_at is not None and effective_at <= (
        state.last_emitted_target_at
    ):
        raise ValueError(
            "Successor residual anchor must follow the publication barrier."
        )
    successor = build_successor_active_residual_anchor_binding(
        active_anchor,
        request,
        receipt,
        effective_for_target_at=effective_at,
    )
    return replace(
        state,
        revision=state.revision + 1,
        active_residual_anchor_binding=successor,
        pending_residual_rebase=None,
    )


def transition_trajectory_session_state(
    state: TrajectoryEditingState,
    *,
    expected_state_hash: str,
    interaction_mode: str,
    base_lifecycle_status: str,
) -> TrajectoryEditingState:
    """Apply one explicit, stale-checked session lifecycle transition.

    This reducer does not grant authority, request HOLD, activate a target, or
    replace a nominal base.  Returning to pass-through requires the later
    typed session-close protocol and is intentionally not implemented here.
    """

    if not isinstance(state, TrajectoryEditingState):
        raise TypeError("state must be a TrajectoryEditingState.")
    if state.interaction_mode == TRAJECTORY_INTERACTION_MODE_PASS_THROUGH:
        raise ValueError("No residual session is active.")
    if _required(expected_state_hash, name="expected_state_hash") != (
        state.semantic_content_hash
    ):
        raise ValueError("Lifecycle transition was built from stale state.")
    next_interaction = _required(
        interaction_mode,
        name="interaction_mode",
    )
    if next_interaction == TRAJECTORY_INTERACTION_MODE_PASS_THROUGH:
        raise ValueError(
            "Pass-through requires the typed session-close protocol."
        )
    next_lifecycle = _required(
        base_lifecycle_status,
        name="base_lifecycle_status",
    )
    validate_trajectory_interaction_base_state(
        next_interaction,
        next_lifecycle,
    )
    return replace(
        state,
        revision=state.revision + 1,
        interaction_mode=next_interaction,
        base_lifecycle_status=next_lifecycle,
    )


def prepare_trajectory_target(
    state: TrajectoryEditingState,
    *,
    target_effective_at: float,
) -> TrajectoryTargetPreparation:
    """Select one exact lattice target without advancing reducer state."""

    if not isinstance(state, TrajectoryEditingState):
        raise TypeError("state must be a TrajectoryEditingState.")
    target = _finite(
        target_effective_at,
        name="target_effective_at",
    )
    if state.base_lifecycle_status == TRAJECTORY_BASE_LIFECYCLE_EXHAUSTED:
        return TrajectoryTargetPreparation(
            status=TRAJECTORY_TARGET_STATUS_NO_ACTION,
            target_effective_at=target,
            publication_basis_hash=state.publication_basis_hash,
            reason="base_support_exhausted",
            active_trajectory_id=(
                None
                if state.active_trajectory is None
                else state.active_trajectory.trajectory_id
            ),
            active_trajectory_hash=(
                None
                if state.active_trajectory is None
                else state.active_trajectory.semantic_content_hash
            ),
        )
    active = state.active_trajectory
    if active is None:
        return TrajectoryTargetPreparation(
            status=TRAJECTORY_TARGET_STATUS_NO_ACTION,
            target_effective_at=target,
            publication_basis_hash=state.publication_basis_hash,
            reason="no_active_trajectory",
        )
    identity = {
        "active_trajectory_id": active.trajectory_id,
        "active_trajectory_hash": active.semantic_content_hash,
    }
    pending_rebase = state.pending_residual_rebase
    if (
        pending_rebase is not None
        and target >= pending_rebase.effective_not_before_target_at
    ):
        return TrajectoryTargetPreparation(
            status=TRAJECTORY_TARGET_STATUS_NO_ACTION,
            target_effective_at=target,
            publication_basis_hash=state.publication_basis_hash,
            reason="residual_rebase_pending",
            **identity,
        )
    if (
        state.last_emitted_target_at is not None
        and target <= state.last_emitted_target_at
    ):
        return TrajectoryTargetPreparation(
            status=TRAJECTORY_TARGET_STATUS_ALREADY_EMITTED,
            target_effective_at=target,
            publication_basis_hash=state.publication_basis_hash,
            reason="target_not_strictly_after_last_emission",
            **identity,
        )
    keyframe = active.keyframe_at(target)
    if keyframe is None:
        if target < active.effective_from:
            reason = "target_before_trajectory_support"
        elif target >= active.effective_until:
            reason = "trajectory_support_expired"
        else:
            reason = "target_not_on_trajectory_lattice"
        return TrajectoryTargetPreparation(
            status=TRAJECTORY_TARGET_STATUS_NO_ACTION,
            target_effective_at=target,
            publication_basis_hash=state.publication_basis_hash,
            reason=reason,
            **identity,
        )
    return TrajectoryTargetPreparation(
        status=TRAJECTORY_TARGET_STATUS_READY,
        target_effective_at=target,
        publication_basis_hash=state.publication_basis_hash,
        reason="exact_nominal_keyframe",
        keyframe=keyframe,
        action=keyframe.action,
        **identity,
    )


def commit_trajectory_target_emission(
    state: TrajectoryEditingState,
    preparation: TrajectoryTargetPreparation,
    *,
    emission_receipt_id: str,
    emission_receipt_hash: str,
    emitted_action_hash: str | None = None,
) -> TrajectoryEditingState:
    """Advance the publication barrier only after an external receipt.

    ``emitted_action_hash`` is required by corrected production paths because
    their formally published action can differ from the nominal preparation.
    The default retains the historical pure/pass-through behavior.
    """

    if not isinstance(state, TrajectoryEditingState):
        raise TypeError("state must be a TrajectoryEditingState.")
    if not isinstance(preparation, TrajectoryTargetPreparation):
        raise TypeError(
            "preparation must be a TrajectoryTargetPreparation."
        )
    if preparation.status != TRAJECTORY_TARGET_STATUS_READY:
        raise ValueError("Only a ready target can be committed.")
    if state.base_lifecycle_status == TRAJECTORY_BASE_LIFECYCLE_EXHAUSTED:
        raise ValueError("An exhausted nominal base cannot commit an emission.")
    if preparation.publication_basis_hash != state.publication_basis_hash:
        raise ValueError("Preparation was produced from another publication basis.")
    active = state.active_trajectory
    if active is None or preparation.active_trajectory_hash != (
        active.semantic_content_hash
    ):
        raise ValueError("Preparation no longer matches the active trajectory.")
    if preparation.action is None:
        raise ValueError("Ready preparation has no action.")
    if (
        state.last_emitted_target_at is not None
        and preparation.target_effective_at <= state.last_emitted_target_at
    ):
        raise ValueError("Target emission must be strictly monotonic.")
    resolved_action_hash = (
        preparation.action.semantic_content_hash
        if emitted_action_hash is None
        else _required(emitted_action_hash, name="emitted_action_hash")
    )
    return replace(
        state,
        revision=state.revision + 1,
        last_emitted_target_at=preparation.target_effective_at,
        last_emitted_action_hash=resolved_action_hash,
        last_emission_receipt_id=_required(
            emission_receipt_id,
            name="emission_receipt_id",
        ),
        last_emission_receipt_hash=_required(
            emission_receipt_hash,
            name="emission_receipt_hash",
        ),
    )


__all__ = [
    "activate_residual_session",
    "complete_residual_rebase",
    "commit_trajectory_target_emission",
    "empty_trajectory_editing_state",
    "offer_nominal_trajectory",
    "prepare_trajectory_target",
    "request_residual_rebase",
    "transition_trajectory_session_state",
]
