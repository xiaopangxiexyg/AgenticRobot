"""Deterministic reducer for component-keyed composition allocation."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping

from .contracts import (
    COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
    COMPOSITION_ALLOCATION_STATUS_NO_DECISION,
    COMPOSITION_ALLOCATION_STATUS_READY,
    ComponentAllocationPolicy,
    ComponentResidualMagnitudeObservation,
    ComponentSourceActivityObservation,
    CompositionAllocationSpec,
)
from .decision import (
    ComponentAllocationSelection,
    CompositionAllocationDecision,
    CompositionAllocationEvaluation,
    CompositionAllocationState,
)
from .override import CompositionAllocationOverrideState


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


def empty_composition_allocation_state(
    reducer_id: str,
    *,
    spec: CompositionAllocationSpec,
) -> CompositionAllocationState:
    """Create an unclassified state whose explicit default is base-only."""

    if not isinstance(spec, CompositionAllocationSpec):
        raise TypeError("spec must be a CompositionAllocationSpec.")
    component_ids = spec.component_ids
    return CompositionAllocationState(
        reducer_id=_required(reducer_id, name="reducer_id"),
        revision=0,
        allocation_spec_hash=spec.semantic_content_hash,
        component_ids=component_ids,
        mode_by_component={
            component_id: COMPOSITION_ALLOCATION_MODE_BASE_ONLY
            for component_id in component_ids
        },
        mode_entered_at_by_component={
            component_id: None for component_id in component_ids
        },
    )


def _candidate_mode(
    policy: ComponentAllocationPolicy,
    *,
    current_mode: str,
    motion_activity_level: float,
    residual_magnitude_level: float,
) -> tuple[str, str]:
    motion = policy.motion_policy
    magnitude = policy.residual_magnitude_policy
    relative_enter = (
        motion_activity_level >= motion.relative_enter_level
        or residual_magnitude_level >= magnitude.relative_enter_level
    )
    relative_exit = (
        motion_activity_level <= motion.relative_exit_level
        and residual_magnitude_level <= magnitude.relative_exit_level
    )
    absolute_enter = (
        motion_activity_level >= motion.absolute_enter_level
        or residual_magnitude_level >= magnitude.absolute_enter_level
    )
    absolute_exit = (
        motion_activity_level <= motion.absolute_exit_level
        and residual_magnitude_level <= magnitude.absolute_exit_level
    )
    if current_mode == COMPOSITION_ALLOCATION_MODE_BASE_ONLY:
        if absolute_enter:
            candidate = (
                COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
                "evidence_crossed_absolute_enter",
            )
        elif relative_enter:
            candidate = (
                COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
                "evidence_crossed_relative_enter",
            )
        else:
            candidate = (current_mode, "evidence_below_relative_enter")
    elif current_mode == COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY:
        if absolute_enter:
            candidate = (
                COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
                "evidence_crossed_absolute_enter",
            )
        elif relative_exit:
            candidate = (
                COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
                "motion_and_magnitude_crossed_relative_exit",
            )
        else:
            candidate = (current_mode, "evidence_within_relative_hysteresis")
    elif current_mode == COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER:
        if relative_exit:
            candidate = (
                COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
                "motion_and_magnitude_crossed_relative_exit",
            )
        elif absolute_exit:
            candidate = (
                COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
                "motion_and_magnitude_crossed_absolute_exit",
            )
        else:
            candidate = (current_mode, "evidence_within_absolute_hysteresis")
    else:
        raise ValueError(f"Unsupported current allocation mode {current_mode!r}.")

    mode, reason = candidate
    rank = {
        COMPOSITION_ALLOCATION_MODE_BASE_ONLY: 0,
        COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY: 1,
        COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER: 2,
    }
    ceiling = policy.automatic_mode_ceiling
    if rank[mode] <= rank[ceiling]:
        return mode, reason
    return ceiling, f"{reason}:automatic_mode_ceiling"


def evaluate_composition_allocation(
    state: CompositionAllocationState,
    *,
    spec: CompositionAllocationSpec,
    activity_by_component: Mapping[
        str,
        ComponentSourceActivityObservation,
    ],
    residual_magnitude_by_component: Mapping[
        str,
        ComponentResidualMagnitudeObservation,
    ],
    evaluated_at: float,
    clock_domain_id: str,
    override_state: CompositionAllocationOverrideState | None = None,
) -> CompositionAllocationEvaluation:
    """Evaluate one atomic component set without changing authority.

    Missing, not-yet-available, or stale evidence produces ``no_decision`` and
    preserves reducer state.  Fresh zero motion is not sufficient for
    base-only: the anchor-relative magnitude must also be near identity.
    """

    if not isinstance(state, CompositionAllocationState):
        raise TypeError("state must be a CompositionAllocationState.")
    if not isinstance(spec, CompositionAllocationSpec):
        raise TypeError("spec must be a CompositionAllocationSpec.")
    if state.allocation_spec_hash != spec.semantic_content_hash:
        raise ValueError("Allocation state belongs to another spec.")
    if state.component_ids != spec.component_ids:
        raise ValueError("Allocation state component topology changed.")
    now = _finite(evaluated_at, name="evaluated_at")
    clock = _required(clock_domain_id, name="clock_domain_id")
    if override_state is not None:
        if not isinstance(override_state, CompositionAllocationOverrideState):
            raise TypeError("override_state must be typed or None.")
        if (
            override_state.allocation_spec_hash != spec.semantic_content_hash
            or override_state.component_ids != spec.component_ids
            or override_state.clock_domain_id != clock
        ):
            raise ValueError("Allocation override belongs to another spec/topology.")
    override_identity = (
        {}
        if override_state is None
        else {
            "allocation_override_state_id": override_state.state_id,
            "allocation_override_state_hash": (
                override_state.semantic_content_hash
            ),
        }
    )
    observations = dict(activity_by_component)
    magnitudes = dict(residual_magnitude_by_component)
    extra_motion = tuple(sorted(set(observations) - set(spec.component_ids)))
    extra_magnitude = tuple(sorted(set(magnitudes) - set(spec.component_ids)))
    if extra_motion or extra_magnitude:
        raise ValueError(
            "Allocation evidence contains undeclared components: "
            f"motion={extra_motion!r}, magnitude={extra_magnitude!r}."
        )

    unavailable: dict[str, str] = {}
    anchor_identities: set[tuple[str, str, int]] = set()
    for policy in spec.component_policies:
        component_id = policy.component_id
        observation = observations.get(component_id)
        if observation is None:
            unavailable[component_id] = "missing_motion_observation"
            continue
        if not isinstance(observation, ComponentSourceActivityObservation):
            raise TypeError(
                "activity_by_component values must be "
                "ComponentSourceActivityObservation."
            )
        if observation.component_id != component_id:
            raise ValueError("Activity observation component/key mismatch.")
        if observation.source_role != policy.motion_source_role:
            raise ValueError("Motion observation uses another declared source role.")
        if observation.clock_domain_id != clock:
            raise ValueError(
                "Activity observation and allocation evaluation must share "
                "one mapped clock domain."
            )
        if now < observation.available_at:
            unavailable[component_id] = "motion_not_yet_available"
        elif now >= observation.valid_until:
            unavailable[component_id] = "motion_observation_stale"
        magnitude = magnitudes.get(component_id)
        if magnitude is None:
            unavailable.setdefault(
                component_id,
                "missing_residual_magnitude_observation",
            )
            continue
        if not isinstance(magnitude, ComponentResidualMagnitudeObservation):
            raise TypeError(
                "residual_magnitude_by_component values must be "
                "ComponentResidualMagnitudeObservation."
            )
        if magnitude.component_id != component_id:
            raise ValueError("Residual magnitude observation component/key mismatch.")
        if magnitude.source_role != policy.residual_source_role:
            raise ValueError(
                "Residual magnitude observation uses another declared source role."
            )
        if magnitude.clock_domain_id != clock:
            raise ValueError(
                "Residual magnitude and allocation evaluation must share one "
                "mapped clock domain."
            )
        if now < magnitude.available_at:
            unavailable.setdefault(
                component_id,
                "residual_magnitude_not_yet_available",
            )
        elif now >= magnitude.valid_until:
            unavailable.setdefault(
                component_id,
                "residual_magnitude_observation_stale",
            )
        anchor_identities.add(
            (
                magnitude.anchor_receipt_id,
                magnitude.anchor_receipt_hash,
                magnitude.anchor_request_generation,
            )
        )

    if len(anchor_identities) > 1:
        raise ValueError(
            "Atomic allocation components must use one residual anchor receipt."
        )

    if unavailable:
        decision = CompositionAllocationDecision(
            allocation_spec_hash=spec.semantic_content_hash,
            previous_state_hash=state.semantic_content_hash,
            evaluated_at=now,
            clock_domain_id=clock,
            output_status=COMPOSITION_ALLOCATION_STATUS_NO_DECISION,
            reason="allocation_evidence_unavailable",
            unavailable_reason_by_component=unavailable,
            **override_identity,
        )
        return CompositionAllocationEvaluation(state=state, decision=decision)

    selections: list[ComponentAllocationSelection] = []
    next_modes = dict(state.mode_by_component)
    next_entered = dict(state.mode_entered_at_by_component)
    for policy in spec.component_policies:
        component_id = policy.component_id
        observation = observations[component_id]
        magnitude = magnitudes[component_id]
        current_mode = state.mode_by_component[component_id]
        forced_mode = (
            None
            if override_state is None
            else override_state.override_mode_by_component[component_id]
        )
        if forced_mode is None:
            candidate_mode, reason = _candidate_mode(
                policy,
                current_mode=current_mode,
                motion_activity_level=observation.activity_level,
                residual_magnitude_level=magnitude.magnitude_level,
            )
        else:
            candidate_mode = forced_mode
            reason = "persistent_allocation_override"
        entered_at = state.mode_entered_at_by_component[component_id]
        if entered_at is not None and now < entered_at:
            raise ValueError("Allocation evaluation time moved before mode entry.")
        if (
            forced_mode is None
            and
            candidate_mode != current_mode
            and entered_at is not None
            and now - entered_at < policy.minimum_dwell_sec
        ):
            candidate_mode = current_mode
            reason = "minimum_dwell_active"
        transitioned = candidate_mode != current_mode
        if transitioned:
            next_modes[component_id] = candidate_mode
            next_entered[component_id] = now
        selections.append(
            ComponentAllocationSelection(
                component_id=component_id,
                previous_mode=current_mode,
                selected_mode=candidate_mode,
                transitioned=transitioned,
                reason=reason,
                motion_observation_id=observation.observation_id,
                motion_observation_hash=observation.semantic_content_hash,
                motion_activity_level=observation.activity_level,
                residual_magnitude_observation_id=magnitude.observation_id,
                residual_magnitude_observation_hash=(
                    magnitude.semantic_content_hash
                ),
                residual_magnitude_level=magnitude.magnitude_level,
                anchor_receipt_id=magnitude.anchor_receipt_id,
                anchor_receipt_hash=magnitude.anchor_receipt_hash,
                anchor_request_generation=(
                    magnitude.anchor_request_generation
                ),
                valid_until=min(
                    observation.valid_until,
                    magnitude.valid_until,
                ),
            )
        )

    decision = CompositionAllocationDecision(
        allocation_spec_hash=spec.semantic_content_hash,
        previous_state_hash=state.semantic_content_hash,
        evaluated_at=now,
        clock_domain_id=clock,
        output_status=COMPOSITION_ALLOCATION_STATUS_READY,
        reason=(
            "allocation_evaluated_with_persistent_override"
            if override_state is not None
            and any(
                value is not None
                for value in override_state.override_mode_by_component.values()
            )
            else "activity_allocation_evaluated"
        ),
        selections=tuple(selections),
        **override_identity,
    )
    next_state = CompositionAllocationState(
        reducer_id=state.reducer_id,
        revision=state.revision + 1,
        allocation_spec_hash=state.allocation_spec_hash,
        component_ids=state.component_ids,
        mode_by_component=MappingProxyType(next_modes),
        mode_entered_at_by_component=MappingProxyType(next_entered),
        last_decision_hash=decision.semantic_content_hash,
    )
    return CompositionAllocationEvaluation(
        state=next_state,
        decision=decision,
    )


__all__ = [
    "empty_composition_allocation_state",
    "evaluate_composition_allocation",
]
