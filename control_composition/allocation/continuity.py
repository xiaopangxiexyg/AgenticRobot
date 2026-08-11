"""Pure reducer and activation-local runtime for allocation target seams."""

from __future__ import annotations

import math
import threading
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.action.component_action import ComponentAction

from .continuity_contracts import (
    ALLOCATION_CONTINUITY_CORRECTION_DECAYING_HANDOFF,
    ALLOCATION_CONTINUITY_CORRECTION_NONE,
    ALLOCATION_CONTINUITY_CORRECTION_PERSISTENT_ABSOLUTE_ALIGNMENT,
    AllocationComponentContinuityState,
    AllocationTransitionContinuityEvaluation,
    AllocationTransitionContinuityPolicy,
    AllocationTransitionContinuityReceipt,
    AllocationTransitionContinuityState,
    _finite,
    _mode_map,
    _required,
    empty_allocation_transition_continuity_state,
)
from .contracts import COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER


def _canonical_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("Pose continuity requires a finite non-zero quaternion.")
    quaternion = quaternion / norm
    for coordinate in quaternion[::-1]:
        if abs(float(coordinate)) <= 1e-15:
            continue
        if coordinate < 0.0:
            quaternion = -quaternion
        break
    return quaternion


def _alignment(
    previous: np.ndarray,
    raw: np.ndarray,
    *,
    representation_family: str,
) -> tuple[float, ...]:
    if representation_family == "joint_position":
        if previous.shape != raw.shape:
            raise ValueError("Joint-position continuity shape changed.")
        return tuple(float(value) for value in previous - raw)
    if (
        representation_family != "cartesian_pose"
        or previous.shape != (7,)
        or raw.shape != (7,)
    ):
        raise ValueError(
            "Allocation continuity supports cartesian_pose and joint_position."
        )
    translation = previous[:3] - raw[:3]
    rotation = (
        Rotation.from_quat(_canonical_quaternion(previous[3:7]))
        * Rotation.from_quat(_canonical_quaternion(raw[3:7])).inv()
    ).as_quat()
    return tuple(
        float(value)
        for value in np.concatenate((translation, _canonical_quaternion(rotation)))
    )


def _apply_alignment(
    raw: np.ndarray,
    correction: tuple[float, ...],
    *,
    representation_family: str,
    factor: float,
    dtype: str,
) -> np.ndarray:
    if representation_family == "joint_position":
        offset = np.asarray(correction, dtype=np.float64)
        if offset.shape != raw.shape:
            raise ValueError("Joint-position correction shape changed.")
        return (raw + factor * offset).astype(dtype)
    if representation_family != "cartesian_pose" or raw.shape != (7,):
        raise ValueError(
            "Allocation continuity supports cartesian_pose and joint_position."
        )
    payload = np.asarray(correction, dtype=np.float64)
    if payload.shape != (7,):
        raise ValueError("Pose correction must contain translation and quaternion.")
    offset_rotation = Rotation.from_quat(_canonical_quaternion(payload[3:7]))
    scaled_rotation = Rotation.from_rotvec(factor * offset_rotation.as_rotvec())
    raw_rotation = Rotation.from_quat(_canonical_quaternion(raw[3:7]))
    quaternion = _canonical_quaternion((scaled_rotation * raw_rotation).as_quat())
    return np.concatenate(
        (raw[:3] + factor * payload[:3], quaternion)
    ).astype(dtype)


def _validate_actions(
    raw_candidate: ComponentAction,
    baseline_candidate: ComponentAction,
    policy: AllocationTransitionContinuityPolicy,
) -> None:
    for action, name in (
        (raw_candidate, "raw_candidate"),
        (baseline_candidate, "baseline_candidate"),
    ):
        if not isinstance(action, ComponentAction):
            raise TypeError(f"{name} must be a ComponentAction.")
    if raw_candidate.action_schema.semantic_content_hash != (
        baseline_candidate.action_schema.semantic_content_hash
    ):
        raise ValueError("Raw and baseline continuity actions use another schema.")
    if tuple(raw_candidate.component_coverage) != policy.component_ids or tuple(
        baseline_candidate.component_coverage
    ) != policy.component_ids:
        raise ValueError("Continuity actions must cover the policy topology exactly.")


def _correction_for_component(
    previous_state: AllocationComponentContinuityState,
    *,
    selected_mode: str,
    seam_previous: np.ndarray,
    raw: np.ndarray,
    representation_family: str,
    target_effective_at: float,
    handoff_duration_sec: float,
) -> tuple[AllocationComponentContinuityState, float, bool]:
    transitioned = previous_state.selected_mode != selected_mode
    if transitioned:
        correction = _alignment(
            seam_previous,
            raw,
            representation_family=representation_family,
        )
        if selected_mode == COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER:
            return (
                AllocationComponentContinuityState(
                    component_id=previous_state.component_id,
                    selected_mode=selected_mode,
                    correction_kind=(
                        ALLOCATION_CONTINUITY_CORRECTION_PERSISTENT_ABSOLUTE_ALIGNMENT
                    ),
                    correction_values=correction,
                    correction_started_at=target_effective_at,
                ),
                1.0,
                True,
            )
        return (
            AllocationComponentContinuityState(
                component_id=previous_state.component_id,
                selected_mode=selected_mode,
                correction_kind=ALLOCATION_CONTINUITY_CORRECTION_DECAYING_HANDOFF,
                correction_values=correction,
                correction_started_at=target_effective_at,
                correction_ends_at=(
                    target_effective_at + handoff_duration_sec
                ),
            ),
            1.0,
            True,
        )

    if previous_state.correction_kind == (
        ALLOCATION_CONTINUITY_CORRECTION_PERSISTENT_ABSOLUTE_ALIGNMENT
    ):
        return previous_state, 1.0, False
    if previous_state.correction_kind != (
        ALLOCATION_CONTINUITY_CORRECTION_DECAYING_HANDOFF
    ):
        return previous_state, 0.0, False
    assert previous_state.correction_started_at is not None
    assert previous_state.correction_ends_at is not None
    remaining = previous_state.correction_ends_at - target_effective_at
    if remaining <= 0.0:
        return (
            AllocationComponentContinuityState(
                component_id=previous_state.component_id,
                selected_mode=selected_mode,
                correction_kind=ALLOCATION_CONTINUITY_CORRECTION_NONE,
            ),
            0.0,
            False,
        )
    factor = remaining / (
        previous_state.correction_ends_at
        - previous_state.correction_started_at
    )
    return previous_state, factor, False


def evaluate_allocation_transition_continuity(
    state: AllocationTransitionContinuityState,
    *,
    policy: AllocationTransitionContinuityPolicy,
    raw_candidate: ComponentAction,
    baseline_candidate: ComponentAction,
    selected_mode_by_component: Mapping[str, str],
    target_effective_at: float,
    invocation_id: str,
    invocation_hash: str,
) -> AllocationTransitionContinuityEvaluation:
    """Advance one unique target using target time, never wall-clock time."""

    if not isinstance(state, AllocationTransitionContinuityState):
        raise TypeError("state must be typed.")
    if not isinstance(policy, AllocationTransitionContinuityPolicy):
        raise TypeError("policy must be typed.")
    if state.policy_hash != policy.semantic_content_hash:
        raise ValueError("Continuity state belongs to another policy.")
    _validate_actions(raw_candidate, baseline_candidate, policy)
    selected = _mode_map(
        selected_mode_by_component,
        component_ids=policy.component_ids,
    )
    if any(value is None for value in selected.values()):
        raise ValueError("Selected continuity modes cannot be None.")
    target = _finite(target_effective_at, name="target_effective_at")
    normalized_invocation_id = _required(invocation_id, name="invocation_id")
    normalized_invocation_hash = _required(invocation_hash, name="invocation_hash")
    if state.last_invocation_id == normalized_invocation_id:
        if state.last_invocation_hash != normalized_invocation_hash:
            raise ValueError("Continuity invocation ID was reused with another hash.")
        raise ValueError(
            "Use AllocationTransitionContinuityRuntime for idempotent duplicates."
        )
    if state.last_target_effective_at is not None and target <= (
        state.last_target_effective_at
    ):
        raise ValueError("Continuity target time must increase strictly.")

    previous_states = state.component_state_by_id
    previous_modes = {
        component_id: previous_states[component_id].selected_mode
        for component_id in policy.component_ids
    }
    output_values: dict[str, np.ndarray] = {}
    next_states: list[AllocationComponentContinuityState] = []
    transitions: dict[str, str] = {}
    correction_kinds: dict[str, str] = {}
    for component_id in policy.component_ids:
        payload_schema = raw_candidate.action_schema.payload_schema_by_component[
            component_id
        ]
        raw = np.asarray(
            raw_candidate.value(component_id).values,
            dtype=np.float64,
        )
        seam_previous = np.asarray(
            (
                baseline_candidate
                if state.last_output is None
                else state.last_output
            ).value(component_id).values,
            dtype=np.float64,
        )
        mode = str(selected[component_id])
        component_state, factor, transitioned = _correction_for_component(
            previous_states[component_id],
            selected_mode=mode,
            seam_previous=seam_previous,
            raw=raw,
            representation_family=payload_schema.representation_family,
            target_effective_at=target,
            handoff_duration_sec=policy.handoff_duration_sec,
        )
        output_values[component_id] = (
            raw.astype(payload_schema.dtype)
            if component_state.correction_kind
            == ALLOCATION_CONTINUITY_CORRECTION_NONE
            else _apply_alignment(
                raw,
                component_state.correction_values,
                representation_family=payload_schema.representation_family,
                factor=factor,
                dtype=payload_schema.dtype,
            )
        )
        next_states.append(component_state)
        transitions[component_id] = (
            f"{previous_states[component_id].selected_mode or 'uninitialized'}"
            f"->{mode}"
            if transitioned
            else "unchanged"
        )
        correction_kinds[component_id] = component_state.correction_kind

    output = ComponentAction(
        action_schema=raw_candidate.action_schema,
        values_by_component=output_values,
    )
    next_state = AllocationTransitionContinuityState(
        policy_hash=state.policy_hash,
        composition_activation_id=state.composition_activation_id,
        composition_activation_hash=state.composition_activation_hash,
        revision=state.revision + 1,
        component_states=tuple(next_states),
        last_target_effective_at=target,
        last_output=output,
        last_invocation_id=normalized_invocation_id,
        last_invocation_hash=normalized_invocation_hash,
    )
    receipt = AllocationTransitionContinuityReceipt(
        policy_hash=state.policy_hash,
        composition_activation_id=state.composition_activation_id,
        composition_activation_hash=state.composition_activation_hash,
        invocation_id=normalized_invocation_id,
        invocation_hash=normalized_invocation_hash,
        target_effective_at=target,
        state_before_hash=state.semantic_content_hash,
        state_after_hash=next_state.semantic_content_hash,
        raw_candidate_hash=raw_candidate.semantic_content_hash,
        output_candidate_hash=output.semantic_content_hash,
        previous_mode_by_component=previous_modes,
        selected_mode_by_component={
            component_id: str(selected[component_id])
            for component_id in policy.component_ids
        },
        transition_by_component=transitions,
        correction_kind_by_component=correction_kinds,
    )
    return AllocationTransitionContinuityEvaluation(
        state=next_state,
        output=output,
        receipt=receipt,
    )


class AllocationTransitionContinuityRuntime:
    """Activation-scoped idempotent owner for the pure continuity reducer."""

    def __init__(self, policy: AllocationTransitionContinuityPolicy) -> None:
        if not isinstance(policy, AllocationTransitionContinuityPolicy):
            raise TypeError("policy must be typed.")
        self.policy = policy
        self._lock = threading.RLock()
        self._state: AllocationTransitionContinuityState | None = None
        self._last_evaluation: AllocationTransitionContinuityEvaluation | None = None
        self._prepared_evaluation: AllocationTransitionContinuityEvaluation | None = None

    def activate(
        self,
        *,
        composition_activation_id: str,
        composition_activation_hash: str,
    ) -> AllocationTransitionContinuityState:
        with self._lock:
            self._state = empty_allocation_transition_continuity_state(
                self.policy,
                composition_activation_id=composition_activation_id,
                composition_activation_hash=composition_activation_hash,
            )
            self._last_evaluation = None
            self._prepared_evaluation = None
            return self._state

    def prepare(self, **kwargs) -> AllocationTransitionContinuityEvaluation:
        """Prepare one candidate without advancing the published-target state."""

        with self._lock:
            if self._state is None:
                raise RuntimeError("Continuity runtime has no active composition.")
            invocation_id = _required(kwargs.get("invocation_id"), name="invocation_id")
            invocation_hash = _required(
                kwargs.get("invocation_hash"),
                name="invocation_hash",
            )
            if self._state.last_invocation_id == invocation_id:
                if self._state.last_invocation_hash != invocation_hash:
                    raise ValueError(
                        "Continuity invocation ID was reused with another hash."
                    )
                assert self._last_evaluation is not None
                return self._last_evaluation
            prepared = self._prepared_evaluation
            if prepared is not None and prepared.receipt.invocation_id == invocation_id:
                if prepared.receipt.invocation_hash != invocation_hash:
                    raise ValueError(
                        "Continuity invocation ID was reused with another hash."
                    )
                return prepared
            result = evaluate_allocation_transition_continuity(
                self._state,
                policy=self.policy,
                **kwargs,
            )
            # One runtime has one current publication candidate. A later formal
            # evaluation supersedes an uncommitted one; neither affects the
            # published-target baseline until exact proposal admission succeeds.
            self._prepared_evaluation = result
            return result

    def validate_prepared(self, *, receipt_hash: str) -> None:
        """Fail before external admission if the candidate is stale or unknown."""

        expected = _required(receipt_hash, name="receipt_hash")
        with self._lock:
            if self._state is None:
                raise RuntimeError("Continuity runtime has no active composition.")
            prepared = self._prepared_evaluation
            if prepared is None or prepared.receipt.semantic_content_hash != expected:
                raise ValueError("Continuity publication is not the current preparation.")
            if prepared.receipt.state_before_hash != self._state.semantic_content_hash:
                raise ValueError("Continuity preparation starts from stale state.")

    def commit_prepared(
        self,
        *,
        receipt_hash: str,
    ) -> AllocationTransitionContinuityEvaluation:
        """Commit the exact candidate only after proposal admission accepted it."""

        expected = _required(receipt_hash, name="receipt_hash")
        with self._lock:
            if (
                self._last_evaluation is not None
                and self._last_evaluation.receipt.semantic_content_hash == expected
            ):
                return self._last_evaluation
            self.validate_prepared(receipt_hash=expected)
            assert self._prepared_evaluation is not None
            result = self._prepared_evaluation
            self._state = result.state
            self._last_evaluation = result
            self._prepared_evaluation = None
            return result

    def evaluate(self, **kwargs) -> AllocationTransitionContinuityEvaluation:
        """Prepare and immediately commit for standalone reducer users/tests."""

        result = self.prepare(**kwargs)
        return self.commit_prepared(
            receipt_hash=result.receipt.semantic_content_hash,
        )

    @property
    def state(self) -> AllocationTransitionContinuityState | None:
        with self._lock:
            return self._state


__all__ = [
    "AllocationTransitionContinuityRuntime",
    "evaluate_allocation_transition_continuity",
]
