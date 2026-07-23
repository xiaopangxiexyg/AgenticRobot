"""Passive policy safety comparison against the current legacy owner.

This module is intentionally frontend-specific: it understands the existing
``DeploymentObservation`` and ``ActionSafetyFilter`` anchor behavior so it can
evaluate the shared canonical numeric primitive with exactly the same legacy
reference.  It owns no queue, anchor lifecycle, execution target, permit, or
backend and must remain safe to disable without changing legacy execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    CanonicalActionSafetyAnchor,
    CanonicalKeyframeSafetyProfile,
    SharedCanonicalSafetyResult,
    apply_canonical_keyframe_safety,
)
from robot_motion.action.proposal import robot_action_semantic_payload
from robot_motion.position_control import PositionKeyframeSafetyConfig
from robot_motion.deploy.runtime import PolicyActionEmission
from robot_motion.runtime.immutable import deep_freeze

from .actions import ActionSafetyFilter
from .arbiter_shadow import ActionComponentParityReceipt
from .canonical_action import CanonicalDeployActionResult
from .observation import DeploymentObservation
from .schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    STATE_ARM_POSE_SLICES,
    STATE_ARM_QPOS_SLICES,
    vector_or_none,
)


POLICY_SAFETY_COMPARISON_HASH_DOMAIN_V1 = (
    "dexteleop.policy_safety_shadow_comparison.v1"
)
POLICY_SAFETY_COMPARISON_MODE_LEGACY_ALIGNED = "legacy_aligned"
POLICY_SAFETY_NUMERIC_PARITY_HASH_DOMAIN_V1 = (
    "dexteleop.policy_safety_numeric_parity.v1"
)


@dataclass(frozen=True)
class PolicySafetyShadowComparison:
    comparison_id: str
    comparison_mode: str
    non_authoritative: bool
    emission_id: str
    arbitration_evaluation_id: str
    status: str
    mismatch_kind: str | None
    legacy_anchor_id: str | None
    legacy_anchor_kind: str | None
    legacy_anchor_hash: str | None
    legacy_faults: tuple[str, ...]
    legacy_action_present: bool
    shared_safety_result: SharedCanonicalSafetyResult | None
    component_parity: ActionComponentParityReceipt | None
    details: Mapping[str, Any]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if self.comparison_mode != POLICY_SAFETY_COMPARISON_MODE_LEGACY_ALIGNED:
            raise ValueError("Unsupported policy safety comparison mode.")
        if not self.non_authoritative:
            raise ValueError("Policy safety comparison must remain non-authoritative.")
        object.__setattr__(self, "legacy_faults", tuple(self.legacy_faults))
        object.__setattr__(self, "details", deep_freeze(dict(self.details)))

    def to_debug_payload(self) -> dict[str, Any]:
        safety = self.shared_safety_result
        parity = self.component_parity
        return {
            "comparison_id": self.comparison_id,
            "comparison_mode": self.comparison_mode,
            "non_authoritative": self.non_authoritative,
            "emission_id": self.emission_id,
            "arbitration_evaluation_id": self.arbitration_evaluation_id,
            "status": self.status,
            "mismatch_kind": self.mismatch_kind,
            "legacy_anchor_id": self.legacy_anchor_id,
            "legacy_anchor_kind": self.legacy_anchor_kind,
            "legacy_anchor_hash": self.legacy_anchor_hash,
            "legacy_faults": list(self.legacy_faults),
            "legacy_action_present": self.legacy_action_present,
            "shared_safety": (
                None
                if safety is None
                else {
                    "input_action_id": safety.input_action_id,
                    "status": safety.status,
                    "modified_components": list(safety.modified_components),
                    "reasons": list(safety.reasons),
                    "safety_contract_id": safety.safety_contract_id,
                    "safety_contract_hash": safety.safety_contract_hash,
                    "anchor_id": safety.anchor_id,
                    "anchor_kind": safety.anchor_kind,
                    "anchor_hash": safety.anchor_hash,
                    "safe_action_present": safety.safe_action is not None,
                }
            ),
            "component_parity": (
                None
                if parity is None
                else {
                    "matched": parity.matched,
                    "compared_components": list(parity.compared_components),
                    "mismatched_components": list(parity.mismatched_components),
                    "max_abs_error": parity.max_abs_error,
                    "absolute_tolerance": parity.absolute_tolerance,
                    "semantic_content_hash": parity.semantic_content_hash,
                }
            ),
            "details": dict(self.details),
            "semantic_content_hash": self.semantic_content_hash,
        }


def _applied_limits(
    safety_filter: ActionSafetyFilter,
    legacy_result: CanonicalDeployActionResult,
) -> PositionKeyframeSafetyConfig:
    metadata = dict(legacy_result.filtered.metadata)
    override = metadata.get("safety_overrides", {})
    applied = override.get("applied", {}) if isinstance(override, Mapping) else {}
    if not isinstance(applied, Mapping):
        applied = {}
    config = safety_filter.config
    return PositionKeyframeSafetyConfig(
        workspace_min=config.workspace_min,
        workspace_max=config.workspace_max,
        workspace_soft_margin_m=config.workspace_soft_margin_m,
        max_translation_step_m=float(
            applied.get("max_arm_xyz_step_m", config.max_arm_xyz_step_m)
        ),
        max_rotation_step_deg=float(
            applied.get("max_arm_rot_step_deg", config.max_arm_rot_step_deg)
        ),
        max_joint_step_rad=float(
            applied.get("max_arm_joint_step_rad", config.max_arm_joint_step_rad)
        ),
        joint_lower_rad=config.arm_joint_lower_rad,
        joint_upper_rad=config.arm_joint_upper_rad,
    )


def _legacy_aligned_anchor(
    emission: PolicyActionEmission,
    observation: DeploymentObservation,
) -> tuple[CanonicalActionSafetyAnchor | None, str | None]:
    action = emission.pre_safety_robot_action
    if action is None:
        return None, "pre_safety_action_unavailable"
    if action.hand_qpos_by_side or action.end_effector_targets_by_side:
        return None, "hand_or_end_effector_unsupported"
    if bool(action.arm_pose_by_side) == bool(action.arm_qpos_by_side):
        return None, "arm_modality_unsupported"

    if action.arm_qpos_by_side:
        qpos_by_side: dict[str, np.ndarray] = {}
        for side in action.arm_qpos_by_side:
            value = np.asarray(
                observation.state[STATE_ARM_QPOS_SLICES[side]], dtype=float
            ).reshape(6)
            if not np.all(np.isfinite(value)):
                return None, f"{side}_measured_qpos_unavailable"
            qpos_by_side[side] = value
        return (
            CanonicalActionSafetyAnchor(
                anchor_id=(
                    f"legacy-policy-measured-qpos:{emission.emission_id}:"
                    f"observation:{observation.step_index}"
                ),
                anchor_kind="measured_state",
                action_space=action.action_space,
                arm_qpos_by_side=qpos_by_side,
                observed_at=observation.timestamp,
            ),
            None,
        )

    # Match ActionSafetyFilter._reference_action_slice exactly.  The legacy
    # filter accepts last_action only when the *whole* deploy vector is finite;
    # sparse vectors containing NaNs therefore fall back to measured pose even
    # when the active side's pose slice itself is finite.
    last_action = vector_or_none(observation.last_action, length=ACTION_DIM)
    pose_by_side: dict[str, np.ndarray] = {}
    frame_by_side: dict[str, str] = {}
    kinds: set[str] = set()
    for side in action.arm_pose_by_side:
        if last_action is not None:
            last_pose = last_action[ACTION_ARM_POSE_SLICES[side]]
            pose = last_pose.copy()
            kinds.add("previous_safe")
        else:
            pose = np.asarray(
                observation.state[STATE_ARM_POSE_SLICES[side]], dtype=float
            ).reshape(7)
            if not np.all(np.isfinite(pose)):
                return None, f"{side}_measured_pose_unavailable"
            kinds.add("measured_state")
        pose_by_side[side] = pose
        frame = action.arm_frame(side)
        if frame is None or not str(frame).strip():
            return None, f"{side}_pose_frame_unavailable"
        frame_by_side[side] = str(frame)
    if len(kinds) != 1:
        return None, "mixed_legacy_pose_anchor_provenance"
    kind = next(iter(kinds))
    return (
        CanonicalActionSafetyAnchor(
            anchor_id=(
                f"legacy-policy-{kind}:{emission.emission_id}:"
                f"observation:{observation.step_index}"
            ),
            anchor_kind=kind,
            action_space=action.action_space,
            arm_pose_by_side=pose_by_side,
            frame_by_side=frame_by_side,
            observed_at=observation.timestamp,
        ),
        None,
    )


def _component_payload(action: Any, component: str) -> np.ndarray | None:
    side, kind = str(component).split("_", 1)
    if kind != "arm":
        return None
    value = action.arm_pose_by_side.get(side)
    if value is None:
        value = action.arm_qpos_by_side.get(side)
    return None if value is None else np.asarray(value, dtype=float).reshape(-1)


def _compare_numeric_action_components(
    expected: Any,
    observed: Any,
    *,
    atol: float,
) -> ActionComponentParityReceipt:
    """Compare safety-stage values without folding representation into it."""

    compared = tuple(
        dict.fromkeys((*expected.active_effectors, *observed.active_effectors))
    )
    mismatched: list[str] = []
    max_error = 0.0
    if expected.action_space != observed.action_space:
        mismatched.append("action_space")
    for component in compared:
        expected_value = _component_payload(expected, component)
        observed_value = _component_payload(observed, component)
        if (
            expected_value is None
            or observed_value is None
            or expected_value.shape != observed_value.shape
        ):
            mismatched.append(component)
            continue
        difference = float(
            np.max(np.abs(expected_value - observed_value), initial=0.0)
        )
        max_error = max(max_error, difference)
        if not np.allclose(
            expected_value,
            observed_value,
            rtol=0.0,
            atol=float(atol),
        ):
            mismatched.append(component)
    mismatched = list(dict.fromkeys(mismatched))
    payload = {
        "comparison_scope": "numeric_action_components_v1",
        "matched": not mismatched,
        "compared_components": compared,
        "mismatched_components": tuple(mismatched),
        "max_abs_error": max_error,
        "absolute_tolerance": float(atol),
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=POLICY_SAFETY_NUMERIC_PARITY_HASH_DOMAIN_V1,
    )
    return ActionComponentParityReceipt(
        matched=not mismatched,
        compared_components=compared,
        mismatched_components=tuple(mismatched),
        max_abs_error=max_error,
        absolute_tolerance=float(atol),
        semantic_content_hash=semantic_hash,
    )


def _representation_mismatches(expected: Any, observed: Any) -> tuple[str, ...]:
    mismatched: list[str] = []
    if expected.action_space != observed.action_space:
        mismatched.append("action_space")
    if tuple(expected.active_effectors) != tuple(observed.active_effectors):
        mismatched.append("active_effectors")
    components = tuple(
        dict.fromkeys((*expected.active_effectors, *observed.active_effectors))
    )
    for component in components:
        if expected.frame_by_effector.get(component) != observed.frame_by_effector.get(
            component
        ):
            mismatched.append(f"{component}:frame")
        if expected.controlled_frame_by_effector.get(
            component
        ) != observed.controlled_frame_by_effector.get(component):
            mismatched.append(f"{component}:controlled_frame")
    return tuple(dict.fromkeys(mismatched))


def _comparison(
    *,
    emission: PolicyActionEmission,
    arbitration_evaluation_id: str,
    status: str,
    mismatch_kind: str | None,
    anchor: CanonicalActionSafetyAnchor | None,
    legacy_result: CanonicalDeployActionResult,
    shared_result: SharedCanonicalSafetyResult | None,
    parity: ActionComponentParityReceipt | None,
    details: Mapping[str, Any],
) -> PolicySafetyShadowComparison:
    legacy_action_hash = (
        None
        if legacy_result.robot_action is None
        else canonical_content_hash(
            robot_action_semantic_payload(legacy_result.robot_action),
            domain="dexteleop.policy_safety_legacy_action.v1",
        )
    )
    shared_action_hash = (
        None
        if shared_result is None or shared_result.safe_action is None
        else canonical_content_hash(
            robot_action_semantic_payload(shared_result.safe_action),
            domain="dexteleop.policy_safety_shared_action.v1",
        )
    )
    payload = {
        "comparison_mode": POLICY_SAFETY_COMPARISON_MODE_LEGACY_ALIGNED,
        "non_authoritative": True,
        "emission_id": emission.emission_id,
        "arbitration_evaluation_id": arbitration_evaluation_id,
        "status": status,
        "mismatch_kind": mismatch_kind,
        "legacy_anchor_id": None if anchor is None else anchor.anchor_id,
        "legacy_anchor_kind": None if anchor is None else anchor.anchor_kind,
        "legacy_anchor_hash": (
            None if anchor is None else anchor.semantic_content_hash
        ),
        "legacy_faults": tuple(str(value) for value in legacy_result.faults),
        "legacy_action_present": legacy_result.robot_action is not None,
        "legacy_action_hash": legacy_action_hash,
        "shared_safety_status": None if shared_result is None else shared_result.status,
        "shared_safety_contract_hash": (
            None if shared_result is None else shared_result.safety_contract_hash
        ),
        "shared_anchor_hash": (
            None if shared_result is None else shared_result.anchor_hash
        ),
        "shared_action_hash": shared_action_hash,
        "component_parity_hash": None if parity is None else parity.semantic_content_hash,
        "details": dict(details),
    }
    semantic_hash = canonical_content_hash(
        payload, domain=POLICY_SAFETY_COMPARISON_HASH_DOMAIN_V1
    )
    return PolicySafetyShadowComparison(
        comparison_id=f"policy-safety-comparison:{semantic_hash}",
        comparison_mode=POLICY_SAFETY_COMPARISON_MODE_LEGACY_ALIGNED,
        non_authoritative=True,
        emission_id=emission.emission_id,
        arbitration_evaluation_id=arbitration_evaluation_id,
        status=status,
        mismatch_kind=mismatch_kind,
        legacy_anchor_id=None if anchor is None else anchor.anchor_id,
        legacy_anchor_kind=None if anchor is None else anchor.anchor_kind,
        legacy_anchor_hash=None if anchor is None else anchor.semantic_content_hash,
        legacy_faults=tuple(str(value) for value in legacy_result.faults),
        legacy_action_present=legacy_result.robot_action is not None,
        shared_safety_result=shared_result,
        component_parity=parity,
        details=dict(details),
        semantic_content_hash=semantic_hash,
    )


def evaluate_policy_legacy_aligned_safety(
    *,
    emission: PolicyActionEmission,
    arbitration_evaluation_id: str,
    observation: DeploymentObservation,
    legacy_result: CanonicalDeployActionResult,
    safety_filter: ActionSafetyFilter,
    atol: float = 1e-6,
) -> PolicySafetyShadowComparison:
    """Compare shared numeric safety using the exact current legacy anchor."""

    if not isinstance(emission, PolicyActionEmission):
        raise TypeError("emission must be a PolicyActionEmission.")
    if not str(arbitration_evaluation_id).strip():
        raise ValueError("arbitration_evaluation_id must be non-empty.")
    if not isinstance(observation, DeploymentObservation):
        raise TypeError("observation must be a DeploymentObservation.")
    if not isinstance(legacy_result, CanonicalDeployActionResult):
        raise TypeError("legacy_result must be a CanonicalDeployActionResult.")
    if not isinstance(safety_filter, ActionSafetyFilter):
        raise TypeError("safety_filter must be an ActionSafetyFilter.")
    action = emission.pre_safety_robot_action
    anchor, not_comparable_reason = _legacy_aligned_anchor(emission, observation)
    if action is None or anchor is None:
        return _comparison(
            emission=emission,
            arbitration_evaluation_id=arbitration_evaluation_id,
            status="not_comparable",
            mismatch_kind="unsupported_legacy_anchor",
            anchor=None,
            legacy_result=legacy_result,
            shared_result=None,
            parity=None,
            details={"reason": not_comparable_reason or "pre_safety_action_unavailable"},
        )

    profile = CanonicalKeyframeSafetyProfile(
        profile_id="policy-legacy-aligned-arm-keyframe-v1",
        limits=_applied_limits(safety_filter, legacy_result),
    )
    shared = apply_canonical_keyframe_safety(
        action,
        f"legacy-aligned:{emission.emission_id}",
        anchor=anchor,
        profile=profile,
    )
    legacy_blocked = bool(legacy_result.faults) or legacy_result.command is None
    shared_blocked = shared.safe_action is None or shared.status in {"hold", "fault"}
    parity = None
    if not legacy_blocked and not shared_blocked:
        if legacy_result.robot_action is None or shared.safe_action is None:
            status = "mismatch"
            mismatch_kind = "accepted_action_missing"
        else:
            parity = _compare_numeric_action_components(
                legacy_result.robot_action,
                shared.safe_action,
                atol=float(atol),
            )
            status = "matched" if parity.matched else "mismatch"
            mismatch_kind = None if parity.matched else "numeric_primitive"
    elif legacy_blocked == shared_blocked:
        status = "matched_blocked"
        mismatch_kind = None
    else:
        status = "mismatch"
        mismatch_kind = "acceptance_outcome"
    return _comparison(
        emission=emission,
        arbitration_evaluation_id=arbitration_evaluation_id,
        status=status,
        mismatch_kind=mismatch_kind,
        anchor=anchor,
        legacy_result=legacy_result,
        shared_result=shared,
        parity=parity,
        details={
            "legacy_blocked": legacy_blocked,
            "shared_blocked": shared_blocked,
            "profile_hash": profile.semantic_content_hash,
            "parity_scope": "numeric_action_components_v1",
            "representation_mismatches": (
                ()
                if legacy_result.robot_action is None or shared.safe_action is None
                else _representation_mismatches(
                    legacy_result.robot_action,
                    shared.safe_action,
                )
            ),
        },
    )


__all__ = [
    "POLICY_SAFETY_COMPARISON_MODE_LEGACY_ALIGNED",
    "PolicySafetyShadowComparison",
    "evaluate_policy_legacy_aligned_safety",
]
