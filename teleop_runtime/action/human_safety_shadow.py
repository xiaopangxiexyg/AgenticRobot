"""Typed observe-only evidence for the canonical GUI's legacy arm limiter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_QPOS_V1,
    CanonicalActionSafetyAnchor,
    CanonicalKeyframeSafetyProfile,
    RobotAction,
    SharedCanonicalSafetyResult,
    apply_canonical_keyframe_safety,
    split_effector_name,
)
from robot_motion.action.proposal import robot_action_semantic_payload
from robot_motion.position_control import PositionKeyframeSafetyConfig
from teleop_runtime.real_teleop.temporary_dual_arm_safety import (
    TemporaryDualArmSafetyEvaluation,
)

from .adapters import matrix_to_xyzquat
from .human_proposal import HumanCommandEmission, HumanCommandFilterObservation


HUMAN_LEGACY_SAFETY_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.human_legacy_safety_binding.v1"
)
HUMAN_LEGACY_SAFETY_COMPARISON_HASH_DOMAIN_V1 = (
    "dexteleop.human_legacy_safety_comparison.v1"
)


@dataclass(frozen=True)
class HumanLegacySafetyBindingEvidence:
    """Bind exact legacy limiter evidence to one immutable human emission.

    ``comparable`` only means the legacy stage is isolated well enough for a
    later numeric dual-run.  It never means the filtered command was accepted
    by the command guard, execution target, or backend.
    """

    evidence_id: str
    emission_sequence: int
    source_emission_semantic_hash: str
    filter_observed_at: float
    limiter_evaluation_id: str
    limiter_evaluation_hash: str
    input_matches_emission: bool
    output_matches_post_filter: bool
    previous_anchor_available: bool
    comparable: bool
    reasons: tuple[str, ...]
    non_authoritative: bool
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if not self.non_authoritative:
            raise ValueError("Human legacy safety evidence must remain non-authoritative.")
        object.__setattr__(self, "reasons", tuple(str(value) for value in self.reasons))


@dataclass(frozen=True)
class HumanLegacySafetyComparison:
    comparison_id: str
    binding_evidence_id: str
    arbitration_evaluation_id: str
    status: str
    mismatch_kind: str | None
    anchor_id: str | None
    shared_safety_result: SharedCanonicalSafetyResult | None
    mismatched_components: tuple[str, ...]
    non_authoritative: bool
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if self.status not in {"matched", "mismatch", "not_comparable"}:
            raise ValueError("Unsupported human legacy safety comparison status.")
        if not self.non_authoritative:
            raise ValueError("Human safety comparison must remain non-authoritative.")
        object.__setattr__(
            self,
            "mismatched_components",
            tuple(str(value) for value in self.mismatched_components),
        )


def _snapshot_values(snapshot: object) -> tuple[tuple[int, ...] | None, tuple[float, ...] | None]:
    try:
        array = np.asarray(getattr(snapshot, "value"), dtype=float)
    except (AttributeError, TypeError, ValueError):
        return None, None
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None, None
    return tuple(int(item) for item in array.shape), tuple(
        float(item) for item in array.reshape(-1)
    )


def _matches(
    shape: tuple[int, ...] | None,
    values: tuple[float, ...] | None,
    snapshot: object | None,
) -> bool:
    if shape is None or values is None or snapshot is None:
        return False
    other_shape, other_values = _snapshot_values(snapshot)
    if other_shape != shape or other_values is None:
        return False
    return bool(
        np.array_equal(
            np.asarray(values, dtype=float),
            np.asarray(other_values, dtype=float),
        )
    )


def bind_human_legacy_safety_evidence(
    *,
    emission: HumanCommandEmission,
    filter_observation: HumanCommandFilterObservation,
    limiter_evaluation: TemporaryDualArmSafetyEvaluation,
) -> HumanLegacySafetyBindingEvidence:
    """Join one limiter receipt without inferring an anchor from final commands."""

    if filter_observation.emission_sequence != emission.emission_sequence:
        raise ValueError("Human filter observation references another emission.")
    source_by_effector = {
        item.effector_name: item for item in emission.merged_commands
    }
    filtered_by_effector = {
        item.effector_name: item for item in filter_observation.filtered_commands
    }
    stages = tuple(limiter_evaluation.stages)
    reasons: list[str] = []
    if not limiter_evaluation.enabled:
        reasons.append("legacy_temporary_limiter_disabled")
    if not stages:
        reasons.append("legacy_arm_stage_absent")

    input_matches = bool(stages) and all(
        _matches(
            stage.input_shape,
            stage.input_values,
            source_by_effector.get(stage.effector_name),
        )
        for stage in stages
    )
    output_matches = bool(stages) and all(
        _matches(
            stage.output_shape,
            stage.output_values,
            filtered_by_effector.get(stage.effector_name),
        )
        for stage in stages
    )
    previous_available = bool(stages) and all(
        stage.previous_anchor_shape is not None
        and stage.previous_anchor_values is not None
        for stage in stages
    )
    if not input_matches:
        reasons.append("pre_limiter_command_differs_from_human_emission")
    if not output_matches:
        reasons.append("post_limiter_command_modified_by_later_filter")
    if not previous_available:
        reasons.append("legacy_previous_anchor_unavailable")
    if not np.isclose(float(limiter_evaluation.command_scale), 1.0):
        reasons.append("legacy_command_scale_not_represented_by_shared_keyframe_safety")
    unsupported = tuple(
        stage.effector_name
        for stage in stages
        if stage.command_type not in {"cartesian_pose", "joint_vector"}
    )
    if unsupported:
        reasons.append("unsupported_legacy_arm_command_type:" + ",".join(unsupported))

    comparable = not reasons
    payload = {
        "emission_sequence": emission.emission_sequence,
        "source_emission_semantic_hash": emission.semantic_content_hash,
        "filter_observed_at": filter_observation.observed_at,
        "limiter_evaluation_id": limiter_evaluation.evaluation_id,
        "limiter_evaluation_hash": limiter_evaluation.semantic_content_hash,
        "input_matches_emission": input_matches,
        "output_matches_post_filter": output_matches,
        "previous_anchor_available": previous_available,
        "comparable": comparable,
        "reasons": tuple(reasons),
        "non_authoritative": True,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=HUMAN_LEGACY_SAFETY_BINDING_HASH_DOMAIN_V1,
    )
    return HumanLegacySafetyBindingEvidence(
        evidence_id=f"human-legacy-safety-binding:{semantic_hash}",
        **payload,
        semantic_content_hash=semantic_hash,
    )


def _legacy_anchor(
    action: RobotAction,
    evaluation: TemporaryDualArmSafetyEvaluation,
) -> CanonicalActionSafetyAnchor:
    if action.hand_qpos_by_side or action.end_effector_targets_by_side:
        raise ValueError("Human arm safety comparison does not cover hand components.")
    poses: dict[str, np.ndarray] = {}
    qpos: dict[str, np.ndarray] = {}
    frames: dict[str, str] = {}
    stage_by_effector = {stage.effector_name: stage for stage in evaluation.stages}
    if set(stage_by_effector) != set(action.active_effectors):
        raise ValueError("Legacy limiter stages do not match active action effectors.")
    for effector in action.active_effectors:
        side, kind = split_effector_name(effector)
        if kind != "arm":
            raise ValueError("Human legacy safety comparison is arm-only.")
        stage = stage_by_effector[effector]
        if stage.previous_anchor_shape is None or stage.previous_anchor_values is None:
            raise ValueError("Legacy previous-command anchor is unavailable.")
        previous = np.asarray(stage.previous_anchor_values, dtype=float).reshape(
            stage.previous_anchor_shape
        )
        if side in action.arm_pose_by_side:
            if stage.command_type != "cartesian_pose" or previous.shape != (4, 4):
                raise ValueError("Legacy pose anchor representation mismatch.")
            poses[side] = matrix_to_xyzquat(previous)
            frame = action.arm_frame(side)
            if frame is None or not str(frame).strip():
                raise ValueError("Human pose action lacks an explicit reference frame.")
            frames[side] = str(frame)
        elif side in action.arm_qpos_by_side:
            if stage.command_type != "joint_vector" or previous.shape != (6,):
                raise ValueError("Legacy qpos anchor representation mismatch.")
            qpos[side] = previous.copy()
        else:
            raise ValueError("Legacy limiter stage lacks matching action payload.")
    return CanonicalActionSafetyAnchor(
        anchor_id=f"legacy-human-previous-command:{evaluation.evaluation_id}",
        anchor_kind="previous_commanded",
        action_space=action.action_space,
        arm_pose_by_side=poses,
        arm_qpos_by_side=qpos,
        frame_by_side=frames,
        observed_at=evaluation.evaluated_at,
    )


def _component_value(action: RobotAction, effector: str) -> np.ndarray | None:
    side, kind = split_effector_name(effector)
    if kind != "arm":
        return None
    value = action.arm_pose_by_side.get(side)
    if value is None:
        value = action.arm_qpos_by_side.get(side)
    return None if value is None else np.asarray(value, dtype=float).reshape(-1)


def evaluate_human_legacy_aligned_safety(
    *,
    source_action: RobotAction,
    legacy_filtered_action: RobotAction,
    binding_evidence: HumanLegacySafetyBindingEvidence,
    limiter_evaluation: TemporaryDualArmSafetyEvaluation,
    arbitration_evaluation_id: str,
    profile: CanonicalKeyframeSafetyProfile,
    atol: float = 1e-6,
) -> HumanLegacySafetyComparison:
    """Run shared numeric safety against the exact legacy previous command."""

    if not str(arbitration_evaluation_id).strip():
        raise ValueError("arbitration_evaluation_id must be non-empty.")
    if binding_evidence.limiter_evaluation_id != limiter_evaluation.evaluation_id:
        raise ValueError("Binding evidence references another limiter evaluation.")
    shared: SharedCanonicalSafetyResult | None = None
    anchor: CanonicalActionSafetyAnchor | None = None
    mismatched: list[str] = []
    mismatch_kind: str | None = None
    status = "not_comparable"
    if binding_evidence.comparable:
        try:
            anchor = _legacy_anchor(source_action, limiter_evaluation)
            shared = apply_canonical_keyframe_safety(
                source_action,
                f"human-legacy-aligned:{binding_evidence.evidence_id}",
                anchor=anchor,
                profile=profile,
            )
            if shared.safe_action is None or shared.status in {"hold", "fault"}:
                status = "mismatch"
                mismatch_kind = "shared_safety_blocked_legacy_output"
            else:
                components = tuple(
                    dict.fromkeys(
                        (*shared.safe_action.active_effectors, *legacy_filtered_action.active_effectors)
                    )
                )
                if shared.safe_action.action_space != legacy_filtered_action.action_space:
                    mismatched.append("action_space")
                for component in components:
                    expected = _component_value(shared.safe_action, component)
                    observed = _component_value(legacy_filtered_action, component)
                    if (
                        expected is None
                        or observed is None
                        or expected.shape != observed.shape
                        or not np.allclose(expected, observed, rtol=0.0, atol=float(atol))
                    ):
                        mismatched.append(component)
                status = "matched" if not mismatched else "mismatch"
                mismatch_kind = None if not mismatched else "numeric_primitive"
        except (TypeError, ValueError):
            status = "not_comparable"
            mismatch_kind = "legacy_anchor_representation"
    payload = {
        "binding_evidence_id": binding_evidence.evidence_id,
        "arbitration_evaluation_id": arbitration_evaluation_id,
        "status": status,
        "mismatch_kind": mismatch_kind,
        "anchor_id": None if anchor is None else anchor.anchor_id,
        "source_action": robot_action_semantic_payload(source_action),
        "legacy_filtered_action": robot_action_semantic_payload(legacy_filtered_action),
        "shared_safe_action": (
            None
            if shared is None or shared.safe_action is None
            else robot_action_semantic_payload(shared.safe_action)
        ),
        "profile_hash": profile.semantic_content_hash,
        "mismatched_components": tuple(mismatched),
        "non_authoritative": True,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=HUMAN_LEGACY_SAFETY_COMPARISON_HASH_DOMAIN_V1,
    )
    return HumanLegacySafetyComparison(
        comparison_id=f"human-legacy-safety-comparison:{semantic_hash}",
        binding_evidence_id=binding_evidence.evidence_id,
        arbitration_evaluation_id=arbitration_evaluation_id,
        status=status,
        mismatch_kind=mismatch_kind,
        anchor_id=None if anchor is None else anchor.anchor_id,
        shared_safety_result=shared,
        mismatched_components=tuple(mismatched),
        non_authoritative=True,
        semantic_content_hash=semantic_hash,
    )


def build_human_legacy_aligned_safety_profile(
    *,
    source_action: RobotAction,
    limiter_evaluation: TemporaryDualArmSafetyEvaluation,
) -> CanonicalKeyframeSafetyProfile:
    """Translate only the numeric bounds actually owned by the legacy limiter.

    This helper intentionally disables shared workspace bounds because the
    temporary GUI limiter does not own a workspace envelope.  Collision,
    resolver, reference, backend, and measured-divergence rules remain outside
    this narrow legacy-aligned profile.
    """

    if source_action.hand_qpos_by_side or source_action.end_effector_targets_by_side:
        raise ValueError("Human legacy-aligned profile is arm-only.")
    stage_types = {stage.command_type for stage in limiter_evaluation.stages}
    if source_action.action_space == BIMANUAL_EE_POSE_V1:
        if not source_action.arm_pose_by_side or source_action.arm_qpos_by_side:
            raise ValueError("Pose profile requires a pure arm-pose action.")
        if stage_types and stage_types != {"cartesian_pose"}:
            raise ValueError("Legacy limiter evidence is not pose-only.")
        return CanonicalKeyframeSafetyProfile(
            profile_id="human-temporary-legacy-aligned-pose-v1",
            limits=PositionKeyframeSafetyConfig(
                workspace_enabled=False,
                max_translation_step_m=float(limiter_evaluation.max_xyz_step_m),
                max_rotation_step_deg=float(limiter_evaluation.max_rot_step_deg),
            ),
        )
    if source_action.action_space == BIMANUAL_QPOS_V1:
        if not source_action.arm_qpos_by_side or source_action.arm_pose_by_side:
            raise ValueError("Qpos profile requires a pure arm-qpos action.")
        if stage_types and stage_types != {"joint_vector"}:
            raise ValueError("Legacy limiter evidence is not qpos-only.")
        return CanonicalKeyframeSafetyProfile(
            profile_id="human-temporary-legacy-aligned-qpos-v1",
            limits=PositionKeyframeSafetyConfig(
                workspace_enabled=False,
                max_joint_step_rad=float(
                    np.deg2rad(limiter_evaluation.max_joint_step_deg)
                ),
            ),
        )
    raise ValueError(
        f"Unsupported human legacy-aligned action space {source_action.action_space!r}."
    )


__all__ = [
    "HUMAN_LEGACY_SAFETY_BINDING_HASH_DOMAIN_V1",
    "HUMAN_LEGACY_SAFETY_COMPARISON_HASH_DOMAIN_V1",
    "HumanLegacySafetyBindingEvidence",
    "HumanLegacySafetyComparison",
    "bind_human_legacy_safety_evidence",
    "build_human_legacy_aligned_safety_profile",
    "evaluate_human_legacy_aligned_safety",
]
