from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    ActionPersistencePolicy,
    ActiveExecutionSnapshot,
    CanonicalActionSafetyAnchor,
    CanonicalKeyframeSafetyProfile,
    EndEffectorSafetyCapability,
    EndEffectorTarget,
    HandActionSpec,
    HandQposSafetyLimits,
    RobotAction,
    apply_canonical_keyframe_safety,
    evaluate_shared_execution_shadow,
    project_robot_action_v1_to_component_action,
)
from robot_motion.position_control import (
    PositionKeyframeSafetyConfig,
    limit_joint_keyframes,
)


def profile() -> CanonicalKeyframeSafetyProfile:
    return CanonicalKeyframeSafetyProfile(
        profile_id="rm65-qpos-shadow-v1",
        limits=PositionKeyframeSafetyConfig(
            max_joint_step_rad=0.2,
            joint_lower_rad=(-1.0,) * 6,
            joint_upper_rad=(1.0,) * 6,
        ),
    )


def test_workspace_opt_out_preserves_the_original_default_profile_hash_payload() -> None:
    limits = PositionKeyframeSafetyConfig()
    enabled = CanonicalKeyframeSafetyProfile(
        profile_id="workspace-hash-compat-v1",
        limits=limits,
    )
    original_payload_hash = canonical_content_hash(
        {
            "profile_id": "workspace-hash-compat-v1",
            "workspace_min": limits.workspace_min,
            "workspace_max": limits.workspace_max,
            "workspace_soft_margin_m": limits.workspace_soft_margin_m,
            "max_translation_step_m": limits.max_translation_step_m,
            "max_rotation_step_deg": limits.max_rotation_step_deg,
            "max_joint_step_rad": limits.max_joint_step_rad,
            "joint_lower_rad": limits.joint_lower_rad,
            "joint_upper_rad": limits.joint_upper_rad,
        },
        domain="dexteleop.keyframe_safety_profile.v1",
    )
    disabled = CanonicalKeyframeSafetyProfile(
        profile_id="workspace-hash-compat-v1",
        limits=PositionKeyframeSafetyConfig(workspace_enabled=False),
    )

    assert enabled.semantic_content_hash == original_payload_hash
    assert disabled.semantic_content_hash != enabled.semantic_content_hash


def qpos_anchor() -> CanonicalActionSafetyAnchor:
    return CanonicalActionSafetyAnchor(
        anchor_id="measured-qpos-cut-0",
        anchor_kind="measured_state",
        action_space="bimanual_qpos_v1",
        arm_qpos_by_side={
            "left": np.zeros(6),
            "right": np.zeros(6),
        },
        observed_at=10.0,
    )


def test_canonical_qpos_safety_reuses_existing_keyframe_constraints() -> None:
    action = RobotAction.bimanual_qpos(
        left_arm=np.asarray([2.0, -2.0, 0.5, 0.0, 0.0, 0.0]),
        right_arm=np.asarray([-2.0, 2.0, -0.5, 0.0, 0.0, 0.0]),
        source="fixture-policy",
    )

    result = apply_canonical_keyframe_safety(
        action,
        "arbitration-result-0",
        anchor=qpos_anchor(),
        profile=profile(),
    )
    primitive = limit_joint_keyframes(
        action.arm_qpos_by_side,
        reference_by_side=qpos_anchor().arm_qpos_by_side,
        config=profile().limits,
    )

    assert result.status == "modified"
    assert result.safe_action is not None
    assert result.modified_components == ("left_arm", "right_arm")
    for side in ("left", "right"):
        np.testing.assert_allclose(
            result.safe_action.arm_qpos_by_side[side],
            primitive.values_by_side[side],
        )
    assert result.anchor_kind == "measured_state"
    assert result.anchor_id == "measured-qpos-cut-0"
    assert result.safety_contract_hash == profile().semantic_content_hash
    assert "shared_keyframe_safety" not in result.safe_action.metadata


def test_canonical_keyframe_safety_fails_closed_on_uncovered_semantics() -> None:
    partial_action = RobotAction.bimanual_qpos(
        left_arm=np.zeros(6),
        source="partial-policy",
    )
    mismatch = apply_canonical_keyframe_safety(
        partial_action,
        "result-component-mismatch",
        anchor=qpos_anchor(),
        profile=profile(),
    )
    assert mismatch.status == "fault"
    assert mismatch.safe_action is None
    assert mismatch.reasons == ("qpos_anchor_component_mismatch",)
    assert mismatch.anchor_id == qpos_anchor().anchor_id
    assert mismatch.anchor_hash == qpos_anchor().semantic_content_hash
    assert mismatch.safety_contract_id == profile().profile_id
    assert mismatch.safety_contract_hash == profile().semantic_content_hash

    hand_action = RobotAction.bimanual_qpos(
        left_arm=np.zeros(6),
        right_arm=np.zeros(6),
        left_hand=np.zeros(6),
        source="arm-hand-policy",
    )
    unsupported = apply_canonical_keyframe_safety(
        hand_action,
        "result-hand-unsupported",
        anchor=qpos_anchor(),
        profile=profile(),
    )
    assert unsupported.status == "fault"
    assert unsupported.safe_action is None
    assert unsupported.reasons == ("keyframe_safety_v1_hand_unsupported",)


def test_explicit_hand_profile_reuses_position_clamp_and_optional_step_semantics() -> None:
    hand_spec = HandActionSpec.dmtac_gripper("left")
    action = RobotAction.bimanual_ee_pose(
        left_hand=np.asarray([1.4, 0.0, 0.0, 0.0, 0.0, 0.0]),
        source="angle-trigger",
        metadata={
            "hand_action_specs_by_side": {
                "left": hand_spec.to_metadata(),
            }
        },
    )
    anchor = CanonicalActionSafetyAnchor(
        anchor_id="measured-left-gripper-0",
        anchor_kind="measured_state",
        action_space=action.action_space,
        hand_qpos_by_side={
            "left": np.asarray([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        },
        hand_action_specs_by_side={"left": hand_spec},
        observed_at=10.0,
    )
    hand_limits = HandQposSafetyLimits.uniform(
        hand_spec,
        lower=0.0,
        upper=1.0,
        max_step=0.1,
    )
    result = apply_canonical_keyframe_safety(
        action,
        "result-left-gripper",
        anchor=anchor,
        profile=CanonicalKeyframeSafetyProfile(
            profile_id="dmtac-position-compatible-v1",
            hand_limits_by_side={"left": hand_limits},
        ),
    )

    assert result.status == "modified"
    assert result.modified_components == ("left_hand",)
    assert result.reasons == (
        "left_hand:absolute_hand_qpos_limit",
        "left_hand:hand_qpos_step_limit",
    )
    assert result.safe_action is not None
    np.testing.assert_allclose(
        result.safe_action.hand_qpos_by_side["left"],
        [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert result.details["hand_metadata_by_side"]["left"]["command_space"] == (
        hand_spec.command_space
    )


def test_compound_pose_and_hand_safety_preserves_arm_and_padding_contracts() -> None:
    hand_spec = HandActionSpec.q16("right", active_dof=2)
    hand_target = np.full(16, np.nan, dtype=np.float32)
    hand_target[:2] = [0.25, 0.75]
    hand_anchor = np.full(16, np.nan, dtype=np.float32)
    hand_anchor[:2] = [0.2, 0.7]
    action = RobotAction.bimanual_ee_pose_hand16(
        right_arm=np.asarray([0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]),
        right_hand=hand_target,
        source="vive-q6",
        metadata={
            "hand_action_specs_by_side": {
                "right": hand_spec.to_metadata(),
            }
        },
    )
    anchor = CanonicalActionSafetyAnchor(
        anchor_id="measured-pose-hand-0",
        anchor_kind="measured_state",
        action_space=action.action_space,
        arm_pose_by_side={
            "right": np.asarray([0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]),
        },
        hand_qpos_by_side={"right": hand_anchor},
        hand_action_specs_by_side={"right": hand_spec},
        frame_by_side={"right": "right_arm_base"},
    )
    result = apply_canonical_keyframe_safety(
        action,
        "result-pose-hand",
        anchor=anchor,
        profile=CanonicalKeyframeSafetyProfile(
            profile_id="pose-q16-v1",
            hand_limits_by_side={
                "right": HandQposSafetyLimits.uniform(hand_spec),
            },
        ),
    )

    assert result.status == "safe"
    assert result.safe_action is not None
    np.testing.assert_allclose(
        result.safe_action.arm_pose_by_side["right"],
        action.arm_pose_by_side["right"],
    )
    np.testing.assert_allclose(
        result.safe_action.hand_qpos_by_side["right"][:2],
        [0.25, 0.75],
    )
    assert np.all(np.isnan(result.safe_action.hand_qpos_by_side["right"][2:]))


def _continuous_target(value: float) -> EndEffectorTarget:
    return EndEffectorTarget(
        value_type="continuous",
        values=np.asarray([value], dtype=np.float32),
        adapter="fake_parallel_gripper_v1",
        names=("opening",),
        metadata={"units": ("normalized",)},
    )


def _continuous_capability() -> EndEffectorSafetyCapability:
    return EndEffectorSafetyCapability(
        adapter="fake_parallel_gripper_v1",
        value_type="continuous",
        names=("opening",),
        units=("normalized",),
        lower=(0.0,),
        upper=(1.0,),
        max_step=(0.5,),
        max_rate_per_sec=(2.0,),
        maximum_rate_interval_sec=0.1,
    )


def test_native_continuous_safety_limits_absolute_step_and_bounded_rate() -> None:
    pose = np.asarray([0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
    action = RobotAction(
        action_space="bimanual_ee_pose_v1",
        arm_pose_by_side={"right": pose},
        end_effector_targets_by_side={"right": _continuous_target(1.4)},
        source="typed-policy",
    )
    anchor = CanonicalActionSafetyAnchor(
        anchor_id="previous-safe-native-0",
        anchor_kind="previous_safe_committed",
        action_space=action.action_space,
        arm_pose_by_side={"right": pose},
        end_effector_targets_by_side={"right": _continuous_target(0.2)},
        frame_by_side={"right": "right_arm_base"},
        observed_at=10.0,
    )
    safety_profile = CanonicalKeyframeSafetyProfile(
        profile_id="fake-native-continuous-v1",
        end_effector_capabilities_by_side={
            "right": _continuous_capability(),
        },
    )

    result = apply_canonical_keyframe_safety(
        action,
        "typed-emission-0",
        anchor=anchor,
        profile=safety_profile,
        # The elapsed time is deliberately large. The capability caps the
        # rate interval at 0.1 s, so the allowed rate delta remains 0.2.
        target_effective_at=10.4,
    )

    assert result.status == "modified"
    assert result.modified_components == ("right_hand",)
    assert result.reasons == (
        "right_hand:absolute_end_effector_limit",
        "right_hand:end_effector_step_limit",
        "right_hand:end_effector_rate_limit",
    )
    assert result.safe_action is not None
    np.testing.assert_allclose(
        result.safe_action.end_effector_targets_by_side["right"].values,
        [0.4],
    )
    details = result.details["end_effector_metadata_by_side"]["right"]
    assert details["elapsed_sec"] == pytest.approx(0.4)
    assert details["bounded_elapsed_sec"] == pytest.approx(0.1)
    assert result.safety_contract_hash == safety_profile.semantic_content_hash
    assert result.anchor_hash == anchor.semantic_content_hash


def test_native_continuous_safety_fails_closed_without_matching_capability() -> None:
    action = RobotAction(
        action_space="bimanual_ee_pose_v1",
        end_effector_targets_by_side={"right": _continuous_target(0.4)},
        source="typed-policy",
    )
    anchor = CanonicalActionSafetyAnchor(
        anchor_id="native-anchor-missing-profile",
        anchor_kind="previous_safe_committed",
        action_space=action.action_space,
        end_effector_targets_by_side={"right": _continuous_target(0.2)},
        observed_at=10.0,
    )

    result = apply_canonical_keyframe_safety(
        action,
        "typed-emission-missing-profile",
        anchor=anchor,
        profile=CanonicalKeyframeSafetyProfile(
            profile_id="no-native-capability-v1"
        ),
        target_effective_at=10.1,
    )

    assert result.status == "fault"
    assert result.safe_action is None
    assert result.reasons == (
        "end_effector_safety_profile_component_mismatch",
    )


def _discrete_target(value: int) -> EndEffectorTarget:
    return EndEffectorTarget(
        value_type="discrete",
        values=np.asarray([value], dtype=np.int64),
        adapter="fake_discrete_gripper_v1",
        names=("mode",),
        metadata={"units": ("token",)},
    )


def test_native_discrete_safety_accepts_only_allowlisted_transitions() -> None:
    capability = EndEffectorSafetyCapability(
        adapter="fake_discrete_gripper_v1",
        value_type="discrete",
        names=("mode",),
        units=("token",),
        allowed_tokens_by_component=((0, 1, 2),),
        allowed_transitions_by_component=(
            ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
        ),
    )
    pose = np.asarray([0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
    anchor = CanonicalActionSafetyAnchor(
        anchor_id="discrete-anchor-0",
        anchor_kind="previous_safe_committed",
        action_space="bimanual_ee_pose_v1",
        arm_pose_by_side={"right": pose},
        end_effector_targets_by_side={"right": _discrete_target(1)},
        frame_by_side={"right": "right_arm_base"},
        observed_at=10.0,
    )
    safety_profile = CanonicalKeyframeSafetyProfile(
        profile_id="fake-native-discrete-v1",
        end_effector_capabilities_by_side={"right": capability},
    )

    accepted = apply_canonical_keyframe_safety(
        RobotAction(
            action_space="bimanual_ee_pose_v1",
            arm_pose_by_side={"right": pose},
            end_effector_targets_by_side={"right": _discrete_target(2)},
        ),
        "typed-discrete-accepted",
        anchor=anchor,
        profile=safety_profile,
        target_effective_at=10.1,
    )
    rejected = apply_canonical_keyframe_safety(
        RobotAction(
            action_space="bimanual_ee_pose_v1",
            arm_pose_by_side={"right": pose},
            end_effector_targets_by_side={"right": _discrete_target(0)},
        ),
        "typed-discrete-rejected",
        anchor=anchor,
        profile=safety_profile,
        target_effective_at=10.1,
    )

    assert accepted.status == "safe"
    assert accepted.safe_action is not None
    assert rejected.status == "fault"
    assert rejected.safe_action is None
    assert "transition is not allowed" in rejected.reasons[0]


def test_canonical_keyframe_safety_rejects_invalid_anchor_evidence() -> None:
    with pytest.raises(ValueError, match="observed_at must be finite"):
        CanonicalActionSafetyAnchor(
            anchor_id="invalid-time",
            anchor_kind="measured_state",
            action_space="bimanual_qpos_v1",
            arm_qpos_by_side={"right": np.zeros(6)},
            observed_at=float("nan"),
        )

    with pytest.raises(ValueError, match="frames must be non-empty"):
        CanonicalActionSafetyAnchor(
            anchor_id="invalid-frame",
            anchor_kind="previous_selected",
            action_space="bimanual_ee_pose_v1",
            arm_pose_by_side={
                "right": np.asarray([0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
            },
            frame_by_side={"right": ""},
        )


def test_pose_safety_requires_exact_anchor_frame_evidence() -> None:
    action = RobotAction.bimanual_ee_pose(
        right_arm=np.asarray([0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]),
        source="pose-policy",
    )
    anchor = CanonicalActionSafetyAnchor(
        anchor_id="pose-anchor-0",
        anchor_kind="previous_selected",
        action_space=action.action_space,
        arm_pose_by_side={
            "right": np.asarray([0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
        },
        frame_by_side={"right": "unexpected_frame"},
    )

    result = apply_canonical_keyframe_safety(
        action,
        "result-pose-frame",
        anchor=anchor,
        profile=CanonicalKeyframeSafetyProfile(profile_id="pose-shadow-v1"),
    )

    assert result.status == "fault"
    assert result.safe_action is None
    assert result.reasons == ("right_pose_anchor_frame_mismatch",)


def test_explicit_anchor_safety_runs_inside_non_authoritative_execution_shadow() -> None:
    action = RobotAction.bimanual_qpos(
        left_arm=np.full(6, 0.3),
        right_arm=np.full(6, -0.3),
        source="fixture-policy",
    )
    arbitration_result = SimpleNamespace(
        result_id="arbitration-result-qpos-0",
        semantic_content_hash="arbitration-result-qpos-hash-0",
        output_status="selected",
        status_by_group={"arms": "selected"},
        selected_by_group={"arms": "proposal-qpos-0"},
        target_effective_at=10.0,
        supported_until=10.1,
        composed_candidate=project_robot_action_v1_to_component_action(action),
    )

    evaluated = evaluate_shared_execution_shadow(
        arbitration_result,
        active_execution=ActiveExecutionSnapshot(
            control_domain_id="robot-0",
            execution_epoch=0,
            active_segment_by_group={},
        ),
        persistence_policy=ActionPersistencePolicy(
            policy_id="qpos-shadow-segment-v1",
            selected_segment_max_lifetime_sec=0.1,
        ),
        shared_safety=partial(
            apply_canonical_keyframe_safety,
            anchor=qpos_anchor(),
            profile=profile(),
        ),
        now=10.0,
        supports_local_group_stop=False,
        source_arbitration_evaluation_id="arbitration-evaluation-qpos-0",
    )

    assert evaluated.non_authoritative
    assert evaluated.output_status == "command_intent_ready"
    assert evaluated.safety_result is not None
    assert evaluated.safety_result.status == "modified"
    assert evaluated.command_intent is not None
    np.testing.assert_allclose(
        evaluated.command_intent["commands"]["left_arm"]["value"],
        np.full(6, 0.2),
    )
