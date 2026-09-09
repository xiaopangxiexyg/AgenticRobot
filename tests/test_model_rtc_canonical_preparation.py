from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from dataset_contracts import PoseFrameContract, PoseSourceProfile
from dataset_contracts.v22 import EndEffectorSpec, PolicyIOLayout
from robot_motion.action import EndEffectorTarget, RobotAction
from robot_motion.deploy import (
    ACTION_ARM_POSE_SLICES,
    ACTION_HAND_SLICES,
    DEPLOY_SCHEMA_V2,
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
    POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
    POLICY_CONTRACT_V22_NATIVE,
    POLICY_PLAN_KEYFRAME_MUTABLE,
    RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    CanonicalDeployActionPipeline,
    CanonicalPolicyActionResult,
    FilteredNormalizedAction,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    ResolvedDeploySessionContract,
    RuntimeEndEffectorCapability,
    TimedPolicyActionKeyframe,
    build_canonical_policy_action,
    build_deployment_observation,
)
from teleop_runtime.deploy.action_execution import PreparedActionExecutor
from teleop_runtime.deploy.hand_backend_config import HAND_BACKEND_DMTAC_GRIPPER
from teleop_runtime.deploy.run_policy import (
    _ModelRtcCanonicalActionPreparer,
    _prepared_action_last_action,
    _rtc_action_preparer_for_profile,
    _runtime_end_effector_safety_capability,
)


class _Clock:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _layout() -> PolicyIOLayout:
    return PolicyIOLayout(
        name="dual_dmtac_model_rtc_test",
        end_effectors={
            side: EndEffectorSpec.continuous(
                side,
                adapter="dmtac_gripper_v1",
                target_dim=1,
                target_names=("gripper_close",),
                target_source_indices=(0,),
                state_source_indices=(0,),
                target_legacy_q6_slot_indices=(0,),
                units="normalized_0_1",
            )
            for side in ("left", "right")
        },
    )


def _runtime_pose_contract() -> PoseFrameContract:
    left = np.eye(4, dtype=float)
    left[:3, 3] = [0.4, -0.1, 0.05]
    right = np.eye(4, dtype=float)
    right[:3, 3] = [-0.3, 0.2, -0.02]
    return PoseFrameContract.v23_forced_raw(
        PoseSourceProfile(
            source_id="model-rtc-runtime-test",
            source_family="test",
            source_description="non-identity runtime frame fixture",
            topology="dual_arm",
            active_sides=("left", "right"),
            raw_absolute_frame_by_side={
                "left": "left_arm_base",
                "right": "right_arm_base",
            },
            raw_to_desk_transform_by_side={"left": left, "right": right},
        )
    )


def _backend_args() -> Namespace:
    return Namespace(
        hand_backend=HAND_BACKEND_DMTAC_GRIPPER,
        left_hand_backend=None,
        right_hand_backend=None,
    )


def _session(
    layout: PolicyIOLayout,
    pose_contract: PoseFrameContract,
) -> ResolvedDeploySessionContract:
    args = _backend_args()
    capabilities = {
        side: RuntimeEndEffectorCapability(
            side=side,
            adapter="dmtac_gripper_v1",
            backend=HAND_BACKEND_DMTAC_GRIPPER,
            safety=_runtime_end_effector_safety_capability(
                args=args,
                deploy_layout=layout,
                side=side,
            ),
        )
        for side in ("left", "right")
    }
    return ResolvedDeploySessionContract(
        deploy_schema_version=DEPLOY_SCHEMA_V2,
        policy_contract_mode=POLICY_CONTRACT_V22_NATIVE,
        active_sides=("left", "right"),
        action_dim=layout.action_dim,
        action_discrete_dim=layout.action_discrete_dim,
        state_dim=layout.state_dim,
        state_discrete_dim=layout.state_discrete_dim,
        policy_absolute_frame=POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
        runtime_pose_frame_contract=pose_contract,
        policy_io_layout=layout,
        runtime_end_effectors=capabilities,
        hardware_output=True,
    )


def _pose(x: float, y: float, z: float) -> np.ndarray:
    return np.asarray([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _observation():
    arm_poses = {
        "left": _pose(-0.20, -0.25, 0.35),
        "right": _pose(0.20, 0.25, 0.35),
    }
    arm_states = {}
    for side, pose in arm_poses.items():
        matrix = np.eye(4, dtype=float)
        matrix[:3, 3] = pose[:3]
        arm_states[side] = {
            "current_qpos": np.zeros(6, dtype=np.float32),
            "current_tcp_pose": matrix,
            "actual_tcp_pose": matrix,
        }
    hand_states = {
        "left": {
            "joint_pos": np.asarray([0.25, 0, 0, 0, 0, 0], dtype=np.float32),
            "joint_timestamp": 10.0,
        },
        "right": {
            "joint_pos": np.asarray([0.75, 0, 0, 0, 0, 0], dtype=np.float32),
            "joint_timestamp": 10.0,
        },
    }
    return build_deployment_observation(
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
        end_effector_sides=("left", "right"),
        step_index=7,
        arm_states=arm_states,
        hand_states=hand_states,
        timestamp=10.0,
    )


def _canonical_action(pose_contract: PoseFrameContract):
    runtime_targets = {
        "left": _pose(0.10, -0.25, 0.35),
        "right": _pose(0.50, 0.25, 0.35),
    }
    desk_targets = {
        side: pose_contract.storage_pose_vector_to_canonical_desk(pose, side=side)
        for side, pose in runtime_targets.items()
    }
    return build_canonical_policy_action(
        RobotAction(
            action_space="dexteleop_policy_io_v2.2",
            timestamp=10.0,
            arm_pose_by_side=desk_targets,
            end_effector_targets_by_side={
                "left": EndEffectorTarget(
                    value_type="continuous",
                    values=np.asarray([1.4], dtype=np.float32),
                    adapter="dmtac_gripper_v1",
                    names=("gripper_close",),
                    metadata={"units": ("normalized_0_1",)},
                ),
                "right": EndEffectorTarget(
                    value_type="continuous",
                    values=np.asarray([-0.2], dtype=np.float32),
                    adapter="dmtac_gripper_v1",
                    names=("gripper_close",),
                    metadata={"units": ("normalized_0_1",)},
                ),
            },
            frame_by_effector={
                "left_arm": "canonical_desk",
                "right_arm": "canonical_desk",
            },
            source="model-rtc-test",
        ),
        source_pose_mode="absolute",
        observation_cut_id="observation:model-rtc:7",
        source_action_hash="source:model-rtc:7",
        action_codec_hash="codec:model-rtc",
        frame_contract_hash="frame:model-rtc",
        layout_hash="layout:model-rtc",
        normalization_hash="normalization:model-rtc",
        conversion_contract_hash="conversion:model-rtc",
    )


def _keyframe(canonical_action) -> TimedPolicyActionKeyframe:
    return TimedPolicyActionKeyframe(
        target_effective_at=10.01,
        effective_until=10.10,
        effective_clock_domain_id="monotonic:test",
        action_period_sec=0.1,
        decoded_absolute_action=None,
        request_id="request:model-rtc:7",
        inference_id="inference:model-rtc:7",
        policy_output_id="output:model-rtc:7",
        chunk_id=1,
        chunk_keyframe_index=0,
        chunk_size=1,
        observation_cut_id="observation:model-rtc:7",
        producer_epoch=1,
        policy_node_id="policy:model-rtc",
        artifact_hash="artifact:model-rtc",
        produced_at=10.0,
        schedule_spec_id="schedule:model-rtc",
        schedule_spec_hash="schedule-hash:model-rtc",
        admitted_plan_generation=1,
        state=POLICY_PLAN_KEYFRAME_MUTABLE,
        valid_until=10.10,
        canonical_action=canonical_action,
    )


def _safety_filter() -> ActionSafetyFilter:
    return ActionSafetyFilter(
        ActionSafetyFilterConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
            workspace_min=(-1.0, -1.0, 0.0),
            workspace_max=(1.0, 1.0, 1.0),
            workspace_soft_margin_m=0.0,
            max_arm_xyz_step_m=0.05,
            max_arm_rot_step_deg=15.0,
            max_arm_joint_step_rad=0.2,
        )
    )


def _legacy_pipeline(
    safety_filter: ActionSafetyFilter,
) -> CanonicalDeployActionPipeline:
    return CanonicalDeployActionPipeline(
        safety_filter=safety_filter,
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
    )


def _runtime() -> PolicyDeployRuntime:
    return PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
        ),
        monotonic_clock=_Clock(10.001),
    )


def test_legacy_preparer_keeps_typed_emission_fail_closed() -> None:
    pose_contract = _runtime_pose_contract()
    canonical_action = _canonical_action(pose_contract)
    result = _runtime().prepare_timed_action_keyframe(
        _keyframe(canonical_action),
        _observation(),
        preparer=_legacy_pipeline(_safety_filter()),
        policy_name="model-rtc-test",
        plan_keyframe_count_before=1,
        plan_keyframe_count_after=0,
        plan_pop_count=1,
    )

    assert result.emission is not None
    assert result.prepared_action is None
    assert isinstance(result.emission_error, RuntimeError)
    assert "Typed canonical preparation is unavailable" in str(result.emission_error)


def test_model_rtc_preparer_converts_frames_and_applies_measured_safety() -> None:
    layout = _layout()
    pose_contract = _runtime_pose_contract()
    session = _session(layout, pose_contract)
    observation = _observation()
    canonical_action = _canonical_action(pose_contract)
    preparer = _ModelRtcCanonicalActionPreparer(
        deploy_session_contract=session,
        safety_filter=_safety_filter(),
    )

    anchor = preparer._measured_anchor(
        preparer._runtime_action(canonical_action),
        observation,
        10.01,
    )
    np.testing.assert_allclose(
        anchor.end_effector_targets_by_side["left"].values,
        [0.25],
    )
    np.testing.assert_allclose(
        anchor.end_effector_targets_by_side["right"].values,
        [0.75],
    )

    result = _runtime().prepare_timed_action_keyframe(
        _keyframe(canonical_action),
        observation,
        preparer=preparer,
        policy_name="model-rtc-test",
        plan_keyframe_count_before=1,
        plan_keyframe_count_after=0,
        plan_pop_count=1,
    )

    assert result.emission_error is None
    assert isinstance(result.prepared_action, CanonicalPolicyActionResult)
    prepared = result.prepared_action
    assert prepared.faults == ()
    assert prepared.safety_result.status == "modified"
    assert prepared.robot_action is not None
    safe_action = prepared.robot_action
    np.testing.assert_allclose(
        safe_action.arm_pose_by_side["left"],
        _pose(-0.15, -0.25, 0.35),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        safe_action.arm_pose_by_side["right"],
        _pose(0.25, 0.25, 0.35),
        atol=1e-6,
    )
    assert safe_action.arm_frame("left") == "left_arm_base"
    assert safe_action.arm_frame("right") == "right_arm_base"
    np.testing.assert_allclose(
        safe_action.end_effector_targets_by_side["left"].values,
        [1.0],
    )
    np.testing.assert_allclose(
        safe_action.end_effector_targets_by_side["right"].values,
        [0.0],
    )
    assert prepared.command is not None
    for side, value in (("left", 1.0), ("right", 0.0)):
        command = prepared.command.commands[f"{side}_hand"]
        assert command.command_type == "gripper_scalar"
        np.testing.assert_allclose(command.value, [value])

    last_action = _prepared_action_last_action(
        prepared,
        observation,
        deploy_layout=layout,
    )
    for side in ("left", "right"):
        np.testing.assert_allclose(
            last_action[ACTION_ARM_POSE_SLICES[side]],
            safe_action.arm_pose_by_side[side],
        )
    assert last_action[ACTION_HAND_SLICES["left"]][0] == pytest.approx(1.0)
    assert last_action[ACTION_HAND_SLICES["right"]][0] == pytest.approx(0.0)

    execution = PreparedActionExecutor(
        position_deploy_executor=None,
        send_actions=False,
        arm_backends={},
        hand_backends={},
        active_sides=("left", "right"),
        position_worker_binding=None,
        submit_legacy_owner_state=lambda **_kwargs: None,
    ).execute(
        prepared,
        observation,
        policy_name="model-rtc-test",
    )
    assert execution.accepted is True
    assert execution.mode == "dry_run"
    assert execution.dispatch_attempted is False

    position_executor = Mock()
    position_executor.submit.return_value = 9
    position_execution = PreparedActionExecutor(
        position_deploy_executor=position_executor,
        send_actions=False,
        arm_backends={},
        hand_backends={},
        active_sides=("left", "right"),
        position_worker_binding=None,
        submit_legacy_owner_state=lambda **_kwargs: None,
    ).execute(
        prepared,
        observation,
        policy_name="model-rtc-test",
    )
    assert position_execution.accepted is True
    assert position_execution.mode == "position_dry_run_queue"
    assert position_execution.position_source_index == 9
    filtered = position_executor.submit.call_args.args[0]
    assert isinstance(filtered, FilteredNormalizedAction)
    assert filtered.faults == []
    for side in ("left", "right"):
        np.testing.assert_allclose(
            filtered.action.vector[ACTION_ARM_POSE_SLICES[side]],
            safe_action.arm_pose_by_side[side],
        )
        spec = filtered.action.metadata["arm_action_specs_by_side"][side]
        assert spec["reference_frame"] == f"{side}_arm_base"
        assert spec["controlled_frame"] == "tool_tcp"
    assert set(
        position_executor.submit.call_args.kwargs["discrete_command"].commands
    ) == {
        "left_hand",
        "right_hand",
    }

    inconsistent = replace(
        prepared,
        safety_result=replace(
            prepared.safety_result,
            status="hold",
            safe_action=None,
        ),
        faults=(),
    )
    with pytest.raises(RuntimeError, match="no executable safe action"):
        PreparedActionExecutor(
            position_deploy_executor=position_executor,
            send_actions=False,
            arm_backends={},
            hand_backends={},
            active_sides=("left", "right"),
            position_worker_binding=None,
            submit_legacy_owner_state=lambda **_kwargs: None,
        ).execute(
            inconsistent,
            observation,
            policy_name="model-rtc-test",
        )


def test_only_conditioned_model_rtc_selects_typed_preparation() -> None:
    layout = _layout()
    session = _session(layout, _runtime_pose_contract())
    safety_filter = _safety_filter()
    legacy = _legacy_pipeline(safety_filter)

    for profile in ("off", RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1):
        assert (
            _rtc_action_preparer_for_profile(
                rtc_profile_id=profile,
                canonical_action_pipeline=legacy,
                deploy_session_contract=session,
                safety_filter=safety_filter,
            )
            is legacy
        )
    assert isinstance(
        _rtc_action_preparer_for_profile(
            rtc_profile_id=MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
            canonical_action_pipeline=legacy,
            deploy_session_contract=session,
            safety_filter=safety_filter,
        ),
        _ModelRtcCanonicalActionPreparer,
    )
