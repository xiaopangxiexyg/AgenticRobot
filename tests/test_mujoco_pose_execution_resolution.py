from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.action import (
    ActionResolutionError,
    CanonicalDeskToRuntimePoseTransformer,
    DecisionScheduleSpec,
    HandActionSpec,
    RobotAction,
    build_authoritative_command_dispatch_context,
    build_authoritative_receiver_state_snapshot,
    build_scheduled_reference_segment_activation_spec,
)
from robot_motion.pose_frame_profiles import (
    default_rm65_workspace_near_boundary_collection_pose_frame_contract,
    default_rm65_workspace_near_boundary_runtime_pose_frame_contract,
)
from robot_motion.pose_smoothing import BimanualPoseSmoother, PoseSmoothingConfig
from robot_motion.position_control import ExecutionCapabilityProfile
from sim.contracts import TimingSpec
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.deploy.pose_resolution import (
    MujocoCoupledPoseToQposResolver,
    POSE_TO_QPOS_RESOLVER_CONTRACT_HASH_V1,
    build_mujoco_pose_qpos_execution_target,
)
from sim.mujoco.deploy.resolved_reference_receiver import (
    MujocoScheduledResolvedQposReferenceReceiver,
)
from sim.mujoco.ik import RM65CanonicalIK


class _Scene:
    arm_specs = {"left": object(), "right": object()}
    gripper_specs = {}

    def __init__(self) -> None:
        self.qpos = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }

    def get_arm_qpos(self, side: str) -> np.ndarray:
        return self.qpos[side].copy()

    def step_kinematic_direct_qpos_debug(
        self,
        qpos_by_side,
        *,
        gripper_close_by_name,
        zero_controlled_qvel,
    ) -> None:
        del gripper_close_by_name, zero_controlled_qvel
        self.qpos = {
            side: np.asarray(value, dtype=float).reshape(6).copy()
            for side, value in qpos_by_side.items()
        }


class _Planner:
    collision_pair_count = 1

    def __init__(self, *, hold: bool = False) -> None:
        self.hold = bool(hold)
        self.targets: list[dict[str, np.ndarray]] = []

    def reset(self, _qpos_by_side):
        return {"initialized": True}

    def plan_pose_reference(self, pose_by_side, **_kwargs):
        self.targets.append(
            {
                side: np.asarray(pose, dtype=float).copy()
                for side, pose in pose_by_side.items()
            }
        )
        identity = np.eye(4, dtype=float)
        return SimpleNamespace(
            valid=True,
            hold=self.hold,
            status="hold_last_safe" if self.hold else "tracking",
            q_ref_by_side={
                "left": np.full(6, 0.15, dtype=float),
                "right": np.full(6, -0.15, dtype=float),
            },
            target_endpoint_pose_world_by_side={
                side: identity.copy() for side in pose_by_side
            },
            realized_endpoint_pose_world_by_side={
                side: identity.copy() for side in pose_by_side
            },
            modification_reasons=("hold_last_safe",) if self.hold else (),
            metadata={
                "planner": "fixture-coupled-rm65",
                "failure": (
                    {
                        "code": "collision_margin_low",
                        "error_type": "_PlanningFailure",
                    }
                    if self.hold
                    else None
                ),
                "collision_min_distance_m": 0.03,
                "collision_distance_epsilon_m": 0.0001,
                "collision_distance_m": 0.031,
                "collision_check_attempts": (
                    [
                        {"scale": 1.0, "distance_m": 0.022},
                        {
                            "scale": 0.25,
                            "distance_m": 0.029,
                            "collision_pair_index": 4,
                            "collision_geometry_1": "right_hand_proxy_0",
                            "collision_geometry_2": "table_0",
                        },
                    ]
                    if self.hold
                    else []
                ),
            },
        )


def _pose_action(*, left_frame: str = "left_arm_base") -> RobotAction:
    identity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return RobotAction(
        action_space="bimanual_ee_pose_v1",
        arm_pose_by_side={"left": identity, "right": identity},
        timestamp=10.0,
        source="shared_safe_pose",
        frame_by_effector={
            "left_arm": left_frame,
            "right_arm": "right_arm_base",
        },
        controlled_frame_by_effector={
            "left_arm": "tool_tcp",
            "right_arm": "tool_tcp",
        },
    )


def test_pose_resolution_preserves_safe_pose_and_emits_separate_qpos_command() -> None:
    scene = _Scene()
    planner = _Planner()
    resolver = MujocoCoupledPoseToQposResolver(RM65CanonicalIK(scene, planner))
    safe_pose = _pose_action()

    command = resolver.resolve(safe_pose)
    resolution = resolver.resolution_for_command(command)

    assert resolution.input_safe_action_hash == command.metadata["input_safe_action_hash"]
    assert resolution.input_safe_action.action_space == "bimanual_ee_pose_v1"
    assert resolution.output_qpos_action.action_space == "bimanual_qpos_v1"
    assert resolution.resolver_contract_hash == POSE_TO_QPOS_RESOLVER_CONTRACT_HASH_V1
    assert resolution.transform_contract_hash is None
    assert set(command.commands) == {"left_arm", "right_arm"}
    assert all(item.command_type == "joint_vector" for item in command.commands.values())
    np.testing.assert_allclose(command.commands["left_arm"].value, np.full(6, 0.15))
    np.testing.assert_allclose(command.commands["right_arm"].value, np.full(6, -0.15))
    assert planner.targets and set(planner.targets[0]) == {"left", "right"}
    # Resolution is pure with respect to the MuJoCo plant.
    np.testing.assert_allclose(scene.qpos["left"], np.zeros(6))
    np.testing.assert_allclose(scene.qpos["right"], np.zeros(6))
    timing = resolver.last_resolve_diagnostics
    assert timing["resolve_invoked"] is True
    assert timing["resolve_status"] == "accepted"
    assert timing["ik_status"] == "tracking"
    assert float(timing["resolve_total_elapsed_sec"]) >= float(
        timing["ik_elapsed_sec"]
    )
    assert float(timing["resolution_build_elapsed_sec"]) >= 0.0
    assert float(timing["command_resolve_elapsed_sec"]) >= 0.0
    assert float(timing["command_metadata_elapsed_sec"]) >= 0.0
    assert float(timing["resolution_cache_insert_elapsed_sec"]) >= 0.0
    assert timing["resolution_cache_size_before"] == 0
    assert timing["resolution_cache_size_after"] == 1
    assert len(timing["gc_count_before"]) == 3
    assert len(timing["gc_count_after"]) == 3
    assert all(
        isinstance(value, int) and value >= 0
        for value in timing["gc_count_before"]
    )
    assert all(
        isinstance(value, int) and value >= 0
        for value in timing["gc_count_after"]
    )
    resolver.clear_last_resolve_diagnostics()
    assert resolver.last_resolve_diagnostics == {}


def test_pose_resolution_smooths_before_ik_and_resets_temporal_state() -> None:
    scene = _Scene()
    planner = _Planner()
    smoother = BimanualPoseSmoother(
        PoseSmoothingConfig(
            time_constant_sec=1.0,
            nominal_dt_sec=0.1,
            max_dt_sec=0.1,
            deadband_translation_m=0.0,
            deadband_rotation_deg=0.0,
            snap_translation_m=0.0,
            snap_rotation_deg=0.0,
        )
    )
    resolver = MujocoCoupledPoseToQposResolver(
        RM65CanonicalIK(scene, planner),
        target_pose_smoother=smoother,
    )

    first = _pose_action()
    resolver.resolve(first)
    moved = RobotAction(
        action_space=first.action_space,
        timestamp=10.1,
        arm_pose_by_side={
            side: np.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            for side in ("left", "right")
        },
        frame_by_effector=first.frame_by_effector,
        controlled_frame_by_effector=first.controlled_frame_by_effector,
        source=first.source,
    )

    command = resolver.resolve(moved)
    resolution = resolver.resolution_for_command(command)
    solver_x = float(
        resolution.solver_input_pose_action.arm_pose_by_side["left"][0]
    )
    assert 0.0 < solver_x < 0.1
    assert float(
        resolution.input_safe_action.arm_pose_by_side["left"][0]
    ) == pytest.approx(0.1)
    assert resolution.solver_metadata["target_pose_smoothing"]["left"][
        "status"
    ] == "smoothed"
    assert planner.targets[1]["left"][0, 3] == pytest.approx(solver_x)

    resolver.reset_temporal_state()
    reset_target = RobotAction(
        action_space=moved.action_space,
        timestamp=10.2,
        arm_pose_by_side={
            side: np.asarray([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            for side in ("left", "right")
        },
        frame_by_effector=moved.frame_by_effector,
        controlled_frame_by_effector=moved.controlled_frame_by_effector,
        source=moved.source,
    )
    reset_resolution = resolver.resolution_for_command(
        resolver.resolve(reset_target)
    )
    assert float(
        reset_resolution.solver_input_pose_action.arm_pose_by_side["left"][0]
    ) == pytest.approx(0.2)
    assert reset_resolution.solver_metadata["target_pose_smoothing"]["left"][
        "status"
    ] == "initialized"


def test_pose_resolution_preserves_safe_q6_gripper_components() -> None:
    scene = _Scene()
    resolver = MujocoCoupledPoseToQposResolver(
        RM65CanonicalIK(scene, _Planner())
    )
    hand_specs = {
        side: HandActionSpec.dmtac_gripper(side)
        for side in ("left", "right")
    }
    safe_pose = _pose_action()
    safe_pose = RobotAction(
        action_space=safe_pose.action_space,
        timestamp=safe_pose.timestamp,
        arm_pose_by_side=safe_pose.arm_pose_by_side,
        hand_qpos_by_side={
            "left": np.asarray([0.25, 0, 0, 0, 0, 0], dtype=np.float32),
            "right": np.asarray([0.75, 0, 0, 0, 0, 0], dtype=np.float32),
        },
        frame_by_effector=safe_pose.frame_by_effector,
        controlled_frame_by_effector=safe_pose.controlled_frame_by_effector,
        source=safe_pose.source,
        metadata={
            "hand_action_specs_by_side": {
                side: spec.to_metadata() for side, spec in hand_specs.items()
            }
        },
    )

    command = resolver.resolve(safe_pose)
    resolution = resolver.resolution_for_command(command)

    assert set(command.commands) == {
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    }
    np.testing.assert_array_equal(command.commands["left_hand"].value, [0.25])
    np.testing.assert_array_equal(command.commands["right_hand"].value, [0.75])
    np.testing.assert_array_equal(
        resolution.output_qpos_action.hand_qpos_by_side["left"],
        safe_pose.hand_qpos_by_side["left"],
    )
    assert resolution.resolver_contract_hash != resolver.resolver_contract_hash
    assert (
        "safe-hand-qpos-passthrough-v1"
        in resolution.resolver_contract_id
    )


def test_pose_resolution_refuses_unverified_frame_conversion() -> None:
    resolver = MujocoCoupledPoseToQposResolver(
        RM65CanonicalIK(_Scene(), _Planner())
    )
    with pytest.raises(ValueError, match="exact arm-base input"):
        resolver.resolve(_pose_action(left_frame="canonical_desk"))


def test_verified_canonical_desk_transform_occurs_after_safe_action() -> None:
    source_contract = default_rm65_workspace_near_boundary_collection_pose_frame_contract(
        ("left", "right")
    )
    assert source_contract is not None
    target_contract = (
        default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
    )
    transformer = CanonicalDeskToRuntimePoseTransformer(
        source_pose_frame_contract=source_contract,
        target_pose_frame_contract=target_contract,
    )
    planner = _Planner()
    resolver = MujocoCoupledPoseToQposResolver(
        RM65CanonicalIK(_Scene(), planner),
        pose_transformer=transformer,
    )
    safe_pose = _pose_action(left_frame="canonical_desk")
    safe_pose = RobotAction(
        action_space=safe_pose.action_space,
        timestamp=safe_pose.timestamp,
        arm_pose_by_side=safe_pose.arm_pose_by_side,
        frame_by_effector={
            "left_arm": "canonical_desk",
            "right_arm": "canonical_desk",
        },
        controlled_frame_by_effector=safe_pose.controlled_frame_by_effector,
        source=safe_pose.source,
    )

    command = resolver.resolve(safe_pose)
    resolution = resolver.resolution_for_command(command)

    assert resolution.input_safe_action.arm_frame("left") == "canonical_desk"
    assert resolution.input_safe_action.arm_frame("right") == "canonical_desk"
    assert resolution.solver_input_pose_action.arm_frame("left") == "left_arm_base"
    assert resolution.solver_input_pose_action.arm_frame("right") == "right_arm_base"
    assert resolution.transform_receipt_id == transformer.binding_receipt.receipt_id
    assert (
        resolution.transform_contract_hash
        == transformer.binding_receipt.semantic_content_hash
    )
    assert resolution.resolver_contract_hash != POSE_TO_QPOS_RESOLVER_CONTRACT_HASH_V1
    assert command.metadata["transform_contract_hash"] == (
        transformer.binding_receipt.semantic_content_hash
    )
    expected_left = target_contract.canonical_desk_pose_vector_to_storage(
        safe_pose.arm_pose_by_side["left"], side="left"
    )
    np.testing.assert_allclose(
        resolution.solver_input_pose_action.arm_pose_by_side["left"],
        expected_left,
        atol=1e-6,
    )


def test_verified_canonical_desk_transform_preserves_safe_q6_gripper() -> None:
    source_contract = default_rm65_workspace_near_boundary_collection_pose_frame_contract(
        ("left", "right")
    )
    assert source_contract is not None
    transformer = CanonicalDeskToRuntimePoseTransformer(
        source_pose_frame_contract=source_contract,
        target_pose_frame_contract=(
            default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
        ),
    )
    hand_spec = HandActionSpec.dmtac_gripper("left")
    safe_pose = _pose_action(left_frame="canonical_desk")
    safe_pose = RobotAction(
        action_space=safe_pose.action_space,
        timestamp=safe_pose.timestamp,
        arm_pose_by_side=safe_pose.arm_pose_by_side,
        hand_qpos_by_side={
            "left": np.asarray([0.4, 0, 0, 0, 0, 0], dtype=np.float32)
        },
        frame_by_effector={
            "left_arm": "canonical_desk",
            "right_arm": "canonical_desk",
        },
        controlled_frame_by_effector=safe_pose.controlled_frame_by_effector,
        source=safe_pose.source,
        metadata={
            "hand_action_specs_by_side": {
                "left": hand_spec.to_metadata()
            }
        },
    )
    resolver = MujocoCoupledPoseToQposResolver(
        RM65CanonicalIK(_Scene(), _Planner()),
        pose_transformer=transformer,
    )

    resolution = resolver.resolution_for_command(resolver.resolve(safe_pose))

    np.testing.assert_array_equal(
        resolution.solver_input_pose_action.hand_qpos_by_side["left"],
        safe_pose.hand_qpos_by_side["left"],
    )
    assert resolution.solver_input_pose_action.arm_frame("left") == "left_arm_base"


def test_pose_resolution_hold_never_creates_executable_command() -> None:
    resolver = MujocoCoupledPoseToQposResolver(
        RM65CanonicalIK(_Scene(), _Planner(hold=True))
    )
    with pytest.raises(ActionResolutionError) as captured:
        resolver.resolve(_pose_action())
    failure = captured.value.failure
    assert failure.reason_code == "collision_margin_low"
    assert failure.resolver_status == "hold_last_safe"
    assert failure.outcome_code == (
        "target_private_resolution_failure.collision_margin_low"
    )
    assert failure.diagnostics["best_candidate_distance_m"] == pytest.approx(
        0.029
    )
    assert failure.diagnostics["collision_attempt_count"] == 2
    assert failure.diagnostics["collision_threshold_m"] == pytest.approx(
        0.0299
    )
    assert failure.diagnostics["collision_pair_index"] == 4
    assert failure.diagnostics["collision_geometry_1"] == (
        "right_hand_proxy_0"
    )
    assert failure.diagnostics["collision_geometry_2"] == "table_0"
    timing = resolver.last_resolve_diagnostics
    assert timing["resolve_status"] == "rejected"
    assert timing["resolve_failure_reason_code"] == "collision_margin_low"
    assert float(timing["ik_elapsed_sec"]) >= 0.0


def test_resolution_lookup_rejects_untyped_command() -> None:
    resolver = MujocoCoupledPoseToQposResolver(
        RM65CanonicalIK(_Scene(), _Planner())
    )
    command = resolver.resolve(_pose_action())
    command.metadata.pop("pose_to_qpos_resolution_hash")
    with pytest.raises(ValueError, match="lacks typed"):
        resolver.resolution_for_command(command)


def test_resolved_receiver_keeps_pose_anchor_separate_from_qpos_segment() -> None:
    clock_value = 10.0

    def clock() -> float:
        return clock_value

    scene = _Scene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=TimingSpec(
                sample_hz=15.0,
                policy_hz=15.0,
                reference_hz=60.0,
                ik_hz=60.0,
                command_hz=120.0,
                physics_hz=1000.0,
            ),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
        ),
        monotonic_clock=clock,
    )
    capability = ExecutionCapabilityProfile(
        profile_id="resolved-pose-qpos-fixture-v1",
        validated_max_selected_keyframe_hz=60.0,
        configured_selected_keyframe_hz=30.0,
        validated_max_coupled_ik_hz=60.0,
        configured_coupled_ik_hz=30.0,
        reference_resample_hz=60.0,
        command_hz=120.0,
        backend_servo_hz=1000.0,
        validated_action_spaces=("bimanual_ee_pose_v1",),
        validated_selection_groups=("arms",),
        supports_local_group_stop=False,
        acceptance_evidence_id="resolved-pose-qpos-fixture-v1",
        decision_runtime_budget_sec=0.005,
        ik_runtime_budget_sec=1.0 / 60.0,
        end_to_end_control_budget_sec=1.0 / 30.0,
        required_cancellation_ack_levels=("software_queue",),
    )
    schedule = DecisionScheduleSpec(
        spec_id="resolved-pose-qpos-30hz-v1",
        decision_hz=30.0,
        target_lead_time_sec=0.05,
    )
    activation = build_scheduled_reference_segment_activation_spec(
        schedule,
        capability,
        minimum_activation_margin_sec=1.0 / 60.0,
        activation_margin_evidence_id="resolved-pose-qpos-margin-v1",
    )
    receiver = MujocoScheduledResolvedQposReferenceReceiver(
        rig,
        activation_spec=activation,
        action_schema_hash="qpos-reference-schema-v1",
        pose_action_schema_hash="pose-action-schema-v1",
        clock_domain_id="local_monotonic",
        live_state_provider=lambda: build_authoritative_receiver_state_snapshot(
            runtime_session_id="session-0",
            control_domain_id="domain-0",
            current_owner_token_id="token-0",
            current_owner_generation=1,
            current_authority_epoch=2,
            current_execution_epoch=3,
            expected_next_group_generation_by_group={"arms": 1},
            captured_at=clock(),
        ),
        activation_callback=lambda _group, _generation: None,
        clock=clock,
    )
    rig.start()
    safe_pose = _pose_action()
    context = build_authoritative_command_dispatch_context(
        runtime_session_id="session-0",
        control_domain_id="domain-0",
        embodiment_instance_id="embodiment-0",
        decision_id="decision-0",
        directive_id="directive-0",
        safety_result_id="safety-0",
        command_id="command-0",
        dispatch_permit_id="permit-0",
        owner_token_id="token-0",
        owner_generation=1,
        target_effective_at=10.05,
        authority_epoch=2,
        execution_epoch=3,
        group_segment_generation_by_group={"arms": 1},
        selected_action_ref_by_group={"arms": "proposal-0"},
        safe_action=safe_pose,
    )
    resolver = MujocoCoupledPoseToQposResolver(
        RM65CanonicalIK(scene, _Planner())
    )
    command = resolver.resolve(safe_pose)
    resolution = resolver.resolution_for_command(command)
    target = build_mujoco_pose_qpos_execution_target(
        resolution,
        context,
        resolved_at=clock(),
        target_validity_sec=0.1,
        clock_domain_id="local_monotonic",
        input_action_schema_hash="pose-action-schema-v1",
        output_target_schema_hash="qpos-reference-schema-v1",
    )

    receipts = receiver.dispatch_resolved_target(
        command,
        context,
        target=target,
        resolution=resolution,
    )

    assert all(receipt.accepted for receipt in receipts.values())
    result = receiver.last_resolved_result
    assert result is not None
    assert result.target_admission.accepted
    assert result.segment_result.admission is not None
    assert result.segment_result.admission.accepted
    assert result.segment_result.anchor_candidate is None
    assert result.anchor_commit_evidence is not None
    assert result.anchor_commit_evidence.safe_action_hash == (
        resolution.input_safe_action_hash
    )
    assert result.anchor_commit_evidence.effective_from == pytest.approx(10.05)
    segment = result.segment_result.segment
    assert segment is not None
    assert segment.end_parent.source_kind == "resolved_execution_target"
    assert segment.end_parent.source_fact_id == target.target_id
    assert segment.end_parent.projected_group_action_hash == (
        resolution.output_qpos_action_hash
    )
    assert segment.end_parent.projected_group_action_hash != (
        resolution.input_safe_action_hash
    )
    rig.close()
