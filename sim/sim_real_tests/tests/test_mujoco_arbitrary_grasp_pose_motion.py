from __future__ import annotations

from dataclasses import replace
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.path_planning import (
    JointConfigurationCheck,
    JointPathPlannerConfig,
    JointPathPlanResult,
    JointPathStatus,
)
from robot_motion.rm65.grasp_reachability import (
    GraspReachabilityReceipt,
    GraspReachabilityStatus,
)
from robot_motion.shadow_execution import (
    AuthoritativeRobotStateSnapshot,
    RegisteredJointTrajectoryPoint,
    RegisteredShadowJointTrajectory,
    ShadowExecutionIdentity,
    ShadowSimulationStatus,
)
from sim.mujoco.collision_safety import CollisionCheckResult
from sim.mujoco.control.desk_tcp_position import (
    DEFAULT_INSPIRE_V3_DUAL_ARM_URDF,
    DEFAULT_SIM_HOME_QPOS_BY_SIDE,
    DeskTcpPositionControlConfig,
)
from sim.sim_real_tests.mujoco import arbitrary_grasp_pose_motion as motion
from sim.sim_real_tests.mujoco import (
    continuous_random_grasp_pose_motion as random_motion,
)
from sim.sim_real_tests.mujoco.arbitrary_grasp_pose_motion import (
    MOTION_STATUS_ENDPOINT_REJECTED,
    MOTION_STATUS_PATH_COLLISION_BLOCKED,
    GraspMotionPlayback,
    GraspMotionTarget,
    ToolTcpFrameBinding,
    build_shadow_initialized_control,
    build_shadow_simulation_receipt_from_playback,
    build_smooth_se3_trajectory,
    execute_grasp_motion,
    pose_from_xyz_quat_xyzw,
    pose_from_xyz_rpy_deg,
    quintic_progress,
    required_motion_duration_sec,
    shadow_control_initialization_observation,
)
from sim.sim_real_tests.mujoco.continuous_random_grasp_pose_motion import (
    ContinuousRandomGraspPoseConfig,
    RawCartesianGraspTargetStream,
    classify_random_grasp_attempt,
    sample_random_joint_fk_target,
)
from sim.sim_real_tests.mujoco.view_arbitrary_grasp_pose_motion import (
    _obstacle_scene_from_args,
    _uses_continuous_random_mode,
    _validate_args,
    build_parser,
)
from sim.sim_real_tests.rm65 import (
    DEFAULT_REFERENCE_FRAME_ID,
    DEFAULT_REFERENCE_FRAME_SPEC_HASH,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _shadow_identity() -> ShadowExecutionIdentity:
    return ShadowExecutionIdentity(
        station_revision="canonical-station:test",
        reference_frame_id="canonical_station",
        reference_frame_spec_hash="frame-spec:test",
        frame_graph_hash="frame-graph:test",
        tool_profile_hash_by_side={"left": "tool:left", "right": "tool:right"},
        robot_model_hash="robot-model:test",
        collision_model_hash="collision-model:test",
        joint_layout_hash="joint-layout:test",
        gripper_model_hash="gripper-model:test",
    )


def _shadow_home_snapshot() -> AuthoritativeRobotStateSnapshot:
    return AuthoritativeRobotStateSnapshot(
        qpos_by_side=DEFAULT_SIM_HOME_QPOS_BY_SIDE,
        gripper_close_by_side={"left": 0.2, "right": 0.3},
        arm_measured_at=1.0,
        gripper_measured_at_by_side={"left": 1.0, "right": 1.0},
        available_at=1.01,
        arm_evidence_refs=("arm:test",),
        gripper_evidence_refs=("gripper:left", "gripper:right"),
        identity=_shadow_identity(),
    )


def _registered_shadow_trajectory() -> RegisteredShadowJointTrajectory:
    snapshot = _shadow_home_snapshot()
    target = {side: snapshot.qpos_by_side[side].copy() for side in ("left", "right")}
    target["left"][0] += 0.01
    return RegisteredShadowJointTrajectory(
        trajectory_id="trajectory:test",
        created_at=2.0,
        initial_snapshot=snapshot,
        scene_hash="scene:test",
        target_hash="target:test",
        planner_config_hash="planner:test",
        moving_sides=("left",),
        points=(
            RegisteredJointTrajectoryPoint(
                time_from_start_sec=0.0,
                qpos_by_side=snapshot.qpos_by_side,
            ),
            RegisteredJointTrajectoryPoint(
                time_from_start_sec=0.1,
                qpos_by_side=target,
            ),
        ),
    )


@pytest.mark.skipif(
    not DEFAULT_INSPIRE_V3_DUAL_ARM_URDF.is_file(),
    reason="local canonical Inspire-v3 URDF asset is unavailable",
)
def test_shadow_control_cold_starts_from_exact_arm_and_gripper_state() -> None:
    snapshot = _shadow_home_snapshot()
    control = build_shadow_initialized_control(snapshot)
    try:
        observation = shadow_control_initialization_observation(control, snapshot)
        assert observation["exact_state_match"] is True
        assert observation["collision_safe"] is True
        assert observation["maximum_qvel_rad_s"] == pytest.approx(0.0, abs=1e-12)
        assert max(observation["qpos_max_abs_error_rad_by_side"].values()) <= 1e-12
        assert max(observation["gripper_close_abs_error_by_side"].values()) <= 1e-12
    finally:
        control.close()


def test_shadow_control_rejects_realtime_initialization_before_build() -> None:
    with pytest.raises(
        motion.ShadowMujocoInitializationError,
        match="realtime_shadow_initialization_forbidden",
    ):
        build_shadow_initialized_control(
            _shadow_home_snapshot(),
            config=DeskTcpPositionControlConfig(realtime_worker=True),
        )


def test_shadow_control_labels_measured_initial_collision_at_build_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_error = (
        "simulation home qpos is collision-unsafe: "
        "{'geom1': 'left_arm_link3_proxy_collision_0', "
        "'geom2': 'left_arm_link5_collision_0', "
        "'collision_category': 'robot_self'}"
    )

    def reject_collision(_config):
        raise RuntimeError(backend_error)

    monkeypatch.setattr(motion.DeskTcpPositionControl, "build", reject_collision)

    with pytest.raises(
        motion.ShadowMujocoInitializationError,
        match="measured_initial_state_collision_blocked",
    ) as raised:
        build_shadow_initialized_control(_shadow_home_snapshot())

    assert raised.value.reason_code == "measured_initial_state_collision_blocked"
    assert raised.value.details == {
        "backend_error": backend_error,
        "collision_check": {
            "geom1": "left_arm_link3_proxy_collision_0",
            "geom2": "left_arm_link5_collision_0",
            "collision_category": "robot_self",
        },
    }


@pytest.mark.skipif(
    not DEFAULT_INSPIRE_V3_DUAL_ARM_URDF.is_file(),
    reason="local canonical Inspire-v3 URDF asset is unavailable",
)
def test_shadow_control_retries_known_zero_distance_build_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _shadow_home_snapshot()
    backend_error = (
        "simulation home qpos is collision-unsafe: "
        "{'minimum_distance_m': 0.0, 'pair_class': '2:2', "
        "'collision_category': 'robot_self', "
        "'geom1': 'right_arm_link2_collision_0', "
        "'geom2': 'right_arm_link4_collision_0'}"
    )
    original_build = motion.DeskTcpPositionControl.build
    calls: list[DeskTcpPositionControlConfig] = []

    def reject_measured_once(config):
        calls.append(config)
        if len(calls) == 1:
            raise RuntimeError(backend_error)
        return original_build(config)

    monkeypatch.setattr(motion.DeskTcpPositionControl, "build", reject_measured_once)

    control = build_shadow_initialized_control(snapshot)
    try:
        assert len(calls) == 2
        assert calls[0].left_home_qpos == tuple(snapshot.qpos_by_side["left"])
        assert calls[0].right_home_qpos == tuple(snapshot.qpos_by_side["right"])
        assert calls[1].left_home_qpos == tuple(
            float(value) for value in DEFAULT_SIM_HOME_QPOS_BY_SIDE["left"]
        )
        assert calls[1].right_home_qpos == tuple(
            float(value) for value in DEFAULT_SIM_HOME_QPOS_BY_SIDE["right"]
        )
        assert control.config.left_home_qpos == calls[0].left_home_qpos
        assert control.config.right_home_qpos == calls[0].right_home_qpos
        observation = shadow_control_initialization_observation(control, snapshot)
        assert observation["exact_state_match"] is True
        assert observation["collision_safe"] is True
    finally:
        control.close()


def _robot_self_zero_distance_check() -> CollisionCheckResult:
    return CollisionCheckResult(
        safe=False,
        minimum_distance_m=0.0,
        geom1_id=11,
        geom2_id=12,
        geom1_name="right_arm_link2_collision_0",
        geom2_name="right_arm_link4_collision_0",
        body1_name="right_arm_link2",
        body2_name="right_arm_link4",
        pair_class="2:2",
    )


@pytest.mark.parametrize("pair_class", ["1:2", "1:4", "2:2", "4:4"])
def test_zero_distance_mesh_query_recovers_only_with_separated_witness(
    pair_class: str,
) -> None:
    check = replace(_robot_self_zero_distance_check(), pair_class=pair_class)
    recovered = motion._recover_zero_distance_mesh_query(
        check,
        pair_has_contact=False,
        witness_distance_m=0.0387,
        minimum_clearance_m=0.002,
    )

    assert recovered.safe is True
    assert recovered.minimum_distance_m == pytest.approx(0.0387)


def test_zero_distance_mesh_query_stays_blocked_on_contact_or_small_gap() -> None:
    check = _robot_self_zero_distance_check()
    assert (
        motion._recover_zero_distance_mesh_query(
            check,
            pair_has_contact=True,
            witness_distance_m=0.0387,
            minimum_clearance_m=0.002,
        )
        == check
    )
    unsupported = replace(check, pair_class="1:1")
    assert (
        motion._recover_zero_distance_mesh_query(
            unsupported,
            pair_has_contact=False,
            witness_distance_m=0.0387,
            minimum_clearance_m=0.002,
        )
        == unsupported
    )
    assert (
        motion._recover_zero_distance_mesh_query(
            check,
            pair_has_contact=False,
            witness_distance_m=0.001,
            minimum_clearance_m=0.002,
        )
        == check
    )


def test_build_error_artifact_gate_requires_zero_robot_self_distance() -> None:
    details = {
        "collision_check": {
            "minimum_distance_m": 0.0,
            "pair_class": "2:2",
            "collision_category": "robot_self",
        }
    }
    assert motion._is_zero_distance_robot_self_query_artifact(details) is True
    assert (
        motion._is_zero_distance_robot_self_query_artifact(
            {
                "collision_check": {
                    "minimum_distance_m": -0.001,
                    "pair_class": "2:2",
                    "collision_category": "robot_self",
                }
            }
        )
        is False
    )
    assert (
        motion._is_zero_distance_robot_self_query_artifact(
            {
                "collision_check": {
                    "minimum_distance_m": 0.0,
                    "pair_class": "1:2",
                    "collision_category": "robot_environment",
                }
            }
        )
        is False
    )


def test_pose_builders_share_explicit_base_from_dmtac_tcp_semantics() -> None:
    xyz = (0.42, -0.18, 0.51)
    from_rpy = pose_from_xyz_rpy_deg(xyz, (20.0, -35.0, 80.0))
    quaternion = Rotation.from_euler("xyz", (20.0, -35.0, 80.0), degrees=True).as_quat()
    from_quaternion = pose_from_xyz_quat_xyzw(xyz, quaternion)

    np.testing.assert_allclose(from_rpy, from_quaternion, atol=1e-12)
    np.testing.assert_allclose(from_rpy[:3, 3], xyz, atol=0.0)
    with pytest.raises(ValueError, match="nonzero norm"):
        pose_from_xyz_quat_xyzw(xyz, (0.0, 0.0, 0.0, 0.0))


def test_quintic_se3_reference_has_exact_endpoints_and_respects_speed_limits() -> None:
    start = np.eye(4, dtype=float)
    target = pose_from_xyz_rpy_deg((0.36, -0.21, 0.17), (65.0, -30.0, 95.0))
    linear_limit = 0.11
    angular_limit_deg = 38.0
    duration, metadata = required_motion_duration_sec(
        start,
        target,
        linear_speed_m_s=linear_limit,
        angular_speed_deg_s=angular_limit_deg,
        minimum_duration_sec=0.5,
    )
    trajectory = build_smooth_se3_trajectory(
        start,
        target,
        reference_hz=60.0,
        duration_sec=duration,
        terminal_hold_sec=0.0,
    )

    np.testing.assert_allclose(trajectory[0], start, atol=1e-12)
    np.testing.assert_allclose(trajectory[-1], target, atol=1e-12)
    assert quintic_progress(0.0) == 0.0
    assert quintic_progress(1.0) == 1.0
    dt = duration / (len(trajectory) - 1)
    translation_speed = [
        np.linalg.norm(rhs[:3, 3] - lhs[:3, 3]) / dt
        for lhs, rhs in pairwise(trajectory)
    ]
    angular_speed_deg = [
        np.rad2deg(Rotation.from_matrix(lhs[:3, :3].T @ rhs[:3, :3]).magnitude()) / dt
        for lhs, rhs in pairwise(trajectory)
    ]
    assert max(translation_speed) <= linear_limit * 1.0001
    assert max(angular_speed_deg) <= angular_limit_deg * 1.0001
    assert metadata["translation_distance_m"] > 0.4
    assert metadata["rotation_distance_deg"] > 90.0
    # Discrete endpoint speeds converge to the analytical zero derivative.
    assert translation_speed[0] < 2e-4
    assert translation_speed[-1] < 2e-4
    assert angular_speed_deg[0] < 0.02
    assert angular_speed_deg[-1] < 0.02


def test_tool_tcp_binding_converts_actual_dmtac_target_to_solver_frame() -> None:
    single_from_dmtac = np.eye(4, dtype=float)
    single_from_dmtac[:3, 3] = (0.0, 0.0, 0.131)
    binding = ToolTcpFrameBinding(
        side="left",
        single_tool_from_dmtac_tool=single_from_dmtac,
        position_offset_m=0.131,
        orientation_offset_deg=0.0,
    )
    base_from_single = pose_from_xyz_rpy_deg((0.32, -0.08, 0.56), (15.0, 30.0, -25.0))
    base_from_dmtac = base_from_single @ single_from_dmtac

    recovered = binding.solver_target_from_dmtac_target(base_from_dmtac)

    np.testing.assert_allclose(recovered, base_from_single, atol=1e-12)
    assert binding.to_payload()["position_offset_m"] == pytest.approx(0.131)


def test_final_settled_frame_does_not_erase_intermediate_ik_failure() -> None:
    assert (
        motion.classify_motion_status(
            final_endpoint_status="settled",
            endpoint_status_counts={"tracking": 20, "ik_target_unmet": 1, "settled": 5},
        )
        == motion.MOTION_STATUS_IK_TARGET_UNMET
    )
    assert (
        motion.classify_motion_status(
            final_endpoint_status="settled",
            endpoint_status_counts={"tracking": 20, "settled": 5},
        )
        == motion.MOTION_STATUS_COMPLETED
    )


def test_recovered_ik_miss_is_visible_but_not_terminal() -> None:
    assert (
        motion.classify_motion_status(
            final_endpoint_status="settled",
            endpoint_status_counts={"ik_target_unmet": 1, "tracking": 1, "settled": 2},
            unrecovered_ik_target_unmet_count=0,
        )
        == motion.MOTION_STATUS_COMPLETED
    )
    assert (
        motion.classify_motion_status(
            final_endpoint_status="settled",
            endpoint_status_counts={"ik_target_unmet": 2, "settled": 1},
            unrecovered_ik_target_unmet_count=1,
        )
        == motion.MOTION_STATUS_IK_TARGET_UNMET
    )


class _FrozenReferenceRecoveryControl:
    def __init__(self, endpoint_statuses: tuple[str, ...]) -> None:
        self.endpoint_statuses = list(endpoint_statuses)
        self.references: list[object] = []
        self.dispatch_kwargs: list[dict[str, object]] = []

    def solve_and_dispatch(self, reference, **kwargs):
        self.references.append(reference)
        self.dispatch_kwargs.append(dict(kwargs))
        return SimpleNamespace(status="tracking")

    def endpoint_tracking_observation(self) -> dict[str, object]:
        return {"status": self.endpoint_statuses.pop(0)}


def test_frozen_reference_retry_recovers_without_bypassing_lower_owner() -> None:
    control = _FrozenReferenceRecoveryControl(("ik_target_unmet", "tracking"))
    frozen_reference = object()
    settle_calls = 0

    def settle_once() -> int:
        nonlocal settle_calls
        settle_calls += 1
        return 17

    recovery = motion._dispatch_frozen_reference_with_bounded_ik_recovery(
        control,
        frozen_reference,
        inactive_side="right",
        retry_limit=2,
        settle_once=settle_once,
    )

    assert recovery["status"] == "transient_recovered"
    assert recovery["retry_count"] == 1
    assert recovery["physics_ticks"] == 34
    assert recovery["retry_physics_ticks"] == 17
    assert settle_calls == 2
    assert len(control.references) == 2
    assert all(reference is frozen_reference for reference in control.references)
    assert all(
        kwargs
        == {
            "gripper_close_by_side": {"left": 0.0, "right": 0.0},
            "hold_sides": ("right",),
        }
        for kwargs in control.dispatch_kwargs
    )


def test_frozen_reference_retry_exhaustion_remains_unrecovered() -> None:
    control = _FrozenReferenceRecoveryControl(
        ("ik_target_unmet", "ik_target_unmet", "ik_target_unmet")
    )
    recovery = motion._dispatch_frozen_reference_with_bounded_ik_recovery(
        control,
        object(),
        inactive_side="right",
        retry_limit=2,
        settle_once=lambda: 16,
    )

    assert recovery["status"] == "unrecovered"
    assert recovery["unrecovered"] is True
    assert recovery["retry_count"] == 2
    assert recovery["attempt_count"] == 3
    assert len(control.references) == 3


class _FakeScene:
    def get_arm_qpos(self, _side: str) -> np.ndarray:
        return np.zeros(6, dtype=float)


class _FakeRetargeter:
    def tool_tcp_world_targets_for_qpos_pair(
        self, *, left_qpos: np.ndarray, right_qpos: np.ndarray
    ) -> dict[str, np.ndarray]:
        del left_qpos, right_qpos
        return {"left": np.eye(4), "right": np.eye(4)}


class _FakeControl:
    def __init__(self) -> None:
        self.scene = _FakeScene()
        self.retargeter = _FakeRetargeter()
        self.world_to_base_by_side = {"left": np.eye(4), "right": np.eye(4)}
        self.world_from_desk = np.eye(4)
        self.contract = SimpleNamespace(
            timing=SimpleNamespace(ik_hz=60.0, command_hz=120.0, physics_hz=1000.0)
        )
        self.dispatch_count = 0

    def solve_and_dispatch(self, *args, **kwargs):
        del args, kwargs
        self.dispatch_count += 1
        raise AssertionError("rejected endpoint must not dispatch")


def _blocked_receipt() -> GraspReachabilityReceipt:
    return GraspReachabilityReceipt(
        status=GraspReachabilityStatus.COLLISION_BLOCKED,
        reason_code="all_accepted_ik_candidates_collision_blocked",
        proof_scope="endpoint_collision_candidates",
        reference_frame_id=DEFAULT_REFERENCE_FRAME_ID,
        reference_frame_spec_hash=DEFAULT_REFERENCE_FRAME_SPEC_HASH,
        target_pose=tuple(tuple(row) for row in np.eye(4)),
    )


def test_urdf_segment_recheck_uses_exact_point_one_degree_resolution() -> None:
    checked: list[dict[str, np.ndarray]] = []

    class Oracle:
        def check(self, candidate):
            checked.append(candidate)
            return JointConfigurationCheck(
                safe=True,
                collision_check_available=True,
                reason="safe",
                minimum_distance_m=0.05,
            )

    target = np.zeros(6)
    target[0] = np.deg2rad(0.35)
    result = motion._direct_joint_path_oracle_assessment(
        _FakeControl(),
        side="left",
        start_qpos_rad=np.zeros(6),
        target_qpos_rad=target,
        oracle=Oracle(),
    )

    assert result["safe"] is True
    assert result["collision_sample_count"] == 5
    assert len(checked) == 5
    assert max(
        np.rad2deg(checked[index + 1]["left"][0] - checked[index]["left"][0])
        for index in range(4)
    ) <= 0.1


def test_rejected_endpoint_emits_no_motion_or_dispatch(monkeypatch) -> None:
    binding = ToolTcpFrameBinding(
        side="left",
        single_tool_from_dmtac_tool=np.eye(4),
        position_offset_m=0.0,
        orientation_offset_deg=0.0,
    )
    monkeypatch.setattr(
        motion,
        "evaluate_motion_target_endpoint",
        lambda _control, _target: (_blocked_receipt(), binding, None),
    )
    control = _FakeControl()
    target = GraspMotionTarget(
        side="left",
        target_pose_base_from_dmtac_tcp=np.eye(4),
        source="test",
    )

    playback = execute_grasp_motion(control, target)

    assert playback.motion_status == motion.MOTION_STATUS_ENDPOINT_REJECTED
    assert playback.frame_count == 0
    assert playback.summary["motion_emitted"] is False
    assert control.dispatch_count == 0


class _RecoveryExecutionControl(_FakeControl):
    def __init__(self, endpoint_statuses: tuple[str, ...]) -> None:
        super().__init__()
        self.endpoint_statuses = list(endpoint_statuses)
        self.references: list[object] = []
        self.physics_ticks = 0

    def solve_and_dispatch(self, reference, **_kwargs):
        self.dispatch_count += 1
        self.references.append(reference)
        return SimpleNamespace(status="tracking")

    def step_physics(self) -> None:
        self.physics_ticks += 1

    def endpoint_tracking_observation(self) -> dict[str, object]:
        return {"status": self.endpoint_statuses.pop(0)}


def _reachable_receipt_and_binding() -> tuple[
    GraspReachabilityReceipt, ToolTcpFrameBinding
]:
    receipt = GraspReachabilityReceipt(
        status=GraspReachabilityStatus.REACHABLE,
        reason_code="test_reachable",
        proof_scope="test",
        reference_frame_id=DEFAULT_REFERENCE_FRAME_ID,
        reference_frame_spec_hash=DEFAULT_REFERENCE_FRAME_SPEC_HASH,
        target_pose=tuple(tuple(row) for row in np.eye(4)),
        selected_candidate_index=0,
        candidates=(
            {
                "candidate_index": 0,
                "qpos_rad": [0.0] * 6,
            },
        ),
    )
    binding = ToolTcpFrameBinding(
        side="left",
        single_tool_from_dmtac_tool=np.eye(4),
        position_offset_m=0.0,
        orientation_offset_deg=0.0,
    )
    return receipt, binding


def _install_two_reference_motion(monkeypatch) -> None:
    receipt, binding = _reachable_receipt_and_binding()
    monkeypatch.setattr(
        motion,
        "evaluate_motion_target_endpoint",
        lambda _control, _target: (receipt, binding, None),
    )
    monkeypatch.setattr(
        motion,
        "build_collision_checked_joint_fk_reference_plan",
        lambda *_args, **_kwargs: motion.JointFkReferencePlan(
            status="safe",
            pose_desk=(np.eye(4), np.eye(4)),
            qpos_rad=(np.zeros(6), np.zeros(6)),
            metadata={"planner": "test_two_reference_plan"},
        ),
    )
    monkeypatch.setattr(
        motion,
        "required_motion_duration_sec",
        lambda *_args, **_kwargs: (0.25, {"duration_sec": 0.25}),
    )
    monkeypatch.setattr(
        motion,
        "build_smooth_se3_trajectory",
        lambda start, *_args, **_kwargs: (np.asarray(start).copy(),),
    )


def test_execute_reports_transient_recovery_and_continues(monkeypatch) -> None:
    _install_two_reference_motion(monkeypatch)
    control = _RecoveryExecutionControl(
        ("ik_target_unmet", "tracking", "settled", "settled")
    )
    target = GraspMotionTarget(
        side="left",
        target_pose_base_from_dmtac_tcp=np.eye(4),
        source="test_recovery",
    )

    playback = execute_grasp_motion(
        control,
        target,
        ik_target_unmet_retry_limit=2,
    )

    recovery = playback.summary["ik_target_unmet_recovery"]
    assert playback.motion_status == motion.MOTION_STATUS_COMPLETED
    assert playback.endpoint_receipt.status is GraspReachabilityStatus.REACHABLE
    assert control.dispatch_count == 3
    assert control.references[0] is control.references[1]
    assert control.references[2] is not control.references[1]
    assert recovery["transient_recovered_reference_count"] == 1
    assert recovery["unrecovered_reference_count"] == 0
    assert recovery["records"][0]["status"] == "transient_recovered"
    assert playback.summary["planned_reference_frame_count"] == 2
    assert playback.summary["dispatched_reference_count"] == 2


def test_execute_stops_after_unrecovered_reference_without_reclassifying_endpoint(
    monkeypatch,
) -> None:
    _install_two_reference_motion(monkeypatch)
    control = _RecoveryExecutionControl(
        ("ik_target_unmet", "ik_target_unmet", "ik_target_unmet")
    )
    target = GraspMotionTarget(
        side="left",
        target_pose_base_from_dmtac_tcp=np.eye(4),
        source="test_unrecovered",
    )

    playback = execute_grasp_motion(
        control,
        target,
        ik_target_unmet_retry_limit=1,
    )

    recovery = playback.summary["ik_target_unmet_recovery"]
    assert playback.motion_status == motion.MOTION_STATUS_IK_TARGET_UNMET
    assert playback.endpoint_receipt.status is GraspReachabilityStatus.REACHABLE
    assert control.dispatch_count == 2
    assert control.references[0] is control.references[1]
    assert playback.summary["planned_reference_frame_count"] == 2
    assert playback.summary["dispatched_reference_count"] == 1
    assert playback.summary["reference_frame_count"] == 1
    assert playback.summary["trajectory"]["total_reference_duration_sec"] == 0.0
    assert playback.summary["trajectory"]["planned_reference_duration_sec"] > 0.0
    assert recovery["unrecovered_reference_count"] == 1
    assert recovery["records"][0]["status"] == "unrecovered"


def test_cli_requires_position_and_orientation_as_one_explicit_pose() -> None:
    parser = build_parser()

    _validate_args(parser.parse_args([]))
    _validate_args(
        parser.parse_args(
            [
                "--target-xyz-m",
                "0.3",
                "0.1",
                "0.5",
                "--target-rpy-deg",
                "0",
                "90",
                "0",
            ]
        )
    )
    with pytest.raises(ValueError, match="explicit target requires"):
        _validate_args(parser.parse_args(["--target-xyz-m", "0.3", "0.1", "0.5"]))
    with pytest.raises(ValueError, match="explicit target requires"):
        _validate_args(parser.parse_args(["--target-rpy-deg", "0", "90", "0"]))


def test_raw_cartesian_target_stream_is_deterministic_and_never_prefilters() -> None:
    config = ContinuousRandomGraspPoseConfig(
        seed=17,
        position_min_m=(-0.4, -0.3, 0.1),
        position_max_m=(0.2, 0.5, 0.9),
    )
    lhs = RawCartesianGraspTargetStream(side="left", config=config)
    rhs = RawCartesianGraspTargetStream(side="left", config=config)

    lhs_first = lhs.next_target()
    lhs_second = lhs.next_target()
    rhs_first = rhs.next_target()

    np.testing.assert_array_equal(
        lhs_first.target_pose_base_from_dmtac_tcp,
        rhs_first.target_pose_base_from_dmtac_tcp,
    )
    assert not np.array_equal(
        lhs_first.target_pose_base_from_dmtac_tcp,
        lhs_second.target_pose_base_from_dmtac_tcp,
    )
    xyz = lhs_first.target_pose_base_from_dmtac_tcp[:3, 3]
    assert np.all(xyz >= np.asarray(config.position_min_m))
    assert np.all(xyz < np.asarray(config.position_max_m))
    rotation = lhs_first.target_pose_base_from_dmtac_tcp[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert lhs_first.known_qpos_rad is None
    assert lhs_first.metadata["sample_index"] == 0
    assert lhs_first.metadata["rejection_sampled"] is False
    assert len(str(lhs_first.metadata["target_pose_content_hash"])) == 64


def test_joint_fk_target_is_one_draw_without_collision_prefilter(monkeypatch) -> None:
    monkeypatch.setattr(
        random_motion,
        "build_rm65_single_arm_ik_solver",
        lambda _config: SimpleNamespace(
            q_min=np.full(6, -2.0),
            q_max=np.full(6, 2.0),
        ),
    )
    target = sample_random_joint_fk_target(
        _FakeControl(),
        side="left",
        seed=23,
        sample_index=5,
        maximum_joint_offset_deg=30.0,
    )

    assert target.known_qpos_rad is not None
    assert target.metadata["sample_index"] == 5
    assert target.metadata["sample_kind"] == "random_joint_fk_candidate"
    assert target.metadata["rejection_sampled"] is False
    assert target.metadata["selected_max_joint_motion_deg"] <= 30.0


def _random_attempt_playback(
    *,
    reachability_status: GraspReachabilityStatus,
    motion_status: str = MOTION_STATUS_ENDPOINT_REJECTED,
    summary: dict[str, object] | None = None,
    frame_count: int = 0,
) -> GraspMotionPlayback:
    target = GraspMotionTarget(
        side="left",
        target_pose_base_from_dmtac_tcp=np.eye(4),
        source="test_random_cartesian",
        metadata={
            "sample_index": 4,
            "target_pose_content_hash": "a" * 64,
        },
    )
    receipt = GraspReachabilityReceipt(
        status=reachability_status,
        reason_code=f"test_{reachability_status.value}",
        proof_scope="test",
        reference_frame_id=DEFAULT_REFERENCE_FRAME_ID,
        reference_frame_spec_hash=DEFAULT_REFERENCE_FRAME_SPEC_HASH,
        target_pose=tuple(tuple(row) for row in np.eye(4)),
    )
    binding = ToolTcpFrameBinding(
        side="left",
        single_tool_from_dmtac_tool=np.eye(4),
        position_offset_m=0.0,
        orientation_offset_deg=0.0,
    )
    frames = tuple(
        {"left": np.zeros(6), "right": np.zeros(6)} for _ in range(frame_count)
    )
    return GraspMotionPlayback(
        motion_status=motion_status,
        side="left",
        target=target,
        endpoint_receipt=receipt,
        frame_binding=binding,
        target_pose_world_by_side={"left": np.eye(4), "right": np.eye(4)},
        reference_pose_world=(),
        actual_pose_world=(),
        actual_qpos_by_frame=frames,
        summary=summary or {"motion_emitted": False},
    )


def test_current_cartesian_re_ik_playback_cannot_claim_exact_replay_evidence() -> None:
    trajectory = _registered_shadow_trajectory()
    playback = _random_attempt_playback(
        reachability_status=GraspReachabilityStatus.REACHABLE,
        motion_status=motion.MOTION_STATUS_COMPLETED,
        frame_count=len(trajectory.points),
        summary={
            "terminal_hold_sec": 0.5,
            "terminal_hold_confirmed": True,
            "trajectory": {
                "execution_contract": {
                    # Free-form summary flags cannot upgrade the current
                    # playback into typed lower-owned evidence.
                    "planner_execution_certified": True,
                    "exact_planned_qpos_dispatch": True,
                }
            },
        },
    )

    receipt = build_shadow_simulation_receipt_from_playback(
        playback=playback,
        trajectory=trajectory,
        evidence_refs=("mujoco-current-run:test",),
    )

    assert receipt.status is ShadowSimulationStatus.EXECUTION_UNCERTIFIED
    assert receipt.exact_planned_qpos_execution_confirmed is False
    assert receipt.model_identity_confirmed is False
    assert receipt.terminal_hold_confirmed is False
    assert receipt.executed_point_count == 0


def test_path_collision_playback_keeps_distinct_shadow_failure_status() -> None:
    trajectory = _registered_shadow_trajectory()
    playback = _random_attempt_playback(
        reachability_status=GraspReachabilityStatus.REACHABLE,
        motion_status=motion.MOTION_STATUS_PATH_COLLISION_BLOCKED,
    )

    receipt = build_shadow_simulation_receipt_from_playback(
        playback=playback,
        trajectory=trajectory,
        evidence_refs=("mujoco-path-collision:test",),
    )

    assert receipt.status is ShadowSimulationStatus.PATH_COLLISION_BLOCKED
    assert receipt.executed_point_count == 0


def test_random_attempt_classification_never_calls_ik_exhaustion_unreachable() -> None:
    proven = classify_random_grasp_attempt(
        _random_attempt_playback(
            reachability_status=GraspReachabilityStatus.KINEMATICALLY_UNREACHABLE
        )
    )
    exhausted = classify_random_grasp_attempt(
        _random_attempt_playback(
            reachability_status=GraspReachabilityStatus.IK_SEARCH_EXHAUSTED
        )
    )
    collision = classify_random_grasp_attempt(
        _random_attempt_playback(
            reachability_status=GraspReachabilityStatus.COLLISION_BLOCKED
        )
    )

    assert proven.proven_unreachable is True
    assert proven.terminal_classification == "proven_kinematically_unreachable"
    assert exhausted.proven_unreachable is False
    assert exhausted.terminal_classification == (
        "ik_search_exhausted_not_proven_unreachable"
    )
    assert collision.terminal_classification == "endpoint_collision_blocked"
    assert all(not outcome.motion_emitted for outcome in (proven, exhausted, collision))


def test_random_attempt_path_collision_is_distinct_and_emits_zero_motion() -> None:
    outcome = classify_random_grasp_attempt(
        _random_attempt_playback(
            reachability_status=GraspReachabilityStatus.REACHABLE,
            motion_status=MOTION_STATUS_PATH_COLLISION_BLOCKED,
            summary={
                "motion_emitted": False,
                "path_preflight": {"reason_code": "blocked_at_progress_0_4"},
            },
        )
    )

    assert outcome.reachability_status == "reachable"
    assert outcome.terminal_classification == "direct_path_collision_blocked"
    assert outcome.reason_code == "blocked_at_progress_0_4"
    assert outcome.motion_emitted is False


@pytest.mark.parametrize(
    ("motion_status", "classification"),
    (
        (
            motion.MOTION_STATUS_PATH_SEARCH_EXHAUSTED,
            "path_search_exhausted_not_proven_unreachable",
        ),
        (
            motion.MOTION_STATUS_PATH_START_BLOCKED,
            "path_start_collision_blocked",
        ),
        (
            motion.MOTION_STATUS_PATH_TARGET_BLOCKED,
            "path_target_collision_blocked",
        ),
        (motion.MOTION_STATUS_PATH_PLANNER_ERROR, "planner_error"),
    ),
)
def test_obstacle_path_failures_are_typed_zero_motion(
    motion_status: str,
    classification: str,
) -> None:
    outcome = classify_random_grasp_attempt(
        _random_attempt_playback(
            reachability_status=GraspReachabilityStatus.REACHABLE,
            motion_status=motion_status,
        )
    )

    assert outcome.terminal_classification == classification
    assert outcome.motion_emitted is False
    assert outcome.frame_count == 0


class _SequenceOracle:
    scene_hash = "test-scene"

    def __init__(self, checks: list[JointConfigurationCheck]) -> None:
        self.checks = list(checks)
        self.queries: list[dict[str, np.ndarray]] = []

    def check(self, qpos_by_component):
        self.queries.append(
            {key: np.asarray(value).copy() for key, value in qpos_by_component.items()}
        )
        return self.checks.pop(0)


def test_combined_obstacle_oracle_reuses_exact_configuration_checks() -> None:
    oracle = object.__new__(motion._MujocoFclObstacleOracle)
    oracle._check_cache = {}
    queries: list[dict[str, np.ndarray]] = []
    expected = JointConfigurationCheck(safe=True, minimum_distance_m=0.08)

    def check_uncached(qpos_by_component):
        queries.append(
            {key: np.asarray(value).copy() for key, value in qpos_by_component.items()}
        )
        return expected

    oracle._check_uncached = check_uncached
    first = {"left": np.zeros(6), "right": np.ones(6)}
    same = {key: value.copy() for key, value in first.items()}

    assert oracle.check(first) is expected
    assert oracle.check(same) is expected
    assert len(queries) == 1

    same["left"][0] = 0.1
    assert oracle.check(same) is expected
    assert len(queries) == 2


def test_final_obstacle_revalidation_is_dense_and_fail_closed() -> None:
    safe = JointConfigurationCheck(safe=True, minimum_distance_m=0.08)
    blocked = JointConfigurationCheck(
        safe=False,
        minimum_distance_m=0.01,
        reason="collision_clearance_violated",
    )
    oracle = _SequenceOracle([safe, safe, blocked])
    waypoints = (
        {"left": np.zeros(6), "right": np.ones(6)},
        {"left": np.full(6, np.deg2rad(0.3)), "right": np.ones(6)},
    )

    result = motion.revalidate_obstacle_path(
        waypoints,
        oracle=oracle,
        edge_resolution_rad=np.deg2rad(0.1),
    )

    assert result["safe"] is False
    assert result["collision_check_available"] is True
    assert result["reason_code"] == "final_obstacle_revalidation_blocked"
    assert len(oracle.queries) == 3
    assert all(set(query) == {"left", "right"} for query in oracle.queries)
    assert all(np.array_equal(query["right"], np.ones(6)) for query in oracle.queries)


def test_obstacle_plan_is_revalidated_without_a_second_global_search(
    monkeypatch,
) -> None:
    waypoints = (
        {"left": np.zeros(6), "right": np.zeros(6)},
        {"left": np.full(6, 0.2), "right": np.zeros(6)},
    )
    planned_configs: list[JointPathPlannerConfig] = []

    def plan(*_args, planner_config, **_kwargs):
        planned_configs.append(planner_config)
        return JointPathPlanResult(
            status=JointPathStatus.PATH_FOUND,
            start_qpos_by_component=waypoints[0],
            target_qpos_by_component=waypoints[-1],
            movable_components=("left",),
            waypoints=waypoints,
            scene_hash="test-scene",
            reason="bounded_search_found_path",
        )

    final_check = {
        "safe": False,
        "collision_check_available": True,
        "reason_code": "final_obstacle_revalidation_blocked",
    }
    monkeypatch.setattr(motion, "plan_obstacle_aware_joint_path", plan)
    monkeypatch.setattr(
        motion,
        "revalidate_obstacle_path",
        lambda *_args, **_kwargs: final_check,
    )

    path, final, attempts = motion.plan_revalidated_obstacle_aware_joint_path(
        SimpleNamespace(),
        side="left",
        target_qpos_rad=np.full(6, 0.2),
        single_arm_solver=SimpleNamespace(),
        obstacle_oracle=SimpleNamespace(),
        planner_config=JointPathPlannerConfig(shortcut_attempts=8),
    )

    assert path.status is JointPathStatus.PATH_FOUND
    assert final is final_check
    assert len(planned_configs) == 1
    assert planned_configs[0].edge_resolution_rad == pytest.approx(np.deg2rad(1.0))
    assert [attempt["phase"] for attempt in attempts] == ["planning"]


def test_obstacle_search_bounds_stay_local_and_within_joint_limits(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return "planned"

    monkeypatch.setattr(motion, "plan_collision_aware_joint_path", capture)
    control = _FakeControl()
    control.scene.get_arm_qpos = lambda side: (
        np.zeros(6) if side == "left" else np.ones(6)
    )
    padding = np.deg2rad(20.0)
    result = motion.plan_obstacle_aware_joint_path(
        control,
        side="left",
        target_qpos_rad=np.full(6, 0.2),
        single_arm_solver=SimpleNamespace(
            q_min=np.full(6, -1.0), q_max=np.full(6, 1.0)
        ),
        obstacle_oracle=SimpleNamespace(scene_hash="scene"),
        planner_config=JointPathPlannerConfig(sample_padding_rad=padding),
    )

    assert result == "planned"
    np.testing.assert_allclose(
        captured["joint_lower_by_component"]["left"], np.full(6, -padding)
    )
    np.testing.assert_allclose(
        captured["joint_upper_by_component"]["left"], np.full(6, 0.2 + padding)
    )
    assert captured["movable_components"] == ("left",)
    np.testing.assert_array_equal(
        captured["target_qpos_by_component"]["right"], np.ones(6)
    )


def test_execute_path_search_exhaustion_emits_no_motion(monkeypatch) -> None:
    receipt, binding = _reachable_receipt_and_binding()
    endpoint_oracle = SimpleNamespace(scene_hash="test-scene")
    path_oracle = SimpleNamespace(scene_hash="test-scene")
    monkeypatch.setattr(
        motion, "build_obstacle_collision_oracle", lambda *_: endpoint_oracle
    )
    monkeypatch.setattr(
        motion,
        "build_path_obstacle_collision_oracle",
        lambda *_args, **_kwargs: path_oracle,
    )
    monkeypatch.setattr(
        motion, "_MujocoFclObstacleOracle", lambda _control, value: value
    )
    monkeypatch.setattr(
        motion,
        "evaluate_motion_target_endpoint",
        lambda _control, _target, **_kwargs: (
            receipt,
            binding,
            SimpleNamespace(q_min=-np.ones(6), q_max=np.ones(6)),
        ),
    )
    result = JointPathPlanResult(
        status=JointPathStatus.PATH_SEARCH_EXHAUSTED,
        start_qpos_by_component={"left": np.zeros(6), "right": np.zeros(6)},
        target_qpos_by_component={"left": np.zeros(6), "right": np.zeros(6)},
        movable_components=("left",),
        scene_hash="test-scene",
        reason="bounded_search_exhausted",
    )
    monkeypatch.setattr(
        motion, "plan_obstacle_aware_joint_path", lambda *_args, **_kwargs: result
    )
    control = _FakeControl()
    target = GraspMotionTarget(
        side="left",
        target_pose_base_from_dmtac_tcp=np.eye(4),
        source="test",
    )
    scene = SimpleNamespace(
        reference_frame_id=motion.OBSTACLE_SCENE_FRAME_ID,
        reference_frame_spec_hash=motion.OBSTACLE_SCENE_FRAME_SPEC_HASH,
        semantic_content_hash="test-scene",
        to_dict=lambda: {"obstacles": []},
    )

    playback = execute_grasp_motion(control, target, obstacle_scene=scene)

    assert playback.motion_status == motion.MOTION_STATUS_PATH_SEARCH_EXHAUSTED
    assert playback.frame_count == 0
    assert playback.summary["motion_emitted"] is False
    assert control.dispatch_count == 0


def test_cli_builds_repeatable_world_frame_obstacle_cubes() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--obstacle-cube-world-m",
            "-0.4",
            "-0.2",
            "0.4",
            "0.12",
            "--obstacle-cube-world-m",
            "-0.3",
            "-0.1",
            "0.5",
            "0.08",
            "--obstacle-clearance-m",
            "0.015",
        ]
    )
    _validate_args(args)
    scene = _obstacle_scene_from_args(args)

    assert scene is not None
    assert scene.reference_frame_id == motion.OBSTACLE_SCENE_FRAME_ID
    assert scene.minimum_clearance_m == pytest.approx(0.015)
    assert tuple(item.obstacle_id for item in scene.obstacles) == ("cube-0", "cube-1")
    assert scene.obstacles[0].size_xyz_m == pytest.approx((0.12, 0.12, 0.12))
    with pytest.raises(ValueError, match="positive SIDE"):
        _validate_args(
            parser.parse_args(["--obstacle-cube-world-m", "0", "0", "0", "0"])
        )


def test_cli_can_bind_fixed_scene_to_a_deterministic_joint_target() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--side",
            "right",
            "--target-qpos-rad",
            "-1.6",
            "0.8",
            "-2.1",
            "2.2",
            "-0.7",
            "1.2",
            "--include-fixed-scene-obstacles",
            "--planner-seed",
            "7",
            "--planner-max-iterations",
            "80",
            "--planner-max-nodes",
            "120",
        ]
    )

    _validate_args(args)
    scene = _obstacle_scene_from_args(args)

    assert _uses_continuous_random_mode(args) is False
    assert args.planner_seed == 7
    assert args.planner_max_iterations == 80
    assert args.planner_max_nodes == 120
    assert scene is not None
    assert {item.obstacle_id for item in scene.obstacles} == {
        "fixed-scene:camera_pole",
        "fixed-scene:left_camera_pole",
        "fixed-scene:right_camera_pole",
        "fixed-scene:table",
    }


def test_cli_fixed_scene_review_owns_target_and_enables_fixed_obstacles() -> None:
    args = build_parser().parse_args(
        ["--fixed-scene-review", "table-detour-left", "--headless"]
    )

    _validate_args(args)
    scene = _obstacle_scene_from_args(args)

    assert _uses_continuous_random_mode(args) is False
    assert scene is not None
    assert "fixed-scene:table" in {item.obstacle_id for item in scene.obstacles}


def test_continuous_cli_is_default_and_headless_requires_a_finite_count() -> None:
    parser = build_parser()
    continuous = parser.parse_args([])
    assert _uses_continuous_random_mode(continuous) is True
    _validate_args(continuous)

    single = parser.parse_args(["--single-reachable-demo"])
    assert _uses_continuous_random_mode(single) is False
    _validate_args(single)

    with pytest.raises(ValueError, match="headless continuous mode requires"):
        _validate_args(parser.parse_args(["--headless"]))
    finite_headless = parser.parse_args(["--headless", "--random-sample-count", "3"])
    _validate_args(finite_headless)
