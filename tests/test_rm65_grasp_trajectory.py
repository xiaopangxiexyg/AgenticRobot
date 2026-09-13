import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_contracts.pose_frames import load_pose_frame_contract
from robot_motion.path_planning import (
    JointConfigurationCheck,
    JointPathPlannerConfig,
    JointPathPlanResult,
    JointPathStatus,
)
from robot_motion.pose import pose_delta_metrics
from robot_motion.relative_action import xyzquat_to_pose_matrix
from robot_motion.rm65 import (
    DEFAULT_COLLISION_SCENE,
    load_rm65_tool_profile,
    tool_tcp_pose_flange_m_rad,
)
from robot_motion.scene_obstacles import ObstacleSceneSnapshot
from robot_motion.shadow_execution import (
    AuthoritativeRobotStateSnapshot,
    RegisteredJointTrajectoryPoint,
    RegisteredShadowJointTrajectory,
    ShadowExecutionIdentity,
    ShadowSimulationStatus,
)
from sim.contracts import SimulationContract
from sim.mujoco.actuation import (
    cad_bimanual_dmtac_gripper_profile,
    rm65_bimanual_udp_feedback_profile,
)
from sim.mujoco.control.desk_tcp_position import (
    DEFAULT_INSPIRE_V3_DUAL_ARM_URDF,
    DEFAULT_SIMULATION_CONTRACT_PATH,
    DEFAULT_STATION_POSE_CONTRACT_PATH,
    DEFAULT_TOOL_PROFILE_BY_SIDE,
)
from sim.sim_real_tests.mujoco.arbitrary_grasp_pose_motion import (
    OBSTACLE_SCENE_FRAME_ID,
    OBSTACLE_SCENE_FRAME_SPEC_HASH,
    build_obstacle_collision_oracle,
    build_shadow_initialized_control,
)
from sim.sim_real_tests.mujoco.exact_shadow_trajectory import (
    MujocoExactQposExecutionConfig,
    MujocoShadowCompatibilityExpectation,
    execute_registered_shadow_trajectory,
)
from teleop_runtime.skill_deploy import rm65_grasp_trajectory as grasp_trajectory_module
from teleop_runtime.skill_deploy.motion_port import RealAgentJointTrajectory
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspJointTrajectoryPlanner,
    RM65GraspReachabilityConfig,
    RM65GraspTrajectoryPlanningConfig,
    RM65GraspTrajectoryPlanningError,
)
from teleop_runtime.skill_deploy.rm65_lower_assembly import (
    build_exclusive_rm65_skill_target_builder,
)
from teleop_runtime.skill_deploy.rm65_ports import RM65SkillTargetBuilder
from teleop_runtime.skill_deploy.rm65_profile import (
    ExclusiveRM65SkillDeploymentProfile,
    resolve_exclusive_rm65_skill_deployment_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.integration, pytest.mark.contract]
SAFE_QPOS = {
    "left": np.array(
        [
            1.5853514942,
            0.1381550277,
            2.1352611562,
            0.0699109098,
            -0.7493894092,
            -0.0572276017,
        ]
    ),
    "right": np.array(
        [
            -1.4983346097,
            0.0517420315,
            2.1069143897,
            0.0504173256,
            -0.6155243686,
            -0.0529943045,
        ]
    ),
}

# The first pre-IK qpos cut and selected GraspNet target from the bounded
# 2026-09-09 real-motion trace.  The old online target path oscillated before a
# long solver cycle expired its measurement; this fixture remains offline and
# proves that the replacement can fully plan the same request before motion.
FIELD_TRACE_QPOS = {
    "left": np.array(
        [
            2.3890416528,
            0.1777966957,
            2.1860074444,
            1.9001574483,
            -1.4231240444,
            -0.5993635115,
        ]
    ),
    "right": np.array(
        [
            -2.2347546108,
            0.5760459383,
            1.4952934462,
            -2.0872916528,
            -1.5589455263,
            0.8234812338,
        ]
    ),
}
FIELD_TRACE_TARGET_XYZQUAT = (
    -0.1112588662,
    -0.0110550193,
    0.1506269683,
    0.2346013309,
    0.8368666035,
    0.0734558775,
    0.4891019705,
)


def _field_trace_snapshot() -> AuthoritativeRobotStateSnapshot:
    identity = ShadowExecutionIdentity(
        station_revision="field-trace-station:test",
        reference_frame_id="desk",
        reference_frame_spec_hash="field-trace-frame-spec:test",
        frame_graph_hash="field-trace-frame-graph:test",
        tool_profile_hash_by_side={
            "left": "field-trace-tool:left",
            "right": "field-trace-tool:right",
        },
        robot_model_hash="field-trace-robot:test",
        collision_model_hash="field-trace-collision:test",
        joint_layout_hash="field-trace-joints:test",
        gripper_model_hash="field-trace-gripper:test",
    )
    return AuthoritativeRobotStateSnapshot(
        qpos_by_side=FIELD_TRACE_QPOS,
        gripper_close_by_side={"left": 0.2, "right": 0.3},
        arm_measured_at=1.0,
        gripper_measured_at_by_side={"left": 1.0, "right": 1.0},
        available_at=1.01,
        arm_evidence_refs=("field-trace-arm:test",),
        gripper_evidence_refs=(
            "field-trace-gripper:left",
            "field-trace-gripper:right",
        ),
        identity=identity,
    )


def _field_trace_mujoco_expectation(
    *,
    scene_revision: str,
) -> MujocoShadowCompatibilityExpectation:
    station = load_pose_frame_contract(DEFAULT_STATION_POSE_CONTRACT_PATH)
    arm_profile = rm65_bimanual_udp_feedback_profile()
    gripper_profile = cad_bimanual_dmtac_gripper_profile()
    simulation_contract = SimulationContract.load(DEFAULT_SIMULATION_CONTRACT_PATH)
    tool_profile_by_side = {
        side: load_rm65_tool_profile(DEFAULT_TOOL_PROFILE_BY_SIDE[side], side=side)
        for side in ("left", "right")
    }
    return MujocoShadowCompatibilityExpectation(
        station_revision=station.source_id,
        reference_frame_id=station.canonical_absolute_frame,
        desk_from_arm_base_by_side={
            side: tuple(
                station.source.effective_raw_to_canonical_desk_transform(side).reshape(
                    -1
                )
            )
            for side in ("left", "right")
        },
        joint_names_by_side={
            side: arm_profile.joint_names_by_side[side]
            for side in ("left", "right")
        },
        tool_profile_id_by_side={
            "left": "DMTAC_M:left",
            "right": "DMTAC_M:right",
        },
        tool_tcp_pose_flange_m_rad_by_side={
            side: tool_tcp_pose_flange_m_rad(tool_profile_by_side[side])
            for side in ("left", "right")
        },
        gripper_profile_name=gripper_profile.name,
        gripper_side_by_name=gripper_profile.side_by_gripper,
        scene_revision=scene_revision,
        simulation_scene_profile_name=simulation_contract.scene.profile_name,
        collision_profile_name=DEFAULT_COLLISION_SCENE,
        evidence_refs=("field-trace-owner-compatibility:test",),
    )


class _ReachabilityBuilder:
    def __init__(self, realized_pose: np.ndarray) -> None:
        self.realized_pose = np.asarray(realized_pose, dtype=float)

    def set_current_qpos_pair(self, **_kwargs) -> None:
        return None

    def solver_world_targets_for_qpos_pair(self, **_kwargs):
        return {"left": np.eye(4), "right": np.eye(4)}

    def solver_world_target_from_tool_tcp_world_target(self, **kwargs):
        return np.asarray(kwargs["tool_tcp_world_target"], dtype=float)

    def solve_target_pair(self, _target):
        side = type("Side", (), {"qpos": np.zeros(6)})
        return type("Result", (), {"side_results": {"left": side, "right": side}})

    def tool_tcp_world_targets_for_qpos_pair(self, **_kwargs):
        return {"left": np.eye(4), "right": self.realized_pose.copy()}


def _real_profile():
    return resolve_exclusive_rm65_skill_deployment_profile(
        ExclusiveRM65SkillDeploymentProfile(
            arm_ip_assignment_path=(
                REPO_ROOT / "teleop_runtime/configs/hardware/rm65_arm_ips/latest.json"
            ),
            tool_profile_path_by_side={
                side: (
                    REPO_ROOT
                    / f"hardware/rm65/utils/configs/calibrations/rm65_{side}_DMTAC_M.json"
                )
                for side in ("left", "right")
            },
            envelope_profile_path=(
                REPO_ROOT
                / "teleop_runtime/configs/skill_deploy/agent_primitive_right_relative_tight_v1.json"
            ),
        )
    )


@pytest.fixture(scope="module")
def field_trace_plan() -> tuple[RealAgentJointTrajectory, RM65SkillTargetBuilder]:
    profile = _real_profile()
    builder = build_exclusive_rm65_skill_target_builder(profile)
    trajectory = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=builder,
        target_envelope_filter=builder.pre_ik_target_safety_filter,
    )(
        qpos_by_side=FIELD_TRACE_QPOS,
        side="right",
        target_tool_tcp_world=xyzquat_to_pose_matrix(
            FIELD_TRACE_TARGET_XYZQUAT,
            name="field_trace_grasp_target",
        ),
        speed_scale=0.2,
    )
    return trajectory, builder


def test_urdf_grasp_planner_handles_hold_and_moving_paths_without_mujoco() -> None:
    profile = _real_profile()
    builder = build_exclusive_rm65_skill_target_builder(profile)
    planner = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=builder,
        target_envelope_filter=builder.pre_ik_target_safety_filter,
    )
    initial_collision_check = planner.oracle.check(SAFE_QPOS)
    assert initial_collision_check.collision_check_available is True
    assert initial_collision_check.metadata["attached_proxy_geometry_count"] == 2
    current_target = builder.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=SAFE_QPOS["left"],
        right_qpos=SAFE_QPOS["right"],
    )["right"]
    hold = planner(
        qpos_by_side=SAFE_QPOS,
        side="right",
        target_tool_tcp_world=current_target,
        speed_scale=0.25,
    )
    current_reachability = planner.assess_endpoint_ik(
        qpos_by_side=SAFE_QPOS,
        side="right",
        target_tool_tcp_world=current_target,
    )

    moved_qpos = {side: values.copy() for side, values in SAFE_QPOS.items()}
    moved_qpos["right"][0] += 0.04
    moved_target = builder.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=moved_qpos["left"],
        right_qpos=moved_qpos["right"],
    )["right"]
    moving = planner(
        qpos_by_side=SAFE_QPOS,
        side="right",
        target_tool_tcp_world=moved_target,
        speed_scale=0.25,
    )

    assert hold.time_from_start_sec.tolist() == [0.0, 0.25]
    assert current_reachability.classification == "standard_reachable"
    assert moving.time_from_start_sec.size > 2
    assert np.all(np.diff(moving.time_from_start_sec) > 0.0)
    np.testing.assert_allclose(
        moving.qpos_by_side["left"],
        np.broadcast_to(SAFE_QPOS["left"], moving.qpos_by_side["left"].shape),
    )


def test_dense_grasp_path_cannot_bypass_tcp_safety_envelope() -> None:
    profile = resolve_exclusive_rm65_skill_deployment_profile(
        ExclusiveRM65SkillDeploymentProfile(
            arm_ip_assignment_path=(
                REPO_ROOT / "teleop_runtime/configs/hardware/rm65_arm_ips/latest.json"
            ),
            tool_profile_path_by_side={
                side: (
                    REPO_ROOT
                    / f"hardware/rm65/utils/configs/calibrations/rm65_{side}_DMTAC_M.json"
                )
                for side in ("left", "right")
            },
            envelope_profile_path=(
                REPO_ROOT
                / "teleop_runtime/configs/skill_deploy/agent_primitive_right_relative_tight_v1.json"
            ),
        )
    )
    builder = build_exclusive_rm65_skill_target_builder(profile)
    calls = 0

    def envelope_filter(targets):
        nonlocal calls
        calls += 1
        projected = {side: pose.copy() for side, pose in targets.items()}
        if calls == 2:
            projected["right"][0, 3] += 0.01
        return projected

    planner = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=builder,
        target_envelope_filter=envelope_filter,
    )
    moved = {side: qpos.copy() for side, qpos in SAFE_QPOS.items()}
    moved["right"][0] += 0.02

    with pytest.raises(
        RM65GraspTrajectoryPlanningError,
        match="path_target_safety_envelope_blocked",
    ) as raised:
        planner._revalidate((SAFE_QPOS, moved))

    assert raised.value.reason_code == "path_target_safety_envelope_blocked"
    assert raised.value.details["modified_sides"] == ("right",)


def test_field_trace_grasp_target_is_fully_planned_before_motion(
    field_trace_plan: tuple[RealAgentJointTrajectory, RM65SkillTargetBuilder],
) -> None:
    trajectory, builder = field_trace_plan
    target = xyzquat_to_pose_matrix(
        FIELD_TRACE_TARGET_XYZQUAT,
        name="field_trace_grasp_target",
    )

    final_tcp = builder.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=trajectory.qpos_by_side["left"][-1],
        right_qpos=trajectory.qpos_by_side["right"][-1],
    )["right"]
    position_error_m, orientation_error_deg = pose_delta_metrics(final_tcp, target)
    maximum_step_deg = max(
        float(np.rad2deg(np.max(np.abs(np.diff(trajectory.qpos_by_side[side], axis=0)))))
        for side in ("left", "right")
    )

    assert trajectory.time_from_start_sec.size > 2
    assert float(trajectory.time_from_start_sec[-1]) < 30.0
    assert position_error_m <= 0.003
    assert orientation_error_deg <= 1.0
    assert maximum_step_deg <= 0.1
    np.testing.assert_allclose(
        trajectory.qpos_by_side["left"],
        np.broadcast_to(
            FIELD_TRACE_QPOS["left"],
            trajectory.qpos_by_side["left"].shape,
        ),
    )


@pytest.mark.slow_sim
@pytest.mark.skipif(
    not DEFAULT_INSPIRE_V3_DUAL_ARM_URDF.is_file(),
    reason="local canonical Inspire-v3 MuJoCo asset is unavailable",
)
def test_field_trace_grasp_trajectory_executes_as_exact_qpos_in_mujoco(
    field_trace_plan: tuple[RealAgentJointTrajectory, RM65SkillTargetBuilder],
) -> None:
    planned, _builder = field_trace_plan
    obstacle_scene = ObstacleSceneSnapshot(
        reference_frame_id=OBSTACLE_SCENE_FRAME_ID,
        reference_frame_spec_hash=OBSTACLE_SCENE_FRAME_SPEC_HASH,
        revision=0,
    )
    snapshot = _field_trace_snapshot()
    registered = RegisteredShadowJointTrajectory(
        trajectory_id="field-trace-grasp:exact-qpos",
        created_at=2.0,
        initial_snapshot=snapshot,
        scene_hash=obstacle_scene.semantic_content_hash,
        target_hash="field-trace-graspnet-candidate-0",
        planner_config_hash=planned.semantic_content_hash,
        moving_sides=("right",),
        points=tuple(
            RegisteredJointTrajectoryPoint(
                time_from_start_sec=float(planned.time_from_start_sec[index]),
                qpos_by_side={
                    side: planned.qpos_by_side[side][index]
                    for side in ("left", "right")
                },
            )
            for index in range(planned.time_from_start_sec.size)
        ),
    )
    control = build_shadow_initialized_control(snapshot)
    control.shadow_obstacle_oracle = build_obstacle_collision_oracle(
        control,
        obstacle_scene,
    )
    try:
        receipt = execute_registered_shadow_trajectory(
            control=control,
            trajectory=registered,
            model_binding=_field_trace_mujoco_expectation(
                scene_revision=obstacle_scene.semantic_content_hash,
            ),
            config=MujocoExactQposExecutionConfig(
                tracking_tolerance_rad=5e-3,
                final_settle_timeout_sec=8.0,
                terminal_hold_timeout_sec=1.0,
                terminal_hold_dwell_sec=0.02,
                terminal_velocity_tolerance_rad_s=5e-3,
            ),
        )
    finally:
        control.close()

    assert receipt.status is ShadowSimulationStatus.COMPLETED
    assert receipt.executed_point_count == len(registered.points)
    assert receipt.exact_planned_qpos_execution_confirmed is True
    assert receipt.model_identity_confirmed is True
    assert receipt.terminal_hold_confirmed is True


@pytest.mark.parametrize(
    ("translation_m", "rotation_deg", "expected"),
    [
        (0.001, 0.5, "standard_reachable"),
        (0.03, 10.0, "basically_reachable"),
        (0.06, 25.0, "difficult_to_reach"),
    ],
)
def test_reachability_classification_uses_one_solution_residual_pair(
    translation_m: float,
    rotation_deg: float,
    expected: str,
) -> None:
    profile = _real_profile()
    initial_builder = build_exclusive_rm65_skill_target_builder(profile)
    planner = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=initial_builder,
        target_envelope_filter=None,
        reachability_config=RM65GraspReachabilityConfig(max_solve_cycles=1),
    )
    realized = np.eye(4)
    realized[0, 3] = translation_m
    angle = np.deg2rad(rotation_deg)
    realized[:2, :2] = ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    planner.target_builder = _ReachabilityBuilder(realized)

    result = planner.assess_endpoint_ik(
        qpos_by_side={side: np.zeros(6) for side in ("left", "right")},
        side="right",
        target_tool_tcp_world=np.eye(4),
        monotonic=lambda: 0.0,
    )

    assert result.classification == expected
    assert result.position_residual_m == pytest.approx(translation_m)
    assert result.orientation_residual_deg == pytest.approx(rotation_deg)
    assert result.solve_cycles == 1


def test_reachability_stops_on_wall_time_budget_and_returns_best_basic_solution() -> (
    None
):
    profile = _real_profile()
    initial_builder = build_exclusive_rm65_skill_target_builder(profile)
    planner = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=initial_builder,
        target_envelope_filter=None,
        reachability_config=RM65GraspReachabilityConfig(
            time_budget_ms=1_000.0,
            max_solve_cycles=120,
        ),
    )
    realized = np.eye(4)
    realized[0, 3] = 0.03
    planner.target_builder = _ReachabilityBuilder(realized)
    times = iter((0.0, 2.0, 2.0, 2.0))

    result = planner.assess_endpoint_ik(
        qpos_by_side={side: np.zeros(6) for side in ("left", "right")},
        side="right",
        target_tool_tcp_world=np.eye(4),
        monotonic=lambda: next(times),
    )

    assert result.classification == "basically_reachable"
    assert result.termination_reason == "time_budget_exhausted"
    assert result.solve_cycles == 1


class _MultistartBuilder:
    def __init__(self) -> None:
        self.current = {side: np.zeros(6) for side in ("left", "right")}
        self.seed_history: list[dict[str, np.ndarray]] = []

    def set_current_qpos_pair(self, *, left_qpos, right_qpos, **_kwargs) -> None:
        self.current = {
            "left": np.asarray(left_qpos, dtype=float).copy(),
            "right": np.asarray(right_qpos, dtype=float).copy(),
        }
        self.seed_history.append(
            {side: value.copy() for side, value in self.current.items()}
        )

    def solver_world_targets_for_qpos_pair(self, **_kwargs):
        return {"left": np.eye(4), "right": np.eye(4)}

    def solver_world_target_from_tool_tcp_world_target(self, **kwargs):
        return np.asarray(kwargs["tool_tcp_world_target"], dtype=float)

    def solve_target_pair(self, _target):
        side_results = {
            side: SimpleNamespace(qpos=value.copy())
            for side, value in self.current.items()
        }
        return SimpleNamespace(side_results=side_results)

    def tool_tcp_world_targets_for_qpos_pair(self, **_kwargs):
        return {"left": np.eye(4), "right": np.eye(4)}


def _planner_without_assets(
    *,
    config: RM65GraspTrajectoryPlanningConfig,
) -> RM65GraspJointTrajectoryPlanner:
    planner = object.__new__(RM65GraspJointTrajectoryPlanner)
    planner.config = config
    planner.reachability_config = RM65GraspReachabilityConfig()
    planner.profile = SimpleNamespace(
        station_world_frame_id="canonical_station",
        position_profile=SimpleNamespace(
            rates=SimpleNamespace(ik_hz=20.0),
            ik_execution=SimpleNamespace(
                joint_velocity_limit_profile="test-velocity"
            ),
        ),
    )
    planner.target_envelope_filter = None
    planner._solver_lock = threading.Lock()
    return planner


def test_endpoint_planner_collects_distinct_exact_multistart_solutions() -> None:
    planner = _planner_without_assets(
        config=RM65GraspTrajectoryPlanningConfig(
            maximum_endpoint_solve_cycles=1,
            endpoint_seed_count=5,
        )
    )
    builder = _MultistartBuilder()
    planner.target_builder = builder
    planner.oracle = SimpleNamespace(
        check=lambda _qpos: JointConfigurationCheck(safe=True)
    )
    current = {side: np.zeros(6) for side in ("left", "right")}

    candidates = planner._endpoint_qpos_candidates(
        qpos_by_side=current,
        side="right",
        target_tool_tcp_world=np.eye(4),
    )

    assert [candidate.seed_index for candidate in candidates] == [0, 1, 2, 3, 4]
    assert all(candidate.solve_cycles == 1 for candidate in candidates)
    assert all(
        np.array_equal(candidate.qpos_by_side["left"], current["left"])
        for candidate in candidates
    )
    assert len({tuple(candidate.qpos_by_side["right"]) for candidate in candidates}) == 5
    np.testing.assert_array_equal(builder.seed_history[-1]["right"], current["right"])


def test_endpoint_planner_reports_best_multistart_residuals_on_exhaustion() -> None:
    planner = _planner_without_assets(
        config=RM65GraspTrajectoryPlanningConfig(
            maximum_endpoint_solve_cycles=1,
            endpoint_seed_count=3,
        )
    )
    planner.target_builder = _MultistartBuilder()
    planner.oracle = SimpleNamespace(
        check=lambda _qpos: JointConfigurationCheck(safe=True)
    )
    current = {side: np.zeros(6) for side in ("left", "right")}
    target = np.eye(4, dtype=float)
    target[0, 3] = 0.1

    with pytest.raises(
        RM65GraspTrajectoryPlanningError,
        match="ik_search_exhausted",
    ) as error:
        planner._endpoint_qpos_candidates(
            qpos_by_side=current,
            side="right",
            target_tool_tcp_world=target,
        )

    assert error.value.details["cycles"] == 3
    assert error.value.details["best_position_residual_m"] == pytest.approx(0.1)
    assert error.value.details["best_orientation_residual_deg"] == pytest.approx(0.0)
    assert error.value.details["endpoint_seed_count"] == 3
    assert len(error.value.details["attempts"]) == 3


def test_endpoint_planner_uses_all_structured_seed_basins_by_default() -> None:
    assert RM65GraspTrajectoryPlanningConfig().endpoint_seed_count == 7


def test_path_planner_widens_corridor_then_uses_alternate_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _planner_without_assets(
        config=RM65GraspTrajectoryPlanningConfig(
            endpoint_seed_count=2,
            path_search_attempts_per_endpoint=4,
        )
    )
    planner.planner_config = JointPathPlannerConfig(
        max_iterations=1,
        max_nodes=2,
        shortcut_attempts=0,
    )
    planner.scene = SimpleNamespace(semantic_content_hash="scene:test")
    planner.oracle = SimpleNamespace()
    planner.target_builder = _MultistartBuilder()
    current = {side: np.zeros(6) for side in ("left", "right")}
    first_target = {side: value.copy() for side, value in current.items()}
    first_target["right"] = np.full(6, 0.1)
    second_target = {side: value.copy() for side, value in current.items()}
    second_target["right"] = np.full(6, 0.2)
    endpoints = (
        grasp_trajectory_module._EndpointIKCandidate(
            qpos_by_side=first_target,
            seed_index=0,
            solve_cycles=10,
            position_residual_m=0.001,
            orientation_residual_deg=0.2,
        ),
        grasp_trajectory_module._EndpointIKCandidate(
            qpos_by_side=second_target,
            seed_index=1,
            solve_cycles=12,
            position_residual_m=0.001,
            orientation_residual_deg=0.2,
        ),
    )
    planner._endpoint_qpos_candidates = lambda **_kwargs: endpoints
    planner._revalidate = lambda _waypoints: {
        "sample_count": 2,
        "minimum_distance_m": 0.03,
    }
    calls: list[dict[str, object]] = []

    def fake_plan(**kwargs):
        calls.append(kwargs)
        target = kwargs["target_qpos_by_component"]
        found = np.array_equal(target["right"], second_target["right"])
        return JointPathPlanResult(
            status=(
                JointPathStatus.PATH_FOUND
                if found
                else JointPathStatus.PATH_SEARCH_EXHAUSTED
            ),
            start_qpos_by_component=kwargs["start_qpos_by_component"],
            target_qpos_by_component=target,
            movable_components=("right",),
            waypoints=(kwargs["start_qpos_by_component"], target) if found else (),
            scene_hash="scene:test",
            reason=None if found else "bounded_search_exhausted",
        )

    monkeypatch.setattr(
        grasp_trajectory_module,
        "plan_collision_aware_joint_path",
        fake_plan,
    )
    monkeypatch.setattr(
        grasp_trajectory_module,
        "load_joint_velocity_limit_profiles",
        lambda: {
            "test-velocity": SimpleNamespace(max_velocity_deg_s=(30.0,) * 6)
        },
    )

    trajectory = planner(
        qpos_by_side=current,
        side="right",
        target_tool_tcp_world=np.eye(4),
        speed_scale=0.2,
    )

    assert len(calls) == 5
    first_widths = [
        np.asarray(call["joint_upper_by_component"]["right"])
        - np.asarray(call["joint_lower_by_component"]["right"])
        for call in calls[:4]
    ]
    assert np.all(first_widths[1] > first_widths[0])
    assert np.all(first_widths[2] >= first_widths[1])
    assert np.any(first_widths[2] > first_widths[1])
    assert np.all(first_widths[3] >= first_widths[2])
    np.testing.assert_allclose(trajectory.qpos_by_side["right"][-1], 0.2)
