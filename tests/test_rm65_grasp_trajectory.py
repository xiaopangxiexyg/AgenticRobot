import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_contracts.pose_frames import load_pose_frame_contract
from robot_motion.joint_limits import load_joint_velocity_limit_profiles
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
    RM65_GRASP_ENDPOINT_ORDER_ESTIMATED_EXECUTION_TIME,
    RM65_GRASP_ENDPOINT_ORDER_JOINT_DISTANCE,
    RM65_GRASP_TRAJECTORY_PARAMETERIZATION_CONTINUOUS_C2,
    RM65_GRASP_TRAJECTORY_PARAMETERIZATION_SEGMENTED_STOP,
    RM65_GRASP_TRAJECTORY_SPEED_PROFILE_FAR_4X_APPROACH_1X,
    RM65_GRASP_TRAJECTORY_SPEED_PROFILE_UNIFORM,
)
from teleop_runtime.skill_deploy.rm65_lower_assembly import (
    build_exclusive_rm65_skill_target_builder,
)
from teleop_runtime.skill_deploy.rm65_ports import RM65SkillTargetBuilder
from teleop_runtime.skill_deploy.rm65_profile import (
    ExclusiveRM65SkillDeploymentProfile,
    resolve_exclusive_rm65_skill_deployment_profile,
)
from teleop_runtime.skill_deploy.rm65_target_safety import (
    ExclusiveRM65SkillTargetSafetyRuntime,
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

# Read-only state and selected candidate 3 from the 2026-09-14 10:56
# real-camera capture.  The right TCP starts 154.16 mm beyond the task
# envelope's y-max; the 160 mm ingress cap admits this measured staging pose
# while preserving monotonic envelope entry and dense collision revalidation.
OFFLINE_INGRESS_QPOS = {
    "left": np.array(
        [
            2.2549480297663664,
            0.4383219939384942,
            2.2667114786173554,
            1.8238690683340744,
            -1.3990559923145207,
            -0.41160100069361605,
        ]
    ),
    "right": np.array(
        [
            -2.441157052133034,
            0.5387831547380355,
            1.8238167372186542,
            -2.1215349775880297,
            -1.4772815556456362,
            0.6232221724979286,
        ]
    ),
}
OFFLINE_INGRESS_TARGET = np.array(
    [
        [
            0.7536026758524672,
            -0.6323477267158206,
            0.17949751938490785,
            -0.1292053338337324,
        ],
        [
            0.13750307481077162,
            -0.11537878437095266,
            -0.9837584259027543,
            -0.07293043669658317,
        ],
        [
            0.6427876098413727,
            0.7660444429890577,
            -3.865527780625074e-09,
            0.14860936225424548,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Exact read-only qpos and candidate 3 from the 2026-09-14 13:49 blue-cube
# capture.  Candidate 0 exposed a large TCP-envelope excursion along a
# collision-clear joint interpolation; candidate 3 provides a short strict
# route from the same measured HOLD posture.
LATEST_BLUE_CUBE_QPOS = {
    "left": np.array(
        [
            2.254965340313452,
            0.4383045502333542,
            2.2665719289762354,
            1.8238690683340744,
            -1.398881422105066,
            -0.41174061691376346,
        ]
    ),
    "right": np.array(
        [
            -2.441104854175668,
            0.5387307903331019,
            1.8238516246289345,
            -2.1214477590623293,
            -1.4772641119404961,
            0.6231174436880611,
        ]
    ),
}
LATEST_BLUE_CUBE_CANDIDATE_3 = np.array(
    [
        [
            0.7702649959168447,
            0.2803535459146707,
            -0.5727946624736426,
            -0.33077911609627286,
        ],
        [
            -0.5382509172192118,
            -0.19590731346984203,
            -0.819698892668342,
            0.05540680494080945,
        ],
        [
            -0.34202015463697494,
            0.9396926166689296,
            7.224301935798734e-09,
            0.1543208538538673,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Exact measured HOLD state and selected candidate 0 from the 2026-09-19
# blue-cube run.  The terminal grasp itself is reachable within the strict
# 3 mm / 1 degree contract.  The old planner nevertheless rejected every
# pregrasp distance because it also imposed the terminal 1-degree limit on
# the intermediate pregrasp and approach waypoints.
RECORDED_CANDIDATE_0_QPOS = {
    "left": np.array(
        [
            2.3181637508776864,
            0.6254911191676488,
            1.9098614746682967,
            2.0429602738380046,
            -1.5188728081796763,
            -0.4476071383685691,
        ]
    ),
    "right": np.array(
        [
            -2.4927142529403135,
            0.5396732497533673,
            1.826958335198566,
            -2.0757026410180384,
            -1.4138213563462472,
            0.3305478926541362,
        ]
    ),
}
RECORDED_CANDIDATE_0_TARGET = np.array(
    [
        [
            0.7544351172666891,
            0.6563746291827534,
            0.0,
            -0.3478102578835386,
        ],
        [
            0.6563746291827535,
            -0.7544351172666891,
            0.0,
            0.040734236366776144,
        ],
        [0.0, 0.0, -1.0, 0.16809674027061464],
        [0.0, 0.0, 0.0, 1.0],
    ]
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
            side: arm_profile.joint_names_by_side[side] for side in ("left", "right")
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


def _real_profile(
    *,
    envelope_profile_name: str = "agent_primitive_right_relative_tight_v1.json",
):
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
                / "teleop_runtime/configs/skill_deploy"
                / envelope_profile_name
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


@pytest.fixture(scope="module")
def continuous_field_trace_plan() -> tuple[
    RealAgentJointTrajectory,
    RM65SkillTargetBuilder,
]:
    profile = _real_profile()
    builder = build_exclusive_rm65_skill_target_builder(profile)
    trajectory = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=builder,
        target_envelope_filter=builder.pre_ik_target_safety_filter,
        config=RM65GraspTrajectoryPlanningConfig(
            trajectory_parameterization=(
                RM65_GRASP_TRAJECTORY_PARAMETERIZATION_CONTINUOUS_C2
            ),
            endpoint_candidate_ordering=(
                RM65_GRASP_ENDPOINT_ORDER_ESTIMATED_EXECUTION_TIME
            ),
        ),
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


def test_grasp_oracle_uses_ik_sparse_proxy_pairs_and_reports_coverage():
    profile = _real_profile()
    builder = build_exclusive_rm65_skill_target_builder(profile)
    planner = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=builder,
        target_envelope_filter=builder.pre_ik_target_safety_filter,
    )
    from robot_motion.rm65.scene_profiles import DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS

    sparse_model = planner.oracle._collision_model
    sparse_names = {
        frozenset((
            sparse_model.geometryObjects[int(pair.first)].name,
            sparse_model.geometryObjects[int(pair.second)].name,
        ))
        for pair in sparse_model.collisionPairs
    }
    assert {
        frozenset(pair) for pair in DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS
    } <= sparse_names
    assert frozenset(("right_arm_link3_0", "right_arm_link5_0")) not in sparse_names
    check = planner.oracle.check(FIELD_TRACE_QPOS)
    assert check.collision_check_available
    assert check.metadata["collision_scope"] == "profile_sparse_proxy_pairs"
    assert check.metadata["collision_pair_count"] == len(sparse_model.collisionPairs)


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
        planner._revalidate((SAFE_QPOS, moved), moving_side="right")

    assert raised.value.reason_code == "path_target_safety_envelope_blocked"
    assert raised.value.details["modified_sides"] == ("right",)
    assert (
        raised.value.details["envelope_ingress_rejection"]
        == "projection_diagnostics_unavailable"
    )


class _IngressEnvelopeFilter:
    def __init__(
        self,
        adjustments_m: tuple[float, ...],
        *,
        constraints_by_call: tuple[tuple[str, ...], ...] | None = None,
    ) -> None:
        self.adjustments_m = adjustments_m
        self.constraints_by_call = constraints_by_call
        self.calls = 0
        self._projection = None

    def __call__(self, targets):
        adjustment = self.adjustments_m[min(self.calls, len(self.adjustments_m) - 1)]
        constraints = (
            ("absolute:y:max",)
            if self.constraints_by_call is None
            else self.constraints_by_call[
                min(self.calls, len(self.constraints_by_call) - 1)
            ]
        )
        self.calls += 1
        projected = {side: pose.copy() for side, pose in targets.items()}
        raw_pose = {side: np.eye(4) for side in ("left", "right")}
        effective_pose = {side: np.eye(4) for side in ("left", "right")}
        raw_pose["right"][1, 3] = 0.1 + adjustment
        effective_pose["right"][1, 3] = 0.1
        if adjustment > 0.0:
            projected["right"][0, 3] += 0.01
        self._projection = {
            "raw_tcp_pose_by_side": {
                side: pose.tolist() for side, pose in raw_pose.items()
            },
            "effective_tcp_pose_by_side": {
                side: pose.tolist() for side, pose in effective_pose.items()
            },
            "side_projection_by_side": {
                "left": {
                    "modified": False,
                    "translation_modified": False,
                    "rotation_modified": False,
                    "translation_adjustment_m": 0.0,
                    "rotation_adjustment_rad": 0.0,
                    "violated_translation_constraints": (),
                    "active_translation_constraints": (),
                },
                "right": {
                    "modified": adjustment > 0.0,
                    "translation_modified": adjustment > 0.0,
                    "rotation_modified": False,
                    "translation_adjustment_m": adjustment,
                    "rotation_adjustment_rad": 0.0,
                    "violated_translation_constraints": (
                        constraints if adjustment > 0.0 else ()
                    ),
                    "active_translation_constraints": (
                        constraints if adjustment > 0.0 else ()
                    ),
                },
            },
        }
        return projected

    def snapshot(self):
        return SimpleNamespace(projection=self._projection)


def test_dense_grasp_path_accepts_bounded_monotonic_envelope_ingress() -> None:
    profile = _real_profile()
    builder = build_exclusive_rm65_skill_target_builder(profile)
    envelope_filter = _IngressEnvelopeFilter((0.12, 0.0))
    planner = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=builder,
        target_envelope_filter=envelope_filter,
    )

    result = planner._revalidate(
        (SAFE_QPOS, SAFE_QPOS),
        moving_side="right",
    )

    assert result["sample_count"] == 2
    assert result["envelope_ingress"] == {
        "side": "right",
        "initial_translation_adjustment_m": 0.12,
        "violated_translation_constraints": ["absolute:y:max"],
        "observed_translation_constraints": ["absolute:y:max"],
        "entered_at": {"segment_index": 0, "edge_sample_index": 1},
    }


def test_dense_grasp_path_rejects_motion_away_from_envelope() -> None:
    profile = _real_profile()
    builder = build_exclusive_rm65_skill_target_builder(profile)
    envelope_filter = _IngressEnvelopeFilter((0.12, 0.119, 0.1215))
    planner = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=builder,
        target_envelope_filter=envelope_filter,
    )

    with pytest.raises(
        RM65GraspTrajectoryPlanningError,
        match="path_target_safety_envelope_blocked",
    ) as raised:
        planner._revalidate(
            (SAFE_QPOS, SAFE_QPOS, SAFE_QPOS),
            moving_side="right",
        )

    assert (
        raised.value.details["envelope_ingress_rejection"]
        == "path_moved_away_from_envelope"
    )


def test_dense_grasp_path_allows_submillimeter_tangent_during_ingress() -> None:
    profile = _real_profile()
    builder = build_exclusive_rm65_skill_target_builder(profile)
    envelope_filter = _IngressEnvelopeFilter(
        (0.12, 0.00025, 0.0),
        constraints_by_call=(
            ("absolute:y:max",),
            ("absolute:x:min", "absolute:y:max"),
            (),
        ),
    )
    planner = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=builder,
        target_envelope_filter=envelope_filter,
    )

    result = planner._revalidate(
        (SAFE_QPOS, SAFE_QPOS, SAFE_QPOS),
        moving_side="right",
    )

    assert result["envelope_ingress"]["observed_translation_constraints"] == [
        "absolute:x:min",
        "absolute:y:max",
    ]


def test_dense_grasp_path_rejects_submillimeter_excursion_after_starting_inside() -> (
    None
):
    profile = _real_profile()
    builder = build_exclusive_rm65_skill_target_builder(profile)
    envelope_filter = _IngressEnvelopeFilter(
        (0.0, 0.000335),
        constraints_by_call=((), ("absolute:x:min",)),
    )
    planner = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=builder,
        target_envelope_filter=envelope_filter,
    )

    with pytest.raises(RM65GraspTrajectoryPlanningError) as raised:
        planner._revalidate((SAFE_QPOS, SAFE_QPOS), moving_side="right")

    assert (
        raised.value.details["envelope_ingress_rejection"]
        == "path_did_not_start_at_the_violation"
    )


def test_dense_grasp_path_rejects_larger_new_constraint_during_ingress() -> None:
    profile = _real_profile()
    builder = build_exclusive_rm65_skill_target_builder(profile)
    envelope_filter = _IngressEnvelopeFilter(
        (0.12, 0.0008),
        constraints_by_call=(
            ("absolute:y:max",),
            ("absolute:x:min", "absolute:y:max"),
        ),
    )
    planner = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=builder,
        target_envelope_filter=envelope_filter,
    )

    with pytest.raises(RM65GraspTrajectoryPlanningError) as raised:
        planner._revalidate(
            (SAFE_QPOS, SAFE_QPOS),
            moving_side="right",
        )

    assert (
        raised.value.details["envelope_ingress_rejection"]
        == "violation_constraint_set_expanded"
    )


@pytest.mark.parametrize(
    ("details", "expected_rejection"),
    [
        (
            {"modified_sides": ("left", "right")},
            "modified_sides_are_not_the_moving_side_only",
        ),
        (
            {
                "modified_sides": ("right",),
                "envelope_projection_by_side": {
                    "right": {
                        "translation_modified": True,
                        "rotation_modified": True,
                        "translation_adjustment_m": 0.1,
                        "violated_translation_constraints": ["absolute:y:max"],
                    }
                },
            },
            "violation_is_not_translation_only",
        ),
        (
            {
                "modified_sides": ("right",),
                "envelope_projection_by_side": {
                    "right": {
                        "translation_modified": True,
                        "rotation_modified": False,
                        "translation_adjustment_m": 0.161,
                        "violated_translation_constraints": ["absolute:y:max"],
                    }
                },
            },
            "initial_distance_exceeds_limit",
        ),
        (
            {
                "modified_sides": ("right",),
                "envelope_projection_by_side": {
                    "right": {
                        "translation_modified": True,
                        "rotation_modified": False,
                        "translation_adjustment_m": 0.1,
                        "violated_translation_constraints": [
                            "absolute:x:min",
                            "absolute:y:max",
                        ],
                    }
                },
            },
            "multiple_initial_constraints_are_not_supported",
        ),
    ],
)
def test_envelope_ingress_requires_one_bounded_moving_arm_translation_violation(
    details: dict[str, object],
    expected_rejection: str,
) -> None:
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())

    with pytest.raises(RM65GraspTrajectoryPlanningError) as raised:
        planner._bounded_ingress_projection(
            moving_side="right",
            details=details,
        )

    assert raised.value.reason_code == "path_target_safety_envelope_blocked"
    assert raised.value.details["envelope_ingress_rejection"] == expected_rejection


@pytest.mark.parametrize("adjustment", [0.159, 0.16, 0.1607339041])
def test_start_envelope_preflight_distinguishes_current_tcp_from_grasp_target(
    monkeypatch, adjustment,
) -> None:
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    targets = {side: np.eye(4) for side in ("left", "right")}
    planner.target_builder = SimpleNamespace(solver_world_targets_for_qpos_pair=lambda **_: targets)
    planner.target_envelope_filter = object()
    projected = {side: pose.copy() for side, pose in targets.items()}
    projected["right"][1, 3] = -adjustment
    monkeypatch.setattr(planner, "_project_target_envelope", lambda *_args, **_: projected)
    monkeypatch.setattr(planner, "_target_envelope_projection_details", lambda *_: {
        "envelope_projection_by_side": {"right": {
            "translation_modified": True, "rotation_modified": False,
            "translation_adjustment_m": adjustment,
            "violated_translation_constraints": ["absolute:y:max"],
        }}
    })
    qpos = {side: np.zeros(6) for side in targets}
    if adjustment <= 0.16:
        planner._preflight_start_envelope(qpos_by_side=qpos, side="right")
    else:
        with pytest.raises(RM65GraspTrajectoryPlanningError) as raised:
            planner._preflight_start_envelope(qpos_by_side=qpos, side="right")
        assert raised.value.reason_code == "start_state_safety_envelope_blocked"
        assert raised.value.details["envelope_projection_subject"] == "current_measured_tool_tcp_in_envelope_reference_frame"
        assert raised.value.details["maximum_path_envelope_ingress_distance_m"] == 0.16


def test_envelope_ingress_interpolates_actual_tool_tcp_space() -> None:
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    builder = _ToolOffsetCartesianBuilder()
    planner.target_builder = builder
    planner.target_envelope_filter = _ToolSpaceAbsoluteEnvelopeFilter(
        builder,
        minimum_x=-0.39,
    )
    planner._target_envelope_projection_details = lambda _modified: {
        "envelope_projection_by_side": {
            "right": {
                "translation_modified": True,
                "rotation_modified": False,
                "translation_adjustment_m": 0.1217,
                "violated_translation_constraints": ["absolute:x:min"],
            }
        }
    }
    current = {
        "left": np.zeros(6),
        "right": np.array([-0.3717, 0.0, 0.0, 0.0, -np.pi / 2.0, 0.0]),
    }
    final_tool = np.eye(4)
    final_tool[0, 3] = -0.28

    ingress = planner._envelope_ingress_targets(
        qpos_by_side=current,
        side="right",
        final_tool_tcp_world=final_tool,
    )

    assert ingress
    fraction, first_tool = ingress[0]
    assert fraction == pytest.approx(0.05)
    # The first ingress waypoint must be inside the actual tool-TCP envelope.
    assert first_tool[0, 3] == pytest.approx(-0.3845, abs=1e-8)
    assert first_tool[0, 3] > -0.39
    solver_waypoint = builder.solver_world_target_from_tool_tcp_world_target(
        side="right",
        tool_tcp_world_target=first_tool,
    )
    roundtrip_tool = builder.tool_tcp_world_target_from_solver_world_target(
        side="right",
        solver_world_target=solver_waypoint,
    )
    assert roundtrip_tool[0, 3] > -0.39


def test_real_snapshot_plans_via_strict_envelope_ingress_waypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _real_profile(envelope_profile_name="agent_grasp_right_workspace_v1.json")
    builder = build_exclusive_rm65_skill_target_builder(profile)
    target_safety = ExclusiveRM65SkillTargetSafetyRuntime.from_profile_path(
        target_builder=builder,
        station_world_from_arm_base_by_side=profile.scene_world_from_arm_base_by_side,
        station_world_frame_id=profile.station_world_frame_id,
        envelope_profile_path=profile.envelope_profile_path,
    )
    envelope_filter = target_safety.envelope_filter
    planner = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=builder,
        target_envelope_filter=envelope_filter,
    )
    phases: list[str] = []
    original_plan_path_leg = planner._plan_path_leg

    def tracked_plan_path_leg(**kwargs):
        phases.append(kwargs["planning_phase"])
        return original_plan_path_leg(**kwargs)

    monkeypatch.setattr(planner, "_plan_path_leg", tracked_plan_path_leg)
    start_solver = builder.solver_world_targets_for_qpos_pair(
        left_qpos=OFFLINE_INGRESS_QPOS["left"],
        right_qpos=OFFLINE_INGRESS_QPOS["right"],
    )
    projected_start = envelope_filter(start_solver)
    initial_projection = envelope_filter.snapshot().projection

    assert not np.allclose(projected_start["right"], start_solver["right"])
    assert initial_projection["side_projection_by_side"]["right"][
        "translation_adjustment_m"
    ] == pytest.approx(0.15415882876337186)
    assert initial_projection["side_projection_by_side"]["right"][
        "violated_translation_constraints"
    ] == ("absolute:y:max",)

    trajectory = planner(
        qpos_by_side=OFFLINE_INGRESS_QPOS,
        side="right",
        target_tool_tcp_world=OFFLINE_INGRESS_TARGET,
        speed_scale=0.2,
    )

    assert phases[:2] == ["envelope_ingress", "pregrasp_target"]
    assert trajectory.time_from_start_sec.size > 2
    assert np.all(np.diff(trajectory.time_from_start_sec) > 0.0)
    np.testing.assert_allclose(
        trajectory.qpos_by_side["left"],
        np.broadcast_to(
            OFFLINE_INGRESS_QPOS["left"],
            trajectory.qpos_by_side["left"].shape,
        ),
    )
    final_tcp = builder.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=trajectory.qpos_by_side["left"][-1],
        right_qpos=trajectory.qpos_by_side["right"][-1],
    )["right"]
    position_error_m, orientation_error_deg = pose_delta_metrics(
        final_tcp,
        OFFLINE_INGRESS_TARGET,
    )
    sampled_velocity = (
        np.abs(np.diff(trajectory.qpos_by_side["right"], axis=0))
        / np.diff(trajectory.time_from_start_sec)[:, None]
    )
    velocity_limit = (
        np.deg2rad(
            np.asarray(
                load_joint_velocity_limit_profiles()[
                    profile.position_profile.ik_execution.joint_velocity_limit_profile
                ].max_velocity_deg_s,
                dtype=float,
            )
        )
        * 0.2
    )

    assert position_error_m <= planner.config.endpoint_position_tolerance_m
    assert orientation_error_deg <= planner.config.endpoint_orientation_tolerance_deg
    assert np.all(sampled_velocity <= velocity_limit + 1e-9)

    entered = False
    for index in range(trajectory.time_from_start_sec.size):
        qpos = {
            side: trajectory.qpos_by_side[side][index] for side in ("left", "right")
        }
        solver_targets = builder.solver_world_targets_for_qpos_pair(
            left_qpos=qpos["left"],
            right_qpos=qpos["right"],
        )
        projected = envelope_filter(solver_targets)
        modified = not np.allclose(
            projected["right"],
            solver_targets["right"],
            atol=1e-9,
            rtol=0.0,
        )
        if entered:
            assert modified is False
        elif not modified:
            entered = True
    assert entered is True


def test_latest_blue_cube_candidate_has_short_strict_read_only_path() -> None:
    profile = _real_profile(envelope_profile_name="agent_grasp_right_workspace_v1.json")
    builder = build_exclusive_rm65_skill_target_builder(profile)
    target_safety = ExclusiveRM65SkillTargetSafetyRuntime.from_profile_path(
        target_builder=builder,
        station_world_from_arm_base_by_side=profile.scene_world_from_arm_base_by_side,
        station_world_frame_id=profile.station_world_frame_id,
        envelope_profile_path=profile.envelope_profile_path,
    )
    envelope_filter = target_safety.envelope_filter
    planner = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=builder,
        target_envelope_filter=envelope_filter,
    )

    trajectory = planner(
        qpos_by_side=LATEST_BLUE_CUBE_QPOS,
        side="right",
        target_tool_tcp_world=LATEST_BLUE_CUBE_CANDIDATE_3,
        speed_scale=0.2,
    )

    assert trajectory.time_from_start_sec.size > 2
    assert float(trajectory.time_from_start_sec[-1]) < 35.0
    final_tcp = builder.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=trajectory.qpos_by_side["left"][-1],
        right_qpos=trajectory.qpos_by_side["right"][-1],
    )["right"]
    position_error_m, orientation_error_deg = pose_delta_metrics(
        final_tcp,
        LATEST_BLUE_CUBE_CANDIDATE_3,
    )
    assert position_error_m <= planner.config.endpoint_position_tolerance_m
    assert orientation_error_deg <= planner.config.endpoint_orientation_tolerance_deg

    entered = False
    for index in range(trajectory.time_from_start_sec.size):
        solver_targets = builder.solver_world_targets_for_qpos_pair(
            left_qpos=trajectory.qpos_by_side["left"][index],
            right_qpos=trajectory.qpos_by_side["right"][index],
        )
        projected = envelope_filter(solver_targets)
        modified = not np.allclose(
            projected["right"],
            solver_targets["right"],
            atol=1e-9,
            rtol=0.0,
        )
        if entered:
            assert modified is False
        elif not modified:
            entered = True
    assert entered is True


def test_latest_blue_cube_candidate_supports_continuous_c2_parameterization() -> None:
    profile = _real_profile(envelope_profile_name="agent_grasp_right_workspace_v1.json")
    builder = build_exclusive_rm65_skill_target_builder(profile)
    target_safety = ExclusiveRM65SkillTargetSafetyRuntime.from_profile_path(
        target_builder=builder,
        station_world_from_arm_base_by_side=profile.scene_world_from_arm_base_by_side,
        station_world_frame_id=profile.station_world_frame_id,
        envelope_profile_path=profile.envelope_profile_path,
    )
    planner = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=builder,
        target_envelope_filter=target_safety.envelope_filter,
        config=RM65GraspTrajectoryPlanningConfig(
            trajectory_parameterization=(
                RM65_GRASP_TRAJECTORY_PARAMETERIZATION_CONTINUOUS_C2
            ),
            endpoint_candidate_ordering=(
                RM65_GRASP_ENDPOINT_ORDER_ESTIMATED_EXECUTION_TIME
            ),
        ),
    )

    trajectory = planner(
        qpos_by_side=LATEST_BLUE_CUBE_QPOS,
        side="right",
        target_tool_tcp_world=LATEST_BLUE_CUBE_CANDIDATE_3,
        speed_scale=0.2,
    )

    assert trajectory.time_from_start_sec.size > 2
    assert float(trajectory.time_from_start_sec[-1]) < 35.0
    assert np.all(np.diff(trajectory.time_from_start_sec) > 0.0)
    final_tcp = builder.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=trajectory.qpos_by_side["left"][-1],
        right_qpos=trajectory.qpos_by_side["right"][-1],
    )["right"]
    position_error_m, orientation_error_deg = pose_delta_metrics(
        final_tcp,
        LATEST_BLUE_CUBE_CANDIDATE_3,
    )
    assert position_error_m <= planner.config.endpoint_position_tolerance_m
    assert orientation_error_deg <= planner.config.endpoint_orientation_tolerance_deg


def test_real_snapshot_ingress_does_not_admit_an_outside_grasp_target() -> None:
    profile = _real_profile(envelope_profile_name="agent_grasp_right_workspace_v1.json")
    builder = build_exclusive_rm65_skill_target_builder(profile)
    target_safety = ExclusiveRM65SkillTargetSafetyRuntime.from_profile_path(
        target_builder=builder,
        station_world_from_arm_base_by_side=profile.scene_world_from_arm_base_by_side,
        station_world_frame_id=profile.station_world_frame_id,
        envelope_profile_path=profile.envelope_profile_path,
    )
    planner = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=builder,
        target_envelope_filter=target_safety.envelope_filter,
    )
    outside_target = OFFLINE_INGRESS_TARGET.copy()
    outside_target[0, 3] = 0.5

    with pytest.raises(
        RM65GraspTrajectoryPlanningError,
        match="target_safety_envelope_blocked",
    ) as raised:
        planner(
            qpos_by_side=OFFLINE_INGRESS_QPOS,
            side="right",
            target_tool_tcp_world=outside_target,
            speed_scale=0.2,
        )

    assert raised.value.reason_code == "target_safety_envelope_blocked"
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
        float(
            np.rad2deg(np.max(np.abs(np.diff(trajectory.qpos_by_side[side], axis=0))))
        )
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
@pytest.mark.parametrize(
    "plan_fixture_name",
    ("field_trace_plan", "continuous_field_trace_plan"),
)
def test_field_trace_grasp_trajectory_executes_as_exact_qpos_in_mujoco(
    request,
    plan_fixture_name: str,
) -> None:
    planned, _builder = request.getfixturevalue(plan_fixture_name)
    obstacle_scene = ObstacleSceneSnapshot(
        reference_frame_id=OBSTACLE_SCENE_FRAME_ID,
        reference_frame_spec_hash=OBSTACLE_SCENE_FRAME_SPEC_HASH,
        revision=0,
    )
    snapshot = _field_trace_snapshot()
    registered = RegisteredShadowJointTrajectory(
        trajectory_id=f"field-trace-grasp:{plan_fixture_name}:exact-qpos",
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
    # These residuals come from the controlled builder below, not a real model.
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    planner.reachability_config = RM65GraspReachabilityConfig(
        max_solve_cycles=1,
        seed_count=1,
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
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    planner.reachability_config = RM65GraspReachabilityConfig(
        time_budget_ms=1_000.0,
        max_solve_cycles=120,
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


def test_reachability_multistart_respects_total_evaluation_budget(
    monkeypatch,
) -> None:
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    planner.reachability_config = RM65GraspReachabilityConfig(
        time_budget_ms=1_000.0,
        max_solve_cycles=5,
        max_solve_evaluations_per_seed=3,
        seed_count=3,
    )
    requested_budgets = []
    seeds = tuple(
        {side: np.full(6, index, dtype=float) for side in ("left", "right")}
        for index in range(3)
    )
    monkeypatch.setattr(
        planner,
        "_endpoint_seed_qpos",
        lambda *_args, **_kwargs: seeds,
    )

    def solve(**kwargs):
        budget = kwargs["maximum_evaluations"]
        requested_budgets.append(budget)
        return ({}, budget, 0.06, 25.0)

    monkeypatch.setattr(planner, "_least_squares_endpoint_from_seed", solve)

    result = planner.assess_endpoint_ik(
        qpos_by_side={side: np.zeros(6) for side in ("left", "right")},
        side="right",
        target_tool_tcp_world=np.eye(4),
        monotonic=lambda: 0.0,
    )

    assert requested_budgets == [3, 2]
    assert result.solve_cycles == 5
    assert result.termination_reason == "cycle_budget_exhausted"


def test_reachability_reports_seed_budget_without_invalidating_result(
    monkeypatch,
) -> None:
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    planner.reachability_config = RM65GraspReachabilityConfig(
        time_budget_ms=1_000.0,
        max_solve_cycles=10,
        max_solve_evaluations_per_seed=3,
        seed_count=2,
    )
    seeds = tuple(
        {side: np.full(6, index, dtype=float) for side in ("left", "right")}
        for index in range(2)
    )
    monkeypatch.setattr(
        planner,
        "_endpoint_seed_qpos",
        lambda *_args, **_kwargs: seeds,
    )
    monkeypatch.setattr(
        planner,
        "_least_squares_endpoint_from_seed",
        lambda **_kwargs: ({}, 1, 0.06, 25.0),
    )

    result = planner.assess_endpoint_ik(
        qpos_by_side={side: np.zeros(6) for side in ("left", "right")},
        side="right",
        target_tool_tcp_world=np.eye(4),
        monotonic=lambda: 0.0,
    )

    assert result.solve_cycles == 2
    assert result.termination_reason == "seed_budget_exhausted"


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


class _LinearQposBuilder(_MultistartBuilder):
    @staticmethod
    def _poses(*, left_qpos, right_qpos):
        poses = {"left": np.eye(4), "right": np.eye(4)}
        poses["left"][0, 3] = float(np.asarray(left_qpos)[0])
        poses["right"][0, 3] = float(np.asarray(right_qpos)[0])
        return poses

    def solver_world_targets_for_qpos_pair(self, **kwargs):
        return self._poses(**kwargs)

    def tool_tcp_world_targets_for_qpos_pair(self, **kwargs):
        return self._poses(**kwargs)


class _CartesianQposBuilder(_MultistartBuilder):
    @staticmethod
    def _poses(*, left_qpos, right_qpos):
        poses = {"left": np.eye(4), "right": np.eye(4)}
        for side, qpos in (("left", left_qpos), ("right", right_qpos)):
            values = np.asarray(qpos, dtype=float)
            poses[side][:3, :3] = grasp_trajectory_module.Rotation.from_rotvec(
                values[3:]
            ).as_matrix()
            poses[side][:3, 3] = values[:3]
        return poses

    def solver_world_targets_for_qpos_pair(self, **kwargs):
        return self._poses(**kwargs)

    def tool_tcp_world_targets_for_qpos_pair(self, **kwargs):
        return self._poses(**kwargs)


class _ToolOffsetCartesianBuilder(_CartesianQposBuilder):
    """Small synthetic builder exposing the solver-endpoint/tool offset bug."""

    @staticmethod
    def _tool_offset() -> np.ndarray:
        offset = np.eye(4)
        offset[2, 3] = 0.14
        return offset

    def tool_tcp_world_target_from_solver_world_target(
        self,
        *,
        side,
        solver_world_target,
    ):
        assert side in {"left", "right"}
        return np.asarray(solver_world_target, dtype=float) @ self._tool_offset()

    def solver_world_target_from_tool_tcp_world_target(
        self,
        *,
        side,
        tool_tcp_world_target,
    ):
        assert side in {"left", "right"}
        return np.asarray(tool_tcp_world_target, dtype=float) @ np.linalg.inv(
            self._tool_offset()
        )

    def tool_tcp_world_targets_for_qpos_pair(self, **kwargs):
        return {
            side: self.tool_tcp_world_target_from_solver_world_target(
                side=side,
                solver_world_target=pose,
            )
            for side, pose in self._poses(**kwargs).items()
        }


class _ToolSpaceAbsoluteEnvelopeFilter:
    def __init__(self, builder: _ToolOffsetCartesianBuilder, minimum_x: float) -> None:
        self.builder = builder
        self.minimum_x = float(minimum_x)

    def __call__(self, targets):
        projected = {
            side: np.asarray(pose, dtype=float).copy()
            for side, pose in targets.items()
        }
        for side, solver_pose in targets.items():
            tool_pose = self.builder.tool_tcp_world_target_from_solver_world_target(
                side=side,
                solver_world_target=solver_pose,
            )
            if float(tool_pose[0, 3]) < self.minimum_x:
                effective_tool_pose = tool_pose.copy()
                effective_tool_pose[0, 3] = self.minimum_x
                projected[side] = (
                    self.builder.solver_world_target_from_tool_tcp_world_target(
                        side=side,
                        tool_tcp_world_target=effective_tool_pose,
                    )
                )
        return projected


class _MinimumRightXEnvelopeFilter:
    def __init__(self, minimum_x: float) -> None:
        self.minimum_x = float(minimum_x)

    def __call__(self, targets):
        projected = {
            side: np.asarray(pose, dtype=float).copy() for side, pose in targets.items()
        }
        projected["right"][0, 3] = max(
            self.minimum_x,
            float(projected["right"][0, 3]),
        )
        return projected


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
            ik_execution=SimpleNamespace(joint_velocity_limit_profile="test-velocity"),
        ),
    )
    planner.target_envelope_filter = None
    planner.path_collision_scene_profile = "test_sparse_proxy"
    planner.trajectory_record_dir = None
    planner._solver_lock = threading.Lock()
    return planner


@pytest.mark.parametrize("atol", (1e-9, 1e-5))
def test_joint_proposal_dedup_matches_original_absolute_tolerance(atol):
    random = np.random.default_rng(45)
    lower = np.asarray(grasp_trajectory_module.RM65_ABSOLUTE_Q_MIN_RAD)
    upper = np.asarray(grasp_trajectory_module.RM65_ABSOLUTE_Q_MAX_RAD)
    centers = [np.zeros(6), lower, upper, *random.uniform(lower, upper, (40, 6))]
    for first in centers:
        for joint in range(6):
            for delta in (
                0.0,
                np.nextafter(atol, 0.0),
                atol,
                np.nextafter(atol, np.inf),
                -atol,
                -2 * atol,
            ):
                second = first.copy()
                second[joint] += delta
                before = first.copy(), second.copy()
                expected = np.allclose(first, second, atol=atol, rtol=0.0)
                assert grasp_trajectory_module._joint_proposals_match(
                    first, second, atol=atol
                ) == expected
                np.testing.assert_array_equal(first, before[0])
                np.testing.assert_array_equal(second, before[1])


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize("count", (1, 8, 24, 64))
def test_endpoint_seed_dedup_preserves_sequence_and_detached_guesses(monkeypatch, side, count):
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    lower = np.asarray(grasp_trajectory_module.RM65_ABSOLUTE_Q_MIN_RAD)
    upper = np.asarray(grasp_trajectory_module.RM65_ABSOLUTE_Q_MAX_RAD)
    original_match = grasp_trajectory_module._joint_proposals_match
    default_pair = {
        name: np.asarray(getattr(grasp_trajectory_module, f"DEFAULT_{name.upper()}_ARM_QPOS"))
        for name in ("left", "right")
    }
    for pair in (
        FIELD_TRACE_QPOS,
        default_pair,
        {"left": lower.copy(), "right": upper.copy()},
    ):
        before = {name: value.copy() for name, value in pair.items()}
        # The canonical NumPy predicate reconstructs the old sequence using
        # the unchanged local/default/Halton seed-generation algorithm.
        with monkeypatch.context() as context:
            context.setattr(
                grasp_trajectory_module,
                "_joint_proposals_match",
                lambda a, b, *, atol: bool(np.allclose(a, b, atol=atol, rtol=0.0)),
            )
            expected = planner._endpoint_seed_qpos(pair, side=side, seed_count=count)
        actual = planner._endpoint_seed_qpos(pair, side=side, seed_count=count)
        assert grasp_trajectory_module._joint_proposals_match is original_match
        assert len(actual) == len(expected) == count
        for first, second in zip(actual, expected, strict=True):
            for name in ("left", "right"):
                np.testing.assert_array_equal(first[name], second[name])
                assert not np.shares_memory(first[name], pair[name])
        actual[0][side][0] = 0.0
        for name, value in pair.items():
            np.testing.assert_array_equal(value, before[name])
        assert np.all(actual[-1][side] >= lower) and np.all(actual[-1][side] <= upper)


@pytest.mark.parametrize("phase", ["envelope_ingress", "pregrasp_target", "grasp_target"])
@pytest.mark.parametrize("obstruction", ["clear", "visible_target", "visible_background", "visible_start", "production"])
def test_path_leg_search_uses_actual_composed_oracle_without_duplicate_scene_binding(phase, obstruction):
    from robot_motion.grasp_visible_scene import ParallelJawSolidGeometry, VisibleGraspScene

    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig(
        path_search_attempts_per_endpoint=4 if obstruction == "visible_start" else 1,
    ))
    planner.planner_config = JointPathPlannerConfig()
    planner.scene = SimpleNamespace(semantic_content_hash="production:test")
    checked = []

    class ProductionOracle:
        scene_hash = "production:test"

        def check(self, qpos):
            checked.append(qpos["right"].copy())
            return JointConfigurationCheck(
                safe=not (obstruction == "production" and qpos["right"][0] > .05),
                collision_check_available=True,
                reason="production_block" if obstruction == "production" else None,
            )

    class Builder:
        def tool_tcp_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
            poses = {side: np.eye(4) for side in ("left", "right")}
            poses["right"][0, 3] = right_qpos[0]
            return poses

    planner.oracle = ProductionOracle()
    planner.target_builder = Builder()
    planner.target_envelope_filter = lambda targets: targets
    planner._require_qpos_within_target_envelope = lambda *_args, **_kwargs: None
    planner._revalidate = lambda *_args, **_kwargs: {}
    scene = VisibleGraspScene(
        points=np.array([[.04 if obstruction == "visible_start" else .14, 0, -.02]])
        if obstruction.startswith("visible") else np.empty((0, 3)),
        target_mask=np.array([obstruction == "visible_target"], dtype=bool)
        if obstruction.startswith("visible") else np.empty(0, dtype=bool),
        geometry=ParallelJawSolidGeometry(-.043, .05, .01, .0255, .07),
    )
    start = {side: np.zeros(6) for side in ("left", "right")}
    target = {side: q.copy() for side, q in start.items()}
    target["right"][0] = .1
    endpoint = grasp_trajectory_module._EndpointIKCandidate(
        qpos_by_side=target, seed_index=0, solve_cycles=1,
        position_residual_m=0, orientation_residual_deg=0,
    )
    arguments = dict(qpos_by_side=start, side="right", target_tool_tcp_world=np.eye(4),
                     planning_phase=phase, visible_scene=scene, endpoint_candidates=(endpoint,))
    if obstruction == "clear":
        leg = planner._plan_path_leg(**arguments)
        assert leg.path.status is JointPathStatus.PATH_FOUND
        assert leg.path.scene_hash == planner._path_search_oracle(
            planning_phase=phase, visible_scene=scene, side="right",
            start_qpos_by_side=start,
            terminal_tool_tcp_world=(
                np.eye(4) if phase == "grasp_target" else None
            ),
        ).scene_hash
        assert leg.path.scene_hash != planner.scene.semantic_content_hash
        assert leg.path.metadata["requested_scene_hash"] is None
        assert len(leg.search_attempts) == 1
    elif obstruction == "visible_start":
        with pytest.raises(RM65GraspTrajectoryPlanningError, match="path_start_collision_blocked") as error:
            planner._plan_path_leg(**arguments)
        assert len(error.value.details["search_attempts"]) == 1
        assert error.value.details["path"]["status"] == JointPathStatus.START_BLOCKED.value
        assert "visible_gripper_path_collision" in str(error.value.details["path"])
    else:
        with pytest.raises(RM65GraspTrajectoryPlanningError, match="path_target_collision_blocked"):
            planner._plan_path_leg(**arguments)
    assert checked


@pytest.mark.parametrize("reject_path", [False, True])
def test_sparse_proxy_search_still_requires_dense_revalidation(reject_path):
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig(
        path_search_attempts_per_endpoint=1,
    ))
    planner.planner_config = JointPathPlannerConfig(shortcut_attempts=0)
    planner.scene = SimpleNamespace(minimum_clearance_m=0.01)
    checked = []

    class ProbeOracle:
        scene_hash = "same-obstacle-scene"

        def check(self, qpos):
            checked.append(qpos["right"].copy())
            return JointConfigurationCheck(safe=True, collision_check_available=True)

    planner.oracle = ProbeOracle()
    validations = []

    def revalidate(*_args, **_kwargs):
        validations.append(True)
        if reject_path:
            raise RM65GraspTrajectoryPlanningError("path_collision_blocked")
        return {"sample_count": 1}

    planner._revalidate = revalidate
    start = {side: np.zeros(6) for side in ("left", "right")}
    target = {side: q.copy() for side, q in start.items()}
    target["right"][0] = 0.1
    endpoint = grasp_trajectory_module._EndpointIKCandidate(
        qpos_by_side=target, seed_index=0, solve_cycles=1,
        position_residual_m=0.0, orientation_residual_deg=0.0,
    )
    kwargs = dict(
        qpos_by_side=start, side="right", target_tool_tcp_world=np.eye(4),
        planning_phase="pregrasp_target", endpoint_candidates=(endpoint,),
    )
    if reject_path:
        with pytest.raises(RM65GraspTrajectoryPlanningError, match="path_collision_blocked") as error:
            planner._plan_path_leg(**kwargs)
        assert error.value.details["search_attempts"][0]["search_collision_model"] == "ik_sparse_proxy_pairs"
    else:
        leg = planner._plan_path_leg(**kwargs)
        assert leg.path.status is JointPathStatus.PATH_FOUND
        assert len(leg.search_attempts) == 1
        assert leg.search_attempts[0]["search_collision_model"] == "ik_sparse_proxy_pairs"
    assert checked
    assert len(validations) == 1


@pytest.mark.parametrize("has_capture_qpos", [False, True])
@pytest.mark.parametrize("failure_reason", [
    "path_collision_check_unavailable",
    "endpoint_collision_check_unavailable",
    "approach_endpoint_safety_envelope_unavailable",
])
def test_grasp_planner_filters_once_at_capture_pose_before_path_search(has_capture_qpos, failure_reason):
    from robot_motion.grasp_visible_scene import ParallelJawSolidGeometry, VisibleGraspScene

    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    planner._solver_lock = threading.Lock()
    class Builder:
        def tool_tcp_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
            poses = {side: np.eye(4) for side in ("left", "right")}
            poses["right"][0, 3] = right_qpos[0]
            poses["left"][0, 3] = .5 + left_qpos[0]
            return poses
        def solver_world_targets_for_qpos_pair(self, **qpos):
            return self.tool_tcp_world_targets_for_qpos_pair(**qpos)
        def solver_world_target_from_tool_tcp_world_target(self, *, side, tool_tcp_world_target):
            return tool_tcp_world_target
    planner.target_builder = Builder()
    planner._require_unmodified_target_envelope = lambda *_args, **_kwargs: None
    planner._preflight_start_envelope = lambda **_kwargs: None
    capture = {side: np.zeros(6) for side in ("left", "right")}
    current = {side: q.copy() for side, q in capture.items()}
    current["right"][0] = .2
    view = VisibleGraspScene(points=np.array([[.04, 0, -.02], [.54, 0, -.02], [.44, 0, -.02]]),
                             target_mask=np.zeros(3, dtype=bool),
                             geometry=ParallelJawSolidGeometry(-.043, .05, .01, .0255, .07),
                             capture_qpos_by_side=capture if has_capture_qpos else None)
    checked = []
    def inspect_chain(**arguments):
        filtered = arguments["visible_scene"]
        np.testing.assert_array_equal(filtered.points, view.points[[2]] if has_capture_qpos else view.points)
        np.testing.assert_array_equal(arguments["qpos_by_side"]["right"], current["right"])
        checked.append(filtered)
        raise RM65GraspTrajectoryPlanningError(failure_reason)
    planner._plan_chain_for_pregrasp = inspect_chain
    target = np.eye(4)
    target[0, 3] = .8
    with pytest.raises(RM65GraspTrajectoryPlanningError, match=failure_reason):
        planner(qpos_by_side=current, side="right", target_tool_tcp_world=target,
                speed_scale=.2, visible_scene=view, monotonic=lambda: 0)
    assert len(checked) == 1
    assert len(view.points) == 3


def test_real_grasp_capture_self_filter_margin_covers_measured_six_millimeters():
    assert (
        grasp_trajectory_module.RM65_GRASP_CAPTURE_SELF_OBSERVATION_SURFACE_MARGIN_M
        == pytest.approx(.006)
    )


def test_visible_grasp_diagnostics_separate_terminal_and_straight_ingress_collision():
    from robot_motion.grasp_visible_scene import (
        ParallelJawSolidGeometry,
        VisibleGraspScene,
    )

    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    scene = VisibleGraspScene(
        points=np.array([[.04, 0.0, -.08]]),
        target_mask=np.array([False]),
        geometry=ParallelJawSolidGeometry(-.043, .05, .01, .0255, .07),
    )

    diagnostics = planner._visible_grasp_collision_diagnostics(
        scene,
        grasp_tool_tcp_world=np.eye(4),
        distance_m=.06,
    )

    assert diagnostics["terminal_target_noncontact_points"] == 0
    assert diagnostics["terminal_background_points"] == 0
    assert diagnostics["approach_background_points"] == 1
    assert diagnostics["straight_ingress_only"] is True
    assert diagnostics["terminal_collision_blocked"] is False


def test_straight_ingress_collision_defers_to_global_grasp_target_search(monkeypatch):
    from robot_motion.grasp_visible_scene import (
        ParallelJawSolidGeometry,
        VisibleGraspScene,
    )

    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig(
        allow_unconstrained_terminal_fallback=True,
    ))
    planner.scene = SimpleNamespace(semantic_content_hash="scene:test")
    planner.profile.position_profile.ik_execution.joint_velocity_limit_profile = (
        "test-velocity"
    )
    current = {side: np.zeros(6) for side in ("left", "right")}

    class Builder:
        def solver_world_targets_for_qpos_pair(self, **_kwargs):
            return {side: np.eye(4) for side in ("left", "right")}

        def solver_world_target_from_tool_tcp_world_target(self, *, tool_tcp_world_target, **_kwargs):
            return np.asarray(tool_tcp_world_target, dtype=float)

        def tool_tcp_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
            poses = {side: np.eye(4) for side in ("left", "right")}
            poses["right"][0, 3] = float(np.asarray(right_qpos)[0])
            return poses

    planner.target_builder = Builder()
    planner._require_unmodified_target_envelope = lambda *_args, **_kwargs: None
    planner._preflight_start_envelope = lambda **_kwargs: None
    planner._revalidate = lambda *_args, **_kwargs: {"sample_count": 2}
    planner._plan_constrained_approach_leg = lambda **_kwargs: pytest.fail(
        "a straight-sweep diagnostic must defer to the global target search"
    )

    target = np.eye(4)
    target[0, 3] = .1
    endpoint_qpos = {side: value.copy() for side, value in current.items()}
    endpoint_qpos["right"][0] = .1
    endpoint = grasp_trajectory_module._EndpointIKCandidate(
        qpos_by_side=endpoint_qpos,
        seed_index=0,
        solve_cycles=1,
        position_residual_m=0.0,
        orientation_residual_deg=0.0,
    )
    leg = grasp_trajectory_module._PlannedPathLeg(
        endpoint=endpoint,
        path=SimpleNamespace(
            waypoints=(current, endpoint_qpos),
            metadata={"planning_phase": "grasp_target"},
            semantic_content_hash="path:test",
        ),
        dense_revalidation={},
        search_attempts=(),
    )
    calls = []

    def direct_target_search(**kwargs):
        calls.append(kwargs)
        assert kwargs["planning_phase"] == "grasp_target"
        return leg

    planner._plan_path_leg = direct_target_search
    monkeypatch.setattr(
        grasp_trajectory_module,
        "load_joint_velocity_limit_profiles",
        lambda: {"test-velocity": SimpleNamespace(max_velocity_deg_s=(30.0,) * 6)},
    )
    scene = VisibleGraspScene(
        points=np.array([[.14, 0.0, -.08]]),
        target_mask=np.array([False]),
        geometry=ParallelJawSolidGeometry(-.043, .05, .01, .0255, .07),
    )

    trajectory = planner(
        qpos_by_side=current,
        side="right",
        target_tool_tcp_world=target,
        speed_scale=.2,
        visible_scene=scene,
        monotonic=lambda: 0.0,
    )

    assert len(calls) == 1
    np.testing.assert_allclose(trajectory.qpos_by_side["right"][-1][0], .1)


def test_local_pregrasp_failure_falls_back_to_global_grasp_target_search(monkeypatch):
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig(
        allow_unconstrained_terminal_fallback=True,
    ))
    planner.scene = SimpleNamespace(semantic_content_hash="scene:test")
    planner.profile.position_profile.ik_execution.joint_velocity_limit_profile = (
        "test-velocity"
    )
    current = {side: np.zeros(6) for side in ("left", "right")}

    class Builder:
        def solver_world_targets_for_qpos_pair(self, **_kwargs):
            return {side: np.eye(4) for side in ("left", "right")}

        def solver_world_target_from_tool_tcp_world_target(
            self, *, tool_tcp_world_target, **_kwargs
        ):
            return np.asarray(tool_tcp_world_target, dtype=float)

        def tool_tcp_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
            poses = {side: np.eye(4) for side in ("left", "right")}
            poses["right"][0, 3] = float(np.asarray(right_qpos)[0])
            return poses

    planner.target_builder = Builder()
    planner._require_unmodified_target_envelope = lambda *_args, **_kwargs: None
    planner._preflight_start_envelope = lambda **_kwargs: None
    planner._pregrasp_distance_candidates = lambda: (0.02,)
    planner._revalidate = lambda *_args, **_kwargs: {"sample_count": 2}

    target = np.eye(4)
    target[0, 3] = .1
    endpoint_qpos = {side: value.copy() for side, value in current.items()}
    endpoint_qpos["right"][0] = .1
    endpoint = grasp_trajectory_module._EndpointIKCandidate(
        qpos_by_side=endpoint_qpos,
        seed_index=0,
        solve_cycles=1,
        position_residual_m=0.0,
        orientation_residual_deg=0.0,
    )
    leg = grasp_trajectory_module._PlannedPathLeg(
        endpoint=endpoint,
        path=SimpleNamespace(
            waypoints=(current, endpoint_qpos),
            metadata={"planning_phase": "grasp_target"},
            semantic_content_hash="path:test",
        ),
        dense_revalidation={},
        search_attempts=(),
    )
    calls = []

    def local_rejection(**_kwargs):
        raise RM65GraspTrajectoryPlanningError(
            "ik_search_exhausted",
            details={"planning_phase": "local_pregrasp_ik"},
        )

    def global_target_search(**kwargs):
        calls.append(kwargs)
        assert kwargs["planning_phase"] == "grasp_target"
        return leg

    planner._plan_chain_for_pregrasp = local_rejection
    planner._plan_path_leg = global_target_search
    monkeypatch.setattr(
        grasp_trajectory_module,
        "load_joint_velocity_limit_profiles",
        lambda: {"test-velocity": SimpleNamespace(max_velocity_deg_s=(30.0,) * 6)},
    )

    trajectory = planner(
        qpos_by_side=current,
        side="right",
        target_tool_tcp_world=target,
        speed_scale=.2,
        monotonic=lambda: 0.0,
    )

    assert len(calls) == 1
    np.testing.assert_allclose(trajectory.qpos_by_side["right"][-1][0], .1)


def test_pregrasp_branch_selection_validates_local_stage_before_exact_global_connection():
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    current = {side: np.zeros(6) for side in ("left", "right")}
    endpoints = tuple(
        grasp_trajectory_module._EndpointIKCandidate(
            qpos_by_side={"left": current["left"], "right": np.full(6, value)},
            seed_index=index,
            solve_cycles=10,
            position_residual_m=0.0,
            orientation_residual_deg=0.0,
        )
        for index, value in enumerate((0.1, 0.2))
    )
    planner._endpoint_qpos_candidates = lambda **_kwargs: endpoints
    calls = []

    def leg(endpoint):
        return grasp_trajectory_module._PlannedPathLeg(
            endpoint=endpoint,
            path=SimpleNamespace(waypoints=(endpoint.qpos_by_side,)),
            dense_revalidation={},
            search_attempts=(),
        )

    def local(**kwargs):
        qpos = kwargs["qpos_by_side"]
        calls.append(("local", float(qpos["right"][0])))
        if float(qpos["right"][0]) == pytest.approx(0.1):
            raise RM65GraspTrajectoryPlanningError("approach_branch_discontinuous")
        return leg(endpoints[1])

    def global_connection(**kwargs):
        selected = kwargs["endpoint_candidates"]
        assert len(selected) == 1
        assert selected[0] is endpoints[1]
        np.testing.assert_array_equal(kwargs["qpos_by_side"]["right"], current["right"])
        calls.append(("global", float(selected[0].qpos_by_side["right"][0])))
        return leg(selected[0])

    planner._plan_constrained_approach_leg = local
    planner._plan_path_leg = global_connection
    pregrasp, approach = planner._plan_pregrasp_and_approach(
        qpos_by_side=current,
        side="right",
        pregrasp_tool_tcp_world=np.eye(4),
        grasp_tool_tcp_world=np.eye(4),
        deadline=10.0,
        monotonic=lambda: 0.0,
    )

    assert calls == [("local", 0.1), ("local", 0.2), ("global", 0.2)]
    assert pregrasp.endpoint is endpoints[1]
    assert approach.endpoint is endpoints[1]
    assert pregrasp.search_attempts[0]["reason_code"] == "approach_branch_discontinuous"
    assert pregrasp.search_attempts[1]["status"] == "local_chain_valid"
    assert len(pregrasp.search_attempts) == 2


def test_pregrasp_shared_connection_reuses_exact_selected_cached_local_chain():
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    current = {side: np.zeros(6) for side in ("left", "right")}
    endpoints = tuple(grasp_trajectory_module._EndpointIKCandidate(
        qpos_by_side={"left": current["left"], "right": np.full(6, value)},
        seed_index=index, solve_cycles=1, position_residual_m=0, orientation_residual_deg=0,
    ) for index, value in enumerate((.1, .2)))
    planner._endpoint_qpos_candidates = lambda **_: endpoints
    calls = []
    local_legs = []

    def local(**arguments):
        index = len(local_legs)
        calls.append(("local", index))
        leg = grasp_trajectory_module._PlannedPathLeg(
            endpoint=endpoints[index], path=SimpleNamespace(waypoints=(arguments["qpos_by_side"],)),
            dense_revalidation={}, search_attempts=(),
        )
        local_legs.append(leg)
        return leg

    def connect(**arguments):
        assert arguments["endpoint_candidates"] == endpoints
        calls.append(("global", len(arguments["endpoint_candidates"])))
        return grasp_trajectory_module._PlannedPathLeg(
            endpoint=endpoints[1], path=SimpleNamespace(waypoints=(current, endpoints[1].qpos_by_side)),
            dense_revalidation={}, search_attempts=({"status": "path_found"},),
        )

    planner._plan_constrained_approach_leg = local
    planner._plan_path_leg = connect
    pregrasp, approach = planner._plan_pregrasp_and_approach(
        qpos_by_side=current, side="right", pregrasp_tool_tcp_world=np.eye(4),
        grasp_tool_tcp_world=np.eye(4), deadline=10.0, monotonic=lambda: 0.0,
    )
    assert calls == [("local", 0), ("local", 1), ("global", 2)]
    assert pregrasp.endpoint is endpoints[1]
    assert approach is local_legs[1]
    assert pregrasp.search_attempts[-1]["status"] == "path_found"


@pytest.mark.parametrize("failure_stage", ["pregrasp_ik", "approach"])
def test_local_chain_failure_never_spends_envelope_ingress_search_budget(failure_stage):
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    current = {side: np.zeros(6) for side in ("left", "right")}
    endpoint = grasp_trajectory_module._EndpointIKCandidate(
        qpos_by_side=current, seed_index=0, solve_cycles=1,
        position_residual_m=0, orientation_residual_deg=0,
    )
    def endpoints(**_):
        if failure_stage == "pregrasp_ik":
            raise RM65GraspTrajectoryPlanningError("ik_search_exhausted")
        return (endpoint,)
    planner._endpoint_qpos_candidates = endpoints
    def approach(**_):
        raise RM65GraspTrajectoryPlanningError("approach_branch_discontinuous")
    planner._plan_constrained_approach_leg = approach
    def must_not_search(**_):
        pytest.fail("a rejected local chain must not search global/ingress paths")
    planner._envelope_ingress_targets = must_not_search
    planner._plan_path_leg = must_not_search
    with pytest.raises(RM65GraspTrajectoryPlanningError) as error:
        planner._plan_chain_for_pregrasp(
            qpos_by_side=current, side="right", pregrasp_tool_tcp_world=np.eye(4),
            grasp_tool_tcp_world=np.eye(4), deadline=10, monotonic=lambda: 0,
        )
    assert error.value.details["planning_phase"] == (
        "terminal_branch_selection"
        if failure_stage == "pregrasp_ik"
        else "local_approach_validation"
    )


def test_local_chain_is_prevalidated_and_reused_after_envelope_ingress():
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    current = {side: np.zeros(6) for side in ("left", "right")}
    def qpos(value):
        pair = {side: q.copy() for side, q in current.items()}
        pair["right"][0] = value
        return pair
    def endpoint(value):
        return grasp_trajectory_module._EndpointIKCandidate(
            qpos_by_side=qpos(value), seed_index=0, solve_cycles=1,
            position_residual_m=0, orientation_residual_deg=0,
        )
    pre, final, ingress = (endpoint(value) for value in (.1, .12, .05))
    def leg(start, end, name):
        return grasp_trajectory_module._PlannedPathLeg(
            endpoint=end, path=SimpleNamespace(waypoints=(start, end.qpos_by_side), semantic_content_hash=name),
            dense_revalidation={}, search_attempts=(),
        )
    calls = []
    def endpoints(**arguments):
        np.testing.assert_array_equal(arguments["qpos_by_side"]["right"], current["right"])
        calls.append("local_ik")
        return (pre,)
    planner._endpoint_qpos_candidates = endpoints
    local = leg(pre.qpos_by_side, final, "local:test")
    def approach(**arguments):
        np.testing.assert_array_equal(arguments["qpos_by_side"]["right"], pre.qpos_by_side["right"])
        calls.append("local_approach")
        return local
    planner._plan_constrained_approach_leg = approach
    planner._envelope_ingress_targets = lambda **_: ((.05, np.eye(4)),)
    def global_path(**arguments):
        if arguments["planning_phase"] == "envelope_ingress":
            calls.append("ingress")
            return leg(current, ingress, "ingress:test")
        assert arguments["endpoint_candidates"] == (pre,)
        np.testing.assert_array_equal(arguments["qpos_by_side"]["right"], ingress.qpos_by_side["right"])
        calls.append("global_connection")
        return leg(ingress.qpos_by_side, pre, "global:test")
    planner._plan_path_leg = global_path
    planner._revalidate = lambda *_args, **_kwargs: {}
    legs, route, _ = planner._plan_chain_for_pregrasp(
        qpos_by_side=current, side="right", pregrasp_tool_tcp_world=np.eye(4),
        grasp_tool_tcp_world=np.eye(4), deadline=10, monotonic=lambda: 0,
    )
    assert calls == ["local_ik", "local_approach", "ingress", "global_connection"]
    assert legs[1].endpoint is pre
    assert legs[2] is local
    assert route["ingress_path_hash"] == "ingress:test"


@pytest.mark.parametrize("failure_stage", [
    "ingress", "connection", "revalidation", "unavailable", "endpoint_unavailable",
    "envelope_unavailable", "target_envelope_unavailable", "global", "global_boundary",
])
def test_ingress_route_slice_preserves_other_entries_exact_branch_and_global_deadline(failure_stage):
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    current = {side: np.zeros(6) for side in ("left", "right")}
    endpoint = grasp_trajectory_module._EndpointIKCandidate(
        qpos_by_side=current, seed_index=0, solve_cycles=1,
        position_residual_m=0, orientation_residual_deg=0,
    )
    leg = grasp_trajectory_module._PlannedPathLeg(
        endpoint=endpoint, path=SimpleNamespace(waypoints=(current, current), semantic_content_hash="path:test"),
        dense_revalidation={}, search_attempts=(),
    )
    prepared = grasp_trajectory_module._PreparedGraspApproaches(((endpoint, leg),), ())
    clock = [0.0]
    prepare_calls = []
    def prepare(**arguments):
        prepare_calls.append(arguments["deadline"])
        return prepared
    planner._prepare_grasp_approaches = prepare
    planner._envelope_ingress_targets = lambda **_: ((.05, np.eye(4)), (.1, np.eye(4)))
    route_calls = []
    unavailable_reasons = {
        "unavailable": "path_collision_check_unavailable",
        "endpoint_unavailable": "endpoint_collision_check_unavailable",
        "envelope_unavailable": "path_target_safety_envelope_unavailable",
        "target_envelope_unavailable": "target_safety_envelope_unavailable",
    }
    def fail_first(stage, arguments):
        if len(route_calls) != 1:
            return
        selected = (
            "ingress" if failure_stage in unavailable_reasons
            or failure_stage in {"global", "global_boundary"} else failure_stage
        )
        if stage != selected:
            return
        clock[0] = 10.0 if failure_stage in {"global", "global_boundary"} else arguments["deadline"]
        if failure_stage in unavailable_reasons:
            clock[0] = 1.0
        raise RM65GraspTrajectoryPlanningError(
            unavailable_reasons.get(
                failure_stage,
                "path_search_exhausted" if failure_stage == "global_boundary"
                else "path_planning_time_budget_exhausted",
            ),
            details={"planning_phase": stage, "first_failure_retained": True},
        )
    def ingress(**arguments):
        route_calls.append(arguments["deadline"])
        fail_first("ingress", arguments)
        return leg
    def connect(**arguments):
        assert arguments["deadline"] == route_calls[-1]
        assert arguments["prepared_approaches"] is prepared
        fail_first("connection", arguments)
        return leg, leg
    def dense(*_, **arguments):
        assert arguments["deadline"] == route_calls[-1]
        fail_first("revalidation", arguments)
        return {"minimum_distance_m": .05}
    planner._plan_path_leg = ingress
    planner._plan_pregrasp_and_approach = connect
    planner._revalidate = dense
    arguments = dict(qpos_by_side=current, side="right", pregrasp_tool_tcp_world=np.eye(4),
                     grasp_tool_tcp_world=np.eye(4), deadline=10.0, monotonic=lambda: clock[0])
    if failure_stage in unavailable_reasons or failure_stage in {"global", "global_boundary"}:
        with pytest.raises(RM65GraspTrajectoryPlanningError) as error:
            planner._plan_chain_for_pregrasp(**arguments)
        assert route_calls == [5.0]
        attempts = error.value.details["envelope_ingress_route_attempts"]
        assert attempts[0]["details"]["first_failure_retained"]
        if failure_stage == "global":
            assert attempts[0]["budget_scope"] == "global"
        if failure_stage == "global_boundary":
            assert error.value.reason_code == "path_planning_time_budget_exhausted"
            assert len(attempts) == 1
            assert attempts[0]["reason_code"] == "path_search_exhausted"
    else:
        legs, route, attempts = planner._plan_chain_for_pregrasp(**arguments)
        assert route_calls == [5.0, 10.0]
        assert route["inward_fraction"] == .1
        assert legs == (leg, leg, leg)
        assert attempts[0]["budget_scope"] == "ingress_route"
        assert attempts[0]["details"]["first_failure_retained"]
        assert attempts[1]["status"] == "path_found"
    assert prepare_calls == [10.0]


def test_path_leg_submits_all_ik_branches_to_one_budgeted_search(monkeypatch):
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig(path_search_attempts_per_endpoint=2))
    planner.planner_config = JointPathPlannerConfig()
    planner.oracle = SimpleNamespace(scene_hash="scene:test")
    current = {side: np.zeros(6) for side in ("left", "right")}
    endpoints = tuple(grasp_trajectory_module._EndpointIKCandidate(
        qpos_by_side={"left": current["left"], "right": np.full(6, value)},
        seed_index=index, solve_cycles=1, position_residual_m=0, orientation_residual_deg=0,
    ) for index, value in enumerate((.1, .2)))
    clock = [0.0]
    calls = []
    def plan(**arguments):
        value = float(arguments["target_qpos_by_component"]["right"][0])
        alternative = arguments["alternative_target_qpos_by_component"][0]
        calls.append((value, float(alternative["right"][0]), arguments["config"].seed, arguments["config"].time_budget_sec))
        clock[0] += 1.0
        return JointPathPlanResult(
            status=JointPathStatus.PATH_FOUND,
            start_qpos_by_component=current, target_qpos_by_component=alternative,
            movable_components=("right",), scene_hash="scene:test",
            reason="bounded_search_found_path", metadata={"selected_target_index": 1},
            waypoints=(current, alternative),
        )
    monkeypatch.setattr(grasp_trajectory_module, "plan_collision_aware_joint_path", plan)
    planner._revalidate = lambda *_args, **_kwargs: {"minimum_distance_m": .05}
    leg = planner._plan_path_leg(qpos_by_side=current, side="right", target_tool_tcp_world=np.eye(4),
        planning_phase="grasp_target", endpoint_candidates=endpoints, deadline=10.0, monotonic=lambda: clock[0])
    assert calls == [(.1, .2, 0, 10.0)]
    assert leg.endpoint is endpoints[1]
    assert [a["endpoint_seed_index"] for a in leg.search_attempts] == [1]
    assert leg.search_attempts[0]["endpoint_seed_indices"] == (0, 1)
    assert all(a["path_attempt"] == 0 for a in leg.search_attempts)
    assert leg.dense_revalidation["minimum_distance_m"] == .05


@pytest.mark.parametrize("failure_reason", [
    "path_collision_check_unavailable", "path_target_safety_envelope_unavailable",
])
def test_path_leg_dense_query_fault_stops_before_other_branch(monkeypatch, failure_reason):
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    planner.planner_config = JointPathPlannerConfig()
    planner.oracle = SimpleNamespace(scene_hash="scene:test")
    current = {side: np.zeros(6) for side in ("left", "right")}
    endpoints = tuple(grasp_trajectory_module._EndpointIKCandidate(
        qpos_by_side={"left": current["left"], "right": np.full(6, value)},
        seed_index=index, solve_cycles=1, position_residual_m=0, orientation_residual_deg=0,
    ) for index, value in enumerate((.1, .2)))
    calls = []
    def plan(**arguments):
        calls.append(arguments["target_qpos_by_component"])
        return JointPathPlanResult(
            status=JointPathStatus.PATH_FOUND, start_qpos_by_component=current,
            target_qpos_by_component=arguments["target_qpos_by_component"],
            movable_components=("right",), scene_hash="scene:test", reason="direct_path_clear",
            waypoints=(current, arguments["target_qpos_by_component"]),
        )
    def dense(*_args, **_kwargs):
        raise RM65GraspTrajectoryPlanningError(failure_reason, details={"query_fault": True})
    monkeypatch.setattr(grasp_trajectory_module, "plan_collision_aware_joint_path", plan)
    planner._revalidate = dense
    with pytest.raises(RM65GraspTrajectoryPlanningError, match=failure_reason) as error:
        planner._plan_path_leg(qpos_by_side=current, side="right", target_tool_tcp_world=np.eye(4),
            planning_phase="grasp_target", endpoint_candidates=endpoints,
            deadline=10.0, monotonic=lambda: 0.0)
    assert len(calls) == 1
    assert calls[0] is endpoints[0].qpos_by_side
    assert error.value.details["last_dense_revalidation_error"]["details"]["query_fault"]


def test_shared_goal_dense_rejection_prioritizes_other_branch_without_dropping_goals(monkeypatch):
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig(path_search_attempts_per_endpoint=2))
    planner.planner_config = JointPathPlannerConfig()
    planner.oracle = SimpleNamespace(scene_hash="scene:test")
    current = {side: np.zeros(6) for side in ("left", "right")}
    endpoints = tuple(grasp_trajectory_module._EndpointIKCandidate(
        qpos_by_side={"left": current["left"], "right": np.full(6, value)},
        seed_index=index, solve_cycles=1, position_residual_m=0, orientation_residual_deg=0,
    ) for index, value in enumerate((.1, .2)))
    calls = []

    def plan(**arguments):
        targets = (arguments["target_qpos_by_component"],) + arguments["alternative_target_qpos_by_component"]
        calls.append(tuple(float(target["right"][0]) for target in targets))
        return JointPathPlanResult(
            status=JointPathStatus.PATH_FOUND, start_qpos_by_component=current,
            target_qpos_by_component=targets[0], movable_components=("right",),
            waypoints=(current, targets[0]), metadata={"selected_target_index": 0},
        )

    def dense(waypoints, **_):
        if float(waypoints[-1]["right"][0]) == .1:
            raise RM65GraspTrajectoryPlanningError("path_target_safety_envelope_blocked")
        return {"sample_count": 2}

    monkeypatch.setattr(grasp_trajectory_module, "plan_collision_aware_joint_path", plan)
    planner._revalidate = dense
    leg = planner._plan_path_leg(
        qpos_by_side=current, side="right", target_tool_tcp_world=np.eye(4),
        planning_phase="pregrasp_target", endpoint_candidates=endpoints,
        deadline=10.0, monotonic=lambda: 0.0,
    )
    assert calls == [(.1, .2), (.2, .1)]
    assert leg.endpoint is endpoints[1]
    assert leg.search_attempts[0]["dense_revalidation_error"] == "path_target_safety_envelope_blocked"
    assert leg.search_attempts[1]["endpoint_seed_indices"] == (1, 0)


@pytest.mark.parametrize("selected_index", [0, 2])
def test_shared_goal_result_cannot_claim_a_different_endpoint(monkeypatch, selected_index):
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    planner.planner_config = JointPathPlannerConfig()
    planner.oracle = SimpleNamespace(scene_hash="scene:test")
    current = {side: np.zeros(6) for side in ("left", "right")}
    endpoints = tuple(grasp_trajectory_module._EndpointIKCandidate(
        qpos_by_side={"left": current["left"], "right": np.full(6, value)},
        seed_index=index, solve_cycles=1, position_residual_m=0, orientation_residual_deg=0,
    ) for index, value in enumerate((.1, .2)))
    def plan(**_):
        return JointPathPlanResult(
            status=JointPathStatus.PATH_FOUND, start_qpos_by_component=current,
            target_qpos_by_component=endpoints[1].qpos_by_side, movable_components=("right",),
            waypoints=(current, endpoints[1].qpos_by_side), metadata={"selected_target_index": selected_index},
        )
    def unexpected_dense(*_, **__):
        pytest.fail("a mismatched goal must not reach dense validation")
    monkeypatch.setattr(grasp_trajectory_module, "plan_collision_aware_joint_path", plan)
    planner._revalidate = unexpected_dense
    with pytest.raises(RM65GraspTrajectoryPlanningError, match="path_planner_error"):
        planner._plan_path_leg(
            qpos_by_side=current, side="right", target_tool_tcp_world=np.eye(4),
            planning_phase="pregrasp_target", endpoint_candidates=endpoints,
            deadline=10.0, monotonic=lambda: 0.0,
        )


def test_singularity_replay_time_scaling_slows_without_rejecting_path() -> None:
    planner = object.__new__(RM65GraspJointTrajectoryPlanner)
    planner.config = RM65GraspTrajectoryPlanningConfig(
        singularity_soft_margin_deg=6.0,
        singularity_replay_slowdown=0.5,
    )
    samples = {
        "left": [
            np.array([0.0, 0.0, 0.20, 0.0, 0.20, 0.0]),
            np.array([0.0, 0.0, 0.02, 0.0, 0.20, 0.0]),
            np.array([0.0, 0.0, 0.20, 0.0, 0.20, 0.0]),
        ],
        "right": [np.zeros(6), np.zeros(6), np.zeros(6)],
    }
    original = np.asarray(samples["left"], dtype=float).copy()
    scaled, metadata = planner._apply_singularity_replay_time_scaling(
        times=[0.0, 1.0, 2.0],
        samples=samples,
        moving_side="left",
    )

    assert scaled == pytest.approx([0.0, 2.0, 4.0])
    assert metadata["policy"] == "real_teleop_soft_traversal_v1"
    assert metadata["scaled_interval_count"] == 2
    assert metadata["collision_rechecked"] is True
    np.testing.assert_allclose(np.asarray(samples["left"]), original)


def test_continuous_c2_parameterization_keeps_internal_knot_moving() -> None:
    planner = object.__new__(RM65GraspJointTrajectoryPlanner)
    planner.config = RM65GraspTrajectoryPlanningConfig(
        trajectory_parameterization=(
            RM65_GRASP_TRAJECTORY_PARAMETERIZATION_CONTINUOUS_C2
        )
    )
    planner.target_builder = _CartesianQposBuilder()
    left = np.zeros(6)
    waypoints = tuple(
        {
            "left": left.copy(),
            "right": np.asarray((x, 0.0, 0.0, 0.0, 0.3, 0.0)),
        }
        for x in (0.0, 0.05, 0.10)
    )
    velocity_limit = np.full(6, 0.5)

    continuous, segments, tcp_time_scale = planner._continuous_leg_trajectory(
        waypoints=waypoints,
        moving_side="right",
        speed_scale=0.5,
        joint_velocity_rad_s=velocity_limit,
        reference_hz=60.0,
    )

    assert continuous.metadata["internal_knot_zero_velocity_required"] is False
    assert abs(float(continuous.knot_qd_rad_s[1, 6])) > 1e-6
    assert float(np.max(np.abs(continuous.qd_rad_s[:, 6:]))) <= 0.5 + 1e-7
    assert len(segments) == 2
    assert tcp_time_scale >= 1.0
    np.testing.assert_allclose(continuous.waypoint_q_rad[:, 6], (0.0, 0.05, 0.10))


def test_grasp_trajectory_parameterization_defaults_to_legacy_until_promoted() -> None:
    config = RM65GraspTrajectoryPlanningConfig()

    assert config.trajectory_parameterization == (
        RM65_GRASP_TRAJECTORY_PARAMETERIZATION_SEGMENTED_STOP
    )

    with pytest.raises(ValueError, match="trajectory_parameterization"):
        RM65GraspTrajectoryPlanningConfig(trajectory_parameterization="unknown")

    assert config.endpoint_candidate_ordering == (
        RM65_GRASP_ENDPOINT_ORDER_JOINT_DISTANCE
    )
    with pytest.raises(ValueError, match="endpoint_candidate_ordering"):
        RM65GraspTrajectoryPlanningConfig(endpoint_candidate_ordering="unknown")


def test_grasp_trajectory_speed_profile_accelerates_only_far_field() -> None:
    planner = RM65GraspJointTrajectoryPlanner.__new__(RM65GraspJointTrajectoryPlanner)
    planner.config = RM65GraspTrajectoryPlanningConfig(
        trajectory_speed_profile=RM65_GRASP_TRAJECTORY_SPEED_PROFILE_FAR_4X_APPROACH_1X
    )

    assert planner._effective_trajectory_speed_scale(
        base_speed_scale=0.20, planning_phase="envelope_ingress"
    ) == 0.80
    assert planner._effective_trajectory_speed_scale(
        base_speed_scale=0.20, planning_phase="pregrasp_target"
    ) == 0.80
    assert planner._effective_trajectory_speed_scale(
        base_speed_scale=0.20, planning_phase="constrained_approach"
    ) == 0.20
    assert planner._effective_trajectory_speed_scale(
        base_speed_scale=0.20, planning_phase="grasp_target"
    ) == 0.20

    planner.config = RM65GraspTrajectoryPlanningConfig(
        trajectory_speed_profile=RM65_GRASP_TRAJECTORY_SPEED_PROFILE_UNIFORM
    )
    assert planner._effective_trajectory_speed_scale(
        base_speed_scale=0.20, planning_phase="pregrasp_target"
    ) == 0.20


def test_estimated_execution_time_orders_ik_branches_by_joint_speed(
    monkeypatch,
) -> None:
    planner = _planner_without_assets(
        config=RM65GraspTrajectoryPlanningConfig(
            endpoint_candidate_ordering=(
                RM65_GRASP_ENDPOINT_ORDER_ESTIMATED_EXECUTION_TIME
            )
        )
    )
    current = np.zeros(6)

    def candidate(seed_index: int, joint_index: int, value: float):
        right = current.copy()
        right[joint_index] = value
        return grasp_trajectory_module._EndpointIKCandidate(
            qpos_by_side={"left": current.copy(), "right": right},
            seed_index=seed_index,
            solve_cycles=1,
            position_residual_m=0.0,
            orientation_residual_deg=0.0,
        )

    smaller_joint_distance_but_slow = candidate(0, 1, 0.30)
    larger_joint_distance_but_fast = candidate(1, 0, 0.40)
    monkeypatch.setattr(
        grasp_trajectory_module,
        "load_joint_velocity_limit_profiles",
        lambda: {
            "test-velocity": SimpleNamespace(
                max_velocity_deg_s=(100.0, 10.0, 100.0, 100.0, 100.0, 100.0)
            )
        },
    )

    ordered = planner._ordered_endpoint_candidates(
        candidates=(smaller_joint_distance_but_slow, larger_joint_distance_but_fast),
        current_qpos=current,
        side="right",
    )

    assert tuple(item.seed_index for item in ordered) == (1, 0)


def test_planner_entrypoint_records_continuous_parameterization(
    monkeypatch,
    tmp_path,
) -> None:
    planner = _planner_without_assets(
        config=RM65GraspTrajectoryPlanningConfig(
            trajectory_parameterization=(
                RM65_GRASP_TRAJECTORY_PARAMETERIZATION_CONTINUOUS_C2
            )
        )
    )
    planner.target_builder = _LinearQposBuilder()
    planner.scene = SimpleNamespace(semantic_content_hash="scene:test")
    planner.trajectory_record_dir = tmp_path
    planner._preflight_start_envelope = lambda **_kwargs: None
    planner._require_unmodified_target_envelope = lambda *_args, **_kwargs: None

    left = np.zeros(6)

    def waypoint(value: float) -> dict[str, np.ndarray]:
        right = np.zeros(6)
        right[0] = value
        return {"left": left.copy(), "right": right}

    global_waypoints = tuple(waypoint(value) for value in (0.0, 0.05, 0.10))
    approach_waypoints = tuple(waypoint(value) for value in (0.10, 0.15, 0.20))

    def leg(phase: str, waypoints) -> grasp_trajectory_module._PlannedPathLeg:
        endpoint = grasp_trajectory_module._EndpointIKCandidate(
            qpos_by_side=waypoints[-1],
            seed_index=0,
            solve_cycles=1,
            position_residual_m=0.0,
            orientation_residual_deg=0.0,
        )
        path = JointPathPlanResult(
            status=JointPathStatus.PATH_FOUND,
            start_qpos_by_component=waypoints[0],
            target_qpos_by_component=waypoints[-1],
            movable_components=("right",),
            waypoints=waypoints,
            scene_hash="scene:test",
            metadata={"planning_phase": phase},
        )
        return grasp_trajectory_module._PlannedPathLeg(
            endpoint=endpoint,
            path=path,
            dense_revalidation={"safe": True},
            search_attempts=(),
        )

    legs = (
        leg("pregrasp_target", global_waypoints),
        leg("constrained_approach", approach_waypoints),
    )
    planner._plan_chain_for_pregrasp = lambda **_kwargs: (legs, None, [])
    revalidated_sample_counts = []

    def revalidate(waypoints, **_kwargs):
        revalidated_sample_counts.append(len(waypoints))
        return {"safe": True, "sample_count": len(waypoints)}

    planner._revalidate = revalidate
    monkeypatch.setattr(
        grasp_trajectory_module,
        "load_joint_velocity_limit_profiles",
        lambda: {
            "test-velocity": SimpleNamespace(max_velocity_deg_s=(30.0,) * 6)
        },
    )
    recorded = {}
    monkeypatch.setattr(
        grasp_trajectory_module,
        "write_grasp_trajectory_record",
        lambda *_args, **kwargs: recorded.update(kwargs),
    )

    target = np.eye(4)
    target[0, 3] = 0.20
    trajectory = planner(
        qpos_by_side=waypoint(0.0),
        side="right",
        target_tool_tcp_world=target,
        speed_scale=0.5,
        monotonic=lambda: 0.0,
    )

    assert revalidated_sample_counts[0] == 5
    assert len(revalidated_sample_counts) == 3
    assert (
        sum(revalidated_sample_counts[1:]) - 1
        == trajectory.time_from_start_sec.size
    )
    assert recorded["parameterization"]["mode"] == (
        RM65_GRASP_TRAJECTORY_PARAMETERIZATION_CONTINUOUS_C2
    )
    assert recorded["parameterization"]["internal_path_waypoint_stops"] is False
    assert recorded["planning_timing"]["clock"] == "host_perf_counter"
    assert recorded["planning_timing"]["path_search_wall_duration_sec"] >= 0.0
    assert recorded["planning_timing"]["planning_wall_duration_sec"] >= recorded[
        "planning_timing"
    ]["path_search_wall_duration_sec"]
    assert tuple(
        item["planning_phase"] for item in recorded["stage_boundaries"]
    ) == ("pregrasp_target", "constrained_approach")
    assert recorded["source_waypoints_by_side"]["right"].shape == (5, 6)
    assert trajectory.time_from_start_sec.size > 5
    np.testing.assert_allclose(trajectory.qpos_by_side["right"][-1, 0], 0.20)


def test_pregrasp_backs_away_along_tool_approach_axis() -> None:
    planner = _planner_without_assets(
        config=RM65GraspTrajectoryPlanningConfig(pregrasp_distance_m=0.06)
    )
    target = np.eye(4)
    target[:3, :3] = grasp_trajectory_module.Rotation.from_euler(
        "y", 90.0, degrees=True
    ).as_matrix()
    target[:3, 3] = (0.2, -0.1, 0.3)

    pregrasp = planner._pregrasp_target_pose(target)

    np.testing.assert_allclose(pregrasp[:3, :3], target[:3, :3], atol=1e-12)
    np.testing.assert_allclose(
        target[:3, 3] - pregrasp[:3, 3],
        0.06 * target[:3, 2],
        atol=1e-12,
    )
    assert planner._pregrasp_distance_candidates() == pytest.approx(
        (0.06, 0.05, 0.04, 0.03, 0.02)
    )


def test_constrained_approach_plans_fixed_orientation_monotonic_tcp_path() -> None:
    planner = _planner_without_assets(
        config=RM65GraspTrajectoryPlanningConfig(
            pregrasp_distance_m=0.03,
            approach_waypoint_spacing_m=0.01,
        )
    )
    planner.target_builder = _CartesianQposBuilder()
    planner.scene = SimpleNamespace(semantic_content_hash="scene:test")
    planner.oracle = SimpleNamespace(
        check=lambda _qpos: JointConfigurationCheck(
            safe=True,
            minimum_distance_m=0.04,
        )
    )
    pregrasp = np.eye(4)
    grasp = np.eye(4)
    grasp[2, 3] = 0.03
    current = {side: np.zeros(6) for side in ("left", "right")}

    leg = planner._plan_constrained_approach_leg(
        qpos_by_side=current,
        side="right",
        pregrasp_tool_tcp_world=pregrasp,
        grasp_tool_tcp_world=grasp,
    )

    assert leg.path.status is JointPathStatus.PATH_FOUND
    assert len(leg.path.waypoints) == 4
    assert leg.path.metadata["planning_phase"] == "constrained_approach"
    geometry = leg.dense_revalidation["approach_geometry"]
    assert geometry["approach_distance_m"] == pytest.approx(0.03)
    assert geometry["maximum_lateral_error_m"] < 1e-7
    assert geometry["maximum_orientation_error_deg"] < 1e-7
    assert geometry["maximum_reverse_progress_m"] <= 1e-12
    np.testing.assert_allclose(
        leg.endpoint.qpos_by_side["right"][:3],
        (0, 0, 0.03),
    )
    np.testing.assert_allclose(leg.endpoint.qpos_by_side["left"], np.zeros(6))


def test_constrained_approach_can_propagate_backward_from_terminal_branch() -> None:
    planner = _planner_without_assets(
        config=RM65GraspTrajectoryPlanningConfig(
            pregrasp_distance_m=0.03,
            approach_waypoint_spacing_m=0.01,
        )
    )
    planner.target_builder = _CartesianQposBuilder()
    planner.scene = SimpleNamespace(semantic_content_hash="scene:test")
    planner.oracle = SimpleNamespace(
        check=lambda _qpos: JointConfigurationCheck(
            safe=True,
            minimum_distance_m=0.04,
        )
    )
    pregrasp = np.eye(4)
    grasp = np.eye(4)
    grasp[2, 3] = 0.03
    terminal_qpos = {side: np.zeros(6) for side in ("left", "right")}
    terminal_qpos["right"][2] = 0.03
    terminal = grasp_trajectory_module._EndpointIKCandidate(
        qpos_by_side=terminal_qpos,
        seed_index=7,
        solve_cycles=3,
        position_residual_m=0.0,
        orientation_residual_deg=0.0,
    )

    leg = planner._plan_constrained_approach_leg(
        qpos_by_side=terminal_qpos,
        side="right",
        pregrasp_tool_tcp_world=pregrasp,
        grasp_tool_tcp_world=grasp,
        terminal_endpoint=terminal,
    )

    assert leg.endpoint is terminal
    assert leg.path.metadata["solve_direction"] == "reverse_from_terminal"
    assert len(leg.path.waypoints) == 4
    np.testing.assert_allclose(
        leg.path.waypoints[0]["right"][:3],
        (0.0, 0.0, 0.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        leg.path.waypoints[-1]["right"][:3],
        (0.0, 0.0, 0.03),
    )
    assert all(
        item["solve_direction"] == "reverse_from_terminal"
        for item in leg.search_attempts
    )


def test_local_chain_recovers_terminal_branch_hidden_by_pregrasp_ik(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _planner_without_assets(
        config=RM65GraspTrajectoryPlanningConfig(
            pregrasp_distance_m=0.03,
            approach_waypoint_spacing_m=0.01,
        )
    )
    planner.target_builder = _CartesianQposBuilder()
    planner.scene = SimpleNamespace(semantic_content_hash="scene:test")
    planner.oracle = SimpleNamespace(
        check=lambda _qpos: JointConfigurationCheck(
            safe=True,
            minimum_distance_m=0.04,
        )
    )
    pregrasp = np.eye(4)
    grasp = np.eye(4)
    grasp[2, 3] = 0.03
    current = {side: np.zeros(6) for side in ("left", "right")}
    terminal_qpos = {side: values.copy() for side, values in current.items()}
    terminal_qpos["right"][2] = 0.03
    terminal = grasp_trajectory_module._EndpointIKCandidate(
        qpos_by_side=terminal_qpos,
        seed_index=4,
        solve_cycles=2,
        position_residual_m=0.0,
        orientation_residual_deg=0.0,
    )

    def endpoints(*, target_tool_tcp_world, **_kwargs):
        target = np.asarray(target_tool_tcp_world, dtype=float)
        if target[2, 3] == pytest.approx(0.0):
            raise RM65GraspTrajectoryPlanningError(
                "endpoint_collision_blocked"
            )
        return (terminal,)

    monkeypatch.setattr(planner, "_endpoint_qpos_candidates", endpoints)

    prepared = planner._prepare_grasp_approaches(
        qpos_by_side=current,
        side="right",
        pregrasp_tool_tcp_world=pregrasp,
        grasp_tool_tcp_world=grasp,
        deadline=10.0,
        monotonic=lambda: 0.0,
    )

    assert len(prepared.candidates) == 1
    pregrasp_endpoint, approach = prepared.candidates[0]
    np.testing.assert_allclose(
        pregrasp_endpoint.qpos_by_side["right"][:3],
        (0.0, 0.0, 0.0),
        atol=1e-12,
    )
    assert approach.endpoint is terminal
    assert tuple(
        item["solve_direction"] for item in prepared.validation_attempts
    ) == ("forward_from_pregrasp", "reverse_from_terminal")
    assert prepared.validation_attempts[-1]["status"] == "local_chain_valid"


@pytest.mark.parametrize("terminal_found", [False, True])
def test_local_chain_reuses_terminal_search_only_within_one_assessment(terminal_found):
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    current = {side: np.zeros(6) for side in ("left", "right")}
    target = np.eye(4)
    terminal = grasp_trajectory_module._EndpointIKCandidate(
        qpos_by_side=current, seed_index=0, solve_cycles=1,
        position_residual_m=0, orientation_residual_deg=0,
    )
    calls = []

    def endpoints(*, target_tool_tcp_world, side, **kwargs):
        is_terminal = np.array_equal(target_tool_tcp_world, target)
        calls.append((side, is_terminal))
        assert kwargs["prefer_closest_tolerance"] is False
        if is_terminal and terminal_found:
            return (terminal,)
        raise RM65GraspTrajectoryPlanningError("ik_search_exhausted")

    reverse_attempts = []

    def approach(**kwargs):
        assert kwargs["terminal_endpoint"] is terminal
        reverse_attempts.append(kwargs["pregrasp_tool_tcp_world"].copy())
        raise RM65GraspTrajectoryPlanningError("approach_branch_discontinuous")

    planner._endpoint_qpos_candidates = endpoints
    planner._plan_constrained_approach_leg = approach
    distances = len(planner._pregrasp_distance_candidates())
    for side in ("left", "right", "left"):
        calls.clear()
        reverse_attempts.clear()
        result = planner.assess_local_grasp_chain(
            qpos_by_side=current, side=side, target_tool_tcp_world=target,
            time_budget_sec=5, monotonic=lambda: 0,
        )
        assert result.feasible is False
        assert len(result.attempts) == distances
        assert calls.count((side, True)) == 1
        assert calls.count((side, False)) == distances
        assert len(reverse_attempts) == (distances if terminal_found else 0)


def test_cached_terminal_miss_does_not_veto_a_different_forward_branch():
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    current = {side: np.zeros(6) for side in ("left", "right")}
    target = np.eye(4)
    endpoint = grasp_trajectory_module._EndpointIKCandidate(
        qpos_by_side=current, seed_index=0, solve_cycles=1,
        position_residual_m=0, orientation_residual_deg=0,
    )
    calls = []

    def endpoints(*, target_tool_tcp_world, **_kwargs):
        calls.append(float(target_tool_tcp_world[2, 3]))
        if len(calls) <= 2:
            raise RM65GraspTrajectoryPlanningError("ik_search_exhausted")
        return (endpoint,)

    planner._endpoint_qpos_candidates = endpoints
    planner._plan_constrained_approach_leg = lambda **_: SimpleNamespace()
    result = planner.assess_local_grasp_chain(
        qpos_by_side=current, side="left", target_tool_tcp_world=target,
        time_budget_sec=5, monotonic=lambda: 0,
    )
    assert result.feasible is True
    assert len(calls) == 3
    assert calls[1] == 0
    assert calls[0] != calls[2]


def test_constrained_approach_rejects_lateral_tcp_bow() -> None:
    planner = _planner_without_assets(
        config=RM65GraspTrajectoryPlanningConfig(
            maximum_approach_lateral_error_m=0.002,
        )
    )
    planner.target_builder = _CartesianQposBuilder()
    pregrasp = np.eye(4)
    grasp = np.eye(4)
    grasp[2, 3] = 0.06
    start = {side: np.zeros(6) for side in ("left", "right")}
    bowed = {side: values.copy() for side, values in start.items()}
    bowed["right"][:3] = (0.003, 0.0, 0.03)
    end = {side: values.copy() for side, values in start.items()}
    end["right"][2] = 0.06

    with pytest.raises(
        RM65GraspTrajectoryPlanningError,
        match="approach_lateral_deviation",
    ) as error:
        planner._validate_constrained_approach_geometry(
            (start, bowed, end),
            moving_side="right",
            pregrasp_tool_tcp_world=pregrasp,
            grasp_tool_tcp_world=grasp,
        )

    assert error.value.details["lateral_error_m"] > 0.002


@pytest.mark.parametrize("adaptive", [False, True])
def test_clearance_guard_is_confined_to_search_and_uses_actual_refinement_boundary(adaptive) -> None:
    from robot_motion.rm65.obstacle_collision import RM65ObstacleCollisionOracle
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    planner.planner_config = JointPathPlannerConfig(
        adaptive_edge_resolution_rad=0.001 if adaptive else None,
        adaptive_refinement_distance_m=0.03 if adaptive else None,
    )
    planner.scene = SimpleNamespace(minimum_clearance_m=0.02)
    class RecordingOracle(RM65ObstacleCollisionOracle):
        def __init__(self):
            self.calls = []
        @property
        def scene_hash(self):
            return "scene:test"
        def check_for_search(self, qpos, *, refinement_distance_m):
            self.calls.append(("search", refinement_distance_m))
            return JointConfigurationCheck(safe=True, minimum_distance_m=0.04)
        def check(self, qpos):
            self.calls.append(("exact", None))
            return JointConfigurationCheck(safe=True, minimum_distance_m=0.05)
    planner.oracle = RecordingOracle()
    planner.target_envelope_filter = None
    qpos = {side: np.zeros(6) for side in ("left", "right")}
    search = planner._path_search_oracle(planning_phase="pregrasp_target")
    assert search.scene_hash == planner.oracle.scene_hash
    assert search.check(qpos).safe
    assert planner.oracle.calls == [("search", 0.03 if adaptive else 0.02)]
    assert planner.oracle.check(qpos).minimum_distance_m == 0.05
    assert planner.oracle.calls[-1] == ("exact", None)
    planner.target_builder = _CartesianQposBuilder()
    planner.oracle.calls.clear()
    end = {side: values.copy() for side, values in qpos.items()}
    end["right"][0] = 0.002
    dense = planner._revalidate((qpos, end), moving_side="right")
    assert dense["minimum_distance_m"] == 0.05
    assert dense["sample_count"] >= 2
    assert all(mode == "exact" for mode, _ in planner.oracle.calls)


def test_grasp_search_oracle_composes_collision_and_target_envelope() -> None:
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    planner.target_builder = _LinearQposBuilder()
    collision_oracle = SimpleNamespace(
        scene_hash="scene:test",
        check=lambda _qpos: JointConfigurationCheck(
            safe=True,
            minimum_distance_m=0.03,
        ),
    )
    planner.oracle = collision_oracle
    planner.target_envelope_filter = _MinimumRightXEnvelopeFilter(0.8)
    outside = {side: np.zeros(6) for side in ("left", "right")}
    outside["right"][0] = 0.5
    inside = {side: value.copy() for side, value in outside.items()}
    inside["right"][0] = 0.9

    search_oracle = planner._path_search_oracle(planning_phase="grasp_target")

    assert search_oracle is not collision_oracle
    assert search_oracle.scene_hash == "scene:test"
    outside_check = search_oracle.check(outside)
    assert outside_check.safe is False
    assert outside_check.collision_check_available is True
    assert outside_check.reason == "path_target_safety_envelope_blocked"
    assert search_oracle.check(inside).safe is True
    ingress_oracle = planner._path_search_oracle(
        planning_phase="envelope_ingress", side="right", start_qpos_by_side=inside,
    )
    assert ingress_oracle is not collision_oracle
    assert ingress_oracle.check(inside).safe
    assert not ingress_oracle.check(outside).safe


@pytest.mark.parametrize("initial,next_adjustment,constraints,reason", [
    (.12, .10, ("absolute:y:max",), None),
    (.12, .123, ("absolute:y:max",), "path_exceeded_initial_outer_boundary"),
    (.12, .00025, ("absolute:y:max", "absolute:x:min"), None),
    (.12, .0008, ("absolute:y:max", "absolute:x:min"), "violation_constraint_set_expanded"),
    (.12, 0, (), None),
    (0, .000335, ("absolute:y:max",), "path_did_not_start_at_the_violation"),
])
def test_ingress_search_rejects_outer_excursions_before_path_found(initial, next_adjustment, constraints, reason):
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    planner.target_builder = _CartesianQposBuilder()
    planner.oracle = SimpleNamespace(scene_hash="scene:test", check=lambda _: JointConfigurationCheck(safe=True))
    planner.target_envelope_filter = _IngressEnvelopeFilter(
        (initial, next_adjustment),
        constraints_by_call=(("absolute:y:max",), constraints),
    )
    qpos = {side: np.zeros(6) for side in ("left", "right")}
    oracle = planner._path_search_oracle(planning_phase="envelope_ingress", side="right", start_qpos_by_side=qpos)
    check = oracle.check(qpos)
    assert check.collision_check_available
    assert check.safe == (reason is None)
    if reason is not None:
        assert check.reason == "path_target_safety_envelope_blocked"
        assert check.metadata["target_envelope"]["envelope_ingress_rejection"] == reason


def test_ingress_search_queries_do_not_mutate_global_best_or_entry_state():
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    planner.target_builder = _CartesianQposBuilder()
    planner.oracle = SimpleNamespace(scene_hash="scene:test", check=lambda _: JointConfigurationCheck(safe=True))
    planner.target_envelope_filter = _IngressEnvelopeFilter((.12, .05, 0, .11, .123))
    qpos = {side: np.zeros(6) for side in ("left", "right")}
    oracle = planner._path_search_oracle(planning_phase="envelope_ingress", side="right", start_qpos_by_side=qpos)
    # A random inside or near-boundary query is not committed path progress.
    assert oracle.check(qpos).safe
    assert oracle.check(qpos).safe
    assert oracle.check(qpos).safe
    assert not oracle.check(qpos).safe


def test_ingress_search_requires_a_valid_initial_projection():
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    planner.target_builder = _CartesianQposBuilder()
    planner.oracle = SimpleNamespace(scene_hash="scene:test", check=lambda _: JointConfigurationCheck(safe=True))
    planner.target_envelope_filter = _IngressEnvelopeFilter((.17,))
    qpos = {side: np.zeros(6) for side in ("left", "right")}
    with pytest.raises(RM65GraspTrajectoryPlanningError) as error:
        planner._path_search_oracle(planning_phase="envelope_ingress", side="right", start_qpos_by_side=qpos)
    assert error.value.details["envelope_ingress_rejection"] == "initial_distance_exceeds_limit"


def test_ingress_search_never_overrides_production_collision():
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    planner.target_builder = _CartesianQposBuilder()
    refusal = JointConfigurationCheck(safe=False, collision_check_available=True, reason="production_collision")
    planner.oracle = SimpleNamespace(scene_hash="scene:test", check=lambda _: refusal)
    planner.target_envelope_filter = _IngressEnvelopeFilter((.12, .10))
    qpos = {side: np.zeros(6) for side in ("left", "right")}
    oracle = planner._path_search_oracle(planning_phase="envelope_ingress", side="right", start_qpos_by_side=qpos)
    assert oracle.check(qpos) is refusal
    assert planner.target_envelope_filter.calls == 1


def test_grasp_search_oracle_fails_closed_when_envelope_is_unavailable() -> None:
    planner = _planner_without_assets(config=RM65GraspTrajectoryPlanningConfig())
    planner.target_builder = _LinearQposBuilder()
    planner.oracle = SimpleNamespace(
        scene_hash="scene:test",
        check=lambda _qpos: JointConfigurationCheck(safe=True),
    )

    def unavailable(_targets):
        raise RuntimeError("test envelope failure")

    planner.target_envelope_filter = unavailable
    qpos = {side: np.zeros(6) for side in ("left", "right")}

    check = planner._path_search_oracle(planning_phase="grasp_target").check(qpos)

    assert check.safe is False
    assert check.collision_check_available is False
    assert check.reason == "path_target_safety_envelope_unavailable"


def test_closest_tolerance_endpoint_must_also_be_inside_envelope() -> None:
    planner = _planner_without_assets(
        config=RM65GraspTrajectoryPlanningConfig(
            endpoint_position_tolerance_m=0.3,
        )
    )
    planner.target_builder = _LinearQposBuilder()
    planner.target_envelope_filter = _MinimumRightXEnvelopeFilter(0.8)
    current = {side: np.zeros(6) for side in ("left", "right")}
    target = np.eye(4)
    target[0, 3] = 1.0
    solved = np.zeros(6)
    solved[0] = 1.0

    candidate, position, orientation = planner._closest_tolerance_endpoint(
        qpos_by_side=current,
        side="right",
        target_tool_tcp_world=target,
        seed_qpos=np.zeros(6),
        solved_qpos=solved,
    )

    assert candidate["right"][0] == pytest.approx(0.8, abs=1e-8)
    assert position == pytest.approx(0.2, abs=1e-8)
    assert orientation == pytest.approx(0.0)


def test_endpoint_candidate_rejects_realized_pose_outside_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _planner_without_assets(
        config=RM65GraspTrajectoryPlanningConfig(
            maximum_endpoint_solve_cycles=1,
            endpoint_seed_count=1,
        )
    )
    planner.target_builder = _LinearQposBuilder()
    planner.target_envelope_filter = _MinimumRightXEnvelopeFilter(0.8)
    planner.oracle = SimpleNamespace(
        check=lambda _qpos: JointConfigurationCheck(safe=True)
    )
    current = {side: np.zeros(6) for side in ("left", "right")}
    unsafe_candidate = {side: value.copy() for side, value in current.items()}
    unsafe_candidate["right"][0] = 0.5
    monkeypatch.setattr(
        planner,
        "_endpoint_seed_qpos",
        lambda *_args, **_kwargs: (current,),
    )
    monkeypatch.setattr(
        planner,
        "_least_squares_endpoint_from_seed",
        lambda **_kwargs: (unsafe_candidate, 1, 0.001, 0.0),
    )
    target = np.eye(4)
    target[0, 3] = 1.0

    with pytest.raises(
        RM65GraspTrajectoryPlanningError,
        match="endpoint_safety_envelope_blocked",
    ) as raised:
        planner._endpoint_qpos_candidates(
            qpos_by_side=current,
            side="right",
            target_tool_tcp_world=target,
        )

    assert raised.value.details["attempts"][0]["outcome"] == (
        "endpoint_safety_envelope_blocked"
    )


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

    assert candidates[0].seed_index == 0
    assert {candidate.seed_index for candidate in candidates} == {0, 1, 2, 3, 4}
    assert all(candidate.solve_cycles == 1 for candidate in candidates)
    assert all(
        np.array_equal(candidate.qpos_by_side["left"], current["left"])
        for candidate in candidates
    )
    assert (
        len({tuple(candidate.qpos_by_side["right"]) for candidate in candidates}) == 5
    )
    # Endpoint optimization uses the read-only FK port and must not mutate the
    # online target builder's tracked state.
    assert builder.seed_history == []


@pytest.mark.parametrize(
    ("minimum_distance_m", "expected_reason"),
    (
        (0.003, "endpoint_clearance_blocked"),
        (0.0, "endpoint_collision_blocked"),
        (-0.003, "endpoint_collision_blocked"),
    ),
)
def test_endpoint_planner_distinguishes_positive_gap_from_penetration(
    monkeypatch: pytest.MonkeyPatch,
    minimum_distance_m: float,
    expected_reason: str,
) -> None:
    planner = _planner_without_assets(
        config=RM65GraspTrajectoryPlanningConfig(
            maximum_endpoint_solve_cycles=1,
            endpoint_seed_count=1,
        )
    )
    planner.target_builder = _MultistartBuilder()
    planner.target_envelope_filter = lambda targets: targets
    planner.oracle = SimpleNamespace(
        check=lambda _qpos: JointConfigurationCheck(
            safe=False,
            collision_check_available=True,
            reason="collision_clearance_violated",
            minimum_distance_m=minimum_distance_m,
            metadata={
                "closest_geometry_names": ("arm", "camera_pole"),
                "closest_obstacle_id": "camera_pole",
            },
        )
    )
    current = {side: np.zeros(6) for side in ("left", "right")}
    solved = {side: value.copy() for side, value in current.items()}
    monkeypatch.setattr(
        planner,
        "_endpoint_seed_qpos",
        lambda *_args, **_kwargs: (current,),
    )
    monkeypatch.setattr(
        planner,
        "_least_squares_endpoint_from_seed",
        lambda **_kwargs: (solved, 1, 0.001, 0.1),
    )

    with pytest.raises(
        RM65GraspTrajectoryPlanningError,
        match=expected_reason,
    ) as error:
        planner._endpoint_qpos_candidates(
            qpos_by_side=current,
            side="right",
            target_tool_tcp_world=np.eye(4),
        )

    assert error.value.details["attempts"][0]["outcome"] == expected_reason
    assert error.value.details["collision_examples"][0][
        "minimum_distance_m"
    ] == pytest.approx(minimum_distance_m)


def test_endpoint_planner_uses_pose_tolerance_for_positive_gap_clearance_rescue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _planner_without_assets(
        config=RM65GraspTrajectoryPlanningConfig(
            maximum_endpoint_solve_cycles=1,
            endpoint_seed_count=1,
        )
    )
    planner.target_builder = _CartesianQposBuilder()
    planner.target_envelope_filter = lambda targets: targets
    current = {side: np.zeros(6) for side in ("left", "right")}

    def solve(*, target_tool_tcp_world, **_kwargs):
        target = np.asarray(target_tool_tcp_world, dtype=float)
        candidate = {side: value.copy() for side, value in current.items()}
        candidate["right"][:3] = target[:3, 3]
        candidate["right"][3:] = grasp_trajectory_module.Rotation.from_matrix(
            target[:3, :3]
        ).as_rotvec()
        return candidate, 1, 0.0, 0.0

    def check(qpos):
        displaced = not np.allclose(qpos["right"], 0.0, atol=1e-12, rtol=0.0)
        return JointConfigurationCheck(
            safe=displaced,
            collision_check_available=True,
            reason=(
                "collision_clearance_satisfied"
                if displaced
                else "collision_clearance_violated"
            ),
            minimum_distance_m=0.03 if displaced else 0.003,
            metadata={
                "closest_geometry_names": ("arm", "camera_pole"),
                "closest_obstacle_id": "camera_pole",
            },
        )

    planner.oracle = SimpleNamespace(check=check)
    monkeypatch.setattr(
        planner,
        "_endpoint_seed_qpos",
        lambda *_args, **_kwargs: (current,),
    )
    monkeypatch.setattr(planner, "_least_squares_endpoint_from_seed", solve)

    candidates = planner._endpoint_qpos_candidates(
        qpos_by_side=current,
        side="right",
        target_tool_tcp_world=np.eye(4),
    )

    assert len(candidates) == 1
    assert candidates[0].orientation_residual_deg < (
        planner.config.endpoint_orientation_tolerance_deg
    )
    assert planner.oracle.check(candidates[0].qpos_by_side).safe is True


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


def test_endpoint_planner_uses_global_deterministic_seed_coverage_by_default() -> None:
    config = RM65GraspTrajectoryPlanningConfig()

    assert config.endpoint_seed_count == 24
    assert config.maximum_endpoint_candidates == 8
    assert config.path_planning_time_budget_sec == pytest.approx(120.0)
    assert config.pregrasp_distance_m == pytest.approx(0.06)
    assert config.minimum_pregrasp_distance_m == pytest.approx(0.02)
    assert config.pregrasp_orientation_tolerance_deg == pytest.approx(3.0)
    assert config.maximum_approach_orientation_error_deg == pytest.approx(3.0)
    assert config.endpoint_orientation_tolerance_deg == pytest.approx(1.0)
    assert config.maximum_path_envelope_ingress_distance_m == pytest.approx(0.16)


def test_recorded_candidate_zero_uses_bounded_pregrasp_convergence() -> None:
    profile = _real_profile(
        envelope_profile_name="agent_grasp_right_absolute_standard_v1.json"
    )
    builder = build_exclusive_rm65_skill_target_builder(profile)
    planner = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=builder,
        target_envelope_filter=builder.pre_ik_target_safety_filter,
    )

    trajectory = planner(
        qpos_by_side=RECORDED_CANDIDATE_0_QPOS,
        side="right",
        target_tool_tcp_world=RECORDED_CANDIDATE_0_TARGET,
        speed_scale=0.2,
    )

    assert len(trajectory.time_from_start_sec) > 1
    final_tcp = builder.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=trajectory.qpos_by_side["left"][-1],
        right_qpos=trajectory.qpos_by_side["right"][-1],
    )["right"]
    position_error_m, orientation_error_deg = pose_delta_metrics(
        final_tcp,
        RECORDED_CANDIDATE_0_TARGET,
    )
    assert position_error_m <= planner.config.endpoint_position_tolerance_m
    assert orientation_error_deg <= planner.config.endpoint_orientation_tolerance_deg


def test_path_leg_shares_remaining_wall_time_and_stops_after_timeout(
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
    current = {side: np.zeros(6) for side in ("left", "right")}
    target = {side: value.copy() for side, value in current.items()}
    target["right"] = np.full(6, 0.1)
    endpoint = grasp_trajectory_module._EndpointIKCandidate(
        qpos_by_side=target,
        seed_index=0,
        solve_cycles=10,
        position_residual_m=0.001,
        orientation_residual_deg=0.2,
    )
    planner._endpoint_qpos_candidates = lambda **_kwargs: (endpoint,)
    calls: list[dict[str, object]] = []

    def fake_plan(**kwargs):
        calls.append(kwargs)
        return JointPathPlanResult(
            status=JointPathStatus.PATH_SEARCH_EXHAUSTED,
            start_qpos_by_component=kwargs["start_qpos_by_component"],
            target_qpos_by_component=kwargs["target_qpos_by_component"],
            movable_components=("right",),
            scene_hash="scene:test",
            reason="time_budget_exhausted",
            metadata={"search_termination": "time_budget_exhausted"},
        )

    monkeypatch.setattr(
        grasp_trajectory_module,
        "plan_collision_aware_joint_path",
        fake_plan,
    )

    with pytest.raises(
        RM65GraspTrajectoryPlanningError,
        match="path_planning_time_budget_exhausted",
    ) as error:
        planner._plan_path_leg(
            qpos_by_side=current,
            side="right",
            target_tool_tcp_world=np.eye(4),
            planning_phase="grasp_target",
            deadline=10.0,
            monotonic=lambda: 2.0,
        )

    assert len(calls) == 1
    assert calls[0]["config"].time_budget_sec == pytest.approx(8.0)
    assert calls[0]["monotonic"]() == pytest.approx(2.0)
    assert error.value.details["search_attempts"][0]["reason"] == (
        "time_budget_exhausted"
    )


def test_endpoint_planner_solves_real_workspace_pose_outside_local_qp_basin() -> None:
    profile = _real_profile()
    builder = build_exclusive_rm65_skill_target_builder(profile)
    planner = RM65GraspJointTrajectoryPlanner(
        profile=profile,
        planning_target_builder=builder,
        target_envelope_filter=builder.pre_ik_target_safety_filter,
        config=RM65GraspTrajectoryPlanningConfig(
            endpoint_seed_count=16,
            maximum_endpoint_candidates=4,
        ),
    )
    target = np.eye(4, dtype=float)
    target[:3, :3] = np.asarray(
        (
            (-0.9396926168, 0.3420201541, 0.0),
            (0.3420201541, 0.9396926168, 0.0),
            (0.0, 0.0, -1.0),
        )
    )
    target[:3, 3] = (-0.3287666177, 0.0735600430, 0.1556847929)

    candidates = planner._endpoint_qpos_candidates(
        qpos_by_side=SAFE_QPOS,
        side="right",
        target_tool_tcp_world=target,
    )

    assert candidates
    for candidate in candidates:
        realized = builder.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=candidate.qpos_by_side["left"],
            right_qpos=candidate.qpos_by_side["right"],
        )["right"]
        position_error_m, orientation_error_deg = pose_delta_metrics(realized, target)
        assert position_error_m <= planner.config.endpoint_position_tolerance_m
        assert (
            orientation_error_deg <= planner.config.endpoint_orientation_tolerance_deg
        )
        assert planner.oracle.check(candidate.qpos_by_side).safe is True

    reachability = planner.assess_endpoint_ik(
        qpos_by_side=SAFE_QPOS,
        side="right",
        target_tool_tcp_world=target,
    )
    assert reachability.classification == "standard_reachable"


def test_path_planner_shares_goal_set_in_every_expanding_corridor(
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
        # Keep intermediate mock corridors distinct from the final full range.
        sample_padding_rad=0.1,
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
    planner._revalidate = lambda _waypoints, **_kwargs: {
        "sample_count": 2,
        "minimum_distance_m": 0.03,
    }
    calls: list[dict[str, object]] = []

    def fake_plan(**kwargs):
        calls.append(kwargs)
        target = kwargs["alternative_target_qpos_by_component"][0]
        found = (
            np.array_equal(target["right"], second_target["right"])
            and np.array_equal(
                kwargs["joint_lower_by_component"]["right"],
                grasp_trajectory_module.RM65_ABSOLUTE_Q_MIN_RAD,
            )
            and np.array_equal(
                kwargs["joint_upper_by_component"]["right"],
                grasp_trajectory_module.RM65_ABSOLUTE_Q_MAX_RAD,
            )
        )
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
            metadata={"selected_target_index": 1} if found else {},
        )

    monkeypatch.setattr(
        grasp_trajectory_module,
        "plan_collision_aware_joint_path",
        fake_plan,
    )
    monkeypatch.setattr(
        grasp_trajectory_module,
        "load_joint_velocity_limit_profiles",
        lambda: {"test-velocity": SimpleNamespace(max_velocity_deg_s=(30.0,) * 6)},
    )

    trajectory = planner(
        qpos_by_side=current,
        side="right",
        target_tool_tcp_world=np.eye(4),
        speed_scale=0.2,
    )

    assert len(calls) == 4
    assert [call["config"].seed for call in calls] == [0, 1, 2, 3]
    for index, call in enumerate(calls):
        np.testing.assert_array_equal(
            call["target_qpos_by_component"]["right"],
            first_target["right"],
        )
        np.testing.assert_array_equal(
            call["alternative_target_qpos_by_component"][0]["right"],
            second_target["right"],
        )
    first_widths = [
        np.asarray(call["joint_upper_by_component"]["right"])
        - np.asarray(call["joint_lower_by_component"]["right"])
        for call in calls
    ]
    assert np.all(first_widths[1] > first_widths[0])
    assert np.all(first_widths[2] >= first_widths[1])
    assert np.any(first_widths[2] > first_widths[1])
    assert np.all(first_widths[3] >= first_widths[2])
    np.testing.assert_allclose(trajectory.qpos_by_side["right"][-1], 0.2)
