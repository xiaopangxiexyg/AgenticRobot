from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_contracts.pose_frames import load_pose_frame_contract
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
from sim.sim_real_tests.mujoco import exact_shadow_trajectory as exact_shadow
from sim.sim_real_tests.mujoco.arbitrary_grasp_pose_motion import (
    OBSTACLE_SCENE_FRAME_ID,
    OBSTACLE_SCENE_FRAME_SPEC_HASH,
    build_obstacle_collision_oracle,
    build_shadow_initialized_control,
)
from sim.sim_real_tests.mujoco.exact_shadow_trajectory import (
    MujocoExactQposExecutionConfig,
    MujocoShadowCompatibilityExpectation,
    MujocoShadowModelBinding,
    execute_registered_shadow_trajectory,
    verify_mujoco_shadow_compatibility,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_JOINT_NAMES_BY_SIDE = {
    side: tuple(f"{side}_joint{i}" for i in range(1, 7))
    for side in ("left", "right")
}
_DESK_FROM_BASE_BY_SIDE = {
    side: tuple(np.eye(4).reshape(-1)) for side in ("left", "right")
}
_TCP_BY_SIDE = {
    "left": (0.0, 0.0, 0.14, 0.0, 0.0, 0.0),
    "right": (0.0, 0.0, 0.14, 0.0, 0.0, 0.0),
}
_GRIPPER_SIDE_BY_NAME = {
    "left_gripper": "left",
    "right_gripper": "right",
}


def _identity(*, robot_model_hash: str = "robot:test") -> ShadowExecutionIdentity:
    return ShadowExecutionIdentity(
        station_revision="station:test",
        reference_frame_id="desk",
        reference_frame_spec_hash="frame-spec:test",
        frame_graph_hash="frame-graph:test",
        tool_profile_hash_by_side={"left": "tool:left", "right": "tool:right"},
        robot_model_hash=robot_model_hash,
        collision_model_hash="collision:test",
        joint_layout_hash="joints:test",
        gripper_model_hash="gripper:test",
    )


def _snapshot(identity: ShadowExecutionIdentity | None = None):
    from sim.mujoco.control.desk_tcp_position import DEFAULT_SIM_HOME_QPOS_BY_SIDE

    return AuthoritativeRobotStateSnapshot(
        qpos_by_side=DEFAULT_SIM_HOME_QPOS_BY_SIDE,
        gripper_close_by_side={"left": 0.2, "right": 0.3},
        arm_measured_at=1.0,
        gripper_measured_at_by_side={"left": 1.0, "right": 1.0},
        available_at=1.01,
        arm_evidence_refs=("arm:test",),
        gripper_evidence_refs=("gripper:left", "gripper:right"),
        identity=identity or _identity(),
    )


def _trajectory(snapshot=None, *, delta: float = 0.004, scene_hash="scene:test"):
    initial = snapshot or _snapshot()
    target = {side: initial.qpos_by_side[side].copy() for side in ("left", "right")}
    target["left"][0] += delta
    return RegisteredShadowJointTrajectory(
        trajectory_id="trajectory:test",
        created_at=2.0,
        initial_snapshot=initial,
        scene_hash=scene_hash,
        target_hash="target:test",
        planner_config_hash="planner:test",
        moving_sides=("left",),
        points=(
            RegisteredJointTrajectoryPoint(
                time_from_start_sec=0.0,
                qpos_by_side=initial.qpos_by_side,
            ),
            RegisteredJointTrajectoryPoint(
                time_from_start_sec=0.05,
                qpos_by_side=target,
            ),
        ),
    )


def _binding(*, scene_revision="scene:test", **changes):
    values = {
        "station_revision": "station:test",
        "reference_frame_id": "desk",
        "desk_from_arm_base_by_side": _DESK_FROM_BASE_BY_SIDE,
        "joint_names_by_side": _JOINT_NAMES_BY_SIDE,
        "tool_profile_id_by_side": {
            "left": "DMTAC_M:left",
            "right": "DMTAC_M:right",
        },
        "tool_tcp_pose_flange_m_rad_by_side": _TCP_BY_SIDE,
        "gripper_profile_name": "dmtac:test",
        "gripper_side_by_name": _GRIPPER_SIDE_BY_NAME,
        "scene_revision": scene_revision,
        "simulation_scene_profile_name": "scene:test",
        "collision_profile_name": "collision:test",
        "evidence_refs": ("model-binding:test",),
    }
    values.update(changes)
    return MujocoShadowModelBinding(**values)


def _binding_from_independent_config(*, scene_revision):
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
        evidence_refs=("real-owner-compatibility:test",),
    )


@pytest.mark.skipif(
    not DEFAULT_INSPIRE_V3_DUAL_ARM_URDF.is_file(),
    reason="local canonical Inspire-v3 URDF asset is unavailable",
)
def test_exact_registered_qpos_executes_in_mujoco_and_finishes_in_hold() -> None:
    obstacle_scene = ObstacleSceneSnapshot(
        reference_frame_id=OBSTACLE_SCENE_FRAME_ID,
        reference_frame_spec_hash=OBSTACLE_SCENE_FRAME_SPEC_HASH,
        revision=0,
    )
    scene_revision = obstacle_scene.semantic_content_hash
    trajectory = _trajectory(scene_hash=scene_revision)
    control = build_shadow_initialized_control(trajectory.initial_snapshot)
    control.shadow_obstacle_oracle = build_obstacle_collision_oracle(
        control,
        obstacle_scene,
    )
    try:
        gripper_profile = control.rig.config.gripper_actuation_profile
        assert (
            gripper_profile.name
            == "dmtac_bimanual_position_eg2_4b2_cad_20260818.v1"
        )
        assert gripper_profile.side_by_gripper == {
            "left_eg2_4b2_cad": "left",
            "right_eg2_4b2_cad": "right",
        }
        assert all(
            len(gripper_profile.joint_names_by_gripper[name]) == 6
            for name in gripper_profile.side_by_gripper
        )
        receipt = execute_registered_shadow_trajectory(
            control=control,
            trajectory=trajectory,
            model_binding=_binding_from_independent_config(
                scene_revision=scene_revision,
            ),
            config=MujocoExactQposExecutionConfig(
                tracking_tolerance_rad=3e-3,
                final_settle_timeout_sec=4.0,
                terminal_hold_timeout_sec=1.0,
                terminal_hold_dwell_sec=0.02,
                terminal_velocity_tolerance_rad_s=5e-3,
            ),
        )
    finally:
        control.close()

    assert receipt.status is ShadowSimulationStatus.COMPLETED
    assert receipt.executed_point_count == len(trajectory.points)
    assert receipt.exact_planned_qpos_execution_confirmed is True
    assert receipt.model_identity_confirmed is True
    assert receipt.terminal_hold_confirmed is True


class _Collision:
    def __init__(self, safe: bool):
        self.safe = safe

    def to_dict(self):
        return {
            "safe": self.safe,
            "minimum_distance_m": 0.02 if self.safe else -0.001,
            "pair_class": None if self.safe else "robot:obstacle",
        }


class _Projector:
    def __init__(self, blocked_call: int | None = None):
        self.calls = 0
        self.blocked_call = blocked_call

    def check_qpos(self, _data, _target):
        self.calls += 1
        return _Collision(self.calls != self.blocked_call)


class _ObstacleOracle:
    def __init__(self, scene_hash, blocked_call=None):
        self.scene_hash = scene_hash
        self.collision_model_is_independent = True
        self.calls = 0
        self.blocked_call = blocked_call

    def check(self, _qpos_by_side):
        self.calls += 1
        safe = self.calls != self.blocked_call
        return SimpleNamespace(
            safe=safe,
            collision_check_available=True,
            to_dict=lambda: {
                "safe": safe,
                "collision_check_available": True,
                "minimum_distance_m": 0.03 if safe else -0.002,
            },
        )


class _FakeScene:
    def __init__(self, projector):
        self.data = SimpleNamespace(time=0.0)
        self._projector = projector
        self.grippers = {"left_gripper": 0.2, "right_gripper": 0.3}
        self.arm_specs = {
            side: SimpleNamespace(joint_names=_JOINT_NAMES_BY_SIDE[side])
            for side in ("left", "right")
        }
        self.scene_spec = SimpleNamespace(profile_name="scene:test")
        self.conversion = SimpleNamespace(scene_profile_name="scene:test")

    def collision_projector(self):
        return self._projector

    def kinematic_controlled_qpos(self):
        return np.zeros(14)

    def kinematic_target_vector(self, _base, qpos_by_side, **_kwargs):
        return np.concatenate((qpos_by_side["left"], qpos_by_side["right"]))

    def get_gripper_close(self, name):
        return self.grippers[name]


class _FakeRig:
    def __init__(self, scene, initial, *, frozen=False, hold_drift=False):
        self.scene = scene
        self.config = SimpleNamespace(
            timing=SimpleNamespace(physics_timestep_sec=0.01),
            actuation_profile=SimpleNamespace(
                joint_names_by_side=_JOINT_NAMES_BY_SIDE,
            ),
            gripper_actuation_profile=SimpleNamespace(
                name="dmtac:test",
                side_by_gripper=_GRIPPER_SIDE_BY_NAME,
            ),
        )
        self.measured = {side: value.copy() for side, value in initial.items()}
        self.accepted = {side: value.copy() for side, value in initial.items()}
        self.frozen = frozen
        self.hold_drift = hold_drift
        self.hold_active = False
        self.hold_calls = 0

    def measured_qpos_by_side(self):
        return {side: value.copy() for side, value in self.measured.items()}

    def accepted_qcmd_by_side(self):
        return {side: value.copy() for side, value in self.accepted.items()}

    def send_commands(self, command):
        for side in ("left", "right"):
            self.accepted[side] = np.asarray(
                command.commands[f"{side}_arm"].value, dtype=float
            ).copy()
        return {
            name: SimpleNamespace(accepted=True)
            for name in ("left_arm", "right_arm", "left_hand", "right_hand")
        }

    def hold_motion(self, _reason):
        self.hold_calls += 1
        self.hold_active = True
        self.accepted = {
            side: value.copy() for side, value in self.measured.items()
        }


class _FakeControl:
    def __init__(
        self,
        trajectory,
        *,
        blocked_call=None,
        frozen=False,
        hold_drift=False,
        gripper_drift_at_tick=None,
        obstacle_blocked_call=None,
    ):
        self.scene = _FakeScene(_Projector(blocked_call))
        self.rig = _FakeRig(
            self.scene,
            trajectory.initial_snapshot.qpos_by_side,
            frozen=frozen,
            hold_drift=hold_drift,
        )
        self.station_pose_contract = SimpleNamespace(
            source_id="station:test",
            canonical_absolute_frame="desk",
            source=SimpleNamespace(
                effective_raw_to_canonical_desk_transform=lambda _side: np.eye(4)
            ),
        )
        self.contract = SimpleNamespace(
            scene=SimpleNamespace(profile_name="scene:test")
        )
        self.config = SimpleNamespace(
            left_tool_profile=DEFAULT_TOOL_PROFILE_BY_SIDE["left"],
            right_tool_profile=DEFAULT_TOOL_PROFILE_BY_SIDE["right"],
        )
        self.retargeter = SimpleNamespace(
            config=SimpleNamespace(
                left_tool_tcp_pose_flange_m_rad=np.asarray(_TCP_BY_SIDE["left"]),
                right_tool_tcp_pose_flange_m_rad=np.asarray(_TCP_BY_SIDE["right"]),
            ),
            profile=SimpleNamespace(name="collision:test"),
        )
        self.shadow_obstacle_oracle = _ObstacleOracle(
            trajectory.scene_hash,
            blocked_call=obstacle_blocked_call,
        )
        self.tick = 0
        self.gripper_drift_at_tick = gripper_drift_at_tick

    def step_physics(self):
        self.tick += 1
        self.scene.data.time += self.rig.config.timing.physics_timestep_sec
        if self.gripper_drift_at_tick == self.tick:
            self.scene.grippers["right_gripper"] += 0.1
        if self.rig.hold_active and self.rig.hold_drift:
            self.rig.measured["left"][0] += 0.01
            self.rig.accepted["left"][0] += 0.01
            return
        if not self.rig.frozen:
            self.rig.measured = {
                side: value.copy() for side, value in self.rig.accepted.items()
            }


def test_artifact_identity_hash_does_not_gate_compatible_execution() -> None:
    trajectory = _trajectory(
        _snapshot(identity=_identity(robot_model_hash="unrelated-artifact-hash"))
    )
    control = _FakeControl(trajectory)

    receipt = execute_registered_shadow_trajectory(
        control=control,
        trajectory=trajectory,
        model_binding=_binding(),
        config=MujocoExactQposExecutionConfig(
            terminal_hold_timeout_sec=0.1,
            terminal_hold_dwell_sec=0.01,
        ),
    )

    assert receipt.status is ShadowSimulationStatus.COMPLETED
    assert receipt.executed_point_count == len(trajectory.points)
    assert receipt.model_identity_confirmed is True
    assert receipt.terminal_hold_confirmed is True
    assert control.rig.hold_calls == 1


@pytest.mark.parametrize(
    ("changes", "reason"),
    (
        ({"station_revision": "station:other"}, "station_revision_mismatch"),
        ({"reference_frame_id": "world"}, "reference_frame_mismatch"),
        (
            {
                "desk_from_arm_base_by_side": {
                    "left": tuple((np.eye(4) + np.eye(4) * 0.01).reshape(-1)),
                    "right": _DESK_FROM_BASE_BY_SIDE["right"],
                }
            },
            "reference_frame_spec_mismatch",
        ),
        (
            {
                "joint_names_by_side": {
                    "left": tuple(reversed(_JOINT_NAMES_BY_SIDE["left"])),
                    "right": _JOINT_NAMES_BY_SIDE["right"],
                }
            },
            "joint_layout_mismatch",
        ),
        (
            {
                "tool_profile_id_by_side": {
                    "left": "other:left",
                    "right": "DMTAC_M:right",
                }
            },
            "tool_profile_mismatch",
        ),
        (
            {
                "tool_tcp_pose_flange_m_rad_by_side": {
                    "left": (0.0, 0.0, 0.15, 0.0, 0.0, 0.0),
                    "right": _TCP_BY_SIDE["right"],
                }
            },
            "tool_tcp_mismatch",
        ),
        ({"gripper_profile_name": "other"}, "gripper_profile_mismatch"),
        (
            {
                "gripper_side_by_name": {
                    "left_gripper": "right",
                    "right_gripper": "left",
                }
            },
            "gripper_assignment_mismatch",
        ),
        (
            {"simulation_scene_profile_name": "scene:other"},
            "simulation_scene_profile_mismatch",
        ),
        (
            {"scene_revision": "scene:other"},
            "simulation_scene_revision_mismatch",
        ),
        ({"collision_profile_name": "other"}, "collision_profile_mismatch"),
    ),
)
def test_minimal_compatibility_mismatch_holds_without_dispatch(
    changes, reason
) -> None:
    trajectory = _trajectory()
    control = _FakeControl(trajectory)
    receipt = execute_registered_shadow_trajectory(
        control=control,
        trajectory=trajectory,
        model_binding=_binding(**changes),
        config=MujocoExactQposExecutionConfig(
            terminal_hold_timeout_sec=0.1,
            terminal_hold_dwell_sec=0.01,
        ),
    )

    assert receipt.status is ShadowSimulationStatus.MODEL_IDENTITY_MISMATCH
    assert receipt.model_identity_confirmed is False
    assert receipt.executed_point_count == 0
    assert any(reason in ref for ref in receipt.evidence_refs)
    assert control.rig.hold_calls == 1


def test_internal_joint_layout_disagreement_fails_closed() -> None:
    trajectory = _trajectory()
    control = _FakeControl(trajectory)
    control.rig.config.actuation_profile.joint_names_by_side = {
        "left": tuple(reversed(_JOINT_NAMES_BY_SIDE["left"])),
        "right": _JOINT_NAMES_BY_SIDE["right"],
    }

    compatible, reasons, _evidence = verify_mujoco_shadow_compatibility(
        control, trajectory, _binding()
    )

    assert compatible is False
    assert reasons == ("joint_layout_mismatch",)


def test_missing_runtime_observable_fails_closed() -> None:
    trajectory = _trajectory()
    control = _FakeControl(trajectory)
    del control.retargeter

    compatible, reasons, evidence = verify_mujoco_shadow_compatibility(
        control, trajectory, _binding()
    )

    assert compatible is False
    assert reasons == ("compatibility_observation_failed",)
    assert any("compatibility_observation_failed" in ref for ref in evidence)


def test_expectation_cannot_inject_executor_owned_reason_codes() -> None:
    with pytest.raises(ValueError, match="executor-owned reason namespace"):
        _binding(
            evidence_refs=(
                "mujoco-shadow-compatibility-reason:forged_reason",
            )
        )


def test_dynamic_scene_revision_is_observed_from_simulation_control() -> None:
    trajectory = _trajectory()
    control = _FakeControl(trajectory)
    control.shadow_obstacle_oracle.scene_hash = "scene:stale"

    compatible, reasons, _evidence = verify_mujoco_shadow_compatibility(
        control, trajectory, _binding()
    )

    assert compatible is False
    assert reasons == ("simulation_scene_revision_mismatch",)


def test_trajectory_revision_is_distinct_from_observed_scene_revision() -> None:
    trajectory = _trajectory()
    control = _FakeControl(trajectory)
    control.shadow_obstacle_oracle.scene_hash = "scene:new"

    compatible, reasons, _evidence = verify_mujoco_shadow_compatibility(
        control,
        trajectory,
        _binding(scene_revision="scene:new"),
    )

    assert compatible is False
    assert reasons == ("trajectory_scene_revision_mismatch",)


def test_loaded_tool_profile_tcp_must_match_effective_runtime_tcp() -> None:
    trajectory = _trajectory()
    control = _FakeControl(trajectory)
    changed_tcp = {
        "left": (0.0, 0.0, 0.15, 0.0, 0.0, 0.0),
        "right": _TCP_BY_SIDE["right"],
    }
    control.retargeter.config.left_tool_tcp_pose_flange_m_rad = np.asarray(
        changed_tcp["left"]
    )

    compatible, reasons, _evidence = verify_mujoco_shadow_compatibility(
        control,
        trajectory,
        _binding(tool_tcp_pose_flange_m_rad_by_side=changed_tcp),
    )

    assert compatible is False
    assert reasons == ("tool_profile_runtime_tcp_mismatch",)


def test_compatibility_is_observed_once_not_per_trajectory_point(monkeypatch) -> None:
    trajectory = _trajectory()
    control = _FakeControl(trajectory)
    original = exact_shadow._observe_mujoco_shadow_compatibility
    calls = 0

    def counted(observed_control):
        nonlocal calls
        calls += 1
        return original(observed_control)

    monkeypatch.setattr(
        exact_shadow,
        "_observe_mujoco_shadow_compatibility",
        counted,
    )
    receipt = execute_registered_shadow_trajectory(
        control=control,
        trajectory=trajectory,
        model_binding=_binding(),
        config=MujocoExactQposExecutionConfig(
            terminal_hold_timeout_sec=0.1,
            terminal_hold_dwell_sec=0.01,
        ),
    )

    assert receipt.status is ShadowSimulationStatus.COMPLETED
    assert calls == 1


def test_intermediate_collision_is_distinct_from_target_collision() -> None:
    base = _trajectory()
    middle_qpos = {
        side: base.points[0].qpos_by_side[side].copy() for side in ("left", "right")
    }
    middle_qpos["left"][0] += 0.002
    trajectory = replace(
        base,
        points=(
            base.points[0],
            RegisteredJointTrajectoryPoint(
                time_from_start_sec=0.025,
                qpos_by_side=middle_qpos,
            ),
            RegisteredJointTrajectoryPoint(
                time_from_start_sec=0.05,
                qpos_by_side=base.points[-1].qpos_by_side,
            ),
        ),
    )

    receipt = execute_registered_shadow_trajectory(
        control=_FakeControl(trajectory, blocked_call=2),
        trajectory=trajectory,
        model_binding=_binding(),
        config=MujocoExactQposExecutionConfig(
            terminal_hold_timeout_sec=0.1,
            terminal_hold_dwell_sec=0.01,
            collision_sample_max_joint_step_rad=1.0,
        ),
    )

    assert receipt.status is ShadowSimulationStatus.PATH_COLLISION_BLOCKED
    assert receipt.executed_point_count == 0
    assert receipt.minimum_clearance_m == pytest.approx(-0.001)


def test_target_collision_is_typed_before_any_point_is_dispatched() -> None:
    trajectory = _trajectory()
    receipt = execute_registered_shadow_trajectory(
        control=_FakeControl(trajectory, blocked_call=2),
        trajectory=trajectory,
        model_binding=_binding(),
        config=MujocoExactQposExecutionConfig(
            terminal_hold_timeout_sec=0.1,
            terminal_hold_dwell_sec=0.01,
            collision_sample_max_joint_step_rad=1.0,
        ),
    )

    assert receipt.status is ShadowSimulationStatus.PATH_TARGET_COLLISION_BLOCKED
    assert receipt.executed_point_count == 0


def test_bound_obstacle_oracle_collision_is_checked_before_dispatch() -> None:
    trajectory = _trajectory()
    control = _FakeControl(trajectory, obstacle_blocked_call=2)
    receipt = execute_registered_shadow_trajectory(
        control=control,
        trajectory=trajectory,
        model_binding=_binding(),
        config=MujocoExactQposExecutionConfig(
            terminal_hold_timeout_sec=0.1,
            terminal_hold_dwell_sec=0.01,
            collision_sample_max_joint_step_rad=1.0,
        ),
    )

    assert receipt.status is ShadowSimulationStatus.PATH_TARGET_COLLISION_BLOCKED
    assert receipt.executed_point_count == 0
    assert receipt.minimum_clearance_m == pytest.approx(-0.002)
    assert control.scene._projector.calls == 2
    assert control.shadow_obstacle_oracle.calls == 2


def test_between_point_collision_is_found_by_dense_joint_interpolation() -> None:
    trajectory = _trajectory()
    receipt = execute_registered_shadow_trajectory(
        control=_FakeControl(trajectory, blocked_call=2),
        trajectory=trajectory,
        model_binding=_binding(),
        config=MujocoExactQposExecutionConfig(
            terminal_hold_timeout_sec=0.1,
            terminal_hold_dwell_sec=0.01,
            collision_sample_max_joint_step_rad=0.001,
        ),
    )

    assert receipt.status is ShadowSimulationStatus.PATH_COLLISION_BLOCKED
    assert receipt.executed_point_count == 0


def test_start_state_drift_is_rejected_before_collision_or_dispatch() -> None:
    trajectory = _trajectory()
    control = _FakeControl(trajectory)
    control.rig.measured["left"][0] += 0.01

    receipt = execute_registered_shadow_trajectory(
        control=control,
        trajectory=trajectory,
        model_binding=_binding(),
        config=MujocoExactQposExecutionConfig(
            terminal_hold_timeout_sec=0.1,
            terminal_hold_dwell_sec=0.01,
        ),
    )

    assert receipt.status is ShadowSimulationStatus.TRAJECTORY_START_MISMATCH
    assert receipt.executed_point_count == 0
    assert control.scene._projector.calls == 0


def test_final_tracking_timeout_is_typed_and_terminally_holds() -> None:
    trajectory = _trajectory(delta=0.1)
    control = _FakeControl(trajectory, frozen=True)

    receipt = execute_registered_shadow_trajectory(
        control=control,
        trajectory=trajectory,
        model_binding=_binding(),
        config=MujocoExactQposExecutionConfig(
            final_settle_timeout_sec=0.03,
            terminal_hold_timeout_sec=0.1,
            terminal_hold_dwell_sec=0.01,
        ),
    )

    assert receipt.status is ShadowSimulationStatus.POINT_TRACKING_TIMEOUT
    assert receipt.executed_point_count == len(trajectory.points)
    assert receipt.exact_planned_qpos_execution_confirmed is False
    assert receipt.terminal_hold_confirmed is True


def test_measured_gripper_drift_aborts_the_registered_timeline() -> None:
    trajectory = _trajectory()
    control = _FakeControl(trajectory, gripper_drift_at_tick=2)

    receipt = execute_registered_shadow_trajectory(
        control=control,
        trajectory=trajectory,
        model_binding=_binding(),
        config=MujocoExactQposExecutionConfig(
            final_settle_timeout_sec=0.1,
            terminal_hold_timeout_sec=0.1,
            terminal_hold_dwell_sec=0.01,
        ),
    )

    assert receipt.status is ShadowSimulationStatus.GRIPPER_HOLD_DRIFT
    assert receipt.executed_point_count == 1
    assert receipt.exact_planned_qpos_execution_confirmed is False


def test_terminal_hold_failure_cannot_be_reported_as_completed() -> None:
    trajectory = _trajectory()
    control = _FakeControl(trajectory, hold_drift=True)

    receipt = execute_registered_shadow_trajectory(
        control=control,
        trajectory=trajectory,
        model_binding=_binding(),
        config=MujocoExactQposExecutionConfig(
            final_settle_timeout_sec=0.1,
            terminal_hold_timeout_sec=0.03,
            terminal_hold_dwell_sec=0.02,
        ),
    )

    assert receipt.status is ShadowSimulationStatus.TERMINAL_HOLD_FAILED
    assert receipt.executed_point_count == len(trajectory.points)
    assert receipt.exact_planned_qpos_execution_confirmed is True
    assert receipt.terminal_hold_confirmed is False
