from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from control_station.real_shadow_acceptance import (
    RealMotionDisposition,
    RealShadowAcceptanceStatus,
    run_real_shadow_acceptance,
)
from robot_motion.shadow_execution import (
    AuthoritativeRobotStateSnapshot,
    RegisteredJointTrajectoryPoint,
    RegisteredShadowJointTrajectory,
    ShadowExecutionIdentity,
    ShadowSimulationStatus,
)
from sim.mujoco.control.desk_tcp_position import DEFAULT_INSPIRE_V3_DUAL_ARM_URDF
from sim.sim_real_tests.mujoco.arbitrary_grasp_pose_motion import (
    build_shadow_initialized_control,
)
from sim.sim_real_tests.mujoco.exact_shadow_trajectory import (
    MujocoExactQposExecutionConfig,
    MujocoShadowModelBinding,
    execute_registered_shadow_trajectory,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


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


def _trajectory(snapshot=None, *, delta: float = 0.004):
    initial = snapshot or _snapshot()
    target = {side: initial.qpos_by_side[side].copy() for side in ("left", "right")}
    target["left"][0] += delta
    return RegisteredShadowJointTrajectory(
        trajectory_id="trajectory:test",
        created_at=2.0,
        initial_snapshot=initial,
        scene_hash="scene:test",
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


def _binding(identity=None):
    return MujocoShadowModelBinding(
        observed_identity=identity or _identity(),
        scene_hash="scene:test",
        evidence_refs=("model-binding:test",),
        verified=True,
        verification_method="canonical_model_identity_v1",
    )


@pytest.mark.skipif(
    not DEFAULT_INSPIRE_V3_DUAL_ARM_URDF.is_file(),
    reason="local canonical Inspire-v3 URDF asset is unavailable",
)
def test_exact_registered_qpos_executes_in_mujoco_and_finishes_in_hold() -> None:
    trajectory = _trajectory()
    control = build_shadow_initialized_control(trajectory.initial_snapshot)
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
            model_binding=_binding(),
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


@pytest.mark.skipif(
    not DEFAULT_INSPIRE_V3_DUAL_ARM_URDF.is_file(),
    reason="local canonical Inspire-v3 URDF asset is unavailable",
)
def test_application_acceptance_initializes_from_snapshot_and_stops_after_mujoco() -> None:
    snapshot = _snapshot()
    real_factory_calls: list[str] = []

    receipt = run_real_shadow_acceptance(
        snapshot_provider=lambda: snapshot,
        trajectory_builder=_trajectory,
        simulation_control_factory=build_shadow_initialized_control,
        model_binding_provider=lambda _control, trajectory: _binding(
            trajectory.initial_snapshot.identity
        ),
        real_ports_factory=lambda trajectory: real_factory_calls.append(
            trajectory.semantic_content_hash
        ),
    )

    assert receipt.status is RealShadowAcceptanceStatus.SIMULATION_ACCEPTED_NO_MOTION
    assert receipt.simulation_receipt.status is ShadowSimulationStatus.COMPLETED
    assert receipt.real_motion_disposition is RealMotionDisposition.NONE
    assert receipt.terminal_hold_confirmed is True
    assert real_factory_calls == []


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


class _FakeScene:
    def __init__(self, projector):
        self.data = SimpleNamespace(time=0.0)
        self._projector = projector
        self.grippers = {"left_gripper": 0.2, "right_gripper": 0.3}

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
            gripper_actuation_profile=SimpleNamespace(
                side_by_gripper={
                    "left_gripper": "left",
                    "right_gripper": "right",
                }
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
    ):
        self.scene = _FakeScene(_Projector(blocked_call))
        self.rig = _FakeRig(
            self.scene,
            trajectory.initial_snapshot.qpos_by_side,
            frozen=frozen,
            hold_drift=hold_drift,
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


def test_model_identity_mismatch_holds_without_dispatching_points() -> None:
    trajectory = _trajectory()
    control = _FakeControl(trajectory)

    receipt = execute_registered_shadow_trajectory(
        control=control,
        trajectory=trajectory,
        model_binding=_binding(_identity(robot_model_hash="changed")),
        config=MujocoExactQposExecutionConfig(
            terminal_hold_timeout_sec=0.1,
            terminal_hold_dwell_sec=0.01,
        ),
    )

    assert receipt.status is ShadowSimulationStatus.MODEL_IDENTITY_MISMATCH
    assert receipt.executed_point_count == 0
    assert receipt.model_identity_confirmed is False
    assert receipt.terminal_hold_confirmed is True
    assert control.rig.hold_calls == 1


def test_matching_but_unverified_model_identity_cannot_confirm_execution() -> None:
    trajectory = _trajectory()
    control = _FakeControl(trajectory)
    receipt = execute_registered_shadow_trajectory(
        control=control,
        trajectory=trajectory,
        model_binding=MujocoShadowModelBinding(
            observed_identity=_identity(),
            scene_hash="scene:test",
            evidence_refs=("unverified-model-binding:test",),
        ),
        config=MujocoExactQposExecutionConfig(
            terminal_hold_timeout_sec=0.1,
            terminal_hold_dwell_sec=0.01,
        ),
    )

    assert receipt.status is ShadowSimulationStatus.MODEL_IDENTITY_MISMATCH
    assert receipt.model_identity_confirmed is False
    assert receipt.executed_point_count == 0


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
