from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.runtime import EffectorCommand, TeleopCommand
from sim.contracts import TimingSpec
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.control.desk_tcp_position import DeskTcpPositionControl


class _FakeScene:
    arm_specs = {"left": object(), "right": object()}
    gripper_specs = {}

    def __init__(self) -> None:
        self.qpos = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }
        self.physics_commands: list[dict[str, np.ndarray]] = []

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
        command = {
            side: np.asarray(qpos, dtype=float).reshape(6).copy()
            for side, qpos in qpos_by_side.items()
        }
        self.physics_commands.append(command)
        self.qpos = {side: qpos.copy() for side, qpos in command.items()}


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _timing() -> TimingSpec:
    return TimingSpec(
        sample_hz=15.0,
        policy_hz=15.0,
        reference_hz=60.0,
        ik_hz=60.0,
        command_hz=120.0,
        physics_hz=1000.0,
    )


def _command(left: float, right: float, **metadata) -> TeleopCommand:
    command = TeleopCommand(timestamp=0.0)
    command.add_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.full(6, left, dtype=float),
            metadata=dict(metadata),
        )
    )
    command.add_command(
        EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="joint_vector",
            value=np.full(6, right, dtype=float),
            metadata=dict(metadata),
        )
    )
    return command


def test_mujoco_position_rig_runs_atomic_60_to_120_to_1000_pipeline() -> None:
    scene = _FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
    )
    rig.start()
    receipts = rig.send_commands(_command(0.2, -0.2))
    assert set(receipts) == {"left_arm", "right_arm"}
    assert all(receipt.accepted for receipt in receipts.values())

    for _ in range(12):
        rig.step_physics()

    accepted = rig.accepted_qcmd_by_side()
    assert np.all(accepted["left"] > 0.0)
    assert np.all(accepted["right"] < 0.0)
    assert len(scene.physics_commands) == 12
    assert all(set(command) == {"left", "right"} for command in scene.physics_commands)
    snapshot = rig.snapshot()
    assert snapshot["timing"]["reference_hz"] == 60.0
    assert snapshot["timing"]["command_hz"] == 120.0
    assert snapshot["physics_ticks"] == 12
    assert rig._direct.ticks == ()


def test_mujoco_position_rig_mailbox_keeps_only_latest_qref() -> None:
    scene = _FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
    )
    rig.start()
    rig.send_commands(_command(0.1, -0.1))
    rig.send_commands(_command(0.3, -0.3))
    for _ in range(12):
        rig.step_physics()

    snapshot = rig.snapshot()
    assert snapshot["last_reference_sequence"] == 2
    assert snapshot["pending_reference_sequence"] is None
    assert snapshot["planner_by_side"]["left"]["target_q"] == [0.3] * 6
    assert snapshot["planner_by_side"]["right"]["target_q"] == [-0.3] * 6


def test_offline_immediate_qref_realigns_after_wall_clock_outpaces_physics() -> None:
    scene = _FakeScene()
    clock = _Clock()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    rig.start()
    rig.send_commands(_command(0.1, -0.1))
    rig.step_physics()

    # Model a slow IK/Agent planning interval without advancing MuJoCo.  The
    # second immediate qref must activate on the next explicit physics tick,
    # not remain pending behind the external clock forever.
    clock.value += 1.0
    receipts = rig.send_commands(_command(0.3, -0.3))
    rig.step_physics()

    snapshot = rig.snapshot()
    assert all(
        item.metadata["activate_at"] == pytest.approx(clock.value)
        for item in receipts.values()
    )
    assert snapshot["last_reference_sequence"] == 2
    assert snapshot["pending_reference_sequence"] is None
    assert snapshot["planner_by_side"]["left"]["target_q"] == [0.3] * 6
    assert snapshot["planner_by_side"]["right"]["target_q"] == [-0.3] * 6


def test_worker_deadline_uses_absolute_execution_time_after_snapshot_restore() -> None:
    scene = _FakeScene()
    clock = _Clock(10.0)
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
        ),
        monotonic_clock=clock,
    )

    class _RestoredExecution:
        physics_time_sec = 0.5

    rig._execution = _RestoredExecution()
    rig._clock_origin = 9.5
    calls: list[bool] = []

    def one_tick() -> None:
        calls.append(True)
        rig._stop_event.set()

    rig.step_physics = one_tick
    rig._worker_loop()

    assert len(calls) == 1
    assert rig._worker_deadline_misses == 0


def test_snapshot_restore_primes_planner_at_prior_command_phase() -> None:
    scene = _FakeScene()
    clock = _Clock(10.0)
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
        ),
        monotonic_clock=clock,
        initial_qcmd_by_side={
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        },
        initial_physics_time_sec=0.5,
    )
    rig.start()
    try:
        planner = rig._planners["right"]
        # The command clock is sampled before each physics step.  A snapshot
        # at t=0.500 therefore has its preceding 120 Hz proposal at the
        # physics-aligned t=0.492 tick, not at t=0.499 or t=0.500.
        assert planner._last_commit_monotonic == pytest.approx(9.992)
        assert planner._target_published_monotonic == pytest.approx(
            9.483333333333333
        )
    finally:
        rig.close()


def test_mujoco_position_rig_rejects_partial_bimanual_reference() -> None:
    scene = _FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
        ),
    )
    rig.start()
    partial = TeleopCommand()
    partial.add_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.zeros(6),
        )
    )
    try:
        rig.send_commands(partial)
    except ValueError as exc:
        assert "atomic qref" in str(exc)
    else:
        raise AssertionError("partial bimanual reference must be rejected")


def test_mujoco_position_rig_reports_actual_forbidden_contact_separately() -> None:
    scene = _FakeScene()
    scene.data = object()
    check = SimpleNamespace(
        to_dict=lambda: {
            "safe": False,
            "minimum_distance_m": -0.0015,
            "penetration_depth_m": 0.0015,
            "geom1": "right_gripper_collision",
            "geom2": "table_collision",
            "body1": "right_gripper",
            "body2": "table",
            "pair_class": "1:4",
            "collision_category": "gripper_environment",
        }
    )
    projector = SimpleNamespace(
        oracle=SimpleNamespace(check_contacts=lambda _data: check),
        policy=SimpleNamespace(hard_penetration_m=0.001),
    )
    scene.collision_projector = lambda: projector
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
        ),
    )

    observation = rig.capture_active_contact_observation()

    assert observation["available"] is True
    assert observation["safe"] is False
    assert observation["semantics"] == "mujoco_active_forbidden_contacts"
    assert observation["hard_penetration_limit_m"] == pytest.approx(0.001)
    assert observation["penetration_depth_m"] == pytest.approx(0.0015)


def test_mujoco_position_rig_close_does_not_wait_on_live_worker_lock() -> None:
    scene = _FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            worker_join_timeout_sec=0.01,
        ),
    )
    rig.start()
    release = threading.Event()
    worker = threading.Thread(
        target=release.wait,
        daemon=True,
        name="synthetic-stuck-mujoco-worker",
    )
    worker.start()
    rig._worker = worker

    started_at = time.monotonic()
    with pytest.raises(TimeoutError, match="did not stop"):
        rig.close()
    assert time.monotonic() - started_at < 0.5
    assert rig._worker is worker

    release.set()
    worker.join(timeout=1.0)
    rig.close()
    assert rig._worker is None


def test_mujoco_position_rig_restarts_from_measured_state_without_stale_mailbox() -> None:
    scene = _FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
    )
    rig.start()
    rig.send_commands(_command(0.2, -0.2))
    for _ in range(12):
        rig.step_physics()
    assert rig.snapshot()["physics_ticks"] == 12

    # A close/start is a new execution epoch.  The scene may have been moved
    # by an offline reset or viewer before the restart; neither the old
    # accepted qcmd nor a queued reference may leak across that boundary.
    scene.qpos = {
        "left": np.full(6, 0.31, dtype=float),
        "right": np.full(6, -0.27, dtype=float),
    }
    rig.close()
    rig._last_worker_error = "synthetic previous worker fault"
    rig.start()

    snapshot = rig.snapshot()
    assert snapshot["physics_ticks"] == 0
    assert snapshot["deadline_misses"] == 0
    assert snapshot["last_worker_error"] is None
    assert snapshot["last_reference_sequence"] is None
    assert snapshot["pending_reference_sequence"] is None
    assert snapshot["accepted_qcmd_by_side"]["left"] == [0.31] * 6
    assert snapshot["accepted_qcmd_by_side"]["right"] == [-0.27] * 6
    assert snapshot["measured_qpos_by_side"]["left"] == [0.31] * 6
    assert snapshot["measured_qpos_by_side"]["right"] == [-0.27] * 6

    rig.close()


def test_mujoco_position_rig_rebases_to_measured_state_after_tick_failure() -> None:
    scene = _FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
    )
    rig.start()
    rig.send_commands(_command(0.4, -0.4))

    def fail_step(command=None):
        del command
        raise RuntimeError("synthetic physics failure")

    rig._execution.step_physics = fail_step
    with pytest.raises(RuntimeError, match="synthetic physics failure"):
        rig.step_physics()

    # The failed qcmd was never applied by the scene.  Recovery must use the
    # measured scene state and discard the reference/proposal that led to the
    # failed transaction instead of retrying it implicitly.
    snapshot = rig.snapshot()
    assert snapshot["accepted_qcmd_by_side"]["left"] == [0.0] * 6
    assert snapshot["accepted_qcmd_by_side"]["right"] == [0.0] * 6
    assert snapshot["last_worker_error"] == "RuntimeError: synthetic physics failure"
    assert snapshot["last_reference_sequence"] is None
    assert snapshot["pending_reference_sequence"] is None
    assert snapshot["planner_by_side"]["left"]["target_q"] == [0.0] * 6
    assert snapshot["planner_by_side"]["right"]["target_q"] == [0.0] * 6

    with pytest.raises(RuntimeError, match="synthetic physics failure"):
        rig.step_physics()

    # A fresh lifecycle resets the underlying executor as well as the
    # planner-facing accepted qcmd; merely rebasing the planner is not enough
    # because the actuator transport owns its own accepted sequence/timeline.
    rig.close()
    rig._execution.step_physics = type(rig._execution).step_physics.__get__(
        rig._execution,
        type(rig._execution),
    )
    rig.start()
    assert rig.snapshot()["last_worker_error"] is None
    rig.step_physics()

    rig.close()


def test_desk_tcp_endpoint_observation_separates_ik_and_servo_residuals() -> None:
    class _EndpointRig:
        def __init__(self) -> None:
            self.measured = {
                "left": np.zeros(6, dtype=float),
                "right": np.zeros(6, dtype=float),
            }

        def measured_qpos_by_side(self):
            return {side: values.copy() for side, values in self.measured.items()}

    class _EndpointRetargeter:
        config = SimpleNamespace(
            solve_translation_tolerance_m=0.002,
            solve_orientation_tolerance_deg=0.5,
        )

        @staticmethod
        def tool_tcp_world_targets_for_qpos_pair(*, left_qpos, right_qpos):
            poses = {}
            for side, qpos in (("left", left_qpos), ("right", right_qpos)):
                pose = np.eye(4, dtype=float)
                pose[0, 3] = float(np.asarray(qpos, dtype=float).reshape(6)[0])
                poses[side] = pose
            return poses

    control = DeskTcpPositionControl.__new__(DeskTcpPositionControl)
    control.rig = _EndpointRig()
    control.retargeter = _EndpointRetargeter()
    control.scene = SimpleNamespace()
    control._last_tool_world_target_by_side = {
        side: np.eye(4, dtype=float) for side in ("left", "right")
    }
    control._last_active_sides = ("left", "right")
    control._last_ik_seed_source = "accepted_qcmd"
    control._last_ik_seed_by_side = {
        side: np.zeros(6, dtype=float) for side in ("left", "right")
    }
    control._last_ik_measured_qpos_by_side = {
        side: np.zeros(6, dtype=float) for side in ("left", "right")
    }
    control._last_ik_target_error_by_side = {
        side: {"position_error_m": 0.0, "orientation_error_deg": 0.0}
        for side in ("left", "right")
    }

    settled = control.endpoint_tracking_observation()
    assert settled["status"] == "settled"
    assert settled["measured_source"] == "mujoco_data_qpos"

    # The IK target was reached, but the plant has not yet caught up.  This is
    # a transport/servo residual, not an IK failure.
    control.rig.measured["left"][0] = 0.01
    tracking = control.endpoint_tracking_observation()
    assert tracking["status"] == "tracking"
    assert tracking["by_side"]["left"]["settled"] is False
    assert tracking["by_side"]["left"]["ik_target_settled"] is True

    # If the accepted qref itself missed the requested pose, report that
    # separately so a caller does not wait forever for servo convergence.
    control._last_ik_target_error_by_side["left"] = {
        "position_error_m": 0.01,
        "orientation_error_deg": 0.0,
    }
    unmet = control.endpoint_tracking_observation()
    assert unmet["status"] == "ik_target_unmet"

    # A held side has no active Cartesian solve target.  Its stale/requested
    # pose residual must not turn a valid single-arm replay into an IK failure;
    # physical drift from the effective hold pose is still reported normally.
    control._last_active_sides = ("right",)
    control.rig.measured["left"][0] = 0.0
    held = control.endpoint_tracking_observation()
    assert held["status"] == "settled"
    assert held["active_sides"] == ["right"]
    assert held["by_side"]["left"]["active"] is False
    assert held["by_side"]["left"]["held"] is True
    assert held["by_side"]["left"]["ik_target_settled"] is True
