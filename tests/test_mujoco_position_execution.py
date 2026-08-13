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
