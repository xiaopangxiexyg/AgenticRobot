from __future__ import annotations

import numpy as np
import pytest

from sim.mujoco.deploy.measurement import (
    MUJOCO_BACKEND_MEASUREMENT_VERSION,
    build_mujoco_backend_measurement,
    capture_mujoco_qpos_measurement,
)


class FakeScene:
    def __init__(self, qpos_by_side) -> None:
        self.qpos_by_side = qpos_by_side

    def get_arm_qpos(self, side: str):
        return np.asarray(self.qpos_by_side[side], dtype=float)


class FakeRig:
    def __init__(self, snapshot, scene_qpos) -> None:
        self._snapshot = snapshot
        self.scene = FakeScene(scene_qpos)

    def snapshot(self):
        return self._snapshot


def _qpos(value: float) -> list[float]:
    return [value] * 6


def test_actuated_measurement_keeps_command_target_and_plant_state_distinct() -> None:
    rig = FakeRig(
        {
            "execution_mode": "actuated_position_servo",
            "robot_dynamics": "mujoco_position_actuator",
            "accepted_qcmd_by_side": {"left": _qpos(0.3), "right": _qpos(-0.3)},
            "execution": {
                "physics_tick": 48,
                "actuator_target_by_side": {
                    "left": _qpos(0.2),
                    "right": _qpos(-0.2),
                },
                "measured_qpos_by_side": {
                    "left": _qpos(0.1),
                    "right": _qpos(-0.1),
                },
            },
            "gripper_execution": {
                "accepted_close_by_gripper": {"left_gripper": 0.8},
                "actuator_target_close_by_gripper": {"left_gripper": 0.6},
                "measured_close_by_gripper": {"left_gripper": 0.4},
            },
        },
        # The locked actuated snapshot must win over a later scene sample.
        scene_qpos={"left": _qpos(9.0), "right": _qpos(9.0)},
    )

    measurement = build_mujoco_backend_measurement(
        rig,
        physics_ticks_after_dispatch=12,
    )

    assert measurement["schema_version"] == MUJOCO_BACKEND_MEASUREMENT_VERSION
    assert measurement["accepted_qcmd_by_side"]["left"] == _qpos(0.3)
    assert measurement["actuator_target_qpos_by_side"]["left"] == _qpos(0.2)
    assert measurement["measured_qpos_by_side"]["left"] == _qpos(0.1)
    assert measurement["arm_qpos_by_side"] == measurement["measured_qpos_by_side"]
    assert measurement["physics_ticks_after_dispatch"] == 12
    assert measurement["physics_tick"] == 48
    assert measurement["gripper"] == {
        "accepted_close_by_gripper": {"left_gripper": 0.8},
        "actuator_target_close_by_gripper": {"left_gripper": 0.6},
        "measured_close_by_gripper": {"left_gripper": 0.4},
    }


def test_direct_qpos_debug_measurement_marks_actuator_target_unavailable() -> None:
    rig = FakeRig(
        {
            "execution_mode": "kinematic_direct_qpos_debug",
            "robot_dynamics": "not_simulated",
            "accepted_qcmd_by_side": {"left": _qpos(0.3)},
            "execution": {
                "execution_mode": "kinematic_direct_qpos_debug",
                "robot_dynamics": "not_simulated",
            },
            "gripper_execution": None,
        },
        scene_qpos={"left": _qpos(0.25)},
    )

    measurement = build_mujoco_backend_measurement(
        rig,
        physics_ticks_after_dispatch=1,
    )

    assert measurement["accepted_qcmd_by_side"]["left"] == _qpos(0.3)
    assert measurement["actuator_target_qpos_by_side"] is None
    assert measurement["measured_qpos_by_side"]["left"] == _qpos(0.25)
    assert measurement["robot_dynamics"] == "not_simulated"


def test_measurement_rejects_incomplete_actuator_targets() -> None:
    rig = FakeRig(
        {
            "accepted_qcmd_by_side": {"left": _qpos(0.0), "right": _qpos(0.0)},
            "execution": {
                "actuator_target_by_side": {"left": _qpos(0.0)},
                "measured_qpos_by_side": {
                    "left": _qpos(0.0),
                    "right": _qpos(0.0),
                },
            },
        },
        scene_qpos={"left": _qpos(0.0), "right": _qpos(0.0)},
    )
    with pytest.raises(ValueError, match="same sides"):
        build_mujoco_backend_measurement(rig, physics_ticks_after_dispatch=0)


def test_raw_qpos_measurement_is_source_agnostic_and_prefers_locked_snapshot() -> None:
    rig = FakeRig(
        {
            "execution": {
                "physics_tick": 7,
                "measured_qpos_by_side": {
                    "left": _qpos(0.1),
                    "right": _qpos(-0.1),
                },
            },
        },
        scene_qpos={"left": _qpos(9.0), "right": _qpos(9.0)},
    )

    measurement = capture_mujoco_qpos_measurement(
        rig,
        sides=("left", "right"),
        clock_domain_id="sim-clock",
        measured_at=12.5,
    )

    assert measurement.qpos_by_side["left"] == tuple(_qpos(0.1))
    assert measurement.physics_tick == 7
    assert measurement.measured_at == 12.5
    assert measurement.clock_domain_id == "sim-clock"
    assert measurement.measurement_id.startswith("mujoco-qpos-measurement:")
    with pytest.raises(TypeError):
        measurement.qpos_by_side["left"] = tuple(_qpos(1.0))
