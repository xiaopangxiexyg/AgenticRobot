from __future__ import annotations

import numpy as np
import pytest

from sim.contracts import TimingSpec
from sim.mujoco.control import (
    DIRECT_QPOS_DEBUG_EXECUTION_MODE,
    DirectQposCommand,
    DirectQposDebugController,
)


class _FakeScene:
    arm_specs = {"left": object(), "right": object()}
    gripper_specs = {"right_gripper": object()}

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, list[float]], dict[str, float], bool]] = []

    def step_kinematic_direct_qpos_debug(
        self,
        qpos_by_side,
        *,
        gripper_close_by_name,
        zero_controlled_qvel,
    ) -> None:
        self.calls.append(
            (
                {side: np.asarray(qpos).tolist() for side, qpos in qpos_by_side.items()},
                dict(gripper_close_by_name),
                bool(zero_controlled_qvel),
            )
        )


class _FakeModel:
    class opt:
        timestep = 1.0 / 500.0


def _command(sequence: int, *, value: float = 0.0) -> DirectQposCommand:
    return DirectQposCommand(
        sequence=sequence,
        qpos_by_side={
            "left": np.full(6, value),
            "right": np.full(6, -value),
        },
        gripper_close_by_name={"right_gripper": 0.25 + value},
        source="test",
    )


def test_direct_qpos_debug_holds_150hz_command_across_1000hz_physics_ticks() -> None:
    scene = _FakeScene()
    controller = DirectQposDebugController(
        scene,
        TimingSpec(command_hz=150.0, physics_hz=1000.0),
    )
    controller.reset(_command(0))

    applied: list[int] = []
    next_sequence = 1
    for _ in range(15):
        command = None
        if controller.command_due:
            command = _command(next_sequence, value=next_sequence * 0.1)
            next_sequence += 1
        tick = controller.step_physics(command)
        if tick.command_applied:
            applied.append(tick.physics_tick)

    assert applied == [7, 14]
    assert len(scene.calls) == 15
    assert scene.calls[0][0]["right"] == pytest.approx([0.0] * 6)
    assert scene.calls[7][0]["right"] == pytest.approx([-0.1] * 6)
    assert scene.calls[13][0]["right"] == pytest.approx([-0.1] * 6)
    assert scene.calls[14][0]["right"] == pytest.approx([-0.2] * 6)
    assert controller.ticks[7].scheduled_command_time_sec == pytest.approx(1.0 / 150.0)
    assert controller.execution_metadata["execution_mode"] == DIRECT_QPOS_DEBUG_EXECUTION_MODE
    assert controller.execution_metadata["robot_dynamics"] == "not_simulated"


def test_direct_qpos_debug_rejects_early_or_non_atomic_commands() -> None:
    controller = DirectQposDebugController(
        _FakeScene(),
        TimingSpec(command_hz=150.0, physics_hz=1000.0),
    )
    controller.reset(_command(0))

    with pytest.raises(RuntimeError, match="before the next command deadline"):
        controller.step_physics(_command(1))

    invalid = DirectQposCommand(
        sequence=1,
        qpos_by_side={"right": np.zeros(6)},
        source="test",
    )
    for _ in range(7):
        controller.step_physics()
    with pytest.raises(ValueError, match="exactly match active_sides"):
        controller.step_physics(invalid)


def test_direct_qpos_command_rejects_invalid_gripper_close() -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        DirectQposCommand(
            sequence=0,
            qpos_by_side={"right": np.zeros(6)},
            gripper_close_by_name={"right_gripper": 1.1},
        )


def test_direct_qpos_debug_rejects_scene_timing_mismatch() -> None:
    scene = _FakeScene()
    scene.model = _FakeModel()

    with pytest.raises(ValueError, match="timestep does not match"):
        DirectQposDebugController(
            scene,
            TimingSpec(command_hz=150.0, physics_hz=1000.0),
        )
