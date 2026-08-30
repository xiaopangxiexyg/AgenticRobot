from __future__ import annotations

import numpy as np
import pytest

from sim.rm65.actuation import (
    Rm65PositionActuator,
    rm65_bimanual_udp_feedback_profile,
)


def test_rm65_actuator_core_is_backend_independent_and_causal() -> None:
    profile = rm65_bimanual_udp_feedback_profile()
    actuator = Rm65PositionActuator(
        profile,
        command_hz=120.0,
        physics_hz=1000.0,
    )
    initial = {side: np.zeros(6) for side in ("left", "right")}
    target = {side: values.copy() for side, values in initial.items()}
    target["left"][5] = 0.02
    actuator.reset(command_sequence=0, qpos_by_side=initial)

    while not actuator.command_due:
        actuator.step()
    accepted_tick = actuator.step(command_sequence=1, qpos_by_side=target)

    assert accepted_tick.command_applied
    assert np.array_equal(actuator.accepted_qpos("left"), target["left"])
    assert actuator.snapshot()["raw_actuator_target_by_side"]["left"][5] == 0.0

    sequence = 2
    saw_delayed_raw_target = False
    saw_filtered_target = False
    while actuator.physics_time_sec < 0.06:
        if actuator.command_due:
            tick = actuator.step(command_sequence=sequence, qpos_by_side=target)
            sequence += 1
        else:
            tick = actuator.step()
        snapshot = actuator.snapshot()
        raw = snapshot["raw_actuator_target_by_side"]["left"][5]
        filtered = snapshot["actuator_target_by_side"]["left"][5]
        if raw == pytest.approx(target["left"][5]):
            saw_delayed_raw_target = True
            if not np.isclose(raw, filtered):
                saw_filtered_target = True

    assert saw_delayed_raw_target
    assert saw_filtered_target
    assert tick.physics_tick == actuator.snapshot()["physics_tick"]
    assert actuator.snapshot()["pending_delayed_targets"] < 36


def test_rm65_actuator_core_supports_one_arm_without_backend_dependencies() -> None:
    actuator = Rm65PositionActuator(
        rm65_bimanual_udp_feedback_profile(),
        command_hz=120.0,
        physics_hz=1000.0,
        active_sides=("right",),
    )
    actuator.reset(command_sequence=7, qpos_by_side={"right": np.zeros(6)})

    assert actuator.active_sides == ("right",)
    assert set(actuator.actuator_target_by_side) == {"right"}
    with pytest.raises(ValueError, match="exactly match active_sides"):
        actuator.reset(
            command_sequence=8,
            qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        )


def test_rm65_actuator_snapshot_bootstrap_preserves_absolute_command_clock() -> None:
    actuator = Rm65PositionActuator(
        rm65_bimanual_udp_feedback_profile(),
        command_hz=120.0,
        physics_hz=1000.0,
    )
    initial = {side: np.zeros(6) for side in ("left", "right")}
    actuator.reset(
        command_sequence=0,
        qpos_by_side=initial,
        initial_physics_time_sec=0.5,
    )

    assert actuator.physics_time_sec == pytest.approx(0.5)
    assert actuator.command_due is False
    while actuator.physics_time_sec < (0.5 + 1.0 / 120.0):
        actuator.step()
    assert actuator.command_due is True

    # The historical cold-start contract remains unchanged: t=0 begins with
    # the first command deadline one period later.
    actuator.reset(command_sequence=0, qpos_by_side=initial)
    assert actuator.physics_time_sec == pytest.approx(0.0)
    assert actuator.command_due is False
