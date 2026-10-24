"""Public-core clock regressions: pending events can fall inside the last tick."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from sim.rm65.actuation import (
    DmtacPositionActuator,
    Rm65PositionActuator,
    dmtac_bimanual_position_profile,
    rm65_bimanual_udp_feedback_profile,
)

pytestmark = pytest.mark.unit


def _arm():
    profile = replace(
        rm65_bimanual_udp_feedback_profile(),
        name="fractional-delay-test",
        transport_delay_sec_by_side={"left": 0.0005, "right": 0.0005},
        transport_delay_sec_by_side_joint={},
    )
    return Rm65PositionActuator(profile, command_hz=120, physics_hz=1000)


def _grip():
    original = dmtac_bimanual_position_profile()
    profile = replace(
        original,
        name="fractional-delay-test",
        source_artifact="synthetic-test",
        source_payload_sha256="0" * 64,
        parameters_by_side={
            side: {
                key: replace(value, delay_sec=0.0005)
                for key, value in directions.items()
            }
            for side, directions in original.parameters_by_side.items()
        },
    )
    return DmtacPositionActuator(profile)


def test_arm_restores_event_due_after_last_processed_tick_before_current_time():
    source = _arm()
    initial = {side: np.zeros(6) for side in source.active_sides}
    source.reset(command_sequence=0, qpos_by_side=initial)
    while not source.command_due:
        source.step()
    source.step(
        command_sequence=1,
        qpos_by_side={side: np.full(6, 0.01) for side in source.active_sides},
    )
    saved = source.snapshot()
    assert (
        source.physics_time_sec - 0.001
        < saved["delayed_targets"][0]["apply_time_sec"]
        < source.physics_time_sec
    )
    restored = _arm()
    restored.reset(
        command_sequence=saved["accepted_command_sequence"],
        qpos_by_side=saved["accepted_qcmd_by_side"],
        initial_physics_time_sec=source.physics_time_sec,
        initial_state=saved,
    )
    assert restored.snapshot() == saved
    for _ in range(30):
        for owner in (source, restored):
            if owner.command_due:
                owner.step(
                    command_sequence=owner.snapshot()["accepted_command_sequence"] + 1,
                    qpos_by_side=initial,
                )
            else:
                owner.step()
        assert restored.snapshot() == source.snapshot()


@pytest.mark.parametrize("dt", [0.001, 0.0025])
def test_gripper_restores_event_inside_completed_step_without_reapplying_it(dt):
    source = _grip()
    initial = {side: 0.0 for side in source.active_sides}
    source.reset(initial)
    source.queue_targets({side: 0.5 for side in source.active_sides})
    source.step(physics_time_sec=0.0, timestep_sec=dt)
    saved = source.snapshot()
    assert 0.0 < saved["delayed_targets"][0]["apply_time_sec"] < dt
    restored = _grip()
    restored.reset(
        saved["actuator_position_by_side"],
        initial_physics_time_sec=dt,
        initial_state=saved,
    )
    assert restored.snapshot() == saved
    for i in range(1, 20):
        for owner in (source, restored):
            owner.step(physics_time_sec=i * dt, timestep_sec=dt)
        assert restored.snapshot() == source.snapshot()


@pytest.mark.parametrize(
    "damage", ["processed-event", "missing-step", "wrong-end", "negative-dt"]
)
def test_gripper_rejects_processed_or_unproven_past_events(damage):
    source = _grip()
    source.reset({side: 0.0 for side in source.active_sides})
    source.queue_targets({side: 0.5 for side in source.active_sides})
    source.step(physics_time_sec=0.0, timestep_sec=0.0025)
    saved = copy.deepcopy(source.snapshot())
    if damage == "processed-event":
        saved["delayed_targets"][0]["apply_time_sec"] = 0.0
    elif damage == "missing-step":
        saved.pop("last_step")  # Legacy v1 must retain the old conservative bound.
    elif damage == "wrong-end":
        saved["last_step"]["timestep_sec"] = 0.003
    else:
        saved["last_step"]["timestep_sec"] = -0.001
    with pytest.raises(ValueError):
        _grip().reset(
            saved["actuator_position_by_side"],
            initial_physics_time_sec=0.0025,
            initial_state=saved,
        )


def test_arm_still_rejects_event_at_an_already_processed_tick():
    source = _arm()
    source.reset(
        command_sequence=0,
        qpos_by_side={side: np.zeros(6) for side in source.active_sides},
    )
    while not source.command_due:
        source.step()
    source.step(
        command_sequence=1,
        qpos_by_side={side: np.full(6, 0.01) for side in source.active_sides},
    )
    saved = copy.deepcopy(source.snapshot())
    saved["delayed_targets"][0]["apply_time_sec"] = (saved["physics_tick"] - 1) / 1000
    with pytest.raises(ValueError, match="delayed_targets"):
        _arm().reset(
            command_sequence=saved["accepted_command_sequence"],
            qpos_by_side=saved["accepted_qcmd_by_side"],
            initial_physics_time_sec=saved["physics_time_sec"],
            initial_state=saved,
        )
