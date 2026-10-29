from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_gateway.profiles.mujoco.gripper_skill import MujocoGripperControlPort

pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _Rig:
    def __init__(self, timestep=0.00025):
        self.config = SimpleNamespace(timing=SimpleNamespace(physics_timestep_sec=timestep))
        self.steps = 0

    def step_physics(self):
        self.steps += 1


class _Observer:
    """Release window that becomes stable after a fixed number of rig steps."""

    def __init__(self, rig, *, stable_after_steps):
        self.rig = rig
        self.stable_after_steps = stable_after_steps
        self.reads = 0

    def release_snapshot(self):
        self.reads += 1
        stable = (self.stable_after_steps is not None
                  and self.rig.steps >= self.stable_after_steps)
        return {"status": "stable_supported_release" if stable else "insufficient_window",
                "stable_supported_release": stable}


def _port(rig, observer):
    port = object.__new__(MujocoGripperControlPort)
    port.rig = rig
    port._release_observer = observer
    return port


def test_named_release_steps_the_owned_rig_until_the_window_is_stable():
    rig = _Rig()
    observer = _Observer(rig, stable_after_steps=2400)
    snapshot = _port(rig, observer).observe_named_release(cancel_requested=lambda: False)
    assert snapshot["stable_supported_release"] is True
    # Checked every 10 ms of simulated time (40 steps at 4 kHz): stops at the first check.
    assert rig.steps == 2400


def test_named_release_observation_is_bounded_and_reports_the_last_window():
    rig = _Rig()
    observer = _Observer(rig, stable_after_steps=None)
    snapshot = _port(rig, observer).observe_named_release(
        cancel_requested=lambda: False, maximum_observation_sec=0.5)
    assert snapshot["stable_supported_release"] is False
    assert rig.steps == 2000


def test_named_release_observation_stops_on_cancellation_without_stepping_further():
    rig = _Rig()
    observer = _Observer(rig, stable_after_steps=None)
    calls = []

    def cancelled():
        calls.append(rig.steps)
        return rig.steps >= 100

    _port(rig, observer).observe_named_release(cancel_requested=cancelled)
    assert rig.steps == 100


def test_already_stable_release_does_not_advance_physics():
    rig = _Rig()
    observer = _Observer(rig, stable_after_steps=0)
    _port(rig, observer).observe_named_release(cancel_requested=lambda: False)
    assert rig.steps == 0


def test_named_release_observation_requires_the_bound_observer():
    with pytest.raises(RuntimeError, match="observer"):
        _port(_Rig(), None).observe_named_release(cancel_requested=lambda: False)
