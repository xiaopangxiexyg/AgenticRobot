from __future__ import annotations

from teleop_runtime.deploy.run_policy import (
    enable_impedance_force_control_backends,
    warm_start_impedance_force_control_backends,
)


class _FakeImpedanceBackend:
    def __init__(self) -> None:
        self.events: list[object] = []

    def warm_start_force_control(self) -> None:
        self.events.append("warm_start")

    def enable_force_control(self) -> None:
        self.events.append("enable")

    def wait_until_force_control_ready(self, *, timeout_sec: float) -> None:
        self.events.append(("ready", timeout_sec))


def test_warm_start_does_not_start_impedance_periodic_output() -> None:
    left = _FakeImpedanceBackend()
    right = _FakeImpedanceBackend()

    warmed = warm_start_impedance_force_control_backends({"left": left, "right": right})

    assert warmed == ("left", "right")
    assert left.events == ["warm_start"]
    assert right.events == ["warm_start"]

    enable_impedance_force_control_backends({"left": left, "right": right})

    assert left.events == ["warm_start", "enable", ("ready", 1.0)]
    assert right.events == ["warm_start", "enable", ("ready", 1.0)]
