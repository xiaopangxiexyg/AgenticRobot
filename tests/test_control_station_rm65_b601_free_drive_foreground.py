from __future__ import annotations

from types import SimpleNamespace

from control_station.sim_hil_profile.rm65_b601_free_drive import (
    RM65FreeDriveAdmissionError,
    RM65ToB601FreeDriveBridge,
)
from control_station.sim_hil_profile.rm65_b601_free_drive_foreground import (
    run_rm65_to_b601_foreground,
)


class _Bridge(RM65ToB601FreeDriveBridge):
    def __init__(self, *, reject_activation: bool = False) -> None:
        self._state = "inactive"
        self.reject_activation = reject_activation
        self.activate_count = 0
        self.deactivate_count = 0
        self.step_count = 0

    @property
    def state(self) -> str:
        return self._state

    def activate(self) -> None:
        self.activate_count += 1
        if self.reject_activation:
            self._state = "held"
            self.reject_activation = False
            raise RM65FreeDriveAdmissionError("fresh RM65 observation unavailable")
        self._state = "active"

    def engage_clutch(self) -> None:
        self._state = "clutched"

    def release_clutch(self) -> None:
        self._state = "active"

    def deactivate(self) -> None:
        self.deactivate_count += 1
        self._state = "inactive"

    def step(self):
        self.step_count += 1
        return SimpleNamespace(
            state=self._state,
            source_step=None,
            retarget_result=None,
            pending_command=False,
            position_tick=SimpleNamespace(
                command_applied=False,
                rejection_reason=None,
            ),
        )


class _Viewer:
    def __init__(self, *, stop_on_sync: bool = False) -> None:
        self.started = 0
        self.sync_count = 0
        self.closed = 0
        self.stop_on_sync = stop_on_sync
        self.running = True

    def start(self):
        self.started += 1
        return {"running": True}

    def sync(self) -> bool:
        self.sync_count += 1
        if self.stop_on_sync:
            self.running = False
            return False
        return self.running

    def status(self):
        return {"state": "closed_by_user" if not self.running else "running"}

    def close(self) -> None:
        self.closed += 1
        self.running = False


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += max(float(duration), 0.0)


def _commands(values: list[str | None]):
    pending = iter(values)

    def poll(_timeout: float) -> str | None:
        return next(pending, None)

    return poll


def test_foreground_owns_viewer_and_bridge_lifecycle_without_enter() -> None:
    bridge = _Bridge()
    viewer = _Viewer()
    clock = _Clock()
    events: list[dict[str, object]] = []
    messages: list[str] = []

    run_rm65_to_b601_foreground(
        bridge,
        viewer=viewer,
        rate_hz=100.0,
        poll_command=_commands(["a", None, "q"]),
        emit=messages.append,
        record=events.append,
        clock=clock.now,
        sleeper=clock.sleep,
    )

    assert viewer.started == 1
    assert viewer.closed == 1
    assert viewer.sync_count >= 2
    assert bridge.activate_count == 1
    assert bridge.step_count == 1
    assert bridge.deactivate_count >= 2  # startup and final cleanup
    assert bridge.state == "inactive"
    assert any(event["event"] == "operator_command" for event in events)
    assert events[-1]["event"] == "foreground_stopped"
    assert any("Keys (no Enter)" in message for message in messages)


def test_admission_rejection_stays_held_and_a_retries_without_old_target() -> None:
    bridge = _Bridge(reject_activation=True)
    viewer = _Viewer()
    clock = _Clock()
    events: list[dict[str, object]] = []
    messages: list[str] = []

    run_rm65_to_b601_foreground(
        bridge,
        viewer=viewer,
        rate_hz=100.0,
        poll_command=_commands(["a", "a", "q"]),
        emit=messages.append,
        record=events.append,
        clock=clock.now,
        sleeper=clock.sleep,
    )

    assert bridge.activate_count == 2
    assert any("fresh RM65 observation unavailable" in message for message in messages)
    assert any(event["event"] == "admission_rejected" for event in events)
    assert bridge.state == "inactive"


def test_viewer_close_stops_following_without_a_bridge_step() -> None:
    bridge = _Bridge()
    viewer = _Viewer(stop_on_sync=True)
    clock = _Clock()
    events: list[dict[str, object]] = []

    run_rm65_to_b601_foreground(
        bridge,
        viewer=viewer,
        rate_hz=100.0,
        poll_command=_commands([]),
        record=events.append,
        clock=clock.now,
        sleeper=clock.sleep,
    )

    assert bridge.step_count == 0
    assert bridge.state == "inactive"
    assert viewer.closed == 1
    assert any(event["event"] == "viewer_stopped" for event in events)
