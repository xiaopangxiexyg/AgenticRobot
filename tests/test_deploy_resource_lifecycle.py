from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from teleop_runtime.deploy import run_policy
from teleop_runtime.deploy import resource_lifecycle as lifecycle_module
from teleop_runtime.deploy.resource_lifecycle import PolicyDeployResourceLifecycle


class _DebugLogger:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.logged: list[tuple[str, object]] = []

    def log_event(self, event: str, payload: object) -> None:
        self.logged.append((event, payload))

    def close(self) -> None:
        self.events.append("debug_logger")


class _Closeable:
    def __init__(
        self,
        label: str,
        events: list[str],
        *,
        close_error: Exception | None = None,
    ) -> None:
        self.label = label
        self.events = events
        self.close_error = close_error

    def close(self) -> None:
        self.events.append(self.label)
        if self.close_error is not None:
            raise self.close_error


class _RtcFrontend(_Closeable):
    def close(self, *, now: float) -> SimpleNamespace:
        assert now > 0.0
        super().close()
        return SimpleNamespace(
            semantic_content_hash="stop-hash",
            session_state="closed",
            thread_exited=True,
            execution_drain_evidence={"drained": True},
            status="closed",
        )


class _RtcDebugObserver(_Closeable):
    enabled = True

    def offer(self, event: str, payload: object) -> None:
        assert event == "rtc_frontend_close"
        assert isinstance(payload, dict)

    def close(self) -> SimpleNamespace:
        super().close()
        return SimpleNamespace(to_payload=lambda: {"closed": True})


class _PositionExecutor(_Closeable):
    def __init__(
        self,
        events: list[str],
        *,
        close_error: Exception | None = None,
        worker_alive: bool = False,
    ) -> None:
        super().__init__("position_executor", events, close_error=close_error)
        self.worker_alive = worker_alive

    def snapshot(self) -> dict[str, bool]:
        return {
            "reference_worker_alive": self.worker_alive,
            "discrete_worker_alive": False,
        }


class _Guard(_Closeable):
    def snapshot(self) -> dict[str, str]:
        return {"status": "closed"}


class _FixedInit:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def restore_speed(self, arm_backends: object) -> None:
        assert arm_backends
        self.events.append("fixed_init_restore")


class _ArmBackend(_Closeable):
    def __init__(
        self,
        events: list[str],
        *,
        ready: bool = True,
    ) -> None:
        super().__init__("left_arm_close", events)
        self.ready = ready

    def start(self) -> None:
        self.events.append("arm_start")

    def is_ready(self) -> bool:
        return self.ready

    def get_state(self) -> dict[str, bool]:
        return {"ready": self.ready}


class _StartingPositionExecutor(_PositionExecutor):
    def __init__(
        self,
        events: list[str],
        *,
        affinity: object = None,
    ) -> None:
        super().__init__(events)
        self.affinity = affinity

    def prepare_collision_planner(self) -> None:
        self.events.append("position_prepare")

    def start(self) -> None:
        self.events.append("position_start")

    def snapshot(self) -> dict[str, object]:
        return {
            "reference_worker_affinity_receipt": self.affinity,
            "reference_worker_alive": False,
            "discrete_worker_alive": False,
        }


class _StartingGuard(_Guard):
    def start(self, *, arm_stale_watchdog: bool) -> dict[str, object]:
        assert arm_stale_watchdog is False
        self.events.append("guard_start_without_stale_watchdog")
        return {"config": {}, "last_reading_by_side": {}}

    def arm_stale_watchdog(self) -> dict[str, object]:
        self.events.append("guard_arm_stale_watchdog")
        return {"config": {}}


class _Startable(_Closeable):
    def start(self) -> None:
        self.events.append(f"{self.label}_start")


class _Preview(_Closeable):
    disabled_reason = None

    def ensure_available(self) -> bool:
        self.events.append("camera_preview_preflight")
        return True


def _startup_args(**overrides: object) -> SimpleNamespace:
    values = {
        "allow_arm_connect_failure": False,
        "position_reference_worker_cpu": None,
        "position_continuous_waypoint_worker_cpu": None,
        "position_continuous_waypoint_worker_isolation": "thread",
        "position_worker_cpu_reservation": False,
        "position_worker_cpu_reservation_cpu_ids": (),
        "position_continuous_waypoint_no_debug_deadline_soak": False,
        "max_steps": 1,
        "allow_hand_connect_failure": False,
        "debug": False,
        "camera_mode": "multi",
        "vis_cam": True,
        "camera_start_stagger_sec": 0.5,
        "rtc_profile": "off",
        "rtc_real_shadow": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _startup_lifecycle(
    events: list[str],
    *,
    arm_backend: _ArmBackend | None = None,
    position_executor: _StartingPositionExecutor | None = None,
) -> PolicyDeployResourceLifecycle:
    logger = _DebugLogger(events)
    return PolicyDeployResourceLifecycle(
        debug_logger=logger,
        rtc_frontend=_Startable("rtc", events),
        rtc_debug_observer=None,
        position_deploy_executor=(
            position_executor or _StartingPositionExecutor(events)
        ),
        position_force_guard_monitor=_StartingGuard("guard_close", events),
        impedance_force_guard_monitor=None,
        fixed_init_controller=None,
        hand_state_poller=_Startable("hand_poller", events),
        arm_backends={"left": arm_backend or _ArmBackend(events)},
        hand_backends={"left": _Closeable("left_hand_close", events)},
        camera_sources={"front": _Closeable("camera_close", events)},
        camera_previewer=_Preview("camera_preview_close", events),
        tactile_visualizer=None,
        submit_legacy_owner_state=lambda **_kwargs: None,
        flush_position_executor_debug_events=lambda: None,
    )


def _lifecycle(
    events: list[str],
    *,
    position_executor: _PositionExecutor | None = None,
    arm_backend: _Closeable | None = None,
    hand_backend: _Closeable | None = None,
) -> tuple[PolicyDeployResourceLifecycle, _DebugLogger]:
    logger = _DebugLogger(events)
    lifecycle = PolicyDeployResourceLifecycle(
        debug_logger=logger,
        rtc_frontend=_RtcFrontend("rtc_frontend", events),
        rtc_debug_observer=_RtcDebugObserver("rtc_debug_observer", events),
        position_deploy_executor=position_executor or _PositionExecutor(events),
        position_force_guard_monitor=_Guard("position_force_guard", events),
        impedance_force_guard_monitor=_Closeable("impedance_force_guard", events),
        fixed_init_controller=_FixedInit(events),
        hand_state_poller=_Closeable("hand_state_poller", events),
        arm_backends={"left": arm_backend or _Closeable("left_arm", events)},
        hand_backends={"left": hand_backend or _Closeable("left_hand", events)},
        camera_sources={"front": _Closeable("camera_front", events)},
        camera_previewer=_Closeable("camera_previewer", events),
        tactile_visualizer=_Closeable("tactile_visualizer", events),
        submit_legacy_owner_state=lambda **_kwargs: events.append("legacy_owner"),
        flush_position_executor_debug_events=lambda: events.append(
            "position_debug_flush"
        ),
    )
    return lifecycle, logger


def test_resource_lifecycle_preserves_startup_safety_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    lifecycle = _startup_lifecycle(events)

    def start_hand_backends_checked(
        hand_backends: object,
        **kwargs: object,
    ) -> None:
        assert hand_backends
        assert kwargs == {
            "disconnected_gripper_sides": (),
            "allow_connect_failure": False,
            "debug": False,
        }
        events.append("hand_start_checked")

    def start_camera_sources(
        camera_sources: object,
        *,
        stagger_sec: float,
    ) -> None:
        assert camera_sources
        assert stagger_sec == 0.5
        events.append("camera_start")

    def run_camera_startup_self_check(camera_sources: object) -> None:
        assert camera_sources
        events.append("camera_self_check")

    monkeypatch.setattr(
        lifecycle_module,
        "start_hand_backends_checked",
        start_hand_backends_checked,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "start_camera_sources",
        start_camera_sources,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "run_camera_startup_self_check",
        run_camera_startup_self_check,
    )

    lifecycle.start(
        args=_startup_args(),
        disconnected_gripper_sides=(),
        impedance_force_control_is_deferred=True,
        warm_start_impedance_force_control_backends=lambda arm_backends: (
            events.append("impedance_warm_start") or ("left",)
        ),
    )

    assert events == [
        "arm_start",
        "position_prepare",
        "position_start",
        "guard_start_without_stale_watchdog",
        "impedance_warm_start",
        "hand_start_checked",
        "hand_poller_start",
        "camera_preview_preflight",
        "camera_start",
        "camera_self_check",
        "guard_arm_stale_watchdog",
        "rtc_start",
    ]


def test_arm_not_ready_fails_before_downstream_startup() -> None:
    events: list[str] = []
    lifecycle = _startup_lifecycle(
        events,
        arm_backend=_ArmBackend(events, ready=False),
    )

    with pytest.raises(RuntimeError, match="left arm backend is not ready"):
        lifecycle.start(
            args=_startup_args(),
            disconnected_gripper_sides=(),
            impedance_force_control_is_deferred=False,
            warm_start_impedance_force_control_backends=lambda _backends: (),
        )

    assert events == ["arm_start"]


def test_missing_affinity_receipt_fails_closed_before_guard_start() -> None:
    events: list[str] = []
    lifecycle = _startup_lifecycle(
        events,
        position_executor=_StartingPositionExecutor(events, affinity=None),
    )

    with pytest.raises(RuntimeError, match="affinity receipt is missing"):
        lifecycle.start(
            args=_startup_args(position_reference_worker_cpu=6),
            disconnected_gripper_sides=(),
            impedance_force_control_is_deferred=False,
            warm_start_impedance_force_control_backends=lambda _backends: (),
        )

    assert events == ["arm_start", "position_prepare", "position_start"]


def test_resource_lifecycle_preserves_ordered_shutdown() -> None:
    events: list[str] = []
    lifecycle, _logger = _lifecycle(events)

    errors = lifecycle.close(keyboard_interrupt_stop_error=None)

    assert errors == ()
    assert events == [
        "legacy_owner",
        "rtc_frontend",
        "rtc_debug_observer",
        "position_executor",
        "position_debug_flush",
        "position_force_guard",
        "impedance_force_guard",
        "fixed_init_restore",
        "hand_state_poller",
        "left_arm",
        "left_hand",
        "camera_front",
        "camera_previewer",
        "tactile_visualizer",
        "debug_logger",
    ]


def test_live_position_worker_blocks_backend_close_but_not_other_cleanup() -> None:
    events: list[str] = []
    position_executor = _PositionExecutor(
        events,
        close_error=RuntimeError("worker did not stop"),
        worker_alive=True,
    )
    lifecycle, logger = _lifecycle(events, position_executor=position_executor)

    errors = lifecycle.close(keyboard_interrupt_stop_error="arm stop incomplete")

    assert "left_arm" not in events
    assert "left_hand" not in events
    assert "camera_front" in events
    assert events[-1] == "debug_logger"
    assert errors == (
        "keyboard_interrupt_emergency_stop: arm stop incomplete",
        "position_executor: RuntimeError: worker did not stop",
        "position_backends: close skipped because a dispatch worker is still alive",
    )
    assert any(
        event == "position_executor_close_error"
        and payload["backend_close_skipped"] is True
        for event, payload in logger.logged
    )


def test_independent_close_failure_is_recorded_and_cleanup_continues() -> None:
    events: list[str] = []
    failing_arm = _Closeable(
        "left_arm",
        events,
        close_error=RuntimeError("arm close failed"),
    )
    lifecycle, logger = _lifecycle(events, arm_backend=failing_arm)

    errors = lifecycle.close(keyboard_interrupt_stop_error=None)

    assert errors == ("left_arm_backend: RuntimeError: arm close failed",)
    assert "left_hand" in events
    assert "camera_front" in events
    assert events[-1] == "debug_logger"
    assert (
        "cleanup_error",
        {
            "resource": "left_arm_backend",
            "error_type": "RuntimeError",
            "error": "arm close failed",
        },
    ) in logger.logged


def test_run_policy_delegates_resource_startup_and_shutdown() -> None:
    source = inspect.getsource(run_policy.main)
    assert source.count("resource_lifecycle.start(") == 1
    assert source.count("resource_lifecycle.close(") == 1
    assert "position_deploy_executor.prepare_collision_planner()" not in source
    assert "run_camera_startup_self_check(camera_sources)" not in source
    assert "position_workers_stopped" not in source
