from __future__ import annotations

import inspect
from types import SimpleNamespace

from teleop_runtime.deploy import run_policy
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


def test_run_policy_delegates_shutdown_to_resource_lifecycle() -> None:
    source = inspect.getsource(run_policy.main)
    assert source.count("resource_lifecycle.close(") == 1
    assert "position_workers_stopped" not in source
