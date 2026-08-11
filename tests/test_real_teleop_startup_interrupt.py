from __future__ import annotations

from teleop_runtime.real_teleop import runtime
from teleop_runtime.real_teleop import startup_runtime


def test_runtime_startup_cleanup_exports_are_owner_compatibility_aliases() -> None:
    assert runtime._StartupCleanupRegistry is startup_runtime.StartupCleanupRegistry
    assert runtime._activate_startup_cleanup is startup_runtime.activate_startup_cleanup
    assert runtime.cleanup_active_startup is startup_runtime.cleanup_active_startup


def test_startup_cleanup_registry_is_lifo_and_idempotent() -> None:
    calls: list[str] = []
    registry = runtime._activate_startup_cleanup()
    registry.add("first", lambda: calls.append("first"))
    registry.add("second", lambda: calls.append("second"))

    runtime.cleanup_active_startup()
    runtime.cleanup_active_startup()

    assert calls == ["second", "first"]


def test_startup_cleanup_registry_disarm_transfers_ownership() -> None:
    calls: list[str] = []
    registry = runtime._activate_startup_cleanup()
    registry.add("owned-by-normal-shutdown", lambda: calls.append("unexpected"))

    registry.disarm()
    runtime.cleanup_active_startup()

    assert calls == []


def test_startup_cleanup_registry_reports_failures_without_masking_cleanup() -> None:
    calls: list[str] = []
    logs: list[tuple[str, str]] = []
    registry = startup_runtime.activate_startup_cleanup(
        logger=lambda tag, message: logs.append((tag, message))
    )
    registry.add("after-failure", lambda: calls.append("after-failure"))

    def fail() -> None:
        calls.append("failure")
        raise RuntimeError("expected cleanup failure")

    registry.add("failure", fail)
    registry.add("before-failure", lambda: calls.append("before-failure"))

    startup_runtime.cleanup_active_startup()

    assert calls == ["before-failure", "failure", "after-failure"]
    assert logs == [
        (
            "Shutdown",
            "startup Ctrl+C cleanup completed with recoverable failures: "
            "failure: RuntimeError: expected cleanup failure",
        )
    ]
