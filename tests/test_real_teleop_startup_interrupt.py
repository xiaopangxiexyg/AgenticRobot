from __future__ import annotations

from teleop_runtime.real_teleop import runtime


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
