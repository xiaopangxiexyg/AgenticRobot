from __future__ import annotations

from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint
from teleop_runtime.real_teleop.main_loop_watchdog import MainLoopStallWatchdog


def test_disabled_watchdog_does_not_start_a_worker() -> None:
    assert legacy_entrypoint.MainLoopStallWatchdog is MainLoopStallWatchdog
    watchdog = MainLoopStallWatchdog(enabled=False)

    watchdog.start()
    watchdog.beat("unit", 1)
    watchdog.stop()

    assert watchdog._thread is None


def test_enabled_watchdog_starts_beats_and_stops_cleanly() -> None:
    watchdog = MainLoopStallWatchdog(enabled=True, timeout_sec=10.0)

    watchdog.start()
    watchdog.beat("unit", 7)
    watchdog.stop()

    assert watchdog._thread is None
