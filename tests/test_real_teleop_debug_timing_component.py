from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint
from teleop_runtime.real_teleop import debug_timing


def _args(**updates: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "debug_timing": False,
        "debug_timing_rm65": False,
        "debug_timing_ik": False,
        "debug_timing_auto_log": True,
        "debug_timing_console": False,
        "debug_timing_log": False,
        "debug_ik_timing": False,
        "debug_timing_watchdog": False,
        "debug_timing_threshold_sec": 0.025,
        "debug_timing_report_interval_sec": 1.0,
        "control_debug": "off",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_extracted_debug_timing_metadata_matches_legacy_contract() -> None:
    assert legacy_entrypoint.DebugTimingModules is debug_timing.DebugTimingModules
    assert legacy_entrypoint.RealDebugTimingState is debug_timing.RealDebugTimingState
    assert legacy_entrypoint._record_position_loop_timings is debug_timing.record_position_loop_timings
    legacy_args = _args(debug_timing_rm65=True, debug_timing_ik=True)
    extracted_args = _args(debug_timing_rm65=True, debug_timing_ik=True)

    assert asdict(debug_timing.debug_timing_modules(extracted_args)) == asdict(
        legacy_entrypoint._debug_timing_modules(legacy_args)
    )
    assert debug_timing.debug_timing_metadata(extracted_args) == legacy_entrypoint._debug_timing_metadata(
        legacy_args
    )


def test_extracted_debug_timing_metadata_merge_matches_legacy_contract() -> None:
    legacy_args = _args(control_debug_runtime_metadata={"requested": True})
    extracted_args = _args(control_debug_runtime_metadata={"requested": True})

    expected = legacy_entrypoint._merge_control_debug_runtime_metadata(
        legacy_args,
        attached=True,
    )
    actual = debug_timing.merge_control_debug_runtime_metadata(
        extracted_args,
        attached=True,
    )

    assert actual == expected == {"requested": True, "attached": True}
