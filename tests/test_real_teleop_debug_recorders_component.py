from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from teleop_runtime.real_teleop.debug_recorders import (
    ActionDebugLogRecorder,
    DebugTimingLogRecorder,
    IkDebugLogRecorder,
)
from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint


def test_legacy_recorder_wrappers_delegate_to_extracted_implementations() -> None:
    assert legacy_entrypoint.DebugTimingLogRecorder.__bases__ == (DebugTimingLogRecorder,)
    assert legacy_entrypoint.ActionDebugLogRecorder.__bases__ == (ActionDebugLogRecorder,)
    assert legacy_entrypoint.IkDebugLogRecorder.__bases__ == (IkDebugLogRecorder,)


@pytest.mark.parametrize(
    ("recorder_type", "schema"),
    (
        (DebugTimingLogRecorder, "dexteleop_debug_timing_v1"),
        (ActionDebugLogRecorder, "dexteleop_action_debug_v1"),
        (IkDebugLogRecorder, "dexteleop_ik_debug_v1"),
    ),
)
def test_extracted_debug_recorders_write_and_stop_cleanly(
    tmp_path,
    recorder_type,
    schema: str,
) -> None:
    recorder = recorder_type(root_dir=tmp_path)
    path = recorder.start(now=1_700_000_000.125)
    recorder.record_frame(
        {
            "schema": schema,
            "event": "frame",
            "payload": {"numpy": np.asarray([1.0, 2.0])},
        }
    )
    assert recorder.stop(reason="test", now=1_700_000_000.5) == path
    assert recorder.active is False

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["start", "frame", "stop"]
    assert rows[1]["payload"]["numpy"] == [1.0, 2.0]
    assert rows[-1]["payload"]["reason"] == "test"


def test_extracted_timing_recorder_selects_state_change_and_heartbeat(tmp_path) -> None:
    recorder = DebugTimingLogRecorder(root_dir=tmp_path, heartbeat_sec=1.0)
    recorder.start(now=1.0)
    try:
        state = SimpleNamespace(value="RUNNING")
        assert (
            recorder.select_frame_reason(
                now=1.0,
                elapsed_sec=0.0,
                phases=(),
                state=state,
                fault_codes=(),
                threshold_sec=1.0,
            )
            == "state_change"
        )
        assert (
            recorder.select_frame_reason(
                now=1.5,
                elapsed_sec=0.0,
                phases=(),
                state=state,
                fault_codes=(),
                threshold_sec=1.0,
            )
            is None
        )
        assert (
            recorder.select_frame_reason(
                now=2.0,
                elapsed_sec=0.0,
                phases=(),
                state=state,
                fault_codes=(),
                threshold_sec=1.0,
            )
            == "heartbeat"
        )
    finally:
        recorder.stop(reason="test", now=2.0)


def test_action_debug_sampling_gates_before_payload_construction(tmp_path) -> None:
    recorder = ActionDebugLogRecorder(root_dir=tmp_path, sample_hz=10.0)
    recorder.start(now=1.0)
    try:
        assert recorder.should_record_frame(now=1.0) is True
        assert recorder.should_record_frame(now=1.05) is False
        assert recorder.should_record_frame(now=1.10) is True
    finally:
        recorder.stop(reason="test", now=1.2)


def test_timing_sampling_preserves_state_and_fault_transitions(tmp_path) -> None:
    recorder = DebugTimingLogRecorder(
        root_dir=tmp_path,
        heartbeat_sec=1.0,
        sample_hz=5.0,
    )
    recorder.start(now=1.0)
    try:
        state = SimpleNamespace(value="RUNNING")
        assert recorder.select_frame_reason(
            now=1.0,
            elapsed_sec=0.1,
            phases=(),
            state=state,
            fault_codes=(),
            threshold_sec=0.04,
        ) == "slow+state_change"
        assert recorder.select_frame_reason(
            now=1.05,
            elapsed_sec=0.1,
            phases=(),
            state=state,
            fault_codes=(),
            threshold_sec=0.04,
        ) is None
        assert recorder.select_frame_reason(
            now=1.21,
            elapsed_sec=0.1,
            phases=(),
            state=state,
            fault_codes=(),
            threshold_sec=0.04,
        ) == "slow"
        assert recorder.select_frame_reason(
            now=1.22,
            elapsed_sec=0.1,
            phases=(),
            state=state,
            fault_codes=("fault",),
            threshold_sec=0.04,
        ) == "slow+fault_change"
    finally:
        recorder.stop(reason="test", now=1.3)
