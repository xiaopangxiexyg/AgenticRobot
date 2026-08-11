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
from teleop_runtime.real_teleop import action_debug_payload, runtime_debug
from teleop_runtime.real_teleop.diagnostic_setup import (
    initialize_control_debug_runtime_metadata,
    setup_actuator_command_trace,
    setup_control_debug_runtime,
)
from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint


@pytest.mark.parametrize(
    ("enabled", "episode_sidecar", "no_collection", "expected_episode"),
    (
        (False, True, False, False),
        (True, False, False, False),
        (True, True, True, False),
        (True, True, False, True),
    ),
)
def test_control_debug_metadata_baseline_owns_sidecar_admission(
    enabled: bool,
    episode_sidecar: bool,
    no_collection: bool,
    expected_episode: bool,
) -> None:
    args = SimpleNamespace(
        control_debug_config=SimpleNamespace(enabled=enabled),
        control_debug_episode_sidecar=episode_sidecar,
        no_collection=no_collection,
    )
    calls: list[tuple[object, dict[str, object]]] = []

    result = initialize_control_debug_runtime_metadata(
        args,
        merge_runtime_metadata=lambda value, **updates: (
            calls.append((value, updates)) or updates
        ),
    )

    assert calls[0][0] is args
    assert result == calls[0][1]
    assert result["attached"] is False
    assert result["session_sidecar_requested"] is enabled
    assert result["session_sidecar_attached"] is False
    assert result["episode_sidecar_requested"] is expected_episode
    assert result["episode_sidecar_attached"] is False
    assert result["episode_sidecars"] is False


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


def test_action_debug_backend_snapshot_reuses_preloaded_state_and_bounds_fallback_reads() -> None:
    assert (
        action_debug_payload._action_debug_backend_snapshot_from_state
        is runtime_debug._action_debug_backend_snapshot_from_state
    )
    assert (
        action_debug_payload._preloaded_action_debug_backend_snapshot
        is runtime_debug._preloaded_action_debug_backend_snapshot
    )

    class CountingBackend:
        def __init__(self) -> None:
            self.reads = 0

        def get_state(self) -> dict[str, object]:
            self.reads += 1
            return {
                "last_command_timestamp": 1.0,
                "current_tcp_pose": None,
                "last_target_tcp_pose": None,
            }

    backend = CountingBackend()
    preloaded_state = backend.get_state()
    backend.reads = 0

    preloaded = runtime_debug._preloaded_action_debug_backend_snapshot(
        backend,
        preloaded_state,
        side="left",
        role="command_backend",
        sample_timestamp=2.0,
        record_wall_time=2.1,
    )
    assert backend.reads == 0
    assert preloaded["available"] is True
    assert preloaded["read_elapsed_s"] == 0.0

    fallback = runtime_debug._preloaded_action_debug_backend_snapshot(
        backend,
        None,
        side="left",
        role="command_backend",
        sample_timestamp=2.0,
        record_wall_time=2.1,
    )
    assert backend.reads == 1
    assert fallback["available"] is True
    assert fallback["last_command_timestamp"] == 1.0


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


class _FakeActuatorTraceRecorder:
    def __init__(self) -> None:
        self.config = SimpleNamespace(capacity_per_side=64)
        self.closed = False

    def record_command(self, **_kwargs: object) -> None:
        return None

    def record_singularity_transition(self, **_kwargs: object) -> None:
        return None

    def record_ik_shadow(self, **_kwargs: object) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_actuator_trace_setup_attaches_all_observers_as_one_transaction(
    tmp_path,
) -> None:
    recorder = _FakeActuatorTraceRecorder()
    cleanup_steps = []
    attached = []
    logs = []
    components = SimpleNamespace(active_sides=("left", "right"))

    def attach(kind: str):
        def callback(_components, sink):
            attached.append((kind, sink))
            return 1 if kind == "ik" else 2

        return callback

    result = setup_actuator_command_trace(
        SimpleNamespace(
            actuator_trace=True,
            actuator_trace_root=tmp_path,
            actuator_singularity_trace=True,
        ),
        components=components,
        startup_cleanup=SimpleNamespace(
            add=lambda label, callback: cleanup_steps.append((label, callback))
        ),
        timestamp="20260808_010203_000004",
        expand_cli_path=lambda value: value,
        new_recorder=lambda *_args, **_kwargs: recorder,
        attach_command_sink=attach("command"),
        attach_singularity_sink=attach("singularity"),
        attach_ik_shadow_sink=attach("ik"),
        log=lambda tag, message: logs.append((tag, message)),
    )

    assert result.recorder is recorder
    assert result.session_dir == (
        tmp_path / "20260808_010203_000004_vive_rm65_actuator_trace"
    )
    assert cleanup_steps == [("actuator_trace_recorder", recorder.close)]
    assert [kind for kind, sink in attached if sink is not None] == [
        "command",
        "singularity",
        "ik",
    ]
    assert logs[0][0] == "ActuatorTrace"
    assert "capacity_per_side=64" in logs[0][1]
    assert "hot_path_disk_write=false" in logs[0][1]


def test_actuator_trace_setup_rolls_back_every_sink_on_partial_failure(
    tmp_path,
) -> None:
    recorder = _FakeActuatorTraceRecorder()
    detach_calls = []
    logs = []
    components = SimpleNamespace(active_sides=("left", "right"))

    def attach_command(_components, sink):
        if sink is None:
            detach_calls.append("command")
            return 0
        return 1

    def detacher(kind: str):
        def callback(_components, sink):
            assert sink is None
            detach_calls.append(kind)
            return 0

        return callback

    result = setup_actuator_command_trace(
        SimpleNamespace(
            actuator_trace=True,
            actuator_trace_root=tmp_path,
            actuator_singularity_trace=True,
        ),
        components=components,
        startup_cleanup=SimpleNamespace(add=lambda *_args: None),
        timestamp="20260808_010203_000004",
        expand_cli_path=lambda value: value,
        new_recorder=lambda *_args, **_kwargs: recorder,
        attach_command_sink=attach_command,
        attach_singularity_sink=detacher("singularity"),
        attach_ik_shadow_sink=detacher("ik"),
        log=lambda tag, message: logs.append((tag, message)),
    )

    assert result.recorder is None
    assert recorder.closed is True
    assert detach_calls == ["command", "singularity", "ik"]
    assert logs[0][0] == "ActuatorTrace"
    assert "did not attach to every active command backend" in logs[0][1]


def test_actuator_trace_setup_disabled_has_no_side_effects(tmp_path) -> None:
    result = setup_actuator_command_trace(
        SimpleNamespace(actuator_trace=False),
        components=SimpleNamespace(active_sides=("left", "right")),
        startup_cleanup=SimpleNamespace(
            add=lambda *_args: pytest.fail("cleanup registration should be skipped")
        ),
        timestamp="",
        expand_cli_path=lambda _value: pytest.fail("path resolution should be skipped"),
        new_recorder=lambda *_args, **_kwargs: pytest.fail("recorder should be skipped"),
        attach_command_sink=lambda *_args: pytest.fail("attach should be skipped"),
        attach_singularity_sink=lambda *_args: pytest.fail("attach should be skipped"),
        attach_ik_shadow_sink=lambda *_args: pytest.fail("attach should be skipped"),
        log=lambda *_args: pytest.fail("logging should be skipped"),
    )

    assert result.recorder is None
    assert result.session_dir is None


class _FakePositionDebugRecorder:
    def __init__(self) -> None:
        self.close_timeouts: list[float] = []

    def close(self, *, timeout_sec: float) -> None:
        self.close_timeouts.append(timeout_sec)


def test_control_debug_setup_attaches_observer_and_metadata_transaction(
    tmp_path,
) -> None:
    recorder = _FakePositionDebugRecorder()
    cleanup_steps = []
    attached = []
    metadata = []
    logs = []
    fanout = object()

    result = setup_control_debug_runtime(
        SimpleNamespace(
            control_debug_config=SimpleNamespace(
                enabled=True,
                root=tmp_path,
                level="summary",
            )
        ),
        components=object(),
        startup_cleanup=SimpleNamespace(
            add=lambda label, callback: cleanup_steps.append((label, callback))
        ),
        timestamp="20260808_010203_000004",
        expand_cli_path=lambda value: value,
        new_recorder=lambda *_args, **_kwargs: recorder,
        audit_fanout_type=lambda recorders: (
            fanout if tuple(recorders) == (recorder,) else pytest.fail("bad fanout")
        ),
        attach_position_sink=lambda _components, sink: attached.append(sink),
        update_runtime_metadata=lambda **updates: metadata.append(updates) or updates,
        wall_time=lambda: 10.0,
        monotonic_time=lambda: 20.0,
        log=lambda tag, message: logs.append((tag, message)),
    )

    assert result.audit_fanout is fanout
    assert result.recorder is recorder
    assert result.session_dir == (
        tmp_path / "20260808_010203_000004_vive_rm65_real_teleop"
    )
    assert attached == [fanout]
    assert cleanup_steps[0][0] == "position_debug_session_recorder"
    assert metadata == [
        {
            "attached": True,
            "session_output_dir": str(result.session_dir),
            "session_sidecar_attached": True,
            "session_sidecar_error": None,
            "session_sidecar_output_dir": str(result.session_dir),
            "session_sidecar_attached_wall_time": 10.0,
            "session_sidecar_attached_monotonic_time": 20.0,
        }
    ]
    assert logs[0][0] == "ControlDebug"


def test_control_debug_setup_failure_closes_recorder_and_detaches_sink(
    tmp_path,
) -> None:
    recorder = _FakePositionDebugRecorder()
    attached = []
    metadata = []
    logs = []

    def attach(_components, sink) -> None:
        attached.append(sink)
        if sink is not None:
            raise RuntimeError("synthetic attach failure")

    result = setup_control_debug_runtime(
        SimpleNamespace(
            control_debug_config=SimpleNamespace(
                enabled=True,
                root=tmp_path,
                level="summary",
            )
        ),
        components=object(),
        startup_cleanup=SimpleNamespace(add=lambda *_args: None),
        timestamp="20260808_010203_000004",
        expand_cli_path=lambda value: value,
        new_recorder=lambda *_args, **_kwargs: recorder,
        audit_fanout_type=lambda _recorders: object(),
        attach_position_sink=attach,
        update_runtime_metadata=lambda **updates: metadata.append(updates) or updates,
        wall_time=lambda: 10.0,
        monotonic_time=lambda: 20.0,
        log=lambda tag, message: logs.append((tag, message)),
    )

    assert result.audit_fanout is None
    assert result.recorder is None
    assert recorder.close_timeouts == [1.0]
    assert attached[-1] is None
    assert metadata[0]["attached"] is False
    assert "synthetic attach failure" in metadata[0]["error"]
    assert "synthetic attach failure" in logs[0][1]


def test_control_debug_setup_disabled_has_no_side_effects(tmp_path) -> None:
    result = setup_control_debug_runtime(
        SimpleNamespace(
            control_debug_config=SimpleNamespace(enabled=False),
        ),
        components=object(),
        startup_cleanup=SimpleNamespace(
            add=lambda *_args: pytest.fail("cleanup registration should be skipped")
        ),
        timestamp="",
        expand_cli_path=lambda _value: pytest.fail("path resolution should be skipped"),
        new_recorder=lambda *_args, **_kwargs: pytest.fail("recorder should be skipped"),
        audit_fanout_type=lambda *_args: pytest.fail("fanout should be skipped"),
        attach_position_sink=lambda *_args: pytest.fail("attach should be skipped"),
        update_runtime_metadata=lambda **_kwargs: pytest.fail("metadata should be skipped"),
        wall_time=lambda: pytest.fail("clock should be skipped"),
        monotonic_time=lambda: pytest.fail("clock should be skipped"),
        log=lambda *_args: pytest.fail("logging should be skipped"),
    )

    assert result.audit_fanout is None
    assert result.recorder is None
    assert result.session_dir is None
