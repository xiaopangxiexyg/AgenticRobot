from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hardware.DMTAC_gripper.examples import identify_gripper_actuator as identify


class _FakeMB:
    REG_CATCH_MODE = 5
    REG_STOP = 6
    REG_SPEED_SET = 11
    REG_FORCE_SET = 12
    REG_OPENLEN_ACT = 61

    def __init__(self) -> None:
        self.registers = {
            self.REG_CATCH_MODE: 0,
            self.REG_SPEED_SET: 321,
            self.REG_FORCE_SET: 456,
        }
        self.writes: list[tuple[int, int]] = []

    def read_holding_registers(self, start: int, count: int) -> list[int]:
        return [self.registers[start + offset] for offset in range(count)]

    def write_single_register(self, register: int, value: int) -> None:
        self.writes.append((register, value))
        self.registers[register] = value


class _FakeGripper:
    STATUS_TEXT = {2: "closed", 4: "closing"}

    def __init__(self) -> None:
        self.mb = _FakeMB()
        self.stop_count = 0
        self.close_count = 0
        self.max_open_mm = 70.0
        self.last_joint_pos_cmd = 1.0
        self.last_speed_cmd = 1.0
        self.last_torque_cmd = 0.1
        self.position_calls: list[tuple[float, dict[str, object]]] = []

    def set_joint_pos(self, joint_pos: float, **kwargs: object) -> None:
        self.position_calls.append((float(joint_pos), dict(kwargs)))
        self.last_joint_pos_cmd = float(joint_pos)

    def set_speed(self, speed: float) -> None:
        self.last_speed_cmd = float(speed)
        self.mb.write_single_register(self.mb.REG_SPEED_SET, int(round(speed * 1000)))

    def set_torque(self, torque: float) -> None:
        self.last_torque_cmd = float(torque)
        self.mb.write_single_register(self.mb.REG_FORCE_SET, int(round(torque * 1000)))

    def set_continuous_grasp(self, continuous: bool) -> None:
        self.mb.write_single_register(self.mb.REG_CATCH_MODE, int(bool(continuous)))

    def get_open_limits_raw(self) -> tuple[int, int]:
        return (0, 1000)

    def estop(self) -> None:
        self.stop_count += 1

    def close(self) -> None:
        self.close_count += 1


class _CollectWriter:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def enqueue(self, row: object) -> None:
        self.rows.append(dict(row))


def _protocol_args(**overrides) -> argparse.Namespace:
    values = {
        "protocol": "contact",
        "force_levels": "0.10,0.20,0.35",
        "continuous_modes": "false,true",
        "free_continuous_grasp": False,
        "host_hold_modes": "one-shot,resend",
        "repeats": 1,
        "target_position": 1.0,
        "open_position": 0.0,
        "free_duration_sec": 3.0,
        "hold_duration_sec": 5.0,
        "sample_hz": 30.0,
        "resend_hz": 15.0,
        "object_label": "foam_block",
        "object_width_mm": 40.0,
        "allow_high_force": False,
        "high_force_confirmation": "",
        "command_mode": "runtime-equivalent",
        "target_sequence": "",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _journal(path: Path, *, status: str = "configured") -> None:
    identify._atomic_write_json(
        path,
        {
            "schema": identify.JOURNAL_SCHEMA,
            "status": status,
            "endpoint": {
                "side": "left",
                "port": "/dev/fake-gripper",
                "gripper_id": 1,
                "baudrate": 115200,
                "timeout": 0.05,
                "source": "test",
            },
            "original_settings_raw": {
                "speed_raw": 123,
                "force_raw": 234,
                "continuous_raw": 1,
            },
        },
    )


def test_contact_matrix_separates_device_and_host_hold() -> None:
    specs = identify.build_trial_specs(_protocol_args())

    assert len(specs) == 12
    assert {spec.force for spec in specs} == {0.1, 0.2, 0.35}
    assert {spec.continuous_grasp for spec in specs} == {False, True}
    assert {spec.host_position_hold for spec in specs} == {"one-shot", "resend"}
    assert all(spec.duration_sec == 5.0 for spec in specs)


def test_free_protocol_never_enables_hold_modes() -> None:
    specs = identify.build_trial_specs(
        _protocol_args(protocol="free", object_label="", object_width_mm=0.0)
    )

    assert len(specs) == 3
    assert all(spec.continuous_grasp is False for spec in specs)
    assert all(spec.host_position_hold == "one-shot" for spec in specs)


def test_free_protocol_accepts_ordered_partial_and_reversal_targets() -> None:
    specs = identify.build_trial_specs(
        _protocol_args(
            protocol="free",
            object_label="",
            object_width_mm=0.0,
            force_levels="0.2",
            target_sequence="0.25,0.5,0.75,1.0,0.75,0.5,0.25,0.0",
        )
    )

    assert len(specs) == 1
    assert specs[0].motion_targets == (0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0)


def test_free_protocol_can_match_runtime_continuous_setting() -> None:
    specs = identify.build_trial_specs(
        _protocol_args(
            protocol="free",
            object_label="",
            object_width_mm=0.0,
            force_levels="0.2",
            free_continuous_grasp=True,
        )
    )

    assert len(specs) == 1
    assert specs[0].force == 0.2
    assert specs[0].continuous_grasp is True
    assert specs[0].host_position_hold == "one-shot"


def test_contact_protocol_rejects_multi_target_sequence() -> None:
    with pytest.raises(ValueError, match="only supported by protocol=free"):
        identify.build_trial_specs(_protocol_args(target_sequence="0.5,1.0"))


def test_force_above_half_requires_two_explicit_acknowledgements() -> None:
    with pytest.raises(ValueError, match="force above 0.5"):
        identify.build_trial_specs(_protocol_args(force_levels="0.7"))

    with pytest.raises(ValueError, match="force above 0.5"):
        identify.build_trial_specs(
            _protocol_args(force_levels="0.7", allow_high_force=True)
        )

    specs = identify.build_trial_specs(
        _protocol_args(
            force_levels="0.7",
            allow_high_force=True,
            high_force_confirmation="HIGH_FORCE",
        )
    )
    assert {spec.force for spec in specs} == {0.7}


def test_invalid_timing_is_rejected_while_building_plan() -> None:
    with pytest.raises(ValueError, match="sample-hz"):
        identify.build_trial_specs(_protocol_args(sample_hz=0.0))
    with pytest.raises(ValueError, match="resend-hz"):
        identify.build_trial_specs(_protocol_args(resend_hz=0.0))
    with pytest.raises(ValueError, match="hold-duration-sec"):
        identify.build_trial_specs(_protocol_args(hold_duration_sec=0.0))


def test_resolve_endpoint_uses_selected_side_from_saved_config(tmp_path: Path) -> None:
    config_path = tmp_path / "hardware.json"
    config_path.write_text(
        json.dumps(
            {
                "created_at": "test",
                "assignments": {
                    "right": {
                        "gripper": {
                            "enabled": True,
                            "port": "/dev/right-by-path",
                            "gripper_id": 7,
                            "baudrate": 57600,
                            "timeout": 0.08,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        side="right",
        port=None,
        config=str(config_path),
        gripper_id=1,
        baudrate=115200,
        timeout=0.05,
    )

    endpoint, config = identify.resolve_endpoint(args)

    assert endpoint == identify.Endpoint(
        side="right",
        port="/dev/right-by-path",
        gripper_id=7,
        baudrate=57600,
        timeout=0.08,
        source=str(config_path),
    )
    assert config is not None


def test_restore_raw_settings_writes_and_verifies_exact_register_values() -> None:
    gripper = _FakeGripper()

    identify.restore_raw_settings(
        gripper,
        {"speed_raw": 111, "force_raw": 222, "continuous_raw": 1},
    )

    assert gripper.mb.writes == [(11, 111), (12, 222), (5, 1)]
    assert identify.read_raw_settings(gripper) == {
        "speed_raw": 111,
        "force_raw": 222,
        "continuous_raw": 1,
    }


def test_state_reader_decodes_uint16_negative_endpoint_without_false_error() -> None:
    gripper = _FakeGripper()
    gripper.mb.registers.update(
        {
            61: 0xFFFE,  # uint16 representation of -2 at the closed endpoint
            62: 0xFFFA,  # signed current -6
            63: 32,
            64: 0,
            65: 2,
        }
    )

    state = identify.read_state_safely(gripper)

    assert state["opening_register_raw_u16"] == 65534
    assert state["opening_register_signed"] == -2
    assert state["opening_raw"] == 0
    assert state["opening_endpoint_clamped"] is True
    assert state["joint_pos"] == 1.0
    assert state["opening_mm"] == 0.0
    assert state["current"] == -6
    assert state["status_code"] == 2


def test_state_reader_still_rejects_implausible_opening_values() -> None:
    gripper = _FakeGripper()
    gripper.mb.registers.update({61: 5000, 62: 0, 63: 32, 64: 0, 65: 2})

    with pytest.raises(RuntimeError, match="outside tolerated endpoint range"):
        identify.read_state_safely(gripper)


def test_runtime_temperature_limit_is_enforced_after_state_is_recorded() -> None:
    gripper = _FakeGripper()
    gripper.mb.registers.update({61: 500, 62: 10, 63: 56, 64: 0, 65: 4})
    writer = _CollectWriter()
    spec = identify.TrialSpec(
        trial_id="trial_hot",
        protocol="free",
        repeat=1,
        force=0.2,
        continuous_grasp=False,
        host_position_hold="one-shot",
        target_position=1.0,
        duration_sec=1.0,
    )

    with pytest.raises(RuntimeError, match="temperature=56C"):
        identify._sample_state(
            gripper,
            writer,
            spec,
            stage="motion_and_hold",
            sequence=1,
            max_temperature_c=55.0,
        )

    assert writer.rows[-1]["event"] == "state"
    assert writer.rows[-1]["read_midpoint_monotonic_ns"] > 0


def test_suite_preflight_tolerates_transient_moving_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = [
        {"is_moving": True, "error_code": 0, "temperature_c": 35},
        {"is_moving": False, "error_code": 0, "temperature_c": 35},
        {"is_moving": True, "error_code": 0, "temperature_c": 35},
        {"is_moving": False, "error_code": 0, "temperature_c": 35},
        {"is_moving": False, "error_code": 0, "temperature_c": 35},
        {"is_moving": False, "error_code": 0, "temperature_c": 35},
    ]
    iterator = iter(states)
    monkeypatch.setattr(identify, "read_state_safely", lambda gripper: next(iterator))
    monkeypatch.setattr(identify.time, "sleep", lambda duration: None)

    observed = identify._wait_for_suite_preflight_idle(
        _FakeGripper(),
        timeout_sec=1.0,
        max_temperature_c=55.0,
    )

    assert observed == states


def test_suite_preflight_accepts_sticky_status_only_at_safe_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = [
        {
            "joint_pos": 0.0,
            "current": current,
            "is_moving": True,
            "error_code": 0,
            "temperature_c": 34,
        }
        for current in (21, 0, 0, 0, 0)
    ]
    iterator = iter(states)
    evidence_path = tmp_path / "preflight.json"
    monkeypatch.setattr(identify, "read_state_safely", lambda gripper: next(iterator))
    monkeypatch.setattr(identify.time, "sleep", lambda duration: None)

    observed = identify._wait_for_suite_preflight_idle(
        _FakeGripper(),
        timeout_sec=1.0,
        max_temperature_c=55.0,
        evidence_path=evidence_path,
    )

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert observed == states
    assert evidence["accepted"] is True
    assert evidence["decision"] == "stable_safe_open_low_current_status_ignored"


def test_suite_preflight_rejects_sticky_status_away_from_safe_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "joint_pos": 0.5,
        "current": 0,
        "is_moving": True,
        "error_code": 0,
        "temperature_c": 34,
    }
    clock = iter((0.0, 0.0, 0.1, 0.2, 0.3, 0.4))
    evidence_path = tmp_path / "preflight.json"
    monkeypatch.setattr(identify, "read_state_safely", lambda gripper: dict(state))
    monkeypatch.setattr(identify.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(identify.time, "sleep", lambda duration: None)

    with pytest.raises(RuntimeError, match="neither status-idle nor stably safe-open"):
        identify._wait_for_suite_preflight_idle(
            _FakeGripper(),
            timeout_sec=0.35,
            max_temperature_c=55.0,
            evidence_path=evidence_path,
        )

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["accepted"] is False
    assert evidence["decision"] == "timeout_not_stationary"


def test_runtime_equivalent_command_times_only_position_transaction() -> None:
    gripper = _FakeGripper()
    writer = _CollectWriter()
    spec = identify.TrialSpec(
        trial_id="trial_001",
        protocol="free",
        repeat=1,
        force=0.2,
        continuous_grasp=False,
        host_position_hold="one-shot",
        target_position=1.0,
        duration_sec=1.0,
    )

    identify._send_position(
        gripper,
        writer,
        spec,
        stage="motion_initial",
        target=0.5,
        force=0.2,
        continuous=True,
        command_mode="runtime-equivalent",
        scheduled_monotonic_ns=123,
        segment_index=2,
    )

    assert gripper.position_calls == [
        (
            0.5,
            {
                "speed": None,
                "torque": None,
                "continuous": None,
                "force_send": True,
            },
        )
    ]
    assert [row["event"] for row in writer.rows] == ["command_begin", "command_end"]
    assert writer.rows[-1]["scheduled_monotonic_ns"] == 123
    assert (
        writer.rows[-1]["position_write_timing_semantics"]
        == "position_register_transaction_bounds"
    )


def test_runtime_equivalent_config_is_preapplied_and_read_back() -> None:
    gripper = _FakeGripper()
    writer = _CollectWriter()
    spec = identify.TrialSpec(
        trial_id="trial_001",
        protocol="contact",
        repeat=1,
        force=0.35,
        continuous_grasp=True,
        host_position_hold="one-shot",
        target_position=1.0,
        duration_sec=1.0,
    )

    observed = identify._configure_motion(
        gripper,
        writer,
        spec,
        stage="trial_motion",
        force=0.35,
        continuous=True,
    )

    assert observed == {
        "speed_raw": 1000,
        "force_raw": 350,
        "continuous_raw": 1,
    }
    assert [row["event"] for row in writer.rows] == [
        "motion_config_begin",
        "motion_config_end",
    ]


def test_async_writer_assigns_ordered_trace_sequence(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    writer = identify.AsyncJsonlWriter(trace_path, queue_size=16)
    writer.start()
    writer.enqueue({"event": "first"})
    writer.enqueue({"event": "second"})
    writer.close()

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [row["trace_sequence"] for row in rows] == [0, 1]
    assert [row["event"] for row in rows] == ["first", "second"]
    assert writer.written == 2
    assert writer.dropped == 0


def test_restore_journal_stops_then_restores_and_marks_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "settings.recovery.json"
    _journal(path)
    gripper = _FakeGripper()
    monkeypatch.setattr(identify, "_open_gripper", lambda endpoint: gripper)
    monkeypatch.setattr(identify, "_exclusive_serial", lambda value: None)

    identify.restore_journal(path, actor="test", stop_first=True)

    restored = json.loads(path.read_text(encoding="utf-8"))
    assert gripper.stop_count == 1
    assert gripper.close_count == 1
    assert restored["status"] == "restored"
    assert restored["restore_actor"] == "test"
    assert restored["restored_settings_raw"] == {
        "speed_raw": 123,
        "force_raw": 234,
        "continuous_raw": 1,
    }


def test_guardian_signals_readiness_before_any_experiment_write(tmp_path: Path) -> None:
    path = tmp_path / "settings.recovery.json"
    _journal(path, status="restored")

    process, pipe = identify._start_guardian(path)
    try:
        ready = json.loads(
            identify._guardian_ready_path(path).read_text(encoding="utf-8")
        )
        assert ready["guardian_pid"] == process.pid
        assert process.poll() is None
        identify._notify_guardian_restored(process, pipe)
        pipe = None
        assert process.returncode == 0
    finally:
        if pipe is not None:
            pipe.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)


def test_trial_summary_preserves_force_stop_drift_and_timing() -> None:
    spec = identify.TrialSpec(
        trial_id="trial_001",
        protocol="contact",
        repeat=1,
        force=0.2,
        continuous_grasp=False,
        host_position_hold="one-shot",
        target_position=1.0,
        duration_sec=5.0,
    )
    rows = [
        {
            "monotonic_time_ns": 1_000_000_000,
            "read_midpoint_monotonic_ns": 2_000_000_000,
            "state": {
                "joint_pos": 0.4,
                "current": 100,
                "temperature_c": 31,
                "status_code": 4,
            },
        },
        {
            "monotonic_time_ns": 1_033_000_000,
            "read_midpoint_monotonic_ns": 2_040_000_000,
            "state": {
                "joint_pos": 0.6,
                "current": 250,
                "temperature_c": 32,
                "status_code": 6,
            },
        },
        {
            "monotonic_time_ns": 1_066_000_000,
            "read_midpoint_monotonic_ns": 2_080_000_000,
            "state": {
                "joint_pos": 0.61,
                "current": 200,
                "temperature_c": 32,
                "status_code": 6,
            },
        },
    ]

    summary = identify.summarize_trial(
        spec,
        rows,
        resend_count=0,
        measured_force_n=4.2,
    )

    assert summary["force_stop_seen"] is True
    assert summary["max_abs_current"] == 250
    assert summary["hold_position_drift"] == pytest.approx(0.01)
    assert summary["sample_gap_ms"]["median"] == 40.0
    assert summary["measured_force_n"] == 4.2


def test_parser_defaults_to_runtime_equivalent_timing() -> None:
    parser = identify.build_parser()
    args = parser.parse_args(["run", "--side", "left"])

    assert args.command_mode == "runtime-equivalent"
    assert args.target_sequence == ""
    assert args.manual_gauge_timeout_sec == 30.0


def test_run_requires_motion_acknowledgement_before_hardware() -> None:
    parser = identify.build_parser()
    args = parser.parse_args(["run", "--side", "left"])

    with pytest.raises(RuntimeError, match="requires --apply"):
        identify.run(args)
