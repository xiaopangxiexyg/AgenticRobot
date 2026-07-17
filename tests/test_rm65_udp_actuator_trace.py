from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from hardware.rm65.utils.scripts import rm65_udp_actuator_trace as trace


def _config(*, ip: str, cycle: int = 20, joint_speed: int = 0) -> dict:
    return {
        "cycle": cycle,
        "enable": True,
        "port": 8089,
        "force_coordinate": -1,
        "ip": ip,
        "custom_config": {
            "joint_speed": joint_speed,
            "lift_state": 0,
            "expand_state": 1,
            "hand_state": 0,
            "arm_current_status": 1,
            "aloha_state": 0,
            "plus_base": 1,
            "plus_state": 0,
        },
    }


def _journal(path: Path, *, status: str = "configured") -> None:
    payload = {
        "schema": trace.JOURNAL_SCHEMA,
        "status": status,
        "arms": [
            {
                "side": "left",
                "arm_ip": "192.168.101.21",
                "original_config": _config(ip="192.168.101.10"),
            },
            {
                "side": "right",
                "arm_ip": "192.168.1.20",
                "original_config": _config(ip="192.168.1.10"),
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_direct_script_bootstraps_repository_import_path(tmp_path: Path) -> None:
    script = Path(trace.__file__).resolve()
    code = (
        "import runpy; "
        f"runpy.run_path({str(script)!r}, run_name='trace_import_test'); "
        "import hardware.rm65"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_temporary_push_config_preserves_all_unrelated_fields() -> None:
    original = _config(ip="10.1.2.3", cycle=40, joint_speed=0)

    target = trace.temporary_push_config(
        original,
        local_ip="192.168.101.10",
        push_port=8099,
        cycle_ms=5,
    )

    assert target == {
        **original,
        "cycle": 1,
        "port": 8099,
        "ip": "192.168.101.10",
        "custom_config": {**original["custom_config"], "joint_speed": 1},
    }
    assert original["custom_config"]["joint_speed"] == 0


def test_controller_cycle_one_means_one_five_ms_interval() -> None:
    original = _config(ip="10.1.2.3", cycle=1, joint_speed=0)

    normalized = trace.normalize_push_config(original)
    target = trace.temporary_push_config(
        original,
        local_ip="192.168.101.10",
        push_port=8099,
        cycle_ms=5,
    )

    assert normalized["cycle"] == 1
    assert target["cycle"] == 1


def test_temporary_push_config_refuses_unrestorable_joint_speed_setting() -> None:
    original = _config(ip="10.1.2.3", joint_speed=-1)

    with pytest.raises(ValueError, match="could not be restored exactly"):
        trace.temporary_push_config(
            original,
            local_ip="192.168.101.10",
            push_port=8099,
            cycle_ms=5,
        )


def test_decode_udp_payload_keeps_raw_and_promotes_joint_fields() -> None:
    payload = json.dumps(
        {
            "arm_ip": "192.168.1.20",
            "joint_status": {
                "joint_position": [1, 2, 3, 4, 5, 6, 99],
                "joint_speed": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 9.9],
                "joint_current": [10, 20, 30, 40, 50, 60, 90],
            },
        }
    ).encode()

    decoded = trace.decode_udp_payload(payload)

    assert decoded["payload_format"] == "json"
    assert decoded["joint_position_deg"] == [1, 2, 3, 4, 5, 6]
    assert decoded["joint_speed_deg_s"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert decoded["joint_current_ma"] == [10, 20, 30, 40, 50, 60]
    assert decoded["payload_base64"]


def test_restore_journal_restores_right_then_left_and_marks_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "capture.recovery.json"
    _journal(path)
    calls: list[str] = []

    def fake_set_and_verify(arm_ip: str, expected: dict) -> dict:
        calls.append(arm_ip)
        return trace.normalize_push_config(expected)

    monkeypatch.setattr(trace, "set_and_verify_config", fake_set_and_verify)

    trace.restore_journal(path, actor="test")

    assert calls == ["192.168.1.20", "192.168.101.21"]
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["status"] == "restored"
    assert restored["restore_actor"] == "test"
    assert set(restored["restored_config_by_side"]) == {"left", "right"}
    assert restored["restore_failures"] == []


def test_restore_failure_is_durable_and_never_marked_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "capture.recovery.json"
    _journal(path)

    def fake_set_and_verify(arm_ip: str, expected: dict) -> dict:
        if arm_ip == "192.168.1.20":
            raise RuntimeError("synthetic readback mismatch")
        return trace.normalize_push_config(expected)

    monkeypatch.setattr(trace, "set_and_verify_config", fake_set_and_verify)

    with pytest.raises(trace.RestoreError, match="synthetic readback mismatch"):
        trace.restore_journal(path, actor="test")

    failed = json.loads(path.read_text(encoding="utf-8"))
    assert failed["status"] == "restore_failed"
    assert failed["restored_config_by_side"].keys() == {"left"}
    assert "synthetic readback mismatch" in failed["restore_failures"][0]


def test_detached_guardian_must_signal_readiness_before_capture(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.recovery.json"
    _journal(path, status="restored")

    process, pipe = trace._start_guardian(path)
    try:
        ready_path = path.with_suffix(".guardian.ready.json")
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        assert ready["guardian_pid"] == process.pid
        assert process.poll() is None
        trace._notify_guardian_restored(process, pipe)
        pipe = None
        assert process.returncode == 0
    finally:
        if pipe is not None:
            pipe.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)


def test_capture_without_both_acknowledgements_fails_before_hardware() -> None:
    parser = trace.build_parser()
    args = parser.parse_args(["capture", "--duration-sec", "30"])
    with pytest.raises(RuntimeError, match="explicit --apply"):
        trace.capture(args)

    args = parser.parse_args(
        ["capture", "--duration-sec", "30", "--apply"]
    )
    with pytest.raises(RuntimeError, match="--i-understand-no-motion"):
        trace.capture(args)
