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
        # ``-I`` ignores PYTHON* environment variables, including the
        # snapshot runner's PYTHONPYCACHEPREFIX.  Keep the direct-script
        # import probe isolated and explicitly prevent source-tree bytecode.
        [sys.executable, "-I", "-B", "-c", code],
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


def test_decode_udp_payload_compact_omits_redundant_raw_json() -> None:
    payload = json.dumps(
        {
            "joint_status": {
                "joint_position": [1, 2, 3, 4, 5, 6],
                "joint_speed": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "joint_current": [10, 20, 30, 40, 50, 60],
            }
        }
    ).encode()

    decoded = trace.decode_udp_payload(payload, compact=True)

    assert decoded["payload_format"] == "json"
    assert decoded["joint_position_deg"] == [1, 2, 3, 4, 5, 6]
    assert decoded["joint_speed_deg_s"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert decoded["joint_current_ma"] == [10, 20, 30, 40, 50, 60]
    assert "payload_base64" not in decoded
    assert "payload_json" not in decoded


def test_decode_udp_payload_compact_retains_schema_sample_and_binary_error() -> None:
    payload = json.dumps(
        {"joint_status": {"joint_position": [1, 2, 3, 4, 5, 6]}}
    ).encode()

    sample = trace.decode_udp_payload(payload, compact=True, retain_raw=True)
    invalid = trace.decode_udp_payload(b"\xffinvalid", compact=True)

    assert sample["schema_sample"] is True
    assert sample["payload_base64"]
    assert sample["payload_json"]
    assert invalid["payload_format"] == "binary"
    assert invalid["payload_base64"]


def test_capture_parser_exposes_backward_compatible_full_and_compact_format() -> None:
    parser = trace.build_parser()

    default_args = parser.parse_args(["capture", "--duration-sec", "30"])
    compact_args = parser.parse_args(
        ["capture", "--duration-sec", "30", "--format", "compact"]
    )

    assert default_args.trace_format == "full"
    assert compact_args.trace_format == "compact"


def test_capture_output_root_defaults_to_configured_legacy_location() -> None:
    args = trace.build_parser().parse_args(["capture", "--duration-sec", "30"])

    resolution = trace.resolve_output_root(
        args,
        date_partition="2026-08-09",
    )

    assert resolution.path == trace.legacy_output_root()
    assert resolution.source == "legacy"
    assert args.output_root is None


def test_capture_output_root_supports_explicit_canonical_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical" / "2026-08-09"
    monkeypatch.setattr(
        trace,
        "canonical_output_root",
        lambda *, date_partition=None: canonical,
    )
    args = trace.build_parser().parse_args(
        [
            "capture",
            "--duration-sec",
            "30",
            "--artifact-layout",
            trace.RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
        ]
    )

    resolution = trace.resolve_output_root(
        args,
        date_partition="2026-08-09",
    )

    assert resolution.path == canonical
    assert resolution.source == "canonical"
    assert not (tmp_path / "canonical").exists()


def test_explicit_capture_output_root_wins_over_layout(tmp_path: Path) -> None:
    explicit = tmp_path / "exact"
    args = trace.build_parser().parse_args(
        [
            "capture",
            "--duration-sec",
            "30",
            "--artifact-layout",
            trace.RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
            "--output-root",
            str(explicit),
        ]
    )

    resolution = trace.resolve_output_root(args)

    assert resolution.path == explicit
    assert resolution.source == "explicit"


def test_canonical_capture_is_allocated_as_one_run_directory(tmp_path: Path) -> None:
    resolution = trace.ArtifactWriteResolution(
        path=tmp_path / "canonical-date",
        source="canonical",
        layout_name=trace.RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
    )

    paths = trace.allocate_capture_artifact_paths(
        resolution,
        timestamp="20260809_010203_000004",
    )

    expected_run = (
        tmp_path
        / "canonical-date"
        / "rm65_udp_actuator_trace_20260809_010203_000004"
    )
    assert paths.run_directory == expected_run
    assert paths.trace_path.parent == expected_run
    assert paths.journal_path.parent == expected_run
    assert not expected_run.exists()


def test_legacy_capture_keeps_flat_file_layout(tmp_path: Path) -> None:
    resolution = trace.ArtifactWriteResolution(
        path=tmp_path / "legacy",
        source="legacy",
        layout_name=trace.LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
    )

    paths = trace.allocate_capture_artifact_paths(
        resolution,
        timestamp="20260809_010203_000004",
    )

    assert paths.run_directory == tmp_path / "legacy"
    assert paths.trace_path.parent == tmp_path / "legacy"
    assert paths.journal_path.parent == tmp_path / "legacy"


def test_recovery_discovery_covers_nested_canonical_runs(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    canonical = tmp_path / "canonical"
    legacy.mkdir()
    nested = canonical / "2026-08-09" / "run-001"
    nested.mkdir(parents=True)
    legacy_journal = legacy / "legacy.recovery.json"
    canonical_journal = nested / "canonical.recovery.json"
    legacy_journal.write_text("{}", encoding="utf-8")
    canonical_journal.write_text("{}", encoding="utf-8")

    paths = trace.recovery_journal_paths((legacy, canonical, legacy))

    assert paths == tuple(
        sorted((legacy_journal.resolve(), canonical_journal.resolve()))
    )


def test_recovery_roots_cover_legacy_canonical_family_and_explicit_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = tmp_path / "legacy"
    canonical = tmp_path / "canonical"
    explicit = tmp_path / "explicit"
    monkeypatch.setattr(trace, "legacy_output_root", lambda: legacy)
    monkeypatch.setattr(trace, "canonical_output_family_root", lambda: canonical)

    roots = trace.recovery_journal_roots(explicit, legacy)

    assert roots == (legacy.resolve(), canonical.resolve(), explicit.resolve())
    assert all(not root.exists() for root in roots)


def test_capture_lock_prevents_parallel_capture_across_output_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trace, "legacy_output_root", lambda: tmp_path / "legacy")
    monkeypatch.setattr(trace, "_process_lock_root", lambda: tmp_path / "locks")

    with trace.exclusive_capture_locks(
        trace._capture_lock_roots(tmp_path / "layout-a")
    ):
        with pytest.raises(RuntimeError, match="another UDP actuator capture"):
            with trace.exclusive_capture_locks(
                trace._capture_lock_roots(tmp_path / "layout-b")
            ):
                pytest.fail("capture must have one cross-layout process owner")


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
