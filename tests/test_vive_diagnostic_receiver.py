from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import project_paths
from hardware.VIVE.mocap import vive_diagnostic_receiver
from runtime_artifact_paths import RUNTIME_ARTIFACT_LAYOUT_CLI_NAME


def _pose(*, x: float = 0.0) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _packet(
    seq: int,
    *,
    sample_time: float,
    x: float = 0.0,
    linear_velocity: list[float] | None = None,
) -> bytes:
    linear_velocity = linear_velocity or [0.0, 0.0, 0.0]
    return json.dumps(
        {
            "protocol": "teleop_runtime.vive/v1",
            "seq": seq,
            "sender_timestamp": 1_800_000_000.0 + sample_time,
            "pose_sample_timestamp": 1_800_000_000.0 + sample_time,
            "pose_sample_monotonic_time": sample_time,
            "trackers": [
                {
                    "tracker_name": "tracker_1",
                    "tracker_role": "left_wrist",
                    "valid": True,
                    "pose_matrix_4x4": _pose(x=x),
                    "error": None,
                    "read_elapsed_ms": 0.2,
                    "linear_velocity_m_s": linear_velocity,
                    "angular_velocity_rad_s": [0.0, 0.0, 0.0],
                    "velocity_frame_name": "vive_world",
                    "velocity_source": "openvr_tracked_device_pose",
                    "velocity_valid": True,
                    "velocity_error": None,
                }
            ],
        }
    ).encode("utf-8")


def _jsonl(path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_receiver_cli_defaults_to_300hz_and_teleop_records() -> None:
    args = vive_diagnostic_receiver.parse_args(["--label", "static"])

    assert args.expected_hz == 300.0
    assert args.output_root is None
    assert args.artifact_layout == "legacy"
    assert args.port == 9999
    assert args.duration_sec == 0.0


def test_receiver_canonical_artifact_layout_is_explicit_and_side_effect_free(
    monkeypatch,
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    monkeypatch.setattr(
        project_paths,
        "local_config",
        lambda: {
            "runtime_artifacts_root": str(canonical),
            "teleop_records_root": str(legacy),
        },
    )

    legacy_args = vive_diagnostic_receiver.parse_args(["--label", "static"])
    canonical_args = vive_diagnostic_receiver.parse_args(
        [
            "--label",
            "static",
            "--artifact-layout",
            RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
        ]
    )

    assert vive_diagnostic_receiver.resolve_diagnostic_output_root(legacy_args) == (
        legacy / "vive_diagnostics"
    )
    assert vive_diagnostic_receiver.resolve_diagnostic_output_root(canonical_args) == (
        canonical / "evidence" / "preflight" / "vive"
    )
    assert not canonical.exists()
    assert not legacy.exists()


def test_receiver_explicit_output_root_precedes_artifact_layout(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    args = vive_diagnostic_receiver.parse_args(
        [
            "--label",
            "static",
            "--artifact-layout",
            RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
            "--output-root",
            str(explicit),
        ]
    )

    assert vive_diagnostic_receiver.resolve_diagnostic_output_root(args) == explicit
    assert not explicit.exists()


def test_recorder_keeps_every_packet_and_derives_contiguous_pose_velocity(tmp_path) -> None:
    recorder = vive_diagnostic_receiver.ViveDiagnosticRecorder(
        output_root=tmp_path,
        label="motion",
        expected_hz=300.0,
        flush_every_packets=1,
    )
    source = ("192.168.0.10", 54321)
    recorder.record_datagram(
        _packet(10, sample_time=100.0, x=0.0, linear_velocity=[0.9, 0.0, 0.0]),
        source=source,
        receiver_wall_time=1_800_000_100.0,
        receiver_monotonic_time=200.0,
    )
    recorder.record_datagram(
        _packet(
            11,
            sample_time=100.0 + 1.0 / 300.0,
            x=0.003,
            linear_velocity=[0.9, 0.0, 0.0],
        ),
        source=source,
        receiver_wall_time=1_800_000_100.0 + 1.0 / 300.0,
        receiver_monotonic_time=200.0 + 1.0 / 300.0,
    )
    summary_path = recorder.close(reason="test")

    packet_rows = _jsonl(recorder.packet_path)
    tracker_rows = _jsonl(recorder.tracker_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert [row["seq"] for row in packet_rows] == [10, 11]
    assert len(tracker_rows) == 2
    second = tracker_rows[1]
    assert second["pose_pair_contiguous"] is True
    assert second["position_step_m"] == pytest.approx(0.003)
    assert second["pose_delta_time_sec"] == pytest.approx(1.0 / 300.0)
    assert second["pose_derived_linear_speed_m_s"] == pytest.approx(0.9)
    assert second["linear_speed_m_s"] == pytest.approx(0.9)
    assert second["pose_repeated"] is False
    assert summary["counts"]["datagrams"] == 2
    assert summary["counts"]["sequence_missing_estimate"] == 0
    tracker_summary = summary["trackers"]["left_wrist:tracker_1"]
    assert tracker_summary["counts"]["contiguous_pose_pairs"] == 1
    assert tracker_summary["metrics"]["contiguous_position_step_m"]["p99"] == pytest.approx(
        0.003
    )


def test_recorder_reports_sequence_loss_repeated_pose_and_malformed_datagram(tmp_path) -> None:
    recorder = vive_diagnostic_receiver.ViveDiagnosticRecorder(
        output_root=tmp_path,
        label="static trajectory",
        flush_every_packets=1,
    )
    source = ("192.168.0.10", 54321)
    recorder.record_datagram(
        _packet(5, sample_time=10.0),
        source=source,
        receiver_wall_time=100.0,
        receiver_monotonic_time=20.0,
    )
    recorder.record_datagram(
        _packet(8, sample_time=10.01),
        source=source,
        receiver_wall_time=100.01,
        receiver_monotonic_time=20.01,
    )
    recorder.record_datagram(
        b"not-json",
        source=source,
        receiver_wall_time=100.02,
        receiver_monotonic_time=20.02,
    )
    summary_path = recorder.close(reason="test")

    packet_rows = _jsonl(recorder.packet_path)
    tracker_rows = _jsonl(recorder.tracker_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert packet_rows[1]["sequence_status"] == "gap"
    assert packet_rows[1]["missing_packets_before"] == 2
    assert packet_rows[2]["decode_ok"] is False
    assert "raw_payload_base64" in packet_rows[2]
    assert tracker_rows[1]["pose_repeated"] is True
    assert tracker_rows[1]["pose_pair_contiguous"] is False
    assert summary["counts"]["sequence_missing_estimate"] == 2
    assert summary["counts"]["decode_errors"] == 1
    assert summary["trackers"]["left_wrist:tracker_1"]["counts"][
        "repeated_pose_rows"
    ] == 1
    assert math.isclose(summary["observed_receive_hz"], 100.0)
