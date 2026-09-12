from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hardware.rebot_b601_rs import (
    B601ActiveReportReceipt,
    B601JointConfig,
    B601RSBackendConfig,
    evaluate_active_report_capture,
)
from scripts.hardware import rebot_b601_rs_active_report_probe as active_probe

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _joints() -> tuple[B601JointConfig, ...]:
    return tuple(
        B601JointConfig(
            name=f"joint{index}",
            motor_id=index,
            feedback_id=0xFD,
            model="rs-06" if index <= 3 else "rs-00",
            position_velocity_limit_rad_s=1.0,
        )
        for index in range(1, 7)
    )


def _capture(*, joint2_hz: float = 125.0) -> dict[str, object]:
    return {
        "robstride_active_report_per_device_feedback": {
            "1": {
                "sample_count": 100,
                "observed_hz": 125.0,
                "p99_interval_sec": 0.009,
            },
            "2": {
                "sample_count": 100,
                "observed_hz": joint2_hz,
                "p99_interval_sec": 0.009,
            },
        },
        "robstride_active_report_frames_by_host_id": {"0xfd": 200},
        "robstride_short_active_report_frame_count": 0,
    }


def test_active_report_acceptance_requires_every_selected_device() -> None:
    acceptance = evaluate_active_report_capture(
        _capture(),
        _joints(),
        ("joint1", "joint2"),
        target_hz=125.0,
    )

    assert acceptance.passed is True
    assert all(device.passed for device in acceptance.devices)


def test_active_report_acceptance_fails_slow_or_missing_device() -> None:
    acceptance = evaluate_active_report_capture(
        _capture(joint2_hz=20.0),
        _joints(),
        ("joint1", "joint2", "joint3"),
        target_hz=125.0,
    )

    assert acceptance.passed is False
    assert acceptance.devices[0].passed is True
    assert acceptance.devices[1].passed is False
    assert acceptance.devices[2].passed is False
    assert any("joint2: observed_hz=20.000" in reason for reason in acceptance.reasons)
    assert any("joint3: no complete" in reason for reason in acceptance.reasons)


def test_active_report_acceptance_does_not_use_operation_status_as_active_report() -> None:
    operation_status_only = {
        "robstride_operation_status_per_device_feedback": {
            "1": {
                "sample_count": 100,
                "observed_hz": 125.0,
                "p99_interval_sec": 0.009,
            },
            "2": {
                "sample_count": 100,
                "observed_hz": 125.0,
                "p99_interval_sec": 0.009,
            },
        },
        "robstride_operation_status_frames_by_host_id": {"0xfd": 200},
        "robstride_short_operation_status_frame_count": 0,
    }

    acceptance = evaluate_active_report_capture(
        operation_status_only,
        _joints(),
        ("joint1", "joint2"),
        target_hz=125.0,
    )

    assert acceptance.passed is False
    assert all(device.passed is False for device in acceptance.devices)
    assert all(
        any("no complete type-24 active-report" in reason for reason in device.reasons)
        for device in acceptance.devices
    )


def test_active_report_acceptance_rejects_short_active_report_frames() -> None:
    capture = _capture()
    capture["robstride_short_active_report_frame_count"] = 1

    acceptance = evaluate_active_report_capture(
        capture,
        _joints(),
        ("joint1", "joint2"),
        target_hz=125.0,
    )

    assert acceptance.passed is False
    assert acceptance.short_active_report_frame_count == 1
    assert all(
        any("short type-24 active-report" in reason for reason in device.reasons)
        for device in acceptance.devices
    )


class _FakeBackend:
    def __init__(self, config: B601RSBackendConfig) -> None:
        self.config = config
        self.connected = False
        self.calls: list[tuple[str, object]] = []
        self.state_observation_metadata_verified = True

    def connect(self) -> None:
        self.calls.append(("connect", None))
        self.connected = True

    def set_active_reporting(
        self,
        enabled: bool,
        *,
        joint_names: tuple[str, ...],
    ) -> B601ActiveReportReceipt:
        self.calls.append(("set_active_reporting", (enabled, joint_names)))
        return B601ActiveReportReceipt(
            operation=(
                "enable_active_reporting" if enabled else "disable_active_reporting"
            ),
            requested_joint_names=joint_names,
            completed_joint_names=joint_names,
            failed_joint_names=(),
            errors=(),
            completed=True,
        )

    def close(self) -> None:
        self.calls.append(("close", None))
        self.connected = False


def _args(config_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config=str(config_path),
        channel="can0",
        duration_sec=0.01,
        target_hz=125.0,
        minimum_rate_ratio=0.90,
        maximum_p99_factor=2.0,
        minimum_sample_count=10,
        joint=("joint1", "joint2"),
        allow_active_report=True,
        confirmation="B601_RS_ACTIVE_REPORT_TELEMETRY",
        auto_disable_active_report=True,
    )


def _install_probe_fakes(
    monkeypatch: pytest.MonkeyPatch,
    config_path: Path,
) -> _FakeBackend:
    config = B601RSBackendConfig(
        channel="can0",
        joints=_joints(),
        allow_active_reporting=True,
        source_config=str(config_path),
    )
    backend = _FakeBackend(config)
    monkeypatch.setattr(
        active_probe.B601RSBackendConfig,
        "from_rs_yaml",
        staticmethod(lambda _path, **_kwargs: config),
    )
    monkeypatch.setattr(active_probe, "B601RSBackend", lambda _config: backend)
    monkeypatch.setattr(active_probe.time, "sleep", lambda _seconds: None)
    return backend


def test_probe_pass_is_transport_only_and_always_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "rebotarm_rs.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")
    backend = _install_probe_fakes(monkeypatch, config_path)
    monkeypatch.setattr(active_probe.frequency_probe, "_read_only_capture", lambda *_: _capture())

    result = active_probe._run_active_report_probe(_args(config_path))

    assert result["execution_status"] == "COMPLETED"
    assert result["stream_transport_acceptance"] == "PASS"
    assert result["control_feedback_admission"] == "NOT_ADMITTED"
    assert result["motion_commands_sent"] is False
    assert result["motors_enabled_by_probe"] is False
    assert result["motorbridge_state_observation_metadata_available"] is True
    assert "type-24 active-report" in result["notes"][0]
    assert backend.calls == [
        ("connect", None),
        ("set_active_reporting", (True, ("joint1", "joint2"))),
        ("set_active_reporting", (False, ("joint1", "joint2"))),
        ("close", None),
    ]


def test_probe_capture_failure_still_disables_active_reporting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "rebotarm_rs.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")
    backend = _install_probe_fakes(monkeypatch, config_path)

    def fail_capture(*_args: object) -> dict[str, object]:
        raise RuntimeError("synthetic capture failure")

    monkeypatch.setattr(active_probe.frequency_probe, "_read_only_capture", fail_capture)

    result = active_probe._run_active_report_probe(_args(config_path))

    assert result["execution_status"] == "ERROR"
    assert result["stream_transport_acceptance"] == "FAIL"
    assert "synthetic capture failure" in result["errors"]
    assert backend.calls[-2:] == [
        ("set_active_reporting", (False, ("joint1", "joint2"))),
        ("close", None),
    ]


def test_probe_rejects_missing_cleanup_authorization_before_connect(tmp_path: Path) -> None:
    config_path = tmp_path / "rebotarm_rs.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")
    args = _args(config_path)
    args.auto_disable_active_report = False

    with pytest.raises(active_probe.ActiveReportProbeError, match="auto-disable"):
        active_probe._run_active_report_probe(args)
