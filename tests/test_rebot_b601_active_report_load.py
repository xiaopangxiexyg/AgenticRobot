from __future__ import annotations

import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from hardware.rebot_b601_rs import (
    B601ActiveReportReceipt,
    B601CommandReceipt,
    B601JointConfig,
    B601ModeReceipt,
    B601RSBackendConfig,
    FrequencyLedger,
)
from scripts.hardware import rebot_b601_rs_active_report_load_probe as load_probe

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


def _capture() -> dict[str, object]:
    return {
        "robstride_active_report_per_device_feedback": {
            str(index): {
                "sample_count": 120,
                "observed_hz": 40.0,
                "p99_interval_sec": 0.025,
            }
            for index in range(1, 7)
        },
        "robstride_active_report_frames_by_host_id": {"0xfd": 720},
        "robstride_short_active_report_frame_count": 0,
    }


class _FakeBackend:
    def __init__(self, config: B601RSBackendConfig) -> None:
        self.config = config
        self.connected = False
        self.calls: list[tuple[str, object]] = []
        self.state_observation_metadata_verified = False
        self._command_sequence = 0

    def connect(self) -> None:
        self.calls.append(("connect", None))
        self.connected = True

    def poll_feedback(self, *, request_feedback: bool) -> SimpleNamespace:
        assert request_feedback is True
        self.calls.append(("poll_feedback", request_feedback))
        return SimpleNamespace(
            valid=True,
            positions_rad=(0.0,) * 6,
            sequence=17,
            receive_timestamp_sec=time.monotonic(),
            fault_reason=None,
        )

    def safe_stop(self, *, joint_names: tuple[str, ...]) -> B601ModeReceipt:
        self.calls.append(("safe_stop", joint_names))
        return B601ModeReceipt(
            operation="safe_stop",
            requested_joint_names=joint_names,
            completed_joint_names=joint_names,
            failed_joint_names=(),
            errors=(),
            completed=True,
        )

    def prepare_mit_mode(self, *, joint_names: tuple[str, ...]) -> B601ModeReceipt:
        self.calls.append(("prepare_mit_mode", joint_names))
        return B601ModeReceipt(
            operation="prepare_mit_mode",
            requested_joint_names=joint_names,
            completed_joint_names=joint_names,
            failed_joint_names=(),
            errors=(),
            completed=True,
        )

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

    def send_mit(
        self,
        *_args: object,
        joint_names: tuple[str, ...],
        **_kwargs: object,
    ) -> B601CommandReceipt:
        self._command_sequence += 1
        self.calls.append(("send_mit", joint_names))
        completed = time.monotonic()
        return B601CommandReceipt(
            sequence=self._command_sequence,
            mode="mit",
            frame_id="b601_rs/joint_mit_reference",
            requested_at_sec=completed,
            completed_at_sec=completed,
            requested_joint_names=joint_names,
            sent_joint_names=joint_names,
            failed_joint_names=(),
            errors=(),
            accepted=True,
            source_sequence=17,
        )

    def close(self) -> None:
        self.calls.append(("close", None))
        self.connected = False


def _args(config_path: Path, source_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config=str(config_path),
        rebot_source_root=source_root,
        channel="can0",
        duration_sec=0.03,
        warmup_sec=0.05,
        target_tx_hz=125.0,
        target_report_hz=40.0,
        minimum_tx_rate_ratio=0.01,
        minimum_report_rate_ratio=0.90,
        maximum_tx_deadline_miss_ratio=1.0,
        maximum_report_p99_factor=2.0,
        minimum_tx_sample_count=2,
        minimum_report_sample_count=100,
        joint=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        allow_motion=True,
        allow_active_report=True,
        confirmation="B601_RS_ACTIVE_REPORT_TX_LOAD",
        auto_disable=True,
        auto_disable_active_report=True,
        output=None,
        raw_output=None,
    )


def test_validation_requires_explicit_motion_and_stream_cleanup_flags(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path / "rebotarm_rs.yaml", tmp_path)
    args.allow_motion = False

    with pytest.raises(load_probe.ActiveReportLoadProbeError, match="allow-motion"):
        load_probe._validate_args(args)


def test_evaluate_load_requires_both_tx_clocks_and_type24_stream() -> None:
    ledger = FrequencyLedger()
    ledger.mark("compute", 0.0)
    ledger.mark("compute", 0.01)
    ledger.mark("tx", 0.0)
    ledger.mark("tx", 0.01)
    config = B601RSBackendConfig(channel="can0", joints=_joints())

    frequency, tx_acceptance, report_acceptance, passed = load_probe._evaluate_load(
        capture=_capture(),
        ledger=ledger,
        config=config,
        selected=tuple(f"joint{index}" for index in range(1, 7)),
        target_tx_hz=100.0,
        target_report_hz=40.0,
        minimum_tx_rate_ratio=0.90,
        minimum_report_rate_ratio=0.90,
        maximum_tx_deadline_miss_ratio=0.01,
        maximum_report_p99_factor=2.0,
        minimum_tx_sample_count=2,
        minimum_report_sample_count=100,
    )

    assert passed is True
    assert frequency["tx"]["observed_hz"] == pytest.approx(100.0)
    assert tx_acceptance["passed"] is True
    assert report_acceptance is not None
    assert report_acceptance["passed"] is True


def test_load_probe_keeps_motors_disabled_and_joins_receiver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "rebotarm_rs.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")
    source_root = tmp_path / "rebot-source"
    source_root.mkdir()
    config = B601RSBackendConfig(
        channel="can0",
        joints=_joints(),
        allow_motion=True,
        allow_active_reporting=True,
        position_limits_rad=((-1.0, 1.0),) * 6,
        source_config=str(config_path),
    )
    backend = _FakeBackend(config)

    monkeypatch.setattr(
        load_probe.B601RSBackendConfig,
        "from_rs_yaml",
        staticmethod(lambda _path, **_kwargs: config),
    )
    monkeypatch.setattr(load_probe, "B601RSBackend", lambda _config: backend)
    monkeypatch.setattr(
        load_probe.frequency_probe,
        "_resolve_position_limits",
        lambda _path, _root: ((-1.0, 1.0),) * 6,
    )
    monkeypatch.setattr(load_probe.time, "sleep", lambda _seconds: None)

    def fake_capture(
        _channel: str,
        _duration_sec: float,
        *,
        raw_output: Path | None = None,
        ready: object = None,
        stop: Event | None = None,
    ) -> dict[str, object]:
        del raw_output
        assert ready is not None
        ready.set()  # type: ignore[union-attr]
        assert stop is not None
        stop.wait(1.0)
        return _capture()

    monkeypatch.setattr(
        load_probe.frequency_probe,
        "_read_only_capture",
        fake_capture,
    )

    result = load_probe._run_active_report_load_probe(_args(config_path, source_root))

    assert result["execution_status"] == "COMPLETED"
    assert result["acceptance_status"] == "PASS"
    assert result["stream_transport_acceptance"] == "PASS"
    assert result["tx_acceptance"] == "PASS"
    assert result["motion_commands_sent"] is False
    assert result["motors_enabled_by_probe"] is False
    assert result["tx_commands_sent"] is True
    assert result["tx_mode"] == "mit-zero-disabled"
    assert backend.calls[0] == ("connect", None)
    assert (
        "prepare_mit_mode",
        tuple(f"joint{index}" for index in range(1, 7)),
    ) in backend.calls
    assert (
        backend.calls.count(
            ("safe_stop", tuple(f"joint{index}" for index in range(1, 7)))
        )
        == 3
    )
    assert backend.calls[-1] == ("close", None)
    assert (
        backend.calls.count(
            (
                "set_active_reporting",
                (True, tuple(f"joint{index}" for index in range(1, 7))),
            )
        )
        == 1
    )
    assert (
        backend.calls.count(
            (
                "set_active_reporting",
                (False, tuple(f"joint{index}" for index in range(1, 7))),
            )
        )
        == 1
    )
