from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hardware.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601HardwareError,
    B601JointConfig,
    B601ModeReceipt,
    B601MotionAuthorizationError,
    B601RSBackend,
    B601RSBackendConfig,
    FrequencyLedger,
)
from robot_motion.rebot_b601_rs import B601_RS_JOINT_MIT_FRAME, B601MitReference
from scripts.hardware import rebot_b601_rs_frequency_probe as frequency_probe

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _joint_configs() -> tuple[B601JointConfig, ...]:
    return tuple(
        B601JointConfig(
            name=name,
            motor_id=index,
            feedback_id=0xFD,
            model="rs-06" if index <= 3 else "rs-00",
            mit_kp=50.0,
            mit_kd=3.0,
            position_velocity_limit_rad_s=10.0,
        )
        for index, name in enumerate(B601_RS_ARM_JOINT_NAMES, start=1)
    )


def _config(*, allow_motion: bool = False) -> B601RSBackendConfig:
    return B601RSBackendConfig(
        channel="can0",
        rate_hz=500.0,
        joints=_joint_configs(),
        allow_motion=allow_motion,
        position_limits_rad=((-3.0, 3.0),) * 6,
    )


def _yaml_mapping() -> dict[str, object]:
    joints = []
    for index, name in enumerate(B601_RS_ARM_JOINT_NAMES, start=1):
        joints.append(
            {
                "name": name,
                "motor_id": f"0x{index:02x}",
                "feedback_id": "0xFD",
                "model": "rs-06" if index <= 3 else "rs-00",
                "vendor": "robstride",
                "MIT": {"kp": 50.0, "kd": 3.0},
                "POS_VEL": {"vlim": 10.0},
            }
        )
    return {
        "channel": "can0",
        "rate": 500,
        "groups": {"arm": {"joints": list(B601_RS_ARM_JOINT_NAMES)}},
        "joints": joints,
    }


class _FakeMotor:
    def __init__(self, motor_id: int, *, state: object | None = None) -> None:
        self.motor_id = motor_id
        self.state = state
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.fail_method: str | None = None

    def _call(self, method: str, *args: object) -> None:
        self.calls.append((method, args))
        if self.fail_method == method:
            raise RuntimeError(f"synthetic {method} failure")

    def get_state(self) -> object | None:
        self._call("get_state")
        return self.state

    def ensure_mode(self, *args: object) -> None:
        self._call("ensure_mode", *args)

    def enable(self) -> None:
        self._call("enable")

    def disable(self) -> None:
        self._call("disable")

    def send_pos_vel(self, *args: object) -> None:
        self._call("send_pos_vel", *args)

    def send_mit(self, *args: object) -> None:
        self._call("send_mit", *args)


class _FakeController:
    def __init__(self) -> None:
        self.motors: dict[int, _FakeMotor] = {}
        self.calls: list[str] = []

    def add_robstride_motor(self, motor_id: int, feedback_id: int, model: str) -> _FakeMotor:
        del feedback_id, model
        self.calls.append(f"add:{motor_id}")
        motor = _FakeMotor(motor_id)
        self.motors[motor_id] = motor
        return motor

    def poll_feedback_once(self) -> None:
        self.calls.append("poll_feedback_once")

    def close(self) -> None:
        self.calls.append("close")


def _populate_states(controller: _FakeController) -> None:
    for index, motor in controller.motors.items():
        motor.state = SimpleNamespace(
            can_id=index,
            arbitration_id=0x18000000 + index,
            status_code=0,
            pos=0.1 * index,
            vel=0.0,
            torq=0.0,
        )


def test_config_parser_keeps_explicit_arm_order_and_excludes_gripper() -> None:
    config = B601RSBackendConfig.from_mapping(_yaml_mapping())

    assert config.channel == "can0"
    assert config.rate_hz == pytest.approx(500.0)
    assert tuple(joint.name for joint in config.joints) == B601_RS_ARM_JOINT_NAMES
    assert config.joints[0].motor_id == 1
    assert config.joints[-1].motor_id == 6
    assert config.allow_motion is False


def test_connect_and_poll_do_not_request_feedback_or_enable() -> None:
    controller = _FakeController()
    backend = B601RSBackend(_config(), controller_factory=lambda _channel: controller)

    backend.connect()
    _populate_states(controller)
    state = backend.poll_feedback()

    assert backend.connected is True
    assert controller.calls == [f"add:{index}" for index in range(1, 7)] + [
        "poll_feedback_once"
    ]
    assert state.valid is True
    assert state.stale is False
    assert state.freshness_verified is False
    assert state.freshness_semantics == "poll_observation_unverified"
    assert state.positions_rad == pytest.approx(tuple(0.1 * index for index in range(1, 7)))
    assert all(call[0] == "get_state" for motor in controller.motors.values() for call in motor.calls)


def test_empty_feedback_is_invalid_and_never_silently_zero_filled() -> None:
    controller = _FakeController()
    backend = B601RSBackend(_config(), controller_factory=lambda _channel: controller)
    backend.connect()

    state = backend.poll_feedback()

    assert state.valid is False
    assert state.stale is True
    assert state.positions_rad is None
    assert all(joint.position_rad is None for joint in state.joints)
    assert "feedback cache is empty" in (state.fault_reason or "")


def test_wrong_motor_identity_and_vendor_status_are_invalid() -> None:
    controller = _FakeController()
    backend = B601RSBackend(_config(), controller_factory=lambda _channel: controller)
    backend.connect()
    _populate_states(controller)
    controller.motors[2].state.can_id = 99  # type: ignore[union-attr]
    controller.motors[3].state.status_code = 7  # type: ignore[union-attr]

    state = backend.poll_feedback()

    assert state.valid is False
    assert state.joint("joint2").valid is False
    assert state.joint("joint3").valid is False
    assert "can_id=99" in (state.fault_reason or "")
    assert "status_code=7" in (state.fault_reason or "")


def test_command_paths_require_explicit_motion_authorization() -> None:
    controller = _FakeController()
    backend = B601RSBackend(_config(), controller_factory=lambda _channel: controller)
    backend.connect()

    with pytest.raises(B601MotionAuthorizationError):
        backend.send_position_velocity((0.0,) * 6)
    reference = B601MitReference(
        sequence=1,
        source_timestamp_sec=1.0,
        receive_timestamp_sec=1.0,
        qpos_rad=(0.0,) * 6,
    )
    with pytest.raises(B601MotionAuthorizationError):
        backend.send_mit(
            reference,
            kp=(0.0,) * 6,
            kd=(0.0,) * 6,
            tau_nm=(0.0,) * 6,
            torque_limits_nm=(1.0,) * 6,
        )
    assert all(not motor.calls for motor in controller.motors.values())


def test_safe_stop_is_available_without_motion_start_authorization() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(),
        controller_factory=lambda _channel: controller,
    )
    backend.connect()

    receipt = backend.safe_stop(joint_names=("joint1", "joint2"))

    assert receipt.completed is True
    assert receipt.operation == "safe_stop"
    assert receipt.completed_joint_names == ("joint1", "joint2")
    assert all(
        motor.calls == [("disable", ())]
        for index, motor in controller.motors.items()
        if index in (1, 2)
    )
    assert all(
        not motor.calls
        for index, motor in controller.motors.items()
        if index not in (1, 2)
    )


def test_motion_requires_urdf_limits_and_respects_velocity_envelope() -> None:
    controller = _FakeController()
    config_without_limits = B601RSBackendConfig(
        channel="can0",
        rate_hz=500.0,
        joints=_joint_configs(),
        allow_motion=True,
    )
    backend_without_limits = B601RSBackend(
        config_without_limits,
        controller_factory=lambda _channel: controller,
    )
    backend_without_limits.connect()

    with pytest.raises(B601HardwareError, match="position_limits_rad"):
        backend_without_limits.send_position_velocity((0.0,) * 6)

    bounded_backend = B601RSBackend(
        _config(allow_motion=True),
        controller_factory=lambda _channel: controller,
    )
    bounded_backend.connect()
    with pytest.raises(ValueError, match="configured per-joint limit"):
        bounded_backend.send_position_velocity(
            (0.0,) * 6,
            velocity_limits_rad_s=(11.0,) * 6,
        )
    reference = B601MitReference(
        sequence=1,
        source_timestamp_sec=1.0,
        receive_timestamp_sec=1.0,
        qpos_rad=(0.0,) * 6,
        qvel_rad_s=(11.0,) * 6,
    )
    with pytest.raises(ValueError, match="reference.qvel_rad_s"):
        bounded_backend.send_mit(
            reference,
            kp=(0.0,) * 6,
            kd=(0.0,) * 6,
            tau_nm=(0.0,) * 6,
            torque_limits_nm=(1.0,) * 6,
        )


def test_position_dispatch_reports_all_sent_joints_and_no_hidden_partial_success() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(allow_motion=True),
        controller_factory=lambda _channel: controller,
        clock=iter([10.0, 10.001]).__next__,
    )
    backend.connect()
    controller.motors[4].fail_method = "send_pos_vel"

    receipt = backend.send_position_velocity((0.1,) * 6, joint_names=("joint1", "joint4"))

    assert receipt.accepted is False
    assert receipt.sent_joint_names == ("joint1",)
    assert receipt.failed_joint_names == ("joint4",)
    assert receipt.errors == ("joint4: synthetic send_pos_vel failure",)
    assert controller.motors[1].calls[-1][0] == "send_pos_vel"


def test_mit_dispatch_enforces_final_torque_limit_and_preserves_reference_frame() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(allow_motion=True),
        controller_factory=lambda _channel: controller,
        clock=iter([20.0, 20.001]).__next__,
    )
    backend.connect()
    reference = B601MitReference(
        sequence=42,
        source_timestamp_sec=1.0,
        receive_timestamp_sec=1.0,
        qpos_rad=(0.1,) * 6,
        qvel_rad_s=(0.0,) * 6,
    )

    with pytest.raises(ValueError, match="final torque limits"):
        backend.send_mit(
            reference,
            kp=(1.0,) * 6,
            kd=(1.0,) * 6,
            tau_nm=(1.1,) * 6,
            torque_limits_nm=(1.0,) * 6,
        )
    receipt = backend.send_mit(
        reference,
        kp=(1.0,) * 6,
        kd=(1.0,) * 6,
        tau_nm=(0.5,) * 6,
        torque_limits_nm=(1.0,) * 6,
        joint_names=("joint1",),
    )

    assert receipt.accepted is True
    assert receipt.source_sequence == 42
    assert receipt.frame_id == B601_RS_JOINT_MIT_FRAME
    assert controller.motors[1].calls[-1][0] == "send_mit"
    assert all(not motor.calls for index, motor in controller.motors.items() if index != 1)


def test_frequency_ledger_keeps_feedback_compute_tx_and_haptic_separate() -> None:
    ledger = FrequencyLedger()
    for timestamp in (0.0, 0.01, 0.02):
        ledger.mark("feedback", timestamp)
    for timestamp in (0.0, 0.02):
        ledger.mark("compute", timestamp)
    ledger.mark("tx", 0.01)

    summary = ledger.as_dict(target_hz=100.0)

    assert summary["feedback"]["sample_count"] == 3
    assert summary["feedback"]["observed_hz"] == pytest.approx(100.0)
    assert summary["compute"]["sample_count"] == 2
    assert summary["tx"]["sample_count"] == 1
    assert summary["haptic"]["sample_count"] == 0
    assert summary["haptic"]["observed_hz"] is None


class _FakePositionProbeBackend:
    def __init__(
        self,
        *,
        enable_receipt: B601ModeReceipt,
        stop_receipt: B601ModeReceipt,
    ) -> None:
        self.enable_receipt = enable_receipt
        self.stop_receipt = stop_receipt
        self.calls: list[str] = []
        self.sequence = 0

    def connect(self) -> None:
        self.calls.append("connect")

    def poll_feedback(self) -> object:
        self.calls.append("poll_feedback")
        self.sequence += 1
        return SimpleNamespace(
            valid=True,
            positions_rad=(0.0,) * 6,
            receive_timestamp_sec=float(self.sequence),
            sequence=self.sequence,
            fault_reason=None,
        )

    def prepare_position_velocity_mode(self, *, joint_names: object) -> B601ModeReceipt:
        del joint_names
        self.calls.append("prepare")
        return B601ModeReceipt(
            "prepare_position_velocity_mode",
            ("joint1", "joint2"),
            ("joint1", "joint2"),
            (),
            (),
            True,
        )

    def enable(self, *, joint_names: object) -> B601ModeReceipt:
        del joint_names
        self.calls.append("enable")
        return self.enable_receipt

    def send_position_velocity(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.calls.append("send")
        return SimpleNamespace(
            accepted=True,
            errors=(),
            completed_at_sec=1.0,
        )

    def safe_stop(self, *, joint_names: object) -> B601ModeReceipt:
        del joint_names
        self.calls.append("safe_stop")
        return self.stop_receipt

    def close(self) -> None:
        self.calls.append("close")


def _position_probe_args(config_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        allow_motion=True,
        confirmation="B601_RS_POSITION_FREQUENCY",
        enable=True,
        auto_disable=True,
        config=str(config_path),
        duration_sec=1e-9,
        warmup_sec=0.01,
        joint=("joint1", "joint2"),
        rebot_source_root=None,
        target_hz=125.0,
    )


def _install_position_probe_fakes(
    monkeypatch: pytest.MonkeyPatch,
    backend: _FakePositionProbeBackend,
) -> None:
    monkeypatch.setattr(
        frequency_probe,
        "_resolve_position_limits",
        lambda _config_path, _source_root: ((-1.0, 1.0),) * 6,
    )
    monkeypatch.setattr(
        frequency_probe.B601RSBackendConfig,
        "from_rs_yaml",
        staticmethod(lambda _path, **_kwargs: SimpleNamespace(channel="can0")),
    )
    monkeypatch.setattr(frequency_probe, "B601RSBackend", lambda _config: backend)


def test_position_probe_stops_all_selected_joints_after_partial_enable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "rebotarm_rs.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")
    backend = _FakePositionProbeBackend(
        enable_receipt=B601ModeReceipt(
            "enable",
            ("joint1", "joint2"),
            ("joint1",),
            ("joint2",),
            ("joint2: synthetic enable failure",),
            False,
        ),
        stop_receipt=B601ModeReceipt(
            "safe_stop",
            ("joint1", "joint2"),
            ("joint1", "joint2"),
            (),
            (),
            True,
        ),
    )
    _install_position_probe_fakes(monkeypatch, backend)

    with pytest.raises(
        frequency_probe.FrequencyProbeError,
        match="motor enable failed",
    ):
        frequency_probe._run_position_probe(_position_probe_args(config_path))

    assert backend.calls[-2:] == ["safe_stop", "close"]


def test_position_probe_reports_incomplete_safe_stop_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "rebotarm_rs.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")
    backend = _FakePositionProbeBackend(
        enable_receipt=B601ModeReceipt(
            "enable",
            ("joint1", "joint2"),
            ("joint1", "joint2"),
            (),
            (),
            True,
        ),
        stop_receipt=B601ModeReceipt(
            "safe_stop",
            ("joint1", "joint2"),
            ("joint1",),
            ("joint2",),
            ("joint2: synthetic disable failure",),
            False,
        ),
    )
    _install_position_probe_fakes(monkeypatch, backend)

    result = frequency_probe._run_position_probe(_position_probe_args(config_path))

    assert result["cleanup_receipt"] == {
        "operation": "safe_stop",
        "requested_joint_names": ("joint1", "joint2"),
        "completed_joint_names": ("joint1",),
        "failed_joint_names": ("joint2",),
        "errors": ("joint2: synthetic disable failure",),
        "completed": False,
    }
    assert result["cleanup_errors"] == (
        "safe_stop: joint2: synthetic disable failure",
    )
    assert result["stop_reason"] == "safe_stop: joint2: synthetic disable failure"


@pytest.mark.parametrize("target_hz", (0.0, float("nan"), float("inf")))
def test_position_probe_rejects_nonfinite_or_nonpositive_target_before_connect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_hz: float,
) -> None:
    config_path = tmp_path / "rebotarm_rs.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")
    backend = _FakePositionProbeBackend(
        enable_receipt=B601ModeReceipt(
            "enable", ("joint1", "joint2"), ("joint1", "joint2"), (), (), True
        ),
        stop_receipt=B601ModeReceipt(
            "safe_stop", ("joint1", "joint2"), ("joint1", "joint2"), (), (), True
        ),
    )
    _install_position_probe_fakes(monkeypatch, backend)
    args = _position_probe_args(config_path)
    args.target_hz = target_hz

    with pytest.raises(ValueError, match="target_hz must be positive"):
        frequency_probe._run_position_probe(args)

    assert backend.calls == []
