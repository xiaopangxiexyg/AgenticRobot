from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from hardware.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601_RS_POS_VEL_LIMIT_SPD_PARAM_ID,
    B601_RS_POS_VEL_POS_KP_PARAM_ID,
    B601_RS_POS_VEL_VEL_KI_PARAM_ID,
    B601_RS_POS_VEL_VEL_KP_PARAM_ID,
    B601HardwareError,
    B601JointConfig,
    B601ModeReceipt,
    B601MotionAuthorizationError,
    B601RSBackend,
    B601RSBackendConfig,
    FrequencyLedger,
    evaluate_frequency_target,
)
from robot_motion.rebot_b601_rs import B601_RS_JOINT_MIT_FRAME, B601MitReference
from scripts.hardware import rebot_b601_rs_frequency_probe as frequency_probe

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _joint_configs() -> tuple[B601JointConfig, ...]:
    velocity_kp = (12.0, 14.0, 14.0, 5.0, 4.0, 4.0)
    position_kp = (13.0, 16.0, 14.0, 20.0, 10.0, 10.0)
    return tuple(
        B601JointConfig(
            name=name,
            motor_id=index,
            feedback_id=0xFD,
            model="rs-06" if index <= 3 else "rs-00",
            mit_kp=50.0,
            mit_kd=3.0,
            position_velocity_limit_rad_s=10.0,
            position_velocity_kp=velocity_kp[index - 1],
            position_velocity_ki=0.1,
            position_kp=position_kp[index - 1],
            position_ki=0.0,
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
    velocity_kp = (12.0, 14.0, 14.0, 5.0, 4.0, 4.0)
    position_kp = (13.0, 16.0, 14.0, 20.0, 10.0, 10.0)
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
                "POS_VEL": {
                    "vel_kp": velocity_kp[index - 1],
                    "vel_ki": 0.1,
                    "pos_kp": position_kp[index - 1],
                    "pos_ki": 0.0,
                    "vlim": 10.0,
                },
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
        self.position_param: float | None = None
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.fail_method: str | None = None

    def _call(self, method: str, *args: object) -> None:
        self.calls.append((method, args))
        if self.fail_method == method:
            raise RuntimeError(f"synthetic {method} failure")

    def get_state(self) -> object | None:
        self._call("get_state")
        return self.state

    def request_feedback(self) -> None:
        self._call("request_feedback")

    def robstride_get_param_f32(self, *args: object) -> float:
        self._call("robstride_get_param_f32", *args)
        if self.position_param is None:
            raise RuntimeError("synthetic position parameter unavailable")
        return self.position_param

    def robstride_write_param_f32(self, *args: object) -> None:
        self._call("robstride_write_param_f32", *args)

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
        motor.position_param = 0.1 * index


def _socketcan_frame(arbitration_id: int, *, data_length: int = 8) -> bytes:
    return frequency_probe._CAN_FRAME.pack(
        arbitration_id,
        data_length,
        b"\x00" * 8,
    )


def test_passive_probe_classifies_robstride_operation_status_id_without_payload_decode() -> None:
    arbitration_id = (2 << 24) | (0x0003 << 8) | 0xFD
    decoded = frequency_probe.decode_can_frame(
        _socketcan_frame(arbitration_id | frequency_probe._CAN_EFF_FLAG)
    )

    identity = frequency_probe.decode_robstride_extended_id(decoded)

    assert decoded.arbitration_id == arbitration_id
    assert decoded.extended is True
    assert decoded.data_length == 8
    assert identity.communication_type == 2
    assert identity.communication_name == "operation_status"
    assert identity.extra_data == 3
    assert identity.node_id == 0xFD
    assert identity.stream_device_id == 3


def test_passive_probe_rejects_standard_frame_as_robstride_private_id() -> None:
    decoded = frequency_probe.decode_can_frame(_socketcan_frame(0x123))

    with pytest.raises(ValueError, match="requires an extended CAN frame"):
        frequency_probe.decode_robstride_extended_id(decoded)


def test_passive_probe_rejects_short_or_invalid_classic_can_frame() -> None:
    with pytest.raises(ValueError, match="too short"):
        frequency_probe.decode_can_frame(b"\x00" * 15)
    with pytest.raises(ValueError, match="data length is invalid"):
        frequency_probe.decode_can_frame(_socketcan_frame(0x123, data_length=9))


def test_passive_summary_keeps_status_timing_fail_closed() -> None:
    summary = frequency_probe._robstride_candidate_summary(
        Counter({2: 4, 24: 1, 31: 2}),
        {
            2: {1: [0.0, 0.01], 2: [0.0, 0.02]},
            24: {7: [0.0]},
        },
        {
            2: Counter({0xFD: 4}),
            24: Counter({0xFD: 1}),
        },
        short_stream_frame_counts=Counter({2: 1}),
    )

    assert summary["robstride_extended_candidate_frame_count"] == 7
    assert summary["frames_by_robstride_communication_type"] == {
        "2:operation_status": 4,
        "24:active_report": 1,
        "31:unknown": 2,
    }
    assert summary["robstride_unknown_communication_type_frame_count"] == 2
    assert summary["robstride_operation_status_frame_count"] == 4
    assert summary["robstride_short_operation_status_frame_count"] == 1
    assert summary["robstride_operation_status_frames_by_device_id"] == {
        "1": 2,
        "2": 2,
    }
    assert summary["robstride_operation_status_frames_by_host_id"] == {"0xfd": 4}
    assert summary["robstride_operation_status_per_device_feedback"]["1"][
        "observed_hz"
    ] == pytest.approx(100.0)
    assert summary["robstride_active_report_frame_count"] == 1
    assert summary["robstride_short_active_report_frame_count"] == 0
    assert summary["robstride_active_report_frames_by_device_id"] == {"7": 1}
    assert summary["robstride_active_report_frames_by_host_id"] == {"0xfd": 1}
    assert summary["feedback_path_classification"] == (
        "ROBSTRIDE_OPERATION_STATUS_AND_ACTIVE_REPORT_CANDIDATES_OBSERVED"
    )


def test_config_parser_keeps_explicit_arm_order_and_excludes_gripper() -> None:
    config = B601RSBackendConfig.from_mapping(_yaml_mapping())

    assert config.channel == "can0"
    assert config.rate_hz == pytest.approx(500.0)
    assert tuple(joint.name for joint in config.joints) == B601_RS_ARM_JOINT_NAMES
    assert config.joints[0].motor_id == 1
    assert config.joints[-1].motor_id == 6
    assert config.joints[0].position_velocity_kp == pytest.approx(12.0)
    assert config.joints[0].position_velocity_ki == pytest.approx(0.1)
    assert config.joints[0].position_kp == pytest.approx(13.0)
    assert config.joints[0].position_ki == pytest.approx(0.0)
    assert config.allow_motion is False


def test_prepare_position_velocity_mode_mirrors_rebot_parameter_order() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(allow_motion=True),
        controller_factory=lambda _channel: controller,
        sleeper=lambda _seconds: None,
    )

    backend.connect()
    receipt = backend.prepare_position_velocity_mode(
        joint_names=("joint1",),
        velocity_limits_rad_s=(0.1,) * 6,
    )

    assert receipt.completed is True
    assert receipt.completed_joint_names == ("joint1",)
    assert controller.motors[1].calls == [
        ("robstride_write_param_f32", (B601_RS_POS_VEL_LIMIT_SPD_PARAM_ID, 0.1)),
        ("robstride_write_param_f32", (B601_RS_POS_VEL_VEL_KP_PARAM_ID, 12.0)),
        ("robstride_write_param_f32", (B601_RS_POS_VEL_VEL_KI_PARAM_ID, 0.1)),
        ("robstride_write_param_f32", (B601_RS_POS_VEL_POS_KP_PARAM_ID, 13.0)),
        ("ensure_mode", (2, 1000)),
    ]
    assert all(not motor.calls for index, motor in controller.motors.items() if index != 1)


def test_prepare_position_velocity_mode_fails_closed_on_parameter_write() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(allow_motion=True),
        controller_factory=lambda _channel: controller,
        sleeper=lambda _seconds: None,
    )

    backend.connect()
    controller.motors[1].fail_method = "robstride_write_param_f32"

    receipt = backend.prepare_position_velocity_mode(joint_names=("joint1",))

    assert receipt.completed is False
    assert receipt.failed_joint_names == ("joint1",)
    assert "joint1: synthetic robstride_write_param_f32 failure" in receipt.errors
    assert all(call[0] != "ensure_mode" for call in controller.motors[1].calls)


def test_prepare_mit_mode_mirrors_rebot_transition_and_never_enables() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(allow_motion=True),
        controller_factory=lambda _channel: controller,
        sleeper=lambda _seconds: None,
    )

    backend.connect()
    receipt = backend.prepare_mit_mode(joint_names=("joint1",))

    assert receipt.completed is True
    assert receipt.operation == "prepare_mit_mode"
    assert receipt.completed_joint_names == ("joint1",)
    assert controller.motors[1].calls == [("ensure_mode", (1, 1000))]
    assert all(
        not motor.calls for index, motor in controller.motors.items() if index != 1
    )
    assert all(
        call[0] != "enable"
        for motor in controller.motors.values()
        for call in motor.calls
    )


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


def test_active_feedback_poll_requests_each_configured_arm_joint() -> None:
    controller = _FakeController()
    backend = B601RSBackend(_config(), controller_factory=lambda _channel: controller)

    backend.connect()
    _populate_states(controller)
    state = backend.poll_feedback(request_feedback=True)

    assert state.valid is True
    assert controller.calls == [f"add:{index}" for index in range(1, 7)] + [
        "poll_feedback_once"
    ]
    assert all(
        motor.calls == [
            ("request_feedback", ()),
            ("get_state", ()),
            ("robstride_get_param_f32", (0x7019, 100)),
        ]
        for motor in controller.motors.values()
    )


def test_active_feedback_poll_uses_mech_pos_when_motor_state_cache_is_empty() -> None:
    controller = _FakeController()
    backend = B601RSBackend(_config(), controller_factory=lambda _channel: controller)

    backend.connect()
    for index, motor in controller.motors.items():
        motor.position_param = 0.1 * index

    state = backend.poll_feedback(request_feedback=True)

    assert state.valid is True
    assert state.positions_rad == pytest.approx(tuple(0.1 * index for index in range(1, 7)))
    assert state.velocities_reported is None
    assert all(
        joint.position_source == "robstride_mech_pos_parameter"
        for joint in state.joints
    )


def test_active_feedback_request_failure_is_invalid_and_fail_closed() -> None:
    controller = _FakeController()
    backend = B601RSBackend(_config(), controller_factory=lambda _channel: controller)

    backend.connect()
    _populate_states(controller)
    controller.motors[2].fail_method = "request_feedback"

    state = backend.poll_feedback(request_feedback=True)

    assert state.valid is False
    assert state.stale is True
    assert "joint2.request_feedback: synthetic request_feedback failure" in (
        state.fault_reason or ""
    )


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


def test_position_dispatch_checks_position_limits_only_for_selected_joints() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(allow_motion=True),
        controller_factory=lambda _channel: controller,
    )
    backend.connect()
    qpos = (0.1, 4.0, 0.1, 0.1, 0.1, 0.1)

    receipt = backend.send_position_velocity(qpos, joint_names=("joint1",))

    assert receipt.accepted is True
    assert receipt.sent_joint_names == ("joint1",)
    assert controller.motors[1].calls[-1] == ("send_pos_vel", (0.1, 10.0))
    assert all(
        not motor.calls
        for index, motor in controller.motors.items()
        if index != 1
    )
    with pytest.raises(ValueError, match=r"qpos_rad\[1\].*joint2"):
        backend.send_position_velocity(qpos, joint_names=("joint2",))


def test_mit_dispatch_checks_position_limits_only_for_selected_joints() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(allow_motion=True),
        controller_factory=lambda _channel: controller,
    )
    backend.connect()
    reference = B601MitReference(
        sequence=1,
        source_timestamp_sec=1.0,
        receive_timestamp_sec=1.0,
        qpos_rad=(0.1, 4.0, 0.1, 0.1, 0.1, 0.1),
    )

    receipt = backend.send_mit(
        reference,
        kp=(1.0,) * 6,
        kd=(1.0,) * 6,
        tau_nm=(0.0,) * 6,
        torque_limits_nm=(1.0,) * 6,
        joint_names=("joint1",),
    )

    assert receipt.accepted is True
    assert receipt.sent_joint_names == ("joint1",)
    assert controller.motors[1].calls[-1] == (
        "send_mit",
        (0.1, 0.0, 1.0, 1.0, 0.0),
    )
    assert all(
        not motor.calls
        for index, motor in controller.motors.items()
        if index != 1
    )
    with pytest.raises(ValueError, match=r"reference\.qpos_rad\[1\].*joint2"):
        backend.send_mit(
            reference,
            kp=(1.0,) * 6,
            kd=(1.0,) * 6,
            tau_nm=(0.0,) * 6,
            torque_limits_nm=(1.0,) * 6,
            joint_names=("joint2",),
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


def test_frequency_acceptance_distinguishes_completion_from_target_pass() -> None:
    ledger = FrequencyLedger()
    for timestamp in (0.0, 0.05, 0.10):
        ledger.mark("tx", timestamp)

    acceptance = evaluate_frequency_target(
        ledger.summarize(target_hz=125.0)["tx"]
    )

    assert acceptance.passed is False
    assert acceptance.observed_hz == pytest.approx(20.0)
    assert acceptance.minimum_observed_hz == pytest.approx(118.75)
    assert acceptance.deadline_miss_ratio == pytest.approx(1.0)
    assert len(acceptance.reasons) == 2


def test_frequency_acceptance_passes_explicit_rate_and_deadline_limits() -> None:
    ledger = FrequencyLedger()
    for timestamp in (0.0, 0.008, 0.016, 0.024):
        ledger.mark("tx", timestamp)

    acceptance = evaluate_frequency_target(
        ledger.summarize(target_hz=125.0)["tx"]
    )

    assert acceptance.passed is True
    assert acceptance.reasons == ()


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
        self.send_sequence = 0
        self.mit_send_sequence = 0
        self.prepare_velocity_limits: object = None

    def connect(self) -> None:
        self.calls.append("connect")

    def poll_feedback(self, *, request_feedback: bool = False) -> object:
        if request_feedback:
            self.calls.append("request_feedback")
        self.calls.append("poll_feedback")
        self.sequence += 1
        return SimpleNamespace(
            valid=True,
            positions_rad=(0.0,) * 6,
            receive_timestamp_sec=float(self.sequence),
            sequence=self.sequence,
            fault_reason=None,
        )

    def prepare_position_velocity_mode(
        self,
        *,
        joint_names: object,
        velocity_limits_rad_s: object = None,
    ) -> B601ModeReceipt:
        del joint_names
        self.prepare_velocity_limits = velocity_limits_rad_s
        self.calls.append("prepare")
        return B601ModeReceipt(
            "prepare_position_velocity_mode",
            ("joint1", "joint2"),
            ("joint1", "joint2"),
            (),
            (),
            True,
        )

    def prepare_mit_mode(self, *, joint_names: object) -> B601ModeReceipt:
        del joint_names
        self.calls.append("prepare_mit")
        return B601ModeReceipt(
            "prepare_mit_mode",
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
        self.send_sequence += 1
        return SimpleNamespace(
            accepted=True,
            errors=(),
            completed_at_sec=float(self.send_sequence),
        )

    def send_mit(self, *args: object, **kwargs: object) -> object:
        del args
        self.calls.append("send_mit")
        assert kwargs["kp"] == (0.0,) * 6
        assert kwargs["kd"] == (0.0,) * 6
        assert kwargs["tau_nm"] == (0.0,) * 6
        assert kwargs["torque_limits_nm"] == (0.0,) * 6
        self.mit_send_sequence += 1
        return SimpleNamespace(
            accepted=True,
            errors=(),
            completed_at_sec=float(self.mit_send_sequence),
        )

    def safe_stop(self, *, joint_names: object) -> B601ModeReceipt:
        del joint_names
        self.calls.append("safe_stop")
        return self.stop_receipt

    def close(self) -> None:
        self.calls.append("close")


def _position_probe_args(config_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        mode="position",
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
        velocity_limit_rad_s=0.1,
        minimum_rate_ratio=0.95,
        maximum_deadline_miss_ratio=0.01,
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
        staticmethod(
            lambda _path, **_kwargs: SimpleNamespace(
                channel="can0",
                joints=tuple(
                    SimpleNamespace(position_velocity_limit_rad_s=10.0)
                    for _ in range(6)
                ),
            )
        ),
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
    assert backend.prepare_velocity_limits == (0.1,) * 6


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
    assert backend.prepare_velocity_limits == (0.1,) * 6


def test_disabled_tx_probe_never_enables_or_polls_feedback_after_warmup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "rebotarm_rs.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")
    backend = _FakePositionProbeBackend(
        enable_receipt=B601ModeReceipt(
            "enable", ("joint1", "joint2"), ("joint1", "joint2"), (), (), True
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
    args = _position_probe_args(config_path)
    args.mode = "position-tx-disabled"
    args.enable = False
    args.confirmation = "B601_RS_DISABLED_TX_FREQUENCY"
    args.duration_sec = 0.02

    result = frequency_probe._run_position_probe(args)

    assert "enable" not in backend.calls
    assert backend.calls.count("request_feedback") == 1
    assert backend.calls.count("poll_feedback") == 1
    assert backend.calls.count("safe_stop") == 2
    assert backend.calls.index("safe_stop") < backend.calls.index("prepare")
    assert backend.calls[-2:] == ["safe_stop", "close"]
    assert backend.send_sequence >= 2
    assert result["motors_enabled_by_probe"] is False
    assert result["freshness_semantics"] == (
        "initial_vendor_snapshot_only; no_runtime_feedback"
    )
    assert result["acceptance"]["required_events"] == ("compute", "tx")


def test_disabled_mit_tx_probe_double_disables_and_sends_zero_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    args.mode = "mit-tx-disabled"
    args.enable = False
    args.confirmation = "B601_RS_DISABLED_MIT_TX_FREQUENCY"
    args.duration_sec = 0.02

    result = frequency_probe._run_position_probe(args)

    assert "enable" not in backend.calls
    assert backend.calls.count("request_feedback") == 1
    assert backend.calls.count("poll_feedback") == 1
    assert backend.calls.count("safe_stop") == 3
    assert backend.calls.index("safe_stop") < backend.calls.index("prepare_mit")
    assert backend.calls.index("prepare_mit") < backend.calls.index(
        "safe_stop", backend.calls.index("prepare_mit")
    )
    assert backend.calls[-2:] == ["safe_stop", "close"]
    assert backend.mit_send_sequence >= 2
    assert result["motors_enabled_by_probe"] is False
    assert result["post_mode_disable_receipt"]["completed"] is True
    assert result["mit_command_semantics"]["final_commanded_effort_nm"] == (
        0.0,
    ) * 6
    assert result["acceptance"]["required_events"] == ("compute", "tx")


def test_gravity_offline_probe_reuses_model_and_zero_pd_composition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeGravityModel:
        urdf_path = tmp_path / "robot.urdf"

        def arm_position_limits(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
            return (-1.0,) * 6, (1.0,) * 6

        def arm_generalized_gravity(self, qpos: object) -> tuple[float, ...]:
            assert qpos == (0.0,) * 6
            return (1.0, -2.0, 3.0, -4.0, 5.0, -6.0)

    monkeypatch.setattr(
        frequency_probe.B601PinocchioModel,
        "from_rebot_source_root",
        staticmethod(lambda _root: _FakeGravityModel()),
    )

    result = frequency_probe._gravity_offline_capture(1e-9, 125.0, tmp_path)

    assert result["motion_commands_sent"] is False
    assert result["motors_enabled_by_probe"] is False
    assert result["iterations"] == 1
    assert result["last_gravity_torque_nm"] == (
        1.0,
        -2.0,
        3.0,
        -4.0,
        5.0,
        -6.0,
    )
    assert result["last_requested_torque_nm"] == result["last_gravity_torque_nm"]
    assert result["acceptance"]["required_events"] == ("compute",)


def test_probe_main_returns_nonzero_for_completed_frequency_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        frequency_probe,
        "_run_position_probe",
        lambda _args: {
            "mode": "position",
            "execution_status": "COMPLETED",
            "acceptance_status": "FAIL",
        },
    )

    return_code = frequency_probe.main(["--mode", "position"])

    assert return_code == 1
    assert '"status": "FAIL"' in capsys.readouterr().out


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
