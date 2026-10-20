from __future__ import annotations

import json
import math
import struct
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from hardware.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601_RS_MECH_POS_PARAM_ID,
    B601_RS_POS_VEL_LIMIT_SPD_PARAM_ID,
    B601_RS_POS_VEL_POS_KP_PARAM_ID,
    B601_RS_POS_VEL_VEL_KI_PARAM_ID,
    B601_RS_POS_VEL_VEL_KP_PARAM_ID,
    B601_RS_TORQUE_LIMIT_PARAM_ID,
    B601ActiveReportAuthorizationError,
    B601HardwareDependencyError,
    B601HardwareError,
    B601JointConfig,
    B601JointState,
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


@pytest.fixture
def audit_evidence():
    frames, samples = [], []
    for motor in range(1, 7):
        base = motor * 100
        samples.append({"seq": motor - 1, "motor_id": motor, "completed": True,
                        "request_start_monotonic_ns": base + 10,
                        "request_end_monotonic_ns": base + 50, "mechpos_rad": float(motor)})
        for offset, cid, payload in (
            (20, (17 << 24) | (0xFD << 8) | motor, bytes.fromhex("1970000000000000")),
            (30, (17 << 24) | (motor << 8) | 0xFD, bytes.fromhex("19700000") + struct.pack("<f", motor)),
            (40, (24 << 24) | (motor << 8) | 0xFD, bytes(8)),
        ):
            frames.append({"kind": "frame", "receive_monotonic_ns": base + offset,
                           "clock_pair_end_monotonic_ns": base + offset + 1,
                           "socket_frame_hex": struct.pack("=IB3x8s", cid | 0x80000000, 8, payload).hex()})
    records = [{"kind": "header", "format": "b601_socketcan_rx_v1", "channel": "can-test"},
               *frames, {"kind": "end", "completed": True, "received_frame_count": len(frames)}]
    summary = {"mode": "read-paired", "status": "OK", "motion_commands_sent": False,
               "motors_enabled_by_probe": False, "parameter_samples": samples,
               "passive_capture": {"frame_count": len(frames), "channel": "can-test"}}
    return summary, records


def test_offline_pair_audit_matches_without_promoting_freshness(audit_evidence):
    original = deepcopy(audit_evidence)
    report = frequency_probe.audit_paired_capture(*audit_evidence)
    assert report["host_pairing_passed"] and report["matched_read_count"] == 6
    assert not report["freshness_verified"]
    assert not report["static_decode_validation_passed"]
    assert not report["quantitative_replay_ready"]
    assert audit_evidence == original


def test_offline_pair_audit_inventories_other_stream_host_without_using_it_as_reply(audit_evidence):
    summary, records = audit_evidence
    raw = bytearray.fromhex(records[3]["socket_frame_hex"])
    raw[0] = 0xFE
    records[3]["socket_frame_hex"] = raw.hex()
    report = frequency_probe.audit_paired_capture(summary, records)
    assert report["host_pairing_passed"]
    assert report["mixed_stream_host_ids"]
    assert report["stream_frames_by_host"] == {"0xfe": 1, "0xfd": 5}


@pytest.mark.parametrize("fault", [
    "wrong_motor", "wrong_host", "wrong_param", "wrong_value", "short_frame", "rtr",
    "duplicate_rx", "missing_rx", "truncated", "overlap", "nan", "bad_seq", "j7",
    "incomplete", "wrong_channel", "reverse_clock", "unexpected_tx", "response_before_request",
])
def test_offline_pair_audit_fails_closed(audit_evidence, fault):
    summary, records = audit_evidence
    sample = summary["parameter_samples"][0]
    raw = bytearray.fromhex(records[2]["socket_frame_hex"])
    if fault == "wrong_motor": raw[1] = 2
    elif fault == "wrong_host": raw[0] = 0xFE
    elif fault == "wrong_param": raw[8] = 0x1B
    elif fault == "wrong_value": raw[12:16] = struct.pack("<f", 99)
    elif fault == "short_frame": raw[4] = 4
    elif fault == "rtr": raw[3] |= 0x40
    elif fault == "unexpected_tx": raw[3] = 0x83
    elif fault == "duplicate_rx":
        extra = deepcopy(records[2])
        extra["receive_monotonic_ns"] += 1
        extra["clock_pair_end_monotonic_ns"] += 1
        records.insert(3, extra)
    elif fault == "missing_rx": records.pop(2)
    elif fault == "truncated": records.pop()
    elif fault == "overlap": summary["parameter_samples"][1]["request_start_monotonic_ns"] = 110
    elif fault == "nan": sample["mechpos_rad"] = math.nan
    elif fault == "bad_seq": sample["seq"] = True
    elif fault == "j7": sample["motor_id"] = 7
    elif fault == "incomplete": sample["completed"] = False
    elif fault == "wrong_channel": records[0]["channel"] = "other"
    elif fault == "reverse_clock": records[2]["receive_monotonic_ns"] = 1
    elif fault == "response_before_request":
        records[1]["socket_frame_hex"], records[2]["socket_frame_hex"] = records[2]["socket_frame_hex"], records[1]["socket_frame_hex"]
    if fault in {"wrong_motor", "wrong_host", "wrong_param", "wrong_value", "short_frame", "rtr", "unexpected_tx"}:
        records[2]["socket_frame_hex"] = raw.hex()
    if fault in {"duplicate_rx", "missing_rx"}:
        records[-1]["received_frame_count"] = len(records) - 2
        summary["passive_capture"]["frame_count"] = len(records) - 2
    report = frequency_probe.audit_paired_capture(summary, records)
    assert report["status"] == "FAIL" and report["issues"]
    assert not report["host_pairing_passed"]


def test_audit_cli_never_opens_transport(tmp_path, audit_evidence, monkeypatch):
    summary, records = audit_evidence
    source, raw, output = (tmp_path / name for name in ("summary.json", "raw.jsonl", "audit.json"))
    source.write_text(json.dumps(summary))
    raw.write_text("\n".join(map(json.dumps, records)))
    monkeypatch.setattr(frequency_probe.socket, "socket", lambda *a: pytest.fail("opened CAN"))
    monkeypatch.setattr(frequency_probe, "B601RSBackend", lambda *a: pytest.fail("opened backend"))
    args = ["--mode", "audit-paired", "--input-summary", str(source), "--input-raw", str(raw), "--output", str(output)]
    assert frequency_probe.main(args) == 0
    result = json.loads(output.read_text())
    assert result["matched_read_count"] == 6 and result["input_sha256"]["raw"]
    assert frequency_probe.main(args) == 2


@pytest.fixture
def paired_rig(monkeypatch):
    rig = SimpleNamespace(now=0.0, calls=[], closed=[], fail=None, rx_error=False, nonfinite=False)

    class Motor:
        def __init__(self, index):
            self.index = index

        def robstride_get_param_f32(self, param, timeout_ms):
            assert param == 0x7019 and timeout_ms == 500
            rig.calls.append(self.index)
            rig.now += 0.5
            if rig.fail is not None and self.index == 2:
                raise rig.fail
            return math.nan if rig.nonfinite else self.index * 0.01

        def close(self):
            rig.closed.append(self.index)

    class Controller:
        def __init__(self, channel):
            assert channel == "can-test"

        def add_robstride_motor(self, index, host, model):
            assert host == 0xFD
            assert model == ("rs-06" if index <= 3 else "rs-00")
            return Motor(index)

        def close(self):
            rig.closed.append("controller")

    def capture(channel, duration, *, raw_output, ready, stop):
        if rig.rx_error:
            raise OSError("RX bind failed")
        ready.set()
        assert stop.wait(5.0)
        return {"raw_capture": {"completed": True}, "freshness_verified": False}

    monkeypatch.setitem(sys.modules, "motorbridge", SimpleNamespace(Controller=Controller))
    monkeypatch.setattr(frequency_probe, "_read_only_capture", capture)
    monkeypatch.setattr(frequency_probe.time, "monotonic", lambda: rig.now)
    monkeypatch.setattr(frequency_probe.time, "monotonic_ns", lambda: int(rig.now * 1e9))
    return rig


def test_paired_capture_brackets_all_six_reads_and_joins(tmp_path, paired_rig):
    result = frequency_probe._paired_capture("can-test", 3.0, tmp_path / "raw.jsonl")
    assert result["status"] == "OK"
    assert paired_rig.calls == list(range(1, 7))
    assert paired_rig.closed == [6, 5, 4, 3, 2, 1, "controller"]
    assert not result["motion_commands_sent"]
    assert not result["freshness_verified"]
    for sample in result["parameter_samples"]:
        assert sample["request_end_monotonic_ns"] > sample["request_start_monotonic_ns"]


@pytest.mark.parametrize("failure,status", [(RuntimeError("read failed"), "ERROR"), (KeyboardInterrupt(), "INTERRUPTED")])
def test_paired_capture_retains_partial_samples_and_closes(tmp_path, paired_rig, failure, status):
    paired_rig.fail = failure
    result = frequency_probe._paired_capture("can-test", 3.0, tmp_path / "raw.jsonl")
    assert result["status"] == status
    assert len(result["parameter_samples"]) == 2
    assert result["parameter_samples"][0]["completed"]
    assert not result["parameter_samples"][1]["completed"]
    assert paired_rig.closed[-1] == "controller"
    assert paired_rig.calls == [1, 2]


def test_paired_capture_rx_failure_prevents_sdk_reads(tmp_path, paired_rig):
    paired_rig.rx_error = True
    result = frequency_probe._paired_capture("can-test", 3.0, tmp_path / "raw.jsonl")
    assert result["status"] == "ERROR"
    assert paired_rig.calls == []


def test_paired_capture_rejects_nonfinite_and_stops_reads(tmp_path, paired_rig):
    paired_rig.nonfinite = True
    result = frequency_probe._paired_capture("can-test", 3.0, tmp_path / "raw")
    assert result["status"] == "ERROR"
    assert paired_rig.calls == [1]
    assert not result["parameter_samples"][0]["completed"]


def test_paired_cli_preserves_existing_summary(tmp_path, monkeypatch):
    path = tmp_path / "summary"
    path.write_text("previous evidence")
    monkeypatch.setattr(frequency_probe, "_paired_capture", lambda *a: pytest.fail("unexpected IO"))
    assert frequency_probe.main([
        "--mode", "read-paired", "--raw-output", str(tmp_path / "raw"),
        "--output", str(path), "--confirmation", "B601_RS_PAIRED_STATIC_READ_ONLY",
    ]) == 2
    assert path.read_text() == "previous evidence"


@pytest.mark.parametrize("flag", ["--enable", "--allow-motion", "--auto-disable"])
def test_paired_cli_rejects_motion_flags_before_io(tmp_path, monkeypatch, flag):
    monkeypatch.setattr(frequency_probe, "_paired_capture", lambda *a: pytest.fail("unexpected IO"))
    assert frequency_probe.main([
        "--mode", "read-paired", "--raw-output", str(tmp_path / "raw"),
        "--output", str(tmp_path / "summary"),
        "--confirmation", "B601_RS_PAIRED_STATIC_READ_ONLY", flag,
    ]) == 2


def test_paired_cli_requires_confirmation_before_io(tmp_path, monkeypatch):
    monkeypatch.setattr(frequency_probe, "_paired_capture", lambda *a: pytest.fail("unexpected IO"))
    assert frequency_probe.main(["--mode", "read-paired"]) == 2


def test_paired_cli_writes_interrupted_partial_result(tmp_path, paired_rig):
    paired_rig.fail = KeyboardInterrupt()
    path = tmp_path / "summary.json"
    assert frequency_probe.main([
        "--mode", "read-paired", "--channel", "can-test",
        "--raw-output", str(tmp_path / "raw.jsonl"), "--output", str(path),
        "--confirmation", "B601_RS_PAIRED_STATIC_READ_ONLY",
    ]) == 130
    assert json.loads(path.read_text())["status"] == "INTERRUPTED"


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


def _config(
    *,
    allow_motion: bool = False,
    allow_active_reporting: bool = False,
) -> B601RSBackendConfig:
    return B601RSBackendConfig(
        channel="can0",
        rate_hz=500.0,
        joints=_joint_configs(),
        allow_motion=allow_motion,
        allow_active_reporting=allow_active_reporting,
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
        self.observation: object | None = None
        self.position_param: float | None = None
        self.parameter_values: dict[int, float] = {}
        self.write_without_effect: set[int] = set()
        self.fail_write_values: set[float] = set()
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.fail_method: str | None = None
        self.active_report_call_count = 0
        self.fail_active_report_calls: set[int] = set()

    def _call(self, method: str, *args: object) -> None:
        self.calls.append((method, args))
        if self.fail_method == method:
            raise RuntimeError(f"synthetic {method} failure")

    def get_state(self) -> object | None:
        self._call("get_state")
        return self.state

    def robstride_get_state_observation(self) -> object | None:
        self._call("robstride_get_state_observation")
        return self.observation

    def robstride_get_active_report_observation(self) -> object | None:
        self._call("robstride_get_active_report_observation")
        return self.observation

    def request_feedback(self) -> None:
        self._call("request_feedback")

    def robstride_get_param_f32(self, *args: object) -> float:
        self._call("robstride_get_param_f32", *args)
        param_id = int(args[0])
        if param_id in self.parameter_values:
            return self.parameter_values[param_id]
        if self.position_param is None:
            raise RuntimeError("synthetic position parameter unavailable")
        return self.position_param

    def robstride_write_param_f32(self, *args: object) -> None:
        self._call("robstride_write_param_f32", *args)
        param_id = int(args[0])
        value = float(args[1])
        if value in self.fail_write_values:
            raise RuntimeError(f"synthetic write failure for value {value}")
        if param_id not in self.write_without_effect:
            self.parameter_values[param_id] = value

    def ensure_mode(self, *args: object) -> None:
        self._call("ensure_mode", *args)

    def enable(self) -> None:
        self._call("enable")

    def disable(self) -> None:
        self._call("disable")

    def robstride_set_active_report(self, enabled: bool) -> None:
        self.active_report_call_count += 1
        self.calls.append(("robstride_set_active_report", (enabled,)))
        if self.active_report_call_count in self.fail_active_report_calls:
            raise RuntimeError("synthetic active-report failure")

    def send_pos_vel(self, *args: object) -> None:
        self._call("send_pos_vel", *args)

    def send_mit(self, *args: object) -> None:
        self._call("send_mit", *args)


class _FakeController:
    def __init__(self) -> None:
        self.motors: dict[int, _FakeMotor] = {}
        self.calls: list[str] = []
        self.robstride_capabilities = [
            "parameter_reply_correlation_v1",
            "state_observation_metadata_v1",
        ]

    def motorbridge_abi_capabilities(self) -> dict[str, object]:
        return {
            "schema": 1,
            "abi": {"name": "motor_abi", "version": "test"},
            "features": {"robstride": list(self.robstride_capabilities)},
        }

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


@pytest.fixture
def passive_receiver(monkeypatch):
    class Receiver:
        def __init__(self):
            self.now = 0.0
            self.packets = []
            self.closed = False

        def bind(self, address):
            assert address == ("can-test",)

        def settimeout(self, value):
            assert value > 0

        def recv(self, size):
            self.now += 0.1
            if self.packets:
                packet = self.packets.pop(0)
                if isinstance(packet, BaseException):
                    raise packet
                return packet
            raise frequency_probe.socket.timeout()

        def close(self):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    receiver = Receiver()
    receiver.packets = []
    monkeypatch.setattr(frequency_probe.socket, "socket", lambda *args: receiver)
    monkeypatch.setattr(frequency_probe.time, "monotonic", lambda: receiver.now)
    monkeypatch.setattr(frequency_probe.time, "monotonic_ns", lambda: int(receiver.now * 1e9))
    monkeypatch.setattr(frequency_probe.time, "time_ns", lambda: 1_000_000_000 + int(receiver.now * 1e9))

    def no_backend(*args, **kwargs):
        pytest.fail("passive capture must not construct the motor backend")

    monkeypatch.setattr(frequency_probe, "B601RSBackend", no_backend)
    return receiver


def test_passive_raw_capture_keeps_bytes_clocks_and_malformed_frames(tmp_path, passive_receiver):
    packet = _socketcan_frame((2 << 24) | (3 << 8) | 0xFD | frequency_probe._CAN_EFF_FLAG)
    passive_receiver.packets = [packet, b"bad"]
    path = tmp_path / "capture.jsonl"
    result = frequency_probe._read_only_capture("can-test", 0.5, raw_output=path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["kind"] for r in records] == ["header", "frame", "frame", "end"]
    assert records[1]["socket_frame_hex"] == packet.hex()
    assert records[2]["socket_frame_hex"] == b"bad".hex()
    assert records[1]["receive_monotonic_ns"] <= records[1]["clock_pair_end_monotonic_ns"]
    assert records[1]["observe_realtime_ns"] == records[1]["receive_monotonic_ns"] + 1_000_000_000
    assert records[-1]["received_frame_count"] == 2
    assert result["malformed_frame_count"] == 1
    assert not result["freshness_verified"]
    assert not result["motion_commands_sent"]
    assert passive_receiver.closed


def test_passive_raw_capture_never_overwrites_existing_evidence(tmp_path, passive_receiver):
    path = tmp_path / "capture.jsonl"
    path.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        frequency_probe._read_only_capture("can-test", 0.5, raw_output=path)
    assert path.read_text() == "existing evidence"
    assert passive_receiver.closed


def test_interrupted_raw_capture_has_no_success_footer(tmp_path, passive_receiver):
    passive_receiver.packets = [KeyboardInterrupt()]
    path = tmp_path / "capture.jsonl"
    with pytest.raises(KeyboardInterrupt):
        frequency_probe._read_only_capture("can-test", 0.5, raw_output=path)
    assert [json.loads(x)["kind"] for x in path.read_text().splitlines()] == ["header"]
    assert passive_receiver.closed


def test_empty_raw_capture_is_not_promoted_to_feedback(tmp_path, passive_receiver):
    result = frequency_probe._read_only_capture("can-test", 0.5, raw_output=tmp_path / "empty.jsonl")
    assert result["frame_count"] == 0
    assert not result["freshness_verified"]


@pytest.mark.parametrize("mode", ["offline", "gravity-offline", "position", "mit-tx-disabled"])
def test_raw_output_rejects_other_modes_before_io(mode, tmp_path, monkeypatch):
    monkeypatch.setattr(frequency_probe.socket, "socket", lambda *args: pytest.fail("no I/O expected"))
    assert frequency_probe.main(["--mode", mode, "--raw-output", str(tmp_path / "raw")]) == 2


def test_raw_and_summary_output_cannot_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(frequency_probe.socket, "socket", lambda *args: pytest.fail("no I/O expected"))
    path = str(tmp_path / "capture")
    assert frequency_probe.main(["--raw-output", path, "--output", path]) == 2


def test_passive_bind_error_closes_socket_and_does_not_create_raw_file(tmp_path, passive_receiver, monkeypatch):
    def fail_bind(address):
        raise OSError("interface unavailable")

    monkeypatch.setattr(passive_receiver, "bind", fail_bind)
    path = tmp_path / "raw.jsonl"
    with pytest.raises(frequency_probe.FrequencyProbeError, match="cannot open passive"):
        frequency_probe._read_only_capture("can-test", 0.5, raw_output=path)
    assert passive_receiver.closed
    assert not path.exists()


def test_passive_cli_writes_summary_and_raw_separately(tmp_path, passive_receiver):
    path = tmp_path / "raw.jsonl"
    summary = tmp_path / "summary.json"
    assert frequency_probe.main([
        "--mode", "read-only", "--channel", "can-test", "--duration-sec", "0.5",
        "--raw-output", str(path), "--output", str(summary),
    ]) == 0
    result = json.loads(summary.read_text())
    assert result["raw_capture"]["completed"]
    assert result["frame_count"] == 0
    assert not result["motion_commands_sent"]


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
    assert config.allow_active_reporting is False

    telemetry_config = B601RSBackendConfig.from_mapping(
        _yaml_mapping(),
        allow_active_reporting=True,
    )
    assert telemetry_config.allow_active_reporting is True
    assert telemetry_config.allow_motion is False


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


def test_motion_connect_rejects_runtime_without_strict_parameter_correlation() -> None:
    controller = _FakeController()
    controller.robstride_capabilities.clear()
    backend = B601RSBackend(
        _config(allow_motion=True),
        controller_factory=lambda _channel: controller,
    )

    with pytest.raises(
        B601HardwareDependencyError,
        match="strict RobStride type-17 parameter-reply correlation",
    ):
        backend.connect()

    assert backend.connected is False
    assert backend.motorbridge_runtime_evidence is None
    assert controller.calls == ["close"]
    assert controller.motors == {}


def test_default_motion_connect_checks_capability_before_opening_transport(
    monkeypatch,
) -> None:
    opened_channels: list[str] = []

    class Controller:
        def __init__(self, channel: str) -> None:
            opened_channels.append(channel)

    monkeypatch.setitem(
        sys.modules,
        "motorbridge",
        SimpleNamespace(
            Controller=Controller,
            abi_capabilities=lambda: {
                "schema": 1,
                "abi": {"name": "motor_abi", "version": "0.5.3"},
                "features": {"robstride": ["param_f32_host_id"]},
            },
        ),
    )
    backend = B601RSBackend(_config(allow_motion=True))

    with pytest.raises(B601HardwareDependencyError, match="abi_version='0.5.3'"):
        backend.connect()

    assert opened_channels == []
    assert backend.connected is False


def test_read_only_connect_remains_available_without_strict_parameter_correlation() -> None:
    controller = _FakeController()
    controller.robstride_capabilities.clear()
    backend = B601RSBackend(
        _config(),
        controller_factory=lambda _channel: controller,
    )

    backend.connect()
    _populate_states(controller)
    state = backend.poll_feedback()

    assert state.valid is True
    assert backend.parameter_reply_correlation_verified is False
    assert backend.motorbridge_runtime_evidence == {
        "verified": False,
        "abi_version": "test",
        "robstride_features": (),
        "error": (
            "MotorBridge does not advertise required RobStride capability "
            "'parameter_reply_correlation_v1'"
        ),
    }
    assert backend.safe_stop(joint_names=("joint6",)).completed is True


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


def test_host_fresh_poll_uses_operation_status_sequence_and_receive_age() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(),
        controller_factory=lambda _channel: controller,
        clock=lambda: 10.0,
    )

    backend.connect()
    _populate_states(controller)
    for index, motor in controller.motors.items():
        motor.observation = SimpleNamespace(
            state=motor.state,
            status_sequence=20 + index,
            receive_age_ns=5_000_000,
        )

    state = backend.poll_feedback(require_host_freshness=True)

    assert state.valid is True
    assert state.stale is False
    assert state.freshness_verified is False
    assert state.host_receive_freshness_verified is True
    assert state.freshness_semantics == (
        "host_receive_sequence_and_age_verified; "
        "device_source_timestamp_unavailable"
    )
    assert state.velocity_semantics == (
        "robstride_operation_status_host_receive_verified"
    )
    assert state.positions_rad == pytest.approx(
        tuple(0.1 * index for index in range(1, 7))
    )
    for index, joint in enumerate(state.joints, start=1):
        assert joint.status_sequence == 20 + index
        assert joint.receive_age_sec == pytest.approx(0.005)
        assert joint.observed_at_sec == pytest.approx(9.995)
        assert joint.host_receive_freshness_verified is True
        assert joint.position_source == "robstride_operation_status_cache"
    assert all(
        motor.calls == [("robstride_get_state_observation", ())]
        for motor in controller.motors.values()
    )


@pytest.mark.parametrize("fault", [None, "empty", "stale", "getter_error", "getter_missing", "wrong_type"])
def test_type24_cache_advertisement_selects_only_dedicated_stream_and_never_falls_back(fault):
    from hardware.rebot_b601_rs.active_report import B601ActiveReportControlGate

    controller = _FakeController()
    controller.robstride_capabilities.append("active_report_observation_v1")
    backend = B601RSBackend(_config(), controller_factory=lambda _: controller, clock=lambda: 10.)
    backend.connect()
    _populate_states(controller)
    for index, motor in controller.motors.items():
        motor.observation = SimpleNamespace(state=motor.state, status_sequence=20+index,
                                            receive_age_ns=5_000_000)
        # Generic Type-2 replies remain available but cannot rescue this poll.
        motor.state = SimpleNamespace(**{**vars(motor.state), "arbitration_id": 0x02000000+index})
        if index == 2:
            if fault == "empty": motor.observation = None
            elif fault == "stale": motor.observation.receive_age_ns = 200_000_000
            elif fault == "getter_error": motor.fail_method = "robstride_get_active_report_observation"
            elif fault == "getter_missing": motor.robstride_get_active_report_observation = None
            elif fault == "wrong_type": motor.observation.state = motor.state
    state = backend.poll_feedback(require_host_freshness=True, prefer_active_report=True)
    gate = B601ActiveReportControlGate(B601_RS_ARM_JOINT_NAMES, .1)
    if fault:
        with pytest.raises(ValueError):
            gate.observe(state)
        assert not gate.baseline_established
    else:
        assert gate.observe(state) is False
        # Polling fresh Type-2 replies cannot advance a static Type-24 sequence.
        with pytest.raises(ValueError, match="advance"):
            gate.observe(backend.poll_feedback(require_host_freshness=True, prefer_active_report=True))
        for motor in controller.motors.values():
            motor.observation.status_sequence += 10
        assert gate.observe(backend.poll_feedback(require_host_freshness=True, prefer_active_report=True))
        assert state.joints[1].receive_age_sec == pytest.approx(.005)
    assert all(name != "robstride_get_state_observation"
               for motor in controller.motors.values() for name, _ in motor.calls)
    assert all(name not in {"request_feedback", "get_state", "enable", "send_mit"}
               for motor in controller.motors.values() for name, _ in motor.calls)


def test_host_fresh_generic_contract_does_not_change_when_type24_cache_is_available():
    controller = _FakeController()
    controller.robstride_capabilities.append("active_report_observation_v1")
    backend = B601RSBackend(_config(), controller_factory=lambda _: controller)
    backend.connect()
    _populate_states(controller)
    for index, motor in controller.motors.items():
        motor.observation = SimpleNamespace(state=motor.state, status_sequence=index,
                                            receive_age_ns=1_000_000)
        motor.fail_method = "robstride_get_active_report_observation"
    assert backend.poll_feedback(require_host_freshness=True).valid
    assert all(motor.calls == [("robstride_get_state_observation", ())]
               for motor in controller.motors.values())


@pytest.mark.parametrize("preference,error", [(True, ValueError), (1, TypeError), ("yes", TypeError)])
def test_active_report_cache_preference_requires_explicit_freshness(preference, error):
    backend = B601RSBackend(_config())
    with pytest.raises(error, match="prefer_active_report"):
        backend.poll_feedback(prefer_active_report=preference)
    assert not backend.connected


def test_host_fresh_poll_rejects_operation_status_older_than_timeout() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(),
        controller_factory=lambda _channel: controller,
        clock=lambda: 10.0,
    )

    backend.connect()
    _populate_states(controller)
    for index, motor in controller.motors.items():
        motor.observation = SimpleNamespace(
            state=motor.state,
            status_sequence=index,
            receive_age_ns=(200_000_000 if index == 6 else 5_000_000),
        )

    state = backend.poll_feedback(require_host_freshness=True)

    assert state.valid is False
    assert state.stale is True
    assert state.host_receive_freshness_verified is False
    assert state.joint("joint6").valid is False
    assert state.joint("joint6").host_receive_freshness_verified is False
    assert "receive age 0.200000s exceeds feedback timeout 0.100000s" in (
        state.fault_reason or ""
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("status_sequence", 1.5, "status_sequence must be a positive integer"),
        ("receive_age_ns", -1, "receive_age_ns must be non-negative"),
    ),
)
def test_host_fresh_poll_rejects_malformed_observation_metadata(
    field: str,
    value: object,
    message: str,
) -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(),
        controller_factory=lambda _channel: controller,
        clock=lambda: 10.0,
    )

    backend.connect()
    _populate_states(controller)
    for index, motor in controller.motors.items():
        fields = {
            "state": motor.state,
            "status_sequence": index,
            "receive_age_ns": 5_000_000,
        }
        if index == 6:
            fields[field] = value
        motor.observation = SimpleNamespace(**fields)

    state = backend.poll_feedback(require_host_freshness=True)

    assert state.valid is False
    assert state.stale is True
    assert state.host_receive_freshness_verified is False
    assert state.joint("joint6").valid is False
    assert message in (state.fault_reason or "")


@pytest.mark.parametrize("value", (1.0, 1.5, "1", "1.5", True, 0, -1))
def test_joint_state_rejects_invalid_status_sequence(value: object) -> None:
    with pytest.raises(ValueError, match="status_sequence must be a positive integer"):
        B601JointState(
            name="joint1",
            motor_id=1,
            feedback_id=0xFD,
            arbitration_id=None,
            status_code=0,
            position_rad=0.0,
            velocity_reported=0.0,
            torque_reported_nm=0.0,
            observed_at_sec=0.0,
            sample_sequence=0,
            valid=True,
            stale=False,
            status_sequence=value,
        )


def test_joint_state_preserves_positive_integer_status_sequence() -> None:
    state = B601JointState(
        name="joint1",
        motor_id=1,
        feedback_id=0xFD,
        arbitration_id=None,
        status_code=0,
        position_rad=0.0,
        velocity_reported=0.0,
        torque_reported_nm=0.0,
        observed_at_sec=0.0,
        sample_sequence=0,
        valid=True,
        stale=False,
        status_sequence=7,
    )

    assert state.status_sequence == 7
    assert type(state.status_sequence) is int


def test_host_fresh_poll_requires_advertised_motorbridge_capability() -> None:
    controller = _FakeController()
    controller.robstride_capabilities.remove("state_observation_metadata_v1")
    backend = B601RSBackend(_config(), controller_factory=lambda _channel: controller)
    backend.connect()

    with pytest.raises(
        B601HardwareDependencyError,
        match="state_observation_metadata_v1",
    ):
        backend.poll_feedback(require_host_freshness=True)

    assert controller.calls == [f"add:{index}" for index in range(1, 7)]
    assert all(not motor.calls for motor in controller.motors.values())


def test_feedback_paths_cannot_be_mixed_in_one_poll() -> None:
    controller = _FakeController()
    backend = B601RSBackend(_config(), controller_factory=lambda _channel: controller)
    backend.connect()

    with pytest.raises(ValueError, match="separate feedback paths"):
        backend.poll_feedback(
            request_feedback=True,
            require_host_freshness=True,
        )

    assert controller.calls == [f"add:{index}" for index in range(1, 7)]


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


def test_active_feedback_prefers_host_qualified_mech_pos_when_available() -> None:
    controller = _FakeController()
    backend = B601RSBackend(_config(), controller_factory=lambda _channel: controller)

    backend.connect()
    _populate_states(controller)
    motor = controller.motors[4]

    def host_read(param_id: int, host_id: int, timeout_ms: int) -> float:
        motor._call("robstride_get_param_f32_host_id", param_id, host_id, timeout_ms)
        assert param_id == B601_RS_MECH_POS_PARAM_ID
        assert host_id == 0xFD
        assert timeout_ms == 100
        return float(motor.position_param)

    motor.robstride_get_param_f32_host_id = host_read  # type: ignore[attr-defined]

    state = backend.poll_feedback(request_feedback=True)

    assert state.valid is True
    assert state.joint("joint4").position_source == (
        "robstride_mech_pos_parameter_host_id"
    )
    assert (
        "robstride_get_param_f32_host_id",
        (B601_RS_MECH_POS_PARAM_ID, 0xFD, 100),
    ) in motor.calls


def test_active_feedback_rejects_finite_mech_pos_outside_read_admission_envelope() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        B601RSBackendConfig(
            channel="can0",
            rate_hz=500.0,
            joints=_joint_configs(),
            position_limits_rad=((-3.0, 3.0),) * 6,
            feedback_position_limits_rad=((-3.0, 3.0),) * 6,
        ),
        controller_factory=lambda _channel: controller,
    )

    backend.connect()
    _populate_states(controller)
    corrupted_position = 2.686622237168953e36
    controller.motors[4].position_param = corrupted_position

    state = backend.poll_feedback(request_feedback=True)

    assert state.valid is False
    assert state.stale is True
    assert state.positions_rad is None
    joint = state.joint("joint4")
    assert joint.valid is False
    assert joint.stale is True
    assert joint.position_rad == corrupted_position
    assert joint.position_source == "robstride_mech_pos_parameter"
    assert f"mechPos(0x{B601_RS_MECH_POS_PARAM_ID:04x})" in (joint.error or "")
    assert repr(corrupted_position) in (joint.error or "")
    assert "raw_f32_bits=0x7c015b2d" in (joint.error or "")
    assert "motor_id=4" in (joint.error or "")
    assert "outside feedback acceptance limits" in (joint.error or "")


def test_feedback_admission_window_does_not_relax_outgoing_position_limits() -> None:
    controller = _FakeController()
    strict_limits = (
        (-3.0, 3.0),
        (0.0, 3.14),
        (0.0, 3.14),
        (-math.pi / 2.0, math.pi / 2.0),
        (-1.57, 1.57),
        (-3.14, 3.14),
    )
    feedback_limits = (
        (-3.0, 3.0),
        (-0.01, 3.14),
        (-0.003, 3.14),
        (-math.pi / 2.0 - 0.001, math.pi / 2.0),
        (-1.57, 1.57),
        (-3.14, 3.14),
    )
    backend = B601RSBackend(
        B601RSBackendConfig(
            channel="can0",
            rate_hz=500.0,
            joints=_joint_configs(),
            allow_motion=True,
            position_limits_rad=strict_limits,
            feedback_position_limits_rad=feedback_limits,
        ),
        controller_factory=lambda _channel: controller,
    )

    backend.connect()
    _populate_states(controller)
    controller.motors[2].position_param = -0.004
    controller.motors[3].position_param = -0.002
    controller.motors[4].position_param = -math.pi / 2.0 - 0.0005

    state = backend.poll_feedback(request_feedback=True)

    assert state.valid is True
    assert state.positions_rad is not None
    assert state.positions_rad[1:4] == pytest.approx(
        (-0.004, -0.002, -math.pi / 2.0 - 0.0005)
    )
    with pytest.raises(ValueError, match=r"qpos_rad\[1\].*joint2"):
        backend.send_position_velocity(state.positions_rad)


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


def test_active_reporting_requires_separate_authorization() -> None:
    controller = _FakeController()
    backend = B601RSBackend(_config(), controller_factory=lambda _channel: controller)
    backend.connect()

    with pytest.raises(B601ActiveReportAuthorizationError):
        backend.set_active_reporting(True, joint_names=("joint1",))

    assert controller.motors[1].calls == []


def test_active_reporting_round_trip_never_calls_motion_methods() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(allow_active_reporting=True),
        controller_factory=lambda _channel: controller,
    )
    backend.connect()

    enabled = backend.set_active_reporting(
        True,
        joint_names=("joint1", "joint2"),
    )
    assert enabled.completed is True
    assert backend.active_report_joint_names == ("joint1", "joint2")
    disabled = backend.set_active_reporting(
        False,
        joint_names=("joint1", "joint2"),
    )

    assert disabled.completed is True
    assert backend.active_report_joint_names == ()
    for motor_id in (1, 2):
        methods = [method for method, _args in controller.motors[motor_id].calls]
        assert methods == [
            "robstride_set_active_report",
            "robstride_set_active_report",
        ]
    backend.close()


def test_partial_active_report_enable_rolls_back_every_attempted_joint() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(allow_active_reporting=True),
        controller_factory=lambda _channel: controller,
    )
    backend.connect()
    controller.motors[2].fail_active_report_calls.add(1)

    receipt = backend.set_active_reporting(
        True,
        joint_names=("joint1", "joint2"),
    )

    assert receipt.completed is False
    assert receipt.rollback_attempted is True
    assert receipt.rollback_completed is True
    assert receipt.failed_joint_names == ("joint2",)
    assert backend.active_report_joint_names == ()
    assert controller.motors[1].calls == [
        ("robstride_set_active_report", (True,)),
        ("robstride_set_active_report", (False,)),
    ]
    assert controller.motors[2].calls == [
        ("robstride_set_active_report", (True,)),
        ("robstride_set_active_report", (False,)),
    ]
    backend.close()


def test_failed_active_report_rollback_stays_latched_and_blocks_close() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(allow_active_reporting=True),
        controller_factory=lambda _channel: controller,
    )
    backend.connect()
    controller.motors[1].fail_active_report_calls.add(2)
    controller.motors[2].fail_active_report_calls.add(1)

    receipt = backend.set_active_reporting(
        True,
        joint_names=("joint1", "joint2"),
    )

    assert receipt.rollback_completed is False
    assert receipt.failed_joint_names == ("joint2", "joint1")
    assert backend.active_report_joint_names == ("joint1",)
    with pytest.raises(B601HardwareError, match="active reporting must be disabled"):
        backend.close()

    cleanup = backend.set_active_reporting(False, joint_names=("joint1",))
    assert cleanup.completed is True
    backend.close()


@pytest.mark.parametrize("pending", ["active_report", "torque_limit", "enabled"])
def test_terminal_transport_release_preserves_unconfirmed_device_state_and_sends_nothing(pending):
    controller = _FakeController()
    backend = B601RSBackend(_config(allow_motion=True, allow_active_reporting=True),
                            controller_factory=lambda _channel: controller)
    backend.connect()
    if pending == "active_report":
        assert backend.set_active_reporting(True, joint_names=("joint1",)).completed
    elif pending == "torque_limit":
        controller.motors[1].parameter_values[B601_RS_TORQUE_LIMIT_PARAM_ID] = 36.
        assert backend.safe_stop(joint_names=("joint1",)).completed
        assert backend.apply_temporary_driver_torque_limits((.5,)*6, joint_names=("joint1",)).completed
    else:
        assert backend.enable(joint_names=("joint1",)).completed
    before = {i: list(motor.calls) for i, motor in controller.motors.items()}
    backend.release_transport()
    backend.release_transport()
    assert not backend.connected
    assert controller.calls.count("close") == 1
    assert {i: motor.calls for i, motor in controller.motors.items()} == before
    assert backend.active_report_joint_names == (("joint1",) if pending == "active_report" else ())
    assert backend.temporary_driver_torque_limit_joint_names == (("joint1",) if pending == "torque_limit" else ())
    assert backend.enabled_joint_names == (("joint1",) if pending == "enabled" else ())
    with pytest.raises(B601HardwareError, match="unconfirmed device state"):
        backend.connect()


@pytest.mark.parametrize("method", ["close", "release_transport"])
def test_transport_close_failure_keeps_the_original_handle_until_explicit_retry(method):
    controller = _FakeController()
    backend = B601RSBackend(_config(), controller_factory=lambda _channel: controller)
    backend.connect()
    original = controller.close
    def fail():
        raise OSError("synthetic host release failure")
    controller.close = fail
    with pytest.raises(OSError, match="host release failure"):
        getattr(backend, method)()
    assert backend.connected
    assert backend.registered_joint_names == B601_RS_ARM_JOINT_NAMES
    controller.close = original
    getattr(backend, method)()
    assert not backend.connected
    assert controller.calls.count("close") == 1


def test_incomplete_vendor_release_cannot_clear_the_handle():
    controller = _FakeController()
    backend = B601RSBackend(_config(), controller_factory=lambda _channel: controller)
    backend.connect()
    controller.close = lambda: False
    with pytest.raises(B601HardwareError, match="release is incomplete"):
        backend.release_transport()
    assert backend.connected


@pytest.mark.parametrize("failure", [None, "bus", "motor", "controller"])
def test_native_bus_and_motor_handles_are_released_explicitly_with_retry(failure):
    events, failed = [], [failure]
    class Controller(_FakeController):
        def close_bus(self):
            events.append("bus")
            if failed[0] == "bus":
                raise OSError("synthetic bus close failure")
        def add_robstride_motor(self, motor_id, *args):
            motor = super().add_robstride_motor(motor_id, *args)
            def close():
                events.append(f"motor:{motor_id}")
                if failed[0] == "motor" and motor_id == 3:
                    raise OSError("synthetic motor handle close failure")
            motor.close = close
            return motor
        def close(self):
            events.append("controller")
            if failed[0] == "controller":
                raise OSError("synthetic controller close failure")
    controller = Controller()
    backend = B601RSBackend(_config(allow_active_reporting=True),
                            controller_factory=lambda _channel: controller)
    backend.connect()
    assert backend.set_active_reporting(True, joint_names=("joint1",)).completed
    before = {i: list(m.calls) for i, m in controller.motors.items()}
    if failure:
        with pytest.raises((OSError, B601HardwareError), match="close failure"):
            backend.release_transport()
        assert backend.connected
        assert backend.active_report_joint_names == ("joint1",)
        failed[0] = None
    backend.release_transport()
    assert not backend.connected
    assert events[0] == "bus" and events[-1] == "controller"
    assert all(events.count(f"motor:{i}") == (2 if failure == "motor" and i == 3 else 1)
               for i in range(1, 7))
    assert {i: m.calls for i, m in controller.motors.items()} == before
    assert backend.active_report_joint_names == ("joint1",)


def test_temporary_driver_torque_limit_requires_motion_authorization() -> None:
    controller = _FakeController()
    backend = B601RSBackend(_config(), controller_factory=lambda _channel: controller)
    backend.connect()
    controller.motors[6].parameter_values[B601_RS_TORQUE_LIMIT_PARAM_ID] = 14.0

    with pytest.raises(B601MotionAuthorizationError):
        backend.apply_temporary_driver_torque_limits(
            (0.5,) * 6,
            joint_names=("joint6",),
        )

    assert controller.motors[6].calls == []


def test_temporary_driver_torque_limit_round_trip_requires_disable_before_restore() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(allow_motion=True),
        controller_factory=lambda _channel: controller,
    )
    backend.connect()
    motor = controller.motors[6]
    motor.parameter_values[B601_RS_TORQUE_LIMIT_PARAM_ID] = 14.0
    assert backend.safe_stop(joint_names=("joint6",)).completed is True

    apply_receipt = backend.apply_temporary_driver_torque_limits(
        (0.5,) * 6,
        joint_names=("joint6",),
    )

    assert apply_receipt.completed is True
    assert apply_receipt.target_limits_nm == (0.5,)
    assert apply_receipt.prior_limits_nm == (14.0,)
    assert apply_receipt.verified_limits_nm == (0.5,)
    assert apply_receipt.applied_joint_names == ("joint6",)
    assert backend.temporary_driver_torque_limit_joint_names == ("joint6",)
    assert motor.parameter_values[B601_RS_TORQUE_LIMIT_PARAM_ID] == 0.5

    assert backend.enable(joint_names=("joint6",)).completed is True
    assert backend.enabled_joint_names == ("joint6",)
    with pytest.raises(B601HardwareError, match="only after disable"):
        backend.restore_driver_torque_limits()

    assert backend.safe_stop(joint_names=("joint6",)).completed is True
    restore_receipt = backend.restore_driver_torque_limits()

    assert restore_receipt.completed is True
    assert restore_receipt.target_limits_nm == (14.0,)
    assert restore_receipt.prior_limits_nm == (0.5,)
    assert restore_receipt.verified_limits_nm == (14.0,)
    assert restore_receipt.restored_joint_names == ("joint6",)
    assert backend.temporary_driver_torque_limit_joint_names == ()
    backend.close()


def test_temporary_driver_torque_limit_readback_mismatch_rolls_back_every_touched_joint() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(allow_motion=True),
        controller_factory=lambda _channel: controller,
    )
    backend.connect()
    for motor_id in (5, 6):
        controller.motors[motor_id].parameter_values[
            B601_RS_TORQUE_LIMIT_PARAM_ID
        ] = 14.0
    controller.motors[6].write_without_effect.add(B601_RS_TORQUE_LIMIT_PARAM_ID)
    assert backend.safe_stop(joint_names=("joint5", "joint6")).completed is True

    receipt = backend.apply_temporary_driver_torque_limits(
        (0.5,) * 6,
        joint_names=("joint5", "joint6"),
    )

    assert receipt.completed is False
    assert receipt.applied_joint_names == ("joint5",)
    assert receipt.failed_joint_names == ("joint6",)
    assert receipt.rollback_attempted is True
    assert receipt.rollback_completed is True
    assert set(receipt.restored_joint_names) == {"joint5", "joint6"}
    assert backend.temporary_driver_torque_limit_joint_names == ()
    assert all(
        controller.motors[motor_id].parameter_values[
            B601_RS_TORQUE_LIMIT_PARAM_ID
        ]
        == 14.0
        for motor_id in (5, 6)
    )


def test_failed_driver_torque_limit_restore_stays_latched_and_blocks_close() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(allow_motion=True),
        controller_factory=lambda _channel: controller,
    )
    backend.connect()
    motor = controller.motors[6]
    motor.parameter_values[B601_RS_TORQUE_LIMIT_PARAM_ID] = 14.0
    assert backend.safe_stop(joint_names=("joint6",)).completed is True
    assert backend.apply_temporary_driver_torque_limits(
        (0.5,) * 6,
        joint_names=("joint6",),
    ).completed
    motor.fail_method = "robstride_write_param_f32"

    failed_restore = backend.restore_driver_torque_limits()

    assert failed_restore.completed is False
    assert failed_restore.failed_joint_names == ("joint6",)
    assert backend.temporary_driver_torque_limit_joint_names == ("joint6",)
    with pytest.raises(B601HardwareError, match="must be restored before close"):
        backend.close()
    assert backend.connected is True

    motor.fail_method = None
    assert backend.restore_driver_torque_limits().completed is True
    backend.close()


def test_failed_driver_torque_limit_rollback_marks_final_value_unknown() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(allow_motion=True),
        controller_factory=lambda _channel: controller,
    )
    backend.connect()
    for motor_id in (5, 6):
        controller.motors[motor_id].parameter_values[
            B601_RS_TORQUE_LIMIT_PARAM_ID
        ] = 14.0
    controller.motors[6].write_without_effect.add(B601_RS_TORQUE_LIMIT_PARAM_ID)
    controller.motors[5].fail_write_values.add(14.0)
    assert backend.safe_stop(joint_names=("joint5", "joint6")).completed is True

    receipt = backend.apply_temporary_driver_torque_limits(
        (0.5,) * 6,
        joint_names=("joint5", "joint6"),
    )

    assert receipt.completed is False
    assert receipt.rollback_attempted is True
    assert receipt.rollback_completed is False
    assert receipt.failed_joint_names == ("joint6", "joint5")
    assert backend.temporary_driver_torque_limit_joint_names == ("joint5",)
    with pytest.raises(B601HardwareError, match="must be restored before close"):
        backend.close()

    controller.motors[5].fail_write_values.clear()
    restore_receipt = backend.restore_driver_torque_limits()
    assert restore_receipt.completed is True
    assert restore_receipt.prior_limits_nm == (None,)
    assert restore_receipt.verified_limits_nm == (14.0,)
    backend.close()


def test_temporary_driver_torque_limit_rejects_protocol_range_before_io() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(allow_motion=True),
        controller_factory=lambda _channel: controller,
    )
    backend.connect()
    controller.motors[6].parameter_values[B601_RS_TORQUE_LIMIT_PARAM_ID] = 14.0

    with pytest.raises(ValueError, match="exceeds RobStride"):
        backend.apply_temporary_driver_torque_limits(
            (61.0,) * 6,
            joint_names=("joint6",),
        )

    assert controller.motors[6].calls == []


def test_temporary_driver_torque_limit_requires_confirmed_disabled_state() -> None:
    controller = _FakeController()
    backend = B601RSBackend(
        _config(allow_motion=True),
        controller_factory=lambda _channel: controller,
    )
    backend.connect()
    controller.motors[6].parameter_values[B601_RS_TORQUE_LIMIT_PARAM_ID] = 14.0

    with pytest.raises(B601HardwareError, match="successful disable/safe_stop"):
        backend.apply_temporary_driver_torque_limits(
            (0.5,) * 6,
            joint_names=("joint6",),
        )

    assert backend.confirmed_disabled_joint_names == ()
    assert controller.motors[6].calls == []


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
