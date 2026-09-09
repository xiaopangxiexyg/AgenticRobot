#!/usr/bin/env python3
"""Measure B601-RS feedback/bus timing with a guarded active path.

The default ``read-only`` mode opens a passive SocketCAN receive socket only.
It does not construct MotorBridge, request feedback, enable a motor, or send a
position/MIT frame.  ``offline`` provides deterministic smoke evidence.
``gravity-offline`` measures the existing Pinocchio gravity plus MIT torque
composition path without opening CAN.
``position-tx-disabled`` measures only the POS_VEL dispatch clock after an
explicit successful disable and never enables a motor.  ``mit-tx-disabled``
adds only zero-gain, zero-torque MIT dispatch after two successful disable
receipts.  ``position`` is the separately confirmed enabled commissioning
path.  No active mode is run by automated tests or by this development task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import struct
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

# ``python scripts/hardware/...`` places the script directory, rather than the
# repository root, on ``sys.path``.  Keep the documented direct invocation
# working without requiring an editable package install.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hardware.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601HardwareError,
    B601ModeReceipt,
    B601RSBackend,
    B601RSBackendConfig,
    FrequencyLedger,
    evaluate_frequency_target,
    summarize_timestamps,
)
from robot_motion.rebot_b601_rs import (
    B601MitControlProfile,
    B601MitReference,
    B601PinocchioModel,
    compute_b601_mit_torque,
)

_CAN_RAW = getattr(socket, "CAN_RAW", 1)
_CAN_FRAME = struct.Struct("=IB3x8s")
_CAN_EFF_FLAG = 0x80000000
_CAN_EFF_MASK = 0x1FFFFFFF
_CAN_SFF_MASK = 0x000007FF

_ROBSTRIDE_COMMUNICATION_TYPE_NAMES = {
    0: "get_device_id",
    1: "operation_control",
    2: "operation_status",
    3: "enable",
    4: "disable",
    6: "set_zero_position",
    7: "set_device_id",
    17: "read_parameter",
    18: "write_parameter",
    21: "fault_report",
    22: "save_parameters",
    23: "set_baudrate",
    24: "active_report",
    25: "set_protocol",
}
_ROBSTRIDE_OPERATION_STATUS = 2
_ROBSTRIDE_ACTIVE_REPORT = 24
_ROBSTRIDE_STREAM_TYPES = (
    _ROBSTRIDE_OPERATION_STATUS,
    _ROBSTRIDE_ACTIVE_REPORT,
)


class FrequencyProbeError(RuntimeError):
    """A probe precondition or transport operation failed."""


@dataclass(frozen=True)
class DecodedFrame:
    arbitration_id: int
    data_length: int
    extended: bool


@dataclass(frozen=True)
class RobStrideExtendedId:
    """Protocol fields carried by a candidate RobStride private-protocol ID."""

    communication_type: int
    communication_name: str
    extra_data: int
    node_id: int

    @property
    def stream_device_id(self) -> int | None:
        if self.communication_type not in _ROBSTRIDE_STREAM_TYPES:
            return None
        return self.extra_data & 0xFF


def decode_can_frame(frame: bytes) -> DecodedFrame:
    """Decode a classic 16-byte SocketCAN frame without touching the bus."""

    if len(frame) < _CAN_FRAME.size:
        raise ValueError(f"SocketCAN frame is too short: {len(frame)} bytes")
    can_id, data_length, _data = _CAN_FRAME.unpack(frame[: _CAN_FRAME.size])
    if data_length > 8:
        raise ValueError(f"classic CAN data length is invalid: {data_length}")
    extended = bool(can_id & _CAN_EFF_FLAG)
    arbitration_id = can_id & (_CAN_EFF_MASK if extended else _CAN_SFF_MASK)
    return DecodedFrame(
        arbitration_id=arbitration_id,
        data_length=int(data_length),
        extended=extended,
    )


def decode_robstride_extended_id(frame: DecodedFrame) -> RobStrideExtendedId:
    """Classify a 29-bit ID without interpreting its vendor payload as state."""

    if not frame.extended:
        raise ValueError("RobStride private protocol requires an extended CAN frame")
    communication_type = (frame.arbitration_id >> 24) & 0x1F
    return RobStrideExtendedId(
        communication_type=communication_type,
        communication_name=_ROBSTRIDE_COMMUNICATION_TYPE_NAMES.get(
            communication_type,
            "unknown",
        ),
        extra_data=(frame.arbitration_id >> 8) & 0xFFFF,
        node_id=frame.arbitration_id & 0xFF,
    )


def _communication_type_key(communication_type: int) -> str:
    name = _ROBSTRIDE_COMMUNICATION_TYPE_NAMES.get(communication_type, "unknown")
    return f"{communication_type}:{name}"


def _robstride_candidate_summary(
    communication_type_counts: Counter[int],
    stream_timestamps_by_type_and_device: dict[
        int,
        dict[int, list[float]],
    ],
    stream_host_id_counts_by_type: dict[int, Counter[int]],
    *,
    short_stream_frame_counts: Counter[int],
) -> dict[str, object]:
    unknown_count = sum(
        count
        for communication_type, count in communication_type_counts.items()
        if communication_type not in _ROBSTRIDE_COMMUNICATION_TYPE_NAMES
    )
    result: dict[str, object] = {
        "robstride_extended_candidate_frame_count": sum(
            communication_type_counts.values()
        ),
        "frames_by_robstride_communication_type": {
            _communication_type_key(communication_type): count
            for communication_type, count in sorted(communication_type_counts.items())
        },
        "robstride_unknown_communication_type_frame_count": unknown_count,
    }
    stream_frame_counts: dict[int, int] = {}
    for communication_type, label in (
        (_ROBSTRIDE_OPERATION_STATUS, "operation_status"),
        (_ROBSTRIDE_ACTIVE_REPORT, "active_report"),
    ):
        timestamps_by_device_id = stream_timestamps_by_type_and_device.get(
            communication_type,
            {},
        )
        host_id_counts = stream_host_id_counts_by_type.get(
            communication_type,
            Counter(),
        )
        stream_frame_count = sum(
            len(timestamps) for timestamps in timestamps_by_device_id.values()
        )
        stream_frame_counts[communication_type] = stream_frame_count
        result[f"robstride_{label}_frame_count"] = stream_frame_count
        result[f"robstride_short_{label}_frame_count"] = short_stream_frame_counts[
            communication_type
        ]
        result[f"robstride_{label}_frames_by_device_id"] = {
            str(device_id): len(timestamps)
            for device_id, timestamps in sorted(timestamps_by_device_id.items())
        }
        result[f"robstride_{label}_frames_by_host_id"] = {
            f"0x{host_id:02x}": count
            for host_id, count in sorted(host_id_counts.items())
        }
        result[f"robstride_{label}_per_device_feedback"] = {
            str(device_id): summarize_timestamps("feedback", timestamps).as_dict()
            for device_id, timestamps in sorted(timestamps_by_device_id.items())
        }
    operation_status_observed = bool(stream_frame_counts[_ROBSTRIDE_OPERATION_STATUS])
    active_report_observed = bool(stream_frame_counts[_ROBSTRIDE_ACTIVE_REPORT])
    if operation_status_observed and active_report_observed:
        classification = "ROBSTRIDE_OPERATION_STATUS_AND_ACTIVE_REPORT_CANDIDATES_OBSERVED"
    elif operation_status_observed:
        classification = "ROBSTRIDE_OPERATION_STATUS_CANDIDATE_OBSERVED"
    elif active_report_observed:
        classification = "ROBSTRIDE_ACTIVE_REPORT_CANDIDATE_OBSERVED"
    else:
        classification = "NO_DECODABLE_ROBSTRIDE_STREAM_CANDIDATE_OBSERVED"
    result["feedback_path_classification"] = classification
    return result


def _per_id_summary(
    timestamps_by_id: dict[int, list[float]],
) -> dict[str, dict[str, object]]:
    return {
        f"0x{arbitration_id:x}": summarize_timestamps("feedback", timestamps)
        .as_dict()
        for arbitration_id, timestamps in sorted(timestamps_by_id.items())
    }


def _read_only_capture(
    channel: str, duration_sec: float, *, raw_output: Path | None = None,
) -> dict[str, object]:
    if not channel.strip():
        raise ValueError("channel must not be empty")
    if not math.isfinite(duration_sec) or duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive")
    ledger = FrequencyLedger()
    timestamps_by_id: dict[int, list[float]] = defaultdict(list)
    counts: Counter[int] = Counter()
    extended_count = 0
    standard_count = 0
    malformed_count = 0
    communication_type_counts: Counter[int] = Counter()
    stream_timestamps_by_type_and_device: dict[
        int,
        dict[int, list[float]],
    ] = {
        communication_type: defaultdict(list)
        for communication_type in _ROBSTRIDE_STREAM_TYPES
    }
    stream_host_id_counts_by_type: dict[int, Counter[int]] = {
        communication_type: Counter()
        for communication_type in _ROBSTRIDE_STREAM_TYPES
    }
    short_stream_frame_counts: Counter[int] = Counter()
    started = time.monotonic()
    deadline = started + duration_sec
    receiver = None
    try:
        receiver = socket.socket(socket.AF_CAN, socket.SOCK_RAW, _CAN_RAW)
        receiver.bind((channel,))
        receiver.settimeout(min(0.2, max(0.01, duration_sec)))
    except OSError as exc:
        if receiver is not None:
            receiver.close()
        raise FrequencyProbeError(f"cannot open passive SocketCAN receiver on {channel}: {exc}") from exc
    with receiver, (
        raw_output.open("x", encoding="utf-8")
        if raw_output is not None else nullcontext(None)
    ) as raw_writer:
        if raw_writer is not None:
            raw_writer.write(json.dumps({
                "kind": "header", "format": "b601_socketcan_rx_v1", "channel": channel,
                "timestamp_semantics": "userspace_after_recv_not_motor_acquisition",
                "clock_pair_semantics": "realtime_read_between_receive_and_end_monotonic_ns",
                "transmit_capability_used": False,
            }) + "\n")
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            receiver.settimeout(min(0.2, remaining))
            try:
                raw = receiver.recv(72)
            except socket.timeout:
                continue
            except OSError as exc:
                raise FrequencyProbeError(f"SocketCAN receive failed on {channel}: {exc}") from exc
            received_at = time.monotonic()
            if raw_writer is not None:
                received_ns = time.monotonic_ns()
                realtime_ns = time.time_ns()
                clock_pair_end_ns = time.monotonic_ns()
                raw_writer.write(json.dumps({
                    "kind": "frame",
                    "receive_monotonic_ns": received_ns,
                    "observe_realtime_ns": realtime_ns,
                    "clock_pair_end_monotonic_ns": clock_pair_end_ns,
                    "socket_frame_hex": raw.hex(),
                }) + "\n")
            try:
                decoded = decode_can_frame(raw)
            except ValueError:
                malformed_count += 1
                continue
            ledger.mark("feedback", received_at)
            timestamps_by_id[decoded.arbitration_id].append(received_at)
            counts[decoded.arbitration_id] += 1
            if decoded.extended:
                extended_count += 1
                identity = decode_robstride_extended_id(decoded)
                communication_type_counts[identity.communication_type] += 1
                if identity.communication_type in _ROBSTRIDE_STREAM_TYPES:
                    if decoded.data_length == 8:
                        device_id = identity.stream_device_id
                        assert device_id is not None
                        stream_timestamps_by_type_and_device[
                            identity.communication_type
                        ][device_id].append(received_at)
                        stream_host_id_counts_by_type[
                            identity.communication_type
                        ][identity.node_id] += 1
                    else:
                        short_stream_frame_counts[identity.communication_type] += 1
            else:
                standard_count += 1
        if raw_writer is not None:
            raw_writer.write(json.dumps({
                "kind": "end", "completed": True,
                "received_frame_count": sum(counts.values()) + malformed_count,
            }) + "\n")
    ended = time.monotonic()
    feedback_stats = ledger.summarize()["feedback"].as_dict()
    result = {
        "mode": "read-only",
        "channel": channel,
        "duration_sec": ended - started,
        "transport": "socketcan_passive_receive_only",
        "motion_commands_sent": False,
        "feedback": feedback_stats,
        "frame_count": sum(counts.values()),
        "unique_arbitration_ids": len(counts),
        "frames_by_arbitration_id": {
            f"0x{arbitration_id:x}": count
            for arbitration_id, count in sorted(counts.items())
        },
        "per_id_feedback": _per_id_summary(dict(timestamps_by_id)),
        "extended_frame_count": extended_count,
        "standard_frame_count": standard_count,
        "malformed_frame_count": malformed_count,
        "freshness_verified": False,
        "freshness_semantics": (
            "socket_receive_time_only; robstride_stream_payload_not_decoded; "
            "source_sequence_unavailable"
        ),
        "notes": [
            "This measures received CAN frame timing, not command, compute, or haptic rate.",
            "Communication types are classified from candidate RobStride 29-bit IDs; payload values are not decoded as joint state.",
            "Type-2 operation-status and type-24 active-report candidates are timed separately; neither is promoted to joint state.",
            "A stream candidate establishes per-device receive cadence only, not source freshness or MotorBridge cache freshness.",
            "MotorBridge 0.4.9 does not expose a source timestamp/sequence for MotorState.",
        ],
    }
    result.update(
        _robstride_candidate_summary(
            communication_type_counts,
            {
                communication_type: dict(timestamps_by_device_id)
                for communication_type, timestamps_by_device_id in (
                    stream_timestamps_by_type_and_device.items()
                )
            },
            stream_host_id_counts_by_type,
            short_stream_frame_counts=short_stream_frame_counts,
        )
    )
    if raw_output is not None:
        result["raw_capture"] = {
            "path": str(raw_output.resolve()), "format": "b601_socketcan_rx_v1",
            "completed": True, "hardware_acquisition_time_verified": False,
            "scope": "receive_observations_only_not_dispatch_receipts",
        }
    return result


def _offline_capture(duration_sec: float, target_hz: float) -> dict[str, object]:
    if not math.isfinite(duration_sec) or duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive")
    if not math.isfinite(target_hz) or target_hz <= 0.0:
        raise ValueError("target_hz must be positive")
    period = 1.0 / target_hz
    sample_count = max(2, int(round(duration_sec * target_hz)) + 1)
    timestamps = [index * period for index in range(sample_count)]
    stats = summarize_timestamps("feedback", timestamps, target_hz=target_hz).as_dict()
    return {
        "mode": "offline",
        "duration_sec": duration_sec,
        "transport": "synthetic_clock",
        "motion_commands_sent": False,
        "feedback": stats,
        "frame_count": 0,
        "unique_arbitration_ids": 0,
        "frames_by_arbitration_id": {},
        "per_id_feedback": {},
        "extended_frame_count": 0,
        "standard_frame_count": 0,
        "malformed_frame_count": 0,
        "robstride_extended_candidate_frame_count": 0,
        "frames_by_robstride_communication_type": {},
        "robstride_unknown_communication_type_frame_count": 0,
        "robstride_operation_status_frame_count": 0,
        "robstride_short_operation_status_frame_count": 0,
        "robstride_operation_status_frames_by_device_id": {},
        "robstride_operation_status_frames_by_host_id": {},
        "robstride_operation_status_per_device_feedback": {},
        "robstride_active_report_frame_count": 0,
        "robstride_short_active_report_frame_count": 0,
        "robstride_active_report_frames_by_device_id": {},
        "robstride_active_report_frames_by_host_id": {},
        "robstride_active_report_per_device_feedback": {},
        "feedback_path_classification": "NOT_APPLICABLE_SYNTHETIC",
        "freshness_verified": True,
        "freshness_semantics": "synthetic_only",
        "notes": ["Offline timing is not hardware or CAN evidence."],
    }


def _gravity_offline_capture(
    duration_sec: float,
    target_hz: float,
    source_root: Path | None,
    *,
    minimum_rate_ratio: float = 0.95,
    maximum_deadline_miss_ratio: float = 0.01,
) -> dict[str, object]:
    """Measure the existing gravity/composition path without hardware I/O."""

    if not math.isfinite(duration_sec) or duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive")
    if not math.isfinite(target_hz) or target_hz <= 0.0:
        raise ValueError("target_hz must be positive")
    if source_root is None:
        raise FrequencyProbeError("gravity-offline mode requires --rebot-source-root")
    root = source_root.expanduser().resolve(strict=True)
    try:
        model = B601PinocchioModel.from_rebot_source_root(root)
    except Exception as exc:
        raise FrequencyProbeError(
            f"cannot construct B601-RS gravity model: {exc}"
        ) from exc

    lower, upper = model.arm_position_limits()
    qpos = tuple(
        0.0 if float(lo) <= 0.0 <= float(hi) else 0.5 * (float(lo) + float(hi))
        for lo, hi in zip(lower, upper)
    )
    zero = (0.0,) * 6
    reference = B601MitReference(
        sequence=0,
        source_timestamp_sec=0.0,
        receive_timestamp_sec=0.0,
        qpos_rad=qpos,
        qvel_rad_s=zero,
    )
    profile = B601MitControlProfile(
        name="gravity-offline-zero-pd",
        position_kp_nm_per_rad=zero,
        damping_kd_nm_s_per_rad=zero,
        max_abs_torque_nm=(1.0e6,) * 6,
        max_torque_rate_nm_s=(1.0e9,) * 6,
    )
    ledger = FrequencyLedger()
    schedule_overruns = 0
    iterations = 0
    last_gravity: tuple[float, ...] | None = None
    last_requested: tuple[float, ...] | None = None
    period = 1.0 / target_hz
    started = time.monotonic()
    next_deadline = started
    while iterations == 0 or time.monotonic() - started < duration_sec:
        now = time.monotonic()
        if now < next_deadline:
            time.sleep(next_deadline - now)
        try:
            gravity = tuple(
                float(value) for value in model.arm_generalized_gravity(qpos)
            )
            breakdown = compute_b601_mit_torque(
                reference,
                profile,
                measured_qpos_rad=qpos,
                measured_qvel_rad_s=zero,
                gravity_torque_nm=gravity,
            )
        except Exception as exc:
            raise FrequencyProbeError(
                f"offline gravity/torque computation failed: {exc}"
            ) from exc
        if breakdown.requested_torque_nm != gravity:
            raise FrequencyProbeError(
                "zero-PD offline composition did not preserve gravity torque"
            )
        ledger.mark("compute")
        iterations += 1
        last_gravity = gravity
        last_requested = breakdown.requested_torque_nm
        next_deadline += period
        completed_at = time.monotonic()
        if completed_at > next_deadline:
            schedule_overruns += 1
            next_deadline = completed_at

    stats = ledger.summarize(target_hz=target_hz)
    acceptance = evaluate_frequency_target(
        stats["compute"],
        minimum_rate_ratio=minimum_rate_ratio,
        maximum_deadline_miss_ratio=maximum_deadline_miss_ratio,
    )
    return {
        "mode": "gravity-offline",
        "duration_sec": duration_sec,
        "target_hz": target_hz,
        "rebot_source_root": str(root),
        "urdf_path": str(model.urdf_path),
        "transport": "none_offline_pinocchio",
        "motion_commands_sent": False,
        "motors_enabled_by_probe": False,
        "iterations": iterations,
        "schedule_overruns": schedule_overruns,
        "qpos_rad": qpos,
        "auxiliary_joint_positions": "pinocchio_neutral",
        "last_gravity_torque_nm": last_gravity,
        "last_requested_torque_nm": last_requested,
        "mit_command_semantics": "software_zero_pd_plus_pinocchio_gravity",
        "execution_status": "COMPLETED",
        "acceptance_status": "PASS" if acceptance.passed else "FAIL",
        "acceptance": {
            "required_events": ("compute",),
            "minimum_rate_ratio": minimum_rate_ratio,
            "maximum_deadline_miss_ratio": maximum_deadline_miss_ratio,
            "events": {"compute": acceptance.as_dict()},
        },
        "frequency": {event: value.as_dict() for event, value in stats.items()},
        "freshness_verified": False,
        "freshness_semantics": "not_applicable_offline_fixed_state",
        "notes": [
            "Offline compute evidence only; no CAN socket, MotorBridge controller, mode transition, enable, or command send is used.",
            "The measured path reuses B601PinocchioModel.arm_generalized_gravity and compute_b601_mit_torque with zero position/damping gains.",
            "This does not validate feedback freshness, motor torque sign/scale, damping, friction, or enabled-loop stability.",
        ],
    }


def _parse_joint_names(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return B601_RS_ARM_JOINT_NAMES
    names: list[str] = []
    for value in values:
        names.extend(part.strip() for part in str(value).split(",") if part.strip())
    resolved = tuple(names)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("--joint must contain unique joint names")
    unknown = set(resolved).difference(B601_RS_ARM_JOINT_NAMES)
    if unknown:
        raise ValueError(f"unknown B601-RS joint names: {sorted(unknown)}")
    return resolved


def _mode_receipt_payload(receipt: B601ModeReceipt) -> dict[str, object]:
    return {
        "operation": receipt.operation,
        "requested_joint_names": receipt.requested_joint_names,
        "completed_joint_names": receipt.completed_joint_names,
        "failed_joint_names": receipt.failed_joint_names,
        "errors": receipt.errors,
        "completed": receipt.completed,
    }


def _resolve_position_limits(
    config_path: Path,
    source_root: Path | None,
) -> tuple[tuple[float, float], ...]:
    """Resolve six arm limits from the pinned URDF before enabling motors."""

    root = (
        config_path.parent.parent
        if source_root is None
        else source_root.expanduser().resolve(strict=True)
    )
    try:
        from robot_motion.rebot_b601_rs import B601PinocchioModel

        model = B601PinocchioModel.from_rebot_source_root(root)
        lower, upper = model.arm_position_limits()
    except Exception as exc:
        raise FrequencyProbeError(
            "cannot resolve finite B601-RS URDF position limits before motion: "
            f"{exc}"
        ) from exc
    limits = tuple(
        (float(lower_value), float(upper_value))
        for lower_value, upper_value in zip(lower, upper)
    )
    if len(limits) != 6 or any(
        not math.isfinite(lower_value)
        or not math.isfinite(upper_value)
        or lower_value >= upper_value
        for lower_value, upper_value in limits
    ):
        raise FrequencyProbeError("B601-RS URDF returned invalid arm position limits")
    return limits


def _run_position_probe(args: argparse.Namespace) -> dict[str, object]:
    mode = str(args.mode)
    if mode not in ("position", "position-tx-disabled", "mit-tx-disabled"):
        raise ValueError("active TX probe received an unsupported mode")
    if not args.allow_motion:
        raise FrequencyProbeError(f"{mode} mode requires --allow-motion")
    required_confirmation = {
        "position": "B601_RS_POSITION_FREQUENCY",
        "position-tx-disabled": "B601_RS_DISABLED_TX_FREQUENCY",
        "mit-tx-disabled": "B601_RS_DISABLED_MIT_TX_FREQUENCY",
    }[mode]
    if args.confirmation != required_confirmation:
        raise FrequencyProbeError(
            f"{mode} mode requires --confirmation {required_confirmation}"
        )
    if mode == "position" and not args.enable:
        raise FrequencyProbeError("position mode requires explicit --enable")
    if mode != "position" and args.enable:
        raise FrequencyProbeError(f"{mode} mode rejects --enable")
    if not args.auto_disable:
        raise FrequencyProbeError(f"{mode} mode requires explicit --auto-disable")
    if not args.config:
        raise FrequencyProbeError(
            f"{mode} mode requires --config pointing to rebotarm_rs.yaml"
        )
    if not math.isfinite(args.duration_sec) or args.duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive")
    if not math.isfinite(args.warmup_sec) or args.warmup_sec <= 0.0:
        raise ValueError("warmup_sec must be positive")
    target_hz = float(args.target_hz)
    if not math.isfinite(target_hz) or target_hz <= 0.0:
        raise ValueError("target_hz must be positive")
    velocity_limit_rad_s = float(args.velocity_limit_rad_s)
    if not math.isfinite(velocity_limit_rad_s) or velocity_limit_rad_s <= 0.0:
        raise ValueError("velocity_limit_rad_s must be positive")
    minimum_rate_ratio = float(args.minimum_rate_ratio)
    if (
        not math.isfinite(minimum_rate_ratio)
        or not 0.0 < minimum_rate_ratio <= 1.0
    ):
        raise ValueError("minimum_rate_ratio must be in (0, 1]")
    maximum_deadline_miss_ratio = float(args.maximum_deadline_miss_ratio)
    if (
        not math.isfinite(maximum_deadline_miss_ratio)
        or not 0.0 <= maximum_deadline_miss_ratio <= 1.0
    ):
        raise ValueError("maximum_deadline_miss_ratio must be in [0, 1]")
    selected = _parse_joint_names(args.joint)
    config_path = Path(args.config).expanduser().resolve(strict=True)
    position_limits = _resolve_position_limits(
        config_path,
        args.rebot_source_root,
    )
    config = B601RSBackendConfig.from_rs_yaml(
        config_path,
        allow_motion=True,
        position_limits_rad=position_limits,
    )
    effective_velocity_limits = tuple(
        min(velocity_limit_rad_s, joint.position_velocity_limit_rad_s)
        for joint in config.joints
    )
    backend = B601RSBackend(config)
    ledger = FrequencyLedger()
    cleanup_required = False
    commands_sent = 0
    schedule_overruns = 0
    stop_reason: str | None = None
    cleanup_errors: list[str] = []
    cleanup_receipt: dict[str, object] | None = None
    preflight_disable_receipt: dict[str, object] | None = None
    post_mode_disable_receipt: dict[str, object] | None = None
    mit_reference: B601MitReference | None = None
    primary_error: Exception | None = None
    try:
        backend.connect()
        warmup_deadline = time.monotonic() + args.warmup_sec
        state = None
        while time.monotonic() < warmup_deadline:
            state = backend.poll_feedback(request_feedback=True)
            ledger.mark("feedback", state.receive_timestamp_sec)
            if state.valid and state.positions_rad is not None:
                break
            time.sleep(0.005)
        if state is None or not state.valid or state.positions_rad is None:
            fault_reason = (
                "no feedback observation"
                if state is None
                else state.fault_reason or "snapshot did not contain six valid positions"
            )
            raise FrequencyProbeError(
                "no complete valid feedback snapshot arrived during warmup; "
                f"last_fault={fault_reason}"
            )
        initial_positions = state.positions_rad
        initial_sequence = state.sequence
        if mode == "mit-tx-disabled":
            mit_reference = B601MitReference(
                sequence=initial_sequence,
                source_timestamp_sec=state.receive_timestamp_sec,
                receive_timestamp_sec=state.receive_timestamp_sec,
                qpos_rad=initial_positions,
            )
        if mode != "position":
            cleanup_required = True
            disable_receipt = backend.safe_stop(joint_names=selected)
            preflight_disable_receipt = _mode_receipt_payload(disable_receipt)
            if not disable_receipt.completed:
                raise FrequencyProbeError(
                    "preflight disable failed: " + "; ".join(disable_receipt.errors)
                )
        if mode == "mit-tx-disabled":
            mode_receipt = backend.prepare_mit_mode(joint_names=selected)
        else:
            mode_receipt = backend.prepare_position_velocity_mode(
                joint_names=selected,
                velocity_limits_rad_s=effective_velocity_limits,
            )
        if not mode_receipt.completed:
            mode_name = "MIT" if mode == "mit-tx-disabled" else "POS_VEL"
            raise FrequencyProbeError(
                f"{mode_name} setup failed: " + "; ".join(mode_receipt.errors)
            )
        if mode == "mit-tx-disabled":
            disable_receipt = backend.safe_stop(joint_names=selected)
            post_mode_disable_receipt = _mode_receipt_payload(disable_receipt)
            if not disable_receipt.completed:
                raise FrequencyProbeError(
                    "post-MIT-mode disable failed: "
                    + "; ".join(disable_receipt.errors)
                )
        if mode == "position":
            cleanup_required = True
            enable_receipt = backend.enable(joint_names=selected)
            if not enable_receipt.completed:
                raise FrequencyProbeError(
                    "motor enable failed: " + "; ".join(enable_receipt.errors)
                )
        period = 1.0 / target_hz
        started = time.monotonic()
        next_deadline = started
        while time.monotonic() - started < args.duration_sec:
            now = time.monotonic()
            if now < next_deadline:
                time.sleep(next_deadline - now)
            if mode == "position":
                state = backend.poll_feedback(request_feedback=True)
                ledger.mark("feedback", state.receive_timestamp_sec)
                if not state.valid or state.positions_rad is None:
                    stop_reason = state.fault_reason or "invalid feedback"
                    break
                reference_positions = state.positions_rad
                source_sequence = state.sequence
            else:
                reference_positions = initial_positions
                source_sequence = initial_sequence
            ledger.mark("compute")
            if mode == "mit-tx-disabled":
                if mit_reference is None:
                    raise FrequencyProbeError("MIT reference was not initialized")
                receipt = backend.send_mit(
                    mit_reference,
                    kp=(0.0,) * 6,
                    kd=(0.0,) * 6,
                    tau_nm=(0.0,) * 6,
                    torque_limits_nm=(0.0,) * 6,
                    joint_names=selected,
                )
            else:
                receipt = backend.send_position_velocity(
                    reference_positions,
                    velocity_limits_rad_s=effective_velocity_limits,
                    joint_names=selected,
                    source_sequence=source_sequence,
                )
            if not receipt.accepted:
                command_name = "MIT" if mode == "mit-tx-disabled" else "position"
                stop_reason = (
                    "; ".join(receipt.errors) or f"partial {command_name} dispatch"
                )
                break
            ledger.mark("tx", receipt.completed_at_sec)
            commands_sent += 1
            next_deadline += period
            completed_at = time.monotonic()
            if completed_at > next_deadline:
                schedule_overruns += 1
                next_deadline = completed_at
    except Exception as exc:  # noqa: BLE001 - always reach fail-safe cleanup
        primary_error = exc
    finally:
        if cleanup_required:
            try:
                stop_receipt = backend.safe_stop(joint_names=selected)
                cleanup_receipt = _mode_receipt_payload(stop_receipt)
                if not stop_receipt.completed:
                    cleanup_errors.extend(
                        f"safe_stop: {error}" for error in stop_receipt.errors
                    )
                    if not stop_receipt.errors:
                        cleanup_errors.append(
                            "safe_stop: incomplete without vendor error"
                        )
            except Exception as exc:  # noqa: BLE001 - report vendor cleanup failure
                cleanup_errors.append(f"safe_stop: {exc}")
        try:
            backend.close()
        except Exception as exc:  # noqa: BLE001 - report vendor cleanup failure
            cleanup_errors.append(f"close: {exc}")
        if cleanup_errors and stop_reason is None:
            stop_reason = "; ".join(cleanup_errors)
    if primary_error is not None:
        detail = str(primary_error)
        if cleanup_errors:
            detail += "; cleanup: " + "; ".join(cleanup_errors)
        raise FrequencyProbeError(detail) from primary_error
    frequency_stats = ledger.summarize(target_hz=target_hz)
    required_events = (
        ("feedback", "compute", "tx")
        if mode == "position"
        else ("compute", "tx")
    )
    event_acceptance = {
        event: evaluate_frequency_target(
            frequency_stats[event],
            minimum_rate_ratio=minimum_rate_ratio,
            maximum_deadline_miss_ratio=maximum_deadline_miss_ratio,
        )
        for event in required_events
    }
    acceptance_passed = stop_reason is None and all(
        acceptance.passed for acceptance in event_acceptance.values()
    )
    result: dict[str, object] = {
        "mode": mode,
        "channel": config.channel,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "rebot_source_root": str(
            config_path.parent.parent
            if args.rebot_source_root is None
            else args.rebot_source_root.expanduser().resolve(strict=True)
        ),
        "duration_sec": args.duration_sec,
        "target_hz": target_hz,
        "requested_velocity_limit_rad_s": velocity_limit_rad_s,
        "effective_velocity_limits_rad_s": effective_velocity_limits,
        "transport": "motorbridge_wired_local",
        "motion_commands_sent": commands_sent > 0,
        "motors_enabled_by_probe": mode == "position",
        "commands_sent": commands_sent,
        "schedule_overruns": schedule_overruns,
        "stop_reason": stop_reason,
        "execution_status": "COMPLETED",
        "acceptance_status": "PASS" if acceptance_passed else "FAIL",
        "acceptance": {
            "required_events": required_events,
            "minimum_rate_ratio": minimum_rate_ratio,
            "maximum_deadline_miss_ratio": maximum_deadline_miss_ratio,
            "events": {
                event: acceptance.as_dict()
                for event, acceptance in event_acceptance.items()
            },
        },
        "freshness_verified": False,
        "freshness_semantics": (
            "motorbridge_vendor_cache_unverified"
            if mode == "position"
            else "initial_vendor_snapshot_only; no_runtime_feedback"
        ),
        "frequency": {
            event: stats.as_dict() for event, stats in frequency_stats.items()
        },
        "notes": [
            "Human-run commissioning handoff only; this mode is never executed by automated tests.",
            (
                "It requests one vendor feedback snapshot per arm joint before each active poll and holds the latest valid position."
                if mode == "position"
                else (
                    "It verifies disable, then measures host-to-vendor POS_VEL dispatch while holding the warmup reference without runtime feedback."
                    if mode == "position-tx-disabled"
                    else "It verifies disable before and after MIT mode setup, then dispatches only the warmup position with qvel=kp=kd=tau=0 while the selected motor remains disabled."
                )
            ),
            (
                "The position command is capped by --velocity-limit-rad-s and the YAML per-joint limits."
                if mode != "mit-tx-disabled"
                else "Zero MIT gains and zero feedforward make the commanded effort exactly zero; this is a transport ceiling probe, not gravity or damping control."
            ),
        ],
    }
    if mode == "mit-tx-disabled":
        result["mit_command_semantics"] = {
            "qpos": "warmup_snapshot",
            "qvel_rad_s": (0.0,) * 6,
            "kp": (0.0,) * 6,
            "kd": (0.0,) * 6,
            "tau_nm": (0.0,) * 6,
            "torque_limits_nm": (0.0,) * 6,
            "final_commanded_effort_nm": (0.0,) * 6,
        }
    if preflight_disable_receipt is not None:
        result["preflight_disable_receipt"] = preflight_disable_receipt
    if post_mode_disable_receipt is not None:
        result["post_mode_disable_receipt"] = post_mode_disable_receipt
    if cleanup_receipt is not None:
        result["cleanup_receipt"] = cleanup_receipt
    if cleanup_errors:
        result["cleanup_errors"] = tuple(cleanup_errors)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "B601-RS timing probe; read-only by default, active TX modes are explicitly guarded."
        )
    )
    parser.add_argument(
        "--mode",
        choices=(
            "read-only",
            "offline",
            "gravity-offline",
            "position-tx-disabled",
            "mit-tx-disabled",
            "position",
        ),
        default="read-only",
    )
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--duration-sec", "--duration", type=float, default=3.0)
    parser.add_argument("--target-hz", type=float, default=500.0)
    parser.add_argument(
        "--minimum-rate-ratio",
        type=float,
        default=0.95,
        help="minimum observed/target rate ratio required for a position PASS",
    )
    parser.add_argument(
        "--maximum-deadline-miss-ratio",
        type=float,
        default=0.01,
        help="maximum fraction of intervals allowed beyond the 2x-period deadline",
    )
    parser.add_argument(
        "--velocity-limit-rad-s",
        type=float,
        default=0.1,
        help="maximum POS_VEL command speed; defaults to the conservative 0.1 rad/s",
    )
    parser.add_argument(
        "--config", help="explicit rebotarm_rs.yaml for guarded active mode"
    )
    parser.add_argument(
        "--rebot-source-root",
        type=Path,
        help=(
            "reBotArm source root containing the pinned URDF; defaults to "
            "two parents above --config"
        ),
    )
    parser.add_argument("--joint", action="append", help="joint name or comma-separated names")
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--auto-disable", action="store_true")
    parser.add_argument("--output", type=Path, help="write JSON outside the source worktree")
    parser.add_argument(
        "--raw-output", type=Path,
        help="New JSONL file for raw RX frames and host clocks; read-only mode only, never overwrites.",
    )
    parser.add_argument("--json", action="store_true", help="retain machine-readable JSON output (default)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not math.isfinite(args.duration_sec) or args.duration_sec <= 0.0:
            raise ValueError("duration_sec must be positive")
        if args.raw_output is not None:
            if args.mode != "read-only":
                raise ValueError("--raw-output is supported only in read-only mode")
            args.raw_output = args.raw_output.expanduser().resolve()
            if args.output is not None and args.output.expanduser().resolve() == args.raw_output:
                raise ValueError("--output and --raw-output must be distinct paths")
        if args.mode == "read-only":
            if args.raw_output is None:
                payload = _read_only_capture(args.channel, args.duration_sec)
            else:
                payload = _read_only_capture(
                    args.channel, args.duration_sec, raw_output=args.raw_output,
                )
        elif args.mode == "offline":
            payload = _offline_capture(args.duration_sec, args.target_hz)
        elif args.mode == "gravity-offline":
            payload = _gravity_offline_capture(
                args.duration_sec,
                args.target_hz,
                args.rebot_source_root,
                minimum_rate_ratio=args.minimum_rate_ratio,
                maximum_deadline_miss_ratio=args.maximum_deadline_miss_ratio,
            )
        else:
            payload = _run_position_probe(args)
    except (B601HardwareError, FrequencyProbeError, OSError, ValueError) as exc:
        payload = {"status": "ERROR", "mode": args.mode, "error": str(exc)}
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    exit_code = 0
    status = "OK"
    if payload.get("acceptance_status") == "FAIL":
        status = "FAIL"
        exit_code = 1
    payload = {"status": status, **payload}
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.expanduser().resolve().write_text(encoded + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
