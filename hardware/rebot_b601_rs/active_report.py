"""Fail-closed acceptance for B601 RobStride active-report observations.

The evaluator consumes the existing passive SocketCAN capture summary.  It
only judges whether complete type-24 active-report frames arrived at the
requested cadence for each selected device.  Type-2 operation-status frames
are tracked separately and cannot satisfy this gate.  The evaluator
deliberately does not decode or admit either stream as control state.

The decoded evaluator is a separate gate.  It brackets the raw capture with
MotorBridge state snapshots and cross-checks the per-motor status sequence
against the raw type-24 stream from the same foreground interval.  Passing
that gate proves host receive freshness only; the device still exposes no
source acquisition timestamp.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .backend import B601HardwareState, B601JointConfig


def _positive(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be finite") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


@dataclass(frozen=True)
class B601ActiveReportDeviceAcceptance:
    """Transport-only acceptance for one selected B601 motor."""

    joint_name: str
    motor_id: int
    feedback_id: int
    sample_count: int
    observed_hz: float | None
    p99_interval_sec: float | None
    passed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "joint_name": self.joint_name,
            "motor_id": self.motor_id,
            "feedback_id": self.feedback_id,
            "sample_count": self.sample_count,
            "observed_hz": self.observed_hz,
            "p99_interval_sec": self.p99_interval_sec,
            "passed": self.passed,
            "reasons": self.reasons,
        }


@dataclass(frozen=True)
class B601ActiveReportAcceptance:
    """Aggregate transport gate that makes no control-freshness claim."""

    passed: bool
    target_hz: float
    minimum_observed_hz: float
    maximum_p99_interval_sec: float
    minimum_sample_count: int
    short_active_report_frame_count: int
    expected_feedback_ids: tuple[int, ...]
    observed_feedback_ids: tuple[int, ...]
    devices: tuple[B601ActiveReportDeviceAcceptance, ...]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "target_hz": self.target_hz,
            "minimum_observed_hz": self.minimum_observed_hz,
            "maximum_p99_interval_sec": self.maximum_p99_interval_sec,
            "minimum_sample_count": self.minimum_sample_count,
            "short_active_report_frame_count": (self.short_active_report_frame_count),
            "expected_feedback_ids": self.expected_feedback_ids,
            "observed_feedback_ids": self.observed_feedback_ids,
            "devices": tuple(device.as_dict() for device in self.devices),
            "reasons": self.reasons,
        }


@dataclass(frozen=True)
class B601DecodedActiveReportDeviceAcceptance:
    """Host-receive acceptance for one decoded type-24 stream."""

    joint_name: str
    motor_id: int
    feedback_id: int
    baseline_status_sequence: int | None
    final_status_sequence: int | None
    status_sequence_delta: int | None
    sequence_observed_hz: float | None
    raw_sample_count: int
    raw_observed_hz: float | None
    raw_p99_interval_sec: float | None
    decoded_raw_sample_ratio: float | None
    final_receive_age_sec: float | None
    passed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "joint_name": self.joint_name,
            "motor_id": self.motor_id,
            "feedback_id": self.feedback_id,
            "baseline_status_sequence": self.baseline_status_sequence,
            "final_status_sequence": self.final_status_sequence,
            "status_sequence_delta": self.status_sequence_delta,
            "sequence_observed_hz": self.sequence_observed_hz,
            "raw_sample_count": self.raw_sample_count,
            "raw_observed_hz": self.raw_observed_hz,
            "raw_p99_interval_sec": self.raw_p99_interval_sec,
            "decoded_raw_sample_ratio": self.decoded_raw_sample_ratio,
            "final_receive_age_sec": self.final_receive_age_sec,
            "passed": self.passed,
            "reasons": self.reasons,
        }


@dataclass(frozen=True)
class B601DecodedActiveReportAcceptance:
    """Aggregate decoded type-24 gate; device-source freshness is not claimed."""

    passed: bool
    target_hz: float
    minimum_observed_hz: float
    maximum_p99_interval_sec: float
    maximum_receive_age_sec: float
    minimum_sample_count: int
    minimum_decoded_raw_ratio: float
    capture_duration_sec: float
    short_active_report_frame_count: int
    expected_feedback_ids: tuple[int, ...]
    observed_feedback_ids: tuple[int, ...]
    devices: tuple[B601DecodedActiveReportDeviceAcceptance, ...]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "target_hz": self.target_hz,
            "minimum_observed_hz": self.minimum_observed_hz,
            "maximum_p99_interval_sec": self.maximum_p99_interval_sec,
            "maximum_receive_age_sec": self.maximum_receive_age_sec,
            "minimum_sample_count": self.minimum_sample_count,
            "minimum_decoded_raw_ratio": self.minimum_decoded_raw_ratio,
            "capture_duration_sec": self.capture_duration_sec,
            "short_active_report_frame_count": self.short_active_report_frame_count,
            "expected_feedback_ids": self.expected_feedback_ids,
            "observed_feedback_ids": self.observed_feedback_ids,
            "devices": tuple(device.as_dict() for device in self.devices),
            "reasons": self.reasons,
            "host_receive_freshness_verified": self.passed,
            "device_source_timestamp_verified": False,
        }


def evaluate_active_report_capture(
    capture: Mapping[str, object],
    joints: Sequence[B601JointConfig],
    selected_joint_names: Sequence[str],
    *,
    target_hz: float,
    minimum_rate_ratio: float = 0.90,
    maximum_p99_factor: float = 2.0,
    minimum_sample_count: int = 10,
) -> B601ActiveReportAcceptance:
    """Evaluate type-24 active-report cadence for every selected motor."""

    target = _positive(target_hz, name="target_hz")
    rate_ratio = _positive(minimum_rate_ratio, name="minimum_rate_ratio")
    if rate_ratio > 1.0:
        raise ValueError("minimum_rate_ratio must be at most 1")
    p99_factor = _positive(maximum_p99_factor, name="maximum_p99_factor")
    if (
        isinstance(minimum_sample_count, bool)
        or not isinstance(minimum_sample_count, int)
        or minimum_sample_count < 2
    ):
        raise ValueError("minimum_sample_count must be an integer of at least 2")

    selected = tuple(str(name) for name in selected_joint_names)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected_joint_names must be non-empty and unique")
    by_name = {joint.name: joint for joint in joints}
    unknown = set(selected).difference(by_name)
    if unknown:
        raise ValueError(f"unknown selected joints: {sorted(unknown)}")

    raw_per_device = capture.get(
        "robstride_active_report_per_device_feedback",
        {},
    )
    raw_hosts = capture.get("robstride_active_report_frames_by_host_id", {})
    if not isinstance(raw_per_device, Mapping) or not isinstance(raw_hosts, Mapping):
        raise TypeError("capture summary has malformed active-report mappings")
    short_count = int(capture.get("robstride_short_active_report_frame_count", 0))
    if short_count < 0:
        raise ValueError("short active-report frame count must be non-negative")

    observed_feedback_ids: set[int] = set()
    for raw_host in raw_hosts:
        try:
            observed_feedback_ids.add(int(str(raw_host), 0))
        except ValueError as exc:
            raise ValueError(f"invalid active-report host id: {raw_host}") from exc

    minimum_hz = target * rate_ratio
    maximum_p99 = p99_factor / target
    devices: list[B601ActiveReportDeviceAcceptance] = []
    aggregate_reasons: list[str] = []
    for name in selected:
        joint = by_name[name]
        raw_stats = raw_per_device.get(str(joint.motor_id))
        reasons: list[str] = []
        if not isinstance(raw_stats, Mapping):
            sample_count = 0
            observed_hz = None
            p99_interval = None
            reasons.append("no complete type-24 active-report frames observed")
        else:
            sample_count = int(raw_stats.get("sample_count", 0))
            observed_raw = raw_stats.get("observed_hz")
            p99_raw = raw_stats.get("p99_interval_sec")
            observed_hz = None if observed_raw is None else float(observed_raw)
            p99_interval = None if p99_raw is None else float(p99_raw)
        if sample_count < minimum_sample_count:
            reasons.append(
                f"sample_count={sample_count} below minimum={minimum_sample_count}"
            )
        if observed_hz is None or not math.isfinite(observed_hz):
            reasons.append("observed_hz is unavailable")
        elif observed_hz < minimum_hz:
            reasons.append(
                f"observed_hz={observed_hz:.3f} below minimum={minimum_hz:.3f}"
            )
        if p99_interval is None or not math.isfinite(p99_interval):
            reasons.append("p99_interval_sec is unavailable")
        elif p99_interval > maximum_p99:
            reasons.append(
                f"p99_interval_sec={p99_interval:.6f} exceeds maximum={maximum_p99:.6f}"
            )
        if joint.feedback_id not in observed_feedback_ids:
            reasons.append(
                f"expected feedback_id=0x{joint.feedback_id:02x} not observed"
            )
        if short_count:
            reasons.append(
                f"capture contains {short_count} short type-24 active-report frame(s)"
            )
        device = B601ActiveReportDeviceAcceptance(
            joint_name=name,
            motor_id=joint.motor_id,
            feedback_id=joint.feedback_id,
            sample_count=sample_count,
            observed_hz=observed_hz,
            p99_interval_sec=p99_interval,
            passed=not reasons,
            reasons=tuple(reasons),
        )
        devices.append(device)
        aggregate_reasons.extend(f"{name}: {reason}" for reason in reasons)

    expected_feedback_ids = tuple(
        sorted({by_name[name].feedback_id for name in selected})
    )
    return B601ActiveReportAcceptance(
        passed=not aggregate_reasons,
        target_hz=target,
        minimum_observed_hz=minimum_hz,
        maximum_p99_interval_sec=maximum_p99,
        minimum_sample_count=minimum_sample_count,
        short_active_report_frame_count=short_count,
        expected_feedback_ids=expected_feedback_ids,
        observed_feedback_ids=tuple(sorted(observed_feedback_ids)),
        devices=tuple(devices),
        reasons=tuple(aggregate_reasons),
    )


def evaluate_decoded_active_report_observations(
    capture: Mapping[str, object],
    baseline: B601HardwareState,
    final: B601HardwareState,
    joints: Sequence[B601JointConfig],
    selected_joint_names: Sequence[str],
    *,
    target_hz: float,
    maximum_receive_age_sec: float,
    minimum_rate_ratio: float = 0.80,
    maximum_p99_factor: float = 2.0,
    minimum_sample_count: int = 10,
    minimum_decoded_raw_ratio: float = 0.80,
) -> B601DecodedActiveReportAcceptance:
    """Cross-check decoded observations against raw type-24 active reports.

    ``baseline`` and ``final`` must bracket the foreground raw capture.  The
    MotorBridge status-sequence delta measures decoded frames accepted into
    the per-motor cache, while the raw summary independently measures complete
    type-24 frames received by SocketCAN.  The decoded/raw ratio prevents a
    sparse type-2 stream from masquerading as decoded type-24 progress.

    These are host-side observations.  Neither timestamp represents the
    motor's acquisition time, so a pass does not by itself admit the samples
    to a control loop.
    """

    receive_age_limit = _positive(
        maximum_receive_age_sec,
        name="maximum_receive_age_sec",
    )
    decoded_raw_ratio_limit = _positive(
        minimum_decoded_raw_ratio,
        name="minimum_decoded_raw_ratio",
    )
    if decoded_raw_ratio_limit > 1.0:
        raise ValueError("minimum_decoded_raw_ratio must be at most 1")
    duration = _positive(capture.get("duration_sec"), name="capture.duration_sec")
    selected = tuple(str(name) for name in selected_joint_names)
    raw_acceptance = evaluate_active_report_capture(
        capture,
        joints,
        selected,
        target_hz=target_hz,
        minimum_rate_ratio=minimum_rate_ratio,
        maximum_p99_factor=maximum_p99_factor,
        minimum_sample_count=minimum_sample_count,
    )
    by_name = {joint.name: joint for joint in joints}
    baseline_by_name = {joint.name: joint for joint in baseline.joints}
    final_by_name = {joint.name: joint for joint in final.joints}
    raw_by_name = {device.joint_name: device for device in raw_acceptance.devices}
    devices: list[B601DecodedActiveReportDeviceAcceptance] = []
    aggregate_reasons: list[str] = []
    for name in selected:
        joint = by_name[name]
        before = baseline_by_name.get(name)
        after = final_by_name.get(name)
        raw_device = raw_by_name[name]
        reasons = list(raw_device.reasons)

        baseline_sequence = None if before is None else before.status_sequence
        final_sequence = None if after is None else after.status_sequence
        sequence_delta: int | None = None
        sequence_hz: float | None = None
        if before is None or after is None:
            reasons.append("decoded observation is missing from a bracketing snapshot")
        else:
            if not before.valid or not before.host_receive_freshness_verified:
                reasons.append(
                    "baseline decoded observation is not valid and host-fresh"
                )
            if not after.valid or not after.host_receive_freshness_verified:
                reasons.append("final decoded observation is not valid and host-fresh")
            if baseline_sequence is None or final_sequence is None:
                reasons.append("status_sequence is unavailable")
            elif final_sequence <= baseline_sequence:
                reasons.append(
                    "status_sequence did not advance: "
                    f"baseline={baseline_sequence}, final={final_sequence}"
                )
            else:
                sequence_delta = final_sequence - baseline_sequence
                sequence_hz = sequence_delta / duration
                if sequence_delta < raw_acceptance.minimum_sample_count:
                    reasons.append(
                        f"status_sequence_delta={sequence_delta} below "
                        f"minimum={raw_acceptance.minimum_sample_count}"
                    )
                if sequence_hz < raw_acceptance.minimum_observed_hz:
                    reasons.append(
                        f"sequence_observed_hz={sequence_hz:.3f} below "
                        f"minimum={raw_acceptance.minimum_observed_hz:.3f}"
                    )

        final_receive_age = None if after is None else after.receive_age_sec
        if final_receive_age is None or not math.isfinite(final_receive_age):
            reasons.append("final receive_age_sec is unavailable")
        elif final_receive_age < 0.0:
            reasons.append("final receive_age_sec is negative")
        elif final_receive_age > receive_age_limit:
            reasons.append(
                f"final receive_age_sec={final_receive_age:.6f} exceeds "
                f"maximum={receive_age_limit:.6f}"
            )

        decoded_raw_ratio = None
        if sequence_delta is not None and raw_device.sample_count > 0:
            decoded_raw_ratio = sequence_delta / raw_device.sample_count
            if decoded_raw_ratio < decoded_raw_ratio_limit:
                reasons.append(
                    f"decoded_raw_sample_ratio={decoded_raw_ratio:.3f} below "
                    f"minimum={decoded_raw_ratio_limit:.3f}"
                )
        device = B601DecodedActiveReportDeviceAcceptance(
            joint_name=name,
            motor_id=joint.motor_id,
            feedback_id=joint.feedback_id,
            baseline_status_sequence=baseline_sequence,
            final_status_sequence=final_sequence,
            status_sequence_delta=sequence_delta,
            sequence_observed_hz=sequence_hz,
            raw_sample_count=raw_device.sample_count,
            raw_observed_hz=raw_device.observed_hz,
            raw_p99_interval_sec=raw_device.p99_interval_sec,
            decoded_raw_sample_ratio=decoded_raw_ratio,
            final_receive_age_sec=final_receive_age,
            passed=not reasons,
            reasons=tuple(reasons),
        )
        devices.append(device)
        aggregate_reasons.extend(f"{name}: {reason}" for reason in reasons)

    return B601DecodedActiveReportAcceptance(
        passed=not aggregate_reasons,
        target_hz=raw_acceptance.target_hz,
        minimum_observed_hz=raw_acceptance.minimum_observed_hz,
        maximum_p99_interval_sec=raw_acceptance.maximum_p99_interval_sec,
        maximum_receive_age_sec=receive_age_limit,
        minimum_sample_count=raw_acceptance.minimum_sample_count,
        minimum_decoded_raw_ratio=decoded_raw_ratio_limit,
        capture_duration_sec=duration,
        short_active_report_frame_count=(
            raw_acceptance.short_active_report_frame_count
        ),
        expected_feedback_ids=raw_acceptance.expected_feedback_ids,
        observed_feedback_ids=raw_acceptance.observed_feedback_ids,
        devices=tuple(devices),
        reasons=tuple(aggregate_reasons),
    )
