"""Fail-closed acceptance for B601 RobStride active-report transport timing.

The evaluator consumes the existing passive SocketCAN capture summary.  It
only judges whether complete type-2 operation-status frames arrived at the
requested cadence for each selected device.  It deliberately does not decode
or admit those frames as control state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .backend import B601JointConfig


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
    short_operation_status_frame_count: int
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
            "short_operation_status_frame_count": (
                self.short_operation_status_frame_count
            ),
            "expected_feedback_ids": self.expected_feedback_ids,
            "observed_feedback_ids": self.observed_feedback_ids,
            "devices": tuple(device.as_dict() for device in self.devices),
            "reasons": self.reasons,
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
    """Evaluate type-2 receive cadence for every selected motor."""

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
        "robstride_operation_status_per_device_feedback",
        {},
    )
    raw_hosts = capture.get("robstride_operation_status_frames_by_host_id", {})
    if not isinstance(raw_per_device, Mapping) or not isinstance(raw_hosts, Mapping):
        raise TypeError("capture summary has malformed operation-status mappings")
    short_count = int(capture.get("robstride_short_operation_status_frame_count", 0))
    if short_count < 0:
        raise ValueError("short operation-status frame count must be non-negative")

    observed_feedback_ids: set[int] = set()
    for raw_host in raw_hosts:
        try:
            observed_feedback_ids.add(int(str(raw_host), 0))
        except ValueError as exc:
            raise ValueError(f"invalid operation-status host id: {raw_host}") from exc

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
            reasons.append("no complete type-2 operation-status frames observed")
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
                f"p99_interval_sec={p99_interval:.6f} exceeds "
                f"maximum={maximum_p99:.6f}"
            )
        if joint.feedback_id not in observed_feedback_ids:
            reasons.append(f"expected feedback_id=0x{joint.feedback_id:02x} not observed")
        if short_count:
            reasons.append(
                f"capture contains {short_count} short operation-status frame(s)"
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
        short_operation_status_frame_count=short_count,
        expected_feedback_ids=expected_feedback_ids,
        observed_feedback_ids=tuple(sorted(observed_feedback_ids)),
        devices=tuple(devices),
        reasons=tuple(aggregate_reasons),
    )
