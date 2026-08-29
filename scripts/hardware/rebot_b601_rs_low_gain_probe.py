"""Guarded, human-run B601-RS low-gain gravity/damping commissioning.

This probe is intentionally a slow, single-host wired commissioning tool.  It
reuses the AgenticRobot B601 backend, the pinned Pinocchio model, and the
upstream reBotArm MIT argument order.  It is not a haptic servo, a force
sensor, or a hardware-readiness claim.

The software owns the complete low-gain torque composition:

``tau = gravity_scale * g(q) + kp * (q_ref - q) + kd * (dq_ref - dq) + J.T F``

``J.T F`` is zero in the joint ``hold`` and ``slow-step`` modes.  The guarded
``cartesian-hold`` mode anchors the current ``gripper_end`` position after a
fresh post-enable snapshot and uses the existing translational virtual spring
and world-aligned wrench mapping.  It controls no absolute Cartesian pose and
does not command orientation.

The dispatched RobStride MIT ``kp`` and ``kd`` are always zero, so the final
software torque is the only PD/derivative contribution that this probe owns.
The software then applies a final absolute torque bound and slew-rate bound
before calling :meth:`B601RSBackend.send_mit`.  Feedback uses the existing
explicit ``mechPos`` request path; its source freshness remains unverified by
MotorBridge 0.4.9 and is reported as such.

This module is motion-capable only after all of ``--allow-motion``,
``--auto-disable`` and the exact confirmation string are supplied.  The
``feedback-stability`` mode is a separate observation-only path: it uses the
same explicit ``mechPos`` reads with ``allow_motion=False`` and never changes
the vendor mode or enable state.  Automated tests exercise the pure helpers
only; they never construct a hardware backend.

After MIT mode setup and enable, the active path first requires four consecutive
complete, in-limit, near-zero-velocity feedback snapshots.  No torque command
is sent during this arming-settle gate; the first torque reference is anchored
to the final settled snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np

# Direct ``python scripts/hardware/...`` invocation places the script directory
# on sys.path rather than the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hardware.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601CommandReceipt,
    B601HardwareError,
    B601ModeReceipt,
    B601RSBackend,
    B601RSBackendConfig,
    FrequencyLedger,
    evaluate_frequency_target,
)
from robot_motion.rebot_b601_rs import (
    B601MitControlProfile,
    B601MitReference,
    B601MitTorqueBreakdown,
    B601PinocchioModel,
    B601TranslationalVirtualSpring,
    compute_b601_mit_torque,
    map_world_aligned_ee_wrench_to_joint_torque,
)

CONFIRMATION = "B601_RS_LOW_GAIN_MIT_GRAVITY_DAMPING"
FEEDBACK_STABILITY_CONFIRMATION = "B601_RS_FEEDBACK_STABILITY_READ_ONLY"
_ZERO_SIX = (0.0,) * 6
_ALL_JOINTS = B601_RS_ARM_JOINT_NAMES
_FAULT_HOLD_JOINTS = ("joint2", "joint3")
_NON_FAULT_HOLD_JOINTS = tuple(
    name for name in _ALL_JOINTS if name not in _FAULT_HOLD_JOINTS
)
_DEFAULT_RATE_HZ = 20.0
_DEFAULT_DURATION_SEC = 8.0
_DEFAULT_WARMUP_SEC = 2.0
_DEFAULT_FREEZE_SEC = 1.0
_DEFAULT_KP = 0.5
_DEFAULT_KD = 0.5
_DEFAULT_GRAVITY_SCALE = 1.0
_DEFAULT_TORQUE_LIMIT_SCALE = 0.25
_DEFAULT_TORQUE_RISE_TIME_SEC = 1.0
_DEFAULT_STEP_RAD = 0.05
_DEFAULT_STEP_SPEED_RAD_S = 0.02
_MAX_STEP_RAD = math.radians(20.0)
_DEFAULT_MIN_STEP_RESPONSE_RATIO = 0.10
_DEFAULT_MAX_STEP_RESPONSE_RATIO = 1.50
_DEFAULT_MIN_STEP_RESPONSE_RAD = math.radians(0.1)
_DEFAULT_MAX_SUPPORT_DRIFT_RAD = math.radians(0.5)
_DEFAULT_VELOCITY_ABORT_RAD_S = 0.10
_DEFAULT_MAX_ERROR_RAD = 0.15
_DEFAULT_FAULT_HOLD_MAX_AGE_SEC = 0.15
_DEFAULT_FAULT_HOLD_MAX_VELOCITY_RAD_S = 0.02
_DEFAULT_FAULT_HOLD_MAX_REBASE_RAD = 0.05
_DEFAULT_ARMING_SETTLE_SAMPLES = 4
_DEFAULT_CARTESIAN_STIFFNESS_N_PER_M = 20.0
_DEFAULT_CARTESIAN_DAMPING_N_S_PER_M = 1.0
_DEFAULT_CARTESIAN_MAX_FORCE_N = 2.0
_DEFAULT_CARTESIAN_MAX_DISPLACEMENT_M = 0.02
_DEFAULT_CARTESIAN_OVERLAY_LIMIT_SCALE = 0.25


class LowGainProbeError(RuntimeError):
    """A guarded probe precondition, safety stop, or vendor operation failed."""


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a finite percentile for a non-empty scalar sequence."""

    normalized = tuple(_finite(value, name="percentile_value") for value in values)
    if not normalized:
        raise ValueError("percentile values must not be empty")
    return float(np.percentile(np.asarray(normalized, dtype=float), percentile))


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _six(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    try:
        normalized = tuple(_finite(value, name=name) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain six finite values") from exc
    if len(normalized) != 6:
        raise ValueError(f"{name} must contain six finite values")
    return normalized


def _positive(value: object, *, name: str) -> float:
    normalized = _finite(value, name=name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _bounded(
    value: object,
    *,
    name: str,
    lower: float,
    upper: float,
    lower_inclusive: bool = True,
) -> float:
    normalized = _finite(value, name=name)
    if (normalized < lower if lower_inclusive else normalized <= lower) or normalized > upper:
        qualifier = "[" if lower_inclusive else "("
        raise ValueError(f"{name} must be in {qualifier}{lower}, {upper}]")
    return normalized


def _mode_receipt_payload(receipt: B601ModeReceipt | None) -> dict[str, object] | None:
    if receipt is None:
        return None
    return {
        "operation": receipt.operation,
        "requested_joint_names": receipt.requested_joint_names,
        "completed_joint_names": receipt.completed_joint_names,
        "failed_joint_names": receipt.failed_joint_names,
        "errors": receipt.errors,
        "completed": receipt.completed,
    }


def _command_receipt_payload(
    receipt: B601CommandReceipt | None,
) -> dict[str, object] | None:
    if receipt is None:
        return None
    return {
        "sequence": receipt.sequence,
        "mode": receipt.mode,
        "frame_id": receipt.frame_id,
        "requested_at_sec": receipt.requested_at_sec,
        "completed_at_sec": receipt.completed_at_sec,
        "requested_joint_names": receipt.requested_joint_names,
        "sent_joint_names": receipt.sent_joint_names,
        "failed_joint_names": receipt.failed_joint_names,
        "errors": receipt.errors,
        "accepted": receipt.accepted,
        "source_sequence": receipt.source_sequence,
    }


def _validate_fault_hold_snapshot(
    qpos_rad: Sequence[float] | None,
    *,
    qvel_rad_s: Sequence[float] | None,
    observed_age_sec: float,
    position_limits_rad: Sequence[tuple[float, float]],
    max_age_sec: float = _DEFAULT_FAULT_HOLD_MAX_AGE_SEC,
    max_velocity_rad_s: float = _DEFAULT_FAULT_HOLD_MAX_VELOCITY_RAD_S,
    fault_qpos_rad: Sequence[float] | None = None,
    fault_qvel_rad_s: Sequence[float] | None = None,
    max_rebase_rad: float = _DEFAULT_FAULT_HOLD_MAX_REBASE_RAD,
) -> tuple[float, ...]:
    """Admit only a fresh, stationary, in-limit snapshot for protective hold.

    The returned tuple is the immutable q-reference for the joint2/joint3
    latch.  A missing or invalid snapshot is never substituted with zeros or
    with the most recent command.  ``observed_age_sec`` is measured by this
    process's monotonic clock; it is deliberately not presented as vendor
    source freshness because MotorBridge 0.4.9 provides no source timestamp.
    """

    if qpos_rad is None:
        raise LowGainProbeError(
            "fault hold requires a complete last-safe position snapshot"
        )
    if qvel_rad_s is None:
        raise LowGainProbeError(
            "fault hold requires a complete last-safe velocity observation"
        )
    q = _six(qpos_rad, name="fault_hold_qpos_rad")
    qvel = _six(qvel_rad_s, name="fault_hold_qvel_rad_s")
    limits = tuple(
        (float(lower), float(upper)) for lower, upper in position_limits_rad
    )
    if len(limits) != 6 or any(
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower >= upper
        for lower, upper in limits
    ):
        raise LowGainProbeError("fault hold received invalid position limits")
    age = _finite(observed_age_sec, name="fault_hold_observed_age_sec")
    max_age = _bounded(
        max_age_sec,
        name="fault_hold_max_age_sec",
        lower=0.01,
        upper=1.0,
    )
    if age < 0.0 or age > max_age:
        raise LowGainProbeError(
            "fault hold snapshot is stale: "
            f"age={age:.6f}s > max={max_age:.6f}s"
        )
    max_velocity = _bounded(
        max_velocity_rad_s,
        name="fault_hold_max_velocity_rad_s",
        lower=0.0001,
        upper=0.5,
        lower_inclusive=False,
    )
    observed_max_velocity = max(abs(value) for value in qvel)
    if observed_max_velocity > max_velocity:
        raise LowGainProbeError(
            "fault hold snapshot is moving: "
            f"max_abs_velocity={observed_max_velocity:.6f} rad/s > "
            f"{max_velocity:.6f} rad/s"
        )
    for index, (value, (lower, upper)) in enumerate(zip(q, limits, strict=True)):
        if not lower <= value <= upper:
            raise LowGainProbeError(
                f"fault hold qpos_rad[{index}]={value:.6f} is outside "
                f"URDF limits [{lower:.6f}, {upper:.6f}]"
            )
    if fault_qpos_rad is not None:
        fault_q = _six(fault_qpos_rad, name="fault_qpos_rad")
        if fault_qvel_rad_s is None:
            raise LowGainProbeError(
                "fault snapshot velocity is unavailable; refusing protective rebase"
            )
        fault_qvel = _six(fault_qvel_rad_s, name="fault_qvel_rad_s")
        fault_max_velocity = max(abs(value) for value in fault_qvel)
        if fault_max_velocity > max_velocity:
            raise LowGainProbeError(
                "fault snapshot is moving too quickly for a safe rebase: "
                f"max_abs_velocity={fault_max_velocity:.6f} rad/s > "
                f"{max_velocity:.6f} rad/s"
            )
        for index, (value, (lower, upper)) in enumerate(
            zip(fault_q, limits, strict=True)
        ):
            if not lower <= value <= upper:
                raise LowGainProbeError(
                    f"fault qpos_rad[{index}]={value:.6f} is outside URDF limits"
                )
        max_rebase = _bounded(
            max_rebase_rad,
            name="fault_hold_max_rebase_rad",
            lower=0.001,
            upper=0.5,
            lower_inclusive=False,
        )
        for joint in _FAULT_HOLD_JOINTS:
            index = _ALL_JOINTS.index(joint)
            delta = abs(fault_q[index] - q[index])
            if delta > max_rebase:
                raise LowGainProbeError(
                    f"fault hold rebase for {joint} is too large: "
                    f"{delta:.6f} rad > {max_rebase:.6f} rad"
                )
    return q


def _evaluate_fault_hold_candidate(
    qpos_rad: Sequence[float],
    *,
    qvel_rad_s: Sequence[float],
    velocity_abort_rad_s: float,
    observed_age_sec: float,
    position_limits_rad: Sequence[tuple[float, float]],
    max_age_sec: float,
    max_velocity_rad_s: float,
    max_rebase_rad: float,
) -> tuple[tuple[float, ...] | None, str | None]:
    """Evaluate whether one control sample may replace ``last_safe``.

    Candidate admission is deliberately separate from the normal control-loop
    velocity watchdog. A sample may remain legal for bounded control while
    being unsuitable as a future joint2/joint3 fault-hold reference. In that
    case the immutable last-safe reference is retained; the strict validator
    is still rerun if a protective hold is actually requested.
    """

    qvel = _six(qvel_rad_s, name="control_sample_qvel_rad_s")
    velocity_abort = _positive(
        velocity_abort_rad_s,
        name="velocity_abort_rad_s",
    )
    max_velocity = max(abs(value) for value in qvel)
    if max_velocity > velocity_abort:
        raise LowGainProbeError(
            f"velocity abort: max_abs_fd_velocity={max_velocity:.6f} "
            f"> {velocity_abort:.6f} rad/s"
        )
    try:
        accepted = _validate_fault_hold_snapshot(
            qpos_rad,
            qvel_rad_s=qvel,
            observed_age_sec=observed_age_sec,
            position_limits_rad=position_limits_rad,
            max_age_sec=max_age_sec,
            max_velocity_rad_s=max_velocity_rad_s,
            max_rebase_rad=max_rebase_rad,
        )
    except LowGainProbeError as exc:
        return None, str(exc)
    return accepted, None


def _validate_arming_settle_transition(
    current_qpos_rad: Sequence[float] | None,
    *,
    previous_qpos_rad: Sequence[float],
    current_timestamp_sec: float,
    previous_timestamp_sec: float,
    position_limits_rad: Sequence[tuple[float, float]],
    max_velocity_rad_s: float = _DEFAULT_FAULT_HOLD_MAX_VELOCITY_RAD_S,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate one post-enable feedback transition before torque is allowed.

    The transition uses consecutive host receive timestamps and the same
    fresh/in-limit/near-zero-velocity validator as the protective hold.  This
    helper only validates observations; it never sends a motor command.
    """

    if current_qpos_rad is None:
        raise LowGainProbeError(
            "arming settle requires a complete feedback position snapshot"
        )
    current = _six(current_qpos_rad, name="arming_settle_qpos_rad")
    previous = _six(previous_qpos_rad, name="arming_settle_previous_qpos_rad")
    current_timestamp = _finite(
        current_timestamp_sec,
        name="arming_settle_current_timestamp_sec",
    )
    previous_timestamp = _finite(
        previous_timestamp_sec,
        name="arming_settle_previous_timestamp_sec",
    )
    dt = current_timestamp - previous_timestamp
    if dt <= 0.0:
        raise LowGainProbeError(
            "arming settle feedback timestamp did not advance: "
            f"dt={dt:.6f}s"
        )
    qvel = tuple(
        (current_value - previous_value) / dt
        for current_value, previous_value in zip(current, previous, strict=True)
    )
    accepted = _validate_fault_hold_snapshot(
        current,
        qvel_rad_s=qvel,
        observed_age_sec=0.0,
        position_limits_rad=position_limits_rad,
        max_velocity_rad_s=max_velocity_rad_s,
    )
    return accepted, qvel


def _validate_protective_hold_feedback(
    qpos_rad: Sequence[float] | None,
    *,
    qvel_rad_s: Sequence[float] | None,
    hold_qpos_rad: Sequence[float],
    observed_age_sec: float,
    position_limits_rad: Sequence[tuple[float, float]],
    max_age_sec: float = _DEFAULT_FAULT_HOLD_MAX_AGE_SEC,
    max_velocity_rad_s: float = _DEFAULT_FAULT_HOLD_MAX_VELOCITY_RAD_S,
    max_rebase_rad: float = _DEFAULT_FAULT_HOLD_MAX_REBASE_RAD,
) -> tuple[float, ...]:
    """Validate one complete feedback sample while the protective hold runs.

    ``hold_qpos_rad`` is immutable: this helper never updates it.  Feedback
    is used only as a measured state for damping and as a drift watchdog.  A
    held joint that leaves the small admitted neighbourhood is escalated to
    the caller's full-disable fallback rather than being silently rebased.
    """

    current = _validate_fault_hold_snapshot(
        qpos_rad,
        qvel_rad_s=qvel_rad_s,
        observed_age_sec=observed_age_sec,
        position_limits_rad=position_limits_rad,
        max_age_sec=max_age_sec,
        max_velocity_rad_s=max_velocity_rad_s,
        max_rebase_rad=max_rebase_rad,
    )
    hold_q = _six(hold_qpos_rad, name="protective_hold_qpos_rad")
    max_rebase = _bounded(
        max_rebase_rad,
        name="fault_hold_max_rebase_rad",
        lower=0.001,
        upper=0.5,
        lower_inclusive=False,
    )
    for joint in _FAULT_HOLD_JOINTS:
        index = _ALL_JOINTS.index(joint)
        delta = abs(current[index] - hold_q[index])
        if delta > max_rebase:
            raise LowGainProbeError(
                f"protective hold drift for {joint} is too large: "
                f"{delta:.6f} rad > {max_rebase:.6f} rad"
            )
    return current


def resolve_active_joint_names(
    target_joint: str,
    support_joints: Sequence[str] = _FAULT_HOLD_JOINTS,
) -> tuple[str, ...]:
    """Resolve the explicit command set for a low-gain probe.

    The target is the only joint whose reference may change in ``slow-step``.
    Joint2 and joint3 remain mandatory support joints so the first single-joint
    check cannot accidentally enable a wrist-only subset.  The selected set is
    still dispatched explicitly; ``--joint`` no longer implies six-joint TX.
    """

    target = str(target_joint).strip()
    if target not in _ALL_JOINTS:
        raise ValueError(f"unknown B601-RS joint: {target_joint}")
    try:
        support = tuple(str(name).strip() for name in support_joints)
    except TypeError as exc:
        raise ValueError("support_joints must be a sequence of joint names") from exc
    if any(name not in _ALL_JOINTS for name in support):
        unknown = sorted(set(support).difference(_ALL_JOINTS))
        raise ValueError(f"unknown B601-RS support joints: {unknown}")
    names = tuple(dict.fromkeys((target, *support)))
    if not set(_FAULT_HOLD_JOINTS).issubset(names):
        raise ValueError(
            "joint2 and joint3 are mandatory support joints for the fault hold"
        )
    return names


def _validate_cli(args: argparse.Namespace) -> None:
    if args.mode == "feedback-stability":
        if args.allow_motion:
            raise LowGainProbeError(
                "feedback-stability rejects --allow-motion; it is observation-only"
            )
        if args.auto_disable:
            raise LowGainProbeError(
                "feedback-stability rejects --auto-disable; it does not change motor state"
            )
        if args.confirmation != FEEDBACK_STABILITY_CONFIRMATION:
            raise LowGainProbeError(
                "feedback-stability requires "
                f"--confirmation {FEEDBACK_STABILITY_CONFIRMATION}"
            )
        if not args.config:
            raise LowGainProbeError("feedback-stability requires --config")
        if not args.rebot_source_root:
            raise LowGainProbeError(
                "feedback-stability requires --rebot-source-root"
            )
        _bounded(args.rate_hz, name="rate_hz", lower=1.0, upper=40.0)
        _bounded(args.duration_sec, name="duration_sec", lower=0.1, upper=60.0)
        _bounded(
            args.fault_hold_max_velocity_rad_s,
            name="fault_hold_max_velocity_rad_s",
            lower=0.0001,
            upper=0.5,
            lower_inclusive=False,
        )
        return
    if not args.allow_motion:
        raise LowGainProbeError("low-gain mode requires --allow-motion")
    if not args.auto_disable:
        raise LowGainProbeError("low-gain mode requires explicit --auto-disable")
    if args.confirmation != CONFIRMATION:
        raise LowGainProbeError(
            f"low-gain mode requires --confirmation {CONFIRMATION}"
        )
    if not args.config:
        raise LowGainProbeError("low-gain mode requires --config")
    if not args.rebot_source_root:
        raise LowGainProbeError("low-gain mode requires --rebot-source-root")
    if args.mode not in ("hold", "slow-step", "cartesian-hold"):
        raise ValueError("mode must be hold, slow-step, or cartesian-hold")
    _bounded(args.rate_hz, name="rate_hz", lower=1.0, upper=40.0)
    _bounded(args.duration_sec, name="duration_sec", lower=0.1, upper=60.0)
    _bounded(args.warmup_sec, name="warmup_sec", lower=0.1, upper=15.0)
    _bounded(args.freeze_sec, name="freeze_sec", lower=0.1, upper=10.0)
    _bounded(args.kp, name="kp", lower=0.0, upper=5.0)
    _bounded(args.kd, name="kd", lower=0.0, upper=5.0)
    target_kp = getattr(args, "target_kp", None)
    target_kd = getattr(args, "target_kd", None)
    if target_kp is not None:
        _bounded(target_kp, name="target_kp", lower=0.0, upper=5.0)
    if target_kd is not None:
        _bounded(target_kd, name="target_kd", lower=0.0, upper=5.0)
    _bounded(args.gravity_scale, name="gravity_scale", lower=0.0, upper=1.0)
    _bounded(
        args.torque_limit_scale,
        name="torque_limit_scale",
        lower=0.01,
        upper=0.5,
        lower_inclusive=False,
    )
    _bounded(
        args.torque_rise_time_sec,
        name="torque_rise_time_sec",
        lower=0.1,
        upper=10.0,
    )
    _bounded(
        args.velocity_abort_rad_s,
        name="velocity_abort_rad_s",
        lower=0.01,
        upper=0.5,
    )
    _bounded(
        args.max_error_rad,
        name="max_error_rad",
        lower=0.01,
        upper=0.5,
    )
    _bounded(args.step_rad, name="step_rad", lower=0.0, upper=_MAX_STEP_RAD)
    _bounded(
        args.step_speed_rad_s,
        name="step_speed_rad_s",
        lower=0.001,
        upper=0.05,
    )
    _bounded(
        getattr(args, "fault_hold_max_age_sec", _DEFAULT_FAULT_HOLD_MAX_AGE_SEC),
        name="fault_hold_max_age_sec",
        lower=0.01,
        upper=1.0,
    )
    _bounded(
        getattr(
            args,
            "fault_hold_max_velocity_rad_s",
            _DEFAULT_FAULT_HOLD_MAX_VELOCITY_RAD_S,
        ),
        name="fault_hold_max_velocity_rad_s",
        lower=0.0001,
        upper=0.5,
        lower_inclusive=False,
    )
    _bounded(
        getattr(args, "fault_hold_max_rebase_rad", _DEFAULT_FAULT_HOLD_MAX_REBASE_RAD),
        name="fault_hold_max_rebase_rad",
        lower=0.001,
        upper=0.5,
        lower_inclusive=False,
    )
    _bounded(
        getattr(
            args,
            "cartesian_stiffness_n_per_m",
            _DEFAULT_CARTESIAN_STIFFNESS_N_PER_M,
        ),
        name="cartesian_stiffness_n_per_m",
        lower=0.0,
        upper=100.0,
    )
    _bounded(
        getattr(
            args,
            "cartesian_damping_n_s_per_m",
            _DEFAULT_CARTESIAN_DAMPING_N_S_PER_M,
        ),
        name="cartesian_damping_n_s_per_m",
        lower=0.0,
        upper=10.0,
    )
    _bounded(
        getattr(args, "cartesian_max_force_n", _DEFAULT_CARTESIAN_MAX_FORCE_N),
        name="cartesian_max_force_n",
        lower=0.1,
        upper=5.0,
    )
    _bounded(
        getattr(
            args,
            "cartesian_max_displacement_m",
            _DEFAULT_CARTESIAN_MAX_DISPLACEMENT_M,
        ),
        name="cartesian_max_displacement_m",
        lower=0.001,
        upper=0.05,
        lower_inclusive=False,
    )
    _bounded(
        getattr(
            args,
            "cartesian_overlay_limit_scale",
            _DEFAULT_CARTESIAN_OVERLAY_LIMIT_SCALE,
        ),
        name="cartesian_overlay_limit_scale",
        lower=0.01,
        upper=0.5,
        lower_inclusive=False,
    )
    if args.mode == "slow-step" and args.step_rad <= 0.0:
        raise ValueError("slow-step mode requires step_rad > 0")
    active_joint_names = resolve_active_joint_names(
        args.joint,
        getattr(args, "support_joints", _FAULT_HOLD_JOINTS),
    )
    if args.mode == "cartesian-hold":
        if args.kp != 0.0 or target_kp not in (None, 0.0):
            raise ValueError(
                "cartesian-hold requires --kp 0 and no non-zero --target-kp "
                "to avoid joint-position stiffness"
            )
        if active_joint_names != _ALL_JOINTS:
            raise ValueError(
                "cartesian-hold requires the canonical joint1..joint6 active set; "
                "use joint1 as target and add joint4, joint5, and joint6 as support joints"
            )


def _resolve_model_and_limits(
    source_root: Path,
) -> tuple[B601PinocchioModel, tuple[tuple[float, float], ...]]:
    try:
        model = B601PinocchioModel.from_rebot_source_root(source_root)
        lower, upper = model.arm_position_limits()
    except Exception as exc:
        raise LowGainProbeError(
            f"cannot resolve pinned B601-RS URDF limits: {exc}"
        ) from exc
    limits = tuple(
        (float(lo), float(hi)) for lo, hi in zip(lower, upper, strict=True)
    )
    if len(limits) != 6 or any(
        not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi
        for lo, hi in limits
    ):
        raise LowGainProbeError("pinned B601-RS URDF returned invalid position limits")
    return model, limits


def _resolve_effort_limits(
    urdf_path: Path,
    *,
    scale: float,
) -> tuple[float, ...]:
    """Read URDF effort ceilings and apply a conservative commissioning scale."""

    scale = _bounded(scale, name="torque_limit_scale", lower=0.01, upper=0.5, lower_inclusive=False)
    try:
        root = ET.fromstring(urdf_path.read_text(encoding="utf-8"))
    except (OSError, ET.ParseError) as exc:
        raise LowGainProbeError(f"cannot read pinned B601-RS URDF effort limits: {exc}") from exc
    by_name: dict[str, float] = {}
    for joint in root.findall("joint"):
        name = str(joint.get("name", "")).strip()
        limit = joint.find("limit")
        if not name or limit is None or limit.get("effort") is None:
            continue
        try:
            effort = _finite(limit.get("effort"), name=f"{name}.effort")
        except ValueError:
            continue
        if effort > 0.0:
            by_name[name] = effort
    missing = [name for name in _ALL_JOINTS if name not in by_name]
    if missing:
        raise LowGainProbeError(
            f"pinned B601-RS URDF has no positive effort for {missing}"
        )
    return tuple(by_name[name] * scale for name in _ALL_JOINTS)


def summarize_feedback_stability(
    positions_rad: Sequence[Sequence[float]],
    timestamps_sec: Sequence[float],
    *,
    stationarity_velocity_limit_rad_s: float = _DEFAULT_FAULT_HOLD_MAX_VELOCITY_RAD_S,
) -> dict[str, object]:
    """Summarize synchronous ``mechPos`` finite-difference noise.

    This is deliberately a characterization helper, not a controller.  It
    computes the same first-order finite difference used by the guarded active
    probe, but does not smooth, clip, or reinterpret an excursion.  The
    stationarity gate is reported for review; it never authorizes a motion
    command or changes the protective-hold threshold.
    """

    try:
        samples = tuple(_six(sample, name="positions_rad") for sample in positions_rad)
    except TypeError as exc:
        raise ValueError("positions_rad must be a sequence of six-joint samples") from exc
    times = tuple(_finite(value, name="timestamp_sec") for value in timestamps_sec)
    if len(samples) != len(times):
        raise ValueError("positions_rad and timestamps_sec must have equal length")
    if len(samples) < 2:
        raise ValueError("feedback stability requires at least two complete snapshots")
    if any(current <= previous for previous, current in pairwise(times)):
        raise ValueError("timestamps_sec must be strictly increasing")
    limit = _bounded(
        stationarity_velocity_limit_rad_s,
        name="stationarity_velocity_limit_rad_s",
        lower=0.0001,
        upper=0.5,
        lower_inclusive=False,
    )
    deltas = np.diff(np.asarray(samples, dtype=float), axis=0)
    intervals = np.diff(np.asarray(times, dtype=float))
    velocities = deltas / intervals[:, None]
    per_joint: dict[str, dict[str, object]] = {}
    threshold_exceed_count = 0
    for index, joint_name in enumerate(_ALL_JOINTS):
        abs_step = np.abs(deltas[:, index])
        abs_velocity = np.abs(velocities[:, index])
        exceeds = abs_velocity > limit
        count = int(np.count_nonzero(exceeds))
        threshold_exceed_count += count
        per_joint[joint_name] = {
            "sample_count": len(samples),
            "interval_count": len(intervals),
            "position_first_rad": float(samples[0][index]),
            "position_last_rad": float(samples[-1][index]),
            "max_abs_drift_rad": float(np.max(np.abs(np.asarray(samples)[:, index] - samples[0][index]))),
            "max_abs_step_rad": float(np.max(abs_step)),
            "p50_abs_step_rad": _percentile(abs_step, 50.0),
            "p95_abs_step_rad": _percentile(abs_step, 95.0),
            "p99_abs_step_rad": _percentile(abs_step, 99.0),
            "max_abs_velocity_rad_s": float(np.max(abs_velocity)),
            "p50_abs_velocity_rad_s": _percentile(abs_velocity, 50.0),
            "p95_abs_velocity_rad_s": _percentile(abs_velocity, 95.0),
            "p99_abs_velocity_rad_s": _percentile(abs_velocity, 99.0),
            "stationarity_exceed_count": count,
            "stationarity_exceed_ratio": float(count / len(intervals)),
            "stationarity_passed": count == 0,
        }
    return {
        "sample_count": len(samples),
        "interval_count": len(intervals),
        "stationarity_velocity_limit_rad_s": limit,
        "stationarity_exceed_count": threshold_exceed_count,
        "stationarity_passed": threshold_exceed_count == 0,
        "per_joint": per_joint,
    }


def _run_feedback_stability_probe(
    backend: B601RSBackend,
    *,
    config_path: Path,
    source_root: Path,
    model: B601PinocchioModel,
    position_limits: tuple[tuple[float, float], ...],
    duration_sec: float,
    rate_hz: float,
    stationarity_velocity_limit_rad_s: float,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    max_samples: int | None = None,
) -> dict[str, object]:
    """Capture complete feedback without mode, enable, disable, or TX control."""

    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when provided")
    period = 1.0 / _positive(rate_hz, name="rate_hz")
    started = _finite(clock(), name="clock")
    deadline = started + _positive(duration_sec, name="duration_sec")
    next_deadline = started
    ledger = FrequencyLedger()
    positions: list[tuple[float, ...]] = []
    timestamps: list[float] = []
    errors: list[str] = []
    connected = False
    try:
        backend.connect()
        connected = True
        while clock() < deadline and (
            max_samples is None or len(positions) < max_samples
        ):
            now = _finite(clock(), name="clock")
            if now < next_deadline:
                sleeper(next_deadline - now)
            state = backend.poll_feedback(request_feedback=True)
            if (
                not state.valid
                or getattr(state, "stale", False)
                or state.positions_rad is None
            ):
                errors.append(
                    "feedback snapshot is not complete and valid: "
                    f"{getattr(state, 'fault_reason', None) or 'unknown fault'}"
                )
                break
            try:
                q = _six(state.positions_rad, name="feedback_qpos_rad")
                observed_at = _finite(
                    state.receive_timestamp_sec,
                    name="feedback_receive_timestamp_sec",
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"invalid feedback snapshot: {exc}")
                break
            for index, (value, (lower, upper)) in enumerate(
                zip(q, position_limits, strict=True)
            ):
                if not lower <= value <= upper:
                    errors.append(
                        f"feedback qpos_rad[{index}]={value:.6f} is outside "
                        f"URDF limits [{lower:.6f}, {upper:.6f}]"
                    )
                    break
            if errors:
                break
            positions.append(q)
            timestamps.append(observed_at)
            ledger.mark("feedback", observed_at)
            next_deadline += period
    finally:
        if connected:
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001 - preserve evidence
                errors.append(f"close: {exc}")
    elapsed = max(0.0, _finite(clock(), name="clock") - started)
    payload: dict[str, object] = {
        "status": "OK" if len(positions) >= 2 and not errors else "FAIL",
        "mode": "feedback-stability",
        "confirmation": FEEDBACK_STABILITY_CONFIRMATION,
        "channel": backend.config.channel,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "rebot_source_root": str(source_root),
        "urdf_path": str(model.urdf_path),
        "urdf_sha256": hashlib.sha256(model.urdf_path.read_bytes()).hexdigest(),
        "transport": "motorbridge_wired_local",
        "joint_names": _ALL_JOINTS,
        "active_joint_names": (),
        "support_joint_names": (),
        "motors_enabled_by_probe": False,
        "motion_commands_sent": False,
        "mode_transitions_sent": False,
        "disable_commands_sent": False,
        "feedback_requests_sent": len(positions) * len(_ALL_JOINTS),
        "duration_sec": elapsed,
        "rate_hz": rate_hz,
        "stop_reason": errors[0] if errors else None,
        "errors": tuple(errors),
        "frequency": {"feedback": ledger.summarize(target_hz=rate_hz)["feedback"].as_dict()},
        "stability": (
            summarize_feedback_stability(
                positions,
                timestamps,
                stationarity_velocity_limit_rad_s=stationarity_velocity_limit_rad_s,
            )
            if len(positions) >= 2
            else None
        ),
        "acceptance_status": "CAPTURED" if len(positions) >= 2 and not errors else "FAILED",
        "hardware_acceptance": "NOT_CLAIMED",
        "freshness_verified": False,
        "freshness_semantics": "motorbridge_mechPos_parameter_observation_unverified",
        "samples": [
            {"receive_timestamp_sec": timestamp, "qpos_rad": q}
            for timestamp, q in zip(timestamps, positions, strict=True)
        ],
        "notes": [
            "This mode only requests vendor feedback and reads mechPos; it never changes mode, enables, disables, or sends MIT/POS_VEL control frames.",
            "Run only after the operator has independently made the arm safe; this probe does not alter the existing enable state.",
            "The stationarity gate is diagnostic evidence for protective-hold review, not permission to relax the active safety threshold.",
            "MotorBridge 0.4.9 provides no source timestamp or sequence, so freshness remains unverified.",
        ],
    }
    return payload


def condition_torque(
    requested_nm: Sequence[float],
    previous_nm: Sequence[float],
    *,
    absolute_limits_nm: Sequence[float],
    max_rate_nm_s: Sequence[float],
    dt_sec: float,
) -> tuple[tuple[float, ...], bool, bool]:
    """Apply the final absolute and slew-rate bounds to a six-joint torque."""

    requested = np.asarray(_six(requested_nm, name="requested_nm"), dtype=float)
    previous = np.asarray(_six(previous_nm, name="previous_nm"), dtype=float)
    limits = np.asarray(_six(absolute_limits_nm, name="absolute_limits_nm"), dtype=float)
    rates = np.asarray(_six(max_rate_nm_s, name="max_rate_nm_s"), dtype=float)
    dt = _positive(dt_sec, name="dt_sec")
    if np.any(limits <= 0.0) or np.any(rates <= 0.0):
        raise ValueError("absolute_limits_nm and max_rate_nm_s must be positive")
    clipped = np.clip(requested, -limits, limits)
    torque_limited = not np.array_equal(clipped, requested)
    max_delta = rates * dt
    conditioned = np.clip(clipped, previous - max_delta, previous + max_delta)
    slew_limited = not np.array_equal(conditioned, clipped)
    return tuple(float(value) for value in conditioned), torque_limited, slew_limited


def build_software_profile(
    *,
    kp: float,
    kd: float,
    torque_limits_nm: Sequence[float],
    torque_rise_time_sec: float,
    velocity_limit_rad_s: float,
    overlay_limits_nm: Sequence[float] = _ZERO_SIX,
    target_joint: str | None = None,
    target_kp: float | None = None,
    target_kd: float | None = None,
) -> B601MitControlProfile:
    """Build the low-gain software composition profile used by the probe."""

    kp = _bounded(kp, name="kp", lower=0.0, upper=5.0)
    kd = _bounded(kd, name="kd", lower=0.0, upper=5.0)
    limits = _six(torque_limits_nm, name="torque_limits_nm")
    overlay_limits = _six(overlay_limits_nm, name="overlay_limits_nm")
    if any(value <= 0.0 for value in limits):
        raise ValueError("torque_limits_nm must be positive")
    if any(value < 0.0 for value in overlay_limits):
        raise ValueError("overlay_limits_nm must be non-negative")
    rise_time = _positive(torque_rise_time_sec, name="torque_rise_time_sec")
    velocity_limit = _positive(velocity_limit_rad_s, name="velocity_limit_rad_s")
    position_gains = [kp] * 6
    damping_gains = [kd] * 6
    if target_joint is None:
        if target_kp is not None or target_kd is not None:
            raise ValueError("target_joint is required for target gain overrides")
    else:
        if target_joint not in _ALL_JOINTS:
            raise ValueError(f"unknown B601-RS joint: {target_joint}")
        target_index = _ALL_JOINTS.index(target_joint)
        if target_kp is not None:
            position_gains[target_index] = _bounded(
                target_kp,
                name="target_kp",
                lower=0.0,
                upper=5.0,
            )
        if target_kd is not None:
            damping_gains[target_index] = _bounded(
                target_kd,
                name="target_kd",
                lower=0.0,
                upper=5.0,
            )
    return B601MitControlProfile(
        name="b601_rs_low_gain_software_gravity_damping",
        position_kp_nm_per_rad=tuple(position_gains),
        damping_kd_nm_s_per_rad=tuple(damping_gains),
        max_abs_torque_nm=limits,
        max_torque_rate_nm_s=tuple(value / rise_time for value in limits),
        max_abs_overlay_torque_nm=overlay_limits,
        max_abs_reference_velocity_rad_s=(max(velocity_limit, 0.05),) * 6,
        max_abs_state_velocity_rad_s=(max(velocity_limit, 0.05),) * 6,
        metadata={
            "hardware_safety_claim": False,
            "torque_owner": "software_composed_tau; vendor_kp_kd_zero",
            "torque_limit_source": "pinned_urdf_effort_scaled",
            "torque_rise_time_sec": rise_time,
            "target_joint": target_joint,
        },
    )


def build_reference(
    *,
    sequence: int,
    timestamp_sec: float,
    qpos_rad: Sequence[float],
    qvel_rad_s: Sequence[float] = _ZERO_SIX,
    joint_torque_overlay_nm: Sequence[float] = _ZERO_SIX,
) -> B601MitReference:
    """Build a validated local-clock MIT reference."""

    timestamp = _finite(timestamp_sec, name="timestamp_sec")
    return B601MitReference(
        sequence=int(sequence),
        source_timestamp_sec=timestamp,
        receive_timestamp_sec=timestamp,
        qpos_rad=_six(qpos_rad, name="qpos_rad"),
        qvel_rad_s=_six(qvel_rad_s, name="qvel_rad_s"),
        joint_torque_overlay_nm=_six(
            joint_torque_overlay_nm,
            name="joint_torque_overlay_nm",
        ),
    )


def compute_software_torque_breakdown(
    model: B601PinocchioModel,
    *,
    reference: B601MitReference,
    measured_qpos_rad: Sequence[float],
    measured_qvel_rad_s: Sequence[float],
    profile: B601MitControlProfile,
    gravity_scale: float,
) -> B601MitTorqueBreakdown:
    """Compose gravity, joint PD, and bounded overlay torque in software."""

    scale = _bounded(gravity_scale, name="gravity_scale", lower=0.0, upper=1.0)
    gravity = tuple(
        scale * float(value)
        for value in model.arm_generalized_gravity(measured_qpos_rad)
    )
    return compute_b601_mit_torque(
        reference,
        profile,
        measured_qpos_rad=measured_qpos_rad,
        measured_qvel_rad_s=measured_qvel_rad_s,
        gravity_torque_nm=gravity,
    )


def compute_software_torque(
    model: B601PinocchioModel,
    *,
    reference: B601MitReference,
    measured_qpos_rad: Sequence[float],
    measured_qvel_rad_s: Sequence[float],
    profile: B601MitControlProfile,
    gravity_scale: float,
) -> tuple[float, ...]:
    """Compatibility wrapper returning only the requested total torque."""

    return compute_software_torque_breakdown(
        model,
        reference=reference,
        measured_qpos_rad=measured_qpos_rad,
        measured_qvel_rad_s=measured_qvel_rad_s,
        profile=profile,
        gravity_scale=gravity_scale,
    ).requested_torque_nm


def compute_cartesian_impedance_overlay(
    model: B601PinocchioModel,
    spring: B601TranslationalVirtualSpring,
    *,
    qpos_rad: Sequence[float],
    qvel_rad_s: Sequence[float],
    max_displacement_m: float,
) -> tuple[tuple[float, ...], dict[str, object]]:
    """Evaluate one current-anchor translational impedance sample.

    This helper reuses the model's LOCAL_WORLD_ALIGNED Jacobian and the shared
    ``J.T @ wrench`` implementation.  It rejects excessive TCP displacement
    before returning any torque overlay and verifies the twist/wrench power
    identity whenever joint velocity is available.
    """

    qpos = _six(qpos_rad, name="qpos_rad")
    qvel = _six(qvel_rad_s, name="qvel_rad_s")
    displacement_limit = _bounded(
        max_displacement_m,
        name="cartesian_max_displacement_m",
        lower=0.001,
        upper=0.05,
        lower_inclusive=False,
    )
    transform = np.asarray(model.forward_kinematics(qpos), dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise LowGainProbeError("Cartesian impedance FK returned an invalid transform")
    jacobian = np.asarray(model.arm_jacobian(qpos), dtype=float)
    if jacobian.shape != (6, 6) or not np.all(np.isfinite(jacobian)):
        raise LowGainProbeError("Cartesian impedance Jacobian is invalid")
    translational_jacobian = jacobian[:3]
    singular_values = np.linalg.svd(translational_jacobian, compute_uv=False)
    if singular_values.shape != (3,) or not np.all(np.isfinite(singular_values)):
        raise LowGainProbeError("Cartesian impedance singular values are invalid")
    position = transform[:3, 3]
    linear_velocity = translational_jacobian @ np.asarray(qvel, dtype=float)
    spring_result = spring.evaluate(position, linear_velocity)
    displacement_norm = float(np.linalg.norm(spring_result.displacement_m))
    if displacement_norm > displacement_limit:
        raise LowGainProbeError(
            "Cartesian displacement abort: "
            f"norm={displacement_norm:.6f} m > {displacement_limit:.6f} m"
        )
    mapping = map_world_aligned_ee_wrench_to_joint_torque(
        model,
        qpos,
        spring_result.wrench,
        qvel_rad_s=qvel,
    )
    if mapping.cartesian_power_w is None or mapping.joint_power_w is None:
        raise LowGainProbeError("Cartesian impedance power evidence is unavailable")
    power_error_w = abs(mapping.cartesian_power_w - mapping.joint_power_w)
    power_tolerance_w = 1e-9 + 1e-9 * max(
        abs(mapping.cartesian_power_w),
        abs(mapping.joint_power_w),
    )
    if power_error_w > power_tolerance_w:
        raise LowGainProbeError(
            "Cartesian wrench mapping violated power invariance: "
            f"error={power_error_w:.12g} W"
        )
    return mapping.joint_torque_nm, {
        "frame_id": spring_result.wrench.frame_id,
        "position_m": tuple(float(value) for value in position),
        "linear_velocity_m_s": tuple(float(value) for value in linear_velocity),
        "displacement_m": spring_result.displacement_m,
        "displacement_norm_m": displacement_norm,
        "force_n": spring_result.wrench.force_n,
        "raw_force_n": spring_result.raw_force_n,
        "force_limited": spring_result.force_limited,
        "translational_jacobian_singular_values": tuple(
            float(value) for value in singular_values
        ),
        "cartesian_power_w": mapping.cartesian_power_w,
        "joint_power_w": mapping.joint_power_w,
        "power_error_w": power_error_w,
    }


def plan_reference(
    initial_qpos_rad: Sequence[float],
    *,
    mode: str,
    joint: str,
    elapsed_sec: float,
    step_rad: float,
    step_speed_rad_s: float,
    step_direction: int,
    position_limits_rad: Sequence[tuple[float, float]],
) -> tuple[tuple[float, ...], tuple[float, ...], bool]:
    """Return q-reference, q-velocity reference, and whether the step is done."""

    initial = np.asarray(_six(initial_qpos_rad, name="initial_qpos_rad"), dtype=float)
    limits = tuple((float(lo), float(hi)) for lo, hi in position_limits_rad)
    if len(limits) != 6:
        raise ValueError("position_limits_rad must contain six pairs")
    if joint not in _ALL_JOINTS:
        raise ValueError(f"unknown B601-RS joint: {joint}")
    if mode in ("hold", "cartesian-hold"):
        return tuple(float(value) for value in initial), _ZERO_SIX, True
    if mode != "slow-step":
        raise ValueError("mode must be hold, slow-step, or cartesian-hold")
    direction = 1 if int(step_direction) >= 0 else -1
    step = _bounded(step_rad, name="step_rad", lower=0.0, upper=_MAX_STEP_RAD)
    speed = _bounded(step_speed_rad_s, name="step_speed_rad_s", lower=0.001, upper=0.05)
    index = _ALL_JOINTS.index(joint)
    target = float(initial[index] + direction * step)
    lower, upper = limits[index]
    if not lower <= target <= upper:
        raise ValueError(
            f"slow-step target for {joint} is outside URDF limits: "
            f"{target:.6f} not in [{lower:.6f}, {upper:.6f}]"
        )
    progress = min(step, max(0.0, _finite(elapsed_sec, name="elapsed_sec") * speed))
    done = progress >= step
    qref = initial.copy()
    qref[index] += direction * progress
    qvel = np.zeros(6, dtype=float)
    if not done:
        qvel[index] = direction * speed
    return tuple(float(value) for value in qref), tuple(float(value) for value in qvel), done


def validate_target_excursion(
    qpos_rad: Sequence[float],
    *,
    anchor_qpos_rad: Sequence[float],
    target_joint: str,
    max_excursion_rad: float = _MAX_STEP_RAD,
) -> float:
    """Return target excursion or reject motion beyond the hard relative limit."""

    current = _six(qpos_rad, name="qpos_rad")
    anchor = _six(anchor_qpos_rad, name="anchor_qpos_rad")
    if target_joint not in _ALL_JOINTS:
        raise ValueError(f"unknown B601-RS joint: {target_joint}")
    limit = _bounded(
        max_excursion_rad,
        name="max_excursion_rad",
        lower=0.0,
        upper=_MAX_STEP_RAD,
        lower_inclusive=False,
    )
    index = _ALL_JOINTS.index(target_joint)
    excursion = abs(current[index] - anchor[index])
    if excursion > limit:
        raise LowGainProbeError(
            f"target excursion abort: {target_joint} moved {excursion:.6f} rad "
            f"from the post-enable anchor, exceeding the {limit:.6f} rad "
            "(20 deg maximum) hard limit"
        )
    return excursion


def evaluate_slow_step_response(
    *,
    mode: str,
    target_joint: str,
    step_rad: float,
    step_direction: int,
    initial_qpos_rad: Sequence[float] | None,
    samples: Sequence[dict[str, object]],
    active_joint_names: Sequence[str],
    min_response_ratio: float = _DEFAULT_MIN_STEP_RESPONSE_RATIO,
    max_response_ratio: float = _DEFAULT_MAX_STEP_RESPONSE_RATIO,
    min_response_rad: float = _DEFAULT_MIN_STEP_RESPONSE_RAD,
    max_support_drift_rad: float = _DEFAULT_MAX_SUPPORT_DRIFT_RAD,
) -> dict[str, object]:
    """Classify whether a completed relative step produced a bounded response.

    This is commissioning evidence only.  It deliberately uses the mean of the
    final five feedback samples so one quantized sample cannot turn a no-motion
    run into an apparent response pass.
    """

    if mode != "slow-step":
        return {"status": "NOT_APPLICABLE", "passed": True}
    if target_joint not in _ALL_JOINTS:
        raise ValueError(f"unknown B601-RS joint: {target_joint}")
    if initial_qpos_rad is None:
        return {
            "status": "NO_INITIAL_REFERENCE",
            "passed": False,
            "sample_count": len(samples),
        }
    initial = np.asarray(_six(initial_qpos_rad, name="initial_qpos_rad"), dtype=float)
    step = _bounded(step_rad, name="step_rad", lower=0.0, upper=_MAX_STEP_RAD)
    if step <= 0.0:
        raise ValueError("slow-step response requires step_rad > 0")
    minimum_ratio = _bounded(
        min_response_ratio,
        name="min_response_ratio",
        lower=0.0,
        upper=1.0,
    )
    maximum_ratio = _bounded(
        max_response_ratio,
        name="max_response_ratio",
        lower=1.0,
        upper=5.0,
    )
    minimum_rad = _bounded(
        min_response_rad,
        name="min_response_rad",
        lower=0.0,
        upper=0.1,
    )
    support_limit = _bounded(
        max_support_drift_rad,
        name="max_support_drift_rad",
        lower=0.0,
        upper=0.1,
    )
    qpos_samples = tuple(
        np.asarray(_six(sample["qpos_rad"], name="sample_qpos_rad"), dtype=float)
        for sample in samples
    )
    if not qpos_samples:
        return {
            "status": "NO_FEEDBACK_SAMPLES",
            "passed": False,
            "sample_count": 0,
        }
    tail_size = min(5, len(qpos_samples))
    settled = np.mean(np.stack(qpos_samples[-tail_size:]), axis=0)
    target_index = _ALL_JOINTS.index(target_joint)
    direction = 1 if int(step_direction) >= 0 else -1
    signed_displacement = direction * float(
        settled[target_index] - initial[target_index]
    )
    response_ratio = signed_displacement / step
    peak_signed_displacement = max(
        direction * float(qpos[target_index] - initial[target_index])
        for qpos in qpos_samples
    )
    max_abs_target_excursion = max(
        abs(float(qpos[target_index] - initial[target_index]))
        for qpos in qpos_samples
    )
    support_names = tuple(
        name
        for name in active_joint_names
        if name in _ALL_JOINTS and name != target_joint
    )
    support_peak_drift = {
        name: max(
            abs(float(qpos[_ALL_JOINTS.index(name)] - initial[_ALL_JOINTS.index(name)]))
            for qpos in qpos_samples
        )
        for name in support_names
    }
    max_support_drift = max(support_peak_drift.values(), default=0.0)
    reference_completed = any(bool(sample.get("step_done", False)) for sample in samples)
    minimum_required = max(minimum_rad, minimum_ratio * step)
    maximum_allowed = maximum_ratio * step
    if max_abs_target_excursion > _MAX_STEP_RAD:
        status = "EXCURSION_LIMIT_EXCEEDED"
    elif not reference_completed:
        status = "REFERENCE_INCOMPLETE"
    elif signed_displacement < -minimum_rad:
        status = "WRONG_DIRECTION"
    elif signed_displacement < minimum_required:
        status = "INSUFFICIENT_RESPONSE"
    elif signed_displacement > maximum_allowed:
        status = "EXCESSIVE_RESPONSE"
    elif max_support_drift > support_limit:
        status = "SUPPORT_DRIFT_EXCEEDED"
    else:
        status = "PASS"
    return {
        "status": status,
        "passed": status == "PASS",
        "sample_count": len(qpos_samples),
        "settled_sample_count": tail_size,
        "reference_completed": reference_completed,
        "target_joint": target_joint,
        "step_direction": direction,
        "commanded_step_rad": step,
        "hard_excursion_limit_rad": _MAX_STEP_RAD,
        "settled_signed_displacement_rad": signed_displacement,
        "peak_signed_displacement_rad": peak_signed_displacement,
        "max_abs_target_excursion_rad": max_abs_target_excursion,
        "response_ratio": response_ratio,
        "minimum_response_rad": minimum_required,
        "maximum_response_rad": maximum_allowed,
        "final_tracking_error_rad": abs(step - signed_displacement),
        "support_joint_names": support_names,
        "support_peak_drift_rad": support_peak_drift,
        "max_support_drift_rad": max_support_drift,
        "max_support_drift_limit_rad": support_limit,
    }


def _output_payload(payload: dict[str, object], output: Path | None) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=list)
    print(encoded)
    if output is None:
        return
    output_path = output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(encoded + "\n", encoding="utf-8")


def _signal_handler(stop: dict[str, bool], signum: int, _frame: object) -> None:
    """Latch a protective stop; SIGUSR1 is the explicit all-disable request."""

    stop["requested"] = True
    if getattr(signal, "SIGUSR1", None) == signum:
        stop["full_disable_requested"] = True


def _freeze_and_stop(
    backend: B601RSBackend,
    model: B601PinocchioModel,
    *,
    qpos_rad: tuple[float, ...] | None,
    profile: B601MitControlProfile,
    gravity_scale: float,
    previous_tau: tuple[float, ...],
    torque_limits_nm: tuple[float, ...],
    freeze_sec: float,
    rate_hz: float,
    joint_names: Sequence[str] = _ALL_JOINTS,
) -> tuple[int, tuple[float, ...], list[str]]:
    """Hold the last known pose briefly before the explicit safe stop.

    joint_names is explicit so a single-joint commissioning run never
    re-enables or dispatches a frame to an unselected motor during cleanup.
    """

    if qpos_rad is None or freeze_sec <= 0.0:
        return 0, previous_tau, []
    errors: list[str] = []
    max_rate = profile.max_torque_rate_nm_s
    count = 0
    latest_tau = previous_tau
    period = 1.0 / rate_hz
    deadline = time.monotonic() + freeze_sec
    while time.monotonic() < deadline:
        try:
            reference = build_reference(
                sequence=0,
                timestamp_sec=time.monotonic(),
                qpos_rad=qpos_rad,
                qvel_rad_s=_ZERO_SIX,
            )
            requested = compute_software_torque(
                model,
                reference=reference,
                measured_qpos_rad=qpos_rad,
                measured_qvel_rad_s=_ZERO_SIX,
                profile=profile,
                gravity_scale=gravity_scale,
            )
            latest_tau, _torque_limited, _slew_limited = condition_torque(
                requested,
                latest_tau,
                absolute_limits_nm=torque_limits_nm,
                max_rate_nm_s=max_rate,
                dt_sec=period,
            )
            receipt = backend.send_mit(
                reference,
                kp=_ZERO_SIX,
                kd=_ZERO_SIX,
                tau_nm=latest_tau,
                torque_limits_nm=torque_limits_nm,
                joint_names=joint_names,
            )
            if not receipt.accepted:
                errors.extend(receipt.errors or ("freeze MIT dispatch was incomplete",))
                break
            count += 1
            time.sleep(period)
        except Exception as exc:  # noqa: BLE001 - report cleanup failure
            errors.append(f"freeze: {exc}")
            break
    return count, latest_tau, errors


def _send_fault_hold_command(
    backend: B601RSBackend,
    model: B601PinocchioModel,
    *,
    qpos_rad: tuple[float, ...],
    measured_qpos_rad: tuple[float, ...],
    measured_qvel_rad_s: tuple[float, ...],
    profile: B601MitControlProfile,
    gravity_scale: float,
    previous_tau: tuple[float, ...],
    torque_limits_nm: tuple[float, ...],
    rate_hz: float,
) -> tuple[B601CommandReceipt, tuple[float, ...]]:
    """Send one bounded fixed-reference MIT frame to joint2 and joint3 only."""

    period = 1.0 / _positive(rate_hz, name="rate_hz")
    reference = build_reference(
        sequence=0,
        timestamp_sec=time.monotonic(),
        qpos_rad=qpos_rad,
        qvel_rad_s=_ZERO_SIX,
    )
    requested = compute_software_torque(
        model,
        reference=reference,
        measured_qpos_rad=measured_qpos_rad,
        measured_qvel_rad_s=measured_qvel_rad_s,
        profile=profile,
        gravity_scale=gravity_scale,
    )
    conditioned, _torque_limited, _slew_limited = condition_torque(
        requested,
        previous_tau,
        absolute_limits_nm=torque_limits_nm,
        max_rate_nm_s=profile.max_torque_rate_nm_s,
        dt_sec=period,
    )
    receipt = backend.send_mit(
        reference,
        kp=_ZERO_SIX,
        kd=_ZERO_SIX,
        tau_nm=conditioned,
        torque_limits_nm=torque_limits_nm,
        joint_names=_FAULT_HOLD_JOINTS,
    )
    return receipt, conditioned


def _run_latched_fault_hold(
    backend: B601RSBackend,
    model: B601PinocchioModel,
    *,
    qpos_rad: tuple[float, ...] | None,
    qvel_rad_s: tuple[float, ...] | None,
    observed_age_sec: float,
    fault_qpos_rad: tuple[float, ...] | None,
    fault_qvel_rad_s: tuple[float, ...] | None,
    position_limits_rad: tuple[tuple[float, float], ...],
    max_age_sec: float,
    max_velocity_rad_s: float,
    max_rebase_rad: float,
    profile: B601MitControlProfile,
    gravity_scale: float,
    previous_tau: tuple[float, ...],
    torque_limits_nm: tuple[float, ...],
    rate_hz: float,
    stop: dict[str, bool],
    max_iterations: int | None = None,
) -> tuple[
    int,
    tuple[float, ...],
    B601CommandReceipt | None,
    B601ModeReceipt | None,
    B601ModeReceipt | None,
    list[str],
]:
    """Latch joint2/joint3 at a fresh, stationary, validated pose until SIGUSR1.

    The first hold frame is sent before the non-hold joints are disabled.  The
    loop has no time-based release: only the explicit SIGUSR1 handler may
    request safe_stop for all registered arm joints.  Each subsequent hold
    cycle validates a complete feedback snapshot and the fixed-reference
    drift before computing measured-state damping.  If a hold frame, feedback
    sample, or initial reference is invalid, the fail-closed fallback is an
    immediate all-arm safe_stop because a partial hold cannot be claimed as
    safe.

    max_iterations exists only for bounded offline tests; production calls
    leave it as None and therefore remain latched until release.
    """

    errors: list[str] = []
    latest_tau = previous_tau
    hold_commands = 0
    first_hold_receipt: B601CommandReceipt | None = None
    non_hold_disable_receipt: B601ModeReceipt | None = None
    full_disable_receipt: B601ModeReceipt | None = None

    def full_disable_fallback() -> None:
        nonlocal full_disable_receipt
        try:
            full_disable_receipt = backend.safe_stop(joint_names=_ALL_JOINTS)
            if not full_disable_receipt.completed:
                errors.extend(
                    f"full safe_stop: {error}"
                    for error in full_disable_receipt.errors
                )
                if not full_disable_receipt.errors:
                    errors.append("full safe_stop incomplete without vendor error")
        except Exception as exc:  # noqa: BLE001 - preserve safety evidence
            errors.append(f"full safe_stop: {exc}")

    try:
        hold_q = _validate_fault_hold_snapshot(
            qpos_rad,
            qvel_rad_s=qvel_rad_s,
            observed_age_sec=observed_age_sec,
            position_limits_rad=position_limits_rad,
            max_age_sec=max_age_sec,
            max_velocity_rad_s=max_velocity_rad_s,
            fault_qpos_rad=fault_qpos_rad,
            fault_qvel_rad_s=fault_qvel_rad_s,
            max_rebase_rad=max_rebase_rad,
        )
        measured_qpos = (
            fault_qpos_rad if fault_qpos_rad is not None else hold_q
        )
        measured_qvel = (
            fault_qvel_rad_s if fault_qvel_rad_s is not None else qvel_rad_s
        )
        first_hold_receipt, latest_tau = _send_fault_hold_command(
            backend,
            model,
            qpos_rad=hold_q,
            measured_qpos_rad=_six(measured_qpos, name="fault_measured_qpos_rad"),
            measured_qvel_rad_s=_six(measured_qvel, name="fault_measured_qvel_rad_s"),
            profile=profile,
            gravity_scale=gravity_scale,
            previous_tau=latest_tau,
            torque_limits_nm=torque_limits_nm,
            rate_hz=rate_hz,
        )
        if not first_hold_receipt.accepted:
            errors.extend(
                first_hold_receipt.errors
                or ("fault hold MIT dispatch was incomplete",)
            )
            full_disable_fallback()
            return (
                hold_commands,
                latest_tau,
                first_hold_receipt,
                non_hold_disable_receipt,
                full_disable_receipt,
                errors,
            )
        hold_commands = 1
    except Exception as exc:  # noqa: BLE001 - preserve safety evidence
        errors.append(f"fault hold: {exc}")
        full_disable_fallback()
        return (
            hold_commands,
            latest_tau,
            first_hold_receipt,
            non_hold_disable_receipt,
            full_disable_receipt,
            errors,
        )

    try:
        non_hold_disable_receipt = backend.safe_stop(
            joint_names=_NON_FAULT_HOLD_JOINTS,
        )
        if not non_hold_disable_receipt.completed:
            errors.extend(
                f"non-hold safe_stop: {error}"
                for error in non_hold_disable_receipt.errors
            )
            if not non_hold_disable_receipt.errors:
                errors.append("non-hold safe_stop incomplete without vendor error")
            full_disable_fallback()
            return (
                hold_commands,
                latest_tau,
                first_hold_receipt,
                non_hold_disable_receipt,
                full_disable_receipt,
                errors,
            )
    except Exception as exc:  # noqa: BLE001 - preserve safety evidence
        errors.append(f"non-hold safe_stop: {exc}")
        full_disable_fallback()
        return (
            hold_commands,
            latest_tau,
            first_hold_receipt,
            non_hold_disable_receipt,
            full_disable_receipt,
            errors,
        )

    print(
        "B601 protective hold latched: joint2/joint3 remain at the last valid "
        "position; joint1/joint4/joint5/joint6 are disabled. "
        f"pid={os.getpid()}; send SIGUSR1 to this process for explicit full disable.",
        flush=True,
    )
    period = 1.0 / _positive(rate_hz, name="rate_hz")
    monitor_q = _six(measured_qpos, name="fault_measured_qpos_rad")
    monitor_observed_at = time.monotonic() - max(0.0, observed_age_sec)
    while not stop.get("full_disable_requested", False):
        if max_iterations is not None and hold_commands >= max_iterations:
            break
        try:
            state = backend.poll_feedback(request_feedback=True)
            observed_at = time.monotonic()
            if (
                not state.valid
                or getattr(state, "stale", False)
                or state.positions_rad is None
            ):
                raise LowGainProbeError(
                    "protective hold feedback is not complete and valid: "
                    f"{state.fault_reason or 'unknown fault'}"
                )
            monitor_dt = observed_at - monitor_observed_at
            if not math.isfinite(monitor_dt) or monitor_dt <= 0.0:
                raise LowGainProbeError(
                    "protective hold feedback timestamp did not advance"
                )
            monitor_qvel = tuple(
                (current - previous) / monitor_dt
                for current, previous in zip(
                    state.positions_rad,
                    monitor_q,
                    strict=True,
                )
            )
            measured_q = _validate_protective_hold_feedback(
                state.positions_rad,
                qvel_rad_s=monitor_qvel,
                hold_qpos_rad=hold_q,
                observed_age_sec=0.0,
                position_limits_rad=position_limits_rad,
                max_age_sec=max_age_sec,
                max_velocity_rad_s=max_velocity_rad_s,
                max_rebase_rad=max_rebase_rad,
            )
            if stop.get("full_disable_requested", False):
                break
            receipt, latest_tau = _send_fault_hold_command(
                backend,
                model,
                qpos_rad=hold_q,
                measured_qpos_rad=measured_q,
                measured_qvel_rad_s=monitor_qvel,
                profile=profile,
                gravity_scale=gravity_scale,
                previous_tau=latest_tau,
                torque_limits_nm=torque_limits_nm,
                rate_hz=rate_hz,
            )
            if not receipt.accepted:
                errors.extend(
                    receipt.errors or ("fault hold MIT dispatch was incomplete",)
                )
                full_disable_fallback()
                break
            hold_commands += 1
            monitor_q = measured_q
            monitor_observed_at = observed_at
        except Exception as exc:  # noqa: BLE001 - preserve safety evidence
            errors.append(f"fault hold: {exc}")
            full_disable_fallback()
            break
        if max_iterations is not None and hold_commands >= max_iterations:
            break
        time.sleep(period)

    if stop.get("full_disable_requested", False):
        full_disable_fallback()
    return (
        hold_commands,
        latest_tau,
        first_hold_receipt,
        non_hold_disable_receipt,
        full_disable_receipt,
        errors,
    )


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    """Run the guarded active probe; callers must provide explicit authority."""

    _validate_cli(args)
    config_path = Path(args.config).expanduser().resolve(strict=True)
    source_root = Path(args.rebot_source_root).expanduser().resolve(strict=True)
    model, position_limits = _resolve_model_and_limits(source_root)
    if args.mode == "feedback-stability":
        config = B601RSBackendConfig.from_rs_yaml(
            config_path,
            allow_motion=False,
            position_limits_rad=position_limits,
        )
        return _run_feedback_stability_probe(
            B601RSBackend(config),
            config_path=config_path,
            source_root=source_root,
            model=model,
            position_limits=position_limits,
            duration_sec=args.duration_sec,
            rate_hz=args.rate_hz,
            stationarity_velocity_limit_rad_s=args.fault_hold_max_velocity_rad_s,
        )
    torque_limits = _resolve_effort_limits(
        model.urdf_path,
        scale=args.torque_limit_scale,
    )
    overlay_limits = (
        tuple(
            value * args.cartesian_overlay_limit_scale
            for value in torque_limits
        )
        if args.mode == "cartesian-hold"
        else _ZERO_SIX
    )
    profile = build_software_profile(
        kp=args.kp,
        kd=args.kd,
        torque_limits_nm=torque_limits,
        torque_rise_time_sec=args.torque_rise_time_sec,
        velocity_limit_rad_s=args.velocity_abort_rad_s,
        overlay_limits_nm=overlay_limits,
        target_joint=args.joint,
        target_kp=args.target_kp,
        target_kd=args.target_kd,
    )
    config = B601RSBackendConfig.from_rs_yaml(
        config_path,
        allow_motion=True,
        position_limits_rad=position_limits,
    )
    backend = B601RSBackend(config)
    ledger = FrequencyLedger()
    support_joint_names = tuple(
        getattr(args, "support_joints", _FAULT_HOLD_JOINTS)
    )
    active_joint_names = resolve_active_joint_names(
        args.joint,
        support_joints=support_joint_names,
    )
    stop = {"requested": False, "full_disable_requested": False}
    previous_q: tuple[float, ...] | None = None
    previous_feedback_time: float | None = None
    previous_tau = _ZERO_SIX
    initial_q: tuple[float, ...] | None = None
    final_q: tuple[float, ...] | None = None
    cartesian_anchor_position_m: tuple[float, ...] | None = None
    cartesian_spring: B601TranslationalVirtualSpring | None = None
    fault_qpos_for_hold: tuple[float, ...] | None = None
    fault_qvel_for_hold: tuple[float, ...] | None = None
    last_safe_q: tuple[float, ...] | None = None
    last_safe_qvel: tuple[float, ...] | None = None
    last_safe_observed_at_sec: float | None = None
    commands_sent = 0
    freeze_commands = 0
    schedule_overruns = 0
    torque_limit_count = 0
    slew_limit_count = 0
    overlay_limit_count = 0
    fault_hold_candidate_accepted_count = 0
    fault_hold_candidate_rejected_count = 0
    fault_hold_candidate_last_rejection: str | None = None
    samples: list[dict[str, object]] = []
    stop_reason: str | None = None
    primary_error: Exception | None = None
    preflight_disable: B601ModeReceipt | None = None
    mode_receipt: B601ModeReceipt | None = None
    enable_receipt: B601ModeReceipt | None = None
    cleanup_receipt: B601ModeReceipt | None = None
    cleanup_errors: list[str] = []
    protective_hold_latched = False
    protective_hold_commands = 0
    protective_hold_admission_age_sec: float | None = None
    protective_hold_first_receipt: B601CommandReceipt | None = None
    protective_hold_non_hold_receipt: B601ModeReceipt | None = None
    protective_hold_full_disable_receipt: B601ModeReceipt | None = None
    arming_settle_required_samples = _DEFAULT_ARMING_SETTLE_SAMPLES
    arming_settle_samples = 0
    arming_settle_passed = False
    arming_settle_elapsed_sec: float | None = None
    arming_settle_velocity_limit_rad_s = args.fault_hold_max_velocity_rad_s
    arming_settle_last_qvel: tuple[float, ...] | None = None
    connected = False
    enabled = False
    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigusr1 = signal.getsignal(signal.SIGUSR1)
    signal.signal(
        signal.SIGINT,
        lambda signum, frame: _signal_handler(stop, signum, frame),
    )
    signal.signal(
        signal.SIGUSR1,
        lambda signum, frame: _signal_handler(stop, signum, frame),
    )
    try:
        backend.connect()
        connected = True
        warmup_deadline = time.monotonic() + args.warmup_sec
        warmup_state = None
        warmup_observed_at_sec: float | None = None
        while time.monotonic() < warmup_deadline:
            if stop["full_disable_requested"]:
                raise LowGainProbeError(
                    "explicit full disable requested before motor enable"
                )
            if stop["requested"]:
                raise LowGainProbeError("operator interrupt before motor enable")
            warmup_state = backend.poll_feedback(request_feedback=True)
            warmup_observed_at_sec = time.monotonic()
            if (
                warmup_state.valid
                and not getattr(warmup_state, "stale", False)
                and warmup_state.positions_rad is not None
            ):
                break
            time.sleep(0.01)
        if (
            warmup_state is None
            or not warmup_state.valid
            or getattr(warmup_state, "stale", False)
            or warmup_state.positions_rad is None
        ):
            reason = (
                "no complete valid feedback snapshot"
                if warmup_state is None
                else warmup_state.fault_reason or "snapshot did not contain six positions"
            )
            raise LowGainProbeError(
                "no complete valid feedback snapshot arrived during warmup; "
                f"last_fault={reason}"
            )
        warmup_q = _validate_fault_hold_snapshot(
            warmup_state.positions_rad,
            qvel_rad_s=_ZERO_SIX,
            observed_age_sec=0.0,
            position_limits_rad=position_limits,
            max_age_sec=args.fault_hold_max_age_sec,
            max_velocity_rad_s=args.fault_hold_max_velocity_rad_s,
            max_rebase_rad=args.fault_hold_max_rebase_rad,
        )
        initial_q = warmup_q
        final_q = warmup_q
        if stop["full_disable_requested"]:
            raise LowGainProbeError(
                "explicit full disable requested before motor enable"
            )
        if stop["requested"]:
            raise LowGainProbeError("operator interrupt before motor enable")
        # A preflight stop makes the subsequent mode transition explicit.  It
        # is intentionally done only after the warmup snapshot and requires
        # the operator's physical support/estop procedure from the handoff.
        preflight_disable = backend.safe_stop(joint_names=_ALL_JOINTS)
        if not preflight_disable.completed:
            raise LowGainProbeError(
                "preflight safe_stop failed: " + "; ".join(preflight_disable.errors)
            )
        mode_receipt = backend.prepare_mit_mode(joint_names=active_joint_names)
        if not mode_receipt.completed:
            raise LowGainProbeError(
                "MIT mode setup failed: " + "; ".join(mode_receipt.errors)
            )
        if stop["full_disable_requested"]:
            raise LowGainProbeError(
                "explicit full disable requested before motor enable"
            )
        if stop["requested"]:
            raise LowGainProbeError("operator interrupt before motor enable")
        enable_receipt = backend.enable(joint_names=active_joint_names)
        if not enable_receipt.completed:
            raise LowGainProbeError(
                "motor enable failed: " + "; ".join(enable_receipt.errors)
            )
        enabled = True
        arming_state = backend.poll_feedback(request_feedback=True)
        arming_observed_at_sec = time.monotonic()
        if (
            not arming_state.valid
            or getattr(arming_state, "stale", False)
            or arming_state.positions_rad is None
        ):
            raise LowGainProbeError(
                "post-enable feedback snapshot was not complete and valid; "
                f"last_fault={arming_state.fault_reason or 'unknown'}"
            )
        if warmup_observed_at_sec is None:
            raise LowGainProbeError("missing warmup observation timestamp")
        arm_dt = arming_observed_at_sec - warmup_observed_at_sec
        if not math.isfinite(arm_dt) or arm_dt <= 0.0:
            raise LowGainProbeError("non-positive warmup-to-enable observation interval")
        arming_q = _six(arming_state.positions_rad, name="arming_qpos_rad")
        arming_qvel = tuple(
            (current - previous) / arm_dt
            for current, previous in zip(arming_q, warmup_q, strict=True)
        )
        arming_q = _validate_fault_hold_snapshot(
            arming_q,
            qvel_rad_s=arming_qvel,
            observed_age_sec=0.0,
            position_limits_rad=position_limits,
            max_age_sec=args.fault_hold_max_age_sec,
            max_velocity_rad_s=args.fault_hold_max_velocity_rad_s,
            max_rebase_rad=args.fault_hold_max_rebase_rad,
        )
        initial_q = arming_q
        final_q = arming_q
        last_safe_q = arming_q
        last_safe_qvel = tuple(float(value) for value in arming_qvel)
        last_safe_observed_at_sec = arming_observed_at_sec
        previous_q = arming_q
        previous_feedback_time = arming_state.receive_timestamp_sec
        arming_settle_started_at_sec = time.monotonic()
        arming_settle_previous_q = arming_q
        arming_settle_previous_feedback_time = arming_state.receive_timestamp_sec
        arming_settle_last_qvel = tuple(float(value) for value in arming_qvel)
        arming_settle_last_observed_at_sec = arming_observed_at_sec
        arming_settle_samples = 1
        arming_settle_period = 1.0 / args.rate_hz
        arming_settle_deadline = arming_settle_started_at_sec + arming_settle_period
        while arming_settle_samples < arming_settle_required_samples:
            if stop["full_disable_requested"]:
                stop_reason = "explicit_full_disable during arming settle"
                break
            if stop["requested"]:
                stop_reason = "operator_interrupt during arming settle"
                break
            now = time.monotonic()
            if now < arming_settle_deadline:
                time.sleep(arming_settle_deadline - now)
            if stop["full_disable_requested"]:
                stop_reason = "explicit_full_disable during arming settle"
                break
            if stop["requested"]:
                stop_reason = "operator_interrupt during arming settle"
                break
            settle_state = backend.poll_feedback(request_feedback=True)
            settle_observed_at_sec = time.monotonic()
            if (
                not settle_state.valid
                or getattr(settle_state, "stale", False)
                or settle_state.positions_rad is None
            ):
                stop_reason = (
                    "arming settle feedback snapshot was not complete and valid; "
                    f"last_fault={settle_state.fault_reason or 'unknown'}"
                )
                break
            try:
                (
                    arming_settle_previous_q,
                    arming_settle_last_qvel,
                ) = _validate_arming_settle_transition(
                    settle_state.positions_rad,
                    previous_qpos_rad=arming_settle_previous_q,
                    current_timestamp_sec=settle_state.receive_timestamp_sec,
                    previous_timestamp_sec=arming_settle_previous_feedback_time,
                    position_limits_rad=position_limits,
                    max_velocity_rad_s=arming_settle_velocity_limit_rad_s,
                )
            except (LowGainProbeError, TypeError, ValueError) as exc:
                stop_reason = f"arming settle rejected: {exc}"
                break
            arming_settle_previous_feedback_time = settle_state.receive_timestamp_sec
            arming_settle_last_observed_at_sec = settle_observed_at_sec
            arming_settle_samples += 1
            arming_settle_deadline += arming_settle_period
            completed_at = time.monotonic()
            arming_settle_deadline = max(arming_settle_deadline, completed_at)
        arming_settle_elapsed_sec = max(
            0.0,
            time.monotonic() - arming_settle_started_at_sec,
        )
        arming_settle_passed = (
            stop_reason is None
            and arming_settle_samples >= arming_settle_required_samples
        )
        if arming_settle_passed:
            arming_q = arming_settle_previous_q
            arming_qvel = arming_settle_last_qvel
            initial_q = arming_settle_previous_q
            final_q = arming_settle_previous_q
            last_safe_q = arming_settle_previous_q
            last_safe_qvel = arming_settle_last_qvel
            last_safe_observed_at_sec = arming_settle_last_observed_at_sec
            previous_q = arming_settle_previous_q
            previous_feedback_time = arming_settle_previous_feedback_time
        if stop_reason is None and args.mode == "cartesian-hold":
            arming_transform = np.asarray(
                model.forward_kinematics(arming_q),
                dtype=float,
            )
            if (
                arming_transform.shape != (4, 4)
                or not np.all(np.isfinite(arming_transform))
            ):
                raise LowGainProbeError(
                    "post-enable Cartesian anchor FK returned an invalid transform"
                )
            cartesian_anchor_position_m = tuple(
                float(value) for value in arming_transform[:3, 3]
            )
            cartesian_spring = B601TranslationalVirtualSpring(
                anchor_position_m=cartesian_anchor_position_m,
                stiffness_n_per_m=(args.cartesian_stiffness_n_per_m,) * 3,
                damping_n_s_per_m=(args.cartesian_damping_n_s_per_m,) * 3,
                max_abs_force_n=(args.cartesian_max_force_n,) * 3,
            )
        period = 1.0 / args.rate_hz
        started = time.monotonic()
        next_deadline = started
        while stop_reason is None and time.monotonic() - started < args.duration_sec:
            fault_qpos_for_hold = None
            fault_qvel_for_hold = None
            if stop["full_disable_requested"]:
                stop_reason = "explicit_full_disable"
                break
            if stop["requested"]:
                stop_reason = "operator_interrupt"
                break
            now = time.monotonic()
            if now < next_deadline:
                time.sleep(next_deadline - now)
            state = backend.poll_feedback(request_feedback=True)
            observed_at_sec = time.monotonic()
            ledger.mark("feedback", state.receive_timestamp_sec)
            if not state.valid or state.positions_rad is None:
                stop_reason = state.fault_reason or "invalid feedback snapshot"
                break
            if getattr(state, "stale", False):
                stop_reason = "stale feedback snapshot"
                break
            q = state.positions_rad
            final_q = q
            try:
                q = _six(q, name="feedback_qpos_rad")
            except (TypeError, ValueError) as exc:
                stop_reason = f"invalid feedback position: {exc}"
                break
            fault_qpos_for_hold = q
            for index, (value, (lower, upper)) in enumerate(
                zip(q, position_limits, strict=True)
            ):
                if not lower <= value <= upper:
                    stop_reason = (
                        f"feedback qpos_rad[{index}]={value:.6f} is outside "
                        f"URDF limits [{lower:.6f}, {upper:.6f}]"
                    )
                    break
            if stop_reason is not None:
                break
            if previous_q is None:
                qvel = _ZERO_SIX
                dt = period
            else:
                if previous_feedback_time is None:
                    raise LowGainProbeError("missing previous feedback timestamp")
                dt = state.receive_timestamp_sec - previous_feedback_time
                if not math.isfinite(dt) or dt <= 0.0:
                    stop_reason = "non-increasing feedback timestamp"
                    break
                qvel = tuple(
                    (current - previous) / dt
                    for current, previous in zip(q, previous_q, strict=True)
                )
            fault_qvel_for_hold = tuple(float(value) for value in qvel)
            if any(not math.isfinite(value) for value in qvel):
                stop_reason = "non-finite finite-difference velocity"
                break
            if args.mode == "slow-step" and initial_q is not None:
                try:
                    validate_target_excursion(
                        q,
                        anchor_qpos_rad=initial_q,
                        target_joint=args.joint,
                    )
                except LowGainProbeError as exc:
                    stop_reason = str(exc)
                    break
            try:
                (
                    safe_candidate_q,
                    fault_hold_candidate_rejection,
                ) = _evaluate_fault_hold_candidate(
                    q,
                    qvel_rad_s=qvel,
                    velocity_abort_rad_s=args.velocity_abort_rad_s,
                    observed_age_sec=max(
                        0.0,
                        time.monotonic() - observed_at_sec,
                    ),
                    position_limits_rad=position_limits,
                    max_age_sec=args.fault_hold_max_age_sec,
                    max_velocity_rad_s=args.fault_hold_max_velocity_rad_s,
                    max_rebase_rad=args.fault_hold_max_rebase_rad,
                )
            except LowGainProbeError as exc:
                stop_reason = str(exc)
                break
            if safe_candidate_q is None:
                fault_hold_candidate_rejected_count += 1
                fault_hold_candidate_last_rejection = fault_hold_candidate_rejection
            else:
                fault_hold_candidate_accepted_count += 1
            elapsed = time.monotonic() - started
            qref, qvel_ref, step_done = plan_reference(
                initial_q,
                mode=args.mode,
                joint=args.joint,
                elapsed_sec=elapsed,
                step_rad=args.step_rad,
                step_speed_rad_s=args.step_speed_rad_s,
                step_direction=args.step_direction,
                position_limits_rad=position_limits,
            )
            max_error = max(
                abs(current - reference)
                for index, (current, reference) in enumerate(
                    zip(q, qref, strict=True)
                )
                if _ALL_JOINTS[index] in active_joint_names
            )
            if max_error > args.max_error_rad:
                stop_reason = (
                    f"position error abort: max_abs_error={max_error:.6f} "
                    f"> {args.max_error_rad:.6f} rad"
                )
                break
            cartesian_audit: dict[str, object] | None = None
            joint_torque_overlay_nm = _ZERO_SIX
            if args.mode == "cartesian-hold":
                if cartesian_spring is None:
                    raise LowGainProbeError(
                        "Cartesian impedance spring was not armed"
                    )
                joint_torque_overlay_nm, cartesian_audit = (
                    compute_cartesian_impedance_overlay(
                        model,
                        cartesian_spring,
                        qpos_rad=q,
                        qvel_rad_s=qvel,
                        max_displacement_m=args.cartesian_max_displacement_m,
                    )
                )
            reference = build_reference(
                sequence=state.sequence,
                timestamp_sec=state.receive_timestamp_sec,
                qpos_rad=qref,
                qvel_rad_s=qvel_ref,
                joint_torque_overlay_nm=joint_torque_overlay_nm,
            )
            torque_breakdown = compute_software_torque_breakdown(
                model,
                reference=reference,
                measured_qpos_rad=q,
                measured_qvel_rad_s=qvel,
                profile=profile,
                gravity_scale=args.gravity_scale,
            )
            requested_tau = torque_breakdown.requested_torque_nm
            conditioned_tau, torque_limited, slew_limited = condition_torque(
                requested_tau,
                previous_tau,
                absolute_limits_nm=torque_limits,
                max_rate_nm_s=profile.max_torque_rate_nm_s,
                dt_sec=dt,
            )
            compute_time = time.monotonic()
            ledger.mark("compute", compute_time)
            receipt = backend.send_mit(
                reference,
                # Software owns the complete composed torque.  Passing
                # non-zero vendor gains would double-count PD and bypass the
                # final total-torque bound above.
                kp=_ZERO_SIX,
                kd=_ZERO_SIX,
                tau_nm=conditioned_tau,
                torque_limits_nm=torque_limits,
                joint_names=active_joint_names,
            )
            if not receipt.accepted:
                stop_reason = "; ".join(receipt.errors) or "incomplete MIT dispatch"
                break
            ledger.mark("tx", receipt.completed_at_sec)
            commands_sent += 1
            torque_limit_count += int(torque_limited)
            slew_limit_count += int(slew_limited)
            overlay_limit_count += int(torque_breakdown.overlay_limited)
            samples.append(
                {
                    "sequence": state.sequence,
                    "elapsed_sec": elapsed,
                    "feedback_timestamp_sec": state.receive_timestamp_sec,
                    "qpos_rad": q,
                    "qvel_fd_rad_s": qvel,
                    "qref_rad": qref,
                    "qvel_ref_rad_s": qvel_ref,
                    "gravity_tau_nm": torque_breakdown.gravity_torque_nm,
                    "position_tau_nm": torque_breakdown.position_torque_nm,
                    "damping_tau_nm": torque_breakdown.damping_torque_nm,
                    "overlay_requested_tau_nm": torque_breakdown.overlay_requested_nm,
                    "overlay_effective_tau_nm": torque_breakdown.overlay_effective_nm,
                    "overlay_limited": torque_breakdown.overlay_limited,
                    "requested_tau_nm": requested_tau,
                    "conditioned_tau_nm": conditioned_tau,
                    "torque_limited": torque_limited,
                    "slew_limited": slew_limited,
                    "fault_hold_candidate_accepted": safe_candidate_q is not None,
                    "fault_hold_candidate_rejection": fault_hold_candidate_rejection,
                    "step_done": step_done,
                    "cartesian_impedance": cartesian_audit,
                    "freshness_verified": state.freshness_verified,
                    "freshness_semantics": state.freshness_semantics,
                }
            )
            previous_q = q
            previous_feedback_time = state.receive_timestamp_sec
            previous_tau = conditioned_tau
            if safe_candidate_q is not None:
                last_safe_q = safe_candidate_q
                last_safe_qvel = tuple(float(value) for value in qvel)
                last_safe_observed_at_sec = observed_at_sec
            next_deadline += period
            completed_at = time.monotonic()
            if completed_at > next_deadline:
                schedule_overruns += 1
                next_deadline = completed_at
    except LowGainProbeError as exc:
        primary_error = exc
    except (B601HardwareError, OSError, TypeError, ValueError) as exc:
        primary_error = exc
    except Exception as exc:  # noqa: BLE001 - report unexpected probe failure
        primary_error = exc
    finally:
        full_stop_attempts = 0

        def issue_full_safe_stop() -> None:
            nonlocal cleanup_receipt, full_stop_attempts
            if cleanup_receipt is not None and cleanup_receipt.completed:
                return
            if full_stop_attempts >= 2:
                return
            full_stop_attempts += 1
            try:
                cleanup_receipt = backend.safe_stop(joint_names=_ALL_JOINTS)
                if not cleanup_receipt.completed:
                    cleanup_errors.extend(
                        f"safe_stop: {error}"
                        for error in cleanup_receipt.errors
                    )
                    if not cleanup_receipt.errors:
                        cleanup_errors.append(
                            "safe_stop incomplete without vendor error"
                        )
            except Exception as exc:  # noqa: BLE001 - preserve cleanup evidence
                cleanup_errors.append(f"safe_stop: {exc}")

        if connected:
            exceptional_stop = primary_error is not None or stop_reason is not None
            explicit_full_disable = stop.get("full_disable_requested", False)
            if enabled and exceptional_stop and not explicit_full_disable:
                # A communication/feedback fault without a complete current
                # sample cannot safely rebase a hold; use the upstream-style
                # immediate emergency disable fallback instead.
                can_attempt_hold = (
                    fault_qpos_for_hold is not None
                    or stop_reason == "operator_interrupt"
                )
                if can_attempt_hold:
                    if last_safe_observed_at_sec is None:
                        safe_age_sec = math.inf
                    else:
                        safe_age_sec = max(
                            0.0,
                            time.monotonic() - last_safe_observed_at_sec,
                        )
                    protective_hold_admission_age_sec = safe_age_sec
                    try:
                        (
                            protective_hold_commands,
                            previous_tau,
                            protective_hold_first_receipt,
                            protective_hold_non_hold_receipt,
                            protective_hold_full_disable_receipt,
                            hold_errors,
                        ) = _run_latched_fault_hold(
                            backend,
                            model,
                            qpos_rad=last_safe_q,
                            qvel_rad_s=last_safe_qvel,
                            observed_age_sec=safe_age_sec,
                            fault_qpos_rad=fault_qpos_for_hold,
                            fault_qvel_rad_s=fault_qvel_for_hold,
                            position_limits_rad=position_limits,
                            max_age_sec=args.fault_hold_max_age_sec,
                            max_velocity_rad_s=args.fault_hold_max_velocity_rad_s,
                            max_rebase_rad=args.fault_hold_max_rebase_rad,
                            profile=profile,
                            gravity_scale=args.gravity_scale,
                            previous_tau=previous_tau,
                            torque_limits_nm=torque_limits,
                            rate_hz=args.rate_hz,
                            stop=stop,
                        )
                        cleanup_errors.extend(hold_errors)
                        protective_hold_latched = (
                            protective_hold_first_receipt is not None
                            and protective_hold_non_hold_receipt is not None
                            and protective_hold_non_hold_receipt.completed
                            and protective_hold_commands > 0
                        )
                        if protective_hold_full_disable_receipt is not None:
                            cleanup_receipt = protective_hold_full_disable_receipt
                    except Exception as exc:  # noqa: BLE001 - preserve safety evidence
                        cleanup_errors.append(f"protective hold: {exc}")
                else:
                    cleanup_errors.append(
                        "protective hold refused: no complete current fault snapshot"
                    )
                issue_full_safe_stop()
            elif enabled and not exceptional_stop:
                freeze_commands, previous_tau, freeze_errors = _freeze_and_stop(
                    backend,
                    model,
                    qpos_rad=last_safe_q,
                    profile=profile,
                    gravity_scale=args.gravity_scale,
                    previous_tau=previous_tau,
                    torque_limits_nm=torque_limits,
                    freeze_sec=args.freeze_sec,
                    rate_hz=args.rate_hz,
                    joint_names=active_joint_names,
                )
                cleanup_errors.extend(freeze_errors)
                issue_full_safe_stop()
            else:
                issue_full_safe_stop()
        try:
            backend.close()
        except Exception as exc:  # noqa: BLE001 - preserve cleanup evidence
            cleanup_errors.append(f"close: {exc}")
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGUSR1, old_sigusr1)

    if primary_error is not None:
        detail = str(primary_error)
        if cleanup_errors:
            detail += "; cleanup: " + "; ".join(cleanup_errors)
        raise LowGainProbeError(detail) from primary_error

    frequency = ledger.summarize(target_hz=args.rate_hz)
    required_events = ("feedback", "compute", "tx")
    frequency_acceptance = {
        event: evaluate_frequency_target(
            frequency[event],
            minimum_rate_ratio=0.95,
            maximum_deadline_miss_ratio=0.02,
        )
        for event in required_events
    }
    cleanup_ok = cleanup_receipt is not None and cleanup_receipt.completed and not cleanup_errors
    control_passed = (
        stop_reason is None
        and commands_sent > 0
        and cleanup_ok
        and all(result.passed for result in frequency_acceptance.values())
    )
    slow_step_response = evaluate_slow_step_response(
        mode=args.mode,
        target_joint=args.joint,
        step_rad=args.step_rad,
        step_direction=args.step_direction,
        initial_qpos_rad=initial_q,
        samples=samples,
        active_joint_names=active_joint_names,
    )
    response_passed = bool(slow_step_response["passed"])
    acceptance_passed = control_passed and response_passed
    max_abs_requested = max(
        (max(abs(value) for value in sample["requested_tau_nm"]) for sample in samples),
        default=0.0,
    )
    max_abs_conditioned = max(
        (max(abs(value) for value in sample["conditioned_tau_nm"]) for sample in samples),
        default=0.0,
    )
    return {
        "status": "OK" if acceptance_passed else "FAIL",
        "mode": args.mode,
        "confirmation": CONFIRMATION,
        "channel": config.channel,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "rebot_source_root": str(source_root),
        "urdf_path": str(model.urdf_path),
        "urdf_sha256": hashlib.sha256(model.urdf_path.read_bytes()).hexdigest(),
        "transport": "motorbridge_wired_local",
        "joint_names": _ALL_JOINTS,
        "active_joint_names": active_joint_names,
        "support_joint_names": support_joint_names,
        "fault_hold_joint_names": _FAULT_HOLD_JOINTS,
        "fault_hold_non_hold_joint_names": _NON_FAULT_HOLD_JOINTS,
        "target_joint": args.joint,
        "arming_settle": {
            "required_samples": arming_settle_required_samples,
            "samples": arming_settle_samples,
            "passed": arming_settle_passed,
            "elapsed_sec": arming_settle_elapsed_sec,
            "velocity_limit_rad_s": arming_settle_velocity_limit_rad_s,
            "last_qvel_rad_s": arming_settle_last_qvel,
            "torque_commands_sent": 0,
        },
        "motors_enabled_by_probe": enabled,
        "motion_commands_sent": commands_sent > 0,
        "commands_sent": commands_sent,
        "freeze_commands_sent": freeze_commands,
        "duration_sec": args.duration_sec,
        "rate_hz": args.rate_hz,
        "stop_reason": stop_reason,
        "cleanup_errors": tuple(cleanup_errors),
        "preflight_disable_receipt": _mode_receipt_payload(preflight_disable),
        "mode_receipt": _mode_receipt_payload(mode_receipt),
        "enable_receipt": _mode_receipt_payload(enable_receipt),
        "cleanup_receipt": _mode_receipt_payload(cleanup_receipt),
        "protective_hold": {
            "latched": protective_hold_latched,
            "command_count": protective_hold_commands,
            "first_command_receipt": _command_receipt_payload(
                protective_hold_first_receipt
            ),
            "non_hold_disable_receipt": _mode_receipt_payload(
                protective_hold_non_hold_receipt
            ),
            "full_disable_receipt": _mode_receipt_payload(
                protective_hold_full_disable_receipt
            ),
            "last_safe_qpos_rad": last_safe_q,
            "last_safe_qvel_rad_s": last_safe_qvel,
            "last_safe_age_sec": (
                None
                if last_safe_observed_at_sec is None
                else max(0.0, time.monotonic() - last_safe_observed_at_sec)
            ),
            "admission_age_sec": protective_hold_admission_age_sec,
            "max_age_sec": args.fault_hold_max_age_sec,
            "max_velocity_rad_s": args.fault_hold_max_velocity_rad_s,
            "max_rebase_rad": args.fault_hold_max_rebase_rad,
            "vendor_kp_nm_per_rad": _ZERO_SIX,
            "vendor_kd_nm_s_per_rad": _ZERO_SIX,
            "reference_velocity_rad_s": _ZERO_SIX,
        },
        "fault_hold_candidate_admission": {
            "accepted_count": fault_hold_candidate_accepted_count,
            "rejected_count": fault_hold_candidate_rejected_count,
            "last_rejection": fault_hold_candidate_last_rejection,
            "max_age_sec": args.fault_hold_max_age_sec,
            "max_velocity_rad_s": args.fault_hold_max_velocity_rad_s,
            "semantics": "rejected_sample_retains_previous_last_safe",
        },
        "initial_qpos_rad": initial_q,
        "final_qpos_rad": final_q,
        "profile": {
            "software_kp_nm_per_rad": profile.position_kp_nm_per_rad,
            "software_kd_nm_s_per_rad": profile.damping_kd_nm_s_per_rad,
            "gravity_scale": args.gravity_scale,
            "torque_limits_nm": torque_limits,
            "torque_limit_scale": args.torque_limit_scale,
            "max_torque_rate_nm_s": profile.max_torque_rate_nm_s,
            "max_abs_overlay_torque_nm": profile.max_abs_overlay_torque_nm,
            "vendor_kp_nm_per_rad": _ZERO_SIX,
            "vendor_kd_nm_s_per_rad": _ZERO_SIX,
            "torque_owner": "software_composed_tau_only",
        },
        "slow_step_response": slow_step_response,
        "cartesian_impedance": {
            "enabled": args.mode == "cartesian-hold",
            "translation_only": True,
            "anchor_source": "fresh_post_enable_fk",
            "anchor_position_m": cartesian_anchor_position_m,
            "stiffness_n_per_m": (args.cartesian_stiffness_n_per_m,) * 3,
            "damping_n_s_per_m": (args.cartesian_damping_n_s_per_m,) * 3,
            "max_abs_force_n": (args.cartesian_max_force_n,) * 3,
            "max_displacement_m": args.cartesian_max_displacement_m,
            "overlay_limit_scale": args.cartesian_overlay_limit_scale,
            "requires_all_six_arm_joints": True,
        },
        "torque_limit_count": torque_limit_count,
        "slew_limit_count": slew_limit_count,
        "overlay_limit_count": overlay_limit_count,
        "max_abs_requested_tau_nm": max_abs_requested,
        "max_abs_conditioned_tau_nm": max_abs_conditioned,
        "frequency": {event: stats.as_dict() for event, stats in frequency.items()},
        "frequency_acceptance": {
            event: result.as_dict()
            for event, result in frequency_acceptance.items()
        },
        "control_acceptance_status": "PASS" if control_passed else "FAIL",
        "motion_response_status": slow_step_response["status"],
        "acceptance_status": "PASS" if acceptance_passed else "FAIL",
        "hardware_acceptance": "NOT_CLAIMED",
        "freshness_verified": False,
        "freshness_semantics": "motorbridge_mechPos_parameter_observation_unverified",
        "samples": samples,
        "notes": [
            "Human-run commissioning only; automated tests never execute this active path.",
            "The active set is the target plus explicitly selected support joints; scheme-1 J6 commissioning selects all six arm joints while only the J6 reference changes.",
            "On an exceptional stop, only joint2/joint3 may be latched; joint1/joint4/joint5/joint6 are safe-stopped, and SIGUSR1 is required for full disable.",
            "Protective hold admits only a host-fresh, in-limit, near-zero-velocity last-safe snapshot and a small fault rebase; otherwise it falls back to full safe_stop.",
            "A control sample that fails only fault-hold candidate admission is not promoted to last-safe; normal control remains governed by its separate velocity abort and all other watchdogs.",
            "While latched, every hold cycle validates a complete feedback snapshot and held-joint drift before sending a fixed-reference, measured-state torque frame; any failure falls back to full safe_stop.",
            "Gripper motor ID7 is outside this backend and is not included in any all-arm receipt.",
            "The probe never chooses an absolute pose. hold starts at the post-enable settled pose; slow-step uses a bounded relative step with a 20 degree command and observed-excursion hard limit.",
            "Before any torque command, the active path requires four consecutive complete feedback snapshots after enable; no torque command is sent during this arming-settle gate.",
            "cartesian-hold anchors fresh post-enable FK position, controls translation only, and requires all six arm joints.",
            "Software composes gravity, position, finite-difference damping, and an optional bounded J.T F overlay, then dispatches vendor MIT kp=kd=0.",
            "Gravity uses B601PinocchioModel with Pinocchio-neutral auxiliary gripper coordinates; actual gripper load is not identified.",
            "The explicit mechPos path is synchronous and normally around the observed low-bandwidth feedback rate; this is not a high-rate haptic loop.",
            "A slow-step PASS additionally requires a completed reference, correct-direction bounded target response, and bounded support-joint drift. It is not a HARDWARE_PASS or force-sensor validation.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "B601-RS low-gain gravity/damping commissioning; motion-capable and "
            "human-run only."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("hold", "slow-step", "cartesian-hold", "feedback-stability"),
        default="hold",
    )
    parser.add_argument("--config", required=True, help="explicit rebotarm_rs.yaml")
    parser.add_argument(
        "--rebot-source-root",
        type=Path,
        required=True,
        help="pinned reBotArm_control_py source root containing the URDF",
    )
    parser.add_argument("--duration-sec", type=float, default=_DEFAULT_DURATION_SEC)
    parser.add_argument("--warmup-sec", type=float, default=_DEFAULT_WARMUP_SEC)
    parser.add_argument("--freeze-sec", type=float, default=_DEFAULT_FREEZE_SEC)
    parser.add_argument("--rate-hz", type=float, default=_DEFAULT_RATE_HZ)
    parser.add_argument(
        "--joint",
        choices=_ALL_JOINTS,
        default="joint1",
        help="joint whose reference may change; joint2/joint3 remain support joints",
    )
    parser.add_argument(
        "--support-joint",
        dest="support_joints",
        action="append",
        default=list(_FAULT_HOLD_JOINTS),
        help="additional support joint; joint2 and joint3 are always mandatory",
    )
    parser.add_argument("--step-rad", type=float, default=_DEFAULT_STEP_RAD)
    parser.add_argument(
        "--step-speed-rad-s",
        type=float,
        default=_DEFAULT_STEP_SPEED_RAD_S,
    )
    parser.add_argument("--step-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--kp", type=float, default=_DEFAULT_KP)
    parser.add_argument("--kd", type=float, default=_DEFAULT_KD)
    parser.add_argument(
        "--target-kp",
        type=float,
        help="optional software position gain override for --joint only",
    )
    parser.add_argument(
        "--target-kd",
        type=float,
        help="optional software damping gain override for --joint only",
    )
    parser.add_argument("--gravity-scale", type=float, default=_DEFAULT_GRAVITY_SCALE)
    parser.add_argument(
        "--torque-limit-scale",
        type=float,
        default=_DEFAULT_TORQUE_LIMIT_SCALE,
        help="fraction of the pinned URDF effort ceiling; max 0.5 for this probe",
    )
    parser.add_argument(
        "--torque-rise-time-sec",
        type=float,
        default=_DEFAULT_TORQUE_RISE_TIME_SEC,
    )
    parser.add_argument(
        "--velocity-abort-rad-s",
        type=float,
        default=_DEFAULT_VELOCITY_ABORT_RAD_S,
    )
    parser.add_argument("--max-error-rad", type=float, default=_DEFAULT_MAX_ERROR_RAD)
    parser.add_argument(
        "--fault-hold-max-age-sec",
        type=float,
        default=_DEFAULT_FAULT_HOLD_MAX_AGE_SEC,
        help="maximum host-observation age allowed for a protective hold",
    )
    parser.add_argument(
        "--fault-hold-max-velocity-rad-s",
        type=float,
        default=_DEFAULT_FAULT_HOLD_MAX_VELOCITY_RAD_S,
        help="maximum absolute observed velocity for a protective hold",
    )
    parser.add_argument(
        "--fault-hold-max-rebase-rad",
        type=float,
        default=_DEFAULT_FAULT_HOLD_MAX_REBASE_RAD,
        help="maximum joint2/joint3 fault-to-last-safe rebase distance",
    )
    parser.add_argument(
        "--cartesian-stiffness-n-per-m",
        type=float,
        default=_DEFAULT_CARTESIAN_STIFFNESS_N_PER_M,
        help="isotropic translational stiffness for cartesian-hold",
    )
    parser.add_argument(
        "--cartesian-damping-n-s-per-m",
        type=float,
        default=_DEFAULT_CARTESIAN_DAMPING_N_S_PER_M,
        help="isotropic translational damping for cartesian-hold",
    )
    parser.add_argument(
        "--cartesian-max-force-n",
        type=float,
        default=_DEFAULT_CARTESIAN_MAX_FORCE_N,
        help="per-axis Cartesian force cap for cartesian-hold",
    )
    parser.add_argument(
        "--cartesian-max-displacement-m",
        type=float,
        default=_DEFAULT_CARTESIAN_MAX_DISPLACEMENT_M,
        help="TCP displacement-norm abort threshold for cartesian-hold",
    )
    parser.add_argument(
        "--cartesian-overlay-limit-scale",
        type=float,
        default=_DEFAULT_CARTESIAN_OVERLAY_LIMIT_SCALE,
        help="fraction of the already-scaled total torque bound available to J.T F",
    )
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--auto-disable", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--output", type=Path, help="write JSON outside the source worktree")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_probe(args)
    except (LowGainProbeError, B601HardwareError, OSError, ValueError) as exc:
        payload = {"status": "ERROR", "mode": args.mode, "error": str(exc)}
        _output_payload(payload, args.output)
        return 2
    _output_payload(payload, args.output)
    return 0 if payload.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
