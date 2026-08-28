"""Guarded, human-run B601-RS low-gain gravity/damping commissioning.

This probe is intentionally a slow, single-host wired commissioning tool.  It
reuses the AgenticRobot B601 backend, the pinned Pinocchio model, and the
upstream reBotArm MIT argument order.  It is not a haptic servo, a force
sensor, or a hardware-readiness claim.

The software owns the complete low-gain torque composition:

``tau = gravity_scale * g(q) + kp * (q_ref - q) + kd * (dq_ref - dq)``

The dispatched RobStride MIT ``kp`` and ``kd`` are always zero, so the final
software torque is the only PD/derivative contribution that this probe owns.
The software then applies a final absolute torque bound and slew-rate bound
before calling :meth:`B601RSBackend.send_mit`.  Feedback uses the existing
explicit ``mechPos`` request path; its source freshness remains unverified by
MotorBridge 0.4.9 and is reported as such.

This module is motion-capable only after all of ``--allow-motion``,
``--auto-disable`` and the exact confirmation string are supplied.  Automated
tests exercise the pure helpers only; they never construct a hardware backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# Direct ``python scripts/hardware/...`` invocation places the script directory
# on sys.path rather than the repository root.
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
)
from robot_motion.rebot_b601_rs import (
    B601MitControlProfile,
    B601MitReference,
    B601PinocchioModel,
    compute_b601_mit_torque,
)

CONFIRMATION = "B601_RS_LOW_GAIN_MIT_GRAVITY_DAMPING"
_ZERO_SIX = (0.0,) * 6
_ALL_JOINTS = B601_RS_ARM_JOINT_NAMES
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
_DEFAULT_VELOCITY_ABORT_RAD_S = 0.10
_DEFAULT_MAX_ERROR_RAD = 0.15


class LowGainProbeError(RuntimeError):
    """A guarded probe precondition, safety stop, or vendor operation failed."""


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


def _validate_cli(args: argparse.Namespace) -> None:
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
    if args.mode not in ("hold", "slow-step"):
        raise ValueError("mode must be hold or slow-step")
    _bounded(args.rate_hz, name="rate_hz", lower=1.0, upper=40.0)
    _bounded(args.duration_sec, name="duration_sec", lower=0.1, upper=60.0)
    _bounded(args.warmup_sec, name="warmup_sec", lower=0.1, upper=15.0)
    _bounded(args.freeze_sec, name="freeze_sec", lower=0.1, upper=10.0)
    _bounded(args.kp, name="kp", lower=0.0, upper=5.0)
    _bounded(args.kd, name="kd", lower=0.0, upper=5.0)
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
    _bounded(args.step_rad, name="step_rad", lower=0.0, upper=0.2)
    _bounded(
        args.step_speed_rad_s,
        name="step_speed_rad_s",
        lower=0.001,
        upper=0.05,
    )
    if args.mode == "slow-step" and args.step_rad <= 0.0:
        raise ValueError("slow-step mode requires step_rad > 0")
    if args.joint not in _ALL_JOINTS:
        raise ValueError(f"unknown B601-RS joint: {args.joint}")


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
) -> B601MitControlProfile:
    """Build the low-gain software composition profile used by the probe."""

    kp = _bounded(kp, name="kp", lower=0.0, upper=5.0)
    kd = _bounded(kd, name="kd", lower=0.0, upper=5.0)
    limits = _six(torque_limits_nm, name="torque_limits_nm")
    if any(value <= 0.0 for value in limits):
        raise ValueError("torque_limits_nm must be positive")
    rise_time = _positive(torque_rise_time_sec, name="torque_rise_time_sec")
    velocity_limit = _positive(velocity_limit_rad_s, name="velocity_limit_rad_s")
    return B601MitControlProfile(
        name="b601_rs_low_gain_software_gravity_damping",
        position_kp_nm_per_rad=(kp,) * 6,
        damping_kd_nm_s_per_rad=(kd,) * 6,
        max_abs_torque_nm=limits,
        max_torque_rate_nm_s=tuple(value / rise_time for value in limits),
        max_abs_overlay_torque_nm=_ZERO_SIX,
        max_abs_reference_velocity_rad_s=(max(velocity_limit, 0.05),) * 6,
        max_abs_state_velocity_rad_s=(max(velocity_limit, 0.05),) * 6,
        metadata={
            "hardware_safety_claim": False,
            "torque_owner": "software_composed_tau; vendor_kp_kd_zero",
            "torque_limit_source": "pinned_urdf_effort_scaled",
            "torque_rise_time_sec": rise_time,
        },
    )


def build_reference(
    *,
    sequence: int,
    timestamp_sec: float,
    qpos_rad: Sequence[float],
    qvel_rad_s: Sequence[float] = _ZERO_SIX,
) -> B601MitReference:
    """Build a validated local-clock MIT reference."""

    timestamp = _finite(timestamp_sec, name="timestamp_sec")
    return B601MitReference(
        sequence=int(sequence),
        source_timestamp_sec=timestamp,
        receive_timestamp_sec=timestamp,
        qpos_rad=_six(qpos_rad, name="qpos_rad"),
        qvel_rad_s=_six(qvel_rad_s, name="qvel_rad_s"),
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
    """Compose gravity, low-gain position, and damping torque in software."""

    scale = _bounded(gravity_scale, name="gravity_scale", lower=0.0, upper=1.0)
    gravity = tuple(
        scale * float(value)
        for value in model.arm_generalized_gravity(measured_qpos_rad)
    )
    breakdown = compute_b601_mit_torque(
        reference,
        profile,
        measured_qpos_rad=measured_qpos_rad,
        measured_qvel_rad_s=measured_qvel_rad_s,
        gravity_torque_nm=gravity,
    )
    return breakdown.requested_torque_nm


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
    if mode == "hold":
        return tuple(float(value) for value in initial), _ZERO_SIX, True
    if mode != "slow-step":
        raise ValueError("mode must be hold or slow-step")
    direction = 1 if int(step_direction) >= 0 else -1
    step = _bounded(step_rad, name="step_rad", lower=0.0, upper=0.2)
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


def _output_payload(payload: dict[str, object], output: Path | None) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=list)
    print(encoded)
    if output is None:
        return
    output_path = output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(encoded + "\n", encoding="utf-8")


def _signal_handler(stop: dict[str, bool], _signum: int, _frame: object) -> None:
    stop["requested"] = True


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
) -> tuple[int, tuple[float, ...], list[str]]:
    """Hold the last known pose briefly before the explicit safe stop."""

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
                joint_names=_ALL_JOINTS,
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


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    """Run the guarded active probe; callers must provide explicit authority."""

    _validate_cli(args)
    config_path = Path(args.config).expanduser().resolve(strict=True)
    source_root = Path(args.rebot_source_root).expanduser().resolve(strict=True)
    model, position_limits = _resolve_model_and_limits(source_root)
    torque_limits = _resolve_effort_limits(
        model.urdf_path,
        scale=args.torque_limit_scale,
    )
    profile = build_software_profile(
        kp=args.kp,
        kd=args.kd,
        torque_limits_nm=torque_limits,
        torque_rise_time_sec=args.torque_rise_time_sec,
        velocity_limit_rad_s=args.velocity_abort_rad_s,
    )
    config = B601RSBackendConfig.from_rs_yaml(
        config_path,
        allow_motion=True,
        position_limits_rad=position_limits,
    )
    backend = B601RSBackend(config)
    ledger = FrequencyLedger()
    stop = {"requested": False}
    previous_q: tuple[float, ...] | None = None
    previous_feedback_time: float | None = None
    previous_tau = _ZERO_SIX
    initial_q: tuple[float, ...] | None = None
    final_q: tuple[float, ...] | None = None
    commands_sent = 0
    freeze_commands = 0
    schedule_overruns = 0
    torque_limit_count = 0
    slew_limit_count = 0
    samples: list[dict[str, object]] = []
    stop_reason: str | None = None
    primary_error: Exception | None = None
    preflight_disable: B601ModeReceipt | None = None
    mode_receipt: B601ModeReceipt | None = None
    enable_receipt: B601ModeReceipt | None = None
    cleanup_receipt: B601ModeReceipt | None = None
    cleanup_errors: list[str] = []
    enable_attempted = False
    enabled = False
    old_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda signum, frame: _signal_handler(stop, signum, frame))
    try:
        backend.connect()
        warmup_deadline = time.monotonic() + args.warmup_sec
        warmup_state = None
        while time.monotonic() < warmup_deadline:
            warmup_state = backend.poll_feedback(request_feedback=True)
            if warmup_state.valid and warmup_state.positions_rad is not None:
                break
            time.sleep(0.01)
        if (
            warmup_state is None
            or not warmup_state.valid
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
        initial_q = warmup_state.positions_rad
        final_q = initial_q
        for index, (value, (lower, upper)) in enumerate(
            zip(initial_q, position_limits, strict=True)
        ):
            if not lower <= value <= upper:
                raise LowGainProbeError(
                    f"warmup qpos_rad[{index}]={value:.6f} is outside URDF limits"
                )
        # A preflight stop makes the subsequent mode transition explicit.  It
        # is intentionally done only after the warmup snapshot and requires
        # the operator's physical support/estop procedure from the handoff.
        preflight_disable = backend.safe_stop(joint_names=_ALL_JOINTS)
        if not preflight_disable.completed:
            raise LowGainProbeError(
                "preflight safe_stop failed: " + "; ".join(preflight_disable.errors)
            )
        mode_receipt = backend.prepare_mit_mode(joint_names=_ALL_JOINTS)
        if not mode_receipt.completed:
            raise LowGainProbeError(
                "MIT mode setup failed: " + "; ".join(mode_receipt.errors)
            )
        enable_attempted = True
        enable_receipt = backend.enable(joint_names=_ALL_JOINTS)
        if not enable_receipt.completed:
            raise LowGainProbeError(
                "motor enable failed: " + "; ".join(enable_receipt.errors)
            )
        enabled = True
        period = 1.0 / args.rate_hz
        started = time.monotonic()
        next_deadline = started
        while time.monotonic() - started < args.duration_sec:
            if stop["requested"]:
                stop_reason = "operator_interrupt"
                break
            now = time.monotonic()
            if now < next_deadline:
                time.sleep(next_deadline - now)
            state = backend.poll_feedback(request_feedback=True)
            ledger.mark("feedback", state.receive_timestamp_sec)
            if not state.valid or state.positions_rad is None:
                stop_reason = state.fault_reason or "invalid feedback snapshot"
                break
            q = state.positions_rad
            final_q = q
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
            if any(not math.isfinite(value) for value in qvel):
                stop_reason = "non-finite finite-difference velocity"
                break
            max_velocity = max(abs(value) for value in qvel)
            if max_velocity > args.velocity_abort_rad_s:
                stop_reason = (
                    f"velocity abort: max_abs_fd_velocity={max_velocity:.6f} "
                    f"> {args.velocity_abort_rad_s:.6f} rad/s"
                )
                break
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
                for current, reference in zip(q, qref, strict=True)
            )
            if max_error > args.max_error_rad:
                stop_reason = (
                    f"position error abort: max_abs_error={max_error:.6f} "
                    f"> {args.max_error_rad:.6f} rad"
                )
                break
            reference = build_reference(
                sequence=state.sequence,
                timestamp_sec=state.receive_timestamp_sec,
                qpos_rad=qref,
                qvel_rad_s=qvel_ref,
            )
            requested_tau = compute_software_torque(
                model,
                reference=reference,
                measured_qpos_rad=q,
                measured_qvel_rad_s=qvel,
                profile=profile,
                gravity_scale=args.gravity_scale,
            )
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
                joint_names=_ALL_JOINTS,
            )
            if not receipt.accepted:
                stop_reason = "; ".join(receipt.errors) or "incomplete MIT dispatch"
                break
            ledger.mark("tx", receipt.completed_at_sec)
            commands_sent += 1
            torque_limit_count += int(torque_limited)
            slew_limit_count += int(slew_limited)
            samples.append(
                {
                    "sequence": state.sequence,
                    "elapsed_sec": elapsed,
                    "feedback_timestamp_sec": state.receive_timestamp_sec,
                    "qpos_rad": q,
                    "qvel_fd_rad_s": qvel,
                    "qref_rad": qref,
                    "qvel_ref_rad_s": qvel_ref,
                    "requested_tau_nm": requested_tau,
                    "conditioned_tau_nm": conditioned_tau,
                    "torque_limited": torque_limited,
                    "slew_limited": slew_limited,
                    "step_done": step_done,
                    "freshness_verified": state.freshness_verified,
                    "freshness_semantics": state.freshness_semantics,
                }
            )
            previous_q = q
            previous_feedback_time = state.receive_timestamp_sec
            previous_tau = conditioned_tau
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
        if enable_attempted:
            if enabled:
                freeze_commands, previous_tau, freeze_errors = _freeze_and_stop(
                    backend,
                    model,
                    qpos_rad=final_q,
                    profile=profile,
                    gravity_scale=args.gravity_scale,
                    previous_tau=previous_tau,
                    torque_limits_nm=torque_limits,
                    freeze_sec=args.freeze_sec,
                    rate_hz=args.rate_hz,
                )
                cleanup_errors.extend(freeze_errors)
            try:
                cleanup_receipt = backend.safe_stop(joint_names=_ALL_JOINTS)
                if not cleanup_receipt.completed:
                    cleanup_errors.extend(
                        f"safe_stop: {error}"
                        for error in cleanup_receipt.errors
                    )
                    if not cleanup_receipt.errors:
                        cleanup_errors.append("safe_stop incomplete without vendor error")
            except Exception as exc:  # noqa: BLE001 - preserve cleanup evidence
                cleanup_errors.append(f"safe_stop: {exc}")
        try:
            backend.close()
        except Exception as exc:  # noqa: BLE001 - preserve cleanup evidence
            cleanup_errors.append(f"close: {exc}")
        signal.signal(signal.SIGINT, old_sigint)

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
    max_abs_requested = max(
        (max(abs(value) for value in sample["requested_tau_nm"]) for sample in samples),
        default=0.0,
    )
    max_abs_conditioned = max(
        (max(abs(value) for value in sample["conditioned_tau_nm"]) for sample in samples),
        default=0.0,
    )
    return {
        "status": "OK" if control_passed else "FAIL",
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
        "target_joint": args.joint,
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
        "initial_qpos_rad": initial_q,
        "final_qpos_rad": final_q,
        "profile": {
            "software_kp_nm_per_rad": profile.position_kp_nm_per_rad,
            "software_kd_nm_s_per_rad": profile.damping_kd_nm_s_per_rad,
            "gravity_scale": args.gravity_scale,
            "torque_limits_nm": torque_limits,
            "torque_limit_scale": args.torque_limit_scale,
            "max_torque_rate_nm_s": profile.max_torque_rate_nm_s,
            "vendor_kp_nm_per_rad": _ZERO_SIX,
            "vendor_kd_nm_s_per_rad": _ZERO_SIX,
            "torque_owner": "software_composed_tau_only",
        },
        "torque_limit_count": torque_limit_count,
        "slew_limit_count": slew_limit_count,
        "max_abs_requested_tau_nm": max_abs_requested,
        "max_abs_conditioned_tau_nm": max_abs_conditioned,
        "frequency": {event: stats.as_dict() for event, stats in frequency.items()},
        "frequency_acceptance": {
            event: result.as_dict()
            for event, result in frequency_acceptance.items()
        },
        "acceptance_status": "PASS" if control_passed else "FAIL",
        "hardware_acceptance": "NOT_CLAIMED",
        "freshness_verified": False,
        "freshness_semantics": "motorbridge_mechPos_parameter_observation_unverified",
        "samples": samples,
        "notes": [
            "Human-run commissioning only; automated tests never execute this active path.",
            "All six arm motors are used as bounded support joints; gripper motor ID7 is excluded.",
            "The probe never chooses an absolute pose. hold starts at the warmup pose; slow-step uses a bounded relative step.",
            "Software composes gravity, position, and finite-difference damping, then dispatches vendor MIT kp=kd=0.",
            "Gravity uses B601PinocchioModel with Pinocchio-neutral auxiliary gripper coordinates; actual gripper load is not identified.",
            "The explicit mechPos path is synchronous and normally around the observed low-bandwidth feedback rate; this is not a high-rate haptic loop.",
            "A PASS means the guarded software run and cleanup met their declared bounds. It is not a HARDWARE_PASS or force-sensor validation.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "B601-RS low-gain gravity/damping commissioning; motion-capable and "
            "human-run only."
        )
    )
    parser.add_argument("--mode", choices=("hold", "slow-step"), default="hold")
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
    parser.add_argument("--joint", choices=_ALL_JOINTS, default="joint3")
    parser.add_argument("--step-rad", type=float, default=_DEFAULT_STEP_RAD)
    parser.add_argument(
        "--step-speed-rad-s",
        type=float,
        default=_DEFAULT_STEP_SPEED_RAD_S,
    )
    parser.add_argument("--step-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--kp", type=float, default=_DEFAULT_KP)
    parser.add_argument("--kd", type=float, default=_DEFAULT_KD)
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
