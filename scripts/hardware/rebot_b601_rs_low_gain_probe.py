"""Guarded, human-run B601-RS low-gain gravity/damping commissioning.

This probe is intentionally a slow, single-host wired commissioning tool.  It
reuses the AgenticRobot B601 backend, the pinned Pinocchio model, and the
upstream reBotArm MIT argument order.  It is not a haptic servo, a force
sensor, or a hardware-readiness claim.

The software owns the complete low-gain torque composition:

``tau = gravity_scale * g(q) + kp * (q_ref - q) + kd * (dq_ref - dq) + J.T F``

During startup, the complete scaled gravity vector is present from the first
post-enable lock frame.  The final total-torque conditioner is the only
startup slew, so gravity is not attenuated twice by an independent scalar ramp
and a total-torque rate limit.  Damping remains immediately active while the
temporary joint lock hands off to the final controller.

``J.T F`` is zero in the joint ``hold`` and ``slow-step`` modes.  The guarded
``cartesian-hold`` mode anchors the current ``gripper_end`` position after a
fresh post-enable snapshot and uses the existing translational virtual spring
and world-aligned wrench mapping.  It controls no absolute Cartesian pose and
does not command orientation.  The separate ``pose-hold`` mode anchors the
complete current ``base_link_T_gripper_end`` transform and adds a bounded 6D
restoring wrench through the same ``J.T @ wrench`` path.  Both task-space modes
keep software joint stiffness at zero so there is only one spring owner.

The dispatched RobStride MIT ``kp`` is always zero.  By default ``kd`` is also
zero and the software owns all damping.  An explicit driver ``kd`` may instead
provide motor-local damping while software damping is disabled; that path
requires verified volatile driver torque limits on all six joints.  The host
still applies its final absolute torque and slew-rate bounds to the gravity and
task-space wrench it owns before calling :meth:`B601RSBackend.send_mit`.
Feedback uses the existing explicit ``mechPos`` request path; its source
freshness remains unverified by MotorBridge 0.4.9 and is reported as such.

``gravity-calibration`` hands the startup lock to a normal draggable
gravity/damping controller.  Once ACTIVE, the operator may move joint1..joint6,
release the arm, label the approach path with ``P`` or ``N``, and press ``R``
without Enter.  That fresh feedback pose becomes
the fixed software position reference for a settling interval followed by one
recording window (20 seconds by default); completion removes the position term
and returns to draggable gravity compensation so multiple poses can be collected
in one run.  The configured ``--kp`` is used while the fixed reference is
settling and recording.  ``gravity-calibration-fit`` is an offline-only mode: it
reduces complete windows, pairs repeated poses, selects informative poses, and
fits a candidate effective-gravity overlay without opening CAN.

Reaching ``--duration-sec`` ends the active experiment, but does not authorize
drive disable.  Like the first Ctrl-C or an ordinary handled control fault, it
enters the same latched all-arm gravity/damping protective hold.  A subsequent
Ctrl-C/SIGUSR1 is still required for the final six-joint safe stop unless both
live and cached complete-frame dispatch have become impossible.

When the operator presses Ctrl-C once, the probe enters a latched protective
hold for all six arm joints (the gripper motor, joint7, is outside this
backend).  This is deliberately not a position lock: software Kp is zero,
gravity is recomputed from the current measured pose every cycle, and one
moderate damping owner opposes motion.  The current feedback pose is used only
as the next MIT reference so a human can drag the arm while gravity remains
compensated.  A receive, numeric, motion-monitor, or model failure first falls
back to the last completely dispatched pose and gravity-only torque while
motor-local damping remains active when configured.  A second Ctrl-C/SIGUSR1,
or repeated inability to dispatch both live and cached complete six-joint
frames, requests the full six-joint safe stop.

This module is motion-capable only after all of ``--allow-motion``,
``--auto-disable`` and the exact confirmation string are supplied.  The
``feedback-stability`` mode is a separate observation-only path: it uses the
same explicit ``mechPos`` reads with ``allow_motion=False`` and never changes
the vendor mode or enable state.  Automated tests exercise the pure helpers
only; they never construct a hardware backend.

At feedback admission, the observed J2 zero-state boundary is represented with
a calibrated ``[-6e-3, 0)`` rad projection to the URDF lower limit.  The latest
zero-state and post-fault captures extend the same treatment to J3 with a
calibrated ``[-3e-3, 0)`` compatibility window.  J4's rounded ``-1.57/1.57``
URDF entries are represented by the authoritative ``+/-pi/2`` nominal limits
inside this probe, with a separate ``0.001`` rad lower-bound feedback window
projected to ``-pi/2``.  Raw values are retained in evidence records.  Values
beyond each joint's compatibility window remain fail-closed, and outgoing
references are still bounded by the effective nominal limits without applying
this feedback projection.

After MIT mode setup and enable, the active path captures a fresh post-enable
snapshot and immediately sends a bounded, all-six-joint software position lock
at that snapshot.  The lock frame is repeated while the feedback gate is
checked, then the complete gravity vector is ramped in with the lock gains
still active.  A second, same-duration handoff ramps the temporary joint lock
out while the requested final controller (including pose impedance) ramps in.
This closes the enable-to-control torque gap without adding an unbounded vendor
position loop: the startup gains are read from the pinned YAML MIT configuration
and remain inside the existing final software torque conditioner.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import sys
import termios
import time
import tty
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TextIO

import numpy as np

# Direct ``python scripts/hardware/...`` invocation places the script directory
# on sys.path rather than the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hardware.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601CommandReceipt,
    B601DriverTorqueLimitReceipt,
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
    B601PoseVirtualSpring,
    B601TranslationalVirtualSpring,
    compute_b601_mit_torque,
    map_world_aligned_ee_wrench_to_joint_torque,
)

CONFIRMATION = "B601_RS_LOW_GAIN_MIT_GRAVITY_DAMPING"
FEEDBACK_STABILITY_CONFIRMATION = "B601_RS_FEEDBACK_STABILITY_READ_ONLY"
DRIVER_TORQUE_LIMIT_CONFIRMATION = "B601_RS_DRIVER_TORQUE_LIMIT_ROUNDTRIP"
_ZERO_SIX = (0.0,) * 6
_ALL_JOINTS = B601_RS_ARM_JOINT_NAMES
_FAULT_HOLD_JOINTS = _ALL_JOINTS
_FEEDBACK_BOUNDARY_NORMALIZATION_JOINTS = ("joint2", "joint3", "joint4")
# J2 was observed at -4.943 mrad and -4.517 mrad in two independent
# zero-state, read-only captures.  J3 was observed at -0.682 mrad in the latest
# zero-state capture and -2.216 mrad after the gravity fault.  Keep small,
# explicit margins for those measured states only; these are feedback-admission
# boundaries, not relaxed model limits.
_FEEDBACK_BOUNDARY_NORMALIZATION_MAX_NEGATIVE_RAD = {
    "joint2": 6.0e-3,
    "joint3": 3.0e-3,
    "joint4": 1.0e-3,
}
# The compatibility windows are immediately below each joint's nominal lower
# boundary.  J2/J3 are zero-referenced joints; J4's nominal lower boundary is
# the precise -pi/2 rating rather than the URDF's rounded -1.57 entry.
_FEEDBACK_BOUNDARY_NORMALIZATION_REFERENCE_RAD = {
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": -math.pi / 2.0,
}
# The source URDF stores J4 as +/-1.57 (rounded to two decimals), while the
# vendor rating and commissioning documentation specify +/-90 degrees.  Keep
# the source URDF immutable and use this precise nominal boundary only for the
# probe's effective feedback/command-limit representation.
_AUTHORITATIVE_POSITION_LIMIT_OVERRIDES_RAD = {
    "joint4": (-math.pi / 2.0, math.pi / 2.0),
}
_NON_FAULT_HOLD_JOINTS: tuple[str, ...] = ()
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
_DEFAULT_VELOCITY_ABORT_RAD_S = 0.12
_DEFAULT_ACCELERATION_ABORT_RAD_S2 = 4.0
_DEFAULT_MAX_ERROR_RAD = 0.15
_DEFAULT_FAULT_HOLD_MAX_AGE_SEC = 0.15
_DEFAULT_FAULT_HOLD_MAX_VELOCITY_RAD_S = 0.02
_DEFAULT_FAULT_HOLD_MAX_REBASE_RAD = 0.05
_DEFAULT_PROTECTIVE_HOLD_DAMPING_KD = 1.0
_DEFAULT_CALIBRATION_DURATION_SEC = 20.0
_DEFAULT_CALIBRATION_SETTLING_SEC = 2.0
_DEFAULT_CALIBRATION_POSE_TOLERANCE_RAD = 0.08
_DEFAULT_ARMING_SETTLE_SAMPLES = 4
_DRIVE_ENABLE_COUNTDOWN_SECONDS = 3
_OPERATOR_TTY_PATH = Path("/dev/tty")
_DEFAULT_CARTESIAN_STIFFNESS_N_PER_M = 20.0
_DEFAULT_CARTESIAN_DAMPING_N_S_PER_M = 1.0
_DEFAULT_CARTESIAN_MAX_FORCE_N = 2.0
_DEFAULT_CARTESIAN_MAX_DISPLACEMENT_M = 0.02
_DEFAULT_CARTESIAN_OVERLAY_LIMIT_SCALE = 0.25
_DEFAULT_ROTATIONAL_STIFFNESS_NM_PER_RAD = 0.5
_DEFAULT_ROTATIONAL_DAMPING_NM_S_PER_RAD = 0.05
_DEFAULT_ROTATIONAL_MAX_TORQUE_NM = 0.25
_DEFAULT_ROTATIONAL_MAX_DISPLACEMENT_RAD = math.radians(5.0)


class LowGainProbeError(RuntimeError):
    """A guarded probe precondition, safety stop, or vendor operation failed."""


class OperatorKeyReader:
    """Read single operator keys from the controlling terminal without Enter."""

    def __init__(self, tty_path: Path = _OPERATOR_TTY_PATH) -> None:
        self._fd: int | None = None
        self._saved_attributes: list[object] | None = None
        self._pending_approach_direction: str | None = None
        try:
            self._fd = os.open(tty_path, os.O_RDONLY | os.O_NONBLOCK)
            self._saved_attributes = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            termios.tcflush(self._fd, termios.TCIFLUSH)
        except OSError as exc:
            self.close()
            raise LowGainProbeError(
                "gravity-calibration requires an interactive controlling terminal "
                "for the R capture key"
            ) from exc

    def flush(self) -> None:
        if self._fd is not None:
            termios.tcflush(self._fd, termios.TCIFLUSH)

    def poll_record_request(self) -> bool:
        """Return whether ``R`` was pressed, preserving the legacy API.

        Formal calibration should use :meth:`poll_capture_request` so the
        operator can label the approach path with ``P`` or ``N`` immediately
        before ``R``.  Keeping this wrapper avoids changing the small terminal
        reader contract used by existing callers and tests.
        """

        requested, _direction = self.poll_capture_request()
        return requested

    def poll_capture_request(self) -> tuple[bool, str | None]:
        """Read one non-blocking key batch and return ``(R, approach)``.

        ``P`` and ``N`` label the manually chosen positive/negative approach
        path for the next recording.  The label is consumed by ``R``; an
        unlabelled ``R`` remains valid for diagnostic capture but cannot form
        a formally bidirectional pair during offline fitting.
        """

        if self._fd is None:
            return False, None
        try:
            pressed = os.read(self._fd, 64)
        except BlockingIOError:
            return False, None
        except OSError as exc:
            raise LowGainProbeError(
                "failed to read the gravity-calibration operator terminal"
            ) from exc
        direction: str | None = None
        requested = False
        for key in pressed.lower():
            if key == ord("p"):
                self._pending_approach_direction = "positive"
            elif key == ord("n"):
                self._pending_approach_direction = "negative"
            elif key == ord("r"):
                requested = True
                direction = self._pending_approach_direction
                self._pending_approach_direction = None
        return requested, direction

    def close(self) -> None:
        fd = self._fd
        saved_attributes = self._saved_attributes
        self._fd = None
        self._saved_attributes = None
        if fd is None:
            return
        if saved_attributes is not None:
            with contextlib.suppress(OSError, termios.error):
                termios.tcsetattr(fd, termios.TCSADRAIN, saved_attributes)
        with contextlib.suppress(OSError):
            os.close(fd)


@dataclass
class GravityCalibrationCapture:
    """Own the FREE_DRAG/SETTLING/RECORDING capture state machine.

    ``SETTLING`` is intentionally separate from ``RECORDING``.  The operator
    captures a pose only after releasing the arm, then the fixed reference is
    allowed to settle under the non-zero software Kp.  Samples collected while
    the reference is settling are retained in the run evidence but are not
    included in the reduced calibration observation.  A zero settling duration
    keeps the historical helper behaviour used by older callers/tests.
    """

    record_duration_sec: float
    settling_duration_sec: float = 0.0
    state: str = "FREE_DRAG"
    reference_qpos_rad: tuple[float, ...] | None = None
    started_elapsed_sec: float | None = None
    recording_started_elapsed_sec: float | None = None
    active_record_index: int | None = None
    records: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.record_duration_sec = _positive(
            self.record_duration_sec,
            name="calibration_duration_sec",
        )
        self.settling_duration_sec = _non_negative(
            self.settling_duration_sec,
            name="calibration_settling_sec",
        )

    def trigger(
        self,
        qpos_rad: Sequence[float],
        *,
        elapsed_sec: float,
        sample_start_index: int,
        approach_direction: str | None = None,
    ) -> dict[str, object] | None:
        if self.state != "FREE_DRAG":
            return None
        reference = _six(qpos_rad, name="calibration_reference_qpos_rad")
        started = _non_negative(elapsed_sec, name="calibration_trigger_elapsed_sec")
        if approach_direction is not None:
            approach_direction = str(approach_direction).strip().lower()
            if approach_direction not in ("positive", "negative"):
                raise ValueError(
                    "calibration approach_direction must be positive or negative"
                )
        record_index = len(self.records) + 1
        record: dict[str, object] = {
            "index": record_index,
            "trigger_elapsed_sec": started,
            "reference_qpos_rad": reference,
            "approach_direction": approach_direction,
            # ``sample_start_index`` is filled when RECORDING starts.  Keep the
            # trigger index separately so the settling interval can be audited.
            "trigger_sample_index": int(sample_start_index),
            "sample_start_index": None,
            "sample_end_index": None,
            "completed": False,
            "observed_duration_sec": 0.0,
            "settling_duration_sec": self.settling_duration_sec,
            "settling_sample_start_index": int(sample_start_index),
            "recording_started_elapsed_sec": None,
            "completion_reason": None,
            "valid": None,
            "invalid_reason": None,
        }
        self.records.append(record)
        self.state = (
            "SETTLING" if self.settling_duration_sec > 0.0 else "RECORDING"
        )
        self.reference_qpos_rad = reference
        self.started_elapsed_sec = started
        self.recording_started_elapsed_sec = (
            started if self.state == "RECORDING" else None
        )
        self.active_record_index = record_index
        if self.state == "RECORDING":
            record["sample_start_index"] = int(sample_start_index)
            record["recording_started_elapsed_sec"] = started
        return record

    def start_recording_if_due(
        self,
        *,
        elapsed_sec: float,
        sample_start_index: int,
    ) -> dict[str, object] | None:
        """Transition a captured reference from SETTLING to RECORDING."""

        if self.state != "SETTLING" or self.started_elapsed_sec is None:
            return None
        elapsed = _non_negative(elapsed_sec, name="calibration_elapsed_sec")
        if elapsed - self.started_elapsed_sec < self.settling_duration_sec:
            return None
        if self.active_record_index is None:
            raise ValueError("settling calibration record is missing")
        record = self.records[self.active_record_index - 1]
        self.state = "RECORDING"
        self.recording_started_elapsed_sec = elapsed
        record["sample_start_index"] = int(sample_start_index)
        record["recording_started_elapsed_sec"] = elapsed
        return record

    def complete_if_due(
        self,
        *,
        elapsed_sec: float,
        sample_end_index: int,
    ) -> dict[str, object] | None:
        if self.state != "RECORDING" or self.recording_started_elapsed_sec is None:
            return None
        elapsed = _non_negative(elapsed_sec, name="calibration_elapsed_sec")
        if elapsed - self.recording_started_elapsed_sec < self.record_duration_sec:
            return None
        return self._finish(
            elapsed_sec=elapsed,
            sample_end_index=sample_end_index,
            completed=True,
            reason="duration_complete",
        )

    def interrupt(
        self,
        *,
        elapsed_sec: float,
        sample_end_index: int,
        reason: str,
    ) -> dict[str, object] | None:
        if self.state not in ("SETTLING", "RECORDING"):
            return None
        return self._finish(
            elapsed_sec=max(0.0, float(elapsed_sec)),
            sample_end_index=sample_end_index,
            completed=False,
            reason=reason,
        )

    def reference(self, current_qpos_rad: Sequence[float]) -> tuple[float, ...]:
        current = _six(current_qpos_rad, name="calibration_current_qpos_rad")
        if self.state == "RECORDING":
            if self.reference_qpos_rad is None:
                raise ValueError("recording calibration reference is missing")
            return self.reference_qpos_rad
        if self.state == "SETTLING":
            if self.reference_qpos_rad is None:
                raise ValueError("settling calibration reference is missing")
            return self.reference_qpos_rad
        return current

    def record_elapsed_sec(self, elapsed_sec: float) -> float | None:
        if self.state != "RECORDING" or self.recording_started_elapsed_sec is None:
            return None
        return max(0.0, float(elapsed_sec) - self.recording_started_elapsed_sec)

    def _finish(
        self,
        *,
        elapsed_sec: float,
        sample_end_index: int,
        completed: bool,
        reason: str,
    ) -> dict[str, object]:
        if self.started_elapsed_sec is None or self.active_record_index is None:
            raise ValueError("active calibration record is incomplete")
        record = self.records[self.active_record_index - 1]
        if record["sample_start_index"] is None:
            # A stop during SETTLING has no valid recording samples.  Use the
            # trigger index as a closed empty interval for deterministic audit.
            record["sample_start_index"] = int(record["trigger_sample_index"])
        record["sample_end_index"] = int(sample_end_index)
        record["completed"] = completed
        record["valid"] = completed
        if not completed:
            record["invalid_reason"] = reason
        record["observed_duration_sec"] = max(
            0.0,
            elapsed_sec - self.started_elapsed_sec,
        )
        record["recording_duration_sec"] = (
            max(
                0.0,
                elapsed_sec - self.recording_started_elapsed_sec,
            )
            if self.recording_started_elapsed_sec is not None
            else 0.0
        )
        record["completion_reason"] = reason
        self.state = "FREE_DRAG"
        self.reference_qpos_rad = None
        self.started_elapsed_sec = None
        self.recording_started_elapsed_sec = None
        self.active_record_index = None
        return record


def evaluate_gravity_calibration_capture(
    capture: GravityCalibrationCapture | None,
) -> dict[str, object]:
    """Summarize whether an interactive capture is complete enough to accept."""

    if capture is None:
        return {
            "status": "NOT_APPLICABLE",
            "completed_record_count": 0,
            "incomplete_record_count": 0,
            "reasons": (),
        }
    completed_count = sum(bool(record["completed"]) for record in capture.records)
    incomplete_count = len(capture.records) - completed_count
    reasons: list[str] = []
    if completed_count == 0:
        reasons.append("no complete calibration recording window")
    if incomplete_count:
        reasons.append(
            f"{incomplete_count} calibration recording window(s) are incomplete"
        )
    return {
        "status": "PASS" if not reasons else "FAIL",
        "completed_record_count": completed_count,
        "incomplete_record_count": incomplete_count,
        "reasons": tuple(reasons),
    }


def _calibration_sample_vector(
    sample: Mapping[str, object],
    key: str,
    *,
    name: str | None = None,
) -> np.ndarray:
    """Read one finite six-joint vector from a captured sample."""

    value = sample.get(key)
    if value is None:
        raise ValueError(f"calibration sample is missing {key}")
    return np.asarray(_six(value, name=name or key), dtype=float)


def _calibration_interval(
    sample_count: int,
    sample_start_index: int,
    sample_end_index: int,
) -> tuple[int, int]:
    """Validate a half-open calibration sample interval."""

    def integer_index(value: object, *, name: str) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        try:
            normalized = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        try:
            exact = float(value) == float(normalized)
        except (TypeError, ValueError, OverflowError):
            exact = False
        if not exact:
            raise ValueError(f"{name} must be an integer")
        return normalized

    start = integer_index(sample_start_index, name="sample_start_index")
    end = integer_index(sample_end_index, name="sample_end_index")
    if start < 0 or end <= start or end > sample_count:
        raise ValueError(
            "calibration sample interval must satisfy "
            f"0 <= start < end <= {sample_count}"
        )
    return start, end


def evaluate_calibration_window(
    samples: Sequence[Mapping[str, object]],
    *,
    sample_start_index: int,
    sample_end_index: int,
    max_velocity_rad_s: float | None = None,
    max_position_span_rad: float | None = None,
    reject_torque_saturation: bool = True,
) -> dict[str, object]:
    """Reduce and quality-check one fixed-reference recording window.

    The function deliberately does not invent a universal stability threshold.
    A caller may provide commissioning-specific velocity/span criteria; when a
    criterion is omitted it is reported as ``NOT_CONFIGURED`` instead of being
    silently tightened in code.  Every frame in the interval is still required
    to be finite and complete.  The returned medians represent one physical
    pose observation; individual 20 Hz frames are not treated as independent
    poses by the fitter.
    """

    start, end = _calibration_interval(
        len(samples),
        sample_start_index,
        sample_end_index,
    )
    window = tuple(samples[start:end])
    if not window:
        raise ValueError("calibration window must contain at least one sample")

    qpos = np.vstack(
        [_calibration_sample_vector(sample, "qpos_rad") for sample in window]
    )
    qvel = np.vstack(
        [_calibration_sample_vector(sample, "qvel_fd_rad_s") for sample in window]
    )
    gravity = np.vstack(
        [_calibration_sample_vector(sample, "gravity_tau_nm") for sample in window]
    )
    position = np.vstack(
        [_calibration_sample_vector(sample, "position_tau_nm") for sample in window]
    )
    requested = np.vstack(
        [_calibration_sample_vector(sample, "requested_tau_nm") for sample in window]
    )
    conditioned = np.vstack(
        [
            _calibration_sample_vector(sample, "conditioned_tau_nm")
            for sample in window
        ]
    )
    if any(
        not np.all(np.isfinite(values))
        for values in (qpos, qvel, gravity, position, requested, conditioned)
    ):
        raise ValueError("calibration window contains non-finite numeric values")

    qref_value = window[0].get("qref_rad")
    if qref_value is None:
        raise ValueError("calibration window is missing qref_rad")
    qref = np.asarray(_six(qref_value, name="qref_rad"), dtype=float)
    if any(
        not np.allclose(
            _calibration_sample_vector(sample, "qref_rad"),
            qref,
            rtol=0.0,
            atol=1e-12,
        )
        for sample in window
    ):
        raise ValueError("calibration qref changed inside a recording window")

    reasons: list[str] = []
    max_velocity = float(np.max(np.abs(qvel)))
    position_span = np.max(qpos, axis=0) - np.min(qpos, axis=0)
    max_span = float(np.max(position_span))
    torque_limited_count = sum(bool(sample.get("torque_limited")) for sample in window)
    slew_limited_count = sum(bool(sample.get("slew_limited")) for sample in window)
    if max_velocity_rad_s is None:
        velocity_gate: dict[str, object] = {
            "status": "NOT_CONFIGURED",
            "limit_rad_s": None,
            "observed_max_rad_s": max_velocity,
        }
    else:
        velocity_limit = _positive(
            max_velocity_rad_s,
            name="calibration_max_velocity_rad_s",
        )
        velocity_passed = max_velocity <= velocity_limit
        velocity_gate = {
            "status": "PASS" if velocity_passed else "FAIL",
            "limit_rad_s": velocity_limit,
            "observed_max_rad_s": max_velocity,
        }
        if not velocity_passed:
            reasons.append(
                "finite-difference velocity exceeds calibration stability criterion"
            )
    if max_position_span_rad is None:
        span_gate: dict[str, object] = {
            "status": "NOT_CONFIGURED",
            "limit_rad": None,
            "observed_max_span_rad": max_span,
        }
    else:
        span_limit = _positive(
            max_position_span_rad,
            name="calibration_max_position_span_rad",
        )
        span_passed = max_span <= span_limit
        span_gate = {
            "status": "PASS" if span_passed else "FAIL",
            "limit_rad": span_limit,
            "observed_max_span_rad": max_span,
        }
        if not span_passed:
            reasons.append("position span exceeds calibration stability criterion")
    if reject_torque_saturation and torque_limited_count:
        reasons.append("calibration window contains absolute torque saturation")
    if reject_torque_saturation and slew_limited_count:
        reasons.append("calibration window contains torque slew saturation")

    def median_vector(values: np.ndarray) -> tuple[float, ...]:
        return tuple(float(value) for value in np.median(values, axis=0))

    def mad_vector(values: np.ndarray) -> tuple[float, ...]:
        med = np.median(values, axis=0)
        return tuple(
            float(value)
            for value in np.median(np.abs(values - med), axis=0)
        )

    elapsed_values = [
        _finite(sample.get("elapsed_sec"), name="calibration_sample_elapsed_sec")
        for sample in window
        if sample.get("elapsed_sec") is not None
    ]
    duration_sec = (
        max(elapsed_values) - min(elapsed_values) if len(elapsed_values) >= 2 else None
    )
    return {
        "status": "PASS" if not reasons else "INVALID",
        "valid": not reasons,
        "sample_start_index": start,
        "sample_end_index": end,
        "sample_count": len(window),
        "duration_sec": duration_sec,
        "reference_qpos_rad": tuple(float(value) for value in qref),
        "qpos_median_rad": median_vector(qpos),
        "qpos_mad_rad": mad_vector(qpos),
        "qpos_span_rad": tuple(float(value) for value in position_span),
        "qvel_median_rad_s": median_vector(qvel),
        "qvel_mad_rad_s": mad_vector(qvel),
        "gravity_tau_median_nm": median_vector(gravity),
        "gravity_tau_mad_nm": mad_vector(gravity),
        "position_tau_median_nm": median_vector(position),
        "position_tau_mad_nm": mad_vector(position),
        "requested_tau_median_nm": median_vector(requested),
        "conditioned_tau_median_nm": median_vector(conditioned),
        "max_abs_velocity_rad_s": max_velocity,
        "max_position_span_rad": max_span,
        "stability_velocity_gate": velocity_gate,
        "stability_position_span_gate": span_gate,
        "torque_limited_count": torque_limited_count,
        "slew_limited_count": slew_limited_count,
        "reasons": tuple(reasons),
        "approach_direction": window[0].get("approach_direction"),
    }


def reduce_gravity_calibration_windows(
    payload: Mapping[str, object],
    *,
    max_velocity_rad_s: float | None = None,
    max_position_span_rad: float | None = None,
    reject_torque_saturation: bool = True,
) -> dict[str, object]:
    """Extract complete records and reduce each to one pose observation."""

    calibration = payload.get("gravity_calibration")
    if not isinstance(calibration, Mapping):
        raise TypeError("payload is missing gravity_calibration metadata")
    records = calibration.get("records")
    samples = payload.get("samples")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("gravity_calibration.records must be a sequence")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise TypeError("payload.samples must be a sequence")
    reduced: list[dict[str, object]] = []
    invalid: list[dict[str, object]] = []
    completed_count = 0
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            invalid.append({"reason": "record is not an object"})
            continue
        if not bool(raw_record.get("completed")):
            invalid.append(
                {
                    "index": raw_record.get("index"),
                    "reason": raw_record.get("completion_reason")
                    or "recording window is incomplete",
                }
            )
            continue
        completed_count += 1
        start = raw_record.get("sample_start_index")
        end = raw_record.get("sample_end_index")
        if start is None or end is None:
            invalid.append(
                {
                    "index": raw_record.get("index"),
                    "reason": "complete record has no sample interval",
                }
            )
            continue
        try:
            start_index, end_index = _calibration_interval(
                len(samples),
                start,
                end,
            )
            summary = evaluate_calibration_window(
                samples,
                sample_start_index=start_index,
                sample_end_index=end_index,
                max_velocity_rad_s=max_velocity_rad_s,
                max_position_span_rad=max_position_span_rad,
                reject_torque_saturation=reject_torque_saturation,
            )
        except (TypeError, ValueError) as exc:
            invalid.append(
                {
                    "index": raw_record.get("index"),
                    "reason": str(exc),
                }
            )
            continue
        summary["record_index"] = raw_record.get("index")
        summary["trigger_elapsed_sec"] = raw_record.get("trigger_elapsed_sec")
        summary["settling_duration_sec"] = raw_record.get("settling_duration_sec", 0.0)
        summary["recording_started_elapsed_sec"] = raw_record.get(
            "recording_started_elapsed_sec"
        )
        summary["pair_id"] = raw_record.get("pair_id")
        summary["approach_direction"] = raw_record.get("approach_direction")
        if bool(summary["valid"]):
            reduced.append(summary)
        else:
            invalid.append(summary)
    return {
        "windows": reduced,
        "invalid_records": invalid,
        "completed_record_count": completed_count,
        "valid_record_count": len(reduced),
    }


def pair_gravity_calibration_windows(
    windows: Sequence[Mapping[str, object]],
    *,
    pose_tolerance_rad: float,
    require_opposite_direction: bool = False,
) -> dict[str, object]:
    """Pair opposite-direction captures of the same pose by reference pose.

    Direction is an operator action and is not inferable from a static JSON
    window.  Explicit ``pair_id`` values are preferred; otherwise this helper
    greedily pairs the nearest unpaired references within the caller-supplied
    tolerance.  When ``require_opposite_direction`` is true, only windows
    explicitly labelled ``positive`` and ``negative`` can form a pair.  The
    default remains geometric-only for diagnostics and backwards-compatible
    helper use; the formal fitter enables the strict mode.
    """

    tolerance = _positive(
        pose_tolerance_rad,
        name="calibration_pose_tolerance_rad",
    )
    normalized: list[dict[str, object]] = []
    for index, window in enumerate(windows):
        qref = np.asarray(
            _six(window.get("reference_qpos_rad"), name="reference_qpos_rad"),
            dtype=float,
        )
        normalized.append({"index": index, "window": window, "qref": qref})

    by_pair_id: dict[object, list[dict[str, object]]] = {}
    unlabelled: list[dict[str, object]] = []
    for item in normalized:
        pair_id = item["window"].get("pair_id")
        if pair_id is None:
            unlabelled.append(item)
        else:
            by_pair_id.setdefault(pair_id, []).append(item)

    pairs: list[dict[str, object]] = []
    unpaired: list[dict[str, object]] = []
    direction_rejected_count = 0

    def directions_are_opposite(
        first_window: Mapping[str, object],
        second_window: Mapping[str, object],
    ) -> bool:
        first_direction = first_window.get("approach_direction")
        second_direction = second_window.get("approach_direction")
        return (
            first_direction in ("positive", "negative")
            and second_direction in ("positive", "negative")
            and first_direction != second_direction
        )

    for pair_id, items in by_pair_id.items():
        if len(items) == 2:
            if require_opposite_direction and not directions_are_opposite(
                items[0]["window"], items[1]["window"]
            ):
                direction_rejected_count += 1
                unpaired.extend(item["window"] for item in items)
                continue
            pairs.append(
                {
                    "pair_id": pair_id,
                    "window_indices": tuple(int(item["index"]) for item in items),
                    "windows": tuple(item["window"] for item in items),
                    "pairing": "explicit_pair_id",
                    "direction_verified": directions_are_opposite(
                        items[0]["window"], items[1]["window"]
                    ),
                }
            )
        else:
            unpaired.extend(item["window"] for item in items)

    remaining = list(unlabelled)
    while remaining:
        first = remaining.pop(0)
        if not remaining:
            unpaired.append(first["window"])
            break
        candidate_items = [
            item
            for item in remaining
            if not require_opposite_direction
            or directions_are_opposite(first["window"], item["window"])
        ]
        if not candidate_items:
            if require_opposite_direction and remaining:
                nearest_any_distance = min(
                    float(np.max(np.abs(first["qref"] - item["qref"])))
                    for item in remaining
                )
                if nearest_any_distance <= tolerance:
                    direction_rejected_count += 1
            unpaired.append(first["window"])
            continue
        distances = [
            float(np.max(np.abs(first["qref"] - item["qref"])))
            for item in candidate_items
        ]
        nearest_index = int(np.argmin(np.asarray(distances, dtype=float)))
        nearest_distance = distances[nearest_index]
        if nearest_distance <= tolerance:
            second = candidate_items[nearest_index]
            remaining.remove(second)
            pairs.append(
                {
                    "pair_id": None,
                    "window_indices": (int(first["index"]), int(second["index"])),
                    "windows": (first["window"], second["window"]),
                    "pairing": "nearest_reference_pose",
                    "reference_distance_rad": nearest_distance,
                    "direction_verified": directions_are_opposite(
                        first["window"], second["window"]
                    ),
                }
            )
        else:
            unpaired.append(first["window"])

    return {
        "pairs": tuple(pairs),
        "unpaired": tuple(unpaired),
        "pair_count": len(pairs),
        "unpaired_count": len(unpaired),
        "pose_tolerance_rad": tolerance,
        "require_opposite_direction": require_opposite_direction,
        "direction_rejected_count": direction_rejected_count,
        "verified_pair_count": sum(
            bool(pair.get("direction_verified")) for pair in pairs
        ),
    }


def _calibration_pair_observation(
    pair: Mapping[str, object],
) -> dict[str, object]:
    windows = pair.get("windows")
    if not isinstance(windows, Sequence) or len(windows) != 2:
        raise ValueError("calibration pair must contain exactly two windows")

    def average(key: str) -> tuple[float, ...]:
        values = np.vstack(
            [
                np.asarray(_six(window.get(key), name=key), dtype=float)
                for window in windows
            ]
        )
        return tuple(float(value) for value in np.mean(values, axis=0))

    first = windows[0]
    return {
        "pair_id": pair.get("pair_id"),
        "pairing": pair.get("pairing"),
        "record_indices": tuple(window.get("record_index") for window in windows),
        "reference_qpos_rad": average("reference_qpos_rad"),
        "gravity_tau_nm": average("gravity_tau_median_nm"),
        "position_tau_nm": average("position_tau_median_nm"),
        "position_tau_mad_nm": average("position_tau_mad_nm"),
        "approach_directions": tuple(
            window.get("approach_direction") for window in windows
        ),
        "direction_verified": all(
            window.get("approach_direction") is not None for window in windows
        )
        and len(
            {
                window.get("approach_direction") for window in windows
            }
        )
        == 2,
        "source_window_count": 2,
        "source_record": first.get("record_index"),
    }


def payload_gravity_torque_regressor(
    model: B601PinocchioModel,
    q_arm: Sequence[float],
) -> np.ndarray:
    """Return the 6-by-4 static torque basis for an unknown tool payload.

    Columns are ``[mass_kg, first_moment_x, first_moment_y,
    first_moment_z]`` with first moments expressed in the ``gripper_end``
    frame.  The basis uses the existing world-aligned end-effector Jacobian;
    it does not mutate the URDF or contact the vendor backend.
    """

    q = _six(q_arm, name="payload_regressor_q_arm")
    transform = np.asarray(model.forward_kinematics(q), dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("payload regressor FK returned an invalid transform")
    rotation = transform[:3, :3]
    try:
        gravity = np.asarray(
            tuple(float(value) for value in model.gravity_mps2),
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("gravity_mps2 must contain three finite values") from exc
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity_mps2 must contain three finite values")
    wrench_basis = np.zeros((6, 4), dtype=float)
    wrench_basis[:3, 0] = gravity
    for column in range(3):
        wrench_basis[3:, column + 1] = np.cross(rotation[:, column], gravity)
    # Pinocchio's static equation is g - J.T f_ext, so an added payload's
    # gravity contribution is -J.T times its downward gravity wrench.
    regressor = -np.asarray(model.arm_jacobian(q), dtype=float).T @ wrench_basis
    if regressor.shape != (6, 4) or not np.all(np.isfinite(regressor)):
        raise ValueError("payload gravity regressor is invalid")
    return regressor


def select_calibration_poses(
    pose_observations: Sequence[Mapping[str, object]],
    *,
    max_pose_count: int | None = None,
    model: str = "global-scale-joint-bias",
    kinematic_model: B601PinocchioModel | None = None,
    pose_tolerance_rad: float | None = None,
) -> dict[str, object]:
    """Select information-rich, non-duplicate poses.

    The greedy SVD/log-volume criterion ranks candidates by the information
    added to the regression.  Once a candidate is selected, any remaining
    reference pose within ``pose_tolerance_rad`` is rejected as a duplicate.
    This makes a long run of repeated windows contribute one physical pose,
    rather than silently turning repeated 20 Hz observations into artificial
    identifiability.
    """

    if model not in (
        "global-scale",
        "global-scale-joint-bias",
        "global-scale-payload",
    ):
        raise ValueError("unknown calibration model")
    if model == "global-scale-payload" and kinematic_model is None:
        raise ValueError(
            "global-scale-payload pose selection requires a kinematic model"
        )
    if pose_tolerance_rad is not None:
        pose_tolerance = _positive(
            pose_tolerance_rad,
            name="calibration_pose_tolerance_rad",
        )
    else:
        pose_tolerance = None
    feature_width = {
        "global-scale": 1,
        "global-scale-joint-bias": 7,
        "global-scale-payload": 5,
    }[model]
    if max_pose_count is None:
        selected_limit = len(pose_observations)
    else:
        if isinstance(max_pose_count, bool) or int(max_pose_count) <= 0:
            raise ValueError("max_pose_count must be positive when provided")
        selected_limit = min(int(max_pose_count), len(pose_observations))
    if not pose_observations or selected_limit == 0:
        return {
            "selected_indices": (),
            "rejected_indices": tuple(range(len(pose_observations))),
            "duplicate_rejected_indices": (),
            "pose_dedup_tolerance_rad": pose_tolerance,
            "rank": 0,
            "singular_values": (),
            "condition_number": math.inf,
            "criterion": "greedy_svd_log_volume_with_pose_dedup",
        }

    features: list[np.ndarray] = []
    references: list[np.ndarray] = []
    for observation in pose_observations:
        references.append(
            np.asarray(
                _six(observation.get("reference_qpos_rad"), name="reference_qpos_rad"),
                dtype=float,
            )
        )
        gravity = np.asarray(
            _six(observation.get("gravity_tau_nm"), name="gravity_tau_nm"),
            dtype=float,
        )
        block = np.zeros((6, feature_width), dtype=float)
        block[:, 0] = gravity
        if model == "global-scale-joint-bias":
            block[:, 1:] = np.eye(6, dtype=float)
        elif model == "global-scale-payload":
            block[:, 1:] = payload_gravity_torque_regressor(
                kinematic_model,
                observation.get("reference_qpos_rad"),
            )
        features.append(block)

    selected: list[int] = []
    remaining = set(range(len(features)))
    duplicate_rejected: set[int] = set()
    best_signature: tuple[int, float] | None = None
    while remaining and len(selected) < selected_limit:
        candidate_scores: list[tuple[tuple[int, float], int, np.ndarray]] = []
        for index in remaining:
            candidate_indices = selected + [index]
            matrix = np.vstack([features[item] for item in candidate_indices])
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            tolerance = max(matrix.shape) * np.finfo(float).eps * (
                float(singular_values[0]) if singular_values.size else 0.0
            )
            rank = int(np.count_nonzero(singular_values > tolerance))
            positive = singular_values[singular_values > tolerance]
            volume = float(np.sum(np.log(positive))) if positive.size else -math.inf
            candidate_scores.append(((rank, volume), index, singular_values))
        candidate_scores.sort(key=lambda item: (item[0][0], item[0][1], -item[1]), reverse=True)
        signature, chosen, singular_values = candidate_scores[0]
        if best_signature is not None and signature == best_signature and len(selected) > 0:
            # Once every remaining candidate has the same information volume,
            # retaining more near-duplicate poses adds no identifiability.
            break
        selected.append(chosen)
        remaining.remove(chosen)
        if pose_tolerance is not None:
            duplicate_indices = {
                index
                for index in remaining
                if float(np.max(np.abs(references[index] - references[chosen])))
                <= pose_tolerance
            }
            remaining.difference_update(duplicate_indices)
            duplicate_rejected.update(duplicate_indices)
        best_signature = signature

    matrix = (
        np.vstack([features[item] for item in selected])
        if selected
        else np.empty((0, feature_width))
    )
    singular_values = (
        np.linalg.svd(matrix, compute_uv=False) if matrix.size else np.asarray(())
    )
    if singular_values.size:
        tolerance = max(matrix.shape) * np.finfo(float).eps * float(singular_values[0])
        rank = int(np.count_nonzero(singular_values > tolerance))
        condition = (
            math.inf
            if rank == 0 or singular_values[-1] <= tolerance
            else float(singular_values[0] / singular_values[-1])
        )
    else:
        rank = 0
        condition = math.inf
    return {
        "selected_indices": tuple(selected),
        "rejected_indices": tuple(sorted(remaining | duplicate_rejected)),
        "duplicate_rejected_indices": tuple(sorted(duplicate_rejected)),
        "pose_dedup_tolerance_rad": pose_tolerance,
        "rank": rank,
        "singular_values": tuple(float(value) for value in singular_values),
        "condition_number": condition,
        "criterion": "greedy_svd_log_volume_with_pose_dedup",
    }


def fit_gravity_calibration(
    payload: Mapping[str, object],
    *,
    pose_tolerance_rad: float,
    model: str = "global-scale-joint-bias",
    kinematic_model: B601PinocchioModel | None = None,
    max_pose_count: int | None = None,
    max_velocity_rad_s: float | None = None,
    max_position_span_rad: float | None = None,
    require_bidirectional: bool = True,
) -> dict[str, object]:
    """Fit an auditable effective gravity correction from reduced windows."""

    if model not in (
        "global-scale",
        "global-scale-joint-bias",
        "global-scale-payload",
    ):
        raise ValueError(
            "calibration model must be global-scale, global-scale-joint-bias, "
            "or global-scale-payload"
        )
    if model == "global-scale-payload" and kinematic_model is None:
        raise ValueError("global-scale-payload fitting requires a kinematic model")
    reduced = reduce_gravity_calibration_windows(
        payload,
        max_velocity_rad_s=max_velocity_rad_s,
        max_position_span_rad=max_position_span_rad,
    )
    windows = reduced["windows"]
    invalid_record_count = len(reduced["invalid_records"])
    if not windows:
        return {
            "status": "FAIL",
            "model": model,
            "reasons": ("no valid complete calibration windows",),
            "reduction": reduced,
        }
    paired = pair_gravity_calibration_windows(
        windows,
        pose_tolerance_rad=pose_tolerance_rad,
        require_opposite_direction=require_bidirectional,
    )
    if require_bidirectional and not paired["pairs"]:
        return {
            "status": "FAIL",
            "model": model,
            "reasons": ("no bidirectional pose pair was identified",),
            "reduction": reduced,
            "pairing": paired,
        }
    observations = [
        _calibration_pair_observation(pair) for pair in paired["pairs"]
    ]
    if not require_bidirectional:
        observations.extend(
            {
                "pair_id": None,
                "pairing": "unpaired_single_window",
                "record_indices": (window.get("record_index"),),
                "reference_qpos_rad": window.get("reference_qpos_rad"),
                "gravity_tau_nm": window.get("gravity_tau_median_nm"),
                "position_tau_nm": window.get("position_tau_median_nm"),
                "position_tau_mad_nm": window.get("position_tau_mad_nm"),
                "approach_directions": (window.get("approach_direction"),),
                "direction_verified": False,
                "source_window_count": 1,
            }
            for window in paired["unpaired"]
        )
    selection = select_calibration_poses(
        observations,
        max_pose_count=max_pose_count,
        model=model,
        kinematic_model=kinematic_model,
        pose_tolerance_rad=pose_tolerance_rad,
    )
    selected = [observations[index] for index in selection["selected_indices"]]
    if not selected:
        return {
            "status": "FAIL",
            "model": model,
            "reasons": ("pose selection returned no observations",),
            "reduction": reduced,
            "pairing": paired,
            "pose_selection": selection,
        }

    profile = payload.get("profile")
    if not isinstance(profile, Mapping):
        return {
            "status": "FAIL",
            "model": model,
            "reasons": ("payload is missing profile metadata",),
            "reduction": reduced,
            "pairing": paired,
            "pose_selection": selection,
        }
    try:
        nominal_scale = _positive(profile.get("gravity_scale"), name="gravity_scale")
        software_kp = _six(
            profile.get("software_kp_nm_per_rad"),
            name="software_kp_nm_per_rad",
        )
    except (TypeError, ValueError) as exc:
        return {
            "status": "FAIL",
            "model": model,
            "reasons": (str(exc),),
            "reduction": reduced,
            "pairing": paired,
            "pose_selection": selection,
        }
    if not any(abs(value) > 0.0 for value in software_kp):
        return {
            "status": "FAIL",
            "model": model,
            "reasons": (
                "software Kp is zero; position residual is unobservable",
            ),
            "reduction": reduced,
            "pairing": paired,
            "pose_selection": selection,
        }

    rows: list[list[float]] = []
    target: list[float] = []
    for observation in selected:
        gravity = np.asarray(
            _six(observation["gravity_tau_nm"], name="gravity_tau_nm"),
            dtype=float,
        )
        residual = np.asarray(
            _six(observation["position_tau_nm"], name="position_tau_nm"),
            dtype=float,
        )
        nominal_gravity = gravity / nominal_scale
        if model == "global-scale":
            rows.extend([[float(value)] for value in nominal_gravity])
        elif model == "global-scale-joint-bias":
            for joint_index, value in enumerate(nominal_gravity):
                row = [0.0] * 7
                row[0] = float(value)
                row[joint_index + 1] = 1.0
                rows.append(row)
        else:
            payload_basis = payload_gravity_torque_regressor(
                kinematic_model,
                observation.get("reference_qpos_rad"),
            )
            for joint_index, value in enumerate(nominal_gravity):
                row = [float(value)] + [float(item) for item in payload_basis[joint_index]]
                rows.append(row)
        target.extend(float(value) for value in residual)
    design = np.asarray(rows, dtype=float)
    observed = np.asarray(target, dtype=float)
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(observed)):
        return {
            "status": "FAIL",
            "model": model,
            "reasons": ("calibration design or residual contains non-finite values",),
            "reduction": reduced,
            "pairing": paired,
            "pose_selection": selection,
        }
    if np.linalg.norm(observed) <= np.finfo(float).eps:
        return {
            "status": "FAIL",
            "model": model,
            "reasons": (
                "position residual is numerically zero; no gravity correction is observable",
            ),
            "reduction": reduced,
            "pairing": paired,
            "pose_selection": selection,
        }

    solution, _residuals, rank, singular_values = np.linalg.lstsq(
        design,
        observed,
        rcond=None,
    )
    delta_scale = float(solution[0])
    bias = np.zeros(6, dtype=float)
    payload_parameters = np.zeros(4, dtype=float)
    if model == "global-scale-joint-bias":
        bias[:] = solution[1:]
    elif model == "global-scale-payload":
        payload_parameters[:] = solution[1:]
        if payload_parameters[0] < 0.0:
            # Mass is physically non-negative.  Hold it at the boundary and
            # re-solve the remaining first-moment/global-scale columns.
            payload_parameters[0] = 0.0
            free_solution, *_ = np.linalg.lstsq(
                design[:, (0, 2, 3, 4)],
                observed,
                rcond=None,
            )
            delta_scale = float(free_solution[0])
            payload_parameters[1:] = free_solution[1:]
    if delta_scale < -nominal_scale:
        delta_scale = -nominal_scale
        if model == "global-scale-joint-bias":
            corrected = observed - design[:, 0] * delta_scale
            for joint_index in range(6):
                bias[joint_index] = float(
                    np.mean(
                        corrected[joint_index :: 6]
                    )
                )
        elif model == "global-scale-payload":
            payload_parameters[1:] = np.linalg.lstsq(
                design[:, 2:],
                observed - design[:, 0] * delta_scale,
                rcond=None,
            )[0]
    prediction = design[:, 0] * delta_scale
    if model == "global-scale-joint-bias":
        prediction += np.asarray(
            [bias[index % 6] for index in range(len(observed))],
            dtype=float,
        )
    elif model == "global-scale-payload":
        prediction += design[:, 1:] @ payload_parameters
    residual_after = observed - prediction
    tolerance = max(design.shape) * np.finfo(float).eps * float(singular_values[0])
    effective_rank = int(np.count_nonzero(singular_values > tolerance))
    condition_number = (
        math.inf
        if not singular_values.size or singular_values[-1] <= tolerance
        else float(singular_values[0] / singular_values[-1])
    )
    per_joint_before = np.sqrt(
        np.mean(np.asarray(observed).reshape(-1, 6) ** 2, axis=0)
    )
    per_joint_after = np.sqrt(
        np.mean(np.asarray(residual_after).reshape(-1, 6) ** 2, axis=0)
    )
    rms_before = float(np.sqrt(np.mean(observed**2)))
    rms_after = float(np.sqrt(np.mean(residual_after**2)))
    expected_rank = {
        "global-scale": 1,
        "global-scale-joint-bias": 7,
        "global-scale-payload": 5,
    }[model]
    formal_direction_ok = require_bidirectional and all(
        bool(observation["direction_verified"]) for observation in selected
    )
    fit_reasons: list[str] = []
    if invalid_record_count:
        fit_reasons.append(
            f"{invalid_record_count} calibration record(s) are incomplete or invalid"
        )
    if not formal_direction_ok:
        fit_reasons.append("selected observations are not explicitly bidirectional")
    if effective_rank < expected_rank:
        fit_reasons.append(
            f"regression rank {effective_rank} is below expected rank {expected_rank}"
        )
    if rms_after >= rms_before:
        fit_reasons.append("training residual did not decrease")
    fit_passed = not fit_reasons
    return {
        "status": "PASS" if fit_passed else "CANDIDATE",
        "model": model,
        "reasons": tuple(fit_reasons),
        "nominal_gravity_scale": nominal_scale,
        "gravity_scale_correction": delta_scale,
        "effective_gravity_scale": nominal_scale + delta_scale,
        "joint_bias_nm": tuple(float(value) for value in bias),
        "payload_mass_kg": (
            float(payload_parameters[0]) if model == "global-scale-payload" else None
        ),
        "payload_first_moment_kg_m": (
            tuple(float(value) for value in payload_parameters[1:])
            if model == "global-scale-payload"
            else None
        ),
        "payload_com_m": (
            tuple(
                float(value / payload_parameters[0])
                for value in payload_parameters[1:]
            )
            if model == "global-scale-payload" and payload_parameters[0] > 0.0
            else None
        ),
        "software_kp_nm_per_rad": software_kp,
        "observation_count": len(selected),
        "source_window_count": sum(
            int(observation["source_window_count"]) for observation in selected
        ),
        "paired_observation_count": sum(
            int(observation["source_window_count"]) == 2 for observation in selected
        ),
        "direction_verified_pair_count": sum(
            bool(observation["direction_verified"]) for observation in selected
        ),
        "pose_selection": selection,
        "fit": {
            "rank": effective_rank,
            "least_squares_rank": int(rank),
            "expected_rank": expected_rank,
            "singular_values": tuple(float(value) for value in singular_values),
            "condition_number": condition_number,
            "rms_residual_before_nm": rms_before,
            "rms_residual_after_nm": rms_after,
            "residual_reduced": rms_after < rms_before,
            "invalid_record_count": invalid_record_count,
            "per_joint_rms_before_nm": tuple(float(value) for value in per_joint_before),
            "per_joint_rms_after_nm": tuple(float(value) for value in per_joint_after),
        },
        "reduction": reduced,
        "pairing": paired,
        "selected_observations": tuple(selected),
        "unpaired_used": not require_bidirectional and bool(paired["unpaired"]),
        "formal_direction_verified": formal_direction_ok,
    }


def evaluate_gravity_calibration_model(
    fit: Mapping[str, object],
    payload: Mapping[str, object],
    *,
    kinematic_model: B601PinocchioModel | None = None,
    pose_tolerance_rad: float = _DEFAULT_CALIBRATION_POSE_TOLERANCE_RAD,
    max_velocity_rad_s: float | None = None,
    max_position_span_rad: float | None = None,
    require_bidirectional: bool = True,
) -> dict[str, object]:
    """Evaluate a fitted correction on an independent capture payload."""

    if fit.get("status") not in ("PASS", "CANDIDATE"):
        return {"status": "NOT_APPLICABLE", "reason": "fit did not produce a model"}
    model = str(fit.get("model"))
    nominal_scale = _positive(
        fit.get("nominal_gravity_scale"),
        name="nominal_gravity_scale",
    )
    delta = _finite(fit.get("gravity_scale_correction"), name="gravity_scale_correction")
    bias = np.asarray(
        _six(fit.get("joint_bias_nm"), name="joint_bias_nm"),
        dtype=float,
    )
    if model == "global-scale-payload":
        first_moment_value = fit.get("payload_first_moment_kg_m")
        try:
            first_moment = tuple(
                _finite(value, name="payload_first_moment_kg_m")
                for value in first_moment_value
            )
        except TypeError as exc:
            raise ValueError(
                "payload_first_moment_kg_m must contain three finite values"
            ) from exc
        if len(first_moment) != 3:
            raise ValueError(
                "payload_first_moment_kg_m must contain three finite values"
            )
        payload_parameters = np.asarray(
            (_finite(fit.get("payload_mass_kg"), name="payload_mass_kg"),)
            + first_moment,
            dtype=float,
        )
    else:
        payload_parameters = np.zeros(4, dtype=float)
    if model == "global-scale-payload" and kinematic_model is None:
        return {
            "status": "FAIL",
            "reason": "payload validation requires a kinematic model",
        }
    reduced = reduce_gravity_calibration_windows(
        payload,
        max_velocity_rad_s=max_velocity_rad_s,
        max_position_span_rad=max_position_span_rad,
    )
    windows = reduced["windows"]
    invalid_record_count = len(reduced["invalid_records"])
    if not windows:
        return {
            "status": "FAIL",
            "reason": "validation payload has no valid windows",
            "reduction": reduced,
        }
    paired = pair_gravity_calibration_windows(
        windows,
        pose_tolerance_rad=pose_tolerance_rad,
        require_opposite_direction=require_bidirectional,
    )
    if require_bidirectional and not paired["pairs"]:
        return {
            "status": "FAIL",
            "reason": "validation has no explicitly bidirectional pose pair",
            "reduction": reduced,
            "pairing": paired,
        }
    observations = [
        _calibration_pair_observation(pair) for pair in paired["pairs"]
    ]
    if not require_bidirectional:
        observations.extend(
            {
                "reference_qpos_rad": window.get("reference_qpos_rad"),
                "gravity_tau_nm": window.get("gravity_tau_median_nm"),
                "position_tau_nm": window.get("position_tau_median_nm"),
                "source_window_count": 1,
                "direction_verified": False,
            }
            for window in paired["unpaired"]
        )
    if not observations:
        return {
            "status": "FAIL",
            "reason": "validation pose pairing returned no observations",
            "reduction": reduced,
            "pairing": paired,
        }
    before: list[np.ndarray] = []
    after: list[np.ndarray] = []
    for observation in observations:
        gravity = np.asarray(
            _six(observation["gravity_tau_nm"], name="gravity_tau_nm"),
            dtype=float,
        )
        residual = np.asarray(
            _six(observation["position_tau_nm"], name="position_tau_nm"),
            dtype=float,
        )
        nominal_gravity = gravity / nominal_scale
        correction = nominal_gravity * delta
        if model == "global-scale-joint-bias":
            correction = correction + bias
        elif model == "global-scale-payload":
            correction = correction + payload_gravity_torque_regressor(
                kinematic_model,
                observation.get("reference_qpos_rad"),
            ) @ payload_parameters
        before.append(residual)
        after.append(residual - correction)
    before_array = np.vstack(before)
    after_array = np.vstack(after)
    rms_before = float(np.sqrt(np.mean(before_array**2)))
    rms_after = float(np.sqrt(np.mean(after_array**2)))
    residual_reduced = rms_after < rms_before
    validation_reasons: list[str] = []
    if invalid_record_count:
        validation_reasons.append(
            f"{invalid_record_count} validation record(s) are incomplete or invalid"
        )
    if not residual_reduced:
        validation_reasons.append("validation residual did not decrease")
    return {
        "status": "PASS" if not validation_reasons else "FAIL",
        "window_count": len(windows),
        "observation_count": len(observations),
        "rms_residual_before_nm": rms_before,
        "rms_residual_after_nm": rms_after,
        "residual_reduced": residual_reduced,
        "reasons": tuple(validation_reasons),
        "invalid_record_count": invalid_record_count,
        "pairing": paired,
        "per_joint_rms_before_nm": tuple(
            float(value) for value in np.sqrt(np.mean(before_array**2, axis=0))
        ),
        "per_joint_rms_after_nm": tuple(
            float(value) for value in np.sqrt(np.mean(after_array**2, axis=0))
        ),
        "reduction": reduced,
    }


def _load_json_object(path: Path, *, name: str) -> dict[str, object]:
    """Load a JSON object for the offline calibration path."""

    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LowGainProbeError(f"cannot read {name}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise LowGainProbeError(f"{name} must contain a JSON object")
    return decoded


def _write_json_object(path: Path, payload: Mapping[str, object]) -> None:
    """Write one calibration artifact, creating only its requested directory."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise LowGainProbeError(
            f"cannot write calibration artifact {destination}: {exc}"
        ) from exc


def _calibration_source_identity(
    payload: Mapping[str, object],
    *,
    config_path: Path,
    model: B601PinocchioModel,
) -> dict[str, object]:
    """Compare capture provenance with the currently pinned offline inputs."""

    expected = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "urdf_sha256": hashlib.sha256(model.urdf_path.read_bytes()).hexdigest(),
    }
    observed = {
        key: payload.get(key)
        for key in expected
    }
    mismatches = {
        key: {
            "expected": expected[key],
            "observed": observed[key],
        }
        for key in expected
        if observed[key] is not None and observed[key] != expected[key]
    }
    missing = tuple(key for key in expected if observed[key] is None)
    return {
        "status": (
            "FAIL" if mismatches else ("CANDIDATE" if missing else "PASS")
        ),
        "expected": expected,
        "observed": observed,
        "missing": missing,
        "mismatches": mismatches,
    }


def _run_gravity_calibration_fit(
    *,
    args: argparse.Namespace,
    model: B601PinocchioModel,
    source_root: Path,
) -> dict[str, object]:
    """Fit and report an effective gravity correction without opening CAN."""

    input_path = Path(args.calibration_input).expanduser().resolve(strict=True)
    payload = _load_json_object(input_path, name="calibration input")
    train_identity = _calibration_source_identity(
        payload,
        config_path=Path(args.config).expanduser().resolve(strict=True),
        model=model,
    )
    if train_identity["status"] == "FAIL":
        raise LowGainProbeError(
            "calibration input provenance does not match the pinned config/URDF: "
            f"{train_identity['mismatches']}"
        )
    fit = fit_gravity_calibration(
        payload,
        pose_tolerance_rad=args.calibration_pose_tolerance_rad,
        model=args.calibration_model,
        kinematic_model=model,
        max_pose_count=args.calibration_max_poses,
        max_velocity_rad_s=args.calibration_max_velocity_rad_s,
        max_position_span_rad=args.calibration_max_position_span_rad,
        require_bidirectional=not args.calibration_allow_unpaired,
    )
    validation_path = getattr(args, "calibration_validation_input", None)
    validation: dict[str, object]
    validation_identity: dict[str, object] | None = None
    validation_dataset_sha256: str | None = None
    if validation_path is None:
        validation = {
            "status": "NOT_PROVIDED",
            "reason": "independent validation capture was not supplied",
        }
    else:
        validation_resolved = (
            Path(validation_path).expanduser().resolve(strict=True)
        )
        validation_payload = _load_json_object(
            validation_resolved,
            name="calibration validation input",
        )
        validation_identity = _calibration_source_identity(
            validation_payload,
            config_path=Path(args.config).expanduser().resolve(strict=True),
            model=model,
        )
        if validation_identity["status"] == "FAIL":
            raise LowGainProbeError(
                "calibration validation provenance does not match the pinned "
                f"config/URDF: {validation_identity['mismatches']}"
            )
        validation_dataset_sha256 = hashlib.sha256(
            validation_resolved.read_bytes()
        ).hexdigest()
        validation = evaluate_gravity_calibration_model(
            fit,
            validation_payload,
            kinematic_model=model,
            pose_tolerance_rad=args.calibration_pose_tolerance_rad,
            max_velocity_rad_s=args.calibration_max_velocity_rad_s,
            max_position_span_rad=args.calibration_max_position_span_rad,
            require_bidirectional=not args.calibration_allow_unpaired,
        )
    fit_status = str(fit.get("status"))
    acceptance_status = (
        "PASS"
        if (
            fit_status == "PASS"
            and validation.get("status") == "PASS"
            and train_identity["status"] == "PASS"
            and (
                validation_identity is None
                or validation_identity["status"] == "PASS"
            )
        )
        else "CANDIDATE"
    )
    dataset_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    overlay = {
        "schema": "b601.gravity-calibration-overlay.v1",
        "activation": "candidate_only_until_independent_validation",
        "model": fit.get("model"),
        "nominal_gravity_scale": fit.get("nominal_gravity_scale"),
        "gravity_scale_correction": fit.get("gravity_scale_correction"),
        "effective_gravity_scale": fit.get("effective_gravity_scale"),
        "joint_bias_nm": fit.get("joint_bias_nm"),
        "payload_mass_kg": fit.get("payload_mass_kg"),
        "payload_first_moment_kg_m": fit.get("payload_first_moment_kg_m"),
        "payload_com_m": fit.get("payload_com_m"),
        "source": {
            "dataset_path": str(input_path),
            "dataset_sha256": dataset_sha256,
            "urdf_path": str(model.urdf_path),
            "urdf_sha256": hashlib.sha256(model.urdf_path.read_bytes()).hexdigest(),
            "rebot_source_root": str(source_root),
            "config_sha256": train_identity["expected"]["config_sha256"],
            "validation_dataset_sha256": validation_dataset_sha256,
        },
        "fit": fit.get("fit"),
        "validation": validation,
        "source_identity": {
            "training": train_identity,
            "validation": validation_identity,
        },
        "pose_selection": fit.get("pose_selection"),
        "selected_observations": fit.get("selected_observations"),
    }
    overlay_path = Path(args.calibration_overlay_output).expanduser().resolve()
    _write_json_object(overlay_path, overlay)
    report_path = getattr(args, "calibration_report_output", None)
    if report_path is not None:
        _write_json_object(
            Path(report_path),
            {
                "schema": "b601.gravity-calibration-report.v1",
                "acceptance_status": acceptance_status,
                "fit": fit,
                "validation": validation,
                "overlay_path": str(overlay_path),
                "dataset_sha256": dataset_sha256,
            },
        )
    return {
        "status": "OK" if acceptance_status == "PASS" else "FAIL",
        "mode": "gravity-calibration-fit",
        "acceptance_status": acceptance_status,
        "input_path": str(input_path),
        "overlay_path": str(overlay_path),
        "report_path": (
            str(Path(report_path).expanduser().resolve())
            if report_path is not None
            else None
        ),
        "dataset_sha256": dataset_sha256,
        "source_identity": {
            "training": train_identity,
            "validation": validation_identity,
        },
        "fit": fit,
        "validation": validation,
        "hardware_acceptance": "NOT_CLAIMED",
    }


def complete_active_duration(
    *,
    stop: dict[str, bool],
    calibration_capture: GravityCalibrationCapture | None,
    elapsed_sec: float,
    sample_end_index: int,
) -> str:
    """End ACTIVE without treating the duration boundary as disable authority."""

    if calibration_capture is not None and calibration_capture.state in (
        "SETTLING",
        "RECORDING",
    ):
        calibration_capture.interrupt(
            elapsed_sec=elapsed_sec,
            sample_end_index=sample_end_index,
            reason="probe_duration_complete",
        )
    # Duration completion itself is the first protective event.  A subsequent
    # Ctrl-C can therefore request the explicit final stop.
    stop["requested"] = True
    return "duration_complete"


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


def _non_negative(value: object, *, name: str) -> float:
    normalized = _finite(value, name=name)
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
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


def normalize_feedback_boundary_qpos(
    qpos_rad: Sequence[float],
    *,
    name: str = "feedback_qpos_rad",
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """Normalize only the explicitly admitted encoder boundary observations.

    RobStride ``mechPos`` may report a value just below a physical/model lower
    boundary after zeroing or at a rounded nominal limit.  J2 uses the measured
    ``-6e-3 <= q < 0`` boundary, J3 uses ``-3e-3 <= q < 0``, and J4 uses a
    separate ``0.001`` rad window immediately below its precise ``-pi/2``
    nominal boundary.  Every other joint and every more-negative value is
    returned unchanged so the caller's existing exact position-limit check
    still rejects it.  The raw value must be retained by callers that write an
    evidence record.
    """

    q = _six(qpos_rad, name=name)
    normalized = list(q)
    normalized_joint_names: list[str] = []
    for index, joint_name in enumerate(_ALL_JOINTS):
        value = q[index]
        boundary = _FEEDBACK_BOUNDARY_NORMALIZATION_MAX_NEGATIVE_RAD.get(joint_name)
        reference = _FEEDBACK_BOUNDARY_NORMALIZATION_REFERENCE_RAD.get(joint_name)
        if (
            boundary is not None
            and reference is not None
            and reference - boundary <= value < reference
        ):
            normalized[index] = reference
            normalized_joint_names.append(joint_name)
    return tuple(normalized), tuple(normalized_joint_names)


def build_feedback_admission_position_limits(
    position_limits_rad: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Build the read-side envelope without changing command limits.

    The three documented encoder-boundary windows are the only values that
    may extend the read-side lower boundary.  Outgoing MIT/POS_VEL references
    continue to use the strict ``position_limits_rad`` supplied by the pinned
    URDF.  Keeping this envelope explicit also lets the wired backend reject a
    finite but impossible parameter response before it reaches FK, gravity,
    impedance, or dispatch.
    """

    limits = tuple(
        (_finite(lower, name="position_limit_lower"),
         _finite(upper, name="position_limit_upper"))
        for lower, upper in position_limits_rad
    )
    if len(limits) != 6 or any(lower >= upper for lower, upper in limits):
        raise ValueError("position_limits_rad must contain six valid intervals")
    admission = list(limits)
    for index, joint_name in enumerate(_ALL_JOINTS):
        boundary = _FEEDBACK_BOUNDARY_NORMALIZATION_MAX_NEGATIVE_RAD.get(joint_name)
        reference = _FEEDBACK_BOUNDARY_NORMALIZATION_REFERENCE_RAD.get(joint_name)
        if boundary is None or reference is None:
            continue
        lower, upper = admission[index]
        admission[index] = (min(lower, reference - boundary), upper)
    return tuple(admission)


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


def _backend_feedback_diagnostics(state: object) -> tuple[dict[str, object], ...]:
    """Serialize the backend's per-joint admission evidence for a bad frame."""

    diagnostics: list[dict[str, object]] = []
    for joint in getattr(state, "joints", ()):
        diagnostics.append(
            {
                "name": getattr(joint, "name", None),
                "motor_id": getattr(joint, "motor_id", None),
                "feedback_id": getattr(joint, "feedback_id", None),
                "position_rad": getattr(joint, "position_rad", None),
                "position_source": getattr(joint, "position_source", None),
                "valid": getattr(joint, "valid", None),
                "stale": getattr(joint, "stale", None),
                "error": getattr(joint, "error", None),
            }
        )
    return tuple(diagnostics)


def _driver_torque_limit_receipt_payload(
    receipt: B601DriverTorqueLimitReceipt | None,
) -> dict[str, object] | None:
    if receipt is None:
        return None
    return {
        "operation": receipt.operation,
        "requested_joint_names": receipt.requested_joint_names,
        "target_limits_nm": receipt.target_limits_nm,
        "prior_limits_nm": receipt.prior_limits_nm,
        "verified_limits_nm": receipt.verified_limits_nm,
        "applied_joint_names": receipt.applied_joint_names,
        "restored_joint_names": receipt.restored_joint_names,
        "failed_joint_names": receipt.failed_joint_names,
        "errors": receipt.errors,
        "completed": receipt.completed,
        "rollback_attempted": receipt.rollback_attempted,
        "rollback_completed": receipt.rollback_completed,
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

    The returned tuple is the initial six-joint snapshot admitted for a
    protective hold.  A missing or invalid snapshot is never substituted with
    zeros or with the most recent command.  ``observed_age_sec`` is measured by
    this process's monotonic clock; it is deliberately not presented as vendor
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
    q, _ = normalize_feedback_boundary_qpos(
        qpos_rad,
        name="fault_hold_qpos_rad",
    )
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
        fault_q, _ = normalize_feedback_boundary_qpos(
            fault_qpos_rad,
            name="fault_qpos_rad",
        )
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


def _validate_fault_entry_snapshot(
    qpos_rad: Sequence[float] | None,
    *,
    qvel_rad_s: Sequence[float] | None,
    observed_age_sec: float,
    position_limits_rad: Sequence[tuple[float, float]],
    max_age_sec: float,
    max_velocity_rad_s: float,
) -> tuple[float, ...]:
    """Admit a complete current fault frame as a first protective reference.

    Unlike the stationary ``last_safe`` gate, a handled control fault may
    arrive while the arm is still moving.  The normal motion watchdog has
    already rejected anything above ``max_velocity_rad_s`` before this helper
    is called; this entry gate therefore checks that configured abort limit
    without applying the much tighter stationary-candidate threshold.
    """

    if qpos_rad is None:
        raise LowGainProbeError(
            "fault hold requires a complete current fault position snapshot"
        )
    if qvel_rad_s is None:
        raise LowGainProbeError(
            "fault hold requires a complete current fault velocity observation"
        )
    q, _ = normalize_feedback_boundary_qpos(
        qpos_rad,
        name="fault_entry_qpos_rad",
    )
    qvel = _six(qvel_rad_s, name="fault_entry_qvel_rad_s")
    age = _finite(observed_age_sec, name="fault_entry_observed_age_sec")
    max_age = _bounded(
        max_age_sec,
        name="fault_hold_max_age_sec",
        lower=0.01,
        upper=1.0,
    )
    if age < 0.0 or age > max_age:
        raise LowGainProbeError(
            "fault entry snapshot is stale: "
            f"age={age:.6f}s > max={max_age:.6f}s"
        )
    velocity_limit = _positive(
        max_velocity_rad_s,
        name="fault_entry_max_velocity_rad_s",
    )
    observed_max_velocity = max(abs(value) for value in qvel)
    if observed_max_velocity > velocity_limit:
        raise LowGainProbeError(
            "fault entry snapshot is moving too quickly: "
            f"max_abs_velocity={observed_max_velocity:.6f} rad/s > "
            f"{velocity_limit:.6f} rad/s"
        )
    limits = tuple(
        (float(lower), float(upper)) for lower, upper in position_limits_rad
    )
    if len(limits) != 6 or any(
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower >= upper
        for lower, upper in limits
    ):
        raise LowGainProbeError("fault entry received invalid position limits")
    for index, (value, (lower, upper)) in enumerate(zip(q, limits, strict=True)):
        if not lower <= value <= upper:
            raise LowGainProbeError(
                f"fault entry qpos_rad[{index}]={value:.6f} is outside "
                f"URDF limits [{lower:.6f}, {upper:.6f}]"
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
    being unsuitable as a future protective-hold admission reference. In that
    case the immutable last-safe candidate is retained; the strict validator
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


def evaluate_feedback_motion_guard(
    *,
    sequence: int,
    feedback_timestamp_sec: float,
    dt_sec: float,
    raw_qpos_rad: Sequence[float],
    normalized_qpos_rad: Sequence[float],
    previous_qvel_rad_s: Sequence[float] | None,
    current_qvel_rad_s: Sequence[float],
    velocity_abort_rad_s: float,
    acceleration_abort_rad_s2: float,
) -> tuple[tuple[float, ...] | None, dict[str, object] | None]:
    """Evaluate finite-difference velocity/acceleration before dispatch.

    The first accepted control sample has no previous velocity observation and
    therefore reports ``qacc_fd_rad_s2=None``. A rejected sample is returned
    separately and must not enter the commanded sample list or last-safe state.
    """

    timestamp = _finite(feedback_timestamp_sec, name="feedback_timestamp_sec")
    dt = _positive(dt_sec, name="feedback_dt_sec")
    raw_q = _six(raw_qpos_rad, name="raw_qpos_rad")
    normalized_q = _six(normalized_qpos_rad, name="normalized_qpos_rad")
    current_qvel = _six(current_qvel_rad_s, name="current_qvel_rad_s")
    velocity_limit = _positive(
        velocity_abort_rad_s,
        name="velocity_abort_rad_s",
    )
    acceleration_limit = _positive(
        acceleration_abort_rad_s2,
        name="acceleration_abort_rad_s2",
    )
    previous_qvel: tuple[float, ...] | None = None
    qacc: tuple[float, ...] | None = None
    if previous_qvel_rad_s is not None:
        previous_qvel = _six(previous_qvel_rad_s, name="previous_qvel_rad_s")
        qacc = tuple(
            (current - previous) / dt
            for current, previous in zip(
                current_qvel,
                previous_qvel,
                strict=True,
            )
        )
        if any(not math.isfinite(value) for value in qacc):
            raise ValueError("qacc_fd_rad_s2 must contain six finite values")

    guard_name: str | None = None
    offending_index: int | None = None
    observed_value: float | None = None
    limit_value: float | None = None
    velocity_index = int(np.argmax(np.abs(np.asarray(current_qvel, dtype=float))))
    maximum_velocity = abs(current_qvel[velocity_index])
    if maximum_velocity > velocity_limit:
        guard_name = "velocity"
        offending_index = velocity_index
        observed_value = maximum_velocity
        limit_value = velocity_limit
    elif qacc is not None:
        acceleration_index = int(np.argmax(np.abs(np.asarray(qacc, dtype=float))))
        maximum_acceleration = abs(qacc[acceleration_index])
        if maximum_acceleration > acceleration_limit:
            guard_name = "acceleration"
            offending_index = acceleration_index
            observed_value = maximum_acceleration
            limit_value = acceleration_limit

    if guard_name is None:
        return qacc, None
    assert offending_index is not None
    assert observed_value is not None
    assert limit_value is not None
    units = "rad/s" if guard_name == "velocity" else "rad/s^2"
    field = "qvel" if guard_name == "velocity" else "qacc"
    reason = (
        f"{guard_name} abort: {B601_RS_ARM_JOINT_NAMES[offending_index]} "
        f"abs_{field}={observed_value:.6f} > {limit_value:.6f} {units}"
    )
    return qacc, {
        "reason": reason,
        "guard": guard_name,
        "offending_joint": B601_RS_ARM_JOINT_NAMES[offending_index],
        "offending_joint_index": offending_index,
        "observed_abs_value": observed_value,
        "limit": limit_value,
        "units": units,
        "sequence": int(sequence),
        "feedback_timestamp_sec": timestamp,
        "feedback_dt_sec": dt,
        "raw_qpos_rad": raw_q,
        "normalized_qpos_rad": normalized_q,
        "previous_qvel_rad_s": previous_qvel,
        "current_qvel_rad_s": current_qvel,
        "qacc_fd_rad_s2": qacc,
        "command_dispatched": False,
        "eligible_for_last_safe": False,
    }


def evaluate_feedback_position_guard(
    *,
    sequence: int,
    feedback_timestamp_sec: float,
    raw_qpos_rad: Sequence[float],
    normalized_qpos_rad: Sequence[float],
    position_limits_rad: Sequence[tuple[float, float]],
    previous_qvel_rad_s: Sequence[float] | None = None,
) -> dict[str, object] | None:
    """Reject an out-of-limit position before it can enter active state.

    Feedback admission is deliberately separate from the motion watchdog.  A
    finite but implausible ``mechPos`` value must be retained as evidence and
    must never replace ``final_q``, a fault-hold snapshot, or ``last_safe``.
    URDF limits remain the sole position boundary; this helper adds no
    independent magnitude threshold.
    """

    timestamp = _finite(feedback_timestamp_sec, name="feedback_timestamp_sec")
    raw_q = _six(raw_qpos_rad, name="raw_qpos_rad")
    normalized_q = _six(normalized_qpos_rad, name="normalized_qpos_rad")
    limits = tuple(
        (_finite(lower, name="position_limit_lower"),
         _finite(upper, name="position_limit_upper"))
        for lower, upper in position_limits_rad
    )
    if len(limits) != 6 or any(lower >= upper for lower, upper in limits):
        raise ValueError("position_limits_rad must contain six valid intervals")
    previous_qvel = (
        None
        if previous_qvel_rad_s is None
        else _six(previous_qvel_rad_s, name="previous_qvel_rad_s")
    )
    for index, (value, (lower, upper)) in enumerate(
        zip(normalized_q, limits, strict=True)
    ):
        if lower <= value <= upper:
            continue
        reason = (
            f"feedback qpos_rad[{index}]={value:.6f} is outside "
            f"URDF limits [{lower:.6f}, {upper:.6f}]"
        )
        return {
            "reason": reason,
            "guard": "position_limits",
            "offending_joint": B601_RS_ARM_JOINT_NAMES[index],
            "offending_joint_index": index,
            "observed_value": value,
            "lower_limit": lower,
            "upper_limit": upper,
            "sequence": int(sequence),
            "feedback_timestamp_sec": timestamp,
            "feedback_dt_sec": None,
            "raw_qpos_rad": raw_q,
            "normalized_qpos_rad": normalized_q,
            "previous_qvel_rad_s": previous_qvel,
            "current_qvel_rad_s": None,
            "qacc_fd_rad_s2": None,
            "command_dispatched": False,
            "eligible_for_last_safe": False,
        }
    return None


def _validate_arming_settle_transition(
    current_qpos_rad: Sequence[float] | None,
    *,
    previous_qpos_rad: Sequence[float],
    current_timestamp_sec: float,
    previous_timestamp_sec: float,
    position_limits_rad: Sequence[tuple[float, float]],
    max_velocity_rad_s: float = _DEFAULT_VELOCITY_ABORT_RAD_S,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate one post-enable feedback transition before torque is allowed.

    The transition uses consecutive host receive timestamps and the normal
    active-path velocity abort threshold.  It deliberately does not reuse the
    stricter protective-hold stationary-admission gate: enabling the drives can
    produce a short, bounded transient before the startup lock settles.  This
    helper only validates observations; it never sends a motor command.
    """

    current = _validate_in_limit_qpos(
        current_qpos_rad,
        position_limits_rad=position_limits_rad,
        name="arming_settle_qpos_rad",
    )
    previous = _validate_in_limit_qpos(
        previous_qpos_rad,
        position_limits_rad=position_limits_rad,
        name="arming_settle_previous_qpos_rad",
    )
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
    qvel = _six(qvel, name="arming_settle_qvel_rad_s")
    velocity_limit = _positive(
        max_velocity_rad_s,
        name="arming_settle_velocity_limit_rad_s",
    )
    observed_max_velocity = max(abs(value) for value in qvel)
    if observed_max_velocity > velocity_limit:
        raise LowGainProbeError(
            "arming settle feedback is moving: "
            f"max_abs_velocity={observed_max_velocity:.6f} rad/s > "
            f"{velocity_limit:.6f} rad/s"
        )
    return current, qvel


def _validate_in_limit_qpos(
    qpos_rad: Sequence[float] | None,
    *,
    position_limits_rad: Sequence[tuple[float, float]],
    name: str,
) -> tuple[float, ...]:
    """Normalize and validate a complete position snapshot without a motion gate."""

    if qpos_rad is None:
        raise LowGainProbeError(f"{name} requires a complete feedback position snapshot")
    qpos, _ = normalize_feedback_boundary_qpos(qpos_rad, name=name)
    limits = tuple(
        (float(lower), float(upper)) for lower, upper in position_limits_rad
    )
    if len(limits) != 6 or any(
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower >= upper
        for lower, upper in limits
    ):
        raise LowGainProbeError(f"{name} received invalid position limits")
    for index, (value, (lower, upper)) in enumerate(zip(qpos, limits, strict=True)):
        if not lower <= value <= upper:
            raise LowGainProbeError(
                f"{name}[{index}]={value:.6f} is outside URDF limits "
                f"[{lower:.6f}, {upper:.6f}]"
            )
    return qpos


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

    The first Ctrl-C snapshot is retained for admission evidence, but it is not
    a fixed position target.  Every in-limit measured pose may become the next
    gravity-compensation reference, which is what makes the hold draggable.
    ``hold_qpos_rad`` and ``max_rebase_rad`` remain in the helper signature so
    callers and older evidence readers can pass the original admission data;
    they are deliberately not used as a drift watchdog.  A feedback, age, or
    velocity failure rejects only this live sample; the caller may continue
    with its previously verified gravity-only cache.
    """

    # Validate the initial reference shape for evidence compatibility, but do
    # not compare subsequent positions against it: manual dragging is an
    # intended protective-hold operation.
    _six(hold_qpos_rad, name="protective_hold_qpos_rad")
    _ = max_rebase_rad
    current = _validate_in_limit_qpos(
        qpos_rad,
        position_limits_rad=position_limits_rad,
        name="protective_hold_feedback_qpos_rad",
    )
    if qvel_rad_s is None:
        raise LowGainProbeError(
            "protective hold requires a complete feedback velocity observation"
        )
    qvel = _six(qvel_rad_s, name="protective_hold_feedback_qvel_rad_s")
    age = _finite(observed_age_sec, name="protective_hold_observed_age_sec")
    max_age = _bounded(
        max_age_sec,
        name="fault_hold_max_age_sec",
        lower=0.01,
        upper=1.0,
    )
    if age < 0.0 or age > max_age:
        raise LowGainProbeError(
            "protective hold feedback is stale: "
            f"age={age:.6f}s > max={max_age:.6f}s"
        )
    velocity_limit = _positive(
        max_velocity_rad_s,
        name="protective_hold_velocity_limit_rad_s",
    )
    observed_max_velocity = max(abs(value) for value in qvel)
    if observed_max_velocity > velocity_limit:
        raise LowGainProbeError(
            "protective hold velocity abort: "
            f"max_abs_velocity={observed_max_velocity:.6f} rad/s > "
            f"{velocity_limit:.6f} rad/s"
        )
    return current


def resolve_active_joint_names() -> tuple[str, ...]:
    """Return the complete B601 arm command set used by impedance modes."""

    return _ALL_JOINTS


def _validate_cli(args: argparse.Namespace) -> None:
    if args.mode == "gravity-calibration-fit":
        if getattr(args, "allow_motion", False):
            raise LowGainProbeError(
                "gravity-calibration-fit is offline and rejects --allow-motion"
            )
        if getattr(args, "auto_disable", False):
            raise LowGainProbeError(
                "gravity-calibration-fit is offline and rejects --auto-disable"
            )
        calibration_input = getattr(args, "calibration_input", None)
        if calibration_input is None:
            raise LowGainProbeError(
                "gravity-calibration-fit requires --calibration-input"
            )
        if not Path(calibration_input).expanduser().is_file():
            raise LowGainProbeError(
                "gravity-calibration-fit calibration input does not exist: "
                f"{calibration_input}"
            )
        if getattr(args, "calibration_overlay_output", None) is None:
            raise LowGainProbeError(
                "gravity-calibration-fit requires --calibration-overlay-output"
            )
        _positive(
            getattr(
                args,
                "calibration_pose_tolerance_rad",
                _DEFAULT_CALIBRATION_POSE_TOLERANCE_RAD,
            ),
            name="calibration_pose_tolerance_rad",
        )
        if (
            getattr(args, "calibration_max_poses", None) is not None
            and (
                isinstance(args.calibration_max_poses, bool)
                or int(args.calibration_max_poses) <= 0
            )
        ):
            raise ValueError("calibration_max_poses must be positive")
        for name in (
            "calibration_max_velocity_rad_s",
            "calibration_max_position_span_rad",
        ):
            value = getattr(args, name, None)
            if value is not None:
                _positive(value, name=name)
        return
    if args.mode == "feedback-stability":
        if getattr(args, "driver_kd", 0.0) != 0.0:
            raise LowGainProbeError(
                "feedback-stability rejects --driver-kd; it is observation-only"
            )
        if getattr(args, "apply_driver_torque_limits", False):
            raise LowGainProbeError(
                "feedback-stability rejects --apply-driver-torque-limits; "
                "it is observation-only"
            )
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
        _positive(args.rate_hz, name="rate_hz")
        _positive(args.duration_sec, name="duration_sec")
        _positive(
            args.fault_hold_max_velocity_rad_s,
            name="fault_hold_max_velocity_rad_s",
        )
        return
    if args.mode == "driver-torque-limit-roundtrip":
        if not args.allow_motion:
            raise LowGainProbeError(
                "driver-torque-limit-roundtrip requires --allow-motion"
            )
        if not args.auto_disable:
            raise LowGainProbeError(
                "driver-torque-limit-roundtrip requires explicit --auto-disable"
            )
        if args.confirmation != DRIVER_TORQUE_LIMIT_CONFIRMATION:
            raise LowGainProbeError(
                "driver-torque-limit-roundtrip requires --confirmation "
                f"{DRIVER_TORQUE_LIMIT_CONFIRMATION}"
            )
        if getattr(args, "apply_driver_torque_limits", False):
            raise LowGainProbeError(
                "driver-torque-limit-roundtrip already performs the explicit "
                "temporary write; omit --apply-driver-torque-limits"
            )
        _bounded(
            args.torque_limit_scale,
            name="torque_limit_scale",
            lower=0.0,
            upper=1.0,
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
    if args.mode not in (
        "hold",
        "slow-step",
        "cartesian-hold",
        "pose-hold",
        "gravity-calibration",
        "gravity-calibration-fit",
    ):
        raise ValueError(
            "mode must be hold, slow-step, cartesian-hold, pose-hold, or "
            "gravity-calibration"
        )
    _positive(args.rate_hz, name="rate_hz")
    _positive(args.duration_sec, name="duration_sec")
    _positive(args.warmup_sec, name="warmup_sec")
    _non_negative(args.freeze_sec, name="freeze_sec")
    _non_negative(args.kp, name="kp")
    _non_negative(args.kd, name="kd")
    if args.mode == "gravity-calibration":
        _positive(
            getattr(
                args,
                "calibration_duration_sec",
                _DEFAULT_CALIBRATION_DURATION_SEC,
            ),
            name="calibration_duration_sec",
        )
        _non_negative(
            getattr(
                args,
                "calibration_settling_sec",
                _DEFAULT_CALIBRATION_SETTLING_SEC,
            ),
            name="calibration_settling_sec",
        )
    _non_negative(
        getattr(
            args,
            "protective_hold_kd",
            _DEFAULT_PROTECTIVE_HOLD_DAMPING_KD,
        ),
        name="protective_hold_kd",
    )
    driver_kd = _non_negative(
        getattr(args, "driver_kd", 0.0),
        name="driver_kd",
    )
    if driver_kd > 0.0 and not args.apply_driver_torque_limits:
        raise LowGainProbeError(
            "non-zero --driver-kd requires --apply-driver-torque-limits"
        )
    target_kp = getattr(args, "target_kp", None)
    target_kd = getattr(args, "target_kd", None)
    if target_kp is not None:
        _non_negative(target_kp, name="target_kp")
    if target_kd is not None:
        _non_negative(target_kd, name="target_kd")
    if driver_kd > 0.0 and (
        args.kd > 0.0 or (target_kd is not None and target_kd > 0.0)
    ):
        raise LowGainProbeError(
            "non-zero --driver-kd requires software --kd 0 and no non-zero "
            "--target-kd"
        )
    _non_negative(args.gravity_scale, name="gravity_scale")
    _bounded(
        args.torque_limit_scale,
        name="torque_limit_scale",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
    )
    _positive(args.torque_rise_time_sec, name="torque_rise_time_sec")
    _positive(args.velocity_abort_rad_s, name="velocity_abort_rad_s")
    _positive(args.acceleration_abort_rad_s2, name="acceleration_abort_rad_s2")
    _positive(args.max_error_rad, name="max_error_rad")
    _bounded(args.step_rad, name="step_rad", lower=0.0, upper=_MAX_STEP_RAD)
    _positive(args.step_speed_rad_s, name="step_speed_rad_s")
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
    _non_negative(
        getattr(
            args,
            "cartesian_stiffness_n_per_m",
            _DEFAULT_CARTESIAN_STIFFNESS_N_PER_M,
        ),
        name="cartesian_stiffness_n_per_m",
    )
    _non_negative(
        getattr(
            args,
            "cartesian_damping_n_s_per_m",
            _DEFAULT_CARTESIAN_DAMPING_N_S_PER_M,
        ),
        name="cartesian_damping_n_s_per_m",
    )
    _positive(
        getattr(args, "cartesian_max_force_n", _DEFAULT_CARTESIAN_MAX_FORCE_N),
        name="cartesian_max_force_n",
    )
    _positive(
        getattr(
            args,
            "cartesian_max_displacement_m",
            _DEFAULT_CARTESIAN_MAX_DISPLACEMENT_M,
        ),
        name="cartesian_max_displacement_m",
    )
    _positive(
        getattr(
            args,
            "cartesian_overlay_limit_scale",
            _DEFAULT_CARTESIAN_OVERLAY_LIMIT_SCALE,
        ),
        name="cartesian_overlay_limit_scale",
    )
    _non_negative(
        getattr(
            args,
            "rotational_stiffness_nm_per_rad",
            _DEFAULT_ROTATIONAL_STIFFNESS_NM_PER_RAD,
        ),
        name="rotational_stiffness_nm_per_rad",
    )
    _non_negative(
        getattr(
            args,
            "rotational_damping_nm_s_per_rad",
            _DEFAULT_ROTATIONAL_DAMPING_NM_S_PER_RAD,
        ),
        name="rotational_damping_nm_s_per_rad",
    )
    _positive(
        getattr(
            args,
            "rotational_max_torque_nm",
            _DEFAULT_ROTATIONAL_MAX_TORQUE_NM,
        ),
        name="rotational_max_torque_nm",
    )
    _positive(
        getattr(
            args,
            "rotational_max_displacement_rad",
            _DEFAULT_ROTATIONAL_MAX_DISPLACEMENT_RAD,
        ),
        name="rotational_max_displacement_rad",
    )
    if args.mode == "slow-step" and args.step_rad <= 0.0:
        raise ValueError("slow-step mode requires step_rad > 0")


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
    limits_list = [
        (float(lo), float(hi)) for lo, hi in zip(lower, upper, strict=True)
    ]
    # The pinned source URDF rounds J4's +/-90-degree rating to +/-1.57.  Use
    # the precise nominal boundary in this probe without modifying the source
    # model; the feedback-admission layer below still supplies only the
    # separately documented 0.001 rad measurement window.
    for index, joint_name in enumerate(_ALL_JOINTS):
        override = _AUTHORITATIVE_POSITION_LIMIT_OVERRIDES_RAD.get(joint_name)
        if override is not None:
            limits_list[index] = override
    limits = tuple(limits_list)
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

    scale = _bounded(
        scale,
        name="torque_limit_scale",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
    )
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
    limit = _positive(
        stationarity_velocity_limit_rad_s,
        name="stationarity_velocity_limit_rad_s",
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
    feedback_position_limits = build_feedback_admission_position_limits(
        position_limits
    )
    period = 1.0 / _positive(rate_hz, name="rate_hz")
    started = _finite(clock(), name="clock")
    deadline = started + _positive(duration_sec, name="duration_sec")
    next_deadline = started
    ledger = FrequencyLedger()
    positions: list[tuple[float, ...]] = []
    raw_positions: list[tuple[float, ...]] = []
    normalization_events: list[tuple[str, ...]] = []
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
                raw_q = _six(state.positions_rad, name="feedback_qpos_rad")
                q, normalized_joint_names = normalize_feedback_boundary_qpos(
                    raw_q,
                    name="feedback_qpos_rad",
                )
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
            raw_positions.append(raw_q)
            normalization_events.append(normalized_joint_names)
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
        "position_limits_rad": position_limits,
        "feedback_position_limits_rad": feedback_position_limits,
        "position_limit_semantics": (
            "pinned URDF limits with precise J4 +/-pi/2 nominal override; "
            "feedback-only boundary windows are reported separately"
        ),
        "feedback_admission_semantics": (
            "backend rejects finite positions outside the explicit read-side "
            "envelope before FK/gravity/impedance/dispatch; approved J2/J3/J4 "
            "boundary windows are read-only and do not relax outgoing limits"
        ),
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
        "feedback_boundary_normalization": {
            "joints": _FEEDBACK_BOUNDARY_NORMALIZATION_JOINTS,
            "intervals_rad": {
                joint_name: (
                    _FEEDBACK_BOUNDARY_NORMALIZATION_REFERENCE_RAD[joint_name]
                    - _FEEDBACK_BOUNDARY_NORMALIZATION_MAX_NEGATIVE_RAD[joint_name],
                    _FEEDBACK_BOUNDARY_NORMALIZATION_REFERENCE_RAD[joint_name],
                )
                for joint_name in _FEEDBACK_BOUNDARY_NORMALIZATION_JOINTS
            },
            "projection_rad": {
                joint_name: _FEEDBACK_BOUNDARY_NORMALIZATION_REFERENCE_RAD[joint_name]
                for joint_name in _FEEDBACK_BOUNDARY_NORMALIZATION_JOINTS
            },
            "interval_semantics": "inclusive_lower_exclusive_upper",
            "applied_sample_count": sum(
                bool(names) for names in normalization_events
            ),
            "applied_joint_counts": {
                joint_name: sum(joint_name in names for names in normalization_events)
                for joint_name in _FEEDBACK_BOUNDARY_NORMALIZATION_JOINTS
            },
            "raw_values_preserved": True,
            "semantics": (
                "feedback_admission_only; URDF limits and outgoing references remain strict"
            ),
        },
        "samples": [
            {
                "receive_timestamp_sec": timestamp,
                "raw_qpos_rad": raw_q,
                "qpos_rad": q,
                "boundary_normalized_joint_names": normalized_joint_names,
            }
            for timestamp, raw_q, q, normalized_joint_names in zip(
                timestamps,
                raw_positions,
                positions,
                normalization_events,
                strict=True,
            )
        ],
        "notes": [
            "This mode only requests vendor feedback and reads mechPos; it never changes mode, enables, disables, or sends MIT/POS_VEL control frames.",
            "Run only after the operator has independently made the arm safe; this probe does not alter the existing enable state.",
            "The stationarity gate is diagnostic evidence for protective-hold review, not permission to relax the active safety threshold.",
            "J2 feedback readings in [-6e-3, 0) rad and J3 readings in [-3e-3, 0) rad are represented as 0 rad at admission; J4 readings in [-pi/2-1e-3, -pi/2) rad are represented as -pi/2. Raw readings are retained per sample, and values beyond each compatibility window remain outside the strict effective limit.",
            "MotorBridge 0.4.9 provides no source timestamp or sequence, so freshness remains unverified.",
        ],
    }
    return payload


def _run_driver_torque_limit_roundtrip_probe(
    backend: B601RSBackend,
    *,
    config_path: Path,
    source_root: Path,
    model: B601PinocchioModel,
    position_limits: tuple[tuple[float, float], ...],
    torque_limits_nm: tuple[float, ...],
) -> dict[str, object]:
    """Verify the six-arm-joint ``0x700B`` transaction while disabled."""

    target_names = _ALL_JOINTS
    connected = False
    preflight_disable: B601ModeReceipt | None = None
    cleanup_receipt: B601ModeReceipt | None = None
    apply_receipt: B601DriverTorqueLimitReceipt | None = None
    restore_receipt: B601DriverTorqueLimitReceipt | None = None
    feedback_validated = False
    errors: list[str] = []
    try:
        backend.connect()
        connected = True
        state = backend.poll_feedback(request_feedback=True)
        if (
            not state.valid
            or getattr(state, "stale", False)
            or state.positions_rad is None
        ):
            raise LowGainProbeError(
                "driver torque-limit roundtrip requires one complete valid feedback "
                f"snapshot; last_fault={state.fault_reason or 'unknown'}"
            )
        qpos, _ = normalize_feedback_boundary_qpos(
            state.positions_rad,
            name="driver_limit_feedback_qpos_rad",
        )
        for index, (value, (lower, upper)) in enumerate(
            zip(qpos, position_limits, strict=True)
        ):
            if not lower <= value <= upper:
                raise LowGainProbeError(
                    f"driver limit feedback qpos_rad[{index}]={value:.6f} is "
                    f"outside URDF limits [{lower:.6f}, {upper:.6f}]"
                )
        feedback_validated = True
        preflight_disable = backend.safe_stop(joint_names=target_names)
        if not preflight_disable.completed:
            raise LowGainProbeError(
                "preflight safe_stop failed: "
                + "; ".join(preflight_disable.errors)
            )
        apply_receipt = backend.apply_temporary_driver_torque_limits(
            torque_limits_nm,
            joint_names=target_names,
        )
        if not apply_receipt.completed:
            raise LowGainProbeError(
                "temporary driver torque-limit apply failed: "
                + "; ".join(apply_receipt.errors)
            )
    except (B601HardwareError, LowGainProbeError, OSError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    except Exception as exc:  # noqa: BLE001 - preserve vendor failure evidence
        errors.append(f"unexpected roundtrip failure: {exc}")
    finally:
        if connected:
            try:
                cleanup_receipt = backend.safe_stop(joint_names=target_names)
                if not cleanup_receipt.completed:
                    errors.extend(
                        f"cleanup safe_stop: {error}"
                        for error in cleanup_receipt.errors
                    )
                    if not cleanup_receipt.errors:
                        errors.append(
                            "cleanup safe_stop incomplete without vendor error"
                        )
            except Exception as exc:  # noqa: BLE001 - preserve cleanup evidence
                errors.append(f"cleanup safe_stop: {exc}")
            try:
                restore_receipt = backend.restore_driver_torque_limits()
                if not restore_receipt.completed:
                    errors.extend(
                        f"restore driver torque limit: {error}"
                        for error in restore_receipt.errors
                    )
                    if not restore_receipt.errors:
                        errors.append(
                            "restore driver torque limit incomplete without vendor error"
                        )
            except Exception as exc:  # noqa: BLE001 - preserve cleanup evidence
                errors.append(f"restore driver torque limit: {exc}")
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001 - preserve cleanup evidence
                errors.append(f"close: {exc}")

    completed = (
        feedback_validated
        and preflight_disable is not None
        and preflight_disable.completed
        and apply_receipt is not None
        and apply_receipt.completed
        and cleanup_receipt is not None
        and cleanup_receipt.completed
        and restore_receipt is not None
        and restore_receipt.completed
        and not errors
    )
    return {
        "status": "OK" if completed else "FAIL",
        "mode": "driver-torque-limit-roundtrip",
        "confirmation": DRIVER_TORQUE_LIMIT_CONFIRMATION,
        "channel": backend.config.channel,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "rebot_source_root": str(source_root),
        "urdf_path": str(model.urdf_path),
        "urdf_sha256": hashlib.sha256(model.urdf_path.read_bytes()).hexdigest(),
        "transport": "motorbridge_wired_local",
        "joint_names": target_names,
        "target_limits_nm": torque_limits_nm,
        "parameter": "RobStride limit_torque 0x700B",
        "parameter_persistence": "volatile_not_saved",
        "feedback_validated": feedback_validated,
        "motors_enabled_by_probe": False,
        "motion_commands_sent": False,
        "mode_transitions_sent": False,
        "parameter_write_attempted": apply_receipt is not None,
        "preflight_disable_receipt": _mode_receipt_payload(preflight_disable),
        "apply_receipt": _driver_torque_limit_receipt_payload(apply_receipt),
        "cleanup_receipt": _mode_receipt_payload(cleanup_receipt),
        "restore_receipt": _driver_torque_limit_receipt_payload(restore_receipt),
        "errors": tuple(errors),
        "acceptance_status": "PASS" if completed else "FAIL",
        "hardware_acceptance": "NOT_CLAIMED",
        "notes": [
            "This mode never changes control mode, enables a motor, or sends MIT/POS_VEL commands.",
            "It writes every arm motor's volatile RobStride limit_torque (0x700B), verifies every readback, safe-stops again, and restores every exact observed original value with readback.",
            "A PASS validates the guarded parameter transaction only; it is not a motion, force-control, or haptic acceptance result.",
        ],
    }


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


def gravity_ramp_fraction(*, elapsed_sec: float, rise_time_sec: float) -> float:
    """Return one shared linear startup scale for the complete gravity vector."""

    elapsed = _non_negative(elapsed_sec, name="elapsed_sec")
    rise_time = _positive(rise_time_sec, name="rise_time_sec")
    return min(1.0, elapsed / rise_time)


def apply_shared_gravity_ramp(
    breakdown: B601MitTorqueBreakdown,
    *,
    fraction: float,
) -> B601MitTorqueBreakdown:
    """Scale gravity as one vector while leaving damping and impedance live."""

    if not isinstance(breakdown, B601MitTorqueBreakdown):
        raise TypeError("breakdown must be a B601MitTorqueBreakdown")
    scale = _bounded(fraction, name="fraction", lower=0.0, upper=1.0)
    position = np.asarray(breakdown.position_torque_nm, dtype=float)
    damping = np.asarray(breakdown.damping_torque_nm, dtype=float)
    gravity = scale * np.asarray(breakdown.gravity_torque_nm, dtype=float)
    overlay = np.asarray(breakdown.overlay_effective_nm, dtype=float)
    requested = position + damping + gravity + overlay
    return B601MitTorqueBreakdown(
        position_torque_nm=breakdown.position_torque_nm,
        damping_torque_nm=breakdown.damping_torque_nm,
        gravity_torque_nm=tuple(float(value) for value in gravity),
        overlay_requested_nm=breakdown.overlay_requested_nm,
        overlay_effective_nm=breakdown.overlay_effective_nm,
        requested_torque_nm=tuple(float(value) for value in requested),
        overlay_limited=breakdown.overlay_limited,
    )


def startup_handoff_fraction(*, elapsed_sec: float, rise_time_sec: float) -> float:
    """Return the second-phase blend from temporary joint lock to final control."""

    elapsed = _non_negative(elapsed_sec, name="elapsed_sec")
    rise_time = _positive(rise_time_sec, name="rise_time_sec")
    if elapsed <= rise_time:
        return 0.0
    return min(1.0, (elapsed - rise_time) / rise_time)


def build_startup_lock_profile(
    *,
    mit_kp_nm_per_rad: Sequence[float],
    mit_kd_nm_s_per_rad: Sequence[float],
    torque_limits_nm: Sequence[float],
    torque_rise_time_sec: float,
    velocity_limit_rad_s: float,
) -> B601MitControlProfile:
    """Build the temporary all-joint lock from the pinned YAML MIT gains.

    The gains are configuration inputs, not probe constants.  They are used as
    software gains so the same absolute and slew conditioners bound the startup
    lock as the final impedance command; the vendor MIT ``kp`` remains zero.
    """

    kp = _six(mit_kp_nm_per_rad, name="mit_kp_nm_per_rad")
    kd = _six(mit_kd_nm_s_per_rad, name="mit_kd_nm_s_per_rad")
    limits = _six(torque_limits_nm, name="torque_limits_nm")
    if any(value < 0.0 for value in kp) or any(value < 0.0 for value in kd):
        raise ValueError("startup MIT gains must be non-negative")
    if any(value <= 0.0 for value in limits):
        raise ValueError("torque_limits_nm must be positive")
    rise_time = _positive(torque_rise_time_sec, name="torque_rise_time_sec")
    velocity_limit = _positive(velocity_limit_rad_s, name="velocity_limit_rad_s")
    return B601MitControlProfile(
        name="b601_rs_startup_yaml_mit_position_lock",
        position_kp_nm_per_rad=kp,
        damping_kd_nm_s_per_rad=kd,
        max_abs_torque_nm=limits,
        max_torque_rate_nm_s=tuple(value / rise_time for value in limits),
        max_abs_overlay_torque_nm=_ZERO_SIX,
        max_abs_reference_velocity_rad_s=(velocity_limit,) * 6,
        max_abs_state_velocity_rad_s=(velocity_limit,) * 6,
        metadata={
            "gain_source": "rebotarm_rs_yaml.MIT",
            "torque_owner": "software_startup_lock",
            "vendor_kp": "zero",
            "vendor_kd": "optional_driver_damping_only",
        },
    )


def build_protective_hold_profile(
    startup_profile: B601MitControlProfile,
    *,
    driver_kd_nm_s_per_rad: Sequence[float],
    default_damping_kd_nm_s_per_rad: float = _DEFAULT_PROTECTIVE_HOLD_DAMPING_KD,
) -> B601MitControlProfile:
    """Build the draggable all-arm gravity/damping protective hold.

    The normal active controller may use either software damping or an
    explicitly selected RobStride driver ``Kd``.  Protective hold intentionally
    removes the startup position lock (software ``Kp=0``) and keeps one
    moderate damping owner while gravity is recomputed from each measured
    pose.  This prevents the first Ctrl-C state from fighting an operator who
    is trying to reposition the arm.

    For each joint, explicitly selected driver damping owns the derivative
    term; joints without driver damping use ``default_damping_kd`` in the
    host-composed torque.  The startup profile still supplies the absolute and
    slew bounds, so this helper introduces no additional safety threshold.
    """

    if not isinstance(startup_profile, B601MitControlProfile):
        raise TypeError("startup_profile must be a B601MitControlProfile")
    driver_kd = _six(
        driver_kd_nm_s_per_rad,
        name="driver_kd_nm_s_per_rad",
    )
    default_damping_kd = _non_negative(
        default_damping_kd_nm_s_per_rad,
        name="default_damping_kd_nm_s_per_rad",
    )
    software_kd = tuple(
        0.0 if driver_value > 0.0 else default_damping_kd
        for driver_value in driver_kd
    )
    metadata = dict(startup_profile.metadata)
    metadata.update(
        {
            "gain_source": "protective_hold_default_or_explicit_driver_kd",
            "torque_owner": "software_gravity_plus_single_damping_owner",
            "software_damping_disabled_where_driver_kd_nonzero": True,
            "driver_kd_nm_s_per_rad": driver_kd,
            "position_stiffness_disabled": True,
            "default_software_damping_kd_nm_s_per_rad": default_damping_kd,
            "reference_semantics": "current_feedback_pose_each_cycle",
        }
    )
    return B601MitControlProfile(
        name="b601_rs_protective_hold_gravity_damping",
        position_kp_nm_per_rad=_ZERO_SIX,
        damping_kd_nm_s_per_rad=software_kd,
        max_abs_torque_nm=startup_profile.max_abs_torque_nm,
        max_torque_rate_nm_s=startup_profile.max_torque_rate_nm_s,
        max_abs_overlay_torque_nm=startup_profile.max_abs_overlay_torque_nm,
        max_abs_reference_velocity_rad_s=startup_profile.max_abs_reference_velocity_rad_s,
        max_abs_state_velocity_rad_s=startup_profile.max_abs_state_velocity_rad_s,
        max_abs_state_acceleration_rad_s2=startup_profile.max_abs_state_acceleration_rad_s2,
        joint_names=startup_profile.joint_names,
        metadata=metadata,
    )


def blend_startup_lock_torque_breakdown(
    startup_lock: B601MitTorqueBreakdown,
    final_control: B601MitTorqueBreakdown,
    *,
    gravity_fraction: float,
    handoff_fraction: float,
) -> B601MitTorqueBreakdown:
    """Blend a bounded startup lock into the final six-joint controller.

    Gravity is always scaled as one complete vector.  Position, damping and
    task-space overlay terms move together from the temporary lock to the final
    controller, which prevents a discontinuity when the end-effector spring
    becomes the position-restoring owner.
    """

    if not isinstance(startup_lock, B601MitTorqueBreakdown):
        raise TypeError("startup_lock must be a B601MitTorqueBreakdown")
    if not isinstance(final_control, B601MitTorqueBreakdown):
        raise TypeError("final_control must be a B601MitTorqueBreakdown")
    gravity_scale = _bounded(
        gravity_fraction,
        name="gravity_fraction",
        lower=0.0,
        upper=1.0,
    )
    handoff = _bounded(
        handoff_fraction,
        name="handoff_fraction",
        lower=0.0,
        upper=1.0,
    )
    lock_weight = 1.0 - handoff
    lock_position = np.asarray(startup_lock.position_torque_nm, dtype=float)
    final_position = np.asarray(final_control.position_torque_nm, dtype=float)
    lock_damping = np.asarray(startup_lock.damping_torque_nm, dtype=float)
    final_damping = np.asarray(final_control.damping_torque_nm, dtype=float)
    gravity = gravity_scale * np.asarray(final_control.gravity_torque_nm, dtype=float)
    overlay_requested = handoff * np.asarray(
        final_control.overlay_requested_nm,
        dtype=float,
    )
    overlay_effective = handoff * np.asarray(
        final_control.overlay_effective_nm,
        dtype=float,
    )
    position = lock_weight * lock_position + handoff * final_position
    damping = lock_weight * lock_damping + handoff * final_damping
    requested = position + damping + gravity + overlay_effective
    return B601MitTorqueBreakdown(
        position_torque_nm=tuple(float(value) for value in position),
        damping_torque_nm=tuple(float(value) for value in damping),
        gravity_torque_nm=tuple(float(value) for value in gravity),
        overlay_requested_nm=tuple(float(value) for value in overlay_requested),
        overlay_effective_nm=tuple(float(value) for value in overlay_effective),
        requested_torque_nm=tuple(float(value) for value in requested),
        overlay_limited=final_control.overlay_limited,
    )


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

    kp = _non_negative(kp, name="kp")
    kd = _non_negative(kd, name="kd")
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
            position_gains[target_index] = _non_negative(
                target_kp, name="target_kp"
            )
        if target_kd is not None:
            damping_gains[target_index] = _non_negative(
                target_kd, name="target_kd"
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
            "torque_owner": "software_composed_tau_excludes_vendor_pd",
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

    scale = _non_negative(gravity_scale, name="gravity_scale")
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
    displacement_limit = _positive(
        max_displacement_m, name="cartesian_max_displacement_m"
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


def compute_pose_impedance_overlay(
    model: B601PinocchioModel,
    spring: B601PoseVirtualSpring,
    *,
    qpos_rad: Sequence[float],
    qvel_rad_s: Sequence[float],
    max_displacement_m: float,
    max_rotation_rad: float,
) -> tuple[tuple[float, ...], dict[str, object]]:
    """Evaluate one anchored 6D pose-impedance sample without joint motion."""

    qpos = _six(qpos_rad, name="qpos_rad")
    qvel = _six(qvel_rad_s, name="qvel_rad_s")
    displacement_limit = _positive(
        max_displacement_m, name="cartesian_max_displacement_m"
    )
    rotation_limit = _positive(
        max_rotation_rad, name="rotational_max_displacement_rad"
    )
    transform = np.asarray(model.forward_kinematics(qpos), dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise LowGainProbeError("Pose impedance FK returned an invalid transform")
    jacobian = np.asarray(model.arm_jacobian(qpos), dtype=float)
    if jacobian.shape != (6, 6) or not np.all(np.isfinite(jacobian)):
        raise LowGainProbeError("Pose impedance Jacobian is invalid")
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if singular_values.shape != (6,) or not np.all(np.isfinite(singular_values)):
        raise LowGainProbeError("Pose impedance singular values are invalid")
    spatial_velocity = jacobian @ np.asarray(qvel, dtype=float)
    spring_result = spring.evaluate(transform, spatial_velocity)
    displacement_norm = float(np.linalg.norm(spring_result.displacement_m))
    rotation_error_norm = float(np.linalg.norm(spring_result.rotation_error_rad))
    if displacement_norm > displacement_limit:
        raise LowGainProbeError(
            "Pose translation abort: "
            f"norm={displacement_norm:.6f} m > {displacement_limit:.6f} m"
        )
    if rotation_error_norm > rotation_limit:
        raise LowGainProbeError(
            "Pose rotation abort: "
            f"norm={rotation_error_norm:.6f} rad > {rotation_limit:.6f} rad"
        )
    mapping = map_world_aligned_ee_wrench_to_joint_torque(
        model,
        qpos,
        spring_result.wrench,
        qvel_rad_s=qvel,
    )
    if mapping.cartesian_power_w is None or mapping.joint_power_w is None:
        raise LowGainProbeError("Pose impedance power evidence is unavailable")
    power_error_w = abs(mapping.cartesian_power_w - mapping.joint_power_w)
    power_tolerance_w = 1e-9 + 1e-9 * max(
        abs(mapping.cartesian_power_w),
        abs(mapping.joint_power_w),
    )
    if power_error_w > power_tolerance_w:
        raise LowGainProbeError(
            "Pose wrench mapping violated power invariance: "
            f"error={power_error_w:.12g} W"
        )
    return mapping.joint_torque_nm, {
        "frame_id": spring_result.wrench.frame_id,
        "pose": tuple(tuple(float(value) for value in row) for row in transform),
        "spatial_velocity": tuple(float(value) for value in spatial_velocity),
        "displacement_m": spring_result.displacement_m,
        "displacement_norm_m": displacement_norm,
        "rotation_error_rad": spring_result.rotation_error_rad,
        "rotation_error_norm_rad": rotation_error_norm,
        "force_n": spring_result.wrench.force_n,
        "torque_nm": spring_result.wrench.torque_nm,
        "raw_force_n": spring_result.raw_force_n,
        "raw_torque_nm": spring_result.raw_torque_nm,
        "force_limited": spring_result.force_limited,
        "torque_limited": spring_result.torque_limited,
        "jacobian_singular_values": tuple(float(value) for value in singular_values),
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
    if mode in ("hold", "cartesian-hold", "pose-hold"):
        return tuple(float(value) for value in initial), _ZERO_SIX, True
    if mode != "slow-step":
        raise ValueError(
            "mode must be hold, slow-step, cartesian-hold, or pose-hold"
        )
    direction = 1 if int(step_direction) >= 0 else -1
    step = _bounded(step_rad, name="step_rad", lower=0.0, upper=_MAX_STEP_RAD)
    speed = _positive(step_speed_rad_s, name="step_speed_rad_s")
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
    minimum_ratio = _non_negative(min_response_ratio, name="min_response_ratio")
    maximum_ratio = _positive(max_response_ratio, name="max_response_ratio")
    if maximum_ratio < minimum_ratio:
        raise ValueError("max_response_ratio must be >= min_response_ratio")
    minimum_rad = _non_negative(min_response_rad, name="min_response_rad")
    support_limit = _non_negative(
        max_support_drift_rad,
        name="max_support_drift_rad",
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
    if output is None:
        print(encoded)
        return
    output_path = output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(encoded + "\n", encoding="utf-8")
    summary = {
        key: payload[key]
        for key in ("status", "mode", "acceptance_status", "stop_reason", "error")
        if key in payload
    }
    summary["result_path"] = str(output_path)
    if "runtime_log_path" in payload:
        summary["runtime_log_path"] = payload["runtime_log_path"]
    print(json.dumps(summary, sort_keys=True, default=list), flush=True)


def _operator_notice(message: str) -> None:
    """Write an infrequent safety notice to the log and controlling terminal."""

    print(message, flush=True)
    if sys.stdout.isatty():
        return
    try:
        with _OPERATOR_TTY_PATH.open("w", encoding="utf-8") as operator_stream:
            print(message, file=operator_stream, flush=True)
    except OSError:
        return


def _signal_handler(stop: dict[str, bool], signum: int, _frame: object) -> None:
    """Latch protective hold; a second SIGINT or SIGUSR1 requests full stop."""

    already_requested = stop.get("requested", False)
    stop["requested"] = True
    if getattr(signal, "SIGUSR1", None) == signum or (
        signum == signal.SIGINT and already_requested
    ):
        stop["full_disable_requested"] = True


def resolve_shutdown_action(
    *,
    enabled: bool,
    full_disable_requested: bool,
) -> str:
    """Choose the only permitted first action when leaving active control."""

    if enabled and not full_disable_requested:
        return "protective_hold"
    return "safe_stop"


def _drive_enable_countdown(
    stop: dict[str, bool],
    *,
    seconds: int = _DRIVE_ENABLE_COUNTDOWN_SECONDS,
    sleeper: Callable[[float], None] = time.sleep,
    log_stream: TextIO | None = None,
    operator_tty_path: Path | None = _OPERATOR_TTY_PATH,
) -> bool:
    """Log and foreground-mirror the final interruptible enable countdown."""

    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
        raise ValueError("drive enable countdown seconds must be a positive integer")
    output = sys.stdout if log_stream is None else log_stream
    operator_stream: TextIO | None = None
    try:
        if operator_tty_path is not None and not output.isatty():
            try:
                operator_stream = operator_tty_path.open("w", encoding="utf-8")
            except OSError:
                # A missing controlling terminal must not hide the audit log or
                # turn a valid foreground invocation into an implementation error.
                operator_stream = None

        def emit(message: str) -> None:
            print(message, file=output, flush=True)
            if operator_stream is not None:
                print(message, file=operator_stream, flush=True)

        for remaining in range(seconds, 0, -1):
            if stop.get("requested", False) or stop.get(
                "full_disable_requested", False
            ):
                emit("[DRIVE] CANCELLED")
                return False
            emit(f"[DRIVE] {remaining}")
            sleeper(1.0)
        if stop.get("requested", False) or stop.get("full_disable_requested", False):
            emit("[DRIVE] CANCELLED")
            return False
        emit("[DRIVE] ENABLE")
        return True
    finally:
        if operator_stream is not None:
            operator_stream.close()


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
    driver_kd_nm_s_per_rad: Sequence[float] = _ZERO_SIX,
) -> tuple[B601CommandReceipt, tuple[float, ...], tuple[float, ...]]:
    """Send one bounded all-arm gravity/damping MIT frame.

    ``qpos_rad`` is the current measured pose used as the next reference.  The
    protective-hold profile has zero software Kp, so this reference does not
    create a position lock; it keeps the complete MIT frame self-consistent
    while the measured pose changes under manual dragging.
    """

    period = 1.0 / _positive(rate_hz, name="rate_hz")
    reference = build_reference(
        sequence=0,
        timestamp_sec=time.monotonic(),
        qpos_rad=qpos_rad,
        qvel_rad_s=_ZERO_SIX,
    )
    breakdown = compute_software_torque_breakdown(
        model,
        reference=reference,
        measured_qpos_rad=measured_qpos_rad,
        measured_qvel_rad_s=measured_qvel_rad_s,
        profile=profile,
        gravity_scale=gravity_scale,
    )
    conditioned, _torque_limited, _slew_limited = condition_torque(
        breakdown.requested_torque_nm,
        previous_tau,
        absolute_limits_nm=torque_limits_nm,
        max_rate_nm_s=profile.max_torque_rate_nm_s,
        dt_sec=period,
    )
    receipt = backend.send_mit(
        reference,
        kp=_ZERO_SIX,
        kd=_six(driver_kd_nm_s_per_rad, name="driver_kd_nm_s_per_rad"),
        tau_nm=conditioned,
        torque_limits_nm=torque_limits_nm,
        joint_names=_FAULT_HOLD_JOINTS,
    )
    return receipt, conditioned, _six(
        breakdown.gravity_torque_nm,
        name="fault_hold_gravity_tau_nm",
    )


def _send_cached_fault_hold_command(
    backend: B601RSBackend,
    *,
    qpos_rad: tuple[float, ...],
    gravity_tau_nm: tuple[float, ...],
    previous_tau: tuple[float, ...],
    profile: B601MitControlProfile,
    torque_limits_nm: tuple[float, ...],
    rate_hz: float,
    driver_kd_nm_s_per_rad: Sequence[float] = _ZERO_SIX,
) -> tuple[B601CommandReceipt, tuple[float, ...], tuple[float, ...]]:
    """Send a model-free degraded hold frame from the last verified cache.

    The cached term is gravity only: position, Cartesian impedance, and stale
    software-velocity damping are deliberately excluded.  Motor-local damping
    remains active when configured.  The cached target still passes through
    the existing absolute and slew-rate conditioner before all six joints are
    dispatched.
    """

    period = 1.0 / _positive(rate_hz, name="rate_hz")
    cached_q = _six(qpos_rad, name="cached_fault_hold_qpos_rad")
    cached_gravity = _six(
        gravity_tau_nm,
        name="cached_fault_hold_gravity_tau_nm",
    )
    conditioned, _torque_limited, _slew_limited = condition_torque(
        cached_gravity,
        previous_tau,
        absolute_limits_nm=torque_limits_nm,
        max_rate_nm_s=profile.max_torque_rate_nm_s,
        dt_sec=period,
    )
    reference = build_reference(
        sequence=0,
        timestamp_sec=time.monotonic(),
        qpos_rad=cached_q,
        qvel_rad_s=_ZERO_SIX,
    )
    receipt = backend.send_mit(
        reference,
        kp=_ZERO_SIX,
        kd=_six(driver_kd_nm_s_per_rad, name="driver_kd_nm_s_per_rad"),
        tau_nm=conditioned,
        torque_limits_nm=torque_limits_nm,
        joint_names=_FAULT_HOLD_JOINTS,
    )
    return receipt, conditioned, cached_gravity


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
    driver_kd_nm_s_per_rad: Sequence[float] = _ZERO_SIX,
    hold_profile: B601MitControlProfile | None = None,
    operator_interrupt: bool = False,
    fault_entry: bool = False,
    fault_entry_max_velocity_rad_s: float | None = None,
    monitor_velocity_rad_s: float | None = None,
    monitor_acceleration_rad_s2: float | None = None,
    cached_qpos_rad: tuple[float, ...] | None = None,
    cached_gravity_tau_nm: tuple[float, ...] | None = None,
    degraded_entry_reason: str | None = None,
    degraded_telemetry: dict[str, object] | None = None,
) -> tuple[
    int,
    tuple[float, ...],
    B601CommandReceipt | None,
    B601ModeReceipt | None,
    B601ModeReceipt | None,
    list[str],
]:
    """Latch all six arm joints in a draggable gravity/damping hold.

    The first hold frame normally comes from the already validated
    ``last_safe`` snapshot; the interrupted loop's instantaneous velocity is
    not used to reject it.  If that candidate has expired while a complete
    current frame remains available, the current fault-entry gate is used
    instead.  For a normal control/position fault, a complete current fault
    snapshot is likewise used as the first reference, provided it is fresh, in
    limits, and below the configured motion-abort velocity.  The profile passed
    as ``hold_profile`` has software Kp=0 and one damping owner.  Each later
    feedback sample becomes the next reference, and gravity is recomputed from
    that pose, so an operator can drag the arm without a fixed-position fight.
    A configured velocity/acceleration monitor still catches untrusted motion.
    Receive, numeric, and model failures enter a model-free degraded hold using
    the last verified pose and gravity-only torque.  A complete-frame dispatch
    is retried once; full disable is reserved for an explicit request or when
    neither the live nor cached all-six-joint frame can be sent completely.
    The loop has no time-based release.

    max_iterations exists only for bounded offline tests; production calls
    leave it as None and therefore remain latched until release.
    """

    errors: list[str] = []
    latest_tau = previous_tau
    command_profile = profile if hold_profile is None else hold_profile
    monitor_velocity_limit = _positive(
        max_velocity_rad_s
        if monitor_velocity_rad_s is None
        else monitor_velocity_rad_s,
        name="protective_hold_velocity_limit_rad_s",
    )
    monitor_acceleration_limit = (
        None
        if monitor_acceleration_rad_s2 is None
        else _positive(
            monitor_acceleration_rad_s2,
            name="protective_hold_acceleration_limit_rad_s2",
        )
    )
    hold_commands = 0
    first_hold_receipt: B601CommandReceipt | None = None
    non_hold_disable_receipt: B601ModeReceipt | None = None
    full_disable_receipt: B601ModeReceipt | None = None
    telemetry = degraded_telemetry if degraded_telemetry is not None else {}
    telemetry.setdefault("entry_count", 0)
    telemetry.setdefault("recovery_count", 0)
    telemetry.setdefault("command_count", 0)
    telemetry.setdefault("reason_counts", {})
    telemetry.setdefault("last_reason", None)
    telemetry.setdefault("tx_retry_count", 0)
    telemetry.setdefault("tx_retry_recovered_count", 0)
    telemetry.setdefault("escalation_reason", None)
    driver_damping_available = any(
        value > 0.0
        for value in _six(
            driver_kd_nm_s_per_rad,
            name="driver_kd_nm_s_per_rad",
        )
    )
    telemetry["motor_local_damping_available"] = driver_damping_available
    degraded_active = False

    def record_degraded(category: str, detail: str) -> None:
        nonlocal degraded_active
        reason_counts = telemetry["reason_counts"]
        if not isinstance(reason_counts, dict):
            raise TypeError("degraded telemetry reason_counts must be a dict")
        reason_counts[category] = int(reason_counts.get(category, 0)) + 1
        telemetry["last_reason"] = detail
        if degraded_active:
            return
        degraded_active = True
        telemetry["entry_count"] = int(telemetry["entry_count"]) + 1
        if int(telemetry["entry_count"]) > 1:
            print(f"[B601] protective hold degraded again: {detail}", flush=True)
            return
        damping_note = (
            "motor-local damping remains active"
            if driver_damping_available
            else "no motor-local damping is configured"
        )
        _operator_notice(
            "[B601] PROTECTIVE HOLD DEGRADED: using last verified gravity; "
            f"{damping_note} while live feedback/control recovers."
        )

    def record_recovery() -> None:
        nonlocal degraded_active
        if not degraded_active:
            return
        degraded_active = False
        telemetry["recovery_count"] = int(telemetry["recovery_count"]) + 1
        if int(telemetry["recovery_count"]) == 1:
            _operator_notice(
                "[B601] PROTECTIVE HOLD RECOVERED: live gravity update resumed."
            )
        else:
            print("[B601] protective hold live update recovered again", flush=True)

    def attempt_complete_frame(
        sender: Callable[
            [],
            tuple[B601CommandReceipt, tuple[float, ...], tuple[float, ...]],
        ],
        *,
        context: str,
    ) -> tuple[
        tuple[B601CommandReceipt, tuple[float, ...], tuple[float, ...]] | None,
        tuple[str, ...],
    ]:
        """Try one complete frame and one immediate same-frame retry."""

        failures: list[str] = []
        for attempt in range(2):
            try:
                result = sender()
            except Exception as exc:  # noqa: BLE001 - retry then preserve evidence
                failures.append(f"{context}: {exc}")
            else:
                receipt = result[0]
                if receipt.accepted:
                    if attempt:
                        telemetry["tx_retry_recovered_count"] = (
                            int(telemetry["tx_retry_recovered_count"]) + 1
                        )
                    return result, tuple(failures)
                failures.extend(
                    receipt.errors or (f"{context}: incomplete six-joint dispatch",)
                )
            if attempt == 0:
                telemetry["tx_retry_count"] = int(telemetry["tx_retry_count"]) + 1
        return None, tuple(failures)

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

    cached_q: tuple[float, ...] | None = None
    cached_gravity: tuple[float, ...] | None = (
        None
        if cached_gravity_tau_nm is None
        else _six(cached_gravity_tau_nm, name="cached_gravity_tau_nm")
    )

    def attempt_cached_frame(
        context: str,
    ) -> tuple[
        tuple[B601CommandReceipt, tuple[float, ...], tuple[float, ...]] | None,
        tuple[str, ...],
    ]:
        if cached_q is None or cached_gravity is None:
            return None, (f"{context}: no verified pose/gravity cache",)
        return attempt_complete_frame(
            lambda: _send_cached_fault_hold_command(
                backend,
                qpos_rad=cached_q,
                gravity_tau_nm=cached_gravity,
                previous_tau=latest_tau,
                profile=command_profile,
                torque_limits_nm=torque_limits_nm,
                rate_hz=rate_hz,
                driver_kd_nm_s_per_rad=driver_kd_nm_s_per_rad,
            ),
            context=context,
        )

    try:
        cache_source_q = cached_qpos_rad if cached_qpos_rad is not None else qpos_rad
        if cache_source_q is not None:
            cached_q = _validate_in_limit_qpos(
                cache_source_q,
                position_limits_rad=position_limits_rad,
                name="cached_fault_hold_qpos_rad",
            )
        if degraded_entry_reason is not None:
            if cached_q is None or cached_gravity is None:
                raise LowGainProbeError(
                    "degraded fault hold requires a verified pose and gravity cache"
                )
            record_degraded("entry_without_live_feedback", degraded_entry_reason)
            first_result, first_failures = attempt_cached_frame(
                "degraded fault hold entry"
            )
            if first_result is None:
                errors.extend(first_failures)
                telemetry["escalation_reason"] = (
                    "cached all-six-joint hold dispatch failed after retry"
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
            first_hold_receipt, latest_tau, cached_gravity = first_result
            hold_commands = 1
            telemetry["command_count"] = int(telemetry["command_count"]) + 1
            measured_qpos = cached_q
            measured_qvel = _ZERO_SIX
            hold_q = cached_q
        elif fault_entry:
            if fault_qpos_rad is None or fault_qvel_rad_s is None:
                raise LowGainProbeError(
                    "fault hold requires a complete current fault snapshot"
                )
            entry_velocity_limit = (
                max_velocity_rad_s
                if fault_entry_max_velocity_rad_s is None
                else fault_entry_max_velocity_rad_s
            )
            hold_q = _validate_fault_entry_snapshot(
                fault_qpos_rad,
                qvel_rad_s=fault_qvel_rad_s,
                observed_age_sec=observed_age_sec,
                position_limits_rad=position_limits_rad,
                max_age_sec=max_age_sec,
                max_velocity_rad_s=entry_velocity_limit,
            )
            measured_qpos = hold_q
            measured_qvel = _six(
                fault_qvel_rad_s,
                name="fault_measured_qvel_rad_s",
            )
        else:
            hold_q = _validate_fault_hold_snapshot(
                qpos_rad,
                qvel_rad_s=qvel_rad_s,
                observed_age_sec=observed_age_sec,
                position_limits_rad=position_limits_rad,
                max_age_sec=max_age_sec,
                max_velocity_rad_s=max_velocity_rad_s,
                fault_qpos_rad=None if operator_interrupt else fault_qpos_rad,
                fault_qvel_rad_s=None if operator_interrupt else fault_qvel_rad_s,
                max_rebase_rad=max_rebase_rad,
            )
            measured_qpos = hold_q
            if fault_qpos_rad is not None and not operator_interrupt:
                measured_qpos, _ = normalize_feedback_boundary_qpos(
                    fault_qpos_rad,
                    name="fault_measured_qpos_rad",
                )
            measured_qvel = (
                fault_qvel_rad_s
                if fault_qvel_rad_s is not None and not operator_interrupt
                else qvel_rad_s
            )
        if degraded_entry_reason is None:
            first_result, first_failures = attempt_complete_frame(
                lambda: _send_fault_hold_command(
                    backend,
                    model,
                    qpos_rad=hold_q,
                    measured_qpos_rad=_six(
                        measured_qpos,
                        name="fault_measured_qpos_rad",
                    ),
                    measured_qvel_rad_s=_six(
                        measured_qvel,
                        name="fault_measured_qvel_rad_s",
                    ),
                    profile=command_profile,
                    gravity_scale=gravity_scale,
                    previous_tau=latest_tau,
                    torque_limits_nm=torque_limits_nm,
                    rate_hz=rate_hz,
                    driver_kd_nm_s_per_rad=driver_kd_nm_s_per_rad,
                ),
                context="fault hold entry",
            )
            if first_result is None:
                if cached_q is None or cached_gravity is None:
                    errors.extend(first_failures)
                    telemetry["escalation_reason"] = (
                        "no verified gravity cache after hold entry failure"
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
                record_degraded("entry_control_failure", "; ".join(first_failures))
                first_result, cached_failures = attempt_cached_frame(
                    "cached fault hold entry"
                )
                if first_result is None:
                    errors.extend(first_failures)
                    errors.extend(cached_failures)
                    telemetry["escalation_reason"] = (
                        "live and cached all-six-joint hold dispatch failed after retry"
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
            first_hold_receipt, latest_tau, cached_gravity = first_result
            cached_q = hold_q
            hold_commands = 1
            telemetry["command_count"] = int(telemetry["command_count"]) + 1
    except Exception as exc:  # noqa: BLE001 - preserve safety evidence
        detail = f"fault hold entry: {exc}"
        record_degraded("entry_validation_or_control_failure", detail)
        first_result, cached_failures = attempt_cached_frame(
            "cached fault hold after entry failure"
        )
        if first_result is None:
            errors.append(detail)
            errors.extend(cached_failures)
            telemetry["escalation_reason"] = (
                "entry failed and cached six-joint hold could not be dispatched"
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
        first_hold_receipt, latest_tau, cached_gravity = first_result
        measured_qpos = cached_q
        measured_qvel = _ZERO_SIX
        hold_q = cached_q
        hold_commands = 1
        telemetry["command_count"] = int(telemetry["command_count"]) + 1

    # The protective hold owns the complete arm set.  There is intentionally
    # no partial ``safe_stop`` here: disabling joints 1/4/5/6 would turn a
    # draggable gravity hold into an asymmetric, unstable load path.

    _operator_notice(
        "B601 protective hold latched: joint1..joint6 remain enabled with "
        "gravity compensation and damping; joint7 is not part of this backend. "
        f"pid={os.getpid()}; press Ctrl-C again or send SIGUSR1 for full disable."
    )
    period = 1.0 / _positive(rate_hz, name="rate_hz")
    monitor_q = _six(measured_qpos, name="fault_measured_qpos_rad")
    monitor_observed_at = time.monotonic() - max(0.0, observed_age_sec)
    monitor_baseline_pending = operator_interrupt or fault_entry
    monitor_previous_qvel = _six(
        measured_qvel,
        name="fault_measured_qvel_rad_s",
    )
    while not stop.get("full_disable_requested", False):
        if max_iterations is not None and hold_commands >= max_iterations:
            break
        cycle_error: Exception | None = None
        cycle_result: tuple[
            B601CommandReceipt,
            tuple[float, ...],
            tuple[float, ...],
        ] | None = None
        cycle_failures: tuple[str, ...] = ()
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
            current_q, _ = normalize_feedback_boundary_qpos(
                state.positions_rad,
                name="protective_hold_feedback_qpos_rad",
            )
            if monitor_baseline_pending:
                # The first post-interrupt sample spans the transition from
                # the active loop to protective hold. It is a measured-state
                # baseline only; deriving velocity over that transition would
                # turn an otherwise valid first hold frame into an immediate
                # fallback. The next sample is monitored with the configured
                # protective-hold motion limits.
                measured_q = _validate_protective_hold_feedback(
                    current_q,
                    qvel_rad_s=_ZERO_SIX,
                    hold_qpos_rad=hold_q,
                    observed_age_sec=0.0,
                    position_limits_rad=position_limits_rad,
                    max_age_sec=max_age_sec,
                    max_velocity_rad_s=monitor_velocity_limit,
                    max_rebase_rad=max_rebase_rad,
                )
                monitor_q = measured_q
                monitor_observed_at = observed_at
                monitor_baseline_pending = False
                monitor_qvel = _ZERO_SIX
            else:
                monitor_dt = observed_at - monitor_observed_at
                if not math.isfinite(monitor_dt) or monitor_dt <= 0.0:
                    raise LowGainProbeError(
                        "protective hold feedback timestamp did not advance"
                    )
                monitor_qvel = tuple(
                    (current - previous) / monitor_dt
                    for current, previous in zip(
                        current_q,
                        monitor_q,
                        strict=True,
                    )
                )
                if monitor_acceleration_limit is not None:
                    monitor_qacc = tuple(
                        (current - previous) / monitor_dt
                        for current, previous in zip(
                            monitor_qvel,
                            monitor_previous_qvel,
                            strict=True,
                        )
                    )
                    if any(not math.isfinite(value) for value in monitor_qacc):
                        raise LowGainProbeError(
                            "protective hold finite-difference acceleration is not finite"
                        )
                    observed_max_acceleration = max(abs(value) for value in monitor_qacc)
                    if observed_max_acceleration > monitor_acceleration_limit:
                        raise LowGainProbeError(
                            "protective hold acceleration abort: "
                            f"max_abs_acceleration={observed_max_acceleration:.6f} "
                            f"rad/s^2 > {monitor_acceleration_limit:.6f} rad/s^2"
                        )
                measured_q = _validate_protective_hold_feedback(
                    current_q,
                    qvel_rad_s=monitor_qvel,
                    hold_qpos_rad=hold_q,
                    observed_age_sec=0.0,
                    position_limits_rad=position_limits_rad,
                    max_age_sec=max_age_sec,
                    max_velocity_rad_s=monitor_velocity_limit,
                    max_rebase_rad=max_rebase_rad,
                )
            if stop.get("full_disable_requested", False):
                break
            cycle_result, cycle_failures = attempt_complete_frame(
                lambda measured_q=measured_q,
                monitor_qvel=monitor_qvel,
                previous_tau=latest_tau: _send_fault_hold_command(
                    backend,
                    model,
                    qpos_rad=measured_q,
                    measured_qpos_rad=measured_q,
                    measured_qvel_rad_s=monitor_qvel,
                    profile=command_profile,
                    gravity_scale=gravity_scale,
                    previous_tau=previous_tau,
                    torque_limits_nm=torque_limits_nm,
                    rate_hz=rate_hz,
                    driver_kd_nm_s_per_rad=driver_kd_nm_s_per_rad,
                ),
                context="live protective hold",
            )
        except Exception as exc:  # noqa: BLE001 - preserve safety evidence
            cycle_error = exc
        if cycle_result is not None:
            _receipt, latest_tau, cached_gravity = cycle_result
            cached_q = measured_q
            hold_commands += 1
            monitor_q = measured_q
            monitor_previous_qvel = _six(
                monitor_qvel,
                name="protective_hold_monitor_qvel_rad_s",
            )
            monitor_observed_at = observed_at
            telemetry["command_count"] = int(telemetry["command_count"]) + 1
            record_recovery()
        else:
            detail = (
                f"fault hold: {cycle_error}"
                if cycle_error is not None
                else "; ".join(cycle_failures)
            )
            category = (
                "feedback_or_control_failure"
                if cycle_error is not None
                else "live_dispatch_failure"
            )
            record_degraded(category, detail)
            monitor_baseline_pending = True
            if cached_q is None or cached_gravity is None:
                errors.append(detail)
                telemetry["escalation_reason"] = "no verified degraded hold cache"
                full_disable_fallback()
                break
            cached_result, cached_failures = attempt_cached_frame(
                "degraded protective hold"
            )
            if cached_result is None:
                errors.append(detail)
                errors.extend(cached_failures)
                telemetry["escalation_reason"] = (
                    "live and cached all-six-joint hold dispatch failed after retry"
                )
                full_disable_fallback()
                break
            _receipt, latest_tau, cached_gravity = cached_result
            hold_commands += 1
            telemetry["command_count"] = int(telemetry["command_count"]) + 1
        if max_iterations is not None and hold_commands >= max_iterations:
            break
        time.sleep(period)

    if stop.get("full_disable_requested", False):
        telemetry["escalation_reason"] = "explicit second stop request"
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
    driver_kd_nm_s_per_rad = (
        _non_negative(getattr(args, "driver_kd", 0.0), name="driver_kd"),
    ) * 6
    config_path = Path(args.config).expanduser().resolve(strict=True)
    source_root = Path(args.rebot_source_root).expanduser().resolve(strict=True)
    model, position_limits = _resolve_model_and_limits(source_root)
    feedback_position_limits = build_feedback_admission_position_limits(
        position_limits
    )
    if args.mode == "gravity-calibration-fit":
        return _run_gravity_calibration_fit(
            args=args,
            model=model,
            source_root=source_root,
        )
    if args.mode == "feedback-stability":
        config = B601RSBackendConfig.from_rs_yaml(
            config_path,
            allow_motion=False,
            position_limits_rad=position_limits,
            feedback_position_limits_rad=feedback_position_limits,
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
    if args.mode == "driver-torque-limit-roundtrip":
        config = B601RSBackendConfig.from_rs_yaml(
            config_path,
            allow_motion=True,
            position_limits_rad=position_limits,
            feedback_position_limits_rad=feedback_position_limits,
        )
        return _run_driver_torque_limit_roundtrip_probe(
            B601RSBackend(config),
            config_path=config_path,
            source_root=source_root,
            model=model,
            position_limits=position_limits,
            torque_limits_nm=torque_limits,
        )
    overlay_limits = (
        tuple(
            value * args.cartesian_overlay_limit_scale
            for value in torque_limits
        )
        if args.mode in ("cartesian-hold", "pose-hold")
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
        feedback_position_limits_rad=feedback_position_limits,
    )
    startup_lock_profile = build_startup_lock_profile(
        mit_kp_nm_per_rad=tuple(joint.mit_kp for joint in config.joints),
        mit_kd_nm_s_per_rad=tuple(joint.mit_kd for joint in config.joints),
        torque_limits_nm=torque_limits,
        torque_rise_time_sec=args.torque_rise_time_sec,
        velocity_limit_rad_s=args.velocity_abort_rad_s,
    )
    protective_hold_profile = build_protective_hold_profile(
        startup_lock_profile,
        driver_kd_nm_s_per_rad=driver_kd_nm_s_per_rad,
        default_damping_kd_nm_s_per_rad=getattr(
            args,
            "protective_hold_kd",
            _DEFAULT_PROTECTIVE_HOLD_DAMPING_KD,
        ),
    )
    backend = B601RSBackend(config)
    ledger = FrequencyLedger()
    active_joint_names = resolve_active_joint_names()
    support_joint_names = tuple(
        name for name in active_joint_names if name != args.joint
    )
    stop = {"requested": False, "full_disable_requested": False}
    previous_q: tuple[float, ...] | None = None
    previous_feedback_time: float | None = None
    previous_guard_qvel: tuple[float, ...] | None = None
    previous_tau = _ZERO_SIX
    last_verified_gravity_tau: tuple[float, ...] | None = None
    initial_q: tuple[float, ...] | None = None
    final_q: tuple[float, ...] | None = None
    cartesian_anchor_position_m: tuple[float, ...] | None = None
    cartesian_spring: B601TranslationalVirtualSpring | None = None
    pose_anchor: tuple[tuple[float, ...], ...] | None = None
    pose_spring: B601PoseVirtualSpring | None = None
    fault_qpos_for_hold: tuple[float, ...] | None = None
    fault_qvel_for_hold: tuple[float, ...] | None = None
    fault_observed_at_sec: float | None = None
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
    rejected_motion_sample: dict[str, object] | None = None
    rejected_motion_sample_count = 0
    backend_feedback_diagnostics: tuple[dict[str, object], ...] = ()
    samples: list[dict[str, object]] = []
    stop_reason: str | None = None
    control_completion_reason: str | None = None
    primary_error: Exception | None = None
    preflight_disable: B601ModeReceipt | None = None
    mode_receipt: B601ModeReceipt | None = None
    enable_receipt: B601ModeReceipt | None = None
    cleanup_receipt: B601ModeReceipt | None = None
    driver_limit_apply_receipt: B601DriverTorqueLimitReceipt | None = None
    driver_limit_restore_receipt: B601DriverTorqueLimitReceipt | None = None
    cleanup_errors: list[str] = []
    protective_hold_latched = False
    protective_hold_commands = 0
    protective_hold_admission_age_sec: float | None = None
    protective_hold_first_receipt: B601CommandReceipt | None = None
    protective_hold_non_hold_receipt: B601ModeReceipt | None = None
    protective_hold_full_disable_receipt: B601ModeReceipt | None = None
    protective_hold_reference_source: str | None = None
    protective_hold_entry_reason: str | None = None
    protective_hold_degraded: dict[str, object] = {
        "entry_count": 0,
        "recovery_count": 0,
        "command_count": 0,
        "reason_counts": {},
        "last_reason": None,
        "tx_retry_count": 0,
        "tx_retry_recovered_count": 0,
        "escalation_reason": None,
    }
    startup_lock_first_receipt: B601CommandReceipt | None = None
    startup_lock_commands = 0
    arming_settle_torque_commands = 0
    startup_state_history: list[str] = []
    arming_settle_required_samples = _DEFAULT_ARMING_SETTLE_SAMPLES
    arming_settle_samples = 0
    arming_settle_passed = False
    arming_settle_elapsed_sec: float | None = None
    # Startup settle admits the bounded enable transient under the normal
    # active-path motion watchdog.  The stricter fault-hold value remains only
    # the first-Ctrl-C/last-safe stationary admission gate.
    arming_settle_velocity_limit_rad_s = args.velocity_abort_rad_s
    arming_settle_last_qvel: tuple[float, ...] | None = None
    calibration_capture = (
        GravityCalibrationCapture(
            args.calibration_duration_sec,
            settling_duration_sec=args.calibration_settling_sec,
        )
        if args.mode == "gravity-calibration"
        else None
    )
    calibration_keys: OperatorKeyReader | None = None
    calibration_ready = False
    last_control_elapsed_sec = 0.0
    connected = False
    enabled = False
    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigusr1 = signal.getsignal(signal.SIGUSR1)

    def set_startup_state(state: str) -> None:
        if startup_state_history and startup_state_history[-1] == state:
            return
        startup_state_history.append(state)
        _operator_notice(f"[B601] STARTUP {state}")

    signal.signal(
        signal.SIGINT,
        lambda signum, frame: _signal_handler(stop, signum, frame),
    )
    signal.signal(
        signal.SIGUSR1,
        lambda signum, frame: _signal_handler(stop, signum, frame),
    )
    try:
        if calibration_capture is not None:
            # Establish the interactive input path before opening the backend or
            # changing any drive state.
            calibration_keys = OperatorKeyReader()
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
        if getattr(args, "apply_driver_torque_limits", False):
            driver_limit_apply_receipt = (
                backend.apply_temporary_driver_torque_limits(
                    torque_limits,
                    joint_names=active_joint_names,
                )
            )
            if not driver_limit_apply_receipt.completed:
                detail = "; ".join(driver_limit_apply_receipt.errors)
                raise LowGainProbeError(
                    "temporary driver torque-limit apply failed"
                    + (f": {detail}" if detail else "")
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
        if not _drive_enable_countdown(stop):
            if stop["full_disable_requested"]:
                raise LowGainProbeError(
                    "explicit full disable requested during drive countdown"
                )
            raise LowGainProbeError("operator interrupt during drive countdown")
        enable_receipt = backend.enable(joint_names=active_joint_names)
        if not enable_receipt.completed:
            raise LowGainProbeError(
                "motor enable failed: " + "; ".join(enable_receipt.errors)
            )
        enabled = True
        set_startup_state("ENABLED_UNARMED")
        if stop["full_disable_requested"]:
            raise LowGainProbeError(
                "explicit full disable requested immediately after motor enable"
            )
        if stop["requested"]:
            raise LowGainProbeError(
                "operator interrupt immediately after motor enable"
            )
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
        arming_q = _validate_in_limit_qpos(
            arming_state.positions_rad,
            position_limits_rad=position_limits,
            name="arming_qpos_rad",
        )
        arming_qvel = tuple(
            (current - previous) / arm_dt
            for current, previous in zip(arming_q, warmup_q, strict=True)
        )
        if max(abs(value) for value in arming_qvel) > args.velocity_abort_rad_s:
            raise LowGainProbeError(
                "post-enable feedback is moving too quickly for startup lock: "
                f"max_abs_velocity={max(abs(value) for value in arming_qvel):.6f} "
                f"> {args.velocity_abort_rad_s:.6f} rad/s"
            )
        initial_q = arming_q
        final_q = arming_q
        last_safe_q = arming_q
        last_safe_qvel = tuple(float(value) for value in arming_qvel)
        last_safe_observed_at_sec = arming_observed_at_sec
        previous_q = arming_q
        previous_feedback_time = arming_state.receive_timestamp_sec
        set_startup_state("LOCKING")
        lock_reference = build_reference(
            sequence=arming_state.sequence,
            timestamp_sec=arming_state.receive_timestamp_sec,
            qpos_rad=initial_q,
            qvel_rad_s=_ZERO_SIX,
        )
        lock_breakdown = compute_software_torque_breakdown(
            model,
            reference=lock_reference,
            measured_qpos_rad=arming_q,
            measured_qvel_rad_s=arming_qvel,
            profile=startup_lock_profile,
            gravity_scale=args.gravity_scale,
        )
        lock_tau, _lock_torque_limited, _lock_slew_limited = condition_torque(
            lock_breakdown.requested_torque_nm,
            _ZERO_SIX,
            absolute_limits_nm=torque_limits,
            max_rate_nm_s=startup_lock_profile.max_torque_rate_nm_s,
            dt_sec=arm_dt,
        )
        startup_lock_first_receipt = backend.send_mit(
            lock_reference,
            kp=_ZERO_SIX,
            kd=driver_kd_nm_s_per_rad,
            tau_nm=lock_tau,
            torque_limits_nm=torque_limits,
            joint_names=active_joint_names,
        )
        if not startup_lock_first_receipt.accepted:
            raise LowGainProbeError(
                "; ".join(startup_lock_first_receipt.errors)
                or "startup lock MIT dispatch was incomplete"
            )
        startup_lock_commands = 1
        arming_settle_torque_commands = 1
        previous_tau = lock_tau
        last_verified_gravity_tau = _six(
            lock_breakdown.gravity_torque_nm,
            name="startup_lock_gravity_tau_nm",
        )
        set_startup_state("LOCKED")
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
            previous_settle_feedback_time = arming_settle_previous_feedback_time
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
            settle_dt = (
                settle_state.receive_timestamp_sec - previous_settle_feedback_time
            )
            lock_reference = build_reference(
                sequence=settle_state.sequence,
                timestamp_sec=settle_state.receive_timestamp_sec,
                qpos_rad=initial_q,
                qvel_rad_s=_ZERO_SIX,
            )
            lock_breakdown = compute_software_torque_breakdown(
                model,
                reference=lock_reference,
                measured_qpos_rad=arming_settle_previous_q,
                measured_qvel_rad_s=arming_settle_last_qvel,
                profile=startup_lock_profile,
                gravity_scale=args.gravity_scale,
            )
            lock_tau, _lock_torque_limited, _lock_slew_limited = condition_torque(
                lock_breakdown.requested_torque_nm,
                previous_tau,
                absolute_limits_nm=torque_limits,
                max_rate_nm_s=startup_lock_profile.max_torque_rate_nm_s,
                dt_sec=settle_dt,
            )
            lock_receipt = backend.send_mit(
                lock_reference,
                kp=_ZERO_SIX,
                kd=driver_kd_nm_s_per_rad,
                tau_nm=lock_tau,
                torque_limits_nm=torque_limits,
                joint_names=active_joint_names,
            )
            if not lock_receipt.accepted:
                stop_reason = "; ".join(lock_receipt.errors) or (
                    "arming settle startup lock dispatch was incomplete"
                )
                break
            startup_lock_commands += 1
            arming_settle_torque_commands += 1
            previous_tau = lock_tau
            last_verified_gravity_tau = _six(
                lock_breakdown.gravity_torque_nm,
                name="arming_settle_gravity_tau_nm",
            )
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
            arming_q = initial_q
            arming_qvel = arming_settle_last_qvel
            final_q = initial_q
            last_safe_q = initial_q
            last_safe_qvel = arming_settle_last_qvel
            last_safe_observed_at_sec = arming_settle_last_observed_at_sec
            previous_q = arming_settle_previous_q
            previous_feedback_time = arming_settle_previous_feedback_time
        if stop_reason is None and args.mode in ("cartesian-hold", "pose-hold"):
            arming_transform = np.asarray(
                model.forward_kinematics(initial_q),
                dtype=float,
            )
            if (
                arming_transform.shape != (4, 4)
                or not np.all(np.isfinite(arming_transform))
            ):
                raise LowGainProbeError(
                    "post-enable Cartesian anchor FK returned an invalid transform"
                )
            if args.mode == "cartesian-hold":
                cartesian_anchor_position_m = tuple(
                    float(value) for value in arming_transform[:3, 3]
                )
                cartesian_spring = B601TranslationalVirtualSpring(
                    anchor_position_m=cartesian_anchor_position_m,
                    stiffness_n_per_m=(args.cartesian_stiffness_n_per_m,) * 3,
                    damping_n_s_per_m=(args.cartesian_damping_n_s_per_m,) * 3,
                    max_abs_force_n=(args.cartesian_max_force_n,) * 3,
                )
            else:
                pose_anchor = tuple(
                    tuple(float(value) for value in row)
                    for row in arming_transform
                )
                pose_spring = B601PoseVirtualSpring(
                    anchor_pose=pose_anchor,
                    translational_stiffness_n_per_m=(
                        args.cartesian_stiffness_n_per_m,
                    )
                    * 3,
                    translational_damping_n_s_per_m=(
                        args.cartesian_damping_n_s_per_m,
                    )
                    * 3,
                    rotational_stiffness_nm_per_rad=(
                        args.rotational_stiffness_nm_per_rad,
                    )
                    * 3,
                    rotational_damping_nm_s_per_rad=(
                        args.rotational_damping_nm_s_per_rad,
                    )
                    * 3,
                    max_abs_force_n=(args.cartesian_max_force_n,) * 3,
                    max_abs_torque_nm=(args.rotational_max_torque_nm,) * 3,
                )
        period = 1.0 / args.rate_hz
        started = time.monotonic()
        next_deadline = started
        while stop_reason is None and time.monotonic() - started < args.duration_sec:
            fault_qpos_for_hold = None
            fault_qvel_for_hold = None
            fault_observed_at_sec = None
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
                backend_feedback_diagnostics = _backend_feedback_diagnostics(state)
                rejected_motion_sample = {
                    "reason": stop_reason,
                    "guard": "backend_feedback_admission",
                    "sequence": int(state.sequence),
                    "feedback_timestamp_sec": state.receive_timestamp_sec,
                    "feedback_dt_sec": None,
                    "raw_qpos_rad": tuple(
                        joint.position_rad for joint in state.joints
                    ),
                    "normalized_qpos_rad": None,
                    "previous_qvel_rad_s": previous_guard_qvel,
                    "current_qvel_rad_s": None,
                    "qacc_fd_rad_s2": None,
                    "backend_fault_reason": state.fault_reason,
                    "backend_joint_diagnostics": backend_feedback_diagnostics,
                    "command_dispatched": False,
                    "eligible_for_last_safe": False,
                }
                rejected_motion_sample_count += 1
                break
            if getattr(state, "stale", False):
                stop_reason = "stale feedback snapshot"
                break
            try:
                raw_q = _six(state.positions_rad, name="feedback_qpos_rad")
                q, normalized_joint_names = normalize_feedback_boundary_qpos(
                    raw_q,
                    name="feedback_qpos_rad",
                )
            except (TypeError, ValueError) as exc:
                stop_reason = f"invalid feedback position: {exc}"
                rejected_motion_sample = {
                    "reason": stop_reason,
                    "guard": "position_sample",
                    "sequence": int(state.sequence),
                    "feedback_timestamp_sec": state.receive_timestamp_sec,
                    "feedback_dt_sec": None,
                    "raw_qpos_observed": repr(state.positions_rad),
                    "normalized_qpos_rad": None,
                    "previous_qvel_rad_s": previous_guard_qvel,
                    "current_qvel_rad_s": None,
                    "qacc_fd_rad_s2": None,
                    "command_dispatched": False,
                    "eligible_for_last_safe": False,
                }
                rejected_motion_sample_count += 1
                break
            position_rejection = evaluate_feedback_position_guard(
                sequence=state.sequence,
                feedback_timestamp_sec=state.receive_timestamp_sec,
                raw_qpos_rad=raw_q,
                normalized_qpos_rad=q,
                position_limits_rad=position_limits,
                previous_qvel_rad_s=previous_guard_qvel,
            )
            if position_rejection is not None:
                rejected_motion_sample = position_rejection
                rejected_motion_sample_count += 1
                stop_reason = str(position_rejection["reason"])
                break
            # Only an in-limit, finite sample may become the active state or a
            # candidate protective-hold snapshot.  An invalid finite value is
            # evidence only; it must never contaminate the final pose or hold
            # admission path.
            final_q = q
            fault_qpos_for_hold = q
            fault_observed_at_sec = observed_at_sec
            if previous_q is None:
                qvel = _ZERO_SIX
                dt = period
            else:
                if previous_feedback_time is None:
                    raise LowGainProbeError("missing previous feedback timestamp")
                dt = state.receive_timestamp_sec - previous_feedback_time
                if not math.isfinite(dt) or dt <= 0.0:
                    stop_reason = "non-increasing feedback timestamp"
                    rejected_motion_sample = {
                        "reason": stop_reason,
                        "guard": "feedback_dt",
                        "sequence": int(state.sequence),
                        "feedback_timestamp_sec": state.receive_timestamp_sec,
                        "feedback_dt_sec": dt if math.isfinite(dt) else None,
                        "feedback_dt_observed": repr(dt),
                        "raw_qpos_rad": raw_q,
                        "normalized_qpos_rad": q,
                        "previous_qvel_rad_s": previous_guard_qvel,
                        "current_qvel_rad_s": None,
                        "qacc_fd_rad_s2": None,
                        "command_dispatched": False,
                        "eligible_for_last_safe": False,
                    }
                    rejected_motion_sample_count += 1
                    break
                qvel = tuple(
                    (current - previous) / dt
                    for current, previous in zip(q, previous_q, strict=True)
                )
            fault_qvel_for_hold = tuple(float(value) for value in qvel)
            if any(not math.isfinite(value) for value in qvel):
                stop_reason = "non-finite finite-difference velocity"
                rejected_motion_sample = {
                    "reason": stop_reason,
                    "guard": "sample_validation",
                    "sequence": int(state.sequence),
                    "feedback_timestamp_sec": state.receive_timestamp_sec,
                    "feedback_dt_sec": dt,
                    "raw_qpos_rad": raw_q,
                    "normalized_qpos_rad": q,
                    "previous_qvel_rad_s": previous_guard_qvel,
                    "current_qvel_rad_s": tuple(
                        value if math.isfinite(value) else None for value in qvel
                    ),
                    "current_qvel_observed": tuple(repr(value) for value in qvel),
                    "qacc_fd_rad_s2": None,
                    "command_dispatched": False,
                    "eligible_for_last_safe": False,
                }
                rejected_motion_sample_count += 1
                break
            try:
                qacc, motion_rejection = evaluate_feedback_motion_guard(
                    sequence=state.sequence,
                    feedback_timestamp_sec=state.receive_timestamp_sec,
                    dt_sec=dt,
                    raw_qpos_rad=raw_q,
                    normalized_qpos_rad=q,
                    previous_qvel_rad_s=previous_guard_qvel,
                    current_qvel_rad_s=qvel,
                    velocity_abort_rad_s=args.velocity_abort_rad_s,
                    acceleration_abort_rad_s2=args.acceleration_abort_rad_s2,
                )
            except (TypeError, ValueError) as exc:
                stop_reason = f"invalid feedback motion sample: {exc}"
                rejected_motion_sample = {
                    "reason": stop_reason,
                    "guard": "sample_validation",
                    "sequence": int(state.sequence),
                    "feedback_timestamp_sec": state.receive_timestamp_sec,
                    "feedback_dt_sec": dt,
                    "raw_qpos_rad": raw_q,
                    "normalized_qpos_rad": q,
                    "previous_qvel_rad_s": previous_guard_qvel,
                    "current_qvel_rad_s": qvel,
                    "qacc_fd_rad_s2": None,
                    "command_dispatched": False,
                    "eligible_for_last_safe": False,
                }
                rejected_motion_sample_count += 1
                break
            if motion_rejection is not None:
                rejected_motion_sample = motion_rejection
                rejected_motion_sample_count += 1
                stop_reason = str(motion_rejection["reason"])
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
            last_control_elapsed_sec = elapsed
            handoff_fraction = startup_handoff_fraction(
                elapsed_sec=elapsed,
                rise_time_sec=args.torque_rise_time_sec,
            )
            if elapsed < args.torque_rise_time_sec:
                set_startup_state("GRAVITY_RAMP")
            elif handoff_fraction < 1.0:
                set_startup_state("HANDOFF")
            else:
                set_startup_state("ACTIVE")
            if calibration_capture is not None:
                if handoff_fraction >= 1.0 and not calibration_ready:
                    if calibration_keys is None:
                        raise LowGainProbeError(
                            "gravity-calibration operator terminal is unavailable"
                        )
                    calibration_keys.flush()
                    calibration_ready = True
                    _operator_notice(
                        "[B601] CALIBRATION FREE_DRAG: move joint1..joint6, "
                        "release external force, press P or N to label the "
                        "approach path, then press R to record "
                        f"{calibration_capture.record_duration_sec:g}s. "
                        "Unlabelled R is diagnostic only."
                    )
                started_record = calibration_capture.start_recording_if_due(
                    elapsed_sec=elapsed,
                    sample_start_index=len(samples),
                )
                if started_record is not None:
                    _operator_notice(
                        "[B601] CALIBRATION RECORDING "
                        f"#{started_record['index']}: settling complete; do not touch."
                    )
                completed_record = calibration_capture.complete_if_due(
                    elapsed_sec=elapsed,
                    sample_end_index=len(samples),
                )
                if completed_record is not None:
                    _operator_notice(
                        "[B601] CALIBRATION COMPLETE "
                        f"#{completed_record['index']}: returned to FREE_DRAG."
                    )
                capture_request = (False, None)
                if calibration_ready and calibration_keys is not None:
                    capture_request = calibration_keys.poll_capture_request()
                if capture_request[0]:
                    new_record = calibration_capture.trigger(
                        q,
                        elapsed_sec=elapsed,
                        sample_start_index=len(samples),
                        approach_direction=capture_request[1],
                    )
                    if new_record is not None:
                        direction_note = (
                            new_record["approach_direction"] or "unlabelled"
                        )
                        _operator_notice(
                            "[B601] CALIBRATION CAPTURED "
                            f"#{new_record['index']}: approach={direction_note}; "
                            "reference fixed; settling."
                        )
                qref = calibration_capture.reference(q)
                qvel_ref = _ZERO_SIX
                step_done = True
            else:
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
            pose_audit: dict[str, object] | None = None
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
            elif args.mode == "pose-hold":
                if pose_spring is None:
                    raise LowGainProbeError("Pose impedance spring was not armed")
                joint_torque_overlay_nm, pose_audit = compute_pose_impedance_overlay(
                    model,
                    pose_spring,
                    qpos_rad=q,
                    qvel_rad_s=qvel,
                    max_displacement_m=args.cartesian_max_displacement_m,
                    max_rotation_rad=args.rotational_max_displacement_rad,
                )
            reference = build_reference(
                sequence=state.sequence,
                timestamp_sec=state.receive_timestamp_sec,
                qpos_rad=qref,
                qvel_rad_s=qvel_ref,
                joint_torque_overlay_nm=joint_torque_overlay_nm,
            )
            target_torque_breakdown = compute_software_torque_breakdown(
                model,
                reference=reference,
                measured_qpos_rad=q,
                measured_qvel_rad_s=qvel,
                profile=profile,
                gravity_scale=args.gravity_scale,
            )
            # Gravity is present at its requested scale from the first active
            # frame.  The total-torque conditioner below is the single startup
            # slew; applying a second scalar gravity ramp here would create a
            # compensation gap and allow the arm to sag during handoff.
            gravity_ramp = 1.0
            startup_lock_reference = build_reference(
                sequence=state.sequence,
                timestamp_sec=state.receive_timestamp_sec,
                qpos_rad=initial_q,
                qvel_rad_s=_ZERO_SIX,
            )
            startup_lock_torque_breakdown = compute_software_torque_breakdown(
                model,
                reference=startup_lock_reference,
                measured_qpos_rad=q,
                measured_qvel_rad_s=qvel,
                profile=startup_lock_profile,
                gravity_scale=args.gravity_scale,
            )
            torque_breakdown = blend_startup_lock_torque_breakdown(
                startup_lock_torque_breakdown,
                target_torque_breakdown,
                gravity_fraction=gravity_ramp,
                handoff_fraction=handoff_fraction,
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
                # Driver Kp stays zero so the end-effector spring remains the
                # sole position-restoring owner.  Optional driver Kd supplies
                # only motor-local damping behind verified 0x700B bounds.
                kp=_ZERO_SIX,
                kd=driver_kd_nm_s_per_rad,
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
                    "feedback_dt_sec": dt,
                    "raw_qpos_rad": raw_q,
                    "qpos_rad": q,
                    "boundary_normalized_joint_names": normalized_joint_names,
                    "qvel_fd_rad_s": qvel,
                    "qacc_fd_rad_s2": qacc,
                    "qref_rad": qref,
                    "qvel_ref_rad_s": qvel_ref,
                    "startup_state": startup_state_history[-1],
                    "gravity_ramp_fraction": gravity_ramp,
                    "startup_handoff_fraction": handoff_fraction,
                    "calibration_state": (
                        calibration_capture.state
                        if calibration_capture is not None
                        else None
                    ),
                    "calibration_record_index": (
                        calibration_capture.active_record_index
                        if calibration_capture is not None
                        else None
                    ),
                    "calibration_record_elapsed_sec": (
                        calibration_capture.record_elapsed_sec(elapsed)
                        if calibration_capture is not None
                        else None
                    ),
                    "startup_lock_position_tau_nm": (
                        startup_lock_torque_breakdown.position_torque_nm
                    ),
                    "startup_lock_damping_tau_nm": (
                        startup_lock_torque_breakdown.damping_torque_nm
                    ),
                    "gravity_target_tau_nm": (
                        target_torque_breakdown.gravity_torque_nm
                    ),
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
                    "pose_impedance": pose_audit,
                    "freshness_verified": state.freshness_verified,
                    "freshness_semantics": state.freshness_semantics,
                }
            )
            previous_q = q
            previous_feedback_time = state.receive_timestamp_sec
            previous_guard_qvel = tuple(float(value) for value in qvel)
            previous_tau = conditioned_tau
            last_verified_gravity_tau = _six(
                torque_breakdown.gravity_torque_nm,
                name="active_gravity_tau_nm",
            )
            if safe_candidate_q is not None:
                last_safe_q = safe_candidate_q
                last_safe_qvel = tuple(float(value) for value in qvel)
                last_safe_observed_at_sec = observed_at_sec
            next_deadline += period
            completed_at = time.monotonic()
            if completed_at > next_deadline:
                schedule_overruns += 1
                next_deadline = completed_at
        if stop_reason is None:
            control_completion_reason = complete_active_duration(
                stop=stop,
                calibration_capture=calibration_capture,
                elapsed_sec=last_control_elapsed_sec,
                sample_end_index=len(samples),
            )
            _operator_notice(
                "[B601] CONTROL DURATION COMPLETE: entering protective hold; "
                "Ctrl-C or SIGUSR1 requests final full disable."
            )
    except LowGainProbeError as exc:
        primary_error = exc
    except (B601HardwareError, OSError, TypeError, ValueError) as exc:
        primary_error = exc
    except Exception as exc:  # noqa: BLE001 - report unexpected probe failure
        primary_error = exc
    finally:
        if calibration_keys is not None:
            calibration_keys.close()
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
            explicit_full_disable = stop.get("full_disable_requested", False)
            shutdown_action = resolve_shutdown_action(
                enabled=enabled,
                full_disable_requested=explicit_full_disable,
            )
            protective_hold_remains_latched = False
            if shutdown_action == "protective_hold":
                # A complete current fault sample is preferred for a normal
                # control/position error.  An explicit operator interrupt
                # normally remains anchored to the previously admitted
                # stationary sample; if that sample aged out, a complete
                # current frame is admitted instead so the first Ctrl-C does
                # not unnecessarily drop the whole arm.
                operator_interrupt = bool(
                    stop.get("requested", False)
                    and control_completion_reason is None
                    and (
                        stop_reason is None
                        or stop_reason.startswith("operator_interrupt")
                    )
                )
                duration_complete = control_completion_reason == "duration_complete"
                protective_hold_entry_reason = (
                    control_completion_reason
                    or stop_reason
                    or (str(primary_error) if primary_error is not None else None)
                    or "enabled_control_exit"
                )
                if last_safe_observed_at_sec is None:
                    safe_age_sec = math.inf
                else:
                    safe_age_sec = max(
                        0.0,
                        time.monotonic() - last_safe_observed_at_sec,
                    )
                safe_snapshot_available = (
                    last_safe_q is not None
                    and last_safe_qvel is not None
                    and safe_age_sec <= args.fault_hold_max_age_sec
                )
                fault_snapshot_available = (
                    fault_qpos_for_hold is not None
                    and fault_qvel_for_hold is not None
                )
                # A Ctrl-C normally uses the stationary last-safe snapshot.
                # If that candidate has aged out while a complete current
                # frame is still available, use the same bounded current-fault
                # admission as a handled position error instead of dropping
                # the whole arm immediately.
                degraded_entry_reason: str | None = None
                if (operator_interrupt or duration_complete) and safe_snapshot_available:
                    fault_entry = False
                    can_attempt_hold = True
                elif fault_snapshot_available:
                    fault_entry = True
                    can_attempt_hold = True
                elif safe_snapshot_available:
                    # Startup and other enabled-path exceptions can occur
                    # before the active loop has produced a fault snapshot.
                    # A still-fresh admitted pose can nevertheless seed a live
                    # gravity/damping hold instead of forcing drive disable.
                    fault_entry = False
                    can_attempt_hold = True
                elif previous_q is not None and last_verified_gravity_tau is not None:
                    # Loss of the current RX/control sample does not prove loss
                    # of the TX path.  Enter a zero-Kp cached-gravity hold and
                    # continue polling instead of dropping a loaded arm.
                    fault_entry = False
                    can_attempt_hold = True
                    degraded_entry_reason = (
                        stop_reason
                        or str(primary_error)
                        or "live fault snapshot unavailable"
                    )
                else:
                    fault_entry = False
                    can_attempt_hold = False
                if can_attempt_hold:
                    if degraded_entry_reason is not None:
                        stop["requested"] = True
                        hold_qpos = previous_q
                        hold_qvel = _ZERO_SIX
                        hold_age_sec = math.inf
                        protective_hold_reference_source = (
                            "last_complete_command_cache"
                        )
                    elif fault_entry:
                        # A handled control/position fault already consumed
                        # the first protective event.  Treat the latched hold
                        # as the first stop request so one subsequent Ctrl-C
                        # (or SIGUSR1) escalates directly to full disable.
                        stop["requested"] = True
                    if degraded_entry_reason is not None:
                        pass
                    elif fault_entry:
                        if fault_observed_at_sec is None:
                            fault_age_sec = math.inf
                        else:
                            fault_age_sec = max(
                                0.0,
                                time.monotonic() - fault_observed_at_sec,
                            )
                        hold_qpos = fault_qpos_for_hold
                        hold_qvel = fault_qvel_for_hold
                        hold_age_sec = fault_age_sec
                        protective_hold_reference_source = (
                            "current_snapshot_after_stale_duration_last_safe"
                            if duration_complete
                            else (
                                "current_fault_snapshot"
                                if not operator_interrupt
                                else "current_fault_snapshot_after_stale_last_safe"
                            )
                        )
                    else:
                        hold_qpos = last_safe_q
                        hold_qvel = last_safe_qvel
                        hold_age_sec = safe_age_sec
                        protective_hold_reference_source = (
                            "duration_complete_last_safe_snapshot"
                            if duration_complete
                            else "last_safe_snapshot"
                        )
                    if hold_age_sec is None:
                        hold_age_sec = math.inf
                    protective_hold_admission_age_sec = hold_age_sec
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
                            qpos_rad=hold_qpos,
                            qvel_rad_s=hold_qvel,
                            observed_age_sec=hold_age_sec,
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
                            driver_kd_nm_s_per_rad=driver_kd_nm_s_per_rad,
                            hold_profile=protective_hold_profile,
                            # A duration boundary, like an operator transition,
                            # needs a new monitoring baseline on the first hold
                            # feedback sample.
                            operator_interrupt=(
                                operator_interrupt or duration_complete
                            ),
                            fault_entry=fault_entry,
                            fault_entry_max_velocity_rad_s=args.velocity_abort_rad_s,
                            monitor_velocity_rad_s=args.velocity_abort_rad_s,
                            monitor_acceleration_rad_s2=args.acceleration_abort_rad_s2,
                            cached_qpos_rad=previous_q,
                            cached_gravity_tau_nm=last_verified_gravity_tau,
                            degraded_entry_reason=degraded_entry_reason,
                            degraded_telemetry=protective_hold_degraded,
                        )
                        cleanup_errors.extend(hold_errors)
                        protective_hold_latched = (
                            protective_hold_first_receipt is not None
                            and protective_hold_commands > 0
                            and (
                                protective_hold_non_hold_receipt is None
                                or protective_hold_non_hold_receipt.completed
                            )
                        )
                        if protective_hold_full_disable_receipt is not None:
                            cleanup_receipt = protective_hold_full_disable_receipt
                        # A successful operator hold is intentionally
                        # long-lived. The second Ctrl-C/SIGUSR1 or inability
                        # to send both live and cached complete frames is what
                        # authorizes full disable.
                        protective_hold_remains_latched = (
                            protective_hold_latched
                            and not hold_errors
                            and protective_hold_full_disable_receipt is None
                            and not stop.get("full_disable_requested", False)
                        )
                    except Exception as exc:  # noqa: BLE001 - preserve safety evidence
                        # Expected RX, numeric, model, and dispatch failures are
                        # handled inside _run_latched_fault_hold.  Reaching this
                        # boundary means the hold implementation itself failed.
                        cleanup_errors.append(f"protective hold: {exc}")
                        protective_hold_remains_latched = False
                else:
                    cleanup_errors.append(
                        "protective hold refused: no verified pose/gravity cache"
                    )
                    protective_hold_degraded["escalation_reason"] = (
                        "no verified pose/gravity cache"
                    )
                    protective_hold_remains_latched = False
                if not protective_hold_remains_latched:
                    issue_full_safe_stop()
            else:
                issue_full_safe_stop()
            if getattr(args, "apply_driver_torque_limits", False):
                try:
                    driver_limit_restore_receipt = (
                        backend.restore_driver_torque_limits()
                    )
                    if not driver_limit_restore_receipt.completed:
                        cleanup_errors.extend(
                            f"restore driver torque limit: {error}"
                            for error in driver_limit_restore_receipt.errors
                        )
                        if not driver_limit_restore_receipt.errors:
                            cleanup_errors.append(
                                "restore driver torque limit incomplete without vendor error"
                            )
                except Exception as exc:  # noqa: BLE001 - preserve cleanup evidence
                    cleanup_errors.append(f"restore driver torque limit: {exc}")
        try:
            backend.close()
        except Exception as exc:  # noqa: BLE001 - preserve cleanup evidence
            cleanup_errors.append(f"close: {exc}")
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGUSR1, old_sigusr1)

    if calibration_capture is not None and calibration_capture.state in (
        "SETTLING",
        "RECORDING",
    ):
        calibration_capture.interrupt(
            elapsed_sec=last_control_elapsed_sec,
            sample_end_index=len(samples),
            reason=stop_reason or "probe_duration_complete",
        )

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
    driver_limit_cleanup_ok = (
        not getattr(args, "apply_driver_torque_limits", False)
        or (
            driver_limit_restore_receipt is not None
            and driver_limit_restore_receipt.completed
        )
    )
    cleanup_ok = (
        cleanup_receipt is not None
        and cleanup_receipt.completed
        and driver_limit_cleanup_ok
        and not cleanup_errors
    )
    calibration_acceptance = evaluate_gravity_calibration_capture(
        calibration_capture
    )
    calibration_passed = calibration_acceptance["status"] in (
        "PASS",
        "NOT_APPLICABLE",
    )
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
    acceptance_passed = control_passed and response_passed and calibration_passed
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
        "position_limits_rad": position_limits,
        "feedback_position_limits_rad": feedback_position_limits,
        "position_limit_semantics": (
            "pinned URDF limits with precise J4 +/-pi/2 nominal override; "
            "feedback-only boundary windows are reported separately"
        ),
        "feedback_admission_semantics": (
            "backend rejects finite positions outside the explicit read-side "
            "envelope before FK/gravity/impedance/dispatch; approved J2/J3/J4 "
            "boundary windows are read-only and do not relax outgoing limits"
        ),
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
            "velocity_limit_source": "normal_motion_velocity_abort",
            "last_qvel_rad_s": arming_settle_last_qvel,
            "torque_commands_sent": arming_settle_torque_commands,
            "reference_qpos_rad": initial_q,
            "first_lock_command_receipt": _command_receipt_payload(
                startup_lock_first_receipt
            ),
        },
        "startup_state_machine": {
            "state_history": tuple(startup_state_history),
            "startup_lock_commands": startup_lock_commands,
            "startup_lock_gain_source": "rebotarm_rs_yaml.MIT",
            "startup_lock_software_kp_nm_per_rad": (
                startup_lock_profile.position_kp_nm_per_rad
            ),
            "startup_lock_software_kd_nm_s_per_rad": (
                startup_lock_profile.damping_kd_nm_s_per_rad
            ),
            "vendor_kp_nm_per_rad": _ZERO_SIX,
            "vendor_kd_nm_s_per_rad": driver_kd_nm_s_per_rad,
            "gravity_ramp_duration_sec": args.torque_rise_time_sec,
            "gravity_ramp_note": (
                "state label covers the final total-torque slew; the requested "
                "gravity vector is not independently attenuated"
            ),
            "handoff_duration_sec": args.torque_rise_time_sec,
            "gravity_ramp_semantics": "complete_six_joint_vector",
            "handoff_semantics": (
                "temporary_joint_lock_to_final_controller; "
                "end_effector_impedance_restores_pose"
            ),
            "all_six_joint_dispatch": True,
        },
        "motors_enabled_by_probe": enabled,
        "motion_commands_sent": commands_sent > 0,
        "commands_sent": commands_sent,
        "freeze_commands_sent": freeze_commands,
        "duration_sec": args.duration_sec,
        "control_completion_reason": control_completion_reason,
        "rate_hz": args.rate_hz,
        "gravity_calibration": {
            "enabled": calibration_capture is not None,
            "interaction": "FREE_DRAG -> R -> SETTLING -> RECORDING -> FREE_DRAG",
            "record_duration_sec": (
                calibration_capture.record_duration_sec
                if calibration_capture is not None
                else None
            ),
            "settling_duration_sec": (
                calibration_capture.settling_duration_sec
                if calibration_capture is not None
                else None
            ),
            "records": (
                calibration_capture.records
                if calibration_capture is not None
                else []
            ),
            "completed_record_count": calibration_acceptance[
                "completed_record_count"
            ],
            "incomplete_record_count": calibration_acceptance[
                "incomplete_record_count"
            ],
            "acceptance_status": calibration_acceptance["status"],
            "acceptance_reasons": calibration_acceptance["reasons"],
            "final_state": (
                calibration_capture.state
                if calibration_capture is not None
                else None
            ),
            "free_drag_reference_semantics": "current_feedback_pose_each_cycle",
            "recording_reference_semantics": "qpos_captured_on_R_after_settling",
            "settling_reference_semantics": "qpos_captured_on_R_fixed_with_nonzero_software_kp",
            "sample_reduction_semantics": (
                "one_median_observation_per_window; frames are not independent poses"
            ),
        },
        "stop_reason": stop_reason,
        "cleanup_errors": tuple(cleanup_errors),
        "preflight_disable_receipt": _mode_receipt_payload(preflight_disable),
        "mode_receipt": _mode_receipt_payload(mode_receipt),
        "enable_receipt": _mode_receipt_payload(enable_receipt),
        "cleanup_receipt": _mode_receipt_payload(cleanup_receipt),
        "driver_torque_limits": {
            "requested": bool(
                getattr(args, "apply_driver_torque_limits", False)
            ),
            "source": "pinned_urdf_effort_scaled",
            "requested_limits_nm": torque_limits,
            "volatile_parameter": "RobStride limit_torque 0x700B",
            "apply_receipt": _driver_torque_limit_receipt_payload(
                driver_limit_apply_receipt
            ),
            "restore_receipt": _driver_torque_limit_receipt_payload(
                driver_limit_restore_receipt
            ),
        },
        "protective_hold": {
            "latched": protective_hold_latched,
            "entry_reason": protective_hold_entry_reason,
            "joint_names": _FAULT_HOLD_JOINTS,
            "reference_source": protective_hold_reference_source,
            "command_count": protective_hold_commands,
            "first_command_receipt": _command_receipt_payload(
                protective_hold_first_receipt
            ),
            "non_hold_disable_required": bool(_NON_FAULT_HOLD_JOINTS),
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
            "admission_max_velocity_rad_s": args.fault_hold_max_velocity_rad_s,
            # Kept as an alias for existing evidence consumers; this is the
            # stationary-admission gate, not the draggable-hold motion limit.
            "max_velocity_rad_s": args.fault_hold_max_velocity_rad_s,
            "max_rebase_rad": args.fault_hold_max_rebase_rad,
            "vendor_kp_nm_per_rad": _ZERO_SIX,
            "vendor_kd_nm_s_per_rad": driver_kd_nm_s_per_rad,
            "software_kp_nm_per_rad": protective_hold_profile.position_kp_nm_per_rad,
            "software_kd_nm_s_per_rad": protective_hold_profile.damping_kd_nm_s_per_rad,
            "damping_owner": (
                "driver_kd_where_nonzero_else_protective_hold_software_kd"
            ),
            "default_software_damping_kd_nm_s_per_rad": getattr(
                args,
                "protective_hold_kd",
                _DEFAULT_PROTECTIVE_HOLD_DAMPING_KD,
            ),
            "reference_semantics": "current_feedback_pose_each_cycle",
            "position_lock": False,
            "reference_velocity_rad_s": _ZERO_SIX,
            "motion_velocity_limit_rad_s": args.velocity_abort_rad_s,
            "motion_acceleration_limit_rad_s2": args.acceleration_abort_rad_s2,
            "degraded": {
                **protective_hold_degraded,
                "cache_semantics": (
                    "last_complete_command_pose_plus_gravity_only; no position or "
                    "Cartesian overlay; configured driver damping remains active"
                ),
                "software_damping_when_feedback_untrusted": False,
                "complete_frame_retry_policy": "one immediate retry",
                "full_disable_policy": (
                    "explicit second stop or live-and-cached six-joint dispatch failure"
                ),
            },
        },
        "fault_hold_candidate_admission": {
            "accepted_count": fault_hold_candidate_accepted_count,
            "rejected_count": fault_hold_candidate_rejected_count,
            "last_rejection": fault_hold_candidate_last_rejection,
            "max_age_sec": args.fault_hold_max_age_sec,
            "max_velocity_rad_s": args.fault_hold_max_velocity_rad_s,
            "semantics": "rejected_sample_retains_previous_last_safe",
        },
        "motion_guard": {
            "velocity_abort_rad_s": args.velocity_abort_rad_s,
            "acceleration_abort_rad_s2": args.acceleration_abort_rad_s2,
            "acceleration_threshold_status": "commissioning_seed_not_hardware_certified",
            "qvel_source": "normalized_qpos_finite_difference_over_feedback_dt",
            "qacc_source": "consecutive_qvel_finite_difference_over_feedback_dt",
            "first_sample_qacc_semantics": "unavailable_without_previous_control_qvel",
            "rejected_sample_count": rejected_motion_sample_count,
            "rejected_sample": rejected_motion_sample,
            "backend_feedback_diagnostics": backend_feedback_diagnostics,
            "rejected_sample_command_semantics": "not_dispatched",
            "rejected_sample_last_safe_semantics": "ineligible",
        },
        "initial_qpos_rad": initial_q,
        "final_qpos_rad": final_q,
        "profile": {
            "software_kp_nm_per_rad": profile.position_kp_nm_per_rad,
            "software_kd_nm_s_per_rad": profile.damping_kd_nm_s_per_rad,
            "gravity_scale": args.gravity_scale,
            "gravity_ramp": {
                "type": "full_vector_immediate_with_total_torque_slew",
                "duration_sec": args.torque_rise_time_sec,
                "applies_to": "complete_six_joint_gravity_vector",
                "scalar_gravity_at_first_active_frame": args.gravity_scale,
                "independent_scalar_ramp": False,
                "non_gravity_terms_scaled": False,
            },
            "torque_limits_nm": torque_limits,
            "torque_limit_scale": args.torque_limit_scale,
            "max_torque_rate_nm_s": profile.max_torque_rate_nm_s,
            "max_abs_overlay_torque_nm": profile.max_abs_overlay_torque_nm,
            "vendor_kp_nm_per_rad": _ZERO_SIX,
            "vendor_kd_nm_s_per_rad": driver_kd_nm_s_per_rad,
            "torque_owner": (
                "software_position_gravity_task_wrench; optional_driver_damping"
            ),
            "startup_lock_profile": {
                "software_kp_nm_per_rad": startup_lock_profile.position_kp_nm_per_rad,
                "software_kd_nm_s_per_rad": startup_lock_profile.damping_kd_nm_s_per_rad,
                "gain_source": "rebotarm_rs_yaml.MIT",
                "vendor_kp_nm_per_rad": _ZERO_SIX,
            },
        },
        "slow_step_response": slow_step_response,
        "joint_impedance": {
            "enabled": args.mode in ("hold", "gravity-calibration"),
            "reference_source": (
                "interactive_current_or_R_captured_qpos"
                if args.mode == "gravity-calibration"
                else "fresh_post_enable_settled_qpos"
            ),
            "reference_qpos_rad": initial_q if args.mode == "hold" else None,
            "reference_velocity_rad_s": _ZERO_SIX,
            "software_kp_nm_per_rad": profile.position_kp_nm_per_rad,
            "software_kd_nm_s_per_rad": profile.damping_kd_nm_s_per_rad,
            "vendor_kp_nm_per_rad": _ZERO_SIX,
            "vendor_kd_nm_s_per_rad": driver_kd_nm_s_per_rad,
            "final_total_torque_bounded": (
                all(value == 0.0 for value in driver_kd_nm_s_per_rad)
                or bool(getattr(args, "apply_driver_torque_limits", False))
            ),
            "final_total_torque_bound_owner": (
                "driver_limit_torque_0x700B"
                if any(value > 0.0 for value in driver_kd_nm_s_per_rad)
                else "software_condition_torque"
            ),
            "final_driver_total_torque_bounded": bool(
                getattr(args, "apply_driver_torque_limits", False)
            ),
            "absolute_pose_commanded": False,
        },
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
        "pose_impedance": {
            "enabled": args.mode == "pose-hold",
            "translation_only": False,
            "anchor_source": "fresh_post_enable_fk",
            "anchor_pose": pose_anchor,
            "translational_stiffness_n_per_m": (
                args.cartesian_stiffness_n_per_m,
            )
            * 3,
            "translational_damping_n_s_per_m": (
                args.cartesian_damping_n_s_per_m,
            )
            * 3,
            "rotational_stiffness_nm_per_rad": (
                args.rotational_stiffness_nm_per_rad,
            )
            * 3,
            "rotational_damping_nm_s_per_rad": (
                args.rotational_damping_nm_s_per_rad,
            )
            * 3,
            "max_abs_force_n": (args.cartesian_max_force_n,) * 3,
            "max_abs_torque_nm": (args.rotational_max_torque_nm,) * 3,
            "max_translation_m": args.cartesian_max_displacement_m,
            "max_rotation_rad": args.rotational_max_displacement_rad,
            "overlay_limit_scale": args.cartesian_overlay_limit_scale,
            "requires_all_six_arm_joints": True,
            "joint_position_stiffness_disabled": all(
                gain == 0.0 for gain in profile.position_kp_nm_per_rad
            ),
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
        "feedback_boundary_normalization": {
            "joints": _FEEDBACK_BOUNDARY_NORMALIZATION_JOINTS,
            "intervals_rad": {
                joint_name: (
                    _FEEDBACK_BOUNDARY_NORMALIZATION_REFERENCE_RAD[joint_name]
                    - _FEEDBACK_BOUNDARY_NORMALIZATION_MAX_NEGATIVE_RAD[joint_name],
                    _FEEDBACK_BOUNDARY_NORMALIZATION_REFERENCE_RAD[joint_name],
                )
                for joint_name in _FEEDBACK_BOUNDARY_NORMALIZATION_JOINTS
            },
            "projection_rad": {
                joint_name: _FEEDBACK_BOUNDARY_NORMALIZATION_REFERENCE_RAD[joint_name]
                for joint_name in _FEEDBACK_BOUNDARY_NORMALIZATION_JOINTS
            },
            "interval_semantics": "inclusive_lower_exclusive_upper",
            "applied_sample_count": sum(
                bool(sample.get("boundary_normalized_joint_names"))
                for sample in samples
            ),
            "applied_joint_counts": {
                joint_name: sum(
                    joint_name in sample.get("boundary_normalized_joint_names", ())
                    for sample in samples
                )
                for joint_name in _FEEDBACK_BOUNDARY_NORMALIZATION_JOINTS
            },
            "raw_values_preserved": True,
            "semantics": (
                "feedback_admission_only; URDF limits and outgoing references remain strict"
            ),
        },
        "samples": samples,
        "notes": [
            "Human-run commissioning only; automated tests never execute this active path.",
            "The active set is the target plus explicitly selected support joints; scheme-1 J6 commissioning selects all six arm joints while only the J6 reference changes.",
            "After enable, duration completion, an ordinary control exit, or the first Ctrl-C keeps joint1..joint6 in the protective gravity/damping hold; joint7 is outside this backend, and SIGUSR1 or the next Ctrl-C requests full disable.",
            "Protective hold normally admits a fresh stationary last-safe snapshot for an operator stop or a complete current in-limit fault snapshot; if RX/control has already failed, it can enter degraded hold from the pose and pure-gravity cache paired with the last complete six-joint dispatch.",
            "A control sample that fails only fault-hold candidate admission is not promoted to last-safe; normal control remains governed by its separate velocity abort and all other watchdogs.",
            "While latched, a valid cycle updates gravity from current feedback; RX, numeric, motion-monitor, or model failure keeps sending cached gravity with zero Kp and configured motor-local damping while feedback recovery is retried.",
            "A complete six-joint MIT command is retried once; only failure of both live and cached complete-frame dispatch, absence of any verified cache, or an explicit second stop request permits full safe_stop.",
            "Protective hold has software Kp=0 and one derivative owner per joint: explicit driver Kd suppresses the default software damping for that joint; no two damping terms are stacked.",
            "J2 feedback readings in [-6e-3, 0) rad and J3 readings in [-3e-3, 0) rad are represented as 0 rad at admission; J4 readings in [-pi/2-1e-3, -pi/2) rad are represented as -pi/2. Raw active samples are retained, and values beyond each compatibility window remain outside the strict effective limit.",
            "Gripper motor ID7 is outside this backend and is not included in any all-arm receipt.",
            "The probe never chooses an absolute pose. hold starts at the post-enable settled pose; slow-step uses a bounded relative step with a 20 degree command and observed-excursion hard limit.",
            "After enable, the active path captures a fresh q0 and immediately sends an all-six-joint bounded software position lock at q0; lock frames continue during the four-sample arming-settle gate.",
            "cartesian-hold anchors fresh post-enable FK position, controls translation only, and requires all six arm joints.",
            "pose-hold anchors the complete fresh post-enable FK transform and applies bounded world-aligned force and torque through the same J.T wrench path.",
            "Software composes gravity, position, optional finite-difference damping, and a bounded J.T wrench overlay; vendor MIT kp remains zero, while explicit vendor kd may own motor-local damping behind verified 0x700B limits.",
            "Startup keeps the YAML-derived software joint lock while the requested complete six-joint gravity vector is present from the first lock frame, then linearly hands position/damping/pose terms over to the final controller; every frame remains under the same final absolute and slew-rate conditioner.",
            "When explicitly requested, the probe temporarily mirrors every arm joint's software total-torque envelope into volatile RobStride limit_torque (0x700B), verifies every readback before enable, and restores every original value after full safe_stop.",
            "Gravity uses B601PinocchioModel with Pinocchio-neutral auxiliary gripper coordinates; actual gripper load is not identified.",
            "The explicit mechPos path is synchronous and normally around the observed low-bandwidth feedback rate; this is not a high-rate haptic loop.",
            "A slow-step PASS additionally requires a completed reference, correct-direction bounded target response, and bounded support-joint drift. It is not a HARDWARE_PASS or force-sensor validation.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "B601-RS low-gain gravity/damping commissioning; motion-capable and "
            "human-run only, with an offline gravity-calibration-fit mode."
        )
    )
    parser.add_argument(
        "--mode",
        choices=(
            "hold",
            "slow-step",
            "cartesian-hold",
            "pose-hold",
            "gravity-calibration",
            "gravity-calibration-fit",
            "feedback-stability",
            "driver-torque-limit-roundtrip",
        ),
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
    parser.add_argument(
        "--calibration-duration-sec",
        type=float,
        default=_DEFAULT_CALIBRATION_DURATION_SEC,
        help=(
            "fixed-reference recording duration after each R key in "
            "gravity-calibration mode"
        ),
    )
    parser.add_argument(
        "--calibration-settling-sec",
        type=float,
        default=_DEFAULT_CALIBRATION_SETTLING_SEC,
        help=(
            "time after R during which the fixed reference settles under the "
            "non-zero calibration Kp; these samples are excluded from the fit"
        ),
    )
    parser.add_argument(
        "--calibration-input",
        type=Path,
        help="completed gravity-calibration JSON for offline fitting",
    )
    parser.add_argument(
        "--calibration-validation-input",
        type=Path,
        help="independent gravity-calibration JSON used for holdout evaluation",
    )
    parser.add_argument(
        "--calibration-overlay-output",
        type=Path,
        help="candidate gravity overlay JSON written by gravity-calibration-fit",
    )
    parser.add_argument(
        "--calibration-report-output",
        type=Path,
        help="optional full calibration fit report JSON",
    )
    parser.add_argument(
        "--calibration-model",
        choices=(
            "global-scale",
            "global-scale-joint-bias",
            "global-scale-payload",
        ),
        default="global-scale-joint-bias",
        help=(
            "effective gravity correction parameterization for offline fitting; "
            "payload adds an unknown non-negative tool mass and first moment"
        ),
    )
    parser.add_argument(
        "--calibration-pose-tolerance-rad",
        type=float,
        default=_DEFAULT_CALIBRATION_POSE_TOLERANCE_RAD,
        help="maximum per-joint reference difference used to infer a pose pair",
    )
    parser.add_argument(
        "--calibration-max-poses",
        type=int,
        help="optional cap on selected information-rich pose observations",
    )
    parser.add_argument(
        "--calibration-max-velocity-rad-s",
        type=float,
        help="optional per-window stability velocity criterion for fitting",
    )
    parser.add_argument(
        "--calibration-max-position-span-rad",
        type=float,
        help="optional per-window stability position-span criterion for fitting",
    )
    parser.add_argument(
        "--calibration-allow-unpaired",
        action="store_true",
        help="allow single-direction windows as a diagnostic candidate; not formal acceptance",
    )
    parser.add_argument("--warmup-sec", type=float, default=_DEFAULT_WARMUP_SEC)
    parser.add_argument(
        "--freeze-sec",
        type=float,
        default=_DEFAULT_FREEZE_SEC,
        help=(
            "legacy command-line compatibility value; an enabled probe now "
            "enters latched protective hold instead of timed freeze-and-disable"
        ),
    )
    parser.add_argument("--rate-hz", type=float, default=_DEFAULT_RATE_HZ)
    parser.add_argument(
        "--joint",
        choices=_ALL_JOINTS,
        default="joint1",
        help=(
            "slow-step target and optional gain-override joint; all impedance "
            "modes always enable and command joint1..joint6"
        ),
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
        "--driver-kd",
        type=float,
        default=0.0,
        help=(
            "optional all-arm motor-local MIT damping; requires "
            "--apply-driver-torque-limits and does not add position stiffness"
        ),
    )
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
    parser.add_argument(
        "--protective-hold-kd",
        type=float,
        default=_DEFAULT_PROTECTIVE_HOLD_DAMPING_KD,
        help=(
            "software damping used by the draggable Ctrl-C protective hold "
            "while feedback remains valid and no non-zero --driver-kd is selected"
        ),
    )
    parser.add_argument("--gravity-scale", type=float, default=_DEFAULT_GRAVITY_SCALE)
    parser.add_argument(
        "--torque-limit-scale",
        type=float,
        default=_DEFAULT_TORQUE_LIMIT_SCALE,
        help="fraction of the pinned URDF effort ceiling; must be in (0, 1]",
    )
    parser.add_argument(
        "--apply-driver-torque-limits",
        action="store_true",
        help=(
            "temporarily mirror all six scaled URDF torque envelopes into "
            "volatile RobStride limit_torque (0x700B), with per-joint readback "
            "and restoration"
        ),
    )
    parser.add_argument(
        "--torque-rise-time-sec",
        type=float,
        default=_DEFAULT_TORQUE_RISE_TIME_SEC,
        help=(
            "shared six-joint startup total-torque slew and lock-handoff "
            "duration; gravity is not independently attenuated"
        ),
    )
    parser.add_argument(
        "--velocity-abort-rad-s",
        type=float,
        default=_DEFAULT_VELOCITY_ABORT_RAD_S,
        help=(
            "normal-control finite-difference velocity abort threshold; "
            "independent from the stricter fault-hold stationarity gate"
        ),
    )
    parser.add_argument(
        "--acceleration-abort-rad-s2",
        type=float,
        default=_DEFAULT_ACCELERATION_ABORT_RAD_S2,
        help=(
            "finite-difference acceleration abort threshold; the default is "
            "an unverified commissioning seed, not a hardware-certified limit"
        ),
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
        help=(
            "maximum absolute observed velocity for the stationary protective-"
            "hold admission snapshot"
        ),
    )
    parser.add_argument(
        "--fault-hold-max-rebase-rad",
        type=float,
        default=_DEFAULT_FAULT_HOLD_MAX_REBASE_RAD,
        help=(
            "maximum fault-to-last-safe rebase distance used by admission; "
            "not a drag limit once protective hold is active"
        ),
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
    parser.add_argument(
        "--rotational-stiffness-nm-per-rad",
        type=float,
        default=_DEFAULT_ROTATIONAL_STIFFNESS_NM_PER_RAD,
        help="isotropic world-aligned rotational stiffness for pose-hold",
    )
    parser.add_argument(
        "--rotational-damping-nm-s-per-rad",
        type=float,
        default=_DEFAULT_ROTATIONAL_DAMPING_NM_S_PER_RAD,
        help="isotropic world-aligned rotational damping for pose-hold",
    )
    parser.add_argument(
        "--rotational-max-torque-nm",
        type=float,
        default=_DEFAULT_ROTATIONAL_MAX_TORQUE_NM,
        help="per-axis end-effector torque cap for pose-hold",
    )
    parser.add_argument(
        "--rotational-max-displacement-rad",
        type=float,
        default=_DEFAULT_ROTATIONAL_MAX_DISPLACEMENT_RAD,
        help="orientation-error-norm abort threshold for pose-hold",
    )
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument(
        "--auto-disable",
        action="store_true",
        help=(
            "required acknowledgement that an explicit final stop may disable "
            "the arm; duration completion itself enters protective hold"
        ),
    )
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--output", type=Path, help="write JSON outside the source worktree")
    parser.add_argument(
        "--log",
        type=Path,
        help="write detailed stdout/stderr; defaults beside --output with a .log suffix",
    )
    return parser


def _resolve_runtime_log_path(args: argparse.Namespace) -> Path:
    configured = getattr(args, "log", None)
    if configured is not None:
        return Path(configured).expanduser().resolve()
    if args.output is not None:
        return Path(args.output).expanduser().resolve().with_suffix(".log")
    return Path(f"/tmp/b601-rs-low-gain-probe-{os.getpid()}.log")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime_log_path = _resolve_runtime_log_path(args)
    runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[B601] mode={args.mode} pid={os.getpid()} log={runtime_log_path}",
        flush=True,
    )
    if args.mode == "gravity-calibration-fit":
        print("[B601] OFFLINE: no CAN connection or motor state change.", flush=True)
    else:
        print(
            "[B601] STOP: Ctrl-C requests protective hold; Ctrl-C again or "
            "SIGUSR1 requests full six-joint disable. Physical E-stop remains primary.",
            flush=True,
        )
    with (
        runtime_log_path.open("a", encoding="utf-8") as runtime_log,
        contextlib.redirect_stdout(runtime_log),
        contextlib.redirect_stderr(runtime_log),
    ):
        print(
            f"[B601] run started mode={args.mode} pid={os.getpid()}",
            flush=True,
        )
        try:
            payload = run_probe(args)
        except (LowGainProbeError, B601HardwareError, OSError, ValueError) as exc:
            payload = {"status": "ERROR", "mode": args.mode, "error": str(exc)}
            exit_code = 2
        else:
            exit_code = 0 if payload.get("status") == "OK" else 1
        print(
            f"[B601] run finished status={payload.get('status')} exit={exit_code}",
            flush=True,
        )
    payload["runtime_log_path"] = str(runtime_log_path)
    _output_payload(payload, args.output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
