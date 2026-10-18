"""Bounded, ephemeral operator torque intent for B601 commissioning.

This module never owns CAN or computes gravity. The existing hardware owner
samples it once per admitted feedback cycle and remains the only dispatcher.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

from .joint_motion import B601WindowVelocityEstimator, window_stationarity_gate_reason
from .model import B601_RS_ARM_JOINT_NAMES

_ZERO_SIX = (0.0,) * 6
# Bounds for the opt-in J3 diagnostic, not a new default velocity source.
_J3_POSITION_COUNT_RAD = 0.000384
J3_HELD_PROGRESS_STEP_RAD = .99 * _J3_POSITION_COUNT_RAD
_VELOCITY_CHECK_WINDOW_SEC = 0.08
_VELOCITY_CHECK_GAP_SEC = 0.04
# Withdrawal, first post-zero feedback and final observation each need one
# bounded gap in addition to the full 80 ms measured window. No assist is kept.
_VELOCITY_CONFIRMATION_TIMEOUT_SEC = _VELOCITY_CHECK_WINDOW_SEC + 3 * _VELOCITY_CHECK_GAP_SEC
# Preserve the previously admitted post-zero observation envelope when the
# explicit acquisition's raw hard limit is independently raised.
_RELEASE_SUPPORT_RAW_SPEED_CEILING_RAD_S = 0.14
# Only these motion-readiness failures may wait BEFORE an automatic trial.
_ONSET_BASELINE_TRANSIENT_REASONS = frozenset({
    "ready", "velocity_window_not_ready", "not_stationary",
    "velocity_window_displacement_exceeded", "onset_capture_position_not_settled",
})


def begin_held_progress(now: float, q: float, hold_nm: float) -> dict[str, object]:
    """Canonical observation state shared by the two explicit J3 probes."""
    return {
        "stage": "hold", "entered_at_sec": now, "hold_nm": hold_nm,
        "candidate_qpos_rad": q, "first_hold_completed_at_sec": None,
        "first_feedback_at_sec": None, "min_qpos_rad": q, "max_qpos_rad": q,
        "maximum_progress_rad": 0., "last_counted_progress_rad": 0.,
        "distinct_progress_samples": 0, "last_progress_at_sec": None,
    }


def observe_held_progress(probe: dict[str, object], *, now: float, q: float, direction: float, post_hold: bool | None) -> float:
    """Count only new post-receipt frontiers, never repeated positions."""
    progress = direction * (q - probe["candidate_qpos_rad"])
    probe["maximum_progress_rad"] = max(probe["maximum_progress_rad"], progress)
    probe["min_qpos_rad"] = min(probe["min_qpos_rad"], q)
    probe["max_qpos_rad"] = max(probe["max_qpos_rad"], q)
    if post_hold is False:
        probe["last_counted_progress_rad"] = max(probe["last_counted_progress_rad"], progress)
    elif post_hold:
        if probe["first_feedback_at_sec"] is None:
            probe["first_feedback_at_sec"] = now
        if progress >= probe["last_counted_progress_rad"] + J3_HELD_PROGRESS_STEP_RAD:
            probe["distinct_progress_samples"] += 1
            probe["last_counted_progress_rad"] = progress
            probe["last_progress_at_sec"] = now
    return progress


def _bounded_count_sample(dt: float, delta: float, velocity: float, hard_limit: float) -> bool:
    """Recognize the observed count/dt shape, not physical speed safety."""
    return (
        0 < dt <= _VELOCITY_CHECK_GAP_SEC
        and .99 * _J3_POSITION_COUNT_RAD <= abs(delta) <= _J3_POSITION_COUNT_RAD
        and abs(velocity) <= 2 * hard_limit
        and math.isclose(velocity, delta / dt, rel_tol=1e-5, abs_tol=1e-9)
    )


def joint_command_snapshot(values: dict[str, tuple[float, ...]] | None) -> dict[str, tuple[float, ...]] | None:
    """Validate an owner-confirmed diagnostic copy of the six-joint command."""
    if values is None:
        return None
    if set(values) != {"gravity_nm", "overlay_nm", "total_nm"}:
        raise ValueError("joint commands require gravity, overlay and total torque")
    result = {key: tuple(_finite(item, key) for item in vector) for key, vector in values.items()}
    if any(len(vector) != 6 for vector in result.values()):
        raise ValueError("joint commands require six values per component")
    return result


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class B601TorqueExperimentLimits:
    joint_name: str
    max_abs_assist_nm: float = 0.30
    rise_slew_nm_s: float = 0.15
    release_slew_nm_s: float = 2.0
    command_timeout_sec: float = 0.35
    stationarity_limit_rad_s: float = 0.02
    max_joint_velocity_rad_s: float = 0.05
    max_excursion_rad: float = 0.15
    max_support_drift_rad: float = 0.008726646259971648

    def __post_init__(self) -> None:
        if self.joint_name not in B601_RS_ARM_JOINT_NAMES:
            raise ValueError("joint_name must be a B601 arm joint")
        for name in (
            "max_abs_assist_nm",
            "rise_slew_nm_s",
            "release_slew_nm_s",
            "command_timeout_sec",
            "stationarity_limit_rad_s",
            "max_joint_velocity_rad_s",
            "max_excursion_rad",
            "max_support_drift_rad",
        ):
            if _finite(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.release_slew_nm_s < self.rise_slew_nm_s:
            raise ValueError("release slew must be at least rise slew")
        if self.max_joint_velocity_rad_s <= self.stationarity_limit_rad_s:
            raise ValueError("joint velocity guard must exceed arming stationarity limit")


@dataclass(frozen=True)
class B601MotionOnsetCaptureConfig:
    """Diagnostic onset criterion, not a friction or external-force observer."""

    displacement_rad: float = 0.0012
    confirm_samples: int = 3
    window_velocity_rad_s: float = 0.002
    settle_timeout_sec: float = 1.0
    # Explicit automatic acquisition only; other axes keep the original guard.
    raw_joint_velocity_limit_rad_s: float | None = None
    release_support_confirmation: bool = False

    def __post_init__(self) -> None:
        for name in ("displacement_rad", "window_velocity_rad_s", "settle_timeout_sec"):
            if _finite(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.displacement_rad < 3 * _J3_POSITION_COUNT_RAD:
            raise ValueError("onset displacement must cover at least three position counts")
        if isinstance(self.confirm_samples, bool) or not isinstance(self.confirm_samples, int) or self.confirm_samples < 3:
            raise ValueError("onset confirmation requires at least three samples")
        if self.settle_timeout_sec < _VELOCITY_CONFIRMATION_TIMEOUT_SEC:
            raise ValueError("onset settle timeout must allow final zero and a full window")
        if self.raw_joint_velocity_limit_rad_s is not None:
            value = _finite(self.raw_joint_velocity_limit_rad_s, "onset raw velocity limit")
            if not 0.05 <= value <= 0.15:
                raise ValueError("onset raw velocity limit must be within 0.05..0.15 rad/s")
            object.__setattr__(self, "raw_joint_velocity_limit_rad_s", value)
        if type(self.release_support_confirmation) is not bool:
            raise TypeError("release support confirmation must be boolean")
        if self.release_support_confirmation and self.raw_joint_velocity_limit_rad_s is None:
            raise ValueError("release support confirmation requires explicit acquisition velocity limits")


@dataclass(frozen=True)
class B601OnsetProbeConfig:
    """Finite observation of a held software overlay, not measured force."""

    hold_sec: float = 0.12
    max_trial_sec: float = 10.0
    max_command_exposure_nm_sec: float = 4.0
    max_displacement_rad: float | None = None
    observation_timeout_sec: float | None = None

    def __post_init__(self) -> None:
        for name in ("hold_sec", "max_trial_sec", "max_command_exposure_nm_sec"):
            if _finite(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hold_sec < _VELOCITY_CHECK_WINDOW_SEC:
            raise ValueError("probe hold must cover a full observation window")
        if self.max_trial_sec <= self.hold_sec + 2 * _VELOCITY_CHECK_GAP_SEC:
            raise ValueError("probe trial must allow hold dispatch, observation and withdrawal")
        if self.max_displacement_rad is not None and _finite(self.max_displacement_rad, "max_displacement_rad") < 3 * _J3_POSITION_COUNT_RAD:
            raise ValueError("probe displacement bound must cover at least three position counts")
        if self.observation_timeout_sec is not None:
            timeout = _finite(self.observation_timeout_sec, "observation_timeout_sec")
            if not self.hold_sec + _VELOCITY_CHECK_WINDOW_SEC + _VELOCITY_CHECK_GAP_SEC <= timeout < self.max_trial_sec - _VELOCITY_CHECK_GAP_SEC:
                raise ValueError("observation timeout must cover quiet confirmation and fit the trial deadline")
            object.__setattr__(self, "observation_timeout_sec", timeout)

    @property
    def observation_limit_sec(self) -> float:
        # Legacy defaults stay exact; an explicit margin accommodates the
        # final quiet window without renewing the original trial clock.
        return (self.hold_sec + _VELOCITY_CHECK_WINDOW_SEC
                if self.observation_timeout_sec is None else self.observation_timeout_sec)


class B601TorqueExperiment:
    """Thread-safe intent mailbox with owner-cycle-only state transitions.

    HTTP handlers may submit intent but cannot arm a motor or dispatch torque.
    Arming is completed only inside ``step`` with fresh measured state.
    """

    def __init__(
        self, limits: B601TorqueExperimentLimits, *, quantized_velocity_debug: bool = False,
        onset_capture: B601MotionOnsetCaptureConfig | None = None,
        onset_probe: B601OnsetProbeConfig | None = None,
    ) -> None:
        if not isinstance(quantized_velocity_debug, bool):
            raise TypeError("quantized_velocity_debug must be boolean")
        # Acquisition has a separate onset speed guard; retain the original
        # ceiling for ordinary quantized-velocity commissioning.
        velocity_ceiling = 0.05
        if isinstance(onset_capture, B601MotionOnsetCaptureConfig) and onset_capture.raw_joint_velocity_limit_rad_s is not None:
            velocity_ceiling = 0.13 if onset_capture.release_support_confirmation else 0.07
        if quantized_velocity_debug and (
            limits.joint_name != "joint3" or limits.max_joint_velocity_rad_s > velocity_ceiling
        ):
            raise ValueError(f"quantized velocity debug requires J3 and a hard limit at most {velocity_ceiling} rad/s")
        if onset_capture is not None:
            if not isinstance(onset_capture, B601MotionOnsetCaptureConfig) or not quantized_velocity_debug:
                raise ValueError("onset capture requires explicit J3 quantized velocity debug")
            if onset_capture.displacement_rad >= limits.max_excursion_rad or onset_capture.window_velocity_rad_s >= limits.stationarity_limit_rad_s:
                raise ValueError("onset criterion must lie within excursion and stationarity bounds")
        self.limits = limits
        self.onset_capture = onset_capture
        if onset_capture is not None and onset_capture.release_support_confirmation and onset_probe is None:
            raise ValueError("release support confirmation requires explicit onset probe")
        if onset_probe is not None:
            if not isinstance(onset_probe, B601OnsetProbeConfig) or onset_capture is None:
                raise ValueError("onset probe requires explicit onset capture")
            if onset_probe.max_command_exposure_nm_sec <= 2 * _VELOCITY_CHECK_GAP_SEC * limits.max_abs_assist_nm:
                raise ValueError("probe exposure must leave a full dispatch/withdrawal reserve")
            bound = onset_probe.max_displacement_rad if onset_probe.max_displacement_rad is not None else onset_capture.displacement_rad
            if not onset_capture.displacement_rad <= bound < limits.max_excursion_rad:
                raise ValueError("probe displacement bound must cover onset and lie inside the hard excursion limit")
        self.onset_probe = onset_probe
        self._human_intervention: bool | None = None
        self._probe: dict[str, object] | None = None
        self._probe_frontier_rad = 0.0
        self._probe_reason = "not_started"
        self._probe_exposure_nm_sec = 0.0
        self._probe_last_completed_sec: float | None = None
        self._probe_last_delta_nm = 0.0
        self._lock = threading.Lock()
        self._joint_name = limits.joint_name
        self._phase: Literal["inactive", "arm_requested", "armed", "velocity_check", "onset_release", "fault"] = "inactive"
        self._ready = False
        self._desired_nm = 0.0
        self._effective_nm = 0.0
        self._last_intent_sec: float | None = None
        self._last_step_sec: float | None = None
        self._anchor_rad: float | None = None
        self._anchor_qpos: tuple[float, ...] | None = None
        self._fault: str | None = None
        self._last_qpos: tuple[float, ...] | None = None
        self._last_qvel: tuple[float, ...] | None = None
        self._last_dispatch: dict[str, object] | None = None
        self._dispatch_count = 0
        self._velocity_estimators = (
            tuple(B601WindowVelocityEstimator(_VELOCITY_CHECK_WINDOW_SEC) for _ in range(6))
            if quantized_velocity_debug else None
        )
        self._window_velocity = _ZERO_SIX
        self._velocity_check_since: float | None = None
        self._velocity_check_min_q: tuple[float, ...] | None = None
        self._velocity_check_max_q: tuple[float, ...] | None = None
        self._velocity_check_axes: tuple[int, ...] = (2,)
        self._velocity_first_feedback_sec: float | None = None
        self._velocity_zero_receipt_sec: float | None = None
        self._velocity_first_zero_sec: float | None = None
        self._velocity_receipt_sequence: int | None = None
        self._velocity_reason: str | None = None
        self._release_support_observation: dict[str, object] | None = None
        self._capture_trial: dict[str, object] | None = None
        self._capture_trials: deque[dict[str, object]] = deque(maxlen=64)
        self._capture_sequence = 0
        self._capture_streak = 0
        self._capture_release_sec: float | None = None
        self._capture_first_zero_sec: float | None = None
        self._capture_start_wait_reason: str | None = None
        # The UI receives a short rolling view; the dynamics trace retains
        # the full per-cycle record on disk.
        self._history: deque[dict[str, object]] = deque(maxlen=240)
        self._events: deque[dict[str, object]] = deque(maxlen=100)

    @property
    def joint_index(self) -> int:
        with self._lock:
            return B601_RS_ARM_JOINT_NAMES.index(self._joint_name)

    def owner_ready(self) -> None:
        """Called only after the existing startup handoff has completed."""

        with self._lock:
            if self._phase != "fault":
                self._ready = True

    def owner_snapshot(self, *, observed_at_sec: float) -> dict[str, object]:
        """Small owner-cycle view for batch sequencing, without UI history."""
        now = _finite(observed_at_sec, "observed_at_sec")
        with self._lock:
            return {
                "ready": self._ready, "phase": self._phase, "fault": self._fault,
                "feedback_timestamp_sec": self._last_step_sec,
                "qpos_rad": self._last_qpos, "qvel_rad_s": self._last_qvel,
                "window_velocity_rad_s": self._window_velocity,
                "prepare_gate": self._preparation_gate_reason(now),
                "position_count_bound_rad": _J3_POSITION_COUNT_RAD,
                "onset_start_wait_reason": self._capture_start_wait_reason,
                "desired_nm": self._desired_nm, "effective_nm": self._effective_nm,
                "last_dispatch": None if self._last_dispatch is None else dict(self._last_dispatch),
                "capture": self._capture_snapshot().get("onset_capture"),
                "last_completed_trial": (
                    dict(self._capture_trials[-1]) if self._phase == "inactive" and self._capture_trials else None
                ),
            }

    def set_human_intervention(self, human: bool) -> None:
        """Explicit operator context; never an automatic contact detector."""
        if not isinstance(human, bool):
            raise TypeError("human intervention must be boolean")
        with self._lock:
            if self.onset_probe is None:
                raise ValueError("control context requires explicit onset probe")
            self._human_intervention = human
            if human and self._phase == "arm_requested":
                self._phase = "inactive"
                self._desired_nm = self._effective_nm = 0.0
                self._last_intent_sec = None
                self._probe_reason = "human_cancelled_pending_arm"
            if human and self._phase == "armed":
                self._begin_capture_release(time.monotonic(), "human_intervention")

    def submit(
        self,
        action: str,
        *,
        now_sec: float,
        torque_nm: float | None = None,
        joint_name: str | None = None,
        withdraw_source: str = "operator_button",
    ) -> None:
        now = _finite(now_sec, "now_sec")
        if now < 0:
            raise ValueError("now_sec must be non-negative")
        if action == "withdraw" and withdraw_source not in {
            "operator_button", "page_hidden", "page_unload", "owner_cancel"
        }:
            raise ValueError("unsupported withdrawal source")
        with self._lock:
            if action == "select_joint":
                if joint_name not in B601_RS_ARM_JOINT_NAMES:
                    raise ValueError("joint_name must be a B601 arm joint")
                if self._velocity_estimators is not None and joint_name != "joint3":
                    raise ValueError("quantized velocity debug is J3 only")
                if self._phase != "inactive" or self._effective_nm != 0.0:
                    raise ValueError("joint selection requires inactive, zero-assist state")
                self._joint_name = joint_name
                self._desired_nm = 0.0
                self._anchor_rad = None
                self._anchor_qpos = None
                self._last_dispatch = None
                self._history.clear()
            elif action == "arm":
                if self.onset_probe is not None and self._human_intervention is not False:
                    raise ValueError("probe requires an explicit no-human-intervention declaration")
                if not self._ready:
                    raise ValueError("arm requires active gravity owner")
                if self._phase != "inactive":
                    raise ValueError("arm requires inactive state")
                if self._effective_nm != 0.0:
                    raise ValueError("arm requires completed torque withdrawal")
                self._desired_nm = 0.0
                self._fault = None
                self._phase = "arm_requested"
                self._last_intent_sec = now
            elif action == "set":
                if self._phase != "armed":
                    raise ValueError("set requires armed state")
                if torque_nm is None:
                    raise ValueError("set requires torque_nm")
                requested = _finite(torque_nm, "torque_nm")
                if abs(requested) > self.limits.max_abs_assist_nm:
                    raise ValueError("torque_nm exceeds configured assist limit")
                self._desired_nm = requested
                self._last_intent_sec = now
                if self.onset_capture is not None and requested == 0.0:
                    self._begin_capture_release(now, "operator_zero")
            elif action == "heartbeat":
                if self._phase != "armed":
                    raise ValueError("heartbeat requires armed state")
                self._last_intent_sec = now
            elif action in ("zero", "withdraw"):
                self._desired_nm = 0.0
                self._last_intent_sec = now
                if self.onset_capture is not None and self._phase == "armed":
                    reason = (
                        "operator_zero" if action == "zero" else {
                            "operator_button": "operator_withdraw",
                            "page_hidden": "page_hidden",
                            "page_unload": "page_unload",
                            "owner_cancel": "owner_cancel",
                        }[withdraw_source]
                    )
                    self._begin_capture_release(now, reason)
                if action == "withdraw":
                    if self._phase not in {"fault", "velocity_check", "onset_release"}:
                        self._phase = "inactive"
                    if self._phase not in {"velocity_check", "onset_release"}:
                        self._anchor_rad = None
                        self._anchor_qpos = None
            else:
                raise ValueError("unsupported torque-console action")
            if action != "heartbeat":
                event = {
                    "time_sec": now,
                    "action": action,
                    "joint_name": self._joint_name,
                    "requested_nm": self._desired_nm,
                }
                if action == "withdraw":
                    event["source"] = withdraw_source
                self._events.append(event)

    def _velocity_debug_snapshot(self, now_sec: float) -> dict[str, object]:
        if self._velocity_estimators is None:
            return {}
        return {
            "velocity_window": {
                "enabled": True,
                "ready": all(e.ready for e in self._velocity_estimators),
                "span_sec": min(e.span_sec for e in self._velocity_estimators),
                "position_span_rad": tuple(e.position_span_rad for e in self._velocity_estimators),
            },
            "control_qvel_rad_s": self._window_velocity,
            "quantized_velocity_debug": {
                "reason": self._velocity_reason,
                "pending": self._phase == "velocity_check",
                "started_at_sec": self._velocity_check_since,
                "zero_dispatch_confirmed": self._velocity_zero_receipt_sec is not None and (
                    self._phase != "velocity_check" or self._velocity_first_zero_sec is not None
                ),
                "first_zero_completed_at_sec": self._velocity_first_zero_sec,
                "confirmation_first_feedback_at_sec": self._velocity_first_feedback_sec,
                "confirmation_position_span_rad": (
                    None if self._velocity_check_min_q is None
                    else self._velocity_check_max_q[2] - self._velocity_check_min_q[2]
                ),
                **({"withdrawal_axes": tuple(B601_RS_ARM_JOINT_NAMES[i] for i in self._velocity_check_axes),
                    "confirmation_position_span_rad_six": None if self._velocity_check_min_q is None else tuple(
                        high-low for high, low in zip(self._velocity_check_max_q, self._velocity_check_min_q, strict=True)
                    )} if self.onset_capture is not None else {}),
                "position_count_bound_rad": _J3_POSITION_COUNT_RAD,
                "confirmation_window_sec": _VELOCITY_CHECK_WINDOW_SEC,
                "confirmation_timeout_sec": _VELOCITY_CONFIRMATION_TIMEOUT_SEC,
                "dispatch_watchdog_sec": _VELOCITY_CHECK_GAP_SEC,
                "preparation_gate_reason": self._preparation_gate_reason(now_sec),
                "semantics": "withdraw first; window is diagnostic, not measured physical velocity",
            },
        }

    def _preparation_gate_reason(self, now: float) -> str:
        assert self._velocity_estimators is not None
        if self._last_qvel is None or self._last_step_sec is None:
            return "feedback_unavailable"
        if not 0 <= now - self._last_step_sec <= _VELOCITY_CHECK_GAP_SEC:
            return "feedback_not_fresh"
        reason = window_stationarity_gate_reason(
            self._velocity_estimators, self._last_qvel, self._window_velocity,
            soft_limit_rad_s=self.limits.stationarity_limit_rad_s,
            hard_limit_rad_s=self.limits.max_joint_velocity_rad_s,
        )
        if reason != "ready":
            return reason
        if self.onset_capture is not None and any(e.position_span_rad > _J3_POSITION_COUNT_RAD for e in self._velocity_estimators):
            return "onset_capture_position_not_settled"
        if self._velocity_zero_receipt_sec is None or not 0 <= now - self._velocity_zero_receipt_sec <= _VELOCITY_CHECK_GAP_SEC:
            return "zero_dispatch_unconfirmed"
        return "ready"

    def _capture_snapshot(self, *, include_trials: bool = False) -> dict[str, object]:
        if self.onset_capture is None:
            return {}
        trial_qpos = (self._last_qpos if self._capture_trial is None
            else self._capture_trial.get("final_qpos_rad", self._last_qpos))
        return {"onset_capture": {
            "settings": vars(self.onset_capture),
            "trial": None if self._capture_trial is None else dict(self._capture_trial),
            "completed_trials": len(self._capture_trials),
            **({"trials": [dict(trial) for trial in self._capture_trials]} if include_trials else {}),
            "confirmation_streak": self._capture_streak,
            "release_started_at_sec": self._capture_release_sec,
            "first_zero_completed_at_sec": self._capture_first_zero_sec,
            "release_support_observation": None if self._release_support_observation is None else dict(self._release_support_observation),
            "semantics": ("software command at post-hold encoder progress; not output friction measurement"
                          if self.onset_probe is not None else "software command at sustained motion onset; not measured friction torque"),
            **({"probe": {
                "settings": vars(self.onset_probe), "state": None if self._probe is None else dict(self._probe),
                "reason": self._probe_reason, "human_intervention": self._human_intervention,
                "maximum_hold_sec": self.onset_probe.observation_limit_sec,
                "command_exposure_nm_sec": self._probe_exposure_nm_sec,
                "trial_displacement_rad": (None if self._capture_trial is None or trial_qpos is None
                    else trial_qpos[2]-self._capture_trial["baseline_qpos_rad"][2]),
                "cumulative_displacement_limit_rad": self._probe_displacement_limit_rad(),
                "semantics": "held overlay and new position progress; no force/contact sensing",
            }} if self.onset_probe is not None else {}),
        }}

    def _probe_displacement_limit_rad(self) -> float:
        assert self.onset_probe is not None and self.onset_capture is not None
        configured = self.onset_probe.max_displacement_rad
        return self.onset_capture.displacement_rad if configured is None else configured

    def _begin_capture_release(self, now: float, reason: str) -> None:
        # Keep the original arm/support anchors until withdrawal and rest are
        # confirmed. HTTP zero/withdraw cannot reset this observation clock.
        self._phase = "onset_release"
        self._desired_nm = self._effective_nm = 0.0
        self._capture_release_sec = now
        self._capture_first_zero_sec = None
        self._velocity_zero_receipt_sec = None
        if self.onset_probe is not None:
            self._probe_reason = reason
            if self._probe is not None:
                self._probe["stage"] = "released"
        if self._capture_trial is not None:
            self._capture_trial.update(outcome=reason, withdrawal_started_at_sec=now)
        self._events.append({"time_sec": now, "action": "onset_withdraw", "reason": reason, "requested_nm": 0.0})

    def _capture_release_failure(self, now: float) -> str | None:
        if self._capture_release_sec is None:
            return None
        elapsed = now-self._capture_release_sec
        zero_fresh = self._velocity_zero_receipt_sec is not None and 0 <= now-self._velocity_zero_receipt_sec <= _VELOCITY_CHECK_GAP_SEC
        if elapsed >= _VELOCITY_CHECK_GAP_SEC and (not zero_fresh or self._capture_first_zero_sec is None):
            return "onset_capture_zero_dispatch_timeout"
        if elapsed > self.onset_capture.settle_timeout_sec:
            return "onset_capture_settle_timeout"
        return None

    def _hard_velocity_limit(self, index: int) -> float:
        if index == 2 and self.onset_capture is not None and self.onset_capture.raw_joint_velocity_limit_rad_s is not None:
            return self.onset_capture.raw_joint_velocity_limit_rad_s
        return self.limits.max_joint_velocity_rad_s

    def _step_onset_capture(self, now: float, q: tuple[float, ...], v: tuple[float, ...], onset_start_permitted: bool | None = None) -> None:
        assert self.onset_capture is not None
        config = self.onset_capture
        if self._phase not in {"armed", "onset_release"}:
            return
        if abs(v[2]) > self._hard_velocity_limit(2):
            self._phase, self._fault = "fault", "onset_capture_raw_velocity_exceeded"
            return
        if self._phase == "armed":
            if self._desired_nm == 0.0:
                self._capture_streak = 0
                return
            if self._capture_trial is None:
                baseline_reason = self._preparation_gate_reason(now)
                if (
                    onset_start_permitted is not None
                    and baseline_reason in _ONSET_BASELINE_TRANSIENT_REASONS
                    and (not onset_start_permitted or baseline_reason != "ready")
                ):
                    self._capture_start_wait_reason = (
                        "continuous_baseline_dwell" if baseline_reason == "ready" else baseline_reason
                    )
                    return
                if baseline_reason != "ready":
                    self._phase, self._fault = "fault", "onset_capture_baseline_"+baseline_reason
                    return
                self._capture_sequence += 1
                self._capture_trial = {
                    "trial_sequence": self._capture_sequence,
                    "started_at_sec": now, "baseline_qpos_rad": q,
                    "baseline_control_qvel_rad_s": self._window_velocity,
                    "baseline_dispatch": None if self._last_dispatch is None else dict(self._last_dispatch),
                    "direction": 1 if self._desired_nm > 0 else -1,
                    "outcome": "observing", "peak_dispatched_overlay_nm": 0.0,
                    "peak_conditioned_delta_nm": 0.0,
                }
            trial = self._capture_trial
            direction = trial["direction"]
            if self._desired_nm * direction <= 0:
                self._begin_capture_release(now, "direction_changed")
                return
            displacement = (q[2] - trial["baseline_qpos_rad"][2]) * direction
            fresh = self._last_dispatch is not None and 0 <= now-self._last_dispatch["completed_at_sec"] <= _VELOCITY_CHECK_GAP_SEC
            matching_command = fresh and self._last_dispatch["overlay_after_profile_nm"] * direction > 0
            if fresh:
                # These first crossings are diagnostics only. A single count
                # can be quantization or preload, never a control onset gate.
                for count in (1, 2, 3):
                    key = f"first_{count}_count_observation"
                    if key not in trial and displacement >= count * .99 * _J3_POSITION_COUNT_RAD:
                        trial[key] = {"feedback_timestamp_sec": now, "displacement_rad": displacement,
                                      "window_velocity_rad_s": self._window_velocity[2],
                                      "dispatch": dict(self._last_dispatch)}
            confirmed = matching_command and all(e.ready for e in self._velocity_estimators) and (
                displacement >= config.displacement_rad
                and self._window_velocity[2] * direction >= config.window_velocity_rad_s
            )
            self._capture_streak = self._capture_streak + 1 if confirmed else 0
            if confirmed and "first_criterion_observation" not in trial:
                trial["first_criterion_observation"] = {
                    "feedback_timestamp_sec": now, "displacement_rad": displacement,
                    "window_velocity_rad_s": self._window_velocity[2], "dispatch": dict(self._last_dispatch),
                }
            if self.onset_probe is not None:
                # A quiet local plateau cannot erase earlier trial progress.
                # Slow staircase motion must stop further force escalation,
                # without claiming sustained motion or measured friction.
                cumulative_limit_reached = abs(displacement) >= self._probe_displacement_limit_rad()
                self._step_probe(now, q, displacement, fresh, matching_command, cumulative_limit_reached)
                if self._phase == "armed" and cumulative_limit_reached:
                    trial.update(cumulative_displacement_at_withdrawal_rad=q[2]-trial["baseline_qpos_rad"][2],
                        cumulative_withdrawal_feedback_at_sec=now,
                        cumulative_withdrawal_dispatch=None if self._last_dispatch is None else dict(self._last_dispatch))
                    self._begin_capture_release(now, "probe_cumulative_displacement_withdrawn")
                return
            if self._capture_streak >= config.confirm_samples:
                trial.update(onset_confirmed_at_sec=now, onset_displacement_rad=displacement,
                             onset_qpos_rad=q, onset_qvel_fd_rad_s=v,
                             onset_window_velocity_rad_s=self._window_velocity[2],
                             onset_dispatch=dict(self._last_dispatch))
                self._begin_capture_release(now, "motion_onset_confirmed")
        else:
            release_failure = self._capture_release_failure(now)
            if release_failure is not None:
                self._phase, self._fault = "fault", release_failure
            elif (self._capture_first_zero_sec is not None
                and now-self._capture_first_zero_sec >= _VELOCITY_CHECK_WINDOW_SEC
                and (self._release_support_observation is None or self._release_support_observation["state"] == "confirmed")
                and self._preparation_gate_reason(now) == "ready"):
                if self._capture_trial is not None:
                    self._capture_trial.update(completed_at_sec=now, final_qpos_rad=q)
                    self._capture_trials.append(dict(self._capture_trial))
                self._events.append({"time_sec": now, "action": "onset_trial_complete", "requested_nm": 0.0})
                self._phase = "inactive"
                self._anchor_rad = None
                self._anchor_qpos = None

    def _step_probe(self, now: float, q: tuple[float, ...], displacement: float, fresh: bool, matching: bool, cumulative_limit_reached: bool) -> None:
        config, trial = self.onset_probe, self._capture_trial
        assert config is not None and trial is not None
        direction = trial["direction"]
        if self._human_intervention is not False:
            self._begin_capture_release(now, "human_intervention_unknown")
            return
        clock_now = max(now, self._probe_last_completed_sec or now)
        projected = self._probe_exposure_nm_sec + abs(self._probe_last_delta_nm) * max(0., clock_now-(self._probe_last_completed_sec or clock_now))
        reserve = 2 * _VELOCITY_CHECK_GAP_SEC * self.limits.max_abs_assist_nm
        if clock_now-trial["started_at_sec"] >= config.max_trial_sec or projected + reserve >= config.max_command_exposure_nm_sec:
            self._begin_capture_release(clock_now, "probe_trial_budget_exhausted")
            return
        self._capture_streak = 0  # Old overlapping-window counts are diagnostic only in this opt-in mode.
        if not fresh:
            if self._probe is not None and 0 < self._last_dispatch["completed_at_sec"]-now <= _VELOCITY_CHECK_GAP_SEC:
                # Already-held commands may encounter buffered feedback, but
                # it cannot seed or confirm progress, or renew any clock.
                # Remember its high-water mark so a repeated cached position
                # cannot later masquerade as a new post-hold advance.
                observe_held_progress(self._probe, now=now, q=q[2], direction=direction, post_hold=False)
                self._probe_reason = "waiting_post_dispatch_feedback"
                if clock_now-self._probe["entered_at_sec"] >= config.hold_sec + 2 * _VELOCITY_CHECK_GAP_SEC:
                    self._begin_capture_release(clock_now, "probe_inconclusive")
                return
            self._phase, self._fault = "fault", "probe_dispatch_not_fresh"
            return
        if self._probe is None:
            if cumulative_limit_reached:
                return  # The caller withdraws; do not create another hold.
            if not matching or displacement < self._probe_frontier_rad + .99 * _J3_POSITION_COUNT_RAD:
                return
            delta = self._last_dispatch["conditioned_total_nm"]-self._last_dispatch["gravity_nm"]
            if delta * direction <= 0:
                self._begin_capture_release(now, "probe_overlay_unconfirmed")
                return
            hold = direction * min(abs(delta), abs(self._last_dispatch["overlay_after_profile_nm"]), abs(self._effective_nm), abs(self._desired_nm))
            self._probe_frontier_rad = displacement
            self._probe = begin_held_progress(now, q[2], hold)
            trial["probe_count"] = trial.get("probe_count", 0) + 1
            self._probe_reason = "candidate_hold_pending_dispatch"
            self._events.append({"time_sec": now, "action": "probe_hold", "requested_nm": hold})
            return
        probe = self._probe
        completed = probe["first_hold_completed_at_sec"]
        progress = observe_held_progress(probe, now=now, q=q[2], direction=direction,
            post_hold=False if completed is None or now <= completed else None)
        if completed is None or now <= completed:
            # Completion-time equality is still pre-hold evidence. Remember
            # known progress without counting it as a post-hold advance.
            probe["last_counted_progress_rad"] = max(probe["last_counted_progress_rad"], progress)
        if completed is None:
            if progress >= 2 * .99 * _J3_POSITION_COUNT_RAD:
                self._begin_capture_release(now, "probe_motion_before_hold_confirmation")
            elif now-probe["entered_at_sec"] >= _VELOCITY_CHECK_GAP_SEC:
                self._phase, self._fault = "fault", "probe_hold_dispatch_timeout"
            return
        if now <= completed:
            return  # Buffered feedback is not post-hold evidence.
        # A late quiet sample cannot renew an expired held-force observation.
        # Equality consumes the deadline; only earlier feedback may resume.
        if now-completed >= config.observation_limit_sec:
            self._begin_capture_release(now, "probe_inconclusive")
            return
        # Only now is this feedback causally after a held-command receipt.
        observe_held_progress(probe, now=now, q=q[2], direction=direction, post_hold=True)
        if progress < -2 * .99 * _J3_POSITION_COUNT_RAD:
            self._begin_capture_release(now, "probe_opposing_motion")
            return
        self._capture_streak = probe["distinct_progress_samples"]
        if abs(self._window_velocity[2]) >= self.limits.stationarity_limit_rad_s:
            self._begin_capture_release(now, "probe_soft_motion_withdrawn")
            return
        if self._capture_streak >= self.onset_capture.confirm_samples and displacement >= self.onset_capture.displacement_rad:
            trial.update(onset_confirmed_at_sec=now, onset_displacement_rad=displacement,
                onset_qpos_rad=q, onset_qvel_fd_rad_s=self._last_qvel, onset_window_velocity_rad_s=self._window_velocity[2],
                onset_dispatch=dict(self._last_dispatch), onset_confirmation_method="post_hold_distinct_progress")
            self._begin_capture_release(now, "motion_onset_confirmed")
            return
        if cumulative_limit_reached:
            return  # No confirmed onset; the caller withdraws before resume.
        self._probe_reason = "observing_new_feedback_at_held_overlay"
        if now-completed >= config.hold_sec:
            # Past pre-slip does not imply current motion. Require a full
            # causal window without a new progress frontier, at held force.
            plateau_since = probe["last_progress_at_sec"] or probe["first_feedback_at_sec"]
            quiet = now-plateau_since >= _VELOCITY_CHECK_WINDOW_SEC and (
                all(e.ready and e.position_span_rad <= _J3_POSITION_COUNT_RAD
                        and abs(v) <= self.limits.stationarity_limit_rad_s
                        for e,v in zip(self._velocity_estimators,self._window_velocity,strict=True))
            )
            if quiet:
                trial["last_probe"] = {**probe, "ended_at_sec": now, "outcome": "quiet_resume"}
                # Do not re-observe the same plateau as a fresh candidate.
                frontier = direction * (probe["candidate_qpos_rad"]-trial["baseline_qpos_rad"][2]) + probe["maximum_progress_rad"]
                self._probe_frontier_rad = max(self._probe_frontier_rad, frontier)
                self._probe_reason = "quiet_resume_from_held_overlay"
                self._probe = None
                self._events.append({"time_sec": now, "action": "probe_resume", "requested_nm": self._desired_nm})
            else:
                self._probe_reason = "waiting_recent_plateau"

    def _begin_velocity_check(self, now: float, dt: float, q: tuple[float, ...], v: tuple[float, ...], previous_q: tuple[float, ...], axes: tuple[int, ...]) -> None:
        self._velocity_check_axes = axes
        self._phase = "velocity_check"
        self._desired_nm = self._effective_nm = 0.0
        self._last_intent_sec = None
        self._velocity_check_since = now
        self._velocity_check_min_q = self._velocity_check_max_q = None
        self._velocity_first_feedback_sec = None
        # Retain the last actual zero receipt for any already-running capture
        # release deadline. The new diagnostic still requires its OWN first
        # post-spike zero receipt before declaring confirmation.
        if self._capture_release_sec is None:
            self._velocity_zero_receipt_sec = None
        self._velocity_first_zero_sec = None
        self._velocity_reason = "single_count_speed_spike_withdrawn"
        if self._capture_trial is not None:
            self._capture_trial.update(outcome="inconclusive_velocity_spike", ended_at_sec=now)
        self._events.append({"time_sec": now, "action": "velocity_check", "requested_nm": 0.0,
                             "raw_qvel_rad_s": v[2], "delta_rad": q[2]-previous_q[2], "feedback_dt_sec": dt,
                             **({"withdrawal_axes": tuple(B601_RS_ARM_JOINT_NAMES[i] for i in axes),
                                 "raw_qvel_rad_s_six": v,
                                 "delta_rad_six": tuple(a-b for a,b in zip(q,previous_q,strict=True))} if self.onset_capture is not None else {})})

    def _observe_release_support(self, now: float, dt: float, q: tuple[float, ...], v: tuple[float, ...], previous_q: tuple[float, ...], axes: tuple[int, ...]) -> bool:
        """One post-zero micro-motion episode; never a propulsion permission.

        Input v may use paired per-joint receive clocks, not the snapshot dt.
        Bound its magnitude, sign and effective interval along with the actual
        position counts. The owner independently admits that paired timebase.
        """
        observation = self._release_support_observation
        commands = None if self._last_dispatch is None else self._last_dispatch.get("joint_commands")
        zero_verified = (
            self._capture_first_zero_sec is not None and self._velocity_zero_receipt_sec is not None
            and 0 <= now-self._velocity_zero_receipt_sec <= _VELOCITY_CHECK_GAP_SEC
            and self._last_dispatch is not None and self._last_dispatch["completed_at_sec"] < now
            and commands is not None and all(
                overlay == 0.0 and abs(total-gravity) <= 1e-9
                for overlay,total,gravity in zip(commands["overlay_nm"], commands["total_nm"], commands["gravity_nm"], strict=True)
            ) and self._desired_nm == self._effective_nm == 0.0
        )
        if observation is None and (self._capture_trial is None or not axes or not zero_verified or self._human_intervention is not False):
            return False
        count_limit = 2 * _J3_POSITION_COUNT_RAD
        qualified = not any(i == 2 for i in axes) and 0 < dt <= _VELOCITY_CHECK_GAP_SEC
        for i in axes:
            delta = q[i]-previous_q[i]
            counts = round(abs(delta)/_J3_POSITION_COUNT_RAD)
            qualified = qualified and (
                counts in (1, 2) and .99*counts*_J3_POSITION_COUNT_RAD <= abs(delta) <= counts*_J3_POSITION_COUNT_RAD
                and 0 < abs(v[i]) <= min(2*self._hard_velocity_limit(i), _RELEASE_SUPPORT_RAW_SPEED_CEILING_RAD_S)
                and delta*v[i] > 0 and 0 < delta/v[i] <= _VELOCITY_CHECK_GAP_SEC
            )
        if observation is None:
            if not qualified:
                return False  # Existing raw guard still owns every other shape.
            observation = self._release_support_observation = {
                "state": "observing", "severity": "warning", "action": "keep_zero_and_damping",
                "started_at_sec": now, "deadline_sec": now+_VELOCITY_CONFIRMATION_TIMEOUT_SEC,
                "axes": tuple(B601_RS_ARM_JOINT_NAMES[i] for i in axes),
                "min_qpos_rad": previous_q, "max_qpos_rad": previous_q,
                "position_range_limit_rad": count_limit,
            }
            self._events.append({"time_sec": now, "action": "release_support_observation", "requested_nm": 0.0})
        reason = None
        if not zero_verified:
            reason = "onset_release_support_zero_dispatch_unconfirmed"
        elif not qualified:
            reason = "onset_release_support_raw_speed_exceeded"
        elif observation["state"] == "confirmed" and axes:
            reason = "onset_release_support_repeated_raw_violation"
        observation["min_qpos_rad"] = tuple(min(a,b) for a,b in zip(observation["min_qpos_rad"],q,strict=True))
        observation["max_qpos_rad"] = tuple(max(a,b) for a,b in zip(observation["max_qpos_rad"],q,strict=True))
        if reason is None and any(observation["max_qpos_rad"][i]-observation["min_qpos_rad"][i] > count_limit+1e-12 for i in range(6) if i != 2):
            reason = "onset_release_support_continued_motion"
        if reason is None and observation["state"] == "observing":
            # Completion is a fresh full window AFTER the event, never the
            # pre-withdrawal history. Repeated crossings cannot renew deadlines.
            if now >= observation["deadline_sec"]-1e-9:
                reason = "onset_release_support_observation_timeout"
            elif now-observation["started_at_sec"] >= _VELOCITY_CHECK_WINDOW_SEC and all(
                e.ready and e.position_span_rad <= _J3_POSITION_COUNT_RAD
                and abs(speed) <= self.limits.stationarity_limit_rad_s
                for i,(e,speed) in enumerate(zip(self._velocity_estimators,self._window_velocity,strict=True)) if i != 2
            ):
                observation.update(state="confirmed", confirmed_at_sec=now)
        if reason is not None:
            observation.update(state="fault", severity="error", reason=reason)
            self._phase, self._fault = "fault", reason
        if self._capture_trial is not None:
            self._capture_trial["release_support_observation"] = dict(observation)
        return True

    def _capture_raw_guard(self, now: float, dt: float, q: tuple[float, ...], v: tuple[float, ...], previous_q: tuple[float, ...]) -> None:
        axes = tuple(i for i, raw in enumerate(v) if abs(raw) > self._hard_velocity_limit(i))
        if self._phase == "onset_release" and self.onset_capture.release_support_confirmation:
            release_failure = self._capture_release_failure(now)
            if release_failure is not None:
                self._phase, self._fault = "fault", release_failure
                return
            if self._observe_release_support(now, dt, q, v, previous_q, axes):
                return
        if not axes:
            return
        release_failure = self._capture_release_failure(now)
        if release_failure is not None:
            self._phase, self._fault = "fault", release_failure
            return
        # Classify ALL axes before withdrawing. One qualified spike must never
        # mask a simultaneous multi-count or otherwise invalid hard fault.
        quiet = all(e.ready and e.position_span_rad <= _J3_POSITION_COUNT_RAD
                    and abs(speed) <= self.limits.stationarity_limit_rad_s
                    for e,speed in zip(self._velocity_estimators,self._window_velocity,strict=True))
        rejected = tuple(i for i in axes if not _bounded_count_sample(
            dt, q[i]-previous_q[i], v[i], self._hard_velocity_limit(i),
        ))
        if not rejected and self._phase != "velocity_check" and not quiet:
            rejected = axes
        if rejected:
            # Keep support fault priority, but report an actually invalid
            # sample before rejecting other axes for the combined window.
            i = min(rejected, key=lambda axis: axis == 2)
            prefix = "onset_capture_raw_velocity_exceeded"
            if i == 2 and self._phase == "armed":
                prefix = "joint_velocity_exceeded"
            elif i == 2 and self._phase == "velocity_check":
                prefix = "velocity_check_raw_speed_exceeded"
            self._phase = "fault"
            self._fault = f"{prefix}: {B601_RS_ARM_JOINT_NAMES[i]} abs_qvel={abs(v[i]):.6f}rad/s limit={self._hard_velocity_limit(i):.6f}rad/s"
            return
        if self._phase != "velocity_check":
            self._begin_velocity_check(now, dt, q, v, previous_q, axes)

    def _withdraw_single_count_spike(self, now: float, dt: float, q: tuple[float, ...], v: tuple[float, ...], previous_q: tuple[float, ...]) -> bool:
        if self._velocity_estimators is None or not 0 < dt <= _VELOCITY_CHECK_GAP_SEC:
            return False
        estimator = self._velocity_estimators[2]
        if not (
            all(e.ready for e in self._velocity_estimators)
            # Allow decoding roundoff, not arbitrary sub-count motion from a
            # different position source with finer resolution.
            and _bounded_count_sample(dt,q[2]-previous_q[2],v[2],self._hard_velocity_limit(2))
            and estimator.position_span_rad <= _J3_POSITION_COUNT_RAD
            and abs(self._window_velocity[2]) <= self.limits.stationarity_limit_rad_s
        ):
            return False
        self._begin_velocity_check(now,dt,q,v,previous_q,(2,))
        return True

    def _check_zero_velocity_withdrawal(self, now: float, dt: float, q: tuple[float, ...], v: tuple[float, ...], previous_q: tuple[float, ...]) -> None:
        assert self._velocity_check_since is not None
        # Do not mix the pre-withdrawal spike with post-zero confirmation.
        # Keep the original total-excursion/support guards and raw speed
        # checks active; seed this range only with actual post-zero feedback.
        if self._velocity_first_zero_sec is not None and now > self._velocity_first_zero_sec:
            if self._velocity_first_feedback_sec is None:
                self._velocity_first_feedback_sec = now
                self._velocity_check_min_q = self._velocity_check_max_q = q
            else:
                self._velocity_check_min_q = tuple(min(a,b) for a,b in zip(self._velocity_check_min_q,q,strict=True))
                self._velocity_check_max_q = tuple(max(a,b) for a,b in zip(self._velocity_check_max_q,q,strict=True))
        elapsed = now - self._velocity_check_since
        zero_fresh = (
            self._velocity_first_zero_sec is not None
            and self._velocity_zero_receipt_sec is not None
            and 0 <= now - self._velocity_zero_receipt_sec <= _VELOCITY_CHECK_GAP_SEC
        )
        reason = None
        capture_failure = self._capture_release_failure(now) if self.onset_capture is not None else None
        if abs(v[2]) > self._hard_velocity_limit(2) and not _bounded_count_sample(dt,q[2]-previous_q[2],v[2],self._hard_velocity_limit(2)):
            reason = "velocity_check_raw_speed_exceeded"
        elif self._velocity_check_min_q is not None and (
            any(self._velocity_check_max_q[i]-self._velocity_check_min_q[i] > _J3_POSITION_COUNT_RAD
                for i in (range(6) if self.onset_capture is not None else (2,)))
        ):
            reason = "velocity_check_continued_motion"
        elif self._velocity_estimators[2].ready and abs(self._window_velocity[2]) > self._hard_velocity_limit(2):
            reason = "velocity_check_window_speed_exceeded"
        elif elapsed >= _VELOCITY_CHECK_GAP_SEC - 1e-9 and not zero_fresh:
            reason = "velocity_check_zero_dispatch_timeout"
        elif elapsed > _VELOCITY_CONFIRMATION_TIMEOUT_SEC + 1e-9:
            reason = "velocity_check_confirmation_timeout"
        elif capture_failure is not None:
            reason = capture_failure
        elif zero_fresh and now - self._velocity_first_zero_sec >= _VELOCITY_CHECK_WINDOW_SEC - 1e-9 and all(
            e.ready and abs(velocity) <= self.limits.stationarity_limit_rad_s
            for e, velocity in zip(self._velocity_estimators, self._window_velocity, strict=True)
        ):
            self._phase = "inactive"
            self._anchor_rad = None
            self._anchor_qpos = None
            self._velocity_reason = "zero_confirmed_manual_rearm_required"
            if self._capture_trial is not None:
                self._capture_trial.update(completed_at_sec=now, final_qpos_rad=q)
                self._capture_trials.append(dict(self._capture_trial))
            self._events.append({"time_sec": now, "action": "velocity_check_complete", "requested_nm": 0.0})
        elif elapsed >= _VELOCITY_CONFIRMATION_TIMEOUT_SEC - 1e-9:
            reason = "velocity_check_confirmation_timeout"
        if reason is not None:
            self._phase = "fault"
            self._fault = self._velocity_reason = reason

    def step(
        self,
        *,
        now_sec: float,
        qpos_rad: tuple[float, ...],
        qvel_rad_s: tuple[float, ...],
        onset_start_permitted: bool | None = None,
    ) -> tuple[tuple[float, ...], dict[str, object]]:
        # None preserves the manual fail-closed contract. Automatic admission
        # may only delay a new trial; current feedback still owns its gate.
        if onset_start_permitted is not None and (
            type(onset_start_permitted) is not bool or self.onset_capture is None
        ):
            raise ValueError("onset start permission requires boolean and onset capture")
        now = _finite(now_sec, "now_sec")
        q = tuple(_finite(value, "qpos_rad") for value in qpos_rad)
        v = tuple(_finite(value, "qvel_rad_s") for value in qvel_rad_s)
        if len(q) != 6 or len(v) != 6:
            raise ValueError("feedback must contain six joints")
        with self._lock:
            self._capture_start_wait_reason = None
            index = B601_RS_ARM_JOINT_NAMES.index(self._joint_name)
            if self._last_step_sec is not None and now < self._last_step_sec:
                raise ValueError("non-monotonic owner step timestamp")
            dt = 0.0 if self._last_step_sec is None else now - self._last_step_sec
            previous_q = q if self._last_qpos is None else self._last_qpos
            if self._velocity_estimators is not None:
                if self._last_step_sec is not None and dt <= 0:
                    raise ValueError("debug feedback timestamps must advance")
                if dt > _VELOCITY_CHECK_GAP_SEC:
                    for estimator in self._velocity_estimators:
                        estimator.reset()
                    if self._phase in {"armed", "arm_requested", "velocity_check", "onset_release"}:
                        self._phase = "fault"
                        self._desired_nm = 0.0
                        self._fault = "velocity_debug_feedback_gap_exceeded"
                # A buffered snapshot may predate the already completed
                # zero dispatch. It still feeds the original safety guards,
                # but cannot seed either post-zero observation statistic.
                if not (
                    (self._phase == "velocity_check" and self._velocity_first_zero_sec is not None
                     and now <= self._velocity_first_zero_sec)
                    or (self._phase == "onset_release" and self._capture_first_zero_sec is not None
                        and now <= self._capture_first_zero_sec)
                ):
                    self._window_velocity = tuple(
                        e.update(timestamp_sec=now, qpos_rad=position, fallback_rad_s=velocity)
                        for e, position, velocity in zip(self._velocity_estimators, q, v, strict=True)
                    )
            self._last_step_sec = now
            self._last_qpos, self._last_qvel = q, v
            if self.onset_capture is not None and self._phase in {"armed", "velocity_check", "onset_release"}:
                self._capture_raw_guard(now,dt,q,v,previous_q)
            if (
                onset_start_permitted is not None and self._phase == "arm_requested"
                and max(abs(value) for value in v) > self.limits.max_joint_velocity_rad_s
            ):
                # A cold readiness window cannot hide a raw hard violation.
                self._phase = "fault"
                self._fault = "onset_capture_baseline_raw_velocity_hard_guard"
            if self._phase == "arm_requested":
                preparation_reason = (
                    self._preparation_gate_reason(now) if self._velocity_estimators is not None
                    else "ready" if max(abs(value) for value in v) <= self.limits.stationarity_limit_rad_s
                    else "not_stationary"
                )
                if preparation_reason == "ready" and onset_start_permitted is not False:
                    self._phase = "armed"
                    self._anchor_rad = q[index]
                    self._anchor_qpos = q
                    self._velocity_check_since = None
                    self._velocity_reason = None
                    self._capture_trial = None
                    self._capture_streak = 0
                    self._capture_release_sec = self._capture_first_zero_sec = None
                    self._release_support_observation = None
                    self._probe = None
                    self._probe_frontier_rad = 0.0
                    self._probe_reason = "ramp"
                    self._probe_exposure_nm_sec = 0.0
                    self._probe_last_completed_sec = None
                    self._probe_last_delta_nm = 0.0
                    if onset_start_permitted is not None:
                        self._last_intent_sec = now
                    self._events.append({"time_sec": now, "action": "armed", "joint_name": self._joint_name, "requested_nm": 0.0})
                elif onset_start_permitted is not None and preparation_reason in _ONSET_BASELINE_TRANSIENT_REASONS:
                    self._capture_start_wait_reason = (
                        "continuous_baseline_dwell" if preparation_reason == "ready" else preparation_reason
                    )
                else:
                    self._phase = "inactive"
                    self._fault = f"arm_rejected_{preparation_reason}"
                    if onset_start_permitted is not None:
                        self._phase = "fault"
            if self._phase in {"armed", "velocity_check", "onset_release"} and self._anchor_rad is not None:
                excursion = q[index] - self._anchor_rad
                if (
                    (self._desired_nm != 0.0 or self._effective_nm != 0.0)
                    and abs(v[index]) > self._hard_velocity_limit(index)
                    and not self._withdraw_single_count_spike(now, dt, q, v, previous_q)
                ):
                    self._phase = "fault"
                    self._desired_nm = 0.0
                    self._fault = (
                        f"joint_velocity_exceeded: {self._joint_name} "
                        f"abs_qvel={abs(v[index]):.6f}rad/s "
                        f"limit={self._hard_velocity_limit(index):.6f}rad/s"
                    )
                if self._phase != "fault" and abs(excursion) > self.limits.max_excursion_rad:
                    self._phase = "fault"
                    self._desired_nm = 0.0
                    self._fault = (
                        f"joint_excursion_exceeded: {self._joint_name} "
                        f"delta={excursion:+.6f}rad limit={self.limits.max_excursion_rad:.6f}rad"
                    )
                if self._anchor_qpos is not None and self._phase != "fault":
                    for support_index, anchor in enumerate(self._anchor_qpos):
                        if support_index == index:
                            continue
                        drift = q[support_index] - anchor
                        if abs(drift) > self.limits.max_support_drift_rad:
                            self._phase = "fault"
                            self._desired_nm = 0.0
                            self._fault = (
                                "support_joint_drift_exceeded: "
                                f"{B601_RS_ARM_JOINT_NAMES[support_index]} "
                                f"delta={drift:+.6f}rad "
                                f"limit={self.limits.max_support_drift_rad:.6f}rad"
                            )
                            break
            if self._phase == "velocity_check":
                self._check_zero_velocity_withdrawal(now, dt, q, v, previous_q)
            if self.onset_capture is not None:
                self._step_onset_capture(now, q, v, onset_start_permitted)
            if self._phase == "fault":
                # Fault withdrawal is immediate; the owner aborts this cycle
                # before dispatch and enters its existing protective hold.
                self._desired_nm = self._effective_nm = 0.0
                if self._release_support_observation is not None:
                    self._release_support_observation.update(state="fault", severity="error", reason=self._fault)
                    if self._capture_trial is not None:
                        self._capture_trial["release_support_observation"] = dict(self._release_support_observation)
                if self._capture_trial is not None and "completed_at_sec" not in self._capture_trial:
                    self._capture_trial.update(outcome="guard_fault", failure_reason=self._fault)
            if (
                self._phase == "fault"
                and self._fault is not None
                and (not self._events or self._events[-1].get("action") != "guard_fault")
            ):
                self._events.append(
                    {
                        "time_sec": now,
                        "action": "guard_fault",
                        "joint_name": self._joint_name,
                        "reason": self._fault,
                        "requested_nm": 0.0,
                    }
                )
            age = None if self._last_intent_sec is None else now - self._last_intent_sec
            if self._phase == "armed" and (age is None or age > self.limits.command_timeout_sec):
                if self.onset_capture is not None:
                    self._begin_capture_release(now, "gui_intent_stale")
                else:
                    self._desired_nm = 0.0
                    self._phase = "inactive"
                    self._anchor_rad = None
                    self._anchor_qpos = None
                self._fault = "gui_intent_stale"
                self._events.append({"time_sec": now, "action": "stale_withdraw", "requested_nm": 0.0})
            target = self._desired_nm if self._phase == "armed" else 0.0
            if self._capture_start_wait_reason is not None:
                target = 0.0
            if self.onset_probe is not None and self._phase == "armed" and self._probe is not None:
                target = math.copysign(min(abs(target), abs(self._probe["hold_nm"])), target)
            reversing = target * self._effective_nm < 0.0
            if reversing:
                # Release the old sign before allowing a new signed assist.
                change = max(
                    -self.limits.release_slew_nm_s * dt,
                    min(self.limits.release_slew_nm_s * dt, -self._effective_nm),
                )
            else:
                slope = (
                    self.limits.rise_slew_nm_s
                    if abs(target) > abs(self._effective_nm)
                    else self.limits.release_slew_nm_s
                )
                change = max(-slope * dt, min(slope * dt, target - self._effective_nm))
            self._effective_nm += change
            if abs(self._effective_nm) < 1e-12:
                self._effective_nm = 0.0
            overlay = list(_ZERO_SIX)
            overlay[index] = self._effective_nm
            audit = {
                "time_sec": now,
                "phase": self._phase,
                "owner_ready": self._ready,
                "joint_name": self._joint_name,
                "desired_assist_nm": target,
                "effective_assist_nm": self._effective_nm,
                "qpos_rad": q,
                "qvel_fd_rad_s": v,
                "anchor_rad": self._anchor_rad,
                "displacement_rad": (
                    None if self._anchor_rad is None else q[index] - self._anchor_rad
                ),
                "support_drift_rad": (
                    None
                    if self._anchor_qpos is None
                    else {
                        name: q[i] - self._anchor_qpos[i]
                        for i, name in enumerate(B601_RS_ARM_JOINT_NAMES)
                        if i != index
                    }
                ),
                "intent_age_sec": age,
                "fault": self._fault,
                "onset_start_wait_reason": self._capture_start_wait_reason,
                **self._velocity_debug_snapshot(now),
                **self._capture_snapshot(),
            }
            self._history.append(audit)
            return tuple(overlay), audit

    def validate_final_command(
        self, *, gravity_nm: tuple[float, ...], total_nm: tuple[float, ...],
    ) -> None:
        """Check the opt-in probe envelope before the sole owner sends MIT."""
        if self.onset_probe is None:
            return
        commands = joint_command_snapshot({"gravity_nm": gravity_nm, "overlay_nm": _ZERO_SIX, "total_nm": total_nm})
        with self._lock:
            if self._capture_trial is None or "completed_at_sec" in self._capture_trial:
                return
            index = B601_RS_ARM_JOINT_NAMES.index(self._joint_name)
            delta = commands["total_nm"][index] - commands["gravity_nm"][index]
            if abs(delta) > self.limits.max_abs_assist_nm + 1e-9:
                self._phase, self._fault = "fault", "probe_final_command_envelope_exceeded"
                self._desired_nm = self._effective_nm = 0.0
                raise ValueError("probe final command exceeds assist envelope before dispatch")

    def record_dispatch(
        self,
        *,
        joint_name: str,
        receipt_sequence: int,
        feedback_timestamp_sec: float,
        overlay_after_profile_nm: float,
        gravity_nm: float,
        requested_total_nm: float,
        conditioned_total_nm: float,
        joint_commands: dict[str, tuple[float, ...]] | None = None,
        completed_at_sec: float | None = None,
    ) -> None:
        """Publish only owner-confirmed software dispatch values to the GUI."""

        if isinstance(receipt_sequence, bool) or not isinstance(receipt_sequence, int) or receipt_sequence < 0:
            raise ValueError("receipt_sequence must be non-negative")

        record = {
            "receipt_sequence": int(receipt_sequence),
            "feedback_timestamp_sec": _finite(
                feedback_timestamp_sec, "feedback_timestamp_sec"
            ),
            "overlay_after_profile_nm": _finite(
                overlay_after_profile_nm, "overlay_after_profile_nm"
            ),
            "gravity_nm": _finite(gravity_nm, "gravity_nm"),
            "requested_total_nm": _finite(requested_total_nm, "requested_total_nm"),
            "conditioned_total_nm": _finite(
                conditioned_total_nm, "conditioned_total_nm"
            ),
            "joint_commands": joint_command_snapshot(joint_commands),
        }
        with self._lock:
            # A zero-assist joint switch may arrive between step and receipt.
            # Never display the old joint's receipt as the new joint's output.
            if joint_name != self._joint_name:
                return
            if self._velocity_estimators is not None:
                completed = _finite(completed_at_sec, "completed_at_sec")
                timestamp = record["feedback_timestamp_sec"]
                if timestamp != self._last_step_sec or not timestamp <= completed <= timestamp + _VELOCITY_CHECK_GAP_SEC:
                    raise ValueError("debug dispatch requires matching feedback and timely completion")
                if self._velocity_receipt_sequence is not None and receipt_sequence <= self._velocity_receipt_sequence:
                    raise ValueError("debug dispatch receipt sequence must advance")
                if self._last_dispatch is not None and timestamp == self._last_dispatch["feedback_timestamp_sec"]:
                    raise ValueError("debug feedback cycle already has a dispatch receipt")
                commands = record["joint_commands"]
                if commands is None:
                    raise ValueError("debug dispatch requires complete six-joint command snapshot")
                index = B601_RS_ARM_JOINT_NAMES.index(joint_name)
                if any(abs(record[scalar] - commands[vector][index]) > 1e-9 for scalar, vector in (
                    ("gravity_nm", "gravity_nm"), ("overlay_after_profile_nm", "overlay_nm"),
                    ("conditioned_total_nm", "total_nm"),
                )):
                    raise ValueError("debug scalar receipt must match six-joint snapshot")
                zero = record["overlay_after_profile_nm"] == 0.0 and all(
                    abs(total - gravity) <= 1e-9 and overlay == 0.0
                    for total, gravity, overlay in zip(commands["total_nm"], commands["gravity_nm"], commands["overlay_nm"], strict=True)
                )
                if self._phase == "velocity_check" and self._velocity_first_zero_sec is not None and not zero:
                    self._phase = "fault"
                    self._fault = "velocity_check_nonzero_after_zero"
                    self._velocity_zero_receipt_sec = None
                    raise ValueError("debug withdrawal zero command cannot reappear as nonzero")
                if self._phase in {"onset_release", "velocity_check"} and self._capture_first_zero_sec is not None and not zero:
                    self._phase, self._fault = "fault", "onset_capture_nonzero_after_zero"
                    self._velocity_zero_receipt_sec = None
                    raise ValueError("onset withdrawal zero command cannot reappear as nonzero")
                self._velocity_receipt_sequence = receipt_sequence
                record["completed_at_sec"] = completed
                if self.onset_probe is not None and self._capture_trial is not None and "completed_at_sec" not in self._capture_trial:
                    delta = record["conditioned_total_nm"]-record["gravity_nm"]
                    if self._probe_last_completed_sec is not None and completed < self._probe_last_completed_sec:
                        self._phase, self._fault = "fault", "probe_nonmonotonic_dispatch"
                        self._desired_nm = self._effective_nm = 0.0
                        raise ValueError("probe dispatch completion timestamps must advance")
                    if self._probe_last_completed_sec is not None:
                        self._probe_exposure_nm_sec += abs(self._probe_last_delta_nm) * (completed-self._probe_last_completed_sec)
                    self._probe_last_completed_sec, self._probe_last_delta_nm = completed, delta
                    self._capture_trial["command_exposure_nm_sec"] = self._probe_exposure_nm_sec
                    if abs(delta) > self.limits.max_abs_assist_nm + 1e-9 or self._probe_exposure_nm_sec > self.onset_probe.max_command_exposure_nm_sec:
                        self._phase, self._fault = "fault", "probe_final_command_envelope_exceeded"
                        self._desired_nm = self._effective_nm = 0.0
                        raise ValueError("probe final command exceeds assist/exposure envelope")
                    if self._probe is not None and self._phase == "armed":
                        hold = self._probe["hold_nm"]
                        matches = abs(delta-hold) <= 1e-9 and abs(record["overlay_after_profile_nm"]-hold) <= 1e-9
                        if self._probe["first_hold_completed_at_sec"] is not None and not matches:
                            self._begin_capture_release(completed, "probe_hold_command_changed")
                        elif matches and self._probe["first_hold_completed_at_sec"] is None:
                            self._probe["first_hold_completed_at_sec"] = completed
                self._velocity_zero_receipt_sec = completed if zero else None
                if self._capture_trial is not None:
                    self._capture_trial["peak_dispatched_overlay_nm"] = max(
                        self._capture_trial["peak_dispatched_overlay_nm"], abs(record["overlay_after_profile_nm"]),
                    )
                    self._capture_trial["peak_conditioned_delta_nm"] = max(
                        self._capture_trial["peak_conditioned_delta_nm"], abs(record["conditioned_total_nm"]-record["gravity_nm"]),
                    )
                if zero and self._capture_release_sec is not None and self._phase in {"onset_release", "velocity_check"} and self._capture_first_zero_sec is None and self._capture_release_sec <= completed <= self._capture_release_sec+_VELOCITY_CHECK_GAP_SEC:
                    self._capture_first_zero_sec = completed
                    if self._phase == "onset_release":
                        for estimator in self._velocity_estimators:
                            estimator.reset()
                        self._window_velocity = _ZERO_SIX
                if (
                    zero and self._phase == "velocity_check" and self._velocity_first_zero_sec is None
                    and completed <= self._velocity_check_since + _VELOCITY_CHECK_GAP_SEC
                ):
                    self._velocity_first_zero_sec = completed
                    # Observe only feedback admitted AFTER final zero dispatch,
                    # never reuse the pre-withdrawal window to declare rest.
                    for estimator in self._velocity_estimators:
                        estimator.reset()
                    self._window_velocity = _ZERO_SIX
            self._dispatch_count += 1
            record["joint_name"] = joint_name
            record["dispatch_count"] = self._dispatch_count
            record["recorded_at_sec"] = time.monotonic()
            self._last_dispatch = record
            if self._history and self._history[-1]["time_sec"] == record["feedback_timestamp_sec"]:
                self._history[-1] = {**self._history[-1], **record}

    def status(self) -> dict[str, object]:
        with self._lock:
            # Dispatch completes after feedback. GUI readiness is observed now,
            # not at the earlier feedback timestamp used by owner admission.
            observed_at = time.monotonic()
            return {
                "phase": self._phase,
                "owner_ready": self._ready,
                "joint_name": self._joint_name,
                "limits": vars(self.limits),
                "desired_assist_nm": self._desired_nm,
                "effective_assist_nm": self._effective_nm,
                "fault": self._fault,
                "qpos_rad": self._last_qpos,
                "qvel_fd_rad_s": self._last_qvel,
                "anchor_rad": self._anchor_rad,
                "displacement_rad": (
                    None
                    if self._anchor_rad is None or self._last_qpos is None
                    else self._last_qpos[B601_RS_ARM_JOINT_NAMES.index(self._joint_name)]
                    - self._anchor_rad
                ),
                "last_observed_displacement_rad": (
                    None if not self._history else self._history[-1]["displacement_rad"]
                ),
                "support_drift_rad": (
                    None
                    if self._anchor_qpos is None or self._last_qpos is None
                    else {
                        name: self._last_qpos[i] - self._anchor_qpos[i]
                        for i, name in enumerate(B601_RS_ARM_JOINT_NAMES)
                        if name != self._joint_name
                    }
                ),
                "last_observed_support_drift_rad": (
                    None if not self._history else self._history[-1]["support_drift_rad"]
                ),
                "feedback_age_sec": (
                    None
                    if self._last_step_sec is None
                    else max(0.0, time.monotonic() - self._last_step_sec)
                ),
                "last_dispatch": None if self._last_dispatch is None else dict(self._last_dispatch),
                "dispatch_age_sec": (
                    None
                    if self._last_dispatch is None
                    else max(0.0, time.monotonic() - float(self._last_dispatch["recorded_at_sec"]))
                ),
                "history": [dict(item) for item in self._history],
                "events": [dict(item) for item in self._events],
                **self._velocity_debug_snapshot(observed_at),
                **self._capture_snapshot(include_trials=True),
            }

    def owner_stopped(self, *, now_sec: float, reason: str) -> None:
        """Invalidate all GUI authority before protective-hold/cleanup begins."""

        now = _finite(now_sec, "now_sec")
        with self._lock:
            self._phase = "fault"
            self._ready = False
            self._desired_nm = 0.0
            self._effective_nm = 0.0
            self._anchor_rad = None
            self._anchor_qpos = None
            self._fault = str(reason).strip() or "owner_stopped"
            if self._capture_trial is not None and "completed_at_sec" not in self._capture_trial:
                self._capture_trial.update(outcome="owner_stopped", failure_reason=self._fault)
            self._events.append(
                {"time_sec": now, "action": "owner_stopped", "requested_nm": 0.0}
            )
