"""Opt-in staged single-joint policy assist over the existing B601 runtime.

This module is a pure control component.  It owns no CAN, motor, gravity,
watchdog, or final torque-conditioning path.  In particular, the impulse
accounting below bounds the software overlay request; it is not a measurement
of motor torque, output-shaft torque, or physical energy.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum

from .joint_motion import (
    B601JointMotionConfig,
    B601JointMotionFrame,
    B601JointMotionTracker,
    B601WindowStopEvidence,
    slew_driver_damping,
)
from .mit import B601MitReference
from .model import B601_RS_ARM_JOINT_NAMES
from .shared_control import (
    B601PolicyAssistCommand,
    B601PolicyCommandReceipt,
    B601SharedControlConfig,
    B601SharedControlMode,
)
from .shared_control_runtime import B601SharedControlFrame, B601SharedControlRuntime
from .torque_experiment import (
    J3_HELD_PROGRESS_STEP_RAD,
    B601OnsetProbeConfig,
    begin_held_progress,
    observe_held_progress,
)


class B601StagedAssistPhase(str, Enum):
    PASSIVE = "passive"
    CUE = "cue"
    BREAKAWAY = "breakaway"
    RELEASING = "releasing"
    CAPTURE = "capture"
    DROP = "drop"
    TRACK = "track"
    SETTLE = "settle"
    BRAKE = "brake"
    SOFT_HOLD = "soft_hold"
    HUMAN_OVERRIDE = "human_override"
    AUTHORITY_EXHAUSTED = "authority_exhausted"


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class B601StagedAssistConfig:
    """Explicit one-joint limits; experiment settings are not safety certification."""

    joint_name: str
    cue_torque_nm: float
    cue_duration_sec: float
    cue_rise_slew_nm_s: float
    breakaway_torque_nm: float
    breakaway_rise_slew_nm_s: float
    breakaway_max_duration_sec: float
    breakaway_max_impulse_nm_sec: float
    minimum_breakaway_error_rad: float
    stuck_max_velocity_rad_s: float
    motion_onset_displacement_rad: float
    motion_onset_velocity_rad_s: float
    motion_onset_confirm_samples: int
    new_intent_delta_rad: float
    tracking_overlay_cap_nm: float
    tracking_slew_nm_s: float
    settle_error_rad: float
    settle_capture_error_rad: float
    settle_velocity_rad_s: float
    settle_dwell_sec: float
    settle_integral_gain_nm_per_rad_sec: float
    settle_integral_leak_per_sec: float
    settle_integral_cap_nm: float
    settle_overlay_cap_nm: float
    max_total_overlay_nm: float
    motion_control: B601JointMotionConfig | None = None
    motion_onset_confirm_displacement_rad: float | None = None
    continuous_motion_handoff: bool = False
    capture_overlay_cap_nm: float = 0.0
    breakaway_probe: B601OnsetProbeConfig | None = None
    tracking_recovery_gain_nm_per_rad: float | None = None
    tracking_recovery_velocity_feedback_nm_s_per_rad: float = 0.0
    breakaway_soft_withdrawal_confirmation: bool = False
    onset_handoff: str = "stop_and_zero"
    tracking_release_slew_nm_s: float | None = None

    def __post_init__(self) -> None:
        joint_name = str(self.joint_name).strip()
        if joint_name not in B601_RS_ARM_JOINT_NAMES:
            raise ValueError("joint_name must be one of B601 J1-J6")
        object.__setattr__(self, "joint_name", joint_name)
        for name in (
            "cue_torque_nm",
            "cue_duration_sec",
            "cue_rise_slew_nm_s",
            "breakaway_torque_nm",
            "breakaway_rise_slew_nm_s",
            "breakaway_max_duration_sec",
            "breakaway_max_impulse_nm_sec",
            "minimum_breakaway_error_rad",
            "stuck_max_velocity_rad_s",
            "motion_onset_displacement_rad",
            "motion_onset_velocity_rad_s",
            "new_intent_delta_rad",
            "tracking_overlay_cap_nm",
            "tracking_slew_nm_s",
            "settle_error_rad",
            "settle_capture_error_rad",
            "settle_velocity_rad_s",
            "settle_dwell_sec",
            "settle_integral_gain_nm_per_rad_sec",
            "settle_integral_leak_per_sec",
            "settle_integral_cap_nm",
            "settle_overlay_cap_nm",
            "max_total_overlay_nm",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name=name))
        if self.tracking_release_slew_nm_s is not None:
            rate = _positive(self.tracking_release_slew_nm_s, name="tracking_release_slew_nm_s")
            if (not isinstance(self.motion_control, B601JointMotionConfig) or not self.motion_control.responsive_tracking_brake
                or not self.tracking_slew_nm_s <= rate <= self.motion_control.braking_slew_nm_s):
                raise ValueError("normal tracking release must be between guidance rise and protective braking slew")
        samples = self.motion_onset_confirm_samples
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("motion_onset_confirm_samples must be a positive integer")
        if max(
            self.cue_torque_nm,
            self.breakaway_torque_nm,
            self.tracking_overlay_cap_nm,
            self.settle_overlay_cap_nm,
        ) > self.max_total_overlay_nm:
            raise ValueError("all phase torque caps must be within max_total_overlay_nm")
        if self.settle_overlay_cap_nm > self.tracking_overlay_cap_nm:
            raise ValueError("settle_overlay_cap_nm must not exceed tracking_overlay_cap_nm")
        if self.settle_capture_error_rad <= self.settle_error_rad:
            raise ValueError("settle_capture_error_rad must exceed settle_error_rad")
        if self.settle_integral_cap_nm > self.settle_overlay_cap_nm:
            raise ValueError("settle_integral_cap_nm must not exceed settle_overlay_cap_nm")
        if self.motion_control is not None:
            if not isinstance(self.motion_control, B601JointMotionConfig):
                raise TypeError("motion_control must be B601JointMotionConfig")
            if self.motion_control.arrival_margin_rad >= self.settle_error_rad:
                raise ValueError("arrival margin must be below final stop tolerance")
            if self.motion_control.hold_torque_cap_nm > self.tracking_overlay_cap_nm:
                raise ValueError("hold torque cap exceeds tracking cap")
        if self.motion_onset_confirm_displacement_rad is not None:
            threshold = _positive(self.motion_onset_confirm_displacement_rad, name="motion_onset_confirm_displacement_rad")
            object.__setattr__(self, "motion_onset_confirm_displacement_rad", threshold)
            if threshold < self.motion_onset_displacement_rad:
                raise ValueError("motion confirmation displacement must not precede fast pulse withdrawal")
            if self.motion_control is None or not self.motion_control.windowed_soft_velocity:
                raise ValueError("displacement confirmation requires windowed motion control")
            if self.motion_onset_confirm_samples < 2:
                raise ValueError("displacement confirmation requires at least two samples")
        if self.motion_control is not None and self.motion_control.position_resolution_rad is not None:
            resolution = self.motion_control.position_resolution_rad
            if self.motion_onset_confirm_displacement_rad is None:
                raise ValueError("quantized motion decisions require displacement confirmation")
            if resolution > self.motion_onset_displacement_rad or resolution > self.settle_velocity_rad_s * self.motion_control.velocity_window_sec:
                raise ValueError("position resolution exceeds onset/stop observation envelope")
            if self.motion_control.responsive_tracking_brake and self.motion_control.braking_slew_nm_s < self.tracking_slew_nm_s:
                raise ValueError("responsive withdrawal rate must not be below guidance rise rate")
        if not isinstance(self.continuous_motion_handoff, bool):
            raise TypeError("continuous_motion_handoff must be boolean")
        if self.continuous_motion_handoff and self.motion_onset_confirm_displacement_rad is None:
            raise ValueError("continuous handoff requires explicit windowed displacement confirmation")
        capture = _finite(self.capture_overlay_cap_nm, name="capture_overlay_cap_nm")
        if not 0 <= capture <= min(0.10, self.tracking_overlay_cap_nm):
            raise ValueError("capture cap must be within 0..0.10 Nm and the tracking cap")
        if capture and not self.continuous_motion_handoff:
            raise ValueError("low-force capture requires the offline continuous handoff candidate")
        object.__setattr__(self, "capture_overlay_cap_nm", capture)
        if self.breakaway_probe is not None:
            probe = self.breakaway_probe
            if not isinstance(probe, B601OnsetProbeConfig):
                raise TypeError("breakaway probe must be B601OnsetProbeConfig")
            if self.joint_name not in {"joint2", "joint3"} or self.motion_control is None or self.motion_control.position_resolution_rad is None:
                raise ValueError("held breakaway probe requires explicit resolved J2/J3 motion")
            if self.capture_overlay_cap_nm or self.continuous_motion_handoff:
                raise ValueError("held probe uses the existing stop-and-zero handoff")
            if (probe.max_trial_sec > self.breakaway_max_duration_sec
                or probe.max_command_exposure_nm_sec > self.breakaway_max_impulse_nm_sec
                or max(probe.observation_limit_sec, probe.hold_sec + self.motion_control.velocity_window_sec) >= probe.max_trial_sec
                or probe.max_displacement_rad is None
                or probe.max_displacement_rad <= self.motion_onset_confirm_displacement_rad):
                raise ValueError("held probe must fit the existing pulse budgets and displacement envelope")
        if self.tracking_recovery_gain_nm_per_rad is not None:
            object.__setattr__(self, "tracking_recovery_gain_nm_per_rad", _positive(
                self.tracking_recovery_gain_nm_per_rad, name="tracking recovery gain"))
            if self.breakaway_probe is None or not self.motion_control.responsive_tracking_brake:
                raise ValueError("tracking recovery requires the explicit resolved J2/J3 held candidate")
        feedback = _finite(self.tracking_recovery_velocity_feedback_nm_s_per_rad, name="tracking recovery velocity feedback")
        if feedback < 0 or (feedback and self.tracking_recovery_gain_nm_per_rad is None):
            raise ValueError("velocity feedback requires nonnegative gain and explicit tracking recovery")
        object.__setattr__(self, "tracking_recovery_velocity_feedback_nm_s_per_rad", feedback)
        if type(self.breakaway_soft_withdrawal_confirmation) is not bool:
            raise TypeError("soft withdrawal confirmation must be boolean")
        if self.breakaway_soft_withdrawal_confirmation and (
            self.tracking_recovery_gain_nm_per_rad is None
            or not self.motion_control.reference_pacing
        ):
            raise ValueError("soft withdrawal confirmation requires explicit paced tracking recovery")
        if self.onset_handoff not in {"stop_and_zero", "track"}:
            raise ValueError("onset handoff must be stop_and_zero or track")
        if self.onset_handoff == "track" and (
            self.breakaway_probe is None
            or self.tracking_recovery_gain_nm_per_rad is None
            or not self.motion_control.reference_pacing
            or self.continuous_motion_handoff
        ):
            raise ValueError("tracking handoff requires the resolved held probe and paced tracking recovery")
        if (self.motion_control is not None and self.motion_control.speed_governor
            and self.onset_handoff != "track"):
            raise ValueError("speed governor requires explicit normal tracking handoff")
        if (self.motion_control is not None and self.motion_control.arrival_stabilization
            and self.tracking_overlay_cap_nm / self.motion_control.arrival_unload_slew_nm_s
                + 2*self.settle_dwell_sec >= self.motion_control.braking_timeout_sec):
            raise ValueError("arrival unload and post-zero dwell must fit the braking deadline")

    def breakaway_envelope(self, *, control_period_sec: float) -> dict[str, object]:
        """Optimistic no-motion command bounds, not a dispatch or force claim.

        The pulse starts at zero. Duration is checked before the next request,
        and impulse integrates absolute requests, including ramp samples. Full
        authority and perfect timing are assumed; final conditioning, onset and
        driver saturation can only reduce the achievable command.
        """
        dt = _positive(control_period_sec, name="control_period_sec")
        peak = self.breakaway_torque_nm
        slew = self.breakaway_rise_slew_nm_s
        step_rise = _positive(slew * dt, name="per_tick_breakaway_rise")
        available_ticks = max(0, math.ceil((self.breakaway_max_duration_sec - 1e-9) / dt) - 1)
        peak_tick = max(1, math.ceil(peak / step_rise - 1e-9))
        ramp_impulse = dt * (step_rise * peak_tick * (peak_tick - 1) / 2 + peak)
        # Sum the same discrete ramp used by the runtime, without an unbounded
        # simulation loop. A fractional last budget sample cannot increase a
        # peak already reached on an earlier tick.
        impulse_unit = _positive(step_rise * dt, name="per_tick_impulse_unit")
        budget_ticks = math.floor((math.sqrt(1 + 8 * self.breakaway_max_impulse_nm_sec / impulse_unit) - 1) / 2 + 1e-9)
        full_ticks = min(available_ticks, peak_tick - 1, budget_ticks)
        ceiling = full_ticks * step_rise
        if full_ticks < available_ticks:
            remaining = max(0.0, self.breakaway_max_impulse_nm_sec - impulse_unit * full_ticks * (full_ticks + 1) / 2)
            ceiling = max(ceiling, min(peak, (full_ticks + 1) * step_rise, remaining / dt))
        reasons = []
        if available_ticks < peak_tick:
            reasons.append("duration_limited")
        if self.breakaway_max_impulse_nm_sec + 1e-12 < ramp_impulse:
            reasons.append("impulse_limited")
        return {
            "nominal_peak_nm": peak,
            "optimistic_command_peak_ceiling_nm": ceiling,
            "minimum_rise_time_sec": peak / slew,
            "minimum_continuous_ramp_impulse_nm_sec": peak * peak / (2 * slew),
            "nominal_tick_ramp_impulse_nm_sec": ramp_impulse,
            "control_period_sec": dt,
            "nominal_peak_within_optimistic_ceiling": not reasons,
            "limiting_reasons": reasons,
            "semantics": "No-motion full-authority upper bound; not predicted motion, accepted dispatch or measured torque.",
        }


@dataclass(frozen=True)
class B601StagedAssistFrame:
    """Auditable owner-tick output; torque fields are requested software overlays."""

    reference: B601MitReference
    base_frame: B601SharedControlFrame
    phase: B601StagedAssistPhase
    reason: str
    joint_name: str
    policy_sequence: int | None
    intent_generation: int
    overlay_torque_nm: tuple[float, ...]
    breakaway_impulse_nm_sec: float
    settle_integral_torque_nm: float
    motion: B601JointMotionFrame | None = None
    drive_driver_kd_nm_s_per_rad: tuple[float, ...] = (0.0,) * 6
    support_max_velocity_rad_s: float = 0.0
    support_raw_max_velocity_rad_s: float = 0.0
    motion_onset_confirm_count: int = 0
    motion_onset_required_samples: int = 1
    release_terminal: bool = False
    release_completed: bool = False
    release_failed: bool = False
    release_restoring_base_damping: bool = False
    conditioned_command_delta_nm: tuple[float, ...] | None = None
    conditioned_command_impulse_nm_sec: float = 0.0
    breakaway_probe: dict[str, object] | None = None
    tracking_recovery: dict[str, object] | None = None


class B601StagedSingleJointAssistRuntime:
    """Stages an existing shared-control intent without taking actuator ownership.

    The caller must explicitly call :meth:`authorize_breakaway` during an
    active cue.  The grant is single-use and invalidated by a materially new
    target, clutch, human override, stale command, or command withdrawal.
    Explicit revocation prevents a pending/current cue grant from starting; a
    caller must clear or withdraw policy separately to stop an active pulse.
    """

    def __init__(
        self,
        shared_config: B601SharedControlConfig,
        staged_config: B601StagedAssistConfig,
    ) -> None:
        if not isinstance(shared_config, B601SharedControlConfig):
            raise TypeError("shared_config must be a B601SharedControlConfig")
        if not isinstance(staged_config, B601StagedAssistConfig):
            raise TypeError("staged_config must be a B601StagedAssistConfig")
        self.shared_config = shared_config
        self.config = staged_config
        self.joint_index = B601_RS_ARM_JOINT_NAMES.index(staged_config.joint_name)
        if any(
            value != 0.0
            for values in (
                shared_config.position_kp_nm_per_rad,
                shared_config.velocity_kd_nm_s_per_rad,
            )
            for index, value in enumerate(values)
            if index != self.joint_index
        ):
            raise ValueError("shared_config must disable assist on every non-target joint")
        if staged_config.max_total_overlay_nm > shared_config.max_abs_assist_torque_nm[self.joint_index]:
            raise ValueError("max_total_overlay_nm exceeds the existing shared-control joint cap")
        self._runtime = B601SharedControlRuntime(shared_config)
        self._phase = B601StagedAssistPhase.PASSIVE
        self._reason = "no_policy_command"
        self._intent_generation = 0
        self._intent_target_rad: float | None = None
        self._intent_started_sec: float | None = None
        self._intent_direction = 0.0
        self._authorized_generation: int | None = None
        self._request_lock = threading.Lock()
        self._latest_published_command: B601PolicyAssistCommand | None = None
        self._authorization_request: tuple[int, int, float] | None = None
        self._breakaway_revocation_requested = False
        self._published_phase = B601StagedAssistPhase.PASSIVE
        self._published_generation = 0
        self._breakaway_started_sec: float | None = None
        self._breakaway_anchor_rad: float | None = None
        self._breakaway_impulse_nm_sec = 0.0
        self._motion_onset_count = 0
        self._settle_started_sec: float | None = None
        self._settle_integral_torque_nm = 0.0
        self._previous_overlay_nm = 0.0
        self._tracking_bias_nm = 0.0
        self._tracking_recovery_permitted = False
        self._tracking_recovery: dict[str, object] | None = None
        self._motion = (
            None if staged_config.motion_control is None
            else B601JointMotionTracker(staged_config.motion_control)
        )
        self._motion_frame: B601JointMotionFrame | None = None
        self._stop_evidence: B601WindowStopEvidence | None = None
        self._brake_requested = False
        self._drive_driver_kd = (0.0,) * 6
        self._support_max_velocity = 0.0
        self._support_raw_max_velocity = 0.0
        self._tick_dt = 0.0
        self._withdrawal_request: str | None = None
        self._release_active = False
        self._release_terminal = False
        self._release_completed = False
        self._release_failed = False
        self._release_first = False
        self._release_reason = ""
        self._release_transition_ready = False
        self._release_restore = False
        self._final_zero_confirmed = False
        self._last_output_sequence: int | None = None
        self._last_output_timestamp: float | None = None
        self._last_output_zero = False
        self._command_delta: tuple[float, ...] | None = None
        self._command_rate: float | None = None
        self._command_clock: float | None = None
        self._command_impulse = 0.0
        self._command_accounting = False
        self._capture_started_sec: float | None = None
        self._breakaway_probe: dict[str, object] | None = None
        self._breakaway_probe_frontier = 0.0
        self._breakaway_probe_confirmed = False

    @property
    def release_completed(self) -> bool:
        return self._release_completed

    def request_withdrawal(self, reason: str) -> None:
        if self.config.motion_onset_confirm_displacement_rad is None:
            raise ValueError("confirmed withdrawal requires explicit motion confirmation")
        with self._request_lock:
            self._withdrawal_request = str(reason)
            self._authorization_request = None
        self.clear_policy()

    @property
    def phase(self) -> B601StagedAssistPhase:
        with self._request_lock:
            return self._published_phase

    def set_policy_clutch(self, active: bool) -> None:
        self._runtime.set_policy_clutch(active)
        if active and self.config.capture_overlay_cap_nm and self.phase in {
            B601StagedAssistPhase.RELEASING, B601StagedAssistPhase.CAPTURE,
        }:
            self.request_withdrawal("policy_clutch")

    def request_stop_and_hold(self) -> None:
        if self._motion is None:
            raise ValueError("stop-and-hold requires opt-in motion control")
        with self._request_lock:
            self._brake_requested = True
            self._authorization_request = None

    def record_dispatch(
        self, control_sequence: int, *,
        conditioned_torque_nm: Sequence[float] | None = None,
        passive_torque_nm: Sequence[float] | None = None,
        driver_kd_nm_s_per_rad: Sequence[float] | None = None,
        torque_rate_nm_s: Sequence[float] | None = None,
        completed_at_sec: float | None = None,
    ) -> None:
        if self.config.motion_onset_confirm_displacement_rad is None:
            if self._motion is not None:
                self._motion.record_dispatch(control_sequence)
            return
        if isinstance(control_sequence, bool) or not isinstance(control_sequence, int):
            raise TypeError("conditioned dispatch sequence must be an integer")
        if control_sequence != self._last_output_sequence:
            raise ValueError("conditioned dispatch must match the current owner output")
        vectors = (conditioned_torque_nm, passive_torque_nm, driver_kd_nm_s_per_rad, torque_rate_nm_s)
        if any(v is None for v in vectors) or completed_at_sec is None:
            raise ValueError("confirmed withdrawal requires complete six-joint conditioned dispatch")
        final, passive, kd, rates = (
            tuple(_finite(x, name="conditioned dispatch") for x in v) for v in vectors
        )
        if any(len(v) != 6 for v in (final, passive, kd, rates)) or min(kd) < 0 or min(rates) <= 0:
            raise ValueError("conditioned dispatch requires six finite values and valid gains/rates")
        now = _finite(completed_at_sec, name="dispatch clock")
        if self._last_output_timestamp is None or not (
            self._last_output_timestamp <= now
            <= self._last_output_timestamp + self.config.motion_control.max_control_gap_sec
        ):
            raise ValueError("conditioned dispatch must complete within the current owner watchdog")
        if self._command_clock is not None and now < self._command_clock:
            raise ValueError("conditioned dispatch clock must advance")
        # Validate the complete final receipt before committing any planner state.
        self._motion.record_dispatch(control_sequence)
        self._charge_command(now)
        self._command_delta = tuple(a-b for a,b in zip(final, passive))
        self._command_clock = now
        self._command_rate = rates[self.joint_index]
        self._consume_handoff_stop()
        self._check_handoff_intent_at_receipt(now)
        self._expire_release_budget(now)
        self._final_zero_confirmed = self._last_output_zero and max(abs(x) for x in self._command_delta) <= 1e-9
        probe = self._breakaway_probe
        if probe is not None and probe["stage"] == "hold":
            delta = self._command_delta[self.joint_index]
            matches = math.isclose(delta, probe["hold_nm"], rel_tol=0, abs_tol=1e-9)
            if probe["first_hold_completed_at_sec"] is not None and not matches:
                self._begin_release(now, self._stop_evidence.qpos_rad[self.joint_index], "held_probe_final_command_changed", terminal=True)
            elif matches and probe["first_hold_completed_at_sec"] is None:
                probe["first_hold_completed_at_sec"] = now
        self._last_output_sequence = None  # A duplicate receipt cannot commit twice.
        if self._release_active and not self._release_failed and self._release_transition_ready and self._final_zero_confirmed:
            if self._breakaway_probe_confirmed and (
                self._intent_direction * (self._stop_evidence.qpos_rad[self.joint_index] - self._breakaway_anchor_rad)
                < self.config.motion_onset_confirm_displacement_rad
            ):
                # Check the measured pose at the FIRST accepted zero ACK,
                # too: the previous cycle may still have a conditioner tail.
                self._release_terminal = True
                self._release_restore = True
                self._release_reason = "held_probe_progress_returned_after_withdrawal"
                self._reason = self._release_reason
            if not self._release_terminal and self.config.capture_overlay_cap_nm:
                self._release_active = False
                self._phase = B601StagedAssistPhase.CAPTURE
                self._reason = "final_zero_confirmed_low_force_capture_pending"
            elif not self._release_terminal and self._motion_onset_count >= self.config.motion_onset_confirm_samples:
                self._release_active = False
                self._command_accounting = False
                self._phase = B601StagedAssistPhase.DROP
                self._reason = "conditioned_pulse_withdrawal_confirmed"
            elif self._release_restore and not any(kd):
                self.clear_policy()
                self._reset(B601StagedAssistPhase.PASSIVE, "conditioned_withdrawal_and_stop_confirmed")
                self._release_completed = True
            with self._request_lock:
                self._published_phase = self._phase
        elif self._phase is B601StagedAssistPhase.CAPTURE and not self._release_terminal:
            if self._command_impulse > self.config.breakaway_max_impulse_nm_sec + 1e-9:
                self._release_terminal = True
                self._release_reason = "capture_conditioned_command_budget_exceeded"
                self.clear_policy()
            elif abs(self._command_delta[self.joint_index]) > self.config.capture_overlay_cap_nm + 1e-9:
                # A changing gravity baseline can invalidate a relative cap.
                # Do not silently grant TRACK from an out-of-envelope receipt.
                self._release_terminal = True
                self._release_reason = "capture_conditioned_overlay_cap_exceeded"
                self.clear_policy()
            elif self._motion_onset_count >= self.config.motion_onset_confirm_samples:
                self._phase = B601StagedAssistPhase.TRACK
                self._reason = "capture_displacement_confirmed_reference_retained"
                self._command_accounting = False
            if self._release_terminal:
                self._reason = self._release_reason
            with self._request_lock:
                self._published_phase = self._phase

    def _charge_command(self, now: float) -> None:
        if self._command_clock is not None and now >= self._command_clock:
            if self._command_accounting and self._command_delta is not None:
                self._command_impulse += abs(self._command_delta[self.joint_index]) * (now-self._command_clock)
            self._command_clock = now

    def _check_handoff_intent_at_receipt(self, now: float) -> None:
        pending_soft = self._breakaway_probe is not None and self._breakaway_probe.get("withdrawal_confirmation_pending")
        if (not self.config.capture_overlay_cap_nm and not pending_soft) or self._release_terminal or self._phase not in {
            B601StagedAssistPhase.RELEASING, B601StagedAssistPhase.CAPTURE,
        }:
            return
        with self._request_lock:
            withdrawal = self._withdrawal_request
        sample = self._runtime.sample_policy(now_sec=now)
        command = sample.command
        reason = (
            withdrawal if withdrawal is not None else
            "policy_clutch" if self._runtime.policy_clutch else
            sample.reason if not sample.accepted else
            "capture_target_changed_withdrawal" if abs(
                command.qpos_target_rad[self.joint_index]-self._intent_target_rad
            ) >= self.config.new_intent_delta_rad else None
        )
        if reason is not None:
            self._release_terminal = True
            self._release_transition_ready = False
            self._release_reason = reason
            self._reason = reason
            self.clear_policy()

    def _consume_handoff_stop(self) -> None:
        capturing = self._phase is B601StagedAssistPhase.CAPTURE
        if not self.config.continuous_motion_handoff or not (self._release_active or capturing):
            return
        with self._request_lock:
            requested = self._brake_requested
            self._brake_requested = False
        if requested and not self._release_terminal:
            # A stop arriving before the complete ACK must veto DROP, too.
            self._release_terminal = True
            self._release_transition_ready = False
            self._release_reason = "operator_stop_during_low_force_capture" if capturing else "operator_stop_during_continuous_handoff"
            self._reason = self._release_reason
            self.clear_policy()

    def _breakaway_expiry_reason(self, now: float) -> str | None:
        duration = (
            self._breakaway_started_sec is not None
            and now - self._breakaway_started_sec >= self.config.breakaway_max_duration_sec - 1e-9
        )
        impulse = self._breakaway_impulse_nm_sec >= self.config.breakaway_max_impulse_nm_sec
        if duration or impulse:
            return "breakaway_impulse_budget_exhausted" if impulse else "breakaway_duration_exhausted_before_impulse_budget"
        return None

    def _expire_release_budget(self, now: float) -> None:
        if (self._release_active or self._phase is B601StagedAssistPhase.CAPTURE) and not self._release_terminal:
            reason = self._breakaway_expiry_reason(now)
            if reason is not None:
                self._release_terminal = True
                self._release_reason = reason
                self._reason = reason
                self._authorized_generation = None
                self.clear_policy()

    def _begin_release(
        self, now: float, q: float, reason: str, *, terminal: bool,
        preserve_motion: bool = False,
    ) -> None:
        self._tracking_bias_nm = 0.0
        self._tracking_recovery = None
        if self._breakaway_probe is not None and self._breakaway_probe["stage"] == "hold":
            self._breakaway_probe.update(stage="released", outcome=reason, ended_at_sec=now)
        if not self._release_active:
            # A failed motion stop already owns its braking clock. Starting
            # another tracker here would give an expired stop a new deadline.
            if not preserve_motion:
                self._motion.start(now=now, q=q, target=q)
            self._release_first = not preserve_motion
            self._release_transition_ready = False
            self._release_restore = False
            self._final_zero_confirmed = False
        self._release_active = True
        self._release_terminal = terminal or self._release_terminal
        self._release_completed = False
        self._release_reason = reason
        self._phase = B601StagedAssistPhase.RELEASING
        self._authorized_generation = None
        self._previous_overlay_nm = 0.0

    def _release_output(
        self, base_frame: B601SharedControlFrame, now: float, q: float,
        raw_v: float, control_qvel: Sequence[float],
    ) -> float:
        if (self.config.tracking_recovery_gain_nm_per_rad is not None
            and self._release_terminal and self._release_failed):
            # Observation recovery is not a new motion grant. Keep this failed
            # trial latched for the outer owner shutdown; do not advance the
            # old tracker or turn a later zero receipt into stop completion.
            self._release_transition_ready = False
            self._release_restore = False
            return 0.0
        if self._command_accounting and self._command_impulse > self.config.breakaway_max_impulse_nm_sec + 1e-9:
            # Changing gravity or delayed dispatch may invalidate the constant-
            # passive tail reserve. Fail closed, never claim a force guarantee.
            self._release_failed = True
            self._reason = "conditioned_command_impulse_budget_exceeded"
            return 0.0
        if self.config.continuous_motion_handoff and not self._release_terminal:
            speed = max(abs(float(control_qvel[self.joint_index])), self._support_max_velocity)
            if speed < self.config.motion_control.soft_velocity_rad_s:
                # Withdraw the high pulse, but do not force a confirmed normal
                # handoff back into rest/stiction. Complete final zero receipt
                # and sustained displacement are still mandatory; the caller
                # retains its independent raw hard guards and stage damping.
                self._release_transition_ready = True
                self._release_restore = False
                self._reason = "continuous_handoff_final_zero_pending"
                return 0.0
            self._release_terminal = True
            self._release_reason = "continuous_handoff_soft_velocity_braking"
            self.clear_policy()
        if self.config.continuous_motion_handoff and self._release_first:
            # The waiting handoff has no planner advancement. Start terminal
            # braking from THIS fresh state, not from the old waiting timestamp.
            self._motion.start(now=now, q=q, target=q)
        motion = self._motion.update(
            now=now, q=q, raw_v=raw_v,
            control_v=float(control_qvel[self.joint_index]),
            control_sequence=base_frame.reference.sequence, authority=1.0, kp=0.0, kd=0.0,
            stop_error_rad=self.config.settle_error_rad,
            stop_velocity_rad_s=self.config.settle_velocity_rad_s,
            stop_dwell_sec=self.config.settle_dwell_sec,
            request_brake=self._release_first, brake_reason=self._release_reason,
            support_max_velocity_rad_s=self._support_max_velocity,
            support_raw_max_velocity_rad_s=self._support_raw_max_velocity,
            stop_evidence=self._stop_evidence,
        )
        first = self._release_first
        self._release_first = False
        self._motion_frame = motion
        self._release_failed = motion.phase == "authority_exhausted"
        resolution = self.config.motion_control.position_resolution_rad
        stationary = (
            self._stop_evidence is not None and self._stop_evidence.gate_reason(
                window_sec=self.config.motion_control.velocity_window_sec,
                velocity_limit=self.config.settle_velocity_rad_s, resolution=resolution,
            ) is None
        ) if resolution is not None else max(
            abs(raw_v), abs(float(control_qvel[self.joint_index])),
            self._support_max_velocity, self._support_raw_max_velocity,
        ) <= self.config.settle_velocity_rad_s
        self._release_transition_ready = motion.phase == "soft_hold" and stationary
        if self._release_transition_ready and self._final_zero_confirmed:
            if not self._release_terminal and self._motion_onset_count < self.config.motion_onset_confirm_samples:
                self._release_terminal = True
                self._release_reason = "motion_onset_unconfirmed_withdrawn"
            self._release_restore = self._release_terminal
        else:
            self._release_restore = False
        self._reason = motion.reason if self._release_failed else self._release_reason
        # No stop-point spring during withdrawal. The existing brake supplies
        # bounded dissipative torque; a stationary frame requests pure gravity.
        return 0.0 if first or motion.phase in {"soft_hold", "authority_exhausted"} else max(
            -self.config.tracking_overlay_cap_nm,
            min(self.config.tracking_overlay_cap_nm, motion.requested_torque_nm),
        )

    def publish_policy(
        self,
        command: B601PolicyAssistCommand,
    ) -> B601PolicyCommandReceipt:
        receipt = self._runtime.publish_policy(command)
        with self._request_lock:
            self._latest_published_command = command if receipt.accepted else None
            if not receipt.accepted:
                self._authorization_request = None
        return receipt

    def clear_policy(self) -> int | None:
        cleared_sequence = self._runtime.clear_policy()
        with self._request_lock:
            self._latest_published_command = None
            self._authorization_request = None
        return cleared_sequence

    def authorize_breakaway(self, *, sequence: int) -> bool:
        """Grant one breakaway attempt for the current cue's intent."""

        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise TypeError("sequence must be a non-negative integer")
        if sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        with self._request_lock:
            command = self._latest_published_command
            if (
                self._published_phase is not B601StagedAssistPhase.CUE
                or command is None
                or command.sequence != sequence
            ):
                return False
            self._authorization_request = (
                self._published_generation,
                sequence,
                command.qpos_target_rad[self.joint_index],
            )
        return True

    def revoke_breakaway_authorization(self) -> None:
        """Revoke a pending or current-cue grant before a pulse starts."""

        with self._request_lock:
            self._authorization_request = None
            self._breakaway_revocation_requested = True

    def step(
        self,
        *,
        control_sequence: int,
        control_timestamp_sec: float,
        measured_qpos_rad: Sequence[float],
        measured_qvel_rad_s: Sequence[float],
        dt_sec: float,
        human_override_detected: bool = False,
        control_qvel_rad_s: Sequence[float] | None = None,
        control_qvel_window_span_sec: float | None = None,
        stop_evidence: B601WindowStopEvidence | None = None,
        tracking_recovery_permitted: bool = False,
    ) -> B601StagedAssistFrame:
        if type(tracking_recovery_permitted) is not bool:
            raise TypeError("tracking recovery permission must be boolean")
        self._tracking_recovery_permitted = tracking_recovery_permitted
        base_frame = self._runtime.step(
            control_sequence=control_sequence,
            control_timestamp_sec=control_timestamp_sec,
            measured_qpos_rad=measured_qpos_rad,
            measured_qvel_rad_s=measured_qvel_rad_s,
            dt_sec=dt_sec,
            human_override_detected=human_override_detected,
        )
        index = self.joint_index
        measured_q = float(measured_qpos_rad[index])
        measured_v = float(measured_qvel_rad_s[index])
        decision = base_frame.decision
        now = float(control_timestamp_sec)
        dt = float(dt_sec)
        self._tick_dt = dt
        if stop_evidence is not None and (
            stop_evidence.timestamp_sec != now
            or stop_evidence.joint_index != self.joint_index
            or stop_evidence.qpos_rad != tuple(measured_qpos_rad)
            or stop_evidence.raw_qvel_rad_s != tuple(measured_qvel_rad_s)
            or stop_evidence.window_qvel_rad_s != (() if control_qvel_rad_s is None else tuple(control_qvel_rad_s))
        ):
            raise ValueError("stop evidence must match this six-axis feedback tick")
        self._stop_evidence = stop_evidence
        motion_config = self.config.motion_control
        if motion_config is not None and motion_config.continuous_speed_limit_rad_s is not None:
            # Same observer/clock in CUE, breakaway and TRACK. Its lifecycle is
            # reset by a new intent, not by the normal onset handoff.
            self._motion.observe_feedback(now=now, q=measured_q, raw_v=measured_v)
        if control_qvel_rad_s is not None:
            control_qvel_rad_s = tuple(_finite(v, name="control velocity") for v in control_qvel_rad_s)
            if len(control_qvel_rad_s) != 6:
                raise ValueError("control velocity requires six joints")
        windowed = motion_config is not None and motion_config.windowed_soft_velocity
        if control_qvel_window_span_sec is not None:
            control_qvel_window_span_sec = _finite(control_qvel_window_span_sec, name="control velocity window span")
            if control_qvel_window_span_sec < 0:
                raise ValueError("control velocity window span must be non-negative")
        soft_qvel = control_qvel_rad_s if windowed and control_qvel_rad_s is not None else measured_qvel_rad_s
        monitor_support = motion_config is not None and any(motion_config.drive_driver_kd_nm_s_per_rad)
        self._support_max_velocity = (
            max(abs(float(v)) for i, v in enumerate(soft_qvel) if i != index)
            if monitor_support else 0.0
        )
        self._support_raw_max_velocity = (
            max(abs(float(v)) for i, v in enumerate(measured_qvel_rad_s) if i != index)
            if monitor_support else 0.0
        )
        self._motion_frame = None

        self._charge_command(now)
        strict = self.config.motion_onset_confirm_displacement_rad is not None
        active = self._phase not in {
            B601StagedAssistPhase.PASSIVE, B601StagedAssistPhase.HUMAN_OVERRIDE,
            B601StagedAssistPhase.AUTHORITY_EXHAUSTED,
        }
        if strict and active and self._last_output_sequence is not None:
            raise ValueError("previous owner output lacks a complete conditioned dispatch receipt")
        with self._request_lock:
            withdrawal_request = self._withdrawal_request
            self._withdrawal_request = None
        if withdrawal_request is not None:
            self._begin_release(now, measured_q, withdrawal_request, terminal=True)
        elif (strict and (active or decision.reason in {"policy_assist", "recovery_ramp"}) and not self._release_active
              and self.config.tracking_recovery_gain_nm_per_rad is not None
              and not tracking_recovery_permitted):
            self._begin_release(now, measured_q, "tracking_recovery_permission_withdrawn", terminal=True)
        elif (strict and active and not self._release_active
              and self.config.tracking_recovery_gain_nm_per_rad is not None
              and stop_evidence is not None
              and any(abs(v) > stop_evidence.raw_velocity_limit_rad_s for v in measured_qvel_rad_s)):
            # Defense at the independently callable control boundary; the
            # hardware owner still retains its original six-axis hard guard.
            self._begin_release(now, measured_q, "tracking_recovery_raw_velocity_guard", terminal=True)
        elif strict and active and not self._release_active and decision.reason not in {"policy_assist", "recovery_ramp"}:
            reason = self._release_reason if self._release_terminal else decision.reason
            self._begin_release(now, measured_q, reason, terminal=True)
        if self._release_active:
            self._consume_handoff_stop()
            self._expire_release_budget(now)
            if (
                dt > motion_config.max_control_gap_sec
                or control_qvel_rad_s is None or control_qvel_window_span_sec is None
                or control_qvel_window_span_sec < motion_config.velocity_window_sec - 1e-9
            ):
                self._release_failed = True
                self._reason = "withdrawal_feedback_or_control_gap_invalid"
                return self._frame(base_frame, 0.0)
            if decision.reason not in {"policy_assist", "recovery_ramp"} and not self._release_terminal:
                self._begin_release(now, measured_q, decision.reason, terminal=True)
            probe = self._breakaway_probe
            if not self._release_terminal and probe is not None and probe.get("withdrawal_confirmation_pending"):
                self._observe_soft_withdrawal(now, measured_q, float(soft_qvel[index]))
            elif not self._release_terminal and self._breakaway_anchor_rad is not None and not self._breakaway_probe_confirmed:
                confirmed = (
                    self._intent_direction * (measured_q-self._breakaway_anchor_rad)
                    >= self.config.motion_onset_confirm_displacement_rad
                    and self._intent_direction * float(soft_qvel[index]) >= 0.0
                )
                self._motion_onset_count = self._motion_onset_count+1 if confirmed else 0
            overlay = self._release_output(base_frame, now, measured_q, measured_v, soft_qvel)
            self._previous_overlay_nm = overlay
            return self._frame(base_frame, overlay)

        if decision.state.mode is B601SharedControlMode.HUMAN_OVERRIDE:
            self._reset(B601StagedAssistPhase.HUMAN_OVERRIDE, decision.reason)
            return self._frame(base_frame, None)
        if decision.reason not in {"policy_assist", "recovery_ramp"}:
            phase = (
                B601StagedAssistPhase.PASSIVE
                if decision.state.mode is B601SharedControlMode.PASSIVE
                else B601StagedAssistPhase.HUMAN_OVERRIDE
            )
            self._reset(phase, decision.reason)
            return self._frame(base_frame, None)

        target_q = measured_q + decision.raw_target_error_rad[index]
        target_error = decision.raw_target_error_rad[index]
        direction = 1.0 if target_error > 0.0 else -1.0 if target_error < 0.0 else 0.0
        if self._phase is B601StagedAssistPhase.CAPTURE:
            if dt > motion_config.max_control_gap_sec or (
                control_qvel_rad_s is None or control_qvel_window_span_sec is None
                or control_qvel_window_span_sec < motion_config.velocity_window_sec - 1e-9
            ):
                self._begin_release(now, measured_q, "capture_feedback_or_control_gap_invalid", terminal=True)
                self._release_failed = True
                self._reason = "capture_feedback_or_control_gap_invalid"
                return self._frame(base_frame, 0.0)
            self._consume_handoff_stop()
            self._expire_release_budget(now)
            changed = abs(target_q - self._intent_target_rad) >= self.config.new_intent_delta_rad
            if self._release_terminal or changed or direction != self._intent_direction:
                reason = self._release_reason if self._release_terminal else "capture_target_changed_withdrawal"
                self._begin_release(now, measured_q, reason, terminal=True)
                overlay = 0.0
            else:
                overlay = self._capture_output(base_frame, now, dt, measured_q, measured_v, control_qvel_rad_s)
            if self._release_active:
                overlay = self._release_output(base_frame, now, measured_q, measured_v, soft_qvel)
            self._previous_overlay_nm = overlay
            return self._frame(base_frame, overlay)
        if (
            self._intent_target_rad is None
            or abs(target_q - self._intent_target_rad)
            >= self.config.new_intent_delta_rad
        ):
            self._start_intent(
                now,
                target_q,
                direction=direction,
                measured_q=measured_q,
            )
        with self._request_lock:
            revoke_breakaway = self._breakaway_revocation_requested
            self._breakaway_revocation_requested = False
        if revoke_breakaway:
            self._authorized_generation = None
        self._consume_authorization_request(decision.state.last_policy_sequence)
        with self._request_lock:
            brake_requested = self._brake_requested
            self._brake_requested = False
        if (
            self._motion is not None
            and dt > self.config.motion_control.max_control_gap_sec
        ):
            if strict:
                self._begin_release(now, measured_q, "control_gap_assist_withdrawn", terminal=True)
                self._release_failed = True
                self._reason = "control_gap_assist_withdrawn"
                return self._frame(base_frame, 0.0)
            self._phase = B601StagedAssistPhase.AUTHORITY_EXHAUSTED
            self._reason = "control_gap_assist_withdrawn"
            self._authorized_generation = None
            self._previous_overlay_nm = 0.0
            self._motion.reset()
            return self._frame(base_frame, None)
        if windowed and (
            control_qvel_rad_s is None or control_qvel_window_span_sec is None
            or control_qvel_window_span_sec < motion_config.velocity_window_sec - 1e-9
        ):
            if strict:
                self._begin_release(now, measured_q, "control_velocity_window_unavailable_assist_withdrawn", terminal=True)
                self._release_failed = True
                self._reason = "control_velocity_window_unavailable_assist_withdrawn"
                return self._frame(base_frame, 0.0)
            # Keep this intent's identity exhausted. Losing observation quality
            # must not make the same old published target a new grant later.
            self._phase = B601StagedAssistPhase.AUTHORITY_EXHAUSTED
            self._reason = "control_velocity_window_unavailable_assist_withdrawn"
            self._authorized_generation = None
            self._previous_overlay_nm = 0.0
            self._motion.reset()
            return self._frame(base_frame, None)
        cue_moved = (
            self._motion is not None
            and self._phase is B601StagedAssistPhase.CUE
            and abs(measured_q - self._intent_anchor_rad)
            >= self.config.motion_onset_displacement_rad
        )
        soft_speed = (
            self._motion is not None
            and self._phase in {
                B601StagedAssistPhase.CUE, B601StagedAssistPhase.BREAKAWAY,
                B601StagedAssistPhase.DROP,
            }
            and max(abs(float(soft_qvel[index])), self._support_max_velocity) >= self.config.motion_control.soft_velocity_rad_s
        )
        brake_reason = (
            "operator_stop_and_hold" if brake_requested else
            "cue_motion_braking" if cue_moved else
            "support_velocity_braking" if soft_speed and self._support_max_velocity >= self.config.motion_control.soft_velocity_rad_s else
            "pretracking_soft_velocity_braking" if soft_speed else
            "operator_stop_and_hold"
        )
        if (brake_requested or cue_moved or soft_speed) and self._motion is not None:
            # Braking discards this target's spring, not its diagnostic identity.
            self._motion.start(now=now, q=measured_q, target=target_q)
            self._phase = B601StagedAssistPhase.BRAKE
            self._authorized_generation = None
            self._settle_integral_torque_nm = 0.0
            brake_requested = True

        if self._phase is B601StagedAssistPhase.CUE:
            cue_elapsed = now - float(self._intent_started_sec)
            if cue_elapsed < self.config.cue_duration_sec:
                overlay = self._cue_output(target_error, decision.state.authority, dt)
            else:
                # The cue is a short force pulse, but the operator's one-shot
                # authorization must remain available after that pulse. Do not
                # silently progress into target tracking before explicit grant.
                overlay = self._slew(0.0, self.config.tracking_slew_nm_s, dt)
                can_break = (
                    self._authorized_generation == self._intent_generation
                    and abs(target_error) >= self.config.minimum_breakaway_error_rad
                    and abs(float(soft_qvel[index]) if windowed else measured_v) <= self.config.stuck_max_velocity_rad_s
                )
                if can_break:
                    self._phase = B601StagedAssistPhase.BREAKAWAY
                    self._reason = "authorized_breakaway_started"
                    self._breakaway_started_sec = now
                    self._breakaway_anchor_rad = measured_q
                    self._motion_onset_count = 0
                    self._breakaway_impulse_nm_sec = 0.0
                    self._command_accounting = True
                    self._command_impulse = 0.0
                    self._authorized_generation = None
                    overlay = 0.0
                elif self._authorized_generation == self._intent_generation:
                    self._authorized_generation = None
                    self._reason = (
                        "breakaway_authorization_rejected_small_target_error"
                        if abs(target_error) < self.config.minimum_breakaway_error_rad
                        else "breakaway_authorization_rejected_not_stationary"
                    )
                    overlay = 0.0
                else:
                    self._reason = "awaiting_breakaway_authorization"
        elif self._phase is B601StagedAssistPhase.BREAKAWAY:
            overlay = self._breakaway_output(
                base_frame,
                now=now,
                dt=dt,
                measured_q=measured_q,
                measured_v=measured_v,
                target_error=target_error,
                authority=decision.state.authority,
                control_v=float(soft_qvel[index]),
            )
        elif self._phase is B601StagedAssistPhase.DROP:
            self._phase = B601StagedAssistPhase.TRACK
            self._reason = "motion_onset_confirmed_assist_dropped"
            self._settle_started_sec = None
            if self._motion is not None:
                self._motion.start(now=now, q=measured_q, target=target_q)
                overlay = self._motion_output(
                    base_frame, now, dt, measured_q, measured_v, control_qvel_rad_s
                )
            else:
                overlay = self._tracking_output(base_frame, dt)
        elif self._phase is B601StagedAssistPhase.AUTHORITY_EXHAUSTED:
            overlay = 0.0
        elif self._motion is not None and self._phase in {
            B601StagedAssistPhase.TRACK,
            B601StagedAssistPhase.BRAKE,
            B601StagedAssistPhase.SOFT_HOLD,
        }:
            overlay = self._motion_output(
                base_frame, now, dt, measured_q, measured_v, control_qvel_rad_s,
                request_brake=brake_requested, brake_reason=brake_reason,
            )
        else:
            overlay = self._tracking_or_settle_output(
                base_frame,
                now=now,
                dt=dt,
                target_error=target_error,
                measured_v=measured_v,
            )

        self._previous_overlay_nm = overlay
        if self._release_active and not (
            self.config.continuous_motion_handoff and self._release_failed
        ):
            overlay = self._release_output(base_frame, now, measured_q, measured_v, soft_qvel)
            self._previous_overlay_nm = overlay
        return self._frame(base_frame, overlay)

    def _motion_output(
        self,
        base_frame: B601SharedControlFrame,
        now: float,
        dt: float,
        q: float,
        raw_v: float,
        control_qvel: Sequence[float] | None,
        request_brake: bool = False,
        brake_reason: str = "operator_stop_and_hold",
        capture: bool = False,
        protect_maintenance: bool = True,
    ) -> float:
        control_v = raw_v if control_qvel is None else float(control_qvel[self.joint_index])
        if not math.isfinite(control_v):
            raise ValueError("control velocity must be finite")
        motion = self._motion.update(
            now=now,
            q=q,
            raw_v=raw_v,
            control_v=control_v,
            control_sequence=base_frame.reference.sequence,
            authority=base_frame.decision.state.authority,
            kp=self.shared_config.position_kp_nm_per_rad[self.joint_index],
            kd=self.shared_config.velocity_kd_nm_s_per_rad[self.joint_index],
            stop_error_rad=self.config.settle_error_rad,
            stop_velocity_rad_s=self.config.settle_velocity_rad_s,
            stop_dwell_sec=self.config.settle_dwell_sec,
            request_brake=request_brake,
            brake_reason=brake_reason,
            support_max_velocity_rad_s=self._support_max_velocity,
            support_raw_max_velocity_rad_s=self._support_raw_max_velocity,
            stop_evidence=self._stop_evidence,
            speed_governor_permitted=(self.config.onset_handoff == "track"
                                      and self._tracking_recovery_permitted),
            zero_dispatch_confirmed=self._final_zero_confirmed,
        )
        self._motion_frame = motion
        self._phase = B601StagedAssistPhase(motion.phase)
        self._reason = motion.reason
        if capture and motion.phase == "track":
            self._phase = B601StagedAssistPhase.CAPTURE
            self._reason = "bounded_low_force_capture_awaiting_displacement"
        self._settle_integral_torque_nm = 0.0
        if self._phase is B601StagedAssistPhase.AUTHORITY_EXHAUSTED:
            if (self.config.continuous_motion_handoff
                or self.config.tracking_recovery_gain_nm_per_rad is not None):
                # Explicit candidates only: keep the original failed stop and
                # stage damping until explicit shutdown, never rearm from a
                # new target or mistake zero request for completed withdrawal.
                self._begin_release(now, q, motion.reason, terminal=True, preserve_motion=True)
                self._release_failed = True
                self.clear_policy()
            return 0.0
        cap = self.config.capture_overlay_cap_nm if capture else self.config.tracking_overlay_cap_nm
        requested = motion.requested_torque_nm
        if self.config.tracking_recovery_gain_nm_per_rad is not None:
            requested = self._tracking_recovery_output(motion, base_frame, dt, control_v, cap)
        continuous_arrival = (motion.arrival_stabilization_active
            and self.config.motion_control.continuous_speed_limit_rad_s is not None)
        if (motion.phase == "track" and motion.continuous_speed_scale is not None
            or continuous_arrival):
            # The bounded integral learns load/velocity error separately.
            # Speed scaling removes PD propulsion only. Proportional velocity
            # error must still reduce the complete same-direction output:
            # a bounded integral is not a guaranteed physical load torque.
            # Neither correction is debited recursively from integral memory.
            # Arrival still drains the complete request before final zero.
            direction = math.copysign(1.0, motion.target_error_rad)
            maintenance = (base_frame.decision.state.authority
                * (self._tracking_recovery or {}).get("bias_request_nm", 0.0)
                if motion.phase == "track" and protect_maintenance else 0.0)
            propulsion = max(0.0, direction * (requested-maintenance))
            # Use the SAME causal window as the maintenance integrator.
            # A one-frame count step or a shorter slope must not become a
            # different velocity-error controller superimposed on that loop.
            # Recent travel still owns continuous_scale and soft braking;
            # raw speed still owns the independent hard guard in the tracker.
            feedback_velocity = control_v
            # Fast velocity-error feedback is an OUTPUT correction, not a
            # permanent debit of low-frequency maintenance. The integral still
            # learns velocity error once below, with the existing finite cap.
            target_velocity = (self._tracking_recovery or {}).get("target_velocity_rad_s", 0.0)
            aligned_request = max(0.0, direction * requested)
            velocity_cut = min(aligned_request,
                self.config.tracking_recovery_velocity_feedback_nm_s_per_rad
                * max(0.0, direction * (feedback_velocity-target_velocity)))
            # A short window may still contain the preceding forward burst.
            # Never use it to cut a request opposing CURRENT reverse motion.
            drive = propulsion if direction * control_v >= 0.0 else 0.0
            cut = drive * (1.0 - (motion.continuous_speed_scale
                if motion.continuous_speed_scale is not None else 1.0))
            # Apply the stricter output constraint once, without creating a
            # torque reversal or erasing a separate opposing PD request.
            applied_cut = min(aligned_request, max(velocity_cut, cut))
            requested -= direction * applied_cut
            if self._tracking_recovery is not None:
                self._tracking_recovery.update(
                    tracking_cut_velocity_rad_s=feedback_velocity,
                    tracking_cut_velocity_source="causal_window_same_as_maintenance",
                    unscaled_maintenance_nm=maintenance,
                    regulatable_propulsion_nm=propulsion,
                    tracking_velocity_correction_nm=velocity_cut,
                    continuous_speed_bound_cut_nm=cut,
                    continuous_speed_drive_cut_nm=applied_cut - velocity_cut,
                    applied_drive_cut_nm=applied_cut,
                    combined_request_nm=requested,
                )
        requested = max(-cap, min(cap, requested))
        if self.config.motion_control.responsive_tracking_brake:
            return self._responsive_tracking_output(requested, control_v, dt)
        rate = self.config.tracking_slew_nm_s
        if self._phase is B601StagedAssistPhase.BRAKE:
            # Never let slew retention keep driving into a requested brake.
            if self._previous_overlay_nm * requested <= 0.0:
                self._previous_overlay_nm = 0.0
            rate = self.config.motion_control.braking_slew_nm_s
        return self._slew(requested, rate, dt)

    def _tracking_recovery_output(
        self, motion: B601JointMotionFrame, base: B601SharedControlFrame,
        dt: float, velocity: float, cap: float,
    ) -> float:
        """Bounded velocity-error bias, not a learned physical friction value.

        The existing tracker owns reference, arrival and braking. The automatic
        session owns lease, leg deadline and final-command exposure. Only its
        explicit fresh no-contact permit enables this extra command demand.
        """
        config = self.config.motion_control
        reason = "tracking_bias_reset_outside_track"
        target_velocity = 0.0
        feedback_cut = 0.0
        antiwindup_cut = 0.0
        dispatch_tracking_cut = 0.0
        requested_bias = 0.0
        execution_tracking = self.config.onset_handoff == "track"
        frontier_feedback = execution_tracking and config.speed_governor
        previous_recovery = self._tracking_recovery or {}
        feedback_velocity = previous_recovery.get("feedback_velocity_rad_s", 0.0)
        arrival_error = (
            self.config.settle_error_rad if motion.arrival_brake_error_rad is None
            else motion.arrival_brake_error_rad
        )
        arrival_active = frontier_feedback and motion.arrival_stabilization_active
        arrival_projection = (arrival_active and (
            not str(previous_recovery.get("reason", "")).startswith("arrival_")
            or motion.arrival_unloading))
        if ((self._phase is B601StagedAssistPhase.TRACK or arrival_active)
            and self._tracking_recovery_permitted
            and self._stop_evidence is not None
            and self._stop_evidence.window_span_sec >= config.velocity_window_sec - 1e-9
            and motion.target_error_rad is not None
            and (abs(motion.target_error_rad) > arrival_error or arrival_active)):
            # A transfer can coast across the goal while unloading. Its
            # bounded maintenance must then correct the CURRENT error, not
            # remain locked to the old travel direction. Final reversal still
            # passes the native zero-receipt gate in responsive output.
            direction = (self._intent_direction
                if arrival_active and config.continuous_speed_limit_rad_s is None
                else math.copysign(1.0, motion.target_error_rad))
            target_velocity = math.copysign(1.0, motion.target_error_rad) * min(
                config.max_reference_velocity_rad_s,
                math.sqrt(2 * config.max_reference_acceleration_rad_s2 * max(
                    0.0, abs(motion.target_error_rad) - arrival_error)),
            )
            if (motion.phase == "track" and config.continuous_speed_limit_rad_s is not None
                and self.config.tracking_recovery_velocity_feedback_nm_s_per_rad > 0):
                # Track the dispatched trajectory, not a constant goalward
                # speed after the measured joint has passed that reference.
                # Ki/Kv is the existing PI velocity-loop crossover (1/s).
                # A stuck joint behind a paused reference still has a positive
                # demand; a joint ahead of it cannot keep accumulating drive.
                position_gain = (self.config.tracking_recovery_gain_nm_per_rad
                    / self.config.tracking_recovery_velocity_feedback_nm_s_per_rad)
                reference_demand = motion.qvel_reference_rad_s + position_gain * (
                    motion.q_reference_rad-self._stop_evidence.qpos_rad[self.joint_index])
                target_velocity = direction * min(abs(target_velocity), max(0., direction*reference_demand))
            if (motion.phase == "track" and motion.continuous_speed_scale is not None
                and target_velocity * velocity >= 0.0):
                target_velocity *= motion.continuous_speed_scale
            if self._tracking_bias_nm * direction < 0:
                self._tracking_bias_nm = 0.0
            if execution_tracking and self._command_delta is not None:
                # A receipt observes the composed output, not its PD/bias
                # decomposition. In governed TRACK, credit already requested
                # maintenance first: total slew may not have served all PD.
                # This can only CUT existing state, never copy PD or gravity
                # into it. Velocity error below remains its only growth source.
                if config.continuous_speed_limit_rad_s is not None and arrival_projection:
                    # Arrival must start from what was actually dispatched,
                    # not latent maintenance hidden behind TRACK regulation.
                    # Project ON ENTRY, then again during final unloading.
                    # Repeating this while stabilizing charges each intentional
                    # velocity cut to memory a second time and loses support.
                    # Final unloading cannot restore hidden drive from memory.
                    delivered_bias = max(0.0,
                        direction * self._command_delta[self.joint_index]
                        - max(0.0, direction * motion.requested_torque_nm))
                elif config.continuous_speed_limit_rad_s is not None:
                    # Compare the preceding owner request with ITS receipt.
                    # Planned speed regulation/slew already reduced that
                    # request; do not charge the same reduction to memory
                    # again every tick. Only an additional downstream shortfall
                    # can cut the bounded maintenance state here.
                    shortfall = max(0.0, direction * (
                        self._previous_overlay_nm - self._command_delta[self.joint_index]))
                    delivered_bias = max(0.0, direction * self._tracking_bias_nm - shortfall)
                else:
                    delivered_bias = max(0.0, direction * self._command_delta[self.joint_index])
                if not frontier_feedback:
                    previous_pd = previous_recovery.get("pd_request_nm", 0.0)
                    delivered_bias = max(0.0, direction * (
                        self._command_delta[self.joint_index] - previous_pd))
                dispatch_tracking_cut = max(
                    0.0, direction * self._tracking_bias_nm - delivered_bias)
                self._tracking_bias_nm = direction * min(
                    direction * self._tracking_bias_nm, delivered_bias)
            # Feed the complete causal window, not single-count onset events.
            # No pulse reauthorization or deadline/budget restart is allowed.
            delta = self.config.tracking_recovery_gain_nm_per_rad * (target_velocity - velocity) * dt
            rate = self.config.tracking_slew_nm_s if delta * direction >= 0 else config.braking_slew_nm_s
            if arrival_active and motion.arrival_unloading:
                # Only drain existing receipt-bounded state. Any renewed motion
                # or target drift cancels unloading in the authoritative tracker.
                delta = -self._tracking_bias_nm
                rate = config.arrival_unload_slew_nm_s
            magnitude = max(0.0, min(cap, direction * (
                self._tracking_bias_nm + max(-rate * dt, min(rate * dt, delta)))))
            # Anti-windup: PD and bias share ONE phase/total cap.
            headroom = max(0.0, cap - max(0.0, direction * motion.requested_torque_nm))
            antiwindup_cut = max(0.0, magnitude-headroom)
            magnitude = min(magnitude, headroom)
            # Continuous execution integrates velocity error once and shapes
            # PD separately. Do not also charge a resolved-speed frontier cut
            # to maintenance: that duplicate cut can re-stick the gearbox.
            if (config.continuous_speed_limit_rad_s is None
                and self._stop_evidence.position_span_rad[self.joint_index] > config.position_resolution_rad):
                # A rate bound alone cannot create fast negative feedback.
                # This proportional cut only removes same-direction bias;
                # it neither strengthens assistance nor requests reversal.
                # Continuous execution carries ONE net maintenance state.
                # The same-goal governor can regulate a new speed frontier
                # without repeatedly charging the same quantized progress.
                # Its original soft brake still owns sustained overspeed.
                velocity_increase = max(0.0, direction * velocity)
                if execution_tracking:
                    previous_speed = max(0.0, direction * feedback_velocity)
                    velocity_increase -= previous_speed
                if frontier_feedback:
                    feedback_velocity = direction * max(
                        previous_speed, direction * velocity, 0.0)
                else:
                    feedback_velocity = velocity
                feedback_cut = min(magnitude, self.config.tracking_recovery_velocity_feedback_nm_s_per_rad * max(0.0, velocity_increase))
            elif execution_tracking and not frontier_feedback:
                feedback_velocity = 0.0
            # In governed TRACK a one-count window cannot establish a stop.
            # Keep the frontier so the next window cannot charge it again.
            requested_bias = direction * (magnitude - feedback_cut)
            # The diagnostic path retains its bounded pre-feedback estimate.
            # Continuous execution stores the net request and tracks delivered
            # demand on the next tick, including downstream conditioning.
            self._tracking_bias_nm = requested_bias if execution_tracking else direction * magnitude
            reason = "bounded_velocity_error_maintenance"
            if arrival_active:
                reason = "arrival_unloading_existing_maintenance" if motion.arrival_unloading else "arrival_bounded_stabilization"
        elif (config.continuous_speed_limit_rad_s is not None
            and motion.speed_governor_active
            and motion.reason in {"soft_velocity_braking", "resolved_tracking_speed_braking",
                                  "speed_governor_waiting_for_zero_dispatch"}
            and not str(previous_recovery.get("reason", "")).startswith("arrival_")
            and self._tracking_recovery_permitted
            and self._stop_evidence is not None
            and self._stop_evidence.window_span_sec >= config.velocity_window_sec - 1e-9):
            # Memory is NOT a brake output. Freeze only previously dispatched
            # same-goal maintenance through an authorized soft stop. No new
            # integration, pulse, lease or budget is granted. Resume still
            # requires native zero/still and the ordinary tracking slew.
            direction = self._intent_direction
            if previous_recovery.get("reason") != "tracking_bias_frozen_for_governed_stop":
                delivered = max(0.0, direction*self._command_delta[self.joint_index])
                self._tracking_bias_nm = direction*min(cap, max(0.0, direction*self._tracking_bias_nm), delivered)
            reason = "tracking_bias_frozen_for_governed_stop"
        else:
            self._tracking_bias_nm = 0.0
            if frontier_feedback:
                feedback_velocity = 0.0
        requested = motion.requested_torque_nm + base.decision.state.authority * requested_bias
        self._tracking_recovery = {
            "reason": reason, "permitted": self._tracking_recovery_permitted,
            "target_velocity_rad_s": target_velocity,
            "window_velocity_rad_s": velocity,
            "bias_request_nm": requested_bias,
            "maintenance_bias_nm": self._tracking_bias_nm,
            "velocity_feedback_cut_nm": feedback_cut,
            "antiwindup_cut_nm": antiwindup_cut,
            "dispatch_tracking_cut_nm": dispatch_tracking_cut,
            "dispatch_credit_mode": ("arrival_unload_sent_total" if motion.arrival_unloading
                and arrival_projection and config.continuous_speed_limit_rad_s is not None
                else "arrival_entry_reserve_pd_from_sent_total"
                if arrival_projection and config.continuous_speed_limit_rad_s is not None
                else "receipt_shortfall_from_previous_owner_request"
                if config.continuous_speed_limit_rad_s is not None
                else "maintenance_first_bounded_by_sent_total" if frontier_feedback else "reserve_requested_pd_first"),
            "feedback_velocity_rad_s": feedback_velocity,
            "velocity_feedback_baseline_mode": (
                "continuous_pd_scaling_with_net_velocity_feedback" if config.continuous_speed_limit_rad_s is not None
                else "track_resolved_speed_frontier" if frontier_feedback else "current_resolved_speed"),
            "maintenance_state_mode": ("receipt_bounded_frozen_memory"
                if reason == "tracking_bias_frozen_for_governed_stop" else "bounded_bias_with_separate_transient_feedback"
                if execution_tracking and config.continuous_speed_limit_rad_s is not None
                else "dispatch_tracked_net_bias" if execution_tracking else "bounded_bias_with_output_feedback"),
            "pd_request_nm": motion.requested_torque_nm,
            "combined_request_nm": requested, "total_cap_nm": cap,
            "semantics": "software command bias, not measured friction or force",
        }
        return requested

    def _responsive_tracking_output(self, requested: float, control_v: float, dt: float) -> float:
        """Slow guidance rise, fast withdrawal, native zero before reversal."""
        if self._command_delta is None:
            raise ValueError("responsive tracking requires a conditioned dispatch receipt")
        previous_sent = self._command_delta[self.joint_index]
        previous = self._previous_overlay_nm
        continuous_arrival = (self._motion_frame.arrival_stabilization_active
            and self.config.motion_control.continuous_speed_limit_rad_s is not None)
        if continuous_arrival and previous * previous_sent >= 0.0:
            # A downstream cut is authoritative during arrival. Do not restore
            # the difference at the fast braking slew on the following tick.
            previous = math.copysign(min(abs(previous), abs(previous_sent)), previous)
            self._previous_overlay_nm = previous
        rate = self.config.tracking_slew_nm_s
        reason = "slow_guidance_rise"
        opposes_motion = requested * control_v < 0.0
        if self.config.motion_control.continuous_speed_limit_rad_s is not None:
            # A delayed window alone cannot authorize the fast braking lane:
            # the latest paired interval must agree that this request opposes
            # motion. Otherwise it is ordinary guidance, with its normal slew.
            opposes_motion = (opposes_motion and self._stop_evidence is not None
                and requested * self._stop_evidence.raw_qvel_rad_s[self.joint_index] < 0.0)
        if (
            requested * previous_sent < 0.0 and not self._final_zero_confirmed
            or requested * previous < 0.0
            or self._last_output_zero and not self._final_zero_confirmed
        ):
            # Do not cross directly into the opposite torque while an older
            # conditioned command remains present. The existing owner ACK
            # must confirm zero, including final-conditioning retention.
            requested = 0.0
            rate = self.config.motion_control.braking_slew_nm_s
            reason = "native_zero_pending_before_torque_reversal"
        elif self._motion_frame.arrival_unloading:
            rate = self.config.motion_control.arrival_unload_slew_nm_s
            reason = "gradual_arrival_unloading"
        elif self._phase is B601StagedAssistPhase.BRAKE:
            if continuous_arrival and abs(requested) > abs(previous):
                # Arrival includes a target spring and maintenance, not just
                # damping. Even a directionally opposing sum cannot authorize
                # restoring that entire drive at the fast braking slew: a
                # velocity-cut/fast-restore cycle can excite gearbox recoil.
                # Actual speed/terminal braking has already left arrival and
                # retains its original fast lane; drive removal is unchanged.
                reason = "bounded_arrival_correction_rise"
            else:
                rate = self.config.motion_control.braking_slew_nm_s
                reason = "fast_tracking_withdrawal_or_brake"
        elif (
            opposes_motion and self._stop_evidence is not None
            and self._stop_evidence.position_span_rad[self.joint_index]
            > self.config.motion_control.position_resolution_rad
        ):
            rate = self.config.motion_control.braking_slew_nm_s
            reason = "fast_resolved_motion_braking"
        elif abs(requested) < abs(previous):
            rate = (self.config.tracking_release_slew_nm_s
                    or self.config.motion_control.braking_slew_nm_s)
            reason = "normal_tracking_drive_reduction" if self.config.tracking_release_slew_nm_s else "fast_tracking_withdrawal_or_brake"
        motion = self._motion_frame
        direction = self._intent_direction
        if (self.config.motion_control.continuous_speed_limit_rad_s is not None
            and direction*previous_sent > 0 and direction*previous > 0
            and direction*requested < direction*previous
            and direction*control_v >= 0.0
            and motion.tracking_recent_velocity_rad_s is not None
            and motion.tracking_recent_position_span_rad > self.config.motion_control.position_resolution_rad
            and direction*motion.tracking_recent_velocity_rad_s >= self.config.motion_control.resolved_soft_velocity_rad_s):
            # Remove the propulsion tail within the existing owner-rate and
            # watchdog envelope. This lane cannot increase magnitude or cross
            # zero; ordinary correction/arrival unloading retains its old slew.
            # A downstream conditioner may already have reduced the actual
            # command below our preceding request. Never restore that tail
            # while claiming to withdraw it; anchor to the complete receipt.
            available = min(direction*previous, direction*previous_sent)
            self._previous_overlay_nm = direction*available
            requested = direction * min(available, max(0.0, direction*requested))
            rate = max(rate, min(self._command_rate,
                self.config.max_total_overlay_nm / self.config.motion_control.max_control_gap_sec))
            if reason != "native_zero_pending_before_torque_reversal":
                reason = "fast_resolved_drive_removal"
        self._motion_frame = replace(
            self._motion_frame, torque_transition_reason=reason,
            torque_transition_target_nm=requested, torque_transition_rate_nm_s=rate,
        )
        return self._slew(requested, rate, dt)

    def _capture_output(
        self, base_frame: B601SharedControlFrame, now: float, dt: float,
        q: float, raw_v: float, control_qvel: Sequence[float],
    ) -> float:
        cap = self.config.capture_overlay_cap_nm
        reserve = cap * self.config.motion_control.max_control_gap_sec + cap * cap / (2 * self._command_rate)
        if self._command_impulse + reserve >= self.config.breakaway_max_impulse_nm_sec:
            self._begin_release(now, q, "capture_conditioned_budget_withdrawal", terminal=True)
            return 0.0
        if self._capture_started_sec is None:
            # Use the existing transactional planner once, then retain its
            # accepted reference at CAPTURE -> TRACK. No second control loop.
            self._motion.start(now=now, q=q, target=self._intent_target_rad)
            self._capture_started_sec = now
        confirmed = (
            self._intent_direction * (q - self._breakaway_anchor_rad)
            >= self.config.motion_onset_confirm_displacement_rad
            and self._intent_direction * float(control_qvel[self.joint_index]) >= 0.0
        )
        self._motion_onset_count = self._motion_onset_count + 1 if confirmed else 0
        overlay = self._motion_output(base_frame, now, dt, q, raw_v, control_qvel, capture=True)
        if self._phase is not B601StagedAssistPhase.CAPTURE:
            self._begin_release(now, q, self._reason, terminal=True)
            return 0.0
        remaining = max(0.0, self.config.breakaway_max_impulse_nm_sec - self._breakaway_impulse_nm_sec)
        overlay = max(-remaining / dt, min(remaining / dt, overlay))
        self._breakaway_impulse_nm_sec += abs(overlay) * dt
        return overlay

    def _start_intent(
        self,
        now: float,
        target_q: float,
        *,
        direction: float,
        measured_q: float,
    ) -> None:
        self._intent_generation += 1
        self._tracking_bias_nm = 0.0
        self._tracking_recovery = None
        self._release_completed = False
        self._release_failed = False
        self._release_terminal = False
        self._release_restore = False
        self._capture_started_sec = None
        self._command_accounting = False
        self._intent_target_rad = target_q
        self._intent_started_sec = now
        self._intent_direction = direction
        self._intent_anchor_rad = measured_q
        self._phase = B601StagedAssistPhase.CUE
        self._reason = "new_policy_intent_cue"
        self._breakaway_probe = None
        self._breakaway_probe_frontier = 0.0
        self._breakaway_probe_confirmed = False
        self._authorized_generation = None
        self._breakaway_started_sec = None
        self._breakaway_anchor_rad = None
        self._breakaway_impulse_nm_sec = 0.0
        self._motion_onset_count = 0
        self._settle_started_sec = None
        self._settle_integral_torque_nm = 0.0
        self._previous_overlay_nm = 0.0
        if self._motion is not None:
            self._motion.reset()
        self._motion_frame = None
        self._drive_driver_kd = (0.0,) * 6
        with self._request_lock:
            self._authorization_request = None
            self._breakaway_revocation_requested = False

    def _consume_authorization_request(self, current_sequence: int | None) -> None:
        with self._request_lock:
            request = self._authorization_request
            self._authorization_request = None
        if request is None or self._phase is not B601StagedAssistPhase.CUE:
            return
        generation, requested_sequence, target = request
        if (
            generation == self._intent_generation
            and current_sequence is not None
            and requested_sequence <= current_sequence
            and self._intent_target_rad is not None
            and abs(target - self._intent_target_rad)
            < self.config.new_intent_delta_rad
        ):
            self._authorized_generation = generation

    def _cue_output(self, target_error: float, authority: float, dt: float) -> float:
        direction = 1.0 if target_error > 0.0 else -1.0 if target_error < 0.0 else 0.0
        requested = direction * min(self.config.cue_torque_nm, self.config.max_total_overlay_nm) * authority
        return self._slew(requested, self.config.cue_rise_slew_nm_s, dt)

    def _breakaway_output(
        self,
        base_frame: B601SharedControlFrame,
        *,
        now: float,
        dt: float,
        measured_q: float,
        measured_v: float,
        target_error: float,
        authority: float,
        control_v: float,
    ) -> float:
        direction = self._intent_direction
        anchor = float(self._breakaway_anchor_rad)
        aligned_displacement = direction * (measured_q - anchor)
        aligned_velocity = direction * measured_v
        if abs(target_error) < self.config.minimum_breakaway_error_rad:
            self._phase = B601StagedAssistPhase.DROP
            self._reason = "breakaway_cancelled_target_within_deadband"
            self._previous_overlay_nm = 0.0
            if self.config.motion_onset_confirm_displacement_rad is not None:
                self._begin_release(now, measured_q, self._reason, terminal=True)
            return 0.0
        if direction * target_error <= 0.0:
            self._phase = B601StagedAssistPhase.DROP
            self._reason = "breakaway_cancelled_target_direction_changed"
            self._previous_overlay_nm = 0.0
            if self.config.motion_onset_confirm_displacement_rad is not None:
                self._begin_release(now, measured_q, self._reason, terminal=True)
            return 0.0
        expiry_reason = self._breakaway_expiry_reason(now)
        if expiry_reason is not None:
            self._phase = B601StagedAssistPhase.AUTHORITY_EXHAUSTED
            self._reason = expiry_reason
            self._previous_overlay_nm = 0.0
            if self.config.motion_onset_confirm_displacement_rad is not None:
                self._begin_release(now, measured_q, self._reason, terminal=True)
            return 0.0
        onset_velocity = self.config.motion_onset_velocity_rad_s
        if self.config.motion_control is not None:
            # Stop the pulse as soon as the requested tracking speed is
            # reached; waiting for a higher legacy onset threshold injects
            # unnecessary energy before the slow reference can take over.
            onset_velocity = min(
                onset_velocity,
                self.config.motion_control.max_reference_velocity_rad_s,
            )
        confirmation_displacement = self.config.motion_onset_confirm_displacement_rad
        resolution = None if self.config.motion_control is None else self.config.motion_control.position_resolution_rad
        if resolution is None:
            velocity_candidate = aligned_velocity >= onset_velocity
        else:
            # A one-count return can exceed the raw onset speed while net
            # progress is zero. Keep raw hardware protection at the owner;
            # only a complete causal window with resolved range can pause here.
            evidence = self._stop_evidence
            if evidence is None or evidence.window_span_sec < self.config.motion_control.velocity_window_sec - 1e-9:
                self._begin_release(now, measured_q, "onset_window_evidence_unavailable", terminal=True)
                return 0.0
            if max(abs(v) for v in evidence.raw_qvel_rad_s) > evidence.raw_velocity_limit_rad_s:
                self._begin_release(now, measured_q, "onset_raw_velocity_hard_guard", terminal=True)
                return 0.0
            velocity_candidate = (
                direction * control_v >= onset_velocity
                and evidence.position_span_rad[self.joint_index] > resolution
            )
        candidate = (
            aligned_displacement >= self.config.motion_onset_displacement_rad
            or velocity_candidate
        )
        if self.config.breakaway_probe is not None:
            if not self._breakaway_command_budget_available(now, measured_q):
                return 0.0
            observed = self._held_breakaway_output(now, dt, measured_q, control_v, authority)
            if observed is not None:
                return observed
            if self._breakaway_probe_confirmed and self.config.onset_handoff == "track":
                return self._handoff_to_tracking(base_frame, now, dt, measured_q, measured_v)
            candidate = False  # A plateau resumes the SAME pulse, not a grant.
        # A raw sample can stop energy injection without authorizing TRACK.
        # A separately configured net displacement rejects one-count speed
        # spikes; a non-opposing window prevents reversal from granting TRACK.
        onset = self.config.breakaway_probe is None and (
            aligned_displacement >= confirmation_displacement
            and direction * control_v >= 0.0
        ) if confirmation_displacement is not None else candidate
        self._motion_onset_count = self._motion_onset_count + 1 if onset else 0
        if self._motion_onset_count >= self.config.motion_onset_confirm_samples:
            self._phase = B601StagedAssistPhase.DROP
            self._reason = "motion_onset_confirmed_drop_breakaway"
            self._previous_overlay_nm = 0.0
            if confirmation_displacement is not None:
                self._begin_release(now, measured_q, "motion_onset_conditioned_withdrawal_pending", terminal=False)
            return 0.0
        if candidate:
            # Confirmation waits with ZERO breakaway, never with a large pulse.
            # Pauses consume wall-clock duration and do not refund impulse.
            self._reason = "motion_onset_candidate_pulse_paused"
            if confirmation_displacement is not None:
                self._begin_release(now, measured_q, "motion_onset_conditioned_withdrawal_pending", terminal=False)
            return 0.0
        if (confirmation_displacement is not None and self.config.breakaway_probe is None
            and not self._breakaway_command_budget_available(now, measured_q)):
            return 0.0
        remaining_impulse = max(
            0.0,
            self.config.breakaway_max_impulse_nm_sec - self._breakaway_impulse_nm_sec,
        )
        previous_magnitude = max(0.0, direction * self._previous_overlay_nm)
        requested_magnitude = min(
            self.config.breakaway_torque_nm * authority,
            previous_magnitude + self.config.breakaway_rise_slew_nm_s * dt,
            remaining_impulse / dt,
        )
        overlay = direction * requested_magnitude
        self._breakaway_impulse_nm_sec += abs(overlay) * dt
        self._reason = (
            "held_probe_quiet_resume_same_budget"
            if self._breakaway_probe is not None and self._breakaway_probe.get("ended_at_sec") == now
            else "bounded_breakaway_pulse"
        )
        return overlay

    def _handoff_to_tracking(
        self, base: B601SharedControlFrame, now: float, dt: float, q: float, raw_v: float,
    ) -> float:
        """Transfer one confirmed normal onset to the existing tracking servo.

        Seed its bounded maintenance memory from the latest complete dispatch,
        not the nominal pulse cap. Velocity feedback and responsive withdrawal
        remove excess demand; the automatic session keeps charging the original
        leg/batch exposure budgets through this handoff.
        """
        sent = self._command_delta[self.joint_index]
        cap = self.config.tracking_overlay_cap_nm
        if (not self._tracking_recovery_permitted or sent*self._intent_direction <= 0
            or abs(sent) > cap + 1e-9
            or any(abs(v) > 1e-9 for i, v in enumerate(self._command_delta) if i != self.joint_index)):
            self._begin_release(now, q, "tracking_handoff_dispatch_or_permission_invalid", terminal=True)
            return 0.0
        self._command_accounting = False
        self._tracking_bias_nm = sent
        self._previous_overlay_nm = sent
        self._phase = B601StagedAssistPhase.TRACK
        self._motion.start(now=now, q=q, target=self._intent_target_rad,
            preserve_velocity_history=self.config.motion_control.continuous_speed_limit_rad_s is not None)
        overlay = self._motion_output(base, now, dt, q, raw_v, self._stop_evidence.window_qvel_rad_s,
            protect_maintenance=False)
        if self._phase is B601StagedAssistPhase.TRACK:
            # A one-shot breakaway pulse is not a validated running load.
            # Keep the existing fast onset DROP and do not retain the removed
            # pulse tail as latent protected maintenance on the next tick.
            if self.config.motion_control.continuous_speed_limit_rad_s is not None:
                self._tracking_bias_nm = self._intent_direction*min(abs(self._tracking_bias_nm),
                    max(0.0, self._intent_direction*overlay))
            self._reason = "confirmed_onset_transferred_to_tracking"
            self._tracking_recovery["handoff_seed_command_nm"] = sent
        return overlay

    def _breakaway_command_budget_available(self, now: float, q: float) -> bool:
        if self._command_delta is None or self._command_rate is None:
            self._begin_release(now, q, "conditioned_dispatch_unavailable", terminal=True)
            return False
        peak = max(self.config.breakaway_torque_nm, abs(self._command_delta[self.joint_index]))
        reserve = peak*self.config.motion_control.max_control_gap_sec + peak*peak/(2*self._command_rate)
        if self._command_impulse + reserve >= self.config.breakaway_max_impulse_nm_sec:
            self._begin_release(now, q, "conditioned_command_budget_withdrawal", terminal=True)
            return False
        return True

    def _held_breakaway_output(self, now: float, dt: float, q: float, v: float, authority: float) -> float | None:
        """Observe actual held commands without renewing any pulse budget."""
        config, motion, evidence = self.config.breakaway_probe, self.config.motion_control, self._stop_evidence
        clock_now = max(now, self._command_clock)
        direction = self._intent_direction
        displacement = direction * (q - self._breakaway_anchor_rad)
        def withdraw(reason: str, *, terminal: bool = True) -> float:
            self._begin_release(now, q, reason, terminal=terminal)
            return 0.0
        if (evidence is None or evidence.timestamp_sec != now
            or evidence.window_span_sec < motion.velocity_window_sec - 1e-9
            or max(abs(x) for x in evidence.raw_qvel_rad_s) > evidence.raw_velocity_limit_rad_s):
            return withdraw("held_probe_feedback_unavailable_or_unsafe")
        if clock_now-self._breakaway_started_sec >= config.max_trial_sec - 1e-9 or self._command_impulse >= config.max_command_exposure_nm_sec:
            return withdraw("held_probe_budget_exhausted")
        if abs(displacement) >= config.max_displacement_rad:
            return withdraw("held_probe_cumulative_displacement_withdrawn")
        speed_limit = motion.resolved_soft_velocity_rad_s
        probe = self._breakaway_probe
        if probe is not None and probe["stage"] == "hold":
            completed = probe["first_hold_completed_at_sec"]
            fresh = completed is not None and now > completed and now >= self._command_clock
            progress = observe_held_progress(probe, now=now, q=q, direction=direction,
                post_hold=None if fresh else False)
            if completed is None:
                if now-probe["entered_at_sec"] >= motion.max_control_gap_sec:
                    return withdraw("held_probe_dispatch_timeout")
            elif clock_now-completed >= config.observation_limit_sec:
                return withdraw("held_probe_observation_timeout")
            if fresh:
                if progress < -2*J3_HELD_PROGRESS_STEP_RAD:
                    return withdraw("held_probe_opposing_motion")
                observe_held_progress(probe, now=now, q=q, direction=direction, post_hold=True)
                self._motion_onset_count = probe["distinct_progress_samples"]
                if (self._motion_onset_count >= self.config.motion_onset_confirm_samples
                    and displacement >= self.config.motion_onset_confirm_displacement_rad and direction*v >= 0):
                    self._breakaway_probe_confirmed = True
                    if self.config.onset_handoff == "track":
                        # The observer has confirmed new, post-dispatch frontiers.
                        # Normal execution retains the accepted maintenance command;
                        # unloading belongs to diagnostic trials and terminal stops.
                        probe.update(stage="tracking_handoff", outcome="distinct_progress_confirmed",
                                     ended_at_sec=now)
                        return None
                    return withdraw("held_probe_distinct_progress_confirmed", terminal=False)
            if abs(v) >= speed_limit and evidence.position_span_rad[self.joint_index] > motion.position_resolution_rad:
                if (self.config.breakaway_soft_withdrawal_confirmation and fresh
                    and direction*v > 0
                    and displacement >= self.config.motion_onset_confirm_displacement_rad
                    and probe["distinct_progress_samples"] > 0):
                    # Withdraw propulsion NOW; new causal frontiers may still
                    # complete confirmation while the original brake stops it.
                    self._begin_release(now, q, "held_probe_soft_speed_confirmation_pending", terminal=False)
                    probe["withdrawal_confirmation_pending"] = True
                    return 0.0
                return withdraw("held_probe_soft_speed_withdrawn_unconfirmed")
            if fresh and now-completed >= config.hold_sec:
                plateau = probe["last_progress_at_sec"] or probe["first_feedback_at_sec"]
                if now-plateau >= motion.velocity_window_sec and all(
                    span <= motion.position_resolution_rad and abs(speed) <= self.config.stuck_max_velocity_rad_s
                    for span, speed in zip(evidence.position_span_rad, evidence.window_qvel_rad_s, strict=True)
                ):
                    probe.update(stage="quiet_resume", outcome="quiet_resume", ended_at_sec=now)
                    self._breakaway_probe_frontier = max(self._breakaway_probe_frontier,
                        direction*(probe["candidate_qpos_rad"]-self._breakaway_anchor_rad)+probe["maximum_progress_rad"])
                    self._motion_onset_count = 0
                    self._reason = "held_probe_quiet_resume_same_budget"
                    return None
            held = direction * min(abs(probe["hold_nm"]), self.config.breakaway_torque_nm*authority)
            if not math.isclose(held, probe["hold_nm"], rel_tol=0, abs_tol=1e-9):
                return withdraw("held_probe_authority_changed")
            if self._breakaway_impulse_nm_sec + abs(held)*dt >= self.config.breakaway_max_impulse_nm_sec:
                return withdraw("breakaway_impulse_budget_exhausted")
            self._breakaway_impulse_nm_sec += abs(held)*dt
            self._reason = "held_probe_observing_actual_command"
            return held
        if abs(v) >= speed_limit and evidence.position_span_rad[self.joint_index] > motion.position_resolution_rad:
            return withdraw("held_probe_soft_speed_withdrawn_unconfirmed")
        if displacement >= max(self.config.motion_onset_displacement_rad, self._breakaway_probe_frontier+J3_HELD_PROGRESS_STEP_RAD):
            actual = self._command_delta[self.joint_index]
            held = direction*min(abs(actual), abs(self._previous_overlay_nm), self.config.breakaway_torque_nm*authority)
            if actual*direction <= 0 or held*direction <= 0:
                return withdraw("held_probe_command_unconfirmed")
            if self._breakaway_impulse_nm_sec + abs(held)*dt >= self.config.breakaway_max_impulse_nm_sec:
                return withdraw("breakaway_impulse_budget_exhausted")
            self._breakaway_probe = begin_held_progress(now, q, held)
            self._breakaway_probe_frontier = displacement
            self._breakaway_impulse_nm_sec += abs(held)*dt
            self._reason = "held_probe_pending_actual_dispatch"
            return held
        return None

    def _observe_soft_withdrawal(self, now: float, q: float, velocity: float) -> None:
        """Reuse held frontiers during braking, never restart a pulse or clock."""
        probe, evidence = self._breakaway_probe, self._stop_evidence
        motion = self.config.motion_control
        if not self._tracking_recovery_permitted:
            self._begin_release(now, q, "tracking_recovery_permission_withdrawn", terminal=True)
            return
        if (evidence is None or evidence.timestamp_sec != now
            or evidence.window_span_sec < motion.velocity_window_sec - 1e-9
            or max(abs(v) for v in evidence.raw_qvel_rad_s) > evidence.raw_velocity_limit_rad_s):
            self._begin_release(now, q, "held_probe_withdrawal_feedback_unsafe", terminal=True)
            return
        direction = self._intent_direction
        displacement = direction*(q-self._breakaway_anchor_rad)
        if abs(displacement) >= self.config.breakaway_probe.max_displacement_rad:
            self._begin_release(now, q, "held_probe_cumulative_displacement_withdrawn", terminal=True)
            return
        completed = probe["first_hold_completed_at_sec"]
        fresh = completed is not None and now > completed and now >= self._command_clock
        progress = observe_held_progress(probe, now=now, q=q, direction=direction,
                                         post_hold=fresh)
        if progress < -2*J3_HELD_PROGRESS_STEP_RAD or (
            probe["maximum_progress_rad"]-progress > 2*J3_HELD_PROGRESS_STEP_RAD
        ):
            self._begin_release(now, q, "held_probe_withdrawal_progress_returned", terminal=True)
            return
        if (fresh and probe["distinct_progress_samples"] >= self.config.motion_onset_confirm_samples
            and displacement >= self.config.motion_onset_confirm_displacement_rad
            and direction*velocity >= 0):
            self._breakaway_probe_confirmed = True
            self._motion_onset_count = probe["distinct_progress_samples"]
            self._release_reason = "held_probe_post_withdrawal_progress_confirmed"
            probe["outcome"] = self._release_reason

    def _tracking_or_settle_output(
        self,
        base_frame: B601SharedControlFrame,
        *,
        now: float,
        dt: float,
        target_error: float,
        measured_v: float,
    ) -> float:
        in_capture_region = abs(target_error) <= self.config.settle_capture_error_rad
        nearly_stationary = abs(measured_v) <= self.config.settle_velocity_rad_s
        if self._phase is B601StagedAssistPhase.SETTLE and not in_capture_region:
            self._phase = B601StagedAssistPhase.TRACK
            self._reason = "settle_conditions_lost"
            self._settle_started_sec = None
        elif (
            self._phase is B601StagedAssistPhase.TRACK
            and in_capture_region
            and nearly_stationary
        ):
            if self._settle_started_sec is None:
                self._settle_started_sec = now
            elif now - self._settle_started_sec >= self.config.settle_dwell_sec:
                self._phase = B601StagedAssistPhase.SETTLE
                self._reason = "settled_with_bounded_low_gain_assist"
        elif self._phase is B601StagedAssistPhase.TRACK:
            self._settle_started_sec = None
        if self._phase is B601StagedAssistPhase.SETTLE:
            base_torque = self._base_overlay(base_frame)
            if nearly_stationary and abs(target_error) > self.config.settle_error_rad:
                candidate_integral = self._settle_integral_torque_nm + (
                    self.config.settle_integral_gain_nm_per_rad_sec
                    * target_error
                    * dt
                )
                would_saturate_with_error = (
                    target_error > 0.0
                    and base_torque + candidate_integral
                    > self.config.settle_overlay_cap_nm
                ) or (
                    target_error < 0.0
                    and base_torque + candidate_integral
                    < -self.config.settle_overlay_cap_nm
                )
                if not would_saturate_with_error:
                    self._settle_integral_torque_nm = candidate_integral
            else:
                self._settle_integral_torque_nm *= math.exp(
                    -self.config.settle_integral_leak_per_sec * dt
                )
            self._settle_integral_torque_nm = max(
                -self.config.settle_integral_cap_nm,
                min(
                    self.config.settle_integral_cap_nm,
                    self._settle_integral_torque_nm,
                ),
            )
            requested = base_torque + self._settle_integral_torque_nm
            requested = max(-self.config.settle_overlay_cap_nm, min(self.config.settle_overlay_cap_nm, requested))
        else:
            self._settle_integral_torque_nm *= math.exp(
                -self.config.settle_integral_leak_per_sec * dt
            )
            requested = self._base_overlay(base_frame)
            requested = max(-self.config.tracking_overlay_cap_nm, min(self.config.tracking_overlay_cap_nm, requested))
        return self._slew(requested, self.config.tracking_slew_nm_s, dt)

    def _tracking_output(self, base_frame: B601SharedControlFrame, dt: float) -> float:
        requested = self._base_overlay(base_frame)
        requested = max(-self.config.tracking_overlay_cap_nm, min(self.config.tracking_overlay_cap_nm, requested))
        return self._slew(requested, self.config.tracking_slew_nm_s, dt)

    def _base_overlay(self, base_frame: B601SharedControlFrame) -> float:
        return base_frame.decision.assist_torque_nm[self.joint_index]

    def _slew(self, requested: float, rate: float, dt: float) -> float:
        delta = rate * dt
        return max(self._previous_overlay_nm - delta, min(self._previous_overlay_nm + delta, requested))

    def _reset(self, phase: B601StagedAssistPhase, reason: str) -> None:
        self._tracking_bias_nm = 0.0
        self._tracking_recovery = None
        self._phase = phase
        self._release_completed = False
        self._release_active = False
        self._capture_started_sec = None
        self._command_accounting = False
        self._reason = reason
        self._intent_target_rad = None
        self._intent_started_sec = None
        self._intent_direction = 0.0
        self._authorized_generation = None
        self._breakaway_started_sec = None
        self._breakaway_anchor_rad = None
        self._breakaway_impulse_nm_sec = 0.0
        self._motion_onset_count = 0
        self._settle_started_sec = None
        self._settle_integral_torque_nm = 0.0
        self._previous_overlay_nm = 0.0
        self._breakaway_probe = None
        self._breakaway_probe_frontier = 0.0
        self._breakaway_probe_confirmed = False
        if self._motion is not None:
            self._motion.reset()
        self._motion_frame = None
        with self._request_lock:
            self._brake_requested = False
            self._authorization_request = None
            self._breakaway_revocation_requested = False

    def _frame(
        self,
        base_frame: B601SharedControlFrame,
        selected_overlay_nm: float | None,
    ) -> B601StagedAssistFrame:
        overlay = [0.0] * 6
        if selected_overlay_nm is not None:
            overlay[self.joint_index] = selected_overlay_nm
        motion = self.config.motion_control
        if motion is not None and self._phase in {
            B601StagedAssistPhase.CUE, B601StagedAssistPhase.BREAKAWAY,
            B601StagedAssistPhase.DROP, B601StagedAssistPhase.TRACK,
            B601StagedAssistPhase.BRAKE, B601StagedAssistPhase.SOFT_HOLD,
            B601StagedAssistPhase.RELEASING,
            B601StagedAssistPhase.CAPTURE,
        }:
            self._drive_driver_kd = slew_driver_damping(
                self._drive_driver_kd, motion.drive_driver_kd_nm_s_per_rad,
                rate_per_sec=motion.drive_damping_rise_per_sec, dt_sec=self._tick_dt,
            )
        else:
            # Explicit withdrawal/stale/fault restores the calibrated base Kd.
            self._drive_driver_kd = (0.0,) * 6
        requested_driver_kd = (
            (0.0,) * 6 if self._release_active and self._release_restore else self._drive_driver_kd
        )
        reference = replace(
            base_frame.reference,
            joint_torque_overlay_nm=tuple(overlay),
        )
        self._last_output_sequence = reference.sequence
        self._last_output_timestamp = reference.receive_timestamp_sec
        self._last_output_zero = not any(overlay)
        with self._request_lock:
            self._published_phase = self._phase
            self._published_generation = self._intent_generation
        return B601StagedAssistFrame(
            reference=reference,
            base_frame=base_frame,
            phase=self._phase,
            reason=self._reason,
            joint_name=self.config.joint_name,
            policy_sequence=base_frame.decision.state.last_policy_sequence,
            intent_generation=self._intent_generation,
            overlay_torque_nm=tuple(overlay),
            breakaway_impulse_nm_sec=self._breakaway_impulse_nm_sec,
            settle_integral_torque_nm=self._settle_integral_torque_nm,
            motion=self._motion_frame,
            drive_driver_kd_nm_s_per_rad=requested_driver_kd,
            support_max_velocity_rad_s=self._support_max_velocity,
            support_raw_max_velocity_rad_s=self._support_raw_max_velocity,
            motion_onset_confirm_count=self._motion_onset_count,
            motion_onset_required_samples=self.config.motion_onset_confirm_samples,
            release_terminal=self._release_terminal,
            release_completed=self._release_completed,
            release_failed=self._release_failed,
            release_restoring_base_damping=self._release_restore,
            conditioned_command_delta_nm=self._command_delta,
            conditioned_command_impulse_nm_sec=self._command_impulse,
            breakaway_probe=None if self._breakaway_probe is None else dict(self._breakaway_probe),
            tracking_recovery=None if self._tracking_recovery is None else dict(self._tracking_recovery),
        )


__all__ = [
    "B601StagedAssistConfig",
    "B601StagedAssistFrame",
    "B601StagedAssistPhase",
    "B601StagedSingleJointAssistRuntime",
]
