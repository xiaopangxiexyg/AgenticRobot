"""One owner-held positioning/acquisition/return cycle; never actuator I/O."""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import asdict, dataclass, replace
from typing import Any

from robot_motion.rebot_b601_rs.calibration import CalibrationCondition, finite, six

from .automatic_calibration import B601AutomaticCalibrationController
from .calibration_session import (
    B601AutomaticCalibrationSession,
    CalibrationBatchLimits,
    calibration_outcome,
    calibration_receipt_rejection,
)
from .staged_controller import B601StagedTorqueConsoleController

ZERO = (0.0,) * 6


@dataclass(frozen=True)
class CalibrationCycleConfig:
    offset_rad: float = math.radians(20)
    positioning_velocity_rad_s: float = 0.010
    positioning_timeout_sec: float = 180.0
    positioning_command_exposure_nm_sec: float = 80.0
    background_permit_timeout_sec: float = 70.0
    gravity_observation_sec: float = 5.0
    gravity_drift_limit_rad: float = math.radians(1)
    max_cycle_sec: float = 900.0
    supported_origin_disable_confirmation: bool = False
    positioning_support_soft_speed_recovery: bool = False
    positioning_continuous_speed_limit_rad_s: float | None = None
    positioning_torque_rise_nm_s: float | None = None
    positioning_torque_fall_nm_s: float | None = None

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name in {"positioning_continuous_speed_limit_rad_s", "positioning_torque_rise_nm_s", "positioning_torque_fall_nm_s"} and value is None:
                continue
            if name in {"supported_origin_disable_confirmation", "positioning_support_soft_speed_recovery"}:
                if type(value) is not bool:
                    raise TypeError(f"{name} must be boolean")
            elif isinstance(value, bool) or finite(value, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.offset_rad > math.radians(20) + 1e-12:
            raise ValueError("cycle positioning offset must not exceed 20 degrees")
        if (
            self.gravity_observation_sec < 0.08
            or self.gravity_drift_limit_rad > math.radians(3)
        ):
            raise ValueError(
                "gravity observation requires a full window and drift within three degrees"
            )
        if (
            self.positioning_timeout_sec
            <= self.offset_rad / self.positioning_velocity_rad_s
        ):
            raise ValueError("positioning timeout cannot cover the reference travel")


class B601AutomaticCalibrationCycleController:
    """Sequence existing intent controllers, with one active dispatch consumer.

    The original hardware owner retains gravity, admission, final limiting and
    disable. GUI start grants one finite cycle; contact/lost lease never starts
    an automatic return. A known acquisition warning may return after its native
    zero/base-damping stop, without rewriting the acquisition result.
    """

    controller_mode = "automatic_calibration"

    def __init__(
        self,
        calibration: B601AutomaticCalibrationController,
        cycle: CalibrationCycleConfig,
    ) -> None:
        if not isinstance(
            calibration, B601AutomaticCalibrationController
        ) or not isinstance(cycle, CalibrationCycleConfig):
            raise TypeError(
                "cycle requires the existing calibration controller and explicit cycle settings"
            )
        self.calibration, self.cycle = calibration, cycle
        self.maximum_offset_rad = cycle.offset_rad
        self.config, self.limits = calibration.config, calibration.limits
        if cycle.positioning_torque_rise_nm_s is not None:
            self.limits = replace(self.limits, rise_slew_nm_s=cycle.positioning_torque_rise_nm_s)
        if (
            self.config.motion_acquisition_arrival_tolerance_rad is None
            or self.config.motion_plan != "local_identification"
        ):
            raise ValueError(
                "cycle requires explicit local identification and arrival tolerance"
            )
        if (
            cycle.positioning_velocity_rad_s
            > calibration.settings.staged_assist.motion_control.max_reference_velocity_rad_s
        ):
            raise ValueError("cycle speed exceeds the existing motion profile")
        if (
            cycle.max_cycle_sec
            < 2 * cycle.positioning_timeout_sec
            + self.config.max_batch_sec
            + 2 * cycle.gravity_observation_sec
        ):
            raise ValueError(
                "cycle deadline must cover both moves, acquisition and observations"
            )
        span = (
            cycle.offset_rad
            + self.config.global_excursion_rad
            + self.config.motion_acquisition_arrival_tolerance_rad
        )
        if span > 0.405:
            raise ValueError("cycle absolute excursion must remain within 0.405 rad")
        # The full-cycle controller has a wider, cycle-specific envelope than
        # the underlying fixed-position controller. Keep that envelope in the
        # canonical limits so pre-start lease changes cannot accidentally
        # replace it with the narrower base profile and fail settings checks.
        self.limits = replace(self.limits, max_excursion_rad=span)
        self._base_limits = self.limits
        self.settings = replace(
            calibration.settings,
            limits=self.limits,
            maximum_target_offset_rad=span,
            staged_assist=replace(
                calibration.settings.staged_assist,
                tracking_slew_nm_s=(calibration.settings.staged_assist.tracking_slew_nm_s
                    if cycle.positioning_torque_rise_nm_s is None else cycle.positioning_torque_rise_nm_s),
                tracking_release_slew_nm_s=cycle.positioning_torque_fall_nm_s,
                motion_control=replace(
                    calibration.settings.staged_assist.motion_control,
                    support_soft_speed_recovery=cycle.positioning_support_soft_speed_recovery,
                    continuous_speed_limit_rad_s=cycle.positioning_continuous_speed_limit_rad_s,
                ),
            ),
        )
        self._position = B601StagedTorqueConsoleController(self.settings)
        self._background_mode_enabled = False
        self._page_visible = True
        # Background timers can be throttled far more aggressively than the
        # foreground lease. Keep a finite crash fallback while allowing one
        # minute-scale browser timer interval; pagehide still withdraws now.
        self._background_permit_timeout_sec = max(
            self.limits.command_timeout_sec, cycle.background_permit_timeout_sec
        )
        self._session: B601AutomaticCalibrationSession | None = None
        self._lock = threading.RLock()
        self._phase, self._reason = "idle", "not_started"
        self._ready = False
        self._human: bool | None = None
        self._permit = None
        self._start_requested = False
        self._cancel_requested: str | None = None
        self._cancel_withdraw_source = "owner_cancel"
        self._using_calibration = False
        self._condition: CalibrationCondition | None = None
        self._position_limits = None
        self._origin = self._q = None
        self._started = self._phase_started = None
        self._pending = None
        self._last_feedback = None
        self._receipt = None
        self._last_sent_delta = ZERO
        self._exposure = 0.0
        self._zero_base = False
        self._observation_anchor = None
        self._observation_started = None
        self._observations = []
        self._observation_min = self._observation_max = None
        self._moves = []
        self._last_positioning_motion = None
        self._records = []
        self._events = []

    def set_background_mode_enabled(self, enabled: bool) -> None:
        """Use a bounded browser-throttle lease only for the full-cycle mode."""
        if type(enabled) is not bool:
            raise TypeError("background mode must be boolean")
        with self._lock:
            if self._active or self._start_requested:
                raise ValueError("background behavior is locked after cycle start")
            timeout = (
                self._background_permit_timeout_sec
                if enabled else self._base_limits.command_timeout_sec
            )
            self.calibration.set_command_timeout_sec(timeout)
            self.limits = replace(self._base_limits, command_timeout_sec=timeout)
            self.settings = replace(self.settings, limits=self.limits)
            self._position.set_command_timeout_sec(timeout)
            self._background_mode_enabled = enabled

    @property
    def _active(self):
        return self._phase not in {"idle", "complete", "failed", "cancelled"}

    @property
    def cycle_disable_confirmation_ready(self) -> bool:
        with self._lock:
            return (
                self.cycle.supported_origin_disable_confirmation
                and self._phase == "complete"
                and self._zero_base
                and self._pending is None
                and self._human is False
            )

    @property
    def cycle_disable_confirmation_required(self) -> bool:
        return self.cycle.supported_origin_disable_confirmation

    def configure_context(
        self,
        condition: CalibrationCondition,
        position_limits_rad: tuple[tuple[float, float], ...],
    ) -> None:
        with self._lock:
            self.calibration.configure_context(condition, position_limits_rad)
            self._condition, self._position_limits = condition, position_limits_rad

    def owner_ready(self) -> None:
        with self._lock:
            self._ready = True
            self._position.owner_ready()
            self.calibration.owner_ready()

    def set_human_intervention(self, human: bool) -> None:
        if type(human) is not bool:
            raise TypeError("human intervention must be boolean")
        with self._lock:
            self._human = human
            self.calibration.set_human_intervention(human)
            if human:
                self._start_requested = False
                if self._active:
                    self._cancel_requested = "human_intervention"
                    self._cancel_withdraw_source = "owner_cancel"

    def submit(self, action: str, *, now_sec: float, **fields: Any) -> None:
        now = finite(now_sec, "intent time")
        with self._lock:
            if action == "arm":
                allowed = {"cycle_angle_deg"}
            elif action == "withdraw":
                allowed = {"withdraw_source"}
            elif action == "visibility":
                allowed = {"visible"}
            else:
                allowed = set()
            if set(fields) - allowed or (action == "arm" and fields and set(fields) != allowed):
                raise ValueError("cycle accepts no manual target or torque fields")
            if action == "arm":
                if (
                    self._phase != "idle"
                    or self._start_requested
                    or not self._ready
                    or self._human is not False
                    or self._condition is None
                ):
                    raise ValueError(
                        "cycle start requires ready owner and declared no intervention"
                    )
                angle = fields.get("cycle_angle_deg", math.degrees(self.cycle.offset_rad))
                if (isinstance(angle, bool) or not isinstance(angle, (int, float))
                    or not math.isfinite(angle) or not 0 < angle <= math.degrees(self.maximum_offset_rad)):
                    raise ValueError(f"cycle angle must be numeric, positive and at most {math.degrees(self.maximum_offset_rad):g} degrees")
                # The owner already admitted the maximum envelope before CAN.
                # Select a smaller goal atomically with start; never rebuild an
                # active controller, renew budgets or change the return origin.
                self.cycle = replace(self.cycle, offset_rad=math.radians(angle))
                self._start_requested, self._permit = True, now
            elif action == "heartbeat":
                if not self._active and not self._start_requested:
                    raise ValueError("cycle heartbeat requires an active cycle")
                self._permit = now
                if self._using_calibration:
                    self.calibration.submit("heartbeat", now_sec=now)
            elif action == "visibility":
                visible = fields.get("visible")
                if type(visible) is not bool:
                    raise ValueError("visibility intent requires a boolean visible field")
                if not self._background_mode_enabled:
                    raise ValueError("background visibility is available only for the full-cycle mode")
                if not self._active and not self._start_requested:
                    raise ValueError("visibility heartbeat requires a pending or active cycle")
                self._page_visible = visible
                self._permit = now
                self._records.append({
                    "kind": "automatic_cycle_visibility",
                    "recorded_monotonic_sec": now,
                    "visible": visible,
                    "mode": "background_permit" if not visible else "foreground_permit",
                })
                if self._using_calibration:
                    self.calibration.submit("heartbeat", now_sec=now)
            elif action in {"zero", "withdraw"}:
                self._start_requested = False
                if self._active:
                    source = fields.get("withdraw_source", "operator_button")
                    reasons = {
                        "operator_button": "operator_withdraw",
                        "page_unload": "page_unload",
                        "page_hidden": "page_hidden",
                        "owner_cancel": "owner_cancel",
                    }
                    if action == "zero":
                        reason = "operator_zero"
                    elif source not in reasons:
                        raise ValueError("unsupported withdrawal source")
                    else:
                        reason = reasons[source]
                    if self._cancel_requested is None:
                        self._cancel_requested = reason
                        self._cancel_withdraw_source = source
            else:
                raise ValueError("unsupported cycle action")

    def _transition(self, phase, reason, now):
        self._phase, self._reason, self._phase_started = phase, reason, now
        event = {"time_sec": now, "phase": phase, "reason": reason}
        self._events.append(event)
        self._records.append(
            {
                "kind": "automatic_calibration_cycle_transition",
                "recorded_monotonic_sec": now,
                **event,
            }
        )

    def _fail(self, reason, now):
        # Keep the first causal guard, not a later generic child-withdrawal
        # label from the same tick. Withdrawal and zero/base checks still run.
        if self._phase != "failed":
            self._transition("failed", reason, now)
        self._zero_base = False

    def _make_move(self, target, now, role):
        # A new phase starts only after a complete zero/base receipt. It does
        # not renew a local acquisition budget or a failed target.
        anchor = self._q
        self._last_positioning_motion = None
        target_q = list(anchor)
        target_q[2] = target
        if any(
            not lo <= v <= hi
            for v, (lo, hi) in zip(target_q, self._position_limits, strict=True)
        ):
            raise ValueError("cycle target outside model joint limits")
        plan = {
            "schema": "b601.local-calibration-plan.v1",
            "execution_enabled": False,
            "condition": asdict(self._condition),
            "joint_name": "joint3",
            "anchor_qpos_rad": anchor,
            "steps": [
                {
                    "trial_id": role,
                    "role": role,
                    "target_qpos_rad": tuple(target_q),
                    "max_reference_velocity_rad_s": self.cycle.positioning_velocity_rad_s,
                }
            ],
        }
        self._session = B601AutomaticCalibrationSession(
            self._position,
            plan=plan,
            condition=self._condition,
            position_limits_rad=self._position_limits,
            limits=CalibrationBatchLimits(
                self.cycle.positioning_timeout_sec,
                self.cycle.positioning_timeout_sec,
                self.config.prepare_timeout_sec,
                self.limits.command_timeout_sec,
                self.settings.staged_assist.motion_control.max_control_gap_sec,
                self.cycle.positioning_command_exposure_nm_sec,
                self.cycle.positioning_command_exposure_nm_sec,
            ),
            acquisition_arrival_tolerance_rad=self.config.motion_acquisition_arrival_tolerance_rad,
        )
        self._transition(
            "positioning_out" if role == "positioning_not_for_fit" else "returning",
            "waiting_for_native_zero_and_still_before_move",
            now,
        )

    def _start_observation(self, now, returning=False):
        self._session = None
        self._observation_started = None
        self._observation_anchor = None
        self._observation_min = self._observation_max = None
        self._transition(
            "return_observe" if returning else "gravity_observe",
            "gravity_only_observation",
            now,
        )

    def step(
        self,
        *,
        now_sec: float,
        qpos_rad: tuple[float, ...],
        qvel_rad_s: tuple[float, ...],
        feedback_sequence: int,
        control_now_sec: float,
    ) -> tuple[tuple[float, ...], dict[str, Any]]:
        now, clock = (
            finite(now_sec, "feedback time"),
            finite(control_now_sec, "control time"),
        )
        q = six(qpos_rad, "cycle position")
        velocity = six(qvel_rad_s, "cycle velocity")
        with self._lock:
            if self._pending is not None:
                raise ValueError("cycle previous six-axis dispatch not confirmed")
            if (
                not now
                <= clock
                <= now + self.settings.staged_assist.motion_control.max_control_gap_sec
            ):
                raise ValueError(
                    "cycle observation clock does not follow fresh feedback"
                )
            self._q = q
            previous_feedback = self._last_feedback
            if previous_feedback is not None and now <= previous_feedback:
                raise ValueError("cycle feedback clock must advance")
            self._last_feedback = now
            if self._start_requested:
                self._start_requested = False
                self._origin, self._started = q, clock
                target = q[2] + self.cycle.offset_rad
                # Validate the complete local-test range BEFORE any propulsion.
                lo, hi = self._position_limits[2]
                if (
                    not lo
                    <= target - self.config.global_excursion_rad
                    <= target + self.config.global_excursion_rad
                    <= hi
                ):
                    self._fail(
                        "cycle_target_and_test_range_outside_joint_limits", clock
                    )
                else:
                    self._make_move(target, clock, "positioning_not_for_fit")
            if self._active:
                gap = self.settings.staged_assist.motion_control.max_control_gap_sec
                if previous_feedback is not None and now - previous_feedback > gap:
                    self._fail("cycle_feedback_gap_exceeded", clock)
                elif (
                    not self._using_calibration
                    and max(abs(v) for v in velocity)
                    > self.limits.max_joint_velocity_rad_s
                ):
                    self._fail("cycle_raw_velocity_exceeded", clock)
                elif self._human is not False:
                    self._cancel_requested = "human_intervention"
                    self._cancel_withdraw_source = "owner_cancel"
                elif self._cancel_requested is None and (
                    self._permit is None
                    or not 0 <= clock - self._permit <= self.limits.command_timeout_sec
                ):
                    self._cancel_requested = "gui_permit_expired"
                    self._cancel_withdraw_source = "owner_cancel"
                elif self._cancel_requested is None and clock - self._started > self.cycle.max_cycle_sec:
                    self._cancel_requested = "cycle_timeout"
                    self._cancel_withdraw_source = "owner_cancel"
                elif (
                    not self._origin[2] - self.config.global_excursion_rad
                    <= q[2]
                    <= self._origin[2] + self.settings.maximum_target_offset_rad
                ):
                    self._fail("cycle_absolute_excursion", clock)
                elif any(
                    abs(q[i] - self._origin[i]) > self.limits.max_support_drift_rad
                    for i in range(6)
                    if i != 2
                ):
                    self._fail("cycle_support_excursion", clock)
                elif self._cancel_requested is None and (
                    self._exposure + 6 * self.limits.max_abs_assist_nm * gap
                    >= 2 * self.cycle.positioning_command_exposure_nm_sec
                    + self.config.max_sweep_command_exposure_nm_sec
                ):
                    self._cancel_requested = "cycle_command_exposure_budget"
                if self._cancel_requested is not None and self._phase != "stopping":
                    if self._using_calibration:
                        self.calibration.submit(
                            "withdraw",
                            now_sec=clock,
                            withdraw_source=self._cancel_withdraw_source,
                        )
                    elif self._session is not None:
                        self._session.cancel(
                            now_sec=clock, reason=self._cancel_requested
                        )
                    else:
                        self._position.submit(
                            "withdraw",
                            now_sec=clock,
                            withdraw_source=self._cancel_withdraw_source,
                        )
                    self._transition("stopping", self._cancel_requested, clock)
            self._pending = (
                feedback_sequence,
                now,
                self._using_calibration,
                self._session,
            )
            if self._using_calibration:
                overlay, audit = self.calibration.step(
                    now_sec=now,
                    qpos_rad=q,
                    qvel_rad_s=qvel_rad_s,
                    feedback_sequence=feedback_sequence,
                    control_now_sec=clock,
                )
            elif self._session is not None:
                overlay, audit = self._session.step(
                    now_sec=now,
                    qpos_rad=q,
                    qvel_rad_s=qvel_rad_s,
                    feedback_sequence=feedback_sequence,
                    condition=self._condition,
                    human_intervention=self._human,
                    operator_permit_timestamp_sec=self._permit,
                    operator_observed_at_sec=clock,
                )
            else:
                overlay, audit = self._position.step(
                    now_sec=now, qpos_rad=q, qvel_rad_s=qvel_rad_s
                )
            if (not self._using_calibration and audit.get("motion_control") is not None
                and (audit.get("calibration_trial") or {}).get("state") == "moving"):
                # One diagnostic snapshot survives withdrawal resetting the
                # child controller/counter for its zero-only release. It never
                # participates in control decisions.
                self._last_positioning_motion = {
                    **audit["motion_control"], "observed_at_sec": now,
                }
            if audit["phase"] == "fault":
                self._fail(str(audit["fault"]), clock)
            if self._phase == "failed":
                overlay = ZERO
                audit = {**audit, "phase": "fault", "fault": self._reason}
            return overlay, {**audit, "automatic_calibration": self._batch_status()}

    def validate_final_command(self, **values: Any) -> None:
        with self._lock:
            if self._using_calibration:
                self.calibration.validate_final_command(**values)

    def active_driver_damping(
        self,
        base_kd: tuple[float, ...],
        *,
        final_command_delta_nm: tuple[float, ...] | None = None,
    ) -> tuple[float, ...]:
        with self._lock:
            delegate = self.calibration if self._using_calibration else self._position
            return delegate.active_driver_damping(
                base_kd, final_command_delta_nm=final_command_delta_nm
            )

    def validate_dispatch_start(
        self,
        *,
        feedback_sequence: int,
        feedback_timestamp_sec: float,
        now_sec: float | None = None,
    ) -> None:
        with self._lock:
            if self._using_calibration:
                self.calibration.validate_dispatch_start(
                    feedback_sequence=feedback_sequence,
                    feedback_timestamp_sec=feedback_timestamp_sec,
                    **({"now_sec": now_sec} if now_sec is not None else {}),
                )
            elif self._pending is None or self._pending[:2] != (
                feedback_sequence,
                feedback_timestamp_sec,
            ):
                raise ValueError("cycle requires matching pending feedback")
            elif (
                not feedback_timestamp_sec
                <= (
                    time.monotonic()
                    if now_sec is None
                    else finite(now_sec, "dispatch start")
                )
                <= feedback_timestamp_sec
                + self.settings.staged_assist.motion_control.max_control_gap_sec
            ):
                self._fail(
                    "cycle_dispatch_start_deadline_exceeded", feedback_timestamp_sec
                )
                raise ValueError(self._reason)

    def record_dispatch(self, *, native_receipt: dict[str, Any], **values: Any) -> None:
        with self._lock:
            if self._pending is None:
                raise ValueError("cycle has no pending dispatch")
            sequence, timestamp, using_calibration, session = self._pending
            now = finite(values.get("completed_at_sec"), "dispatch completion")
            reason = calibration_receipt_rejection(
                native_receipt,
                sequence,
                None if self._receipt is None else self._receipt["sequence"],
            )
            requested = finite(
                native_receipt.get("requested_at_sec"), "dispatch request"
            )
            if (
                reason
                or native_receipt.get("completed_at_sec") != now
                or native_receipt.get("sequence") != values.get("receipt_sequence")
                or values.get("feedback_timestamp_sec") != timestamp
                or not timestamp
                <= requested
                <= now
                <= timestamp
                + self.settings.staged_assist.motion_control.max_control_gap_sec
            ):
                self._fail(reason or "cycle_dispatch_clock_or_identity_mismatch", now)
                raise ValueError(self._reason)
            if using_calibration:
                self.calibration.record_dispatch(
                    native_receipt=native_receipt, **values
                )
            elif session is not None:
                session.record_dispatch(
                    receipt=native_receipt,
                    joint_commands=values["joint_commands"],
                    torque_rate_nm_s=values["torque_rate_nm_s"],
                )
            else:
                self._position.record_dispatch(**values)
            commands = values["joint_commands"]
            gravity = six(commands["gravity_nm"], "gravity")
            total = six(commands["total_nm"], "total")
            overlay = six(commands["overlay_nm"], "overlay")
            delta = tuple(t - g for t, g in zip(total, gravity, strict=True))
            if self._receipt is not None and self._started is not None:
                self._exposure += sum(abs(v) for v in self._last_sent_delta) * (
                    now - max(self._receipt["completed_at_sec"], self._started)
                )
            self._last_sent_delta = delta
            self._zero_base = (
                all(
                    o == 0 and abs(d) <= 1e-9
                    for o, d in zip(overlay, delta, strict=True)
                )
                and tuple(commands["driver_kd_nm_s_per_rad"])
                == self._condition.driver_kd_nm_s_per_rad
            )
            self._receipt = dict(native_receipt)
            self._pending = None
            self._advance(now)

    def _position_ready(self):
        view = self._position.owner_snapshot()
        return (
            self._zero_base
            and view["operator_phase"] == "inactive"
            and view["prepare_gate"] == "ready"
            and not view["fault"]
        )

    def _advance(self, now):
        if self._phase in {"positioning_out", "returning"}:
            status = self._session.owner_snapshot()
            if status["state"] == "idle" and self._position_ready():
                self._session.begin(
                    now_sec=now,
                    allow_motion=True,
                    human_intervention=self._human,
                    allow_breakaway=True,
                )
            elif (
                status["state"] == "idle"
                and now - self._phase_started > self.config.prepare_timeout_sec
            ):
                self._fail("cycle_positioning_preparation_timeout", now)
            elif status["state"] == "complete":
                returning = self._phase == "returning"
                self._moves.append({
                    **self._session.result_record(),
                    "motion_control": copy.deepcopy(self._last_positioning_motion),
                })
                self._start_observation(now, returning)
            elif status["state"] in {"failed", "cancelled"}:
                self._fail("cycle_positioning:" + status["reason"], now)
        elif self._phase in {"gravity_observe", "return_observe"}:
            if self._observation_anchor is not None:
                self._observation_min = min(self._observation_min, self._q[2])
                self._observation_max = max(self._observation_max, self._q[2])
                if (
                    not self._zero_base
                    or max(
                        abs(self._observation_min - self._observation_anchor[2]),
                        abs(self._observation_max - self._observation_anchor[2]),
                    )
                    > self.cycle.gravity_drift_limit_rad
                ):
                    self._fail("cycle_gravity_only_drift_or_nonzero_command", now)
                    return
            if not self._position_ready():
                if now - self._phase_started > self.config.prepare_timeout_sec:
                    self._fail("cycle_zero_base_or_stillness_not_confirmed", now)
                return
            if self._observation_started is None:
                self._observation_started, self._observation_anchor = now, self._q
                self._observation_min = self._observation_max = self._q[2]
            drift = self._q[2] - self._observation_anchor[2]
            if abs(drift) > self.cycle.gravity_drift_limit_rad:
                self._fail("cycle_gravity_only_drift", now)
            elif now - self._observation_started >= self.cycle.gravity_observation_sec:
                self._observations.append(
                    {
                        "phase": self._phase,
                        "started_at_sec": self._observation_started,
                        "completed_at_sec": now,
                        "anchor_qpos_rad": self._observation_anchor,
                        "final_qpos_rad": self._q,
                        "final_drift_rad": drift,
                        "minimum_J3_rad": self._observation_min,
                        "maximum_J3_rad": self._observation_max,
                        "zero_assist_and_base_damping_confirmed": True,
                    }
                )
                if self._phase == "return_observe":
                    if (
                        abs(self._q[2] - self._origin[2])
                        > self.config.motion_acquisition_arrival_tolerance_rad
                    ):
                        self._fail("cycle_return_outside_origin_tolerance", now)
                    else:
                        self._transition(
                            "complete", "returned_zero_base_and_still_confirmed", now
                        )
                else:
                    self._using_calibration = True
                    self.calibration.submit("arm", now_sec=now)
                    self._transition(
                        "calibration",
                        "gravity_observation_complete_start_existing_batch",
                        now,
                    )
        elif self._phase == "calibration":
            batch = self.calibration.owner_snapshot()
            if (
                batch["state"] == "complete"
                or batch["state"] == "failed"
                and batch["outcome"]["severity"] == "warning"
            ):
                # The session's terminal state follows its own final-zero/base
                # stop. Do not recover a control fault or cancelled permission.
                if not self._zero_base or batch.get("fault"):
                    self._fail("cycle_acquisition_terminal_without_zero_base", now)
                    return
                self._using_calibration = False
                self._position = B601StagedTorqueConsoleController(self.settings)
                self._position.owner_ready()
                if (
                    abs(self._q[2] - self._origin[2])
                    <= self.config.motion_acquisition_arrival_tolerance_rad
                ):
                    self._start_observation(now, True)
                else:
                    self._make_move(self._origin[2], now, "return_not_for_fit")
            elif batch["state"] in {"failed", "cancelled"}:
                self._fail("cycle_acquisition:" + batch["reason"], now)
        elif self._phase == "stopping":
            ready = (
                self._zero_base
                and self.calibration.owner_snapshot()["state"]
                in {"cancelled", "failed"}
                if self._using_calibration
                else self._position_ready()
            )
            if ready:
                self._transition("cancelled", self._cancel_requested, now)
            elif (
                now - self._phase_started
                > self.config.prepare_timeout_sec
                + self.settings.staged_assist.motion_control.braking_timeout_sec
            ):
                self._fail("cycle_withdrawal_not_confirmed", now)

    def _cycle_status(self, *, detailed=False):
        return {
            "phase": self._phase,
            "reason": self._reason,
            "page_visible": self._page_visible,
            "background_permit_timeout_sec": self._background_permit_timeout_sec,
            "settings": asdict(self.cycle),
            "maximum_offset_rad": self.maximum_offset_rad,
            "origin_qpos_rad": self._origin,
            "current_qpos_rad": self._q,
            "outbound_target_rad": None
            if self._origin is None
            else self._origin[2] + self.cycle.offset_rad,
            "gravity_observation": None
            if self._observation_anchor is None
            else {
                "started_at_sec": self._observation_started,
                "anchor_qpos_rad": self._observation_anchor,
                "minimum_J3_rad": self._observation_min,
                "maximum_J3_rad": self._observation_max,
                "drift_limit_rad": self.cycle.gravity_drift_limit_rad,
            },
            "positioning": None
            if self._session is None
            else self._session.status()
            if detailed
            else self._session.owner_snapshot(),
            "positioning_limits": {
                "support_soft_speed_recovery": self.settings.staged_assist.motion_control.support_soft_speed_recovery,
                "continuous_speed_limit_rad_s": self.settings.staged_assist.motion_control.continuous_speed_limit_rad_s,
                "tracking_torque_rise_nm_s": self.settings.staged_assist.tracking_slew_nm_s,
                "tracking_torque_fall_nm_s": self.settings.staged_assist.tracking_release_slew_nm_s,
                "arrival_tolerance_rad": self.config.motion_acquisition_arrival_tolerance_rad,
            },
            **({"positioning_control": copy.deepcopy(self._last_positioning_motion)} if detailed else {}),
            **(
                {
                    "moves": copy.deepcopy(self._moves),
                    "gravity_observations": copy.deepcopy(self._observations),
                }
                if detailed
                else {}
            ),
            "completed_positioning_moves": len(self._moves),
            **({"events": copy.deepcopy(self._events)} if detailed else {}),
            "command_exposure_nm_sec": self._exposure,
            "zero_assist_base_damping_confirmed": self._zero_base,
            "disable_confirmation_required": self.cycle.supported_origin_disable_confirmation,
            "disable_confirmation_available": self.cycle_disable_confirmation_ready,
            "semantics": "software_driven_positioning_not_external_force_measurement",
        }

    def _batch_status(self, *, detailed=False):
        batch = (
            self.calibration.status()["automatic_calibration"]
            if detailed
            else self.calibration.owner_snapshot()
        )
        acquisition_warning = (
            batch["state"] == "failed" and batch["outcome"]["severity"] == "warning"
        )
        return {
            **batch,
            "operation_mode": "supported_cycle",
            "state": batch["state"] if self._phase == "calibration" else self._phase,
            "reason": batch["reason"] if self._phase == "calibration" else self._reason,
            "outcome": batch["outcome"]
            if self._phase == "calibration"
            or self._phase == "complete"
            and acquisition_warning
            else calibration_outcome(self._phase, self._reason),
            "calibration_result": {
                "state": batch["state"],
                "reason": batch["reason"],
                "outcome": batch["outcome"],
            },
            "stage": batch["stage"] if self._using_calibration else "cycle",
            "active": self._active,
            "start_requested": self._start_requested,
            "human_intervention": self._human,
            "cycle": self._cycle_status(detailed=detailed),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            view = (
                self.calibration if self._using_calibration else self._position
            ).status()
            return {
                **view,
                "owner_ready": self._ready,
                "controller_mode": self.controller_mode,
                "automatic_calibration": self._batch_status(detailed=True),
                "events": list(self._events),
            }

    def drain_records(self) -> list[dict[str, Any]]:
        with self._lock:
            records, self._records = self._records, []
            for record in self.calibration.drain_records():
                if record["kind"] == "automatic_calibration_result":
                    record = {**record, "kind": "automatic_calibration_substage_result"}
                records.append(record)
            return records

    def result_record(self) -> dict[str, Any]:
        with self._lock:
            result = self.calibration.result_record()
            if self._phase != "complete":
                state = (
                    "failed"
                    if self._phase == "failed"
                    else "cancelled"
                    if self._phase == "cancelled"
                    else self._phase
                )
                result = {
                    **result,
                    "state": state,
                    "reason": self._reason,
                    "outcome": calibration_outcome(state, self._reason),
                }
            return {
                **result,
                "operation_mode": "supported_cycle",
                "recorded_monotonic_sec": None
                if self._receipt is None
                else self._receipt["completed_at_sec"],
                "cycle": self._cycle_status(detailed=True),
            }

    def recording_failed(self, reason: str) -> None:
        with self._lock:
            self.calibration.recording_failed(reason)
            self._cancel_requested = "recording_failed:" + reason

    def owner_stopped(self, *, now_sec: float, reason: str) -> None:
        with self._lock:
            self._ready = False
            if self._phase not in {"complete", "failed", "cancelled"}:
                self._transition("cancelled", reason, now_sec)
            self.calibration.owner_stopped(now_sec=now_sec, reason=reason)
            self._position.owner_stopped(now_sec=now_sec, reason=reason)
