"""Offline automatic acquisition adapter around the existing staged owner port.

There is intentionally no hardware launcher or CAN writer. A bounded batch
authorization permits sequencing intents, not changing gains/guards or loading
compensation. Feedback, tracking, braking and dispatch acknowledgement remain
with B601StagedTorqueConsoleController and its existing motion runtime.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from typing import Any

from robot_motion.rebot_b601_rs.calibration import CalibrationCondition, finite, six
from robot_motion.rebot_b601_rs.holding_calibration import (
    HoldingForceSearch,
    HoldingSearchConfig,
)
from robot_motion.rebot_b601_rs.mit import B601_RS_JOINT_MIT_FRAME
from robot_motion.rebot_b601_rs.model import B601_RS_ARM_JOINT_NAMES

from .staged_controller import B601StagedTorqueConsoleController


def calibration_outcome(state: str, reason: str) -> dict[str, str]:
    """Describe acquisition separately; never override an owner fault or exit.

    Known capacity/quality stops finish the existing withdrawal protocol before
    reaching a terminal state. Unknown failures stay errors, not warnings.
    """
    incomplete = {
        "stopped_short_of_target", "insufficient_authority_no_automatic_retry",
        "batch_timeout", "leg_timeout", "preparation_timeout",
        "active_command_exposure_budget", "leg_command_exposure_budget",
        "workspace_timeout", "workspace_command_exposure_budget",
        "holding_search_incomplete",
    }
    if state == "failed":
        severity = "warning" if reason in incomplete else "error"
        action = "retain_partial_data_stop_acquisition" if severity == "warning" else "withdraw_and_require_fault_resolution"
    elif state == "cancelled":
        if reason in {
            "human_intervention", "human_contact_or_unknown",
            "human_intervention_unknown", "operator_withdraw", "operator_zero",
            "page_hidden", "page_unload", "owner_cancel",
        }:
            severity, action = "info", "operator_withdrawal"
        elif reason in incomplete or reason in {"operator_permit_expired", "gui_permit_expired"}:
            severity, action = "warning", "retain_partial_data_stop_acquisition"
        else:
            severity, action = "error", "withdraw_and_require_fault_resolution"
    else:
        severity, action = "info", "complete" if state == "complete" else "continue"
    return {"severity": severity, "scope": "acquisition", "reason": reason, "action": action}


def complete_calibration_receipt(
    receipt: dict[str, Any], source_sequence: int, previous_sequence: int | None
) -> bool:
    """Both acquisition stages require the same complete native owner receipt."""
    return calibration_receipt_rejection(receipt, source_sequence, previous_sequence) is None


def calibration_receipt_rejection(
    receipt: dict[str, Any], source_sequence: int, previous_sequence: int | None
) -> str | None:
    """Explain the existing admission gates without relaxing any of them."""
    if not isinstance(receipt, dict):
        return "invalid_native_receipt"
    if receipt.get("accepted") is not True:
        return "native_dispatch_not_accepted"
    if any(
        not isinstance(receipt.get(key), (list, tuple))
        or tuple(receipt[key]) != B601_RS_ARM_JOINT_NAMES
        for key in ("requested_joint_names", "sent_joint_names")
    ):
        return "incomplete_six_axis_dispatch"
    if any(
        not isinstance(receipt.get(key), (list, tuple)) or receipt[key]
        for key in ("errors", "failed_joint_names")
    ):
        return "native_dispatch_errors_or_failed_joints"
    if receipt.get("mode") != "mit" or receipt.get("frame_id") != B601_RS_JOINT_MIT_FRAME:
        return "native_command_identity_mismatch"
    if type(receipt.get("source_sequence")) is not int or receipt["source_sequence"] != source_sequence:
        return "dispatch_source_sequence_mismatch"
    if type(receipt.get("sequence")) is not int or receipt["sequence"] < 0:
        return "invalid_dispatch_receipt_sequence"
    if previous_sequence is not None and receipt["sequence"] <= previous_sequence:
        return "dispatch_receipt_sequence_not_advancing"
    return None


@dataclass(frozen=True)
class CalibrationBatchLimits:
    max_batch_sec: float
    max_leg_sec: float
    prepare_timeout_sec: float
    operator_lease_sec: float
    max_feedback_gap_sec: float
    max_active_command_exposure_nm_sec: float
    max_leg_command_exposure_nm_sec: float | None = None

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name == "max_leg_command_exposure_nm_sec" and value is None:
                continue
            if finite(value, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_batch_sec < self.max_leg_sec:
            raise ValueError("batch duration must cover a complete leg")
        if self.max_leg_command_exposure_nm_sec is not None and self.max_leg_command_exposure_nm_sec > self.max_active_command_exposure_nm_sec:
            raise ValueError("leg exposure cannot exceed the complete batch budget")


class B601AutomaticCalibrationSession:
    """One-shot offline batch; terminal states never rearm or retry themselves."""

    hardware_execution_allowed = False

    def __init__(
        self,
        controller: B601StagedTorqueConsoleController,
        *,
        plan: dict[str, Any],
        condition: CalibrationCondition,
        position_limits_rad: tuple[tuple[float, float], ...],
        limits: CalibrationBatchLimits,
        acquisition_arrival_tolerance_rad: float | None = None,
        holding_search: HoldingSearchConfig | None = None,
    ) -> None:
        if not isinstance(controller, B601StagedTorqueConsoleController):
            raise TypeError(
                "automatic acquisition requires the existing staged controller"
            )
        motion = controller.settings.staged_assist.motion_control
        if holding_search is not None and (
            not isinstance(holding_search, HoldingSearchConfig)
            or motion is None
            or not motion.responsive_tracking_brake
            or motion.continuous_speed_limit_rad_s is None
            or holding_search.movement_rad < 4 * (motion.position_resolution_rad or 0.0)
        ):
            raise ValueError("holding search requires explicit continuous resolved motion and at least four position counts")
        self.holding_search = holding_search
        if (
            motion is None
            or controller.settings.staged_assist.motion_onset_confirm_displacement_rad
            is None
        ):
            raise ValueError(
                "automatic acquisition requires confirmed-motion, causal-window feedback and protected withdrawal"
            )
        if controller.limits.joint_name not in {"joint2", "joint3"}:
            raise ValueError(
                "automatic acquisition supports J2/J3 only"
            )
        self._joint_index = j = B601_RS_ARM_JOINT_NAMES.index(controller.limits.joint_name)
        if (
            limits.operator_lease_sec > controller.limits.command_timeout_sec
            or limits.max_feedback_gap_sec > motion.max_control_gap_sec
        ):
            raise ValueError("batch leases must not relax the original owner timeouts")
        if (
            plan.get("schema") != "b601.local-calibration-plan.v1"
            or plan.get("execution_enabled") is not False
        ):
            raise ValueError(
                "expected an offline draft plan, not an execution capability"
            )
        if (
            CalibrationCondition.from_mapping(
                plan.get("condition"), run_id=condition.run_id
            )
            != condition
        ):
            raise ValueError(
                "plan gravity/source condition differs from current condition"
            )
        if (
            plan["condition"].get("run_id") != condition.run_id
            or plan.get("joint_name") != controller.limits.joint_name
        ):
            raise ValueError("plan run or joint differs from the owner")
        anchor = six(plan.get("anchor_qpos_rad"), "anchor_qpos_rad")
        if len(position_limits_rad) != 6:
            raise ValueError("six owner position limits required")
        for q, interval in zip(anchor, position_limits_rad):
            if (
                len(interval) != 2
                or not finite(interval[0], "lower") <= q <= finite(interval[1], "upper")
                or interval[0] >= interval[1]
            ):
                raise ValueError("anchor outside owner position limits")
        steps = plan.get("steps")
        if not isinstance(steps, list) or not 1 <= len(steps) <= 128:
            raise ValueError("batch requires 1..128 finite steps")
        seen = set()
        previous = anchor
        spans = []
        reference_duration = 0.0
        for step in steps:
            if not isinstance(step, dict) or step.get("role") not in {
                "training",
                "validation",
                "positioning_not_for_fit",
                "return_not_for_fit",
            }:
                raise ValueError("invalid calibration step role")
            ident = step.get("trial_id")
            if not isinstance(ident, str) or not ident or ident in seen:
                raise ValueError("step identifiers must be non-empty and unique")
            seen.add(ident)
            target = six(step.get("target_qpos_rad"), "target_qpos_rad")
            speed = finite(
                step.get("max_reference_velocity_rad_s"), "reference velocity"
            )
            if not 0 < speed <= motion.max_reference_velocity_rad_s:
                raise ValueError(
                    "plan reference speed exceeds the existing motion profile"
                )
            if any(target[i] != anchor[i] for i in range(6) if i != j):
                raise ValueError(
                    "single-joint acquisition cannot move support-joint references"
                )
            if any(
                not lo <= q <= hi for q, (lo, hi) in zip(target, position_limits_rad)
            ):
                raise ValueError("plan target outside owner position limits")
            if (
                abs(target[j] - anchor[j]) > controller.limits.max_excursion_rad
                or abs(target[j] - previous[j])
                > controller.settings.maximum_target_offset_rad
            ):
                raise ValueError(
                    "plan exceeds global excursion or per-leg target envelope"
                )
            span = abs(target[j] - previous[j])
            if span:
                spans.append(span)
            reference_duration += span / speed
            previous = target
        strict_tolerance = controller.settings.staged_assist.settle_error_rad
        arrival_tolerance = strict_tolerance if acquisition_arrival_tolerance_rad is None else finite(
            acquisition_arrival_tolerance_rad, "acquisition arrival tolerance"
        )
        if acquisition_arrival_tolerance_rad is not None and (
            arrival_tolerance + 1e-12 < (strict_tolerance if holding_search is None else holding_search.movement_rad) or not spans
            or arrival_tolerance > min(spans) / 2 + 1e-12
        ):
            raise ValueError("acquisition tolerance must lie between the applicable tracking/holding tolerance and half the smallest planned move")
        if (
            acquisition_arrival_tolerance_rad is not None
            and max(spans) + arrival_tolerance
            > controller.settings.maximum_target_offset_rad + 1e-12
        ):
            raise ValueError("acquisition tolerance must leave room for the next leg's actual anchor")
        if limits.max_leg_command_exposure_nm_sec is not None and limits.max_leg_command_exposure_nm_sec <= 6 * controller.limits.max_abs_assist_nm * limits.max_feedback_gap_sec:
            raise ValueError("leg exposure must allow a complete withdrawal reserve")
        self.acquisition_arrival_tolerance_rad = arrival_tolerance
        self._budget_plan = {
            "step_count": len(steps),
            "reference_motion_only_sec": reference_duration,
            "maximum_leg_budgets_sum_nm_sec": None if limits.max_leg_command_exposure_nm_sec is None else len(steps) * limits.max_leg_command_exposure_nm_sec,
            "batch_command_budget_nm_sec": limits.max_active_command_exposure_nm_sec,
            "all_leg_caps_fit_batch": None if limits.max_leg_command_exposure_nm_sec is None else len(steps) * limits.max_leg_command_exposure_nm_sec <= limits.max_active_command_exposure_nm_sec,
            "semantics": "reference duration excludes startup, friction, braking and rest; finite batch budget may stop acquisition before every leg",
        }
        self.controller = controller
        self.condition = condition
        self.limits = limits
        self._steps = copy.deepcopy(steps)
        self._anchor = anchor
        self._position_limits = tuple(
            tuple(interval) for interval in position_limits_rad
        )
        self._lock = threading.RLock()
        self._state = "idle"
        self._reason = "not_authorized"
        self._index = 0
        self._batch_id: str | None = None
        self._started: float | None = None
        self._phase_started: float | None = None
        self._leg_started: float | None = None
        self._last_feedback: float | None = None
        self._last_feedback_sequence: int | None = None
        self._pending: tuple[int, float] | None = None
        self._last_receipt_sequence: int | None = None
        self._last_completed: float | None = None
        self._sent_delta = (0.0,) * 6
        self._sent_driver_kd: tuple[float, ...] | None = None
        self._last_dispatch_was_releasing = False
        self._active_exposure = 0.0
        self._release_exposure = 0.0
        self._leg_exposure = 0.0
        self._breakaway_allowed = False
        self._breakaway_consumed = False
        self._stop_destination: str | None = None
        self._events: list[dict[str, Any]] = []
        self._completed_legs: list[dict[str, Any]] = []
        self._arrival_stationary_since: float | None = None
        self._arrival_min: tuple[float, ...] | None = None
        self._arrival_max: tuple[float, ...] | None = None
        self._holding = None
        self._holding_probe_nm = None
        self._holding_observation = None
        self._holding_pose = None
        self._holding_leg_result = None
        self._holding_attempts = 0
        self._holding_reacquire_bias_nm = 0.0
        self._holding_approach_origin = None
        self._previous_sent_delta = (0.0,) * 6
        self._previous_completed = None
        self._previous_sent_driver_kd = None

    def _transition(self, state: str, reason: str, now: float) -> None:
        self._state, self._reason, self._phase_started = state, reason, now
        self._events.append(
            {
                "time_sec": now,
                "state": state,
                "reason": reason,
                "step_index": self._index,
            }
        )

    def begin(
        self,
        *,
        now_sec: float,
        allow_motion: bool,
        human_intervention: bool,
        allow_breakaway: bool,
    ) -> None:
        now = finite(now_sec, "now_sec")
        with self._lock:
            if self._state != "idle":
                raise ValueError(
                    "a batch is one-shot; completed/failed batches cannot restart"
                )
            if (
                allow_motion is not True
                or human_intervention is not False
                or not isinstance(allow_breakaway, bool)
            ):
                raise ValueError(
                    "batch requires explicit motion permission and declared no human contact"
                )
            snapshot = self.controller.owner_snapshot()
            stamp = snapshot["feedback_timestamp_sec"]
            if (
                not snapshot["ready"]
                or snapshot["operator_phase"] != "inactive"
                or snapshot["fault"]
                or stamp is None
                or not 0 <= now - stamp <= self.limits.operator_lease_sec
                or self._pending is not None
                or not self._zero_dispatch()
                or now < self._last_completed
                or self._sent_driver_kd != self.condition.driver_kd_nm_s_per_rad
            ):
                raise ValueError(
                    "batch requires a fresh inactive owner with confirmed zero assist"
                )
            self._breakaway_allowed = allow_breakaway
            self._started = now
            self._batch_id = f"batch-{now:.9f}"
            self._transition("preparing", "bounded_batch_authorized", now)

    def _abort(self, now: float, reason: str, *, failed: bool = True) -> None:
        if self._state in {"complete", "failed", "cancelled", "idle"}:
            return
        if self._stop_destination is None:
            self._stop_destination = "failed" if failed else "cancelled"
            if self._state == "releasing":
                # Do not restart a withdrawal or its deadline after a late
                # contact/fault. Preserve the first physical release request.
                self._reason = reason
            else:
                if not self.controller.owner_snapshot()["release_terminal"]:
                    self.controller.submit("withdraw", now_sec=now)
                self._transition("releasing", reason, now)

    def cancel(self, *, now_sec: float, reason: str = "operator_cancel") -> None:
        with self._lock:
            self._abort(finite(now_sec, "now_sec"), reason, failed=False)

    def _marker(self) -> dict[str, Any] | None:
        if self._batch_id is None:
            return None
        step = self._steps[min(self._index, len(self._steps) - 1)]
        return {
            "batch_id": self._batch_id,
            "trial_id": step["trial_id"],
            "role": step["role"],
            "state": self._state,
        }

    def step(
        self,
        *,
        now_sec: float,
        feedback_sequence: int,
        qpos_rad: tuple[float, ...],
        qvel_rad_s: tuple[float, ...],
        condition: CalibrationCondition,
        human_intervention: bool | None,
        operator_permit_timestamp_sec: float | None,
        operator_observed_at_sec: float | None = None,
    ) -> tuple[tuple[float, ...], dict[str, Any]]:
        try:
            now = finite(now_sec, "now_sec")
        except ValueError:
            with self._lock:
                if self._last_feedback is not None:
                    self._abort(
                        max(
                            self._last_feedback,
                            self._last_completed or self._last_feedback,
                        ),
                        "invalid_feedback_clock",
                    )
            raise
        with self._lock:
            try:
                q, velocity = six(qpos_rad, "qpos_rad"), six(qvel_rad_s, "qvel_rad_s")
                if type(feedback_sequence) is not int or feedback_sequence < 0:
                    raise ValueError("feedback sequence must be non-negative")
                permit = (
                    None
                    if operator_permit_timestamp_sec is None
                    else finite(operator_permit_timestamp_sec, "operator permit")
                )
                permit_clock = (
                    now
                    if operator_observed_at_sec is None
                    else finite(operator_observed_at_sec, "operator observation clock")
                )
                if not now <= permit_clock <= now + self.limits.max_feedback_gap_sec:
                    raise ValueError(
                        "operator observation must follow fresh feedback within one control gap"
                    )
            except ValueError:
                self._abort(
                    max(now, self._last_feedback or now), "invalid_feedback_or_permit"
                )
                raise
            if self._last_feedback is not None and now <= self._last_feedback:
                self._abort(self._last_feedback, "feedback_clock_not_advancing")
                raise ValueError("feedback timestamps must advance")
            if (
                self._last_feedback_sequence is not None
                and feedback_sequence <= self._last_feedback_sequence
            ):
                self._abort(now, "feedback_sequence_not_advancing")
                raise ValueError("feedback sequences must advance")
            active = self._state not in {"idle", "complete", "failed", "cancelled"}
            if active:
                if condition != self.condition:
                    self._abort(now, "gravity_or_source_condition_changed")
                elif human_intervention is not False:
                    self._abort(now, "human_contact_or_unknown", failed=False)
                elif (
                    permit is None
                    or not 0 <= permit_clock - permit <= self.limits.operator_lease_sec
                ):
                    self._abort(now, "operator_permit_expired")
                elif self._pending is not None:
                    self._abort(now, "previous_dispatch_not_confirmed")
                elif (
                    self._last_feedback is not None
                    and now - self._last_feedback > self.limits.max_feedback_gap_sec
                ):
                    self._abort(now, "feedback_gap")
                elif any(
                    not lo <= position <= hi
                    for position, (lo, hi) in zip(q, self._position_limits)
                ):
                    self._abort(now, "outside_batch_position_envelope")
                elif (
                    abs(q[self._joint_index] - self._anchor[self._joint_index])
                    > self.controller.limits.max_excursion_rad
                ):
                    self._abort(now, "global_batch_excursion")
                elif any(
                    abs(q[i] - self._anchor[i])
                    > self.controller.limits.max_support_drift_rad
                    for i in range(6)
                    if i != self._joint_index
                ):
                    self._abort(now, "global_support_drift")
                elif now - self._started > self.limits.max_batch_sec:
                    self._abort(now, "batch_timeout")
                elif (
                    self._state in {"preparing", "arming"}
                    and now - self._phase_started > self.limits.prepare_timeout_sec
                ):
                    self._abort(now, "preparation_timeout")
                elif (
                    self._state in {"moving", "observing"}
                    and now - self._leg_started > self.limits.max_leg_sec
                ):
                    self._abort(now, "leg_timeout")
                if self._state != "releasing" and self._last_completed is not None:
                    pending_exposure = sum(abs(v) for v in self._sent_delta) * max(
                        0.0, now - self._last_completed
                    )
                    # Reserve one entire worst-case six-axis dispatch interval.
                    reserve = (
                        6
                        * self.controller.limits.max_abs_assist_nm
                        * self.limits.max_feedback_gap_sec
                    )
                    budget_reason = self._exposure_budget_reason(pending_exposure + reserve)
                    if budget_reason:
                        self._abort(now, budget_reason)
                snapshot = self.controller.owner_snapshot()
                if (
                    self._state in {"arming", "moving"}
                    and snapshot["operator_phase"] == "armed"
                ):
                    self.controller.submit("heartbeat", now_sec=now)
            self._last_feedback = now
            self._last_feedback_sequence = feedback_sequence
            overlay, audit = self.controller.step(
                now_sec=now, qpos_rad=q, qvel_rad_s=velocity,
                tracking_recovery_permitted=(
                    self._state in {"arming", "moving"}
                    and human_intervention is False and permit is not None
                    and 0 <= permit_clock - permit <= self.limits.operator_lease_sec
                ),
                **({"calibration_hold_enabled": self._state == "moving",
                    "calibration_probe_nm": self._holding_probe_nm if self._state == "moving" else None,
                    "calibration_approach_bias_nm": self._holding_reacquire_bias_nm if self._state == "moving" else 0.0}
                   if self.holding_search is not None else {}),
            )
            self._pending = (feedback_sequence, now)
            audit = {**audit, "calibration_trial": self._marker(),
                "holding_calibration": self._holding_status()}
            return overlay, audit

    def _zero_dispatch(self) -> bool:
        return self._last_completed is not None and all(
            abs(v) <= 1e-9 for v in self._sent_delta
        )

    def _exposure_budget_reason(self, reserve: float = 0.0) -> str | None:
        # Legacy batches retain their active-only accounting. Explicit
        # per-leg acquisition also counts withdrawal commands in both bounds.
        leg_limit = self.limits.max_leg_command_exposure_nm_sec
        total = self._active_exposure + (self._release_exposure if leg_limit is not None else 0.0)
        if total + reserve >= self.limits.max_active_command_exposure_nm_sec:
            return "active_command_exposure_budget"
        if leg_limit is not None and self._leg_exposure + reserve >= leg_limit:
            return "leg_command_exposure_budget"
        return None

    def _arrival_still(self, view: dict[str, Any], *, eligible: bool, now: float) -> bool:
        """Reuse owner window gates, then dwell only AFTER final zero dispatch.

        The captured hold point is not the plan target. The six-axis range also
        prevents consecutive quiet windows from accepting a slow staircase.
        """
        if not eligible or view["arrival_stop_gate"] != "ready":
            self._arrival_stationary_since = None
            self._arrival_min = self._arrival_max = None
            return False
        clock, q = view["feedback_timestamp_sec"], view["qpos_rad"]
        staged = self.controller.settings.staged_assist
        if self._arrival_stationary_since is None:
            self._arrival_stationary_since = now
            self._arrival_min = self._arrival_max = q
        else:
            self._arrival_min = tuple(min(a, b) for a, b in zip(self._arrival_min, q))
            self._arrival_max = tuple(max(a, b) for a, b in zip(self._arrival_max, q))
            allowed = max(
                staged.motion_control.position_resolution_rad or 0.0,
                staged.settle_velocity_rad_s * staged.settle_dwell_sec,
            )
            if max(hi - lo for lo, hi in zip(self._arrival_min, self._arrival_max)) > allowed:
                self._arrival_stationary_since = now
                self._arrival_min = self._arrival_max = q
        return clock - self._arrival_stationary_since >= staged.settle_dwell_sec

    def record_dispatch(
        self,
        *,
        receipt: dict[str, Any],
        joint_commands: dict[str, tuple[float, ...]],
        torque_rate_nm_s: tuple[float, ...],
    ) -> None:
        """Consume the actual complete receipt; failed sends never advance a leg."""
        with self._lock:
            if self._pending is None:
                raise ValueError("no pending control frame")
            sequence, timestamp = self._pending
            if not isinstance(receipt, dict) or not isinstance(joint_commands, dict):
                self._abort(timestamp, "invalid_dispatch_record")
                raise TypeError("receipt and joint commands must be objects")
            receipt_sequence = receipt.get("sequence")
            complete = complete_calibration_receipt(
                receipt, sequence, self._last_receipt_sequence
            )
            if not complete:
                self._abort(timestamp, "incomplete_or_mismatched_dispatch")
                raise ValueError(
                    "a complete advancing six-axis MIT receipt is required"
                )
            try:
                requested = finite(receipt.get("requested_at_sec"), "requested_at_sec")
                completed = finite(receipt.get("completed_at_sec"), "completed_at_sec")
                gravity = six(joint_commands.get("gravity_nm"), "gravity_nm")
                total = six(joint_commands.get("total_nm"), "total_nm")
                overlay = six(joint_commands.get("overlay_nm"), "overlay_nm")
                driver_kd = six(
                    joint_commands.get("driver_kd_nm_s_per_rad"),
                    "driver_kd_nm_s_per_rad",
                )
                rates = six(torque_rate_nm_s, "torque_rate_nm_s")
                if min(driver_kd) < 0 or min(rates) < 0:
                    raise ValueError(
                        "driver damping and torque rate must be non-negative"
                    )
            except ValueError:
                self._abort(timestamp, "invalid_dispatch_command")
                raise
            if (
                not timestamp
                <= requested
                <= completed
                <= timestamp + self.limits.max_feedback_gap_sec
                or self._last_completed is not None
                and requested < self._last_completed
            ):
                self._abort(timestamp, "dispatch_clock_or_deadline")
                raise ValueError(
                    "receipt must follow its feedback within the dispatch deadline"
                )
            try:
                self.controller.record_dispatch(
                    joint_name=self.controller.limits.joint_name,
                    receipt_sequence=receipt_sequence,
                    feedback_timestamp_sec=timestamp,
                    overlay_after_profile_nm=overlay[self._joint_index],
                    gravity_nm=gravity[self._joint_index],
                    requested_total_nm=gravity[self._joint_index] + overlay[self._joint_index],
                    conditioned_total_nm=total[self._joint_index],
                    joint_commands=joint_commands,
                    completed_at_sec=completed,
                    torque_rate_nm_s=rates,
                )
            except (ValueError, TypeError):
                self._abort(completed, "native_dispatch_rejected")
                raise
            if self._last_completed is not None and self._started is not None:
                exposure = sum(abs(v) for v in self._sent_delta) * (
                    completed - max(self._last_completed, self._started)
                )
                self._leg_exposure += exposure
                # Attribute the interval to the command that actually remained
                # dispatched, not the state requested at the current tick.
                if self._last_dispatch_was_releasing:
                    self._release_exposure += exposure
                else:
                    self._active_exposure += exposure
            self._previous_sent_delta, self._previous_completed = self._sent_delta, self._last_completed
            self._previous_sent_driver_kd = self._sent_driver_kd
            self._sent_delta = tuple(t - g for t, g in zip(total, gravity))
            self._sent_driver_kd = driver_kd
            self._last_dispatch_was_releasing = self._state == "releasing"
            self._last_completed, self._last_receipt_sequence = (
                completed,
                receipt_sequence,
            )
            self._pending = None
            self._advance(completed, driver_kd)

    def _advance(self, now: float, driver_kd: tuple[float, ...]) -> None:
        if self._state in {"idle", "complete", "failed", "cancelled"}:
            return
        view = self.controller.owner_snapshot()
        if view["fault"] or view["release_failed"]:
            self._abort(now, f"owner_fault:{view['fault'] or view['reason']}")
        if self._state == "releasing":
            zero_and_base = (
                view["operator_phase"] == "inactive"
                and self._zero_dispatch()
                and driver_kd == self.condition.driver_kd_nm_s_per_rad
                and view["prepare_gate"] == "ready"
            )
            arrived_still = self._arrival_still(
                view, eligible=zero_and_base and self._stop_destination is None, now=now
            )
            if zero_and_base and (self._stop_destination is not None or arrived_still):
                if self._stop_destination is not None:
                    self._transition(self._stop_destination, self._reason, now)
                else:
                    step = self._steps[self._index]
                    error = step["target_qpos_rad"][self._joint_index] - view["qpos_rad"][self._joint_index]
                    budget_reason = (
                        self._exposure_budget_reason()
                        if self.limits.max_leg_command_exposure_nm_sec is not None else None
                    )
                    if budget_reason:
                        self._transition("failed", budget_reason, now)
                        return
                    if self._holding_leg_result is None and abs(error) > self.acquisition_arrival_tolerance_rad:
                        self._transition("failed", "stopped_short_of_target", now)
                        return
                    self._completed_legs.append(
                        {
                            "trial_id": step["trial_id"],
                            "role": step["role"],
                            "completed_at_sec": now,
                            "target_position_rad": step["target_qpos_rad"][self._joint_index],
                            "measured_position_rad": view["qpos_rad"][self._joint_index],
                            "target_error_rad": error,
                            "final_zero_and_still_confirmed": True,
                            "tracking_target_reached": abs(error) <= self.controller.settings.staged_assist.settle_error_rad,
                            "tracking_arrival_tolerance_rad": self.controller.settings.staged_assist.settle_error_rad,
                            "acquisition_arrival_tolerance_rad": self.acquisition_arrival_tolerance_rad,
                            "command_exposure_nm_sec": self._leg_exposure,
                            **({"holding_calibration": self._holding_leg_result,
                                "arrival_semantics": (
                                    "verified_zero_hold_without_motion_then_separate_withdrawal"
                                    if self._holding_leg_result.get("observation_mode") == "passive_zero_assist"
                                    else "verified_nonzero_hold_then_separate_safe_withdrawal")}
                               if self._holding_leg_result is not None else {}),
                        }
                    )
                    self._index += 1
                    self._leg_exposure = 0.0
                    self._holding = self._holding_probe_nm = self._holding_observation = None
                    self._holding_pose = self._holding_leg_result = None
                    self._holding_attempts = 0
                    self._holding_reacquire_bias_nm = 0.0
                    self._holding_approach_origin = None
                    if self._index == len(self._steps):
                        self._transition(
                            "complete", (
                                "holding_points_and_withdrawals_confirmed" if self.holding_search is not None else
                                "all_targets_and_withdrawals_confirmed"
                                if all(leg["tracking_target_reached"] for leg in self._completed_legs)
                                else "all_legs_collected_with_target_deviation"
                            ), now
                        )
                    else:
                        self._transition(
                            "preparing", "previous_leg_zero_and_still_confirmed", now
                        )
                    self._arrival_stationary_since = None
                    self._arrival_min = self._arrival_max = None
            elif (
                now - self._phase_started
                > self.controller.settings.staged_assist.motion_control.braking_timeout_sec
                + self.limits.max_feedback_gap_sec
            ):
                self._transition(
                    "failed", "withdrawal_not_confirmed_owner_protection_required", now
                )
            return
        step = self._steps[self._index]
        if self._state == "observing":
            self._advance_holding(view, now, driver_kd, passive=True)
            return
        if self._state == "preparing":
            if (
                view["operator_phase"] != "inactive"
                or view["prepare_gate"] != "ready"
                or not self._zero_dispatch()
                or driver_kd != self.condition.driver_kd_nm_s_per_rad
            ):
                return
            if (self.holding_search is not None
                and abs(step["target_qpos_rad"][self._joint_index] - view["qpos_rad"][self._joint_index])
                <= min(self.holding_search.movement_rad, self.acquisition_arrival_tolerance_rad)):
                if view["arrival_stop_gate"] != "ready":
                    return
                # Recoil can already have reached the next sample. Measure
                # zero without creating a CUE/pulse that has no position error.
                self._leg_started = now
                self._holding_approach_origin = view["qpos_rad"]
                self._transition("observing", "already_at_point_zero_assist_observation", now)
                self._advance_holding(view, now, driver_kd, passive=True)
                return
            self.controller.configure_reference_velocity(
                step["max_reference_velocity_rad_s"]
            )
            self.controller.submit("arm", now_sec=now)
            self._transition("arming", "stationary_zero_confirmed", now)
        elif self._state == "arming":
            if view["operator_phase"] != "armed":
                return
            target = step["target_qpos_rad"][self._joint_index]
            offset = target - view["anchor_qpos_rad"][self._joint_index]
            if abs(offset) > self.controller.settings.maximum_target_offset_rad:
                self._abort(now, "actual_anchor_exceeds_leg_envelope")
                return
            self.controller.submit("set_target", now_sec=now, target_offset_rad=offset)
            self._breakaway_consumed = False
            self._leg_started = now
            self._holding_approach_origin = view["qpos_rad"]
            self._transition("moving", "target_submitted_after_arm_receipt", now)
        elif self._state == "moving":
            if view["release_terminal"]:
                self._abort(now, f"owner_terminal_withdrawal:{view['reason']}")
            elif view["phase"] == "authority_exhausted":
                self._abort(now, "insufficient_authority_no_automatic_retry")
            elif self.holding_search is not None and view["phase"] in {"track", "brake"}:
                self._advance_holding(view, now, driver_kd)
            elif view["phase"] == "soft_hold" and view["motion_phase"] == "soft_hold":
                # Always unload first. A stop point captured with a conditioner
                # tail or soft-hold torque cannot establish final arrival.
                self.controller.submit("withdraw", now_sec=now)
                self._transition(
                    "releasing", "stop_point_captured_waiting_for_zero_dispatch", now
                )
            elif (
                self._breakaway_allowed
                and not self._breakaway_consumed
                and view["breakaway_gate"] == "ready"
            ):
                self.controller.submit("authorize_breakaway", now_sec=now)
                self._breakaway_consumed = True

    def _holding_status(self, *, detailed=False):
        if self.holding_search is None:
            return None
        return {"enabled": True, "search": None if self._holding is None else self._holding.status(detailed=detailed),
                "probe_command_nm": self._holding_probe_nm,
                "stage": ("withdrawal" if self._state == "releasing" else self._state if self._state != "moving"
                    else "reacquire" if self._holding_probe_nm is None else "fixed_command_observation"),
                "attempt_count": self._holding_attempts, "max_attempts": self.holding_search.max_trials,
                "reacquire_bias_nm": self._holding_reacquire_bias_nm,
                "sample_pose_rad": self._holding_pose,
                "nominal_target_rad": self._steps[min(self._index, len(self._steps)-1)]["target_qpos_rad"],
                "quantity": "software_command_not_measured_force", "auto_apply": False}

    def _advance_holding(self, view, now, driver_kd, *, passive=False):
        config, j = self.holding_search, self._joint_index
        q, stamp = view["qpos_rad"], view["feedback_timestamp_sec"]
        target = self._steps[self._index]["target_qpos_rad"]
        point = target if self._holding_pose is None else self._holding_pose
        expected_kd = (self.condition.driver_kd_nm_s_per_rad if passive else
            tuple(max(a,b) for a,b in zip(self.condition.driver_kd_nm_s_per_rad,
            self.controller.settings.staged_assist.motion_control.drive_driver_kd_nm_s_per_rad)))
        if passive:
            if any(abs(v) > self.controller.limits.max_joint_velocity_rad_s for v in view["raw_qvel_rad_s"]):
                self._abort(now, "holding_zero_observation_velocity_exceeded")
                return
            if view["operator_phase"] != "inactive" or not self._zero_dispatch() or tuple(driver_kd) != expected_kd:
                self._abort(now, "holding_zero_observation_command_changed")
                return
        if self._holding_probe_nm is None:
            if (abs(q[j]-point[j]) > min(config.movement_rad, self.acquisition_arrival_tolerance_rad)
                or view["arrival_stop_gate"] != "ready" or tuple(driver_kd) != expected_kd
                or not passive and view["motion_reason"] not in {"bounded_reference_tracking", "target_reached_braking"}):
                return
            if self._holding is None:
                self._holding = HoldingForceSearch(config, seed_nm=self._sent_delta[j],
                    cap_nm=self.controller.settings.staged_assist.tracking_overlay_cap_nm)
                self._holding_pose = q
            elif max(abs(a-b) for a,b in zip(q, self._holding_pose)) > config.movement_rad:
                return  # Do not compare different full-arm poses as one point.
            if self._holding_attempts >= config.max_trials:
                self._abort(now, "holding_search_incomplete")
                return
            self._holding_attempts += 1
            self._holding_probe_nm = self._holding.candidate_nm
            self._holding_observation = {"started_at_sec": None, "anchor_qpos_rad": q,
                "minimum_rad": q, "maximum_rad": q, "ramp_frozen": False,
                "requested_command_nm": self._holding_probe_nm,
                "observation_mode": "passive_zero_assist" if passive else "active_fixed_command",
                "transition_start_command_nm": self._sent_delta[j],
                "transition_minimum_command_nm": self._sent_delta[j],
                "transition_maximum_command_nm": self._sent_delta[j],
                "transition_minimum_rad": q, "transition_maximum_rad": q,
                "approach_from_qpos_rad": self._holding_approach_origin,
                "reacquired_qpos_rad": q,
                "reacquired_at_sec": now}
            return
        observation = self._holding_observation
        if not passive and view["motion_reason"] != "calibration_fixed_command_observation":
            # Real speed braking is never turned into a signed force sample.
            # Resume only via the existing finite governor/point budgets.
            observation.update(outcome="censored_owner_brake", completed_at_sec=now)
            self._events.append({"time_sec": now, "state": "holding_observation", **observation})
            self._holding_probe_nm = self._holding_observation = None
            self._holding_reacquire_bias_nm = 0.0  # A real brake invalidates the probe handoff.
            self._holding_approach_origin = q
            return
        if (observation["started_at_sec"] is not None
            and now - observation["started_at_sec"] > config.observation_timeout_sec):
            self._abort(now, "holding_search_incomplete")
            return
        # Feedback precedes THIS receipt. Only the preceding complete command
        # can have caused it; starting at this receipt would backdate evidence.
        previous = self._previous_sent_delta[j]
        observation["transition_minimum_command_nm"] = min(observation["transition_minimum_command_nm"], previous)
        observation["transition_maximum_command_nm"] = max(observation["transition_maximum_command_nm"], previous)
        observation["transition_minimum_rad"] = tuple(min(a,b) for a,b in zip(observation["transition_minimum_rad"],q))
        observation["transition_maximum_rad"] = tuple(max(a,b) for a,b in zip(observation["transition_maximum_rad"],q))
        plateau = (self._previous_completed is not None and stamp > self._previous_completed
                   and abs(previous-self._holding_probe_nm) <= 1e-9
                   and abs(self._sent_delta[j]-self._holding_probe_nm) <= 1e-9
                   and all(abs(self._previous_sent_delta[i]) <= 1e-9 and abs(self._sent_delta[i]) <= 1e-9
                           for i in range(6) if i != j)
                   and self._previous_sent_driver_kd == expected_kd
                   and tuple(driver_kd) == expected_kd)
        if not plateau:
            observation["started_at_sec"] = None
            if (not observation["ramp_frozen"]
                and abs(q[j]-observation["anchor_qpos_rad"][j]) >= config.movement_rad
                and (self._holding_probe_nm-previous)*(q[j]-observation["anchor_qpos_rad"][j]) > 0):
                # Stop the ramp at the command that was actually in force.
                # Subsequent evidence must come from its constant plateau.
                observation["ramp_frozen"] = True
                self._holding_probe_nm = previous
            return
        if observation["started_at_sec"] is None:
            if max(abs(a-b) for a,b in zip(q,point)) > config.movement_rad:
                self._refine_holding_transition(view, now, driver_kd, "censored_plateau_outside_point")
                if passive:
                    self._abort(now, "holding_search_incomplete")
                return
            observation.update(started_at_sec=stamp, first_causal_receipt_at_sec=self._previous_completed,
                anchor_qpos_rad=q, minimum_rad=q, maximum_rad=q)
            return
        observation["minimum_rad"] = tuple(min(a,b) for a,b in zip(observation["minimum_rad"],q))
        observation["maximum_rad"] = tuple(max(a,b) for a,b in zip(observation["maximum_rad"],q))
        delta = q[j]-observation["anchor_qpos_rad"][j]
        elapsed = stamp-observation["started_at_sec"]
        outcome = None
        velocity = view["control_qvel_rad_s"][j]
        if (elapsed >= self.controller.settings.staged_assist.motion_control.velocity_window_sec
            and abs(delta) >= config.movement_rad and delta*velocity > 0
            and delta*view["raw_qvel_rad_s"][j] > 0):
            outcome = "positive_motion" if delta > 0 else "negative_motion"
        elif elapsed >= config.observation_sec:
            span = max(b-a for a,b in zip(observation["minimum_rad"],observation["maximum_rad"]))
            if (span <= config.movement_rad and view["arrival_stop_gate"] == "ready"
                and abs(q[j]-point[j]) <= config.movement_rad):
                outcome = "stationary"
            elif view["arrival_stop_gate"] == "ready" and abs(q[j]-point[j]) > config.movement_rad:
                # A preload release can move during the ramp or briefly at
                # its plateau and then stop outside the sampled point.
                # Reacquire and repeat; this is not evidence of sustained
                # movement at the final plateau or an in-pose hold sample.
                self._refine_holding_transition(view, now, driver_kd, "censored_settled_outside_point")
                if passive:
                    self._abort(now, "holding_search_incomplete")
                return
        if outcome is None:
            return
        observation.update(completed_at_sec=now, final_qpos_rad=q,
            gravity_nm=view["gravity_command_nm"],
            driver_kd_nm_s_per_rad=tuple(driver_kd),
            net_approach_direction=(0 if abs(observation["reacquired_qpos_rad"][j]-observation["approach_from_qpos_rad"][j])
                < self.controller.settings.staged_assist.motion_control.position_resolution_rad else
                1 if observation["reacquired_qpos_rad"][j] > observation["approach_from_qpos_rad"][j] else -1),
            nominal_target_rad=target)
        # Guard against a changing support posture during the observation.
        if max(abs(q[i]-self._holding_pose[i]) for i in range(6) if i != j) > config.movement_rad:
            self._abort(now, "holding_search_incomplete")
            return
        self._holding.record(command_nm=self._holding_probe_nm, outcome=outcome, evidence=dict(observation))
        self._events.append({"time_sec": now, "state": "holding_observation", **observation,
            "command_nm": self._holding_probe_nm, "outcome": outcome})
        self._holding_reacquire_bias_nm = self._holding_return_command()
        self._holding_probe_nm = self._holding_observation = None
        self._holding_approach_origin = q
        if self._holding.state == "complete":
            self._holding_leg_result = {**self._holding.status(detailed=True),
                "sample_pose_rad": self._holding_pose, "nominal_target_rad": target,
                "observation_mode": "passive_zero_assist" if passive else "active_fixed_command"}
            self.controller.submit("withdraw", now_sec=now)
            self._transition("releasing", "holding_point_verified_withdrawal_pending", now)
        elif self._holding.state == "incomplete" or passive and self._holding.candidate_nm != 0:
            self._abort(now, "holding_search_incomplete")

    def _holding_return_command(self):
        command = self._holding.return_command_nm
        return self._sent_delta[self._joint_index] if command is None else command

    def _refine_holding_transition(self, view, now, driver_kd, outcome):
        j, q = self._joint_index, view["qpos_rad"]
        observation = {**self._holding_observation, "outcome": outcome,
            "completed_at_sec": now, "final_qpos_rad": q,
            "actual_plateau_command_nm": self._previous_sent_delta[j],
            "first_causal_receipt_at_sec": self._previous_completed,
            "gravity_nm": view["gravity_command_nm"],
            "driver_kd_nm_s_per_rad": tuple(driver_kd),
            "nominal_target_rad": self._steps[self._index]["target_qpos_rad"]}
        self._events.append({"time_sec": now, "state": "holding_observation", **observation})
        # A different support posture cannot guide a same-pose command search.
        if any(max(abs(observation[key][i]-self._holding_pose[i])
                   for key in ("transition_minimum_rad", "transition_maximum_rad")) > self.holding_search.movement_rad
               for i in range(6) if i != j):
            self._abort(now, "holding_search_incomplete")
            return
        self._holding.record_transition(command_nm=self._previous_sent_delta[j],
            requested_command_nm=observation["requested_command_nm"], evidence=observation)
        self._holding_reacquire_bias_nm = self._holding_return_command()
        self._holding_probe_nm = self._holding_observation = None
        self._holding_approach_origin = q
        if self._holding.state == "incomplete":
            self._abort(now, "holding_search_incomplete")

    def result_record(self) -> dict[str, Any]:
        """Return one trace record; the caller reuses its existing log writer."""
        with self._lock:
            if (
                self._state not in {"complete", "failed", "cancelled"}
                or self._pending is not None
            ):
                raise ValueError(
                    "batch result requires a terminal state and consumed receipt"
                )
            return {
                "kind": "calibration_batch_result",
                "schema": "b601.calibration-batch-result.v1",
                "recorded_monotonic_sec": self._last_completed,
                **self.status(),
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "reason": self._reason,
                "step_index": self._index,
                "batch_id": self._batch_id,
                "outcome": calibration_outcome(self._stop_destination or self._state, self._reason),
                "step_count": len(self._steps),
                "calibration_trial": self._marker(),
                "completed_legs": copy.deepcopy(self._completed_legs),
                "events": copy.deepcopy(self._events),
                "active_command_exposure_nm_sec": self._active_exposure,
                "withdrawal_command_exposure_nm_sec": self._release_exposure,
                "leg_command_exposure_nm_sec": self._leg_exposure,
                "max_leg_command_exposure_nm_sec": self.limits.max_leg_command_exposure_nm_sec,
                "max_batch_command_exposure_nm_sec": self.limits.max_active_command_exposure_nm_sec,
                "batch_command_exposure_nm_sec": self._active_exposure + self._release_exposure,
                "batch_budget_counts_withdrawal": self.limits.max_leg_command_exposure_nm_sec is not None,
                "acquisition_arrival_tolerance_rad": self.acquisition_arrival_tolerance_rad,
                "tracking_arrival_tolerance_rad": self.controller.settings.staged_assist.settle_error_rad,
                "budget_plan": dict(self._budget_plan),
                "exposure_semantics": "software_command_not_mechanical_energy_or_measured_torque",
                "pending_dispatch": self._pending is not None,
                "hardware_execution_allowed": False,
                "holding_calibration": self._holding_status(detailed=True),
            }

    def owner_snapshot(self) -> dict[str, Any]:
        """Bounded control-tick view; detailed evidence stays in status()."""
        with self._lock:
            return {
                "state": self._state,
                "reason": self._reason,
                "step_index": self._index,
                "holding_calibration": self._holding_status(),
            }
