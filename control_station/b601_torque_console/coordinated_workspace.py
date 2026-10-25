"""Opt-in coordinated acquisition intents for the existing six-axis owner."""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import asdict

from robot_motion.rebot_b601_rs.calibration import finite, six
from robot_motion.rebot_b601_rs.coordinated_holding import (
    AXES,
    ZERO,
    CoordinatedPoseHolding,
)
from robot_motion.rebot_b601_rs.holding_calibration import HoldingForceSearch
from robot_motion.rebot_b601_rs.joint_motion import slew_driver_damping

from .calibration_session import calibration_outcome, calibration_receipt_rejection

# Only acquisition capability/quality stops permit a new, bounded return.
# Authority loss, transport/receipt faults and motion guards never do.
RETURNABLE_STOPS = frozenset({
    "leg_timeout", "leg_command_exposure_budget", "active_command_exposure_budget",
    "batch_timeout", "holding_search_incomplete", "stopped_short_of_target",
})


class CoordinatedWorkspaceController:
    """One batch/receipt consumer; no backend, thread, or second CAN owner.

    Normal point/axis handoffs retain explicit pose bias. Only batch completion,
    operator withdrawal, expiry or a fault initiates all-axis withdrawal.
    """

    controller_mode = "automatic_calibration"

    def __init__(self, settings, config, workspace, *, command_field_source=None):
        workspace.sweep_speeds("joint3", config)
        if workspace.holding_search is None:
            raise ValueError("coordinated acquisition requires explicit holding search")
        if config.max_sweep_leg_command_exposure_nm_sec is None:
            raise ValueError("coordinated acquisition requires the explicit per-leg exposure budget")
        self.settings, self.limits = settings, settings.limits
        self.config, self.workspace = config, workspace
        self.pose = CoordinatedPoseHolding(settings, workspace.coordinated_holding,
            arrival_tolerance_rad=(workspace.holding_search.movement_rad / 2
                if workspace.holding_arrival_tolerance_rad is None else workspace.holding_arrival_tolerance_rad),
            command_field_source=command_field_source)
        self._field_pose_indices = None
        self._condition = self._position_limits = None
        self._ready = self._closed = self._start = False
        self._human = self._permit = self._origin = self._started = None
        self._phase, self._reason = "idle", "waiting_for_operator"
        self._destination = None
        self._tasks, self._steps = [], []
        self._index = self._leg_index = self._segments = 0
        self._leg_started = self._quiet_since = None
        self._sweep_started = None
        self._search = self._probe = self._observation = None
        self._attempts = 0
        self._sample_pose = None
        self._nominal_target = None
        self._pose_window = None
        self._leg_anchor = None
        self._pending = self._receipt = None
        self._last_commands = None
        self._receipt_feedback_stamp = None
        self._last_delta = ZERO
        self._last_kd = None
        self._last_sequence = None
        self._exposure = self._leg_exposure = self._sweep_exposure = 0.0
        self._zero_base = False
        self._batch_id = None
        self._records, self._acquisitions, self._legs, self._pose_observations = [], [], [], []
        self._audit = {}
        self._supported_origin = None
        self._return_and_disable = False
        self._return_context = None
        self._return_completed = False
        self._return_route_complete = False
        self._return_completion_observation = None
        self._return_revoked = None
        self._manual_return_requested = None
        self._dispatch_withdrawal = None
        self._lock = threading.RLock()

    @property
    def _active(self):
        return self._phase not in {"idle", "complete", "failed", "cancelled"}

    @property
    def _continuous_identification(self):
        return self.workspace.holding_sampling_plan == "identification"

    def configure_context(self, condition, position_limits_rad):
        with self._lock:
            if self._condition is not None:
                raise ValueError("coordinated context is immutable")
            limits = [tuple(v) for v in position_limits_rad]
            for i, bounds in zip(AXES, (self.workspace.joint2_acquisition_range_deg, self.workspace.joint3_acquisition_range_deg)):
                lo, hi = map(math.radians, bounds)
                if not limits[i][0] <= lo < hi <= limits[i][1]:
                    raise ValueError("coordinated range exceeds owner limits")
                limits[i] = (lo, hi)
            field_pose_indices = None
            if self.pose.command_field_source is not None:
                field_pose_indices = self.pose.command_field_source.validate_context(condition,
                    setup_label=self.workspace.setup_label, pose_centres_rad=self.workspace.pose_centres())
            self._condition, self._position_limits = condition, tuple(limits)
            self._field_pose_indices = field_pose_indices

    def owner_ready(self):
        self._ready = True

    def set_human_intervention(self, human):
        if type(human) is not bool:
            raise TypeError("human intervention must be boolean")
        with self._lock:
            self._human = human
            if human:
                self._return_revoked = "human_intervention"
                self._manual_return_requested = None
                self._start = False
                if self._active:
                    self._stop("human_intervention", "cancelled")

    def submit(self, action, *, now_sec, **fields):
        now = finite(now_sec, "coordinated intent")
        with self._lock:
            allowed = ({"return_and_disable"} if action == "arm" else
                {"withdraw_source"} if action == "withdraw" else {"visible"} if action == "visibility" else set())
            if set(fields) - allowed:
                raise ValueError("workspace accepts no manual targets or torque fields")
            if action == "arm":
                return_and_disable = fields.get("return_and_disable", False)
                if type(return_and_disable) is not bool:
                    raise TypeError("return_and_disable must be boolean")
                if self._phase != "idle" or self._start or self._manual_return_requested is not None or self._closed or not self._ready or self._human is not False or self._condition is None:
                    raise ValueError("coordinated start requires idle owner and no human intervention")
                if self.pose.last_time is None or any(abs(self.pose.q[i]) > math.radians(self.workspace.maximum_start_angle_deg) for i in AXES):
                    raise ValueError("coordinated start requires the near-zero start window")
                if return_and_disable and (self._supported_origin is None or any(
                    abs(self._supported_origin[i]) > math.radians(self.workspace.maximum_start_angle_deg) for i in AXES)):
                    raise ValueError("supported origin requires a fresh stationary zero-assist receipt")
                self._return_and_disable = return_and_disable
                self._return_revoked = None
                self._start, self._permit = True, now
            elif action in {"heartbeat", "visibility"}:
                if not self._active and not self._start and self._manual_return_requested is None:
                    raise ValueError("workspace permit requires an active or pending batch")
                if action == "visibility" and type(fields.get("visible")) is not bool:
                    raise ValueError("visibility must be boolean")
                self._permit = now
            elif action == "return_to_origin":
                reason = self._manual_return_gate()
                if reason != "ready":
                    raise ValueError(f"supported return unavailable: {reason}")
                if self.pose.last_time is None or not 0 <= now-self.pose.last_time <= self.pose.motion.max_control_gap_sec:
                    raise ValueError("supported return requires fresh owner feedback")
                self._manual_return_requested = (self._phase, self._reason, now)
                self._permit = now
                self._return_revoked = None
            elif action in {"withdraw", "zero"}:
                source = fields.get("withdraw_source", "operator_button")
                reasons = {"operator_button": "operator_withdraw", "page_unload": "page_unload", "page_hidden": "page_hidden", "owner_cancel": "owner_cancel"}
                if source not in reasons:
                    raise ValueError("unsupported withdrawal source")
                self._return_revoked = "operator_zero" if action == "zero" else reasons[source]
                self._manual_return_requested = None
                self._start = False
                if self._active:
                    self._stop("operator_zero" if action == "zero" else reasons[source], "cancelled")
            else:
                raise ValueError("unsupported workspace action")

    def _transition(self, phase, reason):
        self._phase, self._reason = phase, reason
        self._quiet_since = None
        self._records.append({"kind": "workspace_calibration_transition", "recorded_monotonic_sec": self.pose.last_time,
            "phase": phase, "reason": reason, "task_index": self._index, "batch_id": self._batch_id})

    def _stop(self, reason, destination):
        finishing_return = (self._return_context is not None and self._return_route_complete
            and reason == self._return_context["reason"] and destination == self._return_context["state"])
        if not finishing_return and reason not in RETURNABLE_STOPS and reason != "all_targets_and_withdrawals_confirmed":
            self._return_revoked = reason
        if self._phase != "stopping":
            self._withdraw_started = self.pose.last_time
            self._destination = destination
            self._transition("stopping", reason)
        self._start = False
        if self._observation is not None:
            self._records.append({"kind": "coordinated_holding_observation", "recorded_monotonic_sec": self.pose.last_time,
                **self._observation, "outcome": "censored_batch_withdrawal", "reason": reason,
                "actual_delta_nm": self._last_delta, "final_qpos_rad": self.pose.q})
        self._probe = self._observation = None

    def _begin_task(self):
        self._pose_window = None
        if self._index == len(self._tasks):
            self._return_route_complete = True
            if self._return_context is not None:
                self._stop(self._return_context["reason"], self._return_context["state"])
            else:
                self._stop("all_targets_and_withdrawals_confirmed", "complete")
            return
        task = self._tasks[self._index]
        self._leg_anchor = self.pose.q
        self._leg_started = self.pose.last_time
        self._leg_exposure = 0.0
        if task["kind"] in {"observe", "return_observe"}:
            self._transition(task["kind"], "coordinated_pose_observation")
            return
        if task["kind"] == "sweep":
            self._sweep_exposure = 0.0
            self._sweep_started = self.pose.last_time
            plan = self.workspace.sweep_plan(condition=self._condition, joint_name=task["joint"],
                anchor_qpos_rad=self._nominal_target or self.pose.goal, position_limits_rad=self._position_limits,
                calibration=self.config)
            self._steps = plan["steps"]
            for step in self._steps:
                step["trial_id"] = f"cw{self._index}:{task['joint']}:{step['trial_id']}"
            self._leg_index, self._legs = 0, []
            self._begin_leg()
        else:
            goal = list(self._nominal_target or self.pose.goal)
            i = int(task["joint"][-1]) - 1
            distance = task["target"] - self.pose.q[i]
            goal[i] = self.pose.q[i] + max(-self.workspace.maximum_positioning_step_rad,
                                         min(self.workspace.maximum_positioning_step_rad, distance))
            self.pose.set_goal(goal,
                max_reference_velocity_rad_s=(self.workspace.positioning_velocity_rad_s
                    if (self._return_and_disable or self._return_context is not None) and task["kind"] == "return" else None),
                support_soft_speed_recovery=self.workspace.positioning_support_soft_speed_recovery)
            self._nominal_target = self.pose.goal
            self._segments += 1
            if self._segments > 64:
                self._stop("workspace_positioning_segment_budget", "failed")
            else:
                self._transition(task["kind"], "coordinated_pose_positioning")

    def _begin_return(self):
        """Reuse the admitted two-axis return route, budgets and dispatch owner."""
        partial = None
        if self._index < len(self._tasks) and self._tasks[self._index]["kind"] == "sweep":
            partial = {"task": copy.deepcopy(self._tasks[self._index]), "partial_stage": True,
                "result": {"state": self._destination, "reason": self._reason,
                    "completed_legs": copy.deepcopy(self._legs),
                    "holding_calibration": self._holding_status(detailed=True)}}
        self._return_context = {"state": self._destination, "reason": self._reason,
            "task_index": self._index, "task_count": len(self._tasks),
            "started_at_sec": self.pose.last_time, "partial_acquisition": partial}
        self._records.append({"kind": "workspace_return_started", "recorded_monotonic_sec": self.pose.last_time,
            **copy.deepcopy(self._return_context), "target_qpos_rad": self._supported_origin})
        # This is the same segmented J2-then-J3 route used by normal completion,
        # not a new straight-line six-axis move or an arbitrary-pose planner.
        self._tasks = self.workspace.tasks(self._supported_origin)[-3:]
        self._return_route_complete = self._return_completed = False
        self._return_completion_observation = None
        self._index = self._segments = 0
        self._nominal_target = self.pose.q
        self._search = self._probe = self._observation = self._sample_pose = None
        self._begin_task()

    def _manual_return_gate(self):
        if self._closed or not self._ready:
            return "owner_unavailable"
        if self._return_context is not None or self._manual_return_requested is not None:
            return "return_already_requested"
        if self._human is not False:
            return "human_intervention_or_unknown"
        if self._supported_origin is None or self._condition is None:
            return "origin_unavailable"
        if any(abs(self._supported_origin[i]) > math.radians(self.workspace.maximum_start_angle_deg) for i in AXES):
            return "origin_outside_start_window"
        if self._phase not in {"idle", "complete", "failed", "cancelled"} or self._start:
            return "stop_acquisition_first"
        if self._phase == "failed" and self._reason not in RETURNABLE_STOPS:
            return "fault_requires_resolution"
        if self._phase == "cancelled" and self._reason not in {"operator_withdraw", "operator_zero"}:
            return "authority_revoked"
        if self._started is not None and self.pose.last_time-self._started >= self.workspace.max_workspace_sec:
            return "workspace_timeout"
        if self._exposure + self.workspace.coordinated_holding.total_assist_cap_nm*self.pose.motion.max_control_gap_sec >= self.workspace.max_command_exposure_nm_sec:
            return "workspace_command_exposure_budget"
        if any(not lo <= q <= hi for q, (lo, hi) in zip(self.pose.q, self._position_limits)):
            return "workspace_position_envelope"
        if any(abs(self.pose.q[i]-self._supported_origin[i]) > self.limits.max_support_drift_rad for i in (0, 3, 4, 5)):
            return "workspace_support_excursion"
        if not self._zero_base or not self.pose.quiet():
            return "waiting_for_zero_base_and_stillness"
        return "ready"

    def _begin_leg(self):
        self._leg_anchor = self.pose.q
        step = self._steps[self._leg_index]
        self.pose.set_goal(step["target_qpos_rad"],
            max_reference_velocity_rad_s=step["max_reference_velocity_rad_s"])
        self._nominal_target = self.pose.goal
        self._search = self._probe = self._observation = self._sample_pose = None
        self._attempts = 0
        self._leg_started, self._leg_exposure = self.pose.last_time, 0.0
        self._transition("sweep", "coordinated_reacquire")

    def _at_goal(self, radius, *, require_quiet=True):
        # Only J2/J3 have position authority. Passive axes remain inside their
        # safety envelope; static acquisition checks their full position window.
        return (self.pose.goal is not None and (not require_quiet or self.pose.quiet())
                and max(abs(self.pose.q[i]-self.pose.goal[i]) for i in AXES) <= radius)

    def _arrival_status(self):
        radius = self.pose.arrival_tolerance_rad
        goal = self.pose.goal
        quiet = self.pose.quiet()
        error = None if goal is None else tuple(q-g for q, g in zip(self.pose.q, goal))
        return {"controlled_joint_names": ["joint2", "joint3"],
            "tolerance_rad": radius, "six_axis_stationary": quiet,
            "controlled_error_rad": None if error is None else [error[i] for i in AXES],
            "passive_offset_from_nominal_rad": None if error is None else {f"joint{i+1}": error[i] for i in (0, 3, 4, 5)},
            "reason": ("no_goal" if goal is None else "six_axis_not_stationary" if not quiet
                       else "controlled_target_error" if max(abs(error[i]) for i in AXES) > radius else "ready")}

    def _passive_sample_changed(self):
        return (self._sample_pose is not None and any(
            abs(self.pose.q[i]-self._sample_pose[i]) > self.workspace.holding_search.movement_rad
            for i in (0, 3, 4, 5)))

    def step(self, *, now_sec, qpos_rad, qvel_rad_s, feedback_sequence, control_now_sec):
        with self._lock:
            now, clock = finite(now_sec, "feedback time"), finite(control_now_sec, "control time")
            gap = self.pose.motion.max_control_gap_sec
            if self._closed or self._pending is not None:
                raise ValueError("workspace closed or previous dispatch not confirmed")
            if type(feedback_sequence) is not int or feedback_sequence < 0 or self._last_sequence is not None and feedback_sequence <= self._last_sequence:
                self.recording_failed("workspace_feedback_sequence")
                raise ValueError(self._reason)
            if not now <= clock <= now + gap:
                self.recording_failed("workspace_feedback_clock_or_gap")
                raise ValueError(self._reason)
            try:
                dt = self.pose.observe(now, qpos_rad, qvel_rad_s,
                    idle=self._phase == "idle" and not self._start)
            except ValueError:
                self.recording_failed("workspace_feedback_clock_or_gap")
                raise
            self._last_sequence = feedback_sequence
            if self._manual_return_requested is not None:
                state, reason, requested = self._manual_return_requested
                if clock-requested > self.config.prepare_timeout_sec:
                    self._manual_return_requested = None
                    self._return_revoked = "return_preparation_timeout"
                    self._records.append({"kind": "workspace_return_rejected", "recorded_monotonic_sec": clock,
                        "reason": self._return_revoked})
                elif self._zero_base and self.pose.quiet():
                    self._manual_return_requested = None
                    if self._started is None:
                        self._started, self._origin = clock, self.pose.q
                        self._batch_id = f"supported-return-{clock:.9f}"
                    self._destination = "cancelled" if state == "idle" else state
                    self._reason = "operator_return" if state == "idle" else reason
                    self._begin_return()
            if self._start:
                if any(abs(self.pose.q[i]) > math.radians(self.workspace.maximum_start_angle_deg) for i in AXES):
                    self._start = False
                    raise ValueError("workspace origin moved outside the near-zero start window")
                self._start, self._origin, self._started = False, self.pose.q, clock
                self._batch_id = f"coordinated-{clock:.9f}"
                self._tasks = self.workspace.tasks(self._supported_origin if self._return_and_disable else self._origin)
                self._transition("preparing", "coordinated_batch_authorized")
                self._records.append({**self.result_record(), "kind": "automatic_calibration_started", "recorded_monotonic_sec": clock})
            if self._active:
                if any(not lo <= value <= hi for value, (lo, hi) in zip(self.pose.q, self._position_limits)):
                    self.recording_failed("workspace_position_envelope")
                elif any(abs(self.pose.q[i] - self._origin[i]) > self.limits.max_support_drift_rad for i in (0, 3, 4, 5)):
                    self.recording_failed("workspace_support_excursion")
                elif max(abs(v) for v in self.pose.raw) > self.limits.max_joint_velocity_rad_s:
                    self.recording_failed("joint_velocity_exceeded")
                elif self._leg_anchor is not None and self._index < len(self._tasks) and any(
                    abs(self.pose.q[i]-self._leg_anchor[i]) > (
                        self.limits.max_excursion_rad if self._tasks[self._index].get("joint") == f"joint{i+1}"
                        else self.limits.max_support_drift_rad) for i in AXES):
                    self.recording_failed("coordinated_leg_excursion")
                elif self._human is not False:
                    self._stop("human_intervention", "cancelled")
                elif self._permit is None or not 0 <= clock-self._permit <= self.limits.command_timeout_sec:
                    self._stop("gui_permit_expired", "cancelled")
                elif clock-self._started > self.workspace.max_workspace_sec:
                    self._stop("workspace_timeout", "failed")
                else:
                    pending = sum(abs(v) for v in self._last_delta) * max(0, now-self._receipt["completed_at_sec"]) if self._receipt else 0
                    reserve = self.workspace.coordinated_holding.total_assist_cap_nm * gap
                    if self._exposure+pending+reserve >= self.workspace.max_command_exposure_nm_sec:
                        self._stop("workspace_command_exposure_budget", "failed")
                    elif self._phase == "preparing" and clock-self._started > self.config.prepare_timeout_sec:
                        self._stop("preparation_timeout", "failed")
                    elif self._phase not in {"preparing", "stopping"} and now-self._leg_started > self.config.max_sweep_leg_sec:
                        self._stop("leg_timeout", "failed")
                    elif self._phase == "sweep" and self._leg_exposure+pending+reserve >= self.config.max_sweep_leg_command_exposure_nm_sec:
                        self._stop("leg_command_exposure_budget", "failed")
                    elif self._phase == "sweep" and self._sweep_exposure+pending+reserve >= self.config.max_sweep_command_exposure_nm_sec:
                        self._stop("active_command_exposure_budget", "failed")
                    elif self._phase == "sweep" and now-self._sweep_started > self.config.max_batch_sec:
                        self._stop("batch_timeout", "failed")
            prepare_confirmed = self._zero_base
            if self.workspace.coordinated_holding.motor_local_recovery:
                prepare_confirmed = (max(abs(v) for v in self._last_delta) <= 1e-9 and self._last_commands is not None
                    and self._last_commands["overlay_nm"] == ZERO and self._last_kd == self._drive_kd_target())
            if self._phase == "preparing" and self.pose.quiet() and prepare_confirmed:
                # Parameter changes and the first goal belong after preparation,
                # never to the GUI Start request's possibly moving feedback.
                self.pose.set_goal(self._origin)
                self._begin_task()
            if self._phase == "failed" and (calibration_outcome(self._phase, self._reason)["severity"] == "error" or not self._zero_base):
                return ZERO, self._make_audit(ZERO, {}, fault=self._reason)
            permitted = self._active and self._phase not in {"preparing", "stopping"}
            joint = self._tasks[self._index].get("joint", "joint2") if self._index < len(self._tasks) else "joint2"
            try:
                command, audit = self.pose.step(sequence=feedback_sequence, dt=dt, permitted=permitted,
                    probe_joint=int(joint[-1])-1 if self._probe is not None else None, probe_nm=self._probe,
                    # Once a point is frozen, returning after a probe is local
                    # force search, not another transit. Its departure bias
                    # may be exactly the command that moved away from the point.
                    point_search=(not self._continuous_identification and self._phase == "sweep" and (
                        self.workspace.holding_sampling_plan != "coverage" or self._search is not None)),
                    # Reacquiring the frozen point is continuous transport.
                    # Only actual plateau observations own force bisection.
                    point_reacquire=(self.workspace.coordinated_holding.motor_local_recovery
                        and self._phase == "sweep" and self._sample_pose is not None and self._probe is None),
                    cruise_joint=(int(joint[-1])-1 if self._continuous_identification and self._phase == "sweep"
                        and self._leg_index < len(self._steps)
                        and self._steps[self._leg_index]["role"] in {"training", "validation"} else None),
                    field_pose_index=(self._field_pose_indices[self._tasks[self._index]["pose"]]
                        if self._field_pose_indices is not None and self._index < len(self._tasks)
                        and self._phase == "sweep" else None),
                    field_driver_kd=self._last_kd)
            except ValueError as exc:
                self.recording_failed(str(exc))
                return ZERO, self._make_audit(ZERO, {}, fault=self._reason)
            self._pending = (feedback_sequence, now, clock, command)
            self._audit = self._make_audit(command, audit)
            return command, self._audit

    def validate_final_command(self, *, gravity_nm, total_nm):
        delta = tuple(t-g for t, g in zip(six(total_nm, "total"), six(gravity_nm, "gravity")))
        if any(abs(delta[i]) > 1e-9 for i in (0, 3, 4, 5)) or sum(abs(v) for v in delta) > self.workspace.coordinated_holding.total_assist_cap_nm + 1e-9:
            self.recording_failed("coordinated_final_command_envelope")
            raise ValueError(self._reason)

    def _drive_kd_target(self):
        return tuple(max(b, a, c) for b, a, c in zip(self._condition.driver_kd_nm_s_per_rad,
            self.workspace.joint2_drive_kd, self.workspace.joint3_drive_kd))

    def active_driver_damping(self, base_kd, *, final_command_delta_nm=None):
        base = six(base_kd, "base damping")
        target = base
        if self._active and (self._phase != "preparing" or self.workspace.coordinated_holding.motor_local_recovery):
            delta = self._last_delta if final_command_delta_nm is None else six(final_command_delta_nm, "final delta")
            if self._phase != "stopping" or any(abs(v) > 1e-9 for v in delta) or any(abs(v) > 1e-9 for v in self._last_delta):
                target = tuple(max(b, a) for b, a in zip(base, self._drive_kd_target()))
        dt = (self.settings.nominal_control_period_sec if self._receipt_feedback_stamp is None else
              self.pose.last_time-self._receipt_feedback_stamp)
        return slew_driver_damping(self._last_kd or base, target,
            rate_per_sec=self.pose.motion.drive_damping_rise_per_sec, dt_sec=dt)

    def validate_dispatch_start(self, *, feedback_sequence, feedback_timestamp_sec, now_sec=None):
        """False vetoes this unsent frame on normal lease expiry; faults raise."""
        with self._lock:
            clock = time.monotonic() if now_sec is None else finite(now_sec, "dispatch start")
            if self._pending is None or self._pending[:2] != (feedback_sequence, feedback_timestamp_sec) or not feedback_timestamp_sec <= clock <= feedback_timestamp_sec + self.pose.motion.max_control_gap_sec:
                self.recording_failed("workspace_dispatch_start_identity_or_deadline")
                raise ValueError(self._reason)
            nonzero = any(abs(v) > 1e-9 for v in self._pending[3])
            if (nonzero and self._human is False and not self._closed and self._active
                and self._phase not in {"preparing", "stopping"} and self._permit is not None
                and clock-self._permit > self.limits.command_timeout_sec):
                self._dispatch_withdrawal = {"reason": "gui_permit_expired", "feedback_sequence": feedback_sequence,
                    "feedback_timestamp_sec": feedback_timestamp_sec, "observed_at_sec": clock,
                    "last_permit_at_sec": self._permit, "permit_age_sec": clock-self._permit,
                    "permit_timeout_sec": self.limits.command_timeout_sec,
                    "command_sent": False, "withdrawal_receipt_confirmed": False}
                self._stop("gui_permit_expired", "cancelled")
                # The owner must leave the loop without sending this pending
                # command. Physical withdrawal belongs to its existing hold/
                # cleanup protocol; cancellation is NOT a zero receipt.
                self._transition("cancelled", "gui_permit_expired")
                self._records.append({"kind": "workspace_dispatch_withdrawn", "recorded_monotonic_sec": clock,
                    **self._dispatch_withdrawal})
                return False
            if nonzero and (self._human is not False or self._closed or self._phase in {"stopping", "failed", "cancelled"}
                or self._permit is None or not 0 <= clock-self._permit <= self.limits.command_timeout_sec):
                self.recording_failed("coordinated_authority_changed_before_dispatch")
                raise ValueError(self._reason)

    def record_dispatch(self, *, native_receipt, **values):
        with self._lock:
            if self._pending is None:
                raise ValueError("workspace has no pending dispatch")
            sequence, stamp, _, _ = self._pending
            completed = finite(values.get("completed_at_sec"), "dispatch completion")
            reason = calibration_receipt_rejection(native_receipt, sequence, None if self._receipt is None else self._receipt["sequence"])
            if (reason or native_receipt.get("completed_at_sec") != completed
                or native_receipt.get("sequence") != values.get("receipt_sequence")
                or values.get("feedback_timestamp_sec") != stamp
                or not stamp <= finite(native_receipt.get("requested_at_sec"), "dispatch request") <= completed <= stamp+self.pose.motion.max_control_gap_sec
                or self._receipt is not None and native_receipt["requested_at_sec"] < self._receipt["completed_at_sec"]):
                self.recording_failed(reason or "workspace_dispatch_clock_or_identity")
                raise ValueError(self._reason)
            commands = values["joint_commands"]
            overlay = six(commands["overlay_nm"], "requested overlay")
            if any(abs(overlay[i]) > 1e-9 for i in (0, 3, 4, 5)) or sum(abs(v) for v in overlay) > self.workspace.coordinated_holding.total_assist_cap_nm + 1e-9:
                self.recording_failed("coordinated_requested_command_envelope")
                raise ValueError(self._reason)
            rates = six(values.get("torque_rate_nm_s"), "torque rates")
            if min(rates) <= 0:
                self.recording_failed("invalid_dispatch_command")
                raise ValueError(self._reason)
            self.validate_final_command(gravity_nm=commands["gravity_nm"], total_nm=commands["total_nm"])
            delta = tuple(t-g for t, g in zip(six(commands["total_nm"], "total"), six(commands["gravity_nm"], "gravity")))
            kd = six(commands["driver_kd_nm_s_per_rad"], "driver damping")
            if kd != self.active_driver_damping(self._condition.driver_kd_nm_s_per_rad, final_command_delta_nm=delta):
                self.recording_failed("coordinated_damping_receipt_mismatch")
                raise ValueError(self._reason)
            previous, previous_receipt = self._last_delta, self._receipt
            if previous_receipt is not None and self._started is not None:
                exposure = sum(abs(v) for v in previous) * (completed-max(previous_receipt["completed_at_sec"], self._started))
                self._exposure += exposure
                self._leg_exposure += exposure
                if self._phase == "sweep":
                    self._sweep_exposure += exposure
            self.pose.record_dispatch(sequence, delta)
            self._receipt, self._pending, self._last_delta, self._last_kd = dict(native_receipt), None, delta, kd
            self._last_commands = {"gravity_nm": six(commands["gravity_nm"], "gravity"),
                "overlay_nm": overlay, "total_nm": six(commands["total_nm"], "total"),
                "driver_kd_nm_s_per_rad": kd}
            self._receipt_feedback_stamp = stamp
            self._zero_base = (max(abs(v) for v in delta) <= 1e-9 and six(commands["overlay_nm"], "overlay") == ZERO
                               and kd == self._condition.driver_kd_nm_s_per_rad)
            if self._supported_origin is None and self._phase == "idle" and self._zero_base and self.pose.quiet():
                self._supported_origin = self.pose.q
            self._advance(stamp, completed, previous, previous_receipt, delta, commands)

    def _advance(self, stamp, completed, previous, previous_receipt, delta, commands):
        if not self._active:
            return
        if self._phase == "stopping":
            if self._zero_base and self.pose.quiet():
                if self._quiet_since is None:
                    self._quiet_since = stamp
                if stamp-self._quiet_since >= self.settings.staged_assist.settle_dwell_sec:
                    if (self._return_and_disable and self._return_revoked is None and self._return_context is None
                        and self._reason in RETURNABLE_STOPS):
                        self._begin_return()
                    else:
                        window = self._supported_origin_window()
                        self._return_completed = self._return_route_complete and window is not None and window["within_window"]
                        if self._return_route_complete:
                            self._return_completion_observation = {
                                "qpos_rad": self.pose.q, "origin_window": window,
                                "zero_assist_base_damping_confirmed": self._zero_base,
                                "six_axis_stationary": self.pose.quiet(),
                                "automatic_origin_confirmed": self._return_completed,
                                "completed_at_sec": completed}
                            self._records.append({"kind": "workspace_return_evaluation",
                                "recorded_monotonic_sec": completed, **copy.deepcopy(self._return_completion_observation)})
                        self._transition(self._destination, self._reason)
            else:
                self._quiet_since = None
            if self._phase == "stopping" and stamp-self._withdraw_started > self.pose.motion.braking_timeout_sec:
                self.recording_failed("workspace_withdrawal_not_confirmed")
            return
        if self._phase in {"idle", "preparing"}:
            return
        if self.pose.braking:
            if self._observation is not None:
                j = int(self._tasks[self._index]["joint"][-1])-1
                observation = {**self._observation, "outcome": "censored_native_brake",
                    "completed_at_sec": completed, "final_qpos_rad": self.pose.q,
                    "actual_delta_nm": previous,
                    "net_displacement_rad": self.pose.q[j]-self._observation["anchor_qpos_rad"][j]}
                self._records.append({"kind": "coordinated_holding_observation", "recorded_monotonic_sec": completed,
                    **observation})
                selected = self.pose.frames[AXES.index(j)]
                if (selected.reason in {"resolved_tracking_speed_braking", "soft_velocity_braking"}
                    and abs(observation["net_displacement_rad"]) >= self.workspace.holding_search.movement_rad/2):
                    # This response preceded the new braking command. Retain
                    # it as censored progress, never as a resolved force bound.
                    self._search.record_transition(command_nm=previous[j],
                        requested_command_nm=observation["requested_command_nm"], evidence=observation)
            self._probe = self._observation = None
            self._quiet_since = None
            if self._search is not None and self._search.state == "incomplete":
                self._stop("holding_search_incomplete", "failed")
            return
        task = self._tasks[self._index]
        if self._phase in {"position", "return", "observe", "return_observe"}:
            pose_observation = self._phase in {"observe", "return_observe"}
            # Static acquisition measures the entire six-axis position window.
            # Do not restart it for a short-window velocity/count fluctuation
            # inside that band. Native braking, fresh complete receipts and all
            # owner guards still precede this branch. Transit handoffs and fixed
            # command bisection keep their original quiet gates.
            if not self._at_goal(self.pose.arrival_tolerance_rad,
                                 require_quiet=not pose_observation):
                self._quiet_since = self._pose_window = None
                return
            if self._quiet_since is None:
                self._quiet_since = stamp
                self._pose_window = {"started_at_sec": stamp, "anchor_qpos_rad": self.pose.q,
                    "minimum_rad": self.pose.q, "maximum_rad": self.pose.q}
            window = self._pose_window
            window["minimum_rad"] = tuple(min(a, b) for a, b in zip(window["minimum_rad"], self.pose.q))
            window["maximum_rad"] = tuple(max(a, b) for a, b in zip(window["maximum_rad"], self.pose.q))
            if max(b-a for a, b in zip(window["minimum_rad"], window["maximum_rad"])) > self.workspace.holding_search.movement_rad:
                # A rejected dwell never contributes to a completed static
                # window. Waiting for a new dwell retains the original budget.
                self._records.append({"kind": "workspace_pose_observation_rejected",
                    "recorded_monotonic_sec": completed, "reason": "pose_changed_during_dwell",
                    "task": copy.deepcopy(task), **window, "final_qpos_rad": self.pose.q})
                self._quiet_since = self._pose_window = None
                return
            dwell = self.workspace.gravity_observation_sec if pose_observation else self.settings.staged_assist.settle_dwell_sec
            if stamp-self._quiet_since < dwell:
                return
            # Reference speed/damping changes at the next task still require
            # the original fresh stillness check. Wait without discarding the
            # valid long window when its last frame has count jitter.
            if pose_observation and not self.pose.quiet():
                return
            if self._phase in {"position", "return"} and abs(task["target"]-self.pose.q[int(task["joint"][-1])-1]) > self.pose.arrival_tolerance_rad:
                self._begin_task()
                return
            if pose_observation:
                observation = {"task": copy.deepcopy(task), "qpos_rad": self.pose.q, **window,
                    "completion_criterion": "bounded_six_axis_position_window",
                    "handoff_quiet_confirmed": True,
                    "allowed_position_span_rad": self.workspace.holding_search.movement_rad,
                    "controlled_target_tolerance_rad": self.pose.arrival_tolerance_rad,
                    "nominal_target_rad": self._nominal_target,
                    "command_delta_nm": delta, "gravity_command_nm": commands["gravity_nm"],
                    "driver_kd_nm_s_per_rad": self._last_kd, "completed_at_sec": completed,
                    "quantity": "coupled_command_hold_not_gravity_only_or_measured_force"}
                self._pose_observations.append(observation)
                self._records.append({"kind": "workspace_pose_observation", "recorded_monotonic_sec": completed, **observation})
            self._index += 1
            self._segments = 0
            self._begin_task()
            return
        if self._continuous_identification:
            # Transport retains the other axis' support and the same final
            # envelope. Endpoint dwell is only a handoff, not force bisection.
            if not self._at_goal(self.pose.arrival_tolerance_rad):
                self._quiet_since = None
                return
            if self._quiet_since is None:
                self._quiet_since = stamp
            if stamp-self._quiet_since >= self.settings.staged_assist.settle_dwell_sec:
                j = int(task["joint"][-1])-1
                self._complete_leg(completed, {"sampling_mode": "continuous_identification",
                    "target_position_rad": self._nominal_target[j],
                    "measured_position_rad": self.pose.q[j],
                    "target_error_rad": self._nominal_target[j]-self.pose.q[j],
                    "tracking_target_reached": True})
        else:
            self._observe_probe(stamp, completed, previous, previous_receipt, delta, commands)

    def _observe_probe(self, stamp, completed, previous, previous_receipt, delta, commands):
        cfg = self.workspace.holding_search
        j = int(self._tasks[self._index]["joint"][-1])-1
        if self._probe is None:
            if self._passive_sample_changed():
                # These axes cannot be repositioned by this controller. Do not
                # repeatedly probe or reanchor an already frozen sample point.
                self._stop("holding_search_incomplete", "failed")
                return
            kd_target = self._drive_kd_target()
            # Nominal placement is not the measured point's stability band.
            # Freeze the actual pose below; keep its fine dwell and subsequent
            # full-pose/context checks independent of this entry tolerance.
            tolerance = self.pose.arrival_tolerance_rad
            if self._sample_pose is not None:
                # Return near the frozen point, not merely inside its outer
                # transit band. Otherwise the ramp consumes the observation's
                # displacement room before a causal plateau can even start.
                resolution = self.pose.motion.position_resolution_rad
                tolerance = min(tolerance, 2*resolution if resolution is not None else cfg.movement_rad/2)
            if not self._at_goal(tolerance) or self._last_kd != kd_target:
                self._quiet_since = None
                return
            if self._quiet_since is None:
                self._quiet_since = stamp
            if stamp-self._quiet_since < self.settings.staged_assist.settle_dwell_sec:
                return
            if self._attempts >= cfg.max_trials:
                self._stop("holding_search_incomplete", "failed")
                return
            if self._search is None:
                self._search = HoldingForceSearch(cfg, seed_nm=delta[j],
                    cap_nm=self.workspace.coordinated_holding.total_assist_cap_nm,
                    recheck_interior_on_conflict=True)
                # Freeze the complete measured pose and hold its controlled
                # axes, not alternating edges of the nominal arrival band.
                # Keep the nominal plan and all passive targets separately.
                self._sample_pose = self.pose.q
                hold_goal = list(self.pose.goal)
                for i in AXES:
                    hold_goal[i] = self._sample_pose[i]
                self.pose.set_goal(hold_goal,
                    max_reference_velocity_rad_s=self.pose.trackers[0].config.max_reference_velocity_rad_s,
                    support_soft_speed_recovery=self.pose.trackers[0].config.support_soft_speed_recovery)
            self._attempts += 1
            self._probe = self._search.candidate_nm
            self._observation = {"requested_command_nm": self._probe, "ramp_frozen": False,
                "reacquired_qpos_rad": self.pose.q, "anchor_qpos_rad": self.pose.q,
                "minimum_rad": self.pose.q, "maximum_rad": self.pose.q, "started_at_sec": None}
            return
        obs = self._observation
        q = self.pose.q
        support_changed = max(abs(q[i]-self._sample_pose[i]) for i in range(6) if i != j) > cfg.movement_rad
        if support_changed:
            self._records.append({"kind": "coordinated_holding_observation", "recorded_monotonic_sec": completed,
                **obs, "outcome": "censored_support_pose_changed", "final_qpos_rad": q, "actual_delta_nm": previous})
            self._probe = self._observation = None
            self._quiet_since = None
            if self._passive_sample_changed():
                self._stop("holding_search_incomplete", "failed")
            return
        plateau = previous_receipt is not None and previous_receipt["completed_at_sec"] < stamp and abs(previous[j]-self._probe) <= 1e-9 and abs(delta[j]-self._probe) <= 1e-9
        if not plateau:
            obs["started_at_sec"] = None
            if not obs["ramp_frozen"] and abs(q[j]-obs["anchor_qpos_rad"][j]) >= cfg.movement_rad and (self._probe-previous[j])*(q[j]-obs["anchor_qpos_rad"][j]) > 0:
                self._probe, obs["ramp_frozen"] = previous[j], True
            return
        outcome = None
        if obs["started_at_sec"] is None:
            if max(abs(a-b) for a, b in zip(q, self._sample_pose)) > cfg.movement_rad:
                outcome = "censored_plateau_outside_point"
            else:
                obs.update(started_at_sec=stamp, first_causal_receipt_at_sec=previous_receipt["completed_at_sec"],
                    anchor_qpos_rad=q, minimum_rad=q, maximum_rad=q,
                    stationary_since_sec=stamp, stationary_minimum_rad=q[j], stationary_maximum_rad=q[j],
                    minimum_delta_nm=previous, maximum_delta_nm=previous)
                return
        if outcome is None:
            obs["minimum_rad"] = tuple(min(a,b) for a,b in zip(obs["minimum_rad"],q))
            obs["maximum_rad"] = tuple(max(a,b) for a,b in zip(obs["maximum_rad"],q))
            obs["minimum_delta_nm"] = tuple(min(a,b) for a,b in zip(obs["minimum_delta_nm"],previous))
            obs["maximum_delta_nm"] = tuple(max(a,b) for a,b in zip(obs["maximum_delta_nm"],previous))
            elapsed = stamp-obs["started_at_sec"]
            displacement = q[j]-obs["anchor_qpos_rad"][j]
            # A tolerated pose error is not proof of a constant-command hold.
            # Use the selected axis' fine position band for its dwell, rather
            # than repeatedly clearing it for any axis' short-window velocity
            # fluctuation. Whole-pose and support-command bounds below remain
            # independent; completion still requires fresh six-axis stillness.
            obs["net_displacement_rad"] = displacement
            obs["stationary_minimum_rad"] = min(obs["stationary_minimum_rad"], q[j])
            obs["stationary_maximum_rad"] = max(obs["stationary_maximum_rad"], q[j])
            span = obs["stationary_maximum_rad"]-obs["stationary_minimum_rad"]
            resolution = self.pose.motion.position_resolution_rad
            if (span > 2*resolution + 1e-9 if resolution is not None else not self.pose.quiet()):
                obs.update(stationary_since_sec=stamp, stationary_minimum_rad=q[j], stationary_maximum_rad=q[j])
            obs["stationary_dwell_sec"] = stamp-obs["stationary_since_sec"]
            if elapsed >= self.pose.motion.velocity_window_sec and abs(displacement) >= cfg.movement_rad and displacement*self.pose.velocity[j] > 0 and displacement*self.pose.raw[j] > 0:
                outcome = "positive_motion" if displacement > 0 else "negative_motion"
            elif elapsed >= cfg.observation_sec and self.pose.quiet():
                if abs(q[j]-self._sample_pose[j]) > cfg.movement_rad:
                    outcome = "censored_settled_outside_point"
                elif (obs["stationary_dwell_sec"] >= cfg.observation_sec
                      and max(b-a for a,b in zip(obs["minimum_rad"],obs["maximum_rad"])) <= cfg.movement_rad
                      and max(obs["maximum_delta_nm"][i]-obs["minimum_delta_nm"][i] for i in range(6) if i != j) <= cfg.torque_resolution_nm):
                    outcome = "stationary"
            if outcome is None and elapsed > cfg.observation_timeout_sec:
                outcome = "censored_context_not_stationary"
        if outcome is None:
            return
        if outcome == "stationary":
            obs.update(completion_criterion=("fixed_command_fine_position_dwell" if resolution is not None
                                            else "fixed_command_quiet_dwell"),
                allowed_stationary_span_rad=None if resolution is None else 2*resolution,
                handoff_quiet_confirmed=True)
        obs.update(outcome=outcome, completed_at_sec=completed, final_qpos_rad=q,
            net_displacement_rad=q[j]-obs["anchor_qpos_rad"][j],
            actual_delta_nm=previous, gravity_command_nm=commands["gravity_nm"],
            driver_kd_nm_s_per_rad=self._last_kd, nominal_target_rad=self._nominal_target)
        self._records.append({"kind": "coordinated_holding_observation", "recorded_monotonic_sec": completed, **obs})
        if outcome in {"stationary", "positive_motion", "negative_motion"}:
            self._search.record(command_nm=self._probe, requested_command_nm=obs["requested_command_nm"], outcome=outcome, evidence=dict(obs))
        elif outcome in {"censored_plateau_outside_point", "censored_settled_outside_point"}:
            self._search.record_transition(command_nm=self._probe, requested_command_nm=obs["requested_command_nm"], evidence=dict(obs))
        self._probe = self._observation = None
        self._quiet_since = None
        if self._search.state == "incomplete":
            self._stop("holding_search_incomplete", "failed")
        elif self._search.state == "complete":
            self.pose.adopt_hold(j, self._search.status()["usable_hold_nm"])
            self._complete_leg(completed, {"target_position_rad": self._nominal_target[j], "measured_position_rad": q[j],
                "target_error_rad": self._nominal_target[j]-q[j],
                "tracking_target_reached": abs(self._nominal_target[j]-q[j]) <= self.settings.staged_assist.settle_error_rad,
                "holding_calibration": {**self._search.status(detailed=True), "sample_pose_rad": self._sample_pose,
                    "nominal_target_rad": self._nominal_target, "support_bias_nm": self.pose.bias,
                    "observation_mode": "coordinated_pose_hold", "withdrawal_confirmed": False}})

    def _complete_leg(self, completed, evidence):
        step = self._steps[self._leg_index]
        self._legs.append({"trial_id": step["trial_id"], "role": step["role"],
            "max_reference_velocity_rad_s": step["max_reference_velocity_rad_s"],
            "completed_at_sec": completed, "final_zero_and_still_confirmed": False,
            "handoff_full_dispatch_confirmed": True, "withdrawal_confirmed": False,
            "command_exposure_nm_sec": self._leg_exposure, **evidence})
        self._leg_index += 1
        if self._leg_index < len(self._steps):
            self._begin_leg()
        else:
            result = {"state": "complete", "reason": ("continuous_sweeps_and_handoffs_confirmed"
                if self._continuous_identification else "holding_points_and_handoffs_confirmed"),
                "completed_legs": copy.deepcopy(self._legs)}
            self._acquisitions.append({"task": copy.deepcopy(self._tasks[self._index]), "result": result, "final_qpos_rad": self.pose.q})
            self._records.append({"kind": "workspace_calibration_stage_result", "recorded_monotonic_sec": completed, **self._acquisitions[-1]})
            self._index += 1
            self._begin_task()

    def _holding_status(self, detailed=False):
        return {"enabled": not self._continuous_identification, "stage": "fixed_command_observation" if self._probe is not None else "reacquire",
            "search": None if self._search is None else self._search.status(detailed=detailed),
            "probe_command_nm": self._probe, "sample_pose_rad": self._sample_pose,
            "nominal_target_rad": self._nominal_target, "holding_reference_rad": self.pose.goal,
            "attempt_count": self._attempts,
            "context_reason": ("inactive" if self._phase != "sweep" and self._reason != "holding_search_incomplete" else
                               "awaiting_sample_pose" if self._sample_pose is None else
                               "support_pose_changed" if self._passive_sample_changed() else "ready"),
            "max_attempts": self.workspace.holding_search.max_trials,
            "reacquire_bias_nm": self.pose.bias, "quantity": "software_command_not_measured_force", "auto_apply": False}

    def _static_observation_status(self):
        active = self._phase in {"observe", "return_observe"}
        window = self._pose_window if active and self._quiet_since is not None else None
        return {"active": active, "completion_criterion": "bounded_six_axis_position_window",
            "required_dwell_sec": self.workspace.gravity_observation_sec,
            "elapsed_sec": 0.0 if window is None else max(0.0, self.pose.last_time-self._quiet_since),
            "allowed_position_span_rad": self.workspace.holding_search.movement_rad,
            "controlled_target_tolerance_rad": self.pose.arrival_tolerance_rad,
            "position_span_rad": None if window is None else tuple(b-a for a,b in zip(window["minimum_rad"], window["maximum_rad"])),
            "reason": ("inactive" if not active else "native_braking" if self.pose.braking else
                       "controlled_target_error" if not self._at_goal(self.pose.arrival_tolerance_rad, require_quiet=False)
                       else "waiting_for_window" if window is None else
                       "awaiting_handoff_stillness" if self.pose.last_time-self._quiet_since >= self.workspace.gravity_observation_sec and not self.pose.quiet()
                       else "collecting_window")}

    def _workspace_status(self, detailed=False):
        acquisitions = copy.deepcopy(self._acquisitions) if detailed else []
        if detailed and self._return_context is not None and self._return_context["partial_acquisition"] is not None:
            acquisitions.append(copy.deepcopy(self._return_context["partial_acquisition"]))
        if detailed and self._index < len(self._tasks) and self._tasks[self._index]["kind"] == "sweep":
            acquisitions.append({"task": copy.deepcopy(self._tasks[self._index]), "partial_stage": True,
                "result": {"state": self._phase, "reason": self._reason, "completed_legs": copy.deepcopy(self._legs),
                    "holding_calibration": self._holding_status(detailed=True)}})
        return {"phase": self._phase, "reason": self._reason, "coordinated_holding_enabled": True,
            "holding_sampling_plan": self.workspace.holding_sampling_plan,
            "recovery_damping_owner": ("motor_local_profile" if self.workspace.coordinated_holding.motor_local_recovery
                else "software_window"),
            "supported_soft_stop": (self._active and self._phase != "stopping"
                and bool(self._audit.get("coordinated_holding", {}).get("supported_soft_stop"))),
            "setup_label": self.workspace.setup_label, "settings": asdict(self.workspace) if detailed else None,
            "task_index": self._index, "task_count": len(self._tasks), "pose_count": len(self.workspace.pose_centres()),
            "current_task": copy.deepcopy(self._tasks[self._index]) if self._index < len(self._tasks) else None,
            "completed_joint_acquisitions": len(self._acquisitions), "expected_joint_acquisitions": 2*len(self.workspace.pose_centres()),
            "origin_qpos_rad": self._origin, "current_qpos_rad": self.pose.q,
            "supported_return": {"enabled": self._return_and_disable,
                "supported_origin_qpos_rad": self._supported_origin,
                "route_completed": self._return_route_complete,
                "origin_window": self._supported_origin_window(),
                "completion_observation": copy.deepcopy(self._return_completion_observation) if detailed else None,
                "manual_disable_confirmation_available": self.cycle_disable_confirmation_ready,
                "active": self._return_context is not None and self._active,
                "completed": self._return_completed,
                "revoked_reason": self._return_revoked,
                "acquisition_stop": None if self._return_context is None else {
                    k: v for k, v in self._return_context.items() if k != "partial_acquisition"},
                "automatic_disable_ready": self.automatic_disable_ready,
                "manual_available": self._manual_return_gate() == "ready",
                "manual_gate_reason": self._manual_return_gate(),
                "manual_requested": self._manual_return_requested is not None,
                "max_cycles": 1},
            "dispatch_withdrawal": copy.deepcopy(self._dispatch_withdrawal),
            "holding_search_enabled": not self._continuous_identification, "holding_calibration": self._holding_status(detailed),
            "continuous_identification": {"enabled": self._continuous_identification,
                "command_field_mode": self.workspace.coordinated_holding.command_field_mode,
                "leg_index": self._leg_index, "leg_count": len(self._steps),
                "speeds_rad_s": self.workspace.sweep_speeds(
                    self._tasks[self._index].get("joint") if self._index < len(self._tasks) else None, self.config),
                "fits_after_cleanup_only": True, "auto_apply": False},
            **({"command_field_source": self.pose.command_field_source.metadata(),
                "command_field_mode": self.workspace.coordinated_holding.command_field_mode}
               if self.pose.command_field_source is not None and detailed else {}),
            "arrival": self._arrival_status(),
            "static_observation": self._static_observation_status(),
            "maximum_start_angle_deg": self.workspace.maximum_start_angle_deg,
            "acquisition_ranges_deg": {"joint2": self.workspace.joint2_acquisition_range_deg, "joint3": self.workspace.joint3_acquisition_range_deg},
            "pose_centres_rad": self.workspace.pose_centres(), "local_excursion_rad": self.workspace.local_excursion_rad(self.config),
            "holding_bias_nm": self.pose.bias, "last_confirmed_delta_nm": self._last_delta,
            "combined_l1_cap_nm": self.workspace.coordinated_holding.total_assist_cap_nm,
            "command_exposure_nm_sec": self._exposure, "max_command_exposure_nm_sec": self.workspace.max_command_exposure_nm_sec,
            "leg_command_exposure_nm_sec": self._leg_exposure,
            "max_sweep_leg_command_exposure_nm_sec": self.config.max_sweep_leg_command_exposure_nm_sec,
            "max_sweep_leg_sec": self.config.max_sweep_leg_sec,
            "acquisitions": acquisitions, "moves": [], "gravity_observations": [],
            "coordinated_pose_observations": copy.deepcopy(self._pose_observations) if detailed else [],
            "gravity_parameters_frozen": True, "auto_apply": False}

    def _batch_status(self, detailed=False):
        return {"operation_mode": "workspace", "state": self._phase, "reason": self._reason,
            "active": self._active, "start_requested": self._start or self._manual_return_requested is not None, "stage": "motion", "human_intervention": self._human,
            "outcome": calibration_outcome(self._phase, self._reason), "onset_trials": [], "onset_index": 0, "expected_onset_trials": 0,
            "workspace": self._workspace_status(detailed), "cycle": {"phase": self._phase,
                "background_permit_timeout_sec": self.limits.command_timeout_sec,
                "zero_assist_base_damping_confirmed": self._zero_base, "disable_confirmation_required": True,
                "disable_confirmation_available": self.cycle_disable_confirmation_ready}}

    def _make_audit(self, command, pose_audit, fault=None):
        task = self._tasks[self._index] if self._index < len(self._tasks) else {}
        joint = task.get("joint", "joint2")
        j = int(joint[-1])-1
        motion_fault = self.pose.fault_frame
        if fault == "joint_velocity_exceeded":
            # The task joint is not necessarily the joint that tripped.
            failed = max(range(6), key=lambda i: abs(self.pose.raw[i]))
            motion_fault = {"joint_name": f"joint{failed+1}", "reason": fault,
                "feedback_timestamp_sec": self.pose.last_time,
                "raw_qvel_rad_s": self.pose.raw[failed],
                "window_qvel_rad_s": self.pose.velocity[failed],
                "limit_rad_s": self.limits.max_joint_velocity_rad_s}
        moving = self._phase in {"position", "return", "sweep", "observe", "return_observe"}
        marker = None if self._batch_id is None else {
            "batch_id": self._batch_id,
            "trial_id": (self._steps[self._leg_index]["trial_id"] if self._phase == "sweep" and self._leg_index < len(self._steps)
                         else f"cw{self._index}:{self._phase}"),
            "role": "holding_not_for_running_fit", "state": "observing" if self._probe is not None else self._phase}
        if marker is not None and self._continuous_identification:
            step = self._steps[self._leg_index] if self._phase == "sweep" and self._leg_index < len(self._steps) else None
            marker.update(role=step["role"] if step else "return_not_for_fit" if self._phase.startswith("return") else "positioning_not_for_fit",
                state="moving" if step and not self.pose.braking else self._phase,
                reference_speed_rad_s=None if step is None else step["max_reference_velocity_rad_s"])
        return {"time_sec": self.pose.last_time, "controller_mode": self.controller_mode, "joint_name": joint,
            "owner_ready": self._ready, "operator_armed": moving, "phase": "fault" if fault else "track" if moving else "inactive",
            "reason": fault or self._reason, "fault": fault, "qpos_rad": self.pose.q, "qvel_fd_rad_s": self.pose.raw,
            "control_qvel_rad_s": self.pose.velocity, "desired_assist_nm": command[j], "effective_assist_nm": command[j],
            "overlay_torque_nm": command, "coordinated_holding": pose_audit,
            "motion_fault": motion_fault,
            "holding_calibration": self._holding_status(), "automatic_calibration": self._batch_status(),
            "calibration_trial": marker}

    def status(self):
        with self._lock:
            fault = self._reason if self._phase == "failed" and calibration_outcome(self._phase, self._reason)["severity"] == "error" else None
            requested = self._pending[3] if self._pending is not None else self._last_delta
            view = self._make_audit(requested, self._audit.get("coordinated_holding", {}), fault=fault)
            j = int(view["joint_name"][-1])-1
            sent = None if self._receipt is None else {
                "joint_name": view["joint_name"], "receipt_sequence": self._receipt["sequence"],
                "recorded_at_sec": self._receipt["completed_at_sec"],
                "overlay_after_profile_nm": self._last_commands["overlay_nm"][j],
                "gravity_nm": self._last_commands["gravity_nm"][j],
                "conditioned_total_nm": self._last_commands["total_nm"][j],
                "joint_commands": copy.deepcopy(self._last_commands)}
            return {**view, "limits": asdict(self.limits), "history": [], "events": [],
                "effective_assist_nm": self._last_delta[j],
                "last_dispatch": sent, "motion_control_enabled": True,
                "motion_control": asdict(self.pose.frames[AXES.index(j)]) if self.pose.frames else None,
                "feedback_age_sec": None if self.pose.last_time is None else max(0, time.monotonic()-self.pose.last_time),
                "dispatch_age_sec": None if sent is None else max(0, time.monotonic()-sent["recorded_at_sec"])}

    def result_record(self):
        with self._lock:
            batch = self._batch_status(True)
            return {"schema": "b601.automatic-joint-calibration.v1", "kind": "automatic_calibration_result",
                "recorded_monotonic_sec": None if self._receipt is None else self._receipt["completed_at_sec"],
                "batch_id": self._batch_id, "state": self._phase, "reason": self._reason,
                "condition": None if self._condition is None else asdict(self._condition),
                "settings": asdict(self.config), "onset_trials": [], "incomplete_onset_trial": None,
                "onset_damping": {"stage": "not_requested_gravity_first"},
                "last_complete_dispatch": copy.deepcopy(self._receipt), "pending_dispatch": None if self._pending is None else {
                    "feedback_sequence": self._pending[0], "feedback_timestamp_sec": self._pending[1], "receipt_confirmed": False},
                "motion": {"completed_legs": [{**leg, "joint_name": item["task"]["joint"], "pose_index": item["task"]["pose"]}
                    for item in batch["workspace"]["acquisitions"] for leg in item["result"]["completed_legs"]]}, **batch}

    def drain_records(self):
        with self._lock:
            records, self._records = self._records, []
            return records

    def recording_failed(self, reason):
        with self._lock:
            self._start = False
            self._manual_return_requested = None
            self._return_revoked = str(reason)
            self._transition("failed", str(reason))

    def owner_stopped(self, *, now_sec, reason):
        with self._lock:
            self._closed, self._ready, self._start = True, False, False
            self._manual_return_requested = None
            if self._active:
                self.recording_failed(f"owner_stopped:{reason}")

    @property
    def cycle_disable_confirmation_required(self):
        return True

    @property
    def cycle_disable_confirmation_ready(self):
        origin = self._supported_origin if self._return_context is not None else self._origin
        finished = (self._phase == "complete" or self._return_context is not None
            and self._return_route_complete and not self._active and self._return_revoked is None)
        return (finished and self._pending is None and self._zero_base
            and origin is not None and self.pose.quiet()
            and max(abs(self.pose.q[i]-origin[i]) for i in AXES) <= self.workspace.holding_search.movement_rad)

    @property
    def automatic_disable_ready(self):
        return (self._return_and_disable and self._return_completed and not self._active
            and self._return_revoked is None
            and self._phase in {"complete", "failed"} and self._human is False and not self._closed
            and self._pending is None and self._zero_base and self.pose.quiet()
            and self._permit is not None and self.pose.last_time is not None
            and 0 <= self.pose.last_time-self._permit <= self.limits.command_timeout_sec
            and self._supported_origin is not None and all(abs(q-o) <= self.workspace.holding_search.movement_rad
                for q, o in zip(self.pose.q, self._supported_origin))
            and (self._phase == "complete" or self._return_context is not None
                 and self._reason == self._return_context["reason"] and self._reason in RETURNABLE_STOPS))

    def _supported_origin_window(self):
        """Describe the existing six-axis gate without widening or bypassing it."""
        if self._supported_origin is None:
            return None
        limit = self.workspace.holding_search.movement_rad
        blocking = [{"joint_name": f"joint{i+1}", "offset_rad": q-o, "limit_rad": limit}
            for i, (q, o) in enumerate(zip(self.pose.q, self._supported_origin)) if abs(q-o) > limit]
        return {"tolerance_rad": limit, "within_window": not blocking, "blocking_joints": blocking,
            "semantics": "encoder_window_not_physical_support_detection"}
