"""Finite J2/J3 acquisition intents; the existing owner alone sends commands."""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import asdict, dataclass, replace

from robot_motion.rebot_b601_rs.calibration import finite, plan_local_sweeps, six
from robot_motion.rebot_b601_rs.coordinated_holding import CoordinatedHoldingConfig
from robot_motion.rebot_b601_rs.holding_calibration import HoldingSearchConfig
from robot_motion.rebot_b601_rs.model import B601_RS_ARM_JOINT_NAMES

from .calibration_session import (
    B601AutomaticCalibrationSession,
    CalibrationBatchLimits,
    calibration_outcome,
    calibration_receipt_rejection,
)
from .staged_controller import B601StagedTorqueConsoleController

ZERO = (0.0,) * 6
RANGES = ((0.0, math.radians(120)), (0.0, math.radians(90)))
COORDINATED_CALIBRATION_OVERRIDES = (
    "prepare_timeout_sec", "max_batch_sec", "max_sweep_leg_sec",
    "max_sweep_command_exposure_nm_sec", "max_sweep_leg_command_exposure_nm_sec",
    "motion_braking_timeout_sec", "motion_soft_velocity_rad_s",
)


@dataclass(frozen=True)
class WorkspaceCalibrationConfig:
    joint2_centres_deg: tuple[float, ...] = (3.0, 6.0)
    joint3_centres_deg: tuple[float, ...] = (3.0, 6.0)
    joint2_acquisition_range_deg: tuple[float, float] = (0.0, 10.0)
    joint3_acquisition_range_deg: tuple[float, float] = (0.0, 10.0)
    maximum_start_angle_deg: float = 2.0
    positioning_joint_order: tuple[str, str] = ("joint3", "joint2")
    joint2_drive_kd: tuple[float, ...] = (0.4, 0.6, 0.4, 0.6, 0.2, 0.1)
    joint3_drive_kd: tuple[float, ...] = (0.4, 0.6, 0.4, 0.6, 0.2, 0.1)
    maximum_positioning_step_rad: float = 0.06
    positioning_velocity_rad_s: float = 0.015
    positioning_torque_rise_nm_s: float = 1.0
    positioning_torque_fall_nm_s: float = 8.0
    positioning_support_soft_speed_recovery: bool = False
    gravity_observation_sec: float = 5.0
    max_workspace_sec: float = 7200.0
    max_command_exposure_nm_sec: float = 2400.0
    background_permit_timeout_sec: float = 70.0
    setup_label: str = "temporary_end_effector"
    holding_search: HoldingSearchConfig | None = None
    holding_excursion_rad: float | None = None
    coordinated_holding: CoordinatedHoldingConfig | None = None
    max_sweep_leg_sec: float | None = None
    max_sweep_leg_command_exposure_nm_sec: float | None = None
    max_sweep_command_exposure_nm_sec: float | None = None
    max_batch_sec: float | None = None
    prepare_timeout_sec: float | None = None
    motion_braking_timeout_sec: float | None = None
    motion_soft_velocity_rad_s: float | None = None

    def __post_init__(self):
        if isinstance(self.coordinated_holding, dict):
            object.__setattr__(self, "coordinated_holding", CoordinatedHoldingConfig(**self.coordinated_holding))
        if self.coordinated_holding is not None and not isinstance(self.coordinated_holding, CoordinatedHoldingConfig):
            raise TypeError("coordinated holding requires an explicit configuration object")
        if isinstance(self.holding_search, dict):
            object.__setattr__(self, "holding_search", HoldingSearchConfig(**self.holding_search))
        if self.holding_search is not None and not isinstance(self.holding_search, HoldingSearchConfig):
            raise TypeError("holding search must be an explicit configuration object")
        if self.coordinated_holding is not None and self.holding_search is None:
            raise ValueError("coordinated holding requires explicit holding search")
        for name in COORDINATED_CALIBRATION_OVERRIDES:
            value = getattr(self, name)
            if value is not None and (self.coordinated_holding is None or finite(value, name) <= 0):
                raise ValueError("workspace overrides require explicit coordinated holding and positive limits")
        if self.holding_excursion_rad is not None and (
            self.holding_search is None or finite(self.holding_excursion_rad, "holding excursion") <= 0
        ):
            raise ValueError("holding excursion requires the explicit holding search")
        if type(self.positioning_support_soft_speed_recovery) is not bool:
            raise TypeError("workspace positioning support recovery must be boolean")
        for name, upper in (("joint2_centres_deg", 120), ("joint3_centres_deg", 90)):
            values = tuple(finite(v, name) for v in getattr(self, name))
            if not 1 <= len(values) <= 12 or list(values) != sorted(set(values)):
                raise ValueError("workspace centres must be sorted, distinct and finite")
            if not all(0 < v < upper for v in values):
                raise ValueError("workspace centres must stay inside J2 0..120 / J3 0..90 degrees")
            object.__setattr__(self, name, values)
        for name, upper in (("joint2_acquisition_range_deg", 120), ("joint3_acquisition_range_deg", 90)):
            values = tuple(finite(v, name) for v in getattr(self, name))
            if len(values) != 2 or not 0 <= values[0] < values[1] <= upper:
                raise ValueError("workspace acquisition ranges must stay inside declared joint ranges")
            object.__setattr__(self, name, values)
        order = tuple(self.positioning_joint_order)
        if len(order) != 2 or set(order) != {"joint2", "joint3"}:
            raise ValueError("positioning order must contain J2 and J3 exactly once")
        object.__setattr__(self, "positioning_joint_order", order)
        if len(self.joint2_centres_deg) * len(self.joint3_centres_deg) > 64:
            raise ValueError("workspace batch permits at most 64 pose centres")
        for name in ("joint2_drive_kd", "joint3_drive_kd"):
            gains = six(getattr(self, name), name)
            if min(gains) < 0:
                raise ValueError("workspace damping must be nonnegative")
            object.__setattr__(self, name, gains)
        for name in (
            "maximum_positioning_step_rad", "positioning_velocity_rad_s",
            "positioning_torque_rise_nm_s", "positioning_torque_fall_nm_s",
            "gravity_observation_sec", "max_workspace_sec",
            "max_command_exposure_nm_sec", "background_permit_timeout_sec",
            "maximum_start_angle_deg",
        ):
            if finite(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.setup_label, str) or not self.setup_label.strip() or len(self.setup_label) > 120:
            raise ValueError("workspace setup label must be 1..120 characters")

    def pose_centres(self):
        return tuple(
            (math.radians(j2), math.radians(j3))
            for row, j2 in enumerate(self.joint2_centres_deg)
            for j3 in (self.joint3_centres_deg if row % 2 == 0 else reversed(self.joint3_centres_deg))
        )

    def local_excursion_rad(self, calibration):
        if self.holding_excursion_rad is None:
            return calibration.sweep_excursion_rad
        if not 2 * self.holding_search.movement_rad < self.holding_excursion_rad <= calibration.sweep_excursion_rad:
            raise ValueError("holding targets must fit the existing local sweep and measurement neighbourhood")
        return self.holding_excursion_rad

    def tasks(self, origin_qpos_rad):
        origin = six(origin_qpos_rad, "workspace origin")
        tasks = []
        for pose, (j2, j3) in enumerate(self.pose_centres()):
            centres = {"joint2": j2, "joint3": j3}
            tasks.extend({"kind": "position", "pose": pose, "joint": joint, "target": centres[joint]}
                for joint in self.positioning_joint_order)
            tasks.extend([
                {"kind": "observe", "pose": pose},
                {"kind": "sweep", "pose": pose, "joint": "joint2"},
                {"kind": "sweep", "pose": pose, "joint": "joint3"},
            ])
        tasks.extend({"kind": "return", "joint": joint,
                      "target": origin[B601_RS_ARM_JOINT_NAMES.index(joint)]}
                     for joint in reversed(self.positioning_joint_order))
        tasks.append({"kind": "return_observe"})
        return tasks


class B601WorkspaceCalibrationController:
    """Position one axis at a time, observe gravity, sweep both axes, return.

    Each child is the existing receipt-driven staged/session implementation.
    A child is replaced only after confirmed zero assist, base damping and
    stationarity. Cancellation never initiates an automatic return.
    """

    controller_mode = "automatic_calibration"

    def __init__(self, settings, config, workspace: WorkspaceCalibrationConfig):
        overrides = {name: getattr(workspace, name) for name in COORDINATED_CALIBRATION_OVERRIDES
                     if getattr(workspace, name) is not None}
        if overrides:
            from .automatic_calibration import B601AutomaticCalibrationController

            config = replace(config, **overrides)
            # Reuse canonical pre-CAN validation and settings resolution;
            # fixed-position/supported-cycle controllers retain their settings.
            settings = B601AutomaticCalibrationController(settings, config).settings
        self.config, self.workspace = config, workspace
        self.limits = replace(settings.limits, command_timeout_sec=max(
            settings.limits.command_timeout_sec, workspace.background_permit_timeout_sec))
        self.settings = replace(settings, limits=self.limits)
        motion = self.settings.staged_assist.motion_control
        if motion is None or not motion.responsive_tracking_brake or config.motion_plan != "local_identification":
            raise ValueError("workspace acquisition requires the explicit local identification profile")
        if workspace.holding_search is not None and (
            motion.continuous_speed_limit_rad_s is None
            or config.motion_acquisition_arrival_tolerance_rad is None
            or workspace.holding_search.movement_rad < 4 * (motion.position_resolution_rad or 0.0)
            or workspace.holding_search.observation_timeout_sec >= config.max_sweep_leg_sec
            or workspace.holding_search.movement_rad >= config.motion_acquisition_arrival_tolerance_rad
        ):
            raise ValueError("holding search must fit resolved continuous motion, point duration and arrival envelope")
        workspace.local_excursion_rad(config)
        tolerance = config.motion_acquisition_arrival_tolerance_rad
        if tolerance is None or not 2 * tolerance < workspace.maximum_positioning_step_rad <= settings.maximum_target_offset_rad:
            raise ValueError("workspace positioning steps must fit the existing local target envelope and tolerance")
        if workspace.positioning_velocity_rad_s > motion.max_reference_velocity_rad_s:
            raise ValueError("workspace positioning speed exceeds the existing motion profile")
        if not workspace.positioning_torque_rise_nm_s <= workspace.positioning_torque_fall_nm_s <= motion.braking_slew_nm_s:
            raise ValueError("workspace normal torque release must remain inside the existing protective braking slew")
        for centres, (lo, hi) in zip(
            (workspace.joint2_centres_deg, workspace.joint3_centres_deg),
            (workspace.joint2_acquisition_range_deg, workspace.joint3_acquisition_range_deg), strict=True,
        ):
            if any(not math.radians(lo) + config.global_excursion_rad < math.radians(q) < math.radians(hi) - config.global_excursion_rad for q in centres):
                raise ValueError("workspace centre leaves insufficient room for local sweeps and withdrawal")
        # Validate every child configuration before the owner connects to CAN,
        # not lazily after the first physical positioning leg has completed.
        for joint in ("joint2", "joint3"):
            for positioning in (True, False):
                self._delegate = self._make_controller(joint, positioning=positioning)
        self._session = None
        self._condition = self._position_limits = None
        self._ready = self._start_requested = self._closed = False
        self._human = None
        self._permit = self._started = self._last_feedback = None
        self._phase_started = self._child_created_at = None
        self._phase, self._reason = "idle", "waiting_for_operator"
        self._origin = self._q = None
        self._batch_id = None
        self._tasks = []
        self._index = self._segments = 0
        self._segment_limit = None
        self._pending = self._receipt = None
        self._last_delta = ZERO
        self._exposure = 0.0
        self._zero_base = False
        self._cancel = self._stop_destination = None
        self._observation = None
        self._observations, self._acquisitions, self._moves = [], [], []
        self._records = []
        self._lock = threading.RLock()
        self._coordinated = None
        if workspace.coordinated_holding is not None:
            from .coordinated_workspace import CoordinatedWorkspaceController
            self._coordinated = CoordinatedWorkspaceController(self.settings, config, workspace)

    @property
    def _active(self):
        return self._phase not in {"idle", "complete", "failed", "cancelled"}

    def _make_controller(self, joint, *, positioning):
        gains = getattr(self.workspace, f"{joint}_drive_kd")
        motion = replace(self.settings.staged_assist.motion_control,
                         drive_driver_kd_nm_s_per_rad=gains,
                         support_soft_speed_recovery=(self.workspace.positioning_support_soft_speed_recovery
                             if positioning else self.settings.staged_assist.motion_control.support_soft_speed_recovery))
        staged = replace(self.settings.staged_assist, joint_name=joint, motion_control=motion)
        if positioning:
            staged = replace(staged,
                tracking_slew_nm_s=self.workspace.positioning_torque_rise_nm_s,
                tracking_release_slew_nm_s=self.workspace.positioning_torque_fall_nm_s)
        settings = replace(self.settings, limits=replace(self.limits, joint_name=joint), staged_assist=staged)
        controller = B601StagedTorqueConsoleController(settings)
        if getattr(self, "_ready", False):
            controller.owner_ready()
        return controller

    def validate_position_limits(self, position_limits):
        if len(position_limits) != 6:
            raise ValueError("workspace requires six owner position intervals")
        for j, bounds in enumerate(RANGES, 1):
            lo, hi = position_limits[j]
            if lo > bounds[0] + 1e-12 or hi < bounds[1] - 1e-12:
                raise ValueError("workspace J2/J3 range exceeds the pinned owner position limits")

    def configure_context(self, condition, position_limits_rad):
        if self._coordinated is not None:
            self.validate_position_limits(position_limits_rad)
            return self._coordinated.configure_context(condition, position_limits_rad)
        with self._lock:
            if self._condition is not None:
                raise ValueError("workspace context is immutable")
            self.validate_position_limits(position_limits_rad)
            limits = [tuple(v) for v in position_limits_rad]
            limits[1:3] = [tuple(math.radians(v) for v in bounds) for bounds in (
                self.workspace.joint2_acquisition_range_deg, self.workspace.joint3_acquisition_range_deg)]
            self._condition, self._position_limits = condition, tuple(limits)

    def owner_ready(self):
        if self._coordinated is not None:
            return self._coordinated.owner_ready()
        with self._lock:
            self._ready = True
            self._delegate.owner_ready()

    def set_human_intervention(self, human):
        if self._coordinated is not None:
            return self._coordinated.set_human_intervention(human)
        if type(human) is not bool:
            raise TypeError("human intervention must be boolean")
        with self._lock:
            self._human = human
            if human:
                self._start_requested = False
                if self._active:
                    self._cancel = "human_intervention"

    def submit(self, action, *, now_sec, **fields):
        if self._coordinated is not None:
            return self._coordinated.submit(action, now_sec=now_sec, **fields)
        now = finite(now_sec, "intent time")
        with self._lock:
            allowed = {"withdraw_source"} if action == "withdraw" else {"visible"} if action == "visibility" else set()
            if set(fields) - allowed:
                raise ValueError("workspace accepts no manual targets or torque fields")
            if action == "arm":
                if self._phase != "idle" or self._start_requested or self._closed or not self._ready or self._human is not False or self._condition is None:
                    raise ValueError("workspace start requires idle owner and no human intervention")
                if self._q is None or any(abs(self._q[j]) > math.radians(self.workspace.maximum_start_angle_deg) for j in (1, 2)):
                    raise ValueError(f"workspace start requires J2/J3 within {self.workspace.maximum_start_angle_deg:g} degrees of zero")
                self._start_requested, self._permit = True, now
            elif action in {"heartbeat", "visibility"}:
                if not self._active and not self._start_requested:
                    raise ValueError("workspace permit requires an active or pending batch")
                if action == "visibility" and type(fields.get("visible")) is not bool:
                    raise ValueError("visibility must be boolean")
                self._permit = now
            elif action in {"withdraw", "zero"}:
                source = fields.get("withdraw_source", "operator_button")
                reasons = {"operator_button": "operator_withdraw", "page_unload": "page_unload", "page_hidden": "page_hidden", "owner_cancel": "owner_cancel"}
                if source not in reasons:
                    raise ValueError("unsupported withdrawal source")
                self._start_requested = False
                if self._active:
                    self._cancel = "operator_zero" if action == "zero" else reasons[source]
            else:
                raise ValueError("unsupported workspace action")

    def _build_tasks(self):
        return self.workspace.tasks(self._origin)

    def _transition(self, phase, reason, now):
        if (phase, reason) != (self._phase, self._reason):
            self._phase, self._reason = phase, reason
            self._phase_started = now
            self._records.append({"kind": "workspace_calibration_transition",
                "recorded_monotonic_sec": now, "phase": phase, "reason": reason,
                "task_index": self._index, "batch_id": self._batch_id})

    def _retain_session_result(self, session, now):
        snapshot = session.status()
        terminal = snapshot["state"] in {"complete", "failed", "cancelled"} and not snapshot["pending_dispatch"]
        record = {"task": copy.deepcopy(self._tasks[self._index]),
            "result": session.result_record() if terminal else snapshot, "final_qpos_rad": self._q}
        if not terminal:
            record["partial_stage"] = True
        destination = self._acquisitions if record["task"]["kind"] == "sweep" else self._moves
        destination.append(record)
        self._records.append({"kind": "workspace_calibration_stage_result", "recorded_monotonic_sec": now, **record})
        if terminal:
            self._records.append({**record["result"], "workspace_task": record["task"]})
        self._session = None
        return record

    def _begin_task(self, now):
        while self._index < len(self._tasks):
            task = self._tasks[self._index]
            kind = task["kind"]
            self._transition(kind, "workspace_task_started", now)
            if kind in {"observe", "return_observe"}:
                return
            joint = task["joint"]
            j = B601_RS_ARM_JOINT_NAMES.index(joint)
            if kind in {"position", "return"}:
                distance = task["target"] - self._q[j]
                tolerance = self.config.motion_acquisition_arrival_tolerance_rad
                if abs(distance) <= tolerance:
                    self._index += 1
                    self._segments, self._segment_limit = 0, None
                    continue
                if self._segment_limit is None:
                    self._segment_limit = math.ceil(abs(distance) / (self.workspace.maximum_positioning_step_rad - 2 * tolerance)) + 2
                if self._segments >= self._segment_limit:
                    self.recording_failed("workspace_positioning_segment_budget")
                    return
                self._segments += 1
                target = list(self._q)
                target[j] += math.copysign(min(abs(distance), self.workspace.maximum_positioning_step_rad), distance)
                steps = [{"trial_id": f"w{self._index}-move{self._segments}",
                          "role": "return_not_for_fit" if kind == "return" else "positioning_not_for_fit",
                          "target_qpos_rad": tuple(target),
                          "max_reference_velocity_rad_s": self.workspace.positioning_velocity_rad_s}]
                plan = {"schema": "b601.local-calibration-plan.v1", "execution_enabled": False,
                        "condition": asdict(self._condition), "joint_name": joint,
                        "anchor_qpos_rad": self._q, "steps": steps}
                tolerance = None
            else:
                plan = plan_local_sweeps(condition=self._condition, joint_name=joint,
                    anchor_qpos_rad=self._q, position_limits_rad=self._position_limits,
                    excursion_rad=self.workspace.local_excursion_rad(self.config),
                    speeds_rad_s=self.config.sweep_speeds_rad_s,
                    max_reference_velocity_rad_s=self.settings.staged_assist.motion_control.max_reference_velocity_rad_s,
                    purpose="local_identification")
                for step in plan["steps"]:
                    step["trial_id"] = f"w{self._index}:{joint}:{step['trial_id']}"
                tolerance = (self.workspace.holding_search.movement_rad if self.workspace.holding_search is not None
                    else self.config.motion_acquisition_arrival_tolerance_rad)
            controller = self._make_controller(joint, positioning=kind != "sweep")
            session = B601AutomaticCalibrationSession(controller, plan=plan,
                condition=self._condition, position_limits_rad=self._position_limits,
                limits=CalibrationBatchLimits(self.config.max_batch_sec, self.config.max_sweep_leg_sec,
                    self.config.prepare_timeout_sec, self.limits.command_timeout_sec,
                    self.settings.staged_assist.motion_control.max_control_gap_sec,
                    self.config.max_sweep_command_exposure_nm_sec,
                    self.config.max_sweep_leg_command_exposure_nm_sec),
                acquisition_arrival_tolerance_rad=tolerance,
                holding_search=self.workspace.holding_search if kind == "sweep" else None)
            self._delegate, self._session = controller, session
            self._child_created_at = now
            return
        if (not self._zero_base
            or self._delegate.owner_snapshot()["prepare_gate"] != "ready"
            or any(abs(self._q[j] - self._origin[j]) > self.config.motion_acquisition_arrival_tolerance_rad for j in (1, 2))):
            self.recording_failed("workspace_return_not_confirmed")
        else:
            self._transition("complete", "all_targets_and_withdrawals_confirmed", now)

    def step(self, *, now_sec, qpos_rad, qvel_rad_s, feedback_sequence, control_now_sec):
        if self._coordinated is not None:
            return self._coordinated.step(now_sec=now_sec, qpos_rad=qpos_rad, qvel_rad_s=qvel_rad_s,
                feedback_sequence=feedback_sequence, control_now_sec=control_now_sec)
        now, clock = finite(now_sec, "feedback time"), finite(control_now_sec, "control time")
        q = six(qpos_rad, "workspace position")
        gap = self.settings.staged_assist.motion_control.max_control_gap_sec
        with self._lock:
            if self._closed:
                raise ValueError("workspace owner has stopped")
            if self._pending is not None:
                raise ValueError("workspace previous six-axis dispatch is not confirmed")
            if (not now <= clock <= now + gap
                or (self._active and self._last_feedback is not None
                    and not self._last_feedback < now <= self._last_feedback + gap)):
                self.recording_failed("workspace_feedback_clock_or_gap")
                raise ValueError(self._reason)
            self._last_feedback, self._q = now, q
            if self._start_requested:
                if any(not lo <= value <= hi for value, (lo, hi) in zip(q, self._position_limits, strict=True)):
                    raise ValueError("workspace origin outside the declared position range")
                if any(abs(q[j]) > math.radians(self.workspace.maximum_start_angle_deg) for j in (1, 2)):
                    raise ValueError("workspace origin moved outside the near-zero start window")
                self._start_requested = False
                self._origin, self._started = q, clock
                self._batch_id = f"workspace-{clock:.9f}"
                self._tasks = self._build_tasks()
                self._records.append({**self.result_record(), "kind": "automatic_calibration_started", "recorded_monotonic_sec": clock})
                self._begin_task(clock)
            if self._active:
                if any(not lo <= value <= hi for value, (lo, hi) in zip(q, self._position_limits, strict=True)):
                    self.recording_failed("workspace_position_envelope")
                elif any(abs(q[i] - self._origin[i]) > self.limits.max_support_drift_rad for i in (0, 3, 4, 5)):
                    self.recording_failed("workspace_support_excursion")
                elif self._human is not False:
                    self._cancel = "human_intervention"
                elif self._permit is None or not 0 <= clock - self._permit <= self.limits.command_timeout_sec:
                    self._cancel = "gui_permit_expired"
                elif clock - self._started > self.workspace.max_workspace_sec:
                    self._cancel = "workspace_timeout"
                elif self._exposure + 6 * self.limits.max_abs_assist_nm * gap >= self.workspace.max_command_exposure_nm_sec:
                    self._cancel = "workspace_command_exposure_budget"
                if self._cancel is not None and self._phase != "stopping":
                    self._stop_destination = "failed" if self._cancel in {"workspace_timeout", "workspace_command_exposure_budget"} else "cancelled"
                    if self._session is not None:
                        self._session.cancel(now_sec=clock, reason=self._cancel)
                    else:
                        self._delegate.submit("withdraw", now_sec=clock, withdraw_source="owner_cancel")
                    self._transition("stopping", self._cancel, clock)
            if self._active and self._session is None and self._phase not in {"observe", "return_observe", "stopping"}:
                self._begin_task(clock)
            if (self._session is not None and self._session.owner_snapshot()["state"] == "idle"
                and clock - self._child_created_at > self.config.prepare_timeout_sec):
                self.recording_failed("preparation_timeout")
            if (self._phase == "stopping"
                and clock - self._phase_started > self.settings.staged_assist.motion_control.braking_timeout_sec + gap):
                self.recording_failed("workspace_withdrawal_not_confirmed")
            if self._phase == "failed" and (
                calibration_outcome(self._phase, self._reason)["severity"] == "error"
                or not self._zero_base
            ):
                # A rejected frame must not create a new child pending receipt.
                return ZERO, {**self._delegate.status(), "phase": "fault", "fault": self._reason,
                    "automatic_calibration": self._batch_status(detailed=False)}
            if self._session is None:
                overlay, audit = self._delegate.step(now_sec=now, qpos_rad=q, qvel_rad_s=qvel_rad_s)
            else:
                overlay, audit = self._session.step(now_sec=now, qpos_rad=q,
                    qvel_rad_s=qvel_rad_s, feedback_sequence=feedback_sequence,
                    condition=self._condition, human_intervention=self._human,
                    operator_permit_timestamp_sec=self._permit, operator_observed_at_sec=clock)
            self._pending = (feedback_sequence, now, self._delegate, self._session)
            if audit["phase"] == "fault":
                self.recording_failed(str(audit["fault"]))
            if self._phase == "failed" and (
                calibration_outcome(self._phase, self._reason)["severity"] == "error"
                or not self._zero_base
            ):
                overlay, audit = ZERO, {**audit, "phase": "fault", "fault": self._reason}
            return overlay, {**audit, "automatic_calibration": self._batch_status(detailed=False)}

    def validate_final_command(self, **values):
        if self._coordinated is not None:
            return self._coordinated.validate_final_command(**values)
        # Staged commands remain bounded by their runtime and the owner.
        return None

    def active_driver_damping(self, base_kd, *, final_command_delta_nm=None):
        if self._coordinated is not None:
            return self._coordinated.active_driver_damping(base_kd, final_command_delta_nm=final_command_delta_nm)
        with self._lock:
            return self._delegate.active_driver_damping(base_kd, final_command_delta_nm=final_command_delta_nm)

    def validate_dispatch_start(self, *, feedback_sequence, feedback_timestamp_sec, now_sec=None):
        if self._coordinated is not None:
            return self._coordinated.validate_dispatch_start(feedback_sequence=feedback_sequence,
                feedback_timestamp_sec=feedback_timestamp_sec, now_sec=now_sec)
        with self._lock:
            clock = time.monotonic() if now_sec is None else finite(now_sec, "dispatch start")
            if self._pending is None or self._pending[:2] != (feedback_sequence, feedback_timestamp_sec) or not feedback_timestamp_sec <= clock <= feedback_timestamp_sec + self.settings.staged_assist.motion_control.max_control_gap_sec:
                self.recording_failed("workspace_dispatch_start_identity_or_deadline")
                raise ValueError(self._reason)

    def record_dispatch(self, *, native_receipt, **values):
        if self._coordinated is not None:
            return self._coordinated.record_dispatch(native_receipt=native_receipt, **values)
        with self._lock:
            if self._pending is None:
                raise ValueError("workspace has no pending dispatch")
            sequence, timestamp, delegate, session = self._pending
            now = finite(values.get("completed_at_sec"), "dispatch completion")
            reason = calibration_receipt_rejection(native_receipt, sequence,
                None if self._receipt is None else self._receipt["sequence"])
            requested = finite(native_receipt.get("requested_at_sec"), "dispatch request")
            if reason or native_receipt.get("completed_at_sec") != now or native_receipt.get("sequence") != values.get("receipt_sequence") or values.get("feedback_timestamp_sec") != timestamp or not timestamp <= requested <= now <= timestamp + self.settings.staged_assist.motion_control.max_control_gap_sec:
                self.recording_failed(reason or "workspace_dispatch_clock_or_identity")
                raise ValueError(self._reason)
            if session is None:
                delegate.record_dispatch(**values)
            else:
                session.record_dispatch(receipt=native_receipt, joint_commands=values["joint_commands"], torque_rate_nm_s=values["torque_rate_nm_s"])
            commands = values["joint_commands"]
            delta = tuple(t - g for t, g in zip(six(commands["total_nm"], "total"), six(commands["gravity_nm"], "gravity"), strict=True))
            if self._receipt is not None and self._started is not None:
                self._exposure += sum(abs(v) for v in self._last_delta) * (now - max(self._receipt["completed_at_sec"], self._started))
            self._last_delta = delta
            self._zero_base = (all(abs(v) <= 1e-9 for v in delta)
                and six(commands["overlay_nm"], "overlay") == ZERO
                and six(commands["driver_kd_nm_s_per_rad"], "driver damping") == self._condition.driver_kd_nm_s_per_rad)
            self._receipt, self._pending = dict(native_receipt), None
            if self._phase == "failed":
                return
            if self._phase == "stopping":
                if self._zero_base and delegate.owner_snapshot()["operator_phase"] == "inactive":
                    if session is not None:
                        self._retain_session_result(session, now)
                    self._transition(self._stop_destination, self._cancel, now)
                return
            if session is not None:
                result = session.owner_snapshot()
                if result["state"] == "idle":
                    if self._zero_base and delegate.owner_snapshot()["prepare_gate"] == "ready":
                        session.begin(now_sec=now,
                            allow_motion=True, human_intervention=self._human, allow_breakaway=True)
                elif result["state"] in {"failed", "cancelled"}:
                    self._retain_session_result(session, now)
                    self._transition(result["state"], result["reason"], now)
                elif result["state"] == "complete":
                    self._retain_session_result(session, now)
                    if self._tasks[self._index]["kind"] == "sweep":
                        self._index += 1
                return
            if self._phase in {"observe", "return_observe"} and self._zero_base:
                if self._observation is None:
                    self._observation = {"started_at_sec": now, "anchor_qpos_rad": self._q, "minimum_rad": self._q, "maximum_rad": self._q}
                observation = self._observation
                observation["minimum_rad"] = tuple(min(a, b) for a, b in zip(observation["minimum_rad"], self._q))
                observation["maximum_rad"] = tuple(max(a, b) for a, b in zip(observation["maximum_rad"], self._q))
                if timestamp - observation["started_at_sec"] >= self.workspace.gravity_observation_sec:
                    observation.update(completed_at_sec=now, final_qpos_rad=self._q,
                        task=copy.deepcopy(self._tasks[self._index]), zero_assist_and_base_damping_confirmed=True,
                        stationary_end_confirmed=delegate.owner_snapshot()["prepare_gate"] == "ready",
                        final_gravity_command_nm=tuple(commands["gravity_nm"]))
                    self._observations.append(observation)
                    self._records.append({"kind": "workspace_gravity_observation", "recorded_monotonic_sec": now, **observation})
                    self._observation = None
                    self._index += 1
                    self._transition("advancing", "gravity_observation_complete", now)

    @property
    def cycle_disable_confirmation_required(self):
        return True

    @property
    def cycle_disable_confirmation_ready(self):
        if self._coordinated is not None:
            return self._coordinated.cycle_disable_confirmation_ready
        with self._lock:
            return (self._phase == "complete" and self._zero_base and self._pending is None
                and self._origin is not None and self._q is not None
                and self._delegate.owner_snapshot()["prepare_gate"] == "ready"
                and all(abs(self._q[i] - self._origin[i]) <= self.config.motion_acquisition_arrival_tolerance_rad for i in (1, 2)))

    def _workspace_status(self, *, detailed):
        view = {"phase": self._phase, "reason": self._reason,
            "setup_label": self.workspace.setup_label,
            "joint_ranges_deg": {"joint2": [0, 120], "joint3": [0, 90]},
            "acquisition_ranges_deg": {"joint2": self.workspace.joint2_acquisition_range_deg,
                                       "joint3": self.workspace.joint3_acquisition_range_deg},
            "maximum_start_angle_deg": self.workspace.maximum_start_angle_deg,
            "positioning_joint_order": self.workspace.positioning_joint_order,
            "j2_observed_elastic_below_zero_deg": 0.3,
            "feedback_boundary_policy": "existing_owner_projection_unchanged",
            "task_index": self._index, "task_count": len(self._tasks),
            "current_task": None if self._index >= len(self._tasks) else self._tasks[self._index],
            "pose_count": len(self.workspace.pose_centres()),
            "pose_centres_rad": self.workspace.pose_centres(),
            "completed_joint_acquisitions": sum(r["result"]["state"] == "complete" for r in self._acquisitions),
            "expected_joint_acquisitions": 2 * len(self.workspace.pose_centres()),
            "command_exposure_nm_sec": self._exposure,
            "max_command_exposure_nm_sec": self.workspace.max_command_exposure_nm_sec,
            "origin_qpos_rad": self._origin, "current_qpos_rad": self._q,
            "auto_apply": False, "gravity_parameters_frozen": True,
            "holding_search_enabled": self.workspace.holding_search is not None,
            "local_excursion_rad": self.workspace.local_excursion_rad(self.config),
            "holding_calibration": (None if self._session is None else
                self._session.owner_snapshot().get("holding_calibration")),
            "quantity": "software_command_and_encoder_response_not_measured_force"}
        if detailed:
            acquisitions = list(self._acquisitions)
            moves = list(self._moves)
            if self._session is not None and self._index < len(self._tasks):
                # This is an observation of an interrupted child, never a
                # terminal/zero-dispatch claim. Keep unconfirmed receipts visible.
                partial = {"task": self._tasks[self._index], "result": self._session.status(),
                    "final_qpos_rad": self._q, "partial_stage": True}
                destination = acquisitions if partial["task"]["kind"] == "sweep" else moves
                destination.append(partial)
            view.update(settings=asdict(self.workspace),
                pose_centres_rad=self.workspace.pose_centres(),
                acquisitions=copy.deepcopy(acquisitions),
                gravity_observations=copy.deepcopy(self._observations), moves=copy.deepcopy(moves))
        return view

    def _batch_status(self, *, detailed):
        return {"operation_mode": "workspace", "state": self._phase, "reason": self._reason,
            "outcome": calibration_outcome(self._phase, self._reason),
            "active": self._active, "start_requested": self._start_requested,
            "stage": "motion", "human_intervention": self._human,
            "onset_trials": [], "onset_index": 0, "expected_onset_trials": 0,
            "workspace": self._workspace_status(detailed=detailed),
            "cycle": {"phase": self._phase, "background_permit_timeout_sec": self.limits.command_timeout_sec,
                "zero_assist_base_damping_confirmed": self._zero_base,
                "disable_confirmation_required": True,
                "disable_confirmation_available": self.cycle_disable_confirmation_ready}}

    def status(self):
        if self._coordinated is not None:
            return self._coordinated.status()
        with self._lock:
            return {**self._delegate.status(), "owner_ready": self._ready,
                "controller_mode": self.controller_mode,
                "automatic_calibration": self._batch_status(detailed=False)}

    def result_record(self):
        if self._coordinated is not None:
            return self._coordinated.result_record()
        with self._lock:
            batch = self._batch_status(detailed=True)
            legs = [{**copy.deepcopy(leg), "joint_name": acquisition["task"]["joint"],
                     "pose_index": acquisition["task"]["pose"]}
                    for acquisition in batch["workspace"]["acquisitions"]
                    for leg in acquisition["result"]["completed_legs"]]
            return {"schema": "b601.automatic-joint-calibration.v1", "kind": "automatic_calibration_result",
                "recorded_monotonic_sec": None if self._receipt is None else self._receipt["completed_at_sec"],
                "batch_id": self._batch_id, "state": self._phase, "reason": self._reason,
                "outcome": calibration_outcome(self._phase, self._reason),
                "condition": None if self._condition is None else asdict(self._condition),
                "last_complete_dispatch": copy.deepcopy(self._receipt),
                "pending_dispatch": None if self._pending is None else {
                    "feedback_sequence": self._pending[0], "feedback_timestamp_sec": self._pending[1],
                    "receipt_confirmed": False},
                "settings": asdict(self.config), "onset_trials": [], "incomplete_onset_trial": None,
                "onset_damping": {"stage": "not_requested_gravity_first"},
                "motion": {"completed_legs": legs}, **batch}

    def drain_records(self):
        if self._coordinated is not None:
            return self._coordinated.drain_records()
        with self._lock:
            records, self._records = self._records, []
            return records

    def recording_failed(self, reason):
        if self._coordinated is not None:
            return self._coordinated.recording_failed(reason)
        with self._lock:
            if self._phase != "failed" or calibration_outcome(self._phase, self._reason)["severity"] != "error":
                self._transition("failed", str(reason), self._last_feedback or 0.0)

    def owner_stopped(self, *, now_sec, reason):
        if self._coordinated is not None:
            return self._coordinated.owner_stopped(now_sec=now_sec, reason=reason)
        with self._lock:
            self._closed, self._start_requested, self._ready = True, False, False
            self._delegate.owner_stopped(now_sec=now_sec, reason=reason)
            if self._session is not None:
                self._retain_session_result(self._session, now_sec)
            if self._active:
                self._transition("failed", f"owner_stopped:{reason}", now_sec)
