"""Loopback-console adapter for the opt-in staged single-joint controller.

The adapter owns operator intent and translates it into timestamped policy
references. The hardware owner remains responsible for feedback admission,
gravity compensation, final torque conditioning, and the only motor dispatch.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from typing import Literal

from robot_motion.rebot_b601_rs import (
    B601PolicyAssistCommand,
    B601SharedControlConfig,
    B601StagedAssistConfig,
    B601StagedAssistPhase,
    B601StagedSingleJointAssistRuntime,
)
from robot_motion.rebot_b601_rs.joint_motion import (
    B601WindowStopEvidence,
    B601WindowVelocityEstimator,
    window_stationarity_gate_reason,
)
from robot_motion.rebot_b601_rs.model import B601_RS_ARM_JOINT_NAMES
from robot_motion.rebot_b601_rs.torque_experiment import (
    B601TorqueExperimentLimits,
    joint_command_snapshot,
)

_ZERO_SIX = (0.0,) * 6


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


@dataclass(frozen=True)
class B601StagedTorqueConsoleSettings:
    """Explicit envelope and low-gain tracking settings for the GUI mode."""

    limits: B601TorqueExperimentLimits
    staged_assist: B601StagedAssistConfig
    maximum_target_offset_rad: float
    tracking_kp_nm_per_rad: float
    tracking_kd_nm_s_per_rad: float
    authority_rise_per_sec: float
    nominal_control_period_sec: float

    def __post_init__(self) -> None:
        if not isinstance(self.limits, B601TorqueExperimentLimits):
            raise TypeError("limits must be B601TorqueExperimentLimits")
        if not isinstance(self.staged_assist, B601StagedAssistConfig):
            raise TypeError("staged_assist must be B601StagedAssistConfig")
        if self.staged_assist.continuous_motion_handoff:
            raise ValueError("continuous motion handoff is an offline simulation candidate, not a hardware console mode")
        for name in (
            "maximum_target_offset_rad",
            "tracking_kp_nm_per_rad",
            "authority_rise_per_sec",
            "nominal_control_period_sec",
        ):
            if _finite(getattr(self, name), name=name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        kd = _finite(self.tracking_kd_nm_s_per_rad, name="tracking_kd_nm_s_per_rad")
        if kd < 0.0:
            raise ValueError("tracking_kd_nm_s_per_rad must be non-negative")
        motion = self.staged_assist.motion_control
        if motion is not None:
            if motion.reference_pacing and (
                self.staged_assist.tracking_recovery_gain_nm_per_rad is None
                or self.staged_assist.breakaway_probe is None
                or not motion.responsive_tracking_brake
                or not motion.windowed_soft_velocity
                or motion.position_resolution_rad is None
            ):
                raise ValueError("reference pacing requires explicit windowed tracking recovery")
            if self.limits.joint_name not in {"joint2", "joint3"}:
                raise ValueError("motion-control commissioning supports J2/J3 only")
            if motion.soft_velocity_rad_s >= self.limits.max_joint_velocity_rad_s:
                raise ValueError("soft velocity must be below the existing hard guard")
            if (motion.continuous_speed_limit_rad_s is not None
                and motion.continuous_speed_limit_rad_s >= self.limits.max_joint_velocity_rad_s):
                raise ValueError("continuous speed ceiling must be below the existing hard guard")
            if motion.max_reference_lag_rad > self.maximum_target_offset_rad:
                raise ValueError("reference lag must fit the target excursion envelope")
            if any(motion.drive_driver_kd_nm_s_per_rad) and 1.0 / self.nominal_control_period_sec < motion.minimum_owner_rate_hz:
                raise ValueError("owner rate is below the motion profile minimum")
        if self.maximum_target_offset_rad > self.limits.max_excursion_rad:
            raise ValueError("target offset limit must not exceed the excursion guard")
        confirmation = self.staged_assist.motion_onset_confirm_displacement_rad
        if confirmation is not None and confirmation >= self.maximum_target_offset_rad:
            raise ValueError("motion confirmation displacement must fit the target offset envelope")
        if self.staged_assist.max_total_overlay_nm > self.limits.max_abs_assist_nm:
            raise ValueError("staged overlay cap exceeds torque-console assist limit")
        one_cycle_hard_limit_displacement = (
            self.limits.max_joint_velocity_rad_s
            * self.nominal_control_period_sec
        )
        if (
            self.staged_assist.motion_onset_displacement_rad
            >= one_cycle_hard_limit_displacement
        ):
            raise ValueError(
                "motion-onset displacement must be below the hard-velocity "
                "limit times the owner control period"
            )
        if (
            self.staged_assist.motion_onset_velocity_rad_s
            > 0.5 * self.limits.max_joint_velocity_rad_s
        ):
            raise ValueError(
                "motion-onset velocity must not exceed half the hard-velocity "
                "guard to leave a control-cycle response margin"
            )


class B601StagedTorqueConsoleController:
    """Thread-safe GUI intent bridge, stepped only from the hardware owner."""

    controller_mode = "staged_assist"

    def __init__(self, settings: B601StagedTorqueConsoleSettings) -> None:
        if not isinstance(settings, B601StagedTorqueConsoleSettings):
            raise TypeError("settings must be B601StagedTorqueConsoleSettings")
        self.settings = settings
        self.limits = settings.limits
        self._lock = threading.RLock()
        self._ready = False
        self._operator_phase: Literal["inactive", "arm_requested", "armed", "releasing", "fault"] = "inactive"
        self._fault: str | None = None
        self._anchor_qpos: tuple[float, ...] | None = None
        self._qpos: tuple[float, ...] | None = None
        self._qvel: tuple[float, ...] | None = None
        self._target_offset_rad: float | None = None
        self._target_restart_pending = False
        self._target_update_pending = False
        self._breakaway_authorized = False
        self._last_intent_sec: float | None = None
        self._last_step_sec: float | None = None
        self._sequence = 0
        self._control_sequence = 0
        self._last_frame = None
        self._last_overlay = _ZERO_SIX
        self._last_dispatch: dict[str, object] | None = None
        self._dispatch_count = 0
        self._breakaway_peak_overlay_nm = 0.0
        self._breakaway_peak_dispatched_nm = 0.0
        self._breakaway_peak_conditioned_delta_nm = 0.0
        self._history: deque[dict[str, object]] = deque(maxlen=240)
        self._events: deque[dict[str, object]] = deque(maxlen=100)
        self._runtime = self._build_runtime(self.limits.joint_name)
        motion = settings.staged_assist.motion_control
        self._reference_velocity_ceiling = None if motion is None else motion.max_reference_velocity_rad_s
        self._velocity_estimators = (
            tuple(B601WindowVelocityEstimator(motion.velocity_window_sec) for _ in range(6))
            if motion is not None and motion.windowed_soft_velocity else None
        )
        self._velocity_estimator = (
            self._velocity_estimators[self._joint_index] if self._velocity_estimators is not None
            else None if motion is None else B601WindowVelocityEstimator(motion.velocity_window_sec)
        )
        self._control_qvel = None

    def _build_runtime(self, joint_name: str) -> B601StagedSingleJointAssistRuntime:
        joint_index = B601_RS_ARM_JOINT_NAMES.index(joint_name)
        kp = [0.0] * 6
        kd = [0.0] * 6
        kp[joint_index] = self.settings.tracking_kp_nm_per_rad
        kd[joint_index] = self.settings.tracking_kd_nm_s_per_rad
        shared_config = B601SharedControlConfig(
            position_kp_nm_per_rad=tuple(kp),
            velocity_kd_nm_s_per_rad=tuple(kd),
            max_target_error_rad=(self.settings.maximum_target_offset_rad,) * 6,
            max_abs_assist_torque_nm=(self.limits.max_abs_assist_nm,) * 6,
            max_assist_torque_rate_nm_s=(self.limits.rise_slew_nm_s,) * 6,
            authority_rise_per_sec=self.settings.authority_rise_per_sec,
            policy_stale_timeout_sec=self.limits.command_timeout_sec,
        )
        staged_config = replace(
            self.settings.staged_assist,
            joint_name=joint_name,
            max_total_overlay_nm=self.limits.max_abs_assist_nm,
        )
        return B601StagedSingleJointAssistRuntime(shared_config, staged_config)

    def owner_ready(self) -> None:
        with self._lock:
            if self._operator_phase != "fault":
                self._ready = True

    def set_command_timeout_sec(self, timeout_sec: float) -> None:
        """Update the GUI lease only before this controller starts a target."""
        timeout = _finite(timeout_sec, name="command_timeout_sec")
        if timeout <= 0:
            raise ValueError("command_timeout_sec must be positive")
        with self._lock:
            if self._operator_phase != "inactive" or self._target_offset_rad is not None:
                raise ValueError("command timeout can only change while inactive")
            self.limits = replace(self.limits, command_timeout_sec=timeout)
            self.settings = replace(self.settings, limits=self.limits)
            self._runtime = self._build_runtime(self.limits.joint_name)

    def owner_snapshot(self) -> dict[str, object]:
        """Small owner-tick view; no history copies or additional state owner."""
        with self._lock:
            frame = self._last_frame
            return {
                "ready": self._ready, "operator_phase": self._operator_phase,
                "phase": None if frame is None else frame.phase.value,
                "reason": None if frame is None else frame.reason,
                "fault": self._fault, "feedback_timestamp_sec": self._last_step_sec,
                "qpos_rad": self._qpos, "anchor_qpos_rad": self._anchor_qpos,
                "target_offset_rad": self._target_offset_rad,
                "overlay_torque_nm": self._last_overlay,
                "prepare_gate": self._stationary_gate_reason(self.limits.stationarity_limit_rad_s, all_joints=True),
                "arrival_stop_gate": self._stationary_gate_reason(self.settings.staged_assist.settle_velocity_rad_s, all_joints=True),
                "breakaway_gate": self._breakaway_gate_reason(),
                "motion_phase": None if frame is None or frame.motion is None else frame.motion.phase,
                "target_error_rad": None if frame is None or frame.motion is None else frame.motion.target_error_rad,
                "release_failed": False if frame is None else frame.release_failed,
                "release_terminal": False if frame is None else frame.release_terminal,
            }

    def configure_reference_velocity(self, velocity_rad_s: float) -> None:
        """Bounded offline sequencing seam; only after an acknowledged zero.

        No GUI action or hardware CLI exposes this operation. Rebuilding the
        motion runtime while active would discard withdrawal state, so it is
        deliberately forbidden, including while waiting for a receipt.
        """
        speed = _finite(velocity_rad_s, name="velocity_rad_s")
        with self._lock:
            if self._reference_velocity_ceiling is None or not 0 < speed <= self._reference_velocity_ceiling:
                raise ValueError("reference speed must fit the original motion profile")
            receipt = self._last_dispatch
            commands = None if receipt is None else receipt["joint_commands"]
            if (not self._ready or self._operator_phase != "inactive" or self._fault
                    or any(abs(v) > 1e-9 for v in self._last_overlay)
                    or receipt is None or receipt["feedback_timestamp_sec"] != self._last_step_sec
                    or commands is None
                    or any(abs(t - g) > 1e-9 for t, g in zip(commands["total_nm"], commands["gravity_nm"]))):
                raise ValueError("reference speed changes require inactive acknowledged zero assist")
            motion = replace(self.settings.staged_assist.motion_control, max_reference_velocity_rad_s=speed)
            self.settings = replace(self.settings, staged_assist=replace(self.settings.staged_assist, motion_control=motion))
            self._runtime = self._build_runtime(self.limits.joint_name)
            self._last_frame = None
            self._anchor_qpos = None
            self._target_offset_rad = None
            self._target_update_pending = False
            self._target_restart_pending = False
            self._breakaway_authorized = False

    def submit(
        self,
        action: str,
        *,
        now_sec: float,
        torque_nm: float | None = None,
        target_offset_rad: float | None = None,
        joint_name: str | None = None,
        withdraw_source: str = "operator_button",
    ) -> None:
        now = _finite(now_sec, name="now_sec")
        if now < 0.0:
            raise ValueError("now_sec must be non-negative")
        if action == "withdraw" and withdraw_source not in {
            "operator_button", "page_hidden", "page_unload", "owner_cancel"
        }:
            raise ValueError("unsupported withdrawal source")
        with self._lock:
            if action == "select_joint":
                if self.settings.staged_assist.motion_control is not None and joint_name not in {"joint2", "joint3"}:
                    raise ValueError("motion-control commissioning supports J2/J3 only")
                if joint_name not in B601_RS_ARM_JOINT_NAMES:
                    raise ValueError("joint_name must be a B601 arm joint")
                if self._operator_phase != "inactive" or any(
                    abs(value) > 1e-12 for value in self._last_overlay
                ):
                    raise ValueError("joint selection requires inactive, zero-assist state")
                self.limits = replace(self.limits, joint_name=joint_name)
                self._runtime = self._build_runtime(joint_name)
                self._anchor_qpos = None
                self._target_offset_rad = None
                self._target_restart_pending = False
                self._target_update_pending = False
                self._breakaway_authorized = False
                self._last_frame = None
                self._last_overlay = _ZERO_SIX
                self._last_dispatch = None
                self._breakaway_peak_overlay_nm = 0.0
                self._breakaway_peak_dispatched_nm = 0.0
                self._breakaway_peak_conditioned_delta_nm = 0.0
                self._history.clear()
                self._fault = None
            elif action == "arm":
                if not self._ready:
                    raise ValueError("arm requires active gravity owner")
                if self._operator_phase != "inactive":
                    raise ValueError("arm requires inactive state")
                self._operator_phase = "arm_requested"
                self._fault = None
                self._last_intent_sec = now
            elif action == "set_target":
                if self._operator_phase != "armed":
                    raise ValueError("set_target requires armed state")
                if self.settings.staged_assist.motion_onset_confirm_displacement_rad is not None and self._target_offset_rad is not None:
                    raise ValueError("withdraw and reprepare before replacing a confirmed-motion target")
                if target_offset_rad is None:
                    raise ValueError("set_target requires target_offset_rad")
                offset = _finite(target_offset_rad, name="target_offset_rad")
                if abs(offset) > self.settings.maximum_target_offset_rad:
                    raise ValueError("target_offset_rad exceeds configured limit")
                previous_offset = self._target_offset_rad
                staged_phase = (
                    B601StagedAssistPhase.PASSIVE
                    if self._last_frame is None
                    else self._last_frame.phase
                )
                if staged_phase is B601StagedAssistPhase.AUTHORITY_EXHAUSTED:
                    # The same exhausted intent cannot be retried by merely
                    # clicking Apply again; a materially different target is
                    # required, as in the underlying control runtime.
                    self._target_restart_pending = (
                        previous_offset is None
                        or abs(offset - previous_offset)
                        >= self.settings.staged_assist.new_intent_delta_rad
                    )
                else:
                    self._target_restart_pending = (
                        previous_offset is not None
                        and staged_phase is not B601StagedAssistPhase.PASSIVE
                    )
                self._target_offset_rad = offset
                self._target_update_pending = True
                self._runtime.revoke_breakaway_authorization()
                self._breakaway_authorized = False
                self._last_intent_sec = now
                self._fault = None
            elif action == "authorize_breakaway":
                if self._operator_phase != "armed":
                    raise ValueError("breakaway authorization requires armed state")
                if self._target_update_pending or self._target_restart_pending:
                    raise ValueError("breakaway authorization waits for owner target admission")
                gate_reason = self._breakaway_gate_reason()
                if gate_reason != "ready":
                    raise ValueError(f"breakaway authorization is not ready: {gate_reason}")
                sequence = self._sequence
                if sequence <= 0 or not self._runtime.authorize_breakaway(sequence=sequence):
                    raise ValueError("breakaway is available only during the current CUE phase")
                self._breakaway_authorized = True
                self._last_intent_sec = now
            elif action == "heartbeat":
                if self._operator_phase != "armed":
                    raise ValueError("heartbeat requires armed state")
                self._last_intent_sec = now
            elif action == "stop_and_hold":
                if self._last_frame is not None and self._last_frame.phase is B601StagedAssistPhase.RELEASING:
                    raise ValueError("withdrawal already brakes; use withdraw to finish this trial")
                if self._operator_phase != "armed" or self._target_offset_rad is None:
                    raise ValueError("stop-and-hold requires an active target")
                if self._target_update_pending or self._target_restart_pending:
                    raise ValueError("stop-and-hold waits for owner target admission")
                self._runtime.request_stop_and_hold()
                self._breakaway_authorized = False
                self._last_intent_sec = now
            elif action == "zero":
                if self.settings.staged_assist.motion_onset_confirm_displacement_rad is not None:
                    self._withdraw(now, reason="operator_zero")
                    return
                self._target_offset_rad = None
                self._target_restart_pending = False
                self._target_update_pending = False
                self._breakaway_authorized = False
                self._last_intent_sec = now
                self._runtime.clear_policy()
            elif action == "withdraw":
                reason = {
                    "operator_button": "operator_withdraw",
                    "page_hidden": "page_hidden",
                    "page_unload": "page_unload",
                    "owner_cancel": "owner_cancel",
                }[withdraw_source]
                self._withdraw(now, reason=reason)
            else:
                raise ValueError("unsupported staged-assist console action")
            if action not in {"heartbeat", "withdraw"}:
                self._events.append(
                    {
                        "time_sec": now,
                        "action": action,
                        "joint_name": self.limits.joint_name,
                        "requested_target_offset_rad": self._target_offset_rad,
                    }
                )
            elif action == "withdraw":
                self._events.append(
                    {
                        "time_sec": now,
                        "action": action,
                        "source": withdraw_source,
                        "joint_name": self.limits.joint_name,
                    }
                )

    def step(
        self,
        *,
        now_sec: float,
        qpos_rad: tuple[float, ...],
        qvel_rad_s: tuple[float, ...],
        tracking_recovery_permitted: bool = False,
    ) -> tuple[tuple[float, ...], dict[str, object]]:
        now = _finite(now_sec, name="now_sec")
        qpos = tuple(_finite(value, name="qpos_rad") for value in qpos_rad)
        qvel = tuple(_finite(value, name="qvel_rad_s") for value in qvel_rad_s)
        if len(qpos) != 6 or len(qvel) != 6:
            raise ValueError("feedback must contain six joints")
        with self._lock:
            if self._last_step_sec is not None and now <= self._last_step_sec:
                raise ValueError("owner feedback timestamps must advance")
            dt = (
                self.settings.nominal_control_period_sec
                if self._last_step_sec is None
                else now - self._last_step_sec
            )
            self._last_step_sec = now
            self._qpos, self._qvel = qpos, qvel
            control_qvel = list(qvel)
            if self._velocity_estimators is not None:
                if dt > self.settings.staged_assist.motion_control.max_control_gap_sec:
                    for estimator in self._velocity_estimators:
                        estimator.reset()
                control_qvel = [
                    estimator.update(timestamp_sec=now, qpos_rad=q, fallback_rad_s=v)
                    for estimator, q, v in zip(self._velocity_estimators, qpos, qvel, strict=True)
                ]
            elif self._velocity_estimator is not None:
                control_qvel[self._joint_index] = self._velocity_estimator.update(
                    timestamp_sec=now, qpos_rad=qpos[self._joint_index],
                    fallback_rad_s=qvel[self._joint_index],
                )
            self._control_qvel = tuple(control_qvel)
            if self._operator_phase == "arm_requested":
                preparation_reason = self._stationary_gate_reason(
                    self.limits.stationarity_limit_rad_s, all_joints=True,
                )
                if preparation_reason == "ready":
                    self._anchor_qpos = qpos
                    self._operator_phase = "armed"
                    self._fault = None
                    self._events.append(
                        {
                            "time_sec": now,
                            "action": "armed",
                            "joint_name": self.limits.joint_name,
                            "requested_target_offset_rad": None,
                        }
                    )
                else:
                    self._operator_phase = "inactive"
                    self._fault = f"arm_rejected_{preparation_reason}"
                    self._events.append(
                        {
                            "time_sec": now,
                            "action": self._fault,
                            "joint_name": self.limits.joint_name,
                            "requested_target_offset_rad": None,
                        }
                    )
            if self._operator_phase == "releasing":
                guard = self._motion_guard(qpos, qvel)
                if guard is not None:
                    self._operator_phase = "fault"
                    self._fault = guard
                    self._runtime.clear_policy()
            if self._operator_phase == "armed" and self._target_offset_rad is not None:
                age = None if self._last_intent_sec is None else now - self._last_intent_sec
                if age is None or age > self.limits.command_timeout_sec:
                    self._withdraw(now, reason="gui_intent_stale")
                    self._fault = "gui_intent_stale"
                else:
                    guard = self._motion_guard(qpos, qvel)
                    if guard is not None:
                        self._operator_phase = "fault"
                        self._fault = guard
                        self._target_offset_rad = None
                        self._target_restart_pending = False
                        self._target_update_pending = False
                        self._breakaway_authorized = False
                        self._runtime.clear_policy()
                    else:
                        if self._target_restart_pending:
                            # Let the staged runtime observe a no-command tick
                            # before a revised intent can begin a new cue.
                            self._runtime.clear_policy()
                            self._target_restart_pending = False
                            if self.settings.staged_assist.motion_control is not None:
                                # A new local intent starts at admitted measured
                                # state, not at the old pre-braking anchor.
                                self._anchor_qpos = qpos
                        else:
                            self._sequence += 1
                            target = list(self._anchor_qpos or qpos)
                            target[self._joint_index] += self._target_offset_rad
                            command = B601PolicyAssistCommand(
                                sequence=self._sequence,
                                source_timestamp_sec=now,
                                receive_timestamp_sec=now,
                                qpos_target_rad=tuple(target),
                            )
                            receipt = self._runtime.publish_policy(command)
                            if not receipt.accepted:
                                self._operator_phase = "fault"
                                self._fault = f"staged_policy_rejected:{receipt.reason}"
                                self._target_offset_rad = None
                                self._target_update_pending = False
                                self._breakaway_authorized = False
                                self._runtime.clear_policy()
                            else:
                                self._target_update_pending = False
            if self._operator_phase != "armed" or self._target_offset_rad is None:
                self._runtime.clear_policy()
            self._control_sequence += 1
            motion = self.settings.staged_assist.motion_control
            stop_evidence = None
            if motion is not None and motion.position_resolution_rad is not None and self._velocity_estimators is not None:
                stop_evidence = B601WindowStopEvidence(
                    timestamp_sec=now, qpos_rad=qpos, raw_qvel_rad_s=qvel,
                    window_qvel_rad_s=self._control_qvel,
                    position_span_rad=tuple(e.position_span_rad for e in self._velocity_estimators),
                    window_span_sec=self._window_span(),
                    raw_velocity_limit_rad_s=self.limits.max_joint_velocity_rad_s,
                    joint_index=self._joint_index,
                )
            frame = self._runtime.step(
                control_sequence=self._control_sequence,
                control_timestamp_sec=now,
                measured_qpos_rad=qpos,
                measured_qvel_rad_s=qvel,
                dt_sec=dt,
                control_qvel_rad_s=self._control_qvel,
                control_qvel_window_span_sec=self._window_span(),
                stop_evidence=stop_evidence,
                tracking_recovery_permitted=tracking_recovery_permitted,
            )
            overlay = frame.overlay_torque_nm
            previous_stage_phase = (
                None if self._last_frame is None else self._last_frame.phase.value
            )
            self._last_frame = frame
            self._last_overlay = overlay
            stage_phase = frame.phase.value
            if frame.phase is B601StagedAssistPhase.RELEASING and frame.release_terminal:
                self._operator_phase = "releasing" if self._operator_phase != "fault" else "fault"
                self._target_offset_rad = None
                self._target_restart_pending = False
                self._target_update_pending = False
                self._breakaway_authorized = False
                self._runtime.clear_policy()
            if frame.release_failed:
                # The outer six-axis guard already identifies the first
                # offending joint. A generic brake failure must not erase it.
                if self._operator_phase != "fault":
                    self._fault = frame.reason
                self._operator_phase = "fault"
            if (
                stage_phase == B601StagedAssistPhase.CUE.value
                and previous_stage_phase != B601StagedAssistPhase.CUE.value
            ):
                self._breakaway_peak_overlay_nm = 0.0
                self._breakaway_peak_dispatched_nm = 0.0
                self._breakaway_peak_conditioned_delta_nm = 0.0
            if stage_phase == B601StagedAssistPhase.BREAKAWAY.value:
                self._breakaway_peak_overlay_nm = max(
                    self._breakaway_peak_overlay_nm, abs(overlay[self._joint_index])
                )
            if (
                stage_phase != B601StagedAssistPhase.CUE.value
                or frame.reason.startswith("breakaway_authorization_rejected_")
            ):
                self._breakaway_authorized = False
            phase = (
                "fault"
                if self._operator_phase == "fault"
                else self._operator_phase
                if self._target_offset_rad is None
                else stage_phase
            )
            index = self._joint_index
            audit: dict[str, object] = {
                "time_sec": now,
                "phase": phase,
                "stage_phase": stage_phase,
                "policy_authority": frame.base_frame.decision.state.authority,
                "breakaway_peak_overlay_nm": self._breakaway_peak_overlay_nm,
                "breakaway_peak_dispatched_nm": self._breakaway_peak_dispatched_nm,
                "reason": frame.reason,
                "motion_onset_confirm_count": frame.motion_onset_confirm_count,
                "motion_onset_required_samples": frame.motion_onset_required_samples,
                "breakaway_probe": frame.breakaway_probe,
                "tracking_recovery": frame.tracking_recovery,
                "release_terminal": frame.release_terminal,
                "release_failed": frame.release_failed,
                "release_restoring_base_damping": frame.release_restoring_base_damping,
                "conditioned_command_delta_nm": frame.conditioned_command_delta_nm,
                "conditioned_command_impulse_nm_sec": frame.conditioned_command_impulse_nm_sec,
                "owner_ready": self._ready,
                "operator_armed": self._operator_phase == "armed",
                "controller_mode": self.controller_mode,
                "joint_name": self.limits.joint_name,
                "policy_sequence": frame.policy_sequence,
                "intent_generation": frame.intent_generation,
                "target_offset_rad": self._target_offset_rad,
                "target_update_pending": self._target_update_pending,
                "breakaway_authorized": self._breakaway_authorized,
                "breakaway_eligible": self._breakaway_gate_reason() == "ready",
                "breakaway_gate_reason": self._breakaway_gate_reason(),
                "desired_assist_nm": overlay[index],
                "effective_assist_nm": overlay[index],
                "overlay_torque_nm": overlay,
                "breakaway_impulse_nm_sec": frame.breakaway_impulse_nm_sec,
                "settle_integral_torque_nm": frame.settle_integral_torque_nm,
                "qpos_rad": qpos,
                "qvel_fd_rad_s": qvel,
                "control_qvel_rad_s": self._control_qvel,
                "velocity_window_span_sec": None if self._velocity_estimator is None else self._velocity_estimator.span_sec,
                "velocity_window": self._window_status(),
                "motion_control": None if frame.motion is None else asdict(frame.motion),
                "drive_driver_kd_nm_s_per_rad": frame.drive_driver_kd_nm_s_per_rad,
                "support_max_velocity_rad_s": frame.support_max_velocity_rad_s,
                "support_raw_max_velocity_rad_s": frame.support_raw_max_velocity_rad_s,
                "anchor_rad": None if self._anchor_qpos is None else self._anchor_qpos[index],
                "displacement_rad": (
                    None if self._anchor_qpos is None else qpos[index] - self._anchor_qpos[index]
                ),
                "support_drift_rad": self._support_drift(qpos),
                "fault": self._fault,
            }
            self._history.append(audit)
            return overlay, audit

    @property
    def _joint_index(self) -> int:
        return B601_RS_ARM_JOINT_NAMES.index(self.limits.joint_name)

    def _window_span(self) -> float | None:
        return None if self._velocity_estimators is None else min(e.span_sec for e in self._velocity_estimators)

    def _window_status(self) -> dict[str, object]:
        return {
            "enabled": self._velocity_estimators is not None,
            "ready": None if self._velocity_estimators is None else all(e.ready for e in self._velocity_estimators),
            "span_sec": self._window_span(),
            "position_span_rad": None if self._velocity_estimators is None else tuple(e.position_span_rad for e in self._velocity_estimators),
            "soft_decision_source": "raw_position_finite_difference" if self._velocity_estimators is None else "causal_window_velocity_and_displacement",
            "hard_guard_source": "raw_position_finite_difference",
            "position_resolution_rad": (
                None if self.settings.staged_assist.motion_control is None else
                self.settings.staged_assist.motion_control.position_resolution_rad
            ),
        }

    def _stationary_gate_reason(self, limit: float, *, all_joints: bool) -> str:
        if self._qvel is None:
            return "feedback_unavailable"
        indices = range(6) if all_joints else (self._joint_index,)
        if self._velocity_estimators is None:
            return "ready" if max(abs(self._qvel[i]) for i in indices) <= limit else "not_stationary"
        return window_stationarity_gate_reason(
            self._velocity_estimators, self._qvel, self._control_qvel,
            soft_limit_rad_s=limit, hard_limit_rad_s=self.limits.max_joint_velocity_rad_s,
            joint_index=None if all_joints else self._joint_index,
        )

    def active_driver_damping(
        self, base_kd: tuple[float, ...], *, final_command_delta_nm: tuple[float, ...] | None = None,
    ) -> tuple[float, ...]:
        """Owner-only dispatch setting; GUI cannot submit motor gains."""
        with self._lock:
            drive = (
                _ZERO_SIX if self._last_frame is None or self._operator_phase not in {"armed", "releasing"}
                else self._last_frame.drive_driver_kd_nm_s_per_rad
            )
            if (
                self._operator_phase == "releasing" and self._last_frame is not None
                and self._last_frame.release_restoring_base_damping
            ):
                if final_command_delta_nm is not None:
                    delta = tuple(_finite(v, name="final command delta") for v in final_command_delta_nm)
                    if len(delta) != 6:
                        raise ValueError("final command delta requires six joints")
                else:
                    delta = None
                if delta is None or any(abs(v) > 1e-9 for v in delta):
                    # Last tick's zero receipt does not prove THIS packet is
                    # zero after a changed gravity baseline/final limiter.
                    if self._last_dispatch is None or self._last_dispatch["driver_kd_nm_s_per_rad"] is None:
                        raise ValueError("base damping restoration requires a previous complete gain receipt")
                    drive = self._last_dispatch["driver_kd_nm_s_per_rad"]
            return tuple(max(base, active) for base, active in zip(base_kd, drive, strict=True))

    def _breakaway_gate_reason(self) -> str:
        if self._operator_phase != "armed":
            return "not_armed"
        if self._last_frame is None or self._last_frame.phase is not B601StagedAssistPhase.CUE:
            return "not_in_cue"
        if self._target_update_pending or self._target_restart_pending:
            return "target_update_pending"
        if self._breakaway_authorized:
            return "already_authorized"
        if (
            self._anchor_qpos is None
            or self._qpos is None
            or self._qvel is None
            or self._target_offset_rad is None
        ):
            return "feedback_or_target_unavailable"
        index = self._joint_index
        target_error = self._anchor_qpos[index] + self._target_offset_rad - self._qpos[index]
        minimum_error = self.settings.staged_assist.minimum_breakaway_error_rad
        if abs(target_error) < minimum_error:
            return "target_error_below_breakaway_minimum"
        stationary_reason = self._stationary_gate_reason(
            self.settings.staged_assist.stuck_max_velocity_rad_s, all_joints=False,
        )
        if stationary_reason != "ready":
            return "joint_not_stationary_for_breakaway" if stationary_reason == "not_stationary" else stationary_reason
        return "ready"

    def _motion_guard(
        self,
        qpos: tuple[float, ...],
        qvel: tuple[float, ...],
    ) -> str | None:
        index = self._joint_index
        anchor_q = self._anchor_qpos
        if anchor_q is None:
            return "staged_target_without_stationary_anchor"
        if abs(qvel[index]) > self.limits.max_joint_velocity_rad_s:
            return (
                f"joint_velocity_exceeded: {self.limits.joint_name} "
                f"abs_qvel={abs(qvel[index]):.6f}rad/s "
                f"limit={self.limits.max_joint_velocity_rad_s:.6f}rad/s"
            )
        excursion = qpos[index] - anchor_q[index]
        if abs(excursion) > self.limits.max_excursion_rad:
            return (
                f"joint_excursion_exceeded: {self.limits.joint_name} "
                f"delta={excursion:+.6f}rad "
                f"limit={self.limits.max_excursion_rad:.6f}rad"
            )
        for support_index, name in enumerate(B601_RS_ARM_JOINT_NAMES):
            if support_index == index:
                continue
            motion = self.settings.staged_assist.motion_control
            if motion is not None and any(motion.drive_driver_kd_nm_s_per_rad) and abs(qvel[support_index]) > self.limits.max_joint_velocity_rad_s:
                return (
                    f"support_joint_velocity_exceeded: {name} "
                    f"abs_qvel={abs(qvel[support_index]):.6f}rad/s "
                    f"limit={self.limits.max_joint_velocity_rad_s:.6f}rad/s"
                )
            drift = qpos[support_index] - anchor_q[support_index]
            if abs(drift) > self.limits.max_support_drift_rad:
                return (
                    f"support_joint_drift_exceeded: {name} "
                    f"delta={drift:+.6f}rad "
                    f"limit={self.limits.max_support_drift_rad:.6f}rad"
                )
        return None

    def _support_drift(self, qpos: tuple[float, ...]) -> dict[str, float] | None:
        anchor_q = self._anchor_qpos
        if anchor_q is None:
            return None
        return {
            name: qpos[index] - anchor_q[index]
            for index, name in enumerate(B601_RS_ARM_JOINT_NAMES)
            if index != self._joint_index
        }

    def _withdraw(self, now: float, *, reason: str, immediate: bool = False) -> None:
        self._target_offset_rad = None
        self._target_restart_pending = False
        self._target_update_pending = False
        self._breakaway_authorized = False
        pending_release = (
            not immediate and self._operator_phase in {"armed", "releasing"}
            and self.settings.staged_assist.motion_onset_confirm_displacement_rad is not None
        )
        self._operator_phase = "releasing" if pending_release else "inactive"
        self._last_intent_sec = now
        self._runtime.clear_policy()
        if pending_release:
            self._runtime.request_withdrawal(reason)
        self._events.append(
            {
                "time_sec": now,
                "action": "withdraw",
                "joint_name": self.limits.joint_name,
                "requested_target_offset_rad": None,
                "reason": reason,
            }
        )

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
        torque_rate_nm_s: tuple[float, ...] | None = None,
    ) -> None:
        if isinstance(receipt_sequence, bool) or not isinstance(receipt_sequence, int) or receipt_sequence < 0:
            raise ValueError("receipt_sequence must be non-negative")
        commands = None if joint_commands is None else dict(joint_commands)
        driver_kd = None if commands is None else commands.pop("driver_kd_nm_s_per_rad", None)
        if driver_kd is not None:
            driver_kd = tuple(_finite(v, name="driver_kd_nm_s_per_rad") for v in driver_kd)
            if len(driver_kd) != 6 or min(driver_kd) < 0:
                raise ValueError("driver damping dispatch requires six non-negative gains")
        record = {
            "receipt_sequence": receipt_sequence,
            "feedback_timestamp_sec": _finite(feedback_timestamp_sec, name="feedback_timestamp_sec"),
            "overlay_after_profile_nm": _finite(overlay_after_profile_nm, name="overlay_after_profile_nm"),
            "gravity_nm": _finite(gravity_nm, name="gravity_nm"),
            "requested_total_nm": _finite(requested_total_nm, name="requested_total_nm"),
            "conditioned_total_nm": _finite(conditioned_total_nm, name="conditioned_total_nm"),
            "joint_commands": joint_command_snapshot(commands),
            "driver_kd_nm_s_per_rad": driver_kd,
        }
        with self._lock:
            if joint_name != self.limits.joint_name:
                if self.settings.staged_assist.motion_onset_confirm_displacement_rad is not None:
                    raise ValueError("conditioned receipt joint does not match the active console")
                return
            matched = bool(self._history and self._history[-1]["time_sec"] == record["feedback_timestamp_sec"])
            strict = self.settings.staged_assist.motion_onset_confirm_displacement_rad is not None
            if strict and (not matched or self._last_dispatch is not None and receipt_sequence <= self._last_dispatch["receipt_sequence"]):
                raise ValueError("conditioned receipt must match current feedback and advance its sequence")
            if matched:
                self._runtime.record_dispatch(
                    self._control_sequence,
                    conditioned_torque_nm=None if commands is None else commands.get("total_nm"),
                    passive_torque_nm=None if commands is None else commands.get("gravity_nm"),
                    driver_kd_nm_s_per_rad=driver_kd,
                    torque_rate_nm_s=torque_rate_nm_s,
                    completed_at_sec=completed_at_sec,
                )
            self._dispatch_count += 1
            record.update(
                {
                    "joint_name": joint_name,
                    "dispatch_count": self._dispatch_count,
                    "recorded_at_sec": time.monotonic(),
                }
            )
            self._last_dispatch = record
            if matched:
                if self._operator_phase == "releasing" and self._runtime.release_completed:
                    self._withdraw(float(completed_at_sec), reason="conditioned_withdrawal_and_stop_confirmed", immediate=True)
                history_record = {**self._history[-1], **record}
                if history_record.get("stage_phase") == B601StagedAssistPhase.BREAKAWAY.value:
                    self._breakaway_peak_dispatched_nm = max(
                        self._breakaway_peak_dispatched_nm,
                        abs(record["overlay_after_profile_nm"]),
                    )
                    self._breakaway_peak_conditioned_delta_nm = max(
                        self._breakaway_peak_conditioned_delta_nm,
                        abs(record["conditioned_total_nm"] - record["gravity_nm"]),
                    )
                history_record["breakaway_peak_dispatched_nm"] = (
                    self._breakaway_peak_dispatched_nm
                )
                history_record["breakaway_peak_conditioned_delta_nm"] = (
                    self._breakaway_peak_conditioned_delta_nm
                )
                self._history[-1] = history_record

    def owner_stopped(self, *, now_sec: float, reason: str) -> None:
        now = _finite(now_sec, name="now_sec")
        with self._lock:
            self._withdraw(now, reason=reason, immediate=True)
            self._ready = False
            self._operator_phase = "fault"
            self._fault = str(reason).strip() or "owner_stopped"

    def status(self) -> dict[str, object]:
        with self._lock:
            stage_phase = (
                B601StagedAssistPhase.PASSIVE.value
                if self._last_frame is None
                else self._last_frame.phase.value
            )
            phase = (
                "fault"
                if self._operator_phase == "fault"
                else self._operator_phase
                if self._target_offset_rad is None
                else stage_phase
            )
            index = self._joint_index
            displacement = (
                None
                if self._anchor_qpos is None or self._qpos is None
                else self._qpos[index] - self._anchor_qpos[index]
            )
            feedback_age = (
                None
                if self._last_step_sec is None
                else max(0.0, time.monotonic() - self._last_step_sec)
            )
            dispatch_age = (
                None
                if self._last_dispatch is None
                else max(0.0, time.monotonic() - float(self._last_dispatch["recorded_at_sec"]))
            )
            return {
                "controller_mode": self.controller_mode,
                "motion_control_enabled": self.settings.staged_assist.motion_control is not None,
                "motion_control_settings": None if self.settings.staged_assist.motion_control is None else asdict(self.settings.staged_assist.motion_control),
                "breakaway_probe_settings": None if self.settings.staged_assist.breakaway_probe is None else asdict(self.settings.staged_assist.breakaway_probe),
                "breakaway_command_envelope": self.settings.staged_assist.breakaway_envelope(
                    control_period_sec=self.settings.nominal_control_period_sec,
                ),
                "motion_control": None if self._last_frame is None or self._last_frame.motion is None else asdict(self._last_frame.motion),
                "control_qvel_rad_s": self._control_qvel,
                "velocity_window_span_sec": None if self._velocity_estimator is None else self._velocity_estimator.span_sec,
                "velocity_window": self._window_status(),
                "drive_driver_kd_nm_s_per_rad": (
                    _ZERO_SIX if self._last_frame is None or not self._ready or self._operator_phase not in {"armed", "releasing"}
                    else self._last_frame.drive_driver_kd_nm_s_per_rad
                ),
                "support_max_velocity_rad_s": 0.0 if self._last_frame is None else self._last_frame.support_max_velocity_rad_s,
                "support_raw_max_velocity_rad_s": 0.0 if self._last_frame is None else self._last_frame.support_raw_max_velocity_rad_s,
                "phase": phase,
                "stage_phase": stage_phase,
                "reason": "" if self._last_frame is None else self._last_frame.reason,
                "motion_onset_confirm_count": 0 if self._last_frame is None else self._last_frame.motion_onset_confirm_count,
                "motion_onset_required_samples": None if self._last_frame is None else self._last_frame.motion_onset_required_samples,
                "breakaway_probe": None if self._last_frame is None else self._last_frame.breakaway_probe,
                "tracking_recovery": None if self._last_frame is None else self._last_frame.tracking_recovery,
                "release_terminal": False if self._last_frame is None else self._last_frame.release_terminal,
                "release_failed": False if self._last_frame is None else self._last_frame.release_failed,
                "release_restoring_base_damping": False if self._last_frame is None else self._last_frame.release_restoring_base_damping,
                "conditioned_command_delta_nm": None if self._last_frame is None else self._last_frame.conditioned_command_delta_nm,
                "conditioned_command_impulse_nm_sec": 0.0 if self._last_frame is None else self._last_frame.conditioned_command_impulse_nm_sec,
                "policy_authority": (
                    None
                    if self._last_frame is None
                    else self._last_frame.base_frame.decision.state.authority
                ),
                "owner_ready": self._ready,
                "operator_armed": self._operator_phase == "armed",
                "joint_name": self.limits.joint_name,
                "limits": {
                    **vars(self.limits),
                    "maximum_target_offset_rad": self.settings.maximum_target_offset_rad,
                    "cue_torque_nm": self.settings.staged_assist.cue_torque_nm,
                    "authority_rise_per_sec": self.settings.authority_rise_per_sec,
                    "breakaway_torque_nm": self.settings.staged_assist.breakaway_torque_nm,
                    "breakaway_rise_slew_nm_s": (
                        self.settings.staged_assist.breakaway_rise_slew_nm_s
                    ),
                    "breakaway_max_duration_sec": (
                        self.settings.staged_assist.breakaway_max_duration_sec
                    ),
                    "breakaway_max_impulse_nm_sec": (
                        self.settings.staged_assist.breakaway_max_impulse_nm_sec
                    ),
                    "minimum_breakaway_error_rad": self.settings.staged_assist.minimum_breakaway_error_rad,
                    "new_intent_delta_rad": self.settings.staged_assist.new_intent_delta_rad,
                    "stuck_max_velocity_rad_s": self.settings.staged_assist.stuck_max_velocity_rad_s,
                    "motion_onset_displacement_rad": (
                        self.settings.staged_assist.motion_onset_displacement_rad
                    ),
                    "motion_onset_confirm_displacement_rad": self.settings.staged_assist.motion_onset_confirm_displacement_rad,
                    "motion_onset_velocity_rad_s": (
                        self.settings.staged_assist.motion_onset_velocity_rad_s
                    ),
                    "motion_onset_confirm_samples": (
                        self.settings.staged_assist.motion_onset_confirm_samples
                    ),
                    "tracking_overlay_cap_nm": self.settings.staged_assist.tracking_overlay_cap_nm,
                },
                "target_offset_rad": self._target_offset_rad,
                "target_update_pending": self._target_update_pending,
                "desired_assist_nm": 0.0 if self._target_offset_rad is None else self._last_overlay[index],
                "effective_assist_nm": self._last_overlay[index],
                "fault": self._fault,
                "qpos_rad": self._qpos,
                "qvel_fd_rad_s": self._qvel,
                "anchor_rad": None if self._anchor_qpos is None else self._anchor_qpos[index],
                "displacement_rad": displacement,
                "support_drift_rad": self._support_drift(self._qpos) if self._qpos is not None else None,
                "last_observed_displacement_rad": (
                    None if not self._history else self._history[-1]["displacement_rad"]
                ),
                "last_observed_support_drift_rad": (
                    None if not self._history else self._history[-1]["support_drift_rad"]
                ),
                "feedback_age_sec": feedback_age,
                "last_dispatch": self._last_dispatch,
                "dispatch_age_sec": dispatch_age,
                "history": tuple(self._history),
                "events": tuple(self._events),
                "latest_policy_sequence": self._sequence or None,
                "breakaway_authorized": self._breakaway_authorized,
                "breakaway_eligible": self._breakaway_gate_reason() == "ready",
                "breakaway_gate_reason": self._breakaway_gate_reason(),
                "breakaway_impulse_nm_sec": (
                    0.0 if self._last_frame is None else self._last_frame.breakaway_impulse_nm_sec
                ),
                "breakaway_peak_overlay_nm": self._breakaway_peak_overlay_nm,
                "breakaway_peak_dispatched_nm": self._breakaway_peak_dispatched_nm,
                "breakaway_peak_conditioned_delta_nm": self._breakaway_peak_conditioned_delta_nm,
                "settle_integral_torque_nm": (
                    0.0 if self._last_frame is None else self._last_frame.settle_integral_torque_nm
                ),
            }


__all__ = ["B601StagedTorqueConsoleController", "B601StagedTorqueConsoleSettings"]
