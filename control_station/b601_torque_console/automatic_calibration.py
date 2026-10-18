"""Operator-authorized J3 acquisition through the existing single-owner ports.

This leaf sequences direct onset trials and the existing staged sweep adapter.
It never opens CAN, conditions a motor command, or installs compensation.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from robot_motion.rebot_b601_rs.calibration import (
    CalibrationCondition,
    finite,
    plan_local_sweeps,
    six,
)
from robot_motion.rebot_b601_rs.joint_motion import slew_driver_damping
from robot_motion.rebot_b601_rs.torque_experiment import (
    B601MotionOnsetCaptureConfig,
    B601OnsetProbeConfig,
    B601TorqueExperiment,
)

from .calibration_session import (
    B601AutomaticCalibrationSession,
    CalibrationBatchLimits,
    calibration_outcome,
    calibration_receipt_rejection,
)
from .staged_controller import (
    B601StagedTorqueConsoleController,
    B601StagedTorqueConsoleSettings,
)


def load_automatic_profile(path: Path) -> dict:
    """Load declarative commissioning settings without importing hardware."""
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("automatic profile must be a JSON object")
    if "calibration_profile" in values:
        if not {"calibration_profile", "cycle"} <= set(values) or set(values) - {"calibration_profile", "cycle", "workspace"}:
            raise ValueError("cycle profile requires calibration_profile, cycle and optional workspace")
        name = values["calibration_profile"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("cycle calibration profile must be a sibling filename")
        base = json.loads((path.parent/name).read_text(encoding="utf-8"))
        if not isinstance(base, dict) or "calibration_profile" in base or "cycle" in base:
            raise ValueError("cycle requires an ordinary calibration profile; nesting is unsupported")
        values = {**base, **{k: v for k, v in values.items() if k != "calibration_profile"}}
    if "cycle" in values and not isinstance(values["cycle"], dict):
        raise TypeError("cycle settings must be a JSON object, not a disabled or missing cycle")
    return values


@dataclass(frozen=True)
class AutomaticCalibrationConfig:
    onset_repeats_per_direction: int = 2
    cap_dwell_sec: float = 0.5
    prepare_timeout_sec: float = 5.0
    max_batch_sec: float = 150.0
    global_excursion_rad: float = 0.02
    sweep_excursion_rad: float = 0.008
    sweep_speeds_rad_s: tuple[float, float] = (0.0075, 0.015)
    max_sweep_leg_sec: float = 12.0
    max_sweep_command_exposure_nm_sec: float = 20.0
    onset_displacement_rad: float = 0.0012
    onset_confirm_samples: int = 3
    onset_velocity_rad_s: float = 0.002
    onset_settle_timeout_sec: float = 1.0
    probe_hold_sec: float = 0.12
    probe_max_trial_sec: float = 15.0
    probe_max_command_exposure_nm_sec: float = 6.0
    probe_max_displacement_rad: float = 0.003
    probe_observation_timeout_sec: float | None = None
    motion_position_resolution_rad: float | None = None
    motion_responsive_tracking_brake: bool = False
    motion_reference_pacing: bool = False
    motion_soft_withdrawal_confirmation: bool = False
    motion_arrival_margin_rad: float = 0.0
    motion_breakaway_probe: bool = False
    motion_tracking_recovery_cap_nm: float | None = None
    motion_tracking_recovery_gain_nm_per_rad: float = 20.0
    motion_tracking_recovery_velocity_feedback_nm_s_per_rad: float = 0.0
    onset_release_braking: bool = False
    onset_joint_velocity_limit_rad_s: float | None = None
    motion_acquisition_arrival_tolerance_rad: float | None = None
    max_sweep_leg_command_exposure_nm_sec: float | None = None
    motion_onset_handoff: str = "stop_and_zero"
    motion_plan: str = "model_sweeps"
    motion_braking_slew_nm_s: float | None = None
    motion_speed_governor: bool = False
    feedback_velocity_timebase: str = "snapshot"
    motion_arrival_stabilization: bool = False
    motion_tracking_arrival_tolerance_rad: float | None = None
    motion_velocity_headroom_ratio: float = 0.0
    motion_braking_timeout_sec: float | None = None
    onset_release_support_confirmation: bool = False
    motion_continuous_speed_regulation: bool = False
    motion_soft_velocity_rad_s: float | None = None

    def __post_init__(self) -> None:
        if (
            type(self.onset_repeats_per_direction) is not int
            or not 2 <= self.onset_repeats_per_direction <= 5
        ):
            raise ValueError("onset repeats per direction must be 2..5")
        for name, value in vars(self).items():
            if (
                name
                not in {
                    "onset_repeats_per_direction",
                    "onset_confirm_samples",
                    "sweep_speeds_rad_s",
                    "motion_position_resolution_rad",
                    "motion_responsive_tracking_brake",
                    "motion_reference_pacing",
                    "motion_soft_withdrawal_confirmation",
                    "probe_observation_timeout_sec",
                    "motion_arrival_margin_rad",
                    "motion_breakaway_probe",
                    "motion_tracking_recovery_cap_nm",
                    "motion_tracking_recovery_velocity_feedback_nm_s_per_rad",
                    "onset_release_braking",
                    "onset_joint_velocity_limit_rad_s",
                    "motion_acquisition_arrival_tolerance_rad",
                    "motion_tracking_arrival_tolerance_rad",
                    "motion_velocity_headroom_ratio",
                    "motion_braking_timeout_sec",
                    "onset_release_support_confirmation",
                    "motion_continuous_speed_regulation",
                    "motion_soft_velocity_rad_s",
                    "max_sweep_leg_command_exposure_nm_sec",
                    "motion_onset_handoff",
                    "motion_plan",
                    "motion_braking_slew_nm_s",
                    "motion_speed_governor",
                    "motion_arrival_stabilization",
                    "feedback_velocity_timebase",
                }
                and finite(value, name) <= 0
            ):
                raise ValueError(f"{name} must be positive")
        if len(self.sweep_speeds_rad_s) != 2:
            raise ValueError("two sweep reference speeds required")
        speeds = tuple(finite(v, "sweep speed") for v in self.sweep_speeds_rad_s)
        if not 0 < speeds[0] < speeds[1]:
            raise ValueError("sweep speeds must be positive and increasing")
        object.__setattr__(self, "sweep_speeds_rad_s", speeds)
        if self.max_batch_sec < self.max_sweep_leg_sec:
            raise ValueError("batch duration must cover one complete sweep leg")
        if self.motion_position_resolution_rad is not None and finite(self.motion_position_resolution_rad, "motion position resolution") <= 0:
            raise ValueError("motion position resolution must be positive")
        if type(self.motion_responsive_tracking_brake) is not bool:
            raise TypeError("motion responsive tracking brake must be boolean")
        if self.motion_responsive_tracking_brake and self.motion_position_resolution_rad is None:
            raise ValueError("responsive tracking brake requires a position resolution envelope")
        if type(self.motion_reference_pacing) is not bool:
            raise TypeError("motion reference pacing must be boolean")
        if self.motion_reference_pacing and (
            not self.motion_responsive_tracking_brake
            or self.motion_tracking_recovery_cap_nm is None
        ):
            raise ValueError("motion reference pacing requires explicit responsive tracking recovery")
        if type(self.motion_soft_withdrawal_confirmation) is not bool:
            raise TypeError("motion soft withdrawal confirmation must be boolean")
        if self.motion_soft_withdrawal_confirmation and not self.motion_reference_pacing:
            raise ValueError("motion soft withdrawal confirmation requires explicit pacing")
        margin = finite(self.motion_arrival_margin_rad, "motion arrival margin")
        if margin < 0 or (margin and self.motion_position_resolution_rad is None):
            raise ValueError("arrival margin requires nonnegative margin and a position resolution envelope")
        if type(self.motion_breakaway_probe) is not bool:
            raise TypeError("motion breakaway probe must be boolean")
        if self.motion_breakaway_probe and self.motion_position_resolution_rad is None:
            raise ValueError("held breakaway probe requires a position resolution envelope")
        if self.motion_tracking_recovery_cap_nm is not None:
            cap = finite(self.motion_tracking_recovery_cap_nm, "tracking recovery cap")
            if not 0 < cap <= 1.0 or not self.motion_breakaway_probe or not self.motion_responsive_tracking_brake:
                raise ValueError("tracking recovery cap must be within 0..1 Nm in the explicit held/responsive candidate")
            object.__setattr__(self, "motion_tracking_recovery_cap_nm", cap)
        feedback = finite(self.motion_tracking_recovery_velocity_feedback_nm_s_per_rad, "tracking recovery velocity feedback")
        if feedback < 0 or (feedback and self.motion_tracking_recovery_cap_nm is None):
            raise ValueError("velocity feedback requires nonnegative gain and an explicit recovery cap")
        object.__setattr__(self, "motion_tracking_recovery_velocity_feedback_nm_s_per_rad", feedback)
        if type(self.onset_release_braking) is not bool:
            raise TypeError("onset release braking must be boolean")
        if type(self.onset_release_support_confirmation) is not bool:
            raise TypeError("onset release support confirmation must be boolean")
        if self.onset_release_support_confirmation and (not self.onset_release_braking or self.onset_joint_velocity_limit_rad_s is None):
            raise ValueError("release support confirmation requires explicit onset release braking and velocity limits")
        if self.onset_joint_velocity_limit_rad_s is not None:
            limit = finite(self.onset_joint_velocity_limit_rad_s, "onset joint velocity limit")
            if not self.onset_release_braking or not 0.05 <= limit <= 0.15:
                raise ValueError("onset joint velocity limit requires release braking and 0.05..0.15 rad/s")
            object.__setattr__(self, "onset_joint_velocity_limit_rad_s", limit)
        if self.motion_acquisition_arrival_tolerance_rad is not None:
            tolerance = finite(self.motion_acquisition_arrival_tolerance_rad, "acquisition tolerance")
            if not 0 < tolerance <= self.sweep_excursion_rad / 2 + 1e-12:
                raise ValueError("acquisition tolerance must be positive and no larger than half the sweep excursion")
            object.__setattr__(self, "motion_acquisition_arrival_tolerance_rad", tolerance)
        if self.motion_tracking_arrival_tolerance_rad is not None:
            tolerance = finite(self.motion_tracking_arrival_tolerance_rad, "tracking arrival tolerance")
            if tolerance <= 0:
                raise ValueError("tracking arrival tolerance must be positive")
            object.__setattr__(self, "motion_tracking_arrival_tolerance_rad", tolerance)
        headroom = finite(self.motion_velocity_headroom_ratio, "motion velocity headroom")
        if headroom < 0 or (headroom and not self.motion_reference_pacing):
            raise ValueError("motion velocity headroom requires nonnegative margin and explicit pacing")
        if self.motion_braking_timeout_sec is not None:
            timeout = finite(self.motion_braking_timeout_sec, "motion braking timeout")
            if timeout <= 0 or self.motion_onset_handoff != "track":
                raise ValueError("motion braking timeout requires a positive value and tracking handoff")
        if self.max_sweep_leg_command_exposure_nm_sec is not None:
            limit = finite(self.max_sweep_leg_command_exposure_nm_sec, "sweep leg exposure")
            if not 0 < limit <= self.max_sweep_command_exposure_nm_sec:
                raise ValueError("sweep leg exposure must be positive and inside the total sweep budget")
            object.__setattr__(self, "max_sweep_leg_command_exposure_nm_sec", limit)
        if self.motion_plan not in {"model_sweeps", "motion_check", "local_identification"}:
            raise ValueError("unknown automatic motion plan")
        if self.motion_onset_handoff not in {"stop_and_zero", "track"}:
            raise ValueError("motion onset handoff must be stop_and_zero or track")
        if self.feedback_velocity_timebase not in {"snapshot", "joint_receive"}:
            raise ValueError("unknown feedback velocity timebase")
        if self.feedback_velocity_timebase == "joint_receive" and not self.motion_speed_governor:
            raise ValueError("joint receive velocity requires explicit governed tracking")
        if self.motion_onset_handoff == "track" and (
            not self.motion_breakaway_probe or not self.motion_reference_pacing
        ):
            raise ValueError("tracking handoff requires the held probe and paced tracking recovery")
        if self.motion_braking_slew_nm_s is not None:
            rate = finite(self.motion_braking_slew_nm_s, "motion braking slew")
            if rate <= 0 or self.motion_onset_handoff != "track":
                raise ValueError("motion braking slew requires a positive rate and explicit tracking handoff")
            object.__setattr__(self, "motion_braking_slew_nm_s", rate)
        if type(self.motion_speed_governor) is not bool:
            raise TypeError("motion speed governor must be boolean")
        if self.motion_speed_governor and self.motion_onset_handoff != "track":
            raise ValueError("motion speed governor requires explicit tracking handoff")
        if type(self.motion_arrival_stabilization) is not bool:
            raise TypeError("motion arrival stabilization must be boolean")
        if self.motion_arrival_stabilization and not self.motion_speed_governor:
            raise ValueError("motion arrival stabilization requires governed tracking")
        if type(self.motion_continuous_speed_regulation) is not bool:
            raise TypeError("motion continuous speed regulation must be boolean")
        if self.motion_continuous_speed_regulation and not (
            self.motion_plan == "local_identification"
            and self.motion_speed_governor and self.motion_reference_pacing
            and self.motion_onset_handoff == "track"
            and self.motion_tracking_recovery_cap_nm is not None
            and self.motion_tracking_recovery_velocity_feedback_nm_s_per_rad > 0
        ):
            raise ValueError("continuous acquisition requires explicit local governed tracking recovery with velocity feedback")
        if self.motion_soft_velocity_rad_s is not None:
            speed = finite(self.motion_soft_velocity_rad_s, "motion soft velocity")
            if speed <= 0 or not self.motion_continuous_speed_regulation:
                raise ValueError("motion soft velocity override requires positive explicit continuous acquisition")


class B601AutomaticCalibrationController:
    """Only the owner calls step/record_dispatch; HTTP publishes intent."""

    controller_mode = "automatic_calibration"
    hardware_execution_allowed = True

    def __init__(
        self,
        settings: B601StagedTorqueConsoleSettings,
        config: AutomaticCalibrationConfig,
    ) -> None:
        if config.motion_position_resolution_rad is not None:
            motion = settings.staged_assist.motion_control
            if motion is None:
                raise ValueError("automatic quantized motion requires a motion profile")
            settings = replace(settings, staged_assist=replace(
                settings.staged_assist, motion_control=replace(
                    motion, position_resolution_rad=config.motion_position_resolution_rad,
                    responsive_tracking_brake=config.motion_responsive_tracking_brake,
                    arrival_margin_rad=config.motion_arrival_margin_rad,
                ),
            ))
        self.settings, self.config, self.limits = settings, config, settings.limits
        if config.motion_breakaway_probe:
            staged = settings.staged_assist
            settings = replace(settings, staged_assist=replace(staged,
                breakaway_probe=B601OnsetProbeConfig(
                    hold_sec=config.probe_hold_sec,
                    max_trial_sec=staged.breakaway_max_duration_sec,
                    max_command_exposure_nm_sec=staged.breakaway_max_impulse_nm_sec,
                    max_displacement_rad=config.probe_max_displacement_rad,
                    observation_timeout_sec=config.probe_observation_timeout_sec,
                )))
            self.settings = settings
        if config.motion_tracking_recovery_cap_nm is not None:
            if config.motion_tracking_recovery_cap_nm > self.limits.max_abs_assist_nm:
                raise ValueError("tracking recovery cap exceeds the existing total assist envelope")
            settings = replace(settings, staged_assist=replace(settings.staged_assist,
                tracking_overlay_cap_nm=config.motion_tracking_recovery_cap_nm,
                tracking_recovery_gain_nm_per_rad=config.motion_tracking_recovery_gain_nm_per_rad,
                tracking_recovery_velocity_feedback_nm_s_per_rad=config.motion_tracking_recovery_velocity_feedback_nm_s_per_rad))
            self.settings = settings
        if config.motion_tracking_arrival_tolerance_rad is not None:
            if config.motion_tracking_arrival_tolerance_rad > settings.maximum_target_offset_rad / 4.0:
                raise ValueError("tracking arrival tolerance exceeds the existing target envelope fraction")
            settings = replace(settings, staged_assist=replace(settings.staged_assist,
                settle_error_rad=config.motion_tracking_arrival_tolerance_rad))
            self.settings = settings
        if config.motion_reference_pacing:
            motion = settings.staged_assist.motion_control
            settings = replace(settings, staged_assist=replace(
                settings.staged_assist, motion_control=replace(motion, reference_pacing=True),
                breakaway_soft_withdrawal_confirmation=config.motion_soft_withdrawal_confirmation,
            ))
            self.settings = settings
        if config.motion_onset_handoff == "track":
            settings = replace(settings, staged_assist=replace(
                settings.staged_assist, onset_handoff="track"))
            self.settings = settings
        motion = settings.staged_assist.motion_control
        if (config.motion_braking_slew_nm_s is not None or config.motion_speed_governor
            or config.motion_braking_timeout_sec is not None or config.motion_velocity_headroom_ratio
            or config.motion_continuous_speed_regulation):
            if motion is None:
                raise ValueError("motion dynamic overrides require a motion profile")
            overrides = {}
            if config.motion_braking_timeout_sec is not None:
                overrides["braking_timeout_sec"] = config.motion_braking_timeout_sec
            if config.motion_velocity_headroom_ratio:
                overrides["velocity_headroom_ratio"] = config.motion_velocity_headroom_ratio
            if config.motion_braking_slew_nm_s is not None:
                overrides["braking_slew_nm_s"] = config.motion_braking_slew_nm_s
            if config.motion_speed_governor:
                overrides.update(speed_governor=True,
                    arrival_stabilization=config.motion_arrival_stabilization)
            if config.motion_continuous_speed_regulation:
                if config.motion_soft_velocity_rad_s is not None:
                    if not motion.soft_velocity_rad_s <= config.motion_soft_velocity_rad_s < self.limits.max_joint_velocity_rad_s:
                        raise ValueError("acquisition soft velocity must retain the original soft floor and stay below the hard guard")
                    overrides["soft_velocity_rad_s"] = config.motion_soft_velocity_rad_s
                # Reuse continuous output regulation without raising the
                # resolved absolute soft stop ceiling a second time.
                overrides["continuous_speed_limit_rad_s"] = overrides.get("soft_velocity_rad_s", motion.soft_velocity_rad_s)
            settings = replace(settings, staged_assist=replace(
                settings.staged_assist, motion_control=replace(motion, **overrides)))
            self.settings = settings
            motion = settings.staged_assist.motion_control
        if (
            self.limits.joint_name != "joint3"
            or motion is None
            or settings.staged_assist.motion_onset_confirm_displacement_rad is None
        ):
            raise ValueError(
                "automatic calibration requires the confirmed J3 motion profile"
            )
        if config.onset_release_braking and not 0 < motion.drive_driver_kd_nm_s_per_rad[2] <= 0.4:
            raise ValueError("onset release braking requires the existing J3 drive Kd within 0..0.4")
        if (
            config.motion_acquisition_arrival_tolerance_rad is not None
            and config.motion_acquisition_arrival_tolerance_rad + 1e-12
            < settings.staged_assist.settle_error_rad
        ):
            raise ValueError("acquisition tolerance cannot be smaller than the tracking tolerance")
        if (
            config.motion_acquisition_arrival_tolerance_rad is not None
            and 2 * config.sweep_excursion_rad
            + config.motion_acquisition_arrival_tolerance_rad
            > settings.maximum_target_offset_rad + 1e-12
        ):
            raise ValueError("acquisition tolerance must leave room for the next leg's actual anchor")
        if (
            config.max_sweep_leg_command_exposure_nm_sec is not None
            and config.max_sweep_leg_command_exposure_nm_sec
            <= 6 * self.limits.max_abs_assist_nm * motion.max_control_gap_sec
        ):
            raise ValueError("sweep leg exposure must allow a complete withdrawal reserve")
        if (
            config.global_excursion_rad > self.limits.max_excursion_rad
            or config.sweep_excursion_rad * 2 > settings.maximum_target_offset_rad
        ):
            raise ValueError(
                "automatic acquisition must fit the existing excursion envelope"
            )
        if max(config.sweep_speeds_rad_s) > motion.max_reference_velocity_rad_s:
            raise ValueError("sweep speed exceeds the motion profile")
        support_peak = max(
            v for i, v in enumerate(motion.drive_driver_kd_nm_s_per_rad) if i != 2
        )
        if (
            support_peak / motion.drive_damping_rise_per_sec
            + motion.velocity_window_sec
            + settings.staged_assist.settle_dwell_sec
            + settings.nominal_control_period_sec
            >= config.prepare_timeout_sec
        ):
            raise ValueError(
                "support damping cannot ramp and settle within the preparation timeout"
            )
        cap = self.limits.max_abs_assist_nm
        optimistic_time = cap / self.limits.rise_slew_nm_s + config.cap_dwell_sec
        optimistic_exposure = (
            cap * cap / (2 * self.limits.rise_slew_nm_s) + cap * config.cap_dwell_sec
        )
        if (
            optimistic_time + config.onset_settle_timeout_sec
            >= config.probe_max_trial_sec
            or optimistic_exposure + cap * cap / (2 * self.limits.release_slew_nm_s)
            >= config.probe_max_command_exposure_nm_sec
        ):
            raise ValueError(
                "scan cap is unreachable within the configured trial/exposure budget"
            )
        self._direct = B601TorqueExperiment(
            self.limits,
            quantized_velocity_debug=True,
            onset_capture=B601MotionOnsetCaptureConfig(
                displacement_rad=config.onset_displacement_rad,
                confirm_samples=config.onset_confirm_samples,
                window_velocity_rad_s=config.onset_velocity_rad_s,
                settle_timeout_sec=config.onset_settle_timeout_sec,
                raw_joint_velocity_limit_rad_s=config.onset_joint_velocity_limit_rad_s,
                release_support_confirmation=config.onset_release_support_confirmation,
            ),
            onset_probe=B601OnsetProbeConfig(
                hold_sec=config.probe_hold_sec,
                max_trial_sec=config.probe_max_trial_sec,
                max_command_exposure_nm_sec=config.probe_max_command_exposure_nm_sec,
                max_displacement_rad=config.probe_max_displacement_rad,
                observation_timeout_sec=config.probe_observation_timeout_sec,
            ),
        )
        self._staged = B601StagedTorqueConsoleController(settings)
        self._sweep: B601AutomaticCalibrationSession | None = None
        self._condition: CalibrationCondition | None = None
        self._position_limits: tuple[tuple[float, float], ...] | None = None
        self._lock = threading.RLock()
        self._state, self._reason = "idle", "waiting_for_operator"
        self._human: bool | None = None
        self._permit: float | None = None
        self._start_requested = False
        self._cancel_requested: str | None = None
        self._started: float | None = None
        self._phase_started: float | None = None
        self._anchor: tuple[float, ...] | None = None
        self._batch_id: str | None = None
        self._onset_index = 0
        self._directions = tuple(
            d
            for i in range(config.onset_repeats_per_direction)
            for d in ((-1, 1) if i % 2 == 0 else (1, -1))
        )
        self._onset_trials: list[dict[str, Any]] = []
        self._cap_since: float | None = None
        self._scan_end_reason: str | None = None
        self._scan_started: float | None = None
        self._last_feedback: float | None = None
        self._last_completed: float | None = None
        self._last_qpos: tuple[float, ...] | None = None
        self._last_receipt: dict[str, Any] | None = None
        # Keep J3 transparent during onset characterization. Support axes use
        # the existing v2 profile, never an independent GUI gain source.
        self._support_target_kd = tuple(
            0.0 if i == 2 else v
            for i, v in enumerate(motion.drive_driver_kd_nm_s_per_rad)
        )
        self._release_target_kd = (
            motion.drive_driver_kd_nm_s_per_rad
            if config.onset_release_braking else self._support_target_kd
        )
        self._sent_kd = (0.0,) * 6
        self._requested_kd = (0.0,) * 6
        self._kd_plateau_since: float | None = None
        self._zero_plateau_since: float | None = None
        self._last_sent_zero = False
        self._onset_first_zero: float | None = None
        self._scan_withdrawal_reason: str | None = None
        self._base_restore_completed: float | None = None
        self._pending_sequence: int | None = None
        self._dispatch_rejection: dict[str, Any] | None = None
        self._scan_min: tuple[float, ...] | None = None
        self._scan_max: tuple[float, ...] | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=128)
        self._records: deque[dict[str, Any]] = deque(maxlen=32)
        self._motion_started = False
        self._terminal_recorded = False
        self._ready = False
        self._onset_prepare_deadline: float | None = None
        self._onset_baseline_since: float | None = None
        self._onset_baseline_min: tuple[float, ...] | None = None
        self._onset_baseline_max: tuple[float, ...] | None = None
        self._onset_baseline_reason = "not_preparing"

    @property
    def _motion_active(self) -> bool:
        return self._state in {"motion_prepare", "motion"} or self._motion_started

    @property
    def _active(self) -> bool:
        return self._state not in {"idle", "complete", "cancelled", "failed"}

    def configure_context(
        self,
        condition: CalibrationCondition,
        position_limits_rad: tuple[tuple[float, float], ...],
    ) -> None:
        with self._lock:
            if self._condition is not None or self._state != "idle":
                raise ValueError(
                    "calibration context can only be configured once before acquisition"
                )
            if condition.driver_kd_nm_s_per_rad != (0.0,) * 6:
                raise ValueError(
                    "onset acquisition requires the established zero base driver damping"
                )
            if len(position_limits_rad) != 6 or any(
                not finite(lo, "lower") < finite(hi, "upper")
                for lo, hi in position_limits_rad
            ):
                raise ValueError("six finite owner position intervals required")
            self._condition = condition
            self._position_limits = tuple(tuple(v) for v in position_limits_rad)

    def owner_ready(self) -> None:
        with self._lock:
            self._ready = True
            self._direct.owner_ready()
            self._staged.owner_ready()

    def set_command_timeout_sec(self, timeout_sec: float) -> None:
        """Set the GUI lease before starting a calibration batch."""
        timeout = finite(timeout_sec, "command_timeout_sec")
        if timeout <= 0:
            raise ValueError("command_timeout_sec must be positive")
        with self._lock:
            if self._state != "idle" or self._start_requested:
                raise ValueError("command timeout can only change before calibration starts")
            self.limits = replace(self.limits, command_timeout_sec=timeout)
            self.settings = replace(self.settings, limits=self.limits)
            self._direct.limits = self.limits
            self._staged.set_command_timeout_sec(timeout)

    def set_human_intervention(self, human: bool) -> None:
        if type(human) is not bool:
            raise TypeError("human intervention must be boolean")
        with self._lock:
            self._human = human
            self._direct.set_human_intervention(human)
            if human:
                self._start_requested = False
                if self._active:
                    self._cancel_requested = "human_intervention"

    def submit(self, action: str, *, now_sec: float, **fields: Any) -> None:
        now = finite(now_sec, "now_sec")
        with self._lock:
            withdraw_source = fields.pop("withdraw_source", "operator_button")
            if action == "withdraw" and withdraw_source not in {
                "operator_button", "page_hidden", "page_unload", "owner_cancel"
            }:
                raise ValueError("unsupported withdrawal source")
            if fields:
                raise ValueError(
                    "automatic calibration has no manual torque, target or joint fields"
                )
            if action == "arm":
                if (
                    self._state != "idle"
                    or self._start_requested
                    or not self._ready
                    or self._condition is None
                    or self._human is not False
                ):
                    raise ValueError(
                        "start requires an idle active owner and declared no human intervention"
                    )
                self._start_requested = True
                self._permit = now
            elif action == "heartbeat":
                if not self._active and not self._start_requested:
                    raise ValueError("heartbeat requires an active acquisition")
                self._permit = now
            elif action in {"withdraw", "zero"}:
                self._start_requested = False
                if self._active:
                    reason = "operator_zero" if action == "zero" else {
                        "operator_button": "operator_withdraw",
                        "page_hidden": "page_hidden",
                        "page_unload": "page_unload",
                        "owner_cancel": "owner_cancel",
                    }[withdraw_source]
                    if self._cancel_requested is None:
                        self._cancel_requested = reason
            else:
                raise ValueError("unsupported automatic calibration action")
            if action != "heartbeat":
                event = {"action": action, "time_sec": now}
                if action == "withdraw":
                    event["source"] = withdraw_source
                self._events.append(event)

    def _transition(self, state: str, reason: str, now: float) -> None:
        self._state, self._reason, self._phase_started = state, reason, now
        if state == "onset_prepare":
            # One deadline covers preparation, arming and deferred first force.
            # Baseline chatter cannot renew it or spend a new trial budget.
            if self._onset_prepare_deadline is None:
                self._onset_prepare_deadline = now + self.config.prepare_timeout_sec
            self._onset_baseline_since = None
            self._onset_baseline_min = self._onset_baseline_max = None
            self._onset_baseline_reason = "continuous_baseline_dwell"
        self._events.append(
            {
                "action": "calibration_phase",
                "time_sec": now,
                "state": state,
                "reason": reason,
            }
        )

    def _cancel(self, now: float, reason: str) -> None:
        if not self._active:
            return
        self._cancel_requested = reason
        if self._motion_active and self._sweep is not None:
            if (
                self._sweep.owner_snapshot()["state"] == "idle"
                and self._state != "motion_cancel"
            ):
                self._staged.submit("withdraw", now_sec=now)
                self._transition("motion_cancel", reason, now)
            else:
                self._sweep.cancel(now_sec=now, reason=reason)
        elif self._state not in {"onset_release", "onset_restore"}:
            self._direct.submit("withdraw", now_sec=now)
            self._transition("onset_release", reason, now)

    def _onset_start_pending(self, view: dict[str, Any]) -> bool:
        trial = None if view["capture"] is None else view["capture"]["trial"]
        return self._state in {"onset_prepare", "onset_arm", "onset_scan"} and (
            trial is None or "completed_at_sec" in trial
        )

    def _observe_onset_baseline(self, now: float, view: dict[str, Any]) -> None:
        """Admit first force only after continuous acknowledged zero/stillness.

        The direct controller remains the same-tick gate owner. This dwell
        sequences the batch and never overrides its raw or position checks.
        """
        reason = "trial_started"
        if self._onset_start_pending(view):
            reason = view["prepare_gate"]
            if reason == "ready" and not self._damping_settled(now, self._support_target_kd):
                reason = "support_damping_or_zero_not_settled"
        if reason != "ready":
            self._onset_baseline_since = None
            self._onset_baseline_min = self._onset_baseline_max = None
            self._onset_baseline_reason = reason
            return
        q = view["qpos_rad"]
        if self._onset_baseline_since is None:
            self._onset_baseline_since = now
            self._onset_baseline_min = self._onset_baseline_max = q
        else:
            self._onset_baseline_min = tuple(min(a, b) for a, b in zip(self._onset_baseline_min, q))
            self._onset_baseline_max = tuple(max(a, b) for a, b in zip(self._onset_baseline_max, q))
            if max(hi-lo for lo, hi in zip(self._onset_baseline_min, self._onset_baseline_max)) > view["position_count_bound_rad"]:
                self._onset_baseline_since = now
                self._onset_baseline_min = self._onset_baseline_max = q
        # The first clock is an actual dispatch completion; later evidence is
        # feedback AFTER that zero, not time spent computing/sending a packet.
        self._onset_baseline_reason = (
            "ready" if self._last_feedback - self._onset_baseline_since >= self.settings.staged_assist.settle_dwell_sec
            else "continuous_baseline_dwell"
        )

    def _onset_start_permission(self, q: tuple[float, ...]) -> bool:
        """Include THIS admitted feedback before creating a first force frame.

        Only the prior interval's native zero receipts seed the dwell. Its
        range must also include the current feedback BEFORE dispatch, not be
        checked after a newly created trial has bypassed preparation.
        """
        if self._onset_baseline_reason != "ready":
            return False
        view = self._direct.owner_snapshot(observed_at_sec=self._last_feedback)
        span = max(
            max(hi, value) - min(lo, value)
            for lo, hi, value in zip(self._onset_baseline_min, self._onset_baseline_max, q)
        )
        if span > view["position_count_bound_rad"]:
            self._onset_baseline_since = None
            self._onset_baseline_min = self._onset_baseline_max = None
            self._onset_baseline_reason = "continuous_baseline_position_changed"
            return False
        return True

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
            finite(now_sec, "now_sec"),
            finite(control_now_sec, "control_now_sec"),
        )
        q = six(qpos_rad, "qpos_rad")
        with self._lock:
            dt = (
                self.settings.nominal_control_period_sec
                if self._last_feedback is None
                else now - self._last_feedback
            )
            if (
                not now
                <= clock
                <= now + self.settings.staged_assist.motion_control.max_control_gap_sec
            ):
                raise ValueError(
                    "control observation clock does not follow fresh feedback"
                )
            if self._start_requested:
                # Transfer/GUI idle time did not execute this child's ramp.
                # Give its first packet at most one nominal control interval;
                # subsequent increments remain anchored to acknowledged Kd.
                dt = min(dt, self.settings.nominal_control_period_sec)
                self._start_requested = False
                self._started = clock
                self._batch_id = f"automatic-{clock:.9f}"
                self._anchor = q
                self._transition("onset_prepare", "batch_authorized", clock)
                started = self.result_record()
                started["kind"] = "automatic_calibration_started"
                self._records.append(started)
            if self._active:
                reason = self._cancel_requested or (
                    "previous_dispatch_not_confirmed"
                    if self._pending_sequence is not None
                    else None
                )
                if self._human is not False:
                    reason = "human_intervention_unknown"
                elif reason is None and (
                    self._permit is None
                    or not 0 <= clock - self._permit <= self.limits.command_timeout_sec
                ):
                    reason = "gui_permit_expired"
                elif reason is None and clock - self._started > self.config.max_batch_sec:
                    reason = "batch_timeout"
                elif abs(q[2] - self._anchor[2]) > self.config.global_excursion_rad:
                    reason = "global_batch_excursion"
                elif any(
                    abs(q[i] - self._anchor[i]) > self.limits.max_support_drift_rad
                    for i in range(6)
                    if i != 2
                ):
                    reason = "global_support_drift"
                elif any(
                    not lo <= position <= hi
                    for position, (lo, hi) in zip(q, self._position_limits)
                ):
                    reason = "outside_position_envelope"
                elif (
                    (
                        not self._motion_active
                        and self._onset_start_pending(self._direct.owner_snapshot(observed_at_sec=clock))
                        and self._onset_prepare_deadline is not None
                        and clock > self._onset_prepare_deadline
                    )
                    or (
                        self._state == "motion_prepare"
                        and clock - self._phase_started > self.config.prepare_timeout_sec
                    )
                ):
                    reason = "preparation_timeout"
                if reason:
                    self._cancel(clock, reason)
            self._last_feedback, self._last_qpos = now, q
            self._pending_sequence = feedback_sequence
            if self._motion_active:
                overlay, audit = self._sweep.step(
                    now_sec=now,
                    feedback_sequence=feedback_sequence,
                    qpos_rad=q,
                    qvel_rad_s=qvel_rad_s,
                    condition=self._condition,
                    human_intervention=self._human,
                    operator_permit_timestamp_sec=self._permit,
                    operator_observed_at_sec=clock,
                )
            else:
                if (
                    self._state in {"onset_arm", "onset_scan"}
                    and self._direct.owner_snapshot(observed_at_sec=clock)["phase"]
                    == "armed"
                ):
                    self._direct.submit("heartbeat", now_sec=now)
                overlay, audit = self._direct.step(
                    now_sec=now, qpos_rad=q, qvel_rad_s=qvel_rad_s,
                    onset_start_permitted=(
                        self._onset_start_permission(q)
                        if self._state in {"onset_prepare", "onset_arm", "onset_scan"} else None
                    ),
                )
                view = self._direct.owner_snapshot(observed_at_sec=clock)
                if (
                    view["phase"] in {"onset_release", "velocity_check"}
                    and view["capture"] is not None
                    and view["capture"]["trial"] is not None
                    and self._scan_withdrawal_reason is None
                ):
                    self._scan_withdrawal_reason = view["capture"]["trial"].get(
                        "outcome"
                    )
                if (
                    view["capture"] is not None
                    and view["capture"]["first_zero_completed_at_sec"] is not None
                ):
                    self._onset_first_zero = view["capture"][
                        "first_zero_completed_at_sec"
                    ]
                target_kd = self._support_target_kd if self._active else (0.0,) * 6
                if (
                    self.config.onset_release_braking
                    and view["phase"] in {"onset_release", "velocity_check"}
                    and view["capture"] is not None
                    and view["capture"]["trial"] is not None
                    and "completed_at_sec" not in view["capture"]["trial"]
                ):
                    # Remove propulsion first; add only the existing bounded
                    # motor-local damping. No spring or old position target.
                    target_kd = (
                        self._release_target_kd
                        if self._active and self._human is False and self._cancel_requested is None
                        else self._sent_kd
                    )
                if self._state == "onset_restore":
                    # Do not discard passive damping while force tails or
                    # unconfirmed motion remain. Restoration owns no assist.
                    target_kd = (
                        (0.0,) * 6
                        if view["phase"] == "inactive"
                        and view["prepare_gate"] == "ready"
                        and self._last_sent_zero
                        else self._sent_kd
                    )
                    if clock - self._phase_started > self.config.prepare_timeout_sec:
                        audit = {
                            **audit,
                            "phase": "fault",
                            "fault": "support_damping_restore_timeout",
                        }
                self._requested_kd = slew_driver_damping(
                    self._sent_kd,
                    target_kd,
                    rate_per_sec=self.settings.staged_assist.motion_control.drive_damping_rise_per_sec,
                    dt_sec=dt,
                )
                if self._state in {"onset_scan", "onset_release"}:
                    self._scan_min = (
                        q
                        if self._scan_min is None
                        else tuple(min(a, b) for a, b in zip(self._scan_min, q))
                    )
                    self._scan_max = (
                        q
                        if self._scan_max is None
                        else tuple(max(a, b) for a, b in zip(self._scan_max, q))
                    )
            if audit["phase"] == "fault":
                self._transition("failed", str(audit["fault"]), clock)
            return overlay, {**audit, "automatic_calibration": self._small_status()}

    def validate_final_command(self, **values: Any) -> None:
        if not self._motion_active:
            self._direct.validate_final_command(**values)

    def active_driver_damping(
        self,
        base_kd: tuple[float, ...],
        *,
        final_command_delta_nm: tuple[float, ...] | None = None,
    ) -> tuple[float, ...]:
        with self._lock:
            if self._motion_active:
                return self._staged.active_driver_damping(
                    base_kd, final_command_delta_nm=final_command_delta_nm
                )
            drive = self._requested_kd
            if (
                self._state == "onset_restore"
                or self.config.onset_release_braking and drive[2] > self._sent_kd[2]
            ) and (
                final_command_delta_nm is None
                or any(
                    abs(v) > 1e-9
                    for v in six(final_command_delta_nm, "final command delta")
                )
            ):
                # A prior zero receipt cannot establish this packet is zero.
                # Do not overlap a new brake gain with a propulsion tail.
                drive = self._sent_kd
            return tuple(
                max(base, request) for base, request in zip(base_kd, drive, strict=True)
            )

    def _reject_dispatch(
        self, reason: str, *, stage: str, observed_at_sec: float,
        native_receipt: dict[str, Any] | None = None, supplied: dict[str, Any],
    ) -> None:
        # Fault-only serialization: no history copying or JSON work on the
        # successful control path. Preserve malformed evidence as bounded text.
        def detached(value: Any) -> Any:
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError):
                return repr(value)[:4096]
            return copy.deepcopy(value)

        if self._dispatch_rejection is None:
            self._dispatch_rejection = {
                "kind": "automatic_calibration_dispatch_rejected",
                "recorded_monotonic_sec": observed_at_sec,
                "reason": reason, "stage": stage, "state": self._state,
                "native_send_attempted": stage == "after_dispatch",
                "expected_source_sequence": self._pending_sequence,
                "previous_receipt_sequence": None if self._last_receipt is None else self._last_receipt["sequence"],
                "feedback_timestamp_sec": self._last_feedback,
                "max_control_gap_sec": self.settings.staged_assist.motion_control.max_control_gap_sec,
                "feedback_to_observation_sec": None if self._last_feedback is None else observed_at_sec-self._last_feedback,
                "native_receipt": detached(native_receipt),
                "supplied": detached(supplied),
                "qpos_rad": self._last_qpos,
            }
            self._records.append(copy.deepcopy(self._dispatch_rejection))
        self.recording_failed(reason)
        raise ValueError(f"automatic acquisition dispatch rejected: {reason}")

    def validate_dispatch_start(
        self, *, feedback_sequence: int, feedback_timestamp_sec: float, now_sec: float | None = None,
    ) -> None:
        """The existing feedback deadline also applies immediately before CAN."""
        with self._lock:
            # Include contention for this lock in the live admission age.
            # Explicit timestamps are only needed by deterministic offline callers.
            now = finite(time.monotonic() if now_sec is None else now_sec, "now_sec")
            reason = None
            if self._pending_sequence is None:
                reason = "no_pending_dispatch"
            elif feedback_sequence != self._pending_sequence:
                reason = "dispatch_source_sequence_mismatch"
            elif feedback_timestamp_sec != self._last_feedback:
                reason = "dispatch_feedback_timestamp_mismatch"
            elif not self._last_feedback <= now:
                reason = "dispatch_request_precedes_feedback"
            elif now > self._last_feedback + self.settings.staged_assist.motion_control.max_control_gap_sec:
                reason = "dispatch_start_deadline_exceeded"
            if reason:
                self._reject_dispatch(reason, stage="before_dispatch", observed_at_sec=now,
                    supplied={"feedback_sequence": feedback_sequence, "feedback_timestamp_sec": feedback_timestamp_sec})

    def record_dispatch(self, *, native_receipt: dict[str, Any], **values: Any) -> None:
        with self._lock:
            reason = (
                "no_pending_dispatch" if self._pending_sequence is None
                else calibration_receipt_rejection(
                    native_receipt,
                    self._pending_sequence,
                    None
                    if self._last_receipt is None
                    else self._last_receipt["sequence"],
                )
            )
            try:
                now = finite(values.get("completed_at_sec"), "completed_at_sec")
            except (ValueError, TypeError):
                # Incomplete native sends may have no completion timestamp.
                # Keep their receipt too; never acknowledge an invented time.
                now = time.monotonic()
                reason = reason or "invalid_dispatch_timestamp"
            if reason is None:
                try:
                    requested_at = finite(native_receipt.get("requested_at_sec"), "requested_at_sec")
                    completed_at = finite(native_receipt.get("completed_at_sec"), "completed_at_sec")
                except (ValueError, TypeError):
                    reason = "invalid_dispatch_timestamp"
                else:
                    if native_receipt["sequence"] != values.get("receipt_sequence"):
                        reason = "dispatch_receipt_sequence_mismatch"
                    elif completed_at != now:
                        reason = "dispatch_completion_timestamp_mismatch"
                    elif values.get("feedback_timestamp_sec") != self._last_feedback:
                        reason = "dispatch_feedback_timestamp_mismatch"
                    elif not self._last_feedback <= requested_at <= now:
                        reason = "dispatch_request_timestamp_order"
                    elif now > self._last_feedback + self.settings.staged_assist.motion_control.max_control_gap_sec:
                        reason = "dispatch_completion_deadline_exceeded"
            if reason:
                self._reject_dispatch(reason, stage="after_dispatch", observed_at_sec=now,
                    native_receipt=native_receipt, supplied=values)
            self._pending_sequence = None
            if self._motion_active:
                self._sweep.record_dispatch(
                    receipt=native_receipt,
                    joint_commands=values["joint_commands"],
                    torque_rate_nm_s=values["torque_rate_nm_s"],
                )
                self._last_completed, self._last_receipt = (
                    now,
                    copy.deepcopy(native_receipt),
                )
                if self._state == "motion_cancel":
                    view = self._staged.owner_snapshot()
                    commands = values["joint_commands"]
                    if (
                        view["operator_phase"] == "inactive"
                        and view["prepare_gate"] == "ready"
                        and all(
                            abs(t - g) <= 1e-9
                            for t, g in zip(
                                commands["total_nm"], commands["gravity_nm"]
                            )
                        )
                        and commands["driver_kd_nm_s_per_rad"]
                        == self._condition.driver_kd_nm_s_per_rad
                    ):
                        self._transition("cancelled", self._cancel_requested, now)
                elif self._state == "motion_prepare":
                    view = self._staged.owner_snapshot()
                    if view["prepare_gate"] == "ready":
                        self._sweep.begin(
                            now_sec=now,
                            allow_motion=True,
                            human_intervention=self._human,
                            allow_breakaway=True,
                        )
                        self._transition("motion", "low_speed_sweeps_started", now)
                elif self._state == "motion":
                    result = self._sweep.owner_snapshot()
                    if result["state"] in {"complete", "failed", "cancelled"}:
                        self._records.append(self._sweep.result_record())
                        self._transition(
                            "complete"
                            if result["state"] == "complete"
                            else result["state"],
                            result["reason"],
                            now,
                        )
            else:
                commands = dict(values["joint_commands"])
                driver_kd = six(
                    commands.pop("driver_kd_nm_s_per_rad", None),
                    "driver_kd_nm_s_per_rad",
                )
                zero = all(
                    abs(t - g) <= 1e-9 and o == 0.0
                    for t, g, o in zip(
                        six(commands.get("total_nm"), "total_nm"),
                        six(commands.get("gravity_nm"), "gravity_nm"),
                        six(commands.get("overlay_nm"), "overlay_nm"),
                    )
                )
                expected_kd = self._requested_kd
                if (
                    self._state == "onset_restore"
                    or self.config.onset_release_braking and expected_kd[2] > self._sent_kd[2]
                ) and (
                    not zero or driver_kd == self._sent_kd
                ):
                    expected_kd = self._sent_kd
                if any(abs(a - b) > 1e-9 for a, b in zip(driver_kd, expected_kd)):
                    self.recording_failed("support_damping_dispatch_mismatch")
                    raise ValueError(
                        "support damping receipt does not match this control frame"
                    )
                self._direct.record_dispatch(
                    **{
                        k: v
                        for k, v in values.items()
                        if k != "torque_rate_nm_s" and k != "joint_commands"
                    },
                    joint_commands=commands,
                )
                self._last_completed, self._last_receipt = (
                    now,
                    copy.deepcopy(native_receipt),
                )
                if self._kd_plateau_since is None or driver_kd != self._sent_kd:
                    self._kd_plateau_since = now
                self._sent_kd = driver_kd
                self._last_sent_zero = zero
                self._zero_plateau_since = (
                    now
                    if zero and self._zero_plateau_since is None
                    else self._zero_plateau_since
                    if zero
                    else None
                )
                self._advance_onset(now)
            if (
                self._state in {"complete", "failed", "cancelled"}
                and not self._terminal_recorded
            ):
                self._records.append(self.result_record())
                self._terminal_recorded = True

    def _advance_onset(self, now: float) -> None:
        view = self._direct.owner_snapshot(observed_at_sec=now)
        self._observe_onset_baseline(now, view)
        if self._onset_start_pending(view) and self._state in {"onset_arm", "onset_scan"}:
            if view["phase"] == "velocity_check":
                # The old zero-first raw-count confirmation still owns this
                # interval; it is NOT a completed or newly authorized trial.
                return
            if view["phase"] == "inactive":
                self._transition("onset_prepare", "pending_start_returned_to_zero_baseline", now)
                return
        if (
            self._state == "onset_prepare"
            and view["prepare_gate"] == "ready"
            and self._damping_settled(now, self._support_target_kd)
            and self._onset_baseline_reason == "ready"
        ):
            self._direct.submit("arm", now_sec=now)
            self._transition("onset_arm", "stationary_baseline_confirmed", now)
        elif self._state == "onset_arm" and view["phase"] == "armed" and self._onset_baseline_reason == "ready":
            direction = self._directions[self._onset_index]
            self._direct.submit(
                "set", now_sec=now, torque_nm=direction * self.limits.max_abs_assist_nm
            )
            self._scan_min = self._scan_max = self._last_qpos
            self._cap_since = None
            self._scan_end_reason = None
            self._scan_started = now
            self._onset_first_zero = None
            self._scan_withdrawal_reason = None
            self._transition("onset_scan", "bounded_native_ramp_started", now)
        elif self._state == "onset_scan":
            if view["phase"] in {"onset_release", "velocity_check", "inactive"}:
                self._transition(
                    "onset_release", "native_motion_or_budget_withdrawal", now
                )
            else:
                sent = view["last_dispatch"]
                delta = sent["conditioned_total_nm"] - sent["gravity_nm"]
                at_cap = (
                    abs(delta) >= self.limits.max_abs_assist_nm - 1e-6
                    and abs(view["effective_nm"])
                    >= self.limits.max_abs_assist_nm - 1e-6
                )
                self._cap_since = (
                    now
                    if at_cap and self._cap_since is None
                    else self._cap_since
                    if at_cap
                    else None
                )
                if (
                    self._cap_since is not None
                    and now - self._cap_since >= self.config.cap_dwell_sec
                ):
                    self._scan_end_reason = "cap_dwell_without_confirmed_motion"
                    self._direct.submit("zero", now_sec=now)
                    self._transition("onset_release", self._scan_end_reason, now)
        elif (
            self._state == "onset_release"
            and view["phase"] == "inactive"
            and view["prepare_gate"] == "ready"
        ):
            trial = view["last_completed_trial"]
            if (
                trial is not None
                and self._scan_started is not None
                and trial["started_at_sec"] >= self._scan_started
                and len(self._onset_trials) <= self._onset_index
            ):
                trial = {
                    **trial,
                    "direction": self._directions[self._onset_index],
                    "scan_end_reason": self._scan_end_reason,
                    "qpos_min_rad": self._scan_min,
                    "qpos_max_rad": self._scan_max,
                    "human_intervention": self._human,
                    "active_driver_kd_nm_s_per_rad": self._support_target_kd,
                    "zero_assist_completed_at_sec": self._onset_first_zero,
                    "withdrawal_reason": self._scan_withdrawal_reason,
                }
                self._onset_trials.append(trial)
                self._records.append(
                    {
                        "kind": "automatic_calibration_onset_trial",
                        "schema": "b601.automatic-onset-trial.v1",
                        "batch_id": self._batch_id,
                        "recorded_monotonic_sec": now,
                        "trial": trial,
                    }
                )
            if self._cancel_requested is not None:
                self._transition("onset_restore", self._cancel_requested, now)
                return
            if trial is None or self._scan_started is None or trial["started_at_sec"] < self._scan_started:
                # An unstarted intent cannot consume a calibration trial.
                self._transition("onset_prepare", "unstarted_scan_waiting_for_baseline", now)
                return
            self._onset_index += 1
            self._onset_prepare_deadline = None
            if self._onset_index < len(self._directions):
                self._scan_started = None
                self._transition(
                    "onset_prepare", "previous_scan_zero_and_still_confirmed", now
                )
            else:
                self._transition(
                    "onset_restore", "onset_trials_complete_restoring_base_damping", now
                )
        elif (
            self._state == "onset_restore"
            and view["phase"] == "inactive"
            and view["prepare_gate"] == "ready"
            and self._damping_settled(now, (0.0,) * 6)
        ):
            self._base_restore_completed = now
            if self._cancel_requested is not None:
                self._transition("cancelled", self._cancel_requested, now)
            else:
                plan = plan_local_sweeps(
                    condition=self._condition,
                    joint_name="joint3",
                    anchor_qpos_rad=self._last_qpos,
                    position_limits_rad=self._position_limits,
                    excursion_rad=self.config.sweep_excursion_rad,
                    speeds_rad_s=self.config.sweep_speeds_rad_s,
                    max_reference_velocity_rad_s=self.settings.staged_assist.motion_control.max_reference_velocity_rad_s,
                    purpose=self.config.motion_plan,
                )
                self._sweep = B601AutomaticCalibrationSession(
                    self._staged,
                    plan=plan,
                    condition=self._condition,
                    position_limits_rad=self._position_limits,
                    limits=CalibrationBatchLimits(
                        self.config.max_batch_sec,
                        self.config.max_sweep_leg_sec,
                        self.config.prepare_timeout_sec,
                        self.limits.command_timeout_sec,
                        self.settings.staged_assist.motion_control.max_control_gap_sec,
                        self.config.max_sweep_command_exposure_nm_sec,
                        self.config.max_sweep_leg_command_exposure_nm_sec,
                    ),
                    acquisition_arrival_tolerance_rad=self.config.motion_acquisition_arrival_tolerance_rad,
                )
                self._motion_started = True
                self._transition(
                    "motion_prepare",
                    "onset_trials_recorded_waiting_for_motion_baseline",
                    now,
                )

    def _damping_settled(self, now: float, target: tuple[float, ...]) -> bool:
        return (
            all(abs(a - b) <= 1e-9 for a, b in zip(self._sent_kd, target))
            and self._last_sent_zero
            and self._zero_plateau_since is not None
            and now - self._zero_plateau_since
            >= self.settings.staged_assist.motion_control.velocity_window_sec
            and self._kd_plateau_since is not None
            and now - self._kd_plateau_since
            >= self.settings.staged_assist.motion_control.velocity_window_sec
        )

    def _outcome(self) -> dict[str, str]:
        # A rejected receipt/recording failure can precede a terminal phase.
        # Surface it immediately, without mutating the withdrawal state machine.
        if self._dispatch_rejection is not None:
            return calibration_outcome("failed", "dispatch_rejected:" + self._dispatch_rejection["reason"])
        if self._cancel_requested is not None and self._cancel_requested.startswith("recording_failed:"):
            return calibration_outcome("failed", self._cancel_requested)
        return calibration_outcome(self._state, self._reason)

    def _small_status(self) -> dict[str, Any]:
        return {
            "operation_mode": "fixed_position",
            "state": self._state,
            "reason": self._reason,
            "outcome": self._outcome(),
            "active": self._active,
            "start_requested": self._start_requested,
            "stage": "motion" if self._motion_active else "onset",
            "onset_index": self._onset_index,
            "onset_count": len(self._directions),
            "completed_onset_trials": len(self._onset_trials),
            "human_intervention": self._human,
            "motion_plan": self.config.motion_plan,
            "feedback_velocity_timebase": self.config.feedback_velocity_timebase,
            "planned_motion_steps": (
                2 if self.config.motion_plan == "motion_check" else
                6 * len(self.config.sweep_speeds_rad_s)
                + (2 if self.config.motion_plan == "model_sweeps" else 0)
            ),
            "motion_acquisition_arrival_tolerance_rad": self.config.motion_acquisition_arrival_tolerance_rad,
            "support_damping": self._onset_damping_status(),
            "onset_baseline": {
                "gate": self._onset_baseline_reason,
                "dwell_sec": 0.0 if self._onset_baseline_since is None else max(0.0, self._last_feedback - self._onset_baseline_since),
                "required_dwell_sec": self.settings.staged_assist.settle_dwell_sec,
                "prepare_deadline_sec": self._onset_prepare_deadline,
            },
            "direction": None
            if self._onset_index >= len(self._directions)
            else self._directions[self._onset_index],
        }

    def owner_snapshot(self) -> dict[str, Any]:
        """Bounded sequencing view, without GUI history or copied trial traces."""
        with self._lock:
            view = self._staged.owner_snapshot() if self._motion_active else self._direct.owner_snapshot(
                observed_at_sec=self._last_completed or self._last_feedback or 0.)
            return {**self._small_status(), "fault": view["fault"]}

    def status(self) -> dict[str, Any]:
        with self._lock:
            view = (
                self._staged.status() if self._motion_active else self._direct.status()
            )
            return {
                **view,
                "controller_mode": self.controller_mode,
                "automatic_calibration": {
                    **self._small_status(),
                    "onset_trials": copy.deepcopy(self._onset_trials),
                    "motion": None if self._sweep is None else self._sweep.status(),
                },
                "events": list(self._events),
            }

    def drain_records(self) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self._records)
            self._records.clear()
            return records

    def recording_failed(self, reason: str) -> None:
        with self._lock:
            self._start_requested = False
            if self._active:
                self._cancel_requested = "recording_failed:" + reason

    def result_record(self) -> dict[str, Any]:
        with self._lock:
            capture = self._direct.owner_snapshot(
                observed_at_sec=self._last_completed or self._last_feedback or 0.0
            )["capture"]
            incomplete = (
                None
                if capture is None
                or capture["trial"] is None
                or "completed_at_sec" in capture["trial"]
                else capture["trial"]
            )
            if incomplete is not None:
                incomplete = {
                    **incomplete,
                    "active_driver_kd_nm_s_per_rad": self._support_target_kd,
                    "zero_assist_completed_at_sec": self._onset_first_zero,
                    "withdrawal_reason": self._scan_withdrawal_reason,
                }
            return {
                "kind": "automatic_calibration_result",
                "operation_mode": "fixed_position",
                "schema": "b601.automatic-joint-calibration.v1",
                "batch_id": self._batch_id,
                "recorded_monotonic_sec": self._last_completed,
                "state": self._state,
                "reason": self._reason,
                "outcome": self._outcome(),
                "condition": None
                if self._condition is None
                else asdict(self._condition),
                "settings": asdict(self.config),
                "onset_trials": copy.deepcopy(self._onset_trials),
                "control_limits": asdict(self.limits),
                "motion_settings": asdict(self.settings.staged_assist),
                "onset_damping": self._onset_damping_status(),
                "dispatch_rejection": copy.deepcopy(self._dispatch_rejection),
                "incomplete_onset_trial": incomplete,
                "motion": None if self._sweep is None else self._sweep.status(),
                "motion_state": None
                if not self._motion_active
                else self._staged.owner_snapshot(),
                "auto_apply": False,
                "measured_output_force": False,
            }

    def _onset_damping_status(self) -> dict[str, Any]:
        return {
            "scope": "onset_stage_only",
            "support_target_kd_nm_s_per_rad": self._support_target_kd,
            "last_acknowledged_kd_nm_s_per_rad": None
            if self._last_receipt is None
            else self._sent_kd,
            "last_requested_kd_nm_s_per_rad": self._requested_kd,
            "zero_assist_completed_at_sec": self._onset_first_zero,
            "base_restore_completed_at_sec": self._base_restore_completed,
            "release_braking_enabled": self.config.onset_release_braking,
            "release_target_kd_nm_s_per_rad": self._release_target_kd,
            "selected_joint_velocity_limit_rad_s": (
                self.config.onset_joint_velocity_limit_rad_s
                if self.config.onset_joint_velocity_limit_rad_s is not None
                else self.limits.max_joint_velocity_rad_s
            ),
            "support_joint_velocity_limit_rad_s": self.limits.max_joint_velocity_rad_s,
            "semantics": "motor_local_damping_gains_not_measured_torque",
        }

    def owner_stopped(self, *, now_sec: float, reason: str) -> None:
        with self._lock:
            if self._state not in {"complete", "failed", "cancelled"}:
                self._transition("cancelled", reason, now_sec)
            self._ready = False
            (self._staged if self._motion_active else self._direct).owner_stopped(
                now_sec=now_sec, reason=reason
            )
