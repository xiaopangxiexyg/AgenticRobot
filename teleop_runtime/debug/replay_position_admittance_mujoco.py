"""Replay saved VIVE targets and wrench through 60 Hz Active into MuJoCo.

This diagnostic is deliberately outside every production entrypoint.  It runs
two deterministic, device-free branches from the same recorded initial qpos:

``baseline``
    saved solver-world VIVE targets -> shared V2 Balanced IK -> 120 Hz qcmd
``active``
    the same targets plus saved raw wrench -> production Active pre-IK filter
    -> shared V2 Balanced IK -> 120 Hz qcmd

Both branches use the fitted RM65 position actuator and 1 kHz MuJoCo physics.
Their time-aligned difference is therefore a system-level counterfactual that
includes stateful IK, command shaping, actuator delay, and physical response.
It is not a force/contact identification test and never constructs hardware.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from robot_motion.position_control.cartesian_admittance import (
    CartesianAdmittanceConfig,
)
from robot_motion.position_control.rotational_admittance import (
    RotationalAdmittanceConfig,
)
from robot_motion.rm65.position_profile import (
    RM65_BIMANUAL_POSITION_V2_BALANCED,
    RM65_BIMANUAL_POSITION_V2_BALANCED_60HZ_JERK_V1_NAME,
    RM65_BIMANUAL_POSITION_V2_BALANCED_60HZ_JERK_V2_NAME,
    RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
    RM65_POSITION_CONTROL_PROFILE_CHOICES,
    resolve_rm65_bimanual_position_profile,
)
from sim.contracts import SimulationContract
from sim.mujoco.actuation import (
    identified_bimanual_dmtac_gripper_profile,
    rm65_bimanual_udp_feedback_profile,
)
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.control.desk_tcp_position import (
    DEFAULT_SIMULATION_CONTRACT_PATH,
    position_15_60_120_timing,
)
from sim.mujoco.scene import MujocoScene
from teleop_runtime.real_teleop.position_admittance_shadow import (
    POSITION_ADMITTANCE_ACTIVE_OUTPUT_FOLLOW_TAU_SEC_MAX,
    POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
    POSITION_ADMITTANCE_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_M,
    POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP,
    POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY,
    POSITION_ADMITTANCE_ACTIVE_NOMINAL_ROTATION_LIMIT_DEG,
    POSITION_ADMITTANCE_TRANSLATION_RELEASE_QPOS_FAST_PATH_ALPHA,
    PositionAdmittanceEndpointExcursionGuard,
    PositionAdmittanceShadowTargetFilter,
    admitted_solver_world_poses_from_joint_command,
    install_position_admittance_active,
    position_admittance_active_uses_home_endpoint_guard,
    position_admittance_final_translation_slew_ratio,
    validate_position_admittance_active_authority,
    validate_position_rotational_admittance_active_config,
)

from .replay_position_admittance_active_pipeline import (
    OFFLINE_ACTIVE_ENVELOPES,
    _RecordedWrenchMonitor,
    _cautious_active_config,
    _frame_payload,
    _recorded_inputs,
)
from .replay_position_admittance_force_trace import (
    _position_admittance_receipt,
)
from .replay_position_admittance_ik_trace import (
    DEFAULT_LEFT_TOOL_PROFILE,
    DEFAULT_REPLAY_HZ,
    DEFAULT_RIGHT_TOOL_PROFILE,
    SIDES,
    _build_retargeter,
    _command_qpos,
    _config_payload,
    _context,
    _frame_rows,
    _mapping,
    _qpos,
    _resolve_log,
    _rotation_error_deg,
    _stats,
    _target_at_tick,
    _tool_pose,
)


MUJOCO_INCREMENTAL_TRANSLATION_LIMIT_M = 0.003
MUJOCO_INCREMENTAL_ROTATION_LIMIT_DEG = 0.5
# The Active/qref release contract is 1 mm.  The physical comparison adds a
# separately named 0.5 mm actuator/model margin; it does not enlarge the
# authoritative modifier or admitted-qref envelope.
MUJOCO_RELEASE_QREF_LIMIT_M = 0.001
MUJOCO_RELEASE_PHYSICAL_MARGIN_M = 0.0005
MUJOCO_RELEASE_TRANSLATION_LIMIT_M = (
    MUJOCO_RELEASE_QREF_LIMIT_M + MUJOCO_RELEASE_PHYSICAL_MARGIN_M
)
MUJOCO_RELEASE_ROTATION_LIMIT_DEG = 0.5


_MUJOCO_FRAME_ADMITTANCE_FIELDS = (
    "observed_at_monotonic_sec",
    "raw_target_pose_solver_world",
    "raw_wrench_n_nm",
    "tared_wrench_n_nm",
    "source_time_sec",
    "tare_generation",
)


def _compact_mujoco_frame(row: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only fields consumed by saved-wrench MuJoCo replay.

    Action-debug frames intentionally contain broad runtime evidence.  Keeping
    every nested command/backend/timing payload in memory multiplies a large
    real run into several GiB before MuJoCo rollout arrays are allocated.  The
    replay contract consumes only raw targets/wrenches, initial qref, activity
    evidence, and timestamps; project exactly those fields while streaming.
    """

    admittance_sides = _mapping(
        _mapping(row.get("position_admittance")).get("sides")
    )
    commands = _mapping(row.get("command_by_side"))
    tracker_channels = _mapping(row.get("tracker_channels"))
    return {
        "event": "frame",
        "teleop_command_timestamp": row.get("teleop_command_timestamp"),
        "sample_timestamp": row.get("sample_timestamp"),
        "position_admittance": {
            "sides": {
                side: {
                    field: _mapping(admittance_sides.get(side)).get(field)
                    for field in _MUJOCO_FRAME_ADMITTANCE_FIELDS
                }
                for side in SIDES
            }
        },
        "command_by_side": {
            side: {
                "value_qpos": _mapping(commands.get(side)).get("value_qpos"),
                "metadata": {
                    "target_activity": _mapping(
                        _mapping(commands.get(side)).get("metadata")
                    ).get("target_activity")
                },
            }
            for side in SIDES
        },
        "tracker_channels": {
            f"{side}_wrist_pose": {
                "motion_window": _mapping(
                    tracker_channels.get(f"{side}_wrist_pose")
                ).get("motion_window")
            }
            for side in SIDES
        },
    }


def _read_mujoco_rows(path: Path) -> list[Mapping[str, Any]]:
    """Stream and compact action-debug rows for bounded-memory replay."""

    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(
                _compact_mujoco_frame(value)
                if value.get("event") == "frame"
                else value
            )
    return rows
# A force release must not expose a long-lived downstream IK/actuator backlog.
# These are physical counterfactual limits, evaluated against the no-modifier
# branch after the filtered force has crossed back inside its return deadband.
MUJOCO_FORCE_RELEASE_CATCHUP_LIMIT_M = 0.001
MUJOCO_FORCE_RELEASE_CATCHUP_DURATION_SEC = 0.1
MUJOCO_FORCE_RELEASE_OBSERVATION_SEC = 1.0
MUJOCO_FORCE_RELEASE_DURATION_MIN_CATCHUP_M = (
    MUJOCO_RELEASE_PHYSICAL_MARGIN_M
)
MUJOCO_TORQUE_RELEASE_CATCHUP_LIMIT_DEG = 0.2
MUJOCO_TORQUE_RELEASE_CATCHUP_DURATION_SEC = 0.1
MUJOCO_TORQUE_RELEASE_OBSERVATION_SEC = 1.0
MUJOCO_TORQUE_RELEASE_DURATION_MIN_CATCHUP_DEG = 0.05
# Offline timestamps are large absolute monotonic values.  Subtracting them
# can put an exact 0.1 s boundary a few picoseconds above its decimal value.
# Keep the acceptance contract exact to one nanosecond without widening any
# physical displacement or angle threshold.
MUJOCO_TIME_COMPARISON_TOLERANCE_SEC = 1e-9
MUJOCO_FINAL_QCMD_TRACKING_LIMIT_RAD = 0.01
MUJOCO_FIXED_SIDE_COUNTERFACTUAL_LIMIT_RAD = 1e-5
MUJOCO_FRAME_TRANSLATION_TOLERANCE_M = 1e-8
MUJOCO_FRAME_ROTATION_TOLERANCE_DEG = 1e-6
# Ignore only numerical-scale decay when asking whether the rotational
# modifier requested motion on a replay tick.  This keeps the qref hold metric
# about visible control motion instead of counting sub-micro-radian tails.
MUJOCO_ROTATION_MOTION_STEP_FLOOR_DEG = 1e-4
MUJOCO_ROTATIONAL_QPOS_FAST_PATH_ALPHA = 0.80
OFFLINE_NORMAL_PARITY_ENVELOPE = "normal-parity-offline"
PRODUCTION_NORMAL_PARITY_ENVELOPE = "normal-parity-active"
OFFLINE_NORMAL_PARITY_MAX_TOOL_EXCURSION_M = 0.08
ADMITTANCE_ENVELOPES = (
    "cautious-active",
    "recorded-shadow",
    OFFLINE_NORMAL_PARITY_ENVELOPE,
    PRODUCTION_NORMAL_PARITY_ENVELOPE,
)


@dataclass
class _ReplayClock:
    value: float

    def __call__(self) -> float:
        return float(self.value)


@dataclass
class _PhysicalRollout:
    name: str
    report: dict[str, Any]
    sample_times_sec: list[float]
    measured_qpos: dict[str, list[np.ndarray]]
    measured_pose: dict[str, list[np.ndarray]]
    qref: dict[str, list[np.ndarray]]
    qcmd: dict[str, list[np.ndarray]]
    raw_target_pose: dict[str, list[np.ndarray]]
    applied_target_pose: dict[str, list[np.ndarray]]
    filtered_force_control_n: list[np.ndarray]
    active_offset_m: list[np.ndarray]
    filtered_torque_control_nm: list[np.ndarray] = field(default_factory=list)
    active_rotation_rotvec_rad: list[np.ndarray] = field(default_factory=list)


def _finite_max(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return max(finite) if finite else None


def _qcmd_dynamics(
    *,
    times_sec: Sequence[float],
    qcmd_by_side: Mapping[str, Sequence[np.ndarray]],
    nominal_sample_hz: float,
) -> dict[str, dict[str, Any]]:
    """Summarize logical qcmd samples and quantized physical apply timing."""

    times = np.asarray(times_sec, dtype=float).reshape(-1)
    nominal_hz = float(nominal_sample_hz)
    if not math.isfinite(nominal_hz) or nominal_hz <= 0.0:
        raise ValueError("qcmd nominal_sample_hz must be positive and finite")
    result: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        qcmd = np.asarray(qcmd_by_side[side], dtype=float)
        if qcmd.ndim != 2 or qcmd.shape[1:] != (6,) or len(qcmd) != len(times):
            raise ValueError(f"{side} qcmd dynamics require aligned Nx6 samples")
        if len(times) < 4:
            result[side] = {
                "sample_count": int(len(times)),
                "acceleration_deg_s2": _stats(()),
                "jerk_deg_s3": _stats(()),
                "power_fraction_27_30hz": None,
            }
            continue
        dt = np.diff(times)
        if np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
            raise ValueError("qcmd dynamics require strictly increasing times")
        velocity = np.diff(qcmd, axis=0) / dt[:, None]
        velocity_times = times[1:]
        acceleration_dt = np.diff(velocity_times)
        acceleration = np.diff(velocity, axis=0) / acceleration_dt[:, None]
        acceleration_times = velocity_times[1:]
        jerk_dt = np.diff(acceleration_times)
        jerk = np.diff(acceleration, axis=0) / jerk_dt[:, None]
        acceleration_deg = np.rad2deg(np.max(np.abs(acceleration), axis=1))
        jerk_deg = np.rad2deg(np.max(np.abs(jerk), axis=1))

        median_interval_hz = float(1.0 / np.median(dt))
        effective_sample_hz = float((len(times) - 1) / (times[-1] - times[0]))
        nominal_period_sec = 1.0 / nominal_hz
        apply_interval_error_sec = dt - nominal_period_sec
        sample_index = np.arange(len(qcmd), dtype=float)
        design = np.column_stack((sample_index, np.ones_like(sample_index)))
        trend, *_ = np.linalg.lstsq(design, qcmd, rcond=None)
        detrended = qcmd - design @ trend
        power = np.abs(np.fft.rfft(detrended, axis=0)) ** 2
        # The planner emits one logical sample per nominal command cut.  The
        # MuJoCo actuator applies those samples on the first following physics
        # tick, so 1000 Hz physics yields alternating 8/9 ms intervals.  Use
        # the logical rate for spectrum bins and report application-time
        # quantization separately; the inverse median interval (125 Hz here)
        # is not the long-run command rate.
        frequencies = np.fft.rfftfreq(len(qcmd), d=nominal_period_sec)
        total_mask = (frequencies >= 1.0) & (frequencies <= 60.0)
        band_mask = (frequencies >= 27.0) & (frequencies <= 30.0)
        total_by_joint = np.sum(power[total_mask], axis=0)
        band_by_joint = np.sum(power[band_mask], axis=0)
        total = float(np.sum(total_by_joint))
        band = float(np.sum(band_by_joint))
        wrist_total = float(np.sum(total_by_joint[3:6]))
        wrist_band = float(np.sum(band_by_joint[3:6]))
        result[side] = {
            "sample_count": int(len(times)),
            "sample_hz": effective_sample_hz,
            "nominal_sample_hz": nominal_hz,
            "effective_sample_hz": effective_sample_hz,
            "median_interval_hz": median_interval_hz,
            "physical_apply_interval_sec": _stats(dt.tolist()),
            "physical_apply_interval_error_sec": _stats(
                apply_interval_error_sec.tolist()
            ),
            "acceleration_deg_s2": _stats(acceleration_deg.tolist()),
            "jerk_deg_s3": _stats(jerk_deg.tolist()),
            "acceleration_by_joint_deg_s2": [
                _stats(np.rad2deg(np.abs(acceleration[:, joint])).tolist())
                for joint in range(6)
            ],
            "jerk_by_joint_deg_s3": [
                _stats(np.rad2deg(np.abs(jerk[:, joint])).tolist())
                for joint in range(6)
            ],
            "wrist_acceleration_deg_s2": _stats(
                np.rad2deg(
                    np.max(np.abs(acceleration[:, 3:6]), axis=1)
                ).tolist()
            ),
            "wrist_jerk_deg_s3": _stats(
                np.rad2deg(np.max(np.abs(jerk[:, 3:6]), axis=1)).tolist()
            ),
            "power_fraction_27_30hz": (
                None if total <= 0.0 else band / total
            ),
            "power_fraction_27_30hz_by_joint": [
                (
                    None
                    if float(total_by_joint[joint]) <= 0.0
                    else float(band_by_joint[joint] / total_by_joint[joint])
                )
                for joint in range(6)
            ],
            "wrist_power_fraction_27_30hz": (
                None if wrist_total <= 0.0 else wrist_band / wrist_total
            ),
        }
    return result


def _force_release_transients(
    *,
    baseline: _PhysicalRollout,
    candidate: _PhysicalRollout,
    selected_side: str,
    release_force_deadband_n: Sequence[float],
) -> dict[str, Any]:
    """Measure physical outward catch-up after a sustained force release."""

    sample_count = len(candidate.sample_times_sec)
    aligned_lengths = (
        len(baseline.sample_times_sec),
        len(baseline.measured_pose[selected_side]),
        len(candidate.measured_pose[selected_side]),
        len(candidate.filtered_force_control_n),
        len(candidate.active_offset_m),
    )
    if any(length != sample_count for length in aligned_lengths):
        raise ValueError("force-release analysis requires aligned rollout samples")
    deadband = np.asarray(release_force_deadband_n, dtype=float).reshape(-1)
    if deadband.shape != (3,) or not np.all(np.isfinite(deadband)):
        raise ValueError("release force deadband must contain three finite values")
    if np.any(deadband < 0.0):
        raise ValueError("release force deadband must be non-negative")

    physical_effect_m = [
        float(
            np.linalg.norm(
                candidate.measured_pose[selected_side][index][:3, 3]
                - baseline.measured_pose[selected_side][index][:3, 3]
            )
        )
        for index in range(sample_count)
    ]
    force_active = [
        bool(
            np.any(
                np.abs(np.asarray(force, dtype=float).reshape(-1)[:3])
                > deadband + 1e-12
            )
        )
        for force in candidate.filtered_force_control_n
    ]
    events: list[dict[str, Any]] = []
    ignored_short_releases = 0
    for release_index in range(1, sample_count):
        if not force_active[release_index - 1] or force_active[release_index]:
            continue
        if (
            float(
                np.linalg.norm(
                    np.asarray(
                        candidate.active_offset_m[release_index - 1],
                        dtype=float,
                    ).reshape(3)
                )
            )
            <= 1e-9
        ):
            continue
        release_time = float(candidate.sample_times_sec[release_index])
        end_index = release_index
        while end_index + 1 < sample_count:
            next_index = end_index + 1
            if force_active[next_index]:
                break
            if (
                float(candidate.sample_times_sec[next_index]) - release_time
                > MUJOCO_FORCE_RELEASE_OBSERVATION_SEC + 1e-12
            ):
                break
            end_index = next_index
        observation_duration = (
            float(candidate.sample_times_sec[end_index]) - release_time
        )
        if observation_duration + 1e-12 < MUJOCO_FORCE_RELEASE_CATCHUP_DURATION_SEC:
            ignored_short_releases += 1
            continue

        window = np.asarray(
            physical_effect_m[release_index : end_index + 1],
            dtype=float,
        )
        outward_growth = window - float(window[0])
        peak_offset = int(np.argmax(outward_growth))
        peak_index = release_index + peak_offset
        catchup_m = max(0.0, float(outward_growth[peak_offset]))
        catchup_duration_sec = (
            float(candidate.sample_times_sec[peak_index]) - release_time
            if catchup_m > 1e-12
            else 0.0
        )
        duration_relevant = bool(
            catchup_m
            > MUJOCO_FORCE_RELEASE_DURATION_MIN_CATCHUP_M + 1e-12
        )
        duration_passed = bool(
            not duration_relevant
            or catchup_duration_sec
            <= MUJOCO_FORCE_RELEASE_CATCHUP_DURATION_SEC
            + MUJOCO_TIME_COMPARISON_TOLERANCE_SEC
        )
        event_passed = bool(
            catchup_m <= MUJOCO_FORCE_RELEASE_CATCHUP_LIMIT_M + 1e-12
            and duration_passed
        )
        events.append(
            {
                "release_index": release_index,
                "release_time_sec": release_time,
                "observation_duration_sec": observation_duration,
                "physical_effect_at_release_m": float(window[0]),
                "peak_physical_effect_m": float(window[peak_offset]),
                "outward_catchup_m": catchup_m,
                "outward_catchup_duration_sec": catchup_duration_sec,
                "duration_relevant": duration_relevant,
                "duration_passed": duration_passed,
                "passed": event_passed,
            }
        )

    return {
        "passed": bool(events and all(event["passed"] for event in events)),
        "event_count": len(events),
        "ignored_short_release_count": ignored_short_releases,
        "max_outward_catchup_m": _finite_max(
            [float(event["outward_catchup_m"]) for event in events]
        ),
        "max_outward_catchup_duration_sec": _finite_max(
            [float(event["outward_catchup_duration_sec"]) for event in events]
        ),
        "events": events,
    }


def _torque_release_transients(
    *,
    baseline: _PhysicalRollout,
    candidate: _PhysicalRollout,
    selected_side: str,
    release_torque_deadband_nm: Sequence[float],
) -> dict[str, Any]:
    """Measure physical angular catch-up after sustained torque release."""

    sample_count = len(candidate.sample_times_sec)
    aligned_lengths = (
        len(baseline.sample_times_sec),
        len(baseline.measured_pose[selected_side]),
        len(candidate.measured_pose[selected_side]),
        len(candidate.filtered_torque_control_nm),
        len(candidate.active_rotation_rotvec_rad),
    )
    if any(length != sample_count for length in aligned_lengths):
        raise ValueError("torque-release analysis requires aligned rollout samples")
    deadband = np.asarray(release_torque_deadband_nm, dtype=float).reshape(-1)
    if deadband.shape != (3,) or not np.all(np.isfinite(deadband)):
        raise ValueError("release torque deadband must contain three finite values")
    if np.any(deadband < 0.0):
        raise ValueError("release torque deadband must be non-negative")

    physical_effect_deg = [
        _rotation_error_deg(
            baseline.measured_pose[selected_side][index][:3, :3],
            candidate.measured_pose[selected_side][index][:3, :3],
        )
        for index in range(sample_count)
    ]
    torque_active = [
        bool(
            np.any(
                np.abs(np.asarray(torque, dtype=float).reshape(-1)[:3])
                > deadband + 1e-12
            )
        )
        for torque in candidate.filtered_torque_control_nm
    ]
    events: list[dict[str, Any]] = []
    ignored_short_releases = 0
    for release_index in range(1, sample_count):
        if not torque_active[release_index - 1] or torque_active[release_index]:
            continue
        if (
            float(
                np.linalg.norm(
                    np.asarray(
                        candidate.active_rotation_rotvec_rad[release_index - 1],
                        dtype=float,
                    ).reshape(3)
                )
            )
            <= 1e-9
        ):
            continue
        release_time = float(candidate.sample_times_sec[release_index])
        end_index = release_index
        while end_index + 1 < sample_count:
            next_index = end_index + 1
            if torque_active[next_index]:
                break
            if (
                float(candidate.sample_times_sec[next_index]) - release_time
                > MUJOCO_TORQUE_RELEASE_OBSERVATION_SEC + 1e-12
            ):
                break
            end_index = next_index
        observation_duration = (
            float(candidate.sample_times_sec[end_index]) - release_time
        )
        if observation_duration + 1e-12 < MUJOCO_TORQUE_RELEASE_CATCHUP_DURATION_SEC:
            ignored_short_releases += 1
            continue

        window = np.asarray(
            physical_effect_deg[release_index : end_index + 1],
            dtype=float,
        )
        outward_growth = window - float(window[0])
        peak_offset = int(np.argmax(outward_growth))
        peak_index = release_index + peak_offset
        catchup_deg = max(0.0, float(outward_growth[peak_offset]))
        catchup_duration_sec = (
            float(candidate.sample_times_sec[peak_index]) - release_time
            if catchup_deg > 1e-12
            else 0.0
        )
        duration_relevant = bool(
            catchup_deg
            > MUJOCO_TORQUE_RELEASE_DURATION_MIN_CATCHUP_DEG + 1e-12
        )
        duration_passed = bool(
            not duration_relevant
            or catchup_duration_sec
            <= MUJOCO_TORQUE_RELEASE_CATCHUP_DURATION_SEC
            + MUJOCO_TIME_COMPARISON_TOLERANCE_SEC
        )
        event_passed = bool(
            catchup_deg <= MUJOCO_TORQUE_RELEASE_CATCHUP_LIMIT_DEG + 1e-12
            and duration_passed
        )
        events.append(
            {
                "release_index": release_index,
                "release_time_sec": release_time,
                "observation_duration_sec": observation_duration,
                "physical_effect_at_release_deg": float(window[0]),
                "peak_physical_effect_deg": float(window[peak_offset]),
                "outward_catchup_deg": catchup_deg,
                "outward_catchup_duration_sec": catchup_duration_sec,
                "duration_relevant": duration_relevant,
                "duration_passed": duration_passed,
                "passed": event_passed,
            }
        )

    return {
        "passed": bool(events and all(event["passed"] for event in events)),
        "event_count": len(events),
        "ignored_short_release_count": ignored_short_releases,
        "max_outward_catchup_deg": _finite_max(
            [float(event["outward_catchup_deg"]) for event in events]
        ),
        "max_outward_catchup_duration_sec": _finite_max(
            [float(event["outward_catchup_duration_sec"]) for event in events]
        ),
        "events": events,
    }


def _profile_execution_config(
    *,
    timing,
    arm_profile,
    gripper_profile,
    position_execution_profile_name: str = RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
) -> MujocoPositionExecutionConfig:
    profile = resolve_rm65_bimanual_position_profile(
        position_execution_profile_name
    )
    execution = profile.command_execution
    return MujocoPositionExecutionConfig(
        timing=timing,
        max_velocity_rad_s=tuple(execution.max_velocity_rad_s),
        max_acceleration_rad_s2=tuple(execution.max_acceleration_rad_s2),
        max_command_step_deg=float(execution.max_command_step_deg),
        velocity_ramp_sec=float(execution.velocity_ramp_sec),
        target_velocity_filter_tau_sec=float(
            execution.target_velocity_filter_tau_sec
        ),
        degraded_speed_scale=float(execution.degraded_speed_scale),
        follow_qpos_alpha=float(execution.follow_qpos_alpha),
        position_command_planner=str(execution.planner),
        sample_aligned_source_hz=float(
            profile.rates.sample_hz
            if execution.sample_aligned_source_hz is None
            else execution.sample_aligned_source_hz
        ),
        sample_aligned_source_latch_mode=(
            execution.sample_aligned_source_latch_mode
        ),
        sample_aligned_horizon_source_steps=int(
            execution.sample_aligned_horizon_source_steps
        ),
        sample_aligned_safety_horizon_source_steps=(
            execution.sample_aligned_safety_horizon_source_steps
        ),
        sample_aligned_scale_degraded_target=bool(
            execution.sample_aligned_scale_degraded_target
        ),
        sample_aligned_jerk_regularization=float(
            execution.sample_aligned_jerk_regularization
        ),
        sample_aligned_jerk_delta_regularization=float(
            execution.sample_aligned_jerk_delta_regularization
        ),
        actuation_profile=arm_profile,
        gripper_actuation_profile=gripper_profile,
        realtime_worker=False,
    )


def _admittance_config_for_envelope(
    recorded: CartesianAdmittanceConfig,
    *,
    admittance_envelope: str,
) -> CartesianAdmittanceConfig:
    envelope = str(admittance_envelope).strip().lower()
    if envelope == "cautious-active":
        return _cautious_active_config(recorded)
    if envelope == "recorded-shadow":
        return recorded
    if envelope in {
        OFFLINE_NORMAL_PARITY_ENVELOPE,
        PRODUCTION_NORMAL_PARITY_ENVELOPE,
    }:
        # Reconstruct the established 15--40 Hz normal-parity motion envelope
        # even when the source run's receipt records a cautious or
        # translation-disabled Active configuration.  The production-named
        # branch is admitted again by the real Active installer below; the
        # offline-named branch retains the historical counterfactual bypass.
        limits = OFFLINE_ACTIVE_ENVELOPES[
            "normal_parity_release_braked"
            if envelope == PRODUCTION_NORMAL_PARITY_ENVELOPE
            else "normal_parity"
        ]

        def vector(name: str) -> tuple[float, float, float]:
            value = float(limits[name])
            return (value, value, value)

        return replace(
            recorded,
            enabled_axes=(True, True, True),
            max_offset_m=vector("max_offset_m"),
            max_velocity_m_s=vector("max_velocity_m_s"),
            max_return_velocity_m_s=vector("max_return_velocity_m_s"),
            max_output_slew_m_s=vector("max_output_slew_m_s"),
            return_rate_s=(
                4.0
                if envelope == PRODUCTION_NORMAL_PARITY_ENVELOPE
                else recorded.return_rate_s
            ),
        )
    raise ValueError(
        "admittance_envelope must be cautious-active, recorded-shadow, "
        "normal-parity-offline, or normal-parity-active"
    )


def _install_offline_recorded_shadow_counterfactual(
    *,
    retargeter: Any,
    monitor: Any,
    active_sides: Sequence[str],
    config: CartesianAdmittanceConfig,
    rotational_config: RotationalAdmittanceConfig | None = None,
    side_activity: Callable[[str], tuple[bool, str]],
    apply_sides: Sequence[str],
    endpoint_policy: str = POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
    now_sec: float,
) -> PositionAdmittanceShadowTargetFilter:
    """Apply an explicit counterfactual envelope only in device-free replay.

    This intentionally does not call or weaken the production Active installer.
    The debug-local counterfactual retains the same controller, final modifier
    checks, IK, collision admission, command shaping, and MuJoCo execution while
    using the explicitly selected offline config.
    """

    tool_tcp_from_solver = getattr(retargeter, "_tool_tcp_from_solver", {})
    rotations = {
        side: np.asarray(
            tool_tcp_from_solver.get(side, np.eye(4)),
            dtype=float,
        )[:3, :3]
        for side in active_sides
    }
    observer = PositionAdmittanceShadowTargetFilter(
        monitor=monitor,
        active_sides=active_sides,
        config=config,
        rotational_config=rotational_config,
        previous_filter=getattr(retargeter, "target_override_filter", None),
        side_activity=side_activity,
        tool_rotation_from_solver_by_side=rotations,
        apply_to_target=True,
        apply_sides=apply_sides,
        qpos_fast_path_alpha=(
            None
            if rotational_config is None
            else MUJOCO_ROTATIONAL_QPOS_FAST_PATH_ALPHA
        ),
        active_endpoint_policy=endpoint_policy,
    )
    observer.reset_from_monitor(now_sec=now_sec)
    retargeter.target_override_filter = observer
    return observer


def _joint_bounds(scene: MujocoScene) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in SIDES:
        lower: list[float] = []
        upper: list[float] = []
        for name in scene.arm_specs[side].joint_names:
            joint_id = scene.mujoco.mj_name2id(
                scene.model,
                scene.mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            if joint_id < 0 or not bool(scene.model.jnt_limited[joint_id]):
                raise RuntimeError(f"MuJoCo arm joint has no finite limit: {name}")
            values = np.asarray(scene.model.jnt_range[joint_id], dtype=float)
            lower.append(float(values[0]))
            upper.append(float(values[1]))
        result[side] = (np.asarray(lower), np.asarray(upper))
    return result


def _initial_frame_check(
    *,
    scene: MujocoScene,
    retargeter,
    initial_qpos: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    solver_poses = retargeter.solver_world_targets_for_qpos_pair(
        left_qpos=initial_qpos["left"],
        right_qpos=initial_qpos["right"],
    )
    by_side: dict[str, dict[str, float]] = {}
    for side in SIDES:
        mujoco_pose = scene.body_pose(scene.arm_specs[side].controlled_body)
        by_side[side] = {
            "translation_m": float(
                np.linalg.norm(
                    solver_poses[side][:3, 3] - mujoco_pose[:3, 3]
                )
            ),
            "rotation_deg": _rotation_error_deg(
                solver_poses[side][:3, :3],
                mujoco_pose[:3, :3],
            ),
        }
    return {
        "passed": all(
            values["translation_m"] <= MUJOCO_FRAME_TRANSLATION_TOLERANCE_M
            and values["rotation_deg"]
            <= MUJOCO_FRAME_ROTATION_TOLERANCE_DEG
            for values in by_side.values()
        ),
        "by_side": by_side,
        "translation_tolerance_m": MUJOCO_FRAME_TRANSLATION_TOLERANCE_M,
        "rotation_tolerance_deg": MUJOCO_FRAME_ROTATION_TOLERANCE_DEG,
        "meaning": "shared_solver_world_endpoint_equals_mujoco_link6_world",
    }


def _physical_rollout(
    *,
    name: str,
    active: bool,
    mode: str,
    frames: Sequence[Mapping[str, Any]],
    frame_times: Sequence[float],
    raw_targets: Sequence[Mapping[str, np.ndarray]],
    tare_by_side: Mapping[str, np.ndarray],
    initial_source_time_by_side: Mapping[str, float],
    selected_side: str,
    config: CartesianAdmittanceConfig,
    rotational_config: RotationalAdmittanceConfig | None = None,
    combined_translation_rotation_active: bool = False,
    admittance_envelope: str,
    config_payload: Mapping[str, Any],
    replay_hz: float,
    release_tail_sec: float,
    left_tool_pose: tuple[float, ...],
    right_tool_pose: tuple[float, ...],
    simulation_contract_path: Path,
    max_tool_excursion_m: float | None,
    max_tool_rotation_excursion_deg: float | None = None,
    endpoint_policy: str = POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
    position_execution_profile_name: str = RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
) -> _PhysicalRollout:
    position_execution_profile = resolve_rm65_bimanual_position_profile(
        position_execution_profile_name
    )
    execution = position_execution_profile.command_execution
    initial_qpos = {
        side: _qpos(_mapping(frames[0].get("command_by_side")), side=side)
        for side in SIDES
    }
    retargeter, qref_step_limit = _build_retargeter(
        initial_qpos=initial_qpos,
        config_payload=config_payload,
        left_tool_pose=left_tool_pose,
        right_tool_pose=right_tool_pose,
    )
    initial_tool_targets = retargeter.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=initial_qpos["left"],
        right_qpos=initial_qpos["right"],
    )

    base_contract = SimulationContract.load(simulation_contract_path)
    timing = position_15_60_120_timing(
        base_contract.timing,
        parameter_source=position_execution_profile.name,
    )
    contract = replace(base_contract, timing=timing)
    arm_profile = rm65_bimanual_udp_feedback_profile()
    gripper_profile = identified_bimanual_dmtac_gripper_profile()
    scene = MujocoScene.from_contract(
        contract,
        position_actuation_profile=arm_profile,
        gripper_actuation_profile=gripper_profile,
    )
    scene.set_qpos_by_side(initial_qpos)
    projector = scene.collision_projector()
    if projector is None:
        raise RuntimeError("MuJoCo replay requires collision safety")
    initial_collision = projector.check_qpos(
        scene.data,
        scene.kinematic_controlled_qpos(),
    )
    if not initial_collision.safe:
        raise RuntimeError(
            "recorded initial qpos is collision-unsafe in MuJoCo: "
            f"{initial_collision.to_dict()}"
        )
    frame_check = _initial_frame_check(
        scene=scene,
        retargeter=retargeter,
        initial_qpos=initial_qpos,
    )
    if not frame_check["passed"]:
        raise RuntimeError(
            "shared IK and MuJoCo world frames disagree: "
            f"{frame_check}"
        )

    clock = _ReplayClock(float(frame_times[0]))
    rig = MujocoPositionExecutionRig(
        scene,
        _profile_execution_config(
            timing=timing,
            arm_profile=arm_profile,
            gripper_profile=gripper_profile,
            position_execution_profile_name=position_execution_profile.name,
        ),
        monotonic_clock=clock,
    )
    rig.start()
    bounds = _joint_bounds(scene)
    monitor = _RecordedWrenchMonitor(
        tare_by_side=tare_by_side,
        initial_source_time_by_side=initial_source_time_by_side,
    )
    observer = None
    endpoint_excursion_guard = None
    production_rotational_active_installer = False
    if active:
        install_kwargs = {
            "retargeter": retargeter,
            "monitor": monitor,
            "active_sides": SIDES,
            "config": config,
            "side_activity": lambda side: (
                (True, "offline_active_side")
                if side == selected_side
                else (False, "offline_fixed_side")
            ),
            "apply_sides": (selected_side,),
            "now_sec": frame_times[0] - 1.0 / replay_hz,
        }
        if (
            rotational_config is not None
            and admittance_envelope == PRODUCTION_NORMAL_PARITY_ENVELOPE
            and combined_translation_rotation_active
        ):
            observer = install_position_admittance_active(
                **install_kwargs,
                rotational_config=rotational_config,
                active_envelope=(
                    POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY
                ),
                combined_translation_rotation_active=True,
                endpoint_policy=endpoint_policy,
            )
            production_rotational_active_installer = True
        elif (
            rotational_config is not None
            and admittance_envelope == OFFLINE_NORMAL_PARITY_ENVELOPE
        ):
            observer = _install_offline_recorded_shadow_counterfactual(
                **install_kwargs,
                rotational_config=rotational_config,
                endpoint_policy=endpoint_policy,
            )
        elif rotational_config is not None:
            if (
                endpoint_policy
                == POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP
            ):
                observer = _install_offline_recorded_shadow_counterfactual(
                    **install_kwargs,
                    rotational_config=rotational_config,
                    endpoint_policy=endpoint_policy,
                )
            else:
                try:
                    validate_position_rotational_admittance_active_config(
                        rotational_config
                    )
                except ValueError:
                    observer = _install_offline_recorded_shadow_counterfactual(
                        **install_kwargs,
                        rotational_config=rotational_config,
                        endpoint_policy=endpoint_policy,
                    )
                else:
                    observer = install_position_admittance_active(
                        **install_kwargs,
                        rotational_config=rotational_config,
                        combined_translation_rotation_active=(
                            combined_translation_rotation_active
                        ),
                    )
                    production_rotational_active_installer = True
        else:
            install = (
                install_position_admittance_active
                if admittance_envelope == "cautious-active"
                else _install_offline_recorded_shadow_counterfactual
            )
            observer = install(
                **install_kwargs,
                **(
                    {"endpoint_policy": endpoint_policy}
                    if install is _install_offline_recorded_shadow_counterfactual
                    else {}
                ),
            )
        if (
            position_admittance_active_uses_home_endpoint_guard(endpoint_policy)
            and (
                max_tool_excursion_m is not None
                or (
                    production_rotational_active_installer
                    and max_tool_rotation_excursion_deg is not None
                )
            )
        ):
            endpoint_excursion_guard = PositionAdmittanceEndpointExcursionGuard(
                retargeter=retargeter,
                active_side=selected_side,
                max_tool_excursion_m=float(
                    POSITION_ADMITTANCE_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_M
                    if max_tool_excursion_m is None
                    else max_tool_excursion_m
                ),
                max_tool_rotation_excursion_deg=(
                    None
                    if (
                        max_tool_rotation_excursion_deg is None
                        or not production_rotational_active_installer
                    )
                    else float(max_tool_rotation_excursion_deg)
                ),
                maximum_allowed_tool_rotation_excursion_deg=(
                    float(max_tool_rotation_excursion_deg)
                    if (
                        production_rotational_active_installer
                        and max_tool_rotation_excursion_deg is not None
                    )
                    else POSITION_ADMITTANCE_ACTIVE_NOMINAL_ROTATION_LIMIT_DEG
                    + 0.5
                ),
            )
            endpoint_excursion_guard.reset_anchor_from_qpos(initial_qpos)

    duration_sec = float(frame_times[-1] - frame_times[0]) + float(
        release_tail_sec
    )
    tick_count = int(math.ceil(duration_sec * replay_hz)) + 1
    tail_update_stride = max(1, int(round(replay_hz / 20.0)))
    event_index = 0
    frame_index = 0
    current_qref = {side: initial_qpos[side].copy() for side in SIDES}
    previous_qcmd = {side: initial_qpos[side].copy() for side in SIDES}
    previous_qcmd_velocity = {side: np.zeros(6, dtype=float) for side in SIDES}
    previous_qcmd_time = 0.0
    previous_observer_time = float(frame_times[0]) - 1.0 / replay_hz

    sample_times: list[float] = []
    measured_qpos = {side: [] for side in SIDES}
    measured_pose = {side: [] for side in SIDES}
    qref_samples = {side: [] for side in SIDES}
    qcmd_samples = {side: [] for side in SIDES}
    qcmd_command_times: list[float] = []
    qcmd_command_samples = {side: [] for side in SIDES}
    raw_target_pose_samples = {side: [] for side in SIDES}
    applied_target_pose_samples = {side: [] for side in SIDES}
    filtered_force_samples: list[np.ndarray] = []
    active_offset_samples: list[np.ndarray] = []
    filtered_torque_samples: list[np.ndarray] = []
    active_rotation_samples: list[np.ndarray] = []
    qref_step_ratios: list[float] = []
    qcmd_step_ratios: list[float] = []
    qcmd_velocity_ratios: list[float] = []
    qcmd_acceleration_ratios: list[float] = []
    qcmd_tracking_errors: list[float] = []
    timing_jitter_sec: list[float] = []
    active_offsets: list[float] = []
    active_slew_ratios: list[float] = []
    active_return_slew_ratios: list[float] = []
    active_standard_slew_ratios: list[float] = []
    passive_return_direction_mismatch_ticks = 0
    active_rotation_offsets: list[float] = []
    active_rotation_slew_ratios: list[float] = []
    previous_offset = np.zeros(3, dtype=float)
    previous_rotation_offset = np.zeros(3, dtype=float)
    nonzero_modifier_ticks = 0
    active_qref_motion_ticks = 0
    feedback_frame_counts: Counter[str] = Counter()
    feedback_state_update_count = 0
    max_feedback_anchor_norm_m = 0.0
    fixed_qref_delta: list[float] = []
    joint_limit_margin_rad: list[float] = []
    preventive_clearance_m: list[float] = []
    preventive_unsafe_ticks = 0
    preventive_unsafe_examples: list[dict[str, Any]] = []
    hard_contact_minimum_m = float("inf")
    hard_contact_faults: list[dict[str, Any]] = []
    branch_counts: Counter[str] = Counter()
    selected_side_branch_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    admittance_micro_reversal_hold_ticks: Counter[str] = Counter()
    admittance_micro_reversal_held_joints: Counter[str] = Counter()
    admittance_micro_reversal_soften_ticks: Counter[str] = Counter()
    admittance_micro_reversal_softened_joints: Counter[str] = Counter()
    model_errors: list[dict[str, Any]] = []
    endpoint_excursions_m: list[float] = []
    endpoint_dispatched_excursions_m: list[float] = []
    tool_rotation_excursions_deg: list[float] = []
    active_rotation_motion_qref_steps_deg: list[float] = []
    active_rotation_motion_qref_hold_ticks = 0
    ik_execution_sync_qcmd_to_previous_qref: list[float] = []
    ik_execution_sync_count = 0

    command_step_limit = math.radians(
        execution.max_command_step_deg
    )
    max_velocity = np.asarray(
        execution.max_velocity_rad_s,
        dtype=float,
    )
    max_acceleration = np.asarray(
        execution.max_acceleration_rad_s2,
        dtype=float,
    )
    opposite_side = "right" if selected_side == "left" else "left"

    try:
        for tick_index in range(tick_count):
            scheduled_relative = tick_index / replay_hz
            physics_relative = float(rig._execution.physics_time_sec)  # noqa: SLF001
            now = float(frame_times[0]) + physics_relative
            clock.value = now
            timing_jitter_sec.append(physics_relative - scheduled_relative)

            # Match the real 60 Hz loop: coupled IK is seeded from the latest
            # transport-accepted 120 Hz qcmd, never from the prior qref.  The
            # acceleration-limited transport may intentionally lag qref.
            accepted_qcmd = rig.accepted_qcmd_by_side()
            ik_execution_sync_qcmd_to_previous_qref.append(
                max(
                    float(
                        np.max(
                            np.abs(
                                accepted_qcmd[side] - current_qref[side]
                            )
                        )
                    )
                    for side in SIDES
                )
            )
            retargeter.set_current_qpos_pair(
                left_qpos=accepted_qcmd["left"],
                right_qpos=accepted_qcmd["right"],
                reset_tracking=False,
            )
            ik_execution_sync_count += 1

            while event_index < len(frames) and frame_times[event_index] <= now + 1e-9:
                monitor.update_from_frame(frames[event_index])
                event_index += 1
            if (
                event_index == len(frames)
                and now > frame_times[-1]
                and tick_index % tail_update_stride == 0
            ):
                tail_delta = now - frame_times[-1]
                monitor.update_zero_tail(
                    tare_by_side=tare_by_side,
                    source_time_by_side={
                        side: float(
                            _frame_payload(frames[-1], side=side).get(
                                "source_time_sec"
                            )
                        )
                        + tail_delta
                        for side in SIDES
                    },
                )
            raw, frame_index = _target_at_tick(
                tick_time=now,
                frame_times=frame_times,
                targets=raw_targets,
                frame_index=frame_index,
                mode=mode,
            )
            frame = frames[frame_index]
            context = _context(frame=frame, targets=raw)
            applied_targets = raw
            force_control = np.zeros(3, dtype=float)
            torque_control = np.zeros(3, dtype=float)
            offset = np.zeros(3, dtype=float)
            rotation_offset = np.zeros(3, dtype=float)
            if observer is not None:
                resolved = observer(raw, object(), object(), now_sec=now)
                if resolved is None:
                    raise RuntimeError("Active observer returned no target pair")
                applied_targets = {
                    side: np.asarray(resolved[side], dtype=float).reshape(4, 4)
                    for side in SIDES
                }
            retargeter._apply_target_override_activity_metadata(context)  # noqa: SLF001
            command = retargeter.build_command_for_context(
                context,
                targets=dict(applied_targets),
            )
            if endpoint_excursion_guard is not None:
                command = endpoint_excursion_guard.filter_command(
                    command,
                    now=now,
                )
                endpoint_snapshot = endpoint_excursion_guard.snapshot()
                endpoint_excursion = endpoint_snapshot.get("last_excursion_m")
                if endpoint_excursion is not None:
                    endpoint_excursions_m.append(float(endpoint_excursion))
                dispatched_excursion = endpoint_snapshot.get(
                    "last_accepted_excursion_m"
                )
                if dispatched_excursion is not None:
                    endpoint_dispatched_excursions_m.append(
                        float(dispatched_excursion)
                    )
            next_qref: dict[str, np.ndarray] = {}
            for side in SIDES:
                next_qref[side], metadata = _command_qpos(command, side=side)
                branch_counts[str(metadata.get("ik_branch_candidate"))] += 1
                if side == selected_side:
                    selected_side_branch_counts[
                        str(metadata.get("ik_branch_candidate"))
                    ] += 1
                status_counts[str(metadata.get("collision_avoidance_status"))] += 1
                output_smoothing = _mapping(
                    metadata.get("output_qpos_smoothing")
                )
                admittance_fast_path = _mapping(
                    output_smoothing.get("admittance_fast_path")
                )
                held_joint_indices = admittance_fast_path.get(
                    "micro_reversal_held_joint_indices",
                    (),
                )
                held_joint_count = (
                    len(held_joint_indices)
                    if isinstance(held_joint_indices, Sequence)
                    and not isinstance(held_joint_indices, (str, bytes))
                    else 0
                )
                admittance_micro_reversal_hold_ticks[side] += int(
                    held_joint_count > 0
                )
                admittance_micro_reversal_held_joints[side] += held_joint_count
                softened_joint_indices = admittance_fast_path.get(
                    "micro_reversal_softened_joint_indices",
                    (),
                )
                softened_joint_count = (
                    len(softened_joint_indices)
                    if isinstance(softened_joint_indices, Sequence)
                    and not isinstance(softened_joint_indices, (str, bytes))
                    else 0
                )
                admittance_micro_reversal_soften_ticks[side] += int(
                    softened_joint_count > 0
                )
                admittance_micro_reversal_softened_joints[side] += (
                    softened_joint_count
                )
            qref_step_ratios.append(
                max(
                    float(
                        np.max(
                            np.abs(next_qref[side] - current_qref[side])
                            / qref_step_limit
                        )
                    )
                    for side in SIDES
                )
            )
            selected_qref_step = float(
                np.max(
                    np.abs(
                        next_qref[selected_side] - current_qref[selected_side]
                    )
                )
            )
            fixed_qref_delta.append(
                float(
                    np.max(
                        np.abs(next_qref[opposite_side] - initial_qpos[opposite_side])
                    )
                )
            )
            admitted_tool_targets = (
                retargeter.tool_tcp_world_targets_for_qpos_pair(
                    left_qpos=next_qref["left"],
                    right_qpos=next_qref["right"],
                )
            )
            tool_rotation_excursion_deg = _rotation_error_deg(
                initial_tool_targets[selected_side][:3, :3],
                admitted_tool_targets[selected_side][:3, :3],
            )
            tool_rotation_excursions_deg.append(tool_rotation_excursion_deg)
            if (
                position_admittance_active_uses_home_endpoint_guard(endpoint_policy)
                and max_tool_rotation_excursion_deg is not None
                and tool_rotation_excursion_deg
                > float(max_tool_rotation_excursion_deg) + 1e-12
            ):
                raise RuntimeError(
                    "Active tool rotation excursion guard rejected MuJoCo qref: "
                    f"{tool_rotation_excursion_deg:.9f} deg exceeds "
                    f"{float(max_tool_rotation_excursion_deg):.9f} deg"
                )
            if observer is not None:
                admitted = admitted_solver_world_poses_from_joint_command(
                    retargeter=retargeter,
                    command=command,
                )
                observer.observe_admitted_poses(admitted, now_sec=now)
                snapshot = _mapping(
                    _mapping(observer.snapshot(now_sec=now).get("sides")).get(
                        selected_side
                    )
                )
                offset = np.asarray(
                    snapshot.get("applied_total_offset_m"), dtype=float
                ).reshape(3)
                force_control = np.asarray(
                    snapshot.get("filtered_force_control_n"), dtype=float
                ).reshape(3)
                if rotational_config is not None:
                    torque_control = np.asarray(
                        snapshot.get("filtered_torque_control_nm"),
                        dtype=float,
                    ).reshape(3)
                    rotation_offset = np.asarray(
                        snapshot.get("applied_rotation_rotvec_rad"),
                        dtype=float,
                    ).reshape(3)
                offset_norm = float(np.linalg.norm(offset))
                rotation_offset_norm = float(np.linalg.norm(rotation_offset))
                active_offsets.append(offset_norm)
                active_rotation_offsets.append(rotation_offset_norm)
                modifier_nonzero = bool(
                    offset_norm > 1e-12 or rotation_offset_norm > 1e-12
                )
                nonzero_modifier_ticks += int(modifier_nonzero)
                active_qref_motion_ticks += int(
                    modifier_nonzero and selected_qref_step > 1e-12
                )
                observer_dt = now - previous_observer_time
                passive_return_active = bool(
                    snapshot.get("passive_return_active")
                )
                slew_ratio, returning_to_zero = (
                    position_admittance_final_translation_slew_ratio(
                        config,
                        previous_offset_m=previous_offset,
                        candidate_offset_m=offset,
                        passive_return_active=passive_return_active,
                        dt_sec=max(observer_dt, 1e-12),
                    )
                )
                active_slew_ratios.append(slew_ratio)
                (
                    active_return_slew_ratios
                    if returning_to_zero
                    else active_standard_slew_ratios
                ).append(slew_ratio)
                passive_return_direction_mismatch_ticks += int(
                    passive_return_active and not returning_to_zero
                )
                if rotational_config is not None:
                    rotation_motion_step = float(
                        np.linalg.norm(
                            rotation_offset - previous_rotation_offset
                        )
                    )
                    rotation_slew_limit = float(
                        max(rotational_config.max_output_slew_rad_s)
                    ) * max(observer_dt, 1e-12)
                    active_rotation_slew_ratios.append(
                        rotation_motion_step / rotation_slew_limit
                    )
                    if (
                        float(np.degrees(rotation_motion_step))
                        > MUJOCO_ROTATION_MOTION_STEP_FLOOR_DEG
                    ):
                        qref_step_deg = float(
                            np.degrees(selected_qref_step)
                        )
                        active_rotation_motion_qref_steps_deg.append(
                            qref_step_deg
                        )
                        active_rotation_motion_qref_hold_ticks += int(
                            qref_step_deg < 0.001
                        )
                previous_offset = offset
                previous_rotation_offset = rotation_offset
                previous_observer_time = now
                feedback = _mapping(snapshot.get("admission_feedback"))
                feedback_frame_counts[str(feedback.get("frame"))] += 1
                feedback_state_update_count += int(
                    bool(feedback.get("state_update_applied"))
                )
                anchor = np.asarray(
                    snapshot.get("anchor_offset_m"), dtype=float
                ).reshape(3)
                max_feedback_anchor_norm_m = max(
                    max_feedback_anchor_norm_m,
                    float(np.linalg.norm(anchor)),
                )
            current_qref = {side: values.copy() for side, values in next_qref.items()}
            rig.send_commands(command)

            next_boundary = (tick_index + 1) / replay_hz
            while float(rig._execution.physics_time_sec) + 1e-12 < next_boundary:  # noqa: SLF001
                clock.value = float(frame_times[0]) + float(
                    rig._execution.physics_time_sec  # noqa: SLF001
                )
                command_due = bool(rig._execution.command_due)  # noqa: SLF001
                command_time = float(rig._execution.physics_time_sec)  # noqa: SLF001
                rig.step_physics()
                if not scene.state_is_finite():
                    raise RuntimeError("MuJoCo state became non-finite")
                hard = projector.oracle.check_contacts(scene.data)
                if math.isfinite(float(hard.minimum_distance_m)):
                    hard_contact_minimum_m = min(
                        hard_contact_minimum_m,
                        float(hard.minimum_distance_m),
                    )
                if not hard.safe:
                    hard_contact_faults.append(hard.to_dict())
                    raise RuntimeError(
                        "forbidden MuJoCo contact crossed the hard penetration limit: "
                        f"{hard.to_dict()}"
                    )
                after_qcmd = rig.accepted_qcmd_by_side()
                if command_due:
                    command_dt = command_time - previous_qcmd_time
                    if command_dt <= 0.0:
                        raise RuntimeError("MuJoCo 120 Hz command time did not advance")
                    velocity_by_side: dict[str, np.ndarray] = {}
                    for side in SIDES:
                        delta = after_qcmd[side] - previous_qcmd[side]
                        velocity = delta / command_dt
                        velocity_by_side[side] = velocity
                        qcmd_step_ratios.append(
                            float(np.max(np.abs(delta)) / command_step_limit)
                        )
                        qcmd_velocity_ratios.append(
                            float(np.max(np.abs(velocity) / max_velocity))
                        )
                        qcmd_acceleration_ratios.append(
                            float(
                                np.max(
                                    np.abs(
                                        velocity - previous_qcmd_velocity[side]
                                    )
                                    / command_dt
                                    / max_acceleration
                                )
                            )
                        )
                    previous_qcmd = {
                        side: after_qcmd[side].copy() for side in SIDES
                    }
                    qcmd_command_times.append(command_time)
                    for side in SIDES:
                        qcmd_command_samples[side].append(
                            after_qcmd[side].copy()
                        )
                    previous_qcmd_velocity = {
                        side: velocity_by_side[side].copy() for side in SIDES
                    }
                    previous_qcmd_time = command_time

            clock.value = float(frame_times[0]) + float(
                rig._execution.physics_time_sec  # noqa: SLF001
            )
            preventive = projector.oracle.check(scene.data)
            preventive_clearance_m.append(float(preventive.minimum_distance_m))
            preventive_unsafe_ticks += int(not preventive.safe)
            if not preventive.safe and len(preventive_unsafe_examples) < 8:
                preventive_unsafe_examples.append(
                    {
                        "tick_index": tick_index,
                        "time_sec": clock.value,
                        "collision": preventive.to_dict(),
                    }
                )
            qcmd_now = rig.accepted_qcmd_by_side()
            sample_times.append(clock.value)
            for side in SIDES:
                measured = scene.get_arm_qpos(side)
                lower, upper = bounds[side]
                joint_limit_margin_rad.append(
                    float(np.min(np.minimum(measured - lower, upper - measured)))
                )
                qcmd_tracking_errors.append(
                    float(np.max(np.abs(measured - qcmd_now[side])))
                )
                measured_qpos[side].append(measured.copy())
                measured_pose[side].append(
                    scene.body_pose(scene.arm_specs[side].controlled_body)
                )
                qref_samples[side].append(next_qref[side].copy())
                qcmd_samples[side].append(qcmd_now[side].copy())
                raw_target_pose_samples[side].append(raw[side].copy())
                applied_target_pose_samples[side].append(
                    np.asarray(applied_targets[side], dtype=float)
                    .reshape(4, 4)
                    .copy()
                )
            filtered_force_samples.append(force_control.copy())
            active_offset_samples.append(offset.copy())
            filtered_torque_samples.append(torque_control.copy())
            active_rotation_samples.append(rotation_offset.copy())
    except Exception as exc:  # noqa: BLE001 - diagnostic must preserve failure.
        model_errors.append(
            {
                "tick_index": len(sample_times),
                "frame_index": frame_index,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        rig.close()

    completed = len(sample_times) == tick_count and not model_errors
    final_tracking = (
        max(
            float(
                np.max(
                    np.abs(
                        measured_qpos[side][-1] - qcmd_samples[side][-1]
                    )
                )
            )
            for side in SIDES
        )
        if completed
        else None
    )
    max_active_offset = _finite_max(active_offsets)
    max_active_slew = _finite_max(active_slew_ratios)
    final_active_offset = active_offsets[-1] if active_offsets else None
    max_active_rotation = _finite_max(active_rotation_offsets)
    max_active_rotation_slew = _finite_max(active_rotation_slew_ratios)
    final_active_rotation = (
        active_rotation_offsets[-1] if active_rotation_offsets else None
    )
    qcmd_dynamics = _qcmd_dynamics(
        times_sec=qcmd_command_times,
        qcmd_by_side=qcmd_command_samples,
        nominal_sample_hz=timing.command_hz,
    )
    report = {
        "name": name,
        "active": active,
        "mode": mode,
        "completed": completed,
        "expected_reference_ticks": tick_count,
        "completed_reference_ticks": len(sample_times),
        "physics_ticks": int(rig.snapshot()["physics_ticks"]),
        "model_errors": model_errors,
        "endpoint_excursion_guard": (
            None
            if endpoint_excursion_guard is None
            else endpoint_excursion_guard.snapshot()
        ),
        "tool_endpoint_excursion_m": _stats(endpoint_excursions_m),
        "dispatched_tool_endpoint_excursion_m": _stats(
            endpoint_dispatched_excursions_m
        ),
        "tool_rotation_excursion_deg": _stats(tool_rotation_excursions_deg),
        "frame_check": frame_check,
        "initial_collision": initial_collision.to_dict(),
        "timing_jitter_sec": _stats(timing_jitter_sec),
        "qref_step_ratio": _stats(qref_step_ratios),
        "qcmd_step_ratio": _stats(qcmd_step_ratios),
        "qcmd_velocity_ratio": _stats(qcmd_velocity_ratios),
        "qcmd_acceleration_ratio": _stats(qcmd_acceleration_ratios),
        "qcmd_tracking_error_rad": _stats(qcmd_tracking_errors),
        "final_qcmd_tracking_error_rad": final_tracking,
        "qcmd_dynamics_120hz": qcmd_dynamics,
        "ik_execution_state_sync": {
            "source": "transport_accepted_qcmd",
            "count": ik_execution_sync_count,
            "qcmd_to_previous_qref_rad": _stats(
                ik_execution_sync_qcmd_to_previous_qref
            ),
        },
        "fixed_side_qref_delta_rad": _stats(fixed_qref_delta),
        "joint_limit_margin_rad": _stats(joint_limit_margin_rad),
        "minimum_joint_limit_margin_rad": (
            min(joint_limit_margin_rad) if joint_limit_margin_rad else None
        ),
        "preventive_clearance_m": _stats(preventive_clearance_m),
        "minimum_preventive_clearance_m": (
            min(preventive_clearance_m) if preventive_clearance_m else None
        ),
        "preventive_unsafe_reference_ticks": preventive_unsafe_ticks,
        "preventive_unsafe_examples": preventive_unsafe_examples,
        "hard_contact_minimum_m": (
            None
            if not math.isfinite(hard_contact_minimum_m)
            else hard_contact_minimum_m
        ),
        "hard_contact_faults": hard_contact_faults,
        "active_offset_norm_m": _stats(active_offsets),
        "final_active_offset_m": final_active_offset,
        "active_slew_ratio": _stats(active_slew_ratios),
        "active_return_slew_ratio": _stats(active_return_slew_ratios),
        "active_standard_slew_ratio": _stats(active_standard_slew_ratios),
        "passive_return_direction_mismatch_ticks": int(
            passive_return_direction_mismatch_ticks
        ),
        "active_rotation_offset_norm_rad": _stats(active_rotation_offsets),
        "final_active_rotation_offset_rad": final_active_rotation,
        "active_rotation_slew_ratio": _stats(active_rotation_slew_ratios),
        "nonzero_modifier_ticks": nonzero_modifier_ticks,
        "active_qref_motion_ticks": active_qref_motion_ticks,
        "production_rotational_active_installer": bool(
            production_rotational_active_installer
        ),
        "admission_feedback_frame_counts": dict(feedback_frame_counts),
        "admission_feedback_state_update_count": feedback_state_update_count,
        "maximum_feedback_anchor_norm_m": max_feedback_anchor_norm_m,
        "branch_counts": dict(branch_counts),
        "selected_side_branch_counts": dict(selected_side_branch_counts),
        "active_rotation_motion_step_floor_deg": (
            MUJOCO_ROTATION_MOTION_STEP_FLOOR_DEG
        ),
        "active_rotation_motion_qref_step_deg": _stats(
            active_rotation_motion_qref_steps_deg
        ),
        "active_rotation_motion_qref_hold_ticks": int(
            active_rotation_motion_qref_hold_ticks
        ),
        "active_rotation_motion_qref_hold_fraction": (
            None
            if not active_rotation_motion_qref_steps_deg
            else float(
                active_rotation_motion_qref_hold_ticks
                / len(active_rotation_motion_qref_steps_deg)
            )
        ),
        "collision_status_counts": dict(status_counts),
        "admittance_micro_reversal_hold_ticks_by_side": {
            side: int(admittance_micro_reversal_hold_ticks[side])
            for side in SIDES
        },
        "admittance_micro_reversal_held_joints_by_side": {
            side: int(admittance_micro_reversal_held_joints[side])
            for side in SIDES
        },
        "admittance_micro_reversal_soften_ticks_by_side": {
            side: int(admittance_micro_reversal_soften_ticks[side])
            for side in SIDES
        },
        "admittance_micro_reversal_softened_joints_by_side": {
            side: int(admittance_micro_reversal_softened_joints[side])
            for side in SIDES
        },
        "checks": {
            "completed": completed,
            "frame_contract": bool(frame_check["passed"]),
            "state_finite_and_hard_contact_safe": bool(
                completed and not hard_contact_faults
            ),
            "preventive_clearance": bool(
                completed and preventive_unsafe_ticks == 0
            ),
            "joint_limits": bool(
                joint_limit_margin_rad and min(joint_limit_margin_rad) >= -1e-9
            ),
            "formal_qref_step": bool(
                qref_step_ratios and max(qref_step_ratios) <= 1.0 + 1e-6
            ),
            "formal_qcmd_step": bool(
                qcmd_step_ratios and max(qcmd_step_ratios) <= 1.0 + 1e-6
            ),
            "formal_qcmd_velocity": bool(
                qcmd_velocity_ratios
                and max(qcmd_velocity_ratios) <= 1.0 + 1e-6
            ),
            "formal_qcmd_acceleration": bool(
                qcmd_acceleration_ratios
                and max(qcmd_acceleration_ratios) <= 1.0 + 1e-5
            ),
            "final_physical_settle": bool(
                final_tracking is not None
                and final_tracking <= MUJOCO_FINAL_QCMD_TRACKING_LIMIT_RAD
            ),
            "active_final_envelope": bool(
                not active
                or (
                    max_active_offset is not None
                    and max_active_offset
                    <= max(config.max_offset_m) + 1e-9
                    and max_active_slew is not None
                    and max_active_slew <= 1.0 + 1e-6
                    and final_active_offset is not None
                    and final_active_offset <= 1e-5
                    and set(feedback_frame_counts) == {"solver_world"}
                    and feedback_state_update_count == 0
                    and max_feedback_anchor_norm_m <= 1e-12
                    and (
                        rotational_config is None
                        or (
                            max_active_rotation is not None
                            and max_active_rotation
                            <= max(rotational_config.max_offset_rad) + 1e-9
                            and max_active_rotation_slew is not None
                            and max_active_rotation_slew <= 1.0 + 1e-6
                            and final_active_rotation is not None
                            and final_active_rotation <= 1e-5
                        )
                    )
                )
            ),
            "active_tool_endpoint_excursion": bool(
                not active
                or not position_admittance_active_uses_home_endpoint_guard(
                    endpoint_policy
                )
                or max_tool_excursion_m is None
                or (
                    endpoint_dispatched_excursions_m
                    and max(endpoint_dispatched_excursions_m)
                    <= float(max_tool_excursion_m) + 1e-12
                    and endpoint_excursion_guard is not None
                    and endpoint_excursion_guard.snapshot()["last_check_error"]
                    is None
                )
            ),
            "active_tool_rotation_excursion": bool(
                not active
                or not position_admittance_active_uses_home_endpoint_guard(
                    endpoint_policy
                )
                or max_tool_rotation_excursion_deg is None
                or (
                    tool_rotation_excursions_deg
                    and max(tool_rotation_excursions_deg)
                    <= float(max_tool_rotation_excursion_deg) + 1e-12
                )
            ),
        },
        "execution": {
            "position_profile": position_execution_profile.name,
            "position_command_planner": execution.planner,
            "admittance_envelope": admittance_envelope,
            "active_endpoint_policy": endpoint_policy,
            "active_modifier_qpos_fast_path": {
                "enabled": bool(
                    retargeter.config.output_qpos_admittance_fast_path_enabled
                ),
                "configured_alpha_cap": float(
                    retargeter.config.output_qpos_admittance_fast_path_alpha
                ),
                "micro_reversal_hold_deg": float(
                    retargeter.config.output_qpos_admittance_micro_reversal_hold_deg
                ),
                "micro_reversal_creep_deg": float(
                    retargeter.config.output_qpos_admittance_micro_reversal_creep_deg
                ),
                "requested_alpha": (
                    None
                    if observer is None
                    else float(observer.qpos_fast_path_alpha)
                ),
                "rotation_release_requested_alpha": (
                    None
                    if observer is None or rotational_config is None
                    else float(
                        observer.rotation_release_qpos_fast_path_alpha
                    )
                ),
                "translation_release_requested_alpha": (
                    None
                    if observer is None
                    or rotational_config is None
                    or not any(config.enabled_axes)
                    else float(
                        POSITION_ADMITTANCE_TRANSLATION_RELEASE_QPOS_FAST_PATH_ALPHA
                    )
                ),
                "eligibility": "modifier_nonzero_plus_hard_solver_guards",
            },
            "execution_mode": "actuated_position_servo",
            "robot_dynamics": "mujoco_position_actuator",
            "arm_actuation_profile": arm_profile.name,
            "arm_actuation_calibration_status": arm_profile.calibration_status,
            "gripper_actuation_profile": gripper_profile.name,
            "rates_hz": {
                "saved_target_nominal": 15.0,
                "reference_ik": timing.ik_hz,
                "command": timing.command_hz,
                "physics": timing.physics_hz,
            },
        },
    }
    return _PhysicalRollout(
        name=name,
        report=report,
        sample_times_sec=sample_times,
        measured_qpos=measured_qpos,
        measured_pose=measured_pose,
        qref=qref_samples,
        qcmd=qcmd_samples,
        raw_target_pose=raw_target_pose_samples,
        applied_target_pose=applied_target_pose_samples,
        filtered_force_control_n=filtered_force_samples,
        active_offset_m=active_offset_samples,
        filtered_torque_control_nm=filtered_torque_samples,
        active_rotation_rotvec_rad=active_rotation_samples,
    )


def _compare_rollouts(
    *,
    baseline: _PhysicalRollout,
    candidate: _PhysicalRollout,
    selected_side: str,
    incremental_translation_limit_m: float,
    incremental_rotation_limit_deg: float,
    release_force_deadband_n: Sequence[float],
    release_torque_deadband_nm: Sequence[float] | None = None,
    force_release_baseline: _PhysicalRollout | None = None,
    torque_release_baseline: _PhysicalRollout | None = None,
) -> dict[str, Any]:
    if baseline.sample_times_sec != candidate.sample_times_sec:
        return {
            "passed": False,
            "safety_passed": False,
            "error": "baseline and Active MuJoCo sample clocks differ",
        }
    release_clock_mismatches = [
        {
            "baseline": label,
            "baseline_samples": len(rollout.sample_times_sec),
            "candidate_samples": len(candidate.sample_times_sec),
        }
        for label, rollout in (
            ("force_release_baseline", force_release_baseline),
            ("torque_release_baseline", torque_release_baseline),
        )
        if rollout is not None
        and rollout.sample_times_sec != candidate.sample_times_sec
    ]
    if release_clock_mismatches:
        return {
            "passed": False,
            "safety_passed": False,
            "error": "component release baseline and Active MuJoCo sample clocks differ",
            "sample_clock_mismatches": release_clock_mismatches,
        }
    opposite_side = "right" if selected_side == "left" else "left"
    selected_translation: list[float] = []
    selected_rotation: list[float] = []
    opposite_translation: list[float] = []
    opposite_rotation: list[float] = []
    selected_qpos: list[float] = []
    opposite_qpos: list[float] = []
    selected_qref: list[float] = []
    opposite_qref: list[float] = []
    selected_qcmd: list[float] = []
    opposite_qcmd: list[float] = []
    for index in range(len(baseline.sample_times_sec)):
        for side, translation, rotation, qpos, qref, qcmd in (
            (
                selected_side,
                selected_translation,
                selected_rotation,
                selected_qpos,
                selected_qref,
                selected_qcmd,
            ),
            (
                opposite_side,
                opposite_translation,
                opposite_rotation,
                opposite_qpos,
                opposite_qref,
                opposite_qcmd,
            ),
        ):
            baseline_pose = baseline.measured_pose[side][index]
            candidate_pose = candidate.measured_pose[side][index]
            translation.append(
                float(
                    np.linalg.norm(
                        candidate_pose[:3, 3] - baseline_pose[:3, 3]
                    )
                )
            )
            rotation.append(
                _rotation_error_deg(
                    baseline_pose[:3, :3], candidate_pose[:3, :3]
                )
            )
            qpos.append(
                float(
                    np.max(
                        np.abs(
                            candidate.measured_qpos[side][index]
                            - baseline.measured_qpos[side][index]
                        )
                    )
                )
            )
            qref.append(
                float(
                    np.max(
                        np.abs(
                            candidate.qref[side][index]
                            - baseline.qref[side][index]
                        )
                    )
                )
            )
            qcmd.append(
                float(
                    np.max(
                        np.abs(
                            candidate.qcmd[side][index]
                            - baseline.qcmd[side][index]
                        )
                    )
                )
            )
    completed = bool(
        baseline.report["completed"]
        and candidate.report["completed"]
        and selected_translation
    )
    final_selected_translation = (
        selected_translation[-1] if selected_translation else None
    )
    final_selected_rotation = selected_rotation[-1] if selected_rotation else None
    max_selected_translation = _finite_max(selected_translation)
    max_selected_rotation = _finite_max(selected_rotation)
    max_opposite_qpos = _finite_max(opposite_qpos)
    opposite_raw_target_translation_m = [
        float(
            np.linalg.norm(
                pose[:3, 3]
                - baseline.raw_target_pose[opposite_side][0][:3, 3]
            )
        )
        for pose in baseline.raw_target_pose[opposite_side]
    ]
    opposite_raw_target_rotation_deg = [
        _rotation_error_deg(
            baseline.raw_target_pose[opposite_side][0][:3, :3],
            pose[:3, :3],
        )
        for pose in baseline.raw_target_pose[opposite_side]
    ]
    max_opposite_raw_target_translation_m = _finite_max(
        opposite_raw_target_translation_m
    )
    max_opposite_raw_target_rotation_deg = _finite_max(
        opposite_raw_target_rotation_deg
    )
    opposite_fixed_gate_applicable = bool(
        max_opposite_raw_target_translation_m is not None
        and max_opposite_raw_target_translation_m
        <= MUJOCO_FRAME_TRANSLATION_TOLERANCE_M
        and max_opposite_raw_target_rotation_deg is not None
        and max_opposite_raw_target_rotation_deg
        <= MUJOCO_FRAME_ROTATION_TOLERANCE_DEG
    )
    nonzero_physical_ticks = sum(
        translation > 1e-6 or rotation > 1e-4
        for translation, rotation in zip(
            selected_translation,
            selected_rotation,
            strict=True,
        )
    )
    release_transients = _force_release_transients(
        baseline=(
            baseline
            if force_release_baseline is None
            else force_release_baseline
        ),
        candidate=candidate,
        selected_side=selected_side,
        release_force_deadband_n=release_force_deadband_n,
    )
    translation_modifier_present = any(
        float(np.linalg.norm(value)) > 1e-9
        for value in candidate.active_offset_m
    )
    rotation_modifier_present = any(
        float(np.linalg.norm(value)) > 1e-9
        for value in candidate.active_rotation_rotvec_rad
    )
    rotation_release_transients = (
        None
        if release_torque_deadband_nm is None
        else _torque_release_transients(
            baseline=(
                baseline
                if torque_release_baseline is None
                else torque_release_baseline
            ),
            candidate=candidate,
            selected_side=selected_side,
            release_torque_deadband_nm=release_torque_deadband_nm,
        )
    )
    checks = {
        "system_level_increment_bounded": bool(
            completed
            and max_selected_translation is not None
            and max_selected_translation
            <= incremental_translation_limit_m
            and max_selected_rotation is not None
            and max_selected_rotation <= incremental_rotation_limit_deg
        ),
        "opposite_side_counterfactual_fixed": bool(
            completed
            and (
                not opposite_fixed_gate_applicable
                or (
                    max_opposite_qpos is not None
                    and max_opposite_qpos
                    <= MUJOCO_FIXED_SIDE_COUNTERFACTUAL_LIMIT_RAD
                )
            )
        ),
        "zero_force_release_rejoins_baseline": bool(
            completed
            and final_selected_translation is not None
            and final_selected_translation
            <= MUJOCO_RELEASE_TRANSLATION_LIMIT_M
            and final_selected_rotation is not None
            and final_selected_rotation <= MUJOCO_RELEASE_ROTATION_LIMIT_DEG
        ),
        "force_release_no_delayed_outward_catchup": bool(
            completed
            and (
                release_transients["passed"]
                if translation_modifier_present
                else True
            )
        ),
        "torque_release_no_delayed_outward_catchup": bool(
            completed
            and (
                True
                if (
                    rotation_release_transients is None
                    or not rotation_modifier_present
                )
                else rotation_release_transients["passed"]
            )
        ),
        "nonzero_active_physical_coverage": bool(
            candidate.report["nonzero_modifier_ticks"] > 0
            and candidate.report["active_qref_motion_ticks"] > 0
            and nonzero_physical_ticks > 0
        ),
    }
    safety_checks = {
        key: value
        for key, value in checks.items()
        if key != "nonzero_active_physical_coverage"
    }
    return {
        "passed": all(checks.values()),
        "safety_passed": all(safety_checks.values()),
        "checks": checks,
        "nonzero_active_physical_ticks": nonzero_physical_ticks,
        "selected_measured_translation_m": _stats(selected_translation),
        "selected_measured_rotation_deg": _stats(selected_rotation),
        "selected_measured_qpos_delta_rad": _stats(selected_qpos),
        "selected_qref_delta_rad": _stats(selected_qref),
        "selected_qcmd_delta_rad": _stats(selected_qcmd),
        "final_selected_measured_translation_m": final_selected_translation,
        "final_selected_measured_rotation_deg": final_selected_rotation,
        "final_selected_measured_qpos_delta_rad": (
            selected_qpos[-1] if selected_qpos else None
        ),
        "final_selected_qref_delta_rad": (
            selected_qref[-1] if selected_qref else None
        ),
        "final_selected_qcmd_delta_rad": (
            selected_qcmd[-1] if selected_qcmd else None
        ),
        "opposite_measured_translation_m": _stats(opposite_translation),
        "opposite_measured_rotation_deg": _stats(opposite_rotation),
        "opposite_measured_qpos_delta_rad": _stats(opposite_qpos),
        "opposite_qref_delta_rad": _stats(opposite_qref),
        "opposite_qcmd_delta_rad": _stats(opposite_qcmd),
        "opposite_fixed_gate_applicable": opposite_fixed_gate_applicable,
        "opposite_raw_target_translation_m": _stats(
            opposite_raw_target_translation_m
        ),
        "opposite_raw_target_rotation_deg": _stats(
            opposite_raw_target_rotation_deg
        ),
        "force_release_transients": release_transients,
        "torque_release_transients": rotation_release_transients,
        "release_gate_applicability": {
            "force": bool(translation_modifier_present),
            "torque": bool(
                release_torque_deadband_nm is not None
                and rotation_modifier_present
            ),
        },
        "release_counterfactual_baselines": {
            "force": (
                baseline.name
                if force_release_baseline is None
                else force_release_baseline.name
            ),
            "torque": (
                None
                if release_torque_deadband_nm is None
                else baseline.name
                if torque_release_baseline is None
                else torque_release_baseline.name
            ),
        },
        "limits": {
            "incremental_translation_m": incremental_translation_limit_m,
            "incremental_rotation_deg": incremental_rotation_limit_deg,
            "release_translation_m": MUJOCO_RELEASE_TRANSLATION_LIMIT_M,
            "release_qref_contract_m": MUJOCO_RELEASE_QREF_LIMIT_M,
            "release_physical_margin_m": MUJOCO_RELEASE_PHYSICAL_MARGIN_M,
            "release_rotation_deg": MUJOCO_RELEASE_ROTATION_LIMIT_DEG,
            "force_release_catchup_m": MUJOCO_FORCE_RELEASE_CATCHUP_LIMIT_M,
            "force_release_catchup_duration_sec": (
                MUJOCO_FORCE_RELEASE_CATCHUP_DURATION_SEC
            ),
            "force_release_duration_min_catchup_m": (
                MUJOCO_FORCE_RELEASE_DURATION_MIN_CATCHUP_M
            ),
            "force_release_observation_sec": (
                MUJOCO_FORCE_RELEASE_OBSERVATION_SEC
            ),
            "torque_release_catchup_deg": (
                MUJOCO_TORQUE_RELEASE_CATCHUP_LIMIT_DEG
            ),
            "torque_release_catchup_duration_sec": (
                MUJOCO_TORQUE_RELEASE_CATCHUP_DURATION_SEC
            ),
            "torque_release_duration_min_catchup_deg": (
                MUJOCO_TORQUE_RELEASE_DURATION_MIN_CATCHUP_DEG
            ),
            "torque_release_observation_sec": (
                MUJOCO_TORQUE_RELEASE_OBSERVATION_SEC
            ),
            "opposite_qpos_rad": MUJOCO_FIXED_SIDE_COUNTERFACTUAL_LIMIT_RAD,
        },
    }


def _recorded_input_contract(
    *,
    receipt: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    rotational_replay: bool,
) -> tuple[bool, tuple[str, ...]]:
    """Validate the saved-input provenance accepted by an offline replay.

    Translation Active development still requires a non-applying shadow log.
    Rotational replay may additionally consume an exact ``rotation_active``
    record because the log retains the pre-modifier raw VIVE target and raw
    wrench.  The force remains open-loop replay evidence, as declared in the
    report limitations; accepting its provenance does not promote it to a
    counterfactual contact measurement.
    """

    accepted_modes = (
        ("shadow", "rotation_active", "combined_active")
        if rotational_replay
        else ("shadow",)
    )
    passed = bool(
        receipt
        and receipt.get("enabled") is True
        and receipt.get("requested_mode") in accepted_modes
        and str(
            _mapping(config_payload.get("position_control_profile")).get(
                "name"
            )
        )
        in {
            RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
            RM65_BIMANUAL_POSITION_V2_BALANCED_60HZ_JERK_V1_NAME,
            RM65_BIMANUAL_POSITION_V2_BALANCED_60HZ_JERK_V2_NAME,
        }
    )
    return passed, accepted_modes


def _replay_recorded_wrench_through_mujoco_evaluation(
    path: Path,
    *,
    side: str = "left",
    admittance_envelope: str = "cautious-active",
    replay_hz: float = DEFAULT_REPLAY_HZ,
    release_tail_sec: float = 10.0,
    interpolation_modes: Sequence[str] = ("zoh", "linear"),
    left_tool_profile: Path = Path(DEFAULT_LEFT_TOOL_PROFILE),
    right_tool_profile: Path = Path(DEFAULT_RIGHT_TOOL_PROFILE),
    simulation_contract_path: Path = DEFAULT_SIMULATION_CONTRACT_PATH,
    max_tool_excursion_m: float | None = None,
    rotational_config: RotationalAdmittanceConfig | None = None,
    max_tool_rotation_excursion_deg: float | None = None,
    combined_translation_rotation_active: bool = False,
    endpoint_policy: str = POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
    position_execution_profile_name: str = RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
    translation_filter_tau_sec: float | None = None,
    rotation_filter_tau_sec: float | None = None,
    translation_output_follow_tau_sec: float | None = None,
    rotation_output_follow_tau_sec: float | None = None,
    translation_max_output_acceleration_m_s2: float | None = None,
    rotation_max_output_angular_acceleration_rad_s2: float | None = None,
) -> tuple[dict[str, Any], dict[str, _PhysicalRollout]]:
    """Run both physical branches and retain the trajectory evidence."""

    selected_side = str(side).strip().lower()
    if selected_side not in SIDES:
        raise ValueError("side must be left or right")
    if not math.isfinite(replay_hz) or replay_hz <= 0.0:
        raise ValueError("replay_hz must be positive and finite")
    if not math.isclose(replay_hz, 60.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("the current Active and V2 Balanced IK replay requires 60 Hz")
    if not math.isfinite(release_tail_sec) or release_tail_sec <= 0.0:
        raise ValueError("release_tail_sec must be positive and finite")
    if max_tool_excursion_m is not None and (
        not math.isfinite(float(max_tool_excursion_m))
        or float(max_tool_excursion_m) <= 0.0
    ):
        raise ValueError("max_tool_excursion_m must be positive and finite")
    if max_tool_rotation_excursion_deg is not None and (
        not math.isfinite(float(max_tool_rotation_excursion_deg))
        or float(max_tool_rotation_excursion_deg) <= 0.0
    ):
        raise ValueError(
            "max_tool_rotation_excursion_deg must be positive and finite"
        )
    if rotational_config is not None and max_tool_rotation_excursion_deg is None:
        raise ValueError(
            "rotational replay requires max_tool_rotation_excursion_deg"
        )
    if combined_translation_rotation_active and rotational_config is None:
        raise ValueError(
            "combined translation and rotation replay requires rotational_config"
        )
    for name, value in (
        ("translation_filter_tau_sec", translation_filter_tau_sec),
        ("rotation_filter_tau_sec", rotation_filter_tau_sec),
        ("translation_output_follow_tau_sec", translation_output_follow_tau_sec),
        ("rotation_output_follow_tau_sec", rotation_output_follow_tau_sec),
    ):
        if value is not None and (
            not math.isfinite(float(value))
            or float(value) < 0.0
            or float(value)
            > POSITION_ADMITTANCE_ACTIVE_OUTPUT_FOLLOW_TAU_SEC_MAX + 1e-12
        ):
            raise ValueError(
                f"{name} must be finite and within [0, "
                f"{POSITION_ADMITTANCE_ACTIVE_OUTPUT_FOLLOW_TAU_SEC_MAX:g}]"
            )
    if (
        rotation_filter_tau_sec is not None
        or rotation_output_follow_tau_sec is not None
    ) and rotational_config is None:
        raise ValueError(
            "rotation filter/output tau requires rotational_config"
        )
    for name, value in (
        (
            "translation_max_output_acceleration_m_s2",
            translation_max_output_acceleration_m_s2,
        ),
        (
            "rotation_max_output_angular_acceleration_rad_s2",
            rotation_max_output_angular_acceleration_rad_s2,
        ),
    ):
        if value is not None and (
            not math.isfinite(float(value)) or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    if (
        rotation_max_output_angular_acceleration_rad_s2 is not None
        and rotational_config is None
    ):
        raise ValueError(
            "rotation_max_output_angular_acceleration_rad_s2 requires "
            "rotational_config"
        )
    position_admittance_active_uses_home_endpoint_guard(endpoint_policy)
    modes = tuple(str(mode).strip().lower() for mode in interpolation_modes)
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"zoh", "linear"}:
        raise ValueError("interpolation_modes must be a unique subset of zoh/linear")
    envelope = str(admittance_envelope).strip().lower()
    if envelope not in ADMITTANCE_ENVELOPES:
        raise ValueError(
            "admittance_envelope must be cautious-active, recorded-shadow, "
            "normal-parity-offline, or normal-parity-active"
        )
    if (
        combined_translation_rotation_active
        and envelope
        not in {"cautious-active", PRODUCTION_NORMAL_PARITY_ENVELOPE}
    ):
        raise ValueError(
            "production combined Active replay requires cautious-active or "
            "normal-parity-active; use normal-parity-offline only for the "
            "device-free counterfactual"
        )
    if (
        endpoint_policy == POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP
        and (
            envelope != PRODUCTION_NORMAL_PARITY_ENVELOPE
            or not combined_translation_rotation_active
            or rotational_config is None
        )
    ):
        raise ValueError(
            "formal_teleop replay requires production combined normal-parity "
            "translation and rotation Active"
        )

    log_path = _resolve_log(Path(path))
    rows = _read_mujoco_rows(log_path)
    recorded_frames = _frame_rows(rows)
    frames: list[Mapping[str, Any]] = []
    cached_snapshot_frames = 0
    previous_observed_at: float | None = None
    previous_admittance: object = None
    for frame_index, frame in enumerate(recorded_frames):
        left_payload = _frame_payload(frame, side="left")
        observed_at = float(left_payload.get("observed_at_monotonic_sec"))
        admittance_snapshot = frame.get("position_admittance")
        if (
            previous_observed_at is not None
            and observed_at == previous_observed_at
        ):
            if admittance_snapshot != previous_admittance:
                raise ValueError(
                    "cached frame changed position admittance without a newer "
                    f"observation at frame {frame_index}"
                )
            cached_snapshot_frames += 1
            continue
        frames.append(frame)
        previous_observed_at = observed_at
        previous_admittance = admittance_snapshot
    if not frames:
        raise ValueError("MuJoCo replay requires action-debug frames")
    receipt = _position_admittance_receipt(rows)
    recorded_config = _mapping(receipt.get("config"))
    recorded_admittance_config = CartesianAdmittanceConfig(
        **dict(recorded_config)
    )
    config = _admittance_config_for_envelope(
        recorded_admittance_config,
        admittance_envelope=envelope,
    )
    if translation_filter_tau_sec is not None:
        config = replace(
            config,
            force_filter_tau_sec=float(translation_filter_tau_sec),
        )
    if translation_output_follow_tau_sec is not None:
        config = replace(
            config,
            output_follow_tau_sec=float(translation_output_follow_tau_sec),
        )
    if translation_max_output_acceleration_m_s2 is not None:
        config = replace(
            config,
            max_output_acceleration_m_s2=(
                float(translation_max_output_acceleration_m_s2),
            )
            * 3,
        )
    if rotational_config is not None:
        if rotation_filter_tau_sec is not None:
            rotational_config = replace(
                rotational_config,
                torque_filter_tau_sec=float(rotation_filter_tau_sec),
            )
        if rotation_output_follow_tau_sec is not None:
            rotational_config = replace(
                rotational_config,
                output_follow_tau_sec=float(rotation_output_follow_tau_sec),
            )
        if rotation_max_output_angular_acceleration_rad_s2 is not None:
            rotational_config = replace(
                rotational_config,
                max_output_angular_acceleration_rad_s2=(
                    float(rotation_max_output_angular_acceleration_rad_s2),
                )
                * 3,
            )
        rotational_config_is_production = True
        if envelope == PRODUCTION_NORMAL_PARITY_ENVELOPE:
            validate_position_admittance_active_authority(
                config,
                active_envelope=(
                    POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY
                ),
                rotational_config=rotational_config,
                combined_translation_rotation_active=True,
                endpoint_policy=endpoint_policy,
                max_tool_excursion_m=max_tool_excursion_m,
                max_tool_rotation_excursion_deg=(
                    max_tool_rotation_excursion_deg
                ),
            )
        else:
            try:
                validate_position_rotational_admittance_active_config(
                    rotational_config
                )
            except ValueError:
                rotational_config_is_production = False
                if combined_translation_rotation_active:
                    raise ValueError(
                        "combined translation and rotation replay requires "
                        "the production cautious-3deg rotational config or "
                        "an admitted normal-parity-active 5-degree config"
                    )
        if combined_translation_rotation_active:
            # Production rotation-only replay preserves its established
            # translation-disabled semantics.  The explicit combined branch
            # retains the cautious translation axes and therefore exercises
            # the same installer used by the proposed real command.
            if envelope == "cautious-active":
                config = _cautious_active_config(config)
            # A rotation-only Active receipt records the post-resolution
            # disabled axes, even though the raw six-axis wrench and the
            # original command's xyz selection are retained.  The explicit
            # combined replay restores the proposed command's xyz axes;
            # no production or implicit replay path reaches this branch.
            config = replace(config, enabled_axes=(True, True, True))
        elif envelope == OFFLINE_NORMAL_PARITY_ENVELOPE:
            # Keep the explicitly reconstructed normal-parity xyz envelope.
            # Both cautious and larger SO(3) candidates use the replay-only
            # installer and cannot grant hardware authority.
            pass
        else:
            # Rotation-only and larger-candidate legacy semantics remain
            # translation-disabled.  A production-valid cautious candidate
            # still receives the production installer in _physical_rollout.
            if rotational_config_is_production:
                config = _cautious_active_config(config)
            config = replace(config, enabled_axes=(False, False, False))
    frame_times, raw_targets, tare_by_side, initial_source_times = (
        _recorded_inputs(frames)
    )
    config_payload = _config_payload(rows)
    left_tool_pose = _tool_pose(Path(left_tool_profile), side="left")
    right_tool_pose = _tool_pose(Path(right_tool_profile), side="right")
    sample_deltas = np.diff(np.asarray(frame_times, dtype=float))
    sampled_hz = (
        None
        if sample_deltas.size == 0
        else float(1.0 / np.median(sample_deltas))
    )

    rollouts: dict[str, Any] = {}
    physical_rollouts: dict[str, _PhysicalRollout] = {}
    comparisons: dict[str, Any] = {}
    component_release_isolation = bool(
        rotational_config is not None
        and envelope
        in {
            OFFLINE_NORMAL_PARITY_ENVELOPE,
            PRODUCTION_NORMAL_PARITY_ENVELOPE,
        }
    )
    for mode in modes:
        baseline = _physical_rollout(
            name=f"{mode}_baseline",
            active=False,
            mode=mode,
            frames=frames,
            frame_times=frame_times,
            raw_targets=raw_targets,
            tare_by_side=tare_by_side,
            initial_source_time_by_side=initial_source_times,
            selected_side=selected_side,
            config=config,
            rotational_config=rotational_config,
            combined_translation_rotation_active=(
                combined_translation_rotation_active
            ),
            admittance_envelope=envelope,
            config_payload=config_payload,
            replay_hz=float(replay_hz),
            release_tail_sec=float(release_tail_sec),
            left_tool_pose=left_tool_pose,
            right_tool_pose=right_tool_pose,
            simulation_contract_path=Path(simulation_contract_path),
            max_tool_excursion_m=None,
            max_tool_rotation_excursion_deg=None,
            endpoint_policy=endpoint_policy,
            position_execution_profile_name=position_execution_profile_name,
        )
        candidate = _physical_rollout(
            name=f"{mode}_active",
            active=True,
            mode=mode,
            frames=frames,
            frame_times=frame_times,
            raw_targets=raw_targets,
            tare_by_side=tare_by_side,
            initial_source_time_by_side=initial_source_times,
            selected_side=selected_side,
            config=config,
            rotational_config=rotational_config,
            combined_translation_rotation_active=(
                combined_translation_rotation_active
            ),
            admittance_envelope=envelope,
            config_payload=config_payload,
            replay_hz=float(replay_hz),
            release_tail_sec=float(release_tail_sec),
            left_tool_pose=left_tool_pose,
            right_tool_pose=right_tool_pose,
            simulation_contract_path=Path(simulation_contract_path),
            max_tool_excursion_m=max_tool_excursion_m,
            max_tool_rotation_excursion_deg=max_tool_rotation_excursion_deg,
            endpoint_policy=endpoint_policy,
            position_execution_profile_name=position_execution_profile_name,
        )
        force_release_baseline = None
        torque_release_baseline = None
        if component_release_isolation:
            # A no-modifier baseline cannot attribute delayed physical motion
            # to force or torque while both modifiers are active.  These two
            # extra device-free branches subtract the other modifier before
            # evaluating each release transient.
            translation_only = _physical_rollout(
                name=f"{mode}_translation_only",
                active=True,
                mode=mode,
                frames=frames,
                frame_times=frame_times,
                raw_targets=raw_targets,
                tare_by_side=tare_by_side,
                initial_source_time_by_side=initial_source_times,
                selected_side=selected_side,
                config=config,
                rotational_config=None,
                combined_translation_rotation_active=False,
                admittance_envelope=envelope,
                config_payload=config_payload,
                replay_hz=float(replay_hz),
                release_tail_sec=float(release_tail_sec),
                left_tool_pose=left_tool_pose,
                right_tool_pose=right_tool_pose,
                simulation_contract_path=Path(simulation_contract_path),
                max_tool_excursion_m=max_tool_excursion_m,
                max_tool_rotation_excursion_deg=None,
                endpoint_policy=endpoint_policy,
                position_execution_profile_name=position_execution_profile_name,
            )
            rotation_only = _physical_rollout(
                name=f"{mode}_rotation_only",
                active=True,
                mode=mode,
                frames=frames,
                frame_times=frame_times,
                raw_targets=raw_targets,
                tare_by_side=tare_by_side,
                initial_source_time_by_side=initial_source_times,
                selected_side=selected_side,
                config=replace(config, enabled_axes=(False, False, False)),
                rotational_config=rotational_config,
                combined_translation_rotation_active=False,
                admittance_envelope=envelope,
                config_payload=config_payload,
                replay_hz=float(replay_hz),
                release_tail_sec=float(release_tail_sec),
                left_tool_pose=left_tool_pose,
                right_tool_pose=right_tool_pose,
                simulation_contract_path=Path(simulation_contract_path),
                max_tool_excursion_m=None,
                max_tool_rotation_excursion_deg=max_tool_rotation_excursion_deg,
                endpoint_policy=endpoint_policy,
                position_execution_profile_name=position_execution_profile_name,
            )
            rollouts[translation_only.name] = translation_only.report
            rollouts[rotation_only.name] = rotation_only.report
            physical_rollouts[translation_only.name] = translation_only
            physical_rollouts[rotation_only.name] = rotation_only
            force_release_baseline = rotation_only
            torque_release_baseline = translation_only
        rollouts[baseline.name] = baseline.report
        rollouts[candidate.name] = candidate.report
        physical_rollouts[baseline.name] = baseline
        physical_rollouts[candidate.name] = candidate
        comparisons[mode] = _compare_rollouts(
            baseline=baseline,
            candidate=candidate,
            selected_side=selected_side,
            incremental_translation_limit_m=(
                max(config.max_offset_m) + MUJOCO_RELEASE_PHYSICAL_MARGIN_M
                if envelope
                in {
                    OFFLINE_NORMAL_PARITY_ENVELOPE,
                    PRODUCTION_NORMAL_PARITY_ENVELOPE,
                }
                else 0.005
                if rotational_config is not None
                else MUJOCO_INCREMENTAL_TRANSLATION_LIMIT_M
                if envelope == "cautious-active"
                else max(config.max_offset_m)
                + MUJOCO_RELEASE_PHYSICAL_MARGIN_M
            ),
            incremental_rotation_limit_deg=(
                float(max_tool_rotation_excursion_deg)
                if rotational_config is not None
                else MUJOCO_INCREMENTAL_ROTATION_LIMIT_DEG
                if envelope == "cautious-active"
                else POSITION_ADMITTANCE_ACTIVE_NOMINAL_ROTATION_LIMIT_DEG
            ),
            release_force_deadband_n=config.return_force_deadband_n,
            release_torque_deadband_nm=(
                None
                if rotational_config is None
                else rotational_config.return_torque_deadband_nm
            ),
            force_release_baseline=force_release_baseline,
            torque_release_baseline=torque_release_baseline,
        )

    input_contract_passed, accepted_recorded_modes = (
        _recorded_input_contract(
            receipt=receipt,
            config_payload=config_payload,
            rotational_replay=rotational_config is not None,
        )
    )
    rollout_checks = [
        bool(value)
        for rollout in rollouts.values()
        for value in _mapping(rollout.get("checks")).values()
    ]
    safety_passed = bool(
        input_contract_passed
        and rollout_checks
        and all(rollout_checks)
        and all(bool(value["safety_passed"]) for value in comparisons.values())
    )
    coverage_passed = all(
        bool(
            _mapping(value.get("checks")).get(
                "nonzero_active_physical_coverage",
                False,
            )
        )
        for value in comparisons.values()
    )
    production_rotation_active_exercised = bool(
        rotational_config is not None
        and all(
            bool(
                _mapping(rollouts.get(f"{mode}_active")).get(
                    "production_rotational_active_installer",
                    False,
                )
            )
            for mode in modes
        )
    )
    report = {
        "schema": (
            "dexteleop.position_rotational_admittance_mujoco_replay/v1"
            if rotational_config is not None
            else "dexteleop.position_admittance_mujoco_replay/v1"
        ),
        "passed": bool(safety_passed and coverage_passed),
        "safety_passed": safety_passed,
        "coverage_passed": coverage_passed,
        "evidence_level": "offline_saved_wrench_shared_ik_actuated_mujoco",
        "admittance_envelope": envelope,
        "hardware_authorization": False,
        "position_execution_profile": position_execution_profile_name,
        "production_rotational_active_installer_exercised": (
            production_rotation_active_exercised
        ),
        "combined_translation_rotation_active": bool(
            combined_translation_rotation_active
        ),
        "offline_normal_parity_counterfactual": bool(
            envelope == OFFLINE_NORMAL_PARITY_ENVELOPE
        ),
        "production_normal_parity_active": bool(
            envelope == PRODUCTION_NORMAL_PARITY_ENVELOPE
        ),
        "component_release_isolation": {
            "enabled": component_release_isolation,
            "force_baseline": (
                "rotation_only" if component_release_isolation else "baseline"
            ),
            "torque_baseline": (
                "translation_only" if component_release_isolation else "baseline"
            ),
            "meaning": (
                "release gates subtract the other simultaneously active modifier"
                if component_release_isolation
                else "release gates compare against the no-modifier baseline"
            ),
        },
        "path": str(log_path),
        "side": selected_side,
        "input": {
            "recorded_frame_count": len(recorded_frames),
            "frame_count": len(frames),
            "collapsed_cached_snapshot_frames": cached_snapshot_frames,
            "recorded_duration_sec": float(frame_times[-1] - frame_times[0]),
            "sampled_action_debug_hz_median": sampled_hz,
            "recorded_position_profile": _mapping(
                config_payload.get("position_control_profile")
            ).get("name"),
            "recorded_admittance_mode": receipt.get("requested_mode"),
            "accepted_recorded_admittance_modes": list(
                accepted_recorded_modes
            ),
            "contract_passed": input_contract_passed,
        },
        "recorded_config": recorded_config,
        "active_config": asdict(config),
        "active_rotational_config": (
            None if rotational_config is None else asdict(rotational_config)
        ),
        "simulation_contract": str(Path(simulation_contract_path)),
        "rollouts": rollouts,
        "comparisons": comparisons,
        "limitations": [
            "saved wrench is played back open-loop; MuJoCo contact force does not replace it",
            "saved target/wrench samples are reconstructed to 60 Hz with declared ZOH/linear modes",
            "the fitted RM65 actuator profile has no held-out validation for this trace",
            "MuJoCo force/contact and arm torque parameters remain provisional",
            (
                "saved torque is interpreted as a tool-frame pseudovector; "
                "the sensor moment-origin/TCP shift is not independently identified"
                if rotational_config is not None
                else "rotation admittance is disabled in this evaluation"
            ),
            "simulation evidence does not authorize hardware motion by itself",
            (
                "cautious rotation uses the production Active installer and "
                "runtime-Home endpoint guard with cautious translation axes enabled"
                if production_rotation_active_exercised
                and combined_translation_rotation_active
                else "cautious rotation uses the production Active installer and "
                "runtime-Home endpoint guard with translation axes disabled"
                if production_rotation_active_exercised
                else "recorded-shadow is an offline counterfactual and bypasses the "
                "production 2 mm / 5 mm/s Active admission envelope"
                if envelope == "recorded-shadow"
                else "normal-parity-active uses the production Active installer "
                "with formal teleop safety and without the temporary runtime-Home "
                "endpoint guard"
                if (
                    envelope == PRODUCTION_NORMAL_PARITY_ENVELOPE
                    and endpoint_policy
                    == POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP
                )
                else "normal-parity-active uses the production Active installer "
                "and runtime-Home 8 cm / bounded rotation endpoint guard"
                if envelope == PRODUCTION_NORMAL_PARITY_ENVELOPE
                else "normal-parity-offline reconstructs the 8 cm / 60 mm/s "
                "translation envelope only inside device-free MuJoCo and does not "
                "change production Active admission"
                if envelope == OFFLINE_NORMAL_PARITY_ENVELOPE
                else "cautious-active retains the production Active admission envelope"
            ),
        ],
    }
    return report, physical_rollouts


def replay_recorded_wrench_through_mujoco(
    path: Path,
    *,
    side: str = "left",
    admittance_envelope: str = "cautious-active",
    replay_hz: float = DEFAULT_REPLAY_HZ,
    release_tail_sec: float = 10.0,
    interpolation_modes: Sequence[str] = ("zoh", "linear"),
    left_tool_profile: Path = Path(DEFAULT_LEFT_TOOL_PROFILE),
    right_tool_profile: Path = Path(DEFAULT_RIGHT_TOOL_PROFILE),
    simulation_contract_path: Path = DEFAULT_SIMULATION_CONTRACT_PATH,
    max_tool_excursion_m: float | None = None,
    rotational_config: RotationalAdmittanceConfig | None = None,
    max_tool_rotation_excursion_deg: float | None = None,
    combined_translation_rotation_active: bool = False,
    endpoint_policy: str = POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
    position_execution_profile_name: str = RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
    translation_filter_tau_sec: float | None = None,
    rotation_filter_tau_sec: float | None = None,
    translation_output_follow_tau_sec: float | None = None,
    rotation_output_follow_tau_sec: float | None = None,
    translation_max_output_acceleration_m_s2: float | None = None,
    rotation_max_output_angular_acceleration_rad_s2: float | None = None,
) -> dict[str, Any]:
    """Run baseline and production-Active candidates through physical MuJoCo."""

    report, _ = _replay_recorded_wrench_through_mujoco_evaluation(
        path,
        side=side,
        admittance_envelope=admittance_envelope,
        replay_hz=replay_hz,
        release_tail_sec=release_tail_sec,
        interpolation_modes=interpolation_modes,
        left_tool_profile=left_tool_profile,
        right_tool_profile=right_tool_profile,
        simulation_contract_path=simulation_contract_path,
        max_tool_excursion_m=max_tool_excursion_m,
        rotational_config=rotational_config,
        max_tool_rotation_excursion_deg=max_tool_rotation_excursion_deg,
        combined_translation_rotation_active=(
            combined_translation_rotation_active
        ),
        endpoint_policy=endpoint_policy,
        position_execution_profile_name=position_execution_profile_name,
        translation_filter_tau_sec=translation_filter_tau_sec,
        rotation_filter_tau_sec=rotation_filter_tau_sec,
        translation_output_follow_tau_sec=(
            translation_output_follow_tau_sec
        ),
        rotation_output_follow_tau_sec=rotation_output_follow_tau_sec,
        translation_max_output_acceleration_m_s2=(
            translation_max_output_acceleration_m_s2
        ),
        rotation_max_output_angular_acceleration_rad_s2=(
            rotation_max_output_angular_acceleration_rad_s2
        ),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--side", choices=SIDES, default="left")
    parser.add_argument(
        "--admittance-envelope",
        choices=ADMITTANCE_ENVELOPES,
        default="cautious-active",
    )
    parser.add_argument("--replay-hz", type=float, default=DEFAULT_REPLAY_HZ)
    parser.add_argument("--release-tail-sec", type=float, default=10.0)
    parser.add_argument(
        "--interpolation",
        choices=("zoh", "linear", "both"),
        default="both",
    )
    parser.add_argument(
        "--left-tool-profile",
        type=Path,
        default=Path(DEFAULT_LEFT_TOOL_PROFILE),
    )
    parser.add_argument(
        "--right-tool-profile",
        type=Path,
        default=Path(DEFAULT_RIGHT_TOOL_PROFILE),
    )
    parser.add_argument(
        "--simulation-contract",
        type=Path,
        default=DEFAULT_SIMULATION_CONTRACT_PATH,
    )
    parser.add_argument(
        "--position-execution-profile",
        choices=tuple(
            value
            for value in RM65_POSITION_CONTROL_PROFILE_CHOICES
            if value != "legacy"
        ),
        default=RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
    )
    parser.add_argument(
        "--translation-output-follow-tau-sec",
        type=float,
        default=None,
        help="Optional offline/Active-valid 60 Hz translation output follower.",
    )
    parser.add_argument(
        "--translation-max-output-acceleration-m-s2",
        type=float,
        default=None,
        help="Optional 60 Hz translation modifier acceleration limit.",
    )
    args = parser.parse_args()
    modes = ("zoh", "linear") if args.interpolation == "both" else (args.interpolation,)
    report = replay_recorded_wrench_through_mujoco(
        args.log,
        side=args.side,
        admittance_envelope=args.admittance_envelope,
        replay_hz=args.replay_hz,
        release_tail_sec=args.release_tail_sec,
        interpolation_modes=modes,
        left_tool_profile=args.left_tool_profile,
        right_tool_profile=args.right_tool_profile,
        simulation_contract_path=args.simulation_contract,
        position_execution_profile_name=args.position_execution_profile,
        translation_output_follow_tau_sec=(
            args.translation_output_follow_tau_sec
        ),
        translation_max_output_acceleration_m_s2=(
            args.translation_max_output_acceleration_m_s2
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
