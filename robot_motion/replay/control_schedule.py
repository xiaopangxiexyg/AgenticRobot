from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from robot_motion.position_control.interpolation import (
    DEFAULT_POSITION_COMMAND_HZ,
    DEFAULT_POSITION_REFERENCE_HZ,
    JointCommandInterpolator,
)
from robot_motion.position_control.horizon_governor import JointHorizonLimits
from robot_motion.position_control.offline_waypoint_trajectory import (
    OFFLINE_WAYPOINT_TIMING_STRICT,
    build_offline_quintic_waypoint_trajectory,
)
from robot_motion.position_control.resampling import (
    TimestampedMotionTarget,
    build_uniform_time_grid,
    resample_motion_targets,
    validate_resample_rate,
)


DEFAULT_REPLAY_ACTION_HZ = 15.0
DEFAULT_REPLAY_REFERENCE_HZ = DEFAULT_POSITION_REFERENCE_HZ
DEFAULT_POSITION_QPOS_CONTROL_HZ = DEFAULT_POSITION_COMMAND_HZ
_TIME_EPSILON_SEC = 1e-12

REPLAY_QPOS_EXECUTION_LEGACY_LINEAR = "legacy_linear"
REPLAY_QPOS_EXECUTION_CONTINUOUS_QUINTIC = "continuous_quintic"
REPLAY_QPOS_EXECUTION_MODES = (
    REPLAY_QPOS_EXECUTION_LEGACY_LINEAR,
    REPLAY_QPOS_EXECUTION_CONTINUOUS_QUINTIC,
)


@dataclass(frozen=True)
class ReplayControlAction:
    action_index: int
    frame_index: int
    relative_time_sec: float
    qpos_by_side: Mapping[str, Sequence[float]]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayControlReference:
    """One high-rate reference produced from timestamped replay actions."""

    reference_index: int
    relative_time_sec: float
    action_index: int
    action_frame_index: int
    action_relative_time_sec: float
    next_action_index: int
    next_action_frame_index: int
    next_action_relative_time_sec: float
    action_interpolation_alpha: float
    qpos_by_side: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayControlTick:
    """One command tick emitted toward an active high-rate reference."""

    control_index: int
    action_index: int
    action_frame_index: int
    relative_time_sec: float
    action_relative_time_sec: float
    reference_index: int
    reference_relative_time_sec: float
    reference_action_interpolation_alpha: float
    segment_elapsed_sec: float
    segment_duration_sec: float
    interpolation_alpha: float
    qpos_by_side: dict[str, np.ndarray]
    target_qpos_by_side: dict[str, np.ndarray]
    reference_qpos_by_side: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_replay_rate(value: float, *, name: str) -> float:
    return validate_resample_rate(value, name=name)


def normalize_replay_qpos_execution_mode(value: str | None) -> str:
    mode = str(value or REPLAY_QPOS_EXECUTION_LEGACY_LINEAR).strip().lower()
    aliases = {
        "legacy": REPLAY_QPOS_EXECUTION_LEGACY_LINEAR,
        "linear": REPLAY_QPOS_EXECUTION_LEGACY_LINEAR,
        "continuous": REPLAY_QPOS_EXECUTION_CONTINUOUS_QUINTIC,
        "quintic": REPLAY_QPOS_EXECUTION_CONTINUOUS_QUINTIC,
        "c2": REPLAY_QPOS_EXECUTION_CONTINUOUS_QUINTIC,
    }
    mode = aliases.get(mode, mode)
    if mode not in REPLAY_QPOS_EXECUTION_MODES:
        raise ValueError(
            "qpos execution mode must be one of "
            f"{REPLAY_QPOS_EXECUTION_MODES}, got {value!r}."
        )
    return mode


def position_qpos_segment_duration(
    *,
    control_hz: float,
    reference_hz: float = DEFAULT_REPLAY_REFERENCE_HZ,
    segment_duration_sec: float | None = None,
) -> float:
    """Return the reference -> command interpolation duration.

    ``control_hz`` remains a required compatibility argument and is validated,
    but the default segment duration now has reference semantics: one high-rate
    period, rather than one 120 Hz command period.
    """

    validate_replay_rate(control_hz, name="control_hz")
    reference_rate = validate_replay_rate(reference_hz, name="reference_hz")
    if segment_duration_sec is None:
        return 1.0 / reference_rate
    duration = float(segment_duration_sec)
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("segment_duration_sec must be finite and >= 0.")
    return duration


def _copy_qpos_by_side(qpos_by_side: Mapping[str, Sequence[float]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for side, values in qpos_by_side.items():
        qpos = np.asarray(values, dtype=float).reshape(-1)
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            raise ValueError(f"{side} qpos must be a finite 6-vector.")
        out[str(side)] = qpos.astype(float, copy=True)
    if not out:
        raise ValueError("Replay control action must include at least one side qpos.")
    return out


def _normalize_actions(actions: Sequence[ReplayControlAction]) -> list[ReplayControlAction]:
    normalized: list[ReplayControlAction] = []
    last_time = -float("inf")
    seen_action_indices: set[int] = set()
    for fallback_index, action in enumerate(actions):
        relative_time = float(action.relative_time_sec)
        if not np.isfinite(relative_time):
            raise ValueError("Replay action relative_time_sec must be finite.")
        if relative_time <= last_time:
            raise ValueError("Replay action times must be strictly increasing.")
        action_index = int(action.action_index)
        if action_index < 0 or action_index in seen_action_indices:
            raise ValueError("Replay action indices must be unique non-negative integers.")
        seen_action_indices.add(action_index)
        last_time = relative_time
        metadata = dict(action.metadata or {})
        metadata.setdefault("source_index", int(fallback_index))
        normalized.append(
            ReplayControlAction(
                action_index=action_index,
                frame_index=int(action.frame_index),
                relative_time_sec=relative_time,
                qpos_by_side=_copy_qpos_by_side(action.qpos_by_side),
                metadata=metadata,
            )
        )
    if not normalized:
        raise ValueError("At least one replay action is required.")
    first_time = float(normalized[0].relative_time_sec)
    if abs(first_time) <= _TIME_EPSILON_SEC:
        return normalized
    return [
        ReplayControlAction(
            action_index=action.action_index,
            frame_index=action.frame_index,
            relative_time_sec=max(0.0, float(action.relative_time_sec) - first_time),
            qpos_by_side=action.qpos_by_side,
            metadata=action.metadata,
        )
        for action in normalized
    ]


def _command_time_grid_through(
    start_time: float,
    deadline: float,
    *,
    control_hz: float,
) -> tuple[float, ...]:
    """Return fixed-rate ticks through the first grid point at/after deadline."""

    rate = validate_replay_rate(control_hz, name="control_hz")
    span = max(0.0, float(deadline) - float(start_time))
    final_step = max(0, int(math.ceil(span * rate - _TIME_EPSILON_SEC)))
    grid_end = float(start_time) + float(final_step) / rate
    return build_uniform_time_grid(
        float(start_time),
        grid_end,
        hz=rate,
        include_final=False,
    )


def resample_position_qpos_references(
    actions: Sequence[ReplayControlAction],
    *,
    reference_hz: float = DEFAULT_REPLAY_REFERENCE_HZ,
) -> list[ReplayControlReference]:
    """Resample timestamped action qpos into an offline reference stream."""

    action_list = _normalize_actions(actions)
    reference_rate = validate_replay_rate(reference_hz, name="reference_hz")
    action_by_index = {action.action_index: action for action in action_list}
    targets = [
        TimestampedMotionTarget(
            timestamp=action.relative_time_sec,
            source_index=action.action_index,
            qpos_by_side=action.qpos_by_side,
            metadata={
                "action_index": int(action.action_index),
                "action_frame_index": int(action.frame_index),
                "action_metadata": dict(action.metadata or {}),
            },
        )
        for action in action_list
    ]
    resampled = resample_motion_targets(
        targets,
        reference_hz=reference_rate,
        include_final=True,
    )

    references: list[ReplayControlReference] = []
    for item in resampled:
        left_action = action_by_index[item.left_source_index]
        right_action = action_by_index[item.right_source_index]
        active_action = left_action
        metadata = dict(active_action.metadata or {})
        metadata.update(
            {
                "position_control_pipeline": "action_to_reference_to_command",
                "reference_hz": float(reference_rate),
                "reference_resampling": dict(item.metadata),
                "reference_left_action_index": int(left_action.action_index),
                "reference_left_action_frame_index": int(left_action.frame_index),
                "reference_right_action_index": int(right_action.action_index),
                "reference_right_action_frame_index": int(right_action.frame_index),
                "reference_action_interpolation_alpha": float(item.interpolation_alpha),
            }
        )
        references.append(
            ReplayControlReference(
                reference_index=int(item.sequence),
                relative_time_sec=float(item.timestamp),
                action_index=int(active_action.action_index),
                action_frame_index=int(active_action.frame_index),
                action_relative_time_sec=float(active_action.relative_time_sec),
                next_action_index=int(right_action.action_index),
                next_action_frame_index=int(right_action.frame_index),
                next_action_relative_time_sec=float(right_action.relative_time_sec),
                action_interpolation_alpha=float(item.interpolation_alpha),
                qpos_by_side=_copy_qpos_by_side(item.qpos_by_side),
                metadata=metadata,
            )
        )
    return references


def iter_position_qpos_control_ticks(
    actions: Sequence[ReplayControlAction],
    *,
    reference_hz: float = DEFAULT_REPLAY_REFERENCE_HZ,
    control_hz: float = DEFAULT_POSITION_QPOS_CONTROL_HZ,
    segment_duration_sec: float | None = None,
    include_final_settle: bool = True,
    execution_mode: str = REPLAY_QPOS_EXECUTION_LEGACY_LINEAR,
    continuous_limits: JointHorizonLimits | None = None,
    continuous_timing_mode: str = OFFLINE_WAYPOINT_TIMING_STRICT,
    continuous_max_time_scale: float = 4.0,
) -> list[ReplayControlTick]:
    """Build the offline action -> reference -> command path.

    Stage one uses actual action timestamps and offline interpolation. Stage two
    has the same online semantics as ``JointCommandInterpolator``: each newly
    due reference preempts from the last command that was actually emitted,
    then is approached over one reference period (or
    ``segment_duration_sec``). No unobserved theoretical point is used as a
    segment anchor, and no integer frequency ratio is assumed.
    """

    mode = normalize_replay_qpos_execution_mode(execution_mode)
    if mode == REPLAY_QPOS_EXECUTION_CONTINUOUS_QUINTIC:
        if segment_duration_sec is not None:
            raise ValueError(
                "segment_duration_sec is a legacy-linear option and cannot be "
                "combined with continuous_quintic."
            )
        if continuous_limits is None:
            raise ValueError(
                "continuous_quintic requires continuous_limits for a hard "
                "q/v/a/jerk certificate."
            )
        return _iter_continuous_position_qpos_control_ticks(
            actions,
            reference_hz=reference_hz,
            control_hz=control_hz,
            include_final_settle=include_final_settle,
            limits=continuous_limits,
            timing_mode=continuous_timing_mode,
            max_time_scale=continuous_max_time_scale,
        )

    reference_rate = validate_replay_rate(reference_hz, name="reference_hz")
    control_rate = validate_replay_rate(control_hz, name="control_hz")
    segment_duration = position_qpos_segment_duration(
        control_hz=control_rate,
        reference_hz=reference_rate,
        segment_duration_sec=segment_duration_sec,
    )
    references = resample_position_qpos_references(
        actions,
        reference_hz=reference_rate,
    )

    active = references[0]
    active_target = _copy_qpos_by_side(active.qpos_by_side)
    segment_start_time = float(active.relative_time_sec)
    next_reference_index = 0
    final_deadline = float(references[-1].relative_time_sec)
    if include_final_settle:
        final_deadline += segment_duration
    command_times = _command_time_grid_through(
        float(references[0].relative_time_sec),
        final_deadline,
        control_hz=control_rate,
    )
    planner = JointCommandInterpolator(
        segment_duration_sec=segment_duration,
        stale_after_sec=None,
    )
    planner.reset(
        references[0].qpos_by_side,
        epoch=0,
        monotonic_time=float(references[0].relative_time_sec),
    )

    ticks: list[ReplayControlTick] = []
    for control_index, tick_time in enumerate(command_times):
        while (
            next_reference_index < len(references)
            and float(references[next_reference_index].relative_time_sec)
            <= tick_time + _TIME_EPSILON_SEC
        ):
            next_reference = references[next_reference_index]
            reference_time = float(next_reference.relative_time_sec)
            active = next_reference
            active_target = _copy_qpos_by_side(active.qpos_by_side)
            segment_start_time = reference_time
            planner.accept_qpos_target(
                active.qpos_by_side,
                epoch=0,
                sequence=int(active.reference_index),
                published_monotonic_time=reference_time,
                monotonic_time=reference_time,
                source="offline_replay_reference",
            )
            next_reference_index += 1

        planned_tick = planner.step(monotonic_time=tick_time)
        qpos_by_side = {
            side: np.asarray(qpos, dtype=float).copy()
            for side, qpos in planned_tick.q_cmd_by_side.items()
        }
        alpha = float(planned_tick.interpolation_alpha)
        metadata = dict(active.metadata or {})
        metadata.update(
            {
                "reference_hz": float(reference_rate),
                "control_hz": float(control_rate),
                "command_hz": float(control_rate),
                "reference_index": int(active.reference_index),
                "reference_relative_time_sec": float(active.relative_time_sec),
                "reference_action_interpolation_alpha": float(
                    active.action_interpolation_alpha
                ),
                "command_resampling_mode": (
                    "shared_joint_command_interpolator"
                ),
                "command_segment_anchor": "last_emitted_command",
                "command_planner": (
                    "robot_motion.position_control.JointCommandInterpolator"
                ),
                "reference_command_tie_break": (
                    "accept_reference_before_command_emit"
                ),
                "command_interpolation_alpha": float(alpha),
                "command_time_grid_rule": (
                    "fixed_rate_first_tick_at_or_after_final_deadline"
                ),
                "final_deadline_sec": float(final_deadline),
                "include_final_settle": bool(include_final_settle),
            }
        )
        ticks.append(
            ReplayControlTick(
                control_index=int(control_index),
                action_index=int(active.action_index),
                action_frame_index=int(active.action_frame_index),
                relative_time_sec=float(tick_time),
                action_relative_time_sec=float(active.action_relative_time_sec),
                reference_index=int(active.reference_index),
                reference_relative_time_sec=float(active.relative_time_sec),
                reference_action_interpolation_alpha=float(
                    active.action_interpolation_alpha
                ),
                segment_elapsed_sec=max(0.0, float(tick_time) - segment_start_time),
                segment_duration_sec=float(segment_duration),
                interpolation_alpha=float(alpha),
                qpos_by_side=qpos_by_side,
                target_qpos_by_side=_copy_qpos_by_side(active_target),
                reference_qpos_by_side=_copy_qpos_by_side(active.qpos_by_side),
                metadata=metadata,
            )
        )
    return ticks


def _iter_continuous_position_qpos_control_ticks(
    actions: Sequence[ReplayControlAction],
    *,
    reference_hz: float,
    control_hz: float,
    include_final_settle: bool,
    limits: JointHorizonLimits,
    timing_mode: str,
    max_time_scale: float,
) -> list[ReplayControlTick]:
    """Build one exact-waypoint C2 trajectory on the command clock.

    ``reference_hz`` remains in the public replay contract and determines the
    optional final hold duration.  It does not create a second interpolation
    owner in this mode.
    """

    action_list = _normalize_actions(actions)
    reference_rate = validate_replay_rate(reference_hz, name="reference_hz")
    control_rate = validate_replay_rate(control_hz, name="control_hz")
    sides = tuple(sorted(action_list[0].qpos_by_side))
    if not sides:
        raise ValueError("continuous qpos replay requires at least one side.")
    for action in action_list:
        if tuple(sorted(action.qpos_by_side)) != sides:
            raise ValueError(
                "continuous qpos replay actions must keep a stable side set."
            )
    dimension_by_side = {
        side: int(np.asarray(action_list[0].qpos_by_side[side]).size)
        for side in sides
    }
    total_dimension = sum(dimension_by_side.values())
    if limits.dimension != total_dimension:
        raise ValueError(
            "continuous_limits dimension does not match replay qpos sides: "
            f"limits={limits.dimension}, qpos={total_dimension}."
        )

    waypoint_times = np.asarray(
        [action.relative_time_sec for action in action_list],
        dtype=float,
    )
    waypoint_q = np.asarray(
        [
            np.concatenate(
                [
                    np.asarray(action.qpos_by_side[side], dtype=float)
                    for side in sides
                ]
            )
            for action in action_list
        ],
        dtype=float,
    )
    trajectory = build_offline_quintic_waypoint_trajectory(
        waypoint_times,
        waypoint_q,
        command_hz=control_rate,
        limits=limits,
        timing_mode=timing_mode,
        max_time_scale=max_time_scale,
        terminal_hold_sec=(1.0 / reference_rate if include_final_settle else 0.0),
    )
    execution_waypoint_times = trajectory.waypoint_timestamps
    final_deadline = float(execution_waypoint_times[-1])
    if include_final_settle:
        final_deadline += 1.0 / reference_rate

    side_slices: dict[str, slice] = {}
    offset = 0
    for side in sides:
        next_offset = offset + dimension_by_side[side]
        side_slices[side] = slice(offset, next_offset)
        offset = next_offset

    ticks: list[ReplayControlTick] = []
    for control_index, timestamp in enumerate(trajectory.timestamps):
        active_position = int(
            np.clip(
                np.searchsorted(
                    execution_waypoint_times,
                    timestamp,
                    side="right",
                )
                - 1,
                0,
                len(action_list) - 1,
            )
        )
        next_position = min(active_position + 1, len(action_list) - 1)
        active = action_list[active_position]
        successor = action_list[next_position]
        interval_duration = float(
            execution_waypoint_times[next_position]
            - execution_waypoint_times[active_position]
        )
        alpha = (
            0.0
            if next_position == active_position or interval_duration <= 0.0
            else float(
                np.clip(
                    (
                        timestamp
                        - execution_waypoint_times[active_position]
                    )
                    / interval_duration,
                    0.0,
                    1.0,
                )
            )
        )
        qpos_by_side = {
            side: np.asarray(
                trajectory.q_rad[control_index, side_slices[side]],
                dtype=float,
            ).copy()
            for side in sides
        }
        target_qpos_by_side = {
            side: np.asarray(successor.qpos_by_side[side], dtype=float).copy()
            for side in sides
        }
        metadata = dict(active.metadata or {})
        metadata.update(
            {
                "position_control_pipeline": (
                    "action_waypoints_to_continuous_command"
                ),
                "reference_hz": float(reference_rate),
                "control_hz": float(control_rate),
                "command_hz": float(control_rate),
                "command_planner": trajectory.metadata["planner"],
                "command_resampling_mode": (
                    REPLAY_QPOS_EXECUTION_CONTINUOUS_QUINTIC
                ),
                "command_segment_anchor": "global_timestamped_waypoints",
                "continuous_qpos_hard_limit_certificate": dict(
                    trajectory.metadata["continuous_hard_limit_certificate"]
                ),
                "continuous_qd_rad_s": trajectory.qd_rad_s[
                    control_index
                ].tolist(),
                "continuous_qdd_rad_s2": trajectory.qdd_rad_s2[
                    control_index
                ].tolist(),
                "continuous_jerk_rad_s3": trajectory.jerk_rad_s3[
                    control_index
                ].tolist(),
                "continuous_internal_knot_zero_velocity_required": False,
                "continuous_timing_mode": trajectory.metadata["timing_mode"],
                "continuous_time_scale": trajectory.metadata["time_scale"],
                "continuous_retimed": trajectory.metadata["retimed"],
                "continuous_source_duration_sec": trajectory.metadata[
                    "source_duration_sec"
                ],
                "continuous_execution_duration_sec": trajectory.metadata[
                    "execution_duration_sec"
                ],
                "reference_action_interpolation_alpha": alpha,
                "final_deadline_sec": float(final_deadline),
                "include_final_settle": bool(include_final_settle),
            }
        )
        ticks.append(
            ReplayControlTick(
                control_index=int(control_index),
                action_index=int(active.action_index),
                action_frame_index=int(active.frame_index),
                relative_time_sec=float(timestamp),
                action_relative_time_sec=float(active.relative_time_sec),
                reference_index=int(active_position),
                reference_relative_time_sec=float(timestamp),
                reference_action_interpolation_alpha=alpha,
                segment_elapsed_sec=max(
                    0.0,
                    float(timestamp)
                    - float(execution_waypoint_times[active_position]),
                ),
                segment_duration_sec=max(0.0, interval_duration),
                interpolation_alpha=alpha,
                qpos_by_side=qpos_by_side,
                target_qpos_by_side=target_qpos_by_side,
                reference_qpos_by_side={
                    side: value.copy() for side, value in qpos_by_side.items()
                },
                metadata=metadata,
            )
        )
    return ticks


__all__ = [
    "DEFAULT_POSITION_QPOS_CONTROL_HZ",
    "DEFAULT_REPLAY_ACTION_HZ",
    "DEFAULT_REPLAY_REFERENCE_HZ",
    "ReplayControlAction",
    "ReplayControlReference",
    "ReplayControlTick",
    "REPLAY_QPOS_EXECUTION_CONTINUOUS_QUINTIC",
    "REPLAY_QPOS_EXECUTION_LEGACY_LINEAR",
    "REPLAY_QPOS_EXECUTION_MODES",
    "iter_position_qpos_control_ticks",
    "normalize_replay_qpos_execution_mode",
    "position_qpos_segment_duration",
    "resample_position_qpos_references",
    "validate_replay_rate",
]
