#!/usr/bin/env python3
"""Visualize recorded RM65 deploy jitter against causal IK-rate candidates.

This is an offline, no-hardware research probe.  It reads one deployment debug
JSONL and plays three aligned trajectories sequentially in the existing RM65
MuJoCo scene:

1. the qpos references actually dispatched by the recorded deployment, using
   their recorded (possibly irregular) emission timing;
2. an ideal-clock reconstruction of the current 15 Hz source all-IK chain;
3. a causal higher-rate candidate that only subdivides the next visible 15 Hz
   Pose interval and never inspects the following policy target.

MuJoCo is a kinematic arm-qpos and gripper proxy viewer here.  Arm motion comes
from the compared reference path, while gripper close values come from the
recorded deployment's discrete hand dispatches.  No device backend is created
and no command is dispatched.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teleop_runtime.debug.view_rm65_pose_lift_sim_replay import (  # noqa: E402
    DEFAULT_VIEW_URDF,
    PoseLiftSimFrame,
    PoseLiftSimTrajectory,
    view_trajectories,
)
from robot_motion.position_control import (  # noqa: E402
    POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED,
    POSITION_PLAN_GAP_NONE,
    POSITION_PLAN_RELATION_REPLACEMENT,
    POSITION_PLAN_RELATION_STREAM_START,
    PositionPlanSpliceDynamicsConfig,
    PositionStreamStartAcquisitionConfig,
    PositionSourcePlanCut,
    build_position_stream_start_acquisition,
    resolve_position_plan_splice,
)
from robot_motion.rm65 import (  # noqa: E402
    RM65_QREF_ACCELERATION_HARD_V1,
    RM65_QREF_ACCELERATION_PROFILE_CHOICES,
)
from todos.probes.compare_rm65_official_ik_rates import (  # noqa: E402
    COMMAND_HZ,
    INPUT_HZ,
    SIDES,
    PoseTrack,
    RM65PositionCollisionPlanner,
    RolloutArtifact,
    _load_debug_tracks,
    _arm_base_tool_poses,
    _planner_config,
    _percentiles,
    _pose_errors,
    _rollout_track_artifact,
)


RECORDED_PROFILE = "A_RECORDED_DEPLOY_ACTUAL_TIMING"
IDEAL_15HZ_PROFILE = "B_CAUSAL_15HZ_IDEAL_CLOCK"
CANDIDATE_PROFILE_PREFIX = "C_CAUSAL_MIDPOINT_ALL_IK"
NON_RTC_SPLICE_PROFILE_PREFIX = "C_NON_RTC_HOLD_RESTART_SPLICE"
NON_RTC_CLOCK_REBASE_PROFILE = "C_NON_RTC_CLOCK_REBASE_AND_TAIL_DRAIN"
STREAM_START_ACQUISITION_PROFILE = (
    "C_STREAM_START_ACQUISITION_RTC_COMPATIBLE"
)
GRIPPER_NAME_BY_SIDE = {
    "left": "left_gripper",
    "right": "right_gripper",
}
INITIAL_GRIPPER_CLOSE_BY_NAME = {
    "left_gripper": 0.0,
    "right_gripper": 0.0,
}


@dataclass(frozen=True)
class RecordedChunk:
    chunk_id: int
    dispatch_events: tuple[Mapping[str, Any], ...]
    gap_hold_wall_time: float | None
    discrete_dispatch_events: tuple[Mapping[str, Any], ...] = ()


def _finite_positive(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return parsed


def _finite_nonnegative(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("expected a finite non-negative number")
    return parsed


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_events(path: Path) -> tuple[dict[str, Any], ...]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid deployment debug JSONL at line {line_number}: {exc}"
                ) from exc
            if isinstance(event, dict):
                events.append(event)
    return tuple(events)


def recorded_chunks_from_events(
    events: Sequence[Mapping[str, Any]],
) -> tuple[RecordedChunk, ...]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    discrete_grouped: dict[int, list[Mapping[str, Any]]] = {}
    gap_hold_by_chunk: dict[int, float] = {}
    current_chunk: int | None = None
    next_chunk = 0
    reset_boundaries = any(
        str(event.get("event", ""))
        == "position_executor_collision_planner_reset"
        for event in events
    )
    for event in events:
        event_name = str(event.get("event", ""))
        if (
            reset_boundaries
            and event_name == "position_executor_collision_planner_reset"
        ):
            current_chunk = next_chunk
            next_chunk += 1
            grouped.setdefault(current_chunk, [])
            discrete_grouped.setdefault(current_chunk, [])
            continue
        if event_name == "policy_chunk" and not reset_boundaries:
            current_chunk = next_chunk
            next_chunk += 1
            grouped.setdefault(current_chunk, [])
            discrete_grouped.setdefault(current_chunk, [])
            continue
        if current_chunk is None:
            continue
        if event_name == "position_executor_arm_reference_dispatch":
            grouped[current_chunk].append(event)
        elif event_name == "position_executor_discrete_dispatch":
            discrete_grouped.setdefault(current_chunk, []).append(event)
        elif event_name == "position_chunk_gap_hold":
            wall_time = event.get("wall_time")
            if wall_time is not None and math.isfinite(float(wall_time)):
                gap_hold_by_chunk[current_chunk] = float(wall_time)

    return tuple(
        RecordedChunk(
            chunk_id=chunk_id,
            dispatch_events=tuple(grouped[chunk_id]),
            gap_hold_wall_time=gap_hold_by_chunk.get(chunk_id),
            discrete_dispatch_events=tuple(discrete_grouped.get(chunk_id, ())),
        )
        for chunk_id in sorted(grouped)
        if grouped[chunk_id]
    )


def _event_qpos(event: Mapping[str, Any]) -> dict[str, np.ndarray]:
    commands = dict(event.get("position_arm_commands") or {})
    result: dict[str, np.ndarray] = {}
    for side in SIDES:
        value = dict(commands.get(f"{side}_arm") or {}).get("value")
        qpos = np.asarray(value, dtype=float).reshape(-1)
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            raise ValueError(f"recorded dispatch has invalid {side} qpos")
        result[side] = qpos.copy()
    return result


def _event_timestamp(event: Mapping[str, Any]) -> float:
    for key in (
        "position_reference_emitted_monotonic_time",
        "observed_monotonic_time",
        "wall_time",
    ):
        value = event.get(key)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    raise ValueError("recorded dispatch has no finite timestamp")


def _discrete_event_timestamp(event: Mapping[str, Any]) -> float:
    for key in (
        "position_discrete_logical_tick_monotonic_time",
        "position_discrete_callback_completed_monotonic_time",
        "observed_monotonic_time",
        "wall_time",
    ):
        value = event.get(key)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    raise ValueError("recorded discrete dispatch has no finite timestamp")


def _discrete_event_source_index(event: Mapping[str, Any]) -> int:
    value = event.get("position_discrete_source_index")
    if isinstance(value, bool) or value is None:
        raise ValueError("recorded discrete dispatch has no source index")
    index = int(value)
    if index < 0 or float(index) != float(value):
        raise ValueError("recorded discrete dispatch source index is invalid")
    return index


def _discrete_event_gripper_targets(
    event: Mapping[str, Any],
) -> dict[str, float]:
    commands = event.get("position_discrete_commands")
    if not isinstance(commands, Mapping):
        raise ValueError("recorded discrete dispatch has no command mapping")
    result: dict[str, float] = {}
    for side, gripper_name in GRIPPER_NAME_BY_SIDE.items():
        command = commands.get(f"{side}_hand")
        if not isinstance(command, Mapping):
            continue
        command_type = str(command.get("command_type", ""))
        if command_type not in {"joint_vector", "gripper_scalar"}:
            raise ValueError(
                f"recorded {side} hand command type is unsupported: "
                f"{command_type!r}"
            )
        value = np.asarray(command.get("value"), dtype=float).reshape(-1)
        if value.size < 1 or not math.isfinite(float(value[0])):
            raise ValueError(f"recorded {side} hand command is invalid")
        close = float(value[0])
        if close < -1e-9 or close > 1.0 + 1e-9:
            raise ValueError(
                f"recorded {side} hand close value is outside [0, 1]: {close}"
            )
        result[gripper_name] = float(np.clip(close, 0.0, 1.0))
    if not result:
        raise ValueError("recorded discrete dispatch has no supported hand command")
    return result


def _gripper_targets_by_source_index(
    chunk: RecordedChunk,
    *,
    source_index_shift: int = 0,
) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for event in sorted(
        chunk.discrete_dispatch_events,
        key=_discrete_event_timestamp,
    ):
        index = _discrete_event_source_index(event) + int(source_index_shift)
        result.setdefault(index, {}).update(
            _discrete_event_gripper_targets(event)
        )
    return result


def _advance_gripper_source_state(
    state: dict[str, float],
    targets_by_source_index: Mapping[int, Mapping[str, float]],
    *,
    source_index: int,
) -> dict[str, float]:
    for index in sorted(targets_by_source_index):
        if index <= int(source_index):
            state.update(
                {
                    str(name): float(close)
                    for name, close in targets_by_source_index[index].items()
                }
            )
    return dict(state)


def _gripper_states_for_arm_events(
    arm_events: Sequence[Mapping[str, Any]],
    discrete_events: Sequence[Mapping[str, Any]],
    *,
    initial_state: Mapping[str, float],
) -> tuple[dict[str, float], ...]:
    state = {str(name): float(value) for name, value in initial_state.items()}
    ordered_discrete = sorted(discrete_events, key=_discrete_event_timestamp)
    cursor = 0
    result: list[dict[str, float]] = []
    for arm_event in arm_events:
        arm_timestamp = _event_timestamp(arm_event)
        while (
            cursor < len(ordered_discrete)
            and _discrete_event_timestamp(ordered_discrete[cursor])
            <= arm_timestamp + 1e-9
        ):
            state.update(
                _discrete_event_gripper_targets(ordered_discrete[cursor])
            )
            cursor += 1
        result.append(dict(state))
    return tuple(result)


def _gripper_metrics(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values: dict[str, list[float]] = {
        name: [] for name in GRIPPER_NAME_BY_SIDE.values()
    }
    for event in events:
        for name, close in _discrete_event_gripper_targets(event).items():
            values[name].append(float(close))
    return {
        "recorded_discrete_dispatch_count": len(events),
        "normalized_close_range_by_name": {
            name: (
                None
                if not closes
                else {"min": min(closes), "max": max(closes)}
            )
            for name, closes in values.items()
        },
        "viewer_geometry": "parallel_gripper_proxy_for_dmtac_normalized_close",
    }


def _time_based_motion_metrics(
    times_sec: np.ndarray,
    qpos_rad: np.ndarray,
) -> dict[str, float | int | None]:
    times = np.asarray(times_sec, dtype=float).reshape(-1)
    qpos = np.asarray(qpos_rad, dtype=float).reshape(-1, 12)
    if times.shape[0] != qpos.shape[0] or times.shape[0] < 2:
        return {
            "dispatch_count": int(times.shape[0]),
            "duration_sec": 0.0,
            "max_gap_ms": None,
            "gaps_over_12p5ms": 0,
            "max_joint_step_deg": None,
            "max_joint_velocity_deg_s": None,
            "max_joint_acceleration_deg_s2": None,
            "max_joint_jerk_deg_s3": None,
        }
    dt = np.diff(times)
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
        raise ValueError("recorded dispatch timestamps must be strictly increasing")
    dq = np.diff(qpos, axis=0)
    velocity = dq / dt[:, None]
    velocity_time = 0.5 * (times[:-1] + times[1:])
    velocity_dt = np.diff(velocity_time)
    acceleration = (
        np.diff(velocity, axis=0) / velocity_dt[:, None]
        if velocity.shape[0] >= 2
        else np.empty((0, qpos.shape[1]), dtype=float)
    )
    acceleration_time = (
        0.5 * (velocity_time[:-1] + velocity_time[1:])
        if velocity_time.shape[0] >= 2
        else np.empty((0,), dtype=float)
    )
    acceleration_dt = np.diff(acceleration_time)
    jerk = (
        np.diff(acceleration, axis=0) / acceleration_dt[:, None]
        if acceleration.shape[0] >= 2
        else np.empty((0, qpos.shape[1]), dtype=float)
    )
    scale = 180.0 / math.pi

    def maximum(array: np.ndarray) -> float | None:
        return None if array.size == 0 else float(np.max(np.abs(array)) * scale)

    return {
        "dispatch_count": int(times.shape[0]),
        "duration_sec": float(times[-1] - times[0]),
        "max_gap_ms": float(np.max(dt) * 1000.0),
        "gaps_over_12p5ms": int(np.sum(dt > 1.5 / COMMAND_HZ)),
        "max_joint_step_deg": maximum(dq),
        "max_joint_velocity_deg_s": maximum(velocity),
        "max_joint_acceleration_deg_s2": maximum(acceleration),
        "max_joint_jerk_deg_s3": maximum(jerk),
    }


def _recorded_chunk_arrays(
    chunk: RecordedChunk,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ordered = sorted(chunk.dispatch_events, key=_event_timestamp)
    absolute_times = np.asarray([_event_timestamp(event) for event in ordered])
    qpos = np.asarray(
        [
            np.concatenate(
                [_event_qpos(event)[side] for side in SIDES],
            )
            for event in ordered
        ],
        dtype=float,
    ).reshape(-1, 12)
    relative_times = absolute_times - absolute_times[0] + (1.0 / COMMAND_HZ)
    metrics = _time_based_motion_metrics(absolute_times, qpos)
    if chunk.gap_hold_wall_time is not None:
        final_event_wall = float(ordered[-1].get("wall_time", 0.0))
        tail_hold = max(0.0, chunk.gap_hold_wall_time - final_event_wall)
        metrics["recorded_tail_hold_sec"] = tail_hold
    else:
        metrics["recorded_tail_hold_sec"] = 0.0
    return relative_times, qpos, metrics


def _auto_select_chunk(
    chunks: Sequence[RecordedChunk],
    tracks_by_chunk: Mapping[int, PoseTrack],
) -> tuple[RecordedChunk, dict[str, Any]]:
    ranked: list[tuple[tuple[float, float, float], RecordedChunk, dict[str, Any]]] = []
    for chunk in chunks:
        if chunk.chunk_id not in tracks_by_chunk:
            continue
        _times, _qpos, metrics = _recorded_chunk_arrays(chunk)
        step = float(metrics.get("max_joint_step_deg") or 0.0)
        acceleration = float(metrics.get("max_joint_acceleration_deg_s2") or 0.0)
        gap = float(metrics.get("max_gap_ms") or 0.0)
        ranked.append(((step, acceleration, gap), chunk, metrics))
    if not ranked:
        raise RuntimeError("deployment debug contains no aligned action/dispatch chunk")
    _score, selected, metrics = max(ranked, key=lambda item: item[0])
    return selected, metrics


def _chunk_id_from_track(track: PoseTrack) -> int:
    marker = ":chunk="
    if marker not in track.name:
        raise ValueError(f"debug track has no chunk identity: {track.name}")
    return int(track.name.rsplit(marker, 1)[1])


def _trajectory_from_recorded(
    track: PoseTrack,
    chunk: RecordedChunk,
) -> tuple[PoseLiftSimTrajectory, dict[str, Any]]:
    ordered_events = sorted(chunk.dispatch_events, key=_event_timestamp)
    times, qpos, metrics = _recorded_chunk_arrays(chunk)
    gripper_states = _gripper_states_for_arm_events(
        ordered_events,
        chunk.discrete_dispatch_events,
        initial_state=INITIAL_GRIPPER_CLOSE_BY_NAME,
    )
    metrics["gripper"] = _gripper_metrics(chunk.discrete_dispatch_events)
    frames = tuple(
        PoseLiftSimFrame(
            sequence=index + 1,
            timestamp=float(times[index]),
            source_frame_index=min(
                track.frames - 1,
                int(index * INPUT_HZ / max(COMMAND_HZ, 1.0)) + 1,
            ),
            interpolation_alpha=0.0,
            status="recorded_dispatch",
            hold=False,
            qpos_by_side={
                "left": qpos[index, :6].copy(),
                "right": qpos[index, 6:].copy(),
            },
            modification_reasons=("recorded_actual_emission_timing",),
            pose_lift_reason="recorded_deploy_reference",
            collision_distance_m=None,
            gripper_close_by_name=gripper_states[index],
        )
        for index in range(qpos.shape[0])
    )
    return (
        PoseLiftSimTrajectory(
            profile_name=RECORDED_PROFILE,
            source_episode_index=int(chunk.chunk_id),
            source_path=str(track.source),
            source_frame_count=int(track.frames),
            reference_hz=COMMAND_HZ,
            initial_qpos_by_side={
                side: np.asarray(
                    track.initial_qpos_by_side[side],
                    dtype=float,
                ).copy()
                for side in SIDES
            },
            frames=frames,
            status_counts={"recorded_dispatch": len(frames)},
            hold_count=0,
            pose_lift_applied_count=0,
            initial_gripper_close_by_name=dict(
                INITIAL_GRIPPER_CLOSE_BY_NAME
            ),
        ),
        metrics,
    )


def _trajectory_from_recorded_full(
    tracks_by_chunk: Mapping[int, PoseTrack],
    chunks: Sequence[RecordedChunk],
) -> tuple[PoseLiftSimTrajectory, dict[str, Any]]:
    aligned = [chunk for chunk in chunks if chunk.chunk_id in tracks_by_chunk]
    if not aligned:
        raise RuntimeError("deployment debug has no complete recorded trajectory")
    events = sorted(
        [event for chunk in aligned for event in chunk.dispatch_events],
        key=_event_timestamp,
    )
    absolute_times = np.asarray([_event_timestamp(event) for event in events])
    qpos = np.asarray(
        [
            np.concatenate([_event_qpos(event)[side] for side in SIDES])
            for event in events
        ],
        dtype=float,
    ).reshape(-1, 12)
    relative_times = absolute_times - absolute_times[0] + (1.0 / COMMAND_HZ)
    metrics = _time_based_motion_metrics(absolute_times, qpos)
    discrete_events = sorted(
        [
            event
            for chunk in aligned
            for event in chunk.discrete_dispatch_events
        ],
        key=_discrete_event_timestamp,
    )
    gripper_states = _gripper_states_for_arm_events(
        events,
        discrete_events,
        initial_state=INITIAL_GRIPPER_CLOSE_BY_NAME,
    )
    metrics["gripper"] = _gripper_metrics(discrete_events)
    metrics["chunk_count"] = len(aligned)
    metrics["source_pose_frames_15hz"] = int(
        sum(tracks_by_chunk[chunk.chunk_id].frames for chunk in aligned)
    )
    recorded_translation_errors: list[float] = []
    recorded_rotation_errors: list[float] = []
    for event in events:
        collision_debug = dict(event.get("position_collision_debug") or {})
        target = dict(
            collision_debug.get("target_endpoint_pose_world_by_side") or {}
        )
        realized = dict(
            collision_debug.get("realized_endpoint_pose_world_by_side") or {}
        )
        for side in SIDES:
            if side not in target or side not in realized:
                continue
            translation, rotation = _pose_errors(
                np.asarray(target[side], dtype=float).reshape(4, 4),
                np.asarray(realized[side], dtype=float).reshape(4, 4),
            )
            recorded_translation_errors.append(translation)
            recorded_rotation_errors.append(rotation)
    metrics["pose_tracking"] = {
        "translation_error_mm": _percentiles(recorded_translation_errors),
        "rotation_error_deg": _percentiles(recorded_rotation_errors),
    }
    recorded_boundaries: list[dict[str, Any]] = []
    previous_last_event: Mapping[str, Any] | None = None
    for chunk_position, chunk in enumerate(aligned):
        ordered = sorted(chunk.dispatch_events, key=_event_timestamp)
        first_event = ordered[0]
        planner_metadata = dict(
            (first_event.get("position_collision_debug") or {}).get(
                "planner_metadata"
            )
            or {}
        )
        if chunk_position > 0:
            assert previous_last_event is not None
            previous_qpos = np.concatenate(
                [_event_qpos(previous_last_event)[side] for side in SIDES]
            )
            first_qpos = np.concatenate(
                [_event_qpos(first_event)[side] for side in SIDES]
            )
            recorded_boundaries.append(
                {
                    "chunk_id": int(chunk.chunk_id),
                    "recorded_gap_sec": float(
                        _event_timestamp(first_event)
                        - _event_timestamp(previous_last_event)
                    ),
                    "first_dispatch_interpolation_alpha": float(
                        first_event.get("position_interpolation_alpha", 0.0)
                    ),
                    "first_dispatch_source_cut_command_offset": int(
                        planner_metadata.get("source_cut_command_offset", -1)
                    ),
                    "first_dispatch_plan_row": int(
                        planner_metadata.get("source_cut_command_offset", -1)
                    )
                    + 1,
                    "first_dispatch_step_deg": float(
                        np.max(np.abs(np.rad2deg(first_qpos - previous_qpos)))
                    ),
                    "governor_execution_mode": planner_metadata.get(
                        "governor_execution_mode"
                    ),
                    "continuous_waypoint_status": dict(
                        planner_metadata.get("continuous_waypoint_shadow") or {}
                    ).get("status"),
                }
            )
        previous_last_event = ordered[-1]
    metrics["post_hold_boundaries"] = recorded_boundaries
    metrics["post_hold_boundary_worst"] = {
        "first_dispatch_interpolation_alpha": max(
            (
                float(item["first_dispatch_interpolation_alpha"])
                for item in recorded_boundaries
            ),
            default=0.0,
        ),
        "first_dispatch_plan_row": max(
            (int(item["first_dispatch_plan_row"]) for item in recorded_boundaries),
            default=0,
        ),
        "first_dispatch_step_deg": max(
            (float(item["first_dispatch_step_deg"]) for item in recorded_boundaries),
            default=0.0,
        ),
    }
    frames: list[PoseLiftSimFrame] = []
    for index, (event, timestamp) in enumerate(zip(events, relative_times)):
        parent_indices = [
            int(value) for value in (event.get("position_parent_source_indices") or ())
        ]
        frames.append(
            PoseLiftSimFrame(
                sequence=index + 1,
                timestamp=float(timestamp),
                source_frame_index=(max(parent_indices) if parent_indices else index),
                interpolation_alpha=float(event.get("position_interpolation_alpha", 0.0)),
                status="recorded_dispatch",
                hold=False,
                qpos_by_side={
                    "left": qpos[index, :6].copy(),
                    "right": qpos[index, 6:].copy(),
                },
                modification_reasons=("recorded_actual_emission_timing",),
                pose_lift_reason="recorded_deploy_reference",
                collision_distance_m=(
                    None
                    if event.get("position_collision_distance_m") is None
                    else float(event["position_collision_distance_m"])
                ),
                gripper_close_by_name=gripper_states[index],
            )
        )
    first_track = tracks_by_chunk[aligned[0].chunk_id]
    return (
        PoseLiftSimTrajectory(
            profile_name=RECORDED_PROFILE,
            source_episode_index=-1,
            source_path=str(first_track.source),
            source_frame_count=int(metrics["source_pose_frames_15hz"]),
            reference_hz=COMMAND_HZ,
            initial_qpos_by_side={
                side: np.asarray(
                    first_track.initial_qpos_by_side[side],
                    dtype=float,
                ).copy()
                for side in SIDES
            },
            frames=tuple(frames),
            status_counts={"recorded_dispatch": len(frames)},
            hold_count=0,
            pose_lift_applied_count=0,
            initial_gripper_close_by_name=dict(
                INITIAL_GRIPPER_CLOSE_BY_NAME
            ),
        ),
        metrics,
    )


def _trajectory_from_rollout(
    track: PoseTrack,
    artifact: RolloutArtifact,
    *,
    profile_name: str,
    ik_hz: float,
    chunk: RecordedChunk | None = None,
) -> PoseLiftSimTrajectory:
    qpos = np.asarray(artifact.q_command_rad, dtype=float).reshape(-1, 12)
    command_hz = float(artifact.summary.get("command_hz", COMMAND_HZ))
    commands_per_source = max(1, int(round(command_hz / INPUT_HZ)))
    gripper_targets = (
        {}
        if chunk is None
        else _gripper_targets_by_source_index(chunk)
    )
    gripper_state = dict(INITIAL_GRIPPER_CLOSE_BY_NAME)
    frame_grippers: list[dict[str, float]] = []
    for index in range(qpos.shape[0]):
        frame_grippers.append(
            _advance_gripper_source_state(
                gripper_state,
                gripper_targets,
                source_index=min(
                    track.frames - 1,
                    index // commands_per_source + 1,
                ),
            )
        )
    frames = tuple(
        PoseLiftSimFrame(
            sequence=index + 1,
            timestamp=(index + 1) / command_hz,
            source_frame_index=min(
                track.frames - 1,
                index // commands_per_source + 1,
            ),
            interpolation_alpha=float(
                ((index % commands_per_source) + 1) / commands_per_source
            ),
            status="offline_causal_rollout",
            hold=False,
            qpos_by_side={
                "left": qpos[index, :6].copy(),
                "right": qpos[index, 6:].copy(),
            },
            modification_reasons=(f"source_all_ik_{ik_hz:g}hz",),
            pose_lift_reason="next_15hz_target_only",
            collision_distance_m=None,
            gripper_close_by_name=frame_grippers[index],
        )
        for index in range(qpos.shape[0])
    )
    return PoseLiftSimTrajectory(
        profile_name=profile_name,
        source_episode_index=_chunk_id_from_track(track),
        source_path=str(track.source),
        source_frame_count=int(track.frames),
        reference_hz=command_hz,
        initial_qpos_by_side={
            side: np.asarray(track.initial_qpos_by_side[side], dtype=float).copy()
            for side in SIDES
        },
        frames=frames,
        status_counts={"offline_causal_rollout": len(frames)},
        hold_count=0,
        pose_lift_applied_count=len(artifact.source_qpos_rad_by_index),
        initial_gripper_close_by_name=dict(INITIAL_GRIPPER_CLOSE_BY_NAME),
    )


def _trajectory_from_rollouts_full(
    tracks_by_chunk: Mapping[int, PoseTrack],
    chunks: Sequence[RecordedChunk],
    *,
    ik_hz: float,
    profile_name: str,
    args: argparse.Namespace,
    trajectory_mode: str = "formal",
    plan_splice_dynamics_config: PositionPlanSpliceDynamicsConfig | None = None,
    stream_start_acquisition_config: (
        PositionStreamStartAcquisitionConfig | None
    ) = None,
) -> tuple[PoseLiftSimTrajectory, dict[str, Any]]:
    command_hz = float(getattr(args, "command_hz", COMMAND_HZ))
    commands_per_source = max(1, int(round(command_hz / INPUT_HZ)))
    aligned = [chunk for chunk in chunks if chunk.chunk_id in tracks_by_chunk]
    if not aligned:
        raise RuntimeError("deployment debug has no complete candidate trajectory")
    first_track = tracks_by_chunk[aligned[0].chunk_id]
    initial = {
        side: np.asarray(first_track.initial_qpos_by_side[side], dtype=float).copy()
        for side in SIDES
    }
    anchor = {side: value.copy() for side, value in initial.items()}
    gripper_state = dict(INITIAL_GRIPPER_CLOSE_BY_NAME)
    frames: list[PoseLiftSimFrame] = []
    per_chunk: list[dict[str, Any]] = []
    time_cursor = 0.0
    source_offset = 0
    pose_lift_count = 0
    gap_holds = 0
    previous_recorded_end: float | None = None
    post_hold_boundaries: list[dict[str, Any]] = []
    previous_chunk: RecordedChunk | None = None
    stream_start_acquisition_metadata: dict[str, Any] | None = None
    for chunk_position, chunk in enumerate(aligned):
        track = tracks_by_chunk[chunk.chunk_id]
        recorded_events = sorted(chunk.dispatch_events, key=_event_timestamp)
        recorded_start = _event_timestamp(recorded_events[0])
        if previous_recorded_end is not None:
            gap_sec = max(0.0, recorded_start - previous_recorded_end)
            if gap_sec > 0.0:
                time_cursor += gap_sec
                frames.append(
                    PoseLiftSimFrame(
                        sequence=len(frames) + 1,
                        timestamp=time_cursor,
                        source_frame_index=source_offset,
                        interpolation_alpha=0.0,
                        status="recorded_policy_gap_hold",
                        hold=True,
                        qpos_by_side={side: value.copy() for side, value in anchor.items()},
                        modification_reasons=("recorded_policy_gap_hold",),
                        pose_lift_reason="preserve_recorded_inference_gap",
                        collision_distance_m=None,
                        gripper_close_by_name=dict(gripper_state),
                    )
                )
                gap_holds += 1
        causal_track = replace(
            track,
            initial_qpos_by_side={side: value.copy() for side, value in anchor.items()},
        )
        gripper_source_index_shift = 0
        if (
            chunk_position == 0
            and stream_start_acquisition_config is not None
            and stream_start_acquisition_config.enabled
        ):
            acquisition_oracle = RM65PositionCollisionPlanner(
                _planner_config(args),
                endpoint_to_tool_by_side=track.endpoint_to_tool_by_side,
            )
            measured_tool_pose = _arm_base_tool_poses(
                acquisition_oracle,
                causal_track.initial_qpos_by_side,
                causal_track.endpoint_to_tool_by_side,
            )
            acquisition = build_position_stream_start_acquisition(
                measured_tool_pose,
                {
                    side: causal_track.pose_15hz_by_side[side][0]
                    for side in SIDES
                },
                config=stream_start_acquisition_config,
            )
            stream_start_acquisition_metadata = acquisition.to_metadata()
            if acquisition.active:
                gripper_source_index_shift = len(
                    acquisition.prelude_pose_by_side
                )
                causal_track = replace(
                    causal_track,
                    pose_15hz_by_side={
                        side: np.concatenate(
                            (
                                np.asarray(
                                    [
                                        prelude[side]
                                        for prelude in (
                                            acquisition.prelude_pose_by_side
                                        )
                                    ],
                                    dtype=float,
                                ),
                                causal_track.pose_15hz_by_side[side],
                            ),
                            axis=0,
                        )
                        for side in SIDES
                    },
                )
        gripper_targets = _gripper_targets_by_source_index(
            chunk,
            source_index_shift=gripper_source_index_shift,
        )
        splice_decision = None
        if (
            chunk_position == 0
            and stream_start_acquisition_metadata is not None
            and bool(stream_start_acquisition_metadata.get("active", False))
        ):
            splice_decision = resolve_position_plan_splice(
                PositionSourcePlanCut(
                    cut_id=f"offline-stream-start:{chunk.chunk_id}",
                    current_source_plan_id=f"policy_chunk:{chunk.chunk_id}",
                    current_source_step=0,
                    previous_source_plan_id=None,
                    previous_source_step=None,
                    relation=POSITION_PLAN_RELATION_STREAM_START,
                    execution_gap=POSITION_PLAN_GAP_NONE,
                    detected_at=recorded_start,
                    evidence={
                        "offline_only": True,
                        "stream_start_acquisition": True,
                    },
                )
            )
        elif chunk_position > 0:
            assert previous_chunk is not None
            previous_track = tracks_by_chunk[previous_chunk.chunk_id]
            splice_decision = resolve_position_plan_splice(
                PositionSourcePlanCut(
                    cut_id=(
                        f"offline-nonrtc:{previous_chunk.chunk_id}"
                        f"->{chunk.chunk_id}"
                    ),
                    current_source_plan_id=f"policy_chunk:{chunk.chunk_id}",
                    current_source_step=0,
                    previous_source_plan_id=(
                        f"policy_chunk:{previous_chunk.chunk_id}"
                    ),
                    previous_source_step=max(0, previous_track.frames - 1),
                    relation=POSITION_PLAN_RELATION_REPLACEMENT,
                    execution_gap=POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED,
                    detected_at=recorded_start,
                    evidence={
                        "offline_only": True,
                        "recorded_policy_gap_preserved": True,
                    },
                )
            )
        artifact = _rollout_track_artifact(
            causal_track,
            ik_hz=float(ik_hz),
            mode=trajectory_mode,
            args=args,
            plan_splice_decision=splice_decision,
            plan_splice_dynamics_config=plan_splice_dynamics_config,
        )
        if artifact.summary.get("error") is not None:
            raise RuntimeError(
                f"full trajectory chunk {chunk.chunk_id} failed: "
                f"{artifact.summary['error']}"
            )
        qpos = np.asarray(artifact.q_command_rad, dtype=float).reshape(-1, 12)
        for local_index, row in enumerate(qpos):
            time_cursor += 1.0 / command_hz
            local_source_index = min(
                causal_track.frames - 1,
                local_index // commands_per_source + 1,
            )
            frames.append(
                PoseLiftSimFrame(
                    sequence=len(frames) + 1,
                    timestamp=time_cursor,
                    source_frame_index=(
                        source_offset
                        + local_source_index
                    ),
                    interpolation_alpha=float(
                        ((local_index % commands_per_source) + 1)
                        / commands_per_source
                    ),
                    status="offline_causal_rollout",
                    hold=False,
                    qpos_by_side={
                        "left": row[:6].copy(),
                        "right": row[6:].copy(),
                    },
                    modification_reasons=(f"source_all_ik_{ik_hz:g}hz",),
                    pose_lift_reason="next_15hz_target_only",
                    collision_distance_m=None,
                    gripper_close_by_name=_advance_gripper_source_state(
                        gripper_state,
                        gripper_targets,
                        source_index=local_source_index,
                    ),
                )
            )
        if qpos.shape[0] == 0:
            raise RuntimeError(f"full trajectory chunk {chunk.chunk_id} is empty")
        if chunk_position > 0:
            first_interval = min(commands_per_source, qpos.shape[0])
            first_steps = np.diff(
                np.vstack((
                    np.concatenate([anchor[side] for side in SIDES])[None, :],
                    qpos[:first_interval],
                )),
                axis=0,
            )
            raw = artifact.raw_metrics
            post_hold_boundaries.append(
                {
                    "chunk_id": int(chunk.chunk_id),
                    "recorded_gap_sec": float(gap_sec),
                    "first_interval_command_count": int(first_interval),
                    "max_joint_step_deg": float(
                        np.max(np.abs(np.rad2deg(first_steps)))
                    ),
                    "max_joint_velocity_deg_s": float(
                        max(raw["velocity_deg_s"][:first_interval], default=0.0)
                    ),
                    "max_joint_acceleration_deg_s2": float(
                        max(
                            raw["acceleration_deg_s2"][:first_interval],
                            default=0.0,
                        )
                    ),
                    "max_joint_jerk_deg_s3": float(
                        max(raw["jerk_deg_s3"][:first_interval], default=0.0)
                    ),
                    "translation_error_mm_max": float(
                        max(
                            raw["translation_error_mm"][: 2 * first_interval],
                            default=0.0,
                        )
                    ),
                    "rotation_error_deg_max": float(
                        max(
                            raw["rotation_error_deg"][: 2 * first_interval],
                            default=0.0,
                        )
                    ),
                    "regularized_cut_count": int(
                        artifact.summary["plan_splice_regularized_cut_count"]
                    ),
                    "regularization_fallback_cut_count": int(
                        artifact.summary["plan_splice_fallback_cut_count"]
                    ),
                }
            )
        anchor = {"left": qpos[-1, :6].copy(), "right": qpos[-1, 6:].copy()}
        pose_lift_count += len(artifact.source_qpos_rad_by_index)
        per_chunk.append(dict(artifact.summary))
        source_offset += causal_track.frames
        previous_recorded_end = _event_timestamp(recorded_events[-1])
        previous_chunk = chunk
    trajectory = PoseLiftSimTrajectory(
        profile_name=profile_name,
        source_episode_index=-1,
        source_path=str(first_track.source),
        source_frame_count=source_offset,
        reference_hz=command_hz,
        initial_qpos_by_side=initial,
        frames=tuple(frames),
        status_counts={
            "offline_causal_rollout": len(frames) - gap_holds,
            "recorded_policy_gap_hold": gap_holds,
        },
        hold_count=gap_holds,
        pose_lift_applied_count=pose_lift_count,
        initial_gripper_close_by_name=dict(INITIAL_GRIPPER_CLOSE_BY_NAME),
    )
    summary = {
        "error": None,
        "ik_hz": float(ik_hz),
        "command_hz": command_hz,
        "trajectory_mode": trajectory_mode,
        "chunk_count": len(aligned),
        "source_pose_frames_15hz": source_offset,
        "command_count": len(frames),
        "duration_sec": time_cursor,
        "cross_chunk_anchor": "previous_candidate_endpoint",
        "recorded_policy_gap_holds": gap_holds,
        "gripper": _gripper_metrics(
            [
                event
                for chunk in aligned
                for event in chunk.discrete_dispatch_events
            ]
        ),
        "stream_start_acquisition": stream_start_acquisition_metadata,
        "plan_splice_initial_jerk_regularization": float(
            0.0
            if plan_splice_dynamics_config is None
            else plan_splice_dynamics_config
            .hold_restart_initial_jerk_delta_regularization
        ),
        "plan_splice_jerk_delta_regularization": float(
            0.0
            if plan_splice_dynamics_config is None
            else plan_splice_dynamics_config
            .hold_restart_jerk_delta_regularization
        ),
        "post_hold_boundaries": post_hold_boundaries,
        "post_hold_boundary_worst": {
            key: max(
                (float(item[key]) for item in post_hold_boundaries),
                default=0.0,
            )
            for key in (
                "max_joint_step_deg",
                "max_joint_velocity_deg_s",
                "max_joint_acceleration_deg_s2",
                "max_joint_jerk_deg_s3",
                "translation_error_mm_max",
                "rotation_error_deg_max",
            )
        },
        "per_chunk": per_chunk,
    }
    return trajectory, summary


def _with_viewer_startup_hold(
    trajectory: PoseLiftSimTrajectory,
    *,
    hold_sec: float,
) -> PoseLiftSimTrajectory:
    """Add a viewer-only initial dwell without changing the control artifact."""

    duration = max(0.0, float(hold_sec))
    if duration == 0.0:
        return trajectory
    initial_frame = PoseLiftSimFrame(
        sequence=0,
        timestamp=duration,
        source_frame_index=0,
        interpolation_alpha=0.0,
        status="viewer_startup_hold",
        hold=True,
        qpos_by_side={
            side: np.asarray(
                trajectory.initial_qpos_by_side[side],
                dtype=float,
            ).copy()
            for side in SIDES
        },
        modification_reasons=("viewer_startup_hold",),
        pose_lift_reason="viewer_only_no_control_semantics",
        collision_distance_m=None,
        gripper_close_by_name=dict(
            trajectory.initial_gripper_close_by_name
        ),
    )
    shifted = tuple(
        replace(
            frame,
            sequence=int(frame.sequence) + 1,
            timestamp=float(frame.timestamp) + duration,
        )
        for frame in trajectory.frames
    )
    return replace(
        trajectory,
        frames=(initial_frame, *shifted),
        status_counts={
            **dict(trajectory.status_counts),
            "viewer_startup_hold": 1,
        },
        hold_count=int(trajectory.hold_count) + 1,
    )


def _rollout_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        scene_profile=str(args.scene_profile),
        collision_min_distance_m=float(args.collision_min_distance_m),
        collision_distance_epsilon_m=float(args.collision_distance_epsilon_m),
        jerk_ramp_sec=float(args.jerk_ramp_sec),
        batched_solver_tolerance=float(args.batched_solver_tolerance),
        command_hz=float(args.planner_command_hz),
        qref_acceleration_profile=str(
            args.candidate_qref_acceleration_profile
        ),
    )


def _validate_mujoco_playback(
    args: argparse.Namespace,
    trajectories: Sequence[PoseLiftSimTrajectory],
) -> None:
    """Load the real viewer scene and apply every qpos without opening GLFW."""

    from sim.contracts import SceneSpec
    from sim.mujoco.controllers import GripperController, QposController
    from sim.mujoco.scene import MujocoScene

    scene_spec = SceneSpec.kinematic_replay()
    scene = MujocoScene.from_urdf(
        Path(args.urdf).expanduser().resolve(),
        camera_names=(),
        timestep=1.0 / COMMAND_HZ,
        gravity=scene_spec.gravity_mps2,
        scene_spec=scene_spec,
    )
    missing = sorted(set(SIDES) - set(scene.arm_specs))
    if missing:
        raise RuntimeError(f"MuJoCo view URDF is missing arm specs: {missing}")
    controller = QposController(scene, collision_mode="raw")
    gripper_controller = GripperController(scene, collision_mode="raw")
    applied = 0
    gripper_applied = 0
    for trajectory in trajectories:
        scene.reset_data()
        controller.apply(trajectory.initial_qpos_by_side)
        if trajectory.initial_gripper_close_by_name:
            missing_grippers = sorted(
                set(trajectory.initial_gripper_close_by_name)
                - set(scene.gripper_specs)
            )
            if missing_grippers:
                raise RuntimeError(
                    "MuJoCo view URDF is missing gripper specs: "
                    f"{missing_grippers}"
                )
            gripper_controller.apply(
                trajectory.initial_gripper_close_by_name
            )
        for frame in trajectory.frames:
            controller.apply(frame.qpos_by_side)
            if frame.gripper_close_by_name:
                gripper_controller.apply(frame.gripper_close_by_name)
                gripper_applied += 1
            applied += 1
    print(
        "[DeployIKRateSim] headless_mujoco_validation=passed "
        f"trajectories={len(trajectories)} qpos_frames={applied} "
        f"gripper_frames={gripper_applied}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-jsonl", type=Path, required=True)
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="Policy chunk to play. Default: auto-select the largest recorded qpos step.",
    )
    parser.add_argument(
        "--full-deployment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Play all aligned policy chunks and their recorded inference gaps.",
    )
    parser.add_argument(
        "--candidate-ik-hz",
        type=float,
        choices=(30.0, 60.0, 120.0),
        default=30.0,
    )
    parser.add_argument(
        "--planner-command-hz",
        type=float,
        choices=(60.0, 120.0),
        default=COMMAND_HZ,
        help=(
            "Offline planner output cadence. Use 60 to match the current "
            "deployment reference layer."
        ),
    )
    parser.add_argument(
        "--candidate-trajectory-mode",
        choices=("formal", "c2"),
        default="formal",
        help="Select the formal third-order or admitted continuous C2 cut.",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=(
            "ik-rate",
            "non-rtc-clock-rebase",
            "non-rtc-splice",
            "stream-start-acquisition",
        ),
        default="ik-rate",
        help=(
            "Compare a higher-rate all-IK candidate or a 15 Hz candidate "
            "that only regularizes the first cut after a recorded non-RTC Hold."
        ),
    )
    parser.add_argument(
        "--stream-start-max-translation-step-m",
        type=_finite_positive,
        default=0.005,
    )
    parser.add_argument(
        "--stream-start-max-rotation-step-deg",
        type=_finite_positive,
        default=0.5,
    )
    parser.add_argument(
        "--stream-start-max-source-intervals",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--non-rtc-splice-initial-jerk-regularization",
        type=_finite_nonnegative,
        default=0.0,
        help=(
            "Default-off first-jerk continuity weight used only by "
            "--candidate-mode non-rtc-splice."
        ),
    )
    parser.add_argument(
        "--non-rtc-splice-jerk-delta-regularization",
        type=_finite_nonnegative,
        default=0.0,
        help=(
            "Default-off jerk-delta continuity weight across the first "
            "post-Hold planning horizon."
        ),
    )
    parser.add_argument(
        "--play",
        choices=("comparison", "recorded", "ideal15", "candidate"),
        default="comparison",
        help=(
            "Select one trajectory or the sequential comparison. "
            "The two-command handoff uses recorded and candidate separately."
        ),
    )
    parser.add_argument(
        "--include-ideal-15hz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Play an ideal-clock 15 Hz reconstruction between A and C.",
    )
    parser.add_argument("--scene-profile", default="dual_arm_hand_proxy_open")
    parser.add_argument("--collision-min-distance-m", type=float, default=0.03)
    parser.add_argument("--collision-distance-epsilon-m", type=float, default=1e-4)
    parser.add_argument("--jerk-ramp-sec", type=_finite_positive, default=0.05)
    parser.add_argument(
        "--candidate-qref-acceleration-profile",
        choices=RM65_QREF_ACCELERATION_PROFILE_CHOICES,
        default=RM65_QREF_ACCELERATION_HARD_V1,
        help=(
            "Deploy-only planned-qref acceleration profile used by the "
            "offline candidate. hard_v1 is the exact rollback."
        ),
    )
    parser.add_argument("--batched-solver-tolerance", type=float, default=1e-6)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_VIEW_URDF)
    parser.add_argument(
        "--reference-hz",
        type=_finite_positive,
        default=COMMAND_HZ,
        help="MuJoCo qpos playback rate; fixed to 120 Hz by the handoff commands.",
    )
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--speed", type=_finite_positive, default=1.0)
    parser.add_argument("--no-sleep", action="store_true")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replay continuously until the viewer is closed or Ctrl+C is pressed.",
    )
    parser.add_argument(
        "--viewer-startup-hold-sec",
        type=float,
        default=0.0,
        help="Viewer-only dwell at the initial qpos before each replay pass.",
    )
    parser.add_argument("--pause-between-profiles-sec", type=float, default=0.75)
    parser.add_argument("--hold-viewer-at-end", action="store_true")
    parser.add_argument("--hide-left-ui", action="store_true")
    parser.add_argument("--hide-right-ui", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    debug_path = Path(args.debug_jsonl).expanduser().resolve()
    if not debug_path.is_file():
        raise FileNotFoundError(debug_path)
    events = _read_events(debug_path)
    chunks = recorded_chunks_from_events(events)
    load_args = SimpleNamespace(
        exclude_debug_chunk=[],
        max_input_frames=None,
        min_input_frames=2,
    )
    tracks = _load_debug_tracks((debug_path,), args=load_args)
    tracks_by_chunk = {_chunk_id_from_track(track): track for track in tracks}
    rollout_args = _rollout_args(args)
    non_rtc_splice_candidate = str(args.candidate_mode) in {
        "non-rtc-clock-rebase",
        "non-rtc-splice",
        "stream-start-acquisition",
    }
    if non_rtc_splice_candidate and not bool(args.full_deployment):
        raise ValueError(
            "the non-RTC candidate requires --full-deployment so "
            "an acknowledged recorded Hold boundary exists."
        )
    need_ideal = args.play == "ideal15" or (
        args.play == "comparison" and bool(args.include_ideal_15hz)
    )
    need_candidate = args.play in {"comparison", "candidate"}
    candidate_ik_hz = (
        15.0 if non_rtc_splice_candidate else float(args.candidate_ik_hz)
    )
    splice_config = (
        PositionPlanSpliceDynamicsConfig(
            hold_restart_initial_jerk_delta_regularization=float(
                args.non_rtc_splice_initial_jerk_regularization
            ),
            hold_restart_jerk_delta_regularization=float(
                args.non_rtc_splice_jerk_delta_regularization
            ),
        )
        if non_rtc_splice_candidate
        else None
    )
    stream_start_config = (
        PositionStreamStartAcquisitionConfig(
            enabled=True,
            max_translation_step_m=float(
                args.stream_start_max_translation_step_m
            ),
            max_rotation_step_rad=math.radians(
                float(args.stream_start_max_rotation_step_deg)
            ),
            max_source_intervals=int(args.stream_start_max_source_intervals),
        )
        if str(args.candidate_mode) == "stream-start-acquisition"
        else None
    )
    candidate_name = (
        STREAM_START_ACQUISITION_PROFILE
        if str(args.candidate_mode) == "stream-start-acquisition"
        else
        NON_RTC_CLOCK_REBASE_PROFILE
        if (
            str(args.candidate_mode) == "non-rtc-clock-rebase"
            and float(args.non_rtc_splice_initial_jerk_regularization) == 0.0
            and float(args.non_rtc_splice_jerk_delta_regularization) == 0.0
        )
        else (
            f"{NON_RTC_SPLICE_PROFILE_PREFIX}_"
            f"i{float(args.non_rtc_splice_initial_jerk_regularization):g}_"
            f"d{float(args.non_rtc_splice_jerk_delta_regularization):g}"
        )
        if non_rtc_splice_candidate
        else (
            f"{CANDIDATE_PROFILE_PREFIX}_{candidate_ik_hz:g}HZ_"
            f"{str(args.candidate_trajectory_mode).upper()}"
        )
    )
    candidate_name = (
        f"{candidate_name}_QDD_"
        f"{str(args.candidate_qref_acceleration_profile).upper()}"
    )
    trajectories: list[PoseLiftSimTrajectory] = []
    ideal_summary: dict[str, Any] | None = None
    candidate_summary: dict[str, Any] | None = None
    if bool(args.full_deployment):
        if args.chunk is not None:
            raise ValueError("--full-deployment cannot be combined with --chunk")
        recorded_trajectory, recorded_metrics = _trajectory_from_recorded_full(
            tracks_by_chunk,
            chunks,
        )
        selection = "full_deployment_all_aligned_chunks"
        selected_chunk_id: int | None = None
        source_pose_frames = int(recorded_trajectory.source_frame_count)
        if args.play in {"comparison", "recorded"}:
            trajectories.append(recorded_trajectory)
        if need_ideal:
            ideal_trajectory, ideal_summary = _trajectory_from_rollouts_full(
                tracks_by_chunk,
                chunks,
                ik_hz=15.0,
                profile_name=IDEAL_15HZ_PROFILE,
                args=rollout_args,
                trajectory_mode="formal",
            )
            trajectories.append(ideal_trajectory)
        if need_candidate:
            candidate_trajectory, candidate_summary = _trajectory_from_rollouts_full(
                tracks_by_chunk,
                chunks,
                ik_hz=candidate_ik_hz,
                profile_name=candidate_name,
                args=rollout_args,
                trajectory_mode=str(args.candidate_trajectory_mode),
                plan_splice_dynamics_config=splice_config,
                stream_start_acquisition_config=stream_start_config,
            )
            trajectories.append(candidate_trajectory)
    else:
        if args.chunk is None:
            selected_chunk, recorded_metrics = _auto_select_chunk(
                chunks,
                tracks_by_chunk,
            )
            selection = "auto_largest_recorded_joint_step"
        else:
            selected = [chunk for chunk in chunks if chunk.chunk_id == int(args.chunk)]
            if not selected:
                raise ValueError(f"recorded deployment has no dispatch chunk {args.chunk}")
            selected_chunk = selected[0]
            _times, _qpos, recorded_metrics = _recorded_chunk_arrays(selected_chunk)
            selection = "explicit"
        if selected_chunk.chunk_id not in tracks_by_chunk:
            raise ValueError(
                f"deployment chunk {selected_chunk.chunk_id} has no aligned Pose track"
            )
        track = tracks_by_chunk[selected_chunk.chunk_id]
        selected_chunk_id = int(selected_chunk.chunk_id)
        source_pose_frames = int(track.frames)
        recorded_trajectory, recorded_metrics = _trajectory_from_recorded(
            track,
            selected_chunk,
        )
        if args.play in {"comparison", "recorded"}:
            trajectories.append(recorded_trajectory)
        if need_ideal:
            ideal_artifact = _rollout_track_artifact(
                track,
                ik_hz=15.0,
                mode="formal",
                args=rollout_args,
            )
            if ideal_artifact.summary.get("error") is not None:
                raise RuntimeError(
                    f"15 Hz reconstruction failed: {ideal_artifact.summary['error']}"
                )
            ideal_summary = dict(ideal_artifact.summary)
            trajectories.append(
                _trajectory_from_rollout(
                    track,
                    ideal_artifact,
                    profile_name=IDEAL_15HZ_PROFILE,
                    ik_hz=15.0,
                    chunk=selected_chunk,
                )
            )
        if need_candidate:
            candidate_artifact = _rollout_track_artifact(
                track,
                ik_hz=candidate_ik_hz,
                mode=str(args.candidate_trajectory_mode),
                args=rollout_args,
            )
            if candidate_artifact.summary.get("error") is not None:
                raise RuntimeError(
                    f"{candidate_ik_hz:g} Hz candidate failed: "
                    f"{candidate_artifact.summary['error']}"
                )
            candidate_summary = dict(candidate_artifact.summary)
            trajectories.append(
                _trajectory_from_rollout(
                    track,
                    candidate_artifact,
                    profile_name=candidate_name,
                    ik_hz=candidate_ik_hz,
                    chunk=selected_chunk,
                )
            )

    payload = {
        "schema": "dexteleop.rm65_deploy_ik_rate_mujoco_comparison/v1",
        "no_hardware": True,
        "dispatch": False,
        "mujoco_role": "kinematic_arm_qpos_and_gripper_proxy_viewer",
        "source_debug_jsonl": str(debug_path),
        "selected_chunk": selected_chunk_id,
        "chunk_selection": selection,
        "source_pose_frames_15hz": source_pose_frames,
        "play_selection": str(args.play),
        "play_order": [trajectory.profile_name for trajectory in trajectories],
        "contracts": {
            "recorded_baseline": "actual_qpos_reference_and_emission_timing",
            "ideal_15hz": "same_pose_and_initial_qpos_ideal_120hz_clock",
            "candidate": "next_original_15hz_target_only",
            "candidate_mode": str(args.candidate_mode),
            "candidate_ik_hz": candidate_ik_hz,
            "candidate_trajectory_mode": str(
                args.candidate_trajectory_mode
            ),
            "candidate_qref_acceleration_profile": str(
                args.candidate_qref_acceleration_profile
            ),
            "planner_command_hz": float(args.planner_command_hz),
            "candidate_non_rtc_splice_initial_jerk_regularization": float(
                args.non_rtc_splice_initial_jerk_regularization
            ),
            "candidate_non_rtc_splice_jerk_delta_regularization": float(
                args.non_rtc_splice_jerk_delta_regularization
            ),
            "non_rtc_gap_contract": "recorded_hold_preserved_exactly",
            "command_hz": float(args.planner_command_hz),
            "collision_and_dynamics": "same_offline_official_all_ik_governor",
            "gripper": (
                "recorded_15hz_discrete_dmtac_normalized_close_on_"
                "parallel_gripper_proxy"
            ),
        },
        "recorded_metrics": recorded_metrics,
        "ideal_15hz_summary": ideal_summary,
        "candidate_summary": candidate_summary,
    }
    print(
        "[DeployIKRateSim] selected "
        f"chunk={selected_chunk_id} selection={selection} "
        f"recorded_dispatches={recorded_metrics['dispatch_count']} "
        f"recorded_max_gap_ms={recorded_metrics['max_gap_ms']} "
        f"recorded_max_step_deg={recorded_metrics['max_joint_step_deg']}",
        flush=True,
    )
    for trajectory in trajectories:
        print(
            "[DeployIKRateSim] play_order "
            f"profile={trajectory.profile_name} "
            f"duration_sec={trajectory.summary()['duration_sec']:.3f} "
            f"frames={len(trajectory.frames)} "
            f"gripper_frames="
            f"{trajectory.summary()['gripper_reference_count']}",
            flush=True,
        )
    if args.summary_json is not None:
        output = Path(args.summary_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable)
            + "\n",
            encoding="utf-8",
        )
        print(f"[DeployIKRateSim] summary_json={output}", flush=True)
    if bool(args.viewer):
        viewer_trajectories = tuple(
            _with_viewer_startup_hold(
                trajectory,
                hold_sec=float(args.viewer_startup_hold_sec),
            )
            for trajectory in trajectories
        )
        if bool(args.loop):
            args.repeat = 2_147_483_647
            print(
                "[DeployIKRateSim] viewer_loop=enabled; "
                "close the window or press Ctrl+C to stop",
                flush=True,
            )
        view_trajectories(args, viewer_trajectories)
    else:
        _validate_mujoco_playback(args, trajectories)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.repeat) <= 0:
        raise ValueError("--repeat must be positive")
    if float(args.pause_between_profiles_sec) < 0.0:
        raise ValueError("--pause-between-profiles-sec must be non-negative")
    run(args)
    return 0


def cli_main(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except KeyboardInterrupt:
        print(
            "\n[DeployIKRateSim] interrupted; viewer closed cleanly",
            flush=True,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
