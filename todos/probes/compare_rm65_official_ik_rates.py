#!/usr/bin/env python3
"""Compare RM65 all-IK source rates with one common 120 Hz trajectory layer.

This is a no-hardware research probe.  It uses the production official all-IK
candidate provider, collision model, branch selector and jerk-limited governor,
but never creates a backend or dispatches a command.

The compared variable is the Cartesian Pose -> selected QPOS sampling rate.
Input Pose paths are sampled at 15 Hz, resampled on SE(3) to 15/30/60/120 Hz,
and all variants are emitted and evaluated at 120 Hz.  At every instant only
the next original 15 Hz target is visible.  Higher IK rates may subdivide that
known interval, but may not inspect the following 15 Hz target.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_motion.position_control import (  # noqa: E402
    PositionPlanSpliceDecision,
    PositionPlanSpliceDynamicsConfig,
    ResampledMotionReference,
)
from robot_motion.rm65 import (  # noqa: E402
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
    RM65_QREF_ACCELERATION_HARD_V1,
    RM65_QREF_ACCELERATION_PROFILE_CHOICES,
    RM65PositionCollisionPlanner,
    RM65PositionCollisionPlannerConfig,
    endpoint_to_tool_transform,
    load_rm65_tool_profile,
    resolve_rm65_qref_acceleration_profile,
)
from teleop_runtime.position_control.official_all_ik_governor import (  # noqa: E402
    RM65OfficialAllIKGovernorPlanner,
)


SIDES = ("left", "right")
INPUT_HZ = 15.0
COMMAND_HZ = 120.0
DEFAULT_IK_RATES = (15.0, 30.0, 60.0, 120.0)


@dataclass(frozen=True)
class PoseTrack:
    name: str
    pose_15hz_by_side: Mapping[str, np.ndarray]
    initial_qpos_by_side: Mapping[str, np.ndarray]
    endpoint_to_tool_by_side: Mapping[str, np.ndarray]
    source: str

    @property
    def frames(self) -> int:
        return int(self.pose_15hz_by_side["left"].shape[0])


@dataclass(frozen=True)
class RolloutArtifact:
    """One no-hardware rollout plus the paths needed by visual diagnostics."""

    summary: Mapping[str, Any]
    raw_metrics: Mapping[str, list[float]]
    q_command_rad: np.ndarray
    q_ik_raw_rad: np.ndarray
    source_qpos_rad_by_index: Mapping[int, Mapping[str, np.ndarray]]


def _finite_qpos(value: Any) -> np.ndarray | None:
    try:
        qpos = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
        return None
    return qpos.copy()


def _planner_config(
    args: argparse.Namespace,
    *,
    reference_hz: float | None = None,
) -> RM65PositionCollisionPlannerConfig:
    effective_reference_hz = (
        COMMAND_HZ if reference_hz is None else float(reference_hz)
    )
    return RM65PositionCollisionPlannerConfig(
        reference_hz=effective_reference_hz,
        scene_profile=str(args.scene_profile),
        collision_min_distance_m=float(args.collision_min_distance_m),
        collision_distance_epsilon_m=float(args.collision_distance_epsilon_m),
        max_joint_velocity_rad_s=tuple(
            float(value) for value in RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S
        ),
        max_joint_acceleration_rad_s2=tuple(
            float(value)
            for value in RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2
        ),
        solve_max_iterations=1,
        solve_min_iterations=1,
        debug_profile=False,
    )


def _arm_base_endpoint_poses(
    planner: RM65PositionCollisionPlanner,
    qpos_by_side: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    world = planner.observe_endpoint_poses_world(qpos_by_side)
    return {
        side: (
            np.linalg.inv(
                np.asarray(planner.world_from_arm_base_by_side[side], dtype=float)
            )
            @ np.asarray(world[side], dtype=float)
        )
        for side in SIDES
    }


def _arm_base_tool_poses(
    planner: RM65PositionCollisionPlanner,
    qpos_by_side: Mapping[str, np.ndarray],
    endpoint_to_tool_by_side: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    endpoint = _arm_base_endpoint_poses(planner, qpos_by_side)
    return {
        side: endpoint[side]
        @ np.asarray(endpoint_to_tool_by_side[side], dtype=float).reshape(4, 4)
        for side in SIDES
    }


def _load_parquet_tracks(
    paths: Sequence[Path],
    *,
    args: argparse.Namespace,
) -> list[PoseTrack]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for --parquet") from exc

    endpoint_to_tool = {side: np.eye(4, dtype=float) for side in SIDES}
    oracle = RM65PositionCollisionPlanner(
        _planner_config(args),
        endpoint_to_tool_by_side=endpoint_to_tool,
    )
    tracks: list[PoseTrack] = []
    for path in paths:
        rows = pq.read_table(
            path,
            columns=[
                "collection.continuity_segment_index",
                "action.reference.left_arm.q_ref",
                "action.reference.right_arm.q_ref",
            ],
        ).to_pylist()
        grouped: dict[int, list[dict[str, np.ndarray]]] = {}
        for row in rows:
            left = _finite_qpos(row["action.reference.left_arm.q_ref"])
            right = _finite_qpos(row["action.reference.right_arm.q_ref"])
            if left is None or right is None:
                continue
            segment = int(row["collection.continuity_segment_index"])
            grouped.setdefault(segment, []).append({"left": left, "right": right})
        if not grouped:
            continue
        segment, qrows = max(grouped.items(), key=lambda item: len(item[1]))
        if args.max_input_frames is not None:
            qrows = qrows[: int(args.max_input_frames)]
        if len(qrows) < int(args.min_input_frames):
            continue
        poses = {side: [] for side in SIDES}
        for qpos in qrows:
            observed = _arm_base_endpoint_poses(oracle, qpos)
            for side in SIDES:
                poses[side].append(observed[side])
        tracks.append(
            PoseTrack(
                name=f"{path.stem}:segment={segment}",
                pose_15hz_by_side={
                    side: np.asarray(poses[side], dtype=float) for side in SIDES
                },
                initial_qpos_by_side={
                    side: np.asarray(qrows[0][side], dtype=float).copy()
                    for side in SIDES
                },
                endpoint_to_tool_by_side=endpoint_to_tool,
                source=str(path),
            )
        )
    return tracks


def _tool_transforms_from_debug_startup(startup: Mapping[str, Any]) -> dict[str, np.ndarray]:
    args = dict(startup.get("args") or {})
    result: dict[str, np.ndarray] = {}
    for side in SIDES:
        profile_path = args.get(f"{side}_tool_profile")
        if not profile_path:
            raise ValueError(f"debug startup is missing {side}_tool_profile")
        requested = Path(str(profile_path))
        local_equivalent = (
            PROJECT_ROOT
            / "hardware/rm65/utils/configs/calibrations"
            / requested.name
        )
        profile = load_rm65_tool_profile(
            requested if requested.exists() else local_equivalent,
            side=side,
        )
        result[side] = endpoint_to_tool_transform(profile)
    return result


def _load_debug_tracks(
    paths: Sequence[Path],
    *,
    args: argparse.Namespace,
) -> list[PoseTrack]:
    tracks: list[PoseTrack] = []
    for path in paths:
        startup: dict[str, Any] | None = None
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("event") == "startup" and startup is None:
                    startup = event
                elif event.get("event") == "action_step":
                    rows.append(event)
        if startup is None or not rows:
            continue
        endpoint_to_tool = _tool_transforms_from_debug_startup(startup)
        by_chunk: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            metadata = dict((row.get("command") or {}).get("metadata") or {})
            chunk = int(metadata.get("deploy_action_chunk_id", -1))
            by_chunk.setdefault(chunk, []).append(row)
        for chunk, chunk_rows in sorted(by_chunk.items()):
            if chunk in set(int(value) for value in args.exclude_debug_chunk):
                continue
            chunk_rows.sort(key=lambda row: int(row.get("position_source_index", -1)))
            unique: list[dict[str, Any]] = []
            seen: set[int] = set()
            for row in chunk_rows:
                source_index = int(row.get("position_source_index", -1))
                if source_index in seen:
                    continue
                seen.add(source_index)
                unique.append(row)
            if args.max_input_frames is not None:
                unique = unique[: int(args.max_input_frames)]
            if len(unique) < int(args.min_input_frames):
                continue
            poses: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
            initial: dict[str, np.ndarray] = {}
            valid = True
            for index, row in enumerate(unique):
                commands = dict((row.get("command") or {}).get("commands") or {})
                for side in SIDES:
                    item = dict(commands.get(f"{side}_arm") or {})
                    try:
                        pose = np.asarray(item.get("value"), dtype=float).reshape(4, 4)
                    except Exception:
                        valid = False
                        break
                    poses[side].append(pose)
                    if index == 0:
                        state = dict((row.get("pre_arm_state_by_side") or {}).get(side) or {})
                        qpos = _finite_qpos(
                            state.get("hardware_qpos", state.get("measured_qpos"))
                        )
                        if qpos is None:
                            valid = False
                            break
                        initial[side] = qpos
                if not valid:
                    break
            if not valid:
                continue
            tracks.append(
                PoseTrack(
                    name=f"{path.stem}:chunk={chunk}",
                    pose_15hz_by_side={
                        side: np.asarray(poses[side], dtype=float) for side in SIDES
                    },
                    initial_qpos_by_side=initial,
                    endpoint_to_tool_by_side=endpoint_to_tool,
                    source=str(path),
                )
            )
    return tracks


def _resample_pose_sequence(poses: np.ndarray, target_hz: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp

    source = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
    duration = (source.shape[0] - 1) / INPUT_HZ
    count = int(round(duration * float(target_hz))) + 1
    target_times = np.arange(count, dtype=float) / float(target_hz)
    source_times = np.arange(source.shape[0], dtype=float) / INPUT_HZ
    result = np.repeat(np.eye(4, dtype=float)[None, :, :], count, axis=0)
    for axis in range(3):
        result[:, axis, 3] = np.interp(
            target_times,
            source_times,
            source[:, axis, 3],
        )
    rotations = Rotation.from_matrix(source[:, :3, :3])
    result[:, :3, :3] = Slerp(source_times, rotations)(target_times).as_matrix()
    return result


def _pose_errors(
    target: np.ndarray,
    realized: np.ndarray,
) -> tuple[float, float]:
    translation_mm = 1000.0 * float(
        np.linalg.norm(target[:3, 3] - realized[:3, 3])
    )
    relative = target[:3, :3].T @ realized[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return translation_mm, float(np.rad2deg(math.acos(cosine)))


def _percentiles(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"p50": None, "p95": None, "p99": None, "max": None}
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _derivative_max_samples(
    q_path: np.ndarray,
    initial_qpos: np.ndarray,
    *,
    sample_hz: float = COMMAND_HZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.vstack((np.asarray(initial_qpos, dtype=float).reshape(1, -1), q_path))
    hz = float(sample_hz)
    velocity = np.diff(q, axis=0) * hz
    acceleration = np.diff(
        np.vstack((np.zeros((1, q.shape[1]), dtype=float), velocity)),
        axis=0,
    ) * hz
    jerk = np.diff(
        np.vstack((np.zeros((1, q.shape[1]), dtype=float), acceleration)),
        axis=0,
    ) * hz
    scale = 180.0 / math.pi
    return (
        np.max(np.abs(velocity), axis=1) * scale,
        np.max(np.abs(acceleration), axis=1) * scale,
        np.max(np.abs(jerk), axis=1) * scale,
    )


def _zero_crossings(values: np.ndarray, *, deadband_deg: float = 0.1) -> int:
    degrees = np.rad2deg(np.asarray(values, dtype=float))
    sign = np.where(degrees > deadband_deg, 1, np.where(degrees < -deadband_deg, -1, 0))
    previous = 0
    count = 0
    for value in sign:
        if value == 0:
            continue
        if previous != 0 and value != previous:
            count += 1
        previous = int(value)
    return count


def _rollout_track_artifact(
    track: PoseTrack,
    *,
    ik_hz: float,
    mode: str,
    args: argparse.Namespace,
    plan_splice_decision: PositionPlanSpliceDecision | None = None,
    plan_splice_dynamics_config: PositionPlanSpliceDynamicsConfig | None = None,
) -> RolloutArtifact:
    command_hz = float(getattr(args, "command_hz", COMMAND_HZ))
    qref_acceleration_profile = str(
        getattr(
            args,
            "qref_acceleration_profile",
            RM65_QREF_ACCELERATION_HARD_V1,
        )
    )
    (
        qref_acceleration_scale,
        qref_max_acceleration_rad_s2,
    ) = resolve_rm65_qref_acceleration_profile(
        qref_acceleration_profile,
        hard_max_acceleration_rad_s2=(
            RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2
        ),
    )
    source_poses = {
        side: _resample_pose_sequence(track.pose_15hz_by_side[side], ik_hz)
        for side in SIDES
    }
    target_120 = {
        side: _resample_pose_sequence(track.pose_15hz_by_side[side], command_hz)
        for side in SIDES
    }
    source_count = int(source_poses["left"].shape[0])
    command_count = int(target_120["left"].shape[0] - 1)
    subdivisions_per_15hz_interval = int(round(ik_hz / INPUT_HZ))
    # Match the production 15 Hz governor's 133 ms dynamic horizon at every
    # rate.  Only the first 66.7 ms contains visible, interpolated targets; the
    # second half repeats the last visible target and exposes no future input.
    lookahead_steps = 2 * subdivisions_per_15hz_interval
    collision = RM65PositionCollisionPlanner(
        _planner_config(args, reference_hz=command_hz),
        endpoint_to_tool_by_side=track.endpoint_to_tool_by_side,
    )
    planner = RM65OfficialAllIKGovernorPlanner(
        collision,
        endpoint_to_tool_by_side=track.endpoint_to_tool_by_side,
        source_hz=ik_hz,
        command_hz=command_hz,
        lookahead_source_steps=lookahead_steps,
        jerk_ramp_sec=float(args.jerk_ramp_sec),
        batched_solver_tolerance=float(args.batched_solver_tolerance),
        continuous_waypoint_shadow_enabled=(mode == "c2"),
        continuous_waypoint_admission_enabled=(mode == "c2"),
        continuous_waypoint_async_precompute_enabled=False,
        plan_splice_dynamics_config=plan_splice_dynamics_config,
        max_velocity_rad_s=RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
        max_acceleration_rad_s2=qref_max_acceleration_rad_s2,
    )
    planner.reset(track.initial_qpos_by_side)
    last_staged = -1
    q_path: list[np.ndarray] = []
    q_raw_path: list[np.ndarray] = []
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    tick_latencies: list[float] = []
    cut_latencies: list[float] = []
    new_source_solutions = 0
    candidate_counts: list[int] = []
    execution_modes: Counter[str] = Counter()
    continuous_statuses: Counter[str] = Counter()
    continuous_admission_statuses: Counter[str] = Counter()
    splice_regularized_cuts = 0
    splice_fallback_cuts = 0
    error: str | None = None
    try:
        for command_index in range(command_count):
            call_time = command_index / command_hz
            source_index = min(
                int(math.floor(call_time * ik_hz + 1e-9)),
                source_count - 1,
            )
            current_input_interval = min(
                int(math.floor(call_time * INPUT_HZ + 1e-9)),
                track.frames - 2,
            )
            next_visible_input_index = current_input_interval + 1
            stage_through = min(
                source_count - 1,
                next_visible_input_index * subdivisions_per_15hz_interval,
            )
            while last_staged < stage_through:
                last_staged += 1
                planner.stage_source_pose(
                    source_index=last_staged,
                    source_timestamp=last_staged / ik_hz,
                    pose_by_side={
                        side: source_poses[side][last_staged] for side in SIDES
                    },
                    plan_splice_decision=(
                        plan_splice_decision if last_staged == 0 else None
                    ),
                )
            reference = ResampledMotionReference(
                sequence=command_index,
                timestamp=call_time,
                left_source_index=source_index,
                right_source_index=source_index,
                left_source_timestamp=source_index / ik_hz,
                right_source_timestamp=source_index / ik_hz,
                interpolation_alpha=float(
                    np.clip((call_time * ik_hz) - source_index, 0.0, 1.0)
                ),
                pose_by_side={
                    side: target_120[side][command_index] for side in SIDES
                },
                qpos_by_side={},
                metadata={"probe": "compare_rm65_official_ik_rates"},
            )
            started = time.perf_counter()
            result = planner.plan_pose_reference_horizon(
                reference,
                source_timestamp=call_time,
                source="offline_ik_rate_comparison",
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            metadata = dict(result.metadata)
            tick_latencies.append(elapsed_ms)
            solved = int(metadata.get("new_source_solutions", 0))
            new_source_solutions += solved
            if solved:
                cut_latencies.append(elapsed_ms)
            for counts in dict(metadata.get("candidate_counts_by_source") or {}).values():
                for side in SIDES:
                    if side in counts:
                        candidate_counts.append(int(counts[side]))
            execution_modes[str(metadata.get("governor_execution_mode"))] += 1
            continuous = dict(metadata.get("continuous_waypoint_shadow") or {})
            if continuous:
                continuous_statuses[str(continuous.get("status"))] += 1
                continuous_admission_statuses[
                    str(continuous.get("admission_status"))
                ] += 1
            if bool(metadata.get("source_cut_replanned", False)):
                splice_regularized_cuts += int(
                    bool(
                        metadata.get(
                            "plan_splice_initial_jerk_regularization_applied",
                            False,
                        )
                    )
                )
                splice_fallback_cuts += int(
                    bool(
                        metadata.get(
                            "plan_splice_initial_jerk_regularization_fallback",
                            False,
                        )
                    )
                )
            qcmd = np.concatenate(
                [np.asarray(result.q_ref_by_side[side], dtype=float) for side in SIDES]
            )
            qraw = np.concatenate(
                [np.asarray(result.q_ik_raw_by_side[side], dtype=float) for side in SIDES]
            )
            q_path.append(qcmd)
            q_raw_path.append(qraw)
            realized = _arm_base_tool_poses(
                collision,
                result.q_ref_by_side,
                track.endpoint_to_tool_by_side,
            )
            desired_index = command_index + 1
            for side in SIDES:
                translation, rotation = _pose_errors(
                    target_120[side][desired_index],
                    realized[side],
                )
                translation_errors.append(translation)
                rotation_errors.append(rotation)
    except Exception as exc:  # noqa: BLE001 - failures are comparison evidence.
        error = f"{type(exc).__name__}: {exc}"
    finally:
        planner.close()

    q_array = np.asarray(q_path, dtype=float).reshape(-1, 12)
    q_raw_array = np.asarray(q_raw_path, dtype=float).reshape(-1, 12)
    initial_dual = np.concatenate(
        [track.initial_qpos_by_side[side] for side in SIDES]
    )
    velocity, acceleration, jerk = _derivative_max_samples(
        q_array,
        initial_dual,
        sample_hz=command_hz,
    ) if q_array.size else (np.asarray([]), np.asarray([]), np.asarray([]))
    selected = {
        int(index): {
            side: np.asarray(values[side], dtype=float)
            for side in SIDES
        }
        for index, values in planner._source_qpos.items()
    }
    selected_order = sorted(selected)
    selected_dual = np.asarray(
        [
            np.concatenate([selected[index][side] for side in SIDES])
            for index in selected_order
        ],
        dtype=float,
    ).reshape(-1, 12)
    selected_steps_deg = (
        np.max(np.abs(np.diff(selected_dual, axis=0)), axis=1) * 180.0 / math.pi
        if selected_dual.shape[0] >= 2
        else np.asarray([], dtype=float)
    )
    source_translation_errors: list[float] = []
    source_rotation_errors: list[float] = []
    for index in selected_order:
        realized_source = _arm_base_tool_poses(
            collision,
            selected[index],
            track.endpoint_to_tool_by_side,
        )
        for side in SIDES:
            translation, rotation = _pose_errors(
                source_poses[side][index],
                realized_source[side],
            )
            source_translation_errors.append(translation)
            source_rotation_errors.append(rotation)
    singularity_values: dict[str, list[float]] = {side: [] for side in SIDES}
    for row in q_array[:: max(1, int(command_hz // 30))]:
        diagnostics = collision.analyze_configuration(
            {"left": row[:6], "right": row[6:]}
        )
        for side in SIDES:
            singularity_values[side].append(
                float(diagnostics["sides"][side]["sigma_min"])
            )
    raw_error_deg = (
        np.max(np.abs(q_array - q_raw_array), axis=1) * 180.0 / math.pi
        if q_array.size
        else np.asarray([], dtype=float)
    )
    steady_skip = int(round(command_hz / INPUT_HZ))
    steady_translation_errors = translation_errors[2 * steady_skip :]
    steady_rotation_errors = rotation_errors[2 * steady_skip :]
    steady_velocity = velocity[steady_skip:]
    steady_acceleration = acceleration[steady_skip:]
    steady_jerk = jerk[steady_skip:]
    summary = {
        "track": track.name,
        "source": track.source,
        "input_frames_15hz": track.frames,
        "duration_sec": (track.frames - 1) / INPUT_HZ,
        "ik_hz": float(ik_hz),
        "command_hz": command_hz,
        "maximum_visible_future_sec": 1.0 / INPUT_HZ,
        "dynamic_planning_horizon_sec": lookahead_steps / ik_hz,
        "terminal_hold_after_visible_target_sec": 1.0 / INPUT_HZ,
        "visible_future_contract": "next_original_15hz_target_only",
        "subdivisions_per_15hz_interval": subdivisions_per_15hz_interval,
        "lookahead_source_steps": lookahead_steps,
        "trajectory_mode_requested": mode,
        "qref_acceleration_profile": qref_acceleration_profile,
        "qref_acceleration_scale": float(qref_acceleration_scale),
        "qref_max_acceleration_rad_s2": list(
            qref_max_acceleration_rad_s2
        ),
        "qref_max_jerk_rad_s3": [
            float(value) / float(args.jerk_ramp_sec)
            for value in qref_max_acceleration_rad_s2
        ],
        "plan_splice_mode": (
            None if plan_splice_decision is None else plan_splice_decision.mode
        ),
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
        "plan_splice_regularized_cut_count": int(splice_regularized_cuts),
        "plan_splice_fallback_cut_count": int(splice_fallback_cuts),
        "error": error,
        "command_count": int(q_array.shape[0]),
        "source_solution_count": int(new_source_solutions),
        "candidate_count_per_side": dict(Counter(candidate_counts)),
        "execution_mode_counts": dict(execution_modes),
        "continuous_waypoint_status_counts": dict(continuous_statuses),
        "continuous_waypoint_admission_status_counts": dict(
            continuous_admission_statuses
        ),
        "pose_tracking": {
            "translation_error_mm": _percentiles(translation_errors),
            "rotation_error_deg": _percentiles(rotation_errors),
        },
        "pose_tracking_after_first_15hz_interval": {
            "translation_error_mm": _percentiles(steady_translation_errors),
            "rotation_error_deg": _percentiles(steady_rotation_errors),
        },
        "source_ik_pose_residual": {
            "translation_error_mm": _percentiles(source_translation_errors),
            "rotation_error_deg": _percentiles(source_rotation_errors),
        },
        "joint_dynamics": {
            "max_joint_velocity_deg_s": _percentiles(velocity),
            "max_joint_acceleration_deg_s2": _percentiles(acceleration),
            "max_joint_jerk_deg_s3": _percentiles(jerk),
            "governor_vs_raw_ik_max_joint_error_deg": _percentiles(raw_error_deg),
        },
        "joint_dynamics_after_first_15hz_interval": {
            "max_joint_velocity_deg_s": _percentiles(steady_velocity),
            "max_joint_acceleration_deg_s2": _percentiles(steady_acceleration),
            "max_joint_jerk_deg_s3": _percentiles(steady_jerk),
        },
        "source_branch_continuity": {
            "max_joint_step_deg": _percentiles(selected_steps_deg),
            "steps_over_10deg": int(np.sum(selected_steps_deg > 10.0)),
            "steps_over_30deg": int(np.sum(selected_steps_deg > 30.0)),
            "left_q3_zero_crossings": (
                _zero_crossings(selected_dual[:, 2]) if selected_dual.size else 0
            ),
            "left_q5_zero_crossings": (
                _zero_crossings(selected_dual[:, 4]) if selected_dual.size else 0
            ),
            "right_q3_zero_crossings": (
                _zero_crossings(selected_dual[:, 8]) if selected_dual.size else 0
            ),
            "right_q5_zero_crossings": (
                _zero_crossings(selected_dual[:, 10]) if selected_dual.size else 0
            ),
        },
        "singularity": {
            side: {
                "sigma_min": _percentiles(singularity_values[side]),
                "minimum": (
                    None
                    if not singularity_values[side]
                    else float(min(singularity_values[side]))
                ),
            }
            for side in SIDES
        },
        "compute": {
            "all_ticks_ms": _percentiles(tick_latencies),
            "source_cut_ms": _percentiles(cut_latencies),
            "ticks_over_command_period": int(
                np.sum(np.asarray(tick_latencies) > 1000.0 / command_hz)
            ),
            "source_cuts_over_period": int(
                np.sum(np.asarray(cut_latencies) > 1000.0 / ik_hz)
            ),
            "estimated_total_compute_ms": float(np.sum(tick_latencies)),
            "estimated_compute_ms_per_wall_sec": (
                None
                if q_array.shape[0] == 0
                else float(np.sum(tick_latencies))
                / (q_array.shape[0] / command_hz)
            ),
        },
    }
    raw = {
        "translation_error_mm": translation_errors,
        "rotation_error_deg": rotation_errors,
        "steady_translation_error_mm": steady_translation_errors,
        "steady_rotation_error_deg": steady_rotation_errors,
        "source_translation_error_mm": source_translation_errors,
        "source_rotation_error_deg": source_rotation_errors,
        "velocity_deg_s": velocity.tolist(),
        "acceleration_deg_s2": acceleration.tolist(),
        "jerk_deg_s3": jerk.tolist(),
        "steady_velocity_deg_s": steady_velocity.tolist(),
        "steady_acceleration_deg_s2": steady_acceleration.tolist(),
        "steady_jerk_deg_s3": steady_jerk.tolist(),
        "source_step_deg": selected_steps_deg.tolist(),
        "tick_latency_ms": tick_latencies,
        "cut_latency_ms": cut_latencies,
    }
    return RolloutArtifact(
        summary=summary,
        raw_metrics=raw,
        q_command_rad=q_array.copy(),
        q_ik_raw_rad=q_raw_array.copy(),
        source_qpos_rad_by_index={
            index: {
                side: np.asarray(values[side], dtype=float).copy()
                for side in SIDES
            }
            for index, values in selected.items()
        },
    )


def _rollout_track(
    track: PoseTrack,
    *,
    ik_hz: float,
    mode: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    artifact = _rollout_track_artifact(
        track,
        ik_hz=ik_hz,
        mode=mode,
        args=args,
    )
    return dict(artifact.summary), {
        key: list(values) for key, values in artifact.raw_metrics.items()
    }


def _aggregate(
    entries: Sequence[tuple[dict[str, Any], dict[str, list[float]]]],
    *,
    ik_hz: float,
    mode: str,
) -> dict[str, Any]:
    successful = [entry for entry in entries if entry[0]["error"] is None]
    command_hz = (
        COMMAND_HZ
        if not successful
        else float(successful[0][0]["command_hz"])
    )
    def values(name: str) -> list[float]:
        return [value for _summary, raw in successful for value in raw[name]]

    def merged_counts(name: str) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for summary, _raw in successful:
            counts.update(dict(summary.get(name) or {}))
        return dict(counts)

    return {
        "ik_hz": float(ik_hz),
        "trajectory_mode": mode,
        "tracks": len(entries),
        "successful_tracks": len(successful),
        "failed_tracks": len(entries) - len(successful),
        "failures": [
            {"track": summary["track"], "error": summary["error"]}
            for summary, _raw in entries
            if summary["error"] is not None
        ],
        "commands": int(sum(summary["command_count"] for summary, _raw in successful)),
        "execution_mode_counts": merged_counts("execution_mode_counts"),
        "continuous_waypoint_status_counts": merged_counts(
            "continuous_waypoint_status_counts"
        ),
        "continuous_waypoint_admission_status_counts": merged_counts(
            "continuous_waypoint_admission_status_counts"
        ),
        "pose_tracking": {
            "translation_error_mm": _percentiles(values("translation_error_mm")),
            "rotation_error_deg": _percentiles(values("rotation_error_deg")),
        },
        "pose_tracking_after_first_15hz_interval": {
            "translation_error_mm": _percentiles(values("steady_translation_error_mm")),
            "rotation_error_deg": _percentiles(values("steady_rotation_error_deg")),
        },
        "source_ik_pose_residual": {
            "translation_error_mm": _percentiles(values("source_translation_error_mm")),
            "rotation_error_deg": _percentiles(values("source_rotation_error_deg")),
        },
        "joint_dynamics": {
            "max_joint_velocity_deg_s": _percentiles(values("velocity_deg_s")),
            "max_joint_acceleration_deg_s2": _percentiles(values("acceleration_deg_s2")),
            "max_joint_jerk_deg_s3": _percentiles(values("jerk_deg_s3")),
        },
        "joint_dynamics_after_first_15hz_interval": {
            "max_joint_velocity_deg_s": _percentiles(values("steady_velocity_deg_s")),
            "max_joint_acceleration_deg_s2": _percentiles(values("steady_acceleration_deg_s2")),
            "max_joint_jerk_deg_s3": _percentiles(values("steady_jerk_deg_s3")),
        },
        "source_branch_continuity": {
            "max_joint_step_deg": _percentiles(values("source_step_deg")),
            "steps_over_10deg": int(sum(value > 10.0 for value in values("source_step_deg"))),
            "steps_over_30deg": int(sum(value > 30.0 for value in values("source_step_deg"))),
        },
        "compute": {
            "all_ticks_ms": _percentiles(values("tick_latency_ms")),
            "source_cut_ms": _percentiles(values("cut_latency_ms")),
            "ticks_over_command_period": int(
                sum(
                    value > 1000.0 / command_hz
                    for value in values("tick_latency_ms")
                )
            ),
            "source_cuts_over_period": int(sum(value > 1000.0 / ik_hz for value in values("cut_latency_ms"))),
            "estimated_compute_ms_per_wall_sec": (
                None
                if not successful
                else float(sum(values("tick_latency_ms")))
                / float(sum(summary["duration_sec"] for summary, _raw in successful))
            ),
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", action="append", default=[])
    parser.add_argument("--debug-jsonl", action="append", default=[])
    parser.add_argument("--exclude-debug-chunk", action="append", type=int, default=[])
    parser.add_argument("--ik-hz", action="append", type=float, default=[])
    parser.add_argument(
        "--command-hz",
        type=float,
        choices=(60.0, 120.0),
        default=COMMAND_HZ,
        help=(
            "Planner output cadence. Use 60 for the current deploy reference "
            "layer; 120 preserves the legacy probe contract."
        ),
    )
    parser.add_argument("--trajectory-mode", choices=("formal", "c2", "both"), default="formal")
    parser.add_argument("--min-input-frames", type=int, default=8)
    parser.add_argument("--max-input-frames", type=int, default=None)
    parser.add_argument("--scene-profile", default="dual_arm_hand_proxy_open")
    parser.add_argument("--collision-min-distance-m", type=float, default=0.03)
    parser.add_argument("--collision-distance-epsilon-m", type=float, default=1e-4)
    parser.add_argument("--jerk-ramp-sec", type=float, default=0.05)
    parser.add_argument(
        "--qref-acceleration-profile",
        choices=RM65_QREF_ACCELERATION_PROFILE_CHOICES,
        default=RM65_QREF_ACCELERATION_HARD_V1,
        help=(
            "Scale only the offline planned-qref acceleration and derived "
            "jerk envelope; the hard_v1 default preserves prior results."
        ),
    )
    parser.add_argument("--batched-solver-tolerance", type=float, default=1e-6)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args(argv)
    if not args.parquet and not args.debug_jsonl:
        parser.error("provide at least one --parquet or --debug-jsonl")
    rates = tuple(args.ik_hz or DEFAULT_IK_RATES)
    if any(rate not in DEFAULT_IK_RATES for rate in rates):
        parser.error("--ik-hz must be one of 15, 30, 60, 120")
    args.ik_hz = rates
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    tracks = _load_parquet_tracks(
        [Path(value).resolve() for value in args.parquet],
        args=args,
    )
    tracks.extend(
        _load_debug_tracks(
            [Path(value).resolve() for value in args.debug_jsonl],
            args=args,
        )
    )
    if not tracks:
        raise RuntimeError("no usable tracks were loaded")
    modes = ("formal", "c2") if args.trajectory_mode == "both" else (args.trajectory_mode,)
    per_track: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    started = time.perf_counter()
    for mode in modes:
        for ik_hz in args.ik_hz:
            entries: list[tuple[dict[str, Any], dict[str, list[float]]]] = []
            for track in tracks:
                print(
                    f"[IKRate] mode={mode} ik_hz={ik_hz:g} track={track.name}",
                    flush=True,
                )
                entry = _rollout_track(
                    track,
                    ik_hz=float(ik_hz),
                    mode=mode,
                    args=args,
                )
                entries.append(entry)
                per_track.append(entry[0])
            aggregate.append(_aggregate(entries, ik_hz=float(ik_hz), mode=mode))
    result = {
        "schema": "dexteleop.rm65_official_ik_rate_comparison/v1",
        "no_hardware": True,
        "contract": {
            "input_pose_hz": INPUT_HZ,
            "command_output_hz": float(args.command_hz),
            "ik_rates_hz": list(args.ik_hz),
            "pose_resampling": "linear_translation_shortest_quaternion_slerp",
            "maximum_visible_future_sec": 1.0 / INPUT_HZ,
            "fixed_dynamic_planning_horizon_sec": 2.0 / INPUT_HZ,
            "terminal_hold_after_visible_target_sec": 1.0 / INPUT_HZ,
            "future_visibility": (
                "Only the next original 15 Hz target is visible. Higher IK "
                "rates subdivide that interval and never see the target after it."
            ),
            "same_candidate_provider": "realman_official_all_ik",
            "same_branch_selector": "nearest_collision_safe_pair",
            "same_dynamics": "formal_third_order_jerk_horizon_for_primary_comparison",
            "current_runtime_note": (
                "The current deployment planner emits at 60 Hz and the RM65 "
                "backend owns the separate downstream 120 Hz qcmd stream."
            ),
        },
        "elapsed_sec": time.perf_counter() - started,
        "inputs": [
            {
                "track": track.name,
                "source": track.source,
                "frames_15hz": track.frames,
            }
            for track in tracks
        ],
        "aggregate": aggregate,
        "per_track": per_track,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(payload + "\n", encoding="utf-8")
    for item in aggregate:
        print(
            "[IKRateSummary] "
            f"mode={item['trajectory_mode']} ik_hz={item['ik_hz']:g} "
            f"ok={item['successful_tracks']}/{item['tracks']} "
            f"steady_translation_p95_mm={item['pose_tracking_after_first_15hz_interval']['translation_error_mm']['p95']} "
            f"steady_acceleration_p95_deg_s2={item['joint_dynamics_after_first_15hz_interval']['max_joint_acceleration_deg_s2']['p95']} "
            f"steady_jerk_p95_deg_s3={item['joint_dynamics_after_first_15hz_interval']['max_joint_jerk_deg_s3']['p95']} "
            f"cut_p95_ms={item['compute']['source_cut_ms']['p95']}",
            flush=True,
        )
    print(f"[IKRate] result_json={args.output_json.resolve()}", flush=True)
    if args.print_json:
        print(payload)
    return 0 if all(item["failed_tracks"] == 0 for item in aggregate) else 1


if __name__ == "__main__":
    raise SystemExit(main())
