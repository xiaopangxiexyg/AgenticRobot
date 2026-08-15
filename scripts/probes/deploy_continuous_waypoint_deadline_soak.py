#!/usr/bin/env python3
"""No-hardware wall-clock soak for the 15->120 Hz RM65 deploy executor.

This probe intentionally exercises the real :class:`PositionDeployExecutor`,
official all-IK governor, coupled collision model, and optional asynchronous C2
waypoint precompute worker.  The only substituted boundary is arm dispatch: a
bounded in-memory callback returns the same accepted receipt shape as the real
backend and never opens a device or network connection.

Recorded accepted q_ref is used only to construct reachable endpoint targets
through the shared RM65 FK observer.  The executor still has to lift those
poses with official all-IK and execute its normal 15 Hz source / 120 Hz command
contracts.  This makes the probe suitable for scheduler/deadline and lifecycle
soak testing, not for judging model-vs-real tracking error.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_motion import pose_matrix_to_xyzquat  # noqa: E402
from robot_motion.deploy.actions import (  # noqa: E402
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    FilteredNormalizedAction,
    NormalizedAction,
)
from robot_motion.deploy.canonical_action import (  # noqa: E402
    CanonicalDeployActionPipeline,
)
from robot_motion.deploy.observation import DeploymentObservation  # noqa: E402
from robot_motion.deploy.schema import (  # noqa: E402
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    OBSERVATION_STATE_DIM,
    STATE_ARM_POSE_SLICES,
    STATE_ARM_QPOS_SLICES,
)
from robot_motion.rm65 import (  # noqa: E402
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
    RM65PositionCollisionPlanner,
    RM65PositionCollisionPlannerConfig,
)
from teleop_runtime.position_control.deploy_executor import (  # noqa: E402
    PositionDeployExecutor,
    PositionDeployExecutorConfig,
)
from teleop_runtime.position_control.official_all_ik_governor import (  # noqa: E402
    RM65OfficialAllIKGovernorPlanner,
)


SIDES = ("left", "right")
SOURCE_HZ = 15.0
COMMAND_HZ = 120.0


@dataclass(frozen=True)
class Track:
    path: Path
    segment: int
    q_left: np.ndarray
    q_right: np.ndarray
    clipped_values: int

    @property
    def source_frames(self) -> int:
        return int(self.q_left.shape[0])


def _finite_qpos(value: Any) -> np.ndarray | None:
    try:
        qpos = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
        return None
    return qpos


def _clip_roundoff(qpos: np.ndarray, *, tolerance_rad: float) -> tuple[np.ndarray, int]:
    lower = np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float)
    upper = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float)
    below = lower - qpos
    above = qpos - upper
    if np.any(below > tolerance_rad) or np.any(above > tolerance_rad):
        raise ValueError("recorded q_ref exceeds an absolute joint limit beyond tolerance")
    clipped = np.clip(qpos, lower, upper)
    return clipped, int(np.count_nonzero(clipped != qpos))


def _load_tracks(
    paths: Sequence[Path],
    *,
    min_source_frames: int,
    max_source_frames: int | None,
    joint_limit_tolerance_rad: float,
) -> list[Track]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read episode parquet files; run this probe "
            "inside the DexTeleop runtime environment"
        ) from exc
    tracks: list[Track] = []
    for path in paths:
        table = pq.read_table(
            path,
            columns=[
                "collection.continuity_segment_index",
                "action.reference.left_arm.q_ref",
                "action.reference.right_arm.q_ref",
            ],
        ).to_pylist()
        by_segment: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        clipped_by_segment: Counter[int] = Counter()
        for row in table:
            q_left = _finite_qpos(row["action.reference.left_arm.q_ref"])
            q_right = _finite_qpos(row["action.reference.right_arm.q_ref"])
            if q_left is None or q_right is None:
                continue
            segment = int(row["collection.continuity_segment_index"])
            try:
                q_left, left_clipped = _clip_roundoff(
                    q_left,
                    tolerance_rad=joint_limit_tolerance_rad,
                )
                q_right, right_clipped = _clip_roundoff(
                    q_right,
                    tolerance_rad=joint_limit_tolerance_rad,
                )
            except ValueError:
                continue
            by_segment.setdefault(segment, []).append((q_left, q_right))
            clipped_by_segment[segment] += left_clipped + right_clipped
        if not by_segment:
            continue
        segment, rows = max(by_segment.items(), key=lambda item: len(item[1]))
        if len(rows) < min_source_frames:
            continue
        if max_source_frames is not None:
            rows = rows[:max_source_frames]
        tracks.append(
            Track(
                path=path,
                segment=segment,
                q_left=np.asarray([row[0] for row in rows], dtype=float),
                q_right=np.asarray([row[1] for row in rows], dtype=float),
                clipped_values=int(clipped_by_segment[segment]),
            )
        )
    if not tracks:
        raise RuntimeError("no usable dual-arm q_ref continuity segment was found")
    return tracks


def _planner_config(args: argparse.Namespace) -> RM65PositionCollisionPlannerConfig:
    return RM65PositionCollisionPlannerConfig(
        reference_hz=COMMAND_HZ,
        scene_profile=str(args.scene_profile),
        collision_min_distance_m=float(args.collision_min_distance_m),
        collision_distance_epsilon_m=float(args.collision_distance_epsilon_m),
        max_joint_velocity_rad_s=tuple(
            float(value) for value in RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S
        ),
        max_joint_acceleration_rad_s2=tuple(
            float(value) for value in RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2
        ),
        solve_max_iterations=1,
        solve_min_iterations=1,
        debug_profile=False,
    )


class _DelayedAsyncGovernor(RM65OfficialAllIKGovernorPlanner):
    """Probe-only deterministic delay injection at the proposal worker."""

    def __init__(
        self,
        *args: Any,
        precompute_delay_ms: float,
        precompute_spike_every: int,
        precompute_spike_ms: float,
        **kwargs: Any,
    ) -> None:
        self._probe_precompute_delay_sec = float(precompute_delay_ms) / 1000.0
        self._probe_precompute_spike_every = int(precompute_spike_every)
        self._probe_precompute_spike_sec = float(precompute_spike_ms) / 1000.0
        self._probe_precompute_build_count = 0
        self._probe_precompute_lock = threading.Lock()
        self.probe_anchor_mismatches: list[dict[str, float | int]] = []
        super().__init__(*args, **kwargs)

    def _build_continuous_waypoint_request(self, request):  # type: ignore[override]
        with self._probe_precompute_lock:
            self._probe_precompute_build_count += 1
            build_index = self._probe_precompute_build_count
        delay = self._probe_precompute_delay_sec
        if (
            self._probe_precompute_spike_every > 0
            and build_index % self._probe_precompute_spike_every == 0
        ):
            delay += self._probe_precompute_spike_sec
        if delay > 0.0:
            time.sleep(delay)
        return super()._build_continuous_waypoint_request(request)

    def _consume_continuous_waypoint_precompute(self, **kwargs):  # type: ignore[override]
        request, _ = self._make_continuous_waypoint_build_request(
            interval_index=int(kwargs["interval_index"]),
            horizon_last_index=int(kwargs["horizon_last_index"]),
            interval_timestamp=float(kwargs["interval_timestamp"]),
            initial_state=kwargs["initial_state"],
        )
        if request is not None:
            with self._continuous_precompute_condition:
                candidates = list(self._continuous_precompute_results.values())
            if request.key not in self._continuous_precompute_results and candidates:
                previous = candidates[-1].request
                self.probe_anchor_mismatches.append(
                    {
                        "interval_index": int(request.interval_index),
                        "candidate_interval_index": int(previous.interval_index),
                        "q_max_abs": float(
                            np.max(
                                np.abs(
                                    request.dual_waypoints[0]
                                    - previous.dual_waypoints[0]
                                )
                            )
                        ),
                        "qd_max_abs": float(
                            np.max(
                                np.abs(
                                    request.initial_velocity_rad_s
                                    - previous.initial_velocity_rad_s
                                )
                            )
                        ),
                        "qdd_max_abs": float(
                            np.max(
                                np.abs(
                                    request.initial_acceleration_rad_s2
                                    - previous.initial_acceleration_rad_s2
                                )
                            )
                        ),
                        "waypoints_max_abs": float(
                            np.max(
                                np.abs(
                                    request.dual_waypoints
                                    - previous.dual_waypoints
                                )
                            )
                        ),
                        "timestamps_max_abs": float(
                            np.max(np.abs(request.timestamps - previous.timestamps))
                        ),
                        "terminal_velocity_max_abs": float(
                            np.max(
                                np.abs(
                                    request.terminal_velocity_rad_s
                                    - previous.terminal_velocity_rad_s
                                )
                            )
                        ),
                        "sample_timestamps_max_abs": float(
                            np.max(
                                np.abs(
                                    request.sample_timestamps
                                    - previous.sample_timestamps
                                )
                            )
                        ),
                    }
                )
        return super()._consume_continuous_waypoint_precompute(**kwargs)


def _make_governor_factory(
    args: argparse.Namespace,
    created_governors: list[_DelayedAsyncGovernor],
):
    endpoint_to_tool = {side: np.eye(4, dtype=float) for side in SIDES}

    def factory() -> RM65OfficialAllIKGovernorPlanner:
        canonical = RM65PositionCollisionPlanner(
            _planner_config(args),
            endpoint_to_tool_by_side=endpoint_to_tool,
        )
        proposal = RM65PositionCollisionPlanner(
            _planner_config(args),
            endpoint_to_tool_by_side=endpoint_to_tool,
        )
        governor = _DelayedAsyncGovernor(
            canonical,
            continuous_waypoint_precompute_collision_planner=proposal,
            endpoint_to_tool_by_side=endpoint_to_tool,
            source_hz=SOURCE_HZ,
            command_hz=COMMAND_HZ,
            lookahead_source_steps=2,
            jerk_ramp_sec=float(args.jerk_ramp_sec),
            batched_solver_tolerance=float(args.batched_solver_tolerance),
            continuous_waypoint_shadow_enabled=True,
            continuous_waypoint_admission_enabled=True,
            continuous_waypoint_async_precompute_enabled=True,
            max_velocity_rad_s=RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
            max_acceleration_rad_s2=(
                RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2
            ),
            precompute_delay_ms=float(args.precompute_delay_ms),
            precompute_spike_every=int(args.precompute_spike_every),
            precompute_spike_ms=float(args.precompute_spike_ms),
        )
        created_governors.append(governor)
        return governor

    return factory


def _pipeline() -> CanonicalDeployActionPipeline:
    return CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=SIDES,
                arms_enabled=True,
                hands_enabled=False,
                max_arm_xyz_step_m=0.0,
                max_arm_rot_step_deg=0.0,
                max_arm_joint_step_rad=0.0,
            )
        ),
        active_sides=SIDES,
        arms_enabled=True,
        hands_enabled=False,
    )


def _filtered_pose(poses: Mapping[str, np.ndarray]) -> FilteredNormalizedAction:
    vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
    for side in SIDES:
        vector[ACTION_ARM_POSE_SLICES[side]] = pose_matrix_to_xyzquat(poses[side])
    return FilteredNormalizedAction(
        action=NormalizedAction(
            vector=vector,
            metadata={"source": "no_hardware_deadline_soak"},
        ),
        metadata={"safety_pass_count": 1},
    )


def _observation(
    qpos_by_side: Mapping[str, np.ndarray],
    pose_by_side: Mapping[str, np.ndarray],
    *,
    timestamp_monotonic: float,
    step_index: int,
) -> DeploymentObservation:
    state = np.full((OBSERVATION_STATE_DIM,), np.nan, dtype=np.float32)
    valid = np.zeros((OBSERVATION_STATE_DIM,), dtype=bool)
    arm_states: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        qpos = np.asarray(qpos_by_side[side], dtype=float).reshape(6)
        xyzquat = pose_matrix_to_xyzquat(pose_by_side[side])
        state[STATE_ARM_QPOS_SLICES[side]] = qpos
        state[STATE_ARM_POSE_SLICES[side]] = xyzquat
        valid[STATE_ARM_QPOS_SLICES[side]] = True
        valid[STATE_ARM_POSE_SLICES[side]] = True
        arm_states[side] = {
            "hardware_qpos": qpos.copy(),
            "measured_qpos": qpos.copy(),
            "hardware_read_success": True,
            "hardware_measurement_valid": True,
            "hardware_ready": True,
            "hardware_read_sequence": max(1, int(step_index) + 1),
            "measured_sequence": max(1, int(step_index) + 1),
            "hardware_read_timestamp_monotonic": float(timestamp_monotonic),
            "measured_timestamp_monotonic": float(timestamp_monotonic),
            "hardware_result_code": 0,
            "hardware_arm_err": 0,
            "hardware_sys_err": 0,
            "emergency_stopped": False,
        }
    return DeploymentObservation(
        timestamp=time.time(),
        step_index=int(step_index),
        active_sides=SIDES,
        arms_enabled=True,
        hands_enabled=False,
        state=state,
        state_valid=valid,
        arm_states=arm_states,
    )


class _MockDispatch:
    def __init__(self, args: argparse.Namespace) -> None:
        self._lock = threading.Lock()
        self._qpos = {side: np.zeros(6, dtype=float) for side in SIDES}
        self.records: list[dict[str, Any]] = []
        self.hold_count = 0
        self.teleport_count = 0
        self._generation = 0
        self._generation_dispatch_count = 0
        self._base_delay_sec = float(args.dispatch_delay_ms) / 1000.0
        self._spike_every = int(args.dispatch_spike_every)
        self._spike_sec = float(args.dispatch_spike_ms) / 1000.0

    def teleport(self, qpos_by_side: Mapping[str, np.ndarray]) -> None:
        with self._lock:
            self._qpos = {
                side: np.asarray(qpos_by_side[side], dtype=float).reshape(6).copy()
                for side in SIDES
            }
            self.teleport_count += 1
            self._generation += 1
            self._generation_dispatch_count = 0

    def qpos(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {side: value.copy() for side, value in self._qpos.items()}

    def hold(self) -> None:
        with self._lock:
            self.hold_count += 1

    def __call__(self, command) -> dict[str, Any]:
        started = time.monotonic()
        with self._lock:
            dispatch_index = len(self.records) + 1
            generation = self._generation
            generation_dispatch_index = self._generation_dispatch_count
            self._generation_dispatch_count += 1
        delay = self._base_delay_sec
        if self._spike_every > 0 and dispatch_index % self._spike_every == 0:
            delay += self._spike_sec
        if delay > 0.0:
            time.sleep(delay)

        requested: dict[str, np.ndarray] = {}
        for side in SIDES:
            name = f"{side}_arm"
            item = command.commands[name]
            requested[name] = np.asarray(item.value, dtype=float).reshape(6).copy()
        ended = time.monotonic()
        metadata = dict(command.metadata)
        collision_debug = dict(metadata.get("position_collision_debug") or {})
        planner_metadata = dict(collision_debug.get("planner_metadata") or {})
        waypoint_shadow = dict(
            planner_metadata.get("continuous_waypoint_shadow") or {}
        )
        target_poses = dict(
            collision_debug.get("target_endpoint_pose_world_by_side") or {}
        )
        realized_poses = dict(
            collision_debug.get("realized_endpoint_pose_world_by_side") or {}
        )
        pose_errors = {
            side: _pose_error(target_poses.get(side), realized_poses.get(side))
            for side in SIDES
        }
        record = {
            "dispatch_index": dispatch_index,
            "generation": generation,
            "generation_dispatch_index": generation_dispatch_index,
            "started_monotonic": started,
            "duration_ms": (ended - started) * 1000.0,
            "deadline_lateness_ms": (
                started
                - float(metadata.get("position_reference_scheduled_monotonic_time", started))
            )
            * 1000.0,
            "frame_total_ms": metadata.get("position_collision_frame_total_ms"),
            "time_budget_exceeded": bool(
                metadata.get("position_collision_time_budget_exceeded", False)
            ),
            "mode": planner_metadata.get("governor_execution_mode"),
            "precompute_submitted": planner_metadata.get(
                "continuous_waypoint_precompute_submitted_count"
            ),
            "precompute_completed": planner_metadata.get(
                "continuous_waypoint_precompute_completed_count"
            ),
            "precompute_misses": planner_metadata.get(
                "continuous_waypoint_precompute_miss_count"
            ),
            "precompute_superseded": planner_metadata.get(
                "continuous_waypoint_precompute_superseded_count"
            ),
            "precompute_status": waypoint_shadow.get("status"),
            "precompute_miss_reason": waypoint_shadow.get(
                "precompute_miss_reason"
            ),
            "formal_reference_translation_error_m": max(
                (
                    value[0]
                    for value in pose_errors.values()
                    if value[0] is not None
                ),
                default=None,
            ),
            "formal_reference_rotation_error_deg": max(
                (
                    value[1]
                    for value in pose_errors.values()
                    if value[1] is not None
                ),
                default=None,
            ),
            "q_left": requested["left_arm"],
            "q_right": requested["right_arm"],
        }
        with self._lock:
            self._qpos = {
                "left": requested["left_arm"].copy(),
                "right": requested["right_arm"].copy(),
            }
            self.records.append(record)
        return {
            "accepted": True,
            "effectors": {
                name: {
                    "accepted": True,
                    "backend_ready": True,
                    "command_type": "joint_vector",
                    "requested_value": value.copy(),
                    "final_value": value.copy(),
                    "rewritten": False,
                    "singularity_gate_status": "pass",
                    "state_error": None,
                    "background_streaming": {},
                }
                for name, value in requested.items()
            },
        }


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


def _pose_error(
    target: Any,
    realized: Any,
) -> tuple[float | None, float | None]:
    try:
        target_pose = np.asarray(target, dtype=float).reshape(4, 4)
        realized_pose = np.asarray(realized, dtype=float).reshape(4, 4)
    except Exception:
        return None, None
    if not np.all(np.isfinite(target_pose)) or not np.all(np.isfinite(realized_pose)):
        return None, None
    translation_m = float(
        np.linalg.norm(target_pose[:3, 3] - realized_pose[:3, 3])
    )
    relative_rotation = target_pose[:3, :3].T @ realized_pose[:3, :3]
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
    rotation_deg = float(np.rad2deg(math.acos(cosine)))
    return translation_m, rotation_deg


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _compact_executor_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    scalar_fields = (
        "started",
        "active",
        "closing",
        "closed",
        "queue_depth",
        "submitted_count",
        "reference_count",
        "arm_dispatch_attempt_count",
        "arm_dispatch_failed_count",
        "dispatch_count",
        "queue_full_count",
        "clear_count",
        "hold_count",
        "hold_failure_count",
        "missed_reference_ticks",
        "fault_latched",
        "fault_stage",
        "fault_type",
        "fault",
        "collision_reset_count",
        "collision_tracking_count",
        "collision_limited_count",
        "collision_backoff_count",
        "collision_hold_count",
        "reference_worker_alive",
    )
    compact = {name: snapshot.get(name) for name in scalar_fields}
    last_collision = dict(snapshot.get("last_collision") or {})
    compact["last_collision"] = {
        name: last_collision.get(name)
        for name in (
            "position_collision_status",
            "position_collision_hold",
            "position_collision_distance_m",
            "position_collision_frame_total_ms",
            "position_collision_time_budget_exceeded",
            "position_collision_certification_status",
            "temporal_smoothing",
        )
    }
    last_reset = dict(snapshot.get("last_collision_reset") or {})
    compact["last_collision_reset"] = {
        name: last_reset.get(name)
        for name in (
            "authoritative",
            "source",
            "pair_skew_sec",
            "measurement_max_age_sec",
        )
    }
    return compact


def _track_poses(
    oracle: RM65PositionCollisionPlanner,
    track: Track,
) -> list[dict[str, np.ndarray]]:
    return [
        _observe_endpoint_poses_arm_base(
            oracle,
            {"left": track.q_left[index], "right": track.q_right[index]},
        )
        for index in range(track.source_frames)
    ]


def _observe_endpoint_poses_arm_base(
    oracle: RM65PositionCollisionPlanner,
    qpos_by_side: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return the same arm-base target frame consumed by official all-IK."""

    world = oracle.observe_endpoint_poses_world(qpos_by_side)
    return {
        side: (
            np.linalg.inv(np.asarray(oracle.world_from_arm_base_by_side[side], dtype=float))
            @ np.asarray(world[side], dtype=float)
        )
        for side in SIDES
    }


def _speculative_replacement_pose(
    oracle: RM65PositionCollisionPlanner,
    track: Track,
    index: int,
    *,
    perturbation_rad: float,
) -> dict[str, np.ndarray]:
    q_left = track.q_left[index].copy()
    q_right = track.q_right[index].copy()
    # J6 is chosen only to make the speculative source hash differ.  The
    # perturbed target is never committed and remains inside hard limits.
    lower = np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float)
    upper = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float)
    q_right[5] = np.clip(q_right[5] + perturbation_rad, lower[5], upper[5])
    return _observe_endpoint_poses_arm_base(
        oracle,
        {"left": q_left, "right": q_right},
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    paths = [Path(value).resolve() for value in args.parquet]
    for pattern in args.glob:
        paths.extend(Path(value).resolve() for value in sorted(glob.glob(pattern)))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise ValueError("provide at least one --parquet or --glob input")
    tracks = _load_tracks(
        paths,
        min_source_frames=int(args.min_source_frames),
        max_source_frames=args.max_source_frames,
        joint_limit_tolerance_rad=float(args.joint_limit_tolerance_rad),
    )

    oracle = RM65PositionCollisionPlanner(
        _planner_config(args),
        endpoint_to_tool_by_side={side: np.eye(4, dtype=float) for side in SIDES},
    )
    dispatch = _MockDispatch(args)
    created_governors: list[_DelayedAsyncGovernor] = []
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=SOURCE_HZ,
            reference_hz=COMMAND_HZ,
            command_hz=COMMAND_HZ,
            stale_after_sec=0.25,
            queue_size=32,
            debug_events_enabled=True,
            debug_event_queue_size=128,
            shutdown_timeout_sec=2.0,
            collision_required=True,
            chunk_lookahead_enabled=True,
        ),
        canonical_pipeline=_pipeline(),
        dispatch_callback=dispatch,
        hold_callback=dispatch.hold,
        collision_planner_factory=_make_governor_factory(args, created_governors),
    )
    executor.prepare_collision_planner()
    executor.start()

    started = time.monotonic()
    deadline = (
        math.inf
        if args.duration_sec is None
        else started + float(args.duration_sec)
    )
    source_period = 1.0 / SOURCE_HZ
    source_submit_count = 0
    replacement_count = 0
    clear_injection_count = 0
    completed_tracks = 0
    interrupted = False
    run_error: str | None = None
    try:
        repeat_index = 0
        while repeat_index < int(args.repeat) and time.monotonic() < deadline:
            for track in tracks:
                if time.monotonic() >= deadline:
                    break
                q0 = {"left": track.q_left[0], "right": track.q_right[0]}
                dispatch.teleport(q0)
                if executor.snapshot()["active"]:
                    executor.clear()
                poses = _track_poses(oracle, track)
                next_submit_at = time.monotonic()
                for index in range(track.source_frames - 3):
                    if time.monotonic() >= deadline:
                        break
                    if (
                        int(args.clear_every_source_frames) > 0
                        and source_submit_count > 0
                        and source_submit_count
                        % int(args.clear_every_source_frames)
                        == 0
                    ):
                        executor.clear()
                        clear_injection_count += 1
                    now = time.monotonic()
                    if now < next_submit_at:
                        time.sleep(next_submit_at - now)
                    submit_time = time.monotonic()
                    current_qpos = dispatch.qpos()
                    current_pose = _observe_endpoint_poses_arm_base(
                        oracle,
                        current_qpos,
                    )
                    observation = _observation(
                        current_qpos,
                        current_pose,
                        timestamp_monotonic=submit_time,
                        step_index=source_submit_count,
                    )
                    second_pose = poses[index + 2]
                    if (
                        int(args.replace_every_source_frames) > 0
                        and source_submit_count > 0
                        and source_submit_count
                        % int(args.replace_every_source_frames)
                        == 0
                    ):
                        second_pose = _speculative_replacement_pose(
                            oracle,
                            track,
                            index + 2,
                            perturbation_rad=float(args.replacement_perturbation_rad),
                        )
                        replacement_count += 1
                    executor.submit(
                        _filtered_pose(poses[index]),
                        observation,
                        "deadline_soak",
                        timestamp_monotonic=submit_time,
                        lookahead_filtered=_filtered_pose(poses[index + 1]),
                        lookahead_observation=observation,
                        second_lookahead_filtered=_filtered_pose(second_pose),
                        second_lookahead_observation=observation,
                        third_lookahead_filtered=_filtered_pose(poses[index + 3]),
                        third_lookahead_observation=observation,
                    )
                    source_submit_count += 1
                    next_submit_at += source_period
                # Drain the last source interval, then invalidate before the
                # stale watchdog can turn an intentional dataset boundary into
                # a false failure.
                time.sleep(min(0.10, max(0.0, deadline - time.monotonic())))
                if executor.snapshot()["active"]:
                    executor.clear()
                completed_tracks += 1
            repeat_index += 1
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:  # noqa: BLE001 - result must preserve soak fault.
        run_error = f"{type(exc).__name__}: {exc}"
    finally:
        before_close = executor.snapshot()
        close_started = time.monotonic()
        close_error: str | None = None
        try:
            executor.close()
        except Exception as exc:  # noqa: BLE001 - close is part of the verdict.
            close_error = f"{type(exc).__name__}: {exc}"
        close_ms = (time.monotonic() - close_started) * 1000.0
        after_close = executor.snapshot()

    records = list(dispatch.records)
    steady_records = [
        record
        for record in records
        if int(record["generation_dispatch_index"])
        >= int(args.warmup_command_frames)
    ]
    modes = Counter(str(record["mode"]) for record in records)
    precompute_statuses = Counter(
        str(record["precompute_status"])
        for record in records
        if record.get("precompute_status") is not None
    )
    precompute_miss_reasons = Counter(
        str(record["precompute_miss_reason"])
        for record in records
        if record.get("precompute_miss_reason") is not None
    )
    frame_ms = [record["frame_total_ms"] for record in records]
    c2_mode_names = {
        "continuous_c2_waypoint",
        "experimental_continuous_c2_waypoint",
    }
    selected = int(sum(modes.get(name, 0) for name in c2_mode_names))
    c2_records = [
        record
        for record in records
        if record.get("mode") in c2_mode_names
    ]
    last_record = records[-1] if records else {}
    elapsed_sec = time.monotonic() - started
    summary = {
        "schema": "dexteleop.deploy_continuous_waypoint_deadline_soak/v1",
        "no_hardware": True,
        "interrupted": interrupted,
        "run_error": run_error,
        "close_error": close_error,
        "elapsed_sec": float(elapsed_sec),
        "input": {
            "paths": [str(track.path) for track in tracks],
            "tracks": len(tracks),
            "completed_tracks": completed_tracks,
            "source_frames_available": int(sum(track.source_frames for track in tracks)),
            "source_submissions": source_submit_count,
            "roundoff_clipped_values": int(sum(track.clipped_values for track in tracks)),
        },
        "injections": {
            "speculative_replacements": replacement_count,
            "clear_reset": clear_injection_count,
            "dispatch_delay_ms": float(args.dispatch_delay_ms),
            "dispatch_spike_every": int(args.dispatch_spike_every),
            "dispatch_spike_ms": float(args.dispatch_spike_ms),
            "precompute_delay_ms": float(args.precompute_delay_ms),
            "precompute_spike_every": int(args.precompute_spike_every),
            "precompute_spike_ms": float(args.precompute_spike_ms),
        },
        "executor": {
            "before_close": _compact_executor_snapshot(before_close),
            "after_close": _compact_executor_snapshot(after_close),
            "close_ms": float(close_ms),
        },
        "dispatch": {
            "count": len(records),
            "rate_hz": (None if elapsed_sec <= 0.0 else len(records) / elapsed_sec),
            "deadline_lateness_ms": _percentiles(
                record["deadline_lateness_ms"] for record in records
            ),
            "callback_duration_ms": _percentiles(
                record["duration_ms"] for record in records
            ),
            "planner_frame_total_ms": _percentiles(frame_ms),
            "time_budget_exceeded_count": int(
                sum(bool(record["time_budget_exceeded"]) for record in records)
            ),
            "warmup_command_frames_per_track": int(args.warmup_command_frames),
            "steady_state_count": len(steady_records),
            "steady_state_deadline_lateness_ms": _percentiles(
                record["deadline_lateness_ms"] for record in steady_records
            ),
            "steady_state_planner_frame_total_ms": _percentiles(
                record["frame_total_ms"] for record in steady_records
            ),
            "steady_state_late_over_period_count": int(
                sum(
                    float(record["deadline_lateness_ms"])
                    > (1000.0 / COMMAND_HZ)
                    for record in steady_records
                )
            ),
            "steady_state_time_budget_exceeded_count": int(
                sum(bool(record["time_budget_exceeded"]) for record in steady_records)
            ),
            "mode_counts": dict(modes),
            "precompute_status_counts": dict(precompute_statuses),
            "precompute_miss_reason_counts": dict(precompute_miss_reasons),
            "continuous_c2_selected_fraction": (
                None if not records else selected / len(records)
            ),
            "continuous_c2_vs_formal_reference": {
                "translation_error_mm": _percentiles(
                    float(record["formal_reference_translation_error_m"]) * 1000.0
                    for record in c2_records
                    if record.get("formal_reference_translation_error_m") is not None
                ),
                "rotation_error_deg": _percentiles(
                    record["formal_reference_rotation_error_deg"]
                    for record in c2_records
                    if record.get("formal_reference_rotation_error_deg") is not None
                ),
            },
            "last_precompute_counters": {
                "submitted": last_record.get("precompute_submitted"),
                "completed": last_record.get("precompute_completed"),
                "misses": last_record.get("precompute_misses"),
                "superseded": last_record.get("precompute_superseded"),
            },
        },
        "mock_boundary": {
            "hold_count": dispatch.hold_count,
            "teleport_count": dispatch.teleport_count,
        },
        "probe_diagnostics": {
            "anchor_mismatch_count": int(
                sum(len(item.probe_anchor_mismatches) for item in created_governors)
            ),
            "anchor_mismatch_examples": [
                value
                for item in created_governors
                for value in item.probe_anchor_mismatches[:3]
            ][:6],
        },
    }
    summary["verdict"] = {
        "passed": bool(
            run_error is None
            and close_error is None
            and not bool(before_close["fault_latched"])
            and int(before_close["arm_dispatch_failed_count"]) == 0
            and int(before_close["queue_full_count"]) == 0
            and not bool(after_close["reference_worker_alive"])
        ),
        "deadline_clean": bool(
            int(before_close["missed_reference_ticks"]) == 0
            and not any(bool(record["time_budget_exceeded"]) for record in records)
        ),
        "steady_state_deadline_clean": bool(
            steady_records
            and not any(
                float(record["deadline_lateness_ms"])
                > (1000.0 / COMMAND_HZ)
                for record in steady_records
            )
            and not any(
                bool(record["time_budget_exceeded"])
                for record in steady_records
            )
        ),
        "notes": [
            "passed checks lifecycle/dispatch integrity; deadline_clean is reported separately",
            "injected dispatch spikes are expected to make deadline_clean false",
            "this is a no-hardware scheduler/planner soak, not a real actuator timing certificate",
        ],
    }
    return _json_safe(summary)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real 15->120 Hz deploy executor and official all-IK/C2 "
            "planner against recorded q_ref-derived poses with mock dispatch only."
        )
    )
    parser.add_argument("--parquet", action="append", default=[], help="Episode parquet path; repeatable.")
    parser.add_argument("--glob", action="append", default=[], help="Episode parquet glob; repeatable.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--min-source-frames", type=int, default=8)
    parser.add_argument("--max-source-frames", type=int, default=None)
    parser.add_argument("--warmup-command-frames", type=int, default=16)
    parser.add_argument("--joint-limit-tolerance-rad", type=float, default=1e-6)
    parser.add_argument("--scene-profile", default="dual_arm_hand_proxy_open")
    parser.add_argument("--collision-min-distance-m", type=float, default=0.03)
    parser.add_argument("--collision-distance-epsilon-m", type=float, default=1e-4)
    parser.add_argument("--jerk-ramp-sec", type=float, default=0.05)
    parser.add_argument("--batched-solver-tolerance", type=float, default=1e-6)
    parser.add_argument("--dispatch-delay-ms", type=float, default=0.0)
    parser.add_argument("--dispatch-spike-every", type=int, default=0)
    parser.add_argument("--dispatch-spike-ms", type=float, default=0.0)
    parser.add_argument("--precompute-delay-ms", type=float, default=0.0)
    parser.add_argument("--precompute-spike-every", type=int, default=0)
    parser.add_argument("--precompute-spike-ms", type=float, default=0.0)
    parser.add_argument("--replace-every-source-frames", type=int, default=0)
    parser.add_argument("--replacement-perturbation-rad", type=float, default=0.002)
    parser.add_argument("--clear-every-source-frames", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)
    for name in (
        "repeat",
        "min_source_frames",
        "warmup_command_frames",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "dispatch_spike_every",
        "precompute_spike_every",
        "replace_every_source_frames",
        "clear_every_source_frames",
    ):
        if int(getattr(args, name)) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in (
        "dispatch_delay_ms",
        "dispatch_spike_ms",
        "precompute_delay_ms",
        "precompute_spike_ms",
    ):
        if float(getattr(args, name)) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.duration_sec is not None and float(args.duration_sec) <= 0.0:
        parser.error("--duration-sec must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    print(
        "[DeadlineSoak] NO HARDWARE: real executor/all-IK/collision; "
        "mock arm dispatch and hold only.",
        flush=True,
    )
    summary = _run(args)
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    return 0 if bool(summary["verdict"]["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
