"""Model-independent safe-move scene from already-running RGB-D sources."""

from __future__ import annotations

import logging
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from PIL import Image
import time
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass

import numpy as np

from robot_motion.grasp_visible_scene import (
    PRODUCTION_VISIBLE_GRASP_DEPTH_FILTER,
    filter_isolated_background_depth,
)
from robot_motion.observed_obstacles import ObservedObstacleCloud
from robot_motion.pose import as_strict_pose_matrix
from teleop_runtime.skill_deploy.motion_port import (
    REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD,
)
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspTrajectoryPlanningError,
)

from .motion_observed_scene_debug import SafeMotionDebugRecord, validate_debug_directory
from .rm65_camera_geometry import RM65StationaryGeometryUnavailable


@dataclass(frozen=True)
class SafeMotionObservationConfig:
    roles: tuple[str, ...] = ("main", "wrist_left", "wrist_right")
    maximum_depth_m: float = 1.5
    maximum_scene_age_sec: float = 30.0
    voxel_size_m: float = 0.004

    def __post_init__(self):
        roles = tuple(self.roles)
        if (not roles or len(set(roles)) != len(roles)
                or any(role not in {"main", "wrist_left", "wrist_right"} for role in roles)):
            raise ValueError("Safe motion requires explicit unique RGB-D observation roles.")
        for name in ("maximum_depth_m", "maximum_scene_age_sec", "voxel_size_m"):
            value = float(getattr(self, name))
            if isinstance(getattr(self, name), bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "roles", roles)


@contextmanager
def _timed_phase(record_timing, phase, *, role=None):
    if record_timing is None:
        yield
        return
    started = time.perf_counter()
    primary_error = None
    try:
        yield
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            record_timing({"phase": phase, "role": role,
                           "wall_sec": time.perf_counter() - started})
        except BaseException as recording_error:
            if isinstance(primary_error, (KeyboardInterrupt, SystemExit)):
                raise primary_error from recording_error
            raise

def obstacle_cloud_from_captures(*, captures, config, reference_frame_id, self_filter,
                                 planning_qpos_by_side, max_inter_camera_skew_sec, debug_record=None,
                                 record_timing=None):
    """Same pure geometry path for live captures and archived read-only replay."""
    if tuple(capture.role for capture in captures) != config.roles:
        raise ValueError("Required RGB-D views are incomplete or reordered.")
    if (len({capture.station_revision for capture in captures}) != 1
            or len({capture.source_binding_hash for capture in captures}) != 1):
        raise ValueError("RGB-D views belong to different Station revisions or source bindings.")
    acquired = np.asarray([capture.acquired_at for capture in captures], dtype=float)
    if not np.isfinite(acquired).all() or np.ptp(acquired) > max_inter_camera_skew_sec:
        raise ValueError("RGB-D inter-camera acquisition skew exceeded.")
    points, evidence = [], []
    for capture in captures:
        pair = capture.capture_qpos_by_side
        if pair is None or set(pair) != {"left", "right"}:
            raise ValueError("Captured obstacles require capture-time dual-arm qpos.")
        for side in ("left", "right"):
            q = np.asarray(pair[side], dtype=float)
            if (q.shape != (6,) or not np.isfinite(q).all()
                    or np.max(np.abs(q - planning_qpos_by_side[side])) > REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD):
                raise ValueError("Capture-time qpos differs from safe-move start.")
        depth = np.asarray(capture.depth_m, dtype=float).copy()
        if (depth.shape != (capture.height, capture.width)
                or not np.isfinite(depth).all() or np.any(depth < 0)):
            raise ValueError("Aligned depth must contain finite nonnegative meters.")
        pose = as_strict_pose_matrix(np.asarray(capture.source_from_camera).reshape(4, 4))
        if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-8)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-8)):
            raise ValueError("Camera geometry must be a rigid world transform.")
        intrinsics = capture.intrinsics
        values = [float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy")]
        if not np.isfinite(values).all() or min(values[:2]) <= 0:
            raise ValueError("Camera intrinsics must be finite with positive focal lengths.")
        range_removed = int(np.count_nonzero(depth > config.maximum_depth_m))
        depth[depth > config.maximum_depth_m] = 0
        target = np.zeros(depth.shape, dtype=bool)
        if debug_record is not None:
            debug_record.phase("self_filter", role=capture.role)
        with _timed_phase(record_timing, "self_filter", role=capture.role):
            filtered = self_filter(depth_m=depth, target_mask=target, intrinsics=capture.intrinsics,
                source_from_camera=pose, capture_qpos_by_side=pair)
        if debug_record is not None:
            debug_record.phase("depth_preprocessing", role=capture.role)
        with _timed_phase(record_timing, "depth_preprocessing", role=capture.role):
            valid, preprocessing = filter_isolated_background_depth(
                depth_m=filtered.depth_m, target_mask=target, config=PRODUCTION_VISIBLE_GRASP_DEPTH_FILTER)
        rows, columns = np.nonzero(valid)
        if not len(rows):
            raise ValueError(f"Required view {capture.role} has no usable obstacle depth.")
        z = filtered.depth_m[rows, columns]
        fx, fy, cx, cy = values
        camera_points = np.column_stack(((columns-cx)*z/fx, (rows-cy)*z/fy, z))
        world_points = camera_points @ pose[:3, :3].T + pose[:3, 3]
        points.append(world_points)
        if debug_record is not None:
            debug_record.filtered(capture, filtered, valid, world_points, preprocessing)
        evidence.append({"role": capture.role, "source_id": capture.source_id,
            "source_generation": capture.source_generation, "frame_index": capture.frame_index,
            "acquired_at": capture.acquired_at, "station_revision": capture.station_revision,
            "source_binding_hash": capture.source_binding_hash,
            "depth_content_hash": capture.depth_content_hash,
            "T_world_from_camera": pose.tolist(), "intrinsics": dict(intrinsics),
            "capture_qpos_by_side": {side: pair[side].tolist() for side in pair},
            "pose_measured_at": capture.pose_measured_at,
            "pose_evidence_refs": list(capture.pose_evidence_refs),
            "out_of_range_points": range_removed,
            "self_filter": dict(filtered.evidence), "depth_preprocessing": dict(preprocessing)})
    with _timed_phase(record_timing, "voxelize_and_bind_evidence"):
        return ObservedObstacleCloud(reference_frame_id=reference_frame_id,
            points_m=np.concatenate(points), voxel_size_m=config.voxel_size_m,
            evidence={"views": evidence, "policy": asdict(config),
                      "scope": "observed_surfaces_only_no_target_contact_exemptions"})


class ObservedSafeMovePlanner:
    """Freeze one scene and pass it through the existing lower planner."""

    def __init__(self, *, provider, current_station_revision, reference_frame_id,
                 self_filter, config, planner_factory, clock, non_motion_read_window=None,
                 debug_record_dir=None, debug_context=None, debug_context_files=None,
                 observation_record_dir=None):
        self.provider = provider
        self.current_station_revision = current_station_revision
        self.reference_frame_id = reference_frame_id
        self.self_filter = self_filter
        self.config = config
        self.planner_factory = planner_factory
        self.clock = clock
        self.non_motion_read_window = non_motion_read_window or nullcontext
        self.debug_record_dir = None if debug_record_dir is None else validate_debug_directory(debug_record_dir)
        self.debug_context = debug_context
        self.debug_context_files = debug_context_files
        self.observation_record_dir = observation_record_dir

    def __call__(self, **kwargs):
        if kwargs.get("require_grasp_approach") is not False:
            raise ValueError("Observed safe-move planner is not a grasp contact planner.")
        started = time.monotonic()
        wall_started = time.perf_counter()
        timings = []
        capturing = True
        captures = []
        record = None
        observation_record = None
        inputs_attempted = False
        primary_error = None
        budget = kwargs.get("time_budget_sec")
        if budget is not None and (
                not np.isfinite(budget) or budget <= 0):
            raise RM65GraspTrajectoryPlanningError("path_planning_time_budget_exhausted")

        def remaining_budget(stage):
            if budget is None:
                return None
            remaining = float(budget) - (time.monotonic() - started)
            if not np.isfinite(remaining) or remaining <= 0:
                raise RM65GraspTrajectoryPlanningError(
                    "path_planning_time_budget_exhausted",
                    details={"stage": stage, "time_budget_sec": budget})
            return remaining

        def record_timing(entry):
            timings.append(entry)
            if record is not None and not capturing:
                record.timing(entry)

        def remember_primary(error):
            nonlocal primary_error
            if primary_error is None or (
                    isinstance(error, (KeyboardInterrupt, SystemExit))
                    and not isinstance(primary_error, (KeyboardInterrupt, SystemExit))):
                primary_error = error

        revision = self.current_station_revision()
        try:
            with ExitStack() as read_scope:
                try:
                    if self.observation_record_dir is not None:
                        parent = Path(self.observation_record_dir)
                        parent.mkdir(parents=True, exist_ok=True)
                        observation_record = Path(tempfile.mkdtemp(prefix="attempt-", dir=parent))
                        logging.getLogger(__name__).info("Safe-move RGB-D record: %s", observation_record)
                    if self.debug_record_dir is not None:
                        record = SafeMotionDebugRecord(self.debug_record_dir,
                            reference_frame_id=self.reference_frame_id, config=self.config, request=kwargs,
                            context={**dict(self.debug_context or {}), "station_revision": revision,
                                     "max_inter_camera_skew_ms": self.provider.max_inter_camera_skew_ms},
                            context_files=self.debug_context_files)
                    try:
                        try:
                            # Freeze correlated views without intervening disk I/O.
                            for role in self.config.roles:
                                try:
                                    with _timed_phase(record_timing, "capture_rgbd", role=role):
                                        captures.append(self.provider.capture_aligned_rgbd(role=role,
                                            expected_station_revision=revision, require_stationary=True))
                                except (RuntimeError, ValueError) as error:
                                    raise self._observation_error(error, stage="capture_rgbd", role=role,
                                                                  captures=captures) from error
                        except BaseException as error:
                            remember_primary(error)
                            raise
                        finally:
                            # Resolve frame-near qpos before pausing its sole producer.
                            # Retain partial captures even if entering the window fails.
                            try:
                                try:
                                    read_scope.enter_context(self.non_motion_read_window())
                                except BaseException as error:
                                    if isinstance(error, (RuntimeError, ValueError)) and not isinstance(
                                            error, RM65GraspTrajectoryPlanningError):
                                        observation_error = self._observation_error(
                                            error, stage="observation_window", role=None, captures=captures)
                                        remember_primary(observation_error)
                                        raise observation_error from error
                                    remember_primary(error)
                                    raise
                            finally:
                                capturing = False
                                if observation_record is not None:
                                    inputs_attempted = True
                                    self._save_inputs(observation_record, captures, kwargs)
                                if record is not None:
                                    for entry in timings:
                                        record.timing(entry)
                                    for capture in captures:
                                        with _timed_phase(record_timing, "save_input", role=capture.role):
                                            record.capture(capture)
                        self._require_current(captures, revision)
                        try:
                            cloud = obstacle_cloud_from_captures(captures=captures, config=self.config,
                                reference_frame_id=self.reference_frame_id, self_filter=self.self_filter,
                                planning_qpos_by_side=kwargs["qpos_by_side"],
                                max_inter_camera_skew_sec=self.provider.max_inter_camera_skew_ms / 1000,
                                debug_record=record, record_timing=record_timing)
                        except RM65GraspTrajectoryPlanningError:
                            raise
                        except (RuntimeError, ValueError) as error:
                            raise self._observation_error(error, stage="obstacle_cloud", role=None,
                                                          captures=captures) from error
                    except RM65GraspTrajectoryPlanningError:
                        raise
                    except (RuntimeError, ValueError) as error:
                        raise self._observation_error(error, stage="observation_window", role=None,
                                                      captures=captures) from error
                    if observation_record is not None:
                        np.savez(observation_record / "obstacles.npz", occupied_centers_m=cloud.points_m)
                        _save_json(observation_record / "obstacles.json", cloud.evidence_payload())
                    if record is not None:
                        with _timed_phase(record_timing, "save_obstacles"):
                            record.scene(cloud)
                    self._require_current(captures, revision)
                    remaining_budget("observed_scene_preparation")
                    with _timed_phase(record_timing, "planner_prepare"):
                        planner = self.planner_factory(cloud)
                    if record is not None:
                        record.planner(planner, kwargs["qpos_by_side"])
                    planning_kwargs = dict(kwargs)
                    remaining = remaining_budget("observed_scene_preparation")
                    if remaining is not None:
                        planning_kwargs["time_budget_sec"] = remaining
                    with _timed_phase(record_timing, "path_planning"):
                        trajectory = planner(**planning_kwargs)
                    self._require_current(captures, revision)
                    remaining_budget("observed_scene_planning")
                    if record is not None:
                        record.finish(trajectory=trajectory)
                    if observation_record is not None:
                        _save_json(observation_record / "result.json",
                                   {"status": "path_found", "execution_verified": False})
                except BaseException as error:
                    remember_primary(error)
                    raise
            if primary_error is not None:
                raise primary_error
            # Window exit may resume sampling and consume budget or invalidate the scene.
            self._require_current(captures, revision)
            remaining_budget("observed_scene_recording")
            return trajectory
        except BaseException as caught_error:  # Retain evidence, then preserve failure/interrupt.
            remember_primary(caught_error)
            error = primary_error
            try:
                if observation_record is not None:
                    if not inputs_attempted:
                        inputs_attempted = True
                        self._save_inputs(observation_record, captures, kwargs)
                    _save_json(observation_record / "result.json", {"status": "failed",
                        "exception_type": type(error).__name__, "message": str(error),
                        "reason_code": getattr(error, "reason_code", None),
                        "details": getattr(error, "details", {})})
                if record is not None:
                    if isinstance(error, RM65GraspTrajectoryPlanningError):
                        error.details["debug_record_dir"] = str(record.directory)
                    if not (isinstance(error, RM65GraspTrajectoryPlanningError)
                            and error.reason_code == "safe_motion_debug_record_failed"):
                        record.finish(error=error)
            except BaseException as recording_error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise error from recording_error
                if isinstance(recording_error, RM65GraspTrajectoryPlanningError):
                    recording_error.details["original_error"] = str(error)
                raise recording_error from error
            if error is not caught_error:
                raise error from caught_error
            raise
        finally:
            logging.getLogger(__name__).info("Safe-move preparation %.3fs; phases=%s",
                time.perf_counter() - wall_started, timings)
    def _save_inputs(self, record, captures, request):
        views = []
        for cut in captures:
            np.savez(record / f"{cut.role}.npz", rgb=cut.rgb, depth_m=cut.depth_m)
            Image.fromarray(cut.rgb).save(record / f"{cut.role}.jpg")
            views.append({"role": cut.role, "source_id": cut.source_id,
                "file": f"{cut.role}.npz", "acquired_at": cut.acquired_at,
                "station_revision": cut.station_revision, "source_generation": cut.source_generation,
                "frame_index": cut.frame_index, "source_binding_hash": cut.source_binding_hash,
                "rgb_content_hash": cut.rgb_content_hash, "depth_content_hash": cut.depth_content_hash,
                "intrinsics": dict(cut.intrinsics), "T_world_from_camera": cut.source_from_camera,
                "capture_qpos_by_side": cut.capture_qpos_by_side,
                "pose_measured_at": cut.pose_measured_at, "pose_available_at": cut.pose_available_at,
                "pose_evidence_refs": cut.pose_evidence_refs})
        _save_json(record / "snapshot.json", {"schema": "agenticrobot.safe_move_inputs.v1",
            "reference_frame_id": self.reference_frame_id, "configuration": asdict(self.config),
            "recorded_at": self.clock(),
            "request": {key: request[key] for key in ("qpos_by_side", "side", "target_tool_tcp_world",
                "speed_scale") if key in request}, "views": views})


    def _observation_error(self, error, *, stage, role, captures):
        return RM65GraspTrajectoryPlanningError("safe_motion_observation_unavailable",
            details={"stage": stage, "role": role, "exception_type": type(error).__name__,
                "message": str(error), "required_roles": self.config.roles,
                **({"observation_reason_code": error.reason_code,
                    "observation_diagnostic": error.diagnostic}
                   if isinstance(error, RM65StationaryGeometryUnavailable) else {}),
                "capture_frames": [{"role": cut.role, "source_id": cut.source_id,
                    "frame_index": cut.frame_index, "acquired_at": cut.acquired_at,
                    "pose_measured_at": cut.pose_measured_at,
                    "valid_depth_pixels": int(np.count_nonzero(cut.depth_m > 0)),
                    "in_range_depth_pixels": int(np.count_nonzero(
                        (cut.depth_m > 0) & (cut.depth_m <= self.config.maximum_depth_m)))}
                    for cut in captures]})


    def _require_current(self, captures, revision):
        ages = [float(self.clock()) - capture.acquired_at for capture in captures]
        if (self.current_station_revision() != revision or any(
                not np.isfinite(age) or age < 0 or age > self.config.maximum_scene_age_sec for age in ages)):
            raise RM65GraspTrajectoryPlanningError("safe_motion_observation_expired",
                details={"scene_age_sec": ages, "maximum_scene_age_sec": self.config.maximum_scene_age_sec})


def _save_json(path, value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Mapping):
            return dict(item)
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, default=encode, ensure_ascii=False, indent=2, allow_nan=False))

