"""Model-independent safe-move scene from already-running RGB-D sources."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from PIL import Image
import time
from contextlib import ExitStack, nullcontext
from dataclasses import asdict

import numpy as np

from robot_motion.observed_scene import (
    SafeMotionObservationConfig as SafeMotionObservationConfig,
    obstacle_cloud_from_captures as _obstacle_cloud_from_captures,
    timed_observation_phase as _timed_phase,
)
from teleop_runtime.skill_deploy.motion_port import (
    REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD,
)
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspTrajectoryPlanningError,
)

from .motion_observed_scene_debug import (
    SafeMotionDebugRecord,
    encode_record_json,
    validate_debug_directory,
)
from .rm65_camera_geometry import RM65StationaryGeometryUnavailable


def obstacle_cloud_from_captures(*, captures, config, reference_frame_id, self_filter,
                                 planning_qpos_by_side, max_inter_camera_skew_sec, debug_record=None,
                                 record_timing=None):
    """Compatibility adapter retaining the original real deployment drift bound."""
    return _obstacle_cloud_from_captures(
        captures=captures, config=config, reference_frame_id=reference_frame_id,
        self_filter=self_filter, planning_qpos_by_side=planning_qpos_by_side,
        max_inter_camera_skew_sec=max_inter_camera_skew_sec,
        max_qpos_drift_rad=REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD,
        debug_record=debug_record, record_timing=record_timing)


class MotionPlanningGeometryResources:
    """Prepare shared static geometry before any camera or arm is started."""

    def __init__(self, *, self_filter, planner_factory=None):
        self.self_filter = self_filter
        self.planner_factory = planner_factory

    def start(self):
        started = time.perf_counter()
        phases = self.self_filter.prepare()
        if self.planner_factory is not None:
            before = time.perf_counter()
            self.planner_factory()
            phases['planner_prepare'] = time.perf_counter() - before
        logging.getLogger(__name__).warning(
            'Motion geometry ready %.3fs; phases=%s; no observation or motion cached',
            time.perf_counter() - started, phases)

    def close(self):
        # No device, worker, file or request state is owned by this resource.
        # The in-memory references belong to the enclosing session.
        pass


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
                            self._capture_views(captures, revision, record_timing)
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
                        cloud = self._build_obstacle_cloud(
                            captures, kwargs["qpos_by_side"], record, record_timing)
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

    def _capture_views(self, captures, revision, record_timing):
        """Append each acquired view so partial evidence survives a later failure."""
        for role in self.config.roles:
            try:
                with _timed_phase(record_timing, "capture_rgbd", role=role):
                    captures.append(self.provider.capture_aligned_rgbd(
                        role=role, expected_station_revision=revision, require_stationary=True))
            except (RuntimeError, ValueError) as error:
                raise self._observation_error(
                    error, stage="capture_rgbd", role=role, captures=captures) from error

    def _build_obstacle_cloud(self, captures, qpos_by_side, record, record_timing):
        """Preserve lower planning refusals; classify observation processing errors."""
        try:
            return obstacle_cloud_from_captures(
                captures=captures, config=self.config,
                reference_frame_id=self.reference_frame_id, self_filter=self.self_filter,
                planning_qpos_by_side=qpos_by_side,
                max_inter_camera_skew_sec=self.provider.max_inter_camera_skew_ms / 1000,
                debug_record=record, record_timing=record_timing)
        except RM65GraspTrajectoryPlanningError:
            raise
        except (RuntimeError, ValueError) as error:
            raise self._observation_error(
                error, stage="obstacle_cloud", role=None, captures=captures) from error

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
    path.write_text(encode_record_json(value), encoding="utf-8")
