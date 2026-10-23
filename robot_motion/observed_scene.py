"""Build observed obstacle geometry without a device, model or physics owner.

Adapters supply calibrated RGB-D cuts, capture-time joint measurements and the
existing robot self filter. This module never acquires new observations and
does not infer unobserved geometry. Retained cuts may be reused; admission and
freshness for the intended operation remain the caller's responsibility.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np

from robot_motion.grasp_visible_scene import (
    PRODUCTION_VISIBLE_GRASP_DEPTH_FILTER,
    filter_isolated_background_depth,
)
from robot_motion.observed_obstacles import ObservedObstacleCloud
from robot_motion.pose import as_strict_pose_matrix


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
def timed_observation_phase(record_timing, phase, *, role=None):
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
                                 planning_qpos_by_side, max_inter_camera_skew_sec,
                                 max_qpos_drift_rad, debug_record=None, record_timing=None):
    """Project supplied observations; no native scene or acquisition is accepted.

    The profile owns joint-drift admission. Requiring its bound explicitly keeps
    the reusable geometry independent of real deployment or simulator defaults.
    """
    for name, value in (("max_qpos_drift_rad", max_qpos_drift_rad),
                        ("max_inter_camera_skew_sec", max_inter_camera_skew_sec)):
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if set(planning_qpos_by_side) != {"left", "right"} or any(
            np.asarray(q).shape != (6,) or not np.isfinite(q).all()
            for q in planning_qpos_by_side.values()):
        raise ValueError("Planning requires finite measured dual-arm qpos.")
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
                    or np.max(np.abs(q - planning_qpos_by_side[side])) > max_qpos_drift_rad):
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
        with timed_observation_phase(record_timing, "self_filter", role=capture.role):
            filtered = self_filter(depth_m=depth, target_mask=target, intrinsics=capture.intrinsics,
                source_from_camera=pose, capture_qpos_by_side=pair)
        if debug_record is not None:
            debug_record.phase("depth_preprocessing", role=capture.role)
        with timed_observation_phase(record_timing, "depth_preprocessing", role=capture.role):
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
            "capture_qpos_by_side": {side: np.asarray(pair[side]).tolist() for side in pair},
            "pose_measured_at": capture.pose_measured_at,
            "pose_evidence_refs": list(capture.pose_evidence_refs),
            "out_of_range_points": range_removed,
            "self_filter": dict(filtered.evidence), "depth_preprocessing": dict(preprocessing)})
    with timed_observation_phase(record_timing, "voxelize_and_bind_evidence"):
        return ObservedObstacleCloud(reference_frame_id=reference_frame_id,
            points_m=np.concatenate(points), voxel_size_m=config.voxel_size_m,
            evidence={"views": evidence, "policy": asdict(config),
                      "scope": "observed_surfaces_only_no_target_contact_exemptions"})
