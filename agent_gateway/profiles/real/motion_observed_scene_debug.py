"""Opt-in local evidence for the exact frozen safe-move scene, not a planner."""

from __future__ import annotations

import json
import logging
import os
import stat
import tempfile
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np

from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspTrajectoryPlanningError,
)

_LOGGER = logging.getLogger(__name__)


def validate_debug_directory(value: str | Path) -> Path:
    path = Path(value)
    info = path.lstat()
    if not path.is_absolute() or not stat.S_ISDIR(info.st_mode) or info.st_mode & stat.S_IWOTH:
        raise ValueError("Safe-motion debug root must be an existing absolute non-world-writable real directory.")
    return path


def _json_default(value):
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported debug value: {type(value).__name__}")


class SafeMotionDebugRecord:
    """Write private stage checkpoints before planning can return a trajectory."""

    def __init__(self, root, *, reference_frame_id, config, request, context=None, context_files=None):
        root = validate_debug_directory(root)
        try:
            self.directory = Path(tempfile.mkdtemp(prefix="scene-", dir=root))
        except OSError as error:
            raise RM65GraspTrajectoryPlanningError("safe_motion_debug_record_failed", details={
                "debug_record_root": str(root), "exception_type": type(error).__name__, "message": str(error),
            }) from error
        self.started = time.perf_counter()
        self.io_sec = 0.
        self.captures = {}
        self.payload = {
            "schema": "agenticrobot.safe_motion_debug.v1", "state": "recording",
            "reference_frame_id": reference_frame_id, "depth_unit": "meters",
            "rgb_order": "RGB", "depth_alignment": "aligned_to_rgb_not_raw_sensor_depth",
            "policy": asdict(config), "request": request, "context": dict(context or {}),
            "views": {}, "stages": [], "context_files": {},
            "scope": "frozen_planning_inputs_and_outcome_not_execution_evidence",
        }
        self.phase("capture")
        for name, source in (context_files or {}).items():
            # File names belong to the trusted composition, not capture metadata.
            if Path(name).name != name or name in {"", ".", ".."}:
                raise ValueError("Debug context names must be plain filenames.")
            with self._atomic_file(name) as output, Path(source).open("rb") as incoming:
                output.write(incoming.read())
            self.payload["context_files"][name] = {"source": str(source), "file": name}
        self._checkpoint()

    @contextmanager
    def _atomic_file(self, name):
        started = time.perf_counter()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".partial-", delete=False) as stream:
                temporary = Path(stream.name)
                yield stream
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / name)
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (OSError, TypeError, ValueError) as error:
            raise RM65GraspTrajectoryPlanningError("safe_motion_debug_record_failed", details={
                "debug_record_dir": str(self.directory), "file": name,
                "exception_type": type(error).__name__, "message": str(error),
            }) from error
        finally:
            self.io_sec += time.perf_counter() - started
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _checkpoint(self):
        self.payload["elapsed_sec"] = time.perf_counter() - self.started
        self.payload["recording_io_sec"] = self.io_sec
        with self._atomic_file("record.json") as stream:
            stream.write(json.dumps(self.payload, default=_json_default, allow_nan=False,
                                    ensure_ascii=False, indent=2).encode("utf-8"))

    def phase(self, name, *, role=None):
        self.payload["stages"].append({"phase": name, "role": role,
            "wall_time_sec": time.time(), "elapsed_sec": time.perf_counter() - self.started})
        self._checkpoint()

    def timing(self, entry):
        self.payload.setdefault("timing", []).append(dict(entry))
        self._checkpoint()

    def capture(self, capture):
        role = capture.role
        if role not in {"main", "wrist_left", "wrist_right"} or role in self.captures:
            raise ValueError("Debug capture roles must be unique known camera roles.")
        if (np.asarray(capture.rgb).dtype != np.uint8
                or np.asarray(capture.depth_m).dtype.kind not in "fiu"):
            raise ValueError("Debug RGB-D arrays must be uint8 RGB and numeric metric depth, never objects.")
        with self._atomic_file(f"{role}.npz") as stream:
            np.savez(stream, rgb=capture.rgb, depth_m=capture.depth_m)
        from PIL import Image
        with self._atomic_file(f"{role}.png") as stream:
            Image.fromarray(capture.rgb).save(stream, format="PNG")
        self.captures[role] = capture
        self.payload["views"][role] = {
            "source_id": capture.source_id, "role": role, "width": capture.width, "height": capture.height,
            "source_generation": capture.source_generation, "frame_index": capture.frame_index,
            "capture_group_id": capture.capture_group_id, "acquired_at": capture.acquired_at,
            "station_revision": capture.station_revision, "source_binding_hash": capture.source_binding_hash,
            "rgb_content_hash": capture.rgb_content_hash, "depth_content_hash": capture.depth_content_hash,
            "intrinsics": capture.intrinsics, "T_world_from_camera": np.asarray(capture.source_from_camera).reshape(4, 4),
            "capture_qpos_by_side": capture.capture_qpos_by_side,
            "pose_measured_at": capture.pose_measured_at, "pose_available_at": capture.pose_available_at,
            "pose_evidence_refs": capture.pose_evidence_refs,
            "rgbd_file": f"{role}.npz", "rgb_preview": f"{role}.png",
        }
        self.phase("captured", role=role)

    def filtered(self, capture, filtered, valid, points, preprocessing):
        role = capture.role
        rows, columns = np.nonzero(valid)
        with self._atomic_file(f"{role}-filtered.npz") as stream:
            np.savez(stream, self_removed_mask=filtered.removed_mask,
                self_filtered_depth_m=filtered.depth_m, background_valid_mask=valid,
                depth_range_rejected_mask=np.asarray(capture.depth_m) > self.payload["policy"]["maximum_depth_m"],
                denoise_removed_mask=(filtered.depth_m > 0) & ~valid,
                points_world_m=points, pixel_uv=np.column_stack((columns, rows)))
        self.payload["views"][role].update({"filtered_file": f"{role}-filtered.npz",
            "self_filter": filtered.evidence, "depth_preprocessing": preprocessing})
        self.phase("filtered", role=role)

    def scene(self, cloud):
        with self._atomic_file("obstacles.npz") as stream:
            np.savez(stream, points_m=cloud.points_m)
        self.payload["cloud"] = {"file": "obstacles.npz", "scene_hash": cloud.semantic_content_hash,
            "voxel_size_m": cloud.voxel_size_m, "quantization_margin_m": cloud.quantization_margin_m,
            "evidence": cloud.evidence_payload()}
        self.phase("scene_ready")

    def planner(self, planner, qpos):
        # A scene-bound partial uses session-static CAD for this diagnostic only.
        if isinstance(planner, partial):
            planner = planner.func.__self__
        self.payload["planner"] = {"config": asdict(planner.config),
            "search_config": asdict(planner.planner_config), "fixed_scene": planner.scene.to_dict(),
            "deployment_profile_hash": planner.profile.semantic_content_hash,
            "collision_scene_profile": planner.path_collision_scene_profile,
            "tool_tcp_pose_flange_m_rad_by_side": planner.profile.tool_tcp_pose_flange_m_rad_by_side,
            "start_obstacle_proxy_boxes": [
                {"name": name, "T_world_from_box": pose, "half_size_m": half}
                for name, pose, half in planner.oracle.world_obstacle_proxy_boxes(qpos)]}
        self.phase("planning")

    def _projections(self, details):
        results = []
        if isinstance(details, (list, tuple)):
            return [item for value in details for item in self._projections(value)]
        if not isinstance(details, Mapping):
            return results
        nearest = details.get("nearest_observed")
        if nearest is not None:
            point = np.asarray(nearest["occupied_center_world_m"], dtype=float)
            for role, capture in self.captures.items():
                pose = np.asarray(capture.source_from_camera).reshape(4, 4)
                local = pose[:3, :3].T @ (point - pose[:3, 3])
                projection = {"role": role, "geometry_name": nearest["geometry_name"],
                    "point_world_m": point, "projected_depth_m": float(local[2])}
                if local[2] > 0:
                    k = capture.intrinsics
                    uv = np.array([k["fx"] * local[0] / local[2] + k["cx"],
                                   k["fy"] * local[1] / local[2] + k["cy"]])
                    u, v = np.rint(uv).astype(int)
                    inside = 0 <= u < capture.width and 0 <= v < capture.height
                    projection.update(pixel_uv=uv, sampled_pixel_uv=[int(u), int(v)], in_image=inside)
                    if inside:
                        observed = float(capture.depth_m[v, u])
                        projection.update(observed_depth_m=observed,
                            depth_residual_m=None if observed <= 0 else observed - float(local[2]))
                else:
                    projection["in_image"] = False
                results.append(projection)
        for name, value in details.items():
            if name != "nearest_observed":
                results.extend(self._projections(value))
        return results

    def finish(self, *, error=None, trajectory=None):
        self.payload["state"] = ("interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit))
            else "rejected" if isinstance(error, RM65GraspTrajectoryPlanningError)
            else "error" if error is not None else "planned")
        if error is not None:
            self.payload["error"] = {"exception_type": type(error).__name__, "message": str(error)}
            if isinstance(error, RM65GraspTrajectoryPlanningError):
                self.payload["error"].update(reason_code=error.reason_code, details=error.details)
                self.payload["nearest_point_projections"] = self._projections(error.details)
        else:
            self.payload["trajectory_hash"] = trajectory.semantic_content_hash
        self.phase(self.payload["state"])
        _LOGGER.info("Safe-motion debug record: %s state=%s", self.directory, self.payload["state"])
