"""RM65-owned static and measured camera poses for RGB grasp review.

The mutable part is published only from the lower control cycle after it has
consumed one authoritative dual-arm qpos cut.  Render requests only read the
latest immutable cut; they never call the IK/FK solver or accept camera poses
from a caller.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash

from .realsense_perception import (
    RealPerceptionSourceBinding,
    RealRgbCameraGeometryCut,
    real_camera_role_for_source,
)


def _finite_time(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _pose(value: object, *, field_name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field_name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{field_name} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError(f"{field_name} rotation is not rigid")
    return matrix.copy()


def _numbers(value: str | None, *, count: int) -> tuple[float, ...]:
    fields = () if value is None else tuple(value.split())
    if not fields:
        return (0.0,) * count
    if len(fields) != count:
        raise ValueError("URDF origin vector has the wrong length")
    result = tuple(float(item) for item in fields)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("URDF origin vector must be finite")
    return result


def _origin_matrix(origin: ET.Element | None) -> np.ndarray:
    xyz = _numbers(None if origin is None else origin.get("xyz"), count=3)
    roll, pitch, yaw = _numbers(None if origin is None else origin.get("rpy"), count=3)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = xyz
    return matrix


def _fixed_transform_from_ancestor(
    *, urdf_root: ET.Element, ancestor: str, descendant: str
) -> np.ndarray:
    parent_joint_by_child: dict[str, tuple[str, str, np.ndarray]] = {}
    for joint in urdf_root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        child_name = str(child.get("link") or "")
        if child_name in parent_joint_by_child:
            raise ValueError(f"URDF child {child_name!r} has multiple parent joints")
        parent_joint_by_child[child_name] = (
            str(parent.get("link") or ""),
            str(joint.get("type") or ""),
            _origin_matrix(joint.find("origin")),
        )
    current = descendant
    ancestor_from_current = np.eye(4, dtype=np.float64)
    visited: set[str] = set()
    while current != ancestor:
        if current in visited:
            raise ValueError("URDF camera fixed chain contains a cycle")
        visited.add(current)
        try:
            parent, joint_type, parent_from_child = parent_joint_by_child[current]
        except KeyError as error:
            raise ValueError(
                f"URDF frame {descendant!r} is not connected to {ancestor!r}"
            ) from error
        if joint_type != "fixed":
            raise ValueError(
                f"URDF camera chain {ancestor!r}->{descendant!r} is not fixed"
            )
        ancestor_from_current = parent_from_child @ ancestor_from_current
        current = parent
    return _pose(ancestor_from_current, field_name=f"{ancestor}_from_{descendant}")


def _placement_matches_selected_urdf(value: object, *, urdf_path: Path) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    declared = Path(value)
    if declared.is_absolute():
        try:
            return declared.resolve(strict=True) == urdf_path
        except OSError:
            return False
    parts = declared.parts
    return bool(parts) and tuple(urdf_path.parts[-len(parts) :]) == parts


def _require_registry_placement(
    *,
    frame_spec: Mapping[str, object],
    source_id: str,
    camera_frame: str,
    expected_parent: str,
    urdf_path: Path,
) -> None:
    placements = frame_spec.get("placements")
    matches = tuple(
        item
        for item in (placements if isinstance(placements, list) else ())
        if isinstance(item, Mapping)
        and str(item.get("frame")) == camera_frame
        and str(item.get("parent_frame")) == expected_parent
        and _placement_matches_selected_urdf(item.get("urdf"), urdf_path=urdf_path)
    )
    if len(matches) != 1:
        raise ValueError(
            f"camera registry does not bind {source_id!r} once to the selected URDF"
        )


def _require_same_pose(
    actual: object,
    expected: object,
    *,
    field_name: str,
    atol: float = 1e-8,
) -> np.ndarray:
    actual_pose = _pose(actual, field_name=f"{field_name}.actual")
    expected_pose = _pose(expected, field_name=f"{field_name}.expected")
    if not np.allclose(actual_pose, expected_pose, rtol=0.0, atol=atol):
        raise ValueError(f"{field_name} differs between the selected URDF and runtime")
    return actual_pose


@dataclass(frozen=True)
class _DynamicCameraMount:
    source_id: str
    role: str
    side: str
    parent_frame: str
    camera_frame: str
    parent_from_camera: tuple[float, ...]


class RM65RgbCameraGeometry:
    """Thread-safe geometry cut store shared by control and render threads."""

    def __init__(
        self,
        *,
        source_binding: RealPerceptionSourceBinding,
        target_builder: object,
        static_source_from_camera: Mapping[str, np.ndarray],
        dynamic_mounts: Sequence[_DynamicCameraMount],
        urdf_sha256: str,
        max_pose_age_ms: int = 250,
        max_frame_pose_skew_ms: int = 150,
    ) -> None:
        if not isinstance(source_binding, RealPerceptionSourceBinding):
            raise TypeError("source_binding must be typed")
        if not callable(
            getattr(target_builder, "solver_world_targets_for_qpos_pair", None)
        ):
            raise TypeError("target_builder lacks read-only coupled link6 FK")
        for field_name, value in (
            ("max_pose_age_ms", max_pose_age_ms),
            ("max_frame_pose_skew_ms", max_frame_pose_skew_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be positive")
        static = {
            source_id: _pose(value, field_name=f"static_pose.{source_id}")
            for source_id, value in static_source_from_camera.items()
        }
        mounts = tuple(dynamic_mounts)
        if len({item.source_id for item in mounts}) != len(mounts):
            raise ValueError("dynamic camera source IDs must be unique")
        all_sources = set(static) | {item.source_id for item in mounts}
        if all_sources != set(source_binding.camera_sources):
            raise ValueError("every configured RGB source requires one geometry owner")
        self.source_binding = source_binding
        self.target_builder = target_builder
        self._static = static
        self._mount_by_source = {item.source_id: item for item in mounts}
        self.max_pose_age_ms = max_pose_age_ms
        self.max_frame_pose_skew_ms = max_frame_pose_skew_ms
        self._lock = threading.Lock()
        self._dynamic_cut_by_source: dict[str, RealRgbCameraGeometryCut] = {}
        self._semantic_content_hash = canonical_content_hash(
            {
                "source_binding_hash": source_binding.semantic_content_hash,
                "static_source_from_camera": {
                    key: value.tolist() for key, value in sorted(static.items())
                },
                "dynamic_mounts": [item.__dict__ for item in mounts],
                "urdf_sha256": urdf_sha256,
                "pose_semantics": "solver_world_from_camera_optical_frame",
                "publisher": "exclusive_lower_control_cycle_measured_qpos",
                "max_pose_age_ms": max_pose_age_ms,
                "max_frame_pose_skew_ms": max_frame_pose_skew_ms,
            },
            domain="dexteleop.real_rm65.rgb_camera_geometry_binding.v1",
        )

    @property
    def semantic_content_hash(self) -> str:
        return self._semantic_content_hash

    def source_catalog(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            {
                "source_id": source_id,
                "role": real_camera_role_for_source(source_id),
                "native_width": int(source.config.width),
                "native_height": int(source.config.height),
            }
            for source_id, source in sorted(
                self.source_binding.camera_sources.items(),
                key=lambda item: real_camera_role_for_source(item[0]),
            )
        )

    def publish_measured_qpos(
        self,
        *,
        qpos_by_side: Mapping[str, np.ndarray],
        measured_at: float,
        available_at: float,
        evidence_refs: Sequence[str],
    ) -> None:
        if set(qpos_by_side) != {"left", "right"}:
            raise ValueError("camera geometry qpos must contain left and right")
        measured = _finite_time(measured_at, field_name="measured_at")
        available = _finite_time(available_at, field_name="available_at")
        if available < measured:
            raise ValueError("camera pose availability precedes measurement")
        targets = self.target_builder.solver_world_targets_for_qpos_pair(
            left_qpos=np.asarray(qpos_by_side["left"], dtype=np.float64),
            right_qpos=np.asarray(qpos_by_side["right"], dtype=np.float64),
        )
        if set(targets) != {"left", "right"}:
            raise RuntimeError("coupled FK did not return exact left/right link6 poses")
        refs = tuple(sorted({str(item) for item in evidence_refs if str(item)}))
        cuts: dict[str, RealRgbCameraGeometryCut] = {}
        for source_id, mount in self._mount_by_source.items():
            parent_from_camera = np.asarray(
                mount.parent_from_camera, dtype=np.float64
            ).reshape(4, 4)
            source_from_camera = _pose(
                np.asarray(targets[mount.side], dtype=np.float64) @ parent_from_camera,
                field_name=f"dynamic_pose.{source_id}",
            )
            cut_hash = canonical_content_hash(
                {
                    "geometry_binding_hash": self.semantic_content_hash,
                    "source_id": source_id,
                    "source_from_camera": source_from_camera.tolist(),
                    "measured_at": measured,
                    "available_at": available,
                    "source_evidence_refs": refs,
                },
                domain="dexteleop.real_rm65.rgb_camera_geometry_cut.v1",
            )
            cuts[source_id] = RealRgbCameraGeometryCut(
                source_id=source_id,
                source_from_camera=tuple(
                    float(value) for value in source_from_camera.reshape(-1)
                ),
                measured_at=measured,
                available_at=available,
                geometry_binding_hash=self.semantic_content_hash,
                geometry_cut_hash=cut_hash,
                evidence_refs=(*refs, "real-rm65-camera-pose:" + cut_hash),
            )
        with self._lock:
            self._dynamic_cut_by_source = cuts

    def pose_cut(
        self,
        *,
        source_id: str,
        frame_acquired_at: float,
        now: float,
    ) -> RealRgbCameraGeometryCut:
        captured = _finite_time(frame_acquired_at, field_name="frame_acquired_at")
        current = _finite_time(now, field_name="now")
        if source_id in self._static:
            pose = self._static[source_id]
            cut_hash = canonical_content_hash(
                {
                    "geometry_binding_hash": self.semantic_content_hash,
                    "source_id": source_id,
                    "source_from_camera": pose.tolist(),
                },
                domain="dexteleop.real_rm65.rgb_camera_static_cut.v1",
            )
            return RealRgbCameraGeometryCut(
                source_id=source_id,
                source_from_camera=tuple(float(value) for value in pose.reshape(-1)),
                measured_at=captured,
                available_at=captured,
                geometry_binding_hash=self.semantic_content_hash,
                geometry_cut_hash=cut_hash,
                evidence_refs=("real-rm65-static-camera-pose:" + cut_hash,),
            )
        with self._lock:
            cut = self._dynamic_cut_by_source.get(source_id)
        if cut is None:
            raise RuntimeError(f"real_rgb_camera_pose_unavailable:{source_id}")
        if (current - cut.measured_at) * 1000.0 > self.max_pose_age_ms:
            raise RuntimeError(f"real_rgb_camera_pose_stale:{source_id}")
        if abs(captured - cut.measured_at) * 1000.0 > self.max_frame_pose_skew_ms:
            raise RuntimeError(f"real_rgb_camera_frame_pose_skew_exceeded:{source_id}")
        if cut.available_at > current:
            raise RuntimeError(f"real_rgb_camera_pose_from_future:{source_id}")
        return cut


def build_rm65_rgb_camera_geometry(
    *,
    source_binding: RealPerceptionSourceBinding,
    target_builder: object,
    camera_registry_path: str | Path,
    camera_mount_urdf_path: str | Path,
    scene_world_from_arm_base_by_side: Mapping[str, np.ndarray],
    scene_world_from_canonical_station: np.ndarray,
) -> RM65RgbCameraGeometry:
    """Resolve all camera mounts from one Station-compatible selected URDF."""

    registry_path = Path(camera_registry_path).resolve(strict=True)
    urdf_path = Path(camera_mount_urdf_path).resolve(strict=True)
    if not registry_path.is_file() or not urdf_path.is_file():
        raise ValueError("camera registry and mount URDF must be regular files")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    cameras = registry.get("cameras")
    if not isinstance(cameras, Mapping):
        raise TypeError("camera registry lacks cameras")
    urdf_bytes = urdf_path.read_bytes()
    root = ET.fromstring(urdf_bytes)
    if set(scene_world_from_arm_base_by_side) != {"left", "right"}:
        raise ValueError("scene arm-base poses must contain exactly left and right")
    for side in ("left", "right"):
        _require_same_pose(
            _fixed_transform_from_ancestor(
                urdf_root=root,
                ancestor="world",
                descendant=f"{side}_arm_base_link",
            ),
            scene_world_from_arm_base_by_side[side],
            field_name=f"selected_urdf.world_from_{side}_arm_base_link",
        )
    _require_same_pose(
        _fixed_transform_from_ancestor(
            urdf_root=root,
            ancestor="world",
            descendant="workspace_near_boundary_frame",
        ),
        scene_world_from_canonical_station,
        field_name="selected_urdf.world_from_canonical_station",
    )

    for (
        source_id,
        source_from_camera,
    ) in source_binding.world_from_depth_camera_by_source.items():
        spec = cameras.get(source_id)
        frames = (
            None
            if not isinstance(spec, Mapping)
            else (spec.get("extrinsics") or {}).get("frames")
        )
        candidates = tuple(
            (str(frame_name), frame_spec)
            for frame_name, frame_spec in (
                frames.items() if isinstance(frames, Mapping) else ()
            )
            if isinstance(frame_spec, Mapping) and "T_camera_to_parent" in frame_spec
        )
        if len(candidates) != 1:
            raise ValueError(
                f"static camera {source_id!r} requires one registry pose frame"
            )
        camera_frame, frame_spec = candidates[0]
        expected_parent = str(frame_spec.get("parent_frame") or "")
        _require_registry_placement(
            frame_spec=frame_spec,
            source_id=source_id,
            camera_frame=camera_frame,
            expected_parent=expected_parent,
            urdf_path=urdf_path,
        )
        parent_from_camera = _fixed_transform_from_ancestor(
            urdf_root=root,
            ancestor=expected_parent,
            descendant=camera_frame,
        )
        _require_same_pose(
            parent_from_camera,
            frame_spec["T_camera_to_parent"],
            field_name=f"selected_urdf.{expected_parent}_from_{camera_frame}",
        )
        _require_same_pose(
            _fixed_transform_from_ancestor(
                urdf_root=root,
                ancestor="world",
                descendant=camera_frame,
            ),
            source_from_camera,
            field_name=f"selected_urdf.world_from_{camera_frame}",
        )

    dynamic_mounts: list[_DynamicCameraMount] = []
    for source_id in source_binding.camera_sources:
        if source_id in source_binding.world_from_depth_camera_by_source:
            continue
        role = real_camera_role_for_source(source_id)
        side_by_role = {"wrist_left": "left", "wrist_right": "right"}
        if role not in side_by_role:
            raise ValueError(f"RGB source {source_id!r} has no trusted pose policy")
        side = side_by_role[role]
        camera_frame = f"{side}_camera_rgb"
        expected_parent = f"{side}_arm_link6"
        spec = cameras.get(source_id)
        frames = (
            None
            if not isinstance(spec, Mapping)
            else (spec.get("extrinsics") or {}).get("frames")
        )
        frame_spec = (
            None if not isinstance(frames, Mapping) else frames.get(camera_frame)
        )
        if not isinstance(frame_spec, Mapping):
            raise TypeError(
                f"camera registry lacks the reviewed {camera_frame!r} frame"
            )
        _require_registry_placement(
            frame_spec=frame_spec,
            source_id=source_id,
            camera_frame=camera_frame,
            expected_parent=expected_parent,
            urdf_path=urdf_path,
        )
        parent_from_camera = _fixed_transform_from_ancestor(
            urdf_root=root,
            ancestor=expected_parent,
            descendant=camera_frame,
        )
        dynamic_mounts.append(
            _DynamicCameraMount(
                source_id=source_id,
                role=role,
                side=side,
                parent_frame=expected_parent,
                camera_frame=camera_frame,
                parent_from_camera=tuple(
                    float(value) for value in parent_from_camera.reshape(-1)
                ),
            )
        )
    return RM65RgbCameraGeometry(
        source_binding=source_binding,
        target_builder=target_builder,
        static_source_from_camera=(source_binding.world_from_depth_camera_by_source),
        dynamic_mounts=dynamic_mounts,
        urdf_sha256=hashlib.sha256(urdf_bytes).hexdigest(),
    )


__all__ = [
    "RM65RgbCameraGeometry",
    "build_rm65_rgb_camera_geometry",
]
