"""URDF-derived absolute TCP workspace preview and MuJoCo decoration.

This module is simulation-only.  It derives review geometry from fixed URDF
links and writes passive ``user_scn`` decorations.  It owns no production
profile selection, control authority, IK, physics advancement, or dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from project_paths import resolve_path
from robot_motion.target_safety.tcp_pose_envelope import (
    TCP_CONTROLLED_FRAME_ID,
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeSideConfig,
    project_tcp_pose_envelope_side,
)
from robot_motion.transforms import as_pose_matrix
from sim.mujoco.urdf import URDFModel, URDFVisual, parse_urdf


PREVIEW_SCHEMA_VERSION = "sim_real_tests.tcp_pose_absolute_workspace_preview.v1"
PREVIEW_PROFILE_NAMES = (
    "absolute_loose",
    "absolute_standard",
    "absolute_tight",
)
_REFERENCE_FRAME_ID = "desk"
_PREVIEW_TARGET_BINDING_HASH = "preview:urdf_workspace_near_boundary_frame"
_PREVIEW_TOOL_BINDING_HASH = "preview:tool_tcp"
_RAW_COLOR = np.array([1.0, 0.28, 0.03, 0.94], dtype=np.float32)
_EFFECTIVE_COLOR = np.array([0.10, 1.0, 0.22, 0.96], dtype=np.float32)
_UNREACHABLE_COLOR = np.array([1.0, 0.05, 0.05, 0.96], dtype=np.float32)
_ACTUAL_COLOR = np.array([0.92, 0.10, 1.0, 0.96], dtype=np.float32)
_DESK_AXIS_COLORS = (
    np.array([1.0, 0.05, 0.05, 0.95], dtype=np.float32),
    np.array([0.05, 0.85, 0.05, 0.95], dtype=np.float32),
    np.array([0.08, 0.3, 1.0, 0.95], dtype=np.float32),
)


def _required_text(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _finite_scalar(value: object, *, name: str, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        suffix = " and positive" if positive else ""
        raise ValueError(f"{name} must be finite{suffix}")
    return result


def _rgba(value: object, *, name: str) -> tuple[float, float, float, float]:
    try:
        result = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain four finite values") from exc
    if len(result) != 4 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain four finite values")
    if any(item < 0.0 or item > 1.0 for item in result):
        raise ValueError(f"{name} values must be within [0, 1]")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class AbsoluteWorkspacePreviewSpec:
    profile_name: str
    xy_table_scale: float
    z_pole_height_scale: float
    x_alignment: str
    z_min_clearance_m: float
    rgba: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile_name",
            _required_text(self.profile_name, name="profile_name"),
        )
        object.__setattr__(
            self,
            "xy_table_scale",
            _finite_scalar(
                self.xy_table_scale,
                name="xy_table_scale",
                positive=True,
            ),
        )
        z_scale = _finite_scalar(
            self.z_pole_height_scale,
            name="z_pole_height_scale",
            positive=True,
        )
        if z_scale > 1.0:
            raise ValueError("z_pole_height_scale must not exceed 1")
        object.__setattr__(self, "z_pole_height_scale", z_scale)
        x_alignment = _required_text(self.x_alignment, name="x_alignment")
        if x_alignment not in {"center", "positive_open"}:
            raise ValueError(
                "x_alignment must be either 'center' or 'positive_open'"
            )
        object.__setattr__(self, "x_alignment", x_alignment)
        z_min_clearance_m = _finite_scalar(
            self.z_min_clearance_m,
            name="z_min_clearance_m",
        )
        if z_min_clearance_m < 0.0:
            raise ValueError("z_min_clearance_m must be non-negative")
        object.__setattr__(self, "z_min_clearance_m", z_min_clearance_m)
        object.__setattr__(self, "rgba", _rgba(self.rgba, name="rgba"))


@dataclass(frozen=True)
class AbsoluteWorkspacePreviewConfig:
    urdf_path: Path
    reference_frame_link: str
    table_link: str
    camera_pole_link: str
    profile_by_name: Mapping[str, AbsoluteWorkspacePreviewSpec]

    def __post_init__(self) -> None:
        path = Path(self.urdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        profiles = dict(self.profile_by_name)
        if set(profiles) != set(PREVIEW_PROFILE_NAMES):
            raise ValueError(
                "preview profiles must contain exactly "
                f"{list(PREVIEW_PROFILE_NAMES)}"
            )
        object.__setattr__(self, "urdf_path", path)
        object.__setattr__(
            self,
            "reference_frame_link",
            _required_text(self.reference_frame_link, name="reference_frame_link"),
        )
        object.__setattr__(
            self,
            "table_link",
            _required_text(self.table_link, name="table_link"),
        )
        object.__setattr__(
            self,
            "camera_pole_link",
            _required_text(self.camera_pole_link, name="camera_pole_link"),
        )
        object.__setattr__(
            self,
            "profile_by_name",
            MappingProxyType(dict(sorted(profiles.items()))),
        )


@dataclass(frozen=True)
class DerivedAbsoluteWorkspace:
    profile_name: str
    urdf_path: Path
    reference_frame_link: str
    desk_pose_world: np.ndarray
    absolute_min_xyz_m: tuple[float, float, float]
    absolute_max_xyz_m: tuple[float, float, float]
    table_min_xyz_m: tuple[float, float, float]
    table_max_xyz_m: tuple[float, float, float]
    pole_top_z_m: float
    rgba: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        desk_pose_world = as_pose_matrix(self.desk_pose_world).copy()
        lower = np.asarray(self.absolute_min_xyz_m, dtype=float)
        upper = np.asarray(self.absolute_max_xyz_m, dtype=float)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
            raise ValueError("absolute workspace bounds must be finite non-empty 3D bounds")
        desk_pose_world.setflags(write=False)
        object.__setattr__(self, "desk_pose_world", desk_pose_world)

    @property
    def center_xyz_m(self) -> np.ndarray:
        return 0.5 * (
            np.asarray(self.absolute_min_xyz_m, dtype=float)
            + np.asarray(self.absolute_max_xyz_m, dtype=float)
        )

    @property
    def size_xyz_m(self) -> np.ndarray:
        return (
            np.asarray(self.absolute_max_xyz_m, dtype=float)
            - np.asarray(self.absolute_min_xyz_m, dtype=float)
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "profile_name": self.profile_name,
            "urdf_path": str(self.urdf_path),
            "reference_frame_link": self.reference_frame_link,
            "absolute_min_xyz_m": list(self.absolute_min_xyz_m),
            "absolute_max_xyz_m": list(self.absolute_max_xyz_m),
            "table_min_xyz_m": list(self.table_min_xyz_m),
            "table_max_xyz_m": list(self.table_max_xyz_m),
            "pole_top_z_m": float(self.pole_top_z_m),
            "desk_pose_world": self.desk_pose_world.tolist(),
        }


def load_absolute_workspace_preview_config(
    path: str | Path,
) -> AbsoluteWorkspacePreviewConfig:
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("absolute workspace preview config must be an object")
    allowed = {
        "schema_version",
        "role",
        "urdf_path",
        "reference_frame_link",
        "table_link",
        "camera_pole_link",
        "profiles",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"absolute workspace preview config has unknown fields: {unknown}")
    if payload.get("schema_version") != PREVIEW_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {PREVIEW_SCHEMA_VERSION!r}"
        )
    if payload.get("role") != "visual_review_only_not_a_runtime_safety_profile":
        raise ValueError("preview config must declare its non-production role")
    profiles_payload = payload.get("profiles")
    if not isinstance(profiles_payload, Mapping):
        raise ValueError("profiles must be an object")
    profiles: dict[str, AbsoluteWorkspacePreviewSpec] = {}
    for profile_name, raw_spec in profiles_payload.items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"profiles.{profile_name} must be an object")
        unknown_spec = sorted(
            set(raw_spec)
            - {
                "xy_table_scale",
                "z_pole_height_scale",
                "x_alignment",
                "z_min_clearance_m",
                "rgba",
            }
        )
        if unknown_spec:
            raise ValueError(
                f"profiles.{profile_name} has unknown fields: {unknown_spec}"
            )
        profiles[str(profile_name)] = AbsoluteWorkspacePreviewSpec(
            profile_name=str(profile_name),
            xy_table_scale=raw_spec["xy_table_scale"],
            z_pole_height_scale=raw_spec["z_pole_height_scale"],
            x_alignment=raw_spec["x_alignment"],
            z_min_clearance_m=raw_spec["z_min_clearance_m"],
            rgba=raw_spec["rgba"],
        )
    return AbsoluteWorkspacePreviewConfig(
        urdf_path=resolve_path(str(payload["urdf_path"])),
        reference_frame_link=str(payload["reference_frame_link"]),
        table_link=str(payload["table_link"]),
        camera_pole_link=str(payload["camera_pole_link"]),
        profile_by_name=profiles,
    )


def _fixed_link_pose_world(
    model: URDFModel,
    link_name: str,
    *,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    if link_name in cache:
        return cache[link_name].copy()
    if link_name == model.root_link:
        pose = np.eye(4, dtype=float)
    else:
        joint = model.parent_joint_by_child.get(link_name)
        if joint is None:
            raise ValueError(f"URDF link {link_name!r} is disconnected from the root")
        if joint.joint_type != "fixed":
            raise ValueError(
                f"URDF link {link_name!r} is not fixed to the world; "
                f"joint={joint.name!r} type={joint.joint_type!r}"
            )
        pose = _fixed_link_pose_world(model, joint.parent, cache=cache) @ joint.origin
    cache[link_name] = pose.copy()
    return pose


def _single_box_visual(model: URDFModel, link_name: str) -> URDFVisual:
    link = model.links.get(link_name)
    if link is None:
        raise ValueError(f"URDF lacks required link {link_name!r}")
    boxes = tuple(visual for visual in link.visuals if visual.geometry.kind == "box")
    if len(boxes) != 1 or len(boxes[0].geometry.values) != 3:
        raise ValueError(
            f"URDF link {link_name!r} must contain exactly one box visual"
        )
    return boxes[0]


def _box_corners_in_frame(
    *,
    link_pose_world: np.ndarray,
    visual: URDFVisual,
    frame_pose_world: np.ndarray,
) -> np.ndarray:
    size = np.asarray(visual.geometry.values, dtype=float)
    local_corners = np.array(
        [
            [sx * size[0], sy * size[1], sz * size[2], 1.0]
            for sx in (-0.5, 0.5)
            for sy in (-0.5, 0.5)
            for sz in (-0.5, 0.5)
        ],
        dtype=float,
    )
    world_from_visual = link_pose_world @ visual.origin
    frame_from_world = np.linalg.inv(frame_pose_world)
    return (frame_from_world @ world_from_visual @ local_corners.T).T[:, :3]


def derive_absolute_workspace(
    config: AbsoluteWorkspacePreviewConfig,
    profile_name: str,
) -> DerivedAbsoluteWorkspace:
    normalized_name = _required_text(profile_name, name="profile_name")
    try:
        spec = config.profile_by_name[normalized_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown preview profile {normalized_name!r}; "
            f"choose one of {list(config.profile_by_name)}"
        ) from exc
    model = parse_urdf(config.urdf_path)
    pose_cache: dict[str, np.ndarray] = {}
    desk_pose_world = _fixed_link_pose_world(
        model,
        config.reference_frame_link,
        cache=pose_cache,
    )
    table_corners = _box_corners_in_frame(
        link_pose_world=_fixed_link_pose_world(
            model,
            config.table_link,
            cache=pose_cache,
        ),
        visual=_single_box_visual(model, config.table_link),
        frame_pose_world=desk_pose_world,
    )
    pole_corners = _box_corners_in_frame(
        link_pose_world=_fixed_link_pose_world(
            model,
            config.camera_pole_link,
            cache=pose_cache,
        ),
        visual=_single_box_visual(model, config.camera_pole_link),
        frame_pose_world=desk_pose_world,
    )
    table_lower = np.min(table_corners, axis=0)
    table_upper = np.max(table_corners, axis=0)
    table_center_xy = 0.5 * (table_lower[:2] + table_upper[:2])
    table_size_xy = table_upper[:2] - table_lower[:2]
    half_xy = 0.5 * table_size_xy * float(spec.xy_table_scale)
    if spec.x_alignment == "positive_open":
        loose_xy_scale = float(
            config.profile_by_name["absolute_loose"].xy_table_scale
        )
        x_max = float(
            table_center_xy[0]
            + 0.5 * table_size_xy[0] * loose_xy_scale
        )
        x_min = float(x_max - 2.0 * half_xy[0])
    else:
        x_min = float(table_center_xy[0] - half_xy[0])
        x_max = float(table_center_xy[0] + half_xy[0])
    tabletop_z = float(table_upper[2])
    z_min = tabletop_z + float(spec.z_min_clearance_m)
    pole_top_z = float(np.max(pole_corners[:, 2]))
    if pole_top_z <= tabletop_z:
        raise ValueError(
            "camera pole top must be above the tabletop in the reference frame"
        )
    z_max = tabletop_z + float(spec.z_pole_height_scale) * (
        pole_top_z - tabletop_z
    )
    if z_max <= z_min:
        raise ValueError(
            "z_min_clearance_m leaves no non-empty vertical workspace"
        )
    lower = np.array(
        [x_min, table_center_xy[1] - half_xy[1], z_min],
        dtype=float,
    )
    upper = np.array(
        [x_max, table_center_xy[1] + half_xy[1], z_max],
        dtype=float,
    )
    return DerivedAbsoluteWorkspace(
        profile_name=normalized_name,
        urdf_path=config.urdf_path,
        reference_frame_link=config.reference_frame_link,
        desk_pose_world=desk_pose_world,
        absolute_min_xyz_m=tuple(float(value) for value in lower),
        absolute_max_xyz_m=tuple(float(value) for value in upper),
        table_min_xyz_m=tuple(float(value) for value in table_lower),
        table_max_xyz_m=tuple(float(value) for value in table_upper),
        pole_top_z_m=pole_top_z,
        rgba=spec.rgba,
    )


def preview_envelope_profile(
    workspace: DerivedAbsoluteWorkspace,
    *,
    sides: Sequence[str] = ("left", "right"),
) -> TCPPoseEnvelopeProfile:
    normalized_sides = tuple(_required_text(side, name="side") for side in sides)
    if not normalized_sides or len(set(normalized_sides)) != len(normalized_sides):
        raise ValueError("sides must be non-empty and unique")
    side_config = TCPPoseEnvelopeSideConfig(
        absolute_min_xyz_m=workspace.absolute_min_xyz_m,
        absolute_max_xyz_m=workspace.absolute_max_xyz_m,
    )
    return TCPPoseEnvelopeProfile(
        profile_id=f"preview:{workspace.profile_name}",
        target_frame_binding_hash=_PREVIEW_TARGET_BINDING_HASH,
        tool_frame_binding_hash_by_side={
            side: _PREVIEW_TOOL_BINDING_HASH for side in normalized_sides
        },
        reference_frame_by_side={
            side: _REFERENCE_FRAME_ID for side in normalized_sides
        },
        side_config_by_side={side: side_config for side in normalized_sides},
        controlled_frame_id=TCP_CONTROLLED_FRAME_ID,
    )


def build_review_trajectory(
    workspace: DerivedAbsoluteWorkspace,
    *,
    samples_per_leg: int = 40,
    start_pose_desk: np.ndarray | None = None,
    include_negative_faces: bool = True,
    endpoint_hold_frames: int = 0,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Return a full-range raw/effective desk-frame TCP review trajectory.

    The preview is intentionally a workspace *candidate* rather than an IK
    reachability guarantee.  It visits both the lower and upper face of each
    axis by default, with a small raw overshoot so the envelope projector is
    exercised at both boundaries.  Optional endpoint holds are useful for a
    closed-loop visualizer: the target stays fixed while the simulated servo
    catches up, instead of making a large workspace excursion look small
    merely because the reference timeline was too short.
    """

    if isinstance(samples_per_leg, bool) or int(samples_per_leg) < 2:
        raise ValueError("samples_per_leg must be an integer >= 2")
    if not isinstance(include_negative_faces, bool):
        raise TypeError("include_negative_faces must be a bool")
    if (
        isinstance(endpoint_hold_frames, bool)
        or int(endpoint_hold_frames) != endpoint_hold_frames
        or int(endpoint_hold_frames) < 0
    ):
        raise ValueError("endpoint_hold_frames must be a non-negative integer")
    endpoint_hold_frames = int(endpoint_hold_frames)
    lower = np.asarray(workspace.absolute_min_xyz_m, dtype=float)
    upper = np.asarray(workspace.absolute_max_xyz_m, dtype=float)
    span = upper - lower
    if start_pose_desk is None:
        start_pose = np.eye(4, dtype=float)
        start_pose[:3, 3] = 0.5 * (lower + upper)
    else:
        start_pose = as_pose_matrix(start_pose_desk).copy()
    start = start_pose[:3, 3].copy()
    if np.any(start < lower) or np.any(start > upper):
        raise ValueError("start_pose_desk must begin inside the selected workspace")
    excursions = []
    for axis in range(3):
        lower_point = start.copy()
        lower_point[axis] = lower[axis] - 0.08 * span[axis]
        upper_point = start.copy()
        upper_point[axis] = upper[axis] + 0.08 * span[axis]
        excursions.append((lower_point, upper_point))
    # Start at the actual Home TCP and cross one axis at a time.  Visiting
    # both faces catches the asymmetric reachable/collision boundary that a
    # positive-only sweep hides; returning to Home between axes keeps the IK
    # review interpretable and avoids a diagonal corner jump.
    points: list[tuple[np.ndarray, np.ndarray]] = []
    for lower_point, upper_point in excursions:
        if include_negative_faces:
            points.extend(
                (
                    (start.copy(), lower_point.copy()),
                    (lower_point.copy(), upper_point.copy()),
                    (upper_point.copy(), start.copy()),
                )
            )
        else:
            points.extend(
                (
                    (start.copy(), upper_point.copy()),
                    (upper_point.copy(), start.copy()),
                )
            )

    def _pose(position: np.ndarray) -> np.ndarray:
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = np.asarray(position, dtype=float)
        pose[:3, :3] = start_pose[:3, :3]
        return pose

    raw: list[np.ndarray] = []
    for segment_start, segment_end in points:
        for alpha in np.linspace(0.0, 1.0, int(samples_per_leg), endpoint=False):
            raw.append(
                _pose(
                    (1.0 - alpha) * segment_start + alpha * segment_end
                )
            )
        if endpoint_hold_frames:
            endpoint = _pose(segment_end)
            raw.extend(endpoint.copy() for _ in range(endpoint_hold_frames + 1))
    raw.append(_pose(points[-1][1]))

    profile = preview_envelope_profile(workspace)
    effective = tuple(
        project_tcp_pose_envelope_side(
            "left",
            pose,
            profile=profile,
            target_frame_binding_hash=_PREVIEW_TARGET_BINDING_HASH,
            tool_frame_binding_hash_by_side={
                "left": _PREVIEW_TOOL_BINDING_HASH,
                "right": _PREVIEW_TOOL_BINDING_HASH,
            },
            controlled_frame_id=TCP_CONTROLLED_FRAME_ID,
            reference_frame_by_side={
                "left": _REFERENCE_FRAME_ID,
                "right": _REFERENCE_FRAME_ID,
            },
            execution_epoch=0,
            anchor_receipt=None,
        ).effective_tcp_pose_by_side["left"]
        for pose in raw
    )
    return tuple(raw), effective


def trajectory_to_world(
    workspace: DerivedAbsoluteWorkspace,
    poses_desk: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    return tuple(
        workspace.desk_pose_world @ as_pose_matrix(pose_desk)
        for pose_desk in poses_desk
    )


@dataclass(frozen=True)
class AbsoluteWorkspaceVisualizationConfig:
    edge_radius_m: float = 0.0025
    trail_radius_m: float = 0.002
    point_radius_m: float = 0.012
    residual_radius_m: float = 0.0012
    axis_length_m: float = 0.16
    axis_radius_m: float = 0.0025

    def __post_init__(self) -> None:
        for name in (
            "edge_radius_m",
            "trail_radius_m",
            "point_radius_m",
            "residual_radius_m",
            "axis_length_m",
            "axis_radius_m",
        ):
            value = _finite_scalar(getattr(self, name), name=name, positive=True)
            object.__setattr__(self, name, value)


class AbsoluteWorkspaceVisualizer:
    """Render one absolute box plus raw/effective conceptual TCP trajectories."""

    _BOX_GEOMS = 13
    _AXIS_GEOMS = 3
    _CURRENT_GEOMS = 4

    def __init__(
        self,
        *,
        workspace: DerivedAbsoluteWorkspace,
        raw_pose_world: Sequence[np.ndarray],
        effective_pose_world: Sequence[np.ndarray],
        actual_pose_world: Sequence[np.ndarray],
        config: AbsoluteWorkspaceVisualizationConfig | None = None,
        target_reachability_by_frame: Sequence[str] | None = None,
    ) -> None:
        self.workspace = workspace
        self.raw_pose_world = tuple(as_pose_matrix(pose).copy() for pose in raw_pose_world)
        self.effective_pose_world = tuple(
            as_pose_matrix(pose).copy() for pose in effective_pose_world
        )
        self.actual_pose_world = tuple(
            as_pose_matrix(pose).copy() for pose in actual_pose_world
        )
        counts = {
            len(self.raw_pose_world),
            len(self.effective_pose_world),
            len(self.actual_pose_world),
        }
        if len(self.raw_pose_world) < 2 or len(counts) != 1:
            raise ValueError(
                "raw/effective/actual trajectories must have equal length >= 2"
            )
        self.config = config or AbsoluteWorkspaceVisualizationConfig()
        if target_reachability_by_frame is None:
            reachability = tuple("unknown" for _ in self.raw_pose_world)
        else:
            reachability = tuple(str(value) for value in target_reachability_by_frame)
            if len(reachability) != len(self.raw_pose_world):
                raise ValueError(
                    "target_reachability_by_frame must match trajectory length"
                )
        allowed_reachability = {"unknown", "reachable", "unreachable_or_unmet"}
        unknown_reachability = sorted(set(reachability) - allowed_reachability)
        if unknown_reachability:
            raise ValueError(
                "target_reachability_by_frame contains unknown states: "
                f"{unknown_reachability}"
            )
        self.target_reachability_by_frame = reachability
        self._base_geom_index: int | None = None

    @property
    def frame_count(self) -> int:
        return len(self.raw_pose_world)

    @property
    def geom_count(self) -> int:
        trail_count = 3 * (self.frame_count - 1)
        return self._BOX_GEOMS + self._AXIS_GEOMS + trail_count + self._CURRENT_GEOMS

    def update(self, viewer_handle: Any, *, frame_index: int) -> bool:
        if viewer_handle is None:
            return False
        index = int(frame_index)
        if isinstance(frame_index, bool) or not 0 <= index < self.frame_count:
            raise ValueError(f"frame_index must be within [0, {self.frame_count - 1}]")
        lock = getattr(viewer_handle, "lock", None)
        if not callable(lock):
            raise TypeError("viewer_handle must expose lock()")
        with lock():
            scene = getattr(viewer_handle, "user_scn", None)
            if scene is None:
                raise TypeError("viewer_handle must expose user_scn")
            if self._base_geom_index is None:
                self._base_geom_index = int(scene.ngeom)
            stop = self._base_geom_index + self.geom_count
            if stop > int(scene.maxgeom):
                raise RuntimeError(
                    "MuJoCo viewer user_scn has insufficient geometry capacity: "
                    f"need={stop} max={scene.maxgeom}"
                )
            cursor = self._base_geom_index
            cursor = self._write_box(scene, cursor)
            cursor = self._write_axes(scene, cursor)
            for poses, color, label in (
                (self.raw_pose_world, _RAW_COLOR, "absolute_workspace_raw_trail"),
                (
                    self.effective_pose_world,
                    _EFFECTIVE_COLOR,
                    "absolute_workspace_effective_trail",
                ),
                (
                    self.actual_pose_world,
                    _ACTUAL_COLOR,
                    "absolute_workspace_actual_trail",
                ),
            ):
                for start, end in zip(poses[:-1], poses[1:]):
                    self._write_segment(
                        scene.geoms[cursor],
                        start=start[:3, 3],
                        end=end[:3, 3],
                        radius=float(self.config.trail_radius_m),
                        rgba=color,
                        label=label,
                    )
                    cursor += 1
            raw_pose = self.raw_pose_world[index]
            effective_pose = self.effective_pose_world[index]
            actual_pose = self.actual_pose_world[index]
            self._write_sphere(
                scene.geoms[cursor],
                position=raw_pose[:3, 3],
                radius=float(self.config.point_radius_m),
                rgba=_RAW_COLOR,
                label="absolute_workspace_raw_tcp",
            )
            cursor += 1
            self._write_sphere(
                scene.geoms[cursor],
                position=effective_pose[:3, 3],
                radius=float(self.config.point_radius_m),
                rgba=(
                    _UNREACHABLE_COLOR
                    if self.target_reachability_by_frame[index]
                    == "unreachable_or_unmet"
                    else _EFFECTIVE_COLOR
                ),
                label="absolute_workspace_effective_tcp",
            )
            cursor += 1
            self._write_sphere(
                scene.geoms[cursor],
                position=actual_pose[:3, 3],
                radius=float(self.config.point_radius_m),
                rgba=_ACTUAL_COLOR,
                label="absolute_workspace_actual_tcp",
            )
            cursor += 1
            self._write_segment(
                scene.geoms[cursor],
                start=raw_pose[:3, 3],
                end=effective_pose[:3, 3],
                radius=float(self.config.residual_radius_m),
                rgba=np.array([1.0, 0.9, 0.05, 0.9], dtype=np.float32),
                label="absolute_workspace_projection_residual",
            )
            scene.ngeom = max(int(scene.ngeom), stop)
        return True

    def _write_box(self, scene: Any, cursor: int) -> int:
        import mujoco

        center_desk = np.ones(4, dtype=float)
        center_desk[:3] = self.workspace.center_xyz_m
        center_world = (self.workspace.desk_pose_world @ center_desk)[:3]
        half_size = 0.5 * self.workspace.size_xyz_m
        rgba = np.asarray(self.workspace.rgba, dtype=np.float32)
        mujoco.mjv_initGeom(
            scene.geoms[cursor],
            mujoco.mjtGeom.mjGEOM_BOX,
            half_size,
            center_world,
            self.workspace.desk_pose_world[:3, :3].reshape(-1),
            rgba,
        )
        scene.geoms[cursor].category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        scene.geoms[cursor].label = f"{self.workspace.profile_name}_volume"
        cursor += 1
        lower = np.asarray(self.workspace.absolute_min_xyz_m, dtype=float)
        upper = np.asarray(self.workspace.absolute_max_xyz_m, dtype=float)
        corners_desk = tuple(
            np.array([x, y, z, 1.0], dtype=float)
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        )
        corners_world = tuple(
            (self.workspace.desk_pose_world @ corner)[:3]
            for corner in corners_desk
        )
        edge_indices = (
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7), (6, 7),
        )
        edge_color = rgba.copy()
        edge_color[3] = max(0.65, float(edge_color[3]))
        for first, second in edge_indices:
            self._write_segment(
                scene.geoms[cursor],
                start=corners_world[first],
                end=corners_world[second],
                radius=float(self.config.edge_radius_m),
                rgba=edge_color,
                label=f"{self.workspace.profile_name}_edge",
            )
            cursor += 1
        return cursor

    def _write_axes(self, scene: Any, cursor: int) -> int:
        origin = self.workspace.desk_pose_world[:3, 3]
        for axis_index, rgba in enumerate(_DESK_AXIS_COLORS):
            endpoint = origin + (
                float(self.config.axis_length_m)
                * self.workspace.desk_pose_world[:3, axis_index]
            )
            self._write_segment(
                scene.geoms[cursor],
                start=origin,
                end=endpoint,
                radius=float(self.config.axis_radius_m),
                rgba=rgba,
                label=f"desk_{'xyz'[axis_index]}_axis",
            )
            cursor += 1
        return cursor

    @staticmethod
    def _write_sphere(
        geom: Any,
        *,
        position: np.ndarray,
        radius: float,
        rgba: np.ndarray,
        label: str,
    ) -> None:
        import mujoco

        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius], dtype=float),
            np.asarray(position, dtype=float),
            np.eye(3, dtype=float).reshape(-1),
            rgba,
        )
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        geom.label = label

    @staticmethod
    def _write_segment(
        geom: Any,
        *,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        rgba: np.ndarray,
        label: str,
    ) -> None:
        import mujoco

        start_point = np.asarray(start, dtype=float)
        end_point = np.asarray(end, dtype=float)
        if float(np.linalg.norm(end_point - start_point)) <= 1e-12:
            AbsoluteWorkspaceVisualizer._write_sphere(
                geom,
                position=start_point,
                radius=radius,
                rgba=rgba,
                label=label,
            )
            return
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            np.eye(3, dtype=float).reshape(-1),
            rgba,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            start_point,
            end_point,
        )
        geom.rgba = rgba
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        geom.label = label


__all__ = [
    "AbsoluteWorkspacePreviewConfig",
    "AbsoluteWorkspacePreviewSpec",
    "AbsoluteWorkspaceVisualizationConfig",
    "AbsoluteWorkspaceVisualizer",
    "DerivedAbsoluteWorkspace",
    "PREVIEW_PROFILE_NAMES",
    "PREVIEW_SCHEMA_VERSION",
    "build_review_trajectory",
    "derive_absolute_workspace",
    "load_absolute_workspace_preview_config",
    "preview_envelope_profile",
    "trajectory_to_world",
]
