"""Camera-neutral projection and image overlay for DMTAC M grasp poses.

The renderer is intentionally geometric only.  It does not test scene depth,
occlusion, reachability, collision, or motion feasibility.  A caller supplies
one reference-frame pose for each grasp and the pose of the target camera in
the same reference frame; this module performs the frame transform, clips the
wireframe, and draws it over an already captured image.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from robot_motion.grasp_visible_scene import ParallelJawSolidGeometry

GRASP_VISIBILITY_STATES = frozenset(
    {"visible", "partially_visible", "behind_camera", "out_of_frame"}
)

# The reviewed DMTAC M adapter maps the VGN palm/finger boundary to the
# confirmed 140 mm tool_tcp by a 42.999 mm local approach-axis translation.
# The 97.001 mm source is an internal marker in the current zero-translation
# 4B2 proxy attachment, not an independent flange-mating calibration.  Keep
# the resulting reviewed transform in this geometry owner so planning,
# collision checks, and overlays cannot silently use different TCP origins.
DMTAC_M_TOOL_TCP_FROM_FLANGE_Z_M = 0.140
DMTAC_M_REVIEWED_TOOL_TCP_POSE_FLANGE_M_RAD = (
    0.0,
    0.0,
    DMTAC_M_TOOL_TCP_FROM_FLANGE_Z_M,
    0.0,
    0.0,
    0.0,
)
DMTAC_M_CAD_PALM_FROM_ATTACHMENT_Z_M = 0.097001
DMTAC_M_PALM_FROM_TOOL_TCP_Z_M = (
    DMTAC_M_CAD_PALM_FROM_ATTACHMENT_Z_M - DMTAC_M_TOOL_TCP_FROM_FLANGE_Z_M
)
DMTAC_M_FINGER_LENGTH_M = 0.050
DMTAC_M_FINGER_THICKNESS_M = 0.010
DMTAC_M_PAD_HEIGHT_M = 0.0255

# The saved DMTAC-GraspGenX alignment is a separate model profile.  The
# legacy constants above remain the original 4B2 proxy used by the older
# VGN/GraspNet and diagnostic paths.  These values are copied from the
# reviewed ``graspgenx-dmtac-alignment-right.json`` payload and are expressed
# in the GraspGenX model frame, not in the legacy 4B2 solid frame.
DMTAC_M_GRASPGENX_MAX_OPENING_M = 0.0705
DMTAC_M_GRASPGENX_PALM_Z_M = -0.029
DMTAC_M_GRASPGENX_FINGER_LENGTH_M = 0.100
DMTAC_M_GRASPGENX_FINGER_THICKNESS_M = 0.0199
DMTAC_M_GRASPGENX_PAD_HEIGHT_M = 0.030
DMTAC_M_GRASPGENX_GRASP_CENTER_FROM_TACTILE_XYZ_M = (0.0, 0.0, -0.022)
DMTAC_M_GRASPGENX_GRASP_CENTER_FROM_TACTILE_Z_M = (
    DMTAC_M_GRASPGENX_GRASP_CENTER_FROM_TACTILE_XYZ_M[2]
)
DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID = "DMTAC_M_graspgenx_alignment_v4_q0"

# T_tool_tcp_from_graspgenx_model at the reviewed URDF pose (q=0, master q=0).
# Keeping this as a full rigid transform preserves the small CAD/tactile frame
# tilt that a yaw-only proxy cannot represent.  It is a fixed deployment
# calibration; the current point-grasp contract does not yet carry a live
# DMTAC opening/q-position into the perception planner.
DMTAC_M_GRASPGENX_TOOL_FROM_MODEL_ROTATION = (
    -0.005883664976856,
    0.9999446749409,
    0.008719492172192,
    -0.9999826899785,
    -0.005883853041647,
    -0.000004084338666,
    0.00004722009784,
    -0.008719365268476,
    0.9999619844972,
)
DMTAC_M_GRASPGENX_TOOL_FROM_MODEL_XYZ_M = (
    0.02283489505130,
    -0.000005278121116,
    0.02618158377488,
)


def reviewed_4b2_cad_solid_geometry(*, maximum_opening_m: float = 0.070) -> ParallelJawSolidGeometry:
    """Bind the reviewed full 4B2 assembly to the unchanged 140 mm real TCP.

    The URDF arm_link6 -> CAD base chain translates 24.5 mm laterally and
    8.2 mm axially. CAD jaws close along flange Y, not controller TCP X.
    Keep the model's +X-closing geometry frame distinct from controller TCP;
    this is a nominal solid adapter, not a new hardware calibration.
    Source: rm65_dual_arm_wrist_cameras_v4_head_camera_4b2_cad_with_legacy_housing.urdf.
    """
    return ParallelJawSolidGeometry(
        palm_z_m=DMTAC_M_PALM_FROM_TOOL_TCP_Z_M,
        finger_length_m=DMTAC_M_FINGER_LENGTH_M,
        finger_thickness_m=DMTAC_M_FINGER_THICKNESS_M,
        pad_height_m=DMTAC_M_PAD_HEIGHT_M,
        maximum_opening_m=maximum_opening_m,
        tool_from_solid_xyz_m=(0.0245, -8.9993525e-8, 0.0082),
        solid_yaw_rad=math.pi / 2,
    )


def reviewed_graspgenx_dmtac_solid_geometry(
    *, maximum_opening_m: float = DMTAC_M_GRASPGENX_MAX_OPENING_M
) -> ParallelJawSolidGeometry:
    """Return the saved DMTAC-specific GraspGenX geometry profile.

    This profile is intentionally independent of ``reviewed_4b2_cad_solid_geometry``.
    Its local dimensions and the calibrated TCP-to-model transform come from
    the saved GUI alignment; the legacy proxy remains available to consumers
    that explicitly request it.
    """
    return ParallelJawSolidGeometry(
        palm_z_m=DMTAC_M_GRASPGENX_PALM_Z_M,
        finger_length_m=DMTAC_M_GRASPGENX_FINGER_LENGTH_M,
        finger_thickness_m=DMTAC_M_GRASPGENX_FINGER_THICKNESS_M,
        pad_height_m=DMTAC_M_GRASPGENX_PAD_HEIGHT_M,
        maximum_opening_m=maximum_opening_m,
        tool_from_solid_xyz_m=DMTAC_M_GRASPGENX_TOOL_FROM_MODEL_XYZ_M,
        tool_from_solid_rotation=DMTAC_M_GRASPGENX_TOOL_FROM_MODEL_ROTATION,
    )


def _rigid_transform(value: Any, *, field_name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"{field_name} must contain 16 values.")
    matrix = matrix.reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field_name} must be finite.")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
        raise ValueError(f"{field_name} has an invalid homogeneous row.")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError(f"{field_name} rotation is not orthonormal.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{field_name} rotation is not right-handed.")
    return matrix


def _camera_intrinsics(
    value: Mapping[str, Any],
) -> tuple[int, int, float, float, float, float]:
    required = {"width", "height", "fx", "fy", "cx", "cy"}
    if not required <= set(value):
        raise ValueError("intrinsics must contain width, height, fx, fy, cx, and cy.")
    width = value["width"]
    height = value["height"]
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, (int, np.integer))
        or not isinstance(height, (int, np.integer))
        or int(width) <= 0
        or int(height) <= 0
    ):
        raise ValueError("intrinsics width and height must be positive integers.")
    numeric = tuple(float(value[name]) for name in ("fx", "fy", "cx", "cy"))
    if not all(math.isfinite(item) for item in numeric):
        raise ValueError("intrinsics must be finite.")
    fx, fy, cx, cy = numeric
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("intrinsics focal lengths must be positive.")
    return int(width), int(height), fx, fy, cx, cy


@dataclass(frozen=True)
class DmtacGraspOverlayCandidate:
    """One grasp pose expressed in the caller-selected reference frame."""

    ordinal: int
    score: float
    gripper_width_m: float
    reference_from_tcp: tuple[float, ...]
    selected: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or int(self.ordinal) < 0:
            raise ValueError("ordinal must be a non-negative integer.")
        score = float(self.score)
        width = float(self.gripper_width_m)
        if not math.isfinite(score):
            raise ValueError("score must be finite.")
        if (
            not math.isfinite(width)
            or not 0.005 <= width <= DMTAC_M_GRASPGENX_MAX_OPENING_M
        ):
            raise ValueError(
                "gripper_width_m must be in "
                f"[0.005, {DMTAC_M_GRASPGENX_MAX_OPENING_M:.4f}]."
            )
        pose = _rigid_transform(
            self.reference_from_tcp, field_name="reference_from_tcp"
        )
        object.__setattr__(self, "ordinal", int(self.ordinal))
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "gripper_width_m", width)
        object.__setattr__(
            self,
            "reference_from_tcp",
            tuple(float(item) for item in pose.reshape(-1)),
        )

    @classmethod
    def from_pose(
        cls,
        *,
        ordinal: int,
        score: float,
        gripper_width_m: float,
        reference_from_tcp: Any,
        selected: bool = False,
    ) -> DmtacGraspOverlayCandidate:
        pose = np.asarray(reference_from_tcp, dtype=np.float64).reshape(-1)
        return cls(
            ordinal=ordinal,
            score=score,
            gripper_width_m=gripper_width_m,
            reference_from_tcp=tuple(float(item) for item in pose),
            selected=selected,
        )


@dataclass(frozen=True)
class ProjectedGraspSegment:
    kind: str
    start_uv: tuple[int, int]
    end_uv: tuple[int, int]


@dataclass(frozen=True)
class DmtacGraspProjection:
    ordinal: int
    score: float
    selected: bool
    visibility: str
    total_segment_count: int
    front_segment_count: int
    visible_segment_count: int
    clipped_segment_count: int
    tcp_pixel_uv: tuple[int, int] | None
    camera_from_tcp: tuple[float, ...]
    segments: tuple[ProjectedGraspSegment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "score": self.score,
            "selected": self.selected,
            "visibility": self.visibility,
            "total_segment_count": self.total_segment_count,
            "front_segment_count": self.front_segment_count,
            "visible_segment_count": self.visible_segment_count,
            "clipped_segment_count": self.clipped_segment_count,
            "tcp_pixel_uv": (
                None if self.tcp_pixel_uv is None else list(self.tcp_pixel_uv)
            ),
            "T_camera_from_tcp": list(self.camera_from_tcp),
        }


def camera_from_reference_poses(
    *, reference_from_tcp: Any, reference_from_camera: Any
) -> np.ndarray:
    """Resolve ``camera_from_tcp`` from two poses in one reference frame."""

    reference_tcp = _rigid_transform(
        reference_from_tcp, field_name="reference_from_tcp"
    )
    reference_camera = _rigid_transform(
        reference_from_camera, field_name="reference_from_camera"
    )
    camera_from_tcp = np.linalg.inv(reference_camera) @ reference_tcp
    return _rigid_transform(camera_from_tcp, field_name="camera_from_tcp")


def _box_segments(
    *,
    kind: str,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    vertices = [
        np.asarray((x, y, z), dtype=np.float64)
        for x in x_bounds
        for y in y_bounds
        for z in z_bounds
    ]
    segments: list[tuple[str, np.ndarray, np.ndarray]] = []
    for first in range(len(vertices)):
        for second in range(first + 1, len(vertices)):
            differences = np.count_nonzero(
                np.abs(vertices[first] - vertices[second]) > 1e-12
            )
            if differences == 1:
                segments.append((kind, vertices[first], vertices[second]))
    return segments


def _wireframe_segments(
    width_m: float,
    *,
    geometry: ParallelJawSolidGeometry | None = None,
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    # Geometry is expressed about the actual calibrated tool_tcp.  The palm
    # boundary is behind it; the fingertips extend slightly beyond it.
    geometry = geometry or reviewed_4b2_cad_solid_geometry()
    half_opening = width_m / 2.0
    finger_thickness = geometry.finger_thickness_m
    half_height = geometry.pad_height_m / 2.0
    palm_z = geometry.palm_z_m
    fingertip_z = palm_z + geometry.finger_length_m
    segments: list[tuple[str, np.ndarray, np.ndarray]] = []
    segments.extend(
        _box_segments(
            kind="left_finger",
            x_bounds=(-half_opening - finger_thickness, -half_opening),
            y_bounds=(-half_height, half_height),
            z_bounds=(palm_z, fingertip_z),
        )
    )
    segments.extend(
        _box_segments(
            kind="right_finger",
            x_bounds=(half_opening, half_opening + finger_thickness),
            y_bounds=(-half_height, half_height),
            z_bounds=(palm_z, fingertip_z),
        )
    )
    segments.extend(
        _box_segments(
            kind="palm",
            x_bounds=(
                -half_opening - finger_thickness,
                half_opening + finger_thickness,
            ),
            y_bounds=(-half_height, half_height),
            z_bounds=(palm_z - finger_thickness, palm_z),
        )
    )
    axis_x = max(0.035, half_opening + finger_thickness + 0.005)
    segments.extend(
        (
            ("axis_x", np.zeros(3), np.asarray((axis_x, 0.0, 0.0))),
            ("axis_y", np.zeros(3), np.asarray((0.0, 0.035, 0.0))),
            ("axis_z", np.zeros(3), np.asarray((0.0, 0.0, 0.060))),
        )
    )
    return tuple(segments)


def _clip_near_plane(
    first: np.ndarray, second: np.ndarray, *, near_clip_m: float
) -> tuple[np.ndarray, np.ndarray, bool] | None:
    first_front = bool(first[2] >= near_clip_m)
    second_front = bool(second[2] >= near_clip_m)
    if not first_front and not second_front:
        return None
    if first_front and second_front:
        return first, second, False
    fraction = (near_clip_m - float(first[2])) / float(second[2] - first[2])
    intersection = first + fraction * (second - first)
    if first_front:
        return first, intersection, True
    return intersection, second, True


def _clip_image_segment(
    first: tuple[float, float],
    second: tuple[float, float],
    *,
    width: int,
    height: int,
) -> tuple[tuple[float, float], tuple[float, float], bool] | None:
    # Liang-Barsky clipping against the inclusive image rectangle.
    x0, y0 = first
    x1, y1 = second
    dx = x1 - x0
    dy = y1 - y0
    lower = 0.0
    upper = 1.0
    for p, q in (
        (-dx, x0),
        (dx, float(width - 1) - x0),
        (-dy, y0),
        (dy, float(height - 1) - y0),
    ):
        if abs(p) <= 1e-15:
            if q < 0.0:
                return None
            continue
        ratio = q / p
        if p < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    clipped_first = (x0 + lower * dx, y0 + lower * dy)
    clipped_second = (x0 + upper * dx, y0 + upper * dy)
    clipped = lower > 0.0 or upper < 1.0
    return clipped_first, clipped_second, clipped


def project_dmtac_grasp(
    candidate: DmtacGraspOverlayCandidate,
    *,
    reference_from_camera: Any,
    intrinsics: Mapping[str, Any],
    near_clip_m: float = 0.005,
    solid_geometry: ParallelJawSolidGeometry | None = None,
) -> DmtacGraspProjection:
    """Project one DMTAC M wireframe and classify its image visibility."""

    width, height, fx, fy, cx, cy = _camera_intrinsics(intrinsics)
    near = float(near_clip_m)
    if not math.isfinite(near) or near <= 0.0:
        raise ValueError("near_clip_m must be finite and positive.")
    camera_from_tcp = camera_from_reference_poses(
        reference_from_tcp=candidate.reference_from_tcp,
        reference_from_camera=reference_from_camera,
    )
    local_segments = _wireframe_segments(
        candidate.gripper_width_m,
        geometry=solid_geometry,
    )
    projected: list[ProjectedGraspSegment] = []
    front_count = 0
    clipped_count = 0

    for kind, first_local, second_local in local_segments:
        # TCP axes retain the actual controlled frame. Only physical solids
        # receive the deployment-bound geometry-frame transform.
        body_pose = (camera_from_tcp if solid_geometry is None or kind.startswith("axis_")
                     else solid_geometry.solid_pose(camera_from_tcp))
        first_camera = body_pose[:3, :3] @ first_local + body_pose[:3, 3]
        second_camera = body_pose[:3, :3] @ second_local + body_pose[:3, 3]
        near_segment = _clip_near_plane(first_camera, second_camera, near_clip_m=near)
        if near_segment is None:
            continue
        front_count += 1
        first_camera, second_camera, near_clipped = near_segment
        first_uv = (
            fx * float(first_camera[0]) / float(first_camera[2]) + cx,
            fy * float(first_camera[1]) / float(first_camera[2]) + cy,
        )
        second_uv = (
            fx * float(second_camera[0]) / float(second_camera[2]) + cx,
            fy * float(second_camera[1]) / float(second_camera[2]) + cy,
        )
        image_segment = _clip_image_segment(
            first_uv, second_uv, width=width, height=height
        )
        if image_segment is None:
            if near_clipped:
                clipped_count += 1
            continue
        first_uv, second_uv, image_clipped = image_segment
        if near_clipped or image_clipped:
            clipped_count += 1

        def integer_pixel(value: tuple[float, float]) -> tuple[int, int]:
            return (
                min(width - 1, max(0, round(value[0]))),
                min(height - 1, max(0, round(value[1]))),
            )

        projected.append(
            ProjectedGraspSegment(
                kind=kind,
                start_uv=integer_pixel(first_uv),
                end_uv=integer_pixel(second_uv),
            )
        )

    tcp = camera_from_tcp[:3, 3]
    tcp_pixel: tuple[int, int] | None = None
    if float(tcp[2]) >= near:
        tcp_u = fx * float(tcp[0]) / float(tcp[2]) + cx
        tcp_v = fy * float(tcp[1]) / float(tcp[2]) + cy
        if 0.0 <= tcp_u <= width - 1 and 0.0 <= tcp_v <= height - 1:
            tcp_pixel = (round(tcp_u), round(tcp_v))

    total_count = len(local_segments)
    visible_count = len(projected)
    if front_count == 0:
        visibility = "behind_camera"
    elif visible_count == 0:
        visibility = "out_of_frame"
    elif visible_count == total_count and clipped_count == 0:
        visibility = "visible"
    else:
        visibility = "partially_visible"
    return DmtacGraspProjection(
        ordinal=candidate.ordinal,
        score=candidate.score,
        selected=candidate.selected,
        visibility=visibility,
        total_segment_count=total_count,
        front_segment_count=front_count,
        visible_segment_count=visible_count,
        clipped_segment_count=clipped_count,
        tcp_pixel_uv=tcp_pixel,
        camera_from_tcp=tuple(float(item) for item in camera_from_tcp.reshape(-1)),
        segments=tuple(projected),
    )


def render_dmtac_grasp_overlays(
    image: Image.Image | np.ndarray,
    *,
    candidates: Sequence[DmtacGraspOverlayCandidate],
    reference_from_camera: Any,
    intrinsics: Mapping[str, Any],
    target_pixel_uv: tuple[int, int] | None = None,
    target_bbox_xyxy: tuple[int, int, int, int] | None = None,
    solid_geometry: ParallelJawSolidGeometry | None = None,
) -> tuple[Image.Image, tuple[DmtacGraspProjection, ...]]:
    """Overlay ranked candidates and return machine-readable projection records."""

    width, height, _fx, _fy, _cx, _cy = _camera_intrinsics(intrinsics)
    if isinstance(image, Image.Image):
        output = image.convert("RGB").copy()
    else:
        pixels = np.asarray(image)
        if pixels.shape != (height, width, 3) or pixels.dtype != np.uint8:
            raise ValueError("image array must be uint8 HxWx3 matching intrinsics.")
        output = Image.fromarray(pixels, mode="RGB")
    if output.size != (width, height):
        raise ValueError("image dimensions differ from intrinsics.")
    camera_pose = _rigid_transform(
        reference_from_camera, field_name="reference_from_camera"
    )
    projections = tuple(
        project_dmtac_grasp(
            candidate,
            reference_from_camera=camera_pose,
            intrinsics=intrinsics,
            solid_geometry=solid_geometry,
        )
        for candidate in candidates
    )
    draw = ImageDraw.Draw(output)
    if target_bbox_xyxy is not None:
        if len(target_bbox_xyxy) != 4:
            raise ValueError("target_bbox_xyxy must contain four values.")
        x0, y0, x1, y1 = (int(value) for value in target_bbox_xyxy)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(64, 220, 255), width=2)
    if target_pixel_uv is not None:
        u, v = (int(value) for value in target_pixel_uv)
        draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=(255, 255, 0), width=3)

    secondary_colors = (
        (255, 160, 32),
        (96, 235, 160),
        (96, 190, 255),
        (255, 224, 96),
    )

    def body_color(projection: DmtacGraspProjection) -> tuple[int, int, int]:
        if projection.selected:
            return (255, 64, 255)
        return secondary_colors[projection.ordinal % len(secondary_colors)]

    axis_colors = {
        "axis_x": (255, 48, 48),
        "axis_y": (48, 230, 48),
        "axis_z": (64, 128, 255),
    }
    for projection in reversed(projections):
        line_width = 4 if projection.selected else 2
        for segment in projection.segments:
            if segment.kind in axis_colors and not projection.selected:
                continue
            color = axis_colors.get(segment.kind, body_color(projection))
            draw.line(
                (*segment.start_uv, *segment.end_uv),
                fill=color,
                width=line_width,
            )
        if projection.tcp_pixel_uv is not None:
            u, v = projection.tcp_pixel_uv
            color = body_color(projection)
            draw.ellipse((u - 3, v - 3, u + 3, v + 3), outline=color, width=2)
            draw.text((u + 5, v + 4), f"#{projection.ordinal}", fill=color)

    # Always list every candidate, including ones that could not be drawn.
    row_height = 14
    legend_height = max(row_height, row_height * len(projections))
    legend_top = max(0, height - legend_height - 1)
    draw.rectangle((0, legend_top, min(width - 1, 300), height - 1), fill=(0, 0, 0))
    for row, projection in enumerate(projections):
        prefix = "BEST" if projection.selected else "cand"
        color = body_color(projection)
        draw.text(
            (4, legend_top + row * row_height + 2),
            (
                f"{prefix} #{projection.ordinal} score={projection.score:.3f} "
                f"{projection.visibility} "
                f"{projection.visible_segment_count}/{projection.total_segment_count}"
            ),
            fill=color,
        )
    return output, projections


__all__ = [
    "DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID",
    "DMTAC_M_GRASPGENX_FINGER_LENGTH_M",
    "DMTAC_M_GRASPGENX_FINGER_THICKNESS_M",
    "DMTAC_M_GRASPGENX_GRASP_CENTER_FROM_TACTILE_XYZ_M",
    "DMTAC_M_GRASPGENX_GRASP_CENTER_FROM_TACTILE_Z_M",
    "DMTAC_M_GRASPGENX_MAX_OPENING_M",
    "DMTAC_M_GRASPGENX_PAD_HEIGHT_M",
    "DMTAC_M_GRASPGENX_PALM_Z_M",
    "DMTAC_M_GRASPGENX_TOOL_FROM_MODEL_ROTATION",
    "DMTAC_M_GRASPGENX_TOOL_FROM_MODEL_XYZ_M",
    "GRASP_VISIBILITY_STATES",
    "DmtacGraspOverlayCandidate",
    "DmtacGraspProjection",
    "ProjectedGraspSegment",
    "camera_from_reference_poses",
    "project_dmtac_grasp",
    "render_dmtac_grasp_overlays",
    "reviewed_graspgenx_dmtac_solid_geometry",
    "reviewed_4b2_cad_solid_geometry",
]
