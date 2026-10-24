"""Offline GUI for aligning the GraspGenX model with the real DMTAC end tool.

The viewer intentionally keeps three different frames visible:

* the measured RM65 arm endpoint and the URDF flange;
* the controller TCP and the nominal GraspGenX model frame;
* the DMTAC sensing pair and the derived tactile grasp frame.

It consumes only a URDF and user supplied joint values.  It never opens a
robot connection and never changes a production grasp profile.  Save produces
an independent alignment JSON that can be reviewed before it is wired into a
runtime profile.
"""

from __future__ import annotations

import argparse
import json
import math
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Iterable, Sequence

from deployments.perception.dmtac_grasp_overlay import (
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID,
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_IDS,
    DMTAC_M_GRASPGENX_GRASP_CENTER_FROM_TACTILE_XYZ_M,
    dmtac_graspgenx_alignment_profile,
)

from .gripper_urdf_editor.core import (
    Mat4,
    URDFDocument,
    URDFValidationError,
    identity,
    invert_rigid,
    iter_geometry,
    matmul,
    origin_matrix,
    transform_point,
    transform_vector,
)
from .gripper_urdf_editor.mesh import MeshCache, box_triangles
from .gripper_urdf_editor.viewer import GripperCanvas, SceneMarker, SceneShape


SCHEMA = "dexteleop.graspgenx_dmtac_alignment.v4"
DEFAULT_TCP_Z_M = 0.140
DEFAULT_MODEL_FROM_TCP_Z_M = 0.042999
DEFAULT_OPENING_M = 0.070
# The GraspGenX model frame is anchored at the nominal palm boundary.  The
# palm is therefore z=0 in that frame; the corresponding TCP-local palm
# coordinate is derived from model_from_tcp_z_m below.  Keeping these two
# coordinates separate prevents the old GUI from applying -42.999 mm twice.
DEFAULT_PALM_Z_M = 0.0
DEFAULT_FINGER_LENGTH_M = 0.050
DEFAULT_FINGER_THICKNESS_M = 0.010
DEFAULT_PAD_HEIGHT_M = 0.0255
DEFAULT_PROFILE_ID = "DMTAC_M"
# The two visual models intentionally have separate parameter sets.  The
# ``model_*`` fields below are the DMTAC-aligned model; these constants are the
# legacy/original 4B2 production defaults.
DEFAULT_PRODUCTION_MODEL_FROM_TCP_Z_M = DEFAULT_MODEL_FROM_TCP_Z_M
DEFAULT_PRODUCTION_OPENING_M = DEFAULT_OPENING_M
DEFAULT_PRODUCTION_PALM_Z_M = DEFAULT_PALM_Z_M
DEFAULT_PRODUCTION_FINGER_LENGTH_M = DEFAULT_FINGER_LENGTH_M
DEFAULT_PRODUCTION_FINGER_THICKNESS_M = DEFAULT_FINGER_THICKNESS_M
DEFAULT_PRODUCTION_PAD_HEIGHT_M = DEFAULT_PAD_HEIGHT_M
# These values reproduce the *current production* 4B2 proxy conversion.  They
# are not the transform from the controller TCP to the DMTAC sensing frames.
# In the reviewed URDF the same 24.5/8.2 mm numbers belong to the RM65 endpoint
# -> installed 4B2 chain.  Production currently re-expresses them as a virtual
# solid frame so its short parallel-jaw proxy lands on the old 4B2 fingers.
DMTAC_M_TOOL_FROM_SOLID_XYZ_M = (0.0245, -8.9993525e-8, 0.0082)
DMTAC_M_SOLID_YAW_RAD = math.pi / 2.0
# The visually reviewed grasp centre is 20 mm behind the midpoint of the two
# DMTAC sensing frames.  This is an object/contact target, not the GraspGenX
# gripper-base frame.  GraspGenX emits the gripper base and its nominal proxy
# extends along local +Z, so the base must additionally sit half a proxy depth
# behind this centre.
DEFAULT_GRASP_CENTER_FROM_TACTILE_XYZ_M = (0.0, 0.0, -0.020)
DEFAULT_GRASP_CENTER_FROM_TACTILE_Z_M = DEFAULT_GRASP_CENTER_FROM_TACTILE_XYZ_M[2]
DEFAULT_MODEL_CORRECTION_M = (0.0, 0.0, 0.0)
AXIS_LENGTH_M = 0.030
AXIS_RADIUS_M = 0.0011


def _finite(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _vec3(value: object, default: Sequence[float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return tuple(float(v) for v in default)  # type: ignore[return-value]
    return tuple(_finite(v, float(default[i])) for i, v in enumerate(value))  # type: ignore[return-value]


def _mat4_translation(matrix: Mat4) -> tuple[float, float, float]:
    return (matrix[0][3], matrix[1][3], matrix[2][3])


def _rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> Mat4:
    return origin_matrix((0.0, 0.0, 0.0), (roll, pitch, yaw))


def _pose6_matrix(pose: Sequence[float]) -> Mat4:
    values = tuple(float(value) for value in pose)
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError("tool TCP pose must contain six finite xyz/rpy values")
    return origin_matrix(values[:3], values[3:])


def load_tool_tcp_pose(profile: Path) -> tuple[float, ...]:
    """Load the same endpoint-relative TCP pose consumed by the real backend."""

    payload = json.loads(profile.read_text(encoding="utf-8"))
    tcp = payload.get("tcp") if isinstance(payload, dict) else None
    pose = tcp.get("pose_flange_m_rad") if isinstance(tcp, dict) else None
    if not isinstance(pose, list) or len(pose) != 6:
        raise ValueError(f"tool profile has no six-value tcp.pose_flange_m_rad: {profile}")
    values = tuple(float(value) for value in pose)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"tool profile tcp.pose_flange_m_rad is not finite: {profile}")
    return values


def _axis_box(frame: Mat4, axis: int, length: float = AXIS_LENGTH_M) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    """Return a thin box aligned with one local frame axis."""
    local = matmul(
        origin_matrix((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        origin_matrix(
            ((length / 2.0) if axis == 0 else 0.0,
             (length / 2.0) if axis == 1 else 0.0,
             (length / 2.0) if axis == 2 else 0.0),
            (0.0, 0.0, 0.0),
        ),
    )
    dimensions = [2.0 * AXIS_RADIUS_M] * 3
    dimensions[axis] = length
    # box_triangles is centered on its local origin, so translate its center.
    center = [0.0, 0.0, 0.0]
    center[axis] = length / 2.0
    local = origin_matrix(center, (0.0, 0.0, 0.0))
    geometry = matmul(frame, local)
    return tuple(
        tuple(tuple(transform_point(geometry, vertex) for vertex in triangle) for triangle in box_triangles(dimensions))
    )


def _transform_triangles(matrix: Mat4, triangles: Iterable[Sequence[Sequence[float]]]) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    return tuple(
        tuple(transform_point(matrix, vertex) for vertex in triangle)
        for triangle in triangles
    )


def _tool_palm_z_m(
    state: "AlignmentState",
    *,
    model_from_tcp_z_m: float | None = None,
    model_palm_z_m: float | None = None,
) -> float:
    """Return the palm boundary in the controller tool_tcp frame.

    ``graspgenx_model`` is anchored at the palm boundary, while
    ``model_from_tcp_z_m`` is the distance from TCP to that anchor.  A local
    model palm adjustment consequently maps to this TCP-local coordinate.
    """
    from_tcp = state.model_from_tcp_z_m if model_from_tcp_z_m is None else model_from_tcp_z_m
    palm = state.model_palm_z_m if model_palm_z_m is None else model_palm_z_m
    return float(palm - from_tcp)


def _state_tool_tcp_pose(state: "AlignmentState") -> tuple[float, ...]:
    values = list(state.tool_tcp_pose_endpoint_m_rad or ())
    if len(values) != 6:
        values = [0.0, 0.0, state.tcp_z_m, 0.0, 0.0, 0.0]
    values[2] = float(state.tcp_z_m)
    return tuple(float(value) for value in values)


def _production_solid_adapter() -> Mat4:
    """Return the reviewed neutral-solid -> physical-DMTAC transform."""
    return origin_matrix(
        DMTAC_M_TOOL_FROM_SOLID_XYZ_M,
        (0.0, 0.0, DMTAC_M_SOLID_YAW_RAD),
    )


def _parallel_jaw_triangles(
    frame: Mat4,
    state: "AlignmentState",
    *,
    palm_z_m: float,
    opening_m: float | None = None,
    finger_length_m: float | None = None,
    finger_thickness_m: float | None = None,
    pad_height_m: float | None = None,
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    """Build the same palm/finger proxy topology used by production checks.

    ``box_triangles`` is centered at its origin.  The finger box is therefore
    centered half a finger length after the palm boundary, and the palm box is
    centered half a thickness behind it.  This keeps the visual proxy's
    extents identical to the analytical solid instead of centering a 50 mm
    finger around the palm plane.
    """
    opening = max(0.001, float(state.model_opening_m if opening_m is None else opening_m))
    finger_length = float(state.model_finger_length_m if finger_length_m is None else finger_length_m)
    finger_thickness = float(
        state.model_finger_thickness_m if finger_thickness_m is None else finger_thickness_m
    )
    pad_height = float(state.model_pad_height_m if pad_height_m is None else pad_height_m)
    triangles: list[tuple[tuple[float, float, float], ...]] = []
    finger_mesh = box_triangles((finger_thickness, pad_height, finger_length))
    finger_center_z = palm_z_m + finger_length / 2.0
    for sign in (-1.0, 1.0):
        center = origin_matrix(
            (sign * opening / 2.0, 0.0, finger_center_z),
            (0.0, 0.0, 0.0),
        )
        triangles.extend(_transform_triangles(matmul(frame, center), finger_mesh))
    palm_mesh = box_triangles(
        (opening + 2.0 * finger_thickness, pad_height, finger_thickness)
    )
    palm_center = origin_matrix(
        (0.0, 0.0, palm_z_m - finger_thickness / 2.0),
        (0.0, 0.0, 0.0),
    )
    triangles.extend(_transform_triangles(matmul(frame, palm_center), palm_mesh))
    return tuple(triangles)


def _frame_angle(a: Mat4, b: Mat4) -> float:
    relative = matmul(invert_rigid(a), b)
    trace = relative[0][0] + relative[1][1] + relative[2][2]
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))


def _side_prefix(side: str) -> str:
    return "left" if side == "left" else "right"


def _link_contains(link: str, *tokens: str) -> bool:
    lowered = link.lower()
    return any(token in lowered for token in tokens)


@dataclass
class AlignmentState:
    side: str = "right"
    arm_qpos: list[float] | None = None
    gripper_q: float = 0.0
    tcp_z_m: float = DEFAULT_TCP_Z_M
    tool_tcp_pose_endpoint_m_rad: list[float] | None = None
    model_from_tcp_z_m: float = DEFAULT_MODEL_FROM_TCP_Z_M
    model_opening_m: float = DEFAULT_OPENING_M
    model_palm_z_m: float = DEFAULT_PALM_Z_M
    model_finger_length_m: float = DEFAULT_FINGER_LENGTH_M
    model_finger_thickness_m: float = DEFAULT_FINGER_THICKNESS_M
    model_pad_height_m: float = DEFAULT_PAD_HEIGHT_M
    grasp_center_from_tactile_x_m: float = DEFAULT_GRASP_CENTER_FROM_TACTILE_XYZ_M[0]
    grasp_center_from_tactile_y_m: float = DEFAULT_GRASP_CENTER_FROM_TACTILE_XYZ_M[1]
    grasp_center_from_tactile_z_m: float = DEFAULT_GRASP_CENTER_FROM_TACTILE_Z_M
    production_model_from_tcp_z_m: float = DEFAULT_PRODUCTION_MODEL_FROM_TCP_Z_M
    production_opening_m: float = DEFAULT_PRODUCTION_OPENING_M
    production_palm_z_m: float = DEFAULT_PRODUCTION_PALM_Z_M
    production_finger_length_m: float = DEFAULT_PRODUCTION_FINGER_LENGTH_M
    production_finger_thickness_m: float = DEFAULT_PRODUCTION_FINGER_THICKNESS_M
    production_pad_height_m: float = DEFAULT_PRODUCTION_PAD_HEIGHT_M
    model_correction_xyz_m: list[float] | None = None
    model_correction_rpy_rad: list[float] | None = None
    profile_id: str = DEFAULT_PROFILE_ID

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("Alignment side must be left or right.")
        if self.arm_qpos is None:
            self.arm_qpos = [0.0] * 6
        if self.tool_tcp_pose_endpoint_m_rad is None:
            self.tool_tcp_pose_endpoint_m_rad = [0.0, 0.0, self.tcp_z_m, 0.0, 0.0, 0.0]
        if self.model_correction_xyz_m is None:
            self.model_correction_xyz_m = list(DEFAULT_MODEL_CORRECTION_M)
        if self.model_correction_rpy_rad is None:
            self.model_correction_rpy_rad = [0.0, 0.0, 0.0]


def current_dmtac_state(side: str, tcp_pose: Sequence[float], arm_qpos: Sequence[float], *,
    alignment_profile_id: str = DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID) -> AlignmentState:
    """Read selected nominal geometry without changing hardware calibration."""
    profile = dmtac_graspgenx_alignment_profile(alignment_profile_id)
    _pose6_matrix(tcp_pose)
    if len(arm_qpos) != 6 or not all(math.isfinite(value) for value in arm_qpos):
        raise ValueError("Arm preview requires six finite joint values.")
    geometry = profile.nominal_geometry
    center = DMTAC_M_GRASPGENX_GRASP_CENTER_FROM_TACTILE_XYZ_M
    return AlignmentState(side=side, arm_qpos=list(arm_qpos), tcp_z_m=float(tcp_pose[2]),
        tool_tcp_pose_endpoint_m_rad=list(tcp_pose), model_opening_m=geometry.maximum_opening_m,
        model_palm_z_m=geometry.palm_z_m, model_finger_length_m=geometry.finger_length_m,
        model_finger_thickness_m=geometry.finger_thickness_m, model_pad_height_m=geometry.pad_height_m,
        grasp_center_from_tactile_x_m=center[0], grasp_center_from_tactile_y_m=center[1],
        grasp_center_from_tactile_z_m=center[2])


class AlignmentModel:
    def __init__(self, urdf: Path, state: AlignmentState):
        self.document = URDFDocument.load(urdf)
        self.state = state
        self.mesh_cache = MeshCache(max_triangles=1000)
        self.warnings: list[str] = []
        self._pair = None
        self._pair_endpoints: tuple[object, object] | None = None
        self._refresh_pair()

    def _refresh_pair(self) -> None:
        prefix = _side_prefix(self.state.side)
        pairs = [pair for pair in self.document.detect_gripper_pairs() if prefix in pair.label.lower()]
        if pairs:
            self._pair = pairs[0]
            try:
                self._pair_endpoints = (
                    self.document.default_endpoint(self._pair.joint_a),
                    self.document.default_endpoint(self._pair.joint_b),
                )
            except URDFValidationError:
                self._pair_endpoints = None

    def _link(self, candidates: Sequence[str], fallback_tokens: Sequence[str]) -> str | None:
        for candidate in candidates:
            if candidate in self.document.links:
                return candidate
        prefix = _side_prefix(self.state.side)
        ranked: list[tuple[int, str]] = []
        for name in self.document.links:
            low = name.lower()
            if prefix not in low:
                continue
            score = sum(10 for token in fallback_tokens if token in low)
            if score:
                ranked.append((score, name))
        return max(ranked, default=(0, ""))[1] or None

    def _arm_joint_positions(self) -> dict[str, float]:
        prefix = _side_prefix(self.state.side)
        positions: dict[str, float] = {}
        for index, value in enumerate(self.state.arm_qpos or [0.0] * 6, start=1):
            name = f"{prefix}_arm_joint{index}"
            if name in self.document.joints:
                positions[name] = float(value)
        if self._pair is not None:
            positions.update(self.document.pair_joint_positions(self._pair, self.state.gripper_q))
        return positions

    def _transforms(self) -> dict[str, Mat4]:
        return self.document.link_transforms(self._arm_joint_positions())

    def frame_links(self) -> dict[str, str | None]:
        prefix = _side_prefix(self.state.side)
        return {
            "rm65_arm_endpoint": self._link(
                (f"{prefix}_arm_link6", f"{prefix}_wrist", f"{prefix}_arm_end"),
                ("arm_link6", "wrist", "arm_end"),
            ),
            "real_flange": self._link(
                (
                    f"{prefix}_realman_to_inspire_gripper_flange",
                    f"{prefix}_realman_to_dmtac_flange",
                    f"{prefix}_flange",
                ),
                ("realman_to", "flange"),
            ),
        }

    def _world_from_link(self, name: str | None, transforms: dict[str, Mat4]) -> Mat4:
        return identity() if not name else transforms.get(name, identity())

    def _candidate_sensor_links(self, transforms: dict[str, Mat4]) -> list[str]:
        prefix = _side_prefix(self.state.side)
        scored: list[tuple[int, str]] = []
        for name in self.document.links:
            low = name.lower()
            if prefix not in low or name not in transforms:
                continue
            if any(token in low for token in ("camera", "realsense", "arm_link", "flange", "wrist")):
                continue
            score = 0
            for token, value in (
                ("sensing_frame", 55), ("sensing", 40), ("dmtac", 30),
                ("tactile", 28), ("w2m", 25), ("sensor", 18),
                ("inner_contact", 17), ("contact", 15), ("fingertip", 10),
                ("finger_tip", 10), ("pad", 8), ("tip", 6),
            ):
                if token in low:
                    score += value
            if score:
                scored.append((score, name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [name for _score, name in scored]

    def _tactile_frame(self, transforms: dict[str, Mat4]) -> tuple[Mat4, list[str]]:
        links = self._candidate_sensor_links(transforms)
        notes: list[str] = []
        if len(links) < 2 and self._pair_endpoints is not None:
            links = [self._pair_endpoints[0].link_name, self._pair_endpoints[1].link_name]  # type: ignore[attr-defined]
            notes.append("No explicit DMTAC/tactile pair found; using detected gripper endpoints.")
        if len(links) < 2:
            notes.append("URDF does not expose two tactile links; DMTAC frame is a diagnostic fallback.")
            fallback = self._world_from_link(self.frame_links()["real_flange"], transforms)
            return fallback, notes
        links = links[:2]
        ta, tb = transforms[links[0]], transforms[links[1]]
        pa, pb = _mat4_translation(ta), _mat4_translation(tb)
        x = tuple(pb[i] - pa[i] for i in range(3))
        length = math.sqrt(sum(value * value for value in x))
        if length <= 1e-9:
            notes.append("Tactile pair has zero separation; using flange X axis.")
            x = transform_vector(self._world_from_link(self.frame_links()["real_flange"], transforms), (1.0, 0.0, 0.0))
            length = math.sqrt(sum(value * value for value in x))
        x = tuple(value / max(length, 1e-9) for value in x)
        za = transform_vector(ta, (0.0, 1.0, 0.0))
        zb = transform_vector(tb, (0.0, 1.0, 0.0))
        z0 = tuple(za[i] + zb[i] for i in range(3))
        z_norm = math.sqrt(sum(value * value for value in z0))
        if z_norm <= 1e-9:
            z0 = (0.0, 0.0, 1.0)
            z_norm = 1.0
        z0 = tuple(value / z_norm for value in z0)
        y = (
            z0[1] * x[2] - z0[2] * x[1],
            z0[2] * x[0] - z0[0] * x[2],
            z0[0] * x[1] - z0[1] * x[0],
        )
        y_norm = math.sqrt(sum(value * value for value in y))
        if y_norm <= 1e-9:
            y = (0.0, 1.0, 0.0)
            y_norm = 1.0
        y = tuple(value / y_norm for value in y)
        z = (
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        )
        origin = tuple((pa[i] + pb[i]) / 2.0 for i in range(3))
        # Build the tactile basis as columns in world coordinates.
        tactile = (
            (x[0], y[0], z[0], origin[0]),
            (x[1], y[1], z[1], origin[1]),
            (x[2], y[2], z[2], origin[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
        return tactile, notes

    def frames(self) -> tuple[dict[str, Mat4], dict[str, object]]:
        transforms = self._transforms()
        links = self.frame_links()
        endpoint = self._world_from_link(links["rm65_arm_endpoint"], transforms)
        flange = self._world_from_link(links["real_flange"], transforms)
        # Real RM65 planning applies tcp.pose_flange_m_rad to the solver
        # endpoint (arm_link6), not to the rotated visual installation flange.
        # Applying +140 mm to the latter flips the TCP to the opposite side of
        # the flange, which was the main error in the previous GUI.
        tcp_pose = _state_tool_tcp_pose(self.state)
        tcp = matmul(endpoint, _pose6_matrix(tcp_pose))

        # Reproduce the current production GraspGenX -> controller conversion
        # exactly.  This is deliberately kept separate from the DMTAC tactile
        # frame because the production proxy still describes the old short 4B2
        # fingers rather than the distal DMTAC sensing components.
        production_solid = matmul(tcp, _production_solid_adapter())
        production_model = matmul(
            production_solid,
            origin_matrix(
                (0.0, 0.0, -self.state.production_model_from_tcp_z_m),
                (0.0, 0.0, 0.0),
            ),
        )

        tactile, notes = self._tactile_frame(transforms)
        grasp_center = matmul(
            tactile,
            origin_matrix(
                (
                    self.state.grasp_center_from_tactile_x_m,
                    self.state.grasp_center_from_tactile_y_m,
                    self.state.grasp_center_from_tactile_z_m,
                ),
                (0.0, 0.0, 0.0),
            ),
        )
        nominal_model = matmul(
            grasp_center,
            origin_matrix(
                (0.0, 0.0, -self.state.model_finger_length_m / 2.0),
                (0.0, 0.0, 0.0),
            ),
        )
        correction = origin_matrix(
            self.state.model_correction_xyz_m or (0.0, 0.0, 0.0),
            self.state.model_correction_rpy_rad or (0.0, 0.0, 0.0),
        )
        # GraspGenX emits a gripper-base frame.  The proxy centre is half a
        # finger depth in front of that base, while the reviewed DMTAC grasp
        # centre is a separate contact/object target.  Keeping the frames
        # explicit avoids the previous 25 mm semantic error and makes the
        # direction of local +Z visible.
        model = matmul(nominal_model, correction)
        sweep_center = matmul(
            model,
            origin_matrix(
                (0.0, 0.0, self.state.model_finger_length_m / 2.0),
                (0.0, 0.0, 0.0),
            ),
        )
        sensor_links = self._candidate_sensor_links(transforms)[:2]
        frames = {
            "rm65_arm_endpoint": endpoint,
            "real_flange": flange,
            "tool_tcp": tcp,
            "dmtac_grasp_center": grasp_center,
            "graspgenx_model": model,
            "graspgenx_sweep_center": sweep_center,
            "production_graspgenx_model": production_model,
            "production_solid": production_solid,
            "dmtac_tactile": tactile,
        }
        for index, link in enumerate(sensor_links):
            frames[f"dmtac_sensing_{index}"] = transforms[link]
        details = {
            "transforms": transforms,
            "sensor_links": sensor_links,
            "warnings": notes,
            "sensor_width_m": (
                math.dist(_mat4_translation(transforms[sensor_links[0]]), _mat4_translation(transforms[sensor_links[1]]))
                if len(sensor_links) == 2 else None
            ),
        }
        return frames, details

    def scene(self, flags: dict[str, bool]) -> tuple[list[SceneShape], list[SceneMarker], dict[str, Mat4], dict[str, object]]:
        frames, details = self.frames()
        transforms = details["transforms"]  # type: ignore[assignment]
        shapes: list[SceneShape] = []
        markers: list[SceneMarker] = []
        prefix = _side_prefix(self.state.side)
        side_links = {name for name, _depth in self.document.subtree_links(f"{prefix}_arm_base_link")} if f"{prefix}_arm_base_link" in self.document.links else set(self.document.links)
        for ref in iter_geometry(self.document, side_links):
            if ref.role == "visual" and not flags.get("visual", True):
                continue
            if ref.role == "collision" and not flags.get("collision", False):
                continue
            try:
                triangles = self.mesh_cache.geometry_triangles(self.document, ref)
            except URDFValidationError as exc:
                self.warnings.append(str(exc))
                continue
            if not triangles:
                continue
            low = ref.link_name.lower()
            if "flange" in low:
                color, outline = "#f3a43b", "#ffe0a6"
            elif any(token in low for token in ("dmtac", "tactile", "w2m", "sensor", "contact")):
                color, outline = "#18b9c8", "#b4ffff"
            elif "cad" in low or "gripper" in low or "hand" in low:
                color, outline = "#77869d", "#d8e0eb"
            elif "arm_link" in low:
                color, outline = "#58677c", "#aebcd1"
            else:
                color, outline = "#3c4658", "#8b98ab"
            if ref.role == "collision":
                outline = "#33e28c"
            geometry = self.document.geometry_transform(ref, transforms)
            shapes.append(SceneShape(_transform_triangles(geometry, triangles), color, outline, ref.role == "collision"))

        if flags.get("model", True):
            model = frames["graspgenx_model"]
            shapes.append(
                SceneShape(
                    _parallel_jaw_triangles(
                        model,
                        self.state,
                        palm_z_m=self.state.model_palm_z_m,
                    ),
                    "#d68638",
                    "#ffd78a",
                    True,
                )
            )
        if flags.get("output", True):
            shapes.append(
                SceneShape(
                    _parallel_jaw_triangles(
                        frames["production_solid"],
                        self.state,
                        palm_z_m=_tool_palm_z_m(
                            self.state,
                            model_from_tcp_z_m=self.state.production_model_from_tcp_z_m,
                            model_palm_z_m=self.state.production_palm_z_m,
                        ),
                        opening_m=self.state.production_opening_m,
                        finger_length_m=self.state.production_finger_length_m,
                        finger_thickness_m=self.state.production_finger_thickness_m,
                        pad_height_m=self.state.production_pad_height_m,
                    ),
                    "#e7b63e",
                    "#fff0a6",
                    True,
                )
            )

        frame_colors = {
            "rm65_arm_endpoint": "#e8e8e8",
            "real_flange": "#ff9d38",
            "tool_tcp": "#f04f9b",
            "dmtac_grasp_center": "#a6ef5f",
            "graspgenx_model": "#d68638",
            "graspgenx_sweep_center": "#f1c75b",
            "production_graspgenx_model": "#ff6b57",
            "production_solid": "#f4cf38",
            "dmtac_tactile": "#14cbd2",
        }
        frame_labels = {
            "rm65_arm_endpoint": "RM65 endpoint",
            "real_flange": "real flange",
            "tool_tcp": "TCP",
            "dmtac_grasp_center": "reviewed grasp center",
            "graspgenx_model": "DMTAC-aligned GraspGenX",
            "graspgenx_sweep_center": "GraspGenX sweep center",
            "production_graspgenx_model": "current production GraspGenX",
            "production_solid": "current production 4B2 proxy",
            "dmtac_tactile": "DMTAC",
        }
        for index in range(len(details.get("sensor_links", []))):
            frame_colors[f"dmtac_sensing_{index}"] = "#57e7ed"
            frame_labels[f"dmtac_sensing_{index}"] = f"DMTAC sensing {index + 1}"
        if flags.get("frames", True):
            for name, frame in frames.items():
                if name in {"graspgenx_model", "graspgenx_sweep_center"} and not flags.get("model", True):
                    continue
                if name in {"production_graspgenx_model", "production_solid"} and not flags.get("output", True):
                    continue
                color = frame_colors[name]
                for axis in range(3):
                    axis_color = ("#f05c5c", "#58d875", "#5e9eff")[axis]
                    shapes.append(SceneShape(_axis_box(frame, axis), axis_color, color, True))
                markers.append(SceneMarker(_mat4_translation(frame), color, frame_labels[name]))
        return shapes, markers, frames, details

    def payload(self) -> dict[str, object]:
        frames, details = self.frames()
        return {
            "schema": SCHEMA,
            "profile_id": self.state.profile_id,
            "source_urdf": str(self.document.source_path),
            "side": self.state.side,
            "frame_convention": {
                "tcp": "world_from_rm65_solver_endpoint * endpoint_from_tcp_profile",
                "current_production_graspgenx_model": "world_from_tcp * current_tool_from_solid * translate(0,0,-production_model_from_tcp_z)",
                "dmtac_grasp_center": "world_from_dmtac_tactile * translate(grasp_center_from_tactile_xyz)",
                "graspgenx_model": "world_from_dmtac_grasp_center * translate(0,0,-finger_length/2) * local_alignment_correction",
                "graspgenx_sweep_center": "world_from_graspgenx_model * translate(0,0,finger_length/2)",
                "production_solid": "world_from_tcp * current_tool_from_solid; legacy 4B2 proxy only",
                "dmtac_tactile": "midpoint of selected sensing links; X closes left-to-right",
                "production_effect": "diagnostic-only; does not modify production profile or hardware",
            },
            "coordinate_contract": {
                "version": "dmtac_tactile_alignment_v4",
                "model_frame_anchor": "gripper_base_behind_reviewed_grasp_center",
                "model_palm_z_m": self.state.model_palm_z_m,
                "grasp_center_from_tactile_xyz_m": [
                    self.state.grasp_center_from_tactile_x_m,
                    self.state.grasp_center_from_tactile_y_m,
                    self.state.grasp_center_from_tactile_z_m,
                ],
                "grasp_center_from_tactile_z_m": self.state.grasp_center_from_tactile_z_m,
                "tool_palm_from_tcp_z_m": _tool_palm_z_m(self.state),
                "tool_from_solid_xyz_m": list(DMTAC_M_TOOL_FROM_SOLID_XYZ_M),
                "solid_yaw_rad": DMTAC_M_SOLID_YAW_RAD,
            },
            "arm_qpos_rad": list(self.state.arm_qpos or []),
            "dmtac": {
                "master_q": self.state.gripper_q,
                "selected_sensor_links": list(details["sensor_links"]),
                "sensing_width_m": details["sensor_width_m"],
            },
            "graspgenx_model": {
                "opening_m": self.state.model_opening_m,
                "palm_z_m": self.state.model_palm_z_m,
                "tool_palm_from_tcp_z_m": _tool_palm_z_m(self.state),
                "finger_length_m": self.state.model_finger_length_m,
                "finger_thickness_m": self.state.model_finger_thickness_m,
                "pad_height_m": self.state.model_pad_height_m,
                "model_from_tcp_z_m": self.state.model_from_tcp_z_m,
                "grasp_center_from_tactile_xyz_m": [
                    self.state.grasp_center_from_tactile_x_m,
                    self.state.grasp_center_from_tactile_y_m,
                    self.state.grasp_center_from_tactile_z_m,
                ],
                "grasp_center_from_tactile_z_m": self.state.grasp_center_from_tactile_z_m,
                "correction_xyz_m": list(self.state.model_correction_xyz_m or []),
                "correction_rpy_rad": list(self.state.model_correction_rpy_rad or []),
            },
            "original_model": {
                "role": "legacy_4b2_current_production",
                "opening_m": self.state.production_opening_m,
                "palm_z_m": self.state.production_palm_z_m,
                "finger_length_m": self.state.production_finger_length_m,
                "finger_thickness_m": self.state.production_finger_thickness_m,
                "pad_height_m": self.state.production_pad_height_m,
                "model_from_tcp_z_m": self.state.production_model_from_tcp_z_m,
            },
            "profile_unchanged_fields": {
                # Keep the legacy key readable, but make the actual endpoint
                # semantics explicit for new payload consumers.
                "tool_tcp_from_endpoint_z_m": self.state.tcp_z_m,
                "tool_tcp_from_flange_z_m": self.state.tcp_z_m,
                "tool_tcp_pose_endpoint_m_rad": list(_state_tool_tcp_pose(self.state)),
                "width_clearance_m": 0.006,
                "terminal_target_contact_depth_m": 0.006,
            },
            "frames_world_from": {
                name: [list(row) for row in matrix]
                for name, matrix in frames.items()
            },
            "diagnostics": {
                "sweep_center_to_reviewed_grasp_center_distance_m": math.dist(
                    _mat4_translation(frames["graspgenx_sweep_center"]),
                    _mat4_translation(frames["dmtac_grasp_center"]),
                ),
                "sweep_center_to_reviewed_grasp_center_rotation_deg": _frame_angle(
                    frames["graspgenx_sweep_center"], frames["dmtac_grasp_center"]
                ),
                "current_production_model_to_dmtac_origin_distance_m": math.dist(
                    _mat4_translation(frames["production_graspgenx_model"]),
                    _mat4_translation(frames["dmtac_tactile"]),
                ),
                "current_production_model_to_dmtac_rotation_deg": _frame_angle(
                    frames["production_graspgenx_model"], frames["dmtac_tactile"]
                ),
                "current_production_to_aligned_model_origin_distance_m": math.dist(
                    _mat4_translation(frames["production_graspgenx_model"]),
                    _mat4_translation(frames["graspgenx_model"]),
                ),
                "current_production_to_aligned_model_rotation_deg": _frame_angle(
                    frames["production_graspgenx_model"], frames["graspgenx_model"]
                ),
                "warnings": list(details["warnings"]),
            },
        }


class AlignmentApp:
    def __init__(self, root: tk.Tk, model: AlignmentModel, output: Path | None = None, *,
                 alignment_profile_id: str = DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID):
        self.root = root
        self.model = model
        self.output = output
        self.alignment_profile_id = alignment_profile_id
        self.root.title("GraspGenX ↔ DMTAC 坐标对齐 · offline diagnostic")
        self.root.geometry("1540x940")
        self.root.minsize(1200, 760)
        self.vars: dict[str, tk.Variable] = {}
        self.flags = {
            "visual": tk.BooleanVar(value=True),
            "collision": tk.BooleanVar(value=False),
            "frames": tk.BooleanVar(value=True),
            "model": tk.BooleanVar(value=True),
            "output": tk.BooleanVar(value=True),
        }
        self.status = tk.StringVar(value="")
        self.metric = tk.StringVar(value="")
        self._build()
        self.refresh(fit=True)

    def _var(self, name: str, value: float | str) -> tk.Variable:
        variable = tk.DoubleVar(value=value) if isinstance(value, (float, int)) else tk.StringVar(value=value)
        self.vars[name] = variable
        variable.trace_add("write", lambda *_args: self.refresh())
        return variable

    def _field(self, parent: tk.Misc, row: int, label: str, name: str, value: float, unit: str, minimum: float, maximum: float, scale: float = 1.0) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        variable = self._var(name, value * scale)
        spin = ttk.Spinbox(parent, textvariable=variable, from_=minimum, to=maximum, increment=(0.1 if unit == "mm" else 1.0), width=11)
        spin.grid(row=row, column=1, sticky="ew", padx=5)
        ttk.Label(parent, text=unit).grid(row=row, column=2, sticky="w")

    def _value(self, name: str, default: float) -> float:
        return _finite(self.vars[name].get(), default)

    def _build(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(toolbar, text="URDF:").pack(side=tk.LEFT)
        ttk.Label(toolbar, text=str(self.model.document.source_path), foreground="#43536d").pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="保存 alignment JSON", command=self.save).pack(side=tk.RIGHT, padx=3)
        ttk.Button(toolbar, text="读取 alignment JSON", command=self.load).pack(side=tk.RIGHT, padx=3)
        ttk.Button(toolbar, text="重置名义值", command=self.reset).pack(side=tk.RIGHT, padx=3)

        left = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        left.grid(row=1, column=0, sticky="nsew")
        left.configure(width=410)
        right = ttk.Frame(self.root, padding=(0, 0, 8, 8))
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        base = ttk.LabelFrame(left, text="坐标系与状态", padding=8)
        base.pack(fill=tk.X, pady=(0, 7))
        base.columnconfigure(1, weight=1)
        ttk.Label(base, text="所有值只作用于离线预览；不连接真机.").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        row = ttk.Frame(base)
        row.grid(row=1, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(row, text="机械臂").pack(side=tk.LEFT)
        side_var = self._var("side", self.model.state.side)
        combo = ttk.Combobox(row, textvariable=side_var, values=("right", "left"), state="readonly", width=9)
        combo.pack(side=tk.LEFT, padx=5)
        combo.bind("<<ComboboxSelected>>", lambda _event: self._side_changed())
        self._field(base, 2, "DMTAC master q", "gripper_q", self.model.state.gripper_q, "q", -10.0, 10.0, 1.0)
        self._field(base, 3, "TCP ← flange Z", "tcp_z", self.model.state.tcp_z_m, "mm", 0.0, 300.0, 1000.0)
        qframe = ttk.LabelFrame(left, text="RM65 measured qpos (rad)", padding=8)
        qframe.pack(fill=tk.X, pady=7)
        for index, value in enumerate(self.model.state.arm_qpos or [0.0] * 6):
            self._field(qframe, index, f"joint {index + 1}", f"q{index + 1}", value, "rad", -7.0, 7.0, 1.0)

        model_frame = ttk.LabelFrame(left, text="两套 GraspGenX 模型（参数独立）", padding=8)
        model_frame.pack(fill=tk.X, pady=7)
        ttk.Label(model_frame, text="原始 / 当前生产 4B2 proxy").grid(row=0, column=0, columnspan=3, sticky="w")
        self._field(model_frame, 1, "model ← solid Z", "production_model_from_tcp_z", self.model.state.production_model_from_tcp_z_m, "mm", -200.0, 200.0, 1000.0)
        self._field(model_frame, 2, "最大开口", "production_opening", self.model.state.production_opening_m, "mm", 1.0, 200.0, 1000.0)
        self._field(model_frame, 3, "palm Z", "production_palm_z", self.model.state.production_palm_z_m, "mm", -200.0, 200.0, 1000.0)
        self._field(model_frame, 4, "finger length", "production_finger_length", self.model.state.production_finger_length_m, "mm", 1.0, 200.0, 1000.0)
        self._field(model_frame, 5, "finger thickness", "production_finger_thickness", self.model.state.production_finger_thickness_m, "mm", 1.0, 100.0, 1000.0)
        self._field(model_frame, 6, "pad height", "production_pad_height", self.model.state.production_pad_height_m, "mm", 1.0, 100.0, 1000.0)
        ttk.Separator(model_frame).grid(row=7, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(model_frame, text="DMTAC 对齐 GraspGenX proxy").grid(row=8, column=0, columnspan=3, sticky="w")
        self._field(model_frame, 9, "最大开口", "opening", self.model.state.model_opening_m, "mm", 1.0, 200.0, 1000.0)
        self._field(model_frame, 10, "palm Z", "palm_z", self.model.state.model_palm_z_m, "mm", -200.0, 200.0, 1000.0)
        self._field(model_frame, 11, "finger length", "finger_length", self.model.state.model_finger_length_m, "mm", 1.0, 200.0, 1000.0)
        self._field(model_frame, 12, "finger thickness", "finger_thickness", self.model.state.model_finger_thickness_m, "mm", 1.0, 100.0, 1000.0)
        self._field(model_frame, 13, "pad height", "pad_height", self.model.state.model_pad_height_m, "mm", 1.0, 100.0, 1000.0)
        self._field(model_frame, 14, "抓取中心 ← tactile X", "grasp_center_x", self.model.state.grasp_center_from_tactile_x_m, "mm", -100.0, 100.0, 1000.0)
        self._field(model_frame, 15, "抓取中心 ← tactile Y", "grasp_center_y", self.model.state.grasp_center_from_tactile_y_m, "mm", -100.0, 100.0, 1000.0)
        self._field(model_frame, 16, "抓取中心 ← tactile Z", "grasp_center_z", self.model.state.grasp_center_from_tactile_z_m, "mm", -100.0, 100.0, 1000.0)
        ttk.Separator(model_frame).grid(row=17, column=0, columnspan=3, sticky="ew", pady=5)
        ttk.Label(model_frame, text="DMTAC 对齐 GraspGenX base 的局部微调").grid(row=18, column=0, columnspan=3, sticky="w")
        for row, label, name in (
            (19, "X", "corr_x"), (20, "Y", "corr_y"), (21, "Z", "corr_z"),
        ):
            self._field(model_frame, row, label, name, 0.0, "mm", -100.0, 100.0, 1000.0)
        for row, label, name in (
            (22, "roll", "corr_roll"), (23, "pitch", "corr_pitch"), (24, "yaw", "corr_yaw"),
        ):
            self._field(model_frame, row, label, name, 0.0, "deg", -180.0, 180.0, 1.0)

        visible = ttk.LabelFrame(left, text="显示图层", padding=8)
        visible.pack(fill=tk.X, pady=7)
        for text, name in (
            ("URDF visual / actual hardware", "visual"),
            ("URDF collision", "collision"),
            ("坐标轴与标签", "frames"),
            ("GraspGenX 两指 proxy", "model"),
            ("TCP 输出 solid", "output"),
        ):
            ttk.Checkbutton(visible, text=text, variable=self.flags[name], command=self.refresh).pack(anchor="w")
        ttk.Label(left, textvariable=self.metric, justify=tk.LEFT, foreground="#2b5d75", wraplength=390).pack(fill=tk.X, pady=7)
        ttk.Label(left, textvariable=self.status, justify=tk.LEFT, foreground="#7a4b22", wraplength=390).pack(fill=tk.X, pady=7)

        viewbar = ttk.Frame(right, padding=(0, 0, 0, 6))
        viewbar.grid(row=0, column=0, sticky="ew")
        for text, view in (("等轴", "iso"), ("正视", "front"), ("侧视", "side"), ("俯视", "top")):
            ttk.Button(viewbar, text=text, command=lambda selected=view: self.viewer.set_view(selected)).pack(side=tk.LEFT, padx=2)
        ttk.Button(viewbar, text="适配视图", command=lambda: self.viewer.fit()).pack(side=tk.LEFT, padx=9)
        self.viewer = GripperCanvas(right)
        self.viewer.grid(row=1, column=0, sticky="nsew")
        ttk.Label(right, text="橙色=法兰；粉色=控制 TCP；棕色=GraspGenX 掌面锚定 proxy；黄色=DMTAC/4B2 实体适配 proxy；青色=DMTAC 感知框架；RGB=局部 XYZ。", foreground="#43536d").grid(row=2, column=0, sticky="w", pady=(5, 0))

    def _sync_state(self) -> None:
        self.model.state.side = str(self.vars["side"].get())
        self.model.state.gripper_q = self._value("gripper_q", 0.0)
        self.model.state.tcp_z_m = self._value("tcp_z", 140.0) / 1000.0
        self.model.state.arm_qpos = [self._value(f"q{i}", 0.0) for i in range(1, 7)]
        self.model.state.production_model_from_tcp_z_m = self._value("production_model_from_tcp_z", 42.999) / 1000.0
        self.model.state.production_opening_m = self._value("production_opening", 70.0) / 1000.0
        self.model.state.production_palm_z_m = self._value("production_palm_z", 0.0) / 1000.0
        self.model.state.production_finger_length_m = self._value("production_finger_length", 50.0) / 1000.0
        self.model.state.production_finger_thickness_m = self._value("production_finger_thickness", 10.0) / 1000.0
        self.model.state.production_pad_height_m = self._value("production_pad_height", 25.5) / 1000.0
        self.model.state.model_opening_m = self._value("opening", 70.0) / 1000.0
        self.model.state.model_palm_z_m = self._value("palm_z", DEFAULT_PALM_Z_M * 1000.0) / 1000.0
        self.model.state.model_finger_length_m = self._value("finger_length", 50.0) / 1000.0
        self.model.state.model_finger_thickness_m = self._value("finger_thickness", 10.0) / 1000.0
        self.model.state.model_pad_height_m = self._value("pad_height", 25.5) / 1000.0
        self.model.state.grasp_center_from_tactile_x_m = self._value("grasp_center_x", 0.0) / 1000.0
        self.model.state.grasp_center_from_tactile_y_m = self._value("grasp_center_y", 0.0) / 1000.0
        self.model.state.grasp_center_from_tactile_z_m = self._value("grasp_center_z", -20.0) / 1000.0
        self.model.state.model_correction_xyz_m = [self._value("corr_x", 0.0) / 1000.0, self._value("corr_y", 0.0) / 1000.0, self._value("corr_z", 0.0) / 1000.0]
        self.model.state.model_correction_rpy_rad = [math.radians(self._value("corr_roll", 0.0)), math.radians(self._value("corr_pitch", 0.0)), math.radians(self._value("corr_yaw", 0.0))]

    def _side_changed(self) -> None:
        self._sync_state()
        self.model._refresh_pair()
        self.refresh(fit=True)

    def refresh(self, fit: bool = False) -> None:
        if not hasattr(self, "viewer") or not all(name in self.vars for name in ("side", "tcp_z")):
            return
        self._sync_state()
        self.model._refresh_pair()
        flags = {name: bool(variable.get()) for name, variable in self.flags.items()}
        try:
            shapes, markers, _frames, details = self.model.scene(flags)
            self.viewer.set_scene(shapes, markers, fit=fit)
            width = details.get("sensor_width_m")
            metric = self.model.frames()[0]
            text = (
                f"DMTAC sensing-origin separation: {'—' if width is None else f'{width * 1000:.2f} mm'}\n"
                f"sweep center ↔ reviewed grasp center: {math.dist(_mat4_translation(metric['graspgenx_sweep_center']), _mat4_translation(metric['dmtac_grasp_center'])) * 1000:.2f} mm\n"
                f"sweep center ↔ grasp center rotation: {_frame_angle(metric['graspgenx_sweep_center'], metric['dmtac_grasp_center']):.2f} deg"
            )
            self.metric.set(text)
            warnings = details.get("warnings", [])
            self.status.set("；".join(warnings) if warnings else "离线预览正常。保存 JSON 后再人工确认是否将修正接入 profile。")
        except Exception as exc:
            self.status.set(f"预览失败：{type(exc).__name__}: {exc}")

    def reset(self) -> None:
        self._apply_state(current_dmtac_state(self.model.state.side,
            _state_tool_tcp_pose(self.model.state), [0.] * 6,
            alignment_profile_id=self.alignment_profile_id))

    def save(self) -> None:
        target = self.output or Path(f"graspgenx-dmtac-alignment-{self.model.state.side}.json")
        selected = filedialog.asksaveasfilename(
            title="保存 GraspGenX-DMTAC alignment JSON",
            initialfile=target.name,
            defaultextension=".json",
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
        )
        if not selected:
            return
        self._sync_state()
        Path(selected).write_text(json.dumps(self.model.payload(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.status.set(f"已保存独立 alignment JSON：{selected}")

    def _apply_state(self, state: AlignmentState) -> None:
        values = {
            "side": state.side,
            "gripper_q": state.gripper_q,
            "tcp_z": state.tcp_z_m * 1000.,
            "production_model_from_tcp_z": state.production_model_from_tcp_z_m * 1000.,
            "production_opening": state.production_opening_m * 1000.,
            "production_palm_z": state.production_palm_z_m * 1000.,
            "production_finger_length": state.production_finger_length_m * 1000.,
            "production_finger_thickness": state.production_finger_thickness_m * 1000.,
            "production_pad_height": state.production_pad_height_m * 1000.,
            "opening": state.model_opening_m * 1000.,
            "palm_z": state.model_palm_z_m * 1000.,
            "finger_length": state.model_finger_length_m * 1000.,
            "finger_thickness": state.model_finger_thickness_m * 1000.,
            "pad_height": state.model_pad_height_m * 1000.,
            "grasp_center_x": state.grasp_center_from_tactile_x_m * 1000.,
            "grasp_center_y": state.grasp_center_from_tactile_y_m * 1000.,
            "grasp_center_z": state.grasp_center_from_tactile_z_m * 1000.,
        }
        for axis, value in zip("xyz", state.model_correction_xyz_m):
            values["corr_"+axis] = value * 1000.
        for axis, value in zip(("roll", "pitch", "yaw"), state.model_correction_rpy_rad):
            values["corr_"+axis] = math.degrees(value)
        for index, value in enumerate(state.arm_qpos, start=1):
            values[f"q{index}"] = value
        for name, value in values.items():
            if name in self.vars:
                self.vars[name].set(value)
        # Variable callbacks may have synchronized the previous state.
        # Install the complete parsed pose only after all controls agree.
        self.model.state = state
        self.model._refresh_pair()
        self.refresh(fit=True)

    def load(self) -> None:
        selected = filedialog.askopenfilename(title="读取 alignment JSON", filetypes=(("JSON", "*.json"), ("All files", "*.*")))
        if not selected:
            return
        try:
            payload = json.loads(Path(selected).read_text(encoding="utf-8"))
            state = _state_from_payload(payload)
            self._apply_state(state)
            self.status.set(f"已读取 alignment JSON：{selected}")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            messagebox.showerror("读取失败", f"{type(exc).__name__}: {exc}")


def _payload_model_palm_z(
    payload: dict[str, object],
    grasp: dict[str, object],
    model_from_tcp_z_m: float,
) -> float:
    """Read the model-local palm coordinate with legacy-v1 compatibility."""
    value = _finite(grasp.get("palm_z_m"), DEFAULT_PALM_Z_M)
    contract = payload.get("coordinate_contract")
    if not isinstance(contract, dict):
        # Old GUI payloads stored the TCP-local -42.999 mm value in the field
        # that is now explicitly model-local.  Normalize that representation
        # to the new palm-boundary anchor instead of reintroducing the double
        # offset when an old JSON is loaded.
        if math.isclose(value, -model_from_tcp_z_m, abs_tol=1e-6):
            return DEFAULT_PALM_Z_M
    return value


def _payload_grasp_center_xyz(grasp: dict[str, object]) -> tuple[float, float, float]:
    """Read the tactile-local grasp-center offset with z-only compatibility."""
    value = grasp.get("grasp_center_from_tactile_xyz_m")
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return _vec3(value, DEFAULT_GRASP_CENTER_FROM_TACTILE_XYZ_M)
    return (
        DEFAULT_GRASP_CENTER_FROM_TACTILE_XYZ_M[0],
        DEFAULT_GRASP_CENTER_FROM_TACTILE_XYZ_M[1],
        _finite(
            grasp.get("grasp_center_from_tactile_z_m"),
            DEFAULT_GRASP_CENTER_FROM_TACTILE_Z_M,
        ),
    )


def _state_from_payload(payload: dict[str, object]) -> AlignmentState:
    if not isinstance(payload, dict):
        raise TypeError("Alignment payload must be an object.")
    grasp = payload.get("graspgenx_model", {})
    if not isinstance(grasp, dict):
        grasp = {}
    original = payload.get("original_model")
    if not isinstance(original, dict):
        # v2 payloads had one shared model block.  Read it as both models so
        # loading an old alignment does not silently change its appearance.
        original = grasp
    dmtac = payload.get("dmtac", {})
    if not isinstance(dmtac, dict):
        dmtac = {}
    profile = payload.get("profile_unchanged_fields", {})
    if not isinstance(profile, dict):
        profile = {}
    model_from_tcp_z_m = _finite(
        grasp.get("model_from_tcp_z_m"), DEFAULT_MODEL_FROM_TCP_Z_M
    )
    production_model_from_tcp_z_m = _finite(
        original.get("model_from_tcp_z_m"), model_from_tcp_z_m
    )
    center_xyz = _payload_grasp_center_xyz(grasp)
    if "tool_tcp_pose_endpoint_m_rad" in profile:
        raw_tcp_pose = profile["tool_tcp_pose_endpoint_m_rad"]
        if (not isinstance(raw_tcp_pose, (list, tuple)) or len(raw_tcp_pose) != 6
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       for value in raw_tcp_pose)):
            raise ValueError("tool TCP pose must contain six finite xyz/rpy values")
        tcp_pose = [float(value) for value in raw_tcp_pose]
        _pose6_matrix(tcp_pose)
    else:
        tcp_pose = [0.0, 0.0, _finite(profile.get("tool_tcp_from_endpoint_z_m",
            profile.get("tool_tcp_from_flange_z_m")), DEFAULT_TCP_Z_M), 0.0, 0.0, 0.0]
    return AlignmentState(
        side=str(payload.get("side", "right")),
        arm_qpos=[_finite(v, 0.0) for v in payload.get("arm_qpos_rad", [0.0] * 6)] if isinstance(payload.get("arm_qpos_rad"), list) else [0.0] * 6,
        gripper_q=_finite(dmtac.get("master_q"), 0.0),
        tcp_z_m=tcp_pose[2],
        tool_tcp_pose_endpoint_m_rad=tcp_pose,
        model_from_tcp_z_m=model_from_tcp_z_m,
        model_opening_m=_finite(grasp.get("opening_m"), DEFAULT_OPENING_M),
        model_palm_z_m=_payload_model_palm_z(payload, grasp, model_from_tcp_z_m),
        model_finger_length_m=_finite(grasp.get("finger_length_m"), DEFAULT_FINGER_LENGTH_M),
        model_finger_thickness_m=_finite(grasp.get("finger_thickness_m"), DEFAULT_FINGER_THICKNESS_M),
        model_pad_height_m=_finite(grasp.get("pad_height_m"), DEFAULT_PAD_HEIGHT_M),
        grasp_center_from_tactile_x_m=center_xyz[0],
        grasp_center_from_tactile_y_m=center_xyz[1],
        grasp_center_from_tactile_z_m=_finite(
            center_xyz[2], DEFAULT_GRASP_CENTER_FROM_TACTILE_Z_M
        ),
        model_correction_xyz_m=list(
            _vec3(grasp.get("correction_xyz_m"), DEFAULT_MODEL_CORRECTION_M)
        ),
        model_correction_rpy_rad=list(_vec3(grasp.get("correction_rpy_rad"))),
        production_model_from_tcp_z_m=production_model_from_tcp_z_m,
        production_opening_m=_finite(original.get("opening_m"), DEFAULT_PRODUCTION_OPENING_M),
        production_palm_z_m=_finite(original.get("palm_z_m"), DEFAULT_PRODUCTION_PALM_Z_M),
        production_finger_length_m=_finite(
            original.get("finger_length_m"), DEFAULT_PRODUCTION_FINGER_LENGTH_M
        ),
        production_finger_thickness_m=_finite(
            original.get("finger_thickness_m"), DEFAULT_PRODUCTION_FINGER_THICKNESS_M
        ),
        production_pad_height_m=_finite(
            original.get("pad_height_m"), DEFAULT_PRODUCTION_PAD_HEIGHT_M
        ),
        profile_id=str(payload.get("profile_id", DEFAULT_PROFILE_ID)),
    )


def export_payload(
    urdf: Path,
    output: Path,
    side: str,
    arm_qpos: Sequence[float],
    alignment: Path | None,
    tool_profile: Path | None = None,
    alignment_profile_id: str = DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID,
) -> None:
    tcp_pose = (
        load_tool_tcp_pose(tool_profile)
        if tool_profile is not None
        else (0.0, 0.0, DEFAULT_TCP_Z_M, 0.0, 0.0, 0.0)
    )
    state = current_dmtac_state(side, tcp_pose, arm_qpos, alignment_profile_id=alignment_profile_id)
    if alignment is not None:
        state = _state_from_payload(json.loads(alignment.read_text(encoding="utf-8")))
        state.side = side
        state.arm_qpos = list(arm_qpos)
    model = AlignmentModel(urdf, state)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model.payload(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline GraspGenX ↔ DMTAC coordinate alignment GUI.")
    parser.add_argument("--urdf", type=Path, required=True, help="URDF containing the RM65 arm and DMTAC/hand geometry.")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--arm-qpos", default="0,0,0,0,0,0", help="six RM65 joint values in radians; offline only")
    parser.add_argument("--profile-id", default=DEFAULT_PROFILE_ID)
    parser.add_argument("--alignment-profile", choices=DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_IDS,
        default=DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID)
    parser.add_argument(
        "--tool-profile",
        type=Path,
        help="RM65 tool profile whose tcp.pose_flange_m_rad is applied to arm_link6",
    )
    parser.add_argument("--load-alignment", type=Path)
    parser.add_argument("--headless-export", type=Path, help="write JSON without opening Tk")
    args = parser.parse_args()
    try:
        qpos = [float(value.strip()) for value in args.arm_qpos.split(",")]
        if len(qpos) != 6:
            raise ValueError("--arm-qpos requires six comma-separated values")
        tool_profile = args.tool_profile
        if tool_profile is None:
            tool_profile = (
                Path(__file__).resolve().parents[1]
                / "hardware/rm65/utils/configs/calibrations"
                / f"rm65_{args.side}_DMTAC_M.json"
            )
        tcp_pose = load_tool_tcp_pose(tool_profile)
        if args.headless_export:
            export_payload(
                args.urdf,
                args.headless_export,
                args.side,
                qpos,
                args.load_alignment,
                tool_profile,
                args.alignment_profile,
            )
            print(json.dumps({"status": "ok", "output": str(args.headless_export)}, ensure_ascii=False))
            return 0
        state = current_dmtac_state(args.side, tcp_pose, qpos,
            alignment_profile_id=args.alignment_profile)
        state.profile_id = args.profile_id
        if args.load_alignment:
            state = _state_from_payload(json.loads(args.load_alignment.read_text(encoding="utf-8")))
            state.side = args.side
            state.arm_qpos = qpos
        root = tk.Tk()
        AlignmentApp(root, AlignmentModel(args.urdf, state), None, alignment_profile_id=args.alignment_profile)
        root.mainloop()
        return 0
    except (OSError, ValueError, URDFValidationError) as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        return 2
    except tk.TclError as exc:
        print(f"error: GUI display is unavailable: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
