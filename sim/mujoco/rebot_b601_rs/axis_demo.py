"""Visualize B601 operator-guide directions in the MuJoCo world frame.

The robot remains at one fixed simulated configuration.  A thin frame marks
the current end-effector anchor and a brighter mocap frame demonstrates each
signed translation and right-hand-rule rotation.  This module has no hardware
backend, CAN transport, or motor-control entrypoint.
"""

from __future__ import annotations

import argparse
import math
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.rebot_b601_rs import B601PinocchioModel
from sim.mujoco.viewer import close_passive_viewer

from .scene import (
    B601_RS_ARM_JOINT_NAMES,
    B601_RS_DEFAULT_ARM_QPOS,
    B601_RS_GRIPPER_OPEN_QPOS,
    b601_position_actuator_name,
    build_b601_mjcf_xml,
)

DEFAULT_TRANSLATION_M = 0.03
DEFAULT_ROTATION_RAD = math.radians(8.0)
DEFAULT_SETTLE_SEC = 2.0
DEFAULT_PHASE_SEC = 3.0
_ANCHOR_BODY_NAME = "b601_axis_anchor_frame"
_MOVING_BODY_NAME = "b601_axis_moving_frame"


@dataclass(frozen=True)
class AxisDirectionPhase:
    """One signed base/world-aligned visual motion."""

    kind: str
    axis: str
    direction: int

    @property
    def name(self) -> str:
        sign = "POSITIVE" if self.direction > 0 else "NEGATIVE"
        return f"{self.kind.upper()}_{self.axis.upper()}_{sign}"


def axis_direction_phases() -> tuple[AxisDirectionPhase, ...]:
    """Return the same signed-axis order used by foreground acquisition."""

    return tuple(
        AxisDirectionPhase(kind, axis, direction)
        for kind, axes in (("translation", "xyz"), ("rotation", "xyz"))
        for axis in axes
        for direction in (1, -1)
    )


def direction_frame_pose(
    phase: AxisDirectionPhase,
    *,
    phase_fraction: float,
    anchor_position_m: Sequence[float],
    translation_m: float,
    rotation_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return world position and wxyz quaternion for one out-and-back phase."""

    fraction = float(phase_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("phase_fraction must be finite and inside [0, 1]")
    anchor = np.asarray(anchor_position_m, dtype=float)
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        raise ValueError("anchor_position_m must contain three finite values")
    translation = float(translation_m)
    rotation = float(rotation_rad)
    if not math.isfinite(translation) or translation <= 0.0:
        raise ValueError("translation_m must be finite and positive")
    if not math.isfinite(rotation) or rotation <= 0.0:
        raise ValueError("rotation_rad must be finite and positive")

    # One smooth lobe reaches the signed target halfway through the phase and
    # returns exactly to the anchor before the following direction begins.
    displacement_fraction = math.sin(math.pi * fraction)
    axis_index = "xyz".index(phase.axis)
    position = anchor.copy()
    rotation_vector = np.zeros(3, dtype=float)
    if phase.kind == "translation":
        position[axis_index] += (
            phase.direction * translation * displacement_fraction
        )
    elif phase.kind == "rotation":
        rotation_vector[axis_index] = (
            phase.direction * rotation * displacement_fraction
        )
    else:
        raise ValueError("phase kind must be translation or rotation")
    quaternion_xyzw = Rotation.from_rotvec(rotation_vector).as_quat()
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    return position, quaternion_wxyz


def _add_axis_geoms(
    body: ET.Element,
    *,
    prefix: str,
    alpha: float,
    radius_m: float,
    length_m: float,
) -> None:
    colors = {
        "x": (0.95, 0.12, 0.12),
        "y": (0.12, 0.80, 0.20),
        "z": (0.15, 0.35, 1.00),
    }
    for index, axis in enumerate("xyz"):
        endpoint = np.zeros(3, dtype=float)
        endpoint[index] = length_m
        rgba = " ".join(str(value) for value in (*colors[axis], alpha))
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{prefix}_{axis}_shaft",
                "type": "capsule",
                "fromto": "0 0 0 " + " ".join(f"{value:.8g}" for value in endpoint),
                "size": f"{radius_m:.8g}",
                "rgba": rgba,
                "contype": "0",
                "conaffinity": "0",
                # Group 0 is visible in the stock passive viewer.  Higher
                # decoration groups may start disabled, hiding the guide.
                "group": "0",
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{prefix}_{axis}_tip",
                "type": "sphere",
                "pos": " ".join(f"{value:.8g}" for value in endpoint),
                "size": f"{2.2 * radius_m:.8g}",
                "rgba": rgba,
                "contype": "0",
                "conaffinity": "0",
                "group": "0",
            },
        )


def build_axis_visualization_xml(
    source_root: str | Path,
    *,
    anchor_position_m: Sequence[float],
) -> str:
    """Add non-colliding anchor and mocap triads to the existing B601 scene."""

    anchor = np.asarray(anchor_position_m, dtype=float)
    if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
        raise ValueError("anchor_position_m must contain three finite values")
    root = ET.fromstring(build_b601_mjcf_xml(source_root))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("B601 MJCF has no worldbody")
    anchor_text = " ".join(f"{value:.12g}" for value in anchor)
    anchor_body = ET.SubElement(
        worldbody,
        "body",
        {"name": _ANCHOR_BODY_NAME, "pos": anchor_text},
    )
    _add_axis_geoms(
        anchor_body,
        prefix="b601_anchor",
        alpha=0.35,
        radius_m=0.0025,
        length_m=0.10,
    )
    moving_body = ET.SubElement(
        worldbody,
        "body",
        {"name": _MOVING_BODY_NAME, "mocap": "true", "pos": anchor_text},
    )
    _add_axis_geoms(
        moving_body,
        prefix="b601_moving",
        alpha=0.95,
        radius_m=0.004,
        length_m=0.13,
    )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Show B601 +/-XYZ translations and +/-RxRyRz right-hand rotations "
            "in the MuJoCo base/world-aligned frame. No hardware is accessed."
        )
    )
    parser.add_argument("--rebot-source-root", type=Path, required=True)
    parser.add_argument(
        "--translation-m",
        type=float,
        default=DEFAULT_TRANSLATION_M,
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=math.degrees(DEFAULT_ROTATION_RAD),
    )
    parser.add_argument("--settle-sec", type=float, default=DEFAULT_SETTLE_SEC)
    parser.add_argument("--phase-sec", type=float, default=DEFAULT_PHASE_SEC)
    return parser


def run_axis_visualizer(
    *,
    source_root: str | Path,
    translation_m: float,
    rotation_rad: float,
    settle_sec: float,
    phase_sec: float,
) -> int:
    """Open the foreground viewer and animate only the mocap direction frame."""

    translation = float(translation_m)
    rotation = float(rotation_rad)
    settle = float(settle_sec)
    phase_duration = float(phase_sec)
    for value, name in (
        (translation, "translation_m"),
        (rotation, "rotation_rad"),
        (settle, "settle_sec"),
        (phase_duration, "phase_sec"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuJoCo with viewer support is required for the axis visualizer"
        ) from exc

    qpos = np.asarray(B601_RS_DEFAULT_ARM_QPOS, dtype=float)
    kinematic_model = B601PinocchioModel.from_rebot_source_root(source_root)
    anchor_pose = np.asarray(kinematic_model.forward_kinematics(qpos), dtype=float)
    anchor_position = anchor_pose[:3, 3].copy()
    xml_text = build_axis_visualization_xml(
        source_root,
        anchor_position_m=anchor_position,
    )
    model = mujoco.MjModel.from_xml_string(xml_text)
    data = mujoco.MjData(model)
    for index, joint_name in enumerate(B601_RS_ARM_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = qpos[index]
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            b601_position_actuator_name(joint_name),
        )
        data.ctrl[actuator_id] = qpos[index]
    for joint_name, value in B601_RS_GRIPPER_OPEN_QPOS.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = value
    moving_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        _MOVING_BODY_NAME,
    )
    mocap_id = int(model.body_mocapid[moving_body_id])
    if mocap_id < 0:
        raise RuntimeError("moving direction frame is not a MuJoCo mocap body")
    data.mocap_pos[mocap_id] = anchor_position
    data.mocap_quat[mocap_id] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)

    phases = axis_direction_phases()
    print("[SIM] B601 remains fixed; only the bright direction frame moves.")
    print("[SIM] Red=X, green=Y, blue=Z. Frame: X forward, Y left, Z up.")
    print("[SIM] Rotations use the right-hand rule. Close the viewer to stop.")
    handle = mujoco.viewer.launch_passive(model, data)
    try:
        handle.cam.lookat[:] = anchor_position
        handle.cam.distance = 0.75
        handle.cam.azimuth = 135.0
        handle.cam.elevation = -18.0
        started = time.monotonic()
        announced_phase = -1
        while handle.is_running():
            elapsed = time.monotonic() - started
            if elapsed < settle:
                if announced_phase != 0:
                    print(f"[SIM 00/{len(phases):02d}] anchor frame; no motion")
                    announced_phase = 0
                position = anchor_position
                quaternion = np.asarray((1.0, 0.0, 0.0, 0.0))
            else:
                phase_time = elapsed - settle
                phase_index = int(phase_time // phase_duration)
                if phase_index >= len(phases):
                    print("[SIM] direction sequence complete")
                    break
                phase = phases[phase_index]
                fraction = (phase_time % phase_duration) / phase_duration
                position, quaternion = direction_frame_pose(
                    phase,
                    phase_fraction=fraction,
                    anchor_position_m=anchor_position,
                    translation_m=translation,
                    rotation_rad=rotation,
                )
                if announced_phase != phase_index + 1:
                    sign = "+" if phase.direction > 0 else "-"
                    magnitude = (
                        f"{100.0 * translation:g} cm"
                        if phase.kind == "translation"
                        else f"{math.degrees(rotation):g} deg"
                    )
                    print(
                        f"[SIM {phase_index + 1:02d}/{len(phases):02d}] "
                        f"{phase.kind} {sign}{phase.axis.upper()} {magnitude}"
                    )
                    announced_phase = phase_index + 1
            data.mocap_pos[mocap_id] = position
            data.mocap_quat[mocap_id] = quaternion
            # This is a coordinate explainer, not a controller experiment.
            # Re-forward the display state without advancing robot dynamics.
            mujoco.mj_forward(model, data)
            handle.sync()
            time.sleep(1.0 / 60.0)
    finally:
        close_passive_viewer(handle)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run_axis_visualizer(
        source_root=args.rebot_source_root,
        translation_m=args.translation_m,
        rotation_rad=math.radians(args.rotation_deg),
        settle_sec=args.settle_sec,
        phase_sec=args.phase_sec,
    )


if __name__ == "__main__":
    raise SystemExit(main())
