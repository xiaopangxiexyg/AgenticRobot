"""Build the B601-RS visualization plant from the pinned upstream assets."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np

from robot_motion.rebot_b601_rs.model import (
    B601_RS_ARM_JOINT_NAMES,
    B601_RS_END_EFFECTOR_FRAME,
    B601_RS_URDF_RELATIVE_PATH,
    B601ModelError,
)
from sim.mujoco.config import ArmSpec
from sim.mujoco.mjcf import MJCFConversion, convert_urdf_to_mjcf_xml
from sim.mujoco.scene import MujocoScene, SceneLoadResult
from sim.mujoco.urdf import (
    URDFGeometry,
    URDFModel,
    parse_urdf,
)

B601_RS_SIM_ARM_KEY = "b601_rs"
B601_RS_SIM_EFFECTOR_NAME = "b601_handle"
B601_RS_PHYSICS_HZ = 1000.0
B601_RS_COMMAND_HZ = 500.0
B601_RS_DEFAULT_ARM_QPOS = (0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
B601_RS_GRIPPER_OPEN_QPOS = {
    "joint_left": 0.05,
    "joint_right": 0.0715,
}
_B601_RS_GRIPPER_LOCK_SOLREF = (0.002, 1.0)

# These gains are upstream configuration values used only as an initial
# visualization seed.  They are not identified hardware or haptic gains.
B601_RS_PROVISIONAL_POSITION_KP = (50.0, 150.0, 150.0, 50.0, 50.0, 50.0)
B601_RS_PROVISIONAL_POSITION_KV = (3.0, 10.0, 10.0, 5.0, 4.0, 4.0)
B601_RS_PROVISIONAL_PROFILE_NAME = "rebot_b601_rs_visualization_only_provisional_seed"
B601_RS_EFFORT_PROFILE_NAME = "rebot_b601_rs_ideal_effort_simulation_only"
B601ActuationMode = Literal["position", "effort"]


def b601_position_actuator_name(joint_name: str) -> str:
    """Return the stable actuator name for one B601 arm joint."""

    return f"b601_rs_position_{joint_name}"


def b601_effort_actuator_name(joint_name: str) -> str:
    """Return the stable ideal-effort actuator name for one arm joint."""

    return f"b601_rs_effort_{joint_name}"


def _validate_actuation_mode(value: str) -> B601ActuationMode:
    normalized = str(value).strip()
    if normalized not in {"position", "effort"}:
        raise ValueError("actuation_mode must be 'position' or 'effort'")
    return normalized  # type: ignore[return-value]


def _source_paths(source_root: str | Path) -> tuple[Path, Path, Path]:
    try:
        root = Path(source_root).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise FileNotFoundError(
            f"reBotArm source root not found: {source_root}"
        ) from exc
    if not root.is_dir():
        raise FileNotFoundError(f"reBotArm source root is not a directory: {root}")

    try:
        urdf_path = (root / B601_RS_URDF_RELATIVE_PATH).resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise FileNotFoundError(
            "B601-RS URDF is absent from the supplied reBotArm source root"
        ) from exc
    if not urdf_path.is_file():
        raise FileNotFoundError(f"B601-RS URDF is not a regular file: {urdf_path}")
    package_root = urdf_path.parent.parent
    for name, path in (("URDF", urdf_path), ("model package", package_root)):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise B601ModelError(
                f"B601-RS {name} resolves outside the supplied source root: {path}"
            ) from exc
    return root, urdf_path, package_root


def _normalized_mesh_geometry(
    geometry: URDFGeometry,
    *,
    package_root: Path,
) -> URDFGeometry:
    if geometry.kind != "mesh":
        return geometry
    if not geometry.filename:
        raise B601ModelError("B601-RS mesh geometry has no filename")
    raw_path = Path(geometry.filename)
    candidate = raw_path if raw_path.is_absolute() else package_root / raw_path
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise FileNotFoundError(f"B601-RS mesh not found: {candidate}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"B601-RS mesh is not a regular file: {resolved}")
    try:
        resolved.relative_to(package_root)
    except ValueError as exc:
        raise B601ModelError(
            f"B601-RS mesh resolves outside its model package: {resolved}"
        ) from exc
    return replace(geometry, filename=str(resolved))


def _load_normalized_urdf(
    source_root: str | Path,
) -> tuple[Path, URDFModel]:
    _root, urdf_path, package_root = _source_paths(source_root)
    model = parse_urdf(urdf_path)
    links = {}
    for name, link in model.links.items():
        visuals = tuple(
            replace(
                visual,
                geometry=_normalized_mesh_geometry(
                    visual.geometry,
                    package_root=package_root,
                ),
            )
            for visual in link.visuals
        )
        collisions = tuple(
            replace(
                collision,
                geometry=_normalized_mesh_geometry(
                    collision.geometry,
                    package_root=package_root,
                ),
            )
            for collision in link.collisions
        )
        links[name] = replace(link, visuals=visuals, collisions=collisions)
    return urdf_path, replace(model, links=links)


def _arm_joint_limits(
    model: URDFModel,
    name: str,
) -> tuple[float, float, float]:
    matches = [joint for joint in model.joints if joint.name == name]
    if len(matches) != 1:
        raise B601ModelError(
            f"B601-RS URDF must contain exactly one arm joint {name!r}"
        )
    joint = matches[0]
    limit = joint.limit
    if (
        limit is None
        or limit.lower is None
        or limit.upper is None
        or limit.effort is None
        or not all(
            math.isfinite(float(value))
            for value in (limit.lower, limit.upper, limit.effort)
        )
        or float(limit.lower) >= float(limit.upper)
        or float(limit.effort) <= 0.0
    ):
        raise B601ModelError(
            f"B601-RS arm joint {name!r} lacks finite position/effort limits"
        )
    assert limit.lower is not None
    assert limit.upper is not None
    assert limit.effort is not None
    return float(limit.lower), float(limit.upper), float(limit.effort)


def _add_provisional_position_actuators(
    root: ET.Element,
    model: URDFModel,
) -> None:
    joint_elements = {
        str(element.attrib["name"]): element
        for element in root.iter("joint")
        if "name" in element.attrib
    }
    actuator_root = ET.SubElement(root, "actuator")
    for index, name in enumerate(B601_RS_ARM_JOINT_NAMES):
        lower, upper, effort = _arm_joint_limits(model, name)
        if name not in joint_elements:
            raise B601ModelError(f"converted MJCF is missing arm joint {name!r}")
        ET.SubElement(
            actuator_root,
            "position",
            {
                "name": b601_position_actuator_name(name),
                "joint": name,
                "kp": f"{B601_RS_PROVISIONAL_POSITION_KP[index]:.12g}",
                "kv": f"{B601_RS_PROVISIONAL_POSITION_KV[index]:.12g}",
                "ctrllimited": "true",
                "ctrlrange": f"{lower:.12g} {upper:.12g}",
                "forcelimited": "true",
                "forcerange": f"{-effort:.12g} {effort:.12g}",
            },
        )


def _add_ideal_effort_actuators(
    root: ET.Element,
    model: URDFModel,
) -> None:
    joint_elements = {
        str(element.attrib["name"]): element
        for element in root.iter("joint")
        if "name" in element.attrib
    }
    actuator_root = ET.SubElement(root, "actuator")
    for name in B601_RS_ARM_JOINT_NAMES:
        _lower, _upper, effort = _arm_joint_limits(model, name)
        if name not in joint_elements:
            raise B601ModelError(f"converted MJCF is missing arm joint {name!r}")
        ET.SubElement(
            actuator_root,
            "motor",
            {
                "name": b601_effort_actuator_name(name),
                "joint": name,
                "gear": "1",
                "ctrllimited": "true",
                "ctrlrange": f"{-effort:.12g} {effort:.12g}",
                "forcelimited": "true",
                "forcerange": f"{-effort:.12g} {effort:.12g}",
            },
        )


def _add_fixed_open_gripper_constraints(root: ET.Element) -> None:
    """Keep non-actuated gripper DOFs fixed during ideal-effort simulation.

    Gravity compensation is evaluated with the two auxiliary joints fixed at
    ``B601_RS_GRIPPER_OPEN_QPOS``.  A continuous MuJoCo equality constraint is
    required here: rewriting qpos/qvel between physics steps would inject
    impulses through the coupled arm dynamics.
    """

    equality_root = root.find("equality")
    if equality_root is None:
        equality_root = ET.SubElement(root, "equality")
    solref = " ".join(f"{value:.12g}" for value in _B601_RS_GRIPPER_LOCK_SOLREF)
    for joint_name, open_qpos in B601_RS_GRIPPER_OPEN_QPOS.items():
        ET.SubElement(
            equality_root,
            "joint",
            {
                "name": f"b601_rs_fixed_open_{joint_name}",
                "joint1": joint_name,
                "polycoef": f"{open_qpos:.12g} 0 0 0 0",
                "solref": solref,
            },
        )


def _mjcf_xml_from_model(
    model: URDFModel,
    *,
    timestep: float,
    gravity: np.ndarray,
    actuation_mode: B601ActuationMode,
) -> str:
    xml_text = convert_urdf_to_mjcf_xml(
        model,
        camera_mounts=(),
        timestep=timestep,
        gravity=tuple(float(value) for value in gravity),
    )
    root = ET.fromstring(xml_text)
    option = root.find("option")
    if option is None:
        raise B601ModelError("converted B601-RS MJCF has no option element")
    option.attrib["integrator"] = "implicitfast"
    if actuation_mode == "position":
        _add_provisional_position_actuators(root, model)
    else:
        _add_ideal_effort_actuators(root, model)
        _add_fixed_open_gripper_constraints(root)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def build_b601_mjcf_xml(
    source_root: str | Path,
    *,
    timestep_sec: float = 1.0 / B601_RS_PHYSICS_HZ,
    gravity_mps2: Sequence[float] = (0.0, 0.0, 0.0),
    actuation_mode: B601ActuationMode = "position",
) -> str:
    """Return in-memory MJCF without modifying or copying upstream assets."""

    timestep = float(timestep_sec)
    gravity = np.asarray(gravity_mps2, dtype=float)
    if not math.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("timestep_sec must be finite and positive")
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity_mps2 must contain three finite values")
    mode = _validate_actuation_mode(actuation_mode)
    _urdf_path, urdf_model = _load_normalized_urdf(source_root)
    return _mjcf_xml_from_model(
        urdf_model,
        timestep=timestep,
        gravity=gravity,
        actuation_mode=mode,
    )


def _validate_initial_qpos(values: Sequence[float], model: URDFModel) -> np.ndarray:
    qpos = np.asarray(values, dtype=float)
    if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
        raise ValueError("initial_arm_qpos must contain six finite values")
    for index, name in enumerate(B601_RS_ARM_JOINT_NAMES):
        lower, upper, _effort = _arm_joint_limits(model, name)
        if not lower <= qpos[index] <= upper:
            raise ValueError(f"initial_arm_qpos exceeds the limit for {name}")
    return qpos.copy()


def build_b601_mujoco_scene(
    source_root: str | Path,
    *,
    timestep_sec: float = 1.0 / B601_RS_PHYSICS_HZ,
    gravity_mps2: Sequence[float] = (0.0, 0.0, 0.0),
    initial_arm_qpos: Sequence[float] = B601_RS_DEFAULT_ARM_QPOS,
    actuation_mode: B601ActuationMode = "position",
) -> MujocoScene:
    """Compile the B601 plant and wrap it in the generic scene facade."""

    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuJoCo is required to build the B601-RS simulation scene"
        ) from exc

    urdf_path, urdf_model = _load_normalized_urdf(source_root)
    initial = _validate_initial_qpos(initial_arm_qpos, urdf_model)
    timestep = float(timestep_sec)
    gravity = np.asarray(gravity_mps2, dtype=float)
    if not math.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("timestep_sec must be finite and positive")
    if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
        raise ValueError("gravity_mps2 must contain three finite values")
    mode = _validate_actuation_mode(actuation_mode)
    xml_text = _mjcf_xml_from_model(
        urdf_model,
        timestep=timestep,
        gravity=gravity,
        actuation_mode=mode,
    )
    model = mujoco.MjModel.from_xml_string(xml_text)
    data = mujoco.MjData(model)
    conversion = MJCFConversion(
        urdf_path=urdf_path,
        mjcf_path=Path("<in-memory-b601-rs-mjcf>"),
        camera_mounts=(),
        joint_names=tuple(
            joint.name for joint in urdf_model.joints if joint.joint_type != "fixed"
        ),
        body_names=tuple(urdf_model.links),
        scene_profile_name=(
            B601_RS_PROVISIONAL_PROFILE_NAME
            if mode == "position"
            else B601_RS_EFFORT_PROFILE_NAME
        ),
    )
    arm_spec = ArmSpec(
        side=B601_RS_SIM_ARM_KEY,
        joint_names=B601_RS_ARM_JOINT_NAMES,
        controlled_body=B601_RS_END_EFFECTOR_FRAME,
        effector_name=B601_RS_SIM_EFFECTOR_NAME,
    )
    scene = MujocoScene(
        mujoco=mujoco,
        model=model,
        data=data,
        load_result=SceneLoadResult(
            conversion=conversion,
            cameras=(),
            arms={B601_RS_SIM_ARM_KEY: arm_spec},
            grippers={},
            scene_spec=None,
        ),
    )
    scene.set_arm_qpos(B601_RS_SIM_ARM_KEY, initial)
    for name, value in B601_RS_GRIPPER_OPEN_QPOS.items():
        scene.data.qpos[scene.joint_qpos_addr(name)] = float(value)
        scene.data.qvel[scene.joint_dof_addr(name)] = 0.0
    if mode == "position":
        actuator_ids = [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                b601_position_actuator_name(name),
            )
            for name in B601_RS_ARM_JOINT_NAMES
        ]
        scene.data.ctrl[np.asarray(actuator_ids, dtype=np.int32)] = initial
    else:
        scene.data.ctrl[:] = 0.0
    scene.forward()
    return scene


__all__ = [
    "B601_RS_COMMAND_HZ",
    "B601_RS_DEFAULT_ARM_QPOS",
    "B601_RS_EFFORT_PROFILE_NAME",
    "B601_RS_GRIPPER_OPEN_QPOS",
    "B601_RS_PHYSICS_HZ",
    "B601_RS_PROVISIONAL_POSITION_KP",
    "B601_RS_PROVISIONAL_POSITION_KV",
    "B601_RS_PROVISIONAL_PROFILE_NAME",
    "B601_RS_SIM_ARM_KEY",
    "B601ActuationMode",
    "b601_effort_actuator_name",
    "b601_position_actuator_name",
    "build_b601_mjcf_xml",
    "build_b601_mujoco_scene",
]
