from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from project_paths import repo_path, resolve_path
from sim.contracts import FreeBodyObjectSpec, SimulationContract

from ..mjcf import convert_urdf_to_mjcf_xml
from ..urdf import parse_urdf


DEFAULT_CONTRACT_PATH = repo_path("sim/configs/rm65_bimanual_physics_v1.json")
DEFAULT_TABLE_BODY = "table"


@dataclass(frozen=True)
class IsolatedDropScene:
    """MJCF and metadata for one table-top rigid-body drop diagnostic."""

    xml: str
    object_spec: FreeBodyObjectSpec
    table_body_name: str
    source_profile_name: str
    timestep_sec: float


def _select_object(
    contract: SimulationContract,
    object_name: str | None,
) -> FreeBodyObjectSpec:
    objects = contract.scene.objects
    if not objects:
        raise ValueError("The selected contract has no free-body object to drop.")
    if object_name is None:
        if len(objects) != 1:
            names = ", ".join(item.name for item in objects)
            raise ValueError(
                "The selected contract contains multiple free-body objects; "
                f"choose one with --object. Available: {names}."
            )
        return objects[0]
    requested = str(object_name)
    for item in objects:
        if item.name == requested:
            return item
    names = ", ".join(item.name for item in objects)
    raise ValueError(f"Unknown scene object {requested!r}. Available: {names}.")


def _direct_body(worldbody: ET.Element, name: str) -> ET.Element:
    for body in worldbody.findall("./body"):
        if body.attrib.get("name") == name:
            return body
    nested = next(
        (body for body in worldbody.iter("body") if body.attrib.get("name") == name),
        None,
    )
    if nested is not None:
        raise ValueError(
            f"Body {name!r} is nested in the URDF. The isolated diagnostic only "
            "supports a world-fixed, top-level table body so its pose is preserved."
        )
    raise ValueError(f"The scene URDF does not contain table body {name!r}.")


def _prune_unused_assets(root: ET.Element) -> None:
    """Keep only mesh assets referenced by the retained table/object geoms."""

    asset = root.find("./asset")
    if asset is None:
        return
    referenced_meshes = {
        mesh_name
        for geom in root.findall(".//geom")
        if (mesh_name := geom.attrib.get("mesh")) is not None
    }
    for item in list(asset):
        if item.tag != "mesh" or item.attrib.get("name") not in referenced_meshes:
            asset.remove(item)


def build_isolated_drop_scene(
    contract: SimulationContract,
    *,
    object_name: str | None = None,
    table_body_name: str = DEFAULT_TABLE_BODY,
) -> IsolatedDropScene:
    """Build ground + table + one freejoint object from a versioned contract.

    The regular converter remains the source of collision masks, contacts,
    gravity, table geometry and object inertial parameters. This function only
    removes articulated robot bodies and their now-unused mesh assets.
    """

    if not contract.scene.collision.enabled:
        raise ValueError("The cube-drop diagnostic requires scene collision to be enabled.")
    if not contract.scene.ground.enabled:
        raise ValueError("The cube-drop diagnostic requires the contract ground plane.")

    object_spec = _select_object(contract, object_name)
    isolated_spec = replace(
        contract.scene,
        profile_name=f"{contract.scene.profile_name}__isolated_drop",
        backend_role="isolated_rigid_body_diagnostic",
        objects=(object_spec,),
    )
    urdf_model = parse_urdf(resolve_path(contract.scene.urdf_path))
    xml = convert_urdf_to_mjcf_xml(
        urdf_model,
        camera_mounts=(),
        timestep=contract.timing.physics_timestep_sec,
        scene_spec=isolated_spec,
    )
    root = ET.fromstring(xml)
    root.attrib["model"] = f"{object_spec.name}_isolated_drop"
    worldbody = root.find("./worldbody")
    if worldbody is None:
        raise ValueError("Generated MJCF has no worldbody.")

    table = copy.deepcopy(_direct_body(worldbody, str(table_body_name)))
    # Semantic frames such as table_center_frame are useful in the complete
    # scene but are geometry-free and do not belong in the rigid-body drop
    # oracle. Preserve nested table geometry while pruning empty descendants.
    def prune_empty_descendants(body: ET.Element) -> bool:
        for child in list(body.findall("./body")):
            if not prune_empty_descendants(child):
                body.remove(child)
        return bool(
            body.findall("./geom")
            or body.findall("./site")
            or body.findall("./camera")
            or body.findall("./body")
        )

    prune_empty_descendants(table)
    dropped_object = copy.deepcopy(_direct_body(worldbody, object_spec.name))
    for body in list(worldbody.findall("./body")):
        worldbody.remove(body)
    worldbody.append(table)
    worldbody.append(dropped_object)

    # Parent/child exclusions were generated for the complete URDF and would
    # reference removed robot bodies. The isolated model has no self-collision
    # pairs that need explicit exclusions.
    for contact in list(root.findall("./contact")):
        root.remove(contact)
    _prune_unused_assets(root)
    ET.indent(root, space="  ")
    return IsolatedDropScene(
        xml=ET.tostring(root, encoding="unicode"),
        object_spec=object_spec,
        table_body_name=str(table_body_name),
        source_profile_name=contract.scene.profile_name,
        timestep_sec=contract.timing.physics_timestep_sec,
    )


def _require_interactive_display(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
) -> None:
    env = os.environ if environ is None else environ
    host_platform = sys.platform if platform is None else str(platform)
    if host_platform.startswith("linux") and not (
        env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError(
            "MuJoCo GUI needs an interactive Linux display, but neither DISPLAY "
            "nor WAYLAND_DISPLAY is set. Run this command in a desktop session "
            "or through an X/Wayland forwarding setup; MUJOCO_GL=egl/osmesa is "
            "for offscreen rendering and does not create an interactive viewer."
        )


def _load_mujoco_gui():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuJoCo is required for this diagnostic. Install the project MuJoCo "
            "dependency before launching the GUI."
        ) from exc
    if sys.platform.startswith("linux"):
        try:
            import glfw
        except ModuleNotFoundError as exc:
            raise RuntimeError("MuJoCo GUI requires the Python glfw package.") from exc
        _require_glfw_connection(glfw)
    try:
        import mujoco.viewer
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "The installed MuJoCo package does not provide mujoco.viewer."
        ) from exc
    return mujoco, mujoco.viewer


def _require_glfw_connection(glfw_module) -> None:
    """Fail cleanly when DISPLAY exists but the process cannot connect to it."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        initialized = bool(glfw_module.init())
    if not initialized:
        raise RuntimeError(
            "DISPLAY/WAYLAND_DISPLAY is set, but GLFW cannot connect to an "
            "interactive display. Check X/Wayland access or forwarding."
        )
    glfw_module.terminate()


def _configure_camera(handle, object_spec: FreeBodyObjectSpec) -> None:
    x, y, _z = object_spec.initial_position_m
    handle.cam.lookat[:] = (float(x), float(y), 0.035)
    handle.cam.distance = 0.75
    handle.cam.azimuth = 135.0
    handle.cam.elevation = -25.0


def launch_cube_drop_gui(
    scene: IsolatedDropScene,
    *,
    duration_sec: float | None = None,
    start_delay_sec: float = 1.0,
    repeat_every_sec: float | None = None,
    realtime_factor: float = 1.0,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
) -> None:
    """Step an isolated rigid-body model while a passive GUI displays it."""

    if start_delay_sec < 0.0:
        raise ValueError("--start-delay must be non-negative.")
    if realtime_factor <= 0.0:
        raise ValueError("--realtime-factor must be positive.")
    if repeat_every_sec is not None and repeat_every_sec <= start_delay_sec:
        raise ValueError("--repeat-every must be greater than --start-delay.")

    _require_interactive_display()
    mujoco, viewer = _load_mujoco_gui()
    model = mujoco.MjModel.from_xml_string(scene.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    started_at = time.monotonic()
    cycle_started_at = started_at
    previous_at = started_at
    accumulated_sim_sec = 0.0
    render_period_sec = 1.0 / 60.0
    try:
        with viewer.launch_passive(
            model,
            data,
            show_left_ui=bool(show_left_ui),
            show_right_ui=bool(show_right_ui),
        ) as handle:
            _configure_camera(handle, scene.object_spec)
            while handle.is_running():
                frame_started_at = time.monotonic()
                elapsed_sec = frame_started_at - started_at
                cycle_elapsed_sec = frame_started_at - cycle_started_at
                wall_step_sec = min(frame_started_at - previous_at, 0.1)
                previous_at = frame_started_at

                if duration_sec is not None and elapsed_sec >= duration_sec:
                    break
                if (
                    repeat_every_sec is not None
                    and cycle_elapsed_sec >= repeat_every_sec
                ):
                    mujoco.mj_resetData(model, data)
                    mujoco.mj_forward(model, data)
                    cycle_started_at = frame_started_at
                    previous_at = frame_started_at
                    accumulated_sim_sec = 0.0
                    cycle_elapsed_sec = 0.0

                if cycle_elapsed_sec >= start_delay_sec:
                    accumulated_sim_sec += wall_step_sec * realtime_factor
                    steps = int(accumulated_sim_sec / float(model.opt.timestep))
                    for _ in range(steps):
                        mujoco.mj_step(model, data)
                    accumulated_sim_sec -= steps * float(model.opt.timestep)

                handle.sync()
                remaining = render_period_sec - (time.monotonic() - frame_started_at)
                if remaining > 0.0:
                    time.sleep(remaining)
    except Exception as exc:  # noqa: BLE001 - GLFW failures vary by platform.
        raise RuntimeError(
            "Failed to open or run the MuJoCo interactive viewer. Check the "
            "desktop display/GLFW setup; on macOS launch Python GUI programs "
            "with mjpython when required by the MuJoCo installation."
        ) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Open an isolated MuJoCo ground + table + freejoint-object drop "
            "diagnostic without loading articulated robot bodies."
        )
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="Versioned simulation contract (default: canonical RM65 physics profile).",
    )
    parser.add_argument(
        "--object",
        default=None,
        help="Free-body object name. Required only when the contract contains several.",
    )
    parser.add_argument(
        "--table-body",
        default=DEFAULT_TABLE_BODY,
        help="World-fixed table body name in the contract URDF (default: table).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Viewer duration in seconds. 0 keeps it open until the window closes.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=1.0,
        help="Seconds to show the initial pose before each drop (default: 1).",
    )
    parser.add_argument(
        "--repeat-every",
        type=float,
        default=0.0,
        help="Reset and repeat after this many seconds. 0 performs one drop.",
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=1.0,
        help="Simulation time / wall time ratio (default: 1).",
    )
    parser.add_argument("--hide-left-ui", action="store_true")
    parser.add_argument("--hide-right-ui", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    # Fail before converting the URDF or printing a launch summary when the
    # command cannot possibly create an interactive window.
    _require_interactive_display()
    _load_mujoco_gui()
    contract_path = resolve_path(args.contract)
    contract = SimulationContract.load(contract_path)
    scene = build_isolated_drop_scene(
        contract,
        object_name=args.object,
        table_body_name=str(args.table_body),
    )
    repeat_every = float(args.repeat_every)
    summary = {
        "contract": str(contract_path),
        "source_profile": scene.source_profile_name,
        "diagnostic": "isolated_rigid_body_drop",
        "loaded_bodies": [scene.table_body_name, scene.object_spec.name],
        "object": scene.object_spec.to_dict(),
        "gravity_mps2": list(contract.scene.gravity_mps2),
        "physics_hz": contract.timing.physics_hz,
        "robot_bodies_loaded": False,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    launch_cube_drop_gui(
        scene,
        duration_sec=None if float(args.duration) <= 0.0 else float(args.duration),
        start_delay_sec=float(args.start_delay),
        repeat_every_sec=None if repeat_every <= 0.0 else repeat_every,
        realtime_factor=float(args.realtime_factor),
        show_left_ui=not bool(args.hide_left_ui),
        show_right_ui=not bool(args.hide_right_ui),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
