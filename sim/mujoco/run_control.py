from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from project_paths import resolve_path
from robot_motion.rm65 import RM65PositionCollisionPlannerConfig
from sim.contracts import SceneSpec, SimulationContract

from .config import default_camera_registry_path
from .controllers import GripperController, PoseIKController, QposController
from .ik import RM65CanonicalIK
from .mjcf import write_mjcf
from .recorder import Real2SimRecorder
from .scene import MujocoCameraRenderer, MujocoScene
from .transforms import pose_from_xyz_quat_xyzw, pose_to_xyz_quat_xyzw
from .urdf import discover_camera_mounts, parse_urdf
from .viewer import launch_passive_viewer


def _parse_qpos(values: Sequence[float] | None) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape != (6,):
        raise argparse.ArgumentTypeError("qpos must contain exactly 6 values.")
    return array


def _parse_pose(values: Sequence[float] | None) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape == (7,):
        return pose_from_xyz_quat_xyzw(array)
    if array.shape == (16,):
        return array.reshape(4, 4)
    raise argparse.ArgumentTypeError("pose must contain 7 xyz+qx+qy+qz+qw values or 16 matrix values.")


def _parse_gripper_close_items(values: Sequence[str] | None, scene: MujocoScene) -> dict[str, float]:
    if not values:
        return {}
    targets: dict[str, float] = {}
    for item in values:
        text = str(item).strip()
        if not text:
            continue
        if "=" in text:
            name, value_text = text.split("=", 1)
            gripper_name = name.strip()
        else:
            gripper_name = scene._resolve_gripper_name(None)
            value_text = text
        close = float(value_text)
        if close < 0.0 or close > 1.0:
            raise argparse.ArgumentTypeError("--gripper-close values must be in [0, 1].")
        targets[gripper_name] = close
    return targets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MuJoCo joint execution, canonical RM65 pose planning, and camera recording."
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=None,
        help=(
            "Versioned simulation contract. This is the canonical physical-scene entry; "
            "it is mutually exclusive with legacy --urdf."
        ),
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="Legacy explicit URDF entry for kinematic replay/debug.",
    )
    parser.add_argument(
        "--mode",
        choices=("dual", "right"),
        default="dual",
        help="dual controls both arms; right controls only the right arm and keeps left fixed.",
    )
    parser.add_argument(
        "--control",
        choices=("none", "qpos", "pose-ik"),
        default="qpos",
        help="Control input type.",
    )
    parser.add_argument(
        "--left-qpos",
        type=float,
        nargs=6,
        default=None,
        help="Left qpos command, or explicit planner reset state for pose-ik.",
    )
    parser.add_argument(
        "--right-qpos",
        type=float,
        nargs=6,
        default=None,
        help="Right qpos command, or explicit planner reset state for pose-ik.",
    )
    parser.add_argument(
        "--left-pose",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Left tool-TCP target in left_arm_base as xyz qx qy qz qw, "
            "or a row-major 4x4 matrix."
        ),
    )
    parser.add_argument(
        "--right-pose",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Right tool-TCP target in right_arm_base as xyz qx qy qz qw, "
            "or a row-major 4x4 matrix."
        ),
    )
    parser.add_argument(
        "--left-tool-profile",
        default=None,
        help="Explicit left RM65 tool profile name/path required by canonical pose-ik.",
    )
    parser.add_argument(
        "--right-tool-profile",
        default=None,
        help="Explicit right RM65 tool profile name/path required by canonical pose-ik.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Camera key or URDF link to render. Repeatable. Default: all discovered cameras.",
    )
    parser.add_argument(
        "--camera-registry",
        type=Path,
        default=None,
        help="Legacy --urdf camera registry override.",
    )
    parser.add_argument("--width", type=int, default=640, help="Render width.")
    parser.add_argument("--height", type=int, default=480, help="Render height.")
    parser.add_argument("--fovy", type=float, default=60.0, help="Fallback camera vertical FOV in degrees.")
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of render/record iterations; this command does not call mj_step yet.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.0,
        help="Legacy --urdf timestep. 0 uses 1/30; contracts use TimingSpec.physics_hz.",
    )
    parser.add_argument("--mjcf-out", type=Path, default=None, help="Optional path to write generated MJCF.")
    parser.add_argument("--dump-mjcf-only", action="store_true", help="Only convert URDF to MJCF and exit.")
    parser.add_argument("--record", action="store_true", help="Save frames.jsonl and rendered PNG images.")
    parser.add_argument("--record-dir", type=Path, default=None, help="Record output base directory.")
    parser.add_argument("--no-render", action="store_true", help="Skip camera rendering.")
    parser.add_argument("--viewer", action="store_true", help="Open an interactive MuJoCo GUI viewer.")
    parser.add_argument(
        "--viewer-duration",
        type=float,
        default=0.0,
        help="Seconds to keep the GUI viewer open. 0 means until the window is closed.",
    )
    parser.add_argument("--hide-left-ui", action="store_true", help="Hide MuJoCo viewer left UI panel.")
    parser.add_argument("--hide-right-ui", action="store_true", help="Hide MuJoCo viewer right UI panel.")
    parser.add_argument(
        "--gripper-close",
        action="append",
        default=None,
        help=(
            "Normalized gripper close command in [0, 1]. Use VALUE for the only "
            "gripper or NAME=VALUE when multiple grippers exist. Repeatable."
        ),
    )
    parser.add_argument(
        "--animate-gripper",
        action="store_true",
        help="In viewer mode, continuously animate selected gripper(s) open/closed.",
    )
    parser.add_argument("--gripper-period", type=float, default=3.0, help="Gripper animation period in seconds.")
    parser.add_argument("--gripper-min-close", type=float, default=0.0, help="Animation minimum close value.")
    parser.add_argument("--gripper-max-close", type=float, default=1.0, help="Animation maximum close value.")
    parser.add_argument(
        "--debug-proxy-visuals",
        action="store_true",
        help="Highlight URDF proxy geoms and place them in MuJoCo geom group 4.",
    )
    parser.add_argument("--print-camera-poses", action="store_true", help="Print camera world poses as JSON.")
    return parser


def _active_sides(mode: str) -> tuple[str, ...]:
    return ("right",) if mode == "right" else ("left", "right")


def _scene_configuration(
    args: argparse.Namespace,
) -> tuple[
    SimulationContract | None,
    Path,
    Path,
    tuple[str, ...],
    float,
    SceneSpec | None,
]:
    if args.contract is not None and args.urdf is not None:
        raise SystemExit("--contract and --urdf are mutually exclusive.")
    if args.contract is not None:
        contract = SimulationContract.load(args.contract)
        camera_names = tuple(args.camera) if args.camera else contract.observation.image_keys
        return (
            contract,
            resolve_path(contract.scene.urdf_path),
            resolve_path(contract.scene.camera_registry_path),
            camera_names,
            contract.timing.physics_timestep_sec,
            contract.scene,
        )
    if args.urdf is None:
        raise SystemExit("Provide --contract for the physical mainline or --urdf for kinematic debug.")
    legacy_dt = float(args.dt) if float(args.dt) > 0.0 else 1.0 / 30.0
    camera_names = tuple(args.camera) if args.camera else ("all",)
    registry = args.camera_registry or default_camera_registry_path()
    return (
        None,
        Path(args.urdf),
        Path(registry),
        camera_names,
        legacy_dt,
        SceneSpec.kinematic_replay(),
    )


def _qpos_targets(args: argparse.Namespace, scene: MujocoScene) -> dict[str, np.ndarray]:
    left = _parse_qpos(args.left_qpos)
    right = _parse_qpos(args.right_qpos)
    targets: dict[str, np.ndarray] = {}
    if args.mode == "dual":
        targets["left"] = left if left is not None else scene.get_arm_qpos("left")
    if "right" in scene.arm_specs:
        targets["right"] = right if right is not None else scene.get_arm_qpos("right")
    return targets


def _pose_targets(args: argparse.Namespace, scene: MujocoScene) -> dict[str, np.ndarray]:
    left = _parse_pose(args.left_pose)
    right = _parse_pose(args.right_pose)
    targets: dict[str, np.ndarray] = {}
    if args.mode == "dual":
        if left is None:
            raise SystemExit("--left-pose is required for dual canonical pose-ik.")
        targets["left"] = left
    if right is None:
        raise SystemExit("--right-pose is required for --control pose-ik.")
    targets["right"] = right
    return targets


def _print_scene_summary(
    scene: MujocoScene,
    contract: SimulationContract | None = None,
) -> None:
    summary = {
        "urdf": str(scene.conversion.urdf_path),
        "mjcf": str(scene.conversion.mjcf_path),
        "scene_profile": scene.conversion.scene_profile_name or "legacy_kinematic_urdf",
        "contract": None if contract is None else contract.to_dict(),
        "runtime_boundary": {
            "physics_stepping": False,
            "arm_actuator_model": "pending_real_150hz_system_id",
            "command_application": "joint_qpos_sink_after_canonical_group_planning",
        },
        "arms": {
            side: {
                "joints": list(spec.joint_names),
                "controlled_body": spec.controlled_body,
            }
            for side, spec in sorted(scene.arm_specs.items())
        },
        "cameras": scene.camera_metadata(),
        "grippers": scene.gripper_metadata(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _dump_mjcf_only(
    args: argparse.Namespace,
    *,
    urdf_path: Path,
    camera_registry_path: Path,
    camera_names: Sequence[str],
    timestep: float,
    scene_spec: SceneSpec | None,
) -> None:
    urdf_model = parse_urdf(urdf_path)
    cameras = discover_camera_mounts(
        urdf_model,
        camera_names=camera_names,
        registry_path=camera_registry_path,
        default_width=int(args.width),
        default_height=int(args.height),
        default_fovy_deg=float(args.fovy),
    )
    conversion = write_mjcf(
        urdf_model.path,
        camera_mounts=cameras,
        output_path=args.mjcf_out,
        timestep=timestep,
        gravity=scene_spec.gravity_mps2 if scene_spec is not None else (0.0, 0.0, 0.0),
        debug_proxy_visuals=bool(args.debug_proxy_visuals),
        scene_spec=scene_spec,
    )
    summary = {
        "urdf": str(conversion.urdf_path),
        "mjcf": str(conversion.mjcf_path),
        "scene_profile": conversion.scene_profile_name or "legacy_kinematic_urdf",
        "cameras": [
            {
                "name": camera.name,
                "link_name": camera.link_name,
                "width": int(camera.width),
                "height": int(camera.height),
                "fovy_deg": float(camera.fovy_deg),
                "source": camera.source,
            }
            for camera in cameras
        ],
        "joints": list(conversion.joint_names),
        "bodies": list(conversion.body_names),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _gripper_animation_callback(
    scene: MujocoScene,
    *,
    names: Sequence[str],
    period_s: float,
    min_close: float,
    max_close: float,
):
    gripper_names = tuple(str(name) for name in names)
    if not gripper_names:
        raise ValueError("No grippers available for animation.")
    period = max(float(period_s), 1e-6)
    lower = float(np.clip(float(min_close), 0.0, 1.0))
    upper = float(np.clip(float(max_close), 0.0, 1.0))

    def _on_step(elapsed_s: float) -> None:
        alpha = 0.5 - 0.5 * math.cos(2.0 * math.pi * float(elapsed_s) / period)
        close = lower + alpha * (upper - lower)
        scene.apply_safe_kinematic_target(
            {},
            gripper_close_by_name={name: close for name in gripper_names},
        )

    return _on_step


def run(args: argparse.Namespace) -> None:
    contract, urdf_path, camera_registry, camera_names, timestep, scene_spec = (
        _scene_configuration(args)
    )
    if args.dump_mjcf_only:
        _dump_mjcf_only(
            args,
            urdf_path=urdf_path,
            camera_registry_path=camera_registry,
            camera_names=camera_names,
            timestep=timestep,
            scene_spec=scene_spec,
        )
        return

    scene = MujocoScene.from_urdf(
        urdf_path,
        camera_names=camera_names,
        camera_registry_path=camera_registry,
        camera_width=int(args.width),
        camera_height=int(args.height),
        default_camera_fovy_deg=float(args.fovy),
        timestep=timestep,
        gravity=scene_spec.gravity_mps2 if scene_spec is not None else (0.0, 0.0, 0.0),
        mjcf_path=args.mjcf_out,
        debug_proxy_visuals=bool(args.debug_proxy_visuals),
        scene_spec=scene_spec,
    )
    missing = sorted(set(_active_sides(args.mode)) - set(scene.arm_specs))
    if args.control != "none" and missing:
        raise SystemExit(f"URDF/MJCF is missing arm specs for active side(s): {missing}")

    _print_scene_summary(scene, contract)

    extra: dict[str, object] = {"control": args.control, "mode": args.mode}
    if args.control == "qpos":
        qpos_controller = QposController(scene)
        qpos_result = qpos_controller.apply(_qpos_targets(args, scene))
        extra["qpos_result"] = qpos_result
        extra["qpos_safety"] = (
            None
            if qpos_controller.last_safety_result is None
            else qpos_controller.last_safety_result.to_dict()
        )
    elif args.control == "pose-ik":
        explicit_initial_qpos: dict[str, np.ndarray] = {}
        if args.left_qpos is not None:
            left_initial = _parse_qpos(args.left_qpos)
            assert left_initial is not None
            explicit_initial_qpos["left"] = left_initial
        if args.right_qpos is not None:
            right_initial = _parse_qpos(args.right_qpos)
            assert right_initial is not None
            explicit_initial_qpos["right"] = right_initial
        if explicit_initial_qpos:
            initial_safety = scene.apply_safe_kinematic_target(explicit_initial_qpos)
            if initial_safety is not None and initial_safety.modified:
                raise SystemExit(
                    "Explicit planner reset qpos is collision-unsafe: "
                    f"{initial_safety.to_dict()}"
                )
            extra["planner_reset_qpos_by_side"] = {
                side: values.tolist() for side, values in explicit_initial_qpos.items()
            }
        if args.left_tool_profile is None or args.right_tool_profile is None:
            raise SystemExit(
                "Canonical pose-ik requires both --left-tool-profile and "
                "--right-tool-profile; MuJoCo must not guess tool_tcp transforms."
            )
        planner_config = RM65PositionCollisionPlannerConfig(
            reference_hz=(
                float(contract.timing.reference_hz)
                if contract is not None
                else RM65PositionCollisionPlannerConfig().reference_hz
            )
        )
        canonical_ik = RM65CanonicalIK.from_tool_profiles(
            scene,
            {
                "left": str(args.left_tool_profile),
                "right": str(args.right_tool_profile),
            },
            planner_config=planner_config,
        )
        ik_result = PoseIKController(scene, canonical_ik).apply(
            _pose_targets(args, scene)
        )
        extra["ik_result"] = {
            "success": ik_result.success,
            "iterations": ik_result.iterations,
            "message": ik_result.message,
            "status": ik_result.status,
            "hold": ik_result.hold,
            "solver": ik_result.solver,
            "position_error_m": ik_result.position_error_m,
            "orientation_error_rad": ik_result.orientation_error_rad,
            "controlled_frame_by_side": ik_result.controlled_frame_by_side,
            "qpos_by_side": ik_result.qpos_by_side,
            "metadata": ik_result.metadata,
        }
        print(json.dumps(extra["ik_result"], indent=2, ensure_ascii=False))

    gripper_targets = _parse_gripper_close_items(args.gripper_close, scene)
    if gripper_targets:
        gripper_controller = GripperController(scene)
        extra["gripper_result"] = gripper_controller.apply(gripper_targets)
        extra["gripper_safety"] = (
            None
            if gripper_controller.last_safety_result is None
            else gripper_controller.last_safety_result.to_dict()
        )

    if args.print_camera_poses:
        print(json.dumps(scene.camera_metadata(), indent=2, ensure_ascii=False))

    if args.viewer:
        animate_names: tuple[str, ...] = ()
        on_step = None
        if bool(args.animate_gripper):
            animate_names = tuple(gripper_targets) if gripper_targets else tuple(sorted(scene.gripper_specs))
            on_step = _gripper_animation_callback(
                scene,
                names=animate_names,
                period_s=float(args.gripper_period),
                min_close=float(args.gripper_min_close),
                max_close=float(args.gripper_max_close),
            )
        launch_passive_viewer(
            scene,
            duration_sec=None if float(args.viewer_duration) <= 0.0 else float(args.viewer_duration),
            realtime_hz=1.0 / timestep,
            show_left_ui=not bool(args.hide_left_ui),
            show_right_ui=not bool(args.hide_right_ui),
            on_step=on_step,
        )
        if animate_names:
            extra["gripper_animation"] = {
                "names": list(animate_names),
                "period_s": float(args.gripper_period),
                "min_close": float(args.gripper_min_close),
                "max_close": float(args.gripper_max_close),
            }

    renderer = None
    recorder = None
    try:
        if not args.no_render:
            try:
                renderer = MujocoCameraRenderer(scene, width=int(args.width), height=int(args.height))
            except Exception as exc:  # noqa: BLE001 - give a practical MuJoCo GL hint.
                raise RuntimeError(
                    "Failed to create MuJoCo renderer. In headless sessions try "
                    "`MUJOCO_GL=egl` or `MUJOCO_GL=osmesa`."
                ) from exc
        if args.record:
            recorder = Real2SimRecorder.create_timestamped(
                scene,
                root_dir=args.record_dir,
                save_images=not bool(args.no_render),
            )
            print(f"record_dir={recorder.root_dir}")
        for step in range(int(args.steps)):
            images = renderer.render_all() if renderer is not None else {}
            if recorder is not None:
                recorder.record_step(images=images, extra={**extra, "step": step})
        if not args.record and renderer is not None:
            print(f"rendered cameras={list(scene.camera_names)}")
    finally:
        if recorder is not None:
            recorder.close()
        if renderer is not None:
            renderer.close()

    final = {
        "qpos_by_side": scene.arm_qpos_by_side(),
        "controlled_body_pose_world_xyz_quat_xyzw": {
            side: pose_to_xyz_quat_xyzw(scene.body_pose(spec.controlled_body))
            for side, spec in sorted(scene.arm_specs.items())
        },
        "gripper_close_by_name": scene.gripper_close_by_name(),
        "gripper_qpos_by_name": scene.gripper_qpos_by_name(),
    }
    print(json.dumps(final, indent=2, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
