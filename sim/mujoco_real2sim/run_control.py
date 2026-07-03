from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import IKConfig, default_camera_registry_path
from .controllers import PoseIKController, QposController
from .ik import DampedLeastSquaresIK
from .recorder import Real2SimRecorder
from .scene import MujocoCameraRenderer, MujocoScene
from .transforms import pose_from_xyz_quat_xyzw, pose_to_xyz_quat_xyzw
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MuJoCo real2sim qpos / pose-IK control and camera recording."
    )
    parser.add_argument("--urdf", required=True, type=Path, help="Explicit URDF path to load.")
    parser.add_argument(
        "--mode",
        choices=("dual", "right"),
        default="dual",
        help="dual controls both arms; right controls only the right arm and keeps left fixed.",
    )
    parser.add_argument(
        "--control",
        choices=("qpos", "pose-ik"),
        default="qpos",
        help="Control input type.",
    )
    parser.add_argument("--left-qpos", type=float, nargs=6, default=None)
    parser.add_argument("--right-qpos", type=float, nargs=6, default=None)
    parser.add_argument(
        "--left-pose",
        type=float,
        nargs="+",
        default=None,
        help="Left target pose as xyz qx qy qz qw, or a row-major 4x4 matrix.",
    )
    parser.add_argument(
        "--right-pose",
        type=float,
        nargs="+",
        default=None,
        help="Right target pose as xyz qx qy qz qw, or a row-major 4x4 matrix.",
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
        default=default_camera_registry_path(),
        help="Camera registry JSON used to map logical camera keys to URDF links.",
    )
    parser.add_argument("--width", type=int, default=640, help="Render width.")
    parser.add_argument("--height", type=int, default=480, help="Render height.")
    parser.add_argument("--fovy", type=float, default=60.0, help="Fallback camera vertical FOV in degrees.")
    parser.add_argument("--steps", type=int, default=1, help="Number of frames to simulate/record.")
    parser.add_argument("--dt", type=float, default=1.0 / 30.0, help="MuJoCo timestep.")
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
    parser.add_argument("--print-camera-poses", action="store_true", help="Print camera world poses as JSON.")
    parser.add_argument("--ik-max-iterations", type=int, default=80)
    parser.add_argument("--ik-position-tolerance", type=float, default=2e-3)
    parser.add_argument("--ik-orientation-tolerance", type=float, default=2e-2)
    parser.add_argument("--ik-damping", type=float, default=5e-3)
    parser.add_argument("--ik-max-joint-step", type=float, default=0.04)
    parser.add_argument("--ik-orientation-weight", type=float, default=0.35)
    return parser


def _active_sides(mode: str) -> tuple[str, ...]:
    return ("right",) if mode == "right" else ("left", "right")


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
            left = scene.body_pose(scene.arm_specs["left"].controlled_body)
        targets["left"] = left
    if right is None:
        raise SystemExit("--right-pose is required for --control pose-ik.")
    targets["right"] = right
    return targets


def _print_scene_summary(scene: MujocoScene) -> None:
    summary = {
        "urdf": str(scene.conversion.urdf_path),
        "mjcf": str(scene.conversion.mjcf_path),
        "arms": {
            side: {
                "joints": list(spec.joint_names),
                "controlled_body": spec.controlled_body,
            }
            for side, spec in sorted(scene.arm_specs.items())
        },
        "cameras": scene.camera_metadata(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run(args: argparse.Namespace) -> None:
    camera_names = tuple(args.camera) if args.camera else ("all",)
    scene = MujocoScene.from_urdf(
        args.urdf,
        camera_names=camera_names,
        camera_registry_path=args.camera_registry,
        camera_width=int(args.width),
        camera_height=int(args.height),
        default_camera_fovy_deg=float(args.fovy),
        timestep=float(args.dt),
        gravity=(0.0, 0.0, 0.0),
        mjcf_path=args.mjcf_out,
    )
    missing = sorted(set(_active_sides(args.mode)) - set(scene.arm_specs))
    if missing:
        raise SystemExit(f"URDF/MJCF is missing arm specs for active side(s): {missing}")

    _print_scene_summary(scene)
    if args.dump_mjcf_only:
        return

    ik_config = IKConfig(
        max_iterations=int(args.ik_max_iterations),
        position_tolerance_m=float(args.ik_position_tolerance),
        orientation_tolerance_rad=float(args.ik_orientation_tolerance),
        damping=float(args.ik_damping),
        max_joint_step_rad=float(args.ik_max_joint_step),
        orientation_weight=float(args.ik_orientation_weight),
    )
    extra: dict[str, object] = {"control": args.control, "mode": args.mode}
    if args.control == "qpos":
        qpos_result = QposController(scene).apply(_qpos_targets(args, scene))
        extra["qpos_result"] = qpos_result
    else:
        ik_result = PoseIKController(scene, DampedLeastSquaresIK(scene, ik_config)).apply(
            _pose_targets(args, scene)
        )
        extra["ik_result"] = {
            "success": ik_result.success,
            "iterations": ik_result.iterations,
            "message": ik_result.message,
            "position_error_m": ik_result.position_error_m,
            "orientation_error_rad": ik_result.orientation_error_rad,
            "qpos_by_side": ik_result.qpos_by_side,
        }
        print(json.dumps(extra["ik_result"], indent=2, ensure_ascii=False))

    if args.print_camera_poses:
        print(json.dumps(scene.camera_metadata(), indent=2, ensure_ascii=False))

    if args.viewer:
        launch_passive_viewer(
            scene,
            duration_sec=None if float(args.viewer_duration) <= 0.0 else float(args.viewer_duration),
            realtime_hz=1.0 / float(args.dt),
            show_left_ui=not bool(args.hide_left_ui),
            show_right_ui=not bool(args.hide_right_ui),
        )

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
    }
    print(json.dumps(final, indent=2, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
