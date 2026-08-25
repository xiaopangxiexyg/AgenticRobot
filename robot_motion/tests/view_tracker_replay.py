from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from sim.contracts.scene import DEFAULT_RM65_PHYSICS_URDF
from sim.mujoco.config import default_camera_registry_path
from sim.mujoco.scene import MujocoScene

from .tracker_replay import (
    OfflineTrackerReplayConfig,
    OfflineTrackerReplayFrame,
    OfflineTrackerReplayReport,
    run_offline_tracker_replay,
)


SIDES = ("left", "right")
REPO_ROOT = Path(__file__).resolve().parents[2]
# Keep the viewer aligned with the canonical MuJoCo task-physics model;
# collision avoidance remains a separately selected robot_motion scene profile.
DEFAULT_VIEWER_URDF = REPO_ROOT / DEFAULT_RM65_PHYSICS_URDF
FALLBACK_VIEWER_URDF = DEFAULT_VIEWER_URDF


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay recorded tracker-derived RM65 IK outputs in a MuJoCo viewer."
    )
    parser.add_argument("--dataset-root", required=True, type=Path, help="LeRobot-style dataset root.")
    parser.add_argument("--episode", default="0", help="Episode index/id, e.g. 0 or episode_000000.")
    parser.add_argument("--start-frame", type=int, default=0, help="Sorted row offset to start from.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to replay; 0 means all.")
    parser.add_argument(
        "--active-sides",
        default="left,right",
        help="Comma-separated sides to replay through IK. Default: left,right.",
    )
    parser.add_argument(
        "--initial-qpos-source",
        choices=("first_observation", "calibration_home", "default"),
        default="first_observation",
        help="Initial solver qpos for offline tracker replay.",
    )
    parser.add_argument("--disable-target-smoothing", action="store_true")
    parser.add_argument("--disable-realistic-joint-step", action="store_true")
    parser.add_argument(
        "--solve-max-iterations",
        type=int,
        default=1,
        help="Dual-arm IK solve iterations per frame. Default matches runtime.",
    )
    parser.add_argument(
        "--seed-each-frame-from-observation",
        action="store_true",
        help="Reset solver qpos from the dataset observation qpos before each frame.",
    )
    parser.add_argument("--scene-profile", default="", help="Override collision scene profile.")
    parser.add_argument("--contact-mode", default="", help="Override collision avoidance mode.")
    parser.add_argument("--solver-frequency", type=float, default=0.0, help="Override dual-arm IK Hz.")
    parser.add_argument("--ik-tuning-json", default="", help="Optional IK tuning JSON override.")
    parser.add_argument(
        "--qpos-source",
        choices=("tracker_ik", "dataset_observation"),
        default="tracker_ik",
        help="Qpos stream shown in MuJoCo. tracker_ik uses the offline tracker IK result.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help=(
            "URDF loaded by MuJoCo. Default: "
            f"{DEFAULT_RM65_PHYSICS_URDF}"
        ),
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Camera key or URDF link to register in the scene. Repeatable. Default: all discovered cameras.",
    )
    parser.add_argument(
        "--camera-registry",
        type=Path,
        default=default_camera_registry_path(),
        help="Camera registry JSON used to map logical camera keys to URDF links.",
    )
    parser.add_argument("--width", type=int, default=640, help="Registered camera width.")
    parser.add_argument("--height", type=int, default=480, help="Registered camera height.")
    parser.add_argument("--fovy", type=float, default=60.0, help="Fallback camera vertical FOV in degrees.")
    parser.add_argument("--dt", type=float, default=1.0 / 30.0, help="MuJoCo timestep.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--no-sleep", action="store_true", help="Run as fast as possible.")
    parser.add_argument("--loop", action="store_true", help="Loop the selected frames until the viewer closes.")
    parser.add_argument("--dry-run", action="store_true", help="Build replay and MuJoCo scene without opening the viewer.")
    parser.add_argument(
        "--viewer-duration",
        type=float,
        default=0.0,
        help="Seconds to keep the GUI viewer open. 0 means until replay ends or the window is closed.",
    )
    parser.add_argument("--hide-left-ui", action="store_true", help="Hide MuJoCo viewer left UI panel.")
    parser.add_argument("--hide-right-ui", action="store_true", help="Hide MuJoCo viewer right UI panel.")
    parser.add_argument("--mjcf-out", type=Path, default=None, help="Optional path to write generated MJCF.")
    parser.add_argument("--print-every", type=int, default=30, help="Progress print period in frames.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


def run(args: argparse.Namespace) -> None:
    if float(args.speed) <= 0.0:
        raise SystemExit("--speed must be positive.")
    if float(args.dt) <= 0.0:
        raise SystemExit("--dt must be positive.")

    active_sides = _parse_active_sides(args.active_sides)
    report = run_offline_tracker_replay(
        OfflineTrackerReplayConfig(
            dataset_root=args.dataset_root,
            episode=args.episode,
            start_frame=int(args.start_frame),
            max_frames=int(args.max_frames),
            active_sides=active_sides,
            use_target_smoothing=not bool(args.disable_target_smoothing),
            enforce_realistic_joint_step=not bool(args.disable_realistic_joint_step),
            initial_qpos_source=str(args.initial_qpos_source),
            solve_max_iterations=int(args.solve_max_iterations),
            seed_each_frame_from_observation=bool(args.seed_each_frame_from_observation),
            scene_profile=str(args.scene_profile or "") or None,
            contact_mode=str(args.contact_mode or "") or None,
            solver_frequency=float(args.solver_frequency) if float(args.solver_frequency) > 0.0 else None,
            ik_tuning_json_path=str(args.ik_tuning_json or "") or None,
        )
    )
    scene = _load_scene(args, active_sides)
    _initialize_scene_qpos(scene, report, active_sides=active_sides, qpos_source=str(args.qpos_source))
    _print_start_summary(report, scene=scene, args=args)
    if bool(args.dry_run):
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "initial_qpos_by_side": scene.arm_qpos_by_side(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    _play_report_in_mujoco(scene, report, args=args, active_sides=active_sides)


def _parse_active_sides(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = str(value).replace("both", "left,right").split(",")
    else:
        raw = list(value)
    sides = tuple(side.strip().lower() for side in raw if side.strip())
    unknown = sorted(set(sides) - set(SIDES))
    if unknown:
        raise SystemExit(f"--active-sides must be within {SIDES}, got {unknown}.")
    if not sides:
        raise SystemExit("--active-sides cannot be empty.")
    return sides


def _resolve_viewer_urdf(value: Path | None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    for candidate in (DEFAULT_VIEWER_URDF, FALLBACK_VIEWER_URDF):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "No default RM65 MuJoCo viewer URDF was found. Pass --urdf explicitly."
    )


def _load_scene(args: argparse.Namespace, active_sides: Sequence[str]) -> MujocoScene:
    urdf_path = _resolve_viewer_urdf(args.urdf)
    camera_names = tuple(args.camera) if args.camera else ("all",)
    scene = MujocoScene.from_urdf(
        urdf_path,
        camera_names=camera_names,
        camera_registry_path=args.camera_registry,
        camera_width=int(args.width),
        camera_height=int(args.height),
        default_camera_fovy_deg=float(args.fovy),
        timestep=float(args.dt),
        gravity=(0.0, 0.0, 0.0),
        mjcf_path=args.mjcf_out,
    )
    missing = sorted(set(active_sides) - set(scene.arm_specs))
    if missing:
        raise SystemExit(f"URDF/MJCF is missing arm specs for active side(s): {missing}")
    return scene


def _initialize_scene_qpos(
    scene: MujocoScene,
    report: OfflineTrackerReplayReport,
    *,
    active_sides: Sequence[str],
    qpos_source: str,
) -> None:
    if not report.frames:
        raise ValueError("Offline tracker report contains no frames.")
    qpos = _frame_qpos_by_side(
        report.frames[0],
        active_sides=active_sides,
        qpos_source=qpos_source,
    )
    if qpos:
        scene.set_qpos_by_side(qpos)


def _play_report_in_mujoco(
    scene: MujocoScene,
    report: OfflineTrackerReplayReport,
    *,
    args: argparse.Namespace,
    active_sides: Sequence[str],
) -> None:
    import mujoco.viewer

    viewer_handle = mujoco.viewer.launch_passive(
        scene.model,
        scene.data,
        show_left_ui=not bool(args.hide_left_ui),
        show_right_ui=not bool(args.hide_right_ui),
    )
    started_at = time.time()
    processed = 0
    try:
        while True:
            loop_started_at = time.time()
            first_frame_time = float(report.frames[0].frame_time_sec)
            for frame in report.frames:
                if not viewer_handle.is_running():
                    return
                if float(args.viewer_duration) > 0.0:
                    if time.time() - started_at >= float(args.viewer_duration):
                        return
                playback_time = max(0.0, float(frame.frame_time_sec) - first_frame_time)
                _wait_for_frame(
                    replay_started_at=loop_started_at,
                    playback_time_sec=playback_time,
                    speed=float(args.speed),
                    no_sleep=bool(args.no_sleep),
                )
                qpos_by_side = _frame_qpos_by_side(
                    frame,
                    active_sides=active_sides,
                    qpos_source=str(args.qpos_source),
                )
                if qpos_by_side:
                    scene.set_qpos_by_side(qpos_by_side)
                viewer_handle.sync()
                processed += 1
                if _should_print(processed=processed, total=len(report.frames), print_every=int(args.print_every)):
                    print(_progress_line(frame, processed=processed, total=len(report.frames), args=args))
            if not bool(args.loop):
                return
    finally:
        _close_viewer(viewer_handle)
        print(
            json.dumps(
                {
                    "processed_frames": processed,
                    "final_qpos_by_side": scene.arm_qpos_by_side(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )


def _wait_for_frame(
    *,
    replay_started_at: float,
    playback_time_sec: float,
    speed: float,
    no_sleep: bool,
) -> None:
    if no_sleep:
        return
    target = replay_started_at + playback_time_sec / max(float(speed), 1e-6)
    remaining = target - time.time()
    if remaining > 0.0:
        time.sleep(remaining)


def _close_viewer(viewer_handle) -> None:
    try:
        viewer_handle.close()
    except Exception:
        return
    time.sleep(0.05)


def _frame_qpos_by_side(
    frame: OfflineTrackerReplayFrame,
    *,
    active_sides: Sequence[str],
    qpos_source: str,
) -> dict[str, np.ndarray]:
    if qpos_source == "tracker_ik":
        source = frame.replay_qpos
    elif qpos_source == "dataset_observation":
        source = frame.original_qpos
    else:
        raise ValueError(f"Unsupported qpos source: {qpos_source}")
    out: dict[str, np.ndarray] = {}
    for side in active_sides:
        values = source.get(side)
        if values is None:
            continue
        out[side] = np.asarray(values, dtype=float).reshape(6)
    return out


def _print_start_summary(
    report: OfflineTrackerReplayReport,
    *,
    scene: MujocoScene,
    args: argparse.Namespace,
) -> None:
    print(
        json.dumps(
            {
                "dataset_root": str(report.dataset_root),
                "episode_id": report.episode_id,
                "frame_count": report.frame_count,
                "source_frame_count": report.source_frame_count,
                "nominal_hz": report.nominal_hz,
                "timing_source": report.timing_source,
                "qpos_source": str(args.qpos_source),
                "urdf": str(scene.conversion.urdf_path),
                "mjcf": str(scene.conversion.mjcf_path),
                "arms": {
                    side: {
                        "joints": list(spec.joint_names),
                        "controlled_body": spec.controlled_body,
                    }
                    for side, spec in sorted(scene.arm_specs.items())
                },
                "summary": _summary_excerpt(report.summary),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _summary_excerpt(summary: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {
        "status_counts": summary.get("status_counts"),
        "fallback_reason_counts": summary.get("fallback_reason_counts"),
        "initial_qpos_source": summary.get("initial_qpos_source"),
        "target_smoothing_enabled": summary.get("target_smoothing_enabled"),
        "scene_profile": summary.get("scene_profile"),
    }
    by_side = summary.get("by_side")
    if isinstance(by_side, Mapping):
        out["by_side"] = by_side
    return out


def _should_print(*, processed: int, total: int, print_every: int) -> bool:
    if print_every <= 0:
        return False
    return processed == 1 or processed % print_every == 0 or processed == total


def _progress_line(
    frame: OfflineTrackerReplayFrame,
    *,
    processed: int,
    total: int,
    args: argparse.Namespace,
) -> str:
    side_bits: list[str] = []
    for side in SIDES:
        error = frame.errors.get(side)
        if error is None:
            continue
        pos = error.translation_error_m
        rot = error.rotation_error_deg
        if pos is None and rot is None:
            continue
        side_bits.append(
            f"{side}:ik_pos={_format_number(pos, 'm')} ik_rot={_format_number(rot, 'deg')}"
        )
    suffix = " ".join(side_bits)
    if suffix:
        suffix = " " + suffix
    return (
        f"tracker viewer frame={processed}/{total} dataset_frame={frame.frame_index} "
        f"t={frame.frame_time_sec:.3f}s status={frame.status} qpos={args.qpos_source}{suffix}"
    )


def _format_number(value: float | None, unit: str) -> str:
    if value is None:
        return "n/a"
    if unit == "m":
        return f"{float(value):.4f}m"
    if unit == "deg":
        return f"{float(value):.2f}deg"
    return f"{float(value):.4g}{unit}"


if __name__ == "__main__":
    raise SystemExit(main())
