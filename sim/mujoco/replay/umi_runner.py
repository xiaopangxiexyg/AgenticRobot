"""Replay canonical desk/tool-TCP actions through physical MuJoCo control.

The module name is retained for command compatibility with UMI workflows.  The
default package entrypoint is now ``python -m sim.mujoco.replay``.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

from robot_motion.position_control import resample_motion_targets
from robot_motion.rm65 import (
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_POSITION_CONTROL_PROFILE_CHOICES,
)
from sim.mujoco.control.desk_tcp_position import (
    DeskTcpPositionControl,
    DeskTcpPositionControlConfig,
)
from sim.mujoco.viewer import passive_viewer_handle

from .umi_dataset import load_umi_replay_episode
from .tcp_target_visualization import (
    TcpTargetVisualizationConfig,
    TcpTargetVisualizer,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UMI_DATASET_ROOT = (
    _REPO_ROOT / "remote_data/umi/2026-07-19_umi_dual_wrist_debug_remote_raw"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical 16D/26D desk TCP action -> 60 Hz coupled IK -> "
            "120 Hz actuator command -> "
            "1000 Hz MuJoCo replay. Tactile remains synchronized metadata only."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_UMI_DATASET_ROOT)
    parser.add_argument("--episode", default="0")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument("--left-tool-profile", type=Path, default=None)
    parser.add_argument("--right-tool-profile", type=Path, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--tcp-target-visualization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Show left/right target TCP positions and RGB orientation axes in the "
            "MuJoCo viewer (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Run deterministic physics without wall-clock pacing.",
    )
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--status-hz", type=float, default=2.0)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--position-control-profile",
        choices=RM65_POSITION_CONTROL_PROFILE_CHOICES,
        default=RM65_BIMANUAL_POSITION_V1_NAME,
        help=(
            "Position execution profile. The formal profile consumes the already-shaped "
            "main action without repeating collection-side qref smoothing; legacy restores "
            "the previous replay behavior."
        ),
    )
    args = parser.parse_args(argv)
    if args.start_frame < 0:
        parser.error("--start-frame must be >= 0")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.settle_seconds < 0.0:
        parser.error("--settle-seconds must be >= 0")
    if args.status_hz <= 0.0:
        parser.error("--status-hz must be positive")
    return args


def _control_config(args: argparse.Namespace) -> DeskTcpPositionControlConfig:
    base = DeskTcpPositionControlConfig(realtime_worker=False)
    values = {"position_control_profile_name": args.position_control_profile}
    if args.contract is not None:
        values["simulation_contract_path"] = args.contract
    if args.left_tool_profile is not None:
        values["left_tool_profile"] = args.left_tool_profile
    if args.right_tool_profile is not None:
        values["right_tool_profile"] = args.right_tool_profile
    return DeskTcpPositionControlConfig(**{**base.__dict__, **values})


def _advance_to(
    control: DeskTcpPositionControl,
    target_time: float,
    *,
    viewer_handle,
    no_sleep: bool,
) -> bool:
    dt = control.contract.timing.physics_timestep_sec
    while float(control.scene.data.time) + 0.5 * dt < target_time:
        if viewer_handle is not None and not viewer_handle.is_running():
            return False
        started = time.monotonic()
        control.step_physics()
        control.rig.sync_viewer(viewer_handle)
        if not no_sleep:
            remaining = dt - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)
    return True


def run(args: argparse.Namespace) -> dict[str, object]:
    episode = load_umi_replay_episode(
        args.dataset_root,
        args.episode,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
    )
    references = resample_motion_targets(
        episode.motion_targets(), reference_hz=60.0, include_final=True
    )
    control = DeskTcpPositionControl.build(_control_config(args))
    tcp_target_visualizer = TcpTargetVisualizer(
        TcpTargetVisualizationConfig(enabled=bool(args.tcp_target_visualization))
    )
    started = time.monotonic()
    completed_references = 0
    try:
        viewer_context = (
            passive_viewer_handle(control.scene) if args.viewer else nullcontext(None)
        )
        with viewer_context as viewer_handle:
            print(
                f"[canonical-action-replay] episode={episode.episode.episode_id} "
                f"source_frames={len(episode.frames)} references={len(references)} "
                f"action_dim={episode.action_dim} rates=15/60/120/1000Hz "
                "resampling=offline_timestamp_linear_slerp"
            )
            print(
                "[canonical-action-replay] pose=canonical desk_absolute tool_tcp; "
                "raw_to_desk is intentionally not applied"
            )
            print(
                "[canonical-action-replay] mujoco_scene=inspire_gripper_v3 "
                f"collision_ik={control.retargeter.profile.name}/"
                f"{len(control.retargeter.profile.collision_pairs)}pairs "
                f"tcp=DMTAC_M target_viz={'on' if args.tcp_target_visualization else 'off'}"
            )
            print(
                "[canonical-action-replay] tactile=metadata_only "
                f"storage={dict(episode.tactile_storage_by_side)}"
            )
            next_status = 0.0
            for reference in references:
                if not _advance_to(
                    control,
                    float(reference.timestamp),
                    viewer_handle=viewer_handle,
                    no_sleep=bool(args.no_sleep),
                ):
                    break
                grippers = episode.interpolated_gripper_targets(
                    left_source_index=reference.left_source_index,
                    right_source_index=reference.right_source_index,
                    alpha=reference.interpolation_alpha,
                )
                result = control.solve_and_dispatch(
                    reference, gripper_close_by_side=grippers
                )
                if viewer_handle is not None and tcp_target_visualizer.config.enabled:
                    tcp_target_visualizer.update(
                        viewer_handle,
                        control.tool_tcp_world_targets(reference),
                    )
                completed_references += 1
                if float(reference.timestamp) + 1e-12 >= next_status:
                    print(
                        f"[canonical-action-replay] t={reference.timestamp:.3f}s "
                        f"source={reference.left_source_index}->{reference.right_source_index} "
                        f"alpha={reference.interpolation_alpha:.3f} ik={result.status} "
                        f"fallback={result.fallback_reason or '-'}"
                    )
                    next_status += 1.0 / float(args.status_hz)
            if completed_references:
                end_time = float(references[completed_references - 1].timestamp) + float(
                    args.settle_seconds
                )
                _advance_to(
                    control,
                    end_time,
                    viewer_handle=viewer_handle,
                    no_sleep=bool(args.no_sleep),
                )
            snapshot = control.snapshot()
    finally:
        control.close()
    summary = {
        "schema": "dexteleop.sim.mujoco.canonical_action_replay_summary.v1",
        "dataset_root": str(episode.episode.dataset_root),
        "episode_id": episode.episode.episode_id,
        "source_frame_count": len(episode.frames),
        "source_action_dim": episode.action_dim,
        "replay_role": "canonical_cartesian_action_mainline",
        "qpos_replay_used": False,
        "reference_count_planned": len(references),
        "reference_count_completed": completed_references,
        "wall_time_sec": time.monotonic() - started,
        "resampling": "offline_timestamp_linear_xyz_quaternion_slerp",
        "pose_storage_frame": episode.pose_frame_contract.pose_storage_frame,
        "raw_to_desk_applied": False,
        "tactile_role": "synchronized_metadata_only",
        "tactile_storage_by_side": dict(episode.tactile_storage_by_side),
        "tcp_target_visualization": {
            "enabled": bool(args.tcp_target_visualization),
            "viewer_active": bool(args.viewer and args.tcp_target_visualization),
            "geometry": "left/right origin spheres plus RGB XYZ arrows",
        },
        "control": snapshot,
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(
        f"[canonical-action-replay] done references={completed_references}/{len(references)} "
        f"ik={snapshot['ik_status_counts']} wall={summary['wall_time_sec']:.2f}s"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_UMI_DATASET_ROOT", "main", "parse_args", "run"]
