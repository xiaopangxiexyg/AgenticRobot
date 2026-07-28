"""Visualize versioned RM65 Pose Lift replay trajectories in MuJoCo.

This is an offline debug entrypoint.  It loads a recorded pose action, uses the
saved accepted q_ref for the first-frame initialization, resolves subsequent
60 Hz pose references through the same ``RM65PositionCollisionPlanner`` used
by pose replay/deploy, and only then displays the admitted qpos in MuJoCo.

MuJoCo is a trajectory viewer here.  It does not solve IK again, project the
planner output, open a device, or dispatch a real command.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from robot_motion.position_control import TimestampedMotionTarget, resample_motion_targets
from robot_motion.rm65 import (
    DEFAULT_COLLISION_SCENE,
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_POSE_LIFT_PROFILE_CHOICES,
    RM65_POSE_LIFT_V1_BASELINE_NAME,
    RM65PositionCollisionPlanner,
    RM65PositionCollisionPlannerConfig,
    build_rm65_pose_lift_ik_tuning,
    endpoint_to_tool_transform,
    load_rm65_tool_profile,
)
from robot_motion.replay import ReplayGroupSpec
from robot_motion.transforms import parse_pose_matrix
from teleop_runtime.replay.executor import (
    REPLAY_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S,
    REPLAY_POSITION_IK_DAMPING,
    REPLAY_POSITION_IK_ITERATION_RESERVE_MS,
    REPLAY_POSITION_IK_ORIENTATION_WEIGHT,
    REPLAY_POSITION_IK_SOLVE_MAX_ITERATIONS,
    REPLAY_POSITION_IK_SOLVE_MIN_ITERATIONS,
)
from teleop_runtime.replay.pose_frames import (
    replay_action_poses_in_hardware_base_at_index,
)
from teleop_runtime.replay.trajectory import (
    build_replay_group_for_episode,
    read_replay_dataset_episode_indices,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIEW_URDF = (
    _REPO_ROOT
    / "assets/rm65_inspire_gripper_v3/"
    "rm65_dual_arm_wrist_cameras_v3_head_camera.urdf"
)
DEFAULT_TOOL_PROFILE_BY_SIDE = {
    side: (
        _REPO_ROOT
        / f"hardware/rm65/utils/configs/calibrations/rm65_{side}_DMTAC_M.json"
    )
    for side in ("left", "right")
}


@dataclass(frozen=True)
class PoseLiftSimFrame:
    sequence: int
    timestamp: float
    source_frame_index: int
    interpolation_alpha: float
    status: str
    hold: bool
    qpos_by_side: Mapping[str, np.ndarray]
    modification_reasons: tuple[str, ...]
    pose_lift_reason: str
    collision_distance_m: float | None


@dataclass(frozen=True)
class PoseLiftSimTrajectory:
    profile_name: str
    source_episode_index: int
    source_path: str
    source_frame_count: int
    reference_hz: float
    initial_qpos_by_side: Mapping[str, np.ndarray]
    frames: tuple[PoseLiftSimFrame, ...]
    status_counts: Mapping[str, int]
    hold_count: int
    pose_lift_applied_count: int

    def summary(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "source_episode_index": int(self.source_episode_index),
            "source_path": self.source_path,
            "source_frame_count": int(self.source_frame_count),
            "reference_hz": float(self.reference_hz),
            "reference_count": len(self.frames),
            "duration_sec": (
                0.0 if not self.frames else float(self.frames[-1].timestamp)
            ),
            "status_counts": dict(self.status_counts),
            "hold_count": int(self.hold_count),
            "pose_lift_applied_count": int(self.pose_lift_applied_count),
        }


def _finite_positive(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return parsed


def _finite_non_negative(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("expected a finite non-negative number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _one_or_six_positive_floats(value: str) -> tuple[float, ...]:
    parts = [part for part in str(value).replace(",", " ").split() if part]
    if len(parts) not in {1, 6}:
        raise argparse.ArgumentTypeError("expected one value or six comma-separated values")
    parsed = tuple(float(part) for part in parts)
    if not np.all(np.isfinite(parsed)) or np.any(np.asarray(parsed) <= 0.0):
        raise argparse.ArgumentTypeError("joint limits must be finite and positive")
    return parsed * 6 if len(parsed) == 1 else parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--episode",
        type=int,
        help="Dataset episode index. Use --list-episodes to inspect available values.",
    )
    parser.add_argument(
        "--list-episodes",
        action="store_true",
        help="List episode indices and exit without building IK or MuJoCo.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum source action frames; 0 means through the episode end.",
    )
    parser.add_argument(
        "--pose-lift-profile",
        action="append",
        choices=RM65_POSE_LIFT_PROFILE_CHOICES,
        default=None,
        help=(
            "Pose Lift version to play. Repeat to compare profiles sequentially. "
            "Default: rm65_pose_lift_v1_baseline."
        ),
    )
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Replay all registered Pose Lift profiles sequentially from the same q_ref.",
    )
    parser.add_argument(
        "--left-tool-profile",
        type=Path,
        default=DEFAULT_TOOL_PROFILE_BY_SIDE["left"],
    )
    parser.add_argument(
        "--right-tool-profile",
        type=Path,
        default=DEFAULT_TOOL_PROFILE_BY_SIDE["right"],
    )
    parser.add_argument("--scene-profile", default=DEFAULT_COLLISION_SCENE)
    parser.add_argument("--reference-hz", type=_finite_positive, default=60.0)
    parser.add_argument(
        "--orientation-weight",
        type=_finite_non_negative,
        default=REPLAY_POSITION_IK_ORIENTATION_WEIGHT,
    )
    parser.add_argument(
        "--posture-cost",
        type=_finite_non_negative,
        default=None,
        help="Override the selected profile's IK posture cost.",
    )
    parser.add_argument(
        "--damping",
        type=_finite_positive,
        default=REPLAY_POSITION_IK_DAMPING,
    )
    parser.add_argument(
        "--solve-min-iterations",
        type=_positive_int,
        default=REPLAY_POSITION_IK_SOLVE_MIN_ITERATIONS,
    )
    parser.add_argument(
        "--solve-max-iterations",
        type=_positive_int,
        default=REPLAY_POSITION_IK_SOLVE_MAX_ITERATIONS,
    )
    parser.add_argument(
        "--fixed-solve-iterations",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Offline deterministic override: set min=max=N and disable adaptive stopping.",
    )
    parser.add_argument(
        "--adaptive-iterations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--solve-iteration-reserve-ms",
        type=_finite_non_negative,
        default=REPLAY_POSITION_IK_ITERATION_RESERVE_MS,
    )
    parser.add_argument(
        "--max-joint-velocity-rad-s",
        type=_one_or_six_positive_floats,
        default=REPLAY_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S,
        metavar="V[,V,V,V,V,V]",
    )
    parser.add_argument(
        "--max-joint-step-rad",
        type=_one_or_six_positive_floats,
        default=None,
        metavar="S[,S,S,S,S,S]",
    )
    parser.add_argument(
        "--collision-min-distance-m",
        type=_finite_non_negative,
        default=0.03,
    )
    parser.add_argument(
        "--collision-distance-epsilon-m",
        type=_finite_non_negative,
        default=0.0001,
    )
    parser.add_argument(
        "--debug-profile",
        action="store_true",
        help="Enable planner timing/profile metadata while generating the trajectory.",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_VIEW_URDF)
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the passive MuJoCo viewer (default: enabled).",
    )
    parser.add_argument("--speed", type=_finite_positive, default=1.0)
    parser.add_argument("--no-sleep", action="store_true")
    parser.add_argument("--repeat", type=_positive_int, default=1)
    parser.add_argument(
        "--pause-between-profiles-sec",
        type=_finite_non_negative,
        default=0.5,
    )
    parser.add_argument("--hold-viewer-at-end", action="store_true")
    parser.add_argument("--hide-left-ui", action="store_true")
    parser.add_argument("--hide-right-ui", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--trace-jsonl",
        type=Path,
        default=None,
        help="Write every admitted qpos reference and its bounded diagnostics.",
    )
    return parser


def selected_profiles(args: argparse.Namespace) -> tuple[str, ...]:
    requested = (
        tuple(RM65_POSE_LIFT_PROFILE_CHOICES)
        if bool(args.all_profiles)
        else tuple(args.pose_lift_profile or (RM65_POSE_LIFT_V1_BASELINE_NAME,))
    )
    return tuple(dict.fromkeys(str(value) for value in requested))


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.episode is None and not bool(args.list_episodes):
        parser.error("--episode is required unless --list-episodes is used")
    if int(args.start_frame) < 0:
        parser.error("--start-frame must be >= 0")
    if int(args.max_frames) < 0:
        parser.error("--max-frames must be >= 0")
    if int(args.solve_min_iterations) > int(args.solve_max_iterations):
        parser.error("--solve-min-iterations must be <= --solve-max-iterations")
    if (
        float(args.collision_distance_epsilon_m)
        > float(args.collision_min_distance_m)
    ):
        parser.error(
            "--collision-distance-epsilon-m must not exceed "
            "--collision-min-distance-m"
        )


def load_group(args: argparse.Namespace) -> Any:
    start = int(args.start_frame)
    end = -1 if int(args.max_frames) == 0 else start + int(args.max_frames)
    group_args = argparse.Namespace(
        replay_dataset=str(Path(args.dataset_root).expanduser().resolve()),
        replay_parquet="",
        replay_episode=int(args.episode),
        replay_action_column="action",
        replay_all_episodes=False,
        active_sides=("left", "right"),
    )
    group = build_replay_group_for_episode(
        group_args,
        episode_index=int(args.episode),
        group_spec=ReplayGroupSpec(
            key="pose_lift_sim",
            start_index=start,
            end_index=end,
            episode_range=str(args.episode),
        ),
        balanced_episode_count=1,
        require_arm_joint_targets=True,
    )
    if int(group.frame_count) < 2:
        raise ValueError("Pose Lift sim replay requires at least two source frames.")
    return group


def _saved_qref(group: Any, index: int) -> dict[str, np.ndarray]:
    return {
        side: np.asarray(group.arm_joint_targets[side][int(index)], dtype=float).reshape(6)
        for side in ("left", "right")
    }


def _pose_at(group: Any, index: int) -> dict[str, np.ndarray]:
    raw = replay_action_poses_in_hardware_base_at_index(
        group,
        ("left", "right"),
        index=int(index),
    )
    return {side: parse_pose_matrix(value) for side, value in raw.items()}


def build_runtime_references(group: Any, *, reference_hz: float) -> tuple[Any, ...]:
    samples = [
        TimestampedMotionTarget(
            timestamp=float(group.relative_time_sec(index)),
            source_index=int(index),
            pose_by_side=_pose_at(group, index),
        )
        for index in range(int(group.frame_count))
    ]
    return tuple(
        resample_motion_targets(
            samples,
            reference_hz=float(reference_hz),
            include_final=True,
        )
    )


def planner_config_from_args(
    args: argparse.Namespace,
    *,
    profile_name: str,
) -> RM65PositionCollisionPlannerConfig:
    fixed_iterations = args.fixed_solve_iterations
    solve_min = (
        int(fixed_iterations)
        if fixed_iterations is not None
        else int(args.solve_min_iterations)
    )
    solve_max = (
        int(fixed_iterations)
        if fixed_iterations is not None
        else int(args.solve_max_iterations)
    )
    return RM65PositionCollisionPlannerConfig(
        reference_hz=float(args.reference_hz),
        scene_profile=str(args.scene_profile),
        posture_cost=(
            None if args.posture_cost is None else float(args.posture_cost)
        ),
        collision_min_distance_m=float(args.collision_min_distance_m),
        collision_distance_epsilon_m=float(args.collision_distance_epsilon_m),
        max_joint_velocity_rad_s=tuple(
            float(value) for value in args.max_joint_velocity_rad_s
        ),
        max_joint_step_rad=(
            None
            if args.max_joint_step_rad is None
            else tuple(float(value) for value in args.max_joint_step_rad)
        ),
        damping=float(args.damping),
        solve_min_iterations=solve_min,
        solve_max_iterations=solve_max,
        solve_adaptive_iterations_enabled=(
            bool(args.adaptive_iterations) and fixed_iterations is None
        ),
        solve_iteration_reserve_ms=float(args.solve_iteration_reserve_ms),
        pose_lift_profile=str(profile_name),
        debug_profile=bool(args.debug_profile),
        ik_tuning=build_rm65_pose_lift_ik_tuning(
            position_control_profile=RM65_BIMANUAL_POSITION_V1_NAME,
            pose_lift_profile=str(profile_name),
            orientation_weight=float(args.orientation_weight),
        ),
    )


def build_planner(
    args: argparse.Namespace,
    *,
    profile_name: str,
) -> RM65PositionCollisionPlanner:
    endpoint_to_tool = {
        side: endpoint_to_tool_transform(
            load_rm65_tool_profile(
                getattr(args, f"{side}_tool_profile"),
                side=side,
            )
        )
        for side in ("left", "right")
    }
    return RM65PositionCollisionPlanner(
        planner_config_from_args(args, profile_name=profile_name),
        endpoint_to_tool_by_side=endpoint_to_tool,
    )


def generate_trajectory(
    group: Any,
    *,
    profile_name: str,
    planner: Any,
    references: Sequence[Any],
) -> PoseLiftSimTrajectory:
    initial_qpos = _saved_qref(group, 0)
    planner.reset(initial_qpos)
    frames: list[PoseLiftSimFrame] = []
    status_counts: dict[str, int] = {}
    hold_count = 0
    pose_lift_applied_count = 0
    for reference in references:
        if int(reference.sequence) == 0:
            continue
        result = planner.plan_pose_reference(
            reference.pose_by_side,
            target_reference_frame="arm_base",
            target_controlled_frame="tool_tcp",
            source_timestamp=float(reference.timestamp),
            source="pose_lift_sim_replay",
            metadata={
                "left_source_index": int(reference.left_source_index),
                "right_source_index": int(reference.right_source_index),
                "interpolation_alpha": float(reference.interpolation_alpha),
            },
        )
        status = str(result.status)
        status_counts[status] = status_counts.get(status, 0) + 1
        hold = bool(result.hold) or not bool(result.valid)
        hold_count += int(hold)
        metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
        pose_lift = metadata.get("pose_lift")
        pose_lift = pose_lift if isinstance(pose_lift, Mapping) else {}
        pose_lift_applied_count += int(bool(pose_lift.get("applied", False)))
        collision_distance = metadata.get("collision_distance_m")
        frames.append(
            PoseLiftSimFrame(
                sequence=int(result.sequence),
                timestamp=float(reference.timestamp),
                source_frame_index=int(reference.right_source_index),
                interpolation_alpha=float(reference.interpolation_alpha),
                status=status,
                hold=hold,
                qpos_by_side={
                    side: np.asarray(result.q_ref_by_side[side], dtype=float).copy()
                    for side in ("left", "right")
                },
                modification_reasons=tuple(result.modification_reasons),
                pose_lift_reason=str(pose_lift.get("reason") or "unknown"),
                collision_distance_m=(
                    None
                    if collision_distance is None
                    else float(collision_distance)
                ),
            )
        )
    return PoseLiftSimTrajectory(
        profile_name=str(profile_name),
        source_episode_index=int(group.source_episode_index),
        source_path=str(group.source_path),
        source_frame_count=int(group.frame_count),
        reference_hz=float(getattr(planner.config, "reference_hz", 60.0)),
        initial_qpos_by_side={
            side: value.copy() for side, value in initial_qpos.items()
        },
        frames=tuple(frames),
        status_counts=status_counts,
        hold_count=hold_count,
        pose_lift_applied_count=pose_lift_applied_count,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def write_trace_jsonl(
    path: Path,
    trajectories: Sequence[PoseLiftSimTrajectory],
) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for trajectory in trajectories:
        lines.append(
            json.dumps(
                {
                    "record_type": "initial_qpos",
                    "profile": trajectory.profile_name,
                    "source_episode_index": trajectory.source_episode_index,
                    "qpos_by_side": trajectory.initial_qpos_by_side,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=_jsonable,
            )
        )
        for frame in trajectory.frames:
            lines.append(
                json.dumps(
                    {
                        "record_type": "pose_reference",
                        "profile": trajectory.profile_name,
                        "sequence": frame.sequence,
                        "timestamp": frame.timestamp,
                        "source_frame_index": frame.source_frame_index,
                        "interpolation_alpha": frame.interpolation_alpha,
                        "status": frame.status,
                        "hold": frame.hold,
                        "qpos_by_side": frame.qpos_by_side,
                        "modification_reasons": frame.modification_reasons,
                        "pose_lift_reason": frame.pose_lift_reason,
                        "collision_distance_m": frame.collision_distance_m,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_jsonable,
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sleep_interruptibly(
    duration_sec: float,
    *,
    viewer_handle: Any,
) -> bool:
    deadline = time.monotonic() + max(0.0, float(duration_sec))
    while time.monotonic() < deadline:
        if not viewer_handle.is_running():
            return False
        viewer_handle.sync()
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    return bool(viewer_handle.is_running())


def view_trajectories(
    args: argparse.Namespace,
    trajectories: Sequence[PoseLiftSimTrajectory],
) -> None:
    # Imports stay lazy so --help, --list-episodes, and offline trace generation
    # do not require MuJoCo/GLFW or create a GUI resource.
    from sim.contracts import SceneSpec
    from sim.mujoco.controllers import QposController
    from sim.mujoco.scene import MujocoScene
    from sim.mujoco.viewer import passive_viewer_handle

    scene_spec = SceneSpec.kinematic_replay()
    scene = MujocoScene.from_urdf(
        Path(args.urdf).expanduser().resolve(),
        camera_names=(),
        timestep=1.0 / float(args.reference_hz),
        gravity=scene_spec.gravity_mps2,
        scene_spec=scene_spec,
    )
    missing = sorted({"left", "right"} - set(scene.arm_specs))
    if missing:
        raise RuntimeError(f"MuJoCo view URDF is missing arm specs: {missing}")
    controller = QposController(scene, collision_mode="raw")
    with passive_viewer_handle(
        scene,
        show_left_ui=not bool(args.hide_left_ui),
        show_right_ui=not bool(args.hide_right_ui),
    ) as viewer_handle:
        keep_running = True
        for repeat_index in range(int(args.repeat)):
            for trajectory in trajectories:
                if not keep_running or not viewer_handle.is_running():
                    keep_running = False
                    break
                scene.reset_data()
                controller.apply(trajectory.initial_qpos_by_side)
                viewer_handle.sync()
                print(
                    "[PoseLiftSim] "
                    f"play profile={trajectory.profile_name} "
                    f"episode={trajectory.source_episode_index} "
                    f"repeat={repeat_index + 1}/{int(args.repeat)} "
                    f"references={len(trajectory.frames)}",
                    flush=True,
                )
                previous_timestamp = 0.0
                for frame in trajectory.frames:
                    if not viewer_handle.is_running():
                        keep_running = False
                        break
                    controller.apply(frame.qpos_by_side)
                    viewer_handle.sync()
                    if not bool(args.no_sleep):
                        dt = max(0.0, frame.timestamp - previous_timestamp)
                        if not _sleep_interruptibly(
                            dt / float(args.speed),
                            viewer_handle=viewer_handle,
                        ):
                            keep_running = False
                            break
                    previous_timestamp = frame.timestamp
                if keep_running and float(args.pause_between_profiles_sec) > 0.0:
                    keep_running = _sleep_interruptibly(
                        float(args.pause_between_profiles_sec),
                        viewer_handle=viewer_handle,
                    )
        if keep_running and bool(args.hold_viewer_at_end):
            print("[PoseLiftSim] viewer_hold=close the MuJoCo window to exit", flush=True)
            while viewer_handle.is_running():
                viewer_handle.sync()
                time.sleep(0.02)


def run(
    args: argparse.Namespace,
    *,
    planner_builder: Callable[..., Any] = build_planner,
) -> dict[str, Any]:
    if bool(args.list_episodes):
        episode_indices = read_replay_dataset_episode_indices(args.dataset_root)
        payload = {
            "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
            "episodes": episode_indices,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    group = load_group(args)
    references = build_runtime_references(
        group,
        reference_hz=float(args.reference_hz),
    )
    trajectories: list[PoseLiftSimTrajectory] = []
    for profile_name in selected_profiles(args):
        print(
            f"[PoseLiftSim] solving profile={profile_name} "
            f"episode={group.source_episode_index} source_frames={group.frame_count}",
            flush=True,
        )
        trajectory = generate_trajectory(
            group,
            profile_name=profile_name,
            planner=planner_builder(args, profile_name=profile_name),
            references=references,
        )
        trajectories.append(trajectory)
        print(
            "[PoseLiftSim] solved "
            + json.dumps(trajectory.summary(), ensure_ascii=False, sort_keys=True),
            flush=True,
        )

    payload = {
        "schema": "teleop_runtime.rm65_pose_lift_sim_replay/v1",
        "device_access": False,
        "dispatch": False,
        "mujoco_role": "kinematic_viewer_of_pose_lift_admitted_qpos",
        "first_frame_contract": "saved_accepted_qref",
        "source_action_contract": "accepted_qref_fk_pose",
        "profiles": selected_profiles(args),
        "parameters": {
            "reference_hz": float(args.reference_hz),
            "orientation_weight": float(args.orientation_weight),
            "posture_cost_override": args.posture_cost,
            "damping": float(args.damping),
            "solve_min_iterations": int(args.solve_min_iterations),
            "solve_max_iterations": int(args.solve_max_iterations),
            "fixed_solve_iterations": args.fixed_solve_iterations,
            "adaptive_iterations": bool(args.adaptive_iterations),
            "effective_solve_min_iterations": int(
                args.fixed_solve_iterations
                if args.fixed_solve_iterations is not None
                else args.solve_min_iterations
            ),
            "effective_solve_max_iterations": int(
                args.fixed_solve_iterations
                if args.fixed_solve_iterations is not None
                else args.solve_max_iterations
            ),
            "effective_adaptive_iterations": bool(
                args.adaptive_iterations and args.fixed_solve_iterations is None
            ),
            "max_joint_velocity_rad_s": list(args.max_joint_velocity_rad_s),
            "max_joint_step_rad": args.max_joint_step_rad,
            "scene_profile": str(args.scene_profile),
        },
        "trajectories": [trajectory.summary() for trajectory in trajectories],
    }
    if args.summary_json is not None:
        path = Path(args.summary_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_jsonable),
            encoding="utf-8",
        )
    if args.trace_jsonl is not None:
        write_trace_jsonl(args.trace_jsonl, trajectories)
    if bool(args.viewer):
        view_trajectories(args, trajectories)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    run(args)


if __name__ == "__main__":
    main()


__all__ = [
    "PoseLiftSimFrame",
    "PoseLiftSimTrajectory",
    "build_parser",
    "build_runtime_references",
    "generate_trajectory",
    "main",
    "planner_config_from_args",
    "run",
    "selected_profiles",
]
