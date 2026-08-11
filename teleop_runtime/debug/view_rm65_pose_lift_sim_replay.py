"""Visualize versioned RM65 Pose Lift replay trajectories in MuJoCo.

This is an offline debug entrypoint.  It loads a recorded pose action, uses the
saved accepted q_ref for the first-frame initialization, and displays resolved
qpos in MuJoCo.  Registered V1--V10 profiles use the same 60 Hz
``RM65PositionCollisionPlanner`` as pose replay/deploy.  The explicit
official-all-IK comparison profile instead reproduces the current default-off
research chain: source-rate official all-solution IK, collision-safe causal
branch selection, linear joint reference, and a 120 Hz third-order horizon
governor.

MuJoCo is a trajectory viewer here.  It can also display caller-supplied
normalized gripper close references on the scene's gripper proxy.  It does not
solve IK again, project the planner output, open a device, or dispatch a real
command.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hardware.rm65.official_all_ik import RM65OfficialAllIK
from robot_motion.position_control import (
    JointHorizonLimits,
    JointHorizonState,
    TimestampedMotionTarget,
    plan_jerk_limited_joint_horizon,
    resample_motion_targets,
)
from robot_motion.rm65 import (
    DEFAULT_COLLISION_SCENE,
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_POSE_LIFT_PROFILE_CHOICES,
    RM65_POSE_LIFT_V1_BASELINE_NAME,
    RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
    RM65PositionCollisionPlanner,
    RM65PositionCollisionPlannerConfig,
    build_rm65_pose_lift_ik_tuning,
    endpoint_to_tool_transform,
    load_rm65_tool_profile,
    select_nearest_collision_safe_rm65_ik_pair,
)
from robot_motion.rm65.constants import (
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
)
from robot_motion.rm65.defaults import (
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
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

OFFICIAL_SOURCE_ALL_IK_GOVERNOR_PROFILE_NAME = (
    "official_source_all_ik_causal_governor_ramp_0p05"
)
POSE_LIFT_SIM_PROFILE_CHOICES = (
    *RM65_POSE_LIFT_PROFILE_CHOICES,
    OFFICIAL_SOURCE_ALL_IK_GOVERNOR_PROFILE_NAME,
)
REPRESENTATIVE_SUITE_0727_NAME = "0727-v4-vs-official-governor"


@dataclass(frozen=True)
class RepresentativeEpisodeSpec:
    episode_index: int
    start_frame: int
    max_frames: int
    reason: str


# Each range is exactly one recorded continuity segment.  Keeping collection
# pauses out of the trajectory is part of the A/B contract: a segment restart
# is an initialization boundary, not a physical command transition.
REPRESENTATIVE_SUITE_0727 = (
    RepresentativeEpisodeSpec(
        episode_index=96,
        start_frame=40,
        max_frames=75,
        reason="V4 right-arm structural branch drift into a deep singular tail",
    ),
    RepresentativeEpisodeSpec(
        episode_index=37,
        start_frame=157,
        max_frames=62,
        reason="recorded left-arm action itself approaches the singular region",
    ),
    RepresentativeEpisodeSpec(
        episode_index=68,
        start_frame=0,
        max_frames=163,
        reason="ordinary responsive motion with meaningful dynamic pressure",
    ),
)


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
    gripper_close_by_name: Mapping[str, float] = field(default_factory=dict)


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
    initial_gripper_close_by_name: Mapping[str, float] = field(
        default_factory=dict
    )

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
            "gripper_reference_count": int(
                sum(bool(frame.gripper_close_by_name) for frame in self.frames)
            ),
            "gripper_names": sorted(
                {
                    str(name)
                    for frame in self.frames
                    for name in frame.gripper_close_by_name
                }
            ),
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
        "--representative-suite",
        choices=(REPRESENTATIVE_SUITE_0727_NAME,),
        default=None,
        help=(
            "Run the fixed, continuity-safe 0727 visual A/B suite. Unless "
            "--pose-lift-profile is supplied, each segment plays V4 first and "
            "the current official-all-IK governor candidate second."
        ),
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
        choices=POSE_LIFT_SIM_PROFILE_CHOICES,
        default=None,
        help=(
            "Trajectory solver to play. Repeat to compare profiles sequentially. "
            "Default: V1 for one --episode; V4 then the current official-all-IK "
            "governor candidate for --representative-suite."
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
        "--official-governor-command-hz",
        type=_finite_positive,
        default=120.0,
        help=(
            "Command rate for the official source all-IK horizon governor. "
            "The validated current-best contract is 120 Hz."
        ),
    )
    parser.add_argument(
        "--official-governor-lookahead-source-steps",
        type=_positive_int,
        default=2,
        help="Source intervals in the causal horizon; validated default: 2.",
    )
    parser.add_argument(
        "--official-governor-jerk-ramp-sec",
        type=_finite_positive,
        default=0.05,
        help=(
            "Derive jerk limits as max_acceleration/ramp_sec; validated "
            "current-best value: 0.05 s."
        ),
    )
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
    if bool(args.all_profiles):
        requested = tuple(RM65_POSE_LIFT_PROFILE_CHOICES)
    elif args.pose_lift_profile:
        requested = tuple(args.pose_lift_profile)
    elif args.representative_suite:
        requested = (
            RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
            OFFICIAL_SOURCE_ALL_IK_GOVERNOR_PROFILE_NAME,
        )
    else:
        requested = (RM65_POSE_LIFT_V1_BASELINE_NAME,)
    return tuple(dict.fromkeys(str(value) for value in requested))


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if (
        args.episode is None
        and not bool(args.list_episodes)
        and args.representative_suite is None
    ):
        parser.error(
            "--episode or --representative-suite is required unless "
            "--list-episodes is used"
        )
    if args.episode is not None and args.representative_suite is not None:
        parser.error("--episode and --representative-suite are mutually exclusive")
    if bool(args.all_profiles) and args.representative_suite is not None:
        parser.error(
            "--all-profiles cannot be combined with --representative-suite; "
            "pass explicit --pose-lift-profile values instead"
        )
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
    source_steps = int(args.official_governor_lookahead_source_steps)
    if source_steps < 2:
        parser.error("--official-governor-lookahead-source-steps must be >= 2")
    if (
        OFFICIAL_SOURCE_ALL_IK_GOVERNOR_PROFILE_NAME in selected_profiles(args)
        and not math.isclose(
            float(args.official_governor_command_hz),
            120.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        parser.error(
            "the current official-all-IK comparison contract requires "
            "--official-governor-command-hz 120"
        )


def resolved_episode_specs(
    args: argparse.Namespace,
) -> tuple[RepresentativeEpisodeSpec, ...]:
    if args.representative_suite == REPRESENTATIVE_SUITE_0727_NAME:
        return REPRESENTATIVE_SUITE_0727
    if args.episode is None:
        return ()
    return (
        RepresentativeEpisodeSpec(
            episode_index=int(args.episode),
            start_frame=int(args.start_frame),
            max_frames=int(args.max_frames),
            reason="explicit episode selection",
        ),
    )


def load_group(
    args: argparse.Namespace,
    *,
    episode_spec: RepresentativeEpisodeSpec | None = None,
) -> Any:
    spec = (
        episode_spec
        if episode_spec is not None
        else RepresentativeEpisodeSpec(
            episode_index=int(args.episode),
            start_frame=int(args.start_frame),
            max_frames=int(args.max_frames),
            reason="explicit episode selection",
        )
    )
    start = int(spec.start_frame)
    end = -1 if int(spec.max_frames) == 0 else start + int(spec.max_frames)
    group_args = argparse.Namespace(
        replay_dataset=str(Path(args.dataset_root).expanduser().resolve()),
        replay_parquet="",
        replay_episode=int(spec.episode_index),
        replay_action_column="action",
        replay_all_episodes=False,
        active_sides=("left", "right"),
    )
    group = build_replay_group_for_episode(
        group_args,
        episode_index=int(spec.episode_index),
        group_spec=ReplayGroupSpec(
            key="pose_lift_sim",
            start_index=start,
            end_index=end,
            episode_range=str(spec.episode_index),
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
    endpoint_to_tool = endpoint_to_tool_by_side_from_args(args)
    return RM65PositionCollisionPlanner(
        planner_config_from_args(args, profile_name=profile_name),
        endpoint_to_tool_by_side=endpoint_to_tool,
    )


def endpoint_to_tool_by_side_from_args(
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    return {
        side: endpoint_to_tool_transform(
            load_rm65_tool_profile(
                getattr(args, f"{side}_tool_profile"),
                side=side,
            )
        )
        for side in ("left", "right")
    }


def _build_official_algo() -> Any:
    # The shared adapter remains lazy internally and opens no device.
    return RM65OfficialAllIK()


def _solve_official_all_ik_candidates(
    *,
    algo: Any,
    seed_rad: np.ndarray,
    target_endpoint: np.ndarray,
) -> tuple[np.ndarray, ...]:
    return tuple(
        algo.solve_candidates(
            seed_rad=seed_rad,
            target_endpoint=target_endpoint,
        )
    )


def _require_single_continuity_segment(group: Any) -> int:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(
            Path(group.source_path),
            columns=["collection.continuity_segment_index"],
        )
    except Exception as exc:
        raise RuntimeError(
            "official source all-IK comparison requires the recorded "
            "collection.continuity_segment_index column"
        ) from exc
    all_segment_ids = np.asarray(
        table["collection.continuity_segment_index"].to_numpy(),
        dtype=int,
    )
    selected = all_segment_ids[int(group.start_index) : int(group.end_index)]
    unique = np.unique(selected)
    if selected.shape != (int(group.frame_count),) or unique.size != 1:
        raise ValueError(
            "official source all-IK comparison must select exactly one "
            "continuity segment; choose --start-frame/--max-frames at a "
            f"segment boundary, got segment_ids={unique.tolist()}."
        )
    return int(unique[0])


def _official_source_qpos_path(
    group: Any,
    *,
    algo: Any,
    planner: RM65PositionCollisionPlanner,
    endpoint_to_tool_by_side: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    _require_single_continuity_segment(group)
    source_q: list[dict[str, np.ndarray]] = [_saved_qref(group, 0)]
    planner.reset(source_q[0])
    inverse_endpoint_to_tool = {
        side: np.linalg.inv(
            np.asarray(endpoint_to_tool_by_side[side], dtype=float)
        )
        for side in ("left", "right")
    }
    for frame_index in range(1, int(group.frame_count)):
        tcp_targets = _pose_at(group, frame_index)
        candidate_lists = {
            side: _solve_official_all_ik_candidates(
                algo=algo,
                seed_rad=source_q[-1][side],
                target_endpoint=(
                    np.asarray(tcp_targets[side], dtype=float)
                    @ inverse_endpoint_to_tool[side]
                ),
            )
            for side in ("left", "right")
        }
        receipt = select_nearest_collision_safe_rm65_ik_pair(
            candidate_lists,
            anchor_qpos_by_side=source_q[-1],
            validate_pair=lambda pair: planner.validate_qpos_reference(
                pair,
                source="pose_lift_sim_official_source_all_ik",
            ),
        )
        if not receipt.accepted or receipt.qpos_by_side is None:
            raise RuntimeError(
                "official source all-IK failed to produce a collision-safe "
                f"dual-arm pair at episode={group.source_episode_index} "
                f"source_frame={int(group.start_index) + frame_index} "
                f"reason={receipt.reason} "
                f"candidate_counts={dict(receipt.candidate_count_by_side)} "
                f"rejections={dict(receipt.rejection_reason_counts)}"
            )
        source_q.append(
            {
                side: np.asarray(
                    receipt.qpos_by_side[side],
                    dtype=float,
                ).copy()
                for side in ("left", "right")
            }
        )
    return {
        side: np.asarray([row[side] for row in source_q], dtype=float)
        for side in ("left", "right")
    }


def _linear_source_q_reference_120hz(
    source_qpos_by_side: Mapping[str, np.ndarray],
    *,
    commands_per_source: int = 8,
) -> dict[str, np.ndarray]:
    count = int(
        np.asarray(source_qpos_by_side["left"], dtype=float).shape[0]
    )
    if count < 2:
        raise ValueError("source qpos path must contain at least two frames.")
    result: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        source = np.asarray(source_qpos_by_side[side], dtype=float)
        if source.shape != (count, 6):
            raise ValueError(f"{side} source qpos path must have shape {(count, 6)}.")
        rows = [
            (1.0 - step / commands_per_source) * source[source_index]
            + (step / commands_per_source) * source[source_index + 1]
            for source_index in range(count - 1)
            for step in range(commands_per_source)
        ]
        rows.append(source[-1].copy())
        result[side] = np.asarray(rows, dtype=float)
    return result


def _govern_source_q_reference(
    q_reference_by_side: Mapping[str, np.ndarray],
    *,
    lookahead_source_steps: int,
    jerk_ramp_sec: float,
    commands_per_source: int = 8,
    command_hz: float = 120.0,
) -> dict[str, np.ndarray]:
    max_acceleration = np.asarray(
        RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
        dtype=float,
    )
    limits = JointHorizonLimits(
        min_position_rad=RM65_ABSOLUTE_Q_MIN_RAD,
        max_position_rad=RM65_ABSOLUTE_Q_MAX_RAD,
        max_velocity_rad_s=RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
        max_acceleration_rad_s2=max_acceleration,
        max_jerk_rad_s3=max_acceleration / float(jerk_ramp_sec),
    )
    output: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        reference = np.asarray(q_reference_by_side[side], dtype=float)
        command = np.empty_like(reference)
        command[0] = reference[0]
        state = JointHorizonState(
            q_rad=command[0],
            qd_rad_s=np.zeros(6),
            qdd_rad_s2=np.zeros(6),
        )
        final_index = int(reference.shape[0] - 1)
        start = 0
        while start < final_index:
            horizon_end = min(
                final_index,
                start
                + int(lookahead_source_steps) * int(commands_per_source),
            )
            commit_count = min(
                int(commands_per_source),
                horizon_end - start,
            )
            plan = plan_jerk_limited_joint_horizon(
                reference[start : horizon_end + 1],
                initial_state=state,
                limits=limits,
                dt_sec=1.0 / float(command_hz),
                commit_count=commit_count,
            )
            command[start + 1 : start + commit_count + 1] = (
                plan.q_control_cut_rad
            )
            state = JointHorizonState(
                q_rad=plan.q_path_rad[commit_count],
                qd_rad_s=plan.qd_path_rad_s[commit_count],
                qdd_rad_s2=plan.qdd_path_rad_s2[commit_count],
            )
            start += commit_count
        output[side] = command
    return output


def generate_official_source_all_ik_governed_trajectory(
    group: Any,
    *,
    algo: Any,
    planner: RM65PositionCollisionPlanner,
    endpoint_to_tool_by_side: Mapping[str, np.ndarray],
    command_hz: float = 120.0,
    lookahead_source_steps: int = 2,
    jerk_ramp_sec: float = 0.05,
) -> PoseLiftSimTrajectory:
    source_qpos = _official_source_qpos_path(
        group,
        algo=algo,
        planner=planner,
        endpoint_to_tool_by_side=endpoint_to_tool_by_side,
    )
    commands_per_source = 8
    q_reference = _linear_source_q_reference_120hz(
        source_qpos,
        commands_per_source=commands_per_source,
    )
    q_command = _govern_source_q_reference(
        q_reference,
        lookahead_source_steps=int(lookahead_source_steps),
        jerk_ramp_sec=float(jerk_ramp_sec),
        commands_per_source=commands_per_source,
        command_hz=float(command_hz),
    )
    frame_count = int(q_command["left"].shape[0])
    frames: list[PoseLiftSimFrame] = []
    previous = {
        side: np.asarray(q_command[side][0], dtype=float).copy()
        for side in ("left", "right")
    }
    for command_index in range(1, frame_count):
        qpos = {
            side: np.asarray(q_command[side][command_index], dtype=float).copy()
            for side in ("left", "right")
        }
        validation = planner.validate_qpos_reference(
            qpos,
            source="pose_lift_sim_official_governor_endpoint",
        )
        if not validation.safe:
            raise RuntimeError(
                "official governor endpoint validation failed "
                f"episode={group.source_episode_index} "
                f"command_index={command_index} reason={validation.reason}"
            )
        for alpha in (0.25, 0.5, 0.75):
            sample = {
                side: (1.0 - alpha) * previous[side] + alpha * qpos[side]
                for side in ("left", "right")
            }
            sampled = planner.validate_qpos_reference(
                sample,
                source="pose_lift_sim_official_governor_interval_sample",
            )
            if not sampled.safe:
                raise RuntimeError(
                    "official governor sampled interval validation failed "
                    f"episode={group.source_episode_index} "
                    f"command_index={command_index} alpha={alpha} "
                    f"reason={sampled.reason}"
                )
        source_interval = min(
            int(group.frame_count) - 2,
            (command_index - 1) // commands_per_source,
        )
        interpolation_alpha = (
            ((command_index - 1) % commands_per_source) + 1
        ) / float(commands_per_source)
        frames.append(
            PoseLiftSimFrame(
                sequence=command_index,
                timestamp=command_index / float(command_hz),
                source_frame_index=(
                    int(group.start_index) + source_interval + 1
                ),
                interpolation_alpha=float(interpolation_alpha),
                status="tracking",
                hold=False,
                qpos_by_side=qpos,
                modification_reasons=("third_order_horizon_governor",),
                pose_lift_reason=(
                    "official_source_all_ik_collision_safe_nearest_branch"
                ),
                collision_distance_m=(
                    None
                    if validation.collision_distance_m is None
                    else float(validation.collision_distance_m)
                ),
            )
        )
        previous = qpos
    initial_qpos = {
        side: np.asarray(q_command[side][0], dtype=float).copy()
        for side in ("left", "right")
    }
    return PoseLiftSimTrajectory(
        profile_name=OFFICIAL_SOURCE_ALL_IK_GOVERNOR_PROFILE_NAME,
        source_episode_index=int(group.source_episode_index),
        source_path=str(group.source_path),
        source_frame_count=int(group.frame_count),
        reference_hz=float(command_hz),
        initial_qpos_by_side=initial_qpos,
        frames=tuple(frames),
        status_counts={"tracking": len(frames)},
        hold_count=0,
        pose_lift_applied_count=int(group.frame_count) - 1,
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
                    "gripper_close_by_name": (
                        trajectory.initial_gripper_close_by_name
                    ),
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
                        "gripper_close_by_name": frame.gripper_close_by_name,
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
    from sim.mujoco.controllers import GripperController, QposController
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
    gripper_controller = GripperController(scene, collision_mode="raw")
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
                if trajectory.initial_gripper_close_by_name:
                    missing_grippers = sorted(
                        set(trajectory.initial_gripper_close_by_name)
                        - set(scene.gripper_specs)
                    )
                    if missing_grippers:
                        raise RuntimeError(
                            "MuJoCo view URDF is missing gripper specs: "
                            f"{missing_grippers}"
                        )
                    gripper_controller.apply(
                        trajectory.initial_gripper_close_by_name
                    )
                viewer_handle.sync()
                comparison_label = (
                    "B_CURRENT_BETTER_OFFICIAL_ALL_IK_GOVERNOR"
                    if trajectory.profile_name
                    == OFFICIAL_SOURCE_ALL_IK_GOVERNOR_PROFILE_NAME
                    else (
                        "A_V4_BASELINE"
                        if trajectory.profile_name
                        == RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME
                        else "CUSTOM"
                    )
                )
                print(
                    "\n"
                    "============================================================\n"
                    "[PoseLiftSim] "
                    f"comparison={comparison_label} "
                    f"play profile={trajectory.profile_name} "
                    f"episode={trajectory.source_episode_index} "
                    f"repeat={repeat_index + 1}/{int(args.repeat)} "
                    f"references={len(trajectory.frames)}\n"
                    "============================================================",
                    flush=True,
                )
                previous_timestamp = 0.0
                for frame in trajectory.frames:
                    if not viewer_handle.is_running():
                        keep_running = False
                        break
                    controller.apply(frame.qpos_by_side)
                    if frame.gripper_close_by_name:
                        gripper_controller.apply(frame.gripper_close_by_name)
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

    profiles = selected_profiles(args)
    episode_specs = resolved_episode_specs(args)
    trajectories: list[PoseLiftSimTrajectory] = []
    official_algo = (
        _build_official_algo()
        if OFFICIAL_SOURCE_ALL_IK_GOVERNOR_PROFILE_NAME in profiles
        else None
    )
    endpoint_to_tool = (
        endpoint_to_tool_by_side_from_args(args)
        if official_algo is not None
        else None
    )
    for episode_spec in episode_specs:
        group = load_group(args, episode_spec=episode_spec)
        references = build_runtime_references(
            group,
            reference_hz=float(args.reference_hz),
        )
        print(
            "[PoseLiftSim] segment "
            f"episode={group.source_episode_index} "
            f"source_range=[{group.start_index},{group.end_index}) "
            f"source_frames={group.frame_count} reason={episode_spec.reason}",
            flush=True,
        )
        for profile_name in profiles:
            print(
                f"[PoseLiftSim] solving profile={profile_name} "
                f"episode={group.source_episode_index}",
                flush=True,
            )
            if profile_name == OFFICIAL_SOURCE_ALL_IK_GOVERNOR_PROFILE_NAME:
                if official_algo is None or endpoint_to_tool is None:
                    raise RuntimeError(
                        "official all-IK algorithm was not initialized"
                    )
                trajectory = (
                    generate_official_source_all_ik_governed_trajectory(
                        group,
                        algo=official_algo,
                        planner=planner_builder(
                            args,
                            profile_name=RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
                        ),
                        endpoint_to_tool_by_side=endpoint_to_tool,
                        command_hz=float(args.official_governor_command_hz),
                        lookahead_source_steps=int(
                            args.official_governor_lookahead_source_steps
                        ),
                        jerk_ramp_sec=float(
                            args.official_governor_jerk_ramp_sec
                        ),
                    )
                )
            else:
                trajectory = generate_trajectory(
                    group,
                    profile_name=profile_name,
                    planner=planner_builder(args, profile_name=profile_name),
                    references=references,
                )
            trajectories.append(trajectory)
            print(
                "[PoseLiftSim] solved "
                + json.dumps(
                    trajectory.summary(),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    payload = {
        "schema": "teleop_runtime.rm65_pose_lift_sim_replay/v2",
        "device_access": False,
        "dispatch": False,
        "mujoco_role": "kinematic_viewer_of_pose_lift_admitted_qpos",
        "first_frame_contract": "saved_accepted_qref",
        "source_action_contract": "accepted_qref_fk_pose",
        "representative_suite": args.representative_suite,
        "episode_specs": [
            {
                "episode_index": int(spec.episode_index),
                "start_frame": int(spec.start_frame),
                "max_frames": int(spec.max_frames),
                "reason": str(spec.reason),
            }
            for spec in episode_specs
        ],
        "profiles": profiles,
        "parameters": {
            "reference_hz": float(args.reference_hz),
            "official_governor_command_hz": float(
                args.official_governor_command_hz
            ),
            "official_governor_lookahead_source_steps": int(
                args.official_governor_lookahead_source_steps
            ),
            "official_governor_jerk_ramp_sec": float(
                args.official_governor_jerk_ramp_sec
            ),
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
    "OFFICIAL_SOURCE_ALL_IK_GOVERNOR_PROFILE_NAME",
    "PoseLiftSimFrame",
    "PoseLiftSimTrajectory",
    "REPRESENTATIVE_SUITE_0727",
    "REPRESENTATIVE_SUITE_0727_NAME",
    "RepresentativeEpisodeSpec",
    "build_parser",
    "build_runtime_references",
    "generate_official_source_all_ik_governed_trajectory",
    "generate_trajectory",
    "main",
    "planner_config_from_args",
    "resolved_episode_specs",
    "run",
    "selected_profiles",
]
