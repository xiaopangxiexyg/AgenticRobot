"""Offline RM65 local-singularity wind tunnel seeded by accepted q_ref.

This diagnostic never opens devices or dispatches motion.  It perturbs one
recorded q3/q5 structural coordinate, constructs target motion in the exact
weighted Pink FrameTask singular directions, and runs continuous closed-loop
Pose->QPOS rollouts.  It is a research evaluator, not a production planner or
runtime entrypoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_motion.rm65 import (
    DEFAULT_COLLISION_SCENE,
    RM65_POSE_LIFT_PROFILE_CHOICES,
    RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
    RM65PositionCollisionPlanner,
    RM65PositionCollisionPlannerConfig,
    build_rm65_pose_lift_ik_tuning,
    endpoint_to_tool_transform,
    load_rm65_tool_profile,
)
from robot_motion.rm65.defaults import (
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
)
from robot_motion.replay import ReplayGroupSpec
from teleop_runtime.replay.executor import (
    REPLAY_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S,
    REPLAY_POSITION_IK_DAMPING,
    REPLAY_POSITION_IK_ITERATION_RESERVE_MS,
    REPLAY_POSITION_IK_ORIENTATION_WEIGHT,
)
from teleop_runtime.replay.trajectory import (
    build_replay_group_for_episode,
    read_replay_dataset_episode_indices,
)


SIDES = ("left", "right")
STRUCTURAL_JOINTS = {"q3": 2, "q5": 4}
DIRECTION_CHOICES = ("weakest", "second_weakest", "strongest")
INITIAL_MOTION_CHOICES = ("stationary", "inward", "outward")
SCENARIO_CHOICES = ("persistent_push", "enter_exit")


@dataclass(frozen=True)
class WindSeed:
    episode_index: int
    frame_index: int
    side: str
    structural_joint: str
    source_abs_joint_deg: float
    source_other_structural_abs_deg: float
    split: str


@dataclass(frozen=True)
class WindCase:
    seed: WindSeed
    offset_deg: float
    initial_motion: str
    direction: str
    direction_sign: int
    scenario: str

    @property
    def key(self) -> str:
        return (
            f"ep{self.seed.episode_index}:f{self.seed.frame_index}:"
            f"{self.seed.side}:{self.seed.structural_joint}:"
            f"offset={self.offset_deg:+.6g}:initial={self.initial_motion}:"
            f"direction={self.direction}:{self.direction_sign:+d}:"
            f"scenario={self.scenario}"
        )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(tuple(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def se3_exp_local(twist_linear_angular: Sequence[float]) -> np.ndarray:
    """Return ``SE(3).exp`` for Pinocchio's linear-then-angular convention."""

    twist = np.asarray(twist_linear_angular, dtype=float).reshape(6)
    if not np.all(np.isfinite(twist)):
        raise ValueError("SE(3) twist must be finite.")
    linear = twist[:3]
    angular = twist[3:]
    theta = float(np.linalg.norm(angular))
    skew = np.array(
        [
            [0.0, -angular[2], angular[1]],
            [angular[2], 0.0, -angular[0]],
            [-angular[1], angular[0], 0.0],
        ],
        dtype=float,
    )
    if theta <= 1e-9:
        left_jacobian = (
            np.eye(3, dtype=float)
            + 0.5 * skew
            + (1.0 / 6.0) * (skew @ skew)
        )
    else:
        theta2 = theta * theta
        left_jacobian = (
            np.eye(3, dtype=float)
            + ((1.0 - math.cos(theta)) / theta2) * skew
            + ((theta - math.sin(theta)) / (theta2 * theta)) * (skew @ skew)
        )
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = R.from_rotvec(angular).as_matrix()
    pose[:3, 3] = left_jacobian @ linear
    return pose


def _seed_split(seed_identity: str, *, holdout_modulus: int) -> str:
    modulus = int(holdout_modulus)
    if modulus < 2:
        raise ValueError("holdout_modulus must be at least two.")
    digest = hashlib.sha256(seed_identity.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big") % modulus
    return "heldout" if bucket == 0 else "tune"


def select_wind_seeds(
    groups: Sequence[Any],
    *,
    min_other_structural_abs_deg: float,
    holdout_modulus: int,
    split: str,
    max_seeds: int,
) -> list[WindSeed]:
    """Choose deterministic real-qref seeds near each structural boundary."""

    requested_split = str(split)
    if requested_split not in {"all", "tune", "heldout"}:
        raise ValueError("split must be all, tune, or heldout.")
    minimum_other = float(min_other_structural_abs_deg)
    if not math.isfinite(minimum_other) or minimum_other < 0.0:
        raise ValueError("min_other_structural_abs_deg must be non-negative.")
    seeds: list[WindSeed] = []
    for group in groups:
        for side in SIDES:
            values = np.asarray(group.arm_joint_targets[side], dtype=float)
            if values.ndim != 2 or values.shape[1] != 6:
                raise ValueError(
                    f"episode {group.source_episode_index} {side} q_ref is invalid."
                )
            for structural_joint, joint_index in STRUCTURAL_JOINTS.items():
                other_index = 4 if joint_index == 2 else 2
                primary_abs = np.abs(np.degrees(values[:, joint_index]))
                other_abs = np.abs(np.degrees(values[:, other_index]))
                eligible = np.flatnonzero(
                    np.isfinite(primary_abs)
                    & np.isfinite(other_abs)
                    & (other_abs >= minimum_other)
                )
                if eligible.size == 0:
                    eligible = np.flatnonzero(
                        np.isfinite(primary_abs) & np.isfinite(other_abs)
                    )
                if eligible.size == 0:
                    continue
                frame_index = int(eligible[np.argmin(primary_abs[eligible])])
                identity = (
                    f"{int(group.source_episode_index)}:{frame_index}:"
                    f"{side}:{structural_joint}"
                )
                seed_split = _seed_split(
                    identity,
                    holdout_modulus=int(holdout_modulus),
                )
                if requested_split != "all" and seed_split != requested_split:
                    continue
                seeds.append(
                    WindSeed(
                        episode_index=int(group.source_episode_index),
                        frame_index=frame_index,
                        side=side,
                        structural_joint=structural_joint,
                        source_abs_joint_deg=float(primary_abs[frame_index]),
                        source_other_structural_abs_deg=float(
                            other_abs[frame_index]
                        ),
                        split=seed_split,
                    )
                )
    seeds.sort(
        key=lambda seed: (
            seed.source_abs_joint_deg,
            seed.episode_index,
            seed.side,
            seed.structural_joint,
        )
    )
    limit = int(max_seeds)
    return seeds if limit <= 0 else seeds[:limit]


def build_wind_cases(
    seeds: Sequence[WindSeed],
    *,
    offsets_deg: Sequence[float],
    initial_motions: Sequence[str],
    directions: Sequence[str],
    direction_signs: Sequence[int],
    scenarios: Sequence[str],
    max_cases: int,
) -> list[WindCase]:
    offsets = tuple(float(value) for value in offsets_deg)
    if not offsets or any(
        not math.isfinite(value) or value == 0.0 for value in offsets
    ):
        raise ValueError("offsets_deg must contain finite non-zero values.")
    motions = tuple(str(value) for value in initial_motions)
    direction_names = tuple(str(value) for value in directions)
    signs = tuple(int(value) for value in direction_signs)
    scenario_names = tuple(str(value) for value in scenarios)
    if not motions or any(value not in INITIAL_MOTION_CHOICES for value in motions):
        raise ValueError("invalid or empty initial_motions.")
    if not direction_names or any(
        value not in DIRECTION_CHOICES for value in direction_names
    ):
        raise ValueError("invalid or empty directions.")
    if not signs or any(value not in {-1, 1} for value in signs):
        raise ValueError("direction_signs must contain only -1 and +1.")
    if not scenario_names or any(
        value not in SCENARIO_CHOICES for value in scenario_names
    ):
        raise ValueError("invalid or empty scenarios.")
    cases = [
        WindCase(
            seed=seed,
            offset_deg=offset,
            initial_motion=motion,
            direction=direction,
            direction_sign=sign,
            scenario=scenario,
        )
        for seed in seeds
        for offset in offsets
        for motion in motions
        for direction in direction_names
        for sign in signs
        for scenario in scenario_names
    ]
    limit = int(max_cases)
    return cases if limit <= 0 else cases[:limit]


def _direction_index(name: str, count: int) -> int:
    if count < 2:
        raise ValueError("task conditioning must expose at least two directions.")
    if name == "weakest":
        return count - 1
    if name == "second_weakest":
        return count - 2
    if name == "strongest":
        return 0
    raise ValueError(f"unknown direction {name!r}.")


def _scenario_displacement_time(
    scenario: str,
    *,
    elapsed_sec: float,
    duration_sec: float,
) -> float:
    elapsed = float(np.clip(elapsed_sec, 0.0, duration_sec))
    if scenario == "persistent_push":
        return elapsed
    if scenario == "enter_exit":
        half = 0.5 * duration_sec
        return elapsed if elapsed <= half else duration_sec - elapsed
    raise ValueError(f"unknown scenario {scenario!r}.")


def _rotation_error_deg(target: np.ndarray, realized: np.ndarray) -> float:
    relative = target[:3, :3].T @ realized[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _group_by_episode(groups: Sequence[Any]) -> dict[int, Any]:
    return {int(group.source_episode_index): group for group in groups}


def _case_seed_qpos(case: WindCase, group: Any) -> dict[str, np.ndarray]:
    qpos = {
        side: np.asarray(
            group.arm_joint_targets[side][case.seed.frame_index],
            dtype=float,
        )
        .reshape(6)
        .copy()
        for side in SIDES
    }
    qpos[case.seed.side][
        STRUCTURAL_JOINTS[case.seed.structural_joint]
    ] = math.radians(case.offset_deg)
    return qpos


def _prepare_case_anchor(
    planner: Any,
    case: WindCase,
    source_qpos: Mapping[str, np.ndarray],
    *,
    dt_sec: float,
    warmstart_joint_speed_deg_s: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], Mapping[str, Any] | None]:
    joint_index = STRUCTURAL_JOINTS[case.seed.structural_joint]
    seed = {
        side: np.asarray(value, dtype=float).reshape(6).copy()
        for side, value in source_qpos.items()
    }
    if case.initial_motion == "stationary":
        planner.reset(seed)
        poses = {
            side: np.asarray(value, dtype=float).copy()
            for side, value in planner.observe_endpoint_poses_world(seed).items()
        }
        return seed, poses, None

    direction = 1.0 if case.offset_deg > 0.0 else -1.0
    if case.initial_motion == "inward":
        qdot = -direction * math.radians(warmstart_joint_speed_deg_s)
    elif case.initial_motion == "outward":
        qdot = direction * math.radians(warmstart_joint_speed_deg_s)
    else:
        raise ValueError(f"unknown initial motion {case.initial_motion!r}.")
    previous = {side: value.copy() for side, value in seed.items()}
    previous[case.seed.side][joint_index] -= qdot * dt_sec
    target = {
        side: np.asarray(value, dtype=float).copy()
        for side, value in planner.observe_endpoint_poses_world(seed).items()
    }
    planner.reset(previous)
    warmup = planner.plan_pose_reference(
        target,
        target_reference_frame="world",
        target_controlled_frame="arm_endpoint",
        source_timestamp=0.0,
        source="singularity_wind_tunnel:warmstart",
    )
    if not bool(warmup.valid) or bool(warmup.hold):
        raise RuntimeError(
            f"warmstart_not_admitted:{warmup.status}:"
            f"{warmup.metadata.get('failure')}"
        )
    anchor = {
        side: np.asarray(warmup.q_ref_by_side[side], dtype=float).copy()
        for side in SIDES
    }
    poses = {
        side: np.asarray(value, dtype=float).copy()
        for side, value in planner.observe_endpoint_poses_world(anchor).items()
    }
    return anchor, poses, warmup


def run_wind_case(
    planner: Any,
    group: Any,
    case: WindCase,
    *,
    reference_hz: float,
    duration_sec: float,
    weighted_task_speed_per_sec: float,
    warmstart_joint_speed_deg_s: float,
    include_frames: bool,
    include_constrained_feasibility: bool,
) -> dict[str, Any]:
    """Run one continuous detached rollout from one synthetic structural seed."""

    rate = float(reference_hz)
    duration = float(duration_sec)
    task_speed = float(weighted_task_speed_per_sec)
    warmstart_speed = float(warmstart_joint_speed_deg_s)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("reference_hz must be finite and positive.")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_sec must be finite and positive.")
    if not math.isfinite(task_speed) or task_speed <= 0.0:
        raise ValueError("weighted_task_speed_per_sec must be positive.")
    if not math.isfinite(warmstart_speed) or warmstart_speed < 0.0:
        raise ValueError("warmstart_joint_speed_deg_s must be non-negative.")
    dt_sec = 1.0 / rate
    frame_count = max(1, int(round(duration * rate)))
    seed_qpos = _case_seed_qpos(case, group)
    try:
        anchor, base_poses, warmup = _prepare_case_anchor(
            planner,
            case,
            seed_qpos,
            dt_sec=dt_sec,
            warmstart_joint_speed_deg_s=warmstart_speed,
        )
    except Exception as exc:  # noqa: BLE001 - invalid synthetic seeds are reported.
        return {
            "case_key": case.key,
            "case": _plain(asdict(case)),
            "status": "seed_rejected",
            "error": f"{type(exc).__name__}: {exc}",
            "frame_count": 0,
        }

    zero_conditioning = planner.analyze_task_conditioning(
        anchor,
        base_poses,
        dt_sec=dt_sec,
    )
    side_zero = zero_conditioning["sides"][case.seed.side]
    rows = np.asarray(side_zero["task_direction_rows"], dtype=float)
    costs = np.asarray(side_zero["task_cost"], dtype=float).reshape(6)
    direction_index = _direction_index(case.direction, rows.shape[0])
    weighted_direction = rows[direction_index]
    physical_direction = weighted_direction / costs
    current_q = {side: value.copy() for side, value in anchor.items()}
    previous_q = {side: value.copy() for side, value in anchor.items()}
    previous_velocity = {side: np.zeros(6, dtype=float) for side in SIDES}
    previous_acceleration = {side: np.zeros(6, dtype=float) for side in SIDES}
    initial_branch_sign = 1 if case.offset_deg > 0.0 else -1
    joint_index = STRUCTURAL_JOINTS[case.seed.structural_joint]

    min_sigma: list[float] = []
    max_demand: list[float] = []
    strict_alpha: list[float] = []
    translation_error: list[float] = []
    rotation_error: list[float] = []
    max_velocity_deg_s: list[float] = []
    max_acceleration_deg_s2: list[float] = []
    max_jerk_deg_s3: list[float] = []
    structural_abs_deg: list[float] = []
    hold_count = 0
    branch_crossing_count = 0
    previous_branch_sign = initial_branch_sign
    frames: list[dict[str, Any]] = []

    for frame_index in range(1, frame_count + 1):
        elapsed = min(duration, frame_index * dt_sec)
        displacement_time = _scenario_displacement_time(
            case.scenario,
            elapsed_sec=elapsed,
            duration_sec=duration,
        )
        physical_displacement = (
            float(case.direction_sign)
            * task_speed
            * displacement_time
            * physical_direction
        )
        targets = {side: pose.copy() for side, pose in base_poses.items()}
        targets[case.seed.side] = (
            base_poses[case.seed.side] @ se3_exp_local(physical_displacement)
        )
        conditioning = planner.analyze_task_conditioning(
            current_q,
            targets,
            dt_sec=dt_sec,
        )
        side_conditioning = conditioning["sides"][case.seed.side]
        min_sigma.append(
            float(side_conditioning["sigma_task_velocity_min"])
        )
        max_demand.append(
            float(side_conditioning["max_required_velocity_utilization"])
        )
        feasibility: Mapping[str, Any] | None = None
        if include_constrained_feasibility:
            feasibility = planner.analyze_constrained_task_feasibility(
                current_q,
                targets,
                dt_sec=dt_sec,
            )
            value = feasibility.get("strict_directional_alpha")
            if value is not None:
                strict_alpha.append(float(value))

        result = planner.plan_pose_reference(
            targets,
            target_reference_frame="world",
            target_controlled_frame="arm_endpoint",
            source_timestamp=elapsed,
            source="singularity_wind_tunnel:closed_loop",
            metadata={
                "wind_case_key": case.key,
                "wind_frame_index": frame_index,
            },
        )
        hold = not bool(result.valid) or bool(result.hold)
        hold_count += int(hold)
        next_q = {
            side: np.asarray(result.q_ref_by_side[side], dtype=float).copy()
            for side in SIDES
        }
        for side in SIDES:
            velocity = (next_q[side] - previous_q[side]) / dt_sec
            acceleration = (
                velocity - previous_velocity[side]
            ) / dt_sec
            jerk = (
                acceleration - previous_acceleration[side]
            ) / dt_sec
            max_velocity_deg_s.append(
                float(np.max(np.abs(np.degrees(velocity))))
            )
            max_acceleration_deg_s2.append(
                float(np.max(np.abs(np.degrees(acceleration))))
            )
            max_jerk_deg_s3.append(
                float(np.max(np.abs(np.degrees(jerk))))
            )
            previous_velocity[side] = velocity
            previous_acceleration[side] = acceleration
        selected_joint_deg = float(
            math.degrees(next_q[case.seed.side][joint_index])
        )
        structural_abs_deg.append(abs(selected_joint_deg))
        branch_sign = (
            1
            if selected_joint_deg > 1e-6
            else (-1 if selected_joint_deg < -1e-6 else previous_branch_sign)
        )
        if branch_sign != previous_branch_sign:
            branch_crossing_count += 1
        previous_branch_sign = branch_sign
        target = np.asarray(
            result.target_endpoint_pose_world_by_side[case.seed.side],
            dtype=float,
        )
        realized = np.asarray(
            result.realized_endpoint_pose_world_by_side[case.seed.side],
            dtype=float,
        )
        translation_error.append(
            float(np.linalg.norm(target[:3, 3] - realized[:3, 3]))
        )
        rotation_error.append(_rotation_error_deg(target, realized))
        if include_frames:
            frames.append(
                {
                    "frame_index": frame_index,
                    "elapsed_sec": elapsed,
                    "hold": hold,
                    "status": str(result.status),
                    "sigma_task_velocity_min": min_sigma[-1],
                    "max_required_velocity_utilization": max_demand[-1],
                    "strict_directional_alpha": (
                        None
                        if feasibility is None
                        else feasibility.get("strict_directional_alpha")
                    ),
                    "selected_structural_joint_deg": selected_joint_deg,
                    "translation_error_m": translation_error[-1],
                    "rotation_error_deg": rotation_error[-1],
                    "pose_lift": _plain(
                        result.metadata.get("pose_lift", {})
                    ),
                    "structural_branch_guard_runtime": _plain(
                        result.metadata.get("qp_motion_constraints", {}).get(
                            "structural_branch_guard_runtime",
                            {},
                        )
                    ),
                    "directional_task_guard": _plain(
                        result.metadata.get("qp_motion_constraints", {}).get(
                            "directional_task_guard",
                            {},
                        )
                    ),
                }
            )
        previous_q = {side: value.copy() for side, value in next_q.items()}
        current_q = next_q

    return {
        "case_key": case.key,
        "case": _plain(asdict(case)),
        "status": "completed",
        "frame_count": frame_count,
        "warmstart_applied": warmup is not None,
        "initial_task_conditioning": _plain(side_zero),
        "direction_index": direction_index,
        "weighted_direction": weighted_direction.tolist(),
        "physical_local_direction": physical_direction.tolist(),
        "hold_count": hold_count,
        "branch_crossing_count": branch_crossing_count,
        "sigma_task_velocity_min": _summary(min_sigma),
        "max_required_velocity_utilization": _summary(max_demand),
        "strict_directional_alpha": _summary(strict_alpha),
        "translation_error_m": _summary(translation_error),
        "rotation_error_deg": _summary(rotation_error),
        "selected_structural_abs_deg": _summary(structural_abs_deg),
        "max_velocity_deg_s": _summary(max_velocity_deg_s),
        "max_acceleration_deg_s2": _summary(max_acceleration_deg_s2),
        "max_jerk_deg_s3": _summary(max_jerk_deg_s3),
        **({"frames": frames} if include_frames else {}),
    }


def summarize_profile_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [case for case in cases if case.get("status") == "completed"]
    rejected = [case for case in cases if case.get("status") != "completed"]
    return {
        "case_count": len(cases),
        "completed_count": len(completed),
        "seed_rejected_count": len(rejected),
        "case_with_hold_count": sum(
            int(int(case.get("hold_count", 0)) > 0) for case in completed
        ),
        "case_with_branch_crossing_count": sum(
            int(int(case.get("branch_crossing_count", 0)) > 0)
            for case in completed
        ),
        "minimum_sigma_by_case": _summary(
            [
                float(case["sigma_task_velocity_min"]["min"])
                for case in completed
                if case["sigma_task_velocity_min"]["min"] is not None
            ]
        ),
        "maximum_demand_by_case": _summary(
            [
                float(case["max_required_velocity_utilization"]["max"])
                for case in completed
                if case["max_required_velocity_utilization"]["max"] is not None
            ]
        ),
        "maximum_translation_error_by_case_m": _summary(
            [
                float(case["translation_error_m"]["max"])
                for case in completed
                if case["translation_error_m"]["max"] is not None
            ]
        ),
        "maximum_rotation_error_by_case_deg": _summary(
            [
                float(case["rotation_error_deg"]["max"])
                for case in completed
                if case["rotation_error_deg"]["max"] is not None
            ]
        ),
        "maximum_velocity_by_case_deg_s": _summary(
            [
                float(case["max_velocity_deg_s"]["max"])
                for case in completed
                if case["max_velocity_deg_s"]["max"] is not None
            ]
        ),
        "maximum_acceleration_by_case_deg_s2": _summary(
            [
                float(case["max_acceleration_deg_s2"]["max"])
                for case in completed
                if case["max_acceleration_deg_s2"]["max"] is not None
            ]
        ),
        "rejections": [
            {
                "case_key": case.get("case_key"),
                "error": case.get("error"),
            }
            for case in rejected
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-motion",
        action="store_true",
        required=True,
        help="Required acknowledgement: no device access or dispatch is allowed.",
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode", type=int, action="append")
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument("--left-tool-profile", required=True)
    parser.add_argument("--right-tool-profile", required=True)
    parser.add_argument("--scene-profile", default=DEFAULT_COLLISION_SCENE)
    parser.add_argument(
        "--profile",
        choices=RM65_POSE_LIFT_PROFILE_CHOICES,
        action="append",
        help="Repeat to evaluate identical cases under multiple profiles.",
    )
    parser.add_argument("--reference-hz", type=float, default=60.0)
    parser.add_argument("--duration-sec", type=float, default=0.5)
    parser.add_argument(
        "--weighted-task-speed-per-sec",
        type=float,
        default=0.04,
        help="Norm of the weighted local Pink task velocity.",
    )
    parser.add_argument(
        "--warmstart-joint-speed-deg-s",
        type=float,
        default=5.0,
    )
    parser.add_argument("--offset-deg", type=float, action="append")
    parser.add_argument(
        "--initial-motion",
        choices=INITIAL_MOTION_CHOICES,
        action="append",
    )
    parser.add_argument(
        "--direction",
        choices=DIRECTION_CHOICES,
        action="append",
    )
    parser.add_argument(
        "--direction-sign",
        choices=("negative", "positive", "both"),
        default="both",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_CHOICES,
        action="append",
    )
    parser.add_argument(
        "--min-other-structural-abs-deg",
        type=float,
        default=8.0,
    )
    parser.add_argument("--split", choices=("all", "tune", "heldout"), default="all")
    parser.add_argument("--holdout-modulus", type=int, default=4)
    parser.add_argument("--max-seeds", type=int, default=16)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--include-frames", action="store_true")
    parser.add_argument("--include-constrained-feasibility", action="store_true")
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Write the full report to --output-json but print only bounded "
            "settings and per-profile summaries."
        ),
    )
    parser.add_argument("--output-json", default="")
    return parser


def _build_planner(
    profile_name: str,
    *,
    args: argparse.Namespace,
    endpoint_to_tool: Mapping[str, np.ndarray],
) -> RM65PositionCollisionPlanner:
    return RM65PositionCollisionPlanner(
        RM65PositionCollisionPlannerConfig(
            reference_hz=float(args.reference_hz),
            scene_profile=str(args.scene_profile),
            max_joint_velocity_rad_s=(
                REPLAY_POSITION_COLLISION_MAX_JOINT_VELOCITY_RAD_S
            ),
            max_joint_acceleration_rad_s2=(
                RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2
            ),
            damping=REPLAY_POSITION_IK_DAMPING,
            solve_max_iterations=4,
            solve_min_iterations=4,
            solve_adaptive_iterations_enabled=False,
            solve_iteration_reserve_ms=REPLAY_POSITION_IK_ITERATION_RESERVE_MS,
            pose_lift_profile=str(profile_name),
            ik_tuning=build_rm65_pose_lift_ik_tuning(
                position_control_profile="rm65_bimanual_position_v1",
                pose_lift_profile=str(profile_name),
                orientation_weight=REPLAY_POSITION_IK_ORIENTATION_WEIGHT,
            ),
        ),
        endpoint_to_tool_by_side=endpoint_to_tool,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if bool(args.all_episodes) == bool(args.episode):
        raise SystemExit("provide exactly one of --episode or --all-episodes")
    if int(args.max_seeds) < 0 or int(args.max_cases) < 0:
        raise SystemExit("--max-seeds and --max-cases must be non-negative")
    if int(args.holdout_modulus) < 2:
        raise SystemExit("--holdout-modulus must be at least two")
    for name in (
        "reference_hz",
        "duration_sec",
        "weighted_task_speed_per_sec",
        "warmstart_joint_speed_deg_s",
        "min_other_structural_abs_deg",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0 or (
            name
            in {"reference_hz", "duration_sec", "weighted_task_speed_per_sec"}
            and value <= 0.0
        ):
            raise SystemExit(
                f"--{name.replace('_', '-')} has an invalid value"
            )
    episodes = (
        read_replay_dataset_episode_indices(args.dataset_root)
        if args.all_episodes
        else list(args.episode)
    )
    group_args = argparse.Namespace(
        replay_dataset=str(Path(args.dataset_root).expanduser().resolve()),
        replay_parquet="",
        replay_episode=0,
        replay_action_column="action",
        replay_all_episodes=False,
        active_sides=SIDES,
    )
    groups = [
        build_replay_group_for_episode(
            group_args,
            episode_index=int(episode),
            group_spec=ReplayGroupSpec(
                key="wind_tunnel",
                start_index=0,
                end_index=-1,
                episode_range=str(episode),
            ),
            balanced_episode_count=1,
            require_arm_joint_targets=True,
        )
        for episode in episodes
    ]
    seeds = select_wind_seeds(
        groups,
        min_other_structural_abs_deg=float(
            args.min_other_structural_abs_deg
        ),
        holdout_modulus=int(args.holdout_modulus),
        split=str(args.split),
        max_seeds=int(args.max_seeds),
    )
    signs = (
        (-1, 1)
        if args.direction_sign == "both"
        else ((-1,) if args.direction_sign == "negative" else (1,))
    )
    cases = build_wind_cases(
        seeds,
        offsets_deg=tuple(args.offset_deg or (-0.75, 0.75)),
        initial_motions=tuple(args.initial_motion or ("stationary",)),
        directions=tuple(args.direction or ("weakest",)),
        direction_signs=signs,
        scenarios=tuple(args.scenario or ("persistent_push",)),
        max_cases=int(args.max_cases),
    )
    endpoint_to_tool = {
        side: endpoint_to_tool_transform(
            load_rm65_tool_profile(
                getattr(args, f"{side}_tool_profile"),
                side=side,
            )
        )
        for side in SIDES
    }
    groups_by_episode = _group_by_episode(groups)
    profiles = tuple(
        dict.fromkeys(args.profile or (RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,))
    )
    profile_reports: dict[str, Any] = {}
    for profile_name in profiles:
        profile_cases = [
            run_wind_case(
                _build_planner(
                    profile_name,
                    args=args,
                    endpoint_to_tool=endpoint_to_tool,
                ),
                groups_by_episode[case.seed.episode_index],
                case,
                reference_hz=float(args.reference_hz),
                duration_sec=float(args.duration_sec),
                weighted_task_speed_per_sec=float(
                    args.weighted_task_speed_per_sec
                ),
                warmstart_joint_speed_deg_s=float(
                    args.warmstart_joint_speed_deg_s
                ),
                include_frames=bool(args.include_frames),
                include_constrained_feasibility=bool(
                    args.include_constrained_feasibility
                ),
            )
            for case in cases
        ]
        profile_reports[profile_name] = {
            "summary": summarize_profile_cases(profile_cases),
            "cases": profile_cases,
        }
    report = {
        "schema": "teleop_runtime.rm65_singularity_wind_tunnel/v1",
        "device_access": False,
        "dispatch": False,
        "motion_authority": "none",
        "seed_contract": "recorded_accepted_qref_with_one_q3_or_q5_replaced",
        "target_contract": (
            "fixed_initial_weighted_pink_frame_task_singular_direction"
        ),
        "profiles": list(profiles),
        "episode_count": len(groups),
        "seed_count": len(seeds),
        "case_count_per_profile": len(cases),
        "settings": {
            "reference_hz": float(args.reference_hz),
            "duration_sec": float(args.duration_sec),
            "weighted_task_speed_per_sec": float(
                args.weighted_task_speed_per_sec
            ),
            "warmstart_joint_speed_deg_s": float(
                args.warmstart_joint_speed_deg_s
            ),
            "offsets_deg": list(args.offset_deg or (-0.75, 0.75)),
            "initial_motions": list(
                args.initial_motion or ("stationary",)
            ),
            "directions": list(args.direction or ("weakest",)),
            "direction_signs": list(signs),
            "scenarios": list(
                args.scenario or ("persistent_push",)
            ),
            "split": str(args.split),
            "holdout_modulus": int(args.holdout_modulus),
            "min_other_structural_abs_deg": float(
                args.min_other_structural_abs_deg
            ),
        },
        "seeds": [_plain(asdict(seed)) for seed in seeds],
        "profile_reports": profile_reports,
    }
    payload = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if str(args.output_json).strip():
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    printed_report = (
        report
        if not bool(args.compact)
        else {
            "schema": report["schema"],
            "device_access": False,
            "dispatch": False,
            "profiles": report["profiles"],
            "episode_count": report["episode_count"],
            "seed_count": report["seed_count"],
            "case_count_per_profile": report["case_count_per_profile"],
            "settings": report["settings"],
            "profile_summaries": {
                profile_name: {
                    key: value
                    for key, value in profile_report["summary"].items()
                    if key != "rejections"
                }
                for profile_name, profile_report in profile_reports.items()
            },
            "full_report": (
                str(Path(args.output_json).expanduser().resolve())
                if str(args.output_json).strip()
                else None
            ),
        }
    )
    print(
        json.dumps(
            printed_report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
