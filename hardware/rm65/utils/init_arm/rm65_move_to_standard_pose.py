from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.force_compensation import finite_list, norm  # noqa: E402
from hardware.rm65.ik_provider import RM65SDKIKProvider  # noqa: E402
from hardware.rm65.rm_api2_adapter import LEFT_ARM_IP, RIGHT_ARM_IP  # noqa: E402
from hardware.rm65.safety_diagnostics import check_rm65_joint_safety  # noqa: E402
from rm65_inspire_tac.ik_tuning import pose_delta_metrics  # noqa: E402
from rm65_inspire_tac.pipeline import build_dual_arm_scene_profile  # noqa: E402
from hardware.rm65.utils.init_arm.rm65_init_pose_common import (  # noqa: E402
    connect_robot,
    default_pose_path,
    format_values,
    interpolate_pose,
    joint_delta_deg,
    load_pose_file,
    pose_delta,
    read_force,
    read_state,
    recorded_tool_frame_for_side,
    require_zero,
    selected_sides,
    side_ip,
    target_pose_for_side,
    timestamp,
    use_tool_frame,
    write_json,
)
from rm65_inspire_tac.multitask_ik import MultiTaskIKSolver  # noqa: E402
from robot_motion.composite_provider import CompositeKinematicsProvider  # noqa: E402
from robot_motion.ik_resolver import IKSafetyResolver  # noqa: E402
from robot_motion.joint_limits import (  # noqa: E402
    JointVelocityLimitProfile,
    load_joint_velocity_limit_profiles,
    plan_limited_joint_path,
)
from robot_motion.pose import matrix_to_pose6  # noqa: E402
from robot_motion.pinocchio_provider import PinocchioIKProvider  # noqa: E402
from robot_motion.profiles import IKSafetyProfile, load_ik_safety_profiles  # noqa: E402
from robot_motion.seed_expansion import load_seed_expansion_profiles  # noqa: E402
from robot_motion.types import IKResolveRequest, IKResolveResult, IKResolveStatus  # noqa: E402


CONFIRM_TEXT = "INIT_RM65_STANDARD_POSE"
DUAL_ARM_COLLISION_PROFILES = (
    "dual_arm",
    "dual_arm_hand_proxy_open",
    "dual_arm_hand",
    "dual_arm_cylinder_table",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Move one or both RM65 arms smoothly to a recorded teleoperation "
            "standard pose. The standard file stores pose, not qpos."
        )
    )
    parser.add_argument("--arm", choices=("left", "right", "both"), default="left")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--pose-json", default=default_pose_path())
    parser.add_argument("--tool-frame", default="empty")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Must be exactly {CONFIRM_TEXT} for motion.")
    parser.add_argument(
        "--command-mode",
        choices=("ik-joint", "pose-canfd"),
        default="pose-canfd",
        help=(
            "pose-canfd sends slow absolute pose waypoints with Movep_CANFD; "
            "ik-joint is a fallback that commands online IK joint waypoints."
        ),
    )
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--min-duration-sec", type=float, default=10.0)
    parser.add_argument("--max-linear-velocity-m-s", type=float, default=0.025)
    parser.add_argument("--max-angular-velocity-rad-s", type=float, default=0.12)
    parser.add_argument("--hold-sec", type=float, default=1.0)
    parser.add_argument("--trajectory-mode", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--radio", type=int, default=50)
    parser.add_argument("--max-linear-distance-m", type=float, default=1.20)
    parser.add_argument("--max-angular-distance-rad", type=float, default=6.50)
    parser.add_argument("--max-interp-linear-step-m", type=float, default=0.003)
    parser.add_argument("--max-interp-angular-step-rad", type=float, default=0.020)
    parser.add_argument("--position-tolerance-m", type=float, default=0.050)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.160)
    parser.add_argument(
        "--root-joint-max-delta-deg",
        default="70,90",
        help="J1,J2 max absolute delta from start. Use 0,0 to disable relative root limits.",
    )
    parser.add_argument(
        "--root-joint-absolute-limits-deg",
        default="",
        help="Optional J1_min,J1_max,J2_min,J2_max absolute limits in degrees.",
    )
    parser.add_argument("--max-total-joint-delta-deg", type=float, default=160.0)
    parser.add_argument(
        "--max-ik-step-joint-delta-deg",
        type=float,
        default=5.0,
        help=(
            "Legacy IK refinement diagnostic threshold. Commanded joint steps are "
            "rate-limited by --joint-velocity-limit-profile."
        ),
    )
    parser.add_argument(
        "--joint-velocity-limit-profile",
        default="rm65",
        help="Robot-family joint velocity limit profile used for command rate limiting.",
    )
    parser.add_argument(
        "--joint-velocity-limits-json",
        default="",
        help="Optional joint velocity limit JSON. Empty uses robot_motion/configs/joint_velocity_limits.json.",
    )
    parser.add_argument(
        "--ik-safety-profile",
        default="init_arm",
        help="Profile name from the IK safety profile JSON.",
    )
    parser.add_argument(
        "--ik-safety-profile-json",
        default="",
        help="Optional IK safety profile JSON. Empty uses robot_motion/configs/default_ik_safety_profiles.json.",
    )
    parser.add_argument(
        "--ik-jump-refine-max-depth",
        type=int,
        default=6,
        help=(
            "When safe IK solving itself fails on a Cartesian segment, recursively split "
            "that segment up to 2^N substeps before failing. Solved joint targets that "
            "exceed velocity limits are smoothed by the joint velocity limiter instead."
        ),
    )
    parser.add_argument("--max-observed-joint-step-deg", type=float, default=12.0)
    parser.add_argument("--max-tracking-linear-m", type=float, default=0.20)
    parser.add_argument("--max-tracking-angular-rad", type=float, default=0.80)
    parser.add_argument("--abort-on-singularity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--controller-avoid-singularity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--configure-singularity-thresholds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--singularity-elbow-threshold-deg", type=float, default=0.5)
    parser.add_argument("--singularity-wrist-threshold-deg", type=float, default=0.5)
    parser.add_argument("--singularity-shoulder-threshold-m", type=float, default=0.01)
    parser.add_argument("--check-self-collision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dual-arm-collision-ik",
        choices=("required", "auto", "off"),
        default="required",
        help=(
            "For --arm both, solve joint waypoints with the dual-arm URDF collision model. "
            "Use 'off' only for explicit fallback; 'auto' warns and falls back if dependencies are missing."
        ),
    )
    parser.add_argument(
        "--collision-scene-profile",
        choices=DUAL_ARM_COLLISION_PROFILES,
        default="dual_arm_hand_proxy_open",
        help=(
            "Dual-arm URDF scene profile. dual_arm_hand_proxy_open includes fixed endpoint/hand "
            "collision boxes on arm_link6/right_arm_link6 plus a fixed table box."
        ),
    )
    parser.add_argument("--collision-min-distance-m", type=float, default=0.03)
    parser.add_argument("--collision-distance-epsilon-m", type=float, default=1e-4)
    parser.add_argument("--collision-barrier-gain", type=float, default=10.0)
    parser.add_argument("--collision-safe-displacement-gain", type=float, default=0.001)
    parser.add_argument("--dual-arm-ik-solver", default="")
    parser.add_argument("--dual-arm-ik-frequency", type=float, default=30.0)
    parser.add_argument("--dual-arm-ik-iterations", type=int, default=8)
    parser.add_argument("--dual-arm-ik-position-tolerance-m", type=float, default=0.008)
    parser.add_argument("--dual-arm-ik-orientation-tolerance-rad", type=float, default=0.12)
    parser.add_argument("--dual-arm-posture-cost", type=float, default=0.05)
    parser.add_argument("--monitor-force-source", choices=("zero", "work", "tool"), default="work")
    parser.add_argument("--hard-stop-force-n", type=float, default=40.0)
    parser.add_argument("--hard-stop-force-axis-n", type=float, default=40.0)
    parser.add_argument("--hard-stop-torque-nm", type=float, default=2.0)
    parser.add_argument("--state-read-every", type=int, default=2)
    parser.add_argument("--force-read-every", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--stop-settle-sec", type=float, default=0.5)
    parser.add_argument("--output-json", default="")
    return parser


def _parse_float_csv(text: str, *, expected_len: int, label: str, allow_empty: bool = False) -> list[float]:
    stripped = str(text).strip()
    if not stripped and allow_empty:
        return []
    try:
        values = [float(item.strip()) for item in stripped.split(",") if item.strip()]
    except ValueError as exc:
        raise RuntimeError(f"invalid {label}: '{text}'") from exc
    if len(values) != expected_len:
        raise RuntimeError(f"{label} must contain {expected_len} comma-separated values")
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{label} contains non-finite values")
    return values


def _validate_args(args) -> None:
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"motion requires --confirm {CONFIRM_TEXT}")
    if not str(args.tool_frame).strip():
        raise RuntimeError("--tool-frame must be non-empty; use 'empty' for the bare arm end")
    for name in (
        "control_hz",
        "min_duration_sec",
        "max_linear_velocity_m_s",
        "max_angular_velocity_rad_s",
        "max_linear_distance_m",
        "max_angular_distance_rad",
        "max_interp_linear_step_m",
        "max_interp_angular_step_rad",
        "position_tolerance_m",
        "orientation_tolerance_rad",
        "max_total_joint_delta_deg",
        "max_ik_step_joint_delta_deg",
        "max_observed_joint_step_deg",
        "max_tracking_linear_m",
        "max_tracking_angular_rad",
        "collision_min_distance_m",
        "collision_barrier_gain",
        "collision_safe_displacement_gain",
        "dual_arm_ik_frequency",
        "dual_arm_ik_position_tolerance_m",
        "dual_arm_ik_orientation_tolerance_rad",
        "dual_arm_posture_cost",
        "singularity_elbow_threshold_deg",
        "singularity_wrist_threshold_deg",
        "singularity_shoulder_threshold_m",
        "hard_stop_force_n",
        "hard_stop_force_axis_n",
        "hard_stop_torque_nm",
    ):
        if getattr(args, name) <= 0.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be positive")
    if args.collision_distance_epsilon_m < 0.0:
        raise RuntimeError("--collision-distance-epsilon-m must be >= 0")
    if args.dual_arm_ik_iterations < 1:
        raise RuntimeError("--dual-arm-ik-iterations must be >= 1")
    if args.ik_jump_refine_max_depth < 0:
        raise RuntimeError("--ik-jump-refine-max-depth must be >= 0")
    if args.hold_sec < 0.0:
        raise RuntimeError("--hold-sec must be >= 0")
    if args.stop_settle_sec < 0.0:
        raise RuntimeError("--stop-settle-sec must be >= 0")
    for name in ("state_read_every", "force_read_every", "print_every"):
        if getattr(args, name) < 1:
            raise RuntimeError(f"--{name.replace('_', '-')} must be >= 1")
    args.root_joint_max_delta = _parse_float_csv(
        args.root_joint_max_delta_deg,
        expected_len=2,
        label="--root-joint-max-delta-deg",
    )
    absolute_limits = _parse_float_csv(
        args.root_joint_absolute_limits_deg,
        expected_len=4,
        label="--root-joint-absolute-limits-deg",
        allow_empty=True,
    )
    if absolute_limits and (
        absolute_limits[0] >= absolute_limits[1]
        or absolute_limits[2] >= absolute_limits[3]
    ):
        raise RuntimeError("--root-joint-absolute-limits-deg min values must be less than max values")
    args.root_joint_absolute_limits = absolute_limits


def _load_init_ik_safety_profile(args) -> IKSafetyProfile:
    profiles = load_ik_safety_profiles(args.ik_safety_profile_json)
    profile_name = str(args.ik_safety_profile).strip()
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise RuntimeError(f"--ik-safety-profile '{profile_name}' not found; available={available}")
    profile = profiles[profile_name]
    return replace(profile, reject_singular=bool(args.abort_on_singularity))


def _load_joint_velocity_limit_profile(args) -> JointVelocityLimitProfile:
    profiles = load_joint_velocity_limit_profiles(args.joint_velocity_limits_json)
    profile_name = str(args.joint_velocity_limit_profile).strip()
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise RuntimeError(f"--joint-velocity-limit-profile '{profile_name}' not found; available={available}")
    return profiles[profile_name]


def _build_ik_safety_resolver(robot, args) -> IKSafetyResolver:
    from rm65_inspire_tac.ik import SingleArmIKSolver

    seed_expansion_profile = load_seed_expansion_profiles()["rm65"]
    provider = CompositeKinematicsProvider(
        (
            RM65SDKIKProvider(robot),
            PinocchioIKProvider(
                solver=SingleArmIKSolver(),
                source="rm65_pinocchio",
                qpos_size=6,
            ),
        ),
        names=("rm65_sdk", "rm65_pinocchio"),
        primary_index=0,
    )
    return IKSafetyResolver(
        profile=_load_init_ik_safety_profile(args),
        provider=provider,
        seed_expansion_profile=seed_expansion_profile,
    )


def _defer_single_arm_ik_to_dual_arm_multitask(args) -> bool:
    return str(args.arm) == "both" and str(args.dual_arm_collision_ik) != "off"


def _root_limits(start_joint: list[float], args) -> list[tuple[float, float]]:
    limits = [(-math.inf, math.inf), (-math.inf, math.inf)]
    for index, delta in enumerate(args.root_joint_max_delta):
        if delta > 0.0:
            limits[index] = (start_joint[index] - delta, start_joint[index] + delta)
    absolute = args.root_joint_absolute_limits
    if absolute:
        absolute_pairs = [(absolute[0], absolute[1]), (absolute[2], absolute[3])]
        limited = []
        for index in range(2):
            lo = max(limits[index][0], absolute_pairs[index][0])
            hi = min(limits[index][1], absolute_pairs[index][1])
            if lo > hi:
                raise RuntimeError(f"J{index + 1} root limit window is empty")
            limited.append((lo, hi))
        limits = limited
    for index, (lo, hi) in enumerate(limits):
        if start_joint[index] < lo or start_joint[index] > hi:
            raise RuntimeError(
                f"start J{index + 1}={start_joint[index]:.3f}deg is outside "
                f"root limits [{lo:.3f}, {hi:.3f}]deg"
            )
    return limits


def _check_root_limits(joint: list[float], limits: list[tuple[float, float]], label: str) -> None:
    for index, (lo, hi) in enumerate(limits):
        value = joint[index]
        if value < lo or value > hi:
            raise RuntimeError(
                f"{label} J{index + 1}={value:.3f}deg outside "
                f"root limits [{lo:.3f}, {hi:.3f}]deg"
            )


def _singularity_summary(robot, joint: list[float]) -> tuple[int | None, str]:
    try:
        result = robot.Algo_Kin_Singularity_Analyse(joint)
    except Exception as exc:  # noqa: BLE001 - diagnostic path.
        return None, f"unavailable ({exc})"
    if not isinstance(result, tuple) or len(result) < 2:
        return None, str(result)
    status, distance = result[:2]
    labels = {
        0: "normal",
        -1: "shoulder_singularity",
        -2: "elbow_singularity",
        -3: "wrist_singularity",
    }
    try:
        status_int = int(status)
    except (TypeError, ValueError):
        return None, str(result)
    label = labels.get(status_int, f"unknown_status_{status_int}")
    try:
        return status_int, f"{label} wrist_center_distance_m={float(distance):.6f}"
    except (TypeError, ValueError):
        return status_int, f"{label} wrist_center_distance_m={distance}"


def _configure_singularity_safety(robot, side: str, args) -> None:
    if args.configure_singularity_thresholds:
        setter = getattr(robot, "Algo_Kin_Set_Singularity_Thresholds", None)
        if setter is None:
            print(f"WARNING: {side}: SDK does not expose analytic singularity threshold configuration")
        else:
            setter(
                args.singularity_elbow_threshold_deg,
                args.singularity_wrist_threshold_deg,
                args.singularity_shoulder_threshold_m,
            )
            getter = getattr(robot, "Algo_Kin_Get_Singularity_Thresholds", None)
            if getter is None:
                print(
                    f"{side}: singularity thresholds set "
                    f"elbow={args.singularity_elbow_threshold_deg:.3f}deg "
                    f"wrist={args.singularity_wrist_threshold_deg:.3f}deg "
                    f"shoulder={args.singularity_shoulder_threshold_m:.3f}m"
                )
            else:
                try:
                    qe, qw, d = getter()
                    print(f"{side}: singularity thresholds elbow={qe:.3f}deg wrist={qw:.3f}deg shoulder={d:.3f}m")
                except Exception as exc:  # noqa: BLE001 - diagnostics only.
                    print(f"WARNING: {side}: failed to read singularity thresholds after setting: {exc}")

    if args.controller_avoid_singularity:
        setter = getattr(robot, "Set_Avoid_Singularity_Mode", None)
        if setter is None:
            print(f"WARNING: {side}: SDK does not expose controller avoid-singularity mode")
            return
        code = setter(1)
        if int(code) != 0:
            print(
                f"WARNING: {side}: Set_Avoid_Singularity_Mode(1) failed with return code {code}; "
                "continuing with preflight singularity checks only."
            )
            return
        getter = getattr(robot, "Get_Avoid_Singularity_Mode", None)
        if getter is None:
            print(f"{side}: controller avoid-singularity mode enabled")
            return
        try:
            code, mode = getter()
            require_zero(code, f"{side} Get_Avoid_Singularity_Mode")
            print(f"{side}: controller avoid-singularity mode={mode}")
        except Exception as exc:  # noqa: BLE001 - diagnostics only.
            print(f"WARNING: {side}: failed to read avoid-singularity mode after setting: {exc}")


def _check_self_collision(
    robot,
    joint: list[float],
    *,
    side: str = "",
    step_label: object = "",
    previous_joint: list[float] | None = None,
    start_joint: list[float] | None = None,
    context: dict[str, object] | None = None,
) -> str:
    report = check_rm65_joint_safety(
        robot,
        joint,
        side=side,
        label=step_label,
        previous_joint_deg=previous_joint,
        start_joint_deg=start_joint,
        check_position_limit=True,
        check_self_collision=True,
        context=context,
    )
    report.raise_if_unsafe()
    return report.status_text


def _solve_ik(robot, side: str, seed_joint: list[float], pose: list[float], step: object) -> list[float]:
    code, joint = robot.Algo_Inverse_Kinematics(seed_joint, pose, flag=1)
    if int(code) != 0:
        _, singularity = _singularity_summary(robot, seed_joint)
        raise RuntimeError(
            f"{side}: IK failed at step={step} code={code} "
            f"seed_joint={format_values(seed_joint, precision=3)} "
            f"pose={format_values(pose, precision=6)} seed_singularity={singularity}"
        )
    return finite_list(joint, expected_len=6, label="ik_joint")


def _validate_ik_waypoint(
    robot,
    side: str,
    joint: list[float],
    previous_joint: list[float],
    start_joint: list[float],
    root_limits: list[tuple[float, float]],
    step_label: object,
    args,
    *,
    enforce_step_delta: bool,
    safety_context: dict[str, object] | None = None,
) -> tuple[float, float, str, str]:
    _check_root_limits(joint, root_limits, f"{side}: IK step {step_label}")
    step_delta = joint_delta_deg(previous_joint, joint)
    total_delta = joint_delta_deg(start_joint, joint)
    if enforce_step_delta and step_delta > args.max_ik_step_joint_delta_deg:
        raise RuntimeError(
            f"{side}: IK step joint jump {step_delta:.3f}deg > "
            f"--max-ik-step-joint-delta-deg at step={step_label}"
        )
    if total_delta > args.max_total_joint_delta_deg:
        raise RuntimeError(
            f"{side}: IK total joint delta {total_delta:.3f}deg > "
            f"--max-total-joint-delta-deg at step={step_label}"
        )
    status, singularity = _singularity_summary(robot, joint)
    if args.abort_on_singularity and status not in (0, None):
        raise RuntimeError(
            f"{side}: singularity at IK step {step_label}: {singularity} "
            f"joint_deg={format_values(joint, precision=3)} "
            f"J3={joint[2]:.3f}deg J5={joint[4]:.3f}deg. "
            "Skipping this waypoint is unsafe; reduce the path difficulty, use a staged standard pose, "
            "or adjust the singularity thresholds only if you accept the risk."
        )
    self_collision_status = "not_checked"
    if args.check_self_collision:
        context = {
            "phase": "ik_waypoint",
            "enforce_step_delta": bool(enforce_step_delta),
            "computed_step_delta_deg": float(step_delta),
            "computed_total_delta_deg": float(total_delta),
        }
        if safety_context:
            context.update(dict(safety_context))
        self_collision_status = _check_self_collision(
            robot,
            joint,
            side=side,
            step_label=step_label,
            previous_joint=previous_joint,
            start_joint=start_joint,
            context=context,
        )
    return step_delta, total_delta, singularity, self_collision_status


def _waypoint_record(
    *,
    step: object,
    alpha: float,
    pose: list[float],
    joint: list[float],
    step_delta: float,
    total_delta: float,
    singularity: str,
) -> dict[str, object]:
    return {
        "phase": "move",
        "step": step,
        "alpha": alpha,
        "pose": pose,
        "joint_deg": joint,
        "ik_step_joint_delta_deg": step_delta,
        "ik_total_joint_delta_deg": total_delta,
        "singularity": singularity,
    }


def _refine_ik_jump(
    robot,
    side: str,
    start_pose: list[float],
    target_pose: list[float],
    start_joint: list[float],
    root_limits: list[tuple[float, float]],
    previous_alpha: float,
    target_alpha: float,
    previous_joint: list[float],
    nominal_step: int,
    args,
) -> tuple[list[dict[str, object]], str] | None:
    max_depth = int(args.ik_jump_refine_max_depth)
    if max_depth <= 0:
        return None

    last_failure = ""
    for depth in range(1, max_depth + 1):
        parts = 2**depth
        local_seed = list(previous_joint)
        local_previous = list(previous_joint)
        refined: list[dict[str, object]] = []
        self_collision_status = "not_checked"
        max_refined_step = 0.0
        failed = False
        for sub_index in range(1, parts + 1):
            alpha = previous_alpha + (target_alpha - previous_alpha) * (sub_index / parts)
            pose = interpolate_pose(start_pose, target_pose, alpha)
            step_label = f"{nominal_step}.{sub_index}/{parts}"
            try:
                joint = _solve_ik(robot, side, local_seed, pose, step_label)
                (
                    step_delta,
                    total_delta,
                    singularity,
                    self_collision_status,
                ) = _validate_ik_waypoint(
                    robot,
                    side,
                    joint,
                    local_previous,
                    start_joint,
                    root_limits,
                    step_label,
                    args,
                    enforce_step_delta=True,
                )
            except RuntimeError as exc:
                last_failure = str(exc)
                failed = True
                break
            max_refined_step = max(max_refined_step, step_delta)
            refined.append(
                _waypoint_record(
                    step=step_label,
                    alpha=alpha,
                    pose=pose,
                    joint=joint,
                    step_delta=step_delta,
                    total_delta=total_delta,
                    singularity=singularity,
                )
            )
            local_seed = joint
            local_previous = joint
        if not failed:
            print(
                f"{side}: refined IK jump at step={nominal_step} into {parts} substeps "
                f"max_step_delta={max_refined_step:.3f}deg"
            )
            return refined, self_collision_status

    if last_failure:
        print(
            f"{side}: IK jump refinement failed at step={nominal_step} "
            f"after max_depth={max_depth}: {last_failure}"
        )
    return None


def _pose6_to_matrix(pose: list[float]) -> np.ndarray:
    values = np.asarray(finite_list(pose, expected_len=6, label="pose"), dtype=float)
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = values[:3]
    matrix[:3, :3] = R.from_euler("xyz", values[3:6], degrees=False).as_matrix()
    return matrix


def _invert_rigid_matrix(matrix: np.ndarray) -> np.ndarray:
    transform = np.asarray(matrix, dtype=float).reshape(4, 4)
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -(inverse[:3, :3] @ transform[:3, 3])
    return inverse


def _matrix_to_solver_target(pose: np.ndarray) -> np.ndarray:
    matrix = np.asarray(pose, dtype=float).reshape(4, 4)
    quat_xyzw = R.from_matrix(matrix[:3, :3]).as_quat()
    return np.array(
        [
            matrix[0, 3],
            matrix[1, 3],
            matrix[2, 3],
            quat_xyzw[0],
            quat_xyzw[1],
            quat_xyzw[2],
            quat_xyzw[3],
        ],
        dtype=float,
    )


def _joint_deg_to_rad(joint_deg: list[float]) -> np.ndarray:
    return (
        np.asarray(finite_list(joint_deg, expected_len=6, label="joint_deg"), dtype=float)
        * math.pi
        / 180.0
    )


def _joint_rad_to_deg(joint_rad: np.ndarray) -> list[float]:
    return [float(value) * 180.0 / math.pi for value in np.asarray(joint_rad, dtype=float).reshape(6)]


def _forward_pose_from_joint(robot, joint: list[float], *, label: str) -> list[float]:
    provider = RM65SDKIKProvider(robot)
    matrix = provider.forward_kinematics(_joint_deg_to_rad(joint))
    if matrix is None:
        raise RuntimeError(f"forward kinematics unavailable for {label}")
    return matrix_to_pose6(matrix)


def _ik_result_text(result: IKResolveResult) -> str:
    metadata = dict(result.metadata)
    best = metadata.get("best_rejected")
    if result.accepted:
        return (
            f"status={result.status.value} source={result.candidate_source} "
            f"fraction={result.tolerance_fraction:.2f} "
            f"pose_error={result.pose_error_m}/{result.pose_error_deg} "
            f"rejections={metadata.get('rejection_count')}"
        )
    if isinstance(best, dict):
        return (
            f"status={result.status.value} reason={best.get('reject_reason')} "
            f"source={best.get('candidate_source')} score={best.get('score')}"
        )
    return f"status={result.status.value} message={result.message}"


def _resolve_safe_ik_waypoint(
    *,
    resolver: IKSafetyResolver,
    side: str,
    previous_joint: list[float],
    previous_pose: list[float],
    seed_joint: list[float],
    target_pose: list[float],
    step_label: object,
) -> tuple[list[float], list[float], IKResolveResult]:
    reference_qpos = _joint_deg_to_rad(previous_joint)
    seed_qpos = _joint_deg_to_rad(seed_joint)
    request = IKResolveRequest(
        target_pose=_pose6_to_matrix(target_pose),
        reference_qpos=reference_qpos,
        reference_pose=_pose6_to_matrix(previous_pose),
        seed_qpos=(seed_qpos,),
        seed_labels=("seed",),
        previous_safe_qpos=reference_qpos,
        metadata={"side": side, "step": str(step_label)},
    )
    result = resolver.resolve(request)
    if not result.accepted or result.qpos is None:
        raise RuntimeError(f"{side}: safe IK failed at step={step_label}: {_ik_result_text(result)}")
    joint = _joint_rad_to_deg(result.qpos)
    effective_pose = matrix_to_pose6(result.effective_target_pose)
    if result.status == IKResolveStatus.RELAXED:
        print(
            f"WARNING: {side}: relaxed IK target at step={step_label} "
            f"fraction={result.tolerance_fraction:.2f} "
            f"source={result.candidate_source}"
        )
    return joint, effective_pose, result


def _refine_safe_ik_transition(
    *,
    resolver: IKSafetyResolver,
    robot,
    side: str,
    start_pose: list[float],
    target_pose: list[float],
    start_joint: list[float],
    root_limits: list[tuple[float, float]],
    previous_alpha: float,
    target_alpha: float,
    previous_joint: list[float],
    previous_pose: list[float],
    nominal_step: int,
    args,
) -> tuple[list[dict[str, object]], str] | None:
    max_depth = int(args.ik_jump_refine_max_depth)
    if max_depth <= 0:
        return None

    last_failure = ""
    for depth in range(1, max_depth + 1):
        parts = 2**depth
        local_seed = list(previous_joint)
        local_previous_joint = list(previous_joint)
        local_previous_pose = list(previous_pose)
        refined: list[dict[str, object]] = []
        self_collision_status = "not_checked"
        max_refined_step = 0.0
        failed = False
        for sub_index in range(1, parts + 1):
            alpha = previous_alpha + (target_alpha - previous_alpha) * (sub_index / parts)
            requested_pose = interpolate_pose(start_pose, target_pose, alpha)
            step_label = f"{nominal_step}.{sub_index}/{parts}"
            try:
                joint, effective_pose, result = _resolve_safe_ik_waypoint(
                    resolver=resolver,
                    side=side,
                    previous_joint=local_previous_joint,
                    previous_pose=local_previous_pose,
                    seed_joint=local_seed,
                    target_pose=requested_pose,
                    step_label=step_label,
                )
                (
                    step_delta,
                    total_delta,
                    singularity,
                    self_collision_status,
                ) = _validate_ik_waypoint(
                    robot,
                    side,
                    joint,
                    local_previous_joint,
                    start_joint,
                    root_limits,
                    step_label,
                    args,
                    enforce_step_delta=True,
                )
            except RuntimeError as exc:
                last_failure = str(exc)
                failed = True
                break
            max_refined_step = max(max_refined_step, step_delta)
            record = _waypoint_record(
                step=step_label,
                alpha=alpha,
                pose=effective_pose,
                joint=joint,
                step_delta=step_delta,
                total_delta=total_delta,
                singularity=singularity,
            )
            record["requested_pose"] = requested_pose
            record["ik_safety"] = dict(result.metadata)
            refined.append(record)
            local_seed = joint
            local_previous_joint = joint
            local_previous_pose = effective_pose
        if not failed:
            print(
                f"{side}: refined safe IK transition at step={nominal_step} into {parts} substeps "
                f"max_step_delta={max_refined_step:.3f}deg"
            )
            return refined, self_collision_status

    if last_failure:
        print(
            f"{side}: safe IK transition refinement failed at step={nominal_step} "
            f"after max_depth={max_depth}: {last_failure}"
        )
    return None


def _append_rate_limited_joint_transition(
    *,
    waypoints: list[dict[str, object]],
    robot,
    side: str,
    previous_joint: list[float],
    target_joint: list[float],
    start_joint: list[float],
    root_limits: list[tuple[float, float]],
    step_label: object,
    alpha: float,
    requested_pose: list[float],
    target_pose: list[float],
    ik_safety: dict[str, object],
    joint_velocity_profile: JointVelocityLimitProfile,
    period_sec: float,
    args,
) -> tuple[list[float], list[float], str, bool]:
    start_rad = _joint_deg_to_rad(previous_joint)
    target_rad = _joint_deg_to_rad(target_joint)
    plan = plan_limited_joint_path(
        start_qpos=start_rad,
        target_qpos=target_rad,
        profile=joint_velocity_profile,
        dt_sec=period_sec,
    )
    limited = len(plan) > 1 or bool(plan[0].limited)
    if limited:
        raw_delta = np.rad2deg(np.abs(target_rad - start_rad))
        max_step = joint_velocity_profile.max_step_deg(period_sec)
        print(
            f"WARNING: {side}: IK joint target step={step_label} exceeds joint velocity limit; "
            f"max_delta={float(np.max(raw_delta)):.3f}deg "
            f"max_allowed_step={float(np.max(max_step)):.3f}deg "
            f"inserted_rate_limited_steps={len(plan)} "
            f"profile={joint_velocity_profile.name}. Executing joint-limited path toward latest target."
        )

    current_previous = list(previous_joint)
    final_pose = list(target_pose)
    self_collision_status = "not_checked"
    for plan_index, step in enumerate(plan, start=1):
        joint = _joint_rad_to_deg(step.qpos)
        local_label = (
            step_label
            if len(plan) == 1
            else f"{step_label}.rate{plan_index}/{len(plan)}"
        )
        (
            step_delta,
            total_delta,
            singularity,
            self_collision_status,
        ) = _validate_ik_waypoint(
            robot,
            side,
            joint,
            current_previous,
            start_joint,
            root_limits,
            local_label,
            args,
            enforce_step_delta=False,
            safety_context={
                "phase": "rate_limited_joint_transition" if limited else "ik_waypoint",
                "requested_step": str(step_label),
                "rate_limited_step": str(local_label),
                "requested_pose": requested_pose,
                "ik_target_pose": target_pose,
                "ik_target_joint_deg": list(target_joint),
                "ik_target_delta_from_previous_deg": np.rad2deg(target_rad - start_rad).tolist(),
                "joint_rate_limited": bool(limited),
                "joint_rate_limit_index": int(plan_index),
                "joint_rate_limit_count": int(len(plan)),
                "joint_velocity_limit_profile": joint_velocity_profile.name,
                "joint_velocity_limit_delta_deg": step.delta_deg.tolist(),
                "joint_velocity_limit_max_step_deg": step.max_step_deg.tolist(),
                "joint_velocity_limit_ratio": float(step.limit_ratio),
            },
        )
        pose = (
            list(target_pose)
            if plan_index == len(plan) and not limited
            else _forward_pose_from_joint(robot, joint, label=f"{side} step {local_label}")
        )
        record = _waypoint_record(
            step=local_label,
            alpha=alpha,
            pose=pose,
            joint=joint,
            step_delta=step_delta,
            total_delta=total_delta,
            singularity=singularity,
        )
        record["requested_pose"] = requested_pose
        record["ik_target_pose"] = target_pose
        record["ik_safety"] = dict(ik_safety)
        record["joint_rate_limited"] = bool(limited)
        record["joint_velocity_limit_profile"] = joint_velocity_profile.name
        record["joint_velocity_limit_max_step_deg"] = step.max_step_deg.tolist()
        record["joint_velocity_limit_delta_deg"] = step.delta_deg.tolist()
        record["joint_velocity_limit_ratio"] = float(step.limit_ratio)
        record["joint_rate_limit_index"] = plan_index
        record["joint_rate_limit_count"] = len(plan)
        waypoints.append(record)
        current_previous = joint
        final_pose = pose
    return current_previous, final_pose, self_collision_status, limited


def _embed_dual_arm_qpos(profile, current_full_q: np.ndarray, left_rad: np.ndarray, right_rad: np.ndarray) -> np.ndarray:
    full_q = np.asarray(current_full_q, dtype=float).copy()
    if profile.solution_layout == "dual_arm_hand" and full_q.shape[0] >= 24:
        full_q[0:6] = left_rad
        full_q[18:24] = right_rad
        return full_q
    if full_q.shape[0] == 14:
        full_q[1:7] = left_rad
        full_q[8:14] = right_rad
        return full_q
    if full_q.shape[0] == 12:
        full_q[0:6] = left_rad
        full_q[6:12] = right_rad
        return full_q
    raise RuntimeError(f"unsupported dual-arm solver q shape {full_q.shape}")


def _split_dual_arm_qpos(profile, full_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(full_q, dtype=float).reshape(-1)
    if profile.solution_layout == "dual_arm_hand" and q.shape[0] >= 24:
        return q[0:6].copy(), q[18:24].copy()
    if q.shape[0] == 14:
        return q[1:7].copy(), q[8:14].copy()
    if q.shape[0] == 12:
        return q[0:6].copy(), q[6:12].copy()
    raise RuntimeError(f"unsupported dual-arm solver q shape {q.shape}")


def _duration_for_move(linear: float, angular: float, args) -> float:
    return max(
        args.min_duration_sec,
        linear / args.max_linear_velocity_m_s,
        angular / args.max_angular_velocity_rad_s,
        linear / (args.max_interp_linear_step_m * args.control_hz),
        angular / (args.max_interp_angular_step_rad * args.control_hz),
    )


def _prepare_context(side: str, pose_data: dict[str, object], args) -> dict[str, Any]:
    ip = side_ip(side, args)
    robot = connect_robot(side, ip)
    try:
        active_tool_frame = use_tool_frame(robot, side, args.tool_frame)
        _configure_singularity_safety(robot, side, args)
        recorded_tool_frame = recorded_tool_frame_for_side(pose_data, side)
        if recorded_tool_frame and recorded_tool_frame != args.tool_frame:
            raise RuntimeError(
                f"{side}: standard pose was recorded in tool frame '{recorded_tool_frame}', "
                f"but --tool-frame is '{args.tool_frame}'"
            )

        start_state = read_state(robot, side, tool_frame=args.tool_frame)
        target_pose = target_pose_for_side(pose_data, side)
        linear, angular = pose_delta(start_state["pose"], target_pose)
        print(f"{side}: start_joint_deg={format_values(start_state['joint_deg'], precision=3)}")
        print(f"{side}: start_pose={format_values(start_state['pose'], precision=6)}")
        print(f"{side}: target_pose={format_values(target_pose, precision=6)}")
        print(f"{side}: target_error={linear:.6f}m/{angular:.6f}rad")

        if linear > args.max_linear_distance_m:
            raise RuntimeError(f"{side}: linear distance {linear:.6f}m > --max-linear-distance-m")
        if angular > args.max_angular_distance_rad:
            raise RuntimeError(f"{side}: angular distance {angular:.6f}rad > --max-angular-distance-rad")

        root_limits = _root_limits(start_state["joint_deg"], args)
        root_text = ", ".join(
            f"J{index + 1}=[{lo:.2f},{hi:.2f}]deg"
            for index, (lo, hi) in enumerate(root_limits)
        )
        print(f"{side}: root_limits {root_text}")

        start_force = read_force(robot, side)
        source = start_force[args.monitor_force_source]
        start_force_norm = norm(source[:3])
        start_force_axis_peak = max(abs(float(value)) for value in source[:3])
        start_torque_norm = norm(source[3:])
        print(
            f"{side}: start_{args.monitor_force_source}_force="
            f"{start_force_norm:.3f}N axis_peak={start_force_axis_peak:.3f}N "
            f"torque={start_torque_norm:.3f}Nm"
        )
        if start_force_norm > args.hard_stop_force_n:
            raise RuntimeError(f"{side}: start force {start_force_norm:.3f}N > --hard-stop-force-n")
        if start_force_axis_peak > args.hard_stop_force_axis_n:
            raise RuntimeError(
                f"{side}: start force axis peak {start_force_axis_peak:.3f}N > "
                "--hard-stop-force-axis-n"
            )
        if start_torque_norm > args.hard_stop_torque_nm:
            raise RuntimeError(f"{side}: start torque {start_torque_norm:.3f}Nm > --hard-stop-torque-nm")

        skipped = linear <= args.position_tolerance_m and angular <= args.orientation_tolerance_rad
        duration = 0.0 if skipped else _duration_for_move(linear, angular, args)
        steps = 0 if skipped else max(2, int(math.ceil(duration * args.control_hz)))
        hold_steps = 0 if skipped else max(0, int(math.ceil(args.hold_sec * args.control_hz)))
        period = 1.0 / args.control_hz
        step_linear = 0.0 if steps <= 0 else linear / steps
        step_angular = 0.0 if steps <= 0 else angular / steps
        waypoints: list[dict[str, object]] = []
        self_collision_status = "skipped"
        force_joint_waypoints = False

        if skipped:
            print(f"{side}: already within tolerance; no motion waypoints will be generated")
        else:
            print(
                f"{side}: preflight online IK duration={duration:.3f}s "
                f"steps={steps} hold_steps={hold_steps} "
                f"interp_step={step_linear:.6f}m/{step_angular:.6f}rad"
            )
            if _defer_single_arm_ik_to_dual_arm_multitask(args):
                self_collision_status = "deferred_to_dual_arm_multitask"
                print(
                    f"{side}: deferring single-arm IK candidate generation to "
                    "dual-arm MultiTask collision IK; RM SDK IK provider will not be used "
                    "for this dual-arm solve."
                )
                print(f"{side}: preflight ok self_collision={self_collision_status}")
            else:
                joint_velocity_profile = _load_joint_velocity_limit_profile(args)
                max_joint_step = joint_velocity_profile.max_step_deg(period)
                print(
                    f"{side}: joint_velocity_limit_profile={joint_velocity_profile.name} "
                    f"max_velocity_deg_s={format_values(joint_velocity_profile.max_velocity_deg_s, precision=3)} "
                    f"max_command_step_deg={format_values(max_joint_step.tolist(), precision=3)}"
                )
                resolver = _build_ik_safety_resolver(robot, args)
                provider_names = getattr(resolver.provider, "names", ())
                provider_text = ",".join(str(name) for name in provider_names) if provider_names else type(resolver.provider).__name__
                print(
                    f"{side}: ik_safety_profile={resolver.profile.name} "
                    f"providers={provider_text} "
                    f"score_policy=max_joint_delta_then_norm "
                    f"tolerance={resolver.profile.target_tolerance.position_m:.4f}m/"
                    f"{resolver.profile.target_tolerance.orientation_deg:.2f}deg "
                    f"relaxation={list(resolver.profile.relaxation_fractions)} "
                    f"hard_branch_joints={list(resolver.profile.hard_branch_joint_indices)} "
                    f"hard_branch_delta_deg={list(resolver.profile.hard_branch_max_delta_deg)}"
                )
                seed = list(start_state["joint_deg"])
                previous_joint = list(seed)
                previous_pose = list(start_state["pose"])
                previous_alpha = 0.0
                self_collision_status = "not_checked"
                for index in range(1, steps + 1):
                    alpha = index / steps
                    requested_pose = interpolate_pose(start_state["pose"], target_pose, alpha)
                    try:
                        joint, pose, result = _resolve_safe_ik_waypoint(
                            resolver=resolver,
                            side=side,
                            previous_joint=previous_joint,
                            previous_pose=previous_pose,
                            seed_joint=seed,
                            target_pose=requested_pose,
                            step_label=index,
                        )
                    except RuntimeError as exc:
                        refined = _refine_safe_ik_transition(
                            resolver=resolver,
                            robot=robot,
                            side=side,
                            start_pose=start_state["pose"],
                            target_pose=target_pose,
                            start_joint=start_state["joint_deg"],
                            root_limits=root_limits,
                            previous_alpha=previous_alpha,
                            target_alpha=alpha,
                            previous_joint=previous_joint,
                            previous_pose=previous_pose,
                            nominal_step=index,
                            args=args,
                        )
                        if refined is not None:
                            refined_waypoints, self_collision_status = refined
                            waypoints.extend(refined_waypoints)
                            seed = finite_list(
                                refined_waypoints[-1]["joint_deg"],
                                expected_len=6,
                                label="refined_joint",
                            )
                            previous_joint = list(seed)
                            previous_pose = finite_list(
                                refined_waypoints[-1]["pose"],
                                expected_len=6,
                                label="refined_pose",
                            )
                            previous_alpha = alpha
                            continue
                        raise RuntimeError(
                            f"{side}: safe IK failed at step={index}; "
                            f"requested_pose={format_values(requested_pose, precision=6)} "
                            f"previous_joint={format_values(previous_joint, precision=3)} "
                            f"reason={exc}"
                        ) from exc
                    previous_joint, previous_pose, self_collision_status, rate_limited = (
                        _append_rate_limited_joint_transition(
                            waypoints=waypoints,
                            robot=robot,
                            side=side,
                            previous_joint=previous_joint,
                            target_joint=joint,
                            start_joint=start_state["joint_deg"],
                            root_limits=root_limits,
                            step_label=index,
                            alpha=alpha,
                            requested_pose=requested_pose,
                            target_pose=pose,
                            ik_safety=dict(result.metadata),
                            joint_velocity_profile=joint_velocity_profile,
                            period_sec=period,
                            args=args,
                        )
                    )
                    force_joint_waypoints = force_joint_waypoints or rate_limited
                    seed = list(previous_joint)
                    previous_alpha = alpha

                if hold_steps:
                    final_joint = finite_list(waypoints[-1]["joint_deg"], expected_len=6, label="final_joint")
                    final_pose = finite_list(waypoints[-1]["pose"], expected_len=6, label="final_pose")
                    for hold_index in range(1, hold_steps + 1):
                        waypoints.append(
                            {
                                "phase": "hold",
                                "step": steps + hold_index,
                                "alpha": 1.0,
                                "pose": final_pose,
                                "requested_pose": target_pose,
                                "joint_deg": final_joint,
                                "ik_step_joint_delta_deg": 0.0,
                                "ik_total_joint_delta_deg": joint_delta_deg(start_state["joint_deg"], final_joint),
                                "singularity": waypoints[-1]["singularity"],
                                "ik_safety": dict(waypoints[-1].get("ik_safety", {})),
                            }
                        )
                print(f"{side}: preflight ok self_collision={self_collision_status}")

        return {
            "side": side,
            "ip": ip,
            "robot": robot,
            "tool_frame": args.tool_frame,
            "active_tool_frame": active_tool_frame,
            "recorded_tool_frame": recorded_tool_frame,
            "start_state": start_state,
            "start_force": start_force,
            "target_pose": target_pose,
            "initial_error_linear_m": linear,
            "initial_error_angular_rad": angular,
            "root_limits": root_limits,
            "duration_sec": duration,
            "steps": steps,
            "hold_steps": hold_steps,
            "period_sec": period,
            "interp_step_linear_m": step_linear,
            "interp_step_angular_rad": step_angular,
            "waypoints": waypoints,
            "skipped": skipped,
            "force_joint_waypoints": force_joint_waypoints,
            "self_collision_status": self_collision_status,
            "records": [],
            "error": "",
        }
    except Exception:
        robot.Arm_Socket_Close()
        raise


def _contexts_by_side(contexts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(ctx["side"]): ctx for ctx in contexts}


def _dual_arm_target_transforms(profile, solver, start_full_q: np.ndarray, by_side: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    transforms: dict[str, np.ndarray] = {}
    frame_by_side = {
        "left": profile.left_frame_name,
        "right": profile.right_frame_name,
    }
    nominal_transform_by_side = {
        "left": profile.left_pose_transform,
        "right": profile.right_pose_transform,
    }
    for side in ("left", "right"):
        ctx = by_side[side]
        frame_pose = solver.compute_frame_pose(frame_by_side[side], start_full_q)
        nominal_start = nominal_transform_by_side[side] @ _pose6_to_matrix(ctx["start_state"]["pose"])
        correction = frame_pose @ _invert_rigid_matrix(nominal_start)
        correction_translation = float(np.linalg.norm(correction[:3, 3]))
        correction_rotation_deg = float(
            R.from_matrix(correction[:3, :3]).magnitude() * 180.0 / math.pi
        )
        if correction_rotation_deg > 5.0:
            raise RuntimeError(
                f"{side}: dual-arm IK start pose frame mismatch rotation "
                f"{correction_rotation_deg:.3f}deg > 5deg. Check tool frame, URDF frame, "
                "and controller pose convention before executing initialization."
            )
        transforms[side] = correction @ nominal_transform_by_side[side]
        aligned_start = transforms[side] @ _pose6_to_matrix(ctx["start_state"]["pose"])
        aligned_linear, aligned_angular_deg = pose_delta_metrics(frame_pose, aligned_start)
        print(
            f"{side}: dual-arm IK start-frame alignment "
            f"correction={correction_translation:.4f}m/{correction_rotation_deg:.4f}deg "
            f"residual={aligned_linear:.6f}m/{aligned_angular_deg:.6f}deg"
        )
    return transforms


def _check_dual_arm_collision_with_fixed_peer(
    *,
    profile,
    solver,
    start_full_q: np.ndarray,
    fixed_ctx: dict[str, Any],
    moving_ctx: dict[str, Any],
    args,
    start_distance: float | None,
    hand_proxy_text: str,
) -> bool:
    waypoints = moving_ctx.get("waypoints") or []
    if not waypoints:
        return False

    fixed_side = str(fixed_ctx["side"])
    moving_side = str(moving_ctx["side"])
    fixed_rad = _joint_deg_to_rad(fixed_ctx["start_state"]["joint_deg"])
    min_distance_seen = start_distance
    distance_cache: dict[tuple[float, ...], float | None] = {}

    for index, waypoint in enumerate(waypoints, start=1):
        moving_rad = _joint_deg_to_rad(
            finite_list(waypoint["joint_deg"], expected_len=6, label=f"{moving_side}_waypoint_joint")
        )
        if moving_side == "left":
            full_q = _embed_dual_arm_qpos(profile, start_full_q, moving_rad, fixed_rad)
        else:
            full_q = _embed_dual_arm_qpos(profile, start_full_q, fixed_rad, moving_rad)
        cache_key = tuple(round(float(value), 10) for value in full_q)
        if cache_key in distance_cache:
            min_distance = distance_cache[cache_key]
        else:
            min_distance = solver.compute_min_collision_distance(full_q)
            distance_cache[cache_key] = min_distance
        waypoint["dual_arm_collision_distance_m"] = min_distance
        if min_distance is not None:
            min_distance_seen = (
                min_distance
                if min_distance_seen is None
                else min(float(min_distance_seen), float(min_distance))
            )
            if min_distance < args.collision_min_distance_m - args.collision_distance_epsilon_m:
                raise RuntimeError(
                    f"dual-arm fixed-peer collision distance {min_distance:.4f}m < "
                    f"--collision-min-distance-m {args.collision_min_distance_m:.4f}m "
                    f"at {moving_side} waypoint={index} with {fixed_side} fixed"
                )

    moving_ctx["force_joint_waypoints"] = True
    metadata = {
        "enabled": True,
        "mode": "fixed_peer_collision_check",
        "fixed_side": fixed_side,
        "moving_side": moving_side,
        "scene_profile": profile.name,
        "urdf_path": str(profile.urdf_path),
        "collision_pair_count": len(profile.collision_pairs),
        "start_min_distance_m": start_distance,
        "min_distance_m": min_distance_seen,
        "endpoint_proxy": hand_proxy_text,
    }
    fixed_ctx["dual_arm_collision_ik"] = {
        **metadata,
        "self_collision_status": str(fixed_ctx.get("self_collision_status", "skipped")),
    }
    moving_ctx["dual_arm_collision_ik"] = {
        **metadata,
        "self_collision_status": str(moving_ctx.get("self_collision_status", "not_checked")),
    }
    print(
        "dual-arm collision fast preflight ok "
        f"fixed={fixed_side} moving={moving_side} waypoints={len(waypoints)} "
        f"min_distance={min_distance_seen}. "
        f"{moving_side} will execute the collision-checked joint waypoints with Movej_CANFD."
    )
    return True


def _solve_dual_arm_collision_waypoints(contexts: list[dict[str, Any]], args) -> None:
    by_side = _contexts_by_side(contexts)
    if set(by_side) != {"left", "right"}:
        return
    if args.dual_arm_collision_ik == "off":
        print("dual_arm_collision_ik=off; using independently planned arm waypoints.")
        return

    left_ctx = by_side["left"]
    right_ctx = by_side["right"]
    profile = build_dual_arm_scene_profile(args.collision_scene_profile)
    try:
        solver = MultiTaskIKSolver(
            robot_urdf_path=profile.urdf_path,
            robot_package_dirs=list(profile.package_dirs),
            robot_root_joint=None,
            robot_list_of_joints_to_lock=None,
            robot_collision_pair_names=list(profile.collision_pairs),
            solver=str(args.dual_arm_ik_solver).strip() or None,
            frequency=float(args.dual_arm_ik_frequency),
            debug_viz=False,
            debug_profile=False,
            sleep=False,
            collision_barrier_gain=float(args.collision_barrier_gain),
            collision_safe_displacement_gain=float(args.collision_safe_displacement_gain),
            collision_min_distance_m=float(args.collision_min_distance_m),
        )
    except Exception as exc:  # noqa: BLE001 - dependency/setup errors should be actionable.
        message = (
            "dual-arm collision IK is unavailable. It requires pinocchio, pink, qpsolvers, "
            f"and a QP backend in the active environment. scene_profile={profile.name} "
            f"urdf={profile.urdf_path} error={type(exc).__name__}: {exc}"
        )
        if args.dual_arm_collision_ik == "auto":
            print(f"WARNING: {message}. Falling back to independently planned arm waypoints.")
            return
        raise RuntimeError(message) from exc

    solver.add_frame_task(profile.left_frame_name, position_cost=1.0, orientation_cost=1.0)
    solver.add_frame_task(profile.right_frame_name, position_cost=1.0, orientation_cost=1.0)
    solver.add_posture_task(cost=float(args.dual_arm_posture_cost))

    left_start_rad = _joint_deg_to_rad(left_ctx["start_state"]["joint_deg"])
    right_start_rad = _joint_deg_to_rad(right_ctx["start_state"]["joint_deg"])
    start_full_q = _embed_dual_arm_qpos(profile, solver.current_q, left_start_rad, right_start_rad)
    solver.set_configuration_q(start_full_q)
    target_transforms = _dual_arm_target_transforms(profile, solver, start_full_q, by_side)

    hand_proxy_text = "none"
    if profile.hand_proxy_boxes:
        hand_proxy_text = ", ".join(
            f"{spec.geometry_name}@{spec.parent_link_name}"
            for spec in profile.hand_proxy_boxes
        )
    print(
        "dual-arm collision IK enabled "
        f"profile={profile.name} urdf={profile.urdf_path} "
        f"pairs={len(profile.collision_pairs)} min_distance={args.collision_min_distance_m:.3f}m "
        f"endpoint_proxy={hand_proxy_text}"
    )
    if args.command_mode != "ik-joint":
        print("dual-arm collision guard uses checked joint waypoints; executing them with Movej_CANFD.")

    start_distance = solver.compute_min_collision_distance(start_full_q)
    if start_distance is not None:
        print(f"dual-arm collision IK start_min_distance={start_distance:.4f}m")
        if start_distance < args.collision_min_distance_m - args.collision_distance_epsilon_m:
            raise RuntimeError(
                f"dual-arm start min collision distance {start_distance:.4f}m < "
                f"--collision-min-distance-m {args.collision_min_distance_m:.4f}m"
            )

    skipped_contexts = [ctx for ctx in (left_ctx, right_ctx) if bool(ctx.get("skipped"))]
    moving_contexts = [ctx for ctx in (left_ctx, right_ctx) if not bool(ctx.get("skipped"))]
    if len(skipped_contexts) == 1 and len(moving_contexts) == 1:
        if _check_dual_arm_collision_with_fixed_peer(
            profile=profile,
            solver=solver,
            start_full_q=start_full_q,
            fixed_ctx=skipped_contexts[0],
            moving_ctx=moving_contexts[0],
            args=args,
            start_distance=start_distance,
            hand_proxy_text=hand_proxy_text,
        ):
            return

    steps = max(int(left_ctx["steps"]), int(right_ctx["steps"]))
    hold_steps = max(int(left_ctx["hold_steps"]), int(right_ctx["hold_steps"]))
    if steps <= 0:
        for ctx in (left_ctx, right_ctx):
            ctx["dual_arm_collision_ik"] = {
                "enabled": True,
                "scene_profile": profile.name,
                "urdf_path": str(profile.urdf_path),
                "collision_pair_count": len(profile.collision_pairs),
                "start_min_distance_m": start_distance,
                "min_distance_m": start_distance,
                "endpoint_proxy": hand_proxy_text,
            }
        print("dual-arm collision IK current-pose check complete; both arms are within tolerance.")
        return

    duration = max(float(left_ctx["duration_sec"]), float(right_ctx["duration_sec"]))
    period = 1.0 / args.control_hz
    joint_velocity_profile = _load_joint_velocity_limit_profile(args)
    max_joint_step = joint_velocity_profile.max_step_deg(period)
    print(
        "dual-arm joint velocity limiter "
        f"profile={joint_velocity_profile.name} "
        f"max_command_step_deg={format_values(max_joint_step.tolist(), precision=3)}"
    )
    previous_joint = {
        "left": list(left_ctx["start_state"]["joint_deg"]),
        "right": list(right_ctx["start_state"]["joint_deg"]),
    }
    generated = {"left": [], "right": []}
    min_distance_seen = start_distance
    final_singularity = {"left": "not_checked", "right": "not_checked"}
    self_collision_status = {"left": "not_checked", "right": "not_checked"}
    pose_error = {
        "left": {"translation_m": 0.0, "rotation_deg": 0.0},
        "right": {"translation_m": 0.0, "rotation_deg": 0.0},
    }

    for index in range(1, steps + 1):
        alpha = index / steps
        left_pose = interpolate_pose(left_ctx["start_state"]["pose"], left_ctx["target_pose"], alpha)
        right_pose = interpolate_pose(right_ctx["start_state"]["pose"], right_ctx["target_pose"], alpha)
        left_target = target_transforms["left"] @ _pose6_to_matrix(left_pose)
        right_target = target_transforms["right"] @ _pose6_to_matrix(right_pose)
        targets = [_matrix_to_solver_target(left_target), _matrix_to_solver_target(right_target)]

        solved_q = solver.current_q.copy()
        for iteration in range(1, int(args.dual_arm_ik_iterations) + 1):
            solved_q = solver.solve(targets=targets)
            current_left = solver.compute_frame_pose(profile.left_frame_name, solved_q)
            current_right = solver.compute_frame_pose(profile.right_frame_name, solved_q)
            left_linear, left_angular_deg = pose_delta_metrics(current_left, left_target)
            right_linear, right_angular_deg = pose_delta_metrics(current_right, right_target)
            pose_error = {
                "left": {
                    "translation_m": float(left_linear),
                    "rotation_deg": float(left_angular_deg),
                },
                "right": {
                    "translation_m": float(right_linear),
                    "rotation_deg": float(right_angular_deg),
                },
            }
            if (
                left_linear <= args.dual_arm_ik_position_tolerance_m
                and right_linear <= args.dual_arm_ik_position_tolerance_m
                and math.radians(left_angular_deg) <= args.dual_arm_ik_orientation_tolerance_rad
                and math.radians(right_angular_deg) <= args.dual_arm_ik_orientation_tolerance_rad
            ):
                break

        min_distance = solver.compute_min_collision_distance(solved_q)
        if min_distance is not None:
            min_distance_seen = (
                min_distance
                if min_distance_seen is None
                else min(float(min_distance_seen), float(min_distance))
            )
            if min_distance < args.collision_min_distance_m - args.collision_distance_epsilon_m:
                raise RuntimeError(
                    f"dual-arm collision IK min distance {min_distance:.4f}m < "
                    f"--collision-min-distance-m {args.collision_min_distance_m:.4f}m at step={index}"
                )

        left_rad, right_rad = _split_dual_arm_qpos(profile, solved_q)
        target_joints = {
            "left": _joint_rad_to_deg(left_rad),
            "right": _joint_rad_to_deg(right_rad),
        }
        poses = {"left": left_pose, "right": right_pose}
        plans = {
            side: plan_limited_joint_path(
                start_qpos=_joint_deg_to_rad(previous_joint[side]),
                target_qpos=_joint_deg_to_rad(target_joints[side]),
                profile=joint_velocity_profile,
                dt_sec=period,
            )
            for side in ("left", "right")
        }
        substeps = max(len(plans["left"]), len(plans["right"]))
        if substeps > 1:
            raw_left = np.rad2deg(
                np.abs(_joint_deg_to_rad(target_joints["left"]) - _joint_deg_to_rad(previous_joint["left"]))
            )
            raw_right = np.rad2deg(
                np.abs(_joint_deg_to_rad(target_joints["right"]) - _joint_deg_to_rad(previous_joint["right"]))
            )
            print(
                "WARNING: dual-arm IK joint target exceeds joint velocity limit "
                f"step={index} inserted_rate_limited_steps={substeps} "
                f"left_max_delta={float(np.max(raw_left)):.3f}deg "
                f"right_max_delta={float(np.max(raw_right)):.3f}deg "
                f"profile={joint_velocity_profile.name}"
            )

        for sub_index in range(1, substeps + 1):
            sub_rad = {}
            for side in ("left", "right"):
                plan = plans[side]
                sub_rad[side] = plan[min(sub_index, len(plan)) - 1].qpos
            full_q = _embed_dual_arm_qpos(profile, start_full_q, sub_rad["left"], sub_rad["right"])
            sub_distance = solver.compute_min_collision_distance(full_q)
            if sub_distance is not None:
                min_distance_seen = (
                    sub_distance
                    if min_distance_seen is None
                    else min(float(min_distance_seen), float(sub_distance))
                )
                if sub_distance < args.collision_min_distance_m - args.collision_distance_epsilon_m:
                    raise RuntimeError(
                        f"WARNING: dual-arm rate-limited collision distance {sub_distance:.4f}m < "
                        f"--collision-min-distance-m {args.collision_min_distance_m:.4f}m "
                        f"at step={index}.{sub_index}/{substeps}; stopping before execution"
                    )

            step_label = index if substeps == 1 else f"{index}.rate{sub_index}/{substeps}"
            for side, ctx in (("left", left_ctx), ("right", right_ctx)):
                joint = _joint_rad_to_deg(sub_rad[side])
                _check_root_limits(joint, ctx["root_limits"], f"{side}: dual-arm IK step {step_label}")
                step_delta = joint_delta_deg(previous_joint[side], joint)
                total_delta = joint_delta_deg(ctx["start_state"]["joint_deg"], joint)
                if total_delta > args.max_total_joint_delta_deg:
                    raise RuntimeError(
                        f"{side}: dual-arm IK total joint delta {total_delta:.3f}deg > "
                        f"--max-total-joint-delta-deg at step={step_label}"
                    )
                status, singularity = _singularity_summary(ctx["robot"], joint)
                final_singularity[side] = singularity
                if args.abort_on_singularity and status not in (0, None):
                    raise RuntimeError(
                        f"{side}: singularity at dual-arm IK step {step_label}: {singularity} "
                        f"joint_deg={format_values(joint, precision=3)} "
                        f"J3={joint[2]:.3f}deg J5={joint[4]:.3f}deg"
                )
                if args.check_self_collision:
                    plan_step = plans[side][min(sub_index, len(plans[side])) - 1]
                    self_collision_status[side] = _check_self_collision(
                        ctx["robot"],
                        joint,
                        side=side,
                        step_label=step_label,
                        previous_joint=previous_joint[side],
                        start_joint=ctx["start_state"]["joint_deg"],
                        context={
                            "phase": "dual_arm_rate_limited_collision_ik",
                            "requested_step": str(index),
                            "rate_limited_step": str(step_label),
                            "requested_pose": poses[side],
                            "ik_target_joint_deg": target_joints[side],
                            "joint_rate_limited": bool(substeps > 1),
                            "joint_rate_limit_index": int(sub_index),
                            "joint_rate_limit_count": int(substeps),
                            "joint_velocity_limit_profile": joint_velocity_profile.name,
                            "joint_velocity_limit_delta_deg": plan_step.delta_deg.tolist(),
                            "joint_velocity_limit_max_step_deg": plan_step.max_step_deg.tolist(),
                            "joint_velocity_limit_ratio": float(plan_step.limit_ratio),
                            "dual_arm_collision_distance_m": sub_distance,
                            "dual_arm_collision_min_distance_m": args.collision_min_distance_m,
                        },
                    )
                pose = _forward_pose_from_joint(ctx["robot"], joint, label=f"{side} dual-arm step {step_label}")
                pose_linear, pose_angular = pose_delta(pose, poses[side])
                plan_step = plans[side][min(sub_index, len(plans[side])) - 1]
                generated[side].append(
                    {
                        "phase": "move",
                        "step": step_label,
                        "alpha": alpha,
                        "pose": pose,
                        "requested_pose": poses[side],
                        "joint_deg": joint,
                        "ik_step_joint_delta_deg": step_delta,
                        "ik_total_joint_delta_deg": total_delta,
                        "singularity": singularity,
                        "dual_arm_collision_distance_m": sub_distance,
                        "dual_arm_ik_pose_error": {
                            **dict(pose_error[side]),
                            "rate_limited_tracking_translation_m": pose_linear,
                            "rate_limited_tracking_rotation_rad": pose_angular,
                        },
                        "joint_rate_limited": bool(substeps > 1),
                        "joint_velocity_limit_profile": joint_velocity_profile.name,
                        "joint_velocity_limit_max_step_deg": plan_step.max_step_deg.tolist(),
                        "joint_velocity_limit_delta_deg": plan_step.delta_deg.tolist(),
                        "joint_velocity_limit_ratio": float(plan_step.limit_ratio),
                        "joint_rate_limit_index": sub_index,
                        "joint_rate_limit_count": substeps,
                    }
                )
                previous_joint[side] = joint

    if hold_steps:
        final_distance = solver.compute_min_collision_distance(solver.current_q)
        for side, ctx in (("left", left_ctx), ("right", right_ctx)):
            final_joint = finite_list(generated[side][-1]["joint_deg"], expected_len=6, label=f"{side}_final_joint")
            for hold_index in range(1, hold_steps + 1):
                generated[side].append(
                    {
                        "phase": "hold",
                        "step": steps + hold_index,
                        "alpha": 1.0,
                        "pose": ctx["target_pose"],
                        "joint_deg": final_joint,
                        "ik_step_joint_delta_deg": 0.0,
                        "ik_total_joint_delta_deg": joint_delta_deg(ctx["start_state"]["joint_deg"], final_joint),
                        "singularity": final_singularity[side],
                        "dual_arm_collision_distance_m": final_distance,
                        "dual_arm_ik_pose_error": dict(pose_error[side]),
                    }
                )

    for side, ctx in (("left", left_ctx), ("right", right_ctx)):
        ctx["waypoints"] = generated[side]
        ctx["steps"] = steps
        ctx["hold_steps"] = hold_steps
        ctx["duration_sec"] = duration
        ctx["period_sec"] = period
        ctx["force_joint_waypoints"] = True
        ctx["dual_arm_collision_ik"] = {
            "enabled": True,
            "scene_profile": profile.name,
            "urdf_path": str(profile.urdf_path),
            "collision_pair_count": len(profile.collision_pairs),
            "start_min_distance_m": start_distance,
            "min_distance_m": min_distance_seen,
            "endpoint_proxy": hand_proxy_text,
            "self_collision_status": self_collision_status[side],
        }
    print(
        "dual-arm collision IK preflight ok "
        f"steps={steps} hold_steps={hold_steps} min_distance={min_distance_seen} "
        f"self_collision_left={self_collision_status['left']} "
        f"self_collision_right={self_collision_status['right']}"
    )


def _check_runtime_limits(
    ctx: dict[str, Any],
    waypoint: dict[str, object],
    state: dict[str, list[float]],
    previous_joint: list[float],
    force: dict[str, list[float]] | None,
    args,
) -> tuple[float, float, float, float, float, float, float]:
    side = str(ctx["side"])
    joint = state["joint_deg"]
    _check_root_limits(joint, ctx["root_limits"], f"{side}: observed")
    total_delta = joint_delta_deg(ctx["start_state"]["joint_deg"], joint)
    observed_step = joint_delta_deg(previous_joint, joint)
    tracking_linear, tracking_angular = pose_delta(state["pose"], waypoint["pose"])
    if total_delta > args.max_total_joint_delta_deg:
        raise RuntimeError(f"{side}: observed total joint delta {total_delta:.3f}deg > --max-total-joint-delta-deg")
    if observed_step > args.max_observed_joint_step_deg:
        raise RuntimeError(f"{side}: observed joint step {observed_step:.3f}deg > --max-observed-joint-step-deg")
    if tracking_linear > args.max_tracking_linear_m:
        raise RuntimeError(f"{side}: tracking error {tracking_linear:.5f}m > --max-tracking-linear-m")
    if tracking_angular > args.max_tracking_angular_rad:
        raise RuntimeError(f"{side}: tracking error {tracking_angular:.5f}rad > --max-tracking-angular-rad")

    force_norm = 0.0
    force_axis_peak = 0.0
    torque_norm = 0.0
    if force is not None:
        source = force[args.monitor_force_source]
        force_norm = norm(source[:3])
        force_axis_peak = max(abs(float(value)) for value in source[:3])
        torque_norm = norm(source[3:])
        if force_norm > args.hard_stop_force_n:
            raise RuntimeError(f"{side}: force {force_norm:.3f}N > --hard-stop-force-n")
        if force_axis_peak > args.hard_stop_force_axis_n:
            raise RuntimeError(
                f"{side}: force axis peak {force_axis_peak:.3f}N > "
                "--hard-stop-force-axis-n"
            )
        if torque_norm > args.hard_stop_torque_nm:
            raise RuntimeError(f"{side}: torque {torque_norm:.3f}Nm > --hard-stop-torque-nm")
    return (
        total_delta,
        observed_step,
        tracking_linear,
        tracking_angular,
        force_norm,
        force_axis_peak,
        torque_norm,
    )


def _command_waypoint(ctx: dict[str, Any], waypoint: dict[str, object], args) -> None:
    robot = ctx["robot"]
    side = str(ctx["side"])
    use_joint_waypoint = args.command_mode == "ik-joint" or bool(ctx.get("force_joint_waypoints"))
    if use_joint_waypoint:
        require_zero(
            robot.Movej_CANFD(
                waypoint["joint_deg"],
                follow=False,
                trajectory_mode=args.trajectory_mode,
                radio=args.radio,
            ),
            f"{side} Movej_CANFD",
        )
    else:
        require_zero(
            robot.Movep_CANFD(
                waypoint["pose"],
                follow=False,
                trajectory_mode=args.trajectory_mode,
                radio=args.radio,
            ),
            f"{side} Movep_CANFD",
        )


def _execute_context(ctx: dict[str, Any], args, stop_event: threading.Event) -> None:
    side = str(ctx["side"])
    robot = ctx["robot"]
    waypoints = ctx["waypoints"]
    if ctx["skipped"] or not waypoints:
        return

    previous_observed_joint = list(ctx["start_state"]["joint_deg"])
    last_state = ctx["start_state"]
    next_deadline = time.perf_counter()
    motion_started = False
    max_force_norm = 0.0
    max_force_axis_peak = 0.0
    max_torque_norm = 0.0
    max_total_delta = 0.0
    max_observed_step = 0.0
    max_tracking_linear = 0.0
    max_tracking_angular = 0.0
    missed = 0
    records: list[dict[str, object]] = []
    try:
        for index, waypoint in enumerate(waypoints, start=1):
            if stop_event.is_set():
                break
            loop_start = time.perf_counter()
            _command_waypoint(ctx, waypoint, args)
            motion_started = True

            need_state = (
                index % args.state_read_every == 0
                or index == len(waypoints)
                or index % args.force_read_every == 0
            )
            if need_state:
                last_state = read_state(robot, side, tool_frame=str(ctx["tool_frame"]))
            force = None
            if index % args.force_read_every == 0 or index == len(waypoints):
                force = read_force(robot, side)

            total_delta = 0.0
            observed_step = 0.0
            tracking_linear = 0.0
            tracking_angular = 0.0
            force_norm = 0.0
            force_axis_peak = 0.0
            torque_norm = 0.0
            if need_state:
                (
                    total_delta,
                    observed_step,
                    tracking_linear,
                    tracking_angular,
                    force_norm,
                    force_axis_peak,
                    torque_norm,
                ) = _check_runtime_limits(
                    ctx,
                    waypoint,
                    last_state,
                    previous_observed_joint,
                    force,
                    args,
                )
                previous_observed_joint = list(last_state["joint_deg"])
                max_total_delta = max(max_total_delta, total_delta)
                max_observed_step = max(max_observed_step, observed_step)
                max_tracking_linear = max(max_tracking_linear, tracking_linear)
                max_tracking_angular = max(max_tracking_angular, tracking_angular)
                max_force_norm = max(max_force_norm, force_norm)
                max_force_axis_peak = max(max_force_axis_peak, force_axis_peak)
                max_torque_norm = max(max_torque_norm, torque_norm)

            loop_elapsed = time.perf_counter() - loop_start
            missed_deadline = loop_elapsed > ctx["period_sec"]
            if missed_deadline:
                missed += 1
            if index % args.print_every == 0 or index == len(waypoints):
                print(
                    f"{side}:init {waypoint['phase']} step={index}/{len(waypoints)} "
                    f"track={tracking_linear:.4f}m/{tracking_angular:.4f}rad "
                    f"joint_delta={total_delta:.3f}deg step_delta={observed_step:.3f}deg "
                    f"|F|={force_norm:.3f}N Faxis={force_axis_peak:.3f}N |T|={torque_norm:.3f}Nm"
                )
            records.append(
                {
                    "step": index,
                    "phase": waypoint["phase"],
                    "command_pose": waypoint["pose"],
                    "observed_pose": last_state["pose"] if need_state else None,
                    "observed_joint_deg": last_state["joint_deg"] if need_state else None,
                    "tracking_linear_m": tracking_linear,
                    "tracking_angular_rad": tracking_angular,
                    "joint_total_delta_deg": total_delta,
                    "joint_observed_step_deg": observed_step,
                    "force_norm_n": force_norm,
                    "force_axis_peak_n": force_axis_peak,
                    "torque_norm_nm": torque_norm,
                    "loop_elapsed_sec": loop_elapsed,
                    "missed_deadline": missed_deadline,
                }
            )
            next_deadline += ctx["period_sec"]
            sleep_sec = next_deadline - time.perf_counter()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
    except Exception as exc:  # noqa: BLE001 - worker must report and stop peer arms.
        ctx["error"] = str(exc)
        stop_event.set()
    finally:
        if motion_started:
            code = robot.Arm_Slow_Stop()
            print(f"{side}: Arm_Slow_Stop(init) -> {code}")
            time.sleep(args.stop_settle_sec)

    ctx["records"] = records
    ctx["runtime_summary"] = {
        "max_force_norm_n": max_force_norm,
        "max_force_axis_peak_n": max_force_axis_peak,
        "max_torque_norm_nm": max_torque_norm,
        "max_total_joint_delta_deg": max_total_delta,
        "max_observed_joint_step_deg": max_observed_step,
        "max_tracking_linear_m": max_tracking_linear,
        "max_tracking_angular_rad": max_tracking_angular,
        "missed_deadlines": missed,
    }


def _finalize_context(ctx: dict[str, Any], args) -> None:
    side = str(ctx["side"])
    robot = ctx["robot"]
    final_state = read_state(robot, side, tool_frame=str(ctx["tool_frame"]))
    final_force = read_force(robot, side)
    linear, angular = pose_delta(final_state["pose"], ctx["target_pose"])
    print(f"{side}: final_error={linear:.6f}m/{angular:.6f}rad")
    if not ctx["error"] and linear > args.position_tolerance_m:
        ctx["error"] = f"final position error {linear:.6f}m > --position-tolerance-m"
    if not ctx["error"] and angular > args.orientation_tolerance_rad:
        ctx["error"] = f"final orientation error {angular:.6f}rad > --orientation-tolerance-rad"
    ctx["final_state"] = final_state
    ctx["final_force"] = final_force
    ctx["final_error_linear_m"] = linear
    ctx["final_error_angular_rad"] = angular


def _report_context(ctx: dict[str, Any]) -> dict[str, object]:
    return {
        "side": ctx["side"],
        "ip": ctx["ip"],
        "tool_frame": ctx["tool_frame"],
        "active_tool_frame": ctx["active_tool_frame"],
        "recorded_tool_frame": ctx["recorded_tool_frame"],
        "start_state": ctx["start_state"],
        "target_pose": ctx["target_pose"],
        "initial_error_linear_m": ctx["initial_error_linear_m"],
        "initial_error_angular_rad": ctx["initial_error_angular_rad"],
        "duration_sec": ctx["duration_sec"],
        "steps": ctx["steps"],
        "hold_steps": ctx["hold_steps"],
        "interp_step_linear_m": ctx.get("interp_step_linear_m"),
        "interp_step_angular_rad": ctx.get("interp_step_angular_rad"),
        "root_limits": [
            [
                None if not math.isfinite(limit[0]) else limit[0],
                None if not math.isfinite(limit[1]) else limit[1],
            ]
            for limit in ctx["root_limits"]
        ],
        "skipped": ctx["skipped"],
        "error": ctx["error"],
        "force_joint_waypoints": bool(ctx.get("force_joint_waypoints")),
        "dual_arm_collision_ik": ctx.get("dual_arm_collision_ik", {"enabled": False}),
        "runtime_summary": ctx.get("runtime_summary", {}),
        "final_state": ctx.get("final_state"),
        "final_error_linear_m": ctx.get("final_error_linear_m"),
        "final_error_angular_rad": ctx.get("final_error_angular_rad"),
        "records": ctx.get("records", []),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contexts: list[dict[str, Any]] = []
    pose_data: dict[str, object] | None = None
    exit_code = 0
    try:
        _validate_args(args)
        pose_data = load_pose_file(Path(args.pose_json))
        for side in selected_sides(args.arm):
            contexts.append(_prepare_context(side, pose_data, args))
        _solve_dual_arm_collision_waypoints(contexts, args)

        print(
            f"command_mode={args.command_mode} execute={args.execute} "
            f"arms={','.join(str(ctx['side']) for ctx in contexts)}"
        )
        if not args.execute:
            print("DRY-RUN: preflight complete; no motion commands were sent.")
            return 0

        stop_event = threading.Event()
        threads = [
            threading.Thread(target=_execute_context, args=(ctx, args, stop_event), daemon=False)
            for ctx in contexts
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for ctx in contexts:
            _finalize_context(ctx, args)
            if ctx["error"]:
                print(f"ERROR: {ctx['side']}: {ctx['error']}", file=sys.stderr)
                exit_code = 1
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        if args.output_json:
            payload = {
                "schema_version": 1,
                "generated_by": "hardware/rm65/utils/init_arm/rm65_move_to_standard_pose.py",
                "created_at": timestamp(),
                "pose_json": args.pose_json,
                "executed": bool(args.execute),
                "command_mode": args.command_mode,
                "config": {
                    "control_hz": args.control_hz,
                    "min_duration_sec": args.min_duration_sec,
                    "max_linear_velocity_m_s": args.max_linear_velocity_m_s,
                    "max_angular_velocity_rad_s": args.max_angular_velocity_rad_s,
                    "max_interp_linear_step_m": args.max_interp_linear_step_m,
                    "max_interp_angular_step_rad": args.max_interp_angular_step_rad,
                    "max_angular_distance_rad": args.max_angular_distance_rad,
                    "ik_safety_profile": args.ik_safety_profile,
                    "ik_safety_profile_json": args.ik_safety_profile_json,
                    "controller_avoid_singularity": args.controller_avoid_singularity,
                    "configure_singularity_thresholds": args.configure_singularity_thresholds,
                    "singularity_elbow_threshold_deg": args.singularity_elbow_threshold_deg,
                    "singularity_wrist_threshold_deg": args.singularity_wrist_threshold_deg,
                    "singularity_shoulder_threshold_m": args.singularity_shoulder_threshold_m,
                    "tool_frame": args.tool_frame,
                    "root_joint_max_delta_deg": args.root_joint_max_delta_deg,
                    "root_joint_absolute_limits_deg": args.root_joint_absolute_limits_deg,
                    "hard_stop_force_n": args.hard_stop_force_n,
                    "hard_stop_force_axis_n": args.hard_stop_force_axis_n,
                    "hard_stop_torque_nm": args.hard_stop_torque_nm,
                    "force_read_every": args.force_read_every,
                    "dual_arm_collision_ik": args.dual_arm_collision_ik,
                    "collision_scene_profile": args.collision_scene_profile,
                    "collision_min_distance_m": args.collision_min_distance_m,
                    "collision_distance_epsilon_m": args.collision_distance_epsilon_m,
                    "dual_arm_ik_iterations": args.dual_arm_ik_iterations,
                },
                "arms": [_report_context(ctx) for ctx in contexts],
            }
            write_json(args.output_json, payload)
        for ctx in contexts:
            robot = ctx.get("robot")
            if robot is not None:
                robot.Arm_Socket_Close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
