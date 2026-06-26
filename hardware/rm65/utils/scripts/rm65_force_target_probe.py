from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65


CONFIRM_TEXT = "TEST_RM65_FORCE_TARGET"
AXES = ("x", "y", "z", "rx", "ry", "rz")


def _format(values: Iterable[float], *, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _finite_float_list(values: Iterable[float], *, expected_len: int) -> list[float]:
    out = [float(value) for value in values]
    if len(out) != expected_len:
        raise RuntimeError(f"expected {expected_len} values, got {len(out)}")
    if not all(math.isfinite(value) for value in out):
        raise RuntimeError(f"non-finite values: {out}")
    return out


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _selected_ip(args) -> tuple[str, str]:
    if args.arm == "left":
        return "left", args.left_ip
    return "right", args.right_ip


def _connect_robot(side: str, ip: str) -> Arm:
    robot = Arm(RM65, ip)
    if int(robot.Arm_Socket_State()) != 0:
        robot.Arm_Socket_Close()
        raise RuntimeError(f"{side}: failed to connect ip={ip}")
    print(f"{side}: connected ip={ip}")
    return robot


def _read_state(robot: Arm, side: str) -> dict[str, object]:
    code, joint_deg, pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=2)
    _require_zero(code, f"{side} Get_Current_Arm_State")
    if int(arm_err) != 0 or int(sys_err) != 0:
        raise RuntimeError(f"{side} reports arm_err={arm_err}, sys_err={sys_err}")
    return {
        "joint_deg": _finite_float_list(joint_deg, expected_len=6),
        "pose": _finite_float_list(pose, expected_len=6),
    }


def _read_force(robot: Arm, side: str) -> dict[str, list[float]]:
    code, raw, zero, work, tool = robot.Get_Force_Data()
    _require_zero(code, f"{side} Get_Force_Data")
    return {
        "raw": _finite_float_list(raw, expected_len=6),
        "zero": _finite_float_list(zero, expected_len=6),
        "work": _finite_float_list(work, expected_len=6),
        "tool": _finite_float_list(tool, expected_len=6),
    }


def _print_force(prefix: str, readings: dict[str, list[float]]) -> None:
    for source in ("raw", "zero", "work", "tool"):
        wrench = readings[source]
        print(
            f"{prefix} {source}={_format(wrench)} "
            f"|F|={_norm(wrench[:3]):.4f}N |T|={_norm(wrench[3:]):.4f}Nm"
        )


def _joint_delta_deg(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _pose_delta(start_pose: list[float], current_pose: list[float]) -> tuple[float, float]:
    linear = _norm(current_pose[:3][i] - start_pose[:3][i] for i in range(3))
    angular = _norm(current_pose[3:][i] - start_pose[3:][i] for i in range(3))
    return linear, angular


def _current_tool_name(frame_result: object) -> str:
    if isinstance(frame_result, tuple) and len(frame_result) >= 2:
        frame = frame_result[1]
    else:
        frame = frame_result
    if isinstance(frame, dict):
        value = frame.get("name") or frame.get("frame_name") or frame.get("tool_name")
        if value is not None:
            return str(value).strip("\x00")
    return str(frame_result).strip("\x00")


def _check_expected_tool(robot: Arm, args) -> None:
    if not args.expected_tool:
        return
    code, frame = robot.Get_Current_Tool_Frame()
    _require_zero(code, "Get_Current_Tool_Frame")
    current_name = _current_tool_name((code, frame))
    print(f"current_tool={current_name}")
    if args.expected_tool not in current_name:
        raise RuntimeError(
            f"expected active tool containing '{args.expected_tool}', got '{current_name}'"
        )


def _axis_index(axis: str) -> int:
    if axis not in AXES:
        raise RuntimeError(f"unsupported axis '{axis}', expected one of {AXES}")
    return AXES.index(axis)


def _control_vectors(args) -> tuple[list[int], list[float], list[float]]:
    axis_index = _axis_index(args.force_axis)
    control_mode = [0, 0, 0, 0, 0, 0]
    desired_force = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    limit_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if args.motion_axis:
        motion_axis_index = _axis_index(args.motion_axis)
        control_mode[motion_axis_index] = 3
        if motion_axis_index < 3:
            limit_vel[motion_axis_index] = float(args.motion_limit_vel_mm_s)
        else:
            limit_vel[motion_axis_index] = float(args.motion_limit_vel_mdeg_s)
    control_mode[axis_index] = int(args.force_control_mode)
    desired_force[axis_index] = float(args.force)
    # RealMan docs describe translational limit_vel as 0.001 m/s units and
    # rotational limit_vel as 0.001 deg/s units.
    if axis_index < 3:
        limit_vel[axis_index] = float(args.limit_vel_mm_s)
    else:
        limit_vel[axis_index] = float(args.limit_vel_mdeg_s)
    return control_mode, desired_force, limit_vel


def _target_pose_with_motion_offset(start_pose: list[float], args) -> list[float]:
    target_pose = list(start_pose)
    if args.motion_axis and abs(float(args.motion_offset)) > 0.0:
        target_pose[_axis_index(args.motion_axis)] += float(args.motion_offset)
    return target_pose


def _check_force_start(readings: dict[str, list[float]], args) -> None:
    source = readings[args.monitor_force_source]
    force_norm = _norm(source[:3])
    torque_norm = _norm(source[3:])
    axis_component = float(source[_axis_index(args.force_axis)])
    component_error = abs(axis_component - float(args.force))
    print(
        f"precheck monitor_source={args.monitor_force_source} "
        f"force_norm={force_norm:.4f}N torque_norm={torque_norm:.4f}Nm "
        f"{args.force_axis}={axis_component:.4f} target={float(args.force):.4f} "
        f"component_error={component_error:.4f}"
    )
    if force_norm > args.hard_max_start_force_n:
        raise RuntimeError(
            f"start force {force_norm:.3f}N > "
            f"--hard-max-start-force-n {args.hard_max_start_force_n:.3f}N"
        )
    if torque_norm > args.max_start_torque_nm:
        raise RuntimeError(
            f"start torque {torque_norm:.3f}Nm > "
            f"--max-start-torque-nm {args.max_start_torque_nm:.3f}Nm"
        )
    if component_error > args.max_start_component_error:
        raise RuntimeError(
            f"start force component error {component_error:.3f}N > "
            f"--max-start-component-error {args.max_start_component_error:.3f}N"
        )


def _frame_mode(text: str) -> int:
    return {"work": 0, "tool": 1}[text]


def _run_force_target(robot: Arm, side: str, start_state: dict[str, object], args) -> None:
    control_mode, desired_force, limit_vel = _control_vectors(args)
    target_pose = _target_pose_with_motion_offset(start_state["pose"], args)
    target_joint = list(start_state["joint_deg"])
    flag = 1 if args.target_mode == "pose" else 0
    print(
        f"{side}: starting force target probe target_mode={args.target_mode} "
        f"frame={args.frame} axis={args.force_axis} force={float(args.force):.4f} "
        f"motion_axis={args.motion_axis or '(none)'} motion_offset={float(args.motion_offset):.5f} "
        f"control_mode={control_mode} desired_force={_format(desired_force)} "
        f"limit_vel={_format(limit_vel)} duration={args.duration_sec:.2f}s"
    )
    force_active = False
    try:
        _require_zero(robot.Start_Force_Position_Move(True), f"{side} Start_Force_Position_Move")
        force_active = True
        loops = max(1, int(math.ceil(args.duration_sec / args.cycle_sec)))
        max_joint_delta = 0.0
        max_pose_linear = 0.0
        max_pose_angular = 0.0
        max_force_norm = 0.0
        axis_index = _axis_index(args.force_axis)
        for index in range(loops):
            if args.target_mode == "pose":
                code = robot.Force_Position_Move_New(
                    flag=flag,
                    pose=target_pose,
                    sensor=1,
                    mode=_frame_mode(args.frame),
                    follow=args.force_follow,
                    control_mode=control_mode,
                    desired_force=desired_force,
                    limit_vel=limit_vel,
                    trajectory_mode=args.trajectory_mode,
                    radio=args.radio,
                )
            else:
                code = robot.Force_Position_Move_New(
                    flag=flag,
                    joint=target_joint,
                    sensor=1,
                    mode=_frame_mode(args.frame),
                    follow=args.force_follow,
                    control_mode=control_mode,
                    desired_force=desired_force,
                    limit_vel=limit_vel,
                    trajectory_mode=args.trajectory_mode,
                    radio=args.radio,
                )
            _require_zero(code, f"{side} Force_Position_Move_New")

            time.sleep(args.cycle_sec)
            state = _read_state(robot, side)
            readings = _read_force(robot, side)
            joint_delta = _joint_delta_deg(start_state["joint_deg"], state["joint_deg"])
            pose_linear, pose_angular = _pose_delta(start_state["pose"], state["pose"])
            source = readings[args.monitor_force_source]
            force_norm = _norm(source[:3])
            axis_component = float(source[axis_index])
            axis_error = axis_component - float(args.force)
            max_joint_delta = max(max_joint_delta, joint_delta)
            max_pose_linear = max(max_pose_linear, pose_linear)
            max_pose_angular = max(max_pose_angular, pose_angular)
            max_force_norm = max(max_force_norm, force_norm)
            if index % args.print_every == 0 or index + 1 == loops:
                print(
                    f"{side}:target step={index + 1}/{loops} "
                    f"joint_delta={joint_delta:.4f}deg "
                    f"pose_delta={pose_linear:.5f}m/{pose_angular:.5f}rad "
                    f"{args.monitor_force_source}|F|={force_norm:.4f}N "
                    f"{args.force_axis}={axis_component:.4f}N err={axis_error:+.4f}N "
                    f"wrench={_format(source)}"
                )
            if joint_delta > args.max_joint_delta_deg:
                raise RuntimeError(
                    f"joint delta {joint_delta:.4f}deg > "
                    f"--max-joint-delta-deg {args.max_joint_delta_deg:.4f}deg"
                )
            if pose_linear > args.max_pose_linear_m:
                raise RuntimeError(
                    f"pose linear delta {pose_linear:.5f}m > "
                    f"--max-pose-linear-m {args.max_pose_linear_m:.5f}m"
                )
            if pose_angular > args.max_pose_angular_rad:
                raise RuntimeError(
                    f"pose angular delta {pose_angular:.5f}rad > "
                    f"--max-pose-angular-rad {args.max_pose_angular_rad:.5f}rad"
                )
            if force_norm > args.max_monitor_force_n:
                raise RuntimeError(
                    f"monitor force {force_norm:.4f}N > "
                    f"--max-monitor-force-n {args.max_monitor_force_n:.4f}N"
                )
        print(
            f"{side}: force target complete "
            f"max_joint_delta={max_joint_delta:.4f}deg "
            f"max_pose_delta={max_pose_linear:.5f}m/{max_pose_angular:.5f}rad "
            f"max_{args.monitor_force_source}_force={max_force_norm:.4f}N"
        )
    finally:
        if force_active:
            stop_code = robot.Stop_Force_Position_Move(True)
            print(f"{side}: Stop_Force_Position_Move -> {stop_code}")
            time.sleep(args.stop_settle_sec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safe RM65 force target probe using the official new-parameter "
            "force-position interface. It sends control_mode/desired_force/limit_vel "
            "instead of entering drag teach mode."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--expected-tool", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Must be exactly {CONFIRM_TEXT} for force mode.")
    parser.add_argument("--target-mode", choices=("pose", "joint"), default="pose")
    parser.add_argument("--frame", choices=("work", "tool"), default="tool")
    parser.add_argument("--force-axis", choices=AXES, default="z")
    parser.add_argument("--force", type=float, default=0.5)
    parser.add_argument(
        "--force-control-mode",
        type=int,
        choices=(4, 7, 8),
        default=4,
        help=(
            "4=force tracking, 7=force tracking + motion, 8=adaptive mode; "
            "RealMan docs say mode 8 only applies to tool Fz."
        ),
    )
    parser.add_argument(
        "--motion-axis",
        choices=AXES,
        default="",
        help="Optional pose axis to command as movement mode. Translational offsets are meters; rotations are radians.",
    )
    parser.add_argument(
        "--motion-offset",
        type=float,
        default=0.0,
        help="Target pose offset along --motion-axis. Requires --target-mode pose.",
    )
    parser.add_argument(
        "--motion-limit-vel-mm-s",
        type=float,
        default=5.0,
        help="Motion-axis translational speed limit, in the SDK's 0.001m/s units.",
    )
    parser.add_argument(
        "--motion-limit-vel-mdeg-s",
        type=float,
        default=500.0,
        help="Motion-axis rotational speed limit, in the SDK's 0.001deg/s units.",
    )
    parser.add_argument(
        "--limit-vel-mm-s",
        type=float,
        default=1.0,
        help="Translational force-axis speed limit, in the SDK's 0.001m/s units.",
    )
    parser.add_argument(
        "--limit-vel-mdeg-s",
        type=float,
        default=100.0,
        help="Rotational force-axis speed limit, in the SDK's 0.001deg/s units.",
    )
    parser.add_argument("--force-follow", action="store_true")
    parser.add_argument("--trajectory-mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--radio", type=int, default=0)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--cycle-sec", type=float, default=0.10)
    parser.add_argument("--stop-settle-sec", type=float, default=0.5)
    parser.add_argument("--monitor-force-source", choices=("zero", "work", "tool"), default="tool")
    parser.add_argument("--max-start-component-error", type=float, default=2.0)
    parser.add_argument("--hard-max-start-force-n", type=float, default=5.0)
    parser.add_argument("--max-start-torque-nm", type=float, default=0.20)
    parser.add_argument("--max-monitor-force-n", type=float, default=5.0)
    parser.add_argument("--max-joint-delta-deg", type=float, default=3.0)
    parser.add_argument("--max-pose-linear-m", type=float, default=0.010)
    parser.add_argument("--max-pose-angular-rad", type=float, default=0.10)
    parser.add_argument("--print-every", type=int, default=1)
    return parser


def _validate_args(args) -> None:
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"force target probe requires --confirm {CONFIRM_TEXT}")
    if args.force_control_mode == 8 and not (args.frame == "tool" and args.force_axis == "z"):
        raise RuntimeError("control mode 8 is only documented for tool-frame Fz")
    if args.motion_axis and args.target_mode != "pose":
        raise RuntimeError("--motion-axis currently requires --target-mode pose")
    if abs(float(args.motion_offset)) > 0.0 and not args.motion_axis:
        raise RuntimeError("--motion-offset requires --motion-axis")
    if abs(args.force) > args.max_monitor_force_n:
        raise RuntimeError("--force must not exceed --max-monitor-force-n")
    if args.duration_sec <= 0.0:
        raise RuntimeError("--duration-sec must be positive")
    if args.cycle_sec < 0.05:
        raise RuntimeError("--cycle-sec must be >= 0.05")
    if args.stop_settle_sec < 0.0:
        raise RuntimeError("--stop-settle-sec must be >= 0")
    if args.radio < 0:
        raise RuntimeError("--radio must be >= 0")
    for name in (
        "limit_vel_mm_s",
        "limit_vel_mdeg_s",
        "motion_limit_vel_mm_s",
        "motion_limit_vel_mdeg_s",
        "max_start_component_error",
        "hard_max_start_force_n",
        "max_start_torque_nm",
        "max_monitor_force_n",
        "max_joint_delta_deg",
        "max_pose_linear_m",
        "max_pose_angular_rad",
    ):
        if getattr(args, name) <= 0.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be positive")
    if args.print_every < 1:
        raise RuntimeError("--print-every must be >= 1")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    side, ip = _selected_ip(args)
    try:
        robot = _connect_robot(side, ip)
        try:
            _check_expected_tool(robot, args)
            state = _read_state(robot, side)
            print(f"{side}: start_joint_deg={_format(state['joint_deg'])}")
            print(f"{side}: start_pose_xyz_rpy={_format(state['pose'])}")
            readings = _read_force(robot, side)
            _print_force(f"{side}: start_force", readings)
            _check_force_start(readings, args)
            control_mode, desired_force, limit_vel = _control_vectors(args)
            print(
                f"planned_force_target target_mode={args.target_mode} frame={args.frame} "
                f"motion_axis={args.motion_axis or '(none)'} motion_offset={float(args.motion_offset):.5f} "
                f"control_mode={control_mode} desired_force={_format(desired_force)} "
                f"limit_vel={_format(limit_vel)}"
            )
            if not args.execute:
                print("DRY-RUN: force-position target mode will not be started.")
                return 0
            _run_force_target(robot, side, state, args)
            final = _read_state(robot, side)
            final_force = _read_force(robot, side)
            print(f"{side}: final_joint_delta={_joint_delta_deg(state['joint_deg'], final['joint_deg']):.4f}deg")
            linear, angular = _pose_delta(state["pose"], final["pose"])
            print(f"{side}: final_pose_delta={linear:.5f}m/{angular:.5f}rad")
            _print_force(f"{side}: final_force", final_force)
        finally:
            robot.Arm_Socket_Close()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
