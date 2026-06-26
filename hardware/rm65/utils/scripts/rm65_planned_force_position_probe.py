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


CONFIRM_TEXT = "TEST_RM65_PLANNED_FORCE"
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


def _axis_index(axis: str) -> int:
    if axis not in AXES:
        raise RuntimeError(f"unsupported axis '{axis}', expected one of {AXES}")
    return AXES.index(axis)


def _selected_ip(args) -> tuple[str, str]:
    if args.arm == "left":
        return "left", args.left_ip
    return "right", args.right_ip


def _frame_mode(text: str) -> int:
    return {"work": 0, "tool": 1}[text]


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


def _motion_target_pose(start_pose: list[float], args) -> list[float]:
    target_pose = list(start_pose)
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


def _run_probe(robot: Arm, side: str, start_state: dict[str, object], args) -> None:
    target_pose = _motion_target_pose(start_state["pose"], args)
    force_active = False
    motion_started = False
    axis_index = _axis_index(args.force_axis)
    print(
        f"{side}: starting planned force-position probe frame={args.frame} "
        f"force_axis={args.force_axis} force={float(args.force):.4f} "
        f"motion_axis={args.motion_axis} motion_offset={float(args.motion_offset):.5f} "
        f"move_speed={args.move_speed} target_pose={_format(target_pose)}"
    )
    try:
        _require_zero(
            robot.Set_Force_Position(1, _frame_mode(args.frame), axis_index, float(args.force)),
            f"{side} Set_Force_Position",
        )
        force_active = True
        _require_zero(
            robot.Movel_Cmd(
                target_pose,
                args.move_speed,
                r=0,
                connect=0,
                block=False,
            ),
            f"{side} Movel_Cmd",
        )
        motion_started = True
        loops = max(1, int(math.ceil(args.duration_sec / args.cycle_sec)))
        max_joint_delta = 0.0
        max_pose_linear = 0.0
        max_pose_angular = 0.0
        max_force_norm = 0.0
        for index in range(loops):
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
                    f"{side}:planned step={index + 1}/{loops} "
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
            f"{side}: planned force-position complete "
            f"max_joint_delta={max_joint_delta:.4f}deg "
            f"max_pose_delta={max_pose_linear:.5f}m/{max_pose_angular:.5f}rad "
            f"max_{args.monitor_force_source}_force={max_force_norm:.4f}N"
        )
    finally:
        if motion_started:
            slow_stop_code = robot.Arm_Slow_Stop()
            print(f"{side}: Arm_Slow_Stop -> {slow_stop_code}")
            time.sleep(args.stop_settle_sec)
        if force_active:
            stop_force_code = robot.Stop_Force_Position()
            print(f"{side}: Stop_Force_Position -> {stop_force_code}")
            time.sleep(args.stop_settle_sec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safe planned RM65 force-position probe. It uses the SDK's planned "
            "force-position API: Set_Force_Position followed by a small absolute Movel."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--expected-tool", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Must be exactly {CONFIRM_TEXT} for force mode.")
    parser.add_argument("--frame", choices=("work", "tool"), default="tool")
    parser.add_argument("--force-axis", choices=AXES, default="z")
    parser.add_argument("--force", type=float, default=1.0)
    parser.add_argument("--motion-axis", choices=AXES, default="x")
    parser.add_argument("--motion-offset", type=float, default=0.005)
    parser.add_argument("--move-speed", type=int, default=1)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--cycle-sec", type=float, default=0.10)
    parser.add_argument("--stop-settle-sec", type=float, default=0.5)
    parser.add_argument("--monitor-force-source", choices=("zero", "work", "tool"), default="tool")
    parser.add_argument("--max-start-component-error", type=float, default=2.0)
    parser.add_argument("--hard-max-start-force-n", type=float, default=5.0)
    parser.add_argument("--max-start-torque-nm", type=float, default=0.20)
    parser.add_argument("--max-monitor-force-n", type=float, default=5.0)
    parser.add_argument("--max-joint-delta-deg", type=float, default=5.0)
    parser.add_argument("--max-pose-linear-m", type=float, default=0.020)
    parser.add_argument("--max-pose-angular-rad", type=float, default=0.10)
    parser.add_argument("--print-every", type=int, default=1)
    return parser


def _validate_args(args) -> None:
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"planned force-position probe requires --confirm {CONFIRM_TEXT}")
    if _axis_index(args.force_axis) == _axis_index(args.motion_axis):
        raise RuntimeError("force axis and motion axis must be different for planned force-position")
    if abs(args.force) > args.max_monitor_force_n:
        raise RuntimeError("--force must not exceed --max-monitor-force-n")
    if args.move_speed < 1 or args.move_speed > 10:
        raise RuntimeError("--move-speed must be in [1, 10] for this safety probe")
    if args.duration_sec <= 0.0:
        raise RuntimeError("--duration-sec must be positive")
    if args.cycle_sec < 0.05:
        raise RuntimeError("--cycle-sec must be >= 0.05")
    if args.stop_settle_sec < 0.0:
        raise RuntimeError("--stop-settle-sec must be >= 0")
    for name in (
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
            print(
                f"planned_force_position frame={args.frame} force_axis={args.force_axis} "
                f"force={float(args.force):.4f} motion_axis={args.motion_axis} "
                f"motion_offset={float(args.motion_offset):.5f} move_speed={args.move_speed}"
            )
            if not args.execute:
                print("DRY-RUN: planned force-position mode will not be started.")
                return 0
            _run_probe(robot, side, state, args)
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
