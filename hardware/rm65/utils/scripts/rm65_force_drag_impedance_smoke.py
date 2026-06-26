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


CONFIRM_TEXT = "TEST_RM65_FORCE_DRAG"
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


def _parse_free_axes(text: str) -> list[int]:
    normalized = text.lower().replace(",", " ").replace("+", " ")
    tokens = [token for token in normalized.split() if token]
    if len(tokens) == 1 and tokens[0] in ("xyz", "all_pos"):
        tokens = ["x", "y", "z"]
    elif len(tokens) == 1 and tokens[0] in ("rxyz", "all_rot"):
        tokens = ["rx", "ry", "rz"]
    elif len(tokens) == 1 and tokens[0] == "all":
        tokens = list(AXES)
    axes = [0] * 6
    for token in tokens:
        if token not in AXES:
            raise RuntimeError(f"unsupported free axis '{token}', expected one of {AXES}")
        axes[AXES.index(token)] = 1
    if not any(axes):
        raise RuntimeError("--free-axes must enable at least one axis")
    return axes


def _check_force_start(readings: dict[str, list[float]], args) -> None:
    source = readings[args.monitor_force_source]
    force_norm = _norm(source[:3])
    torque_norm = _norm(source[3:])
    print(
        f"precheck monitor_source={args.monitor_force_source} "
        f"force_norm={force_norm:.4f}N torque_norm={torque_norm:.4f}Nm"
    )
    if force_norm > args.max_start_force_n:
        raise RuntimeError(
            f"start force {force_norm:.3f}N > --max-start-force-n {args.max_start_force_n:.3f}N"
        )
    if torque_norm > args.max_start_torque_nm:
        raise RuntimeError(
            f"start torque {torque_norm:.3f}Nm > "
            f"--max-start-torque-nm {args.max_start_torque_nm:.3f}Nm"
        )


def _drag_mode_value(text: str) -> int:
    return {"fast": 0, "precise": 1}[text]


def _frame_value(text: str) -> int:
    return {"work": 0, "tool": 1}[text]


def _run_drag_test(robot: Arm, side: str, start_state: dict[str, object], args) -> None:
    free_axes = _parse_free_axes(args.free_axes)
    frame = _frame_value(args.frame)
    drag_mode = _drag_mode_value(args.force_drag_mode)
    print(
        f"{side}: starting force-drag impedance smoke "
        f"frame={args.frame} free_axes={_format(free_axes, precision=0)} "
        f"force_drag_mode={args.force_drag_mode} singular_wall={int(args.singular_wall)} "
        f"duration={args.duration_sec:.2f}s"
    )
    drag_active = False
    try:
        _require_zero(robot.Set_Force_Drag_Mode(drag_mode), f"{side} Set_Force_Drag_Mode")
        _require_zero(
            robot.Start_Multi_Drag_Teach_New(free_axes, frame=frame, singular_wall=int(args.singular_wall)),
            f"{side} Start_Multi_Drag_Teach_New",
        )
        drag_active = True
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
            force_norm = _norm(readings[args.monitor_force_source][:3])
            max_joint_delta = max(max_joint_delta, joint_delta)
            max_pose_linear = max(max_pose_linear, pose_linear)
            max_pose_angular = max(max_pose_angular, pose_angular)
            max_force_norm = max(max_force_norm, force_norm)
            if index % args.print_every == 0 or index + 1 == loops:
                wrench = readings[args.monitor_force_source]
                print(
                    f"{side}:drag step={index + 1}/{loops} "
                    f"joint_delta={joint_delta:.4f}deg "
                    f"pose_delta={pose_linear:.5f}m/{pose_angular:.5f}rad "
                    f"{args.monitor_force_source}|F|={force_norm:.4f}N "
                    f"wrench={_format(wrench)}"
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
            f"{side}: force-drag complete "
            f"max_joint_delta={max_joint_delta:.4f}deg "
            f"max_pose_delta={max_pose_linear:.5f}m/{max_pose_angular:.5f}rad "
            f"max_{args.monitor_force_source}_force={max_force_norm:.4f}N"
        )
    finally:
        if drag_active:
            stop_code = robot.Stop_Drag_Teach()
            print(f"{side}: Stop_Drag_Teach -> {stop_code}")
            time.sleep(args.stop_settle_sec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "In-place RM65 six-axis force-drag smoke test. This enters official "
            "force drag-teach mode and sends no motion target; the operator can "
            "lightly push the enabled axes to feel compliance."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--expected-tool", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Must be exactly {CONFIRM_TEXT} for drag mode.")
    parser.add_argument("--frame", choices=("work", "tool"), default="tool")
    parser.add_argument(
        "--free-axes",
        default="z",
        help="Enabled drag axes, e.g. z, x,y,z, xyz, rx,rz, or all. Default: z.",
    )
    parser.add_argument("--force-drag-mode", choices=("precise", "fast"), default="precise")
    parser.add_argument("--singular-wall", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--cycle-sec", type=float, default=0.10)
    parser.add_argument("--stop-settle-sec", type=float, default=0.5)
    parser.add_argument("--monitor-force-source", choices=("zero", "work", "tool"), default="tool")
    parser.add_argument("--max-start-force-n", type=float, default=1.0)
    parser.add_argument("--max-start-torque-nm", type=float, default=0.10)
    parser.add_argument("--max-monitor-force-n", type=float, default=8.0)
    parser.add_argument("--max-joint-delta-deg", type=float, default=8.0)
    parser.add_argument("--max-pose-linear-m", type=float, default=0.050)
    parser.add_argument("--max-pose-angular-rad", type=float, default=0.50)
    parser.add_argument("--print-every", type=int, default=5)
    return parser


def _validate_args(args) -> None:
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"force drag test requires --confirm {CONFIRM_TEXT}")
    _parse_free_axes(args.free_axes)
    if args.duration_sec <= 0.0:
        raise RuntimeError("--duration-sec must be positive")
    if args.cycle_sec < 0.05:
        raise RuntimeError("--cycle-sec must be >= 0.05")
    if args.stop_settle_sec < 0.0:
        raise RuntimeError("--stop-settle-sec must be >= 0")
    for name in (
        "max_start_force_n",
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
            free_axes = _parse_free_axes(args.free_axes)
            print(
                f"planned_force_drag frame={args.frame} "
                f"free_axes={dict(zip(AXES, free_axes))} "
                f"force_drag_mode={args.force_drag_mode} singular_wall={args.singular_wall}"
            )
            if not args.execute:
                print("DRY-RUN: force drag mode will not be started.")
                return 0
            _run_drag_test(robot, side, state, args)
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
