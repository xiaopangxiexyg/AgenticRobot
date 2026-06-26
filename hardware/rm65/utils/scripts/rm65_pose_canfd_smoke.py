from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65


CONFIRM_TEXT = "TEST_RM65_POSE_CANFD"
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


def _parse_offsets(text: str) -> list[float]:
    try:
        offsets = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise RuntimeError(f"invalid --offsets '{text}'") from exc
    if not offsets:
        raise RuntimeError("--offsets must contain at least one value")
    if not all(math.isfinite(value) for value in offsets):
        raise RuntimeError("--offsets contains a non-finite value")
    return offsets


def _connect_robot(side: str, ip: str) -> Arm:
    robot = Arm(RM65, ip)
    if int(robot.Arm_Socket_State()) != 0:
        robot.Arm_Socket_Close()
        raise RuntimeError(f"{side}: failed to connect ip={ip}")
    print(f"{side}: connected ip={ip} api={robot.API_Version()}")
    return robot


def _read_state(robot: Arm, side: str) -> dict[str, list[float]]:
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


def _joint_delta_deg(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _pose_delta(start_pose: list[float], current_pose: list[float]) -> tuple[float, float]:
    linear = _norm(current_pose[i] - start_pose[i] for i in range(3))
    angular = _norm(current_pose[i] - start_pose[i] for i in range(3, 6))
    return linear, angular


def _target_pose(start_pose: list[float], axis: str, offset: float) -> list[float]:
    pose = list(start_pose)
    pose[_axis_index(axis)] += float(offset)
    return pose


def _lerp_pose(a: list[float], b: list[float], alpha: float) -> list[float]:
    return [float(x) + (float(y) - float(x)) * alpha for x, y in zip(a, b)]


def _check_start_force(readings: dict[str, list[float]], args) -> None:
    wrench = readings[args.monitor_force_source]
    force_norm = _norm(wrench[:3])
    torque_norm = _norm(wrench[3:])
    print(
        f"precheck monitor_source={args.monitor_force_source} "
        f"force_norm={force_norm:.4f}N torque_norm={torque_norm:.4f}Nm"
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


def _check_motion_limits(start_state: dict[str, list[float]], state: dict[str, list[float]], args) -> tuple[float, float, float]:
    joint_delta = _joint_delta_deg(start_state["joint_deg"], state["joint_deg"])
    pose_linear, pose_angular = _pose_delta(start_state["pose"], state["pose"])
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
    return joint_delta, pose_linear, pose_angular


def _build_targets(start_pose: list[float], args) -> list[list[float]]:
    offsets = list(args.offset_values)
    if args.return_to_start and abs(offsets[-1]) > 1e-12:
        offsets.append(0.0)
    return [_target_pose(start_pose, args.axis, offset) for offset in offsets]


def _run_pose_canfd(robot: Arm, side: str, start_state: dict[str, list[float]], args) -> dict[str, object]:
    targets = _build_targets(start_state["pose"], args)
    print(
        f"{side}: starting pose CANFD smoke axis={args.axis} offsets={_format(args.offset_values, precision=5)} "
        f"segment_sec={args.segment_sec:.3f} cycle_sec={args.cycle_sec:.3f} "
        f"follow={args.follow} trajectory_mode={args.trajectory_mode} radio={args.radio}"
    )
    print(f"{side}: start_pose={_format(start_state['pose'])}")
    for index, target in enumerate(targets, start=1):
        print(f"{side}: target[{index}]={_format(target)}")

    records: list[dict[str, object]] = []
    max_joint_delta = 0.0
    max_pose_linear = 0.0
    max_pose_angular = 0.0
    max_force_norm = 0.0
    motion_started = False
    command_pose = list(start_state["pose"])
    axis_index = _axis_index(args.axis)
    steps_per_segment = max(1, int(math.ceil(args.segment_sec / args.cycle_sec)))
    step_counter = 0

    try:
        for segment_index, target_pose in enumerate(targets, start=1):
            segment_start = list(command_pose)
            for step_index in range(1, steps_per_segment + 1):
                loop_start = time.time()
                alpha = step_index / steps_per_segment
                command_pose = _lerp_pose(segment_start, target_pose, alpha)
                _require_zero(
                    robot.Movep_CANFD(
                        command_pose,
                        follow=args.follow,
                        trajectory_mode=args.trajectory_mode,
                        radio=args.radio,
                    ),
                    f"{side} Movep_CANFD",
                )
                motion_started = True
                step_counter += 1

                state = _read_state(robot, side)
                readings = _read_force(robot, side)
                source = readings[args.monitor_force_source]
                force_norm = _norm(source[:3])
                torque_norm = _norm(source[3:])
                joint_delta, pose_linear, pose_angular = _check_motion_limits(
                    start_state,
                    state,
                    args,
                )
                if force_norm > args.max_monitor_force_n:
                    raise RuntimeError(
                        f"monitor force {force_norm:.4f}N > "
                        f"--max-monitor-force-n {args.max_monitor_force_n:.4f}N"
                    )
                if torque_norm > args.max_monitor_torque_nm:
                    raise RuntimeError(
                        f"monitor torque {torque_norm:.4f}Nm > "
                        f"--max-monitor-torque-nm {args.max_monitor_torque_nm:.4f}Nm"
                    )

                max_joint_delta = max(max_joint_delta, joint_delta)
                max_pose_linear = max(max_pose_linear, pose_linear)
                max_pose_angular = max(max_pose_angular, pose_angular)
                max_force_norm = max(max_force_norm, force_norm)
                observed_axis_offset = state["pose"][axis_index] - start_state["pose"][axis_index]
                command_axis_offset = command_pose[axis_index] - start_state["pose"][axis_index]

                if step_counter % args.print_every == 0 or step_index == steps_per_segment:
                    print(
                        f"{side}:pose_canfd seg={segment_index}/{len(targets)} "
                        f"step={step_index}/{steps_per_segment} "
                        f"cmd_{args.axis}={command_axis_offset:+.5f} "
                        f"obs_{args.axis}={observed_axis_offset:+.5f} "
                        f"joint_delta={joint_delta:.4f}deg "
                        f"pose_delta={pose_linear:.5f}m/{pose_angular:.5f}rad "
                        f"{args.monitor_force_source}|F|={force_norm:.4f}N "
                        f"|T|={torque_norm:.4f}Nm"
                    )

                records.append(
                    {
                        "segment": segment_index,
                        "segment_step": step_index,
                        "command_pose": command_pose,
                        "observed_pose": state["pose"],
                        "observed_joint_deg": state["joint_deg"],
                        "command_axis_offset": command_axis_offset,
                        "observed_axis_offset": observed_axis_offset,
                        "joint_delta_deg": joint_delta,
                        "pose_linear_delta_m": pose_linear,
                        "pose_angular_delta_rad": pose_angular,
                        "force_source": args.monitor_force_source,
                        "force": source,
                        "force_norm_n": force_norm,
                        "torque_norm_nm": torque_norm,
                    }
                )

                elapsed = time.time() - loop_start
                time.sleep(max(args.cycle_sec - elapsed, 0.0))
    finally:
        if motion_started:
            slow_stop_code = robot.Arm_Slow_Stop()
            print(f"{side}: Arm_Slow_Stop -> {slow_stop_code}")
            time.sleep(args.stop_settle_sec)

    print(
        f"{side}: pose CANFD complete max_joint_delta={max_joint_delta:.4f}deg "
        f"max_pose_delta={max_pose_linear:.5f}m/{max_pose_angular:.5f}rad "
        f"max_{args.monitor_force_source}_force={max_force_norm:.4f}N"
    )
    return {
        "records": records,
        "max_joint_delta_deg": max_joint_delta,
        "max_pose_linear_delta_m": max_pose_linear,
        "max_pose_angular_delta_rad": max_pose_angular,
        "max_force_norm_n": max_force_norm,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safe RM65 pose CANFD smoke test. It sends small absolute pose targets "
            "through Movep_CANFD without force mode."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--expected-tool", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Must be exactly {CONFIRM_TEXT} for motion.")
    parser.add_argument("--axis", choices=AXES, default="x")
    parser.add_argument(
        "--offsets",
        default="0.01,0.02,0.05,0.0",
        help="Comma-separated target offsets from the start pose. Units are m for x/y/z, rad for rx/ry/rz.",
    )
    parser.add_argument("--return-to-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment-sec", type=float, default=1.0)
    parser.add_argument("--cycle-sec", type=float, default=0.05)
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Disabled in this safety smoke; reserved for a later high-rate controller test.",
    )
    parser.add_argument("--trajectory-mode", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--radio", type=int, default=50)
    parser.add_argument("--monitor-force-source", choices=("zero", "work", "tool"), default="tool")
    parser.add_argument("--hard-max-start-force-n", type=float, default=5.0)
    parser.add_argument("--max-start-torque-nm", type=float, default=0.20)
    parser.add_argument("--max-monitor-force-n", type=float, default=5.0)
    parser.add_argument("--max-monitor-torque-nm", type=float, default=0.30)
    parser.add_argument("--max-joint-delta-deg", type=float, default=15.0)
    parser.add_argument("--max-pose-linear-m", type=float, default=0.15)
    parser.add_argument("--max-pose-angular-rad", type=float, default=0.10)
    parser.add_argument("--max-command-offset-m", type=float, default=0.15)
    parser.add_argument("--max-command-offset-rad", type=float, default=0.05)
    parser.add_argument("--stop-settle-sec", type=float, default=0.5)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--output-json", default="")
    return parser


def _validate_args(args) -> None:
    args.offset_values = _parse_offsets(args.offsets)
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"pose CANFD smoke requires --confirm {CONFIRM_TEXT}")
    if args.segment_sec <= 0.0:
        raise RuntimeError("--segment-sec must be positive")
    if args.cycle_sec < 0.02:
        raise RuntimeError("--cycle-sec must be >= 0.02")
    if args.follow:
        raise RuntimeError("--follow is disabled in this safety smoke; validate low-follow pose CANFD first")
    if args.print_every < 1:
        raise RuntimeError("--print-every must be >= 1")
    if args.stop_settle_sec < 0.0:
        raise RuntimeError("--stop-settle-sec must be >= 0")
    for name in (
        "hard_max_start_force_n",
        "max_start_torque_nm",
        "max_monitor_force_n",
        "max_monitor_torque_nm",
        "max_joint_delta_deg",
        "max_pose_linear_m",
        "max_pose_angular_rad",
        "max_command_offset_m",
        "max_command_offset_rad",
    ):
        if getattr(args, name) <= 0.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be positive")
    max_abs_offset = max(abs(value) for value in args.offset_values)
    if _axis_index(args.axis) < 3:
        if max_abs_offset > args.max_command_offset_m:
            raise RuntimeError(
                f"max translational command offset {max_abs_offset:.5f}m > "
                f"--max-command-offset-m {args.max_command_offset_m:.5f}m"
            )
    else:
        if max_abs_offset > args.max_command_offset_rad:
            raise RuntimeError(
                f"max rotational command offset {max_abs_offset:.5f}rad > "
                f"--max-command-offset-rad {args.max_command_offset_rad:.5f}rad"
            )


def _write_report(path_text: str, report: dict[str, object]) -> None:
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote report: {path}")


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
            _check_start_force(readings, args)
            targets = _build_targets(state["pose"], args)
            print(
                f"pose_canfd axis={args.axis} offsets={_format(args.offset_values, precision=5)} "
                f"targets={len(targets)} segment_sec={args.segment_sec:.3f}"
            )
            report: dict[str, object] = {
                "arm": side,
                "ip": ip,
                "axis": args.axis,
                "offsets": args.offset_values,
                "return_to_start": args.return_to_start,
                "segment_sec": args.segment_sec,
                "cycle_sec": args.cycle_sec,
                "follow": args.follow,
                "trajectory_mode": args.trajectory_mode,
                "radio": args.radio,
                "start_state": state,
                "start_force": readings,
                "targets": targets,
                "executed": bool(args.execute),
            }
            if not args.execute:
                print("DRY-RUN: pose CANFD commands will not be sent.")
                _write_report(args.output_json, report)
                return 0
            result = _run_pose_canfd(robot, side, state, args)
            final = _read_state(robot, side)
            final_force = _read_force(robot, side)
            linear, angular = _pose_delta(state["pose"], final["pose"])
            print(f"{side}: final_joint_delta={_joint_delta_deg(state['joint_deg'], final['joint_deg']):.4f}deg")
            print(f"{side}: final_pose_delta={linear:.5f}m/{angular:.5f}rad")
            _print_force(f"{side}: final_force", final_force)
            report.update(result)
            report["final_state"] = final
            report["final_force"] = final_force
            _write_report(args.output_json, report)
        finally:
            robot.Arm_Socket_Close()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
