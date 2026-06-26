from __future__ import annotations

import argparse
import json
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
from hardware.rm65.standard_paths import resolve_fit_path


CONFIRM_TEXT = "TEST_RM65_FORCE_CONTROL"


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


def _default_fit_path(arm: str, tool_label: str) -> Path:
    return resolve_fit_path(arm, tool_label or "unknown", require_existing=False)


def _load_fit(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    fit = payload.get("fit", {})
    print(f"fit_json={path}")
    print(f"fit_tool_label={payload.get('tool_label')} source={fit.get('source')} mode={fit.get('mode')}")
    if fit.get("mode") == "tool_gravity_model":
        print(f"fit_payload_kg={float(fit['mass_kg_abs']):.6f}")
        print(f"fit_com_tool_m={_format(fit['com_tool_m'])}")
        print(f"fit_force_bias_tool_n={_format(fit.get('force_bias_tool_n', [0, 0, 0]))}")
        print(
            "fit_corrected_force_norm mean/rms/max="
            f"{float(fit.get('corrected_force_norm_mean_n', math.nan)):.4f}/"
            f"{float(fit.get('corrected_force_norm_rms_n', math.nan)):.4f}/"
            f"{float(fit.get('corrected_force_norm_max_n', math.nan)):.4f}N"
        )
    return payload


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


def _controlled_force_source(args) -> str:
    if args.force_command_source != "auto":
        return args.force_command_source
    return "work" if args.force_mode == 0 else "tool"


def _resolve_desired_force(readings: dict[str, list[float]], args, label: str) -> float:
    if not args.hold_start_force_component:
        return float(args.force)
    source = _controlled_force_source(args)
    force = float(readings[source][args.force_dir])
    print(
        f"hold_start_force_component=True sample={label} source={source} dir={args.force_dir} "
        f"desired_force={force:.4f}"
    )
    return force


def _check_force_start(
    readings: dict[str, list[float]],
    args,
    desired_force: float,
) -> None:
    source_name = _controlled_force_source(args)
    source = readings[source_name]
    force_norm = _norm(source[:3])
    torque_norm = _norm(source[3:])
    current_component = float(source[args.force_dir])
    component_error = abs(current_component - desired_force)
    print(
        f"precheck control_source={source_name} "
        f"force_norm={force_norm:.4f}N torque_norm={torque_norm:.4f}Nm "
        f"component={current_component:.4f} desired={desired_force:.4f} "
        f"component_error={component_error:.4f}"
    )
    if component_error > args.max_start_component_error:
        raise RuntimeError(
            f"start controlled force error {component_error:.3f} > "
            f"--max-start-component-error {args.max_start_component_error:.3f}; "
            "use --hold-start-force-component for a no-motion smoke test when "
            "the controller reports a large calibrated payload bias"
        )
    if force_norm > args.max_start_force_n and not args.hold_start_force_component:
        raise RuntimeError(
            f"start force {force_norm:.3f}N > --max-start-force-n {args.max_start_force_n:.3f}N"
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


def _run_mode(
    robot: Arm,
    side: str,
    mode_name: str,
    start_joint: list[float],
    start_pose: list[float],
    desired_force: float,
    args,
) -> None:
    print(
        f"\n{side}: starting official force-position smoke mode={mode_name} "
        f"sensor={args.force_sensor} force_mode={args.force_mode} "
        f"dir={args.force_dir} force={desired_force:.4f} "
        f"duration={args.duration_sec:.2f}s cycle={args.cycle_sec:.3f}s"
    )
    force_active = False
    try:
        _require_zero(robot.Start_Force_Position_Move(True), f"{side} Start_Force_Position_Move")
        force_active = True
        if args.hold_start_force_component and args.hold_force_sample == "post-start":
            if args.post_start_sample_sec:
                time.sleep(args.post_start_sample_sec)
            post_start_readings = _read_force(robot, side)
            desired_force = _resolve_desired_force(post_start_readings, args, "post-start")
            _check_force_start(post_start_readings, args, desired_force)
            print(f"{side}: mode={mode_name} post-start desired_force={desired_force:.4f}")

        loops = max(1, int(math.ceil(args.duration_sec / args.cycle_sec)))
        max_joint_delta = 0.0
        max_pose_linear = 0.0
        max_pose_angular = 0.0
        max_force_norm = 0.0
        for index in range(loops):
            if mode_name == "joint":
                _require_zero(
                    robot.Force_Position_Move_Joint(
                        start_joint,
                        sensor=args.force_sensor,
                        mode=args.force_mode,
                        dir=args.force_dir,
                        force=desired_force,
                        follow=args.force_follow,
                    ),
                    f"{side} Force_Position_Move_Joint",
                )
            elif mode_name == "pose":
                _require_zero(
                    robot.Force_Position_Move_Pose(
                        start_pose,
                        sensor=args.force_sensor,
                        mode=args.force_mode,
                        dir=args.force_dir,
                        force=desired_force,
                        follow=args.force_follow,
                    ),
                    f"{side} Force_Position_Move_Pose",
                )
            else:
                raise RuntimeError(f"unsupported mode: {mode_name}")

            time.sleep(args.cycle_sec)
            state = _read_state(robot, side)
            readings = _read_force(robot, side)
            joint_delta = _joint_delta_deg(start_joint, state["joint_deg"])
            pose_linear, pose_angular = _pose_delta(start_pose, state["pose"])
            force_norm = _norm(readings[args.monitor_force_source][:3])
            controlled_source = _controlled_force_source(args)
            controlled_component = float(readings[controlled_source][args.force_dir])
            controlled_error = controlled_component - desired_force
            max_joint_delta = max(max_joint_delta, joint_delta)
            max_pose_linear = max(max_pose_linear, pose_linear)
            max_pose_angular = max(max_pose_angular, pose_angular)
            max_force_norm = max(max_force_norm, force_norm)
            if index % args.print_every == 0 or index + 1 == loops:
                print(
                    f"{side}:{mode_name} step={index + 1}/{loops} "
                    f"joint_delta={joint_delta:.4f}deg "
                    f"pose_delta={pose_linear:.5f}m/{pose_angular:.5f}rad "
                    f"{args.monitor_force_source}|F|={force_norm:.4f}N "
                    f"{controlled_source}[{args.force_dir}]={controlled_component:.4f}N "
                    f"err={controlled_error:+.4f}N"
                )
            if joint_delta > args.max_joint_drift_deg:
                raise RuntimeError(
                    f"{mode_name} joint drift {joint_delta:.4f}deg > "
                    f"{args.max_joint_drift_deg:.4f}deg"
                )
            if pose_linear > args.max_pose_linear_drift_m:
                raise RuntimeError(
                    f"{mode_name} pose linear drift {pose_linear:.5f}m > "
                    f"{args.max_pose_linear_drift_m:.5f}m"
                )
            if pose_angular > args.max_pose_angular_drift_rad:
                raise RuntimeError(
                    f"{mode_name} pose angular drift {pose_angular:.5f}rad > "
                    f"{args.max_pose_angular_drift_rad:.5f}rad"
                )
            if force_norm > args.max_monitor_force_n:
                raise RuntimeError(
                    f"{mode_name} monitor force {force_norm:.4f}N > "
                    f"{args.max_monitor_force_n:.4f}N"
                )

        print(
            f"{side}: mode={mode_name} complete "
            f"max_joint_delta={max_joint_delta:.4f}deg "
            f"max_pose_delta={max_pose_linear:.5f}m/{max_pose_angular:.5f}rad "
            f"max_{args.monitor_force_source}_force={max_force_norm:.4f}N"
        )
    finally:
        if force_active:
            stop_code = robot.Stop_Force_Position_Move(True)
            print(f"{side}: Stop_Force_Position_Move({mode_name}) -> {stop_code}")
            time.sleep(args.stop_settle_sec)


def _mode_sequence(args) -> list[str]:
    if args.test_mode == "both":
        return ["joint", "pose"]
    return [args.test_mode]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safe calibrated RM65 official force-position smoke test. The target "
            "position is frozen at script startup; real force mode requires explicit confirmation."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--tool-label", default="")
    parser.add_argument("--fit-json", default="")
    parser.add_argument("--expected-tool", default="")
    parser.add_argument("--test-mode", choices=("joint", "pose", "both"), default="joint")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Must be exactly {CONFIRM_TEXT} for force mode.")
    parser.add_argument("--force-sensor", type=int, choices=(0, 1), default=1)
    parser.add_argument("--force-mode", type=int, choices=(0, 1), default=1)
    parser.add_argument("--force-dir", type=int, choices=range(6), default=2)
    parser.add_argument("--force", type=float, default=0.0)
    parser.add_argument("--allow-nonzero-force", action="store_true")
    parser.add_argument(
        "--hold-start-force-component",
        action="store_true",
        help="Use the startup force component in the controlled direction as the desired force.",
    )
    parser.add_argument(
        "--hold-force-sample",
        choices=("pre-start", "post-start"),
        default="post-start",
        help="When holding startup force, choose whether the desired force is sampled before or after Start_Force_Position_Move.",
    )
    parser.add_argument("--post-start-sample-sec", type=float, default=0.10)
    parser.add_argument(
        "--force-command-source",
        choices=("auto", "zero", "work", "tool"),
        default="auto",
        help="Force source used to derive/check desired force. auto: work for base mode, tool for tool mode.",
    )
    parser.add_argument("--force-follow", action="store_true")
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--cycle-sec", type=float, default=0.10)
    parser.add_argument("--stop-settle-sec", type=float, default=0.5)
    parser.add_argument(
        "--clear-force-data",
        action="store_true",
        help="Clear force data at the frozen start pose before precheck/start.",
    )
    parser.add_argument("--clear-settle-sec", type=float, default=0.30)
    parser.add_argument("--monitor-force-source", choices=("zero", "work", "tool"), default="tool")
    parser.add_argument("--max-start-component-error", type=float, default=0.5)
    parser.add_argument("--max-start-force-n", type=float, default=8.0)
    parser.add_argument("--hard-max-start-force-n", type=float, default=25.0)
    parser.add_argument("--max-start-torque-nm", type=float, default=1.2)
    parser.add_argument("--max-monitor-force-n", type=float, default=15.0)
    parser.add_argument("--max-joint-drift-deg", type=float, default=0.20)
    parser.add_argument("--max-pose-linear-drift-m", type=float, default=0.005)
    parser.add_argument("--max-pose-angular-drift-rad", type=float, default=0.05)
    parser.add_argument("--print-every", type=int, default=5)
    return parser


def _validate_args(args) -> None:
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"real force control requires --confirm {CONFIRM_TEXT}")
    if (
        abs(args.force) > 1e-9
        and not args.allow_nonzero_force
        and not args.hold_start_force_component
    ):
        raise RuntimeError("nonzero force requires --allow-nonzero-force")
    if args.duration_sec <= 0.0:
        raise RuntimeError("--duration-sec must be positive")
    if args.cycle_sec < 0.05:
        raise RuntimeError("--cycle-sec must be >= 0.05 because force reads should not be faster")
    if args.post_start_sample_sec < 0.0:
        raise RuntimeError("--post-start-sample-sec must be >= 0")
    if args.stop_settle_sec < 0.0:
        raise RuntimeError("--stop-settle-sec must be >= 0")
    if args.clear_settle_sec < 0.0:
        raise RuntimeError("--clear-settle-sec must be >= 0")
    if args.max_start_component_error <= 0.0:
        raise RuntimeError("--max-start-component-error must be positive")
    if (
        args.max_start_force_n <= 0.0
        or args.hard_max_start_force_n <= 0.0
        or args.max_monitor_force_n <= 0.0
    ):
        raise RuntimeError("force limits must be positive")
    if args.hard_max_start_force_n < args.max_start_force_n:
        raise RuntimeError("--hard-max-start-force-n must be >= --max-start-force-n")
    if args.max_start_torque_nm <= 0.0:
        raise RuntimeError("--max-start-torque-nm must be positive")
    if args.max_joint_drift_deg <= 0.0:
        raise RuntimeError("--max-joint-drift-deg must be positive")
    if args.max_pose_linear_drift_m <= 0.0:
        raise RuntimeError("--max-pose-linear-drift-m must be positive")
    if args.max_pose_angular_drift_rad <= 0.0:
        raise RuntimeError("--max-pose-angular-drift-rad must be positive")
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
    tool_label = args.tool_label or args.expected_tool
    fit_path = Path(args.fit_json) if args.fit_json else _default_fit_path(side, tool_label)
    try:
        if fit_path.exists():
            _load_fit(fit_path)
        else:
            print(f"WARNING: fit_json not found: {fit_path}")
        robot = _connect_robot(side, ip)
        try:
            _check_expected_tool(robot, args)
            state = _read_state(robot, side)
            start_joint = state["joint_deg"]
            start_pose = state["pose"]
            print(f"{side}: frozen_start_joint_deg={_format(start_joint)}")
            print(f"{side}: frozen_start_pose_xyz_rpy={_format(start_pose)}")
            if args.clear_force_data:
                _require_zero(robot.Clear_Force_Data(True), f"{side} Clear_Force_Data")
                if args.clear_settle_sec:
                    time.sleep(args.clear_settle_sec)
                print(f"{side}: Clear_Force_Data applied at frozen start pose")
            readings = _read_force(robot, side)
            _print_force(f"{side}: start_force", readings)
            desired_force = _resolve_desired_force(readings, args, "pre-start")
            _check_force_start(readings, args, desired_force)

            if not args.execute:
                print("DRY-RUN: force-position mode will not be started.")
                print(f"planned_modes={_mode_sequence(args)}")
                print(f"planned_desired_force={desired_force:.4f}")
                if args.hold_start_force_component and args.hold_force_sample == "post-start":
                    print("NOTE: real execution will resample desired_force after Start_Force_Position_Move.")
                return 0

            for mode_name in _mode_sequence(args):
                _run_mode(robot, side, mode_name, start_joint, start_pose, desired_force, args)
            final = _read_state(robot, side)
            final_force = _read_force(robot, side)
            print(f"{side}: final_joint_delta={_joint_delta_deg(start_joint, final['joint_deg']):.4f}deg")
            linear, angular = _pose_delta(start_pose, final["pose"])
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
