from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65


CONFIRM_TEXT = "MOVE_RM65"


@dataclass(frozen=True)
class ArmTarget:
    side: str
    ip: str


def _finite_float_list(values: Iterable[float], *, expected_len: int) -> list[float]:
    out = [float(value) for value in values]
    if len(out) != expected_len:
        raise RuntimeError(f"expected {expected_len} values, got {len(out)}")
    if not all(math.isfinite(value) for value in out):
        raise RuntimeError(f"non-finite values: {out}")
    return out


def _format(values: Iterable[float], *, precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _nudge_target(joint_deg: list[float], *, joint_index: int, delta_deg: float) -> list[float]:
    target = list(joint_deg)
    target[joint_index - 1] += delta_deg
    return target


def _connect_and_read(arm: ArmTarget) -> tuple[Arm, list[float], list[float]]:
    robot = Arm(RM65, arm.ip)
    code, joint_deg, pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=2)
    _require_zero(code, f"{arm.side} Get_Current_Arm_State")
    joint_deg = _finite_float_list(joint_deg, expected_len=6)
    pose = _finite_float_list(pose, expected_len=6)
    if int(arm_err) != 0 or int(sys_err) != 0:
        raise RuntimeError(f"{arm.side} reports arm_err={arm_err}, sys_err={sys_err}")
    print(f"{arm.side}: ip={arm.ip}")
    print(f"{arm.side}: joint_deg={_format(joint_deg)}")
    print(f"{arm.side}: pose_xyz_rpy={_format(pose, precision=4)}")
    return robot, joint_deg, pose


def _read_force(robot: Arm, side: str) -> dict[str, list[float]]:
    code, force, zero_force, work_zero, tool_zero = robot.Get_Force_Data()
    _require_zero(code, f"{side} Get_Force_Data")
    readings = {
        "raw": _finite_float_list(force, expected_len=6),
        "zero": _finite_float_list(zero_force, expected_len=6),
        "work": _finite_float_list(work_zero, expected_len=6),
        "tool": _finite_float_list(tool_zero, expected_len=6),
    }
    print(f"{side}: force_raw={_format(force, precision=4)}")
    print(f"{side}: force_zero={_format(zero_force, precision=4)}")
    print(f"{side}: force_work_zero={_format(work_zero, precision=4)}")
    print(f"{side}: force_tool_zero={_format(tool_zero, precision=4)}")
    return readings


def _wrench_norms(wrench: list[float]) -> tuple[float, float]:
    force_norm = math.sqrt(sum(value * value for value in wrench[:3]))
    torque_norm = math.sqrt(sum(value * value for value in wrench[3:]))
    return force_norm, torque_norm


def _check_start_force_safe(robot: Arm, side: str, args) -> None:
    if not args.check_start_force:
        print(f"{side}: start-force safety check skipped")
        return
    readings = _read_force(robot, side)
    sources = ("zero", "work", "tool") if args.force_precheck_source == "all" else (args.force_precheck_source,)
    failures = []
    for source in sources:
        force_norm, torque_norm = _wrench_norms(readings[source])
        print(
            f"{side}: precheck {source}_force_norm={force_norm:.4f}N "
            f"{source}_torque_norm={torque_norm:.4f}Nm"
        )
        if force_norm > args.max_start_force_n:
            failures.append(f"{source} force {force_norm:.3f}N > {args.max_start_force_n:.3f}N")
        if torque_norm > args.max_start_torque_nm:
            failures.append(
                f"{source} torque {torque_norm:.3f}Nm > {args.max_start_torque_nm:.3f}Nm"
            )
    if failures:
        raise RuntimeError(
            "force-position precheck failed; remove contact/load before starting force mode: "
            + "; ".join(failures)
        )


def _position_nudge(robot: Arm, arm: ArmTarget, joint_deg: list[float], args) -> None:
    target = _nudge_target(joint_deg, joint_index=args.joint_index, delta_deg=args.position_step_deg)
    print(
        f"{arm.side}: position test joint{args.joint_index} "
        f"{args.position_step_deg:+.3f} deg at speed={args.movej_speed}"
    )
    _require_zero(
        robot.Movej_Cmd(target, args.movej_speed, block=True),
        f"{arm.side} position nudge",
    )
    time.sleep(args.settle_sec)
    _require_zero(
        robot.Movej_Cmd(joint_deg, args.movej_speed, block=True),
        f"{arm.side} position return",
    )
    time.sleep(args.settle_sec)
    print(f"{arm.side}: position test complete and returned to start")


def _force_position_hold(robot: Arm, arm: ArmTarget, joint_deg: list[float], args) -> None:
    if abs(args.force) > 1e-9 and not args.allow_nonzero_force:
        raise RuntimeError("nonzero force requires --allow-nonzero-force")
    target = _nudge_target(
        joint_deg,
        joint_index=args.joint_index,
        delta_deg=args.force_position_step_deg,
    )
    print(
        f"{arm.side}: force-position test sensor={args.force_sensor} mode={args.force_mode} "
        f"dir={args.force_dir} force={args.force:.3f}N nudge={args.force_position_step_deg:+.3f}deg"
    )
    force_active = False
    try:
        _check_start_force_safe(robot, arm.side, args)
        if args.clear_force:
            _require_zero(robot.Clear_Force_Data(True), f"{arm.side} Clear_Force_Data")
        _require_zero(robot.Start_Force_Position_Move(True), f"{arm.side} Start_Force_Position_Move")
        force_active = True

        loops = max(1, int(round(args.force_position_hold_sec / args.cycle_sec)))
        for _ in range(loops):
            _require_zero(
                robot.Force_Position_Move_Joint(
                    target,
                    sensor=args.force_sensor,
                    mode=args.force_mode,
                    dir=args.force_dir,
                    force=args.force,
                    follow=args.force_follow,
                ),
                f"{arm.side} Force_Position_Move_Joint",
            )
            time.sleep(args.cycle_sec)

        if abs(args.force_position_step_deg) > 1e-9:
            _require_zero(
                robot.Force_Position_Move_Joint(
                    joint_deg,
                    sensor=args.force_sensor,
                    mode=args.force_mode,
                    dir=args.force_dir,
                    force=args.force,
                    follow=args.force_follow,
                ),
                f"{arm.side} force-position return command",
            )
            time.sleep(args.settle_sec)
        print(f"{arm.side}: force-position test complete")
    finally:
        if force_active:
            stop_code = robot.Stop_Force_Position_Move(True)
            print(f"{arm.side}: Stop_Force_Position_Move -> {stop_code}")


def _selected_arms(args) -> list[ArmTarget]:
    arms = {
        "left": ArmTarget("left", args.left_ip),
        "right": ArmTarget("right", args.right_ip),
    }
    if args.arm == "both":
        return [arms["left"], arms["right"]]
    return [arms[args.arm]]


def _validate_motion_args(args) -> None:
    if not args.execute:
        return
    if args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"real motion requires --confirm {CONFIRM_TEXT}")
    if args.arm == "both" and not args.allow_both_execute:
        raise RuntimeError("executing both arms at once requires --allow-both-execute")
    if not args.position and not args.force_position:
        raise RuntimeError("use --position, --force-position, or omit --execute for read-only")
    if not 1 <= args.joint_index <= 6:
        raise RuntimeError("--joint-index must be in [1, 6]")
    if abs(args.position_step_deg) > args.max_position_step_deg:
        raise RuntimeError("--position-step-deg exceeds --max-position-step-deg")
    if abs(args.force_position_step_deg) > args.max_force_position_step_deg:
        raise RuntimeError("--force-position-step-deg exceeds --max-force-position-step-deg")
    if args.movej_speed < 1 or args.movej_speed > 10:
        raise RuntimeError("--movej-speed must be in [1, 10] for this safety test")
    if args.cycle_sec < 0.01:
        raise RuntimeError("--cycle-sec must be >= 0.01")
    if args.max_start_force_n <= 0.0:
        raise RuntimeError("--max-start-force-n must be positive")
    if args.max_start_torque_nm <= 0.0:
        raise RuntimeError("--max-start-torque-nm must be positive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safe RM65 real-robot smoke test. Defaults to read-only state/force reads; "
            "real motion requires --execute --confirm MOVE_RM65."
        )
    )
    parser.add_argument("--arm", choices=("left", "right", "both"), default="both")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--execute", action="store_true", help="Allow real robot commands.")
    parser.add_argument("--confirm", default="", help=f"Must be exactly {CONFIRM_TEXT} for motion.")
    parser.add_argument("--allow-both-execute", action="store_true")
    parser.add_argument("--position", action="store_true", help="Run a small MoveJ position nudge.")
    parser.add_argument("--force-position", action="store_true", help="Run force-position mode hold.")
    parser.add_argument("--joint-index", type=int, default=6, help="1-based joint index for tiny nudges.")
    parser.add_argument("--position-step-deg", type=float, default=0.5)
    parser.add_argument("--max-position-step-deg", type=float, default=1.0)
    parser.add_argument("--movej-speed", type=int, default=3, help="RealMan MoveJ speed percent.")
    parser.add_argument("--force-position-step-deg", type=float, default=0.0)
    parser.add_argument("--max-force-position-step-deg", type=float, default=0.2)
    parser.add_argument("--force-sensor", type=int, choices=(0, 1), default=1)
    parser.add_argument("--force-mode", type=int, choices=(0, 1), default=0)
    parser.add_argument("--force-dir", type=int, choices=range(6), default=2)
    parser.add_argument("--force", type=float, default=0.0)
    parser.add_argument("--allow-nonzero-force", action="store_true")
    parser.add_argument("--force-follow", action="store_true")
    parser.add_argument("--check-start-force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-precheck-source", choices=("all", "zero", "work", "tool"), default="all")
    parser.add_argument("--max-start-force-n", type=float, default=5.0)
    parser.add_argument("--max-start-torque-nm", type=float, default=0.5)
    parser.add_argument("--clear-force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-position-hold-sec", type=float, default=1.0)
    parser.add_argument("--cycle-sec", type=float, default=0.05)
    parser.add_argument("--settle-sec", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_motion_args(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    selected_arms = _selected_arms(args)
    print("RM65 safe real-robot test")
    print(f"execute={args.execute} position={args.position} force_position={args.force_position}")
    print("Selected arms:", ", ".join(f"{arm.side}={arm.ip}" for arm in selected_arms))
    if not args.execute:
        print("READ-ONLY: no motion or force-position mode will be started.")

    for arm in selected_arms:
        robot = None
        try:
            try:
                robot, joint_deg, _ = _connect_and_read(arm)
                _read_force(robot, arm.side)
                if args.execute and args.position:
                    _position_nudge(robot, arm, joint_deg, args)
                    code, joint_deg, _, _, _ = robot.Get_Current_Arm_State(retry=2)
                    _require_zero(code, f"{arm.side} state after position test")
                if args.execute and args.force_position:
                    _force_position_hold(robot, arm, joint_deg, args)
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
        finally:
            if robot is not None:
                robot.Arm_Socket_Close()

    print("RM65 safe test finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
