from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable


def _repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("could not locate the DexTeleop repository root")


REPO_ROOT = _repository_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65


CONFIRM_TEXT = "CONFIG_RM65_FORCE"


def _format(values: Iterable[float], *, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _force_norms(wrench: Iterable[float]) -> tuple[float, float]:
    values = [float(value) for value in wrench]
    return (
        math.sqrt(sum(value * value for value in values[:3])),
        math.sqrt(sum(value * value for value in values[3:])),
    )


def _selected_ip(args) -> tuple[str, str]:
    if args.arm == "left":
        return "left", args.left_ip
    return "right", args.right_ip


def _print_robot_state(robot: Arm, side: str) -> None:
    code, joint_deg, pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=2)
    _require_zero(code, f"{side} Get_Current_Arm_State")
    print(f"{side}: joint_deg={_format(joint_deg, precision=3)}")
    print(f"{side}: pose_xyz_rpy={_format(pose, precision=4)}")
    print(f"{side}: arm_err={arm_err} sys_err={sys_err}")

    install = robot.Get_Install_Pose()
    print(f"{side}: install_pose={install}")

    tools = robot.Get_Total_Tool_Frame()
    print(f"{side}: tool_frames={tools}")

    code, current_tool = robot.Get_Current_Tool_Frame()
    _require_zero(code, f"{side} Get_Current_Tool_Frame")
    print(f"{side}: current_tool={current_tool}")

    code, force, zero_force, work_zero, tool_zero = robot.Get_Force_Data()
    _require_zero(code, f"{side} Get_Force_Data")
    print(f"{side}: force_raw={_format(force)}")
    print(f"{side}: force_zero={_format(zero_force)} norm={_format(_force_norms(zero_force))}")
    print(f"{side}: force_work_zero={_format(work_zero)} norm={_format(_force_norms(work_zero))}")
    print(f"{side}: force_tool_zero={_format(tool_zero)} norm={_format(_force_norms(tool_zero))}")


def _configure_tool(robot: Arm, side: str, args) -> None:
    pose = [
        args.tool_x,
        args.tool_y,
        args.tool_z,
        args.tool_rx,
        args.tool_ry,
        args.tool_rz,
    ]
    if args.update_tool:
        code = robot.Update_Tool_Frame(
            args.tool_name,
            pose,
            args.payload_kg,
            args.com_x,
            args.com_y,
            args.com_z,
        )
        label = "Update_Tool_Frame"
    else:
        code = robot.Manual_Set_Tool_Frame(
            args.tool_name,
            pose,
            args.payload_kg,
            args.com_x,
            args.com_y,
            args.com_z,
        )
        label = "Manual_Set_Tool_Frame"
    _require_zero(code, f"{side} {label}")
    _require_zero(robot.Change_Tool_Frame(args.tool_name), f"{side} Change_Tool_Frame")
    print(
        f"{side}: active tool '{args.tool_name}' payload={args.payload_kg:.4f}kg "
        f"com_m={_format([args.com_x, args.com_y, args.com_z])} "
        f"pose={_format(pose)}"
    )


def _set_install_pose(robot: Arm, side: str, args) -> None:
    _require_zero(
        robot.Set_Install_Pose(args.install_x, args.install_y, args.install_z),
        f"{side} Set_Install_Pose",
    )
    print(
        f"{side}: install pose set to "
        f"x={args.install_x:.3f}deg y={args.install_y:.3f}deg z={args.install_z:.3f}deg"
    )


def _auto_force_calibration(robot: Arm, side: str) -> None:
    print(f"{side}: starting vendor automatic six-axis force gravity calibration")
    print(f"{side}: robot will move through RealMan calibration poses; keep workspace clear")
    _require_zero(robot.Set_Force_Sensor(True), f"{side} Set_Force_Sensor")
    print(f"{side}: force sensor gravity calibration complete")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Configure RM65 force-reading standard: install pose, tool payload/CoM, "
            "and force-sensor gravity calibration."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Must be exactly {CONFIRM_TEXT} for writes/motion.")

    parser.add_argument("--set-install-pose", action="store_true")
    parser.add_argument("--install-x", type=float, default=0.0)
    parser.add_argument("--install-y", type=float, default=0.0)
    parser.add_argument("--install-z", type=float, default=0.0)

    parser.add_argument("--set-tool", action="store_true")
    parser.add_argument("--update-tool", action="store_true")
    parser.add_argument("--tool-name", default="dexee")
    parser.add_argument("--tool-x", type=float, default=0.0, help="TCP X offset in meters.")
    parser.add_argument("--tool-y", type=float, default=0.0, help="TCP Y offset in meters.")
    parser.add_argument("--tool-z", type=float, default=0.0, help="TCP Z offset in meters.")
    parser.add_argument("--tool-rx", type=float, default=0.0, help="TCP RX offset in radians.")
    parser.add_argument("--tool-ry", type=float, default=0.0, help="TCP RY offset in radians.")
    parser.add_argument("--tool-rz", type=float, default=0.0, help="TCP RZ offset in radians.")
    parser.add_argument("--payload-kg", type=float, default=0.0)
    parser.add_argument("--com-x", type=float, default=0.0, help="Payload CoM X in meters.")
    parser.add_argument("--com-y", type=float, default=0.0, help="Payload CoM Y in meters.")
    parser.add_argument("--com-z", type=float, default=0.0, help="Payload CoM Z in meters.")

    parser.add_argument("--auto-force-calibration", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    writes_or_motion = args.set_install_pose or args.set_tool or args.auto_force_calibration
    if writes_or_motion and (not args.execute or args.confirm != CONFIRM_TEXT):
        print(
            f"ERROR: writes/calibration require --execute --confirm {CONFIRM_TEXT}",
            file=sys.stderr,
        )
        return 2
    if len(args.tool_name) > 10:
        print("ERROR: --tool-name must be 10 characters or fewer", file=sys.stderr)
        return 2
    if args.payload_kg < 0.0:
        print("ERROR: --payload-kg must be nonnegative", file=sys.stderr)
        return 2

    side, ip = _selected_ip(args)
    robot = None
    try:
        robot = Arm(RM65, ip)
        print(f"{side}: connected ip={ip}")
        _print_robot_state(robot, side)
        if args.set_install_pose:
            _set_install_pose(robot, side, args)
        if args.set_tool:
            _configure_tool(robot, side, args)
        if args.auto_force_calibration:
            _auto_force_calibration(robot, side)
        if writes_or_motion:
            print(f"{side}: post-configuration state")
            _print_robot_state(robot, side)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if robot is not None:
            robot.Arm_Socket_Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
