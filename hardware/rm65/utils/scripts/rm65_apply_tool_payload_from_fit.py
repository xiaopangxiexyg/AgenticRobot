from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
UTILS_DIR = SCRIPT_DIR.parent
RESULTS_DIR = UTILS_DIR / "results"
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65
from hardware.rm65.standard_paths import resolve_fit_path
from hardware.rm65.calibration_profiles import (
    load_profile,
    make_payload,
    make_profile,
    make_tcp_from_fit,
    save_profile,
    update_payload,
    update_tcp,
)


CONFIRM_TEXT = "APPLY_RM65_TOOL_PAYLOAD"


def _format(values: Iterable[float], *, precision: int = 5) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _finite_float_list(values: Iterable[float], *, expected_len: int) -> list[float]:
    out = [float(value) for value in values]
    if len(out) != expected_len:
        raise RuntimeError(f"expected {expected_len} values, got {len(out)}")
    if not all(math.isfinite(value) for value in out):
        raise RuntimeError(f"non-finite values: {out}")
    return out


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _selected_ip(args, arm: str) -> tuple[str, str]:
    if arm == "left":
        return "left", args.left_ip
    if arm == "right":
        return "right", args.right_ip
    raise RuntimeError(f"unsupported arm: {arm}")


def _connect_robot(side: str, ip: str) -> Arm:
    robot = Arm(RM65, ip)
    if int(robot.Arm_Socket_State()) != 0:
        robot.Arm_Socket_Close()
        raise RuntimeError(f"{side}: failed to connect ip={ip}")
    print(f"{side}: connected ip={ip}")
    return robot


def _latest_fit_path(arm: str, tool_label: str | None) -> Path:
    return resolve_fit_path(arm, tool_label, require_existing=True)


def _load_fit(path: Path, args) -> tuple[dict[str, object], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fit = payload.get("fit")
    if not isinstance(fit, dict):
        raise RuntimeError(f"{path} does not contain a fit object")
    if fit.get("mode") != "tool_gravity_model":
        raise RuntimeError(f"{path} is not a tool gravity model fit")
    if fit.get("source") != "tool" and not args.allow_non_tool_source:
        raise RuntimeError(
            f"{path} was fitted from source={fit.get('source')}; "
            "use --allow-non-tool-source only after reviewing frame semantics"
        )
    return payload, fit


def _pose_from_tool_frame(frame: object) -> list[float] | None:
    if isinstance(frame, tuple) and len(frame) >= 2:
        frame = frame[1]
    if isinstance(frame, dict):
        pose = frame.get("pose")
        if pose is not None:
            return _finite_float_list(pose, expected_len=6)
    return None


def _tool_name_from_fit(payload: dict[str, object], args) -> str:
    name = args.tool_name or args.expected_tool or str(payload.get("tool_label") or "")
    if not name:
        raise RuntimeError("pass --tool-name or use a fit with tool_label")
    if len(name) > 10:
        raise RuntimeError("--tool-name must be 10 characters or fewer")
    return name


def _fit_tool_params(fit: dict[str, object]) -> tuple[float, list[float]]:
    payload_kg = float(fit["mass_kg_abs"])
    com_m = _finite_float_list(fit["com_tool_m"], expected_len=3)
    if payload_kg <= 0.0:
        raise RuntimeError(f"invalid payload mass: {payload_kg}")
    return payload_kg, com_m


def _load_or_create_profile(args, arm: str, tool_name: str, fit_path: Path) -> dict[str, object]:
    profile_ref = args.profile or tool_name
    try:
        profile = load_profile(profile_ref, arm=arm)
    except (FileNotFoundError, ValueError):
        profile = make_profile(
            name=tool_name if not args.profile else Path(args.profile).stem,
            arm=arm,
            tool_name=tool_name,
            fit_json=str(fit_path),
        )
        if args.profile and Path(args.profile).suffix.lower() == ".json":
            profile["_profile_path"] = str(Path(args.profile))
    if str(profile.get("arm")) != arm:
        raise RuntimeError(f"profile arm={profile.get('arm')} does not match --arm {arm}")
    if str(profile.get("tool_name") or "") != tool_name:
        raise RuntimeError(f"profile tool_name={profile.get('tool_name')} does not match tool {tool_name}")
    paths = profile.setdefault("paths", {})
    if isinstance(paths, dict):
        paths["fit_json"] = str(fit_path)
    return profile


def _profile_tcp_is_confirmed(profile: dict[str, object]) -> bool:
    tcp = profile.get("tcp")
    return isinstance(tcp, dict) and bool(tcp.get("confirmed"))


def _profile_update_preview(profile: dict[str, object]) -> str:
    tcp = profile.get("tcp") if isinstance(profile.get("tcp"), dict) else {}
    payload = profile.get("payload") if isinstance(profile.get("payload"), dict) else {}
    return json.dumps(
        {
            "profile": profile.get("_profile_path") or f"{profile.get('arm')}:{profile.get('name')}",
            "payload": {
                "mass_kg": payload.get("mass_kg"),
                "com_tool_m": payload.get("com_tool_m"),
            },
            "tcp": {
                "pose_flange_m_rad": tcp.get("pose_flange_m_rad"),
                "position_source": tcp.get("position_source"),
                "confirmed": tcp.get("confirmed"),
                "warnings": tcp.get("warnings", []),
            },
        },
        indent=2,
    )


def _warn_fit_quality(fit: dict[str, object]) -> None:
    force_bias = _finite_float_list(fit.get("force_bias_tool_n", [0, 0, 0]), expected_len=3)
    torque_bias = _finite_float_list(fit.get("torque_bias_tool_nm", [0, 0, 0]), expected_len=3)
    gravity_fraction = float(fit.get("gravity_fraction_of_mean_force", 0.0))
    corrected_max = float(fit.get("corrected_force_norm_max_n", math.nan))
    print(f"fit gravity_fraction={gravity_fraction:.5f}")
    print(f"fit corrected_force_norm_max={corrected_max:.5f}N")
    print(f"fit fixed_force_bias_tool_n={_format(force_bias)} |bias|={_norm(force_bias):.5f}N")
    print(
        f"fit fixed_torque_bias_tool_nm={_format(torque_bias)} "
        f"|bias|={_norm(torque_bias):.5f}Nm"
    )
    if gravity_fraction < 0.2:
        print(
            "WARNING: fitted gravity term is weak; payload mass/CoM may be less reliable "
            "than an empirical bias model"
        )
    if corrected_max > 1.0:
        print("WARNING: corrected max force is >1N; inspect visualization before applying")
    print(
        "NOTE: fixed force/torque bias is reported for diagnosis only. "
        "The RealMan tool frame accepts payload mass and CoM, not this bias term."
    )


def _resolve_tool_pose(robot: Arm | None, args) -> list[float]:
    if args.tool_pose_source == "cli":
        return [
            args.tool_x,
            args.tool_y,
            args.tool_z,
            args.tool_rx,
            args.tool_ry,
            args.tool_rz,
        ]
    if robot is None:
        raise RuntimeError("--tool-pose-source current/given requires robot connection")
    if args.tool_pose_source == "given":
        code, frame = robot.Get_Given_Tool_Frame(args.tool_name)
        _require_zero(code, "Get_Given_Tool_Frame")
        pose = _pose_from_tool_frame(frame)
        if pose is None:
            raise RuntimeError(f"could not read pose from tool frame {args.tool_name}")
        return pose
    code, frame = robot.Get_Current_Tool_Frame()
    _require_zero(code, "Get_Current_Tool_Frame")
    pose = _pose_from_tool_frame(frame)
    if pose is None:
        raise RuntimeError("could not read current tool pose")
    return pose


def _tool_exists(robot: Arm, tool_name: str) -> bool:
    frames = robot.Get_Total_Tool_Frame()
    if isinstance(frames, dict):
        names = [str(name).strip("\x00") for name in frames.get("tool_names", [])]
        if int(frames.get("return_code", 0)) == 0:
            print(f"controller_tool_frames={names}")
            return tool_name in names
    code, _frame = robot.Get_Given_Tool_Frame(tool_name)
    return int(code) == 0


def _resolve_write_mode(robot: Arm, args) -> str:
    if args.write_mode != "auto":
        return args.write_mode
    return "update" if _tool_exists(robot, args.tool_name) else "create"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply RM65 payload mass/CoM estimated by rm65_force_payload_calibrate.py "
            "to an official RealMan tool frame. Defaults to dry-run."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), default="")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--fit-json", default="")
    parser.add_argument("--tool-label", default="")
    parser.add_argument("--tool-name", default="")
    parser.add_argument("--expected-tool", default="", help="Alias for --tool-name in standard calibration commands.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Must be exactly {CONFIRM_TEXT} for writes.")
    parser.add_argument(
        "--write-mode",
        choices=("auto", "update", "create"),
        default="auto",
        help="auto updates an existing tool frame or creates it when absent.",
    )
    parser.add_argument(
        "--update-tool",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated compatibility flag; use --write-mode instead.",
    )
    parser.add_argument("--change-tool", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clear-force-data", action="store_true")
    parser.add_argument(
        "--tool-pose-source",
        choices=("current", "given", "cli"),
        default="current",
        help="TCP pose to preserve while updating payload/CoM.",
    )
    parser.add_argument("--tool-x", type=float, default=0.0)
    parser.add_argument("--tool-y", type=float, default=0.0)
    parser.add_argument("--tool-z", type=float, default=0.0)
    parser.add_argument("--tool-rx", type=float, default=0.0)
    parser.add_argument("--tool-ry", type=float, default=0.0)
    parser.add_argument("--tool-rz", type=float, default=0.0)
    parser.add_argument("--allow-non-tool-source", action="store_true")
    parser.add_argument("--profile", default="", help="Calibration profile name or path. Defaults to the tool name.")
    parser.add_argument(
        "--save-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update the arm/tool calibration profile with payload and a default TCP candidate.",
    )
    parser.add_argument(
        "--overwrite-tcp",
        action="store_true",
        help="Replace an existing confirmed TCP in the profile. By default confirmed TCP entries are preserved.",
    )
    parser.add_argument(
        "--tcp-seed-source",
        choices=("auto", "com", "hand-base", "arm-endpoint"),
        default="auto",
        help=(
            "Initial TCP source saved to the profile. auto uses COM unless the COM is unsafe; "
            "inspire defaults to hand-base."
        ),
    )
    parser.add_argument("--max-tcp-com-distance-m", type=float, default=0.20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.execute or args.clear_force_data) and args.confirm != CONFIRM_TEXT:
        print(f"ERROR: writes require --confirm {CONFIRM_TEXT}", file=sys.stderr)
        return 2

    try:
        fit_path = (
            Path(args.fit_json)
            if args.fit_json
            else _latest_fit_path(args.arm or "right", args.tool_label or args.tool_name or args.expected_tool or None)
        )
        payload, fit = _load_fit(fit_path, args)
        arm = args.arm or str(payload.get("arm") or "")
        side, ip = _selected_ip(args, arm)
        tool_name = _tool_name_from_fit(payload, args)
        args.tool_name = tool_name
        payload_kg, com_m = _fit_tool_params(fit)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"fit_json={fit_path}")
    print(f"arm={side} ip={ip}")
    print(f"tool_name={tool_name}")
    print(f"payload_kg={payload_kg:.6f}")
    print(f"com_tool_m={_format(com_m)}")
    _warn_fit_quality(fit)

    robot = None
    try:
        if args.execute or args.tool_pose_source in ("current", "given"):
            robot = _connect_robot(side, ip)
        tool_pose = _resolve_tool_pose(robot, args)
        print(f"tool_pose={_format(tool_pose)} source={args.tool_pose_source}")
        profile = None
        if args.save_profile:
            profile = _load_or_create_profile(args, side, tool_name, fit_path)
            update_payload(
                profile,
                make_payload(
                    mass_kg=payload_kg,
                    com_tool_m=com_m,
                    source_fit_json=str(fit_path),
                    fit_source=str(fit.get("source") or ""),
                    fit_mode=str(fit.get("mode") or ""),
                ),
            )
            if args.overwrite_tcp or not _profile_tcp_is_confirmed(profile):
                update_tcp(
                    profile,
                    make_tcp_from_fit(
                        tool_name=tool_name,
                        tool_pose_flange_m_rad=tool_pose,
                        com_tool_m=com_m,
                        source_fit_json=str(fit_path),
                        seed_source=args.tcp_seed_source,
                        max_com_distance_m=args.max_tcp_com_distance_m,
                    ),
                )
            else:
                print("profile TCP is confirmed; preserving it (pass --overwrite-tcp to replace)")
            print("profile_update_preview=")
            print(_profile_update_preview(profile))
        if args.update_tool is not None:
            args.write_mode = "update" if args.update_tool else "create"
            print(f"WARNING: --update-tool is deprecated; using write_mode={args.write_mode}")
        write_mode = _resolve_write_mode(robot, args) if robot is not None else args.write_mode

        if not args.execute:
            print(f"DRY-RUN: would {write_mode} and activate tool frame '{tool_name}'")
            if args.save_profile:
                print("DRY-RUN: would save calibration profile after execute")
            return 0

        if write_mode == "update":
            code = robot.Update_Tool_Frame(tool_name, tool_pose, payload_kg, *com_m)
            label = "Update_Tool_Frame"
        else:
            code = robot.Manual_Set_Tool_Frame(tool_name, tool_pose, payload_kg, *com_m)
            label = "Manual_Set_Tool_Frame"
        _require_zero(code, f"{side} {label}")
        if args.change_tool:
            _require_zero(robot.Change_Tool_Frame(tool_name), f"{side} Change_Tool_Frame")
        if args.clear_force_data:
            _require_zero(robot.Clear_Force_Data(True), f"{side} Clear_Force_Data")
        print(f"{side}: applied payload/CoM to tool '{tool_name}'")
        if args.save_profile and profile is not None:
            saved = save_profile(profile, profile.get("_profile_path"))
            print(f"saved payload/TCP profile: {saved}")
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if robot is not None:
            robot.Arm_Socket_Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
