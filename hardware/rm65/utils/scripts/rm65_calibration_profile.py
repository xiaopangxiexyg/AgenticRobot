from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.calibration_profiles import (  # noqa: E402
    ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD,
    finite_float_list,
    list_profiles,
    load_profile,
    make_profile,
    save_profile,
    update_tcp,
    update_impedance,
)
from hardware.rm65.standard_paths import (  # noqa: E402
    force_sensor_calib4_fit_path,
    force_sensor_calib4_measurements_path,
    force_sensor_calib4_pose_sequence_path,
    official_force_sensor_report_path,
    tool_payload_validation_path,
)


IMPEDANCE_FIELDS = {
    "force_source",
    "axes",
    "force_sign",
    "angular_force_sign",
    "angular_axis_signs",
    "angular_axis_velocity_scales",
    "control_hz",
    "action_hz",
    "lowpass_alpha",
    "force_deadband_n",
    "torque_deadband_nm",
    "linear_gain_m_s_per_n",
    "angular_gain_rad_s_per_nm",
    "linear_damping_n_s_per_m",
    "angular_damping_nm_s_per_rad",
    "linear_compliance_m_per_n",
    "angular_compliance_rad_per_nm",
    "max_linear_velocity_m_s",
    "max_angular_velocity_rad_s",
    "max_linear_offset_m",
    "max_angular_offset_rad",
    "return_rate_s",
    "trajectory_mode",
    "radio",
    "hard_stop_force_n",
    "hard_stop_torque_nm",
    "max_command_joint_step_deg",
    "ik_resolver_local_step_max_position_worsen_m",
    "ik_resolver_local_step_component_progress_max_position_worsen_m",
}


def _parse_value(text: str):
    text = text.strip()
    if "," in text:
        return [float(item.strip()) for item in text.split(",") if item.strip()]
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_set_items(items: list[str]) -> dict[str, object]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set item must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if key not in IMPEDANCE_FIELDS:
            raise ValueError(f"unsupported impedance field: {key}")
        out[key] = _parse_value(value)
    return out


def _parse_float_csv(text: str, *, expected_len: int, label: str) -> list[float]:
    return finite_float_list(
        [item.strip() for item in text.split(",") if item.strip()],
        expected_len=expected_len,
        label=label,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create, inspect, and update RM65 calibration profiles.")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create or replace one calibration profile.")
    create.add_argument("--name", required=True)
    create.add_argument("--arm", choices=("left", "right"), required=True)
    create.add_argument("--tool-name", default="")
    create.add_argument("--expected-tool", default="")
    create.add_argument("--pose-sequence-json", default="")
    create.add_argument("--force-measurements-json", default="")
    create.add_argument("--fit-json", default="")
    create.add_argument("--official-report-json", default="")
    create.add_argument("--tool-payload-validation-json", default="")
    create.add_argument("--notes", default="")
    create.add_argument("--output-json", default="")

    show = sub.add_parser("show", help="Print one profile JSON.")
    show.add_argument("--name", required=True)
    show.add_argument("--arm", choices=("left", "right"), default="")

    sub.add_parser("list", help="List saved profiles.")

    update = sub.add_parser("set-impedance", help="Update pose or joint impedance parameters in a profile.")
    update.add_argument("--name", required=True)
    update.add_argument("--arm", choices=("left", "right"), default="")
    update.add_argument("--mode", choices=("pose", "joint"), required=True)
    update.add_argument("--set", action="append", default=[], help="key=value. Example: --set angular_axis_signs=1,1,-1")
    update.add_argument("--from-json", default="", help="Optional JSON object with impedance fields.")

    tcp = sub.add_parser("set-tcp", help="Update one profile TCP pose.")
    tcp.add_argument("--name", required=True)
    tcp.add_argument("--arm", choices=("left", "right"), default="")
    tcp.add_argument("--pose-flange-m-rad", default="", help="x,y,z,rx,ry,rz in meters/radians.")
    tcp.add_argument("--position-flange-m", default="", help="x,y,z in meters; preserves current TCP rotation.")
    tcp.add_argument("--position-source", default="manual")
    tcp.add_argument("--confirmed", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "create":
            tool_name = args.tool_name or args.expected_tool or args.name
            profile = make_profile(
                name=args.name,
                arm=args.arm,
                tool_name=tool_name,
                pose_sequence_json=args.pose_sequence_json or str(force_sensor_calib4_pose_sequence_path(args.arm)),
                force_measurements_json=args.force_measurements_json or str(force_sensor_calib4_measurements_path(args.arm, tool_name)),
                fit_json=args.fit_json or str(force_sensor_calib4_fit_path(args.arm, tool_name)),
                official_report_json=args.official_report_json or str(official_force_sensor_report_path(args.arm, tool_name)),
                tool_payload_validation_json=args.tool_payload_validation_json or str(tool_payload_validation_path(args.arm, tool_name)),
                notes=args.notes,
            )
            path = save_profile(profile, args.output_json or None)
            print(f"wrote profile: {path}")
            return 0

        if args.command == "show":
            profile = load_profile(args.name, arm=args.arm or None)
            print(json.dumps(profile, indent=2))
            return 0

        if args.command == "list":
            profiles = list_profiles()
            if not profiles:
                print("no calibration profiles found")
            for path in profiles:
                try:
                    payload = load_profile(path)
                    print(f"{payload['arm']}: {payload['name']} tool={payload.get('tool_name', '')} path={path}")
                except Exception as exc:
                    print(f"{path}: ERROR {exc}")
            return 0

        if args.command == "set-impedance":
            profile = load_profile(args.name, arm=args.arm or None)
            values = {}
            if args.from_json:
                values.update(json.loads(Path(args.from_json).read_text(encoding="utf-8")))
            values.update(_parse_set_items(args.set))
            update_impedance(profile, args.mode, values)
            path = save_profile(profile, profile.get("_profile_path"))
            print(f"updated {args.mode} impedance in {path}")
            return 0

        if args.command == "set-tcp":
            profile = load_profile(args.name, arm=args.arm or None)
            current_tcp = dict(profile.get("tcp") or {})
            if args.pose_flange_m_rad:
                pose = _parse_float_csv(args.pose_flange_m_rad, expected_len=6, label="pose-flange-m-rad")
            elif args.position_flange_m:
                position = _parse_float_csv(args.position_flange_m, expected_len=3, label="position-flange-m")
                current_pose = current_tcp.get("pose_flange_m_rad") or ARM_ENDPOINT_TCP_POSE_FLANGE_M_RAD
                current_pose = finite_float_list(current_pose, expected_len=6, label="current tcp pose")
                pose = position + current_pose[3:6]
            else:
                raise ValueError("pass --pose-flange-m-rad or --position-flange-m")
            current_tcp.update(
                {
                    "schema_version": 1,
                    "frame_name": current_tcp.get("frame_name") or "tool_tcp",
                    "pose_flange_m_rad": pose,
                    "position_source": args.position_source,
                    "rotation_policy": current_tcp.get("rotation_policy") or "preserve_existing_tool_rotation",
                    "confirmed": bool(args.confirmed),
                }
            )
            update_tcp(profile, current_tcp)
            path = save_profile(profile, profile.get("_profile_path"))
            print(f"updated TCP in {path}")
            return 0
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
