from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.calibration_profiles import load_profile, save_profile, update_impedance  # noqa: E402
from hardware.rm65.impedance import (  # noqa: E402
    RM65ImpedanceConfig,
    RM65JointImpedanceController,
    RM65PoseImpedanceController,
)


CONFIRM_TEXT = "TEST_RM65_FIXED_POINT_IMPEDANCE"


def _normalize_argv(argv: list[str] | None) -> list[str] | None:
    if argv is None:
        argv = sys.argv[1:]
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--angular-axis-signs" and index + 1 < len(argv):
            value = argv[index + 1]
            if value.startswith("-") and "," in value:
                normalized.append(f"{item}={value}")
                index += 2
                continue
        normalized.append(item)
        index += 1
    return normalized


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "max": math.nan}
    return {"mean": statistics.fmean(values), "max": max(values)}


def _config_dict(config: RM65ImpedanceConfig) -> dict[str, object]:
    profile_metadata_fields = {
        "tcp_pose_flange_m_rad",
        "tcp_confirmed",
        "payload_mass_kg",
        "payload_com_tool_m",
        "debug_arm_endpoint_tcp",
        "debug_tool_frame_name",
    }
    return {
        key: value
        for key, value in vars(config).items()
        if not key.startswith("_") and key not in profile_metadata_fields
    }


def _profile_metadata_dict(config: RM65ImpedanceConfig) -> dict[str, object]:
    return {
        "tcp_pose_flange_m_rad": config.tcp_pose_flange_m_rad,
        "tcp_confirmed": config.tcp_confirmed,
        "payload_mass_kg": config.payload_mass_kg,
        "payload_com_tool_m": config.payload_com_tool_m,
    }


def _print_saturation_estimates(config: RM65ImpedanceConfig) -> None:
    thresholds = config.saturation_thresholds()
    if not any(value is not None for value in thresholds.values()):
        return
    print("admittance_saturation_estimates=")
    print(json.dumps(thresholds, indent=2))
    linear_filtered = thresholds.get("linear_filtered_force_n_for_max_offset")
    angular_filtered = thresholds.get("angular_filtered_torque_nm_for_max_offset")
    if linear_filtered is not None and linear_filtered <= 2.0:
        print(
            "WARNING: current linear compliance is aggressive; "
            f"filtered force above about {linear_filtered:.3f}N already saturates max_linear_offset_m"
        )
    if angular_filtered is not None and angular_filtered <= 0.2:
        print(
            "WARNING: current angular compliance is aggressive; "
            f"filtered torque above about {angular_filtered:.3f}Nm already saturates max_angular_offset_rad"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed-position impedance tuning for RM65. It holds the current pose or "
            "joint target and applies Cartesian compliance from external wrench."
        )
    )
    parser.add_argument("--mode", choices=("pose", "joint"), default="pose")
    parser.add_argument("--profile", default="", help="Calibration profile name or path.")
    parser.add_argument("--arm", choices=("left", "right"), default="")
    parser.add_argument("--tool-name", default="")
    parser.add_argument("--duration-sec", type=float, default=20.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--save-to-profile", action="store_true")

    parser.add_argument("--control-hz", type=float, default=None)
    parser.add_argument("--action-hz", type=float, default=None)
    parser.add_argument("--force-source", choices=("zero", "work", "tool"), default="")
    parser.add_argument("--axes", default="")
    parser.add_argument("--force-sign", type=float, choices=(-1.0, 1.0), default=None)
    parser.add_argument("--angular-force-sign", type=float, choices=(-1.0, 1.0), default=None)
    parser.add_argument("--angular-axis-signs", default="")
    parser.add_argument("--lowpass-alpha", type=float, default=None)
    parser.add_argument("--force-deadband-n", type=float, default=None)
    parser.add_argument("--torque-deadband-nm", type=float, default=None)
    parser.add_argument("--linear-compliance-m-per-n", type=float, default=None)
    parser.add_argument("--angular-compliance-rad-per-nm", type=float, default=None)
    parser.add_argument("--linear-damping-n-s-per-m", type=float, default=None)
    parser.add_argument("--angular-damping-nm-s-per-rad", type=float, default=None)
    parser.add_argument("--linear-gain-m-s-per-n", type=float, default=None)
    parser.add_argument("--angular-gain-rad-s-per-nm", type=float, default=None)
    parser.add_argument("--max-linear-velocity-m-s", type=float, default=None)
    parser.add_argument("--max-angular-velocity-rad-s", type=float, default=None)
    parser.add_argument("--max-linear-offset-m", type=float, default=None)
    parser.add_argument("--max-angular-offset-rad", type=float, default=None)
    parser.add_argument("--hard-stop-force-n", type=float, default=None)
    parser.add_argument("--hard-stop-torque-nm", type=float, default=None)
    parser.add_argument("--max-command-joint-step-deg", type=float, default=None)
    parser.add_argument(
        "--ik-offset-backoff-min-scale",
        type=float,
        default=None,
        help=(
            "When IK fails for a Cartesian admittance offset, retry smaller offset scales "
            "down to this value. Default keeps the original fail-fast behavior."
        ),
    )
    parser.add_argument(
        "--debug-arm-endpoint-tcp",
        action="store_true",
        help=(
            "Temporarily switch the active RealMan tool frame to a debug frame with "
            "TCP pose [0,0,0,0,0,0], then restore the original tool frame on exit."
        ),
    )
    parser.add_argument(
        "--debug-tool-frame-name",
        default="",
        help=(
            "Optional RealMan tool frame name used with --debug-arm-endpoint-tcp. "
            "RealMan frame names are limited to 10 characters; longer names are truncated."
        ),
    )
    parser.add_argument("--print-every", type=int, default=10)
    return parser


def _parse_signs(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _overrides_from_args(args) -> dict[str, object]:
    mapping = {
        "control_hz": args.control_hz,
        "action_hz": args.action_hz,
        "force_source": args.force_source or None,
        "axes": args.axes or None,
        "force_sign": args.force_sign,
        "angular_force_sign": args.angular_force_sign,
        "lowpass_alpha": args.lowpass_alpha,
        "force_deadband_n": args.force_deadband_n,
        "torque_deadband_nm": args.torque_deadband_nm,
        "linear_compliance_m_per_n": args.linear_compliance_m_per_n,
        "angular_compliance_rad_per_nm": args.angular_compliance_rad_per_nm,
        "linear_damping_n_s_per_m": args.linear_damping_n_s_per_m,
        "angular_damping_nm_s_per_rad": args.angular_damping_nm_s_per_rad,
        "linear_gain_m_s_per_n": args.linear_gain_m_s_per_n,
        "angular_gain_rad_s_per_nm": args.angular_gain_rad_s_per_nm,
        "max_linear_velocity_m_s": args.max_linear_velocity_m_s,
        "max_angular_velocity_rad_s": args.max_angular_velocity_rad_s,
        "max_linear_offset_m": args.max_linear_offset_m,
        "max_angular_offset_rad": args.max_angular_offset_rad,
        "hard_stop_force_n": args.hard_stop_force_n,
        "hard_stop_torque_nm": args.hard_stop_torque_nm,
        "max_command_joint_step_deg": args.max_command_joint_step_deg,
        "ik_offset_backoff_min_scale": args.ik_offset_backoff_min_scale,
        "debug_arm_endpoint_tcp": args.debug_arm_endpoint_tcp,
        "debug_tool_frame_name": args.debug_tool_frame_name or None,
    }
    if args.angular_axis_signs:
        mapping["angular_axis_signs"] = _parse_signs(args.angular_axis_signs)
    return {key: value for key, value in mapping.items() if value is not None}


def _build_config(args) -> tuple[RM65ImpedanceConfig, dict[str, object] | None, str]:
    overrides = _overrides_from_args(args)
    if args.profile:
        profile = load_profile(args.profile, arm=args.arm or None)
        config = RM65ImpedanceConfig.from_profile(profile, mode=args.mode, **overrides)
        arm = args.arm or str(profile["arm"])
        return config, profile, arm
    if not args.arm or not args.tool_name:
        raise RuntimeError("without --profile, pass both --arm and --tool-name")
    config = RM65ImpedanceConfig(tool_name=args.tool_name, **overrides)
    return config, None, args.arm


def _write_report(path_text: str, report: dict[str, object]) -> None:
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote report: {path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(_normalize_argv(argv))
    if args.execute and args.confirm != CONFIRM_TEXT:
        print(f"ERROR: execution requires --confirm {CONFIRM_TEXT}", file=sys.stderr)
        return 2
    if args.duration_sec <= 0.0:
        print("ERROR: --duration-sec must be positive", file=sys.stderr)
        return 2
    try:
        config, profile, arm = _build_config(args)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"mode={args.mode} arm={arm} tool={config.tool_name} duration_sec={args.duration_sec:.3f}")
    print(json.dumps(_config_dict(config), indent=2))
    metadata = _profile_metadata_dict(config)
    if any(value is not None and value is not False for value in metadata.values()):
        print("profile_metadata=")
        print(json.dumps(metadata, indent=2))
    _print_saturation_estimates(config)
    if not args.execute:
        print("DRY-RUN: no impedance command will be sent.")
        return 0

    controller_cls = RM65PoseImpedanceController if args.mode == "pose" else RM65JointImpedanceController
    records = []
    try:
        with controller_cls(arm=arm, config=config) as controller:
            records = controller.run(args.duration_sec, print_every=args.print_every)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = {
        "samples": len(records),
        "force_norm_n": _stats([float(row["force_norm_n"]) for row in records]),
        "torque_norm_nm": _stats([float(row["torque_norm_nm"]) for row in records]),
        "active_force_norm_n": _stats(
            [
                math.sqrt(sum(float(value) * float(value) for value in row["active_wrench"][:3]))
                for row in records
            ]
        ),
        "active_torque_norm_nm": _stats(
            [
                math.sqrt(sum(float(value) * float(value) for value in row["active_wrench"][3:]))
                for row in records
            ]
        ),
        "linear_offset_m": _stats(
            [
                math.sqrt(sum(float(value) * float(value) for value in row["offset"][:3]))
                for row in records
            ]
        ),
        "angular_offset_rad": _stats(
            [
                math.sqrt(sum(float(value) * float(value) for value in row["offset"][3:]))
                for row in records
            ]
        ),
        "command_joint_step_deg": _stats([float(row["command_joint_step_deg"]) for row in records]),
        "pose_tracking_linear_m": _stats([float(row["pose_tracking_linear_m"]) for row in records]),
        "pose_tracking_angular_rad": _stats([float(row["pose_tracking_angular_rad"]) for row in records]),
    }
    report = {
        "schema_version": 1,
        "data_type": "rm65_fixed_point_impedance_tune",
        "mode": args.mode,
        "arm": arm,
        "tool_name": config.tool_name,
        "duration_sec": args.duration_sec,
        "config": _config_dict(config),
        "profile_metadata": _profile_metadata_dict(config),
        "summary": summary,
        "records": records,
    }
    print(f"summary={summary}")
    _write_report(args.output_json, report)

    if args.save_to_profile:
        if profile is None:
            print("WARNING: --save-to-profile ignored because no --profile was provided")
        else:
            update_impedance(profile, args.mode, _config_dict(config))
            path = save_profile(profile, profile.get("_profile_path"))
            print(f"saved tuned {args.mode} impedance to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
