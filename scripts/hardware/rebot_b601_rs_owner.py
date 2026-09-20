"""Run the guarded B601 gravity/damping owner in the foreground.

This file is only an operator launcher.  The existing
``rebot_b601_rs_low_gain_probe.py`` remains the sole implementation of the
hardware state machine, feedback validation, protective hold, and cleanup
receipts.  The launcher supplies the agreed low-gain profile and keeps the
terminal and evidence paths usable without a long pasted shell command.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hardware.rebot_b601_rm65_mujoco_follow import (
    _run_foreground,
)

DEFAULT_REBOT_SOURCE_ROOT = REPO_ROOT / "hardware" / "reBotArm_control_py"
DEFAULT_CONFIG = DEFAULT_REBOT_SOURCE_ROOT / "config" / "rebotarm_rs.yaml"
DEFAULT_OUTPUT_ROOT = Path("/tmp/b601-rs-commissioning")
DEFAULT_CONFIRMATION = "B601_RS_LOW_GAIN_MIT_GRAVITY_DAMPING"
DEFAULT_ASSIST_CONFIRMATION = "B601_RS_SINGLE_JOINT_POLICY_ASSIST"
DEFAULT_CARTESIAN_ASSIST_CONFIRMATION = "B601_RS_CARTESIAN_POLICY_ASSIST"
DEFAULT_DRIVER_KD = (0.20, 0.30, 0.20, 0.15, 0.10, 0.05)
DEFAULT_GRAVITY_SCALES = (1.0, 0.95, 1.0, 1.0, 1.0, 1.0)
DEFAULT_GRAVITY_OFFSETS = (0.0,) * 6
DEFAULT_GRAVITY_BIAS = (0.0, -0.105, -0.1, -0.05, 0.0, 0.0)
DEFAULT_COULOMB = (0.0,) * 6
DEFAULT_VELOCITY_ABORT_RAD_S = 5.0


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Foreground B601-RS gravity/damping owner using the existing guarded "
            "low-gain probe; this command can enable real motors."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_env_path("REBOT_RS_CONFIG", DEFAULT_CONFIG),
        help="rebotarm_rs.yaml",
    )
    parser.add_argument(
        "--rebot-source-root",
        type=Path,
        default=_env_path("REBOT_SOURCE_ROOT", DEFAULT_REBOT_SOURCE_ROOT),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get("B601_RM65_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("hold", "startup-replay"),
        default="hold",
        help="hold is the normal owner; startup-replay is an explicit diagnostic",
    )
    parser.add_argument("--duration-sec", type=float, default=900.0)
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument(
        "--feedback-source",
        choices=("active-report", "parameter"),
        default="active-report",
        help="active-report is the normal host-fresh source; parameter is explicit fallback",
    )
    parser.add_argument("--kp", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.0)
    parser.add_argument(
        "--single-joint-assist",
        action="store_true",
        help=(
            "run one operator-armed relative joint assist trial inside this "
            "same gravity/damping owner"
        ),
    )
    parser.add_argument(
        "--assist-joint",
        choices=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        default="joint6",
    )
    parser.add_argument("--assist-offset-rad", type=float, default=0.10)
    parser.add_argument("--assist-kp-nm-per-rad", type=float, default=1.0)
    parser.add_argument("--assist-kd-nm-s-per-rad", type=float, default=0.0)
    parser.add_argument("--assist-max-target-error-rad", type=float, default=0.05)
    parser.add_argument("--assist-max-torque-nm", type=float, default=0.05)
    parser.add_argument("--assist-torque-slew-nm-s", type=float, default=0.10)
    parser.add_argument("--assist-authority-rise-per-sec", type=float, default=0.5)
    parser.add_argument(
        "--assist-command-stale-timeout-sec",
        type=float,
        default=0.20,
    )
    parser.add_argument("--assist-countdown-sec", type=float, default=3.0)
    parser.add_argument("--assist-duration-sec", type=float, default=4.0)
    parser.add_argument(
        "--assist-stationarity-limit-rad-s",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--assist-min-response-rad",
        type=float,
        default=0.0017453292519943296,
        help="minimum settled response required for hardware acceptance (0.1 deg)",
    )
    parser.add_argument("--assist-max-response-ratio", type=float, default=1.5)
    parser.add_argument(
        "--assist-max-support-drift-rad",
        type=float,
        default=0.008726646259971648,
        help="maximum non-target joint drift accepted during the trial (0.5 deg)",
    )
    parser.add_argument(
        "--assist-confirmation",
        default="",
        help=(
            "required exact acknowledgement for --single-joint-assist: "
            f"{DEFAULT_ASSIST_CONFIRMATION}"
        ),
    )
    parser.add_argument(
        "--cartesian-assist",
        action="store_true",
        help=(
            "run one operator-armed world-aligned translational assist trial "
            "inside this same gravity/damping owner"
        ),
    )
    parser.add_argument(
        "--cartesian-assist-axis",
        choices=("x", "y", "z"),
        default="x",
    )
    parser.add_argument("--cartesian-assist-offset-m", type=float, default=0.01)
    parser.add_argument(
        "--cartesian-assist-stiffness-n-per-m",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--cartesian-assist-damping-n-s-per-m",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--cartesian-assist-lateral-stiffness-n-per-m",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--cartesian-assist-lateral-damping-n-s-per-m",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--cartesian-assist-max-force-n",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--cartesian-assist-max-joint-torque-nm",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--cartesian-assist-joint-torque-slew-nm-s",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--cartesian-assist-authority-rise-per-sec",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--cartesian-assist-countdown-sec",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--cartesian-assist-duration-sec",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--cartesian-assist-stationarity-limit-rad-s",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--cartesian-assist-min-response-m",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--cartesian-assist-max-response-ratio",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--cartesian-assist-max-lateral-drift-m",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--cartesian-assist-max-rotation-drift-rad",
        type=float,
        default=0.08726646259971647,
        help="maximum TCP orientation drift during a translation-only trial (5 deg)",
    )
    parser.add_argument(
        "--cartesian-assist-confirmation",
        default="",
        help=(
            "required exact acknowledgement for --cartesian-assist: "
            f"{DEFAULT_CARTESIAN_ASSIST_CONFIRMATION}"
        ),
    )
    parser.add_argument(
        "--driver-kd",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=DEFAULT_DRIVER_KD,
    )
    parser.add_argument("--gravity-scale", type=float, default=1.0)
    parser.add_argument(
        "--gravity-joint-scales",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=DEFAULT_GRAVITY_SCALES,
    )
    parser.add_argument(
        "--gravity-joint-offsets-rad",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=DEFAULT_GRAVITY_OFFSETS,
    )
    parser.add_argument(
        "--gravity-joint-bias-nm",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=DEFAULT_GRAVITY_BIAS,
    )
    parser.add_argument(
        "--gravity-coulomb-friction-nm",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=DEFAULT_COULOMB,
    )
    parser.add_argument(
        "--gravity-coulomb-velocity-threshold-rad-s",
        type=float,
        default=0.02,
    )
    parser.add_argument("--torque-limit-scale", type=float, default=0.40)
    parser.add_argument("--torque-rise-time-sec", type=float, default=0.5)
    parser.add_argument(
        "--velocity-abort-rad-s",
        type=float,
        default=DEFAULT_VELOCITY_ABORT_RAD_S,
        help=(
            "normal-control finite-difference velocity abort threshold; "
            "the default is a manual-drag commissioning envelope, not a "
            "hardware-certified limit"
        ),
    )
    parser.add_argument("--acceleration-abort-rad-s2", type=float, default=80.0)
    parser.add_argument("--max-error-rad", type=float, default=2.0)
    parser.add_argument("--fault-hold-max-age-sec", type=float, default=0.15)
    parser.add_argument("--fault-hold-max-velocity-rad-s", type=float, default=0.02)
    parser.add_argument("--fault-hold-max-rebase-rad", type=float, default=0.05)
    parser.add_argument(
        "--allow-motion",
        action="store_true",
        help="required explicit acknowledgement that six real joints may enable",
    )
    parser.add_argument(
        "--without-driver-torque-limits",
        action="store_true",
        help="do not apply the normal volatile 0x700B envelope",
    )
    parser.add_argument(
        "--confirmation",
        default=DEFAULT_CONFIRMATION,
    )
    parser.add_argument(
        "--pause-on-exit",
        action="store_true",
        help="wait for Enter after the owner exits (useful for a throw-away terminal)",
    )
    return parser


def build_probe_command(
    args: argparse.Namespace,
    *,
    run_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "hardware" / "rebot_b601_rs_low_gain_probe.py"),
        "--mode",
        args.mode,
        "--config",
        str(args.config),
        "--rebot-source-root",
        str(args.rebot_source_root),
        "--duration-sec",
        str(args.duration_sec),
        "--warmup-sec",
        str(args.warmup_sec),
        "--rate-hz",
        str(args.rate_hz),
        "--feedback-source",
        args.feedback_source,
        "--kp",
        str(args.kp),
        "--kd",
        str(args.kd),
        "--driver-kd",
        *(str(value) for value in args.driver_kd),
        "--gravity-scale",
        str(args.gravity_scale),
        "--gravity-joint-scales",
        *(str(value) for value in args.gravity_joint_scales),
        "--gravity-joint-offsets-rad",
        *(str(value) for value in args.gravity_joint_offsets_rad),
        "--gravity-joint-bias-nm",
        *(str(value) for value in args.gravity_joint_bias_nm),
        "--gravity-coulomb-friction-nm",
        *(str(value) for value in args.gravity_coulomb_friction_nm),
        "--gravity-coulomb-velocity-threshold-rad-s",
        str(args.gravity_coulomb_velocity_threshold_rad_s),
        "--torque-limit-scale",
        str(args.torque_limit_scale),
        "--torque-rise-time-sec",
        str(args.torque_rise_time_sec),
        "--velocity-abort-rad-s",
        str(args.velocity_abort_rad_s),
        "--acceleration-abort-rad-s2",
        str(args.acceleration_abort_rad_s2),
        "--max-error-rad",
        str(args.max_error_rad),
        "--fault-hold-max-age-sec",
        str(args.fault_hold_max_age_sec),
        "--fault-hold-max-velocity-rad-s",
        str(args.fault_hold_max_velocity_rad_s),
        "--fault-hold-max-rebase-rad",
        str(args.fault_hold_max_rebase_rad),
        "--allow-motion",
        "--auto-disable",
        "--confirmation",
        args.confirmation,
        "--output",
        str(run_dir / "b601_owner.json"),
        "--log",
        str(run_dir / "b601_owner.log"),
    ]
    if args.feedback_source == "active-report":
        command.extend(("--allow-active-report", "--auto-disable-active-report"))
    if not args.without_driver_torque_limits:
        command.append("--apply-driver-torque-limits")
    if args.single_joint_assist:
        command.extend(
            (
                "--single-joint-assist",
                "--assist-joint",
                args.assist_joint,
                "--assist-offset-rad",
                str(args.assist_offset_rad),
                "--assist-kp-nm-per-rad",
                str(args.assist_kp_nm_per_rad),
                "--assist-kd-nm-s-per-rad",
                str(args.assist_kd_nm_s_per_rad),
                "--assist-max-target-error-rad",
                str(args.assist_max_target_error_rad),
                "--assist-max-torque-nm",
                str(args.assist_max_torque_nm),
                "--assist-torque-slew-nm-s",
                str(args.assist_torque_slew_nm_s),
                "--assist-authority-rise-per-sec",
                str(args.assist_authority_rise_per_sec),
                "--assist-command-stale-timeout-sec",
                str(args.assist_command_stale_timeout_sec),
                "--assist-countdown-sec",
                str(args.assist_countdown_sec),
                "--assist-duration-sec",
                str(args.assist_duration_sec),
                "--assist-stationarity-limit-rad-s",
                str(args.assist_stationarity_limit_rad_s),
                "--assist-min-response-rad",
                str(args.assist_min_response_rad),
                "--assist-max-response-ratio",
                str(args.assist_max_response_ratio),
                "--assist-max-support-drift-rad",
                str(args.assist_max_support_drift_rad),
                "--assist-confirmation",
                args.assist_confirmation,
            )
        )
    if args.cartesian_assist:
        command.extend(
            (
                "--cartesian-assist",
                "--cartesian-assist-axis",
                args.cartesian_assist_axis,
                "--cartesian-assist-offset-m",
                str(args.cartesian_assist_offset_m),
                "--cartesian-assist-stiffness-n-per-m",
                str(args.cartesian_assist_stiffness_n_per_m),
                "--cartesian-assist-damping-n-s-per-m",
                str(args.cartesian_assist_damping_n_s_per_m),
                "--cartesian-assist-lateral-stiffness-n-per-m",
                str(args.cartesian_assist_lateral_stiffness_n_per_m),
                "--cartesian-assist-lateral-damping-n-s-per-m",
                str(args.cartesian_assist_lateral_damping_n_s_per_m),
                "--cartesian-assist-max-force-n",
                str(args.cartesian_assist_max_force_n),
                "--cartesian-assist-max-joint-torque-nm",
                str(args.cartesian_assist_max_joint_torque_nm),
                "--cartesian-assist-joint-torque-slew-nm-s",
                str(args.cartesian_assist_joint_torque_slew_nm_s),
                "--cartesian-assist-authority-rise-per-sec",
                str(args.cartesian_assist_authority_rise_per_sec),
                "--cartesian-assist-countdown-sec",
                str(args.cartesian_assist_countdown_sec),
                "--cartesian-assist-duration-sec",
                str(args.cartesian_assist_duration_sec),
                "--cartesian-assist-stationarity-limit-rad-s",
                str(args.cartesian_assist_stationarity_limit_rad_s),
                "--cartesian-assist-min-response-m",
                str(args.cartesian_assist_min_response_m),
                "--cartesian-assist-max-response-ratio",
                str(args.cartesian_assist_max_response_ratio),
                "--cartesian-assist-max-lateral-drift-m",
                str(args.cartesian_assist_max_lateral_drift_m),
                "--cartesian-assist-max-rotation-drift-rad",
                str(args.cartesian_assist_max_rotation_drift_rad),
                "--cartesian-assist-confirmation",
                args.cartesian_assist_confirmation,
            )
        )
    return command


def _motorbridge_environment() -> dict[str, str]:
    binding_root = REPO_ROOT / "hardware" / "MotorBridge" / "bindings" / "python" / "src"
    abi_path = REPO_ROOT / "hardware" / "MotorBridge" / "target" / "release" / "libmotor_abi.so"
    overrides: dict[str, str] = {}
    if binding_root.is_dir():
        current = os.environ.get("PYTHONPATH")
        overrides["PYTHONPATH"] = (
            str(binding_root) if not current else f"{binding_root}{os.pathsep}{current}"
        )
    if abi_path.is_file() and not os.environ.get("MOTORBRIDGE_LIB"):
        overrides["MOTORBRIDGE_LIB"] = str(abi_path)
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.allow_motion:
        _parser().error(
            "real B601 owner is motion-capable; rerun with explicit --allow-motion"
        )
    if args.single_joint_assist:
        if args.mode != "hold":
            _parser().error("--single-joint-assist requires --mode hold")
        if args.assist_confirmation != DEFAULT_ASSIST_CONFIRMATION:
            _parser().error(
                "--single-joint-assist requires --assist-confirmation "
                f"{DEFAULT_ASSIST_CONFIRMATION}"
            )
    elif args.assist_confirmation:
        _parser().error(
            "--assist-confirmation requires --single-joint-assist"
        )
    if args.cartesian_assist:
        if args.single_joint_assist:
            _parser().error(
                "--cartesian-assist and --single-joint-assist are mutually exclusive"
            )
        if args.mode != "hold":
            _parser().error("--cartesian-assist requires --mode hold")
        if (
            args.cartesian_assist_confirmation
            != DEFAULT_CARTESIAN_ASSIST_CONFIRMATION
        ):
            _parser().error(
                "--cartesian-assist requires --cartesian-assist-confirmation "
                f"{DEFAULT_CARTESIAN_ASSIST_CONFIRMATION}"
            )
    elif args.cartesian_assist_confirmation:
        _parser().error(
            "--cartesian-assist-confirmation requires --cartesian-assist"
        )
    config = Path(args.config).expanduser()
    source_root = Path(args.rebot_source_root).expanduser()
    missing = [
        f"config={config}" for path in (config,) if not path.is_file()
    ] + [f"rebot-source-root={source_root}" for path in (source_root,) if not path.is_dir()]
    if missing:
        _parser().error("missing local resource(s): " + ", ".join(missing))
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="b601-owner-", dir=output_root))
    command = build_probe_command(args, run_dir=run_dir)
    print("[B601] foreground gravity/damping owner starts", flush=True)
    if args.single_joint_assist:
        print(
            "[B601] one-shot relative joint assist is enabled; press A only "
            "after STARTUP ACTIVE and ASSIST READY",
            flush=True,
        )
    if args.cartesian_assist:
        print(
            "[B601] one-shot Cartesian assist is enabled; press A only after "
            "STARTUP ACTIVE and CARTESIAN ASSIST READY",
            flush=True,
        )
    print("[B601] THIS COMMAND MAY ENABLE SIX REAL JOINTS; keep E-stop ready", flush=True)
    print("[B601] first Ctrl-C requests protective hold; second Ctrl-C/SIGUSR1 requests full disable", flush=True)
    print(f"[B601] run directory: {run_dir}", flush=True)
    print(f"[B601] terminal log: {run_dir / 'terminal.log'}", flush=True)
    status = _run_foreground(
        command,
        cwd=REPO_ROOT,
        terminal_log=run_dir / "terminal.log",
        environment_overrides=_motorbridge_environment(),
    )
    print(f"[B601] launcher exit code: {status}", flush=True)
    print(f"[B601] result: {run_dir / 'b601_owner.json'}", flush=True)
    print(f"[B601] log: {run_dir / 'b601_owner.log'}", flush=True)
    print(f"[B601] terminal log: {run_dir / 'terminal.log'}", flush=True)
    print("[B601] parent shell remains active", flush=True)
    if args.pause_on_exit and sys.stdin.isatty():
        try:
            input("[B601] press Enter to return to the shell...")
        except EOFError:
            pass
    return status


if __name__ == "__main__":
    raise SystemExit(main())
