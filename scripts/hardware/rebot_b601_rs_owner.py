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

from robot_motion.rebot_b601_rs.gravity_defaults import (
    B601_DEFAULT_DRIVER_KD,
    B601_DEFAULT_GRAVITY_COULOMB_FRICTION_NM,
    B601_DEFAULT_GRAVITY_COULOMB_VELOCITY_THRESHOLD_RAD_S,
    B601_DEFAULT_GRAVITY_JOINT_BIAS_NM,
    B601_DEFAULT_GRAVITY_JOINT_OFFSETS_RAD,
    B601_DEFAULT_GRAVITY_JOINT_SCALES,
    B601_DEFAULT_GRAVITY_SCALE,
    B601_DEFAULT_KD,
    B601_DEFAULT_KP,
)
from scripts.hardware.rebot_b601_rm65_mujoco_follow import (
    _run_foreground,
)

DEFAULT_REBOT_SOURCE_ROOT = REPO_ROOT / "hardware" / "reBotArm_control_py"
DEFAULT_CONFIG = DEFAULT_REBOT_SOURCE_ROOT / "config" / "rebotarm_rs.yaml"
DEFAULT_OUTPUT_ROOT = Path("/tmp/b601-rs-commissioning")
DEFAULT_CONFIRMATION = "B601_RS_LOW_GAIN_MIT_GRAVITY_DAMPING"
DEFAULT_ASSIST_CONFIRMATION = "B601_RS_SINGLE_JOINT_POLICY_ASSIST"
DEFAULT_CARTESIAN_ASSIST_CONFIRMATION = "B601_RS_CARTESIAN_POLICY_ASSIST"
DEFAULT_TORQUE_CONSOLE_CONFIRMATION = "B601_RS_TORQUE_CONSOLE"
DEFAULT_DRIVER_KD = B601_DEFAULT_DRIVER_KD
DEFAULT_GRAVITY_SCALES = B601_DEFAULT_GRAVITY_JOINT_SCALES
DEFAULT_GRAVITY_OFFSETS = B601_DEFAULT_GRAVITY_JOINT_OFFSETS_RAD
DEFAULT_GRAVITY_BIAS = B601_DEFAULT_GRAVITY_JOINT_BIAS_NM
DEFAULT_COULOMB = B601_DEFAULT_GRAVITY_COULOMB_FRICTION_NM
DEFAULT_GRAVITY_SCALE = B601_DEFAULT_GRAVITY_SCALE
DEFAULT_GRAVITY_COULOMB_VELOCITY_THRESHOLD_RAD_S = (
    B601_DEFAULT_GRAVITY_COULOMB_VELOCITY_THRESHOLD_RAD_S
)
DEFAULT_GRAVITY_DEBUG_MODE = True
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
        "--motorbridge-root",
        type=Path,
        help="built MotorBridge checkout; useful before this worktree's submodule is built",
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
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="validate the exact child arguments/resources/ABI without opening CAN or GUI",
    )
    parser.add_argument(
        "--debug-dynamics-trace",
        action="store_true",
        help="write per-cycle feedback/torque and protective-hold diagnostics to JSONL",
    )
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument(
        "--show-joint-angles",
        action="store_true",
        help="display measured J1-J6 angles in degrees at 2 Hz without extra CAN reads",
    )
    parser.add_argument(
        "--feedback-source",
        choices=("active-report", "parameter"),
        default="active-report",
        help="active-report is the normal host-fresh source; parameter is explicit fallback",
    )
    parser.add_argument("--kp", type=float, default=B601_DEFAULT_KP)
    parser.add_argument("--kd", type=float, default=B601_DEFAULT_KD)
    parser.add_argument(
        "--torque-console",
        action="store_true",
        help="open the local browser for bounded live torque-intent commissioning",
    )
    parser.add_argument(
        "--torque-console-staged-assist",
        action="store_true",
        help="use staged relative-target assistance instead of direct torque setpoints",
    )
    parser.add_argument(
        "--torque-console-quantized-velocity-debug", action="store_true",
        help="opt-in direct J3 speed-spike withdrawal/confirmation; manual rearm only",
    )
    parser.add_argument("--torque-console-onset-capture", action="store_true", help="withdraw on sustained J3 onset; retain GUI and gravity for manual trials")
    parser.add_argument("--torque-console-onset-displacement-rad", type=float, default=0.0012)
    parser.add_argument("--torque-console-onset-confirm-samples", type=int, default=3)
    parser.add_argument("--torque-console-onset-velocity-rad-s", type=float, default=0.002)
    parser.add_argument("--torque-console-onset-settle-timeout-sec", type=float, default=1.0)
    parser.add_argument("--torque-console-onset-probe", action="store_true", help="opt-in bounded held-overlay observation before capture withdrawal")
    parser.add_argument("--torque-console-onset-probe-hold-sec", type=float, default=0.12)
    parser.add_argument("--torque-console-onset-probe-max-displacement-rad", type=float,
                        help="cumulative observation bound; defaults to onset displacement, below the hard excursion limit")
    parser.add_argument("--torque-console-onset-probe-max-trial-sec", type=float, default=10.0)
    parser.add_argument("--torque-console-onset-probe-max-command-exposure-nm-sec", type=float, default=4.0)
    parser.add_argument(
        "--torque-console-joint",
        choices=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        default="joint3",
    )
    parser.add_argument("--torque-console-max-assist-nm", type=float, default=0.30)
    parser.add_argument(
        "--torque-console-motion-profile", type=Path,
        help="opt-in J3 reference/brake/stop-point hold JSON profile; implies staged assist",
    )
    parser.add_argument("--torque-console-auto-calibration-profile", type=Path,
                        help="opt-in one-click J3 onset, low-speed and withdrawal acquisition profile")
    parser.add_argument("--torque-console-staged-target-offset-limit-rad", type=float, default=0.065)
    parser.add_argument(
        "--torque-console-staged-authority-rise-per-sec",
        type=float,
        default=3.0,
        help="authority ramp for the opt-in staged console policy intent",
    )
    parser.add_argument("--torque-console-staged-cue-torque-nm", type=float, default=0.04)
    parser.add_argument("--torque-console-staged-cue-duration-sec", type=float, default=0.30)
    parser.add_argument("--torque-console-staged-breakaway-torque-nm", type=float, default=0.20)
    parser.add_argument(
        "--torque-console-staged-breakaway-rise-slew-nm-s",
        type=float,
        default=2.0,
        help="independent rise slew for the one-shot breakaway pulse",
    )
    parser.add_argument("--torque-console-staged-breakaway-duration-sec", type=float, default=0.20)
    parser.add_argument("--torque-console-staged-breakaway-impulse-nm-sec", type=float, default=0.04)
    parser.add_argument("--torque-console-staged-motion-onset-rad", type=float, default=0.0007)
    parser.add_argument("--torque-console-staged-motion-confirm-rad", type=float, default=None)
    parser.add_argument("--torque-console-staged-motion-confirm-samples", type=int, default=1)
    parser.add_argument("--torque-console-staged-motion-onset-velocity-rad-s", type=float, default=0.025)
    parser.add_argument("--torque-console-staged-tracking-kp-nm-per-rad", type=float, default=0.6)
    parser.add_argument("--torque-console-staged-tracking-overlay-cap-nm", type=float, default=0.10)
    parser.add_argument("--torque-console-rise-slew-nm-s", type=float, default=0.15)
    parser.add_argument("--torque-console-release-slew-nm-s", type=float, default=2.0)
    parser.add_argument("--torque-console-timeout-sec", type=float, default=0.35)
    parser.add_argument(
        "--torque-console-stationarity-limit-rad-s", type=float, default=0.02
    )
    parser.add_argument("--torque-console-max-joint-velocity-rad-s", type=float, default=0.05)
    parser.add_argument("--torque-console-max-excursion-rad", type=float, default=0.15)
    parser.add_argument(
        "--torque-console-max-support-drift-rad",
        type=float,
        default=0.008726646259971648,
    )
    parser.add_argument("--torque-console-confirmation", default="")
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
        "--cartesian-assist-max-offset-m",
        type=float,
        default=0.05,
        help=(
            "explicit Cartesian assist excursion envelope; the requested "
            "offset must not exceed this value"
        ),
    )
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
        help="maximum energy-injecting Cartesian assist torque rise rate",
    )
    parser.add_argument(
        "--cartesian-assist-joint-torque-release-slew-nm-s",
        type=float,
        default=None,
        help=(
            "optional faster release/dissipative-brake rate; defaults to the "
            "rise rate"
        ),
    )
    parser.add_argument(
        "--cartesian-assist-breakaway-positive-nm",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=(0.0,) * 6,
        help=(
            "positive task-intent breakaway torque; only J2/J3 may be non-zero"
        ),
    )
    parser.add_argument(
        "--cartesian-assist-breakaway-negative-nm",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=(0.0,) * 6,
        help=(
            "negative task-intent breakaway torque magnitudes; only J2/J3 "
            "may be non-zero"
        ),
    )
    parser.add_argument(
        "--cartesian-assist-breakaway-velocity-fade-rad-s",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--cartesian-assist-breakaway-intent-deadband-nm",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--cartesian-assist-breakaway-intent-ramp-nm",
        type=float,
        default=0.04,
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
    parser.add_argument("--gravity-scale", type=float, default=DEFAULT_GRAVITY_SCALE)
    parser.add_argument(
        "--gravity-debug-mode",
        action="store_true",
        default=DEFAULT_GRAVITY_DEBUG_MODE,
        help=(
            "freeze this run's gravity/driver-Kd tuning parameters in live and "
            "protective hold; enabled by default for the operator-confirmed "
            "gravity profile"
        ),
    )
    parser.add_argument(
        "--no-gravity-debug-mode",
        dest="gravity_debug_mode",
        action="store_false",
        help=(
            "opt out of the default frozen gravity profile and restore the "
            "ordinary protective-hold damping fallback"
        ),
    )
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
        default=DEFAULT_GRAVITY_COULOMB_VELOCITY_THRESHOLD_RAD_S,
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
    if args.torque_console_auto_calibration_profile is not None:
        args.torque_console = args.torque_console_staged_assist = True
        if args.torque_console_staged_motion_confirm_rad is None:
            args.torque_console_staged_motion_confirm_rad = .0007
        if args.torque_console_staged_motion_confirm_samples == 1:
            args.torque_console_staged_motion_confirm_samples = 2
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
    if args.preflight_only:
        command.append("--preflight-only")
    if args.feedback_source == "active-report":
        command.extend(("--allow-active-report", "--auto-disable-active-report"))
    if not args.without_driver_torque_limits:
        command.append("--apply-driver-torque-limits")
    if args.gravity_debug_mode:
        command.append("--gravity-debug-mode")
    if args.debug_dynamics_trace:
        command.append("--debug-dynamics-trace")
    if args.show_joint_angles:
        command.append("--show-joint-angles")
    if args.torque_console_quantized_velocity_debug:
        command.append("--torque-console-quantized-velocity-debug")
    if args.torque_console_onset_capture:
        command.extend((
            "--torque-console-onset-capture",
            "--torque-console-onset-displacement-rad", str(args.torque_console_onset_displacement_rad),
            "--torque-console-onset-confirm-samples", str(args.torque_console_onset_confirm_samples),
            "--torque-console-onset-velocity-rad-s", str(args.torque_console_onset_velocity_rad_s),
            "--torque-console-onset-settle-timeout-sec", str(args.torque_console_onset_settle_timeout_sec),
        ))
    if args.torque_console_onset_probe:
        command.extend((
            "--torque-console-onset-probe",
            "--torque-console-onset-probe-hold-sec", str(args.torque_console_onset_probe_hold_sec),
            "--torque-console-onset-probe-max-trial-sec", str(args.torque_console_onset_probe_max_trial_sec),
            "--torque-console-onset-probe-max-command-exposure-nm-sec", str(args.torque_console_onset_probe_max_command_exposure_nm_sec),
        ))
    if args.torque_console_onset_probe_max_displacement_rad is not None:
        command.extend(("--torque-console-onset-probe-max-displacement-rad",
                        str(args.torque_console_onset_probe_max_displacement_rad)))
    if args.torque_console:
        if args.torque_console_auto_calibration_profile is not None:
            command.extend(("--torque-console-auto-calibration-profile", str(args.torque_console_auto_calibration_profile.resolve())))
        command.extend(
            (
                "--torque-console",
                "--torque-console-joint",
                args.torque_console_joint,
                "--torque-console-max-assist-nm",
                str(args.torque_console_max_assist_nm),
                "--torque-console-rise-slew-nm-s",
                str(args.torque_console_rise_slew_nm_s),
                "--torque-console-release-slew-nm-s",
                str(args.torque_console_release_slew_nm_s),
                "--torque-console-timeout-sec",
                str(args.torque_console_timeout_sec),
                "--torque-console-stationarity-limit-rad-s",
                str(args.torque_console_stationarity_limit_rad_s),
                "--torque-console-max-joint-velocity-rad-s",
                str(args.torque_console_max_joint_velocity_rad_s),
                "--torque-console-max-excursion-rad",
                str(args.torque_console_max_excursion_rad),
                "--torque-console-max-support-drift-rad",
                str(args.torque_console_max_support_drift_rad),
                "--torque-console-confirmation",
                args.torque_console_confirmation,
            )
        )
        if (
            args.torque_console_staged_assist
            or args.torque_console_motion_profile is not None
        ):
            command.extend(
                (
                    "--torque-console-staged-assist",
                    "--torque-console-staged-authority-rise-per-sec",
                    str(args.torque_console_staged_authority_rise_per_sec),
                    "--torque-console-staged-target-offset-limit-rad",
                    str(args.torque_console_staged_target_offset_limit_rad),
                    "--torque-console-staged-cue-torque-nm",
                    str(args.torque_console_staged_cue_torque_nm),
                    "--torque-console-staged-cue-duration-sec",
                    str(args.torque_console_staged_cue_duration_sec),
                    "--torque-console-staged-breakaway-torque-nm",
                    str(args.torque_console_staged_breakaway_torque_nm),
                    "--torque-console-staged-breakaway-rise-slew-nm-s",
                    str(args.torque_console_staged_breakaway_rise_slew_nm_s),
                    "--torque-console-staged-breakaway-duration-sec",
                    str(args.torque_console_staged_breakaway_duration_sec),
                    "--torque-console-staged-breakaway-impulse-nm-sec",
                    str(args.torque_console_staged_breakaway_impulse_nm_sec),
                    "--torque-console-staged-motion-onset-rad",
                    str(args.torque_console_staged_motion_onset_rad),
                    "--torque-console-staged-motion-confirm-samples",
                    str(args.torque_console_staged_motion_confirm_samples),
                    "--torque-console-staged-motion-onset-velocity-rad-s",
                    str(args.torque_console_staged_motion_onset_velocity_rad_s),
                    "--torque-console-staged-tracking-kp-nm-per-rad",
                    str(args.torque_console_staged_tracking_kp_nm_per_rad),
                    "--torque-console-staged-tracking-overlay-cap-nm",
                    str(args.torque_console_staged_tracking_overlay_cap_nm),
                )
            )
        if args.torque_console_staged_motion_confirm_rad is not None:
            command.extend(("--torque-console-staged-motion-confirm-rad", str(args.torque_console_staged_motion_confirm_rad)))
        if args.torque_console_motion_profile is not None:
            command.extend((
                "--torque-console-motion-profile",
                str(args.torque_console_motion_profile.resolve()),
            ))
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
                "--cartesian-assist-max-offset-m",
                str(args.cartesian_assist_max_offset_m),
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
                "--cartesian-assist-breakaway-positive-nm",
                *(str(value) for value in args.cartesian_assist_breakaway_positive_nm),
                "--cartesian-assist-breakaway-negative-nm",
                *(str(value) for value in args.cartesian_assist_breakaway_negative_nm),
                "--cartesian-assist-breakaway-velocity-fade-rad-s",
                str(args.cartesian_assist_breakaway_velocity_fade_rad_s),
                "--cartesian-assist-breakaway-intent-deadband-nm",
                str(args.cartesian_assist_breakaway_intent_deadband_nm),
                "--cartesian-assist-breakaway-intent-ramp-nm",
                str(args.cartesian_assist_breakaway_intent_ramp_nm),
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
        if args.cartesian_assist_joint_torque_release_slew_nm_s is not None:
            command.extend(
                (
                    "--cartesian-assist-joint-torque-release-slew-nm-s",
                    str(
                        args.cartesian_assist_joint_torque_release_slew_nm_s
                    ),
                )
            )
    return command


def _motorbridge_environment(root: Path | None = None) -> dict[str, str]:
    motorbridge_root = (
        Path(root).expanduser().resolve()
        if root is not None
        else REPO_ROOT / "hardware" / "MotorBridge"
    )
    binding_root = motorbridge_root / "bindings" / "python" / "src"
    abi_path = motorbridge_root / "target" / "release" / "libmotor_abi.so"
    if root is not None and (
        not (binding_root / "motorbridge").is_dir() or not abi_path.is_file()
    ):
        raise ValueError(
            "--motorbridge-root requires Python bindings and built libmotor_abi.so"
        )
    overrides: dict[str, str] = {}
    if binding_root.is_dir():
        current = os.environ.get("PYTHONPATH")
        overrides["PYTHONPATH"] = (
            str(binding_root) if not current else f"{binding_root}{os.pathsep}{current}"
        )
    if abi_path.is_file() and (root is not None or not os.environ.get("MOTORBRIDGE_LIB")):
        overrides["MOTORBRIDGE_LIB"] = str(abi_path)
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.torque_console_auto_calibration_profile is not None:
        args.torque_console = args.torque_console_staged_assist = True
    if args.torque_console:
        args.debug_dynamics_trace = True
    if args.torque_console_onset_probe and not args.torque_console_onset_capture:
        _parser().error("onset probe requires explicit --torque-console-onset-capture")
    if args.torque_console_onset_capture and not args.torque_console_quantized_velocity_debug:
        _parser().error("onset capture requires explicit direct J3 quantized velocity debug")
    if args.torque_console_quantized_velocity_debug and (
        not args.torque_console or args.torque_console_staged_assist
        or args.torque_console_motion_profile is not None
    ):
        _parser().error("quantized velocity debug requires direct --torque-console, not staged motion")
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
    if args.torque_console:
        if args.mode != "hold" or args.single_joint_assist or args.cartesian_assist:
            _parser().error("--torque-console requires hold and excludes other assist")
        if not args.gravity_debug_mode or args.without_driver_torque_limits:
            _parser().error(
                "--torque-console requires frozen gravity and driver torque limits"
            )
        if args.torque_console_confirmation != DEFAULT_TORQUE_CONSOLE_CONFIRMATION:
            _parser().error(
                "--torque-console requires --torque-console-confirmation "
                f"{DEFAULT_TORQUE_CONSOLE_CONFIRMATION}"
            )
        if (
            args.torque_console_staged_motion_confirm_rad is not None
            or args.torque_console_staged_motion_confirm_samples != 1
        ) and not (args.torque_console_staged_assist or args.torque_console_motion_profile is not None):
            _parser().error("motion confirmation options require staged torque console")
    elif args.torque_console_staged_assist or args.torque_console_motion_profile is not None:
        _parser().error("--torque-console-staged-assist requires --torque-console")
    elif args.torque_console_confirmation:
        _parser().error("--torque-console-confirmation requires --torque-console")
    elif args.torque_console_staged_motion_confirm_rad is not None or args.torque_console_staged_motion_confirm_samples != 1:
        _parser().error("motion confirmation options require staged torque console")
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
    try:
        motorbridge_environment = _motorbridge_environment(args.motorbridge_root)
    except ValueError as exc:
        _parser().error(str(exc))
    if args.torque_console and not (
        motorbridge_environment.get("MOTORBRIDGE_LIB")
        or os.environ.get("MOTORBRIDGE_LIB")
    ):
        _parser().error(
            "torque console needs a built MotorBridge ABI; supply --motorbridge-root"
        )
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="b601-owner-", dir=output_root))
    command = build_probe_command(args, run_dir=run_dir)
    if args.preflight_only:
        print("[B601] OFFLINE PREFLIGHT: no CAN connection or motor command", flush=True)
        status = _run_foreground(
            command, cwd=REPO_ROOT, terminal_log=run_dir / "terminal.log",
            environment_overrides=motorbridge_environment,
        )
        print(f"[B601] preflight exit code: {status}", flush=True)
        print(f"[B601] preflight result: {run_dir / 'b601_owner.json'}", flush=True)
        return status
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
    if args.torque_console_auto_calibration_profile is not None:
        print(
            "[B601] automatic calibration: select the configured GUI mode, declare no human intervention and start one batch; "
            "no manual target entry; keep the page visible and clear both directions",
            flush=True,
        )
    elif args.torque_console:
        print(
            "[B601] local torque GUI will open before CAN connection; "
            "arm assist only after STARTUP ACTIVE and TORQUE CONSOLE READY",
            flush=True,
        )
    print("[B601] THIS COMMAND MAY ENABLE SIX REAL JOINTS; keep E-stop ready", flush=True)
    print("[B601] first Ctrl-C requests protective hold; second Ctrl-C/SIGUSR1 requests full disable", flush=True)
    print(f"[B601] run directory: {run_dir}", flush=True)
    print(f"[B601] terminal log: {run_dir / 'terminal.log'}", flush=True)
    if args.debug_dynamics_trace:
        print(f"[B601] dynamics trace: {run_dir / 'b601_owner.debug.jsonl'}", flush=True)
    if args.torque_console:
        print(f"[B601] transparency annotations: {run_dir / 'b601_owner.transparency.jsonl'}", flush=True)
    status = _run_foreground(
        command,
        cwd=REPO_ROOT,
        terminal_log=run_dir / "terminal.log",
        environment_overrides=motorbridge_environment,
    )
    print(f"[B601] launcher exit code: {status}", flush=True)
    print(f"[B601] result: {run_dir / 'b601_owner.json'}", flush=True)
    print(f"[B601] log: {run_dir / 'b601_owner.log'}", flush=True)
    if args.debug_dynamics_trace:
        print(f"[B601] dynamics trace: {run_dir / 'b601_owner.debug.jsonl'}", flush=True)
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
