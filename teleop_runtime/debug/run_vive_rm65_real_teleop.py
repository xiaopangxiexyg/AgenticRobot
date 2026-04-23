from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from ..backends.hands import InspireTacHandBackend, InspireTacHandBackendConfig
from ..calibration import TeleopCalibration
from ..configs import (
    ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
    build_annotated_dual_rm65_calibration,
)
from ..diagnostics import TeleopRunState
from ..episodes import EpisodeRecorder, EpisodeRecorderConfig, compare_start_qpos_pair
from ..hand_mapping import (
    ALL_TACTILE_REGION_NAMES,
    DEFAULT_INSPIRE_HAND_JOINT_ORDER,
    TARGET_INSPIRE_HAND_Q6_ORDER,
)
from ..hand_render import (
    LEFT_HAND_RENDER_JOINT_NAMES,
    RIGHT_HAND_RENDER_JOINT_NAMES,
)
from ..hand_tcp_alignment import tracker_hand_to_rm65_tcp_rotation
from ..initial_alignment import TeleopInitialAlignment
from ..retargeters import (
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    JointsToDexHandRetargeter,
    JointsToDexHandRetargeterConfig,
    PoseToArmRetargeter,
    PoseToArmRetargeterConfig,
)
from ..rigs import BimanualRig
from ..safety import TeleopSafetyConfig
from ..schemas import TeleopCommand, merge_teleop_commands
from ..session import TeleopSession, TeleopSessionConfig
from ..sources.hand import HandQ6Source
from ..sources.misc import KeyboardButtonSource
from ..sources.pose import VivePoseSource, ViveTrackerConfig
from ..tcp import HAND_BASE_TCP_FRAME, legacy_tcp_mode_requested
from .log_utils import ChangeGate
from .run_vive_sapien_virtual import (
    RecoveryPlan,
    TerminalClutchInput,
    _active_sides_from_args,
    _advance_recovery_plan,
    _build_episode_frame,
    _build_renderer_debug_boxes,
    _build_renderer_debug_frames,
    _collision_debug_box_templates,
    _emit_collision_proxy_logs,
    _emit_home_alignment_logs,
    _emit_runtime_orientation_logs,
    _emit_tracker_axis_mapping_logs,
    _episode_root_dir,
    _log,
    _manual_calibration_confirmation_required,
    _non_negative_int,
    _alpha_float,
    _parse_active_sides,
    _parse_arm_max_angvel,
    _parse_joint_order,
    _positive_float,
    _qpos_summary,
    _runtime_home_calibration,
    _start_recovery_plan,
    _state_qpos,
    _sync_dual_arm_retargeter_from_preview,
    _hand_render_qpos,
    _wait_for_calibration_start,
)
from rm65_inspire_tac.ik_tuning import load_ik_tuning_config
from rm65_inspire_tac.paths import require_asset_path
from rm65_inspire_tac.pipeline import (
    DEFAULT_LEFT_ARM_QPOS,
    DEFAULT_RIGHT_ARM_QPOS,
    DualArmSceneProfile,
    build_dual_arm_scene_profile,
)
from rm65_inspire_tac.rendering import SapienArmHandRenderer


@dataclass(frozen=True)
class RealTeleopComponents:
    session: TeleopSession
    active_sides: tuple[str, ...]
    left_command_backend: RM65ArmBackend | None
    right_command_backend: RM65ArmBackend | None
    left_preview_backend: RM65ArmBackend
    right_preview_backend: RM65ArmBackend
    left_state_backend: RM65ArmBackend | None
    right_state_backend: RM65ArmBackend | None
    left_hand_backend: InspireTacHandBackend | None
    right_hand_backend: InspireTacHandBackend | None
    initial_alignment: TeleopInitialAlignment | None
    loaded_initial_alignment: TeleopInitialAlignment | None
    logical_arm_ip_mapping: dict[str, str]
    collision_scene_profile: DualArmSceneProfile
    dual_arm_retargeter: DualArmCollisionAwareRetargeter | None
    collision_guard_requested: bool
    collision_guard_setup_error: str | None
    renderer: SapienArmHandRenderer | None
    keyboard_source: KeyboardButtonSource
    left_retargeter: PoseToArmRetargeter
    right_retargeter: PoseToArmRetargeter
    left_hand_retargeter: JointsToDexHandRetargeter | None
    right_hand_retargeter: JointsToDexHandRetargeter | None
    left_hand_control_via_rig: bool
    right_hand_control_via_rig: bool
    joint_distance_weights: np.ndarray
    arm_state_read_executor: ThreadPoolExecutor | None
    arm_loop_step_dt_tracker: dict[str, float | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Control real dual RM65 arms from live VIVE wrist trackers, with optional "
            "additional hand q6 teleop and InspireTac hand control."
        )
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="UDP bind host for the VIVE receiver")
    parser.add_argument("--port", type=int, default=9999, help="UDP bind port for the VIVE receiver")
    parser.add_argument(
        "--active-sides",
        type=_parse_active_sides,
        default=("left", "right"),
        help="One of: left, right, both.",
    )
    parser.add_argument("--left-arm-ip", type=str, default="", help="Left RM65 IP")
    parser.add_argument("--right-arm-ip", type=str, default="", help="Right RM65 IP")
    parser.add_argument("--left-hand-ip", type=str, default="", help="Optional left InspireTac IP")
    parser.add_argument("--right-hand-ip", type=str, default="", help="Optional right InspireTac IP")
    parser.add_argument("--left-hand-port", type=str, default="", help="Optional left hand teleop serial port")
    parser.add_argument("--right-hand-port", type=str, default="", help="Optional right hand teleop serial port")
    parser.add_argument("--hand-reader-baudrate", type=int, default=115200, help="Hand teleop serial baudrate")
    parser.add_argument("--hand-reader-timeout", type=float, default=0.05, help="Hand teleop serial timeout in seconds")
    parser.add_argument("--show-hand-raw-lines", action="store_true", help="Print every raw line from hand teleop serial devices")
    parser.add_argument(
        "--hand-joint-order",
        type=_parse_joint_order,
        default=DEFAULT_INSPIRE_HAND_JOINT_ORDER,
        help=(
            "Permutation applied to hand q6 before mapping to InspireTac q6. "
            f"Default maps into InspireTac order {list(TARGET_INSPIRE_HAND_Q6_ORDER)}."
        ),
    )
    parser.add_argument(
        "--hand-invert",
        dest="hand_invert",
        action="store_true",
        help="Invert normalized q6 before sending to the InspireTac hand. Default is enabled.",
    )
    parser.add_argument(
        "--hand-no-invert",
        dest="hand_invert",
        action="store_false",
        help="Disable q6 inversion if the hand already moves in the correct direction.",
    )
    parser.set_defaults(hand_invert=True)
    parser.add_argument(
        "--hand-command-alpha",
        type=_alpha_float,
        default=1.0,
        help="EMA smoothing factor for hand q6 mapping.",
    )
    parser.add_argument(
        "--hand-send-threshold",
        type=float,
        default=0.002,
        help="Only send/update hand q6 when max delta exceeds this threshold.",
    )
    parser.add_argument(
        "--hand-telemetry-hz",
        type=_positive_float,
        default=30.0,
        help="Telemetry polling rate for each real InspireTac hand backend.",
    )
    parser.add_argument(
        "--hand-tactile-regions",
        type=str,
        default="",
        help=(
            "Optional comma-separated tactile regions to poll, for example "
            "'thumb_1,index_1,palm' or 'all'. Empty means skip tactile reads for lower latency."
        ),
    )
    parser.add_argument(
        "--print-tactile-sum-only",
        action="store_true",
        help=(
            "Suppress normal stdout logs and print only real-time left/right hand tactile sums. "
            "Teleop behavior is unchanged. Disabled by default."
        ),
    )
    parser.add_argument(
        "--max-fingertip-force",
        type=int,
        default=800,
        help="Max fingertip force passed to InspireTac hand backends on startup.",
    )
    parser.add_argument(
        "--teleop-hardware-ip",
        type=str,
        default="",
        help="Reserved teleop-hardware endpoint IP, currently unused.",
    )
    parser.add_argument(
        "--teleop-hardware-port",
        type=int,
        default=0,
        help="Reserved teleop-hardware endpoint port, currently unused.",
    )
    parser.add_argument(
        "--swap-arm-sides",
        action="store_true",
        help=(
            "Swap logical left/right assignment for the provided arm IPs. "
            "Useful when the physical arm connected as --left-arm-ip should behave as logical right."
        ),
    )
    parser.add_argument(
        "--control-mode",
        choices=["absolute", "delta_clutch"],
        default="absolute",
        help="Arm retargeting mode. Use delta_clutch with --keyboard-clutch for relative control.",
    )
    parser.add_argument(
        "--tcp-mode",
        type=str,
        default=HAND_BASE_TCP_FRAME,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--translation-scale", type=float, default=1.0, help="Scale applied to desk-frame translation")
    parser.add_argument(
        "--no-collision",
        "--disable-collision-avoidance",
        "--disable-dual-arm-collision-avoidance",
        dest="disable_dual_arm_collision_avoidance",
        action="store_true",
        help="Disable joint-space dual-arm collision avoidance and fall back to two independent arm retargeters.",
    )
    parser.add_argument(
        "--collision-avoidance-mode",
        choices=["boundary_slide", "freeze_on_contact"],
        default="boundary_slide",
        help="Behavior when the operator pushes both arms toward collision.",
    )
    parser.add_argument(
        "--collision-scene-profile",
        choices=[
            "dual_arm",
            "dual_arm_hand_proxy_open",
            "dual_arm_hand",
            "dual_arm_cylinder_table",
        ],
        default="dual_arm_hand_proxy_open",
        help=(
            "Joint-space collision scene profile used by the dual-arm solver. "
            "'dual_arm_hand_proxy_open' keeps arm IK at 12DoF while attaching fixed open-hand "
            "proxy geometry to both wrists for collision."
        ),
    )
    parser.add_argument("--hz", type=_positive_float, default=30.0, help="Target control/update rate")
    parser.add_argument(
        "--arm-max-angvel",
        type=_parse_arm_max_angvel,
        default=(0.35, 0.35, 0.35, 0.5, 0.5, 0.5),
        help="Scalar or 6 comma-separated arm joint velocity limits in rad/s.",
    )
    parser.add_argument(
        "--minimum-control-hz",
        type=_positive_float,
        default=15.0,
        help="Freeze the session when measured control rate drops below this threshold.",
    )
    parser.add_argument(
        "--status-sec",
        type=_positive_float,
        default=1.0,
        help="Heartbeat interval for teleop status logs.",
    )
    parser.add_argument(
        "--left-right-base-distance",
        type=float,
        default=ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
        help="Annotated dual-arm base distance used by the SAPIEN scene and default calibration.",
    )
    parser.add_argument("--left-tracker-name", type=str, default="tracker_1", help="Left wrist tracker device name")
    parser.add_argument("--right-tracker-name", type=str, default="tracker_2", help="Right wrist tracker device name")
    parser.add_argument(
        "--aux-tracker-name",
        type=str,
        default="tracker_3",
        help="Optional auxiliary tracker name. Pass empty string to disable.",
    )
    parser.add_argument("--headless", action="store_true", help="Disable the SAPIEN viewer window")
    parser.add_argument(
        "--skip-renderer",
        action="store_true",
        help="Skip SAPIEN renderer creation. Useful for environments without a render device.",
    )
    parser.add_argument(
        "--calibration-timeout",
        type=_positive_float,
        default=10.0,
        help="Timeout for runtime home calibration after the receiver starts.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=20,
        help="Number of valid wrist frames to average per side during runtime home calibration.",
    )
    parser.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="Start runtime home calibration immediately without waiting for a key press.",
    )
    parser.add_argument("--keyboard-clutch", action="store_true", help="Enable terminal clutch toggles: z/x/c/r, q to quit")
    parser.add_argument("--no-record", action="store_true", help="Disable JSONL recording for this session")
    parser.add_argument(
        "--no-episode-recording",
        action="store_true",
        help="Disable episode trajectory recording support for this run.",
    )
    parser.add_argument(
        "--episode-root",
        type=str,
        default="",
        help=(
            "Optional directory used to store episode_XXXXXX subdirectories. "
            "Defaults to <session_record_root>/trajectories when session recording is enabled."
        ),
    )
    parser.add_argument(
        "--record-auto-start",
        action="store_true",
        help="Start an episode automatically when the control loop begins.",
    )
    parser.add_argument(
        "--load-profile",
        type=str,
        default="",
        help="Optional calibration profile to load before runtime home calibration.",
    )
    parser.add_argument(
        "--load-initial-alignment",
        type=str,
        default="",
        help=(
            "Optional initial-alignment JSON used only for startup joint mismatch validation. "
            "Runtime home capture still uses the live real-arm pose at 'p' time."
        ),
    )
    parser.add_argument(
        "--ik-tuning-json",
        type=str,
        default="",
        help="Optional RM65 IK tuning JSON. Defaults to teleop_runtime/configs/rm65_ik_tuning.json.",
    )
    parser.add_argument(
        "--rm65-hardware-read-retries",
        type=_non_negative_int,
        default=0,
        help=(
            "Extra retry count for each Get_Current_Arm_State call. "
            "Default 0 avoids retry-induced control-rate collapse when the controller returns transient nonzero codes."
        ),
    )
    parser.add_argument(
        "--rm65-hardware-read-retry-delay-sec",
        type=_positive_float,
        default=0.02,
        help="Delay inserted between repeated Get_Current_Arm_State retries.",
    )
    parser.add_argument(
        "--arm-io-threads",
        type=_non_negative_int,
        default=2,
        help=(
            "Worker count used for left/right arm state refresh and dispatch. "
            "Set 0 or 1 to force sequential IO."
        ),
    )
    parser.add_argument(
        "--disable-arm-step-dt-scaling",
        dest="arm_step_dt_scaling",
        action="store_false",
        help="Disable real-arm step scaling based on measured send dt.",
    )
    parser.set_defaults(arm_step_dt_scaling=True)
    parser.add_argument(
        "--arm-step-dt-alpha",
        type=_alpha_float,
        default=0.25,
        help="EMA factor used when estimating real-arm send dt for step scaling.",
    )
    parser.add_argument(
        "--arm-step-dt-max-scale",
        type=_positive_float,
        default=2.0,
        help="Maximum multiplier applied to arm step limits when real send dt runs slower than target.",
    )
    parser.add_argument(
        "--arm-step-dt-stall-reset-sec",
        type=_positive_float,
        default=0.25,
        help="Reset actual-dt scaling back to the nominal control dt after longer send stalls.",
    )
    parser.add_argument(
        "--max-start-joint-error-deg",
        type=_positive_float,
        default=5.0,
        help="Abort teleop startup if any current joint differs from the initial-alignment qpos by more than this threshold.",
    )
    parser.add_argument(
        "--max-start-weighted-error-deg",
        type=_positive_float,
        default=12.0,
        help="Abort teleop startup if the weighted 6D start-state error exceeds this threshold.",
    )
    parser.add_argument(
        "--ignore-start-mismatch",
        action="store_true",
        help="Allow teleop startup to continue even when the initial-alignment start-state check fails.",
    )
    parser.add_argument(
        "--recovery-healthy-frames",
        type=int,
        default=8,
        help="Number of consecutive fault-free control frames required before auto recovery starts.",
    )
    parser.add_argument(
        "--recovery-ramp-sec",
        type=_positive_float,
        default=1.0,
        help="Absolute-control recovery ramp duration.",
    )
    parser.add_argument("--max-steps", type=_non_negative_int, default=0, help="Stop after N control steps, 0 means infinite")
    args = parser.parse_args()
    args.requested_tcp_mode = str(getattr(args, "tcp_mode", HAND_BASE_TCP_FRAME))
    args.tcp_mode = HAND_BASE_TCP_FRAME
    active_sides = _active_sides_from_args(args)
    logical_ip_by_side: dict[str, tuple[str, str]] = {}
    inactive_config: list[str] = []
    for source_name, (logical_side, arm_ip) in _resolve_logical_side_assignments(args).items():
        if not arm_ip:
            continue
        if logical_side not in active_sides:
            inactive_config.append(f"{source_name}->{logical_side}")
            continue
        logical_ip_by_side[logical_side] = (source_name, arm_ip)
    missing = [side for side in active_sides if side not in logical_ip_by_side]
    if missing:
        raise SystemExit(
            "Real teleop requires an RM65 IP for every active logical side; "
            f"missing={missing} active_sides={list(active_sides)} "
            f"swap_arm_sides={bool(args.swap_arm_sides)}."
        )
    if inactive_config:
        raise SystemExit(
            "Inactive logical side received an arm IP; remove it or include that side "
            f"in --active-sides. inactive={inactive_config}"
        )
    return args


def _resolve_logical_side_assignments(args: argparse.Namespace) -> dict[str, tuple[str, str]]:
    return {
        "left_arm_ip": ("right" if args.swap_arm_sides else "left", args.left_arm_ip),
        "right_arm_ip": ("left" if args.swap_arm_sides else "right", args.right_arm_ip),
    }


def _parse_hand_tactile_regions(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (tuple, list)):
        parts = [str(item).strip() for item in value if str(item).strip()]
    else:
        parts = [part.strip() for part in str(value).split(",") if part.strip()]
    lowered = {part.lower() for part in parts}
    if "all" in lowered:
        return ALL_TACTILE_REGION_NAMES
    return tuple(parts)


def _alignment_qpos_pair(alignment: TeleopInitialAlignment | None) -> dict[str, np.ndarray] | None:
    if alignment is None:
        return None
    qpos_pair: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        side_state = alignment.get_side(side)
        if side_state is None or side_state.arm_qpos is None:
            return None
        qpos_pair[side] = np.asarray(side_state.arm_qpos, dtype=float).reshape(6).copy()
    return qpos_pair


def _start_state_summary(checks: dict[str, object]) -> str:
    parts: list[str] = []
    for side in ("left", "right"):
        if side not in checks:
            continue
        check = checks[side]
        parts.append(
            f"{side}: max_abs={check.max_abs_joint_error_deg:.2f}deg "
            f"weighted={check.weighted_joint_error_deg:.2f}deg "
            f"within_limits={check.within_limits}"
        )
    return "; ".join(parts)


def _build_episode_metadata(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    start_reason: str,
) -> dict[str, object]:
    calibration = components.session.calibration
    assert calibration is not None
    session_record_root = (
        None if components.session.recorder is None else str(components.session.recorder.root_dir)
    )
    return {
        "source_entrypoint": "teleop_runtime.debug.run_vive_rm65_real_teleop",
        "start_reason": start_reason,
        "control_mode": args.control_mode,
        "arm_mode": "real",
        "active_sides": list(components.active_sides),
        "runtime_arm_state_source": "last_commanded_action",
        "hz": float(args.hz),
        "translation_scale": float(args.translation_scale),
        "left_right_base_distance_m": float(args.left_right_base_distance),
        "collision_scene_profile": args.collision_scene_profile,
        "collision_avoidance_mode": args.collision_avoidance_mode,
        "collision_guard_requested": bool(components.collision_guard_requested),
        "collision_guard_enabled": bool(components.dual_arm_retargeter is not None),
        "session_record_root": session_record_root,
        "load_profile": args.load_profile or None,
        "load_initial_alignment": args.load_initial_alignment or None,
        "logical_arm_ip_mapping": dict(components.logical_arm_ip_mapping),
        "left_hand_ip": args.left_hand_ip or None,
        "right_hand_ip": args.right_hand_ip or None,
        "left_hand_port": getattr(args, "left_hand_port", "") or None,
        "right_hand_port": getattr(args, "right_hand_port", "") or None,
        "hand_reader_baudrate": int(getattr(args, "hand_reader_baudrate", 115200)),
        "hand_joint_order": list(getattr(args, "hand_joint_order", DEFAULT_INSPIRE_HAND_JOINT_ORDER)),
        "hand_invert": bool(getattr(args, "hand_invert", True)),
        "hand_command_alpha": float(getattr(args, "hand_command_alpha", 1.0)),
        "hand_send_threshold": float(getattr(args, "hand_send_threshold", 0.002)),
        "hand_telemetry_hz": float(getattr(args, "hand_telemetry_hz", 30.0)),
        "hand_tactile_regions": list(_parse_hand_tactile_regions(getattr(args, "hand_tactile_regions", ()))),
        "teleop_hardware_ip": args.teleop_hardware_ip or None,
        "teleop_hardware_port": int(args.teleop_hardware_port) if args.teleop_hardware_port else None,
        "swap_arm_sides": bool(args.swap_arm_sides),
        "calibration_snapshot": calibration.to_dict(),
        "initial_alignment_snapshot": (
            None
            if components.loaded_initial_alignment is None
            else components.loaded_initial_alignment.to_dict()
        ),
    }


def _start_episode_recording(
    recorder: EpisodeRecorder,
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    now: float,
    start_reason: str,
) -> Path | None:
    episode_dir = recorder.start_episode(
        metadata=_build_episode_metadata(
            args,
            components,
            start_reason=start_reason,
        ),
        timestamp=now,
    )
    if episode_dir is not None:
        _log("Episode", f"recording started -> {episode_dir}", now=now)
    return episode_dir


def _stop_episode_recording(
    recorder: EpisodeRecorder,
    *,
    now: float,
    end_reason: str,
    summary: dict[str, object] | None = None,
) -> Path | None:
    episode_dir = recorder.stop_episode(
        end_reason=end_reason,
        summary=summary,
        timestamp=now,
    )
    if episode_dir is not None:
        _log("Episode", f"recording stopped reason={end_reason} -> {episode_dir}", now=now)
    return episode_dir


def _build_components(args: argparse.Namespace) -> RealTeleopComponents:
    active_sides = _active_sides_from_args(args)
    logical_ip_by_side: dict[str, tuple[str, str]] = {}
    inactive_arm_config: list[str] = []
    for source_name, (logical_side, arm_ip) in _resolve_logical_side_assignments(args).items():
        if not arm_ip:
            continue
        if logical_side not in active_sides:
            inactive_arm_config.append(f"{source_name}->{logical_side}")
            continue
        logical_ip_by_side[logical_side] = (source_name, arm_ip)
    if inactive_arm_config:
        raise ValueError(
            "Inactive logical side received an arm IP; remove it or include that side "
            f"in --active-sides. inactive={inactive_arm_config}"
        )
    missing_arm_sides = [side for side in active_sides if side not in logical_ip_by_side]
    if missing_arm_sides:
        raise ValueError(
            "Real teleop requires an RM65 IP for every active logical side; "
            f"missing={missing_arm_sides} active_sides={list(active_sides)} "
            f"swap_arm_sides={bool(args.swap_arm_sides)}."
        )
    for side in ("left", "right"):
        if side in active_sides:
            continue
        present = [
            f"--{side}-hand-ip" if str(getattr(args, f"{side}_hand_ip", "") or "") else "",
            f"--{side}-hand-port" if str(getattr(args, f"{side}_hand_port", "") or "") else "",
        ]
        present = [value for value in present if value]
        if present:
            raise ValueError(
                f"Inactive side '{side}' received hand configuration {present}; "
                "remove those flags or include the side in --active-sides."
            )
    control_dt = 1.0 / float(args.hz)
    arm_io_threads = max(1, int(args.arm_io_threads))
    collision_guard_requested = (
        set(active_sides) == {"left", "right"}
        and not args.disable_dual_arm_collision_avoidance
    )
    arm_max_angvel = np.asarray(
        getattr(args, "arm_max_angvel", (0.35, 0.35, 0.35, 0.5, 0.5, 0.5)),
        dtype=float,
    ).reshape(6)
    realistic_joint_step = tuple((arm_max_angvel * control_dt).tolist())
    collision_scene_profile = build_dual_arm_scene_profile(args.collision_scene_profile)
    ik_tuning = load_ik_tuning_config(args.ik_tuning_json or None)
    loaded_initial_alignment = (
        TeleopInitialAlignment.load_json(args.load_initial_alignment)
        if args.load_initial_alignment
        else None
    )
    if args.load_profile:
        calibration = TeleopCalibration.load_json(args.load_profile)
    else:
        calibration = build_annotated_dual_rm65_calibration(
            left_right_base_distance=args.left_right_base_distance
        )
    calibration.metadata.update(
        {
            "source_entrypoint": "teleop_runtime.debug.run_vive_rm65_real_teleop",
            "left_tracker_name": args.left_tracker_name,
            "right_tracker_name": args.right_tracker_name,
            "aux_tracker_name": args.aux_tracker_name,
            "active_sides": list(active_sides),
            "arm_mode": "real",
            "left_arm_ip": args.left_arm_ip,
            "right_arm_ip": args.right_arm_ip,
            "left_hand_ip": args.left_hand_ip,
            "right_hand_ip": args.right_hand_ip,
            "left_hand_port": getattr(args, "left_hand_port", ""),
            "right_hand_port": getattr(args, "right_hand_port", ""),
            "hand_reader_baudrate": int(getattr(args, "hand_reader_baudrate", 115200)),
            "hand_joint_order": list(getattr(args, "hand_joint_order", DEFAULT_INSPIRE_HAND_JOINT_ORDER)),
            "hand_invert": bool(getattr(args, "hand_invert", True)),
            "hand_command_alpha": float(getattr(args, "hand_command_alpha", 1.0)),
            "hand_send_threshold": float(getattr(args, "hand_send_threshold", 0.002)),
            "hand_telemetry_hz": float(getattr(args, "hand_telemetry_hz", 30.0)),
            "hand_tactile_regions": list(_parse_hand_tactile_regions(getattr(args, "hand_tactile_regions", ()))),
            "max_fingertip_force": int(getattr(args, "max_fingertip_force", 800)),
            "teleop_hardware_ip": args.teleop_hardware_ip,
            "teleop_hardware_port": int(args.teleop_hardware_port),
            "swap_arm_sides": bool(args.swap_arm_sides),
            "control_dt": control_dt,
            "arm_io_threads": int(arm_io_threads),
            "arm_step_dt_scaling": bool(args.arm_step_dt_scaling),
            "arm_step_dt_alpha": float(args.arm_step_dt_alpha),
            "arm_step_dt_max_scale": float(args.arm_step_dt_max_scale),
            "arm_step_dt_stall_reset_sec": float(args.arm_step_dt_stall_reset_sec),
            "dual_arm_collision_avoidance_requested": collision_guard_requested,
            "collision_avoidance_mode": args.collision_avoidance_mode,
            "collision_scene_profile": args.collision_scene_profile,
            "ik_tuning_source": ik_tuning.source_path,
            "ik_objective_mode": ik_tuning.ik_objective_mode,
            "initial_alignment_source": args.load_initial_alignment,
        }
    )

    keyboard_source = KeyboardButtonSource(name="terminal_clutch")
    default_clutch_pressed = args.control_mode == "delta_clutch" and not args.keyboard_clutch
    keyboard_source.set_button(
        "left_clutch_button",
        default_clutch_pressed,
        side="left",
        critical=False,
    )
    keyboard_source.set_button(
        "right_clutch_button",
        default_clutch_pressed,
        side="right",
        critical=False,
    )

    tracker_configs: list[ViveTrackerConfig] = []
    if "left" in active_sides:
        tracker_configs.append(
            ViveTrackerConfig(
                name=args.left_tracker_name,
                side="left",
                role="left_wrist",
                channel_name="left_wrist_pose",
                legacy_index=0,
            )
        )
    if "right" in active_sides:
        tracker_configs.append(
            ViveTrackerConfig(
                name=args.right_tracker_name,
                side="right",
                role="right_wrist",
                channel_name="right_wrist_pose",
                legacy_index=1,
            )
        )
    if args.aux_tracker_name:
        tracker_configs.append(
            ViveTrackerConfig(
                name=args.aux_tracker_name,
                side=None,
                role="auxiliary",
                channel_name="auxiliary_tracker_pose",
                legacy_index=2,
                critical=False,
            )
        )

    left_retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="left",
            input_channel_name="left_wrist_pose",
            effector_name="left_arm",
            control_mode=args.control_mode,
            translation_scale=args.translation_scale,
            input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
        )
    )
    right_retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="right",
            input_channel_name="right_wrist_pose",
            effector_name="right_arm",
            control_mode=args.control_mode,
            translation_scale=args.translation_scale,
            input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
        )
    )

    left_seed_qpos = DEFAULT_LEFT_ARM_QPOS
    right_seed_qpos = DEFAULT_RIGHT_ARM_QPOS
    alignment_qpos_pair = _alignment_qpos_pair(loaded_initial_alignment)
    if alignment_qpos_pair is not None:
        left_seed_qpos = alignment_qpos_pair["left"]
        right_seed_qpos = alignment_qpos_pair["right"]

    dual_arm_retargeter = None
    collision_guard_setup_error = None
    retargeters = []
    if collision_guard_requested:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="no QP solver found on your system.*",
                    category=UserWarning,
                )
                dual_arm_retargeter = DualArmCollisionAwareRetargeter(
                    DualArmCollisionAwareRetargeterConfig(
                        control_mode=args.control_mode,
                        translation_scale=args.translation_scale,
                        scene_profile=collision_scene_profile.name,
                        contact_mode=args.collision_avoidance_mode,
                        solver_frequency=args.hz,
                        enforce_realistic_joint_step=True,
                        left_execution_max_joint_step_rad=realistic_joint_step,
                        right_execution_max_joint_step_rad=realistic_joint_step,
                        left_initial_qpos=left_seed_qpos,
                        right_initial_qpos=right_seed_qpos,
                        ik_tuning=ik_tuning,
                    ),
                    left_pose_retargeter=left_retargeter,
                    right_pose_retargeter=right_retargeter,
                )
        except (ModuleNotFoundError, RuntimeError, ValueError) as exc:
            collision_guard_setup_error = f"{type(exc).__name__}: {exc}"
        else:
            retargeters.append(dual_arm_retargeter)
    if dual_arm_retargeter is None:
        side_retargeters = {"left": left_retargeter, "right": right_retargeter}
        retargeters.extend(side_retargeters[side] for side in active_sides)

    calibration.metadata.update(
        {
            "dual_arm_collision_avoidance_enabled": dual_arm_retargeter is not None,
            "collision_avoidance_setup_error": collision_guard_setup_error,
        }
    )

    left_hand_port = getattr(args, "left_hand_port", "") or ""
    right_hand_port = getattr(args, "right_hand_port", "") or ""
    hand_source = None
    if left_hand_port or right_hand_port:
        hand_source = HandQ6Source.from_bimanual(
            left_port=(left_hand_port or None) if "left" in active_sides else None,
            right_port=(right_hand_port or None) if "right" in active_sides else None,
            baudrate=int(getattr(args, "hand_reader_baudrate", 115200)),
            timeout=float(getattr(args, "hand_reader_timeout", 0.05)),
            show_raw_lines=bool(getattr(args, "show_hand_raw_lines", False)),
            critical=False,
            strict_expected=False,
        )

    left_hand_retargeter = None
    right_hand_retargeter = None
    if "left" in active_sides and left_hand_port:
        left_hand_retargeter = JointsToDexHandRetargeter(
            JointsToDexHandRetargeterConfig(
                side="left",
                input_channel_name="left_hand_q6",
                effector_name="left_hand",
                joint_order=tuple(getattr(args, "hand_joint_order", DEFAULT_INSPIRE_HAND_JOINT_ORDER)),
                invert=bool(getattr(args, "hand_invert", True)),
                alpha=float(getattr(args, "hand_command_alpha", 1.0)),
            )
        )
    if "right" in active_sides and right_hand_port:
        right_hand_retargeter = JointsToDexHandRetargeter(
            JointsToDexHandRetargeterConfig(
                side="right",
                input_channel_name="right_hand_q6",
                effector_name="right_hand",
                joint_order=tuple(getattr(args, "hand_joint_order", DEFAULT_INSPIRE_HAND_JOINT_ORDER)),
                invert=bool(getattr(args, "hand_invert", True)),
                alpha=float(getattr(args, "hand_command_alpha", 1.0)),
            )
        )
    if left_hand_retargeter is not None:
        retargeters.append(left_hand_retargeter)
    if right_hand_retargeter is not None:
        retargeters.append(right_hand_retargeter)

    hand_tactile_regions = _parse_hand_tactile_regions(getattr(args, "hand_tactile_regions", ()))
    hand_telemetry_hz = float(getattr(args, "hand_telemetry_hz", max(float(args.hz), 10.0)))

    left_hand_backend = None
    if "left" in active_sides and (left_hand_port or args.left_hand_ip):
        left_hand_backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip=args.left_hand_ip or "0.0.0.0",
                side="left",
                connect_hardware=bool(args.left_hand_ip),
                max_fingertip_force=int(getattr(args, "max_fingertip_force", 800)),
                send_threshold=float(getattr(args, "hand_send_threshold", 0.002)),
                enable_telemetry=bool(args.left_hand_ip),
                telemetry_hz=hand_telemetry_hz,
                tactile_regions=hand_tactile_regions,
            )
        )
    right_hand_backend = None
    if "right" in active_sides and (right_hand_port or args.right_hand_ip):
        right_hand_backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip=args.right_hand_ip or "0.0.0.0",
                side="right",
                connect_hardware=bool(args.right_hand_ip),
                max_fingertip_force=int(getattr(args, "max_fingertip_force", 800)),
                send_threshold=float(getattr(args, "hand_send_threshold", 0.002)),
                enable_telemetry=bool(args.right_hand_ip),
                telemetry_hz=hand_telemetry_hz,
                tactile_regions=hand_tactile_regions,
            )
        )

    session_sources = [
        VivePoseSource(
            tracker_configs=tracker_configs,
            host=args.host,
            port=args.port,
        ),
        keyboard_source,
    ]
    if hand_source is not None:
        session_sources.append(hand_source)

    arm_urdf_path = require_asset_path("rm65_arm.urdf")
    side_assignments = _resolve_logical_side_assignments(args)
    logical_arm_ip_mapping: dict[str, str] = {}
    left_command_backend = None
    right_command_backend = None
    left_state_backend = None
    right_state_backend = None
    left_preview_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            default_tcp_frame=args.tcp_mode,
            arm_urdf_path=arm_urdf_path,
            connect_hardware=False,
            control_dt=control_dt,
            max_angvel=tuple(arm_max_angvel.tolist()),
            dry_run_fast=True,
            initial_qpos=tuple(np.asarray(left_seed_qpos, dtype=float).reshape(6).tolist()),
            use_actual_send_dt_scaling=False,
            ik_tuning=ik_tuning,
        )
    )
    right_preview_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="right",
            default_tcp_frame=args.tcp_mode,
            arm_urdf_path=arm_urdf_path,
            connect_hardware=False,
            control_dt=control_dt,
            max_angvel=tuple(arm_max_angvel.tolist()),
            dry_run_fast=True,
            initial_qpos=tuple(np.asarray(right_seed_qpos, dtype=float).reshape(6).tolist()),
            use_actual_send_dt_scaling=False,
            ik_tuning=ik_tuning,
        )
    )
    for _source_name, (logical_side, arm_ip) in side_assignments.items():
        if not arm_ip or logical_side not in active_sides:
            continue
        command_backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip=arm_ip,
                side=logical_side,
                default_tcp_frame=args.tcp_mode,
                arm_urdf_path=arm_urdf_path,
                connect_hardware=True,
                control_dt=control_dt,
                max_angvel=tuple(arm_max_angvel.tolist()),
                dry_run_fast=False,
                hardware_read_retries=int(args.rm65_hardware_read_retries),
                hardware_read_retry_delay_sec=float(args.rm65_hardware_read_retry_delay_sec),
                initial_qpos=None,
                use_actual_send_dt_scaling=bool(args.arm_step_dt_scaling),
                actual_send_dt_alpha=float(args.arm_step_dt_alpha),
                actual_send_dt_min_scale=1.0,
                actual_send_dt_max_scale=float(args.arm_step_dt_max_scale),
                actual_send_dt_stall_reset_sec=float(args.arm_step_dt_stall_reset_sec),
                background_streaming_enabled=True,
                background_streaming_hz=float(args.hz),
                ik_tuning=ik_tuning,
            )
        )
        state_backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip=arm_ip,
                side=logical_side,
                default_tcp_frame=args.tcp_mode,
                arm_urdf_path=arm_urdf_path,
                connect_hardware=True,
                control_dt=control_dt,
                max_angvel=tuple(arm_max_angvel.tolist()),
                dry_run_fast=False,
                hardware_read_retries=int(args.rm65_hardware_read_retries),
                hardware_read_retry_delay_sec=float(args.rm65_hardware_read_retry_delay_sec),
                initial_qpos=None,
                use_actual_send_dt_scaling=False,
                ik_tuning=ik_tuning,
            )
        )
        logical_arm_ip_mapping[logical_side] = arm_ip
        if logical_side == "left":
            left_command_backend = command_backend
            left_state_backend = state_backend
        else:
            right_command_backend = command_backend
            right_state_backend = state_backend
    for side in active_sides:
        command_backend = left_command_backend if side == "left" else right_command_backend
        state_backend = left_state_backend if side == "left" else right_state_backend
        if command_backend is None or state_backend is None:
            raise RuntimeError(f"Failed to resolve logical {side} arm assignment.")

    left_hand_control_via_rig = left_hand_retargeter is not None and left_hand_backend is not None
    right_hand_control_via_rig = right_hand_retargeter is not None and right_hand_backend is not None
    rig_backends = {}
    if "left" in active_sides and left_command_backend is not None:
        rig_backends["left_arm"] = left_command_backend
    if "right" in active_sides and right_command_backend is not None:
        rig_backends["right_arm"] = right_command_backend
    if left_hand_control_via_rig:
        rig_backends["left_hand"] = left_hand_backend
    if right_hand_control_via_rig:
        rig_backends["right_hand"] = right_hand_backend

    rig = BimanualRig(
        backends=rig_backends,
        parallel_dispatch=arm_io_threads > 1,
        parallel_workers=max(arm_io_threads, len(rig_backends)),
    )
    session = TeleopSession(
        config=TeleopSessionConfig(
            mode="vive_rm65_real_teleop",
            tcp_mode=args.tcp_mode,
            record_enabled=not args.no_record,
            recovery_healthy_frame_count=max(1, int(args.recovery_healthy_frames)),
            non_freezing_fault_codes=("low_control_rate",),
        ),
        sources=session_sources,
        retargeters=retargeters,
        rig=rig,
        calibration=calibration,
        safety_config=TeleopSafetyConfig(
            freeze_on_fault=True,
            required_channels=tuple(f"{side}_wrist_pose" for side in active_sides),
            invalid_frame_limit=3,
            minimum_control_hz=float(args.minimum_control_hz),
            target_control_hz=float(args.hz),
        ),
    )

    renderer = None
    if not args.skip_renderer:
        try:
            renderer = SapienArmHandRenderer(
                left_urdf_path=require_asset_path("rm65_inspire_tac_left.urdf"),
                left_retargeting_joint_names=LEFT_HAND_RENDER_JOINT_NAMES,
                right_urdf_path=require_asset_path("rm65_inspire_tac_right.urdf"),
                right_retargeting_joint_names=RIGHT_HAND_RENDER_JOINT_NAMES,
                left_right_base_distance=args.left_right_base_distance,
                show_viewer=not args.headless,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to create the SAPIEN renderer. Check that the current environment "
                "has a usable Vulkan/OpenGL render device, or rerun with --skip-renderer "
                "to validate the real teleop loop without visualization."
            ) from exc
        renderer.reserve_debug_frames(
            (
                "desk",
                "left_arm_base",
                "right_arm_base",
                "left_tracker",
                "right_tracker",
                "left_hand_home",
                "right_hand_home",
                "left_robot_home",
                "right_robot_home",
                "left_tcp_actual",
                "right_tcp_actual",
                "left_tcp_target",
                "right_tcp_target",
            )
        )
        renderer.reserve_debug_boxes(_collision_debug_box_templates(collision_scene_profile))

    if dual_arm_retargeter is not None:
        dual_arm_retargeter.set_current_qpos_pair(
            left_qpos=left_seed_qpos,
            right_qpos=right_seed_qpos,
        )

    arm_state_read_executor = (
        ThreadPoolExecutor(
            max_workers=min(arm_io_threads, len(active_sides)),
            thread_name_prefix="rm65_state",
        )
        if arm_io_threads > 1 and len(active_sides) > 1
        else None
    )

    return RealTeleopComponents(
        session=session,
        active_sides=active_sides,
        left_command_backend=left_command_backend,
        right_command_backend=right_command_backend,
        left_preview_backend=left_preview_backend,
        right_preview_backend=right_preview_backend,
        left_state_backend=left_state_backend,
        right_state_backend=right_state_backend,
        left_hand_backend=left_hand_backend,
        right_hand_backend=right_hand_backend,
        initial_alignment=None,
        loaded_initial_alignment=loaded_initial_alignment,
        logical_arm_ip_mapping=logical_arm_ip_mapping,
        collision_scene_profile=collision_scene_profile,
        dual_arm_retargeter=dual_arm_retargeter,
        collision_guard_requested=collision_guard_requested,
        collision_guard_setup_error=collision_guard_setup_error,
        renderer=renderer,
        keyboard_source=keyboard_source,
        left_retargeter=left_retargeter,
        right_retargeter=right_retargeter,
        left_hand_retargeter=left_hand_retargeter,
        right_hand_retargeter=right_hand_retargeter,
        left_hand_control_via_rig=left_hand_control_via_rig,
        right_hand_control_via_rig=right_hand_control_via_rig,
        joint_distance_weights=np.asarray(
            ik_tuning.branch_normalization.joint_distance_weights,
            dtype=float,
        ),
        arm_state_read_executor=arm_state_read_executor,
        arm_loop_step_dt_tracker={
            "last_loop_time": None,
            "smoothed_loop_dt": control_dt,
        },
    )


def _run_initial_alignment_start_check(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> str:
    recorded_qpos_pair = _alignment_qpos_pair(components.loaded_initial_alignment)
    if recorded_qpos_pair is None:
        if components.loaded_initial_alignment is not None:
            _log(
                "StartCheck",
                "loaded initial alignment does not contain both left/right arm_qpos entries; startup joint check skipped.",
            )
        return "skipped"

    current_qpos_pair = {
        "left": _state_qpos(components.left_preview_backend, DEFAULT_LEFT_ARM_QPOS),
        "right": _state_qpos(components.right_preview_backend, DEFAULT_RIGHT_ARM_QPOS),
    }
    inactive_sides = set(("left", "right")) - set(components.active_sides)
    for side in inactive_sides:
        current_qpos_pair[side] = recorded_qpos_pair[side]
    checks = compare_start_qpos_pair(
        current_qpos_pair=current_qpos_pair,
        recorded_qpos_pair=recorded_qpos_pair,
        joint_distance_weights=components.joint_distance_weights,
        max_abs_threshold_deg=float(args.max_start_joint_error_deg),
        weighted_threshold_deg=float(args.max_start_weighted_error_deg),
    )
    checks = {side: checks[side] for side in components.active_sides}
    _log("StartCheck", f"initial-alignment check: {_start_state_summary(checks)}")
    for side in components.active_sides:
        check = checks[side]
        _log(
            "StartCheck",
            f"{side} current_q={_qpos_summary(check.current_qpos)} "
            f"alignment_q={_qpos_summary(check.recorded_qpos)} "
            f"abs_joint_error_deg={_qpos_summary(check.abs_joint_error_deg)}",
        )
    start_ok = all(check.within_limits for check in checks.values())
    if not start_ok and not args.ignore_start_mismatch:
        raise RuntimeError(
            "Real teleop aborted because the current start-state joint error exceeds the configured threshold."
        )
    if not start_ok:
        return "ignored_mismatch"
    return "passed"


def _sync_preview_from_real_state_real(
    components: RealTeleopComponents,
    *,
    now: float | None = None,
    required: bool,
) -> dict[str, bool]:
    synced = {side: False for side in components.active_sides}
    backends = (
        ("left", components.left_state_backend, components.left_preview_backend),
        ("right", components.right_state_backend, components.right_preview_backend),
    )
    active = [
        (side, state_backend, preview_backend)
        for side, state_backend, preview_backend in backends
        if side in components.active_sides and state_backend is not None
    ]
    parallel_submit_failed = False
    if components.arm_state_read_executor is not None and len(active) > 1:
        try:
            future_map = {
                components.arm_state_read_executor.submit(state_backend.refresh_state_from_hardware): (
                    side,
                    preview_backend,
                )
                for side, state_backend, preview_backend in active
            }
        except RuntimeError:
            parallel_submit_failed = True
        else:
            for future, (side, preview_backend) in future_map.items():
                state = future.result()
                qpos = state.get("current_qpos")
                if qpos is None:
                    if required:
                        raise RuntimeError(
                            f"{side}_arm real-state sync failed: no current_qpos available from hardware."
                        )
                    continue
                preview_backend.set_local_qpos(qpos)
                synced[side] = True
    if components.arm_state_read_executor is None or len(active) <= 1 or parallel_submit_failed:
        for side, state_backend, preview_backend in backends:
            if side not in components.active_sides or state_backend is None:
                continue
            state = state_backend.refresh_state_from_hardware()
            qpos = state.get("current_qpos")
            if qpos is None:
                if required:
                    raise RuntimeError(
                        f"{side}_arm real-state sync failed: no current_qpos available from hardware."
                    )
                continue
            preview_backend.set_local_qpos(qpos)
            synced[side] = True
    _sync_dual_arm_retargeter_from_preview(components)
    return synced


def _sync_real_execution_step_limits(
    components: RealTeleopComponents,
    *,
    now: float | None = None,
) -> None:
    if components.dual_arm_retargeter is None:
        return
    if not components.dual_arm_retargeter.config.enforce_realistic_joint_step:
        return
    if components.left_command_backend is None or components.right_command_backend is None:
        return
    tracker = components.arm_loop_step_dt_tracker
    current_time = time.time() if now is None else float(now)
    command_config = components.left_command_backend.config
    base_dt = float(command_config.control_dt)
    smoothed_loop_dt = float(tracker.get("smoothed_loop_dt", base_dt) or base_dt)
    last_loop_time = tracker.get("last_loop_time")
    if last_loop_time is None:
        smoothed_loop_dt = base_dt
    else:
        raw_loop_dt = max(0.0, current_time - float(last_loop_time))
        if raw_loop_dt > float(command_config.actual_send_dt_stall_reset_sec):
            smoothed_loop_dt = base_dt
        else:
            alpha = float(command_config.actual_send_dt_alpha)
            smoothed_loop_dt = alpha * raw_loop_dt + (1.0 - alpha) * smoothed_loop_dt
    tracker["last_loop_time"] = current_time
    tracker["smoothed_loop_dt"] = smoothed_loop_dt
    effective_loop_dt = float(
        np.clip(
            smoothed_loop_dt,
            base_dt * float(command_config.actual_send_dt_min_scale),
            base_dt * float(command_config.actual_send_dt_max_scale),
        )
    )
    components.dual_arm_retargeter.set_execution_max_joint_step(
        left_step_rad=np.asarray(components.left_command_backend.config.max_angvel, dtype=float)
        * effective_loop_dt,
        right_step_rad=np.asarray(components.right_command_backend.config.max_angvel, dtype=float)
        * effective_loop_dt,
    )


def _apply_active_arm_command_to_preview_real(
    components: RealTeleopComponents,
    active_command,
) -> None:
    if active_command is None:
        return
    left_command = active_command.commands.get("left_arm")
    if left_command is not None:
        components.left_preview_backend.send_command(left_command)
    right_command = active_command.commands.get("right_arm")
    if right_command is not None:
        components.right_preview_backend.send_command(right_command)


def _command_backend_for_side(
    components: RealTeleopComponents,
    side: str,
) -> RM65ArmBackend | None:
    return components.left_command_backend if side == "left" else components.right_command_backend


def _active_command_backend(components: RealTeleopComponents) -> RM65ArmBackend | None:
    for side in components.active_sides:
        backend = _command_backend_for_side(components, side)
        if backend is not None:
            return backend
    return None


def _extract_hand_command_from_command(command: TeleopCommand | None) -> TeleopCommand | None:
    if command is None:
        return None
    hand_command = TeleopCommand(timestamp=command.timestamp, metadata=dict(command.metadata))
    for effector_name in ("left_hand", "right_hand"):
        effector_command = command.commands.get(effector_name)
        if effector_command is not None:
            hand_command.add_command(effector_command.copy())
    if not hand_command.commands:
        return None
    return hand_command


def _build_runtime_hand_command_for_recovery(
    components: RealTeleopComponents,
    observation,
) -> TeleopCommand | None:
    if observation is None:
        return None
    hand_commands: list[TeleopCommand] = []
    if components.left_hand_retargeter is not None:
        hand_commands.append(
            components.left_hand_retargeter.retarget(
                observation,
                components.session.calibration,
            )
        )
    if components.right_hand_retargeter is not None:
        hand_commands.append(
            components.right_hand_retargeter.retarget(
                observation,
                components.session.calibration,
            )
        )
    if not hand_commands:
        return None
    merged = merge_teleop_commands(hand_commands)
    if not merged.commands:
        return None
    return merged


def _merge_runtime_hand_command(
    active_command: TeleopCommand | None,
    hand_command: TeleopCommand | None,
) -> TeleopCommand | None:
    if hand_command is None:
        return active_command
    if active_command is None:
        return None
    return merge_teleop_commands((active_command, hand_command))


def _hand_tactile_sum(backend: InspireTacHandBackend | None) -> float | None:
    if backend is None:
        return None
    state = backend.get_state()
    tactile = state.get("tactile")
    if tactile is not None:
        array = np.asarray(tactile, dtype=float).reshape(-1)
        if array.size > 0 and np.all(np.isfinite(array)):
            return float(np.sum(array))
    tactile_dict = state.get("tactile_dict")
    if isinstance(tactile_dict, dict) and tactile_dict:
        total = 0.0
        found = False
        for value in tactile_dict.values():
            array = np.asarray(value, dtype=float).reshape(-1)
            if array.size == 0 or not np.all(np.isfinite(array)):
                continue
            total += float(np.sum(array))
            found = True
        if found:
            return total
    return None


def _emit_tactile_sum_only_status(
    components: RealTeleopComponents,
    *,
    now: float,
    stream,
) -> None:
    timestamp = time.strftime("%H:%M:%S", time.localtime(now))
    left_sum = _hand_tactile_sum(components.left_hand_backend)
    right_sum = _hand_tactile_sum(components.right_hand_backend)
    left_text = "None" if left_sum is None else f"{left_sum:.4f}"
    right_text = "None" if right_sum is None else f"{right_sum:.4f}"
    stream.write(f"[{timestamp}] [TactileSum] left={left_text} right={right_text}\n")
    stream.flush()


def _release_runtime_state_readers(components: RealTeleopComponents) -> None:
    if components.arm_state_read_executor is not None:
        components.arm_state_read_executor.shutdown(wait=True)


def main() -> None:
    args = parse_args()
    components = _build_components(args)
    clutch_input = TerminalClutchInput(components.keyboard_source)
    episode_recorder = EpisodeRecorder(
        root_dir=_episode_root_dir(args, components),
        config=EpisodeRecorderConfig(enabled=not args.no_episode_recording),
    )
    console_stream = sys.__stdout__
    silenced_stdout = None
    original_stdout = None
    if args.print_tactile_sum_only:
        console_stream.write(
            "[TactileSum] tactile-sum-only mode enabled; press 'p' or Enter for calibration, 'q' to quit.\n"
        )
        console_stream.flush()
        silenced_stdout = open(os.devnull, "w")
        original_stdout = sys.stdout
        sys.stdout = silenced_stdout
    try:
        if components.left_state_backend is not None:
            components.left_state_backend.start()
        if components.right_state_backend is not None:
            components.right_state_backend.start()
        if components.left_hand_backend is not None and not components.left_hand_control_via_rig:
            components.left_hand_backend.start()
        if components.right_hand_backend is not None and not components.right_hand_control_via_rig:
            components.right_hand_backend.start()
        components.session.start()
        clutch_input.start()
        _sync_preview_from_real_state_real(components, required=True)
        _sync_dual_arm_retargeter_from_preview(components)
        _sync_real_execution_step_limits(components)

        for source_name, (logical_side, arm_ip) in _resolve_logical_side_assignments(args).items():
            if not arm_ip or logical_side not in components.active_sides:
                continue
            _log("ArmMode", f"{source_name}={arm_ip} -> logical_{logical_side}_arm")
        if components.left_hand_retargeter is not None or components.right_hand_retargeter is not None:
            _log(
                "HandTeleop",
                "enabled "
                f"left_port={getattr(args, 'left_hand_port', '') or None} "
                f"right_port={getattr(args, 'right_hand_port', '') or None} "
                f"left_hand_ip={args.left_hand_ip or None} "
                f"right_hand_ip={args.right_hand_ip or None} "
                f"joint_order={list(getattr(args, 'hand_joint_order', DEFAULT_INSPIRE_HAND_JOINT_ORDER))} "
                f"invert={bool(getattr(args, 'hand_invert', True))} "
                f"alpha={float(getattr(args, 'hand_command_alpha', 1.0)):.3f} "
                f"telemetry_hz={float(getattr(args, 'hand_telemetry_hz', 30.0)):.1f} "
                f"tactile_regions={list(_parse_hand_tactile_regions(getattr(args, 'hand_tactile_regions', ()))) or []} "
                f"dispatch=rig_parallel",
            )
        elif components.left_hand_backend is not None or components.right_hand_backend is not None:
            _log(
                "HandTeleop",
                "telemetry-only "
                f"left_hand_ip={args.left_hand_ip or None} "
                f"right_hand_ip={args.right_hand_ip or None} "
                f"telemetry_hz={float(getattr(args, 'hand_telemetry_hz', 30.0)):.1f} "
                f"tactile_regions={list(_parse_hand_tactile_regions(getattr(args, 'hand_tactile_regions', ()))) or []}",
            )
        else:
            _log("HandTeleop", "disabled; arm teleop path remains unchanged.")
        if args.teleop_hardware_ip or args.teleop_hardware_port:
            _log(
                "ReservedIO",
                "teleop-hardware endpoint is still reserved on this entrypoint and remains inactive.",
            )
        if components.loaded_initial_alignment is not None:
            _log(
                "InitialAlignment",
                f"source={args.load_initial_alignment} "
                f"left_q={_qpos_summary(components.loaded_initial_alignment.get_side('left').arm_qpos if components.loaded_initial_alignment.get_side('left') is not None else None)} "
                f"right_q={_qpos_summary(components.loaded_initial_alignment.get_side('right').arm_qpos if components.loaded_initial_alignment.get_side('right') is not None else None)}",
            )
            start_check_status = _run_initial_alignment_start_check(args, components)
            if start_check_status == "passed":
                _log(
                    "InitialAlignment",
                    "startup joint check passed; runtime home capture at 'p' will still use the live real-arm pose.",
                )
            elif start_check_status == "ignored_mismatch":
                _log(
                    "InitialAlignment",
                    "startup joint mismatch was ignored by CLI; runtime home capture at 'p' will still use the live real-arm pose.",
                )
        _emit_tracker_axis_mapping_logs(components)
        _log(
            "IK",
            f"objective_mode={components.session.calibration.metadata.get('ik_objective_mode')} "
            f"tuning={components.session.calibration.metadata.get('ik_tuning_source')}",
        )
        _log(
            "ArmMode",
            f"state_read_retries={args.rm65_hardware_read_retries} "
            f"retry_delay_sec={args.rm65_hardware_read_retry_delay_sec:.3f} "
            f"io_threads={max(1, int(args.arm_io_threads))} "
            f"streaming_hz={float(args.hz):.1f} "
            f"minimum_control_hz={args.minimum_control_hz:.1f} "
            f"arm_max_angvel_rad_s={np.round(np.asarray(args.arm_max_angvel, dtype=float), 3).tolist()} "
            f"actual_dt_scaling={'on' if args.arm_step_dt_scaling else 'off'} "
            f"dt_alpha={args.arm_step_dt_alpha:.2f} "
            f"dt_max_scale={args.arm_step_dt_max_scale:.2f} "
            f"dt_stall_reset_sec={args.arm_step_dt_stall_reset_sec:.3f}",
        )
        if components.collision_guard_setup_error is not None:
            _log(
                "CollisionGuard",
                "disabled automatically; "
                f"{components.collision_guard_setup_error}. "
                "Falling back to two independent arm retargeters. "
                "Use --no-collision to disable this explicitly.",
            )
        if components.dual_arm_retargeter is not None:
            _log(
                "CollisionGuard",
                "enabled "
                f"scene_profile={args.collision_scene_profile} "
                f"pairs={len(components.dual_arm_retargeter.profile.collision_pairs)} "
                f"mode={args.collision_avoidance_mode}",
            )
        else:
            _log("CollisionGuard", "disabled; using two independent arm retargeters.")
        _emit_collision_proxy_logs(components)
        if _manual_calibration_confirmation_required(args, components):
            should_continue = _wait_for_calibration_start(
                components,
                clutch_input,
                arm_mode="real",
            )
            if not should_continue:
                return
        _runtime_home_calibration(
            components,
            arm_mode="real",
            timeout_sec=args.calibration_timeout,
            num_samples=args.calibration_samples,
        )
        _release_runtime_state_readers(components)
        _log(
            "ArmMode",
            "runtime arm state source=last_commanded_action; startup hardware-state readers are now idle after calibration.",
        )
        _emit_home_alignment_logs(components)
        if legacy_tcp_mode_requested(args.requested_tcp_mode):
            _log(
                "TCP",
                "legacy --tcp-mode rm_eef was requested; runtime now uses hand_base only.",
            )
        components.session.set_calibration_ready()
        components.session.mark_debug_passed()
        components.session.arm_hardware()
        clutch_input.consume_episode_start_request()
        clutch_input.consume_episode_stop_request()

        if args.keyboard_clutch:
            _log(
                "Keys",
                "p/Enter=start calibration, t=start episode, s=stop episode, z=left clutch, x=right clutch, c=both, r=release, q=quit",
            )
        elif args.control_mode == "delta_clutch":
            _log("Info", "delta_clutch is running with both clutch buttons held true by default.")
        if episode_recorder.config.enabled:
            auto_start_note = " auto_start=on" if args.record_auto_start else " auto_start=off"
            _log(
                "Episode",
                f"recording ready root={episode_recorder.root_dir or _episode_root_dir(args, components)} "
                f"start_key='t' stop_key='s'{auto_start_note}",
            )

        status_gate = ChangeGate()
        orientation_gate = ChangeGate()
        tactile_sum_gate = ChangeGate()
        step_index = 0
        recovery_plan: RecoveryPlan | None = None
        episode_last_state: str | None = None
        if args.record_auto_start and episode_recorder.config.enabled:
            _start_episode_recording(
                episode_recorder,
                args,
                components,
                now=time.time(),
                start_reason="auto_start",
            )

        while True:
            loop_start = time.time()
            clutch_input.poll()
            if clutch_input.should_quit:
                if episode_recorder.is_recording:
                    _stop_episode_recording(
                        episode_recorder,
                        now=loop_start,
                        end_reason="quit_requested",
                        summary={
                            "last_step_index": step_index,
                            "last_state": episode_last_state,
                        },
                    )
                break
            if clutch_input.consume_episode_stop_request() and episode_recorder.config.enabled:
                if episode_recorder.is_recording:
                    _stop_episode_recording(
                        episode_recorder,
                        now=loop_start,
                        end_reason="manual_key_stop",
                        summary={
                            "last_step_index": step_index,
                            "last_state": episode_last_state,
                        },
                    )
                    episode_last_state = None
            if clutch_input.consume_episode_start_request() and episode_recorder.config.enabled:
                if not episode_recorder.is_recording:
                    _start_episode_recording(
                        episode_recorder,
                        args,
                        components,
                        now=loop_start,
                        start_reason="manual_key_start",
                    )

            _sync_dual_arm_retargeter_from_preview(components)
            _sync_real_execution_step_limits(components, now=loop_start)

            step_result = components.session.run_step(now=loop_start)
            active_command = step_result.command
            hand_command = _extract_hand_command_from_command(step_result.command)
            if step_result.state == TeleopRunState.RECOVERING:
                hand_command = _build_runtime_hand_command_for_recovery(
                    components,
                    step_result.observation,
                )
                if step_result.observation is None:
                    active_command = None
                elif recovery_plan is None:
                    recovery_plan, active_command, recovery_done = _start_recovery_plan(
                        components,
                        step_result.observation,
                        now=loop_start,
                        arm_mode="real",
                        control_mode=args.control_mode,
                        ramp_sec=args.recovery_ramp_sec,
                    )
                    if not episode_recorder.is_recording:
                        if recovery_plan is None:
                            _log(
                                "Recovery",
                                f"direct mode control_mode={args.control_mode}",
                                now=loop_start,
                            )
                        else:
                            _log(
                                "Recovery",
                                f"start mode={recovery_plan.mode} duration={recovery_plan.duration_sec:.2f}s",
                                now=loop_start,
                            )
                    active_command = _merge_runtime_hand_command(active_command, hand_command)
                    if active_command is not None:
                        assert components.session.rig is not None
                        components.session.rig.send_commands(active_command)
                    if recovery_done:
                        components.session.finish_recovery("recovery completed")
                        recovery_plan = None
                else:
                    active_command, recovery_done = _advance_recovery_plan(
                        components,
                        step_result.observation,
                        recovery_plan,
                        now=loop_start,
                    )
                    active_command = _merge_runtime_hand_command(active_command, hand_command)
                    if active_command is not None:
                        assert components.session.rig is not None
                        components.session.rig.send_commands(active_command)
                    if recovery_done:
                        components.session.finish_recovery("recovery completed")
                        if not episode_recorder.is_recording:
                            _log("Recovery", "completed", now=loop_start)
                        recovery_plan = None
            else:
                recovery_plan = None

            _apply_active_arm_command_to_preview_real(components, active_command)
            _sync_dual_arm_retargeter_from_preview(components)

            if args.print_tactile_sum_only:
                tactile_snapshot = (
                    _hand_tactile_sum(components.left_hand_backend),
                    _hand_tactile_sum(components.right_hand_backend),
                )
                if tactile_sum_gate.should_emit(
                    "tactile_sum_only_status",
                    tactile_snapshot,
                    now=loop_start,
                    heartbeat_sec=0.1,
                ):
                    _emit_tactile_sum_only_status(
                        components,
                        now=loop_start,
                        stream=console_stream,
                    )

            left_qpos = _state_qpos(components.left_preview_backend, DEFAULT_LEFT_ARM_QPOS)
            right_qpos = _state_qpos(components.right_preview_backend, DEFAULT_RIGHT_ARM_QPOS)
            if components.renderer is not None:
                components.renderer.set_debug_frames(
                    _build_renderer_debug_frames(
                        components,
                        channels=step_result.channels,
                    )
                )
                components.renderer.set_debug_boxes(_build_renderer_debug_boxes(components))
                components.renderer.render_data(
                    left_qpos_arm=left_qpos,
                    left_qpos_hand=_hand_render_qpos(components.left_hand_backend, "left"),
                    right_qpos_arm=right_qpos,
                    right_qpos_hand=_hand_render_qpos(components.right_hand_backend, "right"),
                    capture_image=False,
                )
            if not episode_recorder.is_recording:
                _emit_runtime_orientation_logs(
                    components,
                    channels=step_result.channels,
                    now=loop_start,
                    gate=orientation_gate,
                )

            fault_codes = tuple(
                f"{fault.code}:{fault.channel_name}" if fault.channel_name else fault.code
                for fault in step_result.faults
            )
            channel_valid = tuple(
                sorted((name, bool(channel.valid)) for name, channel in step_result.channels.items())
            )
            clutch_channels = components.keyboard_source.get_latest_channels()
            left_clutch_channel = clutch_channels.get("left_clutch_button")
            right_clutch_channel = clutch_channels.get("right_clutch_button")
            left_clutch = bool(left_clutch_channel.value) if left_clutch_channel is not None else False
            right_clutch = bool(right_clutch_channel.value) if right_clutch_channel is not None else False
            if episode_recorder.is_recording:
                if episode_last_state != step_result.state.value:
                    episode_recorder.record_event(
                        "state_change",
                        payload={
                            "old_state": episode_last_state,
                            "new_state": step_result.state.value,
                        },
                        timestamp=loop_start,
                    )
                    episode_last_state = step_result.state.value
                episode_recorder.record_frame(
                    _build_episode_frame(
                        components,
                        step_index=step_index,
                        step_result=step_result,
                        active_command=active_command,
                        hand_command=hand_command,
                        recovery_plan=recovery_plan,
                        left_clutch=left_clutch,
                        right_clutch=right_clutch,
                    ),
                    timestamp=loop_start,
                )

            collision_status = None
            collision_fallback = None
            jump_gate_status = None
            branch_candidate = None
            if active_command is not None:
                collision_status = active_command.metadata.get("collision_avoidance_status")
                collision_fallback = active_command.metadata.get(
                    "collision_avoidance_fallback_reason"
                )
                jump_gate_status = active_command.metadata.get("ik_jump_gate_status")
                branch_candidate = active_command.metadata.get("ik_branch_candidate")
            left_hw_state = components.left_state_backend.get_state() if components.left_state_backend is not None else {}
            right_hw_state = components.right_state_backend.get_state() if components.right_state_backend is not None else {}
            left_command_state = (
                components.left_command_backend.get_state()
                if components.left_command_backend is not None
                else {}
            )
            right_command_state = (
                components.right_command_backend.get_state()
                if components.right_command_backend is not None
                else {}
            )
            active_command_backend = _active_command_backend(components)
            base_control_dt = (
                float(active_command_backend.config.control_dt)
                if active_command_backend is not None
                else 1.0 / float(args.hz)
            )
            hardware_snapshot = (
                left_hw_state.get("hardware_result_code"),
                right_hw_state.get("hardware_result_code"),
            )
            step_scale_snapshot = (
                round(float(left_command_state.get("effective_step_scale", 1.0)), 3),
                round(float(right_command_state.get("effective_step_scale", 1.0)), 3),
                round(
                    float(
                        components.arm_loop_step_dt_tracker.get("smoothed_loop_dt", args.hz and (1.0 / float(args.hz)))
                    )
                    / base_control_dt,
                    3,
                ),
            )
            status_snapshot = (
                step_result.state.value,
                fault_codes,
                channel_valid,
                left_clutch,
                right_clutch,
                collision_status,
                collision_fallback,
                jump_gate_status,
                branch_candidate,
                hardware_snapshot,
                step_scale_snapshot,
            )
            if status_gate.should_emit(
                "real_teleop_status",
                status_snapshot,
                now=loop_start,
                heartbeat_sec=args.status_sec,
            ):
                collision_bits = ""
                if collision_status is not None:
                    collision_bits = f" collision={collision_status}"
                    if collision_fallback:
                        collision_bits += f" fallback={collision_fallback}"
                if jump_gate_status is not None:
                    collision_bits += f" ik={jump_gate_status}"
                if branch_candidate is not None:
                    collision_bits += f" branch={branch_candidate}"
                _log(
                    "RealTeleop",
                    f"step={step_index} hz={step_result.control_hz:.1f} "
                    f"state={step_result.state.value} "
                    f"left_hw_init={left_hw_state.get('hardware_result_code')}:{left_hw_state.get('hardware_result_text')} "
                    f"right_hw_init={right_hw_state.get('hardware_result_code')}:{right_hw_state.get('hardware_result_text')} "
                    f"left_step_scale={float(left_command_state.get('effective_step_scale', 1.0)):.2f} "
                    f"right_step_scale={float(right_command_state.get('effective_step_scale', 1.0)):.2f} "
                    f"target_step_scale={float(components.arm_loop_step_dt_tracker.get('smoothed_loop_dt', base_control_dt)) / base_control_dt:.2f} "
                    f"left_clutch={left_clutch} "
                    f"right_clutch={right_clutch} "
                    f"faults={list(fault_codes)}"
                    f"{collision_bits}",
                    now=loop_start,
                )

            step_index += 1
            if args.max_steps > 0 and step_index >= args.max_steps:
                break

            remaining = (1.0 / args.hz) - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        if episode_recorder.is_recording:
            _stop_episode_recording(
                episode_recorder,
                now=time.time(),
                end_reason="shutdown",
                summary={"reason": "finally"},
            )
        episode_recorder.close()
        clutch_input.close()
        components.session.close()
        if components.arm_state_read_executor is not None:
            components.arm_state_read_executor.shutdown(wait=True)
        if components.left_state_backend is not None:
            components.left_state_backend.close()
        if components.right_state_backend is not None:
            components.right_state_backend.close()
        if components.left_hand_backend is not None and not components.left_hand_control_via_rig:
            components.left_hand_backend.close()
        if components.right_hand_backend is not None and not components.right_hand_control_via_rig:
            components.right_hand_backend.close()
        if original_stdout is not None:
            sys.stdout = original_stdout
        if silenced_stdout is not None:
            silenced_stdout.close()


if __name__ == "__main__":
    main()
