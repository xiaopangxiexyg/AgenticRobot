"""Argument-parser construction for the real VIVE-to-RM65 frontend.

This module registers CLI actions only. Parsing, validation, configuration
loading, device preflight, and all runtime side effects remain in the canonical
entrypoint parse_args coordinator.
"""

from __future__ import annotations

import argparse
import os

from dataset_contracts.pose_frames import load_pose_frame_contract
from dataset_contracts.v25p_native_storage import (
    V25P_NATIVE_STORAGE_LAYOUTS_V1,
    V25P_RECORD_CJSON_LAYOUT_V1,
)
from runtime_artifact_paths import (
    LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
    RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
    configured_runtime_artifact_layout,
)

from robot_motion.position_control import (
    DEFAULT_POSITION_DEGRADED_SPEED_SCALE,
    DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
    DEFAULT_POSITION_VELOCITY_RAMP_SEC,
    POSITION_COMMAND_PLANNER_CHOICES,
    POSITION_COMMAND_PLANNER_LINEAR,
    ROTATIONAL_ADMITTANCE_PROFILE_BALANCED_5DEG,
    ROTATIONAL_ADMITTANCE_PROFILE_CHOICES,
)
from robot_motion.rm65 import (
    IK_STEP_MODE_CHOICES,
    IK_STEP_MODE_LEGACY_ITERATIVE,
    JOINT_LIMIT_GUARD_LEGACY,
    JOINT_LIMIT_GUARD_MODE_CHOICES,
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_POSITION_CONTROL_PROFILE_CHOICES,
)

from ..action.human_shadow_frontend import add_human_shadow_arguments
from ..arm_frames import TOOL_TCP_FRAME
from ..backends.hands import DEFAULT_INSPIRE_HAND_MAX_FORCE
from ..cli_value_parsers import (
    _alpha_float,
    _non_negative_int,
    _parse_active_sides,
    _parse_arm_max_angvel,
    _positive_float,
)
from ..configs import (
    ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
    DEFAULT_ARM_WORKSPACE_CONFIG_PATH,
)
from ..defaults import (
    DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
    DEFAULT_TELEOP_SAMPLE_HZ,
)
from ..hand_tcp_alignment import add_vive_tracker_height_correction_args
from ..position_control import (
    ARM_EXECUTION_PROFILE_CHOICES,
    CONTROL_DEBUG_LEVELS,
    DEFAULT_CONTROL_DEBUG_FLUSH_EVERY,
    DEFAULT_CONTROL_DEBUG_QUEUE_SIZE,
    DEFAULT_CONTROL_DEBUG_ROOT,
)
from ..sources.camera import (
    REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
    REALSENSE_STARTUP_WARMUP_POLICIES,
)
from ..sources.pose import DEFAULT_VIVE_CLOCK_SYNC_PORT
from .actuator_command_trace import (
    DEFAULT_ACTUATOR_TRACE_MAX_DURATION_SEC,
    DEFAULT_ACTUATOR_TRACE_ROOT,
)
from .cli_value_parsers import (
    _canfd_radio,
    _canfd_trajectory_mode,
    _dmtac_worker_cpu_affinity,
    _dmtac_worker_nice,
    _local_step_clip_scale_arg,
    _movement_multiple_arg,
    _non_negative_float,
    _positive_finite_float_arg,
    _positive_int,
    _positive_unit_scale_arg,
)
from .debug_recorders import DEFAULT_ACTION_DEBUG_LOG_ROOT
from .defaults import (
    DEFAULT_CAMERA_CONFIG_PATH,
    DEFAULT_COLLECTION_PROCESS_STARTUP_TIMEOUT_SEC,
    DEFAULT_DATASET_ROOT,
    DEFAULT_INIT_ARM_PROFILE_PATH,
    DEFAULT_INIT_ARM_QPOS_SPEED_SCALE,
    DEFAULT_POSITION_MAX_ACCELERATION_RAD_S2,
    DEFAULT_REAL_TELEOP_MAX_TRACKER_ROTATION_JUMP_DEG,
    DEFAULT_REAL_TELEOP_MAX_TRACKER_TRANSLATION_JUMP_M,
    INIT_ARM_QPOS_SPEED_BASELINE_PROFILE,
)
from .foot_pad_controls import SPECIAL_ARM_CONTROL_MODE_CHOICES
from .hand_cli import (
    add_dmtac_arguments,
    add_gripper_policy_arguments,
    add_hand_backend_arguments,
    add_inspire_hand_arguments,
)
from .movement import MOVEMENT_MULTIPLE_DEFAULT
from .position_force_guard import POSITION_FORCE_SOURCES
from .position_admittance_shadow import (
    POSITION_ADMITTANCE_ACCEPTED_FILTER_TAU_SEC,
    POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENVELOPE,
    POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
    POSITION_ADMITTANCE_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_M,
    POSITION_ADMITTANCE_ACTIVE_ENVELOPE_CHOICES,
    POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_CHOICES,
    POSITION_ADMITTANCE_ACTIVE_SIDE_CHOICES,
    POSITION_ADMITTANCE_ROTATION_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_DEG,
    POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE,
    POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE_CHOICES,
    POSITION_ADMITTANCE_RESPONSE_WEIGHT_DEFAULT,
)
from .progress_adapter import load_real_teleop_progress_profile
from .wrist_depth_freshness import (
    DEFAULT_WRIST_DEPTH_MAX_M,
    DEFAULT_WRIST_DEPTH_MIN_M,
    DEFAULT_WRIST_DEPTH_STREAM_FPS,
)


def build_real_teleop_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Control real dual RM65 arms from live VIVE wrist trackers, with optional "
            "additional hand q6 teleop and InspireTac hand control."
        )
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="UDP bind host for the VIVE receiver")
    parser.add_argument("--port", type=int, default=9999, help="UDP bind port for the VIVE receiver")
    parser.add_argument(
        "--vive-clock-sync-port",
        type=int,
        default=DEFAULT_VIVE_CLOCK_SYNC_PORT,
        help=(
            "Separate VIVE clock-control UDP port for the default new protocol; "
            "set 0 to select the legacy v1-only receiver."
        ),
    )
    parser.add_argument(
        "--vive-clock-sync-bind-host",
        type=str,
        default="",
        help="Optional VIVE clock-control bind host; defaults to --host.",
    )
    parser.add_argument(
        "--vive-clock-sync-expected-sender-host",
        type=str,
        default="",
        help="Optional VIVE sender IP allowed to submit clock-control results.",
    )
    parser.add_argument(
        "--vive-clock-sync-min-samples",
        type=int,
        default=4,
        help="Accepted four-timestamp samples required before sync is ready.",
    )
    parser.add_argument(
        "--vive-clock-sync-max-rtt-ms",
        type=float,
        default=25.0,
        help="Reject VIVE clock samples above this RTT; <=0 disables the cap.",
    )
    parser.add_argument(
        "--vive-clock-sync-stale-after-sec",
        type=float,
        default=15.0,
        help="Mark remote capture-time mapping stale after this interval.",
    )
    parser.add_argument(
        "--vive-clock-sync-required",
        action="store_true",
        help="Fail startup when VIVE clock-control does not become ready.",
    )
    parser.add_argument(
        "--vive-clock-sync-startup-timeout-sec",
        type=float,
        default=5.0,
        help="Maximum startup wait for required VIVE clock calibration.",
    )
    parser.add_argument(
        "--active-sides",
        type=_parse_active_sides,
        default=("left", "right"),
        help="One of: left, right, both.",
    )
    parser.add_argument("--left-arm-ip", type=str, default="", help="Left RM65 IP")
    parser.add_argument("--right-arm-ip", type=str, default="", help="Right RM65 IP")
    parser.add_argument(
        "--arm-ip-config",
        type=str,
        default="latest",
        help="RM65 arm IP assignment config path, or 'latest'. Missing --left/right-arm-ip values are filled from it.",
    )
    parser.add_argument(
        "--no-arm-ip-default-config",
        dest="arm_ip_default_config",
        action="store_false",
        default=True,
        help="Do not auto-fill missing RM65 arm IPs from the latest saved assignment.",
    )
    parser.add_argument(
        "--no-arm-ip-save-latest",
        dest="arm_ip_save_latest",
        action="store_false",
        default=True,
        help="Do not save explicitly provided RM65 arm IPs to the latest assignment.",
    )
    parser.add_argument("--left-hand-ip", type=str, default="", help="Optional left InspireTac IP")
    parser.add_argument("--right-hand-ip", type=str, default="", help="Optional right InspireTac IP")
    parser.add_argument("--left-hand-port", type=str, default="", help="Optional left hand teleop serial port")
    parser.add_argument("--right-hand-port", type=str, default="", help="Optional right hand teleop serial port")
    parser.add_argument(
        "--fix-arm",
        dest="fix_arm",
        action="store_true",
        default=True,
        help=(
            "Enable fixed-arm collection mode (default: enabled for dual-arm GUI). "
            "In this mode GUI C freezes the selected arm/hand at the last dispatched "
            "action, and C again recaptures that side's runtime home before releasing it."
        ),
    )
    parser.add_argument(
        "--no-fix-arm",
        dest="fix_arm",
        action="store_false",
        help=(
            "Disable the default fix-arm controller and restore the legacy GUI controls."
        ),
    )
    parser.add_argument(
        "--fix-arm-side",
        choices=("left", "right"),
        default="left",
        help="Initial side selected by --fix-arm. The GUI can switch it while no side is fixed.",
    )
    parser.add_argument(
        "--init-arm-available",
        "--init-arm-availabe",
        dest="init_arm_available",
        action="store_true",
        default=True,
        help=(
            "Enable init-arm mode (default: enabled for dual-arm GUI). GUI G routes "
            "the selected arm to the init pose, and G again recaptures that side's "
            "runtime home before resuming teleop."
        ),
    )
    parser.add_argument(
        "--no-init-arm-available",
        "--no-init-arm-availabe",
        dest="init_arm_available",
        action="store_false",
        help="Disable the default init-arm controller.",
    )
    parser.add_argument(
        "--init-arm-profile",
        type=str,
        default=str(DEFAULT_INIT_ARM_PROFILE_PATH),
        help=(
            "Dual-arm init pose profile JSON used by --init-arm-available. "
            "Each side may provide arm_qpos, tcp_pose_base, or tcp_pose_desk. "
            f"Default: {DEFAULT_INIT_ARM_PROFILE_PATH}."
        ),
    )
    parser.add_argument(
        "--init-arm-side",
        choices=("left", "right"),
        default="left",
        help="Initial side selected by --init-arm-available. The GUI can switch it while init-arm is idle.",
    )
    parser.add_argument(
        "--init-arm-left-qpos-speed-scale",
        type=_positive_float,
        default=DEFAULT_INIT_ARM_QPOS_SPEED_SCALE,
        help=(
            "Left-arm exact Init-qpos return speed multiplier relative to "
            f"{INIT_ARM_QPOS_SPEED_BASELINE_PROFILE}=1.0. Default: "
            f"{DEFAULT_INIT_ARM_QPOS_SPEED_SCALE:g}. The GUI may adjust it while "
            "no Init-qpos operation is active."
        ),
    )
    parser.add_argument(
        "--init-arm-right-qpos-speed-scale",
        type=_positive_float,
        default=DEFAULT_INIT_ARM_QPOS_SPEED_SCALE,
        help=(
            "Right-arm exact Init-qpos return speed multiplier relative to "
            f"{INIT_ARM_QPOS_SPEED_BASELINE_PROFILE}=1.0. Default: "
            f"{DEFAULT_INIT_ARM_QPOS_SPEED_SCALE:g}. The GUI may adjust it while "
            "no Init-qpos operation is active."
        ),
    )
    parser.add_argument(
        "--special-arm-control-mode",
        choices=SPECIAL_ARM_CONTROL_MODE_CHOICES,
        default=None,
        help=(
            "Optional pedal mapping for special-arm actions. Four pedals map "
            "F13/F14/F15/F16 to Fix-left/Fix-right/Init-left/Init-right; two "
            "pedals map F13/F14 to Fix-left/Fix-right. A five-pedal topology "
            "uses the same two-pedal side-switch semantics for F13/F14 and "
            "B/C, while F15/F16/F17 map to Advance/Failure/Undo only after a "
            "progress profile is ready. Pedal topology or "
            "availability never disables configured GUI/keyboard controls."
        ),
    )
    add_hand_backend_arguments(parser)
    add_dmtac_arguments(parser)
    add_gripper_policy_arguments(parser)
    add_inspire_hand_arguments(parser)
    parser.add_argument(
        "--camera-config",
        type=str,
        default=str(DEFAULT_CAMERA_CONFIG_PATH),
        help="JSON config for RealSense RGB capture/video and optional preview windows.",
    )
    parser.add_argument(
        "--camera-color-format",
        choices=("bgr8", "rgb8", "yuyv", "uyvy", "y8", "rgba8", "bgra8"),
        default="bgr8",
        help=(
            "RealSense color transport format for real-teleop cameras. "
            "Default bgr8 avoids unstable multi-camera YUYV negotiation; "
            "use yuyv only for an explicitly validated station."
        ),
    )
    parser.add_argument(
        "--camera-startup-warmup-policy",
        choices=REALSENSE_STARTUP_WARMUP_POLICIES,
        default=REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
        help=(
            "RealSense post-pipeline startup warmup. legacy preserves the "
            "historical synchronous N-frame wait, reset retry, any-frameset "
            "admission, and unbounded runtime recovery; bounded_first_frame "
            "requires one complete configured color/depth frameset within each "
            "startup_frame_grace_sec attempt and enables bounded incomplete-stream "
            "recovery, using short cancellable SDK waits; "
            "one failed pipeline is cleanly reopened once before admission fails, "
            "then the unchanged global multi-frame camera self-check runs. "
            "Pipeline negotiation remains serial and no USB reset is implied. "
            "Default: bounded_first_frame; select legacy for atomic rollback."
        ),
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Disable RealSense capture, preview, and video writing.",
    )
    parser.add_argument(
        "--ignore-camera",
        action="append",
        default=[],
        metavar="CAMERA_KEY",
        help=(
            "Explicitly omit one camera view for the complete target dataset. "
            "The dataset must be new/blank or already declare the exact same ignore set."
        ),
    )
    parser.add_argument(
        "--ignore-tactile",
        action="append",
        choices=("left", "right"),
        default=[],
        metavar="SIDE",
        help=(
            "Explicitly omit one tactile side for the complete target dataset. "
            "Arm/gripper state remains required."
        ),
    )
    parser.add_argument(
        "--vis_cam",
        "--vis-cam",
        dest="vis_cam",
        action="store_true",
        help="Open optional RealSense RGB preview windows at startup.",
    )
    parser.add_argument(
        "--vis-cam-key",
        "--vis-camera-key",
        type=str,
        default="",
        help=(
            "Camera key to visualize when camera preview is opened. "
            "Defaults to all configured cameras; use 'front', 'left_wrist', or 'right_wrist' to show one."
        ),
    )
    parser.add_argument(
        "--vis-cam-hz",
        type=_positive_float,
        default=10.0,
        help="Maximum camera preview update rate.",
    )
    parser.add_argument(
        "--inspire-hand-max-force",
        "--max-force",
        "--max-fingertip-force",
        dest="max_fingertip_force",
        type=_non_negative_int,
        default=DEFAULT_INSPIRE_HAND_MAX_FORCE,
        help="Max force value written to every InspireTac finger on startup.",
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
        "--swap-operator-sides",
        dest="swap_operator_sides",
        action="store_true",
        help=(
            "Before teleop starts, swap only VIVE tracker inputs: logical left uses the right VIVE "
            "tracker and logical right uses the left VIVE tracker. Arms, hands, serial ports, and "
            "profiles are unchanged. This mapping is fixed after startup."
        ),
    )
    parser.add_argument(
        "--control-mode",
        choices=["absolute"],
        default="absolute",
        help=(
            "Arm retargeting mode. The active runtime uses the archived absolute pose mapping."
        ),
    )
    parser.add_argument(
        "--arm-execution-profile",
        choices=ARM_EXECUTION_PROFILE_CHOICES,
        default=None,
        help=(
            "Mode-aware real-arm defaults. impedance_40hz resolves to a 15 Hz source "
            "and 40 Hz impedance backend; position resolves to the configurable "
            "position chain (defaults: 15 Hz sampling, 60 Hz references, "
            "120 Hz CANFD stream). "
            "position_45_150hz remains an explicit legacy profile, and fine-grained "
            "flags remain compatible overrides."
        ),
    )
    parser.add_argument(
        "--position-control-profile",
        choices=RM65_POSITION_CONTROL_PROFILE_CHOICES,
        default=RM65_BIMANUAL_POSITION_V1_NAME,
        help=(
            "Versioned RM65 bimanual position behavior. The formal v1 profile "
            "preserves the stable broad source smoothing; v2_responsive keeps the "
            "same 15/60/120 hard execution envelope but enables a guarded per-side "
            "normal-zone fast path and escape-latched q5 hysteresis. "
            "legacy preserves fine-grained historical defaults for rollback/debug. "
            "This option is ignored unless position arm control is selected."
        ),
    )
    parser.add_argument(
        "--tcp-pose-envelope-profile",
        type=str,
        default="",
        help=(
            "Optional strict JSON profile for the final tool-TCP pose envelope "
            "before IK. Providing a profile alone installs shadow observation; "
            "the default empty value leaves the existing control path unchanged."
        ),
    )
    parser.add_argument(
        "--tcp-pose-envelope-active",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Explicitly project final tool-TCP pose targets into the selected "
            "envelope before IK. Requires --tcp-pose-envelope-profile and is "
            "default-off."
        ),
    )
    parser.add_argument(
        "--arm-control-mode",
        choices=["impedance", "position"],
        default="impedance",
        help=(
            "Real-arm execution mode. Default impedance uses the outer-loop pose impedance profile; "
            "position uses the legacy IK + Movej_CANFD path."
        ),
    )
    parser.add_argument(
        "--safe-debug",
        dest="safe_debug",
        action="store_true",
        help=(
            "Compatibility diagnostic marker for position CANFD passthrough. Target pose "
            "smoothing, local-step candidate selection, and inactive-side hold/cap are now "
            "enabled by default in live teleop."
        ),
    )
    parser.add_argument(
        "--disable-hold",
        action="store_true",
        help=(
            "When used with --safe-debug in position CANFD passthrough, disable small-motion "
            "hold behavior: current-pose hold, pose-smoother deadband hold, and inactive-side hold/cap."
        ),
    )
    parser.add_argument(
        "--strict-degree-limit",
        action="store_true",
        help="Scale qpos velocity/step limits to one fifth of their configured values.",
    )
    parser.add_argument(
        "--impedance-mode",
        choices=["pose"],
        default="pose",
        help="Impedance controller reference mode. Pose is the default and matches the unified arm action.",
    )
    parser.add_argument(
        "--arm-canfd-follow",
        dest="arm_canfd_follow",
        action="store_true",
        help=(
            "Send RM65 CANFD commands with SDK follow=True. The SDK requires a stable "
            "transparent transmission period <=10ms for high-follow use. Defaults to true "
            "for --arm-control-mode position and false for force-admittance impedance."
        ),
    )
    parser.add_argument(
        "--no-arm-canfd-follow",
        dest="arm_canfd_follow",
        action="store_false",
        help="Send RM65 impedance CANFD commands with SDK follow=False.",
    )
    parser.set_defaults(arm_canfd_follow=None)
    parser.add_argument(
        "--position-post-ik-qpos-smoothing-alpha",
        type=_alpha_float,
        default=0.14,
        help=(
            "EMA alpha applied after IK/safety step limits and before the outer qpos "
            "reference is created. Position teleop defaults to the adaptive slow endpoint 0.14. Set 1.0 "
            "to disable this pre-trajectory smoothing."
        ),
    )
    parser.add_argument(
        "--position-post-ik-qpos-smoothing-fast-alpha",
        type=_alpha_float,
        default=0.45,
        help="Fast-motion endpoint for adaptive post-IK qpos smoothing (position teleop only).",
    )
    parser.add_argument(
        "--position-post-ik-qpos-smoothing-adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Adapt post-IK qpos EMA from the slow alpha to --position-post-ik-qpos-"
            "smoothing-fast-alpha using validated tracker linear/angular velocity. "
            "Only position teleoperation enables the policy."
        ),
    )
    parser.add_argument(
        "--arm-follow-qpos-alpha",
        type=_alpha_float,
        default=None,
        help=(
            "Final high-rate qpos EMA alpha applied directly at the configured "
            "--arm-control-hz SDK send boundary. Defaults to 0.1 in position streaming; "
            "the filtered qpos is committed back to the planner. Quick-stop/hold bypasses "
            "smoothing. Set 1.0 to disable."
        ),
    )
    parser.add_argument(
        "--arm-canfd-follow-stream-hz",
        type=_non_negative_float,
        default=None,
        help=(
            "High-rate Movej_CANFD stream frequency used when RM65 CANFD follow=True. "
            "In position mode this controls the background qpos stream; in impedance "
            "joint transport this controls the follow stream. Backend default is mode-specific; "
            "set 0 to disable streaming and fall back to one command per outer/backend step."
        ),
    )
    parser.add_argument(
        "--arm-canfd-follow-warning-ratio",
        type=_alpha_float,
        default=None,
        help=(
            "Warn when actual CANFD follow stream send rate falls below "
            "ratio * --arm-canfd-follow-stream-hz. Backend default is 0.9."
        ),
    )
    parser.add_argument(
        "--arm-canfd-trajectory-mode",
        type=_canfd_trajectory_mode,
        default=None,
        help=(
            "RM65 CANFD high-follow trajectory mode: 0 passthrough, 1 curve fitting, "
            "2 filtering. Backend/profile default is 0."
        ),
    )
    parser.add_argument(
        "--arm-canfd-radio",
        type=_canfd_radio,
        default=None,
        help=(
            "RM65 CANFD smoothing parameter for curve fitting/filtering modes. "
            "Valid range is 0..100. Backend/profile default is 0."
        ),
    )
    parser.add_argument(
        "--arm-state-read-every",
        type=_positive_int,
        default=None,
        help=(
            "RM65 impedance state read decimation. 1 reads arm state every backend step. "
            "When --arm-canfd-follow is enabled, backend default is 4 to reduce SDK lock "
            "contention with the high-rate CANFD stream."
        ),
    )
    parser.add_argument(
        "--arm-disable-force-admittance",
        action="store_true",
        help=(
            "Diagnostic impedance-mode override: keep pose-action execution but zero the "
            "outer-loop force/admittance offset so RM65 commands are driven only by the "
            "retargeted pose, IK, and CANFD follow path."
        ),
    )
    parser.add_argument(
        "--force-tare-sec",
        type=_positive_float,
        default=3.0,
        help="Required stillness/tare duration after runtime home before enabling impedance force control.",
    )
    parser.add_argument(
        "--position-force-guard",
        dest="position_force_guard",
        action="store_true",
        help=(
            "Require the latched RM65 force/torque emergency-stop monitor in real "
            "position mode. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-position-force-guard",
        dest="position_force_guard",
        action="store_false",
        help=(
            "Disable the real-position RM65 force emergency-stop monitor. "
            "Unsafe; intended only for explicit diagnostics."
        ),
    )
    parser.set_defaults(position_force_guard=True)
    parser.add_argument(
        "--position-force-guard-source",
        choices=POSITION_FORCE_SOURCES,
        default="tool",
        help="RM65 force source used by the position emergency-stop monitor.",
    )
    parser.add_argument(
        "--position-force-guard-max-force-n",
        type=_positive_float,
        default=30.0,
        help="Position-mode bimanual emergency-stop threshold for force-vector norm.",
    )
    parser.add_argument(
        "--position-force-guard-max-torque-nm",
        type=_positive_float,
        default=5.0,
        help="Position-mode bimanual emergency-stop threshold for torque-vector norm.",
    )
    parser.add_argument(
        "--position-force-guard-hz",
        type=_positive_float,
        default=20.0,
        help=(
            "Per-arm RM65 force polling rate. Must be <=20 Hz because the SDK "
            "requires at least 50 ms between repeated force queries."
        ),
    )
    parser.add_argument(
        "--position-force-guard-consecutive-samples",
        type=_positive_int,
        default=1,
        help="Consecutive over-limit samples required before the latched bimanual stop.",
    )
    parser.add_argument(
        "--position-force-guard-max-read-errors",
        type=_positive_int,
        default=3,
        help="Consecutive RM65 force-read errors allowed before fail-closed bimanual stop.",
    )
    parser.add_argument(
        "--position-force-guard-stale-sec",
        type=_positive_float,
        default=0.4,
        help="Maximum age of the latest valid force sample before fail-closed stop.",
    )
    parser.add_argument(
        "--position-force-guard-startup-timeout-sec",
        type=_positive_float,
        default=2.0,
        help=(
            "Startup deadline for one valid force sample from every active arm; "
            "failure stops both arms and aborts arming."
        ),
    )
    parser.add_argument(
        "--position-admittance-shadow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Observe a bounded translation-only Cartesian admittance overlay "
            "from the existing position force-guard samples without changing "
            "any pre-IK target, qref, or command. Default off."
        ),
    )
    parser.add_argument(
        "--position-admittance-rotation-shadow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Observe a bounded SO(3) rotational admittance response from the "
            "same tared tool wrench. Requires --position-admittance-shadow and "
            "disables the translation shadow axes so the receipt is isolated. "
            "It never changes a target, qref, or hardware command. Default off."
        ),
    )
    parser.add_argument(
        "--position-admittance-rotation-shadow-profile",
        choices=ROTATIONAL_ADMITTANCE_PROFILE_CHOICES,
        default=ROTATIONAL_ADMITTANCE_PROFILE_BALANCED_5DEG,
        help=(
            "Named rotation-shadow response profile. balanced-5deg is the "
            "saved-wrench MuJoCo candidate; this option has no Active authority."
        ),
    )
    parser.add_argument(
        "--position-admittance-active",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply a bounded Cartesian admittance overlay to copied pre-IK "
            "targets. Translation is the default; the separate rotation-active "
            "flag selects the isolated cautious rotational envelope. This is "
            "an explicit, default-off research mode; IK, collision, qref "
            "dynamics, and the independent position force guard remain authoritative."
        ),
    )
    parser.add_argument(
        "--position-admittance-rotation-active",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply one admitted rotational admittance envelope to the explicitly "
            "selected Active side authority. Cautious rotation remains isolated by default; the "
            "bounded 5-degree profiles require explicit combined normal-parity "
            "authority. All ordinary IK, collision, qref, endpoint, and "
            "force-stop gates remain active. Default off."
        ),
    )
    parser.add_argument(
        "--position-admittance-combined-active",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Explicitly authorize translation and rotation admittance together "
            "on the selected Active side authority. The exact envelope/profile pair is "
            "validated by the production installer. Requires both Active flags; "
            "without this flag, cautious rotation retains its established "
            "translation-disabled behavior. Default off."
        ),
    )
    parser.add_argument(
        "--position-admittance-rotation-active-profile",
        choices=POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE_CHOICES,
        default=POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE,
        help=(
            "Named production rotation Active profile. cautious-3deg is the "
            "default; balanced-5deg and rotation-emphasis-5deg are admitted "
            "only with combined normal-parity translation."
        ),
    )
    parser.add_argument(
        "--position-admittance-active-side",
        choices=POSITION_ADMITTANCE_ACTIVE_SIDE_CHOICES,
        default=None,
        help=(
            "Explicit side authority for the active overlay. left/right preserve "
            "the staged single-side paths; both is admitted only for the exact "
            "formal_teleop combined normal-parity translation/rotation profile. "
            "Required with --position-admittance-active; shadow observes both sides."
        ),
    )
    parser.add_argument(
        "--position-admittance-active-envelope",
        choices=POSITION_ADMITTANCE_ACTIVE_ENVELOPE_CHOICES,
        default=POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENVELOPE,
        help=(
            "Explicit production Active envelope. cautious remains the default; "
            "half_normal requires its audited 60 mm/s, return-rate 2 release "
            "brake; normal_parity requires its audited 420 mm/s, return-rate 4 "
            "release brake and is restricted to an admitted combined 5-degree "
            "rotation Active profile. Both faster rates are return-only final "
            "bounds and never expand outward wrench motion."
        ),
    )
    parser.add_argument(
        "--position-admittance-active-endpoint-policy",
        choices=POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_CHOICES,
        default=POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
        help=(
            "Final tool endpoint policy for Active admittance. runtime_home_hold "
            "keeps the temporary test envelope around runtime Home. formal_teleop "
            "removes only that Home-relative test envelope and retains the ordinary "
            "formal teleop workspace, collision, joint, dynamics, force-stop, and "
            "bounded-modifier safety. formal_teleop is admitted only for exact "
            "combined normal-parity translation and 5-degree rotation."
        ),
    )
    parser.add_argument(
        "--position-admittance-active-max-tool-excursion-m",
        type=_positive_float,
        default=POSITION_ADMITTANCE_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_M,
        help=(
            "Maximum final calibrated tool-TCP qref distance from runtime Home "
            "when the endpoint policy is runtime_home_hold. A larger candidate "
            "holds the last safe qref and requests the ordinary GUI Hold/Home flow."
        ),
    )
    parser.add_argument(
        "--position-admittance-active-max-tool-rotation-excursion-deg",
        type=_positive_float,
        default=(
            POSITION_ADMITTANCE_ROTATION_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_DEG
        ),
        help=(
            "Maximum final calibrated tool-TCP qref rotation from runtime Home "
            "during rotation Active when the endpoint policy is runtime_home_hold. "
            "A larger candidate holds the last safe qref and requests the ordinary "
            "GUI Hold/Home flow."
        ),
    )
    parser.add_argument(
        "--position-admittance-shadow-axes",
        type=str,
        default="xyz",
        help="Translation axes used by shadow/active admittance; non-empty subset of xyz.",
    )
    parser.add_argument(
        "--position-admittance-response-model",
        choices=("second_order", "profile_compliance"),
        default="second_order",
        help=(
            "Response law for the default-off shadow/active modifier. "
            "second_order preserves the initial research implementation; "
            "profile_compliance reproduces the established filtered "
            "compliance-target and passive-return semantics."
        ),
    )
    parser.add_argument(
        "--position-admittance-translation-response-weight",
        type=_positive_float,
        default=POSITION_ADMITTANCE_RESPONSE_WEIGHT_DEFAULT,
        help=(
            "Translation compliance multiplier relative to the accepted 250 N/m "
            "profile. 1.0 preserves accepted behavior; 2.0 doubles the "
            "unconstrained offset for the same force. Final offset, velocity, "
            "slew, IK, collision, and force-stop limits are unchanged."
        ),
    )
    parser.add_argument(
        "--position-admittance-rotation-response-weight",
        type=_positive_float,
        default=POSITION_ADMITTANCE_RESPONSE_WEIGHT_DEFAULT,
        help=(
            "Rotation compliance multiplier relative to the selected accepted "
            "profile. 1.0 preserves accepted behavior; larger values emphasize "
            "rotation without changing the profile's final rotation/slew bounds."
        ),
    )
    parser.add_argument(
        "--position-admittance-shadow-mass-kg",
        type=_positive_float,
        default=2.0,
        help="Isotropic virtual translation mass used by shadow/active admittance.",
    )
    parser.add_argument(
        "--position-admittance-shadow-damping-n-s-m",
        type=_non_negative_float,
        default=50.0,
        help=(
            "Isotropic physical damping used only by the second_order research "
            "model. It is intentionally inactive in the accepted "
            "profile_compliance path; use the translation/rotation filter tau "
            "options for that path's effective input damping."
        ),
    )
    parser.add_argument(
        "--position-admittance-shadow-stiffness-n-m",
        type=_non_negative_float,
        default=250.0,
        help=(
            "Isotropic virtual translation stiffness. The 250 N/m default "
            "matches the established DMTAC_M 0.004 m/N compliance scale."
        ),
    )
    parser.add_argument(
        "--position-admittance-shadow-deadband-n",
        type=_non_negative_float,
        default=0.45,
        help="Per-axis force deadband for shadow/active admittance.",
    )
    parser.add_argument(
        "--position-admittance-shadow-filter-tau-sec",
        "--position-admittance-translation-filter-tau-sec",
        dest="position_admittance_shadow_filter_tau_sec",
        type=_non_negative_float,
        default=0.022,
        help=(
            "Translation force-input damping/filter time constant; the production-"
            "neutral option name is an alias for the historical shadow name. "
            "Larger values are smoother and slower. For exact DMTAC_M 0.9-at-40-Hz "
            "pole parity use approximately 0.010857362 s."
        ),
    )
    parser.add_argument(
        "--position-admittance-rotation-filter-tau-sec",
        type=_non_negative_float,
        default=POSITION_ADMITTANCE_ACCEPTED_FILTER_TAU_SEC,
        help=(
            "Rotation torque-input damping/filter time constant. Larger values "
            "are smoother and slower; the default preserves the accepted profile."
        ),
    )
    parser.add_argument(
        "--position-admittance-translation-output-follow-tau-sec",
        type=_non_negative_float,
        default=0.0,
        help=(
            "Optional 60 Hz first-order follower on the already bounded "
            "translation modifier. Zero preserves exact accepted behavior; "
            "a small positive value smooths 20 Hz wrench sample-and-hold "
            "without changing input filtering, IK, or final offset/slew bounds."
        ),
    )
    parser.add_argument(
        "--position-admittance-rotation-output-follow-tau-sec",
        type=_non_negative_float,
        default=0.0,
        help=(
            "Optional 60 Hz first-order follower on the already bounded "
            "rotation modifier. Zero preserves exact accepted behavior."
        ),
    )
    parser.add_argument(
        "--position-admittance-translation-max-output-acceleration-m-s2",
        type=_non_negative_float,
        default=0.0,
        help=(
            "Optional acceleration limit for the 60 Hz translation modifier "
            "output velocity. Zero preserves the accepted follower exactly."
        ),
    )
    parser.add_argument(
        "--position-admittance-rotation-max-output-angular-acceleration-rad-s2",
        type=_non_negative_float,
        default=0.0,
        help=(
            "Optional angular-acceleration limit for the 60 Hz rotation "
            "modifier output velocity. Zero preserves accepted behavior."
        ),
    )
    parser.add_argument(
        "--position-admittance-shadow-max-offset-m",
        type=_positive_float,
        default=0.005,
        help="Per-axis translation-offset limit (hypothetical in shadow mode).",
    )
    parser.add_argument(
        "--position-admittance-shadow-max-velocity-m-s",
        type=_positive_float,
        default=0.01,
        help="Per-axis offset-velocity limit (hypothetical in shadow mode).",
    )
    parser.add_argument(
        "--position-admittance-shadow-max-return-velocity-m-s",
        type=_positive_float,
        default=0.01,
        help=(
            "Passive return velocity limit for profile_compliance. It is "
            "independent from the active offset-velocity limit."
        ),
    )
    parser.add_argument(
        "--position-admittance-shadow-max-slew-m-s",
        type=_positive_float,
        default=0.01,
        help="60 Hz output-slew limit (hypothetical in shadow mode).",
    )
    parser.add_argument(
        "--position-admittance-shadow-return-deadband-n",
        type=_non_negative_float,
        default=0.45,
        help=(
            "Filtered-force activity threshold that pauses passive return in "
            "profile_compliance mode."
        ),
    )
    parser.add_argument(
        "--position-admittance-shadow-return-rate-s",
        type=_non_negative_float,
        default=1.0,
        help="Passive zero-force offset return rate for profile_compliance.",
    )
    parser.add_argument(
        "--position-admittance-response-reference-hz",
        type=_positive_float,
        default=40.0,
        help=(
            "Reference rate whose discrete passive-return dynamics are "
            "preserved when wrench samples arrive at another rate."
        ),
    )
    parser.add_argument(
        "--position-admittance-shadow-stale-sec",
        type=_positive_float,
        default=0.15,
        help="Admittance freshness limit; does not replace the force guard.",
    )
    parser.add_argument(
        "--position-admittance-shadow-max-force-dt-sec",
        type=_positive_float,
        default=0.15,
        help="Maximum new-wrench integration interval used by the admittance state.",
    )
    parser.add_argument(
        "--strict-tool-profile",
        dest="strict_tool_profile",
        action="store_true",
        help="Require active RM65 tool, confirmed TCP, payload, and impedance profile before arming. Default.",
    )
    parser.add_argument(
        "--no-strict-tool-profile",
        dest="strict_tool_profile",
        action="store_false",
        help="Allow incomplete tool profiles for lab debugging. Not recommended for real teleop.",
    )
    parser.set_defaults(strict_tool_profile=True)
    parser.add_argument(
        "--auto-sync-tool-frame",
        dest="auto_sync_tool_frame",
        action="store_true",
        help=(
            "Before impedance arming, write/create and activate the RM65 controller tool frame "
            "from the selected calibration profile. Default."
        ),
    )
    parser.add_argument(
        "--no-auto-sync-tool-frame",
        dest="auto_sync_tool_frame",
        action="store_false",
        help="Disable automatic RM65 controller tool-frame sync for impedance debugging.",
    )
    parser.set_defaults(auto_sync_tool_frame=True)
    parser.add_argument(
        "--left-tool-profile",
        type=str,
        default="",
        help=(
            "Optional explicit left RM65 calibration/impedance profile name or JSON path. "
            "Use position_endpoint with --arm-control-mode position for legacy endpoint-TCP position control."
        ),
    )
    parser.add_argument(
        "--right-tool-profile",
        type=str,
        default="",
        help=(
            "Optional explicit right RM65 calibration/impedance profile name or JSON path. "
            "Use position_endpoint with --arm-control-mode position for legacy endpoint-TCP position control."
        ),
    )
    parser.add_argument(
        "--tcp-mode",
        type=str,
        default=TOOL_TCP_FRAME,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--movement-multiple",
        type=_movement_multiple_arg,
        default=None,
        help=(
            "倍率 for VIVE tracker translation around runtime home. Rotation is unchanged; "
            "robot target translation = robot_home + movement_multiple * (tracker - tracker_home)."
        ),
    )
    parser.add_argument(
        "--translation-scale",
        type=_movement_multiple_arg,
        default=MOVEMENT_MULTIPLE_DEFAULT,
        help=argparse.SUPPRESS,
    )
    add_vive_tracker_height_correction_args(parser)
    parser.add_argument(
        "--virtual-desk-tracker-role",
        choices=["auto", "auxiliary", "left_wrist", "right_wrist"],
        default="auto",
        help=(
            "Tracker role used by the GUI VIVE coordinate calibration sweeps. "
            "auto asks in the GUI and defaults to right_wrist; auxiliary explicitly skips the hand prompt."
        ),
    )
    parser.add_argument(
        "--virtual-desk-post-yaw-deg",
        type=float,
        default=180.0,
        help=(
            "Post-rotation about virtual desk +Z after GUI VIVE coordinate calibration. "
            "Default 180 matches the standalone calibration GUI/SAPIEN scene convention."
        ),
    )
    parser.add_argument(
        "--sweep-capture-sec",
        type=_positive_float,
        default=5.0,
        help="Timed capture duration for each GUI VIVE coordinate X/Y/Z sweep.",
    )
    parser.add_argument(
        "--capture-prep-sec",
        type=_non_negative_float,
        default=3.0,
        help="Preparation countdown before each GUI VIVE coordinate capture starts.",
    )
    parser.add_argument(
        "--capture-extra-wait-sec",
        type=_non_negative_float,
        default=0.0,
        help="Extra wait time if GUI VIVE coordinate capture ends before enough fresh samples arrive.",
    )
    parser.add_argument(
        "--min-sweep-extent-m",
        type=_positive_float,
        default=0.15,
        help="Minimum motion extent required for each GUI VIVE coordinate sweep.",
    )
    parser.add_argument(
        "--min-axis-variance-ratio",
        type=float,
        default=0.5,
        help="Minimum PCA variance ratio required for each GUI VIVE coordinate sweep.",
    )
    parser.add_argument(
        "--min-start-end-distance-m",
        type=_positive_float,
        default=0.08,
        help="Minimum start/end separation required for each GUI VIVE coordinate sweep.",
    )
    parser.add_argument(
        "--max-axis-non-orthogonality-deg",
        type=_positive_float,
        default=20.0,
        help="Maximum allowed axis correction when fitting GUI VIVE coordinate sweeps.",
    )
    parser.add_argument(
        "--min-sweep-samples",
        type=int,
        default=3,
        help="Minimum valid fresh samples for each GUI VIVE coordinate sweep.",
    )
    parser.add_argument(
        "--wrist-capture-prep-sec",
        type=_non_negative_float,
        default=5.0,
        help="Preparation countdown before GUI wrist tracker offset capture.",
    )
    parser.add_argument(
        "--hold-capture-sec",
        type=_positive_float,
        default=4.0,
        help="Timed capture duration for GUI wrist tracker offset capture.",
    )
    parser.add_argument(
        "--min-hold-samples",
        type=int,
        default=3,
        help="Minimum valid samples for each side during GUI wrist tracker offset capture.",
    )
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
            "proxy geometry to both wrists and a fixed table box for collision."
        ),
    )
    parser.add_argument(
        "--rm65-ik-framework",
        choices=["robot_motion"],
        default="robot_motion",
        help=(
            "RM65 IK implementation used by real/sim arm backends and dual-arm collision IK. "
            "robot_motion routes all shared RM65 IK construction and solving through robot_motion.rm65."
        ),
    )
    parser.add_argument(
        "--dual-arm-temporary-safety",
        dest="dual_arm_temporary_safety",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable a temporary conservative output limiter for initial real dual-arm bring-up. "
            "Use --no-dual-arm-temporary-safety to remove it without editing code."
        ),
    )
    parser.add_argument(
        "--dual-arm-temp-max-xyz-step-m",
        type=_positive_float,
        default=0.03,
        help="Temporary dual-arm safety max per-frame TCP translation step in meters.",
    )
    parser.add_argument(
        "--dual-arm-temp-max-rot-step-deg",
        type=_positive_float,
        default=8.0,
        help="Temporary dual-arm safety max per-frame TCP rotation step in degrees.",
    )
    parser.add_argument(
        "--dual-arm-temp-max-joint-step-deg",
        type=_positive_float,
        default=6.0,
        help="Temporary dual-arm safety max per-frame joint-vector step in degrees.",
    )
    parser.add_argument(
        "--dual-arm-temp-command-scale",
        type=_alpha_float,
        default=1.0,
        help="Temporary dual-arm safety scale applied before max-step clipping.",
    )
    parser.add_argument(
        "--hz",
        type=_positive_float,
        default=DEFAULT_TELEOP_SAMPLE_HZ,
        help=(
            "Dataset sample/save rate. Defaults to 15 Hz and remains independent from "
            "the impedance teleop/IK rate."
        ),
    )
    parser.add_argument(
        "--impedance-teleop-control-hz",
        type=_positive_float,
        default=None,
        help=(
            "Force-teleop outer command and dual-arm IK rate. Defaults to 30 Hz when "
            "--hz is left at its default; use any positive value. An explicit legacy --hz "
            "without this flag keeps its historical coupled force-control rate."
        ),
    )
    parser.add_argument(
        "--position-reference-hz",
        type=_positive_float,
        default=DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
        help="Position-mode IK/safety/qpos reference rate. Defaults to 60 Hz.",
    )
    parser.add_argument(
        "--position-reference-max-age-sec",
        type=_positive_float,
        default=0.2,
        help=(
            "Maximum causal qpos-reference age accepted by 15 Hz position-control collection. "
            "Missing or older references drop the training row."
        ),
    )
    parser.add_argument(
        "--position-command-planner",
        choices=POSITION_COMMAND_PLANNER_CHOICES,
        default=POSITION_COMMAND_PLANNER_LINEAR,
        help=(
            "Final position-command planner. 'linear' preserves the existing pure-position "
            "reference interpolation; 'acceleration_limited' adds velocity-continuous, "
            "acceleration-limited online 60->120 Hz planning; "
            "'sample_aligned_jerk_limited' is the opt-in 15 Hz admitted-action / "
            "120 Hz jerk-limited cached-control-cut experiment."
        ),
    )
    parser.add_argument(
        "--position-execution-qvel-feedback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt in to synchronizing coupled position IK from the authoritative "
            "planner qpos and its matching qvel. By default, accepted qcmd still "
            "anchors IK position but planner velocity is not read or fed back."
        ),
    )
    parser.add_argument(
        "--position-velocity-ramp-sec",
        type=_positive_float,
        default=DEFAULT_POSITION_VELOCITY_RAMP_SEC,
        help=(
            "Position-error feedback horizon for acceleration_limited, or the acceleration-"
            "to-jerk ramp used by sample_aligned_jerk_limited. Defaults to 0.10 s; "
            "acceleration is limited independently by --position-max-acceleration-rad-s2."
        ),
    )
    parser.add_argument(
        "--position-target-velocity-filter-tau-sec",
        type=_non_negative_float,
        default=DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
        help=(
            "Time constant in seconds for the bounded causal low-pass applied to "
            "target velocity inferred from accepted position qrefs. Defaults to 0.03 s; "
            "set 0 to reproduce the unfiltered planner behavior. The option is consumed "
            "only by the acceleration-limited position planner; the sample-aligned jerk "
            "planner requires zero and does not apply a second target filter."
        ),
    )
    parser.add_argument(
        "--position-degraded-speed-scale",
        type=_alpha_float,
        default=DEFAULT_POSITION_DEGRADED_SPEED_SCALE,
        help=(
            "Desired speed scale for valid but anomalous position targets. Defaults to 0.50; "
            "manual/collision/no-valid-solution holds use immediate software quick stop instead."
        ),
    )
    parser.add_argument(
        "--position-max-acceleration-rad-s2",
        type=_parse_arm_max_angvel,
        default=DEFAULT_POSITION_MAX_ACCELERATION_RAD_S2,
        help=(
            "Scalar or 6 comma-separated acceleration limits for the optional position "
            "trajectory planner in rad/s^2. Defaults to 7,7,9,15,15,20 and is not "
            "scaled down when --position-degraded-speed-scale is active."
        ),
    )
    parser.add_argument(
        "--position-max-command-step-deg",
        type=_positive_float,
        default=None,
        help=(
            "Optional uniform per-joint command-step envelope, in degrees, applied by the "
            "high-rate acceleration-limited position planner before the final follow EMA. "
            "It is independent of --arm-max-angvel; omit it to retain the existing "
            "max_angvel * command_dt envelope. Position mode only."
        ),
    )
    parser.add_argument(
        "--arm-control-hz",
        type=_positive_float,
        default=None,
        help=(
            "RM65 backend control/streaming rate after each outer action update. "
            "Defaults to 40 Hz for impedance and 120 Hz for position control."
        ),
    )
    parser.add_argument(
        "--arm-max-angvel",
        type=_parse_arm_max_angvel,
        default=(0.7, 0.7, 0.9, 1.5, 1.5, 2.0),
        help="Scalar or 6 comma-separated arm joint velocity limits in rad/s.",
    )
    parser.add_argument(
        "--minimum-control-hz",
        type=_positive_float,
        default=15.0,
        help=(
            "Report low-control-rate faults below this threshold. If omitted, real teleop uses "
            "min(15, 0.9 * effective outer control Hz) so normal scheduler jitter is not flagged."
        ),
    )
    parser.add_argument(
        "--status-sec",
        type=_non_negative_float,
        default=0.0,
        help=(
            "Terminal status heartbeat interval in seconds. Use 0 (the default) to print "
            "only state changes, faults, and lifecycle events."
        ),
    )
    parser.add_argument(
        "--status-change-min-sec",
        type=_non_negative_float,
        default=1.0,
        help=(
            "Minimum interval between change-driven [RealTeleop] diagnostic lines "
            "(default: 1.0). State-machine, fault, collision, Hold, and device-health "
            "changes bypass this coalescing interval. Use 0 for legacy every-change output."
        ),
    )
    parser.add_argument(
        "--debug-orientation-console",
        action="store_true",
        help=(
            "Print periodic raw/retargeted/preview wrist-orientation diagnostics while not "
            "recording. Disabled by default to keep real fault output visible."
        ),
    )
    parser.add_argument(
        "--no-debug-disk-write",
        "--disable-debug-disk-write",
        dest="no_debug_disk_write",
        action="store_true",
        help=(
            "Disable all standalone debug disk outputs for this run: control/action/timing/IK "
            "sidecars, actuator traces, legacy debug session JSONL, fix-arm pose snapshots, "
            "and lightweight debug trajectories. Console/timing instrumentation and formal "
            "dataset collection remain available. This switch overrides individual "
            "debug-write flags."
        ),
    )
    parser.add_argument(
        "--control-debug",
        choices=CONTROL_DEBUG_LEVELS,
        default="off",
        help=(
            "Observational runtime control-audit level: off, summary, or full. It never changes "
            "safety, smoothing, hold, IK, or dispatch behavior. Full is a heavyweight forensic "
            "trace; use --actuator-trace by itself for real2sim actuator identification."
        ),
    )
    parser.add_argument(
        "--control-debug-root",
        type=str,
        default=str(DEFAULT_CONTROL_DEBUG_ROOT),
        help="Output root for runtime control-audit records.",
    )
    parser.add_argument(
        "--control-debug-queue-size",
        type=_positive_int,
        default=DEFAULT_CONTROL_DEBUG_QUEUE_SIZE,
        help="Bounded queue capacity for non-blocking control-audit records.",
    )
    parser.add_argument(
        "--control-debug-flush-every",
        type=_positive_int,
        default=DEFAULT_CONTROL_DEBUG_FLUSH_EVERY,
        help="Control-audit writer flush interval in rows.",
    )
    parser.add_argument(
        "--control-debug-episode-sidecar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Attach a second episode-scoped position audit during formal collection. "
            "Disable it to retain session audit data without duplicate JSON work."
        ),
    )
    parser.add_argument(
        "--actuator-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record post-SDK RM65 qcmd/result/timestamps, 60 Hz IK shadow, and pre-filter "
            "VIVE pose/OpenVR velocity metadata into preallocated memory, then write once "
            "after command workers stop. Intended for low-disturbance actuator and source "
            "identification; do not combine with --control-debug summary/full."
        ),
    )
    parser.add_argument(
        "--actuator-trace-root",
        type=str,
        default=str(DEFAULT_ACTUATOR_TRACE_ROOT),
        help="Output root for low-overhead real-arm actuator command traces.",
    )
    parser.add_argument(
        "--actuator-trace-max-duration-sec",
        type=_positive_float,
        default=DEFAULT_ACTUATOR_TRACE_MAX_DURATION_SEC,
        help=(
            "Preallocated trace duration per arm. Longer runs continue normally but excess "
            "debug samples are counted and dropped instead of growing memory or writing disk."
        ),
    )
    parser.add_argument(
        "--actuator-singularity-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --actuator-trace is enabled, also capture sparse singularity transitions "
            "and every non-normal singular frame in the same preallocated NPZ. Disable with "
            "--no-actuator-singularity-trace for exact qcmd-only rollback."
        ),
    )
    parser.add_argument(
        "--debug-timing",
        "--debug-frozen",
        dest="debug_timing",
        action="store_true",
        help=(
            "Compatibility switch that enables all timing debug modules: console slow-frame "
            "reports, JSONL timing log, RM65 backend timing, IK profiling, and the main-loop "
            "watchdog. Prefer the narrower --debug-timing-* flags for routine diagnosis."
        ),
    )
    parser.add_argument(
        "--debug-timing-console",
        action="store_true",
        help=(
            "Print slow-frame real-loop phase/latency reports without enabling timing "
            "JSONL or backend profiling."
        ),
    )
    parser.add_argument(
        "--debug-timing-log",
        "--debug-timing-light-log",
        dest="debug_timing_log",
        action="store_true",
        help=(
            "Write sampled real-loop timing JSONL frames under the selected artifact "
            "layout (legacy: teleop_records/debug_timing)."
        ),
    )
    parser.add_argument(
        "--no-debug-timing-auto-log",
        "--no-debug-timing-log",
        dest="debug_timing_auto_log",
        action="store_false",
        help=(
            "Do not automatically start the JSONL timing log for narrow timing modules "
            "such as --debug-timing-ik or --debug-timing-rm65. Explicit --debug-timing-log "
            "and legacy --debug-timing still write the JSONL log."
        ),
    )
    parser.add_argument(
        "--debug-timing-rm65",
        "--rm65-backend-debug-timing",
        dest="debug_timing_rm65",
        action="store_true",
        help=(
            "Enable RM65 impedance backend timing instrumentation and start the sampled "
            "JSONL timing log unless --no-debug-timing-auto-log is set."
        ),
    )
    parser.add_argument(
        "--debug-action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record extra action pose source/timing diagnostics to an independent JSONL log. "
            "When formal collection is active, the same diagnostics are also added to sample_metadata_json."
        ),
    )
    parser.add_argument(
        "--debug-action-log-root",
        type=str,
        default=str(DEFAULT_ACTION_DEBUG_LOG_ROOT),
        help="Directory for the standalone --debug-action JSONL log.",
    )
    parser.add_argument(
        "--debug-action-flush-every",
        type=_positive_int,
        default=1,
        help="Flush the standalone --debug-action JSONL log every N frames.",
    )
    parser.add_argument(
        "--debug-action-queue-size",
        type=_positive_int,
        default=2048,
        help="Maximum queued standalone --debug-action rows before frames are dropped.",
    )
    parser.add_argument(
        "--debug-action-sample-hz",
        type=_non_negative_float,
        default=0.0,
        help=(
            "Maximum standalone action-debug frame rate. 0 preserves every-outer-loop "
            "recording; formal dataset sampling keeps its own sample clock."
        ),
    )
    parser.add_argument(
        "--debug-timing-ik",
        dest="debug_timing_ik",
        action="store_true",
        help=(
            "Enable IK timing/profile collection as a timing debug module. "
            "This starts the sampled JSONL timing log unless --no-debug-timing-auto-log "
            "is set. --debug-ik-timing remains available for GUI/status-only IK diagnostics."
        ),
    )
    parser.add_argument(
        "--debug-ik-0827",
        action="store_true",
        help=(
            "Explicitly opt in to the 2026-08-27 IK/jitter investigation log. "
            "The runtime writes an additional bounded asynchronous JSONL control-chain "
            "record (including loop phases, VIVE channel ages, IK timing, and RM65 "
            "backend/CANFD timing) without changing the default control, safety, "
            "writer, progress, or formal-collection behavior."
        ),
    )
    parser.add_argument(
        "--debug-timing-watchdog",
        action="store_true",
        help="Enable the background main-loop stall watchdog without enabling other timing logs.",
    )
    parser.add_argument(
        "--debug-timing-threshold-sec",
        "--debug-frozen-threshold-sec",
        dest="debug_timing_threshold_sec",
        type=_positive_float,
        default=1.0 / DEFAULT_TELEOP_SAMPLE_HZ,
        help="Slow-frame threshold for --debug-timing diagnostics; default matches one outer action period.",
    )
    parser.add_argument(
        "--debug-timing-report-interval-sec",
        "--debug-frozen-report-interval-sec",
        dest="debug_timing_report_interval_sec",
        type=_positive_float,
        default=1.0,
        help="Minimum seconds between repeated --debug-timing reports.",
    )
    parser.add_argument(
        "--debug-timing-log-sample-hz",
        type=_non_negative_float,
        default=0.0,
        help=(
            "Maximum routine timing-JSONL frame rate. 0 preserves unbounded slow-frame "
            "logging; state and fault transitions always bypass this limit."
        ),
    )
    parser.add_argument(
        "--debug-ik-timing",
        action="store_true",
        help=(
            "Enable IK timing diagnostics in status metadata and the browser GUI. "
            "This turns on the dual-arm retargeter IK profile without enabling full loop timing logs."
        ),
    )
    parser.add_argument(
        "--debug-ik-timing-print-every",
        type=_non_negative_int,
        default=0,
        help=(
            "Print one dual-arm IK timing profile every N retargeter frames when --debug-ik-timing is enabled. "
            "0 keeps the diagnostics in status/GUI only."
        ),
    )
    parser.add_argument(
        "--ik-debug-log-full",
        action="store_true",
        help=(
            "Keep the GUI IK debug log in the historical full-detail format. "
            "By default the log is compact to avoid adding main-loop latency while diagnosing teleop lag."
        ),
    )
    parser.add_argument(
        "--ik-time-budget-ms",
        type=_non_negative_float,
        default=None,
        help=(
            "Optional IK frame time budget in milliseconds. Omit to use one control period; "
            "pass 0 to disable over-budget warnings."
        ),
    )
    parser.add_argument(
        "--position-ik-orientation-weight",
        type=_positive_finite_float_arg,
        default=None,
        help=(
            "Position mode only: override both single- and dual-arm IK orientation "
            "weights while fixing the corresponding position weights at 1.0. "
            "Omit to preserve the selected IK tuning JSON."
        ),
    )
    parser.add_argument(
        "--position-ik-solve-max-iterations",
        "--ik-solve-max-iterations",
        "--solve-max-iterations",
        dest="position_ik_solve_max_iterations",
        type=_positive_int,
        default=2,
        help=(
            "Position-mode dual-arm IK solve iteration upper bound (default 2). "
            "This option never changes the force/impedance IK path."
        ),
    )
    parser.add_argument(
        "--position-ik-step-mode",
        choices=IK_STEP_MODE_CHOICES,
        default=IK_STEP_MODE_LEGACY_ITERATIVE,
        help=(
            "Position-mode IK time semantics. legacy_iterative preserves the current "
            "min/max repeated integration. single_step_singularity_aware advances one "
            "physical IK step per reference tick, enables admitted-qdot acceleration "
            "state, and suppresses only the RM65 q4-q6 coaxial null motion. Switching "
            "back to legacy_iterative is the atomic rollback."
        ),
    )
    parser.add_argument(
        "--position-ik-admitted-motion-limit",
        dest="position_ik_admitted_motion_limit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Position single-step IK only: put the formal qref velocity and "
            "acceleration interval directly into the Pink QP, anchored to the "
            "last formally admitted qref/qdot. Disabled by default; "
            "--no-position-ik-admitted-motion-limit is the atomic rollback."
        ),
    )
    parser.add_argument(
        "--position-ik-branch-continuity-anchor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Position single-step IK only: add the experimental q3/q5 soft "
            "branch-continuity task. Disabled by default because it changes "
            "the stable raw IK solution; --no-position-ik-branch-continuity-anchor "
            "is the atomic rollback."
        ),
    )
    parser.add_argument(
        "--position-joint-limit-guard-mode",
        choices=JOINT_LIMIT_GUARD_MODE_CHOICES,
        default=JOINT_LIMIT_GUARD_LEGACY,
        help=(
            "Position-mode physical joint-limit behavior. legacy preserves "
            "the stable selector; boundary_hold removes only inward motion "
            "inside the narrow guard band; forced_escape_latched retains the "
            "2026-08-02 outward-step experiment for offline comparison."
        ),
    )
    parser.add_argument(
        "--position-joint-limit-boundary-margin-deg",
        type=_non_negative_float,
        default=1.5,
        help=(
            "Guard width used only by --position-joint-limit-guard-mode "
            "boundary_hold (default 1.5 degrees)."
        ),
    )
    parser.add_argument(
        "--position-ik-solve-min-iterations",
        type=_positive_int,
        default=2,
        help=(
            "Position-mode dual-arm IK hard minimum iteration count (default 2). "
            "The time budget may stop only later optional iterations; force/impedance remains at 1."
        ),
    )
    parser.add_argument(
        "--position-ik-damping",
        type=_positive_finite_float_arg,
        default=0.03,
        help=(
            "Position-mode Pink QP damping (default 0.03). "
            "Force/impedance keeps the historical 0.01."
        ),
    )
    parser.add_argument(
        "--position-ik-adaptive-iterations",
        dest="position_ik_adaptive_iterations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Position mode only: when enabled, run up to --position-ik-solve-max-iterations "
            "while the IK frame budget still has time after the safety-selection reserve."
        ),
    )
    parser.add_argument(
        "--position-ik-iteration-reserve-ms",
        type=_non_negative_float,
        default=5.0,
        help=(
            "Position adaptive-iteration time reserved for candidate/safety selection after "
            "IK solving. Defaults to 5 ms, matching the existing candidate-selection budget."
        ),
    )
    parser.add_argument(
        "--position-ik-candidate-certification-reserve-ms",
        type=_non_negative_float,
        default=0.0,
        help=(
            "Position mode only: portion of the total IK iteration reserve dedicated to "
            "base-candidate step/collision certification. The formal profile uses 3.5 ms; "
            "set 0 to restore the historical candidate deadline split."
        ),
    )
    parser.add_argument(
        "--position-ik-local-step-clip-scale",
        "--position-ik-local-step-scale",
        dest="position_ik_local_step_clip_scale",
        type=_local_step_clip_scale_arg,
        default=1.0,
        help=(
            "Position mode only: multiplier for the candidate-local IK step clip used "
            "during safe-branch selection. Default 1 preserves behavior; the final qref "
            "execution velocity cap is unchanged."
        ),
    )
    parser.add_argument(
        "--position-prefer-non-wrist-singular-branch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Position mode only: prefer a feasible IK candidate with fewer RM65 wrist "
            "singular sides. If none exists, accept the best singular candidate at normal "
            "speed. Use --no-position-prefer-non-wrist-singular-branch to atomically "
            "restore the previous branch ordering and backend hold-entry gate."
        ),
    )
    parser.add_argument(
        "--position-wrist-branch-switch-guard-deg",
        type=_non_negative_float,
        default=12.0,
        help=(
            "Position single-step mode only: suppress +/-pi wrist-flip "
            "candidates while either the previous or raw q5 is within this "
            "guard band (default 12 deg). Set 0 to restore historical branch "
            "selection; the final execution step limit remains unchanged."
        ),
    )
    parser.add_argument(
        "--position-wrist-q5-hysteresis-release-deg",
        type=_non_negative_float,
        default=6.0,
        help=(
            "Position single-step mode only: keep an already escaped q5 on its "
            "current side when an inward IK candidate remains inside this release "
            "band (default 6 deg). Set 0 to disable this singularity-only hysteresis."
        ),
    )
    parser.add_argument(
        "--position-prefer-non-elbow-singular-branch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Position mode only: prefer collision-safe candidates outside the RM65 "
            "elbow-singularity neighborhood. If no non-singular candidate is feasible "
            "within the frame budget, explicitly allow the selected elbow-singular "
            "candidate through the backend SDK gate. Use "
            "--no-position-prefer-non-elbow-singular-branch to restore the previous "
            "hard elbow hold-entry behavior."
        ),
    )
    parser.add_argument(
        "--position-prefer-non-shoulder-singular-branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Position mode only: use the RM65 kinematics analyzer to prefer a "
            "collision-safe candidate outside the shoulder/coaxial singularity "
            "region. The formal rm65_bimanual_position_v1 profile enables it. "
            "If every evaluated feasible candidate is singular, certify and "
            "pass the selected safe candidate instead of entering hold. Use "
            "--no-position-prefer-non-shoulder-singular-branch for atomic rollback "
            "to the previous shoulder hold-entry behavior."
        ),
    )
    parser.add_argument(
        "--position-singularity-candidate-certificate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Position mode only: require singular fallback metadata to match the exact "
            "final qref fingerprint emitted by the coupled candidate selector. The formal "
            "profile enables this; use --no-position-singularity-candidate-certificate "
            "to restore the previous metadata-only fallback gate."
        ),
    )
    parser.add_argument(
        "--position-inactive-enter-threshold-scale",
        type=_positive_unit_scale_arg,
        default=1.0,
        help=(
            "Position mode only: tighten tracker INACTIVE entry evidence thresholds. "
            "1 preserves the current thresholds; 0.5 requires linear/angular speed "
            "and translation/rotation span to be at most half their defaults. "
            "Exit thresholds and dwell times are unchanged."
        ),
    )
    parser.add_argument(
        "--position-joint-velocity-limit-profile",
        type=str,
        default="rm65",
        help=(
            "Position mode only: joint-velocity profile used by the 60 Hz accepted-qref "
            "final execution-step cap. Default rm65 preserves existing behavior. "
            "The rm65_position_2x/4x/8x profiles are explicit position-only test levels; "
            "the per-arm --arm-max-angvel step remains an independent upper bound."
        ),
    )
    parser.add_argument(
        "--dual-arm-ik-hz",
        type=_positive_float,
        default=None,
        help=(
            "Dual-arm IK solver nominal frequency/step budget, not a separate worker clock. "
            "Actual IK calls follow the outer loop. Defaults to --position-reference-hz in "
            "position mode and --impedance-teleop-control-hz in impedance mode; normally keep them equal."
        ),
    )
    parser.add_argument(
        "--left-right-base-distance",
        type=float,
        default=ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
        help="Annotated dual-arm base distance used by the SAPIEN scene and default calibration.",
    )
    parser.add_argument(
        "--arm-workspace-config",
        type=str,
        default=str(DEFAULT_ARM_WORKSPACE_CONFIG_PATH),
        help=(
            "Desk-frame arm workspace JSON config. Defaults to "
            "teleop_runtime/configs/arm_workspace.json; pass 'none' to skip it."
        ),
    )
    parser.add_argument(
        "--arm-workspace-min",
        nargs="+",
        default=None,
        help=(
            "Runtime override for both arms' desk-frame workspace minimum xyz. "
            "Accepts either three values or --arm-workspace-min=x,y,z."
        ),
    )
    parser.add_argument(
        "--arm-workspace-max",
        nargs="+",
        default=None,
        help=(
            "Runtime override for both arms' desk-frame workspace maximum xyz. "
            "Accepts either three values or --arm-workspace-max=x,y,z."
        ),
    )
    parser.add_argument(
        "--left-arm-workspace-min",
        nargs="+",
        default=None,
        help="Runtime override for the left arm desk-frame workspace minimum xyz.",
    )
    parser.add_argument(
        "--left-arm-workspace-max",
        nargs="+",
        default=None,
        help="Runtime override for the left arm desk-frame workspace maximum xyz.",
    )
    parser.add_argument(
        "--right-arm-workspace-min",
        nargs="+",
        default=None,
        help="Runtime override for the right arm desk-frame workspace minimum xyz.",
    )
    parser.add_argument(
        "--right-arm-workspace-max",
        nargs="+",
        default=None,
        help="Runtime override for the right arm desk-frame workspace maximum xyz.",
    )
    parser.add_argument("--left-tracker-name", type=str, default="tracker_1", help="Left wrist tracker device name")
    parser.add_argument("--right-tracker-name", type=str, default="tracker_2", help="Right wrist tracker device name")
    parser.add_argument(
        "--aux-tracker-name",
        type=str,
        default="",
        help="Optional auxiliary tracker name. Empty by default; pass tracker_3 to enable a third tracker.",
    )
    parser.add_argument(
        "--gui",
        dest="open_browser_gui",
        action="store_true",
        help=(
            "Open the browser controls for VIVE tracker swap/start. "
            "SAPIEN rendering stays disabled unless --show-viewer or --no-skip-renderer is passed."
        ),
    )
    parser.add_argument(
        "--gui-resizable-sidebars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow the shared RealTeleop/UMI browser GUI columns to be resized "
            "with a draggable divider. Enabled by default; pass "
            "--no-gui-resizable-sidebars for the fixed legacy layout."
        ),
    )
    parser.add_argument(
        "--gui-input-activity-indicators",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Show passive A/B/C/G and P1--P5 activity blocks above the left "
            "status column. Enabled by default; pass "
            "--no-gui-input-activity-indicators for exact display rollback."
        ),
    )
    parser.add_argument(
        "--hardware-admission-mode",
        choices=("capability", "strict"),
        default="capability",
        help=(
            "capability keeps the GUI and unrelated controls available when optional "
            "camera/hand/collection devices fail; strict restores the legacy all-or-nothing startup."
        ),
    )
    parser.set_defaults(open_browser_gui=True)
    parser.add_argument(
        "--show-viewer",
        dest="headless",
        action="store_false",
        help="Show the SAPIEN real-teleop viewer window; also requires --no-skip-renderer.",
    )
    parser.add_argument(
        "--no-gui",
        "--headless",
        dest="headless",
        action="store_true",
        help="Disable the browser controls and the SAPIEN viewer window.",
    )
    parser.set_defaults(headless=True)
    parser.add_argument(
        "--skip-renderer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip SAPIEN renderer creation. Enabled by default to reduce real-teleop latency.",
    )
    parser.add_argument(
        "--calibration-timeout",
        type=_positive_float,
        default=10.0,
        help=(
            "Overall timeout for runtime Home stability/capture. Entering or "
            "resuming real teleoperation additionally fails if any required "
            "VIVE tracker timestamp stops advancing for 1 second."
        ),
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=20,
        help=(
            "Number of valid wrist frames used per side to validate runtime Home. "
            "On the default real-arm path, the exact anchor is a fresh stable "
            "VIVE packet captured after the atomic arm snapshot."
        ),
    )
    parser.add_argument(
        "--runtime-home-atomic-arm-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use one dual-arm hardware snapshot for runtime robot Home, preview/IK "
            "anchors, and command-backend reset. Enabled by default; "
            "--no-runtime-home-atomic-arm-snapshot restores the legacy reread path."
        ),
    )
    parser.add_argument(
        "--runtime-home-late-weighted-vive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use fresh VIVE packets for runtime Home. With the default atomic "
            "real-arm snapshot, historical packets validate stillness and one "
            "new stable post-snapshot packet becomes the exact human anchor. "
            "The option name is retained for command compatibility. Enabled by "
            "default; --no-runtime-home-late-weighted-vive restores the legacy "
            "uniform latest-value polling average and disables the exact cut."
        ),
    )
    parser.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="Start runtime home calibration immediately without waiting for a key press.",
    )
    parser.add_argument(
        "--keyboard-clutch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--artifact-layout",
        choices=(
            LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
            RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
        ),
        default=configured_runtime_artifact_layout(),
        help=(
            "Select the default storage layout for optional runtime/debug artifacts. "
            "Formal collection datasets are never redirected; explicit per-sink roots win."
        ),
    )
    parser.add_argument(
        "--record",
        dest="no_record",
        action="store_false",
        help=(
            "Enable legacy debug session JSONL recording. This writes raw_inputs/faults/"
            "state_transitions/calibration under --session-record-root when provided, or "
            "the selected artifact layout otherwise."
        ),
    )
    parser.add_argument(
        "--no-record",
        dest="no_record",
        action="store_true",
        help="Disable legacy debug session JSONL recording. This is the default for real teleop.",
    )
    parser.set_defaults(no_record=True)
    parser.add_argument(
        "--no-session-record-commands",
        action="store_true",
        help=(
            "Do not write the synchronous debug commands.jsonl stream. Formal LeRobot dataset "
            "recording is unaffected."
        ),
    )
    parser.add_argument(
        "--session-record-flush-every",
        type=_positive_int,
        default=1,
        help=(
            "Flush debug session JSONL streams every N rows. Default 1 preserves the previous "
            "flush-after-each-row behavior."
        ),
    )
    parser.add_argument(
        "--session-record-compact-json",
        action="store_true",
        help="Write compact debug session JSONL without spaces to reduce serialization and disk volume.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help=(
            "Base output directory for formal LeRobot dataset collection. Actual dataset path is "
            "<dataset-root>/<YYYY-MM-DD>_<collection-task-name>."
        ),
    )
    parser.add_argument(
        "--plus-episode-curation",
        action="store_true",
        help=(
            "Enable the optional Plus episode curation sidecar and GUI panel. "
            "Default off; raw dataset rows and lifecycle are unchanged."
        ),
    )
    parser.add_argument(
        "--plus-annotation-root",
        type=str,
        default="",
        help="External directory for Plus manifest/episode_curation.jsonl (required with --plus-episode-curation).",
    )
    parser.add_argument(
        "--plus-task-profile",
        type=str,
        default="",
        help="Optional finite task-variant profile JSON used by the Plus GUI.",
    )
    parser.add_argument(
        "--plus-source-binding",
        choices=(
            "v23_compat_content_locator_v1",
            "v24_committed_capture_membership_v1",
            "v24_compatibility_content_locator_v1",
        ),
        default="v23_compat_content_locator_v1",
        help="Explicit Plus source binding; it does not infer v2.4 native identity from a path.",
    )
    parser.add_argument(
        "--collection-task-name",
        type=str,
        default="",
        help="Task name used for formal dataset folder naming and metadata.",
    )
    parser.add_argument(
        "--collection-task-name-zh",
        type=str,
        default="",
        help="Optional Chinese task name written into dataset/session metadata and shown in the GUI.",
    )
    parser.add_argument(
        "--collection-task-id",
        type=str,
        default="",
        help="Optional formal dataset task id. Defaults to a slug from --collection-task-name.",
    )
    parser.add_argument(
        "--collection-task-description",
        type=str,
        default="",
        help="Optional formal dataset task description.",
    )
    parser.add_argument(
        "--collection-progress-profile",
        type=load_real_teleop_progress_profile,
        default=None,
        metavar="JSON_OR_CATALOG",
        help=(
            "Opt in to the default-off RealTeleop collection-progress sidecar "
            "using a one-or-more-step LINEAR FULL_TASK profile from an "
            "explicit JSON path or catalog:<profile_id>. The "
            "formal dataset remains authoritative; typed Advance/Undo/Failure "
            "and Failure/Retry wiring are enabled. The independent rendered "
            "panel requires --collection-progress-gui; Tree profiles and UMI "
            "remain disabled."
        ),
    )
    parser.add_argument(
        "--collection-progress-gui",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Render the optional Control Station long-horizon progress panel. "
            "A blank, unbound dataset defaults to the managed four-stage "
            "writing profile, which can be changed in the GUI before the "
            "first Episode; --collection-progress-profile remains an optional "
            "explicit override. Default off; disabling it preserves the "
            "existing RealTeleop collection GUI and A/B/C semantics exactly."
        ),
    )
    parser.add_argument(
        "--collection-progress-arm-control-mode",
        choices=("strict_profile", "profile", "ignore_profile"),
        default="profile",
        help=(
            "How a selected coarse progress profile controls left/right arms. "
            "strict_profile enforces declared arm modes and disables manual "
            "Fix/single-arm pedals; profile (default) applies declarations on "
            "progress transitions but keeps manual controls; ignore_profile "
            "preserves pedal/button-only behavior."
        ),
    )
    parser.add_argument(
        "--collection-stage-profile",
        type=str,
        default="",
        metavar="JSON_OR_CATALOG",
        help=(
            "Optional ordered fine-stage profile for post-capture annotation. "
            "Managed collection progress profiles automatically use their "
            "same-name .stages.json asset; use this option only for an "
            "explicit external profile."
        ),
    )
    parser.add_argument(
        "--collection-progress-entry-step",
        type=str,
        default="",
        help=(
            "Advanced partial/recovery capture entry progress_step_id. Empty "
            "uses the profile initial step."
        ),
    )
    parser.add_argument(
        "--collection-progress-entry-stage",
        type=str,
        default="",
        help=(
            "Advanced partial/recovery capture entry stage_id. Empty uses the "
            "first stage of the selected progress step."
        ),
    )
    parser.add_argument(
        "--collection-progress-entry-condition",
        choices=("nominal", "recovery_from_failure"),
        default="nominal",
        help=(
            "Capture entry condition. Non-initial fine-stage entries require "
            "recovery_from_failure; this records setup semantics but does not "
            "rewrite a prior Episode's Failure fact."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Optional formal dataset name written into meta/info.json.",
    )
    parser.add_argument(
        "--session-record-root",
        type=str,
        default="",
        help=(
            "Optional base directory for legacy debug JSONL session output when --record is set. "
            "Session JSONL is written to <session-record-root>/<timestamp>_vive_rm65_real_teleop; "
            "when omitted, --artifact-layout selects the default."
        ),
    )
    parser.add_argument(
        "--no-collection",
        action="store_true",
        help="Disable formal dataset collection; GUI episode buttons will only use debug trajectory recording.",
    )
    parser.add_argument("--write-parquet", dest="write_parquet", action="store_true")
    parser.add_argument("--no-parquet", dest="write_parquet", action="store_false")
    parser.set_defaults(write_parquet=True)
    parser.add_argument(
        "--v25p-native-shadow",
        action="store_true",
        help=(
            "Opt in to provisional embodiment-neutral v2.5-p native dual-write "
            "and exact v2.3 row parity receipts. Legacy collection remains "
            "authoritative; no v2.4 commit or long-horizon GUI is enabled."
        ),
    )
    parser.add_argument(
        "--v25p-native-storage-layout",
        choices=V25P_NATIVE_STORAGE_LAYOUTS_V1,
        default=V25P_RECORD_CJSON_LAYOUT_V1,
        help=(
            "Physical layout for the explicitly enabled provisional v2.5-p "
            "shadow. Default record_cjson_v1 preserves the current path; "
            "parquet_stream_chunks_v1 enables bounded background chunks."
        ),
    )
    parser.add_argument(
        "--policy-io-layout",
        type=str,
        default="",
        help=(
            "Optional DexTeleop v2.2 policy-I/O layout JSON, or literal 'auto' to resolve a "
            "strict layout from the instantiated backend/retargeter pairs. It is opt-in and "
            "leaves the existing real-teleop command path unchanged."
        ),
    )
    pose_frame_group = parser.add_mutually_exclusive_group()
    pose_frame_group.add_argument(
        "--pose-frame-contract",
        type=load_pose_frame_contract,
        default=None,
        metavar="JSON",
        help=(
            "Explicit v2.3 pose-frame contract JSON for formal collection. Requires "
            "--policy-io-layout auto or a layout JSON. It changes only detached dataset "
            "rows, never the live teleop command path. Dual-RM65 collection defaults to "
            "the calibrated URDF workspace-near-boundary desk contract."
        ),
    )
    pose_frame_group.add_argument(
        "--no-pose-frame-contract",
        action="store_true",
        help=(
            "Disable the default dual-RM65 v2.3 desk contract and retain the legacy "
            "v2.2/raw collection path."
        ),
    )
    parser.add_argument("--write-video", dest="write_video", action="store_true")
    parser.add_argument("--no-video", dest="write_video", action="store_false")
    parser.set_defaults(write_video=True)
    parser.add_argument(
        "--wrist-depth-mask",
        "--wrsit-depth-mask",
        dest="wrist_depth_mask",
        action="store_true",
        help=(
            "Save 15 Hz bit-packed 0/1 depth-range masks for left/right D405 wrist cameras only. "
            "The front camera depth stream is not enabled by this option."
        ),
    )
    parser.add_argument(
        "--wrist-depth-raw",
        "--wrsit-depth-raw",
        dest="wrist_depth_raw",
        action="store_true",
        help=(
            "Debug mode: save 15 Hz aligned raw z16 depth for left/right D405 wrist cameras only. "
            "This can be large; the front camera depth stream is not enabled."
        ),
    )
    parser.add_argument(
        "--wrist-depth-min-m",
        type=_positive_float,
        default=DEFAULT_WRIST_DEPTH_MIN_M,
        help="Minimum metric depth included in --wrist-depth-mask.",
    )
    parser.add_argument(
        "--wrist-depth-max-m",
        type=_positive_float,
        default=DEFAULT_WRIST_DEPTH_MAX_M,
        help="Maximum metric depth included in --wrist-depth-mask.",
    )
    parser.add_argument(
        "--wrist-depth-stream-fps",
        type=_positive_int,
        default=DEFAULT_WRIST_DEPTH_STREAM_FPS,
        help=(
            "RealSense depth stream fps for left/right D405 wrist cameras. "
            "Saved wrist-depth mask/raw frames follow the 15 Hz collection sample rate."
        ),
    )
    parser.add_argument(
        "--wrist-cam-width",
        dest="wrist_cam_width",
        type=_positive_int,
        default=None,
        help=(
            "Requested native RealSense stream width for left/right D405 wrist cameras when "
            "wrist depth is enabled. D405 RGB and depth are started at the same resolution. "
            "Defaults to the configured wrist RGB stream width."
        ),
    )
    parser.add_argument(
        "--wrist-depth-width",
        dest="wrist_cam_width",
        type=_positive_int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wrist-cam-height",
        dest="wrist_cam_height",
        type=_positive_int,
        default=None,
        help=(
            "Requested native RealSense stream height for left/right D405 wrist cameras when "
            "wrist depth is enabled. D405 RGB and depth are started at the same resolution. "
            "Defaults to the configured wrist RGB stream height."
        ),
    )
    parser.add_argument(
        "--wrist-depth-height",
        dest="wrist_cam_height",
        type=_positive_int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wrist-depth-align-to-color",
        dest="wrist_depth_align_to_color",
        action="store_true",
        default=True,
        help="Align wrist depth frames to the RGB stream before saving mask/raw depth.",
    )
    parser.add_argument(
        "--wrist-depth-no-align-to-color",
        dest="wrist_depth_align_to_color",
        action="store_false",
        help=(
            "Do not align wrist depth to RGB. Use with --wrist-cam-width/--wrist-cam-height "
            "when the saved mask/raw stream should stay at the lower wrist stream resolution."
        ),
    )
    parser.add_argument(
        "--tactile-storage",
        choices=("external", "parquet"),
        default="external",
        help="How high-dimensional tactile frames are stored in the formal dataset.",
    )
    parser.add_argument("--external-tactile", dest="tactile_storage", action="store_const", const="external")
    parser.add_argument(
        "--tactile-compression",
        choices=("zstd", "lz4", "none"),
        default="none",
        help=(
            "Compression for external tactile streams. none is the default and writes "
            "raw .bin streams; use lz4 or zstd for lossless chunk compression."
        ),
    )
    parser.add_argument(
        "--idle-tactile-compression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Losslessly recompress raw tactile streams only while RealTeleop is in "
            "startup-wait/Hold with no active, countdown, or finalizing Episode. "
            "The worker is quiesced before runtime Home. Default off; "
            "--no-idle-tactile-compression is the rollback path."
        ),
    )
    parser.add_argument(
        "--tactile-artifact-integrity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Incrementally hash exact external tactile bytes in the collection writer "
            "and publish a detachable-modality inventory after every sealed Episode. "
            "Default off; --no-tactile-artifact-integrity is the rollback path."
        ),
    )
    parser.add_argument(
        "--tactile-bundle-root",
        type=str,
        default=os.environ.get("TACTILE_BUNDLE_ROOT", ""),
        help=(
            "Pre-existing local storage root for native tactile separation. Tactile "
            "payloads and portable bundle metadata are written under "
            "datasets/<dataset_uuid>; the core dataset stores logical paths and "
            "mirrored commit receipts only. The two roots must be disjoint real "
            "directories (neither may be a symlink); they may be on the same "
            "filesystem for live collection. Defaults to TACTILE_BUNDLE_ROOT when "
            "set, otherwise empty (native separation disabled)."
        ),
    )
    parser.add_argument(
        "--tactile-bundle-min-free-gb",
        type=_non_negative_float,
        default=os.environ.get("TACTILE_BUNDLE_MIN_FREE_GB", "0"),
        help=(
            "Minimum free GiB required on --tactile-bundle-root at startup. "
            "Defaults to TACTILE_BUNDLE_MIN_FREE_GB when set. Zero performs "
            "the normal writable/device checks without an extra reserve."
        ),
    )
    parser.add_argument(
        "--writer-workers",
        type=_positive_int,
        default=4,
        help="Number of async formal dataset writer workers.",
    )
    parser.add_argument(
        "--collection-worker-mode",
        choices=("auto", "process", "thread"),
        default="auto",
        help=(
            "Formal-collection execution boundary. auto uses a spawned process for "
            "position teleop and preserves the existing thread pipeline for force/impedance. "
            "thread is the rollback path."
        ),
    )
    parser.add_argument(
        "--collection-process-queue-size",
        type=_positive_int,
        default=4,
        help=(
            "Bounded whole-sample queue for process-isolated position collection. "
            "Overload pauses collection and never blocks motion."
        ),
    )
    parser.add_argument(
        "--collection-process-shared-slots",
        type=_positive_int,
        default=4,
        help="Preallocated shared-memory camera slots for process-isolated collection.",
    )
    parser.add_argument(
        "--collection-process-extra-array-mb",
        type=_positive_int,
        default=8,
        help=(
            "Extra bytes per shared slot, in MiB, for tactile and other nested ndarray "
            "payloads. These arrays bypass multiprocessing.Queue pickle."
        ),
    )
    parser.add_argument(
        "--collection-process-startup-timeout-sec",
        type=_positive_float,
        default=DEFAULT_COLLECTION_PROCESS_STARTUP_TIMEOUT_SEC,
        help=(
            "Maximum ready-handshake time for a process-isolated formal "
            "collection writer, including validation and persisted-dataset "
            "metadata discovery. Writer preparation remains outside the "
            "realtime control loop."
        ),
    )
    parser.add_argument(
        "--collection-worker-cpus",
        type=_dmtac_worker_cpu_affinity,
        default=(),
        help=(
            "Optional comma-separated CPU affinity for the formal collection process. "
            "Leave empty for scheduler default."
        ),
    )
    parser.add_argument(
        "--collection-worker-nice",
        type=_dmtac_worker_nice,
        default=5,
        help=(
            "Nice increment for the process-isolated collection worker, in [0,19]. "
            "Defaults to 5 so control remains preferred."
        ),
    )
    parser.add_argument(
        "--writer-queue-size",
        type=_positive_int,
        default=512,
        help="Max async formal dataset writer queue size.",
    )
    parser.add_argument(
        "--storage-queue-size",
        type=_non_negative_int,
        default=64,
        help=(
            "Per-camera video storage queue size before the async writer pool. "
            "Queue-full behavior is selected separately; use 0 for direct submit."
        ),
    )
    parser.add_argument(
        "--storage-queue-backpressure-policy",
        choices=("block", "drop_oldest"),
        default="block",
        help=(
            "Video/depth storage queue behavior when full. block is the formal-"
            "collection default: sample assembly waits off the control hot path so "
            "one camera cannot independently lose a row. drop_oldest preserves the "
            "legacy best-effort behavior and may create per-camera frame gaps."
        ),
    )
    parser.add_argument(
        "--collection-warning-hz",
        type=_positive_float,
        default=15.0,
        help="GUI warning threshold for actual dataset recording rate. Defaults to 15 Hz.",
    )
    parser.add_argument(
        "--record-actual-wrist-camera-pose",
        "--record-actual-wrist-pose",
        dest="record_actual_wrist_camera_pose",
        action="store_true",
        default=True,
        help=(
            "Annotate each dataset frame with the latest background-polled RM65 feedback "
            "TCP pose as observation.wrist_camera.* without changing training state/action fields. "
            "Enabled by default; pass --no-record-actual-wrist-camera-pose to disable."
        ),
    )
    parser.add_argument(
        "--no-record-actual-wrist-camera-pose",
        "--no-record-actual-wrist-pose",
        dest="record_actual_wrist_camera_pose",
        action="store_false",
    )
    parser.add_argument(
        "--actual-wrist-camera-pose-hz",
        "--actual-wrist-pose-hz",
        type=_positive_float,
        default=30.0,
        help="Background feedback polling rate for actual wrist camera pose annotation.",
    )
    parser.add_argument(
        "--actual-wrist-camera-pose-max-age-sec",
        "--actual-wrist-pose-max-age-sec",
        type=_positive_float,
        default=0.2,
        help="Mark actual wrist camera pose annotations stale when their sample-time age exceeds this value.",
    )
    parser.add_argument(
        "--no-episode-recording",
        action="store_true",
        help="Disable lightweight debug trajectory recording support for this run.",
    )
    parser.add_argument(
        "--debug-trajectory-recording",
        action="store_true",
        help=(
            "Also save lightweight debug trajectory episode files. Formal LeRobot dataset "
            "collection remains independent; debug trajectories still default on when "
            "--no-collection is used."
        ),
    )
    parser.add_argument(
        "--episode-root",
        type=str,
        default="",
        help=(
            "Optional directory used to store episode_XXXXXX subdirectories. "
            "Providing this option also enables lightweight debug trajectory recording; "
            "it overrides --dataset-root for episode trajectory files only."
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
        help=(
            "Calibration profile to load before runtime home calibration. "
            "When omitted, the newest JSON under teleop_records/calibration_profiles is loaded."
        ),
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
        help="Optional RM65 IK tuning JSON. Defaults to robot_motion/configs/rm65_ik_tuning.json.",
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
        default=3.0,
        help=(
            "Maximum multiplier applied to arm step limits when real send dt runs slower than target. "
            "This applies to position control; impedance force control keeps its outer IK step at "
            "the nominal control dt. "
            "If omitted, real teleop uses max(default, 1.1 * arm_control_hz / effective_outer_hz) "
            "so a slower outer reference loop with a faster inner arm stream is not capped as if too few "
            "inner cycles elapsed."
        ),
    )
    parser.add_argument(
        "--arm-step-dt-stall-reset-sec",
        type=_positive_float,
        default=0.25,
        help="Reset actual-dt scaling back to the nominal control dt after longer send stalls.",
    )
    parser.add_argument(
        "--arm-singularity-protection",
        dest="arm_singularity_protection",
        action="store_true",
        help="Enable RM65 non-impedance backend singularity entry gate.",
    )
    parser.add_argument(
        "--no-arm-singularity-protection",
        dest="arm_singularity_protection",
        action="store_false",
        help=(
            "Disable the RM65 non-impedance backend singularity entry gate. "
            "Use only for diagnostics; this does not affect impedance mode."
        ),
    )
    parser.set_defaults(arm_singularity_protection=True)
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
        help=(
            "Number of consecutive fault-free control frames required before auto recovery starts. "
            "During FROZEN, RM65 real teleop stops new command dispatch and then enters RECOVERING "
            "after this many healthy frames."
        ),
    )
    parser.add_argument(
        "--recovery-ramp-sec",
        type=_positive_float,
        default=1.0,
        help="Absolute-control recovery ramp duration.",
    )
    parser.add_argument(
        "--tracker-auto-hold-sec",
        type=_positive_float,
        default=0.10,
        help="Automatically hold arm motion after a tracker target is unavailable for this long.",
    )
    parser.add_argument(
        "--tracker-manual-hold-sec",
        type=_positive_float,
        default=1.0,
        help=(
            "Escalate tracker loss to a latched manual hold after this long; "
            "runtime home must then be captured again. The default leaves a 0.1 s "
            "automatic-hold/recovery window before escalating at 1.0 s."
        ),
    )
    parser.add_argument(
        "--max-tracker-translation-jump-m",
        type=_positive_float,
        default=DEFAULT_REAL_TELEOP_MAX_TRACKER_TRANSLATION_JUMP_M,
        help=(
            "Maximum single-frame VIVE tracker translation discontinuity before FROZEN. "
            "This is intentionally larger than normal fast hand motion; lower it only when "
            "tracker teleport jumps slip through."
        ),
    )
    parser.add_argument(
        "--max-tracker-rotation-jump-deg",
        type=_positive_float,
        default=DEFAULT_REAL_TELEOP_MAX_TRACKER_ROTATION_JUMP_DEG,
        help=(
            "Maximum single-frame VIVE tracker rotation discontinuity before FROZEN. "
            "Large fast wrist rotations should be allowed; near-flip discontinuities should freeze."
        ),
    )
    parser.add_argument("--max-steps", type=_non_negative_int, default=0, help="Stop after N control steps, 0 means infinite")
    add_human_shadow_arguments(parser)
    return parser


__all__ = ["build_real_teleop_argument_parser"]
