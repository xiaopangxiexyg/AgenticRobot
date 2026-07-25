from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from ..backends.hands import InspireTacHandBackend, InspireTacHandBackendConfig
from ..arm_frames import TOOL_TCP_FRAME
from ..calibration import TeleopCalibration
from ..configs import (
    ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
    build_annotated_dual_rm65_calibration,
)
from ..diagnostics import TeleopRunState
from ..episodes import EpisodeRecorder, EpisodeRecorderConfig
from ..hand_mapping import DEFAULT_INSPIRE_HAND_JOINT_ORDER, TARGET_INSPIRE_HAND_Q6_ORDER
from ..hand_tcp_alignment import (
    DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG,
    add_vive_tracker_height_correction_args,
    tracker_hand_to_rm65_tcp_rotation,
    vive_tracker_local_translation_offset_from_args,
)
from ..hand_render import (
    LEFT_HAND_RENDER_JOINT_NAMES,
    RIGHT_HAND_RENDER_JOINT_NAMES,
    inspire_q6_to_render_qpos,
    open_hand_render_qpos,
)
from ..initial_alignment import TeleopInitialAlignment
from ..retargeters import (
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    JointsToDexHandRetargeter,
    JointsToDexHandRetargeterConfig,
    PoseToArmRetargeter,
    PoseToArmRetargeterConfig,
)
from ..safety import TeleopSafetyConfig
from ..schemas import TeleopObservation, latest_channel_timestamp, merge_teleop_commands
from ..session import TeleopSession, TeleopSessionConfig
from ..sources.hand import HandQ6Source
from ..sources.misc import KeyboardButtonSource
from ..sources.pose import (
    DEFAULT_VIVE_CLOCK_SYNC_PORT,
    VivePoseSource,
    ViveTrackerConfig,
)
from ..special_tasks.home_capture_stability import HomeCaptureStabilityState
from ..tcp import legacy_tcp_mode_requested
from ..transforms import as_pose_matrix, identity_pose, interpolate_pose
from .log_utils import ChangeGate, log_event
from robot_motion.rm65 import build_rm65_dual_arm_scene_profile
from robot_motion.rm65.ik_tuning import load_ik_tuning_config
from robot_motion.rm65.assets import require_asset_path
from robot_motion.rm65 import (
    DEFAULT_LEFT_ARM_QPOS,
    DEFAULT_RIGHT_ARM_QPOS,
    DualArmSceneProfile,
)
from sim.rm65.sapien_renderer import DebugBoxOverlay, DebugBoxTemplate, SapienArmHandRenderer


_TRACKER_BEFORE_CORRECTION_MARKER_SIZE_M = (0.018, 0.018, 0.018)
_TRACKER_AFTER_CORRECTION_MARKER_SIZE_M = (0.024, 0.024, 0.024)
_TRACKER_BEFORE_CORRECTION_COLOR_RGBA = (1.0, 0.55, 0.05, 1.0)
_TRACKER_AFTER_CORRECTION_COLOR_RGBA = (0.05, 0.75, 1.0, 1.0)
DEFAULT_RUNTIME_HOME_START_DELAY_SEC = 2.0


from ..cli_value_parsers import (
    _active_sides_from_args,
    _alpha_float,
    _non_negative_int,
    _normalize_active_sides,
    _parse_active_sides,
    _parse_arm_max_angvel,
    _parse_joint_order,
    _positive_float,
)


def _active_side_set(components) -> set[str]:
    return set(_normalize_active_sides(getattr(components, "active_sides", ("left", "right"))))


def _side_retargeter(components, side: str) -> PoseToArmRetargeter:
    return components.left_retargeter if side == "left" else components.right_retargeter


def _side_preview_backend(components, side: str) -> RM65ArmBackend:
    return components.left_preview_backend if side == "left" else components.right_preview_backend


def _side_state_backend(components, side: str) -> RM65ArmBackend | None:
    return components.left_state_backend if side == "left" else components.right_state_backend


def _assert_inactive_side_config_empty(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
    *,
    suffixes: tuple[str, ...],
) -> None:
    active = set(active_sides)
    for side in ("left", "right"):
        if side in active:
            continue
        present = [
            f"--{side.replace('_', '-')}-{suffix.replace('_', '-')}"
            for suffix in suffixes
            if str(getattr(args, f"{side}_{suffix}", "") or "")
        ]
        if present:
            raise ValueError(
                f"Inactive side '{side}' received configuration {present}; "
                "remove those flags or include the side in --active-sides."
            )


def _log(tag: str, message: str, *, now: float | None = None) -> None:
    log_event(tag, message, now=now)


def _qpos_summary(qpos: np.ndarray | None) -> str:
    if qpos is None:
        return "None"
    return str(np.round(np.asarray(qpos, dtype=float), 3).tolist())


class TerminalClutchInput:
    _MAX_CHARS_PER_POLL = 64

    def __init__(self, keyboard_source: KeyboardButtonSource):
        self.keyboard_source = keyboard_source
        self.enabled = sys.stdin.isatty()
        self._old_attrs: list[int] | None = None
        self.should_quit = False
        self.left_pressed = False
        self.right_pressed = False
        self._calibration_requested = False
        self._episode_start_requested = False
        self._episode_stop_requested = False

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            self._old_attrs = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        except (OSError, termios.error):
            self.enabled = False
            self._old_attrs = None

    def close(self) -> None:
        if self.enabled and self._old_attrs is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_attrs)
            except (OSError, termios.error):
                pass
            self._old_attrs = None

    def poll(self) -> None:
        if not self.enabled:
            return
        # Bound work per control frame. A held key, key-repeat stream, or pasted
        # terminal input must not keep this method draining stdin indefinitely
        # while the robot control loop is waiting to run.
        for _ in range(self._MAX_CHARS_PER_POLL):
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            except OSError:
                self.enabled = False
                self._old_attrs = None
                _log("Keys", "terminal input became unavailable; keyboard controls disabled for this run.")
                return
            if not ready:
                return
            try:
                char = sys.stdin.read(1)
            except OSError:
                self.enabled = False
                self._old_attrs = None
                _log("Keys", "terminal input became unavailable; keyboard controls disabled for this run.")
                return
            if not char:
                return
            if char == "q":
                self.should_quit = True
                return
            if char in {"p", "\n", "\r"}:
                self._calibration_requested = True
                continue
            if char == "t":
                self._episode_start_requested = True
                continue
            if char == "s":
                self._episode_stop_requested = True
                continue
            if char == "z":
                self.left_pressed = not self.left_pressed
            elif char == "x":
                self.right_pressed = not self.right_pressed
            elif char == "b":
                new_state = not (self.left_pressed and self.right_pressed)
                self.left_pressed = new_state
                self.right_pressed = new_state
            elif char == "r":
                self.left_pressed = False
                self.right_pressed = False
            else:
                continue

            self.keyboard_source.set_button(
                "left_clutch_button",
                self.left_pressed,
                side="left",
                critical=False,
            )
            self.keyboard_source.set_button(
                "right_clutch_button",
                self.right_pressed,
                side="right",
                critical=False,
            )

    def consume_calibration_request(self) -> bool:
        requested = self._calibration_requested
        self._calibration_requested = False
        return requested

    def consume_episode_start_request(self) -> bool:
        requested = self._episode_start_requested
        self._episode_start_requested = False
        return requested

    def consume_episode_stop_request(self) -> bool:
        requested = self._episode_stop_requested
        self._episode_stop_requested = False
        return requested


@dataclass(frozen=True)
class VirtualRunComponents:
    session: TeleopSession
    active_sides: tuple[str, ...]
    left_preview_backend: RM65ArmBackend
    right_preview_backend: RM65ArmBackend
    left_state_backend: RM65ArmBackend | None
    right_state_backend: RM65ArmBackend | None
    left_hand_backend: InspireTacHandBackend | None
    right_hand_backend: InspireTacHandBackend | None
    initial_alignment: TeleopInitialAlignment | None
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


@dataclass
class RecoveryPlan:
    mode: str
    started_at: float
    duration_sec: float = 0.0
    payload: dict[str, object] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use real VIVE wrist trackers as input and preview dual RM65 arm motion "
            "in SAPIEN. In arm-mode=real, the script can read the current real-arm start pose "
            "or initialize from a saved initial-alignment file, but never sends any motion command to hardware."
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
        "--active-sides",
        type=_parse_active_sides,
        default=("left", "right"),
        help="One of: left, right, both.",
    )
    parser.add_argument(
        "--control-mode",
        choices=["absolute"],
        default="absolute",
        help="Virtual retargeting mode. The active runtime uses absolute pose mapping.",
    )
    parser.add_argument(
        "--tcp-mode",
        type=str,
        default=TOOL_TCP_FRAME,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--arm-mode",
        choices=["real", "virtual"],
        default="real",
        help="Use real RM65 state as the rendered start pose, or pure virtual preview only.",
    )
    parser.add_argument("--left-arm-ip", type=str, default="", help="Optional left RM65 IP for read-only start-pose sync")
    parser.add_argument("--right-arm-ip", type=str, default="", help="Optional right RM65 IP for read-only start-pose sync")
    parser.add_argument("--left-hand-port", type=str, default="", help="Optional left hand teleop serial port")
    parser.add_argument("--right-hand-port", type=str, default="", help="Optional right hand teleop serial port")
    parser.add_argument("--hand-reader-baudrate", type=int, default=115200, help="Hand teleop serial baudrate")
    parser.add_argument("--hand-reader-timeout", type=float, default=0.05, help="Hand teleop serial timeout in seconds")
    parser.add_argument(
        "--exo-hand-calibration",
        action="store_true",
        help="Re-run exoskeleton hand zero-pose calibration and update the fixed calibration file.",
    )
    parser.add_argument(
        "--exo-hand-calibration-path",
        type=str,
        default="",
        help="Optional fixed exoskeleton hand zero-pose calibration JSON path.",
    )
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
        help="Invert normalized q6 before sending/rendering. Default is enabled.",
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
        help="Only update preview hand state when max q6 delta exceeds this threshold.",
    )
    parser.add_argument("--translation-scale", type=float, default=1.0, help="Scale applied to desk-frame translation")
    add_vive_tracker_height_correction_args(parser)
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
    parser.add_argument("--hz", type=_positive_float, default=30.0, help="Target control/update rate")
    parser.add_argument(
        "--arm-max-angvel",
        type=_parse_arm_max_angvel,
        default=(0.35, 0.35, 0.35, 0.5, 0.5, 0.5),
        help="Scalar or 6 comma-separated arm joint velocity limits in rad/s.",
    )
    parser.add_argument(
        "--print-hz",
        type=_positive_float,
        default=2.0,
        help="Legacy console progress throttle. Most logs are now emitted only on change.",
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
        default="",
        help="Optional auxiliary tracker name. Empty by default; pass tracker_3 to enable a third tracker.",
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
    parser.add_argument(
        "--keyboard-clutch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
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
            "Optional initial-alignment JSON saved by rm65_real_debug. "
            "When provided, preview startup uses the saved arm qpos and skips real-arm start-state reads. "
            "This does not skip tracker home capture."
        ),
    )
    parser.add_argument(
        "--ik-tuning-json",
        type=str,
        default="",
        help="Optional RM65 IK tuning JSON. Defaults to robot_motion/configs/rm65_ik_tuning.json.",
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
        help="Absolute-control recovery ramp duration used in arm-mode=real style recovery.",
    )
    parser.add_argument("--max-steps", type=_non_negative_int, default=0, help="Stop after N control steps, 0 means infinite")
    args = parser.parse_args()
    args.requested_tcp_mode = str(getattr(args, "tcp_mode", TOOL_TCP_FRAME))
    args.tcp_mode = TOOL_TCP_FRAME
    return args


def _vive_tracker_local_translation_offset(args: argparse.Namespace) -> np.ndarray | None:
    return vive_tracker_local_translation_offset_from_args(args)


def build_components(args: argparse.Namespace) -> VirtualRunComponents:
    active_sides = _active_sides_from_args(args)
    _assert_inactive_side_config_empty(
        args,
        active_sides,
        suffixes=("arm_ip", "hand_port"),
    )
    preview_control_dt = 1.0 / float(args.hz)
    preview_dry_run_fast = args.arm_mode != "real"
    collision_guard_requested = (
        set(active_sides) == {"left", "right"}
        and not args.disable_dual_arm_collision_avoidance
    )
    arm_max_angvel = np.asarray(
        getattr(args, "arm_max_angvel", (0.35, 0.35, 0.35, 0.5, 0.5, 0.5)),
        dtype=float,
    ).reshape(6)
    realistic_joint_step = tuple((arm_max_angvel * preview_control_dt).tolist())
    collision_scene_profile = build_rm65_dual_arm_scene_profile(args.collision_scene_profile)
    ik_tuning = load_ik_tuning_config(args.ik_tuning_json or None)
    initial_alignment = (
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
    tracker_translation_offset = _vive_tracker_local_translation_offset(args)
    calibration.metadata.update(
        {
            "left_tracker_name": args.left_tracker_name,
            "right_tracker_name": args.right_tracker_name,
            "aux_tracker_name": args.aux_tracker_name,
            "active_sides": list(active_sides),
            "arm_mode": args.arm_mode,
            "left_arm_ip": args.left_arm_ip,
            "right_arm_ip": args.right_arm_ip,
            "left_hand_port": getattr(args, "left_hand_port", ""),
            "right_hand_port": getattr(args, "right_hand_port", ""),
            "hand_reader_baudrate": int(getattr(args, "hand_reader_baudrate", 115200)),
            "hand_joint_order": list(getattr(args, "hand_joint_order", DEFAULT_INSPIRE_HAND_JOINT_ORDER)),
            "hand_invert": bool(getattr(args, "hand_invert", True)),
            "hand_command_alpha": float(getattr(args, "hand_command_alpha", 1.0)),
            "hand_send_threshold": float(getattr(args, "hand_send_threshold", 0.002)),
            "preview_control_dt": preview_control_dt,
            "preview_dry_run_fast": preview_dry_run_fast,
            "dual_arm_collision_avoidance_requested": collision_guard_requested,
            "collision_avoidance_mode": args.collision_avoidance_mode,
            "collision_scene_profile": args.collision_scene_profile,
            "ik_tuning_source": ik_tuning.source_path,
            "ik_objective_mode": ik_tuning.ik_objective_mode,
            "initial_alignment_source": args.load_initial_alignment,
            "vive_tracker_height_correction_enabled": tracker_translation_offset is not None,
            "vive_tracker_height_correction_m": float(
                getattr(
                    args,
                    "vive_tracker_height_correction_m",
                    DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG.height_correction_m,
                )
            ),
            "vive_tracker_local_translation_offset_m": None
            if tracker_translation_offset is None
            else tracker_translation_offset.tolist(),
        }
    )
    keyboard_source = KeyboardButtonSource(name="terminal_clutch")
    default_clutch_pressed = False
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
            input_local_translation_offset=tracker_translation_offset,
            input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
            tracker_motion_intent_enabled=True,
        )
    )
    right_retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="right",
            input_channel_name="right_wrist_pose",
            effector_name="right_arm",
            control_mode=args.control_mode,
            translation_scale=args.translation_scale,
            input_local_translation_offset=tracker_translation_offset,
            input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
            tracker_motion_intent_enabled=True,
        )
    )
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
                        enforce_realistic_joint_step=bool(args.arm_mode == "real"),
                        left_execution_max_joint_step_rad=realistic_joint_step,
                        right_execution_max_joint_step_rad=realistic_joint_step,
                        left_initial_qpos=DEFAULT_LEFT_ARM_QPOS,
                        right_initial_qpos=DEFAULT_RIGHT_ARM_QPOS,
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
            exo_hand_calibration=bool(getattr(args, "exo_hand_calibration", False)),
            calibration_path=str(getattr(args, "exo_hand_calibration_path", "") or "") or None,
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

    left_hand_backend = None
    if "left" in active_sides and left_hand_port:
        left_hand_backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                send_threshold=float(getattr(args, "hand_send_threshold", 0.002)),
            )
        )
    right_hand_backend = None
    if "right" in active_sides and right_hand_port:
        right_hand_backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip="0.0.0.0",
                side="right",
                connect_hardware=False,
                send_threshold=float(getattr(args, "hand_send_threshold", 0.002)),
            )
        )

    session_sources = [
        VivePoseSource(
            tracker_configs=tracker_configs,
            host=args.host,
            port=args.port,
            clock_sync_port=int(
                getattr(args, "vive_clock_sync_port", DEFAULT_VIVE_CLOCK_SYNC_PORT)
            ),
        ),
        keyboard_source,
    ]
    if hand_source is not None:
        session_sources.append(hand_source)

    session = TeleopSession(
        config=TeleopSessionConfig(
            mode="vive_sapien_virtual",
            tcp_mode=args.tcp_mode,
            record_enabled=not args.no_record,
            allow_hardware_after_debug_pass=False,
            recovery_healthy_frame_count=max(1, int(args.recovery_healthy_frames)),
            non_freezing_fault_codes=("low_control_rate",),
        ),
        sources=session_sources,
        retargeters=retargeters,
        rig=None,
        calibration=calibration,
        safety_config=TeleopSafetyConfig(
            freeze_on_fault=True,
            required_channels=tuple(f"{side}_wrist_pose" for side in active_sides),
            invalid_frame_limit=3,
        ),
    )

    arm_urdf_path = require_asset_path("rm65_arm.urdf")
    left_initial_qpos = tuple(DEFAULT_LEFT_ARM_QPOS.tolist())
    right_initial_qpos = tuple(DEFAULT_RIGHT_ARM_QPOS.tolist())
    if initial_alignment is not None:
        left_alignment = initial_alignment.get_side("left")
        right_alignment = initial_alignment.get_side("right")
        if left_alignment is not None and left_alignment.arm_qpos is not None:
            left_initial_qpos = tuple(
                np.asarray(left_alignment.arm_qpos, dtype=float).reshape(6).tolist()
            )
        if right_alignment is not None and right_alignment.arm_qpos is not None:
            right_initial_qpos = tuple(
                np.asarray(right_alignment.arm_qpos, dtype=float).reshape(6).tolist()
            )
    left_preview_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            default_tcp_frame=args.tcp_mode,
            arm_urdf_path=arm_urdf_path,
            connect_hardware=False,
            control_dt=preview_control_dt,
            max_angvel=tuple(arm_max_angvel.tolist()),
            dry_run_fast=preview_dry_run_fast,
            initial_qpos=left_initial_qpos,
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
            control_dt=preview_control_dt,
            max_angvel=tuple(arm_max_angvel.tolist()),
            dry_run_fast=preview_dry_run_fast,
            initial_qpos=right_initial_qpos,
            ik_tuning=ik_tuning,
        )
    )
    left_state_backend = None
    right_state_backend = None
    if initial_alignment is None:
        if "left" in active_sides and args.left_arm_ip:
            left_state_backend = RM65ArmBackend(
                RM65ArmBackendConfig(
                    arm_ip=args.left_arm_ip,
                    side="left",
                    default_tcp_frame=args.tcp_mode,
                    arm_urdf_path=arm_urdf_path,
                    connect_hardware=True,
                    max_angvel=tuple(arm_max_angvel.tolist()),
                    initial_qpos=None,
                    ik_tuning=ik_tuning,
                )
            )
        if "right" in active_sides and args.right_arm_ip:
            right_state_backend = RM65ArmBackend(
                RM65ArmBackendConfig(
                    arm_ip=args.right_arm_ip,
                    side="right",
                    default_tcp_frame=args.tcp_mode,
                    arm_urdf_path=arm_urdf_path,
                    connect_hardware=True,
                    max_angvel=tuple(arm_max_angvel.tolist()),
                    initial_qpos=None,
                    ik_tuning=ik_tuning,
                )
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
                "to validate the teleop loop without visualization."
            ) from exc
        renderer.reserve_debug_frames(
            (
                "desk",
                "left_arm_base",
                "right_arm_base",
                "left_tracker",
                "right_tracker",
                "left_tracker_before_correction",
                "right_tracker_before_correction",
                "left_tracker_after_correction",
                "right_tracker_after_correction",
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
        debug_box_templates = _collision_debug_box_templates(collision_scene_profile)
        debug_box_templates.update(_tracker_correction_debug_box_templates(active_sides))
        renderer.reserve_debug_boxes(debug_box_templates)
        correction_label = (
            "disabled; before/after markers overlap"
            if tracker_translation_offset is None
            else f"offset_m={np.round(tracker_translation_offset, 4).tolist()}"
        )
        _log(
            "TrackerCorrectionViz",
            "orange marker/frame=before VIVE height correction; "
            f"cyan marker/frame=after local -Z height correction ({correction_label}).",
        )
    if dual_arm_retargeter is not None:
        dual_arm_retargeter.set_current_qpos_pair(
            left_qpos=_state_qpos(left_preview_backend, DEFAULT_LEFT_ARM_QPOS),
            right_qpos=_state_qpos(right_preview_backend, DEFAULT_RIGHT_ARM_QPOS),
        )
    return VirtualRunComponents(
        session=session,
        active_sides=active_sides,
        left_preview_backend=left_preview_backend,
        right_preview_backend=right_preview_backend,
        left_state_backend=left_state_backend,
        right_state_backend=right_state_backend,
        left_hand_backend=left_hand_backend,
        right_hand_backend=right_hand_backend,
        initial_alignment=initial_alignment,
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
    )


def _state_qpos(backend: RM65ArmBackend, fallback: np.ndarray) -> np.ndarray:
    state = backend.get_state()
    qpos = state["current_qpos"]
    if qpos is None:
        return fallback.copy()
    return np.asarray(qpos, dtype=float).copy()


def _has_live_real_state(components: VirtualRunComponents) -> bool:
    return any(_side_state_backend(components, side) is not None for side in components.active_sides)


def _preview_should_follow_live_real_state(
    *,
    arm_mode: str,
    components: VirtualRunComponents,
) -> bool:
    return arm_mode == "real" and _has_live_real_state(components)


def _manual_calibration_confirmation_required(
    args: argparse.Namespace,
    components: VirtualRunComponents,
) -> bool:
    _ = components
    return not bool(args.auto_calibrate)


def _pose_translation_summary(pose: np.ndarray | None) -> str:
    if pose is None:
        return "None"
    return str(np.round(np.asarray(pose[:3, 3], dtype=float), 4).tolist())


def _pose_rotation_summary(pose: np.ndarray | None) -> str:
    if pose is None:
        return "None"
    pose_matrix = as_pose_matrix(pose)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        euler_deg = R.from_matrix(pose_matrix[:3, :3]).as_euler("xyz", degrees=True)
    return str(np.round(np.asarray(euler_deg, dtype=float), 1).tolist())


def _relative_rotation_summary(reference_pose: np.ndarray | None, pose: np.ndarray | None) -> str:
    if reference_pose is None or pose is None:
        return "None"
    relative_pose = np.linalg.inv(as_pose_matrix(reference_pose)) @ as_pose_matrix(pose)
    return _pose_rotation_summary(relative_pose)


def _dominant_axis_label(axis: np.ndarray) -> str:
    vector = np.asarray(axis, dtype=float).reshape(3)
    dominant_index = int(np.argmax(np.abs(vector)))
    dominant_value = float(vector[dominant_index])
    sign = "+" if dominant_value >= 0.0 else "-"
    return f"{sign}{'xyz'[dominant_index]}({abs(dominant_value):.3f})"


def _tracker_axis_mapping_summary(offset: np.ndarray | None) -> str:
    if offset is None:
        return "None"
    rotation = as_pose_matrix(offset)[:3, :3]
    tracker_axes = np.eye(3, dtype=float)
    return ", ".join(
        f"tracker_{axis_name}->hand_{_dominant_axis_label(rotation.T @ tracker_axes[index])}"
        for index, axis_name in enumerate(("x", "y", "z"))
    )


def _tracker_rpy_mapping_summary(offset: np.ndarray | None) -> str:
    if offset is None:
        return "None"
    rotation = as_pose_matrix(offset)[:3, :3]
    tracker_axes = np.eye(3, dtype=float)
    labels = ("roll(x)", "pitch(y)", "yaw(z)")
    return ", ".join(
        f"{labels[index]}->{_dominant_axis_label(rotation.T @ tracker_axes[index])}"
        for index in range(3)
    )


def _channel_pose_raw_in_desk(channel, calibration: TeleopCalibration) -> np.ndarray:
    pose = as_pose_matrix(channel.value)
    frame_name = channel.frame_name or ""
    if frame_name in {"vive_world", "world"}:
        pose_operator = calibration.world_to_operator(pose, tracker_role=None)
        return calibration.operator_to_desk(pose_operator)
    if frame_name.endswith("_operator") or frame_name == "operator":
        return calibration.operator_to_desk(pose)
    if frame_name in {calibration.desk_frame.name, "desk"}:
        return pose
    raise ValueError(f"Unsupported calibration input frame: {frame_name or '<none>'}")


def _robot_home_mode_label(components: VirtualRunComponents) -> str:
    if components.initial_alignment is not None:
        return "tracker_home + alignment_robot_home"
    if _has_live_real_state(components):
        return "tracker_home + real_robot_home"
    return "tracker_home + preview_robot_home"


def _resolve_robot_home_for_side(
    components: VirtualRunComponents,
    *,
    side: str,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    calibration = components.session.calibration
    assert calibration is not None
    preview_backend = (
        components.left_preview_backend
        if side == "left"
        else components.right_preview_backend
    )
    state_backend = (
        components.left_state_backend
        if side == "left"
        else components.right_state_backend
    )

    if components.initial_alignment is not None:
        alignment_state = components.initial_alignment.get_side(side)
        if alignment_state is not None:
            qpos = None
            if alignment_state.arm_qpos is not None:
                qpos = np.asarray(alignment_state.arm_qpos, dtype=float).copy()
                preview_backend.set_local_qpos(qpos)
                _sync_dual_arm_retargeter_from_preview(components)
            if alignment_state.tcp_pose_desk is not None:
                return alignment_state.tcp_pose_desk.copy(), qpos, "initial_alignment_pose_desk"
            if alignment_state.tcp_pose_base is not None:
                return (
                    calibration.get_arm_for_side(side).T_D_B @ alignment_state.tcp_pose_base,
                    qpos,
                    "initial_alignment_pose_base",
                )
            if qpos is not None:
                return _preview_tcp_pose_desk(components, side=side), qpos, "initial_alignment_qpos"

    if state_backend is not None:
        state = state_backend.refresh_state_from_hardware()
        qpos = state.get("current_qpos")
        if qpos is not None:
            qpos_array = np.asarray(qpos, dtype=float).copy()
            preview_backend.set_local_qpos(qpos_array)
            # A preview backend may normalize a tiny SDK feedback quantization
            # overshoot at a hard endpoint. Persist that bounded anchor for
            # this session's home/IK seed rather than the raw observation.
            preview_state = preview_backend.get_state()
            anchor_qpos = preview_state.get("current_qpos")
            if anchor_qpos is None:
                raise RuntimeError(f"{side}_arm preview did not retain a local qpos anchor.")
            anchor_qpos = np.asarray(anchor_qpos, dtype=float).reshape(6).copy()
            normalization = preview_state.get("local_qpos_normalization")
            if isinstance(normalization, dict) and normalization.get("status") == "clipped":
                clipped_joints = normalization.get("clipped_joint_indices", [])
                _log(
                    "Calibration",
                    f"{side}_arm hardware qpos had a tiny endpoint quantization overshoot; "
                    f"preview/home anchor clipped at joint(s) {clipped_joints}. "
                    "The arm remains at the hard limit; move it away before normal tracking.",
                )
            _sync_dual_arm_retargeter_from_preview(components)
            return _preview_tcp_pose_desk(components, side=side), anchor_qpos, "real_current_state"

    qpos = preview_backend.get_state()["current_qpos"]
    return _preview_tcp_pose_desk(components, side=side), qpos, "preview_current_state"


def _tracker_after_height_correction_visual_pose(
    retargeter: PoseToArmRetargeter,
    tracker_pose_desk: np.ndarray,
) -> np.ndarray:
    corrected = retargeter.prepare_input_pose_desk(tracker_pose_desk)
    visual_pose = corrected.copy()
    visual_pose[:3, :3] = tracker_pose_desk[:3, :3]
    return visual_pose


def _debug_marker_pose_at_origin(pose: np.ndarray) -> np.ndarray:
    marker_pose = identity_pose()
    marker_pose[:3, 3] = as_pose_matrix(pose)[:3, 3]
    return marker_pose


def _tracker_correction_visual_poses(
    components: VirtualRunComponents,
    *,
    side: str,
    channels: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not channels:
        return None
    calibration = components.session.calibration
    assert calibration is not None
    channel = channels.get(f"{side}_wrist_pose")
    if channel is None or not channel.valid:
        return None
    before_pose = _channel_pose_in_desk(channel, calibration)
    after_pose = _tracker_after_height_correction_visual_pose(
        _side_retargeter(components, side),
        before_pose,
    )
    return before_pose, after_pose


def _build_renderer_debug_frames(
    components: VirtualRunComponents,
    *,
    channels: dict[str, object] | None,
) -> dict[str, np.ndarray]:
    calibration = components.session.calibration
    assert calibration is not None
    frames: dict[str, np.ndarray] = {"desk": identity_pose()}
    for side in components.active_sides:
        retargeter = _side_retargeter(components, side)
        arm = calibration.get_arm_for_side(side)
        if arm.T_D_B is not None:
            frames[f"{side}_arm_base"] = arm.T_D_B.copy()
        frames[f"{side}_tcp_actual"] = _preview_tcp_pose_desk(components, side=side)
        if retargeter.last_target_pose_desk is not None:
            frames[f"{side}_tcp_target"] = retargeter.last_target_pose_desk
        if retargeter.home_human_pose_desk is not None:
            frames[f"{side}_hand_home"] = retargeter.home_human_pose_desk
        if retargeter.home_robot_pose_desk is not None:
            frames[f"{side}_robot_home"] = retargeter.home_robot_pose_desk
        try:
            tracker_poses = _tracker_correction_visual_poses(
                components,
                side=side,
                channels=channels,
            )
        except Exception:
            continue
        if tracker_poses is None:
            continue
        before_pose, after_pose = tracker_poses
        frames[f"{side}_tracker_before_correction"] = before_pose
        frames[f"{side}_tracker_after_correction"] = after_pose
        frames[f"{side}_tracker"] = after_pose
    return frames


def _tracker_correction_debug_box_templates(
    active_sides: tuple[str, ...],
) -> dict[str, DebugBoxTemplate]:
    boxes: dict[str, DebugBoxTemplate] = {}
    for side in active_sides:
        boxes[f"{side}_tracker_before_correction_marker"] = DebugBoxTemplate(
            size_xyz=_TRACKER_BEFORE_CORRECTION_MARKER_SIZE_M,
            color_rgba=_TRACKER_BEFORE_CORRECTION_COLOR_RGBA,
        )
        boxes[f"{side}_tracker_after_correction_marker"] = DebugBoxTemplate(
            size_xyz=_TRACKER_AFTER_CORRECTION_MARKER_SIZE_M,
            color_rgba=_TRACKER_AFTER_CORRECTION_COLOR_RGBA,
        )
    return boxes


def _collision_debug_box_templates(
    profile: DualArmSceneProfile,
) -> dict[str, DebugBoxTemplate]:
    boxes = {
        spec.name: DebugBoxTemplate(
            size_xyz=spec.box_size_xyz,
            color_rgba=spec.color_rgba,
        )
        for spec in profile.hand_proxy_boxes
    }
    boxes.update(
        {
            spec.name: DebugBoxTemplate(
                size_xyz=spec.box_size_xyz,
                color_rgba=spec.color_rgba,
            )
            for spec in profile.fixed_scene_boxes
        }
    )
    return boxes


def _build_renderer_debug_boxes(
    components: VirtualRunComponents,
    *,
    channels: dict[str, object] | None = None,
) -> dict[str, DebugBoxOverlay]:
    profile = components.collision_scene_profile
    boxes = {
        spec.name: DebugBoxOverlay(
            pose_matrix=_preview_tcp_pose_desk(components, side=spec.side)
            @ spec.parent_to_box_pose_matrix(),
            size_xyz=spec.box_size_xyz,
            color_rgba=spec.color_rgba,
        )
        for spec in profile.hand_proxy_boxes
    }
    boxes.update(
        {
            spec.name: DebugBoxOverlay(
                pose_matrix=spec.world_to_box_pose_matrix(),
                size_xyz=spec.box_size_xyz,
                color_rgba=spec.color_rgba,
            )
            for spec in profile.fixed_scene_boxes
        }
    )
    for side in components.active_sides:
        try:
            tracker_poses = _tracker_correction_visual_poses(
                components,
                side=side,
                channels=channels,
            )
        except Exception:
            continue
        if tracker_poses is None:
            continue
        before_pose, after_pose = tracker_poses
        boxes[f"{side}_tracker_before_correction_marker"] = DebugBoxOverlay(
            pose_matrix=_debug_marker_pose_at_origin(before_pose),
            size_xyz=_TRACKER_BEFORE_CORRECTION_MARKER_SIZE_M,
            color_rgba=_TRACKER_BEFORE_CORRECTION_COLOR_RGBA,
        )
        boxes[f"{side}_tracker_after_correction_marker"] = DebugBoxOverlay(
            pose_matrix=_debug_marker_pose_at_origin(after_pose),
            size_xyz=_TRACKER_AFTER_CORRECTION_MARKER_SIZE_M,
            color_rgba=_TRACKER_AFTER_CORRECTION_COLOR_RGBA,
        )
    return boxes


def _emit_tracker_axis_mapping_logs(components: VirtualRunComponents) -> None:
    calibration = components.session.calibration
    assert calibration is not None
    for tracker_role in tuple(f"{side}_wrist" for side in components.active_sides):
        offset = calibration.tracker_offsets.get(tracker_role)
        if offset is None:
            continue
        _log(
            "TrackerAxes",
            f"{tracker_role} tracker_mount_compensation_rpy_xyz_deg={_pose_rotation_summary(offset)} "
            f"tracker_axis_to_hand_axis={_tracker_axis_mapping_summary(offset)} "
            f"tracker_rpy_to_hand_rpy={_tracker_rpy_mapping_summary(offset)}",
        )


def _emit_home_alignment_logs(components: VirtualRunComponents) -> None:
    for side in components.active_sides:
        retargeter = _side_retargeter(components, side)
        if retargeter.home_human_pose_desk is None or retargeter.home_robot_pose_desk is None:
            continue
        relative_pose = (
            np.linalg.inv(retargeter.home_human_pose_desk) @ retargeter.home_robot_pose_desk
        )
        _log(
            "HomeAxes",
            f"{side} home_human_rpy_xyz_deg={_pose_rotation_summary(retargeter.home_human_pose_desk)} "
            f"home_robot_rpy_xyz_deg={_pose_rotation_summary(retargeter.home_robot_pose_desk)} "
            f"robot_from_human_rpy_xyz_deg={_pose_rotation_summary(relative_pose)}",
        )


def _emit_collision_proxy_logs(components: VirtualRunComponents) -> None:
    profile = components.collision_scene_profile
    if not profile.hand_proxy_boxes and not profile.fixed_scene_boxes:
        return
    expected_self_pairs = {
        (
            spec.geometry_name,
            "arm_link2_0" if spec.side == "left" else "right_arm_link2_0",
        )
        for spec in profile.hand_proxy_boxes
    }
    self_pairs = tuple(pair for pair in profile.collision_pairs if pair in expected_self_pairs)
    _log(
        "CollisionProxy",
        f"profile={profile.name} urdf={profile.urdf_path} self_pairs={self_pairs}",
    )
    for spec in profile.hand_proxy_boxes:
        _log(
            "CollisionProxy",
            f"{spec.side} geometry={spec.geometry_name} parent={spec.parent_link_name} "
            f"size_m={np.round(np.asarray(spec.box_size_xyz, dtype=float), 4).tolist()} "
            f"joint_xyz_m={np.round(np.asarray(spec.joint_origin_xyz, dtype=float), 4).tolist()} "
            f"joint_rpy_deg={np.round(np.rad2deg(np.asarray(spec.joint_origin_rpy, dtype=float)), 1).tolist()} "
            f"box_xyz_m={np.round(np.asarray(spec.box_origin_xyz, dtype=float), 4).tolist()} "
            f"box_rpy_deg={np.round(np.rad2deg(np.asarray(spec.box_origin_rpy, dtype=float)), 1).tolist()}",
        )
    for spec in profile.fixed_scene_boxes:
        _log(
            "CollisionProxy",
            f"fixed_box geometry={spec.geometry_name} parent={spec.parent_link_name} "
            f"size_m={np.round(np.asarray(spec.box_size_xyz, dtype=float), 4).tolist()} "
            f"joint_xyz_m={np.round(np.asarray(spec.joint_origin_xyz, dtype=float), 4).tolist()} "
            f"joint_rpy_deg={np.round(np.rad2deg(np.asarray(spec.joint_origin_rpy, dtype=float)), 1).tolist()} "
            f"box_xyz_m={np.round(np.asarray(spec.box_origin_xyz, dtype=float), 4).tolist()} "
            f"box_rpy_deg={np.round(np.rad2deg(np.asarray(spec.box_origin_rpy, dtype=float)), 1).tolist()}",
        )


def _emit_runtime_orientation_logs(
    components: VirtualRunComponents,
    *,
    channels: dict[str, object],
    now: float,
    gate: ChangeGate,
) -> None:
    calibration = components.session.calibration
    assert calibration is not None
    for side in components.active_sides:
        retargeter = _side_retargeter(components, side)
        channel = channels.get(f"{side}_wrist_pose")
        if channel is None or not channel.valid:
            continue
        try:
            raw_pose = _channel_pose_raw_in_desk(channel, calibration)
            human_pose = _channel_pose_in_desk(channel, calibration)
        except Exception:
            continue
        target_pose = retargeter.last_target_pose_desk
        actual_pose = _preview_tcp_pose_desk(components, side=side)
        snapshot = (
            tuple(np.round(raw_pose[:3, :3].reshape(-1), 2).tolist()),
            tuple(np.round(human_pose[:3, :3].reshape(-1), 2).tolist()),
            tuple(np.round(actual_pose[:3, :3].reshape(-1), 2).tolist()),
            None
            if target_pose is None
            else tuple(np.round(target_pose[:3, :3].reshape(-1), 2).tolist()),
        )
        if not gate.should_emit(
            f"orientation_debug_{side}",
            snapshot,
            now=now,
            heartbeat_sec=1.5,
        ):
            continue
        _log(
            "Orientation",
            f"{side} raw_rpy_xyz_deg={_pose_rotation_summary(raw_pose)} "
            f"hand_rpy_xyz_deg={_pose_rotation_summary(human_pose)} "
            f"target_rpy_xyz_deg={_pose_rotation_summary(target_pose)} "
            f"actual_rpy_xyz_deg={_pose_rotation_summary(actual_pose)} "
            f"human_rel_rpy_xyz_deg={_relative_rotation_summary(retargeter.home_human_pose_desk, human_pose)} "
            f"target_rel_rpy_xyz_deg={_relative_rotation_summary(retargeter.home_robot_pose_desk, target_pose)} "
            f"actual_rel_rpy_xyz_deg={_relative_rotation_summary(retargeter.home_robot_pose_desk, actual_pose)}",
            now=now,
        )


def _sync_preview_from_real_state(
    components: VirtualRunComponents,
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
        if side in _active_side_set(components) and state_backend is not None
    ]
    state_read_executor = getattr(components, "arm_state_read_executor", None)
    parallel_submit_failed = False
    if state_read_executor is not None and len(active) > 1:
        try:
            future_map = {
                state_read_executor.submit(state_backend.refresh_state_from_hardware): (
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
    if state_read_executor is None or len(active) <= 1 or parallel_submit_failed:
        for side, state_backend, preview_backend in active:
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
    if components.dual_arm_retargeter is not None:
        left_qpos = components.left_preview_backend.get_state()["current_qpos"]
        right_qpos = components.right_preview_backend.get_state()["current_qpos"]
        if left_qpos is not None and right_qpos is not None:
            components.dual_arm_retargeter.set_current_qpos_pair(
                left_qpos=left_qpos,
                right_qpos=right_qpos,
            )
    return synced


def _mean_pose(poses: list[np.ndarray]) -> np.ndarray:
    if not poses:
        raise ValueError("Cannot average an empty pose list.")
    mean_pose = np.eye(4, dtype=float)
    mean_pose[:3, 3] = np.mean([pose[:3, 3] for pose in poses], axis=0)
    rot_sum = np.sum([pose[:3, :3] for pose in poses], axis=0)
    u, _, vt = np.linalg.svd(rot_sum)
    mean_pose[:3, :3] = u @ vt
    if np.linalg.det(mean_pose[:3, :3]) < 0.0:
        u[:, -1] *= -1.0
        mean_pose[:3, :3] = u @ vt
    return mean_pose


def _channel_pose_in_desk(channel, calibration) -> np.ndarray:
    pose = as_pose_matrix(channel.value)
    frame_name = channel.frame_name or ""
    metadata = channel.metadata or {}
    tracker_role = metadata.get("tracker_role")

    if frame_name in {"vive_world", "world"}:
        return calibration.world_to_desk(
            pose,
            tracker_role=str(tracker_role) if tracker_role else None,
        )
    if frame_name.endswith("_operator") or frame_name == "operator":
        return calibration.operator_to_desk(pose)
    if frame_name in {calibration.desk_frame.name, "desk"}:
        return pose
    raise ValueError(f"Unsupported calibration input frame: {frame_name or '<none>'}")


def _runtime_home_calibration(
    components: VirtualRunComponents,
    *,
    arm_mode: str,
    timeout_sec: float,
    num_samples: int,
    progress_callback=None,
    require_stable: bool = False,
    capture_sides: tuple[str, ...] | None = None,
    stability_profile: str = "strict",
) -> None:
    calibration = components.session.calibration
    assert calibration is not None
    active_sides = components.active_sides
    if capture_sides is None:
        home_sides = tuple(active_sides)
    else:
        requested_sides = tuple(dict.fromkeys(str(side) for side in capture_sides))
        invalid_sides = tuple(side for side in requested_sides if side not in active_sides)
        if invalid_sides or not requested_sides:
            raise ValueError(
                "runtime Home capture_sides must be a non-empty subset of active sides; "
                f"requested={requested_sides!r} active={active_sides!r}"
            )
        home_sides = requested_sides
    tracker_expectations = ", ".join(
        f"{side}_wrist_pose<-{calibration.metadata.get(f'{side}_tracker_name') or f'{side}_wrist role'}"
        for side in home_sides
    )
    _log(
        "Calibration",
        f"expecting {tracker_expectations}",
    )
    mapping_summary = ", ".join(
        f"{side}_wrist_pose -> {side}_arm" for side in home_sides
    )
    _log(
        "Calibration",
        f"logical arm mapping: {mapping_summary}",
    )
    try:
        left_arm = calibration.get_arm_for_side("left")
        right_arm = calibration.get_arm_for_side("right")
        if left_arm.T_D_B is not None and right_arm.T_D_B is not None:
            _log(
                "Calibration",
                "visual slots: "
                f"left_arm desk_y={left_arm.T_D_B[1, 3]:+.4f}, "
                f"right_arm desk_y={right_arm.T_D_B[1, 3]:+.4f}",
            )
    except Exception:
        pass
    _log(
        "Calibration",
        "keep active wrist tracker(s) steady in the desired home pose "
        f"for about {max(num_samples / 60.0, 0.5):.1f}s. "
        "This captures per-session tracker home on top of the saved tracker/station calibration.",
    )
    _log(
        "Calibration",
        f"home_capture_mode={_robot_home_mode_label(components)}",
    )
    if _preview_should_follow_live_real_state(arm_mode=arm_mode, components=components):
        synced = _sync_preview_from_real_state(components, required=False)
        _log(
            "Calibration",
            f"start-pose source=real synced={synced}; pressing 'p' will capture tracker home and current real-arm home",
        )
    elif _has_live_real_state(components):
        _log(
            "Calibration",
            "start-pose source=virtual preview; pressing 'p' will capture tracker home and read current real-arm home from the provided arm IPs",
        )
    elif components.initial_alignment is not None:
        _log(
            "Calibration",
            "start-pose source=initial_alignment; pressing 'p' will capture tracker home only, while robot home comes from the saved alignment",
        )
    else:
        _log(
            "Calibration",
            "start-pose source=virtual preview defaults/current local state; pressing 'p' will capture tracker home and current preview arm home",
        )
    calibration_timing_start = time.perf_counter()
    start_time = time.time()
    samples: dict[str, list[np.ndarray]] = {side: [] for side in home_sides}
    profile_name = str(stability_profile or "strict").strip().lower()
    if profile_name not in {"strict", "transition"}:
        raise ValueError(
            "runtime Home stability_profile must be 'strict' or 'transition', "
            f"got {stability_profile!r}"
        )
    _log(
        "Calibration",
        f"stability_profile={profile_name} capture_sides={','.join(home_sides)}",
    )
    stability_factory = (
        HomeCaptureStabilityState.transition_profile
        if profile_name == "transition"
        else HomeCaptureStabilityState
    )
    stability_by_side = {side: stability_factory() for side in home_sides}
    observed_channels: set[str] = set()
    sample_markers = sorted({1, max(1, num_samples // 2), num_samples})
    reported: dict[str, set[int]] = {side: set() for side in home_sides}

    def publish_progress(*, now: float, channels=None, waiting: bool = False) -> None:
        if progress_callback is None:
            return
        valid = {}
        if channels is not None:
            valid = {
                side: bool(
                    channels.get(f"{side}_wrist_pose") is not None
                    and channels[f"{side}_wrist_pose"].valid
                )
                for side in home_sides
            }
        progress_callback(
            sample_counts={side: len(samples[side]) for side in home_sides},
            valid_by_side=valid,
            min_samples=int(num_samples),
            elapsed_sec=max(0.0, now - start_time),
            timeout_sec=float(timeout_sec),
            waiting_for_samples=bool(waiting),
            now=now,
        )

    publish_progress(now=start_time, waiting=False)

    while any(len(samples[side]) < num_samples for side in home_sides):
        now = time.time()
        if now - start_time > timeout_sec:
            counts = {side: len(samples[side]) for side in home_sides}
            publish_progress(now=now, waiting=True)
            raise TimeoutError(
                "Timed out waiting for active wrist tracker(s) during runtime calibration. "
                f"sample_counts={counts}. "
                "Verify sender tracker_role values and receiver tracker-name arguments."
            )
        channels, _ = components.session.poll_sources(now)
        _update_optional_hand_preview_from_channels(
            components,
            channels,
            timestamp=now,
        )
        if components.renderer is not None:
            left_qpos = _state_qpos(components.left_preview_backend, DEFAULT_LEFT_ARM_QPOS)
            right_qpos = _state_qpos(components.right_preview_backend, DEFAULT_RIGHT_ARM_QPOS)
            components.renderer.set_debug_frames(
                _build_renderer_debug_frames(
                    components,
                    channels=channels,
                )
            )
            components.renderer.set_debug_boxes(
                _build_renderer_debug_boxes(
                    components,
                    channels=channels,
                )
            )
            components.renderer.render_data(
                left_qpos_arm=left_qpos,
                left_qpos_hand=_hand_render_qpos(components.left_hand_backend, "left"),
                right_qpos_arm=right_qpos,
                right_qpos_hand=_hand_render_qpos(components.right_hand_backend, "right"),
                capture_image=False,
            )
        for side in home_sides:
            channel = channels.get(f"{side}_wrist_pose")
            if channel is None:
                continue
            if channel.channel_name not in observed_channels:
                metadata = channel.metadata or {}
                _log(
                    "Calibration",
                    "observed "
                    f"{channel.channel_name} <- "
                    f"{metadata.get('tracker_name')}:{metadata.get('tracker_role')} "
                    f"frame={channel.frame_name}",
                    now=now,
                )
                observed_channels.add(channel.channel_name)
            if channel.valid and len(samples[side]) < num_samples:
                pose_desk = _channel_pose_in_desk(channel, calibration)
                if require_stable:
                    decision = stability_by_side[side].observe(
                        pose_desk,
                        source_timestamp=float(channel.timestamp),
                        first_accepted_pose=(
                            samples[side][0] if samples[side] else None
                        ),
                    )
                    if decision.reset_samples:
                        samples[side].clear()
                    if not decision.accepted:
                        continue
                samples[side].append(pose_desk)
                for marker in sample_markers:
                    if len(samples[side]) >= marker and marker not in reported[side]:
                        reported[side].add(marker)
                        _log(
                            "Calibration",
                            f"{side} progress {len(samples[side])}/{num_samples}",
                            now=now,
                        )
        if any(len(samples[side]) < num_samples for side in home_sides):
            publish_progress(
                now=now,
                channels=channels,
                waiting=True,
            )
            time.sleep(0.01)

    sample_done_perf = time.perf_counter()
    sample_elapsed_ms = (sample_done_perf - calibration_timing_start) * 1000.0
    publish_progress(now=time.time(), waiting=False)

    sync_elapsed_ms = 0.0
    if _preview_should_follow_live_real_state(arm_mode=arm_mode, components=components):
        sync_start = time.perf_counter()
        synced = _sync_preview_from_real_state(components, required=False)
        sync_elapsed_ms = (time.perf_counter() - sync_start) * 1000.0
        _log(
            "Calibration",
            f"locked preview start pose from real state synced={synced} "
            f"elapsed={sync_elapsed_ms:.1f}ms",
        )
    home_summaries: list[str] = []
    qpos_summaries: list[str] = []
    side_timing_summaries: list[str] = []
    post_sample_start = time.perf_counter()
    for side in home_sides:
        side_start = time.perf_counter()
        mean_start = time.perf_counter()
        human_home = _mean_pose(samples[side])
        mean_elapsed_ms = (time.perf_counter() - mean_start) * 1000.0
        robot_home_start = time.perf_counter()
        robot_home_desk, qpos, robot_source = _resolve_robot_home_for_side(
            components,
            side=side,
        )
        robot_home_elapsed_ms = (time.perf_counter() - robot_home_start) * 1000.0
        anchor_start = time.perf_counter()
        _side_retargeter(components, side).set_home_anchors(
            human_pose_desk=human_home,
            robot_pose_desk=robot_home_desk,
        )
        anchor_elapsed_ms = (time.perf_counter() - anchor_start) * 1000.0
        capture_start = time.perf_counter()
        calibration.capture_home_pose(f"{side}_arm", robot_home_desk)
        if qpos is not None:
            calibration.capture_home_qpos(f"{side}_arm", qpos)
        capture_elapsed_ms = (time.perf_counter() - capture_start) * 1000.0
        side_elapsed_ms = (time.perf_counter() - side_start) * 1000.0
        home_summaries.append(
            f"{side}_source={robot_source} {side}_xyz={_pose_translation_summary(robot_home_desk)}"
        )
        qpos_summaries.append(f"{side}_q={_qpos_summary(qpos)}")
        side_timing_summaries.append(
            f"{side}:total={side_elapsed_ms:.1f}ms "
            f"mean={mean_elapsed_ms:.1f}ms "
            f"robot_home={robot_home_elapsed_ms:.1f}ms "
            f"anchors={anchor_elapsed_ms:.1f}ms "
            f"capture={capture_elapsed_ms:.1f}ms "
            f"source={robot_source}"
        )
    post_sample_elapsed_ms = (time.perf_counter() - post_sample_start) * 1000.0
    _log(
        "CalibrationTiming",
        f"runtime_home sampling={sample_elapsed_ms:.1f}ms "
        f"post_sample={post_sample_elapsed_ms:.1f}ms "
        f"real_state_sync={sync_elapsed_ms:.1f}ms "
        + " | ".join(side_timing_summaries),
    )
    _log("Calibration", "robot_home " + " ".join(home_summaries))
    sample_counts = {side: len(values) for side, values in samples.items()}
    _log(
        "Calibration",
        f"runtime home captured sample_counts={sample_counts} "
        + " ".join(qpos_summaries),
    )


def _wait_for_calibration_start(
    components: VirtualRunComponents,
    clutch_input: TerminalClutchInput,
    *,
    arm_mode: str,
) -> bool:
    if not clutch_input.enabled:
        _log(
            "Calibration",
            "stdin is not interactive; starting automatically. "
            "Use --auto-calibrate explicitly in non-interactive runs.",
        )
        return True

    _log("Calibration", "HOME CAPTURE IS ARMED. Position robot and trackers first, then press 'p' or Enter.",)
    if components.initial_alignment is not None:
        _log(
            "Calibration",
            "P action: capture tracker home only. Robot home will be loaded from --load-initial-alignment.",
        )
    elif _has_live_real_state(components):
        _log(
            "Calibration",
            "P action: capture tracker home and current real-arm home together.",
        )
    else:
        _log(
            "Calibration",
            "P action: capture tracker home and current preview arm home together.",
        )
    _log("Calibration", "Press 'q' to quit.")
    status_gate = ChangeGate()

    def wait_after_start_request(
        delay_sec: float = DEFAULT_RUNTIME_HOME_START_DELAY_SEC,
    ) -> bool:
        hook = getattr(clutch_input, "before_calibration_start_delay", None)
        if callable(hook):
            return bool(hook(delay_sec, arm_mode=arm_mode))
        deadline = time.time() + max(0.0, float(delay_sec))
        _log(
            "Calibration",
            f"start requested; waiting {delay_sec:.1f}s before capture so the operator can react.",
        )
        while True:
            remaining = deadline - time.time()
            if remaining <= 0.0:
                return True
            clutch_input.poll()
            if clutch_input.should_quit:
                return False
            time.sleep(min(0.05, remaining))

    while True:
        loop_start = time.time()
        clutch_input.poll()
        if clutch_input.should_quit:
            return False
        if clutch_input.consume_calibration_request():
            return wait_after_start_request()

        if _preview_should_follow_live_real_state(arm_mode=arm_mode, components=components):
            _sync_preview_from_real_state(components, required=False)

        channels, _ = components.session.poll_sources(loop_start)
        publish_raw_tracker_pose_state = getattr(
            clutch_input,
            "publish_raw_tracker_pose_state",
            None,
        )
        if callable(publish_raw_tracker_pose_state):
            publish_raw_tracker_pose_state(channels=channels, now=loop_start)
        _update_optional_hand_preview_from_channels(
            components,
            channels,
            timestamp=loop_start,
        )
        if components.renderer is not None:
            left_qpos = _state_qpos(components.left_preview_backend, DEFAULT_LEFT_ARM_QPOS)
            right_qpos = _state_qpos(components.right_preview_backend, DEFAULT_RIGHT_ARM_QPOS)
            components.renderer.set_debug_frames(
                _build_renderer_debug_frames(
                    components,
                    channels=channels,
                )
            )
            components.renderer.set_debug_boxes(
                _build_renderer_debug_boxes(
                    components,
                    channels=channels,
                )
            )
            components.renderer.render_data(
                left_qpos_arm=left_qpos,
                left_qpos_hand=_hand_render_qpos(components.left_hand_backend, "left"),
                right_qpos_arm=right_qpos,
                right_qpos_hand=_hand_render_qpos(components.right_hand_backend, "right"),
                capture_image=False,
            )

        ready = {
            name: bool(channels.get(name) is not None and channels[name].valid)
            for name in tuple(f"{side}_wrist_pose" for side in components.active_sides)
        }
        status_snapshot = (
            tuple(sorted(ready.items())),
        )
        if status_gate.should_emit("wait_for_key_status", status_snapshot, now=loop_start):
            _log(
                "Calibration",
                f"waiting_for_key tracker_ready={ready} ready_key='p/Enter' arm_mode={arm_mode}",
                now=loop_start,
            )
        elif status_gate.should_emit(
            "wait_for_key_heartbeat",
            "waiting",
            now=loop_start,
            heartbeat_sec=3.0,
        ):
            _log(
                "Calibration",
                f"still_waiting ready={ready} ready_key='p/Enter'",
                now=loop_start,
            )
        time.sleep(0.02)


def _sync_dual_arm_retargeter_from_preview(
    components: VirtualRunComponents,
    *,
    reset_tracking: bool = True,
) -> None:
    if components.dual_arm_retargeter is None:
        return
    left_qpos = components.left_preview_backend.get_state()["current_qpos"]
    right_qpos = components.right_preview_backend.get_state()["current_qpos"]
    if left_qpos is None or right_qpos is None:
        return
    try:
        components.dual_arm_retargeter.set_current_qpos_pair(
            left_qpos=left_qpos,
            right_qpos=right_qpos,
            reset_tracking=reset_tracking,
        )
    except TypeError as exc:
        if "reset_tracking" not in str(exc):
            raise
        components.dual_arm_retargeter.set_current_qpos_pair(
            left_qpos=left_qpos,
            right_qpos=right_qpos,
        )


def _preview_tcp_pose_desk(
    components: VirtualRunComponents,
    *,
    side: str,
) -> np.ndarray:
    calibration = components.session.calibration
    assert calibration is not None
    backend = (
        components.left_preview_backend
        if side == "left"
        else components.right_preview_backend
    )
    pose_base = backend.compute_tcp_pose()
    return calibration.get_arm_for_side(side).T_D_B @ pose_base


def _send_preview_command(
    components: VirtualRunComponents,
    command,
) -> None:
    left_command = command.commands.get("left_arm")
    if left_command is not None:
        components.left_preview_backend.send_command(left_command)
    right_command = command.commands.get("right_arm")
    if right_command is not None:
        components.right_preview_backend.send_command(right_command)


def _hand_observation_for_step(
    components: VirtualRunComponents,
    step_result,
) -> TeleopObservation | None:
    if step_result.observation is not None:
        return step_result.observation
    return _hand_observation_from_channels(
        components,
        step_result.channels,
        timestamp=step_result.timestamp,
    )


def _hand_observation_from_channels(
    components: VirtualRunComponents,
    channels,
    *,
    timestamp: float | None = None,
) -> TeleopObservation | None:
    if not channels:
        return None
    observation_timestamp = latest_channel_timestamp(channels)
    if observation_timestamp is None:
        observation_timestamp = time.time() if timestamp is None else float(timestamp)
    return TeleopObservation(
        channels={name: channel.copy() for name, channel in channels.items()},
        timestamp=observation_timestamp,
        metadata={"tcp_mode": components.session.config.tcp_mode},
    )


def _build_optional_hand_command_from_observation(
    components: VirtualRunComponents,
    observation: TeleopObservation | None,
):
    if (
        components.left_hand_retargeter is None
        and components.right_hand_retargeter is None
    ):
        return None
    if observation is None:
        return None
    commands = []
    if components.left_hand_retargeter is not None:
        commands.append(
            components.left_hand_retargeter.retarget(
                observation,
                components.session.calibration,
            )
        )
    if components.right_hand_retargeter is not None:
        commands.append(
            components.right_hand_retargeter.retarget(
                observation,
                components.session.calibration,
            )
        )
    if not commands:
        return None
    command = merge_teleop_commands(commands)
    if not command.commands:
        return None
    return command


def _build_optional_hand_command(
    components: VirtualRunComponents,
    step_result,
):
    return _build_optional_hand_command_from_observation(
        components,
        _hand_observation_for_step(components, step_result),
    )


def _send_optional_hand_command(
    components: VirtualRunComponents,
    hand_command,
) -> None:
    if hand_command is None:
        return
    left_command = hand_command.commands.get("left_hand")
    if left_command is not None and components.left_hand_backend is not None:
        components.left_hand_backend.send_command(left_command)
    right_command = hand_command.commands.get("right_hand")
    if right_command is not None and components.right_hand_backend is not None:
        components.right_hand_backend.send_command(right_command)


def _update_optional_hand_preview_from_channels(
    components: VirtualRunComponents,
    channels,
    *,
    timestamp: float | None = None,
):
    hand_command = _build_optional_hand_command_from_observation(
        components,
        _hand_observation_from_channels(
            components,
            channels,
            timestamp=timestamp,
        ),
    )
    _send_optional_hand_command(components, hand_command)
    return hand_command


def _hand_state_q6(backend: InspireTacHandBackend | None) -> np.ndarray | None:
    if backend is None:
        return None
    joint_pos = backend.get_state().get("joint_pos")
    if joint_pos is None:
        return None
    q6 = np.asarray(joint_pos, dtype=float).reshape(-1)
    if q6.shape != (6,) or not np.all(np.isfinite(q6)):
        return None
    return np.clip(q6, 0.0, 1.0)


def _hand_render_qpos(backend: InspireTacHandBackend | None, side: str) -> np.ndarray:
    q6 = _hand_state_q6(backend)
    if q6 is None:
        return open_hand_render_qpos(side)
    return inspire_q6_to_render_qpos(q6)


def _episode_root_dir(
    args: argparse.Namespace,
    components: VirtualRunComponents,
) -> Path | None:
    if args.episode_root:
        return Path(args.episode_root)
    if components.session.recorder is not None:
        return Path(components.session.recorder.root_dir) / "trajectories"
    return None


def _build_episode_metadata(
    args: argparse.Namespace,
    components: VirtualRunComponents,
    *,
    start_reason: str,
) -> dict[str, object]:
    calibration = components.session.calibration
    assert calibration is not None
    session_record_root = (
        None if components.session.recorder is None else str(components.session.recorder.root_dir)
    )
    return {
        "source_entrypoint": "teleop_runtime.debug.run_vive_sapien_virtual",
        "start_reason": start_reason,
        "control_mode": args.control_mode,
        "arm_mode": args.arm_mode,
        "active_sides": list(components.active_sides),
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
        "left_hand_port": getattr(args, "left_hand_port", "") or None,
        "right_hand_port": getattr(args, "right_hand_port", "") or None,
        "hand_reader_baudrate": int(getattr(args, "hand_reader_baudrate", 115200)),
        "hand_joint_order": list(getattr(args, "hand_joint_order", DEFAULT_INSPIRE_HAND_JOINT_ORDER)),
        "hand_invert": bool(getattr(args, "hand_invert", True)),
        "hand_command_alpha": float(getattr(args, "hand_command_alpha", 1.0)),
        "hand_send_threshold": float(getattr(args, "hand_send_threshold", 0.002)),
        "calibration_snapshot": calibration.to_dict(),
        "initial_alignment_snapshot": (
            None if components.initial_alignment is None else components.initial_alignment.to_dict()
        ),
    }


def _start_episode_recording(
    recorder: EpisodeRecorder,
    args: argparse.Namespace,
    components: VirtualRunComponents,
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


def _backend_tcp_pose_base_or_none(backend: RM65ArmBackend) -> np.ndarray | None:
    try:
        return backend.compute_tcp_pose()
    except Exception:
        return None


def _backend_tcp_pose_desk_or_none(
    components: VirtualRunComponents,
    *,
    side: str,
) -> np.ndarray | None:
    try:
        return _preview_tcp_pose_desk(components, side=side)
    except Exception:
        return None


def _episode_arm_payload(
    components: VirtualRunComponents,
    *,
    side: str,
) -> dict[str, object]:
    backend = (
        components.left_preview_backend
        if side == "left"
        else components.right_preview_backend
    )
    state = backend.get_state()
    return {
        "current_qpos": state.get("current_qpos"),
        "last_target_qpos": state.get("last_target_qpos"),
        "last_command": state.get("last_command"),
        "ik_metadata": state.get("ik_metadata"),
        "tcp_pose_base": _backend_tcp_pose_base_or_none(backend),
        "tcp_pose_desk": _backend_tcp_pose_desk_or_none(components, side=side),
    }


def _episode_hand_payload(
    components: VirtualRunComponents,
    *,
    side: str,
) -> dict[str, object] | None:
    backend = (
        components.left_hand_backend
        if side == "left"
        else components.right_hand_backend
    )
    if backend is None:
        return None
    state = backend.get_state()
    return {
        "joint_pos": state.get("joint_pos"),
        "last_sent_command": state.get("last_sent_command"),
        "tactile": state.get("tactile"),
        "tactile_dict": state.get("tactile_dict"),
        "telemetry_error": state.get("telemetry_error"),
    }


def _build_episode_frame(
    components: VirtualRunComponents,
    *,
    step_index: int,
    step_result,
    active_command,
    hand_command,
    recovery_plan: RecoveryPlan | None,
    left_clutch: bool,
    right_clutch: bool,
) -> dict[str, object]:
    if step_result.state == TeleopRunState.RECOVERING:
        if active_command is None:
            active_command_source = "recovery_wait"
        elif recovery_plan is None:
            active_command_source = "recovery_direct"
        else:
            active_command_source = "recovery_plan"
    elif active_command is None:
        active_command_source = "none"
    else:
        active_command_source = "session"

    return {
        "step_index": int(step_index),
        "state": step_result.state.value,
        "active_sides": list(components.active_sides),
        "control_hz": float(step_result.control_hz),
        "active_command_source": active_command_source,
        "recovery_plan_mode": None if recovery_plan is None else recovery_plan.mode,
        "channels": dict(step_result.channels),
        "channel_validity": {
            name: bool(channel.valid)
            for name, channel in sorted(step_result.channels.items())
        },
        "clutch": {
            "left": bool(left_clutch),
            "right": bool(right_clutch),
        },
        "faults": list(step_result.faults),
        "session_command": step_result.command,
        "active_command": active_command,
        "hand_command": hand_command,
        "arms": {
            "left": _episode_arm_payload(components, side="left"),
            "right": _episode_arm_payload(components, side="right"),
        },
        "hands": {
            "left": _episode_hand_payload(components, side="left"),
            "right": _episode_hand_payload(components, side="right"),
        },
    }


def _start_recovery_plan(
    components: VirtualRunComponents,
    observation,
    *,
    now: float,
    arm_mode: str,
    control_mode: str,
    ramp_sec: float,
) -> tuple[RecoveryPlan | None, object | None, bool]:
    if control_mode != "absolute":
        raise ValueError("delta_clutch has been hard-archived; recovery only supports 'absolute'.")
    calibration = components.session.calibration
    assert calibration is not None
    _sync_dual_arm_retargeter_from_preview(components)

    if arm_mode == "virtual":
        return None, components.session.build_command(observation), True

    if components.dual_arm_retargeter is not None:
        context = components.dual_arm_retargeter.build_target_context(observation, calibration)
        return (
            RecoveryPlan(
                mode="absolute_ramp_dual",
                started_at=now,
                duration_sec=float(ramp_sec),
                payload={
                    "context": context,
                    "start_targets": components.dual_arm_retargeter.current_world_targets(),
                    "end_targets": {
                        "left": context.left_target.copy(),
                        "right": context.right_target.copy(),
                    },
                },
            ),
            None,
            False,
        )

    end_resolutions: dict[str, dict[str, object]] = {}
    for side in components.active_sides:
        resolution = _side_retargeter(components, side).resolve_absolute_target_from_observation(
            observation,
            calibration,
        )
        if resolution is None:
            raise RuntimeError(
                f"Absolute recovery requires active wrist channel '{side}_wrist_pose' to be valid."
            )
        end_resolutions[side] = resolution
    return (
        RecoveryPlan(
            mode="absolute_ramp_independent",
            started_at=now,
            duration_sec=float(ramp_sec),
            payload={
                "start_targets": {
                    side: _preview_tcp_pose_desk(components, side=side)
                    for side in components.active_sides
                },
                "end_resolutions": end_resolutions,
            },
        ),
        None,
        False,
    )


def _advance_recovery_plan(
    components: VirtualRunComponents,
    observation,
    plan: RecoveryPlan,
    *,
    now: float,
) -> tuple[object, bool]:
    calibration = components.session.calibration
    assert calibration is not None
    alpha = 1.0
    if plan.duration_sec > 1e-6:
        alpha = float(np.clip((now - plan.started_at) / plan.duration_sec, 0.0, 1.0))

    if plan.mode == "absolute_ramp_dual":
        assert components.dual_arm_retargeter is not None
        context = plan.payload["context"]
        start_targets = plan.payload["start_targets"]
        end_targets = plan.payload["end_targets"]
        command = components.dual_arm_retargeter.build_command_for_context(
            context,
            targets=components.dual_arm_retargeter.interpolate_world_targets(
                start_targets,
                end_targets,
                alpha,
            ),
        )
        command.timestamp = observation.timestamp
        return command, alpha >= 1.0

    if plan.mode == "absolute_ramp_independent":
        start_targets = plan.payload["start_targets"]
        end_resolutions = plan.payload["end_resolutions"]
        commands = []
        for side, resolution in end_resolutions.items():
            target_pose = interpolate_pose(
                start_targets[side],
                resolution["target_pose_desk"],
                alpha,
            )
            commands.append(
                _side_retargeter(components, side).build_command_from_target_pose_desk(
                    target_pose_desk=target_pose,
                    observation=observation,
                    calibration=calibration,
                    pose_operator=resolution.get("pose_operator"),
                )
            )
        command = merge_teleop_commands(commands)
        command.timestamp = observation.timestamp
        return command, alpha >= 1.0

    raise ValueError(f"Unsupported recovery plan mode: {plan.mode}")


def main() -> None:
    args = parse_args()
    components = build_components(args)
    clutch_input = TerminalClutchInput(components.keyboard_source)
    episode_recorder = EpisodeRecorder(
        root_dir=_episode_root_dir(args, components),
        config=EpisodeRecorderConfig(enabled=not args.no_episode_recording),
    )
    try:
        components.left_preview_backend.start()
        components.right_preview_backend.start()
        if components.left_state_backend is not None:
            components.left_state_backend.start()
        if components.right_state_backend is not None:
            components.right_state_backend.start()
        if components.left_hand_backend is not None:
            components.left_hand_backend.start()
        if components.right_hand_backend is not None:
            components.right_hand_backend.start()
        components.session.start()
        clutch_input.start()
        if args.arm_mode == "real":
            if components.initial_alignment is not None:
                _log(
                    "ArmMode",
                    "real-mode recovery behavior is enabled, but preview startup pose is loaded from "
                    f"{args.load_initial_alignment}; no real-arm state read is needed, and robot home at 'p' time will also come from this alignment.",
                )
            else:
                _log(
                    "ArmMode",
                    "real mode: will read current RM65 start pose and copy it into the local preview; no motion command is sent to hardware.",
                )
                if not args.left_arm_ip and not args.right_arm_ip:
                    _log(
                        "ArmMode",
                        "no real arm IP provided; falling back to virtual preview state for both sides.",
                    )
            _log(
                "ArmMode",
                f"preview motion is step-limited at hz={args.hz:.1f} using dry-run IK, rather than instant dry-run convergence.",
            )
        else:
            _log("ArmMode", "virtual mode: rendering uses local unlimited-speed preview backends only.")
            if _has_live_real_state(components):
                _log(
                    "ArmMode",
                    "arm IPs are present in virtual mode; preview startup pose stays local, but pressing 'p' will sample current real-arm home read-only.",
                )
            if components.initial_alignment is not None:
                _log(
                    "ArmMode",
                    "virtual mode already avoids any real-arm state read; "
                    "load-initial-alignment will set the initial preview qpos and also provide robot home at 'p' time.",
                )
        if components.initial_alignment is not None:
            _log(
                "InitialAlignment",
                f"source={args.load_initial_alignment} "
                f"left_q={_qpos_summary(components.initial_alignment.get_side('left').arm_qpos if components.initial_alignment.get_side('left') is not None else None)} "
                f"right_q={_qpos_summary(components.initial_alignment.get_side('right').arm_qpos if components.initial_alignment.get_side('right') is not None else None)}",
            )
        if components.left_hand_retargeter is not None or components.right_hand_retargeter is not None:
            _log(
                "HandTeleop",
                "enabled "
                f"left_port={getattr(args, 'left_hand_port', '') or None} "
                f"right_port={getattr(args, 'right_hand_port', '') or None} "
                f"joint_order={list(getattr(args, 'hand_joint_order', DEFAULT_INSPIRE_HAND_JOINT_ORDER))} "
                f"invert={bool(getattr(args, 'hand_invert', True))} "
                f"alpha={float(getattr(args, 'hand_command_alpha', 1.0)):.3f}",
            )
        else:
            _log("HandTeleop", "disabled; arm teleop path remains unchanged.")
        if components.session.calibration.metadata.get("virtual_desk_scene_alignment_defaulted"):
            _log(
                "VirtualDesk",
                "applied default desk-to-scene yaw 180deg so tracker poses align with the SAPIEN/RM65 scene.",
            )
        _emit_tracker_axis_mapping_logs(components)
        _log(
            "IK",
            f"objective_mode={components.session.calibration.metadata.get('ik_objective_mode')} "
            f"tuning={components.session.calibration.metadata.get('ik_tuning_source')}",
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
        if components.initial_alignment is not None:
            _log(
                "Calibration",
                "load-initial-alignment will provide robot home at 'p' time. "
                "Press 'p' after both wrist trackers are in the desired session home pose.",
            )
        if _manual_calibration_confirmation_required(args, components):
            should_continue = _wait_for_calibration_start(
                components,
                clutch_input,
                arm_mode=args.arm_mode,
            )
            if not should_continue:
                return
        _runtime_home_calibration(
            components,
            arm_mode=args.arm_mode,
            timeout_sec=args.calibration_timeout,
            num_samples=args.calibration_samples,
        )
        _emit_home_alignment_logs(components)
        if legacy_tcp_mode_requested(args.requested_tcp_mode):
            _log(
                "TCP",
                "legacy --tcp-mode rm_eef was requested; runtime now uses tool_tcp actions.",
            )
        components.session.set_calibration_ready()
        components.session.mark_debug_passed()
        components.session.arm_hardware()
        clutch_input.consume_episode_start_request()
        clutch_input.consume_episode_stop_request()

        if args.keyboard_clutch:
            _log(
                "Keys",
                "p/Enter=start calibration, t=start episode, s=stop episode, z=left clutch, x=right clutch, b=both clutch, r=release, q=quit",
            )
        if episode_recorder.config.enabled:
            auto_start_note = " auto_start=on" if args.record_auto_start else " auto_start=off"
            _log(
                "Episode",
                f"recording ready root={episode_recorder.root_dir or _episode_root_dir(args, components)} "
                f"start_key='t' stop_key='s'{auto_start_note}",
            )

        status_gate = ChangeGate()
        orientation_gate = ChangeGate()
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

            step_result = components.session.run_step(now=loop_start)
            active_command = step_result.command
            if step_result.state == TeleopRunState.RECOVERING:
                if step_result.observation is None:
                    active_command = None
                elif recovery_plan is None:
                    recovery_plan, active_command, recovery_done = _start_recovery_plan(
                        components,
                        step_result.observation,
                        now=loop_start,
                        arm_mode=args.arm_mode,
                        control_mode=args.control_mode,
                        ramp_sec=args.recovery_ramp_sec,
                    )
                    if not episode_recorder.is_recording:
                        if recovery_plan is None:
                            _log(
                                "Recovery",
                                f"direct mode arm_mode={args.arm_mode} control_mode={args.control_mode}",
                                now=loop_start,
                            )
                        else:
                            _log(
                                "Recovery",
                                f"start mode={recovery_plan.mode} duration={recovery_plan.duration_sec:.2f}s",
                                now=loop_start,
                            )
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
                    if recovery_done:
                        components.session.finish_recovery("recovery completed")
                        if not episode_recorder.is_recording:
                            _log("Recovery", "completed", now=loop_start)
                        recovery_plan = None
            else:
                recovery_plan = None

            if active_command is not None:
                _send_preview_command(components, active_command)
                if args.arm_mode == "real":
                    _sync_dual_arm_retargeter_from_preview(components)

            hand_command = _build_optional_hand_command(components, step_result)
            _send_optional_hand_command(components, hand_command)

            left_qpos = _state_qpos(components.left_preview_backend, DEFAULT_LEFT_ARM_QPOS)
            right_qpos = _state_qpos(components.right_preview_backend, DEFAULT_RIGHT_ARM_QPOS)
            if components.renderer is not None:
                components.renderer.set_debug_frames(
                    _build_renderer_debug_frames(
                        components,
                        channels=step_result.channels,
                    )
                )
                components.renderer.set_debug_boxes(
                    _build_renderer_debug_boxes(
                        components,
                        channels=step_result.channels,
                    )
                )
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
            )
            if (
                not episode_recorder.is_recording
                and status_gate.should_emit("virtual_run_status", status_snapshot, now=loop_start)
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
                    "VirtualRun",
                    f"step={step_index} hz={step_result.control_hz:.1f} "
                    f"state={step_result.state.value} "
                    f"channels={sorted(step_result.channels)} "
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
        if components.left_hand_backend is not None:
            components.left_hand_backend.close()
        if components.right_hand_backend is not None:
            components.right_hand_backend.close()
        components.left_preview_backend.close()
        components.right_preview_backend.close()
        if components.left_state_backend is not None:
            components.left_state_backend.close()
        if components.right_state_backend is not None:
            components.right_state_backend.close()


if __name__ == "__main__":
    main()
