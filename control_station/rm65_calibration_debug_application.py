from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty
import warnings
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation

from project_paths import calibration_profiles_dir

from teleop_runtime.backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.calibration import TeleopCalibration
from teleop_runtime.configs import (
    ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
    build_annotated_dual_rm65_calibration,
)
from teleop_runtime.arm_frames import TOOL_TCP_FRAME
from teleop_runtime.hand_tcp_alignment import tracker_hand_to_rm65_tcp_rotation
from teleop_runtime.retargeters import PoseToArmRetargeter, PoseToArmRetargeterConfig
from teleop_runtime.schemas import TeleopObservation, latest_channel_timestamp, merge_teleop_commands
from teleop_runtime.sources.pose import (
    DEFAULT_VIVE_CLOCK_SYNC_PORT,
    VivePoseSource,
    ViveTrackerConfig,
)
from teleop_runtime.tcp import legacy_tcp_mode_requested
from teleop_runtime.tracker_hand import (
    TRACKER_OFFSET_CALIBRATION_VERSION,
    canonical_hand_rpy_default,
)
from teleop_runtime.transforms import as_pose_matrix, identity_pose, invert_pose
from teleop_runtime.debug.calibration_flow import (
    average_pose,
    compose_tracker_offset,
    euler_xyz_deg_to_matrix,
    fit_signed_axis_from_poses,
    fit_virtual_desk_frame,
    pose_from_rotation_translation,
    tracker_offset_from_neutral_pose,
)
from control_station.web.calibration_web_ui import CalibrationWebUi, WebUiKeyInput
from teleop_runtime.debug.log_utils import DEFAULT_LOG_HEARTBEAT_SEC, ChangeGate, log_event, log_stage
from robot_motion.rm65.assets import require_asset_path
from robot_motion.rm65 import DEFAULT_LEFT_ARM_QPOS, DEFAULT_RIGHT_ARM_QPOS
from sim.rm65.sapien_renderer import SapienArmHandRenderer

DEFAULT_CALIBRATION_PROFILE_DIR = calibration_profiles_dir()
_VIRTUAL_DESK_SWEEP_CONVENTION_VERSION = 2


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _non_negative_float(text: str) -> float:
    value = float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _format_wall_time(timestamp: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(float(timestamp)))


def _channel_sample_key(channel) -> tuple[object, object, object]:
    metadata = channel.metadata or {}
    seq = metadata.get("seq")
    receiver_timestamp = metadata.get("receiver_timestamp")
    return (seq, receiver_timestamp, channel.timestamp)


def _parse_xyz(text: str) -> np.ndarray:
    parts = [item.strip() for item in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected 3 comma-separated values")
    try:
        return np.asarray([float(item) for item in parts], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric comma-separated values") from exc


def _pose_xyz_rpy_summary(channel, *, now: float) -> dict[str, object]:
    if channel is None:
        return {
            "valid": False,
            "xyz_m": None,
            "rpy_deg": None,
            "age_sec": None,
            "tracker_name": "",
            "tracker_role": "",
            "seq": None,
        }
    metadata = channel.metadata or {}
    pose = as_pose_matrix(channel.value)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpy_deg = SciPyRotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
    except ValueError:
        rpy_deg = np.full(3, np.nan, dtype=float)
    rpy_deg = np.round(rpy_deg.astype(float), 1)
    rpy_values = [float(value) if np.isfinite(value) else None for value in rpy_deg]
    timestamp = metadata.get("receiver_timestamp")
    if timestamp is None:
        timestamp = channel.timestamp
    age_sec = max(0.0, now - float(timestamp)) if timestamp is not None else None
    return {
        "valid": bool(channel.valid),
        "xyz_m": np.round(pose[:3, 3].astype(float), 4).tolist(),
        "rpy_deg": rpy_values,
        "age_sec": round(age_sec, 3) if age_sec is not None else None,
        "tracker_name": str(metadata.get("tracker_name") or ""),
        "tracker_role": str(metadata.get("tracker_role") or ""),
        "seq": metadata.get("seq"),
    }


def _pose_matrix_xyz_rpy_text(pose: np.ndarray) -> str:
    pose_matrix = as_pose_matrix(pose)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpy_deg = SciPyRotation.from_matrix(pose_matrix[:3, :3]).as_euler("xyz", degrees=True)
    except ValueError:
        rpy_deg = np.full(3, np.nan, dtype=float)
    xyz = np.array2string(pose_matrix[:3, 3], precision=4, suppress_small=True)
    rpy = np.array2string(rpy_deg, precision=1, suppress_small=True)
    return f"xyz_m={xyz} rpy_deg={rpy}"


def _qpos_deg_text(qpos: np.ndarray | None) -> str:
    if qpos is None:
        return "None"
    return np.array2string(np.rad2deg(np.asarray(qpos, dtype=float)), precision=1, suppress_small=True)


class TerminalKeyInput:
    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self._old_attrs: list[int] | None = None
        self.should_quit = False
        self._events: list[str] = []

    def start(self) -> None:
        if not self.enabled:
            return
        self._old_attrs = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())

    def close(self) -> None:
        if self.enabled and self._old_attrs is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_attrs)
            self._old_attrs = None

    def poll(self) -> None:
        if not self.enabled:
            return
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                return
            char = sys.stdin.read(1)
            if not char:
                return
            if char == "q":
                self.should_quit = True
            self._events.append(char)

    def consume_events(self) -> list[str]:
        events = list(self._events)
        self._events.clear()
        return events


@dataclass(frozen=True)
class CalibrationDebugComponents:
    calibration: TeleopCalibration
    source: VivePoseSource
    left_backend: RM65ArmBackend
    right_backend: RM65ArmBackend
    left_retargeter: PoseToArmRetargeter
    right_retargeter: PoseToArmRetargeter
    renderer: SapienArmHandRenderer | None
    ui: CalibrationWebUi | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive VIVE -> RM65 calibration debug tool. "
            "This app calibrates a virtual desk frame from an auxiliary tracker, "
            "captures left/right wrist tracker offsets, captures session home anchors, "
            "and saves a reusable profile."
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
        "--tcp-mode",
        type=str,
        default=TOOL_TCP_FRAME,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--arm-mode",
        choices=["real", "virtual"],
        default="real",
        help="Render read-only real RM65 state when IPs are provided, or pure virtual arm poses.",
    )
    parser.add_argument(
        "--left-arm-ip",
        type=str,
        default=os.environ.get("LEFT_ARM_IP", ""),
        help="Optional left RM65 IP for read-only rendering; defaults to LEFT_ARM_IP.",
    )
    parser.add_argument(
        "--right-arm-ip",
        type=str,
        default=os.environ.get("RIGHT_ARM_IP", ""),
        help="Optional right RM65 IP for read-only rendering; defaults to RIGHT_ARM_IP.",
    )
    parser.add_argument(
        "--left-right-base-distance",
        type=float,
        default=ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
        help="Annotated dual-arm base distance used by the SAPIEN scene and preset calibration.",
    )
    parser.add_argument("--left-tracker-name", type=str, default="tracker_1", help="Left wrist tracker device name")
    parser.add_argument("--right-tracker-name", type=str, default="tracker_2", help="Right wrist tracker device name")
    parser.add_argument(
        "--aux-tracker-name",
        type=str,
        default="",
        help="Optional auxiliary calibration tracker device name. Empty disables the third tracker.",
    )
    parser.add_argument(
        "--virtual-desk-tracker-role",
        choices=["auto", "auxiliary", "left_wrist", "right_wrist"],
        default="auto",
        help=(
            "Tracker role used for virtual-desk sweeps. 'auto' prefers auxiliary when present, "
            "then left_wrist, then right_wrist, so a two-tracker setup is sufficient."
        ),
    )
    parser.add_argument("--headless", action="store_true", help="Disable the SAPIEN viewer window")
    parser.add_argument(
        "--skip-renderer",
        action="store_true",
        help="Skip SAPIEN renderer creation. Useful when only terminal validation is needed.",
    )
    parser.add_argument(
        "--ui-mode",
        choices=["terminal", "web"],
        default="web",
        help="Choose terminal key input or a browser-based control panel.",
    )
    parser.add_argument(
        "--gui",
        dest="ui_mode",
        action="store_const",
        const="web",
        help="Use the browser-based calibration control panel. Alias for --ui-mode web.",
    )
    parser.add_argument(
        "--no-gui",
        dest="ui_mode",
        action="store_const",
        const="terminal",
        help="Use terminal keyboard controls only. Alias for --ui-mode terminal.",
    )
    parser.add_argument(
        "--ui-host",
        type=str,
        default="127.0.0.1",
        help="Host used by the browser control panel HTTP server.",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=0,
        help="Port used by the browser control panel HTTP server; 0 picks an available port.",
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Do not auto-open the browser when using --ui-mode web.",
    )
    parser.add_argument(
        "--show-tracker-frame-viz",
        action="store_true",
        help=(
            "After coarse wrist calibration, enter an interactive SAPIEN stage that visualizes "
            "desk/base frames plus left/right raw tracker and calibrated hand frames."
        ),
    )
    parser.add_argument(
        "--frame-viz-mode",
        choices=["calibrated", "raw", "both"],
        default="calibrated",
        help="Which tracker-related frames to show in SAPIEN debug visualization.",
    )
    parser.add_argument("--hz", type=_positive_float, default=30.0, help="Target update rate for capture and validation")
    parser.add_argument(
        "--print-hz",
        type=_positive_float,
        default=2.0,
        help="Legacy console progress throttle. Most logs are now emitted only on change.",
    )
    parser.add_argument("--translation-scale", type=float, default=1.0, help="Translation scale used during validation teleop")
    parser.add_argument(
        "--sweep-capture-sec",
        type=_positive_float,
        default=5.0,
        help="Timed capture duration for each VirtualDesk X/Y/Z sweep",
    )
    parser.add_argument(
        "--capture-prep-sec",
        type=_non_negative_float,
        default=3.0,
        help="Preparation countdown before each timed capture starts.",
    )
    parser.add_argument(
        "--wrist-capture-prep-sec",
        type=_non_negative_float,
        default=5.0,
        help="Preparation countdown before the combined wrist/home capture starts.",
    )
    parser.add_argument(
        "--hold-capture-sec",
        type=_positive_float,
        default=4.0,
        help="Timed capture duration for the combined wrist/home capture",
    )
    parser.add_argument(
        "--capture-extra-wait-sec",
        type=_non_negative_float,
        default=0.0,
        help="Extra time to keep a capture open when the timed window ends before enough fresh samples arrive.",
    )
    parser.add_argument("--min-sweep-extent-m", type=_positive_float, default=0.15, help="Minimum extent required for each auxiliary sweep")
    parser.add_argument(
        "--min-axis-variance-ratio",
        type=float,
        default=0.5,
        help="Minimum PCA variance ratio required for each auxiliary sweep",
    )
    parser.add_argument(
        "--min-start-end-distance-m",
        type=_positive_float,
        default=0.08,
        help="Minimum start/end separation required for each auxiliary sweep",
    )
    parser.add_argument(
        "--max-axis-non-orthogonality-deg",
        type=_positive_float,
        default=20.0,
        help=(
            "Maximum allowed angular correction per axis when fitting an orthonormal "
            "virtual-desk frame from raw X/Y/Z sweep directions"
        ),
    )
    parser.add_argument(
        "--min-sweep-samples",
        type=int,
        default=3,
        help="Minimum valid fresh samples required for each VirtualDesk X/Y/Z sweep.",
    )
    parser.add_argument(
        "--min-hold-samples",
        type=int,
        default=3,
        help="Minimum valid samples required for the combined wrist/home capture",
    )
    parser.add_argument(
        "--home-capture-sec",
        type=_positive_float,
        default=4.0,
        help="Timed capture duration for left/right session home anchors",
    )
    parser.add_argument(
        "--home-samples",
        type=int,
        default=3,
        help="Minimum valid samples required for each side during session home capture",
    )
    parser.add_argument(
        "--skip-session-home",
        action="store_true",
        help="Skip home capture and validation, then save only tracker/station calibration.",
    )
    parser.add_argument(
        "--station-only-profile",
        action="store_true",
        help="Compatibility flag; calibration profiles are always saved without session home anchors.",
    )
    parser.add_argument(
        "--fine-tune-step-deg",
        type=_positive_float,
        default=2.0,
        help="Fine tuning step for wrist orientation offsets during validation",
    )
    parser.add_argument(
        "--default-side",
        choices=["left", "right"],
        default="right",
        help="Initial hand selected for wrist/frame fine tuning in the web and terminal controls.",
    )
    parser.add_argument(
        "--debug-frame-axis-length",
        type=_positive_float,
        default=0.08,
        help="Axis length, meters, for SAPIEN debug frame visualization",
    )
    parser.add_argument(
        "--left-tracker-translation",
        type=_parse_xyz,
        default=np.zeros(3, dtype=float),
        help="Left tracker-to-hand translation offset in tracker coordinates, meters",
    )
    parser.add_argument(
        "--right-tracker-translation",
        type=_parse_xyz,
        default=np.zeros(3, dtype=float),
        help="Right tracker-to-hand translation offset in tracker coordinates, meters",
    )
    parser.add_argument(
        "--left-hand-canonical-rpy-deg",
        type=_parse_xyz,
        default=canonical_hand_rpy_default(),
        help=(
            "Canonical left hand orientation in virtual desk XYZ Euler degrees. "
            "Defaults to [0, 0, 0], so the captured neutral pose itself defines the wrist mount correction."
        ),
    )
    parser.add_argument(
        "--right-hand-canonical-rpy-deg",
        type=_parse_xyz,
        default=canonical_hand_rpy_default(),
        help=(
            "Canonical right hand orientation in virtual desk XYZ Euler degrees. "
            "Defaults to [0, 0, 0], so the captured neutral pose itself defines the wrist mount correction."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help=(
            "Optional calibration profile output path. Defaults to "
            f"{DEFAULT_CALIBRATION_PROFILE_DIR}/<timestamp>.json"
        ),
    )
    parser.add_argument(
        "--load-profile",
        type=str,
        default="",
        help="Optional existing calibration profile to load and reuse.",
    )
    parser.add_argument(
        "--virtual-desk-post-yaw-deg",
        type=float,
        default=180.0,
        help=(
            "Post-rotation about virtual desk +Z, degrees. Defaults to 180 so human desk coordinates "
            "are aligned to the SAPIEN/RM65 scene by flipping X/Y signs."
        ),
    )
    parser.add_argument("--max-steps", type=_non_negative_int, default=0, help="Optional limit for validation loop steps, 0 means infinite")
    args = parser.parse_args()
    args.requested_tcp_mode = str(getattr(args, "tcp_mode", TOOL_TCP_FRAME))
    args.tcp_mode = TOOL_TCP_FRAME
    return args


def build_components(args: argparse.Namespace) -> CalibrationDebugComponents:
    if args.load_profile:
        calibration = TeleopCalibration.load_json(args.load_profile)
    else:
        calibration = build_annotated_dual_rm65_calibration(
            left_right_base_distance=args.left_right_base_distance
        )
    calibration.metadata.update(
        {
            "calibration_mode": "vive_rm65_virtual_desk",
            "left_tracker_name": args.left_tracker_name,
            "right_tracker_name": args.right_tracker_name,
            "aux_tracker_name": args.aux_tracker_name,
            "virtual_desk_tracker_role": args.virtual_desk_tracker_role,
            "tcp_mode": args.tcp_mode,
            "arm_mode": args.arm_mode,
            "left_arm_ip": args.left_arm_ip,
            "right_arm_ip": args.right_arm_ip,
        }
    )
    tracker_configs = [
        ViveTrackerConfig(
            name=args.left_tracker_name,
            side="left",
            role="left_wrist",
            channel_name="left_wrist_pose",
            legacy_index=0,
        ),
        ViveTrackerConfig(
            name=args.right_tracker_name,
            side="right",
            role="right_wrist",
            channel_name="right_wrist_pose",
            legacy_index=1,
        ),
    ]
    if str(args.aux_tracker_name or "").strip():
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
    source = VivePoseSource(
        tracker_configs=tracker_configs,
        host=args.host,
        port=args.port,
        clock_sync_port=int(
            getattr(args, "vive_clock_sync_port", DEFAULT_VIVE_CLOCK_SYNC_PORT)
        ),
    )
    arm_urdf_path = require_asset_path("rm65_arm.urdf")
    left_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip=args.left_arm_ip or "0.0.0.0",
            side="left",
            default_tcp_frame=args.tcp_mode,
            arm_urdf_path=arm_urdf_path,
            connect_hardware=bool(args.arm_mode == "real" and args.left_arm_ip),
            initial_qpos=None if args.arm_mode == "real" and args.left_arm_ip else tuple(DEFAULT_LEFT_ARM_QPOS.tolist()),
        )
    )
    right_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip=args.right_arm_ip or "0.0.0.0",
            side="right",
            default_tcp_frame=args.tcp_mode,
            arm_urdf_path=arm_urdf_path,
            connect_hardware=bool(args.arm_mode == "real" and args.right_arm_ip),
            initial_qpos=None if args.arm_mode == "real" and args.right_arm_ip else tuple(DEFAULT_RIGHT_ARM_QPOS.tolist()),
        )
    )

    left_retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="left",
            input_channel_name="left_wrist_pose",
            effector_name="left_arm",
            control_mode="absolute",
            translation_scale=args.translation_scale,
            input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
        )
    )
    right_retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="right",
            input_channel_name="right_wrist_pose",
            effector_name="right_arm",
            control_mode="absolute",
            translation_scale=args.translation_scale,
            input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
        )
    )
    return CalibrationDebugComponents(
        calibration=calibration,
        source=source,
        left_backend=left_backend,
        right_backend=right_backend,
        left_retargeter=left_retargeter,
        right_retargeter=right_retargeter,
        renderer=None,
        ui=None,
    )


def _state_qpos(backend: RM65ArmBackend, fallback: np.ndarray) -> np.ndarray:
    qpos = backend.get_state()["current_qpos"]
    if qpos is None:
        return fallback.copy()
    return np.asarray(qpos, dtype=float).copy()


class CalibrationApp:
    def __init__(self, args: argparse.Namespace, components: CalibrationDebugComponents):
        self.args = args
        self.components = components
        self.ui: CalibrationWebUi | None = None
        if self.args.ui_mode == "web":
            self.ui = CalibrationWebUi(
                host=self.args.ui_host,
                port=self.args.ui_port,
                open_browser=not self.args.no_open_browser,
            )
            self.key_input = WebUiKeyInput(self.ui)
            self.components = replace(self.components, ui=self.ui)
        else:
            self.key_input = TerminalKeyInput()
        self.virtual_desk_result = None
        self.coarse_tracker_offsets: dict[str, np.ndarray] = {}
        self.fine_tune_rpy_deg = {
            "left": np.zeros(3, dtype=float),
            "right": np.zeros(3, dtype=float),
        }
        self.home_ready = False
        self.empty_hand = np.empty(0, dtype=float)
        self.selected_side = str(self.args.default_side)
        self.log_gate = ChangeGate()
        self._render_enabled = False
        self._debug_frames: dict[str, np.ndarray] = {}
        self._last_validation_target_poses: dict[str, np.ndarray] = {}
        self._last_arm_refresh_time: float | None = None
        self._arm_refresh_period_sec = 0.10
        self._arm_visualization_enabled = {"left": False, "right": False}
        self.frame_viz_mode = str(self.args.frame_viz_mode)
        self.virtual_desk_post_yaw_deg = float(self.args.virtual_desk_post_yaw_deg)
        self._base_world_to_desk_pose: np.ndarray | None = None
        self._renderer_initialized = False
        self._simulate_arm_preview = False
        self._restore_loaded_profile_state()
        self._publish_ui_state(stage="Starting...")

    def _log(self, tag: str, message: str, *, now: float | None = None) -> None:
        log_event(tag, message, now=now)
        if self.ui is not None:
            self.ui.append_message(tag, message)

    def _stage(self, title: str, *, now: float | None = None) -> None:
        log_stage("STAGE", title, now=now)
        self._publish_ui_state(
            stage=title,
            active_controls=["q"],
            control_hint="",
            **self._empty_capture_ui_state(),
        )

    @staticmethod
    def _empty_capture_ui_state() -> dict[str, object]:
        return {
            "capture_stage": "",
            "capture_phase": "idle",
            "capturing": False,
            "capture_channel": "",
            "capture_samples": None,
            "capture_left_samples": None,
            "capture_right_samples": None,
            "capture_min_samples": None,
            "capture_valid": None,
            "capture_left_valid": None,
            "capture_right_valid": None,
            "capture_elapsed_sec": None,
            "capture_duration_sec": None,
            "capture_prep_remaining_sec": None,
            "capture_prep_total_sec": None,
            "capture_waiting_for_samples": False,
        }

    def _publish_ui_state(self, **updates: object) -> None:
        if self.ui is None:
            return
        state = {
            "selected_side": self.selected_side,
            "frame_viz_mode": self.frame_viz_mode,
            "virtual_desk_post_yaw_deg": self.virtual_desk_post_yaw_deg,
            "renderer_ready": bool(self.components.renderer is not None),
            "ui_mode": self.args.ui_mode,
            "host": self.args.host,
            "port": self.args.port,
            "load_profile": self.args.load_profile,
            "show_tracker_frame_viz": bool(self.args.show_tracker_frame_viz),
            "home_ready": bool(self.home_ready),
        }
        state.update(updates)
        self.ui.publish(**state)

    def _source_debug_stats(self) -> dict[str, object]:
        get_stats = getattr(self.components.source, "debug_stats", None)
        if get_stats is None:
            return {}
        stats = get_stats()
        return stats if isinstance(stats, dict) else {}

    def _publish_raw_tracker_pose_state(
        self,
        channels: Mapping[str, object],
        *,
        now: float | None = None,
    ) -> None:
        if self.ui is None:
            return
        current_time = time.time() if now is None else float(now)
        self._publish_ui_state(
            raw_tracker_poses={
                "left": _pose_xyz_rpy_summary(
                    channels.get("left_wrist_pose"),
                    now=current_time,
                ),
                "right": _pose_xyz_rpy_summary(
                    channels.get("right_wrist_pose"),
                    now=current_time,
                ),
            }
        )

    def _format_source_rates(
        self,
        *,
        start_stats: dict[str, object],
        channel_names: tuple[str, ...],
        elapsed_sec: float,
        now: float,
    ) -> str:
        current_stats = self._source_debug_stats()
        if not current_stats:
            return "source_stats=unavailable"
        elapsed = max(float(elapsed_sec), 1e-6)
        packet_count = int(current_stats.get("packet_count") or 0)
        start_packet_count = int(start_stats.get("packet_count") or 0)
        packet_hz = (packet_count - start_packet_count) / elapsed
        update_counts = current_stats.get("channel_update_counts") or {}
        start_update_counts = start_stats.get("channel_update_counts") or {}
        last_updates = current_stats.get("last_channel_update_at") or {}
        channel_parts = []
        if isinstance(update_counts, dict) and isinstance(start_update_counts, dict):
            for channel_name in channel_names:
                count = int(update_counts.get(channel_name) or 0)
                start_count = int(start_update_counts.get(channel_name) or 0)
                channel_hz = (count - start_count) / elapsed
                last_update_at = last_updates.get(channel_name) if isinstance(last_updates, dict) else None
                age = (now - float(last_update_at)) if last_update_at is not None else float("nan")
                channel_parts.append(f"{channel_name}_hz={channel_hz:.1f} age={age:.2f}s")
        last_packet_at = current_stats.get("last_packet_received_at")
        packet_age = (now - float(last_packet_at)) if last_packet_at is not None else float("nan")
        return f"source_udp_hz={packet_hz:.1f} packet_age={packet_age:.2f}s " + " ".join(channel_parts)

    def _ensure_renderer(self) -> None:
        if self.components.renderer is not None or self.args.skip_renderer:
            return
        self._refresh_arm_render_state(force=True)
        left_urdf_path = self._renderer_arm_urdf_path("left")
        right_urdf_path = self._renderer_arm_urdf_path("right")
        try:
            renderer = SapienArmHandRenderer(
                left_urdf_path=left_urdf_path,
                left_retargeting_joint_names=() if left_urdf_path is not None else None,
                right_urdf_path=right_urdf_path,
                right_retargeting_joint_names=() if right_urdf_path is not None else None,
                left_right_base_distance=self.args.left_right_base_distance,
                show_viewer=not self.args.headless,
                debug_frame_axis_length=self.args.debug_frame_axis_length,
            )
            renderer.reserve_debug_frames(
                (
                    "desk",
                    "left_arm_base",
                    "right_arm_base",
                    "left_tracker_raw",
                    "right_tracker_raw",
                    "left_hand_calibrated",
                    "right_hand_calibrated",
                    "left_tcp_actual",
                    "right_tcp_actual",
                    "left_tcp_target",
                    "right_tcp_target",
                    "aux_tracker_raw",
                )
            )
            self.components = replace(self.components, renderer=renderer)
            self._renderer_initialized = True
            self._ensure_renderer_arm_models()
            self._publish_ui_state(renderer_ready=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to create the SAPIEN renderer. Check that the current environment "
                "has a usable Vulkan/OpenGL render device, or rerun with --skip-renderer "
                "to validate the calibration flow without 3D visualization."
            ) from exc

    def _backend_for_side(self, side: str) -> RM65ArmBackend:
        return self.components.left_backend if side == "left" else self.components.right_backend

    def _refresh_arm_visualization_availability(self, *, now: float | None = None) -> None:
        for side in ("left", "right"):
            backend = self._backend_for_side(side)
            state = backend.get_state()
            if self.args.arm_mode == "real":
                if not backend.config.connect_hardware:
                    enabled = False
                    reason = "no real RM65 IP/configured hardware connection"
                elif backend.arm is None:
                    enabled = False
                    reason = "RM65 SDK arm object is unavailable"
                elif state.get("hardware_qpos") is None:
                    enabled = False
                    reason = (
                        "hardware_qpos unavailable "
                        f"result={state.get('hardware_result_code')} text={state.get('hardware_result_text')}"
                    )
                else:
                    enabled = True
                    reason = "hardware_qpos readable"
            else:
                enabled = state.get("current_qpos") is not None
                reason = "local_qpos readable" if enabled else "local_qpos unavailable"
            previous = self._arm_visualization_enabled.get(side, False)
            self._arm_visualization_enabled[side] = enabled
            if previous != enabled:
                if enabled:
                    self._log(
                        "ArmViz",
                        f"{side} arm visualization enabled from "
                        f"{'hardware_qpos' if self.args.arm_mode == 'real' else 'local_qpos'} "
                        f"qpos_deg={_qpos_deg_text(state.get('hardware_qpos') if self.args.arm_mode == 'real' else state.get('current_qpos'))}",
                        now=now,
                    )
                else:
                    self._log(
                        "ArmViz",
                        f"{side} arm visualization disabled; reason={reason}; showing VIVE frames only for this side",
                        now=now,
                    )
            elif not enabled and self.log_gate.should_emit(f"arm_viz_{side}_disabled_reason", reason, now=now):
                self._log(
                    "ArmViz",
                    f"{side} arm visualization disabled; reason={reason}; showing VIVE frames only for this side",
                    now=now,
                )

    def _ensure_renderer_arm_models(self) -> None:
        renderer = self.components.renderer
        if renderer is None:
            return
        ensure_arm_loaded = getattr(renderer, "ensure_arm_loaded", None)
        if ensure_arm_loaded is None:
            return
        for side in ("left", "right"):
            if not self._arm_visualization_enabled.get(side, False):
                continue
            robot_attr = "left_robot" if side == "left" else "right_robot"
            if getattr(renderer, robot_attr, None) is not None:
                continue
            try:
                loaded = bool(
                    ensure_arm_loaded(
                        side=side,
                        urdf_path=require_asset_path("rm65_arm.urdf"),
                        retargeting_joint_names=(),
                    )
                )
            except Exception as exc:
                if self.log_gate.should_emit(f"arm_viz_{side}_dynamic_load_failed", str(exc)):
                    self._log("ArmViz", f"{side} arm model load failed: {exc}")
                continue
            if loaded:
                self._log("ArmViz", f"{side} arm model loaded into existing SAPIEN scene")

    def _renderer_arm_urdf_path(self, side: str) -> str | None:
        if not self._arm_visualization_enabled.get(side, False):
            return None
        return require_asset_path("rm65_arm.urdf")

    def _restore_loaded_profile_state(self) -> None:
        calibration = self.components.calibration
        virtual_desk_metadata = calibration.metadata.get("virtual_desk", {})
        base_world_to_desk = virtual_desk_metadata.get("T_W_V")
        if base_world_to_desk is not None:
            self._base_world_to_desk_pose = as_pose_matrix(base_world_to_desk)
        else:
            world_to_desk = calibration.world_to_desk_pose()
            if world_to_desk is not None:
                self._base_world_to_desk_pose = world_to_desk.copy()
        if "virtual_desk_post_yaw_deg" in calibration.metadata:
            self.virtual_desk_post_yaw_deg = float(
                calibration.metadata.get("virtual_desk_post_yaw_deg", self.virtual_desk_post_yaw_deg)
            )

        tracker_metadata = calibration.metadata.get("tracker_offset_calibration", {})
        for side in ("left", "right"):
            tracker_role = f"{side}_wrist"
            tracker_info = tracker_metadata.get(tracker_role, {})
            final_offset = tracker_info.get("final_offset")
            if final_offset is not None:
                offset_matrix = as_pose_matrix(final_offset)
                self.coarse_tracker_offsets[side] = offset_matrix.copy()
                calibration.set_tracker_offset(tracker_role, offset_matrix)
            elif tracker_role in calibration.tracker_offsets:
                offset_matrix = calibration.get_tracker_offset(tracker_role)
                self.coarse_tracker_offsets[side] = offset_matrix.copy()
            self.fine_tune_rpy_deg[side] = np.zeros(3, dtype=float)

        if self._base_world_to_desk_pose is not None and abs(self.virtual_desk_post_yaw_deg) > 1e-6:
            self._apply_virtual_desk_post_rotation(update_offsets=False)

    def _has_loaded_virtual_desk(self) -> bool:
        return self.components.calibration.world_to_desk_pose() is not None

    def _has_loaded_tracker_offsets(self) -> bool:
        return "left" in self.coarse_tracker_offsets and "right" in self.coarse_tracker_offsets

    def run(self) -> None:
        if self.ui is None and not self.key_input.enabled:
            raise RuntimeError("Calibration debug requires an interactive terminal (TTY).")
        try:
            self.key_input.start()
            if self.ui is not None:
                self.ui.start()
            self._publish_ui_state(stage="Link Check")
            self._start_arm_backend("left", self.components.left_backend)
            self._start_arm_backend("right", self.components.right_backend)
            self._refresh_arm_visualization_availability()
            self.components.source.start()
            if legacy_tcp_mode_requested(self.args.requested_tcp_mode):
                self._log(
                    "TCP",
                    "legacy --tcp-mode rm_eef was requested; runtime now uses tool_tcp actions.",
                )
            self._stage("Link Check")
            self._run_link_check()
            if self.args.arm_mode == "real":
                if self.args.left_arm_ip or self.args.right_arm_ip:
                    self._log(
                        "ArmMode",
                        "real mode: rendering read-only RM65 state where IPs are provided; no motion command is sent.",
                    )
                else:
                    self._log(
                        "ArmMode",
                        "arm-mode=real but no arm IPs provided; robot arms will not be visualized. "
                        "Only VIVE trackers and coordinate frames are shown.",
                    )
            else:
                self._log("ArmMode", "virtual mode: rendering uses local unlimited-speed preview state only.")
            if self.args.load_profile and self._has_loaded_virtual_desk():
                self._log("Load", f"loaded virtual desk from {self.args.load_profile}")
            else:
                self._stage("Virtual Desk Calibration")
                self._run_virtual_desk_calibration()
            if self.components.calibration.metadata.get("virtual_desk_scene_alignment_defaulted"):
                self._log(
                    "Load",
                    "applied default desk-to-scene yaw 180deg so tracker poses align with the SAPIEN/RM65 scene.",
                )
            if self.args.load_profile and self._has_loaded_tracker_offsets():
                self._log("Load", f"loaded wrist tracker offsets from {self.args.load_profile}")
            else:
                self._stage("Wrist/Home Capture")
                self._run_dual_wrist_calibration()
            self._ensure_renderer()
            if self.args.show_tracker_frame_viz and self.components.renderer is not None:
                self._stage("Tracker Frame Viz")
                self._run_tracker_frame_visualization()
            if self.args.skip_session_home:
                self._stage("Save")
                self._save_profile()
            else:
                if not self.home_ready and self.args.load_profile and self._has_loaded_tracker_offsets():
                    self._stage("Home Capture")
                    self._capture_home()
                elif not self.home_ready:
                    self._log(
                        "HomeCapture",
                        "home anchors are not ready because robot home qpos was unavailable; "
                        "validation will show VIVE/coordinate frames only until you reconnect arms or press h after hardware is ready.",
                    )
                self._stage("Validate")
                self._run_validation_loop()
        finally:
            self.key_input.close()
            if self.ui is not None:
                self.ui.mark_shutdown("标定服务已退出，这个网页已经失效。")
                time.sleep(0.35)
                self.ui.close()
            self.components.source.close()
            self.components.left_backend.close()
            self.components.right_backend.close()

    def _start_arm_backend(self, side: str, backend: RM65ArmBackend) -> None:
        try:
            backend.start()
        except Exception as exc:
            self._arm_visualization_enabled[side] = False
            self._log(
                "ArmMode",
                f"{side} RM65 read-only connection failed: {exc}. "
                "This side will not be visualized as a robot arm.",
            )

    def _poll_channels(self) -> dict[str, object]:
        self.key_input.poll()
        channels = self.components.source.get_latest_channels()
        self._publish_raw_tracker_pose_state(channels, now=time.time())
        return channels

    def _render(self) -> None:
        if not self._render_enabled or self.components.renderer is None:
            return
        left_qpos = self._render_qpos_for_side("left")
        right_qpos = self._render_qpos_for_side("right")
        self.components.renderer.set_debug_frames(self._debug_frames)
        self.components.renderer.render_data(
            left_qpos_arm=left_qpos,
            left_qpos_hand=self.empty_hand,
            right_qpos_arm=right_qpos,
            right_qpos_hand=self.empty_hand,
            capture_image=False,
        )

    def _render_qpos_for_side(self, side: str) -> np.ndarray | None:
        if not self._arm_visualization_enabled.get(side, False):
            return None
        backend = self._backend_for_side(side)
        state = backend.get_state()
        if self.args.arm_mode == "real":
            qpos = state.get("current_qpos") if self._simulate_arm_preview else state.get("hardware_qpos")
            if qpos is None:
                qpos = state.get("hardware_qpos")
        else:
            fallback = DEFAULT_LEFT_ARM_QPOS if side == "left" else DEFAULT_RIGHT_ARM_QPOS
            qpos = state.get("current_qpos")
            if qpos is None:
                qpos = fallback
        if qpos is None:
            return None
        return np.asarray(qpos, dtype=float).copy()

    def _refresh_arm_render_state(self, *, now: float | None = None, force: bool = False) -> None:
        if not force and not self._render_enabled:
            return
        current_time = time.time() if now is None else float(now)
        if (
            not force
            and self._last_arm_refresh_time is not None
            and current_time - self._last_arm_refresh_time < self._arm_refresh_period_sec
        ):
            return
        for backend in (self.components.left_backend, self.components.right_backend):
            if backend.config.connect_hardware:
                backend.refresh_state_from_hardware()
        self._last_arm_refresh_time = current_time
        self._refresh_arm_visualization_availability(now=current_time)
        self._ensure_renderer_arm_models()

    def _set_debug_frames(self, frames: dict[str, np.ndarray] | None) -> None:
        self._debug_frames = {} if frames is None else {key: value.copy() for key, value in frames.items()}

    def _cycle_frame_viz_mode(self, *, now: float | None = None) -> None:
        order = ("calibrated", "raw", "both")
        current_index = order.index(self.frame_viz_mode)
        self.frame_viz_mode = order[(current_index + 1) % len(order)]
        self._log("FrameViz", f"frame_viz_mode={self.frame_viz_mode}", now=now)
        self._publish_ui_state()

    def _virtual_desk_post_rotation_pose(self) -> np.ndarray:
        yaw_rad = np.deg2rad(self.virtual_desk_post_yaw_deg)
        return pose_from_rotation_translation(
            euler_xyz_deg_to_matrix([0.0, 0.0, np.rad2deg(yaw_rad)]),
            [0.0, 0.0, 0.0],
        )

    def _apply_virtual_desk_post_rotation(self, *, update_offsets: bool) -> None:
        if self._base_world_to_desk_pose is None:
            return
        post_rotation = self._virtual_desk_post_rotation_pose()
        self.components.calibration.set_world_to_desk(self._base_world_to_desk_pose @ post_rotation)
        self.components.calibration.metadata["virtual_desk_post_yaw_deg"] = float(
            self.virtual_desk_post_yaw_deg
        )
        self.components.calibration.metadata["virtual_desk_sweep_convention_version"] = (
            _VIRTUAL_DESK_SWEEP_CONVENTION_VERSION
        )
        if not update_offsets:
            return
        rotation_only = pose_from_rotation_translation(post_rotation[:3, :3], [0.0, 0.0, 0.0])
        tracker_metadata = self.components.calibration.metadata.setdefault(
            "tracker_offset_calibration", {}
        )
        for side in ("left", "right"):
            tracker_role = f"{side}_wrist"
            current_offset = self.components.calibration.get_tracker_offset(tracker_role)
            updated_offset = current_offset @ rotation_only
            self.coarse_tracker_offsets[side] = updated_offset.copy()
            self.fine_tune_rpy_deg[side] = np.zeros(3, dtype=float)
            self.components.calibration.set_tracker_offset(tracker_role, updated_offset)
            tracker_info = tracker_metadata.setdefault(tracker_role, {})
            tracker_info["coarse_offset"] = updated_offset.tolist()
            tracker_info["canonical_hand_frame_version"] = TRACKER_OFFSET_CALIBRATION_VERSION
            tracker_info["fine_tune_rpy_deg"] = [0.0, 0.0, 0.0]
            tracker_info["final_offset"] = updated_offset.tolist()
        self._clear_home()

    def _toggle_virtual_desk_xy_flip(self, *, now: float | None = None) -> None:
        self.virtual_desk_post_yaw_deg = 0.0 if abs(self.virtual_desk_post_yaw_deg) > 90.0 else 180.0
        self._apply_virtual_desk_post_rotation(update_offsets=True)
        self._log(
            "VirtualDesk",
            f"virtual_desk_post_yaw_deg={self.virtual_desk_post_yaw_deg:.0f}; "
            "XY signs updated and home anchors cleared",
            now=now,
        )
        self._publish_ui_state()

    def _desk_debug_frames(self) -> dict[str, np.ndarray]:
        frames = {"desk": identity_pose()}
        for side in ("left", "right"):
            arm = self.components.calibration.get_arm_for_side(side)
            if arm.T_D_B is not None:
                frames[f"{side}_arm_base"] = arm.T_D_B.copy()
        return frames

    def _current_actual_tcp_pose_desk(self, side: str) -> np.ndarray | None:
        if not self._arm_visualization_enabled.get(side, False):
            return None
        backend = self.components.left_backend if side == "left" else self.components.right_backend
        arm = self.components.calibration.get_arm_for_side(side)
        if arm.T_D_B is None:
            return None
        qpos = self._render_qpos_for_side(side)
        if qpos is None:
            return None
        try:
            pose_base = backend.compute_tcp_pose(self.args.tcp_mode, qpos=qpos)
        except Exception:
            return None
        return arm.T_D_B @ as_pose_matrix(pose_base)

    def _build_debug_frames(
        self,
        channels: dict[str, object],
        *,
        include_actual_tcp: bool = False,
        include_target_tcp: bool = False,
    ) -> dict[str, np.ndarray]:
        frames = self._desk_debug_frames()
        show_raw = self.frame_viz_mode in {"raw", "both"}
        show_calibrated = self.frame_viz_mode in {"calibrated", "both"}
        for side in ("left", "right"):
            channel = channels.get(f"{side}_wrist_pose")
            if channel is not None and channel.valid:
                if show_raw:
                    frames[f"{side}_tracker_raw"] = self._channel_pose_virtual_raw(channel)
                if show_calibrated:
                    frames[f"{side}_hand_calibrated"] = self._channel_pose_virtual_calibrated(channel)
            if include_actual_tcp:
                actual_tcp = self._current_actual_tcp_pose_desk(side)
                if actual_tcp is not None:
                    frames[f"{side}_tcp_actual"] = actual_tcp
            if include_target_tcp and side in self._last_validation_target_poses:
                frames[f"{side}_tcp_target"] = self._last_validation_target_poses[side].copy()
        aux_channel = channels.get("auxiliary_tracker_pose")
        if (
            aux_channel is not None
            and aux_channel.valid
            and self.components.calibration.world_to_desk_pose() is not None
        ):
            frames["aux_tracker_raw"] = self._channel_pose_virtual_raw(aux_channel)
        return frames

    def _world_pose_to_virtual(self, pose_world: np.ndarray) -> np.ndarray:
        T_W_V = self.components.calibration.world_to_desk_pose()
        if T_W_V is None:
            raise RuntimeError("Virtual desk is not calibrated yet.")
        return invert_pose(T_W_V) @ as_pose_matrix(pose_world)

    def _channel_pose_virtual_raw(self, channel) -> np.ndarray:
        return self._world_pose_to_virtual(as_pose_matrix(channel.value))

    def _channel_pose_virtual_calibrated(self, channel) -> np.ndarray:
        tracker_role = None
        if channel.metadata:
            tracker_role = channel.metadata.get("tracker_role")
        return self.components.calibration.world_to_desk(
            as_pose_matrix(channel.value),
            tracker_role=str(tracker_role) if tracker_role else None,
        )

    def _clear_home(self) -> None:
        self.home_ready = False
        self.components.left_retargeter.clear_home_anchors()
        self.components.right_retargeter.clear_home_anchors()
        self._last_validation_target_poses = {}
        self._publish_ui_state(home_ready=False)

    def _set_selected_side(self, side: str, stage_name: str, *, now: float | None = None) -> None:
        if side not in {"left", "right"}:
            return
        if self.selected_side == side:
            return
        self.selected_side = side
        self._log(stage_name, f"selected_side={side}", now=now)
        self._publish_ui_state()

    def _toggle_selected_side(self, stage_name: str, *, now: float | None = None) -> None:
        self._set_selected_side(
            "left" if self.selected_side == "right" else "right",
            stage_name,
            now=now,
        )

    @staticmethod
    def _channel_for_tracker_role(role: str) -> str:
        return {
            "auxiliary": "auxiliary_tracker_pose",
            "left_wrist": "left_wrist_pose",
            "right_wrist": "right_wrist_pose",
        }[str(role)]

    def _resolve_virtual_desk_channel_name(self, channels: Mapping[str, object] | None = None) -> str:
        requested = str(getattr(self.args, "virtual_desk_tracker_role", "auto") or "auto")
        if requested != "auto":
            return self._channel_for_tracker_role(requested)
        if channels:
            for candidate in ("auxiliary_tracker_pose", "left_wrist_pose", "right_wrist_pose"):
                channel = channels.get(candidate)
                if channel is not None and getattr(channel, "valid", False):
                    return candidate
        if str(getattr(self.args, "aux_tracker_name", "") or "").strip():
            return "auxiliary_tracker_pose"
        return "left_wrist_pose"

    @staticmethod
    def _tracker_role_for_channel(channel_name: str) -> str:
        return {
            "auxiliary_tracker_pose": "auxiliary",
            "left_wrist_pose": "left_wrist",
            "right_wrist_pose": "right_wrist",
        }.get(str(channel_name), str(channel_name))

    def _run_link_check(self) -> None:
        self._publish_ui_state(
            stage="Link Check",
            active_controls=["n", "q"],
            control_hint="等待左右 wrist tracker 在线；在线后按 Next/n 进入三轴标定。",
            **self._empty_capture_ui_state(),
        )
        self._log("LinkCheck", "READY: PRESS N TO CONTINUE AFTER left/right wrist tracker channels are present.")
        self._log(
            "LinkCheck",
            "q=quit, n=continue when left/right tracker channels are present. "
            "Virtual desk sweeps can use auxiliary, left_wrist, or right_wrist.",
        )
        self._log(
            "LinkCheck",
            "logical arm mapping: left_wrist_pose -> left_arm(slot at desk -Y / screen-left), "
            "right_wrist_pose -> right_arm(slot at desk +Y / screen-right)",
        )
        try:
            left_arm = self.components.calibration.get_arm_for_side("left")
            right_arm = self.components.calibration.get_arm_for_side("right")
            if left_arm.T_D_B is not None and right_arm.T_D_B is not None:
                self._log(
                    "LinkCheck",
                    "visual slots: "
                    f"left_arm desk_y={left_arm.T_D_B[1, 3]:+.4f}, "
                    f"right_arm desk_y={right_arm.T_D_B[1, 3]:+.4f}",
                )
        except Exception:
            pass
        while True:
            loop_start = time.time()
            channels = self._poll_channels()
            events = self.key_input.consume_events()
            if self.key_input.should_quit:
                raise KeyboardInterrupt
            ready = {
                name: bool(channels.get(name) is not None and channels[name].valid)
                for name in ("left_wrist_pose", "right_wrist_pose")
            }
            desk_channel = self._resolve_virtual_desk_channel_name(channels)
            desk_ready = bool(channels.get(desk_channel) is not None and channels[desk_channel].valid)
            status_snapshot = (
                tuple(sorted(ready.items())),
                desk_channel,
                desk_ready,
                tuple(sorted(channels)),
            )
            if self.log_gate.should_emit("link_check_status", status_snapshot, now=loop_start):
                self._log(
                    "LinkCheck",
                    f"ready={ready} virtual_desk_channel={desk_channel} "
                    f"virtual_desk_ready={desk_ready} channels={sorted(channels)} press 'n' to continue",
                    now=loop_start,
                )
            self._publish_ui_state(
                stage="Link Check",
                active_controls=["n", "q"],
                control_hint=(
                    f"left={ready['left_wrist_pose']} right={ready['right_wrist_pose']} "
                    f"desk_channel={desk_channel} desk_ready={desk_ready}"
                ),
            )
            if "n" in events and all(ready.values()):
                self._log("LinkCheck", "required wrist trackers are ready; continuing", now=loop_start)
                return
            if "n" in events and not all(ready.values()):
                missing = [name for name, is_ready in ready.items() if not is_ready]
                self._log("LinkCheck", f"cannot continue; missing={missing}", now=loop_start)
            self._render()
            time.sleep(max(0.0, (1.0 / self.args.hz) - (time.time() - loop_start)))

    def _capture_single_channel(
        self,
        *,
        stage_name: str,
        instructions: str,
        timed_flow_hint: str,
        direction_hint: str,
        channel_name: str,
        min_samples: int,
        capture_duration_sec: float,
        transform,
        validator=None,
    ) -> list[np.ndarray]:
        self._log(stage_name, instructions)
        self._log(stage_name, timed_flow_hint)
        self._log(stage_name, "READY: timed capture auto-starts; press r to reset, then c to recapture.")
        self._log(
            stage_name,
            f"c=start {self.args.capture_prep_sec:.1f}s prep, then timed capture ({capture_duration_sec:.1f}s); "
            "r=abort/reset, q=quit",
        )
        poses: list[np.ndarray] = []
        preparing = False
        capturing = False
        auto_start_pending = True
        capture_start_time = 0.0
        capture_deadline = 0.0
        capture_extension_deadline = 0.0
        source_stats_at_start: dict[str, object] = {}
        last_sample_key: tuple[object, object, object] | None = None
        stage_gate = ChangeGate()
        min_samples_reported = False
        waiting_for_samples_reported = False
        while True:
            loop_start = time.time()
            channels = self._poll_channels()
            events = self.key_input.consume_events()
            if self.key_input.should_quit:
                raise KeyboardInterrupt
            if "r" in events:
                poses = []
                preparing = False
                capturing = False
                auto_start_pending = False
                min_samples_reported = False
                waiting_for_samples_reported = False
                self._log(stage_name, "reset", now=loop_start)
            start_requested = bool(auto_start_pending or "c" in events)
            if start_requested:
                requested_by_auto = bool(auto_start_pending)
                auto_start_pending = False
                if not capturing and not preparing:
                    poses = []
                    preparing = True
                    capture_start_time = loop_start + self.args.capture_prep_sec
                    capture_deadline = capture_start_time + capture_duration_sec
                    capture_extension_deadline = capture_deadline + self.args.capture_extra_wait_sec
                    source_stats_at_start = {}
                    last_sample_key = None
                    min_samples_reported = False
                    waiting_for_samples_reported = False
                    self._log(
                        stage_name,
                        f"{'AUTO ' if requested_by_auto else ''}CAPTURE PREP prep={self.args.capture_prep_sec:.1f}s "
                        f"start_at={_format_wall_time(capture_start_time)} "
                        f"auto_end_at={_format_wall_time(capture_deadline)}",
                        now=loop_start,
                    )
                else:
                    self._log(
                        stage_name,
                        "CAPTURE ALREADY ACTIVE; wait for auto-end or press r to abort",
                        now=loop_start,
                    )

            channel = channels.get(channel_name)
            channel_valid = bool(channel is not None and channel.valid)
            if preparing and loop_start >= capture_start_time:
                preparing = False
                capturing = True
                source_stats_at_start = self._source_debug_stats()
                if channel_valid:
                    last_sample_key = _channel_sample_key(channel)
                self._log(
                    stage_name,
                    f"CAPTURE STARTED window={capture_duration_sec:.1f}s "
                    f"auto_end_at={_format_wall_time(capture_deadline)}",
                    now=loop_start,
                )
            elapsed = max(0.0, loop_start - capture_start_time) if capturing else 0.0
            waiting_for_samples = bool(capturing and loop_start >= capture_deadline and len(poses) < min_samples)
            capture_active = preparing or capturing
            capture_phase = (
                "preparing"
                if preparing
                else "waiting"
                if waiting_for_samples
                else "capturing"
                if capturing
                else "ready"
            )
            self._publish_ui_state(
                stage=stage_name,
                active_controls=["r", "q"] if capture_active else ["c", "r", "q"],
                control_hint=direction_hint,
                capture_stage=stage_name,
                capture_phase=capture_phase,
                capturing=capture_active,
                capture_channel=channel_name,
                capture_samples=len(poses),
                capture_left_samples=None,
                capture_right_samples=None,
                capture_min_samples=min_samples,
                capture_valid=channel_valid,
                capture_left_valid=None,
                capture_right_valid=None,
                capture_elapsed_sec=elapsed if capturing else None,
                capture_duration_sec=(
                    capture_duration_sec + self.args.capture_extra_wait_sec
                    if waiting_for_samples
                    else capture_duration_sec
                ),
                capture_prep_remaining_sec=max(0.0, capture_start_time - loop_start) if preparing else None,
                capture_prep_total_sec=self.args.capture_prep_sec,
                capture_waiting_for_samples=waiting_for_samples,
            )
            if capturing and stage_gate.should_emit(f"{stage_name}_channel_valid", channel_valid, now=loop_start):
                self._log(stage_name, f"channel_valid={channel_valid}", now=loop_start)
            if capturing and channel_valid:
                sample_key = _channel_sample_key(channel)
                if sample_key != last_sample_key:
                    last_sample_key = sample_key
                    poses.append(transform(channel))
                if len(poses) >= min_samples and not min_samples_reported:
                    min_samples_reported = True
                    self._log(
                        stage_name,
                        f"minimum sample target reached samples={len(poses)}/{min_samples}",
                        now=loop_start,
                    )
            if capturing and stage_gate.should_emit(
                f"{stage_name}_live",
                channel_valid,
                now=loop_start,
                heartbeat_sec=DEFAULT_LOG_HEARTBEAT_SEC,
            ):
                elapsed = max(0.0, loop_start - capture_start_time)
                self._log(
                    stage_name,
                    f"LIVE time={elapsed:.1f}/{capture_duration_sec:.1f}s samples={len(poses)} "
                    f"valid={channel_valid} "
                    f"{self._format_source_rates(start_stats=source_stats_at_start, channel_names=(channel_name,), elapsed_sec=elapsed, now=loop_start)}",
                    now=loop_start,
                )
            if capturing:
                if loop_start >= capture_deadline:
                    capturing = False
                    effective_hz = len(poses) / max(elapsed, 1e-6)
                    if len(poses) < min_samples:
                        if loop_start < capture_extension_deadline:
                            capturing = True
                            if not waiting_for_samples_reported:
                                waiting_for_samples_reported = True
                                self._log(
                                    stage_name,
                                    f"AUTO-END WAITING samples={len(poses)} need>={min_samples} "
                                    f"effective_hz={effective_hz:.1f} "
                                    f"extra_until={_format_wall_time(capture_extension_deadline)}",
                                    now=loop_start,
                                )
                            self._render()
                            time.sleep(max(0.0, (1.0 / self.args.hz) - (time.time() - loop_start)))
                            continue
                        self._log(
                            stage_name,
                            f"AUTO-END INSUFFICIENT samples={len(poses)} need>={min_samples} "
                            f"effective_hz={effective_hz:.1f} "
                            f"{self._format_source_rates(start_stats=source_stats_at_start, channel_names=(channel_name,), elapsed_sec=elapsed, now=loop_start)}",
                            now=loop_start,
                        )
                        poses = []
                        min_samples_reported = False
                        waiting_for_samples_reported = False
                        last_sample_key = None
                    elif validator is None:
                        self._log(
                            stage_name,
                            f"AUTO-END ACCEPTED samples={len(poses)} effective_hz={effective_hz:.1f} "
                            f"{self._format_source_rates(start_stats=source_stats_at_start, channel_names=(channel_name,), elapsed_sec=elapsed, now=loop_start)}",
                            now=loop_start,
                        )
                        return poses
                    else:
                        error = validator(poses)
                        if error is None:
                            self._log(
                                stage_name,
                                f"AUTO-END ACCEPTED samples={len(poses)} effective_hz={effective_hz:.1f} "
                                f"{self._format_source_rates(start_stats=source_stats_at_start, channel_names=(channel_name,), elapsed_sec=elapsed, now=loop_start)}",
                                now=loop_start,
                            )
                            return poses
                        self._log(
                            stage_name,
                            f"AUTO-END REJECTED: {error} effective_hz={effective_hz:.1f} "
                            f"{self._format_source_rates(start_stats=source_stats_at_start, channel_names=(channel_name,), elapsed_sec=elapsed, now=loop_start)}",
                            now=loop_start,
                        )
                        poses = []
                        min_samples_reported = False
                        waiting_for_samples_reported = False
                        last_sample_key = None
            self._render()
            time.sleep(max(0.0, (1.0 / self.args.hz) - (time.time() - loop_start)))

    def _capture_dual_home_holds(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        self._log("HomeCapture", "move robot to session home and place both hands in the desired start pose.")
        self._log(
            "HomeCapture",
            "When ready, press c once, then keep both hands steady until the timed capture ends automatically.",
        )
        self._log("HomeCapture", "READY: PRESS C TO START TIMED CAPTURE.")
        self._log(
            "HomeCapture",
            f"c=start {self.args.capture_prep_sec:.1f}s prep, then timed capture ({self.args.home_capture_sec:.1f}s); "
            "r=abort/reset, q=quit",
        )
        left_poses: list[np.ndarray] = []
        right_poses: list[np.ndarray] = []
        preparing = False
        capturing = False
        capture_start_time = 0.0
        capture_deadline = 0.0
        capture_extension_deadline = 0.0
        source_stats_at_start: dict[str, object] = {}
        left_last_sample_key: tuple[object, object, object] | None = None
        right_last_sample_key: tuple[object, object, object] | None = None
        stage_gate = ChangeGate()
        left_min_reported = False
        right_min_reported = False
        waiting_for_samples_reported = False
        while True:
            loop_start = time.time()
            channels = self._poll_channels()
            events = self.key_input.consume_events()
            if self.key_input.should_quit:
                raise KeyboardInterrupt
            if "r" in events:
                left_poses = []
                right_poses = []
                preparing = False
                capturing = False
                left_min_reported = False
                right_min_reported = False
                waiting_for_samples_reported = False
                self._log("HomeCapture", "reset", now=loop_start)
            if "c" in events:
                if not capturing and not preparing:
                    left_poses = []
                    right_poses = []
                    preparing = True
                    capture_start_time = loop_start + self.args.capture_prep_sec
                    capture_deadline = capture_start_time + self.args.home_capture_sec
                    capture_extension_deadline = capture_deadline + self.args.capture_extra_wait_sec
                    source_stats_at_start = {}
                    left_last_sample_key = None
                    right_last_sample_key = None
                    left_min_reported = False
                    right_min_reported = False
                    waiting_for_samples_reported = False
                    self._log(
                        "HomeCapture",
                        f"CAPTURE PREP prep={self.args.capture_prep_sec:.1f}s "
                        f"start_at={_format_wall_time(capture_start_time)} "
                        f"auto_end_at={_format_wall_time(capture_deadline)}",
                        now=loop_start,
                    )
                else:
                    self._log(
                        "HomeCapture",
                        "CAPTURE ALREADY ACTIVE; wait for auto-end or press r to abort",
                        now=loop_start,
                    )

            left_channel = channels.get("left_wrist_pose")
            right_channel = channels.get("right_wrist_pose")
            left_valid = bool(left_channel is not None and left_channel.valid)
            right_valid = bool(right_channel is not None and right_channel.valid)
            validity_snapshot = (left_valid, right_valid)
            if preparing and loop_start >= capture_start_time:
                preparing = False
                capturing = True
                source_stats_at_start = self._source_debug_stats()
                if left_valid:
                    left_last_sample_key = _channel_sample_key(left_channel)
                if right_valid:
                    right_last_sample_key = _channel_sample_key(right_channel)
                self._log(
                    "HomeCapture",
                    f"CAPTURE STARTED window={self.args.home_capture_sec:.1f}s "
                    f"auto_end_at={_format_wall_time(capture_deadline)}",
                    now=loop_start,
                )
            elapsed = max(0.0, loop_start - capture_start_time) if capturing else 0.0
            waiting_for_samples = bool(
                capturing
                and loop_start >= capture_deadline
                and (len(left_poses) < self.args.home_samples or len(right_poses) < self.args.home_samples)
            )
            capture_active = preparing or capturing
            capture_phase = (
                "preparing"
                if preparing
                else "waiting"
                if waiting_for_samples
                else "capturing"
                if capturing
                else "ready"
            )
            self._publish_ui_state(
                stage="Home Capture",
                active_controls=["r", "q"] if capture_active else ["c", "r", "q"],
                control_hint=(
                    f"把左右手放到本次 teleop 起始姿态；开始采集后先准备 {self.args.capture_prep_sec:.1f}s，"
                    f"再保持 {self.args.home_capture_sec:.1f}s。"
                ),
                capture_stage="HomeCapture",
                capture_phase=capture_phase,
                capturing=capture_active,
                capture_channel="left_wrist_pose,right_wrist_pose",
                capture_samples=None,
                capture_left_samples=len(left_poses),
                capture_right_samples=len(right_poses),
                capture_min_samples=self.args.home_samples,
                capture_valid=None,
                capture_left_valid=left_valid,
                capture_right_valid=right_valid,
                capture_elapsed_sec=elapsed if capturing else None,
                capture_duration_sec=(
                    self.args.home_capture_sec + self.args.capture_extra_wait_sec
                    if waiting_for_samples
                    else self.args.home_capture_sec
                ),
                capture_prep_remaining_sec=max(0.0, capture_start_time - loop_start) if preparing else None,
                capture_prep_total_sec=self.args.capture_prep_sec,
                capture_waiting_for_samples=waiting_for_samples,
            )
            if capturing and stage_gate.should_emit("home_capture_validity", validity_snapshot, now=loop_start):
                self._log(
                    "HomeCapture",
                    f"channel_valid={{'left': {left_valid}, 'right': {right_valid}}}",
                    now=loop_start,
                )
            if capturing and left_valid:
                left_sample_key = _channel_sample_key(left_channel)
                if left_sample_key != left_last_sample_key:
                    left_last_sample_key = left_sample_key
                    left_poses.append(self._channel_pose_virtual_calibrated(left_channel))
                if len(left_poses) >= self.args.home_samples and not left_min_reported:
                    left_min_reported = True
                    self._log(
                        "HomeCapture",
                        f"left minimum sample target reached {len(left_poses)}/{self.args.home_samples}",
                        now=loop_start,
                    )
            if capturing and right_valid:
                right_sample_key = _channel_sample_key(right_channel)
                if right_sample_key != right_last_sample_key:
                    right_last_sample_key = right_sample_key
                    right_poses.append(self._channel_pose_virtual_calibrated(right_channel))
                if len(right_poses) >= self.args.home_samples and not right_min_reported:
                    right_min_reported = True
                    self._log(
                        "HomeCapture",
                        f"right minimum sample target reached {len(right_poses)}/{self.args.home_samples}",
                        now=loop_start,
                    )
            if capturing and stage_gate.should_emit(
                "home_capture_live",
                validity_snapshot,
                now=loop_start,
                heartbeat_sec=DEFAULT_LOG_HEARTBEAT_SEC,
            ):
                elapsed = max(0.0, loop_start - capture_start_time)
                self._log(
                    "HomeCapture",
                    f"LIVE time={elapsed:.1f}/{self.args.home_capture_sec:.1f}s "
                    f"left={len(left_poses)} right={len(right_poses)} "
                    f"valid={{'left': {left_valid}, 'right': {right_valid}}} "
                    f"{self._format_source_rates(start_stats=source_stats_at_start, channel_names=('left_wrist_pose', 'right_wrist_pose'), elapsed_sec=elapsed, now=loop_start)}",
                    now=loop_start,
                )
            if capturing:
                if loop_start >= capture_deadline:
                    capturing = False
                    left_hz = len(left_poses) / max(elapsed, 1e-6)
                    right_hz = len(right_poses) / max(elapsed, 1e-6)
                    if len(left_poses) >= self.args.home_samples and len(right_poses) >= self.args.home_samples:
                        self._log(
                            "HomeCapture",
                            f"AUTO-END ACCEPTED left={len(left_poses)} right={len(right_poses)} "
                            f"left_hz={left_hz:.1f} right_hz={right_hz:.1f} "
                            f"{self._format_source_rates(start_stats=source_stats_at_start, channel_names=('left_wrist_pose', 'right_wrist_pose'), elapsed_sec=elapsed, now=loop_start)}",
                            now=loop_start,
                        )
                        return left_poses, right_poses
                    if loop_start < capture_extension_deadline:
                        capturing = True
                        if not waiting_for_samples_reported:
                            waiting_for_samples_reported = True
                            self._log(
                                "HomeCapture",
                                f"AUTO-END WAITING left={len(left_poses)} right={len(right_poses)} "
                                f"need>={self.args.home_samples} left_hz={left_hz:.1f} right_hz={right_hz:.1f} "
                                f"extra_until={_format_wall_time(capture_extension_deadline)}",
                                now=loop_start,
                            )
                        self._render()
                        time.sleep(max(0.0, (1.0 / self.args.hz) - (time.time() - loop_start)))
                        continue
                    self._log(
                        "HomeCapture",
                        f"AUTO-END INSUFFICIENT left={len(left_poses)} right={len(right_poses)} "
                        f"need>={self.args.home_samples} left_hz={left_hz:.1f} right_hz={right_hz:.1f} "
                        f"{self._format_source_rates(start_stats=source_stats_at_start, channel_names=('left_wrist_pose', 'right_wrist_pose'), elapsed_sec=elapsed, now=loop_start)}",
                        now=loop_start,
                    )
                    left_poses = []
                    right_poses = []
                    left_min_reported = False
                    right_min_reported = False
                    waiting_for_samples_reported = False
                    left_last_sample_key = None
                    right_last_sample_key = None
            self._render()
            time.sleep(max(0.0, (1.0 / self.args.hz) - (time.time() - loop_start)))

    def _run_virtual_desk_calibration(self) -> None:
        def _sweep_validator(poses: list[np.ndarray]) -> str | None:
            fit = fit_signed_axis_from_poses(poses)
            quality = fit.quality
            if quality.extent_m < self.args.min_sweep_extent_m:
                return f"extent {quality.extent_m:.3f}m < {self.args.min_sweep_extent_m:.3f}m"
            if quality.variance_ratio < self.args.min_axis_variance_ratio:
                return f"variance_ratio {quality.variance_ratio:.3f} < {self.args.min_axis_variance_ratio:.3f}"
            if quality.start_end_distance_m < self.args.min_start_end_distance_m:
                return (
                    f"start_end_distance {quality.start_end_distance_m:.3f}m "
                    f"< {self.args.min_start_end_distance_m:.3f}m"
                )
            return None

        channels = self._poll_channels()
        desk_channel_name = self._resolve_virtual_desk_channel_name(channels)
        desk_tracker_role = self._tracker_role_for_channel(desk_channel_name)
        self._log(
            "VirtualDesk",
            f"using {desk_channel_name} ({desk_tracker_role}) for virtual-desk sweeps",
        )

        x_sweep = self._capture_single_channel(
            stage_name="VirtualDeskX",
            instructions=(
                f"使用 {desk_tracker_role} 做 X 轴扫动，让拟合后的 +X 指向身体前方："
                "从靠近身体后侧的位置开始，向身体前方移动并结束。"
            ),
            timed_flow_hint=(
                "按一次开始采集/c 后，按“后 -> 前”扫动。"
                "采集自动结束前，请停在身体前方。"
            ),
            direction_hint="后 -> 前",
            channel_name=desk_channel_name,
            min_samples=self.args.min_sweep_samples,
            capture_duration_sec=self.args.sweep_capture_sec,
            transform=lambda channel: as_pose_matrix(channel.value),
            validator=_sweep_validator,
        )
        y_sweep = self._capture_single_channel(
            stage_name="VirtualDeskY",
            instructions=(
                "做 Y 轴扫动，让拟合后的 +Y 指向你的左侧："
                "从身体右侧开始，向身体左侧移动并结束。"
            ),
            timed_flow_hint=(
                "按一次开始采集/c 后，按“右 -> 左”扫动。"
                "采集自动结束前，请停在身体左侧。"
            ),
            direction_hint="右 -> 左",
            channel_name=desk_channel_name,
            min_samples=self.args.min_sweep_samples,
            capture_duration_sec=self.args.sweep_capture_sec,
            transform=lambda channel: as_pose_matrix(channel.value),
            validator=_sweep_validator,
        )
        z_sweep = self._capture_single_channel(
            stage_name="VirtualDeskZ",
            instructions=(
                "做 Z 轴扫动，让拟合后的 +Z 指向上方：从低处开始，向上移动并结束。"
            ),
            timed_flow_hint=(
                "按一次开始采集/c 后，按“低 -> 高”扫动。"
                "采集自动结束前不要再向下返回。"
            ),
            direction_hint="低 -> 高",
            channel_name=desk_channel_name,
            min_samples=self.args.min_sweep_samples,
            capture_duration_sec=self.args.sweep_capture_sec,
            transform=lambda channel: as_pose_matrix(channel.value),
            validator=_sweep_validator,
        )
        result = fit_virtual_desk_frame(
            x_sweep_poses=x_sweep,
            y_sweep_poses=y_sweep,
            z_sweep_poses=z_sweep,
        )
        max_raw_non_orthogonality = max(
            result.xy_non_orthogonality_deg,
            result.xz_non_orthogonality_deg,
            result.yz_non_orthogonality_deg,
        )
        max_axis_correction = max(
            result.x_axis_correction_deg,
            result.y_axis_correction_deg,
            result.z_axis_correction_deg,
        )
        self._log(
            "VirtualDesk",
            "raw sweep non-orthogonality "
            f"xy={result.xy_non_orthogonality_deg:.2f}deg "
            f"xz={result.xz_non_orthogonality_deg:.2f}deg "
            f"yz={result.yz_non_orthogonality_deg:.2f}deg; "
            "axis corrections "
            f"x={result.x_axis_correction_deg:.2f}deg "
            f"y={result.y_axis_correction_deg:.2f}deg "
            f"z={result.z_axis_correction_deg:.2f}deg; "
            f"z_alignment_before_flip={result.z_axis_alignment_before_flip_deg:.2f}deg "
            f"handedness_flip={result.handedness_flip_applied}",
        )
        if max_axis_correction > self.args.max_axis_non_orthogonality_deg:
            worst_axis_name, worst_axis_correction = max(
                (
                    ("x", result.x_axis_correction_deg),
                    ("y", result.y_axis_correction_deg),
                    ("z", result.z_axis_correction_deg),
                ),
                key=lambda item: item[1],
            )
            raise RuntimeError(
                "Virtual desk fit failed axis-correction check: "
                f"worst_axis={worst_axis_name} correction={worst_axis_correction:.2f}deg "
                f"> {self.args.max_axis_non_orthogonality_deg:.2f}deg. "
                "Raw sweep non-orthogonality was "
                f"xy={result.xy_non_orthogonality_deg:.2f}deg "
                f"xz={result.xz_non_orthogonality_deg:.2f}deg "
                f"yz={result.yz_non_orthogonality_deg:.2f}deg. "
                f"z_alignment_before_flip={result.z_axis_alignment_before_flip_deg:.2f}deg "
                f"handedness_flip={result.handedness_flip_applied}. "
                "If handedness_flip=True or z_alignment_before_flip is near 180deg, "
                "one of X/Y/Z sweep directions is likely reversed."
            )
        if max_raw_non_orthogonality > self.args.max_axis_non_orthogonality_deg:
            self._log(
                "VirtualDesk",
                f"accepted after orthonormal fit despite raw sweep inconsistency "
                f"(max_raw_non_orthogonality={max_raw_non_orthogonality:.2f}deg)",
            )
        self.virtual_desk_result = result
        self._base_world_to_desk_pose = result.T_W_V.copy()
        self.components.calibration.set_world_to_desk(result.T_W_V)
        if abs(self.virtual_desk_post_yaw_deg) > 1e-6:
            self._apply_virtual_desk_post_rotation(update_offsets=False)
        self.components.calibration.metadata["virtual_desk"] = result.to_metadata()
        self.components.calibration.metadata["virtual_desk_tracker_role"] = desk_tracker_role
        self.components.calibration.metadata["virtual_desk_channel_name"] = desk_channel_name
        self.components.calibration.metadata["virtual_desk_sweep_convention_version"] = (
            _VIRTUAL_DESK_SWEEP_CONVENTION_VERSION
        )
        self._log(
            "VirtualDesk",
            f"calibrated x={np.array2string(result.x_axis_world, precision=3)} "
            f"y={np.array2string(result.y_axis_world, precision=3)} "
            f"z={np.array2string(result.z_axis_world, precision=3)} "
            f"origin_source={result.origin_source}",
        )
        if abs(self.virtual_desk_post_yaw_deg) > 1e-6:
            self._log(
                "VirtualDesk",
                f"applied post yaw correction {self.virtual_desk_post_yaw_deg:.0f}deg",
            )

    def _capture_dual_wrist_coarse_holds(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        wrist_prep_sec = float(self.args.wrist_capture_prep_sec)
        self._log(
            "WristHome",
            "Place both hands in the canonical session start pose: forearms forward, hand backs up, "
            "and fingers roughly aligned with virtual-desk +X. This single capture defines both wrist offsets "
            "and the session home anchor when real arm qpos is readable.",
        )
        self._log(
            "WristHome",
            "Default canonical wrist frame uses XYZ Euler [0, 0, 0] deg; the captured neutral pose directly defines the tracker-to-hand correction.",
        )
        self._log(
            "WristHome",
            "Press c once, then keep both wrist trackers and both hand poses steady until the timed capture ends automatically.",
        )
        self._log("WristHome", "READY: PRESS C TO START TIMED CAPTURE.")
        self._log(
            "WristHome",
            f"c=start {wrist_prep_sec:.1f}s prep, then timed capture ({self.args.hold_capture_sec:.1f}s); "
            "r=abort/reset, q=quit",
        )
        left_poses: list[np.ndarray] = []
        right_poses: list[np.ndarray] = []
        preparing = False
        capturing = False
        capture_start_time = 0.0
        capture_deadline = 0.0
        capture_extension_deadline = 0.0
        source_stats_at_start: dict[str, object] = {}
        left_last_sample_key: tuple[object, object, object] | None = None
        right_last_sample_key: tuple[object, object, object] | None = None
        stage_gate = ChangeGate()
        left_min_reported = False
        right_min_reported = False
        waiting_for_samples_reported = False
        while True:
            loop_start = time.time()
            channels = self._poll_channels()
            events = self.key_input.consume_events()
            if self.key_input.should_quit:
                raise KeyboardInterrupt
            if "r" in events:
                left_poses = []
                right_poses = []
                preparing = False
                capturing = False
                left_min_reported = False
                right_min_reported = False
                waiting_for_samples_reported = False
                self._log("WristHome", "reset", now=loop_start)
            if "c" in events:
                if not capturing and not preparing:
                    left_poses = []
                    right_poses = []
                    preparing = True
                    capture_start_time = loop_start + wrist_prep_sec
                    capture_deadline = capture_start_time + self.args.hold_capture_sec
                    capture_extension_deadline = capture_deadline + self.args.capture_extra_wait_sec
                    source_stats_at_start = {}
                    left_last_sample_key = None
                    right_last_sample_key = None
                    left_min_reported = False
                    right_min_reported = False
                    waiting_for_samples_reported = False
                    self._log(
                        "WristHome",
                        f"CAPTURE PREP prep={wrist_prep_sec:.1f}s "
                        f"start_at={_format_wall_time(capture_start_time)} "
                        f"auto_end_at={_format_wall_time(capture_deadline)}",
                        now=loop_start,
                    )
                else:
                    self._log(
                        "WristHome",
                        "CAPTURE ALREADY ACTIVE; wait for auto-end or press r to abort",
                        now=loop_start,
                    )

            left_channel = channels.get("left_wrist_pose")
            right_channel = channels.get("right_wrist_pose")
            left_valid = bool(left_channel is not None and left_channel.valid)
            right_valid = bool(right_channel is not None and right_channel.valid)
            validity_snapshot = (left_valid, right_valid)
            if preparing and loop_start >= capture_start_time:
                preparing = False
                capturing = True
                source_stats_at_start = self._source_debug_stats()
                if left_valid:
                    left_last_sample_key = _channel_sample_key(left_channel)
                if right_valid:
                    right_last_sample_key = _channel_sample_key(right_channel)
                self._log(
                    "WristHome",
                    f"CAPTURE STARTED window={self.args.hold_capture_sec:.1f}s "
                    f"auto_end_at={_format_wall_time(capture_deadline)}",
                    now=loop_start,
                )
            elapsed = max(0.0, loop_start - capture_start_time) if capturing else 0.0
            waiting_for_samples = bool(
                capturing
                and loop_start >= capture_deadline
                and (len(left_poses) < self.args.min_hold_samples or len(right_poses) < self.args.min_hold_samples)
            )
            capture_active = preparing or capturing
            capture_phase = (
                "preparing"
                if preparing
                else "waiting"
                if waiting_for_samples
                else "capturing"
                if capturing
                else "ready"
            )
            self._publish_ui_state(
                stage="Wrist/Home Capture",
                active_controls=["r", "q"] if capture_active else ["c", "r", "q"],
                control_hint="左右手起始姿态",
                capture_stage="WristHome",
                capture_phase=capture_phase,
                capturing=capture_active,
                capture_channel="left_wrist_pose,right_wrist_pose",
                capture_samples=None,
                capture_left_samples=len(left_poses),
                capture_right_samples=len(right_poses),
                capture_min_samples=self.args.min_hold_samples,
                capture_valid=None,
                capture_left_valid=left_valid,
                capture_right_valid=right_valid,
                capture_elapsed_sec=elapsed if capturing else None,
                capture_duration_sec=(
                    self.args.hold_capture_sec + self.args.capture_extra_wait_sec
                    if waiting_for_samples
                    else self.args.hold_capture_sec
                ),
                capture_prep_remaining_sec=max(0.0, capture_start_time - loop_start) if preparing else None,
                capture_prep_total_sec=wrist_prep_sec,
                capture_waiting_for_samples=waiting_for_samples,
            )
            if capturing and stage_gate.should_emit("wrist_coarse_validity", validity_snapshot, now=loop_start):
                self._log(
                    "WristHome",
                    f"channel_valid={{'left': {left_valid}, 'right': {right_valid}}}",
                    now=loop_start,
                )
            if capturing and left_valid:
                left_sample_key = _channel_sample_key(left_channel)
                if left_sample_key != left_last_sample_key:
                    left_last_sample_key = left_sample_key
                    left_poses.append(self._channel_pose_virtual_raw(left_channel))
                if len(left_poses) >= self.args.min_hold_samples and not left_min_reported:
                    left_min_reported = True
                    self._log(
                        "WristHome",
                        f"left minimum sample target reached {len(left_poses)}/{self.args.min_hold_samples}",
                        now=loop_start,
                    )
            if capturing and right_valid:
                right_sample_key = _channel_sample_key(right_channel)
                if right_sample_key != right_last_sample_key:
                    right_last_sample_key = right_sample_key
                    right_poses.append(self._channel_pose_virtual_raw(right_channel))
                if len(right_poses) >= self.args.min_hold_samples and not right_min_reported:
                    right_min_reported = True
                    self._log(
                        "WristHome",
                        f"right minimum sample target reached {len(right_poses)}/{self.args.min_hold_samples}",
                        now=loop_start,
                    )
            if capturing and stage_gate.should_emit(
                "wrist_coarse_live",
                validity_snapshot,
                now=loop_start,
                heartbeat_sec=DEFAULT_LOG_HEARTBEAT_SEC,
            ):
                elapsed = max(0.0, loop_start - capture_start_time)
                self._log(
                    "WristHome",
                    f"LIVE time={elapsed:.1f}/{self.args.hold_capture_sec:.1f}s "
                    f"left={len(left_poses)} right={len(right_poses)} "
                    f"valid={{'left': {left_valid}, 'right': {right_valid}}} "
                    f"{self._format_source_rates(start_stats=source_stats_at_start, channel_names=('left_wrist_pose', 'right_wrist_pose'), elapsed_sec=elapsed, now=loop_start)}",
                    now=loop_start,
                )
            if capturing:
                if loop_start >= capture_deadline:
                    capturing = False
                    left_hz = len(left_poses) / max(elapsed, 1e-6)
                    right_hz = len(right_poses) / max(elapsed, 1e-6)
                    if len(left_poses) >= self.args.min_hold_samples and len(right_poses) >= self.args.min_hold_samples:
                        self._log(
                            "WristHome",
                            f"AUTO-END ACCEPTED left={len(left_poses)} right={len(right_poses)} "
                            f"left_hz={left_hz:.1f} right_hz={right_hz:.1f} "
                            f"{self._format_source_rates(start_stats=source_stats_at_start, channel_names=('left_wrist_pose', 'right_wrist_pose'), elapsed_sec=elapsed, now=loop_start)}",
                            now=loop_start,
                        )
                        return left_poses, right_poses
                    if loop_start < capture_extension_deadline:
                        capturing = True
                        if not waiting_for_samples_reported:
                            waiting_for_samples_reported = True
                            self._log(
                                "WristHome",
                                f"AUTO-END WAITING left={len(left_poses)} right={len(right_poses)} "
                                f"need>={self.args.min_hold_samples} left_hz={left_hz:.1f} right_hz={right_hz:.1f} "
                                f"extra_until={_format_wall_time(capture_extension_deadline)}",
                                now=loop_start,
                            )
                        self._render()
                        time.sleep(max(0.0, (1.0 / self.args.hz) - (time.time() - loop_start)))
                        continue
                    self._log(
                        "WristHome",
                        f"AUTO-END INSUFFICIENT left={len(left_poses)} right={len(right_poses)} "
                        f"need>={self.args.min_hold_samples} left_hz={left_hz:.1f} right_hz={right_hz:.1f} "
                        f"{self._format_source_rates(start_stats=source_stats_at_start, channel_names=('left_wrist_pose', 'right_wrist_pose'), elapsed_sec=elapsed, now=loop_start)}",
                        now=loop_start,
                    )
                    left_poses = []
                    right_poses = []
                    left_min_reported = False
                    right_min_reported = False
                    waiting_for_samples_reported = False
                    left_last_sample_key = None
                    right_last_sample_key = None
            self._render()
            time.sleep(max(0.0, (1.0 / self.args.hz) - (time.time() - loop_start)))

    def _run_dual_wrist_calibration(self) -> None:
        left_poses, right_poses = self._capture_dual_wrist_coarse_holds()
        per_side_inputs = {
            "left": (
                left_poses,
                self.args.left_tracker_translation,
                self.args.left_hand_canonical_rpy_deg,
            ),
            "right": (
                right_poses,
                self.args.right_tracker_translation,
                self.args.right_hand_canonical_rpy_deg,
            ),
        }
        tracker_metadata = self.components.calibration.metadata.setdefault(
            "tracker_offset_calibration", {}
        )
        for side, (poses, translation, canonical_rpy_deg) in per_side_inputs.items():
            tracker_role = f"{side}_wrist"
            hand_rotation_virtual = euler_xyz_deg_to_matrix(canonical_rpy_deg)
            tracker_pose_virtual = average_pose(poses)
            coarse_offset = tracker_offset_from_neutral_pose(
                tracker_pose_virtual,
                hand_rotation_virtual=hand_rotation_virtual,
                translation_tracker_to_hand=translation,
            )
            self.coarse_tracker_offsets[side] = coarse_offset
            self.fine_tune_rpy_deg[side] = np.zeros(3, dtype=float)
            self.components.calibration.set_tracker_offset(tracker_role, coarse_offset)
            tracker_metadata[tracker_role] = {
                "canonical_hand_rpy_deg": canonical_rpy_deg.tolist(),
                "canonical_hand_frame_version": TRACKER_OFFSET_CALIBRATION_VERSION,
                "translation_tracker_to_hand": np.asarray(translation, dtype=float).tolist(),
                "coarse_offset": coarse_offset.tolist(),
                "fine_tune_rpy_deg": [0.0, 0.0, 0.0],
                "final_offset": coarse_offset.tolist(),
                "captured_tracker_pose_virtual": tracker_pose_virtual.tolist(),
            }
            self._log(
                "WristHome",
                f"{side} tracker_offset captured translation="
                f"{np.array2string(np.asarray(translation, dtype=float), precision=3)} "
                f"canonical_rpy_deg={np.array2string(np.asarray(canonical_rpy_deg, dtype=float), precision=1)} "
                f"canonical_version={TRACKER_OFFSET_CALIBRATION_VERSION}",
            )
        if not self.args.skip_session_home:
            left_human_home = average_pose(
                [
                    pose @ self.components.calibration.get_tracker_offset("left_wrist")
                    for pose in left_poses
                ]
            )
            right_human_home = average_pose(
                [
                    pose @ self.components.calibration.get_tracker_offset("right_wrist")
                    for pose in right_poses
                ]
            )
            self._set_home_anchors_from_human_poses(
                left_human_home=left_human_home,
                right_human_home=right_human_home,
                source="wrist_home_capture",
            )

    def _run_tracker_frame_visualization(self) -> None:
        self._render_enabled = True
        self._publish_ui_state(
            stage="Tracker Frame Viz",
            active_controls=["1", "2", "t", "j", "l", "i", "k", "u", "o", "x", "v", "f", "s", "q"],
            control_hint="检查左右 tracker/frame；微调后按 Save/s 继续。",
            **self._empty_capture_ui_state(),
        )
        self._log(
            "TrackerFrameViz",
            "Viewer is interactive. Move the camera in SAPIEN to inspect desk/base frames, raw wrist tracker frames, "
            "and calibrated hand-control frames.",
        )
        self._log(
            "TrackerFrameViz",
            "Scene convention: +X=forward, +Y=left, +Z=up; left arm base is at -Y, right arm base is at +Y.",
        )
        self._log(
            "TrackerFrameViz",
            "1/2=select left/right, t=toggle selected side, "
            "j/l=yaw-/+, i/k=pitch+/-, u/o=roll-/+, x=reset selected fine tune, "
            f"v=cycle frame_viz_mode(current={self.frame_viz_mode}), "
            f"f=toggle XY sign flip(current_yaw={self.virtual_desk_post_yaw_deg:.0f}), "
            "s=save+exit, q=quit",
        )
        while True:
            loop_start = time.time()
            channels = self._poll_channels()
            events = self.key_input.consume_events()
            if self.key_input.should_quit:
                raise KeyboardInterrupt

            tuned = False
            channel_valid = {
                name: bool(channels.get(name) is not None and channels[name].valid)
                for name in ("left_wrist_pose", "right_wrist_pose")
            }
            if self.log_gate.should_emit(
                "tracker_frame_viz_channel_valid",
                tuple(sorted(channel_valid.items())),
                now=loop_start,
            ):
                self._log("TrackerFrameViz", f"channel_valid={channel_valid}", now=loop_start)
            for event in events:
                if event == "1":
                    self._set_selected_side("left", "TrackerFrameViz", now=loop_start)
                elif event == "2":
                    self._set_selected_side("right", "TrackerFrameViz", now=loop_start)
                elif event == "t":
                    self._toggle_selected_side("TrackerFrameViz", now=loop_start)
                elif event == "j":
                    self.fine_tune_rpy_deg[self.selected_side][2] -= self.args.fine_tune_step_deg
                    tuned = True
                elif event == "l":
                    self.fine_tune_rpy_deg[self.selected_side][2] += self.args.fine_tune_step_deg
                    tuned = True
                elif event == "i":
                    self.fine_tune_rpy_deg[self.selected_side][1] += self.args.fine_tune_step_deg
                    tuned = True
                elif event == "k":
                    self.fine_tune_rpy_deg[self.selected_side][1] -= self.args.fine_tune_step_deg
                    tuned = True
                elif event == "u":
                    self.fine_tune_rpy_deg[self.selected_side][0] -= self.args.fine_tune_step_deg
                    tuned = True
                elif event == "o":
                    self.fine_tune_rpy_deg[self.selected_side][0] += self.args.fine_tune_step_deg
                    tuned = True
                elif event == "x":
                    self.fine_tune_rpy_deg[self.selected_side] = np.zeros(3, dtype=float)
                    tuned = True
            if "v" in events:
                self._cycle_frame_viz_mode(now=loop_start)
            if "f" in events:
                self._toggle_virtual_desk_xy_flip(now=loop_start)
            if tuned:
                self._apply_side_fine_tune(self.selected_side)
                self._log(
                    "TrackerFrameViz",
                    f"updated {self.selected_side} fine tune to "
                    f"{np.array2string(self.fine_tune_rpy_deg[self.selected_side], precision=1)} deg",
                    now=loop_start,
                )

            self._set_debug_frames(
                self._build_debug_frames(
                    channels,
                    include_actual_tcp=True,
                    include_target_tcp=False,
                )
            )
            if "s" in events:
                self._log("TrackerFrameViz", "saving calibration profile and exiting", now=loop_start)
                self._render_enabled = False
                self._set_debug_frames({})
                return
            self._render()
            time.sleep(max(0.0, (1.0 / self.args.hz) - (time.time() - loop_start)))

    def _capture_home(self) -> None:
        left_poses, right_poses = self._capture_dual_home_holds()
        left_human_home = average_pose(left_poses)
        right_human_home = average_pose(right_poses)
        self._set_home_anchors_from_human_poses(
            left_human_home=left_human_home,
            right_human_home=right_human_home,
            source="home_recapture",
        )

    def _robot_home_pose_desk_for_side(self, side: str) -> np.ndarray | None:
        if not self._arm_visualization_enabled.get(side, False):
            return None
        backend = self._backend_for_side(side)
        arm = self.components.calibration.get_arm_for_side(side)
        if arm.T_D_B is None:
            return None
        state = backend.get_state()
        qpos = state.get("hardware_qpos") if self.args.arm_mode == "real" else state.get("current_qpos")
        if qpos is None:
            return None
        try:
            robot_home_base = backend.compute_tcp_pose(self.args.tcp_mode, qpos=qpos)
        except Exception as exc:
            self._log("HomeCapture", f"{side} robot TCP pose unavailable: {exc}")
            return None
        return arm.T_D_B @ robot_home_base

    def _robot_home_qpos_for_side(self, side: str) -> np.ndarray | None:
        if not self._arm_visualization_enabled.get(side, False):
            return None
        backend = self._backend_for_side(side)
        state = backend.get_state()
        qpos = state.get("hardware_qpos") if self.args.arm_mode == "real" else state.get("current_qpos")
        return None if qpos is None else np.asarray(qpos, dtype=float).copy()

    def _set_home_anchors_from_human_poses(
        self,
        *,
        left_human_home: np.ndarray,
        right_human_home: np.ndarray,
        source: str,
    ) -> bool:
        self._refresh_arm_render_state(force=True)
        left_robot_home_desk = self._robot_home_pose_desk_for_side("left")
        right_robot_home_desk = self._robot_home_pose_desk_for_side("right")
        if left_robot_home_desk is None or right_robot_home_desk is None:
            self._clear_home()
            missing = []
            if left_robot_home_desk is None:
                missing.append("left_robot_home")
            if right_robot_home_desk is None:
                missing.append("right_robot_home")
            self._log(
                "HomeCapture",
                f"cannot set home anchors; missing={missing}. "
                "SAPIEN will show VIVE/coordinate frames only until real arm qpos is readable.",
            )
            return False

        self.components.left_retargeter.set_home_anchors(
            human_pose_desk=left_human_home,
            robot_pose_desk=left_robot_home_desk,
        )
        self.components.right_retargeter.set_home_anchors(
            human_pose_desk=right_human_home,
            robot_pose_desk=right_robot_home_desk,
        )
        self.components.calibration.capture_home_pose("left_arm", left_robot_home_desk, tcp_mode=self.args.tcp_mode)
        self.components.calibration.capture_home_pose("right_arm", right_robot_home_desk, tcp_mode=self.args.tcp_mode)
        left_qpos = self._robot_home_qpos_for_side("left")
        right_qpos = self._robot_home_qpos_for_side("right")
        if left_qpos is not None:
            self.components.calibration.capture_home_qpos("left_arm", left_qpos)
        if right_qpos is not None:
            self.components.calibration.capture_home_qpos("right_arm", right_qpos)
        self.components.calibration.metadata["home_capture"] = {
            "source": source,
            "tcp_mode": self.args.tcp_mode,
            "left_human_home_virtual": left_human_home.tolist(),
            "right_human_home_virtual": right_human_home.tolist(),
            "left_robot_home_desk": left_robot_home_desk.tolist(),
            "right_robot_home_desk": right_robot_home_desk.tolist(),
        }
        self.home_ready = True
        left_human_tcp_home = self.components.left_retargeter.home_human_pose_desk
        right_human_tcp_home = self.components.right_retargeter.home_human_pose_desk
        self._log(
            "HomeCapture",
            f"home anchors updated source={source} "
            f"left_human_tcp=({_pose_matrix_xyz_rpy_text(left_human_tcp_home) if left_human_tcp_home is not None else 'None'}) "
            f"left_robot_tcp=({_pose_matrix_xyz_rpy_text(left_robot_home_desk)}) "
            f"left_qpos_deg={_qpos_deg_text(left_qpos)}",
        )
        self._log(
            "HomeCapture",
            f"home anchors updated source={source} "
            f"right_human_tcp=({_pose_matrix_xyz_rpy_text(right_human_tcp_home) if right_human_tcp_home is not None else 'None'}) "
            f"right_robot_tcp=({_pose_matrix_xyz_rpy_text(right_robot_home_desk)}) "
            f"right_qpos_deg={_qpos_deg_text(right_qpos)}",
        )
        self._publish_ui_state(
            home_ready=True,
            active_controls=["1", "2", "t", "j", "l", "i", "k", "u", "o", "x", "v", "f", "h", "s", "q"],
            control_hint="左右起始坐标系点已采集；可移动手柄验证三轴方向，按 Save/s 保存。",
            **self._empty_capture_ui_state(),
        )
        return True

    def _apply_side_fine_tune(self, side: str) -> None:
        if side not in self.coarse_tracker_offsets:
            return
        tracker_role = f"{side}_wrist"
        offset = compose_tracker_offset(
            self.coarse_tracker_offsets[side],
            self.fine_tune_rpy_deg[side],
        )
        self.components.calibration.set_tracker_offset(tracker_role, offset)
        tracker_metadata = self.components.calibration.metadata.setdefault("tracker_offset_calibration", {})
        tracker_metadata.setdefault(tracker_role, {})
        tracker_metadata[tracker_role]["canonical_hand_frame_version"] = (
            TRACKER_OFFSET_CALIBRATION_VERSION
        )
        tracker_metadata[tracker_role]["fine_tune_rpy_deg"] = self.fine_tune_rpy_deg[side].tolist()
        tracker_metadata[tracker_role]["final_offset"] = offset.tolist()

    def _save_profile(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            Path(self.args.output_path)
            if self.args.output_path
            else DEFAULT_CALIBRATION_PROFILE_DIR / f"{timestamp}_vive_rm65_virtual_desk.json"
        )
        runtime_home_was_available = bool(self.home_ready)
        self.components.calibration.clear_home_state()
        self.components.calibration.metadata["profile_scope"] = "tracker_station_only"
        self.components.calibration.metadata["runtime_home_used_for_visual_check"] = runtime_home_was_available
        self.components.calibration.metadata["home_policy"] = (
            "session home anchors are intentionally not saved; real teleop must recapture wrist home"
        )
        self.components.calibration.metadata["saved_at"] = datetime.now().isoformat(timespec="seconds")
        self.components.calibration.save_json(output_path)
        self._log("Save", f"calibration profile written to {output_path}")
        self._publish_ui_state(
            stage="Saved",
            active_controls=["q"],
            control_hint=f"saved to {output_path}",
            **self._empty_capture_ui_state(),
        )
        return output_path

    def _preview_command_locally(
        self,
        *,
        side: str,
        backend: RM65ArmBackend,
        command,
        now: float,
    ) -> None:
        state = backend.get_state()
        qpos = state.get("current_qpos")
        if qpos is None:
            qpos = state.get("hardware_qpos")
        if qpos is None:
            qpos = DEFAULT_LEFT_ARM_QPOS if side == "left" else DEFAULT_RIGHT_ARM_QPOS
        backend.set_local_qpos(qpos)
        try:
            if command.command_type == "joint_vector":
                target = np.asarray(command.value, dtype=float).reshape(6)
                max_step = backend.get_effective_max_joint_step(now=now)
                current = np.asarray(backend.get_state()["current_qpos"], dtype=float).reshape(6)
                qpos_target = current + np.clip(target - current, -max_step, max_step)
            elif command.command_type == "cartesian_pose":
                max_step = backend.get_effective_max_joint_step(now=now)
                qpos_target = backend._solve_cartesian_command(command, max_joint_step=max_step)
            else:
                return
        except Exception as exc:
            if self.log_gate.should_emit(f"{side}_preview_ik_failed", str(exc), now=now):
                self._log("Validate", f"{side} SAPIEN preview IK failed: {exc}", now=now)
            return
        backend.set_local_qpos(qpos_target)

    def _run_validation_step(self, channels: dict[str, object]) -> None:
        now = time.time()
        observation_channels = {}
        for name in ("left_wrist_pose", "right_wrist_pose"):
            channel = channels.get(name)
            if channel is not None:
                observation_channels[name] = channel.copy()
        timestamp = latest_channel_timestamp(observation_channels)
        if timestamp is None:
            timestamp = time.time()
        observation = TeleopObservation(
            channels=observation_channels,
            timestamp=timestamp,
            metadata={"tcp_mode": self.args.tcp_mode},
        )
        command = merge_teleop_commands(
            [
                self.components.left_retargeter.retarget(observation, self.components.calibration),
                self.components.right_retargeter.retarget(observation, self.components.calibration),
            ]
        )
        left_command = command.commands.get("left_arm")
        right_command = command.commands.get("right_arm")
        target_poses_desk: dict[str, np.ndarray] = {}
        if left_command is not None:
            target_pose_desk = left_command.metadata.get("target_pose_desk")
            if target_pose_desk is not None:
                target_poses_desk["left"] = as_pose_matrix(target_pose_desk)
            if self.components.left_backend.config.connect_hardware:
                if self.log_gate.should_emit("left_read_only_skip_send", True, now=time.time()):
                    self._log("Validate", "left hardware command skipped; SAPIEN preview updates locally only.")
                self._preview_command_locally(
                    side="left",
                    backend=self.components.left_backend,
                    command=left_command,
                    now=now,
                )
            else:
                self.components.left_backend.send_command(left_command)
        if right_command is not None:
            target_pose_desk = right_command.metadata.get("target_pose_desk")
            if target_pose_desk is not None:
                target_poses_desk["right"] = as_pose_matrix(target_pose_desk)
            if self.components.right_backend.config.connect_hardware:
                if self.log_gate.should_emit("right_read_only_skip_send", True, now=time.time()):
                    self._log("Validate", "right hardware command skipped; SAPIEN preview updates locally only.")
                self._preview_command_locally(
                    side="right",
                    backend=self.components.right_backend,
                    command=right_command,
                    now=now,
                )
            else:
                self.components.right_backend.send_command(right_command)
        self._last_validation_target_poses = target_poses_desk

    def _run_validation_loop(self) -> None:
        self._render_enabled = True
        self._simulate_arm_preview = True
        self._publish_ui_state(
            stage="Validate",
            active_controls=["1", "2", "t", "j", "l", "i", "k", "u", "o", "x", "v", "f", "h", "s", "q"],
            control_hint="SAPIEN 本地预览跟随 VIVE；真机不动。",
            **self._empty_capture_ui_state(),
        )
        self._log(
            "Validate",
            "Viewer is interactive. 1=select left, 2=select right, t=toggle selected side, "
            "j/l=yaw-/+, i/k=pitch+/-, "
            "u/o=roll-/+, x=reset selected fine tune, h=recapture home, v=cycle frame_viz_mode, "
            f"f=toggle XY sign flip(current_yaw={self.virtual_desk_post_yaw_deg:.0f}), "
            "s=save+exit, q=quit",
        )
        self._log(
            "Validate",
            "Scene convention: +X=forward, +Y=left, +Z=up; left arm base is at -Y, right arm base is at +Y.",
        )
        self._log("Validate", "SAPIEN preview uses local qpos after calibration; hardware state is not polled continuously.")
        step_index = 0
        while True:
            loop_start = time.time()
            channels = self._poll_channels()
            events = self.key_input.consume_events()
            if self.key_input.should_quit:
                return

            tuned = False
            for event in events:
                if event == "1":
                    self._set_selected_side("left", "Validate", now=loop_start)
                elif event == "2":
                    self._set_selected_side("right", "Validate", now=loop_start)
                elif event == "t":
                    self._toggle_selected_side("Validate", now=loop_start)
                elif event == "j":
                    self.fine_tune_rpy_deg[self.selected_side][2] -= self.args.fine_tune_step_deg
                    tuned = True
                elif event == "l":
                    self.fine_tune_rpy_deg[self.selected_side][2] += self.args.fine_tune_step_deg
                    tuned = True
                elif event == "i":
                    self.fine_tune_rpy_deg[self.selected_side][1] += self.args.fine_tune_step_deg
                    tuned = True
                elif event == "k":
                    self.fine_tune_rpy_deg[self.selected_side][1] -= self.args.fine_tune_step_deg
                    tuned = True
                elif event == "u":
                    self.fine_tune_rpy_deg[self.selected_side][0] -= self.args.fine_tune_step_deg
                    tuned = True
                elif event == "o":
                    self.fine_tune_rpy_deg[self.selected_side][0] += self.args.fine_tune_step_deg
                    tuned = True
                elif event == "x":
                    self.fine_tune_rpy_deg[self.selected_side] = np.zeros(3, dtype=float)
                    tuned = True
                elif event == "h":
                    self._capture_home()
                elif event == "v":
                    self._cycle_frame_viz_mode(now=loop_start)
                elif event == "f":
                    self._toggle_virtual_desk_xy_flip(now=loop_start)
                elif event == "s":
                    self._save_profile()
                    return

            if tuned:
                self._apply_side_fine_tune(self.selected_side)
                self._clear_home()
                self._log(
                    "Validate",
                    f"updated {self.selected_side} fine tune to "
                    f"{np.array2string(self.fine_tune_rpy_deg[self.selected_side], precision=1)} deg. "
                    "Press 'h' to recapture home.",
                    now=loop_start,
                )

            channel_valid = {
                name: bool(channels.get(name) is not None and channels[name].valid)
                for name in ("left_wrist_pose", "right_wrist_pose")
            }
            channel_snapshot = tuple(sorted(channel_valid.items()))
            if self.log_gate.should_emit("validate_channel_valid", channel_snapshot, now=loop_start):
                self._log("Validate", f"channel_valid={channel_valid}", now=loop_start)
            if self.log_gate.should_emit("validate_home_ready", self.home_ready, now=loop_start):
                self._log("Validate", f"home_ready={self.home_ready}", now=loop_start)

            if self.home_ready:
                self._run_validation_step(channels)

            self._set_debug_frames(
                self._build_debug_frames(
                    channels,
                    include_actual_tcp=True,
                    include_target_tcp=True,
                )
            )
            self._render()
            step_index += 1
            if self.args.max_steps > 0 and step_index >= self.args.max_steps:
                return
            time.sleep(max(0.0, (1.0 / self.args.hz) - (time.time() - loop_start)))


def main() -> None:
    try:
        args = parse_args()
        components = build_components(args)
        app = CalibrationApp(args, components)
        app.run()
    except KeyboardInterrupt:
        print("\n[Exit] calibration stopped by user")


if __name__ == "__main__":
    main()
