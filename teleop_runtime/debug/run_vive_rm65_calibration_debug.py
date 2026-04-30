from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from project_paths import calibration_profiles_dir

from ..backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from ..calibration import TeleopCalibration
from ..configs import (
    ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
    build_annotated_dual_rm65_calibration,
)
from ..hand_tcp_alignment import tracker_hand_to_rm65_tcp_rotation
from ..retargeters import PoseToArmRetargeter, PoseToArmRetargeterConfig
from ..schemas import TeleopObservation, latest_channel_timestamp, merge_teleop_commands
from ..sources.pose import VivePoseSource, ViveTrackerConfig
from ..tcp import HAND_BASE_TCP_FRAME, legacy_tcp_mode_requested
from ..tracker_hand import (
    TRACKER_OFFSET_CALIBRATION_VERSION,
    canonical_hand_rpy_default,
)
from ..transforms import as_pose_matrix, identity_pose, invert_pose
from .calibration_flow import (
    average_pose,
    compose_tracker_offset,
    euler_xyz_deg_to_matrix,
    fit_signed_axis_from_poses,
    fit_virtual_desk_frame,
    pose_from_rotation_translation,
    tracker_offset_from_neutral_pose,
)
from .log_utils import DEFAULT_LOG_HEARTBEAT_SEC, ChangeGate, log_event, log_stage
from rm65_inspire_tac.paths import require_asset_path
from rm65_inspire_tac.pipeline import DEFAULT_LEFT_ARM_QPOS, DEFAULT_RIGHT_ARM_QPOS
from rm65_inspire_tac.rendering import SapienArmHandRenderer

DEFAULT_CALIBRATION_PROFILE_DIR = calibration_profiles_dir()
_VIRTUAL_DESK_SWEEP_CONVENTION_VERSION = 2


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive VIVE -> RM65 calibration debug tool. "
            "This app calibrates a virtual desk frame from an auxiliary tracker, "
            "captures left/right wrist tracker offsets, and saves a reusable profile. "
            "Session home capture is intentionally not part of this tool."
        )
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="UDP bind host for the VIVE receiver")
    parser.add_argument("--port", type=int, default=9999, help="UDP bind port for the VIVE receiver")
    parser.add_argument(
        "--tcp-mode",
        type=str,
        default=HAND_BASE_TCP_FRAME,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--arm-mode",
        choices=["real", "virtual"],
        default="real",
        help="Render read-only real RM65 state when IPs are provided, or pure virtual arm poses.",
    )
    parser.add_argument("--left-arm-ip", type=str, default="", help="Optional left RM65 IP for read-only rendering")
    parser.add_argument("--right-arm-ip", type=str, default="", help="Optional right RM65 IP for read-only rendering")
    parser.add_argument(
        "--left-right-base-distance",
        type=float,
        default=ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
        help="Annotated dual-arm base distance used by the SAPIEN scene and preset calibration.",
    )
    parser.add_argument("--left-tracker-name", type=str, default="tracker_1", help="Left wrist tracker device name")
    parser.add_argument("--right-tracker-name", type=str, default="tracker_2", help="Right wrist tracker device name")
    parser.add_argument("--aux-tracker-name", type=str, default="tracker_3", help="Auxiliary calibration tracker device name")
    parser.add_argument("--headless", action="store_true", help="Disable the SAPIEN viewer window")
    parser.add_argument(
        "--skip-renderer",
        action="store_true",
        help="Skip SAPIEN renderer creation. Useful when only terminal validation is needed.",
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
        default=3.0,
        help="Timed capture duration for each VirtualDesk X/Y/Z sweep",
    )
    parser.add_argument(
        "--hold-capture-sec",
        type=_positive_float,
        default=2.0,
        help="Timed capture duration for neutral and wrist coarse captures",
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
        "--min-hold-samples",
        type=int,
        default=20,
        help="Minimum valid samples required for neutral and wrist hold captures",
    )
    parser.add_argument(
        "--fine-tune-step-deg",
        type=_positive_float,
        default=2.0,
        help="Fine tuning step for wrist orientation offsets during validation",
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
    args.requested_tcp_mode = str(getattr(args, "tcp_mode", HAND_BASE_TCP_FRAME))
    args.tcp_mode = HAND_BASE_TCP_FRAME
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
            "tcp_mode": args.tcp_mode,
            "arm_mode": args.arm_mode,
            "left_arm_ip": args.left_arm_ip,
            "right_arm_ip": args.right_arm_ip,
        }
    )
    source = VivePoseSource(
        tracker_configs=[
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
            ViveTrackerConfig(
                name=args.aux_tracker_name,
                side=None,
                role="auxiliary",
                channel_name="auxiliary_tracker_pose",
                legacy_index=2,
                critical=False,
            ),
        ],
        host=args.host,
        port=args.port,
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
    renderer = None
    if not args.skip_renderer:
        try:
            renderer = SapienArmHandRenderer(
                left_urdf_path=arm_urdf_path,
                left_retargeting_joint_names=(),
                right_urdf_path=arm_urdf_path,
                right_retargeting_joint_names=(),
                left_right_base_distance=args.left_right_base_distance,
                show_viewer=not args.headless,
                debug_frame_axis_length=args.debug_frame_axis_length,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to create the SAPIEN renderer. Check that the current environment "
                "has a usable Vulkan/OpenGL render device, or rerun with --skip-renderer "
                "to validate the calibration flow without 3D visualization."
            ) from exc

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
        renderer=renderer,
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
        self.key_input = TerminalKeyInput()
        self.virtual_desk_result = None
        self.coarse_tracker_offsets: dict[str, np.ndarray] = {}
        self.fine_tune_rpy_deg = {
            "left": np.zeros(3, dtype=float),
            "right": np.zeros(3, dtype=float),
        }
        self.home_ready = False
        self.empty_hand = np.empty(0, dtype=float)
        self.selected_side = "left"
        self.log_gate = ChangeGate()
        self._render_enabled = False
        self._debug_frames: dict[str, np.ndarray] = {}
        self._last_validation_target_poses: dict[str, np.ndarray] = {}
        self._last_arm_refresh_time: float | None = None
        self._arm_refresh_period_sec = 0.10
        self.frame_viz_mode = str(self.args.frame_viz_mode)
        self.virtual_desk_post_yaw_deg = float(self.args.virtual_desk_post_yaw_deg)
        self._base_world_to_desk_pose: np.ndarray | None = None
        if self.components.renderer is not None:
            self.components.renderer.reserve_debug_frames(
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
        self._restore_loaded_profile_state()

    def _log(self, tag: str, message: str, *, now: float | None = None) -> None:
        log_event(tag, message, now=now)

    def _stage(self, title: str, *, now: float | None = None) -> None:
        log_stage("STAGE", title, now=now)

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
        if not self.key_input.enabled:
            raise RuntimeError("Calibration debug requires an interactive terminal (TTY).")
        try:
            self.components.left_backend.start()
            self.components.right_backend.start()
            self.components.source.start()
            self.key_input.start()
            if legacy_tcp_mode_requested(self.args.requested_tcp_mode):
                self._log(
                    "TCP",
                    "legacy --tcp-mode rm_eef was requested; runtime now uses hand_base only.",
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
                        "arm-mode=real but no arm IPs provided; rendering falls back to virtual default poses.",
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
                self._stage("Wrist Coarse")
                self._run_dual_wrist_calibration()
            if self.args.show_tracker_frame_viz and self.components.renderer is not None:
                self._stage("Tracker Frame Viz")
                self._run_tracker_frame_visualization()
            self._stage("Save")
            self._save_profile()
        finally:
            self.key_input.close()
            self.components.source.close()
            self.components.left_backend.close()
            self.components.right_backend.close()

    def _poll_channels(self) -> dict[str, object]:
        self.key_input.poll()
        return self.components.source.get_latest_channels()

    def _render(self) -> None:
        if not self._render_enabled or self.components.renderer is None:
            return
        left_qpos = _state_qpos(self.components.left_backend, DEFAULT_LEFT_ARM_QPOS)
        right_qpos = _state_qpos(self.components.right_backend, DEFAULT_RIGHT_ARM_QPOS)
        self.components.renderer.set_debug_frames(self._debug_frames)
        self.components.renderer.render_data(
            left_qpos_arm=left_qpos,
            left_qpos_hand=self.empty_hand,
            right_qpos_arm=right_qpos,
            right_qpos_hand=self.empty_hand,
            capture_image=False,
        )

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

    def _set_debug_frames(self, frames: dict[str, np.ndarray] | None) -> None:
        self._debug_frames = {} if frames is None else {key: value.copy() for key, value in frames.items()}

    def _cycle_frame_viz_mode(self, *, now: float | None = None) -> None:
        order = ("calibrated", "raw", "both")
        current_index = order.index(self.frame_viz_mode)
        self.frame_viz_mode = order[(current_index + 1) % len(order)]
        self._log("FrameViz", f"frame_viz_mode={self.frame_viz_mode}", now=now)

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

    def _desk_debug_frames(self) -> dict[str, np.ndarray]:
        frames = {"desk": identity_pose()}
        for side in ("left", "right"):
            arm = self.components.calibration.get_arm_for_side(side)
            if arm.T_D_B is not None:
                frames[f"{side}_arm_base"] = arm.T_D_B.copy()
        return frames

    def _current_actual_tcp_pose_desk(self, side: str) -> np.ndarray | None:
        backend = self.components.left_backend if side == "left" else self.components.right_backend
        arm = self.components.calibration.get_arm_for_side(side)
        if arm.T_D_B is None:
            return None
        try:
            pose_base = backend.compute_tcp_pose(self.args.tcp_mode)
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

    def _run_link_check(self) -> None:
        self._log("LinkCheck", "READY: PRESS N TO CONTINUE AFTER left/right/aux tracker channels are present.")
        self._log("LinkCheck", "q=quit, n=continue when left/right/aux tracker channels are present.")
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
                for name in ("left_wrist_pose", "right_wrist_pose", "auxiliary_tracker_pose")
            }
            status_snapshot = (
                tuple(sorted(ready.items())),
                tuple(sorted(channels)),
            )
            if self.log_gate.should_emit("link_check_status", status_snapshot, now=loop_start):
                self._log(
                    "LinkCheck",
                    f"ready={ready} channels={sorted(channels)} press 'n' to continue",
                    now=loop_start,
                )
            if "n" in events and all(ready.values()):
                self._log("LinkCheck", "all required trackers are ready; continuing", now=loop_start)
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
        channel_name: str,
        min_samples: int,
        capture_duration_sec: float,
        transform,
        validator=None,
    ) -> list[np.ndarray]:
        self._log(stage_name, instructions)
        self._log(stage_name, timed_flow_hint)
        self._log(stage_name, "READY: PRESS C TO START TIMED CAPTURE.")
        self._log(
            stage_name,
            f"c=start timed capture ({capture_duration_sec:.1f}s), r=abort/reset, q=quit",
        )
        poses: list[np.ndarray] = []
        capturing = False
        capture_start_time = 0.0
        capture_deadline = 0.0
        last_sample_key: tuple[object, object, object] | None = None
        stage_gate = ChangeGate()
        min_samples_reported = False
        while True:
            loop_start = time.time()
            channels = self._poll_channels()
            events = self.key_input.consume_events()
            if self.key_input.should_quit:
                raise KeyboardInterrupt
            if "r" in events:
                poses = []
                capturing = False
                min_samples_reported = False
                self._log(stage_name, "reset", now=loop_start)
            if "c" in events:
                if not capturing:
                    poses = []
                    capturing = True
                    capture_start_time = loop_start
                    capture_deadline = capture_start_time + capture_duration_sec
                    last_sample_key = None
                    min_samples_reported = False
                    self._log(
                        stage_name,
                        f"CAPTURE STARTED window={capture_duration_sec:.1f}s auto_end_at={_format_wall_time(capture_deadline)}",
                        now=loop_start,
                    )
                else:
                    self._log(
                        stage_name,
                        "CAPTURE ALREADY RUNNING; wait for auto-end or press r to abort",
                        now=loop_start,
                    )

            channel = channels.get(channel_name)
            channel_valid = bool(channel is not None and channel.valid)
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
                    f"LIVE time={elapsed:.1f}/{capture_duration_sec:.1f}s samples={len(poses)} valid={channel_valid}",
                    now=loop_start,
                )
            if capturing:
                elapsed = max(0.0, loop_start - capture_start_time)
                if loop_start >= capture_deadline:
                    capturing = False
                    effective_hz = len(poses) / max(elapsed, 1e-6)
                    if len(poses) < min_samples:
                        self._log(
                            stage_name,
                            f"AUTO-END INSUFFICIENT samples={len(poses)} need>={min_samples} "
                            f"effective_hz={effective_hz:.1f}",
                            now=loop_start,
                        )
                        poses = []
                        min_samples_reported = False
                        last_sample_key = None
                    elif validator is None:
                        self._log(
                            stage_name,
                            f"AUTO-END ACCEPTED samples={len(poses)} effective_hz={effective_hz:.1f}",
                            now=loop_start,
                        )
                        return poses
                    else:
                        error = validator(poses)
                        if error is None:
                            self._log(
                                stage_name,
                                f"AUTO-END ACCEPTED samples={len(poses)} effective_hz={effective_hz:.1f}",
                                now=loop_start,
                            )
                            return poses
                        self._log(
                            stage_name,
                            f"AUTO-END REJECTED: {error} effective_hz={effective_hz:.1f}",
                            now=loop_start,
                        )
                        poses = []
                        min_samples_reported = False
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
            f"c=start timed capture ({self.args.home_capture_sec:.1f}s), r=abort/reset, q=quit",
        )
        left_poses: list[np.ndarray] = []
        right_poses: list[np.ndarray] = []
        capturing = False
        capture_start_time = 0.0
        capture_deadline = 0.0
        left_last_sample_key: tuple[object, object, object] | None = None
        right_last_sample_key: tuple[object, object, object] | None = None
        stage_gate = ChangeGate()
        left_min_reported = False
        right_min_reported = False
        while True:
            loop_start = time.time()
            channels = self._poll_channels()
            events = self.key_input.consume_events()
            if self.key_input.should_quit:
                raise KeyboardInterrupt
            if "r" in events:
                left_poses = []
                right_poses = []
                capturing = False
                left_min_reported = False
                right_min_reported = False
                self._log("HomeCapture", "reset", now=loop_start)
            if "c" in events:
                if not capturing:
                    left_poses = []
                    right_poses = []
                    capturing = True
                    capture_start_time = loop_start
                    capture_deadline = capture_start_time + self.args.home_capture_sec
                    left_last_sample_key = None
                    right_last_sample_key = None
                    left_min_reported = False
                    right_min_reported = False
                    self._log(
                        "HomeCapture",
                        f"CAPTURE STARTED window={self.args.home_capture_sec:.1f}s "
                        f"auto_end_at={_format_wall_time(capture_deadline)}",
                        now=loop_start,
                    )
                else:
                    self._log(
                        "HomeCapture",
                        "CAPTURE ALREADY RUNNING; wait for auto-end or press r to abort",
                        now=loop_start,
                    )

            left_channel = channels.get("left_wrist_pose")
            right_channel = channels.get("right_wrist_pose")
            left_valid = bool(left_channel is not None and left_channel.valid)
            right_valid = bool(right_channel is not None and right_channel.valid)
            validity_snapshot = (left_valid, right_valid)
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
                    f"valid={{'left': {left_valid}, 'right': {right_valid}}}",
                    now=loop_start,
                )
            if capturing:
                elapsed = max(0.0, loop_start - capture_start_time)
                if loop_start >= capture_deadline:
                    capturing = False
                    left_hz = len(left_poses) / max(elapsed, 1e-6)
                    right_hz = len(right_poses) / max(elapsed, 1e-6)
                    if len(left_poses) >= self.args.home_samples and len(right_poses) >= self.args.home_samples:
                        self._log(
                            "HomeCapture",
                            f"AUTO-END ACCEPTED left={len(left_poses)} right={len(right_poses)} "
                            f"left_hz={left_hz:.1f} right_hz={right_hz:.1f}",
                            now=loop_start,
                        )
                        return left_poses, right_poses
                    self._log(
                        "HomeCapture",
                        f"AUTO-END INSUFFICIENT left={len(left_poses)} right={len(right_poses)} "
                        f"need>={self.args.home_samples} left_hz={left_hz:.1f} right_hz={right_hz:.1f}",
                        now=loop_start,
                    )
                    left_poses = []
                    right_poses = []
                    left_min_reported = False
                    right_min_reported = False
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

        x_sweep = self._capture_single_channel(
            stage_name="VirtualDeskX",
            instructions=(
                "Hold the auxiliary tracker and perform the X sweep so that the fitted +X points forward: "
                "start nearer your back and finish in front of your body."
            ),
            timed_flow_hint=(
                "Press c once, then sweep BACK -> FRONT. "
                "Start nearer your back side and finish in front of your body before auto-end."
            ),
            channel_name="auxiliary_tracker_pose",
            min_samples=max(8, self.args.min_hold_samples),
            capture_duration_sec=self.args.sweep_capture_sec,
            transform=lambda channel: as_pose_matrix(channel.value),
            validator=_sweep_validator,
        )
        y_sweep = self._capture_single_channel(
            stage_name="VirtualDeskY",
            instructions=(
                "Perform the Y sweep so that the fitted +Y points to your left: "
                "start on your right side and finish on your left side."
            ),
            timed_flow_hint=(
                "Press c once, then sweep RIGHT -> LEFT. "
                "Start near your right side and finish near your left side before auto-end."
            ),
            channel_name="auxiliary_tracker_pose",
            min_samples=max(8, self.args.min_hold_samples),
            capture_duration_sec=self.args.sweep_capture_sec,
            transform=lambda channel: as_pose_matrix(channel.value),
            validator=_sweep_validator,
        )
        z_sweep = self._capture_single_channel(
            stage_name="VirtualDeskZ",
            instructions=(
                "Move the auxiliary tracker upward and downward. Start low and finish high to define +Z."
            ),
            timed_flow_hint=(
                "Press c once, start low, move upward, and finish high at the end of the timed window. "
                "Do not return downward before auto-end."
            ),
            channel_name="auxiliary_tracker_pose",
            min_samples=max(8, self.args.min_hold_samples),
            capture_duration_sec=self.args.sweep_capture_sec,
            transform=lambda channel: as_pose_matrix(channel.value),
            validator=_sweep_validator,
        )
        neutral_hold = self._capture_single_channel(
            stage_name="VirtualDeskNeutral",
            instructions=(
                "Hold the auxiliary tracker near your preferred teleop center. "
                "This sets the session origin of the virtual desk."
            ),
            timed_flow_hint=(
                "Press c once, then hold the tracker as still as possible until the timed capture ends automatically."
            ),
            channel_name="auxiliary_tracker_pose",
            min_samples=self.args.min_hold_samples,
            capture_duration_sec=self.args.hold_capture_sec,
            transform=lambda channel: as_pose_matrix(channel.value),
        )
        result = fit_virtual_desk_frame(
            x_sweep_poses=x_sweep,
            y_sweep_poses=y_sweep,
            z_sweep_poses=z_sweep,
            neutral_poses=neutral_hold,
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
        self.components.calibration.metadata["virtual_desk_sweep_convention_version"] = (
            _VIRTUAL_DESK_SWEEP_CONVENTION_VERSION
        )
        self._log(
            "VirtualDesk",
            f"calibrated x={np.array2string(result.x_axis_world, precision=3)} "
            f"y={np.array2string(result.y_axis_world, precision=3)} "
            f"z={np.array2string(result.z_axis_world, precision=3)}",
        )
        if abs(self.virtual_desk_post_yaw_deg) > 1e-6:
            self._log(
                "VirtualDesk",
                f"applied post yaw correction {self.virtual_desk_post_yaw_deg:.0f}deg",
            )

    def _capture_dual_wrist_coarse_holds(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        self._log(
            "WristCoarse",
            "Place both hands in the canonical neutral pose: forearms forward, hand backs up, "
            "and fingers roughly aligned with virtual-desk +X.",
        )
        self._log(
            "WristCoarse",
            "Default canonical wrist frame uses XYZ Euler [0, 0, 0] deg; the captured neutral pose directly defines the tracker-to-hand correction.",
        )
        self._log(
            "WristCoarse",
            "Press c once, then keep both wrist trackers and both hand poses steady until the timed capture ends automatically.",
        )
        self._log("WristCoarse", "READY: PRESS C TO START TIMED CAPTURE.")
        self._log(
            "WristCoarse",
            f"c=start timed capture ({self.args.hold_capture_sec:.1f}s), r=abort/reset, q=quit",
        )
        left_poses: list[np.ndarray] = []
        right_poses: list[np.ndarray] = []
        capturing = False
        capture_start_time = 0.0
        capture_deadline = 0.0
        left_last_sample_key: tuple[object, object, object] | None = None
        right_last_sample_key: tuple[object, object, object] | None = None
        stage_gate = ChangeGate()
        left_min_reported = False
        right_min_reported = False
        while True:
            loop_start = time.time()
            channels = self._poll_channels()
            events = self.key_input.consume_events()
            if self.key_input.should_quit:
                raise KeyboardInterrupt
            if "r" in events:
                left_poses = []
                right_poses = []
                capturing = False
                left_min_reported = False
                right_min_reported = False
                self._log("WristCoarse", "reset", now=loop_start)
            if "c" in events:
                if not capturing:
                    left_poses = []
                    right_poses = []
                    capturing = True
                    capture_start_time = loop_start
                    capture_deadline = capture_start_time + self.args.hold_capture_sec
                    left_last_sample_key = None
                    right_last_sample_key = None
                    left_min_reported = False
                    right_min_reported = False
                    self._log(
                        "WristCoarse",
                        f"CAPTURE STARTED window={self.args.hold_capture_sec:.1f}s "
                        f"auto_end_at={_format_wall_time(capture_deadline)}",
                        now=loop_start,
                    )
                else:
                    self._log(
                        "WristCoarse",
                        "CAPTURE ALREADY RUNNING; wait for auto-end or press r to abort",
                        now=loop_start,
                    )

            left_channel = channels.get("left_wrist_pose")
            right_channel = channels.get("right_wrist_pose")
            left_valid = bool(left_channel is not None and left_channel.valid)
            right_valid = bool(right_channel is not None and right_channel.valid)
            validity_snapshot = (left_valid, right_valid)
            if capturing and stage_gate.should_emit("wrist_coarse_validity", validity_snapshot, now=loop_start):
                self._log(
                    "WristCoarse",
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
                        "WristCoarse",
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
                        "WristCoarse",
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
                    "WristCoarse",
                    f"LIVE time={elapsed:.1f}/{self.args.hold_capture_sec:.1f}s "
                    f"left={len(left_poses)} right={len(right_poses)} "
                    f"valid={{'left': {left_valid}, 'right': {right_valid}}}",
                    now=loop_start,
                )
            if capturing:
                elapsed = max(0.0, loop_start - capture_start_time)
                if loop_start >= capture_deadline:
                    capturing = False
                    left_hz = len(left_poses) / max(elapsed, 1e-6)
                    right_hz = len(right_poses) / max(elapsed, 1e-6)
                    if len(left_poses) >= self.args.min_hold_samples and len(right_poses) >= self.args.min_hold_samples:
                        self._log(
                            "WristCoarse",
                            f"AUTO-END ACCEPTED left={len(left_poses)} right={len(right_poses)} "
                            f"left_hz={left_hz:.1f} right_hz={right_hz:.1f}",
                            now=loop_start,
                        )
                        return left_poses, right_poses
                    self._log(
                        "WristCoarse",
                        f"AUTO-END INSUFFICIENT left={len(left_poses)} right={len(right_poses)} "
                        f"need>={self.args.min_hold_samples} left_hz={left_hz:.1f} right_hz={right_hz:.1f}",
                        now=loop_start,
                    )
                    left_poses = []
                    right_poses = []
                    left_min_reported = False
                    right_min_reported = False
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
                "WristCoarse",
                f"{side} tracker_offset captured translation="
                f"{np.array2string(np.asarray(translation, dtype=float), precision=3)} "
                f"canonical_rpy_deg={np.array2string(np.asarray(canonical_rpy_deg, dtype=float), precision=1)} "
                f"canonical_version={TRACKER_OFFSET_CALIBRATION_VERSION}",
            )

    def _run_tracker_frame_visualization(self) -> None:
        self._render_enabled = True
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
            "1/2=select left/right, j/l=yaw-/+, i/k=pitch+/-, u/o=roll-/+, x=reset selected fine tune, "
            f"v=cycle frame_viz_mode(current={self.frame_viz_mode}), "
            f"f=toggle XY sign flip(current_yaw={self.virtual_desk_post_yaw_deg:.0f}), "
            "s=save+exit, q=quit",
        )
        while True:
            loop_start = time.time()
            self._refresh_arm_render_state(now=loop_start)
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
                    if self.selected_side != "left":
                        self.selected_side = "left"
                        self._log("TrackerFrameViz", "selected_side=left", now=loop_start)
                elif event == "2":
                    if self.selected_side != "right":
                        self.selected_side = "right"
                        self._log("TrackerFrameViz", "selected_side=right", now=loop_start)
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
        left_robot_home_base = self.components.left_backend.compute_tcp_pose(self.args.tcp_mode)
        right_robot_home_base = self.components.right_backend.compute_tcp_pose(self.args.tcp_mode)
        left_robot_home_desk = self.components.calibration.get_arm_for_side("left").T_D_B @ left_robot_home_base
        right_robot_home_desk = self.components.calibration.get_arm_for_side("right").T_D_B @ right_robot_home_base

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
        left_qpos = self.components.left_backend.get_state()["current_qpos"]
        right_qpos = self.components.right_backend.get_state()["current_qpos"]
        if left_qpos is not None:
            self.components.calibration.capture_home_qpos("left_arm", left_qpos)
        if right_qpos is not None:
            self.components.calibration.capture_home_qpos("right_arm", right_qpos)
        self.components.calibration.metadata["home_capture"] = {
            "tcp_mode": self.args.tcp_mode,
            "left_human_home_virtual": left_human_home.tolist(),
            "right_human_home_virtual": right_human_home.tolist(),
            "left_robot_home_desk": left_robot_home_desk.tolist(),
            "right_robot_home_desk": right_robot_home_desk.tolist(),
        }
        self.home_ready = True
        self._log("HomeCapture", "home anchors updated")

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
        self.components.calibration.clear_home_state()
        self.components.calibration.metadata["profile_scope"] = "tracker_station_only"
        self.components.calibration.metadata["saved_at"] = datetime.now().isoformat(timespec="seconds")
        self.components.calibration.save_json(output_path)
        self._log("Save", f"calibration profile written to {output_path}")
        return output_path

    def _run_validation_step(self, channels: dict[str, object]) -> None:
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
            self.components.left_backend.send_command(left_command)
        if right_command is not None:
            target_pose_desk = right_command.metadata.get("target_pose_desk")
            if target_pose_desk is not None:
                target_poses_desk["right"] = as_pose_matrix(target_pose_desk)
            self.components.right_backend.send_command(right_command)
        self._last_validation_target_poses = target_poses_desk

    def _run_validation_loop(self) -> None:
        self._render_enabled = True
        self._log(
            "Validate",
            "Viewer is interactive. 1=select left, 2=select right, j/l=yaw-/+, i/k=pitch+/-, "
            "u/o=roll-/+, x=reset selected fine tune, h=recapture home, v=cycle frame_viz_mode, "
            f"f=toggle XY sign flip(current_yaw={self.virtual_desk_post_yaw_deg:.0f}), "
            "s=save+exit, q=quit",
        )
        self._log(
            "Validate",
            "Scene convention: +X=forward, +Y=left, +Z=up; left arm base is at -Y, right arm base is at +Y.",
        )
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
                    if self.selected_side != "left":
                        self.selected_side = "left"
                        self._log("Validate", "selected_side=left", now=loop_start)
                elif event == "2":
                    if self.selected_side != "right":
                        self.selected_side = "right"
                        self._log("Validate", "selected_side=right", now=loop_start)
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
    args = parse_args()
    components = build_components(args)
    app = CalibrationApp(args, components)
    app.run()


if __name__ == "__main__":
    main()
