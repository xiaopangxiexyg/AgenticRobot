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

from ..backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from ..calibration import TeleopCalibration
from ..configs import (
    ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
    build_annotated_dual_rm65_calibration,
)
from ..retargeters import PoseToArmRetargeter, PoseToArmRetargeterConfig
from ..schemas import TeleopObservation, latest_channel_timestamp, merge_teleop_commands
from ..sources.pose import VivePoseSource, ViveTrackerConfig
from ..transforms import as_pose_matrix, invert_pose
from .calibration_flow import (
    average_pose,
    compose_tracker_offset,
    euler_xyz_deg_to_matrix,
    fit_signed_axis_from_poses,
    fit_virtual_desk_frame,
    tracker_offset_from_neutral_pose,
)
from rm65_inspire_tac.paths import require_asset_path
from rm65_inspire_tac.pipeline import DEFAULT_LEFT_ARM_QPOS, DEFAULT_RIGHT_ARM_QPOS
from rm65_inspire_tac.rendering import SapienArmHandRenderer


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
            "captures left/right wrist tracker offsets, then validates the mapping "
            "with a SAPIEN dry-run teleop loop."
        )
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="UDP bind host for the VIVE receiver")
    parser.add_argument("--port", type=int, default=9999, help="UDP bind port for the VIVE receiver")
    parser.add_argument(
        "--tcp-mode",
        choices=["rm_eef", "hand_base"],
        default="rm_eef",
        help="TCP frame used for home capture and validation.",
    )
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
    parser.add_argument("--hz", type=_positive_float, default=30.0, help="Target update rate for capture and validation")
    parser.add_argument("--print-hz", type=_positive_float, default=2.0, help="Console print rate")
    parser.add_argument("--translation-scale", type=float, default=1.0, help="Translation scale used during validation teleop")
    parser.add_argument("--min-sweep-extent-m", type=_positive_float, default=0.15, help="Minimum extent required for each auxiliary sweep")
    parser.add_argument(
        "--min-axis-variance-ratio",
        type=float,
        default=0.85,
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
        help="Maximum allowed non-orthogonality between recovered virtual desk axes",
    )
    parser.add_argument(
        "--min-hold-samples",
        type=int,
        default=20,
        help="Minimum valid samples required for neutral and wrist hold captures",
    )
    parser.add_argument(
        "--home-samples",
        type=int,
        default=20,
        help="Minimum valid samples required per side for home capture",
    )
    parser.add_argument(
        "--fine-tune-step-deg",
        type=_positive_float,
        default=2.0,
        help="Fine tuning step for wrist orientation offsets during validation",
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
        default=np.zeros(3, dtype=float),
        help="Canonical left hand orientation in virtual desk XYZ Euler degrees",
    )
    parser.add_argument(
        "--right-hand-canonical-rpy-deg",
        type=_parse_xyz,
        default=np.zeros(3, dtype=float),
        help="Canonical right hand orientation in virtual desk XYZ Euler degrees",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="Optional calibration profile output path. Defaults to teleop_records/calibration_profiles/<timestamp>.json",
    )
    parser.add_argument("--max-steps", type=_non_negative_int, default=0, help="Optional limit for validation loop steps, 0 means infinite")
    return parser.parse_args()


def build_components(args: argparse.Namespace) -> CalibrationDebugComponents:
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
            arm_ip="0.0.0.0",
            side="left",
            default_tcp_frame=args.tcp_mode,
            arm_urdf_path=arm_urdf_path,
            connect_hardware=False,
            initial_qpos=tuple(DEFAULT_LEFT_ARM_QPOS.tolist()),
        )
    )
    right_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="right",
            default_tcp_frame=args.tcp_mode,
            arm_urdf_path=arm_urdf_path,
            connect_hardware=False,
            initial_qpos=tuple(DEFAULT_RIGHT_ARM_QPOS.tolist()),
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
        )
    )
    right_retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="right",
            input_channel_name="right_wrist_pose",
            effector_name="right_arm",
            control_mode="absolute",
            translation_scale=args.translation_scale,
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

    def run(self) -> None:
        if not self.key_input.enabled:
            raise RuntimeError("Calibration debug requires an interactive terminal (TTY).")
        try:
            self.components.left_backend.start()
            self.components.right_backend.start()
            self.components.source.start()
            self.key_input.start()
            self._run_link_check()
            self._run_virtual_desk_calibration()
            self._run_wrist_calibration("left")
            self._run_wrist_calibration("right")
            self._capture_home()
            self._run_validation_loop()
        finally:
            self.key_input.close()
            self.components.source.close()
            self.components.left_backend.close()
            self.components.right_backend.close()

    def _poll_channels(self) -> dict[str, object]:
        self.key_input.poll()
        return self.components.source.get_latest_channels()

    def _render(self) -> None:
        if self.components.renderer is None:
            return
        left_qpos = _state_qpos(self.components.left_backend, DEFAULT_LEFT_ARM_QPOS)
        right_qpos = _state_qpos(self.components.right_backend, DEFAULT_RIGHT_ARM_QPOS)
        self.components.renderer.render_data(
            left_qpos_arm=left_qpos,
            left_qpos_hand=self.empty_hand,
            right_qpos_arm=right_qpos,
            right_qpos_hand=self.empty_hand,
        )

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

    def _run_link_check(self) -> None:
        print("[LinkCheck] q=quit, n=continue when left/right/aux tracker channels are present.")
        next_print_time = 0.0
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
            if "n" in events and all(ready.values()):
                return
            if loop_start >= next_print_time:
                print(
                    "[LinkCheck] "
                    f"ready={ready} "
                    f"channels={sorted(channels)} "
                    "press 'n' to continue"
                )
                next_print_time = loop_start + (1.0 / self.args.print_hz)
            self._render()
            time.sleep(max(0.0, (1.0 / self.args.hz) - (time.time() - loop_start)))

    def _capture_single_channel(
        self,
        *,
        stage_name: str,
        instructions: str,
        channel_name: str,
        min_samples: int,
        transform,
        validator=None,
    ) -> list[np.ndarray]:
        print(f"[{stage_name}] {instructions}")
        print(f"[{stage_name}] c=start/stop capture, r=reset, q=quit")
        poses: list[np.ndarray] = []
        capturing = False
        next_print_time = 0.0
        while True:
            loop_start = time.time()
            channels = self._poll_channels()
            events = self.key_input.consume_events()
            if self.key_input.should_quit:
                raise KeyboardInterrupt
            if "r" in events:
                poses = []
                capturing = False
                print(f"[{stage_name}] reset")
            if "c" in events:
                if not capturing:
                    poses = []
                    capturing = True
                    print(f"[{stage_name}] capture started")
                else:
                    capturing = False
                    if len(poses) < min_samples:
                        print(f"[{stage_name}] only {len(poses)} samples, need at least {min_samples}")
                        poses = []
                    elif validator is None:
                        return poses
                    else:
                        error = validator(poses)
                        if error is None:
                            return poses
                        print(f"[{stage_name}] rejected: {error}")
                        poses = []

            channel = channels.get(channel_name)
            if capturing and channel is not None and channel.valid:
                poses.append(transform(channel))

            if loop_start >= next_print_time:
                status = "capturing" if capturing else "idle"
                channel_ok = channel is not None and channel.valid
                print(
                    f"[{stage_name}] status={status} "
                    f"samples={len(poses)} "
                    f"channel_valid={channel_ok}"
                )
                next_print_time = loop_start + (1.0 / self.args.print_hz)
            self._render()
            time.sleep(max(0.0, (1.0 / self.args.hz) - (time.time() - loop_start)))

    def _capture_dual_home_holds(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        print("[HomeCapture] move robot to session home and place both hands in the desired start pose.")
        print("[HomeCapture] c=start/stop capture, r=reset, q=quit")
        left_poses: list[np.ndarray] = []
        right_poses: list[np.ndarray] = []
        capturing = False
        next_print_time = 0.0
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
                print("[HomeCapture] reset")
            if "c" in events:
                if not capturing:
                    left_poses = []
                    right_poses = []
                    capturing = True
                    print("[HomeCapture] capture started")
                else:
                    capturing = False
                    if len(left_poses) >= self.args.home_samples and len(right_poses) >= self.args.home_samples:
                        return left_poses, right_poses
                    print(
                        "[HomeCapture] insufficient samples "
                        f"left={len(left_poses)} right={len(right_poses)} "
                        f"need>={self.args.home_samples}"
                    )
                    left_poses = []
                    right_poses = []

            left_channel = channels.get("left_wrist_pose")
            right_channel = channels.get("right_wrist_pose")
            if capturing and left_channel is not None and left_channel.valid:
                left_poses.append(self._channel_pose_virtual_calibrated(left_channel))
            if capturing and right_channel is not None and right_channel.valid:
                right_poses.append(self._channel_pose_virtual_calibrated(right_channel))

            if loop_start >= next_print_time:
                print(
                    "[HomeCapture] "
                    f"status={'capturing' if capturing else 'idle'} "
                    f"left_samples={len(left_poses)} right_samples={len(right_poses)}"
                )
                next_print_time = loop_start + (1.0 / self.args.print_hz)
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
                "Hold the auxiliary tracker and sweep mostly along the direction you want to map to robot +X "
                "(operator forward). Start near the negative side and finish near the positive side."
            ),
            channel_name="auxiliary_tracker_pose",
            min_samples=max(8, self.args.min_hold_samples),
            transform=lambda channel: as_pose_matrix(channel.value),
            validator=_sweep_validator,
        )
        y_sweep = self._capture_single_channel(
            stage_name="VirtualDeskY",
            instructions=(
                "Sweep the auxiliary tracker along the direction you want to map to robot +Y "
                "(operator left). Start near the negative side and finish near the positive side."
            ),
            channel_name="auxiliary_tracker_pose",
            min_samples=max(8, self.args.min_hold_samples),
            transform=lambda channel: as_pose_matrix(channel.value),
            validator=_sweep_validator,
        )
        z_sweep = self._capture_single_channel(
            stage_name="VirtualDeskZ",
            instructions=(
                "Move the auxiliary tracker upward and downward. Start low and finish high to define +Z."
            ),
            channel_name="auxiliary_tracker_pose",
            min_samples=max(8, self.args.min_hold_samples),
            transform=lambda channel: as_pose_matrix(channel.value),
            validator=_sweep_validator,
        )
        neutral_hold = self._capture_single_channel(
            stage_name="VirtualDeskNeutral",
            instructions=(
                "Hold the auxiliary tracker near your preferred teleop center. "
                "This sets the session origin of the virtual desk."
            ),
            channel_name="auxiliary_tracker_pose",
            min_samples=self.args.min_hold_samples,
            transform=lambda channel: as_pose_matrix(channel.value),
        )
        result = fit_virtual_desk_frame(
            x_sweep_poses=x_sweep,
            y_sweep_poses=y_sweep,
            z_sweep_poses=z_sweep,
            neutral_poses=neutral_hold,
        )
        max_non_orthogonality = max(
            result.xy_non_orthogonality_deg,
            result.xz_non_orthogonality_deg,
            result.yz_non_orthogonality_deg,
        )
        if max_non_orthogonality > self.args.max_axis_non_orthogonality_deg:
            raise RuntimeError(
                "Virtual desk fit failed orthogonality check: "
                f"{max_non_orthogonality:.2f}deg > {self.args.max_axis_non_orthogonality_deg:.2f}deg."
            )
        self.virtual_desk_result = result
        self.components.calibration.set_world_to_desk(result.T_W_V)
        self.components.calibration.metadata["virtual_desk"] = result.to_metadata()
        print(
            "[VirtualDesk] calibrated "
            f"x={np.array2string(result.x_axis_world, precision=3)} "
            f"y={np.array2string(result.y_axis_world, precision=3)} "
            f"z={np.array2string(result.z_axis_world, precision=3)}"
        )

    def _run_wrist_calibration(self, side: str) -> None:
        channel_name = f"{side}_wrist_pose"
        tracker_role = f"{side}_wrist"
        translation = (
            self.args.left_tracker_translation
            if side == "left"
            else self.args.right_tracker_translation
        )
        canonical_rpy_deg = (
            self.args.left_hand_canonical_rpy_deg
            if side == "left"
            else self.args.right_hand_canonical_rpy_deg
        )
        hand_rotation_virtual = euler_xyz_deg_to_matrix(canonical_rpy_deg)
        poses = self._capture_single_channel(
            stage_name=f"{side.title()}WristCoarse",
            instructions=(
                f"Place the {side} hand in the canonical neutral pose: forearm forward, "
                "hand orientation aligned with the virtual desk. Hold steady for capture."
            ),
            channel_name=channel_name,
            min_samples=self.args.min_hold_samples,
            transform=self._channel_pose_virtual_raw,
        )
        tracker_pose_virtual = average_pose(poses)
        coarse_offset = tracker_offset_from_neutral_pose(
            tracker_pose_virtual,
            hand_rotation_virtual=hand_rotation_virtual,
            translation_tracker_to_hand=translation,
        )
        self.coarse_tracker_offsets[side] = coarse_offset
        self.fine_tune_rpy_deg[side] = np.zeros(3, dtype=float)
        self.components.calibration.set_tracker_offset(tracker_role, coarse_offset)
        self.components.calibration.metadata.setdefault("tracker_offset_calibration", {})[tracker_role] = {
            "canonical_hand_rpy_deg": canonical_rpy_deg.tolist(),
            "translation_tracker_to_hand": np.asarray(translation, dtype=float).tolist(),
            "coarse_offset": coarse_offset.tolist(),
            "fine_tune_rpy_deg": [0.0, 0.0, 0.0],
            "final_offset": coarse_offset.tolist(),
            "captured_tracker_pose_virtual": tracker_pose_virtual.tolist(),
        }
        print(
            f"[{side.title()}WristCoarse] tracker_offset captured "
            f"translation={np.array2string(np.asarray(translation, dtype=float), precision=3)}"
        )

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
        print("[HomeCapture] home anchors updated")

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
        tracker_metadata[tracker_role]["fine_tune_rpy_deg"] = self.fine_tune_rpy_deg[side].tolist()
        tracker_metadata[tracker_role]["final_offset"] = offset.tolist()

    def _save_profile(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(self.args.output_path) if self.args.output_path else Path(
            "teleop_records/calibration_profiles"
        ) / f"{timestamp}_vive_rm65_virtual_desk.json"
        self.components.calibration.metadata["saved_at"] = datetime.now().isoformat(timespec="seconds")
        self.components.calibration.metadata["home_ready"] = bool(self.home_ready)
        self.components.calibration.save_json(output_path)
        print(f"[Save] calibration profile written to {output_path}")
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
        if left_command is not None:
            self.components.left_backend.send_command(left_command)
        if right_command is not None:
            self.components.right_backend.send_command(right_command)

    def _run_validation_loop(self) -> None:
        print(
            "[Validate] 1=select left, 2=select right, j/l=yaw-/+, i/k=pitch+/-, "
            "u/o=roll-/+, x=reset selected fine tune, h=recapture home, s=save+exit, q=quit"
        )
        next_print_time = 0.0
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
                    self.selected_side = "left"
                elif event == "2":
                    self.selected_side = "right"
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
                elif event == "s":
                    self._save_profile()
                    return

            if tuned:
                self._apply_side_fine_tune(self.selected_side)
                self._clear_home()
                print(
                    f"[Validate] updated {self.selected_side} fine tune to "
                    f"{np.array2string(self.fine_tune_rpy_deg[self.selected_side], precision=1)} deg. "
                    "Press 'h' to recapture home."
                )

            if self.home_ready:
                self._run_validation_step(channels)

            if loop_start >= next_print_time:
                print(
                    "[Validate] "
                    f"selected_side={self.selected_side} "
                    f"home_ready={self.home_ready} "
                    f"left_rpy_deg={np.array2string(self.fine_tune_rpy_deg['left'], precision=1)} "
                    f"right_rpy_deg={np.array2string(self.fine_tune_rpy_deg['right'], precision=1)} "
                    f"channels={sorted(channels)}"
                )
                next_print_time = loop_start + (1.0 / self.args.print_hz)

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
