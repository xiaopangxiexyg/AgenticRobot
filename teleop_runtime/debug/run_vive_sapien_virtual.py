from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass

import numpy as np

from ..backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from ..configs import (
    ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
    build_annotated_dual_rm65_calibration,
)
from ..retargeters import PoseToArmRetargeter, PoseToArmRetargeterConfig
from ..safety import TeleopSafetyConfig
from ..session import TeleopSession, TeleopSessionConfig
from ..sources.misc import KeyboardButtonSource
from ..sources.pose import VivePoseSource, ViveTrackerConfig
from ..transforms import as_pose_matrix
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


class TerminalClutchInput:
    def __init__(self, keyboard_source: KeyboardButtonSource):
        self.keyboard_source = keyboard_source
        self.enabled = sys.stdin.isatty()
        self._old_attrs: list[int] | None = None
        self.should_quit = False
        self.left_pressed = False
        self.right_pressed = False
        self._calibration_requested = False

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
                return
            if char in {"p", "\n", "\r"}:
                self._calibration_requested = True
                continue
            if char == "z":
                self.left_pressed = not self.left_pressed
            elif char == "x":
                self.right_pressed = not self.right_pressed
            elif char == "c":
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


@dataclass(frozen=True)
class VirtualRunComponents:
    session: TeleopSession
    left_backend: RM65ArmBackend
    right_backend: RM65ArmBackend
    renderer: SapienArmHandRenderer | None
    keyboard_source: KeyboardButtonSource
    left_retargeter: PoseToArmRetargeter
    right_retargeter: PoseToArmRetargeter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use real VIVE wrist trackers as input and visualize dual RM65 arm motion "
            "in SAPIEN without connecting real robot hardware."
        )
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="UDP bind host for the VIVE receiver")
    parser.add_argument("--port", type=int, default=9999, help="UDP bind port for the VIVE receiver")
    parser.add_argument(
        "--control-mode",
        choices=["absolute", "delta_clutch"],
        default="absolute",
        help="Virtual retargeting mode. Use delta_clutch with --keyboard-clutch for relative control.",
    )
    parser.add_argument(
        "--tcp-mode",
        choices=["rm_eef", "hand_base"],
        default="rm_eef",
        help="TCP frame passed into IK and visualization.",
    )
    parser.add_argument("--translation-scale", type=float, default=1.0, help="Scale applied to desk-frame translation")
    parser.add_argument("--hz", type=_positive_float, default=30.0, help="Target control/update rate")
    parser.add_argument("--print-hz", type=_positive_float, default=2.0, help="Console print rate")
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
    parser.add_argument("--max-steps", type=_non_negative_int, default=0, help="Stop after N control steps, 0 means infinite")
    return parser.parse_args()


def build_components(args: argparse.Namespace) -> VirtualRunComponents:
    calibration = build_annotated_dual_rm65_calibration(
        left_right_base_distance=args.left_right_base_distance
    )
    calibration.metadata.update(
        {
            "left_tracker_name": args.left_tracker_name,
            "right_tracker_name": args.right_tracker_name,
            "aux_tracker_name": args.aux_tracker_name,
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
        )
    )
    right_retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="right",
            input_channel_name="right_wrist_pose",
            effector_name="right_arm",
            control_mode=args.control_mode,
            translation_scale=args.translation_scale,
        )
    )

    session = TeleopSession(
        config=TeleopSessionConfig(
            mode="vive_sapien_virtual",
            tcp_mode=args.tcp_mode,
            record_enabled=not args.no_record,
            allow_hardware_after_debug_pass=False,
        ),
        sources=[
            VivePoseSource(
                tracker_configs=tracker_configs,
                host=args.host,
                port=args.port,
            ),
            keyboard_source,
        ],
        retargeters=[left_retargeter, right_retargeter],
        rig=None,
        calibration=calibration,
        safety_config=TeleopSafetyConfig(
            freeze_on_fault=True,
            required_channels=("left_wrist_pose", "right_wrist_pose"),
        ),
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
                "to validate the teleop loop without visualization."
            ) from exc
    return VirtualRunComponents(
        session=session,
        left_backend=left_backend,
        right_backend=right_backend,
        renderer=renderer,
        keyboard_source=keyboard_source,
        left_retargeter=left_retargeter,
        right_retargeter=right_retargeter,
    )


def _state_qpos(backend: RM65ArmBackend, fallback: np.ndarray) -> np.ndarray:
    state = backend.get_state()
    qpos = state["current_qpos"]
    if qpos is None:
        return fallback.copy()
    return np.asarray(qpos, dtype=float).copy()


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
    tcp_mode: str,
    timeout_sec: float,
    num_samples: int,
) -> None:
    calibration = components.session.calibration
    assert calibration is not None
    left_tracker = calibration.metadata.get("left_tracker_name")
    right_tracker = calibration.metadata.get("right_tracker_name")
    print(
        "[Calibration] expecting "
        f"left_wrist_pose<-{left_tracker or 'left_wrist role'}, "
        f"right_wrist_pose<-{right_tracker or 'right_wrist role'}"
    )
    print(
        "[Calibration] keep both wrist trackers steady in the desired home pose "
        f"for about {max(num_samples / 60.0, 0.5):.1f}s. "
        "This captures per-session home anchors; it does not yet solve a full desk/world extrinsic."
    )
    start_time = time.time()
    left_samples: list[np.ndarray] = []
    right_samples: list[np.ndarray] = []
    observed_channels: set[str] = set()

    while len(left_samples) < num_samples or len(right_samples) < num_samples:
        now = time.time()
        if now - start_time > timeout_sec:
            raise TimeoutError(
                "Timed out waiting for both wrist trackers during runtime calibration. "
                f"left_samples={len(left_samples)} right_samples={len(right_samples)}. "
                "Verify sender tracker_role values and receiver tracker-name arguments."
            )
        channels, _ = components.session.poll_sources(now)
        left_channel = channels.get("left_wrist_pose")
        if left_channel is not None:
            if left_channel.channel_name not in observed_channels:
                metadata = left_channel.metadata or {}
                print(
                    "[Calibration] observed "
                    f"{left_channel.channel_name} <- "
                    f"{metadata.get('tracker_name')}:{metadata.get('tracker_role')} "
                    f"frame={left_channel.frame_name}"
                )
                observed_channels.add(left_channel.channel_name)
            if left_channel.valid and len(left_samples) < num_samples:
                left_samples.append(_channel_pose_in_desk(left_channel, calibration))
        right_channel = channels.get("right_wrist_pose")
        if right_channel is not None:
            if right_channel.channel_name not in observed_channels:
                metadata = right_channel.metadata or {}
                print(
                    "[Calibration] observed "
                    f"{right_channel.channel_name} <- "
                    f"{metadata.get('tracker_name')}:{metadata.get('tracker_role')} "
                    f"frame={right_channel.frame_name}"
                )
                observed_channels.add(right_channel.channel_name)
            if right_channel.valid and len(right_samples) < num_samples:
                right_samples.append(_channel_pose_in_desk(right_channel, calibration))
        if len(left_samples) < num_samples or len(right_samples) < num_samples:
            time.sleep(0.01)

    left_human_home = _mean_pose(left_samples)
    right_human_home = _mean_pose(right_samples)
    left_robot_home_base = components.left_backend.compute_tcp_pose(tcp_mode)
    right_robot_home_base = components.right_backend.compute_tcp_pose(tcp_mode)
    left_robot_home_desk = calibration.get_arm_for_side("left").T_D_B @ left_robot_home_base
    right_robot_home_desk = calibration.get_arm_for_side("right").T_D_B @ right_robot_home_base

    components.left_retargeter.set_home_anchors(
        human_pose_desk=left_human_home,
        robot_pose_desk=left_robot_home_desk,
    )
    components.right_retargeter.set_home_anchors(
        human_pose_desk=right_human_home,
        robot_pose_desk=right_robot_home_desk,
    )
    calibration.capture_home_pose("left_arm", left_robot_home_desk, tcp_mode=tcp_mode)
    calibration.capture_home_pose("right_arm", right_robot_home_desk, tcp_mode=tcp_mode)
    print(
        "[Calibration] runtime home captured "
        f"left_samples={len(left_samples)} right_samples={len(right_samples)}"
    )


def _wait_for_calibration_start(
    components: VirtualRunComponents,
    clutch_input: TerminalClutchInput,
) -> bool:
    if not clutch_input.enabled:
        print(
            "[Calibration] stdin is not interactive; starting automatically. "
            "Use --auto-calibrate explicitly in non-interactive runs."
        )
        return True

    print(
        "[Calibration] position robot and trackers first, then press 'p' or Enter to start home calibration. "
        "Press 'q' to quit."
    )
    empty_hand = np.empty(0, dtype=float)
    next_print_time = 0.0
    while True:
        loop_start = time.time()
        clutch_input.poll()
        if clutch_input.should_quit:
            return False
        if clutch_input.consume_calibration_request():
            return True

        if components.renderer is not None:
            left_qpos = _state_qpos(components.left_backend, DEFAULT_LEFT_ARM_QPOS)
            right_qpos = _state_qpos(components.right_backend, DEFAULT_RIGHT_ARM_QPOS)
            components.renderer.render_data(
                left_qpos_arm=left_qpos,
                left_qpos_hand=empty_hand,
                right_qpos_arm=right_qpos,
                right_qpos_hand=empty_hand,
            )

        if loop_start >= next_print_time:
            channels, _ = components.session.poll_sources(loop_start)
            print(
                "[Calibration] waiting_for_key "
                f"channels={sorted(channels)} "
                "ready_key='p/Enter'"
            )
            next_print_time = loop_start + 1.0
        time.sleep(0.02)


def main() -> None:
    args = parse_args()
    components = build_components(args)
    clutch_input = TerminalClutchInput(components.keyboard_source)
    empty_hand = np.empty(0, dtype=float)
    try:
        components.left_backend.start()
        components.right_backend.start()
        components.session.start()
        clutch_input.start()
        if not args.auto_calibrate:
            should_continue = _wait_for_calibration_start(components, clutch_input)
            if not should_continue:
                return
        _runtime_home_calibration(
            components,
            tcp_mode=args.tcp_mode,
            timeout_sec=args.calibration_timeout,
            num_samples=args.calibration_samples,
        )
        components.session.set_calibration_ready()
        components.session.mark_debug_passed()
        components.session.arm_hardware()

        if args.keyboard_clutch:
            print("[Keys] p/Enter=start calibration, z=left clutch, x=right clutch, c=both, r=release, q=quit")
        elif args.control_mode == "delta_clutch":
            print("[Info] delta_clutch is running with both clutch buttons held true by default.")

        next_print_time = 0.0
        step_index = 0
        while True:
            loop_start = time.time()
            clutch_input.poll()
            if clutch_input.should_quit:
                break

            step_result = components.session.run_step(now=loop_start)
            if step_result.command is not None:
                left_command = step_result.command.commands.get("left_arm")
                if left_command is not None:
                    components.left_backend.send_command(left_command)
                right_command = step_result.command.commands.get("right_arm")
                if right_command is not None:
                    components.right_backend.send_command(right_command)

            left_qpos = _state_qpos(components.left_backend, DEFAULT_LEFT_ARM_QPOS)
            right_qpos = _state_qpos(components.right_backend, DEFAULT_RIGHT_ARM_QPOS)
            if components.renderer is not None:
                components.renderer.render_data(
                    left_qpos_arm=left_qpos,
                    left_qpos_hand=empty_hand,
                    right_qpos_arm=right_qpos,
                    right_qpos_hand=empty_hand,
                )

            if loop_start >= next_print_time:
                fault_codes = [
                    f"{fault.code}:{fault.channel_name}" if fault.channel_name else fault.code
                    for fault in step_result.faults
                ]
                print(
                    "[VirtualRun] "
                    f"step={step_index} hz={step_result.control_hz:.1f} "
                    f"state={step_result.state.value} "
                    f"channels={sorted(step_result.channels)} "
                    f"left_clutch={components.keyboard_source.get_latest_channels().get('left_clutch_button').value} "
                    f"right_clutch={components.keyboard_source.get_latest_channels().get('right_clutch_button').value} "
                    f"faults={fault_codes}"
                )
                next_print_time = loop_start + (1.0 / args.print_hz)

            step_index += 1
            if args.max_steps > 0 and step_index >= args.max_steps:
                break

            remaining = (1.0 / args.hz) - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        clutch_input.close()
        components.session.close()
        components.left_backend.close()
        components.right_backend.close()


if __name__ == "__main__":
    main()
