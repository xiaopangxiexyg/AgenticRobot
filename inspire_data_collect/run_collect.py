from __future__ import annotations

import argparse
import re
import select
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from inspire_data_collect.arm.exo_joint_source import ExoJointSource, ExoJointSourceConfig
    from inspire_data_collect.arm.fr3_backend import FR3ArmBackend, FR3ArmBackendConfig
    from inspire_data_collect.arm.retargeter import FR3ArmRetargeter, FR3ArmRetargeterConfig
    from inspire_data_collect.async_writer import AsyncWriterConfig, AsyncWriterPool
    from inspire_data_collect.common import (
        ALL_SIDES,
        assert_inactive_side_config_empty,
        parse_active_sides,
    )
    from inspire_data_collect.hand.hand_backend import (
        InspireTacHandBackend,
        InspireTacHandBackendConfig,
    )
    from inspire_data_collect.hand.hand_mapping import (
        DEFAULT_INSPIRE_HAND_JOINT_ORDER,
        parse_tactile_regions,
    )
    from inspire_data_collect.hand.hand_source import HandQ6Source
    from inspire_data_collect.utils.realsense_rgb import (
        RealSenseRGBCameraConfig,
        RealSenseRGBSource,
        load_realsense_camera_config_map,
    )
    from inspire_data_collect.hand.retargeter import HandRetargeter, HandRetargeterConfig
    from inspire_data_collect.utils.tactile_visualizer import (
        TactileUVVisualizer,
        TactileVisualizerConfig,
        tactile_by_side_from_backends,
    )
    from inspire_data_collect.utils.writer import (
        ACTION_DIM,
        ARM_DOF,
        HAND_DOF,
        HAND_VECTOR_ORDER,
        HandOnlyLeRobotWriter,
        HandSampleBuilder,
        WriterConfig,
    )
else:
    from .arm.exo_joint_source import ExoJointSource, ExoJointSourceConfig
    from .arm.fr3_backend import FR3ArmBackend, FR3ArmBackendConfig
    from .arm.retargeter import FR3ArmRetargeter, FR3ArmRetargeterConfig
    from .async_writer import AsyncWriterConfig, AsyncWriterPool
    from .common import ALL_SIDES, assert_inactive_side_config_empty, parse_active_sides
    from .hand.hand_backend import InspireTacHandBackend, InspireTacHandBackendConfig
    from .hand.hand_mapping import DEFAULT_INSPIRE_HAND_JOINT_ORDER, parse_tactile_regions
    from .hand.hand_source import HandQ6Source
    from .utils.realsense_rgb import (
        RealSenseRGBCameraConfig,
        RealSenseRGBSource,
        load_realsense_camera_config_map,
    )
    from .hand.retargeter import HandRetargeter, HandRetargeterConfig
    from .utils.tactile_visualizer import (
        TactileUVVisualizer,
        TactileVisualizerConfig,
        tactile_by_side_from_backends,
    )
    from .utils.writer import (
        ACTION_DIM,
        ARM_DOF,
        HAND_DOF,
        HAND_VECTOR_ORDER,
        HandOnlyLeRobotWriter,
        HandSampleBuilder,
        WriterConfig,
    )


DEFAULT_CAMERA_CONFIG_PATH = Path(__file__).resolve().parent / "realsense.json"


def parse_joint_order(text: str) -> tuple[int, ...]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if len(values) != HAND_DOF:
        raise argparse.ArgumentTypeError("joint order must contain exactly 6 comma-separated ints")
    try:
        order = tuple(int(item) for item in values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("joint order must contain ints") from exc
    if sorted(order) != list(range(HAND_DOF)):
        raise argparse.ArgumentTypeError("joint order must be a permutation of 0,1,2,3,4,5")
    return order


class NonBlockingKeyboard:
    def __init__(self):
        self._fd: int | None = None
        self._old_settings: list[Any] | None = None

    def __enter__(self) -> "NonBlockingKeyboard":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def poll(self) -> str | None:
        if self._fd is None:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None
        return sys.stdin.read(1)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


class LoopRate:
    def __init__(self):
        self._last: float | None = None
        self.hz = 0.0

    def tick(self, now: float) -> float:
        if self._last is not None:
            dt = max(float(now - self._last), 1e-9)
            instant = 1.0 / dt
            self.hz = instant if self.hz <= 0.0 else 0.2 * instant + 0.8 * self.hz
        self._last = float(now)
        return float(self.hz)


def _sanitize_task_name_for_path(task_name: str) -> str:
    text = str(task_name).strip()
    if not text:
        return "task"
    text = re.sub(r"[\\/]+", "-", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("._-")
    return text or "task"


def _default_task_id_from_name(task_name: str) -> str:
    slug = _sanitize_task_name_for_path(task_name).lower()
    slug = re.sub(r"[^0-9a-z_-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "task"


def resolve_collection_layout(
    args: argparse.Namespace,
) -> tuple[Path, str, str, str, str]:
    task_name = str(getattr(args, "task_name", "")).strip()
    task_id = str(getattr(args, "task_id", "")).strip() or _default_task_id_from_name(task_name)
    task_description = str(getattr(args, "task_description", "")).strip() or task_name
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    dataset_dir_name = f"{date_prefix}_{_sanitize_task_name_for_path(task_name)}"
    dataset_root = Path(args.dataset_root) / dataset_dir_name
    dataset_name = str(getattr(args, "dataset_name", "")).strip() or dataset_dir_name
    return dataset_root, task_name, task_id, task_description, dataset_name


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Base output directory. Actual dataset path is <dataset-root>/<YYYY-MM-DD>_<task-name>.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Required task name used for dataset folder naming and meta.",
    )
    parser.add_argument("--task-id", type=str, default="")
    parser.add_argument("--task-description", type=str, default="")
    parser.add_argument("--dataset-name", type=str, default="")
    parser.add_argument("--active-sides", type=parse_active_sides, default=ALL_SIDES)

    parser.add_argument(
        "--enable-arm",
        action="store_true",
        help="Enable exoskeleton -> single-arm FR3 follow control and logging.",
    )
    parser.add_argument(
        "--arm-side",
        choices=["right", "left"],
        default="right",
        help="Single-arm label stored into dataset rows and metadata.",
    )
    parser.add_argument("--exo-port", type=str, default="")
    parser.add_argument("--exo-baudrate", type=int, default=57600)
    parser.add_argument("--fr3-host", type=str, default="192.168.20.172")
    parser.add_argument("--fr3-port", type=int, default=5558)
    parser.add_argument("--fr3-timeout-s", type=float, default=5.0)
    parser.add_argument("--fr3-state-poll-hz", type=float, default=20.0)
    parser.add_argument("--fr3-max-state-age-s", type=float, default=0.5)
    parser.add_argument("--fr3-state-ready-timeout-s", type=float, default=2.0)
    parser.add_argument("--arm-shrink-deg", type=float, default=10.0)
    parser.add_argument(
        "--disable-arm-startup-align",
        action="store_true",
        help="Disable the initial FR3 ramp from current qpos to the exoskeleton-mapped target.",
    )
    parser.add_argument(
        "--arm-startup-align-duration-s",
        type=float,
        default=5.0,
        help="Seconds used to interpolate FR3 from current qpos to the initial exoskeleton target.",
    )
    parser.add_argument(
        "--arm-startup-align-threshold-rad",
        type=float,
        default=0.02,
        help="Skip startup interpolation when max joint mismatch is below this threshold.",
    )

    parser.add_argument("--left-hand-port", type=str, default="")
    parser.add_argument("--right-hand-port", type=str, default="")
    parser.add_argument("--left-hand-ip", type=str, default="")
    parser.add_argument("--right-hand-ip", type=str, default="")
    parser.add_argument("--hand-modbus-port", type=int, default=6000)
    parser.add_argument("--hand-reader-baudrate", type=int, default=115200)
    parser.add_argument("--hand-reader-timeout", type=float, default=0.05)
    parser.add_argument("--show-hand-raw-lines", action="store_true")
    parser.add_argument("--allow-hand-connect-failure", action="store_true")

    parser.add_argument(
        "--hand-joint-order",
        type=parse_joint_order,
        default=DEFAULT_INSPIRE_HAND_JOINT_ORDER,
        help="Source q6 index permutation before sending to Inspire hand",
    )
    parser.add_argument("--hand-invert", dest="hand_invert", action="store_true", default=True)
    parser.add_argument("--hand-no-invert", dest="hand_invert", action="store_false")
    parser.add_argument("--hand-command-alpha", type=float, default=1.0)
    parser.add_argument("--hand-send-threshold", type=float, default=0.002)
    parser.add_argument("--hand-telemetry-hz", type=float, default=10.0)
    parser.add_argument("--hand-tactile-regions", type=str, default="all")
    parser.add_argument("--max-fingertip-force", type=int, default=1200)

    parser.add_argument("--hz", type=float, default=30.0, help="Main control loop rate")
    parser.add_argument("--minimum-control-hz", type=float, default=0.0)
    parser.add_argument("--status-sec", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until stopped")
    parser.add_argument("--record-auto-start", action="store_true")

    parser.add_argument(
        "--camera-mode",
        choices=["none", "single"],
        default="none",
        help="Camera mode. 'single' enables one RealSense RGB camera.",
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=DEFAULT_CAMERA_CONFIG_PATH,
        help="RealSense JSON config. Used when --camera-mode single and --camera-serial is empty.",
    )
    parser.add_argument("--camera-key", type=str, default="front")
    parser.add_argument("--camera-serial", type=str, default="")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-keep-last", type=int, default=4)
    parser.add_argument("--camera-color-space", choices=["rgb", "bgr"], default="rgb")
    parser.add_argument("--camera-wait-timeout-ms", type=int, default=200)
    parser.add_argument("--camera-auto-exposure-priority", type=int, default=None)

    parser.add_argument("--writer-workers", type=int, default=2)
    parser.add_argument("--writer-queue-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--parquet-compression", type=str, default="zstd")
    parser.add_argument("--video-codec", type=str, default="mp4v")
    parser.add_argument("--write-video", dest="write_video", action="store_true", default=True)
    parser.add_argument("--no-video", dest="write_video", action="store_false")
    parser.add_argument("--write-parquet", dest="write_parquet", action="store_true", default=True)
    parser.add_argument("--no-parquet", dest="write_parquet", action="store_false")

    parser.add_argument("--visualize-tactile", action="store_true")
    parser.add_argument("--tactile-vis-value-scale", type=float, default=0.05)
    parser.add_argument("--tactile-vis-size", type=int, default=640)
    parser.add_argument(
        "--tactile-vis-window-mode",
        choices=["resizable", "autosize"],
        default="resizable",
        help="'resizable' lets the OpenCV window be resized by dragging; 'autosize' keeps the rendered image size.",
    )


def validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    active_sides = tuple(args.active_sides)
    assert_inactive_side_config_empty(
        args,
        active_sides,
        suffixes=("hand_port", "hand_ip"),
    )
    if float(args.hz) <= 0.0:
        raise ValueError("--hz must be positive.")
    if float(args.hand_telemetry_hz) <= 0.0:
        raise ValueError("--hand-telemetry-hz must be positive.")
    if args.enable_arm:
        if int(args.exo_baudrate) <= 0:
            raise ValueError("--exo-baudrate must be positive.")
        if int(args.fr3_port) <= 0:
            raise ValueError("--fr3-port must be positive.")
        if float(args.fr3_timeout_s) <= 0.0:
            raise ValueError("--fr3-timeout-s must be positive.")
        if float(args.fr3_state_poll_hz) <= 0.0:
            raise ValueError("--fr3-state-poll-hz must be positive.")
        if float(args.fr3_max_state_age_s) <= 0.0:
            raise ValueError("--fr3-max-state-age-s must be positive.")
        if float(args.fr3_state_ready_timeout_s) <= 0.0:
            raise ValueError("--fr3-state-ready-timeout-s must be positive.")
        if float(args.arm_shrink_deg) < 0.0:
            raise ValueError("--arm-shrink-deg must be non-negative.")
        if float(args.arm_startup_align_duration_s) < 0.0:
            raise ValueError("--arm-startup-align-duration-s must be non-negative.")
        if float(args.arm_startup_align_threshold_rad) < 0.0:
            raise ValueError("--arm-startup-align-threshold-rad must be non-negative.")
    if args.camera_mode == "single":
        if not args.camera_serial and not Path(args.camera_config).exists():
            raise ValueError(
                "--camera-mode single requires --camera-serial or an existing --camera-config."
            )
        if int(args.camera_width) <= 0 or int(args.camera_height) <= 0:
            raise ValueError("--camera-width/--camera-height must be positive.")
        if int(args.camera_fps) <= 0:
            raise ValueError("--camera-fps must be positive.")
        if int(args.camera_keep_last) <= 0:
            raise ValueError("--camera-keep-last must be positive.")
    for side in active_sides:
        if not getattr(args, f"{side}_hand_port"):
            raise ValueError(f"--{side}-hand-port is required for active side '{side}'.")
        if args.visualize_tactile and not getattr(args, f"{side}_hand_ip"):
            raise ValueError(
                f"--{side}-hand-ip is required for tactile visualization on side '{side}'."
            )
    return active_sides


def build_components(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> tuple[HandQ6Source, dict[str, HandRetargeter], dict[str, InspireTacHandBackend]]:
    source = HandQ6Source.from_ports(
        left_port=args.left_hand_port if "left" in active_sides else None,
        right_port=args.right_hand_port if "right" in active_sides else None,
        baudrate=int(args.hand_reader_baudrate),
        timeout=float(args.hand_reader_timeout),
        show_raw_lines=bool(args.show_hand_raw_lines),
        critical=True,
    )

    retargeters: dict[str, HandRetargeter] = {}
    backends: dict[str, InspireTacHandBackend] = {}
    tactile_regions = parse_tactile_regions(args.hand_tactile_regions)
    for side in active_sides:
        effector_name = f"{side}_hand"
        retargeters[side] = HandRetargeter(
            HandRetargeterConfig(
                side=side,
                input_channel_name=f"{side}_hand_q6",
                effector_name=effector_name,
                joint_order=tuple(args.hand_joint_order),
                invert=bool(args.hand_invert),
                alpha=float(args.hand_command_alpha),
            )
        )
        hand_ip = str(getattr(args, f"{side}_hand_ip") or "")
        backends[side] = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip=hand_ip or "0.0.0.0",
                port=int(args.hand_modbus_port),
                side=side,
                effector_name=effector_name,
                connect_hardware=bool(hand_ip),
                max_fingertip_force=int(args.max_fingertip_force),
                send_threshold=float(args.hand_send_threshold),
                enable_telemetry=bool(hand_ip),
                telemetry_hz=float(args.hand_telemetry_hz),
                tactile_regions=tuple(tactile_regions),
            )
        )
    return source, retargeters, backends


def build_arm_components(
    args: argparse.Namespace,
) -> tuple[ExoJointSource | None, FR3ArmRetargeter | None, FR3ArmBackend | None]:
    if not args.enable_arm:
        return None, None, None
    source = ExoJointSource(
        ExoJointSourceConfig(
            side=str(args.arm_side),
            port=str(args.exo_port),
            baudrate=int(args.exo_baudrate),
        )
    )
    retargeter = FR3ArmRetargeter(
        FR3ArmRetargeterConfig(
            side=str(args.arm_side),
            input_channel_name="arm_exo_joint_qpos",
            effector_name="arm",
            shrink_deg=float(args.arm_shrink_deg),
        )
    )
    backend = FR3ArmBackend(
        FR3ArmBackendConfig(
            host=str(args.fr3_host),
            port=int(args.fr3_port),
            timeout_s=float(args.fr3_timeout_s),
            side=str(args.arm_side),
            effector_name="arm",
            state_poll_hz=float(args.fr3_state_poll_hz),
            max_state_age_s=float(args.fr3_max_state_age_s),
            wait_state_timeout_s=float(args.fr3_state_ready_timeout_s),
            startup_align_enabled=not bool(args.disable_arm_startup_align),
            startup_align_duration_s=float(args.arm_startup_align_duration_s),
            startup_align_threshold_rad=float(args.arm_startup_align_threshold_rad),
        )
    )
    return source, retargeter, backend


def build_camera_sources(args: argparse.Namespace) -> dict[str, RealSenseRGBSource]:
    if args.camera_mode == "none":
        return {}
    if args.camera_serial:
        config = RealSenseRGBCameraConfig(
            serial=str(args.camera_serial),
            width=int(args.camera_width),
            height=int(args.camera_height),
            fps=int(args.camera_fps),
            camera_key=str(args.camera_key),
            keep_last=int(args.camera_keep_last),
            color_space=str(args.camera_color_space),
            wait_timeout_ms=int(args.camera_wait_timeout_ms),
            auto_exposure_priority=args.camera_auto_exposure_priority,
            metadata={"source": "cli"},
        )
        return {config.camera_key: RealSenseRGBSource(config)}

    camera_configs = load_realsense_camera_config_map(Path(args.camera_config))
    if len(camera_configs) != 1:
        raise ValueError(
            f"--camera-mode single expects exactly one camera config, got {len(camera_configs)}."
        )
    return {
        camera_key: RealSenseRGBSource(config)
        for camera_key, config in camera_configs.items()
    }


def hand_state_snapshot(backends: dict[str, InspireTacHandBackend]) -> dict[str, dict[str, Any]]:
    return {side: backend.get_state() for side, backend in backends.items()}


def arm_state_snapshot(backend: FR3ArmBackend | None) -> dict[str, Any]:
    if backend is None:
        return {}
    return backend.get_state()


def collect_faults(
    *,
    active_sides: tuple[str, ...],
    control_hz: float,
    minimum_control_hz: float,
    hand_states: dict[str, dict[str, Any]],
    arm_state: dict[str, Any] | None = None,
) -> list[str]:
    faults: list[str] = []
    if minimum_control_hz > 0.0 and control_hz > 0.0 and control_hz < minimum_control_hz:
        faults.append(f"low_control_hz:{control_hz:.2f}")
    for side in active_sides:
        error = hand_states.get(side, {}).get("telemetry_error")
        if error:
            faults.append(f"{side}_hand_telemetry:{error}")
    if arm_state:
        error = arm_state.get("state_error")
        if error:
            faults.append(f"arm_state:{error}")
    return faults


def print_controls() -> None:
    print("\n" + "=" * 50)
    print("数据采集循环已启动 / Starting data collection loop...")
    print("控制说明 / Controls:")
    print("  'Z' = 开始录制 / Start episode")
    print("  'C' = 保存当前片段 / Save current episode")
    print("  'D' = 丢弃当前片段 / Discard current episode")
    print("  'Q' = 退出程序 / Quit program")
    print("=" * 50 + "\n")


def print_standby(next_episode_index: int) -> None:
    print("Standby, Ready to start data collection...")
    print(f"  下一个片段编号 Next episode ID: {next_episode_index}\n")
    print("=" * 100)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspire Tac collection with optional single-arm FR3 teleoperation."
    )
    add_args(parser)
    args = parser.parse_args(argv)
    active_sides = validate_args(args)
    dataset_root, task_name, task_id, task_description, dataset_name = resolve_collection_layout(args)
    camera_sources = build_camera_sources(args)

    writer = HandOnlyLeRobotWriter(
        WriterConfig(
            root_dir=dataset_root,
            chunk_size=int(args.chunk_size),
            nominal_fps=float(args.hz),
            parquet_compression=str(args.parquet_compression),
            video_codec=str(args.video_codec),
            task_name=str(task_name),
            task_id=str(task_id),
            task_description=str(task_description),
            dataset_name=str(dataset_name),
            write_parquet=bool(args.write_parquet),
            write_video=bool(args.write_video and camera_sources),
        )
    )
    writer_pool = AsyncWriterPool(
        writer.handle_job,
        config=AsyncWriterConfig(
            worker_count=int(args.writer_workers),
            max_queue_size=int(args.writer_queue_size),
        ),
        name="hand_only_writer",
    )
    sample_builder = HandSampleBuilder(
        active_sides=active_sides,
        episode_index=writer.next_episode_index,
        camera_buffers={
            camera_key: source.frame_buffer()
            for camera_key, source in camera_sources.items()
        },
    )
    source, retargeters, backends = build_components(args, active_sides)
    arm_source, arm_retargeter, arm_backend = build_arm_components(args)
    visualizer = (
        TactileUVVisualizer(
            TactileVisualizerConfig(
                enabled_sides=active_sides,
                image_size=(int(args.tactile_vis_size), int(args.tactile_vis_size)),
                value_scale=float(args.tactile_vis_value_scale),
                resizable_window=str(args.tactile_vis_window_mode) == "resizable",
            )
        )
        if args.visualize_tactile
        else None
    )

    current_episode: int | None = None
    recording = False
    loop_rate = LoopRate()
    step_count = 0

    def start_episode() -> None:
        nonlocal current_episode, recording
        if recording:
            return
        metadata = {
            "task_name": task_name,
            "task_id": task_id,
            "task_description": task_description,
            "active_sides": list(active_sides),
            "reserved_single_arm_dof": ARM_DOF,
            "action_dim": ACTION_DIM,
            "hand_dof_per_side": HAND_DOF,
            "hand_vector_order": list(HAND_VECTOR_ORDER),
            "joint_order": list(args.hand_joint_order),
            "hand_invert": bool(args.hand_invert),
            "tactile_regions": list(parse_tactile_regions(args.hand_tactile_regions)),
            "camera_mode": str(args.camera_mode),
            "camera_keys": sorted(camera_sources),
            "camera_config_path": None
            if args.camera_mode == "none" or args.camera_serial
            else str(args.camera_config),
            "write_video": bool(args.write_video and camera_sources),
            "arm_enabled": bool(args.enable_arm),
            "arm_side": str(args.arm_side),
            "arm_shrink_deg": float(args.arm_shrink_deg) if args.enable_arm else None,
            "arm_startup_align_enabled": bool(args.enable_arm)
            and not bool(args.disable_arm_startup_align),
            "arm_startup_align_duration_s": float(args.arm_startup_align_duration_s)
            if args.enable_arm
            else None,
            "arm_startup_align_threshold_rad": float(args.arm_startup_align_threshold_rad)
            if args.enable_arm
            else None,
            "exo_port": None if arm_source is None else arm_source.device_info().get("port"),
            "exo_baudrate": None if arm_source is None else arm_source.device_info().get("baudrate"),
            "fr3_host": None if arm_backend is None else arm_backend.device_info().get("host"),
            "fr3_port": None if arm_backend is None else arm_backend.device_info().get("port"),
        }
        current_episode = writer.start_episode(metadata=metadata)
        sample_builder.set_episode(current_episode)
        recording = True
        print(f"\n开始录制片段 Start recording episode: {current_episode}")

    def finish_episode(*, save: bool, show_standby: bool = True) -> None:
        nonlocal current_episode, recording
        if current_episode is None:
            recording = False
            return
        episode_index = int(current_episode)
        frame_count = int(sample_builder.frame_index)
        print(f"\n录制完成 Recording finished for episode {episode_index}")
        print(f"  数据帧数 Data frames: {frame_count}")
        writer_pool.join()
        if save and frame_count > 0:
            print(f"  正在保存片段 Saving episode {episode_index}...")
            for job in writer.plan_finalize_episode(
                episode_index,
                summary={
                    "frame_count": frame_count,
                    "active_sides": list(active_sides),
                    "camera_keys": sorted(camera_sources),
                },
            ):
                writer_pool.submit(job)
            writer_pool.join()
            print(f"  已保存片段 Episode saved {episode_index}")
        else:
            writer.discard_episode(episode_index)
            print(f"  丢弃片段 Discarding episode {episode_index}")
        current_episode = None
        recording = False
        if show_standby:
            print_standby(writer.next_episode_index)

    try:
        source.start()
        if arm_source is not None:
            arm_source.start()
            if not arm_source.is_ready():
                raise RuntimeError(f"arm exoskeleton source is not ready: {arm_source.device_info()}")
        for side, backend in backends.items():
            backend.start()
            if not backend.is_ready() and not args.allow_hand_connect_failure:
                raise RuntimeError(f"{side} hand backend is not ready: {backend.get_state()}")
        if arm_backend is not None:
            arm_backend.start()
            if not arm_backend.is_ready():
                raise RuntimeError(f"FR3 arm backend is not ready: {arm_backend.device_info()}")
        for camera_source in camera_sources.values():
            camera_source.start()

        writer_pool.start()
        if camera_sources:
            writer.write_device_config(
                "realsense",
                {
                    camera_key: source.device_info()
                    for camera_key, source in camera_sources.items()
                },
            )
        if arm_source is not None:
            writer.write_device_config("arm_exo", arm_source.device_info())
        if arm_backend is not None:
            writer.write_device_config("fr3", arm_backend.device_info())
        print(f"数据集目录 / Dataset directory: {dataset_root}")
        print(f"任务名称 / Task name: {task_name}")
        if args.enable_arm:
            print(
                "单臂 FR3 随动 / Single-arm FR3 follow: "
                f"side={args.arm_side}, host={args.fr3_host}:{args.fr3_port}, "
                f"startup_align={'off' if args.disable_arm_startup_align else str(args.arm_startup_align_duration_s) + 's'}"
            )
        print_controls()
        if args.record_auto_start:
            start_episode()
        else:
            print_standby(writer.next_episode_index)
        with NonBlockingKeyboard() as keyboard:
            while True:
                loop_started_at = time.time()
                key = keyboard.poll()
                key = None if key is None else key.lower()
                if key == "z":
                    start_episode()
                elif key == "c":
                    finish_episode(save=True)
                elif key == "d":
                    finish_episode(save=False)
                elif key == "q":
                    print("\n正在退出数据采集 Quitting the data collection...\n")
                    break

                channels = source.get_latest_channels()
                if arm_source is not None:
                    channels.update(arm_source.get_latest_channels())
                commands = {}
                if arm_retargeter is not None and arm_backend is not None:
                    arm_command = arm_retargeter.retarget(channels, timestamp=loop_started_at)
                    if arm_command is not None:
                        commands[arm_command.effector_name] = arm_command
                        arm_backend.send_command(arm_command)
                for side, retargeter in retargeters.items():
                    command = retargeter.retarget(channels, timestamp=loop_started_at)
                    if command is None:
                        continue
                    commands[command.effector_name] = command
                    backends[side].send_command(command)

                hand_states = hand_state_snapshot(backends)
                arm_state = arm_state_snapshot(arm_backend)
                control_hz = loop_rate.tick(loop_started_at)
                faults = collect_faults(
                    active_sides=active_sides,
                    control_hz=control_hz,
                    minimum_control_hz=float(args.minimum_control_hz),
                    hand_states=hand_states,
                    arm_state=arm_state,
                )

                if visualizer is not None:
                    visualizer.update(
                        tactile_by_side_from_backends(active_sides, backends.get)
                    )

                if recording and current_episode is not None:
                    sample = sample_builder.build(
                        timestamp=loop_started_at,
                        session_state="running",
                        control_hz=control_hz,
                        channels=channels,
                        commands=commands,
                        arm_state=arm_state,
                        hand_states=hand_states,
                        faults=faults,
                        metadata={"step_count": step_count},
                    )
                    for job in writer.plan_sample_jobs(sample):
                        writer_pool.submit(job)

                step_count += 1
                if int(args.max_steps) > 0 and step_count >= int(args.max_steps):
                    break
                sleep_time = (1.0 / float(args.hz)) - (time.time() - loop_started_at)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n\n检测到键盘中断，正在停止数据采集 Stopping data collection...\n")
    finally:
        try:
            if recording:
                finish_episode(save=True, show_standby=False)
        finally:
            if arm_backend is not None:
                arm_backend.close()
            if arm_source is not None:
                arm_source.close()
            for backend in backends.values():
                backend.close()
            for camera_source in camera_sources.values():
                camera_source.close()
            source.close()
            if visualizer is not None:
                visualizer.close()
            writer_pool.close(wait=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
