from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from inspire_data_collect.async_writer import AsyncWriterConfig, AsyncWriterPool
    from inspire_data_collect.common import (
        ALL_SIDES,
        assert_inactive_side_config_empty,
        parse_active_sides,
    )
    from inspire_data_collect.hand_backend import (
        InspireTacHandBackend,
        InspireTacHandBackendConfig,
    )
    from inspire_data_collect.hand_mapping import (
        DEFAULT_INSPIRE_HAND_JOINT_ORDER,
        parse_tactile_regions,
    )
    from inspire_data_collect.hand_source import HandQ6Source
    from inspire_data_collect.retargeter import HandRetargeter, HandRetargeterConfig
    from inspire_data_collect.tactile_visualizer import (
        TactileUVVisualizer,
        TactileVisualizerConfig,
        tactile_by_side_from_backends,
    )
    from inspire_data_collect.writer import (
        ACTION_DIM,
        ARM_DOF,
        HAND_DOF,
        HAND_VECTOR_ORDER,
        HandOnlyLeRobotWriter,
        HandSampleBuilder,
        WriterConfig,
    )
else:
    from .async_writer import AsyncWriterConfig, AsyncWriterPool
    from .common import ALL_SIDES, assert_inactive_side_config_empty, parse_active_sides
    from .hand_backend import InspireTacHandBackend, InspireTacHandBackendConfig
    from .hand_mapping import DEFAULT_INSPIRE_HAND_JOINT_ORDER, parse_tactile_regions
    from .hand_source import HandQ6Source
    from .retargeter import HandRetargeter, HandRetargeterConfig
    from .tactile_visualizer import (
        TactileUVVisualizer,
        TactileVisualizerConfig,
        tactile_by_side_from_backends,
    )
    from .writer import (
        ACTION_DIM,
        ARM_DOF,
        HAND_DOF,
        HAND_VECTOR_ORDER,
        HandOnlyLeRobotWriter,
        HandSampleBuilder,
        WriterConfig,
    )


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


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True, help="Output dataset root")
    parser.add_argument("--task-id", type=str, default="hand_only")
    parser.add_argument("--task-description", type=str, default="hand-only Inspire Tac collection")
    parser.add_argument("--dataset-name", type=str, default="")
    parser.add_argument("--active-sides", type=parse_active_sides, default=ALL_SIDES)

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
    parser.add_argument("--max-fingertip-force", type=int, default=800)

    parser.add_argument("--hz", type=float, default=30.0, help="Main control loop rate")
    parser.add_argument("--minimum-control-hz", type=float, default=0.0)
    parser.add_argument("--status-sec", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until stopped")
    parser.add_argument("--record-auto-start", action="store_true")

    parser.add_argument("--writer-workers", type=int, default=2)
    parser.add_argument("--writer-queue-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--parquet-compression", type=str, default="zstd")
    parser.add_argument("--write-parquet", dest="write_parquet", action="store_true", default=True)
    parser.add_argument("--no-parquet", dest="write_parquet", action="store_false")

    parser.add_argument("--visualize-tactile", action="store_true")
    parser.add_argument("--tactile-vis-value-scale", type=float, default=0.05)
    parser.add_argument("--tactile-vis-size", type=int, default=640)


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


def hand_state_snapshot(backends: dict[str, InspireTacHandBackend]) -> dict[str, dict[str, Any]]:
    return {side: backend.get_state() for side, backend in backends.items()}


def collect_faults(
    *,
    active_sides: tuple[str, ...],
    control_hz: float,
    minimum_control_hz: float,
    hand_states: dict[str, dict[str, Any]],
) -> list[str]:
    faults: list[str] = []
    if minimum_control_hz > 0.0 and control_hz > 0.0 and control_hz < minimum_control_hz:
        faults.append(f"low_control_hz:{control_hz:.2f}")
    for side in active_sides:
        error = hand_states.get(side, {}).get("telemetry_error")
        if error:
            faults.append(f"{side}_hand_telemetry:{error}")
    return faults


def format_tactile_sum(hand_state: dict[str, Any] | None) -> str:
    if not hand_state:
        return "-"
    tactile = hand_state.get("tactile")
    if tactile is None:
        return "-"
    values = np.asarray(tactile, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "-"
    return f"{float(np.sum(finite)):.3f}"


def print_status(
    *,
    recording: bool,
    episode_index: int | None,
    frame_index: int,
    control_hz: float,
    active_sides: tuple[str, ...],
    hand_states: dict[str, dict[str, Any]],
) -> None:
    tactile_text = " ".join(
        f"{side}_tac={format_tactile_sum(hand_states.get(side))}"
        for side in active_sides
    )
    episode_text = "-" if episode_index is None else str(episode_index)
    print(
        f"[status] rec={int(recording)} ep={episode_text} frame={frame_index} "
        f"hz={control_hz:.1f} {tactile_text}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hand-only Inspire Tac collection with tactile visualization."
    )
    add_args(parser)
    args = parser.parse_args(argv)
    active_sides = validate_args(args)

    writer = HandOnlyLeRobotWriter(
        WriterConfig(
            root_dir=args.dataset_root,
            chunk_size=int(args.chunk_size),
            nominal_fps=float(args.hz),
            parquet_compression=str(args.parquet_compression),
            task_id=str(args.task_id),
            task_description=str(args.task_description),
            dataset_name=str(args.dataset_name),
            write_parquet=bool(args.write_parquet),
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
    )
    source, retargeters, backends = build_components(args, active_sides)
    visualizer = (
        TactileUVVisualizer(
            TactileVisualizerConfig(
                enabled_sides=active_sides,
                image_size=(int(args.tactile_vis_size), int(args.tactile_vis_size)),
                value_scale=float(args.tactile_vis_value_scale),
            )
        )
        if args.visualize_tactile
        else None
    )

    current_episode: int | None = None
    recording = False
    loop_rate = LoopRate()
    last_status_at = 0.0
    step_count = 0

    def start_episode() -> None:
        nonlocal current_episode, recording
        if recording:
            return
        metadata = {
            "active_sides": list(active_sides),
            "reserved_single_arm_dof": ARM_DOF,
            "action_dim": ACTION_DIM,
            "hand_dof_per_side": HAND_DOF,
            "hand_vector_order": list(HAND_VECTOR_ORDER),
            "joint_order": list(args.hand_joint_order),
            "hand_invert": bool(args.hand_invert),
            "tactile_regions": list(parse_tactile_regions(args.hand_tactile_regions)),
        }
        current_episode = writer.start_episode(metadata=metadata)
        sample_builder.set_episode(current_episode)
        recording = True
        print(f"[record] start episode {current_episode}")

    def finish_episode(*, save: bool) -> None:
        nonlocal current_episode, recording
        if current_episode is None:
            recording = False
            return
        episode_index = int(current_episode)
        frame_count = int(sample_builder.frame_index)
        writer_pool.join()
        if save and frame_count > 0:
            for job in writer.plan_finalize_episode(
                episode_index,
                summary={"frame_count": frame_count, "active_sides": list(active_sides)},
            ):
                writer_pool.submit(job)
            writer_pool.join()
            print(f"[record] saved episode {episode_index} frames={frame_count}")
        else:
            writer.discard_episode(episode_index)
            print(f"[record] discarded episode {episode_index} frames={frame_count}")
        current_episode = None
        recording = False

    try:
        source.start()
        for side, backend in backends.items():
            backend.start()
            if not backend.is_ready() and not args.allow_hand_connect_failure:
                raise RuntimeError(f"{side} hand backend is not ready: {backend.get_state()}")

        writer_pool.start()
        if args.record_auto_start:
            start_episode()

        print(
            "Keys: z=start, c=save current episode, d=discard current episode, q=quit. "
            "The first 7 action/state slots are reserved for one arm and filled with NaN."
        )
        with NonBlockingKeyboard() as keyboard:
            while True:
                loop_started_at = time.time()
                key = keyboard.poll()
                if key == "z":
                    start_episode()
                elif key == "c":
                    finish_episode(save=True)
                elif key == "d":
                    finish_episode(save=False)
                elif key == "q":
                    break

                channels = source.get_latest_channels()
                commands = {}
                for side, retargeter in retargeters.items():
                    command = retargeter.retarget(channels, timestamp=loop_started_at)
                    if command is None:
                        continue
                    commands[command.effector_name] = command
                    backends[side].send_command(command)

                hand_states = hand_state_snapshot(backends)
                control_hz = loop_rate.tick(loop_started_at)
                faults = collect_faults(
                    active_sides=active_sides,
                    control_hz=control_hz,
                    minimum_control_hz=float(args.minimum_control_hz),
                    hand_states=hand_states,
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
                        hand_states=hand_states,
                        faults=faults,
                        metadata={"step_count": step_count},
                    )
                    for job in writer.plan_sample_jobs(sample):
                        writer_pool.submit(job)

                if loop_started_at - last_status_at >= float(args.status_sec):
                    print_status(
                        recording=recording,
                        episode_index=current_episode,
                        frame_index=sample_builder.frame_index,
                        control_hz=control_hz,
                        active_sides=active_sides,
                        hand_states=hand_states,
                    )
                    last_status_at = loop_started_at

                step_count += 1
                if int(args.max_steps) > 0 and step_count >= int(args.max_steps):
                    break
                sleep_time = (1.0 / float(args.hz)) - (time.time() - loop_started_at)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[stop] keyboard interrupt")
    finally:
        try:
            if recording:
                finish_episode(save=True)
        finally:
            for backend in backends.values():
                backend.close()
            source.close()
            if visualizer is not None:
                visualizer.close()
            writer_pool.close(wait=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
