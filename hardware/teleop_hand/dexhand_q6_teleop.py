#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_scalar_or_vector(text: str, name: str) -> np.ndarray:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(f"{name} must be a float or 6 comma-separated floats")

    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid {name}: {text}") from exc

    if len(values) == 1:
        return np.full(6, values[0], dtype=np.float32)
    if len(values) == 6:
        return np.asarray(values, dtype=np.float32)
    raise argparse.ArgumentTypeError(f"{name} must contain 1 or 6 floats, got {len(values)}")


def parse_joint_order(text: str) -> np.ndarray:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("joint order must contain 6 comma-separated indices")

    try:
        order = np.asarray([int(part) for part in parts], dtype=np.int64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid joint order: {text}") from exc

    if sorted(order.tolist()) != [0, 1, 2, 3, 4, 5]:
        raise argparse.ArgumentTypeError("joint order must be a permutation of 0,1,2,3,4,5")
    return order


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


def alpha_float(text: str) -> float:
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("alpha must be in (0, 1]")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read q6 values from hand_q6_reader.py and drive one DexHand from hand_driver.py."
        )
    )

    parser.add_argument("--port", type=str, default="", help="Single-hand serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--left-port", type=str, default="", help="Left-hand serial port")
    parser.add_argument("--right-port", type=str, default="", help="Right-hand serial port")
    parser.add_argument(
        "--side",
        choices=["left", "right"],
        default="right",
        help="Side label used in single-hand mode and default DexHand side.",
    )
    parser.add_argument(
        "--source-side",
        choices=["left", "right"],
        default="right",
        help="Which source hand drives the DexHand in dual-hand reader mode.",
    )
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate for q6 reader")
    parser.add_argument("--timeout", type=float, default=0.05, help="Serial read timeout in seconds")
    parser.add_argument("--show-raw-lines", action="store_true", help="Print raw serial lines from q6 reader")
    parser.add_argument(
        "--exo-hand-calibration",
        action="store_true",
        help="Re-run zero pose calibration and update exo_hand_zero_pose.json before teleop.",
    )
    parser.add_argument(
        "--exo-hand-calibration-path",
        type=str,
        default="",
        help="Optional fixed q6 reader calibration JSON path.",
    )

    parser.add_argument(
        "--hand-type",
        choices=["left", "right"],
        default="",
        help="DexHand side. Defaults to --side in single-hand mode or --source-side in dual-hand mode.",
    )
    parser.add_argument("--can-iface", type=str, default="can0", help="SocketCAN interface for DexHand")
    parser.add_argument("--can-baudrate", type=int, default=1000000, help="DexHand CAN bitrate")

    parser.add_argument(
        "--joint-order",
        type=parse_joint_order,
        default=np.arange(6, dtype=np.int64),
        help="Permutation applied to source q6 before sending, e.g. 1,0,2,3,4,5 swaps thumb joints.",
    )
    parser.add_argument(
        "--gain",
        type=lambda text: parse_scalar_or_vector(text, "gain"),
        default=np.ones(6, dtype=np.float32),
        help="Scalar or 6 comma-separated gains applied before clipping.",
    )
    parser.add_argument(
        "--bias",
        type=lambda text: parse_scalar_or_vector(text, "bias"),
        default=np.zeros(6, dtype=np.float32),
        help="Scalar or 6 comma-separated offsets applied before clipping.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert source q6 before gain/bias if the physical hand moves opposite to the reader.",
    )
    parser.add_argument("--alpha", type=alpha_float, default=1.0, help="EMA smoothing factor in (0, 1]")
    parser.add_argument(
        "--send-threshold",
        type=float,
        default=0.002,
        help="Only send when max joint delta exceeds this threshold.",
    )
    parser.add_argument("--hz", type=positive_float, default=100.0, help="Max command loop rate")
    parser.add_argument("--print-hz", type=positive_float, default=5.0, help="Console print rate")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N valid frames, 0 means infinite")
    parser.add_argument("--dry-run", action="store_true", help="Do not connect to DexHand; only print commands")
    parser.add_argument("--open-on-start", action="store_true", help="Open DexHand once before starting")
    parser.add_argument("--open-on-exit", action="store_true", help="Open DexHand once before exiting")
    return parser.parse_args()


def build_reader(args: argparse.Namespace) -> object:
    from hardware.teleop_hand.hand_q6_reader import BimanualHandQ6Reader, SingleHandQ6Reader

    calibration_path = args.exo_hand_calibration_path or None

    if args.port and (args.left_port or args.right_port):
        raise ValueError("Use either --port for single-hand mode, or --left-port/--right-port for dual-hand mode")

    if args.port:
        return SingleHandQ6Reader(
            port=args.port,
            side=args.side,
            baudrate=args.baudrate,
            timeout=args.timeout,
            raw_only=False,
            show_raw_lines=args.show_raw_lines,
            exo_hand_calibration=args.exo_hand_calibration,
            calibration_path=calibration_path,
        )

    if args.left_port or args.right_port:
        return BimanualHandQ6Reader(
            left_port=args.left_port or None,
            right_port=args.right_port or None,
            baudrate=args.baudrate,
            timeout=args.timeout,
            raw_only=False,
            show_raw_lines=args.show_raw_lines,
            exo_hand_calibration=args.exo_hand_calibration,
            calibration_path=calibration_path,
        )

    raise ValueError("Provide --port, or at least one of --left-port/--right-port")


def extract_source_norm(packet: Dict[str, object], source_side: str) -> Tuple[np.ndarray, str]:
    if "norm" in packet:
        return np.asarray(packet["norm"], dtype=np.float32), str(packet.get("side", source_side))

    if source_side not in packet:
        available = [key for key in ("left", "right") if key in packet]
        raise RuntimeError(
            f"Requested source side '{source_side}' is not available. Available sides: {available or 'none'}"
        )

    side_packet = packet[source_side]
    if not isinstance(side_packet, dict) or "norm" not in side_packet:
        raise RuntimeError(f"Packet for side '{source_side}' does not contain normalized q6 data")

    return np.asarray(side_packet["norm"], dtype=np.float32), source_side


class DexHandQ6Bridge:
    def __init__(
        self,
        *,
        hand: Optional[object],
        invert: bool = False,
        joint_order: Sequence[int] = (0, 1, 2, 3, 4, 5),
        gain: Sequence[float] = (1, 1, 1, 1, 1, 1),
        bias: Sequence[float] = (0, 0, 0, 0, 0, 0),
        alpha: float = 1.0,
        send_threshold: float = 0.002,
    ):
        self.hand = hand
        self.invert = bool(invert)
        self.joint_order = np.asarray(joint_order, dtype=np.int64)
        self.gain = np.asarray(gain, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.alpha = float(alpha)
        self.send_threshold = float(send_threshold)

        self.filtered_cmd: Optional[np.ndarray] = None
        self.last_sent_cmd: Optional[np.ndarray] = None

    def map_source_to_dexhand(self, source_norm: Sequence[float]) -> np.ndarray:
        source = np.asarray(source_norm, dtype=np.float32)
        if source.shape != (6,):
            raise ValueError(f"Expected source q6 shape (6,), got {source.shape}")

        source = np.clip(source, 0.0, 1.0)
        target = source[self.joint_order]
        if self.invert:
            target = 1.0 - target
        target = np.clip(target * self.gain + self.bias, 0.0, 1.0)

        if self.filtered_cmd is None or self.alpha >= 1.0:
            self.filtered_cmd = target
        else:
            self.filtered_cmd = self.alpha * target + (1.0 - self.alpha) * self.filtered_cmd
        return self.filtered_cmd.astype(np.float32)

    def maybe_send(self, target_cmd: Sequence[float]) -> bool:
        target = np.asarray(target_cmd, dtype=np.float32)
        if target.shape != (6,):
            raise ValueError(f"Expected target q6 shape (6,), got {target.shape}")

        if self.last_sent_cmd is not None:
            delta = float(np.max(np.abs(target - self.last_sent_cmd)))
            if delta < self.send_threshold:
                return False

        if self.hand is not None:
            self.hand.set_joint_pos(target.tolist())

        self.last_sent_cmd = target.copy()
        return True

    def send_open_pose(self) -> None:
        open_cmd = np.zeros(6, dtype=np.float32)
        if self.hand is not None:
            self.hand.set_joint_pos(open_cmd.tolist())
        self.filtered_cmd = open_cmd.copy()
        self.last_sent_cmd = open_cmd.copy()


def format_vec(vec: Sequence[float], precision: int = 3) -> str:
    return np.array2string(np.asarray(vec, dtype=np.float32), precision=precision, separator=", ")


def main() -> None:
    args = parse_args()
    reader = build_reader(args)
    from hardware.teleop_hand.hand_q6_reader import Q6_ORDER

    hand_type = args.hand_type or (args.side if args.port else args.source_side)
    if args.dry_run:
        hand = None
    else:
        from hardware.teleop_hand.hand_driver import DexHand

        hand = DexHand(
            hand_type=hand_type,
            can_iface=args.can_iface,
            baudrate=args.can_baudrate,
        )

    bridge = DexHandQ6Bridge(
        hand=hand,
        invert=args.invert,
        joint_order=args.joint_order,
        gain=args.gain,
        bias=args.bias,
        alpha=args.alpha,
        send_threshold=args.send_threshold,
    )

    print(f"[Config] q6_order={Q6_ORDER}")
    print(
        "[Config] "
        f"mode={'dry-run' if args.dry_run else 'hardware'} "
        f"source_side={args.source_side} "
        f"hand_type={hand_type} "
        f"can_iface={args.can_iface} "
        f"invert={args.invert} "
        f"joint_order={args.joint_order.tolist()} "
        f"gain={format_vec(args.gain)} "
        f"bias={format_vec(args.bias)}"
    )

    try:
        if args.open_on_start:
            print("[Init] sending open pose")
            bridge.send_open_pose()
            time.sleep(0.2)

        print("[Start] DexHand q6 teleop loop")
        next_print_t = 0.0
        loop_period = 1.0 / args.hz
        valid_frames = 0

        while True:
            loop_start_t = time.time()

            packet = reader.read_frame()
            source_norm, active_side = extract_source_norm(packet, source_side=args.source_side)
            target_cmd = bridge.map_source_to_dexhand(source_norm)
            sent = bridge.maybe_send(target_cmd)

            now = time.time()
            if now >= next_print_t:
                print(
                    f"[{active_side}->{hand_type}] "
                    f"source={format_vec(source_norm)} target={format_vec(target_cmd)} sent={sent}"
                )
                next_print_t = now + (1.0 / args.print_hz)

            valid_frames += 1
            if args.frames > 0 and valid_frames >= args.frames:
                break

            sleep_time = loop_period - (time.time() - loop_start_t)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Exit] stopped by user")
    finally:
        if args.open_on_exit:
            try:
                print("[Exit] sending open pose")
                bridge.send_open_pose()
                time.sleep(0.2)
            except Exception as exc:
                print(f"[Exit] failed to send open pose: {exc}")

        try:
            reader.close()
        except Exception:
            pass

        if hand is not None:
            try:
                hand.close_can()
            except Exception:
                pass


if __name__ == "__main__":
    main()
