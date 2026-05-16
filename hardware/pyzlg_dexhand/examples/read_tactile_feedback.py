#!/usr/bin/env python3
"""Read DexHand fingertip tactile feedback from one or both hands."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from pyzlg_dexhand.dexhand_interface import (
    FeedbackMode,
    LeftDexHand,
    RightDexHand,
    StampedTouchFeedback,
    ZCANWrapper,
)


FINGER_ORDER = ("th", "ff", "mf", "rf", "lf")


def normalize_direction(direction: int) -> int:
    """Return -1 for invalid force direction readings."""
    if direction == 65535 or direction > 359:
        return -1
    return direction


def touch_to_dict(touch: StampedTouchFeedback) -> Dict[str, Any]:
    """Convert a StampedTouchFeedback dataclass to a JSON-friendly dict."""
    data = asdict(touch)
    data["timestamp"] = data["timestamp"] / 1e9
    data["direction"] = normalize_direction(data["direction"])
    return data


def print_touch_summary(hand_name: str, touches: Dict[str, StampedTouchFeedback]) -> None:
    """Print one compact human-readable sample."""
    parts = []
    for finger in FINGER_ORDER:
        touch = touches.get(finger)
        if touch is None:
            parts.append(f"{finger}: --")
            continue

        direction = normalize_direction(touch.direction)
        parts.append(
            f"{finger}: Fn={touch.normal_force:.3f}N "
            f"Ft={touch.tangential_force:.3f}N "
            f"dFn={touch.normal_force_delta} "
            f"dFt={touch.tangential_force_delta} "
            f"dir={direction} "
            f"prox={touch.proximity} "
            f"temp={touch.temperature}C"
        )

    print(f"{time.time():.6f} {hand_name}: " + " | ".join(parts), flush=True)


def build_hand(hand_name: str, zcan: ZCANWrapper):
    """Create a hand instance that shares an already-open ZCAN device."""
    hand_class = LeftDexHand if hand_name == "left" else RightDexHand
    return hand_class(zcan=zcan, auto_init=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read DexHand fingertip tactile feedback.")
    parser.add_argument(
        "--hands",
        nargs="+",
        choices=["left", "right"],
        default=["left"],
        help="Hands to read. Default: left.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="Polling rate in Hz. Default: 50.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to run. 0 means run until Ctrl-C. Default: 0.",
    )
    parser.add_argument(
        "--period-ms",
        type=int,
        default=20,
        help="Periodic feedback interval to request from each board. Default: 20 ms.",
    )
    parser.add_argument(
        "--no-configure-feedback",
        action="store_true",
        help="Only poll existing feedback frames; do not send feedback-mode commands.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Optional path for JSONL samples.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Read and print one sample per selected hand.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="ZCAN device index. Default: 0.",
    )
    args = parser.parse_args()

    if args.rate <= 0:
        raise ValueError("--rate must be greater than 0")
    if args.period_ms < 0:
        raise ValueError("--period-ms must be non-negative")

    zcan = ZCANWrapper()
    if not zcan.open(device_index=args.device_index):
        raise RuntimeError("Failed to open ZCAN device")

    hands = {}
    jsonl_file: Optional[object] = None
    try:
        for hand_name in args.hands:
            hand = build_hand(hand_name, zcan)
            if not hand.init(device_index=args.device_index):
                raise RuntimeError(f"Failed to initialize {hand_name} hand")
            hands[hand_name] = hand

        if not args.no_configure_feedback:
            for hand in hands.values():
                hand.set_feedback_mode(
                    FeedbackMode.PERIODIC,
                    period_ms=args.period_ms,
                    enable=True,
                )
            time.sleep(max(args.period_ms / 1000.0 * 2.0, 0.05))

        if args.jsonl is not None:
            args.jsonl.parent.mkdir(parents=True, exist_ok=True)
            jsonl_file = args.jsonl.open("a", encoding="utf-8")

        interval = 1.0 / args.rate
        start = time.time()

        while True:
            for hand_name, hand in hands.items():
                touches = hand.get_tactile_feedback()
                print_touch_summary(hand_name, touches)

                if jsonl_file is not None:
                    record = {
                        "timestamp": time.time(),
                        "hand": hand_name,
                        "touch": {
                            finger: touch_to_dict(touch)
                            for finger, touch in touches.items()
                        },
                    }
                    jsonl_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                    jsonl_file.flush()

            if args.once:
                break
            if args.duration > 0 and time.time() - start >= args.duration:
                break
            time.sleep(interval)

    except KeyboardInterrupt:
        pass
    finally:
        if jsonl_file is not None:
            jsonl_file.close()
        zcan.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
