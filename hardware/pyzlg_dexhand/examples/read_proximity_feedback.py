#!/usr/bin/env python3
"""Continuously print DexHand fingertip proximity values."""

import argparse
import time
from typing import Dict

from pyzlg_dexhand.dexhand_interface import (
    FeedbackMode,
    LeftDexHand,
    RightDexHand,
    StampedTouchFeedback,
    ZCANWrapper,
)
from pyzlg_dexhand.zcan import ZCANDeviceType


FINGER_ORDER = ("th", "ff", "mf", "rf", "lf")
DEVICE_TYPE_CHOICES = {
    "100U": ZCANDeviceType.ZCAN_USBCANFD_100U,
    "200U": ZCANDeviceType.ZCAN_USBCANFD_200U,
}


def build_hand(hand_name: str, zcan: ZCANWrapper):
    """Create a hand instance that shares an already-open ZCAN device."""
    hand_class = LeftDexHand if hand_name == "left" else RightDexHand
    return hand_class(zcan=zcan, auto_init=False)


def open_zcan(zcan: ZCANWrapper, device_index: int, device_type: str) -> str:
    """Open the ZCAN adapter and return the matched device type label."""
    if device_type in DEVICE_TYPE_CHOICES:
        selected_type = DEVICE_TYPE_CHOICES[device_type]
        if zcan.open(device_type=selected_type, device_index=device_index):
            return device_type
        raise RuntimeError(
            f"Failed to open ZCAN device as USBCANFD-{device_type}. "
            "Check USB connection, permissions, and whether another process is using it."
        )

    if device_type == "config":
        if zcan.open(device_index=device_index):
            return zcan.zcan_device_type.replace("ZCAN_USBCANFD_", "")
        raise RuntimeError(
            f"Failed to open ZCAN device using config type {zcan.zcan_device_type}."
        )

    configured_type = zcan.device_type
    candidates = [configured_type]
    for candidate in DEVICE_TYPE_CHOICES.values():
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if zcan.open(device_type=candidate, device_index=device_index):
            for label, enum_value in DEVICE_TYPE_CHOICES.items():
                if enum_value == candidate:
                    return label

    raise RuntimeError(
        "Failed to open ZCAN device as USBCANFD-100U or USBCANFD-200U. "
        "Check: 1) `lsusb | grep 3068` can see the adapter; "
        "2) `sudo ./tools/setup_usb_can.sh` has been run; "
        "3) the current user is in `canbus` after logout/login; "
        "4) no other DexHand/ROS process is using the adapter."
    )


def format_proximity_line(
    hand_name: str,
    touches: Dict[str, StampedTouchFeedback],
) -> str:
    """Format one proximity sample for all five fingertips."""
    values = []
    for finger in FINGER_ORDER:
        touch = touches.get(finger)
        value = "--" if touch is None else str(touch.proximity)
        values.append(f"{finger}={value}")
    return f"{time.time():.6f} {hand_name} proximity: " + " ".join(values)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Continuously print five-finger DexHand proximity readings."
    )
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
        default=20.0,
        help="Polling rate in Hz. Default: 20.",
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
    parser.add_argument(
        "--device-type",
        choices=["auto", "config", "100U", "200U"],
        default="auto",
        help="ZCAN adapter type. Default: auto.",
    )
    args = parser.parse_args()

    if args.rate <= 0:
        raise ValueError("--rate must be greater than 0")
    if args.period_ms < 0:
        raise ValueError("--period-ms must be non-negative")

    zcan = ZCANWrapper()
    opened_type = open_zcan(zcan, args.device_index, args.device_type)
    print(f"Opened ZCAN adapter as USBCANFD-{opened_type}", flush=True)

    hands = {}
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

        interval = 1.0 / args.rate
        start = time.time()

        while True:
            for hand_name, hand in hands.items():
                touches = hand.get_tactile_feedback()
                print(format_proximity_line(hand_name, touches), flush=True)

            if args.once:
                break
            if args.duration > 0 and time.time() - start >= args.duration:
                break
            time.sleep(interval)

    except KeyboardInterrupt:
        pass
    finally:
        zcan.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
