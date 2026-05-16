#!/usr/bin/env python3
"""Diagnose and clear DexHand board error states."""

import argparse
import logging
import time
from typing import Dict, Iterable, Optional

from pyzlg_dexhand import ControlMode, LeftDexHand, RightDexHand, ZCANWrapper
from pyzlg_dexhand.zcan import ZCANDeviceType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEVICE_TYPE_CHOICES = {
    "100U": ZCANDeviceType.ZCAN_USBCANFD_100U,
    "200U": ZCANDeviceType.ZCAN_USBCANFD_200U,
}

BOARD_JOINTS = {
    0: ("th_dip", "th_mcp"),
    1: ("th_rot", "ff_spr"),
    2: ("ff_dip", "ff_mcp"),
    3: ("mf_dip", "mf_mcp"),
    4: ("rf_dip", "rf_mcp"),
    5: ("lf_dip", "lf_mcp"),
}

FINGER_OPEN_COMMANDS = {
    "thumb": {"th_rot": 0.0, "th_mcp": 0.0, "th_dip": 0.0},
    "index": {"ff_mcp": 0.0, "ff_dip": 0.0},
    "middle": {"mf_mcp": 0.0, "mf_dip": 0.0},
    "ring": {"rf_mcp": 0.0, "rf_dip": 0.0},
    "little": {"lf_mcp": 0.0, "lf_dip": 0.0},
    "spread": {"ff_spr": 0.0},
}

CONTROL_MODES = {
    "impedance_grasp": ControlMode.IMPEDANCE_GRASP,
    "cascaded_pid": ControlMode.CASCADED_PID,
    "protect_hall_position": ControlMode.PROTECT_HALL_POSITION,
    "hall_position": ControlMode.HALL_POSITION,
}


def open_zcan(zcan: ZCANWrapper, device_index: int, device_type: str) -> str:
    """Open the ZCAN adapter and return the matched device type label."""
    if device_type in DEVICE_TYPE_CHOICES:
        selected_type = DEVICE_TYPE_CHOICES[device_type]
        if zcan.open(device_type=selected_type, device_index=device_index):
            return device_type
        raise RuntimeError(f"Failed to open USBCANFD-{device_type}")

    if device_type == "config":
        if zcan.open(device_index=device_index):
            return zcan.zcan_device_type.replace("ZCAN_USBCANFD_", "")
        raise RuntimeError(f"Failed to open configured adapter type {zcan.zcan_device_type}")

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

    raise RuntimeError("Failed to open USBCANFD-100U or USBCANFD-200U")


def make_hand(hand_name: str, zcan: ZCANWrapper):
    hand_class = LeftDexHand if hand_name == "left" else RightDexHand
    return hand_class(zcan=zcan, auto_init=False)


def describe_errors(hand_name: str, hand) -> bool:
    """Refresh and print board/motor error information. Returns True if any error exists."""
    feedback = hand.get_feedback()
    errors = hand.get_errors()

    any_error = False
    print(f"\n[{time.strftime('%H:%M:%S')}] {hand_name} hand")
    for board_idx in range(6):
        error = errors.get(board_idx)
        joints = "/".join(BOARD_JOINTS[board_idx])
        if error is None:
            print(f"  board {board_idx} ({joints}): OK")
            continue

        any_error = True
        raw = error.error_data.hex() if error.error_data else "--"
        print(f"  board {board_idx} ({joints}): {error.description} raw={raw}")

    motor_error_found = False
    for joint_name, joint_feedback in feedback.joints.items():
        if joint_feedback.error_code:
            motor_error_found = True
            any_error = True
            print(
                f"  motor {joint_name}: error_code={joint_feedback.error_code} "
                f"angle={joint_feedback.angle:.2f} current={joint_feedback.current}"
            )

    if not any_error and not motor_error_found:
        print("  no latched board/motor errors observed")

    return any_error


def clear_errors(hand, repeats: int, settle_s: float) -> None:
    """Clear board errors multiple times with short waits."""
    for idx in range(repeats):
        ok = hand.clear_errors(clear_all=True, use_broadcast=False)
        print(f"  clear attempt {idx + 1}/{repeats}: {'ok' if ok else 'failed'}")
        time.sleep(settle_s)


def open_fingers(hand, fingers: Iterable[str], control_mode: ControlMode) -> None:
    """Send opening commands to selected fingers using reliable per-board commands."""
    command: Dict[str, float] = {}
    for finger in fingers:
        command.update(FINGER_OPEN_COMMANDS[finger])
    if not command:
        return
    print(f"  opening joints: {', '.join(sorted(command))}")
    hand.move_joints(
        **command,
        control_mode=control_mode,
        use_broadcast=False,
        clear_error=False,
        request_feedback=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose and clear DexHand board errors.")
    parser.add_argument("--hands", nargs="+", choices=["left", "right"], default=["left"])
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--device-type", choices=["auto", "config", "100U", "200U"], default="auto")
    parser.add_argument("--monitor", action="store_true", help="Keep printing errors until Ctrl-C.")
    parser.add_argument("--period", type=float, default=0.5, help="Monitor period in seconds.")
    parser.add_argument("--clear", action="store_true", help="Clear all board errors after diagnosis.")
    parser.add_argument("--clear-repeats", type=int, default=3, help="Clear command repetitions.")
    parser.add_argument("--settle", type=float, default=0.05, help="Seconds between clear attempts.")
    parser.add_argument(
        "--open-finger",
        nargs="+",
        choices=list(FINGER_OPEN_COMMANDS.keys()) + ["all"],
        default=[],
        help="Clear and then open selected finger(s).",
    )
    parser.add_argument("--mode", choices=list(CONTROL_MODES.keys()), default="impedance_grasp")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zcan = ZCANWrapper()
    opened_type = open_zcan(zcan, args.device_index, args.device_type)
    print(f"Opened ZCAN adapter as USBCANFD-{opened_type}")

    hands = {}
    try:
        for hand_name in args.hands:
            hand = make_hand(hand_name, zcan)
            if not hand.init(device_index=args.device_index):
                raise RuntimeError(f"Failed to initialize {hand_name} hand")
            hands[hand_name] = hand

        open_finger_args = args.open_finger
        if "all" in open_finger_args:
            open_finger_args = list(FINGER_OPEN_COMMANDS.keys())

        while True:
            for hand_name, hand in hands.items():
                describe_errors(hand_name, hand)
                if args.clear or open_finger_args:
                    clear_errors(hand, args.clear_repeats, args.settle)
                if open_finger_args:
                    open_fingers(hand, open_finger_args, CONTROL_MODES[args.mode])

            if not args.monitor:
                break
            time.sleep(args.period)

    except KeyboardInterrupt:
        pass
    finally:
        zcan.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
