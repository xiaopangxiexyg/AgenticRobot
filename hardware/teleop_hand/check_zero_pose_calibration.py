#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import serial

from hand_q6_reader import auto_calibrate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat hand zero-pose calibration and save each result for stability checks."
    )
    parser.add_argument("--port", type=str, default="", help="Single-hand serial port.")
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument("--left-port", type=str, default="")
    parser.add_argument("--right-port", type=str, default="")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.05)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--stable-window", type=int, default=20)
    parser.add_argument("--stable-eps", type=float, default=0.5)
    parser.add_argument("--show-raw-lines", action="store_true")
    parser.add_argument(
        "--exo-hand-calibration",
        action="store_true",
        help="Also update the fixed exo hand calibration JSON. Default only saves per-run records.",
    )
    parser.add_argument(
        "--exo-hand-calibration-path",
        type=Path,
        default=None,
        help="Optional fixed exo hand calibration JSON path.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default is teleop_records/hand_zero_pose_calibrations.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Run rounds without waiting for Enter between checks.",
    )
    return parser.parse_args()


def _resolve_ports(args: argparse.Namespace) -> dict[str, str]:
    if args.port and (args.left_port or args.right_port):
        raise ValueError("Use either --port/--side or --left-port/--right-port, not both.")
    ports: dict[str, str] = {}
    if args.port:
        ports[str(args.side)] = str(args.port)
    if args.left_port:
        ports["left"] = str(args.left_port)
    if args.right_port:
        ports["right"] = str(args.right_port)
    if not ports:
        raise ValueError("Provide --port or at least one of --left-port/--right-port.")
    return ports


def _summarize(results: dict[str, list[np.ndarray]]) -> None:
    print("\n===== zero pose stability summary =====")
    for side, values in results.items():
        if not values:
            print(f"{side}: no successful calibration")
            continue
        stacked = np.stack(values, axis=0)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        span = np.ptp(stacked, axis=0)
        print(f"{side}: rounds={len(values)}")
        print(f"  mean_deg_at_0={np.round(mean, 3).tolist()}")
        print(f"  std_deg_at_0={np.round(std, 3).tolist()}")
        print(f"  span_deg_at_0={np.round(span, 3).tolist()}")
        print(f"  max_span_deg={float(np.max(span)):.3f}")


def main() -> int:
    args = parse_args()
    ports = _resolve_ports(args)
    if int(args.rounds) <= 0:
        raise ValueError("--rounds must be positive.")

    serials = {
        side: serial.Serial(
            port=port,
            baudrate=int(args.baudrate),
            timeout=float(args.timeout),
        )
        for side, port in ports.items()
    }
    results = {side: [] for side in ports}
    try:
        for round_index in range(1, int(args.rounds) + 1):
            print(f"\n===== round {round_index}/{args.rounds} =====")
            if not args.no_prompt:
                input("Place hand(s) at zero pose, keep still, then press Enter...")
            for side, ser in serials.items():
                calib = auto_calibrate(
                    ser=ser,
                    side=side,
                    port=ports[side],
                    baudrate=int(args.baudrate),
                    min_samples=int(args.min_samples),
                    max_samples=int(args.max_samples),
                    stable_window=int(args.stable_window),
                    stable_eps=float(args.stable_eps),
                    show_raw_lines=bool(args.show_raw_lines),
                    save_dir=args.save_dir,
                    update_fixed=bool(args.exo_hand_calibration),
                    calibration_path=args.exo_hand_calibration_path,
                )
                deg_at_0 = np.asarray(calib["deg_at_0"], dtype=float).reshape(6)
                results[side].append(deg_at_0)
                print(f"[Round {round_index}] {side} deg_at_0={np.round(deg_at_0, 3).tolist()}")
            time.sleep(0.2)
    finally:
        for ser in serials.values():
            try:
                ser.close()
            except Exception:
                pass
    _summarize(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
