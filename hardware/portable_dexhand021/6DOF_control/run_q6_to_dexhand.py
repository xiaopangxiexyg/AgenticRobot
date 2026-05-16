#!/usr/bin/env python3
"""Drive DexHand from a normalized 6D hand command."""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portable_dexhand021 import DexHandController
from portable_dexhand021.tactile_viewer import run_viewer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from q6_to_dexhand import DEFAULT_FINGER_PROFILE_PATH, DexHandQ6Mapper, parse_float_vector


ANGLE_RE = re.compile(r"=\s*([-+]?\d+(?:\.\d+)?)")
TELEOP_OFFSET = (40.0, -70.0, 60.0, 60.0, 60.0, 60.0)
DEFAULT_CALIBRATION_PATH = ROOT / "teleop_hand" / "exo_hand_zero_pose.json"


def parse_q6(text: str) -> Tuple[float, ...]:
    return parse_float_vector(text, 6, "q6")


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


def nonnegative_float(text: str) -> float:
    value = float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map 6D hand q6 commands to DexHand 12-DoF joint commands.")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--q6", type=parse_q6, help="One normalized q6 command, e.g. 0,0,0.5,0.5,0.5,0.5")
    source.add_argument("--port", type=str, help="Serial port for teleop_hand-style ANGLES frames")

    parser.add_argument("--side", choices=["left", "right"], default="right", help="Calibration side for serial input")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.05)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--show-raw-lines", action="store_true")
    parser.add_argument(
        "--zero-on-start",
        dest="zero_on_start",
        action="store_true",
        help="Use the current open hand pose as q6 zero when --port is used. Default is enabled.",
    )
    parser.add_argument(
        "--no-zero-on-start",
        dest="zero_on_start",
        action="store_false",
        help="Use the fixed calibration JSON instead of runtime zero calibration.",
    )
    parser.set_defaults(zero_on_start=True)
    parser.add_argument("--zero-max-samples", type=int, default=200, help="Max valid ANGLES frames for runtime zero calibration")
    parser.add_argument("--zero-min-samples", type=int, default=30, help="Min valid ANGLES frames before accepting a stable zero")
    parser.add_argument("--zero-stable-window", type=int, default=20, help="Recent valid frames used to test zero stability")
    parser.add_argument("--zero-stable-eps", type=float, default=0.5, help="Required max range in degrees for stable runtime zero")
    parser.add_argument(
        "--zero-scale",
        choices=["absolute", "signed"],
        default="absolute",
        help=(
            "Runtime zero scaling. 'absolute' treats motion away from the open zero as closing, "
            "which avoids wrong side/sign keeping q6 clipped at 0. 'signed' matches teleop_hand's side offset exactly."
        ),
    )

    parser.add_argument("--source-order", choices=["auto", "internal", "teleop"], default="auto")
    parser.add_argument("--invert", action="store_true", help="Use when source q6 uses 1=open, 0=closed")
    parser.add_argument("--gain", type=lambda text: parse_float_vector(text, 6, "gain"), default=(1.0,) * 6)
    parser.add_argument("--bias", type=lambda text: parse_float_vector(text, 6, "bias"), default=(0.0,) * 6)
    parser.add_argument("--max-deg", type=lambda text: parse_float_vector(text, 6, "max-deg"), default=(90.0, 150.0, 90.0, 90.0, 90.0, 90.0))
    parser.add_argument("--spread-angle", type=float, default=0.0, help="Fixed ff_spr angle in degrees")
    parser.add_argument(
        "--four-finger-mode",
        choices=["same", "profile"],
        default="profile",
        help=(
            "Four-finger DIP/MCP coupling. 'same' sends equal angles. "
            "'profile' maps DIP linearly and MCP by the saved profile's MCP/DIP ratio."
        ),
    )
    parser.add_argument(
        "--four-finger-profile",
        type=Path,
        default=DEFAULT_FINGER_PROFILE_PATH,
        help="Joint-angle JSON used by --four-finger-mode profile.",
    )
    parser.add_argument("--alpha", type=alpha_float, default=1.0)
    parser.add_argument("--send-threshold", type=float, default=0.002)

    parser.add_argument("--hand", choices=["left", "right"], default="left")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--device-type", choices=["auto", "config", "100U", "200U"], default="auto")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--visualize-tactile", action="store_true", help="Open tactile viewer while controlling, sharing the same DexHand connection")
    parser.add_argument("--state-print-hz", type=nonnegative_float, default=0.0, help="Print compact DexHand state at this rate. 0 disables state printing.")
    parser.add_argument("--tactile-viewer-hz", type=positive_float, default=20.0)
    parser.add_argument("--hz", type=positive_float, default=50.0)
    parser.add_argument("--print-hz", type=positive_float, default=5.0)
    parser.add_argument("--frames", type=int, default=0, help="0 means run until Ctrl-C for serial input")
    parser.add_argument(
        "--clear-errors-on-start",
        dest="clear_errors_on_start",
        action="store_true",
        help="Clear DexHand board errors once after connecting. Default is enabled.",
    )
    parser.add_argument(
        "--no-clear-errors-on-start",
        dest="clear_errors_on_start",
        action="store_false",
        help="Do not clear DexHand errors after connecting.",
    )
    parser.set_defaults(clear_errors_on_start=True)
    parser.add_argument(
        "--error-check-hz",
        type=nonnegative_float,
        default=20.0,
        help="Refresh feedback and clear detected DexHand board errors at this rate. 0 disables runtime checks.",
    )
    return parser.parse_args()


def parse_angles_line(line: str) -> Optional[Tuple[float, ...]]:
    if "ANGLES:" not in line:
        return None
    matches = ANGLE_RE.findall(line)
    if len(matches) != 6:
        return None
    return tuple(float(item) for item in matches)


def load_calibration(path: Path, side: str) -> Dict[str, Tuple[float, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    item = payload["calibrations"][side]
    deg_at_0 = tuple(float(v) for v in item["deg_at_0"])
    if "deg_at_1" in item:
        deg_at_1 = tuple(float(v) for v in item["deg_at_1"])
    else:
        sign = -1.0 if side == "left" else 1.0
        deg_at_1 = tuple(deg_at_0[i] + sign * TELEOP_OFFSET[i] for i in range(6))
    return {"deg_at_0": deg_at_0, "deg_at_1": deg_at_1, "absolute_delta": False}


def offset_for_side(side: str) -> Tuple[float, ...]:
    sign = -1.0 if side == "left" else 1.0
    return tuple(sign * value for value in TELEOP_OFFSET)


def angles_to_norm(angles_deg: Sequence[float], calib: Dict[str, Tuple[float, ...]]) -> Tuple[float, ...]:
    out = []
    absolute_delta = bool(calib.get("absolute_delta", False))
    for angle, deg0, deg1 in zip(angles_deg, calib["deg_at_0"], calib["deg_at_1"]):
        denom = deg1 - deg0
        if abs(denom) < 1e-6:
            out.append(0.0)
        elif absolute_delta:
            out.append(min(max(abs(float(angle) - deg0) / abs(denom), 0.0), 1.0))
        else:
            out.append(min(max((float(angle) - deg0) / denom, 0.0), 1.0))
    return tuple(out)


class SerialQ6Reader:
    """Small read-only clone of teleop_hand hand_q6_reader behavior."""

    def __init__(
        self,
        *,
        port: str,
        side: str,
        baudrate: int,
        timeout: float,
        calibration: Path,
        show_raw_lines: bool,
        zero_on_start: bool,
        zero_max_samples: int,
        zero_min_samples: int,
        zero_stable_window: int,
        zero_stable_eps: float,
        zero_scale: str,
    ) -> None:
        import serial

        self.port = port
        self.side = side
        self.show_raw_lines = show_raw_lines
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        if zero_on_start:
            self.calib = self._calibrate_zero_on_start(
                max_samples=zero_max_samples,
                min_samples=zero_min_samples,
                stable_window=zero_stable_window,
                stable_eps=zero_stable_eps,
                zero_scale=zero_scale,
            )
        else:
            self.calib = load_calibration(calibration, side)

    def _read_angles(self) -> Tuple[float, ...]:
        while True:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if self.show_raw_lines and line:
                print(f"[RAW:{self.side}] {line}")
            angles = parse_angles_line(line)
            if angles is not None:
                return angles

    def _calibrate_zero_on_start(
        self,
        *,
        max_samples: int,
        min_samples: int,
        stable_window: int,
        stable_eps: float,
        zero_scale: str,
    ) -> Dict[str, Tuple[float, ...]]:
        max_samples = max(1, int(max_samples))
        min_samples = max(1, int(min_samples))
        stable_window = max(1, int(stable_window))
        print(f"[Zero:{self.side}] keep hand_q6 fully open; waiting for a stable zero pose...")
        samples = []
        stabilized = False

        while len(samples) < max_samples:
            samples.append(self._read_angles())
            if len(samples) < stable_window or len(samples) < min_samples:
                continue

            window = samples[-stable_window:]
            ranges = tuple(max(sample[i] for sample in window) - min(sample[i] for sample in window) for i in range(6))
            if max(ranges) < stable_eps:
                stabilized = True
                break

        if not stabilized:
            raise RuntimeError(
                f"Runtime zero calibration failed: ANGLES did not stabilize in {len(samples)} samples. "
                "Keep the hand fully open and still, or use --no-zero-on-start."
            )

        used = samples[-stable_window:] if len(samples) >= stable_window else samples
        deg_at_0 = tuple(sum(sample[i] for sample in used) / len(used) for i in range(6))
        offset = offset_for_side(self.side)
        deg_at_1 = tuple(deg_at_0[i] + offset[i] for i in range(6))
        ranges = tuple(max(sample[i] for sample in used) - min(sample[i] for sample in used) for i in range(6))
        print(f"[Zero:{self.side}] finished with {len(samples)} samples, used {len(used)}")
        print(f"[Zero:{self.side}] deg_at_0={[round(v, 3) for v in deg_at_0]}")
        print(f"[Zero:{self.side}] used_range_deg={[round(v, 3) for v in ranges]}, scale={zero_scale}")
        return {
            "deg_at_0": deg_at_0,
            "deg_at_1": deg_at_1,
            "absolute_delta": zero_scale == "absolute",
        }

    def read_q6(self) -> Tuple[float, ...]:
        return angles_to_norm(self._read_angles(), self.calib)

    def close(self) -> None:
        self.ser.close()


def q6_source_order(args: argparse.Namespace) -> str:
    if args.source_order != "auto":
        return args.source_order
    return "teleop" if args.port else "internal"


def format_vec(values: Sequence[float], precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def clear_detected_errors(controller: DexHandController) -> bool:
    """Refresh feedback, clear active board errors, and return whether a clear was sent."""
    controller.read_state()
    errors = controller.read_errors()
    active = {board: error for board, error in errors.items() if error is not None}
    if not active:
        return False

    boards = ", ".join(str(board) for board in sorted(active))
    print(f"[DexHand] detected board error(s) on board(s): {boards}; clearing")
    ok = controller.clear_errors()
    print(f"[DexHand] clear_errors ok={ok}")
    return True


def compact_state_text(state: Dict[str, object]) -> str:
    joints = state.get("joints", {})
    tactile = state.get("tactile", {})
    joint_parts = []
    if isinstance(joints, dict):
        for name in ("th_rot", "th_mcp", "ff_mcp", "mf_mcp", "rf_mcp", "lf_mcp"):
            item = joints.get(name, {})
            angle = item.get("angle") if isinstance(item, dict) else None
            if angle is not None:
                joint_parts.append(f"{name}={float(angle):.1f}")
    tactile_parts = []
    if isinstance(tactile, dict):
        for finger in ("th", "ff", "mf", "rf", "lf"):
            item = tactile.get(finger, {})
            force = item.get("force_z") if isinstance(item, dict) else None
            if force is not None:
                tactile_parts.append(f"{finger}Fz={float(force):.2f}")
    return " ".join(joint_parts + tactile_parts)


def run_control_loop(
    *,
    args: argparse.Namespace,
    mapper: DexHandQ6Mapper,
    source_order: str,
    reader: Optional[SerialQ6Reader],
    controller: Optional[DexHandController],
    stop_event: Optional[threading.Event] = None,
) -> None:
    filtered_q6 = None
    last_sent_q6 = None
    loop_period = 1.0 / args.hz
    next_print_t = 0.0
    next_state_print_t = 0.0
    state_print_period = None if args.state_print_hz <= 0.0 else 1.0 / args.state_print_hz
    error_check_period = None if args.error_check_hz <= 0.0 else 1.0 / args.error_check_hz
    next_error_check_t = time.time() + (error_check_period or 0.0)
    frames = 0

    while stop_event is None or not stop_event.is_set():
        loop_t = time.time()
        q6 = args.q6 if args.q6 is not None else reader.read_q6()
        q6_internal = mapper.normalize_q6(q6, source_order=source_order)
        if filtered_q6 is None or args.alpha >= 1.0:
            filtered_q6 = q6_internal
        else:
            filtered_q6 = tuple(
                args.alpha * q6_internal[i] + (1.0 - args.alpha) * filtered_q6[i]
                for i in range(6)
            )

        action = mapper.normalized_q6_to_dexhand_action(filtered_q6)
        sent = False
        if last_sent_q6 is None or max(abs(filtered_q6[i] - last_sent_q6[i]) for i in range(6)) >= args.send_threshold:
            if controller is not None:
                controller.send_action(action)
            last_sent_q6 = tuple(filtered_q6)
            sent = True

        now = time.time()
        if now >= next_print_t:
            print(f"q6={format_vec(filtered_q6)} action={action} sent={sent}")
            next_print_t = now + 1.0 / args.print_hz

        if controller is not None and state_print_period is not None and now >= next_state_print_t:
            try:
                print(f"[State] {compact_state_text(controller.read_state())}")
            except Exception as exc:
                print(f"[State] read failed: {exc}")
            next_state_print_t = now + state_print_period

        if controller is not None and error_check_period is not None and now >= next_error_check_t:
            try:
                if clear_detected_errors(controller):
                    last_sent_q6 = None
            except Exception as exc:
                print(f"[DexHand] error check failed: {exc}")
            next_error_check_t = now + error_check_period

        frames += 1
        if args.q6 is not None:
            break
        if args.frames > 0 and frames >= args.frames:
            break
        sleep_time = loop_period - (time.time() - loop_t)
        if sleep_time > 0:
            time.sleep(sleep_time)


def main() -> int:
    args = parse_args()
    source_order = q6_source_order(args)
    mapper = DexHandQ6Mapper(
        max_deg=args.max_deg,
        spread_angle_deg=args.spread_angle,
        invert=args.invert,
        gain=args.gain,
        bias=args.bias,
        four_finger_mode=args.four_finger_mode,
        four_finger_profile_path=args.four_finger_profile,
    )

    reader = None
    controller = None
    stop_event = threading.Event()
    worker = None
    try:
        if args.port:
            reader = SerialQ6Reader(
                port=args.port,
                side=args.side,
                baudrate=args.baudrate,
                timeout=args.timeout,
                calibration=args.calibration,
                show_raw_lines=args.show_raw_lines,
                zero_on_start=args.zero_on_start,
                zero_max_samples=args.zero_max_samples,
                zero_min_samples=args.zero_min_samples,
                zero_stable_window=args.zero_stable_window,
                zero_stable_eps=args.zero_stable_eps,
                zero_scale=args.zero_scale,
            )
        if not args.dry_run:
            controller = DexHandController(
                hand=args.hand,
                device_index=args.device_index,
                device_type=args.device_type,
                mock=args.mock,
            ).connect()
            if args.clear_errors_on_start:
                print("[DexHand] clearing board errors on start")
                controller.clear_errors()

        if args.visualize_tactile and controller is not None:
            worker = threading.Thread(
                target=run_control_loop,
                kwargs={
                    "args": args,
                    "mapper": mapper,
                    "source_order": source_order,
                    "reader": reader,
                    "controller": controller,
                    "stop_event": stop_event,
                },
                daemon=True,
            )
            worker.start()
            run_viewer(controller, rate_hz=args.tactile_viewer_hz)
        else:
            run_control_loop(
                args=args,
                mapper=mapper,
                source_order=source_order,
                reader=reader,
                controller=controller,
                stop_event=stop_event,
            )
    except KeyboardInterrupt:
        print("\n[Exit] stopped")
    finally:
        stop_event.set()
        if worker is not None:
            worker.join(timeout=2.0)
        if controller is not None:
            controller.close()
        if reader is not None:
            reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
