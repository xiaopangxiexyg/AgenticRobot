#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TypeVar

import numpy as np
import serial

try:
    import redis
except Exception:
    redis = None

try:
    from project_paths import hand_zero_pose_calibrations_dir
except ModuleNotFoundError:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from project_paths import hand_zero_pose_calibrations_dir


ANGLE_RE = re.compile(r"=\s*([-+]?\d+(?:\.\d+)?)")
_ParsedT = TypeVar("_ParsedT")
OFFSET = [40.0, -70.0, 60.0, 60.0, 60.0, 60.0]
Q6_ORDER = [
    "thumb_yaw",
    "thumb_flex",
    "index",
    "middle",
    "ring",
    "pinky",
]

# teleop_sim.py
SIM_Q6_UPPER_RAD = np.array([1.54, 0.60, 1.57, 1.57, 1.57, 1.57], dtype=np.float32)
SIM_Q6_LOWER_RAD = np.zeros(6, dtype=np.float32)


class MissingPortSpecificCalibrationError(KeyError):
    """Raised when the saved zero pose does not cover the current Q6 device."""


def _default_zero_pose_record_dir() -> Path:
    return hand_zero_pose_calibrations_dir()


def _default_exo_hand_calibration_path() -> Path:
    return Path(__file__).resolve().with_name("exo_hand_zero_pose.json")


def _safe_name(text: object) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text or "unknown")).strip("_") or "unknown"


def _offset_for_side(side: str) -> np.ndarray:
    offset = np.asarray(OFFSET, dtype=np.float32)
    if str(side) == "left":
        offset = -offset
    return offset


def _calibration_from_deg_at_0(side: str, deg_at_0: Sequence[float]) -> Dict[str, np.ndarray]:
    deg0 = np.asarray(deg_at_0, dtype=np.float32).reshape(6)
    offset = _offset_for_side(side)
    return {"deg_at_0": deg0, "deg_at_1": deg0 + offset}


def _calibration_from_payload(side: str, payload: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    deg0 = np.asarray(payload["deg_at_0"], dtype=np.float32).reshape(6)
    if "deg_at_1" in payload:
        deg1 = np.asarray(payload["deg_at_1"], dtype=np.float32).reshape(6)
        return {"deg_at_0": deg0, "deg_at_1": deg1}
    return _calibration_from_deg_at_0(side, deg0)


def _unique_text(values: Sequence[object]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _port_identity(port: str | Path | None) -> dict[str, object]:
    if not port:
        return {"port": None, "port_realpath": None, "port_aliases": []}
    aliases: list[str] = [str(port)]
    realpath: str | None = None
    try:
        path = Path(str(port))
        if path.exists():
            realpath = str(path.resolve())
            aliases.append(realpath)
    except OSError:
        realpath = None
    if realpath is not None:
        for alias_dir in (Path("/dev/serial/by-id"), Path("/dev/serial/by-path")):
            if not alias_dir.exists():
                continue
            try:
                candidates = sorted(alias_dir.iterdir())
            except OSError:
                continue
            for candidate in candidates:
                try:
                    if str(candidate.resolve()) == realpath:
                        aliases.append(str(candidate))
                except OSError:
                    continue
    return {
        "port": str(port),
        "port_realpath": realpath,
        "port_aliases": _unique_text(aliases),
    }


def _stable_port_aliases(aliases: Sequence[object]) -> set[str]:
    stable_prefixes = ("/dev/serial/by-id/", "/dev/serial/by-path/")
    return {
        str(alias)
        for alias in aliases
        if str(alias).startswith(stable_prefixes)
    }


def _find_port_calibration_payload(
    payload: Mapping[str, Any],
    *,
    side: str,
    port: str | Path,
) -> Mapping[str, Any] | None:
    port_calibrations = payload.get("port_calibrations")
    side_registry = (
        port_calibrations.get(str(side))
        if isinstance(port_calibrations, Mapping)
        else None
    )
    if not isinstance(side_registry, Mapping):
        return None
    identity = _port_identity(port)
    aliases = set(str(alias) for alias in identity.get("port_aliases", []))
    stable_aliases = _stable_port_aliases(tuple(aliases))
    preferred_aliases = stable_aliases if stable_aliases else aliases
    for alias in preferred_aliases:
        candidate = side_registry.get(alias)
        if isinstance(candidate, Mapping) and "deg_at_0" in candidate:
            return candidate
    for candidate in side_registry.values():
        if not isinstance(candidate, Mapping) or "deg_at_0" not in candidate:
            continue
        candidate_aliases = set(str(alias) for alias in candidate.get("port_aliases", []))
        for key in ("port", "port_realpath"):
            if candidate.get(key):
                candidate_aliases.add(str(candidate[key]))
        if preferred_aliases.intersection(candidate_aliases):
            return candidate
    return None


def load_exo_hand_calibration(
    side: str,
    calibration_path: str | Path | None = None,
    *,
    port: str | Path | None = None,
) -> Dict[str, np.ndarray]:
    path = Path(calibration_path) if calibration_path is not None else _default_exo_hand_calibration_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    side_name = str(side)
    if port is not None:
        port_payload = _find_port_calibration_payload(payload, side=side_name, port=port)
        if port_payload is None:
            raise MissingPortSpecificCalibrationError(
                f"Missing port-specific exo hand zero pose calibration for "
                f"side={side!r}, port={str(port)!r} in {path}. "
                "Refusing to reuse a saved Q6 normalization from another device; "
                "auto zero-pose calibration is required while holding this Q6 at zero pose."
            )
        calib = _calibration_from_payload(side_name, port_payload)
        print(f"[Calib:{side}] loaded port-specific exo hand zero pose for port={port} <- {path}")
        return calib
    try:
        side_payload = payload["calibrations"][side_name]
    except KeyError as exc:
        raise KeyError(f"Missing fixed exo hand zero pose calibration for side={side!r} in {path}") from exc
    calib = _calibration_from_payload(side_name, side_payload)
    print(f"[Calib:{side}] loaded fixed exo hand zero pose <- {path}")
    return calib


def _update_fixed_exo_hand_calibration(
    *,
    side: str,
    port: str | None,
    baudrate: int | None,
    deg_at_0: np.ndarray,
    record_path: Path | None,
    calibration_path: str | Path | None = None,
) -> Path:
    path = Path(calibration_path) if calibration_path is not None else _default_exo_hand_calibration_path()
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"calibrations": {}}
    payload.setdefault("calibrations", {})
    offset = _offset_for_side(side)
    deg0 = np.asarray(deg_at_0, dtype=float).reshape(6)
    side_entry = {
        "deg_at_0": deg0.tolist(),
        "offset": np.asarray(offset, dtype=float).tolist(),
        "deg_at_1": (deg0 + np.asarray(offset, dtype=float)).tolist(),
    }
    payload["calibrations"][str(side)] = dict(side_entry)
    timestamp = time.time()
    identity = _port_identity(port)
    port_key = str(port) if port is not None else f"{side}_unknown_port"
    port_entry = {
        **side_entry,
        "side": str(side),
        "baudrate": None if baudrate is None else int(baudrate),
        "record_path": None if record_path is None else str(record_path),
        "updated_at": timestamp,
        "updated_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
        **identity,
    }
    payload.setdefault("port_calibrations", {})
    payload["port_calibrations"].setdefault(str(side), {})
    payload["port_calibrations"][str(side)][port_key] = port_entry
    for alias in identity.get("port_aliases", []):
        payload["port_calibrations"][str(side)][str(alias)] = port_entry
    payload["updated_at"] = timestamp
    payload["updated_at_local"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    payload["last_update"] = {
        "side": str(side),
        "port": None if port is None else str(port),
        "baudrate": None if baudrate is None else int(baudrate),
        "record_path": None if record_path is None else str(record_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[Calib:{side}] updated exo hand zero pose registry for port={port} -> {path}")
    return path


def _save_zero_pose_calibration_record(
    *,
    side: str,
    port: str | None,
    baudrate: int | None,
    deg_at_0: np.ndarray,
    deg_at_1: np.ndarray,
    offset: np.ndarray,
    sample_count: int,
    used_samples: list[list[float]],
    stable_window: int,
    stable_eps: float,
    save_dir: str | Path | None = None,
) -> Path:
    record_dir = Path(save_dir) if save_dir is not None else _default_zero_pose_record_dir()
    record_dir.mkdir(parents=True, exist_ok=True)
    used_array = np.asarray(used_samples, dtype=np.float32)
    ranges = np.ptp(used_array, axis=0) if used_array.size else np.zeros(6, dtype=np.float32)
    timestamp = time.time()
    timestamp_text = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
    filename = (
        f"{timestamp_text}_{int((timestamp % 1.0) * 1000):03d}_"
        f"{_safe_name(side)}_{_safe_name(port)}.json"
    )
    path = record_dir / filename
    identity = _port_identity(port)
    payload = {
        "timestamp": timestamp,
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
        "side": str(side),
        "port": None if port is None else str(port),
        "port_realpath": identity.get("port_realpath"),
        "port_aliases": identity.get("port_aliases", []),
        "baudrate": None if baudrate is None else int(baudrate),
        "q6_order": list(Q6_ORDER),
        "zero_pose_required": True,
        "sample_count": int(sample_count),
        "used_count": int(len(used_samples)),
        "stable_window": int(stable_window),
        "stable_eps_deg": float(stable_eps),
        "used_range_deg": np.asarray(ranges, dtype=float).tolist(),
        "used_max_range_deg": float(np.max(ranges)) if ranges.size else 0.0,
        "deg_at_0": np.asarray(deg_at_0, dtype=float).tolist(),
        "offset": np.asarray(offset, dtype=float).tolist(),
        "deg_at_1": np.asarray(deg_at_1, dtype=float).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[Calib:{side}] saved zero pose calibration -> {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one or two hand-only serial devices and expose normalized 6D values "
            "for dexterous-hand control. This is for the glove/hand serial path, not "
            "the arm Dynamixel bus."
        )
    )
    parser.add_argument("--port", type=str, default="", help="Single-hand serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--left-port", type=str, default="", help="Left-hand serial port")
    parser.add_argument("--right-port", type=str, default="", help="Right-hand serial port")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--timeout", type=float, default=0.05, help="Serial read timeout in seconds")
    parser.add_argument("--side", choices=["left", "right"], default="right", help="Used in single-hand mode only")
    parser.add_argument("--raw-only", action="store_true", help="Only print raw parsed angles and skip calibration")
    parser.add_argument(
        "--exo-hand-calibration",
        action="store_true",
        help="Re-run zero pose calibration and update the fixed exo hand calibration file.",
    )
    parser.add_argument(
        "--exo-hand-calibration-path",
        type=str,
        default="",
        help="Optional fixed exo hand calibration JSON path.",
    )
    parser.add_argument("--show-raw-lines", action="store_true", help="Print every raw serial line")
    parser.add_argument("--print-hz", type=float, default=10.0, help="Throttle console output rate")
    parser.add_argument("--frames", type=int, default=0, help="How many valid frames to print; 0 means infinite")
    parser.add_argument("--redis-host", type=str, default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-key", type=str, default="", help="If set, publish parsed packets to Redis")
    return parser.parse_args()


def parse_angles_line(line: str) -> Optional[List[float]]:
    if "ANGLES:" not in line:
        return None
    matches = ANGLE_RE.findall(line)
    if len(matches) != 6:
        return None
    try:
        return [float(x) for x in matches]
    except ValueError:
        return None


class BufferedSerialLineReader:
    """Read complete serial lines while dropping stale backlog.

    Q6 devices can publish faster than preview/control loops consume. Reading
    one ``readline()`` per loop can make the loop act on old buffered frames.
    This reader drains complete lines already available on the serial port and
    returns the newest parsed frame.
    """

    def __init__(
        self,
        ser: Any,
        *,
        show_raw_lines: bool = False,
        raw_prefix: str = "",
    ) -> None:
        self.ser = ser
        self.show_raw_lines = bool(show_raw_lines)
        self.raw_prefix = str(raw_prefix)
        self._pending = bytearray()
        self.last_read_stats: dict[str, int] = {
            "parsed_frames": 0,
            "stale_frames_dropped": 0,
            "pending_bytes": 0,
            "in_waiting": 0,
        }

    def _in_waiting(self) -> int:
        try:
            return max(0, int(getattr(self.ser, "in_waiting", 0) or 0))
        except Exception:
            return 0

    def _read_chunk(self, size: int) -> bytes:
        chunk = self.ser.read(max(1, int(size)))
        return chunk if isinstance(chunk, (bytes, bytearray)) else bytes(chunk or b"")

    def _read_some_blocking(self) -> None:
        waiting = self._in_waiting()
        chunk = self._read_chunk(waiting if waiting > 0 else 1)
        if chunk:
            self._pending.extend(chunk)

    def _drain_immediately_available(self) -> None:
        for _ in range(64):
            waiting = self._in_waiting()
            if waiting <= 0:
                return
            chunk = self._read_chunk(waiting)
            if not chunk:
                return
            self._pending.extend(chunk)

    def _consume_complete_lines(
        self,
        parse_line: Callable[[str], _ParsedT | None],
    ) -> tuple[_ParsedT | None, int]:
        latest: _ParsedT | None = None
        parsed_count = 0
        while True:
            newline_index = self._pending.find(b"\n")
            if newline_index < 0:
                return latest, parsed_count
            raw = bytes(self._pending[: newline_index + 1])
            del self._pending[: newline_index + 1]
            text = raw.decode("utf-8", errors="ignore").strip()
            if self.show_raw_lines and text:
                print(f"{self.raw_prefix}{text}", flush=True)
            parsed = parse_line(text)
            if parsed is not None:
                latest = parsed
                parsed_count += 1

    def read_latest_parsed(
        self,
        parse_line: Callable[[str], _ParsedT | None],
    ) -> _ParsedT:
        latest: _ParsedT | None = None
        parsed_total = 0
        while latest is None:
            latest, parsed_count = self._consume_complete_lines(parse_line)
            parsed_total += parsed_count
            if latest is not None:
                break
            self._read_some_blocking()

        self._drain_immediately_available()
        drained_latest, parsed_count = self._consume_complete_lines(parse_line)
        parsed_total += parsed_count
        if drained_latest is not None:
            latest = drained_latest

        self.last_read_stats = {
            "parsed_frames": int(parsed_total),
            "stale_frames_dropped": max(0, int(parsed_total) - 1),
            "pending_bytes": int(len(self._pending)),
            "in_waiting": int(self._in_waiting()),
        }
        return latest

    def read_latest_parsed_until(
        self,
        parse_line: Callable[[str], _ParsedT | None],
        *,
        deadline: float,
    ) -> _ParsedT | None:
        latest: _ParsedT | None = None
        parsed_total = 0
        while latest is None and time.time() < float(deadline):
            latest, parsed_count = self._consume_complete_lines(parse_line)
            parsed_total += parsed_count
            if latest is not None:
                break
            self._read_some_blocking()

        self._drain_immediately_available()
        drained_latest, parsed_count = self._consume_complete_lines(parse_line)
        parsed_total += parsed_count
        if drained_latest is not None:
            latest = drained_latest

        self.last_read_stats = {
            "parsed_frames": int(parsed_total),
            "stale_frames_dropped": max(0, int(parsed_total) - 1),
            "pending_bytes": int(len(self._pending)),
            "in_waiting": int(self._in_waiting()),
        }
        return latest


def swap01(x6: Sequence[float]) -> np.ndarray:
    y = np.asarray(x6, dtype=np.float32).copy()
    y[0], y[1] = y[1], y[0]
    return y


def norm6_to_q6_rad(norm6: Sequence[float]) -> np.ndarray:
    n = np.asarray(norm6, dtype=np.float32)
    n = np.clip(n, 0.0, 1.0)
    return SIM_Q6_LOWER_RAD + n * (SIM_Q6_UPPER_RAD - SIM_Q6_LOWER_RAD)


def auto_calibrate(
    ser: serial.Serial,
    side: str,
    port: str | None = None,
    baudrate: int | None = None,
    max_samples: int = 200,
    min_samples: int = 30,
    stable_window: int = 20,
    stable_eps: float = 0.5,
    show_raw_lines: bool = False,
    save_dir: str | Path | None = None,
    update_fixed: bool = False,
    calibration_path: str | Path | None = None,
) -> Dict[str, np.ndarray]:
    print(f"[Calib:{side}] keep the hand device at zero pose...")
    samples: List[List[float]] = []
    stabilized = False

    while len(samples) < max_samples:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if show_raw_lines and line:
            print(f"[RAW:{side}] {line}")
        angles = parse_angles_line(line)
        if angles is None:
            continue
        samples.append(angles)

        if len(samples) >= stable_window:
            window = samples[-stable_window:]
            ranges = [max(s[i] for s in window) - min(s[i] for s in window) for i in range(6)]
            if max(ranges) < stable_eps and len(samples) >= min_samples:
                stabilized = True
                break

    if not samples:
        raise RuntimeError("Calibration failed: no valid 'ANGLES:' frames were received")
    if not stabilized:
        raise RuntimeError(
            f"Calibration failed: angles did not stabilize in {len(samples)} samples. "
            "This usually means the protocol/baudrate is wrong, or the hand was moving."
        )

    used = samples[-stable_window:] if len(samples) >= stable_window else samples
    deg_at_0 = np.asarray([sum(s[i] for s in used) / len(used) for i in range(6)], dtype=np.float32)

    offset = _offset_for_side(side)
    deg_at_1 = deg_at_0 + offset

    print(f"[Calib:{side}] finished")
    record_path = _save_zero_pose_calibration_record(
        side=side,
        port=port,
        baudrate=baudrate,
        deg_at_0=deg_at_0,
        deg_at_1=deg_at_1,
        offset=offset,
        sample_count=len(samples),
        used_samples=used,
        stable_window=stable_window,
        stable_eps=stable_eps,
        save_dir=save_dir,
    )
    if update_fixed:
        _update_fixed_exo_hand_calibration(
            side=side,
            port=port,
            baudrate=baudrate,
            deg_at_0=deg_at_0,
            record_path=record_path,
            calibration_path=calibration_path,
        )
    return {"deg_at_0": deg_at_0, "deg_at_1": deg_at_1}


def _load_or_auto_calibrate_exo_hand(
    *,
    ser: serial.Serial,
    side: str,
    port: str,
    baudrate: int,
    show_raw_lines: bool,
    calibration_path: str | Path | None,
) -> Dict[str, np.ndarray]:
    try:
        return load_exo_hand_calibration(side, calibration_path, port=port)
    except MissingPortSpecificCalibrationError as exc:
        print(
            f"[Calib:{side}] {exc}. "
            "Auto-calibrating this Q6 now; keep it at zero pose and still.",
            flush=True,
        )
        return auto_calibrate(
            ser=ser,
            side=side,
            port=port,
            baudrate=baudrate,
            show_raw_lines=show_raw_lines,
            update_fixed=True,
            calibration_path=calibration_path,
        )


def map_angles_to_norm(angles_deg: Sequence[float], deg_at_0: np.ndarray, deg_at_1: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles_deg, dtype=np.float32)
    denom = deg_at_1 - deg_at_0
    denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)
    norm = (angles - deg_at_0) / denom
    return np.clip(norm, 0.0, 1.0)


def build_hand_packet(
    *,
    angles_deg: Sequence[float],
    side: str,
    port: str,
    baudrate: int,
    calib: Optional[Dict[str, np.ndarray]],
) -> Dict[str, object]:
    angles = np.asarray(angles_deg, dtype=np.float32)
    packet: Dict[str, object] = {
        "timestamp": np.asarray([time.time()], dtype=np.float64),
        "angles_deg": angles,
        "side": side,
        "port": port,
        "baudrate": int(baudrate),
    }

    if calib is None:
        return packet

    norm = map_angles_to_norm(
        angles_deg=angles,
        deg_at_0=calib["deg_at_0"],
        deg_at_1=calib["deg_at_1"],
    ).astype(np.float32)
    norm_swap = swap01(norm).astype(np.float32)

    # `norm` is the 6D 0..1 action that can be sent directly to DexHand.set_joint_pos(...).
    packet.update(
        {
            "deg_at_0": calib["deg_at_0"].astype(np.float32),
            "deg_at_1": calib["deg_at_1"].astype(np.float32),
            "norm": norm,
            "dexhand_norm": norm,
            "norm_raw": norm,
            "norm_swap01": norm_swap,
            "q6_order": list(Q6_ORDER),
            "sim_q6_rad": norm6_to_q6_rad(norm_swap).astype(np.float32),
            "q6_rad_swap01": norm6_to_q6_rad(norm_swap).astype(np.float32),
            "q6_rad_raw": norm6_to_q6_rad(norm).astype(np.float32),
        }
    )
    return packet


class SingleHandQ6Reader:
    def __init__(
        self,
        *,
        port: str,
        side: str,
        baudrate: int = 115200,
        timeout: float = 0.05,
        raw_only: bool = False,
        show_raw_lines: bool = False,
        exo_hand_calibration: bool = False,
        calibration_path: str | Path | None = None,
    ):
        self.port = str(port)
        self.side = str(side)
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.raw_only = bool(raw_only)
        self.show_raw_lines = bool(show_raw_lines)
        self.exo_hand_calibration = bool(exo_hand_calibration)
        self.calibration_path = calibration_path

        print(
            f"[Open:{self.side}] port={self.port} baudrate={self.baudrate} "
            "(for ttyACM USB CDC devices, baudrate may be only nominal)"
        )
        self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout)
        reset = getattr(self.ser, "reset_input_buffer", None)
        if callable(reset):
            reset()
        self._line_reader = BufferedSerialLineReader(
            self.ser,
            show_raw_lines=self.show_raw_lines,
            raw_prefix=f"[RAW:{self.side}] ",
        )
        self.calib = (
            None
            if self.raw_only
            else auto_calibrate(
                ser=self.ser,
                side=self.side,
                port=self.port,
                baudrate=self.baudrate,
                show_raw_lines=self.show_raw_lines,
                update_fixed=True,
                calibration_path=self.calibration_path,
            )
            if self.exo_hand_calibration
            else _load_or_auto_calibrate_exo_hand(
                ser=self.ser,
                side=self.side,
                port=self.port,
                baudrate=self.baudrate,
                show_raw_lines=self.show_raw_lines,
                calibration_path=self.calibration_path,
            )
        )

    def read_frame(self) -> Dict[str, object]:
        angles = self._line_reader.read_latest_parsed(parse_angles_line)
        packet = build_hand_packet(
            angles_deg=angles,
            side=self.side,
            port=self.port,
            baudrate=self.baudrate,
            calib=self.calib,
        )
        packet["serial_read_stats"] = dict(self._line_reader.last_read_stats)
        return packet

    def get_norm_values(self) -> np.ndarray:
        packet = self.read_frame()
        if "norm" not in packet:
            raise RuntimeError("raw_only=True does not provide normalized values")
        return np.asarray(packet["norm"], dtype=np.float32)

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass


class BimanualHandQ6Reader:
    def __init__(
        self,
        *,
        left_port: Optional[str] = None,
        right_port: Optional[str] = None,
        baudrate: int = 115200,
        timeout: float = 0.05,
        raw_only: bool = False,
        show_raw_lines: bool = False,
        exo_hand_calibration: bool = False,
        calibration_path: str | Path | None = None,
    ):
        if not left_port and not right_port:
            raise ValueError("At least one of left_port/right_port must be provided")

        self.left = None
        self.right = None
        if left_port:
            self.left = SingleHandQ6Reader(
                port=left_port,
                side="left",
                baudrate=baudrate,
                timeout=timeout,
                raw_only=raw_only,
                show_raw_lines=show_raw_lines,
                exo_hand_calibration=exo_hand_calibration,
                calibration_path=calibration_path,
            )
        if right_port:
            self.right = SingleHandQ6Reader(
                port=right_port,
                side="right",
                baudrate=baudrate,
                timeout=timeout,
                raw_only=raw_only,
                show_raw_lines=show_raw_lines,
                exo_hand_calibration=exo_hand_calibration,
                calibration_path=calibration_path,
            )

    def read_frame(self) -> Dict[str, object]:
        packet: Dict[str, object] = {"timestamp": np.asarray([time.time()], dtype=np.float64)}
        if self.left is not None:
            packet["left"] = self.left.read_frame()
        if self.right is not None:
            packet["right"] = self.right.read_frame()

        if "left" in packet and "right" in packet:
            left_packet = packet["left"]
            right_packet = packet["right"]
            if "norm" in left_packet and "norm" in right_packet:
                packet["norm_action12"] = np.concatenate(
                    [
                        np.asarray(left_packet["norm"], dtype=np.float32),
                        np.asarray(right_packet["norm"], dtype=np.float32),
                    ],
                    axis=0,
                )
                packet["action12_order"] = [f"left_{name}" for name in Q6_ORDER] + [f"right_{name}" for name in Q6_ORDER]

        return packet

    def get_norm_values(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        if self.left is not None:
            out["left"] = self.left.get_norm_values()
        if self.right is not None:
            out["right"] = self.right.get_norm_values()
        return out

    def get_norm_action12(self) -> np.ndarray:
        if self.left is None or self.right is None:
            raise RuntimeError("Both left and right hands are required for action12 output")
        return np.concatenate([self.left.get_norm_values(), self.right.get_norm_values()], axis=0)

    def close(self) -> None:
        if self.left is not None:
            self.left.close()
        if self.right is not None:
            self.right.close()


def open_redis_client(args: argparse.Namespace):
    if not args.redis_key:
        return None
    if redis is None:
        raise RuntimeError("redis package is not available, but --redis-key was provided")
    return redis.Redis(host=args.redis_host, port=args.redis_port)


def print_single_packet(packet: Dict[str, object], raw_only: bool) -> None:
    raw = np.array2string(np.asarray(packet["angles_deg"]), precision=2, separator=", ")
    print(f"[{packet['side']}] angles_deg: {raw}")
    if raw_only:
        return

    norm = np.array2string(np.asarray(packet["norm"]), precision=3, separator=", ")
    norm_swap = np.array2string(np.asarray(packet["norm_swap01"]), precision=3, separator=", ")
    sim_q6 = np.array2string(np.asarray(packet["sim_q6_rad"]), precision=3, separator=", ")

    print(f"[{packet['side']}] norm(用于DexHand.set_joint_pos): {norm}")
    print(f"[{packet['side']}] norm_swap01(仿真兼容):       {norm_swap}")
    print(f"[{packet['side']}] sim_q6_rad:                 {sim_q6}")


def print_bimanual_packet(packet: Dict[str, object], raw_only: bool) -> None:
    if "left" in packet:
        print_single_packet(packet["left"], raw_only=raw_only)
    if "right" in packet:
        print_single_packet(packet["right"], raw_only=raw_only)
    if (not raw_only) and ("norm_action12" in packet):
        action12 = np.array2string(np.asarray(packet["norm_action12"]), precision=3, separator=", ")
        print(f"[bimanual] norm_action12(left6+right6): {action12}")


def build_reader_from_args(args: argparse.Namespace):
    if args.port and (args.left_port or args.right_port):
        raise ValueError("Use either --port for single-hand mode, or --left-port/--right-port for dual-hand mode")
    if args.port:
        return SingleHandQ6Reader(
            port=args.port,
            side=args.side,
            baudrate=args.baudrate,
            timeout=args.timeout,
            raw_only=args.raw_only,
            show_raw_lines=args.show_raw_lines,
            exo_hand_calibration=args.exo_hand_calibration,
            calibration_path=args.exo_hand_calibration_path or None,
        )
    if args.left_port or args.right_port:
        return BimanualHandQ6Reader(
            left_port=args.left_port or None,
            right_port=args.right_port or None,
            baudrate=args.baudrate,
            timeout=args.timeout,
            raw_only=args.raw_only,
            show_raw_lines=args.show_raw_lines,
            exo_hand_calibration=args.exo_hand_calibration,
            calibration_path=args.exo_hand_calibration_path or None,
        )
    raise ValueError("Provide --port, or at least one of --left-port/--right-port")


def main() -> None:
    args = parse_args()
    publisher = open_redis_client(args)
    reader = build_reader_from_args(args)

    try:
        print("[Start] reading hand q6 frames")
        next_print_t = 0.0
        valid_frames = 0
        while True:
            packet = reader.read_frame()

            if publisher is not None:
                publisher.set(args.redis_key, pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL))

            now = time.time()
            if now >= next_print_t:
                if isinstance(reader, SingleHandQ6Reader):
                    print_single_packet(packet, raw_only=args.raw_only)
                else:
                    print_bimanual_packet(packet, raw_only=args.raw_only)
                next_print_t = now + (1.0 / max(args.print_hz, 1e-6))

            valid_frames += 1
            if args.frames > 0 and valid_frames >= args.frames:
                break
    except KeyboardInterrupt:
        print("\n[Exit] stopped by user")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
