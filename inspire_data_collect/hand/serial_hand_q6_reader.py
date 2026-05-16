from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

import numpy as np

from project_paths import hand_zero_pose_calibrations_dir, repo_path

ANGLE_RE = re.compile(r"=\s*([-+]?\d+(?:\.\d+)?)")
_ParsedT = TypeVar("_ParsedT")
OFFSET = [40.0, -70.0, 60.0, 60.0, 60.0, 60.0]
Q6_ORDER = (
    "thumb_yaw",
    "thumb_flex",
    "index",
    "middle",
    "ring",
    "pinky",
)

SIM_Q6_UPPER_RAD = np.array([1.54, 0.60, 1.57, 1.57, 1.57, 1.57], dtype=np.float32)
SIM_Q6_LOWER_RAD = np.zeros(6, dtype=np.float32)


def _default_zero_pose_record_dir() -> Path:
    return hand_zero_pose_calibrations_dir()


def _default_exo_hand_calibration_path() -> Path:
    return repo_path("hardware", "teleop_hand", "exo_hand_zero_pose.json")


def _safe_name(text: object) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text or "unknown")).strip("_") or "unknown"


def _unique_text(values: Sequence[object]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _port_identity(port: str | None) -> dict[str, object]:
    if not port:
        return {"port": None, "port_realpath": None, "port_aliases": []}
    port_text = str(port)
    aliases: list[str] = [port_text]
    realpath: str | None = None
    try:
        path = Path(port_text)
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
        "port": port_text,
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


def _offset_for_side(side: str) -> np.ndarray:
    offset = np.asarray(OFFSET, dtype=np.float32)
    if str(side) == "left":
        offset = -offset
    return offset


def _calibration_from_payload(side: str, payload: dict[str, Any]) -> dict[str, np.ndarray]:
    deg0 = np.asarray(payload["deg_at_0"], dtype=np.float32).reshape(6)
    if "deg_at_1" in payload:
        deg1 = np.asarray(payload["deg_at_1"], dtype=np.float32).reshape(6)
    else:
        deg1 = deg0 + _offset_for_side(side)
    return {"deg_at_0": deg0, "deg_at_1": deg1}


def _find_port_calibration_payload(
    payload: dict[str, Any],
    *,
    side: str,
    port: str,
) -> dict[str, Any] | None:
    side_registry = payload.get("port_calibrations", {}).get(str(side), {})
    if not isinstance(side_registry, dict):
        return None
    identity = _port_identity(port)
    aliases = set(str(alias) for alias in identity.get("port_aliases", []))
    stable_aliases = _stable_port_aliases(tuple(aliases))
    preferred_aliases = stable_aliases if stable_aliases else aliases
    for alias in preferred_aliases:
        value = side_registry.get(alias)
        if isinstance(value, dict):
            return value
    for value in side_registry.values():
        if not isinstance(value, dict):
            continue
        entry_aliases = set(str(alias) for alias in value.get("port_aliases", []))
        for key in ("port", "port_realpath"):
            if value.get(key):
                entry_aliases.add(str(value[key]))
        if preferred_aliases.intersection(entry_aliases):
            return value
    return None


def _record_matches_port_identity(record: dict[str, Any], identity: dict[str, object]) -> bool:
    aliases = set(str(alias) for alias in identity.get("port_aliases", []))
    stable_aliases = _stable_port_aliases(tuple(aliases))
    preferred_aliases = stable_aliases if stable_aliases else aliases
    record_aliases = set(str(alias) for alias in record.get("port_aliases", []))
    for key in ("port", "port_realpath"):
        if record.get(key):
            record_aliases.add(str(record[key]))
    return bool(preferred_aliases.intersection(record_aliases))


def _load_latest_zero_pose_record_for_port(
    *,
    side: str,
    port: str,
    record_dir: str | Path | None = None,
) -> tuple[dict[str, Any], Path] | None:
    root = Path(record_dir) if record_dir is not None else _default_zero_pose_record_dir()
    if not root.exists():
        return None
    identity = _port_identity(port)
    matches: list[tuple[float, Path, dict[str, Any]]] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("side")) != str(side):
            continue
        if not _record_matches_port_identity(payload, identity):
            continue
        try:
            timestamp = float(payload.get("timestamp", 0.0))
        except Exception:
            timestamp = 0.0
        matches.append((timestamp, path, payload))
    if not matches:
        return None
    _, path, payload = max(matches, key=lambda item: item[0])
    return payload, path


def load_exo_hand_calibration(
    side: str,
    calibration_path: str | Path | None = None,
    *,
    port: str | None = None,
) -> dict[str, np.ndarray]:
    path = Path(calibration_path) if calibration_path is not None else _default_exo_hand_calibration_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if port is not None:
        port_payload = _find_port_calibration_payload(payload, side=str(side), port=str(port))
        if port_payload is None:
            record_match = _load_latest_zero_pose_record_for_port(side=str(side), port=str(port))
            if record_match is None:
                raise KeyError(
                    f"Missing port-specific exo hand zero pose calibration for "
                    f"side={side!r}, port={port!r} in {path}"
                )
            record_payload, record_path = record_match
            calib = _calibration_from_payload(str(side), record_payload)
            _update_fixed_exo_hand_calibration(
                side=str(side),
                port=str(port),
                baudrate=record_payload.get("baudrate"),
                deg_at_0=calib["deg_at_0"],
                record_path=record_path,
                calibration_path=path,
            )
            print(
                f"[Calib:{side}] imported latest zero pose record for port={port} "
                f"from {record_path}"
            )
            return calib
        calib = _calibration_from_payload(str(side), port_payload)
        print(f"[Calib:{side}] loaded port-marked exo hand zero pose for port={port} <- {path}")
        return calib
    try:
        side_payload = payload["calibrations"][str(side)]
    except KeyError as exc:
        raise KeyError(f"Missing fixed exo hand zero pose calibration for side={side!r} in {path}") from exc
    calib = _calibration_from_payload(str(side), side_payload)
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


def parse_angles_line(line: str) -> list[float] | None:
    if "ANGLES:" not in line:
        return None
    matches = ANGLE_RE.findall(line)
    if len(matches) != 6:
        return None
    try:
        return [float(value) for value in matches]
    except ValueError:
        return None


class BufferedSerialLineReader:
    """Read complete serial lines while dropping stale backlog.

    The hand-q6 source can publish faster than some control loops consume. If we
    call ``readline()`` only once per control tick, low-Hz loops end up acting on
    the oldest buffered frame instead of the most recent pose. This helper keeps
    an internal byte buffer, drains all complete lines currently waiting on the
    serial port, and returns the latest parsed frame.
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

    def _read_chunk(self, size: int) -> bytes:
        chunk = self.ser.read(max(1, int(size)))
        return chunk if isinstance(chunk, (bytes, bytearray)) else bytes(chunk or b"")

    def _read_some_blocking(self) -> None:
        waiting = int(getattr(self.ser, "in_waiting", 0) or 0)
        self._pending.extend(self._read_chunk(waiting if waiting > 0 else 1))

    def _drain_immediately_available(self) -> None:
        while True:
            waiting = int(getattr(self.ser, "in_waiting", 0) or 0)
            if waiting <= 0:
                return
            chunk = self._read_chunk(waiting)
            if not chunk:
                return
            self._pending.extend(chunk)

    def _consume_complete_lines(
        self,
        parse_line: Callable[[str], _ParsedT | None],
    ) -> _ParsedT | None:
        latest: _ParsedT | None = None
        while True:
            newline_index = self._pending.find(b"\n")
            if newline_index < 0:
                return latest
            raw = bytes(self._pending[: newline_index + 1])
            del self._pending[: newline_index + 1]
            text = raw.decode("utf-8", errors="ignore").strip()
            if self.show_raw_lines and text:
                print(f"{self.raw_prefix}{text}")
            parsed = parse_line(text)
            if parsed is not None:
                latest = parsed

    def read_latest_parsed(
        self,
        parse_line: Callable[[str], _ParsedT | None],
    ) -> _ParsedT:
        latest: _ParsedT | None = None
        while latest is None:
            latest = self._consume_complete_lines(parse_line)
            if latest is not None:
                break
            self._read_some_blocking()
        self._drain_immediately_available()
        drained_latest = self._consume_complete_lines(parse_line)
        return latest if drained_latest is None else drained_latest


def swap01(x6: Sequence[float]) -> np.ndarray:
    values = np.asarray(x6, dtype=np.float32).copy()
    values[0], values[1] = values[1], values[0]
    return values


def norm6_to_q6_rad(norm6: Sequence[float]) -> np.ndarray:
    norm = np.asarray(norm6, dtype=np.float32)
    norm = np.clip(norm, 0.0, 1.0)
    return SIM_Q6_LOWER_RAD + norm * (SIM_Q6_UPPER_RAD - SIM_Q6_LOWER_RAD)


def auto_calibrate(
    ser: Any,
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
) -> dict[str, np.ndarray]:
    print(f"[Calib:{side}] keep the hand device at zero pose...")
    samples: list[list[float]] = []
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
            ranges = [
                max(sample[index] for sample in window)
                - min(sample[index] for sample in window)
                for index in range(6)
            ]
            if max(ranges) < stable_eps and len(samples) >= min_samples:
                stabilized = True
                break

    if not samples:
        raise RuntimeError("Calibration failed: no valid 'ANGLES:' frames were received.")
    if not stabilized:
        raise RuntimeError(
            f"Calibration failed: angles did not stabilize in {len(samples)} samples. "
            "Check protocol/baudrate, and keep the source hand still at zero pose."
        )

    used = samples[-stable_window:] if len(samples) >= stable_window else samples
    deg_at_0 = np.asarray(
        [sum(sample[index] for sample in used) / len(used) for index in range(6)],
        dtype=np.float32,
    )
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


def map_angles_to_norm(
    angles_deg: Sequence[float],
    deg_at_0: np.ndarray,
    deg_at_1: np.ndarray,
) -> np.ndarray:
    angles = np.asarray(angles_deg, dtype=np.float32)
    denom = deg_at_1 - deg_at_0
    denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)
    return np.clip((angles - deg_at_0) / denom, 0.0, 1.0)


def build_hand_packet(
    *,
    angles_deg: Sequence[float],
    side: str,
    port: str,
    baudrate: int,
    calib: dict[str, np.ndarray] | None,
) -> dict[str, object]:
    angles = np.asarray(angles_deg, dtype=np.float32)
    packet: dict[str, object] = {
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
        import serial

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
            "(for ttyACM USB CDC devices, baudrate may be nominal)"
        )
        self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout)
        self._line_reader = BufferedSerialLineReader(
            self.ser,
            show_raw_lines=self.show_raw_lines,
            raw_prefix=f"[RAW:{self.side}] ",
        )
        if self.raw_only:
            self.calib = None
        elif self.exo_hand_calibration:
            self.calib = auto_calibrate(
                ser=self.ser,
                side=self.side,
                port=self.port,
                baudrate=self.baudrate,
                show_raw_lines=self.show_raw_lines,
                update_fixed=True,
                calibration_path=self.calibration_path,
            )
        else:
            try:
                self.calib = load_exo_hand_calibration(
                    self.side,
                    self.calibration_path,
                    port=self.port,
                )
            except KeyError as exc:
                print(f"[Calib:{self.side}] {exc}")
                print(
                    f"[Calib:{self.side}] port={self.port} is not marked; "
                    "running zero pose calibration and recording it."
                )
                self.calib = auto_calibrate(
                    ser=self.ser,
                    side=self.side,
                    port=self.port,
                    baudrate=self.baudrate,
                    show_raw_lines=self.show_raw_lines,
                    update_fixed=True,
                    calibration_path=self.calibration_path,
                )

    def read_frame(self) -> dict[str, object]:
        angles = self._line_reader.read_latest_parsed(parse_angles_line)
        return build_hand_packet(
            angles_deg=angles,
            side=self.side,
            port=self.port,
            baudrate=self.baudrate,
            calib=self.calib,
        )

    def get_norm_values(self) -> np.ndarray:
        packet = self.read_frame()
        if "norm" not in packet:
            raise RuntimeError("raw_only=True does not provide normalized values.")
        return np.asarray(packet["norm"], dtype=np.float32)

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass
