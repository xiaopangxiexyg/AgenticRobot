from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ANGLE_RE = re.compile(r"=\s*([-+]?\d+(?:\.\d+)?)")
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
    return Path(__file__).resolve().parents[2] / "teleop_records" / "hand_zero_pose_calibrations"


def _default_exo_hand_calibration_path() -> Path:
    return Path(__file__).resolve().parents[2] / "hardware" / "teleop_hand" / "exo_hand_zero_pose.json"


def _safe_name(text: object) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text or "unknown")).strip("_") or "unknown"


def _offset_for_side(side: str) -> np.ndarray:
    offset = np.asarray(OFFSET, dtype=np.float32)
    if str(side) == "left":
        offset = -offset
    return offset


def _calibration_from_deg_at_0(side: str, deg_at_0: Sequence[float]) -> dict[str, np.ndarray]:
    deg0 = np.asarray(deg_at_0, dtype=np.float32).reshape(6)
    offset = _offset_for_side(side)
    return {"deg_at_0": deg0, "deg_at_1": deg0 + offset}


def load_exo_hand_calibration(
    side: str,
    calibration_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    path = Path(calibration_path) if calibration_path is not None else _default_exo_hand_calibration_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        side_payload = payload["calibrations"][str(side)]
        deg_at_0 = side_payload["deg_at_0"]
    except KeyError as exc:
        raise KeyError(f"Missing fixed exo hand zero pose calibration for side={side!r} in {path}") from exc
    calib = _calibration_from_deg_at_0(str(side), deg_at_0)
    print(f"[Calib:{side}] loaded fixed exo hand zero pose <- {path}")
    return calib


def _update_fixed_exo_hand_calibration(
    *,
    side: str,
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
    payload["calibrations"][str(side)] = {
        "deg_at_0": deg0.tolist(),
        "offset": np.asarray(offset, dtype=float).tolist(),
        "deg_at_1": (deg0 + np.asarray(offset, dtype=float)).tolist(),
    }
    timestamp = time.time()
    payload["updated_at"] = timestamp
    payload["updated_at_local"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    payload["last_update"] = {
        "side": str(side),
        "record_path": None if record_path is None else str(record_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[Calib:{side}] updated fixed exo hand zero pose -> {path}")
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
    payload = {
        "timestamp": timestamp,
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
        "side": str(side),
        "port": None if port is None else str(port),
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
            else load_exo_hand_calibration(self.side, self.calibration_path)
        )

    def read_frame(self) -> dict[str, object]:
        while True:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if self.show_raw_lines and line:
                print(f"[RAW:{self.side}] {line}")
            angles = parse_angles_line(line)
            if angles is None:
                continue
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
