from __future__ import annotations

import re
import time
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
    max_samples: int = 200,
    min_samples: int = 30,
    stable_window: int = 20,
    stable_eps: float = 0.5,
    show_raw_lines: bool = False,
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
    offset = np.asarray(OFFSET, dtype=np.float32)
    if side == "left":
        offset = -offset
    deg_at_1 = deg_at_0 + offset
    print(f"[Calib:{side}] finished")
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
    ):
        import serial

        self.port = str(port)
        self.side = str(side)
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.raw_only = bool(raw_only)
        self.show_raw_lines = bool(show_raw_lines)

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
                show_raw_lines=self.show_raw_lines,
            )
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
