from __future__ import annotations

import argparse
import json
import math
import os
import select
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence

from serial.tools import list_ports

from .exo_driver import ExoDriver, ExoDriverCfg


SIDES = ("left", "right")
PROBE_SERVO_IDS = (1, 2)
ASSIGNMENT_SCHEMA_VERSION = "angle_trigger_assignment.v1"
ANGLE_SCHEMA_VERSION = "angle_trigger_angle_calibration.v1"
DEFAULT_BAUDRATE = 115200
DEFAULT_SAMPLE_HZ = 30.0
DEFAULT_CAPTURE_SECONDS = 1.0
DEFAULT_ASSIGNMENT_MOTION_MIN_DEG = 3.0
DEFAULT_ENDPOINT_MIN_SEPARATION_DEG = 2.0


class AngleTriggerError(RuntimeError):
    """A recoverable angle-trigger discovery or calibration error."""


@dataclass(frozen=True)
class SensorCandidate:
    device: str
    servo_id: int
    location: str
    serial_number: str
    vid: str
    pid: str
    description: str

    def identity(self) -> dict[str, str | int]:
        return {
            "location": self.location,
            "serial_number": self.serial_number,
            "vid": self.vid,
            "pid": self.pid,
            "servo_id": self.servo_id,
        }

    @property
    def label(self) -> str:
        return f"{self.device} (servo {self.servo_id}, location={self.location or 'unknown'})"


class SensorConnection:
    def __init__(self, candidate: SensorCandidate, *, baudrate: int = DEFAULT_BAUDRATE) -> None:
        self.candidate = candidate
        self.baudrate = baudrate
        self._driver: ExoDriver | None = None

    def open(self) -> None:
        if self._driver is not None:
            return
        driver = ExoDriver(
            ExoDriverCfg(
                port=self.candidate.device,
                baudrate=self.baudrate,
                servo_ids=[self.candidate.servo_id],
                timeout_s=0.12,
                command_delay_s=0.01,
                read_hz=30.0,
            )
        )
        driver.open()
        self._driver = driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def read_angle_deg(self) -> float:
        if self._driver is None:
            raise AngleTriggerError(f"Sensor {self.candidate.label} is not open.")
        reading = self._driver.read_position(self.candidate.servo_id)
        if not reading.ok or reading.angle_deg is None:
            raise AngleTriggerError(
                f"No valid angle response from {self.candidate.label}: {reading.response.strip()!r}"
            )
        return float(reading.angle_deg)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_root() -> Path:
    try:
        root = repo_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from project_paths import hardware_interface_configs_root

        return hardware_interface_configs_root() / "angle_trigger"
    except Exception:
        return repo_root() / "teleop_runtime" / "configs" / "hardware" / "angle_trigger"


def assignment_path(config_root: Path | None = None) -> Path:
    return (config_root or default_config_root()) / "assignment" / "latest.json"


def angle_calibration_path(config_root: Path | None = None) -> Path:
    return (config_root or default_config_root()) / "angles" / "latest.json"


def iso_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _format_usb_id(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{int(value):04x}"
    except (TypeError, ValueError):
        return str(value).strip().lower()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AngleTriggerError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AngleTriggerError(f"{path} must contain a JSON object.")
    return payload


def _probe_port(port: Any, *, baudrate: int) -> SensorCandidate | None:
    successful_ids: list[int] = []
    for servo_id in PROBE_SERVO_IDS:
        driver = ExoDriver(
            ExoDriverCfg(
                port=str(port.device),
                baudrate=baudrate,
                servo_ids=[servo_id],
                timeout_s=0.12,
                command_delay_s=0.01,
                read_hz=30.0,
            )
        )
        try:
            driver.open()
            if driver.read_position(servo_id).ok:
                successful_ids.append(servo_id)
        except (OSError, RuntimeError):
            pass
        finally:
            driver.close()
    if len(successful_ids) != 1:
        return None
    return SensorCandidate(
        device=str(port.device),
        servo_id=successful_ids[0],
        location=str(getattr(port, "location", "") or "").strip(),
        serial_number=str(getattr(port, "serial_number", "") or "").strip(),
        vid=_format_usb_id(getattr(port, "vid", None)),
        pid=_format_usb_id(getattr(port, "pid", None)),
        description=str(getattr(port, "description", "") or "").strip(),
    )


def discover_sensors(*, baudrate: int = DEFAULT_BAUDRATE) -> list[SensorCandidate]:
    """Probe USB serial devices and retain ports with exactly one valid ID 1/2."""

    sensors: list[SensorCandidate] = []
    for port in sorted(list_ports.comports(), key=lambda item: str(item.device)):
        candidate = _probe_port(port, baudrate=baudrate)
        if candidate is not None:
            sensors.append(candidate)
    return sensors


def _normalize_assignment(payload: Mapping[str, Any]) -> dict[str, Any]:
    if str(payload.get("schema_version") or "") != ASSIGNMENT_SCHEMA_VERSION:
        raise AngleTriggerError("Unsupported or missing angle-trigger assignment schema.")
    raw_sides = payload.get("sides")
    if not isinstance(raw_sides, Mapping):
        raise AngleTriggerError("Assignment config must contain a sides object.")
    sides: dict[str, dict[str, str | int]] = {}
    seen_identities: set[tuple[str, str, str, str, int]] = set()
    for side in SIDES:
        raw = raw_sides.get(side)
        if not isinstance(raw, Mapping):
            raise AngleTriggerError(f"Assignment config is missing {side}.")
        identity = raw.get("identity") if isinstance(raw.get("identity"), Mapping) else raw
        location = str(identity.get("location") or "").strip()
        serial_number = str(identity.get("serial_number") or "").strip()
        vid = str(identity.get("vid") or "").strip().lower()
        pid = str(identity.get("pid") or "").strip().lower()
        try:
            servo_id = int(identity.get("servo_id"))
        except (TypeError, ValueError) as exc:
            raise AngleTriggerError(f"Assignment {side} has an invalid servo_id.") from exc
        if servo_id not in PROBE_SERVO_IDS:
            raise AngleTriggerError(f"Assignment {side} servo_id must be one of {PROBE_SERVO_IDS}.")
        if not serial_number and not location:
            raise AngleTriggerError(f"Assignment {side} needs a USB serial number or location.")
        unique_key = (location, serial_number, vid, pid, servo_id)
        if unique_key in seen_identities:
            raise AngleTriggerError("Left and right assignments must refer to different sensors.")
        seen_identities.add(unique_key)
        sides[side] = {
            "location": location,
            "serial_number": serial_number,
            "vid": vid,
            "pid": pid,
            "servo_id": servo_id,
        }
    return {
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "updated_at": str(payload.get("updated_at") or iso_timestamp()),
        "sides": sides,
    }


def _normalize_angle_calibration(payload: Mapping[str, Any]) -> dict[str, Any]:
    if str(payload.get("schema_version") or "") != ANGLE_SCHEMA_VERSION:
        raise AngleTriggerError("Unsupported or missing angle-trigger angle-calibration schema.")
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, Mapping):
        raise AngleTriggerError("Angle calibration config must contain a profiles object.")
    profiles: dict[str, dict[str, float]] = {}
    for side in SIDES:
        raw = raw_profiles.get(side)
        if not isinstance(raw, Mapping):
            raise AngleTriggerError(f"Angle calibration is missing {side}.")
        try:
            zero_deg = float(raw["zero_deg"])
            one_deg = float(raw["one_deg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AngleTriggerError(f"Angle calibration {side} is invalid.") from exc
        if not math.isfinite(zero_deg) or not math.isfinite(one_deg):
            raise AngleTriggerError(f"Angle calibration {side} must be finite.")
        if abs(one_deg - zero_deg) < DEFAULT_ENDPOINT_MIN_SEPARATION_DEG:
            raise AngleTriggerError(f"Angle calibration {side} endpoints are too close together.")
        profiles[side] = {"zero_deg": zero_deg, "one_deg": one_deg}
    return {
        "schema_version": ANGLE_SCHEMA_VERSION,
        "updated_at": str(payload.get("updated_at") or iso_timestamp()),
        "profiles": profiles,
    }


def _candidate_matches(identity: Mapping[str, Any], candidate: SensorCandidate) -> bool:
    if int(identity["servo_id"]) != candidate.servo_id:
        return False
    serial_number = str(identity["serial_number"] or "")
    if serial_number:
        return (
            serial_number == candidate.serial_number
            and str(identity["vid"] or "") == candidate.vid
            and str(identity["pid"] or "") == candidate.pid
        )
    return (
        str(identity["location"] or "") == candidate.location
        and str(identity["vid"] or "") == candidate.vid
        and str(identity["pid"] or "") == candidate.pid
    )


def resolve_assignment(
    assignment: Mapping[str, Any], sensors: Iterable[SensorCandidate]
) -> tuple[dict[str, SensorCandidate] | None, list[str]]:
    normalized = _normalize_assignment(assignment)
    available = list(sensors)
    resolved: dict[str, SensorCandidate] = {}
    errors: list[str] = []
    for side in SIDES:
        matches = [
            sensor
            for sensor in available
            if _candidate_matches(normalized["sides"][side], sensor)
        ]
        if len(matches) != 1:
            errors.append(f"Could not uniquely resolve the configured {side} sensor.")
            continue
        resolved[side] = matches[0]
        available.remove(matches[0])
    return (resolved if not errors else None), errors


def build_assignment(mapping: Mapping[str, SensorCandidate]) -> dict[str, Any]:
    if set(mapping) != set(SIDES) or mapping["left"] == mapping["right"]:
        raise AngleTriggerError("Assignment needs two different left and right sensors.")
    return _normalize_assignment(
        {
            "schema_version": ASSIGNMENT_SCHEMA_VERSION,
            "updated_at": iso_timestamp(),
            "sides": {side: {"identity": mapping[side].identity()} for side in SIDES},
        }
    )


def build_angle_calibration(profiles: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    return _normalize_angle_calibration(
        {
            "schema_version": ANGLE_SCHEMA_VERSION,
            "updated_at": iso_timestamp(),
            "profiles": dict(profiles),
        }
    )


def _open_connections(mapping: Mapping[str, SensorCandidate], *, baudrate: int) -> dict[str, SensorConnection]:
    connections = {side: SensorConnection(mapping[side], baudrate=baudrate) for side in SIDES}
    try:
        for connection in connections.values():
            connection.open()
    except Exception:
        for connection in connections.values():
            connection.close()
        raise
    return connections


def _close_connections(connections: Mapping[str, SensorConnection]) -> None:
    for connection in connections.values():
        connection.close()


def _wait_for_side_motion(
    side: str,
    candidates: Mapping[str, SensorConnection],
    *,
    sample_hz: float,
    min_motion_deg: float,
) -> str:
    if not sys.stdin.isatty():
        raise AngleTriggerError("Side assignment needs an interactive terminal.")
    print(
        f"Move the {side} angle sensor through a clear range, then press Enter to assign it.",
        flush=True,
    )
    period = 1.0 / sample_hz
    while True:
        samples: dict[str, list[float]] = {name: [] for name in candidates}
        next_tick = time.monotonic()
        while True:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick = max(next_tick + period, time.monotonic())
            for name, connection in candidates.items():
                try:
                    samples[name].append(connection.read_angle_deg())
                except AngleTriggerError:
                    continue
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                sys.stdin.readline()
                break
        ranges = {
            name: max(values) - min(values) if len(values) >= 3 else 0.0
            for name, values in samples.items()
        }
        selected, selected_range = max(ranges.items(), key=lambda item: item[1])
        if selected_range >= min_motion_deg:
            return selected
        print(
            f"No sensor moved enough (largest range {selected_range:.2f} deg; "
            f"need {min_motion_deg:.2f} deg). Try again.",
            flush=True,
        )


def assign_sides(
    sensors: Sequence[SensorCandidate],
    *,
    baudrate: int,
    sample_hz: float,
    min_motion_deg: float,
) -> dict[str, SensorCandidate]:
    if len(sensors) != 2:
        raise AngleTriggerError(f"Exactly two angle sensors are required; found {len(sensors)}.")
    temporary_mapping = {f"sensor_{index}": sensor for index, sensor in enumerate(sensors, start=1)}
    connections = {
        name: SensorConnection(sensor, baudrate=baudrate) for name, sensor in temporary_mapping.items()
    }
    try:
        for connection in connections.values():
            connection.open()
        left_name = _wait_for_side_motion(
            "left", connections, sample_hz=sample_hz, min_motion_deg=min_motion_deg
        )
        right_candidates = {name: connection for name, connection in connections.items() if name != left_name}
        right_name = _wait_for_side_motion(
            "right", right_candidates, sample_hz=sample_hz, min_motion_deg=min_motion_deg
        )
        return {"left": temporary_mapping[left_name], "right": temporary_mapping[right_name]}
    finally:
        for connection in connections.values():
            connection.close()


def _capture_endpoint(
    side: str,
    label: str,
    connection: SensorConnection,
    *,
    duration_s: float,
    sample_hz: float,
) -> tuple[float, float, int]:
    if not sys.stdin.isatty():
        raise AngleTriggerError("Angle calibration needs an interactive terminal.")
    input(f"Hold {side} at its intended {label} position, then press Enter to capture {duration_s:.1f}s. ")
    samples: list[float] = []
    deadline = time.monotonic() + duration_s
    period = 1.0 / sample_hz
    next_tick = time.monotonic()
    while time.monotonic() < deadline:
        samples.append(connection.read_angle_deg())
        next_tick = max(next_tick + period, time.monotonic())
        if next_tick < deadline:
            time.sleep(next_tick - time.monotonic())
    minimum_samples = max(3, int(duration_s * sample_hz * 0.6))
    if len(samples) < minimum_samples:
        raise AngleTriggerError(f"Only collected {len(samples)} samples for {side} {label}.")
    mean = sum(samples) / len(samples)
    variance = sum((sample - mean) ** 2 for sample in samples) / len(samples)
    print(f"{side} {label}: {mean:.2f} deg from {len(samples)} samples (std {math.sqrt(variance):.2f} deg).")
    return mean, math.sqrt(variance), len(samples)


def calibrate_angles(
    mapping: Mapping[str, SensorCandidate],
    *,
    baudrate: int,
    duration_s: float,
    sample_hz: float,
) -> dict[str, Any]:
    connections = _open_connections(mapping, baudrate=baudrate)
    profiles: dict[str, dict[str, float]] = {}
    try:
        for side in SIDES:
            zero_deg, zero_std_deg, zero_samples = _capture_endpoint(
                side, "0", connections[side], duration_s=duration_s, sample_hz=sample_hz
            )
            one_deg, one_std_deg, one_samples = _capture_endpoint(
                side, "1", connections[side], duration_s=duration_s, sample_hz=sample_hz
            )
            if abs(one_deg - zero_deg) < DEFAULT_ENDPOINT_MIN_SEPARATION_DEG:
                raise AngleTriggerError(
                    f"{side} 0 and 1 positions differ by less than "
                    f"{DEFAULT_ENDPOINT_MIN_SEPARATION_DEG:.1f} deg; calibration was not saved."
                )
            profiles[side] = {
                "zero_deg": zero_deg,
                "one_deg": one_deg,
                "zero_std_deg": zero_std_deg,
                "one_std_deg": one_std_deg,
                "zero_samples": float(zero_samples),
                "one_samples": float(one_samples),
            }
    finally:
        _close_connections(connections)
    return build_angle_calibration(profiles)


class AngleTriggerReader:
    """Read calibrated left/right angles as independently normalized 0-1 values."""

    def __init__(
        self,
        mapping: Mapping[str, SensorCandidate],
        calibration: Mapping[str, Any],
        *,
        baudrate: int = DEFAULT_BAUDRATE,
    ) -> None:
        self.mapping = dict(mapping)
        self.calibration = _normalize_angle_calibration(calibration)
        self.baudrate = baudrate
        self._connections: dict[str, SensorConnection] | None = None

    def open(self) -> None:
        if self._connections is None:
            self._connections = _open_connections(self.mapping, baudrate=self.baudrate)

    def close(self) -> None:
        if self._connections is not None:
            _close_connections(self._connections)
            self._connections = None

    def __enter__(self) -> "AngleTriggerReader":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def read(self) -> dict[str, float]:
        if self._connections is None:
            raise AngleTriggerError("AngleTriggerReader is not open.")
        values: dict[str, float] = {}
        for side in SIDES:
            profile = self.calibration["profiles"][side]
            angle_deg = self._connections[side].read_angle_deg()
            normalized = (angle_deg - profile["zero_deg"]) / (profile["one_deg"] - profile["zero_deg"])
            values[side] = max(0.0, min(1.0, normalized))
        return values


def _load_optional(path: Path, normalizer: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"No config exists at {path}."
    try:
        return normalizer(_read_json(path)), None
    except AngleTriggerError as exc:
        return None, str(exc)


def prepare_reader(
    *,
    config_root: Path | None = None,
    baudrate: int = DEFAULT_BAUDRATE,
    sample_hz: float = DEFAULT_SAMPLE_HZ,
    capture_seconds: float = DEFAULT_CAPTURE_SECONDS,
    assignment_motion_min_deg: float = DEFAULT_ASSIGNMENT_MOTION_MIN_DEG,
    reassign: bool = False,
    recalibrate_angles: bool = False,
) -> AngleTriggerReader:
    root = (config_root or default_config_root()).expanduser().resolve()
    assign_path = assignment_path(root)
    angles_path = angle_calibration_path(root)
    sensors = discover_sensors(baudrate=baudrate)
    if len(sensors) != 2:
        raise AngleTriggerError(f"Exactly two angle sensors are required; found {len(sensors)}.")

    assignment, assignment_error = _load_optional(assign_path, _normalize_assignment)
    mapping: dict[str, SensorCandidate] | None = None
    if assignment is not None and not reassign:
        mapping, errors = resolve_assignment(assignment, sensors)
        if errors:
            assignment_error = "; ".join(errors)
    if mapping is None:
        if assignment_error:
            print(f"Side assignment is required: {assignment_error}", file=sys.stderr)
        mapping = assign_sides(
            sensors,
            baudrate=baudrate,
            sample_hz=sample_hz,
            min_motion_deg=assignment_motion_min_deg,
        )
        assignment = build_assignment(mapping)
        _write_json(assign_path, assignment)
        print(f"Saved side assignment: {assign_path}")

    calibration, calibration_error = _load_optional(angles_path, _normalize_angle_calibration)
    if calibration is None or recalibrate_angles:
        if calibration_error:
            print(f"Angle calibration is required: {calibration_error}", file=sys.stderr)
        calibration = calibrate_angles(
            mapping,
            baudrate=baudrate,
            duration_s=capture_seconds,
            sample_hz=sample_hz,
        )
        _write_json(angles_path, calibration)
        print(f"Saved angle calibration: {angles_path}")
    return AngleTriggerReader(mapping, calibration, baudrate=baudrate)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover, calibrate, and read two USB angle triggers as left/right 0-1 sensors."
    )
    parser.add_argument("--config-root", type=Path, help="Directory holding assignment/ and angles/ configs.")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_SAMPLE_HZ, help="Read/display rate.")
    parser.add_argument("--capture-seconds", type=float, default=DEFAULT_CAPTURE_SECONDS)
    parser.add_argument("--reassign", action="store_true", help="Redo only left/right USB-port assignment.")
    parser.add_argument("--recalibrate-angles", action="store_true", help="Redo 0/1 angle endpoints.")
    parser.add_argument("--once", action="store_true", help="Print one calibrated reading and exit.")
    parser.add_argument("--list", action="store_true", help="List detected sensor candidates and exit.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.rate_hz <= 0.0 or args.capture_seconds <= 0.0:
        print("angle_trigger: --rate-hz and --capture-seconds must be positive.", file=sys.stderr)
        return 2
    try:
        if args.list:
            sensors = discover_sensors(baudrate=args.baudrate)
            if not sensors:
                print("No compatible angle sensors found.")
            for sensor in sensors:
                print(sensor.label)
            return 0
        with prepare_reader(
            config_root=args.config_root,
            baudrate=args.baudrate,
            sample_hz=args.rate_hz,
            capture_seconds=args.capture_seconds,
            reassign=args.reassign,
            recalibrate_angles=args.recalibrate_angles,
        ) as reader:
            if args.once:
                print(json.dumps(reader.read(), sort_keys=True))
                return 0
            print("Angle-trigger reader started. Press Ctrl-C to stop.")
            period = 1.0 / args.rate_hz
            while True:
                started = time.monotonic()
                values = reader.read()
                print(f"left={values['left']:.4f} right={values['right']:.4f}", end="\r", flush=True)
                remaining = period - (time.monotonic() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print()
        return 0
    except AngleTriggerError as exc:
        print(f"angle_trigger: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
