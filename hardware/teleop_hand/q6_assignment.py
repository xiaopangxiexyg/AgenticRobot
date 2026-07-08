from __future__ import annotations

import argparse
import copy
import json
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import serial
from serial.tools import list_ports

from project_paths import hardware_interface_configs_root, teleop_records_root

from .hand_q6_reader import BufferedSerialLineReader, OFFSET, Q6_ORDER, parse_angles_line


SCHEMA_VERSION = "hand_q6_assignment.v1"
SCAN_SCHEMA_VERSION = "hand_q6_scan.v1"
LATEST_CONFIG_NAME = "latest.json"
SIDES = ("left", "right")
DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT = 0.05
DEFAULT_PROBE_TIMEOUT_S = 1.5
DEFAULT_VISUALIZE_SEC = 20.0
DEFAULT_PREVIEW_REFRESH_HZ = 20.0
DEFAULT_PREVIEW_LOG_HZ = 5.0


def default_config_dir() -> Path:
    return hardware_interface_configs_root() / "q6"


def legacy_config_dirs() -> tuple[Path, ...]:
    return (teleop_records_root() / "q6_hardware_configs",)


def hostname() -> str:
    return socket.gethostname()


def iso_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(dict(payload)), indent=2, ensure_ascii=True, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _unique_text(values: Sequence[object]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def port_aliases(port: str | None) -> list[str]:
    if not port:
        return []
    aliases: list[str] = [str(port)]
    realpath: str | None = None
    try:
        path = Path(str(port))
        if path.exists():
            realpath = str(path.resolve())
            aliases.append(realpath)
    except OSError:
        realpath = None
    if realpath is None:
        return _unique_text(aliases)

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
    return _unique_text(aliases)


def preferred_port(port: str | None) -> str:
    aliases = port_aliases(port)
    for prefix in ("/dev/serial/by-id/", "/dev/serial/by-path/"):
        for alias in aliases:
            if alias.startswith(prefix):
                return alias
    return str(port or "")


def _serial_port_record(item: Any, index: int) -> dict[str, Any]:
    device = str(getattr(item, "device", "") or "")
    record = {
        "index": int(index),
        "device": device,
        "port": device,
        "preferred_port": preferred_port(device),
        "name": str(getattr(item, "name", "") or ""),
        "description": str(getattr(item, "description", "") or ""),
        "hwid": str(getattr(item, "hwid", "") or ""),
        "manufacturer": getattr(item, "manufacturer", None),
        "product": getattr(item, "product", None),
        "serial_number": getattr(item, "serial_number", None),
        "location": getattr(item, "location", None),
        "interface": getattr(item, "interface", None),
        "port_aliases": port_aliases(device),
    }
    record["probable_q6_serial"] = _looks_like_q6_serial(record)
    return record


def _looks_like_q6_serial(port: Mapping[str, Any]) -> bool:
    hwid = str(port.get("hwid") or "").upper()
    description = str(port.get("description") or "").strip().lower()
    product = str(port.get("product") or "").strip().lower()
    serial_number = str(port.get("serial_number") or "").strip()
    if (
        "VID:PID=1A86:7523" in hwid
        and "SER=" not in hwid
        and not serial_number
        and (description == "usb serial" or product == "usb serial")
    ):
        return False
    text = " ".join(
        str(port.get(key) or "")
        for key in (
            "device",
            "name",
            "description",
            "hwid",
            "manufacturer",
            "product",
            "interface",
        )
    ).lower()
    keywords = (
        "q6",
        "hand",
        "glove",
        "ttyacm",
        "usb single serial",
        "usb-single-serial",
        "1a86",
        "ch340",
        "ch341",
    )
    return any(keyword in text for keyword in keywords)


def list_serial_devices() -> list[dict[str, Any]]:
    return [
        _serial_port_record(item, index)
        for index, item in enumerate(list_ports.comports())
    ]


def _read_angles_until(
    reader: BufferedSerialLineReader,
    *,
    deadline: float,
) -> list[float] | None:
    return reader.read_latest_parsed_until(parse_angles_line, deadline=float(deadline))


def probe_q6_port(
    port: str,
    *,
    baudrate: int = DEFAULT_BAUDRATE,
    timeout: float = DEFAULT_TIMEOUT,
    probe_timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
    min_valid_frames: int = 1,
    show_raw_lines: bool = False,
) -> dict[str, Any]:
    started_at = time.time()
    frames: list[list[float]] = []
    try:
        ser = serial.Serial(port=str(port), baudrate=int(baudrate), timeout=float(timeout))
    except Exception as exc:  # noqa: BLE001
        return {
            "reachable": False,
            "valid_frames": 0,
            "latest_angles_deg": None,
            "probe_elapsed_s": round(time.time() - started_at, 6),
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        reset = getattr(ser, "reset_input_buffer", None)
        if callable(reset):
            reset()
        reader = BufferedSerialLineReader(
            ser,
            show_raw_lines=bool(show_raw_lines),
            raw_prefix=f"[RAW:{port}] ",
        )
        deadline = started_at + float(probe_timeout_s)
        while time.time() < deadline and len(frames) < max(1, int(min_valid_frames)):
            parsed = _read_angles_until(
                reader,
                deadline=deadline,
            )
            if parsed is not None:
                frames.append(parsed)
        return {
            "reachable": bool(frames),
            "valid_frames": int(len(frames)),
            "latest_angles_deg": None if not frames else list(frames[-1]),
            "probe_elapsed_s": round(time.time() - started_at, 6),
            "error": None if frames else "no ANGLES frame observed",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "reachable": False,
            "valid_frames": int(len(frames)),
            "latest_angles_deg": None if not frames else list(frames[-1]),
            "probe_elapsed_s": round(time.time() - started_at, 6),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        try:
            ser.close()
        except Exception:
            pass


ProbeFunc = Callable[..., dict[str, Any]]


def scan_q6_devices(
    *,
    baudrate: int = DEFAULT_BAUDRATE,
    timeout: float = DEFAULT_TIMEOUT,
    probe_timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
    probe_all_serial: bool = False,
    show_raw_lines: bool = False,
    serial_devices: Sequence[Mapping[str, Any]] | None = None,
    probe_func: ProbeFunc = probe_q6_port,
) -> dict[str, Any]:
    serial_ports = [
        dict(item)
        for item in (
            list_serial_devices()
            if serial_devices is None
            else serial_devices
        )
    ]
    q6_devices: list[dict[str, Any]] = []
    for index, item in enumerate(serial_ports):
        item.setdefault("index", int(index))
        item.setdefault("port", str(item.get("device") or ""))
        item.setdefault("preferred_port", preferred_port(str(item.get("port") or "")))
        item.setdefault("port_aliases", port_aliases(str(item.get("port") or "")))
        item.setdefault("probable_q6_serial", _looks_like_q6_serial(item))
        should_probe = bool(probe_all_serial or item.get("probable_q6_serial"))
        item["probe_attempted"] = should_probe
        if should_probe and item.get("port"):
            probe = probe_func(
                str(item["port"]),
                baudrate=int(baudrate),
                timeout=float(timeout),
                probe_timeout_s=float(probe_timeout_s),
                show_raw_lines=bool(show_raw_lines),
            )
        else:
            probe = None
        item["probe"] = probe
        if isinstance(probe, Mapping) and probe.get("reachable"):
            device = copy.deepcopy(item)
            device["q6_index"] = int(len(q6_devices))
            q6_devices.append(device)

    return {
        "schema_version": SCAN_SCHEMA_VERSION,
        "scanned_at": iso_timestamp(),
        "hostname": hostname(),
        "baudrate": int(baudrate),
        "timeout": float(timeout),
        "probe_timeout_s": float(probe_timeout_s),
        "serial_ports": serial_ports,
        "q6_devices": q6_devices,
    }


def _normalize_side_assignment(value: Any) -> dict[str, Any] | None:
    if value in (None, "", False):
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Q6 side assignment must be an object or null.")
    out = dict(value)
    port = out.get("port") or out.get("preferred_port") or out.get("device")
    if not port:
        raise ValueError("Q6 side assignment requires port.")
    out["enabled"] = bool(out.get("enabled", True))
    out["port"] = str(port)
    out["baudrate"] = int(out.get("baudrate", DEFAULT_BAUDRATE))
    out["timeout"] = float(out.get("timeout", DEFAULT_TIMEOUT))
    out["identity"] = dict(out.get("identity") or {})
    out["port_aliases"] = _unique_text(tuple(out.get("port_aliases") or (str(port),)))
    out["q6_order"] = list(out.get("q6_order") or Q6_ORDER)
    return out


def normalize_assignment(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.setdefault("schema_version", SCHEMA_VERSION)
    normalized.setdefault("created_at", iso_timestamp())
    normalized.setdefault("hostname", hostname())
    assignments = normalized.get("assignments")
    if not isinstance(assignments, Mapping):
        assignments = {}
    out_assignments = {}
    for side in SIDES:
        out_assignments[side] = _normalize_side_assignment(assignments.get(side))
    normalized["assignments"] = out_assignments
    return normalized


def _assignment_from_scan_item(
    item: Mapping[str, Any] | None,
    *,
    baudrate: int,
    timeout: float,
) -> dict[str, Any] | None:
    if item is None:
        return None
    identity = {
        key: item.get(key)
        for key in (
            "description",
            "hwid",
            "manufacturer",
            "product",
            "serial_number",
            "location",
            "interface",
        )
        if item.get(key) not in (None, "")
    }
    return {
        "enabled": True,
        "port": str(item.get("preferred_port") or item.get("port") or item.get("device")),
        "baudrate": int(baudrate),
        "timeout": float(timeout),
        "identity": identity,
        "port_aliases": list(item.get("port_aliases") or ()),
        "q6_order": list(Q6_ORDER),
        "probe": copy.deepcopy(item.get("probe")),
    }


def resolve_q6_device_token(scan: Mapping[str, Any], token: str | int | None) -> dict[str, Any] | None:
    if token in (None, ""):
        return None
    text = str(token).strip()
    for item in scan.get("q6_devices") or []:
        if text in {
            str(item.get("q6_index")),
            str(item.get("index")),
            str(item.get("port")),
            str(item.get("device")),
            str(item.get("preferred_port")),
        }:
            return dict(item)
    raise ValueError(f"Unknown Q6 token {token!r}. Use Q6 index, serial index, or port path.")


def build_assignment_from_tokens(
    scan: Mapping[str, Any],
    *,
    left: str | int | None = None,
    right: str | int | None = None,
    baudrate: int | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    resolved_baudrate = int(baudrate if baudrate is not None else scan.get("baudrate", DEFAULT_BAUDRATE))
    resolved_timeout = float(timeout if timeout is not None else scan.get("timeout", DEFAULT_TIMEOUT))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": iso_timestamp(),
        "hostname": hostname(),
        "devices_seen": copy.deepcopy(dict(scan)),
        "assignments": {
            "left": _assignment_from_scan_item(
                resolve_q6_device_token(scan, left),
                baudrate=resolved_baudrate,
                timeout=resolved_timeout,
            ),
            "right": _assignment_from_scan_item(
                resolve_q6_device_token(scan, right),
                baudrate=resolved_baudrate,
                timeout=resolved_timeout,
            ),
        },
    }
    return normalize_assignment(payload)


def _latest_config_path_in_dir(root: Path) -> Path | None:
    explicit = root / LATEST_CONFIG_NAME
    if explicit.exists():
        return explicit
    candidates = sorted(root.glob("*_hand_q6_assignment.json"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    return None


def latest_config_path(config_dir: str | Path | None = None) -> Path:
    if config_dir is not None:
        root = Path(config_dir).expanduser().resolve()
        path = _latest_config_path_in_dir(root)
        if path is not None:
            return path
        raise FileNotFoundError(f"No hand Q6 assignment config found in {root}.")

    searched = [default_config_dir(), *legacy_config_dirs()]
    for root in searched:
        path = _latest_config_path_in_dir(root)
        if path is not None:
            return path
    roots = ", ".join(str(root) for root in searched)
    raise FileNotFoundError(f"No hand Q6 assignment config found in {roots}.")


def load_assignment(
    path: str | Path = "latest",
    *,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    if str(path).strip().lower() == "latest":
        resolved = latest_config_path(config_dir)
    else:
        resolved = Path(path).expanduser().resolve()
    payload = _read_json(resolved)
    payload["_config_path"] = str(resolved)
    return normalize_assignment(payload)


def save_assignment(
    assignment: Mapping[str, Any],
    *,
    config_dir: str | Path | None = None,
    update_latest: bool = True,
) -> Path:
    root = Path(config_dir).expanduser().resolve() if config_dir is not None else default_config_dir()
    stamp = timestamp_for_filename()
    path = root / f"{stamp}_hand_q6_assignment.json"
    payload = normalize_assignment(assignment)
    payload["saved_at"] = iso_timestamp()
    _write_json(path, payload)
    if update_latest:
        latest = root / LATEST_CONFIG_NAME
        latest_payload = dict(payload)
        latest_payload["source_config_path"] = str(path)
        _write_json(latest, latest_payload)
    return path


def q6_ports_for_sides(
    assignment: Mapping[str, Any],
    *,
    active_sides: Sequence[str] = SIDES,
) -> dict[str, str]:
    payload = normalize_assignment(assignment)
    ports: dict[str, str] = {}
    for side in active_sides:
        side_name = str(side)
        if side_name not in SIDES:
            raise ValueError(f"Unsupported Q6 side {side_name!r}.")
        item = payload["assignments"].get(side_name)
        if item is not None and bool(item.get("enabled", True)):
            ports[side_name] = str(item["port"])
    return ports


def load_latest_q6_ports(
    *,
    active_sides: Sequence[str] = SIDES,
    config: str | Path = "latest",
    config_dir: str | Path | None = None,
    allow_missing: bool = True,
) -> dict[str, str]:
    try:
        assignment = load_assignment(config, config_dir=config_dir)
    except FileNotFoundError:
        if allow_missing:
            return {}
        raise
    return q6_ports_for_sides(assignment, active_sides=active_sides)


def _identity_matches(configured: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    identity = configured.get("identity") or {}
    for key in ("serial_number", "hwid", "location"):
        value = identity.get(key)
        if value and str(value) == str(current.get(key)):
            return True
    configured_aliases = set(str(alias) for alias in configured.get("port_aliases") or ())
    current_aliases = set(str(alias) for alias in current.get("port_aliases") or ())
    current_aliases.add(str(current.get("port") or ""))
    current_aliases.add(str(current.get("preferred_port") or ""))
    return bool(configured_aliases.intersection(current_aliases))


def _find_current_device(configured: Mapping[str, Any], scan: Mapping[str, Any]) -> dict[str, Any] | None:
    configured_port = str(configured.get("port") or "")
    for item in scan.get("serial_ports") or []:
        candidates = {
            str(item.get("port") or ""),
            str(item.get("device") or ""),
            str(item.get("preferred_port") or ""),
            *(str(alias) for alias in item.get("port_aliases") or ()),
        }
        if configured_port and configured_port in candidates:
            return dict(item)
    for item in scan.get("serial_ports") or []:
        if _identity_matches(configured, item):
            return dict(item)
    return None


def compare_assignment_to_scan(
    assignment: Mapping[str, Any],
    scan: Mapping[str, Any],
) -> dict[str, Any]:
    payload = normalize_assignment(assignment)
    effective = copy.deepcopy(payload)
    missing = []
    matched = []
    remapped = []
    for side in SIDES:
        configured = payload["assignments"].get(side)
        if configured is None:
            continue
        current = _find_current_device(configured, scan)
        if current is None:
            missing.append({"kind": "hand_q6", "side": side, "configured": configured})
            continue
        matched.append({"kind": "hand_q6", "side": side, "configured": configured, "current": current})
        current_port = str(current.get("preferred_port") or current.get("port") or current.get("device"))
        if current_port and current_port != configured.get("port"):
            effective["assignments"][side]["port"] = current_port
            remapped.append({
                "kind": "hand_q6",
                "side": side,
                "from": configured.get("port"),
                "to": current_port,
            })
    return {
        "missing": missing,
        "matched": matched,
        "remapped": remapped,
        "effective_assignment": normalize_assignment(effective),
    }


def format_scan_summary(scan: Mapping[str, Any]) -> str:
    lines = [
        f"Detected Q6 devices: {len(scan.get('q6_devices') or [])}",
    ]
    for item in scan.get("q6_devices") or []:
        probe = item.get("probe") or {}
        lines.append(
            "  "
            f"q6_index={item.get('q6_index')} "
            f"serial_index={item.get('index')} "
            f"port={item.get('preferred_port') or item.get('port')} "
            f"frames={probe.get('valid_frames')} "
            f"angles={probe.get('latest_angles_deg')}"
        )
    attempted = [
        item
        for item in scan.get("serial_ports") or []
        if item.get("probe_attempted")
    ]
    if not attempted:
        lines.append(
            "No serial ports matched the Q6 candidate filter. "
            "Try `scan --probe-all-serial` if the device uses an unexpected USB descriptor."
        )
        return "\n".join(lines)

    failed_attempts = [
        item
        for item in attempted
        if not (
            isinstance(item.get("probe"), Mapping)
            and item.get("probe", {}).get("reachable")
        )
    ]
    if not failed_attempts:
        return "\n".join(lines)

    if scan.get("q6_devices"):
        lines.append("Other Q6 candidate probe attempts failed:")
    else:
        lines.append("No reachable Q6 devices. Probe attempts:")
    permission_denied = False
    for item in failed_attempts:
        probe = item.get("probe") or {}
        error = str(probe.get("error") or "not probed")
        if "permission denied" in error.lower() or "errno 13" in error.lower():
            permission_denied = True
        lines.append(
            "  "
            f"serial_index={item.get('index')} "
            f"port={item.get('preferred_port') or item.get('port')} "
            f"probable_q6={bool(item.get('probable_q6_serial'))} "
            f"error={error}"
        )
    if permission_denied:
        lines.append(
            "Hint: serial permission denied. Check `groups` and `ls -l /dev/ttyACM*`; "
            "add the current user to the serial device group such as dialout, then re-login/replug."
        )
    return "\n".join(lines)


def _q6_preview_value(angles: Sequence[float], baseline: Sequence[float]) -> np.ndarray:
    denom = np.maximum(np.abs(np.asarray(OFFSET, dtype=np.float32)), 1e-6)
    values = np.abs(np.asarray(angles, dtype=np.float32) - np.asarray(baseline, dtype=np.float32)) / denom
    return np.clip(values, 0.0, 1.0)


def _stream_is_tty(stream: Any) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(callable(isatty) and isatty())


def _preview_vector_text(vector: np.ndarray | None) -> str:
    if vector is None:
        return "waiting"
    return np.array2string(vector, precision=3, separator=", ", max_line_width=120)


def _preview_device_rows(
    devices: Sequence[Mapping[str, Any]],
    latest: Mapping[str, Sequence[float]],
    baselines: Mapping[str, Sequence[float]],
    peak_scores: dict[str, float],
    peak_vectors: dict[str, np.ndarray],
    read_stats: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in devices:
        key = str(item.get("q6_index"))
        baseline = baselines.get(key)
        angles = latest.get(key)
        stats = dict((read_stats or {}).get(key) or {})
        vector: np.ndarray | None = None
        score = 0.0
        if baseline is not None and angles is not None:
            vector = _q6_preview_value(angles, baseline)
            score = float(np.max(vector))
            if score >= float(peak_scores.get(key, -1.0)):
                peak_scores[key] = score
                peak_vectors[key] = vector
        rows.append(
            {
                "key": key,
                "port": item.get("preferred_port") or item.get("port"),
                "vector": vector,
                "score": score,
                "peak_score": float(peak_scores.get(key, 0.0)),
                "drop": int(stats.get("stale_frames_dropped", 0) or 0),
            }
        )
    return rows


def visualize_q6_devices(
    devices: Sequence[Mapping[str, Any]],
    *,
    baudrate: int = DEFAULT_BAUDRATE,
    timeout: float = DEFAULT_TIMEOUT,
    duration_s: float = DEFAULT_VISUALIZE_SEC,
    refresh_hz: float = DEFAULT_PREVIEW_REFRESH_HZ,
    log_hz: float = DEFAULT_PREVIEW_LOG_HZ,
    live: bool | None = None,
) -> None:
    if not devices or float(duration_s) <= 0.0:
        return
    use_live = _stream_is_tty(sys.stdout) if live is None else bool(live)
    refresh_interval = 1.0 / max(0.1, float(refresh_hz))
    log_interval = None if float(log_hz) <= 0.0 else 1.0 / max(0.1, float(log_hz))
    serials: list[tuple[Mapping[str, Any], Any, BufferedSerialLineReader]] = []
    try:
        for item in devices:
            port = str(item.get("port") or item.get("device"))
            ser = serial.Serial(port=port, baudrate=int(baudrate), timeout=float(timeout))
            reset = getattr(ser, "reset_input_buffer", None)
            if callable(reset):
                reset()
            reader = BufferedSerialLineReader(ser)
            serials.append((item, ser, reader))
        baselines: dict[str, list[float]] = {}
        latest: dict[str, list[float]] = {}
        read_stats: dict[str, dict[str, Any]] = {}
        peak_scores: dict[str, float] = {str(item.get("q6_index")): 0.0 for item, _ser, _reader in serials}
        peak_vectors: dict[str, np.ndarray] = {}
        print("Q6 preview: move one Q6 at a time to identify left/right. Ctrl-C skips preview.", flush=True)
        if not use_live:
            print("Q6 preview output is throttled because stdout is not an interactive TTY.", flush=True)
            for item, _ser, _reader in serials:
                print(
                    f"  q6_index={item.get('q6_index')} port={item.get('preferred_port') or item.get('port')}",
                    flush=True,
                )
        deadline = time.time() + 2.0
        for item, _ser, reader in serials:
            key = str(item.get("q6_index"))
            angles = _read_angles_until(reader, deadline=deadline)
            if angles is not None:
                baselines[key] = angles
                latest[key] = angles
                read_stats[key] = dict(reader.last_read_stats)
        started_at = time.time()
        end_at = started_at + float(duration_s)
        next_log_at = started_at
        while time.time() < end_at:
            for item, _ser, reader in serials:
                key = str(item.get("q6_index"))
                angles = _read_angles_until(reader, deadline=time.time() + float(timeout))
                if angles is not None:
                    latest[key] = angles
                    read_stats[key] = dict(reader.last_read_stats)
            rows = _preview_device_rows(
                [item for item, _ser, _reader in serials],
                latest,
                baselines,
                peak_scores,
                peak_vectors,
                read_stats,
            )
            now = time.time()
            if use_live:
                print("\033[2J\033[H", end="")
                print("Move one Q6 at a time to identify left/right. Ctrl-C skips preview.")
                print("q6_index | port | now | peak | drop | approx q6 absolute-delta")
                for row in rows:
                    print(
                        f"{row['key']:>8} | {row['port']} | "
                        f"{row['score']:.3f} | {row['peak_score']:.3f} | {row['drop']} | "
                        f"{_preview_vector_text(row['vector'])}"
                    )
                sys.stdout.flush()
            elif log_interval is not None and now >= next_log_at:
                elapsed = max(0.0, now - started_at)
                parts = [
                    f"{row['key']}:now={row['score']:.3f},peak={row['peak_score']:.3f},drop={row['drop']}"
                    for row in rows
                ]
                print(f"Q6 preview t={elapsed:04.1f}s | " + " | ".join(parts), flush=True)
                next_log_at = now + log_interval
            time.sleep(refresh_interval)

        print("Q6 preview peak movement:", flush=True)
        for item, _ser, _reader in serials:
            key = str(item.get("q6_index"))
            vector = peak_vectors.get(key)
            print(
                f"  q6_index={key} port={item.get('preferred_port') or item.get('port')} "
                f"peak={float(peak_scores.get(key, 0.0)):.3f} vector={_preview_vector_text(vector)}",
                flush=True,
            )
    finally:
        for _item, ser, _reader in serials:
            try:
                ser.close()
            except Exception:
                pass


def _prompt_assignment_for_two(default_swap: bool = False) -> bool:
    default_text = "swap" if default_swap else "keep"
    try:
        answer = input(
            "Assign q6_index 0 -> left and q6_index 1 -> right. "
            f"Press Enter to {default_text}, type 's' to swap: "
        ).strip().lower()
    except EOFError:
        return bool(default_swap)
    if not answer:
        return bool(default_swap)
    return answer in {"s", "swap", "y", "yes"}


def _prompt_single_side(default_side: str = "right") -> str:
    try:
        answer = input(
            f"Only one Q6 detected. Assign it to left or right? [{default_side}]: "
        ).strip().lower()
    except EOFError:
        return str(default_side)
    if not answer:
        answer = default_side
    if answer not in SIDES:
        raise ValueError("single Q6 side must be left or right.")
    return answer


def _noninteractive_assignment_error() -> RuntimeError:
    return RuntimeError(
        "Left/right Q6 confirmation requires interactive stdin. "
        "Use `conda run --no-capture-output ...` for the configure command, "
        "or pass --yes with optional --swap/--left/--right to select non-interactively."
    )


def _configure_from_args(args: argparse.Namespace) -> int:
    scan = scan_q6_devices(
        baudrate=int(args.baudrate),
        timeout=float(args.timeout),
        probe_timeout_s=float(args.probe_timeout_s),
        probe_all_serial=bool(args.probe_all_serial),
        show_raw_lines=bool(args.show_raw_lines),
    )
    print(format_scan_summary(scan), flush=True)
    devices = list(scan.get("q6_devices") or [])
    if len(devices) > 2:
        raise RuntimeError(
            f"Expected at most 2 Q6 devices, detected {len(devices)}. "
            "Disconnect extras or pass explicit --left/--right tokens after inspection."
        )
    if not devices:
        raise RuntimeError(
            "No reachable Q6 devices detected. See probe details above; "
            "if the error is Permission denied, fix serial device permissions first."
        )

    left_token = args.left
    right_token = args.right
    if len(devices) == 2 and left_token is None and right_token is None:
        if not bool(args.yes) and not _stream_is_tty(sys.stdin):
            raise _noninteractive_assignment_error()
        if float(args.visualize_sec) > 0.0:
            try:
                visualize_q6_devices(
                    devices,
                    baudrate=int(args.baudrate),
                    timeout=float(args.timeout),
                    duration_s=float(args.visualize_sec),
                    log_hz=float(args.preview_log_hz),
                )
            except KeyboardInterrupt:
                print("\nPreview skipped.", flush=True)
        swap = bool(args.swap)
        if not bool(args.yes):
            swap = _prompt_assignment_for_two(default_swap=swap)
        left_token, right_token = ("1", "0") if swap else ("0", "1")
    elif len(devices) == 1 and left_token is None and right_token is None:
        side = str(args.single_side)
        if not bool(args.yes):
            if not _stream_is_tty(sys.stdin):
                raise _noninteractive_assignment_error()
            side = _prompt_single_side(side)
        if side == "left":
            left_token = "0"
        else:
            right_token = "0"

    assignment = build_assignment_from_tokens(
        scan,
        left=left_token,
        right=right_token,
        baudrate=int(args.baudrate),
        timeout=float(args.timeout),
    )
    ports = q6_ports_for_sides(assignment)
    if not ports:
        raise RuntimeError("No Q6 side assignment was selected.")
    print("Selected Q6 assignment:", flush=True)
    for side in SIDES:
        print(f"  {side}: {ports.get(side)}", flush=True)

    if not bool(args.no_save):
        path = save_assignment(assignment, config_dir=args.config_dir)
        print(f"Saved Q6 assignment: {path}", flush=True)
        print(f"Updated latest: {path.parent / LATEST_CONFIG_NAME}", flush=True)
    return 0


def _scan_from_args(args: argparse.Namespace) -> int:
    scan = scan_q6_devices(
        baudrate=int(args.baudrate),
        timeout=float(args.timeout),
        probe_timeout_s=float(args.probe_timeout_s),
        probe_all_serial=bool(args.probe_all_serial),
        show_raw_lines=bool(args.show_raw_lines),
    )
    if bool(args.json):
        print(json.dumps(_jsonable(scan), indent=2, ensure_ascii=True, sort_keys=True))
    else:
        print(format_scan_summary(scan))
    return 0


def _show_from_args(args: argparse.Namespace) -> int:
    assignment = load_assignment(args.config, config_dir=args.config_dir)
    print(json.dumps(_jsonable(assignment), indent=2, ensure_ascii=True, sort_keys=True))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect and assign hand Q6 serial devices.")
    sub = parser.add_subparsers(dest="command")

    def add_probe_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
        p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
        p.add_argument("--probe-timeout-s", type=float, default=DEFAULT_PROBE_TIMEOUT_S)
        p.add_argument("--probe-all-serial", action="store_true")
        p.add_argument("--show-raw-lines", action="store_true")

    scan_p = sub.add_parser("scan", help="Probe serial ports and print detected Q6 devices.")
    add_probe_args(scan_p)
    scan_p.add_argument("--json", action="store_true")
    scan_p.set_defaults(func=_scan_from_args)

    cfg_p = sub.add_parser("configure", help="Detect Q6 devices, choose left/right, and save latest config.")
    add_probe_args(cfg_p)
    cfg_p.add_argument("--left", default=None, help="Q6 token for left side: q6 index, serial index, or port.")
    cfg_p.add_argument("--right", default=None, help="Q6 token for right side: q6 index, serial index, or port.")
    cfg_p.add_argument("--swap", action="store_true", help="Swap default two-device assignment.")
    cfg_p.add_argument("--single-side", choices=SIDES, default="right")
    cfg_p.add_argument("--visualize-sec", type=float, default=DEFAULT_VISUALIZE_SEC)
    cfg_p.add_argument("--preview-log-hz", type=float, default=DEFAULT_PREVIEW_LOG_HZ)
    cfg_p.add_argument("--yes", action="store_true", help="Use defaults without interactive prompts.")
    cfg_p.add_argument("--no-save", action="store_true")
    cfg_p.add_argument("--config-dir", default=None)
    cfg_p.set_defaults(func=_configure_from_args)

    show_p = sub.add_parser("show", help="Print saved Q6 assignment config.")
    show_p.add_argument("--config", default="latest")
    show_p.add_argument("--config-dir", default=None)
    show_p.set_defaults(func=_show_from_args)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv or raw_argv[0].startswith("-"):
        raw_argv.insert(0, "configure")
    parser = build_arg_parser()
    args = parser.parse_args(raw_argv)
    if not hasattr(args, "func"):
        parser.error("missing command")
    try:
        return int(args.func(args))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
