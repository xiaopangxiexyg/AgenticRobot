from __future__ import annotations

import copy
import json
import re
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from portable_dm_tac import discover_device_candidates, discover_devices

from .gripper_driver import InspireGripper, list_serial_devices


SCHEMA_VERSION = "dmtac_usb_assignment.v1"
SIDES = ("left", "right")
TACTILE_SLOTS = ("inner", "outer")
DEFAULT_GRIPPER_MOTION_AMPLITUDE = 0.3
DEFAULT_GRIPPER_TEST_SPEED = 0.1
DEFAULT_GRIPPER_TEST_TORQUE = 0.1
DEFAULT_GRIPPER_MAX_ACTION = 1.0
LATEST_CONFIG_NAME = "latest.json"
DEFAULT_GRIPPER_ID_SCAN_HINTS = (32, 46)
DEFAULT_GRIPPER_BAUDRATE_SCAN_ORDER = (115200, 57600, 19200)
_UNEXPECTED_SLAVE_ID_RE = re.compile(
    r"Unexpected slave id in response:\s*(\d+)\s*,\s*expect\s*(\d+)"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_config_dir() -> Path:
    try:
        import sys

        root = repo_root()
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from project_paths import hardware_interface_configs_root

        return hardware_interface_configs_root() / "dmtac"
    except Exception:
        return repo_root() / "teleop_runtime" / "configs" / "hardware" / "dmtac"


def legacy_config_dirs() -> tuple[Path, ...]:
    try:
        import sys

        root = repo_root()
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from project_paths import teleop_records_root

        return (teleop_records_root() / "dmtac_hardware_configs",)
    except Exception:
        return (repo_root() / "teleop_records" / "dmtac_hardware_configs",)


def hostname() -> str:
    return socket.gethostname()


def timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(dict(payload)), indent=2, ensure_ascii=True, sort_keys=True),
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


def _has_serial_number(port_info: Mapping[str, Any] | None) -> bool:
    if not isinstance(port_info, Mapping):
        return False
    return str(port_info.get("serial_number") or "").strip() not in {"", "None", "n/a"}


def _is_unstable_gripper_by_id(alias: str, port_info: Mapping[str, Any] | None = None) -> bool:
    path = str(alias or "")
    if not path.startswith("/dev/serial/by-id/"):
        return False
    name = Path(path).name
    if _has_serial_number(port_info):
        return False
    vid = str((port_info or {}).get("vid") or "").lower()
    pid = str((port_info or {}).get("pid") or "").lower()
    if {vid, pid} == {"6790", "29987"}:
        return True
    if name.startswith("usb-1a86_USB_Serial-") or name.startswith("usb-1a86_USB2.0-Serial-"):
        return True
    return name == "usb-1a86_USB_Serial-if00-port0"


def _stable_gripper_aliases(
    aliases: Sequence[str],
    port_info: Mapping[str, Any] | None = None,
) -> list[str]:
    return [
        str(alias)
        for alias in aliases
        if not _is_unstable_gripper_by_id(str(alias), port_info)
    ]


def _is_kernel_assigned_serial_path(path: str) -> bool:
    return str(path or "").startswith(("/dev/ttyUSB", "/dev/ttyACM"))


def _configured_gripper_match_aliases(
    configured: Mapping[str, Any],
    aliases: Sequence[str],
) -> set[str]:
    identity = configured.get("identity") if isinstance(configured.get("identity"), Mapping) else configured
    stable_aliases = _stable_gripper_aliases([str(alias) for alias in aliases], identity)
    topology_or_id_aliases = [
        alias
        for alias in stable_aliases
        if not _is_kernel_assigned_serial_path(alias)
    ]
    if topology_or_id_aliases:
        return set(topology_or_id_aliases)
    return set(stable_aliases)


def _preferred_port_from_aliases(
    aliases: Sequence[str],
    *,
    fallback: str = "",
    port_info: Mapping[str, Any] | None = None,
) -> str:
    stable_aliases = _stable_gripper_aliases(aliases, port_info)
    for prefix in ("/dev/serial/by-id/", "/dev/serial/by-path/"):
        for alias in aliases:
            if alias not in stable_aliases:
                continue
            if alias.startswith(prefix):
                return alias
    return str(fallback or "")


def preferred_port(port: str | None, port_info: Mapping[str, Any] | None = None) -> str:
    aliases = port_aliases(port)
    return _preferred_port_from_aliases(aliases, fallback=str(port or ""), port_info=port_info)


def _latest_config_path_in_dir(root: Path) -> Path | None:
    explicit = root / LATEST_CONFIG_NAME
    if explicit.exists():
        return explicit
    candidates = sorted(root.glob("*_dmtac_usb_assignment.json"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    return None


def latest_config_path(config_dir: str | Path | None = None) -> Path:
    if config_dir is not None:
        root = Path(config_dir).expanduser().resolve()
        path = _latest_config_path_in_dir(root)
        if path is not None:
            return path
        raise FileNotFoundError(f"No DMTac USB assignment config found in {root}.")

    searched = [default_config_dir(), *legacy_config_dirs()]
    for root in searched:
        path = _latest_config_path_in_dir(root)
        if path is not None:
            return path
    roots = ", ".join(str(root) for root in searched)
    raise FileNotFoundError(f"No DMTac USB assignment config found in {roots}.")


def load_assignment(path: str | Path = "latest", *, config_dir: str | Path | None = None) -> dict[str, Any]:
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
    path = root / f"{stamp}_dmtac_usb_assignment.json"
    payload = normalize_assignment(assignment)
    payload["saved_at"] = iso_timestamp()
    _write_json(path, payload)
    if update_latest:
        latest = root / LATEST_CONFIG_NAME
        latest_payload = dict(payload)
        latest_payload["source_config_path"] = str(path)
        _write_json(latest, latest_payload)
    return path


def merge_assignment(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = normalize_assignment(base)
    selected = normalize_assignment(override)
    merged["devices_seen"] = copy.deepcopy(selected.get("devices_seen", merged.get("devices_seen")))
    merged["merged_at"] = iso_timestamp()
    for side in SIDES:
        side_selected = selected["assignments"][side]
        side_merged = merged["assignments"][side]
        if side_selected.get("gripper") is not None:
            side_merged["gripper"] = copy.deepcopy(side_selected["gripper"])
        for slot in TACTILE_SLOTS:
            tactile = side_selected.get("tactile", {}).get(slot)
            if tactile is not None:
                side_merged["tactile"][slot] = copy.deepcopy(tactile)
    return normalize_assignment(merged)


def _looks_like_q6_serial_adapter(port: Mapping[str, Any]) -> bool:
    hwid = str(port.get("hwid") or "").upper()
    description = str(port.get("description") or "").strip().lower()
    product = str(port.get("product") or "").strip().lower()
    if "VID:PID=1A86:55D3" in hwid:
        return True
    return description == "usb single serial" or product == "usb single serial"


def _looks_like_gripper_serial(port: Mapping[str, Any]) -> bool:
    if _looks_like_q6_serial_adapter(port):
        return False
    text = " ".join(
        str(port.get(key) or "")
        for key in ("device", "description", "hwid", "manufacturer", "product")
    ).lower()
    keywords = (
        "ch340",
        "ch341",
        "1a86",
        "cp210",
        "silicon labs",
        "ftdi",
        "0403",
        "pl2303",
        "prolific",
        "067b",
        "usb serial",
        "usb-serial",
        "serial converter",
        "qinhen",
    )
    return any(keyword in text for keyword in keywords)


def _is_system_tty_port(port: str | None) -> bool:
    return str(port or "").startswith("/dev/ttyS")


def _is_probe_error_non_retryable(error: str | None) -> str | None:
    text = str(error or "").lower()
    if "permission denied" in text:
        return "permission_denied"
    if "device or resource busy" in text or "resource busy" in text:
        return "device_busy"
    if "no such file" in text or "no such device" in text:
        return "device_missing"
    if "could not open port" in text and "serialexception" in text:
        return "open_failed"
    return None


def _extract_unexpected_slave_id(error: str | None) -> int | None:
    match = _UNEXPECTED_SLAVE_ID_RE.search(str(error or ""))
    if match is None:
        return None
    detected_id = int(match.group(1))
    if 1 <= detected_id <= 247:
        return detected_id
    return None


def gripper_id_from_probe(probe: Mapping[str, Any] | None) -> int | None:
    if not isinstance(probe, Mapping):
        return None
    for key in ("detected_gripper_id", "gripper_id", "suggested_gripper_id"):
        value = probe.get(key)
        if value in (None, ""):
            continue
        try:
            gripper_id = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= gripper_id <= 247:
            return gripper_id
    return None


def gripper_baudrate_from_probe(probe: Mapping[str, Any] | None) -> int | None:
    if not isinstance(probe, Mapping):
        return None
    for key in ("detected_baudrate", "baudrate"):
        value = probe.get(key)
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def gripper_id_scan_order(requested_id: int = 1) -> list[int]:
    ids = [int(requested_id), *DEFAULT_GRIPPER_ID_SCAN_HINTS, *range(1, 248)]
    seen: set[int] = set()
    ordered: list[int] = []
    for gripper_id in ids:
        if not 1 <= int(gripper_id) <= 247:
            continue
        if int(gripper_id) in seen:
            continue
        seen.add(int(gripper_id))
        ordered.append(int(gripper_id))
    return ordered


def gripper_baudrate_scan_order(
    requested_baudrate: int = 115200,
    scan_baudrates: Sequence[int] | None = None,
) -> list[int]:
    ids = [
        int(requested_baudrate),
        *(int(value) for value in (() if scan_baudrates is None else scan_baudrates)),
    ]
    seen: set[int] = set()
    ordered: list[int] = []
    for baudrate in ids:
        if int(baudrate) in seen:
            continue
        seen.add(int(baudrate))
        ordered.append(int(baudrate))
    return ordered


def gripper_probe_order(
    *,
    requested_id: int = 1,
    requested_baudrate: int = 115200,
    scan_ids: bool = True,
    scan_baudrates: Sequence[int] | None = None,
) -> list[tuple[int, int]]:
    baudrates = gripper_baudrate_scan_order(requested_baudrate, scan_baudrates)
    quick_ids = []
    for value in (int(requested_id), *DEFAULT_GRIPPER_ID_SCAN_HINTS):
        if 1 <= int(value) <= 247 and int(value) not in quick_ids:
            quick_ids.append(int(value))
    full_ids = gripper_id_scan_order(requested_id) if bool(scan_ids) else [int(requested_id)]

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for baudrate in baudrates:
        for gripper_id in quick_ids:
            pair = (int(baudrate), int(gripper_id))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    for baudrate in baudrates:
        for gripper_id in full_ids:
            pair = (int(baudrate), int(gripper_id))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def _probe_gripper(
    port: str,
    *,
    gripper_id: int,
    baudrate: int,
    timeout: float,
    scan_ids: bool = True,
    scan_baudrates: Sequence[int] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    started_at = time.time()
    requested_id = int(gripper_id)
    requested_baudrate = int(baudrate)
    candidate_pairs = gripper_probe_order(
        requested_id=requested_id,
        requested_baudrate=requested_baudrate,
        scan_ids=bool(scan_ids),
        scan_baudrates=scan_baudrates,
    )
    total_pairs = len(candidate_pairs)
    attempted_ids: list[int] = []
    attempted_baudrates: list[int] = []
    attempted_pairs: list[dict[str, int]] = []
    suggested_ids: list[int] = []
    first_error: str | None = None
    last_error: str | None = None

    while candidate_pairs:
        current_baudrate, current_id = candidate_pairs.pop(0)
        attempted_ids.append(current_id)
        attempted_baudrates.append(current_baudrate)
        attempted_pairs.append({"baudrate": int(current_baudrate), "gripper_id": int(current_id)})
        attempt_no = len(attempted_pairs)
        if attempt_no == 1 or attempt_no % 16 == 0 or not candidate_pairs:
            _emit_scan_progress(
                progress_callback,
                {
                    "event": "gripper_setting_probe",
                    "port": str(port),
                    "attempt": int(attempt_no),
                    "total": int(max(total_pairs, attempt_no)),
                    "baudrate": int(current_baudrate),
                    "gripper_id": int(current_id),
                    "scan_ids": bool(scan_ids),
                    "scan_baudrates": list(scan_baudrates or ()),
                },
            )
        try:
            gripper_kwargs = {
                "port": port,
                "gripper_id": current_id,
                "baudrate": current_baudrate,
                "timeout": timeout,
            }
            try:
                gripper_ctx = InspireGripper(
                    **gripper_kwargs,
                    reconnect_on_error=False,
                )
            except TypeError as exc:
                if "reconnect_on_error" not in str(exc):
                    raise
                gripper_ctx = InspireGripper(**gripper_kwargs)
            with gripper_ctx as gripper:
                state = gripper.get_state()
            result = {
                "reachable": True,
                "probe_elapsed_s": round(time.time() - started_at, 6),
                "state": state.to_dict(),
                "error": None,
                "requested_gripper_id": requested_id,
                "requested_baudrate": requested_baudrate,
                "gripper_id": current_id,
                "detected_gripper_id": current_id,
                "attempted_gripper_ids": list(attempted_ids),
                "attempted_gripper_id_count": len(attempted_ids),
                "attempted_baudrates": list(attempted_baudrates),
                "attempted_settings": list(attempted_pairs),
                "attempted_setting_count": len(attempted_pairs),
                "id_scan_enabled": bool(scan_ids),
                "baudrate_scan_enabled": len(set(attempted_baudrates)) > 1,
                "id_autodetected": current_id != requested_id,
                "baudrate": int(current_baudrate),
                "detected_baudrate": int(current_baudrate),
                "timeout": float(timeout),
            }
            if first_error:
                result["id_probe_error"] = first_error
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if first_error is None:
                first_error = last_error
            stop_reason = _is_probe_error_non_retryable(last_error)
            if stop_reason is not None:
                return {
                    "reachable": False,
                    "probe_elapsed_s": round(time.time() - started_at, 6),
                    "state": None,
                    "error": last_error,
                    "requested_gripper_id": requested_id,
                    "requested_baudrate": requested_baudrate,
                    "attempted_gripper_ids": list(attempted_ids),
                    "attempted_gripper_id_count": len(attempted_ids),
                    "attempted_baudrates": list(attempted_baudrates),
                    "attempted_settings": list(attempted_pairs),
                    "attempted_setting_count": len(attempted_pairs),
                    "id_scan_enabled": bool(scan_ids),
                    "baudrate_scan_enabled": len(set(attempted_baudrates)) > 1,
                    "probe_stopped_reason": stop_reason,
                    "baudrate": int(requested_baudrate),
                    "timeout": float(timeout),
                }
            detected_id = _extract_unexpected_slave_id(last_error)
            if detected_id is not None and detected_id not in suggested_ids:
                suggested_ids.append(detected_id)
            suggested_pair = (int(current_baudrate), int(detected_id)) if detected_id is not None else None
            if (
                suggested_pair is not None
                and suggested_pair not in {(item["baudrate"], item["gripper_id"]) for item in attempted_pairs}
                and suggested_pair not in candidate_pairs
            ):
                candidate_pairs.insert(0, suggested_pair)

    return {
        "reachable": False,
        "probe_elapsed_s": round(time.time() - started_at, 6),
        "state": None,
        "error": last_error or first_error,
        "requested_gripper_id": requested_id,
        "requested_baudrate": requested_baudrate,
        "attempted_gripper_ids": list(attempted_ids),
        "attempted_gripper_id_count": len(attempted_ids),
        "attempted_baudrates": list(attempted_baudrates),
        "attempted_settings": list(attempted_pairs),
        "attempted_setting_count": len(attempted_pairs),
        "id_scan_enabled": bool(scan_ids),
        "baudrate_scan_enabled": len(set(attempted_baudrates)) > 1,
        "suggested_gripper_id": suggested_ids[-1] if suggested_ids else None,
        "suggested_gripper_ids": list(suggested_ids),
        "baudrate": int(requested_baudrate),
        "timeout": float(timeout),
    }


def _emit_scan_progress(
    callback: Callable[[Mapping[str, Any]], None] | None,
    payload: Mapping[str, Any],
) -> None:
    if callback is None:
        return
    try:
        callback(dict(payload))
    except Exception:
        return


def scan_usb_devices(
    *,
    probe_grippers: bool = True,
    probe_all_serial: bool = False,
    probe_system_ttys: bool = False,
    gripper_id: int = 1,
    baudrate: int = 115200,
    timeout: float = 0.03,
    scan_gripper_ids: bool = True,
    scan_baudrates: Sequence[int] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    serial_ports = []
    for index, item in enumerate(list_serial_devices()):
        port = dict(item)
        port["index"] = int(index)
        port["port"] = str(port.get("device") or "")
        port["port_aliases"] = port_aliases(port["port"])
        port["preferred_port"] = _preferred_port_from_aliases(
            port["port_aliases"],
            fallback=port["port"],
            port_info=port,
        )
        port["known_q6_serial"] = _looks_like_q6_serial_adapter(port)
        port["probable_gripper_serial"] = _looks_like_gripper_serial(port)
        system_tty = _is_system_tty_port(port["port"])
        port["system_tty"] = bool(system_tty)
        should_probe = bool(
            probe_grippers
            and not (system_tty and not bool(probe_system_ttys))
            and not port["known_q6_serial"]
            and (probe_all_serial or port["probable_gripper_serial"])
        )
        port["probe_attempted"] = should_probe
        port["probe"] = None
        serial_ports.append(port)

    probe_ports = [port for port in serial_ports if bool(port.get("probe_attempted")) and port.get("port")]
    _emit_scan_progress(
        progress_callback,
        {
            "event": "serial_scan_ready",
            "serial_count": int(len(serial_ports)),
            "probe_count": int(len(probe_ports)),
            "probe_all_serial": bool(probe_all_serial),
        },
    )
    probe_index = 0
    for port in serial_ports:
        should_probe = bool(port.get("probe_attempted") and port.get("port"))
        if should_probe and port["port"]:
            probe_index += 1
            _emit_scan_progress(
                progress_callback,
                {
                    "event": "gripper_probe_start",
                    "index": int(probe_index),
                    "total": int(len(probe_ports)),
                    "port": str(port.get("port") or ""),
                    "preferred_port": str(port.get("preferred_port") or ""),
                    "description": str(port.get("description") or ""),
                    "product": str(port.get("product") or ""),
                    "hwid": str(port.get("hwid") or ""),
                    "scan_gripper_ids": bool(scan_gripper_ids),
                    "probe_all_serial": bool(probe_all_serial),
                },
            )
            port["probe"] = _probe_gripper(
                port["port"],
                gripper_id=int(gripper_id),
                baudrate=int(baudrate),
                timeout=float(timeout),
                scan_ids=bool(scan_gripper_ids),
                scan_baudrates=scan_baudrates,
                progress_callback=lambda payload, port=port, probe_index=probe_index: _emit_scan_progress(
                    progress_callback,
                    {
                        **dict(payload),
                        "index": int(probe_index),
                        "total_ports": int(len(probe_ports)),
                        "preferred_port": str(port.get("preferred_port") or ""),
                    },
                ),
            )
            probe = port.get("probe") if isinstance(port.get("probe"), Mapping) else {}
            _emit_scan_progress(
                progress_callback,
                {
                    "event": "gripper_probe_done",
                    "index": int(probe_index),
                    "total": int(len(probe_ports)),
                    "port": str(port.get("port") or ""),
                    "preferred_port": str(port.get("preferred_port") or ""),
                    "reachable": bool(probe.get("reachable")),
                    "error": probe.get("error"),
                    "probe_elapsed_s": probe.get("probe_elapsed_s"),
                    "attempted_setting_count": probe.get("attempted_setting_count"),
                },
            )

    tactile_devices = []
    for index, item in enumerate(discover_devices()):
        record = dict(item) if isinstance(item, Mapping) else {"raw": str(item)}
        dev_id = record.get("serial")
        if dev_id in (None, ""):
            dev_id = record.get("device_index")
        if dev_id in (None, ""):
            dev_id = str(item)
        record["index"] = int(index)
        record["dev_id"] = str(dev_id)
        tactile_devices.append(record)
    tactile_candidates = discover_device_candidates()

    return {
        "schema_version": "dmtac_usb_scan.v1",
        "scanned_at": iso_timestamp(),
        "hostname": hostname(),
        "grippers": serial_ports,
        "tactile": tactile_devices,
        "tactile_candidates": tactile_candidates,
    }


def empty_side_assignment() -> dict[str, Any]:
    return {
        "gripper": None,
        "tactile": {slot: None for slot in TACTILE_SLOTS},
    }


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
        side_payload = assignments.get(side, {}) if isinstance(assignments, Mapping) else {}
        if not isinstance(side_payload, Mapping):
            side_payload = {}
        side_out = empty_side_assignment()
        side_out["gripper"] = _normalize_gripper_assignment(side_payload.get("gripper"))
        side_out["tactile"] = _normalize_tactile_assignment(side_payload.get("tactile"))
        out_assignments[side] = side_out
    normalized["assignments"] = out_assignments
    return normalized


def _normalize_gripper_assignment(value: Any) -> dict[str, Any] | None:
    if value in (None, "", False):
        return None
    if not isinstance(value, Mapping):
        raise ValueError("gripper assignment must be an object or null.")
    out = dict(value)
    port = out.get("port") or out.get("device")
    if not port:
        raise ValueError("gripper assignment requires port.")
    out["enabled"] = bool(out.get("enabled", True))
    out["port"] = str(port)
    out["gripper_id"] = int(out.get("gripper_id", 1))
    out["baudrate"] = int(out.get("baudrate", 115200))
    out["timeout"] = float(out.get("timeout", 0.03))
    out["identity"] = dict(out.get("identity") or {})
    out["port_aliases"] = _unique_text(tuple(out.get("port_aliases") or (str(port),)))
    return out


def _normalize_tactile_assignment(value: Any) -> dict[str, Any]:
    out = {slot: None for slot in TACTILE_SLOTS}
    if value in (None, "", False):
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for slot, item in zip(TACTILE_SLOTS, value):
            out[slot] = _normalize_tactile_slot(item, slot)
        return out
    if not isinstance(value, Mapping):
        raise ValueError("tactile assignment must be an object, array, or null.")
    for slot in TACTILE_SLOTS:
        out[slot] = _normalize_tactile_slot(value.get(slot), slot)
    return out


def _normalize_tactile_slot(value: Any, slot: str) -> dict[str, Any] | None:
    if value in (None, "", False):
        return None
    if isinstance(value, Mapping):
        dev_id = value.get("dev_id") or value.get("serial") or value.get("device_id")
        identity = dict(value.get("identity") or {})
    else:
        dev_id = value
        identity = {}
    if dev_id in (None, ""):
        raise ValueError(f"tactile slot {slot} requires dev_id.")
    return {
        "enabled": True,
        "slot": str(slot),
        "dev_id": str(dev_id),
        "identity": identity,
    }


def _token_text(token: Any) -> str:
    return str(token).strip()


def resolve_gripper_token(scan: Mapping[str, Any], token: str | int | None) -> dict[str, Any] | None:
    if token in (None, ""):
        return None
    text = _token_text(token)
    ports = list(scan.get("grippers") or [])
    for item in ports:
        candidates = {
            str(item.get("index")),
            str(item.get("port")),
            str(item.get("device")),
            str(item.get("preferred_port")),
            *(str(alias) for alias in item.get("port_aliases") or ()),
        }
        if text in candidates:
            return dict(item)
    raise ValueError(f"Unknown gripper token {token!r}. Use scan index or port path.")


def resolve_tactile_token(scan: Mapping[str, Any], token: str | int | None) -> dict[str, Any] | None:
    if token in (None, ""):
        return None
    text = _token_text(token)
    devices = list(scan.get("tactile") or [])
    for item in devices:
        if text == str(item.get("index")) or text == str(item.get("dev_id")) or text == str(item.get("serial")):
            return dict(item)
    raise ValueError(f"Unknown tactile token {token!r}. Use scan index or dev_id.")


def _gripper_assignment_from_scan(item: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if item is None:
        return None
    probe = item.get("probe") if isinstance(item.get("probe"), Mapping) else {}
    detected_gripper_id = gripper_id_from_probe(probe) or 1
    detected_baudrate = gripper_baudrate_from_probe(probe) or 115200
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
        "gripper_id": int(detected_gripper_id),
        "baudrate": int(detected_baudrate),
        "timeout": 0.03,
        "identity": identity,
        "port_aliases": list(item.get("port_aliases") or ()),
        "probe": copy.deepcopy(probe),
    }


def _tactile_assignment_from_scan(
    item: Mapping[str, Any] | None,
    *,
    slot: str,
) -> dict[str, Any] | None:
    if item is None:
        return None
    identity = {
        key: item.get(key)
        for key in ("dev_id", "serial", "device_index", "model", "name")
        if item.get(key) not in (None, "")
    }
    return {
        "enabled": True,
        "slot": str(slot),
        "dev_id": str(item.get("dev_id")),
        "identity": identity,
    }


def build_assignment_from_tokens(
    scan: Mapping[str, Any],
    *,
    left_gripper: str | int | None = None,
    right_gripper: str | int | None = None,
    left_inner: str | int | None = None,
    left_outer: str | int | None = None,
    right_inner: str | int | None = None,
    right_outer: str | int | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": iso_timestamp(),
        "hostname": hostname(),
        "devices_seen": copy.deepcopy(dict(scan)),
        "assignments": {side: empty_side_assignment() for side in SIDES},
    }
    token_map = {
        "left": {
            "gripper": left_gripper,
            "inner": left_inner,
            "outer": left_outer,
        },
        "right": {
            "gripper": right_gripper,
            "inner": right_inner,
            "outer": right_outer,
        },
    }
    for side, tokens in token_map.items():
        payload["assignments"][side]["gripper"] = _gripper_assignment_from_scan(
            resolve_gripper_token(scan, tokens["gripper"])
        )
        for slot in TACTILE_SLOTS:
            payload["assignments"][side]["tactile"][slot] = _tactile_assignment_from_scan(
                resolve_tactile_token(scan, tokens[slot]),
                slot=slot,
            )
    return normalize_assignment(payload)


def gripper_devices_for_auto_assignment(
    scan: Mapping[str, Any],
    *,
    require_reachable: bool = True,
) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for item in scan.get("grippers") or []:
        record = dict(item)
        probe = record.get("probe")
        if bool(require_reachable):
            if not (isinstance(probe, Mapping) and probe.get("reachable") is True):
                continue
        elif not record.get("probable_gripper_serial") and not isinstance(probe, Mapping):
            continue
        devices.append(record)
    return devices


def build_gripper_auto_assignment_tokens(
    scan: Mapping[str, Any],
    *,
    swap: bool = False,
    single_side: str = "right",
    require_reachable: bool = True,
) -> dict[str, str | None]:
    side = str(single_side).strip().lower()
    if side not in SIDES:
        raise ValueError("single_side must be left or right.")
    devices = gripper_devices_for_auto_assignment(scan, require_reachable=require_reachable)
    if len(devices) > 2:
        raise ValueError(
            f"Expected at most 2 grippers for automatic assignment, detected {len(devices)}. "
            "Disconnect extras or pass --left-gripper/--right-gripper explicitly."
        )
    if not devices:
        return {"left": None, "right": None}
    token = lambda item: str(item.get("index"))
    if len(devices) == 1:
        return {
            "left": token(devices[0]) if side == "left" else None,
            "right": token(devices[0]) if side == "right" else None,
        }
    left_item, right_item = (devices[1], devices[0]) if bool(swap) else (devices[0], devices[1])
    return {"left": token(left_item), "right": token(right_item)}


def configured_gripper_sides(assignment: Mapping[str, Any]) -> tuple[str, ...]:
    payload = normalize_assignment(assignment)
    sides = []
    for side in SIDES:
        if payload["assignments"][side]["gripper"] is not None:
            sides.append(side)
    return tuple(sides)


def configured_tactile_sides(assignment: Mapping[str, Any]) -> tuple[str, ...]:
    payload = normalize_assignment(assignment)
    sides = []
    for side in SIDES:
        tactile = payload["assignments"][side]["tactile"]
        if any(tactile.get(slot) is not None for slot in TACTILE_SLOTS):
            sides.append(side)
    return tuple(sides)


def tactile_ids_for_side(
    assignment: Mapping[str, Any],
    side: str,
    *,
    slots: Sequence[str] | None = None,
) -> tuple[str, ...]:
    payload = normalize_assignment(assignment)
    slot_names = tuple(TACTILE_SLOTS if slots is None else slots)
    tactile = payload["assignments"][str(side)]["tactile"]
    ids = []
    for slot in slot_names:
        if slot not in TACTILE_SLOTS:
            raise ValueError(f"Unsupported tactile slot {slot!r}; expected inner/outer.")
        item = tactile.get(slot)
        if item is not None:
            ids.append(str(item["dev_id"]))
    return tuple(ids)


def gripper_for_side(assignment: Mapping[str, Any], side: str) -> dict[str, Any] | None:
    return normalize_assignment(assignment)["assignments"][str(side)]["gripper"]


def _identity_matches(config_identity: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    serial = config_identity.get("serial_number")
    if serial and serial == current.get("serial_number"):
        return True
    hwid = config_identity.get("hwid")
    if hwid and hwid == current.get("hwid"):
        return True
    location = config_identity.get("location")
    if location and location == current.get("location"):
        return True
    return False


def _find_current_gripper(
    configured: Mapping[str, Any],
    scan: Mapping[str, Any],
) -> dict[str, Any] | None:
    ports = list(scan.get("grippers") or [])
    identity = configured.get("identity") or {}
    for item in ports:
        if identity and _identity_matches(identity, item):
            return dict(item)
    configured_port = str(configured.get("port") or "")
    configured_aliases = _configured_gripper_match_aliases(
        configured,
        [str(alias) for alias in configured.get("port_aliases") or ()],
    )
    if not _is_unstable_gripper_by_id(
        configured_port,
        configured.get("identity") if isinstance(configured.get("identity"), Mapping) else configured,
    ) and not (
        _is_kernel_assigned_serial_path(configured_port)
        and any(not _is_kernel_assigned_serial_path(alias) for alias in configured_aliases)
    ):
        configured_aliases.add(configured_port)
    for item in ports:
        raw_candidates = [
            str(item.get("port") or ""),
            str(item.get("device") or ""),
            str(item.get("preferred_port") or ""),
            *(str(alias) for alias in item.get("port_aliases") or ()),
        ]
        candidates = set(_stable_gripper_aliases(raw_candidates, item))
        if configured_aliases.intersection(candidates):
            return dict(item)
    return None


def _find_current_tactile(configured: Mapping[str, Any], scan: Mapping[str, Any]) -> dict[str, Any] | None:
    dev_id = str(configured.get("dev_id") or "")
    for item in scan.get("tactile") or []:
        if dev_id and dev_id in {str(item.get("dev_id")), str(item.get("serial"))}:
            return dict(item)
    return None


def compare_assignment_to_scan(
    assignment: Mapping[str, Any],
    scan: Mapping[str, Any],
) -> dict[str, Any]:
    payload = normalize_assignment(assignment)
    missing = []
    matched = []
    remapped = []
    assigned_gripper_ports = set()
    assigned_tactile_ids = set()
    effective = copy.deepcopy(payload)

    for side in SIDES:
        configured_gripper = payload["assignments"][side]["gripper"]
        if configured_gripper is not None:
            current = _find_current_gripper(configured_gripper, scan)
            if current is None:
                missing.append({"kind": "gripper", "side": side, "configured": configured_gripper})
            else:
                assigned_gripper_ports.add(str(current.get("port") or current.get("device")))
                probe = current.get("probe") or {}
                if probe.get("reachable") is False:
                    missing.append({
                        "kind": "gripper",
                        "side": side,
                        "configured": configured_gripper,
                        "current": current,
                        "reason": "probe_failed",
                    })
                else:
                    matched.append({"kind": "gripper", "side": side, "configured": configured_gripper, "current": current})
                    current_port = str(current.get("preferred_port") or current.get("port") or current.get("device"))
                    if current_port and current_port != configured_gripper.get("port"):
                        remapped.append({
                            "kind": "gripper",
                            "side": side,
                            "from": configured_gripper.get("port"),
                            "to": current_port,
                        })
                        effective["assignments"][side]["gripper"]["port"] = current_port
                    detected_gripper_id = gripper_id_from_probe(current.get("probe"))
                    if detected_gripper_id is not None:
                        effective["assignments"][side]["gripper"]["gripper_id"] = detected_gripper_id
                    detected_baudrate = gripper_baudrate_from_probe(current.get("probe"))
                    if detected_baudrate is not None:
                        effective["assignments"][side]["gripper"]["baudrate"] = detected_baudrate

        for slot in TACTILE_SLOTS:
            configured_tactile = payload["assignments"][side]["tactile"].get(slot)
            if configured_tactile is None:
                continue
            current = _find_current_tactile(configured_tactile, scan)
            if current is None:
                missing.append({
                    "kind": "tactile",
                    "side": side,
                    "slot": slot,
                    "configured": configured_tactile,
                })
            else:
                assigned_tactile_ids.add(str(current.get("dev_id")))
                matched.append({
                    "kind": "tactile",
                    "side": side,
                    "slot": slot,
                    "configured": configured_tactile,
                    "current": current,
                })

    extra = []
    for item in scan.get("grippers") or []:
        probe = item.get("probe") or {}
        if probe.get("reachable") is True:
            port = str(item.get("port") or item.get("device"))
            if port not in assigned_gripper_ports:
                extra.append({"kind": "gripper", "current": item})
    for item in scan.get("tactile") or []:
        dev_id = str(item.get("dev_id"))
        if dev_id not in assigned_tactile_ids:
            extra.append({"kind": "tactile", "current": item})

    return {
        "missing": missing,
        "extra": extra,
        "matched": matched,
        "remapped": remapped,
        "effective_assignment": normalize_assignment(effective),
    }


def format_comparison(comparison: Mapping[str, Any]) -> str:
    lines = []
    for item in comparison.get("missing") or []:
        if item.get("kind") == "gripper":
            configured = item.get("configured") or {}
            lines.append(
                f"[DMTAC CONFIG ERROR] Missing configured {item.get('side')} gripper: "
                f"{configured.get('port')}"
            )
        else:
            configured = item.get("configured") or {}
            lines.append(
                f"[DMTAC CONFIG ERROR] Missing configured {item.get('side')} "
                f"{item.get('slot')} tactile sensor: {configured.get('dev_id')}"
            )
    for item in comparison.get("extra") or []:
        current = item.get("current") or {}
        if item.get("kind") == "gripper":
            lines.append(
                "[DMTAC CONFIG WARNING] Extra unassigned gripper candidate detected: "
                f"{current.get('port') or current.get('device')}"
            )
        else:
            lines.append(
                "[DMTAC CONFIG WARNING] Extra unassigned tactile sensor detected: "
                f"{current.get('dev_id')}"
            )
    for item in comparison.get("remapped") or []:
        lines.append(
            f"[DMTAC CONFIG WARNING] {item.get('side')} {item.get('kind')} path changed: "
            f"{item.get('from')} -> {item.get('to')}"
        )
    return "\n".join(lines)


def require_compatible_or_confirm(
    comparison: Mapping[str, Any],
    *,
    allow_missing: bool = False,
    assume_yes: bool = False,
    input_func: Callable[[str], str] = input,
) -> None:
    text = format_comparison(comparison)
    if text:
        print(text)
    missing = list(comparison.get("missing") or [])
    extra = list(comparison.get("extra") or [])
    if missing and not allow_missing:
        raise RuntimeError("Configured DMTac devices are missing; refusing real-hardware run.")
    if extra:
        if assume_yes:
            return
        answer = input_func("Extra DMTac USB devices were detected. Continue? Type YES: ")
        if answer.strip() != "YES":
            raise RuntimeError("User did not confirm extra DMTac USB devices.")


def gripper_motion_window(
    center: float,
    *,
    amplitude: float = DEFAULT_GRIPPER_MOTION_AMPLITUDE,
    max_action: float = DEFAULT_GRIPPER_MAX_ACTION,
) -> tuple[float, float]:
    amplitude = max(0.0, min(float(amplitude), float(max_action)))
    half = amplitude * 0.5
    lo = float(center) - half
    hi = float(center) + half
    if lo < 0.0:
        hi = min(float(max_action), hi - lo)
        lo = 0.0
    if hi > float(max_action):
        lo = max(0.0, lo - (hi - float(max_action)))
        hi = float(max_action)
    return lo, hi


def run_gripper_motion_test(
    port: str,
    *,
    gripper_id: int = 1,
    baudrate: int = 115200,
    timeout: float = 0.03,
    amplitude: float = DEFAULT_GRIPPER_MOTION_AMPLITUDE,
    speed: float = DEFAULT_GRIPPER_TEST_SPEED,
    torque: float = DEFAULT_GRIPPER_TEST_TORQUE,
    max_action: float = DEFAULT_GRIPPER_MAX_ACTION,
    wait_timeout: float = 3.0,
) -> dict[str, Any]:
    with InspireGripper(
        port=str(port),
        gripper_id=int(gripper_id),
        baudrate=int(baudrate),
        timeout=float(timeout),
    ) as gripper:
        gripper.clear_fault()
        gripper.set_speed(float(speed))
        gripper.set_torque(float(torque))
        start = gripper.get_state()
        lo, hi = gripper_motion_window(
            float(start.joint_pos),
            amplitude=float(amplitude),
            max_action=float(max_action),
        )
        gripper.set_joint_pos(lo, speed=float(speed), torque=float(torque), continuous=False)
        opened = gripper.wait_until_idle(float(wait_timeout))
        gripper.set_joint_pos(hi, speed=float(speed), torque=float(torque), continuous=False)
        closed = gripper.wait_until_idle(float(wait_timeout))
        gripper.set_joint_pos(float(start.joint_pos), speed=float(speed), torque=float(torque), continuous=False)
        restored = gripper.wait_until_idle(float(wait_timeout))
    return {
        "port": str(port),
        "amplitude": float(amplitude),
        "targets": {"open": lo, "close": hi, "restore": float(start.joint_pos)},
        "start": start.to_dict(),
        "opened": opened.to_dict(),
        "closed": closed.to_dict(),
        "restored": restored.to_dict(),
    }
