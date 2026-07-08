from __future__ import annotations

import json
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from project_paths import hardware_interface_configs_root, teleop_records_root


SCHEMA_VERSION = "rm65_arm_ip_assignment.v1"
LATEST_CONFIG_NAME = "latest.json"
SIDES = ("left", "right")


def default_config_dir() -> Path:
    return hardware_interface_configs_root() / "rm65_arm_ips"


def legacy_config_dirs() -> tuple[Path, ...]:
    return (teleop_records_root() / "rm65_arm_ip_configs",)


def hostname() -> str:
    return socket.gethostname()


def iso_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def empty_assignment() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": iso_timestamp(),
        "hostname": hostname(),
        "assignments": {
            "left": {"arm_ip": ""},
            "right": {"arm_ip": ""},
        },
    }


def normalize_assignment(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["schema_version"] = str(normalized.get("schema_version") or SCHEMA_VERSION)
    normalized.setdefault("created_at", iso_timestamp())
    normalized.setdefault("hostname", hostname())

    raw_assignments = normalized.get("assignments")
    if not isinstance(raw_assignments, Mapping):
        raw_assignments = {}
    arm_ip_by_side = normalized.get("arm_ip_by_side")
    if not isinstance(arm_ip_by_side, Mapping):
        arm_ip_by_side = {}

    assignments: dict[str, dict[str, str]] = {}
    for side in SIDES:
        side_payload = raw_assignments.get(side)
        if not isinstance(side_payload, Mapping):
            side_payload = {}
        flat_key = f"{side}_arm_ip"
        arm_ip = (
            side_payload.get("arm_ip")
            or arm_ip_by_side.get(side)
            or normalized.get(flat_key)
            or ""
        )
        assignments[side] = {"arm_ip": str(arm_ip).strip()}
    normalized["assignments"] = assignments
    normalized["arm_ip_by_side"] = {side: assignments[side]["arm_ip"] for side in SIDES}
    normalized["left_arm_ip"] = assignments["left"]["arm_ip"]
    normalized["right_arm_ip"] = assignments["right"]["arm_ip"]
    return normalized


def build_assignment(
    *,
    left_arm_ip: str | None = None,
    right_arm_ip: str | None = None,
    base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = normalize_assignment(base or empty_assignment())
    if left_arm_ip is not None:
        payload["assignments"]["left"]["arm_ip"] = str(left_arm_ip).strip()
    if right_arm_ip is not None:
        payload["assignments"]["right"]["arm_ip"] = str(right_arm_ip).strip()
    return normalize_assignment(payload)


def _latest_config_path_in_dir(root: Path) -> Path | None:
    explicit = root / LATEST_CONFIG_NAME
    if explicit.exists():
        return explicit
    candidates = sorted(root.glob("*_rm65_arm_ip_assignment.json"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    return None


def latest_config_path(config_dir: str | Path | None = None) -> Path:
    if config_dir is not None:
        root = Path(config_dir).expanduser().resolve()
        path = _latest_config_path_in_dir(root)
        if path is not None:
            return path
        raise FileNotFoundError(f"No RM65 arm IP assignment config found in {root}.")

    searched = [default_config_dir(), *legacy_config_dirs()]
    for root in searched:
        path = _latest_config_path_in_dir(root)
        if path is not None:
            return path
    roots = ", ".join(str(root) for root in searched)
    raise FileNotFoundError(f"No RM65 arm IP assignment config found in {roots}.")


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


def _save_root_for_config(config: str | Path = "latest") -> Path:
    text = str(config).strip()
    if not text or text.lower() == "latest":
        return default_config_dir()
    path = Path(text).expanduser()
    if path.suffix:
        return path.parent.resolve()
    return path.resolve()


def save_assignment(
    assignment: Mapping[str, Any],
    *,
    config_dir: str | Path | None = None,
    update_latest: bool = True,
) -> Path:
    root = Path(config_dir).expanduser().resolve() if config_dir is not None else default_config_dir()
    stamp = timestamp_for_filename()
    path = root / f"{stamp}_rm65_arm_ip_assignment.json"
    payload = normalize_assignment(assignment)
    payload["saved_at"] = iso_timestamp()
    _write_json(path, payload)
    if update_latest:
        latest = root / LATEST_CONFIG_NAME
        latest_payload = dict(payload)
        latest_payload["source_config_path"] = str(path)
        _write_json(latest, latest_payload)
    return path


def save_assignment_for_config(
    assignment: Mapping[str, Any],
    *,
    config: str | Path = "latest",
    update_latest: bool = True,
) -> Path:
    return save_assignment(assignment, config_dir=_save_root_for_config(config), update_latest=update_latest)


def arm_ips_for_sides(
    assignment: Mapping[str, Any],
    *,
    sides: Sequence[str] = SIDES,
) -> dict[str, str]:
    payload = normalize_assignment(assignment)
    result: dict[str, str] = {}
    for side in sides:
        side_name = str(side)
        if side_name not in SIDES:
            raise ValueError(f"Unsupported RM65 side {side_name!r}.")
        arm_ip = str(payload["assignments"][side_name].get("arm_ip") or "").strip()
        if arm_ip:
            result[side_name] = arm_ip
    return result
