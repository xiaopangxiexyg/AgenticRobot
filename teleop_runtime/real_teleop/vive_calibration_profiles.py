"""VIVE calibration profile discovery, persistence, and runtime loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from project_paths import calibration_profiles_dir

from ..calibration import TeleopCalibration
from ..log_utils import log_event
from ..sources.pose import VivePoseSource
from .components import RealTeleopComponents
from .movement import movement_multiple_value as _movement_multiple_value
from .side_mapping import (
    tracker_source_side_for_logical_side as _tracker_source_side_for_logical_side,
)
from .workspace import (
    apply_workspace_config as _apply_workspace_config,
    apply_workspace_overrides as _apply_workspace_overrides,
    arm_workspace_config_path as _arm_workspace_config_path,
)


DEFAULT_CALIBRATION_PROFILE_DIR = calibration_profiles_dir()


def _profile_identity(
    path: Path, data: bytes, calibration: TeleopCalibration
) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "offset_role_binding": calibration.metadata.get(
            "tracker_offset_role_binding", "legacy_unmarked"
        ),
        "saved_tracker_input_mapping": calibration.metadata.get(
            "tracker_input_mapping"
        ),
        "saved_movement_multiple": calibration.metadata.get("movement_multiple"),
    }


def load_vive_calibration_profile(
    path: str | Path, *, expected_sha256: str | None = None
) -> TeleopCalibration:
    """Load and identify the same bytes without restoring operator settings."""
    path = Path(path).expanduser()
    data = path.read_bytes()
    if expected_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(
                "Expected VIVE profile SHA256 must be 64 lowercase hexadecimal characters."
            )
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError(
                "VIVE calibration profile SHA256 differs from the approved startup binding."
            )
    calibration = TeleopCalibration.from_dict(json.loads(data))
    if calibration.metadata.get("tracker_offset_role_binding") not in (
        None,
        "source_role",
    ):
        raise ValueError(
            "Real teleop requires source-role tracker offsets, not a logical-channel coordinate profile."
        )
    identity = _profile_identity(path, data, calibration)
    calibration.metadata["runtime_vive_profile"] = identity
    log_event(
        "ViveCalibration", "loaded_profile=" + json.dumps(identity, sort_keys=True)
    )
    return calibration


def vive_configuration_metadata(
    args: argparse.Namespace,
    calibration: TeleopCalibration | None,
    active_sides: tuple[str, ...],
    sources: tuple[object, ...] = (),
) -> dict[str, Any]:
    """Snapshot selected configuration; physical pairing still needs live data."""
    metadata = {} if calibration is None else calibration.metadata
    actual_mapping = next(
        (source.source_role_by_output_channel for source in sources if isinstance(source, VivePoseSource)),
        None,
    )
    return {
        "profile": metadata.get("runtime_vive_profile"),
        "source_role_by_output_channel": actual_mapping,
        "tracker_input_mapping": {
            side: _tracker_source_side_for_logical_side(args, side)
            for side in active_sides
        },
        "swap_operator_sides": bool(getattr(args, "swap_operator_sides", False)),
        "swap_arm_sides": bool(getattr(args, "swap_arm_sides", False)),
        "movement_multiple": _movement_multiple_value(args),
        "physical_pairing_verified": False,
    }


def _calibration_profile_latest_sort_key(path: Path) -> tuple[int, str, float, str]:
    match = re.match(r"^(\d{8}_\d{6})", path.name)
    timestamp_key = match.group(1) if match else ""
    return (
        1 if timestamp_key else 0,
        timestamp_key,
        path.stat().st_mtime,
        path.name,
    )


def _latest_vive_calibration_profile_path(
    *,
    profile_dir: Path | None = None,
) -> Path | None:
    resolved_profile_dir = (
        DEFAULT_CALIBRATION_PROFILE_DIR
        if profile_dir is None
        else Path(profile_dir)
    )
    if not resolved_profile_dir.exists():
        return None
    candidates: list[Path] = []
    seen: set[Path] = set()
    for path in resolved_profile_dir.glob("*.json"):
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=_calibration_profile_latest_sort_key)


def _gui_vive_calibration_load_path(
    args: argparse.Namespace,
    *,
    profile_dir: Path | None = None,
) -> Path:
    explicit = str(getattr(args, "load_profile", "") or "").strip()
    if explicit:
        return Path(os.path.expandvars(explicit)).expanduser()
    last_saved = str(getattr(args, "gui_last_vive_calibration_profile", "") or "").strip()
    if last_saved:
        return Path(os.path.expandvars(last_saved)).expanduser()
    resolved_profile_dir = (
        DEFAULT_CALIBRATION_PROFILE_DIR
        if profile_dir is None
        else Path(profile_dir)
    )
    latest = _latest_vive_calibration_profile_path(
        profile_dir=resolved_profile_dir,
    )
    if latest is None:
        raise FileNotFoundError(
            f"No VIVE calibration profile found in {resolved_profile_dir}"
        )
    return latest


def _runtime_home_was_available(calibration: TeleopCalibration) -> bool:
    if calibration.captured_home_qpos:
        return True
    for arm in calibration.arms.values():
        if arm.home_qpos is not None or arm.home_pose_for_tcp_mode() is not None:
            return True
    return False


def _calibration_profile_payload_without_workspace(
    calibration: TeleopCalibration,
    *,
    workspace_config_path: Path | None,
) -> dict[str, Any]:
    payload = calibration.to_dict()
    for arm_payload in payload.get("arms", {}).values():
        if not isinstance(arm_payload, dict):
            continue
        arm_payload.pop("workspace_min", None)
        arm_payload.pop("workspace_max", None)
        metadata = arm_payload.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("workspace_config_source", None)
            metadata.pop("workspace_override_runtime", None)
    metadata = payload.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata.pop("runtime_vive_profile", None)
        metadata.pop("workspace_config_runtime", None)
        metadata.pop("workspace_override_runtime", None)
        metadata["workspace_policy"] = "managed_by_teleop_runtime_config"
        metadata["workspace_config_path"] = (
            None if workspace_config_path is None else str(workspace_config_path)
        )
    return payload

def _save_gui_vive_calibration_profile(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    profile_dir: Path | None = None,
) -> Path:
    calibration = components.session.calibration
    if calibration is None:
        raise RuntimeError("No active calibration is attached to the teleop session.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    resolved_profile_dir = (
        DEFAULT_CALIBRATION_PROFILE_DIR
        if profile_dir is None
        else Path(profile_dir)
    )
    output_path = resolved_profile_dir / f"{timestamp}_vive_rm65_real_teleop.json"
    profile = TeleopCalibration.from_dict(calibration.to_dict())
    runtime_home_was_available = _runtime_home_was_available(profile)
    profile.clear_home_state()
    profile.metadata.update(
        {
            "profile_scope": "tracker_station_only",
            "runtime_home_used_for_visual_check": runtime_home_was_available,
            "home_policy": (
                "session home anchors are intentionally not saved; real teleop must recapture wrist home"
            ),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "saved_from_entrypoint": "teleop_runtime.debug.run_vive_rm65_real_teleop",
            "saved_from_gui": True,
            "active_sides": list(components.active_sides),
            "movement_multiple": _movement_multiple_value(args),
            "translation_scale": _movement_multiple_value(args),
            "tracker_input_mapping": {
                side: _tracker_source_side_for_logical_side(args, side)
                for side in components.active_sides
            },
        }
    )
    payload = _calibration_profile_payload_without_workspace(
        profile,
        workspace_config_path=_arm_workspace_config_path(args),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2)
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    identity = _profile_identity(output_path, serialized.encode("utf-8"), profile)
    calibration.metadata["runtime_vive_profile"] = identity
    setattr(args, "expected_vive_profile_sha256", identity["sha256"])
    log_event("ViveCalibration", "saved_profile=" + json.dumps(identity, sort_keys=True))
    setattr(args, "gui_last_vive_calibration_profile", str(output_path))
    return output_path


def _load_gui_vive_calibration_profile(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    profile_dir: Path | None = None,
) -> Path:
    path = _gui_vive_calibration_load_path(
        args,
        profile_dir=profile_dir,
    )
    loaded = load_vive_calibration_profile(
        path, expected_sha256=getattr(args, "expected_vive_profile_sha256", None)
    )
    current = components.session.calibration
    if current is None:
        current = loaded
        components.session.calibration = current
    else:
        current.T_W_O = None if loaded.T_W_O is None else loaded.T_W_O.copy()
        current.T_O_D = None if loaded.T_O_D is None else loaded.T_O_D.copy()
        current.tracker_offsets = {
            str(role): np.asarray(offset, dtype=float).copy()
            for role, offset in loaded.tracker_offsets.items()
        }
        for arm_name, loaded_arm in loaded.arms.items():
            current_arm = current.arms.get(arm_name)
            if current_arm is None:
                continue
            current_arm.workspace_min = (
                None
                if loaded_arm.workspace_min is None
                else np.asarray(loaded_arm.workspace_min, dtype=float).copy()
            )
            current_arm.workspace_max = (
                None
                if loaded_arm.workspace_max is None
                else np.asarray(loaded_arm.workspace_max, dtype=float).copy()
            )
        calibration_metadata_keys = (
            "calibration_mode",
            "world_to_desk_mode",
            "virtual_desk",
            "virtual_desk_tracker_role",
            "virtual_desk_channel_name",
            "virtual_desk_sweep_convention_version",
            "virtual_desk_post_yaw_deg",
            "virtual_desk_scene_alignment_defaulted",
            "tracker_offset_calibration",
            "tracker_offset_role_binding",
            "runtime_vive_profile",
            "profile_scope",
            "home_policy",
            "workspace_policy",
            "workspace_config_path",
            "saved_at",
        )
        for key in calibration_metadata_keys:
            if key in loaded.metadata:
                current.metadata[key] = loaded.metadata[key]
            else:
                current.metadata.pop(key, None)
    current.clear_home_state()
    current.metadata.update(
        {
            "runtime_gui_loaded_profile": str(path),
            "runtime_gui_loaded_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _apply_workspace_config(args, current)
    _apply_workspace_overrides(args, current)
    components.left_retargeter.clear_home_anchors()
    components.right_retargeter.clear_home_anchors()
    setattr(args, "load_profile", str(path))
    setattr(args, "gui_last_vive_calibration_profile", str(path))
    setattr(args, "expected_vive_profile_sha256", loaded.metadata["runtime_vive_profile"]["sha256"])
    return path

__all__ = [
    "_calibration_profile_latest_sort_key",
    "_calibration_profile_payload_without_workspace",
    "_gui_vive_calibration_load_path",
    "_latest_vive_calibration_profile_path",
    "_load_gui_vive_calibration_profile",
    "_runtime_home_was_available",
    "_save_gui_vive_calibration_profile",
    "load_vive_calibration_profile",
    "vive_configuration_metadata",
]
