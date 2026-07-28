"""Independent VIVE coordinate-calibration profile storage for Control Station.

The stored payload deliberately reuses :class:`TeleopCalibration` so future
sim teleop adapters can consume the existing coordinate and tracker-offset
semantics.  Storage and profile identity are separate from the real-teleop
``calibration_profiles`` directory, and real-teleop profiles are rejected even
if a caller accidentally copies one into this directory.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from project_paths import (
    calibration_profiles_dir,
    control_station_vive_coordinate_profiles_dir,
)
from teleop_runtime.calibration import ArmCalibration, TeleopCalibration
from teleop_runtime.transforms import as_pose_matrix


VIVE_COORDINATE_PROFILE_KIND = "control_station_vive_coordinate"
VIVE_COORDINATE_PROFILE_SCHEMA = "control_station_vive_coordinate_v1"
_PROFILE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_VALID_SIDES = frozenset({"left", "right"})


def normalize_active_sides(value: str | Iterable[str]) -> tuple[str, ...]:
    """Normalize ``left``, ``right`` or ``both`` to a stable side tuple."""

    if isinstance(value, str):
        text = value.strip().lower()
        if text == "both":
            return ("left", "right")
        values = (text,)
    else:
        values = tuple(str(item).strip().lower() for item in value)
    if not values or any(item not in _VALID_SIDES for item in values):
        raise ValueError("active_sides must contain left and/or right.")
    normalized = tuple(side for side in ("left", "right") if side in set(values))
    if not normalized:
        raise ValueError("active_sides must contain left and/or right.")
    return normalized


def safe_profile_stem(value: str | None) -> str:
    text = _PROFILE_NAME_RE.sub("_", str(value or "").strip()).strip("._-")
    if not text:
        return "vive_coordinate"
    return text[:96]


def new_coordinate_calibration(
    *,
    active_sides: Iterable[str],
    left_tracker_name: str = "tracker_1",
    right_tracker_name: str = "tracker_2",
    auxiliary_tracker_name: str = "",
    desk_tracker: str = "auto",
) -> TeleopCalibration:
    """Build a coordinate-only calibration using the shared payload schema."""

    sides = normalize_active_sides(tuple(active_sides))
    arms = {
        f"{side}_arm": ArmCalibration(
            name=f"{side}_arm",
            side=side,
            # Robot/sim base transforms belong to the target profile, not to
            # this VIVE-only station calibration.
            T_D_B=None,
        )
        for side in sides
    }
    calibration = TeleopCalibration(
        arms=arms,
        metadata={
            "profile_kind": VIVE_COORDINATE_PROFILE_KIND,
            "profile_schema": VIVE_COORDINATE_PROFILE_SCHEMA,
            "profile_scope": "control_station_vive_coordinate_only",
            "active_sides": list(sides),
            "left_tracker_name": str(left_tracker_name),
            "right_tracker_name": str(right_tracker_name),
            "auxiliary_tracker_name": str(auxiliary_tracker_name or ""),
            "desk_tracker": str(desk_tracker),
            "home_policy": "session_home_is_not_part_of_coordinate_profile",
            "target_binding_policy": "target_profile_supplies_desk_to_base",
        },
    )
    return calibration


def _finite_pose(value: Any, *, field_name: str) -> None:
    array = as_pose_matrix(value)
    if array.shape != (4, 4) or not all(
        math.isfinite(float(item))
        for row in array.tolist()
        for item in row
    ):
        raise ValueError(f"{field_name} must be a finite 4x4 pose matrix.")


def validate_coordinate_profile(
    calibration: TeleopCalibration,
    *,
    path: Path | None = None,
) -> tuple[str, ...]:
    """Validate the Control Station profile identity and required components."""

    metadata = calibration.metadata
    if metadata.get("profile_kind") != VIVE_COORDINATE_PROFILE_KIND:
        location = "" if path is None else f" at {path}"
        raise ValueError(
            f"Profile{location} is not a Control Station VIVE coordinate profile."
        )
    if metadata.get("profile_schema") != VIVE_COORDINATE_PROFILE_SCHEMA:
        raise ValueError("Unsupported Control Station VIVE coordinate profile schema.")
    sides = normalize_active_sides(metadata.get("active_sides", ()))
    if calibration.world_to_desk_pose() is None:
        raise ValueError("Coordinate profile is missing the fitted VIVE frame.")
    _finite_pose(calibration.world_to_desk_pose(), field_name="world_to_vive_retarget")
    for side in sides:
        role = f"{side}_wrist"
        if role not in calibration.tracker_offsets:
            raise ValueError(f"Coordinate profile is missing tracker offset {role}.")
        _finite_pose(calibration.tracker_offsets[role], field_name=f"tracker_offsets.{role}")
    return sides


def _profile_root(root: Path | None) -> Path:
    directory = (
        control_station_vive_coordinate_profiles_dir()
        if root is None
        else Path(root)
    ).expanduser().resolve()
    real_root = calibration_profiles_dir().expanduser().resolve()
    if directory == real_root or real_root in directory.parents:
        raise ValueError(
            "Control Station VIVE coordinate profiles must use a directory "
            "outside the real-teleop calibration profile tree."
        )
    return directory


def _profile_path(root: Path, name_or_path: str | Path) -> Path:
    raw = Path(name_or_path).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    candidate = candidate.resolve()
    if candidate.parent != root:
        raise ValueError("VIVE coordinate profile must be stored directly in the Control Station profile directory.")
    if candidate.suffix.lower() != ".json":
        candidate = candidate.with_suffix(".json")
    return candidate


@dataclass(frozen=True)
class CoordinateProfileSummary:
    name: str
    path: str
    profile_name: str
    active_sides: tuple[str, ...]
    saved_at: str
    valid: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "profile_name": self.profile_name,
            "active_sides": list(self.active_sides),
            "saved_at": self.saved_at,
            "valid": self.valid,
            "error": self.error,
        }


def list_coordinate_profiles(root: Path | None = None) -> list[CoordinateProfileSummary]:
    directory = _profile_root(root)
    if not directory.exists():
        return []
    summaries: list[CoordinateProfileSummary] = []
    for path in sorted(directory.glob("*.json"), key=lambda item: item.name, reverse=True):
        try:
            calibration = TeleopCalibration.load_json(path)
            sides = validate_coordinate_profile(calibration, path=path)
            metadata = calibration.metadata
            summaries.append(
                CoordinateProfileSummary(
                    name=path.name,
                    path=str(path),
                    profile_name=str(metadata.get("profile_name") or path.stem),
                    active_sides=sides,
                    saved_at=str(metadata.get("saved_at") or ""),
                )
            )
        except Exception as exc:  # Keep a corrupted profile visible in the UI.
            summaries.append(
                CoordinateProfileSummary(
                    name=path.name,
                    path=str(path),
                    profile_name=path.stem,
                    active_sides=(),
                    saved_at="",
                    valid=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return summaries


def save_coordinate_profile(
    calibration: TeleopCalibration,
    *,
    root: Path | None = None,
    profile_name: str | None = None,
) -> Path:
    directory = _profile_root(root)
    sides = validate_coordinate_profile(calibration)
    stem = safe_profile_stem(profile_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"{timestamp}_{stem}.json"
    suffix = 1
    while path.exists():
        path = directory / f"{timestamp}_{stem}_{suffix}.json"
        suffix += 1

    payload = calibration.to_dict()
    metadata = payload.setdefault("metadata", {})
    metadata.update(
        {
            "profile_kind": VIVE_COORDINATE_PROFILE_KIND,
            "profile_schema": VIVE_COORDINATE_PROFILE_SCHEMA,
            "profile_scope": "control_station_vive_coordinate_only",
            "profile_name": stem,
            "active_sides": list(sides),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "saved_by": "control_station.vive_calibration",
            "real_teleop_profile_path": None,
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_coordinate_profile(
    name_or_path: str | Path,
    *,
    root: Path | None = None,
) -> tuple[TeleopCalibration, Path]:
    directory = _profile_root(root)
    path = _profile_path(directory, name_or_path)
    if not path.exists() and not Path(name_or_path).is_absolute():
        stem = safe_profile_stem(str(name_or_path))
        candidates = sorted(
            directory.glob(f"*_{stem}.json"),
            key=lambda item: item.name,
            reverse=True,
        )
        if candidates:
            path = candidates[0]
    calibration = TeleopCalibration.load_json(path)
    validate_coordinate_profile(calibration, path=path)
    return calibration, path


def coordinate_profile_view(calibration: TeleopCalibration) -> dict[str, Any]:
    sides = validate_coordinate_profile(calibration)
    metadata = calibration.metadata
    world_to_vive = calibration.world_to_desk_pose()
    return {
        "profile_kind": metadata.get("profile_kind"),
        "profile_schema": metadata.get("profile_schema"),
        "profile_name": metadata.get("profile_name", ""),
        "profile_scope": metadata.get("profile_scope", ""),
        "active_sides": list(sides),
        "desk_tracker": metadata.get("desk_tracker", "auto"),
        "saved_at": metadata.get("saved_at", ""),
        "world_to_vive_retarget_pose": None
        if world_to_vive is None
        else world_to_vive.tolist(),
        "tracker_offsets": {
            role: calibration.tracker_offsets[role].tolist()
            for role in sorted(calibration.tracker_offsets)
        },
        "virtual_desk": metadata.get("virtual_desk", {}),
        "tracker_offset_calibration": metadata.get("tracker_offset_calibration", {}),
        "target_binding_policy": metadata.get("target_binding_policy", ""),
        "home_policy": metadata.get("home_policy", ""),
    }


__all__ = [
    "CoordinateProfileSummary",
    "VIVE_COORDINATE_PROFILE_KIND",
    "VIVE_COORDINATE_PROFILE_SCHEMA",
    "coordinate_profile_view",
    "list_coordinate_profiles",
    "load_coordinate_profile",
    "new_coordinate_calibration",
    "normalize_active_sides",
    "safe_profile_stem",
    "save_coordinate_profile",
    "validate_coordinate_profile",
]
