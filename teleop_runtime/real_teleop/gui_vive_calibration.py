"""Read-only Vive calibration profile state for the real-teleop GUI."""

from __future__ import annotations

from pathlib import Path


def vive_calibration_tracker_roles(active_sides: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{side}_wrist" for side in active_sides if side in {"left", "right"})


def vive_calibration_profile_ready(
    calibration: object | None,
    *,
    active_sides: tuple[str, ...],
) -> bool:
    if calibration is None:
        return False
    if calibration.world_to_desk_pose() is None:
        return False
    roles = vive_calibration_tracker_roles(active_sides)
    return all(role in calibration.tracker_offsets for role in roles)


def vive_calibration_profile_status_text(
    calibration: object | None,
    *,
    active_sides: tuple[str, ...],
) -> str:
    if calibration is None:
        return "calibration: unavailable"
    desk_ready = calibration.world_to_desk_pose() is not None
    roles = vive_calibration_tracker_roles(active_sides)
    missing_roles = [role for role in roles if role not in calibration.tracker_offsets]
    offsets_text = "ok" if not missing_roles else "missing " + ",".join(missing_roles)
    return f"desk={'ok' if desk_ready else 'missing'} tracker_offsets={offsets_text}"


def vive_calibration_profile_ui_state(
    args: object,
    components: object,
    *,
    status: str = "",
    path: str | Path | None = None,
    loaded: bool | None = None,
) -> dict[str, object]:
    calibration = getattr(getattr(components, "session"), "calibration")
    active_sides = tuple(getattr(components, "active_sides", ()) or ())
    profile_path = (
        str(path)
        if path is not None
        else str(
            getattr(args, "gui_last_vive_calibration_profile", "")
            or getattr(args, "load_profile", "")
            or ""
        )
    )
    status_text = vive_calibration_profile_status_text(
        calibration,
        active_sides=active_sides,
    )
    if status:
        status_text = f"{status}\n{status_text}"
    if loaded is None:
        loaded = bool(str(getattr(args, "load_profile", "") or "").strip())
    return {
        "vive_calibration_profile_path": profile_path,
        "vive_calibration_profile_status": status_text,
        "vive_calibration_profile_ready": vive_calibration_profile_ready(
            calibration,
            active_sides=active_sides,
        ),
        "vive_calibration_profile_loaded": bool(loaded),
    }


__all__ = [
    "vive_calibration_profile_ready",
    "vive_calibration_profile_status_text",
    "vive_calibration_profile_ui_state",
    "vive_calibration_tracker_roles",
]
