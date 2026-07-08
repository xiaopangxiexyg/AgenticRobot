from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

from ..schemas import TeleopCommand

ARM_MODE_SCHEMA = "dexteleop_arm_mode_v1"

ARM_MODE_IDS: dict[str, int] = {
    "unknown": 0,
    "teleop": 1,
    "fixed_hold": 2,
    "init_pose": 3,
    "stopped": 4,
}

ARM_MODE_ID_TO_LABEL = {mode_id: label for label, mode_id in ARM_MODE_IDS.items()}


def _normalize_side(side: Any) -> str:
    normalized = str(side or "").strip().lower()
    return normalized if normalized in {"left", "right"} else ""


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _metadata_values(command: TeleopCommand | None, key: str) -> list[Any]:
    if command is None:
        return []
    values: list[Any] = []
    if key in (command.metadata or {}):
        values.append(command.metadata.get(key))
    for effector_command in command.commands.values():
        metadata = effector_command.metadata or {}
        if key in metadata:
            values.append(metadata.get(key))
    return values


def _first_metadata(command: TeleopCommand | None, key: str) -> Any:
    for value in _metadata_values(command, key):
        if value is not None:
            return value
    return None


def _active_sides(command: TeleopCommand | None, *, active_key: str, side_key: str) -> set[str]:
    sides: set[str] = set()
    side = _normalize_side(_first_metadata(command, side_key))
    if side:
        sides.add(side)
    if command is None:
        return sides
    for effector_command in command.commands.values():
        metadata = effector_command.metadata or {}
        if not _boolish(metadata.get(active_key)):
            continue
        side = _normalize_side(metadata.get(side_key) or effector_command.side)
        if side:
            sides.add(side)
    return sides


def _side_payload(
    *,
    mode: str,
    source: str,
    tags: Sequence[str] = (),
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    label = str(mode or "unknown")
    if label not in ARM_MODE_IDS:
        label = "unknown"
    mode_id = int(ARM_MODE_IDS[label])
    tag_list = [str(tag) for tag in tags if str(tag)]
    return {
        "mode": label,
        "label": label,
        "mode_id": mode_id,
        "id": mode_id,
        "source": str(source or ""),
        "is_special": bool(label in {"fixed_hold", "init_pose"}),
        "tags": tag_list,
        "detail": dict(detail or {}),
    }


def build_arm_mode_metadata(
    *,
    command: TeleopCommand | None,
    session_state: str,
    active_sides: Sequence[str] = ("left", "right"),
    recording_paused: bool = False,
) -> dict[str, Any]:
    """Build per-side semantic labels without changing action/state vectors."""

    normalized_active_sides = {
        side for side in (_normalize_side(side) for side in active_sides) if side
    }
    state_text = str(session_state or "").strip()
    state_upper = state_text.upper()
    stopped_by_session = bool(recording_paused or state_upper not in {"RUNNING"})
    init_sides = _active_sides(
        command,
        active_key="init_arm_active",
        side_key="init_arm_side",
    )
    init_side = _normalize_side(_first_metadata(command, "init_arm_active_side"))
    if init_side:
        init_sides.add(init_side)
    fix_sides = _active_sides(
        command,
        active_key="fix_arm_active",
        side_key="fix_arm_side",
    )
    fix_side = _normalize_side(_first_metadata(command, "fix_arm_active_side"))
    if fix_side:
        fix_sides.add(fix_side)

    sides: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        enabled = side in normalized_active_sides
        detail = {
            "active_side": enabled,
            "session_state": state_text,
            "recording_paused": bool(recording_paused),
        }
        if stopped_by_session:
            source = "recording_pause" if recording_paused else "session_state"
            sides[side] = _side_payload(
                mode="stopped" if enabled else "unknown",
                source=source,
                tags=("hold",) if recording_paused else (),
                detail=detail,
            )
        elif side in init_sides:
            sides[side] = _side_payload(
                mode="init_pose",
                source="init_arm",
                tags=("init_arm",),
                detail={**detail, "init_arm_mode": _first_metadata(command, "init_arm_mode")},
            )
        elif side in fix_sides:
            sides[side] = _side_payload(
                mode="fixed_hold",
                source="fix_arm",
                tags=("fix_arm",),
                detail={**detail, "fix_arm_mode": _first_metadata(command, "fix_arm_mode")},
            )
        elif enabled:
            sides[side] = _side_payload(mode="teleop", source="normal", detail=detail)
        else:
            sides[side] = _side_payload(mode="unknown", source="inactive_side", detail=detail)

    active_special_tasks = sorted(
        {
            tag
            for side_payload in sides.values()
            for tag in side_payload.get("tags", ())
            if tag in {"fix_arm", "init_arm"}
        }
    )
    return {
        "schema": ARM_MODE_SCHEMA,
        "available": True,
        "sides": sides,
        "active_special_tasks": active_special_tasks,
    }


def normalize_arm_mode_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        raw_sides = value.get("sides")
        if isinstance(raw_sides, Mapping):
            sides = {}
            for side in ("left", "right"):
                raw_side = raw_sides.get(side)
                if isinstance(raw_side, Mapping):
                    raw_mode = raw_side.get("mode", raw_side.get("label", "unknown"))
                    mode = str(raw_mode or "unknown")
                    raw_tags = raw_side.get("tags", ())
                    tags = (
                        raw_tags
                        if isinstance(raw_tags, Sequence) and not isinstance(raw_tags, (str, bytes))
                        else ()
                    )
                    sides[side] = _side_payload(
                        mode=mode,
                        source=str(raw_side.get("source", "")),
                        tags=tags,
                        detail=raw_side.get("detail", {}) if isinstance(raw_side.get("detail"), Mapping) else {},
                    )
                else:
                    sides[side] = _side_payload(mode="unknown", source="missing_side")
            return {
                "schema": str(value.get("schema") or ARM_MODE_SCHEMA),
                "available": bool(value.get("available", True)),
                "sides": sides,
                "active_special_tasks": list(value.get("active_special_tasks", []))
                if isinstance(value.get("active_special_tasks"), Sequence)
                and not isinstance(value.get("active_special_tasks"), (str, bytes))
                else [],
            }
    return {
        "schema": ARM_MODE_SCHEMA,
        "available": False,
        "sides": {
            "left": _side_payload(mode="unknown", source="missing_metadata"),
            "right": _side_payload(mode="unknown", source="missing_metadata"),
        },
        "active_special_tasks": [],
    }


def flatten_arm_mode_columns(payload: Any) -> dict[str, Any]:
    normalized = normalize_arm_mode_metadata(payload)
    columns: dict[str, Any] = {
        "teleop.arm_mode.schema": normalized["schema"],
        "teleop.arm_mode.available": int(bool(normalized.get("available", False))),
    }
    sides = normalized.get("sides", {})
    for side in ("left", "right"):
        side_payload = sides.get(side, {}) if isinstance(sides, Mapping) else {}
        mode = str(side_payload.get("mode") or side_payload.get("label") or "unknown")
        if mode not in ARM_MODE_IDS:
            mode = "unknown"
        columns[f"teleop.arm_mode.{side}.id"] = int(ARM_MODE_IDS[mode])
        columns[f"teleop.arm_mode.{side}.label"] = mode
        columns[f"teleop.arm_mode.{side}.is_special"] = int(
            bool(side_payload.get("is_special", mode in {"fixed_hold", "init_pose"}))
        )
        columns[f"teleop.arm_mode.{side}.source"] = str(side_payload.get("source", ""))
        tags = side_payload.get("tags", ())
        columns[f"teleop.arm_mode.{side}.tags"] = [
            str(tag) for tag in tags
        ] if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)) else []
    return columns
