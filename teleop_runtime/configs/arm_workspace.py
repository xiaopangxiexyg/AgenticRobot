from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_ARM_WORKSPACE_CONFIG_PATH = Path(__file__).resolve().parent / "arm_workspace.json"
SIDES = ("left", "right")


@dataclass(frozen=True)
class ArmWorkspaceBounds:
    side: str
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    source_path: str

    def to_metadata(self) -> dict[str, object]:
        return {
            "side": self.side,
            "workspace_min": list(self.workspace_min),
            "workspace_max": list(self.workspace_max),
            "source_path": self.source_path,
        }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _vector3(value: object, *, key: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key} must contain exactly 3 numbers.")
    try:
        parsed = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must contain numeric values.") from exc
    return (parsed[0], parsed[1], parsed[2])


def _lookup_bound(
    payload: Mapping[str, Any],
    fallback: Mapping[str, Any],
    *,
    bound: str,
    side: str,
) -> tuple[float, float, float]:
    aliases = (f"workspace_{bound}", bound)
    for key in aliases:
        value = payload.get(key)
        if value is not None:
            return _vector3(value, key=f"sides.{side}.{key}")
    for key in aliases:
        value = fallback.get(key)
        if value is not None:
            return _vector3(value, key=f"default.{key}")
    raise ValueError(f"Missing workspace_{bound} for side {side!r}.")


def load_arm_workspace_config(
    path: str | Path = DEFAULT_ARM_WORKSPACE_CONFIG_PATH,
) -> dict[str, ArmWorkspaceBounds]:
    resolved = Path(path).expanduser().resolve()
    payload = _read_json(resolved)
    default_payload = payload.get("default", {})
    if not isinstance(default_payload, Mapping):
        raise ValueError(f"{resolved} field 'default' must be a JSON object.")
    sides_payload = payload.get("sides", {})
    if not isinstance(sides_payload, Mapping):
        raise ValueError(f"{resolved} field 'sides' must be a JSON object.")

    bounds_by_side: dict[str, ArmWorkspaceBounds] = {}
    for side in SIDES:
        side_payload = sides_payload.get(side, {})
        if side_payload is None:
            side_payload = {}
        if not isinstance(side_payload, Mapping):
            raise ValueError(f"{resolved} field 'sides.{side}' must be a JSON object.")
        workspace_min = _lookup_bound(
            side_payload,
            default_payload,
            bound="min",
            side=side,
        )
        workspace_max = _lookup_bound(
            side_payload,
            default_payload,
            bound="max",
            side=side,
        )
        if any(lo > hi for lo, hi in zip(workspace_min, workspace_max)):
            raise ValueError(
                f"Invalid workspace for side {side!r}: workspace_min "
                f"{list(workspace_min)} exceeds workspace_max {list(workspace_max)}."
            )
        bounds_by_side[side] = ArmWorkspaceBounds(
            side=side,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            source_path=str(resolved),
        )
    return bounds_by_side
