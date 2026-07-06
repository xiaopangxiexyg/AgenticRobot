from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..constants import QPOS_UNIT
from ..qpos_layout import as_rm65_qpos, normalize_side, validate_qpos_by_side


QPOS_TARGET_DATA_TYPE = "rm65_arm_init_qpos_target"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ArmQposSpec:
    side: str
    qpos: np.ndarray
    qpos_units: str = QPOS_UNIT
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        side = normalize_side(self.side)
        unit = str(self.qpos_units or QPOS_UNIT)
        if unit != QPOS_UNIT:
            raise ValueError(f"{side}.qpos_units must be {QPOS_UNIT!r}, got {unit!r}.")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "qpos", as_rm65_qpos(self.qpos, label=f"{side}.qpos"))
        object.__setattr__(self, "qpos_units", unit)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "qpos": self.qpos.astype(float).tolist(),
            "qpos_units": self.qpos_units,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, side: str, payload: Mapping[str, Any]) -> "ArmQposSpec":
        if not isinstance(payload, Mapping):
            raise ValueError(f"{side}.qpos spec must be a mapping.")
        qpos = payload.get("qpos")
        if qpos is None:
            raise ValueError(f"{side}.qpos is required.")
        return cls(
            side=side,
            qpos=qpos,
            qpos_units=str(payload.get("qpos_units") or QPOS_UNIT),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), Mapping) else {},
        )


@dataclass(frozen=True)
class QposTargetSpec:
    arms: dict[str, ArmQposSpec]
    name: str = "rm65_init_qpos_target"
    schema_version: int = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.schema_version) != SCHEMA_VERSION:
            raise ValueError(f"unsupported qpos target schema_version={self.schema_version!r}.")
        arms: dict[str, ArmQposSpec] = {}
        for raw_side, spec in self.arms.items():
            side = normalize_side(raw_side)
            arms[side] = spec if isinstance(spec, ArmQposSpec) else ArmQposSpec.from_dict(side, spec)
        if not arms:
            raise ValueError("qpos target must contain at least one arm.")
        object.__setattr__(self, "arms", arms)
        object.__setattr__(self, "name", str(self.name or "rm65_init_qpos_target"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_qpos_by_side(
        cls,
        qpos_by_side: Mapping[str, Any],
        *,
        name: str = "rm65_init_qpos_target",
        metadata: Mapping[str, Any] | None = None,
    ) -> "QposTargetSpec":
        return cls(
            arms={
                side: ArmQposSpec(side=side, qpos=qpos)
                for side, qpos in validate_qpos_by_side(qpos_by_side).items()
            },
            name=name,
            metadata=dict(metadata or {}),
        )

    def qpos_by_side(self) -> dict[str, np.ndarray]:
        return {side: spec.qpos.copy() for side, spec in self.arms.items()}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "data_type": QPOS_TARGET_DATA_TYPE,
            "schema_version": int(self.schema_version),
            "name": self.name,
            "arms": {side: spec.to_dict() for side, spec in self.arms.items()},
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QposTargetSpec":
        if not isinstance(payload, Mapping):
            raise ValueError("qpos target payload must be a mapping.")
        data_type = str(payload.get("data_type") or QPOS_TARGET_DATA_TYPE)
        if data_type != QPOS_TARGET_DATA_TYPE:
            raise ValueError(f"unsupported qpos target data_type={data_type!r}.")
        arms_payload = payload.get("arms")
        if not isinstance(arms_payload, Mapping) or not arms_payload:
            raise ValueError("qpos target payload must contain a non-empty arms mapping.")
        return cls(
            arms={
                normalize_side(str(side)): ArmQposSpec.from_dict(str(side), dict(arm_payload))
                for side, arm_payload in arms_payload.items()
            },
            name=str(payload.get("name") or "rm65_init_qpos_target"),
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), Mapping) else {},
        )


def load_qpos_target(path: str | Path) -> QposTargetSpec:
    return QposTargetSpec.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_qpos_target(spec: QposTargetSpec, path: str | Path) -> None:
    Path(path).write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def qpos_target_schema_path() -> Path:
    return Path(__file__).resolve().parent / "schemas" / "qpos_target.schema.json"


__all__ = [
    "ArmQposSpec",
    "QPOS_TARGET_DATA_TYPE",
    "QposTargetSpec",
    "load_qpos_target",
    "qpos_target_schema_path",
    "save_qpos_target",
]
