from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .schema import HAND16_SLOT_DOF, INSPIRE_HAND_DOF, VALID_SIDES, normalize_side

DEFAULT_HAND_SLOT_SEMANTICS = (
    "Only slots with active_mask=True are valid actuator targets. "
    "Invalid slots are padding for unused hand capacity and must not be interpreted as zero commands."
)


def _default_joint_names(active_dof: int) -> tuple[str, ...]:
    return tuple(f"q{index}" for index in range(int(active_dof)))


@dataclass(frozen=True)
class HandActionSpec:
    side: str
    hand_model: str = "inspire_tac"
    slot_count: int = INSPIRE_HAND_DOF
    active_dof: int = INSPIRE_HAND_DOF
    active_mask: tuple[bool, ...] | None = None
    joint_names: tuple[str, ...] = ()
    command_space: str = "normalized_0_1"
    slot_map: Mapping[str, int] = field(default_factory=dict)
    invalid_slot_value: str = "NaN"
    semantics: str = DEFAULT_HAND_SLOT_SEMANTICS
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        side = normalize_side(self.side)
        slot_count = int(self.slot_count)
        active_dof = int(self.active_dof)
        if slot_count <= 0:
            raise ValueError("HandActionSpec.slot_count must be positive.")
        if active_dof < 0 or active_dof > slot_count:
            raise ValueError(
                f"HandActionSpec.active_dof must be in [0, slot_count], got active_dof={active_dof}."
            )
        if self.active_mask is None:
            active_mask = tuple(index < active_dof for index in range(slot_count))
        else:
            active_mask = tuple(bool(value) for value in self.active_mask)
            if len(active_mask) != slot_count:
                raise ValueError(
                    f"HandActionSpec.active_mask must have length {slot_count}, got {len(active_mask)}."
                )
            if sum(active_mask) != active_dof:
                raise ValueError(
                    "HandActionSpec.active_mask true-count must equal active_dof, "
                    f"got {sum(active_mask)} and {active_dof}."
                )
        joint_names = tuple(str(name) for name in self.joint_names)
        if not joint_names:
            joint_names = _default_joint_names(active_dof)
        if len(joint_names) != active_dof:
            raise ValueError(
                f"HandActionSpec.joint_names must have length active_dof={active_dof}, got {len(joint_names)}."
            )
        if self.slot_map:
            slot_map = {str(name): int(index) for name, index in dict(self.slot_map).items()}
        else:
            active_indices = [index for index, is_active in enumerate(active_mask) if is_active]
            slot_map = {name: int(index) for name, index in zip(joint_names, active_indices)}
        for name, index in slot_map.items():
            if index < 0 or index >= slot_count:
                raise ValueError(f"HandActionSpec.slot_map[{name!r}]={index} outside slot_count={slot_count}.")
            if not active_mask[index]:
                raise ValueError(f"HandActionSpec.slot_map[{name!r}] points to inactive slot {index}.")

        object.__setattr__(self, "side", side)
        object.__setattr__(self, "hand_model", str(self.hand_model))
        object.__setattr__(self, "slot_count", slot_count)
        object.__setattr__(self, "active_dof", active_dof)
        object.__setattr__(self, "active_mask", active_mask)
        object.__setattr__(self, "joint_names", joint_names)
        object.__setattr__(self, "command_space", str(self.command_space))
        object.__setattr__(self, "slot_map", slot_map)
        object.__setattr__(self, "invalid_slot_value", str(self.invalid_slot_value))
        object.__setattr__(self, "semantics", str(self.semantics or DEFAULT_HAND_SLOT_SEMANTICS))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def inspire_tac(cls, side: str) -> "HandActionSpec":
        return cls(
            side=side,
            hand_model="inspire_tac",
            slot_count=INSPIRE_HAND_DOF,
            active_dof=INSPIRE_HAND_DOF,
            joint_names=("pinky", "ring", "middle", "index", "thumb_yaw", "thumb_flex"),
            command_space="normalized_0_1",
            invalid_slot_value="none",
        )

    @classmethod
    def q16(
        cls,
        side: str,
        *,
        hand_model: str = "generic_hand16",
        active_dof: int,
        joint_names: tuple[str, ...] | list[str] | None = None,
        command_space: str = "normalized_0_1",
        metadata: Mapping[str, Any] | None = None,
    ) -> "HandActionSpec":
        return cls(
            side=side,
            hand_model=hand_model,
            slot_count=HAND16_SLOT_DOF,
            active_dof=int(active_dof),
            joint_names=tuple(_default_joint_names(active_dof) if joint_names is None else joint_names),
            command_space=command_space,
            invalid_slot_value="NaN",
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def from_metadata(cls, payload: Mapping[str, Any]) -> "HandActionSpec":
        return cls(
            side=str(payload["side"]),
            hand_model=str(payload.get("hand_model", "unknown")),
            slot_count=int(payload["slot_count"]),
            active_dof=int(payload["active_dof"]),
            active_mask=tuple(bool(value) for value in payload["active_mask"]),
            joint_names=tuple(str(value) for value in payload.get("joint_names", ())),
            command_space=str(payload.get("command_space", "normalized_0_1")),
            slot_map={str(key): int(value) for key, value in dict(payload.get("slot_map", {})).items()},
            invalid_slot_value=str(payload.get("invalid_slot_value", "NaN")),
            semantics=str(payload.get("semantics", DEFAULT_HAND_SLOT_SEMANTICS)),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "hand_model": self.hand_model,
            "slot_count": int(self.slot_count),
            "active_dof": int(self.active_dof),
            "active_mask": list(self.active_mask or ()),
            "joint_names": list(self.joint_names),
            "command_space": self.command_space,
            "slot_map": dict(self.slot_map),
            "invalid_slot_value": self.invalid_slot_value,
            "semantics": self.semantics,
            "metadata": dict(self.metadata),
        }

    def active_values(self, vector: Any) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float32).reshape(-1)
        if values.shape != (self.slot_count,):
            raise ValueError(f"Expected hand vector shape ({self.slot_count},), got {values.shape}.")
        mask = np.asarray(self.active_mask, dtype=bool)
        active = values[mask]
        if active.shape != (self.active_dof,):
            raise ValueError(f"Expected active hand shape ({self.active_dof},), got {active.shape}.")
        if not np.all(np.isfinite(active)):
            raise ValueError("Active hand slots contain non-finite values.")
        return active.astype(np.float32, copy=True)


def normalize_hand_action_specs(
    specs: Mapping[str, HandActionSpec | Mapping[str, Any]] | None,
    *,
    active_sides: tuple[str, ...] = VALID_SIDES,
    default_slot_count: int = INSPIRE_HAND_DOF,
) -> dict[str, HandActionSpec]:
    normalized: dict[str, HandActionSpec] = {}
    for side in active_sides:
        side = normalize_side(side)
        payload = None if specs is None else specs.get(side)
        if isinstance(payload, HandActionSpec):
            normalized[side] = payload
        elif isinstance(payload, Mapping):
            normalized[side] = HandActionSpec.from_metadata(payload)
        elif int(default_slot_count) == HAND16_SLOT_DOF:
            normalized[side] = HandActionSpec.q16(side, active_dof=INSPIRE_HAND_DOF)
        else:
            normalized[side] = HandActionSpec.inspire_tac(side)
    return normalized


def hand_action_specs_metadata(specs: Mapping[str, HandActionSpec]) -> dict[str, Any]:
    return {normalize_side(side): spec.to_metadata() for side, spec in specs.items()}


def hand_action_specs_from_metadata(metadata: Mapping[str, Any] | None) -> dict[str, HandActionSpec]:
    if not metadata:
        return {}
    payload = metadata.get("hand_action_specs_by_side")
    if not isinstance(payload, Mapping):
        return {}
    specs: dict[str, HandActionSpec] = {}
    for side, spec_payload in payload.items():
        if isinstance(spec_payload, HandActionSpec):
            spec = spec_payload
        elif isinstance(spec_payload, Mapping):
            spec = HandActionSpec.from_metadata(spec_payload)
        else:
            continue
        specs[spec.side] = spec
    return specs
