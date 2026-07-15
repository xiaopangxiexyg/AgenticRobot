from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from robot_motion.frames import (
    ARM_ENDPOINT_FRAME,
    ARM_ENDPOINT_SEMANTICS,
    TOOL_TCP_FRAME,
    TOOL_TCP_SEMANTICS,
    canonical_arm_controlled_frame,
)

from .schema import ARM_POSE_DOF, VALID_SIDES, default_arm_frame, normalize_side


@dataclass(frozen=True)
class ArmActionSpec:
    side: str
    controlled_frame: str = TOOL_TCP_FRAME
    reference_frame: str | None = None
    representation: str = "xyz_quat"
    dof: int = ARM_POSE_DOF
    semantics: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        side = normalize_side(self.side)
        controlled_frame = canonical_arm_controlled_frame(self.controlled_frame)
        reference_frame = (
            default_arm_frame(side)
            if self.reference_frame is None
            else str(self.reference_frame)
        )
        dof = int(self.dof)
        if dof != ARM_POSE_DOF:
            raise ValueError(f"ArmActionSpec.dof must be {ARM_POSE_DOF}, got {dof}.")
        if str(self.representation) != "xyz_quat":
            raise ValueError("ArmActionSpec.representation currently must be 'xyz_quat'.")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "controlled_frame", controlled_frame)
        object.__setattr__(self, "reference_frame", reference_frame)
        object.__setattr__(self, "representation", "xyz_quat")
        object.__setattr__(self, "dof", dof)
        default_semantics = (
            ARM_ENDPOINT_SEMANTICS
            if controlled_frame == ARM_ENDPOINT_FRAME
            else TOOL_TCP_SEMANTICS
        )
        object.__setattr__(self, "semantics", str(self.semantics or default_semantics))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def tool_tcp(cls, side: str, *, reference_frame: str | None = None) -> "ArmActionSpec":
        return cls(side=side, controlled_frame=TOOL_TCP_FRAME, reference_frame=reference_frame)

    @classmethod
    def endpoint(cls, side: str, *, reference_frame: str | None = None) -> "ArmActionSpec":
        return cls(
            side=side,
            controlled_frame=ARM_ENDPOINT_FRAME,
            reference_frame=reference_frame,
            semantics=ARM_ENDPOINT_SEMANTICS,
        )

    @classmethod
    def from_metadata(cls, payload: Mapping[str, Any]) -> "ArmActionSpec":
        return cls(
            side=str(payload["side"]),
            controlled_frame=str(payload.get("controlled_frame", TOOL_TCP_FRAME)),
            reference_frame=str(payload.get("reference_frame") or default_arm_frame(str(payload["side"]))),
            representation=str(payload.get("representation", "xyz_quat")),
            dof=int(payload.get("dof", ARM_POSE_DOF)),
            semantics=str(payload.get("semantics", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "controlled_frame": self.controlled_frame,
            "reference_frame": self.reference_frame,
            "representation": self.representation,
            "dof": int(self.dof),
            "semantics": self.semantics,
            "metadata": dict(self.metadata),
        }


def normalize_arm_action_specs(
    specs: Mapping[str, ArmActionSpec | Mapping[str, Any]] | None,
    *,
    active_sides: tuple[str, ...] = VALID_SIDES,
) -> dict[str, ArmActionSpec]:
    normalized: dict[str, ArmActionSpec] = {}
    for side in active_sides:
        side = normalize_side(side)
        payload = None if specs is None else specs.get(side)
        if isinstance(payload, ArmActionSpec):
            normalized[side] = payload
        elif isinstance(payload, Mapping):
            normalized[side] = ArmActionSpec.from_metadata(payload)
        else:
            normalized[side] = ArmActionSpec.tool_tcp(side)
    return normalized


def arm_action_specs_metadata(specs: Mapping[str, ArmActionSpec]) -> dict[str, Any]:
    return {normalize_side(side): spec.to_metadata() for side, spec in specs.items()}


def arm_action_specs_from_metadata(metadata: Mapping[str, Any] | None) -> dict[str, ArmActionSpec]:
    if not metadata:
        return {}
    payload = metadata.get("arm_action_specs_by_side")
    if not isinstance(payload, Mapping):
        return {}
    specs: dict[str, ArmActionSpec] = {}
    for side, spec_payload in payload.items():
        if isinstance(spec_payload, ArmActionSpec):
            spec = spec_payload
        elif isinstance(spec_payload, Mapping):
            spec = ArmActionSpec.from_metadata(spec_payload)
        else:
            continue
        specs[spec.side] = spec
    return specs


__all__ = [
    "ArmActionSpec",
    "arm_action_specs_from_metadata",
    "arm_action_specs_metadata",
    "normalize_arm_action_specs",
]
