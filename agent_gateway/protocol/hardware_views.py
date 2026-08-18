"""Small, non-authoritative hardware views published by a Gateway profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._contract_validation import (
    enum_value,
    identifier,
    nonnegative_integer,
    required_text,
)
from .skill_catalog import (
    AGENT_SKILL_TARGET_KINDS_V1,
    validate_skill_id,
    validate_version,
)


HARDWARE_RESOURCE_VIEW_SCHEMA_V1 = "dexteleop.hardware_resource_view.v1"
HARDWARE_STATUS_VIEW_SCHEMA_V1 = "dexteleop.hardware_status_view.v1"
HARDWARE_HEALTH_STATES_V1 = ("ready", "unavailable", "fault")
HARDWARE_OCCUPANCY_STATES_V1 = ("free", "occupied")


@dataclass(frozen=True, order=True)
class HardwareSkillRef:
    """Exact semantic Skill identity; Route details stay in the Route catalog."""

    skill_id: str
    skill_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_id", validate_skill_id(self.skill_id))
        object.__setattr__(
            self,
            "skill_version",
            validate_version(self.skill_version, field_name="skill_version"),
        )

    def to_wire(self) -> dict[str, str]:
        return {
            "skill_id": self.skill_id,
            "skill_version": self.skill_version,
        }


def _skill_refs(
    values: tuple[HardwareSkillRef, ...],
    *,
    field_name: str,
) -> tuple[HardwareSkillRef, ...]:
    values = tuple(values)
    if any(not isinstance(item, HardwareSkillRef) for item in values):
        raise TypeError(f"{field_name} must contain HardwareSkillRef values.")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return tuple(sorted(values))


@dataclass(frozen=True)
class HardwareResourceView:
    """Redacted read/drive capabilities for one profile-owned resource."""

    resource_id: str
    resource_kind: str
    target_kind: str
    read_skill_refs: tuple[HardwareSkillRef, ...] = ()
    drive_skill_refs: tuple[HardwareSkillRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_id",
            identifier(self.resource_id, field_name="resource_id"),
        )
        object.__setattr__(
            self,
            "resource_kind",
            identifier(self.resource_kind, field_name="resource_kind"),
        )
        object.__setattr__(
            self,
            "target_kind",
            enum_value(
                self.target_kind,
                field_name="target_kind",
                allowed=AGENT_SKILL_TARGET_KINDS_V1,
            ),
        )
        reads = _skill_refs(self.read_skill_refs, field_name="read_skill_refs")
        drives = _skill_refs(self.drive_skill_refs, field_name="drive_skill_refs")
        if set(reads) & set(drives):
            raise ValueError("One Skill cannot be both read and drive.")
        if not reads and not drives:
            raise ValueError("A hardware view must publish a read or drive Skill.")
        object.__setattr__(self, "read_skill_refs", reads)
        object.__setattr__(self, "drive_skill_refs", drives)

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": HARDWARE_RESOURCE_VIEW_SCHEMA_V1,
            "resource_id": self.resource_id,
            "resource_kind": self.resource_kind,
            "target_kind": self.target_kind,
            "properties": {
                "readable": bool(self.read_skill_refs),
                "drivable": bool(self.drive_skill_refs),
            },
            "read_skill_refs": [item.to_wire() for item in self.read_skill_refs],
            "drive_skill_refs": [item.to_wire() for item in self.drive_skill_refs],
        }


@dataclass(frozen=True)
class HardwareStatusView:
    """Bounded occupancy/health projection, never a lower safety receipt."""

    resource_id: str
    occupancy: str
    health: str
    station_revision: int
    reason_code: str
    active_operation_id: str | None = None
    active_skill_ref: HardwareSkillRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_id",
            identifier(self.resource_id, field_name="resource_id"),
        )
        object.__setattr__(
            self,
            "occupancy",
            enum_value(
                self.occupancy,
                field_name="occupancy",
                allowed=HARDWARE_OCCUPANCY_STATES_V1,
            ),
        )
        object.__setattr__(
            self,
            "health",
            enum_value(
                self.health,
                field_name="health",
                allowed=HARDWARE_HEALTH_STATES_V1,
            ),
        )
        object.__setattr__(
            self,
            "station_revision",
            nonnegative_integer(
                self.station_revision,
                field_name="station_revision",
            ),
        )
        object.__setattr__(
            self,
            "reason_code",
            required_text(self.reason_code, field_name="reason_code"),
        )
        if self.active_operation_id is not None:
            object.__setattr__(
                self,
                "active_operation_id",
                identifier(
                    self.active_operation_id,
                    field_name="active_operation_id",
                ),
            )
        if self.active_skill_ref is not None and not isinstance(
            self.active_skill_ref,
            HardwareSkillRef,
        ):
            raise TypeError("active_skill_ref must be HardwareSkillRef or None.")

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": HARDWARE_STATUS_VIEW_SCHEMA_V1,
            "resource_id": self.resource_id,
            "occupancy": self.occupancy,
            "health": self.health,
            "station_revision": self.station_revision,
            "reason_code": self.reason_code,
            "active_operation_id": self.active_operation_id,
            "active_skill_ref": (
                None
                if self.active_skill_ref is None
                else self.active_skill_ref.to_wire()
            ),
        }


__all__ = [
    "HARDWARE_HEALTH_STATES_V1",
    "HARDWARE_OCCUPANCY_STATES_V1",
    "HARDWARE_RESOURCE_VIEW_SCHEMA_V1",
    "HARDWARE_STATUS_VIEW_SCHEMA_V1",
    "HardwareResourceView",
    "HardwareSkillRef",
    "HardwareStatusView",
]
