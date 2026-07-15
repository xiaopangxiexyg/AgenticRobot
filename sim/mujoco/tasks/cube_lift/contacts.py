"""Task-level contact labels built on top of generic MuJoCo contact facts."""

from __future__ import annotations

from dataclasses import dataclass

from ...scene import MujocoContact


@dataclass(frozen=True)
class CubeLiftContactSummary:
    table_contact: bool
    object_contacts: tuple[MujocoContact, ...]

    def to_info(self) -> dict[str, object]:
        return {
            "table_contact": self.table_contact,
            "object_contact_count": len(self.object_contacts),
            "object_contacts": [
                {
                    "geom1": contact.geom1_name,
                    "geom2": contact.geom2_name,
                    "body1": contact.body1_name,
                    "body2": contact.body2_name,
                    "distance_m": contact.distance_m,
                }
                for contact in self.object_contacts
            ],
        }


def summarize_cube_contacts(
    contacts: tuple[MujocoContact, ...],
    *,
    object_body_name: str,
    table_body_name: str,
) -> CubeLiftContactSummary:
    object_contacts = tuple(
        contact
        for contact in contacts
        if object_body_name in (contact.body1_name, contact.body2_name)
    )
    table_contact = any(
        table_body_name in (contact.body1_name, contact.body2_name)
        for contact in object_contacts
    )
    return CubeLiftContactSummary(
        table_contact=table_contact,
        object_contacts=object_contacts,
    )


__all__ = ["CubeLiftContactSummary", "summarize_cube_contacts"]
