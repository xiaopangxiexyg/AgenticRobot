"""Task-level contact labels built on top of generic MuJoCo contact facts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...scene import MujocoContact
from ...transforms import quat_wxyz_to_matrix


def classify_cube_contact_position(
    contact_position_world_m: np.ndarray,
    cube_position_world_m: np.ndarray,
    cube_quaternion_wxyz: np.ndarray,
    cube_half_extents_m: np.ndarray,
    *,
    face_threshold: float = 0.65,
) -> tuple[float, float]:
    """Return ``(side_score, top_bottom_score)`` for one cube contact.

    The contact point is expressed in world coordinates by MuJoCo.  Transforming
    it into the cube frame keeps the label valid if the free cube rotates.  A
    point close to the +/-X or +/-Y faces is a side contact; a point close to
    the +/-Z faces is a top/bottom contact.  Scores ramp over the inner face
    band instead of switching discontinuously at the threshold, which gives
    PPO a useful signal for grazing contacts and cube edges.
    """

    point = np.asarray(contact_position_world_m, dtype=float).reshape(3)
    cube_position = np.asarray(cube_position_world_m, dtype=float).reshape(3)
    half_extents = np.asarray(cube_half_extents_m, dtype=float).reshape(3)
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(cube_position)):
        raise ValueError("Cube contact positions must be finite.")
    if not np.all(np.isfinite(half_extents)) or np.any(half_extents <= 0.0):
        raise ValueError("Cube half extents must be finite and positive.")
    if not 0.0 < float(face_threshold) < 1.0:
        raise ValueError("face_threshold must lie strictly inside (0, 1).")

    rotation_world_from_cube = quat_wxyz_to_matrix(cube_quaternion_wxyz)
    local = rotation_world_from_cube.T @ (point - cube_position)
    normalized = np.clip(np.abs(local) / half_extents, 0.0, 1.5)
    ramp = 1.0 - float(face_threshold)
    top_bottom_score = float(
        np.clip((normalized[2] - float(face_threshold)) / ramp, 0.0, 1.0)
    )
    side_axis = float(max(normalized[0], normalized[1]))
    side_score = float(
        np.clip((side_axis - float(face_threshold)) / ramp, 0.0, 1.0)
    )
    # An edge may be close to both a side and a top face.  Treat the Z face as
    # the dominant class so a top impact cannot earn the full side reward.
    side_score *= 1.0 - top_bottom_score
    return side_score, top_bottom_score


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


__all__ = [
    "CubeLiftContactSummary",
    "classify_cube_contact_position",
    "summarize_cube_contacts",
]
