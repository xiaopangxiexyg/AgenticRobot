"""Geometry-only gripper contact features for simulator rewards.

The extractor deliberately knows nothing about a cube, table, lift height, or
policy observation.  A task supplies the object body and a small surface
contract for each gripper finger.  The surface contract identifies the local
inward normal; the opposite half-space is treated as the outer surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _unit_vector(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    vector = np.asarray(values, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or norm <= 1e-12:
        raise ValueError(f"{name} must be a finite non-zero 3-vector.")
    vector /= norm
    return tuple(float(value) for value in vector)


@dataclass(frozen=True)
class GripperContactSurfaceSpec:
    """Local-frame contract for one finger's inner contact surface."""

    body_name: str
    side: str
    inner_normal_local: tuple[float, ...]
    surface_anchor_local: tuple[float, ...] = (0.0, 0.0, 0.0)
    face_margin_m: float = 0.001
    score_width_m: float = 0.002

    def __post_init__(self) -> None:
        body_name = str(self.body_name).strip()
        side = str(self.side).strip()
        if not body_name:
            raise ValueError("Gripper contact body_name must be non-empty.")
        if not side:
            raise ValueError("Gripper contact side must be non-empty.")
        anchor = np.asarray(self.surface_anchor_local, dtype=float).reshape(3)
        if not np.all(np.isfinite(anchor)):
            raise ValueError("surface_anchor_local must be finite.")
        margin = float(self.face_margin_m)
        width = float(self.score_width_m)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("face_margin_m must be finite and non-negative.")
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("score_width_m must be finite and positive.")
        object.__setattr__(self, "body_name", body_name)
        object.__setattr__(self, "side", side)
        object.__setattr__(
            self,
            "inner_normal_local",
            _unit_vector(self.inner_normal_local, name="inner_normal_local"),
        )
        object.__setattr__(
            self,
            "surface_anchor_local",
            tuple(float(value) for value in anchor),
        )
        object.__setattr__(self, "face_margin_m", margin)
        object.__setattr__(self, "score_width_m", width)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "GripperContactSurfaceSpec":
        return cls(
            body_name=str(values["body_name"]),
            side=str(values["side"]),
            inner_normal_local=tuple(values["inner_normal_local"]),
            surface_anchor_local=tuple(
                values.get("surface_anchor_local", (0.0, 0.0, 0.0))
            ),
            face_margin_m=float(values.get("face_margin_m", 0.001)),
            score_width_m=float(values.get("score_width_m", 0.002)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_name": self.body_name,
            "side": self.side,
            "inner_normal_local": list(self.inner_normal_local),
            "surface_anchor_local": list(self.surface_anchor_local),
            "face_margin_m": self.face_margin_m,
            "score_width_m": self.score_width_m,
        }


@dataclass(frozen=True)
class GripperContactScores:
    """Aggregated inner/outer contact quality for one simulation tick."""

    inner_by_side: Mapping[str, float]
    outer_by_side: Mapping[str, float]
    inner_contact_score: float
    bilateral_inner_contact_score: float
    outer_contact_score: float

    def to_info(self) -> dict[str, Any]:
        return {
            "gripper_inner_contact_score": self.inner_contact_score,
            "gripper_bilateral_inner_contact_score": (
                self.bilateral_inner_contact_score
            ),
            "gripper_outer_contact_score": self.outer_contact_score,
            "gripper_inner_contact_by_side": dict(self.inner_by_side),
            "gripper_outer_contact_by_side": dict(self.outer_by_side),
        }


def classify_gripper_contact_point(
    contact_position_world_m: np.ndarray,
    body_position_world_m: np.ndarray,
    body_rotation_world_from_local: np.ndarray,
    spec: GripperContactSurfaceSpec,
) -> tuple[float, float]:
    """Return ``(inner_score, outer_score)`` for one finger contact point."""

    point = np.asarray(contact_position_world_m, dtype=float).reshape(3)
    body_position = np.asarray(body_position_world_m, dtype=float).reshape(3)
    rotation = np.asarray(body_rotation_world_from_local, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(body_position)):
        raise ValueError("Contact and body positions must be finite.")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("Body rotation must be finite.")
    local_point = rotation.T @ (point - body_position)
    relative = local_point - np.asarray(spec.surface_anchor_local, dtype=float)
    inward_coordinate = float(
        np.dot(relative, np.asarray(spec.inner_normal_local, dtype=float))
    )
    margin = spec.face_margin_m
    width = spec.score_width_m
    inner_score = float(
        1.0 / (1.0 + np.exp(np.clip(-(inward_coordinate - margin) / width, -60.0, 60.0)))
    )
    outer_score = float(
        1.0 / (1.0 + np.exp(np.clip(-(-inward_coordinate - margin) / width, -60.0, 60.0)))
    )
    return inner_score, outer_score


def extract_gripper_contact_scores(
    model: Any,
    data: Any,
    *,
    object_body_id: int,
    surface_specs: Sequence[GripperContactSurfaceSpec],
    surface_body_ids: Mapping[str, int] | None = None,
) -> GripperContactScores:
    """Extract inner/outer finger contact scores from the MuJoCo buffer."""

    if not surface_specs:
        return GripperContactScores({}, {}, 0.0, 0.0, 0.0)
    body_ids = (
        {
            spec.body_name: _body_id(model, spec.body_name)
            for spec in surface_specs
        }
        if surface_body_ids is None
        else {str(name): int(body_id) for name, body_id in surface_body_ids.items()}
    )
    specs_by_body_id = {
        body_ids[spec.body_name]: spec for spec in surface_specs
    }
    inner_by_side: dict[str, float] = {}
    outer_by_side: dict[str, float] = {}
    geom_body_id = model.geom_bodyid
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        body1 = int(geom_body_id[int(contact.geom1)])
        body2 = int(geom_body_id[int(contact.geom2)])
        if body1 != int(object_body_id) and body2 != int(object_body_id):
            continue
        finger_body = body2 if body1 == int(object_body_id) else body1
        spec = specs_by_body_id.get(finger_body)
        if spec is None:
            continue
        inner_score, outer_score = classify_gripper_contact_point(
            np.asarray(contact.pos, dtype=float),
            np.asarray(data.xpos[finger_body], dtype=float),
            np.asarray(data.xmat[finger_body], dtype=float).reshape(3, 3),
            spec,
        )
        inner_by_side[spec.side] = max(inner_by_side.get(spec.side, 0.0), inner_score)
        outer_by_side[spec.side] = max(outer_by_side.get(spec.side, 0.0), outer_score)

    all_inner = tuple(inner_by_side.values())
    inner_contact = float(np.mean(all_inner)) if all_inner else 0.0
    # ``bilateral`` is a physical two-sided signal, not a second aggregate of
    # whichever contacts happened to be observed.  Treating a single finger
    # as bilateral creates a particularly bad CubeLift shortcut: the policy
    # can press one finger into the table/object and receive the lift/contact
    # shaping intended for a stable grasp.  A one-surface contract is kept
    # backwards-compatible, but a multi-finger contract requires every
    # configured side to be present in the same physics tick.
    expected_sides = {spec.side for spec in surface_specs}
    if len(expected_sides) <= 1:
        bilateral = inner_contact
    elif expected_sides.issubset(inner_by_side):
        bilateral = float(min(inner_by_side[side] for side in expected_sides))
    else:
        bilateral = 0.0
    outer_contact = float(max(outer_by_side.values())) if outer_by_side else 0.0
    return GripperContactScores(
        inner_by_side=inner_by_side,
        outer_by_side=outer_by_side,
        inner_contact_score=inner_contact,
        bilateral_inner_contact_score=bilateral,
        outer_contact_score=outer_contact,
    )


def _body_id(model: Any, body_name: str) -> int:
    import mujoco

    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
    if body_id < 0:
        raise KeyError(f"MuJoCo model has no gripper contact body {body_name!r}.")
    return body_id


__all__ = [
    "GripperContactScores",
    "GripperContactSurfaceSpec",
    "classify_gripper_contact_point",
    "extract_gripper_contact_scores",
]
