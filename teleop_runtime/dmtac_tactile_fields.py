"""Pure DMTac tactile field names, shapes, and normalization contracts.

This module deliberately has no backend, SDK, worker, or registry imports so
CLI and configuration code can validate field selections without initializing
the DMTac runtime integration.
"""

from __future__ import annotations

from typing import Any


DMTAC_DISTRIBUTED_FORCE_SHAPE = (288, 384, 3)
DMTAC_RAW_IMAGE_SHAPE = (480, 640)
DMTAC_TACTILE_FIELDS = (
    "raw",
    "infer",
    "deformation2d",
    "shear",
    "depth",
    "force",
    "distributed_force",
    "contact_area",
    "base_frame",
    "events",
)
DMTAC_TACTILE_ARRAY_FIELDS = (
    "raw",
    "infer",
    "deformation2d",
    "shear",
    "depth",
    "force",
    "distributed_force",
)
DMTAC_DEFAULT_TACTILE_FIELDS = ("raw",)
# These solved fields can optionally be interpreted relative to an explicitly
# captured no-contact base frame. Requesting one never authorizes an SDK reset.
DMTAC_TACTILE_CALIBRATED_FIELDS = frozenset(
    {
        "deformation2d",
        "shear",
        "depth",
        "force",
        "distributed_force",
        "contact_area",
        "base_frame",
    }
)
DMTAC_TACTILE_FIELD_DEFAULT_SHAPES = {
    "raw": DMTAC_RAW_IMAGE_SHAPE,
    "infer": (288, 384),
    "deformation2d": (288, 384, 2),
    "shear": (288, 384, 2),
    "depth": (288, 384),
    "force": (6,),
    "distributed_force": DMTAC_DISTRIBUTED_FORCE_SHAPE,
}
DMTAC_TACTILE_FIELD_ALIASES = {
    "dist_force": "distributed_force",
    "distforce": "distributed_force",
    "distributed_force": "distributed_force",
    "distributedforce": "distributed_force",
    "distributed": "distributed_force",
    "deformation": "deformation2d",
    "deformation_2d": "deformation2d",
    "base": "base_frame",
}


def _normalize_dmtac_tactile_fields(fields: Any) -> tuple[str, ...]:
    if fields is None:
        return DMTAC_DEFAULT_TACTILE_FIELDS
    if isinstance(fields, str):
        items = [part.strip() for part in fields.replace(",", " ").split() if part.strip()]
    else:
        items = list(fields)
    normalized: list[str] = []
    for item in items:
        key = str(item).strip().lower().replace("-", "_")
        key = DMTAC_TACTILE_FIELD_ALIASES.get(key, key)
        if key == "all":
            return DMTAC_TACTILE_FIELDS
        if key not in DMTAC_TACTILE_FIELDS:
            raise ValueError(
                f"Unsupported DMTAC tactile field {item!r}. Valid fields: {DMTAC_TACTILE_FIELDS}"
            )
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized) if normalized else DMTAC_DEFAULT_TACTILE_FIELDS


normalize_dmtac_tactile_fields = _normalize_dmtac_tactile_fields


def dmtac_tactile_requires_calibration(fields: Any) -> bool:
    """Whether requested outputs support an optional no-contact calibration."""

    return bool(DMTAC_TACTILE_CALIBRATED_FIELDS.intersection(_normalize_dmtac_tactile_fields(fields)))


__all__ = [
    "DMTAC_DEFAULT_TACTILE_FIELDS",
    "DMTAC_DISTRIBUTED_FORCE_SHAPE",
    "DMTAC_RAW_IMAGE_SHAPE",
    "DMTAC_TACTILE_ARRAY_FIELDS",
    "DMTAC_TACTILE_CALIBRATED_FIELDS",
    "DMTAC_TACTILE_FIELD_ALIASES",
    "DMTAC_TACTILE_FIELD_DEFAULT_SHAPES",
    "DMTAC_TACTILE_FIELDS",
    "dmtac_tactile_requires_calibration",
    "normalize_dmtac_tactile_fields",
]
