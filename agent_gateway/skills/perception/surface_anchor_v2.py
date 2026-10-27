"""Explicit selection and visible-surface evidence for surface anchors v2."""

from __future__ import annotations

import math
from collections.abc import Mapping

from .surface_anchor import ANCHOR_FIELDS, _schema, capture_schema, validate_anchor

SELECTION_FIELDS = {
    "mode",
    "requested_pixel_uv",
    "language_prompt",
    "min_score",
    "algorithm_id",
    "input_rgb_content_hash",
    "mask_media_content_hash",
    "overlay_media_content_hash",
    "model_profile_id",
    "model_profile_hash",
    "mask_bounds_xyxy",
    "mask_pixel_count",
    "interior_distance_px",
    "score",
}
GEOMETRY_VECTOR_FIELDS = {
    "world_center_xyz_m",
    "world_median_xyz_m",
    "world_bounds_low_xyz_m",
    "world_bounds_high_xyz_m",
    "world_extent_xyz_m",
}
GEOMETRY_FIELDS = GEOMETRY_VECTOR_FIELDS | {"median_depth_m", "valid_point_count"}
EXACT_ALGORITHM = "exact_pixel.v1"
INTERIOR_ALGORITHM = "mask_distance_max_valid_depth.v1"
MAX_CLEARANCE_MEDOID_ALGORITHM = "mask_distance_max_valid_depth.plateau_l1_medoid.v1"
INTERIOR_SELECTION_POLICIES = {
    "prompt_nearest": INTERIOR_ALGORITHM,
    "max_clearance_medoid": MAX_CLEARANCE_MEDOID_ALGORITHM,
}


def validate_interior_algorithm(algorithm):
    if not isinstance(algorithm, str) or algorithm not in INTERIOR_SELECTION_POLICIES.values():
        raise ValueError("Unknown surface anchor interior selection algorithm.")


def capture_schema_v2(
    *, definition, back_projection_schema, segmentation_schema, profile_hash,
    interior_algorithm=INTERIOR_ALGORITHM,
):
    """Keep measured-pixel geometry, adding explicit selection evidence."""
    validate_interior_algorithm(interior_algorithm)
    base = capture_schema(
        definition=definition,
        back_projection_schema=back_projection_schema,
        profile_hash=profile_hash,
    )
    segment = segmentation_schema.parameter_schema["properties"]
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    nullable_digest = {"anyOf": [digest, {"type": "null"}]}
    optional_string = {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]}
    vector = {
        "type": "array",
        "minItems": 3,
        "maxItems": 3,
        "items": {"type": "number"},
    }
    selection = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(SELECTION_FIELDS),
        "properties": {
            "mode": {"enum": ["exact", "mask_interior"]},
            "requested_pixel_uv": segment["target_pixel_uv"],
            "language_prompt": segment["language_prompt"],
            "min_score": segment["min_score"],
            "algorithm_id": {"enum": [EXACT_ALGORITHM, interior_algorithm]},
            "input_rgb_content_hash": digest,
            **{
                key: nullable_digest
                for key in ("mask_media_content_hash", "overlay_media_content_hash")
            },
            "model_profile_id": optional_string,
            "model_profile_hash": optional_string,
            "mask_bounds_xyxy": {
                "anyOf": [
                    {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "integer", "minimum": 0},
                    },
                    {"type": "null"},
                ]
            },
            "mask_pixel_count": {
                "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]
            },
            "interior_distance_px": {
                "anyOf": [{"type": "number", "exclusiveMinimum": 0}, {"type": "null"}]
            },
            "score": {
                "anyOf": [
                    {"type": "number", "minimum": 0, "maximum": 1},
                    {"type": "null"},
                ]
            },
        },
    }
    geometry = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(GEOMETRY_FIELDS),
        "properties": {
            **{key: vector for key in GEOMETRY_VECTOR_FIELDS},
            "median_depth_m": {"type": "number", "exclusiveMinimum": 0},
            "valid_point_count": {"type": "integer", "minimum": 1},
        },
    }
    return _schema(
        definition=definition,
        profile_hash=profile_hash,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": [
                "role",
                "selection_mode",
                "target_pixel_uv",
                "language_prompt",
                "min_score",
            ],
            "properties": {
                "role": base.parameter_schema["properties"]["role"],
                "selection_mode": {"enum": ["exact", "mask_interior"]},
                **{
                    key: segment[key]
                    for key in ("target_pixel_uv", "language_prompt", "min_score")
                },
            },
        },
        result={
            "type": "object",
            "additionalProperties": False,
            "required": sorted(ANCHOR_FIELDS | {"selection", "mask_geometry"}),
            "properties": {
                **dict(base.execution_result_schema["properties"]),
                "selection": selection,
                "mask_geometry": {"anyOf": [geometry, {"type": "null"}]},
            },
        },
    )


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _digest(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def validate_anchor_v2(value, *, interior_algorithm=INTERIOR_ALGORITHM):
    """Reject mixed versions and incomplete or contradictory selection evidence."""
    validate_interior_algorithm(interior_algorithm)
    if not isinstance(value, Mapping) or set(value) != ANCHOR_FIELDS | {
        "selection",
        "mask_geometry",
    }:
        raise ValueError("Surface anchor v2 must match its exact schema.")
    validate_anchor({key: value[key] for key in ANCHOR_FIELDS})
    selection = value["selection"]
    if not isinstance(selection, Mapping) or set(selection) != SELECTION_FIELDS:
        raise ValueError("Surface anchor v2 selection evidence is incomplete.")
    if not _digest(selection["input_rgb_content_hash"]):
        raise ValueError("Surface anchor v2 must identify its exact RGB cut.")
    score = selection["min_score"]
    if not _finite(score) or not 0 <= score <= 1:
        raise ValueError("Surface anchor score threshold must be finite in [0, 1].")
    pixel = selection["requested_pixel_uv"]
    if pixel is not None and (
        not isinstance(pixel, (tuple, list))
        or len(pixel) != 2
        or any(type(x) is not int or x < 0 for x in pixel)
    ):
        raise ValueError("Surface anchor requested pixel is invalid.")
    prompt = selection["language_prompt"]
    if prompt is not None and (
        not isinstance(prompt, str)
        or not prompt
        or prompt != prompt.strip()
        or len(prompt) > 256
    ):
        raise ValueError("Surface anchor language prompt is invalid.")
    mask_fields = SELECTION_FIELDS - {
        "mode",
        "requested_pixel_uv",
        "language_prompt",
        "min_score",
        "algorithm_id",
        "input_rgb_content_hash",
    }
    if selection["mode"] == "exact":
        if (
            selection["algorithm_id"] != EXACT_ALGORITHM
            or prompt is not None
            or pixel is None
            or list(pixel) != list(value["target_pixel_uv"])
            or any(selection[key] is not None for key in mask_fields)
            or value["mask_geometry"] is not None
        ):
            raise ValueError(
                "Exact surface selection changed its pixel or added mask evidence."
            )
        return
    if (
        selection["mode"] != "mask_interior"
        or selection["algorithm_id"] != interior_algorithm
        or (pixel is None) == (prompt is None)
    ):
        raise ValueError(
            "Mask interior selection requires exactly one point or text prompt."
        )
    if (
        any(
            not _digest(selection[key])
            for key in ("mask_media_content_hash", "overlay_media_content_hash")
        )
        or not isinstance(selection["model_profile_id"], str)
        or not selection["model_profile_id"]
        or selection["model_profile_id"] != selection["model_profile_id"].strip()
        or not isinstance(selection["model_profile_hash"], str)
        or not selection["model_profile_hash"]
        or selection["model_profile_hash"] != selection["model_profile_hash"].strip()
        or not _finite(selection["score"])
        or not score <= selection["score"] <= 1
        or not _finite(selection["interior_distance_px"])
        or selection["interior_distance_px"] <= 0
        or type(selection["mask_pixel_count"]) is not int
        or selection["mask_pixel_count"] < 1
    ):
        raise ValueError("Mask interior selection metadata is invalid.")
    bounds = selection["mask_bounds_xyxy"]
    if (
        not isinstance(bounds, (tuple, list))
        or len(bounds) != 4
        or any(type(x) is not int or x < 0 for x in bounds)
        or bounds[0] >= bounds[2]
        or bounds[1] >= bounds[3]
    ):
        raise ValueError("Mask interior bounds are invalid.")
    u, v = value["target_pixel_uv"]
    if not bounds[0] <= u < bounds[2] or not bounds[1] <= v < bounds[3]:
        raise ValueError("Selected surface pixel is outside mask bounds.")
    geometry = value["mask_geometry"]
    if not isinstance(geometry, Mapping) or set(geometry) != GEOMETRY_FIELDS:
        raise ValueError("Mask interior selection requires visible depth geometry.")
    for key in GEOMETRY_VECTOR_FIELDS:
        vector = geometry[key]
        if (
            not isinstance(vector, (list, tuple))
            or len(vector) != 3
            or not all(_finite(x) for x in vector)
        ):
            raise ValueError("Visible mask geometry must contain finite 3D vectors.")
    if (
        type(geometry["valid_point_count"]) is not int
        or not 1 <= geometry["valid_point_count"] <= selection["mask_pixel_count"]
        or not _finite(geometry["median_depth_m"])
        or geometry["median_depth_m"] <= 0
    ):
        raise ValueError("Visible mask depth statistics are invalid.")
    for low, high, extent in zip(
        geometry["world_bounds_low_xyz_m"],
        geometry["world_bounds_high_xyz_m"],
        geometry["world_extent_xyz_m"],
        strict=True,
    ):
        if (
            low > high
            or extent < 0
            # The shared geometry owner rounds all three fields to 1 micrometre.
            or not math.isclose(high - low, extent, rel_tol=0, abs_tol=1.5e-6)
        ):
            raise ValueError("Visible mask geometry bounds are inconsistent.")
