"""Authenticated RGB-D/mask reuse parameters with the unchanged v2 anchor document."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

from agent_gateway.protocol._contract_validation import required_text

from .surface_anchor import _schema, capture_definition
from .surface_anchor_v2 import INTERIOR_ALGORITHM, capture_schema_v2

PARAMETER_FIELDS = frozenset(
    (
        "role",
        "selection_mode",
        "target_pixel_uv",
        "rgb_capture_operation_id",
        "segmentation_operation_id",
        "target_index",
    )
)


def capture_definition_v3():
    return replace(
        capture_definition(version="2"),
        skill_version="3",
        parameter_semantics_id="authenticated_rgbd_mask_surface_pixel.v3",
        summary=(
            "Project an authenticated retained capture_rgb v3 paired observation without recapture "
            "or new model inference. Both modes require the capture_rgb v3 paired_observations "
            "output as paired_observation input. "
            "Exact mode preserves the requested pixel and requires null segmentation_operation_id "
            "and target_index. Mask mode reuses one completed segment v3 item and its exact mask; "
            "it also requires the segmentation_mask output at the selected item's "
            "mask_artifact_ordinal as segmentation_mask input. "
            "target_index is the original item index, not its mask Artifact ordinal. A null pixel "
            "inherits that item's original prompt; a non-null pixel must equal its point prompt. "
            "Returns the unchanged v2 measured anchor document with visible mask geometry, not "
            "verified identity, object pose, physical clearance, tracking or action authority."
        ),
    )


def capture_schema_v3(
    *,
    definition,
    back_projection_schema,
    segmentation_schema,
    profile_hash,
    interior_algorithm=INTERIOR_ALGORITHM,
):
    # Adapt the batch's per-target schema, not its result or executable validator.
    prompts = segmentation_schema.parameter_schema["properties"]["targets"]["items"][
        "properties"
    ]
    base = capture_schema_v2(
        definition=definition,
        back_projection_schema=back_projection_schema,
        segmentation_schema=SimpleNamespace(parameter_schema={"properties": prompts}),
        profile_hash=profile_hash,
        interior_algorithm=interior_algorithm,
    )
    identifier = {"type": "string", "minLength": 1, "maxLength": 256}
    properties = {
        "role": back_projection_schema.parameter_schema["properties"]["role"],
        "selection_mode": {"enum": ["exact", "mask_interior"]},
        "target_pixel_uv": {
            **prompts["target_pixel_uv"],
            "description": (
                "Zero-based [u, v] in the source observation's native pixel grid. "
                "Exact mode requires a non-null pixel. In mask_interior mode, null "
                "inherits the selected item's original prompt; a non-null pixel must "
                "equal that item's original point prompt."
            ),
        },
        "rgb_capture_operation_id": identifier,
        "segmentation_operation_id": {"anyOf": [identifier, {"type": "null"}]},
        "target_index": {"type": ["integer", "null"], "minimum": 0, "maximum": 2},
    }
    return _schema(
        definition=definition,
        profile_hash=profile_hash,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": sorted(PARAMETER_FIELDS),
            "properties": properties,
        },
        result=base.execution_result_schema,
    )


def validate_capture_parameters_v3(value, *, dimensions_by_role):
    if not isinstance(value, Mapping) or set(value) != PARAMETER_FIELDS:
        raise ValueError(
            "Surface anchor v3 requires exactly its explicit reuse parameters."
        )
    role = value["role"]
    if not isinstance(role, str) or role not in dimensions_by_role:
        raise ValueError("Surface anchor v3 requires a registered aligned-depth role.")
    required_text(
        value["rgb_capture_operation_id"],
        field_name="rgb_capture_operation_id",
        maximum_length=256,
    )
    pixel = value["target_pixel_uv"]
    if pixel is not None:
        width, height = dimensions_by_role[role]
        if (
            not isinstance(pixel, (tuple, list))
            or len(pixel) != 2
            or any(type(item) is not int for item in pixel)
            or not (0 <= pixel[0] < width and 0 <= pixel[1] < height)
        ):
            raise ValueError(
                "Surface anchor v3 pixel is outside the original native grid."
            )
    mode = value["selection_mode"]
    if mode == "exact":
        if (
            pixel is None
            or value["segmentation_operation_id"] is not None
            or value["target_index"] is not None
        ):
            raise ValueError(
                "Exact surface anchor requires a pixel and no segmentation selector."
            )
    elif mode == "mask_interior":
        required_text(
            value["segmentation_operation_id"],
            field_name="segmentation_operation_id",
            maximum_length=256,
        )
        if (
            type(value["target_index"]) is not int
            or not 0 <= value["target_index"] <= 2
        ):
            raise ValueError(
                "Surface anchor v3 requires an original target index in [0, 2]."
            )
    else:
        raise ValueError("Unknown surface anchor v3 selection mode.")


__all__ = [
    "capture_definition_v3",
    "capture_schema_v3",
    "validate_capture_parameters_v3",
]
