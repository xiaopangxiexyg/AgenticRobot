"""Explicit region samplers over the existing authenticated two-stage query."""

import math
from collections.abc import Mapping
from dataclasses import replace

from agent_gateway.protocol.skill_catalog import (
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)
from agent_gateway.skills.perception.relation_continuation_query import (
    PARAMETER_KEYS as RELATION_KEYS,
)
from agent_gateway.skills.perception.relation_continuation_query import (
    relation_continuation_definition,
    validate_continuation_fields,
)
from agent_gateway.skills.perception.spatial_goal import (
    finite_vector,
    validate_spatial_goal_geometry,
)

REGION_FIELDS = frozenset({
    "offset_bounds_xyz_m", "tool_axis_rotation_bounds_rad", "max_candidates",
})
PARAMETER_KEYS = (RELATION_KEYS - {"offsets_xyz_m", "tool_axis_rotation_offsets_rad"}) | REGION_FIELDS
SAMPLING_METHOD = "center_then_halton_unscrambled.v1"
SAMPLING_METHODS = {
    "4": SAMPLING_METHOD,
    "5": "center_continuation_extrema_then_halton_unscrambled.v1",
}


def validate_region_continuation_version(version):
    if type(version) is not str or version not in SAMPLING_METHODS:
        raise ValueError("Region continuation version must be an explicit '4' or '5'.")


def region_continuation_definition(*, version="4"):
    validate_region_continuation_version(version)
    sampling = (
        "(1..8) paired position/rotation proposals: centre first, then deterministic unscrambled Halton "
        "draws starting at index zero over non-degenerate X/Y/Z/rotation dimensions in that order. "
        "Make exactly max_candidates-1 Halton draws; remove periodic duplicates without replacement "
        "draws (an entirely fixed region returns one proposal). "
        if version == "4" else
        "(1..8) paired position/rotation proposals: centre first, then opposite/along-continuation "
        "centre-line extrema of the offset box with midpoint rotation, then the existing unscrambled "
        "Halton sequence over non-degenerate X/Y/Z/rotation dimensions. The continuation direction "
        "is expressed in the offset frame using the original TCP rotation for tool offsets. Fixed "
        "axes contribute no direction. Zero delta or no free-axis projection keeps v4 sampling. "
        "Remove periodic duplicates and truncate to max_candidates without replacement draws. "
    )
    return replace(relation_continuation_definition(), skill_version=version,
        parameter_semantics_id=f"artifact_relation_region_continuation_query.v{version}",
        execution_result_semantics_id=f"anchored_region_two_stage_evidence.v{version}",
        summary=("Search an allowed goal region relative to one authenticated surface_anchor v2. "
            "Provide three closed [lower,upper] offset intervals in metres and one closed tool-axis "
            "rotation interval in radians (width at most 2*pi), not enumerated poses or guessed XYZ. "
            "Offsets use the public reference or original input-tool axes. Generate at most max_candidates "
            + sampling + "This is not a Cartesian "
            "product or exhaustive region search. Preserve mode requires a single periodic-zero rotation. "
            "Use the freshly frozen measured TCP and the same frozen scene to check both stages of each "
            "proposal. Propagate first endpoint qpos/FK into the continuation, retaining the existing "
            "continuation_delta_xyz_m and optional second-stage tool-axis cone. The 0.1..30 second whole "
            "budget and existing per-path cap, collision, envelope and ranking checks are unchanged. "
            "Return exact generated first_pose_candidates_xyz_quat_xyzw and complete/partial path evidence. "
            "No path among finite samples or budget exhaustion is NOT region infeasibility. Relations "
            "constrain endpoints only; no motion, authority, hidden-object geometry or continuous-path "
            "guarantee. Actual moves still require fresh safe planning and measured arrival."))


def _closed_interval(value, *, name):
    try:
        lower, upper = finite_vector(value, 2, name=name)
    except OverflowError as exc:
        raise ValueError(f"{name} endpoints must be finite numbers.") from exc
    if lower > upper or not math.isfinite(upper - lower):
        raise ValueError(f"{name} must be ordered with a finite width.")
    return lower, upper


def validate_region_continuation_parameters(value, *, version="4"):
    if not isinstance(value, Mapping) or set(value) != PARAMETER_KEYS:
        raise ValueError("Region continuation requires closed bounds, not caller poses, anchors or sample arrays.")
    validate_region_continuation_version(version)
    if type(value["max_candidates"]) is not int or not 1 <= value["max_candidates"] <= 8:
        raise ValueError("max_candidates must be an integer in 1..8.")
    bounds = value["offset_bounds_xyz_m"]
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 3:
        raise ValueError("offset_bounds_xyz_m must contain three lower/upper pairs.")
    for index, interval in enumerate(bounds):
        _closed_interval(interval, name=f"offset_bounds_xyz_m[{index}]")
    lower, upper = _closed_interval(value["tool_axis_rotation_bounds_rad"],
        name="tool_axis_rotation_bounds_rad")
    if upper - lower > math.tau:
        raise ValueError("Tool-axis rotation bounds must span at most 2*pi.")
    validate_spatial_goal_geometry(**{key: value[key] for key in (
        "offset_frame", "orientation_mode", "tool_axis", "axis_reference_xyz")})
    if value["orientation_mode"] == "preserve" and (
            lower != upper or abs((lower + math.pi) % math.tau - math.pi) > 1e-12):
        raise ValueError("Preserve mode requires a fixed zero/periodic-zero rotation.")
    validate_continuation_fields(value)


def region_continuation_schema(*, definition, relation_schema, profile_hash):
    parameters = relation_schema.to_wire()["parameter_schema"]
    properties = parameters["properties"]
    del properties["offsets_xyz_m"], properties["tool_axis_rotation_offsets_rad"]
    interval = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}
    properties.update({
        "offset_bounds_xyz_m": {"type": "array", "minItems": 3, "maxItems": 3, "items": interval},
        "tool_axis_rotation_bounds_rad": interval,
        "max_candidates": {"type": "integer", "minimum": 1, "maximum": 8},
    })
    parameters["required"] = sorted(PARAMETER_KEYS)
    version = definition.skill_version
    validate_region_continuation_version(version)
    return replace(relation_schema, skill_version=version, skill_definition_hash=definition.semantic_content_hash,
        parameter_schema=parameters, parameter_profiles=(SkillParameterProfileRef(
            parameter_profile_id=f"region-tcp-continuation-query-v{version}", parameter_profile_hash=profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter")),))
