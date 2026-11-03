"""Object-relative motion proposals; observation and execution remain lower-owned."""

import math
from collections.abc import Mapping
from dataclasses import replace

from agent_gateway.protocol.skill_catalog import (
    SkillDefinition, agent_skill_schema_document_hash,
)
from agent_gateway.skills.motion.schemas import (
    _pose,
    _pose_schema,
    _required,
    _schema_entry,
)

GRASPED_OBJECT_MOVE_SKILL_ID = "motion.move_grasped_object_to_pose"
OBJECT_MOVE_PHASES = ("lift", "carry", "place")
OBJECT_ATTITUDE_PHASES_V3 = ("lift", "carry", "release_approach")
OBJECT_INCLINATION_DEVIATION_RAD_V3 = math.radians(5.)
OBJECT_MOVE_LINEAGE_FIELDS = (
    "reference_frame_id", "reference_frame_spec_hash", "controlled_frame_id",
    "frame_graph_hash", "tool_config_hash", "calibration_hash",
)


def grasped_object_move_definition(*, controlled_component_ids):
    return SkillDefinition(
        skill_id=GRASPED_OBJECT_MOVE_SKILL_ID, skill_version="1", skill_kind="motion",
        summary=(
            "Move a deployment-bound grasped object to an absolute object-body pose. "
            "The lower owner remembers the measured hand-object transform and checks "
            "the complete carried geometry, robot and world. Lift first establishes "
            "physical following: until holding is verified, a lift must raise the "
            "object at least 3.5 cm above where the grasp was frozen. Carry requires "
            "verified holding; place permits only the named support contact. If the "
            "hand arrives but the measured object state does not, the operation stops "
            "with a reason instead of waiting. Does not close/open the gripper or "
            "certify release. No unchecked TCP fallback."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=tuple(controlled_component_ids),
        parameter_semantics_id="grasped_object.absolute.xyz_quat_xyzw.v1",
        execution_result_semantics_id="component_action_execution.v1",
        task_outcome_semantics_id="grasped_object_pose_arrival.v1",
        entry_predicate_ids=("route_ready.v1", "correlated_hand_object_contact.v1",
                             "configured_scene_collision_free_path_found.v1"),
        task_outcome_predicate_ids=("grasped_object_pose_within_tolerance.v1",),
    )


def grasped_object_place_definition_v2(*, controlled_component_ids):
    """A selected support identity is required; gripper release stays separate."""
    return replace(
        grasped_object_move_definition(controlled_component_ids=controlled_component_ids),
        skill_version="2",
        summary=("Place the already verified held object onto one explicitly bound "
                 "object support. Retain the frozen hand-object relation, complete "
                 "collision path and measured support identity; do not open the hand "
                 "or infer stable release."),
        parameter_semantics_id="grasped_object.place_on_bound_support.v2",
        task_outcome_semantics_id="grasped_object_selected_support_arrival.v2",
    )


def grasped_object_move_definition_v3(*, controlled_component_ids):
    return replace(
        grasped_object_move_definition(controlled_component_ids=controlled_component_ids),
        skill_version="3",
        summary=(
            "Move the held object through its measured hand-object relation, including an "
            "arbitrarily oriented grasp. Lift and carry preserve the inclination of the "
            "same material box axis frozen at grasp establishment within 5 degrees; yaw "
            "about world vertical remains free. This constrains the object, not a horizontal "
            "hand pose or a locked quaternion. Use phase=release_approach and the named "
            "support_object_id for a leveled release goal: the object stays unsupported "
            "while its inclination changes within the bounded interval to that support. "
            "Pass a query_support_release_pose@4 object goal unchanged for this phase, "
            "then open with that release_support_object_id and observe stable release. "
            "The original grasp relation, collision, measured following and HOLD checks "
            "remain active. Unsupported or unreachable orientations are refused; motion "
            "completion alone is not released placement."
        ),
        parameter_semantics_id="grasped_object.yaw_free_inclination.v3",
        task_outcome_semantics_id="grasped_object_inclination_bounded_arrival.v3",
    )


def grasped_object_move_schema_v3(*, definition, profile, object_id,
                                  support_object_ids, observation_source, clock_domain_id):
    base = grasped_object_move_schema(definition=definition, profile=profile, object_id=object_id,
        observation_source=observation_source, clock_domain_id=clock_domain_id)
    parameters = dict(base.parameter_schema)
    properties = dict(parameters["properties"])
    properties["phase"] = {"enum": list(OBJECT_ATTITUDE_PHASES_V3)}
    properties["support_object_id"] = {"enum": list(support_object_ids)}
    parameters["properties"] = properties
    parameters["allOf"] = [{"if": {"properties": {"phase": {"const": "release_approach"}}},
        "then": {"required": ["support_object_id"]},
        "else": {"not": {"required": ["support_object_id"]}}}]
    result = dict(base.execution_result_schema)
    result_properties = dict(result["properties"])
    state = dict(result_properties["object_state"])
    state_properties = dict(state["properties"])
    state_properties["phase"] = {"enum": list(OBJECT_ATTITUDE_PHASES_V3)}
    state_properties["support_contact"] = {"const": False}
    state_properties["support_object_id"] = {"enum": list(support_object_ids)}
    state["properties"] = state_properties
    state["allOf"] = parameters["allOf"]
    result_properties["object_state"] = state
    result["properties"] = result_properties
    schema_hash = agent_skill_schema_document_hash(parameters, schema_kind="parameter")
    return replace(base, skill_version="3", parameter_schema=parameters, execution_result_schema=result,
        parameter_profiles=tuple(replace(item, parameter_schema_hash=schema_hash) for item in base.parameter_profiles))


def validate_grasped_object_move_v3(parameters, *, profile, object_id, support_object_ids):
    if not isinstance(parameters, Mapping) or parameters.get("phase") not in OBJECT_ATTITUDE_PHASES_V3:
        raise ValueError("Object motion v3 requires lift, carry or release_approach.")
    release = parameters["phase"] == "release_approach"
    if (release and parameters.get("support_object_id") not in tuple(support_object_ids)
            or not release and "support_object_id" in parameters):
        raise ValueError("Only release_approach binds an admitted named support.")
    original = {key: value for key, value in parameters.items() if key != "support_object_id"}
    if release:
        original["phase"] = "carry"
    validate_grasped_object_move(original, profile=profile, object_id=object_id)


def validate_object_motion_state_v3(state, *, profile, object_id, observation_source,
                                    clock_domain_id, support_object_ids, parameters=None):
    if not isinstance(state, Mapping) or state.get("phase") not in OBJECT_ATTITUDE_PHASES_V3:
        raise ValueError("Object motion v3 terminal phase is invalid.")
    release = state["phase"] == "release_approach"
    if (release and state.get("support_object_id") not in tuple(support_object_ids)
            or not release and "support_object_id" in state):
        raise ValueError("Object motion v3 terminal support differs from its phase.")
    if parameters is not None and (state["phase"] != parameters["phase"]
            or state.get("support_object_id") != parameters.get("support_object_id")):
        raise ValueError("Object motion v3 terminal context differs from its request.")
    original = {key: value for key, value in state.items() if key != "support_object_id"}
    if release:
        original["phase"] = "carry"
    validate_object_motion_state(original, profile=profile, object_id=object_id,
        observation_source=observation_source, clock_domain_id=clock_domain_id)


def grasped_object_place_schema_v2(*, definition, profile, object_id,
                                    support_object_ids, observation_source, clock_domain_id):
    support_object_ids = tuple(support_object_ids)
    if (definition.skill_version != "2" or not support_object_ids
            or len(set(support_object_ids)) != len(support_object_ids)
            or object_id in support_object_ids):
        raise ValueError("Place v2 requires distinct deployed support object IDs.")
    base = grasped_object_move_schema(
        definition=definition, profile=profile, object_id=object_id,
        observation_source=observation_source, clock_domain_id=clock_domain_id)
    parameters = dict(base.parameter_schema)
    properties = dict(parameters["properties"])
    properties["phase"] = {"const": "place"}
    properties["support_object_id"] = {"type": "string", "enum": list(support_object_ids)}
    parameters["properties"] = properties
    parameters["required"] = [*parameters["required"], "support_object_id"]
    result = dict(base.execution_result_schema)
    result_properties = dict(result["properties"])
    state = dict(result_properties["object_state"])
    state_properties = dict(state["properties"])
    state_properties["support_object_id"] = {"type": "string", "enum": list(support_object_ids)}
    state["properties"] = state_properties
    state["required"] = [*state["required"], "support_object_id"]
    result_properties["object_state"] = state
    result["properties"] = result_properties
    schema_hash = agent_skill_schema_document_hash(parameters, schema_kind="parameter")
    return replace(
        base, skill_version="2", parameter_schema=parameters,
        execution_result_schema=result,
        parameter_profiles=tuple(replace(item, parameter_schema_hash=schema_hash)
                                 for item in base.parameter_profiles),
    )


def validate_grasped_object_place_v2(parameters, *, profile, object_id, support_object_ids):
    if not isinstance(parameters, Mapping) or "support_object_id" not in parameters:
        raise ValueError("Place v2 requires a selected support object ID.")
    support_id = parameters["support_object_id"]
    if support_id not in tuple(support_object_ids) or parameters.get("phase") != "place":
        raise ValueError("Place v2 support and phase must match the deployed allowlist.")
    validate_grasped_object_move(
        {key: value for key, value in parameters.items() if key != "support_object_id"},
        profile=profile, object_id=object_id)


def validate_object_motion_state_v2(state, *, parameters, profile, object_id,
                                    observation_source, clock_domain_id):
    if (not isinstance(state, Mapping)
            or state.get("support_object_id") != parameters["support_object_id"]):
        raise ValueError("Object terminal support differs from the admitted request.")
    validate_object_motion_state(
        {key: value for key, value in state.items() if key != "support_object_id"},
        profile=profile, object_id=object_id,
        observation_source=observation_source, clock_domain_id=clock_domain_id)


def grasped_object_move_schema(*, definition, profile, object_id, observation_source, clock_domain_id):
    for name, value in (("object_id", object_id), ("observation_source", observation_source),
                        ("clock_domain_id", clock_domain_id)):
        _required(value, field_name=name)
    properties = {name: {"const": getattr(profile, name)}
                  for name in OBJECT_MOVE_LINEAGE_FIELDS if name != "controlled_frame_id"}
    properties.update({
        "controlled_frame_id": {"const": "object_body"},
        "component_id": {"const": profile.component_id},
        "object_id": {"const": object_id},
        "phase": {"type": "string", "enum": list(OBJECT_MOVE_PHASES)},
        "object_pose_xyz_quat_xyzw": _pose_schema(workspace_bounds=profile.workspace_bounds),
        "translation_delta_xyz_m": {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": {"type": "number"},
            "description": "Offset the currently measured object in the public reference frame; preserve its orientation.",
        },
    })
    target_fields = {"object_pose_xyz_quat_xyzw", "translation_delta_xyz_m"}
    entry = _schema_entry(
        skill_id=definition.skill_id, skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        parameter_schema={"type": "object", "additionalProperties": False,
                          "properties": properties,
                          "required": [key for key in properties if key not in target_fields],
                          "oneOf": [{"required": [name], "not": {"required": [other]}}
                                    for name, other in (("object_pose_xyz_quat_xyzw", "translation_delta_xyz_m"),
                                                        ("translation_delta_xyz_m", "object_pose_xyz_quat_xyzw"))]},
    )
    state = {
        "object_id": {"const": object_id}, "phase": {"enum": list(OBJECT_MOVE_PHASES)},
        "reference_frame_id": {"const": profile.reference_frame_id},
        "pose_xyz_quat_xyzw": _pose_schema(workspace_bounds=profile.workspace_bounds),
        "measured_at": {"type": "number", "minimum": 0},
        "clock_domain_id": {"const": clock_domain_id},
        "holding_verified": {"const": True}, "support_contact": {"type": "boolean"},
        "source": {"const": observation_source},
    }
    result = dict(entry.execution_result_schema)
    result["properties"] = {**result["properties"], "object_state": {
        "type": "object", "additionalProperties": False,
        "properties": state, "required": list(state)}}
    result["required"] = [*result["required"], "object_state"]
    return replace(entry, execution_result_schema=result)


def validate_object_motion_state(state, *, profile, object_id, observation_source, clock_domain_id):
    fields = {"object_id", "phase", "reference_frame_id", "pose_xyz_quat_xyzw", "measured_at", "clock_domain_id",
              "holding_verified", "support_contact", "source"}
    if not isinstance(state, Mapping) or set(state) != fields:
        raise ValueError("Object motion result needs exact measured object state.")
    if (state["object_id"] != object_id or state["reference_frame_id"] != profile.reference_frame_id
            or state["phase"] not in OBJECT_MOVE_PHASES or state["source"] != observation_source
            or state["clock_domain_id"] != clock_domain_id
            or state["holding_verified"] is not True or type(state["support_contact"]) is not bool):
        raise ValueError("Object motion result identity, holding evidence or contact is invalid.")
    if state["support_contact"] is not (state["phase"] == "place"):
        raise ValueError("Object motion final support contact differs from its phase.")
    stamp = state["measured_at"]
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp < 0:
        raise ValueError("Object result needs finite nonnegative time in the declared clock domain.")
    _pose(state["pose_xyz_quat_xyzw"], field_name="object_state.pose_xyz_quat_xyzw")


def validate_grasped_object_move(parameters, *, profile, object_id):
    exact = {name: getattr(profile, name) for name in OBJECT_MOVE_LINEAGE_FIELDS
             if name != "controlled_frame_id"}
    exact.update(controlled_frame_id="object_body", component_id=profile.component_id,
                 object_id=object_id)
    targets = {"object_pose_xyz_quat_xyzw", "translation_delta_xyz_m"}
    if (not isinstance(parameters, Mapping) or len(targets.intersection(parameters)) != 1
            or set(parameters) - targets != {*exact, "phase"}):
        raise ValueError("Object motion requires the exact deployed object-pose schema.")
    if any(parameters[key] != value for key, value in exact.items()):
        raise ValueError("Object motion identity, frame or calibration mismatch.")
    if parameters["phase"] not in OBJECT_MOVE_PHASES:
        raise ValueError("Object motion phase must be lift, carry or place.")
    if "object_pose_xyz_quat_xyzw" in parameters:
        pose = _pose(parameters["object_pose_xyz_quat_xyzw"], field_name="object_pose_xyz_quat_xyzw")
        if any(not profile.workspace_bounds[2 * i] <= pose[i] <= profile.workspace_bounds[2 * i + 1]
               for i in range(3)):
            raise ValueError("Object goal is outside the deployed workspace.")
    else:
        delta = parameters["translation_delta_xyz_m"]
        if (not isinstance(delta, (tuple, list)) or len(delta) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       for v in delta)):
            raise ValueError("Object translation must contain exactly three finite numbers.")
