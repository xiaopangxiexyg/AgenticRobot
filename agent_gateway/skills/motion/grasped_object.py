"""Object-relative motion proposals; observation and execution remain lower-owned."""

import math
from collections.abc import Mapping
from dataclasses import replace

from agent_gateway.protocol.skill_catalog import SkillDefinition
from agent_gateway.skills.motion.schemas import (
    _pose,
    _pose_schema,
    _required,
    _schema_entry,
)

GRASPED_OBJECT_MOVE_SKILL_ID = "motion.move_grasped_object_to_pose"
OBJECT_MOVE_PHASES = ("lift", "carry", "place")
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
            "physical following; carry requires verified holding; place permits only "
            "the named support contact. Does not close/open the gripper or certify "
            "release. No unchecked TCP fallback."
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
