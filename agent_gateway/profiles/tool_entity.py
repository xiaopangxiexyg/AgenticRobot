"""Shared registrations and executors for the tool-entity perception Skills.

``perception.describe_tool_object_relations@1`` and
``perception.query_tool_hover_paths@1`` run the same code on the RealMan and
MuJoCo Gateways. Targets only supply typed ports:

* a scene port: stationary multi-view RGB-D with capture-time dual-arm qpos
  (``capture_views``), the segmenter's RGB input for one view
  (``segmentation_input``), tool TCP forward kinematics in the capture geometry
  frame (``tcp_pose_by_side``) and the rigid ``public_from_geometry`` transform;
* for hover, a planner port: the deployment's observed-scene cloud and its
  safe-motion planner in plan-only mode (``prepare_scene`` / ``plan``).

Geometry follows ``robot_motion.tool_entity_scene``; hover bounds are evaluated
in the public reference frame. Nothing here moves a robot or retains a plan.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from agent_gateway.profiles.segmentation import decode_segmentation_mask_npy
from agent_gateway.profiles.surface_anchor import (
    MAX_ANCHOR_BYTES,
    resolve_surface_anchor,
)
from agent_gateway.protocol.skill_catalog import SkillRouteManifestEntry
from agent_gateway.protocol.skill_io_contracts import (
    SkillArtifactSlotSpec,
    SkillIOContract,
    inline_only_skill_io_contract,
)
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from agent_gateway.skills.perception.tool_entity import (
    HOVER_SKILL_ID,
    RELATION_SCOPE,
    RELATION_SKILL_ID,
    hover_definition,
    hover_schema,
    relation_definition,
    relation_schema,
    validate_hover_parameters,
    validate_hover_query,
    validate_hover_result_for_parameters,
    validate_relation_parameters,
    validate_relation_result,
    validate_relation_result_for_parameters,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.perception_artifacts import SURFACE_ANCHOR_SCHEMA_ID_V2
from robot_motion.tool_entity_goals import ToolEntityBox
from robot_motion.tool_entity_scene import (
    ENTITY_HOVER_CANDIDATE_BUDGET_SEC,
    ENTITY_HOVER_REGION_BUDGET_SEC,
    OBJECT_MASK_EROSION_PX,
    RELATION_MAXIMUM_DEPTH_M,
    HoverPlanOutcome,
    MeasuredObjectSurface,
    describe_tool_object_relations,
    object_surface_from_mask,
    pose_matrix,
    pose_xyz_quat_xyzw,
    search_entity_hover_paths,
    validate_hover_budgets,
)

TOOL_OBJECT_RELATION_MAX_OPERATION_MS = 90_000
# Capture, cloud and result reserve on top of the hover search budget; the
# default 20 s search gives the original 40 s Route.
TOOL_HOVER_SCENE_RESERVE_SEC = 20.
TOOL_HOVER_MAX_OPERATION_MS = int((ENTITY_HOVER_REGION_BUDGET_SEC + TOOL_HOVER_SCENE_RESERVE_SEC) * 1000)
# Leave the executor time to validate and retain its result before the deadline.
_RESULT_RESERVE_SEC = 1.0
SIDES = ("left", "right")


@dataclass(frozen=True)
class ToolEntityRegistration:
    """One read-only Skill Route with its exact deployment profile identity."""

    skill: SkillRegistrationTemplate
    entry: SkillRouteManifestEntry
    payload_validator: AgentSkillPayloadValidatorSet
    profile_hash: str
    roles: tuple[str, ...]
    component_side_by_id: Mapping[str, str]
    reference_frame_id: str
    # Hover only: the deployment's search budgets (bound into the profile).
    region_budget_sec: float | None = None
    candidate_budget_sec: float | None = None


def _identity(entities):
    return {side: entity.identity() for side, entity in sorted(entities.items())}


def _clone_route(base_route, *, definition, schema, io_contract, suffix, max_operation_duration_ms):
    if base_route.control_effect_class != "read_only" or base_route.may_move_robot:
        raise ValueError("Tool-entity Skills require a read-only base Route.")
    return replace(base_route, skill_id=definition.skill_id, skill_version="1",
        skill_definition_hash=definition.semantic_content_hash, route_id=base_route.route_id + suffix,
        route_version="1", parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles, skill_io_contract_hash=io_contract.semantic_content_hash,
        max_operation_duration_ms=max_operation_duration_ms)


def _clone_binding(base_binding, *, prefix, route):
    binding = base_binding.to_ref() if hasattr(base_binding, "to_ref") else base_binding
    return replace(binding, deployment_binding_id=prefix + ":" + binding.deployment_binding_id,
        deployment_binding_hash=canonical_content_hash(
            {"base": binding.deployment_binding_hash, "route": route.semantic_content_hash},
            domain="agenticrobot.tool_entity.binding.v1"))


def build_tool_object_relation_registration(*, segmentation, back_projection, roles, tool_entity_by_side):
    """Bind relations to the deployment's text segmenter and aligned RGB-D roles."""
    roles = tuple(roles)
    if (not roles or len(set(roles)) != len(roles) or not set(roles) <= set(back_projection.profile.roles)
            or any(segmentation.profile.source_id_by_role.get(role)
                   != back_projection.profile.source_id_by_role.get(role) for role in roles)):
        raise ValueError("Relation roles need matching segmentation and aligned RGB-D sources.")
    if not tool_entity_by_side or not set(tool_entity_by_side) <= set(SIDES) or not all(
            isinstance(entity, ToolEntityBox) for entity in tool_entity_by_side.values()):
        raise ValueError("Relations need at least one reviewed tool entity keyed by arm.")
    for key in ("runtime_session_id", "control_domain_id", "target_binding_id", "station_manifest_hash",
                "execution_manifest_hash", "target_kind"):
        if getattr(segmentation.binding, key) != getattr(back_projection.binding, key):
            raise ValueError("Relation sources belong to another deployment.")
    frame = back_projection.profile.reference_frame_id
    profile_hash = canonical_content_hash({
        "segmentation_route": segmentation.skill.routes[0].semantic_content_hash,
        "segmentation_profile": segmentation.profile.semantic_content_hash,
        "back_projection_profile": back_projection.profile.semantic_content_hash,
        "roles": list(roles), "tool_entities": _identity(tool_entity_by_side),
        "maximum_depth_m": min(RELATION_MAXIMUM_DEPTH_M, back_projection.profile.maximum_depth_m),
        "mask_erosion_px": OBJECT_MASK_EROSION_PX,
        "capture": "stationary_multiview_rgbd_with_capture_qpos",
    }, domain="agenticrobot.tool_object_relations.profile.v1")
    definition = relation_definition()
    schema = relation_schema(definition_hash=definition.semantic_content_hash, profile_hash=profile_hash,
        roles=roles, sides=SIDES, reference_frame_id=frame)
    io_contract = inline_only_skill_io_contract(skill_id=RELATION_SKILL_ID, skill_version="1",
        skill_definition_hash=definition.semantic_content_hash, parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash)
    route = _clone_route(back_projection.skill.routes[0], definition=definition, schema=schema,
        io_contract=io_contract, suffix=".tool_object_relations",
        max_operation_duration_ms=TOOL_OBJECT_RELATION_MAX_OPERATION_MS)
    binding = _clone_binding(back_projection.binding, prefix="tool-object-relations", route=route)

    def validate_segment_prompt(role, prompt, min_score):
        segmentation.payload_validator.validate_parameters(
            {"role": role, "language_prompt": prompt, "target_pixel_uv": None, "min_score": min_score})

    def validate_parameters(value):
        validate_relation_parameters(value, roles=roles, validate_segment_prompt=validate_segment_prompt)

    def validate_result(value):
        validate_relation_result(value, roles=roles, sides=SIDES, reference_frame_id=frame)

    def validate_result_for_parameters(parameters, result):
        validate_parameters(parameters)
        validate_result(result)
        validate_relation_result_for_parameters(parameters, result)

    validator = AgentSkillPayloadValidatorSet(validator_id="tool-object-relations:" + profile_hash,
        skill_id=RELATION_SKILL_ID, skill_version="1", route_id=route.route_id, route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash, validate_parameters=validate_parameters,
        validate_execution_result=validate_result,
        validate_execution_result_for_parameters=validate_result_for_parameters,
        validate_task_evaluation=validate_no_task_evaluation_v1)
    entry = SkillRouteManifestEntry(route=route, deployment_binding=binding, evaluation_plan_hash=None,
        availability_hash=canonical_content_hash({"route": route.semantic_content_hash, "profile": profile_hash,
            "status": "configured_requires_live_stationary_rgbd_text_segmentation_and_tool_entity"},
            domain="agenticrobot.tool_object_relations.availability.v1"))
    return ToolEntityRegistration(SkillRegistrationTemplate(definition, schema, io_contract, (route,)), entry,
        validator, profile_hash, roles, {}, frame)


def build_tool_hover_registration(*, capture_registration, back_projection, component_side_by_id, roles,
                                  tool_entity_by_side, planner_profile_hash,
                                  region_budget_sec=ENTITY_HOVER_REGION_BUDGET_SEC,
                                  candidate_budget_sec=ENTITY_HOVER_CANDIDATE_BUDGET_SEC):
    """Bind hover queries to an authenticated anchor producer and a safe-motion planner profile.

    ``region_budget_sec``/``candidate_budget_sec`` are this deployment's search
    budgets (defaults 20 s/3 s). They enter the profile identity and size the
    Route: max duration = region budget + capture/cloud reserve.
    """
    region_budget_sec, candidate_budget_sec = validate_hover_budgets(
        region_budget_sec=region_budget_sec, candidate_budget_sec=candidate_budget_sec)
    sides = dict(component_side_by_id)
    if (not sides or not set(sides.values()) <= set(SIDES)
            or len(set(sides.values())) != len(sides)
            or not set(sides.values()) <= set(tool_entity_by_side)):
        raise ValueError("Hover components must map to distinct arms with reviewed tool entities.")
    if capture_registration.skill.definition.skill_version != "2":
        raise ValueError("Hover queries consume capture_surface_anchor@2 Artifacts.")
    roles = tuple(roles)
    if not roles or len(set(roles)) != len(roles) or not set(roles) <= set(back_projection.profile.roles):
        raise ValueError("Hover scene roles must be distinct aligned RGB-D roles.")
    if (not isinstance(planner_profile_hash, str) or len(planner_profile_hash) != 64
            or any(c not in "0123456789abcdef" for c in planner_profile_hash)):
        raise ValueError("Hover queries must bind the safe-motion planner profile hash.")
    frame = back_projection.profile.reference_frame_id
    profile_hash = canonical_content_hash({
        "anchor_producer_route": capture_registration.entry.route.semantic_content_hash,
        "back_projection_profile": back_projection.profile.semantic_content_hash,
        "roles": list(roles), "components": dict(sorted(sides.items())),
        "tool_entities": _identity({side: tool_entity_by_side[side] for side in sides.values()}),
        "planner_profile": planner_profile_hash,
        "region_budget_sec": region_budget_sec,
        "candidate_budget_sec": candidate_budget_sec,
        "evaluation_frame": "public_reference_frame",
    }, domain="agenticrobot.tool_entity_hover.profile.v1")
    definition = hover_definition()
    schema = hover_schema(definition_hash=definition.semantic_content_hash, profile_hash=profile_hash,
        component_ids=tuple(sorted(sides)), reference_frame_id=frame,
        anchor_schema=capture_registration.skill.schema_entry.execution_result_schema)
    slot = SkillArtifactSlotSpec(slot_id="surface_anchor", direction="input", min_count=1, max_count=1,
        allowed_media_types=("application/json",), allowed_schema_ids=(SURFACE_ANCHOR_SCHEMA_ID_V2,),
        allowed_encodings=("utf-8",), allowed_compressions=("none",),
        max_artifact_bytes=MAX_ANCHOR_BYTES, max_total_bytes=MAX_ANCHOR_BYTES)
    io_contract = SkillIOContract(skill_id=HOVER_SKILL_ID, skill_version="1",
        skill_definition_hash=definition.semantic_content_hash, parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash, input_artifact_slots=(slot,))
    route = _clone_route(back_projection.skill.routes[0], definition=definition, schema=schema,
        io_contract=io_contract, suffix=".tool_hover_paths",
        max_operation_duration_ms=int(math.ceil((region_budget_sec + TOOL_HOVER_SCENE_RESERVE_SEC) * 1000)))
    binding = _clone_binding(back_projection.binding, prefix="tool-hover-paths", route=route)
    components = tuple(sorted(sides))

    def validate_parameters(value):
        validate_hover_parameters(value, component_ids=components, reference_frame_id=frame)

    def validate_result(value):
        if not isinstance(value, Mapping) or set(value) != {"surface_anchor", "anchor_artifact_id", "hover_query"}:
            raise ValueError("Hover result must retain its anchor evidence and query.")
        capture_registration.payload_validator.validate_execution_result(value["surface_anchor"])
        if not isinstance(value["anchor_artifact_id"], str) or not value["anchor_artifact_id"]:
            raise ValueError("Hover result must identify its exact input Artifact.")
        validate_hover_query(value["hover_query"], component_ids=components, reference_frame_id=frame)
        if value["surface_anchor"]["measurement"]["reference_frame_id"] != frame:
            raise ValueError("Hover anchor uses another reference frame.")
        query = value["hover_query"]
        if (query["region_budget_sec"] != region_budget_sec
                or query["candidate_budget_sec"] != candidate_budget_sec):
            raise ValueError("Hover query used another search budget than its deployment.")

    def validate_result_for_parameters(parameters, result):
        validate_parameters(parameters)
        validate_result(result)
        validate_hover_result_for_parameters(parameters, result)

    validator = AgentSkillPayloadValidatorSet(validator_id="tool-hover-paths:" + profile_hash,
        skill_id=HOVER_SKILL_ID, skill_version="1", route_id=route.route_id, route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash, validate_parameters=validate_parameters,
        validate_execution_result=validate_result,
        validate_execution_result_for_parameters=validate_result_for_parameters,
        validate_task_evaluation=validate_no_task_evaluation_v1)
    entry = SkillRouteManifestEntry(route=route, deployment_binding=binding, evaluation_plan_hash=None,
        availability_hash=canonical_content_hash({"route": route.semantic_content_hash, "profile": profile_hash,
            "status": "configured_requires_surface_anchor_live_rgbd_and_safe_motion_planner"},
            domain="agenticrobot.tool_entity_hover.availability.v1"))
    return ToolEntityRegistration(SkillRegistrationTemplate(definition, schema, io_contract, (route,)), entry,
        validator, profile_hash, roles, sides, frame, region_budget_sec, candidate_budget_sec)


def _transform_points(transform, points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points @ transform[:3, :3].T + transform[:3, 3]


def _transform_pose(transform, pose):
    return pose_xyz_quat_xyzw(transform @ pose_matrix(pose))


def _qpos_payload(qpos_by_side):
    return {side: [float(x) for x in np.asarray(qpos_by_side[side], dtype=float)] for side in sorted(qpos_by_side)}


def _capture_payload(capture):
    return {"role": capture.role, "source_id": capture.source_id, "acquired_at": float(capture.acquired_at),
            "station_revision": int(capture.station_revision),
            "rgb_content_hash": getattr(capture, "rgb_content_hash", None),
            "depth_content_hash": capture.depth_content_hash,
            "capture_qpos_by_side": _qpos_payload(capture.capture_qpos_by_side)}


def _rigid(transform):
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    rotation = transform[:3, :3]
    if (not np.isfinite(transform).all() or not np.allclose(transform[3], (0., 0., 0., 1.), atol=1e-9)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
            or not math.isclose(float(np.linalg.det(rotation)), 1., abs_tol=1e-8)):
        raise ValueError("public_from_geometry must be a rigid transform.")
    return transform


def run_tool_object_relations(*, parameters, roles, reference_frame_id, scene_port, segmentation_provider,
                              tool_entity_by_side, maximum_depth_m, expected_station_revision, cancel_requested):
    """One relation request over typed ports; the Gateway executor and offline replays share it."""
    entities = dict(tool_entity_by_side)
    revision = expected_station_revision
    selected = tuple(roles) if parameters["roles"] is None else tuple(r for r in roles if r in parameters["roles"])
    captures = tuple(scene_port.capture_views(roles=selected, expected_station_revision=revision,
                                              cancel_requested=cancel_requested))
    if tuple(capture.role for capture in captures) != selected:
        raise RuntimeError("Relation capture returned other camera roles.")
    surfaces, segments, unmeasured = [], [], []
    for capture in captures:
        if capture.station_revision != revision or capture.capture_qpos_by_side is None:
            raise RuntimeError("Relation capture needs this revision's capture-time joint state.")
        tcp = scene_port.tcp_pose_by_side(capture.capture_qpos_by_side)
        rgb = scene_port.segmentation_input(capture)
        for item in parameters["objects"]:
            if cancel_requested():
                raise InterruptedError("Relation query cancelled.")
            prepared = segmentation_provider.try_segment_capture(capture=rgb, parameters={
                "role": capture.role, "language_prompt": item["language_prompt"],
                "target_pixel_uv": None, "min_score": parameters["min_score"]})
            if prepared is None:
                unmeasured.append({"label": item["label"], "role": capture.role, "reason": "no_mask_above_threshold"})
                continue
            evidence = prepared.execution_result
            mask = decode_segmentation_mask_npy(prepared.mask_npy, width=int(evidence["image_width"]),
                height=int(evidence["image_height"]), expected_count=int(evidence["mask_pixel_count"]),
                expected_bounds=tuple(int(v) for v in evidence["mask_bounds_xyxy"]))
            points, surface = object_surface_from_mask(mask, capture, maximum_depth_m=maximum_depth_m)
            segments.append({"label": item["label"], "role": capture.role, "score": float(evidence["score"]),
                "mask_pixel_count": int(evidence["mask_pixel_count"]),
                "mask_bounds_xyxy": [int(v) for v in evidence["mask_bounds_xyxy"]],
                "mask_media_content_hash": evidence["mask_media_content_hash"],
                "valid_pixel_count": int(surface["valid_pixel_count"])})
            if points is None:
                unmeasured.append({"label": item["label"], "role": capture.role, "reason": "no_valid_depth"})
                continue
            surfaces.append(MeasuredObjectSurface(reference=item["label"] + "@" + capture.role,
                label=item["label"], role=capture.role, points_m=points, surface_cluster=surface,
                tcp_pose_by_side={side: tcp[side] for side in entities}))
    shared = describe_tool_object_relations(surfaces=surfaces, tool_entity_by_side=entities)
    result = {"scope": RELATION_SCOPE, "semantics": shared["semantics"], "reference_frame_id": reference_frame_id,
              "station_revision": revision, "objects": shared["objects"], "views": shared["views"],
              "evaluated_sides": shared["evaluated_sides"], "unbound_sides": shared["unbound_sides"],
              "gripper_geometry_by_side": shared["gripper_geometry_by_side"], "unmeasured": unmeasured,
              "segments": segments, "captures": [_capture_payload(capture) for capture in captures],
              "evidence_scope": shared["evidence_scope"], "motion_executed": False}
    return result, captures


def run_tool_hover_query(*, parameters, anchor_xyz_m, side, roles, reference_frame_id, scene_port, planner_port,
                         entity, expected_station_revision, cancel_requested, clock, deadline,
                         region_budget_sec=ENTITY_HOVER_REGION_BUDGET_SEC,
                         candidate_budget_sec=ENTITY_HOVER_CANDIDATE_BUDGET_SEC):
    """One hover request over typed ports, evaluated in the public reference frame."""
    public_from_geometry = _rigid(scene_port.public_from_geometry)
    geometry_from_public = np.linalg.inv(public_from_geometry)
    revision = expected_station_revision
    captures = tuple(scene_port.capture_views(roles=tuple(roles), expected_station_revision=revision,
                                              cancel_requested=cancel_requested))
    if tuple(capture.role for capture in captures) != tuple(roles):
        raise RuntimeError("Hover capture returned other camera roles.")
    if any(capture.station_revision != revision or capture.capture_qpos_by_side is None for capture in captures):
        raise RuntimeError("Hover capture needs this revision's capture-time joint state.")
    qpos = {s: np.asarray(captures[0].capture_qpos_by_side[s], dtype=float).copy() for s in SIDES}
    scene = planner_port.prepare_scene(captures=captures, planning_qpos_by_side=qpos,
                                       cancel_requested=cancel_requested)
    points = _transform_points(public_from_geometry, scene.points_m)
    source_tcp = _transform_pose(public_from_geometry, scene_port.tcp_pose_by_side(qpos)[side])
    anchor_xyz = [float(x) for x in anchor_xyz_m]

    def plan_candidate(pose, *, time_budget_sec):
        outcome = planner_port.plan(scene=scene, qpos_by_side=qpos, side=side,
            target_pose_xyz_quat_xyzw=_transform_pose(geometry_from_public, pose),
            time_budget_sec=time_budget_sec, cancel_requested=cancel_requested)
        if outcome.status != "path_found":
            return outcome
        return replace(outcome, endpoint_pose_xyz_quat_xyzw=tuple(
            _transform_pose(public_from_geometry, outcome.endpoint_pose_xyz_quat_xyzw)))

    summary = search_entity_hover_paths(entity=entity, anchor_xyz_m=anchor_xyz,
        current_tcp_pose_xyz_quat_xyzw=source_tcp, points_m=points,
        support_margin_m=float(scene.quantization_margin_m),
        tip_offset_bounds_xy_m=parameters["tip_offset_bounds_xy_m"],
        clearance_bounds_m=parameters["clearance_bounds_m"], orientation_policy=parameters["orientation_policy"],
        tool_axis_max_tilt_rad=float(parameters["tool_axis_max_tilt_rad"]),
        max_candidates=parameters["max_candidates"], translation_tolerance_m=planner_port.translation_tolerance_m,
        rotation_tolerance_rad=planner_port.rotation_tolerance_rad, plan_candidate=plan_candidate,
        cancel_requested=cancel_requested, clock=clock, deadline=deadline,
        region_budget_sec=region_budget_sec, candidate_budget_sec=candidate_budget_sec)
    return {"component_id": parameters["component_id"], "reference_frame_id": reference_frame_id,
            "station_revision": revision, "anchor_xyz_m": anchor_xyz, "source_tcp_pose_xyz_quat_xyzw": source_tcp,
            **summary,
            "tip_offset_bounds_xy_m": [[float(x) for x in item] for item in parameters["tip_offset_bounds_xy_m"]],
            "clearance_bounds_m": [float(x) for x in parameters["clearance_bounds_m"]],
            "tool_entity": entity.identity(), "observed_voxel_count": len(scene.points_m),
            "start_clearance": planner_port.start_clearance, "planning_frame_id": scene.frame_id,
            "motion_executed": False, "execution_authorized": False}


def build_tool_object_relation_executor(*, registration, scene_port, segmentation_provider, tool_entity_by_side,
                                        maximum_depth_m, current_station_revision, clock, max_retained_operations):
    """Capture, segment and describe; one fresh stationary multi-view cut per request."""
    entities = dict(tool_entity_by_side)

    def read(request, operation):
        registration.payload_validator.validate_parameters(request.parameters)
        revision = request.expected_station_revision
        result, captures = run_tool_object_relations(parameters=request.parameters, roles=registration.roles,
            reference_frame_id=registration.reference_frame_id, scene_port=scene_port,
            segmentation_provider=segmentation_provider, tool_entity_by_side=entities,
            maximum_depth_m=maximum_depth_m, expected_station_revision=revision,
            cancel_requested=lambda: (executor.cancel_requested(operation.operation_id)
                                      or clock() >= operation.deadline_at))
        if current_station_revision() != revision:
            raise RuntimeError("Station revision changed during the relation capture; retry.")
        return PassiveReadResult(result, ("tool-object-relations:" + operation.operation_id,
            *("rgbd-capture:" + capture.role + ":" + capture.depth_content_hash for capture in captures)),
            reason_code="tool_object_relations_described")

    executor = PassiveReadSkillExecutor(port_id="agent-gateway:" + registration.entry.deployment_binding.deployment_binding_id,
        executor_id="passive.tool-object-relations.v1", read=read, current_station_revision=current_station_revision,
        clock=clock, max_retained_operations=max_retained_operations)
    return executor


def build_tool_hover_executor(*, registration, capture_registration, scene_port, planner_port, tool_entity_by_side,
                              store, lookup_operation, current_station_revision, clock, clock_domain_id,
                              max_retained_operations):
    """Resolve the anchor, capture the observed scene once, search hover paths without dispatch."""
    entities = dict(tool_entity_by_side)
    _rigid(scene_port.public_from_geometry)

    def read(request, operation):
        registration.payload_validator.validate_parameters(request.parameters)
        anchor, artifact_id = resolve_surface_anchor(request=request, store=store, lookup_operation=lookup_operation,
            capture_registration=capture_registration, current_station_revision=current_station_revision,
            clock_domain_id=clock_domain_id)
        parameters = request.parameters
        revision = request.expected_station_revision
        side = registration.component_side_by_id[parameters["component_id"]]
        entity = entities.get(side)
        if entity is None:
            raise ValueError("No reviewed tool entity is bound for this arm.")
        query = run_tool_hover_query(parameters=parameters, anchor_xyz_m=anchor["measurement"]["world_xyz_m"],
            side=side, roles=registration.roles, reference_frame_id=registration.reference_frame_id,
            scene_port=scene_port, planner_port=planner_port, entity=entity, expected_station_revision=revision,
            cancel_requested=lambda: (executor.cancel_requested(operation.operation_id)
                                      or clock() >= operation.deadline_at),
            clock=clock, deadline=operation.deadline_at - _RESULT_RESERVE_SEC,
            region_budget_sec=registration.region_budget_sec,
            candidate_budget_sec=registration.candidate_budget_sec)
        if current_station_revision() != revision:
            raise RuntimeError("Station revision changed during the hover query; recapture the anchor.")
        payload = {"surface_anchor": anchor, "anchor_artifact_id": artifact_id, "hover_query": query}
        return PassiveReadResult(payload, ("surface-anchor-source:" + anchor["source_operation_id"],
            "tool-hover-query:" + operation.operation_id), reason_code="tool_hover_query_completed")

    executor = PassiveReadSkillExecutor(port_id=registration.entry.route.route_id,
        executor_id="passive.tool-hover-paths.v1", read=read, current_station_revision=current_station_revision,
        clock=clock, max_retained_operations=max_retained_operations)
    return executor


@dataclass(frozen=True)
class ToolHoverScene:
    """One request's observed scene: geometry-frame cloud plus the planner's opaque input."""

    frame_id: str
    points_m: np.ndarray
    quantization_margin_m: float
    planning_input: object


__all__ = [
    "TOOL_HOVER_MAX_OPERATION_MS",
    "TOOL_HOVER_SCENE_RESERVE_SEC",
    "TOOL_OBJECT_RELATION_MAX_OPERATION_MS",
    "HoverPlanOutcome",
    "ToolEntityRegistration",
    "ToolHoverScene",
    "build_tool_hover_executor",
    "build_tool_hover_registration",
    "build_tool_object_relation_executor",
    "build_tool_object_relation_registration",
    "run_tool_hover_query",
    "run_tool_object_relations",
]
