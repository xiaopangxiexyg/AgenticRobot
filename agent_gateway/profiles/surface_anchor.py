"""Measured anchor Artifact adapters sharing existing projection and operation authority."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace

from agent_gateway.profiles.query_geometry import (
    validate_query_selection,
    validate_relation_geometry_parameters,
)
from agent_gateway.profiles.spatial_goal import (
    SpatialGoalRegistration,
    calculate_spatial_goal,
)
from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.skill_catalog import SkillRouteManifestEntry
from agent_gateway.protocol.skill_io_contracts import (
    SkillArtifactSlotSpec,
    SkillIOContract,
)
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception.object_goal_query import (
    RELATION_FIELDS,
    RELATION_SKILL_ID,
    relation_query_definition,
    validate_relation_query_parameters,
)
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from agent_gateway.skills.perception.surface_anchor import (
    CAPTURE_SKILL_ID,
    GOAL_FIELDS,
    capture_definition,
    capture_schema,
    goal_definition,
    goal_schema,
    query_definition,
    query_schema,
    relation_query_schema,
    validate_anchor,
    validate_anchor_query_parameters,
)
from agent_gateway.skills.perception.surface_anchor_v2 import (
    EXACT_ALGORITHM,
    INTERIOR_ALGORITHM,
    INTERIOR_SELECTION_POLICIES,
    validate_interior_algorithm,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.perception_artifacts import (
    SURFACE_ANCHOR_SCHEMA_ID_V1,
    SURFACE_ANCHOR_SCHEMA_ID_V2,
)
from robot_motion.spatial_goals import generate_relation_goals

MAX_ANCHOR_BYTES = 8192


def _v2_profile_identity(*, back_projection, segmentation, interior_algorithm=INTERIOR_ALGORITHM):
    validate_interior_algorithm(interior_algorithm)
    return canonical_content_hash({
        "projection_profile": back_projection.profile.semantic_content_hash,
        "segmentation_profile": segmentation.profile.semantic_content_hash,
        "selection": interior_algorithm,
    }, domain="agenticrobot.surface_anchor.profile.v2")


def _json_bytes(value):
    def mapping(item):
        if isinstance(item, Mapping):
            return dict(item)
        raise TypeError("Surface anchor JSON contains an unsupported value.")

    return json.dumps(
        value,
        default=mapping,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _registration(
    *, definition, schema, base_entry, direction, parameters, result, result_for,
    namespace="surface_anchor", extra_output_slots=(), max_operation_duration_ms=None,
):
    version = definition.skill_version
    anchor_schema = SURFACE_ANCHOR_SCHEMA_ID_V1 if version == "1" else SURFACE_ANCHOR_SCHEMA_ID_V2
    slot = SkillArtifactSlotSpec(
        slot_id="surface_anchor",
        direction=direction,
        min_count=1,
        max_count=1,
        allowed_media_types=("application/json",),
        allowed_schema_ids=(anchor_schema,),
        allowed_encodings=("utf-8",),
        allowed_compressions=("none",),
        max_artifact_bytes=MAX_ANCHOR_BYTES,
        max_total_bytes=MAX_ANCHOR_BYTES,
    )
    io = SkillIOContract(
        skill_id=definition.skill_id,
        skill_version=version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        input_artifact_slots=(slot,) if direction == "input" else (),
        output_artifact_slots=tuple(sorted((slot, *extra_output_slots), key=lambda item: item.slot_id))
            if direction == "output" else (),
    )
    route = replace(
        base_entry.route,
        skill_id=definition.skill_id,
        skill_version=version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=base_entry.route.route_id + "." + namespace + (".v2" if version == "2" else ""),
        route_version=version,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        skill_io_contract_hash=io.semantic_content_hash,
        max_operation_duration_ms=(base_entry.route.max_operation_duration_ms
            if max_operation_duration_ms is None else max_operation_duration_ms),
    )
    binding = replace(
        base_entry.deployment_binding,
        deployment_binding_id=namespace.replace("_", "-") + ":"
        + base_entry.deployment_binding.deployment_binding_id,
        deployment_binding_hash=canonical_content_hash(
            {
                "base": base_entry.deployment_binding.deployment_binding_hash,
                "route": route.semantic_content_hash,
            },
            domain="agenticrobot.surface_anchor.binding.v1",
        ),
    )
    validator = AgentSkillPayloadValidatorSet(
        validator_id=definition.skill_id + ":" + schema.parameter_schema_hash,
        skill_id=definition.skill_id,
        skill_version=version,
        route_id=route.route_id,
        route_version=version,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        validate_parameters=parameters,
        validate_execution_result=result,
        validate_execution_result_for_parameters=result_for,
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    return SpatialGoalRegistration(
        SkillRegistrationTemplate(definition, schema, io, (route,)),
        SkillRouteManifestEntry(
            route=route,
            deployment_binding=binding,
            evaluation_plan_hash=None,
            availability_hash=schema.parameter_profiles[0].parameter_profile_hash,
        ),
        validator,
    )


def build_surface_anchor_registrations(
    *, back_projection, object_query, spatial_goal=None, enable_relation_queries=False,
    version="1", segmentation=None, selection_policy="prompt_nearest",
):
    if (not isinstance(selection_policy, str) or selection_policy not in INTERIOR_SELECTION_POLICIES
            or version != "2" and selection_policy != "prompt_nearest"):
        raise ValueError("An explicit supported surface selection policy requires v2 anchors.")
    interior_algorithm = INTERIOR_SELECTION_POLICIES[selection_policy]
    if version not in {"1", "2"}:
        raise ValueError("Unsupported surface anchor version.")
    if version == "2" and segmentation is None:
        raise ValueError("Surface anchor v2 requires a bound segmentation provider.")
    if version == "2" and (
            any(getattr(segmentation.binding, key) != getattr(back_projection.binding, key)
                for key in ("runtime_session_id", "control_domain_id", "target_binding_id",
                    "station_manifest_hash", "execution_manifest_hash", "target_kind"))
            or any(segmentation.profile.source_id_by_role.get(role) != source
                for role, source in back_projection.profile.source_id_by_role.items())):
        raise ValueError("Surface anchor v2 requires matching RGB/depth deployment and source identities.")
    if (
        object_query.skill.schema_entry.parameter_schema["properties"][
            "reference_frame_id"
        ]["const"]
        != back_projection.profile.reference_frame_id
    ):
        raise ValueError(
            "Surface anchor and path query must share the public reference frame."
        )
    profile_hash = canonical_content_hash(
        {
            "projection": back_projection.profile.semantic_content_hash,
            "query": object_query.entry.route.semantic_content_hash,
        },
        domain="agenticrobot.surface_anchor.profile.v1",
    )
    if version == "2":
        profile_hash = _v2_profile_identity(back_projection=back_projection, segmentation=segmentation,
            interior_algorithm=interior_algorithm)
    capture_def = capture_definition(version=version)
    schema_args = {
        "definition": capture_def,
        "back_projection_schema": back_projection.skill.schema_entry,
        "profile_hash": profile_hash,
    }
    if version == "2":
        from agent_gateway.skills.perception.surface_anchor_v2 import capture_schema_v2
        schema = capture_schema_v2(**schema_args, segmentation_schema=segmentation.skill.schema_entry,
            interior_algorithm=interior_algorithm)
    else:
        schema = capture_schema(**schema_args)

    def parameters(value):
        if version == "2":
            if not isinstance(value, Mapping) or set(value) != {
                    "role", "selection_mode", "target_pixel_uv", "language_prompt", "min_score"}:
                raise ValueError("Surface capture v2 requires explicit selection parameters.")
            if value["selection_mode"] not in {"exact", "mask_interior"}:
                raise ValueError("Unknown surface selection mode.")
            segmentation.payload_validator.validate_parameters(
                {key: value[key] for key in ("role", "target_pixel_uv", "language_prompt", "min_score")})
            if value["selection_mode"] == "exact" and (
                    value["target_pixel_uv"] is None or value["language_prompt"] is not None):
                raise ValueError("Exact surface selection requires a pixel, not a language prompt.")
            if value["role"] not in back_projection.profile.roles:
                raise ValueError("Surface capture requires aligned depth for its role.")
            return
        if (
            not isinstance(value, Mapping)
            or set(value) != {"role", "target_pixel_uv"}
            or value["target_pixel_uv"] is None
        ):
            raise ValueError("Surface capture accepts only a role and one pixel.")
        back_projection.payload_validator.validate_parameters(
            _projection_parameters(value)
        )

    def result(value):
        if version == "2":
            from agent_gateway.skills.perception.surface_anchor_v2 import (
                validate_anchor_v2,
            )
            validate_anchor_v2(value, interior_algorithm=interior_algorithm)
        else:
            validate_anchor(value)
        back_projection.payload_validator.validate_execution_result(
            value["measurement"]
        )
        if version == "2":
            selected = value["selection"]
            parameters({"role": value["measurement"]["role"], "selection_mode": selected["mode"],
                "target_pixel_uv": selected["requested_pixel_uv"],
                "language_prompt": selected["language_prompt"], "min_score": selected["min_score"]})
            if selected["mode"] == "mask_interior" and (
                    selected["model_profile_id"] != segmentation.profile.model_profile_id
                    or selected["model_profile_hash"] != segmentation.profile.model_profile_hash):
                raise ValueError("Surface anchor changed its bound segmentation model.")
            return
        parameters(
            {
                "role": value["measurement"]["role"],
                "target_pixel_uv": value["target_pixel_uv"],
            }
        )

    def result_for(value, answer):
        parameters(value)
        result(answer)
        if version == "2":
            selected = answer["selection"]
            if (answer["measurement"]["role"] != value["role"]
                    or selected["mode"] != value["selection_mode"]
                    or _json_bytes(selected["requested_pixel_uv"]) != _json_bytes(value["target_pixel_uv"])
                    or selected["language_prompt"] != value["language_prompt"]
                    or selected["min_score"] != value["min_score"]):
                raise ValueError("Surface anchor changed its requested selection.")
            return
        if answer["measurement"]["role"] != value["role"] or list(
            answer["target_pixel_uv"]
        ) != list(value["target_pixel_uv"]):
            raise ValueError("Surface anchor changed its requested pixel.")

    base = SkillRouteManifestEntry(
        route=back_projection.skill.routes[0],
        deployment_binding=back_projection.binding.to_ref(),
        evaluation_plan_hash=None,
        availability_hash=back_projection.availability_hash,
    )
    capture = _registration(
        definition=capture_def,
        schema=schema,
        base_entry=base,
        direction="output",
        parameters=parameters,
        result=result,
        result_for=result_for,
        extra_output_slots=(() if version == "1" else tuple(
            replace(slot, min_count=0) for slot in segmentation.skill.io_contract.output_artifact_slots)),
        max_operation_duration_ms=(None if version == "1"
            else segmentation.skill.routes[0].max_operation_duration_ms),
    )
    query_def = query_definition(version=version)
    anchored_schema = query_schema(
        definition=query_def,
        capture=schema,
        query=object_query.skill.schema_entry,
        profile_hash=profile_hash,
    )

    def query_parameters(value):
        validate_anchor_query_parameters(value)
        object_query.payload_validator.validate_parameters(
            {**value, "anchor_xyz_m": [0, 0, 0]}
        )

    def query_result(value):
        if not isinstance(value, Mapping) or set(value) != {
            "surface_anchor",
            "anchor_artifact_id",
            "path_query",
        }:
            raise ValueError("Anchored query result must retain its source evidence.")
        result(value["surface_anchor"])
        if (
            not isinstance(value["anchor_artifact_id"], str)
            or not value["anchor_artifact_id"]
        ):
            raise ValueError("Anchored query must identify its exact input Artifact.")
        object_query.payload_validator.validate_execution_result(value["path_query"])
        if (
            value["path_query"]["reference_frame_id"]
            != value["surface_anchor"]["measurement"]["reference_frame_id"]
        ):
            raise ValueError("Path query changed its anchor reference frame.")
        if list(value["path_query"]["anchor_xyz_m"]) != list(
            value["surface_anchor"]["measurement"]["world_xyz_m"]
        ):
            raise ValueError("Path query changed its measured anchor.")

    def query_result_for(value, answer):
        query_parameters(value)
        query_result(answer)
        object_query.payload_validator.validate_execution_result_for_parameters(
            {
                **value,
                "anchor_xyz_m": answer["surface_anchor"]["measurement"]["world_xyz_m"],
            },
            answer["path_query"],
        )

    query = _registration(
        definition=query_def,
        schema=anchored_schema,
        base_entry=object_query.entry,
        direction="input",
        parameters=query_parameters,
        result=query_result,
        result_for=query_result_for,
    )
    registrations = (capture, query)
    if spatial_goal is not None:
        registrations += (_build_anchored_goal_registration(
            capture=capture, spatial_goal=spatial_goal,
            reference_frame_id=back_projection.profile.reference_frame_id),)
    if enable_relation_queries:
        registrations += (_build_relation_query_registration(capture=capture, object_query=object_query),)
    return registrations


def _build_relation_query_registration(*, capture, object_query):
    definition = relation_query_definition()
    version = capture.skill.definition.skill_version
    if version != "1":
        definition = replace(definition, skill_version=version,
            parameter_semantics_id="artifact_surface_relation_query.v" + version,
            execution_result_semantics_id="anchored_relation_path_evidence.v" + version)
    identity = canonical_content_hash(
        {"capture": capture.entry.route.semantic_content_hash,
         "query": object_query.entry.route.semantic_content_hash},
        domain="agenticrobot.relation_query.profile.v1")
    frame = object_query.skill.schema_entry.parameter_schema["properties"]["reference_frame_id"]["const"]

    def parameters(value):
        validate_relation_query_parameters(value)
        validate_relation_geometry_parameters(value)
        if value["component_id"] != "right_arm" or value["reference_frame_id"] != frame:
            raise ValueError("Relation query component/frame differs from the deployment binding.")

    def result(value):
        if not isinstance(value, Mapping) or set(value) != {
            "surface_anchor", "anchor_artifact_id", "path_query", "relation_constraint_scope"
        } or value["relation_constraint_scope"] != "endpoint_only":
            raise ValueError("Relation query must retain provenance and endpoint-only relation scope.")
        capture.payload_validator.validate_execution_result(value["surface_anchor"])
        if not isinstance(value["anchor_artifact_id"], str) or not value["anchor_artifact_id"]:
            raise ValueError("Relation query must identify its exact input Artifact.")
        object_query.payload_validator.validate_execution_result(value["path_query"])
        measured = value["surface_anchor"]["measurement"]
        if (value["path_query"]["reference_frame_id"] != measured["reference_frame_id"]
                or list(value["path_query"]["anchor_xyz_m"]) != list(measured["world_xyz_m"])):
            raise ValueError("Relation query changed its measured anchor/frame.")

    def result_for(value, answer):
        parameters(value)
        result(answer)
        path = answer["path_query"]
        candidates = generate_relation_goals(
            current_tcp_pose_xyz_quat_xyzw=path["source_tcp_pose_xyz_quat_xyzw"],
            anchor_xyz_m=answer["surface_anchor"]["measurement"]["world_xyz_m"],
            **{key: value[key] for key in RELATION_FIELDS
               if key not in {"component_id", "reference_frame_id", "time_budget_sec"}})
        validate_query_selection({**value, "anchor_xyz_m": answer["surface_anchor"]["measurement"]["world_xyz_m"]},
                                 path, candidates)

    return _registration(definition=definition,
        schema=relation_query_schema(definition=definition, capture=capture.skill.schema_entry,
            query=object_query.skill.schema_entry, profile_hash=identity),
        base_entry=object_query.entry, direction="input", namespace="relation_query",
        parameters=parameters, result=result, result_for=result_for)


def _build_anchored_goal_registration(*, capture, spatial_goal, reference_frame_id):
    schema = spatial_goal.skill.schema_entry
    if (
        schema.parameter_schema["properties"]["reference_frame_id"]["const"]
        != reference_frame_id
    ):
        raise ValueError(
            "Surface anchor and spatial goal must share the public reference frame."
        )
    profile_hash = canonical_content_hash(
        {
            "capture": capture.entry.route.semantic_content_hash,
            "spatial_goal": spatial_goal.entry.route.semantic_content_hash,
        },
        domain="agenticrobot.surface_anchor.spatial_goal_profile.v1",
    )
    definition = goal_definition(version=capture.skill.definition.skill_version)

    def parameters(value):
        if not isinstance(value, Mapping) or set(value) != GOAL_FIELDS:
            raise ValueError("Anchored goal parameters exclude caller-supplied XYZ.")
        spatial_goal.payload_validator.validate_parameters(
            {**value, "anchor_xyz_m": [0, 0, 0]}
        )

    def result(value):
        if not isinstance(value, Mapping) or set(value) != {
            "surface_anchor",
            "anchor_artifact_id",
            "spatial_goal",
        }:
            raise ValueError("Anchored goal result must retain its source evidence.")
        capture.payload_validator.validate_execution_result(value["surface_anchor"])
        if (
            not isinstance(value["anchor_artifact_id"], str)
            or not value["anchor_artifact_id"]
        ):
            raise ValueError("Anchored goal must identify its exact input Artifact.")
        spatial_goal.payload_validator.validate_execution_result(value["spatial_goal"])
        if (
            value["spatial_goal"]["reference_frame_id"]
            != value["surface_anchor"]["measurement"]["reference_frame_id"]
        ):
            raise ValueError("Spatial goal changed its anchor reference frame.")

    def result_for(value, answer):
        parameters(value)
        result(answer)
        spatial_goal.payload_validator.validate_execution_result_for_parameters(
            {
                **value,
                "anchor_xyz_m": answer["surface_anchor"]["measurement"]["world_xyz_m"],
            },
            answer["spatial_goal"],
        )

    return _registration(
        definition=definition,
        schema=goal_schema(
            definition=definition,
            capture=capture.skill.schema_entry,
            spatial_goal=schema,
            profile_hash=profile_hash,
        ),
        base_entry=spatial_goal.entry,
        direction="input",
        parameters=parameters,
        result=result,
        result_for=result_for,
    )


def _projection_parameters(value):
    return {
        **value,
        "pixel_region_xyxy": None,
        "world_z_min_m": None,
        "world_z_max_m": None,
    }


def build_surface_anchor_capture_executor(
    *,
    registration,
    provider,
    store,
    current_station_revision,
    clock,
    clock_domain_id,
    max_retained_operations,
    segmentation_provider=None,
):
    version = registration.skill.definition.skill_version
    if version == "2" and segmentation_provider is None:
        raise ValueError("Surface anchor v2 requires its lifecycle-owned segmentation provider.")
    interior_algorithm = INTERIOR_ALGORITHM
    if version == "2":
        algorithms = registration.skill.schema_entry.execution_result_schema["properties"][
            "selection"]["properties"]["algorithm_id"]["enum"]
        if len(algorithms) != 2 or algorithms[0] != EXACT_ALGORITHM:
            raise ValueError("Surface anchor v2 must publish one exact interior selection algorithm.")
        interior_algorithm = algorithms[1]
        validate_interior_algorithm(interior_algorithm)
    if version == "2" and registration.skill.schema_entry.parameter_profiles[0].parameter_profile_hash != _v2_profile_identity(
            back_projection=provider.registration, segmentation=segmentation_provider.registration,
            interior_algorithm=interior_algorithm):
        raise ValueError("Surface anchor v2 providers differ from the published deployment identity.")
    if version == "2":
        depth_binding = provider.registration.binding
        model_binding = segmentation_provider.registration.binding
        published = registration.entry.deployment_binding
        if (any(getattr(depth_binding, key) != getattr(model_binding, key)
                for key in ("runtime_session_id", "control_domain_id", "target_binding_id",
                    "station_manifest_hash", "execution_manifest_hash", "target_kind"))
                or any(getattr(depth_binding, key) != getattr(published, key)
                    for key in ("target_binding_id", "target_kind", "execution_manifest_hash"))):
            raise ValueError("Surface anchor v2 provider deployment differs from its published binding.")

    def read(request, operation):
        registration.payload_validator.validate_parameters(request.parameters)
        extra = {}
        prepared = None
        if version == "2":
            from agent_gateway.profiles.surface_anchor_selection import (
                capture_selected_surface,
            )
            measurement, capture, extra, prepared = capture_selected_surface(
                provider=provider, segmentation_provider=segmentation_provider,
                parameters=request.parameters, expected_station_revision=request.expected_station_revision,
                interior_algorithm=interior_algorithm)
        else:
            measurement, capture = provider.project_parameters(
                _projection_parameters(request.parameters),
                expected_station_revision=request.expected_station_revision,
            )
        if current_station_revision() != request.expected_station_revision:
            raise RuntimeError("Station revision changed during surface capture.")
        result = {
            "measurement": measurement,
            "target_pixel_uv": list(extra.pop("selected_pixel_uv", request.parameters["target_pixel_uv"])),
            "captured_at": capture.captured_at,
            "clock_domain_id": clock_domain_id,
            "source_operation_id": operation.operation_id,
            "source_request_hash": request.semantic_content_hash,
            "runtime_session_id": request.runtime_session_id,
            **extra,
        }
        registration.payload_validator.validate_execution_result(result)
        registration.payload_validator.validate_execution_result_for_parameters(request.parameters, result)
        data = _json_bytes(result)
        if len(data) > MAX_ANCHOR_BYTES:
            raise ValueError("Surface anchor exceeds its bounded Artifact contract.")
        artifact = artifact_ref_for_bytes(
            data,
            media_type="application/json",
            schema_id=SURFACE_ANCHOR_SCHEMA_ID_V1 if version == "1" else SURFACE_ANCHOR_SCHEMA_ID_V2,
            encoding="utf-8",
            compression="none",
        )
        commit_artifact_bytes(
            store=store,
            data=data,
            artifact=artifact,
            producer_id=registration.skill.definition.skill_id,
            runtime_session_id=request.runtime_session_id,
            idempotency_key=operation.operation_id,
            now=clock(),
        )
        receipt = "surface-anchor-production:" + canonical_content_hash(
            {
                "operation_id": operation.operation_id,
                "request_hash": request.semantic_content_hash,
                "result": result,
                "artifact": artifact.to_wire(),
            },
            domain="agenticrobot.surface_anchor.production.v1",
        )
        outputs = [AgentArtifactBinding(slot_id="surface_anchor", ordinal=0, artifact=artifact,
            production_receipt_ref=receipt)]
        if prepared is not None:
            for slot in registration.skill.io_contract.output_artifact_slots:
                if slot.slot_id == "surface_anchor":
                    continue
                data = prepared.mask_npy if slot.slot_id == "segmentation_mask" else prepared.overlay_jpeg
                output = artifact_ref_for_bytes(data, media_type=slot.allowed_media_types[0],
                    schema_id=slot.allowed_schema_ids[0], encoding=slot.allowed_encodings[0],
                    compression=slot.allowed_compressions[0])
                commit_artifact_bytes(store=store, data=data, artifact=output,
                    producer_id=registration.skill.definition.skill_id,
                    runtime_session_id=request.runtime_session_id,
                    idempotency_key=operation.operation_id + ":" + slot.slot_id, now=clock())
                outputs.append(AgentArtifactBinding(slot_id=slot.slot_id, ordinal=0,
                    artifact=output, production_receipt_ref=receipt))
        if current_station_revision() != request.expected_station_revision:
            raise RuntimeError("Station revision changed during surface Artifact commit.")
        return PassiveReadResult(
            result,
            (receipt,),
            tuple(sorted(outputs, key=lambda item: (item.slot_id, item.ordinal))),
            reason_code="surface_anchor_captured",
        )

    return PassiveReadSkillExecutor(
        port_id=registration.entry.route.route_id,
        executor_id="passive.surface-anchor.v" + version,
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )


def resolve_surface_anchor(
    *,
    request,
    store,
    lookup_operation,
    capture_registration,
    current_station_revision,
    clock_domain_id,
    accepted_capture_registrations=(),
):
    bindings = request.input_artifact_bindings
    if (
        len(bindings) != 1
        or bindings[0].slot_id != "surface_anchor"
        or bindings[0].ordinal != 0
    ):
        raise ValueError("Exactly one surface_anchor Artifact is required.")
    artifact = bindings[0].artifact
    if (
        artifact.schema_id != (SURFACE_ANCHOR_SCHEMA_ID_V1
            if capture_registration.skill.definition.skill_version == "1" else SURFACE_ANCHOR_SCHEMA_ID_V2)
        or artifact.media_type != "application/json"
        or artifact.encoding != "utf-8"
        or artifact.compression != "none"
        or artifact.byte_length > MAX_ANCHOR_BYTES
    ):
        raise ValueError("Surface anchor Artifact descriptor is incompatible.")
    # Read complete authenticated CAS bytes through the bounded store boundary.
    data = store.read_bytes(artifact, maximum=MAX_ANCHOR_BYTES)
    payload = json.loads(data)
    capture_registration.payload_validator.validate_execution_result(payload)
    retained = lookup_operation(payload["source_operation_id"])
    if retained is None:
        raise ValueError("Surface anchor producer is unavailable; recapture the pixel.")
    source_request, source = retained
    if source is None or source_request is None:
        raise ValueError("Surface anchor producer is unavailable; recapture the pixel.")
    if accepted_capture_registrations:
        candidates = (capture_registration, *accepted_capture_registrations)
        matching = [item for item in candidates
            if item.skill.definition.skill_version == source_request.skill_version
            and item.entry.route.semantic_content_hash == source_request.route_hash]
        if len(matching) != 1:
            raise ValueError("Surface anchor producer is outside the registered source set.")
        capture_registration = matching[0]
        capture_registration.payload_validator.validate_execution_result(payload)
    if (
        source.status != "completed"
        or source.execution_outcome != "completed"
        or source_request.skill_id != CAPTURE_SKILL_ID
        or source_request.skill_version != capture_registration.skill.definition.skill_version
        or source_request.route_hash
        != capture_registration.entry.route.semantic_content_hash
        or getattr(source_request, "deployment_binding_hash", None)
        != capture_registration.entry.deployment_binding.deployment_binding_hash
        or source_request.client_id != request.client_id
        or source_request.runtime_session_id != request.runtime_session_id
        or source_request.skill_route_manifest_hash != request.skill_route_manifest_hash
        or source.runtime_session_id != request.runtime_session_id
        or payload["runtime_session_id"] != request.runtime_session_id
        or source.request_hash != payload["source_request_hash"]
        or source.request_hash != source_request.semantic_content_hash
        or payload["clock_domain_id"] != clock_domain_id
        or source.execution_result is None
        or _json_bytes(source.execution_result) != data
        or not any(
            binding.slot_id == "surface_anchor"
            and binding.ordinal == 0
            and binding.artifact == artifact
            for binding in source.output_artifact_bindings
        )
    ):
        raise ValueError(
            "Surface anchor is not an authenticated output of this client/session."
        )
    if (
        payload["measurement"]["station_revision"] != request.expected_station_revision
        or current_station_revision() != request.expected_station_revision
    ):
        raise ValueError(
            "Surface anchor station revision changed; recapture the pixel."
        )
    return payload, artifact.artifact_id


def require_matching_anchor_schema(registration, capture_registration):
    """Reject a different declared anchor policy before passive work starts.

    This checks producer/consumer schema pairing, not Artifact authentication;
    exact completed producer requests remain verified by resolve_surface_anchor.
    """
    properties = registration.skill.schema_entry.execution_result_schema.get("properties", {})
    if (capture_registration.skill.definition.skill_id != CAPTURE_SKILL_ID
            or properties.get("surface_anchor") != capture_registration.skill.schema_entry.execution_result_schema):
        raise ValueError("Anchored consumer and capture require the same exact anchor result schema.")


def _anchor_source_profile_hash(registration, capture_registrations):
    return canonical_content_hash({
        "skill_id": registration.skill.definition.skill_id,
        "skill_version": registration.skill.definition.skill_version,
        "parameter_schema_hash": registration.skill.schema_entry.parameter_schema_hash,
        "execution_result_schema_hash": registration.skill.schema_entry.execution_result_schema_hash,
        "anchor_producer_route_hashes": sorted(
            item.entry.route.semantic_content_hash for item in capture_registrations),
    }, domain="agenticrobot.surface_anchor.consumer_sources.v1")


def build_multi_source_anchor_consumer(*, registration, capture_registrations):
    """Bind a compatible v2 consumer to an explicit, hashed producer set.

    This opt-in changes its deployment profile, not its v2 parameter/result
    semantics. Existing single-producer registrations remain unchanged.
    """
    captures = tuple(capture_registrations)
    if (registration.skill.definition.skill_version != "2"
            or registration.skill.definition.skill_id == CAPTURE_SKILL_ID
            or len(captures) != 2
            or {item.skill.definition.skill_version for item in captures} != {"2", "3"}):
        raise ValueError("The reviewed source set requires compatible anchor v2 and v3 producers.")
    for item in captures:
        require_matching_anchor_schema(registration, item)
        for name in ("target_binding_id", "target_kind", "execution_manifest_hash"):
            if getattr(item.entry.deployment_binding, name) != getattr(registration.entry.deployment_binding, name):
                raise ValueError("Anchor producer set differs from its consumer deployment.")
    schema = registration.skill.schema_entry
    schema = replace(schema, parameter_profiles=tuple(replace(item,
        parameter_profile_id=item.parameter_profile_id + ".producer-set",
        parameter_profile_hash=_anchor_source_profile_hash(registration, captures))
        for item in schema.parameter_profiles))
    validator = registration.payload_validator
    return _registration(definition=registration.skill.definition, schema=schema,
        base_entry=registration.entry, direction="input", namespace="authenticated_anchor_sources",
        parameters=validator.validate_parameters, result=validator.validate_execution_result,
        result_for=validator.validate_execution_result_for_parameters)


def _require_anchor_producer_set(registration, capture_registration, additional):
    require_matching_anchor_schema(registration, capture_registration)
    profiles = registration.skill.schema_entry.parameter_profiles
    if not additional:
        if any(item.parameter_profile_id.endswith(".producer-set") for item in profiles):
            raise ValueError("The published anchor producer set cannot be omitted from its executor.")
        return
    captures = (capture_registration, *additional)
    for item in additional:
        require_matching_anchor_schema(registration, item)
    if (len(captures) != 2
            or {item.skill.definition.skill_version for item in captures} != {"2", "3"}
            or len(profiles) != 1
            or not profiles[0].parameter_profile_id.endswith(".producer-set")
            or profiles[0].parameter_profile_hash != _anchor_source_profile_hash(registration, captures)):
        raise ValueError("Anchor consumers require their exact published producer set.")


def build_anchored_query_executor(
    *,
    registration,
    capture_registration,
    query_port,
    store,
    lookup_operation,
    current_station_revision,
    clock,
    clock_domain_id,
    max_retained_operations,
    accepted_capture_registrations=(),
):
    accepted_capture_registrations = tuple(accepted_capture_registrations)
    _require_anchor_producer_set(registration, capture_registration, accepted_capture_registrations)
    relation = registration.skill.definition.skill_id == RELATION_SKILL_ID
    query = query_port.query_relations if relation else query_port.query

    def read(request, operation):
        registration.payload_validator.validate_parameters(request.parameters)
        anchor, artifact_id = resolve_surface_anchor(
            request=request,
            store=store,
            lookup_operation=lookup_operation,
            capture_registration=capture_registration,
            current_station_revision=current_station_revision,
            clock_domain_id=clock_domain_id,
            accepted_capture_registrations=accepted_capture_registrations,
        )
        result = query(
            {
                **request.parameters,
                "anchor_xyz_m": anchor["measurement"]["world_xyz_m"],
            },
            cancel_requested=lambda: (
                executor.cancel_requested(operation.operation_id)
                or clock() >= operation.deadline_at
            ),
        )
        if current_station_revision() != request.expected_station_revision:
            raise RuntimeError(
                "Station revision changed during anchored path query; recapture the pixel."
            )
        payload = {"surface_anchor": anchor, "anchor_artifact_id": artifact_id, "path_query": result}
        if relation:
            payload["relation_constraint_scope"] = "endpoint_only"
        outputs = ()
        if relation and registration.skill.definition.skill_version == "3":
            from agent_gateway.profiles.pose_candidate_set import publish_candidate_set

            outputs = (publish_candidate_set(request=request, operation=operation, result=payload,
                registration=registration, store=store, clock=clock, clock_domain_id=clock_domain_id),)
            if current_station_revision() != request.expected_station_revision:
                raise RuntimeError("Station revision changed during candidate Artifact commit.")
        return PassiveReadResult(
            payload,
            (
                "surface-anchor-source:" + anchor["source_operation_id"],
                "object-goal-query:" + operation.operation_id,
            ),
            output_artifact_bindings=outputs,
            reason_code="anchored_relation_query_completed" if relation else "anchored_object_goal_query_completed",
        )

    executor = PassiveReadSkillExecutor(
        port_id=registration.entry.route.route_id,
        executor_id=("passive.anchored-relation-query.v" if relation else "passive.anchored-object-query.v")
            + registration.skill.definition.skill_version,
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )
    return executor


def build_anchored_goal_executor(
    *,
    registration,
    capture_registration,
    store,
    lookup_operation,
    current_station_revision,
    clock,
    clock_domain_id,
    max_retained_operations,
    accepted_capture_registrations=(),
):
    """Reuse authenticated anchor resolution and the canonical pure geometry owner."""
    accepted_capture_registrations = tuple(accepted_capture_registrations)
    _require_anchor_producer_set(registration, capture_registration, accepted_capture_registrations)

    def read(request, operation):
        registration.payload_validator.validate_parameters(request.parameters)
        anchor, artifact_id = resolve_surface_anchor(
            request=request,
            store=store,
            lookup_operation=lookup_operation,
            capture_registration=capture_registration,
            current_station_revision=current_station_revision,
            clock_domain_id=clock_domain_id,
            accepted_capture_registrations=accepted_capture_registrations,
        )
        goal = calculate_spatial_goal(
            {
                **request.parameters,
                "anchor_xyz_m": anchor["measurement"]["world_xyz_m"],
            }
        )
        if current_station_revision() != request.expected_station_revision:
            raise RuntimeError(
                "Station revision changed during anchored goal calculation; recapture the pixel."
            )
        return PassiveReadResult(
            {
                "surface_anchor": anchor,
                "anchor_artifact_id": artifact_id,
                "spatial_goal": goal,
            },
            (
                "surface-anchor-source:" + anchor["source_operation_id"],
                "spatial-goal-calculation:" + operation.operation_id,
            ),
            reason_code="anchored_spatial_goal_computed",
        )

    return PassiveReadSkillExecutor(
        port_id=registration.entry.route.route_id,
        executor_id="passive.anchored-spatial-goal.v" + registration.skill.definition.skill_version,
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )
