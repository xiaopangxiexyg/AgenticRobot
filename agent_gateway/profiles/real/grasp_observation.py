"""Frozen, stationary RGB-D observation Skill and session-bound Artifact reader."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.skill_catalog import SkillDeploymentBindingRef
from agent_gateway.protocol.skill_io_contracts import SkillArtifactSlotSpec, SkillIOContract
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.artifact_producer import artifact_ref_for_bytes, commit_artifact_bytes
from agent_gateway.runtime.passive_read_executor import PassiveReadResult, PassiveReadSkillExecutor
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception.grasp_observation import (
    grasp_observation_definition, grasp_observation_schema, validate_observation_parameters,
    validate_observation_result, validate_observation_result_for_parameters,
    frozen_point_grasp_contract,
)
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.grasp_multiview_snapshot import (
    GRASP_MULTIVIEW_SCHEMA_ID, GRASP_MULTIVIEW_ENCODING, GRASP_MULTIVIEW_MAX_BYTES,
    encode_grasp_multiview_arrays, decode_grasp_multiview_snapshot,
    decode_grasp_multiview_snapshot_arrays,
)
from dataset_contracts.grasp_rgbd_snapshot import (
    GRASP_RGBD_SNAPSHOT_SCHEMA_ID,
)
from dataset_contracts.perception_artifacts import (
    RGB_FRAME_SCHEMA_ID_V1, RGB_FRAME_ENCODING_V1, RGB_FRAME_COMPRESSION_V1,
)

OBSERVATION_SLOT = "grasp_observation"


def upgrade_frozen_point_grasp_registration(base):
    definition, schema = frozen_point_grasp_contract(base.skill.definition, base.skill.schema_entry,
                                                    profile_hash=base.profile.semantic_content_hash)
    outputs = tuple(replace(slot, allowed_media_types=("application/zip",),
                            allowed_schema_ids=(GRASP_MULTIVIEW_SCHEMA_ID,), allowed_encodings=(GRASP_MULTIVIEW_ENCODING,))
                    if slot.slot_id == "point_grasp_rgbd_snapshot" else slot
                    for slot in base.skill.io_contract.output_artifact_slots)
    io_contract = replace(base.skill.io_contract, skill_version="4", skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
        output_artifact_slots=outputs)
    route = replace(base.skill.routes[0], skill_version="4", route_version="4", skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles, skill_io_contract_hash=io_contract.semantic_content_hash)
    binding = replace(base.binding, deployment_binding_version="4", route_hash=route.semantic_content_hash,
                      executor_id="real.passive.frozen_point_grasp.v4")
    extra = {"observation_operation_id", "view_policy"}

    def validate_parameters(payload):
        identifier = payload.get("observation_operation_id")
        if (not isinstance(identifier, str) or not 0 < len(identifier) <= 256 or identifier.strip() != identifier
                or payload.get("view_policy") not in {"single", "auto"}):
            raise ValueError("A retained grasp observation and view policy are required")
        base.payload_validator.validate_parameters({k: v for k, v in payload.items() if k not in extra})

    def validate_result(payload):
        evidence = payload.get("view_fusion")
        if not isinstance(evidence, dict) and not hasattr(evidence, "items"):
            raise ValueError("Frozen grasp result lacks view fusion evidence")
        if (set(evidence) != {"observation_operation_id", "view_policy", "diagnostics_json"}
                or evidence["view_policy"] not in {"single", "auto"}
                or not isinstance(evidence["observation_operation_id"], str) or not evidence["observation_operation_id"]
                or not isinstance(evidence["diagnostics_json"], str) or len(evidence["diagnostics_json"]) > 16384):
            raise ValueError("Invalid view fusion evidence")
        diagnostics = json.loads(evidence["diagnostics_json"])
        if not isinstance(diagnostics, list) or not 1 <= len(diagnostics) <= 3:
            raise ValueError("Invalid view fusion diagnostics")
        base.payload_validator.validate_execution_result({k: v for k, v in payload.items() if k != "view_fusion"})

    def validate_for_parameters(parameters, result):
        if any(result["view_fusion"][key] != parameters[key] for key in extra):
            raise ValueError("Grasp observation lineage differs from the request")
        base.payload_validator.validate_execution_result_for_parameters(
            {k: v for k, v in parameters.items() if k not in extra},
            {k: v for k, v in result.items() if k != "view_fusion"})

    validator = replace(base.payload_validator, skill_version="4", route_version="4",
        parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
        validate_parameters=validate_parameters, validate_execution_result=validate_result,
        validate_execution_result_for_parameters=validate_for_parameters)
    registration = replace(base, binding=binding, payload_validator=validator,
        skill=SkillRegistrationTemplate(definition=definition, schema_entry=schema, io_contract=io_contract, routes=(route,)),
        availability_hash=canonical_content_hash({"route": route.semantic_content_hash, "binding": binding.semantic_content_hash},
                                                domain="dexteleop.frozen_grasp.availability.v4"))
    return replace(registration, observation_registration=build_grasp_observation_registration(registration))


@dataclass(frozen=True)
class GraspObservationRegistration:
    skill: SkillRegistrationTemplate
    binding: SkillDeploymentBindingRef
    payload_validator: AgentSkillPayloadValidatorSet
    availability_hash: str
    executor_id: str = "real.passive.grasp_observation.v1"


def build_grasp_observation_registration(point_registration) -> GraspObservationRegistration:
    profile = point_registration.profile
    roles = tuple(profile.source_by_role)
    profile_hash = canonical_content_hash({
        "source_binding_hash": profile.source_binding_hash,
        "frame_binding_hash": profile.frame_binding_hash,
        "roles": roles, "stationary_only": True,
    }, domain="dexteleop.grasp_observation.profile.v1")
    definition = grasp_observation_definition()
    schema = grasp_observation_schema(definition_hash=definition.semantic_content_hash,
                                      roles=roles, profile_hash=profile_hash)
    io_contract = SkillIOContract(
        skill_id=definition.skill_id, skill_version="1", skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash, input_artifact_slots=(),
        output_artifact_slots=(
            SkillArtifactSlotSpec(slot_id=OBSERVATION_SLOT, direction="output", min_count=1, max_count=1,
                allowed_media_types=("application/zip",), allowed_schema_ids=(GRASP_MULTIVIEW_SCHEMA_ID,),
                allowed_encodings=(GRASP_MULTIVIEW_ENCODING,), allowed_compressions=("zip",),
                max_artifact_bytes=GRASP_MULTIVIEW_MAX_BYTES, max_total_bytes=GRASP_MULTIVIEW_MAX_BYTES),
            SkillArtifactSlotSpec(slot_id="rgb_frames", direction="output", min_count=1, max_count=3,
                allowed_media_types=("image/jpeg",), allowed_schema_ids=(RGB_FRAME_SCHEMA_ID_V1,),
                allowed_encodings=(RGB_FRAME_ENCODING_V1,), allowed_compressions=(RGB_FRAME_COMPRESSION_V1,),
                max_artifact_bytes=4*1024*1024, max_total_bytes=12*1024*1024),
        ),
    )
    route = replace(point_registration.skill.routes[0],
        skill_id=definition.skill_id, skill_version="1", skill_definition_hash=definition.semantic_content_hash,
        route_id="real.realsense.perception.capture_grasp_observation", route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash, allowed_parameter_profiles=schema.parameter_profiles,
        skill_io_contract_hash=io_contract.semantic_content_hash, max_operation_duration_ms=10_000)
    skill = SkillRegistrationTemplate(definition=definition, schema_entry=schema, io_contract=io_contract, routes=(route,))
    source_binding = point_registration.binding.to_ref()
    binding = replace(source_binding,
        deployment_binding_id="grasp-observation:"+point_registration.binding.runtime_session_id,
        deployment_binding_hash=canonical_content_hash({"profile": profile_hash, "route": route.semantic_content_hash,
            "session": point_registration.binding.runtime_session_id}, domain="dexteleop.grasp_observation.binding.v1"))
    validator = AgentSkillPayloadValidatorSet(
        validator_id="real.grasp-observation:"+profile_hash,
        skill_id=definition.skill_id, skill_version="1", route_id=route.route_id, route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        validate_parameters=lambda value: validate_observation_parameters(value, roles=roles),
        validate_execution_result=lambda value: validate_observation_result(value, roles=roles),
        validate_execution_result_for_parameters=validate_observation_result_for_parameters,
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    return GraspObservationRegistration(skill, binding, validator, profile_hash)


def build_grasp_observation_executor(*, registration, perception_provider, frame_binding,
                                     artifact_store, runtime_session_id, current_station_revision,
                                     max_retained_operations, clock, non_motion_read_window,
                                     require_stationary=True, observation_slot=OBSERVATION_SLOT):
    capture_name = "grasp-observation" if observation_slot == OBSERVATION_SLOT else "rgbd-observation"
    def commit(data, *, slot, ordinal, schema, encoding, media_type, compression, operation):
        artifact = artifact_ref_for_bytes(data, media_type=media_type, schema_id=schema,
                                         encoding=encoding, compression=compression)
        artifact = commit_artifact_bytes(store=artifact_store, data=data, artifact=artifact,
            producer_id="agent_gateway.grasp_observation", runtime_session_id=runtime_session_id,
            idempotency_key=f"grasp-observation:{operation.operation_id}:{slot}:{ordinal}", now=float(clock()))
        return AgentArtifactBinding(slot_id=slot, ordinal=ordinal, artifact=artifact,
                                    production_receipt_ref="grasp-observation:"+operation.operation_id)

    def read(request, operation):
        views, previews, excluded = [], [], []
        captures = []
        # This window is owned by the existing guard. No HOLD, reconnect, or
        # motion call is made by this read-only executor.
        with non_motion_read_window():
            for role in request.parameters["selected_roles"]:
                try:
                    capture = perception_provider.capture_aligned_rgbd(role=role,
                        expected_station_revision=request.expected_station_revision,
                        require_stationary=require_stationary)
                except RuntimeError as error:
                    reason = str(error).split(":", 1)[0]
                    if reason not in {"real_perception_frame_unavailable", "real_perception_frame_stale",
                                      "real_perception_source_not_running", "real_rgb_camera_pose_unavailable",
                                      "real_rgb_camera_pose_stale", "real_rgb_camera_frame_pose_skew_exceeded",
                                      "grasp_observation_frame_pose_unavailable"} and not (
                        not require_stationary and reason in {
                            "grasp_observation_stationary_geometry_unavailable",
                            "grasp_observation_stationary_history_unavailable",
                            "grasp_observation_robot_not_stationary",
                        }
                    ):
                        raise
                    excluded.append({"role": role, "reason": reason})
                    continue
                captures.append(capture)
            # Acquire all frame cuts before compression so JPEG/NPZ work does
            # not itself increase inter-camera acquisition skew.
            for capture in captures:
                role = capture.role
                transform, _ = frame_binding.frame_graph.transform_pose_matrix(
                    np.asarray(capture.source_from_camera).reshape(4, 4),
                    source_reference_frame_id=frame_binding.source_frame_id,
                    target_reference_frame_id=frame_binding.reference_frame_id)
                metadata = {
                    "schema_id": GRASP_RGBD_SNAPSHOT_SCHEMA_ID, "role": role,
                    "source_id": capture.source_id, "acquired_at": capture.acquired_at,
                    "station_revision": capture.station_revision, "source_generation": capture.source_generation,
                    "frame_index": capture.frame_index, "capture_group_id": capture.capture_group_id,
                    "source_binding_hash": capture.source_binding_hash,
                    "frame_binding_hash": frame_binding.semantic_content_hash,
                    "camera_intrinsics": dict(capture.intrinsics),
                    "T_source_from_camera": list(capture.source_from_camera),
                    "T_reference_from_camera": transform.tolist(),
                    "reference_frame_id": frame_binding.reference_frame_id,
                    "rgb_content_hash": capture.rgb_content_hash, "depth_content_hash": capture.depth_content_hash,
                    "target_mask_available": False,
                    "stationary_only": require_stationary or capture.capture_qpos_by_side is not None,
                    ("measured_qpos" if require_stationary else "camera_pose"): {
                        "measured_at": capture.pose_measured_at, "available_at": capture.pose_available_at,
                        "source_evidence_refs": list(capture.pose_evidence_refs)},
                }
                views.append(dict(rgb=capture.rgb, depth_m=capture.depth_m,
                    target_mask=np.zeros(capture.depth_m.shape, dtype=bool), metadata=metadata,
                    qpos_by_side=capture.capture_qpos_by_side))
                stream = io.BytesIO()
                Image.fromarray(capture.rgb).save(stream, format="JPEG", quality=95)
                previews.append((capture, stream.getvalue()))
        if not views:
            raise RuntimeError("No usable RGB-D view; "+str(excluded))
        if current_station_revision() != request.expected_station_revision:
            raise RuntimeError("Station revision changed during grasp observation")
        observation = encode_grasp_multiview_arrays(views=views, metadata={
            "schema_id": GRASP_MULTIVIEW_SCHEMA_ID, "runtime_session_id": runtime_session_id,
            "station_revision": request.expected_station_revision,
            "frame_binding_hash": frame_binding.semantic_content_hash, "excluded_views": excluded,
            "scope": ("stationary_observation_not_a_validated_plan" if require_stationary
                      else "capture_time_geometry_not_current_scene"),
        })
        outputs = [commit(observation, slot=observation_slot, ordinal=0, schema=GRASP_MULTIVIEW_SCHEMA_ID,
            encoding=GRASP_MULTIVIEW_ENCODING, media_type="application/zip", compression="zip", operation=operation)]
        frames = []
        for ordinal, (capture, data) in enumerate(previews):
            outputs.append(commit(data, slot="rgb_frames", ordinal=ordinal, schema=RGB_FRAME_SCHEMA_ID_V1,
                encoding=RGB_FRAME_ENCODING_V1, media_type="image/jpeg", compression=RGB_FRAME_COMPRESSION_V1, operation=operation))
            frames.append({"role": capture.role, "source_id": capture.source_id, "ordinal": ordinal,
                "width": capture.width, "height": capture.height, "media_type": "image/jpeg",
                "media_content_hash": hashlib.sha256(data).hexdigest(), "acquired_at": capture.acquired_at})
        return PassiveReadResult(execution_result={"station_revision": request.expected_station_revision,
                "frames": frames, "excluded_views": excluded},
            evidence_refs=(capture_name+":"+operation.operation_id,),
            output_artifact_bindings=tuple(sorted(outputs, key=lambda item: (item.slot_id, item.ordinal))),
            reason_code=("passive_grasp_observation_retained" if observation_slot == OBSERVATION_SLOT
                         else "passive_rgbd_observation_retained"))
    return PassiveReadSkillExecutor(port_id="agent-gateway:"+capture_name+":"+runtime_session_id,
        executor_id=registration.executor_id, read=read, current_station_revision=current_station_revision,
        clock=clock, max_retained_operations=max_retained_operations)


def read_retained_grasp_observation(*, executor, operation_id, artifact_store, runtime_session_id,
                                    expected_station_revision, frame_binding_hash,
                                    observation_slot=OBSERVATION_SLOT, decode_arrays=False):
    completion = executor.completion_for(operation_id)
    if completion is None or not completion.accepted:
        raise ValueError("Grasp observation is not retained by this session")
    operation = completion.operation
    if (operation.runtime_session_id != runtime_session_id or operation.status != "completed"
            or operation.execution_outcome != "completed"
            or operation.execution_result["station_revision"] != expected_station_revision):
        raise ValueError("Grasp observation session, revision, or completion differs")
    bindings = [b for b in operation.output_artifact_bindings if b.slot_id == observation_slot]
    if len(bindings) != 1:
        raise ValueError("Grasp observation lacks a unique retained Artifact")
    artifact = bindings[0].artifact
    data = artifact_store.read_bytes(artifact, maximum=GRASP_MULTIVIEW_MAX_BYTES)
    decoder = decode_grasp_multiview_snapshot_arrays if decode_arrays else decode_grasp_multiview_snapshot
    views, metadata, points, _ = decoder(data)
    if (metadata.get("runtime_session_id") != runtime_session_id
            or metadata.get("station_revision") != expected_station_revision
            or metadata.get("frame_binding_hash") != frame_binding_hash or points is not None):
        raise ValueError("Grasp observation provenance differs")
    return views, metadata
