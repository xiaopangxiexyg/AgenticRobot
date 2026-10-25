"""Neutral RGB v3 assembly over existing capture owners, passive executor and CAS.

The RGB-only branch retains ordered legacy captures. The paired branch accepts
only original JPEG/NPY bytes from one correlated cut. No camera, registry or new
lifecycle is created. Assembly must retain at least three producer/consumer
operations.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from agent_gateway.profiles.back_projection import BackProjectionRegistration
from agent_gateway.profiles.spatial_goal import SpatialGoalRegistration
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
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception.definition import capture_rgb_skill_definition_v2
from agent_gateway.skills.perception.rgbd_schemas import (
    capture_rgb_schema_entry_v3,
    capture_rgb_skill_definition_v3,
    validate_capture_rgb_parameters_v3,
    validate_capture_rgb_result_for_parameters_v3,
    validate_capture_rgb_result_v3,
)
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.perception_artifacts import (
    DEPTH_FRAME_COMPRESSION_V1,
    DEPTH_FRAME_ENCODING_V1,
    DEPTH_FRAME_MEDIA_TYPE_V1,
    DEPTH_FRAME_SCHEMA_ID_V1,
    PAIRED_RGBD_SCHEMA_ID_V1,
    RGB_FRAME_COMPRESSION_V1,
    RGB_FRAME_ENCODING_V1,
    RGB_FRAME_MEDIA_TYPE_V1,
    RGB_FRAME_SCHEMA_ID_V1,
)

RGBD_CAPTURE_PROFILE_ID_V3 = "registered-rgb-optional-paired-depth-v3"
RGBD_CAPTURE_EXECUTOR_ID_V3 = "passive.rgbd-capture.v3"


def _frame_identity(profile):
    result = {}
    for name in ("reference_frame_id", "reference_frame_spec_hash", "frame_graph_hash"):
        value = getattr(profile, name, None)
        if (
            not isinstance(value, str)
            or not value
            or value.strip() != value
            or len(value) > 256
        ):
            raise ValueError("RGBD capture requires a bound projection profile " + name)
        result[name] = value
    return result


def _availability(route, back_projection):
    return canonical_content_hash(
        {
            "route": route.semantic_content_hash,
            "projection_route": back_projection.skill.routes[0].semantic_content_hash,
            "projection_profile": back_projection.profile.semantic_content_hash,
            "projection_binding": back_projection.binding.semantic_content_hash,
            "frame_identity": _frame_identity(back_projection.profile),
            "status": "requires_rgb_and_correlated_rgbd_providers_and_artifact_store",
        },
        domain="agenticrobot.rgbd_capture.availability.v3",
    )


def _passive_route(skill):
    if not isinstance(skill, SkillRegistrationTemplate) or len(skill.routes) != 1:
        raise ValueError("RGBD capture requires one exact registered Route.")
    route = skill.routes[0]
    if (
        route.control_effect_class,
        route.privilege_tier,
        route.may_move_robot,
        route.may_open_publication,
        route.requires_operator_confirmation,
        route.requires_deadman,
        route.requires_task_evaluation,
        route.cancellation_policy,
    ) != (
        "read_only",
        "t0_read_only",
        False,
        False,
        False,
        False,
        False,
        "cancel_only",
    ):
        raise ValueError("RGBD capture requires passive read-only source Routes.")
    return route


def build_rgbd_capture_registration(
    *, rgb_registration, back_projection
) -> SpatialGoalRegistration:
    """Derive an explicit v3 surface from exact RGB v2 and projection bindings."""
    if not isinstance(back_projection, BackProjectionRegistration):
        raise TypeError("back_projection must be a BackProjectionRegistration.")
    rgb_skill = rgb_registration.registration
    rgb_route = _passive_route(rgb_skill)
    depth_route = _passive_route(back_projection.skill)
    if rgb_skill.definition != capture_rgb_skill_definition_v2():
        raise ValueError("RGBD capture requires the existing RGB v2 registration.")
    rgb_registration.payload_validator.validate_route(rgb_route)
    back_projection.payload_validator.validate_route(depth_route)
    rgb_binding = rgb_registration.deployment_binding
    depth_binding = back_projection.binding
    for name in (
        "runtime_session_id",
        "control_domain_id",
        "target_binding_id",
        "station_manifest_hash",
        "execution_manifest_hash",
    ):
        if getattr(rgb_binding, name) != getattr(depth_binding, name):
            raise ValueError("RGB and depth deployment identities disagree: " + name)
    if (
        rgb_binding.route_hash != rgb_route.semantic_content_hash
        or rgb_binding.profile_hash != rgb_registration.profile.semantic_content_hash
        or len(rgb_route.allowed_parameter_profiles) != 1
        or rgb_route.allowed_parameter_profiles[0].parameter_profile_hash
        != rgb_registration.profile.semantic_content_hash
        or rgb_route.allowed_parameter_profiles[0].parameter_profile_id
        != rgb_registration.profile.profile_id
        or rgb_route.target_kind != depth_route.target_kind
        or rgb_route.supported_modes != depth_route.supported_modes
    ):
        raise ValueError("RGB and depth Routes or profile bindings disagree.")
    rgb_sources = {
        role: (source, width, height)
        for source, role, width, height in rgb_registration.profile.sources
    }
    depth_dimensions = {}
    for source, role, width, height in back_projection.profile.sources:
        if role in rgb_sources:
            if rgb_sources[role] != (source, width, height):
                raise ValueError(
                    "RGB/depth registered source or native grid disagrees."
                )
            depth_dimensions[role] = (width, height)
    if not depth_dimensions:
        raise ValueError(
            "RGBD capture requires a shared registered aligned-depth role."
        )
    sources = dict(rgb_registration.profile.source_id_by_role)
    roles = tuple(rgb_registration.profile.roles)
    maximum = rgb_registration.profile.max_artifact_bytes
    profile_hash = canonical_content_hash(
        {
            "rgb_route": rgb_route.semantic_content_hash,
            "rgb_profile": rgb_registration.profile.semantic_content_hash,
            "projection_route": depth_route.semantic_content_hash,
            "projection_profile": back_projection.profile.semantic_content_hash,
            "frame_identity": _frame_identity(back_projection.profile),
            "depth_dimensions_by_role": depth_dimensions,
            "total_artifact_byte_limit": maximum,
            "capture_semantics": "optional_single_correlated_cut_original_bytes",
        },
        domain="agenticrobot.rgbd_capture.profile.v3",
    )
    definition = capture_rgb_skill_definition_v3()
    schema = capture_rgb_schema_entry_v3(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=RGBD_CAPTURE_PROFILE_ID_V3,
        parameter_profile_hash=profile_hash,
        allowed_roles=roles,
        max_total_bytes=maximum,
        depth_dimensions_by_role=depth_dimensions,
    )
    # The codec owns the JSON representation identity, not a concrete backend.
    from agent_gateway.profiles.paired_rgbd import PAIRED_RGBD_MAX_BYTES_V1

    formats = (
        (
            "rgb_frames",
            1,
            rgb_registration.profile.max_frames,
            RGB_FRAME_MEDIA_TYPE_V1,
            RGB_FRAME_SCHEMA_ID_V1,
            RGB_FRAME_ENCODING_V1,
            RGB_FRAME_COMPRESSION_V1,
        ),
        (
            "depth_frames",
            0,
            1,
            DEPTH_FRAME_MEDIA_TYPE_V1,
            DEPTH_FRAME_SCHEMA_ID_V1,
            DEPTH_FRAME_ENCODING_V1,
            DEPTH_FRAME_COMPRESSION_V1,
        ),
        (
            "paired_observations",
            0,
            1,
            "application/json",
            PAIRED_RGBD_SCHEMA_ID_V1,
            "utf-8",
            "none",
        ),
    )
    io = SkillIOContract(
        skill_id=definition.skill_id,
        skill_version="3",
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        output_artifact_slots=tuple(
            sorted(
                (
                    SkillArtifactSlotSpec(
                        slot_id=slot,
                        direction="output",
                        min_count=minimum,
                        max_count=count,
                        allowed_media_types=(media,),
                        allowed_schema_ids=(identity,),
                        allowed_encodings=(encoding,),
                        allowed_compressions=(compression,),
                        max_artifact_bytes=(
                            min(maximum, PAIRED_RGBD_MAX_BYTES_V1)
                            if slot == "paired_observations"
                            else maximum
                        ),
                        max_total_bytes=(
                            min(maximum, PAIRED_RGBD_MAX_BYTES_V1)
                            if slot == "paired_observations"
                            else maximum
                        ),
                    )
                    for slot, minimum, count, media, identity, encoding, compression in formats
                ),
                key=lambda item: item.slot_id,
            )
        ),
    )
    route = replace(
        rgb_route,
        skill_version="3",
        route_id=rgb_route.route_id + ".rgbd",
        route_version="3",
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        skill_io_contract_hash=io.semantic_content_hash,
    )
    binding = replace(
        rgb_binding.to_ref(),
        deployment_binding_id="rgbd-capture:" + rgb_binding.deployment_binding_id,
        deployment_binding_hash=canonical_content_hash(
            {
                "rgb": rgb_binding.semantic_content_hash,
                "depth": depth_binding.semantic_content_hash,
                "route": route.semantic_content_hash,
            },
            domain="agenticrobot.rgbd_capture.binding.v3",
        ),
    )
    validator = AgentSkillPayloadValidatorSet(
        validator_id="rgbd-capture:" + profile_hash,
        skill_id=definition.skill_id,
        skill_version="3",
        route_id=route.route_id,
        route_version="3",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        validate_parameters=lambda payload: validate_capture_rgb_parameters_v3(
            payload,
            allowed_roles=roles,
            max_total_bytes=maximum,
            depth_dimensions_by_role=depth_dimensions,
        ),
        validate_execution_result=lambda payload: validate_capture_rgb_result_v3(
            payload,
            source_id_by_role=sources,
            max_total_bytes=maximum,
            depth_dimensions_by_role=depth_dimensions,
        ),
        validate_execution_result_for_parameters=validate_capture_rgb_result_for_parameters_v3,
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    return SpatialGoalRegistration(
        SkillRegistrationTemplate(definition, schema, io, (route,)),
        SkillRouteManifestEntry(
            route=route,
            deployment_binding=binding,
            evaluation_plan_hash=None,
            availability_hash=_availability(route, back_projection),
        ),
        validator,
    )


def build_rgbd_capture_executor(
    *,
    registration,
    rgb_provider,
    back_projection_provider,
    store,
    current_station_revision,
    clock,
    clock_domain_id,
    max_retained_operations,
) -> PassiveReadSkillExecutor:
    """Bind existing owners without publishing cancelled, expired or stale reads."""
    if not isinstance(registration, SpatialGoalRegistration):
        raise TypeError("registration must be a SpatialGoalRegistration.")
    route = _passive_route(registration.skill)
    registration.payload_validator.validate_route(route)
    if (
        registration.skill.definition != capture_rgb_skill_definition_v3()
        or registration.entry.route != route
    ):
        raise ValueError("RGBD executor requires its exact v3 registration.")
    if not isinstance(store, RecoverableAgentArtifactStore):
        raise TypeError("store must be a RecoverableAgentArtifactStore.")
    projection = back_projection_provider.registration
    if not isinstance(
        projection, BackProjectionRegistration
    ) or registration.entry.availability_hash != _availability(route, projection):
        raise ValueError(
            "RGBD provider differs from the exact back-projection binding."
        )
    if not callable(getattr(rgb_provider, "capture", None)):
        raise TypeError("rgb_provider must expose capture().")
    port = back_projection_provider.capture_port
    if not callable(getattr(port, "capture_placement_rgbd", None)):
        raise TypeError("back_projection_provider must expose correlated RGBD capture.")
    if (
        not isinstance(clock_domain_id, str)
        or not clock_domain_id
        or clock_domain_id.strip() != clock_domain_id
    ):
        raise ValueError("clock_domain_id must be a non-empty trimmed string.")

    def read(request: Any, operation: Any) -> PassiveReadResult:
        registration.payload_validator.validate_parameters(request.parameters)
        if request.runtime_session_id != projection.binding.runtime_session_id:
            raise ValueError("RGBD request belongs to another runtime session.")

        def check_active():
            now = clock()
            if (
                isinstance(now, bool)
                or not isinstance(now, (int, float))
                or not math.isfinite(now)
            ):
                raise ValueError("RGBD clock must return a finite number.")
            if (
                executor.cancel_requested(operation.operation_id)
                or now >= operation.deadline_at
            ):
                raise RuntimeError("RGBD capture cancelled or expired.")
            if current_station_revision() != request.expected_station_revision:
                raise RuntimeError("Station revision changed during RGBD capture.")

        check_active()
        include_depth = request.parameters["include_depth"]
        prepared = []
        prepared_bytes = 0
        metadata = []

        def prepare(
            slot_id,
            ordinal,
            data,
            width=None,
            height=None,
            source_id=None,
            role=None,
            acquired_at=None,
        ):
            nonlocal prepared_bytes
            if not isinstance(data, bytes) or not data:
                raise ValueError("RGBD provider must return non-empty original bytes.")
            slot = next(
                item
                for item in registration.skill.io_contract.output_artifact_slots
                if item.slot_id == slot_id
            )
            if len(data) > slot.max_artifact_bytes:
                raise ValueError("RGBD Artifact exceeds its declared slot byte bound.")
            if prepared_bytes + len(data) > request.parameters["max_total_bytes"]:
                raise ValueError(
                    "RGBD Artifacts exceed the requested total byte bound."
                )
            artifact = artifact_ref_for_bytes(
                data,
                media_type=slot.allowed_media_types[0],
                schema_id=slot.allowed_schema_ids[0],
                encoding=slot.allowed_encodings[0],
                compression=slot.allowed_compressions[0],
            )
            prepared.append((slot_id, ordinal, data, artifact))
            prepared_bytes += len(data)
            if slot_id == "rgb_frames":
                metadata.append(
                    {
                        "ordinal": ordinal,
                        "source_id": source_id,
                        "role": role,
                        "acquired_at": acquired_at,
                        "media_type": artifact.media_type,
                        "width": width,
                        "height": height,
                        "media_content_hash": artifact.sha256,
                    }
                )
            return artifact

        pairs = []
        if include_depth:
            role = request.parameters["selected_roles"][0]
            cut = port.capture_placement_rgbd(
                role=role, expected_station_revision=request.expected_station_revision
            )
            check_active()
            profile = projection.profile
            for capture in (cut.rgb, cut.depth):
                if (
                    capture.role != role
                    or capture.source_id != profile.source_id_by_role[role]
                    or capture.station_revision != request.expected_station_revision
                ):
                    raise RuntimeError(
                        "RGBD capture changed its registered source or revision."
                    )
            if (
                cut.rgb.image.size != profile.dimensions_by_role[role]
                or cut.depth.depth_m.shape[::-1] != profile.dimensions_by_role[role]
                or cut.rgb.captured_at != cut.depth.captured_at
            ):
                raise RuntimeError(
                    "RGBD capture changed its native grid or same-cut time."
                )
            started = ended = cut.rgb.captured_at
            revision = cut.rgb.station_revision
            rgb = prepare(
                "rgb_frames",
                0,
                cut.rgb_jpeg,
                *cut.rgb.image.size,
                source_id=cut.rgb.source_id,
                role=role,
                acquired_at=started,
            )
            depth = prepare("depth_frames", 0, cut.depth_npy)
            check_active()
            from agent_gateway.profiles.paired_rgbd import encode_paired_observation

            observation = encode_paired_observation(
                cut,
                rgb_artifact=rgb,
                depth_artifact=depth,
                reference_frame_id=profile.reference_frame_id,
                reference_frame_spec_hash=profile.reference_frame_spec_hash,
                frame_graph_hash=profile.frame_graph_hash,
                clock_domain_id=clock_domain_id,
                source_operation_id=operation.operation_id,
                source_request_hash=request.semantic_content_hash,
                runtime_session_id=request.runtime_session_id,
            )
            check_active()
            observed = prepare("paired_observations", 0, observation)
            pairs = [
                {
                    "role": role,
                    "source_id": cut.rgb.source_id,
                    "rgb_frame_ordinal": 0,
                    "depth_artifact_ordinal": 0,
                    "observation_artifact_ordinal": 0,
                    "observation_sha256": observed.sha256,
                }
            ]
        else:
            parameters = {
                key: value
                for key, value in request.parameters.items()
                if key != "include_depth"
            }
            capture = rgb_provider.capture(
                parameters, expected_station_revision=request.expected_station_revision
            )
            check_active()
            started, ended = (
                capture.capture_window_started_at,
                capture.capture_window_ended_at,
            )
            revision = capture.station_revision
            rgb_slot = next(
                item
                for item in registration.skill.io_contract.output_artifact_slots
                if item.slot_id == "rgb_frames"
            )
            if (
                not isinstance(capture.frames, (list, tuple))
                or not 1 <= len(capture.frames) <= rgb_slot.max_count
            ):
                raise ValueError(
                    "RGB provider returned an unbounded or invalid frame collection."
                )
            for ordinal, item in enumerate(capture.frames):
                check_active()
                if isinstance(item, tuple):
                    source, role, frame = item
                else:
                    frame = item
                    source, role = frame.source_id, frame.role
                if frame.media_type != RGB_FRAME_MEDIA_TYPE_V1:
                    raise ValueError("RGB provider changed the registered media type.")
                prepare(
                    "rgb_frames",
                    ordinal,
                    frame.data,
                    frame.width,
                    frame.height,
                    source_id=source,
                    role=role,
                    acquired_at=getattr(frame, "acquired_at", capture.captured_at),
                )
        if revision != request.expected_station_revision:
            raise RuntimeError("RGBD provider returned another Station revision.")
        result = {
            "station_revision": revision,
            "capture_window_started_at": started,
            "capture_window_ended_at": ended,
            "frames": metadata,
            "total_bytes": prepared_bytes,
            "include_depth": include_depth,
            "paired_observations": pairs,
        }
        identity = canonical_content_hash(
            {
                "operation_id": operation.operation_id,
                "request_hash": request.semantic_content_hash,
                "result": result,
                "artifacts": [item[3].to_wire() for item in prepared],
            },
            domain="agenticrobot.rgbd_capture.production.v3",
        )
        result["snapshot_id"] = "rgb-snapshot:" + identity
        registration.payload_validator.validate_execution_result(result)
        registration.payload_validator.validate_execution_result_for_parameters(
            request.parameters, result
        )
        bindings = []
        for slot, ordinal, data, artifact in prepared:
            check_active()
            committed = commit_artifact_bytes(
                store=store,
                data=data,
                artifact=artifact,
                producer_id="agent_gateway.provider.rgbd_capture.v3",
                runtime_session_id=request.runtime_session_id,
                idempotency_key=f"rgbd-output:{operation.operation_id}:{slot}:{ordinal}",
                now=clock(),
            )
            bindings.append(
                AgentArtifactBinding(
                    slot_id=slot,
                    ordinal=ordinal,
                    artifact=committed,
                    production_receipt_ref=f"rgbd-capture:{identity}:{slot}:{ordinal}",
                )
            )
        check_active()
        return PassiveReadResult(
            result,
            ("rgbd-capture:" + identity,),
            tuple(sorted(bindings, key=lambda item: (item.slot_id, item.ordinal))),
            reason_code="passive_rgbd_capture_completed",
        )

    executor = PassiveReadSkillExecutor(
        port_id="agent-gateway:"
        + registration.entry.deployment_binding.deployment_binding_id,
        executor_id=RGBD_CAPTURE_EXECUTOR_ID_V3,
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )
    return executor


__all__ = ["build_rgbd_capture_executor", "build_rgbd_capture_registration"]
