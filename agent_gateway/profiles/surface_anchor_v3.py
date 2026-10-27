"""Project authenticated retained RGB-D/segment3 evidence without capture or inference."""

from __future__ import annotations

import math
from dataclasses import asdict, replace

from agent_gateway.profiles.back_projection import (
    BackProjectionRegistration,
    LiveBackProjectionProvider,
    project_depth_mask_geometry,
)
from agent_gateway.profiles.paired_rgbd import (
    PAIRED_RGBD_MAX_BYTES_V1,
    PAIRED_RGBD_MAX_DEPTH_BYTES_V1,
    PAIRED_RGBD_MAX_RGB_BYTES_V1,
    decode_paired_observation,
)
from agent_gateway.profiles.rgbd_capture import _availability, _passive_route
from agent_gateway.profiles.segmentation import (
    SegmentationRegistration,
    _validate_overlay_jpeg,
    decode_segmentation_mask_npy,
)
from agent_gateway.profiles.spatial_goal import SpatialGoalRegistration
from agent_gateway.profiles.surface_anchor import (
    MAX_ANCHOR_BYTES,
    _json_bytes,
    _registration,
)
from agent_gateway.profiles.surface_anchor_selection import select_mask_interior_pixel
from agent_gateway.protocol._contract_validation import required_text
from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.skill_catalog import SkillRouteManifestEntry
from agent_gateway.protocol.skill_io_contracts import SkillArtifactSlotSpec
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.runtime.artifact_validation import validate_skill_artifact_bindings
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.skills.perception.rgbd_schemas import capture_rgb_skill_definition_v3
from agent_gateway.skills.perception.surface_anchor_v2 import (
    EXACT_ALGORITHM,
    INTERIOR_SELECTION_POLICIES,
    SELECTION_FIELDS,
    validate_anchor_v2,
)
from agent_gateway.skills.perception.surface_anchor_v3 import (
    capture_definition_v3,
    capture_schema_v3,
    validate_capture_parameters_v3,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.perception_artifacts import (
    PAIRED_RGBD_SCHEMA_ID_V1,
    SURFACE_ANCHOR_SCHEMA_ID_V2,
)


def _profile_identity(back_projection, segmentation, capture_rgb, algorithm):
    return canonical_content_hash(
        {
            "projection_route": back_projection.skill.routes[0].semantic_content_hash,
            "projection_profile": back_projection.profile.semantic_content_hash,
            "segmentation_route": segmentation.skill.routes[0].semantic_content_hash,
            "rgb_route": capture_rgb.entry.route.semantic_content_hash,
            "selection": algorithm,
        },
        domain="agenticrobot.surface_anchor.reuse_profile.v3",
    )


def _dependencies(back_projection, segmentation, capture_rgb):
    if (
        not isinstance(back_projection, BackProjectionRegistration)
        or not isinstance(segmentation, SegmentationRegistration)
        or not isinstance(capture_rgb, SpatialGoalRegistration)
    ):
        raise TypeError(
            "Surface anchor v3 requires typed projection, segment3 and capture3 registrations."
        )
    for skill, validator in (
        (back_projection.skill, back_projection.payload_validator),
        (segmentation.skill, segmentation.payload_validator),
        (capture_rgb.skill, capture_rgb.payload_validator),
    ):
        validator.validate_route(_passive_route(skill))
    if (
        segmentation.skill.definition.skill_version != "3"
        or capture_rgb.skill.definition != capture_rgb_skill_definition_v3()
        or capture_rgb.entry.route != capture_rgb.skill.routes[0]
        or capture_rgb.entry.availability_hash
        != _availability(capture_rgb.entry.route, back_projection)
    ):
        raise ValueError(
            "Surface anchor v3 requires exact capture3 and segment3 projection bindings."
        )
    if (
        segmentation.profile.rgb_capture_route_hash
        != capture_rgb.entry.route.semantic_content_hash
    ):
        raise ValueError("Segment3 is pinned to a different RGB capture Route.")
    for key in (
        "runtime_session_id",
        "control_domain_id",
        "target_binding_id",
        "station_manifest_hash",
        "execution_manifest_hash",
        "target_kind",
    ):
        if getattr(back_projection.binding, key) != getattr(segmentation.binding, key):
            raise ValueError(
                "Surface anchor v3 projection and segmentation deployments differ."
            )
    published = capture_rgb.entry.deployment_binding
    if any(
        getattr(published, key) != getattr(back_projection.binding, key)
        for key in ("target_binding_id", "target_kind", "execution_manifest_hash")
    ) or any(
        segmentation.profile.source_id_by_role.get(role) != source
        for role, source in back_projection.profile.source_id_by_role.items()
    ):
        raise ValueError("Surface anchor v3 sources or published deployment differ.")


def build_surface_anchor_registration_v3(
    *, back_projection, segmentation, capture_rgb, selection_policy="prompt_nearest"
) -> SpatialGoalRegistration:
    _dependencies(back_projection, segmentation, capture_rgb)
    if (
        not isinstance(selection_policy, str)
        or selection_policy not in INTERIOR_SELECTION_POLICIES
    ):
        raise ValueError("Unsupported surface anchor interior selection policy.")
    algorithm = INTERIOR_SELECTION_POLICIES[selection_policy]
    identity = _profile_identity(back_projection, segmentation, capture_rgb, algorithm)
    definition = capture_definition_v3()
    schema = capture_schema_v3(
        definition=definition,
        back_projection_schema=back_projection.skill.schema_entry,
        segmentation_schema=segmentation.skill.schema_entry,
        profile_hash=identity,
        interior_algorithm=algorithm,
    )

    def parameters(value):
        validate_capture_parameters_v3(
            value, dimensions_by_role=back_projection.profile.dimensions_by_role
        )

    def result(value):
        validate_anchor_v2(value, interior_algorithm=algorithm)
        back_projection.payload_validator.validate_execution_result(
            value["measurement"]
        )
        selected = value["selection"]
        if selected["mode"] == "mask_interior" and (
            selected["model_profile_id"] != segmentation.profile.model_profile_id
            or selected["model_profile_hash"] != segmentation.profile.model_profile_hash
        ):
            raise ValueError(
                "Surface anchor changed the authenticated segmentation model."
            )
        if selected["mode"] == "exact" and selected["min_score"] != 0.0:
            raise ValueError("Exact reused surface anchor has no score threshold.")

    def result_for(value, answer):
        parameters(value)
        result(answer)
        selected = answer["selection"]
        if (
            answer["measurement"]["role"] != value["role"]
            or selected["mode"] != value["selection_mode"]
            or value["target_pixel_uv"] is not None
            and _json_bytes(value["target_pixel_uv"])
            != _json_bytes(selected["requested_pixel_uv"])
        ):
            raise ValueError(
                "Surface anchor changed its role, mode or explicit point prompt."
            )

    base = SkillRouteManifestEntry(
        route=back_projection.skill.routes[0],
        deployment_binding=back_projection.binding.to_ref(),
        evaluation_plan_hash=None,
        availability_hash=back_projection.availability_hash,
    )
    registration = _registration(
        definition=definition,
        schema=schema,
        base_entry=base,
        direction="output",
        parameters=parameters,
        result=result,
        result_for=result_for,
        namespace="surface_anchor_v3",
    )
    paired = SkillArtifactSlotSpec(
        slot_id="paired_observation",
        direction="input",
        min_count=1,
        max_count=1,
        allowed_media_types=("application/json",),
        allowed_schema_ids=(PAIRED_RGBD_SCHEMA_ID_V1,),
        allowed_encodings=("utf-8",),
        allowed_compressions=("none",),
        max_artifact_bytes=PAIRED_RGBD_MAX_BYTES_V1,
        max_total_bytes=PAIRED_RGBD_MAX_BYTES_V1,
    )
    mask = next(
        item
        for item in segmentation.skill.io_contract.output_artifact_slots
        if item.slot_id == "segmentation_mask"
    )
    contract = replace(
        registration.skill.io_contract,
        input_artifact_slots=(
            paired,
            replace(mask, direction="input", min_count=0, max_count=1),
        ),
    )
    route = replace(
        registration.entry.route, skill_io_contract_hash=contract.semantic_content_hash
    )
    binding = replace(
        registration.entry.deployment_binding,
        deployment_binding_hash=canonical_content_hash(
            {
                "profile": identity,
                "route": route.semantic_content_hash,
                "projection_binding": back_projection.binding.semantic_content_hash,
                "segmentation_binding": segmentation.binding.semantic_content_hash,
                "rgb_binding": capture_rgb.entry.deployment_binding.deployment_binding_hash,
            },
            domain="agenticrobot.surface_anchor.reuse_binding.v3",
        ),
    )
    return SpatialGoalRegistration(
        SkillRegistrationTemplate(definition, schema, contract, (route,)),
        replace(registration.entry, route=route, deployment_binding=binding),
        registration.payload_validator,
    )


def _request_registration(request, registration, binding):
    route = registration.routes[0]
    profile = registration.schema_entry.parameter_profiles[0]
    if (
        request.skill_id != registration.definition.skill_id
        or request.skill_version != registration.definition.skill_version
        or request.skill_definition_hash
        != registration.definition.semantic_content_hash
        or request.route_id != route.route_id
        or request.route_version != route.route_version
        or request.route_hash != route.semantic_content_hash
        or request.deployment_binding_hash != binding.deployment_binding_hash
        or request.parameter_profile_id != profile.parameter_profile_id
        or request.parameter_profile_hash != profile.parameter_profile_hash
        or request.requested_mode not in route.supported_modes
    ):
        raise ValueError(
            "Reuse request differs from its exact registered Route/profile/deployment."
        )


def _producer(
    *,
    operation_id,
    request,
    registration,
    binding,
    validator,
    lookup_operation,
    check_active,
):
    check_active()
    retained = lookup_operation(operation_id)
    if retained is None:
        raise ValueError("Reuse producer is unavailable in this session.")
    original, source = retained
    check_active()
    if (
        source is None
        or original is None
        or source.operation_id != operation_id
        or source.status != "completed"
        or source.execution_outcome != "completed"
        or source.execution_result is None
        or original.client_id != request.client_id
        or original.runtime_session_id != request.runtime_session_id
        or source.runtime_session_id != request.runtime_session_id
        or original.skill_route_manifest_hash != request.skill_route_manifest_hash
        or original.expected_station_revision != request.expected_station_revision
        or source.request_hash != original.semantic_content_hash
    ):
        raise ValueError(
            "Reuse producer is not retained completed evidence of this client/session/revision."
        )
    _request_registration(original, registration, binding)
    validator.validate_parameters(original.parameters)
    validator.validate_execution_result(source.execution_result)
    validator.validate_execution_result_for_parameters(
        original.parameters, source.execution_result
    )
    validate_skill_artifact_bindings(
        io_contract=registration.io_contract,
        direction="input",
        bindings=original.input_artifact_bindings,
    )
    validate_skill_artifact_bindings(
        io_contract=registration.io_contract,
        direction="output",
        bindings=source.output_artifact_bindings,
    )
    check_active()
    return source, original


def _output(source, slot, ordinal):
    matches = [
        item.artifact
        for item in source.output_artifact_bindings
        if item.slot_id == slot and item.ordinal == ordinal
    ]
    if len(matches) != 1:
        raise ValueError(
            "Producer must retain exactly the specified output slot/ordinal."
        )
    return matches[0]


def _managed_bytes(store, artifact, maximum, check_active):
    if not 0 < artifact.byte_length <= maximum:
        raise ValueError("Reused Artifact exceeds its bounded byte contract.")
    check_active()
    data = store.read_bytes(artifact, maximum=maximum)
    check_active()
    return data

def _paired_cut(
    *,
    request,
    rgb_registration,
    projection,
    store,
    lookup_operation,
    clock_domain_id,
    check_active,
):
    source, original = _producer(
        operation_id=request.parameters["rgb_capture_operation_id"],
        request=request,
        registration=rgb_registration.skill,
        binding=rgb_registration.entry.deployment_binding,
        validator=rgb_registration.payload_validator,
        lookup_operation=lookup_operation,
        check_active=check_active,
    )
    result = source.execution_result
    if result["include_depth"] is not True or len(result["paired_observations"]) != 1:
        raise ValueError(
            "Surface anchor requires a paired capture3, not RGB-only capture."
        )
    pair, frame = result["paired_observations"][0], result["frames"][0]
    artifact = next(
        item.artifact
        for item in request.input_artifact_bindings
        if item.slot_id == "paired_observation"
    )
    if (
        pair["role"] != request.parameters["role"]
        or pair["source_id"] != projection.profile.source_id_by_role[pair["role"]]
        or _output(source, "paired_observations", pair["observation_artifact_ordinal"])
        != artifact
        or artifact.sha256 != pair["observation_sha256"]
    ):
        raise ValueError(
            "Input paired observation differs from the exact capture3 output."
        )
    rgb = _output(source, "rgb_frames", pair["rgb_frame_ordinal"])
    depth = _output(source, "depth_frames", pair["depth_artifact_ordinal"])
    if (
        len(source.output_artifact_bindings) != 3
        or result["total_bytes"]
        != sum(item.artifact.byte_length for item in source.output_artifact_bindings)
        or frame["ordinal"] != pair["rgb_frame_ordinal"]
        or frame["media_content_hash"] != rgb.sha256
    ):
        raise ValueError(
            "Capture3 result does not identify its complete exact paired outputs."
        )
    data = _managed_bytes(store, artifact, PAIRED_RGBD_MAX_BYTES_V1, check_active)
    rgb_data = _managed_bytes(store, rgb, PAIRED_RGBD_MAX_RGB_BYTES_V1, check_active)
    depth_data = _managed_bytes(
        store, depth, PAIRED_RGBD_MAX_DEPTH_BYTES_V1, check_active
    )
    cut, metadata = decode_paired_observation(
        data,
        rgb_jpeg=rgb_data,
        depth_npy=depth_data,
        expected_rgb_artifact=rgb,
        expected_depth_artifact=depth,
    )
    profile = projection.profile
    expected = {
        "source_operation_id": source.operation_id,
        "source_request_hash": original.semantic_content_hash,
        "runtime_session_id": request.runtime_session_id,
        "station_revision": request.expected_station_revision,
        "role": pair["role"],
        "source_id": pair["source_id"],
        "clock_domain_id": clock_domain_id,
        "reference_frame_id": profile.reference_frame_id,
        "reference_frame_spec_hash": profile.reference_frame_spec_hash,
        "frame_graph_hash": profile.frame_graph_hash,
        "captured_at": frame["acquired_at"],
        "width": frame["width"],
        "height": frame["height"],
    }
    if (
        any(metadata[key] != value for key, value in expected.items())
        or (metadata["width"], metadata["height"])
        != profile.dimensions_by_role[pair["role"]]
    ):
        raise ValueError(
            "Paired metadata changed its authenticated producer/frame/clock/role/grid."
        )
    check_active()
    return cut, source, rgb, pair["rgb_frame_ordinal"]


def _selected_mask(
    *,
    request,
    segmentation_registration,
    capture_source,
    rgb,
    rgb_ordinal,
    cut,
    store,
    lookup_operation,
    check_active,
):
    registration = segmentation_registration
    source, original = _producer(
        operation_id=request.parameters["segmentation_operation_id"],
        request=request,
        registration=registration.skill,
        binding=registration.binding.to_ref(),
        validator=registration.payload_validator,
        lookup_operation=lookup_operation,
        check_active=check_active,
    )
    result = source.execution_result
    expected = {
        "rgb_capture_operation_id": capture_source.operation_id,
        "rgb_capture_request_hash": capture_source.request_hash,
        "input_rgb_artifact_id": rgb.artifact_id,
        "input_rgb_frame_ordinal": rgb_ordinal,
        "input_rgb_content_hash": rgb.sha256,
        "source_id": cut.rgb.source_id,
        "role": cut.rgb.role,
        "captured_at": cut.rgb.captured_at,
        "station_revision": request.expected_station_revision,
        "image_width": cut.rgb.image.width,
        "image_height": cut.rgb.image.height,
    }
    if (
        any(result[key] != value for key, value in expected.items())
        or len(original.input_artifact_bindings) != 1
        or original.input_artifact_bindings[0].artifact != rgb
    ):
        raise ValueError(
            "Segment3 did not consume the exact paired capture RGB/frame/grid."
        )
    index = request.parameters["target_index"]
    if index >= len(result["items"]):
        raise ValueError("Selected segmentation target index is unavailable.")
    item = result["items"][index]
    if item["index"] != index or item["status"] != "completed":
        raise ValueError("Selected segmentation item has no completed mask.")
    requested = request.parameters["target_pixel_uv"]
    if requested is not None and _json_bytes(requested) != _json_bytes(
        item["target_pixel_uv"]
    ):
        raise ValueError(
            "Explicit anchor pixel must equal the authenticated item's point prompt."
        )
    artifact = _output(source, "segmentation_mask", item["mask_artifact_ordinal"])
    supplied = next(
        binding.artifact
        for binding in request.input_artifact_bindings
        if binding.slot_id == "segmentation_mask"
    )
    overlay = _output(source, "segmentation_overlay", 0)
    if (
        supplied != artifact
        or artifact.sha256 != item["mask"]["mask_media_content_hash"]
        or overlay.sha256 != result["overlay_media_content_hash"]
    ):
        raise ValueError(
            "Input mask or overlay differs from the selected segment3 output."
        )
    slot = next(
        slot
        for slot in registration.skill.io_contract.output_artifact_slots
        if slot.slot_id == "segmentation_mask"
    )
    mask_data = _managed_bytes(store, artifact, slot.max_artifact_bytes, check_active)
    mask = decode_segmentation_mask_npy(
        mask_data,
        width=cut.rgb.image.width,
        height=cut.rgb.image.height,
        expected_count=item["mask"]["mask_pixel_count"],
        expected_bounds=tuple(item["mask"]["mask_bounds_xyxy"]),
    )
    slot = next(
        slot
        for slot in registration.skill.io_contract.output_artifact_slots
        if slot.slot_id == "segmentation_overlay"
    )
    overlay_data = _managed_bytes(store, overlay, slot.max_artifact_bytes, check_active)
    _validate_overlay_jpeg(
        overlay_data, width=cut.rgb.image.width, height=cut.rgb.image.height
    )
    check_active()
    return item, result, mask


def build_surface_anchor_executor_v3(
    *,
    registration,
    provider,
    segmentation_registration,
    rgb_registration,
    store,
    lookup_operation,
    current_station_revision,
    clock,
    clock_domain_id,
    max_retained_operations,
) -> PassiveReadSkillExecutor:
    if not isinstance(provider, LiveBackProjectionProvider) or not isinstance(
        registration, SpatialGoalRegistration
    ):
        raise TypeError(
            "Surface anchor v3 requires typed registration and the existing projection provider."
        )
    _dependencies(provider.registration, segmentation_registration, rgb_registration)
    required_text(clock_domain_id, field_name="clock_domain_id", maximum_length=256)
    if not isinstance(store, RecoverableAgentArtifactStore) or not callable(
        lookup_operation
    ):
        raise TypeError(
            "Surface anchor v3 requires the managed store and retained operation lookup."
        )
    algorithms = registration.skill.schema_entry.execution_result_schema["properties"][
        "selection"
    ]["properties"]["algorithm_id"]["enum"]
    policies = [
        policy
        for policy, algorithm in INTERIOR_SELECTION_POLICIES.items()
        if list(algorithms) == [EXACT_ALGORITHM, algorithm]
    ]
    if len(policies) != 1:
        raise ValueError("Surface anchor v3 requires one reviewed interior policy.")
    expected = build_surface_anchor_registration_v3(
        back_projection=provider.registration,
        segmentation=segmentation_registration,
        capture_rgb=rgb_registration,
        selection_policy=policies[0],
    )
    if registration.skill != expected.skill or registration.entry != expected.entry:
        raise ValueError(
            "Surface anchor v3 providers differ from the published registration."
        )
    registration.payload_validator.validate_route(registration.entry.route)
    algorithm = INTERIOR_SELECTION_POLICIES[policies[0]]

    def read(request, operation):
        _request_registration(
            request, registration.skill, registration.entry.deployment_binding
        )
        registration.payload_validator.validate_parameters(request.parameters)
        validate_skill_artifact_bindings(
            io_contract=registration.skill.io_contract,
            direction="input",
            bindings=request.input_artifact_bindings,
        )
        slots = [
            (item.slot_id, item.ordinal) for item in request.input_artifact_bindings
        ]
        required = [("paired_observation", 0)]
        if request.parameters["selection_mode"] == "mask_interior":
            required.append(("segmentation_mask", 0))
        if (
            sorted(slots) != required
            or request.runtime_session_id
            != provider.registration.binding.runtime_session_id
        ):
            raise ValueError(
                "Surface anchor inputs must match the exact mode and runtime session."
            )

        def check_active():
            now = clock()
            if type(now) not in (int, float) or not math.isfinite(now):
                raise ValueError("Surface anchor clock must be finite.")
            if (
                executor.cancel_requested(operation.operation_id)
                or now >= operation.deadline_at
            ):
                raise RuntimeError("Surface anchor reuse cancelled or expired.")
            if current_station_revision() != request.expected_station_revision:
                raise RuntimeError(
                    "Station revision changed during surface anchor reuse."
                )

        check_active()
        cut, capture_source, rgb, rgb_ordinal = _paired_cut(
            request=request,
            rgb_registration=rgb_registration,
            projection=provider.registration,
            store=store,
            lookup_operation=lookup_operation,
            clock_domain_id=clock_domain_id,
            check_active=check_active,
        )
        selected = request.parameters["target_pixel_uv"]
        selection = dict.fromkeys(SELECTION_FIELDS)
        selection.update(
            mode=request.parameters["selection_mode"],
            requested_pixel_uv=selected,
            language_prompt=None,
            min_score=0.0,
            algorithm_id=EXACT_ALGORITHM,
            input_rgb_content_hash=rgb.sha256,
        )
        geometry = None
        if request.parameters["selection_mode"] == "mask_interior":
            item, segmentation, mask = _selected_mask(
                request=request,
                segmentation_registration=segmentation_registration,
                capture_source=capture_source,
                rgb=rgb,
                rgb_ordinal=rgb_ordinal,
                cut=cut,
                store=store,
                lookup_operation=lookup_operation,
                check_active=check_active,
            )
            profile = provider.registration.profile
            check_active()
            selected, distance = select_mask_interior_pixel(
                mask,
                cut.depth.depth_m,
                requested_pixel_uv=item["target_pixel_uv"],
                maximum_depth_m=profile.maximum_depth_m,
                interior_algorithm=algorithm,
            )
            check_active()
            geometry = asdict(
                project_depth_mask_geometry(
                    cut.depth,
                    mask,
                    minimum_valid_points=1,
                    maximum_depth_m=profile.maximum_depth_m,
                    robust_lower_quantile=profile.robust_lower_quantile,
                    robust_upper_quantile=profile.robust_upper_quantile,
                )
            )
            selection.update(
                {key: item[key] for key in ("language_prompt", "min_score")}
            )
            selection.update(
                {
                    key: item["mask"][key]
                    for key in (
                        "mask_media_content_hash",
                        "mask_bounds_xyxy",
                        "mask_pixel_count",
                        "score",
                    )
                }
            )
            selection.update(
                {
                    key: segmentation[key]
                    for key in (
                        "overlay_media_content_hash",
                        "model_profile_id",
                        "model_profile_hash",
                    )
                }
            )
            selection.update(
                requested_pixel_uv=item["target_pixel_uv"],
                algorithm_id=algorithm,
                interior_distance_px=distance,
            )
        check_active()
        measurement = provider.project_capture(
            cut.depth,
            {
                "role": request.parameters["role"],
                "target_pixel_uv": selected,
                "pixel_region_xyxy": None,
                "world_z_min_m": None,
                "world_z_max_m": None,
            },
            expected_station_revision=request.expected_station_revision,
        )
        check_active()
        result = {
            "measurement": measurement,
            "target_pixel_uv": list(selected),
            "captured_at": cut.depth.captured_at,
            "clock_domain_id": clock_domain_id,
            "source_operation_id": operation.operation_id,
            "source_request_hash": request.semantic_content_hash,
            "runtime_session_id": request.runtime_session_id,
            "selection": selection,
            "mask_geometry": geometry,
        }
        registration.payload_validator.validate_execution_result_for_parameters(
            request.parameters, result
        )
        data = _json_bytes(result)
        if len(data) > MAX_ANCHOR_BYTES:
            raise ValueError("Surface anchor exceeds its bounded v2 document contract.")
        artifact = artifact_ref_for_bytes(
            data,
            media_type="application/json",
            schema_id=SURFACE_ANCHOR_SCHEMA_ID_V2,
            encoding="utf-8",
            compression="none",
        )
        check_active()
        commit_artifact_bytes(
            store=store,
            data=data,
            artifact=artifact,
            producer_id=registration.skill.definition.skill_id,
            runtime_session_id=request.runtime_session_id,
            idempotency_key=operation.operation_id,
            now=clock(),
        )
        check_active()
        receipt = "surface-anchor-reuse:" + canonical_content_hash(
            {
                "operation_id": operation.operation_id,
                "request_hash": request.semantic_content_hash,
                "artifact": artifact.to_wire(),
                "inputs": [
                    binding.to_wire() for binding in request.input_artifact_bindings
                ],
            },
            domain="agenticrobot.surface_anchor.production.v3",
        )
        return PassiveReadResult(
            result,
            (receipt,),
            (
                AgentArtifactBinding(
                    slot_id="surface_anchor",
                    ordinal=0,
                    artifact=artifact,
                    production_receipt_ref=receipt,
                ),
            ),
            reason_code="surface_anchor_reused",
        )

    executor = PassiveReadSkillExecutor(
        port_id=registration.entry.route.route_id,
        executor_id="passive.surface-anchor.v3",
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )
    return executor


__all__ = ["build_surface_anchor_executor_v3", "build_surface_anchor_registration_v3"]
