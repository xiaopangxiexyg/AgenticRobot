"""Real-camera bindings for the target-neutral perception Artifact Skills.

Construction is pure.  The profile describes the reviewed camera inventory,
finite Artifact bounds, and two passive Routes; it never opens a camera,
Artifact store, socket, or robot resource.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agent_gateway.profiles.back_projection import (
    BackProjectionRegistration,
    build_back_projection_registration,
)
from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
)
from agent_gateway.protocol.skill_io_contracts import (
    SkillArtifactSlotSpec,
    SkillIOContract,
)
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception.definition import (
    capture_depth_skill_definition_v1,
    capture_depth_skill_definition_v2,
    capture_rgb_skill_definition_v1,
    capture_rgb_skill_definition_v2,
)
from agent_gateway.skills.perception.schemas import (
    DEPTH_FRAME_COMPRESSION_V1,
    DEPTH_FRAME_ENCODING_V1,
    DEPTH_FRAME_MEDIA_TYPE_V1,
    DEPTH_FRAME_SCHEMA_ID_V1,
    RGB_FRAME_COMPRESSION_V1,
    RGB_FRAME_ENCODING_V1,
    RGB_FRAME_MEDIA_TYPE_V1,
    RGB_FRAME_SCHEMA_ID_V1,
    capture_depth_schema_entry_v1,
    capture_depth_schema_entry_v2,
    capture_rgb_schema_entry_v1,
    capture_rgb_schema_entry_v2,
    validate_capture_depth_parameters_v1,
    validate_capture_depth_result_for_parameters_v1,
    validate_capture_depth_result_for_parameters_v2,
    validate_capture_depth_result_v1,
    validate_capture_depth_result_v2,
    validate_capture_rgb_parameters_v1,
    validate_capture_rgb_result_for_parameters_v1,
    validate_capture_rgb_result_for_parameters_v2,
    validate_capture_rgb_result_v1,
    validate_capture_rgb_result_v2,
    validate_no_task_evaluation_v1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

from .perception_frames import RealPerceptionFrameBinding

REAL_CAPTURE_RGB_ROUTE_ID_V1 = "real.realsense.perception.capture_rgb"
REAL_CAPTURE_RGB_ROUTE_ID_V2 = "real.realsense.perception.capture_rgb_window"
REAL_CAPTURE_DEPTH_ROUTE_ID_V1 = "real.realsense.perception.capture_depth"
REAL_CAPTURE_DEPTH_ROUTE_ID_V2 = (
    "real.realsense.perception.capture_depth_station"
)
REAL_CAPTURE_RGB_EXECUTOR_ID_V1 = "control-station.real-passive-rgb-capture.v1"
REAL_CAPTURE_RGB_EXECUTOR_ID_V2 = "control-station.real-passive-rgb-capture.v2"
REAL_CAPTURE_DEPTH_EXECUTOR_ID_V1 = "control-station.real-passive-depth-capture.v1"
REAL_CAPTURE_DEPTH_EXECUTOR_ID_V2 = (
    "control-station.real-passive-depth-capture.v2"
)
REAL_CAPTURE_RGB_OUTPUT_SLOT_ID_V1 = "rgb_frames"
REAL_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1 = "depth_frame"
REAL_CAPTURE_MAX_ARTIFACT_BYTES_V1 = 16 * 1024 * 1024
REAL_CAPTURE_MAX_FRAMES_V1 = 8


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


def _sources(
    values: tuple[tuple[str, str, int, int], ...],
    *,
    label: str,
) -> tuple[tuple[str, str, int, int], ...]:
    normalized: list[tuple[str, str, int, int]] = []
    for source_id, role, width, height in values:
        source = _required(source_id, field_name=f"{label}.source_id")
        semantic_role = _required(role, field_name=f"{label}.role")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
        ):
            raise ValueError(f"{label} dimensions must be positive integers.")
        normalized.append((source, semantic_role, width, height))
    normalized.sort(key=lambda item: item[1])
    if (
        not normalized
        or len({item[0] for item in normalized}) != len(normalized)
        or len({item[1] for item in normalized}) != len(normalized)
    ):
        raise ValueError(f"{label} requires unique sources and semantic roles.")
    return tuple(normalized)


@dataclass(frozen=True)
class RealPerceptionCaptureProfile:
    """Reviewed RealSense role catalogs and finite capture bounds."""

    rgb_sources: tuple[tuple[str, str, int, int], ...]
    depth_sources: tuple[tuple[str, str, int, int], ...]
    source_binding_hash: str
    max_artifact_bytes: int = REAL_CAPTURE_MAX_ARTIFACT_BYTES_V1
    max_frame_age_ms: int = 200
    max_inter_camera_skew_ms: int = 100
    profile_id: str = "real-realsense-registered-perception-roles-v1"

    def __post_init__(self) -> None:
        rgb = _sources(self.rgb_sources, label="RGB profile")
        depth = _sources(self.depth_sources, label="Depth profile")
        for field_name in (
            "max_artifact_bytes",
            "max_frame_age_ms",
            "max_inter_camera_skew_ms",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if len(rgb) > REAL_CAPTURE_MAX_FRAMES_V1:
            raise ValueError("RGB source count exceeds the protocol bound of eight.")
        rgb_by_source = {item[0]: item[1] for item in rgb}
        if any(
            source_id not in rgb_by_source or rgb_by_source[source_id] != role
            for source_id, role, _width, _height in depth
        ):
            raise ValueError("Every depth source must reuse its registered RGB role.")
        if any(
            width * height * 4 + 1024 > self.max_artifact_bytes
            for _source_id, _role, width, height in depth
        ):
            raise ValueError("A depth frame exceeds the Artifact byte bound.")
        object.__setattr__(self, "rgb_sources", rgb)
        object.__setattr__(self, "depth_sources", depth)
        object.__setattr__(
            self,
            "source_binding_hash",
            _required(self.source_binding_hash, field_name="source_binding_hash"),
        )
        object.__setattr__(
            self,
            "profile_id",
            _required(self.profile_id, field_name="profile_id"),
        )

    @property
    def rgb_roles(self) -> tuple[str, ...]:
        return tuple(item[1] for item in self.rgb_sources)

    @property
    def depth_roles(self) -> tuple[str, ...]:
        return tuple(item[1] for item in self.depth_sources)

    @property
    def rgb_source_id_by_role(self) -> Mapping[str, str]:
        return {role: source for source, role, _width, _height in self.rgb_sources}

    @property
    def depth_source_id_by_role(self) -> Mapping[str, str]:
        return {role: source for source, role, _width, _height in self.depth_sources}

    @property
    def depth_dimensions_by_role(self) -> Mapping[str, tuple[int, int]]:
        return {
            role: (width, height)
            for _source, role, width, height in self.depth_sources
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "profile_id": self.profile_id,
                "rgb_sources": self.rgb_sources,
                "depth_sources": self.depth_sources,
                "source_binding_hash": self.source_binding_hash,
                "max_artifact_bytes": self.max_artifact_bytes,
                "max_frame_age_ms": self.max_frame_age_ms,
                "max_inter_camera_skew_ms": self.max_inter_camera_skew_ms,
                "rgb_encoding": RGB_FRAME_ENCODING_V1,
                "depth_encoding": DEPTH_FRAME_ENCODING_V1,
                "capture_semantics": "bounded_latest_realsense_frames",
                "downstream_control_effects": "forbidden",
            },
            domain="dexteleop.agent_skill.real_perception_capture_profile.v1",
        )


@dataclass(frozen=True)
class RealPerceptionDeploymentBinding:
    deployment_binding_id: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    profile_hash: str
    executor_id: str
    deployment_binding_version: str = "1"

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                field_name: getattr(self, field_name)
                for field_name in self.__dataclass_fields__
            }
            | {
                "target_kind": "real",
                "control_effect_class": "read_only",
                "downstream_control_effects": "forbidden",
                "capture_owner": "exclusive_realsense_resource_owner",
            },
            domain="dexteleop.agent_skill.real_perception_deployment.v1",
        )

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=self.deployment_binding_id,
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.target_binding_id,
            target_kind="real",
            execution_manifest_hash=self.execution_manifest_hash,
        )


@dataclass(frozen=True)
class RealPerceptionDepthDeploymentBindingV2:
    deployment_binding_id: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    profile_hash: str
    source_frame_id: str
    source_frame_spec_hash: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    frame_graph_hash: str
    executor_id: str = REAL_CAPTURE_DEPTH_EXECUTOR_ID_V2
    deployment_binding_version: str = "2"

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                field_name: getattr(self, field_name)
                for field_name in self.__dataclass_fields__
            }
            | {
                "target_kind": "real",
                "control_effect_class": "read_only",
                "downstream_control_effects": "forbidden",
                "capture_owner": "exclusive_realsense_resource_owner",
                "pose_semantics": "T_reference_camera_optical_frame",
            },
            domain="dexteleop.agent_skill.real_perception_depth_deployment.v2",
        )

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=self.deployment_binding_id,
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.target_binding_id,
            target_kind="real",
            execution_manifest_hash=self.execution_manifest_hash,
        )


@dataclass(frozen=True)
class RealPerceptionCaptureRegistration:
    profile: RealPerceptionCaptureProfile
    rgb: SkillRegistrationTemplate
    depth: SkillRegistrationTemplate
    rgb_binding: RealPerceptionDeploymentBinding
    depth_binding: (
        RealPerceptionDeploymentBinding | RealPerceptionDepthDeploymentBindingV2
    )
    payload_validators: tuple[AgentSkillPayloadValidatorSet, ...]
    availability_hash_by_route_hash: Mapping[str, str]
    frame_binding: RealPerceptionFrameBinding | None = None
    back_projection: BackProjectionRegistration | None = None

    def __post_init__(self) -> None:
        if self.frame_binding is None:
            if self.back_projection is not None:
                raise ValueError("Real Depth v1 cannot expose Station back-projection.")
            if not isinstance(self.depth_binding, RealPerceptionDeploymentBinding):
                raise TypeError("Real Depth v1 requires its v1 deployment binding.")
            return
        if not isinstance(
            self.depth_binding,
            RealPerceptionDepthDeploymentBindingV2,
        ):
            raise TypeError("Real Depth v2 requires its v2 deployment binding.")
        expected = {
            "source_frame_id": self.frame_binding.source_frame_id,
            "source_frame_spec_hash": self.frame_binding.source_frame_spec_hash,
            "reference_frame_id": self.frame_binding.reference_frame_id,
            "reference_frame_spec_hash": (
                self.frame_binding.reference_frame_spec_hash
            ),
            "frame_graph_hash": self.frame_binding.frame_graph.semantic_content_hash,
        }
        if any(
            getattr(self.depth_binding, field_name) != value
            for field_name, value in expected.items()
        ):
            raise ValueError("Real Depth v2 binding differs from its typed frame graph.")
        if self.back_projection is not None:
            profile = self.back_projection.profile
            if (
                profile.target_kind != "real"
                or profile.current_mode not in {"real_active", "real_observe"}
                or profile.reference_frame_id != self.frame_binding.reference_frame_id
                or profile.reference_frame_spec_hash
                != self.frame_binding.reference_frame_spec_hash
                or profile.frame_graph_hash
                != self.frame_binding.frame_graph.semantic_content_hash
            ):
                raise ValueError("Real back-projection frame binding differs.")


def _source_tuples(
    catalog: tuple[Mapping[str, object], ...],
    *,
    label: str,
) -> tuple[tuple[str, str, int, int], ...]:
    fields = {"source_id", "role", "native_width", "native_height"}
    values = []
    for item in catalog:
        if not isinstance(item, Mapping) or set(item) != fields:
            raise ValueError(f"{label} source catalog entry has unexpected fields.")
        values.append(
            (
                item["source_id"],
                item["role"],
                item["native_width"],
                item["native_height"],
            )
        )
    return tuple(values)


def _build_real_perception_capture_registration(
    *,
    rgb_source_catalog: tuple[Mapping[str, object], ...],
    depth_source_catalog: tuple[Mapping[str, object], ...],
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
    source_binding_hash: str,
    current_mode: str = "real_active",
    frame_binding: RealPerceptionFrameBinding | None = None,
    aligned_rgbd_source_catalog: tuple[Mapping[str, object], ...] | None = None,
) -> RealPerceptionCaptureRegistration:
    """Build both exact T0 Routes from one reviewed real-camera inventory."""

    mode = _required(current_mode, field_name="current_mode")
    if mode not in {"real_active", "real_observe"}:
        raise ValueError("current_mode must be real_active or real_observe.")

    is_v2 = frame_binding is not None
    if (aligned_rgbd_source_catalog is None) != (frame_binding is None):
        raise ValueError(
            "Aligned RGB-D sources and the Station frame binding must coexist."
        )
    profile = RealPerceptionCaptureProfile(
        rgb_sources=_source_tuples(rgb_source_catalog, label="RGB"),
        depth_sources=_source_tuples(depth_source_catalog, label="Depth"),
        source_binding_hash=source_binding_hash,
        profile_id=(
            "real-realsense-registered-perception-roles-v2"
            if is_v2
            else "real-realsense-registered-perception-roles-v1"
        ),
    )
    rgb_definition = (
        capture_rgb_skill_definition_v2()
        if is_v2
        else capture_rgb_skill_definition_v1()
    )
    if frame_binding is not None and not isinstance(
        frame_binding,
        RealPerceptionFrameBinding,
    ):
        raise TypeError("frame_binding must be typed or None.")
    depth_definition = (
        capture_depth_skill_definition_v1()
        if frame_binding is None
        else capture_depth_skill_definition_v2()
    )
    rgb_schema_builder = (
        capture_rgb_schema_entry_v2 if is_v2 else capture_rgb_schema_entry_v1
    )
    rgb_schema = rgb_schema_builder(
        skill_definition_hash=rgb_definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id + ".rgb",
        parameter_profile_hash=profile.semantic_content_hash,
        allowed_roles=profile.rgb_roles,
        max_total_bytes=profile.max_artifact_bytes,
    )
    if frame_binding is None:
        depth_schema = capture_depth_schema_entry_v1(
            skill_definition_hash=depth_definition.semantic_content_hash,
            parameter_profile_id=profile.profile_id + ".depth",
            parameter_profile_hash=profile.semantic_content_hash,
            allowed_roles=profile.depth_roles,
        )
    else:
        depth_schema = capture_depth_schema_entry_v2(
            skill_definition_hash=depth_definition.semantic_content_hash,
            parameter_profile_id=profile.profile_id + ".depth_station",
            parameter_profile_hash=profile.semantic_content_hash,
            allowed_roles=profile.depth_roles,
            reference_frame_id=frame_binding.reference_frame_id,
            reference_frame_spec_hash=(
                frame_binding.reference_frame_spec_hash
            ),
            frame_graph_hash=frame_binding.frame_graph.semantic_content_hash,
        )
    rgb_io = SkillIOContract(
        skill_id=rgb_definition.skill_id,
        skill_version=rgb_definition.skill_version,
        skill_definition_hash=rgb_definition.semantic_content_hash,
        parameter_schema_hash=rgb_schema.parameter_schema_hash,
        execution_result_schema_hash=rgb_schema.execution_result_schema_hash,
        task_evaluation_schema_hash=rgb_schema.task_evaluation_schema_hash,
        output_artifact_slots=(
            SkillArtifactSlotSpec(
                slot_id=REAL_CAPTURE_RGB_OUTPUT_SLOT_ID_V1,
                direction="output",
                min_count=1,
                max_count=len(profile.rgb_sources),
                allowed_media_types=(RGB_FRAME_MEDIA_TYPE_V1,),
                allowed_schema_ids=(RGB_FRAME_SCHEMA_ID_V1,),
                allowed_encodings=(RGB_FRAME_ENCODING_V1,),
                allowed_compressions=(RGB_FRAME_COMPRESSION_V1,),
                max_artifact_bytes=profile.max_artifact_bytes,
                max_total_bytes=profile.max_artifact_bytes,
            ),
        ),
    )
    depth_io = SkillIOContract(
        skill_id=depth_definition.skill_id,
        skill_version=depth_definition.skill_version,
        skill_definition_hash=depth_definition.semantic_content_hash,
        parameter_schema_hash=depth_schema.parameter_schema_hash,
        execution_result_schema_hash=depth_schema.execution_result_schema_hash,
        task_evaluation_schema_hash=depth_schema.task_evaluation_schema_hash,
        output_artifact_slots=(
            SkillArtifactSlotSpec(
                slot_id=REAL_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1,
                direction="output",
                min_count=1,
                max_count=1,
                allowed_media_types=(DEPTH_FRAME_MEDIA_TYPE_V1,),
                allowed_schema_ids=(DEPTH_FRAME_SCHEMA_ID_V1,),
                allowed_encodings=(DEPTH_FRAME_ENCODING_V1,),
                allowed_compressions=(DEPTH_FRAME_COMPRESSION_V1,),
                max_artifact_bytes=profile.max_artifact_bytes,
                max_total_bytes=profile.max_artifact_bytes,
            ),
        ),
    )

    def route(definition, schema, io_contract, route_id, *, route_version="1"):
        return SkillExecutionRoute(
            skill_id=definition.skill_id,
            skill_version=definition.skill_version,
            skill_definition_hash=definition.semantic_content_hash,
            route_id=route_id,
            route_version=route_version,
            parameter_schema_hash=schema.parameter_schema_hash,
            execution_result_schema_hash=schema.execution_result_schema_hash,
            task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
            allowed_parameter_profiles=schema.parameter_profiles,
            control_effect_class="read_only",
            privilege_tier="t0_read_only",
            supported_modes=(mode,),
            target_kind="real",
            may_move_robot=False,
            may_open_publication=False,
            requires_operator_confirmation=False,
            requires_deadman=False,
            max_operation_duration_ms=3_000,
            cancellation_policy="cancel_only",
            requires_task_evaluation=False,
            skill_io_contract_hash=io_contract.semantic_content_hash,
        )

    rgb_route = route(
        rgb_definition,
        rgb_schema,
        rgb_io,
        REAL_CAPTURE_RGB_ROUTE_ID_V2 if is_v2 else REAL_CAPTURE_RGB_ROUTE_ID_V1,
        route_version="2" if is_v2 else "1",
    )
    depth_route = route(
        depth_definition,
        depth_schema,
        depth_io,
        (
            REAL_CAPTURE_DEPTH_ROUTE_ID_V1
            if frame_binding is None
            else REAL_CAPTURE_DEPTH_ROUTE_ID_V2
        ),
        route_version="1" if frame_binding is None else "2",
    )
    rgb_registration = SkillRegistrationTemplate(
        definition=rgb_definition,
        schema_entry=rgb_schema,
        io_contract=rgb_io,
        routes=(rgb_route,),
    )
    depth_registration = SkillRegistrationTemplate(
        definition=depth_definition,
        schema_entry=depth_schema,
        io_contract=depth_io,
        routes=(depth_route,),
    )

    def binding(
        route_value,
        executor_id,
        modality,
        *,
        deployment_binding_version="1",
    ):
        return RealPerceptionDeploymentBinding(
            deployment_binding_id=(
                f"real.realsense.passive_{modality}_capture:" + runtime_session_id
            ),
            route_hash=route_value.semantic_content_hash,
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            target_binding_id=target_binding_id,
            station_manifest_hash=station_manifest_hash,
            execution_manifest_hash=execution_manifest_hash,
            profile_hash=profile.semantic_content_hash,
            executor_id=executor_id,
            deployment_binding_version=deployment_binding_version,
        )

    rgb_binding = binding(
        rgb_route,
        REAL_CAPTURE_RGB_EXECUTOR_ID_V2 if is_v2 else REAL_CAPTURE_RGB_EXECUTOR_ID_V1,
        "rgb_window" if is_v2 else "rgb",
        deployment_binding_version="2" if is_v2 else "1",
    )
    if frame_binding is None:
        depth_binding = binding(
            depth_route,
            REAL_CAPTURE_DEPTH_EXECUTOR_ID_V1,
            "depth",
        )
    else:
        depth_binding = RealPerceptionDepthDeploymentBindingV2(
            deployment_binding_id=(
                "real.realsense.passive_depth_capture_station:"
                + runtime_session_id
            ),
            route_hash=depth_route.semantic_content_hash,
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            target_binding_id=target_binding_id,
            station_manifest_hash=station_manifest_hash,
            execution_manifest_hash=execution_manifest_hash,
            profile_hash=profile.semantic_content_hash,
            source_frame_id=frame_binding.source_frame_id,
            source_frame_spec_hash=frame_binding.source_frame_spec_hash,
            reference_frame_id=frame_binding.reference_frame_id,
            reference_frame_spec_hash=frame_binding.reference_frame_spec_hash,
            frame_graph_hash=frame_binding.frame_graph.semantic_content_hash,
        )

    def validator(definition, route_value, *, modality):
        if modality == "rgb":
            validate_parameters = lambda payload: validate_capture_rgb_parameters_v1(
                payload,
                allowed_roles=profile.rgb_roles,
                max_total_bytes=profile.max_artifact_bytes,
            )
            if is_v2:
                validate_result = lambda payload: validate_capture_rgb_result_v2(
                    payload,
                    source_id_by_role=profile.rgb_source_id_by_role,
                    max_total_bytes=profile.max_artifact_bytes,
                )
                validate_for_parameters = (
                    validate_capture_rgb_result_for_parameters_v2
                )
            else:
                validate_result = lambda payload: validate_capture_rgb_result_v1(
                    payload,
                    source_id_by_role=profile.rgb_source_id_by_role,
                    max_total_bytes=profile.max_artifact_bytes,
                )
                validate_for_parameters = (
                    validate_capture_rgb_result_for_parameters_v1
                )
        elif frame_binding is None:
            validate_parameters = lambda payload: validate_capture_depth_parameters_v1(
                payload,
                allowed_roles=profile.depth_roles,
            )
            validate_result = lambda payload: validate_capture_depth_result_v1(
                payload,
                source_id_by_role=profile.depth_source_id_by_role,
                dimensions_by_role=profile.depth_dimensions_by_role,
            )
            validate_for_parameters = validate_capture_depth_result_for_parameters_v1
        else:
            validate_parameters = lambda payload: validate_capture_depth_parameters_v1(
                payload,
                allowed_roles=profile.depth_roles,
            )
            validate_result = lambda payload: validate_capture_depth_result_v2(
                payload,
                source_id_by_role=profile.depth_source_id_by_role,
                dimensions_by_role=profile.depth_dimensions_by_role,
                reference_frame_id=frame_binding.reference_frame_id,
                reference_frame_spec_hash=(
                    frame_binding.reference_frame_spec_hash
                ),
                frame_graph_hash=frame_binding.frame_graph.semantic_content_hash,
            )
            validate_for_parameters = validate_capture_depth_result_for_parameters_v2
        return AgentSkillPayloadValidatorSet(
            validator_id=(
                f"dexteleop.agent_skill.validator.real_{modality}_capture:"
                + runtime_session_id
            ),
            skill_id=definition.skill_id,
            skill_version=definition.skill_version,
            route_id=route_value.route_id,
            route_version=route_value.route_version,
            parameter_schema_hash=route_value.parameter_schema_hash,
            execution_result_schema_hash=route_value.execution_result_schema_hash,
            task_evaluation_schema_hash=route_value.task_evaluation_schema_hash,
            validate_parameters=validate_parameters,
            validate_execution_result=validate_result,
            validate_execution_result_for_parameters=validate_for_parameters,
            validate_task_evaluation=validate_no_task_evaluation_v1,
        )

    availability = {
        route_value.semantic_content_hash: canonical_content_hash(
            {
                "status": "configured_requires_realsense_and_artifact_store",
                "route_hash": route_value.semantic_content_hash,
                "deployment_binding_hash": binding_value.semantic_content_hash,
                "profile_hash": profile.semantic_content_hash,
            },
            domain=(
                "dexteleop.agent_skill.real_rgb_capture_availability.v2"
                if is_v2 and route_value is rgb_route
                else "dexteleop.agent_skill.real_perception_availability.v1"
            ),
        )
        for route_value, binding_value in (
            (rgb_route, rgb_binding),
            (depth_route, depth_binding),
        )
    }
    back_projection = (
        None
        if frame_binding is None or aligned_rgbd_source_catalog is None
        else build_back_projection_registration(
            target_kind="real",
            current_mode=mode,
            source_catalog=aligned_rgbd_source_catalog,
            reference_frame_id=frame_binding.reference_frame_id,
            reference_frame_spec_hash=frame_binding.reference_frame_spec_hash,
            frame_graph_hash=frame_binding.frame_graph.semantic_content_hash,
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            target_binding_id=target_binding_id,
            station_manifest_hash=station_manifest_hash,
            execution_manifest_hash=execution_manifest_hash,
        )
    )
    return RealPerceptionCaptureRegistration(
        profile=profile,
        rgb=rgb_registration,
        depth=depth_registration,
        rgb_binding=rgb_binding,
        depth_binding=depth_binding,
        payload_validators=(
            validator(rgb_definition, rgb_route, modality="rgb"),
            validator(depth_definition, depth_route, modality="depth"),
        ),
        availability_hash_by_route_hash=availability,
        frame_binding=frame_binding,
        back_projection=back_projection,
    )


def build_real_perception_capture_registration(
    *,
    rgb_source_catalog: tuple[Mapping[str, object], ...],
    depth_source_catalog: tuple[Mapping[str, object], ...],
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
    source_binding_hash: str,
    current_mode: str = "real_active",
) -> RealPerceptionCaptureRegistration:
    """Build the legacy v1 depth registration and passive RGB registration."""

    return _build_real_perception_capture_registration(
        rgb_source_catalog=rgb_source_catalog,
        depth_source_catalog=depth_source_catalog,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        target_binding_id=target_binding_id,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        source_binding_hash=source_binding_hash,
        current_mode=current_mode,
        frame_binding=None,
        aligned_rgbd_source_catalog=None,
    )


def build_real_perception_capture_registration_v2(
    *,
    rgb_source_catalog: tuple[Mapping[str, object], ...],
    depth_source_catalog: tuple[Mapping[str, object], ...],
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
    source_binding_hash: str,
    frame_binding: RealPerceptionFrameBinding,
    aligned_rgbd_source_catalog: tuple[Mapping[str, object], ...],
    current_mode: str = "real_active",
) -> RealPerceptionCaptureRegistration:
    """Build RGB/Depth v2 against one exact source-to-Station adapter."""

    return _build_real_perception_capture_registration(
        rgb_source_catalog=rgb_source_catalog,
        depth_source_catalog=depth_source_catalog,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        target_binding_id=target_binding_id,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        source_binding_hash=source_binding_hash,
        current_mode=current_mode,
        frame_binding=frame_binding,
        aligned_rgbd_source_catalog=aligned_rgbd_source_catalog,
    )


__all__ = [
    "REAL_CAPTURE_DEPTH_EXECUTOR_ID_V1",
    "REAL_CAPTURE_DEPTH_EXECUTOR_ID_V2",
    "REAL_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1",
    "REAL_CAPTURE_DEPTH_ROUTE_ID_V1",
    "REAL_CAPTURE_DEPTH_ROUTE_ID_V2",
    "REAL_CAPTURE_MAX_ARTIFACT_BYTES_V1",
    "REAL_CAPTURE_RGB_EXECUTOR_ID_V1",
    "REAL_CAPTURE_RGB_EXECUTOR_ID_V2",
    "REAL_CAPTURE_RGB_OUTPUT_SLOT_ID_V1",
    "REAL_CAPTURE_RGB_ROUTE_ID_V1",
    "REAL_CAPTURE_RGB_ROUTE_ID_V2",
    "RealPerceptionCaptureProfile",
    "RealPerceptionCaptureRegistration",
    "RealPerceptionDeploymentBinding",
    "RealPerceptionDepthDeploymentBindingV2",
    "build_real_perception_capture_registration",
    "build_real_perception_capture_registration_v2",
]
