"""Read-only Route for rendering one retained Station point-grasp candidate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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
    render_point_grasp_candidate_skill_definition_v1,
)
from agent_gateway.skills.perception.point_grasp_render_schemas import (
    point_grasp_candidate_render_schema_entry_v1,
    validate_point_grasp_candidate_render_parameters_v1,
    validate_point_grasp_candidate_render_result_for_parameters_v1,
    validate_point_grasp_candidate_render_result_v1,
)
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.perception_artifacts import (
    LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,
    LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,
    LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
    POINT_GRASP_ANNOTATED_SCHEMA_ID_V1,
)

from .station_point_grasp_deployment import StationBoundPointGraspRegistration

REAL_STATION_POINT_GRASP_RENDER_ROUTE_ID_V1 = (
    "real.realsense.perception.render_point_grasp_candidate"
)
REAL_STATION_POINT_GRASP_RENDER_EXECUTOR_ID_V1 = (
    "control-station.real-passive-station-point-grasp-render.v1"
)
REAL_STATION_POINT_GRASP_RENDER_OUTPUT_SLOT_ID_V1 = "rendered_grasp_candidate"


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


@dataclass(frozen=True)
class StationBoundPointGraspRenderProfile:
    """Exact candidate source and camera geometry allowed by the render Route."""

    render_rgb_sources: tuple[tuple[str, str, int, int], ...]
    camera_geometry_binding_hash: str
    point_grasp_route_hash: str
    point_grasp_profile_hash: str
    point_grasp_deployment_binding_hash: str
    source_binding_hash: str
    frame_binding_hash: str
    frame_graph_hash: str
    source_frame_id: str
    source_frame_spec_hash: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    max_candidates: int
    max_annotated_bytes: int
    profile_id: str = "real-station-point-grasp-candidate-render-v1"

    def __post_init__(self) -> None:
        for field_name in (
            "point_grasp_route_hash",
            "point_grasp_profile_hash",
            "point_grasp_deployment_binding_hash",
            "source_binding_hash",
            "frame_binding_hash",
            "frame_graph_hash",
            "source_frame_id",
            "source_frame_spec_hash",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "profile_id",
            "camera_geometry_binding_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        sources = tuple(sorted(self.render_rgb_sources, key=lambda item: item[1]))
        if not sources:
            raise ValueError(
                "At least one RGB source with trusted geometry is required."
            )
        if any(not isinstance(item, tuple) or len(item) != 4 for item in sources):
            raise ValueError("Render RGB source entries must be 4-tuples.")
        if len({item[0] for item in sources}) != len(sources):
            raise ValueError("Render RGB source IDs must be unique.")
        if len({item[1] for item in sources}) != len(sources):
            raise ValueError("Render RGB roles must be unique.")
        for source_id, role, width, height in sources:
            _required(source_id, field_name="source_id")
            _required(role, field_name="role")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (width, height)
            ):
                raise ValueError("Aligned RGB-D dimensions must be positive integers.")
        object.__setattr__(self, "render_rgb_sources", sources)
        if not 1 <= self.max_candidates <= 5:
            raise ValueError("max_candidates must be in [1, 5].")
        if not 1 <= self.max_annotated_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_annotated_bytes exceeds the reviewed bound.")

    @property
    def source_by_role(self) -> dict[str, str]:
        return {
            role: source_id
            for source_id, role, _width, _height in self.render_rgb_sources
        }

    @property
    def dimensions_by_role(self) -> dict[str, tuple[int, int]]:
        return {
            role: (width, height)
            for _source_id, role, width, height in self.render_rgb_sources
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "profile_id": self.profile_id,
                "render_rgb_sources": self.render_rgb_sources,
                "camera_geometry_binding_hash": self.camera_geometry_binding_hash,
                "point_grasp_route_hash": self.point_grasp_route_hash,
                "point_grasp_profile_hash": self.point_grasp_profile_hash,
                "point_grasp_deployment_binding_hash": (
                    self.point_grasp_deployment_binding_hash
                ),
                "source_binding_hash": self.source_binding_hash,
                "frame_binding_hash": self.frame_binding_hash,
                "frame_graph_hash": self.frame_graph_hash,
                "source_frame_id": self.source_frame_id,
                "source_frame_spec_hash": self.source_frame_spec_hash,
                "reference_frame_id": self.reference_frame_id,
                "reference_frame_spec_hash": self.reference_frame_spec_hash,
                "max_candidates": self.max_candidates,
                "max_annotated_bytes": self.max_annotated_bytes,
                "candidate_source": "retained_completed_station_point_grasp_operation",
                "render_count": 1,
                "view_geometry_owner": "deployment_rgb_camera_geometry_binding",
                "downstream_control_effects": "forbidden",
            },
            domain="dexteleop.agent_skill.station_point_grasp_render_profile.v1",
        )


@dataclass(frozen=True)
class StationBoundPointGraspRenderBinding:
    deployment_binding_id: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    profile_hash: str
    executor_id: str = REAL_STATION_POINT_GRASP_RENDER_EXECUTOR_ID_V1
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
                "candidate_owner": "station_point_grasp_executor",
                "view_geometry_owner": "deployment_aligned_rgbd_binding",
                "downstream_control_effects": "forbidden",
            },
            domain="dexteleop.agent_skill.station_point_grasp_render_deployment.v1",
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
class StationBoundPointGraspRenderRegistration:
    profile: StationBoundPointGraspRenderProfile
    skill: SkillRegistrationTemplate
    binding: StationBoundPointGraspRenderBinding
    payload_validator: AgentSkillPayloadValidatorSet
    availability_hash: str


def build_station_bound_point_grasp_render_registration(
    *,
    point_grasp_registration: StationBoundPointGraspRegistration,
    render_rgb_source_catalog: tuple[Mapping[str, object], ...] | None = None,
    camera_geometry_binding_hash: str | None = None,
) -> StationBoundPointGraspRenderRegistration:
    """Build the optional render Route from the exact proposal Route binding."""

    if not isinstance(point_grasp_registration, StationBoundPointGraspRegistration):
        raise TypeError("point_grasp_registration must be typed.")
    source_profile = point_grasp_registration.profile
    source_binding = point_grasp_registration.binding
    source_route = point_grasp_registration.skill.routes[0]
    selected_catalog = (
        tuple(
            {
                "source_id": source_id,
                "role": role,
                "native_width": width,
                "native_height": height,
            }
            for source_id, role, width, height in source_profile.aligned_rgbd_sources
        )
        if render_rgb_source_catalog is None
        else tuple(render_rgb_source_catalog)
    )
    render_sources = tuple(
        (
            str(item["source_id"]),
            str(item["role"]),
            int(item["native_width"]),
            int(item["native_height"]),
        )
        for item in selected_catalog
    )
    profile = StationBoundPointGraspRenderProfile(
        render_rgb_sources=render_sources,
        camera_geometry_binding_hash=(
            source_profile.source_binding_hash
            if camera_geometry_binding_hash is None
            else camera_geometry_binding_hash
        ),
        point_grasp_route_hash=source_route.semantic_content_hash,
        point_grasp_profile_hash=source_profile.semantic_content_hash,
        point_grasp_deployment_binding_hash=source_binding.semantic_content_hash,
        source_binding_hash=source_profile.source_binding_hash,
        frame_binding_hash=source_profile.frame_binding_hash,
        frame_graph_hash=source_profile.frame_graph_hash,
        source_frame_id=source_profile.source_frame_id,
        source_frame_spec_hash=source_profile.source_frame_spec_hash,
        reference_frame_id=source_profile.reference_frame_id,
        reference_frame_spec_hash=source_profile.reference_frame_spec_hash,
        max_candidates=source_profile.max_candidates,
        max_annotated_bytes=source_profile.max_annotated_bytes,
    )
    definition = render_point_grasp_candidate_skill_definition_v1()
    schema = point_grasp_candidate_render_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        roles=tuple(profile.source_by_role),
        max_candidates=profile.max_candidates,
    )
    io_contract = SkillIOContract(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        input_artifact_slots=(),
        output_artifact_slots=(
            SkillArtifactSlotSpec(
                slot_id=REAL_STATION_POINT_GRASP_RENDER_OUTPUT_SLOT_ID_V1,
                direction="output",
                min_count=1,
                max_count=1,
                allowed_media_types=(LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,),
                allowed_schema_ids=(POINT_GRASP_ANNOTATED_SCHEMA_ID_V1,),
                allowed_encodings=(LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,),
                allowed_compressions=(LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,),
                max_artifact_bytes=profile.max_annotated_bytes,
                max_total_bytes=profile.max_annotated_bytes,
            ),
        ),
    )
    route = SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=REAL_STATION_POINT_GRASP_RENDER_ROUTE_ID_V1,
        route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=source_route.supported_modes,
        target_kind="real",
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=30_000,
        cancellation_policy="cancel_only",
        requires_task_evaluation=False,
        skill_io_contract_hash=io_contract.semantic_content_hash,
    )
    skill = SkillRegistrationTemplate(
        definition=definition,
        schema_entry=schema,
        io_contract=io_contract,
        routes=(route,),
    )
    binding = StationBoundPointGraspRenderBinding(
        deployment_binding_id=(
            "real.station.point_grasp_candidate_render:"
            + source_binding.runtime_session_id
        ),
        route_hash=route.semantic_content_hash,
        runtime_session_id=source_binding.runtime_session_id,
        control_domain_id=source_binding.control_domain_id,
        target_binding_id=source_binding.target_binding_id,
        station_manifest_hash=source_binding.station_manifest_hash,
        execution_manifest_hash=source_binding.execution_manifest_hash,
        profile_hash=profile.semantic_content_hash,
    )
    validator = AgentSkillPayloadValidatorSet(
        validator_id=(
            "dexteleop.agent_skill.validator.station_point_grasp_render:"
            + source_binding.runtime_session_id
        ),
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=lambda payload: (
            validate_point_grasp_candidate_render_parameters_v1(
                payload,
                roles=tuple(profile.source_by_role),
                max_candidates=profile.max_candidates,
            )
        ),
        validate_execution_result=lambda payload: (
            validate_point_grasp_candidate_render_result_v1(
                payload,
                source_by_role=profile.source_by_role,
                dimensions_by_role=profile.dimensions_by_role,
                source_binding_hash=profile.source_binding_hash,
                camera_geometry_binding_hash=(profile.camera_geometry_binding_hash),
                frame_binding_hash=profile.frame_binding_hash,
                frame_graph_hash=profile.frame_graph_hash,
                source_frame_id=profile.source_frame_id,
                source_frame_spec_hash=profile.source_frame_spec_hash,
                reference_frame_id=profile.reference_frame_id,
                reference_frame_spec_hash=profile.reference_frame_spec_hash,
                max_candidates=profile.max_candidates,
            )
        ),
        validate_execution_result_for_parameters=(
            validate_point_grasp_candidate_render_result_for_parameters_v1
        ),
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    availability_hash = canonical_content_hash(
        {
            "status": "configured_requires_retained_point_grasp_and_live_rgb_geometry",
            "route_hash": route.semantic_content_hash,
            "deployment_binding_hash": binding.semantic_content_hash,
            "profile_hash": profile.semantic_content_hash,
        },
        domain="dexteleop.agent_skill.station_point_grasp_render_availability.v1",
    )
    return StationBoundPointGraspRenderRegistration(
        profile=profile,
        skill=skill,
        binding=binding,
        payload_validator=validator,
        availability_hash=availability_hash,
    )


__all__ = [
    "REAL_STATION_POINT_GRASP_RENDER_EXECUTOR_ID_V1",
    "REAL_STATION_POINT_GRASP_RENDER_OUTPUT_SLOT_ID_V1",
    "REAL_STATION_POINT_GRASP_RENDER_ROUTE_ID_V1",
    "StationBoundPointGraspRenderBinding",
    "StationBoundPointGraspRenderProfile",
    "StationBoundPointGraspRenderRegistration",
    "build_station_bound_point_grasp_render_registration",
]
