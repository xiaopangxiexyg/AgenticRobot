"""Pure read-only deployment contract for point-conditioned RGB-D grasps."""

from __future__ import annotations

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
    plan_point_grasp_skill_definition_v1,
)
from agent_gateway.skills.perception.point_grasp_schemas import (
    point_grasp_schema_entry_v1,
    validate_point_grasp_parameters_v1,
    validate_point_grasp_result_for_parameters_v1,
    validate_point_grasp_result_v1,
)
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.perception_artifacts import (
    LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,
    LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,
    LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
    LOCALIZATION_INPUT_IMAGE_COMPRESSION_V1,
    LOCALIZATION_INPUT_JPEG_ENCODING_V1,
    LOCALIZATION_INPUT_JPEG_MEDIA_TYPE_V1,
    LOCALIZATION_INPUT_PNG_ENCODING_V1,
    LOCALIZATION_INPUT_PNG_MEDIA_TYPE_V1,
    POINT_GRASP_ANNOTATED_SCHEMA_ID_V1,
    POINT_GRASP_DEPTH_COMPRESSION_V1,
    POINT_GRASP_DEPTH_ENCODING_V1,
    POINT_GRASP_DEPTH_MEDIA_TYPE_V1,
    POINT_GRASP_DEPTH_SCHEMA_ID_V1,
    POINT_GRASP_RGB_SCHEMA_ID_V1,
)

REAL_POINT_GRASP_ROUTE_ID_V1 = "real.local_model.perception.plan_point_grasp"
REAL_POINT_GRASP_EXECUTOR_ID_V1 = "real-passive-point-grasp-planner.v1"
REAL_POINT_GRASP_RGB_SLOT_ID_V1 = "rgb_image"
REAL_POINT_GRASP_DEPTH_SLOT_ID_V1 = "aligned_depth_m"
REAL_POINT_GRASP_OUTPUT_SLOT_ID_V1 = "annotated_grasps"


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


@dataclass(frozen=True)
class RealPointGraspProfile:
    backend_id: str
    model_profile_id: str
    model_profile_hash: str
    inference_implementation_hash: str
    local_path_enabled: bool
    max_candidates: int = 5
    max_rgb_bytes: int = 16 * 1024 * 1024
    max_depth_bytes: int = 16 * 1024 * 1024
    max_annotated_bytes: int = 16 * 1024 * 1024
    profile_id: str = "real-sam2-point-conditioned-dmtac-m-point-grasp-v1"

    def __post_init__(self) -> None:
        for field in (
            "backend_id",
            "model_profile_id",
            "model_profile_hash",
            "inference_implementation_hash",
            "profile_id",
        ):
            object.__setattr__(
                self, field, _required(getattr(self, field), field_name=field)
            )
        if self.backend_id not in {"vgn", "graspnet"}:
            raise ValueError("point-grasp backend_id is unsupported.")
        if not isinstance(self.local_path_enabled, bool):
            raise TypeError("local_path_enabled must be boolean.")
        if not 1 <= self.max_candidates <= 5:
            raise ValueError("max_candidates must be in [1, 5].")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "profile_id": self.profile_id,
                "backend_id": self.backend_id,
                "model_profile_id": self.model_profile_id,
                "model_profile_hash": self.model_profile_hash,
                "inference_implementation_hash": self.inference_implementation_hash,
                "local_path_test_input": "explicit_allowlisted_roots"
                if self.local_path_enabled
                else "disabled",
                "max_candidates": self.max_candidates,
                "gripper": "DMTAC_M",
                "input": "aligned_rgbd_meters_and_fixed_camera_transform",
                "output": "ranked_camera_and_base_tool_tcp_poses",
                "downstream_control_effects": "forbidden",
            },
            domain="dexteleop.agent_skill.real_point_grasp_profile.v1",
        )


@dataclass(frozen=True)
class RealPointGraspBinding:
    deployment_binding_id: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    profile_hash: str
    executor_id: str = REAL_POINT_GRASP_EXECUTOR_ID_V1
    deployment_binding_version: str = "1"

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            object.__setattr__(
                self, field, _required(getattr(self, field), field_name=field)
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {field: getattr(self, field) for field in self.__dataclass_fields__}
            | {
                "target_kind": "real",
                "control_effect_class": "read_only",
                "downstream_control_effects": "forbidden",
            },
            domain="dexteleop.agent_skill.real_point_grasp_deployment.v1",
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
class RealPointGraspRegistration:
    profile: RealPointGraspProfile
    skill: SkillRegistrationTemplate
    binding: RealPointGraspBinding
    payload_validator: AgentSkillPayloadValidatorSet
    availability_hash: str


def build_real_point_grasp_registration(
    *,
    backend_id: str,
    model_profile_id: str,
    model_profile_hash: str,
    inference_implementation_hash: str,
    local_path_enabled: bool,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
) -> RealPointGraspRegistration:
    profile = RealPointGraspProfile(
        backend_id=backend_id,
        model_profile_id=model_profile_id,
        model_profile_hash=model_profile_hash,
        inference_implementation_hash=inference_implementation_hash,
        local_path_enabled=local_path_enabled,
    )
    definition = plan_point_grasp_skill_definition_v1()
    schema = point_grasp_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        max_candidates=profile.max_candidates,
        backend_id=profile.backend_id,
    )
    io_contract = SkillIOContract(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        input_artifact_slots=(
            SkillArtifactSlotSpec(
                slot_id=REAL_POINT_GRASP_DEPTH_SLOT_ID_V1,
                direction="input",
                min_count=0,
                max_count=1,
                allowed_media_types=(POINT_GRASP_DEPTH_MEDIA_TYPE_V1,),
                allowed_schema_ids=(POINT_GRASP_DEPTH_SCHEMA_ID_V1,),
                allowed_encodings=(POINT_GRASP_DEPTH_ENCODING_V1,),
                allowed_compressions=(POINT_GRASP_DEPTH_COMPRESSION_V1,),
                max_artifact_bytes=profile.max_depth_bytes,
                max_total_bytes=profile.max_depth_bytes,
            ),
            SkillArtifactSlotSpec(
                slot_id=REAL_POINT_GRASP_RGB_SLOT_ID_V1,
                direction="input",
                min_count=0,
                max_count=1,
                allowed_media_types=(
                    LOCALIZATION_INPUT_JPEG_MEDIA_TYPE_V1,
                    LOCALIZATION_INPUT_PNG_MEDIA_TYPE_V1,
                ),
                allowed_schema_ids=(POINT_GRASP_RGB_SCHEMA_ID_V1,),
                allowed_encodings=(
                    LOCALIZATION_INPUT_JPEG_ENCODING_V1,
                    LOCALIZATION_INPUT_PNG_ENCODING_V1,
                ),
                allowed_compressions=(LOCALIZATION_INPUT_IMAGE_COMPRESSION_V1,),
                max_artifact_bytes=profile.max_rgb_bytes,
                max_total_bytes=profile.max_rgb_bytes,
            ),
        ),
        output_artifact_slots=(
            SkillArtifactSlotSpec(
                slot_id=REAL_POINT_GRASP_OUTPUT_SLOT_ID_V1,
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
        skill_version="1",
        skill_definition_hash=definition.semantic_content_hash,
        route_id=REAL_POINT_GRASP_ROUTE_ID_V1,
        route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=("real_observe",),
        target_kind="real",
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=120_000,
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
    binding = RealPointGraspBinding(
        deployment_binding_id="real.local_model.point_grasp:" + runtime_session_id,
        route_hash=route.semantic_content_hash,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        target_binding_id=target_binding_id,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        profile_hash=profile.semantic_content_hash,
    )
    validator = AgentSkillPayloadValidatorSet(
        validator_id="dexteleop.agent_skill.validator.real_point_grasp:"
        + runtime_session_id,
        skill_id=definition.skill_id,
        skill_version="1",
        route_id=route.route_id,
        route_version="1",
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=lambda payload: validate_point_grasp_parameters_v1(
            payload,
            local_path_enabled=profile.local_path_enabled,
            max_candidates=profile.max_candidates,
        ),
        validate_execution_result=lambda payload: validate_point_grasp_result_v1(
            payload,
            backend_id=profile.backend_id,
            model_profile_id=profile.model_profile_id,
            model_profile_hash=profile.model_profile_hash,
            max_candidates=profile.max_candidates,
        ),
        validate_execution_result_for_parameters=validate_point_grasp_result_for_parameters_v1,
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    availability = canonical_content_hash(
        {
            "status": "configured_requires_verified_point_grasp_assets",
            "backend_id": profile.backend_id,
            "route_hash": route.semantic_content_hash,
            "binding_hash": binding.semantic_content_hash,
        },
        domain="dexteleop.agent_skill.real_point_grasp_availability.v1",
    )
    return RealPointGraspRegistration(profile, skill, binding, validator, availability)


__all__ = [
    "REAL_POINT_GRASP_DEPTH_SLOT_ID_V1",
    "REAL_POINT_GRASP_EXECUTOR_ID_V1",
    "REAL_POINT_GRASP_OUTPUT_SLOT_ID_V1",
    "REAL_POINT_GRASP_RGB_SLOT_ID_V1",
    "REAL_POINT_GRASP_ROUTE_ID_V1",
    "RealPointGraspProfile",
    "RealPointGraspRegistration",
    "build_real_point_grasp_registration",
]
