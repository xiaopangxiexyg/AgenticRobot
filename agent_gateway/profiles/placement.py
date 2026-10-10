"""Shared correlated RGB-D placement geometry registration and executor."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from agent_gateway.profiles.back_projection import (
    BackProjectionRegistration,
    CorrelatedPlacementCapturePort,
    LiveAlignedDepthCapture,
    ProjectedDepthGeometry,
    project_depth_mask_geometry,
)
from agent_gateway.profiles.segmentation import (
    SEGMENT_MASK_OUTPUT_SLOT_ID_V1,
    LiveSegmentationProvider,
    SegmentationRegistration,
    decode_segmentation_mask_npy,
)
from agent_gateway.protocol.contracts import AgentSkillOperation
from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
)
from agent_gateway.protocol.skill_io_contracts import SkillIOContract
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.runtime.service import AgentGatewayOperationOutcome
from agent_gateway.skills.perception.definition import (
    plan_placement_skill_definition_v1,
)
from agent_gateway.skills.perception.placement_schemas import (
    plan_placement_schema_entry_v1,
    validate_plan_placement_parameters_v1,
    validate_plan_placement_result_for_parameters_v1,
    validate_plan_placement_result_v1,
)
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from dataset_contracts.canonical_encoding import canonical_content_hash

PLACEMENT_MINIMUM_VALID_MASK_POINTS_V1 = 10
PLACEMENT_MAXIMUM_CLEARANCE_M_V1 = 0.05
PLACEMENT_MAXIMUM_HELD_EEF_OFFSET_M_V1 = 0.20
PLACEMENT_HELD_EEF_BASE_ALLOWANCE_M_V1 = 0.04
PLACEMENT_HELD_EEF_EXTENT_SCALE_V1 = 0.75
PLACEMENT_MINIMUM_HELD_CLOSE_FRACTION_V1 = 0.05
PLACEMENT_MAX_OPERATION_DURATION_MS_V1 = 125_000


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


@dataclass(frozen=True)
class PlacementRobotStateCapture:
    component_id: str
    measured_at: float
    station_revision: int
    tool_tcp_reference_xyz_quat_xyzw: tuple[float, ...]
    gripper_close_fraction: float | None
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _required(self.component_id, field_name="component_id")
        if not math.isfinite(float(self.measured_at)):
            raise ValueError("measured_at must be finite.")
        if (
            isinstance(self.station_revision, bool)
            or not isinstance(self.station_revision, int)
            or self.station_revision < 0
        ):
            raise ValueError("station_revision must be non-negative.")
        pose = tuple(float(value) for value in self.tool_tcp_reference_xyz_quat_xyzw)
        if len(pose) != 7 or not all(math.isfinite(value) for value in pose):
            raise ValueError("Tool TCP pose must contain seven finite values.")
        if not math.isclose(
            math.sqrt(sum(value * value for value in pose[3:])),
            1.0,
            abs_tol=1e-5,
        ):
            raise ValueError("Tool TCP quaternion must be normalized.")
        close = self.gripper_close_fraction
        if close is not None and (
            not math.isfinite(float(close)) or not 0.0 <= float(close) <= 1.0
        ):
            raise ValueError("Gripper close fraction must lie in [0, 1].")
        refs = tuple(
            sorted(
                {
                    _required(item, field_name="evidence_ref")
                    for item in self.evidence_refs
                }
            )
        )
        if not refs:
            raise ValueError("Placement robot state requires evidence.")
        object.__setattr__(self, "tool_tcp_reference_xyz_quat_xyzw", pose)
        object.__setattr__(
            self,
            "gripper_close_fraction",
            None if close is None else float(close),
        )
        object.__setattr__(self, "evidence_refs", refs)


class PlacementRobotStatePort(Protocol):
    def capture_placement_robot_state(
        self, *, expected_station_revision: int
    ) -> PlacementRobotStateCapture: ...


@dataclass(frozen=True)
class PlacementProfile:
    target_kind: str
    current_mode: str
    component_id: str
    destination_role: str
    held_roles: tuple[str, ...]
    reference_frame_id: str
    reference_frame_spec_hash: str
    frame_graph_hash: str
    segmentation_route_hash: str
    segmentation_profile_hash: str
    back_projection_route_hash: str
    back_projection_profile_hash: str
    minimum_valid_mask_points: int = PLACEMENT_MINIMUM_VALID_MASK_POINTS_V1
    maximum_clearance_m: float = PLACEMENT_MAXIMUM_CLEARANCE_M_V1
    maximum_depth_m: float = 10.0
    maximum_held_eef_offset_m: float = PLACEMENT_MAXIMUM_HELD_EEF_OFFSET_M_V1
    held_eef_base_allowance_m: float = PLACEMENT_HELD_EEF_BASE_ALLOWANCE_M_V1
    held_eef_extent_scale: float = PLACEMENT_HELD_EEF_EXTENT_SCALE_V1
    minimum_held_close_fraction: float = PLACEMENT_MINIMUM_HELD_CLOSE_FRACTION_V1
    profile_id: str = "correlated-rgbd-placement-v1"

    def __post_init__(self) -> None:
        if self.target_kind not in {"real", "sim"}:
            raise ValueError("Placement target_kind is unsupported.")
        if self.current_mode not in {"real_active", "sim_active"}:
            raise ValueError("Placement current_mode is unsupported.")
        if (self.target_kind == "real") != self.current_mode.startswith("real_"):
            raise ValueError("Placement target and mode disagree.")
        for name in (
            "component_id",
            "destination_role",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "frame_graph_hash",
            "segmentation_route_hash",
            "segmentation_profile_hash",
            "back_projection_route_hash",
            "back_projection_profile_hash",
            "profile_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), field_name=name),
            )
        roles = tuple(
            sorted(
                {
                    _required(role, field_name="held_role")
                    for role in self.held_roles
                }
            )
        )
        if not roles or self.destination_role not in roles:
            raise ValueError("Placement roles must include the destination role.")
        object.__setattr__(self, "held_roles", roles)
        if (
            isinstance(self.minimum_valid_mask_points, bool)
            or not isinstance(self.minimum_valid_mask_points, int)
            or self.minimum_valid_mask_points <= 0
        ):
            raise ValueError("minimum_valid_mask_points must be positive.")
        for name in (
            "maximum_clearance_m",
            "maximum_depth_m",
            "maximum_held_eef_offset_m",
            "held_eef_base_allowance_m",
            "held_eef_extent_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if not 0.0 <= self.minimum_held_close_fraction <= 1.0:
            raise ValueError("minimum_held_close_fraction must lie in [0, 1].")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
            },
            domain="dexteleop.agent_skill.placement_profile.v1",
        )


@dataclass(frozen=True)
class PlacementDeploymentBinding:
    deployment_binding_id: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    profile_hash: str
    target_kind: str
    executor_id: str
    deployment_binding_version: str = "1"

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), field_name=name),
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {name: getattr(self, name) for name in self.__dataclass_fields__},
            domain="dexteleop.agent_skill.placement_deployment.v1",
        )

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=self.deployment_binding_id,
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.target_binding_id,
            target_kind=self.target_kind,
            execution_manifest_hash=self.execution_manifest_hash,
        )


@dataclass(frozen=True)
class PlacementRegistration:
    profile: PlacementProfile
    skill: SkillRegistrationTemplate
    binding: PlacementDeploymentBinding
    payload_validator: AgentSkillPayloadValidatorSet
    availability_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile, PlacementProfile):
            raise TypeError("profile must be a PlacementProfile.")
        if not isinstance(self.skill, SkillRegistrationTemplate):
            raise TypeError("skill must be a SkillRegistrationTemplate.")
        if not isinstance(self.binding, PlacementDeploymentBinding):
            raise TypeError("binding must be a PlacementDeploymentBinding.")
        if not isinstance(self.payload_validator, AgentSkillPayloadValidatorSet):
            raise TypeError("payload_validator must be typed.")
        if len(self.skill.routes) != 1:
            raise ValueError("Placement registration requires exactly one Route.")
        route = self.skill.routes[0]
        definition = plan_placement_skill_definition_v1()
        if self.skill.definition != definition or (
            route.skill_id,
            route.skill_version,
            route.control_effect_class,
            route.privilege_tier,
            route.may_move_robot,
            route.may_open_publication,
            route.requires_operator_confirmation,
            route.requires_deadman,
            route.max_operation_duration_ms,
            route.cancellation_policy,
            route.requires_task_evaluation,
        ) != (
            definition.skill_id,
            definition.skill_version,
            "read_only",
            "t0_read_only",
            False,
            False,
            False,
            False,
            PLACEMENT_MAX_OPERATION_DURATION_MS_V1,
            "cancel_only",
            False,
        ):
            raise ValueError("Placement registration uses another Skill contract.")
        profile_refs = route.allowed_parameter_profiles
        if len(profile_refs) != 1 or (
            profile_refs[0].parameter_profile_id != self.profile.profile_id
            or profile_refs[0].parameter_profile_hash
            != self.profile.semantic_content_hash
        ):
            raise ValueError("Placement Route does not bind its exact profile.")
        if (
            route.target_kind != self.profile.target_kind
            or route.supported_modes != (self.profile.current_mode,)
            or self.binding.route_hash != route.semantic_content_hash
            or self.binding.profile_hash != self.profile.semantic_content_hash
            or self.binding.target_kind != self.profile.target_kind
        ):
            raise ValueError("Placement registration contracts disagree.")
        self.payload_validator.validate_route(route)
        expected_availability_hash = canonical_content_hash(
            {
                "status": "configured_requires_retained_text_segments_correlated_rgbd_and_tcp_state",
                "route_hash": route.semantic_content_hash,
                "deployment_binding_hash": self.binding.semantic_content_hash,
                "profile_hash": self.profile.semantic_content_hash,
            },
            domain="dexteleop.agent_skill.placement_availability.v1",
        )
        if self.availability_hash != expected_availability_hash:
            raise ValueError("Placement availability changed its contracts.")


def build_placement_registration(
    *,
    segmentation: SegmentationRegistration,
    back_projection: BackProjectionRegistration,
    component_id: str,
    destination_role: str,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
) -> PlacementRegistration:
    if not isinstance(segmentation, SegmentationRegistration):
        raise TypeError("segmentation must be typed.")
    if not isinstance(back_projection, BackProjectionRegistration):
        raise TypeError("back_projection must be typed.")
    if (
        segmentation.profile.target_kind != back_projection.profile.target_kind
        or segmentation.profile.current_mode != back_projection.profile.current_mode
        or segmentation.binding.runtime_session_id != runtime_session_id
        or back_projection.binding.runtime_session_id != runtime_session_id
        or segmentation.binding.control_domain_id != control_domain_id
        or back_projection.binding.control_domain_id != control_domain_id
        or segmentation.binding.target_binding_id != target_binding_id
        or back_projection.binding.target_binding_id != target_binding_id
        or segmentation.binding.station_manifest_hash != station_manifest_hash
        or back_projection.binding.station_manifest_hash != station_manifest_hash
        or segmentation.binding.execution_manifest_hash != execution_manifest_hash
        or back_projection.binding.execution_manifest_hash != execution_manifest_hash
    ):
        raise ValueError("Placement source registrations belong to another deployment.")
    roles = tuple(
        sorted(
            set(segmentation.profile.roles)
            & set(back_projection.profile.roles)
        )
    )
    profile = PlacementProfile(
        target_kind=back_projection.profile.target_kind,
        current_mode=back_projection.profile.current_mode,
        component_id=component_id,
        destination_role=destination_role,
        held_roles=roles,
        reference_frame_id=back_projection.profile.reference_frame_id,
        reference_frame_spec_hash=back_projection.profile.reference_frame_spec_hash,
        frame_graph_hash=back_projection.profile.frame_graph_hash,
        segmentation_route_hash=segmentation.skill.routes[0].semantic_content_hash,
        segmentation_profile_hash=segmentation.profile.semantic_content_hash,
        back_projection_route_hash=(
            back_projection.skill.routes[0].semantic_content_hash
        ),
        back_projection_profile_hash=back_projection.profile.semantic_content_hash,
        maximum_depth_m=back_projection.profile.maximum_depth_m,
    )
    definition = plan_placement_skill_definition_v1()
    schema = plan_placement_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        component_id=profile.component_id,
        reference_frame_id=profile.reference_frame_id,
        reference_frame_spec_hash=profile.reference_frame_spec_hash,
        frame_graph_hash=profile.frame_graph_hash,
        maximum_clearance_m=profile.maximum_clearance_m,
    )
    io_contract = SkillIOContract(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
    )
    route = SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=f"{profile.target_kind}.perception.plan_placement",
        route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=(profile.current_mode,),
        target_kind=profile.target_kind,
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=PLACEMENT_MAX_OPERATION_DURATION_MS_V1,
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
    binding = PlacementDeploymentBinding(
        deployment_binding_id=f"{profile.target_kind}.placement:{runtime_session_id}",
        route_hash=route.semantic_content_hash,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        target_binding_id=target_binding_id,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        profile_hash=profile.semantic_content_hash,
        target_kind=profile.target_kind,
        executor_id=f"{profile.target_kind}-passive-placement-plan.v1",
    )
    validator = AgentSkillPayloadValidatorSet(
        validator_id=(
            "dexteleop.agent_skill.validator."
            f"{profile.target_kind}_placement:{runtime_session_id}"
        ),
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=lambda payload: validate_plan_placement_parameters_v1(
            payload,
            maximum_clearance_m=profile.maximum_clearance_m,
        ),
        validate_execution_result=lambda payload: validate_plan_placement_result_v1(
            payload,
            component_id=profile.component_id,
            reference_frame_id=profile.reference_frame_id,
            reference_frame_spec_hash=profile.reference_frame_spec_hash,
            frame_graph_hash=profile.frame_graph_hash,
            maximum_clearance_m=profile.maximum_clearance_m,
            target_kind=profile.target_kind,
            minimum_held_close_fraction=profile.minimum_held_close_fraction,
        ),
        validate_execution_result_for_parameters=(
            validate_plan_placement_result_for_parameters_v1
        ),
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    availability_hash = canonical_content_hash(
        {
            "status": "configured_requires_retained_text_segments_correlated_rgbd_and_tcp_state",
            "route_hash": route.semantic_content_hash,
            "deployment_binding_hash": binding.semantic_content_hash,
            "profile_hash": profile.semantic_content_hash,
        },
        domain="dexteleop.agent_skill.placement_availability.v1",
    )
    return PlacementRegistration(
        profile=profile,
        skill=skill,
        binding=binding,
        payload_validator=validator,
        availability_hash=availability_hash,
    )


class PlacementProvider:
    def __init__(
        self,
        *,
        registration: PlacementRegistration,
        segmentation_registration: SegmentationRegistration,
        segmentation_executor: PassiveReadSkillExecutor,
        segmentation_provider: LiveSegmentationProvider,
        back_projection_registration: BackProjectionRegistration,
        capture_port: CorrelatedPlacementCapturePort,
        robot_state_port: PlacementRobotStatePort,
    ) -> None:
        if not isinstance(registration, PlacementRegistration):
            raise TypeError("registration must be typed.")
        if not isinstance(segmentation_registration, SegmentationRegistration):
            raise TypeError("segmentation_registration must be typed.")
        if not isinstance(segmentation_executor, PassiveReadSkillExecutor):
            raise TypeError("segmentation_executor must be passive.")
        if not isinstance(segmentation_provider, LiveSegmentationProvider):
            raise TypeError("segmentation_provider must be typed.")
        if not isinstance(back_projection_registration, BackProjectionRegistration):
            raise TypeError("back_projection_registration must be typed.")
        if not callable(getattr(capture_port, "capture_placement_rgbd", None)):
            raise TypeError("capture_port must expose capture_placement_rgbd().")
        if not callable(
            getattr(robot_state_port, "capture_placement_robot_state", None)
        ):
            raise TypeError("robot_state_port must expose placement state capture.")
        profile = registration.profile
        expected_source_binding = (
            registration.binding.runtime_session_id,
            registration.binding.control_domain_id,
            registration.binding.target_binding_id,
            registration.binding.station_manifest_hash,
            registration.binding.execution_manifest_hash,
        )
        source_bindings = (
            segmentation_registration.binding,
            back_projection_registration.binding,
        )
        if (
            profile.segmentation_route_hash
            != segmentation_registration.skill.routes[0].semantic_content_hash
            or profile.segmentation_profile_hash
            != segmentation_registration.profile.semantic_content_hash
            or profile.back_projection_route_hash
            != back_projection_registration.skill.routes[0].semantic_content_hash
            or profile.back_projection_profile_hash
            != back_projection_registration.profile.semantic_content_hash
            or segmentation_executor.executor_id
            != segmentation_registration.binding.executor_id
            or segmentation_provider.registration is not segmentation_registration
            or any(
                (
                    binding.runtime_session_id,
                    binding.control_domain_id,
                    binding.target_binding_id,
                    binding.station_manifest_hash,
                    binding.execution_manifest_hash,
                )
                != expected_source_binding
                for binding in source_bindings
            )
        ):
            raise ValueError("Placement provider uses another bound source contract.")
        self.registration = registration
        self.segmentation_registration = segmentation_registration
        self.segmentation_executor = segmentation_executor
        self.segmentation_provider = segmentation_provider
        self.back_projection_registration = back_projection_registration
        self.capture_port = capture_port
        self.robot_state_port = robot_state_port

    def _retained_segment(
        self,
        operation_id: str,
        *,
        expected_station_revision: int,
    ) -> tuple[AgentSkillOperation, Mapping[str, Any]]:
        completion = self.segmentation_executor.completion_for(operation_id)
        if completion is None:
            raise ValueError("Segment operation is not retained by this session.")
        if not isinstance(completion, AgentGatewayOperationOutcome):
            raise TypeError("Segment completion must be a Gateway outcome.")
        operation = completion.operation
        if (
            not completion.accepted
            or not isinstance(operation, AgentSkillOperation)
            or operation.status != "completed"
            or operation.execution_outcome != "completed"
        ):
            raise ValueError("Retained segment operation did not complete.")
        if operation.runtime_session_id != self.registration.binding.runtime_session_id:
            raise ValueError("Segment operation belongs to another runtime session.")
        result = operation.execution_result
        if not isinstance(result, Mapping):
            raise TypeError("Segment operation has no result mapping.")
        self.segmentation_registration.payload_validator.validate_execution_result(
            result
        )
        if (
            operation.station_revision_at_admission != expected_station_revision
            or result["station_revision"] != expected_station_revision
        ):
            raise ValueError("Segment operation belongs to another Station revision.")
        if result["mode"] != "text":
            raise ValueError("Placement requires text-prompt segment identity anchors.")
        bindings = tuple(
            item
            for item in operation.output_artifact_bindings
            if item.slot_id == SEGMENT_MASK_OUTPUT_SLOT_ID_V1 and item.ordinal == 0
        )
        if len(bindings) != 1:
            raise ValueError("Segment operation has no unique mask output.")
        artifact = bindings[0].artifact
        if artifact.sha256 != result["mask_media_content_hash"]:
            raise ValueError("Segment mask Artifact differs from its result.")
        return operation, result

    def _project(
        self,
        *,
        segment: Mapping[str, Any],
        expected_station_revision: int,
    ) -> tuple[LiveAlignedDepthCapture, ProjectedDepthGeometry, Mapping[str, Any]]:
        role = str(segment["role"])
        correlated = self.capture_port.capture_placement_rgbd(
            role=role,
            expected_station_revision=expected_station_revision,
        )
        depth = correlated.depth
        profile = self.back_projection_registration.profile
        if (
            depth.source_id != profile.source_id_by_role.get(role)
            or depth.station_revision != expected_station_revision
            or depth.depth_m.shape[::-1] != profile.dimensions_by_role.get(role)
        ):
            raise RuntimeError("Placement capture changed its bound depth identity.")
        refreshed = self.segmentation_provider.segment_capture(
            capture=correlated.rgb,
            parameters={
                "role": role,
                "language_prompt": segment["language_prompt"],
                "target_pixel_uv": None,
                "min_score": segment["min_score"],
            },
        )
        refreshed_result = refreshed.execution_result
        mask = decode_segmentation_mask_npy(
            refreshed.mask_npy,
            width=int(refreshed_result["image_width"]),
            height=int(refreshed_result["image_height"]),
            expected_count=int(refreshed_result["mask_pixel_count"]),
            expected_bounds=tuple(
                int(value) for value in refreshed_result["mask_bounds_xyxy"]
            ),
        )
        geometry = project_depth_mask_geometry(
            depth,
            mask,
            minimum_valid_points=self.registration.profile.minimum_valid_mask_points,
            maximum_depth_m=self.registration.profile.maximum_depth_m,
        )
        return depth, geometry, refreshed_result

    @staticmethod
    def _clipped(result: Mapping[str, Any]) -> bool:
        x0, y0, x1, y1 = (int(value) for value in result["mask_bounds_xyxy"])
        return bool(
            x0 == 0
            or y0 == 0
            or x1 == int(result["image_width"])
            or y1 == int(result["image_height"])
        )

    def plan_request(self, request: Any) -> tuple[dict[str, Any], tuple[str, ...]]:
        parameters = request.parameters
        revision = request.expected_station_revision
        profile = self.registration.profile
        held_operation, held_segment = self._retained_segment(
            str(parameters["held_segment_operation_id"]),
            expected_station_revision=revision,
        )
        destination_operation, destination_segment = (
            self._retained_segment(
                str(parameters["destination_segment_operation_id"]),
                expected_station_revision=revision,
            )
        )
        held_role = str(held_segment["role"])
        destination_role = str(destination_segment["role"])
        if held_role not in profile.held_roles:
            raise ValueError("Held segment role lacks aligned depth geometry.")
        if destination_role != profile.destination_role:
            raise ValueError("Destination segment must use the deployment main role.")
        held_capture, held_geometry, held_current_segment = self._project(
            segment=held_segment,
            expected_station_revision=revision,
        )
        destination_capture, destination_geometry, destination_current_segment = (
            self._project(
                segment=destination_segment,
                expected_station_revision=revision,
            )
        )
        state = self.robot_state_port.capture_placement_robot_state(
            expected_station_revision=revision
        )
        if (
            state.component_id != profile.component_id
            or state.station_revision != revision
        ):
            raise RuntimeError("Placement robot state changed its deployment binding.")
        clearance = float(parameters["clearance_m"])
        held_center = np.asarray(held_geometry.world_center_xyz_m, dtype=np.float64)
        held_extent = np.asarray(held_geometry.world_extent_xyz_m, dtype=np.float64)
        destination_center = np.asarray(
            destination_geometry.world_center_xyz_m,
            dtype=np.float64,
        )
        destination_low = np.asarray(
            destination_geometry.world_bounds_low_xyz_m,
            dtype=np.float64,
        )
        destination_high = np.asarray(
            destination_geometry.world_bounds_high_xyz_m,
            dtype=np.float64,
        )
        eef = np.asarray(state.tool_tcp_reference_xyz_quat_xyzw[:3], dtype=np.float64)
        held_offset = held_center - eef
        held_offset_norm = float(np.linalg.norm(held_offset))
        held_extent_norm = float(np.linalg.norm(held_extent))
        held_offset_limit = min(
            profile.maximum_held_eef_offset_m,
            profile.held_eef_base_allowance_m
            + profile.held_eef_extent_scale * held_extent_norm,
        )
        held_clipped = self._clipped(held_current_segment)
        destination_provisional = self._clipped(destination_current_segment)
        close = state.gripper_close_fraction
        rejection = None
        if held_clipped:
            rejection = "held_segment_clipped"
        elif held_offset_norm > held_offset_limit:
            rejection = "held_segment_far_from_eef"
        elif close is None and profile.target_kind == "real":
            rejection = "held_gripper_state_unavailable"
        elif close is not None and close <= profile.minimum_held_close_fraction:
            rejection = "held_gripper_clearly_open"
        source = {
            "held_segment_operation_id": held_operation.operation_id,
            "destination_segment_operation_id": destination_operation.operation_id,
            "station_revision": revision,
            "component_id": profile.component_id,
            "reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "frame_graph_hash": profile.frame_graph_hash,
            "clearance_m": clearance,
            "held_gripper_close_fraction": close,
            "held_gripper_state_checked": close is not None,
            "destination_geometry_provisional": destination_provisional,
            "execution_requires_explicit_motion_skill": True,
            "motion_or_hold_performed": False,
        }
        evidence = tuple(
            sorted(
                {
                    *held_operation.execution_receipt_refs,
                    *destination_operation.execution_receipt_refs,
                    *state.evidence_refs,
                    "placement-held-segment:"
                    + canonical_content_hash(
                        held_current_segment,
                        domain="dexteleop.agent_gateway.passive_read_result.v1",
                    ),
                    "placement-destination-segment:"
                    + canonical_content_hash(
                        destination_current_segment,
                        domain="dexteleop.agent_gateway.passive_read_result.v1",
                    ),
                    "placement-held-capture:" + held_capture.capture_group_id,
                    "placement-destination-capture:"
                    + destination_capture.capture_group_id,
                    "placement-held-depth:" + held_capture.depth_content_hash,
                    "placement-destination-depth:"
                    + destination_capture.depth_content_hash,
                }
            )
        )
        if rejection is not None:
            return (
                {
                    **source,
                    "planning_status": "rejected",
                    "reason_code": rejection,
                    "target_eef_xy": None,
                    "minimum_lateral_eef_z": None,
                    "fits_with_clearance": None,
                    "geometry_eligible_for_motion_planning": False,
                },
                evidence,
            )
        target_eef_xy = destination_center[:2] - held_offset[:2]
        minimum_lateral_eef_z = destination_high[2] + clearance - held_offset[2]
        projected_low = destination_center[:2] - held_extent[:2] / 2.0
        projected_high = destination_center[:2] + held_extent[:2] / 2.0
        margins = np.minimum(
            projected_low - destination_low[:2] - clearance,
            destination_high[:2] - projected_high - clearance,
        )
        fits = bool(np.all(margins >= 0.0))

        def vector(values: np.ndarray) -> list[float]:
            return [round(float(value), 6) for value in values]

        eligible = fits and not destination_provisional
        reason = (
            "placement_geometry_planned"
            if eligible
            else (
                "destination_geometry_provisional"
                if destination_provisional
                else "placement_geometry_does_not_fit"
            )
        )
        return (
            {
                **source,
                "planning_status": "planned",
                "reason_code": reason,
                "target_eef_xy": vector(target_eef_xy),
                "minimum_lateral_eef_z": round(float(minimum_lateral_eef_z), 6),
                "fits_with_clearance": fits,
                "geometry_eligible_for_motion_planning": eligible,
            },
            evidence,
        )


def build_placement_executor(
    *,
    registration: PlacementRegistration,
    provider: PlacementProvider,
    current_station_revision: Callable[[], int],
    max_retained_operations: int,
    clock: Callable[[], float] = time.time,
) -> PassiveReadSkillExecutor:
    if provider.registration is not registration:
        raise ValueError("Placement provider and registration differ.")

    def read(request: Any, _operation: Any) -> PassiveReadResult:
        result, evidence = provider.plan_request(request)
        if current_station_revision() != request.expected_station_revision:
            raise RuntimeError("Station revision changed during placement planning.")
        return PassiveReadResult(
            execution_result=result,
            evidence_refs=evidence,
            reason_code="passive_placement_geometry_planned",
        )

    return PassiveReadSkillExecutor(
        port_id=(
            "agent-gateway:"
            f"{registration.profile.target_kind}-passive-placement:"
            f"{registration.binding.runtime_session_id}"
        ),
        executor_id=registration.binding.executor_id,
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )


__all__ = [
    "PlacementProfile",
    "PlacementProvider",
    "PlacementRegistration",
    "PlacementRobotStateCapture",
    "PlacementRobotStatePort",
    "build_placement_executor",
    "build_placement_registration",
]
