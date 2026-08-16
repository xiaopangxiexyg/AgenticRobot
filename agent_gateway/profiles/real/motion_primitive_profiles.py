"""Exact real-Robot Routes and bindings for the reviewed 6+3 TCP primitives.

Construction is pure and default-off.  It requires an existing real-observe
attachment, lower motion-port binding, exact frame graph, and at least one
lower-owned safety-envelope reference.  No device, control loop, socket, or
motion operation is started here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaCatalog,
    SkillDefinitionCatalog,
    SkillDeploymentBindingRef,
    SkillDeploymentSafetyEnvelopeRef,
    SkillExecutionRoute,
    SkillRouteManifest,
    SkillRouteManifestEntry,
    validate_skill_route_catalog_closure,
)
from agent_gateway.runtime.payload_validation import (
    AgentSkillPayloadValidatorRegistry,
    AgentSkillPayloadValidatorSet,
)
from agent_gateway.skills.motion.definition import (
    FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1,
    TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_SKILL_ID_V1,
    cardinal_tcp_step_skill_definition_v1,
    fixed_level_tcp_orientation_skill_definition_v1,
)
from agent_gateway.skills.motion.schemas import (
    CARDINAL_TCP_STEP_DIRECTIONS_V1,
    cardinal_tcp_step_schema_entry_v1,
    fixed_level_tcp_orientation_schema_entry_v1,
    validate_agent_motion_parameters_v1,
    validate_agent_motion_route_schema_v1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.coordinate_frames import CoordinateFrameGraph
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.primitives import (
    CARDINAL_TCP_SHORT_STEP_PRIMITIVES,
    LevelTCPOrientationPrimitive,
    TCPPoseDisplacementPrimitive,
    cardinal_tcp_short_step,
    level_tcp_orientation,
)
from teleop_runtime.real_teleop.agent_motion_port import RealAgentMotionPortConfig

from .observation_gateway import RealTcpPoseObserveAttachmentSpec

REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1 = (
    "real.rm65.agent_skill_tcp_cardinal_step"
)
REAL_AGENT_TCP_CARDINAL_STEP_EXECUTOR_ID_V1 = (
    "control-station.real-agent-tcp-cardinal-step.v1"
)
REAL_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1 = (
    "real-tool-tcp-cardinal-short-step-v1"
)
REAL_AGENT_TCP_LEVEL_FORWARD_ROUTE_ID_V1 = (
    "real.rm65.agent_skill_tcp_level_forward"
)
REAL_AGENT_TCP_LEVEL_FORWARD_LEFT_30_ROUTE_ID_V1 = (
    "real.rm65.agent_skill_tcp_level_forward_left_30"
)
REAL_AGENT_TCP_LEVEL_FORWARD_RIGHT_30_ROUTE_ID_V1 = (
    "real.rm65.agent_skill_tcp_level_forward_right_30"
)
REAL_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1 = {
    TCP_LEVEL_FORWARD_SKILL_ID_V1: REAL_AGENT_TCP_LEVEL_FORWARD_ROUTE_ID_V1,
    TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1: (
        REAL_AGENT_TCP_LEVEL_FORWARD_LEFT_30_ROUTE_ID_V1
    ),
    TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1: (
        REAL_AGENT_TCP_LEVEL_FORWARD_RIGHT_30_ROUTE_ID_V1
    ),
}
REAL_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1 = {
    TCP_LEVEL_FORWARD_SKILL_ID_V1: "center_forward",
    TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1: "forward_left_30",
    TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1: "forward_right_30",
}
REAL_AGENT_TCP_FIXED_ORIENTATION_EXECUTOR_ID_BY_SKILL_ID_V1 = {
    skill_id: (
        "control-station.real-agent-"
        + skill_id.removeprefix("motion.").replace("_", "-")
        + ".v1"
    )
    for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
}
REAL_AGENT_TCP_FIXED_ORIENTATION_PARAMETER_PROFILE_ID_BY_SKILL_ID_V1 = {
    skill_id: (
        "real-tool-tcp-"
        + skill_id.removeprefix("motion.").replace("_", "-")
        + "-v1"
    )
    for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
}


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


@dataclass(frozen=True)
class RealAgentPrimitiveConvergenceProfile:
    translation_tolerance_m: float = 0.01
    rotation_tolerance_rad: float = 0.08726646259971647
    stable_measurement_count: int = 3
    max_measurement_age_ms: int = 200
    poll_interval_sec: float = 0.01

    def __post_init__(self) -> None:
        for field_name in (
            "translation_tolerance_m",
            "rotation_tolerance_rad",
            "poll_interval_sec",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not value > 0.0:
                raise ValueError(f"{field_name} must be positive.")
            object.__setattr__(self, field_name, value)
        for field_name in ("stable_measurement_count", "max_measurement_age_ms"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.__dict__,
            domain="dexteleop.real_agent_motion.convergence_profile.v1",
        )


@dataclass(frozen=True)
class RealAgentTcpCardinalStepProfile:
    component_id: str
    side: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    execution_reference_frame_id: str
    execution_reference_frame_spec_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    convergence_profile_hash: str
    primitives: tuple[TCPPoseDisplacementPrimitive, ...] = (
        CARDINAL_TCP_SHORT_STEP_PRIMITIVES
    )

    def __post_init__(self) -> None:
        for field_name in (
            "component_id",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "execution_reference_frame_id",
            "execution_reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "convergence_profile_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        side = str(self.side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right.")
        object.__setattr__(self, "side", side)
        primitives = tuple(self.primitives)
        if any(
            not isinstance(item, TCPPoseDisplacementPrimitive)
            for item in primitives
        ):
            raise TypeError("primitives must contain TCP displacement primitives.")
        directions = tuple(item.direction.value for item in primitives)
        if set(directions) != set(CARDINAL_TCP_STEP_DIRECTIONS_V1) or len(
            directions
        ) != len(set(directions)):
            raise ValueError("Profile requires one primitive per cardinal direction.")
        object.__setattr__(
            self,
            "primitives",
            tuple(sorted(primitives, key=lambda item: item.direction.value)),
        )

    @property
    def parameter_profile_id(self) -> str:
        return REAL_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1

    @property
    def primitive_catalog_hash(self) -> str:
        return canonical_content_hash(
            tuple(item.to_payload() for item in self.primitives),
            domain="dexteleop.robot_motion.tcp_displacement_catalog.v1",
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                **self.__dict__,
                "profile_id": self.parameter_profile_id,
                "controlled_frame_id": "tool_tcp",
                "primitive_catalog_hash": self.primitive_catalog_hash,
                "primitives": tuple(item.to_payload() for item in self.primitives),
                "anchor_semantics": "lower_measured_pose_at_skill_activation",
            },
            domain="dexteleop.agent_skill.real_tcp_cardinal_step_profile.v1",
        )

    def primitive(self, direction: str) -> TCPPoseDisplacementPrimitive:
        selected = cardinal_tcp_short_step(direction)
        for item in self.primitives:
            if item.semantic_content_hash == selected.semantic_content_hash:
                return item
        raise ValueError("Direction is absent from the exact primitive profile.")


@dataclass(frozen=True)
class RealAgentTcpFixedOrientationProfile:
    skill_id: str
    component_id: str
    side: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    execution_reference_frame_id: str
    execution_reference_frame_spec_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    convergence_profile_hash: str
    primitive: LevelTCPOrientationPrimitive

    def __post_init__(self) -> None:
        for field_name in (
            "skill_id",
            "component_id",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "execution_reference_frame_id",
            "execution_reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "convergence_profile_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if self.skill_id not in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1:
            raise ValueError("Unsupported fixed-orientation Skill ID.")
        side = str(self.side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right.")
        object.__setattr__(self, "side", side)
        if not isinstance(self.primitive, LevelTCPOrientationPrimitive):
            raise TypeError("primitive must be a fixed TCP orientation primitive.")
        expected = REAL_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1[
            self.skill_id
        ]
        if self.primitive.heading.value != expected:
            raise ValueError("Skill ID and fixed-orientation primitive disagree.")

    @property
    def parameter_profile_id(self) -> str:
        return REAL_AGENT_TCP_FIXED_ORIENTATION_PARAMETER_PROFILE_ID_BY_SKILL_ID_V1[
            self.skill_id
        ]

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                **self.__dict__,
                "profile_id": self.parameter_profile_id,
                "controlled_frame_id": "tool_tcp",
                "primitive": self.primitive.to_payload(),
                "anchor_semantics": "lower_measured_pose_at_skill_activation",
                "target_semantics": "absolute_heading_preserve_translation",
            },
            domain="dexteleop.agent_skill.real_tcp_fixed_orientation_profile.v1",
        )


@dataclass(frozen=True)
class RealAgentPrimitiveDeploymentBinding:
    deployment_binding_id: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    attachment_hash: str
    motion_port_binding_hash: str
    parameter_profile_hash: str
    convergence_profile_hash: str
    component_id: str
    side: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    execution_reference_frame_id: str
    execution_reference_frame_spec_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    executor_id: str
    safety_envelope_refs: tuple[SkillDeploymentSafetyEnvelopeRef, ...]
    deployment_binding_version: str = "1"

    def __post_init__(self) -> None:
        for field_name in (
            "deployment_binding_id",
            "route_hash",
            "runtime_session_id",
            "control_domain_id",
            "target_binding_id",
            "station_manifest_hash",
            "execution_manifest_hash",
            "attachment_hash",
            "motion_port_binding_hash",
            "parameter_profile_hash",
            "convergence_profile_hash",
            "component_id",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "execution_reference_frame_id",
            "execution_reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "executor_id",
            "deployment_binding_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        side = str(self.side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right.")
        object.__setattr__(self, "side", side)
        refs = tuple(self.safety_envelope_refs)
        if not refs or any(
            not isinstance(item, SkillDeploymentSafetyEnvelopeRef) for item in refs
        ):
            raise ValueError("Real motion binding requires typed safety-envelope refs.")
        object.__setattr__(
            self,
            "safety_envelope_refs",
            tuple(sorted(refs, key=lambda item: item.object_id)),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                **{
                    name: getattr(self, name)
                    for name in self.__dataclass_fields__
                    if name != "safety_envelope_refs"
                },
                "safety_envelope_refs": tuple(
                    item.to_wire() for item in self.safety_envelope_refs
                ),
                "target_kind": "real",
                "controlled_frame_id": "tool_tcp",
                "source_kind": "agent_skill_via_existing_realteleop_owner",
            },
            domain="dexteleop.agent_skill.real_tcp_primitive_deployment.v1",
        )

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=self.deployment_binding_id,
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.target_binding_id,
            target_kind="real",
            execution_manifest_hash=self.execution_manifest_hash,
            safety_envelope_refs=self.safety_envelope_refs,
        )


@dataclass(frozen=True)
class RealAgentPrimitiveContractBundle:
    definition_catalog: SkillDefinitionCatalog
    schema_catalog: AgentSkillSchemaCatalog
    route_manifest: SkillRouteManifest
    cardinal_profile: RealAgentTcpCardinalStepProfile
    orientation_profiles: tuple[RealAgentTcpFixedOrientationProfile, ...]
    deployment_bindings: tuple[RealAgentPrimitiveDeploymentBinding, ...]
    convergence_profile: RealAgentPrimitiveConvergenceProfile
    payload_validators: AgentSkillPayloadValidatorRegistry
    frame_graph: CoordinateFrameGraph
    attachment: RealTcpPoseObserveAttachmentSpec

    def __post_init__(self) -> None:
        validate_skill_route_catalog_closure(
            definition_catalog=self.definition_catalog,
            schema_catalog=self.schema_catalog,
            route_manifest=self.route_manifest,
        )
        self.payload_validators.validate_manifest(self.route_manifest)
        if (
            len(self.orientation_profiles) != 3
            or len(self.deployment_bindings) != 4
        ):
            raise ValueError(
                "Real primitive bundle requires one cardinal and three "
                "orientation bindings."
            )

    def binding_for_skill(self, skill_id: str) -> RealAgentPrimitiveDeploymentBinding:
        route = self.route_manifest.route(skill_id, "1").route
        for binding in self.deployment_bindings:
            if binding.route_hash == route.semantic_content_hash:
                return binding
        raise KeyError(skill_id)

    def orientation_profile(
        self,
        skill_id: str,
    ) -> RealAgentTcpFixedOrientationProfile:
        for profile in self.orientation_profiles:
            if profile.skill_id == skill_id:
                return profile
        raise KeyError(skill_id)


def _payload_validator(
    *,
    route: SkillExecutionRoute,
    profile: RealAgentTcpCardinalStepProfile | RealAgentTcpFixedOrientationProfile,
) -> AgentSkillPayloadValidatorSet:
    cardinal = isinstance(profile, RealAgentTcpCardinalStepProfile)
    parameter_fields = (
        {
            "component_id",
            "direction",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "controlled_frame_id",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "primitive_catalog_hash",
        }
        if cardinal
        else {
            "component_id",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "controlled_frame_id",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "primitive_hash",
        }
    )

    def validate_parameters(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != parameter_fields:
            raise ValueError("Real primitive parameters do not match the exact schema.")
        validation_kwargs: dict[str, Any] = {
            "schema_kind": (
                "cardinal_tcp_step"
                if cardinal
                else "fixed_level_tcp_orientation"
            ),
            "parameters": payload,
            "allowed_component_ids": (profile.component_id,),
        }
        if isinstance(profile, RealAgentTcpCardinalStepProfile):
            validation_kwargs["allowed_cardinal_directions"] = tuple(
                item.direction.value for item in profile.primitives
            )
        validate_agent_motion_parameters_v1(**validation_kwargs)
        exact = {
            "component_id": profile.component_id,
            "reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": profile.frame_graph_hash,
            "tool_config_hash": profile.tool_config_hash,
            "calibration_hash": profile.calibration_hash,
            (
                "primitive_catalog_hash" if cardinal else "primitive_hash"
            ): (
                profile.primitive_catalog_hash
                if cardinal
                else profile.primitive.semantic_content_hash
            ),
        }
        for field_name, expected in exact.items():
            if payload[field_name] != expected:
                raise ValueError(f"Real primitive {field_name} mismatch.")
        if cardinal:
            profile.primitive(str(payload["direction"]))

    def validate_execution_result(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != {
            "execution_receipt_refs",
            "final_measured_state_hash",
        }:
            raise ValueError("Real primitive result schema mismatch.")
        refs = payload["execution_receipt_refs"]
        if (
            not isinstance(refs, (tuple, list))
            or not refs
            or len(refs) != len(set(refs))
            or any(not isinstance(item, str) or not item for item in refs)
        ):
            raise ValueError("execution_receipt_refs must be unique and non-empty.")
        _required(
            payload["final_measured_state_hash"],
            field_name="final_measured_state_hash",
        )

    def validate_task_evaluation(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != {
            "aggregate_evaluation_receipt_ref",
            "task_outcome",
        }:
            raise ValueError("Real primitive task-evaluation schema mismatch.")
        _required(
            payload["aggregate_evaluation_receipt_ref"],
            field_name="aggregate_evaluation_receipt_ref",
        )
        if payload["task_outcome"] not in {"failed", "inconclusive", "succeeded"}:
            raise ValueError("Unsupported real primitive task_outcome.")

    return AgentSkillPayloadValidatorSet(
        validator_id=(
            "payload-validator:real-agent-primitive:"
            + profile.semantic_content_hash
        ),
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=validate_parameters,
        validate_execution_result=validate_execution_result,
        validate_task_evaluation=validate_task_evaluation,
    )


def build_real_agent_primitive_contract_bundle(
    *,
    attachment: RealTcpPoseObserveAttachmentSpec,
    port_config: RealAgentMotionPortConfig,
    frame_graph: CoordinateFrameGraph,
    primitive_reference_frame_id: str,
    safety_envelope_refs: tuple[SkillDeploymentSafetyEnvelopeRef, ...],
    convergence_profile: RealAgentPrimitiveConvergenceProfile | None = None,
) -> RealAgentPrimitiveContractBundle:
    """Build exact T2 Routes; all runtime effects remain unattached."""

    if not isinstance(attachment, RealTcpPoseObserveAttachmentSpec):
        raise TypeError("attachment must be typed.")
    if not isinstance(port_config, RealAgentMotionPortConfig):
        raise TypeError("port_config must be typed.")
    if not isinstance(frame_graph, CoordinateFrameGraph):
        raise TypeError("frame_graph must be typed.")
    refs = tuple(safety_envelope_refs)
    if not refs:
        raise ValueError("Real primitive Routes require lower safety-envelope refs.")
    if attachment.frame_graph_hash != frame_graph.semantic_content_hash:
        raise ValueError("Attachment and primitive frame graph disagree.")
    if attachment.reference_frame_id != port_config.execution_frame_id:
        raise ValueError("Real observation must publish in the lower execution frame.")
    if (
        attachment.reference_frame_spec_hash
        != port_config.execution_frame_spec_hash
    ):
        raise ValueError("Real observation execution-frame spec mismatch.")
    if set(attachment.component_ids) != set(
        port_config.component_id_by_side.values()
    ):
        raise ValueError("Observation and motion-port component sets disagree.")
    if len(attachment.component_ids) != 1:
        raise ValueError("The v1 real primitive profile binds exactly one arm component.")
    component_id = attachment.component_ids[0]
    side = next(
        side
        for side, bound_component in port_config.component_id_by_side.items()
        if bound_component == component_id
    )
    primitive_frame_id = _required(
        primitive_reference_frame_id,
        field_name="primitive_reference_frame_id",
    )
    try:
        primitive_frame = frame_graph.frame_by_id[primitive_frame_id]
    except KeyError as exc:
        raise ValueError("Primitive reference frame is absent from the graph.") from exc
    if (
        primitive_frame.semantic_parameters.get("axis_contract")
        != STATION_RETARGET_FRAME_CONTRACT_V1
    ):
        raise ValueError(
            "Primitive reference frame must declare the canonical Station axis contract."
        )
    convergence = convergence_profile or RealAgentPrimitiveConvergenceProfile()
    cardinal_profile = RealAgentTcpCardinalStepProfile(
        component_id=component_id,
        side=side,
        reference_frame_id=primitive_frame.frame_id,
        reference_frame_spec_hash=primitive_frame.semantic_content_hash,
        execution_reference_frame_id=port_config.execution_frame_id,
        execution_reference_frame_spec_hash=port_config.execution_frame_spec_hash,
        frame_graph_hash=frame_graph.semantic_content_hash,
        tool_config_hash=attachment.tool_config_hash,
        calibration_hash=attachment.calibration_hash,
        convergence_profile_hash=convergence.semantic_content_hash,
    )
    cardinal_definition = cardinal_tcp_step_skill_definition_v1(
        controlled_component_ids=(component_id,),
    )
    cardinal_schema = cardinal_tcp_step_schema_entry_v1(
        skill_definition_hash=cardinal_definition.semantic_content_hash,
        parameter_profile_id=cardinal_profile.parameter_profile_id,
        parameter_profile_hash=cardinal_profile.semantic_content_hash,
        allowed_component_ids=(component_id,),
        allowed_reference_frame_ids=(cardinal_profile.reference_frame_id,),
        reference_frame_spec_hash=cardinal_profile.reference_frame_spec_hash,
        frame_graph_hash=cardinal_profile.frame_graph_hash,
        tool_config_hash=cardinal_profile.tool_config_hash,
        calibration_hash=cardinal_profile.calibration_hash,
        primitive_catalog_hash=cardinal_profile.primitive_catalog_hash,
        allowed_directions=tuple(
            item.direction.value for item in cardinal_profile.primitives
        ),
    )

    orientation_profiles = tuple(
        RealAgentTcpFixedOrientationProfile(
            skill_id=skill_id,
            component_id=component_id,
            side=side,
            reference_frame_id=primitive_frame.frame_id,
            reference_frame_spec_hash=primitive_frame.semantic_content_hash,
            execution_reference_frame_id=port_config.execution_frame_id,
            execution_reference_frame_spec_hash=port_config.execution_frame_spec_hash,
            frame_graph_hash=frame_graph.semantic_content_hash,
            tool_config_hash=attachment.tool_config_hash,
            calibration_hash=attachment.calibration_hash,
            convergence_profile_hash=convergence.semantic_content_hash,
            primitive=level_tcp_orientation(
                REAL_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1[skill_id]
            ),
        )
        for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
    )
    orientation_definitions = tuple(
        fixed_level_tcp_orientation_skill_definition_v1(skill_id=profile.skill_id)
        for profile in orientation_profiles
    )
    orientation_schemas = tuple(
        fixed_level_tcp_orientation_schema_entry_v1(
            skill_id=profile.skill_id,
            skill_definition_hash=definition.semantic_content_hash,
            parameter_profile_id=profile.parameter_profile_id,
            parameter_profile_hash=profile.semantic_content_hash,
            allowed_component_ids=(component_id,),
            allowed_reference_frame_ids=(profile.reference_frame_id,),
            reference_frame_spec_hash=profile.reference_frame_spec_hash,
            frame_graph_hash=profile.frame_graph_hash,
            tool_config_hash=profile.tool_config_hash,
            calibration_hash=profile.calibration_hash,
            primitive_hash=profile.primitive.semantic_content_hash,
        )
        for profile, definition in zip(
            orientation_profiles,
            orientation_definitions,
            strict=True,
        )
    )

    def route_for(
        *,
        definition: Any,
        schema: Any,
        route_id: str,
    ) -> SkillExecutionRoute:
        route = SkillExecutionRoute(
            skill_id=definition.skill_id,
            skill_version=definition.skill_version,
            skill_definition_hash=definition.semantic_content_hash,
            route_id=route_id,
            route_version="1",
            parameter_schema_hash=schema.parameter_schema_hash,
            execution_result_schema_hash=schema.execution_result_schema_hash,
            task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
            allowed_parameter_profiles=schema.parameter_profiles,
            control_effect_class="motion",
            privilege_tier="t2_real_motion",
            supported_modes=("real_active",),
            target_kind="real",
            may_move_robot=True,
            may_open_publication=False,
            requires_operator_confirmation=True,
            requires_deadman=False,
            max_operation_duration_ms=10_000,
            cancellation_policy="hold_on_cancel_disconnect_or_timeout",
            requires_task_evaluation=False,
        )
        validate_agent_motion_route_schema_v1(route=route, schema_entry=schema)
        return route

    cardinal_route = route_for(
        definition=cardinal_definition,
        schema=cardinal_schema,
        route_id=REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1,
    )
    orientation_routes = tuple(
        route_for(
            definition=definition,
            schema=schema,
            route_id=REAL_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1[
                profile.skill_id
            ],
        )
        for profile, definition, schema in zip(
            orientation_profiles,
            orientation_definitions,
            orientation_schemas,
            strict=True,
        )
    )
    all_routes = (cardinal_route, *orientation_routes)
    for route in all_routes:
        if not any(route.route_id in ref.allowed_route_ids for ref in refs):
            raise ValueError(
                f"Real primitive Route {route.route_id!r} is absent from all lower envelopes."
            )

    motion_port_binding_hash = canonical_content_hash(
        {
            "active_sides": port_config.active_sides,
            "component_id_by_side": dict(port_config.component_id_by_side),
            "execution_frame_id": port_config.execution_frame_id,
            "execution_frame_spec_hash": port_config.execution_frame_spec_hash,
            "exclusive_command_claim": port_config.exclusive_command_claim,
            "command_path": "teleop_session_filter_guard_admission_dispatch",
            "hold_path": "teleop_session_owner_hold",
        },
        domain="dexteleop.real_agent_motion.port_binding.v1",
    )
    profile_by_route_hash = {
        cardinal_route.semantic_content_hash: cardinal_profile,
        **{
            route.semantic_content_hash: profile
            for route, profile in zip(
                orientation_routes,
                orientation_profiles,
                strict=True,
            )
        },
    }
    executor_by_route_hash = {
        cardinal_route.semantic_content_hash: REAL_AGENT_TCP_CARDINAL_STEP_EXECUTOR_ID_V1,
        **{
            route.semantic_content_hash: (
                REAL_AGENT_TCP_FIXED_ORIENTATION_EXECUTOR_ID_BY_SKILL_ID_V1[
                    profile.skill_id
                ]
            )
            for route, profile in zip(
                orientation_routes,
                orientation_profiles,
                strict=True,
            )
        },
    }
    bindings = tuple(
        RealAgentPrimitiveDeploymentBinding(
            deployment_binding_id=(
                "real-agent-primitive:"
                + route.route_id
                + ":"
                + attachment.runtime_session_id
            ),
            route_hash=route.semantic_content_hash,
            runtime_session_id=attachment.runtime_session_id,
            control_domain_id=attachment.control_domain_id,
            target_binding_id=attachment.target_binding_id,
            station_manifest_hash=attachment.station_manifest_hash,
            execution_manifest_hash=attachment.execution_manifest_hash,
            attachment_hash=attachment.semantic_content_hash,
            motion_port_binding_hash=motion_port_binding_hash,
            parameter_profile_hash=(
                profile_by_route_hash[route.semantic_content_hash].semantic_content_hash
            ),
            convergence_profile_hash=convergence.semantic_content_hash,
            component_id=component_id,
            side=side,
            reference_frame_id=primitive_frame.frame_id,
            reference_frame_spec_hash=primitive_frame.semantic_content_hash,
            execution_reference_frame_id=port_config.execution_frame_id,
            execution_reference_frame_spec_hash=port_config.execution_frame_spec_hash,
            frame_graph_hash=frame_graph.semantic_content_hash,
            tool_config_hash=attachment.tool_config_hash,
            calibration_hash=attachment.calibration_hash,
            executor_id=executor_by_route_hash[route.semantic_content_hash],
            safety_envelope_refs=refs,
        )
        for route in all_routes
    )
    binding_by_route_hash = {item.route_hash: item for item in bindings}
    definitions = SkillDefinitionCatalog(
        catalog_id="dexteleop.agent_skill_definitions.real_motion_primitives",
        catalog_version="1",
        skill_definitions=(cardinal_definition, *orientation_definitions),
    )
    schemas = AgentSkillSchemaCatalog(
        catalog_id="dexteleop.agent_skill_schemas.real_motion_primitives",
        catalog_version="1",
        schema_entries=(cardinal_schema, *orientation_schemas),
    )
    manifest = SkillRouteManifest(
        runtime_session_id=attachment.runtime_session_id,
        control_domain_id=attachment.control_domain_id,
        station_manifest_hash=attachment.station_manifest_hash,
        execution_manifest_hash=attachment.execution_manifest_hash,
        skill_definition_catalog_hash=definitions.semantic_content_hash,
        skill_schema_catalog_hash=schemas.semantic_content_hash,
        target_binding_id=attachment.target_binding_id,
        target_kind="real",
        current_mode="real_active",
        supported_transport_features=(
            "authoritative_port_receipt_v1",
            "idempotency_v1",
            "own_domain_hold_v1",
            "own_operation_cancel_v1",
            "own_operation_status_v1",
            "safety_profile_selection_v1",
        ),
        skill_routes=tuple(
            SkillRouteManifestEntry(
                route=route,
                deployment_binding=(
                    binding_by_route_hash[route.semantic_content_hash].to_ref()
                ),
                evaluation_plan_hash=None,
                availability_hash=canonical_content_hash(
                    {
                        "status": "default_off_requires_realteleop_owner_attachment",
                        "route_hash": route.semantic_content_hash,
                        "deployment_binding_hash": (
                            binding_by_route_hash[
                                route.semantic_content_hash
                            ].semantic_content_hash
                        ),
                    },
                    domain="dexteleop.agent_skill.real_primitive_availability.v1",
                ),
            )
            for route in all_routes
        ),
        explicit_exclusions=(
            "agent_estop",
            "agent_rearm",
            "agent_resume",
            "direct_hardware_refresh",
            "raw_action_submission",
            "second_dispatch_owner",
        ),
    )
    validators = AgentSkillPayloadValidatorRegistry(
        registry_id=(
            "dexteleop.agent_payload_validators.real_motion_primitives:"
            + attachment.runtime_session_id
        ),
        validators=(
            _payload_validator(route=cardinal_route, profile=cardinal_profile),
            *tuple(
                _payload_validator(route=route, profile=profile)
                for route, profile in zip(
                    orientation_routes,
                    orientation_profiles,
                    strict=True,
                )
            ),
        ),
    )
    return RealAgentPrimitiveContractBundle(
        definition_catalog=definitions,
        schema_catalog=schemas,
        route_manifest=manifest,
        cardinal_profile=cardinal_profile,
        orientation_profiles=orientation_profiles,
        deployment_bindings=bindings,
        convergence_profile=convergence,
        payload_validators=validators,
        frame_graph=frame_graph,
        attachment=attachment,
    )


__all__ = [
    "REAL_AGENT_TCP_CARDINAL_STEP_EXECUTOR_ID_V1",
    "REAL_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1",
    "REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1",
    "REAL_AGENT_TCP_FIXED_ORIENTATION_EXECUTOR_ID_BY_SKILL_ID_V1",
    "REAL_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1",
    "REAL_AGENT_TCP_FIXED_ORIENTATION_PARAMETER_PROFILE_ID_BY_SKILL_ID_V1",
    "REAL_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1",
    "REAL_AGENT_TCP_LEVEL_FORWARD_LEFT_30_ROUTE_ID_V1",
    "REAL_AGENT_TCP_LEVEL_FORWARD_RIGHT_30_ROUTE_ID_V1",
    "REAL_AGENT_TCP_LEVEL_FORWARD_ROUTE_ID_V1",
    "RealAgentPrimitiveContractBundle",
    "RealAgentPrimitiveConvergenceProfile",
    "RealAgentPrimitiveDeploymentBinding",
    "RealAgentTcpCardinalStepProfile",
    "RealAgentTcpFixedOrientationProfile",
    "build_real_agent_primitive_contract_bundle",
]
