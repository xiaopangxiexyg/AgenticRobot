"""Embodiment-neutral attach-only real TCP-pose Agent Gateway.

The Gateway consumes an immutable measured-state cut published by the existing
real runtime owner. It does not select a robot model or import an embodiment
backend. It never opens a device connection, refreshes hardware, starts a
backend, dispatches a command, requests HOLD, rearms, or homes.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from agent_gateway.protocol.contracts import (
    AgentObservationSnapshot,
    AgentSkillOperation,
)
from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaCatalog,
    SkillDefinitionCatalog,
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
    SkillRouteManifest,
    SkillRouteManifestEntry,
    validate_skill_route_catalog_closure,
)
from agent_gateway.protocol.state_schemas import (
    tcp_pose_state_definition_v1,
    tcp_pose_state_schema_entry_v1,
    validate_agent_state_parameters_v1,
    validate_agent_state_result_v1,
    validate_agent_state_route_schema_v1,
)
from agent_gateway.runtime.access_policy import AgentSkillAccessPolicy
from agent_gateway.runtime.executor_registry import (
    SkillExecutorBinding,
    SkillExecutorRegistry,
)
from agent_gateway.runtime.operation_port import (
    AgentGatewayNoIoPortReceipt,
)
from agent_gateway.runtime.operation_store import (
    BoundedInMemoryAgentSkillOperationStore,
)
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import (
    AgentSkillPayloadValidatorRegistry,
    AgentSkillPayloadValidatorSet,
)
from agent_gateway.runtime.service import AgentGatewayService
from agent_gateway.transport.unix_transport import (
    AgentGatewayUnixJsonServer,
    AgentGatewayUnixPeerClientMap,
)
from agent_gateway.transport.wire_adapter import (
    BoundAgentGatewayWireAdapter,
)
from dataset_contracts.canonical_encoding import canonical_content_hash


REAL_TCP_POSE_STATE_ROUTE_ID_V1 = "real.passive_tcp_pose"
REAL_TCP_POSE_STATE_EXECUTOR_ID_V1 = "control-station.real-passive-tcp-pose-reader.v1"
REAL_TCP_POSE_STATE_PROFILE_ID_V1 = "real-tool-tcp-pose-read-v1"
REAL_TCP_POSE_STATE_MAX_MEASUREMENT_AGE_MS_V1 = 200
REAL_TCP_POSE_STATE_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.agent_state.real_tcp_pose_deployment_binding.v1"
)
REAL_TCP_POSE_STATE_PROFILE_HASH_DOMAIN_V1 = (
    "dexteleop.agent_state.real_tcp_pose_parameter_profile.v1"
)
REAL_TCP_POSE_STATE_AVAILABILITY_HASH_DOMAIN_V1 = (
    "dexteleop.agent_state.real_tcp_pose_availability.v1"
)
REAL_TCP_POSE_OBSERVE_ATTACHMENT_HASH_DOMAIN_V1 = (
    "dexteleop.agent_state.real_tcp_pose_observe_attachment.v1"
)
REAL_TCP_POSE_OWNER_PUBLICATION_HASH_DOMAIN_V1 = (
    "dexteleop.agent_state.real_tcp_pose_owner_publication.v1"
)
REAL_TCP_POSE_OWNER_CUT_HASH_DOMAIN_V1 = (
    "dexteleop.agent_state.real_tcp_pose_owner_cut.v1"
)
REAL_CONTROL_OWNER_STATE_HASH_DOMAIN_V1 = (
    "dexteleop.agent_state.real_control_owner_state.v1"
)
REAL_TCP_POSE_MEASUREMENT_CUT_HASH_DOMAIN_V1 = (
    "dexteleop.agent_state.real_tcp_pose_measurement_cut.v1"
)
REAL_TCP_POSE_OWNER_PUBLICATION_PORT_HASH_DOMAIN_V1 = (
    "dexteleop.agent_state.real_tcp_pose_owner_publication_port.v1"
)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _component_ids(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field_name} must be a non-empty tuple.")
    normalized = tuple(_required(item, field_name=field_name) for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique values.")
    return tuple(sorted(normalized))


def _pose_by_component(
    value: object,
    *,
    field_name: str,
) -> Mapping[str, tuple[float, ...]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field_name} must be a non-empty mapping.")
    normalized: dict[str, tuple[float, ...]] = {}
    for raw_component_id, raw_pose in value.items():
        component_id = _required(raw_component_id, field_name="component_id")
        if component_id in normalized:
            raise ValueError(f"{field_name} contains a duplicate component.")
        if not isinstance(raw_pose, (list, tuple)) or len(raw_pose) != 7:
            raise ValueError(f"{field_name}.{component_id} must contain xyz+quat_xyzw.")
        pose = tuple(
            _finite_number(item, field_name=f"{field_name}.{component_id}")
            for item in raw_pose
        )
        quaternion_norm = math.sqrt(sum(item * item for item in pose[3:]))
        if abs(quaternion_norm - 1.0) > 1e-3:
            raise ValueError(
                f"{field_name}.{component_id} quaternion must be normalized."
            )
        normalized[component_id] = pose
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class RealTcpPoseObserveAttachmentSpec:
    """Exact deployment identities for one embodiment-neutral adapter."""

    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    target_binding_id: str
    component_ids: tuple[str, ...]
    station_manifest_hash: str
    execution_manifest_hash: str
    backend_inventory_hash: str
    source_binding_id: str
    source_binding_hash: str
    observation_binding_id: str
    observation_binding_hash: str
    observation_plan_hash: str
    observation_capability_hash: str
    pose_schema_hash: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    clock_domain_id: str
    producer_epoch: int
    max_measurement_age_ms: int = REAL_TCP_POSE_STATE_MAX_MEASUREMENT_AGE_MS_V1

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_session_id",
            "control_domain_id",
            "embodiment_instance_id",
            "target_binding_id",
            "station_manifest_hash",
            "execution_manifest_hash",
            "backend_inventory_hash",
            "source_binding_id",
            "source_binding_hash",
            "observation_binding_id",
            "observation_binding_hash",
            "observation_plan_hash",
            "observation_capability_hash",
            "pose_schema_hash",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "clock_domain_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "component_ids",
            _component_ids(self.component_ids, field_name="component_ids"),
        )
        if (
            isinstance(self.producer_epoch, bool)
            or not isinstance(self.producer_epoch, int)
            or self.producer_epoch < 0
        ):
            raise ValueError("producer_epoch must be a non-negative integer.")
        maximum_age = _positive_integer(
            self.max_measurement_age_ms,
            field_name="max_measurement_age_ms",
        )
        if maximum_age > REAL_TCP_POSE_STATE_MAX_MEASUREMENT_AGE_MS_V1:
            raise ValueError("Real TCP-pose freshness exceeds the v1 hard bound.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                **self.__dict__,
                "controlled_frame_id": "tool_tcp",
                "target_kind": "real",
                "adapter_scope": "embodiment_neutral_exact_instance_binding",
                "attachment_semantics": "owner_published_immutable_cut",
                "forbidden_calls": (
                    "backend_start",
                    "command_dispatch",
                    "hardware_refresh",
                    "hold",
                    "home",
                    "rearm",
                ),
            },
            domain=REAL_TCP_POSE_OBSERVE_ATTACHMENT_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class RealPassiveTcpPoseStateDeploymentBinding:
    """Exact real-observe deployment; it carries no motion authority."""

    attachment: RealTcpPoseObserveAttachmentSpec
    route_hash: str
    executor_id: str = REAL_TCP_POSE_STATE_EXECUTOR_ID_V1
    deployment_binding_version: str = "1"
    explicit_exclusions: tuple[str, ...] = (
        "control_authority",
        "device_connection",
        "device_start",
        "direct_hardware_refresh",
        "domain_hold",
        "home",
        "motion",
        "publication_open",
        "rearm",
        "task_evaluation",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.attachment, RealTcpPoseObserveAttachmentSpec):
            raise TypeError("attachment must be typed.")
        for field_name in (
            "route_hash",
            "executor_id",
            "deployment_binding_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        exclusions = tuple(
            sorted(
                {
                    _required(item, field_name="explicit_exclusion")
                    for item in self.explicit_exclusions
                }
            )
        )
        if not exclusions:
            raise ValueError("Real-observe deployment requires explicit exclusions.")
        object.__setattr__(self, "explicit_exclusions", exclusions)

    @property
    def deployment_binding_id(self) -> str:
        return (
            "real.agent_passive_tcp_pose:"
            + self.attachment.target_binding_id
            + ":"
            + self.attachment.runtime_session_id
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "deployment_binding_id": self.deployment_binding_id,
                "deployment_binding_version": self.deployment_binding_version,
                "route_hash": self.route_hash,
                "attachment_hash": self.attachment.semantic_content_hash,
                "executor_id": self.executor_id,
                "target_kind": "real",
                "current_mode": "real_observe",
                "read_semantics": "owner_published_immutable_cut",
                "explicit_exclusions": self.explicit_exclusions,
            },
            domain=REAL_TCP_POSE_STATE_BINDING_HASH_DOMAIN_V1,
        )

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=self.deployment_binding_id,
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.attachment.target_binding_id,
            target_kind="real",
            execution_manifest_hash=self.attachment.execution_manifest_hash,
        )


@dataclass(frozen=True)
class RealTcpPoseObserveContractBundle:
    definition_catalog: SkillDefinitionCatalog
    schema_catalog: AgentSkillSchemaCatalog
    route_manifest: SkillRouteManifest
    deployment_binding: RealPassiveTcpPoseStateDeploymentBinding
    payload_validators: AgentSkillPayloadValidatorRegistry
    attachment: RealTcpPoseObserveAttachmentSpec


def _payload_validators(
    *,
    route: SkillExecutionRoute,
    schema_entry: Any,
    attachment: RealTcpPoseObserveAttachmentSpec,
) -> AgentSkillPayloadValidatorSet:
    parameter_fields = {
        "component_ids",
        "reference_frame_id",
        "controlled_frame_id",
        "max_measurement_age_ms",
    }
    result_fields = set(schema_entry.execution_result_schema["properties"])

    def validate_parameters(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != parameter_fields:
            raise ValueError("Real TCP-pose query must exactly match its schema.")
        validate_agent_state_parameters_v1(
            schema_kind="tcp_pose",
            parameters=payload,
            allowed_component_ids=attachment.component_ids,
        )
        if payload["reference_frame_id"] != attachment.reference_frame_id:
            raise ValueError("Real TCP-pose query uses another reference frame.")
        maximum_age = payload["max_measurement_age_ms"]
        if (
            isinstance(maximum_age, bool)
            or not isinstance(maximum_age, int)
            or maximum_age > attachment.max_measurement_age_ms
        ):
            raise ValueError("Real TCP-pose query exceeds its freshness profile.")

    def validate_result(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != result_fields:
            raise ValueError("Real TCP-pose result must exactly match its schema.")
        exact = {
            "reference_frame_spec_hash": attachment.reference_frame_spec_hash,
            "frame_graph_hash": attachment.frame_graph_hash,
            "tool_config_hash": attachment.tool_config_hash,
            "calibration_hash": attachment.calibration_hash,
            "observation_binding_id": attachment.observation_binding_id,
            "observation_plan_hash": attachment.observation_plan_hash,
        }
        for field_name, expected in exact.items():
            if payload.get(field_name) != expected:
                raise ValueError(f"Real TCP-pose result {field_name} mismatch.")
        result_components = tuple(sorted(payload.get("component_ids", ())))
        if payload.get("pose_schema_hash_by_component") != {
            component_id: attachment.pose_schema_hash
            for component_id in result_components
        }:
            raise ValueError("Real TCP-pose result pose schema mismatch.")

    def validate_result_for_parameters(
        parameters: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> None:
        validate_parameters(parameters)
        validate_agent_state_result_v1(
            schema_kind="tcp_pose",
            parameters=parameters,
            result=payload,
            allowed_component_ids=attachment.component_ids,
        )

    def validate_task_evaluation(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or payload:
            raise ValueError("Real TCP-pose reads do not evaluate a task.")

    return AgentSkillPayloadValidatorSet(
        validator_id="payload-validator:real-passive-tcp-pose:"
        + attachment.runtime_session_id,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=validate_parameters,
        validate_execution_result=validate_result,
        validate_task_evaluation=validate_task_evaluation,
        validate_execution_result_for_parameters=validate_result_for_parameters,
    )


def build_real_tcp_pose_observe_contract_bundle(
    attachment: RealTcpPoseObserveAttachmentSpec,
) -> RealTcpPoseObserveContractBundle:
    """Build one exact real-observe Route without touching a device."""

    if not isinstance(attachment, RealTcpPoseObserveAttachmentSpec):
        raise TypeError("attachment must be typed.")
    definition = tcp_pose_state_definition_v1(
        controlled_component_ids=attachment.component_ids
    )
    profile_hash = canonical_content_hash(
        {
            "profile_id": REAL_TCP_POSE_STATE_PROFILE_ID_V1,
            "attachment_hash": attachment.semantic_content_hash,
            "component_ids": attachment.component_ids,
            "embodiment_instance_id": attachment.embodiment_instance_id,
            "reference_frame_id": attachment.reference_frame_id,
            "controlled_frame_id": "tool_tcp",
            "max_measurement_age_ms": attachment.max_measurement_age_ms,
        },
        domain=REAL_TCP_POSE_STATE_PROFILE_HASH_DOMAIN_V1,
    )
    schema = tcp_pose_state_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=REAL_TCP_POSE_STATE_PROFILE_ID_V1,
        parameter_profile_hash=profile_hash,
        allowed_component_ids=attachment.component_ids,
        allowed_reference_frame_ids=(attachment.reference_frame_id,),
        max_measurement_age_ms=attachment.max_measurement_age_ms,
    )
    route = SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=REAL_TCP_POSE_STATE_ROUTE_ID_V1,
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
        max_operation_duration_ms=1_000,
        cancellation_policy="cancel_only",
        requires_task_evaluation=False,
    )
    validate_agent_state_route_schema_v1(route=route, schema_entry=schema)
    binding = RealPassiveTcpPoseStateDeploymentBinding(
        attachment=attachment,
        route_hash=route.semantic_content_hash,
    )
    definitions = SkillDefinitionCatalog(
        catalog_id="dexteleop.agent_skill_definitions.real_tcp_pose_observe",
        catalog_version="1",
        skill_definitions=(definition,),
    )
    schemas = AgentSkillSchemaCatalog(
        catalog_id="dexteleop.agent_skill_schemas.real_tcp_pose_observe",
        catalog_version="1",
        schema_entries=(schema,),
    )
    availability_hash = canonical_content_hash(
        {
            "status": "attach_only_requires_owner_published_measurement",
            "route_hash": route.semantic_content_hash,
            "deployment_binding_hash": binding.semantic_content_hash,
            "attachment_hash": attachment.semantic_content_hash,
        },
        domain=REAL_TCP_POSE_STATE_AVAILABILITY_HASH_DOMAIN_V1,
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
        current_mode="real_observe",
        supported_transport_features=(
            "idempotency_v1",
            "own_operation_cancel_v1",
            "own_operation_status_v1",
        ),
        skill_routes=(
            SkillRouteManifestEntry(
                route=route,
                deployment_binding=binding.to_ref(),
                evaluation_plan_hash=None,
                availability_hash=availability_hash,
            ),
        ),
        explicit_exclusions=(
            "agent_estop",
            "agent_rearm",
            "agent_resume",
            "control_authority",
            "device_connection",
            "device_start",
            "direct_hardware_refresh",
            "domain_hold",
            "motion",
            "raw_action_submission",
            "second_dispatch_owner",
        ),
    )
    validate_skill_route_catalog_closure(
        definition_catalog=definitions,
        schema_catalog=schemas,
        route_manifest=manifest,
    )
    validators = AgentSkillPayloadValidatorRegistry(
        registry_id=(
            "dexteleop.agent_payload_validators.real_tcp_pose_observe:"
            + attachment.runtime_session_id
        ),
        validators=(
            _payload_validators(
                route=route,
                schema_entry=schema,
                attachment=attachment,
            ),
        ),
    )
    return RealTcpPoseObserveContractBundle(
        definition_catalog=definitions,
        schema_catalog=schemas,
        route_manifest=manifest,
        deployment_binding=binding,
        payload_validators=validators,
        attachment=attachment,
    )


@dataclass(frozen=True)
class RealTcpPosePublishedRead:
    """One immutable read already published by the existing real owner."""

    snapshot: AgentObservationSnapshot
    execution_result: Mapping[str, Any]
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, AgentObservationSnapshot):
            raise TypeError("snapshot must be an AgentObservationSnapshot.")
        if not isinstance(self.execution_result, Mapping):
            raise TypeError("execution_result must be a mapping.")
        object.__setattr__(self, "execution_result", dict(self.execution_result))
        refs = tuple(
            sorted(
                {
                    _required(item, field_name="evidence_ref")
                    for item in self.evidence_refs
                }
            )
        )
        if not refs:
            raise ValueError("A published real read requires lower evidence.")
        object.__setattr__(self, "evidence_refs", refs)


class RealTcpPosePublishedReadSource(Protocol):
    """Existing-owner port; capture must not perform device or control calls."""

    def capture_published_tcp_pose(
        self,
        *,
        active_operation_id: str | None,
        parameters: Mapping[str, Any],
    ) -> RealTcpPosePublishedRead: ...


@dataclass(frozen=True)
class RealControlOwnerState:
    """Exact state already captured by the lower command owner.

    This contract deliberately contains no robot-model or component naming.
    A real launcher must project its existing owner registry into this object;
    the Agent Gateway is not allowed to synthesize these facts.
    """

    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    target_binding_id: str
    execution_manifest_hash: str
    backend_inventory_hash: str
    clock_domain_id: str
    owner_token_id: str
    owner_kind: str
    owner_generation: int
    authority_epoch: int
    execution_epoch: int
    lifecycle_state: str
    execution_gate: str
    owner_state_revision: int
    captured_at: float
    source_owner_state_hash: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_session_id",
            "control_domain_id",
            "embodiment_instance_id",
            "target_binding_id",
            "execution_manifest_hash",
            "backend_inventory_hash",
            "clock_domain_id",
            "owner_token_id",
            "owner_kind",
            "lifecycle_state",
            "execution_gate",
            "source_owner_state_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "owner_generation",
            "authority_epoch",
            "execution_epoch",
            "owner_state_revision",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )
        object.__setattr__(
            self,
            "captured_at",
            _finite_number(self.captured_at, field_name="captured_at"),
        )
        refs = tuple(
            sorted(
                {
                    _required(item, field_name="evidence_ref")
                    for item in self.evidence_refs
                }
            )
        )
        if not refs:
            raise ValueError("Real owner state requires lower-owned evidence.")
        object.__setattr__(self, "evidence_refs", refs)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.__dict__,
            domain=REAL_CONTROL_OWNER_STATE_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class RealTcpPoseMeasurementCut:
    """Immutable measured pose supplied by the existing lower observation cut."""

    embodiment_instance_id: str
    target_binding_id: str
    clock_domain_id: str
    source_binding_hash: str
    observation_binding_hash: str
    observation_plan_hash: str
    pose_schema_hash: str
    measurement_id: str
    source_measurement_hash: str
    measured_at: float
    available_at: float
    pose_xyz_quat_xyzw_by_component: Mapping[str, tuple[float, ...]]
    evidence_refs: tuple[str, ...]
    measurement_valid: bool = True
    fault_class: str | None = None
    fault_rearmable: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "embodiment_instance_id",
            "target_binding_id",
            "clock_domain_id",
            "source_binding_hash",
            "observation_binding_hash",
            "observation_plan_hash",
            "pose_schema_hash",
            "measurement_id",
            "source_measurement_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        measured_at = _finite_number(self.measured_at, field_name="measured_at")
        available_at = _finite_number(
            self.available_at,
            field_name="available_at",
        )
        if available_at < measured_at:
            raise ValueError("Measurement availability cannot precede measurement.")
        object.__setattr__(self, "measured_at", measured_at)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(
            self,
            "pose_xyz_quat_xyzw_by_component",
            _pose_by_component(
                self.pose_xyz_quat_xyzw_by_component,
                field_name="pose_xyz_quat_xyzw_by_component",
            ),
        )
        refs = tuple(
            sorted(
                {
                    _required(item, field_name="evidence_ref")
                    for item in self.evidence_refs
                }
            )
        )
        if not refs:
            raise ValueError("A measured pose cut requires lower-owned evidence.")
        object.__setattr__(self, "evidence_refs", refs)
        if not isinstance(self.measurement_valid, bool):
            raise TypeError("measurement_valid must be a boolean.")
        if self.fault_class is not None:
            object.__setattr__(
                self,
                "fault_class",
                _required(self.fault_class, field_name="fault_class"),
            )
        if not isinstance(self.fault_rearmable, bool):
            raise TypeError("fault_rearmable must be a boolean.")
        if self.fault_class is None and self.fault_rearmable:
            raise ValueError("fault_rearmable requires fault_class.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                **self.__dict__,
                "pose_xyz_quat_xyzw_by_component": dict(
                    self.pose_xyz_quat_xyzw_by_component
                ),
            },
            domain=REAL_TCP_POSE_MEASUREMENT_CUT_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class RealTcpPoseOwnerPublication:
    """One measurement cut explicitly published by the existing real owner."""

    publication_id: str
    attachment_hash: str
    clock_domain_id: str
    station_revision: int
    owner_token_id: str
    owner_kind: str
    source_owner_state_hash: str
    authority_epoch: int
    execution_epoch: int
    dispatch_owner_generation: int
    source_owner_state_revision: int
    lifecycle_state: str
    execution_gate: str
    measurement_id: str
    source_measurement_hash: str
    measured_at: float
    available_at: float
    pose_xyz_quat_xyzw_by_component: Mapping[str, tuple[float, ...]]
    evidence_refs: tuple[str, ...]
    measurement_valid: bool = True
    fault_class: str | None = None
    fault_rearmable: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "publication_id",
            "attachment_hash",
            "clock_domain_id",
            "owner_token_id",
            "owner_kind",
            "source_owner_state_hash",
            "lifecycle_state",
            "execution_gate",
            "measurement_id",
            "source_measurement_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "station_revision",
            "authority_epoch",
            "execution_epoch",
            "dispatch_owner_generation",
            "source_owner_state_revision",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )
        measured_at = _finite_number(self.measured_at, field_name="measured_at")
        available_at = _finite_number(
            self.available_at,
            field_name="available_at",
        )
        if available_at < measured_at:
            raise ValueError("Owner publication cannot precede its measurement.")
        object.__setattr__(self, "measured_at", measured_at)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(
            self,
            "pose_xyz_quat_xyzw_by_component",
            _pose_by_component(
                self.pose_xyz_quat_xyzw_by_component,
                field_name="pose_xyz_quat_xyzw_by_component",
            ),
        )
        refs = tuple(
            sorted(
                {
                    _required(item, field_name="evidence_ref")
                    for item in self.evidence_refs
                }
            )
        )
        if not refs:
            raise ValueError("Owner publication requires immutable evidence.")
        object.__setattr__(self, "evidence_refs", refs)
        if not isinstance(self.measurement_valid, bool):
            raise TypeError("measurement_valid must be a boolean.")
        if self.fault_class is not None:
            object.__setattr__(
                self,
                "fault_class",
                _required(self.fault_class, field_name="fault_class"),
            )
        if not isinstance(self.fault_rearmable, bool):
            raise TypeError("fault_rearmable must be a boolean.")
        if self.fault_class is None and self.fault_rearmable:
            raise ValueError("fault_rearmable requires fault_class.")

    @property
    def measurement_observation_hash(self) -> str:
        return canonical_content_hash(
            {
                "measurement_id": self.measurement_id,
                "source_measurement_hash": self.source_measurement_hash,
                "measured_at": self.measured_at,
                "available_at": self.available_at,
                "pose_xyz_quat_xyzw_by_component": (
                    self.pose_xyz_quat_xyzw_by_component
                ),
                "measurement_valid": self.measurement_valid,
            },
            domain=REAL_TCP_POSE_OWNER_PUBLICATION_HASH_DOMAIN_V1 + ".measurement",
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                **self.__dict__,
                "measurement_observation_hash": (self.measurement_observation_hash),
            },
            domain=REAL_TCP_POSE_OWNER_PUBLICATION_HASH_DOMAIN_V1,
        )


class RealTcpPoseOwnerPublicationBuffer:
    """Thread-safe latest-cut source; capture performs no owner/backend I/O."""

    no_io = True

    def __init__(
        self,
        *,
        contract_bundle: RealTcpPoseObserveContractBundle,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(contract_bundle, RealTcpPoseObserveContractBundle):
            raise TypeError("contract_bundle must be typed.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self.contract_bundle = contract_bundle
        self._clock = clock
        self._lock = threading.RLock()
        self._latest: RealTcpPoseOwnerPublication | None = None

    @property
    def latest_publication(self) -> RealTcpPoseOwnerPublication | None:
        with self._lock:
            return self._latest

    def publish_from_owner(
        self,
        publication: RealTcpPoseOwnerPublication,
    ) -> None:
        """Owner-side call; publication identity must advance monotonically."""

        if not isinstance(publication, RealTcpPoseOwnerPublication):
            raise TypeError("publication must be typed.")
        with self._lock:
            current = self._latest
            attachment = self.contract_bundle.attachment
            if publication.attachment_hash != attachment.semantic_content_hash:
                raise ValueError("Owner publication attachment hash mismatch.")
            if publication.clock_domain_id != attachment.clock_domain_id:
                raise ValueError("Owner publication clock domain mismatch.")
            if set(publication.pose_xyz_quat_xyzw_by_component) != set(
                attachment.component_ids
            ):
                raise ValueError(
                    "Owner publication does not exactly cover attachment components."
                )
            if current is not None and (
                publication.station_revision <= current.station_revision
            ):
                raise ValueError("Owner publication station revision did not advance.")
            if current is not None and publication.available_at < current.available_at:
                raise ValueError("Owner publication availability time regressed.")
            if current is not None:
                for field_name in (
                    "authority_epoch",
                    "execution_epoch",
                    "dispatch_owner_generation",
                    "source_owner_state_revision",
                ):
                    if getattr(publication, field_name) < getattr(
                        current,
                        field_name,
                    ):
                        raise ValueError(
                            f"Owner publication {field_name} regressed."
                        )
                if (
                    publication.source_owner_state_revision
                    == current.source_owner_state_revision
                    and publication.source_owner_state_hash
                    != current.source_owner_state_hash
                ):
                    raise ValueError(
                        "Owner state hash changed without a new owner-state revision."
                    )
                if (
                    publication.source_owner_state_revision
                    > current.source_owner_state_revision
                    and publication.source_owner_state_hash
                    == current.source_owner_state_hash
                ):
                    raise ValueError(
                        "Owner-state revision advanced without new owner state."
                    )
                owner_changed = (
                    publication.owner_token_id != current.owner_token_id
                    or publication.owner_kind != current.owner_kind
                )
                generation_advanced = (
                    publication.dispatch_owner_generation
                    > current.dispatch_owner_generation
                )
                if owner_changed and not generation_advanced:
                    raise ValueError(
                        "Owner identity changed without a new owner generation."
                    )
            self._latest = publication

    def capture_published_tcp_pose(
        self,
        *,
        active_operation_id: str | None,
        parameters: Mapping[str, Any],
    ) -> RealTcpPosePublishedRead:
        with self._lock:
            publication = self._latest
        if publication is None:
            raise RuntimeError("No real-owner TCP-pose publication is available.")
        attachment = self.contract_bundle.attachment
        route = self.contract_bundle.route_manifest.route(
            "state.read_tcp_pose",
            "1",
        ).route
        self.contract_bundle.payload_validators.validate_parameters(
            route=route,
            payload=parameters,
        )
        cutoff_at = _finite_number(self._clock(), field_name="clock result")
        if cutoff_at < publication.available_at:
            raise RuntimeError("Real-owner publication is not causally available.")
        freshness_ms = max(
            0,
            int(math.ceil((cutoff_at - publication.measured_at) * 1_000.0)),
        )
        cut_hash = canonical_content_hash(
            {
                "publication_hash": publication.semantic_content_hash,
                "cutoff_at": cutoff_at,
                "clock_domain_id": attachment.clock_domain_id,
                "active_operation_id": active_operation_id,
            },
            domain=REAL_TCP_POSE_OWNER_CUT_HASH_DOMAIN_V1,
        )
        requested_component_ids = tuple(sorted(parameters["component_ids"]))
        poses = {
            component_id: list(
                publication.pose_xyz_quat_xyzw_by_component[component_id]
            )
            for component_id in requested_component_ids
        }
        result = {
            "observation_cut_id": "real-owner-cut:" + cut_hash,
            "observation_cut_hash": cut_hash,
            "cutoff_at": cutoff_at,
            "cutoff_clock_domain_id": attachment.clock_domain_id,
            "component_ids": list(requested_component_ids),
            "reference_frame_id": attachment.reference_frame_id,
            "reference_frame_spec_hash": attachment.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": attachment.frame_graph_hash,
            "tool_config_hash": attachment.tool_config_hash,
            "calibration_hash": attachment.calibration_hash,
            "measurement_id": publication.measurement_id,
            "measurement_observation_hash": publication.measurement_observation_hash,
            "source_measurement_hash": publication.source_measurement_hash,
            "measured_at": publication.measured_at,
            "available_at": publication.available_at,
            "clock_domain_id": attachment.clock_domain_id,
            "pose_by_component": poses,
            "pose_schema_hash_by_component": {
                component_id: attachment.pose_schema_hash
                for component_id in requested_component_ids
            },
            "representation_by_component": {
                component_id: "cartesian_pose"
                for component_id in requested_component_ids
            },
            "observation_binding_id": attachment.observation_binding_id,
            "observation_plan_hash": attachment.observation_plan_hash,
        }
        summary = {
            "pose_xyz_quat_xyzw_by_component": poses,
            "measured_at": publication.measured_at,
            "attachment_hash": attachment.semantic_content_hash,
            "embodiment_instance_id": attachment.embodiment_instance_id,
            "source_binding_hash": attachment.source_binding_hash,
            "observation_binding_id": attachment.observation_binding_id,
            "observation_binding_hash": attachment.observation_binding_hash,
            "observation_plan_hash": attachment.observation_plan_hash,
            "observation_capability_hash": attachment.observation_capability_hash,
            "owner_publication_id": publication.publication_id,
            "owner_publication_hash": publication.semantic_content_hash,
        }
        snapshot = AgentObservationSnapshot(
            snapshot_id="real-owner-snapshot:" + cut_hash,
            runtime_session_id=attachment.runtime_session_id,
            skill_route_manifest_hash=(
                self.contract_bundle.route_manifest.semantic_content_hash
            ),
            station_revision=publication.station_revision,
            captured_at=cutoff_at,
            available_at=cutoff_at,
            clock_domain_id=attachment.clock_domain_id,
            authority_epoch=publication.authority_epoch,
            execution_epoch=publication.execution_epoch,
            dispatch_owner_generation=publication.dispatch_owner_generation,
            lifecycle_state=publication.lifecycle_state,
            execution_gate=publication.execution_gate,
            active_operation_id=active_operation_id,
            source_binding_id=attachment.source_binding_id,
            producer_epoch=attachment.producer_epoch,
            target_binding_id=attachment.target_binding_id,
            execution_manifest_hash=attachment.execution_manifest_hash,
            frame_graph_hash=attachment.frame_graph_hash,
            tool_config_hash=attachment.tool_config_hash,
            calibration_hash=attachment.calibration_hash,
            measurement_summary=summary,
            measurement_freshness_ms=freshness_ms,
            measurement_valid=publication.measurement_valid,
            fault_class=publication.fault_class,
            fault_rearmable=publication.fault_rearmable,
        )
        return RealTcpPosePublishedRead(
            snapshot=snapshot,
            execution_result=result,
            evidence_refs=publication.evidence_refs,
        )


class RealTcpPoseOwnerPublicationPort:
    """Lower-owner publication seam; owns revision, but no device/control I/O."""

    no_io = True

    def __init__(
        self,
        *,
        contract_bundle: RealTcpPoseObserveContractBundle,
        publication_buffer: RealTcpPoseOwnerPublicationBuffer,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(contract_bundle, RealTcpPoseObserveContractBundle):
            raise TypeError("contract_bundle must be typed.")
        if not isinstance(publication_buffer, RealTcpPoseOwnerPublicationBuffer):
            raise TypeError("publication_buffer must be typed.")
        if publication_buffer.contract_bundle is not contract_bundle:
            raise ValueError("Publication port and buffer must share one bundle.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self.contract_bundle = contract_bundle
        self.publication_buffer = publication_buffer
        self._clock = clock
        self._lock = threading.RLock()

    @property
    def semantic_content_hash(self) -> str:
        attachment = self.contract_bundle.attachment
        return canonical_content_hash(
            {
                "attachment_hash": attachment.semantic_content_hash,
                "route_manifest_hash": (
                    self.contract_bundle.route_manifest.semantic_content_hash
                ),
                "owner_semantics": "existing_lower_owner_only",
                "revision_owner": "lower_owner_publication_port",
                "io_semantics": "no_device_or_control_io",
            },
            domain=REAL_TCP_POSE_OWNER_PUBLICATION_PORT_HASH_DOMAIN_V1,
        )

    def publish(
        self,
        *,
        owner_state: RealControlOwnerState,
        measurement: RealTcpPoseMeasurementCut,
    ) -> RealTcpPoseOwnerPublication:
        """Publish one exact owner+measurement cut into the passive buffer."""

        if not isinstance(owner_state, RealControlOwnerState):
            raise TypeError("owner_state must be typed.")
        if not isinstance(measurement, RealTcpPoseMeasurementCut):
            raise TypeError("measurement must be typed.")
        attachment = self.contract_bundle.attachment
        owner_exact = {
            "runtime_session_id": attachment.runtime_session_id,
            "control_domain_id": attachment.control_domain_id,
            "embodiment_instance_id": attachment.embodiment_instance_id,
            "target_binding_id": attachment.target_binding_id,
            "execution_manifest_hash": attachment.execution_manifest_hash,
            "backend_inventory_hash": attachment.backend_inventory_hash,
            "clock_domain_id": attachment.clock_domain_id,
        }
        for field_name, expected in owner_exact.items():
            if getattr(owner_state, field_name) != expected:
                raise ValueError(f"Real owner {field_name} mismatch.")
        measurement_exact = {
            "embodiment_instance_id": attachment.embodiment_instance_id,
            "target_binding_id": attachment.target_binding_id,
            "clock_domain_id": attachment.clock_domain_id,
            "source_binding_hash": attachment.source_binding_hash,
            "observation_binding_hash": attachment.observation_binding_hash,
            "observation_plan_hash": attachment.observation_plan_hash,
            "pose_schema_hash": attachment.pose_schema_hash,
        }
        for field_name, expected in measurement_exact.items():
            if getattr(measurement, field_name) != expected:
                raise ValueError(f"Real measurement {field_name} mismatch.")
        if set(measurement.pose_xyz_quat_xyzw_by_component) != set(
            attachment.component_ids
        ):
            raise ValueError(
                "Real measurement does not exactly cover attachment components."
            )
        if owner_state.captured_at > measurement.available_at:
            raise ValueError(
                "Owner state was captured after the measurement became available."
            )
        with self._lock:
            latest = self.publication_buffer.latest_publication
            station_revision = 1 if latest is None else latest.station_revision + 1
            published_at = _finite_number(
                self._clock(),
                field_name="publication clock",
            )
            available_at = max(measurement.available_at, owner_state.captured_at)
            if published_at < available_at:
                raise RuntimeError(
                    "Owner publication clock precedes its lower-owned inputs."
                )
            payload = {
                "attachment_hash": attachment.semantic_content_hash,
                "station_revision": station_revision,
                "owner_state_hash": owner_state.semantic_content_hash,
                "measurement_hash": measurement.semantic_content_hash,
                "published_at": published_at,
            }
            publication_hash = canonical_content_hash(
                payload,
                domain=REAL_TCP_POSE_OWNER_PUBLICATION_PORT_HASH_DOMAIN_V1,
            )
            publication = RealTcpPoseOwnerPublication(
                publication_id="real-owner-publication:" + publication_hash,
                attachment_hash=attachment.semantic_content_hash,
                clock_domain_id=attachment.clock_domain_id,
                station_revision=station_revision,
                owner_token_id=owner_state.owner_token_id,
                owner_kind=owner_state.owner_kind,
                source_owner_state_hash=owner_state.source_owner_state_hash,
                authority_epoch=owner_state.authority_epoch,
                execution_epoch=owner_state.execution_epoch,
                dispatch_owner_generation=owner_state.owner_generation,
                source_owner_state_revision=owner_state.owner_state_revision,
                lifecycle_state=owner_state.lifecycle_state,
                execution_gate=owner_state.execution_gate,
                measurement_id=measurement.measurement_id,
                source_measurement_hash=measurement.source_measurement_hash,
                measured_at=measurement.measured_at,
                available_at=available_at,
                pose_xyz_quat_xyzw_by_component=(
                    measurement.pose_xyz_quat_xyzw_by_component
                ),
                evidence_refs=tuple(
                    sorted(
                        {
                            *owner_state.evidence_refs,
                            *measurement.evidence_refs,
                            owner_state.semantic_content_hash,
                            measurement.semantic_content_hash,
                            self.semantic_content_hash,
                        }
                    )
                ),
                measurement_valid=measurement.measurement_valid,
                fault_class=measurement.fault_class,
                fault_rearmable=measurement.fault_rearmable,
            )
            self.publication_buffer.publish_from_owner(publication)
            return publication

    def project_lower_owner_state(
        self,
        lower_owner_snapshot: object,
    ) -> RealControlOwnerState:
        """Project one immutable lower-context snapshot without owning state.

        The method intentionally uses a structural boundary so this pure
        Control Station package does not import a concrete deployment runtime.
        Missing or mismatched lower facts fail closed; attachment identities
        are never used to fill absent owner facts.
        """

        attachment = self.contract_bundle.attachment
        required_fields = (
            "runtime_session_id",
            "control_domain_id",
            "embodiment_instance_id",
            "clock_domain_id",
            "execution_manifest_hash",
            "dispatch_inventory_hash",
            "current_owner_token_id",
            "current_owner_kind",
            "current_owner_generation",
            "authority_epoch",
            "execution_epoch",
            "lifecycle_state",
            "execution_gate",
            "owner_state_revision",
            "execution_gate_snapshot_hash",
            "last_execution_cancellation_hash",
            "last_owner_transition_hash",
            "captured_at",
            "semantic_content_hash",
        )
        missing = tuple(
            field_name
            for field_name in required_fields
            if not hasattr(lower_owner_snapshot, field_name)
        )
        if missing:
            raise TypeError(
                "Lower owner snapshot lacks required fields: " + ", ".join(missing)
            )
        exact = {
            "runtime_session_id": attachment.runtime_session_id,
            "control_domain_id": attachment.control_domain_id,
            "embodiment_instance_id": attachment.embodiment_instance_id,
            "clock_domain_id": attachment.clock_domain_id,
            "execution_manifest_hash": attachment.execution_manifest_hash,
            "dispatch_inventory_hash": attachment.backend_inventory_hash,
        }
        for field_name, expected in exact.items():
            if getattr(lower_owner_snapshot, field_name) != expected:
                raise ValueError(f"Lower owner {field_name} mismatch.")
        lifecycle_state = str(lower_owner_snapshot.lifecycle_state)
        execution_gate = str(lower_owner_snapshot.execution_gate)
        expected_gate = {
            "configured": "hold",
            "connected": "hold",
            "preflighted": "hold",
            "shadow_ready": "hold",
            "armed": "hold",
            "running": "ready",
            "transitioning": "transitioning",
            "hold": "hold",
            "fault": "fault",
            "stopped": "hold",
        }.get(lifecycle_state)
        if expected_gate is None or execution_gate != expected_gate:
            raise ValueError(
                "Lower owner lifecycle/execution-gate pair is unsupported or inconsistent."
            )
        evidence_refs = {
            _required(
                lower_owner_snapshot.semantic_content_hash,
                field_name="semantic_content_hash",
            ),
            _required(
                lower_owner_snapshot.execution_gate_snapshot_hash,
                field_name="execution_gate_snapshot_hash",
            ),
        }
        if lower_owner_snapshot.last_owner_transition_hash is not None:
            evidence_refs.add(
                _required(
                    lower_owner_snapshot.last_owner_transition_hash,
                    field_name="last_owner_transition_hash",
                )
            )
        if lower_owner_snapshot.last_execution_cancellation_hash is not None:
            evidence_refs.add(
                _required(
                    lower_owner_snapshot.last_execution_cancellation_hash,
                    field_name="last_execution_cancellation_hash",
                )
            )
        return RealControlOwnerState(
            runtime_session_id=lower_owner_snapshot.runtime_session_id,
            control_domain_id=lower_owner_snapshot.control_domain_id,
            embodiment_instance_id=lower_owner_snapshot.embodiment_instance_id,
            target_binding_id=attachment.target_binding_id,
            execution_manifest_hash=lower_owner_snapshot.execution_manifest_hash,
            backend_inventory_hash=lower_owner_snapshot.dispatch_inventory_hash,
            clock_domain_id=lower_owner_snapshot.clock_domain_id,
            owner_token_id=lower_owner_snapshot.current_owner_token_id,
            owner_kind=lower_owner_snapshot.current_owner_kind,
            owner_generation=lower_owner_snapshot.current_owner_generation,
            authority_epoch=lower_owner_snapshot.authority_epoch,
            execution_epoch=lower_owner_snapshot.execution_epoch,
            lifecycle_state=lifecycle_state,
            execution_gate=execution_gate,
            owner_state_revision=lower_owner_snapshot.owner_state_revision,
            captured_at=lower_owner_snapshot.captured_at,
            source_owner_state_hash=lower_owner_snapshot.semantic_content_hash,
            evidence_refs=tuple(sorted(evidence_refs)),
        )

    def capture_published_tcp_pose(
        self,
        *,
        active_operation_id: str | None,
        parameters: Mapping[str, Any],
    ) -> RealTcpPosePublishedRead:
        return self.publication_buffer.capture_published_tcp_pose(
            active_operation_id=active_operation_id,
            parameters=parameters,
        )


class RealTcpPoseObservationProducer:
    """Validate owner-published cuts; never refresh or control a backend."""

    def __init__(
        self,
        *,
        source: RealTcpPosePublishedReadSource,
        contract_bundle: RealTcpPoseObserveContractBundle,
    ) -> None:
        capture = getattr(source, "capture_published_tcp_pose", None)
        if not callable(capture):
            raise TypeError("source must expose capture_published_tcp_pose().")
        if not isinstance(contract_bundle, RealTcpPoseObserveContractBundle):
            raise TypeError("contract_bundle must be typed.")
        self.source = source
        self.contract_bundle = contract_bundle
        self._lock = threading.RLock()
        self._latest_station_revision = 0

    @property
    def current_station_revision(self) -> int:
        with self._lock:
            return self._latest_station_revision

    @property
    def default_parameters(self) -> Mapping[str, Any]:
        attachment = self.contract_bundle.attachment
        return {
            "component_ids": list(attachment.component_ids),
            "reference_frame_id": attachment.reference_frame_id,
            "controlled_frame_id": "tool_tcp",
            "max_measurement_age_ms": attachment.max_measurement_age_ms,
        }

    def _capture(
        self,
        *,
        active_operation_id: str | None,
        parameters: Mapping[str, Any],
    ) -> RealTcpPosePublishedRead:
        published = self.source.capture_published_tcp_pose(
            active_operation_id=active_operation_id,
            parameters=parameters,
        )
        if not isinstance(published, RealTcpPosePublishedRead):
            raise TypeError("source returned an invalid published real read.")
        snapshot = published.snapshot
        attachment = self.contract_bundle.attachment
        manifest = self.contract_bundle.route_manifest
        checks = (
            (
                snapshot.runtime_session_id == attachment.runtime_session_id,
                "runtime session",
            ),
            (
                snapshot.skill_route_manifest_hash == manifest.semantic_content_hash,
                "Route manifest",
            ),
            (
                snapshot.target_binding_id == attachment.target_binding_id,
                "target binding",
            ),
            (
                snapshot.execution_manifest_hash == attachment.execution_manifest_hash,
                "execution manifest",
            ),
            (snapshot.clock_domain_id == attachment.clock_domain_id, "clock domain"),
            (
                snapshot.source_binding_id == attachment.source_binding_id,
                "source binding",
            ),
            (snapshot.producer_epoch == attachment.producer_epoch, "producer epoch"),
            (snapshot.frame_graph_hash == attachment.frame_graph_hash, "frame graph"),
            (snapshot.tool_config_hash == attachment.tool_config_hash, "tool config"),
            (snapshot.calibration_hash == attachment.calibration_hash, "calibration"),
            (snapshot.active_operation_id == active_operation_id, "active operation"),
        )
        for matches, label in checks:
            if not matches:
                raise ValueError(f"Published real observation {label} mismatch.")
        if not snapshot.measurement_valid:
            raise RuntimeError("Published real TCP-pose measurement is invalid.")
        if snapshot.measurement_freshness_ms > attachment.max_measurement_age_ms:
            raise RuntimeError("Published real TCP-pose measurement is stale.")
        route = manifest.route("state.read_tcp_pose", "1").route
        self.contract_bundle.payload_validators.validate_execution_result(
            route=route,
            payload=published.execution_result,
            request_parameters=parameters,
        )
        requested_component_ids = tuple(sorted(parameters["component_ids"]))
        result_poses = published.execution_result["pose_by_component"]
        summary_pose = snapshot.measurement_summary.get(
            "pose_xyz_quat_xyzw_by_component"
        )
        if not isinstance(summary_pose, Mapping) or set(summary_pose) != set(
            requested_component_ids
        ):
            raise ValueError("Snapshot and execution result pose cuts differ.")
        for component_id in requested_component_ids:
            if list(summary_pose[component_id]) != list(result_poses[component_id]):
                raise ValueError("Snapshot and execution result pose cuts differ.")
        summary_identity_fields = {
            "attachment_hash": attachment.semantic_content_hash,
            "embodiment_instance_id": attachment.embodiment_instance_id,
            "source_binding_hash": attachment.source_binding_hash,
            "observation_binding_id": attachment.observation_binding_id,
            "observation_binding_hash": attachment.observation_binding_hash,
            "observation_plan_hash": attachment.observation_plan_hash,
            "observation_capability_hash": attachment.observation_capability_hash,
        }
        for field_name, expected in summary_identity_fields.items():
            if snapshot.measurement_summary.get(field_name) != expected:
                raise ValueError(f"Published real observation {field_name} mismatch.")
        if snapshot.measurement_summary.get(
            "measured_at"
        ) != published.execution_result.get("measured_at"):
            raise ValueError("Snapshot and execution result timestamps differ.")
        with self._lock:
            if snapshot.station_revision < self._latest_station_revision:
                raise ValueError("Published real station revision regressed.")
            self._latest_station_revision = snapshot.station_revision
        return published

    def capture_snapshot(
        self,
        *,
        active_operation_id: str | None,
    ) -> AgentObservationSnapshot:
        return self._capture(
            active_operation_id=active_operation_id,
            parameters=self.default_parameters,
        ).snapshot

    def read(
        self,
        *,
        active_operation_id: str,
        parameters: Mapping[str, Any],
    ) -> RealTcpPosePublishedRead:
        return self._capture(
            active_operation_id=active_operation_id,
            parameters=parameters,
        )

    def capture_and_publish(self, service: AgentGatewayService):
        if not isinstance(service, AgentGatewayService):
            raise TypeError("service must be an AgentGatewayService.")
        snapshot = self.capture_snapshot(
            active_operation_id=service.operation_store.active_operation_id,
        )
        return service.publish_observation(snapshot=snapshot, now=snapshot.available_at)


class ObservationOnlyDomainControlPort:
    """No-op domain port required by Gateway lifecycle, with no HOLD authority."""

    no_io = True

    def __init__(self, *, runtime_session_id: str, clock: Callable[[], float]) -> None:
        self.port_id = "agent-gateway:real-observe-no-domain-control:" + _required(
            runtime_session_id,
            field_name="runtime_session_id",
        )
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self._clock = clock
        self._lock = threading.Lock()
        self._next_sequence = 0

    def request_hold(
        self,
        *,
        operation: AgentSkillOperation | None,
        reason: str,
    ) -> AgentGatewayNoIoPortReceipt:
        if operation is not None and not isinstance(operation, AgentSkillOperation):
            raise TypeError("operation must be typed or None.")
        _required(reason, field_name="reason")
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
        return AgentGatewayNoIoPortReceipt(
            port_id=self.port_id,
            operation_kind="hold",
            operation_id=None if operation is None else operation.operation_id,
            succeeded=True,
            status="real_observe_has_no_domain_control_or_hold_authority",
            completed_at=float(self._clock()),
            call_sequence=sequence,
            evidence_refs=("real-observe-boundary:no-domain-control",),
        )


@dataclass
class RealTcpPoseObserveGatewaySession:
    contract_bundle: RealTcpPoseObserveContractBundle
    observation_producer: RealTcpPoseObservationProducer
    state_executor: PassiveReadSkillExecutor
    domain_control_port: ObservationOnlyDomainControlPort
    service: AgentGatewayService
    _close_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _close_outcome: Any = field(default=None, init=False, repr=False)

    def wire_adapter(
        self, *, authenticated_client_id: str
    ) -> BoundAgentGatewayWireAdapter:
        return BoundAgentGatewayWireAdapter(
            authenticated_client_id=authenticated_client_id,
            definition_catalog=self.contract_bundle.definition_catalog,
            schema_catalog=self.contract_bundle.schema_catalog,
            service=self.service,
            publish_observation=lambda: self.observation_producer.capture_and_publish(
                self.service
            ),
            current_station_revision=lambda: self.observation_producer.current_station_revision,
            operation_launcher=self.service.launch_operation,
        )

    def build_unix_json_server(
        self,
        *,
        socket_path: str,
        client_id_by_uid: Mapping[int, str],
        socket_mode: int = 0o600,
        max_frame_bytes: int = 2 * 1024 * 1024,
        peer_idle_timeout_sec: float = 10.0,
    ) -> AgentGatewayUnixJsonServer:
        peer_map = AgentGatewayUnixPeerClientMap(client_id_by_uid)
        if set(peer_map.client_id_by_uid.values()) != set(
            self.service.access_policy.allowed_client_ids
        ):
            raise ValueError("Unix peer map must exactly cover the access policy.")
        return AgentGatewayUnixJsonServer(
            socket_path=socket_path,
            peer_client_map=peer_map,
            facade_factory=lambda client_id: self.wire_adapter(
                authenticated_client_id=client_id
            ),
            close_gateway=self.close,
            socket_mode=socket_mode,
            max_frame_bytes=max_frame_bytes,
            peer_idle_timeout_sec=peer_idle_timeout_sec,
        )

    def close(self):
        with self._close_lock:
            if self._close_outcome is None:
                self._close_outcome = self.service.close(
                    current_station_revision=self.observation_producer.current_station_revision
                )
            return self._close_outcome


def assemble_real_tcp_pose_observe_gateway_session(
    *,
    contract_bundle: RealTcpPoseObserveContractBundle,
    source: RealTcpPosePublishedReadSource,
    allowed_client_ids: tuple[str, ...],
    clock: Callable[[], float] = time.monotonic,
    max_retained_operations: int = 32,
) -> RealTcpPoseObserveGatewaySession:
    """Assemble an attach-only session without starting or reading hardware."""

    if not isinstance(contract_bundle, RealTcpPoseObserveContractBundle):
        raise TypeError("contract_bundle must be typed.")
    if not callable(clock):
        raise TypeError("clock must be callable.")
    clients = tuple(
        _required(item, field_name="allowed_client_id") for item in allowed_client_ids
    )
    if not clients or len(set(clients)) != len(clients):
        raise ValueError("allowed_client_ids must be non-empty and unique.")
    retention = _positive_integer(
        max_retained_operations,
        field_name="max_retained_operations",
    )
    producer = RealTcpPoseObservationProducer(
        source=source,
        contract_bundle=contract_bundle,
    )
    entry = contract_bundle.route_manifest.route("state.read_tcp_pose", "1")

    def read_tcp_pose(request, operation) -> PassiveReadResult:
        published = producer.read(
            active_operation_id=operation.operation_id,
            parameters=request.parameters,
        )
        return PassiveReadResult(
            execution_result=published.execution_result,
            evidence_refs=published.evidence_refs,
            reason_code="passive_real_tcp_pose_read_completed",
        )

    executor = PassiveReadSkillExecutor(
        port_id="agent-gateway:real-passive-tcp-pose:"
        + contract_bundle.attachment.runtime_session_id,
        executor_id=contract_bundle.deployment_binding.executor_id,
        read=read_tcp_pose,
        current_station_revision=lambda: producer.current_station_revision,
        clock=clock,
        max_retained_operations=retention,
    )
    registry = SkillExecutorRegistry(
        manifest=contract_bundle.route_manifest,
        bindings=(
            SkillExecutorBinding.from_manifest_entry(
                manifest=contract_bundle.route_manifest,
                entry=entry,
                execution_port=executor,
            ),
        ),
    )
    domain_port = ObservationOnlyDomainControlPort(
        runtime_session_id=contract_bundle.attachment.runtime_session_id,
        clock=clock,
    )
    service = AgentGatewayService(
        manifest=contract_bundle.route_manifest,
        access_policy=AgentSkillAccessPolicy(
            access_policy_id="real-tcp-pose-observe:"
            + contract_bundle.attachment.runtime_session_id,
            allowed_client_ids=clients,
            allowed_modes=("real_observe",),
            max_request_timeout_ms=1_000,
            max_grant_lifetime_sec=1.0,
        ),
        executor_registry=registry,
        domain_control_port=domain_port,
        payload_validators=contract_bundle.payload_validators,
        operation_store=BoundedInMemoryAgentSkillOperationStore(
            max_retained_operations=retention
        ),
        clock=clock,
        clock_domain_id=contract_bundle.attachment.clock_domain_id,
    )
    return RealTcpPoseObserveGatewaySession(
        contract_bundle=contract_bundle,
        observation_producer=producer,
        state_executor=executor,
        domain_control_port=domain_port,
        service=service,
    )


__all__ = [
    "ObservationOnlyDomainControlPort",
    "REAL_TCP_POSE_STATE_EXECUTOR_ID_V1",
    "REAL_TCP_POSE_STATE_MAX_MEASUREMENT_AGE_MS_V1",
    "REAL_TCP_POSE_STATE_PROFILE_ID_V1",
    "REAL_TCP_POSE_STATE_ROUTE_ID_V1",
    "RealControlOwnerState",
    "RealPassiveTcpPoseStateDeploymentBinding",
    "RealTcpPoseMeasurementCut",
    "RealTcpPoseObservationProducer",
    "RealTcpPoseOwnerPublication",
    "RealTcpPoseOwnerPublicationBuffer",
    "RealTcpPoseOwnerPublicationPort",
    "RealTcpPoseObserveAttachmentSpec",
    "RealTcpPoseObserveContractBundle",
    "RealTcpPoseObserveGatewaySession",
    "RealTcpPosePublishedRead",
    "RealTcpPosePublishedReadSource",
    "assemble_real_tcp_pose_observe_gateway_session",
    "build_real_tcp_pose_observe_contract_bundle",
]
