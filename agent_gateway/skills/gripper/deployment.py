"""Immutable sim/real Route and deployment binding for gripper control."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaCatalog,
    SkillDefinitionCatalog,
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
    SkillRouteManifest,
    SkillRouteManifestEntry,
    validate_skill_route_catalog_closure,
)
from agent_gateway.runtime.payload_validation import (
    AgentSkillPayloadValidatorRegistry,
    AgentSkillPayloadValidatorSet,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

from .definition import (
    GRIPPER_COMPONENT_IDS,
    SET_GRIPPER_STATE_SKILL_ID_V1,
    set_gripper_state_schema_entry_v1,
    set_gripper_state_schema_entry_v2,
    set_gripper_state_skill_definition_v1,
    set_gripper_state_skill_definition_v2,
    validate_no_gripper_task_evaluation_v1,
    validate_set_gripper_state_parameters_v1,
    validate_set_gripper_state_parameters_v2,
    validate_set_gripper_state_result_for_parameters_v1,
    validate_set_gripper_state_result_for_parameters_v2,
    validate_set_gripper_state_result_v1,
    validate_set_gripper_state_result_v2,
)

SIM_GRIPPER_ROUTE_ID_V1 = "sim.mujoco.set_gripper_state"
REAL_GRIPPER_ROUTE_ID_V1 = "real.dmtac.set_gripper_state"


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


@dataclass(frozen=True)
class GripperSkillDeploymentBinding:
    runtime_session_id: str
    control_domain_id: str
    component_id: str
    target_binding_id: str
    target_kind: str
    current_mode: str
    executor_id: str
    port_binding_hash: str
    station_manifest_hash: str
    execution_manifest_hash: str
    skill_version: str = "1"
    port_binding_hash_by_component: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_session_id",
            "control_domain_id",
            "component_id",
            "target_binding_id",
            "executor_id",
            "port_binding_hash",
            "station_manifest_hash",
            "execution_manifest_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        expected_mode = {"sim": "sim_active", "real": "real_active"}
        if self.target_kind not in expected_mode:
            raise ValueError("target_kind must be 'sim' or 'real'.")
        if self.current_mode != expected_mode[self.target_kind]:
            raise ValueError("current_mode does not match target_kind.")
        if self.skill_version == "1":
            if (
                self.component_id != "right_gripper"
                or self.port_binding_hash_by_component is not None
            ):
                raise ValueError("v1 binds exactly the right_gripper component.")
        elif self.skill_version == "2":
            bindings = dict(self.port_binding_hash_by_component or {})
            if self.component_id != "grippers" or set(bindings) != set(
                GRIPPER_COMPONENT_IDS
            ):
                raise ValueError("v2 binds both left_gripper and right_gripper.")
            bindings = {
                key: _required(value, field_name="gripper port binding")
                for key, value in bindings.items()
            }
            object.__setattr__(
                self, "port_binding_hash_by_component", MappingProxyType(bindings)
            )
        else:
            raise ValueError("Unsupported gripper Skill version.")

    def port_hash_for(self, component_id: str) -> str:
        if self.skill_version == "1":
            if component_id != self.component_id:
                raise ValueError("port component does not match v1 binding.")
            return self.port_binding_hash
        return self.port_binding_hash_by_component[component_id]

    @property
    def route_id(self) -> str:
        return (
            SIM_GRIPPER_ROUTE_ID_V1
            if self.target_kind == "sim"
            else REAL_GRIPPER_ROUTE_ID_V1
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "component_id": self.component_id,
                "target_binding_id": self.target_binding_id,
                "target_kind": self.target_kind,
                "current_mode": self.current_mode,
                "executor_id": self.executor_id,
                "port_binding_hash": self.port_binding_hash,
                "station_manifest_hash": self.station_manifest_hash,
                "execution_manifest_hash": self.execution_manifest_hash,
                "route_id": self.route_id,
                **(
                    {
                        "skill_version": "2",
                        "port_binding_hash_by_component": dict(
                            self.port_binding_hash_by_component
                        ),
                    }
                    if self.skill_version == "2"
                    else {}
                ),
            },
            domain="dexteleop.agent_skill.gripper_deployment_binding.v1",
        )

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=(
                f"gripper:{self.target_kind}:{self.runtime_session_id}"
            ),
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.target_binding_id,
            target_kind=self.target_kind,
            execution_manifest_hash=self.execution_manifest_hash,
        )


@dataclass(frozen=True)
class GripperSkillContractBundle:
    definition_catalog: SkillDefinitionCatalog
    schema_catalog: AgentSkillSchemaCatalog
    route_manifest: SkillRouteManifest
    deployment_binding: GripperSkillDeploymentBinding
    payload_validators: AgentSkillPayloadValidatorRegistry

    def __post_init__(self) -> None:
        validate_skill_route_catalog_closure(
            definition_catalog=self.definition_catalog,
            schema_catalog=self.schema_catalog,
            route_manifest=self.route_manifest,
        )
        self.payload_validators.validate_manifest(self.route_manifest)
        entry = self.route_manifest.route(
            SET_GRIPPER_STATE_SKILL_ID_V1, self.deployment_binding.skill_version
        )
        if (
            entry.deployment_binding.deployment_binding_hash
            != self.deployment_binding.semantic_content_hash
        ):
            raise ValueError("manifest does not use this deployment binding.")


def build_gripper_skill_contract_bundle(
    binding: GripperSkillDeploymentBinding,
) -> GripperSkillContractBundle:
    if not isinstance(binding, GripperSkillDeploymentBinding):
        raise TypeError("binding must be a GripperSkillDeploymentBinding.")
    version = binding.skill_version
    definition = (
        set_gripper_state_skill_definition_v1
        if version == "1"
        else set_gripper_state_skill_definition_v2
    )()
    schema = (
        set_gripper_state_schema_entry_v1
        if version == "1"
        else set_gripper_state_schema_entry_v2
    )(
        skill_definition_hash=definition.semantic_content_hash,
    )
    definition_catalog = SkillDefinitionCatalog(
        catalog_id="dexteleop.agent_skill_definitions.gripper_control",
        catalog_version=version,
        skill_definitions=(definition,),
    )
    schema_catalog = AgentSkillSchemaCatalog(
        catalog_id="dexteleop.agent_skill_schemas.gripper_control",
        catalog_version=version,
        schema_entries=(schema,),
    )
    route = SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=binding.route_id,
        route_version=version,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        control_effect_class="motion",
        privilege_tier=(
            "t1_sim_or_no_motion" if binding.target_kind == "sim" else "t2_real_motion"
        ),
        supported_modes=(binding.current_mode,),
        target_kind=binding.target_kind,
        may_move_robot=True,
        may_open_publication=False,
        requires_operator_confirmation=binding.target_kind == "real",
        requires_deadman=False,
        # Deterministic MuJoCo advances one native physics tick per gripper
        # poll.  A detailed contact scene may need several seconds of wall
        # time to complete a short physical close.  Keep the real-device bound.
        max_operation_duration_ms=(15_000 if binding.target_kind == "sim" else 5_000),
        cancellation_policy="hold_on_cancel_disconnect_or_timeout",
        requires_task_evaluation=False,
    )
    availability_hash = canonical_content_hash(
        {
            "route_hash": route.semantic_content_hash,
            "deployment_binding_hash": binding.semantic_content_hash,
            "port_binding_hash": binding.port_binding_hash,
        },
        domain="dexteleop.agent_skill.gripper_route_availability.v1",
    )
    manifest = SkillRouteManifest(
        runtime_session_id=binding.runtime_session_id,
        control_domain_id=binding.control_domain_id,
        station_manifest_hash=binding.station_manifest_hash,
        execution_manifest_hash=binding.execution_manifest_hash,
        skill_definition_catalog_hash=definition_catalog.semantic_content_hash,
        skill_schema_catalog_hash=schema_catalog.semantic_content_hash,
        target_binding_id=binding.target_binding_id,
        target_kind=binding.target_kind,
        current_mode=binding.current_mode,
        supported_transport_features=("authoritative_port_receipt_v1",),
        skill_routes=(
            SkillRouteManifestEntry(
                route=route,
                deployment_binding=binding.to_ref(),
                evaluation_plan_hash=None,
                availability_hash=availability_hash,
            ),
        ),
        explicit_exclusions=(
            "arm_motion",
            "approach_then_close_composition",
            "agent_selected_force_or_speed",
        ),
    )
    validator = AgentSkillPayloadValidatorSet(
        validator_id=f"gripper-control:{binding.target_kind}:v{version}",
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=validate_set_gripper_state_parameters_v1
        if version == "1"
        else validate_set_gripper_state_parameters_v2,
        validate_execution_result=validate_set_gripper_state_result_v1
        if version == "1"
        else validate_set_gripper_state_result_v2,
        validate_task_evaluation=validate_no_gripper_task_evaluation_v1,
        validate_execution_result_for_parameters=(
            validate_set_gripper_state_result_for_parameters_v1
            if version == "1"
            else validate_set_gripper_state_result_for_parameters_v2
        ),
    )
    return GripperSkillContractBundle(
        definition_catalog=definition_catalog,
        schema_catalog=schema_catalog,
        route_manifest=manifest,
        deployment_binding=binding,
        payload_validators=AgentSkillPayloadValidatorRegistry(
            registry_id=f"gripper-control:{binding.runtime_session_id}",
            validators=(validator,),
        ),
    )


__all__ = [
    "REAL_GRIPPER_ROUTE_ID_V1",
    "SIM_GRIPPER_ROUTE_ID_V1",
    "GripperSkillContractBundle",
    "GripperSkillDeploymentBinding",
    "build_gripper_skill_contract_bundle",
]
