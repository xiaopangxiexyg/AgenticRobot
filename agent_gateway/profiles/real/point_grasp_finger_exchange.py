"""Explicit v5 binding; old live and frozen registrations remain unchanged."""

from dataclasses import replace

from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.skills.perception.point_grasp_finger_exchange import (
    FINGER_EXCHANGE_SKILL_VERSION,
    finger_exchange_point_grasp_contract,
    legacy_candidate_validation_projection,
)
from dataset_contracts.canonical_encoding import canonical_content_hash


def upgrade_finger_exchange_point_grasp_registration(base):
    if not base.profile.finger_exchange:
        raise ValueError("The deployment must explicitly enable finger exchange.")
    definition, schema = finger_exchange_point_grasp_contract(
        base.skill.definition,
        base.skill.schema_entry,
        profile_hash=base.profile.semantic_content_hash,
    )
    io_contract = replace(
        base.skill.io_contract,
        skill_version=FINGER_EXCHANGE_SKILL_VERSION,
        skill_definition_hash=definition.semantic_content_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
    )
    route = replace(
        base.skill.routes[0],
        skill_version=FINGER_EXCHANGE_SKILL_VERSION,
        route_version=FINGER_EXCHANGE_SKILL_VERSION,
        skill_definition_hash=definition.semantic_content_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        skill_io_contract_hash=io_contract.semantic_content_hash,
    )
    binding = replace(
        base.binding,
        deployment_binding_version=FINGER_EXCHANGE_SKILL_VERSION,
        route_hash=route.semantic_content_hash,
        executor_id="real.passive.finger_exchange_point_grasp.v5",
    )
    validator = replace(
        base.payload_validator,
        skill_version=FINGER_EXCHANGE_SKILL_VERSION,
        route_version=FINGER_EXCHANGE_SKILL_VERSION,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        validate_execution_result=lambda payload: (
            base.payload_validator.validate_execution_result(
                legacy_candidate_validation_projection(payload)
            )
        ),
        validate_execution_result_for_parameters=lambda parameters, result: (
            base.payload_validator.validate_execution_result_for_parameters(
                parameters, legacy_candidate_validation_projection(result)
            )
        ),
    )
    return replace(
        base,
        binding=binding,
        payload_validator=validator,
        skill=SkillRegistrationTemplate(
            definition=definition,
            schema_entry=schema,
            io_contract=io_contract,
            routes=(route,),
        ),
        availability_hash=canonical_content_hash(
            {
                "route": route.semantic_content_hash,
                "binding": binding.semantic_content_hash,
            },
            domain="dexteleop.finger_exchange_grasp.availability.v5",
        ),
    )
