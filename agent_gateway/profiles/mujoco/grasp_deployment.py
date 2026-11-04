"""Pure registration closure for frozen observation and native fast grasping."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any

from agent_gateway.profiles.grasp_observation import OBSERVATION_SLOT
from agent_gateway.protocol._contract_validation import (
    finite_number,
    freeze_json_object,
    required_text,
    sha256_digest,
    unique_sorted_text,
)
from agent_gateway.protocol.skill_catalog import SkillRouteManifestEntry
from agent_gateway.protocol.skill_io_contracts import (
    SkillArtifactSlotSpec,
    inline_only_skill_io_contract,
)
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.payload_validation import (
    AgentSkillPayloadValidatorRegistry,
    AgentSkillPayloadValidatorSet,
)
from agent_gateway.skills.motion.definition import (
    execute_grasp_approach_plan_skill_definition_v1,
)
from agent_gateway.skills.motion.schemas import (
    grasp_approach_plan_execution_schema_entry_v1,
)
from agent_gateway.skills.perception.grasp_fast_plan_schemas import (
    fast_grasp_plan_schema_entry_v1,
    fast_grasp_plan_skill_definition_v1,
    fast_grasp_plan_skill_definition_v2,
    validate_fast_grasp_plan_parameters_v1,
    validate_fast_grasp_plan_result_for_parameters_v1,
    validate_fast_grasp_plan_result_v1,
)
from agent_gateway.skills.perception.grasp_observation import (
    grasp_observation_definition,
    grasp_observation_schema,
    validate_observation_parameters,
    validate_observation_result,
    validate_observation_result_for_parameters,
)
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.grasp_multiview_snapshot import (
    GRASP_MULTIVIEW_ENCODING,
    GRASP_MULTIVIEW_MAX_BYTES,
    GRASP_MULTIVIEW_SCHEMA_ID,
)
from dataset_contracts.perception_artifacts import (
    RGB_FRAME_COMPRESSION_V1,
    RGB_FRAME_ENCODING_V1,
    RGB_FRAME_SCHEMA_ID_V1,
)
from deployments.perception.grasp_first_path import (
    AXIAL_GRASP_SEARCH_ORDER,
    GRASP_SEARCH_PREFERENCE,
    FirstGraspPathPolicy,
)
from sim.mujoco.deploy.target_measurement import MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1

from .motion_deployment import (
    MujocoAgentSafeTcpPoseContract,
    MujocoAgentTcpPoseContractBundle,
    _motion_validators,
)
from .rgb_deployment import MujocoRgbCaptureRegistration

MUJOCO_CAPTURE_GRASP_OBSERVATION_ROUTE_ID_V1 = (
    "sim.mujoco.perception.capture_grasp_observation.v1"
)
MUJOCO_PLAN_GRASP_FAST_ROUTE_ID_V1 = "sim.mujoco.perception.plan_grasp_fast.v1"
MUJOCO_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1 = (
    "sim.mujoco.motion.execute_grasp_approach_plan.v1"
)
MUJOCO_CAPTURE_GRASP_OBSERVATION_EXECUTOR_ID_V1 = "mujoco-grasp-observation-v1"
MUJOCO_PLAN_GRASP_FAST_EXECUTOR_ID_V1 = "mujoco-fast-grasp-v1"
MUJOCO_EXECUTE_GRASP_APPROACH_PLAN_EXECUTOR_ID_V1 = "mujoco-grasp-plan-execution-v1"
MUJOCO_GRASP_APPROACH_PLAN_ID_PREFIX_V1 = "mujoco-grasp-approach-plan"
MUJOCO_FAST_GRASP_MIN_SPEED_SCALE_V1 = 0.05
MUJOCO_FAST_GRASP_MAX_SPEED_SCALE_V1 = 1.0
MUJOCO_FAST_GRASP_PLAN_TTL_SEC_V1 = 30.0


def _approach_config(value: Mapping[str, Any]) -> Mapping[str, Any]:
    config = freeze_json_object(value, field_name="approach_config")
    if set(config) != {
        "pregrasp_distance_m",
        "near_distance_m",
        "terminal_offset_m",
        "terminal_speed_scale",
    }:
        raise ValueError(
            "approach_config must contain the four native approach settings."
        )
    for name, item in config.items():
        finite_number(item, field_name=name)
    if not 0 < config["near_distance_m"] < config["pregrasp_distance_m"]:
        raise ValueError("Approach requires ordered positive pregrasp/near distances.")
    if not 0 < config["terminal_speed_scale"] <= 1:
        raise ValueError("terminal_speed_scale must be in (0, 1].")
    return config


@dataclass(frozen=True)
class MujocoFastGraspRegistration:
    skill: SkillRegistrationTemplate
    entry: SkillRouteManifestEntry
    payload_validator: AgentSkillPayloadValidatorSet
    executor_id: str

    def __post_init__(self) -> None:
        if self.skill.routes != (self.entry.route,):
            raise ValueError(
                "Grasp registration must advertise its exact single Route."
            )
        self.payload_validator.validate_route(self.entry.route)
        required_text(self.executor_id, field_name="executor_id")


@dataclass(frozen=True)
class MujocoFastGraspContract:
    observation: MujocoFastGraspRegistration
    fast: MujocoFastGraspRegistration
    execute: MujocoFastGraspRegistration
    execution_contract: MujocoAgentSafeTcpPoseContract
    profile_hash: str
    frame_binding_hash: str
    roles: tuple[str, ...]
    policy: FirstGraspPathPolicy
    approach_config: Mapping[str, Any]
    source_by_role: Mapping[str, str]
    model_profile_hash: str
    target_prompt: str | None = None

    def __post_init__(self) -> None:
        for name in ("profile_hash", "frame_binding_hash", "model_profile_hash"):
            sha256_digest(getattr(self, name), field_name=name)
        roles = unique_sorted_text(self.roles, field_name="roles", allow_empty=False)
        if len(roles) > 3 or set(self.source_by_role) != set(roles):
            raise ValueError(
                "Frozen grasp capture requires one to three exact RGB roles."
            )
        sources = {
            role: required_text(self.source_by_role[role], field_name="source_id")
            for role in roles
        }
        if len(set(sources.values())) != len(sources):
            raise ValueError("Frozen grasp RGB sources must be unique.")
        if self.policy != FirstGraspPathPolicy():
            raise ValueError(
                "Fast grasp requires the fixed first_acceptable_v1 policy."
            )
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "source_by_role", MappingProxyType(sources))
        object.__setattr__(
            self, "approach_config", _approach_config(self.approach_config)
        )
        native = self.execution_contract
        if (
            self.execute.entry.deployment_binding != native.deployment_binding.to_ref()
            or self.execute.executor_id != native.deployment_binding.executor_id
            or self.execute.entry.evaluation_plan_hash
            != native.evaluation_plan.semantic_content_hash
            or native.evaluation_plan.route_hash
            != self.execute.entry.route.semantic_content_hash
            or native.evaluation_plan.deployment_binding_hash
            != native.deployment_binding.semantic_content_hash
        ):
            raise ValueError(
                "Native grasp execution binding/evaluation closure differs."
            )


def _preconditions(bundle: MujocoAgentTcpPoseContractBundle) -> None:
    manifest = bundle.route_manifest
    if manifest.target_kind != "sim" or manifest.current_mode != "sim_active":
        raise ValueError("Fast grasp requires the native sim_active owner.")
    rgb = bundle.rgb_capture
    approach = bundle.grasp_approach_contract
    contact = bundle.grasp_contact_contract
    hand = bundle.gripper_contract
    if rgb is None:
        raise ValueError("Fast grasp requires an existing RGB Artifact registration.")
    if not isinstance(rgb, MujocoRgbCaptureRegistration):
        raise TypeError("rgb_capture must be a native RGB Artifact registration.")
    if not rgb.registration.io_contract.output_artifact_slots:
        raise ValueError("Fast grasp RGB registration requires Artifact output.")
    if (
        not isinstance(approach, MujocoAgentSafeTcpPoseContract)
        or not isinstance(contact, MujocoAgentSafeTcpPoseContract)
        or contact.contact_binding is None
    ):
        raise ValueError(
            "Fast grasp requires native approach and target-bound contact."
        )
    if (
        hand is None
        or hand.deployment_binding.skill_version not in {"2", "3"}
        or set(hand.deployment_binding.port_binding_hash_by_component or {})
        != {"left_gripper", "right_gripper"}
    ):
        raise ValueError("Fast grasp requires both native grippers.")
    for binding in (
        bundle.deployment_binding,
        rgb.deployment_binding,
        approach.deployment_binding,
        contact.deployment_binding,
        hand.deployment_binding,
    ):
        for name in (
            "runtime_session_id",
            "control_domain_id",
            "target_binding_id",
            "station_manifest_hash",
            "execution_manifest_hash",
        ):
            if getattr(binding, name) != getattr(manifest, name):
                raise ValueError(f"Fast grasp native owner mismatch: {name}.")
    for native in (approach, contact):
        binding = native.deployment_binding
        if (
            binding.component_id != "right_arm"
            or native.motion_profile.component_id != "right_arm"
        ):
            raise ValueError("Native fast grasp requires the right_arm component.")
        for name in (
            "execution_domain_binding_hash",
            "target_runtime_action_contract_hash",
            "component_target_capability_hash",
            "action_schema_hash",
            "observation_binding_id",
            "observation_binding_hash",
            "observation_plan_hash",
            "observation_capability_hash",
            "pose_schema_hash",
        ):
            if getattr(binding, name) != getattr(bundle.deployment_binding, name):
                raise ValueError(f"Fast grasp native owner mismatch: {name}.")
        for name in (
            "reference_frame_id",
            "reference_frame_spec_hash",
            "execution_reference_frame_id",
            "execution_reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
        ):
            if getattr(native.motion_profile, name) != getattr(
                approach.motion_profile, name
            ):
                raise ValueError(f"Fast grasp native frame mismatch: {name}.")
    if not approach.deployment_binding.safety_envelope_refs or not any(
        MUJOCO_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1 in ref.allowed_route_ids
        for ref in approach.deployment_binding.safety_envelope_refs
    ):
        raise ValueError(
            "Native approach envelope must already allow grasp-plan execution."
        )


def _route(*, base, definition, schema, io, route_id, duration_ms):
    return replace(
        base,
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=route_id,
        route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        skill_io_contract_hash=io.semantic_content_hash,
        max_operation_duration_ms=duration_ms,
    )


def _passive_registration(
    *,
    base_entry,
    definition,
    schema,
    io,
    route_id,
    executor_id,
    profile_hash,
    duration_ms,
    parameters,
    result,
    contextual,
):
    route = _route(
        base=base_entry.route,
        definition=definition,
        schema=schema,
        io=io,
        route_id=route_id,
        duration_ms=duration_ms,
    )
    route = replace(
        route,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        cancellation_policy="cancel_only",
        requires_task_evaluation=False,
    )
    binding = replace(
        base_entry.deployment_binding,
        deployment_binding_id=route_id
        + ":"
        + base_entry.deployment_binding.deployment_binding_id,
        deployment_binding_hash=canonical_content_hash(
            {
                "source_binding_hash": base_entry.deployment_binding.deployment_binding_hash,
                "route_hash": route.semantic_content_hash,
                "profile_hash": profile_hash,
                "executor_id": executor_id,
            },
            domain="dexteleop.mujoco.fast_grasp.passive_binding.v1",
        ),
        safety_envelope_refs=(),
    )
    validator = AgentSkillPayloadValidatorSet(
        validator_id=route_id + ":" + profile_hash,
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        route_id=route_id,
        route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        validate_parameters=parameters,
        validate_execution_result=result,
        validate_execution_result_for_parameters=contextual,
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    return MujocoFastGraspRegistration(
        skill=SkillRegistrationTemplate(definition, schema, io, (route,)),
        entry=SkillRouteManifestEntry(
            route=route,
            deployment_binding=binding,
            evaluation_plan_hash=None,
            availability_hash=canonical_content_hash(
                {
                    "source_availability_hash": base_entry.availability_hash,
                    "binding_hash": binding.deployment_binding_hash,
                    "requires": "native_frozen_rgbd_and_prepared_model",
                },
                domain="dexteleop.mujoco.fast_grasp.passive_availability.v1",
            ),
        ),
        payload_validator=validator,
        executor_id=executor_id,
    )


def _inline_io(definition, schema):
    return inline_only_skill_io_contract(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
    )


def build_mujoco_fast_grasp_contract(
    bundle: MujocoAgentTcpPoseContractBundle,
    *,
    model_profile_hash: str,
    approach_config: Mapping[str, Any],
    target_prompt: str | None = None,
) -> MujocoAgentTcpPoseContractBundle:
    """Append three closed registrations without starting or acquiring resources."""
    model_hash = sha256_digest(model_profile_hash, field_name="model_profile_hash")
    config = _approach_config(approach_config)
    _preconditions(bundle)
    rgb = bundle.rgb_capture
    approach = bundle.grasp_approach_contract
    contact = bundle.grasp_contact_contract
    profile = approach.motion_profile
    target_object_id = None
    named_binding = {}
    if target_prompt is not None:
        target_prompt = required_text(target_prompt, field_name="target_prompt", maximum_length=256)
        segmentation = bundle.segmentation
        if segmentation is None:
            raise ValueError("Named fast grasp requires the existing segmentation deployment.")
        target_object_id = contact.contact_binding.target_body_name
        named_binding = {
            "target_object_id": target_object_id,
            "target_prompt": target_prompt,
            "segmentation_binding_hash": segmentation.binding.semantic_content_hash,
            "pixel_selection": "same_frozen_rgbd_mask_interior",
        }
    if (
        not 0
        < config["near_distance_m"] + config["terminal_offset_m"]
        <= contact.contact_binding.max_approach_m
    ):
        raise ValueError("Near pose is outside the target-bound contact corridor.")
    roles = rgb.profile.roles
    policy = FirstGraspPathPolicy()
    frames = {
        name: getattr(profile, name)
        for name in (
            "reference_frame_id",
            "reference_frame_spec_hash",
            "execution_reference_frame_id",
            "execution_reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
        )
    }
    frame_hash = canonical_content_hash(
        {
            "target_kind": "sim",
            "source_geometry": "native_mujoco_scene_capture",
            "scene_guard_metadata": "private_state_digest_only.v1",
            "source_frame_id": MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1,
            "camera_geometry_owner": "mujoco_scene_camera_metadata",
            "source_catalog": rgb.profile.sources,
            "rgb_profile_hash": rgb.profile.semantic_content_hash,
            "rgb_binding_hash": rgb.deployment_binding.semantic_content_hash,
            "runtime_session_id": bundle.route_manifest.runtime_session_id,
            "control_domain_id": bundle.route_manifest.control_domain_id,
            "target_binding_id": bundle.route_manifest.target_binding_id,
            "station_manifest_hash": bundle.route_manifest.station_manifest_hash,
            "execution_manifest_hash": bundle.route_manifest.execution_manifest_hash,
            "execution_domain_binding_hash": approach.deployment_binding.execution_domain_binding_hash,
            "observation_binding_hash": approach.deployment_binding.observation_binding_hash,
            "controlled_frame_id": "tool_tcp",
            **frames,
        },
        domain="dexteleop.mujoco.fast_grasp.frame_binding.v1",
    )
    profile_hash = canonical_content_hash(
        {
            "model_profile_hash": model_hash,
            "source_catalog": rgb.profile.sources,
            "rgb_profile_hash": rgb.profile.semantic_content_hash,
            "rgb_binding_hash": rgb.deployment_binding.semantic_content_hash,
            "native_approach_binding_hash": approach.deployment_binding.semantic_content_hash,
            "native_approach_profile_hash": profile.semantic_content_hash,
            "native_contact_binding_hash": contact.deployment_binding.semantic_content_hash,
            "native_contact_profile_hash": contact.motion_profile.semantic_content_hash,
            "target_contact_hash": contact.contact_binding.semantic_content_hash,
            "native_gripper_binding_hash": bundle.gripper_contract.deployment_binding.semantic_content_hash,
            "station_manifest_hash": bundle.route_manifest.station_manifest_hash,
            "execution_manifest_hash": bundle.route_manifest.execution_manifest_hash,
            "runtime_session_id": bundle.route_manifest.runtime_session_id,
            "control_domain_id": bundle.route_manifest.control_domain_id,
            "target_binding_id": bundle.route_manifest.target_binding_id,
            "frame_binding_hash": frame_hash,
            "approach_config": config,
            "planning_policy": "first_acceptable_v1",
            "search_policy": asdict(policy),
            "search_preference": GRASP_SEARCH_PREFERENCE,
            "refine_grasp_centers": True,
            "refine_grasp_axial_depth": True,
            "axial_search_order": AXIAL_GRASP_SEARCH_ORDER,
            "min_speed_scale": MUJOCO_FAST_GRASP_MIN_SPEED_SCALE_V1,
            "max_speed_scale": MUJOCO_FAST_GRASP_MAX_SPEED_SCALE_V1,
            "plan_ttl_sec": MUJOCO_FAST_GRASP_PLAN_TTL_SEC_V1,
            "plan_id_prefix": MUJOCO_GRASP_APPROACH_PLAN_ID_PREFIX_V1,
            **named_binding,
        },
        domain="dexteleop.mujoco.fast_grasp.profile.v1",
    )
    rgb_entry = bundle.route_manifest.route(
        rgb.registration.definition.skill_id, rgb.registration.definition.skill_version
    )
    definition = grasp_observation_definition()
    schema = grasp_observation_schema(
        definition_hash=definition.semantic_content_hash,
        roles=roles,
        profile_hash=profile_hash,
    )
    io = replace(
        _inline_io(definition, schema),
        output_artifact_slots=(
            SkillArtifactSlotSpec(
                slot_id=OBSERVATION_SLOT,
                direction="output",
                min_count=1,
                max_count=1,
                allowed_media_types=("application/zip",),
                allowed_schema_ids=(GRASP_MULTIVIEW_SCHEMA_ID,),
                allowed_encodings=(GRASP_MULTIVIEW_ENCODING,),
                allowed_compressions=("zip",),
                max_artifact_bytes=GRASP_MULTIVIEW_MAX_BYTES,
                max_total_bytes=GRASP_MULTIVIEW_MAX_BYTES,
            ),
            SkillArtifactSlotSpec(
                slot_id="rgb_frames",
                direction="output",
                min_count=1,
                max_count=min(3, len(roles)),
                allowed_media_types=("image/jpeg",),
                allowed_schema_ids=(RGB_FRAME_SCHEMA_ID_V1,),
                allowed_encodings=(RGB_FRAME_ENCODING_V1,),
                allowed_compressions=(RGB_FRAME_COMPRESSION_V1,),
                max_artifact_bytes=4 * 1024 * 1024,
                max_total_bytes=12 * 1024 * 1024,
            ),
        ),
    )
    observation = _passive_registration(
        base_entry=rgb_entry,
        definition=definition,
        schema=schema,
        io=io,
        route_id=MUJOCO_CAPTURE_GRASP_OBSERVATION_ROUTE_ID_V1,
        executor_id=MUJOCO_CAPTURE_GRASP_OBSERVATION_EXECUTOR_ID_V1,
        profile_hash=profile_hash,
        duration_ms=10_000,
        parameters=lambda value: validate_observation_parameters(value, roles=roles),
        result=lambda value: validate_observation_result(value, roles=roles),
        contextual=validate_observation_result_for_parameters,
    )
    define_fast_grasp = (fast_grasp_plan_skill_definition_v1 if target_object_id is None
                         else fast_grasp_plan_skill_definition_v2)
    definition = define_fast_grasp(
        controlled_component_ids=(profile.component_id,)
    )
    schema = fast_grasp_plan_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id="mujoco-first-acceptable-grasp-v" + definition.skill_version,
        parameter_profile_hash=profile_hash,
        component_id=profile.component_id,
        side="right",
        roles=roles,
        min_speed_scale=MUJOCO_FAST_GRASP_MIN_SPEED_SCALE_V1,
        max_speed_scale=MUJOCO_FAST_GRASP_MAX_SPEED_SCALE_V1,
        target_object_id=target_object_id,
        **{
            name: frames[name]
            for name in (
                "reference_frame_id",
                "reference_frame_spec_hash",
                "frame_graph_hash",
                "tool_config_hash",
                "calibration_hash",
            )
        },
    )
    fast = _passive_registration(
        base_entry=rgb_entry,
        definition=definition,
        schema=schema,
        io=_inline_io(definition, schema),
        route_id=(MUJOCO_PLAN_GRASP_FAST_ROUTE_ID_V1 if target_object_id is None
                  else "sim.mujoco.perception.plan_grasp_fast.v2"),
        executor_id=MUJOCO_PLAN_GRASP_FAST_EXECUTOR_ID_V1,
        profile_hash=profile_hash,
        duration_ms=45_000,
        parameters=lambda value: validate_fast_grasp_plan_parameters_v1(
            value, schema.parameter_schema
        ),
        result=lambda value: validate_fast_grasp_plan_result_v1(
            value, schema.execution_result_schema
        ),
        contextual=validate_fast_grasp_plan_result_for_parameters_v1,
    )
    definition = execute_grasp_approach_plan_skill_definition_v1(
        controlled_component_ids=(profile.component_id,)
    )
    execution_schema = grasp_approach_plan_execution_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id="mujoco-grasp-plan-execution-v1",
        parameter_profile_hash=profile_hash,
        allowed_component_ids=(profile.component_id,),
        plan_id_prefix=MUJOCO_GRASP_APPROACH_PLAN_ID_PREFIX_V1,
    )
    io = _inline_io(definition, execution_schema)
    route = _route(
        base=bundle.route_manifest.route("motion.approach_grasp_pose", "1").route,
        definition=definition,
        schema=execution_schema,
        io=io,
        route_id=MUJOCO_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1,
        duration_ms=bundle.route_manifest.route(
            "motion.approach_grasp_pose", "1"
        ).route.max_operation_duration_ms,
    )
    binding = replace(
        approach.deployment_binding,
        deployment_binding_id="sim.mujoco.grasp_plan_execution:"
        + bundle.route_manifest.runtime_session_id,
        route_hash=route.semantic_content_hash,
        executor_id=MUJOCO_EXECUTE_GRASP_APPROACH_PLAN_EXECUTOR_ID_V1,
    )
    evaluation = replace(
        approach.evaluation_plan,
        skill_definition_hash=definition.semantic_content_hash,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=binding.semantic_content_hash,
        task_outcome_schema_hash=execution_schema.task_evaluation_schema_hash,
    )
    execution_contract = replace(
        approach, deployment_binding=binding, evaluation_plan=evaluation
    )

    def validate_execution_parameters(value):
        if not isinstance(value, Mapping) or set(value) != {
            "component_id",
            "grasp_approach_plan_id",
            "grasp_approach_plan_hash",
        }:
            raise ValueError(
                "Grasp execution requires exactly a component and immutable plan id/hash."
            )
        if value["component_id"] != profile.component_id:
            raise ValueError(
                "Grasp execution component differs from the native profile."
            )
        identifier = value["grasp_approach_plan_id"]
        if not isinstance(identifier, str) or not re.fullmatch(
            MUJOCO_GRASP_APPROACH_PLAN_ID_PREFIX_V1 + r":[0-9a-f]{64}", identifier
        ):
            raise ValueError("Grasp execution requires a native MuJoCo plan id.")
        sha256_digest(
            value["grasp_approach_plan_hash"], field_name="grasp_approach_plan_hash"
        )

    execute = MujocoFastGraspRegistration(
        skill=SkillRegistrationTemplate(definition, execution_schema, io, (route,)),
        entry=SkillRouteManifestEntry(
            route=route,
            deployment_binding=binding.to_ref(),
            evaluation_plan_hash=evaluation.semantic_content_hash,
            availability_hash=canonical_content_hash(
                {
                    "native_approach_availability_hash": bundle.route_manifest.route(
                        "motion.approach_grasp_pose", "1"
                    ).availability_hash,
                    "profile_hash": profile_hash,
                    "binding_hash": binding.semantic_content_hash,
                },
                domain="dexteleop.mujoco.fast_grasp.execution_availability.v1",
            ),
        ),
        payload_validator=replace(
            _motion_validators(
                route=route, schema_entry=execution_schema, profile=profile
            ),
            validator_id="mujoco-grasp-plan-execution:" + profile_hash,
            validate_parameters=validate_execution_parameters,
        ),
        executor_id=binding.executor_id,
    )
    contract = MujocoFastGraspContract(
        observation,
        fast,
        execute,
        execution_contract,
        profile_hash,
        frame_hash,
        roles,
        policy,
        config,
        rgb.profile.source_id_by_role,
        model_hash,
        target_prompt,
    )
    registrations = (observation, fast, execute)
    definitions = replace(
        bundle.definition_catalog,
        skill_definitions=tuple(
            sorted(
                (
                    *bundle.definition_catalog.skill_definitions,
                    *(r.skill.definition for r in registrations),
                ),
                key=lambda item: (item.skill_id, item.skill_version),
            )
        ),
    )
    schemas = replace(
        bundle.schema_catalog,
        schema_entries=tuple(
            sorted(
                (
                    *bundle.schema_catalog.schema_entries,
                    *(r.skill.schema_entry for r in registrations),
                ),
                key=lambda item: (item.skill_id, item.skill_version),
            )
        ),
    )
    io_catalog = replace(
        bundle.io_catalog,
        io_contracts=tuple(
            sorted(
                (
                    *bundle.io_catalog.io_contracts,
                    *(r.skill.io_contract for r in registrations),
                ),
                key=lambda item: (item.skill_id, item.skill_version),
            )
        ),
    )
    manifest = replace(
        bundle.route_manifest,
        skill_routes=tuple(
            sorted(
                (
                    *bundle.route_manifest.skill_routes,
                    *(r.entry for r in registrations),
                ),
                key=lambda item: (
                    item.route.skill_id,
                    item.route.skill_version,
                    item.route.route_id,
                    item.route.route_version,
                ),
            )
        ),
        skill_definition_catalog_hash=definitions.semantic_content_hash,
        skill_schema_catalog_hash=schemas.semantic_content_hash,
        skill_io_catalog_hash=io_catalog.semantic_content_hash,
    )
    validators = AgentSkillPayloadValidatorRegistry(
        registry_id=bundle.payload_validators.registry_id,
        validators=tuple(
            sorted(
                (
                    *(
                        bundle.payload_validators.validator_for(entry.route)
                        for entry in bundle.route_manifest.skill_routes
                    ),
                    *(r.payload_validator for r in registrations),
                ),
                key=lambda item: item.route_key,
            )
        ),
    )
    return replace(
        bundle,
        definition_catalog=definitions,
        schema_catalog=schemas,
        io_catalog=io_catalog,
        route_manifest=manifest,
        payload_validators=validators,
        fast_grasp_contract=contract,
    )
