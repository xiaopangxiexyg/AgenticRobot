"""Exact immutable deployment artifacts for the CubeLift application.

This module projects one already-constructed policy-only Sim-HIL graph into
the Agent Skill catalogs, Route manifest, evaluator plan, and lower deployment
binding.  It performs no start, inference, physics step, authority transition,
dispatch, or trace write.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from control_station.agent_gateway.protocol.evaluation_contracts import (
    SkillEvaluationPlan,
)
from control_station.agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaCatalog,
    SkillDefinitionCatalog,
    SkillExecutionRoute,
    SkillRouteManifest,
    SkillRouteManifestEntry,
    validate_skill_route_catalog_closure,
)
from control_station.agent_gateway.protocol.state_schemas import (
    tcp_pose_state_definition_v1,
    tcp_pose_state_schema_entry_v1,
    validate_agent_state_route_schema_v1,
)
from control_station.agent_gateway.runtime.payload_validation import (
    AgentSkillPayloadValidatorRegistry,
)
from control_station.composition.mujoco_sim_hil import (
    build_mujoco_sim_hil_agent_skill_execution_manifest,
    build_mujoco_sim_hil_agent_skill_station_manifest,
)
from control_station.tasks.cube_lift.definition import (
    cube_lift_schema_entry_v1,
    cube_lift_skill_definition_v1,
)
from control_station.tasks.cube_lift.sim_route import (
    CubeLiftSimDeploymentBinding,
    CubeLiftTerminalEvaluatorConfig,
    cube_lift_evaluation_plan_v1,
    cube_lift_payload_validator_set_v1,
    cube_lift_sim_route_v1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

from ...agent.state_deployment import (
    MUJOCO_TCP_POSE_STATE_AVAILABILITY_HASH_DOMAIN_V1,
    MUJOCO_TCP_POSE_STATE_EXECUTOR_ID_V1,
    MUJOCO_TCP_POSE_STATE_MAX_MEASUREMENT_AGE_MS_V1,
    MUJOCO_TCP_POSE_STATE_PARAMETER_PROFILE_ID_V1,
    MUJOCO_TCP_POSE_STATE_PROFILE_HASH_DOMAIN_V1,
    MUJOCO_TCP_POSE_STATE_ROUTE_ID_V1,
    MujocoPassiveTcpPoseStateDeploymentBinding,
    build_mujoco_tcp_pose_state_payload_validator_set,
)

MUJOCO_CUBE_LIFT_AGENT_SKILL_EXECUTOR_ID_V1 = (
    "control-station.mujoco-cube-lift-policy-only.v1"
)
MUJOCO_CUBE_LIFT_AGENT_SKILL_CALIBRATION_HASH_DOMAIN_V1 = (
    "dexteleop.sim_hil.mujoco_cube_lift_scene_calibration.v1"
)
MUJOCO_CUBE_LIFT_AGENT_SKILL_AVAILABILITY_HASH_DOMAIN_V1 = (
    "dexteleop.sim_hil.mujoco_cube_lift_route_availability.v1"
)
def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _hash(value: object, *, field_name: str) -> str:
    return _required(
        getattr(value, "semantic_content_hash", None),
        field_name=field_name,
    )


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _default_checkpoint_hash(path: str | Path) -> str:
    from sim.mujoco_temp_pipeline.checkpoint import file_sha256

    return file_sha256(path)


@dataclass(frozen=True)
class MujocoCubeLiftAgentSkillContractBundle:
    """Closed immutable public and lower-owned artifacts for one session."""

    definition_catalog: SkillDefinitionCatalog
    schema_catalog: AgentSkillSchemaCatalog
    route_manifest: SkillRouteManifest
    deployment_binding: CubeLiftSimDeploymentBinding
    state_deployment_binding: MujocoPassiveTcpPoseStateDeploymentBinding
    evaluator_config: CubeLiftTerminalEvaluatorConfig
    evaluation_plan: SkillEvaluationPlan
    payload_validators: AgentSkillPayloadValidatorRegistry
    station_manifest: Any
    execution_manifest: Any

    def __post_init__(self) -> None:
        if not isinstance(self.definition_catalog, SkillDefinitionCatalog):
            raise TypeError("definition_catalog must be typed.")
        if not isinstance(self.schema_catalog, AgentSkillSchemaCatalog):
            raise TypeError("schema_catalog must be typed.")
        if not isinstance(self.route_manifest, SkillRouteManifest):
            raise TypeError("route_manifest must be typed.")
        if not isinstance(self.deployment_binding, CubeLiftSimDeploymentBinding):
            raise TypeError("deployment_binding must be typed.")
        if not isinstance(
            self.state_deployment_binding,
            MujocoPassiveTcpPoseStateDeploymentBinding,
        ):
            raise TypeError("state_deployment_binding must be typed.")
        if not isinstance(self.evaluator_config, CubeLiftTerminalEvaluatorConfig):
            raise TypeError("evaluator_config must be typed.")
        if not isinstance(self.evaluation_plan, SkillEvaluationPlan):
            raise TypeError("evaluation_plan must be typed.")
        if not isinstance(
            self.payload_validators,
            AgentSkillPayloadValidatorRegistry,
        ):
            raise TypeError("payload_validators must be typed.")
        validate_skill_route_catalog_closure(
            definition_catalog=self.definition_catalog,
            schema_catalog=self.schema_catalog,
            route_manifest=self.route_manifest,
        )
        advertised = self.route_manifest.skill_routes
        if len(advertised) != 2:
            raise ValueError(
                "CubeLift bundle must advertise motion and passive state Routes."
            )
        entry = self.route_manifest.route(
            "task.cube_lift",
            "1",
        )
        if (
            entry.deployment_binding.deployment_binding_hash
            != self.deployment_binding.semantic_content_hash
        ):
            raise ValueError("Route manifest uses another deployment binding.")
        if (
            entry.evaluation_plan_hash
            != getattr(self.evaluation_plan, "semantic_content_hash", None)
        ):
            raise ValueError("Route manifest uses another evaluation plan.")
        state_entry = self.route_manifest.route(
            "state.read_tcp_pose",
            "1",
        )
        if state_entry.deployment_binding.deployment_binding_hash != (
            self.state_deployment_binding.semantic_content_hash
        ):
            raise ValueError("State Route manifest uses another binding.")
        if state_entry.evaluation_plan_hash is not None:
            raise ValueError("Passive state Route cannot bind task evaluation.")
        self.payload_validators.validate_manifest(self.route_manifest)


def cube_lift_terminal_evaluator_config_from_task(
    task_config: Any,
    *,
    stable_hold_physics_ticks: int | None = None,
) -> CubeLiftTerminalEvaluatorConfig:
    """Project task-owned geometry while keeping the hold window explicit.

    With no override, the existing task profile's hold length is reused as a
    duration fact.  Success is still re-measured after the lower HOLD and does
    not consume the task kernel's private success counter.
    """

    configured_ticks = (
        getattr(task_config, "success_hold_physics_ticks", None)
        if stable_hold_physics_ticks is None
        else stable_hold_physics_ticks
    )
    return CubeLiftTerminalEvaluatorConfig(
        task_config_hash=_required(
            getattr(task_config, "source_sha256", None),
            field_name="task_config.source_sha256",
        ),
        object_body_name=_required(
            getattr(task_config, "object_body_name", None),
            field_name="task_config.object_body_name",
        ),
        left_finger_body_name=_required(
            getattr(task_config, "grasp_latch_left_finger_body_name", None),
            field_name="task_config.grasp_latch_left_finger_body_name",
        ),
        right_finger_body_name=_required(
            getattr(task_config, "grasp_latch_right_finger_body_name", None),
            field_name="task_config.grasp_latch_right_finger_body_name",
        ),
        table_body_name=_required(
            getattr(task_config, "table_body_name", None),
            field_name="task_config.table_body_name",
        ),
        lift_delta_m=getattr(task_config, "lift_delta_m", None),
        stable_hold_physics_ticks=configured_ticks,
        physics_hz=getattr(task_config, "physics_hz", None),
        success_requires_bilateral_contact=True,
        requires_no_table_contact=True,
    )


def _validate_exact_application(
    application: Any,
    *,
    evaluator_config: CubeLiftTerminalEvaluatorConfig,
) -> dict[str, str]:
    app_config = getattr(application, "config", None)
    if (
        float(getattr(app_config, "initial_human_alpha", -1.0)) != 0.0
        or getattr(app_config, "defer_human_input", None) is not True
        or getattr(app_config, "realtime_worker", None) is not False
        or getattr(app_config, "mujoco_viewer_enabled", None) is not False
        or getattr(app_config, "execution_difference_enabled", None) is not True
    ):
        raise ValueError(
            "CubeLift deployment requires alpha=0, deferred human input, "
            "deterministic physics, no viewer, and execution observations."
        )
    runtime = getattr(application, "runtime", None)
    runtime_config = getattr(runtime, "config", None)
    if not callable(
        getattr(
            getattr(getattr(runtime, "core", None), "coordinator", None),
            "capture_cut",
            None,
        )
    ):
        raise ValueError(
            "Passive state deployment requires the existing control-cut owner."
        )
    runtime_session_id = _required(
        getattr(runtime_config, "runtime_session_id", None),
        field_name="runtime_session_id",
    )
    control_domain_id = _required(
        getattr(runtime_config, "control_domain_id", None),
        field_name="control_domain_id",
    )
    if (
        runtime_session_id != getattr(app_config, "runtime_session_id", None)
        or control_domain_id != getattr(app_config, "control_domain_id", None)
    ):
        raise ValueError("Application/runtime session identity mismatch.")

    task = getattr(application, "task_config", None)
    task_hash = _required(
        getattr(task, "source_sha256", None),
        field_name="task_config.source_sha256",
    )
    expected_task_fields = {
        "object_body_name": evaluator_config.object_body_name,
        "grasp_latch_left_finger_body_name": (
            evaluator_config.left_finger_body_name
        ),
        "grasp_latch_right_finger_body_name": (
            evaluator_config.right_finger_body_name
        ),
        "table_body_name": evaluator_config.table_body_name,
    }
    for field_name, expected in expected_task_fields.items():
        if getattr(task, field_name, None) != expected:
            raise ValueError(f"Evaluator/task {field_name} mismatch.")
    if evaluator_config.task_config_hash != task_hash:
        raise ValueError("Evaluator binds another task configuration.")
    if (
        abs(
            _finite(
                getattr(task, "lift_delta_m", None),
                field_name="task_config.lift_delta_m",
            )
            - evaluator_config.lift_delta_m
        )
        > 1e-12
    ):
        raise ValueError("Evaluator/task lift_delta_m mismatch.")
    if (
        abs(
            _finite(
                getattr(task, "physics_hz", None),
                field_name="task_config.physics_hz",
            )
            - evaluator_config.physics_hz
        )
        > 1e-12
    ):
        raise ValueError("Evaluator/task physics_hz mismatch.")
    if (
        not evaluator_config.success_requires_bilateral_contact
        or not evaluator_config.requires_no_table_contact
    ):
        raise ValueError(
            "CubeLift Route requires bilateral contact and no table contact."
        )

    policy_source = getattr(application, "policy_source", None)
    policy_task_hash = _required(
        getattr(getattr(policy_source, "task_config", None), "source_sha256", None),
        field_name="policy_source.task_config.source_sha256",
    )
    if policy_task_hash != task_hash:
        raise ValueError("Policy source and lower target use different tasks.")
    compatibility = getattr(policy_source, "compatibility", None)
    if not isinstance(compatibility, Mapping):
        raise TypeError("Policy source must expose compatibility evidence.")
    if (
        bool(compatibility.get("task_config_mismatch"))
        or compatibility.get("runtime_task_config_sha256") != task_hash
        or compatibility.get("checkpoint_task_config_sha256") != task_hash
    ):
        raise ValueError("Policy checkpoint/task compatibility is not exact.")

    observation_plan = getattr(runtime_config, "execution_observation_plan", None)
    observation_binding = getattr(runtime, "execution_observation_binding", None)
    observation_capability = getattr(
        runtime,
        "execution_observation_capability",
        None,
    )
    if (
        observation_plan is None
        or not bool(getattr(observation_plan, "enabled", False))
        or getattr(observation_plan, "reference_mode", None) != "full_frames"
        or getattr(observation_plan, "measurement_mode", None) != "full_frames"
        or tuple(getattr(observation_plan, "component_ids", ())) != ("right_arm",)
        or dict(
            getattr(
                observation_plan,
                "required_representation_by_component",
                {},
            )
        )
        != {"right_arm": "cartesian_pose"}
    ):
        raise ValueError("CubeLift deployment requires exact TCP frame observations.")
    observation_plan_hash = _hash(
        observation_plan,
        field_name="execution_observation_plan hash",
    )
    if (
        observation_binding is None
        or getattr(observation_binding, "plan_hash", None) != observation_plan_hash
    ):
        raise ValueError("Lower target did not bind the exact observation plan.")
    observation_binding_id = _required(
        getattr(observation_binding, "binding_id", None),
        field_name="execution observation binding_id",
    )
    observation_binding_hash = _hash(
        observation_binding,
        field_name="execution observation binding hash",
    )
    observation_capability_hash = _hash(
        observation_capability,
        field_name="execution observation capability hash",
    )
    if dict(
        getattr(observation_capability, "representation_by_component", {})
    ) != {"right_arm": "cartesian_pose"}:
        raise ValueError(
            "TCP state deployment requires exact right_arm pose capability."
        )
    pose_schema_by_component = dict(
        getattr(observation_capability, "schema_hash_by_component", {})
    )
    if set(pose_schema_by_component) != {"right_arm"}:
        raise ValueError("TCP state deployment lacks exact pose schema identity.")
    pose_schema_hash = _required(
        pose_schema_by_component["right_arm"],
        field_name="right_arm pose schema hash",
    )
    if getattr(application, "execution_difference_observer", None) is None:
        raise ValueError("CubeLift execution measurement observer is absent.")

    target_profile = getattr(application, "target_frame_profile", None)
    tool_profile = getattr(application, "tool_frame_profile", None)
    target_profile_hash = _hash(
        target_profile,
        field_name="target_frame_profile hash",
    )
    frame_graph_hash = _hash(
        getattr(target_profile, "frame_graph", None),
        field_name="target frame graph hash",
    )
    reference_frame_by_component = dict(
        getattr(target_profile, "reference_frame_by_component", {})
    )
    reference_frame_id = _required(
        reference_frame_by_component.get("right_arm"),
        field_name="right_arm reference frame",
    )
    reference_frame_spec_hash = _hash(
        getattr(
            getattr(target_profile, "frame_graph", None),
            "frame_by_id",
            {},
        ).get(reference_frame_id),
        field_name="right_arm reference frame spec hash",
    )
    tool_profile_hash = _hash(
        tool_profile,
        field_name="tool_frame_profile hash",
    )
    if getattr(tool_profile, "target_frame_profile_hash", None) != target_profile_hash:
        raise ValueError("Tool profile belongs to another target frame profile.")
    pose_frame_contract_hash = _required(
        getattr(runtime_config, "pose_frame_contract_hash", None),
        field_name="pose_frame_contract_hash",
    )
    calibration_hash = canonical_content_hash(
        {
            "calibration_kind": "mujoco_scene_ground_truth",
            "target_frame_profile_hash": target_profile_hash,
            "frame_graph_hash": frame_graph_hash,
            "tool_frame_profile_hash": tool_profile_hash,
            "pose_frame_contract_hash": pose_frame_contract_hash,
            "observation_capability_hash": observation_capability_hash,
        },
        domain=MUJOCO_CUBE_LIFT_AGENT_SKILL_CALIBRATION_HASH_DOMAIN_V1,
    )
    return {
        "runtime_session_id": runtime_session_id,
        "control_domain_id": control_domain_id,
        "task_config_hash": task_hash,
        "observation_plan_hash": observation_plan_hash,
        "observation_binding_id": observation_binding_id,
        "observation_binding_hash": observation_binding_hash,
        "observation_capability_hash": observation_capability_hash,
        "pose_schema_hash": pose_schema_hash,
        "reference_frame_id": reference_frame_id,
        "reference_frame_spec_hash": reference_frame_spec_hash,
        "frame_graph_hash": frame_graph_hash,
        "tool_config_hash": tool_profile_hash,
        "calibration_hash": calibration_hash,
    }




def build_mujoco_cube_lift_agent_skill_contract_bundle(
    *,
    application: Any,
    evaluator_config: CubeLiftTerminalEvaluatorConfig,
    executor_id: str = MUJOCO_CUBE_LIFT_AGENT_SKILL_EXECUTOR_ID_V1,
    checkpoint_hash_provider: Callable[[str | Path], str] = _default_checkpoint_hash,
) -> MujocoCubeLiftAgentSkillContractBundle:
    """Build the exact catalogs/manifests/bindings for one constructed graph."""

    if not isinstance(evaluator_config, CubeLiftTerminalEvaluatorConfig):
        raise TypeError("evaluator_config must be typed.")
    if not callable(checkpoint_hash_provider):
        raise TypeError("checkpoint_hash_provider must be callable.")
    exact = _validate_exact_application(
        application,
        evaluator_config=evaluator_config,
    )
    station_manifest = build_mujoco_sim_hil_agent_skill_station_manifest(
        application.composition
    )
    execution_manifest = build_mujoco_sim_hil_agent_skill_execution_manifest(
        application.composition,
        station_manifest,
    )
    runtime = application.runtime
    execution_binding = station_manifest.execution_binding
    if station_manifest.control_domain_id != exact["control_domain_id"]:
        raise ValueError("Station manifest control domain mismatch.")
    if (
        station_manifest.trace_requirement_profile_id
        != execution_binding.trace_requirement_profile_id
    ):
        raise ValueError("Station/target trace requirement profile mismatch.")
    lower_hash_checks = (
        (
            execution_binding.target_runtime_action_contract_hash,
            _hash(
                runtime.target_runtime_action_contract,
                field_name="target runtime action contract hash",
            ),
            "target runtime action contract",
        ),
        (
            execution_manifest.action_schema_hash,
            _hash(runtime.action_schema, field_name="runtime action schema hash"),
            "action schema",
        ),
        (
            execution_manifest.component_target_capability_hash,
            _hash(
                runtime.component_target_capability,
                field_name="component target capability hash",
            ),
            "component target capability",
        ),
        (
            execution_manifest.component_catalog_hash,
            _hash(runtime.catalog, field_name="component catalog hash"),
            "component catalog",
        ),
        (
            execution_manifest.execution_capability_profile_hash,
            _hash(
                runtime.execution_capability,
                field_name="execution capability hash",
            ),
            "execution capability",
        ),
    )
    for actual, expected, label in lower_hash_checks:
        if actual != expected:
            raise ValueError(f"{label} manifest/runtime mismatch.")

    definition = cube_lift_skill_definition_v1()
    schema_entry = cube_lift_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash
    )
    route = cube_lift_sim_route_v1(
        definition=definition,
        schema_entry=schema_entry,
    )
    state_definition = tcp_pose_state_definition_v1(
        controlled_component_ids=("right_arm",),
    )
    state_profile_hash = canonical_content_hash(
        {
            "parameter_profile_id": (
                MUJOCO_TCP_POSE_STATE_PARAMETER_PROFILE_ID_V1
            ),
            "component_ids": ("right_arm",),
            "reference_frame_ids": (exact["reference_frame_id"],),
            "controlled_frame_id": "tool_tcp",
            "max_measurement_age_ms": (
                MUJOCO_TCP_POSE_STATE_MAX_MEASUREMENT_AGE_MS_V1
            ),
        },
        domain=MUJOCO_TCP_POSE_STATE_PROFILE_HASH_DOMAIN_V1,
    )
    state_schema_entry = tcp_pose_state_schema_entry_v1(
        skill_definition_hash=state_definition.semantic_content_hash,
        parameter_profile_id=MUJOCO_TCP_POSE_STATE_PARAMETER_PROFILE_ID_V1,
        parameter_profile_hash=state_profile_hash,
        allowed_component_ids=("right_arm",),
        allowed_reference_frame_ids=(exact["reference_frame_id"],),
        max_measurement_age_ms=(
            MUJOCO_TCP_POSE_STATE_MAX_MEASUREMENT_AGE_MS_V1
        ),
    )
    state_route = SkillExecutionRoute(
        skill_id=state_definition.skill_id,
        skill_version=state_definition.skill_version,
        skill_definition_hash=state_definition.semantic_content_hash,
        route_id=MUJOCO_TCP_POSE_STATE_ROUTE_ID_V1,
        route_version="1",
        parameter_schema_hash=state_schema_entry.parameter_schema_hash,
        execution_result_schema_hash=(
            state_schema_entry.execution_result_schema_hash
        ),
        task_evaluation_schema_hash=(
            state_schema_entry.task_evaluation_schema_hash
        ),
        allowed_parameter_profiles=state_schema_entry.parameter_profiles,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=("dry_run", "sim_active"),
        target_kind="sim",
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=1_000,
        cancellation_policy="cancel_only",
        requires_task_evaluation=False,
    )
    validate_agent_state_route_schema_v1(
        route=state_route,
        schema_entry=state_schema_entry,
    )
    checkpoint_hash = _required(
        checkpoint_hash_provider(application.policy_source.checkpoint_path),
        field_name="policy checkpoint hash",
    )
    binding = CubeLiftSimDeploymentBinding(
        deployment_binding_id=(
            "sim.mujoco.cube_lift.policy_only:"
            + exact["runtime_session_id"]
        ),
        deployment_binding_version="1",
        route_hash=route.semantic_content_hash,
        runtime_session_id=exact["runtime_session_id"],
        control_domain_id=exact["control_domain_id"],
        target_binding_id=execution_binding.binding_id,
        station_manifest_hash=station_manifest.semantic_content_hash,
        execution_manifest_hash=execution_manifest.semantic_content_hash,
        execution_domain_binding_hash=execution_binding.semantic_content_hash,
        target_runtime_action_contract_hash=(
            execution_binding.target_runtime_action_contract_hash
        ),
        component_target_capability_hash=(
            execution_manifest.component_target_capability_hash
        ),
        action_schema_hash=execution_manifest.action_schema_hash,
        observation_plan_hash=exact["observation_plan_hash"],
        frame_graph_hash=exact["frame_graph_hash"],
        tool_config_hash=exact["tool_config_hash"],
        calibration_hash=exact["calibration_hash"],
        task_config_hash=exact["task_config_hash"],
        policy_checkpoint_hash=checkpoint_hash,
        evaluator_configuration_hash=evaluator_config.semantic_content_hash,
        trace_requirement_profile_id=(
            station_manifest.trace_requirement_profile_id
        ),
        executor_id=_required(executor_id, field_name="executor_id"),
        explicit_exclusions=(
            "hardware_backend",
            "human_input_attachment",
            "raw_action_submission",
            "second_dispatch_owner",
        ),
    )
    state_binding = MujocoPassiveTcpPoseStateDeploymentBinding(
        deployment_binding_id=(
            "sim.mujoco.passive_tcp_pose:" + exact["runtime_session_id"]
        ),
        deployment_binding_version="1",
        route_hash=state_route.semantic_content_hash,
        runtime_session_id=exact["runtime_session_id"],
        control_domain_id=exact["control_domain_id"],
        target_binding_id=execution_binding.binding_id,
        station_manifest_hash=station_manifest.semantic_content_hash,
        execution_manifest_hash=execution_manifest.semantic_content_hash,
        observation_binding_id=exact["observation_binding_id"],
        observation_binding_hash=exact["observation_binding_hash"],
        observation_plan_hash=exact["observation_plan_hash"],
        observation_capability_hash=exact["observation_capability_hash"],
        pose_schema_hash=exact["pose_schema_hash"],
        reference_frame_id=exact["reference_frame_id"],
        reference_frame_spec_hash=exact["reference_frame_spec_hash"],
        frame_graph_hash=exact["frame_graph_hash"],
        tool_config_hash=exact["tool_config_hash"],
        calibration_hash=exact["calibration_hash"],
        executor_id=MUJOCO_TCP_POSE_STATE_EXECUTOR_ID_V1,
        explicit_exclusions=(
            "control_authority",
            "device_start",
            "motion",
            "physics_step",
            "publication_open",
            "task_evaluation",
        ),
    )
    evaluation_plan = cube_lift_evaluation_plan_v1(
        definition=definition,
        route=route,
        deployment_binding=binding,
        evaluator_config=evaluator_config,
        task_evaluation_schema_hash=schema_entry.task_evaluation_schema_hash,
    )
    definition_catalog = SkillDefinitionCatalog(
        catalog_id="dexteleop.agent_skill_definitions.cube_lift_session",
        catalog_version="1",
        skill_definitions=(definition, state_definition),
    )
    schema_catalog = AgentSkillSchemaCatalog(
        catalog_id="dexteleop.agent_skill_schemas.cube_lift_session",
        catalog_version="1",
        schema_entries=(schema_entry, state_schema_entry),
    )
    availability_hash = canonical_content_hash(
        {
            "status": "configured_requires_activation_preflight",
            "runtime_session_id": exact["runtime_session_id"],
            "route_hash": route.semantic_content_hash,
            "deployment_binding_hash": binding.semantic_content_hash,
            "station_manifest_hash": station_manifest.semantic_content_hash,
            "execution_manifest_hash": execution_manifest.semantic_content_hash,
        },
        domain=MUJOCO_CUBE_LIFT_AGENT_SKILL_AVAILABILITY_HASH_DOMAIN_V1,
    )
    state_availability_hash = canonical_content_hash(
        {
            "status": "configured_passive_read_ready",
            "runtime_session_id": exact["runtime_session_id"],
            "route_hash": state_route.semantic_content_hash,
            "deployment_binding_hash": state_binding.semantic_content_hash,
            "observation_binding_hash": exact["observation_binding_hash"],
            "observation_capability_hash": (
                exact["observation_capability_hash"]
            ),
        },
        domain=MUJOCO_TCP_POSE_STATE_AVAILABILITY_HASH_DOMAIN_V1,
    )
    route_manifest = SkillRouteManifest(
        runtime_session_id=exact["runtime_session_id"],
        control_domain_id=exact["control_domain_id"],
        station_manifest_hash=station_manifest.semantic_content_hash,
        execution_manifest_hash=execution_manifest.semantic_content_hash,
        skill_definition_catalog_hash=definition_catalog.semantic_content_hash,
        skill_schema_catalog_hash=schema_catalog.semantic_content_hash,
        target_binding_id=binding.target_binding_id,
        target_kind="sim",
        current_mode="sim_active",
        supported_transport_features=(
            "own_domain_hold_v1",
            "own_operation_cancel_v1",
            "own_operation_status_v1",
            "idempotency_v1",
        ),
        skill_routes=(
            SkillRouteManifestEntry(
                route=route,
                deployment_binding=binding.to_ref(),
                evaluation_plan_hash=evaluation_plan.semantic_content_hash,
                availability_hash=availability_hash,
            ),
            SkillRouteManifestEntry(
                route=state_route,
                deployment_binding=state_binding.to_ref(),
                evaluation_plan_hash=None,
                availability_hash=state_availability_hash,
            ),
        ),
        explicit_exclusions=(
            "agent_estop",
            "agent_rearm",
            "agent_resume",
            "hardware_backend",
            "human_input_attachment",
            "raw_action_submission",
            "second_dispatch_owner",
        ),
    )
    payload_validators = AgentSkillPayloadValidatorRegistry(
        registry_id=(
            "dexteleop.agent_skill_payload_validators.cube_lift_session:"
            + exact["runtime_session_id"]
        ),
        validators=(
            cube_lift_payload_validator_set_v1(
                schema_entry=schema_entry,
                evaluator_config=evaluator_config,
            ),
            build_mujoco_tcp_pose_state_payload_validator_set(
                route=state_route,
                schema_entry=state_schema_entry,
                exact=exact,
            ),
        ),
    )
    return MujocoCubeLiftAgentSkillContractBundle(
        definition_catalog=definition_catalog,
        schema_catalog=schema_catalog,
        route_manifest=route_manifest,
        deployment_binding=binding,
        state_deployment_binding=state_binding,
        evaluator_config=evaluator_config,
        evaluation_plan=evaluation_plan,
        payload_validators=payload_validators,
        station_manifest=station_manifest,
        execution_manifest=execution_manifest,
    )


__all__ = [
    "MUJOCO_CUBE_LIFT_AGENT_SKILL_AVAILABILITY_HASH_DOMAIN_V1",
    "MUJOCO_CUBE_LIFT_AGENT_SKILL_CALIBRATION_HASH_DOMAIN_V1",
    "MUJOCO_CUBE_LIFT_AGENT_SKILL_EXECUTOR_ID_V1",
    "MujocoCubeLiftAgentSkillContractBundle",
    "build_mujoco_cube_lift_agent_skill_contract_bundle",
    "cube_lift_terminal_evaluator_config_from_task",
]
