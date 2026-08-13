"""Exact application graph factory for the no-model Agent TCP-pose Route.

Calling this factory explicitly prepares MuJoCo resources, but it does not
start the runtime, advance physics, bind a transport or submit motion.  The
caller must provide the dedicated target config; ordinary CubeLift Sim-HIL
configuration is rejected instead of being silently repurposed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from collection_runtime.source_observation_history import SQLiteSourceObservationHistoryStore
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ExecutionObservationPlan
from robot_motion.target_safety import (
    TCPPoseEnvelopeCatalogEntry,
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeSafetyRuntime,
    TCPPoseEnvelopeSideConfig,
)
from sim.mujoco.deploy import (
    MujocoComponentActionRobotActionV1Materializer,
    MujocoExecutionDifferenceObserver,
    MujocoSimHILTargetConfig,
    build_mujoco_agent_skill_motion_target,
    build_mujoco_component_execution_binding,
    build_mujoco_pose_component_target_profile,
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
    prepare_mujoco_sim_hil_target,
)

from .motion_application import MujocoAgentSkillMotionApplication
from .motion_deployment import (
    MUJOCO_AGENT_TCP_POSE_MAX_ROTATION_DELTA_RAD_V1,
    MUJOCO_AGENT_TCP_POSE_MAX_TRANSLATION_DELTA_M_V1,
    MUJOCO_AGENT_TCP_POSE_ROUTE_ID_V1,
    MUJOCO_AGENT_TCP_POSE_WORKSPACE_BOUNDS_V1,
    MujocoAgentTcpPoseHistoryProfile,
    build_mujoco_agent_tcp_pose_contract_bundle,
)
from .motion_gateway_session import (
    MujocoAgentTcpPoseGatewaySession,
    assemble_mujoco_agent_tcp_pose_gateway_session,
)

MUJOCO_AGENT_TCP_POSE_CALIBRATION_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.mujoco_tcp_pose_scene_calibration.v1"
)
MUJOCO_AGENT_TCP_POSE_COMPONENT_PROFILE_ID_V1 = (
    "mujoco.rm65.right_agent_tcp_pose.v1"
)
MUJOCO_AGENT_TCP_POSE_OBSERVATION_PLAN_ID_V1 = (
    "control-station-agent-tcp-pose-right-arm-observation-v1"
)
MUJOCO_AGENT_TCP_POSE_STANDARD_ENVELOPE_ID_V1 = (
    "mujoco.agent_tcp_pose.safety.standard.v1"
)


@dataclass(frozen=True)
class MujocoAgentTcpPoseHistoryStorageConfig:
    """Explicit disk binding for the optional history Route; never auto-starts."""

    root: Path
    profile: MujocoAgentTcpPoseHistoryProfile = field(
        default_factory=MujocoAgentTcpPoseHistoryProfile
    )
    filename: str = "planning_history.sqlite3"

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("history root must be pathlib.Path.")
        if not self.root.is_absolute():
            raise ValueError("history root must be an explicit absolute path.")
        if not isinstance(self.profile, MujocoAgentTcpPoseHistoryProfile):
            raise TypeError("history profile must be typed.")
        if (
            not isinstance(self.filename, str)
            or not self.filename.strip()
            or Path(self.filename).name != self.filename
        ):
            raise ValueError("history filename must be a non-empty basename.")


def mujoco_agent_tcp_pose_execution_observation_plan_v1(
) -> ExecutionObservationPlan:
    return ExecutionObservationPlan(
        plan_id=MUJOCO_AGENT_TCP_POSE_OBSERVATION_PLAN_ID_V1,
        reference_mode="full_frames",
        measurement_mode="full_frames",
        component_ids=("right_arm",),
        required_representation_by_component={
            "right_arm": "cartesian_pose",
        },
    )


def validate_mujoco_agent_tcp_pose_target_config(
    config: MujocoSimHILTargetConfig,
) -> None:
    """Fail before scene construction when another Sim-HIL profile is used."""

    if not isinstance(config, MujocoSimHILTargetConfig):
        raise TypeError("config must be a MujocoSimHILTargetConfig.")
    plan = config.execution_observation_plan
    expected = mujoco_agent_tcp_pose_execution_observation_plan_v1()
    if config.allow_scripted_task_reset:
        raise ValueError(
            "Agent TCP pose config requires allow_scripted_task_reset=False."
        )
    if config.exposed_pose_sides != ("right",):
        raise ValueError(
            "Agent TCP pose config must expose exactly the right pose side."
        )
    if (
        not isinstance(plan, ExecutionObservationPlan)
        or plan.semantic_content_hash != expected.semantic_content_hash
    ):
        raise ValueError(
            "Agent TCP pose config requires the exact right-arm Cartesian "
            "execution observation plan."
        )


def _calibration_hash(target: Any) -> str:
    runtime = target.runtime
    target_profile = target.target_frame_profile
    return canonical_content_hash(
        {
            "calibration_kind": "mujoco_scene_ground_truth",
            "target_frame_profile_hash": (
                target_profile.semantic_content_hash
            ),
            "frame_graph_hash": (
                target_profile.frame_graph.semantic_content_hash
            ),
            "tool_frame_profile_hash": (
                target.tool_frame_profile.semantic_content_hash
            ),
            "pose_frame_contract_hash": runtime.config.pose_frame_contract_hash,
            "observation_capability_hash": (
                runtime.execution_observation_capability.semantic_content_hash
            ),
        },
        domain=MUJOCO_AGENT_TCP_POSE_CALIBRATION_HASH_DOMAIN_V1,
    )


def _build_standard_tcp_pose_envelope_runtime(
    *,
    prepared: Any,
    target_config: MujocoSimHILTargetConfig,
) -> TCPPoseEnvelopeSafetyRuntime:
    """Bind the existing v1 motion bounds as one exact lower envelope.

    This intentionally publishes only ``standard``.  Strict/relaxed profiles
    require independently reviewed values; they are not derived by scaling.
    """

    target_profile = prepared.target_frame_profile
    tool_profile = prepared.tool_frame_profile
    reference_frame = target_profile.reference_frame_by_component["right_arm"]
    tool_binding_hash = tool_profile.binding_by_component[
        "right_arm"
    ].semantic_content_hash
    target_binding_hash = canonical_content_hash(
        {
            "target_frame_profile_hash": target_profile.semantic_content_hash,
            "component_id": "right_arm",
            "side": "right",
            "reference_frame_id": reference_frame,
            "controlled_frame_id": "tool_tcp",
        },
        domain="dexteleop.mujoco.agent_tcp_pose_envelope_target_binding.v1",
    )
    bounds = MUJOCO_AGENT_TCP_POSE_WORKSPACE_BOUNDS_V1
    delta = MUJOCO_AGENT_TCP_POSE_MAX_TRANSLATION_DELTA_M_V1
    profile = TCPPoseEnvelopeProfile(
        profile_id="mujoco-agent-tcp-pose-standard-envelope-v1",
        target_frame_binding_hash=target_binding_hash,
        tool_frame_binding_hash_by_side={"right": tool_binding_hash},
        reference_frame_by_side={"right": reference_frame},
        side_config_by_side={
            "right": TCPPoseEnvelopeSideConfig(
                absolute_min_xyz_m=(bounds[0], bounds[2], bounds[4]),
                absolute_max_xyz_m=(bounds[1], bounds[3], bounds[5]),
                relative_min_xyz_m=(-delta, -delta, -delta),
                relative_max_xyz_m=(delta, delta, delta),
                relative_axes="canonical_axes",
                max_rotation_rad=(
                    MUJOCO_AGENT_TCP_POSE_MAX_ROTATION_DELTA_RAD_V1
                ),
            )
        },
    )
    constraint_set_hash = canonical_content_hash(
        {
            "tcp_pose_envelope_profile_hash": profile.semantic_content_hash,
            "legacy_activation_delta_check_retained": True,
            "canonical_keyframe_safety_retained": True,
            "coupled_ik_safety_retained": True,
        },
        domain="dexteleop.mujoco.agent_tcp_pose_constraint_set.v1",
    )
    entry = TCPPoseEnvelopeCatalogEntry(
        envelope_id=MUJOCO_AGENT_TCP_POSE_STANDARD_ENVELOPE_ID_V1,
        profile=profile,
        constraint_set_hash=constraint_set_hash,
        allowed_route_ids=(MUJOCO_AGENT_TCP_POSE_ROUTE_ID_V1,),
    )
    return TCPPoseEnvelopeSafetyRuntime(
        catalog_entries=(entry,),
        target_frame_binding_hash=target_binding_hash,
        tool_frame_binding_hash_by_side={"right": tool_binding_hash},
        reference_frame_by_side={"right": reference_frame},
        clock_domain_id=target_config.clock_domain_id,
    )


def build_mujoco_agent_tcp_pose_gateway_session(
    *,
    target_config: MujocoSimHILTargetConfig,
    allowed_client_ids: tuple[str, ...],
    max_retained_operations: int = 16,
    trace_sink: Callable[[Any], None] | None = None,
    history_storage: MujocoAgentTcpPoseHistoryStorageConfig | None = None,
) -> MujocoAgentTcpPoseGatewaySession:
    """Build the exact default-off graph; do not start, record, or serve it."""

    if history_storage is not None and not isinstance(
        history_storage,
        MujocoAgentTcpPoseHistoryStorageConfig,
    ):
        raise TypeError("history_storage must be typed or None.")
    validate_mujoco_agent_tcp_pose_target_config(target_config)
    prepared = prepare_mujoco_sim_hil_target(target_config)
    component_profile = build_mujoco_pose_component_target_profile(
        prepared.scene,
        profile_id=MUJOCO_AGENT_TCP_POSE_COMPONENT_PROFILE_ID_V1,
        arm_side_by_component={"right_arm": "right"},
        arm_reference_frame_by_component={"right_arm": "right_arm_base"},
        arm_controlled_frame_by_component={"right_arm": "tool_tcp"},
        selection_group_by_component={"right_arm": "arms"},
    )
    runtime_rig_holder: dict[str, Any] = {}

    def capture_support_arm_pose() -> Mapping[str, Any]:
        rig = runtime_rig_holder.get("rig")
        if rig is None:
            raise RuntimeError(
                "MuJoCo support-pose capture requested before target binding."
            )
        return rig.capture_scene(
            lambda scene: {
                "left": capture_mujoco_sim_hil_tool_tcp_pose_by_side(
                    scene,
                    prepared.task_config,
                )["left"]
            }
        )

    materializer = MujocoComponentActionRobotActionV1Materializer(
        prepared.scene,
        component_profile,
        hand_action_spec_by_component={},
        support_arm_pose_provider=capture_support_arm_pose,
        support_arm_sides=("left",),
        support_pose_contract_id=(
            "mujoco.measured_left_tool_tcp_support_for_agent_coupled_ik.v1"
        ),
    )
    component_execution_binding = build_mujoco_component_execution_binding(
        component_profile,
        materializer,
    )
    observation_sink = MujocoExecutionDifferenceObserver(
        selection_group_id="arms",
        component_ids=("right_arm",),
    )
    tcp_pose_envelope_runtime = _build_standard_tcp_pose_envelope_runtime(
        prepared=prepared,
        target_config=target_config,
    )
    target = build_mujoco_agent_skill_motion_target(
        config=target_config,
        component_execution_binding=component_execution_binding,
        prepared_target=prepared,
        execution_observation_sink=observation_sink,
        trace_sink=trace_sink,
        tcp_pose_envelope_runtime=tcp_pose_envelope_runtime,
    )
    runtime_rig_holder["rig"] = target.runtime.rig
    captured_at = float(target.runtime.clock())
    initial_pose = target.runtime.rig.capture_scene(
        lambda scene: capture_mujoco_sim_hil_tool_tcp_pose_by_side(
            scene,
            prepared.task_config,
        )["right"]
    )
    initialization_event_hash = canonical_content_hash(
        {
            "task_initialization_info": dict(prepared.task_initialization_info),
            "measured_right_tool_tcp": initial_pose,
            "captured_at": captured_at,
            "clock_domain_id": target_config.clock_domain_id,
        },
        domain="dexteleop.mujoco.agent_tcp_pose_qpos_initialization.v1",
    )
    tcp_pose_envelope_runtime.commit_reference(
        reference_id=(
            "mujoco-agent-qpos-init-reference:" + initialization_event_hash
        ),
        tcp_pose_by_side={"right": initial_pose},
        captured_at_by_side={"right": captured_at},
        source_event_ids=(
            "mujoco-qpos-initialization-completed:" + initialization_event_hash,
        ),
        lower_state="configured",
    )
    if target.runtime.action_schema.semantic_content_hash != (
        component_profile.action_schema.semantic_content_hash
    ):
        raise RuntimeError("Agent motion target changed its component schema.")
    application = MujocoAgentSkillMotionApplication(
        target=target,
        calibration_hash=_calibration_hash(target),
    )
    contract_bundle = build_mujoco_agent_tcp_pose_contract_bundle(
        application=application,
        history_profile=(None if history_storage is None else history_storage.profile),
    )
    history_store = None
    if history_storage is not None:
        source_binding = contract_bundle.history_source_binding
        if source_binding is None:
            raise RuntimeError("History contract lacks its source binding.")
        history_store = SQLiteSourceObservationHistoryStore(
            root=history_storage.root,
            binding=source_binding,
            limits=history_storage.profile.store_limits,
            filename=history_storage.filename,
        )
    try:
        return assemble_mujoco_agent_tcp_pose_gateway_session(
            application=application,
            contract_bundle=contract_bundle,
            allowed_client_ids=allowed_client_ids,
            max_retained_operations=max_retained_operations,
            source_observation_history_store=history_store,
            owns_source_observation_history_store=history_store is not None,
        )
    except Exception:
        if history_store is not None:
            history_store.close()
        raise


__all__ = [
    "MUJOCO_AGENT_TCP_POSE_CALIBRATION_HASH_DOMAIN_V1",
    "MUJOCO_AGENT_TCP_POSE_COMPONENT_PROFILE_ID_V1",
    "MUJOCO_AGENT_TCP_POSE_OBSERVATION_PLAN_ID_V1",
    "MUJOCO_AGENT_TCP_POSE_STANDARD_ENVELOPE_ID_V1",
    "MujocoAgentTcpPoseHistoryStorageConfig",
    "build_mujoco_agent_tcp_pose_gateway_session",
    "mujoco_agent_tcp_pose_execution_observation_plan_v1",
    "validate_mujoco_agent_tcp_pose_target_config",
]
