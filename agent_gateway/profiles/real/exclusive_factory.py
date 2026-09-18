"""Independent exclusive real Skill application assembly.

Unlike the compatibility RealTeleop attachment, this factory creates the
Gateway and its sole lower control owner directly from typed deployment ports.
It starts no socket, thread, backend, device, or motion.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from agent_gateway.protocol.skill_catalog import SkillDeploymentSafetyEnvelopeRef
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.runtime.station_revision import AgentStationRevisionTracker
from robot_motion.coordinate_frames import CoordinateFrameGraph
from robot_motion.runtime import TeleopCommand, TeleopObservation
from teleop_runtime.skill_deploy.control_runtime import (
    ExclusiveRealSkillControlRuntime,
    RealSkillMeasuredQposObserver,
    RealSkillMotionDispatchReceipt,
    RealSkillMotionEmergencyStopReceipt,
    RealSkillMotionHoldReceipt,
    RealSkillMotionObservationCut,
    RealSkillMotionTargetSynchronizer,
)
from teleop_runtime.skill_deploy.motion_port import (
    RealAgentMotionOwnerIdentity,
    RealAgentMotionPort,
    RealAgentMotionPortConfig,
    RealAgentMotionTargetBuilder,
)
from teleop_runtime.skill_deploy.rm65_ports import (
    ExclusiveRM65SkillDeployPorts,
)

from .localization_deployment import RealObjectLocalizationRegistration
from .grasp_approach_plan import (
    DEFAULT_REAL_GRASP_EXECUTION_OPERATION_TIMEOUT_SEC,
)
from .localization_gateway import ObjectLocalizationProvider
from .motion_gateway_session import (
    RealAgentPrimitiveGatewaySession,
    assemble_real_agent_primitive_gateway_session,
)
from .motion_primitive_profiles import (
    RealAgentPrimitiveConvergenceProfile,
    build_real_agent_primitive_contract_bundle,
)
from .observation_gateway import (
    RealControlOwnerState,
    RealTcpPoseObserveAttachmentSpec,
)
from .perception_deployment import RealPerceptionCaptureRegistration
from .realsense_perception import RealSenseAgentPerceptionProvider
from .station_point_grasp_deployment import StationBoundPointGraspRegistration
from .station_point_grasp_render_deployment import (
    StationBoundPointGraspRenderRegistration,
)


@dataclass(frozen=True)
class ExclusiveRealAgentPrimitiveDeployment:
    """One Gateway session and exactly one independent lower control owner."""

    control_runtime: ExclusiveRealSkillControlRuntime
    gateway_session: RealAgentPrimitiveGatewaySession

    def __post_init__(self) -> None:
        if not isinstance(
            self.control_runtime,
            ExclusiveRealSkillControlRuntime,
        ):
            raise TypeError("control_runtime must be typed.")
        if not isinstance(
            self.gateway_session,
            RealAgentPrimitiveGatewaySession,
        ):
            raise TypeError("gateway_session must be typed.")
        if self.control_runtime.motion_port is not self.gateway_session.port:
            raise ValueError(
                "Gateway and exclusive Skill control runtime must share one port."
            )


def _owner_identity(
    attachment: RealTcpPoseObserveAttachmentSpec,
) -> RealAgentMotionOwnerIdentity:
    return RealAgentMotionOwnerIdentity(
        runtime_session_id=attachment.runtime_session_id,
        control_domain_id=attachment.control_domain_id,
        embodiment_instance_id=attachment.embodiment_instance_id,
        target_binding_id=attachment.target_binding_id,
        execution_manifest_hash=attachment.execution_manifest_hash,
        backend_inventory_hash=attachment.backend_inventory_hash,
        clock_domain_id=attachment.clock_domain_id,
    )


def _owner_state(port: RealAgentMotionPort) -> RealControlOwnerState:
    cut = port.latest_owner_state()
    if cut is None:
        raise RuntimeError("No exclusive lower-owner cut is available.")
    identity = cut.identity
    return RealControlOwnerState(
        runtime_session_id=identity.runtime_session_id,
        control_domain_id=identity.control_domain_id,
        embodiment_instance_id=identity.embodiment_instance_id,
        target_binding_id=identity.target_binding_id,
        execution_manifest_hash=identity.execution_manifest_hash,
        backend_inventory_hash=identity.backend_inventory_hash,
        clock_domain_id=identity.clock_domain_id,
        owner_token_id=cut.owner_token_id,
        owner_kind=cut.owner_kind,
        owner_generation=cut.owner_generation,
        authority_epoch=cut.authority_epoch,
        execution_epoch=cut.execution_epoch,
        lifecycle_state=cut.lifecycle_state,
        execution_gate=cut.execution_gate,
        owner_state_revision=cut.owner_state_revision,
        captured_at=cut.captured_at,
        source_owner_state_hash=cut.semantic_content_hash,
        evidence_refs=cut.evidence_refs,
    )


def assemble_exclusive_real_agent_primitive_deployment(
    *,
    target_builder: RealAgentMotionTargetBuilder,
    port_config: RealAgentMotionPortConfig,
    observation_attachment: RealTcpPoseObserveAttachmentSpec,
    frame_graph: CoordinateFrameGraph,
    primitive_reference_frame_id: str,
    safety_envelope_refs: tuple[SkillDeploymentSafetyEnvelopeRef, ...],
    allowed_client_ids: tuple[str, ...],
    observation_provider: Callable[[], RealSkillMotionObservationCut],
    synchronize_target_builder: RealSkillMotionTargetSynchronizer,
    command_admission: Callable[[TeleopCommand, TeleopObservation, float], bool],
    command_dispatch: Callable[[TeleopCommand], RealSkillMotionDispatchReceipt],
    hold_controller: Callable[[str], RealSkillMotionHoldReceipt],
    emergency_stop_controller: (
        Callable[[str], RealSkillMotionEmergencyStopReceipt] | None
    ) = None,
    measured_qpos_observer: RealSkillMeasuredQposObserver | None = None,
    convergence_profile: RealAgentPrimitiveConvergenceProfile | None = None,
    perception_capture: RealPerceptionCaptureRegistration | None = None,
    localization_registration: RealObjectLocalizationRegistration | None = None,
    localization_provider: ObjectLocalizationProvider | None = None,
    station_point_grasp_registration: StationBoundPointGraspRegistration | None = None,
    station_point_grasp_render_registration: (
        StationBoundPointGraspRenderRegistration | None
    ) = None,
    station_point_grasp_planner: object | None = None,
    station_point_grasp_reachability_evaluator_factory: (
        Callable[[RealAgentMotionPort], Callable[..., object]] | None
    ) = None,
    grasp_trajectory_planner: Callable[..., object] | None = None,
    captured_robot_observation_filter: Callable[..., object] | None = None,
    non_motion_read_window: Callable[[], object] | None = None,
    perception_provider_factory: (
        Callable[
            [RealAgentMotionPort, Callable[[], int]],
            RealSenseAgentPerceptionProvider,
        ]
        | None
    ) = None,
    artifact_store: RecoverableAgentArtifactStore | None = None,
    owns_artifact_store: bool = False,
    grasp_execution_operation_timeout_sec: float = DEFAULT_REAL_GRASP_EXECUTION_OPERATION_TIMEOUT_SEC,
    max_retained_operations: int = 16,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> ExclusiveRealAgentPrimitiveDeployment:
    """Assemble an exclusive Skill owner without constructing RealTeleop."""

    bundle = build_real_agent_primitive_contract_bundle(
        attachment=observation_attachment,
        port_config=port_config,
        frame_graph=frame_graph,
        primitive_reference_frame_id=primitive_reference_frame_id,
        safety_envelope_refs=safety_envelope_refs,
        convergence_profile=convergence_profile,
        perception_capture=perception_capture,
        localization_registration=localization_registration,
        station_point_grasp_registration=station_point_grasp_registration,
        station_point_grasp_render_registration=(
            station_point_grasp_render_registration
        ),
        grasp_execution_operation_timeout_sec=grasp_execution_operation_timeout_sec,
        domain_emergency_stop_available=emergency_stop_controller is not None,
    )
    port = RealAgentMotionPort(
        config=port_config,
        target_builder=target_builder,
        owner_identity=_owner_identity(observation_attachment),
    )
    revision_tracker = AgentStationRevisionTracker()
    perception_provider = (
        None
        if perception_provider_factory is None
        else perception_provider_factory(
            port,
            lambda: revision_tracker.current_revision,
        )
    )
    station_point_grasp_reachability_evaluator = (
        None
        if station_point_grasp_reachability_evaluator_factory is None
        else station_point_grasp_reachability_evaluator_factory(port)
    )
    control_runtime = ExclusiveRealSkillControlRuntime(
        motion_port=port,
        observation_provider=observation_provider,
        synchronize_target_builder=synchronize_target_builder,
        command_admission=command_admission,
        command_dispatch=command_dispatch,
        hold_controller=hold_controller,
        measured_qpos_observer=measured_qpos_observer,
        max_measurement_age_sec=(
            observation_attachment.max_measurement_age_ms / 1_000.0
        ),
        clock=clock,
    )
    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=lambda: _owner_state(port),
        allowed_client_ids=allowed_client_ids,
        max_retained_operations=max_retained_operations,
        perception_provider=perception_provider,
        localization_provider=localization_provider,
        station_point_grasp_planner=station_point_grasp_planner,
        station_point_grasp_reachability_evaluator=(
            station_point_grasp_reachability_evaluator
        ),
        grasp_trajectory_planner=grasp_trajectory_planner,
        captured_robot_observation_filter=captured_robot_observation_filter,
        non_motion_read_window=non_motion_read_window,
        artifact_store=artifact_store,
        owns_artifact_store=owns_artifact_store,
        revision_tracker=revision_tracker,
        emergency_stop=emergency_stop_controller,
        clock=clock,
        sleep=sleep,
    )
    return ExclusiveRealAgentPrimitiveDeployment(
        control_runtime=control_runtime,
        gateway_session=gateway,
    )


def assemble_exclusive_rm65_agent_primitive_deployment(
    *,
    target_builder: RealAgentMotionTargetBuilder,
    rm65_ports: ExclusiveRM65SkillDeployPorts,
    port_config: RealAgentMotionPortConfig,
    observation_attachment: RealTcpPoseObserveAttachmentSpec,
    frame_graph: CoordinateFrameGraph,
    primitive_reference_frame_id: str,
    safety_envelope_refs: tuple[SkillDeploymentSafetyEnvelopeRef, ...],
    allowed_client_ids: tuple[str, ...],
    convergence_profile: RealAgentPrimitiveConvergenceProfile | None = None,
    perception_capture: RealPerceptionCaptureRegistration | None = None,
    localization_registration: RealObjectLocalizationRegistration | None = None,
    localization_provider: ObjectLocalizationProvider | None = None,
    station_point_grasp_registration: StationBoundPointGraspRegistration | None = None,
    station_point_grasp_render_registration: (
        StationBoundPointGraspRenderRegistration | None
    ) = None,
    station_point_grasp_planner: object | None = None,
    station_point_grasp_reachability_evaluator_factory: (
        Callable[[RealAgentMotionPort], Callable[..., object]] | None
    ) = None,
    grasp_trajectory_planner: Callable[..., object] | None = None,
    captured_robot_observation_filter: Callable[..., object] | None = None,
    non_motion_read_window: Callable[[], object] | None = None,
    perception_provider_factory: (
        Callable[
            [RealAgentMotionPort, Callable[[], int]],
            RealSenseAgentPerceptionProvider,
        ]
        | None
    ) = None,
    measured_qpos_observer: RealSkillMeasuredQposObserver | None = None,
    artifact_store: RecoverableAgentArtifactStore | None = None,
    owns_artifact_store: bool = False,
    grasp_execution_operation_timeout_sec: float = DEFAULT_REAL_GRASP_EXECUTION_OPERATION_TIMEOUT_SEC,
    max_retained_operations: int = 16,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> ExclusiveRealAgentPrimitiveDeployment:
    """Bind all real effects to one exact independent RM65 port owner."""

    if not isinstance(rm65_ports, ExclusiveRM65SkillDeployPorts):
        raise TypeError("rm65_ports must be typed.")
    if rm65_ports.target_builder is not target_builder:
        raise ValueError(
            "RM65 ports and Gateway deployment must share one target builder."
        )
    return assemble_exclusive_real_agent_primitive_deployment(
        target_builder=target_builder,
        port_config=port_config,
        observation_attachment=observation_attachment,
        frame_graph=frame_graph,
        primitive_reference_frame_id=primitive_reference_frame_id,
        safety_envelope_refs=safety_envelope_refs,
        allowed_client_ids=allowed_client_ids,
        observation_provider=rm65_ports.observe,
        synchronize_target_builder=rm65_ports.synchronize_target_builder,
        command_admission=rm65_ports.admit_command,
        command_dispatch=rm65_ports.dispatch,
        hold_controller=rm65_ports.hold,
        emergency_stop_controller=rm65_ports.emergency_stop,
        measured_qpos_observer=measured_qpos_observer,
        convergence_profile=convergence_profile,
        perception_capture=perception_capture,
        localization_registration=localization_registration,
        localization_provider=localization_provider,
        station_point_grasp_registration=station_point_grasp_registration,
        station_point_grasp_render_registration=(
            station_point_grasp_render_registration
        ),
        station_point_grasp_planner=station_point_grasp_planner,
        station_point_grasp_reachability_evaluator_factory=(
            station_point_grasp_reachability_evaluator_factory
        ),
        grasp_trajectory_planner=grasp_trajectory_planner,
        captured_robot_observation_filter=captured_robot_observation_filter,
        non_motion_read_window=non_motion_read_window,
        perception_provider_factory=perception_provider_factory,
        artifact_store=artifact_store,
        owns_artifact_store=owns_artifact_store,
        grasp_execution_operation_timeout_sec=grasp_execution_operation_timeout_sec,
        max_retained_operations=max_retained_operations,
        clock=clock,
        sleep=sleep,
    )


__all__ = [
    "ExclusiveRealAgentPrimitiveDeployment",
    "assemble_exclusive_real_agent_primitive_deployment",
    "assemble_exclusive_rm65_agent_primitive_deployment",
]
