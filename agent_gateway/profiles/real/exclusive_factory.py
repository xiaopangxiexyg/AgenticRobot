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
from robot_motion.coordinate_frames import CoordinateFrameGraph
from robot_motion.runtime import TeleopCommand, TeleopObservation
from teleop_runtime.skill_deploy.control_runtime import (
    ExclusiveRealSkillControlRuntime,
    RealSkillMotionDispatchReceipt,
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
    convergence_profile: RealAgentPrimitiveConvergenceProfile | None = None,
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
    )
    port = RealAgentMotionPort(
        config=port_config,
        target_builder=target_builder,
        owner_identity=_owner_identity(observation_attachment),
    )
    control_runtime = ExclusiveRealSkillControlRuntime(
        motion_port=port,
        observation_provider=observation_provider,
        synchronize_target_builder=synchronize_target_builder,
        command_admission=command_admission,
        command_dispatch=command_dispatch,
        hold_controller=hold_controller,
        clock=clock,
    )
    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=lambda: _owner_state(port),
        allowed_client_ids=allowed_client_ids,
        max_retained_operations=max_retained_operations,
        clock=clock,
        sleep=sleep,
    )
    return ExclusiveRealAgentPrimitiveDeployment(
        control_runtime=control_runtime,
        gateway_session=gateway,
    )


__all__ = [
    "ExclusiveRealAgentPrimitiveDeployment",
    "assemble_exclusive_real_agent_primitive_deployment",
]
