"""Explicit composition of the real 6+3 Gateway into RealTeleop."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from agent_gateway.protocol.skill_catalog import SkillDeploymentSafetyEnvelopeRef
from robot_motion.coordinate_frames import CoordinateFrameGraph
from teleop_runtime.real_teleop.agent_motion_port import RealAgentMotionPortConfig
from teleop_runtime.real_teleop.agent_motion_runtime import (
    attach_prepared_real_agent_motion_port,
    build_real_agent_motion_port,
)
from teleop_runtime.real_teleop.components import RealTeleopComponents

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
class RealAgentPrimitiveRealTeleopAttachment:
    """The returned components and Gateway share one lower motion port."""

    components: RealTeleopComponents
    gateway_session: RealAgentPrimitiveGatewaySession

    def __post_init__(self) -> None:
        if not isinstance(self.components, RealTeleopComponents):
            raise TypeError("components must be RealTeleopComponents.")
        if not isinstance(
            self.gateway_session,
            RealAgentPrimitiveGatewaySession,
        ):
            raise TypeError("gateway_session must be typed.")
        if self.components.agent_motion_port is not self.gateway_session.port:
            raise ValueError(
                "RealTeleop and Agent Gateway must share one lower motion port."
            )


def attach_real_agent_primitive_gateway(
    components: RealTeleopComponents,
    *,
    port_config: RealAgentMotionPortConfig,
    observation_attachment: RealTcpPoseObserveAttachmentSpec,
    frame_graph: CoordinateFrameGraph,
    primitive_reference_frame_id: str,
    safety_envelope_refs: tuple[SkillDeploymentSafetyEnvelopeRef, ...],
    owner_state_provider: Callable[[], RealControlOwnerState],
    allowed_client_ids: tuple[str, ...],
    convergence_profile: RealAgentPrimitiveConvergenceProfile | None = None,
    max_retained_operations: int = 16,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> RealAgentPrimitiveRealTeleopAttachment:
    """Preflight and attach the default-off real 6+3 application.

    The function starts no socket, thread, control loop, device, or motion.
    The caller must use the returned ``components`` in the existing RealTeleop
    owner loop; retaining the original aggregate leaves the Gateway unattached.
    """

    bundle = build_real_agent_primitive_contract_bundle(
        attachment=observation_attachment,
        port_config=port_config,
        frame_graph=frame_graph,
        primitive_reference_frame_id=primitive_reference_frame_id,
        safety_envelope_refs=safety_envelope_refs,
        convergence_profile=convergence_profile,
    )
    port = build_real_agent_motion_port(components, config=port_config)
    gateway = assemble_real_agent_primitive_gateway_session(
        contract_bundle=bundle,
        port=port,
        owner_state_provider=owner_state_provider,
        allowed_client_ids=allowed_client_ids,
        max_retained_operations=max_retained_operations,
        clock=clock,
        sleep=sleep,
    )
    attached_components = attach_prepared_real_agent_motion_port(
        components,
        port=port,
    )
    return RealAgentPrimitiveRealTeleopAttachment(
        components=attached_components,
        gateway_session=gateway,
    )


__all__ = [
    "RealAgentPrimitiveRealTeleopAttachment",
    "attach_real_agent_primitive_gateway",
]
