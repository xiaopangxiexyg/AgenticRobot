"""CubeLift policy-source binding for the generic MuJoCo observation owner."""

from __future__ import annotations

from typing import Any

from agent_gateway.runtime.station_revision import (
    AgentStationRevisionTracker,
)

from ...observation import MujocoAgentObservationProducer
from ...observation_binding import MujocoAgentObservationSourceBinding
from ...state_deployment import MujocoPassiveTcpPoseStateDeploymentBinding
from .deployment import MujocoCubeLiftAgentSkillContractBundle


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def build_mujoco_cube_lift_observation_source_binding(
    *,
    application: Any,
    contract_bundle: MujocoCubeLiftAgentSkillContractBundle,
) -> MujocoAgentObservationSourceBinding:
    """Freeze the CubeLift policy input source selected by the application."""

    if not isinstance(contract_bundle, MujocoCubeLiftAgentSkillContractBundle):
        raise TypeError("contract_bundle must be a CubeLift contract bundle.")
    policy_binding = getattr(
        getattr(
            getattr(application, "composition", None),
            "policy_input_adapter",
            None,
        ),
        "binding",
        None,
    )
    binding = contract_bundle.deployment_binding
    return MujocoAgentObservationSourceBinding(
        runtime_session_id=binding.runtime_session_id,
        control_domain_id=binding.control_domain_id,
        source_binding_id=_required(
            getattr(policy_binding, "source_slot_id", None),
            field_name="policy source slot",
        ),
        source_binding_hash=_required(
            getattr(policy_binding, "semantic_content_hash", None),
            field_name="policy source binding hash",
        ),
        producer_epoch=getattr(policy_binding, "producer_epoch", None),
        source_kind="policy",
    )


class MujocoCubeLiftAgentObservationProducer(MujocoAgentObservationProducer):
    """Task-owned constructor over the generic source-bound producer."""

    def __init__(
        self,
        *,
        application: Any,
        contract_bundle: MujocoCubeLiftAgentSkillContractBundle,
        revision_tracker: AgentStationRevisionTracker | None = None,
    ) -> None:
        super().__init__(
            application=application,
            contract_bundle=contract_bundle,
            source_binding=build_mujoco_cube_lift_observation_source_binding(
                application=application,
                contract_bundle=contract_bundle,
            ),
            revision_tracker=revision_tracker,
        )


    def _resolve_observation_component(self, *, application: Any, contract_bundle: Any) -> str:
        # CubeLift retains a task/bimanual action contract. Only its passive
        # observation projection is right-arm TCP; it is not a TCP motion session.
        if not isinstance(contract_bundle, MujocoCubeLiftAgentSkillContractBundle):
            raise TypeError("contract_bundle must be a CubeLift contract bundle.")
        state_binding = contract_bundle.state_deployment_binding
        if (not isinstance(state_binding, MujocoPassiveTcpPoseStateDeploymentBinding)
                or state_binding.component_id != "right_arm"):
            raise ValueError("CubeLift observation requires its typed right_arm state binding.")
        return "right_arm"


__all__ = [
    "MujocoCubeLiftAgentObservationProducer",
    "build_mujoco_cube_lift_observation_source_binding",
]
