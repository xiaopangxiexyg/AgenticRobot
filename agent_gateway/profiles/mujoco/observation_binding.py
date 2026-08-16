"""Frozen source selection for reusable MuJoCo Agent observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .motion_deployment import MujocoAgentTcpPoseContractBundle


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


@dataclass(frozen=True)
class MujocoAgentObservationSourceBinding:
    """Exact lower source identity selected before producer construction."""

    runtime_session_id: str
    control_domain_id: str
    source_binding_id: str
    source_binding_hash: str
    producer_epoch: int
    source_kind: str

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_session_id",
            "control_domain_id",
            "source_binding_id",
            "source_binding_hash",
            "source_kind",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if (
            isinstance(self.producer_epoch, bool)
            or not isinstance(self.producer_epoch, int)
            or self.producer_epoch < 0
        ):
            raise ValueError("producer_epoch must be non-negative.")


def build_mujoco_agent_motion_observation_source_binding(
    *,
    application: Any,
    contract_bundle: MujocoAgentTcpPoseContractBundle,
) -> MujocoAgentObservationSourceBinding:
    """Bind the direct Agent Skill source slot used by the motion Route."""

    from .motion_deployment import MujocoAgentTcpPoseContractBundle

    if not isinstance(contract_bundle, MujocoAgentTcpPoseContractBundle):
        raise TypeError("contract_bundle must be a TCP-pose contract bundle.")
    runtime = getattr(application, "runtime", None)
    runtime_config = getattr(runtime, "config", None)
    binding = contract_bundle.deployment_binding
    source_slots = tuple(contract_bundle.station_manifest.source_slots)
    direct = getattr(runtime_config, "direct_component_proposal_source", None)
    if (
        len(source_slots) != 1
        or source_slots[0].source_kind != "agent_skill"
        or getattr(direct, "producer_kind", None) != "agent_skill"
        or source_slots[0].producer_id != getattr(direct, "producer_id", None)
    ):
        raise ValueError(
            "Agent motion observation source does not match its Station slot."
        )
    return MujocoAgentObservationSourceBinding(
        runtime_session_id=binding.runtime_session_id,
        control_domain_id=binding.control_domain_id,
        source_binding_id=source_slots[0].source_slot_id,
        source_binding_hash=source_slots[0].semantic_content_hash,
        producer_epoch=getattr(direct, "producer_epoch", None),
        source_kind="agent_skill",
    )


__all__ = [
    "MujocoAgentObservationSourceBinding",
    "build_mujoco_agent_motion_observation_source_binding",
]
