"""Typed binding for a lower-owned synchronous policy emission stream."""

from __future__ import annotations

from dataclasses import dataclass

from robot_motion.deploy import PolicyActionEmission


SYNC_POLICY_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs2c_policy_emission_binding.v1"
)
SYNC_POLICY_RUNTIME_CONFIG_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs2c_policy_runtime_config.v1"
)


@dataclass(frozen=True)
class SyncPolicyEmissionBinding:
    """Typed context for one lower-owned sync policy emission stream."""

    source_slot_id: str
    producer_id: str
    producer_epoch: int
    policy_node_id: str
    policy_clock_domain_id: str
    policy_runtime_config_hash: str
    action_adapter_contract_hash: str
    semantic_content_hash: str

    def validate_emission(self, emission: PolicyActionEmission) -> None:
        if not isinstance(emission, PolicyActionEmission):
            raise TypeError("emission must be a PolicyActionEmission.")
        if emission.policy_node_id != self.policy_node_id:
            raise ValueError("Policy emission belongs to another policy node.")
        if (
            emission.produced_clock_domain_id != self.policy_clock_domain_id
            or emission.effective_clock_domain_id != self.policy_clock_domain_id
        ):
            raise ValueError("Policy emission clock domain does not match binding.")
        if emission.queue_pop_count <= 0:
            raise ValueError("Policy emission has no lower-owned queue-pop evidence.")
        receipt = emission.representation_receipt
        if receipt is not None and (
            receipt.adapter_contract_hash != self.action_adapter_contract_hash
        ):
            raise ValueError("Policy emission used another action adapter contract.")


__all__ = [
    "SYNC_POLICY_BINDING_HASH_DOMAIN_V1",
    "SYNC_POLICY_RUNTIME_CONFIG_HASH_DOMAIN_V1",
    "SyncPolicyEmissionBinding",
]
