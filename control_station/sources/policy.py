"""Typed binding for a lower-owned synchronous policy emission stream."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

from dataset_contracts.canonical_encoding import canonical_content_hash
from control_composition.composition import (
    CompositionInputDraft,
    TemporallySampledCompositionInput,
    build_robot_action_v1_composition_input_draft,
    build_temporally_sampled_composition_input,
)
from control_composition.temporal_sampling import GuaranteedNextComponentAction
from control_composition.activation import CompositionActivationScope
from robot_motion.action.component_action import ComponentActionSchema
from robot_motion.action.component_action_v1 import (
    project_robot_action_v1_to_component_action,
)
from robot_motion.deploy import PolicyActionEmission


SYNC_POLICY_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs2c_policy_emission_binding.v1"
)
SYNC_POLICY_RUNTIME_CONFIG_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs2c_policy_runtime_config.v1"
)
POLICY_COMPOSITION_INPUT_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.policy_composition_input_binding.v1"
)


def _policy_representation_metadata(emission: PolicyActionEmission) -> dict[str, object]:
    """Project protocol lineage without copying a policy action payload."""

    canonical = getattr(emission, "canonical_action", None)
    receipt = (
        None
        if canonical is None
        else getattr(canonical, "canonicalization_receipt", None)
    )
    return {
        "action_payload_kind": getattr(emission, "action_payload_kind", None),
        "representation_trace_id": getattr(
            emission, "representation_trace_id", None
        ),
        "source_binding_hash": getattr(emission, "source_binding_hash", None),
        "canonical_policy_action_hash": getattr(
            canonical, "semantic_content_hash", None
        ),
        "canonical_action_hash": getattr(receipt, "canonical_action_hash", None),
        "canonicalization_receipt_hash": getattr(
            receipt, "semantic_content_hash", None
        ),
    }


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


@dataclass(frozen=True)
class PolicyCompositionInputBinding:
    """Bind a lower-owned policy emission stream to one composition role.

    This binding is intentionally not a target proposal binding.  It verifies
    the already-decoded canonical action and preserves queue/chunk lineage,
    then the companion adapter creates a non-authoritative
    ``CompositionInputDraft``.
    """

    source_slot_id: str
    role: str
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    policy_node_id: str
    policy_clock_domain_id: str
    component_catalog_hash: str
    action_space: str
    required_active_effectors: tuple[str, ...]
    action_schema: ComponentActionSchema
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_slot_id",
            "role",
            "runtime_session_id",
            "control_domain_id",
            "producer_id",
            "policy_node_id",
            "policy_clock_domain_id",
            "component_catalog_hash",
            "action_space",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        epoch = int(self.producer_epoch)
        if isinstance(self.producer_epoch, bool) or epoch != self.producer_epoch:
            raise ValueError("producer_epoch must be an integer.")
        if epoch < 0:
            raise ValueError("producer_epoch must be non-negative.")
        object.__setattr__(self, "producer_epoch", epoch)
        effectors = tuple(str(value).strip() for value in self.required_active_effectors)
        if not effectors or any(not value for value in effectors):
            raise ValueError("required_active_effectors must be non-empty.")
        if len(effectors) != len(set(effectors)):
            raise ValueError("required_active_effectors must be unique.")
        object.__setattr__(self, "required_active_effectors", effectors)
        if not isinstance(self.action_schema, ComponentActionSchema):
            raise TypeError("action_schema must be a ComponentActionSchema.")
        if not set(self.required_active_effectors).issubset(
            set(self.action_schema.component_ids)
        ):
            raise ValueError(
                "Policy binding effectors are outside the action schema."
            )
        payload = {
            "source_slot_id": self.source_slot_id,
            "role": self.role,
            "runtime_session_id": self.runtime_session_id,
            "control_domain_id": self.control_domain_id,
            "producer_id": self.producer_id,
            "producer_epoch": self.producer_epoch,
            "policy_node_id": self.policy_node_id,
            "policy_clock_domain_id": self.policy_clock_domain_id,
            "component_catalog_hash": self.component_catalog_hash,
            "action_space": self.action_space,
            "required_active_effectors": self.required_active_effectors,
            "action_schema_hash": self.action_schema.semantic_content_hash,
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=POLICY_COMPOSITION_INPUT_BINDING_HASH_DOMAIN_V1,
            ),
        )

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
            raise ValueError("Policy emission has no queue-pop evidence.")
        if emission.source_validation is None or not emission.source_validation.accepted:
            raise ValueError("Policy emission source validation is not accepted.")
        if emission.pre_safety_faults:
            raise ValueError(
                "Policy emission contains pre-safety faults: "
                + ",".join(emission.pre_safety_faults)
            )
        action = emission.pre_safety_robot_action
        if action is None:
            raise ValueError("Policy emission has no canonical RobotAction.")
        if action.action_space != self.action_space:
            raise ValueError("Policy emission action space does not match binding.")
        if tuple(action.active_effectors) != self.required_active_effectors:
            raise ValueError(
                "Policy emission must exactly cover the declared composition "
                "components."
            )
        if not (
            emission.effective_from
            <= emission.target_effective_at
            < emission.effective_until
        ):
            raise ValueError("Policy emission target is outside its TTL interval.")
        for name in (
            "produced_at",
            "emitted_at",
            "target_effective_at",
            "effective_from",
            "effective_until",
        ):
            if not math.isfinite(float(getattr(emission, name))):
                raise ValueError(f"Policy emission {name} must be finite.")


class PolicyCompositionInputAdapter:
    """Convert policy queue-pop emissions into formal contributor inputs.

    The adapter owns neither policy prediction nor the policy queue.  Its
    local sequence is only the composition-producer sequence required by the
    receiver; chunk and inference identity remain in ``audit_metadata``.
    """

    def __init__(self, binding: PolicyCompositionInputBinding) -> None:
        if not isinstance(binding, PolicyCompositionInputBinding):
            raise TypeError("binding must be a PolicyCompositionInputBinding.")
        self.binding = binding
        self._lock = threading.RLock()
        self._next_sequence = 0
        self._draft_by_emission_id: dict[str, CompositionInputDraft] = {}
        self._hash_by_emission_id: dict[str, str] = {}
        self._temporal_input_by_key: dict[
            tuple[object, ...],
            TemporallySampledCompositionInput,
        ] = {}

    @property
    def last_sequence(self) -> int:
        with self._lock:
            return self._next_sequence - 1

    def adapt(
        self,
        emission: PolicyActionEmission,
        *,
        activation_scope: CompositionActivationScope | None = None,
        source_fact_ids: tuple[str, ...] | None = None,
        source_fact_hashes: tuple[str, ...] | None = None,
    ) -> CompositionInputDraft:
        self.binding.validate_emission(emission)
        if activation_scope is not None and not isinstance(
            activation_scope,
            CompositionActivationScope,
        ):
            raise TypeError(
                "activation_scope must be a CompositionActivationScope or None."
            )
        with self._lock:
            previous_hash = self._hash_by_emission_id.get(emission.emission_id)
            if previous_hash is not None:
                if previous_hash != emission.semantic_content_hash:
                    raise ValueError(
                        "Policy emission ID was reused with another payload."
                    )
                previous = self._draft_by_emission_id[emission.emission_id]
                if (
                    activation_scope is not None
                    and previous.composition_activation_id
                    != activation_scope.activation_id
                ):
                    raise ValueError(
                        "Policy emission was already adapted under another "
                        "composition activation."
                    )
                return previous
            sequence = self._next_sequence
            self._next_sequence += 1
            action = emission.pre_safety_robot_action
            assert action is not None
            normalized_source_fact_ids = (
                (emission.emission_id,)
                if source_fact_ids is None
                else tuple(source_fact_ids)
            )
            normalized_source_fact_hashes = (
                (emission.semantic_content_hash,)
                if source_fact_hashes is None
                else tuple(source_fact_hashes)
            )
            draft = build_robot_action_v1_composition_input_draft(
                runtime_session_id=self.binding.runtime_session_id,
                control_domain_id=self.binding.control_domain_id,
                producer_id=self.binding.producer_id,
                producer_epoch=self.binding.producer_epoch,
                sequence=sequence,
                input_role=self.binding.role,
                action=action,
                action_schema=self.binding.action_schema,
                produced_at=emission.produced_at,
                produced_clock_domain_id=emission.produced_clock_domain_id,
                effective_from=emission.effective_from,
                effective_until=emission.effective_until,
                effective_clock_domain_id=emission.effective_clock_domain_id,
                policy_output_id=emission.policy_output_id,
                source_fact_ids=normalized_source_fact_ids,
                source_fact_hashes=normalized_source_fact_hashes,
                composition_activation_id=(
                    None
                    if activation_scope is None
                    else activation_scope.activation_id
                ),
                composition_activation_hash=(
                    None
                    if activation_scope is None
                    else activation_scope.semantic_content_hash
                ),
                audit_metadata={
                    "composition_contributor": True,
                    "contributor_kind": "policy",
                    "policy_emission_id": emission.emission_id,
                    "policy_emission_hash": emission.semantic_content_hash,
                    "policy_output_id": emission.policy_output_id,
                    "inference_id": emission.inference_id,
                    "policy_node_id": emission.policy_node_id,
                    "chunk_id": emission.chunk_id,
                    "chunk_keyframe_index": emission.chunk_keyframe_index,
                    "chunk_size": emission.chunk_size,
                    "schedule_spec_hash": emission.schedule_spec_hash,
                    **_policy_representation_metadata(emission),
                    "composition_binding_hash": self.binding.semantic_content_hash,
                },
            )
            self._draft_by_emission_id[emission.emission_id] = draft
            self._hash_by_emission_id[emission.emission_id] = (
                emission.semantic_content_hash
            )
            return draft

    def adapt_interpolated_at(
        self,
        emission: PolicyActionEmission,
        *,
        target_effective_at: float,
        effective_until: float,
        sampled_at: float,
        guaranteed_next_step: GuaranteedNextComponentAction,
        activation_scope: CompositionActivationScope | None = None,
    ) -> TemporallySampledCompositionInput:
        """Interpolate a policy source using an explicit next-step guarantee.

        Callers without such a guarantee must use :meth:`adapt`, which preserves
        the established one-current-step draft, sequence, TTL and identity.
        """

        self.binding.validate_emission(emission)
        if not isinstance(
            guaranteed_next_step,
            GuaranteedNextComponentAction,
        ):
            raise TypeError(
                "guaranteed_next_step must be a GuaranteedNextComponentAction."
            )
        if activation_scope is not None and not isinstance(
            activation_scope,
            CompositionActivationScope,
        ):
            raise TypeError(
                "activation_scope must be a CompositionActivationScope or None."
            )
        target = float(target_effective_at)
        until = float(effective_until)
        sampled = float(sampled_at)
        if not all(math.isfinite(value) for value in (target, until, sampled)):
            raise ValueError("Temporal policy sample times must be finite.")
        if until <= target:
            raise ValueError("effective_until must follow target_effective_at.")
        if target >= guaranteed_next_step.target_effective_at:
            raise ValueError(
                "Policy temporal interpolation requires a target before the "
                "guaranteed next step; use that step as current at its target."
            )
        if until > guaranteed_next_step.target_effective_at:
            raise ValueError(
                "Interpolated policy input validity cannot exceed the "
                "guaranteed-next target."
            )
        next_hash = guaranteed_next_step.semantic_content_hash
        activation_id = (
            None if activation_scope is None else activation_scope.activation_id
        )
        cache_key = (
            emission.emission_id,
            emission.semantic_content_hash,
            target,
            until,
            sampled,
            next_hash,
            activation_id,
        )
        with self._lock:
            cached = self._temporal_input_by_key.get(cache_key)
            if cached is not None:
                return cached
            action = emission.pre_safety_robot_action
            assert action is not None
            result = build_temporally_sampled_composition_input(
                current_action=project_robot_action_v1_to_component_action(
                    action,
                    action_schema=self.binding.action_schema,
                ),
                current_target_effective_at=emission.target_effective_at,
                current_source_fact_id=emission.emission_id,
                current_source_fact_hash=emission.semantic_content_hash,
                target_effective_at=target,
                effective_until=until,
                guaranteed_next_step=guaranteed_next_step,
                runtime_session_id=self.binding.runtime_session_id,
                control_domain_id=self.binding.control_domain_id,
                producer_id=self.binding.producer_id,
                producer_epoch=self.binding.producer_epoch,
                sequence=self._next_sequence,
                input_role=self.binding.role,
                produced_at=sampled,
                clock_domain_id=emission.effective_clock_domain_id,
                policy_output_id=emission.policy_output_id,
                composition_activation_id=activation_id,
                composition_activation_hash=(
                    None
                    if activation_scope is None
                    else activation_scope.semantic_content_hash
                ),
                audit_metadata={
                    "composition_contributor": True,
                    "contributor_kind": "policy",
                    "policy_emission_id": emission.emission_id,
                    "policy_emission_hash": emission.semantic_content_hash,
                    "policy_output_id": emission.policy_output_id,
                    "inference_id": emission.inference_id,
                    "policy_node_id": emission.policy_node_id,
                    "chunk_id": emission.chunk_id,
                    "chunk_keyframe_index": emission.chunk_keyframe_index,
                    "chunk_size": emission.chunk_size,
                    "schedule_spec_hash": emission.schedule_spec_hash,
                    **_policy_representation_metadata(emission),
                    "composition_binding_hash": self.binding.semantic_content_hash,
                },
            )
            self._next_sequence += 1
            self._temporal_input_by_key[cache_key] = result
            return result


__all__ = [
    "POLICY_COMPOSITION_INPUT_BINDING_HASH_DOMAIN_V1",
    "PolicyCompositionInputAdapter",
    "PolicyCompositionInputBinding",
    "SYNC_POLICY_BINDING_HASH_DOMAIN_V1",
    "SYNC_POLICY_RUNTIME_CONFIG_HASH_DOMAIN_V1",
    "SyncPolicyEmissionBinding",
]
