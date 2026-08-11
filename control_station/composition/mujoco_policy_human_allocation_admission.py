"""Receiver admission for policy/human allocation contributor inputs.

This activation-local adapter reuses the existing policy and passive-human
source adapters. It records their immutable facts and forwards inputs to the
one MuJoCo composition receiver. It owns no source polling, anchor decision,
allocation evaluation, proposal, authority, safety, or execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from control_composition.activation import CompositionActivationScope
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.deploy import PolicyActionEmission, project_policy_action_emission
from teleop_runtime.action.human_proposal import (
    HumanCommandEmission,
    HumanProposalAdapter,
)

from ..sources.human_component_input import (
    adapt_human_emission_to_component_action,
)
from ..sources.policy import PolicyCompositionInputAdapter
from .mujoco_policy_human_allocation_input import (
    MujocoPolicyHumanAllocationInputController,
    MujocoPolicyHumanAllocationInputCycle,
)
from .mujoco_sim_hil import MujocoSimHILInputSubmission


MUJOCO_ALLOCATION_HUMAN_SUBMISSION_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.control_station.mujoco_allocation_human_submission.development"
)


@dataclass(frozen=True)
class MujocoAllocationHumanInputSubmission:
    source_emission_id: str
    roles: tuple[str, ...]
    input_cycle: MujocoPolicyHumanAllocationInputCycle | None = None
    error: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        emission_id = str(self.source_emission_id).strip()
        roles = tuple(str(role).strip() for role in self.roles)
        if not emission_id:
            raise ValueError("source_emission_id must be non-empty.")
        if not roles or any(not role for role in roles) or len(roles) != len(set(roles)):
            raise ValueError("roles must be non-empty and unique.")
        if self.input_cycle is None and not str(self.error or "").strip():
            raise ValueError("A missing input cycle requires an error.")
        object.__setattr__(self, "source_emission_id", emission_id)
        object.__setattr__(self, "roles", roles)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "source_emission_id": emission_id,
                    "roles": roles,
                    "input_cycle_status": (
                        None if self.input_cycle is None else self.input_cycle.status
                    ),
                    "batch_hash": (
                        None
                        if self.input_cycle is None or self.input_cycle.batch is None
                        else self.input_cycle.batch.semantic_content_hash
                    ),
                    "publication_status": (
                        None
                        if self.publication is None
                        else self.publication.status
                    ),
                    "error": self.error,
                },
                domain=(MUJOCO_ALLOCATION_HUMAN_SUBMISSION_HASH_DOMAIN_DEVELOPMENT),
            ),
        )

    @property
    def accepted(self) -> bool:
        return bool(
            self.input_cycle is not None
            and self.input_cycle.status == "batch_admitted"
        )

    @property
    def publication(self) -> Any | None:
        return None if self.input_cycle is None else self.input_cycle.publication


class MujocoPolicyHumanAllocationInputAdmission:
    """Publish contributor inputs under one exact composition activation."""

    def __init__(
        self,
        *,
        runtime: Any,
        activation_scope: CompositionActivationScope,
        policy_input_adapter: PolicyCompositionInputAdapter,
        human_adapter: HumanProposalAdapter,
        human_input_controller: MujocoPolicyHumanAllocationInputController,
    ) -> None:
        if not isinstance(activation_scope, CompositionActivationScope):
            raise TypeError("activation_scope must be typed.")
        if not isinstance(policy_input_adapter, PolicyCompositionInputAdapter):
            raise TypeError("policy_input_adapter must be typed.")
        if not isinstance(human_adapter, HumanProposalAdapter):
            raise TypeError("human_adapter must be typed.")
        if not isinstance(
            human_input_controller,
            MujocoPolicyHumanAllocationInputController,
        ):
            raise TypeError("human_input_controller must be typed.")
        for method in (
            "record_external_facts",
            "submit_composition_input",
        ):
            if not callable(getattr(runtime, method, None)):
                raise TypeError(f"runtime must expose {method}().")
        config = getattr(runtime, "config", None)
        for name in (
            "runtime_session_id",
            "control_domain_id",
            "embodiment_instance_id",
            "clock_domain_id",
        ):
            if not str(getattr(config, name, "")).strip():
                raise TypeError(f"runtime.config must expose {name}.")
        if human_input_controller.activation_scope.semantic_content_hash != (
            activation_scope.semantic_content_hash
        ):
            raise ValueError("Human input controller belongs to another activation.")
        self.runtime = runtime
        self.activation_scope = activation_scope
        self.policy_input_adapter = policy_input_adapter
        self.human_adapter = human_adapter
        self.human_input_controller = human_input_controller
        self._fact_hash_by_id: dict[str, str] = {}
        self._policy_submission_by_emission_id: dict[
            str,
            MujocoSimHILInputSubmission,
        ] = {}
        self._policy_hash_by_emission_id: dict[str, str] = {}

    def _record_facts(self, facts: tuple[ControlTraceFact, ...]) -> None:
        pending = []
        for fact in facts:
            previous = self._fact_hash_by_id.get(fact.fact_id)
            if previous is not None:
                if previous != fact.semantic_content_hash:
                    raise ValueError("One source fact ID has conflicting content.")
                continue
            pending.append(fact)
        if pending:
            self.runtime.record_external_facts(tuple(pending))
            self._fact_hash_by_id.update(
                {fact.fact_id: fact.semantic_content_hash for fact in pending}
            )

    def publish_policy(
        self,
        emission: PolicyActionEmission,
        observed_at: float,
    ) -> MujocoSimHILInputSubmission:
        if not isinstance(emission, PolicyActionEmission):
            raise TypeError("emission must be PolicyActionEmission.")
        cached_hash = self._policy_hash_by_emission_id.get(emission.emission_id)
        if cached_hash is not None:
            if cached_hash != emission.semantic_content_hash:
                raise ValueError("Policy emission ID has conflicting content.")
            return self._policy_submission_by_emission_id[emission.emission_id]
        try:
            config = self.runtime.config
            facts = project_policy_action_emission(
                emission,
                runtime_session_id=config.runtime_session_id,
                control_domain_id=config.control_domain_id,
                embodiment_instance_id=config.embodiment_instance_id,
                clock_domain_id=config.clock_domain_id,
            )
            self._record_facts(facts)
            self.policy_input_adapter.binding.validate_emission(emission)
            draft = self.policy_input_adapter.adapt(
                emission,
                activation_scope=self.activation_scope,
                source_fact_ids=tuple(fact.fact_id for fact in facts),
                source_fact_hashes=tuple(
                    fact.semantic_content_hash for fact in facts
                ),
            )
            receipt = self.runtime.submit_composition_input(
                draft,
                activation_scope=self.activation_scope,
            )
            submission = MujocoSimHILInputSubmission(
                role=self.policy_input_adapter.binding.role,
                source_emission_id=emission.emission_id,
                input_receipt=receipt,
                error=(
                    None
                    if receipt.accepted
                    else receipt.message
                    or receipt.rejection_code
                    or "composition_input_rejected"
                ),
            )
        except Exception as error:  # noqa: BLE001 - typed source failure result.
            submission = MujocoSimHILInputSubmission(
                role=self.policy_input_adapter.binding.role,
                source_emission_id=emission.emission_id,
                error=f"policy_contributor_error:{type(error).__name__}:{error}",
            )
        self._policy_hash_by_emission_id[emission.emission_id] = (
            emission.semantic_content_hash
        )
        self._policy_submission_by_emission_id[emission.emission_id] = submission
        return submission

    def publish_human(
        self,
        emission: HumanCommandEmission,
        observed_at: float,
        target_effective_at: float,
    ) -> MujocoAllocationHumanInputSubmission:
        if not isinstance(emission, HumanCommandEmission):
            raise TypeError("emission must be HumanCommandEmission.")
        controller = self.human_input_controller
        roles = (
            controller.bundle.config.absolute_overlay_role,
            controller.bundle.config.residual_role,
        )
        try:
            config = self.runtime.config
            effective_until = float(observed_at) + float(
                config.proposal_horizon_sec
            )
            adaptation = adapt_human_emission_to_component_action(
                adapter=self.human_adapter,
                emission=emission,
                effective_from=observed_at,
                effective_until=effective_until,
                target_effective_at=target_effective_at,
                runtime_session_id=config.runtime_session_id,
                control_domain_id=config.control_domain_id,
                embodiment_instance_id=config.embodiment_instance_id,
                occurred_at=observed_at,
                clock_domain_id=config.clock_domain_id,
            )
            self._record_facts(adaptation.trace_facts)
            result = adaptation.adapter_result
            proposal = result.proposal_draft
            action = adaptation.action
            source_id = result.source_adapter_receipt.source_emission_id
            if action is None or proposal is None:
                reasons = ",".join(result.source_validation.reasons)
                return MujocoAllocationHumanInputSubmission(
                    source_emission_id=source_id,
                    roles=roles,
                    error=f"human_source_validation_failed:{reasons or 'no_action'}",
                )
            cycle = controller.process_and_publish(
                action,
                source_action_id=source_id,
                source_action_hash=action.semantic_content_hash,
                source_sequence=proposal.sequence,
                source_available_at=observed_at,
                measured_at=proposal.produced_at,
                measured_clock_domain_id=proposal.produced_clock_domain_id,
                observed_at=observed_at,
                observed_clock_domain_id=config.clock_domain_id,
                runtime_session_id=config.runtime_session_id,
                control_domain_id=config.control_domain_id,
                produced_at=proposal.produced_at,
                produced_clock_domain_id=proposal.produced_clock_domain_id,
                effective_from=proposal.effective_from,
                effective_until=proposal.effective_until,
                effective_clock_domain_id=proposal.effective_clock_domain_id,
                context_ids=proposal.context_ids,
                observation_cut_id=proposal.observation_cut_id,
            )
            actual_source_id = (
                source_id
                if cycle.derivation is None
                else cycle.derivation.profile_step.source_action_id
            )
            error = None
            if cycle.status != "batch_admitted":
                error = f"human_allocation_input_{cycle.status}"
            return MujocoAllocationHumanInputSubmission(
                source_emission_id=actual_source_id,
                roles=roles,
                input_cycle=cycle,
                error=error,
            )
        except Exception as error:  # noqa: BLE001 - typed source failure result.
            return MujocoAllocationHumanInputSubmission(
                source_emission_id=f"human-emission:{emission.emission_sequence}",
                roles=roles,
                error=f"human_contributor_error:{type(error).__name__}:{error}",
            )


__all__ = [
    "MUJOCO_ALLOCATION_HUMAN_SUBMISSION_HASH_DOMAIN_DEVELOPMENT",
    "MujocoAllocationHumanInputSubmission",
    "MujocoPolicyHumanAllocationInputAdmission",
]
