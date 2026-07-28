"""Default-off formal pose composition profile for deterministic MuJoCo.

Human and replay are immutable contributors, never parallel proposal owners.
One composer producer may promote an exact ready composition evaluation; missing
required contributors, numerical conflicts, or promotion errors transition the
shared execution gate to HOLD.  This module owns no device, replay cursor,
policy worker, canonical safety implementation, IK, dispatcher, or episode.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Callable, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from control_composition.activation import (
    ACTIVATION_LOCAL_STRICT_SOURCE_FRESHNESS_POLICY_V2,
    CompositionActivationScope,
    build_composition_activation_scope,
)
from control_composition.composition import (
    CompositionAuthorityDisposition,
    CompositionAuthorityGateSpec,
    CompositionInputAdmissionReceipt,
    CompositionInputDraft,
    CompositionSpec,
    decide_composition_authority_disposition,
)
from control_composition.observability import (
    SourceDifferenceObservation,
    SourceDifferenceSpec,
)
from control_composition.runtime import CompositeProposalSubmission
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    ControlExecutionManifest,
    ExecutionGateTransitionOutcome,
    ExecutionProfileExclusion,
    build_control_execution_manifest,
)
from robot_motion.action.component_action_v1 import (
    project_robot_action_v1_to_component_action,
)
from dataset_interface.composition_control_trace import (
    project_composition_activation_scope,
    project_composition_authority_disposition,
)
from control_composition.lifecycle import (
    CompositionSessionOperationSpec,
    CompositionSessionRearmReceipt,
    CompositionSessionRearmSpec,
    CompositionSessionState,
)
from robot_motion.replay import ReplayProposalSource, ReplaySourceFrameEmission
from robot_motion.replay.control_trace import (
    project_replay_adapter_result,
    project_replay_source_frame_emission,
)
from sim.mujoco.deploy import MujocoDeployRuntime
from teleop_runtime.action.human_proposal import (
    HumanCommandEmission,
    HumanProposalAdapter,
)
from teleop_runtime.control_trace.projection import (
    project_human_proposal_adapter_result,
)

from ..adapters import StationOperationAdapter
from ..composition_session import CompositionSessionController
from ..contracts import (
    ControlStationRunManifest,
    ExecutionDomainBinding,
    SourceSlotSpec,
    StationIntent,
    StationOperationResultDraft,
)
from ..control_trace import (
    project_station_intent_transition,
    project_station_operation_transition,
)
from ..runtime import ControlStationRuntime, StationOperationExecution
from ..state_machine import (
    COMPOSITION_SESSION_OPERATION_HELD_STATUS_V1,
    COMPOSITION_SESSION_OPERATION_READY_STATUS_V1,
    COMPOSITION_SESSION_OPERATION_REARMED_STATUS_V1,
    COMPOSITION_SESSION_OPERATION_STOPPED_STATUS_V1,
    ControlStationTransition,
)
from ..sources.human import HumanPoseSourceContract
from ..sources.replay import ReplaySourceObserverBinding
from ..targets.authoritative import (
    authoritative_authority_epoch_source_id,
    authoritative_execution_epoch_source_id,
    authoritative_owner_registry_id,
)
from .mujoco_human import build_mujoco_pose_human_proposal_adapter


CS6B1H_HUMAN_POSE_ROLE_V1 = "human_pose"
CS6B1H_REPLAY_POSE_ROLE_V1 = "replay_pose"
CS6B1H_BINDING_HASH_DOMAIN_V1 = "dexteleop.cs6b1h_pose_composer_binding.v1"
CS6B1H_BINDING_HASH_DOMAIN_V2 = "dexteleop.cs6b1h_pose_composer_binding.v2"
CS6B1H_STEP_HASH_DOMAIN_V1 = "dexteleop.cs6b1h_pose_composer_step.v1"
CS6B1I_COMPOSITION_SESSION_MANIFEST_ID_V1 = (
    "control_station_cs6b1i_pose_composition_session_mujoco_v1"
)
CS6B1I_COMPOSITION_SESSION_EXECUTION_MANIFEST_ID_V1 = (
    "cs6b1i_pose_composition_session_sim_v1"
)
CS6B1I_COMPOSITION_SESSION_TRACE_PROFILE_ID_V1 = (
    "deployable_sim_scheduled_pose_composer_station_v1"
)
CS6B1I_SIM_REARM_READINESS_CHECKS_V1 = (
    "activation_input_barrier_reset",
    "composer_authority_epoch_advanced",
    "composer_producer_epoch_advanced",
    "execution_cancellation_complete",
    "execution_gate_ready",
    "measured_state_fresh",
)


@dataclass(frozen=True)
class CS6b1hPoseComposerBinding:
    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    composer_producer_id: str
    human_producer_id: str
    replay_producer_id: str
    composition_spec_hash: str
    authority_gate_spec_hash: str
    target_runtime_contract_hash: str
    semantic_content_hash: str
    source_difference_spec_hash: str | None = None
    formal_strategy_spec_hash: str | None = None


@dataclass(frozen=True)
class CS6b1hFormalCompositionStep:
    decision_key: str
    target_effective_at: float
    evaluation: CompositeProposalSubmission | None
    disposition: CompositionAuthorityDisposition | None
    proposal_submission: CompositeProposalSubmission | None
    hold_transition: ExecutionGateTransitionOutcome | None
    blocked_reason: str | None
    error: str | None
    semantic_content_hash: str

    @property
    def proposal_admission(self):
        if self.proposal_submission is None:
            return None
        return self.proposal_submission.proposal_admission

    @property
    def source_difference_observation(
        self,
    ) -> SourceDifferenceObservation | None:
        if self.evaluation is None:
            return None
        return self.evaluation.source_difference_observation

    @property
    def source_difference_error(self) -> str | None:
        if self.evaluation is None:
            return None
        return self.evaluation.source_difference_error


class MujocoPoseComposerAuthorityComposition:
    """Bind human/replay contributors to one formal pose composer producer."""

    def __init__(
        self,
        *,
        runtime: MujocoDeployRuntime,
        human_adapter: HumanProposalAdapter,
        replay_source: ReplayProposalSource,
        composition_spec: CompositionSpec,
        authority_gate_spec: CompositionAuthorityGateSpec,
        source_difference_spec: SourceDifferenceSpec | None = None,
    ) -> None:
        if source_difference_spec is not None and not isinstance(
            source_difference_spec,
            SourceDifferenceSpec,
        ):
            raise TypeError(
                "source_difference_spec must be a SourceDifferenceSpec or None."
            )
        self.runtime = runtime
        self.human_adapter = human_adapter
        self.replay_source = replay_source
        self.composition_spec = composition_spec
        self.authority_gate_spec = authority_gate_spec
        self.source_difference_spec = source_difference_spec
        self._lock = threading.RLock()
        self._started = False
        self._input_receipt_by_role: dict[
            str, CompositionInputAdmissionReceipt
        ] = {}
        self._human_fact_time_by_emission_id: dict[str, float] = {}
        self._step_by_decision_key: dict[str, CS6b1hFormalCompositionStep] = {}
        self._composition_activation_scope: (
            CompositionActivationScope | None
        ) = None
        self._legacy_unscoped_input_barrier_time: float | None = None
        self.binding = self._validate_binding()

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    @property
    def input_receipt_by_role(
        self,
    ) -> Mapping[str, CompositionInputAdmissionReceipt]:
        with self._lock:
            return dict(self._input_receipt_by_role)

    @property
    def composition_activation_scope(
        self,
    ) -> CompositionActivationScope | None:
        with self._lock:
            return self._composition_activation_scope

    def _validate_binding(self) -> CS6b1hPoseComposerBinding:
        runtime = self.runtime
        config = runtime.config
        if not runtime.pose_execution_enabled:
            raise ValueError("CS6b1h requires the pose execution profile.")
        if not runtime.explicit_shared_safety:
            raise ValueError("CS6b1h requires an explicit canonical safety stage.")
        if (
            not config.enable_composer_authority
            or config.enable_replay_authority
            or config.enable_primary_policy_authority
        ):
            raise ValueError(
                "CS6b1h requires composer-only proposal authority."
            )
        spec = self.composition_spec
        if self.source_difference_spec is not None:
            self.source_difference_spec.validate_composition_spec(spec)
        if (
            spec.algebra != "weighted_se3_pose_fusion"
            or spec.input_roles
            != (
                CS6B1H_HUMAN_POSE_ROLE_V1,
                CS6B1H_REPLAY_POSE_ROLE_V1,
            )
            or spec.required_roles != spec.input_roles
        ):
            raise ValueError(
                "CS6b1h v1 requires fail-closed human_pose/replay_pose fusion."
            )
        gate = self.authority_gate_spec
        gate.validate_composition_spec(spec)
        if gate.composer_producer_id != config.composer_producer_id:
            raise ValueError("Composer producer does not match the authority gate.")
        if not math.isclose(
            gate.proposal_ttl_sec,
            config.proposal_horizon_sec,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Authority-gate proposal TTL mismatches the runtime.")
        human = self.human_adapter.config
        replay = self.replay_source.config
        expected_scope = (
            config.runtime_session_id,
            config.control_domain_id,
            config.embodiment_instance_id,
            config.producer_epoch,
        )
        if (
            human.runtime_session_id,
            human.control_domain_id,
            human.embodiment_instance_id,
            human.producer_epoch,
        ) != expected_scope:
            raise ValueError("Human adapter belongs to another runtime scope.")
        if human.producer_id != config.producer_id:
            raise ValueError("Human contributor producer does not match runtime.")
        if (
            replay.runtime_session_id,
            replay.control_domain_id,
            replay.embodiment_instance_id,
            replay.producer_epoch,
        ) != expected_scope:
            raise ValueError("Replay contributor belongs to another runtime scope.")
        if replay.producer_id != config.replay_producer_id:
            raise ValueError("Replay contributor producer does not match runtime.")
        if human.component_catalog.semantic_content_hash != (
            runtime.catalog.semantic_content_hash
        ):
            raise ValueError("Human contributor action catalog mismatches runtime.")
        target_hash = runtime.target_runtime_action_contract.semantic_content_hash
        if replay.target_runtime_contract.semantic_content_hash != target_hash:
            raise ValueError("Replay contributor was assessed against another target.")
        formal_strategy = runtime.policy_topology.composition_strategy
        formal_strategy_spec_hash = (
            None
            if formal_strategy is None
            else formal_strategy.spec.semantic_content_hash
        )
        if (
            formal_strategy is not None
            and formal_strategy.spec.composition_spec_hash
            != spec.semantic_content_hash
        ):
            raise ValueError(
                "Formal strategy is bound to another composition spec."
            )
        payload = {
            "runtime_session_id": config.runtime_session_id,
            "control_domain_id": config.control_domain_id,
            "embodiment_instance_id": config.embodiment_instance_id,
            "composer_producer_id": config.composer_producer_id,
            "human_producer_id": human.producer_id,
            "replay_producer_id": replay.producer_id,
            "composition_spec_hash": spec.semantic_content_hash,
            "authority_gate_spec_hash": gate.semantic_content_hash,
            "target_runtime_contract_hash": target_hash,
        }
        hash_payload = dict(payload)
        difference_spec_hash = (
            None
            if self.source_difference_spec is None
            else self.source_difference_spec.semantic_content_hash
        )
        if difference_spec_hash is not None:
            hash_payload["source_difference_spec_hash"] = difference_spec_hash
        binding_hash_domain = CS6B1H_BINDING_HASH_DOMAIN_V1
        if formal_strategy_spec_hash is not None:
            hash_payload["formal_strategy_spec_hash"] = (
                formal_strategy_spec_hash
            )
            binding_hash_domain = CS6B1H_BINDING_HASH_DOMAIN_V2
        return CS6b1hPoseComposerBinding(
            **payload,
            semantic_content_hash=canonical_content_hash(
                hash_payload,
                domain=binding_hash_domain,
            ),
            source_difference_spec_hash=difference_spec_hash,
            formal_strategy_spec_hash=formal_strategy_spec_hash,
        )

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self.runtime.start()
            try:
                self.runtime.activate_policy_node_producer(
                    self.replay_source.config.producer_id,
                    producer_epoch=self.replay_source.config.producer_epoch,
                    reason="cs6b1h replay contributor startup",
                )
            except Exception:
                self.runtime.close()
                raise
            self._started = True
            self._composition_activation_scope = None
            self._legacy_unscoped_input_barrier_time = float(
                self.runtime.clock()
            )

    def close(self) -> None:
        with self._lock:
            if not self._started:
                return
            self.runtime.close()
            self._started = False
            self._composition_activation_scope = None
            self._legacy_unscoped_input_barrier_time = None

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError(
                "MujocoPoseComposerAuthorityComposition.start() is required."
            )

    def begin_composition_activation(
        self,
        *,
        scope: CompositionActivationScope,
    ) -> float:
        """Bind formal inputs to one explicit start/rearm activation."""

        if not isinstance(scope, CompositionActivationScope):
            raise TypeError("scope must be a CompositionActivationScope.")
        if scope.clock_domain_id != self.runtime.config.clock_domain_id:
            raise ValueError("Composition activation uses another control clock.")
        if scope.required_roles != self.composition_spec.required_roles:
            raise ValueError("Composition activation required roles mismatch.")
        with self._lock:
            self._require_started()
            self._input_receipt_by_role.clear()
            self._human_fact_time_by_emission_id.clear()
            self._step_by_decision_key.clear()
            self._composition_activation_scope = scope
            self.runtime.core.record_external_facts(
                (
                    project_composition_activation_scope(
                        self.runtime.core.trace_context,
                        scope,
                    ),
                )
            )
        return scope.input_barrier_at

    def _submit_input(
        self,
        *,
        role: str,
        producer_id: str,
        producer_epoch: int,
        sequence: int,
        action,
        produced_at: float,
        produced_clock_domain_id: str,
        effective_from: float,
        effective_until: float,
        observation_cut_id: str | None,
        source_fact_ids: tuple[str, ...],
        source_fact_hashes: tuple[str, ...],
        parent_fact_id: str,
        parent_fact_hash: str,
    ) -> CompositionInputAdmissionReceipt:
        activation_scope = self._composition_activation_scope
        if activation_scope is None:
            barrier_time = self._legacy_unscoped_input_barrier_time
            if barrier_time is None:
                raise RuntimeError(
                    "Composition activation input barrier is not armed."
                )
            if float(produced_at) < barrier_time:
                raise ValueError(
                    "Composition input predates the legacy activation barrier."
                )
        previous = self._input_receipt_by_role.get(role)
        if (
            previous is not None
            and previous.accepted
            and previous.composition_input is not None
            and previous.composition_input.audit_metadata.get("parent_fact_id")
            == parent_fact_id
            and previous.composition_input.audit_metadata.get("parent_fact_hash")
            == parent_fact_hash
        ):
            return previous
        receipt = self.runtime.submit_composition_input(
            CompositionInputDraft(
                runtime_session_id=self.runtime.config.runtime_session_id,
                control_domain_id=self.runtime.config.control_domain_id,
                producer_id=producer_id,
                producer_epoch=producer_epoch,
                sequence=sequence,
                input_role=role,
                action=project_robot_action_v1_to_component_action(
                    action,
                    action_schema=self.runtime.action_schema,
                ),
                produced_at=float(produced_at),
                produced_clock_domain_id=produced_clock_domain_id,
                effective_from=float(effective_from),
                effective_until=float(effective_until),
                effective_clock_domain_id=self.runtime.config.clock_domain_id,
                context_ids=(),
                observation_cut_id=observation_cut_id,
                policy_output_id=None,
                source_fact_ids=source_fact_ids,
                source_fact_hashes=source_fact_hashes,
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
                authority_candidate=False,
                audit_metadata={
                    "formal_composition_contributor": True,
                    "authority_gate_spec_hash": (
                        self.authority_gate_spec.semantic_content_hash
                    ),
                    "parent_fact_id": parent_fact_id,
                    "parent_fact_hash": parent_fact_hash,
                },
            ),
            activation_scope=activation_scope,
        )
        if receipt.accepted:
            self._input_receipt_by_role[role] = receipt
        return receipt

    def submit_human_emission(
        self,
        emission: HumanCommandEmission,
        *,
        observed_at: float,
    ) -> CompositionInputAdmissionReceipt:
        with self._lock:
            self._require_started()
            effective_from = float(observed_at)
            effective_until = (
                effective_from + self.runtime.config.proposal_horizon_sec
            )
            adaptation = self.human_adapter.adapt(
                emission,
                effective_from=effective_from,
                effective_until=effective_until,
                target_effective_at=effective_from,
            )
            adapter_receipt = adaptation.source_adapter_receipt
            effective_from = self._human_fact_time_by_emission_id.setdefault(
                adapter_receipt.source_emission_id,
                effective_from,
            )
            effective_until = (
                effective_from + self.runtime.config.proposal_horizon_sec
            )
            source_facts = project_human_proposal_adapter_result(
                emission,
                adaptation,
                runtime_session_id=self.runtime.config.runtime_session_id,
                control_domain_id=self.runtime.config.control_domain_id,
                embodiment_instance_id=(
                    self.runtime.config.embodiment_instance_id
                ),
                occurred_at=effective_from,
                clock_domain_id=self.runtime.config.clock_domain_id,
            )
            self.runtime.core.record_external_facts(source_facts)
            source_fact_hash_by_id = {
                fact.fact_id: fact.semantic_content_hash
                for fact in source_facts
            }
            if (
                adaptation.robot_action is None
                or not adaptation.source_validation.accepted
            ):
                raise ValueError(
                    "Human contributor adaptation failed: "
                    + ",".join(adaptation.source_validation.reasons)
                )
            return self._submit_input(
                role=CS6B1H_HUMAN_POSE_ROLE_V1,
                producer_id=self.human_adapter.config.producer_id,
                producer_epoch=self.human_adapter.config.producer_epoch,
                sequence=emission.emission_sequence,
                action=adaptation.robot_action,
                produced_at=effective_from,
                produced_clock_domain_id=(
                    self.human_adapter.config.produced_clock_domain_id
                ),
                effective_from=effective_from,
                effective_until=effective_until,
                observation_cut_id=None,
                source_fact_ids=(
                    adapter_receipt.source_emission_id,
                    adapter_receipt.representation_trace_id,
                ),
                source_fact_hashes=(
                    source_fact_hash_by_id[adapter_receipt.source_emission_id],
                    source_fact_hash_by_id[
                        adapter_receipt.representation_trace_id
                    ],
                ),
                parent_fact_id=adapter_receipt.source_emission_id,
                parent_fact_hash=adapter_receipt.source_emission_semantic_hash,
            )

    def submit_replay_emission(
        self,
        emission: ReplaySourceFrameEmission,
    ) -> CompositionInputAdmissionReceipt:
        with self._lock:
            self._require_started()
            if emission.clock_domain_id != self.runtime.config.clock_domain_id:
                raise ValueError(
                    "Replay contributor and runtime require one clock domain."
                )
            adaptation = self.replay_source.adapt_source_frame_emission(
                emission,
                sequence=emission.source_action_index,
            )
            source_facts = (
                project_replay_source_frame_emission(
                    emission,
                    runtime_session_id=self.runtime.config.runtime_session_id,
                    control_domain_id=self.runtime.config.control_domain_id,
                    embodiment_instance_id=(
                        self.runtime.config.embodiment_instance_id
                    ),
                ),
                *project_replay_adapter_result(
                    adaptation,
                    runtime_session_id=self.runtime.config.runtime_session_id,
                    control_domain_id=self.runtime.config.control_domain_id,
                    embodiment_instance_id=(
                        self.runtime.config.embodiment_instance_id
                    ),
                    occurred_at=emission.observed_at,
                    clock_domain_id=self.runtime.config.clock_domain_id,
                ),
            )
            existing_by_id = {
                fact.fact_id: fact
                for fact in self.runtime.trace_facts
            }
            new_source_facts = []
            for fact in source_facts:
                existing = existing_by_id.get(fact.fact_id)
                if existing is None:
                    new_source_facts.append(fact)
                elif (
                    existing.event_type != fact.event_type
                    or existing.payload.get("semantic_content_hash")
                    != fact.payload.get("semantic_content_hash")
                ):
                    raise ValueError(
                        "Replay source fact ID was reused with new payload."
                    )
            self.runtime.core.record_external_facts(tuple(new_source_facts))
            source_fact_hash_by_id = {
                fact.fact_id: fact.semantic_content_hash
                for fact in source_facts
            }
            draft = adaptation.proposal_draft
            receipt = adaptation.source_adapter_receipt
            if draft is None or not receipt.accepted:
                raise ValueError(
                    "Replay contributor adaptation failed: "
                    + ",".join(receipt.rejection_reasons)
                )
            return self._submit_input(
                role=CS6B1H_REPLAY_POSE_ROLE_V1,
                producer_id=self.replay_source.config.producer_id,
                producer_epoch=self.replay_source.config.producer_epoch,
                sequence=emission.source_action_index,
                # Formal composition v1 still consumes RobotAction pose
                # values.  The replay proposal itself is ComponentAction-
                # native; retain the immutable source-frame action at this
                # explicitly legacy composition-input edge.
                action=emission.action,
                produced_at=draft.produced_at,
                produced_clock_domain_id=draft.produced_clock_domain_id,
                effective_from=draft.effective_from,
                effective_until=draft.effective_until,
                observation_cut_id=draft.observation_cut_id,
                source_fact_ids=(
                    emission.emission_id,
                    receipt.representation_trace_id,
                ),
                source_fact_hashes=(
                    source_fact_hash_by_id[emission.emission_id],
                    source_fact_hash_by_id[receipt.representation_trace_id],
                ),
                parent_fact_id=emission.emission_id,
                parent_fact_hash=emission.semantic_content_hash,
            )

    def compose_at(
        self,
        *,
        decision_key: str,
        target_effective_at: float,
    ) -> CS6b1hFormalCompositionStep:
        """Evaluate one exact cut, then publish once or transition to HOLD."""

        key = str(decision_key).strip()
        target = float(target_effective_at)
        if not key:
            raise ValueError("decision_key must be non-empty.")
        if not math.isfinite(target):
            raise ValueError("target_effective_at must be finite.")
        with self._lock:
            self._require_started()
            cached = self._step_by_decision_key.get(key)
            if cached is not None:
                if not math.isclose(
                    cached.target_effective_at,
                    target,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "decision_key was reused with another target time."
                    )
                return cached
            evaluation: CompositeProposalSubmission | None = None
            disposition: CompositionAuthorityDisposition | None = None
            proposal: CompositeProposalSubmission | None = None
            hold: ExecutionGateTransitionOutcome | None = None
            blocked_reason: str | None = None
            error_text: str | None = None
            gate_snapshot = self.runtime.core.capture_execution_gate_snapshot(
                now=target
            )
            if gate_snapshot.status != "ready":
                blocked_reason = (
                    f"execution_gate_not_ready:{gate_snapshot.status}"
                )
            else:
                try:
                    evaluation = self.runtime.evaluate_composition_shadow(
                        spec=self.composition_spec,
                        target_effective_at=target,
                        source_difference_spec=self.source_difference_spec,
                        activation_scope=self._composition_activation_scope,
                    )
                    disposition = decide_composition_authority_disposition(
                        evaluation.input_snapshot,
                        evaluation.composition_evaluation,
                        spec=self.composition_spec,
                        gate_spec=self.authority_gate_spec,
                    )
                    self.runtime.core.record_external_facts(
                        (
                            project_composition_authority_disposition(
                                self.runtime.core.trace_context,
                                disposition,
                                occurred_at=float(self.runtime.clock()),
                            ),
                        )
                    )
                    if disposition.downstream_request == "submit_candidate":
                        proposal = self.runtime.submit_evaluated_composition(
                            evaluation,
                            policy_node_id="cs6b1h-formal-pose-composer",
                        )
                        receipt = proposal.proposal_admission
                        if receipt is None or not receipt.accepted:
                            raise RuntimeError(
                                "Formal composite proposal admission was rejected."
                            )
                    else:
                        hold = self.runtime.transition_execution_gate(
                            "hold",
                            reason=f"cs6b1h:{disposition.reason}",
                            source_event_ids=(
                                disposition.disposition_id,
                            ),
                            force_cancellation=True,
                        )
                except Exception as error:
                    error_text = f"{type(error).__name__}:{error}"
                    if hold is None:
                        hold = self.runtime.transition_execution_gate(
                            "hold",
                            reason=f"cs6b1h:composer_error:{key}",
                            force_cancellation=True,
                        )
            payload = {
                "decision_key": key,
                "target_effective_at": target,
                "composition_evaluation_id": (
                    None
                    if evaluation is None
                    else evaluation.composition_evaluation.composition_evaluation_id
                ),
                "composition_evaluation_hash": (
                    None
                    if evaluation is None
                    else evaluation.composition_evaluation.semantic_content_hash
                ),
                "disposition_id": (
                    None if disposition is None else disposition.disposition_id
                ),
                "disposition_hash": (
                    None
                    if disposition is None
                    else disposition.semantic_content_hash
                ),
                "proposal_id": (
                    None
                    if proposal is None
                    or proposal.proposal_admission is None
                    or proposal.proposal_admission.proposal is None
                    else proposal.proposal_admission.proposal.proposal_id
                ),
                "hold_event_id": (
                    None if hold is None else hold.event.event_id
                ),
                "blocked_reason": blocked_reason,
                "error": error_text,
                "composition_activation_id": (
                    None
                    if self._composition_activation_scope is None
                    else self._composition_activation_scope.activation_id
                ),
                "composition_activation_hash": (
                    None
                    if self._composition_activation_scope is None
                    else self._composition_activation_scope.semantic_content_hash
                ),
            }
            if self.source_difference_spec is not None:
                payload.update(
                    {
                        "source_difference_observation_id": (
                            None
                            if evaluation is None
                            or evaluation.source_difference_observation is None
                            else evaluation.source_difference_observation.observation_id
                        ),
                        "source_difference_observation_hash": (
                            None
                            if evaluation is None
                            or evaluation.source_difference_observation is None
                            else evaluation.source_difference_observation.semantic_content_hash
                        ),
                        "source_difference_error": (
                            None
                            if evaluation is None
                            else evaluation.source_difference_error
                        ),
                    }
                )
            step = CS6b1hFormalCompositionStep(
                decision_key=key,
                target_effective_at=target,
                evaluation=evaluation,
                disposition=disposition,
                proposal_submission=proposal,
                hold_transition=hold,
                blocked_reason=blocked_reason,
                error=error_text,
                semantic_content_hash=canonical_content_hash(
                    payload,
                    domain=CS6B1H_STEP_HASH_DOMAIN_V1,
                ),
            )
            self._step_by_decision_key[key] = step
            return step


def _cs6b1i_producer_epoch_policy_hash(
    producer_id: str,
    producer_epoch: int,
    *,
    rearmable: bool = False,
) -> str:
    return canonical_content_hash(
        {
            "producer_id": producer_id,
            "producer_epoch": producer_epoch,
            "activation": (
                "advance_on_typed_composition_activation_rearm"
                if rearmable
                else "fixed_on_composition_session_runtime_start"
            ),
        },
        domain="dexteleop.cs6b1i_fixed_producer_epoch.v1",
    )


def build_cs6b1i_pose_composition_session_manifest(
    composition: MujocoPoseComposerAuthorityComposition,
    human_source_spec: HumanPoseSourceContract,
    replay_observer_binding: ReplaySourceObserverBinding,
) -> ControlStationRunManifest:
    """Freeze a composer-only Station profile without owning an episode."""

    if not isinstance(composition, MujocoPoseComposerAuthorityComposition):
        raise TypeError(
            "composition must be a MujocoPoseComposerAuthorityComposition."
        )
    if not isinstance(human_source_spec, HumanPoseSourceContract):
        raise TypeError("human_source_spec must be a HumanPoseSourceContract.")
    if not isinstance(replay_observer_binding, ReplaySourceObserverBinding):
        raise TypeError(
            "replay_observer_binding must be a ReplaySourceObserverBinding."
        )
    runtime = composition.runtime
    replay = composition.replay_source.config
    human = composition.human_adapter.config
    composer_slot_id = f"composer-slot:{runtime.config.composer_producer_id}"
    binding = ExecutionDomainBinding(
        binding_id=f"mujoco-target:{runtime.config.control_domain_id}",
        target_kind="sim",
        control_domain_id=runtime.config.control_domain_id,
        embodiment_instance_id=runtime.config.embodiment_instance_id,
        target_runtime_action_contract_hash=(
            runtime.target_runtime_action_contract.semantic_content_hash
        ),
        execution_capability_profile_hash=(
            runtime.execution_capability.semantic_content_hash
        ),
        component_catalog_hash=runtime.catalog.semantic_content_hash,
        selection_group_ids=runtime.core.config.active_selection_groups,
        owner_registry_id=authoritative_owner_registry_id(runtime),
        authority_epoch_source_id=authoritative_authority_epoch_source_id(runtime),
        execution_epoch_source_id=authoritative_execution_epoch_source_id(runtime),
        trace_requirement_profile_id=(
            CS6B1I_COMPOSITION_SESSION_TRACE_PROFILE_ID_V1
        ),
        explicit_exclusions=(
            "hardware_device_initialization",
            "real_backend",
        ),
    )
    contributor_trace_refs = (
        "composition_input_admission",
        "composition_contributor_input",
        "composition_input_snapshot",
        "composition_evaluation",
        "composition_authority_disposition",
    )
    return ControlStationRunManifest(
        manifest_id=CS6B1I_COMPOSITION_SESSION_MANIFEST_ID_V1,
        manifest_version="1",
        execution_binding=binding,
        source_slots=(
            SourceSlotSpec(
                source_slot_id=human_source_spec.source_id,
                source_kind="human",
                producer_id=human.producer_id,
                source_action_contract_hash=composition.human_adapter.contract_hash,
                producer_epoch_policy_hash=_cs6b1i_producer_epoch_policy_hash(
                    human.producer_id,
                    human.producer_epoch,
                ),
                supported_uses=("contributor",),
                expected_emission_seam=(
                    "teleop_runtime.action.human_proposal.HumanCommandEmission"
                ),
                required_binding_refs=(
                    human_source_spec.semantic_content_hash,
                    composition.binding.semantic_content_hash,
                ),
                trace_requirement_refs=(
                    "human_command_emission",
                    "source_adapter_receipt",
                    "source_validation",
                    *contributor_trace_refs,
                ),
                explicit_exclusions=(
                    "physical_teleoperation_device",
                    "parallel_motion_proposal",
                    "source_owned_action_queue",
                ),
            ),
            SourceSlotSpec(
                source_slot_id=replay_observer_binding.source_slot_id,
                source_kind="replay",
                producer_id=replay.producer_id,
                source_action_contract_hash=(
                    replay.source_action_contract.semantic_content_hash
                ),
                producer_epoch_policy_hash=_cs6b1i_producer_epoch_policy_hash(
                    replay.producer_id,
                    replay.producer_epoch,
                ),
                supported_uses=("contributor",),
                expected_emission_seam=(
                    "robot_motion.replay.proposal_source.ReplaySourceFrameEmission"
                ),
                required_binding_refs=(
                    replay.source_contract.semantic_content_hash,
                    replay.binding_assessment.semantic_content_hash,
                    replay_observer_binding.semantic_content_hash,
                    composition.binding.semantic_content_hash,
                ),
                trace_requirement_refs=(
                    "replay_source_frame_emission",
                    "replay_source_adapter_receipt",
                    *contributor_trace_refs,
                ),
                explicit_exclusions=(
                    "control_station_dataset_reader",
                    "control_station_replay_cursor",
                    "parallel_motion_proposal",
                ),
            ),
            SourceSlotSpec(
                source_slot_id=composer_slot_id,
                source_kind="composite",
                producer_id=runtime.config.composer_producer_id,
                source_action_contract_hash=canonical_content_hash(
                    {
                        "action_space": runtime.catalog.action_space,
                        "component_catalog_hash": (
                            runtime.catalog.semantic_content_hash
                        ),
                        "composition_spec_hash": (
                            composition.composition_spec.semantic_content_hash
                        ),
                        "authority_gate_spec_hash": (
                            composition.authority_gate_spec.semantic_content_hash
                        ),
                    },
                    domain="dexteleop.cs6b1i_composer_action_contract.v1",
                ),
                producer_epoch_policy_hash=_cs6b1i_producer_epoch_policy_hash(
                    runtime.config.composer_producer_id,
                    runtime.config.producer_epoch,
                    rearmable=True,
                ),
                supported_uses=("candidate", "authority"),
                expected_emission_seam=(
                    "control_composition.composition.CompositionEvaluation"
                ),
                required_binding_refs=(
                    composition.composition_spec.semantic_content_hash,
                    composition.authority_gate_spec.semantic_content_hash,
                    composition.binding.semantic_content_hash,
                ),
                trace_requirement_refs=(
                    *contributor_trace_refs,
                    "action_proposal",
                    "proposal_admission",
                ),
                explicit_exclusions=("direct_contributor_proposal_fallback",),
            ),
        ),
        authority_policy_hash=runtime.core.config.authority_policy.semantic_content_hash,
        decision_schedule_hash=(
            runtime.core.config.decision_schedule.semantic_content_hash
        ),
        trace_requirement_profile_id=(
            CS6B1I_COMPOSITION_SESSION_TRACE_PROFILE_ID_V1
        ),
        capture_policy_id="external_episode_owner_hold_before_unbind_v1",
        allowed_intent_kinds=(
            "acquire_control_lease",
            "connect_source",
            "connect_target",
            "disconnect_source",
            "disconnect_target",
            "estop",
            "hold",
            "release_candidate",
            "release_control_lease",
            "request_candidate",
            "rearm_composition_session",
            "run_preflight",
            "start_composition_session",
            "stop_composition_session",
            "stop_runtime",
        ),
        explicit_exclusions=(
            "browser_raw_action_submission",
            "gui_operation",
            "physical_human_device",
            "policy_model",
            "production_entrypoint",
            "real_motion",
            "replay_cursor_ownership",
            "rtc_policy_source",
        ),
    )


def build_cs6b1i_pose_composition_session_execution_manifest(
    composition: MujocoPoseComposerAuthorityComposition,
    station_manifest: ControlStationRunManifest,
) -> ControlExecutionManifest:
    runtime = composition.runtime
    if (
        station_manifest.manifest_id
        != CS6B1I_COMPOSITION_SESSION_MANIFEST_ID_V1
    ):
        raise ValueError("CS6b1i requires the exact composition-session manifest.")
    if runtime.catalog.action_space != BIMANUAL_EE_POSE_V1:
        raise ValueError("CS6b1i requires pose-native proposal execution.")
    exclusions = tuple(
        ExecutionProfileExclusion.create(
            capability_id=capability_id,
            reason="pose_composition_session_deterministic_sim_only",
            evidence_ids=(station_manifest.semantic_content_hash,),
        )
        for capability_id in (
            "hardware_device_initialization",
            "physical_human_device",
            "policy_model",
            "production_station_entrypoint",
            "real_backend",
            "rtc_policy_source",
        )
    )
    return build_control_execution_manifest(
        manifest_id=CS6B1I_COMPOSITION_SESSION_EXECUTION_MANIFEST_ID_V1,
        manifest_version="v1",
        action_space=runtime.catalog.action_space,
        action_schema=runtime.action_schema,
        component_target_capability=runtime.component_target_capability,
        catalog=runtime.catalog,
        motion_authority_sources=(
            f"composer:{runtime.config.composer_producer_id}",
        ),
        human_motion_proposal_status="not_applicable",
        safety_control_sources=("runtime_safety_supervisor",),
        safety_control_requests=("hold",),
        lifecycle_control_sources=("composition_session", "external_episode_owner"),
        lifecycle_control_requests=(
            "start",
            "hold",
            "rearm_new_activation",
            "stop",
            "release",
        ),
        capability=runtime.execution_capability,
        activation=runtime.reference_activation_spec,
        actual_reference_frame_producer="mujoco_pose_to_qpos_resolver",
        exclusions=exclusions,
    )


class _CS6b1iTargetAdapter:
    def __init__(
        self,
        composition: MujocoPoseComposerAuthorityComposition,
        *,
        subject_id: str,
        clock: Callable[[], float],
        clock_domain_id: str,
    ) -> None:
        self.adapter_id = "cs6b1i-mujoco-target-adapter"
        self.subject_id = subject_id
        self.supported_operations = (
            "connect_target",
            "disconnect_target",
            "run_preflight",
        )
        self.composition = composition
        self.clock = clock
        self.clock_domain_id = clock_domain_id
        self._result_by_intent_id: dict[str, StationOperationResultDraft] = {}

    def perform(self, intent: StationIntent) -> StationOperationResultDraft:
        cached = self._result_by_intent_id.get(intent.intent_id)
        if cached is not None:
            return cached
        if intent.subject_id != self.subject_id:
            raise ValueError("Target operation belongs to another adapter.")
        if intent.intent_kind == "connect_target":
            self.composition.start()
            status = "composition_target_started_without_session_authority"
        elif intent.intent_kind == "run_preflight":
            if not self.composition.started:
                raise RuntimeError("Composition target is not started.")
            status = "composition_target_binding_preflighted"
        elif intent.intent_kind == "disconnect_target":
            self.composition.close()
            status = "composition_target_closed"
        else:
            raise ValueError("Unsupported CS6b1i target operation.")
        result = StationOperationResultDraft(
            source_intent_id=intent.intent_id,
            operation_kind=intent.intent_kind,
            subject_id=self.subject_id,
            succeeded=True,
            completed_at=float(self.clock()),
            completed_clock_domain_id=self.clock_domain_id,
            status=status,
            evidence_refs=(composition_evidence_id(self.composition),),
        )
        self._result_by_intent_id[intent.intent_id] = result
        return result


class _CS6b1iSourceAdapter:
    def __init__(
        self,
        owner: "ScopedMujocoPoseCompositionSession",
        *,
        subject_id: str,
        role: str,
        clock: Callable[[], float],
        clock_domain_id: str,
    ) -> None:
        self.adapter_id = f"cs6b1i-{role}-adapter"
        self.subject_id = subject_id
        self.role = role
        self.clock = clock
        self.clock_domain_id = clock_domain_id
        self.owner = owner
        base = ("connect_source", "disconnect_source", "run_preflight")
        self.supported_operations = (
            (*base, "request_candidate", "release_candidate")
            if role == "composer"
            else base
        )
        if role == "composer":
            self.supported_operations = (
                *self.supported_operations,
                "start_composition_session",
                "rearm_composition_session",
                "stop_composition_session",
            )
        self._result_by_intent_id: dict[str, StationOperationResultDraft] = {}

    def perform(self, intent: StationIntent) -> StationOperationResultDraft:
        cached = self._result_by_intent_id.get(intent.intent_id)
        if cached is not None:
            return cached
        if intent.subject_id != self.subject_id:
            raise ValueError("Source operation belongs to another adapter.")
        kind = intent.intent_kind
        succeeded = True
        evidence = [composition_evidence_id(self.owner.data_plane)]
        if kind == "connect_source":
            status = f"{self.role}_source_connected_without_device_start"
        elif kind == "run_preflight":
            if not self.owner.data_plane.started:
                raise RuntimeError("Composition target is not started.")
            status = f"{self.role}_source_binding_preflighted"
        elif kind == "disconnect_source":
            status = f"{self.role}_source_disconnected"
        elif kind == "request_candidate" and self.role == "composer":
            status = "composer_candidate_bound_without_publication"
        elif kind == "release_candidate" and self.role == "composer":
            status = "composer_candidate_released"
        elif kind == "start_composition_session" and self.role == "composer":
            try:
                spec = self.owner.require_registered_spec(
                    intent.operation_spec_id
                )
                self.owner.validate_start_readiness(spec)
            except Exception as error:
                succeeded = False
                status = COMPOSITION_SESSION_OPERATION_HELD_STATUS_V1
                evidence.append(
                    f"composition-session-start-rejected:{type(error).__name__}"
                )
            else:
                status = COMPOSITION_SESSION_OPERATION_READY_STATUS_V1
                evidence.extend(
                    (spec.operation_spec_id, spec.semantic_content_hash)
                )
        elif kind == "rearm_composition_session" and self.role == "composer":
            try:
                rearm_spec = self.owner.session_controller.require_rearm_spec(
                    intent.operation_spec_id
                )
                receipt = self.owner.acquire_composition_rearm_receipt(
                    intent.operation_spec_id
                )
            except Exception as error:
                self.owner._force_rearm_failure_hold(
                    reason=(
                        "cs6b1i_rearm_readiness_failed:"
                        f"{type(error).__name__}"
                    )
                )
                succeeded = False
                status = COMPOSITION_SESSION_OPERATION_HELD_STATUS_V1
                evidence.append(
                    f"composition-session-rearm-rejected:{type(error).__name__}"
                )
            else:
                status = COMPOSITION_SESSION_OPERATION_REARMED_STATUS_V1
                evidence.extend(
                    (
                        rearm_spec.rearm_spec_id,
                        rearm_spec.semantic_content_hash,
                        receipt.receipt_id,
                        receipt.semantic_content_hash,
                    )
                )
        elif kind == "stop_composition_session" and self.role == "composer":
            try:
                self.owner.require_current_spec(intent.operation_spec_id)
            except Exception as error:
                succeeded = False
                status = f"composition_session_stop_failed:{type(error).__name__}"
                evidence.append(
                    f"composition-session-stop-rejected:{type(error).__name__}"
                )
            else:
                status = COMPOSITION_SESSION_OPERATION_STOPPED_STATUS_V1
        else:
            raise ValueError("Unsupported CS6b1i source operation.")
        result = StationOperationResultDraft(
            source_intent_id=intent.intent_id,
            operation_kind=kind,
            subject_id=self.subject_id,
            succeeded=succeeded,
            completed_at=float(self.clock()),
            completed_clock_domain_id=self.clock_domain_id,
            status=status,
            evidence_refs=tuple(evidence),
        )
        self._result_by_intent_id[intent.intent_id] = result
        return result


def composition_evidence_id(
    composition: MujocoPoseComposerAuthorityComposition,
) -> str:
    return f"cs6b1h-binding:{composition.binding.semantic_content_hash}"


class ScopedMujocoPoseCompositionSession:
    """Station lifecycle around the existing CS6b1h composer data plane."""

    def __init__(
        self,
        *,
        station: ControlStationRuntime,
        data_plane: MujocoPoseComposerAuthorityComposition,
        human_source_slot_id: str,
        replay_source_slot_id: str,
        composer_source_slot_id: str,
        clock: Callable[[], float],
        clock_domain_id: str,
    ) -> None:
        self._station = station
        self._data_plane = data_plane
        self.human_source_slot_id = human_source_slot_id
        self.replay_source_slot_id = replay_source_slot_id
        self.composer_source_slot_id = composer_source_slot_id
        self.clock = clock
        self.clock_domain_id = clock_domain_id
        self.rearm_readiness_observer_binding_id = (
            "cs6b1i-mujoco-composition-rearm-readiness-v1"
        )
        self._last_hold_outcome: ExecutionGateTransitionOutcome | None = None
        self._rearm_receipt_by_spec_id: dict[
            str, CompositionSessionRearmReceipt
        ] = {}
        self.session_controller = CompositionSessionController(
            composer_source_slot_id=composer_source_slot_id,
            contributor_source_slot_ids=(
                human_source_slot_id,
                replay_source_slot_id,
            ),
            composition_spec_hash=(
                data_plane.composition_spec.semantic_content_hash
            ),
            authority_gate_spec_hash=(
                data_plane.authority_gate_spec.semantic_content_hash
            ),
            execution_binding_hash=(
                station.manifest.execution_binding.semantic_content_hash
            ),
            clock=clock,
            clock_domain_id=clock_domain_id,
        )
        self.target_adapter: StationOperationAdapter = _CS6b1iTargetAdapter(
            data_plane,
            subject_id=station.manifest.execution_binding.binding_id,
            clock=clock,
            clock_domain_id=clock_domain_id,
        )
        self.human_adapter: StationOperationAdapter = _CS6b1iSourceAdapter(
            self,
            subject_id=human_source_slot_id,
            role="human_contributor",
            clock=clock,
            clock_domain_id=clock_domain_id,
        )
        self.replay_adapter: StationOperationAdapter = _CS6b1iSourceAdapter(
            self,
            subject_id=replay_source_slot_id,
            role="replay_contributor",
            clock=clock,
            clock_domain_id=clock_domain_id,
        )
        self.composer_adapter: StationOperationAdapter = _CS6b1iSourceAdapter(
            self,
            subject_id=composer_source_slot_id,
            role="composer",
            clock=clock,
            clock_domain_id=clock_domain_id,
        )

    @property
    def station(self) -> ControlStationRuntime:
        return self._station

    @property
    def data_plane(self) -> MujocoPoseComposerAuthorityComposition:
        return self._data_plane

    @property
    def publication_barrier(self):
        """Compatibility view; the generic session controller is the owner."""

        return self.session_controller.publication_barrier

    @property
    def adapters(self) -> tuple[StationOperationAdapter, ...]:
        return (
            self.target_adapter,
            self.human_adapter,
            self.replay_adapter,
            self.composer_adapter,
        )

    @property
    def current_session_state(self) -> CompositionSessionState | None:
        return self.session_controller.current_state

    def bind_session_trace_observers(
        self,
        *,
        spec_observer: Callable[[CompositionSessionOperationSpec], object],
        transition_observer: Callable[..., object],
        rearm_spec_observer: Callable[[CompositionSessionRearmSpec], object]
        | None = None,
        rearm_receipt_observer: Callable[
            [CompositionSessionRearmReceipt], object
        ]
        | None = None,
    ) -> None:
        self.session_controller.bind_session_trace_observers(
            spec_observer=spec_observer,
            transition_observer=transition_observer,
            rearm_spec_observer=rearm_spec_observer,
            rearm_receipt_observer=rearm_receipt_observer,
        )

    def bind_publication_barrier_observer(
        self,
        observer: Callable[..., object],
    ) -> None:
        self.session_controller.bind_publication_barrier_observer(observer)

    def register_composition_session(
        self,
        spec: CompositionSessionOperationSpec,
    ) -> str:
        return self.session_controller.register(spec)

    def register_composition_rearm(
        self,
        spec: CompositionSessionRearmSpec,
    ) -> str:
        if (
            spec.readiness_observer_binding_id
            != self.rearm_readiness_observer_binding_id
        ):
            raise ValueError("Composition rearm readiness provider mismatch.")
        if (
            spec.required_readiness_checks
            != CS6B1I_SIM_REARM_READINESS_CHECKS_V1
        ):
            raise ValueError(
                "MuJoCo rearm cannot attest unsupported readiness checks."
            )
        return self.session_controller.register_rearm(spec)

    def require_registered_spec(
        self,
        operation_spec_id: str | None,
    ) -> CompositionSessionOperationSpec:
        return self.session_controller.require_registered_spec(
            operation_spec_id
        )

    def require_current_spec(
        self,
        operation_spec_id: str | None,
    ) -> CompositionSessionOperationSpec:
        return self.session_controller.require_current_spec(operation_spec_id)

    def validate_start_readiness(
        self,
        spec: CompositionSessionOperationSpec,
    ) -> None:
        state = self.station.state
        if any(
            state.source_state_by_slot[slot_id] != "preflighted"
            for slot_id in spec.contributor_source_slot_ids
        ):
            raise RuntimeError(
                "Composition contributors must be preflighted."
            )
        gate = self.data_plane.runtime.core.capture_execution_gate_snapshot(
            now=float(self.clock())
        )
        if gate.status != "ready":
            raise RuntimeError(f"Execution gate is not ready: {gate.status}.")

    def acquire_composition_rearm_receipt(
        self,
        rearm_spec_id: str | None,
    ) -> CompositionSessionRearmReceipt:
        """Build exact sim-only readiness evidence for one fresh activation."""

        spec = self.session_controller.require_rearm_spec(rearm_spec_id)
        cached = self._rearm_receipt_by_spec_id.get(spec.rearm_spec_id)
        if cached is not None:
            return cached
        state = self.current_session_state
        if state is None or state.phase != "held":
            raise RuntimeError("Composition rearm requires a held activation.")
        hold = self._last_hold_outcome
        if (
            hold is None
            or hold.cancellation is None
            or hold.cancellation_receipt is None
            or not hold.cancellation_receipt.complete
            or hold.cancellation_receipt.requires_fault_escalation
        ):
            raise RuntimeError(
                "Composition rearm requires one complete lower cancellation receipt."
            )
        cancellation = hold.cancellation
        runtime = self.data_plane.runtime
        execution_before = runtime.core.execution_epochs.snapshot()
        if (
            execution_before.fault_latched
            or execution_before.execution_epoch != cancellation.execution_epoch
            or cancellation.execution_epoch
            != cancellation.previous_execution_epoch + 1
        ):
            raise RuntimeError(
                "Composition rearm execution epoch/cancellation evidence is stale."
            )
        measurement = runtime.capture_qpos_measurement()
        if (
            measurement.clock_domain_id != runtime.config.clock_domain_id
            or measurement.measured_at < cancellation.effective_at
            or float(self.clock()) - measurement.measured_at
            > spec.readiness_timeout_sec
        ):
            raise RuntimeError("Composition rearm measured state is stale.")

        composer_id = runtime.config.composer_producer_id
        prior_producer_epoch = runtime.core.producer_epochs.current_epoch(
            composer_id
        )
        if prior_producer_epoch is None:
            raise RuntimeError("Composer has no active producer epoch.")
        producer_activation = runtime.activate_policy_node_producer(
            composer_id,
            producer_epoch=prior_producer_epoch + 1,
            reason=f"cs6b1i_rearm_activation:{spec.rearm_spec_id}",
        )
        receiver_before = runtime.core.capture_receiver_state()
        authority_event = runtime.core.admit_authority_event(
            event_type="mode",
            control_mode=runtime.config.initial_control_mode,
            reason=f"cs6b1i_rearm_authority_epoch:{spec.rearm_spec_id}",
        )
        if authority_event.authority_epoch <= (
            receiver_before.current_authority_epoch
        ):
            raise RuntimeError("Composition rearm authority epoch did not advance.")
        activation_scope = build_composition_activation_scope(
            operation_spec_id=spec.logical_operation_spec_id,
            operation_spec_hash=spec.logical_operation_spec_hash,
            session_epoch=state.session_epoch + 1,
            activated_at=measurement.measured_at,
            clock_domain_id=runtime.config.clock_domain_id,
            input_barrier_at=measurement.measured_at,
            required_roles=self.data_plane.composition_spec.required_roles,
            freshness_policy=(
                ACTIVATION_LOCAL_STRICT_SOURCE_FRESHNESS_POLICY_V2
            ),
        )
        activation_time = self.data_plane.begin_composition_activation(
            scope=activation_scope,
        )
        ready = runtime.transition_execution_gate(
            "ready",
            reason=f"cs6b1i_rearm_ready:{spec.rearm_spec_id}",
            source_event_ids=(
                cancellation.cancellation_id,
                producer_activation.activation_id,
                authority_event.event_id,
            ),
        )
        if ready.snapshot.status != "ready" or ready.cancellation is not None:
            raise RuntimeError("Composition rearm failed to establish READY.")
        receipt = CompositionSessionRearmReceipt(
            rearm_spec_id=spec.rearm_spec_id,
            rearm_spec_hash=spec.semantic_content_hash,
            logical_operation_spec_id=spec.logical_operation_spec_id,
            logical_operation_spec_hash=spec.logical_operation_spec_hash,
            prior_session_epoch=state.session_epoch,
            next_session_epoch=state.session_epoch + 1,
            prior_composer_producer_epoch=prior_producer_epoch,
            composer_producer_epoch=producer_activation.producer_epoch,
            prior_authority_epoch=receiver_before.current_authority_epoch,
            authority_epoch=authority_event.authority_epoch,
            prior_execution_epoch=cancellation.previous_execution_epoch,
            execution_epoch=cancellation.execution_epoch,
            cancellation_receipt_id=cancellation.cancellation_id,
            measurement_id=measurement.measurement_id,
            measurement_hash=measurement.semantic_content_hash,
            producer_activation_id=producer_activation.activation_id,
            authority_event_ids=(authority_event.event_id,),
            ready_gate_event_id=ready.event.event_id,
            activation_input_barrier_time=activation_time,
            completed_at=float(self.clock()),
            clock_domain_id=runtime.config.clock_domain_id,
            satisfied_readiness_checks=CS6B1I_SIM_REARM_READINESS_CHECKS_V1,
        )
        self._rearm_receipt_by_spec_id[spec.rearm_spec_id] = receipt
        return receipt

    def _force_rearm_failure_hold(self, *, reason: str) -> None:
        outcome = self.data_plane.runtime.transition_execution_gate(
            "hold",
            reason=reason,
            force_cancellation=True,
        )
        self._last_hold_outcome = outcome
        self.session_controller.disable_publication(reason=reason)

    def prepare_pending_operation(
        self,
        transition: ControlStationTransition,
    ) -> None:
        if not transition.receipt.accepted:
            return
        kind = transition.intent.intent_kind
        if kind not in {
            "stop_composition_session",
            "hold",
            "estop",
            "disconnect_target",
        }:
            return
        projected = project_station_intent_transition(
            transition,
            embodiment_instance_id=(
                self.data_plane.runtime.config.embodiment_instance_id
            ),
        )
        def lower_hold_or_fault() -> ExecutionGateTransitionOutcome:
            outcome = self.data_plane.runtime.transition_execution_gate(
                "hold" if kind != "estop" else "fault",
                reason=f"cs6b1i_station_{kind}",
                source_event_ids=(transition.intent.intent_id,),
                force_cancellation=True,
            )
            self._last_hold_outcome = outcome
            return outcome

        self.session_controller.prepare_safety_transition(
            intent_kind=kind,
            causal_station_fact_id=transition.intent.intent_id,
            source_fact_id=projected.fact_id,
            source_fact_hash=projected.semantic_content_hash,
            occurred_at=transition.intent.available_at,
            clock_domain_id=transition.intent.available_clock_domain_id,
            lower_hold_or_fault=lower_hold_or_fault,
        )

    def complete_pending_operation(
        self,
        transition: ControlStationTransition,
    ) -> StationOperationExecution:
        self.prepare_pending_operation(transition)
        subject = transition.intent.subject_id
        adapter_by_subject = {
            adapter.subject_id: adapter for adapter in self.adapters
        }
        try:
            adapter = adapter_by_subject[str(subject)]
        except KeyError as exc:
            raise ValueError("Pending operation has no CS6b1i adapter.") from exc
        execution = self.station.execute_pending_operation(transition, adapter)
        completion = execution.completion_transition
        kind = transition.intent.intent_kind
        if kind not in {
            "start_composition_session",
            "rearm_composition_session",
            "stop_composition_session",
        }:
            return execution
        projected = project_station_operation_transition(
            completion,
            embodiment_instance_id=(
                self.data_plane.runtime.config.embodiment_instance_id
            ),
        )
        if kind == "start_composition_session":
            if completion.accepted and completion.result.succeeded:
                spec = self.session_controller.require_registered_spec(
                    transition.intent.operation_spec_id
                )
                state = self.current_session_state
                if state is None or state.phase != "registered":
                    raise RuntimeError(
                        "Composition start lacks a registered activation."
                    )
                activation_scope = build_composition_activation_scope(
                    operation_spec_id=spec.operation_spec_id,
                    operation_spec_hash=spec.semantic_content_hash,
                    session_epoch=state.session_epoch,
                    activated_at=completion.result.available_at,
                    clock_domain_id=(
                        self.data_plane.runtime.config.clock_domain_id
                    ),
                    input_barrier_at=completion.result.available_at,
                    required_roles=(
                        self.data_plane.composition_spec.required_roles
                    ),
                )
                self.data_plane.begin_composition_activation(
                    scope=activation_scope,
                )
            self.session_controller.complete_start_operation(
                operation_spec_id=transition.intent.operation_spec_id,
                result=completion.result,
                completion_accepted=completion.accepted,
                source_fact_id=projected.fact_id,
                source_fact_hash=projected.semantic_content_hash,
                start_intent_id=transition.intent.intent_id,
            )
        elif kind == "rearm_composition_session":
            receipt = self._rearm_receipt_by_spec_id.get(
                str(transition.intent.operation_spec_id)
            )
            if completion.result.succeeded and not completion.accepted:
                self._force_rearm_failure_hold(
                    reason="cs6b1i_rearm_completion_failed"
                )
            self.session_controller.complete_rearm_operation(
                rearm_spec_id=transition.intent.operation_spec_id,
                receipt=receipt,
                result=completion.result,
                completion_accepted=completion.accepted,
                source_fact_id=projected.fact_id,
                source_fact_hash=projected.semantic_content_hash,
                rearm_intent_id=transition.intent.intent_id,
            )
        else:
            self.session_controller.complete_stop_operation(
                result=completion.result,
                completion_accepted=completion.accepted,
                source_fact_id=projected.fact_id,
                source_fact_hash=projected.semantic_content_hash,
            )
        return execution

    def _require_active_session(self) -> None:
        station = self.station.state
        self.session_controller.require_active(
            station_lifecycle_state=station.lifecycle_state,
            composer_runtime_state=(
                station.source_state_by_slot[self.composer_source_slot_id]
            ),
        )

    def submit_human_emission(
        self,
        emission: HumanCommandEmission,
        *,
        observed_at: float,
    ) -> CompositionInputAdmissionReceipt:
        self._require_active_session()
        return self.publication_barrier.execute_if_enabled(
            lambda: self.data_plane.submit_human_emission(
                emission,
                observed_at=observed_at,
            )
        )

    def submit_replay_emission(
        self,
        emission: ReplaySourceFrameEmission,
    ) -> CompositionInputAdmissionReceipt:
        self._require_active_session()
        return self.publication_barrier.execute_if_enabled(
            lambda: self.data_plane.submit_replay_emission(emission)
        )

    def compose_at(
        self,
        *,
        decision_key: str,
        target_effective_at: float,
    ) -> CS6b1hFormalCompositionStep:
        self._require_active_session()
        step = self.publication_barrier.execute_if_enabled(
            lambda: self.data_plane.compose_at(
                decision_key=decision_key,
                target_effective_at=target_effective_at,
            )
        )
        if step.proposal_admission is not None:
            return step
        if step.hold_transition is not None:
            self._last_hold_outcome = step.hold_transition
        barrier = self.session_controller.disable_publication(
            reason="composition_not_publishable"
        )
        state = self.current_session_state
        if state is None or state.phase != "active":
            return step
        source_id = None
        if step.disposition is not None:
            source_id = step.disposition.disposition_id
        elif step.hold_transition is not None:
            source_id = step.hold_transition.event.event_id
        fact = next(
            (
                item
                for item in reversed(self.data_plane.runtime.trace_facts)
                if item.fact_id == source_id
            ),
            None,
        )
        if fact is None:
            raise RuntimeError(
                "Composition HOLD lacks an exact recorded source fact."
            )
        self.session_controller.observe_publication_barrier(
            barrier,
            causal_station_fact_id=fact.fact_id,
            occurred_at=fact.occurred_at,
            clock_domain_id=fact.clock_domain_id,
        )
        self.session_controller.advance(
            transition="hold",
            reason=(
                step.blocked_reason
                or step.error
                or (
                    "composition_not_publishable"
                    if step.disposition is None
                    else step.disposition.reason
                )
            ),
            source_fact_id=fact.fact_id,
            source_fact_hash=fact.semantic_content_hash,
        )
        return step

    def step(
        self,
        *,
        now: float | None = None,
        physics_ticks_after_dispatch: int = 0,
    ):
        self._require_active_session()
        return self.data_plane.runtime.step(
            now=now,
            physics_ticks_after_dispatch=physics_ticks_after_dispatch,
        )

    def close(self) -> None:
        self.session_controller.close()
        self.data_plane.close()


def build_scoped_cs6b1i_mujoco_pose_composition_session(
    runtime: MujocoDeployRuntime,
    human_source_spec: HumanPoseSourceContract,
    replay_source: ReplayProposalSource,
    replay_observer_binding: ReplaySourceObserverBinding,
    composition_spec: CompositionSpec,
    authority_gate_spec: CompositionAuthorityGateSpec,
    *,
    station_clock: Callable[[], float] | None = None,
    station_clock_domain_id: str = "control-station-composition-monotonic",
    source_difference_spec: SourceDifferenceSpec | None = None,
) -> ScopedMujocoPoseCompositionSession:
    data_plane = build_scoped_cs6b1h_mujoco_pose_composer_authority(
        runtime,
        human_source_spec,
        replay_source,
        composition_spec,
        authority_gate_spec,
        source_difference_spec=source_difference_spec,
    )
    manifest = build_cs6b1i_pose_composition_session_manifest(
        data_plane,
        human_source_spec,
        replay_observer_binding,
    )
    clock = runtime.clock if station_clock is None else station_clock
    station = ControlStationRuntime(
        manifest=manifest,
        runtime_session_id=runtime.config.runtime_session_id,
        clock_domain_id=station_clock_domain_id,
        clock=clock,
    )
    return ScopedMujocoPoseCompositionSession(
        station=station,
        data_plane=data_plane,
        human_source_slot_id=human_source_spec.source_id,
        replay_source_slot_id=replay_observer_binding.source_slot_id,
        composer_source_slot_id=(
            f"composer-slot:{runtime.config.composer_producer_id}"
        ),
        clock=clock,
        clock_domain_id=station_clock_domain_id,
    )


def build_scoped_cs6b1h_mujoco_pose_composer_authority(
    runtime: MujocoDeployRuntime,
    human_source_spec: HumanPoseSourceContract,
    replay_source: ReplayProposalSource,
    composition_spec: CompositionSpec,
    authority_gate_spec: CompositionAuthorityGateSpec,
    *,
    source_difference_spec: SourceDifferenceSpec | None = None,
) -> MujocoPoseComposerAuthorityComposition:
    """Build the independent formal sim profile without starting any source."""

    return MujocoPoseComposerAuthorityComposition(
        runtime=runtime,
        human_adapter=build_mujoco_pose_human_proposal_adapter(
            runtime,
            human_source_spec,
        ),
        replay_source=replay_source,
        composition_spec=composition_spec,
        authority_gate_spec=authority_gate_spec,
        source_difference_spec=source_difference_spec,
    )


__all__ = [
    "CS6B1H_BINDING_HASH_DOMAIN_V1",
    "CS6B1H_BINDING_HASH_DOMAIN_V2",
    "CS6B1H_HUMAN_POSE_ROLE_V1",
    "CS6B1H_REPLAY_POSE_ROLE_V1",
    "CS6B1H_STEP_HASH_DOMAIN_V1",
    "CS6B1I_COMPOSITION_SESSION_EXECUTION_MANIFEST_ID_V1",
    "CS6B1I_COMPOSITION_SESSION_MANIFEST_ID_V1",
    "CS6B1I_COMPOSITION_SESSION_TRACE_PROFILE_ID_V1",
    "CS6B1I_SIM_REARM_READINESS_CHECKS_V1",
    "CS6b1hFormalCompositionStep",
    "CS6b1hPoseComposerBinding",
    "MujocoPoseComposerAuthorityComposition",
    "ScopedMujocoPoseCompositionSession",
    "build_cs6b1i_pose_composition_session_execution_manifest",
    "build_cs6b1i_pose_composition_session_manifest",
    "build_scoped_cs6b1i_mujoco_pose_composition_session",
    "build_scoped_cs6b1h_mujoco_pose_composer_authority",
]
