"""Button/trigger operations for one MuJoCo allocation activation.

This Station adapter turns an already-admitted UI/device event into the generic
trigger proposal, then delegates to the existing allocation-override or anchor
owner.  It does not poll devices, compose actions, change authority, create
HOLD, run safety, resolve targets, or dispatch.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash

from control_composition.allocation import (
    COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
    COMPOSITION_ALLOCATION_OVERRIDE_AUTOMATIC,
    COMPOSITION_ALLOCATION_OVERRIDE_OPERATION_KIND,
    COMPOSITION_ALLOCATION_OVERRIDE_PAYLOAD_CONTRACT_DEVELOPMENT,
    CompositionAllocationOverrideReceipt,
    CompositionAllocationOverrideRequest,
)
from control_composition.triggers import (
    COMPOSITION_TRIGGER_EDGE_EACH_EVENT,
    COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT,
    COMPOSITION_TRIGGER_STATUS_TRIGGERED,
    CompositionOperationTriggerProposal,
    CompositionTriggerEvidence,
    CompositionTriggerEvaluation,
    CompositionTriggerRule,
    empty_composition_trigger_state,
    evaluate_composition_operation_trigger,
)
from control_composition.residual_anchor import ResidualAnchorRequest

from .allocation_publication import StationAllocationPublicationPort
from .mujoco_policy_human_allocation import (
    MujocoPolicyHumanAllocationProfileBundle,
)
from .mujoco_policy_human_allocation_input import (
    MujocoPolicyHumanAllocationInputController,
)


MUJOCO_ALLOCATION_SIGNAL_AUTOMATIC = "allocation_automatic"
MUJOCO_ALLOCATION_SIGNAL_BASE_ONLY = "allocation_base_only"
MUJOCO_ALLOCATION_SIGNAL_RELATIVE_OVERLAY = "allocation_relative_overlay"
MUJOCO_ALLOCATION_SIGNAL_ABSOLUTE_OVERLAY_OWNER = (
    "allocation_absolute_overlay_owner"
)
MUJOCO_ALLOCATION_SIGNAL_REBASE_AND_CLEAR = "residual_rebase_and_clear"
MUJOCO_ALLOCATION_BUTTON_SIGNALS = (
    MUJOCO_ALLOCATION_SIGNAL_AUTOMATIC,
    MUJOCO_ALLOCATION_SIGNAL_BASE_ONLY,
    MUJOCO_ALLOCATION_SIGNAL_RELATIVE_OVERLAY,
    MUJOCO_ALLOCATION_SIGNAL_ABSOLUTE_OVERLAY_OWNER,
    MUJOCO_ALLOCATION_SIGNAL_REBASE_AND_CLEAR,
)
MUJOCO_ALLOCATION_REBASE_OPERATION_KIND = "request_residual_rebase_and_clear"
MUJOCO_ALLOCATION_REBASE_PAYLOAD_CONTRACT_DEVELOPMENT = (
    "dexteleop.mujoco_allocation_rebase_trigger_payload.development"
)
MUJOCO_ALLOCATION_BUTTON_SOURCE_SPEC_HASH_DEVELOPMENT = (
    "dexteleop.control_station.allocation_button_source.development"
)
MUJOCO_ALLOCATION_BUTTON_EVENT_PAYLOAD_CONTRACT_DEVELOPMENT = (
    "dexteleop.control_station.allocation_button_event.development"
)
MUJOCO_ALLOCATION_OPERATION_RECEIPT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.mujoco_allocation_operation_receipt.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class MujocoPolicyHumanAllocationOperationReceipt:
    signal_name: str
    event_id: str
    trigger_evaluation: CompositionTriggerEvaluation
    override_receipt: CompositionAllocationOverrideReceipt | None = None
    anchor_request: ResidualAnchorRequest | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        signal = _required(self.signal_name, name="signal_name")
        event_id = _required(self.event_id, name="event_id")
        if signal not in MUJOCO_ALLOCATION_BUTTON_SIGNALS:
            raise ValueError("Unsupported allocation button signal.")
        if not isinstance(self.trigger_evaluation, CompositionTriggerEvaluation):
            raise TypeError("trigger_evaluation must be typed.")
        proposal = self.trigger_evaluation.proposal
        if (
            self.trigger_evaluation.status != COMPOSITION_TRIGGER_STATUS_TRIGGERED
            or not isinstance(proposal, CompositionOperationTriggerProposal)
        ):
            raise ValueError("Operation receipt requires one triggered proposal.")
        if (self.override_receipt is None) == (self.anchor_request is None):
            raise ValueError("Operation receipt requires exactly one owner result.")
        if self.override_receipt is not None and not isinstance(
            self.override_receipt,
            CompositionAllocationOverrideReceipt,
        ):
            raise TypeError("override_receipt must be typed.")
        if self.anchor_request is not None and not isinstance(
            self.anchor_request,
            ResidualAnchorRequest,
        ):
            raise TypeError("anchor_request must be typed.")
        payload = {
            "signal_name": signal,
            "event_id": event_id,
            "trigger_status": self.trigger_evaluation.status,
            "trigger_reason": self.trigger_evaluation.reason,
            "trigger_state_hash": (
                self.trigger_evaluation.state.semantic_content_hash
            ),
            "trigger_evidence_hash": (
                self.trigger_evaluation.evidence.semantic_content_hash
            ),
            "trigger_proposal_id": proposal.proposal_id,
            "trigger_proposal_hash": proposal.semantic_content_hash,
            "override_receipt_id": (
                None
                if self.override_receipt is None
                else self.override_receipt.receipt_id
            ),
            "override_receipt_hash": (
                None
                if self.override_receipt is None
                else self.override_receipt.semantic_content_hash
            ),
            "anchor_request_id": (
                None if self.anchor_request is None else self.anchor_request.request_id
            ),
            "anchor_request_hash": (
                None
                if self.anchor_request is None
                else self.anchor_request.semantic_content_hash
            ),
            "authoritative": False,
        }
        object.__setattr__(self, "signal_name", signal)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    MUJOCO_ALLOCATION_OPERATION_RECEIPT_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def receipt_id(self) -> str:
        return f"mujoco-allocation-operation-receipt:{self.semantic_content_hash}"

    @property
    def authoritative(self) -> bool:
        return False


class MujocoPolicyHumanAllocationOperationController:
    """Activation-local owner adapter for button and future trigger sources."""

    def __init__(
        self,
        *,
        runtime: Any,
        bundle: MujocoPolicyHumanAllocationProfileBundle,
        allocation_publication_port: StationAllocationPublicationPort,
        human_input_controller: MujocoPolicyHumanAllocationInputController,
        proposal_ttl_sec: float = 1.0,
    ) -> None:
        if not isinstance(bundle, MujocoPolicyHumanAllocationProfileBundle):
            raise TypeError("bundle must be typed.")
        if not isinstance(allocation_publication_port, StationAllocationPublicationPort):
            raise TypeError("allocation_publication_port must be typed.")
        if not isinstance(
            human_input_controller,
            MujocoPolicyHumanAllocationInputController,
        ):
            raise TypeError("human_input_controller must be typed.")
        activation = human_input_controller.activation_scope
        if (
            allocation_publication_port.allocation_runtime is None
            or allocation_publication_port.allocation_runtime.activation_scope
            != activation
        ):
            raise ValueError("Operation owners belong to another activation.")
        ttl = float(proposal_ttl_sec)
        if not math.isfinite(ttl) or ttl <= 0.0:
            raise ValueError("proposal_ttl_sec must be positive.")
        self.runtime = runtime
        self.bundle = bundle
        self.allocation_publication_port = allocation_publication_port
        self.human_input_controller = human_input_controller
        self.activation_scope = activation
        self.proposal_ttl_sec = ttl
        self._rules = self._build_rules()
        self._state_by_signal = {
            signal: empty_composition_trigger_state(
                rule,
                activation_scope=activation,
            )
            for signal, rule in self._rules.items()
        }
        self._next_sequence_by_signal = {
            signal: 0 for signal in self._rules
        }
        self._receipt_by_event_key: dict[
            tuple[str, str], MujocoPolicyHumanAllocationOperationReceipt
        ] = {}
        self._source_identity_by_event_key: dict[
            tuple[str, str], tuple[str, str]
        ] = {}
        self._last_receipt: MujocoPolicyHumanAllocationOperationReceipt | None = None
        self._lock = threading.RLock()

    def _build_rules(self) -> dict[str, CompositionTriggerRule]:
        components = self.bundle.allocation_spec.component_ids
        subject = self.bundle.config.profile_id
        requested_mode_by_signal = {
            MUJOCO_ALLOCATION_SIGNAL_AUTOMATIC: (
                COMPOSITION_ALLOCATION_OVERRIDE_AUTOMATIC
            ),
            MUJOCO_ALLOCATION_SIGNAL_BASE_ONLY: (
                COMPOSITION_ALLOCATION_MODE_BASE_ONLY
            ),
            MUJOCO_ALLOCATION_SIGNAL_RELATIVE_OVERLAY: (
                COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY
            ),
            MUJOCO_ALLOCATION_SIGNAL_ABSOLUTE_OVERLAY_OWNER: (
                COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER
            ),
        }
        promotable_modes = frozenset(self.bundle.config.proposal_promotable_modes)
        rules = {
            signal: CompositionTriggerRule(
                rule_id=f"{subject}.button.{signal}",
                trigger_source_id="control-station-gui-button",
                trigger_source_spec_hash=(
                    MUJOCO_ALLOCATION_BUTTON_SOURCE_SPEC_HASH_DEVELOPMENT
                ),
                origin_kind=COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT,
                signal_name=signal,
                edge_policy=COMPOSITION_TRIGGER_EDGE_EACH_EVENT,
                operation_kind=COMPOSITION_ALLOCATION_OVERRIDE_OPERATION_KIND,
                subject_id=subject,
                component_ids=components,
                operation_payload_contract_hash=(
                    COMPOSITION_ALLOCATION_OVERRIDE_PAYLOAD_CONTRACT_DEVELOPMENT
                ),
                operation_payload={"requested_mode": mode},
                proposal_ttl_sec=self.proposal_ttl_sec,
            )
            for signal, mode in requested_mode_by_signal.items()
            if mode == COMPOSITION_ALLOCATION_OVERRIDE_AUTOMATIC
            or mode in promotable_modes
        }
        rules[MUJOCO_ALLOCATION_SIGNAL_REBASE_AND_CLEAR] = CompositionTriggerRule(
            rule_id=f"{subject}.button.residual-rebase-and-clear",
            trigger_source_id="control-station-gui-button",
            trigger_source_spec_hash=(
                MUJOCO_ALLOCATION_BUTTON_SOURCE_SPEC_HASH_DEVELOPMENT
            ),
            origin_kind=COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT,
            signal_name=MUJOCO_ALLOCATION_SIGNAL_REBASE_AND_CLEAR,
            edge_policy=COMPOSITION_TRIGGER_EDGE_EACH_EVENT,
            operation_kind=MUJOCO_ALLOCATION_REBASE_OPERATION_KIND,
            subject_id=subject,
            component_ids=components,
            operation_payload_contract_hash=(
                MUJOCO_ALLOCATION_REBASE_PAYLOAD_CONTRACT_DEVELOPMENT
            ),
            operation_payload={"capture_mode": "first_fresh_after_barrier"},
            proposal_ttl_sec=self.proposal_ttl_sec,
        )
        return rules

    def submit_button_event(
        self,
        *,
        signal_name: str,
        event_id: str,
        source_fact_id: str,
        source_fact_hash: str,
        occurred_at: float,
    ) -> MujocoPolicyHumanAllocationOperationReceipt:
        signal = _required(signal_name, name="signal_name")
        event = _required(event_id, name="event_id")
        source_id = _required(source_fact_id, name="source_fact_id")
        source_hash = _required(source_fact_hash, name="source_fact_hash")
        if signal not in self._rules:
            raise ValueError("Unsupported allocation button signal.")
        with self._lock:
            key = (signal, event)
            previous = self._receipt_by_event_key.get(key)
            if previous is not None:
                if self._source_identity_by_event_key[key] != (
                    source_id,
                    source_hash,
                ):
                    raise ValueError("Button event id was reused for another fact.")
                return previous
            now = max(float(occurred_at), float(self.runtime.clock()))
            cut = self.runtime.capture_control_cut(cutoff_at=now)
            rule = self._rules[signal]
            sequence = self._next_sequence_by_signal[signal]
            evidence = CompositionTriggerEvidence(
                runtime_session_id=cut.runtime_session_id,
                control_domain_id=cut.control_domain_id,
                clock_reset_epoch=cut.clock_reset_epoch,
                composition_activation_id=self.activation_scope.activation_id,
                composition_activation_hash=(
                    self.activation_scope.semantic_content_hash
                ),
                trigger_source_id=rule.trigger_source_id,
                trigger_source_spec_hash=rule.trigger_source_spec_hash,
                origin_kind=COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT,
                signal_name=signal,
                producer_epoch=self.activation_scope.session_epoch,
                sequence=sequence,
                active=True,
                occurred_at=min(float(occurred_at), cut.cutoff_at),
                available_at=cut.cutoff_at,
                valid_until=cut.cutoff_at + self.proposal_ttl_sec,
                clock_domain_id=cut.clock_domain_id,
                control_admission_sequence=cut.cutoff_admission_sequence,
                payload_contract_hash=(
                    MUJOCO_ALLOCATION_BUTTON_EVENT_PAYLOAD_CONTRACT_DEVELOPMENT
                ),
                semantic_payload={"event_id": event, "pressed": True},
                source_fact_ids=(source_id,),
                source_fact_hashes=(source_hash,),
            )
            evaluation = evaluate_composition_operation_trigger(
                self._state_by_signal[signal],
                rule=rule,
                evidence=evidence,
                control_cut=cut,
                activation_scope=self.activation_scope,
            )
            if evaluation.status != COMPOSITION_TRIGGER_STATUS_TRIGGERED:
                raise RuntimeError("Distinct button event did not trigger operation.")
            proposal = evaluation.proposal
            assert proposal is not None
            override_receipt = None
            anchor_request = None
            if proposal.operation_kind == (
                COMPOSITION_ALLOCATION_OVERRIDE_OPERATION_KIND
            ):
                requested_mode = _required(
                    proposal.operation_payload.get("requested_mode"),
                    name="requested_mode",
                )
                override_receipt = self.allocation_publication_port.apply_override(
                    CompositionAllocationOverrideRequest(
                        runtime_session_id=proposal.runtime_session_id,
                        control_domain_id=proposal.control_domain_id,
                        composition_activation_id=(
                            proposal.composition_activation_id
                        ),
                        composition_activation_hash=(
                            proposal.composition_activation_hash
                        ),
                        allocation_spec_hash=(
                            self.bundle.allocation_spec.semantic_content_hash
                        ),
                        component_ids=proposal.component_ids,
                        requested_mode=requested_mode,
                        requested_at=proposal.triggered_at,
                        available_at=proposal.available_at,
                        clock_domain_id=proposal.clock_domain_id,
                        source_fact_ids=(
                            proposal.proposal_id,
                            *proposal.source_fact_ids,
                        ),
                        source_fact_hashes=(
                            proposal.semantic_content_hash,
                            *proposal.source_fact_hashes,
                        ),
                    )
                )
            elif proposal.operation_kind == MUJOCO_ALLOCATION_REBASE_OPERATION_KIND:
                admitted_at = max(proposal.available_at, float(self.runtime.clock()))
                anchor_request = self.human_input_controller.request_anchor(
                    requested_at=admitted_at,
                    effective_not_before_target_at=admitted_at,
                )
            else:
                raise RuntimeError("Trigger proposal names an unsupported owner.")
            receipt = MujocoPolicyHumanAllocationOperationReceipt(
                signal_name=signal,
                event_id=event,
                trigger_evaluation=evaluation,
                override_receipt=override_receipt,
                anchor_request=anchor_request,
            )
            self._state_by_signal[signal] = evaluation.state
            self._next_sequence_by_signal[signal] = sequence + 1
            self._receipt_by_event_key[key] = receipt
            self._source_identity_by_event_key[key] = (source_id, source_hash)
            self._last_receipt = receipt
            return receipt

    def status(self) -> dict[str, object]:
        with self._lock:
            override = self.allocation_publication_port.snapshot().allocation_override_state
            last = self._last_receipt
            return {
                "supported_signals": list(self._rules),
                "allocation_override_mode": (
                    None if override is None else override.requested_mode
                ),
                "allocation_override_revision": (
                    None if override is None else override.revision
                ),
                "last_operation_receipt_id": (
                    None if last is None else last.receipt_id
                ),
                "last_operation_signal": (
                    None if last is None else last.signal_name
                ),
                "last_anchor_request_id": (
                    None
                    if last is None or last.anchor_request is None
                    else last.anchor_request.request_id
                ),
            }


__all__ = [
    "MUJOCO_ALLOCATION_BUTTON_SIGNALS",
    "MUJOCO_ALLOCATION_SIGNAL_AUTOMATIC",
    "MUJOCO_ALLOCATION_SIGNAL_ABSOLUTE_OVERLAY_OWNER",
    "MUJOCO_ALLOCATION_SIGNAL_BASE_ONLY",
    "MUJOCO_ALLOCATION_SIGNAL_REBASE_AND_CLEAR",
    "MUJOCO_ALLOCATION_SIGNAL_RELATIVE_OVERLAY",
    "MujocoPolicyHumanAllocationOperationController",
    "MujocoPolicyHumanAllocationOperationReceipt",
]
