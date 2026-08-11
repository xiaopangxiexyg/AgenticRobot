"""Typed lifecycle evidence for the MuJoCo allocation profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from control_composition.activation import CompositionActivationScope
from control_composition.lifecycle import CompositionSessionOperationSpec
from control_station.contracts import StationOperationResult

from .mujoco_sim_hil import MujocoSimHILPreflightReport


MUJOCO_ALLOCATION_SESSION_START_STATUS = "composition_session_ready"
MUJOCO_ALLOCATION_SESSION_REARM_STATUS = "composition_session_rearmed"


@dataclass(frozen=True)
class MujocoPolicyHumanAllocationActivationEvidence:
    """Exact facts an upper Station operation must acknowledge."""

    operation_kind: str
    operation_spec: CompositionSessionOperationSpec
    activation_scope: CompositionActivationScope
    preflight_report: MujocoSimHILPreflightReport
    producer_activations: tuple[Any, ...]
    execution_gate_snapshot: Any
    prior_activation_scope: CompositionActivationScope | None = None
    execution_gate_transition: Any | None = None

    def __post_init__(self) -> None:
        if self.operation_kind not in {
            "start_composition_session",
            "rearm_composition_session",
        }:
            raise ValueError("Unsupported allocation session operation kind.")
        if not isinstance(self.operation_spec, CompositionSessionOperationSpec):
            raise TypeError("operation_spec must be typed.")
        if not isinstance(self.activation_scope, CompositionActivationScope):
            raise TypeError("activation_scope must be typed.")
        if not isinstance(self.preflight_report, MujocoSimHILPreflightReport):
            raise TypeError("preflight_report must be typed.")
        if not self.preflight_report.passed:
            raise ValueError("Activation evidence requires a passed preflight.")
        activations = tuple(self.producer_activations)
        for activation in activations:
            if not str(getattr(activation, "activation_id", "")).strip() or not str(
                getattr(activation, "semantic_content_hash", "")
            ).strip():
                raise TypeError(
                    "producer_activations must expose typed identity and hash."
                )
            if not hasattr(activation, "available_at") or not str(
                getattr(activation, "available_clock_domain_id", "")
            ).strip():
                raise TypeError(
                    "producer_activations must expose receiver admission time."
                )
        object.__setattr__(self, "producer_activations", activations)
        gate = self.execution_gate_snapshot
        if (
            getattr(gate, "status", None) != "ready"
            or not str(getattr(gate, "semantic_content_hash", "")).strip()
            or not hasattr(gate, "cutoff_at")
        ):
            raise TypeError(
                "execution_gate_snapshot must be an exact lower READY snapshot."
            )
        transition_snapshot = getattr(
            self.execution_gate_transition,
            "snapshot",
            None,
        )
        if transition_snapshot is not None and (
            transition_snapshot.semantic_content_hash
            != gate.semantic_content_hash
        ):
            raise ValueError("Rearm transition and READY snapshot differ.")
        if self.operation_kind == "start_composition_session":
            if self.prior_activation_scope is not None:
                raise ValueError("Session start cannot carry a prior activation.")
        elif not isinstance(
            self.prior_activation_scope,
            CompositionActivationScope,
        ):
            raise TypeError("Session rearm requires the prior activation scope.")

    @property
    def required_evidence_refs(self) -> tuple[str, ...]:
        refs = [
            self.operation_spec.operation_spec_id,
            self.operation_spec.semantic_content_hash,
            self.activation_scope.activation_id,
            self.activation_scope.semantic_content_hash,
            self.preflight_report.semantic_content_hash,
            str(self.execution_gate_snapshot.semantic_content_hash),
        ]
        if self.prior_activation_scope is not None:
            refs.extend(
                (
                    self.prior_activation_scope.activation_id,
                    self.prior_activation_scope.semantic_content_hash,
                )
            )
        for activation in self.producer_activations:
            refs.extend(
                (
                    str(activation.activation_id),
                    str(activation.semantic_content_hash),
                )
            )
        transition = self.execution_gate_transition
        if transition is not None:
            refs.extend(str(value) for value in transition.lower_event_ids)
        return tuple(dict.fromkeys(refs))

    @property
    def causal_ready_at(self) -> float:
        """Latest same-clock fact that the operation result must follow."""

        clock = self.activation_scope.clock_domain_id
        values = [
            float(self.activation_scope.activated_at),
            float(self.execution_gate_snapshot.cutoff_at),
        ]
        values.extend(
            float(activation.available_at)
            for activation in self.producer_activations
            if activation.available_clock_domain_id == clock
        )
        transition_event = getattr(self.execution_gate_transition, "event", None)
        if (
            transition_event is not None
            and getattr(transition_event, "available_clock_domain_id", None) == clock
        ):
            values.append(float(transition_event.available_at))
        return max(values)

    @property
    def causal_clock_domain_id(self) -> str:
        """Receiver clock used by all facts contributing to causal_ready_at."""

        return self.activation_scope.clock_domain_id


AllocationSessionResultProvider = Callable[
    [MujocoPolicyHumanAllocationActivationEvidence],
    StationOperationResult,
]


def validate_allocation_session_operation_result(
    result: StationOperationResult,
    evidence: MujocoPolicyHumanAllocationActivationEvidence,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    clock_domain_id: str,
    composer_source_slot_id: str,
) -> None:
    """Require exact typed acknowledgement before publication can open."""

    if not isinstance(result, StationOperationResult):
        raise TypeError("Operation provider must return StationOperationResult.")
    expected_status = (
        MUJOCO_ALLOCATION_SESSION_START_STATUS
        if evidence.operation_kind == "start_composition_session"
        else MUJOCO_ALLOCATION_SESSION_REARM_STATUS
    )
    exact = (
        result.runtime_session_id == runtime_session_id
        and result.control_domain_id == control_domain_id
        and result.operation_kind == evidence.operation_kind
        and result.subject_id == composer_source_slot_id
        and result.succeeded
        and result.status == expected_status
        and result.available_clock_domain_id == clock_domain_id
    )
    if not exact:
        raise ValueError("Station operation result does not bind this session.")
    if result.available_at < evidence.causal_ready_at:
        raise ValueError("Station operation result predates activation readiness.")
    missing = set(evidence.required_evidence_refs) - set(result.evidence_refs)
    if missing:
        raise ValueError(
            "Station operation result misses activation evidence: "
            + ",".join(sorted(missing))
        )


__all__ = [
    "MUJOCO_ALLOCATION_SESSION_REARM_STATUS",
    "MUJOCO_ALLOCATION_SESSION_START_STATUS",
    "AllocationSessionResultProvider",
    "MujocoPolicyHumanAllocationActivationEvidence",
    "validate_allocation_session_operation_result",
]
