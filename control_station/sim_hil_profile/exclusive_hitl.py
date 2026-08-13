"""Sim application adapter for the backend-neutral exclusive HITL switch.

The adapter translates one Sim-HIL application's existing HOLD, Runtime Home,
source-selection, activation, and fresh-emission receipts into the generic
Station HITL contract.  It does not own composition, trace, model inference,
IK, safety, or dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash

from control_station.hitl import ExclusiveHITLTransitionObservation


_RUNTIME_HOME_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.control_station.runtime_home_capture.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class SimHILExclusiveTransitionPort:
    """Bind a Sim application to one discrete streaming-allocation switch."""

    application: Any
    select_streaming_source: Callable[..., Mapping[str, object]]
    hold_evidence_refs: Callable[[Any], tuple[str, ...]]
    target_source: str
    target_mode: str
    requested_at: float

    def __post_init__(self) -> None:
        if not callable(self.select_streaming_source):
            raise TypeError("select_streaming_source must be callable.")
        if not callable(self.hold_evidence_refs):
            raise TypeError("hold_evidence_refs must be callable.")
        if not callable(getattr(self.application, "clock", None)):
            raise TypeError("application must expose clock().")
        source = _required(self.target_source, name="target_source")
        if source not in {"policy", "human"}:
            raise ValueError("target_source must be policy or human.")
        object.__setattr__(self, "target_source", source)
        object.__setattr__(
            self,
            "target_mode",
            _required(self.target_mode, name="target_mode"),
        )
        object.__setattr__(self, "requested_at", float(self.requested_at))

    def transition(
        self,
        selected_source: str,
    ) -> tuple[dict[str, object], ExclusiveHITLTransitionObservation]:
        if selected_source != self.target_source:
            raise ValueError("Exclusive HITL transition target changed.")
        mode = dict(
            self.select_streaming_source(
                target_source=self.target_source,
                target_mode=self.target_mode,
                requested_at=self.requested_at,
            )
        )
        hold_refs = self.failure_hold_evidence()
        if not hold_refs:
            raise RuntimeError("Exclusive HITL switch has no lower HOLD evidence.")
        selection_id = str(mode.get("selection_record_id") or "").strip()
        selection_hash = str(mode.get("selection_record_hash") or "").strip()
        if not selection_id or not selection_hash:
            raise RuntimeError(
                "Exclusive HITL switch has no typed source-selection receipt."
            )

        prime = getattr(
            self.application,
            "last_activation_prime_evidence",
            None,
        )
        if prime is None:
            raise RuntimeError(
                "Exclusive HITL switch has no activation-prime evidence."
            )
        target_emission_id, target_emission_hash = (
            prime.target_emission_identity(self.target_source)
        )
        if self.target_source == "policy" and not bool(
            getattr(prime, "policy_reset_performed", False)
        ):
            raise RuntimeError(
                "Policy resume did not invalidate the prior policy plan."
            )

        home_capture = None
        home_capture_id = None
        home_capture_hash = None
        if self.target_source == "human":
            preparation = dict(mode.get("human_preparation") or {})
            home_capture = dict(preparation.get("runtime_home") or {})
            home_capture_id = str(home_capture.get("capture_id") or "").strip()
            if not home_capture_id:
                raise RuntimeError("Human takeover did not capture Runtime Home.")
            home_capture_hash = canonical_content_hash(
                home_capture,
                domain=_RUNTIME_HOME_HASH_DOMAIN_DEVELOPMENT,
            )

        execution = self.application.establish_activation_execution(
            target_source=self.target_source,
        )

        return mode, ExclusiveHITLTransitionObservation(
            target_source=self.target_source,
            completed_at=max(
                self.requested_at,
                float(self.application.clock()),
            ),
            hold_evidence_refs=hold_refs,
            home_capture_id=home_capture_id,
            home_capture_hash=home_capture_hash,
            home_capture=home_capture,
            source_selection_record_id=selection_id,
            source_selection_record_hash=selection_hash,
            activation_id=prime.activation_id,
            activation_hash=prime.activation_hash,
            target_emission_id=target_emission_id,
            target_emission_hash=target_emission_hash,
            target_id=execution.target_id,
            target_hash=execution.target_hash,
            target_admission_receipt_id=(
                execution.target_admission_receipt_id
            ),
            target_completion_receipt_id=(
                execution.target_completion_receipt_id
            ),
            command_id=execution.command_id,
            dispatch_receipt_id=execution.dispatch_receipt_id,
            source_reset_performed=bool(
                self.target_source == "policy"
                and getattr(prime, "policy_reset_performed", False)
            ),
        )

    def hold_after_failure(self) -> None:
        if not bool(self.application.status().get("held", False)):
            self.application.hold(reason="exclusive_hitl_switch_failed")

    def failure_hold_evidence(self) -> tuple[str, ...]:
        return tuple(self.hold_evidence_refs(self.application))


__all__ = ["SimHILExclusiveTransitionPort"]
