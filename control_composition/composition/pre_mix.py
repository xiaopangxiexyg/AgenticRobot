"""Optional same-cut composition policy hooks.

This module deliberately does not define motion safety.  A pre-mix policy may
look at the already-admitted action/context cut and decide whether the
composition should continue.  It cannot rewrite source values, issue an
authority or safety receipt, run IK, or dispatch a command.

The first contract only supports ``pass``, ``no_action`` and ``invalid``.
Source-level dropping and bounded previous-result reuse are intentionally left
for a later additive contract so that the initial seam does not add hidden
snapshot or persistence semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.immutable import deep_freeze

from ..information import CompositionInformationPlan
from .context import CompositionContextSnapshot
from .core import CompositionInputSnapshot, CompositionSpec


COMPOSITION_PRE_MIX_POLICY_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.composition_pre_mix_policy_spec.v1"
)
COMPOSITION_PRE_MIX_POLICY_INPUT_HASH_DOMAIN_V1 = (
    "dexteleop.composition_pre_mix_policy_input.v1"
)
COMPOSITION_PRE_MIX_POLICY_DECISION_HASH_DOMAIN_V1 = (
    "dexteleop.composition_pre_mix_policy_decision.v1"
)
COMPOSITION_PRE_MIX_POLICY_DISPOSITIONS_V1 = frozenset(
    {"pass", "no_action", "invalid"}
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


@dataclass(frozen=True)
class CompositionPreMixPolicySpec:
    """Identity/configuration for one integration-provided policy hook.

    The payload is policy metadata only.  It is not a safety contract and
    must not contain target-specific safety ownership or execution settings.
    """

    policy_id: str
    policy_version: str = "1"
    maximum_decision_latency_sec: float | None = None
    parameter_payload: Mapping[str, Any] = field(default_factory=dict)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        policy_id = _required(self.policy_id, name="policy_id")
        policy_version = _required(self.policy_version, name="policy_version")
        maximum_latency = self.maximum_decision_latency_sec
        if maximum_latency is not None:
            maximum_latency = float(maximum_latency)
            if not math.isfinite(maximum_latency) or maximum_latency <= 0.0:
                raise ValueError(
                    "maximum_decision_latency_sec must be finite and positive."
                )
        parameters = deep_freeze(dict(self.parameter_payload))
        payload = {
            "policy_id": policy_id,
            "policy_version": policy_version,
            "maximum_decision_latency_sec": maximum_latency,
            "parameter_payload": parameters,
        }
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "policy_version", policy_version)
        object.__setattr__(
            self,
            "maximum_decision_latency_sec",
            maximum_latency,
        )
        object.__setattr__(self, "parameter_payload", parameters)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COMPOSITION_PRE_MIX_POLICY_SPEC_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class CompositionPreMixPolicyInput:
    """The immutable same-cut view presented to a pre-mix policy."""

    runtime_session_id: str
    control_domain_id: str
    target_effective_at: float
    composition_spec: CompositionSpec
    input_snapshot: CompositionInputSnapshot
    context_snapshot: CompositionContextSnapshot | None = None
    information_plan: CompositionInformationPlan | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        runtime_session_id = _required(
            self.runtime_session_id,
            name="runtime_session_id",
        )
        control_domain_id = _required(
            self.control_domain_id,
            name="control_domain_id",
        )
        target = _finite(
            self.target_effective_at,
            name="target_effective_at",
        )
        if not isinstance(self.composition_spec, CompositionSpec):
            raise TypeError("composition_spec must be a CompositionSpec.")
        if not isinstance(self.input_snapshot, CompositionInputSnapshot):
            raise TypeError("input_snapshot must be a CompositionInputSnapshot.")
        if (
            self.input_snapshot.composition_spec_hash
            != self.composition_spec.semantic_content_hash
        ):
            raise ValueError("Pre-mix policy input/spec hash mismatch.")
        if (
            self.input_snapshot.control_domain_id != control_domain_id
            or self.input_snapshot.target_effective_at != target
        ):
            raise ValueError(
                "Pre-mix policy input belongs to another target/domain."
            )
        context_snapshot_id: str | None = None
        context_snapshot_hash: str | None = None
        if self.context_snapshot is not None:
            if not isinstance(self.context_snapshot, CompositionContextSnapshot):
                raise TypeError(
                    "context_snapshot must be a CompositionContextSnapshot."
                )
            if (
                self.context_snapshot.control_domain_id != control_domain_id
                or self.context_snapshot.cutoff_at
                != self.input_snapshot.cutoff_at
                or self.context_snapshot.cutoff_admission_sequence
                != self.input_snapshot.cutoff_admission_sequence
                or self.context_snapshot.target_effective_at != target
            ):
                raise ValueError("Pre-mix policy context must share the input cut.")
            context_snapshot_id = self.context_snapshot.context_snapshot_id
            context_snapshot_hash = self.context_snapshot.semantic_content_hash
        information_plan_hash = (
            None
            if self.information_plan is None
            else self.information_plan.semantic_content_hash
        )
        payload = {
            "runtime_session_id": runtime_session_id,
            "control_domain_id": control_domain_id,
            "target_effective_at": target,
            "composition_spec_hash": self.composition_spec.semantic_content_hash,
            "composition_input_snapshot_id": (
                self.input_snapshot.composition_input_snapshot_id
            ),
            "composition_input_snapshot_hash": (
                self.input_snapshot.semantic_content_hash
            ),
            "context_snapshot_id": context_snapshot_id,
            "context_snapshot_hash": context_snapshot_hash,
            "information_plan_hash": information_plan_hash,
        }
        object.__setattr__(self, "runtime_session_id", runtime_session_id)
        object.__setattr__(self, "control_domain_id", control_domain_id)
        object.__setattr__(self, "target_effective_at", target)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COMPOSITION_PRE_MIX_POLICY_INPUT_HASH_DOMAIN_V1,
            ),
        )

    @property
    def policy_input_id(self) -> str:
        return f"composition-pre-mix-policy-input:{self.semantic_content_hash}"


@dataclass(frozen=True)
class CompositionPreMixPolicyDecision:
    """A non-authoritative decision for one exact policy input cut."""

    policy_spec_hash: str
    policy_input_hash: str
    disposition: str
    reason: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        policy_spec_hash = _required(
            self.policy_spec_hash,
            name="policy_spec_hash",
        )
        policy_input_hash = _required(
            self.policy_input_hash,
            name="policy_input_hash",
        )
        disposition = _required(self.disposition, name="disposition")
        if disposition not in COMPOSITION_PRE_MIX_POLICY_DISPOSITIONS_V1:
            raise ValueError(f"Unsupported pre-mix disposition {disposition!r}.")
        reason = _required(self.reason, name="reason")
        payload = {
            "policy_spec_hash": policy_spec_hash,
            "policy_input_hash": policy_input_hash,
            "disposition": disposition,
            "reason": reason,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COMPOSITION_PRE_MIX_POLICY_DECISION_HASH_DOMAIN_V1,
            ),
        )

    @property
    def decision_id(self) -> str:
        return (
            "composition-pre-mix-policy-decision:"
            f"{self.semantic_content_hash}"
        )


class CompositionPreMixPolicy(Protocol):
    """Integration-provided, same-cut composition policy port."""

    @property
    def spec(self) -> CompositionPreMixPolicySpec: ...

    def decide(
        self,
        policy_input: CompositionPreMixPolicyInput,
    ) -> CompositionPreMixPolicyDecision: ...


class DefaultPassThroughCompositionPreMixPolicy:
    """Explicit no-op policy useful for adapter tests and profile wiring."""

    def __init__(self) -> None:
        self._spec = CompositionPreMixPolicySpec(
            policy_id="default_pass_through",
            policy_version="1",
            maximum_decision_latency_sec=0.002,
        )

    @property
    def spec(self) -> CompositionPreMixPolicySpec:
        return self._spec

    def decide(
        self,
        policy_input: CompositionPreMixPolicyInput,
    ) -> CompositionPreMixPolicyDecision:
        if not isinstance(policy_input, CompositionPreMixPolicyInput):
            raise TypeError("policy_input must be a CompositionPreMixPolicyInput.")
        return CompositionPreMixPolicyDecision(
            policy_spec_hash=self.spec.semantic_content_hash,
            policy_input_hash=policy_input.semantic_content_hash,
            disposition="pass",
            reason="default_pass",
        )


__all__ = [
    "COMPOSITION_PRE_MIX_POLICY_DECISION_HASH_DOMAIN_V1",
    "COMPOSITION_PRE_MIX_POLICY_DISPOSITIONS_V1",
    "COMPOSITION_PRE_MIX_POLICY_INPUT_HASH_DOMAIN_V1",
    "COMPOSITION_PRE_MIX_POLICY_SPEC_HASH_DOMAIN_V1",
    "CompositionPreMixPolicy",
    "CompositionPreMixPolicyDecision",
    "CompositionPreMixPolicyInput",
    "CompositionPreMixPolicySpec",
    "DefaultPassThroughCompositionPreMixPolicy",
]
