"""Transport-neutral policy-node protocol for synthetic and future runtimes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.composition import CompositionInputDraft
from robot_motion.action.proposal import ActionProposalDraft
from robot_motion.runtime.observation_cut import ObservationBundle

from .context import PolicyContextDraft, PolicyContextSnapshot


POLICY_NODE_OUTPUT_KINDS_V1 = frozenset(
    {"action_proposal", "policy_context", "composition_input", "no_output"}
)
POLICY_NODE_SPEC_HASH_DOMAIN_V1 = "dexteleop.policy_node_spec.v1"


@dataclass(frozen=True)
class PolicyNodeSpec:
    node_id: str
    output_kind: str
    trigger_mode: str
    nominal_hz: float
    observation_contract_hash: str
    output_contract_hash: str
    required_context_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "node_id",
            "output_kind",
            "trigger_mode",
            "observation_contract_hash",
            "output_contract_hash",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        if self.output_kind not in POLICY_NODE_OUTPUT_KINDS_V1 - {"no_output"}:
            raise ValueError("PolicyNodeSpec output_kind must describe a produced fact.")
        if self.trigger_mode not in {"fixed", "event_limited"}:
            raise ValueError("trigger_mode must be fixed or event_limited.")
        rate = float(self.nominal_hz)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("nominal_hz must be finite and positive.")
        channels = tuple(str(value).strip() for value in self.required_context_channels)
        if any(not value for value in channels) or len(channels) != len(set(channels)):
            raise ValueError("required_context_channels must be non-empty and unique.")
        object.__setattr__(self, "nominal_hz", rate)
        object.__setattr__(self, "required_context_channels", channels)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "node_id": self.node_id,
                "output_kind": self.output_kind,
                "trigger_mode": self.trigger_mode,
                "nominal_hz": self.nominal_hz,
                "observation_contract_hash": self.observation_contract_hash,
                "output_contract_hash": self.output_contract_hash,
                "required_context_channels": self.required_context_channels,
            },
            domain=POLICY_NODE_SPEC_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class PolicyNodeRequest:
    request_id: str
    schedule_tick_id: str
    trigger_reason: str
    scheduled_at: float
    target_effective_at: float
    observation: ObservationBundle
    context_snapshot: PolicyContextSnapshot | None = None

    def __post_init__(self) -> None:
        for name in ("request_id", "schedule_tick_id", "trigger_reason"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        for name in ("scheduled_at", "target_effective_at"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite.")


@dataclass(frozen=True)
class PolicyNodeOutput:
    node_id: str
    output_kind: str
    produced_at: float
    policy_context: PolicyContextDraft | None = None
    action_proposal: ActionProposalDraft | None = None
    composition_input: CompositionInputDraft | None = None
    no_output_reason: str | None = None

    def __post_init__(self) -> None:
        if not str(self.node_id).strip():
            raise ValueError("node_id must be non-empty.")
        if self.output_kind not in POLICY_NODE_OUTPUT_KINDS_V1:
            raise ValueError("Unsupported policy node output kind.")
        if not math.isfinite(float(self.produced_at)):
            raise ValueError("produced_at must be finite.")
        payloads = {
            "policy_context": self.policy_context,
            "action_proposal": self.action_proposal,
            "composition_input": self.composition_input,
        }
        present = tuple(name for name, value in payloads.items() if value is not None)
        if self.output_kind == "no_output":
            if present or not str(self.no_output_reason or "").strip():
                raise ValueError("no_output requires only a non-empty reason.")
        elif present != (self.output_kind,) or self.no_output_reason is not None:
            raise ValueError("PolicyNodeOutput kind and typed payload must match exactly.")


@runtime_checkable
class PolicyNodeRuntime(Protocol):
    """One logical node; it owns no authority, dispatcher, or backend."""

    @property
    def spec(self) -> PolicyNodeSpec:
        ...

    def run(self, request: PolicyNodeRequest) -> PolicyNodeOutput:
        ...


__all__ = [
    "POLICY_NODE_OUTPUT_KINDS_V1",
    "POLICY_NODE_SPEC_HASH_DOMAIN_V1",
    "PolicyNodeOutput",
    "PolicyNodeRequest",
    "PolicyNodeRuntime",
    "PolicyNodeSpec",
]
