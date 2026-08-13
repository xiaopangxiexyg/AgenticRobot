"""Pure results for passive policy/human composition shadow binding."""

from __future__ import annotations

import math
from dataclasses import dataclass

from control_composition.composition import CompositionInputAdmissionReceipt
from control_composition.runtime import CompositeProposalSubmission


POLICY_HUMAN_COMPOSITION_SHADOW_PROFILE_OFF = "off"


def normalize_finite_float(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def normalize_required_text(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class PolicyHumanCompositionShadowProfile:
    """Default-off application switch for a no-command shadow session."""

    profile_id: str = POLICY_HUMAN_COMPOSITION_SHADOW_PROFILE_OFF
    enabled: bool = False
    send_actions: bool = False

    def __post_init__(self) -> None:
        profile_id = normalize_required_text(self.profile_id, name="profile_id")
        if not isinstance(self.enabled, bool) or not isinstance(
            self.send_actions,
            bool,
        ):
            raise TypeError("enabled and send_actions must be booleans.")
        if self.send_actions:
            raise ValueError("Composition shadow never supports send_actions=true.")
        if self.enabled and profile_id == POLICY_HUMAN_COMPOSITION_SHADOW_PROFILE_OFF:
            raise ValueError("An enabled composition shadow requires a profile id.")
        object.__setattr__(self, "profile_id", profile_id)


@dataclass(frozen=True)
class PassiveCompositionIngressStatus:
    """Bounded observer status; evictions are visible, never backpressure."""

    source_kind: str
    queue_size: int
    queue_capacity: int
    offered_count: int
    accepted_count: int
    evicted_count: int
    rejected_count: int
    drained_count: int
    ignored_while_disabled_count: int


@dataclass(frozen=True)
class CompositionShadowInputSubmission:
    """Outcome of adapting/admitting one passive source emission."""

    source_kind: str
    input_role: str
    source_emission_id: str
    observed_at: float
    input_receipt: CompositionInputAdmissionReceipt | None = None
    source_fact_ids: tuple[str, ...] = ()
    trace_failures: tuple[str, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_kind",
            normalize_required_text(self.source_kind, name="source_kind"),
        )
        object.__setattr__(
            self,
            "input_role",
            normalize_required_text(self.input_role, name="input_role"),
        )
        object.__setattr__(
            self,
            "source_emission_id",
            normalize_required_text(
                self.source_emission_id,
                name="source_emission_id",
            ),
        )
        object.__setattr__(
            self,
            "observed_at",
            normalize_finite_float(self.observed_at, name="observed_at"),
        )
        if self.input_receipt is not None and not isinstance(
            self.input_receipt,
            CompositionInputAdmissionReceipt,
        ):
            raise TypeError(
                "input_receipt must be CompositionInputAdmissionReceipt or None."
            )
        if self.error is None and (
            self.input_receipt is None or not self.input_receipt.accepted
        ):
            raise ValueError("A successful shadow submission needs an accepted input.")

    @property
    def admitted(self) -> bool:
        return bool(self.input_receipt is not None and self.input_receipt.accepted)


@dataclass(frozen=True)
class PolicyHumanCompositionShadowCycle:
    """One non-promotable shadow cut and its passive ingress outcomes."""

    profile_id: str
    status: str
    target_effective_at: float | None = None
    submissions: tuple[CompositionShadowInputSubmission, ...] = ()
    evaluation: CompositeProposalSubmission | None = None
    policy_ingress: PassiveCompositionIngressStatus | None = None
    human_ingress: PassiveCompositionIngressStatus | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        profile_id = normalize_required_text(self.profile_id, name="profile_id")
        if self.status not in {"disabled", "evaluated", "evaluation_error"}:
            raise ValueError("Unsupported composition shadow cycle status.")
        if self.status == "disabled":
            if any(
                value is not None
                for value in (
                    self.target_effective_at,
                    self.evaluation,
                    self.error,
                )
            ) or self.submissions:
                raise ValueError("Disabled shadow cycle cannot carry evaluation data.")
        else:
            if self.target_effective_at is None:
                raise ValueError("Evaluated shadow cycle requires a target time.")
            normalize_finite_float(
                self.target_effective_at,
                name="target_effective_at",
            )
        if self.status == "evaluated":
            if not isinstance(self.evaluation, CompositeProposalSubmission):
                raise TypeError("Evaluated cycle requires a typed evaluation.")
            if self.evaluation.proposal_admission is not None:
                raise ValueError("Shadow evaluation cannot admit a proposal.")
            if self.error is not None:
                raise ValueError("Evaluated cycle cannot carry an error.")
        elif self.status == "evaluation_error":
            if self.evaluation is not None or not str(self.error or "").strip():
                raise ValueError("Evaluation error cycle requires only an error.")
        object.__setattr__(self, "profile_id", profile_id)

    @property
    def conditional_action(self):
        """Non-authoritative composed candidate for visualization only."""

        if self.evaluation is None:
            return None
        return self.evaluation.composition_evaluation.composed_candidate

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def proposal_admitted(self) -> bool:
        return False

    @property
    def send_actions(self) -> bool:
        return False


__all__ = [
    "POLICY_HUMAN_COMPOSITION_SHADOW_PROFILE_OFF",
    "CompositionShadowInputSubmission",
    "PassiveCompositionIngressStatus",
    "PolicyHumanCompositionShadowCycle",
    "PolicyHumanCompositionShadowProfile",
    "normalize_finite_float",
    "normalize_required_text",
]
