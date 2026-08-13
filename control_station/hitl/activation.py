"""Backend-neutral evidence for source priming after an activation barrier."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash


HITL_ACTIVATION_PRIME_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.control_station.hitl.activation_prime.development"
)


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _optional_identity(value: Any, *names: str) -> str | None:
    if value is None:
        return None
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return None


@dataclass(frozen=True)
class HITLActivationPrimeEvidence:
    """Source identities that became usable after one activation barrier.

    This is readiness evidence only.  It does not grant authority or bypass
    the lower execution gate.  ``policy_reset_performed`` specifically means
    the policy adapter's activation-reset seam was called before the recorded
    policy emission was requested.
    """

    activation_id: str
    activation_hash: str
    barrier_at: float
    primed_at: float
    clock_domain_id: str
    policy_required: bool
    human_required: bool
    policy_reset_performed: bool
    policy_reset_at: float | None = None
    policy_emission_id: str | None = None
    policy_emission_hash: str | None = None
    policy_emitted_at: float | None = None
    human_emission_id: str | None = None
    human_emission_hash: str | None = None
    human_emitted_at: float | None = None
    policy_submission_id: str | None = None
    policy_submission_hash: str | None = None
    human_submission_id: str | None = None
    human_submission_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("activation_id", "activation_hash", "clock_domain_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        barrier_at = _finite(self.barrier_at, name="barrier_at")
        primed_at = _finite(self.primed_at, name="primed_at")
        if primed_at < barrier_at:
            raise ValueError("Activation priming cannot predate its barrier.")
        object.__setattr__(self, "barrier_at", barrier_at)
        object.__setattr__(self, "primed_at", primed_at)
        for name in (
            "policy_required",
            "human_required",
            "policy_reset_performed",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        for name in ("policy_reset_at", "policy_emitted_at", "human_emitted_at"):
            value = getattr(self, name)
            if value is not None:
                normalized = _finite(value, name=name)
                if normalized < barrier_at:
                    raise ValueError(f"{name} must follow the activation barrier.")
                object.__setattr__(self, name, normalized)
        for prefix in (
            "policy_emission",
            "human_emission",
            "policy_submission",
            "human_submission",
        ):
            identifier = getattr(self, f"{prefix}_id")
            content_hash = getattr(self, f"{prefix}_hash")
            if (identifier is None) != (content_hash is None):
                raise ValueError(f"{prefix} ID/hash must appear together.")
            if identifier is not None:
                object.__setattr__(self, f"{prefix}_id", str(identifier).strip())
                object.__setattr__(self, f"{prefix}_hash", str(content_hash).strip())
        if self.policy_required:
            if not self.policy_reset_performed or self.policy_reset_at is None:
                raise ValueError("Required policy priming must reset the policy source.")
            if self.policy_emission_id is None:
                raise ValueError("Required policy priming needs a fresh emission.")
            if (
                self.policy_emitted_at is not None
                and self.policy_emitted_at < self.policy_reset_at
            ):
                raise ValueError("Policy emission predates its activation reset.")
        if self.human_required and self.human_emission_id is None:
            raise ValueError("Required human priming needs a fresh emission.")
        payload = self.to_payload(include_hash=False)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=HITL_ACTIVATION_PRIME_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    def target_emission_identity(self, source: str) -> tuple[str, str]:
        normalized = str(source).strip()
        if normalized == "policy":
            values = (self.policy_emission_id, self.policy_emission_hash)
        elif normalized == "human":
            values = (self.human_emission_id, self.human_emission_hash)
        else:
            raise ValueError("source must be policy or human.")
        if values[0] is None or values[1] is None:
            raise RuntimeError(f"Activation has no fresh {normalized} emission.")
        return str(values[0]), str(values[1])

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "semantic_content_hash"
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


def build_hitl_activation_prime_evidence(
    *,
    activation_scope: Any,
    primed_at: float,
    clock_domain_id: str,
    policy_required: bool,
    human_required: bool,
    policy_reset_performed: bool,
    policy_reset_at: float | None,
    policy_emission: Any | None,
    human_emission: Any | None,
    policy_submission: Any | None,
    human_submission: Any | None,
    strict_source_identity: bool = True,
) -> HITLActivationPrimeEvidence | None:
    """Build evidence without depending on one source/action representation.

    Legacy injected test/profile sources may expose only a local emission ID.
    They can request ``strict_source_identity=False`` and receive ``None``
    instead of a partial fact.  The exclusive-HITL profile always keeps strict
    mode because source switching cannot complete without exact fresh-emission
    identity and hash evidence.
    """

    activation_id = _optional_identity(activation_scope, "activation_id")
    activation_hash = _optional_identity(activation_scope, "semantic_content_hash")
    barrier_at = getattr(activation_scope, "input_barrier_at", None)
    if activation_id is None or activation_hash is None or barrier_at is None:
        raise TypeError("activation_scope must expose identity/hash/input barrier.")
    values = {
        "policy_emission_id": _optional_identity(policy_emission, "emission_id"),
        "policy_emission_hash": _optional_identity(
            policy_emission,
            "semantic_content_hash",
        ),
        # HumanCommandEmission intentionally has a source-local sequence rather
        # than an ``emission_id`` field. The adapter submission exposes the
        # exact source identity admitted by composition, so prefer that generic
        # seam and retain direct emission IDs for alternative source types.
        "human_emission_id": (
            _optional_identity(human_submission, "source_emission_id")
            or _optional_identity(human_emission, "emission_id")
        ),
        "human_emission_hash": _optional_identity(
            human_emission,
            "semantic_content_hash",
        ),
        "policy_submission_id": _optional_identity(
            policy_submission,
            "submission_id",
            "receipt_id",
            "composition_input_id",
            "source_emission_id",
        ),
        "policy_submission_hash": _optional_identity(
            policy_submission,
            "semantic_content_hash",
        ),
        "human_submission_id": _optional_identity(
            human_submission,
            "submission_id",
            "receipt_id",
            "composition_input_id",
            "source_emission_id",
        ),
        "human_submission_hash": _optional_identity(
            human_submission,
            "semantic_content_hash",
        ),
    }
    incomplete_pair = any(
        (values[f"{prefix}_id"] is None)
        != (values[f"{prefix}_hash"] is None)
        for prefix in (
            "policy_emission",
            "human_emission",
            "policy_submission",
            "human_submission",
        )
    )
    missing_required = bool(
        (policy_required and values["policy_emission_id"] is None)
        or (human_required and values["human_emission_id"] is None)
    )
    if (incomplete_pair or missing_required) and not strict_source_identity:
        return None
    return HITLActivationPrimeEvidence(
        activation_id=activation_id,
        activation_hash=activation_hash,
        barrier_at=float(barrier_at),
        primed_at=primed_at,
        clock_domain_id=clock_domain_id,
        policy_required=policy_required,
        human_required=human_required,
        policy_reset_performed=policy_reset_performed,
        policy_reset_at=policy_reset_at,
        policy_emission_id=values["policy_emission_id"],
        policy_emission_hash=values["policy_emission_hash"],
        policy_emitted_at=(
            None
            if policy_emission is None
            else getattr(policy_emission, "emitted_at", None)
        ),
        human_emission_id=values["human_emission_id"],
        human_emission_hash=values["human_emission_hash"],
        human_emitted_at=(
            None
            if human_emission is None
            else getattr(human_emission, "emitted_at", None)
        ),
        policy_submission_id=values["policy_submission_id"],
        policy_submission_hash=values["policy_submission_hash"],
        human_submission_id=values["human_submission_id"],
        human_submission_hash=values["human_submission_hash"],
    )


__all__ = [
    "HITL_ACTIVATION_PRIME_HASH_DOMAIN_DEVELOPMENT",
    "HITLActivationPrimeEvidence",
    "build_hitl_activation_prime_evidence",
]
