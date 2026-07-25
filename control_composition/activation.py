"""Activation-local identity for formal control composition.

An activation is narrower than a runtime session and independent of producer,
authority, and execution epochs.  It binds contributor inputs to one exact
start/rearm instance without changing the numerical ``CompositionSpec``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dataset_contracts.canonical_encoding import canonical_content_hash


COMPOSITION_ACTIVATION_SCOPE_HASH_DOMAIN_V2 = (
    "dexteleop.composition_activation_scope.v2"
)
COMPOSITION_ACTIVATION_ID_HASH_DOMAIN_V2 = (
    "dexteleop.composition_activation_id.v2"
)
ACTIVATION_LOCAL_INPUT_FRESHNESS_POLICY_V2 = (
    "exact_activation_receiver_barrier_v2"
)
ACTIVATION_LOCAL_STRICT_SOURCE_FRESHNESS_POLICY_V2 = (
    "exact_activation_strict_source_barrier_v2"
)
COMPOSITION_ACTIVATION_FRESHNESS_POLICIES_V2 = frozenset(
    {
        ACTIVATION_LOCAL_INPUT_FRESHNESS_POLICY_V2,
        ACTIVATION_LOCAL_STRICT_SOURCE_FRESHNESS_POLICY_V2,
    }
)


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class CompositionActivationScope:
    activation_id: str
    operation_spec_id: str
    operation_spec_hash: str
    session_epoch: int
    activated_at: float
    clock_domain_id: str
    input_barrier_at: float
    freshness_policy: str
    required_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "activation_id",
            "operation_spec_id",
            "operation_spec_hash",
            "clock_domain_id",
            "freshness_policy",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        epoch = int(self.session_epoch)
        if (
            isinstance(self.session_epoch, bool)
            or epoch != self.session_epoch
            or epoch < 0
        ):
            raise ValueError("session_epoch must be a non-negative integer.")
        object.__setattr__(self, "session_epoch", epoch)
        for field_name in ("activated_at", "input_barrier_at"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite.")
            object.__setattr__(self, field_name, value)
        if self.input_barrier_at > self.activated_at:
            raise ValueError("input_barrier_at cannot follow activated_at.")
        if self.freshness_policy not in (
            COMPOSITION_ACTIVATION_FRESHNESS_POLICIES_V2
        ):
            raise ValueError("Unsupported composition activation freshness policy.")
        roles = tuple(
            _required(value, field_name="required_role")
            for value in self.required_roles
        )
        if not roles or len(roles) != len(set(roles)):
            raise ValueError("required_roles must be non-empty and unique.")
        object.__setattr__(self, "required_roles", roles)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "activation_id": self.activation_id,
                "operation_spec_id": self.operation_spec_id,
                "operation_spec_hash": self.operation_spec_hash,
                "session_epoch": self.session_epoch,
                "activated_at": self.activated_at,
                "clock_domain_id": self.clock_domain_id,
                "input_barrier_at": self.input_barrier_at,
                "freshness_policy": self.freshness_policy,
                "required_roles": self.required_roles,
            },
            domain=COMPOSITION_ACTIVATION_SCOPE_HASH_DOMAIN_V2,
        )


@dataclass(frozen=True)
class CompositionActivationInputVerdict:
    """Canonical freshness decision for one input against one activation."""

    accepted: bool
    rejection_code: str | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if self.accepted:
            if self.rejection_code is not None or self.message:
                raise ValueError(
                    "Accepted activation input verdict cannot carry rejection details."
                )
            return
        if self.rejection_code is None:
            raise ValueError(
                "Rejected activation input verdict requires a rejection_code."
            )
        object.__setattr__(
            self,
            "rejection_code",
            _required(self.rejection_code, field_name="rejection_code"),
        )
        object.__setattr__(
            self,
            "message",
            _required(self.message, field_name="message"),
        )


def _reject_input(
    rejection_code: str,
    message: str,
) -> CompositionActivationInputVerdict:
    return CompositionActivationInputVerdict(
        accepted=False,
        rejection_code=rejection_code,
        message=message,
    )


def evaluate_composition_input_activation(
    scope: CompositionActivationScope,
    *,
    composition_activation_id: str | None,
    composition_activation_hash: str | None,
    produced_at: float,
    produced_clock_domain_id: str,
    available_at: float,
    available_clock_domain_id: str,
    effective_from: float,
    effective_clock_domain_id: str,
) -> CompositionActivationInputVerdict:
    """Evaluate activation identity and freshness without owning admission.

    Receiver freshness is defined on the receiver clock by ``available_at`` and
    ``effective_from``.  Strict source freshness additionally compares
    ``produced_at`` only when the producer declares that exact same clock
    domain; no implicit cross-clock conversion is permitted.
    """

    if not isinstance(scope, CompositionActivationScope):
        raise TypeError("scope must be a CompositionActivationScope.")
    produced = float(produced_at)
    available = float(available_at)
    effective = float(effective_from)
    for field_name, value in (
        ("produced_at", produced),
        ("available_at", available),
        ("effective_from", effective),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite.")
    produced_clock = _required(
        produced_clock_domain_id,
        field_name="produced_clock_domain_id",
    )
    available_clock = _required(
        available_clock_domain_id,
        field_name="available_clock_domain_id",
    )
    effective_clock = _required(
        effective_clock_domain_id,
        field_name="effective_clock_domain_id",
    )
    if composition_activation_id != scope.activation_id:
        return _reject_input(
            "activation_identity_mismatch",
            "Composition input is not bound to the current activation id.",
        )
    if composition_activation_hash != scope.semantic_content_hash:
        return _reject_input(
            "activation_hash_mismatch",
            "Composition input is not bound to the current activation hash.",
        )
    if available_clock != scope.clock_domain_id:
        return _reject_input(
            "activation_receiver_clock_mismatch",
            "Composition input admission uses another receiver clock domain.",
        )
    if effective_clock != scope.clock_domain_id:
        return _reject_input(
            "activation_effective_clock_mismatch",
            "Composition input validity uses another activation clock domain.",
        )
    if available < scope.input_barrier_at:
        return _reject_input(
            "activation_receiver_barrier_violation",
            "Composition input became available before the activation barrier.",
        )
    if effective < scope.input_barrier_at:
        return _reject_input(
            "activation_effective_barrier_violation",
            "Composition input becomes effective before the activation barrier.",
        )
    if (
        scope.freshness_policy
        == ACTIVATION_LOCAL_STRICT_SOURCE_FRESHNESS_POLICY_V2
    ):
        if produced_clock != scope.clock_domain_id:
            return _reject_input(
                "activation_source_clock_mismatch",
                "Strict source freshness requires the activation clock domain.",
            )
        if produced <= scope.input_barrier_at:
            return _reject_input(
                "activation_source_barrier_violation",
                "Composition input was not produced after the activation barrier.",
            )
    return CompositionActivationInputVerdict(accepted=True)


def build_composition_activation_scope(
    *,
    operation_spec_id: str,
    operation_spec_hash: str,
    session_epoch: int,
    activated_at: float,
    clock_domain_id: str,
    input_barrier_at: float,
    required_roles: tuple[str, ...],
    freshness_policy: str = ACTIVATION_LOCAL_INPUT_FRESHNESS_POLICY_V2,
) -> CompositionActivationScope:
    """Create one deterministic activation identity from its immutable binding."""

    normalized_epoch = int(session_epoch)
    if (
        isinstance(session_epoch, bool)
        or normalized_epoch != session_epoch
        or normalized_epoch < 0
    ):
        raise ValueError("session_epoch must be a non-negative integer.")
    normalized_activated_at = float(activated_at)
    normalized_input_barrier_at = float(input_barrier_at)
    if not math.isfinite(normalized_activated_at) or not math.isfinite(
        normalized_input_barrier_at
    ):
        raise ValueError("Composition activation times must be finite.")
    normalized_roles = tuple(
        _required(value, field_name="required_role")
        for value in required_roles
    )
    identity_payload = {
        "operation_spec_id": _required(
            operation_spec_id,
            field_name="operation_spec_id",
        ),
        "operation_spec_hash": _required(
            operation_spec_hash,
            field_name="operation_spec_hash",
        ),
        "session_epoch": normalized_epoch,
        "activated_at": normalized_activated_at,
        "clock_domain_id": _required(
            clock_domain_id,
            field_name="clock_domain_id",
        ),
        "input_barrier_at": normalized_input_barrier_at,
        "freshness_policy": _required(
            freshness_policy,
            field_name="freshness_policy",
        ),
        "required_roles": normalized_roles,
    }
    digest = canonical_content_hash(
        identity_payload,
        domain=COMPOSITION_ACTIVATION_ID_HASH_DOMAIN_V2,
    )
    return CompositionActivationScope(
        activation_id=f"composition-activation:{digest}",
        **identity_payload,
    )


__all__ = [
    "ACTIVATION_LOCAL_INPUT_FRESHNESS_POLICY_V2",
    "ACTIVATION_LOCAL_STRICT_SOURCE_FRESHNESS_POLICY_V2",
    "COMPOSITION_ACTIVATION_FRESHNESS_POLICIES_V2",
    "COMPOSITION_ACTIVATION_ID_HASH_DOMAIN_V2",
    "COMPOSITION_ACTIVATION_SCOPE_HASH_DOMAIN_V2",
    "CompositionActivationInputVerdict",
    "CompositionActivationScope",
    "build_composition_activation_scope",
    "evaluate_composition_input_activation",
]
