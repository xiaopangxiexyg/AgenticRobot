"""In-process dispatch ownership and typed dispatch-permit contracts.

M1D deliberately stops before any Rig or SDK integration.  Authorization is
linearized when :meth:`DispatchOwnerRegistry.authorize` consumes a one-shot
permit; a later integration must place that call immediately before its
backend invocation.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable

from dataset_contracts.canonical_encoding import canonical_content_hash

from .cancellation import (
    ExecutionCancellation,
    ExecutionCancellationReceipt,
    ExecutionRecovery,
)
from .identity import RuntimeFactIdentity


DISPATCH_OWNER_TOKEN_HASH_DOMAIN_V1 = "dexteleop.dispatch_owner_token.v1"
DISPATCH_OWNER_TRANSITION_HASH_DOMAIN_V1 = "dexteleop.dispatch_owner_transition.v1"
DISPATCH_PERMIT_HASH_DOMAIN_V1 = "dexteleop.dispatch_permit.v1"
DISPATCH_VALIDATION_HASH_DOMAIN_V1 = "dexteleop.dispatch_validation_receipt.v1"

DISPATCH_OWNER_KINDS = (
    "legacy_control_path",
    "shared_arbitration_runtime",
)
DISPATCH_PERMIT_KINDS = (
    "motion",
    "safety_control",
    "lifecycle_control",
)
SAFETY_CONTROL_OPERATIONS = (
    "stop",
    "hold",
    "disable_motion",
    "collision_stop",
    "cancellation_request",
    "rm65_quick_stop_tick",
)
SAFETY_CONTROL_SOURCES = (
    "execution_cancellation_runtime",
    "shared_safety_guard",
    "runtime_supervisor",
)
LIFECYCLE_CONTROL_OPERATIONS = (
    "startup",
    "home",
    "enable",
    "rearm",
)
OWNER_TRANSITION_KINDS = ("authority_handoff", "owner_rollback")
TRANSITION_GATE_STATUSES = ("hold", "transitioning")


def _identifier(value: str, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _finite_time(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


@dataclass(frozen=True)
class DispatchOwnerToken:
    token_id: str
    runtime_session_id: str
    control_domain_id: str
    owner_runtime_id: str
    owner_kind: str
    owner_generation: int
    authority_epoch_at_acquire: int
    execution_epoch_at_acquire: int
    acquired_at: float
    revoked_at: float | None
    process_scope: str
    semantic_content_hash: str


@dataclass(frozen=True)
class DispatchOwnerTransition:
    transition_id: str
    runtime_session_id: str
    control_domain_id: str
    transition_kind: str
    gate_status: str
    gate_epoch: int
    cancellation_id: str
    previous_authority_epoch: int
    target_execution_epoch: int
    revoked_token: DispatchOwnerToken
    started_at: float
    semantic_content_hash: str


@dataclass(frozen=True)
class DispatchOwnerSnapshot:
    runtime_session_id: str
    control_domain_id: str
    state: str
    current_token: DispatchOwnerToken | None
    current_authority_epoch: int
    current_execution_epoch: int
    pending_transition_id: str | None
    process_scope: str = "in_process_only"


@dataclass(frozen=True)
class MotionDispatchPermit:
    permit_id: str
    runtime_session_id: str
    control_domain_id: str
    permit_kind: str
    token_id: str
    owner_generation: int
    authority_epoch: int
    execution_epoch: int
    command_lineage_id: str
    directive_lineage_id: str
    issued_at: float
    semantic_content_hash: str


@dataclass(frozen=True)
class AdmittedSafetyControlEvidence:
    event_id: str
    runtime_session_id: str
    control_domain_id: str
    execution_epoch: int
    evidence_kind: str
    safety_source: str
    control_admission_sequence: int

    def __post_init__(self) -> None:
        if self.evidence_kind not in (
            "safety_event",
            "execution_cancellation",
            "safety_operation_tick",
        ):
            raise ValueError("Unsupported safety-control evidence kind.")
        if self.safety_source not in SAFETY_CONTROL_SOURCES:
            raise ValueError("Unsupported safety-control source.")
        _non_negative_int(self.execution_epoch, name="execution_epoch")
        if int(self.control_admission_sequence) <= 0:
            raise ValueError("Safety evidence must have a positive admission sequence.")


@dataclass(frozen=True)
class SafetyControlDispatchPermit:
    permit_id: str
    runtime_session_id: str
    control_domain_id: str
    permit_kind: str
    operation: str
    evidence_event_id: str
    evidence_kind: str
    safety_source: str
    target_execution_epoch: int
    issued_at: float
    semantic_content_hash: str


@dataclass(frozen=True)
class LifecycleControlAuthorization:
    authorization_event_id: str
    runtime_session_id: str
    control_domain_id: str
    lifecycle_state_machine_id: str
    requester_kind: str
    operation: str
    target_execution_epoch: int
    operator_or_safety_policy_id: str


@dataclass(frozen=True)
class LifecycleControlDispatchPermit:
    permit_id: str
    runtime_session_id: str
    control_domain_id: str
    permit_kind: str
    operation: str
    lifecycle_state_machine_id: str
    authorization_event_id: str
    target_execution_epoch: int
    operator_or_safety_policy_id: str
    issued_at: float
    semantic_content_hash: str


DispatchPermit = (
    MotionDispatchPermit
    | SafetyControlDispatchPermit
    | LifecycleControlDispatchPermit
)


@dataclass(frozen=True)
class DispatchPermitValidationReceipt:
    receipt_id: str
    permit_id: str
    permit_kind: str
    control_domain_id: str
    checked_at: float
    accepted: bool
    status: str
    current_token_id: str | None
    current_owner_generation: int | None
    current_authority_epoch: int
    current_execution_epoch: int
    semantic_content_hash: str


class DispatchOwnerRegistry:
    """Atomic in-process owner registry and one-shot permit authority."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        control_domain_id: str,
        initial_authority_epoch: int,
        initial_execution_epoch: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runtime_session_id = _identifier(
            runtime_session_id,
            name="runtime_session_id",
        )
        self.control_domain_id = _identifier(
            control_domain_id,
            name="control_domain_id",
        )
        self._authority_epoch = _non_negative_int(
            initial_authority_epoch,
            name="initial_authority_epoch",
        )
        self._execution_epoch = _non_negative_int(
            initial_execution_epoch,
            name="initial_execution_epoch",
        )
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self._clock = clock
        self._current_token: DispatchOwnerToken | None = None
        self._pending_transition: DispatchOwnerTransition | None = None
        self._owner_generation = 0
        self._next_fact_sequence = 0
        self._ever_granted = False
        self._issued_permit_hash_by_id: dict[str, str] = {}
        self._consumed_permit_ids: set[str] = set()
        self._lock = threading.RLock()

    def snapshot(self) -> DispatchOwnerSnapshot:
        with self._lock:
            if self._pending_transition is not None:
                state = "transitioning_hold"
            elif self._current_token is None:
                state = "unowned"
            else:
                state = "active"
            return DispatchOwnerSnapshot(
                runtime_session_id=self.runtime_session_id,
                control_domain_id=self.control_domain_id,
                state=state,
                current_token=self._current_token,
                current_authority_epoch=self._authority_epoch,
                current_execution_epoch=self._execution_epoch,
                pending_transition_id=(
                    None
                    if self._pending_transition is None
                    else self._pending_transition.transition_id
                ),
            )

    def grant_initial_owner(
        self,
        *,
        owner_runtime_id: str,
        owner_kind: str,
    ) -> DispatchOwnerToken:
        with self._lock:
            if self._ever_granted or self._current_token is not None:
                raise RuntimeError("Initial dispatch owner has already been granted.")
            return self._grant_owner_locked(
                owner_runtime_id=owner_runtime_id,
                owner_kind=owner_kind,
            )

    def observe_authority_epoch(
        self,
        *,
        authority_epoch: int,
        evidence_event_id: str,
    ) -> DispatchOwnerSnapshot:
        """Advance authority evidence without changing dispatch ownership."""

        normalized = _non_negative_int(authority_epoch, name="authority_epoch")
        _identifier(evidence_event_id, name="evidence_event_id")
        with self._lock:
            if self._pending_transition is not None:
                raise RuntimeError("Authority is finalized by complete_transition while switching owner.")
            if normalized <= self._authority_epoch:
                raise ValueError("Observed authority epoch must strictly advance.")
            self._authority_epoch = normalized
            return self.snapshot()

    def observe_execution_cancellation(
        self,
        cancellation: ExecutionCancellation,
    ) -> DispatchOwnerSnapshot:
        """Apply a non-owner-transition cancellation to the live permit gate."""

        if not isinstance(cancellation, ExecutionCancellation):
            raise TypeError("cancellation must be an ExecutionCancellation.")
        if cancellation.reason in OWNER_TRANSITION_KINDS:
            raise ValueError("Owner handoff/rollback cancellation must use begin_transition.")
        with self._lock:
            self._validate_cancellation_epoch_locked(cancellation)
            self._execution_epoch = cancellation.execution_epoch
            return self.snapshot()

    def observe_execution_recovery(
        self,
        recovery: ExecutionRecovery,
    ) -> DispatchOwnerSnapshot:
        if not isinstance(recovery, ExecutionRecovery):
            raise TypeError("recovery must be an ExecutionRecovery.")
        with self._lock:
            if recovery.control_domain_id != self.control_domain_id:
                raise ValueError("Recovery belongs to another control domain.")
            if recovery.runtime_session_id != self.runtime_session_id:
                raise ValueError("Recovery belongs to another runtime session.")
            if recovery.previous_execution_epoch != self._execution_epoch:
                raise ValueError("Recovery does not start from the active execution epoch.")
            if recovery.execution_epoch != self._execution_epoch + 1:
                raise ValueError("Recovery must advance one execution epoch.")
            self._execution_epoch = recovery.execution_epoch
            return self.snapshot()

    def begin_transition(
        self,
        *,
        current_token: DispatchOwnerToken,
        transition_kind: str,
        cancellation: ExecutionCancellation,
        gate_status: str,
        gate_epoch: int,
    ) -> DispatchOwnerTransition:
        kind = str(transition_kind).strip()
        status = str(gate_status).strip().lower()
        if kind not in OWNER_TRANSITION_KINDS:
            raise ValueError("Unsupported owner transition kind.")
        if status not in TRANSITION_GATE_STATUSES:
            raise ValueError("Owner transition must begin under HOLD/TRANSITIONING.")
        if not isinstance(cancellation, ExecutionCancellation):
            raise TypeError("cancellation must be an ExecutionCancellation.")
        expected_reason = kind
        if cancellation.reason != expected_reason:
            raise ValueError("Owner transition cancellation reason does not match transition kind.")

        with self._lock:
            self._require_current_token_locked(current_token)
            if self._pending_transition is not None:
                raise RuntimeError("An owner transition is already pending.")
            self._validate_cancellation_epoch_locked(cancellation)

            started_at = _finite_time(self._clock(), name="started_at")
            revoked_payload = {
                name: getattr(current_token, name)
                for name in current_token.__dataclass_fields__
                if name != "semantic_content_hash"
            }
            revoked_payload["revoked_at"] = started_at
            revoked_token = replace(
                current_token,
                revoked_at=started_at,
                semantic_content_hash=canonical_content_hash(
                    revoked_payload,
                    domain=DISPATCH_OWNER_TOKEN_HASH_DOMAIN_V1,
                ),
            )
            sequence = self._take_sequence_locked()
            identity = self._fact_identity(sequence, namespace="dispatch_owner_transition")
            payload = {
                "transition_id": identity.fact_id,
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "transition_kind": kind,
                "gate_status": status,
                "gate_epoch": _non_negative_int(gate_epoch, name="gate_epoch"),
                "cancellation_id": cancellation.cancellation_id,
                "previous_authority_epoch": self._authority_epoch,
                "target_execution_epoch": cancellation.execution_epoch,
                "revoked_token": revoked_token,
                "started_at": started_at,
            }
            semantic_payload = {
                **payload,
                "revoked_token": revoked_token.semantic_content_hash,
            }
            transition = DispatchOwnerTransition(
                **payload,
                semantic_content_hash=canonical_content_hash(
                    semantic_payload,
                    domain=DISPATCH_OWNER_TRANSITION_HASH_DOMAIN_V1,
                ),
            )
            # Revoke and enter the ownerless HOLD interval atomically.
            self._current_token = None
            self._execution_epoch = cancellation.execution_epoch
            self._pending_transition = transition
            return transition

    def complete_transition(
        self,
        *,
        transition_id: str,
        cancellation_receipt: ExecutionCancellationReceipt,
        owner_runtime_id: str,
        owner_kind: str,
        new_authority_epoch: int,
        gate_status: str,
    ) -> DispatchOwnerToken:
        status = str(gate_status).strip().lower()
        if status not in TRANSITION_GATE_STATUSES:
            raise ValueError("Owner grant must complete while the gate remains HOLD/TRANSITIONING.")
        if not isinstance(cancellation_receipt, ExecutionCancellationReceipt):
            raise TypeError("cancellation_receipt must be an ExecutionCancellationReceipt.")
        normalized_owner_runtime_id = _identifier(
            owner_runtime_id,
            name="owner_runtime_id",
        )
        normalized_owner_kind = str(owner_kind).strip()
        if normalized_owner_kind not in DISPATCH_OWNER_KINDS:
            raise ValueError("Dispatch owner must be a legacy or shared runtime.")
        with self._lock:
            transition = self._pending_transition
            if transition is None or transition.transition_id != transition_id:
                raise ValueError("Unknown or inactive owner transition.")
            if cancellation_receipt.cancellation_id != transition.cancellation_id:
                raise ValueError("Cancellation receipt does not belong to this transition.")
            if not cancellation_receipt.complete:
                raise RuntimeError("Owner transition requires complete cancellation acknowledgement.")
            authority_epoch = _non_negative_int(
                new_authority_epoch,
                name="new_authority_epoch",
            )
            if authority_epoch <= transition.previous_authority_epoch:
                raise ValueError("Owner transition must advance authority epoch.")
            token = self._grant_owner_locked(
                owner_runtime_id=normalized_owner_runtime_id,
                owner_kind=normalized_owner_kind,
                authority_epoch=authority_epoch,
            )
            self._pending_transition = None
            return token

    def issue_motion_permit(
        self,
        *,
        token: DispatchOwnerToken,
        authority_epoch: int,
        execution_epoch: int,
        command_lineage_id: str,
        directive_lineage_id: str,
    ) -> MotionDispatchPermit:
        with self._lock:
            self._require_current_token_locked(token)
            sequence = self._take_sequence_locked()
            identity = self._fact_identity(sequence, namespace="motion_dispatch_permit")
            payload = {
                "permit_id": identity.fact_id,
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "permit_kind": "motion",
                "token_id": token.token_id,
                "owner_generation": token.owner_generation,
                "authority_epoch": _non_negative_int(authority_epoch, name="authority_epoch"),
                "execution_epoch": _non_negative_int(execution_epoch, name="execution_epoch"),
                "command_lineage_id": _identifier(command_lineage_id, name="command_lineage_id"),
                "directive_lineage_id": _identifier(directive_lineage_id, name="directive_lineage_id"),
                "issued_at": _finite_time(self._clock(), name="issued_at"),
            }
            permit = MotionDispatchPermit(
                **payload,
                semantic_content_hash=canonical_content_hash(
                    payload,
                    domain=DISPATCH_PERMIT_HASH_DOMAIN_V1,
                ),
            )
            self._record_permit_locked(permit)
            return permit

    def issue_safety_control_permit(
        self,
        *,
        operation: str,
        evidence: AdmittedSafetyControlEvidence | ExecutionCancellation,
        safety_source: str,
    ) -> SafetyControlDispatchPermit:
        normalized_operation = str(operation).strip()
        normalized_source = str(safety_source).strip()
        if normalized_operation not in SAFETY_CONTROL_OPERATIONS:
            raise ValueError("Unsupported safety-control operation.")
        if normalized_source not in SAFETY_CONTROL_SOURCES:
            raise ValueError("Unsupported safety-control source.")
        if isinstance(evidence, ExecutionCancellation):
            evidence_session = evidence.runtime_session_id
            evidence_domain = evidence.control_domain_id
            target_epoch = evidence.execution_epoch
            evidence_id = evidence.cancellation_id
            evidence_kind = "execution_cancellation"
            if normalized_source != "execution_cancellation_runtime":
                raise ValueError("Cancellation evidence requires the cancellation runtime source.")
        elif isinstance(evidence, AdmittedSafetyControlEvidence):
            evidence_session = evidence.runtime_session_id
            evidence_domain = evidence.control_domain_id
            target_epoch = evidence.execution_epoch
            evidence_id = evidence.event_id
            evidence_kind = evidence.evidence_kind
            if evidence.safety_source != normalized_source:
                raise ValueError("Safety evidence source does not match permit source.")
        else:
            raise TypeError("evidence must be admitted safety or cancellation evidence.")
        if evidence_session != self.runtime_session_id or evidence_domain != self.control_domain_id:
            raise ValueError("Safety evidence belongs to another session/domain.")

        with self._lock:
            sequence = self._take_sequence_locked()
            identity = self._fact_identity(sequence, namespace="safety_dispatch_permit")
            payload = {
                "permit_id": identity.fact_id,
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "permit_kind": "safety_control",
                "operation": normalized_operation,
                "evidence_event_id": evidence_id,
                "evidence_kind": evidence_kind,
                "safety_source": normalized_source,
                "target_execution_epoch": target_epoch,
                "issued_at": _finite_time(self._clock(), name="issued_at"),
            }
            permit = SafetyControlDispatchPermit(
                **payload,
                semantic_content_hash=canonical_content_hash(
                    payload,
                    domain=DISPATCH_PERMIT_HASH_DOMAIN_V1,
                ),
            )
            self._record_permit_locked(permit)
            return permit

    def issue_lifecycle_control_permit(
        self,
        authorization: LifecycleControlAuthorization,
    ) -> LifecycleControlDispatchPermit:
        if not isinstance(authorization, LifecycleControlAuthorization):
            raise TypeError("authorization must be a LifecycleControlAuthorization.")
        if authorization.requester_kind != "lifecycle_state_machine":
            raise ValueError("Only a lifecycle state machine may request lifecycle control.")
        if authorization.operation not in LIFECYCLE_CONTROL_OPERATIONS:
            raise ValueError("Unsupported lifecycle-control operation.")
        if (
            authorization.runtime_session_id != self.runtime_session_id
            or authorization.control_domain_id != self.control_domain_id
        ):
            raise ValueError("Lifecycle authorization belongs to another session/domain.")
        with self._lock:
            sequence = self._take_sequence_locked()
            identity = self._fact_identity(sequence, namespace="lifecycle_dispatch_permit")
            payload = {
                "permit_id": identity.fact_id,
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "permit_kind": "lifecycle_control",
                "operation": authorization.operation,
                "lifecycle_state_machine_id": _identifier(
                    authorization.lifecycle_state_machine_id,
                    name="lifecycle_state_machine_id",
                ),
                "authorization_event_id": _identifier(
                    authorization.authorization_event_id,
                    name="authorization_event_id",
                ),
                "target_execution_epoch": _non_negative_int(
                    authorization.target_execution_epoch,
                    name="target_execution_epoch",
                ),
                "operator_or_safety_policy_id": _identifier(
                    authorization.operator_or_safety_policy_id,
                    name="operator_or_safety_policy_id",
                ),
                "issued_at": _finite_time(self._clock(), name="issued_at"),
            }
            permit = LifecycleControlDispatchPermit(
                **payload,
                semantic_content_hash=canonical_content_hash(
                    payload,
                    domain=DISPATCH_PERMIT_HASH_DOMAIN_V1,
                ),
            )
            self._record_permit_locked(permit)
            return permit

    def authorize(
        self,
        permit: DispatchPermit,
        *,
        current_authority_epoch: int,
        current_execution_epoch: int,
    ) -> DispatchPermitValidationReceipt:
        """Consume one permit at the logical backend-call linearization point."""

        authority_epoch = _non_negative_int(
            current_authority_epoch,
            name="current_authority_epoch",
        )
        execution_epoch = _non_negative_int(
            current_execution_epoch,
            name="current_execution_epoch",
        )
        checked_at = _finite_time(self._clock(), name="checked_at")
        with self._lock:
            status = self._validation_status_locked(
                permit,
                current_authority_epoch=authority_epoch,
                current_execution_epoch=execution_epoch,
            )
            accepted = status == "accepted"
            if accepted:
                self._consumed_permit_ids.add(permit.permit_id)
            current = self._current_token
            payload = {
                "permit_id": getattr(permit, "permit_id", "invalid-permit"),
                "permit_kind": getattr(permit, "permit_kind", "unknown"),
                "control_domain_id": self.control_domain_id,
                "checked_at": checked_at,
                "accepted": accepted,
                "status": status,
                "current_token_id": None if current is None else current.token_id,
                "current_owner_generation": (
                    None if current is None else current.owner_generation
                ),
                "current_authority_epoch": authority_epoch,
                "current_execution_epoch": execution_epoch,
            }
            semantic_hash = canonical_content_hash(
                payload,
                domain=DISPATCH_VALIDATION_HASH_DOMAIN_V1,
            )
            return DispatchPermitValidationReceipt(
                receipt_id=f"dispatch-validation:{semantic_hash}",
                **payload,
                semantic_content_hash=semantic_hash,
            )

    def _validation_status_locked(
        self,
        permit: DispatchPermit,
        *,
        current_authority_epoch: int,
        current_execution_epoch: int,
    ) -> str:
        if not isinstance(
            permit,
            (
                MotionDispatchPermit,
                SafetyControlDispatchPermit,
                LifecycleControlDispatchPermit,
            ),
        ):
            return "unsupported_permit_type"
        if permit.runtime_session_id != self.runtime_session_id:
            return "wrong_runtime_session"
        if permit.control_domain_id != self.control_domain_id:
            return "wrong_control_domain"
        admitted_hash = self._issued_permit_hash_by_id.get(permit.permit_id)
        if admitted_hash is None:
            return "permit_not_issued_by_registry"
        if admitted_hash != permit.semantic_content_hash:
            return "permit_content_mismatch"
        if permit.permit_id in self._consumed_permit_ids:
            return "permit_already_consumed"

        if isinstance(permit, MotionDispatchPermit):
            current = self._current_token
            if current is None:
                return "owner_token_revoked"
            if permit.owner_generation != current.owner_generation:
                return "owner_generation_mismatch"
            if permit.token_id != current.token_id:
                return "owner_token_revoked"
            if permit.authority_epoch != current_authority_epoch:
                return "authority_epoch_mismatch"
            if permit.execution_epoch != current_execution_epoch:
                return "execution_epoch_mismatch"
            if current_authority_epoch != self._authority_epoch:
                return "registry_authority_epoch_stale"
            if current_execution_epoch != self._execution_epoch:
                return "registry_execution_epoch_stale"
            return "accepted"

        if isinstance(permit, SafetyControlDispatchPermit):
            if permit.operation not in SAFETY_CONTROL_OPERATIONS:
                return "unauthorized_safety_operation"
            if permit.safety_source not in SAFETY_CONTROL_SOURCES:
                return "unauthorized_safety_source"
            if permit.target_execution_epoch != current_execution_epoch:
                return "execution_epoch_mismatch"
            if current_execution_epoch != self._execution_epoch:
                return "registry_execution_epoch_stale"
            return "accepted"

        if permit.operation not in LIFECYCLE_CONTROL_OPERATIONS:
            return "unauthorized_lifecycle_operation"
        if permit.target_execution_epoch != current_execution_epoch:
            return "execution_epoch_mismatch"
        if current_execution_epoch != self._execution_epoch:
            return "registry_execution_epoch_stale"
        return "accepted"

    def _grant_owner_locked(
        self,
        *,
        owner_runtime_id: str,
        owner_kind: str,
        authority_epoch: int | None = None,
    ) -> DispatchOwnerToken:
        normalized_owner_runtime_id = _identifier(
            owner_runtime_id,
            name="owner_runtime_id",
        )
        normalized_kind = str(owner_kind).strip()
        if normalized_kind not in DISPATCH_OWNER_KINDS:
            raise ValueError("Dispatch owner must be a legacy or shared runtime.")
        if self._current_token is not None:
            raise RuntimeError("A dispatch owner is already active.")
        acquired_at = _finite_time(self._clock(), name="acquired_at")
        token_authority_epoch = (
            self._authority_epoch
            if authority_epoch is None
            else _non_negative_int(authority_epoch, name="authority_epoch")
        )
        next_generation = self._owner_generation + 1
        sequence = self._next_fact_sequence
        identity = self._fact_identity(sequence, namespace="dispatch_owner_token")
        payload = {
            "token_id": identity.fact_id,
            "runtime_session_id": self.runtime_session_id,
            "control_domain_id": self.control_domain_id,
            "owner_runtime_id": normalized_owner_runtime_id,
            "owner_kind": normalized_kind,
            "owner_generation": next_generation,
            "authority_epoch_at_acquire": token_authority_epoch,
            "execution_epoch_at_acquire": self._execution_epoch,
            "acquired_at": acquired_at,
            "revoked_at": None,
            "process_scope": "in_process_only",
        }
        token = DispatchOwnerToken(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=DISPATCH_OWNER_TOKEN_HASH_DOMAIN_V1,
            ),
        )
        self._authority_epoch = token_authority_epoch
        self._owner_generation = next_generation
        self._next_fact_sequence += 1
        self._current_token = token
        self._ever_granted = True
        return token

    def _require_current_token_locked(self, token: DispatchOwnerToken) -> None:
        if not isinstance(token, DispatchOwnerToken):
            raise TypeError("token must be a DispatchOwnerToken.")
        if self._current_token is None:
            raise RuntimeError("No dispatch owner is active.")
        if (
            token.token_id != self._current_token.token_id
            or token.owner_generation != self._current_token.owner_generation
            or token.semantic_content_hash != self._current_token.semantic_content_hash
        ):
            raise ValueError("Dispatch owner token is not current.")

    def _validate_cancellation_epoch_locked(
        self,
        cancellation: ExecutionCancellation,
    ) -> None:
        if cancellation.runtime_session_id != self.runtime_session_id:
            raise ValueError("Cancellation belongs to another runtime session.")
        if cancellation.control_domain_id != self.control_domain_id:
            raise ValueError("Cancellation belongs to another control domain.")
        if cancellation.effective_scope != "control_domain":
            raise ValueError("Dispatch execution epoch requires domain-wide cancellation.")
        if cancellation.previous_execution_epoch != self._execution_epoch:
            raise ValueError("Cancellation does not start from the active execution epoch.")
        if cancellation.execution_epoch != self._execution_epoch + 1:
            raise ValueError("Cancellation must advance one execution epoch.")

    def _record_permit_locked(self, permit: DispatchPermit) -> None:
        if permit.permit_id in self._issued_permit_hash_by_id:
            raise RuntimeError("Dispatch permit ID collision.")
        self._issued_permit_hash_by_id[permit.permit_id] = permit.semantic_content_hash

    def _take_sequence_locked(self) -> int:
        sequence = self._next_fact_sequence
        self._next_fact_sequence += 1
        return sequence

    def _fact_identity(self, sequence: int, *, namespace: str) -> RuntimeFactIdentity:
        return RuntimeFactIdentity(
            runtime_session_id=self.runtime_session_id,
            control_domain_id=self.control_domain_id,
            producer_id="dispatch-owner-registry",
            producer_epoch=0,
            sequence=sequence,
            namespace=namespace,
        )


__all__ = [
    "AdmittedSafetyControlEvidence",
    "DispatchOwnerRegistry",
    "DispatchOwnerSnapshot",
    "DispatchOwnerToken",
    "DispatchOwnerTransition",
    "DispatchPermit",
    "DispatchPermitValidationReceipt",
    "LifecycleControlAuthorization",
    "LifecycleControlDispatchPermit",
    "MotionDispatchPermit",
    "SafetyControlDispatchPermit",
]
