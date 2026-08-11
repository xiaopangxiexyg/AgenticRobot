"""Residual-session anchor revision and rebase transaction contracts.

These contracts bind an already-existing residual anchor protocol to one
trajectory session.  They do not poll a source, capture a device sample,
change authority, create HOLD, interpolate targets, or dispatch motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dataset_contracts.canonical_encoding import canonical_content_hash

from ..residual_anchor import ResidualAnchorReceipt, ResidualAnchorRequest
from ._validation import _finite, _identity_pair, _nonnegative_int, _required


ACTIVE_RESIDUAL_ANCHOR_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.active_residual_anchor_binding.development"
)
RESIDUAL_REBASE_REQUEST_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.residual_rebase_request.development"
)
RESIDUAL_REBASE_SEMANTICS_CLEAR_FUTURE_TARGETS = (
    "clear_residual_from_future_composition_targets"
)
RESIDUAL_REBASE_CAPTURE_FIRST_FRESH_AFTER_BARRIER = (
    "first_fresh_complete_sample_after_barrier"
)
RESIDUAL_REBASE_CAPTURE_TRIGGER_ASSOCIATED_SAMPLE = (
    "exact_trigger_associated_source_sample"
)
RESIDUAL_REBASE_CAPTURE_MODES_DEVELOPMENT = frozenset(
    {
        RESIDUAL_REBASE_CAPTURE_FIRST_FRESH_AFTER_BARRIER,
        RESIDUAL_REBASE_CAPTURE_TRIGGER_ASSOCIATED_SAMPLE,
    }
)


def _same_anchor_protocol(
    left: ResidualAnchorReceipt,
    right: ResidualAnchorRequest | ResidualAnchorReceipt,
) -> bool:
    return all(
        getattr(left, field_name) == getattr(right, field_name)
        for field_name in (
            "source_role",
            "composition_activation_id",
            "composition_activation_hash",
            "source_binding_id",
            "source_binding_hash",
            "absolute_action_schema_hash",
            "relative_action_schema_hash",
            "clock_domain_id",
        )
    ) and set(left.component_binding_by_id) == set(
        right.component_ids
        if isinstance(right, ResidualAnchorRequest)
        else right.component_binding_by_id
    )


@dataclass(frozen=True)
class ActiveResidualAnchorBinding:
    """One immutable active-anchor revision inside a residual session."""

    residual_session_id: str
    binding_revision: int
    residual_anchor_receipt: ResidualAnchorReceipt
    effective_for_target_at: float
    predecessor_binding_id: str | None = None
    predecessor_binding_hash: str | None = None
    rebase_request_id: str | None = None
    rebase_request_hash: str | None = None
    semantic_content_hash: str = field(init=False)
    anchor_binding_id: str = field(init=False)

    def __post_init__(self) -> None:
        session_id = _required(
            self.residual_session_id,
            name="residual_session_id",
        )
        revision = _nonnegative_int(
            self.binding_revision,
            name="binding_revision",
        )
        if not isinstance(self.residual_anchor_receipt, ResidualAnchorReceipt):
            raise TypeError(
                "residual_anchor_receipt must be ResidualAnchorReceipt."
            )
        effective_at = _finite(
            self.effective_for_target_at,
            name="effective_for_target_at",
        )
        predecessor_id, predecessor_hash = _identity_pair(
            self.predecessor_binding_id,
            self.predecessor_binding_hash,
            name="predecessor_binding",
        )
        request_id, request_hash = _identity_pair(
            self.rebase_request_id,
            self.rebase_request_hash,
            name="rebase_request",
        )
        if revision == 0 and any(
            value is not None
            for value in (
                predecessor_id,
                predecessor_hash,
                request_id,
                request_hash,
            )
        ):
            raise ValueError(
                "Initial active anchor cannot carry rebase lineage."
            )
        if revision > 0 and any(
            value is None
            for value in (
                predecessor_id,
                predecessor_hash,
                request_id,
                request_hash,
            )
        ):
            raise ValueError(
                "A successor active anchor requires complete predecessor and "
                "rebase lineage."
            )
        receipt = self.residual_anchor_receipt
        payload = {
            "residual_session_id": session_id,
            "binding_revision": revision,
            "residual_anchor_receipt_id": receipt.receipt_id,
            "residual_anchor_receipt_hash": receipt.semantic_content_hash,
            "anchor_request_generation": receipt.request_generation,
            "component_ids": tuple(receipt.component_binding_by_id),
            "effective_for_target_at": effective_at,
            "predecessor_binding_id": predecessor_id,
            "predecessor_binding_hash": predecessor_hash,
            "rebase_request_id": request_id,
            "rebase_request_hash": request_hash,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=ACTIVE_RESIDUAL_ANCHOR_BINDING_HASH_DOMAIN_DEVELOPMENT,
        )
        object.__setattr__(self, "residual_session_id", session_id)
        object.__setattr__(self, "binding_revision", revision)
        object.__setattr__(self, "effective_for_target_at", effective_at)
        object.__setattr__(self, "predecessor_binding_id", predecessor_id)
        object.__setattr__(self, "predecessor_binding_hash", predecessor_hash)
        object.__setattr__(self, "rebase_request_id", request_id)
        object.__setattr__(self, "rebase_request_hash", request_hash)
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "anchor_binding_id",
            f"active-residual-anchor:{revision}:{semantic_hash}",
        )

    @classmethod
    def initial(
        cls,
        *,
        residual_session_id: str,
        residual_anchor_receipt: ResidualAnchorReceipt,
        effective_for_target_at: float,
    ) -> "ActiveResidualAnchorBinding":
        return cls(
            residual_session_id=residual_session_id,
            binding_revision=0,
            residual_anchor_receipt=residual_anchor_receipt,
            effective_for_target_at=effective_for_target_at,
        )


@dataclass(frozen=True)
class ResidualRebaseRequest:
    """Fence an old anchor and request one fresh successor anchor."""

    residual_session_id: str
    expected_anchor_binding_id: str
    expected_anchor_binding_hash: str
    anchor_request: ResidualAnchorRequest
    effective_not_before_target_at: float
    capture_deadline: float
    trigger_kind: str
    trigger_fact_id: str
    trigger_fact_hash: str
    capture_mode: str = RESIDUAL_REBASE_CAPTURE_FIRST_FRESH_AFTER_BARRIER
    trigger_source_action_id: str | None = None
    trigger_source_action_hash: str | None = None
    semantics: str = RESIDUAL_REBASE_SEMANTICS_CLEAR_FUTURE_TARGETS
    semantic_content_hash: str = field(init=False)
    rebase_request_id: str = field(init=False)

    def __post_init__(self) -> None:
        session_id = _required(
            self.residual_session_id,
            name="residual_session_id",
        )
        binding_id = _required(
            self.expected_anchor_binding_id,
            name="expected_anchor_binding_id",
        )
        binding_hash = _required(
            self.expected_anchor_binding_hash,
            name="expected_anchor_binding_hash",
        )
        if not isinstance(self.anchor_request, ResidualAnchorRequest):
            raise TypeError("anchor_request must be ResidualAnchorRequest.")
        target = _finite(
            self.effective_not_before_target_at,
            name="effective_not_before_target_at",
        )
        deadline = _finite(self.capture_deadline, name="capture_deadline")
        if target < self.anchor_request.requested_at:
            raise ValueError(
                "Rebase target boundary cannot predate the anchor request."
            )
        if deadline <= self.anchor_request.requested_at:
            raise ValueError(
                "Rebase capture_deadline must follow the anchor request."
            )
        trigger_kind = _required(self.trigger_kind, name="trigger_kind")
        trigger_id = _required(self.trigger_fact_id, name="trigger_fact_id")
        trigger_hash = _required(
            self.trigger_fact_hash,
            name="trigger_fact_hash",
        )
        capture_mode = _required(self.capture_mode, name="capture_mode")
        if capture_mode not in RESIDUAL_REBASE_CAPTURE_MODES_DEVELOPMENT:
            raise ValueError("Unsupported residual rebase capture mode.")
        source_action_id, source_action_hash = _identity_pair(
            self.trigger_source_action_id,
            self.trigger_source_action_hash,
            name="trigger_source_action",
        )
        if (
            capture_mode == RESIDUAL_REBASE_CAPTURE_TRIGGER_ASSOCIATED_SAMPLE
        ) != (source_action_id is not None):
            raise ValueError(
                "Trigger-associated capture requires one exact source action; "
                "first-fresh capture must not declare one."
            )
        semantics = _required(self.semantics, name="semantics")
        if semantics != RESIDUAL_REBASE_SEMANTICS_CLEAR_FUTURE_TARGETS:
            raise ValueError("Unsupported residual rebase semantics.")
        payload = {
            "residual_session_id": session_id,
            "expected_anchor_binding_id": binding_id,
            "expected_anchor_binding_hash": binding_hash,
            "anchor_request_id": self.anchor_request.request_id,
            "anchor_request_hash": self.anchor_request.semantic_content_hash,
            "effective_not_before_target_at": target,
            "capture_deadline": deadline,
            "trigger_kind": trigger_kind,
            "trigger_fact_id": trigger_id,
            "trigger_fact_hash": trigger_hash,
            "capture_mode": capture_mode,
            "trigger_source_action_id": source_action_id,
            "trigger_source_action_hash": source_action_hash,
            "semantics": semantics,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=RESIDUAL_REBASE_REQUEST_HASH_DOMAIN_DEVELOPMENT,
        )
        object.__setattr__(self, "residual_session_id", session_id)
        object.__setattr__(self, "expected_anchor_binding_id", binding_id)
        object.__setattr__(self, "expected_anchor_binding_hash", binding_hash)
        object.__setattr__(self, "effective_not_before_target_at", target)
        object.__setattr__(self, "capture_deadline", deadline)
        object.__setattr__(self, "trigger_kind", trigger_kind)
        object.__setattr__(self, "trigger_fact_id", trigger_id)
        object.__setattr__(self, "trigger_fact_hash", trigger_hash)
        object.__setattr__(self, "capture_mode", capture_mode)
        object.__setattr__(
            self,
            "trigger_source_action_id",
            source_action_id,
        )
        object.__setattr__(
            self,
            "trigger_source_action_hash",
            source_action_hash,
        )
        object.__setattr__(self, "semantics", semantics)
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "rebase_request_id",
            f"residual-rebase-request:{semantic_hash}",
        )

    @classmethod
    def for_active_binding(
        cls,
        active_binding: ActiveResidualAnchorBinding,
        *,
        anchor_request: ResidualAnchorRequest,
        effective_not_before_target_at: float,
        capture_deadline: float,
        trigger_kind: str,
        trigger_fact_id: str,
        trigger_fact_hash: str,
        capture_mode: str = RESIDUAL_REBASE_CAPTURE_FIRST_FRESH_AFTER_BARRIER,
        trigger_source_action_id: str | None = None,
        trigger_source_action_hash: str | None = None,
    ) -> "ResidualRebaseRequest":
        if not isinstance(active_binding, ActiveResidualAnchorBinding):
            raise TypeError(
                "active_binding must be ActiveResidualAnchorBinding."
            )
        if not isinstance(anchor_request, ResidualAnchorRequest):
            raise TypeError("anchor_request must be ResidualAnchorRequest.")
        previous = active_binding.residual_anchor_receipt
        if anchor_request.request_generation != previous.request_generation + 1:
            raise ValueError(
                "Rebase anchor generation must advance by exactly one."
            )
        if (
            anchor_request.previous_request_id != previous.request_id
            or anchor_request.previous_request_hash != previous.request_hash
        ):
            raise ValueError(
                "Rebase anchor request does not follow the active anchor request."
            )
        if not _same_anchor_protocol(previous, anchor_request):
            raise ValueError("Rebase cannot change the residual anchor protocol.")
        return cls(
            residual_session_id=active_binding.residual_session_id,
            expected_anchor_binding_id=active_binding.anchor_binding_id,
            expected_anchor_binding_hash=active_binding.semantic_content_hash,
            anchor_request=anchor_request,
            effective_not_before_target_at=effective_not_before_target_at,
            capture_deadline=capture_deadline,
            trigger_kind=trigger_kind,
            trigger_fact_id=trigger_fact_id,
            trigger_fact_hash=trigger_fact_hash,
            capture_mode=capture_mode,
            trigger_source_action_id=trigger_source_action_id,
            trigger_source_action_hash=trigger_source_action_hash,
        )


def build_successor_active_residual_anchor_binding(
    predecessor: ActiveResidualAnchorBinding,
    request: ResidualRebaseRequest,
    receipt: ResidualAnchorReceipt,
    *,
    effective_for_target_at: float,
) -> ActiveResidualAnchorBinding:
    """Validate one receipt and construct the next immutable binding."""

    if not isinstance(predecessor, ActiveResidualAnchorBinding):
        raise TypeError("predecessor must be ActiveResidualAnchorBinding.")
    if not isinstance(request, ResidualRebaseRequest):
        raise TypeError("request must be ResidualRebaseRequest.")
    if not isinstance(receipt, ResidualAnchorReceipt):
        raise TypeError("receipt must be ResidualAnchorReceipt.")
    if request.residual_session_id != predecessor.residual_session_id:
        raise ValueError("Rebase request belongs to another residual session.")
    if (
        request.expected_anchor_binding_id != predecessor.anchor_binding_id
        or request.expected_anchor_binding_hash
        != predecessor.semantic_content_hash
    ):
        raise ValueError("Rebase request was built from another active anchor.")
    receipt.validate_request(request.anchor_request)
    if (
        request.capture_mode
        == RESIDUAL_REBASE_CAPTURE_TRIGGER_ASSOCIATED_SAMPLE
        and (
            receipt.source_action_id != request.trigger_source_action_id
            or receipt.source_action_hash != request.trigger_source_action_hash
        )
    ):
        raise ValueError(
            "Residual rebase receipt is not the trigger-associated source sample."
        )
    if not _same_anchor_protocol(predecessor.residual_anchor_receipt, receipt):
        raise ValueError("Successor receipt changed the residual anchor protocol.")
    if receipt.captured_at > request.capture_deadline:
        raise ValueError("Residual rebase receipt missed its capture deadline.")
    effective_at = _finite(
        effective_for_target_at,
        name="effective_for_target_at",
    )
    if effective_at < request.effective_not_before_target_at:
        raise ValueError("Successor anchor cannot activate before its boundary.")
    if effective_at < receipt.captured_at:
        raise ValueError("Successor anchor cannot activate before capture.")
    return ActiveResidualAnchorBinding(
        residual_session_id=predecessor.residual_session_id,
        binding_revision=predecessor.binding_revision + 1,
        residual_anchor_receipt=receipt,
        effective_for_target_at=effective_at,
        predecessor_binding_id=predecessor.anchor_binding_id,
        predecessor_binding_hash=predecessor.semantic_content_hash,
        rebase_request_id=request.rebase_request_id,
        rebase_request_hash=request.semantic_content_hash,
    )


__all__ = [
    "ACTIVE_RESIDUAL_ANCHOR_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "RESIDUAL_REBASE_REQUEST_HASH_DOMAIN_DEVELOPMENT",
    "RESIDUAL_REBASE_CAPTURE_FIRST_FRESH_AFTER_BARRIER",
    "RESIDUAL_REBASE_CAPTURE_MODES_DEVELOPMENT",
    "RESIDUAL_REBASE_CAPTURE_TRIGGER_ASSOCIATED_SAMPLE",
    "RESIDUAL_REBASE_SEMANTICS_CLEAR_FUTURE_TARGETS",
    "ActiveResidualAnchorBinding",
    "ResidualRebaseRequest",
    "build_successor_active_residual_anchor_binding",
]
