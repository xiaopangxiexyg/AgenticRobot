"""Pure same-cut selection for residual-anchor lifecycle facts."""

from __future__ import annotations

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.control_admission import ControlCut
from robot_motion.runtime.producer_epoch import ProducerEpochSnapshot

from ..activation import CompositionActivationScope
from .contracts import (
    RESIDUAL_ANCHOR_LIFECYCLE_PENDING,
    RESIDUAL_ANCHOR_LIFECYCLE_READY,
    RESIDUAL_ANCHOR_LIFECYCLE_SNAPSHOT_HASH_DOMAIN_DEVELOPMENT,
    ResidualAnchorLifecycleProjection,
    ResidualAnchorLifecycleSnapshot,
    ResidualAnchorLifecycleSpec,
    _finite,
    _required,
)
from .store import ResidualAnchorLifecycleStore


def project_residual_anchor_lifecycle(
    store: ResidualAnchorLifecycleStore,
    cut: ControlCut,
) -> ResidualAnchorLifecycleProjection:
    return ResidualAnchorLifecycleProjection(
        control_domain_id=cut.control_domain_id,
        cutoff_at=cut.cutoff_at,
        cutoff_admission_sequence=cut.cutoff_admission_sequence,
        records=store.records_at_or_before(cut),
    )


def build_residual_anchor_lifecycle_snapshot(
    projection: ResidualAnchorLifecycleProjection,
    *,
    producer_epochs: ProducerEpochSnapshot,
    spec: ResidualAnchorLifecycleSpec,
    activation_scope: CompositionActivationScope,
    lifecycle_scope_id: str,
    target_effective_at: float,
) -> ResidualAnchorLifecycleSnapshot:
    if not isinstance(projection, ResidualAnchorLifecycleProjection):
        raise TypeError("projection must be ResidualAnchorLifecycleProjection.")
    if not isinstance(producer_epochs, ProducerEpochSnapshot):
        raise TypeError("producer_epochs must be ProducerEpochSnapshot.")
    if not isinstance(spec, ResidualAnchorLifecycleSpec):
        raise TypeError("spec must be ResidualAnchorLifecycleSpec.")
    if not isinstance(activation_scope, CompositionActivationScope):
        raise TypeError("activation_scope must be CompositionActivationScope.")
    if (
        producer_epochs.control_domain_id != projection.control_domain_id
        or producer_epochs.cutoff_at != projection.cutoff_at
        or producer_epochs.cutoff_admission_sequence
        != projection.cutoff_admission_sequence
    ):
        raise ValueError("Lifecycle projection and producer epochs differ.")
    scope_id = _required(lifecycle_scope_id, name="lifecycle_scope_id")
    target = _finite(target_effective_at, name="target_effective_at")
    candidates = tuple(
        record
        for record in projection.records
        if (
            record.lifecycle_scope_id == scope_id
            and record.lifecycle_spec_hash == spec.semantic_content_hash
            and record.composition_activation_id == activation_scope.activation_id
            and record.composition_activation_hash
            == activation_scope.semantic_content_hash
            and producer_epochs.active_epoch_by_producer.get(record.producer_id)
            == record.producer_epoch
            and record.effective_for_target_at <= target
        )
    )
    selected = (
        None
        if not candidates
        else max(
            candidates,
            key=lambda item: (
                item.anchor_request_generation,
                item.control_admission_sequence,
                item.lifecycle_record_id,
            ),
        )
    )
    if selected is None:
        status, reason = "missing", "residual_anchor_lifecycle_missing"
    elif selected.phase == RESIDUAL_ANCHOR_LIFECYCLE_PENDING:
        status, reason = RESIDUAL_ANCHOR_LIFECYCLE_PENDING, "residual_anchor_rebase_pending"
    else:
        status, reason = RESIDUAL_ANCHOR_LIFECYCLE_READY, "residual_anchor_ready"
    payload = {
        "control_domain_id": projection.control_domain_id,
        "cutoff_at": projection.cutoff_at,
        "cutoff_admission_sequence": projection.cutoff_admission_sequence,
        "target_effective_at": target,
        "lifecycle_scope_id": scope_id,
        "lifecycle_spec_hash": spec.semantic_content_hash,
        "status": status,
        "reason": reason,
        "selected_record_id": (
            None if selected is None else selected.lifecycle_record_id
        ),
        "selected_record_hash": (
            None if selected is None else selected.semantic_content_hash
        ),
    }
    digest = canonical_content_hash(
        payload,
        domain=RESIDUAL_ANCHOR_LIFECYCLE_SNAPSHOT_HASH_DOMAIN_DEVELOPMENT,
    )
    return ResidualAnchorLifecycleSnapshot(
        lifecycle_snapshot_id=f"residual-anchor-lifecycle-snapshot:{digest}",
        control_domain_id=projection.control_domain_id,
        cutoff_at=projection.cutoff_at,
        cutoff_admission_sequence=projection.cutoff_admission_sequence,
        target_effective_at=target,
        lifecycle_scope_id=scope_id,
        lifecycle_spec_hash=spec.semantic_content_hash,
        status=status,
        reason=reason,
        selected_record=selected,
        semantic_content_hash=digest,
    )


def capture_residual_anchor_lifecycle_snapshot(
    store: ResidualAnchorLifecycleStore,
    cut: ControlCut,
    **kwargs,
) -> ResidualAnchorLifecycleSnapshot:
    return build_residual_anchor_lifecycle_snapshot(
        project_residual_anchor_lifecycle(store, cut),
        **kwargs,
    )


__all__ = [
    "build_residual_anchor_lifecycle_snapshot",
    "capture_residual_anchor_lifecycle_snapshot",
    "project_residual_anchor_lifecycle",
]
