"""Thin Station binding for one human allocation input sample.

The controller derives fixed absolute/residual roles, admits them atomically,
and forwards the accepted receiver stamps into the reusable evidence window.
It neither polls the device nor evaluates allocation, authority, safety, IK,
reference generation, or dispatch.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Sequence

from control_composition import (
    CompositionActivationScope,
    RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND,
    RESIDUAL_ANCHOR_LIFECYCLE_PENDING,
    RESIDUAL_ANCHOR_LIFECYCLE_READY,
    ResidualAnchorLifecycleAdmissionReceipt,
    ResidualAnchorRequest,
)
from control_composition.allocation import (
    AllocationEvidenceSamplePair,
    AllocationPublicationBasis,
)
from control_composition.composition import (
    CompositionInputBatchDraft,
    DerivedCompositionInputRoleBinding,
    build_same_source_composition_input_batch,
)
from robot_motion.action import ComponentAction
from robot_motion.runtime.control_admission import ControlCut

from .derived_input_publication import (
    StationDerivedInputPublicationCycle,
    StationDerivedInputPublicationPort,
)
from .allocation_publication import StationAllocationPublicationPort
from .mujoco_policy_human_allocation import (
    MujocoPolicyHumanAllocationProfileBundle,
)
from .mujoco_policy_human_residual import (
    MujocoPolicyHumanAllocationDerivation,
)


@dataclass(frozen=True)
class MujocoPolicyHumanAllocationInputCycle:
    status: str
    derivation: MujocoPolicyHumanAllocationDerivation | None = None
    batch: CompositionInputBatchDraft | None = None
    publication: StationDerivedInputPublicationCycle | None = None
    evidence_pair: AllocationEvidenceSamplePair | None = None
    anchor_lifecycle_admission: (
        ResidualAnchorLifecycleAdmissionReceipt | None
    ) = None

    def __post_init__(self) -> None:
        if self.status not in {
            "disabled",
            "anchor_captured",
            "batch_rejected",
            "anchor_lifecycle_rejected",
            "batch_admitted",
        }:
            raise ValueError("Unsupported human allocation input cycle status.")
        if self.status == "disabled":
            if any(
                item is not None
                for item in (
                    self.derivation,
                    self.batch,
                    self.publication,
                    self.evidence_pair,
                    self.anchor_lifecycle_admission,
                )
            ):
                raise ValueError("A disabled input cycle carries no work.")
            return
        if not isinstance(
            self.derivation,
            MujocoPolicyHumanAllocationDerivation,
        ):
            raise TypeError("An active cycle requires a typed derivation.")
        if self.status == "anchor_captured":
            if self.derivation.ready or any(
                item is not None
                for item in (
                    self.batch,
                    self.publication,
                    self.evidence_pair,
                    self.anchor_lifecycle_admission,
                )
            ):
                raise ValueError("Anchor capture must not publish input roles.")
            return
        if not isinstance(self.batch, CompositionInputBatchDraft) or not isinstance(
            self.publication,
            StationDerivedInputPublicationCycle,
        ):
            raise TypeError("A publication cycle requires its batch and receipt.")
        if self.status == "batch_rejected":
            if (
                self.publication.status != "rejected"
                or self.evidence_pair is not None
                or self.anchor_lifecycle_admission is not None
            ):
                raise ValueError("Rejected batch cannot create allocation evidence.")
        elif self.status == "anchor_lifecycle_rejected":
            if (
                self.publication.status != "admitted"
                or not isinstance(
                    self.evidence_pair,
                    AllocationEvidenceSamplePair,
                )
                or not isinstance(
                    self.anchor_lifecycle_admission,
                    ResidualAnchorLifecycleAdmissionReceipt,
                )
                or self.anchor_lifecycle_admission.accepted
            ):
                raise ValueError(
                    "Rejected lifecycle requires an admitted capture pair and "
                    "its rejected receipt."
                )
        elif (
            self.publication.status != "admitted"
            or not isinstance(self.evidence_pair, AllocationEvidenceSamplePair)
        ):
            raise ValueError("Admitted batch requires one retained evidence pair.")
        elif self.derivation.captured_anchor and (
            not isinstance(
                self.anchor_lifecycle_admission,
                ResidualAnchorLifecycleAdmissionReceipt,
            )
            or not self.anchor_lifecycle_admission.accepted
        ):
            raise ValueError(
                "Anchor capture becomes usable only after ready lifecycle admission."
            )


@dataclass(frozen=True)
class _PendingAnchorCapturePublication:
    derivation: MujocoPolicyHumanAllocationDerivation
    batch: CompositionInputBatchDraft
    effective_for_target_at: float
    publication: StationDerivedInputPublicationCycle | None = None
    evidence_pair: AllocationEvidenceSamplePair | None = None


class MujocoPolicyHumanAllocationInputController:
    """Bind a concrete human source sample to generic allocation input facts."""

    def __init__(
        self,
        *,
        bundle: MujocoPolicyHumanAllocationProfileBundle,
        activation_scope: CompositionActivationScope,
        role_bindings: Sequence[DerivedCompositionInputRoleBinding],
        publication_port: StationDerivedInputPublicationPort | None = None,
        allocation_publication_port: (
            StationAllocationPublicationPort | None
        ) = None,
        clock_reset_epoch: int = 0,
    ) -> None:
        if not isinstance(bundle, MujocoPolicyHumanAllocationProfileBundle):
            raise TypeError("bundle must be a typed allocation profile bundle.")
        if not isinstance(activation_scope, CompositionActivationScope):
            raise TypeError("activation_scope must be CompositionActivationScope.")
        if (
            isinstance(clock_reset_epoch, bool)
            or not isinstance(clock_reset_epoch, int)
            or clock_reset_epoch < 0
        ):
            raise ValueError("clock_reset_epoch must be a non-negative integer.")
        bindings = tuple(role_bindings)
        expected_roles = {
            bundle.config.absolute_overlay_role,
            bundle.config.residual_role,
        }
        if {binding.input_role for binding in bindings} != expected_roles or (
            len(bindings) != 2
        ):
            raise ValueError("Role bindings must cover fixed human roles exactly.")
        schema_by_role = bundle.input_action_schema_by_role
        if any(
            binding.action_schema_hash
            != schema_by_role[binding.input_role].semantic_content_hash
            for binding in bindings
        ):
            raise ValueError("Derived role binding schema differs from the profile.")
        if bundle.config.enabled:
            if not isinstance(publication_port, StationDerivedInputPublicationPort):
                raise TypeError("Enabled profile requires derived input publication.")
            if not publication_port.profile.enabled:
                raise ValueError("Enabled profile requires an enabled publication port.")
            bound_activation = publication_port.activation_scope
            if (
                bound_activation is None
                or bound_activation.activation_id != activation_scope.activation_id
                or bound_activation.semantic_content_hash
                != activation_scope.semantic_content_hash
            ):
                raise ValueError("Input publisher belongs to another activation.")
            lifecycle_spec = (
                bundle.composition_contracts.residual_anchor_lifecycle_spec
            )
            if (
                not publication_port.lifecycle_enabled
                or publication_port.lifecycle_spec != lifecycle_spec
            ):
                raise ValueError(
                    "Enabled allocation input requires its anchor lifecycle "
                    "publication binding."
                )
            if not isinstance(
                allocation_publication_port,
                StationAllocationPublicationPort,
            ) or not allocation_publication_port.profile.enabled:
                raise TypeError(
                    "Enabled allocation input requires its allocation publication "
                    "port for generation fencing."
                )
            if (
                allocation_publication_port.publication_barrier
                is not publication_port.publication_barrier
            ):
                raise ValueError(
                    "Input and allocation publication must share one Station barrier."
                )
            allocation_runtime = allocation_publication_port.allocation_runtime
            if (
                allocation_runtime is None
                or allocation_runtime.allocation_scope_id
                != publication_port.lifecycle_scope_id
                or allocation_runtime.activation_scope.activation_id
                != activation_scope.activation_id
                or allocation_runtime.activation_scope.semantic_content_hash
                != activation_scope.semantic_content_hash
            ):
                raise ValueError(
                    "Allocation generation fence belongs to another scope."
                )
        self.bundle = bundle
        self.activation_scope = activation_scope
        self.role_bindings = bindings
        self.publication_port = publication_port
        self.allocation_publication_port = allocation_publication_port
        self.clock_reset_epoch = clock_reset_epoch
        self._lock = threading.RLock()
        self._pending_anchor_capture: (
            _PendingAnchorCapturePublication | None
        ) = None
        self._anchor_effective_for_target_at: float | None = None

    def request_anchor(
        self,
        *,
        requested_at: float,
        effective_not_before_target_at: float | None = None,
    ) -> ResidualAnchorRequest:
        """Fence old evidence as soon as a new anchor request is accepted."""

        if not self.bundle.config.enabled:
            raise RuntimeError("Human allocation profile is disabled.")
        with self._lock:
            if self._pending_anchor_capture is not None:
                raise RuntimeError(
                    "Cannot supersede an anchor capture whose publication is pending."
                )
            request = self.bundle.input_contracts.input_adapter.prepare_anchor_request(
                self.activation_scope,
                requested_at=requested_at,
            )
            boundary = float(
                requested_at
                if effective_not_before_target_at is None
                else effective_not_before_target_at
            )
            publisher = self.publication_port
            assert publisher is not None
            admission = publisher.publish_anchor_lifecycle(
                request,
                phase=RESIDUAL_ANCHOR_LIFECYCLE_PENDING,
                effective_for_target_at=boundary,
                observed_at=requested_at,
            )
            if not admission.accepted:
                raise RuntimeError(
                    "Residual anchor pending lifecycle admission failed: "
                    f"{admission.rejection_code}:{admission.message}"
                )
            allocation_port = self.allocation_publication_port
            assert allocation_port is not None
            allocation_port.advance_anchor_generation(
                request.request_generation
            )
            self.bundle.input_contracts.input_adapter.commit_anchor_request(request)
            self.bundle.evidence_window.clear()
            self._anchor_effective_for_target_at = boundary
            return request

    def process_and_publish(
        self,
        absolute_action: ComponentAction,
        *,
        source_action_id: str,
        source_action_hash: str,
        source_sequence: int,
        source_available_at: float,
        measured_at: float,
        measured_clock_domain_id: str,
        observed_at: float,
        observed_clock_domain_id: str,
        runtime_session_id: str,
        control_domain_id: str,
        produced_at: float,
        produced_clock_domain_id: str,
        effective_from: float,
        effective_until: float,
        effective_clock_domain_id: str,
        context_ids: Sequence[str] = (),
        observation_cut_id: str | None = None,
    ) -> MujocoPolicyHumanAllocationInputCycle:
        """Derive once, then atomically admit both fixed human roles."""

        if not self.bundle.config.enabled:
            return MujocoPolicyHumanAllocationInputCycle(status="disabled")
        with self._lock:
            if self._pending_anchor_capture is not None:
                return self._continue_anchor_capture_publication()
            derivation = (
                self.bundle.input_contracts.input_adapter.derive_allocation_actions(
                    absolute_action,
                    source_action_id=source_action_id,
                    source_action_hash=source_action_hash,
                    source_available_at=source_available_at,
                    measured_at=measured_at,
                    measured_clock_domain_id=measured_clock_domain_id,
                    observed_at=observed_at,
                    observed_clock_domain_id=observed_clock_domain_id,
                )
            )
            anchor = self.bundle.input_contracts.input_adapter.anchor
            if anchor is None:
                raise RuntimeError("Residual adapter did not retain its anchor.")
            if not derivation.ready:
                raise RuntimeError(
                    "Anchor capture must produce an identity residual pair."
                )
            batch = build_same_source_composition_input_batch(
                runtime_session_id=runtime_session_id,
                control_domain_id=control_domain_id,
                role_bindings=self.role_bindings,
                action_by_role=derivation.action_by_role,
                source_sequence=source_sequence,
                produced_at=produced_at,
                produced_clock_domain_id=produced_clock_domain_id,
                effective_from=effective_from,
                effective_until=effective_until,
                effective_clock_domain_id=effective_clock_domain_id,
                source_fact_ids=(source_action_id,),
                source_fact_hashes=(source_action_hash,),
                activation_scope=self.activation_scope,
                semantic_binding_id_by_kind={
                    RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND: (
                        anchor.receipt.receipt_id
                    ),
                },
                semantic_binding_hash_by_kind={
                    RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND: (
                        anchor.receipt.semantic_content_hash
                    ),
                },
                context_ids=context_ids,
                observation_cut_id=observation_cut_id,
                audit_metadata={
                    "allocation_derivation_hash": (
                        derivation.semantic_content_hash
                    ),
                },
            )
            publisher = self.publication_port
            assert publisher is not None
            if derivation.captured_anchor:
                boundary = self._anchor_effective_for_target_at
                if boundary is None:
                    raise RuntimeError(
                        "Anchor capture has no admitted target boundary."
                    )
                self._pending_anchor_capture = _PendingAnchorCapturePublication(
                    derivation=derivation,
                    batch=batch,
                    effective_for_target_at=boundary,
                )
                return self._continue_anchor_capture_publication()
            publication = publisher.publish(lambda: batch)
            if publication.status != "admitted":
                return MujocoPolicyHumanAllocationInputCycle(
                    status="batch_rejected",
                    derivation=derivation,
                    batch=batch,
                    publication=publication,
                )
            pair = self._evidence_pair(
                derivation=derivation,
                publication=publication,
                anchor_receipt=anchor.receipt,
            )
            self.bundle.evidence_window.offer(pair)
            return MujocoPolicyHumanAllocationInputCycle(
                status="batch_admitted",
                derivation=derivation,
                batch=batch,
                publication=publication,
                evidence_pair=pair,
            )

    def _continue_anchor_capture_publication(
        self,
    ) -> MujocoPolicyHumanAllocationInputCycle:
        pending = self._pending_anchor_capture
        if pending is None:
            raise RuntimeError("No anchor capture publication is pending.")
        publisher = self.publication_port
        assert publisher is not None
        publication = pending.publication
        if publication is None:
            publication = publisher.publish(lambda: pending.batch)
            if publication.status != "admitted":
                return MujocoPolicyHumanAllocationInputCycle(
                    status="batch_rejected",
                    derivation=pending.derivation,
                    batch=pending.batch,
                    publication=publication,
                )
            anchor = self.bundle.input_contracts.input_adapter.anchor
            if anchor is None:
                raise RuntimeError("Residual adapter lost its captured anchor.")
            pair = self._evidence_pair(
                derivation=pending.derivation,
                publication=publication,
                anchor_receipt=anchor.receipt,
            )
            pending = _PendingAnchorCapturePublication(
                derivation=pending.derivation,
                batch=pending.batch,
                effective_for_target_at=pending.effective_for_target_at,
                publication=publication,
                evidence_pair=pair,
            )
            self._pending_anchor_capture = pending
        pair = pending.evidence_pair
        if pair is None:
            raise RuntimeError("Admitted anchor capture lacks its evidence pair.")
        anchor = self.bundle.input_contracts.input_adapter.anchor
        request = self.bundle.input_contracts.input_adapter.anchor_request
        if anchor is None or request is None:
            raise RuntimeError("Residual adapter lost its anchor lifecycle state.")
        admission = publisher.publish_anchor_lifecycle(
            request,
            phase=RESIDUAL_ANCHOR_LIFECYCLE_READY,
            effective_for_target_at=pending.effective_for_target_at,
            observed_at=anchor.receipt.captured_at,
            anchor_receipt=anchor.receipt,
        )
        if not admission.accepted:
            return MujocoPolicyHumanAllocationInputCycle(
                status="anchor_lifecycle_rejected",
                derivation=pending.derivation,
                batch=pending.batch,
                publication=publication,
                evidence_pair=pair,
                anchor_lifecycle_admission=admission,
            )
        self.bundle.evidence_window.bind_anchor(anchor.receipt)
        self.bundle.evidence_window.offer(pair)
        self._pending_anchor_capture = None
        return MujocoPolicyHumanAllocationInputCycle(
            status="batch_admitted",
            derivation=pending.derivation,
            batch=pending.batch,
            publication=publication,
            evidence_pair=pair,
            anchor_lifecycle_admission=admission,
        )

    def _evidence_pair(
        self,
        *,
        derivation: MujocoPolicyHumanAllocationDerivation,
        publication: StationDerivedInputPublicationCycle,
        anchor_receipt,
    ) -> AllocationEvidenceSamplePair:
        receipt = publication.receipt
        if receipt is None or not receipt.accepted:
            raise ValueError("Evidence requires an accepted batch receipt.")
        records = {
            item.composition_input.input_role: item.composition_input
            for item in receipt.input_receipts
            if item.composition_input is not None
        }
        expected_roles = {
            self.bundle.config.absolute_overlay_role,
            self.bundle.config.residual_role,
        }
        if set(records) != expected_roles:
            raise ValueError("Accepted batch does not contain both human roles.")
        absolute = records[self.bundle.config.absolute_overlay_role]
        residual = records[self.bundle.config.residual_role]
        if (
            absolute.action.semantic_content_hash
            != derivation.absolute_action.semantic_content_hash
            or residual.action.semantic_content_hash
            != derivation.action_by_role[
                self.bundle.config.residual_role
            ].semantic_content_hash
        ):
            raise ValueError("Accepted input action differs from its derivation.")
        shared_fields = (
            "runtime_session_id",
            "control_domain_id",
            "available_at",
            "available_clock_domain_id",
            "effective_from",
            "effective_clock_domain_id",
            "source_fact_ids",
            "source_fact_hashes",
        )
        if any(
            getattr(absolute, name) != getattr(residual, name)
            for name in shared_fields
        ):
            raise ValueError("Accepted human roles do not share one source cut.")
        if (
            absolute.available_at != receipt.available_at
            or absolute.available_clock_domain_id
            != receipt.available_clock_domain_id
            or max(
                absolute.control_admission_sequence,
                residual.control_admission_sequence,
            )
            != receipt.last_control_admission_sequence
        ):
            raise ValueError("Batch receipt stamps differ from admitted records.")
        return AllocationEvidenceSamplePair(
            runtime_session_id=absolute.runtime_session_id,
            control_domain_id=absolute.control_domain_id,
            clock_domain_id=absolute.effective_clock_domain_id,
            clock_reset_epoch=self.clock_reset_epoch,
            sampled_at=absolute.effective_from,
            available_at=absolute.available_at,
            last_control_admission_sequence=(
                receipt.last_control_admission_sequence
            ),
            source_sample_id=derivation.profile_step.source_action_id,
            source_sample_hash=derivation.profile_step.source_action_hash,
            absolute_action=absolute.action,
            residual_action=residual.action,
            anchor_receipt_id=anchor_receipt.receipt_id,
            anchor_receipt_hash=anchor_receipt.semantic_content_hash,
            anchor_request_generation=anchor_receipt.request_generation,
        )

    def capture_basis(
        self,
        *,
        allocation_scope_id: str,
        control_cut: ControlCut,
        requested_effective_from: float,
    ) -> AllocationPublicationBasis | None:
        if not self.bundle.config.enabled:
            return None
        with self._lock:
            return self.bundle.evidence_window.capture_basis(
                allocation_scope_id=allocation_scope_id,
                control_cut=control_cut,
                activation_scope=self.activation_scope,
                requested_effective_from=requested_effective_from,
            )


__all__ = [
    "MujocoPolicyHumanAllocationInputController",
    "MujocoPolicyHumanAllocationInputCycle",
]
