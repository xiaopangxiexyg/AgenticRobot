"""Prepare exact trajectory-derived inputs for one composition target.

This module owns only generic input derivation and admission.  It does not
evaluate authority, publish an action proposal, create an execution target,
or advance the trajectory emission barrier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.component_action import ComponentAction
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator

from ..activation import CompositionActivationScope
from ..composition import (
    CompositionInputAdmissionReceipt,
    CompositionInputDraft,
    CompositionSpec,
)
from ..composition.input_adapter import build_composition_input_draft
from ._validation import _finite, _required
from .contracts import (
    TRAJECTORY_EDITING_MODE_PASS_THROUGH,
    TRAJECTORY_EDITING_MODE_RESIDUAL_ACTIVE,
    TRAJECTORY_TARGET_STATUS_READY,
    TrajectoryEditingState,
    TrajectoryTargetPreparation,
)
from .reducer import prepare_trajectory_target
from .shadow_contracts import TrajectoryCompositionContributorStamp
from .warp import (
    ResidualTemporalWarpSpec,
    TrajectoryResidualTemporalSample,
    sample_residual_for_trajectory_target,
)


TRAJECTORY_COMPOSITION_INPUT_PREPARATION_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.trajectory_composition_input_preparation.development"
)
TRAJECTORY_COMPOSITION_INPUT_ADMISSION_MODES_DEVELOPMENT = frozenset(
    {"shadow", "publication_candidate"}
)
TRAJECTORY_COMPOSITION_INPUT_PREPARATION_STATUSES_DEVELOPMENT = frozenset(
    {"ready", "no_action", "invalid"}
)


@runtime_checkable
class TrajectoryCompositionInputReceiver(Protocol):
    """Minimum receiver surface needed for trajectory input preparation."""

    runtime_session_id: str
    control_domain_id: str
    coordinator: ControlAdmissionCoordinator

    def admit_composition_input(
        self,
        draft: CompositionInputDraft,
        *,
        activation_scope: CompositionActivationScope | None = None,
    ) -> CompositionInputAdmissionReceipt: ...


def _lineage(
    *pairs: tuple[tuple[str, ...], tuple[str, ...]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    hashes_by_id: dict[str, str] = {}
    ordered_ids: list[str] = []
    for ids, hashes in pairs:
        if len(ids) != len(hashes):
            raise ValueError("Source lineage id/hash lengths differ.")
        for fact_id, fact_hash in zip(ids, hashes):
            normalized_id = _required(fact_id, name="source_fact_id")
            normalized_hash = _required(fact_hash, name="source_fact_hash")
            previous = hashes_by_id.get(normalized_id)
            if previous is not None and previous != normalized_hash:
                raise ValueError("Source lineage identity maps to two hashes.")
            if previous is None:
                ordered_ids.append(normalized_id)
                hashes_by_id[normalized_id] = normalized_hash
    return tuple(ordered_ids), tuple(hashes_by_id[value] for value in ordered_ids)


@dataclass(frozen=True)
class TrajectoryCompositionInputPreparation:
    """Typed result before composition evaluation or proposal publication."""

    editing_state_hash: str
    interaction_mode: str
    base_lifecycle_status: str
    preparation: TrajectoryTargetPreparation
    admission_mode: str
    output_status: str
    reason: str
    nominal_role: str
    residual_role: str
    residual_sample: TrajectoryResidualTemporalSample | None = None
    nominal_input_admission: CompositionInputAdmissionReceipt | None = None
    residual_input_admission: CompositionInputAdmissionReceipt | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        state_hash = _required(self.editing_state_hash, name="editing_state_hash")
        interaction_mode = _required(
            self.interaction_mode,
            name="interaction_mode",
        )
        lifecycle = _required(
            self.base_lifecycle_status,
            name="base_lifecycle_status",
        )
        if not isinstance(self.preparation, TrajectoryTargetPreparation):
            raise TypeError("preparation must be TrajectoryTargetPreparation.")
        mode = _required(self.admission_mode, name="admission_mode")
        if mode not in TRAJECTORY_COMPOSITION_INPUT_ADMISSION_MODES_DEVELOPMENT:
            raise ValueError("Unsupported trajectory input admission mode.")
        status = _required(self.output_status, name="output_status")
        if status not in (
            TRAJECTORY_COMPOSITION_INPUT_PREPARATION_STATUSES_DEVELOPMENT
        ):
            raise ValueError("Unsupported trajectory input preparation status.")
        reason = _required(self.reason, name="reason")
        nominal_role = _required(self.nominal_role, name="nominal_role")
        residual_role = _required(self.residual_role, name="residual_role")
        if nominal_role == residual_role:
            raise ValueError("nominal_role and residual_role must differ.")
        sample = self.residual_sample
        if sample is not None and not isinstance(
            sample,
            TrajectoryResidualTemporalSample,
        ):
            raise TypeError(
                "residual_sample must be TrajectoryResidualTemporalSample."
            )
        for name, receipt in (
            ("nominal_input_admission", self.nominal_input_admission),
            ("residual_input_admission", self.residual_input_admission),
        ):
            if receipt is not None and not isinstance(
                receipt,
                CompositionInputAdmissionReceipt,
            ):
                raise TypeError(
                    f"{name} must be CompositionInputAdmissionReceipt."
                )
        if status == "ready" and mode == "publication_candidate":
            if (
                self.nominal_input_admission is None
                or not self.nominal_input_admission.accepted
                or self.residual_input_admission is None
                or not self.residual_input_admission.accepted
            ):
                raise ValueError(
                    "A ready publication candidate requires both accepted inputs."
                )
        payload = {
            "editing_state_hash": state_hash,
            "interaction_mode": interaction_mode,
            "base_lifecycle_status": lifecycle,
            "preparation_hash": self.preparation.semantic_content_hash,
            "admission_mode": mode,
            "output_status": status,
            "reason": reason,
            "nominal_role": nominal_role,
            "residual_role": residual_role,
            "residual_sample_hash": (
                None if sample is None else sample.semantic_content_hash
            ),
            "nominal_input_admission_hash": (
                None
                if self.nominal_input_admission is None
                else self.nominal_input_admission.semantic_content_hash
            ),
            "residual_input_admission_hash": (
                None
                if self.residual_input_admission is None
                else self.residual_input_admission.semantic_content_hash
            ),
        }
        object.__setattr__(self, "editing_state_hash", state_hash)
        object.__setattr__(self, "interaction_mode", interaction_mode)
        object.__setattr__(self, "base_lifecycle_status", lifecycle)
        object.__setattr__(self, "admission_mode", mode)
        object.__setattr__(self, "output_status", status)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "nominal_role", nominal_role)
        object.__setattr__(self, "residual_role", residual_role)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    TRAJECTORY_COMPOSITION_INPUT_PREPARATION_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def selected_input_identity_by_role(
        self,
    ) -> Mapping[str, tuple[str, str]]:
        """Exact admitted records that a later evaluation must select."""

        identities: dict[str, tuple[str, str]] = {}
        for role, receipt in (
            (self.nominal_role, self.nominal_input_admission),
            (self.residual_role, self.residual_input_admission),
        ):
            if receipt is None or not receipt.accepted:
                continue
            record = receipt.composition_input
            if record is None:
                raise ValueError("Accepted composition input has no record.")
            identities[role] = (
                record.composition_input_id,
                record.semantic_content_hash,
            )
        return MappingProxyType(identities)


def _result(
    state: TrajectoryEditingState,
    preparation: TrajectoryTargetPreparation,
    *,
    admission_mode: str,
    output_status: str,
    reason: str,
    nominal_role: str,
    residual_role: str,
    residual_sample: TrajectoryResidualTemporalSample | None = None,
    nominal_input_admission: CompositionInputAdmissionReceipt | None = None,
    residual_input_admission: CompositionInputAdmissionReceipt | None = None,
) -> TrajectoryCompositionInputPreparation:
    return TrajectoryCompositionInputPreparation(
        editing_state_hash=state.semantic_content_hash,
        interaction_mode=state.interaction_mode,
        base_lifecycle_status=state.base_lifecycle_status,
        preparation=preparation,
        admission_mode=admission_mode,
        output_status=output_status,
        reason=reason,
        nominal_role=nominal_role,
        residual_role=residual_role,
        residual_sample=residual_sample,
        nominal_input_admission=nominal_input_admission,
        residual_input_admission=residual_input_admission,
    )


def prepare_trajectory_composition_inputs(
    runtime: TrajectoryCompositionInputReceiver,
    *,
    state: TrajectoryEditingState,
    target_effective_at: float,
    composition_spec: CompositionSpec,
    nominal_role: str,
    residual_role: str,
    nominal_stamp: TrajectoryCompositionContributorStamp,
    residual_stamp: TrajectoryCompositionContributorStamp | None = None,
    residual_action: ComponentAction | None = None,
    residual_fact_id: str | None = None,
    residual_fact_hash: str | None = None,
    residual_anchor_receipt_id: str | None = None,
    residual_anchor_receipt_hash: str | None = None,
    residual_effective_until: float | None = None,
    residual_warp_spec: ResidualTemporalWarpSpec | None = None,
    admission_mode: str = "shadow",
) -> TrajectoryCompositionInputPreparation:
    """Derive and admit one exact nominal/residual pair without evaluation."""

    if not isinstance(runtime, TrajectoryCompositionInputReceiver):
        raise TypeError(
            "runtime must implement TrajectoryCompositionInputReceiver."
        )
    if not isinstance(state, TrajectoryEditingState):
        raise TypeError("state must be a TrajectoryEditingState.")
    if not isinstance(composition_spec, CompositionSpec):
        raise TypeError("composition_spec must be a CompositionSpec.")
    if not isinstance(nominal_stamp, TrajectoryCompositionContributorStamp):
        raise TypeError(
            "nominal_stamp must be TrajectoryCompositionContributorStamp."
        )
    mode = _required(admission_mode, name="admission_mode")
    if mode not in TRAJECTORY_COMPOSITION_INPUT_ADMISSION_MODES_DEVELOPMENT:
        raise ValueError("Unsupported trajectory input admission mode.")
    nominal = _required(nominal_role, name="nominal_role")
    residual = _required(residual_role, name="residual_role")
    if nominal == residual:
        raise ValueError("nominal_role and residual_role must differ.")
    if nominal not in composition_spec.input_roles or residual not in (
        composition_spec.input_roles
    ):
        raise ValueError("Trajectory roles must be declared by CompositionSpec.")
    preparation = prepare_trajectory_target(
        state,
        target_effective_at=target_effective_at,
    )
    if preparation.status != TRAJECTORY_TARGET_STATUS_READY:
        return _result(
            state,
            preparation,
            admission_mode=mode,
            output_status="no_action",
            reason=f"trajectory_preparation:{preparation.reason}",
            nominal_role=nominal,
            residual_role=residual,
        )
    if state.interaction_mode == TRAJECTORY_EDITING_MODE_PASS_THROUGH:
        if mode == "shadow":
            return _result(
                state,
                preparation,
                admission_mode=mode,
                output_status="ready",
                reason="pass_through_nominal_preview",
                nominal_role=nominal,
                residual_role=residual,
            )
        return _result(
            state,
            preparation,
            admission_mode=mode,
            output_status="no_action",
            reason="publication_candidate_requires_residual_active",
            nominal_role=nominal,
            residual_role=residual,
        )
    if state.interaction_mode != TRAJECTORY_EDITING_MODE_RESIDUAL_ACTIVE:
        return _result(
            state,
            preparation,
            admission_mode=mode,
            output_status="no_action",
            reason=f"interaction_mode_not_composable:{state.interaction_mode}",
            nominal_role=nominal,
            residual_role=residual,
        )

    residual_values = (
        residual_stamp,
        residual_action,
        residual_fact_id,
        residual_fact_hash,
        residual_anchor_receipt_id,
        residual_anchor_receipt_hash,
        residual_effective_until,
        residual_warp_spec,
    )
    if any(value is None for value in residual_values):
        return _result(
            state,
            preparation,
            admission_mode=mode,
            output_status="no_action",
            reason="residual_input_unavailable",
            nominal_role=nominal,
            residual_role=residual,
        )
    assert residual_stamp is not None
    assert residual_action is not None
    assert residual_fact_id is not None
    assert residual_fact_hash is not None
    assert residual_anchor_receipt_id is not None
    assert residual_anchor_receipt_hash is not None
    assert residual_effective_until is not None
    assert residual_warp_spec is not None
    residual_until = _finite(
        residual_effective_until,
        name="residual_effective_until",
    )
    if residual_until <= preparation.target_effective_at:
        return _result(
            state,
            preparation,
            admission_mode=mode,
            output_status="no_action",
            reason="residual_support_expired",
            nominal_role=nominal,
            residual_role=residual,
        )

    sample = sample_residual_for_trajectory_target(
        state,
        preparation,
        residual_action=residual_action,
        residual_fact_id=residual_fact_id,
        residual_fact_hash=residual_fact_hash,
        residual_anchor_receipt_id=residual_anchor_receipt_id,
        residual_anchor_receipt_hash=residual_anchor_receipt_hash,
        warp_spec=residual_warp_spec,
    )
    keyframe = preparation.keyframe
    active = state.active_trajectory
    activation_binding = state.residual_session_activation_binding
    assert keyframe is not None and active is not None
    assert activation_binding is not None
    activation_cut = activation_binding.control_cut
    receiver_clock = runtime.coordinator.clock_domain
    if (
        activation_cut.runtime_session_id != runtime.runtime_session_id
        or activation_cut.control_domain_id != runtime.control_domain_id
        or activation_cut.clock_domain_id != receiver_clock.clock_domain_id
        or activation_cut.clock_reset_epoch != receiver_clock.reset_epoch
        or active.clock_domain_id != receiver_clock.clock_domain_id
    ):
        raise ValueError(
            "Trajectory session activation and composition runtime do not share "
            "one receiver/clock domain."
        )
    effective_until = min(keyframe.effective_until, residual_until)
    if effective_until <= preparation.target_effective_at:
        return _result(
            state,
            preparation,
            admission_mode=mode,
            output_status="no_action",
            reason="combined_input_support_expired",
            nominal_role=nominal,
            residual_role=residual,
            residual_sample=sample,
        )

    nominal_fact_ids, nominal_fact_hashes = _lineage(
        (
            (active.fact_id,),
            (active.semantic_content_hash,),
        ),
    )
    residual_fact_ids, residual_fact_hashes = _lineage(
        (
            (residual_fact_id, residual_anchor_receipt_id),
            (residual_fact_hash, residual_anchor_receipt_hash),
        ),
    )
    scope = activation_binding.activation_scope
    common = {
        "runtime_session_id": runtime.runtime_session_id,
        "control_domain_id": runtime.control_domain_id,
        "effective_from": preparation.target_effective_at,
        "effective_until": effective_until,
        "effective_clock_domain_id": active.clock_domain_id,
        "composition_activation_id": scope.activation_id,
        "composition_activation_hash": scope.semantic_content_hash,
    }
    audit_mode = (
        {"trajectory_shadow_only": True}
        if mode == "shadow"
        else {"trajectory_publication_candidate": True}
    )
    nominal_draft = build_composition_input_draft(
        **common,
        producer_id=nominal_stamp.producer_id,
        producer_epoch=nominal_stamp.producer_epoch,
        sequence=nominal_stamp.sequence,
        input_role=nominal,
        action=preparation.action,
        produced_at=nominal_stamp.produced_at,
        produced_clock_domain_id=nominal_stamp.produced_clock_domain_id,
        context_ids=nominal_stamp.context_ids,
        observation_cut_id=nominal_stamp.observation_cut_id,
        policy_output_id=nominal_stamp.policy_output_id,
        source_fact_ids=nominal_fact_ids,
        source_fact_hashes=nominal_fact_hashes,
        audit_metadata={
            **dict(nominal_stamp.audit_metadata),
            "trajectory_id": active.trajectory_id,
            "trajectory_hash": active.semantic_content_hash,
            "trajectory_keyframe_hash": keyframe.semantic_content_hash,
            "trajectory_publication_basis_hash": state.publication_basis_hash,
            **audit_mode,
        },
    )
    residual_draft = build_composition_input_draft(
        **common,
        producer_id=residual_stamp.producer_id,
        producer_epoch=residual_stamp.producer_epoch,
        sequence=residual_stamp.sequence,
        input_role=residual,
        action=sample.action,
        produced_at=residual_stamp.produced_at,
        produced_clock_domain_id=residual_stamp.produced_clock_domain_id,
        context_ids=residual_stamp.context_ids,
        observation_cut_id=residual_stamp.observation_cut_id,
        policy_output_id=residual_stamp.policy_output_id,
        source_fact_ids=residual_fact_ids,
        source_fact_hashes=residual_fact_hashes,
        audit_metadata={
            **dict(residual_stamp.audit_metadata),
            "trajectory_residual_sample_hash": sample.semantic_content_hash,
            "trajectory_residual_weight": sample.weight,
            "active_residual_anchor_binding_id": (
                sample.active_anchor_binding_id
            ),
            "active_residual_anchor_binding_hash": (
                sample.active_anchor_binding_hash
            ),
            "residual_anchor_receipt_id": sample.residual_anchor_receipt_id,
            "residual_anchor_receipt_hash": (
                sample.residual_anchor_receipt_hash
            ),
            "anchor_request_generation": sample.anchor_request_generation,
            "trajectory_publication_basis_hash": state.publication_basis_hash,
            **audit_mode,
        },
    )
    nominal_receipt = runtime.admit_composition_input(
        nominal_draft,
        activation_scope=scope,
    )
    if not nominal_receipt.accepted:
        return _result(
            state,
            preparation,
            admission_mode=mode,
            output_status="invalid",
            reason=(
                "nominal_input_rejected:"
                f"{nominal_receipt.rejection_code or 'unknown'}"
            ),
            nominal_role=nominal,
            residual_role=residual,
            residual_sample=sample,
            nominal_input_admission=nominal_receipt,
        )
    residual_receipt = runtime.admit_composition_input(
        residual_draft,
        activation_scope=scope,
    )
    if not residual_receipt.accepted:
        return _result(
            state,
            preparation,
            admission_mode=mode,
            output_status="invalid",
            reason=(
                "residual_input_rejected:"
                f"{residual_receipt.rejection_code or 'unknown'}"
            ),
            nominal_role=nominal,
            residual_role=residual,
            residual_sample=sample,
            nominal_input_admission=nominal_receipt,
            residual_input_admission=residual_receipt,
        )
    return _result(
        state,
        preparation,
        admission_mode=mode,
        output_status="ready",
        reason="trajectory_inputs_admitted",
        nominal_role=nominal,
        residual_role=residual,
        residual_sample=sample,
        nominal_input_admission=nominal_receipt,
        residual_input_admission=residual_receipt,
    )


__all__ = [
    "TRAJECTORY_COMPOSITION_INPUT_ADMISSION_MODES_DEVELOPMENT",
    "TRAJECTORY_COMPOSITION_INPUT_PREPARATION_HASH_DOMAIN_DEVELOPMENT",
    "TRAJECTORY_COMPOSITION_INPUT_PREPARATION_STATUSES_DEVELOPMENT",
    "TrajectoryCompositionInputReceiver",
    "TrajectoryCompositionInputPreparation",
    "prepare_trajectory_composition_inputs",
]
