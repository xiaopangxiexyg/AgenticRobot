"""Per-target projected inputs derived from one frozen trajectory basis."""

from __future__ import annotations

from dataclasses import dataclass, field

from dataset_contracts.canonical_encoding import canonical_content_hash

from control_composition.composition import ProjectedCompositionInput
from control_composition.residual_anchor import (
    RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND,
)

from .contracts import (
    TRAJECTORY_TARGET_STATUS_READY,
    TrajectoryTargetPreparation,
)
from .frozen_projection import (
    FrozenTrajectoryCompositionProjection,
    _lineage,
    _required,
)
from .reducer import prepare_trajectory_target
from .warp import (
    TrajectoryResidualTemporalSample,
    sample_residual_for_trajectory_target,
)


TRAJECTORY_PROJECTED_INPUT_SET_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.trajectory_projected_input_set.development"
)


@dataclass(frozen=True)
class TrajectoryProjectedInputSet:
    """Projected role inputs for one exact trajectory lattice target."""

    projection_basis_hash: str
    preparation: TrajectoryTargetPreparation
    output_status: str
    reason: str
    inputs: tuple[ProjectedCompositionInput, ...] = ()
    residual_sample: TrajectoryResidualTemporalSample | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        basis_hash = _required(
            self.projection_basis_hash,
            name="projection_basis_hash",
        )
        if not isinstance(self.preparation, TrajectoryTargetPreparation):
            raise TypeError("preparation must be TrajectoryTargetPreparation.")
        status = _required(self.output_status, name="output_status")
        if status not in {"ready", "no_action", "invalid"}:
            raise ValueError("Unsupported projected input-set status.")
        reason = _required(self.reason, name="reason")
        inputs = tuple(self.inputs)
        if any(not isinstance(value, ProjectedCompositionInput) for value in inputs):
            raise TypeError("inputs must contain ProjectedCompositionInput.")
        if len({value.input_role for value in inputs}) != len(inputs):
            raise ValueError("Projected input-set roles must be unique.")
        if any(value.projection_basis_hash != basis_hash for value in inputs):
            raise ValueError("Projected input set mixes projection bases.")
        if status == "ready" and not inputs:
            raise ValueError("Ready projected input set must contain inputs.")
        if status != "ready" and inputs:
            raise ValueError("Non-ready projected input set cannot carry inputs.")
        if self.residual_sample is not None and not isinstance(
            self.residual_sample,
            TrajectoryResidualTemporalSample,
        ):
            raise TypeError("residual_sample must be typed or None.")
        payload = {
            "projection_basis_hash": basis_hash,
            "preparation_hash": self.preparation.semantic_content_hash,
            "output_status": status,
            "reason": reason,
            "projected_input_hashes": tuple(
                value.semantic_content_hash for value in inputs
            ),
            "residual_sample_hash": (
                None
                if self.residual_sample is None
                else self.residual_sample.semantic_content_hash
            ),
        }
        object.__setattr__(self, "projection_basis_hash", basis_hash)
        object.__setattr__(self, "output_status", status)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TRAJECTORY_PROJECTED_INPUT_SET_HASH_DOMAIN_DEVELOPMENT,
            ),
        )


def project_trajectory_composition_inputs(
    projection: FrozenTrajectoryCompositionProjection,
    *,
    target_effective_at: float,
) -> TrajectoryProjectedInputSet:
    """Derive all role inputs for one target using only the frozen basis."""

    if not isinstance(projection, FrozenTrajectoryCompositionProjection):
        raise TypeError("projection must be FrozenTrajectoryCompositionProjection.")
    preparation = prepare_trajectory_target(
        projection.state,
        target_effective_at=target_effective_at,
    )
    if preparation.status != TRAJECTORY_TARGET_STATUS_READY:
        return TrajectoryProjectedInputSet(
            projection_basis_hash=projection.semantic_content_hash,
            preparation=preparation,
            output_status="no_action",
            reason=f"trajectory_preparation:{preparation.reason}",
        )
    active = projection.state.active_trajectory
    keyframe = preparation.keyframe
    assert active is not None and keyframe is not None
    activation_binding = projection.state.residual_session_activation_binding
    activation_scope = (
        None if activation_binding is None else activation_binding.activation_scope
    )
    activation_fields = {
        "composition_activation_id": (
            None if activation_scope is None else activation_scope.activation_id
        ),
        "composition_activation_hash": (
            None
            if activation_scope is None
            else activation_scope.semantic_content_hash
        ),
    }
    nominal_ids, nominal_hashes = _lineage(
        ((active.fact_id,), (active.semantic_content_hash,)),
        (keyframe.source_fact_ids, keyframe.source_fact_hashes),
    )
    nominal_stamp = projection.nominal_stamp
    assert nominal_stamp is not None
    inputs = [
        ProjectedCompositionInput(
            projection_basis_hash=projection.semantic_content_hash,
            runtime_session_id=projection.control_cut.runtime_session_id,
            control_domain_id=projection.control_cut.control_domain_id,
            input_role=projection.nominal_role,
            action=preparation.action,
            producer_id=nominal_stamp.producer_id,
            producer_epoch=nominal_stamp.producer_epoch,
            produced_at=nominal_stamp.produced_at,
            produced_clock_domain_id=nominal_stamp.produced_clock_domain_id,
            effective_from=preparation.target_effective_at,
            effective_until=keyframe.effective_until,
            effective_clock_domain_id=active.clock_domain_id,
            projection_sequence=keyframe.source_index * 16,
            source_kind="trajectory_keyframe",
            source_fact_ids=nominal_ids,
            source_fact_hashes=nominal_hashes,
            context_ids=nominal_stamp.context_ids,
            audit_metadata={
                "trajectory_id": active.trajectory_id,
                "trajectory_hash": active.semantic_content_hash,
                "trajectory_keyframe_hash": keyframe.semantic_content_hash,
                "receiver_admitted": False,
            },
            **activation_fields,
        )
    ]

    residual_sample: TrajectoryResidualTemporalSample | None = None
    residual = projection.residual_input
    if residual is not None:
        if residual.effective_until <= preparation.target_effective_at:
            return TrajectoryProjectedInputSet(
                projection_basis_hash=projection.semantic_content_hash,
                preparation=preparation,
                output_status="no_action",
                reason="residual_support_expired",
            )
        try:
            residual_sample = sample_residual_for_trajectory_target(
                projection.state,
                preparation,
                residual_action=residual.action,
                residual_fact_id=residual.residual_fact_id,
                residual_fact_hash=residual.residual_fact_hash,
                residual_anchor_receipt_id=(
                    residual.residual_anchor_receipt_id
                ),
                residual_anchor_receipt_hash=(
                    residual.residual_anchor_receipt_hash
                ),
                warp_spec=residual.warp_spec,
            )
        except (TypeError, ValueError) as exc:
            return TrajectoryProjectedInputSet(
                projection_basis_hash=projection.semantic_content_hash,
                preparation=preparation,
                output_status="invalid",
                reason=f"residual_projection_invalid:{exc}",
            )
        residual_ids, residual_hashes = _lineage(
            (
                (
                    residual.residual_fact_id,
                    residual.residual_anchor_receipt_id,
                    residual_sample.sample_id,
                ),
                (
                    residual.residual_fact_hash,
                    residual.residual_anchor_receipt_hash,
                    residual_sample.semantic_content_hash,
                ),
            )
        )
        stamp = residual.contributor_stamp
        inputs.append(
            ProjectedCompositionInput(
                projection_basis_hash=projection.semantic_content_hash,
                runtime_session_id=projection.control_cut.runtime_session_id,
                control_domain_id=projection.control_cut.control_domain_id,
                input_role=residual.input_role,
                action=residual_sample.action,
                producer_id=stamp.producer_id,
                producer_epoch=stamp.producer_epoch,
                produced_at=stamp.produced_at,
                produced_clock_domain_id=stamp.produced_clock_domain_id,
                effective_from=preparation.target_effective_at,
                effective_until=min(
                    keyframe.effective_until,
                    residual.effective_until,
                ),
                effective_clock_domain_id=active.clock_domain_id,
                projection_sequence=keyframe.source_index * 16 + 1,
                source_kind="frozen_residual_temporal_sample",
                source_fact_ids=residual_ids,
                source_fact_hashes=residual_hashes,
                context_ids=stamp.context_ids,
                semantic_binding_id_by_kind={
                    RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND: (
                        residual.residual_anchor_receipt_id
                    )
                },
                semantic_binding_hash_by_kind={
                    RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND: (
                        residual.residual_anchor_receipt_hash
                    )
                },
                audit_metadata={
                    "trajectory_residual_sample_hash": (
                        residual_sample.semantic_content_hash
                    ),
                    "receiver_admitted": False,
                },
                **activation_fields,
            )
        )

    for offset, held in enumerate(projection.held_inputs, start=2):
        if not held.effective_from <= preparation.target_effective_at < (
            held.effective_until
        ):
            continue
        stamp = held.contributor_stamp
        inputs.append(
            ProjectedCompositionInput(
                projection_basis_hash=projection.semantic_content_hash,
                runtime_session_id=projection.control_cut.runtime_session_id,
                control_domain_id=projection.control_cut.control_domain_id,
                input_role=held.input_role,
                action=held.action,
                producer_id=stamp.producer_id,
                producer_epoch=stamp.producer_epoch,
                produced_at=stamp.produced_at,
                produced_clock_domain_id=stamp.produced_clock_domain_id,
                effective_from=held.effective_from,
                effective_until=held.effective_until,
                effective_clock_domain_id=active.clock_domain_id,
                projection_sequence=keyframe.source_index * 16 + offset,
                source_kind=held.source_kind,
                source_fact_ids=held.source_fact_ids,
                source_fact_hashes=held.source_fact_hashes,
                context_ids=stamp.context_ids,
                semantic_binding_id_by_kind=(
                    held.semantic_binding_id_by_kind
                ),
                semantic_binding_hash_by_kind=(
                    held.semantic_binding_hash_by_kind
                ),
                audit_metadata={"receiver_admitted": False},
                **activation_fields,
            )
        )

    return TrajectoryProjectedInputSet(
        projection_basis_hash=projection.semantic_content_hash,
        preparation=preparation,
        output_status="ready",
        reason="projected_inputs_ready",
        inputs=tuple(inputs),
        residual_sample=residual_sample,
    )


__all__ = [
    "TRAJECTORY_PROJECTED_INPUT_SET_HASH_DOMAIN_DEVELOPMENT",
    "TrajectoryProjectedInputSet",
    "project_trajectory_composition_inputs",
]
