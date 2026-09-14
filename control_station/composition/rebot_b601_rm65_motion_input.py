"""Adapt one admitted B601/RM65 residual into a composition input.

The B601 source owns host-fresh feedback, FK, frame conversion, and anchor-
relative pose math.  This module only binds that exact current-tick result to
one composition activation.  It does not retain a previous residual, evaluate
IK or safety, publish authority, or dispatch to a simulator or robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from control_composition.activation import CompositionActivationScope
from control_composition.composition import (
    CompositionInputDraft,
    build_composition_input_draft,
)
from control_station.composition.rebot_b601_rm65_residual import (
    B601RM65ResidualStep,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ComponentActionSchema

B601_RM65_MOTION_INPUT_PROFILE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.b601_rm65_motion_input_profile.development"
)
B601_RM65_RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND = (
    "b601_rm65_residual_anchor"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


@dataclass(frozen=True)
class B601RM65MotionInputProfile:
    """Exact composition identity expected from the B601 residual source."""

    profile_id: str
    input_role: str
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    action_schema: ComponentActionSchema
    maximum_adaptation_age_sec: float
    enabled: bool = False
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "profile_id",
                "input_role",
                "runtime_session_id",
                "control_domain_id",
                "producer_id",
            )
        }
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool.")
        epoch = int(self.producer_epoch)
        if (
            isinstance(self.producer_epoch, bool)
            or epoch != self.producer_epoch
            or epoch < 0
        ):
            raise ValueError("producer_epoch must be a non-negative integer.")
        if not isinstance(self.action_schema, ComponentActionSchema):
            raise TypeError("action_schema must be ComponentActionSchema.")
        maximum_age = _finite(
            self.maximum_adaptation_age_sec,
            name="maximum_adaptation_age_sec",
        )
        if maximum_age <= 0.0:
            raise ValueError("maximum_adaptation_age_sec must be positive.")
        for component_id in self.action_schema.component_ids:
            payload = self.action_schema.payload_schema_by_component[component_id]
            if (
                payload.representation_family != "cartesian_pose"
                or payload.value_semantics != "relative_delta"
            ):
                raise ValueError(
                    "B601/RM65 motion inputs require relative-delta Cartesian "
                    f"pose components; got {component_id!r}."
                )
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "producer_epoch", epoch)
        object.__setattr__(self, "maximum_adaptation_age_sec", maximum_age)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    **text,
                    "producer_epoch": epoch,
                    "action_schema_hash": self.action_schema.semantic_content_hash,
                    "maximum_adaptation_age_sec": maximum_age,
                    "enabled": self.enabled,
                },
                domain=(
                    B601_RM65_MOTION_INPUT_PROFILE_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )


@dataclass(frozen=True)
class B601RM65MotionInputAdaptation:
    """One non-authoritative input draft or an explicit current-tick rejection."""

    residual_step: B601RM65ResidualStep
    input_draft: CompositionInputDraft | None
    rejection_reason: str | None
    command_dispatched: bool = False
    authoritative: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.residual_step, B601RM65ResidualStep):
            raise TypeError("residual_step must be B601RM65ResidualStep.")
        if (self.input_draft is None) == (self.rejection_reason is None):
            raise ValueError(
                "A motion input adaptation must contain exactly one of an "
                "input draft or rejection reason."
            )
        if self.input_draft is not None:
            action = self.residual_step.relative_action
            adaptation = self.residual_step.adaptation
            pose_sample = self.residual_step.projection.pose_sample
            if action is None or (
                self.input_draft.action.semantic_content_hash
                != action.semantic_content_hash
            ):
                raise ValueError("Motion input draft differs from its residual step.")
            if adaptation is None or pose_sample is None:
                raise ValueError("Motion input draft lacks admitted residual lineage.")
            receipt = adaptation.receipt
            if self.input_draft.source_fact_ids != (
                pose_sample.sample_id,
                receipt.receipt_id,
            ) or self.input_draft.source_fact_hashes != (
                pose_sample.semantic_content_hash,
                receipt.semantic_content_hash,
            ):
                raise ValueError("Motion input draft lost residual source lineage.")
            kind = B601_RM65_RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND
            if (
                self.input_draft.semantic_binding_id_by_kind.get(kind)
                != receipt.anchor_id
                or self.input_draft.semantic_binding_hash_by_kind.get(kind)
                != receipt.anchor_hash
            ):
                raise ValueError("Motion input draft lost residual anchor binding.")
            if self.input_draft.authority_candidate:
                raise ValueError("B601/RM65 motion input cannot be authoritative.")
        else:
            _required(self.rejection_reason, name="rejection_reason")
        if self.command_dispatched or self.authoritative:
            raise ValueError(
                "B601/RM65 motion input adaptation cannot dispatch or claim authority."
            )

    @property
    def admitted(self) -> bool:
        return self.input_draft is not None


def adapt_b601_rm65_residual_to_composition_input(
    residual_step: B601RM65ResidualStep,
    *,
    profile: B601RM65MotionInputProfile,
    activation_scope: CompositionActivationScope,
    sequence: int,
    effective_from: float,
    effective_until: float,
    context_ids: tuple[str, ...] = (),
) -> B601RM65MotionInputAdaptation:
    """Bind one current residual to one activation without retaining fallback state."""

    if not isinstance(residual_step, B601RM65ResidualStep):
        raise TypeError("residual_step must be B601RM65ResidualStep.")
    if not isinstance(profile, B601RM65MotionInputProfile):
        raise TypeError("profile must be B601RM65MotionInputProfile.")
    if not profile.enabled:
        raise RuntimeError("B601/RM65 motion input profile is disabled.")
    if not isinstance(activation_scope, CompositionActivationScope):
        raise TypeError("activation_scope must be CompositionActivationScope.")
    if profile.input_role not in activation_scope.required_roles:
        raise ValueError("Activation does not require the B601 residual role.")

    adaptation = residual_step.adaptation
    if adaptation is None:
        return _rejected(
            residual_step,
            f"current_residual_unavailable:{residual_step.rejection_reason}",
        )
    projection = residual_step.projection
    pose_sample = projection.pose_sample
    if pose_sample is None:
        return _rejected(residual_step, "current_residual_has_no_pose_sample")

    action = adaptation.relative_action
    receipt = adaptation.receipt
    starts = _finite(effective_from, name="effective_from")
    ends = _finite(effective_until, name="effective_until")
    if ends <= starts:
        raise ValueError("effective_until must follow effective_from.")
    if receipt.adapted_clock_domain_id != activation_scope.clock_domain_id:
        return _rejected(residual_step, "adaptation_clock_mismatch")
    if receipt.adapted_at + 1e-12 < activation_scope.input_barrier_at:
        return _rejected(residual_step, "adaptation_precedes_activation_barrier")
    if starts + 1e-12 < receipt.adapted_at:
        return _rejected(residual_step, "effective_time_precedes_adaptation")
    if starts - receipt.adapted_at > profile.maximum_adaptation_age_sec + 1e-12:
        return _rejected(residual_step, "adaptation_is_stale")
    if starts + 1e-12 < activation_scope.input_barrier_at:
        return _rejected(residual_step, "effective_time_precedes_activation_barrier")
    if (
        action.action_schema.semantic_content_hash
        != profile.action_schema.semantic_content_hash
        or receipt.output_action_schema_hash
        != profile.action_schema.semantic_content_hash
    ):
        return _rejected(residual_step, "residual_action_schema_mismatch")
    if action.component_coverage != profile.action_schema.component_ids:
        return _rejected(residual_step, "residual_component_coverage_mismatch")
    source_hash = receipt.source_sample_hash_by_component.get(
        pose_sample.component_id
    )
    if source_hash != pose_sample.semantic_content_hash:
        return _rejected(residual_step, "residual_source_lineage_mismatch")

    source_observation_id = projection.source_observation_sample_id
    if source_observation_id is None:
        return _rejected(residual_step, "source_observation_lineage_missing")
    kind = B601_RM65_RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND
    draft = build_composition_input_draft(
        runtime_session_id=profile.runtime_session_id,
        control_domain_id=profile.control_domain_id,
        producer_id=profile.producer_id,
        producer_epoch=profile.producer_epoch,
        sequence=sequence,
        input_role=profile.input_role,
        action=action,
        produced_at=receipt.adapted_at,
        produced_clock_domain_id=receipt.adapted_clock_domain_id,
        effective_from=starts,
        effective_until=ends,
        effective_clock_domain_id=activation_scope.clock_domain_id,
        context_ids=context_ids,
        observation_cut_id=source_observation_id,
        source_fact_ids=(pose_sample.sample_id, receipt.receipt_id),
        source_fact_hashes=(
            pose_sample.semantic_content_hash,
            receipt.semantic_content_hash,
        ),
        semantic_binding_id_by_kind={kind: receipt.anchor_id},
        semantic_binding_hash_by_kind={kind: receipt.anchor_hash},
        composition_activation_id=activation_scope.activation_id,
        composition_activation_hash=activation_scope.semantic_content_hash,
        audit_metadata={
            "composition_contributor": True,
            "contributor_kind": "b601_rm65_anchor_relative_residual",
            "motion_input_profile_hash": profile.semantic_content_hash,
            "source_emission_hash": projection.source_emission_hash,
            "source_observation_sample_id": source_observation_id,
            "anchor_generation": receipt.anchor_generation,
            "adapter_contract_hash": receipt.adapter_contract_hash,
            "frame_graph_hash": receipt.frame_graph_hash,
        },
    )
    return B601RM65MotionInputAdaptation(
        residual_step=residual_step,
        input_draft=draft,
        rejection_reason=None,
    )


def _rejected(
    residual_step: B601RM65ResidualStep,
    reason: str,
) -> B601RM65MotionInputAdaptation:
    return B601RM65MotionInputAdaptation(
        residual_step=residual_step,
        input_draft=None,
        rejection_reason=reason,
    )


__all__ = [
    "B601_RM65_MOTION_INPUT_PROFILE_HASH_DOMAIN_DEVELOPMENT",
    "B601_RM65_RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND",
    "B601RM65MotionInputAdaptation",
    "B601RM65MotionInputProfile",
    "adapt_b601_rm65_residual_to_composition_input",
]
