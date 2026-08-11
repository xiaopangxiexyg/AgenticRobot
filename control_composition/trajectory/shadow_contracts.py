"""Contracts for non-authoritative trajectory composition previews."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.component_action import ComponentAction
from robot_motion.runtime.immutable import deep_freeze

from ..composition import CompositionInputAdmissionReceipt
from ..events import CompositionDomainEvent
from ..runtime import CompositeProposalSubmission
from .contracts import (
    TRAJECTORY_EDITING_MODE_PASS_THROUGH,
    TRAJECTORY_EDITING_MODE_RESIDUAL_ACTIVE,
    TrajectoryTargetPreparation,
)
from .warp import TrajectoryResidualTemporalSample


TRAJECTORY_COMPOSITION_SHADOW_RESULT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.trajectory_composition_shadow_result.development"
)
TRAJECTORY_COMPOSITION_SHADOW_EVENT_KIND = (
    "trajectory_composition_shadow_result"
)
TRAJECTORY_SHADOW_EXECUTABLE_PREVIEW_UNAVAILABLE = (
    "unavailable_non_authoritative_shadow"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _optional(value: object | None, *, name: str) -> str | None:
    return None if value is None else _required(value, name=name)


@dataclass(frozen=True)
class TrajectoryCompositionContributorStamp:
    """Caller-owned identity/availability facts for one derived input.

    The stamp does not own a source worker or sequence counter.  A source
    adapter supplies values from its already-active producer incarnation.
    """

    producer_id: str
    producer_epoch: int
    sequence: int
    produced_at: float
    produced_clock_domain_id: str
    context_ids: tuple[str, ...] = ()
    observation_cut_id: str | None = None
    policy_output_id: str | None = None
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        producer_id = _required(self.producer_id, name="producer_id")
        epoch = _nonnegative_int(self.producer_epoch, name="producer_epoch")
        sequence = _nonnegative_int(self.sequence, name="sequence")
        produced_at = _finite(self.produced_at, name="produced_at")
        clock = _required(
            self.produced_clock_domain_id,
            name="produced_clock_domain_id",
        )
        context_ids = tuple(
            _required(value, name="context_id") for value in self.context_ids
        )
        if len(context_ids) != len(set(context_ids)):
            raise ValueError("context_ids must be unique.")
        object.__setattr__(self, "producer_id", producer_id)
        object.__setattr__(self, "producer_epoch", epoch)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "produced_at", produced_at)
        object.__setattr__(self, "produced_clock_domain_id", clock)
        object.__setattr__(self, "context_ids", context_ids)
        object.__setattr__(
            self,
            "observation_cut_id",
            _optional(self.observation_cut_id, name="observation_cut_id"),
        )
        object.__setattr__(
            self,
            "policy_output_id",
            _optional(self.policy_output_id, name="policy_output_id"),
        )
        object.__setattr__(
            self,
            "audit_metadata",
            deep_freeze(dict(self.audit_metadata)),
        )


@dataclass(frozen=True)
class TrajectoryCompositionShadowResult:
    """Complete preview result from one exact target and composition cut."""

    editing_state_hash: str
    interaction_mode: str
    base_lifecycle_status: str
    preparation: TrajectoryTargetPreparation
    output_status: str
    reason: str
    residual_sample: TrajectoryResidualTemporalSample | None = None
    nominal_input_admission: CompositionInputAdmissionReceipt | None = None
    residual_input_admission: CompositionInputAdmissionReceipt | None = None
    composition_submission: CompositeProposalSubmission | None = None
    active_base_binding_id: str | None = None
    active_base_binding_hash: str | None = None
    shadow_trajectory_id: str | None = None
    shadow_trajectory_hash: str | None = None
    shadow_keyframe_hash: str | None = None
    shadow_action: ComponentAction | None = None
    executable_preview_status: str = (
        TRAJECTORY_SHADOW_EXECUTABLE_PREVIEW_UNAVAILABLE
    )
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
            raise TypeError("preparation must be a TrajectoryTargetPreparation.")
        status = _required(self.output_status, name="output_status")
        if status not in {"ready", "no_action", "invalid"}:
            raise ValueError("Unsupported trajectory shadow output status.")
        reason = _required(self.reason, name="reason")
        executable_status = _required(
            self.executable_preview_status,
            name="executable_preview_status",
        )
        if executable_status != TRAJECTORY_SHADOW_EXECUTABLE_PREVIEW_UNAVAILABLE:
            raise ValueError("Trajectory shadow cannot claim executable preview.")

        submission = self.composition_submission
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
        if submission is not None:
            if not isinstance(submission, CompositeProposalSubmission):
                raise TypeError(
                    "composition_submission must be CompositeProposalSubmission."
                )
            if submission.proposal_admission is not None:
                raise ValueError(
                    "Trajectory shadow must stop before proposal admission."
                )
            evaluation = submission.composition_evaluation
            if evaluation.output_status != status:
                raise ValueError(
                    "Trajectory shadow status must match composition evaluation."
                )
            if evaluation.target_effective_at != self.preparation.target_effective_at:
                raise ValueError(
                    "Trajectory shadow evaluation targets another preparation."
                )
        sample = self.residual_sample
        if sample is not None and not isinstance(
            sample,
            TrajectoryResidualTemporalSample,
        ):
            raise TypeError(
                "residual_sample must be TrajectoryResidualTemporalSample."
            )
        if interaction_mode == TRAJECTORY_EDITING_MODE_PASS_THROUGH:
            if sample is not None or submission is not None:
                raise ValueError(
                    "Pass-through shadow must not invoke residual composition."
                )
        elif interaction_mode == TRAJECTORY_EDITING_MODE_RESIDUAL_ACTIVE:
            if status == "ready" and (sample is None or submission is None):
                raise ValueError(
                    "Residual-active ready shadow requires sample/evaluation."
                )

        base_id = _optional(
            self.active_base_binding_id,
            name="active_base_binding_id",
        )
        base_hash = _optional(
            self.active_base_binding_hash,
            name="active_base_binding_hash",
        )
        if (base_id is None) != (base_hash is None):
            raise ValueError("Active base binding identity must be complete.")
        shadow_id = _optional(
            self.shadow_trajectory_id,
            name="shadow_trajectory_id",
        )
        shadow_hash = _optional(
            self.shadow_trajectory_hash,
            name="shadow_trajectory_hash",
        )
        if (shadow_id is None) != (shadow_hash is None):
            raise ValueError("Shadow trajectory identity must be complete.")
        shadow_keyframe_hash = _optional(
            self.shadow_keyframe_hash,
            name="shadow_keyframe_hash",
        )
        if self.shadow_action is not None:
            if not isinstance(self.shadow_action, ComponentAction):
                raise TypeError("shadow_action must be a ComponentAction.")
            if shadow_id is None or shadow_keyframe_hash is None:
                raise ValueError(
                    "Shadow action requires trajectory and keyframe identity."
                )
        elif shadow_keyframe_hash is not None:
            raise ValueError("Shadow keyframe hash requires a shadow action.")

        corrected = self.corrected_action
        if status == "ready" and corrected is None:
            raise ValueError("A ready shadow result requires a corrected action.")
        if status != "ready" and corrected is not None:
            raise ValueError("Only a ready shadow result may expose correction.")

        payload = {
            "editing_state_hash": state_hash,
            "interaction_mode": interaction_mode,
            "base_lifecycle_status": lifecycle,
            "preparation_hash": self.preparation.semantic_content_hash,
            "publication_basis_hash": self.preparation.publication_basis_hash,
            "output_status": status,
            "reason": reason,
            "nominal_action_hash": (
                None
                if self.nominal_action is None
                else self.nominal_action.semantic_content_hash
            ),
            "corrected_action_hash": (
                None if corrected is None else corrected.semantic_content_hash
            ),
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
            "composition_evaluation_hash": (
                None
                if submission is None
                else submission.composition_evaluation.semantic_content_hash
            ),
            "source_difference_observation_hash": (
                None
                if submission is None
                or submission.source_difference_observation is None
                else submission.source_difference_observation.semantic_content_hash
            ),
            "active_base_binding_id": base_id,
            "active_base_binding_hash": base_hash,
            "shadow_trajectory_id": shadow_id,
            "shadow_trajectory_hash": shadow_hash,
            "shadow_keyframe_hash": shadow_keyframe_hash,
            "shadow_action_hash": (
                None
                if self.shadow_action is None
                else self.shadow_action.semantic_content_hash
            ),
            "proposal_admitted": False,
            "authoritative": False,
            "executable_preview_status": executable_status,
        }
        object.__setattr__(self, "editing_state_hash", state_hash)
        object.__setattr__(self, "interaction_mode", interaction_mode)
        object.__setattr__(self, "base_lifecycle_status", lifecycle)
        object.__setattr__(self, "output_status", status)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "active_base_binding_id", base_id)
        object.__setattr__(self, "active_base_binding_hash", base_hash)
        object.__setattr__(self, "shadow_trajectory_id", shadow_id)
        object.__setattr__(self, "shadow_trajectory_hash", shadow_hash)
        object.__setattr__(self, "shadow_keyframe_hash", shadow_keyframe_hash)
        object.__setattr__(self, "executable_preview_status", executable_status)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    TRAJECTORY_COMPOSITION_SHADOW_RESULT_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def result_id(self) -> str:
        return f"trajectory-composition-shadow:{self.semantic_content_hash}"

    @property
    def nominal_action(self) -> ComponentAction | None:
        return self.preparation.action

    @property
    def corrected_action(self) -> ComponentAction | None:
        if self.output_status != "ready":
            return None
        if self.composition_submission is None:
            return self.preparation.action
        return self.composition_submission.composition_evaluation.composed_candidate

    @property
    def executable_action(self) -> None:
        return None

    @property
    def proposal_admitted(self) -> bool:
        return False

    @property
    def authoritative(self) -> bool:
        return False




def trajectory_composition_shadow_domain_event(
    result: TrajectoryCompositionShadowResult,
    *,
    occurred_at: float,
) -> CompositionDomainEvent:
    """Build one protocol-neutral summary event for an external observer."""

    if not isinstance(result, TrajectoryCompositionShadowResult):
        raise TypeError("result must be a TrajectoryCompositionShadowResult.")
    return CompositionDomainEvent(
        event_kind=TRAJECTORY_COMPOSITION_SHADOW_EVENT_KIND,
        subject=result,
        occurred_at=_finite(occurred_at, name="occurred_at"),
    )


def trajectory_composition_shadow_semantic_payload(
    result: TrajectoryCompositionShadowResult,
) -> dict[str, object]:
    """Return the compact, replay-linkable summary for trace projection.

    Full nominal/residual/corrected actions remain in the ordinary composition
    input/evaluation events.  This summary intentionally avoids duplicating
    high-rate action payloads.
    """

    if not isinstance(result, TrajectoryCompositionShadowResult):
        raise TypeError("result must be a TrajectoryCompositionShadowResult.")
    submission = result.composition_submission
    evaluation = None if submission is None else submission.composition_evaluation
    difference = (
        None if submission is None else submission.source_difference_observation
    )
    sample = result.residual_sample
    return {
        "result_id": result.result_id,
        "editing_state_hash": result.editing_state_hash,
        "interaction_mode": result.interaction_mode,
        "base_lifecycle_status": result.base_lifecycle_status,
        "preparation_hash": result.preparation.semantic_content_hash,
        "target_effective_at": result.preparation.target_effective_at,
        "publication_basis_hash": result.preparation.publication_basis_hash,
        "output_status": result.output_status,
        "reason": result.reason,
        "nominal_action_hash": (
            None
            if result.nominal_action is None
            else result.nominal_action.semantic_content_hash
        ),
        "corrected_action_hash": (
            None
            if result.corrected_action is None
            else result.corrected_action.semantic_content_hash
        ),
        "residual_sample_hash": (
            None if sample is None else sample.semantic_content_hash
        ),
        "residual_weight": None if sample is None else sample.weight,
        "active_base_binding_id": result.active_base_binding_id,
        "active_base_binding_hash": result.active_base_binding_hash,
        "shadow_trajectory_id": result.shadow_trajectory_id,
        "shadow_trajectory_hash": result.shadow_trajectory_hash,
        "shadow_keyframe_hash": result.shadow_keyframe_hash,
        "shadow_action_hash": (
            None
            if result.shadow_action is None
            else result.shadow_action.semantic_content_hash
        ),
        "composition_evaluation_id": (
            None if evaluation is None else evaluation.composition_evaluation_id
        ),
        "composition_evaluation_hash": (
            None if evaluation is None else evaluation.semantic_content_hash
        ),
        "source_difference_observation_id": (
            None if difference is None else difference.observation_id
        ),
        "source_difference_observation_hash": (
            None if difference is None else difference.semantic_content_hash
        ),
        "proposal_admitted": result.proposal_admitted,
        "authoritative": result.authoritative,
        "executable_preview_status": result.executable_preview_status,
        "semantic_content_hash": result.semantic_content_hash,
    }




__all__ = [
    "TRAJECTORY_COMPOSITION_SHADOW_EVENT_KIND",
    "TRAJECTORY_COMPOSITION_SHADOW_RESULT_HASH_DOMAIN_DEVELOPMENT",
    "TRAJECTORY_SHADOW_EXECUTABLE_PREVIEW_UNAVAILABLE",
    "TrajectoryCompositionContributorStamp",
    "TrajectoryCompositionShadowResult",
    "trajectory_composition_shadow_domain_event",
    "trajectory_composition_shadow_semantic_payload",
]
