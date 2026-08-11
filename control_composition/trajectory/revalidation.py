"""Final transition revalidation and immutable successor construction."""

from __future__ import annotations

from dataclasses import dataclass, field

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.execution_observation import (
    ExecutionMeasurementObservation,
    ExecutionReferenceFrameObservation,
)
from robot_motion.action.trajectory import ComponentActionTrajectory
from robot_motion.runtime.control_admission import ControlCut

from ._validation import (
    _cut_payload,
    _finite,
    _identity_pair,
    _nonnegative_int,
    _optional,
    _reasons,
    _required,
)
from .lifecycle import NominalBaseBinding
from .transition import (
    TRAJECTORY_TRANSITION_DECISION_AUTO_ELIGIBLE,
    NominalBaseTransitionAssessment,
    TrajectoryTransitionCapabilitySpec,
)


NOMINAL_BASE_TRANSITION_REVALIDATION_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.nominal_base_transition_revalidation.development"
)


@dataclass(frozen=True)
class NominalBaseTransitionRevalidation:
    """One final pure staleness/barrier check immediately before switch."""

    assessment_id: str
    assessment_hash: str
    checked_control_cut: ControlCut
    accepted: bool
    reason_codes: tuple[str, ...]
    current_publication_basis_hash: str
    current_publication_finalized_target_at: float | None
    current_residual_fact_id: str
    current_residual_fact_hash: str
    current_candidate_trajectory_hash: str
    current_candidate_offer_sequence: int
    current_base_binding_hash: str
    current_capability_hash: str
    current_reference_observation_hash: str | None
    current_measurement_observation_hash: str | None
    current_tracking_assessment_hash: str | None
    semantic_content_hash: str = field(init=False)
    revalidation_id: str = field(init=False)

    def __post_init__(self) -> None:
        reasons = _reasons(self.reason_codes)
        if self.accepted and reasons:
            raise ValueError("Accepted revalidation cannot carry reasons.")
        if not self.accepted and not reasons:
            raise ValueError("Rejected revalidation requires reasons.")
        payload = {
            "assessment_id": _required(
                self.assessment_id,
                name="assessment_id",
            ),
            "assessment_hash": _required(
                self.assessment_hash,
                name="assessment_hash",
            ),
            "checked_control_cut": _cut_payload(self.checked_control_cut),
            "accepted": bool(self.accepted),
            "reason_codes": reasons,
            "current_publication_basis_hash": _required(
                self.current_publication_basis_hash,
                name="current_publication_basis_hash",
            ),
            "current_publication_finalized_target_at": (
                None
                if self.current_publication_finalized_target_at is None
                else _finite(
                    self.current_publication_finalized_target_at,
                    name="current_publication_finalized_target_at",
                )
            ),
            "current_residual_fact_id": _required(
                self.current_residual_fact_id,
                name="current_residual_fact_id",
            ),
            "current_residual_fact_hash": _required(
                self.current_residual_fact_hash,
                name="current_residual_fact_hash",
            ),
            "current_candidate_trajectory_hash": _required(
                self.current_candidate_trajectory_hash,
                name="current_candidate_trajectory_hash",
            ),
            "current_candidate_offer_sequence": _nonnegative_int(
                self.current_candidate_offer_sequence,
                name="current_candidate_offer_sequence",
            ),
            "current_base_binding_hash": _required(
                self.current_base_binding_hash,
                name="current_base_binding_hash",
            ),
            "current_capability_hash": _required(
                self.current_capability_hash,
                name="current_capability_hash",
            ),
            "current_reference_observation_hash": _optional(
                self.current_reference_observation_hash,
                name="current_reference_observation_hash",
            ),
            "current_measurement_observation_hash": _optional(
                self.current_measurement_observation_hash,
                name="current_measurement_observation_hash",
            ),
            "current_tracking_assessment_hash": _optional(
                self.current_tracking_assessment_hash,
                name="current_tracking_assessment_hash",
            ),
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=(
                NOMINAL_BASE_TRANSITION_REVALIDATION_HASH_DOMAIN_DEVELOPMENT
            ),
        )
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "revalidation_id",
            f"nominal-base-transition-revalidation:{semantic_hash}",
        )


def revalidate_nominal_base_transition(
    assessment: NominalBaseTransitionAssessment,
    *,
    checked_control_cut: ControlCut,
    current_base_binding: NominalBaseBinding,
    current_candidate_trajectory: ComponentActionTrajectory,
    current_candidate_offer_sequence: int,
    current_residual_fact_id: str,
    current_residual_fact_hash: str,
    current_residual_effective_until: float,
    current_publication_basis_hash: str,
    current_publication_finalized_target_at: float | None,
    current_capability: TrajectoryTransitionCapabilitySpec,
    current_reference_observation: ExecutionReferenceFrameObservation | None = None,
    current_measurement_observation: ExecutionMeasurementObservation | None = None,
    current_tracking_assessment_id: str | None = None,
    current_tracking_assessment_hash: str | None = None,
) -> NominalBaseTransitionRevalidation:
    """Revalidate an advisory assessment without recomputing its seam."""

    if not isinstance(assessment, NominalBaseTransitionAssessment):
        raise TypeError(
            "assessment must be a NominalBaseTransitionAssessment."
        )
    if not isinstance(current_base_binding, NominalBaseBinding):
        raise TypeError("current_base_binding must be a NominalBaseBinding.")
    if not isinstance(current_candidate_trajectory, ComponentActionTrajectory):
        raise TypeError(
            "current_candidate_trajectory must be ComponentActionTrajectory."
        )
    if not isinstance(
        current_capability,
        TrajectoryTransitionCapabilitySpec,
    ):
        raise TypeError(
            "current_capability must be TrajectoryTransitionCapabilitySpec."
        )
    _cut_payload(checked_control_cut)
    reasons: list[str] = []
    if (
        checked_control_cut.runtime_session_id
        != assessment.assessment_control_cut.runtime_session_id
        or checked_control_cut.control_domain_id
        != assessment.assessment_control_cut.control_domain_id
        or checked_control_cut.clock_domain_id
        != assessment.assessment_control_cut.clock_domain_id
        or checked_control_cut.clock_reset_epoch
        != assessment.assessment_control_cut.clock_reset_epoch
    ):
        reasons.append("control_cut_domain_changed")
    if (
        checked_control_cut.cutoff_at
        < assessment.assessment_control_cut.cutoff_at
        or checked_control_cut.cutoff_admission_sequence
        < assessment.assessment_control_cut.cutoff_admission_sequence
    ):
        reasons.append("control_cut_rollback")
    if checked_control_cut.cutoff_at < assessment.available_at:
        reasons.append("assessment_not_yet_visible")
    if checked_control_cut.cutoff_at >= assessment.switch_at:
        reasons.append("switch_boundary_reached")
    if (
        current_base_binding.semantic_content_hash
        != assessment.active_base_binding.semantic_content_hash
    ):
        reasons.append("active_base_binding_changed")
    candidate_sequence = _nonnegative_int(
        current_candidate_offer_sequence,
        name="current_candidate_offer_sequence",
    )
    if (
        current_candidate_trajectory.semantic_content_hash
        != assessment.candidate_trajectory.semantic_content_hash
        or candidate_sequence != assessment.candidate_offer_sequence
    ):
        reasons.append("candidate_shadow_changed")
    residual_id = _required(
        current_residual_fact_id,
        name="current_residual_fact_id",
    )
    residual_hash = _required(
        current_residual_fact_hash,
        name="current_residual_fact_hash",
    )
    residual_until = _finite(
        current_residual_effective_until,
        name="current_residual_effective_until",
    )
    if (
        residual_id != assessment.residual_fact_id
        or residual_hash != assessment.residual_fact_hash
    ):
        reasons.append("residual_basis_changed_requires_reassessment")
    if residual_until <= assessment.switch_at:
        reasons.append("residual_stale_before_switch")
    publication_basis_hash = _required(
        current_publication_basis_hash,
        name="current_publication_basis_hash",
    )
    finalized_target = current_publication_finalized_target_at
    if finalized_target is not None:
        finalized_target = _finite(
            finalized_target,
            name="current_publication_finalized_target_at",
        )
        if finalized_target >= assessment.switch_at:
            reasons.append("publication_barrier_reached_switch")
    if (
        current_capability.semantic_content_hash
        != assessment.transition_capability.semantic_content_hash
    ):
        reasons.append("transition_capability_changed")

    reference_hash = (
        None
        if current_reference_observation is None
        else current_reference_observation.semantic_content_hash
    )
    expected_reference_hash = (
        None
        if assessment.reference_observation is None
        else assessment.reference_observation.semantic_content_hash
    )
    if reference_hash != expected_reference_hash:
        reasons.append("reference_observation_changed")
    measurement_hash = (
        None
        if current_measurement_observation is None
        else current_measurement_observation.semantic_content_hash
    )
    expected_measurement_hash = (
        None
        if assessment.measurement_observation is None
        else assessment.measurement_observation.semantic_content_hash
    )
    if measurement_hash != expected_measurement_hash:
        reasons.append("measurement_observation_changed")
    tracking_id, tracking_hash = _identity_pair(
        current_tracking_assessment_id,
        current_tracking_assessment_hash,
        name="current_tracking_assessment",
    )
    if (
        tracking_id != assessment.tracking_assessment_id
        or tracking_hash != assessment.tracking_assessment_hash
    ):
        reasons.append("tracking_assessment_changed")
    if current_reference_observation is not None:
        max_age = current_capability.maximum_reference_age_sec
        if max_age is None:
            reasons.append("reference_not_supported_by_current_capability")
        elif (
            checked_control_cut.cutoff_at
            - current_reference_observation.available_at
            > max_age
        ):
            reasons.append("reference_observation_stale")
    if current_measurement_observation is not None:
        max_age = current_capability.maximum_measurement_age_sec
        if max_age is None:
            reasons.append("measurement_not_supported_by_current_capability")
        elif (
            checked_control_cut.cutoff_at
            - current_measurement_observation.measured_at
            > max_age
        ):
            reasons.append("measurement_observation_stale")

    return NominalBaseTransitionRevalidation(
        assessment_id=assessment.assessment_id,
        assessment_hash=assessment.semantic_content_hash,
        checked_control_cut=checked_control_cut,
        accepted=not reasons,
        reason_codes=tuple(reasons),
        current_publication_basis_hash=publication_basis_hash,
        current_publication_finalized_target_at=finalized_target,
        current_residual_fact_id=residual_id,
        current_residual_fact_hash=residual_hash,
        current_candidate_trajectory_hash=(
            current_candidate_trajectory.semantic_content_hash
        ),
        current_candidate_offer_sequence=candidate_sequence,
        current_base_binding_hash=current_base_binding.semantic_content_hash,
        current_capability_hash=current_capability.semantic_content_hash,
        current_reference_observation_hash=reference_hash,
        current_measurement_observation_hash=measurement_hash,
        current_tracking_assessment_hash=tracking_hash,
    )


def build_successor_nominal_base_binding(
    assessment: NominalBaseTransitionAssessment,
    revalidation: NominalBaseTransitionRevalidation,
) -> NominalBaseBinding:
    """Build, but do not activate, one immutable successor base binding."""

    if not isinstance(assessment, NominalBaseTransitionAssessment):
        raise TypeError(
            "assessment must be a NominalBaseTransitionAssessment."
        )
    if not isinstance(revalidation, NominalBaseTransitionRevalidation):
        raise TypeError(
            "revalidation must be a NominalBaseTransitionRevalidation."
        )
    if not revalidation.accepted:
        raise ValueError("A rejected transition cannot build a successor base.")
    if assessment.decision != TRAJECTORY_TRANSITION_DECISION_AUTO_ELIGIBLE:
        raise ValueError(
            "A non-automatic assessment requires a separate operator or "
            "HOLD-boundary transition receipt."
        )
    if (
        revalidation.assessment_id != assessment.assessment_id
        or revalidation.assessment_hash != assessment.semantic_content_hash
    ):
        raise ValueError("Transition revalidation belongs to another assessment.")
    predecessor = assessment.active_base_binding
    return NominalBaseBinding(
        residual_session_id=assessment.residual_session_id,
        binding_revision=predecessor.binding_revision + 1,
        trajectory=assessment.candidate_trajectory,
        active_offer_sequence=assessment.candidate_offer_sequence,
        bound_for_target_at=assessment.switch_at,
        predecessor_binding_id=predecessor.base_binding_id,
        predecessor_binding_hash=predecessor.semantic_content_hash,
        transition_assessment_id=assessment.assessment_id,
        transition_assessment_hash=assessment.semantic_content_hash,
        transition_revalidation_id=revalidation.revalidation_id,
        transition_revalidation_hash=revalidation.semantic_content_hash,
    )
__all__ = [
    "NOMINAL_BASE_TRANSITION_REVALIDATION_HASH_DOMAIN_DEVELOPMENT",
    "NominalBaseTransitionRevalidation",
    "build_successor_nominal_base_binding",
    "revalidate_nominal_base_transition",
]
