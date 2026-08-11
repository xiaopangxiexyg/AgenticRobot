"""System-owned seam correction contract, separate from source residuals."""

from __future__ import annotations

from dataclasses import dataclass, field

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.component_action import ComponentAction

from ._validation import _finite, _required
from .lifecycle import NominalBaseBinding
from .revalidation import NominalBaseTransitionRevalidation
from .transition import NominalBaseTransitionAssessment


TRAJECTORY_CONTINUITY_CORRECTION_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.trajectory_continuity_correction.development"
)


@dataclass(frozen=True)
class TrajectoryContinuityCorrection:
    """System-owned correction activation, distinct from human residual."""

    assessment: NominalBaseTransitionAssessment
    revalidation: NominalBaseTransitionRevalidation
    from_base_binding: NominalBaseBinding
    to_base_binding: NominalBaseBinding
    seam_reference_id: str
    seam_reference_hash: str
    residual_fact_id_at_activation: str
    residual_fact_hash_at_activation: str
    correction_action: ComponentAction
    decay_profile_id: str
    decay_profile_hash: str
    effective_until: float
    semantic_content_hash: str = field(init=False)
    correction_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.assessment, NominalBaseTransitionAssessment):
            raise TypeError("assessment must be NominalBaseTransitionAssessment.")
        if not isinstance(
            self.revalidation,
            NominalBaseTransitionRevalidation,
        ):
            raise TypeError(
                "revalidation must be NominalBaseTransitionRevalidation."
            )
        if not self.revalidation.accepted:
            raise ValueError("Continuity correction requires accepted revalidation.")
        if (
            self.revalidation.assessment_id != self.assessment.assessment_id
            or self.revalidation.assessment_hash
            != self.assessment.semantic_content_hash
        ):
            raise ValueError("Continuity correction assessment lineage mismatches.")
        if not isinstance(self.from_base_binding, NominalBaseBinding) or not isinstance(
            self.to_base_binding,
            NominalBaseBinding,
        ):
            raise TypeError("Continuity correction requires nominal base bindings.")
        if (
            self.from_base_binding.semantic_content_hash
            != self.assessment.active_base_binding.semantic_content_hash
            or self.to_base_binding.predecessor_binding_hash
            != self.from_base_binding.semantic_content_hash
            or self.to_base_binding.trajectory.semantic_content_hash
            != self.assessment.candidate_trajectory.semantic_content_hash
        ):
            raise ValueError("Continuity correction base lineage mismatches.")
        if not isinstance(self.correction_action, ComponentAction):
            raise TypeError("correction_action must be a ComponentAction.")
        for component_id in self.correction_action.component_coverage:
            if (
                self.correction_action.value(component_id).payload_schema.value_semantics
                != "relative_delta"
            ):
                raise ValueError(
                    "Continuity correction action must use relative_delta semantics."
                )
        if not set(self.correction_action.component_coverage).issubset(
            set(self.to_base_binding.trajectory.component_coverage)
        ):
            raise ValueError("Continuity correction exceeds base component coverage.")
        effective_until = _finite(self.effective_until, name="effective_until")
        if effective_until <= self.assessment.switch_at:
            raise ValueError("Continuity correction support must follow switch_at.")
        payload = {
            "assessment_id": self.assessment.assessment_id,
            "assessment_hash": self.assessment.semantic_content_hash,
            "revalidation_id": self.revalidation.revalidation_id,
            "revalidation_hash": self.revalidation.semantic_content_hash,
            "from_base_binding_id": self.from_base_binding.base_binding_id,
            "from_base_binding_hash": self.from_base_binding.semantic_content_hash,
            "to_base_binding_id": self.to_base_binding.base_binding_id,
            "to_base_binding_hash": self.to_base_binding.semantic_content_hash,
            "switch_at": self.assessment.switch_at,
            "seam_reference_id": _required(
                self.seam_reference_id,
                name="seam_reference_id",
            ),
            "seam_reference_hash": _required(
                self.seam_reference_hash,
                name="seam_reference_hash",
            ),
            "residual_fact_id_at_activation": _required(
                self.residual_fact_id_at_activation,
                name="residual_fact_id_at_activation",
            ),
            "residual_fact_hash_at_activation": _required(
                self.residual_fact_hash_at_activation,
                name="residual_fact_hash_at_activation",
            ),
            "correction_action_hash": self.correction_action.semantic_content_hash,
            "decay_profile_id": _required(
                self.decay_profile_id,
                name="decay_profile_id",
            ),
            "decay_profile_hash": _required(
                self.decay_profile_hash,
                name="decay_profile_hash",
            ),
            "effective_until": effective_until,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=TRAJECTORY_CONTINUITY_CORRECTION_HASH_DOMAIN_DEVELOPMENT,
        )
        object.__setattr__(self, "effective_until", effective_until)
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "correction_id",
            f"trajectory-continuity-correction:{semantic_hash}",
        )




__all__ = [
    "TRAJECTORY_CONTINUITY_CORRECTION_HASH_DOMAIN_DEVELOPMENT",
    "TrajectoryContinuityCorrection",
]
