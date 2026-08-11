"""Advisory nominal-base transition capability and seam assessment.

These contracts compare immutable, already-admitted facts.  They never switch
the active base, publish a proposal, grant authority, or dispatch motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.component_action import ComponentAction
from robot_motion.action.execution_observation import (
    ExecutionMeasurementObservation,
    ExecutionObservationBinding,
    ExecutionObservationCapabilitySpec,
    ExecutionReferenceFrameObservation,
)
from robot_motion.action.trajectory import ComponentActionTrajectory
from robot_motion.runtime.control_admission import ControlCut

from ._validation import (
    _components,
    _cut_payload,
    _finite,
    _identity_pair,
    _nonnegative_int,
    _positive,
    _reasons,
    _required,
)
from .lifecycle import NominalBaseBinding


TRAJECTORY_TRANSITION_CAPABILITY_HOLD_BOUNDARY_ONLY = "hold_boundary_only"
TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_ONLY = "reference_only"
TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_AND_MEASUREMENT = (
    "reference_and_measurement"
)
TRAJECTORY_TRANSITION_CAPABILITY_LEVELS_DEVELOPMENT = frozenset(
    {
        TRAJECTORY_TRANSITION_CAPABILITY_HOLD_BOUNDARY_ONLY,
        TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_ONLY,
        TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_AND_MEASUREMENT,
    }
)

TRAJECTORY_TRANSITION_KIND_HAND_BACK = "hand_back"
TRAJECTORY_TRANSITION_KIND_BASE_REFRESH = "base_refresh"
TRAJECTORY_TRANSITION_KINDS_DEVELOPMENT = frozenset(
    {
        TRAJECTORY_TRANSITION_KIND_HAND_BACK,
        TRAJECTORY_TRANSITION_KIND_BASE_REFRESH,
    }
)

TRAJECTORY_TRANSITION_DECISION_HARD_INCOMPATIBLE = "hard_incompatible"
TRAJECTORY_TRANSITION_DECISION_AUTO_ELIGIBLE = "auto_eligible"
TRAJECTORY_TRANSITION_DECISION_WAIT = "wait_for_newer_shadow"
TRAJECTORY_TRANSITION_DECISION_OPERATOR_REVIEW = "operator_review"
TRAJECTORY_TRANSITION_DECISION_REJECT = "reject_candidate"
TRAJECTORY_TRANSITION_DECISIONS_DEVELOPMENT = frozenset(
    {
        TRAJECTORY_TRANSITION_DECISION_HARD_INCOMPATIBLE,
        TRAJECTORY_TRANSITION_DECISION_AUTO_ELIGIBLE,
        TRAJECTORY_TRANSITION_DECISION_WAIT,
        TRAJECTORY_TRANSITION_DECISION_OPERATOR_REVIEW,
        TRAJECTORY_TRANSITION_DECISION_REJECT,
    }
)


TRAJECTORY_TRANSITION_CAPABILITY_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.trajectory_transition_capability.development"
)
NOMINAL_BASE_TRANSITION_ASSESSMENT_BASIS_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.nominal_base_transition_assessment_basis.development"
)
NOMINAL_BASE_TRANSITION_ASSESSMENT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.nominal_base_transition_assessment.development"
)


@dataclass(frozen=True)
class TrajectoryTransitionCapabilitySpec:
    """Capability/profile gate for automatic base transition assessment."""

    capability_id: str
    level: str
    component_ids: tuple[str, ...]
    allow_automatic_hand_back: bool
    allow_automatic_base_refresh: bool
    observation_binding: ExecutionObservationBinding | None = None
    observation_capability: ExecutionObservationCapabilitySpec | None = None
    maximum_reference_age_sec: float | None = None
    maximum_measurement_age_sec: float | None = None
    reference_tracking_assumption_id: str | None = None
    reference_tracking_assumption_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        capability_id = _required(self.capability_id, name="capability_id")
        level = _required(self.level, name="level")
        if level not in TRAJECTORY_TRANSITION_CAPABILITY_LEVELS_DEVELOPMENT:
            raise ValueError("Unsupported trajectory transition capability level.")
        components = _components(self.component_ids)
        auto_hand_back = bool(self.allow_automatic_hand_back)
        auto_refresh = bool(self.allow_automatic_base_refresh)
        tracking_id, tracking_hash = _identity_pair(
            self.reference_tracking_assumption_id,
            self.reference_tracking_assumption_hash,
            name="reference_tracking_assumption",
        )
        binding = self.observation_binding
        observation_capability = self.observation_capability
        max_reference_age = self.maximum_reference_age_sec
        max_measurement_age = self.maximum_measurement_age_sec

        if level == TRAJECTORY_TRANSITION_CAPABILITY_HOLD_BOUNDARY_ONLY:
            if binding is not None or observation_capability is not None:
                raise ValueError(
                    "HOLD-boundary capability cannot claim observations."
                )
            if auto_hand_back or auto_refresh:
                raise ValueError(
                    "HOLD-boundary capability cannot allow automatic transition."
                )
            if (
                max_reference_age is not None
                or max_measurement_age is not None
                or tracking_id is not None
            ):
                raise ValueError(
                    "HOLD-boundary capability cannot carry observation policy."
                )
        else:
            if not isinstance(binding, ExecutionObservationBinding):
                raise TypeError(
                    "Reference-capable transition requires observation binding."
                )
            if not isinstance(
                observation_capability,
                ExecutionObservationCapabilitySpec,
            ):
                raise TypeError(
                    "Reference-capable transition requires observation capability."
                )
            if (
                binding.capability_id != observation_capability.capability_id
                or binding.capability_hash
                != observation_capability.semantic_content_hash
                or binding.target_handler_id
                != observation_capability.target_handler_id
            ):
                raise ValueError(
                    "Execution observation binding/capability lineage mismatches."
                )
            if binding.reference_mode == "off":
                raise ValueError(
                    "Reference-capable transition requires reference observations."
                )
            if not set(components).issubset(set(binding.component_ids)):
                raise ValueError(
                    "Transition components exceed the observation binding."
                )
            if not set(components).issubset(
                set(observation_capability.representation_by_component)
            ):
                raise ValueError(
                    "Transition components exceed the observation capability."
                )
            max_reference_age = _positive(
                max_reference_age,
                name="maximum_reference_age_sec",
            )
            if level == TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_ONLY:
                if binding.measurement_mode != "off":
                    raise ValueError(
                        "Reference-only capability cannot bind measurements."
                    )
                if max_measurement_age is not None:
                    raise ValueError(
                        "Reference-only capability has no measurement age."
                    )
                if (auto_hand_back or auto_refresh) and tracking_id is None:
                    raise ValueError(
                        "Automatic reference-only transition requires an explicit "
                        "tracking assumption."
                    )
            else:
                if binding.measurement_mode == "off":
                    raise ValueError(
                        "Reference+measurement capability requires measurements."
                    )
                max_measurement_age = _positive(
                    max_measurement_age,
                    name="maximum_measurement_age_sec",
                )

        payload = {
            "capability_id": capability_id,
            "level": level,
            "component_ids": components,
            "allow_automatic_hand_back": auto_hand_back,
            "allow_automatic_base_refresh": auto_refresh,
            "observation_binding_id": (
                None if binding is None else binding.binding_id
            ),
            "observation_binding_hash": (
                None if binding is None else binding.semantic_content_hash
            ),
            "observation_capability_id": (
                None
                if observation_capability is None
                else observation_capability.capability_id
            ),
            "observation_capability_hash": (
                None
                if observation_capability is None
                else observation_capability.semantic_content_hash
            ),
            "maximum_reference_age_sec": max_reference_age,
            "maximum_measurement_age_sec": max_measurement_age,
            "reference_tracking_assumption_id": tracking_id,
            "reference_tracking_assumption_hash": tracking_hash,
        }
        object.__setattr__(self, "capability_id", capability_id)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "component_ids", components)
        object.__setattr__(
            self,
            "allow_automatic_hand_back",
            auto_hand_back,
        )
        object.__setattr__(
            self,
            "allow_automatic_base_refresh",
            auto_refresh,
        )
        object.__setattr__(
            self,
            "maximum_reference_age_sec",
            max_reference_age,
        )
        object.__setattr__(
            self,
            "maximum_measurement_age_sec",
            max_measurement_age,
        )
        object.__setattr__(
            self,
            "reference_tracking_assumption_id",
            tracking_id,
        )
        object.__setattr__(
            self,
            "reference_tracking_assumption_hash",
            tracking_hash,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TRAJECTORY_TRANSITION_CAPABILITY_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    def allows_automatic(self, transition_kind: str) -> bool:
        kind = _required(transition_kind, name="transition_kind")
        if kind == TRAJECTORY_TRANSITION_KIND_HAND_BACK:
            return self.allow_automatic_hand_back
        if kind == TRAJECTORY_TRANSITION_KIND_BASE_REFRESH:
            return self.allow_automatic_base_refresh
        raise ValueError("Unsupported trajectory transition kind.")


def _validate_observation_binding(
    observation: ExecutionReferenceFrameObservation
    | ExecutionMeasurementObservation,
    capability: TrajectoryTransitionCapabilitySpec,
    *,
    cut: ControlCut,
) -> None:
    binding = capability.observation_binding
    assert binding is not None
    if (
        observation.observation_binding_id != binding.binding_id
        or observation.observation_plan_hash != binding.plan_hash
    ):
        raise ValueError("Transition observation binding mismatches capability.")
    if observation.clock_domain_id != cut.clock_domain_id:
        raise ValueError("Transition observation uses another clock domain.")
    if observation.available_at > cut.cutoff_at:
        raise ValueError("Assessment cut cannot select a future observation.")
    if not set(capability.component_ids).issubset(
        set(observation.value_by_component)
    ):
        raise ValueError("Transition observation misses required components.")


@dataclass(frozen=True)
class NominalBaseTransitionAssessment:
    """Advisory, cut-bound assessment; never a future motion permit."""

    transition_kind: str
    residual_session_id: str
    active_base_binding: NominalBaseBinding
    candidate_trajectory: ComponentActionTrajectory
    candidate_offer_sequence: int
    assessment_control_cut: ControlCut
    residual_fact_id: str
    residual_fact_hash: str
    residual_effective_until: float
    switch_at: float
    available_at: float
    publication_basis_hash: str
    transition_capability: TrajectoryTransitionCapabilitySpec
    old_corrected_action: ComponentAction | None
    new_corrected_action: ComponentAction | None
    decision: str
    reason_codes: tuple[str, ...] = ()
    seam_metrics: Mapping[str, float] = field(default_factory=dict)
    reference_observation: ExecutionReferenceFrameObservation | None = None
    measurement_observation: ExecutionMeasurementObservation | None = None
    tracking_assessment_id: str | None = None
    tracking_assessment_hash: str | None = None
    assessment_basis_hash: str = field(init=False)
    semantic_content_hash: str = field(init=False)
    assessment_id: str = field(init=False)

    def __post_init__(self) -> None:
        kind = _required(self.transition_kind, name="transition_kind")
        if kind not in TRAJECTORY_TRANSITION_KINDS_DEVELOPMENT:
            raise ValueError("Unsupported trajectory transition kind.")
        session_id = _required(
            self.residual_session_id,
            name="residual_session_id",
        )
        if not isinstance(self.active_base_binding, NominalBaseBinding):
            raise TypeError("active_base_binding must be a NominalBaseBinding.")
        active = self.active_base_binding
        if active.residual_session_id != session_id:
            raise ValueError("Active base belongs to another residual session.")
        if not isinstance(self.candidate_trajectory, ComponentActionTrajectory):
            raise TypeError(
                "candidate_trajectory must be a ComponentActionTrajectory."
            )
        candidate = self.candidate_trajectory
        candidate_sequence = _nonnegative_int(
            self.candidate_offer_sequence,
            name="candidate_offer_sequence",
        )
        if candidate_sequence <= active.active_offer_sequence:
            raise ValueError("Candidate offer must follow the active base offer.")
        cut_payload = _cut_payload(self.assessment_control_cut)
        cut = self.assessment_control_cut
        if (
            active.trajectory.clock_domain_id != cut.clock_domain_id
            or candidate.clock_domain_id != cut.clock_domain_id
        ):
            raise ValueError("Assessment trajectories use another clock domain.")
        if candidate.created_at > cut.cutoff_at:
            raise ValueError("Assessment cut cannot select a future candidate.")
        if candidate.source_activation_id is None:
            raise ValueError("Candidate requires source activation lineage.")
        switch_at = _finite(self.switch_at, name="switch_at")
        if switch_at <= cut.cutoff_at:
            raise ValueError("Transition switch_at must be in the future.")
        if active.trajectory.keyframe_at(switch_at) is None:
            raise ValueError("Active base does not cover switch_at exactly.")
        if candidate.keyframe_at(switch_at) is None:
            raise ValueError("Candidate does not cover switch_at exactly.")
        residual_id = _required(self.residual_fact_id, name="residual_fact_id")
        residual_hash = _required(
            self.residual_fact_hash,
            name="residual_fact_hash",
        )
        residual_until = _finite(
            self.residual_effective_until,
            name="residual_effective_until",
        )
        if residual_until <= switch_at:
            raise ValueError("Residual fact must remain valid through switch_at.")
        available_at = _finite(self.available_at, name="available_at")
        if available_at < cut.cutoff_at:
            raise ValueError(
                "Transition assessment cannot be available before its basis cut."
            )
        if available_at >= switch_at:
            raise ValueError(
                "Transition assessment must become available before switch_at."
            )
        publication_basis_hash = _required(
            self.publication_basis_hash,
            name="publication_basis_hash",
        )
        if not isinstance(
            self.transition_capability,
            TrajectoryTransitionCapabilitySpec,
        ):
            raise TypeError(
                "transition_capability must be TrajectoryTransitionCapabilitySpec."
            )
        capability = self.transition_capability
        if (
            capability.observation_binding is not None
            and capability.observation_binding.clock_domain_id
            != cut.clock_domain_id
        ):
            raise ValueError(
                "Transition capability uses another assessment clock domain."
            )
        decision = _required(self.decision, name="decision")
        if decision not in TRAJECTORY_TRANSITION_DECISIONS_DEVELOPMENT:
            raise ValueError("Unsupported trajectory transition decision.")
        reasons = _reasons(self.reason_codes)
        if decision in {
            TRAJECTORY_TRANSITION_DECISION_HARD_INCOMPATIBLE,
            TRAJECTORY_TRANSITION_DECISION_REJECT,
        } and not reasons:
            raise ValueError("Rejected transition assessment requires reasons.")

        old_action = self.old_corrected_action
        new_action = self.new_corrected_action
        if (old_action is None) != (new_action is None):
            raise ValueError(
                "Corrected seam endpoints must be both present or both absent."
            )
        if old_action is None:
            if decision != TRAJECTORY_TRANSITION_DECISION_HARD_INCOMPATIBLE:
                raise ValueError(
                    "Only a hard-incompatible assessment may omit corrected seam "
                    "actions."
                )
        else:
            if not isinstance(old_action, ComponentAction) or not isinstance(
                new_action,
                ComponentAction,
            ):
                raise TypeError(
                    "Corrected seam endpoints must be ComponentAction values."
                )
            assert new_action is not None
            if (
                old_action.action_schema.semantic_content_hash
                != new_action.action_schema.semantic_content_hash
                or old_action.component_coverage
                != new_action.component_coverage
            ):
                raise ValueError(
                    "Corrected seam actions must share schema and coverage."
                )
            if not set(capability.component_ids).issubset(
                set(old_action.component_coverage)
            ):
                raise ValueError(
                    "Corrected seam actions miss transition components."
                )
        if decision == TRAJECTORY_TRANSITION_DECISION_AUTO_ELIGIBLE:
            if old_action is None:
                raise ValueError(
                    "Automatic transition requires corrected seam actions."
                )
            if not capability.allows_automatic(kind):
                raise ValueError(
                    "Transition capability does not allow automatic decision."
                )
            if (
                active.trajectory.source_instance_id
                != candidate.source_instance_id
                or active.trajectory.source_activation_id
                != candidate.source_activation_id
                or active.trajectory.source_activation_hash
                != candidate.source_activation_hash
            ):
                raise ValueError(
                    "Automatic transition cannot cross source activation."
                )

        metrics = {
            _required(name, name="seam_metric_name"): _finite(
                value,
                name=f"seam_metrics[{name!r}]",
            )
            for name, value in dict(self.seam_metrics).items()
        }
        reference = self.reference_observation
        measurement = self.measurement_observation
        tracking_id, tracking_hash = _identity_pair(
            self.tracking_assessment_id,
            self.tracking_assessment_hash,
            name="tracking_assessment",
        )
        if reference is not None:
            if not isinstance(reference, ExecutionReferenceFrameObservation):
                raise TypeError(
                    "reference_observation must be ExecutionReferenceFrameObservation."
                )
            _validate_observation_binding(reference, capability, cut=cut)
            covers_switch = reference.covers(switch_at) or (
                reference.temporal_semantics == "point_sample"
                and math.isclose(
                    reference.scheduled_for,
                    switch_at,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            if not covers_switch:
                raise ValueError("Reference observation does not cover switch_at.")
            assert capability.maximum_reference_age_sec is not None
            if (
                cut.cutoff_at - reference.available_at
                > capability.maximum_reference_age_sec
            ):
                raise ValueError("Reference observation is stale at assessment cut.")
        if measurement is not None:
            if not isinstance(measurement, ExecutionMeasurementObservation):
                raise TypeError(
                    "measurement_observation must be ExecutionMeasurementObservation."
                )
            _validate_observation_binding(measurement, capability, cut=cut)
            if measurement.measured_at > cut.cutoff_at:
                raise ValueError("Assessment cut cannot use a future measurement.")
            if capability.maximum_measurement_age_sec is None:
                raise ValueError("Transition capability does not accept measurements.")
            if (
                cut.cutoff_at - measurement.measured_at
                > capability.maximum_measurement_age_sec
            ):
                raise ValueError("Measurement observation is stale at assessment cut.")

        if decision == TRAJECTORY_TRANSITION_DECISION_AUTO_ELIGIBLE:
            if capability.level != (
                TRAJECTORY_TRANSITION_CAPABILITY_HOLD_BOUNDARY_ONLY
            ) and reference is None:
                raise ValueError("Automatic transition requires a seam reference.")
            if capability.level == (
                TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_AND_MEASUREMENT
            ):
                if measurement is None or tracking_id is None:
                    raise ValueError(
                        "Reference+measurement auto transition requires measurement "
                        "and tracking assessment."
                    )

        basis_payload = {
            "transition_kind": kind,
            "residual_session_id": session_id,
            "active_base_binding_id": active.base_binding_id,
            "active_base_binding_hash": active.semantic_content_hash,
            "candidate_trajectory_id": candidate.trajectory_id,
            "candidate_trajectory_hash": candidate.semantic_content_hash,
            "candidate_offer_sequence": candidate_sequence,
            "assessment_control_cut": cut_payload,
            "residual_fact_id": residual_id,
            "residual_fact_hash": residual_hash,
            "residual_effective_until": residual_until,
            "switch_at": switch_at,
            "available_at": available_at,
            "publication_basis_hash": publication_basis_hash,
            "transition_capability_hash": capability.semantic_content_hash,
            "old_corrected_action_hash": (
                None if old_action is None else old_action.semantic_content_hash
            ),
            "new_corrected_action_hash": (
                None if new_action is None else new_action.semantic_content_hash
            ),
            "reference_observation_id": (
                None if reference is None else reference.reference_frame_id
            ),
            "reference_observation_hash": (
                None if reference is None else reference.semantic_content_hash
            ),
            "measurement_observation_id": (
                None if measurement is None else measurement.measurement_id
            ),
            "measurement_observation_hash": (
                None if measurement is None else measurement.semantic_content_hash
            ),
            "tracking_assessment_id": tracking_id,
            "tracking_assessment_hash": tracking_hash,
        }
        basis_hash = canonical_content_hash(
            basis_payload,
            domain=(
                NOMINAL_BASE_TRANSITION_ASSESSMENT_BASIS_HASH_DOMAIN_DEVELOPMENT
            ),
        )
        semantic_hash = canonical_content_hash(
            {
                "assessment_basis_hash": basis_hash,
                "decision": decision,
                "reason_codes": reasons,
                "seam_metrics": metrics,
            },
            domain=NOMINAL_BASE_TRANSITION_ASSESSMENT_HASH_DOMAIN_DEVELOPMENT,
        )
        object.__setattr__(self, "transition_kind", kind)
        object.__setattr__(self, "residual_session_id", session_id)
        object.__setattr__(self, "candidate_offer_sequence", candidate_sequence)
        object.__setattr__(self, "residual_fact_id", residual_id)
        object.__setattr__(self, "residual_fact_hash", residual_hash)
        object.__setattr__(self, "residual_effective_until", residual_until)
        object.__setattr__(self, "switch_at", switch_at)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "publication_basis_hash", publication_basis_hash)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "seam_metrics", MappingProxyType(metrics))
        object.__setattr__(self, "tracking_assessment_id", tracking_id)
        object.__setattr__(self, "tracking_assessment_hash", tracking_hash)
        object.__setattr__(self, "assessment_basis_hash", basis_hash)
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "assessment_id",
            f"nominal-base-transition-assessment:{semantic_hash}",
        )




__all__ = [
    "NOMINAL_BASE_TRANSITION_ASSESSMENT_BASIS_HASH_DOMAIN_DEVELOPMENT",
    "NOMINAL_BASE_TRANSITION_ASSESSMENT_HASH_DOMAIN_DEVELOPMENT",
    "TRAJECTORY_TRANSITION_CAPABILITY_HASH_DOMAIN_DEVELOPMENT",
    "TRAJECTORY_TRANSITION_CAPABILITY_HOLD_BOUNDARY_ONLY",
    "TRAJECTORY_TRANSITION_CAPABILITY_LEVELS_DEVELOPMENT",
    "TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_AND_MEASUREMENT",
    "TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_ONLY",
    "TRAJECTORY_TRANSITION_DECISION_AUTO_ELIGIBLE",
    "TRAJECTORY_TRANSITION_DECISION_HARD_INCOMPATIBLE",
    "TRAJECTORY_TRANSITION_DECISION_OPERATOR_REVIEW",
    "TRAJECTORY_TRANSITION_DECISION_REJECT",
    "TRAJECTORY_TRANSITION_DECISION_WAIT",
    "TRAJECTORY_TRANSITION_DECISIONS_DEVELOPMENT",
    "TRAJECTORY_TRANSITION_KIND_BASE_REFRESH",
    "TRAJECTORY_TRANSITION_KIND_HAND_BACK",
    "TRAJECTORY_TRANSITION_KINDS_DEVELOPMENT",
    "NominalBaseTransitionAssessment",
    "TrajectoryTransitionCapabilitySpec",
]
