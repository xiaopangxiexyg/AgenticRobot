"""Residual-session and nominal-base lifecycle contracts.

The contracts bind already-admitted source, activation, cut, and anchor facts.
They do not capture devices, advance a source plan, publish a proposal, grant
authority, create a reference, or dispatch motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.trajectory import ComponentActionTrajectory
from robot_motion.runtime.control_admission import ControlCut

from ..activation import CompositionActivationScope
from ..residual_anchor import ResidualAnchorReceipt
from ._validation import (
    _cut_payload,
    _finite,
    _identity_pair,
    _nonnegative_int,
    _required,
)


TRAJECTORY_INTERACTION_MODE_PASS_THROUGH = "pass_through"
TRAJECTORY_INTERACTION_MODE_RESIDUAL_ACTIVE = "residual_active"
TRAJECTORY_INTERACTION_MODE_HAND_BACK = "hand_back"
TRAJECTORY_INTERACTION_MODE_COMPONENT_CONTROL_REQUESTED = (
    "component_control_requested"
)
TRAJECTORY_INTERACTION_MODES_DEVELOPMENT = frozenset(
    {
        TRAJECTORY_INTERACTION_MODE_PASS_THROUGH,
        TRAJECTORY_INTERACTION_MODE_RESIDUAL_ACTIVE,
        TRAJECTORY_INTERACTION_MODE_HAND_BACK,
        TRAJECTORY_INTERACTION_MODE_COMPONENT_CONTROL_REQUESTED,
    }
)

TRAJECTORY_BASE_LIFECYCLE_STABLE = "base_stable"
TRAJECTORY_BASE_LIFECYCLE_REFRESH_PENDING = "base_refresh_pending"
TRAJECTORY_BASE_LIFECYCLE_TRANSITION = "base_transition"
TRAJECTORY_BASE_LIFECYCLE_EXHAUSTED = "base_exhausted"
TRAJECTORY_BASE_LIFECYCLE_STATUSES_DEVELOPMENT = frozenset(
    {
        TRAJECTORY_BASE_LIFECYCLE_STABLE,
        TRAJECTORY_BASE_LIFECYCLE_REFRESH_PENDING,
        TRAJECTORY_BASE_LIFECYCLE_TRANSITION,
        TRAJECTORY_BASE_LIFECYCLE_EXHAUSTED,
    }
)

TRAJECTORY_INTERACTION_BASE_LEGALITY_DEVELOPMENT = MappingProxyType(
    {
        TRAJECTORY_INTERACTION_MODE_PASS_THROUGH: frozenset(
            {TRAJECTORY_BASE_LIFECYCLE_STABLE}
        ),
        TRAJECTORY_INTERACTION_MODE_RESIDUAL_ACTIVE: frozenset(
            TRAJECTORY_BASE_LIFECYCLE_STATUSES_DEVELOPMENT
        ),
        TRAJECTORY_INTERACTION_MODE_HAND_BACK: frozenset(
            {
                TRAJECTORY_BASE_LIFECYCLE_STABLE,
                TRAJECTORY_BASE_LIFECYCLE_TRANSITION,
                TRAJECTORY_BASE_LIFECYCLE_EXHAUSTED,
            }
        ),
        TRAJECTORY_INTERACTION_MODE_COMPONENT_CONTROL_REQUESTED: frozenset(
            {TRAJECTORY_BASE_LIFECYCLE_EXHAUSTED}
        ),
    }
)


NOMINAL_BASE_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.nominal_base_binding.development"
)
RESIDUAL_SESSION_ACTIVATION_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.residual_session_activation_binding.development"
)


def validate_trajectory_interaction_base_state(
    interaction_mode: str,
    base_lifecycle_status: str,
    *,
    allow_component_control_on_stable: bool = False,
) -> None:
    interaction = _required(interaction_mode, name="interaction_mode")
    lifecycle = _required(
        base_lifecycle_status,
        name="base_lifecycle_status",
    )
    if interaction not in TRAJECTORY_INTERACTION_MODES_DEVELOPMENT:
        raise ValueError("Unsupported trajectory interaction mode.")
    if lifecycle not in TRAJECTORY_BASE_LIFECYCLE_STATUSES_DEVELOPMENT:
        raise ValueError("Unsupported trajectory base lifecycle status.")
    profile_exception = (
        interaction
        == TRAJECTORY_INTERACTION_MODE_COMPONENT_CONTROL_REQUESTED
        and lifecycle == TRAJECTORY_BASE_LIFECYCLE_STABLE
        and bool(allow_component_control_on_stable)
    )
    if (
        lifecycle
        not in TRAJECTORY_INTERACTION_BASE_LEGALITY_DEVELOPMENT[interaction]
        and not profile_exception
    ):
        raise ValueError(
            "Illegal trajectory interaction/base lifecycle combination."
        )


@dataclass(frozen=True)
class NominalBaseBinding:
    """One immutable nominal-base revision within a residual session."""

    residual_session_id: str
    binding_revision: int
    trajectory: ComponentActionTrajectory
    active_offer_sequence: int
    bound_for_target_at: float
    predecessor_binding_id: str | None = None
    predecessor_binding_hash: str | None = None
    transition_assessment_id: str | None = None
    transition_assessment_hash: str | None = None
    transition_revalidation_id: str | None = None
    transition_revalidation_hash: str | None = None
    semantic_content_hash: str = field(init=False)
    base_binding_id: str = field(init=False)

    def __post_init__(self) -> None:
        session_id = _required(
            self.residual_session_id,
            name="residual_session_id",
        )
        revision = _nonnegative_int(
            self.binding_revision,
            name="binding_revision",
        )
        if not isinstance(self.trajectory, ComponentActionTrajectory):
            raise TypeError("trajectory must be a ComponentActionTrajectory.")
        if (
            self.trajectory.source_activation_id is None
            or self.trajectory.source_activation_hash is None
        ):
            raise ValueError(
                "A nominal base requires explicit source activation lineage."
            )
        offer_sequence = _nonnegative_int(
            self.active_offer_sequence,
            name="active_offer_sequence",
        )
        bound_for = _finite(
            self.bound_for_target_at,
            name="bound_for_target_at",
        )
        if self.trajectory.keyframe_at(bound_for) is None:
            raise ValueError(
                "bound_for_target_at must be an exact trajectory keyframe."
            )

        predecessor_id, predecessor_hash = _identity_pair(
            self.predecessor_binding_id,
            self.predecessor_binding_hash,
            name="predecessor_binding",
        )
        assessment_id, assessment_hash = _identity_pair(
            self.transition_assessment_id,
            self.transition_assessment_hash,
            name="transition_assessment",
        )
        revalidation_id, revalidation_hash = _identity_pair(
            self.transition_revalidation_id,
            self.transition_revalidation_hash,
            name="transition_revalidation",
        )
        lineage = (
            predecessor_id,
            assessment_id,
            revalidation_id,
        )
        if revision == 0 and any(value is not None for value in lineage):
            raise ValueError(
                "Initial nominal base binding cannot have transition lineage."
            )
        if revision > 0 and any(value is None for value in lineage):
            raise ValueError(
                "Successor nominal base binding requires complete transition lineage."
            )

        payload = {
            "residual_session_id": session_id,
            "binding_revision": revision,
            "trajectory_id": self.trajectory.trajectory_id,
            "trajectory_hash": self.trajectory.semantic_content_hash,
            "active_offer_sequence": offer_sequence,
            "bound_for_target_at": bound_for,
            "effective_from": self.trajectory.effective_from,
            "effective_until": self.trajectory.effective_until,
            "predecessor_binding_id": predecessor_id,
            "predecessor_binding_hash": predecessor_hash,
            "transition_assessment_id": assessment_id,
            "transition_assessment_hash": assessment_hash,
            "transition_revalidation_id": revalidation_id,
            "transition_revalidation_hash": revalidation_hash,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=NOMINAL_BASE_BINDING_HASH_DOMAIN_DEVELOPMENT,
        )
        for name, value in payload.items():
            if hasattr(self, name):
                object.__setattr__(self, name, value)
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "base_binding_id",
            f"nominal-base-binding:{revision}:{semantic_hash}",
        )


@dataclass(frozen=True)
class ResidualSessionActivationBinding:
    """One causal-cut proof for the immutable initial session conditions."""

    residual_session_id: str
    activation_scope: CompositionActivationScope
    control_cut: ControlCut
    residual_anchor_receipt: ResidualAnchorReceipt
    initial_base_binding: NominalBaseBinding
    initial_publication_finalized_target_at: float | None = None
    initial_published_action_hash: str | None = None
    initial_emission_receipt_id: str | None = None
    initial_emission_receipt_hash: str | None = None
    semantic_content_hash: str = field(init=False)
    activation_binding_id: str = field(init=False)

    def __post_init__(self) -> None:
        session_id = _required(
            self.residual_session_id,
            name="residual_session_id",
        )
        if not isinstance(self.activation_scope, CompositionActivationScope):
            raise TypeError(
                "activation_scope must be a CompositionActivationScope."
            )
        cut_payload = _cut_payload(self.control_cut)
        if not isinstance(self.residual_anchor_receipt, ResidualAnchorReceipt):
            raise TypeError(
                "residual_anchor_receipt must be a ResidualAnchorReceipt."
            )
        if not isinstance(self.initial_base_binding, NominalBaseBinding):
            raise TypeError(
                "initial_base_binding must be a NominalBaseBinding."
            )
        base = self.initial_base_binding
        anchor = self.residual_anchor_receipt
        scope = self.activation_scope
        cut = self.control_cut
        if base.binding_revision != 0:
            raise ValueError("Session activation requires base binding revision zero.")
        if base.residual_session_id != session_id:
            raise ValueError("Initial base belongs to another residual session.")
        if (
            anchor.composition_activation_id != scope.activation_id
            or anchor.composition_activation_hash
            != scope.semantic_content_hash
        ):
            raise ValueError(
                "Residual anchor belongs to another composition activation."
            )
        if (
            cut.clock_domain_id != scope.clock_domain_id
            or anchor.clock_domain_id != scope.clock_domain_id
            or base.trajectory.clock_domain_id != scope.clock_domain_id
        ):
            raise ValueError(
                "Session activation facts require one comparable clock domain."
            )
        if cut.cutoff_at < scope.activated_at:
            raise ValueError("Session cut predates composition activation.")
        if anchor.captured_at > cut.cutoff_at:
            raise ValueError("Session cut cannot select a future anchor receipt.")
        if base.trajectory.created_at > cut.cutoff_at:
            raise ValueError("Session cut cannot select a future trajectory.")
        anchor_components = set(anchor.component_binding_by_id)
        if not anchor_components.issubset(set(base.trajectory.component_coverage)):
            raise ValueError(
                "Residual anchor components must be covered by the nominal base."
            )

        barrier_fields = (
            self.initial_publication_finalized_target_at,
            self.initial_published_action_hash,
            self.initial_emission_receipt_id,
            self.initial_emission_receipt_hash,
        )
        if any(value is not None for value in barrier_fields) and not all(
            value is not None for value in barrier_fields
        ):
            raise ValueError(
                "Initial publication barrier lineage must be complete."
            )
        finalized_target = self.initial_publication_finalized_target_at
        published_action_hash = self.initial_published_action_hash
        emission_receipt_id = self.initial_emission_receipt_id
        emission_receipt_hash = self.initial_emission_receipt_hash
        if finalized_target is not None:
            finalized_target = _finite(
                finalized_target,
                name="initial_publication_finalized_target_at",
            )
            published_action_hash = _required(
                published_action_hash,
                name="initial_published_action_hash",
            )
            emission_receipt_id = _required(
                emission_receipt_id,
                name="initial_emission_receipt_id",
            )
            emission_receipt_hash = _required(
                emission_receipt_hash,
                name="initial_emission_receipt_hash",
            )
            if base.bound_for_target_at <= finalized_target:
                raise ValueError(
                    "First editable target must follow the initial publication barrier."
                )

        payload = {
            "residual_session_id": session_id,
            "composition_activation_id": scope.activation_id,
            "composition_activation_hash": scope.semantic_content_hash,
            "control_cut": cut_payload,
            "residual_anchor_receipt_id": anchor.receipt_id,
            "residual_anchor_receipt_hash": anchor.semantic_content_hash,
            "initial_base_binding_id": base.base_binding_id,
            "initial_base_binding_hash": base.semantic_content_hash,
            "component_ids": tuple(sorted(anchor_components)),
            "first_editable_target_at": base.bound_for_target_at,
            "initial_publication_finalized_target_at": finalized_target,
            "initial_published_action_hash": published_action_hash,
            "initial_emission_receipt_id": emission_receipt_id,
            "initial_emission_receipt_hash": emission_receipt_hash,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=(
                RESIDUAL_SESSION_ACTIVATION_BINDING_HASH_DOMAIN_DEVELOPMENT
            ),
        )
        object.__setattr__(self, "residual_session_id", session_id)
        object.__setattr__(
            self,
            "initial_publication_finalized_target_at",
            finalized_target,
        )
        object.__setattr__(
            self,
            "initial_published_action_hash",
            published_action_hash,
        )
        object.__setattr__(
            self,
            "initial_emission_receipt_id",
            emission_receipt_id,
        )
        object.__setattr__(
            self,
            "initial_emission_receipt_hash",
            emission_receipt_hash,
        )
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "activation_binding_id",
            f"residual-session-activation:{semantic_hash}",
        )

    @property
    def first_editable_target_at(self) -> float:
        return self.initial_base_binding.bound_for_target_at

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(self.residual_anchor_receipt.component_binding_by_id)




__all__ = [
    "NOMINAL_BASE_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "RESIDUAL_SESSION_ACTIVATION_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "TRAJECTORY_BASE_LIFECYCLE_EXHAUSTED",
    "TRAJECTORY_BASE_LIFECYCLE_REFRESH_PENDING",
    "TRAJECTORY_BASE_LIFECYCLE_STABLE",
    "TRAJECTORY_BASE_LIFECYCLE_STATUSES_DEVELOPMENT",
    "TRAJECTORY_BASE_LIFECYCLE_TRANSITION",
    "TRAJECTORY_INTERACTION_BASE_LEGALITY_DEVELOPMENT",
    "TRAJECTORY_INTERACTION_MODE_COMPONENT_CONTROL_REQUESTED",
    "TRAJECTORY_INTERACTION_MODE_HAND_BACK",
    "TRAJECTORY_INTERACTION_MODE_PASS_THROUGH",
    "TRAJECTORY_INTERACTION_MODE_RESIDUAL_ACTIVE",
    "TRAJECTORY_INTERACTION_MODES_DEVELOPMENT",
    "NominalBaseBinding",
    "ResidualSessionActivationBinding",
    "validate_trajectory_interaction_base_state",
]
