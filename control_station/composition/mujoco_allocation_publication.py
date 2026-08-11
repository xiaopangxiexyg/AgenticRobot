"""Default-off MuJoCo proposal gate for allocation-aware composition.

This concrete Station binding captures one formal receiver cut, evaluates the
allocation-aware composition on that cut, and may admit that exact result as a
proposal.  It does not promote an earlier shadow result and never advances the
lower control loop; authority selection, target resolution, reference
generation, safety, IK, and dispatch remain downstream-owned.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash

from control_composition.activation import CompositionActivationScope
from control_composition.allocation import (
    COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
    COMPOSITION_ALLOCATION_MODES_DEVELOPMENT,
    CompositionAllocationSpec,
    allocation_publication_control_cut_semantic_hash,
    selected_allocation_modes_from_context,
)
from control_composition.composition import CompositionSpec
from control_composition.eligibility import CompositionSourceEligibilitySpec
from control_composition.observability import SourceDifferenceSpec
from control_composition.runtime import CompositeProposalSubmission
from robot_motion.action import ActionProposal, ComponentAction
from robot_motion.runtime.control_admission import ControlCut
from sim.mujoco.deploy.runtime import MujocoDeployRuntime

from ..publication_barrier import StationPublicationBarrier
from .allocation_publication import StationAllocationPublicationPort


MUJOCO_ALLOCATION_PUBLICATION_PROFILE_OFF = "off"
MUJOCO_ALLOCATION_PUBLICATION_CYCLE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.control_station.mujoco_allocation_publication_cycle.development"
)
_ACTIVE_COMPOSITION_BARRIER_REASONS = frozenset(
    {
        "typed_composition_session_completion",
        "typed_composition_session_rearm_completion",
    }
)
MUJOCO_ALLOCATION_PUBLICATION_ALL_MODES = (
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
    COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class MujocoAllocationPublicationProfile:
    """Explicit switch for formal proposal publication; default is inert."""

    profile_id: str = MUJOCO_ALLOCATION_PUBLICATION_PROFILE_OFF
    enabled: bool = False
    send_actions: bool = False
    allocation_context_channel: str = "source_allocation"
    policy_node_id: str = "allocation-composer"
    promotable_allocation_modes: tuple[str, ...] = (
        MUJOCO_ALLOCATION_PUBLICATION_ALL_MODES
    )

    def __post_init__(self) -> None:
        profile_id = _required(self.profile_id, name="profile_id")
        context_channel = _required(
            self.allocation_context_channel,
            name="allocation_context_channel",
        )
        policy_node_id = _required(self.policy_node_id, name="policy_node_id")
        if not isinstance(self.enabled, bool) or not isinstance(
            self.send_actions,
            bool,
        ):
            raise TypeError("enabled and send_actions must be booleans.")
        if self.send_actions and not self.enabled:
            raise ValueError("send_actions=true requires enabled=true.")
        if self.enabled and not self.send_actions:
            raise ValueError(
                "Use allocation shadow evaluation when send_actions=false."
            )
        if self.enabled and profile_id == MUJOCO_ALLOCATION_PUBLICATION_PROFILE_OFF:
            raise ValueError("Enabled publication requires an explicit profile id.")
        promotable_modes = tuple(
            _required(mode, name="promotable_allocation_mode")
            for mode in self.promotable_allocation_modes
        )
        if (
            not promotable_modes
            or len(promotable_modes) != len(set(promotable_modes))
            or any(
                mode not in COMPOSITION_ALLOCATION_MODES_DEVELOPMENT
                for mode in promotable_modes
            )
        ):
            raise ValueError(
                "promotable_allocation_modes must be a non-empty unique subset "
                "of the allocation mode contract."
            )
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "allocation_context_channel", context_channel)
        object.__setattr__(self, "policy_node_id", policy_node_id)
        object.__setattr__(
            self,
            "promotable_allocation_modes",
            promotable_modes,
        )


@dataclass(frozen=True)
class MujocoAllocationPublicationCycle:
    """One formal evaluation and its optional exact proposal admission."""

    profile_id: str
    status: str
    control_cut: ControlCut | None = None
    submission: CompositeProposalSubmission | None = None
    selected_context_id: str | None = None
    selected_context_hash: str | None = None
    selected_mode_by_component: Mapping[str, str] = field(default_factory=dict)
    promotion_blocked_mode_by_component: Mapping[str, str] = field(default_factory=dict)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        profile_id = _required(self.profile_id, name="profile_id")
        status = _required(self.status, name="status")
        if status not in {
            "disabled",
            "no_action",
            "invalid",
            "promotion_blocked",
            "proposal_not_admitted",
            "proposal_admitted",
        }:
            raise ValueError("Unsupported MuJoCo allocation publication status.")
        modes = dict(self.selected_mode_by_component)
        blocked_modes = dict(self.promotion_blocked_mode_by_component)
        context_id = self.selected_context_id
        context_hash = self.selected_context_hash
        if (context_id is None) != (context_hash is None):
            raise ValueError("Selected allocation context identity is incomplete.")
        if status == "disabled":
            if (
                self.control_cut is not None
                or self.submission is not None
                or context_id is not None
                or modes
                or blocked_modes
            ):
                raise ValueError("A disabled publication cycle has no facts.")
            cut_hash = None
            evaluation_hash = None
            receipt = None
        else:
            if not isinstance(self.control_cut, ControlCut):
                raise TypeError("An enabled cycle requires a ControlCut.")
            if not isinstance(self.submission, CompositeProposalSubmission):
                raise TypeError("An enabled cycle requires a typed submission.")
            cut = self.control_cut
            snapshot = self.submission.input_snapshot
            if (
                snapshot.control_domain_id != cut.control_domain_id
                or snapshot.cutoff_at != cut.cutoff_at
                or snapshot.cutoff_admission_sequence != cut.cutoff_admission_sequence
            ):
                raise ValueError("Composition inputs do not belong to the cut.")
            context_snapshot = self.submission.context_snapshot
            if context_snapshot is not None and (
                context_snapshot.control_domain_id != cut.control_domain_id
                or context_snapshot.cutoff_at != cut.cutoff_at
                or context_snapshot.cutoff_admission_sequence
                != cut.cutoff_admission_sequence
            ):
                raise ValueError("Composition contexts do not belong to the cut.")
            evaluation = self.submission.composition_evaluation
            if context_id is None:
                if modes:
                    recorded_modes = dict(
                        evaluation.strategy_result_data.get(
                            "allocation_selected_mode_by_component",
                            {},
                        )
                    )
                    if modes != recorded_modes or set(modes.values()) != {
                        COMPOSITION_ALLOCATION_MODE_BASE_ONLY
                    }:
                        raise ValueError(
                            "Context-free allocation modes must be the exact "
                            "strategy-recorded base-only fallback."
                        )
            elif context_snapshot is None:
                raise ValueError("Selected allocation context is outside the cut.")
            else:
                matching = tuple(
                    record
                    for record in context_snapshot.selected_context_by_channel.values()
                    if record.context_id == context_id
                )
                if (
                    len(matching) != 1
                    or matching[0].semantic_content_hash != context_hash
                    or not modes
                ):
                    raise ValueError("Selected allocation does not match the cut.")
            receipt = self.submission.proposal_admission
            if status in {"no_action", "invalid"}:
                if (
                    evaluation.output_status != status
                    or receipt is not None
                    or blocked_modes
                ):
                    raise ValueError("Non-ready cycle status/evaluation mismatch.")
            elif evaluation.output_status != "ready":
                raise ValueError("Proposal publication requires a ready evaluation.")
            if status == "promotion_blocked":
                if receipt is not None or not blocked_modes:
                    raise ValueError(
                        "Blocked promotion requires ready modes and no proposal receipt."
                    )
                if any(
                    component not in modes or modes[component] != mode
                    for component, mode in blocked_modes.items()
                ):
                    raise ValueError(
                        "Blocked modes must be an exact subset of selected modes."
                    )
            elif blocked_modes:
                raise ValueError(
                    "Only promotion_blocked cycles may carry blocked modes."
                )
            if status == "proposal_admitted":
                if (
                    receipt is None
                    or not receipt.accepted
                    or receipt.proposal is None
                    or receipt.proposal_id is None
                    or receipt.semantic_content_hash is None
                ):
                    raise ValueError("Admitted cycle requires a complete receipt.")
                candidate = evaluation.composed_candidate
                proposal = receipt.proposal
                if (
                    candidate is None
                    or proposal.action.semantic_content_hash
                    != candidate.semantic_content_hash
                    or not math.isclose(
                        proposal.target_effective_at,
                        evaluation.target_effective_at,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError("Proposal does not bind the exact evaluation.")
            elif status == "proposal_not_admitted" and (
                receipt is not None and receipt.accepted
            ):
                raise ValueError("Accepted receipt needs proposal_admitted status.")
            cut_hash = allocation_publication_control_cut_semantic_hash(cut)
            evaluation_hash = evaluation.semantic_content_hash
        payload = {
            "profile_id": profile_id,
            "status": status,
            "control_cut_hash": cut_hash,
            "composition_evaluation_hash": evaluation_hash,
            "selected_context_id": context_id,
            "selected_context_hash": context_hash,
            "selected_mode_by_component": dict(sorted(modes.items())),
            "promotion_blocked_mode_by_component": dict(sorted(blocked_modes.items())),
            "proposal_accepted": None if receipt is None else receipt.accepted,
            "proposal_id": None if receipt is None else receipt.proposal_id,
            "proposal_receipt_hash": (
                None if receipt is None else receipt.semantic_content_hash
            ),
            "proposal_rejection_code": (
                None if receipt is None else receipt.rejection_code
            ),
            "authoritative": False,
            "executed": False,
        }
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(
            self,
            "selected_mode_by_component",
            MappingProxyType(dict(sorted(modes.items()))),
        )
        object.__setattr__(
            self,
            "promotion_blocked_mode_by_component",
            MappingProxyType(dict(sorted(blocked_modes.items()))),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=MUJOCO_ALLOCATION_PUBLICATION_CYCLE_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def proposal(self) -> ActionProposal | None:
        if self.submission is None or self.submission.proposal_admission is None:
            return None
        return self.submission.proposal_admission.proposal

    @property
    def conditional_action(self) -> ComponentAction | None:
        if self.submission is None:
            return None
        return self.submission.composition_evaluation.composed_candidate

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def executed(self) -> bool:
        return False


class MujocoAllocationProposalPublicationController:
    """Publish one same-cut allocation-aware candidate as a lower proposal."""

    def __init__(
        self,
        *,
        profile: MujocoAllocationPublicationProfile,
        runtime: MujocoDeployRuntime | None = None,
        composition_spec: CompositionSpec | None = None,
        allocation_spec: CompositionAllocationSpec | None = None,
        activation_scope: CompositionActivationScope | None = None,
        allocation_publication_port: StationAllocationPublicationPort | None = None,
        publication_barrier: StationPublicationBarrier | None = None,
        source_difference_spec: SourceDifferenceSpec | None = None,
        source_eligibility_spec: CompositionSourceEligibilitySpec | None = None,
    ) -> None:
        if not isinstance(profile, MujocoAllocationPublicationProfile):
            raise TypeError("profile must be MujocoAllocationPublicationProfile.")
        if profile.enabled:
            if not isinstance(runtime, MujocoDeployRuntime):
                raise TypeError("Enabled publication requires MujocoDeployRuntime.")
            if not runtime.config.enable_composer_authority:
                raise ValueError("MuJoCo composer proposal authority is disabled.")
            if not isinstance(composition_spec, CompositionSpec):
                raise TypeError("Enabled publication requires CompositionSpec.")
            if not isinstance(allocation_spec, CompositionAllocationSpec):
                raise TypeError("Enabled publication requires allocation spec.")
            if not isinstance(activation_scope, CompositionActivationScope):
                raise TypeError("Enabled publication requires activation scope.")
            if not isinstance(
                allocation_publication_port,
                StationAllocationPublicationPort,
            ):
                raise TypeError("Enabled publication requires allocation port.")
            if not isinstance(publication_barrier, StationPublicationBarrier):
                raise TypeError("Enabled publication requires Station barrier.")
            if (
                allocation_publication_port.publication_barrier
                is not publication_barrier
            ):
                raise ValueError("Allocation context and proposal barriers differ.")
            if not allocation_publication_port.profile.enabled:
                raise ValueError("Allocation context publication is disabled.")
            allocation_runtime = allocation_publication_port.allocation_runtime
            assert allocation_runtime is not None
            if (
                allocation_runtime.runtime_session_id
                != runtime.config.runtime_session_id
                or allocation_runtime.control_domain_id
                != runtime.config.control_domain_id
                or allocation_runtime.context_channel
                != profile.allocation_context_channel
                or allocation_runtime.allocation_spec.semantic_content_hash
                != allocation_spec.semantic_content_hash
                or allocation_runtime.activation_scope.activation_id
                != activation_scope.activation_id
                or allocation_runtime.activation_scope.semantic_content_hash
                != activation_scope.semantic_content_hash
            ):
                raise ValueError("Allocation publisher and proposal gate differ.")
            strategy = runtime.composition_strategy
            if (
                strategy is None
                or strategy.spec.composition_spec_hash
                != composition_spec.semantic_content_hash
                or strategy.spec.context_channels
                != (profile.allocation_context_channel,)
            ):
                raise ValueError("MuJoCo strategy is not the allocation profile.")
            barrier_snapshot = publication_barrier.snapshot()
            if (
                not barrier_snapshot.enabled
                or barrier_snapshot.authority_operation_result_id is None
                or barrier_snapshot.transition_reason
                not in _ACTIVE_COMPOSITION_BARRIER_REASONS
            ):
                raise RuntimeError(
                    "Enabled proposal publication requires an active typed "
                    "composition-session barrier."
                )
        else:
            barrier_snapshot = None
        if source_difference_spec is not None and not isinstance(
            source_difference_spec,
            SourceDifferenceSpec,
        ):
            raise TypeError("source_difference_spec must be typed or None.")
        if source_eligibility_spec is not None and not isinstance(
            source_eligibility_spec,
            CompositionSourceEligibilitySpec,
        ):
            raise TypeError("source_eligibility_spec must be typed or None.")
        self.profile = profile
        self.runtime = runtime
        self.composition_spec = composition_spec
        self.allocation_spec = allocation_spec
        self.activation_scope = activation_scope
        self.allocation_publication_port = allocation_publication_port
        self.publication_barrier = publication_barrier
        self.source_difference_spec = source_difference_spec
        self.source_eligibility_spec = source_eligibility_spec
        self._barrier_generation = (
            None if barrier_snapshot is None else barrier_snapshot.generation
        )
        self._authority_operation_result_id = (
            None
            if barrier_snapshot is None
            else barrier_snapshot.authority_operation_result_id
        )
        self._lock = threading.RLock()

    def publish_at(
        self,
        *,
        target_effective_at: float,
        cutoff_at: float | None = None,
    ) -> MujocoAllocationPublicationCycle:
        """Evaluate and propose one target without running downstream control."""

        if not self.profile.enabled:
            return MujocoAllocationPublicationCycle(
                profile_id=self.profile.profile_id,
                status="disabled",
            )
        assert self.publication_barrier is not None
        return self.publication_barrier.execute_if_enabled(
            lambda: self._publish_enabled(
                target_effective_at=float(target_effective_at),
                cutoff_at=cutoff_at,
            )
        )

    def _publish_enabled(
        self,
        *,
        target_effective_at: float,
        cutoff_at: float | None,
    ) -> MujocoAllocationPublicationCycle:
        assert self.runtime is not None
        assert self.composition_spec is not None
        assert self.allocation_spec is not None
        assert self.activation_scope is not None
        assert self.allocation_publication_port is not None
        assert self.publication_barrier is not None
        snapshot = self.publication_barrier.snapshot()
        if (
            snapshot.generation != self._barrier_generation
            or snapshot.authority_operation_result_id
            != self._authority_operation_result_id
        ):
            raise RuntimeError(
                "Proposal publisher belongs to another composition activation."
            )
        if not self.allocation_publication_port.activation_active:
            raise RuntimeError("Allocation context publication is not active.")
        with self._lock:
            cut = self.runtime.capture_control_cut(cutoff_at=cutoff_at)
            evaluation = self.runtime.evaluate_composition_shadow(
                spec=self.composition_spec,
                target_effective_at=target_effective_at,
                required_context_channels=(self.profile.allocation_context_channel,),
                control_cut=cut,
                source_difference_spec=self.source_difference_spec,
                activation_scope=self.activation_scope,
                source_eligibility_spec=self.source_eligibility_spec,
                residual_anchor_lifecycle_scope_id=(
                    self.allocation_publication_port.allocation_runtime.allocation_scope_id
                ),
            )
            context_snapshot = evaluation.context_snapshot
            context = (
                None
                if context_snapshot is None
                else context_snapshot.selected_context_by_channel.get(
                    self.profile.allocation_context_channel
                )
            )
            modes: Mapping[str, str] = {}
            if context is not None:
                modes = selected_allocation_modes_from_context(
                    context,
                    expected_allocation_spec_hash=(
                        self.allocation_spec.semantic_content_hash
                    ),
                    expected_component_ids=self.allocation_spec.component_ids,
                    target_effective_at=target_effective_at,
                    expected_composition_activation_id=(
                        self.activation_scope.activation_id
                    ),
                    expected_composition_activation_hash=(
                        self.activation_scope.semantic_content_hash
                    ),
                )
            output_status = evaluation.composition_evaluation.output_status
            if context is None and output_status == "ready":
                modes = dict(
                    evaluation.composition_evaluation.strategy_result_data.get(
                        "allocation_selected_mode_by_component",
                        {},
                    )
                )
            submission = evaluation
            if output_status == "ready":
                promotable = frozenset(self.profile.promotable_allocation_modes)
                blocked_modes = {
                    component_id: mode
                    for component_id, mode in modes.items()
                    if mode not in promotable
                }
                if blocked_modes:
                    status = "promotion_blocked"
                else:
                    submission = self.runtime.submit_evaluated_composition(
                        evaluation,
                        policy_node_id=self.profile.policy_node_id,
                        component_coverage=None,
                        selection_groups=None,
                    )
                    receipt = submission.proposal_admission
                    status = (
                        "proposal_admitted"
                        if receipt is not None and receipt.accepted
                        else "proposal_not_admitted"
                    )
            else:
                status = output_status
                blocked_modes = {}
            return MujocoAllocationPublicationCycle(
                profile_id=self.profile.profile_id,
                status=status,
                control_cut=cut,
                submission=submission,
                selected_context_id=(None if context is None else context.context_id),
                selected_context_hash=(
                    None if context is None else context.semantic_content_hash
                ),
                selected_mode_by_component=modes,
                promotion_blocked_mode_by_component=blocked_modes,
            )


__all__ = [
    "MUJOCO_ALLOCATION_PUBLICATION_CYCLE_HASH_DOMAIN_DEVELOPMENT",
    "MUJOCO_ALLOCATION_PUBLICATION_ALL_MODES",
    "MUJOCO_ALLOCATION_PUBLICATION_PROFILE_OFF",
    "MujocoAllocationProposalPublicationController",
    "MujocoAllocationPublicationCycle",
    "MujocoAllocationPublicationProfile",
]
