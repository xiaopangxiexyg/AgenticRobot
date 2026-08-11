"""Default-off MuJoCo binding for allocation-context shadow evaluation.

The allocation reducer, publication transaction, formal composition strategy,
and corrected-horizon builder remain reusable ``control_composition`` owners.
This module only binds those ports to an already-started MuJoCo runtime and
checks that a shadow view consumed receiver-admitted facts from one causal cut.
It exposes no proposal, target, reference, safety, IK, or dispatch operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash

from control_composition.activation import CompositionActivationScope
from control_composition.allocation import (
    AllocationPublicationBasis,
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    CompositionAllocationSpec,
    allocation_publication_control_cut_semantic_hash,
    allocation_publication_outcome_domain_event,
    selected_allocation_modes_from_context,
)
from control_composition.composition import CompositionSpec
from control_composition.eligibility import CompositionSourceEligibilitySpec
from control_composition.events import CompositionDomainEvent
from control_composition.observability import SourceDifferenceSpec
from control_composition.runtime import CompositeProposalSubmission
from control_composition.trajectory import (
    TrajectoryCorrectedPreviewSnapshot,
    TrajectoryPreviewFreshnessEvidence,
)
from robot_motion.action import ComponentAction
from robot_motion.runtime.control_admission import ControlCut
from sim.mujoco.deploy.runtime import MujocoDeployRuntime

from .allocation_publication import (
    StationAllocationPublicationCycle,
    StationAllocationPublicationPort,
)
from .trajectory_corrected_preview import (
    TrajectoryCorrectedPreviewController,
    TrajectoryCorrectedPreviewCycle,
)


MUJOCO_ALLOCATION_SHADOW_PROFILE_OFF = "off"
MUJOCO_ALLOCATION_SHADOW_CYCLE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.control_station.mujoco_allocation_shadow_cycle.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class MujocoAllocationShadowProfile:
    """Application switch for non-authoritative allocation observation."""

    profile_id: str = MUJOCO_ALLOCATION_SHADOW_PROFILE_OFF
    enabled: bool = False
    send_actions: bool = False
    allocation_context_channel: str = "source_allocation"

    def __post_init__(self) -> None:
        profile_id = _required(self.profile_id, name="profile_id")
        channel = _required(
            self.allocation_context_channel,
            name="allocation_context_channel",
        )
        if not isinstance(self.enabled, bool) or not isinstance(
            self.send_actions,
            bool,
        ):
            raise TypeError("enabled and send_actions must be booleans.")
        if self.send_actions:
            raise ValueError("MuJoCo allocation shadow cannot send actions.")
        if self.enabled and profile_id == MUJOCO_ALLOCATION_SHADOW_PROFILE_OFF:
            raise ValueError("An enabled allocation shadow profile needs an id.")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "allocation_context_channel", channel)


@dataclass(frozen=True)
class MujocoAllocationShadowCycle:
    """One same-cut, non-promotable formal composition observation."""

    profile_id: str
    status: str
    control_cut: ControlCut | None = None
    submission: CompositeProposalSubmission | None = None
    selected_context_id: str | None = None
    selected_context_hash: str | None = None
    selected_mode_by_component: Mapping[str, str] = field(default_factory=dict)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        profile_id = _required(self.profile_id, name="profile_id")
        status = _required(self.status, name="status")
        modes = dict(self.selected_mode_by_component)
        if status not in {"disabled", "evaluated"}:
            raise ValueError("Unsupported MuJoCo allocation shadow status.")
        if (self.selected_context_id is None) != (self.selected_context_hash is None):
            raise ValueError("Selected allocation context identity is incomplete.")
        if status == "disabled":
            if (
                self.control_cut is not None
                or self.submission is not None
                or self.selected_context_id is not None
                or modes
            ):
                raise ValueError("A disabled allocation shadow has no facts.")
            cut_hash = None
            submission_hash = None
        else:
            if not isinstance(self.control_cut, ControlCut):
                raise TypeError("An evaluated cycle requires a ControlCut.")
            if not isinstance(self.submission, CompositeProposalSubmission):
                raise TypeError("An evaluated cycle requires a typed submission.")
            if self.submission.proposal_admission is not None:
                raise ValueError("Allocation shadow cannot admit a proposal.")
            input_snapshot = self.submission.input_snapshot
            cut = self.control_cut
            if (
                input_snapshot.control_domain_id != cut.control_domain_id
                or input_snapshot.cutoff_at != cut.cutoff_at
                or input_snapshot.cutoff_admission_sequence
                != cut.cutoff_admission_sequence
            ):
                raise ValueError("Shadow inputs do not belong to the declared cut.")
            context_snapshot = self.submission.context_snapshot
            evaluation = self.submission.composition_evaluation
            if context_snapshot is None:
                if self.selected_context_id is not None:
                    raise ValueError(
                        "Missing context snapshot cannot select allocation."
                    )
            else:
                if (
                    context_snapshot.control_domain_id != cut.control_domain_id
                    or context_snapshot.cutoff_at != cut.cutoff_at
                    or context_snapshot.cutoff_admission_sequence
                    != cut.cutoff_admission_sequence
                ):
                    raise ValueError("Shadow contexts do not belong to the input cut.")
                if self.selected_context_id is None and modes:
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
                            "Context-free shadow modes must be the exact "
                            "strategy-recorded base-only fallback."
                        )
                elif (self.selected_context_id is None) != (not modes):
                    raise ValueError(
                        "Selected allocation context/modes are inconsistent."
                    )
                if self.selected_context_id is not None:
                    matching = tuple(
                        record
                        for record in context_snapshot.selected_context_by_channel.values()
                        if record.context_id == self.selected_context_id
                    )
                    if (
                        len(matching) != 1
                        or matching[0].semantic_content_hash
                        != self.selected_context_hash
                    ):
                        raise ValueError(
                            "Selected allocation context does not match the cut."
                        )
            cut_hash = allocation_publication_control_cut_semantic_hash(cut)
            submission_hash = (
                self.submission.composition_evaluation.semantic_content_hash
            )
        payload = {
            "profile_id": profile_id,
            "status": status,
            "control_cut_hash": cut_hash,
            "composition_evaluation_hash": submission_hash,
            "selected_context_id": self.selected_context_id,
            "selected_context_hash": self.selected_context_hash,
            "selected_mode_by_component": dict(sorted(modes.items())),
            "authoritative": False,
            "promotable": False,
            "send_actions": False,
        }
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(
            self,
            "selected_mode_by_component",
            MappingProxyType(dict(sorted(modes.items()))),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=MUJOCO_ALLOCATION_SHADOW_CYCLE_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def conditional_action(self) -> ComponentAction | None:
        if self.submission is None:
            return None
        return self.submission.composition_evaluation.composed_candidate

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def promotable(self) -> bool:
        return False


class MujocoAllocationShadowController:
    """Bind one allocation publisher to one existing MuJoCo topology."""

    def __init__(
        self,
        *,
        profile: MujocoAllocationShadowProfile,
        runtime: MujocoDeployRuntime | None = None,
        composition_spec: CompositionSpec | None = None,
        allocation_spec: CompositionAllocationSpec | None = None,
        activation_scope: CompositionActivationScope | None = None,
        publication_port: StationAllocationPublicationPort | None = None,
        corrected_preview_controller: (
            TrajectoryCorrectedPreviewController | None
        ) = None,
        source_difference_spec: SourceDifferenceSpec | None = None,
        source_eligibility_spec: CompositionSourceEligibilitySpec | None = None,
    ) -> None:
        if not isinstance(profile, MujocoAllocationShadowProfile):
            raise TypeError("profile must be MujocoAllocationShadowProfile.")
        if profile.enabled:
            if not isinstance(runtime, MujocoDeployRuntime):
                raise TypeError("Enabled shadow requires MujocoDeployRuntime.")
            if not isinstance(composition_spec, CompositionSpec):
                raise TypeError("Enabled shadow requires CompositionSpec.")
            if not isinstance(allocation_spec, CompositionAllocationSpec):
                raise TypeError("Enabled shadow requires CompositionAllocationSpec.")
            if not isinstance(activation_scope, CompositionActivationScope):
                raise TypeError("Enabled shadow requires an activation scope.")
            if not isinstance(publication_port, StationAllocationPublicationPort):
                raise TypeError("Enabled shadow requires a publication port.")
            if not publication_port.profile.enabled:
                raise ValueError("Enabled shadow requires enabled context publication.")
            allocation_runtime = publication_port.allocation_runtime
            assert allocation_runtime is not None
            if (
                allocation_runtime.allocation_spec.semantic_content_hash
                != allocation_spec.semantic_content_hash
                or allocation_runtime.context_channel
                != profile.allocation_context_channel
                or allocation_runtime.activation_scope.activation_id
                != activation_scope.activation_id
                or allocation_runtime.activation_scope.semantic_content_hash
                != activation_scope.semantic_content_hash
                or allocation_runtime.runtime_session_id
                != runtime.config.runtime_session_id
                or allocation_runtime.control_domain_id
                != runtime.config.control_domain_id
            ):
                raise ValueError("Allocation publisher and MuJoCo shadow differ.")
            strategy = runtime.composition_strategy
            if (
                strategy is None
                or strategy.spec.composition_spec_hash
                != composition_spec.semantic_content_hash
                or strategy.spec.context_channels
                != (profile.allocation_context_channel,)
            ):
                raise ValueError("MuJoCo strategy is not the allocation profile.")
        if corrected_preview_controller is not None and not isinstance(
            corrected_preview_controller,
            TrajectoryCorrectedPreviewController,
        ):
            raise TypeError("corrected_preview_controller must be typed or None.")
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
        self.publication_port = publication_port
        self.corrected_preview_controller = corrected_preview_controller
        self.source_difference_spec = source_difference_spec
        self.source_eligibility_spec = source_eligibility_spec

    def start(self):
        if not self.profile.enabled:
            return None
        assert self.publication_port is not None
        return self.publication_port.start()

    def close(self, *, timeout_sec: float = 2.0) -> bool:
        if self.publication_port is None:
            return True
        return self.publication_port.close(
            reason="mujoco_allocation_shadow_closed",
            timeout_sec=timeout_sec,
        )

    def offer_basis(
        self,
        basis_factory: Callable[[], AllocationPublicationBasis],
    ) -> StationAllocationPublicationCycle:
        if not callable(basis_factory):
            raise TypeError("basis_factory must be callable.")
        if not self.profile.enabled:
            return StationAllocationPublicationCycle(
                profile_id=self.profile.profile_id,
                status="disabled",
            )
        assert self.publication_port is not None
        return self.publication_port.offer_basis(basis_factory)

    def request_retry(self) -> StationAllocationPublicationCycle:
        if not self.profile.enabled:
            return StationAllocationPublicationCycle(
                profile_id=self.profile.profile_id,
                status="disabled",
            )
        assert self.publication_port is not None
        return self.publication_port.request_retry()

    def drain_publication_events(self) -> tuple[CompositionDomainEvent, ...]:
        if self.publication_port is None:
            return ()
        return tuple(
            allocation_publication_outcome_domain_event(outcome)
            for outcome in self.publication_port.drain_outcomes()
        )

    def _require_active(self) -> None:
        assert self.publication_port is not None
        if not self.publication_port.activation_active:
            self.publication_port.close(
                reason="mujoco_allocation_shadow_barrier_inactive",
                timeout_sec=0.0,
            )
            raise RuntimeError("MuJoCo allocation shadow activation is inactive.")

    def evaluate_at(
        self,
        *,
        target_effective_at: float,
        cutoff_at: float | None = None,
    ) -> MujocoAllocationShadowCycle:
        """Evaluate one current target without proposal admission or dispatch."""

        if not self.profile.enabled:
            return MujocoAllocationShadowCycle(
                profile_id=self.profile.profile_id,
                status="disabled",
            )
        self._require_active()
        assert self.runtime is not None
        assert self.composition_spec is not None
        assert self.allocation_spec is not None
        assert self.activation_scope is not None
        cut = self.runtime.capture_control_cut(cutoff_at=cutoff_at)
        submission = self.runtime.evaluate_composition_shadow(
            spec=self.composition_spec,
            target_effective_at=float(target_effective_at),
            required_context_channels=(self.profile.allocation_context_channel,),
            control_cut=cut,
            source_difference_spec=self.source_difference_spec,
            activation_scope=self.activation_scope,
            source_eligibility_spec=self.source_eligibility_spec,
            residual_anchor_lifecycle_scope_id=(
                self.publication_port.allocation_runtime.allocation_scope_id
            ),
        )
        if submission.proposal_admission is not None:
            raise RuntimeError("MuJoCo shadow unexpectedly admitted a proposal.")
        context_snapshot = submission.context_snapshot
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
                target_effective_at=float(target_effective_at),
                expected_composition_activation_id=(
                    self.activation_scope.activation_id
                ),
                expected_composition_activation_hash=(
                    self.activation_scope.semantic_content_hash
                ),
            )
        elif submission.composition_evaluation.output_status == "ready":
            modes = dict(
                submission.composition_evaluation.strategy_result_data.get(
                    "allocation_selected_mode_by_component",
                    {},
                )
            )
        return MujocoAllocationShadowCycle(
            profile_id=self.profile.profile_id,
            status="evaluated",
            control_cut=cut,
            submission=submission,
            selected_context_id=(None if context is None else context.context_id),
            selected_context_hash=(
                None if context is None else context.semantic_content_hash
            ),
            selected_mode_by_component=modes,
        )

    def run_corrected_preview(
        self,
        preview_factory: Callable[[], TrajectoryCorrectedPreviewSnapshot],
        *,
        freshness_evidence_factory: (
            Callable[[], TrajectoryPreviewFreshnessEvidence | None] | None
        ) = None,
    ) -> TrajectoryCorrectedPreviewCycle:
        """Validate that a UI preview uses only admitted MuJoCo context facts."""

        if not callable(preview_factory):
            raise TypeError("preview_factory must be callable.")
        if not self.profile.enabled:
            return TrajectoryCorrectedPreviewCycle(
                profile_id=self.profile.profile_id,
                status="disabled",
            )
        self._require_active()
        controller = self.corrected_preview_controller
        if controller is None:
            raise RuntimeError("No corrected-preview controller is configured.")
        cycle = controller.run_preview_cycle(
            preview_factory,
            freshness_evidence_factory=freshness_evidence_factory,
        )
        if cycle.status == "projected":
            assert cycle.preview is not None
            self._validate_corrected_preview(cycle.preview)
        return cycle

    def _validate_corrected_preview(
        self,
        preview: TrajectoryCorrectedPreviewSnapshot,
    ) -> None:
        assert self.runtime is not None
        assert self.allocation_spec is not None
        assert self.activation_scope is not None
        cut = preview.projection.control_cut
        config = self.runtime.config
        if (
            cut.runtime_session_id != config.runtime_session_id
            or cut.control_domain_id != config.control_domain_id
            or cut.clock_domain_id != config.clock_domain_id
        ):
            raise ValueError("Corrected preview belongs to another MuJoCo runtime.")
        if preview.authoritative or preview.promotable:
            raise ValueError("Corrected preview cannot become authoritative.")
        for point in preview.points:
            context_snapshot = point.context_snapshot
            if context_snapshot is None:
                if point.output_status == "ready":
                    raise ValueError("Ready allocation preview has no context cut.")
                continue
            context = context_snapshot.selected_context_by_channel.get(
                self.profile.allocation_context_channel
            )
            if context is None:
                if point.output_status == "ready":
                    raise ValueError("Ready allocation preview has no decision.")
                continue
            admitted = self.runtime.resolve_composition_context(context.context_id)
            if (
                admitted is None
                or admitted.semantic_content_hash != context.semantic_content_hash
            ):
                raise ValueError(
                    "Corrected preview contains a non-admitted allocation context."
                )
            selected_allocation_modes_from_context(
                context,
                expected_allocation_spec_hash=(
                    self.allocation_spec.semantic_content_hash
                ),
                expected_component_ids=self.allocation_spec.component_ids,
                target_effective_at=point.intent_point.target_effective_at,
                expected_composition_activation_id=(
                    self.activation_scope.activation_id
                ),
                expected_composition_activation_hash=(
                    self.activation_scope.semantic_content_hash
                ),
            )


__all__ = [
    "MUJOCO_ALLOCATION_SHADOW_CYCLE_HASH_DOMAIN_DEVELOPMENT",
    "MUJOCO_ALLOCATION_SHADOW_PROFILE_OFF",
    "MujocoAllocationShadowController",
    "MujocoAllocationShadowCycle",
    "MujocoAllocationShadowProfile",
]
