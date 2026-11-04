"""Frozen, non-closing trajectory-to-composition input projection.

The capture step reads immutable facts from one receiver cut and then detaches
from the runtime.  Later target projections use only this object: no store is
admitted to, no cursor advances, and no target watermark closes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.component_action import ComponentAction
from robot_motion.runtime.control_admission import ControlCut
from robot_motion.runtime.producer_epoch import ProducerEpochSnapshot

from control_composition.composition import (
    CompositionContextProjection,
    CompositionSpec,
    ProjectedCompositionStrategy,
    build_composition_context_snapshot,
    frozen_control_cut_semantic_hash,
)
from control_composition.eligibility import (
    CompositionSourceEligibilityProjection,
    CompositionSourceEligibilitySpec,
    build_composition_source_eligibility_snapshot,
)
from control_composition.information import CompositionInformationPlan
from control_composition.residual_anchor_lifecycle import (
    ResidualAnchorLifecycleProjection,
    ResidualAnchorLifecycleSpec,
    build_residual_anchor_lifecycle_snapshot,
)
from control_composition.runtime import ControlCompositionRuntime
from control_composition.temporal import CompositionTemporalSelectionSpec

from ._validation import _finite, _required
from .contracts import (
    TrajectoryEditingState,
)
from .shadow_contracts import TrajectoryCompositionContributorStamp
from .warp import ResidualTemporalWarpSpec


FROZEN_TRAJECTORY_HELD_INPUT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.frozen_trajectory_held_input.development"
)
FROZEN_TRAJECTORY_RESIDUAL_INPUT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.frozen_trajectory_residual_input.development"
)
FROZEN_TRAJECTORY_COMPOSITION_PROJECTION_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.frozen_trajectory_composition_projection.development"
)


def _lineage(
    *pairs: tuple[tuple[str, ...], tuple[str, ...]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    hashes_by_id: dict[str, str] = {}
    ordered_ids: list[str] = []
    for ids, hashes in pairs:
        if len(ids) != len(hashes):
            raise ValueError("Source lineage id/hash lengths differ.")
        for raw_id, raw_hash in zip(ids, hashes):
            fact_id = _required(raw_id, name="source_fact_id")
            fact_hash = _required(raw_hash, name="source_fact_hash")
            previous = hashes_by_id.get(fact_id)
            if previous is not None and previous != fact_hash:
                raise ValueError("One source fact id maps to two hashes.")
            if previous is None:
                hashes_by_id[fact_id] = fact_hash
                ordered_ids.append(fact_id)
    if not ordered_ids:
        raise ValueError("Projected source lineage must be non-empty.")
    return tuple(ordered_ids), tuple(
        hashes_by_id[fact_id] for fact_id in ordered_ids
    )


def _stamp_payload(
    stamp: TrajectoryCompositionContributorStamp,
) -> dict[str, object]:
    return {
        "producer_id": stamp.producer_id,
        "producer_epoch": stamp.producer_epoch,
        "sequence": stamp.sequence,
        "produced_at": stamp.produced_at,
        "produced_clock_domain_id": stamp.produced_clock_domain_id,
        "context_ids": stamp.context_ids,
        "observation_cut_id": stamp.observation_cut_id,
        "policy_output_id": stamp.policy_output_id,
    }


@dataclass(frozen=True)
class FrozenTrajectoryHeldInput:
    """One immutable state-valued role held over an explicit support interval."""

    input_role: str
    action: ComponentAction
    contributor_stamp: TrajectoryCompositionContributorStamp
    effective_from: float
    effective_until: float
    source_kind: str
    source_fact_ids: tuple[str, ...]
    source_fact_hashes: tuple[str, ...]
    semantic_binding_id_by_kind: Mapping[str, str] = field(default_factory=dict)
    semantic_binding_hash_by_kind: Mapping[str, str] = field(default_factory=dict)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        role = _required(self.input_role, name="input_role")
        if not isinstance(self.action, ComponentAction):
            raise TypeError("action must be a ComponentAction.")
        if not isinstance(
            self.contributor_stamp,
            TrajectoryCompositionContributorStamp,
        ):
            raise TypeError(
                "contributor_stamp must be TrajectoryCompositionContributorStamp."
            )
        effective_from = _finite(
            self.effective_from,
            name="effective_from",
        )
        effective_until = _finite(
            self.effective_until,
            name="effective_until",
        )
        if effective_until <= effective_from:
            raise ValueError("Frozen held-input interval must be non-empty.")
        source_kind = _required(self.source_kind, name="source_kind")
        fact_ids, fact_hashes = _lineage(
            (tuple(self.source_fact_ids), tuple(self.source_fact_hashes))
        )
        binding_ids = {
            _required(kind, name="semantic_binding_kind"): _required(
                value,
                name="semantic_binding_id",
            )
            for kind, value in dict(
                self.semantic_binding_id_by_kind
            ).items()
        }
        binding_hashes = {
            _required(kind, name="semantic_binding_kind"): _required(
                value,
                name="semantic_binding_hash",
            )
            for kind, value in dict(
                self.semantic_binding_hash_by_kind
            ).items()
        }
        if set(binding_ids) != set(binding_hashes):
            raise ValueError("Frozen held-input semantic binding kinds must match.")
        payload = {
            "input_role": role,
            "action_hash": self.action.semantic_content_hash,
            "contributor_stamp": _stamp_payload(self.contributor_stamp),
            "effective_from": effective_from,
            "effective_until": effective_until,
            "source_kind": source_kind,
            "source_fact_ids": fact_ids,
            "source_fact_hashes": fact_hashes,
            **(
                {
                    "semantic_binding_id_by_kind": dict(
                        sorted(binding_ids.items())
                    ),
                    "semantic_binding_hash_by_kind": dict(
                        sorted(binding_hashes.items())
                    ),
                }
                if binding_ids
                else {}
            ),
        }
        object.__setattr__(self, "input_role", role)
        object.__setattr__(self, "effective_from", effective_from)
        object.__setattr__(self, "effective_until", effective_until)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "source_fact_ids", fact_ids)
        object.__setattr__(self, "source_fact_hashes", fact_hashes)
        object.__setattr__(
            self,
            "semantic_binding_id_by_kind",
            MappingProxyType(dict(sorted(binding_ids.items()))),
        )
        object.__setattr__(
            self,
            "semantic_binding_hash_by_kind",
            MappingProxyType(dict(sorted(binding_hashes.items()))),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=FROZEN_TRAJECTORY_HELD_INPUT_HASH_DOMAIN_DEVELOPMENT,
            ),
        )


@dataclass(frozen=True)
class FrozenTrajectoryResidualInput:
    """One frozen residual fact sampled against each exact nominal keyframe."""

    input_role: str
    action: ComponentAction
    contributor_stamp: TrajectoryCompositionContributorStamp
    residual_fact_id: str
    residual_fact_hash: str
    residual_anchor_receipt_id: str
    residual_anchor_receipt_hash: str
    effective_until: float
    warp_spec: ResidualTemporalWarpSpec
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        role = _required(self.input_role, name="input_role")
        if not isinstance(self.action, ComponentAction):
            raise TypeError("action must be a ComponentAction.")
        if not isinstance(
            self.contributor_stamp,
            TrajectoryCompositionContributorStamp,
        ):
            raise TypeError(
                "contributor_stamp must be TrajectoryCompositionContributorStamp."
            )
        residual_fact_id = _required(
            self.residual_fact_id,
            name="residual_fact_id",
        )
        residual_fact_hash = _required(
            self.residual_fact_hash,
            name="residual_fact_hash",
        )
        anchor_id = _required(
            self.residual_anchor_receipt_id,
            name="residual_anchor_receipt_id",
        )
        anchor_hash = _required(
            self.residual_anchor_receipt_hash,
            name="residual_anchor_receipt_hash",
        )
        effective_until = _finite(
            self.effective_until,
            name="effective_until",
        )
        if not isinstance(self.warp_spec, ResidualTemporalWarpSpec):
            raise TypeError("warp_spec must be ResidualTemporalWarpSpec.")
        payload = {
            "input_role": role,
            "action_hash": self.action.semantic_content_hash,
            "contributor_stamp": _stamp_payload(self.contributor_stamp),
            "residual_fact_id": residual_fact_id,
            "residual_fact_hash": residual_fact_hash,
            "residual_anchor_receipt_id": anchor_id,
            "residual_anchor_receipt_hash": anchor_hash,
            "effective_until": effective_until,
            "warp_spec_hash": self.warp_spec.semantic_content_hash,
        }
        object.__setattr__(self, "input_role", role)
        object.__setattr__(self, "residual_fact_id", residual_fact_id)
        object.__setattr__(self, "residual_fact_hash", residual_fact_hash)
        object.__setattr__(self, "residual_anchor_receipt_id", anchor_id)
        object.__setattr__(self, "residual_anchor_receipt_hash", anchor_hash)
        object.__setattr__(self, "effective_until", effective_until)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    FROZEN_TRAJECTORY_RESIDUAL_INPUT_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )


@dataclass(frozen=True)
class FrozenTrajectoryCompositionProjection:
    """Complete immutable basis for all target evaluations in one horizon."""

    state: TrajectoryEditingState
    composition_spec: CompositionSpec
    strategy: ProjectedCompositionStrategy = field(repr=False, compare=False)
    nominal_role: str = ""
    nominal_stamp: TrajectoryCompositionContributorStamp | None = None
    residual_input: FrozenTrajectoryResidualInput | None = None
    held_inputs: tuple[FrozenTrajectoryHeldInput, ...] = ()
    control_cut: ControlCut | None = None
    producer_epochs: ProducerEpochSnapshot | None = None
    context_projection: CompositionContextProjection | None = None
    source_eligibility_projection: (
        CompositionSourceEligibilityProjection | None
    ) = None
    source_eligibility_spec: CompositionSourceEligibilitySpec | None = None
    residual_anchor_lifecycle_projection: (
        ResidualAnchorLifecycleProjection | None
    ) = None
    residual_anchor_lifecycle_spec: ResidualAnchorLifecycleSpec | None = None
    residual_anchor_lifecycle_scope_id: str | None = None
    temporal_selection_spec: CompositionTemporalSelectionSpec | None = None
    information_plan: CompositionInformationPlan | None = None
    basis_target_effective_at: float = 0.0
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.state, TrajectoryEditingState):
            raise TypeError("state must be TrajectoryEditingState.")
        if not isinstance(self.composition_spec, CompositionSpec):
            raise TypeError("composition_spec must be CompositionSpec.")
        strategy_spec = getattr(self.strategy, "spec", None)
        if strategy_spec is None or strategy_spec.composition_spec_hash != (
            self.composition_spec.semantic_content_hash
        ):
            raise ValueError("Projected strategy and CompositionSpec differ.")
        if not callable(getattr(self.strategy, "decide_projected", None)):
            raise TypeError("strategy must explicitly support projection.")
        nominal_role = _required(self.nominal_role, name="nominal_role")
        if nominal_role not in self.composition_spec.input_roles:
            raise ValueError("nominal_role is not declared by CompositionSpec.")
        if not isinstance(
            self.nominal_stamp,
            TrajectoryCompositionContributorStamp,
        ):
            raise TypeError("nominal_stamp must be a contributor stamp.")
        held_inputs = tuple(self.held_inputs)
        if any(
            not isinstance(item, FrozenTrajectoryHeldInput)
            for item in held_inputs
        ):
            raise TypeError("held_inputs must contain FrozenTrajectoryHeldInput.")
        residual = self.residual_input
        if residual is not None and not isinstance(
            residual,
            FrozenTrajectoryResidualInput,
        ):
            raise TypeError(
                "residual_input must be FrozenTrajectoryResidualInput or None."
            )
        contributor_roles = (
            nominal_role,
            *((residual.input_role,) if residual is not None else ()),
            *(item.input_role for item in held_inputs),
        )
        if len(contributor_roles) != len(set(contributor_roles)):
            raise ValueError("Frozen projection contributor roles must be unique.")
        if not set(contributor_roles).issubset(
            set(self.composition_spec.input_roles)
        ):
            raise ValueError("Frozen projection contains undeclared roles.")
        if not isinstance(self.control_cut, ControlCut):
            raise TypeError("control_cut must be ControlCut.")
        if not isinstance(self.producer_epochs, ProducerEpochSnapshot):
            raise TypeError("producer_epochs must be ProducerEpochSnapshot.")
        cut = self.control_cut
        epochs = self.producer_epochs
        if (
            epochs.control_domain_id != cut.control_domain_id
            or epochs.cutoff_at != cut.cutoff_at
            or epochs.cutoff_admission_sequence
            != cut.cutoff_admission_sequence
        ):
            raise ValueError("Frozen producer epochs use another control cut.")
        if self.context_projection is not None and (
            self.context_projection.control_domain_id != cut.control_domain_id
            or self.context_projection.cutoff_at != cut.cutoff_at
            or self.context_projection.cutoff_admission_sequence
            != cut.cutoff_admission_sequence
        ):
            raise ValueError("Frozen contexts use another control cut.")
        if self.source_eligibility_projection is not None and (
            self.source_eligibility_projection.control_domain_id
            != cut.control_domain_id
            or self.source_eligibility_projection.cutoff_at != cut.cutoff_at
            or self.source_eligibility_projection.cutoff_admission_sequence
            != cut.cutoff_admission_sequence
        ):
            raise ValueError("Frozen source eligibility uses another control cut.")
        if (self.source_eligibility_projection is None) != (
            self.source_eligibility_spec is None
        ):
            raise ValueError(
                "Source eligibility projection/spec must appear together."
            )
        activation = self.state.residual_session_activation_binding
        lifecycle_projection = self.residual_anchor_lifecycle_projection
        lifecycle_spec = self.residual_anchor_lifecycle_spec
        lifecycle_scope = self.residual_anchor_lifecycle_scope_id
        requires_lifecycle = (
            strategy_spec.required_residual_anchor_lifecycle_spec_hash
            is not None
        )
        if requires_lifecycle:
            if not isinstance(
                lifecycle_projection,
                ResidualAnchorLifecycleProjection,
            ) or not isinstance(lifecycle_spec, ResidualAnchorLifecycleSpec):
                raise ValueError(
                    "Projected strategy requires a frozen residual-anchor "
                    "lifecycle projection and spec."
                )
            lifecycle_scope = _required(
                lifecycle_scope,
                name="residual_anchor_lifecycle_scope_id",
            )
            if lifecycle_spec.semantic_content_hash != (
                strategy_spec.required_residual_anchor_lifecycle_spec_hash
            ):
                raise ValueError(
                    "Frozen residual-anchor lifecycle spec differs from strategy."
                )
            if (
                lifecycle_projection.control_domain_id != cut.control_domain_id
                or lifecycle_projection.cutoff_at != cut.cutoff_at
                or lifecycle_projection.cutoff_admission_sequence
                != cut.cutoff_admission_sequence
            ):
                raise ValueError(
                    "Frozen residual-anchor lifecycle uses another control cut."
                )
            if activation is None:
                raise ValueError(
                    "Residual-anchor lifecycle projection requires an active "
                    "residual session."
                )
        elif any(
            value is not None
            for value in (
                lifecycle_projection,
                lifecycle_spec,
                lifecycle_scope,
            )
        ):
            raise ValueError(
                "Frozen projection contains undeclared residual-anchor lifecycle."
            )
        if not isinstance(
            self.temporal_selection_spec,
            CompositionTemporalSelectionSpec,
        ):
            raise TypeError(
                "temporal_selection_spec must be CompositionTemporalSelectionSpec."
            )
        basis_target = _finite(
            self.basis_target_effective_at,
            name="basis_target_effective_at",
        )
        active = self.state.active_trajectory
        if active is None or active.keyframe_at(basis_target) is None:
            raise ValueError("Frozen projection basis must be a trajectory keyframe.")
        if active.clock_domain_id != cut.clock_domain_id:
            raise ValueError("Trajectory and projection cut use different clocks.")
        if activation is not None and (
            activation.control_cut.runtime_session_id != cut.runtime_session_id
            or activation.control_cut.control_domain_id != cut.control_domain_id
            or activation.control_cut.clock_domain_id != cut.clock_domain_id
            or activation.control_cut.clock_reset_epoch != cut.clock_reset_epoch
        ):
            raise ValueError("Residual session and frozen projection differ.")
        payload = {
            "state_hash": self.state.semantic_content_hash,
            "publication_basis_hash": self.state.publication_basis_hash,
            "active_trajectory_hash": active.semantic_content_hash,
            "composition_spec_hash": self.composition_spec.semantic_content_hash,
            "strategy_spec_hash": strategy_spec.semantic_content_hash,
            "nominal_role": nominal_role,
            "nominal_stamp": _stamp_payload(self.nominal_stamp),
            "residual_input_hash": (
                None if residual is None else residual.semantic_content_hash
            ),
            "held_input_hashes": tuple(
                item.semantic_content_hash for item in held_inputs
            ),
            "control_cut_hash": frozen_control_cut_semantic_hash(cut),
            "active_epoch_by_producer": epochs.active_epoch_by_producer,
            "activation_id_by_producer": epochs.activation_id_by_producer,
            "context_record_hashes": (
                ()
                if self.context_projection is None
                else tuple(
                    value.semantic_content_hash
                    for value in self.context_projection.contexts
                )
            ),
            "source_eligibility_state_hashes": (
                ()
                if self.source_eligibility_projection is None
                else tuple(
                    value.semantic_content_hash
                    for value in self.source_eligibility_projection.records
                )
            ),
            "source_eligibility_spec_hash": (
                None
                if self.source_eligibility_spec is None
                else self.source_eligibility_spec.semantic_content_hash
            ),
            "residual_anchor_lifecycle_record_hashes": (
                ()
                if lifecycle_projection is None
                else tuple(
                    value.semantic_content_hash
                    for value in lifecycle_projection.records
                )
            ),
            "residual_anchor_lifecycle_spec_hash": (
                None
                if lifecycle_spec is None
                else lifecycle_spec.semantic_content_hash
            ),
            "residual_anchor_lifecycle_scope_id": lifecycle_scope,
            "temporal_selection_spec_hash": (
                self.temporal_selection_spec.semantic_content_hash
            ),
            "information_plan_hash": (
                None
                if self.information_plan is None
                else self.information_plan.semantic_content_hash
            ),
            "basis_target_effective_at": basis_target,
            "closing": False,
            "promotable": False,
        }
        object.__setattr__(self, "nominal_role", nominal_role)
        object.__setattr__(self, "held_inputs", held_inputs)
        object.__setattr__(self, "basis_target_effective_at", basis_target)
        object.__setattr__(
            self,
            "residual_anchor_lifecycle_scope_id",
            lifecycle_scope,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    FROZEN_TRAJECTORY_COMPOSITION_PROJECTION_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def projection_id(self) -> str:
        return f"frozen-trajectory-composition:{self.semantic_content_hash}"

    @property
    def promotable(self) -> bool:
        return False


def _first_unemitted_target(state: TrajectoryEditingState) -> float:
    active = state.active_trajectory
    if active is None:
        raise ValueError("A frozen trajectory projection requires an active base.")
    for keyframe in active.keyframes:
        if state.last_emitted_target_at is None or (
            keyframe.target_effective_at > state.last_emitted_target_at
        ):
            return keyframe.target_effective_at
    raise ValueError("Active trajectory has no un-emitted target.")


def capture_frozen_trajectory_composition_projection(
    runtime: ControlCompositionRuntime,
    *,
    state: TrajectoryEditingState,
    composition_spec: CompositionSpec,
    nominal_role: str,
    nominal_stamp: TrajectoryCompositionContributorStamp,
    residual_input: FrozenTrajectoryResidualInput | None = None,
    held_inputs: tuple[FrozenTrajectoryHeldInput, ...] = (),
    source_eligibility_spec: CompositionSourceEligibilitySpec | None = None,
    residual_anchor_lifecycle_scope_id: str | None = None,
    basis_target_effective_at: float | None = None,
    cutoff_at: float | None = None,
) -> FrozenTrajectoryCompositionProjection:
    """Capture one coherent non-closing cut, then detach from the runtime."""

    if not isinstance(runtime, ControlCompositionRuntime):
        raise TypeError("runtime must be ControlCompositionRuntime.")
    if runtime.strategy_control is not None:
        raise ValueError(
            "Projected hot-control binding is not implemented; refusing to "
            "bypass its formal semantics."
        )
    if runtime.pre_mix_policy is not None:
        raise ValueError(
            "Projected pre-mix policy is not implemented; refusing to bypass it."
        )
    strategy = runtime.composition_strategy
    if strategy is None or not callable(
        getattr(strategy, "decide_projected", None)
    ):
        raise ValueError(
            "Runtime strategy does not declare projected evaluation support."
        )
    if strategy.spec.composition_spec_hash != composition_spec.semantic_content_hash:
        raise ValueError("Runtime strategy is bound to another CompositionSpec.")
    basis_target = (
        _first_unemitted_target(state)
        if basis_target_effective_at is None
        else _finite(
            basis_target_effective_at,
            name="basis_target_effective_at",
        )
    )
    capture = runtime.capture_non_closing_projection_cut(
        cutoff_at=cutoff_at,
        include_composition_inputs=False,
        include_contexts=True,
        include_source_eligibility=(source_eligibility_spec is not None),
        include_residual_anchor_lifecycle=(
            strategy.spec.required_residual_anchor_lifecycle_spec_hash
            is not None
        ),
    )
    cut = capture.cut
    epochs = capture.projections["producer_epochs"]
    if not isinstance(epochs, ProducerEpochSnapshot):
        raise TypeError("Runtime returned an invalid producer-epoch projection.")

    information_plan = runtime.information_plan
    required_channels = strategy.spec.context_channels
    full_context_projection = capture.projections.get("contexts")
    context_projection: CompositionContextProjection | None = None
    if required_channels:
        if not isinstance(full_context_projection, CompositionContextProjection):
            raise ValueError("Projected strategy requires unavailable contexts.")
        basis_context = build_composition_context_snapshot(
            full_context_projection,
            producer_epochs=epochs,
            target_effective_at=basis_target,
            required_channels=required_channels,
        )
        context_projection = CompositionContextProjection(
            control_domain_id=cut.control_domain_id,
            cutoff_at=cut.cutoff_at,
            cutoff_admission_sequence=cut.cutoff_admission_sequence,
            contexts=tuple(
                basis_context.selected_context_by_channel[channel]
                for channel in required_channels
                if channel in basis_context.selected_context_by_channel
            ),
        )

    eligibility_projection: CompositionSourceEligibilityProjection | None = None
    if source_eligibility_spec is not None:
        source_eligibility_spec.validate_composition_spec(composition_spec)
        full_eligibility = capture.projections.get("source_eligibility")
        if not isinstance(
            full_eligibility,
            CompositionSourceEligibilityProjection,
        ):
            raise ValueError(
                "Source eligibility spec requires an enabled runtime store."
            )
        basis_eligibility = build_composition_source_eligibility_snapshot(
            full_eligibility,
            producer_epochs=epochs,
            spec=source_eligibility_spec,
            composition_spec=composition_spec,
            target_effective_at=basis_target,
        )
        eligibility_projection = CompositionSourceEligibilityProjection(
            control_domain_id=cut.control_domain_id,
            cutoff_at=cut.cutoff_at,
            cutoff_admission_sequence=cut.cutoff_admission_sequence,
            records=tuple(
                basis_eligibility.selected_state_by_role[role]
                for role in composition_spec.input_roles
                if role in basis_eligibility.selected_state_by_role
            ),
        )

    lifecycle_projection = None
    lifecycle_spec = None
    if strategy.spec.required_residual_anchor_lifecycle_spec_hash is not None:
        lifecycle_spec = getattr(
            strategy,
            "residual_anchor_lifecycle_spec",
            None,
        )
        if not isinstance(lifecycle_spec, ResidualAnchorLifecycleSpec):
            raise ValueError(
                "Projected strategy does not expose its lifecycle spec."
            )
        if lifecycle_spec.semantic_content_hash != (
            strategy.spec.required_residual_anchor_lifecycle_spec_hash
        ):
            raise ValueError("Projected strategy lifecycle spec hash mismatch.")
        lifecycle_projection = capture.projections.get(
            "residual_anchor_lifecycle"
        )
        if not isinstance(
            lifecycle_projection,
            ResidualAnchorLifecycleProjection,
        ):
            raise ValueError(
                "Runtime did not return a residual-anchor lifecycle projection."
            )
        if not str(residual_anchor_lifecycle_scope_id or "").strip():
            raise ValueError(
                "Lifecycle-aware trajectory projection requires its scope ID."
            )
        activation_binding = state.residual_session_activation_binding
        if activation_binding is None:
            raise ValueError(
                "Lifecycle-aware trajectory projection requires an active "
                "residual session."
            )
        build_residual_anchor_lifecycle_snapshot(
            lifecycle_projection,
            producer_epochs=epochs,
            spec=lifecycle_spec,
            activation_scope=activation_binding.activation_scope,
            lifecycle_scope_id=residual_anchor_lifecycle_scope_id,
            target_effective_at=basis_target,
        )

    return FrozenTrajectoryCompositionProjection(
        state=state,
        composition_spec=composition_spec,
        strategy=strategy,
        nominal_role=nominal_role,
        nominal_stamp=nominal_stamp,
        residual_input=residual_input,
        held_inputs=held_inputs,
        control_cut=cut,
        producer_epochs=epochs,
        context_projection=context_projection,
        source_eligibility_projection=eligibility_projection,
        source_eligibility_spec=source_eligibility_spec,
        residual_anchor_lifecycle_projection=lifecycle_projection,
        residual_anchor_lifecycle_spec=lifecycle_spec,
        residual_anchor_lifecycle_scope_id=(
            residual_anchor_lifecycle_scope_id
        ),
        temporal_selection_spec=runtime.temporal_selection_spec,
        information_plan=information_plan,
        basis_target_effective_at=basis_target,
    )


__all__ = [
    "FROZEN_TRAJECTORY_COMPOSITION_PROJECTION_HASH_DOMAIN_DEVELOPMENT",
    "FROZEN_TRAJECTORY_HELD_INPUT_HASH_DOMAIN_DEVELOPMENT",
    "FROZEN_TRAJECTORY_RESIDUAL_INPUT_HASH_DOMAIN_DEVELOPMENT",
    "FrozenTrajectoryCompositionProjection",
    "FrozenTrajectoryHeldInput",
    "FrozenTrajectoryResidualInput",
    "capture_frozen_trajectory_composition_projection",
]
