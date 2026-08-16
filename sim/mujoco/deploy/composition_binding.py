"""Optional composition-runtime binding for the MuJoCo target runtime.

The MuJoCo runtime owns target state, authoritative execution, and dispatch.
This module owns only the explicit adapter that connects those lower ports to
the backend-neutral ``control_composition`` runtime and its passive trace
projection. Keeping the construction here prevents the general target runtime
from growing a second implementation of composition retention or lifecycle.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from control_composition.activation import CompositionActivationScope
from control_composition.composition import (
    CompositionContextDraft,
    CompositionDynamicParameterDraft,
    CompositionInputBatchDraft,
    CompositionInputDraft,
    CompositionSpec,
    CompositionStrategyControl,
    FormalCompositionStrategy,
)
from control_composition.eligibility import (
    CompositionSourceEligibilityDraft,
    CompositionSourceEligibilitySpec,
)
from control_composition.information import CompositionInformationPlan
from control_composition.observability import SourceDifferenceSpec
from control_composition.residual_anchor_lifecycle import (
    ResidualAnchorLifecycleDraft,
    ResidualAnchorLifecycleSpec,
)
from control_composition.retention import TimeWindowRetentionSpec
from control_composition.runtime import ControlCompositionRuntime
from dataset_contracts.control_trace import ControlTraceFact
from dataset_interface.composition_control_trace import (
    CompositionControlTraceEmitter,
)
from robot_motion.action import ComponentActionSchema
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext
from robot_motion.runtime.control_admission import (
    ControlAdmissionCoordinator,
    ControlCut,
)
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


@dataclass(frozen=True)
class MujocoCompositionRuntimeBinding:
    """Lifecycle-gated composition port over one authoritative MuJoCo target.

    This port admits and evaluates composition-owned state only. Proposal
    promotion, authority, canonical safety, target execution, and dispatch are
    intentionally absent and remain on ``MujocoDeployRuntime``.
    """

    trace_emitter: CompositionControlTraceEmitter
    strategy_control: CompositionStrategyControl | None
    _topology: ControlCompositionRuntime
    started: Callable[[], bool]

    def _require_started(self) -> None:
        if not self.started():
            raise RuntimeError("MujocoDeployRuntime.start() is required.")

    @property
    def composition_strategy(self) -> FormalCompositionStrategy | None:
        return self._topology.composition_strategy

    @property
    def runtime_session_id(self) -> str:
        return self._topology.runtime_session_id

    @property
    def control_domain_id(self) -> str:
        return self._topology.control_domain_id

    @property
    def coordinator(self) -> ControlAdmissionCoordinator:
        return self._topology.coordinator

    @property
    def source_eligibility(self):
        return self._topology.source_eligibility

    @property
    def input_action_schema_by_role(self) -> Mapping[str, ComponentActionSchema]:
        return self._topology.input_action_schema_by_role

    def admit_composition_context(self, draft: CompositionContextDraft):
        self._require_started()
        return self._topology.admit_composition_context(draft)

    def resolve_composition_context(self, context_id: str):
        """Resolve one admitted context without advancing receiver state."""

        self._require_started()
        return self._topology.contexts.record_by_id(context_id)

    def capture_composition_context_snapshot(
        self,
        *,
        target_effective_at: float,
        required_channels: tuple[str, ...],
    ):
        self._require_started()
        return self._topology.capture_composition_context_snapshot(
            target_effective_at=target_effective_at,
            required_channels=required_channels,
        )

    def admit_composition_input(
        self,
        draft: CompositionInputDraft,
        *,
        activation_scope: CompositionActivationScope | None = None,
    ):
        self._require_started()
        return self._topology.admit_composition_input(
            draft,
            activation_scope=activation_scope,
        )

    def admit_composition_inputs(
        self,
        batch: CompositionInputBatchDraft,
        *,
        activation_scope: CompositionActivationScope | None = None,
    ):
        self._require_started()
        return self._topology.admit_composition_inputs(
            batch,
            activation_scope=activation_scope,
        )

    def admit_source_eligibility(
        self,
        draft: CompositionSourceEligibilityDraft,
    ):
        self._require_started()
        return self._topology.admit_source_eligibility(draft)

    def admit_residual_anchor_lifecycle(
        self,
        draft: ResidualAnchorLifecycleDraft,
        *,
        spec: ResidualAnchorLifecycleSpec,
        activation_scope: CompositionActivationScope,
    ):
        self._require_started()
        return self._topology.admit_residual_anchor_lifecycle(
            draft,
            spec=spec,
            activation_scope=activation_scope,
        )

    def admit_dynamic_parameters(self, draft: CompositionDynamicParameterDraft):
        self._require_started()
        return self._topology.admit_dynamic_parameters(draft)

    def evaluate_composition_shadow(
        self,
        *,
        spec: CompositionSpec,
        target_effective_at: float,
        required_context_channels: tuple[str, ...] = (),
        cutoff_at: float | None = None,
        control_cut: ControlCut | None = None,
        source_difference_spec: SourceDifferenceSpec | None = None,
        activation_scope: CompositionActivationScope | None = None,
        source_eligibility_spec: CompositionSourceEligibilitySpec | None = None,
        residual_anchor_lifecycle_scope_id: str | None = None,
    ):
        """Evaluate one composition cut without proposal or target authority."""

        self._require_started()
        if self.strategy_control is None:
            submission = self._topology.evaluate_composition(
                spec=spec,
                target_effective_at=target_effective_at,
                required_context_channels=required_context_channels,
                cutoff_at=cutoff_at,
                control_cut=control_cut,
                source_difference_spec=source_difference_spec,
                activation_scope=activation_scope,
                source_eligibility_spec=source_eligibility_spec,
                residual_anchor_lifecycle_scope_id=(
                    residual_anchor_lifecycle_scope_id
                ),
            )
        else:
            latest_revision = self.strategy_control.latest_revision
            latest_spec_hash = (
                latest_revision.composition_spec.semantic_content_hash
            )
            if spec.semantic_content_hash != latest_spec_hash:
                raise ValueError(
                    "Hot composition evaluation requested another "
                    "CompositionSpec."
                )
            submission = self._topology.evaluate_hot_composition(
                target_effective_at=target_effective_at,
                required_context_channels=required_context_channels,
                cutoff_at=cutoff_at,
                control_cut=control_cut,
                source_difference_spec=source_difference_spec,
                activation_scope=activation_scope,
                source_eligibility_spec=source_eligibility_spec,
                residual_anchor_lifecycle_scope_id=(
                    residual_anchor_lifecycle_scope_id
                ),
            )
        if submission.proposal_admission is not None:
            raise RuntimeError(
                "Shadow composition unexpectedly admitted a proposal."
            )
        return submission


def build_mujoco_composition_runtime_binding(
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    coordinator: ControlAdmissionCoordinator,
    producer_epochs: ProducerEpochRegistry,
    action_schema: ComponentActionSchema,
    input_action_schema_by_role: Mapping[str, ComponentActionSchema] | None,
    trace_context: AuthoritativeTraceContext,
    trace_sink: Callable[[tuple[ControlTraceFact, ...]], object],
    proposal_admitter: Callable[..., object],
    decision_hz: float,
    enable_source_eligibility: bool,
    information_plan: CompositionInformationPlan | None,
    composition_strategy: FormalCompositionStrategy | None,
    composition_strategy_control_factory: (
        Callable[
            [ControlAdmissionCoordinator, ProducerEpochRegistry, float],
            CompositionStrategyControl,
        ]
        | None
    ),
    clock: Callable[[], float],
    started: Callable[[], bool],
) -> MujocoCompositionRuntimeBinding:
    """Build the optional mixed-control adapter without target ownership."""

    trace_emitter = CompositionControlTraceEmitter(
        trace_context=trace_context,
        trace_sink=trace_sink,
    )
    strategy_control = None
    if composition_strategy_control_factory is not None:
        if composition_strategy is None:
            raise ValueError(
                "A composition strategy control factory requires a "
                "formal composition strategy."
            )
        strategy_control = composition_strategy_control_factory(
            coordinator,
            producer_epochs,
            float(clock()),
        )
        if not isinstance(strategy_control, CompositionStrategyControl):
            raise TypeError(
                "composition_strategy_control_factory must return "
                "CompositionStrategyControl."
            )

    retention_window_sec = max(0.02, 1.0 / float(decision_hz))
    topology = ControlCompositionRuntime(
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        coordinator=coordinator,
        producer_epochs=producer_epochs,
        action_schema=action_schema,
        input_action_schema_by_role=input_action_schema_by_role,
        event_emitter=trace_emitter,
        proposal_admitter=proposal_admitter,
        context_retention_spec=TimeWindowRetentionSpec(
            spec_id="mujoco-composition-context-latest-v1",
            history_window_sec=retention_window_sec,
            max_future_lead_sec=1.0,
            max_effective_duration_sec=1.0,
            max_late_arrival_sec=0.0,
        ),
        composition_input_retention_spec=TimeWindowRetentionSpec(
            spec_id="mujoco-composition-input-latest-v1",
            history_window_sec=retention_window_sec,
            max_future_lead_sec=1.0,
            max_effective_duration_sec=1.0,
            max_late_arrival_sec=0.0,
        ),
        source_eligibility_retention_spec=(
            TimeWindowRetentionSpec(
                spec_id="mujoco-composition-source-eligibility-latest-v1",
                history_window_sec=retention_window_sec,
                max_future_lead_sec=1.0,
                max_effective_duration_sec=1.0,
                max_late_arrival_sec=0.0,
            )
            if enable_source_eligibility
            else None
        ),
        information_plan=information_plan,
        composition_strategy=composition_strategy,
        strategy_control=strategy_control,
        clock=clock,
    )
    return MujocoCompositionRuntimeBinding(
        trace_emitter=trace_emitter,
        strategy_control=strategy_control,
        _topology=topology,
        started=started,
    )


__all__ = [
    "MujocoCompositionRuntimeBinding",
    "build_mujoco_composition_runtime_binding",
]
