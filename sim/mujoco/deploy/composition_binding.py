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

from control_composition.composition import (
    CompositionStrategyControl,
    FormalCompositionStrategy,
)
from control_composition.information import CompositionInformationPlan
from control_composition.retention import TimeWindowRetentionSpec
from control_composition.runtime import ControlCompositionRuntime
from dataset_contracts.control_trace import ControlTraceFact
from dataset_interface.composition_control_trace import (
    CompositionControlTraceEmitter,
)
from robot_motion.action import ComponentActionSchema
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


@dataclass(frozen=True)
class MujocoCompositionRuntimeBinding:
    """Installed composition adapter over one authoritative MuJoCo runtime."""

    trace_emitter: CompositionControlTraceEmitter
    strategy_control: CompositionStrategyControl | None
    topology: ControlCompositionRuntime


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
        topology=topology,
    )


__all__ = [
    "MujocoCompositionRuntimeBinding",
    "build_mujoco_composition_runtime_binding",
]
