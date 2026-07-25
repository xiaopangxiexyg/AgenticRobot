from __future__ import annotations

import numpy as np
import pytest

from control_composition.composition import (
    CompositionContextDraft,
    CompositionInputDraft,
    CompositionSpec,
)
from control_composition.events import CompositionDomainEvent
from control_composition.information import (
    CompositionContextRequirement,
    CompositionInformationPlan,
)
from control_composition.retention import TimeWindowRetentionSpec
from control_composition.runtime import ControlCompositionRuntime
from dataset_interface.composition_control_trace import (
    project_composition_domain_event,
)
from robot_motion.action import (
    BIMANUAL_QPOS_V1,
    ActionComponentCatalog,
    ActionComponentSpec,
    RobotAction,
)
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def requirement(**overrides) -> CompositionContextRequirement:
    values = {
        "channel": "vision.features",
        "accepted_payload_contract_hashes": ("vision-features-v1",),
        "payload_access": "blob_ref",
        "maximum_age_sec": 0.05,
    }
    values.update(overrides)
    return CompositionContextRequirement(**values)


def test_information_plan_is_deterministic_and_channel_addressable() -> None:
    first = CompositionInformationPlan(
        plan_id="visual-guidance-v1",
        context_requirements=(requirement(),),
        diagnostic_level="core",
    )
    second = CompositionInformationPlan(
        plan_id="visual-guidance-v1",
        context_requirements=(requirement(),),
        diagnostic_level="core",
    )
    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.required_context_channels == ("vision.features",)
    assert first.requirement_for_channel("vision.features") is (
        first.context_requirements[0]
    )
    assert first.requirement_for_channel("tactile.features") is None


def test_information_plan_rejects_ambiguous_or_unsupported_demands() -> None:
    with pytest.raises(ValueError, match="non-empty and unique"):
        requirement(accepted_payload_contract_hashes=())
    with pytest.raises(ValueError, match="payload access"):
        requirement(payload_access="live_device")
    with pytest.raises(ValueError, match="maximum_age_sec"):
        requirement(maximum_age_sec=-1.0)
    with pytest.raises(ValueError, match="channels must be unique"):
        CompositionInformationPlan(
            plan_id="duplicate-channel-v1",
            context_requirements=(requirement(), requirement()),
        )


def test_dataset_projection_is_additive_and_outside_composition_domain() -> None:
    plan = CompositionInformationPlan(
        plan_id="visual-guidance-v1",
        context_requirements=(requirement(),),
    )
    event = CompositionDomainEvent(
        event_kind="composition_information_plan",
        subject=plan,
        occurred_at=4.0,
    )
    fact = project_composition_domain_event(
        AuthoritativeTraceContext(
            runtime_session_id="session-0",
            control_domain_id="domain-0",
            embodiment_instance_id="embodiment-0",
            clock_domain_id="control.monotonic",
        ),
        event,
    )
    assert fact.event_type == "composition_information_plan"
    assert fact.payload["plan_id"] == plan.plan_id
    assert fact.payload["semantic_content_hash"] == plan.semantic_content_hash
    assert fact.semantic_content_hash


@pytest.mark.parametrize(
    (
        "context_clock_domain_id",
        "maximum_age_sec",
        "expected_status",
        "expected_reason",
    ),
    (
        (
            "human.monotonic",
            0.2,
            "no_action",
            "context_age_clock_mismatch:operator.intent",
        ),
        ("control.monotonic", 0.2, "ready", "composed"),
        (
            "control.monotonic",
            0.1,
            "no_action",
            "stale_context:operator.intent",
        ),
    ),
)
def test_maximum_age_requires_a_shared_clock_domain(
    context_clock_domain_id: str,
    maximum_age_sec: float,
    expected_status: str,
    expected_reason: str,
) -> None:
    clock = lambda: 10.0
    coordinator = ControlAdmissionCoordinator(
        "robot-0",
        clock_domain=ClockDomainDescriptor(
            clock_domain_id="control.monotonic",
            runtime_session_id="session-0",
            boot_id="boot-0",
            reset_epoch=0,
        ),
        clock=clock,
    )
    epochs = ProducerEpochRegistry(
        runtime_session_id="session-0",
        coordinator=coordinator,
    )
    epochs.activate(
        producer_id="human-producer",
        producer_epoch=1,
        produced_at=9.9,
        produced_clock_domain_id="human.monotonic",
        reason="startup",
    )
    retention = TimeWindowRetentionSpec(
        spec_id="information-plan-clock-test-v1",
        history_window_sec=0.0,
        max_future_lead_sec=2.0,
        max_effective_duration_sec=2.0,
    )
    plan = CompositionInformationPlan(
        plan_id="context-age-clock-test-v1",
        context_requirements=(
            requirement(
                channel="operator.intent",
                accepted_payload_contract_hashes=("operator-intent-v1",),
                payload_access="inline",
                maximum_age_sec=maximum_age_sec,
            ),
        ),
    )
    runtime = ControlCompositionRuntime(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="fixture-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        component_catalog=ActionComponentCatalog(
            catalog_id="left-arm-v1",
            action_space=BIMANUAL_QPOS_V1,
            components=(
                ActionComponentSpec("left_arm", "left_arm", "left_arm"),
            ),
        ),
        event_emitter=lambda _batch: None,
        proposal_admitter=lambda *_args, **_kwargs: None,
        context_retention_spec=retention,
        composition_input_retention_spec=retention,
        information_plan=plan,
        clock=clock,
    )
    context_receipt = runtime.admit_composition_context(
        CompositionContextDraft(
            runtime_session_id="session-0",
            control_domain_id="robot-0",
            producer_id="human-producer",
            producer_epoch=1,
            sequence=0,
            context_channel="operator.intent",
            context_kind="modality",
            semantic_payload={"mode": "assist"},
            produced_at=9.95,
            produced_clock_domain_id=context_clock_domain_id,
            effective_from=10.0,
            effective_until=11.0,
            effective_clock_domain_id="control.monotonic",
            payload_contract_hash="operator-intent-v1",
        )
    )
    assert context_receipt.accepted
    assert context_receipt.context is not None
    input_receipt = runtime.admit_composition_input(
        CompositionInputDraft(
            runtime_session_id="session-0",
            control_domain_id="robot-0",
            producer_id="human-producer",
            producer_epoch=1,
            sequence=0,
            input_role="base",
            action=RobotAction(
                action_space=BIMANUAL_QPOS_V1,
                arm_qpos_by_side={
                    "left": np.zeros(6, dtype=np.float32),
                },
                source="context-age-clock-test",
            ),
            produced_at=9.95,
            produced_clock_domain_id="human.monotonic",
            effective_from=10.0,
            effective_until=11.0,
            effective_clock_domain_id="control.monotonic",
            context_ids=(context_receipt.context.context_id,),
        )
    )
    assert input_receipt.accepted
    evaluation = runtime.evaluate_composition(
        spec=CompositionSpec(
            spec_id="context-age-clock-test-v1",
            algebra="component_union",
            input_roles=("base",),
            required_roles=("base",),
        ),
        target_effective_at=10.1,
    )
    assert evaluation.composition_result.output_status == expected_status
    assert evaluation.composition_result.reason == expected_reason
