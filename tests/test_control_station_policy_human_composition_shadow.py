from __future__ import annotations

import numpy as np
import pytest

from control_composition.composition import CompositionSpec
from control_composition.retention import TimeWindowRetentionSpec
from control_composition.runtime import ControlCompositionRuntime
from control_station.composition import (
    PolicyHumanCompositionShadowProfile,
    PolicyHumanCompositionShadowSession,
)
from control_station.sources import (
    PassiveHumanCommandObserverFanout,
    PolicyCompositionInputAdapter,
    PolicyCompositionInputBinding,
)
from robot_motion.action import (
    ActionComponentCatalog,
    ActionComponentSpec,
    ArmActionSpec,
    BIMANUAL_EE_POSE_V1,
    RobotAction,
    SourceValidationResult,
    build_robot_action_v1_runtime_schema,
)
from robot_motion.deploy import (
    NormalizedAction,
    PolicyActionEmission,
    require_passive_policy_emission_observer,
)
from robot_motion.runtime import EffectorCommand, TeleopCommand
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry
from teleop_runtime.action import (
    HumanCommandEmission,
    HumanProposalAdapter,
    HumanProposalAdapterConfig,
    TeleopCommandActionAdapterConfig,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _contracts():
    arm_spec = ArmActionSpec.tool_tcp(
        "left",
        reference_frame="left_arm_base",
    )
    catalog = ActionComponentCatalog(
        catalog_id="shadow-left-arm",
        action_space=BIMANUAL_EE_POSE_V1,
        components=(
            ActionComponentSpec("left_arm", "left_arm", "left_arm"),
        ),
    )
    schema = build_robot_action_v1_runtime_schema(
        component_catalog=catalog,
        arm_action_specs_by_side={"left": arm_spec},
    )
    spec = CompositionSpec(
        spec_id="policy-human-shadow-pose-fusion",
        algebra="weighted_se3_pose_fusion",
        input_roles=("policy", "human"),
        required_roles=("policy", "human"),
        weight_by_role={"policy": 0.5, "human": 0.5},
        max_translation_disagreement_m=2.0,
        max_rotation_disagreement_rad=1.0,
        pose_component_dependencies=("left_arm",),
    )
    return arm_spec, catalog, schema, spec


def _fixture(
    *,
    enabled: bool = True,
    policy_capacity: int = 4,
    human_capacity: int = 4,
    human_horizon: float = 0.2,
    trace_sink=None,
):
    arm_spec, catalog, schema, spec = _contracts()
    clock = _Clock()
    coordinator = ControlAdmissionCoordinator(
        "shadow-domain",
        clock_domain=ClockDomainDescriptor(
            clock_domain_id="control.monotonic",
            runtime_session_id="shadow-session",
            boot_id="shadow-boot",
            reset_epoch=0,
        ),
        clock=clock,
    )
    epochs = ProducerEpochRegistry(
        runtime_session_id="shadow-session",
        coordinator=coordinator,
    )
    for producer_id in ("policy-producer", "human-producer"):
        epochs.activate(
            producer_id=producer_id,
            producer_epoch=1,
            produced_at=9.9,
            produced_clock_domain_id="control.monotonic",
            reason="virtual_start",
        )
    proposal_calls = []

    def forbidden_proposal(draft, **kwargs):
        proposal_calls.append((draft, kwargs))
        raise AssertionError("shadow evaluation must not publish a proposal")

    retention = TimeWindowRetentionSpec(
        spec_id="policy-human-shadow-window",
        history_window_sec=0.2,
        max_future_lead_sec=1.0,
        max_effective_duration_sec=1.0,
    )
    events = []
    runtime = ControlCompositionRuntime(
        runtime_session_id="shadow-session",
        control_domain_id="shadow-domain",
        embodiment_instance_id="arbitrary-embodiment",
        coordinator=coordinator,
        producer_epochs=epochs,
        action_schema=schema,
        event_emitter=events.extend,
        proposal_admitter=forbidden_proposal,
        context_retention_spec=retention,
        composition_input_retention_spec=retention,
        clock=clock,
    )
    policy_adapter = PolicyCompositionInputAdapter(
        PolicyCompositionInputBinding(
            source_slot_id="policy-source",
            role="policy",
            runtime_session_id="shadow-session",
            control_domain_id="shadow-domain",
            producer_id="policy-producer",
            producer_epoch=1,
            policy_node_id="virtual-policy",
            policy_clock_domain_id="control.monotonic",
            component_catalog_hash=catalog.semantic_content_hash,
            action_space=BIMANUAL_EE_POSE_V1,
            required_active_effectors=("left_arm",),
            action_schema=schema,
        )
    )
    human_adapter = HumanProposalAdapter(
        HumanProposalAdapterConfig(
            runtime_session_id="shadow-session",
            control_domain_id="shadow-domain",
            embodiment_instance_id="arbitrary-embodiment",
            producer_id="human-producer",
            producer_epoch=1,
            produced_clock_domain_id="control.monotonic",
            effective_clock_domain_id="control.monotonic",
            component_catalog=catalog,
            action_schema=schema,
            action_adapter=TeleopCommandActionAdapterConfig(
                include_command_metadata=False,
                source="virtual-human",
                arm_specs={"left": arm_spec},
            ),
        )
    )
    session = PolicyHumanCompositionShadowSession(
        profile=PolicyHumanCompositionShadowProfile(
            profile_id=("policy-human-shadow" if enabled else "off"),
            enabled=enabled,
        ),
        runtime=runtime,
        composition_spec=spec,
        policy_input_adapter=policy_adapter,
        human_input_adapter=human_adapter,
        human_role="human",
        human_effective_horizon_sec=human_horizon,
        policy_queue_capacity=policy_capacity,
        human_queue_capacity=human_capacity,
        source_fact_sink=trace_sink,
        clock=clock,
    )
    return session, clock, proposal_calls, events


def _pose_action(x: float) -> RobotAction:
    return RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={
            "left": np.asarray(
                [x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
        },
        active_effectors=("left_arm",),
        frame_by_effector={"left_arm": "left_arm_base"},
        controlled_frame_by_effector={"left_arm": "tool_tcp"},
        source="virtual-policy",
    )


def _policy_emission(index: int, x: float) -> PolicyActionEmission:
    return PolicyActionEmission(
        emission_id=f"policy-action-emission:{index}",
        admission_mode="on_emission_pop_v1",
        policy_node_id="virtual-policy",
        inference_id=f"inference:{index}",
        policy_output_id=f"policy-output:{index}",
        chunk_id=index,
        chunk_keyframe_index=0,
        chunk_size=1,
        decoded_absolute_action=NormalizedAction(
            vector=np.asarray([x], dtype=np.float32)
        ),
        pre_safety_robot_action=_pose_action(x),
        source_validation=SourceValidationResult(accepted=True),
        representation_receipt=None,
        pre_safety_faults=(),
        produced_at=10.0,
        produced_clock_domain_id="control.monotonic",
        emitted_at=10.0,
        target_effective_at=10.0,
        effective_from=10.0,
        effective_until=10.2,
        effective_clock_domain_id="control.monotonic",
        schedule_spec_id="virtual-schedule",
        schedule_spec_hash="virtual-schedule-hash",
        queue_len_before=1,
        queue_len_after=0,
        queue_pop_count=1,
        semantic_content_hash=f"policy-emission-hash:{index}",
    )


def _human_emission(index: int, x: float) -> HumanCommandEmission:
    arm_spec = ArmActionSpec.tool_tcp(
        "left",
        reference_frame="left_arm_base",
    )
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    command = TeleopCommand(timestamp=10.0)
    command.add_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="cartesian_pose",
            value=pose,
            frame_name="left_arm_base",
            metadata={"arm_action_spec": arm_spec.to_metadata()},
        )
    )
    return HumanCommandEmission.capture(
        emission_sequence=index,
        emitted_at=10.0,
        contributions=(("virtual-human", command),),
        merged_command=command,
    )


def test_shadow_is_default_off_and_does_not_retain_or_evaluate_inputs() -> None:
    session, _, proposal_calls, events = _fixture(enabled=False)

    session.put_nowait(_policy_emission(0, 0.0))
    assert session.observe_human_emission(_human_emission(0, 0.2)) is True
    cycle = session.evaluate_at(target_effective_at=10.02)

    assert cycle.status == "disabled"
    assert cycle.evaluation is None
    assert cycle.policy_ingress.queue_size == 0
    assert cycle.human_ingress.queue_size == 0
    assert cycle.policy_ingress.ignored_while_disabled_count == 1
    assert cycle.human_ingress.ignored_while_disabled_count == 1
    assert proposal_calls == []
    assert events == []


def test_shadow_evaluates_same_cut_without_proposal_target_or_command() -> None:
    session, _, proposal_calls, _ = _fixture()
    session.put_nowait(_policy_emission(0, 0.0))
    assert session.observe_human_emission(_human_emission(0, 0.2)) is True

    cycle = session.evaluate_at(target_effective_at=10.02)

    assert cycle.status == "evaluated"
    assert cycle.evaluation.proposal_admission is None
    assert cycle.evaluation.composition_evaluation.output_status == "ready"
    assert len(cycle.submissions) == 2
    assert all(submission.admitted for submission in cycle.submissions)
    assert cycle.authoritative is False
    assert cycle.proposal_admitted is False
    assert cycle.send_actions is False
    assert proposal_calls == []
    candidate = cycle.conditional_action
    assert candidate is not None
    np.testing.assert_allclose(
        candidate.values_by_component["left_arm"].values[:3],
        [0.1, 0.0, 0.0],
        atol=1e-6,
    )


def test_shadow_exposes_existing_policy_and_human_passive_ports() -> None:
    session, _, _, _ = _fixture()
    assert require_passive_policy_emission_observer(session) is session
    fanout = PassiveHumanCommandObserverFanout(
        {"composition-shadow": session.observe_human_emission}
    )

    session.put_nowait(_policy_emission(0, 0.0))
    assert fanout.observe_emission(_human_emission(0, 0.2)) is True
    cycle = session.evaluate_at(target_effective_at=10.02)

    assert cycle.status == "evaluated"
    assert fanout.status()["subscribers"]["composition-shadow"]["error_count"] == 0


def test_shadow_queue_is_bounded_latest_preserving_and_rate_independent() -> None:
    session, _, proposal_calls, _ = _fixture(policy_capacity=1)
    session.put_nowait(_policy_emission(0, -0.4))
    session.put_nowait(_policy_emission(1, 0.4))
    assert session.observe_human_emission(_human_emission(0, 0.0)) is True

    cycle = session.evaluate_at(target_effective_at=10.02)

    assert cycle.status == "evaluated"
    assert cycle.policy_ingress.evicted_count == 1
    assert cycle.policy_ingress.drained_count == 1
    assert [item.source_kind for item in cycle.submissions] == ["policy", "human"]
    np.testing.assert_allclose(
        cycle.conditional_action.values_by_component["left_arm"].values[:3],
        [0.2, 0.0, 0.0],
        atol=1e-6,
    )
    assert proposal_calls == []


def test_missing_required_source_yields_no_action_not_implicit_hold() -> None:
    session, _, proposal_calls, _ = _fixture()
    session.put_nowait(_policy_emission(0, 0.1))

    cycle = session.evaluate_at(target_effective_at=10.02)

    evaluation = cycle.evaluation.composition_evaluation
    assert evaluation.output_status == "no_action"
    assert evaluation.reason == "missing_required:human"
    assert cycle.conditional_action is None
    assert proposal_calls == []


def test_expired_human_source_is_not_held_or_silently_renormalized() -> None:
    session, _, proposal_calls, _ = _fixture(human_horizon=0.1)
    session.put_nowait(_policy_emission(0, 0.1))
    session.observe_human_emission(_human_emission(0, 0.2))

    cycle = session.evaluate_at(target_effective_at=10.15)

    evaluation = cycle.evaluation.composition_evaluation
    assert evaluation.output_status == "no_action"
    assert evaluation.reason == "missing_required:human"
    assert proposal_calls == []


def test_duplicate_observer_delivery_reuses_source_identity() -> None:
    session, _, proposal_calls, _ = _fixture()
    policy = _policy_emission(0, 0.0)
    human = _human_emission(0, 0.2)
    session.put_nowait(policy)
    session.put_nowait(policy)
    session.observe_human_emission(human)
    session.observe_human_emission(human)

    cycle = session.evaluate_at(target_effective_at=10.02)

    assert cycle.status == "evaluated"
    selected = cycle.evaluation.input_snapshot.selected_input_by_role
    assert set(selected) == {"policy", "human"}
    assert len({item.composition_input_id for item in selected.values()}) == 2
    assert proposal_calls == []


def test_trace_sink_failure_is_visible_but_cannot_gate_shadow_composition() -> None:
    def failing_trace_sink(_fact):
        raise RuntimeError("trace unavailable")

    session, _, proposal_calls, _ = _fixture(trace_sink=failing_trace_sink)
    session.put_nowait(_policy_emission(0, 0.0))
    session.observe_human_emission(_human_emission(0, 0.2))

    cycle = session.evaluate_at(target_effective_at=10.02)

    assert cycle.status == "evaluated"
    assert cycle.evaluation.composition_evaluation.output_status == "ready"
    assert all(item.trace_failures for item in cycle.submissions)
    assert proposal_calls == []


def test_shadow_rejects_same_target_revision_without_replacement_protocol() -> None:
    session, _, _, _ = _fixture()
    session.evaluate_at(target_effective_at=10.02)

    with pytest.raises(ValueError, match="strictly increasing"):
        session.evaluate_at(target_effective_at=10.02)
