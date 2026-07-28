from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from control_composition.composition import (
    COMPOSITION_EVALUATION_HASH_DOMAIN_V7,
    CompositionContextDraft,
    CompositionInputDraft,
    CompositionPreMixPolicyDecision,
    CompositionPreMixPolicyInput,
    CompositionPreMixPolicySpec,
    CompositionSpec,
    CompositionStrategyDecision,
    DeterministicComposer,
    DeterministicCompositionStrategyV2,
    FormalCompositionStrategyInput,
    FormalCompositionStrategySpec,
)
from control_composition.information import (
    CompositionContextRequirement,
    CompositionInformationPlan,
)
from control_composition.retention import TimeWindowRetentionSpec
from control_composition.runtime import ControlCompositionRuntime
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_interface.composition_control_trace import (
    project_composition_domain_event,
)
from robot_motion.action import (
    BIMANUAL_QPOS_V1,
    ActionComponentCatalog,
    ActionComponentSpec,
    ProposalBuffer,
    RobotAction,
    build_robot_action_v1_runtime_schema,
    project_robot_action_v1_to_component_action,
)
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


pytestmark = [pytest.mark.unit, pytest.mark.contract]


class Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def composition_spec() -> CompositionSpec:
    return CompositionSpec(
        spec_id="formal-strategy-component-union-v1",
        algebra="component_union",
        input_roles=("base",),
        required_roles=("base",),
    )


def retention_spec() -> TimeWindowRetentionSpec:
    return TimeWindowRetentionSpec(
        spec_id="formal-strategy-test-window-v1",
        history_window_sec=0.0,
        max_future_lead_sec=2.0,
        max_effective_duration_sec=2.0,
    )


def build_runtime(
    strategy,
    *,
    information_plan: CompositionInformationPlan | None = None,
    pre_mix_policy=None,
):
    clock = Clock()
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
    for producer_id in ("source-0", "composer-0"):
        epochs.activate(
            producer_id=producer_id,
            producer_epoch=1,
            produced_at=9.9,
            produced_clock_domain_id=f"{producer_id}.monotonic",
            reason="startup",
        )
    proposals = ProposalBuffer(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        capacity=16,
    )
    events = []
    component_catalog = ActionComponentCatalog(
        catalog_id="left-arm-v1",
        action_space=BIMANUAL_QPOS_V1,
        components=(
            ActionComponentSpec("left_arm", "left_arm", "left_arm"),
        ),
    )
    runtime = ControlCompositionRuntime(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="fixture-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        action_schema=build_robot_action_v1_runtime_schema(
            component_catalog=component_catalog,
        ),
        event_emitter=lambda batch: events.extend(batch),
        proposal_admitter=lambda draft, **_kwargs: proposals.admit(draft),
        context_retention_spec=retention_spec(),
        composition_input_retention_spec=retention_spec(),
        information_plan=information_plan,
        composition_strategy=strategy,
        pre_mix_policy=pre_mix_policy,
        clock=clock,
    )
    return clock, runtime, proposals, events


def input_draft(
    *,
    context_ids: tuple[str, ...] = (),
    value: float = 0.2,
) -> CompositionInputDraft:
    action_schema = build_robot_action_v1_runtime_schema(
        component_catalog=ActionComponentCatalog(
            catalog_id="left-arm-v1",
            action_space=BIMANUAL_QPOS_V1,
            components=(
                ActionComponentSpec("left_arm", "left_arm", "left_arm"),
            ),
        ),
    )
    return CompositionInputDraft(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        producer_id="source-0",
        producer_epoch=1,
        sequence=0,
        input_role="base",
        action=project_robot_action_v1_to_component_action(
            RobotAction(
                action_space=BIMANUAL_QPOS_V1,
                arm_qpos_by_side={
                    "left": np.full(6, value, dtype=np.float32),
                },
                source="formal-strategy-test",
            ),
            action_schema=action_schema,
        ),
        produced_at=9.9,
        produced_clock_domain_id="source-0.monotonic",
        effective_from=10.0,
        effective_until=11.0,
        effective_clock_domain_id="control.monotonic",
        context_ids=context_ids,
    )


class _PassThroughPolicy:
    def __init__(self) -> None:
        self._spec = CompositionPreMixPolicySpec(
            policy_id="test-pass-through-policy",
            policy_version="1",
            maximum_decision_latency_sec=0.010,
        )
        self.seen_input: CompositionPreMixPolicyInput | None = None

    @property
    def spec(self) -> CompositionPreMixPolicySpec:
        return self._spec

    def decide(
        self,
        policy_input: CompositionPreMixPolicyInput,
    ) -> CompositionPreMixPolicyDecision:
        self.seen_input = policy_input
        return CompositionPreMixPolicyDecision(
            policy_spec_hash=self.spec.semantic_content_hash,
            policy_input_hash=policy_input.semantic_content_hash,
            disposition="pass",
            reason="test_pass",
        )


class _NoActionPolicy:
    def __init__(self) -> None:
        self._spec = CompositionPreMixPolicySpec(
            policy_id="test-no-action-policy",
            policy_version="1",
            maximum_decision_latency_sec=0.010,
        )

    @property
    def spec(self) -> CompositionPreMixPolicySpec:
        return self._spec

    def decide(
        self,
        policy_input: CompositionPreMixPolicyInput,
    ) -> CompositionPreMixPolicyDecision:
        return CompositionPreMixPolicyDecision(
            policy_spec_hash=self.spec.semantic_content_hash,
            policy_input_hash=policy_input.semantic_content_hash,
            disposition="no_action",
            reason="difference_policy_requested_drop",
        )


class _RaisingPolicy:
    def __init__(self) -> None:
        self._spec = CompositionPreMixPolicySpec(
            policy_id="test-raising-policy",
            policy_version="1",
            maximum_decision_latency_sec=0.010,
        )

    @property
    def spec(self) -> CompositionPreMixPolicySpec:
        return self._spec

    def decide(
        self,
        policy_input: CompositionPreMixPolicyInput,
    ) -> CompositionPreMixPolicyDecision:
        raise RuntimeError("test policy failure")


def test_default_off_pre_mix_policy_preserves_existing_result() -> None:
    spec = composition_spec()
    _, runtime, _, _ = build_runtime(None)
    assert runtime.admit_composition_input(input_draft()).accepted

    evaluation = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    )

    assert evaluation.composition_evaluation.output_status == "ready"
    assert evaluation.evaluation_record is not None
    assert evaluation.evaluation_record.pre_mix_policy_input is None
    assert evaluation.evaluation_record.pre_mix_policy_decision is None


def test_pre_mix_policy_uses_same_cut_and_preserves_numeric_input() -> None:
    policy = _PassThroughPolicy()
    spec = composition_spec()
    _, runtime, _, _ = build_runtime(None, pre_mix_policy=policy)
    assert runtime.admit_composition_input(
        input_draft(value=1_000_000.0)
    ).accepted

    evaluation = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    )
    result = evaluation.composition_evaluation

    assert result.output_status == "ready"
    assert evaluation.evaluation_record is not None
    assert evaluation.evaluation_record.pre_mix_policy_input is policy.seen_input
    assert evaluation.evaluation_record.pre_mix_policy_spec is policy.spec
    assert evaluation.evaluation_record.pre_mix_policy_decision is not None
    assert evaluation.evaluation_record.pre_mix_policy_decision.disposition == "pass"
    np.testing.assert_array_equal(
            result.composed_candidate.value("left_arm").values,
        np.full(6, 1_000_000.0, dtype=np.float32),
    )


def test_pre_mix_policy_can_block_without_rewriting_input_snapshot() -> None:
    policy = _NoActionPolicy()
    spec = composition_spec()
    _, runtime, _, _ = build_runtime(None, pre_mix_policy=policy)
    assert runtime.admit_composition_input(input_draft()).accepted

    evaluation = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    )

    assert evaluation.composition_evaluation.output_status == "no_action"
    assert evaluation.composition_evaluation.composed_candidate is None
    assert evaluation.composition_evaluation.reason.startswith(
        "pre_mix_policy:test-no-action-policy:no_action:"
    )
    assert tuple(evaluation.input_snapshot.selected_input_by_role) == ("base",)
    assert evaluation.evaluation_record is not None
    assert evaluation.evaluation_record.pre_mix_policy_decision is not None
    assert not evaluation.evaluation_record.diagnostics


def test_pre_mix_policy_failure_is_invalid_and_diagnostic() -> None:
    policy = _RaisingPolicy()
    spec = composition_spec()
    _, runtime, _, _ = build_runtime(None, pre_mix_policy=policy)
    assert runtime.admit_composition_input(input_draft()).accepted

    evaluation = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    )

    assert evaluation.composition_evaluation.output_status == "invalid"
    assert evaluation.evaluation_record is not None
    decision = evaluation.evaluation_record.pre_mix_policy_decision
    assert decision is not None
    assert decision.disposition == "invalid"
    assert decision.reason == "policy_error:RuntimeError"
    assert evaluation.diagnostics[0].stage == "pre_mix_policy"


@pytest.mark.parametrize(
    ("diagnostic_level", "expected_diagnostic_count"),
    (("core", 0), ("debug", 1)),
)
def test_information_plan_controls_optional_diagnostic_construction(
    diagnostic_level: str,
    expected_diagnostic_count: int,
) -> None:
    policy = _RaisingPolicy()
    plan = CompositionInformationPlan(
        plan_id=f"diagnostic-construction-{diagnostic_level}-v1",
        diagnostic_level=diagnostic_level,
    )
    _, runtime, _, _ = build_runtime(
        None,
        information_plan=plan,
        pre_mix_policy=policy,
    )
    assert runtime.admit_composition_input(input_draft()).accepted

    evaluation = runtime.evaluate_composition(
        spec=composition_spec(),
        target_effective_at=10.1,
    )

    assert evaluation.composition_evaluation.output_status == "invalid"
    assert evaluation.composition_evaluation.reason.endswith(
        "policy_error:RuntimeError"
    )
    assert evaluation.evaluation_record is not None
    assert len(evaluation.diagnostics) == expected_diagnostic_count
    assert (
        evaluation.evaluation_record.pre_mix_policy_elapsed_sec is None
    ) == (diagnostic_level == "core")


def test_pre_mix_policy_latency_overrun_is_no_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _PassThroughPolicy()
    policy._spec = CompositionPreMixPolicySpec(
        policy_id="test-slow-policy",
        policy_version="1",
        maximum_decision_latency_sec=0.001,
    )
    ticks = iter((10.0, 10.010))
    monkeypatch.setattr(
        "control_composition.runtime.time.perf_counter",
        lambda: next(ticks),
    )
    spec = composition_spec()
    _, runtime, _, _ = build_runtime(None, pre_mix_policy=policy)
    assert runtime.admit_composition_input(input_draft()).accepted

    evaluation = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    )

    assert evaluation.composition_evaluation.output_status == "no_action"
    assert evaluation.composition_evaluation.reason.startswith(
        "pre_mix_policy:test-slow-policy:no_action:policy_latency_exceeded"
    )
    assert evaluation.evaluation_record is not None
    assert evaluation.evaluation_record.pre_mix_policy_elapsed_sec == pytest.approx(
        0.010
    )
    assert evaluation.diagnostics[0].code == "latency_exceeded"


def test_formal_deterministic_strategy_is_additive_and_promotable() -> None:
    spec = composition_spec()
    strategy_spec = FormalCompositionStrategySpec(
        strategy_id="deterministic-migration-v2",
        strategy_version="2.0",
        composition_spec_hash=spec.semantic_content_hash,
    )
    strategy = DeterministicCompositionStrategyV2(strategy_spec)
    _, runtime, proposals, events = build_runtime(strategy)
    assert runtime.admit_composition_input(input_draft()).accepted

    evaluation = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    )
    result = evaluation.composition_evaluation
    assert result.output_status == "ready"
    assert result.strategy_id == strategy_spec.strategy_id
    assert result.strategy_spec_hash == strategy_spec.semantic_content_hash
    assert result.strategy_input_id is not None
    assert result.strategy_input_hash is not None
    assert evaluation.evaluation_record is not None
    assert evaluation.evaluation_record.strategy_spec is strategy_spec
    assert evaluation.evaluation_record.strategy_input is not None
    assert evaluation.evaluation_record.strategy_result_binding is not None

    legacy = DeterministicComposer().compose(
        evaluation.input_snapshot,
        spec=spec,
    )
    np.testing.assert_array_equal(
        result.composed_candidate.value("left_arm").values,
        legacy.composed_candidate.value("left_arm").values,
    )
    assert result.semantic_content_hash != legacy.semantic_content_hash
    assert result.semantic_content_hash == canonical_content_hash(
        {
            "composition_input_snapshot_id": (
                result.composition_input_snapshot_id
            ),
            "composition_input_snapshot_hash": (
                result.composition_input_snapshot_hash
            ),
                "composition_spec_hash": result.composition_spec_hash,
                "action_schema_hash": result.action_schema_hash,
                "target_effective_at": result.target_effective_at,
                "target_clock_domain_id": result.target_clock_domain_id,
            "output_status": result.output_status,
            "reason": result.reason,
            "composed_candidate_hash": result.composed_candidate_hash,
            "input_ids": result.input_ids,
            "input_hashes": result.input_hashes,
            "context_ids": result.context_ids,
            "strategy_id": result.strategy_id,
            "strategy_spec_hash": result.strategy_spec_hash,
            "strategy_input_id": result.strategy_input_id,
            "strategy_input_hash": result.strategy_input_hash,
            "strategy_context_snapshot_id": None,
            "strategy_context_snapshot_hash": None,
                "strategy_information_plan_hash": None,
                "strategy_used_context_ids": (),
                "composition_supported_until": (
                    result.composition_supported_until
                ),
            },
                domain=COMPOSITION_EVALUATION_HASH_DOMAIN_V7,
        )

    promoted = runtime.admit_evaluated_composition(
        evaluation,
        runtime_session_id="session-0",
        producer_id="composer-0",
            producer_epoch=1,
            sequence=0,
            effective_until=10.2,
            policy_node_id="formal-strategy-test",
        )
    assert promoted.proposal_admission is not None
    assert promoted.proposal_admission.accepted
    assert promoted.proposal_admission.proposal is not None
    assert promoted.proposal_admission.proposal.composition_evaluation_id == (
        result.composition_evaluation_id
    )
    assert promoted.proposal_admission.proposal.composition_evaluation_hash == (
        result.semantic_content_hash
    )
    assert promoted.proposal_admission.proposal.policy_output_id is None
    assert len(proposals) == 1

    trace_context = AuthoritativeTraceContext(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="fixture-0",
        clock_domain_id="control.monotonic",
    )
    facts = tuple(
        project_composition_domain_event(trace_context, event)
        for event in events
    )
    event_types = tuple(fact.event_type for fact in facts)
    assert event_types.count("formal_composition_strategy_spec") == 1
    assert "formal_composition_strategy_input" in event_types
    assert "formal_composition_strategy_result_binding" in event_types


def test_composition_support_ceiling_and_publication_target_monotonicity() -> None:
    spec = composition_spec()
    strategy_spec = FormalCompositionStrategySpec(
        strategy_id="support-window-strategy",
        strategy_version="1",
        composition_spec_hash=spec.semantic_content_hash,
    )
    _, runtime, proposals, _ = build_runtime(
        DeterministicCompositionStrategyV2(strategy_spec)
    )
    first_draft = replace(input_draft(), effective_until=10.15)
    assert runtime.admit_composition_input(first_draft).accepted
    first = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    )
    assert (
        first.composition_evaluation.composition_supported_until
        == 10.15
    )

    with pytest.raises(ValueError, match="cannot exceed composition support"):
        runtime.admit_evaluated_composition(
            first,
            runtime_session_id="session-0",
            producer_id="composer-0",
            producer_epoch=1,
            sequence=0,
            effective_until=10.16,
        )

    promoted = runtime.admit_evaluated_composition(
        first,
        runtime_session_id="session-0",
        producer_id="composer-0",
        producer_epoch=1,
        sequence=0,
        effective_until=10.14,
    )
    assert promoted.proposal_admission is not None
    assert promoted.proposal_admission.accepted
    assert promoted.proposal_admission.proposal is not None
    assert promoted.proposal_admission.proposal.effective_until == 10.14

    retry = runtime.admit_evaluated_composition(
        first,
        runtime_session_id="session-0",
        producer_id="composer-0",
        producer_epoch=1,
        sequence=0,
        effective_until=10.14,
    )
    assert retry.proposal_admission is promoted.proposal_admission
    with pytest.raises(ValueError, match="publication envelope"):
        runtime.admit_evaluated_composition(
            first,
            runtime_session_id="session-0",
            producer_id="composer-0",
            producer_epoch=1,
            sequence=1,
            effective_until=10.14,
        )

    second_draft = replace(
        input_draft(value=0.3),
        sequence=1,
        effective_until=10.15,
    )
    assert runtime.admit_composition_input(second_draft).accepted
    same_target = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    )
    assert (
        same_target.composition_evaluation.composition_evaluation_id
        != first.composition_evaluation.composition_evaluation_id
    )
    with pytest.raises(ValueError, match="increase strictly"):
        runtime.admit_evaluated_composition(
            same_target,
            runtime_session_id="session-0",
            producer_id="composer-0",
            producer_epoch=1,
            sequence=1,
            effective_until=10.14,
        )
    assert len(proposals) == 1


class ContextScaleStrategy:
    def __init__(self, spec: FormalCompositionStrategySpec) -> None:
        self.spec = spec

    def decide(
        self,
        strategy_input: FormalCompositionStrategyInput,
    ) -> CompositionStrategyDecision:
        context_snapshot = strategy_input.context_snapshot
        assert context_snapshot is not None
        context = context_snapshot.selected_context_by_channel[
            "operator.scale"
        ]
        scale = float(context.semantic_payload["scale"])
        source = strategy_input.input_snapshot.selected_input_by_role[
            "base"
        ].action
        return CompositionStrategyDecision(
            output_status="ready",
            reason="context_scale",
            composed_candidate=type(source)(
                action_schema=source.action_schema,
                values_by_component={
                    "left_arm": source.value("left_arm").values * scale,
                },
            ),
            used_context_ids=(context.context_id,),
        )


def test_formal_strategy_result_binds_exact_context_and_information_plan() -> None:
    spec = composition_spec()
    information_plan = CompositionInformationPlan(
        plan_id="formal-strategy-context-v1",
        context_requirements=(
            CompositionContextRequirement(
                channel="operator.scale",
                accepted_payload_contract_hashes=("operator-scale-v1",),
                payload_access="inline",
                maximum_age_sec=0.3,
            ),
        ),
    )
    strategy_spec = FormalCompositionStrategySpec(
        strategy_id="context-scale-v2",
        strategy_version="2.0",
        composition_spec_hash=spec.semantic_content_hash,
        required_context_channels=("operator.scale",),
    )
    _, runtime, _, _ = build_runtime(
        ContextScaleStrategy(strategy_spec),
        information_plan=information_plan,
    )
    context = runtime.admit_composition_context(
        CompositionContextDraft(
            runtime_session_id="session-0",
            control_domain_id="robot-0",
            producer_id="source-0",
            producer_epoch=1,
            sequence=0,
            context_channel="operator.scale",
            context_kind="modality",
            semantic_payload={"scale": 0.5},
            produced_at=9.9,
            produced_clock_domain_id="control.monotonic",
            effective_from=10.0,
            effective_until=11.0,
            effective_clock_domain_id="control.monotonic",
            payload_contract_hash="operator-scale-v1",
        )
    )
    assert context.accepted
    assert context.context is not None
    assert runtime.admit_composition_input(
        input_draft(context_ids=(context.context.context_id,))
    ).accepted

    evaluation = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    )
    result = evaluation.composition_evaluation
    assert evaluation.context_snapshot is not None
    assert result.strategy_context_snapshot_id == (
        evaluation.context_snapshot.context_snapshot_id
    )
    assert result.strategy_context_snapshot_hash == (
        evaluation.context_snapshot.semantic_content_hash
    )
    assert result.strategy_information_plan_hash == (
        information_plan.semantic_content_hash
    )
    assert result.composition_supported_until == pytest.approx(10.2)
    assert result.strategy_used_context_ids == (
        context.context.context_id,
    )
    np.testing.assert_allclose(
        result.composed_candidate.value("left_arm").values,
        np.full(6, 0.1, dtype=np.float32),
    )


class FailingStrategy:
    def __init__(self, spec: FormalCompositionStrategySpec) -> None:
        self.spec = spec

    def decide(
        self,
        _strategy_input: FormalCompositionStrategyInput,
    ) -> CompositionStrategyDecision:
        raise RuntimeError("synthetic strategy failure")


def test_formal_strategy_failure_is_a_non_promotable_v4_result() -> None:
    spec = composition_spec()
    strategy_spec = FormalCompositionStrategySpec(
        strategy_id="failing-strategy-v2",
        strategy_version="2.0",
        composition_spec_hash=spec.semantic_content_hash,
    )
    _, runtime, proposals, _ = build_runtime(
        FailingStrategy(strategy_spec)
    )
    assert runtime.admit_composition_input(input_draft()).accepted

    evaluation = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    )
    assert evaluation.composition_evaluation.output_status == "invalid"
    assert evaluation.composition_evaluation.reason == (
        "strategy_error:RuntimeError"
    )
    assert evaluation.composition_evaluation.strategy_input_id is not None
    assert evaluation.diagnostics[0].stage == "formal_strategy"
    assert len(proposals) == 0


def test_formal_strategy_latency_overrun_fails_closed(
    monkeypatch,
) -> None:
    spec = composition_spec()
    strategy_spec = FormalCompositionStrategySpec(
        strategy_id="latency-bounded-strategy-v2",
        strategy_version="2.0",
        composition_spec_hash=spec.semantic_content_hash,
        maximum_decision_latency_sec=0.01,
    )
    strategy = DeterministicCompositionStrategyV2(strategy_spec)
    _, runtime, proposals, _ = build_runtime(strategy)
    assert runtime.admit_composition_input(input_draft()).accepted
    timestamps = iter((1.0, 1.02))
    monkeypatch.setattr(
        "control_composition.runtime.time.perf_counter",
        lambda: next(timestamps),
    )

    evaluation = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    )
    assert evaluation.composition_evaluation.output_status == "no_action"
    assert evaluation.composition_evaluation.reason == (
        "strategy_latency_exceeded"
    )
    assert evaluation.composition_evaluation.strategy_input_id is not None
    assert evaluation.diagnostics[0].code == "strategy_latency_exceeded"
    assert len(proposals) == 0
