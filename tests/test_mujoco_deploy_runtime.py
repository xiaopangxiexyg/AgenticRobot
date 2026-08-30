from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import sim.mujoco.deploy.runtime as mujoco_deploy_runtime_module
from control_composition.composition import (
    CompositionContextDraft,
    CompositionInputDraft,
    CompositionSpec,
)
from control_composition.observability import SourceDifferenceSpec
from control_composition.information import (
    CompositionContextRequirement,
    CompositionInformationPlan,
)
from control_composition.eligibility import (
    CompositionSourceEligibilityDraft,
    CompositionSourceEligibilitySpec,
)
from dataset_contracts.replay import (
    REPLAY_ACTION_STAGE_SELECTED,
    REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
    REPLAY_BINDING_VERIFIED_COMPATIBLE,
    REPLAY_MODE_CANONICAL_ACTION,
    REPLAY_SCHEDULE_FIXED_RATE,
    REPLAY_SOURCE_INTERPRETATION_VERIFIED,
    ReplayScheduleSpec,
    ReplaySourceActionContract,
    ReplaySourceContract,
)
from dataset_interface.control_trace import (
    EpisodeControlTrace,
    materialize_actual_decision_view,
)
from dataset_contracts.completeness import (
    evaluate_control_trace_lineage,
    trace_requirement_profile,
)
from robot_motion.action import (
    ActionAdaptationResult,
    ActionProposalDraft,
    AuthoritativeControlRuntime,
    ExecutionObservationPlan,
    IMMEDIATE_QREF_SIM_V1,
    RobotAction,
    SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
    SharedCanonicalSafetyResult,
    assess_component_target_capability,
    build_action_adaptation_receipt,
    project_robot_action_v1_to_component_action,
    resolve_safety_anchor_commit_timeline,
)
from robot_motion.runtime import (
    CancellationAcknowledgementTracker,
    DispatchOwnerRegistry,
    ExecutionCancellationRequest,
    ExecutionEpochRegistry,
    LifecycleStartupError,
)
from robot_motion.replay import (
    ReplayProposalSource,
    ReplayProposalSourceConfig,
    build_replay_binding_assessment,
    create_replay_source_frame_emission,
)
from sim.contracts import TimingSpec
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.deploy import (
    MujocoCompositionRuntimeConfig,
    MujocoDeployConfig,
    MujocoDeployRuntime,
    MujocoDirectComponentProposalSource,
)
from sim.mujoco.deploy.observation import MUJOCO_QPOS_STREAM_ID_V1


class ManualClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class IncrementingClock:
    def __init__(self, value: float = 10.0, step: float = 0.001) -> None:
        self.value = float(value)
        self.step = float(step)

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


class RejectingRepresentationAdapter:
    def __init__(self, capability, *, clock) -> None:
        self._capability = capability
        self._clock = clock

    @property
    def component_target_capability(self):
        return self._capability

    def adapt(self, request):
        started_at = float(self._clock())
        assessment = assess_component_target_capability(
            self._capability,
            request.candidate_action,
        )
        receipt = build_action_adaptation_receipt(
            request=request,
            assessment=assessment,
            receiver_id="fixture-rejecting-adapter",
            processing_started_at=started_at,
            processing_completed_at=float(self._clock()),
            accepted=False,
            rejection_code="fixture_receiver_rejected",
            adapted_candidate=None,
        )
        return ActionAdaptationResult(
            request=request,
            capability_assessment=assessment,
            adapted_candidate=None,
            receipt=receipt,
        )


class FakeScene:
    arm_specs = {"left": object(), "right": object()}
    gripper_specs = {}

    def __init__(self) -> None:
        self.qpos = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }
        self.physics_commands: list[dict[str, np.ndarray]] = []

    def get_arm_qpos(self, side: str) -> np.ndarray:
        return self.qpos[side].copy()

    def step_kinematic_direct_qpos_debug(
        self,
        qpos_by_side,
        *,
        gripper_close_by_name,
        zero_controlled_qvel,
    ) -> None:
        del gripper_close_by_name, zero_controlled_qvel
        command = {
            side: np.asarray(qpos, dtype=float).reshape(6).copy()
            for side, qpos in qpos_by_side.items()
        }
        self.physics_commands.append(command)
        self.qpos = {side: value.copy() for side, value in command.items()}


def timing() -> TimingSpec:
    return TimingSpec(
        sample_hz=15.0,
        policy_hz=15.0,
        reference_hz=60.0,
        ik_hz=60.0,
        command_hz=120.0,
        physics_hz=1000.0,
    )


def build_runtime(
    *,
    clock: ManualClock | None = None,
    domain: str = "mujoco-env-0",
    composition_enabled: bool = False,
) -> tuple[MujocoDeployRuntime, FakeScene, ManualClock]:
    actual_clock = clock or ManualClock()
    scene = FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=actual_clock,
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id=f"session-{domain}",
            control_domain_id=domain,
            embodiment_instance_id=f"embodiment-{domain}",
        ),
        composition_runtime_config=(
            MujocoCompositionRuntimeConfig()
            if composition_enabled
            else None
        ),
        clock=actual_clock,
    )
    return runtime, scene, actual_clock


def test_runtime_start_failure_does_not_leave_a_live_execution_rig(
    monkeypatch,
) -> None:
    runtime, _scene, _clock = build_runtime()

    def fail_activation(**_kwargs):
        raise RuntimeError("synthetic producer activation failure")

    monkeypatch.setattr(runtime.core, "activate_producer", fail_activation)
    with pytest.raises(RuntimeError, match="synthetic producer activation failure"):
        runtime.start()

    assert runtime._started is False
    assert runtime.rig.is_ready() is False
    with pytest.raises(RuntimeError, match="fresh runtime"):
        runtime.start()


def test_runtime_close_stops_execution_before_terminal_trace_and_can_retry(
    monkeypatch,
) -> None:
    runtime, _scene, _clock = build_runtime()
    runtime.start()
    original = runtime.core.close_trace
    fail_once = True

    def fail_terminal_trace(*args, **kwargs):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise RuntimeError("synthetic terminal trace failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime.core, "close_trace", fail_terminal_trace)
    with pytest.raises(RuntimeError, match="synthetic terminal trace failure"):
        runtime.close()

    assert runtime._started is False
    assert runtime.rig.is_ready() is False
    runtime.close()
    assert runtime._lifecycle_terminal is True
    assert runtime._lifecycle_entered is False
    with pytest.raises(RuntimeError, match="fresh runtime"):
        runtime.start()


def test_runtime_rejects_lossy_control_identity_and_tick_counts() -> None:
    runtime, _scene, _clock = build_runtime()
    with pytest.raises(ValueError, match="producer_epoch.*non-negative integer"):
        replace(runtime.config, producer_epoch=1.5)

    runtime.start()
    try:
        with pytest.raises(
            ValueError,
            match="physics_ticks_after_dispatch.*non-negative integer",
        ):
            runtime.step(physics_ticks_after_dispatch=1.5)
        with pytest.raises(ValueError, match="producer_epoch.*non-negative integer"):
            runtime.activate_policy_node_producer(
                "fixture-policy-node",
                producer_epoch=True,
            )
    finally:
        runtime.close()


def test_composition_binding_preserves_runtime_compatibility_views() -> None:
    runtime, _scene, _clock = build_runtime(composition_enabled=True)

    assert runtime.composition_port is runtime._composition_binding
    assert runtime.policy_topology is runtime._composition_binding._topology
    assert (
        runtime.composition_strategy_control
        is runtime.composition_port.strategy_control
    )
    assert (
        runtime._composition_trace_emitter
        is runtime._composition_binding.trace_emitter
    )
    assert not hasattr(runtime.composition_port, "topology")
    with pytest.raises(
        RuntimeError,
        match=r"MujocoDeployRuntime\.start\(\) is required",
    ):
        runtime.composition_port.resolve_composition_context("not-admitted")
    for authority_operation in (
        "admit_evaluated_composition",
        "submit_evaluated_composition",
        "transition_execution_gate",
        "step",
    ):
        assert not hasattr(runtime.composition_port, authority_operation)


def test_authoritative_port_exposes_snapshots_without_mutable_owner_objects() -> None:
    runtime, _scene, _clock = build_runtime()
    port = runtime.authoritative_port

    assert port.profile.initial_control_mode == runtime.core.config.initial_control_mode
    assert port.profile.selection_group_ids == runtime.core.config.active_selection_groups
    assert (
        port.profile.authority_policy_hash
        == runtime.core.config.authority_policy.semantic_content_hash
    )
    assert (
        port.profile.decision_schedule_hash
        == runtime.core.config.decision_schedule.semantic_content_hash
    )
    assert port.capture_receiver_state() == runtime.core.capture_receiver_state()
    assert (
        port.capture_execution_epoch_snapshot()
        == runtime.core.execution_epochs.snapshot()
    )
    assert (
        port.capture_dispatch_owner_snapshot()
        == runtime.core.dispatch_owners.snapshot()
    )
    for mutable_owner in (
        "core",
        "coordinator",
        "dispatch_owners",
        "execution_epochs",
        "producer_epochs",
        "proposals",
        "scheduler",
    ):
        assert not hasattr(port, mutable_owner)


def test_general_runtime_does_not_allocate_optional_composition_state() -> None:
    runtime, scene, _clock = build_runtime()

    assert runtime.composition_runtime_config is None
    assert runtime.composition_port is None
    assert runtime.policy_topology is None
    assert runtime.composition_strategy_control is None
    assert runtime.composition_strategy is None
    with pytest.raises(RuntimeError, match="Composition runtime is disabled"):
        runtime.resolve_composition_context("not-admitted")
    assert scene.physics_commands == []


def test_composer_authority_requires_explicit_composition_runtime_config() -> None:
    runtime, _scene, _clock = build_runtime()

    with pytest.raises(
        ValueError,
        match="Composer authority requires an explicit",
    ):
        MujocoDeployRuntime(
            runtime.rig,
            replace(runtime.config, enable_composer_authority=True),
        )


def test_authoritative_runtime_can_adopt_one_transferred_owner_and_epoch_source() -> None:
    runtime, _scene, clock = build_runtime()
    config = runtime.core.config
    owners = DispatchOwnerRegistry(
        runtime_session_id=config.runtime_session_id,
        control_domain_id=config.control_domain_id,
        initial_authority_epoch=config.initial_authority_epoch,
        initial_execution_epoch=config.initial_execution_epoch,
        clock=clock,
    )
    shared_token = owners.grant_initial_owner(
        owner_runtime_id=config.owner_runtime_id,
        owner_kind="shared_arbitration_runtime",
    )
    epochs = ExecutionEpochRegistry(
        runtime_session_id=config.runtime_session_id,
        control_domain_id=config.control_domain_id,
        clock_domain_id=config.clock_domain.clock_domain_id,
        initial_execution_epoch=config.initial_execution_epoch,
        initial_segment_generation_by_group={"arms": 0},
        clock=clock,
    )

    adopted = AuthoritativeControlRuntime(
        config,
        dispatcher=runtime.rig.send_commands,
        backend_hold=lambda _cancellation: {"software_queue": True},
        backend_name="adopted-mujoco-rig",
        dispatch_owners=owners,
        dispatch_owner_token=shared_token,
        execution_epochs=epochs,
        clock=clock,
    )

    assert adopted.dispatch_owners is owners
    assert adopted.dispatch_owner_token is shared_token
    assert adopted.execution_epochs is epochs
    runtime.close()


def test_authoritative_runtime_refuses_a_legacy_token_or_partial_owner_injection() -> None:
    runtime, _scene, clock = build_runtime()
    config = runtime.core.config
    owners = DispatchOwnerRegistry(
        runtime_session_id=config.runtime_session_id,
        control_domain_id=config.control_domain_id,
        initial_authority_epoch=config.initial_authority_epoch,
        initial_execution_epoch=config.initial_execution_epoch,
        clock=clock,
    )
    legacy_token = owners.grant_initial_owner(
        owner_runtime_id=config.owner_runtime_id,
        owner_kind="legacy_control_path",
    )
    epochs = ExecutionEpochRegistry(
        runtime_session_id=config.runtime_session_id,
        control_domain_id=config.control_domain_id,
        clock_domain_id=config.clock_domain.clock_domain_id,
        initial_execution_epoch=config.initial_execution_epoch,
        initial_segment_generation_by_group={"arms": 0},
        clock=clock,
    )

    with pytest.raises(ValueError, match="shared-arbitration"):
        AuthoritativeControlRuntime(
            config,
            dispatcher=runtime.rig.send_commands,
            backend_hold=lambda _cancellation: {"software_queue": True},
            backend_name="invalid-owner",
            dispatch_owners=owners,
            dispatch_owner_token=legacy_token,
            execution_epochs=epochs,
            clock=clock,
        )
    with pytest.raises(ValueError, match="must be injected together"):
        AuthoritativeControlRuntime(
            config,
            dispatcher=runtime.rig.send_commands,
            backend_hold=lambda _cancellation: {"software_queue": True},
            backend_name="partial-owner",
            dispatch_owners=owners,
            dispatch_owner_token=legacy_token,
            clock=clock,
        )
    runtime.close()


def test_legacy_to_shared_handoff_preserves_the_same_owner_and_epoch_registries() -> None:
    runtime, _scene, clock = build_runtime()
    initial = runtime.core.config
    owners = DispatchOwnerRegistry(
        runtime_session_id=initial.runtime_session_id,
        control_domain_id=initial.control_domain_id,
        initial_authority_epoch=0,
        initial_execution_epoch=0,
        clock=clock,
    )
    legacy_token = owners.grant_initial_owner(
        owner_runtime_id="legacy-real-frontend",
        owner_kind="legacy_control_path",
    )
    epochs = ExecutionEpochRegistry(
        runtime_session_id=initial.runtime_session_id,
        control_domain_id=initial.control_domain_id,
        clock_domain_id=initial.clock_domain.clock_domain_id,
        initial_execution_epoch=0,
        initial_segment_generation_by_group={"arms": 0},
        clock=clock,
    )
    cancellation = epochs.cancel(
        ExecutionCancellationRequest(
            reason="authority_handoff",
            authority_epoch=0,
            requested_scope="control_domain",
            source_event_ids=("operator-approved-handoff",),
        ),
        capability=initial.execution_capability,
    )
    transition = owners.begin_transition(
        current_token=legacy_token,
        transition_kind="authority_handoff",
        cancellation=cancellation,
        gate_status="hold",
        gate_epoch=1,
    )
    acknowledgements = CancellationAcknowledgementTracker(clock=clock)
    acknowledgements.register(cancellation)
    acknowledgements.acknowledge(
        cancellation.cancellation_id,
        acknowledgement_level="software_queue",
        clock_domain_id=initial.clock_domain.clock_domain_id,
        source_id="fake-legacy-worker",
    )
    shared_token = owners.complete_transition(
        transition_id=transition.transition_id,
        cancellation_receipt=acknowledgements.receipt(cancellation.cancellation_id),
        owner_runtime_id=initial.owner_runtime_id,
        owner_kind="shared_arbitration_runtime",
        new_authority_epoch=1,
        gate_status="hold",
    )
    adopted_config = replace(
        initial,
        initial_authority_epoch=1,
        initial_execution_epoch=1,
    )

    adopted = AuthoritativeControlRuntime(
        adopted_config,
        dispatcher=runtime.rig.send_commands,
        backend_hold=lambda _cancellation: {"software_queue": True},
        backend_name="handoff-mujoco-rig",
        dispatch_owners=owners,
        dispatch_owner_token=shared_token,
        execution_epochs=epochs,
        clock=clock,
    )

    assert adopted.dispatch_owners is owners
    assert adopted.execution_epochs is epochs
    assert adopted.dispatch_owner_token.owner_generation == 2
    assert adopted.dispatch_owner_token.owner_kind == "shared_arbitration_runtime"
    assert adopted.execution_epoch == 1
    runtime.close()


def action(left: float = 0.2, right: float = -0.2) -> RobotAction:
    return RobotAction.bimanual_qpos(
        left_arm=np.full(6, left, dtype=np.float32),
        right_arm=np.full(6, right, dtype=np.float32),
        timestamp=10.0,
        source="policy",
    )


def test_direct_component_ingress_has_fixed_agent_skill_identity() -> None:
    clock = ManualClock()
    scene = FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    source = MujocoDirectComponentProposalSource(
        producer_id="control-station-agent-skill-test",
        producer_epoch=7,
        authority_priority=23,
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id="session-agent-skill-direct",
            control_domain_id="domain-agent-skill-direct",
            embodiment_instance_id="embodiment-agent-skill-direct",
            enable_primary_policy_authority=False,
            enable_composer_authority=False,
            enable_replay_authority=False,
            initial_control_mode="agent_skill",
            direct_component_proposal_source=source,
        ),
        clock=clock,
    )
    native = project_robot_action_v1_to_component_action(
        action(),
        action_schema=runtime.action_schema,
    )

    with pytest.raises(RuntimeError, match="start"):
        runtime.submit_component_action(
            native,
            source_event_id="agent-target-before-start",
        )

    runtime.start()
    receipt = runtime.submit_component_action(
        native,
        source_event_id="agent-target:1",
        produced_at=10.0,
        target_effective_at=10.0,
        context_ids=("operation:1", "agent-observation:1"),
    )

    assert receipt.accepted
    assert receipt.proposal is not None
    assert receipt.proposal.producer_kind == "agent_skill"
    assert receipt.proposal.producer_id == source.producer_id
    assert receipt.proposal.producer_epoch == 7
    assert receipt.proposal.authority_priority == 23
    assert receipt.proposal.observation_cut_id is None
    assert receipt.proposal.context_ids == (
        "operation:1",
        "agent-observation:1",
    )
    assert receipt.proposal.audit_metadata == {
        "direct_component_source_event_id": "agent-target:1"
    }

    step = runtime.step(now=10.0)
    assert step.status == "dispatched"
    assert step.decision is not None
    assert step.decision.selected_by_group == {
        "arms": receipt.proposal.proposal_id
    }
    runtime.close()


def replay_source(runtime: MujocoDeployRuntime) -> ReplayProposalSource:
    source_action_contract = ReplaySourceActionContract(
        contract_id="sim-replay-source-action-v1",
        interpretation_status=REPLAY_SOURCE_INTERPRETATION_VERIFIED,
        action_stage=REPLAY_ACTION_STAGE_SELECTED,
        action_space="bimanual_qpos_v1",
        component_ids=("left_arm", "right_arm"),
        selection_group_by_component={
            "left_arm": "arms",
            "right_arm": "arms",
        },
        coupled_selection_groups=("arms",),
        pose_frame_contract_hash=None,
        component_schema_hash=runtime.catalog.semantic_content_hash,
        action_spec_hash_by_component={
            "left_arm": "rm65-arm-qpos6-joint-order-v1",
            "right_arm": "rm65-arm-qpos6-joint-order-v1",
        },
        units_convention_id="si_rad_joint_order_v1",
    )
    source_contract = ReplaySourceContract(
        contract_id="sim-replay-source-v1",
        replay_mode=REPLAY_MODE_CANONICAL_ACTION,
        dataset_schema="dexteleop_dataset_v2.4",
        source_dataset_id="sim-golden-source",
        episode_id="episode-replay-0",
        action_stage=REPLAY_ACTION_STAGE_SELECTED,
        action_field="decision.selected_action",
        source_action_space="bimanual_qpos_v1",
        source_component_schema_hash=runtime.catalog.semantic_content_hash,
    )
    binding, assessment = build_replay_binding_assessment(
        source_action_contract,
        runtime.target_runtime_action_contract,
        binding_id="sim-replay-to-mujoco-target-v1",
    )
    assert assessment.verdict == REPLAY_BINDING_VERIFIED_COMPATIBLE
    return ReplayProposalSource(
        ReplayProposalSourceConfig(
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            embodiment_instance_id=runtime.config.embodiment_instance_id,
            producer_id=runtime.config.replay_producer_id,
            producer_epoch=runtime.config.producer_epoch,
            action_schema=runtime.action_schema,
            source_contract=source_contract,
            source_action_contract=source_action_contract,
            target_runtime_contract=runtime.target_runtime_action_contract,
            derivation_receipt=None,
            schedule=ReplayScheduleSpec(
                schedule_id="sim-replay-30hz-v1",
                schedule_mode=REPLAY_SCHEDULE_FIXED_RATE,
                source_hz=30.0,
                target_hz=None,
                resampling_policy="none",
            ),
            target_binding_spec=binding,
            binding_assessment=assessment,
            requested_use=REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
        )
    )


def test_deployable_sim_runs_one_authoritative_control_and_trace_chain() -> None:
    runtime, scene, _clock = build_runtime()
    runtime.start()
    receipt = runtime.submit_policy_action(
        action(),
        policy_output_id="policy-output-0",
    )
    assert receipt.accepted

    step = runtime.step(now=10.0, physics_ticks_after_dispatch=12)

    assert step.status == "dispatched"
    assert step.decision is not None
    assert step.dispatch_validation is not None and step.dispatch_validation.accepted
    assert set(step.dispatch_receipts or {}) == {"left_arm", "right_arm"}
    assert step.dispatch_context is runtime.last_dispatch_context
    assert step.dispatch_context is not None
    assert step.dispatch_context.decision_id == step.decision.decision_id
    assert step.dispatch_context.safety_result_id in {
        fact.fact_id
        for fact in runtime.trace_facts
        if fact.event_type == "shared_safety_result"
    }
    assert dict(step.dispatch_context.group_segment_generation_by_group) == {
        "arms": 1
    }
    assert step.dispatch_context.execution_epoch == runtime.core.execution_epoch
    assert len(scene.physics_commands) == 12
    assert runtime.core.dispatch_owners.snapshot().current_token is not None
    assert runtime.core.dispatch_owners.snapshot().current_token.owner_kind == (
        "shared_arbitration_runtime"
    )

    event_types = tuple(fact.event_type for fact in runtime.trace_facts)
    for required in (
        "action_proposal",
        "proposal_snapshot",
        "arbitration_input_snapshot",
        "arbitration_computation",
        "arbitration_publication_attempt",
        "arbitration_decision",
        "action_adaptation_request",
        "target_capability_assessment",
        "adapted_action_candidate",
        "action_adaptation_receipt",
        "execution_directive",
        "shared_safety_result",
        "command",
        "dispatch_owner",
        "dispatch_receipt",
        "measurement_alignment",
    ):
        assert required in event_types
    assert "arbitration_evaluation" not in event_types
    decision_fact = next(
        fact for fact in runtime.trace_facts if fact.event_type == "arbitration_decision"
    )
    assert decision_fact.payload["selected_action"] is not None
    assert "would_select_action" not in decision_fact.payload
    runtime.close()
    assert runtime.trace_facts[-1].event_type == "runtime_trace_terminal"


def test_adaptation_rejection_never_reaches_safety_or_dispatch() -> None:
    runtime, scene, clock = build_runtime()
    runtime.core.action_representation_adapter = (
        RejectingRepresentationAdapter(
            runtime.component_target_capability,
            clock=clock,
        )
    )
    runtime.start()
    receipt = runtime.submit_policy_action(
        action(),
        policy_output_id="policy-output-adaptation-rejected",
    )
    assert receipt.accepted

    step = runtime.step(now=10.0, physics_ticks_after_dispatch=12)

    assert step.status in {
        "execution_cancelled",
        "execution_hold_active",
        "action_adaptation_rejected_keep_existing",
    }
    assert step.action_adaptation is not None
    assert step.action_adaptation.receipt.accepted is False
    assert step.safety_result is None
    assert step.command is None
    assert step.dispatch_receipts is None
    assert scene.physics_commands == []
    event_types = tuple(fact.event_type for fact in runtime.trace_facts)
    assert "action_adaptation_receipt" in event_types
    assert "shared_safety_result" not in event_types
    assert "command" not in event_types
    assert "dispatch_receipt" not in event_types
    runtime.close()


def test_sim_policy_chunk_admits_only_prefix_and_schedules_continuous_steps() -> None:
    runtime, scene, clock = build_runtime()
    runtime.start()
    submission = runtime.submit_policy_action_chunk(
        [action(0.1, -0.1), action(0.2, -0.2), action(0.3, -0.3)],
        policy_output_id="policy-chunk-0",
        action_chunk_execution_steps=2,
    )

    assert submission.accepted is True
    assert submission.source_chunk_size == 3
    assert submission.executed_chunk_size == 2
    assert submission.accepted_count == 2
    assert [
        receipt.proposal.chunk_keyframe_index
        for receipt in submission.receipts
        if receipt.proposal is not None
    ] == [0, 1]
    assert [
        receipt.proposal.target_effective_at
        for receipt in submission.receipts
        if receipt.proposal is not None
    ] == pytest.approx([10.0, 10.0 + 1.0 / 30.0])

    first = runtime.step(now=10.0, physics_ticks_after_dispatch=1)
    assert first.status == "dispatched"
    clock.value = 10.0 + 1.0 / 30.0
    second = runtime.step(now=clock.value, physics_ticks_after_dispatch=1)
    assert second.status == "dispatched"
    assert first.decision is not None and second.decision is not None
    assert float(
        first.decision.selected_action.value("left_arm").values[0]
    ) == pytest.approx(0.1)
    assert float(
        second.decision.selected_action.value("left_arm").values[0]
    ) == pytest.approx(0.2)
    next_submission = runtime.submit_policy_action_chunk(
        [action(0.4, -0.4), action(0.5, -0.5)],
        policy_output_id="policy-chunk-1",
        action_chunk_execution_steps=1,
    )
    assert next_submission.accepted is True
    assert next_submission.executed_chunk_size == 1
    assert next_submission.receipts[0].proposal is not None
    assert next_submission.receipts[0].proposal.target_effective_at == pytest.approx(
        10.0 + 2.0 / 30.0
    )
    runtime.close()


def test_sim_policy_chunk_rejects_invalid_lineage_inputs() -> None:
    runtime, _scene, _clock = build_runtime()
    runtime.start()
    with pytest.raises(ValueError, match="policy_output_id"):
        runtime.submit_policy_action_chunk(
            [action()],
            policy_output_id=" ",
        )
    with pytest.raises(ValueError, match="positive integer"):
        runtime.submit_policy_action_chunk(
            [action()],
            policy_output_id="invalid-prefix",
            action_chunk_execution_steps=1.5,
        )
    runtime.close()


def test_sim_policy_chunk_uses_one_default_clock_sample_for_produced_and_target() -> None:
    clock = IncrementingClock()
    runtime, _scene, _clock = build_runtime(clock=clock)
    runtime.start()
    clock.value = 10.0
    submission = runtime.submit_policy_action_chunk(
        [action()],
        policy_output_id="one-clock-sample",
    )
    proposal = submission.receipts[0].proposal
    assert proposal is not None
    assert proposal.produced_at == pytest.approx(10.0)
    assert proposal.target_effective_at == pytest.approx(10.0)
    runtime.close()


def test_policy_observation_cut_is_causal_and_linked_to_the_proposal() -> None:
    runtime, _scene, _clock = build_runtime()
    runtime.start()
    observation = runtime.capture_policy_observation()
    state = observation.payload_handles_by_stream[MUJOCO_QPOS_STREAM_ID_V1]
    qpos = state["arm_qpos_by_side"]
    policy_action = RobotAction.bimanual_qpos(
        left_arm=qpos["left"] + 0.1,
        right_arm=qpos["right"] - 0.1,
        source="fake-policy-from-observation",
    )
    receipt = runtime.submit_policy_action(
        policy_action,
        policy_output_id="policy-output-from-cut-0",
        observation_cut_id=observation.record.observation_cut_id,
    )

    assert receipt.accepted and receipt.proposal is not None
    assert receipt.proposal.observation_cut_id == observation.record.observation_cut_id
    observation_fact = next(
        fact for fact in runtime.trace_facts if fact.event_type == "observation_cut"
    )
    assert observation_fact.fact_id == observation.record.observation_cut_id
    assert observation_fact.payload["selected_sample_by_stream"][
        MUJOCO_QPOS_STREAM_ID_V1
    ]["available_at"] <= observation.record.cutoff_at


def test_decision_gate_does_not_create_a_catch_up_or_second_dispatch() -> None:
    runtime, scene, clock = build_runtime()
    runtime.start()
    runtime.submit_policy_action(action(), policy_output_id="policy-output-0")
    first = runtime.step(now=clock.value)
    second = runtime.step(now=clock.value)

    assert first.status == "dispatched"
    assert second.status == "not_due"
    assert runtime.rig.snapshot()["pending_reference_sequence"] == 1
    assert scene.physics_commands == []


def test_dispatch_epoch_mismatch_rejects_before_mujoco_backend() -> None:
    runtime, _scene, _clock = build_runtime()
    runtime.start()
    runtime.submit_policy_action(action(), policy_output_id="policy-output-0")
    runtime.core.dispatch_owners.observe_authority_epoch(
        authority_epoch=1,
        evidence_event_id="test-authority-epoch-1",
    )

    step = runtime.step(now=10.0)

    assert step.status == "dispatch_rejected"
    assert step.dispatch_validation is not None
    assert step.dispatch_validation.status == "registry_authority_epoch_stale"
    assert runtime.rig.snapshot()["last_reference_sequence"] is None
    assert runtime.last_dispatch_context is None
    assert "dispatch_receipt" not in {
        fact.event_type for fact in runtime.trace_facts
    }


def test_backend_exception_becomes_failed_receipts_without_false_success() -> None:
    runtime, _scene, _clock = build_runtime()
    runtime.start()
    runtime.submit_policy_action(action(), policy_output_id="policy-output-0")

    def fail_backend(_command):
        raise RuntimeError("simulated backend failure")

    runtime.core.dispatcher = fail_backend
    step = runtime.step(now=10.0)

    assert step.status == "dispatch_failed"
    assert step.dispatch_validation is not None and step.dispatch_validation.accepted
    assert set(step.dispatch_receipts or {}) == {"left_arm", "right_arm"}
    assert all(not value.accepted for value in (step.dispatch_receipts or {}).values())
    assert all(
        value.status == "backend_exception"
        for value in (step.dispatch_receipts or {}).values()
    )
    assert not runtime.core._active_segment_by_group
    receipt_fact = next(
        fact for fact in runtime.trace_facts if fact.event_type == "dispatch_receipt"
    )
    assert receipt_fact.payload["all_accepted"] is False


def test_control_domains_are_isolated() -> None:
    left, _left_scene, _ = build_runtime(domain="env-left")
    right, _right_scene, _ = build_runtime(domain="env-right")
    left.start()
    right.start()
    left.submit_policy_action(action(0.1, -0.1), policy_output_id="left-output")
    right.submit_policy_action(action(0.3, -0.3), policy_output_id="right-output")

    left_step = left.step(now=10.0)
    right_step = right.step(now=10.0)

    assert left_step.decision is not None and right_step.decision is not None
    assert left_step.decision.decision_id != right_step.decision.decision_id
    assert {fact.control_domain_id for fact in left.trace_facts} == {"env-left"}
    assert {fact.control_domain_id for fact in right.trace_facts} == {"env-right"}


def test_deployable_sim_preserves_the_arbiter_only_60hz_ceiling() -> None:
    with pytest.raises(ValueError, match="ActionArbiter rate"):
        MujocoDeployConfig(decision_hz=61.0)

    faster_non_arbiter_profile = TimingSpec(
        sample_hz=120.0,
        policy_hz=120.0,
        reference_hz=120.0,
        ik_hz=120.0,
        command_hz=240.0,
        physics_hz=1000.0,
    )
    assert faster_non_arbiter_profile.command_hz == 240.0


def test_target_admission_margin_is_opt_in_and_positive() -> None:
    assert MujocoDeployConfig().target_admission_margin_sec is None
    assert MujocoDeployConfig(
        target_admission_margin_sec=0.05
    ).target_admission_margin_sec == pytest.approx(0.05)
    with pytest.raises(ValueError, match="target_admission_margin_sec"):
        MujocoDeployConfig(target_admission_margin_sec=0.0)


def test_mujoco_reference_activation_profiles_do_not_redefine_existing_default() -> None:
    immediate, _scene, _clock = build_runtime()
    assert immediate.config.reference_activation_profile_id == IMMEDIATE_QREF_SIM_V1
    assert immediate.reference_activation_spec.activation_mode == "immediate_target"
    assert immediate.core.scheduler.spec.target_lead_time_sec == 0.0

    clock = ManualClock()
    scheduled_rig = MujocoPositionExecutionRig(
        FakeScene(),
        MujocoPositionExecutionConfig(
            timing=timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    scheduled = MujocoDeployRuntime(
        scheduled_rig,
        MujocoDeployConfig(
            reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
        ),
        clock=clock,
    )
    assert scheduled.reference_activation_spec.activation_mode == "scheduled_segment"
    assert scheduled.core.scheduler.spec.target_lead_time_sec == pytest.approx(0.05)
    scheduled.start()
    assert scheduled.reference_receiver is not None
    scheduled.close()


def test_non_off_execution_observation_plan_requires_a_sink_at_build_time() -> None:
    clock = ManualClock()
    rig = MujocoPositionExecutionRig(
        FakeScene(),
        MujocoPositionExecutionConfig(
            timing=timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    plan = ExecutionObservationPlan(
        plan_id="required-observation-without-sink",
        measurement_mode="full_frames",
        component_ids=("right_arm",),
    )

    with pytest.raises(
        ValueError,
        match="requires an ExecutionObservationSink",
    ):
        MujocoDeployRuntime(
            rig,
            MujocoDeployConfig(
                reference_activation_profile_id=(
                    SCHEDULED_REFERENCE_SEGMENT_SIM_V1
                ),
                enable_component_execution_handoff=True,
                execution_observation_plan=plan,
            ),
            clock=clock,
        )


def test_scheduled_qpos_receiver_admits_future_segment_without_early_activation() -> None:
    clock = ManualClock()
    scene = FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
        ),
        clock=clock,
    )
    runtime.start()
    runtime.submit_policy_action(action(0.25, -0.25), policy_output_id="scheduled-0")
    step = runtime.step(now=clock.value)

    assert step.status == "dispatched"
    assert runtime.reference_receiver is not None
    result = runtime.reference_receiver.last_result
    assert result is not None
    assert result.context_validation.accepted
    assert result.admission is not None and result.admission.accepted
    assert result.segment is not None
    assert result.segment.scheduled_from == pytest.approx(10.0 + 1.0 / 60.0)
    assert result.segment.scheduled_until == pytest.approx(10.05)
    assert result.segment.start_parent.source_kind == "measured_rebase"
    assert result.anchor_candidate is not None

    pending = resolve_safety_anchor_commit_timeline(
        runtime.reference_receiver.anchor_candidates,
        terminals=runtime.reference_receiver.terminals,
        control_domain_id=runtime.config.control_domain_id,
        selection_group_id="arms",
        safety_cutoff_at=10.01,
        target_effective_at=10.049,
        current_execution_epoch=0,
        expected_group_segment_generation=1,
        minimum_admission_guarantee="interpolation_segment_accepted",
    )
    assert pending.status == "pending_effective"

    active = resolve_safety_anchor_commit_timeline(
        runtime.reference_receiver.anchor_candidates,
        terminals=runtime.reference_receiver.terminals,
        control_domain_id=runtime.config.control_domain_id,
        selection_group_id="arms",
        safety_cutoff_at=10.05,
        target_effective_at=10.05,
        current_execution_epoch=0,
        expected_group_segment_generation=1,
        minimum_admission_guarantee="interpolation_segment_accepted",
    )
    assert active.status == "eligible"

    for _ in range(17):
        rig.step_physics()
    assert rig.snapshot()["last_reference_sequence"] is None
    rig.step_physics()
    assert rig.snapshot()["last_reference_sequence"] == 1
    assert len(rig.reference_activation_observations) == 1
    activation = rig.reference_activation_observations[0]
    assert activation.reference_segment_id == result.segment.reference_segment_id
    assert activation.activated_at_worker_time >= activation.scheduled_for
    assert rig.planner_command_frame_observations
    first_frame = rig.planner_command_frame_observations[0]
    assert first_frame.reference_segment_id == result.segment.reference_segment_id
    assert first_frame.execution_epoch == result.segment.execution_epoch
    assert (
        first_frame.group_segment_generation
        == result.segment.group_segment_generation
    )
    assert set(first_frame.planner_qcmd_by_side) == {"left", "right"}
    for _ in range(20):
        rig.step_physics()
    assert len(rig.planner_command_frame_observations) >= 2
    assert len(
        [
            fact
            for fact in runtime.trace_facts
            if fact.event_type == "arbitration_decision"
        ]
    ) == 1
    runtime.close()
    scheduled_event_types = {fact.event_type for fact in runtime.trace_facts}
    assert {
        "reference_receiver_validation",
        "reference_segment",
        "reference_segment_admission",
        "safety_anchor_commit_candidate",
        "reference_target_activation",
        "planner_command_frame",
    }.issubset(scheduled_event_types)
    profile = replace(
        trace_requirement_profile("deployable_sim_v1"),
        implementation_status="runtime_integrated",
    )
    assert evaluate_control_trace_lineage(runtime.trace_facts, profile=profile) == ()


def test_scheduled_dispatch_does_not_hold_receiver_lock_while_waiting_for_rig(
    monkeypatch,
) -> None:
    """The control-plane and physics lock order must not form a cycle."""

    clock = ManualClock()
    scene = FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
        ),
        clock=clock,
    )
    runtime.start()
    runtime.submit_policy_action(action(0.25, -0.25), policy_output_id="lock-order")
    assert runtime.reference_receiver is not None

    entered_measurement = threading.Event()
    release_measurement = threading.Event()
    original_measurement = rig.measured_qpos_by_side

    def blocked_measurement():
        entered_measurement.set()
        assert release_measurement.wait(timeout=1.0)
        return original_measurement()

    monkeypatch.setattr(rig, "measured_qpos_by_side", blocked_measurement)
    entered_rig_submission = threading.Event()
    release_rig_submission = threading.Event()
    original_submission = rig.send_scheduled_commands

    def blocked_submission(*args, **kwargs):
        entered_rig_submission.set()
        assert release_rig_submission.wait(timeout=1.0)
        return original_submission(*args, **kwargs)

    monkeypatch.setattr(rig, "send_scheduled_commands", blocked_submission)

    result: list[object] = []
    failure: list[BaseException] = []

    def dispatch() -> None:
        try:
            result.append(runtime.step(now=clock.value))
        except BaseException as exc:  # pragma: no cover - surfaced below
            failure.append(exc)

    worker = threading.Thread(target=dispatch, daemon=True)
    worker.start()
    try:
        assert entered_measurement.wait(timeout=1.0)
        measurement_receiver_available = threading.Event()

        def inspect_during_measurement() -> None:
            runtime.reference_receiver.terminals
            measurement_receiver_available.set()

        measurement_inspector = threading.Thread(
            target=inspect_during_measurement,
            daemon=True,
        )
        measurement_inspector.start()
        assert measurement_receiver_available.wait(timeout=0.5), (
            "scheduled dispatch retained the receiver lock while reading the Rig"
        )
        release_measurement.set()
        assert entered_rig_submission.wait(timeout=1.0)

        receiver_available = threading.Event()

        def inspect_receiver() -> None:
            runtime.reference_receiver.terminals
            receiver_available.set()

        inspector = threading.Thread(target=inspect_receiver, daemon=True)
        inspector.start()
        assert receiver_available.wait(timeout=0.5), (
            "scheduled dispatch retained the receiver lock while blocked on "
            "the Rig lock"
        )
    finally:
        release_measurement.set()
        release_rig_submission.set()

    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert not failure
    assert result and result[0].status == "dispatched"
    runtime.close()


def test_scheduled_qpos_receiver_uses_previous_endpoint_and_cancels_pending() -> None:
    clock = ManualClock()
    scene = FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
        ),
        clock=clock,
    )
    runtime.start()
    runtime.submit_policy_action(action(0.1, -0.1), policy_output_id="scheduled-0")
    assert runtime.step(now=10.0).status == "dispatched"
    assert runtime.core.active_execution_segment_by_group == {}
    assert runtime.core.pending_execution_segment_by_group["arms"].group_segment_generation == 1
    for _ in range(18):
        rig.step_physics()
    assert runtime.core.active_execution_segment_by_group["arms"].group_segment_generation == 1
    assert runtime.core.pending_execution_segment_by_group == {}

    clock.value = 10.0 + 1.0 / 30.0
    runtime.submit_policy_action(action(0.2, -0.2), policy_output_id="scheduled-1")
    assert runtime.step(now=clock.value).status == "dispatched"
    assert runtime.reference_receiver is not None
    first, second = runtime.reference_receiver.segments
    assert second.start_parent.source_kind == "committed_safe_endpoint"
    assert second.start_parent.source_fact_id == first.reference_segment_id
    assert second.scheduled_from == pytest.approx(first.scheduled_until)
    assert runtime.reference_receiver.generation_activations[-1].active_generation == 1
    assert runtime.reference_receiver.generation_reservations[-1].reserved_generation == 2
    assert runtime.core.active_execution_segment_by_group["arms"].group_segment_generation == 1
    assert runtime.core.pending_execution_segment_by_group["arms"].group_segment_generation == 2

    # Successor admission reserves generation 2 but generation 1 remains live
    # until the successor's START.  The 120 Hz planner may keep deriving work
    # from generation 1 during that interval.
    for _ in range(34):
        rig.step_physics()
    assert runtime.core.active_execution_segment_by_group["arms"].group_segment_generation == 2
    assert runtime.core.pending_execution_segment_by_group == {}
    assert runtime.reference_receiver.generation_activations[-1].active_generation == 2
    retired = next(
        item
        for item in runtime.reference_receiver.terminals
        if item.reference_segment_id == first.reference_segment_id
    )
    assert retired.status == "retired_by_successor_activation"
    with pytest.raises(RuntimeError, match="generation is not active"):
        runtime.reference_receiver._require_active_generation(
            first.reference_segment_id,
            first.execution_epoch,
            first.group_segment_generation,
            second.scheduled_from,
        )

    clock.value = 10.3
    cancelled = runtime.step(now=clock.value)
    assert cancelled.status == "execution_cancelled"
    assert {
        item.reference_segment_id for item in runtime.reference_receiver.terminals
    } == {first.reference_segment_id, second.reference_segment_id}
    first_terminal = next(
        item
        for item in runtime.reference_receiver.terminals
        if item.reference_segment_id == first.reference_segment_id
    )
    assert first_terminal.last_reference_frame_id is not None
    runtime.close()


def test_second_pending_scheduled_segment_without_relation_fails_closed() -> None:
    clock = ManualClock()
    scene = FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
        ),
        clock=clock,
    )
    runtime.start()
    runtime.submit_policy_action(action(0.1, -0.1), policy_output_id="pending-0")
    assert runtime.step(now=10.0).status == "dispatched"

    # Do not advance physics to the first START.  The second control decision
    # must not silently overwrite the pending mailbox/reservation.
    clock.value = 10.0 + 1.0 / 30.0
    runtime.submit_policy_action(action(0.2, -0.2), policy_output_id="pending-1")
    rejected = runtime.step(now=clock.value)

    assert rejected.status == "dispatch_failed"
    assert runtime.reference_receiver is not None
    assert len(runtime.reference_receiver.segments) == 1
    assert len(runtime.reference_receiver.segment_attempts) == 2
    assert runtime.reference_receiver.admissions[-1].accepted is False
    assert (
        runtime.reference_receiver.admissions[-1].rejection_reason
        == "pending_generation_reservation_exists"
    )
    assert runtime.core.active_execution_segment_by_group == {}
    assert runtime.core.pending_execution_segment_by_group["arms"].group_segment_generation == 1
    runtime.close()


def test_immediate_qref_does_not_invent_scheduled_segment_lineage() -> None:
    runtime, _scene, _clock = build_runtime()
    runtime.start()
    runtime.submit_policy_action(action(), policy_output_id="immediate-0")
    assert runtime.step(now=10.0).status == "dispatched"
    for _ in range(20):
        runtime.rig.step_physics()

    assert runtime.reference_receiver is None
    assert runtime.rig.reference_activation_observations == ()
    assert runtime.rig.planner_command_frame_observations == ()
    runtime.close()
    assert not {
        "reference_receiver_validation",
        "reference_segment",
        "reference_segment_admission",
        "safety_anchor_commit_candidate",
        "reference_target_activation",
        "planner_command_frame",
        "reference_segment_terminal",
    }.intersection(fact.event_type for fact in runtime.trace_facts)


def test_reference_trace_sync_projects_only_new_append_only_suffix(
    monkeypatch,
) -> None:
    runtime, _scene, _clock = build_runtime()
    projected: list[tuple[str, str]] = []
    recorded: list[object] = []

    def project_target(_context, item):
        projected.append(("target", str(item)))
        return SimpleNamespace(fact_id=f"target:{item}")

    def project_admission(_context, item):
        projected.append(("admission", str(item)))
        return SimpleNamespace(fact_id=f"admission:{item}")

    monkeypatch.setattr(
        mujoco_deploy_runtime_module,
        "project_component_execution_target",
        project_target,
    )
    monkeypatch.setattr(
        mujoco_deploy_runtime_module,
        "project_component_execution_target_admission",
        project_admission,
    )
    monkeypatch.setattr(
        runtime.core,
        "record_external_facts",
        lambda facts: recorded.extend(facts),
    )

    history_size = 256
    runtime._immediate_component_targets.extend(
        f"t{index}" for index in range(history_size)
    )
    runtime._immediate_component_target_admissions.extend(
        f"a{index}" for index in range(history_size)
    )
    runtime._sync_reference_trace()

    assert projected[:2] == [("target", "t0"), ("target", "t1")]
    assert projected[history_size - 1] == ("target", "t255")
    assert projected[history_size] == ("admission", "a0")
    assert projected[-1] == ("admission", "a255")
    assert len(projected) == 2 * history_size
    assert len(recorded) == 2 * history_size

    # An idle sync must perform no projection work: it does not reconstruct
    # old facts merely to discover that their IDs were already published.
    runtime._sync_reference_trace()
    assert len(projected) == 2 * history_size
    assert len(recorded) == 2 * history_size

    runtime._immediate_component_targets.append("t256")
    runtime._immediate_component_target_admissions.append("a256")
    runtime._sync_reference_trace()
    assert projected[-2:] == [("target", "t256"), ("admission", "a256")]
    assert len(projected) == 2 * history_size + 2
    assert len(recorded) == 2 * history_size + 2


def test_reference_trace_sync_rejects_history_rewind(monkeypatch) -> None:
    runtime, _scene, _clock = build_runtime()
    monkeypatch.setattr(
        mujoco_deploy_runtime_module,
        "project_component_execution_target",
        lambda _context, item: SimpleNamespace(fact_id=f"target:{item}"),
    )
    monkeypatch.setattr(
        mujoco_deploy_runtime_module,
        "project_component_execution_target_admission",
        lambda _context, item: SimpleNamespace(fact_id=f"admission:{item}"),
    )
    monkeypatch.setattr(
        runtime.core,
        "record_external_facts",
        lambda _facts: None,
    )

    runtime._immediate_component_targets.append("t0")
    runtime._immediate_component_target_admissions.append("a0")
    runtime._sync_reference_trace()
    runtime._immediate_component_targets.clear()

    with pytest.raises(RuntimeError, match="history rewound"):
        runtime._sync_reference_trace()


def test_sim_deploy_module_does_not_import_teleop_runtime() -> None:
    module = importlib.import_module("sim.mujoco.deploy.runtime")
    source_modules = {
        value.__module__
        for value in (module.MujocoDeployConfig, module.MujocoDeployRuntime)
    }
    assert all(not name.startswith("teleop_runtime") for name in source_modules)


def test_actual_decision_view_uses_completed_and_variable_dt_authoritative_facts() -> None:
    runtime, _scene, clock = build_runtime()
    runtime.start()
    runtime.submit_policy_action(action(0.1, -0.1), policy_output_id="output-0")
    first = runtime.step(now=10.0)
    assert first.status == "dispatched"
    clock.value = 10.04
    runtime.submit_policy_action(
        action(0.2, -0.2),
        policy_output_id="output-1",
        target_effective_at=10.04,
    )
    second = runtime.step(now=10.04)
    assert second.status == "dispatched"
    runtime.close()

    records = tuple(
        SimpleNamespace(fact=fact, journal_sequence=index)
        for index, fact in enumerate(runtime.trace_facts, start=1)
    )
    trace = EpisodeControlTrace(
        dataset_root=Path("/unused"),
        episode_uuid="sim-episode-0",
        trace_session_id=runtime.config.runtime_session_id,
        profile_id="deployable_sim_v1",
        committed=False,
        capabilities=("control_trace_core",),
        seal_payload={},
        commit_payload=None,
        episode_records=records,
        dependency_records=(),
        wal_relative_path="unused",
        wal_tail_status="clean_eof",
    )

    rows = materialize_actual_decision_view(trace)

    assert len(rows) == 2
    assert rows[0].dt_sec == pytest.approx(0.04)
    assert rows[0].decided_at == pytest.approx(10.0)
    assert rows[0].published_at == pytest.approx(10.0)
    assert rows[0].effective_interval_end == pytest.approx(10.04)
    assert rows[0].action_stage == "selected_pre_safety"
    assert rows[0].is_resampled is False
    assert rows[0].selected_action is not None
    assert not hasattr(rows[0], "action")
    assert rows[0].dispatch_receipt_ids
    assert rows[0].measurement_alignment_ids
    assert rows[1].dt_sec is None
    assert rows[0].command_id != rows[1].command_id


def test_verified_replay_uses_the_same_authoritative_sim_chain() -> None:
    clock = ManualClock()
    runtime, scene, _ = build_runtime(clock=clock)
    runtime = MujocoDeployRuntime(
        runtime.rig,
        MujocoDeployConfig(
            runtime_session_id="replay-sim-session",
            control_domain_id="replay-sim-domain",
            embodiment_instance_id="replay-sim-embodiment",
            enable_replay_authority=True,
        ),
        clock=clock,
    )
    runtime.start()
    submission = runtime.submit_replay_action(
        replay_source(runtime),
        action(0.35, -0.35),
        source_event_id="episode-replay-0:frame:0",
        source_path="/datasets/sim-golden-source",
        source_episode_index=0,
        group_key="episode-replay-0",
        local_frame_index=0,
        source_action_index=0,
        source_timestamp=0.0,
    )

    assert submission.proposal_admission is not None
    assert submission.proposal_admission.accepted
    assert submission.authorization_admission is not None
    assert submission.authorization_admission.accepted
    step = runtime.step(now=clock.value)

    assert step.status == "dispatched"
    assert step.computation is not None
    selected_ids = set(step.computation.selected_by_group.values())
    assert selected_ids == {submission.proposal_admission.proposal_id}
    assert len(scene.physics_commands) == 0
    event_types = {fact.event_type for fact in runtime.trace_facts}
    assert {
        "replay_binding_assessment",
        "candidate_binding_eligibility",
        "replay_execution_authorization",
        "candidate_authorization_admission",
        "candidate_authorization_snapshot",
        "arbitration_decision",
        "shared_safety_result",
        "dispatch_receipt",
    }.issubset(event_types)
    runtime.close()
    profile = replace(
        trace_requirement_profile("deployable_sim_v1"),
        implementation_status="runtime_integrated",
    )
    assert evaluate_control_trace_lineage(
        runtime.trace_facts,
        profile=profile,
    ) == ()


def test_existing_replay_source_frame_is_admitted_without_rebuilding_identity() -> None:
    clock = ManualClock()
    base, _scene, _ = build_runtime(clock=clock)
    runtime = MujocoDeployRuntime(
        base.rig,
        MujocoDeployConfig(
            runtime_session_id="replay-emission-sim-session",
            control_domain_id="replay-emission-sim-domain",
            embodiment_instance_id="replay-emission-sim-embodiment",
            enable_replay_authority=True,
            initial_control_mode="replay",
            clock_domain_id="local_monotonic",
        ),
        clock=clock,
    )
    runtime.start()
    source = replay_source(runtime)
    emission = create_replay_source_frame_emission(
        source_path="/datasets/sim-golden-source",
        source_episode_index=0,
        group_key="episode-replay-0",
        local_frame_index=0,
        source_action_index=0,
        source_relative_time=0.0,
        action=action(0.35, -0.35),
        observed_at=clock.value,
        target_effective_at=clock.value,
        effective_until=clock.value + 1.0 / 30.0,
        clock_domain_id="local_monotonic",
    )

    submission = runtime.submit_replay_source_frame_emission(source, emission)

    assert submission.source_emission is emission
    assert submission.source_emission.emission_id == emission.emission_id
    assert submission.proposal_admission is not None
    assert submission.proposal_admission.accepted
    assert submission.authorization_admission is not None
    assert submission.authorization_admission.accepted
    source_fact = next(
        fact
        for fact in runtime.trace_facts
        if fact.event_type == "replay_source_frame_emission"
    )
    assert source_fact.fact_id == emission.emission_id
    runtime.close()


def test_existing_replay_source_frame_rejects_implicit_cross_clock_binding() -> None:
    clock = ManualClock()
    base, _scene, _ = build_runtime(clock=clock)
    runtime = MujocoDeployRuntime(
        base.rig,
        MujocoDeployConfig(
            runtime_session_id="replay-clock-mismatch-session",
            control_domain_id="replay-clock-mismatch-domain",
            embodiment_instance_id="replay-clock-mismatch-embodiment",
            enable_replay_authority=True,
            initial_control_mode="replay",
        ),
        clock=clock,
    )
    runtime.start()
    emission = create_replay_source_frame_emission(
        source_path="/datasets/sim-golden-source",
        source_episode_index=0,
        group_key="episode-replay-0",
        local_frame_index=0,
        source_action_index=0,
        source_relative_time=0.0,
        action=action(),
        observed_at=clock.value,
        target_effective_at=clock.value,
        effective_until=clock.value + 1.0 / 30.0,
        clock_domain_id="local_monotonic",
    )
    trace_count = len(runtime.trace_facts)

    with pytest.raises(ValueError, match="explicit clock domain"):
        runtime.submit_replay_source_frame_emission(
            replay_source(runtime),
            emission,
        )

    assert len(runtime.trace_facts) == trace_count
    runtime.close()


def test_external_trace_fact_index_preserves_duplicate_semantics() -> None:
    clock = ManualClock()
    runtime, _scene, _ = build_runtime(clock=clock)
    runtime.start()
    context = runtime.core.trace_context
    first = context.fact(
        fact_id="external-source-validation-1",
        event_type="source_validation",
        occurred_at=clock.value,
        payload={
            "validation_id": "external-source-validation-1",
            "source_event_id": "external-source-1",
            "accepted": True,
            "reasons": (),
        },
    )
    conflicting = context.fact(
        fact_id=first.fact_id,
        event_type="source_validation",
        occurred_at=clock.value,
        payload={
            "validation_id": "external-source-validation-1",
            "source_event_id": "external-source-1",
            "accepted": False,
            "reasons": ("changed",),
        },
    )
    before = len(runtime.trace_facts)

    runtime.record_external_facts((first, first))

    assert len(runtime.trace_facts) == before + 1
    assert runtime.trace_facts[-1].semantic_content_hash == (
        first.semantic_content_hash
    )
    with pytest.raises(ValueError, match="reused with new content"):
        runtime.record_external_facts((conflicting,))
    assert len(runtime.trace_facts) == before + 1
    runtime.close()


def test_replay_publication_fails_closed_without_admitted_target_evidence() -> None:
    clock = ManualClock()
    base, _scene, _ = build_runtime(clock=clock)
    runtime = MujocoDeployRuntime(
        base.rig,
        MujocoDeployConfig(
            runtime_session_id="replay-no-evidence-session",
            control_domain_id="replay-no-evidence-domain",
            embodiment_instance_id="replay-no-evidence-embodiment",
            enable_replay_authority=True,
        ),
        clock=clock,
    )
    runtime.start()
    draft = ActionProposalDraft(
        runtime_session_id=runtime.config.runtime_session_id,
        control_domain_id=runtime.config.control_domain_id,
        embodiment_instance_id=runtime.config.embodiment_instance_id,
        producer_kind="replay",
        producer_id=runtime.config.replay_producer_id,
        producer_epoch=runtime.config.producer_epoch,
        sequence=0,
        action=project_robot_action_v1_to_component_action(
            action(),
            action_schema=runtime.action_schema,
        ),
        produced_at=clock.value,
        produced_clock_domain_id=runtime.config.clock_domain_id,
        effective_from=clock.value,
        effective_until=clock.value + 0.1,
        target_effective_at=clock.value,
        effective_clock_domain_id=runtime.config.clock_domain_id,
    )
    receipt = runtime.core.admit_proposal(draft, source_event_id="unbound-replay")
    assert receipt.accepted

    step = runtime.step(now=clock.value)

    assert step.status == "execution_cancelled"
    assert step.publication_attempt is not None
    assert step.publication_attempt.rejection_reason == (
        "candidate_authorization_rejected:authorization_missing"
    )
    assert runtime.rig.snapshot()["last_reference_sequence"] is None
    assert step.cancellation is not None
    assert step.cancellation_receipt is not None
    assert step.cancellation_receipt.complete
    assert "arbitration_decision" not in {
        fact.event_type for fact in runtime.trace_facts
    }
    assert {
        "execution_directive",
        "execution_cancellation",
        "execution_cancellation_ack",
    }.issubset({fact.event_type for fact in runtime.trace_facts})


def test_no_action_expires_segment_once_then_keeps_backend_hold() -> None:
    runtime, _scene, clock = build_runtime()
    runtime.start()
    runtime.submit_policy_action(action(), policy_output_id="policy-output-0")
    first = runtime.step(now=clock.value)
    assert first.status == "dispatched"
    first_epoch = runtime.core.execution_epoch

    clock.value = 10.1
    expired = runtime.step(now=clock.value)
    assert expired.status == "execution_cancelled"
    assert expired.decision is None
    assert expired.cancellation is not None
    assert runtime.core.execution_epoch == first_epoch + 1
    assert runtime.rig.snapshot()["pending_reference_sequence"] is None

    clock.value += 1.0 / runtime.config.decision_hz
    repeated = runtime.step(now=clock.value)
    assert repeated.status == "execution_hold_active"
    assert repeated.cancellation is None
    assert runtime.core.execution_epoch == first_epoch + 1
    runtime.close()
    profile = replace(
        trace_requirement_profile("deployable_sim_v1"),
        implementation_status="runtime_integrated",
    )
    assert evaluate_control_trace_lineage(
        runtime.trace_facts,
        profile=profile,
    ) == ()

    records = tuple(
        SimpleNamespace(fact=fact, journal_sequence=index)
        for index, fact in enumerate(runtime.trace_facts, start=1)
    )
    trace = EpisodeControlTrace(
        dataset_root=Path("/unused"),
        episode_uuid="sim-cancel-episode-0",
        trace_session_id=runtime.config.runtime_session_id,
        profile_id="deployable_sim_v1",
        committed=False,
        capabilities=("control_trace_core",),
        seal_payload={},
        commit_payload=None,
        episode_records=records,
        dependency_records=(),
        wal_relative_path="unused",
        wal_tail_status="clean_eof",
    )
    rows = materialize_actual_decision_view(trace)
    assert len(rows) == 1
    assert rows[0].decision_id == first.decision.decision_id


def test_execution_gate_hold_cancels_old_epoch_and_rejects_old_permit() -> None:
    runtime, _scene, clock = build_runtime()
    runtime.start()
    runtime.submit_policy_action(action(), policy_output_id="policy-output-0")
    first = runtime.step(now=clock.value)
    assert first.status == "dispatched"
    old_epoch = runtime.core.execution_epoch
    old_permit = runtime.core.dispatch_owners.issue_motion_permit(
        token=runtime.core.dispatch_owner_token,
        authority_epoch=first.decision.authority_epoch,
        execution_epoch=old_epoch,
        command_lineage_id="queued-command-before-hold",
        directive_lineage_id=first.directive.directive_id,
    )

    runtime.set_execution_gate(
        "hold",
        reason="test_operator_hold",
        source_event_ids=("operator-hold-0",),
    )
    assert runtime.core.execution_epoch == old_epoch + 1
    assert runtime.rig.snapshot()["pending_reference_sequence"] is None
    clock.value += 1.0 / runtime.config.decision_hz
    held = runtime.step(now=clock.value)

    assert held.status == "execution_hold_active"
    assert held.decision is None
    assert held.cancellation is None
    assert runtime.core.execution_epoch == old_epoch + 1
    rejected = runtime.core.dispatch_owners.authorize(
        old_permit,
        current_authority_epoch=first.decision.authority_epoch,
        current_execution_epoch=runtime.core.execution_epoch,
    )
    assert not rejected.accepted
    assert rejected.status == "execution_epoch_mismatch"


def test_typed_gate_transition_returns_exact_cancellation_evidence() -> None:
    runtime, _scene, _clock = build_runtime()
    runtime.start()

    outcome = runtime.transition_execution_gate(
        "hold",
        reason="typed_hold_outcome",
        source_event_ids=("operator-hold-typed",),
    )

    assert outcome.event.status == "hold"
    assert outcome.snapshot.status == "hold"
    assert outcome.cancellation is not None
    assert outcome.cancellation_receipt is not None
    assert outcome.cancellation_receipt.complete
    assert outcome.cancellation_dispatch_validation is not None
    assert outcome.cancellation_dispatch_validation.accepted
    assert outcome.lower_event_ids == tuple(
        dict.fromkeys(
            (
                outcome.event.event_id,
                *(fact.fact_id for fact in outcome.trace_facts),
            )
        )
    )
    assert outcome.event.event_id in outcome.lower_event_ids
    assert outcome.cancellation.cancellation_id in outcome.lower_event_ids


def test_authority_transition_requires_completed_lower_execution_fence() -> None:
    runtime, _scene, _clock = build_runtime()
    runtime.start()

    with pytest.raises(RuntimeError, match="execution fence"):
        runtime.core.admit_authority_event(
            event_type="mode",
            control_mode=runtime.config.initial_control_mode,
            reason="unfenced_authority_change",
        )

    hold = runtime.transition_execution_gate(
        "hold",
        reason="authority_transition_fence",
        source_event_ids=("authority-transition-test",),
    )
    assert hold.cancellation_receipt is not None
    assert hold.cancellation_receipt.complete
    before = runtime.core.dispatch_owners.snapshot()

    event = runtime.core.admit_authority_event(
        event_type="mode",
        control_mode=runtime.config.initial_control_mode,
        reason="fenced_authority_change",
    )

    after = runtime.core.dispatch_owners.snapshot()
    assert event.authority_epoch == before.current_authority_epoch + 1
    assert after.current_authority_epoch == event.authority_epoch
    assert after.current_execution_epoch == hold.cancellation.execution_epoch


def test_ready_gate_does_not_reuse_old_hold_as_authority_fence() -> None:
    runtime, _scene, _clock = build_runtime()
    runtime.start()
    hold = runtime.transition_execution_gate(
        "hold",
        reason="establish_old_hold",
    )
    assert hold.cancellation_receipt is not None
    assert hold.cancellation_receipt.complete
    runtime.transition_execution_gate("ready", reason="leave_execution_fence")

    with pytest.raises(RuntimeError, match="execution fence"):
        runtime.core.admit_authority_event(
            event_type="mode",
            control_mode=runtime.config.initial_control_mode,
            reason="ready_is_not_a_fence",
        )


def test_forced_gate_cancellation_advances_epoch_while_backend_already_held() -> None:
    runtime, _scene, _clock = build_runtime()
    runtime.start()
    first = runtime.transition_execution_gate(
        "hold",
        reason="establish_backend_hold",
    )
    assert first.cancellation is not None
    runtime.set_execution_gate("ready", reason="fixture_ready_without_dispatch")

    quiescent = runtime.transition_execution_gate(
        "hold",
        reason="quiescent_hold",
    )
    assert quiescent.cancellation is None
    epoch_before_forced = runtime.core.execution_epoch
    forced = runtime.transition_execution_gate(
        "hold",
        reason="transaction_requires_new_epoch",
        force_cancellation=True,
    )
    assert forced.cancellation is not None
    assert forced.cancellation_receipt is not None
    assert forced.cancellation_receipt.complete
    assert runtime.core.execution_epoch == epoch_before_forced + 1


def test_activate_producer_returns_the_admitted_activation() -> None:
    runtime, _scene, _clock = build_runtime()
    runtime.start()
    producer_id = "hot-swappable-policy"

    activation = runtime.core.activate_producer(
        producer_id=producer_id,
        producer_epoch=1,
        reason="test_hot_swap",
    )

    assert activation.producer_id == producer_id
    assert activation.producer_epoch == 1
    assert runtime.core.producer_epochs.current_epoch(producer_id) == 1
    assert runtime.trace_facts[-1].fact_id == activation.activation_id


def test_latched_fault_cannot_be_cleared_by_a_ready_gate_event() -> None:
    runtime, _scene, _clock = build_runtime()
    runtime.start()
    runtime.set_execution_gate(
        "fault",
        reason="synthetic_external_fault",
        source_event_ids=("external-fault-0",),
    )

    with pytest.raises(RuntimeError, match="explicit recovery"):
        runtime.set_execution_gate(
            "ready",
            reason="unsafe_direct_ready",
            source_event_ids=("unsafe-ready-0",),
        )


def test_slow_context_fast_residual_composes_into_one_authoritative_sim_path() -> None:
    clock = ManualClock()
    scene = FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id="slow-fast-sim-session",
            control_domain_id="slow-fast-sim-domain",
            embodiment_instance_id="slow-fast-sim-embodiment",
            enable_composer_authority=True,
        ),
        composition_runtime_config=MujocoCompositionRuntimeConfig(),
        clock=clock,
    )
    runtime.start()
    for producer_id in ("slow-context", "slow-base", "fast-residual"):
        runtime.activate_policy_node_producer(producer_id)

    slow_observation = runtime.capture_policy_observation(
        policy_node_id="slow-vision",
        trigger_reason="fixed_10hz",
    )
    context_receipt = runtime.submit_composition_context(
        CompositionContextDraft(
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            producer_id="slow-context",
            producer_epoch=runtime.config.producer_epoch,
            sequence=0,
            producer_node_id="slow-vision",
            context_channel="motion-plan",
            context_kind="trajectory_plan",
            semantic_payload={"fixture_plan": "bimanual-base-v1"},
            produced_at=clock.value,
            produced_clock_domain_id=runtime.config.clock_domain_id,
            effective_from=clock.value,
            effective_until=clock.value + 0.2,
            effective_clock_domain_id=runtime.config.clock_domain_id,
            payload_contract_hash="synthetic-motion-plan-v1",
            observation_cut_id=slow_observation.record.observation_cut_id,
            policy_output_id="slow-output-0",
        )
    )
    assert context_receipt.accepted and context_receipt.context is not None
    context = context_receipt.context
    base = RobotAction.bimanual_qpos(
        left_arm=np.full(6, 0.1, dtype=np.float32),
        right_arm=np.full(6, -0.1, dtype=np.float32),
        timestamp=clock.value,
        source="synthetic_slow_base",
    )
    base_receipt = runtime.submit_composition_input(
        CompositionInputDraft(
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            producer_id="slow-base",
            producer_epoch=runtime.config.producer_epoch,
            sequence=0,
            input_role="base",
            action=project_robot_action_v1_to_component_action(
                base,
                action_schema=runtime.action_schema,
            ),
            produced_at=clock.value,
            produced_clock_domain_id=runtime.config.clock_domain_id,
            effective_from=clock.value,
            effective_until=clock.value + 0.2,
            effective_clock_domain_id=runtime.config.clock_domain_id,
            context_ids=(context.context_id,),
            observation_cut_id=slow_observation.record.observation_cut_id,
            policy_output_id="slow-base-output-0",
        )
    )
    assert base_receipt.accepted

    clock.value = 10.025
    fast_observation = runtime.capture_policy_observation(
        policy_node_id="fast-tactile",
        trigger_reason="fixed_40hz",
    )
    context_snapshot = runtime.capture_composition_context_snapshot(
        target_effective_at=clock.value,
        required_channels=("motion-plan",),
    )
    assert context_snapshot.selected_context_id_by_channel["motion-plan"] == (
        context.context_id
    )
    residual = RobotAction.bimanual_qpos(
        left_arm=np.full(6, 0.02, dtype=np.float32),
        right_arm=np.full(6, -0.02, dtype=np.float32),
        timestamp=clock.value,
        source="synthetic_fast_residual",
    )
    residual_receipt = runtime.submit_composition_input(
        CompositionInputDraft(
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            producer_id="fast-residual",
            producer_epoch=runtime.config.producer_epoch,
            sequence=0,
            input_role="residual",
            action=project_robot_action_v1_to_component_action(
                residual,
                action_schema=runtime.action_schema,
            ),
            produced_at=clock.value,
            produced_clock_domain_id=runtime.config.clock_domain_id,
            effective_from=clock.value,
            effective_until=clock.value + 0.05,
            effective_clock_domain_id=runtime.config.clock_domain_id,
            context_ids=(context.context_id,),
            observation_cut_id=fast_observation.record.observation_cut_id,
            policy_output_id="fast-output-0",
        )
    )
    assert residual_receipt.accepted
    spec = CompositionSpec(
        spec_id="mujoco-slow-base-fast-residual-v1",
        algebra="base_plus_bounded_residual",
        input_roles=("base", "residual"),
        required_roles=("base", "residual"),
        residual_limit_by_component={
            "left_arm": 0.05,
            "right_arm": 0.05,
        },
    )
    shadow = runtime.evaluate_composition_shadow(
        spec=spec,
        target_effective_at=clock.value,
    )
    assert shadow.composition_evaluation.output_status == "ready"
    assert shadow.proposal_admission is None
    assert shadow.source_difference_observation is None
    assert shadow.source_difference_error is None
    assert shadow.evaluation_record is not None
    assert shadow.evaluation_record.composition_spec is spec
    assert shadow.evaluation_record.input_snapshot is shadow.input_snapshot
    assert shadow.evaluation_record.composition_evaluation is (
        shadow.composition_evaluation
    )
    assert shadow.evaluation_record.selected_inputs_by_role.keys() == {
        "base",
        "residual",
    }
    assert {
        event.event_kind for event in shadow.domain_events_emitted
        } == {
            "action_payload_schema",
            "component_action_schema",
            "composition_spec",
            "composition_temporal_selection_spec",
            "composition_input_snapshot",
            "composition_evaluation",
        }
    assert shadow.diagnostics == ()
    assert len(runtime.core.proposals) == 0

    invalid_diagnostic = runtime.evaluate_composition_shadow(
        spec=spec,
        target_effective_at=clock.value,
        source_difference_spec=SourceDifferenceSpec(
            spec_id="invalid-diagnostic-must-not-block-composition",
            composition_spec_hash=spec.semantic_content_hash,
            compared_reference_role_pairs=(("base", "unknown-role"),),
        ),
    )
    assert invalid_diagnostic.composition_evaluation.output_status == "ready"
    assert invalid_diagnostic.source_difference_observation is None
    assert invalid_diagnostic.source_difference_error is not None
    assert "unknown composition role" in (
        invalid_diagnostic.source_difference_error
    )
    assert invalid_diagnostic.evaluation_record is not None
    assert len(invalid_diagnostic.diagnostics) == 1
    assert invalid_diagnostic.diagnostics[0].stage == "source_difference"
    assert invalid_diagnostic.diagnostics[0].recoverable
    assert "composition_spec" not in {
        event.event_kind for event in invalid_diagnostic.domain_events_emitted
    }
    assert sum(
        fact.event_type == "composition_spec" for fact in runtime.trace_facts
    ) == 1
    assert len(runtime.core.proposals) == 0

    with pytest.raises(ValueError, match="partially cover coupled"):
        runtime.submit_evaluated_composition(
            shadow,
            component_coverage=("left_arm",),
            selection_groups=("arms",),
        )
    with pytest.raises(ValueError, match="outside the active MuJoCo catalog"):
        runtime.submit_evaluated_composition(
            shadow,
            component_coverage=("left_arm", "right_arm", "left_hand"),
            selection_groups=("arms", "left_hand"),
        )
    assert len(runtime.core.proposals) == 0

    submission = runtime.submit_evaluated_composition(
        shadow,
    )
    assert submission.composition_evaluation.output_status == "ready"
    assert submission.proposal_admission is not None
    assert submission.proposal_admission.accepted
    assert submission.evaluation_record is shadow.evaluation_record
    assert len(runtime.core.proposals) == 1
    duplicate = runtime.submit_evaluated_composition(shadow)
    assert duplicate.proposal_admission is submission.proposal_admission
    assert duplicate.evaluation_record is shadow.evaluation_record
    assert len(runtime.core.proposals) == 1

    step = runtime.step(now=clock.value, physics_ticks_after_dispatch=1)

    assert step.status == "dispatched"
    assert step.decision is not None
    np.testing.assert_allclose(
        step.decision.selected_action.value("left_arm").values,
        np.full(6, 0.12),
    )
    np.testing.assert_allclose(
        step.decision.selected_action.value("right_arm").values,
        np.full(6, -0.12),
    )
    assert step.computation.attribution["left_arm"]["context_ids"] == (
        context.context_id,
    )
    assert slow_observation.record.policy_node_id == "slow-vision"
    assert fast_observation.record.policy_node_id == "fast-tactile"
    assert slow_observation.record.observation_cut_id != (
        fast_observation.record.observation_cut_id
    )
    event_types = {fact.event_type for fact in runtime.trace_facts}
    assert {
        "composition_context",
        "composition_context_snapshot",
        "composition_input",
        "composition_input_snapshot",
        "composition_evaluation",
        "action_proposal",
        "arbitration_decision",
        "dispatch_receipt",
    }.issubset(event_types)
    runtime.close()
    profile = replace(
        trace_requirement_profile("deployable_sim_v1"),
        implementation_status="runtime_integrated",
    )
    assert evaluate_control_trace_lineage(
        runtime.trace_facts,
        profile=profile,
    ) == ()


def test_information_plan_rejects_unrequested_context_before_retention() -> None:
    clock = ManualClock()
    scene = FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    plan = CompositionInformationPlan(
        plan_id="motion-plan-only-v1",
        context_requirements=(
            CompositionContextRequirement(
                channel="motion-plan",
                accepted_payload_contract_hashes=("synthetic-motion-plan-v1",),
                payload_access="inline",
                maximum_age_sec=0.1,
            ),
        ),
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id="information-plan-session",
            control_domain_id="information-plan-domain",
            embodiment_instance_id="information-plan-embodiment",
        ),
        composition_runtime_config=MujocoCompositionRuntimeConfig(
            information_plan=plan,
        ),
        clock=clock,
    )
    runtime.start()
    runtime.activate_policy_node_producer("context-source")

    def context_draft(*, channel: str, contract: str, sequence: int):
        return CompositionContextDraft(
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            producer_id="context-source",
            producer_epoch=runtime.config.producer_epoch,
            sequence=sequence,
            context_channel=channel,
            context_kind="fixture",
            semantic_payload={"value": sequence},
            produced_at=clock.value,
            produced_clock_domain_id=runtime.config.clock_domain_id,
            effective_from=clock.value,
            effective_until=clock.value + 0.5,
            effective_clock_domain_id=runtime.config.clock_domain_id,
            payload_contract_hash=contract,
        )

    unrequested = runtime.submit_composition_context(
        context_draft(
            channel="vision-frame",
            contract="synthetic-vision-v1",
            sequence=0,
        )
    )
    assert not unrequested.accepted
    assert unrequested.rejection_code == "information_channel_not_requested"
    assert len(runtime.policy_topology.contexts) == 0

    wrong_contract = runtime.submit_composition_context(
        context_draft(
            channel="motion-plan",
            contract="unexpected-contract",
            sequence=1,
        )
    )
    assert not wrong_contract.accepted
    assert wrong_contract.rejection_code == (
        "information_payload_contract_mismatch"
    )
    assert len(runtime.policy_topology.contexts) == 0

    accepted = runtime.submit_composition_context(
        context_draft(
            channel="motion-plan",
            contract="synthetic-motion-plan-v1",
            sequence=2,
        )
    )
    assert accepted.accepted
    assert len(runtime.policy_topology.contexts) == 1
    assert (
        runtime.policy_topology.context_requirement_for_channel("motion-plan")
        is plan.context_requirements[0]
    )
    runtime.activate_policy_node_producer("input-source")
    input_receipt = runtime.submit_composition_input(
        CompositionInputDraft(
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            producer_id="input-source",
            producer_epoch=runtime.config.producer_epoch,
            sequence=0,
            input_role="base",
            action=project_robot_action_v1_to_component_action(
                RobotAction.bimanual_qpos(
                    left_arm=np.zeros(6, dtype=np.float32),
                    right_arm=np.zeros(6, dtype=np.float32),
                    timestamp=clock.value,
                    source="information-plan-fixture",
                ),
                action_schema=runtime.action_schema,
            ),
            produced_at=clock.value,
            produced_clock_domain_id=runtime.config.clock_domain_id,
            effective_from=clock.value,
            effective_until=clock.value + 0.5,
            effective_clock_domain_id=runtime.config.clock_domain_id,
            context_ids=(accepted.context_id,),
        )
    )
    assert input_receipt.accepted
    spec = CompositionSpec(
        spec_id="information-plan-component-union-v1",
        algebra="component_union",
        input_roles=("base",),
        required_roles=("base",),
    )
    fresh = runtime.evaluate_composition_shadow(
        spec=spec,
        target_effective_at=clock.value,
    )
    assert fresh.composition_evaluation.output_status == "ready"
    assert fresh.evaluation_record is not None
    assert fresh.evaluation_record.information_plan is plan
    assert fresh.evaluation_record.required_context_channels == ("motion-plan",)

    clock.value += 0.15
    stale = runtime.evaluate_composition_shadow(
        spec=spec,
        target_effective_at=clock.value,
    )
    assert stale.composition_evaluation.output_status == "no_action"
    assert stale.composition_evaluation.reason == "stale_context:motion-plan"
    composition_event_types = [
        fact.event_type
        for fact in runtime.trace_facts
        if fact.event_type.startswith("composition_")
    ]
    assert composition_event_types.count("composition_information_plan") == 1
    assert composition_event_types.index("composition_information_plan") < (
        composition_event_types.index("composition_context_admission")
    )
    runtime.close()


def test_source_eligibility_is_default_off_and_opt_in_shadow_holds_no_motion() -> None:
    default_runtime, default_scene, _ = build_runtime(
        domain="source-eligibility-default-off",
        composition_enabled=True,
    )
    default_runtime.start()
    default_runtime.activate_policy_node_producer("virtual-policy")
    disabled = default_runtime.submit_composition_source_eligibility(
        CompositionSourceEligibilityDraft(
            runtime_session_id=default_runtime.config.runtime_session_id,
            control_domain_id=default_runtime.config.control_domain_id,
            source_slot_id="virtual-policy-slot",
            producer_id="virtual-policy",
            producer_epoch=default_runtime.config.producer_epoch,
            status_epoch=1,
            sequence=0,
            status="available",
            observed_at=10.0,
            observed_clock_domain_id=default_runtime.config.clock_domain_id,
            effective_from=10.0,
            lease_expires_at=10.2,
            effective_clock_domain_id=default_runtime.config.clock_domain_id,
        )
    )
    assert not disabled.accepted
    assert disabled.rejection_code == "source_eligibility_disabled"
    assert default_scene.physics_commands == []
    default_runtime.close()

    clock = ManualClock()
    scene = FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id="source-eligibility-shadow-session",
            control_domain_id="source-eligibility-shadow-domain",
            embodiment_instance_id="source-eligibility-shadow-embodiment",
        ),
        composition_runtime_config=MujocoCompositionRuntimeConfig(
            enable_source_eligibility=True,
        ),
        clock=clock,
    )
    runtime.start()
    runtime.activate_policy_node_producer("virtual-policy")
    spec = CompositionSpec(
        spec_id="virtual-policy-source-eligibility-v1",
        algebra="component_union",
        input_roles=("policy",),
        required_roles=("policy",),
    )
    source_spec = CompositionSourceEligibilitySpec(
        spec_id="virtual-policy-source-slot-v1",
        composition_spec_hash=spec.semantic_content_hash,
        source_slot_by_role={"policy": "virtual-policy-slot"},
        producer_id_by_source_slot={
            "virtual-policy-slot": "virtual-policy",
        },
    )
    action_receipt = runtime.submit_composition_input(
        CompositionInputDraft(
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            producer_id="virtual-policy",
            producer_epoch=runtime.config.producer_epoch,
            sequence=0,
            input_role="policy",
            action=project_robot_action_v1_to_component_action(
                RobotAction.bimanual_qpos(
                    left_arm=np.full(6, 0.1, dtype=np.float32),
                    right_arm=np.full(6, -0.1, dtype=np.float32),
                    timestamp=clock.value,
                    source="virtual-policy",
                ),
                action_schema=runtime.action_schema,
            ),
            produced_at=clock.value,
            produced_clock_domain_id=runtime.config.clock_domain_id,
            effective_from=clock.value,
            effective_until=clock.value + 0.5,
            effective_clock_domain_id=runtime.config.clock_domain_id,
        )
    )
    assert action_receipt.accepted
    available = runtime.submit_composition_source_eligibility(
        CompositionSourceEligibilityDraft(
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            source_slot_id="virtual-policy-slot",
            producer_id="virtual-policy",
            producer_epoch=runtime.config.producer_epoch,
            status_epoch=1,
            sequence=0,
            status="available",
            observed_at=clock.value,
            observed_clock_domain_id=runtime.config.clock_domain_id,
            effective_from=clock.value,
            lease_expires_at=clock.value + 0.2,
            effective_clock_domain_id=runtime.config.clock_domain_id,
        )
    )
    assert available.accepted
    ready = runtime.evaluate_composition_shadow(
        spec=spec,
        source_eligibility_spec=source_spec,
        target_effective_at=clock.value,
    )
    assert ready.composition_evaluation.output_status == "ready"
    assert ready.proposal_admission is None

    clock.value += 0.01
    paused = runtime.submit_composition_source_eligibility(
        CompositionSourceEligibilityDraft(
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            source_slot_id="virtual-policy-slot",
            producer_id="virtual-policy",
            producer_epoch=runtime.config.producer_epoch,
            status_epoch=2,
            sequence=0,
            status="paused",
            observed_at=clock.value,
            observed_clock_domain_id=runtime.config.clock_domain_id,
            effective_from=clock.value,
            lease_expires_at=clock.value + 0.2,
            effective_clock_domain_id=runtime.config.clock_domain_id,
            rearm_required=True,
        )
    )
    assert paused.accepted
    blocked = runtime.evaluate_composition_shadow(
        spec=spec,
        source_eligibility_spec=source_spec,
        target_effective_at=clock.value,
    )
    assert blocked.composition_evaluation.output_status == "no_action"
    assert blocked.input_snapshot.missing_roles == ("policy",)
    assert blocked.input_snapshot.source_ineligibility_reason_by_role[
        "policy"
    ] == "source_status:paused"
    assert blocked.proposal_admission is None
    assert len(runtime.core.proposals) == 0
    assert scene.physics_commands == []
    assert {
        "composition_source_eligibility_spec",
        "composition_source_eligibility_state",
        "composition_source_eligibility_snapshot",
        "composition_source_eligibility_binding",
    }.issubset({fact.event_type for fact in runtime.trace_facts})
    runtime.close()


def test_multirate_topology_coalesces_40hz_inputs_at_the_30hz_arbiter() -> None:
    clock = ManualClock()
    scene = FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id="multirate-sim-session",
            control_domain_id="multirate-sim-domain",
            embodiment_instance_id="multirate-sim-embodiment",
            enable_composer_authority=True,
            decision_hz=30.0,
        ),
        composition_runtime_config=MujocoCompositionRuntimeConfig(),
        clock=clock,
    )
    runtime.start()
    for producer_id in ("slow-context", "slow-base", "fast-residual"):
        runtime.activate_policy_node_producer(producer_id)

    spec = CompositionSpec(
        spec_id="mujoco-multirate-base-residual-v1",
        algebra="base_plus_bounded_residual",
        input_roles=("base", "residual"),
        required_roles=("base", "residual"),
        residual_limit_by_component={
            "left_arm": 0.05,
            "right_arm": 0.05,
        },
    )
    slow_ticks = {index / 10.0: index for index in range(3)}
    fast_ticks = {index / 40.0: index for index in range(9)}
    decision_ticks = {index / 30.0: index for index in range(7)}
    timeline = sorted(set(slow_ticks) | set(fast_ticks) | set(decision_ticks))
    current_context_id: str | None = None
    current_base_value: float | None = None
    current_residual_value: float | None = None
    dispatched_steps = []
    expected_selected_values: list[float] = []

    for offset in timeline:
        clock.value = 10.0 + offset
        if offset in slow_ticks:
            index = slow_ticks[offset]
            observation = runtime.capture_policy_observation(
                policy_node_id="slow-vision",
                trigger_reason="fixed_10hz",
            )
            context_receipt = runtime.submit_composition_context(
                CompositionContextDraft(
                    runtime_session_id=runtime.config.runtime_session_id,
                    control_domain_id=runtime.config.control_domain_id,
                    producer_id="slow-context",
                    producer_epoch=runtime.config.producer_epoch,
                    sequence=index,
                    producer_node_id="slow-vision",
                    context_channel="motion-plan",
                    context_kind="trajectory_plan",
                    semantic_payload={"slow_tick": index},
                    produced_at=clock.value,
                    produced_clock_domain_id=runtime.config.clock_domain_id,
                    effective_from=clock.value,
                    effective_until=clock.value + 0.101,
                    effective_clock_domain_id=runtime.config.clock_domain_id,
                    payload_contract_hash="synthetic-motion-plan-v1",
                    observation_cut_id=observation.record.observation_cut_id,
                    policy_output_id=f"slow-context-output-{index}",
                )
            )
            assert context_receipt.accepted and context_receipt.context is not None
            current_context_id = context_receipt.context.context_id
            current_base_value = 0.1 + 0.01 * index
            base_receipt = runtime.submit_composition_input(
                CompositionInputDraft(
                    runtime_session_id=runtime.config.runtime_session_id,
                    control_domain_id=runtime.config.control_domain_id,
                    producer_id="slow-base",
                    producer_epoch=runtime.config.producer_epoch,
                    sequence=index,
                    input_role="base",
                    action=project_robot_action_v1_to_component_action(
                        RobotAction.bimanual_qpos(
                            left_arm=np.full(6, current_base_value, dtype=np.float32),
                            right_arm=np.full(6, -current_base_value, dtype=np.float32),
                            timestamp=clock.value,
                            source="synthetic_slow_base",
                        ),
                        action_schema=runtime.action_schema,
                    ),
                    produced_at=clock.value,
                    produced_clock_domain_id=runtime.config.clock_domain_id,
                    effective_from=clock.value,
                    effective_until=clock.value + 0.101,
                    effective_clock_domain_id=runtime.config.clock_domain_id,
                    context_ids=(current_context_id,),
                    observation_cut_id=observation.record.observation_cut_id,
                    policy_output_id=f"slow-base-output-{index}",
                )
            )
            assert base_receipt.accepted

        if offset in fast_ticks:
            index = fast_ticks[offset]
            assert current_context_id is not None
            observation = runtime.capture_policy_observation(
                policy_node_id="fast-tactile",
                trigger_reason="fixed_40hz",
            )
            current_residual_value = 0.001 * (index + 1)
            residual_receipt = runtime.submit_composition_input(
                CompositionInputDraft(
                    runtime_session_id=runtime.config.runtime_session_id,
                    control_domain_id=runtime.config.control_domain_id,
                    producer_id="fast-residual",
                    producer_epoch=runtime.config.producer_epoch,
                    sequence=index,
                    input_role="residual",
                    action=project_robot_action_v1_to_component_action(
                        RobotAction.bimanual_qpos(
                            left_arm=np.full(
                                6, current_residual_value, dtype=np.float32
                            ),
                            right_arm=np.full(
                                6, -current_residual_value, dtype=np.float32
                            ),
                            timestamp=clock.value,
                            source="synthetic_fast_residual",
                        ),
                        action_schema=runtime.action_schema,
                    ),
                    produced_at=clock.value,
                    produced_clock_domain_id=runtime.config.clock_domain_id,
                    effective_from=clock.value,
                    effective_until=clock.value + 0.026,
                    effective_clock_domain_id=runtime.config.clock_domain_id,
                    context_ids=(current_context_id,),
                    observation_cut_id=observation.record.observation_cut_id,
                    policy_output_id=f"fast-output-{index}",
                )
            )
            assert residual_receipt.accepted

        if offset in decision_ticks:
            assert current_context_id is not None
            assert current_base_value is not None
            assert current_residual_value is not None
            context_snapshot = runtime.capture_composition_context_snapshot(
                target_effective_at=clock.value,
                required_channels=("motion-plan",),
            )
            assert context_snapshot.selected_context_id_by_channel[
                "motion-plan"
            ] == current_context_id
            submission = runtime.compose_and_submit(
                spec=spec,
                target_effective_at=clock.value,
            )
            assert submission.composition_evaluation.output_status == "ready"
            assert submission.proposal_admission is not None
            assert submission.proposal_admission.accepted
            step = runtime.step(
                now=clock.value,
                physics_ticks_after_dispatch=1,
            )
            assert step.status == "dispatched"
            assert step.computation is not None
            assert step.computation.attribution["left_arm"]["context_ids"] == (
                current_context_id,
            )
            dispatched_steps.append(step)
            expected_selected_values.append(
                current_base_value + current_residual_value
            )

    assert len(dispatched_steps) == 7
    assert len(scene.physics_commands) == 7
    assert len(runtime.core.proposals) == 7
    for step, expected in zip(dispatched_steps, expected_selected_values, strict=True):
        assert step.decision is not None
        np.testing.assert_allclose(
            step.decision.selected_action.value("left_arm").values,
            np.full(6, expected),
        )
    published_at = [step.decision.published_at for step in dispatched_steps]
    assert all(
        later - earlier + 1e-9 >= 1.0 / 30.0
        for earlier, later in zip(published_at, published_at[1:])
    )
    event_types = [fact.event_type for fact in runtime.trace_facts]
    assert event_types.count("composition_context") == 3
    assert event_types.count("composition_input") == 12
    assert event_types.count("composition_evaluation") == 7
    assert event_types.count("action_proposal") == 7
    assert event_types.count("arbitration_decision") == 7
    runtime.close()
    profile = replace(
        trace_requirement_profile("deployable_sim_v1"),
        implementation_status="runtime_integrated",
    )
    assert evaluate_control_trace_lineage(
        runtime.trace_facts,
        profile=profile,
    ) == ()


def test_shared_safety_fault_cancels_and_latches_next_gate_cut() -> None:
    runtime, _scene, clock = build_runtime()
    runtime.start()
    runtime.core.shared_safety = lambda _action, input_id: (
        SharedCanonicalSafetyResult(
            input_action_id=input_id,
            safe_action=None,
            reasons=("synthetic_fault",),
            status="fault",
        )
    )
    runtime.submit_policy_action(action(), policy_output_id="policy-output-0")

    faulted = runtime.step(now=clock.value)

    assert faulted.status == "safety_fault"
    assert faulted.decision is not None
    assert faulted.command is None
    assert faulted.cancellation is not None
    assert faulted.cancellation.fault_latched
    assert runtime.core.execution_epochs.snapshot().fault_latched
    clock.value += 1.0 / runtime.config.decision_hz
    next_step = runtime.step(now=clock.value)
    assert next_step.computation is not None
    assert next_step.computation.execution_gate_status == "fault"
    assert next_step.decision is None
    assert next_step.cancellation is None
    assert runtime.core.execution_epoch == faulted.cancellation.execution_epoch


def test_start_terminalizes_trace_without_starting_rig_after_control_failure() -> None:
    runtime, _scene, _clock = build_runtime()
    events: list[str] = []
    runtime.rig.start = lambda: events.append("rig.start")
    runtime.rig.close = lambda: events.append("rig.close")
    runtime.core.close_trace = lambda: events.append("trace.close")

    def fail_before_rig_start() -> None:
        events.append("core.configure")
        raise RuntimeError("synthetic core startup failure")

    runtime.core.record_compound_authority_profiles = fail_before_rig_start

    with pytest.raises(RuntimeError, match="synthetic core startup failure"):
        runtime.start()

    assert events == [
        "core.configure",
        "trace.close",
    ]
    assert runtime._started is False


def test_close_retries_trace_sync_after_rig_is_quiescent() -> None:
    runtime, _scene, _clock = build_runtime()
    runtime.start()
    events: list[str] = []

    def fail_trace_sync() -> None:
        events.append("trace.sync")
        raise RuntimeError("synthetic trace sync failure")

    runtime._sync_reference_trace = fail_trace_sync
    runtime.core.close_trace = lambda: events.append("trace.close")
    runtime.rig.close = lambda: events.append("rig.close")

    with pytest.raises(RuntimeError, match="synthetic trace sync failure"):
        runtime.close()

    assert events == ["rig.close", "trace.sync"]
    assert runtime._started is False
    assert runtime._rig_close_pending is False
    assert runtime._trace_close_pending is True

    runtime._sync_reference_trace = lambda: events.append("trace.sync:retry")
    runtime.close()

    assert events == [
        "rig.close",
        "trace.sync",
        "trace.sync:retry",
        "trace.close",
    ]


def test_close_retries_only_the_resource_that_remains_pending() -> None:
    runtime, _scene, _clock = build_runtime()
    runtime.start()
    events: list[str] = []
    rig_close_attempts = 0

    runtime._sync_reference_trace = lambda: events.append("trace.sync")
    runtime.core.close_trace = lambda: events.append("trace.close")

    def close_rig() -> None:
        nonlocal rig_close_attempts
        rig_close_attempts += 1
        events.append(f"rig.close:{rig_close_attempts}")
        if rig_close_attempts == 1:
            raise RuntimeError("synthetic rig join timeout")

    runtime.rig.close = close_rig

    with pytest.raises(RuntimeError, match="synthetic rig join timeout"):
        runtime.close()

    assert runtime._started is False
    runtime.close()

    assert events == [
        "rig.close:1",
        "rig.close:2",
        "trace.sync",
        "trace.close",
    ]
    assert runtime._started is False


def test_start_rollback_failure_keeps_only_failed_resource_closeable() -> None:
    runtime, _scene, _clock = build_runtime()
    events: list[str] = []
    rig_close_attempts = 0
    runtime._sync_reference_trace = lambda: events.append("trace.sync")
    runtime.core.close_trace = lambda: events.append("trace.close")

    def fail_rig_start() -> None:
        events.append("rig.start")
        raise RuntimeError("synthetic rig startup failure")

    def close_rig() -> None:
        nonlocal rig_close_attempts
        rig_close_attempts += 1
        events.append(f"rig.close:{rig_close_attempts}")
        if rig_close_attempts == 1:
            raise RuntimeError("synthetic rollback join timeout")

    runtime.rig.start = fail_rig_start
    runtime.rig.close = close_rig

    with pytest.raises(LifecycleStartupError) as caught:
        runtime.start()

    assert isinstance(caught.value.start_error, RuntimeError)
    assert tuple(
        failure.label for failure in caught.value.rollback_failures
    ) == ("mujoco_execution_rig.close",)
    assert events == ["rig.start", "rig.close:1"]

    runtime.close()

    assert events == [
        "rig.start",
        "rig.close:1",
        "rig.close:2",
        "trace.sync",
        "trace.close",
    ]


def test_close_does_not_retry_trace_after_terminal_sink_failure() -> None:
    runtime, _scene, _clock = build_runtime()
    runtime.start()
    events: list[str] = []
    runtime._sync_reference_trace = lambda: events.append("trace.sync")
    runtime.rig.close = lambda: events.append("rig.close")

    def fail_terminal_sink(fact) -> None:
        if fact.event_type == "runtime_trace_terminal":
            events.append("trace.terminal_sink")
            raise RuntimeError("synthetic terminal sink failure")

    runtime.core.trace_sink = fail_terminal_sink

    with pytest.raises(RuntimeError, match="synthetic terminal sink failure"):
        runtime.close()

    assert events == ["rig.close", "trace.sync", "trace.terminal_sink"]
    assert runtime._started is False
    assert sum(
        fact.event_type == "runtime_trace_terminal"
        for fact in runtime.core.trace_facts
    ) == 1

    runtime.close()

    assert events == ["rig.close", "trace.sync", "trace.terminal_sink"]
