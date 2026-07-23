from __future__ import annotations

import math

import numpy as np
import pytest

from robot_motion.action import (
    BIMANUAL_QPOS_V1,
    ActionArbiter,
    ActionComponentCatalog,
    ActionComponentSpec,
    AuthorityPolicySpec,
    AuthorityStore,
    CompositionInputDraft,
    CompositionInputStore,
    CompositionSpec,
    DecisionScheduleSpec,
    DecisionScheduler,
    DeterministicComposer,
    ProposalBuffer,
    RobotAction,
    build_composite_action_proposal_draft,
    build_composition_input_snapshot,
    project_composition_inputs,
)
from robot_motion.action.arbitration_input import capture_arbitration_input
from robot_motion.deploy import (
    PolicyContextDraft,
    PolicyContextStore,
    PolicyNodeOutput,
    PolicyNodeRequest,
    PolicyNodeRuntime,
    PolicyNodeSpec,
    build_policy_context_snapshot,
    project_policy_contexts,
)
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.execution_gate import ExecutionGateStore
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.observation_cut import (
    ObservationBundle,
    ObservationCutRecord,
    ObservationSampleRef,
)
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


class Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class FakeSlowContextNode:
    def __init__(self) -> None:
        self.sequence = 0
        self.spec = PolicyNodeSpec(
            node_id="slow-vision",
            output_kind="policy_context",
            trigger_mode="fixed",
            nominal_hz=10.0,
            observation_contract_hash="slow-observation-v1",
            output_contract_hash="motion-plan-context-v1",
        )

    def run(self, request: PolicyNodeRequest) -> PolicyNodeOutput:
        frame = request.observation.record.selected_sample_by_stream["rgbd"]
        sequence = self.sequence
        self.sequence += 1
        return PolicyNodeOutput(
            node_id=self.spec.node_id,
            output_kind="policy_context",
            produced_at=request.scheduled_at,
            policy_context=PolicyContextDraft(
                runtime_session_id="session-0",
                control_domain_id="robot-0",
                producer_id=self.spec.node_id,
                producer_epoch=1,
                sequence=sequence,
                policy_node_id=self.spec.node_id,
                context_channel="motion-plan",
                context_kind="semantic_subgoal",
                semantic_payload={
                    "rgbd_frame_id": frame.sample_id,
                    "base_qpos": np.full(6, 0.1 + 0.01 * sequence),
                },
                produced_at=request.scheduled_at,
                produced_clock_domain_id="control.monotonic",
                effective_from=request.target_effective_at,
                effective_until=request.target_effective_at + 0.2,
                effective_clock_domain_id="control.monotonic",
                context_contract_hash=self.spec.output_contract_hash,
                observation_cut_id=request.observation.record.observation_cut_id,
                inference_id=f"slow-inference:{sequence}",
                policy_output_id=f"slow-output:{sequence}",
            ),
        )


class FakeFastActionNode:
    def __init__(self) -> None:
        self.sequence = 0
        self.spec = PolicyNodeSpec(
            node_id="fast-tactile",
            output_kind="composition_input",
            trigger_mode="fixed",
            nominal_hz=40.0,
            observation_contract_hash="fast-observation-v1",
            output_contract_hash="bounded-residual-input-v1",
            required_context_channels=("motion-plan",),
        )

    def run(self, request: PolicyNodeRequest) -> PolicyNodeOutput:
        snapshot = request.context_snapshot
        if snapshot is None or snapshot.missing_channels:
            return PolicyNodeOutput(
                node_id=self.spec.node_id,
                output_kind="no_output",
                produced_at=request.scheduled_at,
                no_output_reason="motion_plan_unavailable",
            )
        context_id = snapshot.selected_context_id_by_channel["motion-plan"]
        tactile = request.observation.record.selected_sample_by_stream["tactile"]
        sequence = self.sequence
        self.sequence += 1
        residual = 0.01 + 0.001 * (tactile.sequence % 3)
        return PolicyNodeOutput(
            node_id=self.spec.node_id,
            output_kind="composition_input",
            produced_at=request.scheduled_at,
            composition_input=CompositionInputDraft(
                runtime_session_id="session-0",
                control_domain_id="robot-0",
                producer_id=self.spec.node_id,
                producer_epoch=1,
                sequence=sequence,
                input_role="residual",
                action=RobotAction(
                    action_space=BIMANUAL_QPOS_V1,
                    arm_qpos_by_side={"left": np.full(6, residual)},
                    source="fake_fast_policy",
                ),
                produced_at=request.scheduled_at,
                produced_clock_domain_id="control.monotonic",
                effective_from=request.target_effective_at,
                effective_until=request.target_effective_at + 0.05,
                effective_clock_domain_id="control.monotonic",
                context_ids=(context_id,),
                observation_cut_id=request.observation.record.observation_cut_id,
                policy_output_id=f"fast-output:{sequence}",
            ),
        )


def observation_bundle(
    *,
    coordinator: ControlAdmissionCoordinator,
    policy_node_id: str,
    sequence: int,
    at: float,
    rgbd_sequence: int,
    tactile_sequence: int,
) -> ObservationBundle:
    cut = coordinator.capture_cut(cutoff_at=at).cut
    rgbd_at = 10.0 + rgbd_sequence / 15.0
    refs = {
        "rgbd": ObservationSampleRef(
            stream_id="rgbd",
            sample_id=f"rgbd:{rgbd_sequence}",
            sequence=rgbd_sequence,
            measured_at=rgbd_at,
            measured_clock_domain_id="control.monotonic",
            available_at=rgbd_at,
            available_clock_domain_id="control.monotonic",
            payload_object_id=f"rgbd-payload:{rgbd_sequence}",
        ),
        "tactile": ObservationSampleRef(
            stream_id="tactile",
            sample_id=f"tactile:{tactile_sequence}",
            sequence=tactile_sequence,
            measured_at=10.0 + tactile_sequence / 100.0,
            measured_clock_domain_id="control.monotonic",
            available_at=10.0 + tactile_sequence / 100.0,
            available_clock_domain_id="control.monotonic",
            payload_object_id=f"tactile-payload:{tactile_sequence}",
        ),
    }
    record = ObservationCutRecord.build(
        runtime_session_id="session-0",
        control_cut=cut,
        policy_node_id=policy_node_id,
        policy_node_epoch=1,
        sequence=sequence,
        built_at=at,
        built_clock_domain_id="control.monotonic",
        selected_sample_by_stream=refs,
        observation_contract_hash=f"{policy_node_id}-observation-v1",
        policy_io_layout_hash=f"{policy_node_id}-layout-v1",
        trigger_reason="fixed_schedule",
    )
    return ObservationBundle(
        record=record,
        payload_handles_by_stream={
            "rgbd": {"frame": rgbd_sequence},
            "tactile": {"sample": tactile_sequence},
        },
    )


def test_multirate_slow_context_fast_action_and_composite_lineage() -> None:
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
        runtime_session_id="session-0", coordinator=coordinator
    )
    for producer in ("slow-vision", "slow-base", "fast-tactile", "composer"):
        epochs.activate(
            producer_id=producer,
            producer_epoch=1,
            produced_at=9.0,
            produced_clock_domain_id="control.monotonic",
            reason="startup",
        )
    contexts = PolicyContextStore(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        capacity=16,
    )
    composition_inputs = CompositionInputStore(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        capacity=64,
    )
    proposals = ProposalBuffer(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        capacity=32,
    )
    catalog = ActionComponentCatalog(
        catalog_id="left-arm-qpos-v1",
        action_space=BIMANUAL_QPOS_V1,
        components=(ActionComponentSpec("left_arm", "left_arm", "left_arm"),),
    )
    spec = CompositionSpec(
        spec_id="slow-base-fast-residual-v1",
        algebra="base_plus_bounded_residual",
        input_roles=("base", "residual"),
        required_roles=("base", "residual"),
        residual_limit_by_component={"left_arm": 0.05},
    )
    slow: PolicyNodeRuntime = FakeSlowContextNode()
    fast: PolicyNodeRuntime = FakeFastActionNode()
    composer = DeterministicComposer()
    rgbd_ids_seen: list[str] = []
    tactile_ids_seen: list[str] = []
    context_ids_used: list[str] = []
    proposal_sequence = 0
    base_sequence = 0

    for fast_index in range(8):
        at = 10.0 + fast_index / 40.0
        clock.value = at
        rgbd_sequence = math.floor((at - 10.0) * 15.0 + 1e-9)
        fast_observation = observation_bundle(
            coordinator=coordinator,
            policy_node_id="fast-tactile",
            sequence=fast_index,
            at=at,
            rgbd_sequence=rgbd_sequence,
            tactile_sequence=math.floor((at - 10.0) * 100.0 + 1e-9),
        )
        rgbd_ids_seen.append(
            fast_observation.record.selected_sample_by_stream["rgbd"].sample_id
        )
        tactile_ids_seen.append(
            fast_observation.record.selected_sample_by_stream["tactile"].sample_id
        )

        if fast_index % 4 == 0:
            slow_observation = observation_bundle(
                coordinator=coordinator,
                policy_node_id="slow-vision",
                sequence=fast_index // 4,
                at=at,
                rgbd_sequence=rgbd_sequence,
                tactile_sequence=math.floor((at - 10.0) * 100.0 + 1e-9),
            )
            slow_output = slow.run(
                PolicyNodeRequest(
                    request_id=f"slow-request:{fast_index}",
                    schedule_tick_id=f"slow-tick:{fast_index}",
                    trigger_reason="fixed_10hz",
                    scheduled_at=at,
                    target_effective_at=at,
                    observation=slow_observation,
                )
            )
            context_receipt = contexts.admit(slow_output.policy_context)
            assert context_receipt.context is not None
            context = context_receipt.context
            composition_inputs.admit(
                CompositionInputDraft(
                    runtime_session_id="session-0",
                    control_domain_id="robot-0",
                    producer_id="slow-base",
                    producer_epoch=1,
                    sequence=base_sequence,
                    input_role="base",
                    action=RobotAction(
                        action_space=BIMANUAL_QPOS_V1,
                        arm_qpos_by_side={
                            "left": context.semantic_payload["base_qpos"]
                        },
                        source="fake_slow_base",
                    ),
                    produced_at=at,
                    produced_clock_domain_id="control.monotonic",
                    effective_from=at,
                    effective_until=at + 0.2,
                    effective_clock_domain_id="control.monotonic",
                    context_ids=(context.context_id,),
                    observation_cut_id=context.observation_cut_id,
                    policy_output_id=context.policy_output_id,
                )
            )
            base_sequence += 1

        cut = coordinator.capture_cut(cutoff_at=at).cut
        context_snapshot = build_policy_context_snapshot(
            project_policy_contexts(contexts, cut),
            producer_epochs=epochs.snapshot(cut),
            target_effective_at=at,
            required_channels=("motion-plan",),
        )
        fast_output = fast.run(
            PolicyNodeRequest(
                request_id=f"fast-request:{fast_index}",
                schedule_tick_id=f"fast-tick:{fast_index}",
                trigger_reason="fixed_40hz",
                scheduled_at=at,
                target_effective_at=at,
                observation=fast_observation,
                context_snapshot=context_snapshot,
            )
        )
        assert fast_output.composition_input is not None
        residual_receipt = composition_inputs.admit(fast_output.composition_input)
        assert residual_receipt.accepted
        composition_cut = coordinator.capture_cut(cutoff_at=at).cut
        composition_snapshot = build_composition_input_snapshot(
            project_composition_inputs(composition_inputs, composition_cut),
            producer_epochs=epochs.snapshot(composition_cut),
            spec=spec,
            target_effective_at=at,
        )
        result = composer.compose(composition_snapshot, spec=spec)
        assert result.output_status == "ready"
        context_ids_used.extend(result.context_ids)
        proposal_receipt = proposals.admit(
            build_composite_action_proposal_draft(
                result,
                runtime_session_id="session-0",
                control_domain_id="robot-0",
                embodiment_instance_id="rm65-pair-0",
                producer_id="composer",
                producer_epoch=1,
                sequence=proposal_sequence,
                produced_at=at,
                produced_clock_domain_id="control.monotonic",
                effective_from=at,
                effective_until=at + 0.05,
                effective_clock_domain_id="control.monotonic",
                component_schema_hash=catalog.semantic_content_hash,
                component_coverage=("left_arm",),
                selection_groups=("left_arm",),
                policy_node_id="fast-composer",
            )
        )
        assert proposal_receipt.accepted
        proposal_sequence += 1

    assert len(set(rgbd_ids_seen)) == 3
    assert len(rgbd_ids_seen) == 8
    assert tactile_ids_seen == (
        ["tactile:0", "tactile:2", "tactile:5", "tactile:7"]
        + ["tactile:10", "tactile:12", "tactile:15", "tactile:17"]
    )
    assert len(set(context_ids_used)) == 2
    assert len(proposals) == 8
    final_cut = coordinator.capture_cut(cutoff_at=clock.value).cut
    assert all(
        not record.authority_candidate
        for record in composition_inputs.records_at_or_before(final_cut)
    )

    authority = AuthorityStore(
        runtime_session_id="session-0",
        coordinator=coordinator,
        policy=AuthorityPolicySpec(
            policy_id="composer-authority-v1",
            priority_by_group={"left_arm": {"kind:composer": 10}},
            fallback_by_group={"left_arm": "no_action"},
        ),
    )
    gate = ExecutionGateStore(
        runtime_session_id="session-0", coordinator=coordinator
    )
    clock.value = 10.176
    for decision_hz in (30.0, 45.0, 60.0):
        scheduler = DecisionScheduler(
            DecisionScheduleSpec(
                spec_id=f"synthetic-{decision_hz:g}hz",
                decision_hz=decision_hz,
            ),
            clock=clock,
        )
        snapshot = capture_arbitration_input(
            coordinator=coordinator,
            proposal_buffer=proposals,
            producer_epochs=epochs,
            authority_store=authority,
            execution_gate_store=gate,
            scheduler=scheduler,
            embodiment_instance_id="rm65-pair-0",
            component_catalog=catalog,
            active_selection_groups=("left_arm",),
        )
        assert snapshot is not None
        result = ActionArbiter().arbitrate(snapshot)
        assert result.output_status == "selected"
        assert result.composed_action is not None
        assert result.attribution["left_arm"]["context_ids"]


def test_policy_node_protocol_rejects_mismatched_output_kind() -> None:
    with pytest.raises(ValueError, match="kind and typed payload"):
        PolicyNodeOutput(
            node_id="bad-node",
            output_kind="policy_context",
            produced_at=10.0,
            no_output_reason=None,
        )
