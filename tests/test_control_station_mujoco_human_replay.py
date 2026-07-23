from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from control_station import (
    HumanTakeoverOperationSpec,
    HumanTakeoverReadinessCandidate,
    SideEffectStationOperationCoordinator,
    StationIntentDraft,
)
from control_station.composition import (
    CS6A_HUMAN_REPLAY_TRACE_PROFILE_ID_V1,
    build_cs6a_human_replay_execution_manifest,
    build_scoped_cs6a_mujoco_human_replay_handoff,
    build_scoped_cs6b1_mujoco_human_replay_station,
)
from control_station.composition.mujoco_replay import ReplaySourceObserverBinding
from control_station.handoff import MixedControlBoundaryPolicy
from control_station.sources import (
    SyntheticHumanQposSource,
    SyntheticHumanQposSourceSpec,
)
from dataset_contracts.completeness import trace_requirement_profile
from dataset_contracts.control_trace_episode import (
    CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1,
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
    load_episode_control_trace,
    materialize_actual_decision_view,
)
from robot_motion.action import (
    RobotAction,
    SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
    SharedCanonicalSafetyResult,
)
from robot_motion.action.authoritative_trace import (
    project_control_execution_manifest,
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
from sim.mujoco.deploy import MujocoDeployConfig, MujocoDeployRuntime
from teleop_runtime.control_trace import (
    ControlTraceRuntime,
    ControlTraceSpool,
    ControlTraceSpoolSink,
    ControlTraceWriterHealthLedger,
    EpisodeCaptureManifestStore,
    EpisodeTraceRecorderSink,
    RuntimeTraceSession,
)


class ManualClock:
    def __init__(self, value: float = 20.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class DeterministicSceneFacade:
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
            side: np.asarray(value, dtype=float).reshape(6).copy()
            for side, value in qpos_by_side.items()
        }
        self.physics_commands.append(command)
        self.qpos = {side: value.copy() for side, value in command.items()}


def _timing() -> TimingSpec:
    return TimingSpec(
        sample_hz=15.0,
        policy_hz=15.0,
        reference_hz=60.0,
        ik_hz=60.0,
        command_hz=120.0,
        physics_hz=1000.0,
    )


def _runtime(
    clock: ManualClock,
    *,
    trace_sink=None,
    takeover_allowed: bool = True,
) -> tuple[MujocoDeployRuntime, DeterministicSceneFacade]:
    scene = DeterministicSceneFacade()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
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
            runtime_session_id="cs6a-runtime",
            control_domain_id="cs6a-domain",
            embodiment_instance_id="cs6a-embodiment",
            producer_id="human-main",
            replay_producer_id="replay-main",
            enable_replay_authority=True,
            decision_hz=30.0,
            authority_policy_id="cs6a-human-replay-lease-v1",
            initial_control_mode="hybrid",
            reference_activation_profile_id=(
                SCHEDULED_REFERENCE_SEGMENT_SIM_V1
            ),
            clock_domain_id="local_monotonic",
            authority_takeover_allowed_producers=(
                ("human-main", "replay-main") if takeover_allowed else ()
            ),
        ),
        trace_sink=trace_sink,
        clock=clock,
    )
    return runtime, scene


def _replay_source(runtime: MujocoDeployRuntime) -> ReplayProposalSource:
    action_contract = ReplaySourceActionContract(
        contract_id="cs6a-replay-action:qpos",
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
        contract_id="cs6a-replay-source:qpos",
        replay_mode=REPLAY_MODE_CANONICAL_ACTION,
        dataset_schema="dexteleop_dataset_v2.4",
        source_dataset_id="cs6a-golden-source",
        episode_id="episode-replay-0",
        action_stage=REPLAY_ACTION_STAGE_SELECTED,
        action_field="decision.selected_action",
        source_action_space="bimanual_qpos_v1",
        source_frame_contract_hash=None,
        source_component_schema_hash=runtime.catalog.semantic_content_hash,
    )
    binding, assessment = build_replay_binding_assessment(
        action_contract,
        runtime.target_runtime_action_contract,
        binding_id="cs6a-qpos-replay-to-mujoco",
    )
    assert assessment.verdict == REPLAY_BINDING_VERIFIED_COMPATIBLE
    return ReplayProposalSource(
        ReplayProposalSourceConfig(
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            embodiment_instance_id=runtime.config.embodiment_instance_id,
            producer_id=runtime.config.replay_producer_id,
            producer_epoch=runtime.config.producer_epoch,
            component_schema_hash=runtime.catalog.semantic_content_hash,
            component_coverage=("left_arm", "right_arm"),
            selection_groups=("arms",),
            source_contract=source_contract,
            source_action_contract=action_contract,
            target_runtime_contract=runtime.target_runtime_action_contract,
            derivation_receipt=None,
            schedule=ReplayScheduleSpec(
                schedule_id="cs6a-replay-30hz-v1",
                schedule_mode=REPLAY_SCHEDULE_FIXED_RATE,
                source_hz=30.0,
                target_hz=None,
                resampling_policy="none",
                source_start_index=10,
            ),
            target_binding_spec=binding,
            binding_assessment=assessment,
            requested_use=REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
        )
    )


def _observer_binding(path: Path) -> ReplaySourceObserverBinding:
    return ReplaySourceObserverBinding.create(
        source_slot_id="replay-slot",
        source_locator_id="cs6a-golden-source",
        expected_source_path=str(path),
        source_episode_index=0,
        group_key="replay-group-0",
        source_start_index=10,
        clock_domain_id="local_monotonic",
    )


def _composition(tmp_path: Path, clock: ManualClock, *, trace_sink=None):
    runtime, scene = _runtime(clock, trace_sink=trace_sink)
    human_spec = SyntheticHumanQposSourceSpec(source_id="human-slot")
    human_source = SyntheticHumanQposSource(human_spec)
    replay_source = _replay_source(runtime)
    path = tmp_path / "episode.parquet"
    composition = build_scoped_cs6a_mujoco_human_replay_handoff(
        runtime,
        human_spec,
        replay_source,
        _observer_binding(path),
    )
    return composition, human_source, path, scene


def _station_composition(tmp_path: Path, clock: ManualClock):
    runtime, scene = _runtime(clock)
    human_spec = SyntheticHumanQposSourceSpec(source_id="human-slot")
    human_source = SyntheticHumanQposSource(human_spec)
    replay_source = _replay_source(runtime)
    path = tmp_path / "episode.parquet"
    station = build_scoped_cs6b1_mujoco_human_replay_station(
        runtime,
        human_spec,
        replay_source,
        _observer_binding(path),
        station_clock=clock,
        station_clock_domain_id="local_monotonic",
    )
    return station, human_source, path, scene


def _station_submit(
    station,
    clock: ManualClock,
    *,
    sequence: int,
    kind: str,
    subject_id: str | None = None,
    resource_binding_id: str | None = None,
    operation_spec_id: str | None = None,
):
    state = station.station.state
    transition = station.station.submit_intent(
        StationIntentDraft(
            client_id="operator:cs6b1",
            client_sequence=sequence,
            intent_kind=kind,
            produced_at=clock(),
            produced_clock_domain_id="local-monotonic",
            expected_state_revision=state.state_revision,
            manifest_hash=station.station.manifest.semantic_content_hash,
            subject_id=subject_id,
            resource_binding_id=resource_binding_id,
            operation_spec_id=operation_spec_id,
            reason="cs6b1-test",
        )
    )
    execution = None
    if transition.receipt.status == "operation_requested":
        execution = station.complete_pending_operation(transition)
    return transition, execution


def _advance_station_to_replay_with_human_candidate(station, clock: ManualClock) -> None:
    target = station.station.manifest.execution_binding.binding_id
    operations = (
        ("acquire_control_lease", None),
        ("connect_target", target),
        ("connect_source", "replay-slot"),
        ("connect_source", "human-slot"),
        ("run_preflight", target),
        ("run_preflight", "replay-slot"),
        ("run_preflight", "human-slot"),
        ("request_candidate", "replay-slot"),
        ("request_motion_authority", "replay-slot"),
        ("request_candidate", "human-slot"),
    )
    for sequence, (kind, subject) in enumerate(operations):
        transition, execution = _station_submit(
            station,
            clock,
            sequence=sequence,
            kind=kind,
            subject_id=subject,
        )
        assert transition.receipt.accepted
        if transition.receipt.status == "operation_requested":
            assert execution is not None
            assert execution.completion_transition.accepted
            assert execution.completion_transition.result.succeeded
    assert station.station.state.source_state_by_slot == {
        "human-slot": "candidate",
        "replay-slot": "active",
    }


def _takeover_spec(station, *, observer_binding_id: str):
    manifest = station.station.manifest
    return HumanTakeoverOperationSpec(
        previous_source_slot_id="replay-slot",
        target_human_source_slot_id="human-slot",
        previous_source_action_contract_hash=(
            manifest.source_slot("replay-slot").source_action_contract_hash
        ),
        target_human_action_contract_hash=(
            manifest.source_slot("human-slot").source_action_contract_hash
        ),
        execution_binding_hash=manifest.execution_binding.semantic_content_hash,
        readiness_observer_binding_id=observer_binding_id,
        boundary_policy=_safe_mixed_policy(),
        readiness_timeout_sec=0.2,
        capture_boundary_id="episode-boundary:cs6b1",
    )


class _SyntheticReadinessProvider:
    observer_binding_id = "human-observer:synthetic-cs6b1"

    def __init__(self, source: SyntheticHumanQposSource, clock: ManualClock) -> None:
        self.source = source
        self.clock = clock
        self.call_count = 0

    def acquire(self, *, operation_spec, takeover_state, timeout_sec):
        del operation_spec, timeout_sec
        self.call_count += 1
        assert takeover_state.phase == "awaiting_human_readiness"
        assert takeover_state.takeover_receipt is not None
        self.clock.value = max(
            self.clock.value,
            takeover_state.takeover_receipt.completed_at,
        ) + 0.01
        emission = _human_emission(self.source, self.clock)
        return HumanTakeoverReadinessCandidate(
            observer_binding_id=self.observer_binding_id,
            emission=emission,
            effective_from=self.clock() + 0.03,
            effective_until=self.clock() + 0.13,
            target_effective_at=self.clock() + 0.03,
            deadman_rearm_event_id="virtual-deadman:cs6b1",
            deadman_rearmed_at=self.clock(),
            retargeter_rebase_receipt_id="virtual-rebase:cs6b1",
            retargeter_rebased_at=self.clock(),
        )


class _BlockingSyntheticReadinessProvider(_SyntheticReadinessProvider):
    def __init__(self, source, clock) -> None:
        super().__init__(source, clock)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.worker_thread_id: int | None = None

    def acquire(self, **kwargs):
        self.worker_thread_id = threading.get_ident()
        self.entered.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test provider was not released")
        return super().acquire(**kwargs)


class _LateSyntheticReadinessProvider(_SyntheticReadinessProvider):
    def acquire(self, **kwargs):
        candidate = super().acquire(**kwargs)
        self.clock.value += 0.21
        return candidate


class _StaleSyntheticReadinessProvider(_SyntheticReadinessProvider):
    def acquire(self, *, operation_spec, takeover_state, timeout_sec):
        del operation_spec, timeout_sec
        self.call_count += 1
        assert takeover_state.takeover_receipt is not None
        self.clock.value = takeover_state.takeover_receipt.completed_at
        emission = _human_emission(self.source, self.clock)
        return HumanTakeoverReadinessCandidate(
            observer_binding_id=self.observer_binding_id,
            emission=emission,
            effective_from=self.clock() + 0.03,
            effective_until=self.clock() + 0.13,
            target_effective_at=self.clock() + 0.03,
            deadman_rearm_event_id="virtual-deadman:stale",
            deadman_rearmed_at=self.clock(),
            retargeter_rebase_receipt_id="virtual-rebase:stale",
            retargeter_rebased_at=self.clock(),
        )


def _human_emission(source: SyntheticHumanQposSource, clock: ManualClock):
    return source.emit(
        {
            "left_arm": np.full(6, 0.001, dtype=float),
            "right_arm": np.full(6, -0.001, dtype=float),
        },
        emitted_at=clock(),
    )


def _replay_emission(path: Path, clock: ManualClock):
    return create_replay_source_frame_emission(
        source_path=str(path),
        source_episode_index=0,
        group_key="replay-group-0",
        local_frame_index=0,
        source_action_index=10,
        source_relative_time=0.0,
        action=RobotAction.bimanual_qpos(
            left_arm=np.full(6, 0.002, dtype=float),
            right_arm=np.full(6, -0.002, dtype=float),
            timestamp=clock(),
            source="existing-replay-cursor",
        ),
        observed_at=clock(),
        target_effective_at=clock(),
        effective_until=clock() + 0.1,
        clock_domain_id="local_monotonic",
    )


def _submit_human(composition, emission, clock: ManualClock):
    return composition.submit_human_emission(
        emission,
        effective_from=clock(),
        effective_until=clock() + 0.1,
        target_effective_at=clock(),
    )


def _safe_mixed_policy() -> MixedControlBoundaryPolicy:
    return MixedControlBoundaryPolicy(
        policy_id="cs6b0-safe-mixed-sim-v1",
        policy_suspension_mode="hold_and_release_authority",
        episode_boundary_motion_policy="hold_before_unbind",
    )


def test_cs6b1_station_takeover_commits_once_then_promotes_exact_emission(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    station, human_source, path, _scene = _station_composition(tmp_path, clock)
    _advance_station_to_replay_with_human_candidate(station, clock)
    assert station.publication_barrier_by_source["replay-slot"].snapshot().enabled
    assert not station.publication_barrier_by_source["human-slot"].snapshot().enabled

    replay = station.submit_replay_source_frame_emission(
        _replay_emission(path, clock)
    )
    assert replay.proposal_admission is not None
    assert replay.proposal_admission.accepted
    assert station.step(now=clock(), physics_ticks_after_dispatch=2).status == "dispatched"

    provider = _SyntheticReadinessProvider(human_source, clock)
    spec = _takeover_spec(station, observer_binding_id=provider.observer_binding_id)
    spec_id = station.register_human_takeover_operation(spec, provider)
    clock.value = 20.2
    request, execution = _station_submit(
        station,
        clock,
        sequence=10,
        kind="request_human_takeover",
        subject_id="human-slot",
        resource_binding_id="replay-slot",
        operation_spec_id=spec_id,
    )
    assert request.receipt.status == "operation_requested"
    assert execution is not None
    completion = execution.completion_transition
    assert completion.accepted
    assert completion.result.succeeded
    assert completion.result.status == "human_takeover_ready"
    assert provider.call_count == 1
    assert station.station.state.source_state_by_slot == {
        "human-slot": "active",
        "replay-slot": "preflighted",
    }
    assert not station.publication_barrier_by_source["replay-slot"].snapshot().enabled
    assert station.publication_barrier_by_source["human-slot"].snapshot().enabled
    evidence = station.takeover_evidence_by_intent_id[request.intent.intent_id]
    assert evidence.operation_spec_id == spec_id
    assert evidence.succeeded
    assert evidence.final_phase == "ready"

    human_facts = tuple(
        fact
        for fact in station.data_plane.runtime.trace_facts
        if fact.event_type == "human_command_emission"
    )
    assert human_facts == ()
    promoted = station.promote_human_takeover_emission(spec_id)
    assert promoted.accepted
    with pytest.raises(RuntimeError, match="No READY"):
        station.promote_human_takeover_emission(spec_id)
    human_facts = tuple(
        fact
        for fact in station.data_plane.runtime.trace_facts
        if fact.event_type == "human_command_emission"
    )
    assert len(human_facts) == 1
    with pytest.raises(RuntimeError, match="Station replay authority"):
        station.submit_replay_source_frame_emission(_replay_emission(path, clock))
    station.close()


def test_cs6b1_readiness_wait_runs_off_caller_and_keeps_gate_held(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    station, human_source, _path, _scene = _station_composition(tmp_path, clock)
    _advance_station_to_replay_with_human_candidate(station, clock)
    clock.value = 20.2
    provider = _BlockingSyntheticReadinessProvider(human_source, clock)
    spec = _takeover_spec(station, observer_binding_id=provider.observer_binding_id)
    spec_id = station.register_human_takeover_operation(spec, provider)
    coordinator = SideEffectStationOperationCoordinator(
        coordinator_id="coordinator:cs6b1-mixed",
        runtime=station.station,
        adapters=station.adapters,
        operation_executor=station.complete_pending_operation,
        operation_prepare=station.prepare_pending_operation,
        queue_capacity=2,
        clock=clock,
        clock_domain_id="local-monotonic",
    )
    coordinator.start()
    try:
        caller_thread_id = threading.get_ident()
        state = station.station.state
        submission = coordinator.submit_intent(
            StationIntentDraft(
                client_id="operator:cs6b1",
                client_sequence=10,
                intent_kind="request_human_takeover",
                produced_at=clock(),
                produced_clock_domain_id="local-monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id="human-slot",
                resource_binding_id="replay-slot",
                operation_spec_id=spec_id,
                reason="cs6b1-worker-test",
            )
        )
        assert submission.enqueue_receipt.accepted
        assert provider.entered.wait(timeout=1.0)
        assert provider.worker_thread_id != caller_thread_id
        assert not station.publication_barrier_by_source[
            "replay-slot"
        ].snapshot().enabled
        assert station.data_plane.runtime.core.capture_execution_gate_snapshot().status == (
            "hold"
        )
        # The readiness provider is blocked on the application worker, while
        # the existing control step remains callable and observes lower HOLD.
        assert (
            station.step(now=clock(), physics_ticks_after_dispatch=0).status
            == "execution_hold_active"
        )
        provider.release.set()
        assert coordinator.wait_until_idle(timeout_sec=2.0)
        assert station.station.state.source_state_by_slot["human-slot"] == "active"
        assert station.publication_barrier_by_source["human-slot"].snapshot().enabled
    finally:
        provider.release.set()
        assert coordinator.stop(timeout_sec=2.0)
        station.close()


@pytest.mark.parametrize(
    "provider_type",
    (_LateSyntheticReadinessProvider, _StaleSyntheticReadinessProvider),
)
def test_cs6b1_invalid_readiness_stays_hold_and_emits_no_human_fact(
    tmp_path: Path,
    provider_type,
) -> None:
    clock = ManualClock()
    station, human_source, _path, _scene = _station_composition(tmp_path, clock)
    _advance_station_to_replay_with_human_candidate(station, clock)
    clock.value = 20.2
    provider = provider_type(human_source, clock)
    spec = _takeover_spec(station, observer_binding_id=provider.observer_binding_id)
    spec_id = station.register_human_takeover_operation(spec, provider)
    request, execution = _station_submit(
        station,
        clock,
        sequence=10,
        kind="request_human_takeover",
        subject_id="human-slot",
        resource_binding_id="replay-slot",
        operation_spec_id=spec_id,
    )
    assert execution is not None
    completion = execution.completion_transition
    assert completion.accepted
    assert not completion.result.succeeded
    assert completion.result.status == "human_takeover_held"
    assert completion.state_after.lifecycle_state == "hold"
    assert not any(
        barrier.snapshot().enabled
        for barrier in station.publication_barrier_by_source.values()
    )
    assert not any(
        fact.event_type == "human_command_emission"
        for fact in station.data_plane.runtime.trace_facts
    )
    assert not station.takeover_evidence_by_intent_id[
        request.intent.intent_id
    ].succeeded
    station.close()


@pytest.mark.parametrize(
    ("safety_kind", "station_lifecycle", "lower_gate"),
    (("hold", "hold", "hold"), ("estop", "fault", "fault")),
)
def test_cs6b1_safety_supersedes_waiting_worker_before_lower_ready(
    tmp_path: Path,
    safety_kind: str,
    station_lifecycle: str,
    lower_gate: str,
) -> None:
    clock = ManualClock()
    station, human_source, _path, _scene = _station_composition(tmp_path, clock)
    _advance_station_to_replay_with_human_candidate(station, clock)
    clock.value = 20.2
    provider = _BlockingSyntheticReadinessProvider(human_source, clock)
    spec = _takeover_spec(station, observer_binding_id=provider.observer_binding_id)
    spec_id = station.register_human_takeover_operation(spec, provider)
    coordinator = SideEffectStationOperationCoordinator(
        coordinator_id="coordinator:cs6b1-supersede",
        runtime=station.station,
        adapters=station.adapters,
        operation_executor=station.complete_pending_operation,
        operation_prepare=station.prepare_pending_operation,
        queue_capacity=2,
        clock=clock,
        clock_domain_id="local-monotonic",
    )
    coordinator.start()
    try:
        state = station.station.state
        takeover = coordinator.submit_intent(
            StationIntentDraft(
                client_id="operator:cs6b1",
                client_sequence=10,
                intent_kind="request_human_takeover",
                produced_at=clock(),
                produced_clock_domain_id="local-monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id="human-slot",
                resource_binding_id="replay-slot",
                operation_spec_id=spec_id,
                reason="cs6b1-superseded-test",
            )
        )
        assert takeover.enqueue_receipt.accepted
        assert provider.entered.wait(timeout=1.0)

        state = station.station.state
        safety = coordinator.submit_intent(
            StationIntentDraft(
                client_id="safety:cs6b1",
                client_sequence=0,
                intent_kind=safety_kind,
                produced_at=clock(),
                produced_clock_domain_id="local-monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                reason=f"operator-{safety_kind}-during-readiness",
            )
        )
        assert safety.transition.receipt.accepted
        assert station.station.state.lifecycle_state == station_lifecycle
        assert station.station.state.pending_operation_by_subject == {}

        provider.release.set()
        assert coordinator.wait_until_idle(timeout_sec=2.0)
        completion = coordinator.completions[-1].completion_transition
        assert not completion.accepted
        assert completion.status == "human_takeover_result_after_latched_safety_state"
        assert station.data_plane.safe_human_takeover_state.phase == "failed"
        assert station.data_plane.safe_human_takeover_state.failure_reason == (
            "station_pending_operation_superseded"
        )
        assert (
            station.data_plane.runtime.core.capture_execution_gate_snapshot().status
            == lower_gate
        )
        assert not any(
            barrier.snapshot().enabled
            for barrier in station.publication_barrier_by_source.values()
        )
        assert not any(
            fact.event_type == "human_command_emission"
            for fact in station.data_plane.runtime.trace_facts
        )
    finally:
        provider.release.set()
        assert coordinator.stop(timeout_sec=2.0)
        station.close()


def test_cs6a_discrete_handoff_dispatches_each_source_once(tmp_path: Path) -> None:
    clock = ManualClock()
    composition, human_source, path, scene = _composition(tmp_path, clock)
    assert composition.binding_receipt.discrete_handoff_only
    assert not composition.binding_receipt.source_reactivation_supported
    assert "source_reactivation_within_session" in (
        composition.manifest.explicit_exclusions
    )
    composition.start()

    inactive_replay = _replay_emission(path, clock)
    fact_count = len(composition.runtime.trace_facts)
    with pytest.raises(RuntimeError, match="Replay publication is not"):
        composition.submit_replay_source_frame_emission(inactive_replay)
    assert len(composition.runtime.trace_facts) == fact_count

    human_handoff = composition.handoff_to("human-slot", reason="operator-human")
    assert human_handoff.previous_source_slot_id is None
    human_submission = _submit_human(
        composition,
        _human_emission(human_source, clock),
        clock,
    )
    assert human_submission.accepted
    human_step = composition.step(now=clock(), physics_ticks_after_dispatch=2)
    assert human_step.status == "dispatched"
    assert human_step.decision is not None
    assert human_step.computation is not None
    assert set(human_step.computation.owner_by_component.values()) == {"human-main"}

    clock.value = 20.2
    replay_handoff = composition.handoff_to("replay-slot", reason="operator-replay")
    assert replay_handoff.previous_source_slot_id == "human-slot"
    assert replay_handoff.execution_epoch > human_handoff.execution_epoch

    inactive_human = _human_emission(human_source, clock)
    fact_count = len(composition.runtime.trace_facts)
    with pytest.raises(RuntimeError, match="Human publication is not"):
        _submit_human(composition, inactive_human, clock)
    assert len(composition.runtime.trace_facts) == fact_count

    replay_submission = composition.submit_replay_source_frame_emission(
        _replay_emission(path, clock)
    )
    assert replay_submission.source_emission is not inactive_replay
    assert replay_submission.proposal_admission is not None
    assert replay_submission.proposal_admission.accepted
    assert replay_submission.authorization_admission is not None
    assert replay_submission.authorization_admission.accepted
    replay_step = composition.step(now=clock(), physics_ticks_after_dispatch=2)
    assert replay_step.status == "dispatched"
    assert replay_step.decision is not None
    assert replay_step.computation is not None
    assert set(replay_step.computation.owner_by_component.values()) == {"replay-main"}
    assert len(scene.physics_commands) >= 2

    with pytest.raises(RuntimeError, match="forbids source reactivation"):
        composition.handoff_to("human-slot", reason="unsupported-return")

    event_types = tuple(fact.event_type for fact in composition.runtime.trace_facts)
    assert event_types.count("human_command_emission") == 1
    assert event_types.count("replay_source_frame_emission") == 1
    assert event_types.count("arbitration_decision") == 2
    authority_events = tuple(
        fact.payload["authority_event_type"]
        for fact in composition.runtime.trace_facts
        if fact.event_type == "authority_event"
    )
    assert authority_events == ("takeover", "release", "takeover")
    assert event_types.count("execution_cancellation") == 2
    composition.close()


def test_cs6a_reverse_replay_to_human_handoff_is_also_discrete(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    composition, human_source, path, _scene = _composition(tmp_path, clock)
    composition.start()

    composition.handoff_to("replay-slot", reason="operator-replay-first")
    replay_submission = composition.submit_replay_source_frame_emission(
        _replay_emission(path, clock)
    )
    assert replay_submission.proposal_admission is not None
    assert replay_submission.proposal_admission.accepted
    replay_step = composition.step(now=clock(), physics_ticks_after_dispatch=2)
    assert replay_step.status == "dispatched"
    assert replay_step.computation is not None
    assert set(replay_step.computation.owner_by_component.values()) == {
        "replay-main"
    }

    clock.value = 20.2
    handoff = composition.handoff_to("human-slot", reason="operator-human-second")
    assert handoff.previous_source_slot_id == "replay-slot"
    human_submission = _submit_human(
        composition,
        _human_emission(human_source, clock),
        clock,
    )
    assert human_submission.accepted
    human_step = composition.step(now=clock(), physics_ticks_after_dispatch=2)
    assert human_step.status == "dispatched"
    assert human_step.computation is not None
    assert set(human_step.computation.owner_by_component.values()) == {"human-main"}

    authority_events = tuple(
        fact.payload["authority_event_type"]
        for fact in composition.runtime.trace_facts
        if fact.event_type == "authority_event"
    )
    assert authority_events == ("takeover", "release", "takeover")
    composition.close()


def test_cs6b0_safe_replay_to_human_stays_hold_until_exact_fresh_promotion(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    composition, human_source, path, scene = _composition(tmp_path, clock)
    composition.start()
    composition.handoff_to("replay-slot", reason="operator-replay-first")
    replay_submission = composition.submit_replay_source_frame_emission(
        _replay_emission(path, clock)
    )
    assert replay_submission.proposal_admission is not None
    assert replay_submission.proposal_admission.accepted
    replay_step = composition.step(now=clock(), physics_ticks_after_dispatch=2)
    assert replay_step.status == "dispatched"
    physics_count_before_takeover = len(scene.physics_commands)

    clock.value = 20.2
    state = composition.begin_safe_replay_to_human_takeover(
        _safe_mixed_policy(),
        reason="explicit-human-takeover",
        capture_boundary_id="episode-boundary:0",
    )
    assert state.phase == "awaiting_human_readiness"
    assert state.execution_gate_status == "hold"
    assert composition.active_source_slot_id is None
    assert not state.human_publication_enabled
    assert state.execution_epoch > replay_step.dispatch_context.execution_epoch
    assert composition.runtime.core.producer_epochs.current_epoch(
        "replay-main"
    ) == (composition.runtime.config.producer_epoch + 1)
    assert composition.runtime.rig.snapshot()["last_reference_sequence"] is None

    with pytest.raises(RuntimeError, match="Replay publication is not"):
        composition.submit_replay_source_frame_emission(
            _replay_emission(path, clock)
        )
    too_early_emission = _human_emission(human_source, clock)
    with pytest.raises(RuntimeError, match="Human publication is not"):
        _submit_human(
            composition,
            too_early_emission,
            clock,
        )
    with pytest.raises(ValueError, match="emitted after authority takeover"):
        composition.complete_safe_human_takeover(
            too_early_emission,
            effective_from=clock(),
            effective_until=clock() + 0.1,
            target_effective_at=clock(),
            deadman_rearm_event_id="virtual-deadman:too-early",
            deadman_rearmed_at=clock(),
            retargeter_rebase_receipt_id="virtual-rebase:too-early",
            retargeter_rebased_at=clock(),
        )
    assert len(scene.physics_commands) == physics_count_before_takeover
    assert not any(
        fact.event_type == "human_command_emission"
        for fact in composition.runtime.trace_facts
    )

    clock.value = 20.21
    promoted_emission = _human_emission(human_source, clock)
    state = composition.complete_safe_human_takeover(
        promoted_emission,
        effective_from=20.24,
        effective_until=20.34,
        target_effective_at=20.24,
        deadman_rearm_event_id="virtual-deadman:release-press:0",
        deadman_rearmed_at=clock(),
        retargeter_rebase_receipt_id="virtual-retargeter-rebase:0",
        retargeter_rebased_at=clock(),
    )
    assert state.phase == "ready"
    assert state.human_publication_enabled
    assert composition.active_source_slot_id == "human-slot"
    # The safety preview is deliberately non-publishing.
    assert not any(
        fact.event_type == "human_command_emission"
        for fact in composition.runtime.trace_facts
    )

    clock.value = 20.22
    different_emission = _human_emission(human_source, clock)
    with pytest.raises(RuntimeError, match="exactly promote"):
        composition.submit_human_emission(
            different_emission,
            effective_from=20.24,
            effective_until=20.34,
            target_effective_at=20.24,
        )
    promoted_submission = composition.submit_human_emission(
        promoted_emission,
        effective_from=20.24,
        effective_until=20.34,
        target_effective_at=20.24,
    )
    assert promoted_submission.accepted
    human_emission_id = (
        promoted_submission.adapter_result.source_adapter_receipt.source_emission_id
    )

    clock.value = 20.24
    human_step = composition.step(now=clock(), physics_ticks_after_dispatch=2)
    assert human_step.status == "dispatched"
    assert human_step.computation is not None
    assert set(human_step.computation.owner_by_component.values()) == {"human-main"}

    facts = composition.runtime.trace_facts
    formal_emissions = tuple(
        fact
        for fact in facts
        if fact.event_type == "human_command_emission"
        and fact.fact_id == human_emission_id
    )
    assert len(formal_emissions) == 1
    human_admission = next(
        fact
        for fact in facts
        if fact.event_type == "proposal_admission"
        and fact.payload["emission_id"] == human_emission_id
    )
    human_takeover = tuple(
        fact
        for fact in facts
        if fact.event_type == "authority_event"
        and fact.payload["owner_producer_id"] == "human-main"
    )[-1]
    assert facts.index(human_takeover) < facts.index(formal_emissions[0])
    assert facts.index(formal_emissions[0]) < facts.index(human_admission)
    composition.close()


def test_cs6b0_requires_explicit_mixed_policy_before_any_lower_mutation(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    composition, _human_source, _path, _scene = _composition(tmp_path, clock)
    composition.start()
    legacy_handoff = composition.handoff_to(
        "replay-slot",
        reason="legacy-replay-remains-compatible",
    )
    fact_count = len(composition.runtime.trace_facts)
    legacy_policy = MixedControlBoundaryPolicy(
        policy_id="legacy-capture-only-v1",
        policy_suspension_mode="hold_and_release_authority",
        episode_boundary_motion_policy="capture_only",
    )
    with pytest.raises(ValueError, match="hold_before_unbind"):
        composition.begin_safe_replay_to_human_takeover(
            legacy_policy,
            reason="must-not-mutate",
        )
    assert composition.active_source_slot_id == "replay-slot"
    assert composition.safe_human_takeover_state is None
    assert len(composition.runtime.trace_facts) == fact_count
    assert legacy_handoff.ready_gate_event_id
    composition.close()


def test_cs6b0_unsafe_first_human_preview_never_opens_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clock = ManualClock()
    composition, human_source, path, _scene = _composition(tmp_path, clock)
    composition.start()
    composition.handoff_to("replay-slot", reason="operator-replay-first")
    composition.submit_replay_source_frame_emission(_replay_emission(path, clock))
    assert composition.step(now=clock()).status == "dispatched"

    clock.value = 20.2
    composition.begin_safe_replay_to_human_takeover(
        _safe_mixed_policy(),
        reason="explicit-human-takeover",
    )

    def unsafe_preview(_action, input_action_id):
        return SharedCanonicalSafetyResult(
            input_action_id=input_action_id,
            safe_action=None,
            status="hold",
            reasons=("virtual-unsafe-first-command",),
        )

    monkeypatch.setattr(composition.runtime.core, "shared_safety", unsafe_preview)
    clock.value = 20.21
    state = composition.complete_safe_human_takeover(
        _human_emission(human_source, clock),
        effective_from=clock(),
        effective_until=clock() + 0.1,
        target_effective_at=clock(),
        deadman_rearm_event_id="virtual-deadman:release-press:0",
        deadman_rearmed_at=clock(),
        retargeter_rebase_receipt_id="virtual-retargeter-rebase:0",
        retargeter_rebased_at=clock(),
    )
    assert state.phase == "failed"
    assert state.execution_gate_status == "hold"
    assert state.failure_reason == "human_readiness_hold"
    assert composition.active_source_slot_id is None
    assert not state.human_publication_enabled
    assert not any(
        fact.event_type == "human_command_emission"
        for fact in composition.runtime.trace_facts
    )
    gate_statuses = tuple(
        fact.payload["status"]
        for fact in composition.runtime.trace_facts
        if fact.event_type == "execution_gate_event"
    )
    assert gate_statuses[-1] == "hold"
    composition.close()


def test_cs6a_rejects_implicit_priority_only_authority(tmp_path: Path) -> None:
    clock = ManualClock()
    runtime, _scene = _runtime(clock, takeover_allowed=False)
    human_spec = SyntheticHumanQposSourceSpec(source_id="human-slot")
    with pytest.raises(ValueError, match="takeover allowlist"):
        build_scoped_cs6a_mujoco_human_replay_handoff(
            runtime,
            human_spec,
            _replay_source(runtime),
            _observer_binding(tmp_path / "episode.parquet"),
        )
    assert not runtime._started
    assert MujocoDeployConfig().authority_takeover_allowed_producers == ()


def test_cs6a_submission_and_handoff_are_linearized(
    tmp_path: Path,
    monkeypatch,
) -> None:
    clock = ManualClock()
    composition, human_source, _path, _scene = _composition(tmp_path, clock)
    composition.start()
    composition.handoff_to("human-slot", reason="operator-human")
    emission = _human_emission(human_source, clock)
    entered = threading.Event()
    release = threading.Event()
    original_adapt = composition.human_adapter.adapt

    def blocking_adapt(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2.0)
        return original_adapt(*args, **kwargs)

    monkeypatch.setattr(composition.human_adapter, "adapt", blocking_adapt)
    results: dict[str, object] = {}

    def submit() -> None:
        results["submission"] = _submit_human(composition, emission, clock)

    def handoff() -> None:
        results["handoff"] = composition.handoff_to(
            "replay-slot",
            reason="racing-replay",
        )

    submit_thread = threading.Thread(target=submit)
    handoff_thread = threading.Thread(target=handoff)
    submit_thread.start()
    assert entered.wait(timeout=2.0)
    handoff_thread.start()
    assert handoff_thread.is_alive()
    release.set()
    submit_thread.join(timeout=2.0)
    handoff_thread.join(timeout=2.0)
    assert not submit_thread.is_alive()
    assert not handoff_thread.is_alive()
    assert results["submission"].accepted
    assert results["handoff"].previous_source_slot_id == "human-slot"

    facts = composition.runtime.trace_facts
    submission = results["submission"]
    source_emission_id = (
        submission.adapter_result.source_adapter_receipt.source_emission_id
    )
    human_admission = next(
        fact
        for fact in facts
        if fact.event_type == "proposal_admission"
        and fact.payload["emission_id"] == source_emission_id
    )
    replay_takeover = tuple(
        fact
        for fact in facts
        if fact.event_type == "authority_event"
        and fact.payload["owner_producer_id"] == "replay-main"
    )
    assert len(replay_takeover) == 1
    assert human_admission.payload["control_admission_sequence"] < (
        replay_takeover[0].payload["control_admission_sequence"]
    )
    composition.close()


def test_cs6a_formal_capture_seals_and_rebuilds_both_sources(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    profile = trace_requirement_profile(CS6A_HUMAN_REPLAY_TRACE_PROFILE_ID_V1)
    trace_runtime = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="cs6a-trace",
            runtime_session_id="cs6a-runtime",
            control_domain_id="cs6a-domain",
            profile_id=profile.profile_id,
            started_at=clock(),
            clock_domain_id="local_monotonic",
        ),
        profile=profile,
        capacity=8192,
    )
    session_root = tmp_path / CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1 / "cs6a-trace"
    session_root.mkdir(parents=True)
    spool = ControlTraceSpool(
        session_root / "control_trace.wal",
        trace_session_id="cs6a-trace",
        max_bytes=32 << 20,
        physical_preallocation=False,
    )
    health = ControlTraceWriterHealthLedger(
        session_root / "writer_health.ledger",
        trace_session_id="cs6a-trace",
    )
    durable_sink = ControlTraceSpoolSink(
        sink_id="cs6a-spool",
        runtime=trace_runtime,
        spool=spool,
        health_ledger=health,
        clock=clock,
    )
    dependency_sequences: list[int] = []
    recorder = EpisodeTraceRecorderSink(
        trace_runtime,
        lifecycle_owner_id="cs6a-test-owner",
        flush_to_durable=durable_sink.flush_pending,
        dependency_sequences=lambda _runtime: tuple(dependency_sequences),
        manifest_store=EpisodeCaptureManifestStore(tmp_path),
        uuid_factory=lambda: "cs6a-episode",
        control_clock=clock,
        commit_clock=clock,
    )

    def admit_trace(fact) -> None:
        receipt = trace_runtime.issue(fact)
        if not receipt.accepted:
            raise RuntimeError(
                f"CS6a trace rejected {fact.event_type}: {receipt.rejection_code}"
            )

    composition, human_source, path, _scene = _composition(
        tmp_path,
        clock,
        trace_sink=admit_trace,
    )
    execution_manifest = build_cs6a_human_replay_execution_manifest(
        composition.runtime,
        composition.manifest,
    )
    manifest_fact = project_control_execution_manifest(
        composition.runtime.core.trace_context,
        execution_manifest,
        occurred_at=clock(),
    )
    manifest_receipt = trace_runtime.issue(manifest_fact)
    assert manifest_receipt.accepted
    assert manifest_receipt.sink_accepted_sequence is not None
    dependency_sequences.append(manifest_receipt.sink_accepted_sequence)
    prepared = recorder.prepare_episode(legacy_episode_index=0, prepared_at=clock())
    recorder.bind_prepared_episode(prepared, bound_at=clock())
    try:
        composition.start()
        composition.handoff_to("human-slot", reason="capture-human")
        assert _submit_human(
            composition,
            _human_emission(human_source, clock),
            clock,
        ).accepted
        human_step = composition.step(now=clock(), physics_ticks_after_dispatch=60)
        assert human_step.status == "dispatched"

        clock.value = 20.2
        composition.handoff_to("replay-slot", reason="capture-replay")
        assert composition.runtime.rig.snapshot()["last_reference_sequence"] is None
        replay_submission = composition.submit_replay_source_frame_emission(
            _replay_emission(path, clock)
        )
        assert replay_submission.proposal_admission is not None
        assert replay_submission.proposal_admission.accepted
        replay_step = composition.step(now=clock(), physics_ticks_after_dispatch=60)
        assert replay_step.status == "dispatched"

        clock.value = 20.4
        composition.runtime.set_execution_gate(
            "hold",
            reason="cs6a_capture_terminal_hold",
            source_event_ids=("cs6a-stop",),
        )
        hold_step = composition.step(now=clock())
        assert hold_step.status in {"execution_hold_active", "not_due"}
        released = recorder.release_episode(
            prepared,
            unbound_at=clock(),
            reason="cs6a_finalize",
        )
        assert released.seal.trace_sealed, (
            released.seal.missing_required_event_types,
            released.seal.lineage_issue_codes,
        )
        assert released.seal.capture_commit_eligible
        finalized = recorder.legacy_finalize_completed(released, completed_at=20.5)
        assert finalized.capture_commit_status == "committed"
    finally:
        composition.close()
        durable_sink.flush_pending()
        health.close()
        spool.close()

    captured = load_episode_control_trace(tmp_path, "cs6a-episode")
    assert captured.committed
    assert tuple(
        record.fact.fact_id for record in captured.dependency_records
    ) == (manifest_fact.fact_id,)
    event_types = tuple(
        record.fact.event_type for record in captured.episode_records
    )
    assert event_types.count("human_command_emission") == 1
    assert event_types.count("replay_source_frame_emission") == 1
    assert event_types.count("authority_event") == 3
    assert event_types.count("arbitration_decision") == 2
    decisions = materialize_actual_decision_view(captured)
    assert len(decisions) == 2
    computation_by_result = {
        record.fact.payload["result_id"]: record.fact
        for record in captured.episode_records
        if record.fact.event_type == "arbitration_computation"
    }
    assert tuple(
        tuple(
            sorted(
                set(
                    computation_by_result[
                        item.fact.payload["result_id"]
                    ].payload["owner_by_component"].values()
                )
            )
        )
        for item in captured.episode_records
        if item.fact.event_type == "arbitration_decision"
    ) == (("human-main",), ("replay-main",))
