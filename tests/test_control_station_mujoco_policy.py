from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from control_station import (
    SideEffectStationOperationCoordinator,
    StationControlTraceBridge,
    StationIntentDraft,
    project_station_run_manifest,
)
from control_station.composition import (
    CS2C_SYNC_POLICY_EXECUTION_MANIFEST_ID_V1,
    CS2C_SYNC_POLICY_MUJOCO_MANIFEST_ID_V1,
    CS2C_SYNC_POLICY_MUJOCO_STATION_MANIFEST_ID_V1,
    MujocoPolicyOnlyComposition,
    build_cs2c_sync_policy_execution_manifest,
    build_scoped_cs2c_sync_policy_mujoco,
    build_scoped_cs2c_sync_policy_mujoco_station,
)
from dataset_contracts.completeness import trace_requirement_profile
from dataset_contracts.control_trace_episode import (
    CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1,
)
from dataset_interface.control_trace import (
    load_episode_control_trace,
    materialize_actual_decision_view,
)
from robot_motion.action import (
    IMMEDIATE_QREF_SIM_V1,
    SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
)
from robot_motion.action.authoritative_trace import (
    project_control_execution_manifest,
)
from robot_motion.deploy import (
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    CanonicalDeployActionPipeline,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    build_deployment_observation,
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
    def __init__(self, value: float = 30.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


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


class SyncQposPolicy:
    name = "cs2c-sync-qpos-fixture"

    def predict(self, _observation):
        return {
            "left_arm_qpos": np.full((1, 6), 0.1, dtype=np.float32),
            "right_arm_qpos": np.full((1, 6), -0.1, dtype=np.float32),
        }


class SyncQposChunkPolicy:
    name = "cs2c-sync-qpos-chunk-fixture"

    def predict(self, _observation):
        return {
            "left_arm_qpos": np.asarray(
                [[0.05] * 6, [0.1] * 6], dtype=np.float32
            ),
            "right_arm_qpos": np.asarray(
                [[-0.05] * 6, [-0.1] * 6], dtype=np.float32
            ),
        }


def _observation(timestamp: float = 30.0):
    pose = np.eye(4, dtype=float)
    return build_deployment_observation(
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=False,
        step_index=1,
        arm_states={
            side: {
                "current_qpos": np.zeros(6, dtype=np.float32),
                "current_tcp_pose": pose,
                "actual_tcp_pose": pose,
            }
            for side in ("left", "right")
        },
        hand_states={},
        timestamp=timestamp,
    )


def _pipeline() -> CanonicalDeployActionPipeline:
    return CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("left", "right"),
                arms_enabled=True,
                hands_enabled=False,
                max_arm_xyz_step_m=0.05,
                max_arm_rot_step_deg=20.0,
                max_arm_joint_step_rad=0.2,
                arm_joint_lower_rad=(-3.0,) * 6,
                arm_joint_upper_rad=(3.0,) * 6,
            )
        ),
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=False,
    )


def _policy_runtime(clock: ManualClock) -> PolicyDeployRuntime:
    return PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=False,
            policy_node_id="single-policy",
            policy_clock_domain_id="local_monotonic",
            action_keyframe_period_sec=1.0 / 30.0,
            action_maximum_hold_sec=0.1,
        ),
        monotonic_clock=clock,
    )


def _mujoco_runtime(
    clock: ManualClock,
    *,
    trace_sink=None,
    runtime_session_id: str = "cs2c-runtime",
    control_domain_id: str = "cs2c-mujoco-domain",
    embodiment_instance_id: str = "cs2c-mujoco-rm65",
    scheduled: bool = False,
):
    scene = FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=TimingSpec(
                sample_hz=30.0,
                policy_hz=30.0,
                reference_hz=60.0,
                ik_hz=60.0,
                command_hz=120.0,
                physics_hz=1000.0,
            ),
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
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            producer_id="policy-main",
            producer_epoch=1,
            policy_node_id="single-policy",
            decision_hz=30.0,
            initial_control_mode="policy",
            clock_domain_id="local_monotonic",
            reference_activation_profile_id=(
                SCHEDULED_REFERENCE_SEGMENT_SIM_V1
                if scheduled
                else IMMEDIATE_QREF_SIM_V1
            ),
        ),
        trace_sink=trace_sink,
        clock=clock,
    )
    return runtime, scene


def _emission(policy_runtime, pipeline, clock):
    pop_count = policy_runtime.action_queue_pop_count()
    observation = _observation(clock())
    prediction = policy_runtime.predict_and_queue(
        SyncQposPolicy(),
        observation,
        now=clock(),
        action_keyframe_period_sec=1.0 / 30.0,
    )
    assert prediction.accepted is True
    assert policy_runtime.action_queue_pop_count() == pop_count
    prepared = policy_runtime.prepare_next_action(
        observation,
        preparer=pipeline,
        policy_name=SyncQposPolicy.name,
    )
    assert prepared.emission_error is None
    assert prepared.emission is not None
    assert policy_runtime.action_queue_pop_count() == pop_count + 1
    return prepared.emission


def _chunk_emissions(policy_runtime, pipeline, clock):
    pop_count = policy_runtime.action_queue_pop_count()
    observation = _observation(clock())
    prediction = policy_runtime.predict_and_queue(
        SyncQposChunkPolicy(),
        observation,
        now=clock(),
        action_keyframe_period_sec=1.0 / 30.0,
    )
    assert prediction.accepted is True
    assert prediction.executed_chunk_size == 2
    emissions = []
    for _ in range(2):
        prepared = policy_runtime.prepare_next_action(
            observation,
            preparer=pipeline,
            policy_name=SyncQposChunkPolicy.name,
        )
        assert prepared.emission_error is None
        assert prepared.emission is not None
        emissions.append(prepared.emission)
    assert policy_runtime.action_queue_pop_count() == pop_count + 2
    assert tuple(item.chunk_keyframe_index for item in emissions) == (0, 1)
    return tuple(emissions)


def _submit_station(station, *, sequence: int, kind: str, subject_id=None):
    state = station.station.state
    transition = station.station.submit_intent(
        StationIntentDraft(
            client_id="operator:cs2c-test",
            client_sequence=sequence,
            intent_kind=kind,
            produced_at=station.data_plane.runtime.clock(),
            produced_clock_domain_id="operator-test-monotonic",
            expected_state_revision=state.state_revision,
            manifest_hash=station.station.manifest.semantic_content_hash,
            subject_id=subject_id,
            reason="cs2c sync policy fixture",
        )
    )
    if transition.receipt.status == "operation_requested":
        return transition, station.complete_pending_operation(transition)
    return transition, None


def _advance_station_to_authority(station) -> None:
    target_id = station.station.manifest.execution_binding.binding_id
    source_id = station.source_adapter.subject_id
    for sequence, (kind, subject_id) in enumerate(
        (
            ("acquire_control_lease", None),
            ("connect_target", target_id),
            ("connect_source", source_id),
            ("run_preflight", target_id),
            ("run_preflight", source_id),
            ("request_candidate", source_id),
            ("request_motion_authority", source_id),
        )
    ):
        transition, execution = _submit_station(
            station,
            sequence=sequence,
            kind=kind,
            subject_id=subject_id,
        )
        assert transition.receipt.accepted is True
        if transition.receipt.status == "operation_requested":
            assert execution is not None
            assert execution.completion_transition.accepted is True


def test_cs2c_consumes_exact_sync_policy_emission_without_second_pop() -> None:
    clock = ManualClock()
    policy_runtime = _policy_runtime(clock)
    pipeline = _pipeline()
    emission = _emission(policy_runtime, pipeline, clock)
    pop_count = policy_runtime.action_queue_pop_count()
    queue_length = policy_runtime.action_queue_length()
    runtime, scene = _mujoco_runtime(clock)
    composition = build_scoped_cs2c_sync_policy_mujoco(
        runtime,
        policy_runtime=policy_runtime,
        pipeline=pipeline,
    )

    assert composition.manifest.manifest_id == CS2C_SYNC_POLICY_MUJOCO_MANIFEST_ID_V1
    assert composition.binding_receipt.source_emission_identity_preserved is True
    assert composition.binding_receipt.policy_queue_owned_by_control_station is False
    assert composition.binding_receipt.prediction_owned_by_control_station is False
    assert composition.binding_receipt.station_authority_lifecycle_bound is False

    composition.start()
    submission = composition.submit_emission(emission)
    assert submission.source_emission is emission
    assert submission.accepted is True
    assert policy_runtime.action_queue_pop_count() == pop_count == 1
    assert policy_runtime.action_queue_length() == queue_length == 0

    step = composition.step(now=clock(), physics_ticks_after_dispatch=12)
    assert step.decision is not None
    assert step.decision.selected_by_group["arms"].endswith("policy-main:1:0")
    assert step.safety_result is not None
    np.testing.assert_allclose(
        step.safety_result.safe_action.arm_qpos_by_side["left"], 0.1, atol=1e-6
    )
    np.testing.assert_allclose(
        step.safety_result.safe_action.arm_qpos_by_side["right"], -0.1, atol=1e-6
    )
    assert scene.physics_commands
    assert np.all(scene.physics_commands[-1]["left"] > 0.0)
    assert np.all(scene.physics_commands[-1]["right"] < 0.0)

    facts = {fact.fact_id: fact for fact in runtime.trace_facts}
    assert emission.emission_id in facts
    assert facts[emission.emission_id].event_type == "policy_action_emission"
    assert f"source-validation:{emission.emission_id}" in facts
    proposal = submission.proposal_admission.proposal
    assert proposal is not None
    assert proposal.audit_metadata["policy_action_emission_id"] == emission.emission_id

    with pytest.raises(ValueError, match="submitted only once"):
        composition.submit_emission(emission)
    composition.close()


def test_cs2c_application_composition_has_no_predict_or_pop_call() -> None:
    source = inspect.getsource(MujocoPolicyOnlyComposition)
    assert ".predict(" not in source
    assert "predict_and_queue(" not in source
    assert "prepare_next_action(" not in source
    assert "pop_action(" not in source


def test_cs2c_station_lifecycle_gates_the_same_policy_emission() -> None:
    clock = ManualClock()
    policy_runtime = _policy_runtime(clock)
    pipeline = _pipeline()
    first = _emission(policy_runtime, pipeline, clock)
    runtime, _scene = _mujoco_runtime(clock)
    station = build_scoped_cs2c_sync_policy_mujoco_station(
        runtime,
        policy_runtime=policy_runtime,
        pipeline=pipeline,
        station_clock=clock,
    )
    assert (
        station.station.manifest.manifest_id
        == CS2C_SYNC_POLICY_MUJOCO_STATION_MANIFEST_ID_V1
    )
    assert station.data_plane.binding_receipt.station_authority_lifecycle_bound is True
    with pytest.raises(RuntimeError, match="authority is not active"):
        station.submit_emission(first)

    _advance_station_to_authority(station)
    assert station.publication_barrier.snapshot().enabled is True
    before_pop = policy_runtime.action_queue_pop_count()
    submission = station.submit_emission(first)
    assert submission.accepted is True
    assert policy_runtime.action_queue_pop_count() == before_pop
    step = station.step(now=clock(), physics_ticks_after_dispatch=2)
    assert step.decision is not None

    second = _emission(policy_runtime, pipeline, clock)
    source_id = station.source_adapter.subject_id
    transition, execution = _submit_station(
        station,
        sequence=7,
        kind="release_motion_authority",
        subject_id=source_id,
    )
    assert transition.receipt.accepted is True
    assert execution is not None and execution.completion_transition.accepted is True
    assert station.publication_barrier.snapshot().enabled is False
    with pytest.raises(RuntimeError, match="authority is not active"):
        station.submit_emission(second)
    station.close()


def test_cs2c_policy_station_seals_commits_and_rebuilds_lineage(
    tmp_path: Path,
) -> None:
    """Formal fixture: real sync emission/runtime/WAL/episode/reader only."""

    runtime_session_id = "cs2c-formal-runtime"
    control_domain_id = "cs2c-formal-domain"
    embodiment_instance_id = "cs2c-formal-rm65"
    trace_session_id = "cs2c-formal-trace"
    episode_uuid = "cs2c-formal-episode"
    clock = ManualClock()
    profile = trace_requirement_profile(
        "deployable_sim_scheduled_qpos_policy_station_v1"
    )
    trace_runtime = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id=trace_session_id,
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            profile_id=profile.profile_id,
            started_at=clock(),
            clock_domain_id="local_monotonic",
        ),
        profile=profile,
        capacity=8192,
    )
    session_root = (
        tmp_path / CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1 / trace_session_id
    )
    session_root.mkdir(parents=True)
    spool = ControlTraceSpool(
        session_root / "control_trace.wal",
        trace_session_id=trace_session_id,
        max_bytes=32 << 20,
        physical_preallocation=False,
    )
    health = ControlTraceWriterHealthLedger(
        session_root / "writer_health.ledger",
        trace_session_id=trace_session_id,
    )
    durable_sink = ControlTraceSpoolSink(
        sink_id="cs2c-policy-station-sim-spool",
        runtime=trace_runtime,
        spool=spool,
        health_ledger=health,
        clock=clock,
    )
    dependency_sequences: list[int] = []
    recorder = EpisodeTraceRecorderSink(
        trace_runtime,
        lifecycle_owner_id="cs2c-test-composition-owner",
        flush_to_durable=durable_sink.flush_pending,
        dependency_sequences=lambda _runtime: tuple(dependency_sequences),
        manifest_store=EpisodeCaptureManifestStore(tmp_path),
        uuid_factory=lambda: episode_uuid,
        control_clock=clock,
        commit_clock=clock,
    )

    def admit_trace(fact):
        receipt = trace_runtime.issue(fact)
        if not receipt.accepted:
            raise RuntimeError(
                f"CS2c trace rejected {fact.event_type}: {receipt.rejection_code}"
            )
        return receipt

    sim_runtime, _scene = _mujoco_runtime(
        clock,
        trace_sink=admit_trace,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        scheduled=True,
    )
    policy_runtime = _policy_runtime(clock)
    pipeline = _pipeline()
    station = build_scoped_cs2c_sync_policy_mujoco_station(
        sim_runtime,
        policy_runtime=policy_runtime,
        pipeline=pipeline,
        station_clock=clock,
        station_clock_domain_id="local_monotonic",
    )
    execution_manifest = build_cs2c_sync_policy_execution_manifest(
        sim_runtime,
        station.station.manifest,
    )
    assert execution_manifest.manifest_id == CS2C_SYNC_POLICY_EXECUTION_MANIFEST_ID_V1
    for dependency_fact in (
        project_control_execution_manifest(
            sim_runtime.core.trace_context,
            execution_manifest,
            occurred_at=clock(),
        ),
        project_station_run_manifest(
            station.station.manifest,
            runtime_session_id=runtime_session_id,
            occurred_at=clock(),
            clock_domain_id="local_monotonic",
        ),
    ):
        receipt = admit_trace(dependency_fact)
        assert receipt.sink_accepted_sequence is not None
        dependency_sequences.append(receipt.sink_accepted_sequence)

    bridge = StationControlTraceBridge(
        manifest=station.station.manifest,
        runtime_session_id=runtime_session_id,
        embodiment_instance_id=embodiment_instance_id,
        trace_sink=admit_trace,
        clock=clock,
        clock_domain_id="local_monotonic",
    )
    station.bind_publication_barrier_observer(
        bridge.record_publication_barrier_transition
    )
    coordinator = SideEffectStationOperationCoordinator(
        coordinator_id="cs2c-policy-capture-side-effect-worker",
        runtime=station.station,
        adapters=(station.target_adapter, station.source_adapter),
        operation_prepare=station.prepare_pending_operation,
        operation_executor=station.complete_pending_operation,
        intent_transition_observer=bridge.record_intent_transition,
        operation_transition_observer=bridge.record_operation_transition,
        queue_capacity=2,
        clock=clock,
        clock_domain_id="local_monotonic",
    )

    def station_operation(sequence: int, kind: str, subject_id: str | None):
        before = len(coordinator.completions)
        state = station.station.state
        submission = coordinator.submit_intent(
            StationIntentDraft(
                client_id="operator:cs2c-formal-capture",
                client_sequence=sequence,
                intent_kind=kind,
                produced_at=clock(),
                produced_clock_domain_id="local_monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id=subject_id,
                reason="cs2c policy formal capture fixture",
            )
        )
        if submission.transition.receipt.status != "operation_requested":
            return None
        assert coordinator.wait_until_idle(timeout_sec=2.0)
        assert len(coordinator.completions) == before + 1
        return coordinator.completions[-1]

    prepared = recorder.prepare_episode(legacy_episode_index=0, prepared_at=clock())
    recorder.bind_prepared_episode(prepared, bound_at=clock())
    coordinator.start()
    emissions = ()
    step = None
    try:
        target_id = station.station.manifest.execution_binding.binding_id
        source_id = station.source_adapter.subject_id
        for sequence, (kind, subject_id) in enumerate(
            (
                ("acquire_control_lease", None),
                ("connect_target", target_id),
                ("connect_source", source_id),
                ("run_preflight", target_id),
                ("run_preflight", source_id),
                ("request_candidate", source_id),
                ("request_motion_authority", source_id),
            )
        ):
            completion = station_operation(sequence, kind, subject_id)
            if completion is not None:
                assert completion.completion_transition.accepted

        assert station.publication_barrier.snapshot().enabled
        emissions = _chunk_emissions(policy_runtime, pipeline, clock)
        pop_count = policy_runtime.action_queue_pop_count()
        submissions = tuple(station.submit_emission(item) for item in emissions)
        assert all(item.accepted for item in submissions)
        assert tuple(item.source_emission for item in submissions) == emissions
        assert policy_runtime.action_queue_pop_count() == pop_count == 2
        step = station.step(now=clock(), physics_ticks_after_dispatch=60)
        assert step.status == "dispatched", (
            step.decision,
            step.computation.output_status,
            step.computation.status_by_group,
            step.computation.rejections,
            step.publication_attempt,
            step.directive,
        )

        clock.value = 30.2
        release = station_operation(7, "release_motion_authority", source_id)
        assert release is not None and release.completion_transition.accepted
        station.step(now=clock())
        assert not station.publication_barrier.snapshot().enabled

        clock.value = 30.3
        released = recorder.release_episode(
            prepared,
            unbound_at=clock(),
            reason="cs2c_policy_station_sim_finalize",
        )
        assert released.seal.trace_sealed, (
            released.seal.reason_codes,
            released.seal.missing_required_event_types,
            released.seal.lineage_issue_codes,
        )
        assert released.seal.capture_commit_eligible
        finalized = recorder.legacy_finalize_completed(released, completed_at=30.4)
        assert finalized.capture_commit_status == "committed"
    finally:
        coordinator.stop(timeout_sec=2.0)
        station.close()
        durable_sink.flush_pending()
        health.close()
        spool.close()

    captured = load_episode_control_trace(tmp_path, episode_uuid)
    assert captured.committed
    assert {
        record.fact.event_type for record in captured.dependency_records
    } == {"execution_manifest", "station_run_manifest"}
    event_types = tuple(record.fact.event_type for record in captured.episode_records)
    assert event_types.count("policy_action_emission") == 2
    assert event_types.count("source_validation") == 2
    assert event_types.count("station_intent_transition") == 8
    assert event_types.count("station_operation_transition") == 7
    assert event_types.count("station_publication_barrier_transition") == 2
    assert "reference_segment_admission" in event_types
    assert "planner_command_frame" in event_types
    assert "execution_cancellation" in event_types
    assert len(emissions) == 2
    captured_fact_ids = {
        record.fact.fact_id for record in captured.episode_records
    }
    assert {item.emission_id for item in emissions} <= captured_fact_ids
    decisions = materialize_actual_decision_view(captured)
    assert len(decisions) == 1
    assert step is not None
    assert decisions[0].decision_id == step.decision.decision_id


def test_cs2c_binding_rejects_policy_node_drift_before_runtime_admission() -> None:
    clock = ManualClock()
    policy_runtime = _policy_runtime(clock)
    pipeline = _pipeline()
    emission = _emission(policy_runtime, pipeline, clock)
    runtime, _scene = _mujoco_runtime(clock)
    composition = build_scoped_cs2c_sync_policy_mujoco(
        runtime,
        policy_runtime=policy_runtime,
        pipeline=pipeline,
    )
    composition.start()
    drifted = emission.__class__(
        **{**emission.__dict__, "policy_node_id": "another-policy"}
    )
    before = len(runtime.trace_facts)
    with pytest.raises(ValueError, match="another policy node"):
        composition.submit_emission(drifted)
    assert len(runtime.trace_facts) == before
    composition.close()
