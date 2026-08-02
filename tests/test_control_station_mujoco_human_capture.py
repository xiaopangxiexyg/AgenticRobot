from __future__ import annotations

from pathlib import Path

import numpy as np

from control_station.composition import (
    MujocoHumanOnlyComposition,
    build_cs2a1_mujoco_human_manifest,
    build_cs2a2a_synthetic_human_execution_manifest,
    build_cs2a2b_synthetic_human_execution_manifest,
    build_scoped_cs2a2b_mujoco_human_station,
    build_mujoco_human_proposal_adapter,
)
from control_station import (
    SideEffectStationOperationCoordinator,
    StationControlTraceBridge,
    StationIntentDraft,
    project_station_run_manifest,
)
from control_station.sources import (
    SyntheticHumanQposSource,
    SyntheticHumanQposSourceSpec,
)
from dataset_contracts.completeness import trace_requirement_profile
from dataset_contracts.control_trace_episode import (
    CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1,
)
from dataset_contracts.stream_catalog import NativeStreamContract
from dataset_interface.control_trace import (
    load_episode_control_trace,
    materialize_actual_decision_view,
)
from robot_motion.action import SCHEDULED_REFERENCE_SEGMENT_SIM_V1
from robot_motion.action.authoritative_trace import (
    project_control_execution_manifest,
)
from sim.contracts import TimingSpec
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.deploy import MujocoDeployConfig, MujocoDeployRuntime
from collection_runtime.stream_sink import NativeStreamEpisodeWriter
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
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class DeterministicSceneFacade:
    """Minimal scene only; all control/capture behavior is production code."""

    arm_specs = {"left": object(), "right": object()}
    gripper_specs = {}

    def __init__(self) -> None:
        self.qpos = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }

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
        self.qpos = {
            side: np.asarray(value, dtype=float).reshape(6).copy()
            for side, value in qpos_by_side.items()
        }


def _timing() -> TimingSpec:
    return TimingSpec(
        sample_hz=15.0,
        policy_hz=15.0,
        reference_hz=60.0,
        ik_hz=60.0,
        command_hz=120.0,
        physics_hz=1000.0,
    )


def test_cs2a2a_human_only_sim_seals_commits_and_rebuilds_lineage(
    tmp_path: Path,
) -> None:
    """Test-side composition; no production station entrypoint is implied."""

    runtime_session_id = "cs2a2a-runtime"
    control_domain_id = "cs2a2a-domain"
    trace_session_id = "cs2a2a-trace"
    episode_uuid = "cs2a2a-episode"
    clock = ManualClock()
    profile = trace_requirement_profile("deployable_sim_scheduled_qpos_v1")
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
        capacity=4096,
    )
    session_root = (
        tmp_path / CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1 / trace_session_id
    )
    session_root.mkdir(parents=True)
    spool = ControlTraceSpool(
        session_root / "control_trace.wal",
        trace_session_id=trace_session_id,
        max_bytes=16 << 20,
        physical_preallocation=False,
    )
    health = ControlTraceWriterHealthLedger(
        session_root / "writer_health.ledger",
        trace_session_id=trace_session_id,
    )
    durable_sink = ControlTraceSpoolSink(
        sink_id="cs2a2a-human-sim-spool",
        runtime=trace_runtime,
        spool=spool,
        health_ledger=health,
        clock=clock,
    )
    dependency_sequences: list[int] = []
    sim_state_contract = NativeStreamContract(
        stream_id="sim.arm_qpos",
        modality="robot_state",
        producer_id="mujoco-position-rig",
        value_schema="float64[2,6]",
        shape=(2, 6),
        dtype="<f8",
        units="rad",
        frame=None,
        measured_clock_domain_id="local_monotonic",
        available_clock_domain_id="local_monotonic",
        access_role="actor_input",
    )
    recorder = EpisodeTraceRecorderSink(
        trace_runtime,
        lifecycle_owner_id="cs2a2a-test-composition-owner",
        flush_to_durable=durable_sink.flush_pending,
        dependency_sequences=lambda _runtime: tuple(dependency_sequences),
        manifest_store=EpisodeCaptureManifestStore(tmp_path),
        uuid_factory=lambda: episode_uuid,
        control_clock=clock,
        commit_clock=clock,
        committed_capabilities=("control_trace_core", "native_streams"),
        native_stream_writer_factory=lambda prepared: NativeStreamEpisodeWriter(
            tmp_path,
            episode_uuid=prepared.episode_uuid,
            legacy_episode_index=prepared.legacy_episode_index,
            contracts={sim_state_contract.stream_id: sim_state_contract},
        ),
    )

    def admit_trace(fact) -> None:
        receipt = trace_runtime.issue(fact)
        if not receipt.accepted:
            raise RuntimeError(
                f"CS2a2a trace rejected {fact.event_type}: {receipt.rejection_code}"
            )

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
    sim_runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            embodiment_instance_id="cs2a2a-embodiment",
            producer_id="synthetic-human-main",
            authority_policy_id="mujoco-single-human-authority-v1",
            initial_control_mode="human",
            reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
        ),
        trace_sink=admit_trace,
        clock=clock,
    )
    source_spec = SyntheticHumanQposSourceSpec()
    source = SyntheticHumanQposSource(source_spec)
    adapter = build_mujoco_human_proposal_adapter(sim_runtime)
    station_manifest = build_cs2a1_mujoco_human_manifest(
        sim_runtime,
        adapter,
        source_spec,
    )
    execution_manifest = build_cs2a2a_synthetic_human_execution_manifest(
        sim_runtime,
        station_manifest,
    )
    manifest_fact = project_control_execution_manifest(
        sim_runtime.core.trace_context,
        execution_manifest,
        occurred_at=clock(),
    )
    manifest_receipt = trace_runtime.issue(manifest_fact)
    assert manifest_receipt.accepted
    assert manifest_receipt.sink_accepted_sequence is not None
    dependency_sequences.append(manifest_receipt.sink_accepted_sequence)
    composition = MujocoHumanOnlyComposition(
        manifest=station_manifest,
        runtime=sim_runtime,
        adapter=adapter,
        source_spec=source_spec,
    )
    prepared = recorder.prepare_episode(legacy_episode_index=0, prepared_at=clock())
    recorder.bind_prepared_episode(prepared, bound_at=clock())
    try:
        composition.start()
        emission = source.emit(
            {
                "left_arm": np.full(6, 0.001, dtype=float),
                "right_arm": np.full(6, -0.001, dtype=float),
            },
            emitted_at=clock(),
        )
        submission = composition.submit_emission(
            emission,
            effective_from=clock(),
            effective_until=clock() + 0.1,
            target_effective_at=clock(),
        )
        assert submission.accepted
        step = composition.step(now=clock(), physics_ticks_after_dispatch=12)
        assert step.status == "dispatched"
        recorder.native_stream_capture(prepared).append(
            stream_id=sim_state_contract.stream_id,
            sample_id="sim-arm-qpos-0",
            sequence=0,
            measured_at=clock(),
            measured_clock_domain_id="local_monotonic",
            available_at=clock(),
            available_clock_domain_id="local_monotonic",
            source_payload_object_id="mujoco-scene-arm-qpos-0",
            payload=np.stack(
                [scene.get_arm_qpos(side) for side in ("left", "right")]
            ).astype(np.float64, copy=False),
            written_at=clock(),
            written_clock_domain_id="local_monotonic",
        )

        clock.value = 10.2
        sim_runtime.set_execution_gate(
            "hold",
            reason="cs2a2a_capture_terminal_hold",
            source_event_ids=("cs2a2a-stop",),
        )
        hold_step = composition.step(now=clock())
        assert hold_step.status in {"execution_hold_active", "not_due"}

        clock.value = 10.3
        released = recorder.release_episode(
            prepared,
            unbound_at=clock(),
            reason="cs2a2a_human_sim_finalize",
        )
        assert released.seal.trace_sealed, (
            released.seal.reason_codes,
            released.seal.missing_required_event_types,
            released.seal.lineage_issue_codes,
        )
        assert released.seal.capture_commit_eligible
        finalized = recorder.legacy_finalize_completed(
            released,
            completed_at=10.4,
        )
        assert finalized.capture_commit_status == "committed"
        assert finalized.native_stream_manifest_hash is not None
    finally:
        composition.close()
        durable_sink.flush_pending()
        health.close()
        spool.close()

    captured = load_episode_control_trace(tmp_path, episode_uuid)
    assert captured.committed
    assert captured.capability_artifact_hash_by_name == {
        "native_streams": finalized.native_stream_manifest_hash
    }
    assert tuple(
        item.fact.fact_id for item in captured.dependency_records
    ) == (manifest_fact.fact_id,)
    assert captured.dependency_records[0].fact.payload[
        "semantic_content_hash"
    ] == execution_manifest.semantic_content_hash
    event_types = tuple(
        record.fact.event_type for record in captured.episode_records
    )
    assert event_types.count("human_command_emission") == 1
    assert event_types.count("source_adapter_receipt") == 1
    assert event_types.count("source_validation") == 1
    assert event_types.count("action_proposal") == 1
    assert event_types.count("arbitration_decision") == 1
    assert "policy_output" not in event_types
    decisions = materialize_actual_decision_view(captured)
    assert len(decisions) == 1
    assert decisions[0].decision_id == step.decision.decision_id


def test_cs2a2b_station_scheduled_sim_seals_commits_and_rebuilds_control_plane(
    tmp_path: Path,
) -> None:
    """Formal fixture: real Station/runtime/WAL/seal/commit, no entrypoint."""

    runtime_session_id = "cs2a2b-capture-runtime"
    control_domain_id = "cs2a2b-capture-domain"
    embodiment_instance_id = "cs2a2b-capture-embodiment"
    trace_session_id = "cs2a2b-capture-trace"
    episode_uuid = "cs2a2b-capture-episode"
    clock = ManualClock()
    profile = trace_requirement_profile(
        "deployable_sim_scheduled_qpos_station_v1"
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
        sink_id="cs2a2b-station-sim-spool",
        runtime=trace_runtime,
        spool=spool,
        health_ledger=health,
        clock=clock,
    )
    dependency_sequences: list[int] = []
    recorder = EpisodeTraceRecorderSink(
        trace_runtime,
        lifecycle_owner_id="cs2a2b-test-composition-owner",
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
                f"CS2a2b trace rejected {fact.event_type}: "
                f"{receipt.rejection_code}"
            )
        return receipt

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
    sim_runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            producer_id="synthetic-human-main",
            authority_policy_id="mujoco-single-human-authority-v1",
            initial_control_mode="human",
            reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
        ),
        trace_sink=admit_trace,
        clock=clock,
    )
    source_spec = SyntheticHumanQposSourceSpec()
    source = SyntheticHumanQposSource(source_spec)
    station = build_scoped_cs2a2b_mujoco_human_station(
        sim_runtime,
        source_spec,
        station_clock=clock,
        station_clock_domain_id="local_monotonic",
    )
    execution_manifest = build_cs2a2b_synthetic_human_execution_manifest(
        sim_runtime,
        station.station.manifest,
    )
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

    station_trace_bridge = StationControlTraceBridge(
        manifest=station.station.manifest,
        runtime_session_id=runtime_session_id,
        embodiment_instance_id=embodiment_instance_id,
        trace_sink=admit_trace,
        clock=clock,
        clock_domain_id="local_monotonic",
    )
    station.bind_publication_barrier_observer(
        station_trace_bridge.record_publication_barrier_transition
    )

    coordinator = SideEffectStationOperationCoordinator(
        coordinator_id="cs2a2b-capture-side-effect-worker",
        runtime=station.station,
        adapters=(station.target_adapter, station.source_adapter),
        operation_prepare=station.prepare_pending_operation,
        operation_executor=station.complete_pending_operation,
        intent_transition_observer=(
            station_trace_bridge.record_intent_transition
        ),
        operation_transition_observer=(
            station_trace_bridge.record_operation_transition
        ),
        queue_capacity=2,
        clock=clock,
        clock_domain_id="local_monotonic",
    )

    def submit_station_operation(
        *,
        sequence: int,
        kind: str,
        subject_id: str | None,
    ):
        before_count = len(coordinator.completions)
        state = station.station.state
        submission = coordinator.submit_intent(
            StationIntentDraft(
                client_id="operator:formal-capture",
                client_sequence=sequence,
                intent_kind=kind,
                produced_at=clock(),
                produced_clock_domain_id="local_monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id=subject_id,
                reason="cs2a2b formal capture fixture",
            )
        )
        if submission.transition.receipt.status != "operation_requested":
            return submission, None
        assert coordinator.wait_until_idle(timeout_sec=2.0)
        completions = coordinator.completions
        assert len(completions) == before_count + 1
        completion = completions[-1]
        assert (
            completion.request_transition.intent.intent_id
            == submission.transition.intent.intent_id
        )
        return submission, completion

    prepared = recorder.prepare_episode(legacy_episode_index=0, prepared_at=clock())
    recorder.bind_prepared_episode(prepared, bound_at=clock())
    coordinator.start()
    step = None
    try:
        target_id = station.station.manifest.execution_binding.binding_id
        source_id = station.source_adapter.subject_id
        operations = (
            ("acquire_control_lease", None),
            ("connect_target", target_id),
            ("connect_source", source_id),
            ("run_preflight", target_id),
            ("run_preflight", source_id),
            ("request_candidate", source_id),
            ("request_motion_authority", source_id),
        )
        authority_completion = None
        for sequence, (kind, subject_id) in enumerate(operations):
            _submission, completion = submit_station_operation(
                sequence=sequence,
                kind=kind,
                subject_id=subject_id,
            )
            if kind == "request_motion_authority":
                authority_completion = completion
        assert authority_completion is not None
        assert station.publication_barrier.snapshot().enabled

        emission = source.emit(
            {
                "left_arm": np.full(6, 0.001, dtype=float),
                "right_arm": np.full(6, -0.001, dtype=float),
            },
            emitted_at=clock(),
        )
        submission = station.submit_emission(
            emission,
            effective_from=clock(),
            effective_until=clock() + 0.1,
            target_effective_at=clock(),
        )
        assert submission.accepted
        step = station.step(now=clock(), physics_ticks_after_dispatch=60)
        assert step.status == "dispatched"

        clock.value = 10.2
        release_submission, release_completion = submit_station_operation(
            sequence=7,
            kind="release_motion_authority",
            subject_id=source_id,
        )
        assert release_completion is not None
        hold_step = station.step(now=clock())
        assert hold_step.status in {"execution_hold_active", "not_due"}
        assert not station.publication_barrier.snapshot().enabled
        assert coordinator.transition_observer_failure_count == 0
        assert station.barrier_observer_failure_count == 0
        assert station_trace_bridge.admission_failures == ()

        clock.value = 10.3
        released = recorder.release_episode(
            prepared,
            unbound_at=clock(),
            reason="cs2a2b_station_sim_finalize",
        )
        assert released.seal.trace_sealed, (
            released.seal.reason_codes,
            released.seal.missing_required_event_types,
            released.seal.lineage_issue_codes,
        )
        assert released.seal.capture_commit_eligible
        finalized = recorder.legacy_finalize_completed(
            released,
            completed_at=10.4,
        )
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
    event_types = tuple(
        record.fact.event_type for record in captured.episode_records
    )
    assert event_types.count("station_intent_transition") == 8
    assert event_types.count("station_operation_transition") == 7
    assert event_types.count("station_publication_barrier_transition") == 2
    assert "reference_segment_admission" in event_types
    assert "planner_command_frame" in event_types
    assert "execution_cancellation" in event_types
    decisions = materialize_actual_decision_view(captured)
    assert len(decisions) == 1
    assert step is not None
    assert decisions[0].decision_id == step.decision.decision_id
