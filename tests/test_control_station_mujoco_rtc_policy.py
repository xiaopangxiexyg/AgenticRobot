from __future__ import annotations

import inspect
import time
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
    CS2D_RTC_POLICY_EXECUTION_MANIFEST_ID_V1,
    CS2D_RTC_POLICY_MUJOCO_MANIFEST_ID_V1,
    CS2D_RTC_POLICY_MUJOCO_STATION_MANIFEST_ID_V1,
    MujocoRtcPolicyOnlyComposition,
    build_cs2d_rtc_policy_execution_manifest,
    build_scoped_cs2d_rtc_policy_mujoco,
    build_scoped_cs2d_rtc_policy_mujoco_station,
)
from dataset_contracts.completeness import trace_requirement_profile
from dataset_contracts.control_trace_episode import (
    CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1,
)
from dataset_interface.control_trace import (
    load_episode_control_trace,
    materialize_actual_decision_view,
)
from robot_motion.action import SCHEDULED_REFERENCE_SEGMENT_SIM_V1
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
from teleop_runtime.deploy.rtc_frontend import (
    RtcPolicyShadowConfig,
    RtcPolicyShadowFrontend,
)
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
    def __init__(self, value: float = 40.0) -> None:
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
            side: np.asarray(value, dtype=float).reshape(6).copy()
            for side, value in qpos_by_side.items()
        }
        self.physics_commands.append(command)
        self.qpos = {side: value.copy() for side, value in command.items()}


class RtcBimanualQposPolicy:
    name = "cs2d-rtc-bimanual-qpos"

    def predict(self, _observation):
        return {
            "left_arm_qpos": np.asarray(
                [[0.05] * 6, [0.1] * 6, [0.15] * 6, [0.2] * 6],
                dtype=np.float32,
            ),
            "right_arm_qpos": np.asarray(
                [[-0.05] * 6, [-0.1] * 6, [-0.15] * 6, [-0.2] * 6],
                dtype=np.float32,
            ),
        }


def _observation(timestamp: float, step_index: int):
    pose = np.eye(4, dtype=float)
    return build_deployment_observation(
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=False,
        step_index=step_index,
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
                max_arm_joint_step_rad=0.25,
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
    scheduled: bool = False,
    trace_sink=None,
    runtime_session_id: str = "cs2d-target-runtime",
    control_domain_id: str = "cs2d-domain",
    embodiment_instance_id: str = "cs2d-mujoco-rm65",
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
            producer_id="rtc-policy-main",
            producer_epoch=1,
            policy_node_id="single-policy",
            decision_hz=30.0,
            initial_control_mode="policy",
            clock_domain_id="local_monotonic",
            reference_activation_profile_id=(
                SCHEDULED_REFERENCE_SEGMENT_SIM_V1
                if scheduled
                else "immediate_qref_sim_v1"
            ),
        ),
        trace_sink=trace_sink,
        clock=clock,
    )
    return runtime, scene


def _rtc_config() -> RtcPolicyShadowConfig:
    return RtcPolicyShadowConfig(
        runtime_session_id="cs2d-rtc-source-runtime",
        control_domain_id="cs2d-domain",
        producer_id="rtc-policy-main",
        policy_node_id="single-policy",
        artifact_hash="artifact:cs2d-virtual-qpos",
        action_codec_hash="codec:cs2d-bimanual-qpos-v1",
        frame_contract_hash="frame:joint-space-not-applicable-v1",
        layout_hash="layout:cs2d-bimanual-q6-v1",
        normalization_hash="normalization:none-v1",
        prediction_horizon_steps=4,
        request_validity_steps=10,
        request_trigger_coverage_steps=2,
    )


def _advance_target_physics_to_clock(target, *, clock_origin: float, now: float) -> None:
    """Keep deterministic MuJoCo time aligned with the externally driven RTC.

    A real MuJoCo worker advances while asynchronous policy inference runs.  The
    test clock does not do that automatically, so advance only the target's
    existing physics loop; this must not predict, merge, or pop policy work.
    """

    physics_hz = float(target.rig.config.timing.physics_hz)
    while (
        float(clock_origin)
        + int(target.rig.snapshot()["physics_ticks"]) / physics_hz
        + 1e-12
        < float(now)
    ):
        target.rig.step_physics()


def _next_emitting_step(frontend, clock, *, target, target_clock_origin: float):
    base = clock()
    for index in range(120):
        now = base + index / 30.0
        clock.value = now
        _advance_target_physics_to_clock(
            target,
            clock_origin=target_clock_origin,
            now=now,
        )
        step = frontend.step(
            _observation(timestamp=100.0 + index / 30.0, step_index=index),
            now=now,
        )
        if step.action_preparation.emission is not None:
            return step
        time.sleep(0.001)
    raise AssertionError("RTC frontend did not produce a formal emission.")


def _submit_station(station, *, sequence: int, kind: str, subject_id=None):
    state = station.station.state
    transition = station.station.submit_intent(
        StationIntentDraft(
            client_id="operator:cs2d-test",
            client_sequence=sequence,
            intent_kind=kind,
            produced_at=station.data_plane.runtime.clock(),
            produced_clock_domain_id="operator-test-monotonic",
            expected_state_revision=state.state_revision,
            manifest_hash=station.station.manifest.semantic_content_hash,
            subject_id=subject_id,
            reason="cs2d RTC policy fixture",
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


def test_cs2d_consumes_bound_external_rtc_emission_without_owning_pop() -> None:
    clock = ManualClock()
    policy_runtime = _policy_runtime(clock)
    pipeline = _pipeline()
    rtc_config = _rtc_config()
    frontend = RtcPolicyShadowFrontend(
        config=rtc_config,
        deploy_runtime=policy_runtime,
        policy=RtcBimanualQposPolicy(),
        preparer=pipeline,
        created_at=clock(),
        clock=clock,
    )
    target, scene = _mujoco_runtime(clock)
    composition = build_scoped_cs2d_rtc_policy_mujoco(
        target,
        rtc_config=rtc_config,
        policy_runtime=policy_runtime,
        pipeline=pipeline,
    )
    assert composition.manifest.manifest_id == CS2D_RTC_POLICY_MUJOCO_MANIFEST_ID_V1
    assert composition.binding_receipt.policy_queue_owned_by_control_station is False
    assert composition.binding_receipt.prediction_owned_by_control_station is False
    assert not composition.binding_receipt.station_authority_lifecycle_bound

    target_clock_origin = clock()
    composition.start()
    frontend.start()
    try:
        frontend_step = _next_emitting_step(
            frontend,
            clock,
            target=target,
            target_clock_origin=target_clock_origin,
        )
        emission = frontend_step.action_preparation.emission
        assert emission is not None
        pop_count = emission.queue_pop_count
        submission = composition.submit_emission(emission)
        assert submission.accepted
        assert submission.source_emission is emission
        assert emission.queue_pop_count == pop_count
        assert emission.pre_safety_robot_action is not None
        assert np.all(
            emission.pre_safety_robot_action.arm_qpos_by_side["left"] > 0.0
        ), emission.pre_safety_robot_action

        decision = composition.step(
            now=clock(),
            physics_ticks_after_dispatch=12,
        )
        assert decision.status == "dispatched"
        assert decision.decision is not None
        assert scene.physics_commands
        assert np.all(scene.physics_commands[-1]["left"] > 0.0), (
            decision.safety_result,
            scene.physics_commands[-1],
        )
        assert np.all(scene.physics_commands[-1]["right"] < 0.0)

        with pytest.raises(ValueError, match="submitted only once"):
            composition.submit_emission(emission)
        with pytest.raises(ValueError, match="submitted only once"):
            composition.submit_frontend_step(frontend_step)
    finally:
        frontend.close(now=clock())
        composition.close()


def test_cs2d_composition_never_steps_or_mutates_the_rtc_frontend() -> None:
    source = inspect.getsource(MujocoRtcPolicyOnlyComposition)
    assert "frontend.step(" not in source
    assert "predict(" not in source
    assert "merge_policy" not in source
    assert "pop_due" not in source


def test_cs2d_station_gates_downstream_without_owning_rtc_lifecycle() -> None:
    clock = ManualClock()
    policy_runtime = _policy_runtime(clock)
    pipeline = _pipeline()
    rtc_config = _rtc_config()
    frontend = RtcPolicyShadowFrontend(
        config=rtc_config,
        deploy_runtime=policy_runtime,
        policy=RtcBimanualQposPolicy(),
        preparer=pipeline,
        created_at=clock(),
        clock=clock,
    )
    target, _scene = _mujoco_runtime(clock)
    station = build_scoped_cs2d_rtc_policy_mujoco_station(
        target,
        rtc_config=rtc_config,
        policy_runtime=policy_runtime,
        pipeline=pipeline,
        station_clock=clock,
        station_clock_domain_id="local_monotonic",
    )
    assert (
        station.station.manifest.manifest_id
        == CS2D_RTC_POLICY_MUJOCO_STATION_MANIFEST_ID_V1
    )
    assert station.data_plane.binding_receipt.station_authority_lifecycle_bound
    assert "control_station_rtc_producer_lifecycle" in (
        station.station.manifest.explicit_exclusions
    )

    target_clock_origin = clock()
    _advance_station_to_authority(station)
    assert station.publication_barrier.snapshot().enabled
    frontend.start()
    try:
        first = _next_emitting_step(
            frontend,
            clock,
            target=target,
            target_clock_origin=target_clock_origin,
        )
        pop_count = policy_runtime.action_queue_pop_count()
        first_emission = first.emission_result.emission
        assert first_emission is not None
        submission = station.submit_emission(first_emission)
        assert submission.accepted
        assert policy_runtime.action_queue_pop_count() == pop_count
        decision = station.step(now=clock(), physics_ticks_after_dispatch=12)
        assert decision.status == "dispatched"

        second = _next_emitting_step(
            frontend,
            clock,
            target=target,
            target_clock_origin=target_clock_origin,
        )
        producer_state_before_release = frontend.state.semantic_content_hash
        source_id = station.source_adapter.subject_id
        transition, execution = _submit_station(
            station,
            sequence=7,
            kind="release_motion_authority",
            subject_id=source_id,
        )
        assert transition.receipt.accepted
        assert execution is not None and execution.completion_transition.accepted
        assert not station.publication_barrier.snapshot().enabled
        assert frontend.state.semantic_content_hash == producer_state_before_release
        with pytest.raises(RuntimeError, match="authority is not active"):
            station.submit_frontend_step(second)
        with pytest.raises(RuntimeError, match="authority is not active"):
            station.submit_emission(second.action_preparation.emission)
    finally:
        frontend.close(now=clock())
        station.close()


def test_cs2d_rtc_policy_station_seals_commits_and_rebuilds_lineage(
    tmp_path: Path,
) -> None:
    """Formal fixture over the real RTC frontend, WAL and committed reader."""

    runtime_session_id = "cs2d-formal-runtime"
    control_domain_id = "cs2d-domain"
    embodiment_instance_id = "cs2d-formal-rm65"
    trace_session_id = "cs2d-formal-trace"
    episode_uuid = "cs2d-formal-episode"
    clock = ManualClock()
    profile = trace_requirement_profile(
        "deployable_sim_immediate_qref_rtc_policy_station_v1"
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
        sink_id="cs2d-rtc-policy-station-sim-spool",
        runtime=trace_runtime,
        spool=spool,
        health_ledger=health,
        clock=clock,
    )
    dependency_sequences: list[int] = []
    recorder = EpisodeTraceRecorderSink(
        trace_runtime,
        lifecycle_owner_id="cs2d-test-composition-owner",
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
                f"CS2d trace rejected {fact.event_type}: {receipt.rejection_code}"
            )
        return receipt

    target, _scene = _mujoco_runtime(
        clock,
        trace_sink=admit_trace,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
    )
    policy_runtime = _policy_runtime(clock)
    pipeline = _pipeline()
    rtc_config = _rtc_config()
    frontend = RtcPolicyShadowFrontend(
        config=rtc_config,
        deploy_runtime=policy_runtime,
        policy=RtcBimanualQposPolicy(),
        preparer=pipeline,
        created_at=clock(),
        clock=clock,
    )
    station = build_scoped_cs2d_rtc_policy_mujoco_station(
        target,
        rtc_config=rtc_config,
        policy_runtime=policy_runtime,
        pipeline=pipeline,
        station_clock=clock,
        station_clock_domain_id="local_monotonic",
    )
    execution_manifest = build_cs2d_rtc_policy_execution_manifest(
        target,
        station.station.manifest,
    )
    assert (
        execution_manifest.manifest_id
        == CS2D_RTC_POLICY_EXECUTION_MANIFEST_ID_V1
    )
    for dependency_fact in (
        project_control_execution_manifest(
            target.core.trace_context,
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
        coordinator_id="cs2d-rtc-policy-capture-side-effect-worker",
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
                client_id="operator:cs2d-formal-capture",
                client_sequence=sequence,
                intent_kind=kind,
                produced_at=clock(),
                produced_clock_domain_id="local_monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id=subject_id,
                reason="cs2d RTC policy formal capture fixture",
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
    frontend.start()
    frontend_step = None
    decision = None
    target_clock_origin = clock()
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

        frontend_step = _next_emitting_step(
            frontend,
            clock,
            target=target,
            target_clock_origin=target_clock_origin,
        )
        submission = station.submit_frontend_step(frontend_step)
        assert submission.accepted
        decision = station.step(now=clock(), physics_ticks_after_dispatch=12)
        assert decision.status == "dispatched"

        clock.value += 0.1
        release = station_operation(7, "release_motion_authority", source_id)
        assert release is not None and release.completion_transition.accepted
        station.step(now=clock())
        assert not station.publication_barrier.snapshot().enabled

        clock.value += 0.1
        released = recorder.release_episode(
            prepared,
            unbound_at=clock(),
            reason="cs2d_rtc_policy_station_sim_finalize",
        )
        assert released.seal.trace_sealed, (
            released.seal.reason_codes,
            released.seal.missing_required_event_types,
            released.seal.lineage_issue_codes,
        )
        assert released.seal.capture_commit_eligible
        finalized = recorder.legacy_finalize_completed(
            released,
            completed_at=clock() + 0.1,
        )
        assert finalized.capture_commit_status == "committed"
    finally:
        coordinator.stop(timeout_sec=2.0)
        frontend.close(now=clock())
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
    assert event_types.count("policy_action_emission") == 1
    assert event_types.count("source_validation") == 1
    assert event_types.count("station_intent_transition") == 8
    assert event_types.count("station_operation_transition") == 7
    assert event_types.count("station_publication_barrier_transition") == 2
    assert "reference_segment" not in event_types
    assert "planner_command_frame" not in event_types
    assert "execution_cancellation" in event_types
    assert frontend_step is not None
    emission = frontend_step.action_preparation.emission
    assert emission is not None
    captured_fact_ids = {
        record.fact.fact_id for record in captured.episode_records
    }
    assert emission.emission_id in captured_fact_ids
    decisions = materialize_actual_decision_view(captured)
    assert len(decisions) == 1
    assert decision is not None and decision.decision is not None
    assert decisions[0].decision_id == decision.decision.decision_id


def test_cs2d_scheduled_target_lead_fails_closed_until_explicitly_aligned() -> None:
    clock = ManualClock()
    target, _scene = _mujoco_runtime(clock, scheduled=True)
    with pytest.raises(ValueError, match="immediate-qref only"):
        build_scoped_cs2d_rtc_policy_mujoco(
            target,
            rtc_config=_rtc_config(),
            policy_runtime=_policy_runtime(clock),
            pipeline=_pipeline(),
        )
