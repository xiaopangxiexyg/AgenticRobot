from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from dataset_contracts.completeness import trace_requirement_profile
from dataset_contracts.control_trace_episode import (
    CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1,
)
from dataset_interface.control_trace import (
    ControlTraceReadError,
    load_episode_control_trace,
    materialize_actual_decision_view,
)
from robot_motion.action import (
    RobotAction,
    SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
    build_m6_candidate_policy_arm_qpos_manifest,
)
from robot_motion.action.authoritative_trace import (
    project_control_execution_manifest,
)
from sim.contracts import SimulationContract, TimingSpec
from sim.mujoco.actuation import rm65_bimanual_udp_feedback_profile
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.deploy import MujocoDeployConfig, MujocoDeployRuntime
from sim.mujoco.scene import MujocoScene
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
    """Minimal scene surface; control/capture modules remain production code."""

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


def _action(value: float, *, timestamp: float = 10.0) -> RobotAction:
    return RobotAction.bimanual_qpos(
        left_arm=np.full(6, value, dtype=float),
        right_arm=np.full(6, -value, dtype=float),
        timestamp=timestamp,
        source="formal-sim-capture-fixture",
    )


def _run_scheduled_qpos_formal_capture(
    tmp_path: Path,
    *,
    scene,
    suffix: str,
    actuation_profile=None,
    after_seal: Callable[[], None] | None = None,
    physics_ticks_after_dispatch: int = 60,
    hold_at: float | None = 10.2,
    expect_activation: bool = True,
    action_count: int = 1,
):
    runtime_session_id = f"formal-sim-runtime-{suffix}"
    control_domain_id = f"formal-sim-domain-{suffix}"
    trace_session_id = f"formal-sim-trace-{suffix}"
    episode_uuid = f"formal-sim-episode-{suffix}"
    profile = trace_requirement_profile("deployable_sim_scheduled_qpos_v1")
    clock = ManualClock()

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
        capacity=max(4096, int(action_count) * 64),
    )
    session_root = (
        tmp_path
        / CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1
        / trace_session_id
    )
    session_root.mkdir(parents=True)
    spool = ControlTraceSpool(
        session_root / "control_trace.wal",
        trace_session_id=trace_session_id,
        max_bytes=max(16 << 20, int(action_count) * (512 << 10)),
        physical_preallocation=False,
    )
    health = ControlTraceWriterHealthLedger(
        session_root / "writer_health.ledger",
        trace_session_id=trace_session_id,
    )
    durable_sink = ControlTraceSpoolSink(
        sink_id="formal-sim-fixture-spool",
        runtime=trace_runtime,
        spool=spool,
        health_ledger=health,
        clock=clock,
    )
    dependency_sequences: list[int] = []
    recorder = EpisodeTraceRecorderSink(
        trace_runtime,
        lifecycle_owner_id="formal-sim-fixture-owner",
        flush_to_durable=durable_sink.flush_pending,
        dependency_sequences=lambda _runtime: tuple(dependency_sequences),
        manifest_store=EpisodeCaptureManifestStore(tmp_path),
        uuid_factory=lambda: episode_uuid,
        control_clock=clock,
        commit_clock=clock,
    )

    def admit_trace(fact) -> None:
        receipt = trace_runtime.issue(fact)
        if not receipt.accepted:
            raise RuntimeError(
                f"Formal sim trace rejected {fact.event_type}: "
                f"{receipt.rejection_code}"
            )

    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
            actuation_profile=actuation_profile,
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
            embodiment_instance_id=f"formal-sim-embodiment-{suffix}",
            reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
        ),
        trace_sink=admit_trace,
        clock=clock,
    )
    execution_manifest = build_m6_candidate_policy_arm_qpos_manifest(
        catalog=sim_runtime.catalog,
        capability=sim_runtime.execution_capability,
        activation=sim_runtime.reference_activation_spec,
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

    prepared = recorder.prepare_episode(legacy_episode_index=0, prepared_at=clock())
    recorder.bind_prepared_episode(prepared, bound_at=clock())
    try:
        sim_runtime.start()
        steps = []
        for index in range(int(action_count)):
            clock.value = 10.0 + index / 30.0
            sim_runtime.submit_policy_action(
                _action(0.001 * (index + 1), timestamp=clock()),
                policy_output_id=f"formal-sim-output-{index}",
            )
            step = sim_runtime.step(
                now=clock(),
                physics_ticks_after_dispatch=physics_ticks_after_dispatch,
            )
            assert step.status == "dispatched"
            steps.append(step)

        clock.value = (
            float(hold_at)
            if hold_at is not None
            else clock.value + max(0.2, 2.0 / 30.0)
        )
        sim_runtime.set_execution_gate(
            "hold",
            reason="formal_sim_episode_terminal_hold",
            source_event_ids=("formal-sim-stop",),
        )
        hold_step = sim_runtime.step(now=clock())
        assert hold_step.status in {"execution_hold_active", "not_due"}

        clock.value = 10.3
        released = recorder.release_episode(
            prepared,
            unbound_at=clock(),
            reason="formal_sim_fixture_finalize",
        )
        assert released.seal.trace_sealed
        assert released.seal.capture_commit_eligible
        if after_seal is not None:
            after_seal()
        finalized = recorder.legacy_finalize_completed(
            released,
            completed_at=10.4,
        )
        assert finalized.capture_commit_status == "committed"
    finally:
        sim_runtime.close()
        durable_sink.flush_pending()
        health.close()
        spool.close()

    captured = load_episode_control_trace(tmp_path, episode_uuid)
    assert captured.committed
    assert tuple(
        dependency.fact.fact_id for dependency in captured.dependency_records
    ) == (manifest_fact.fact_id,)
    assert captured.dependency_records[0].fact.payload[
        "semantic_content_hash"
    ] == execution_manifest.semantic_content_hash
    event_types = {record.fact.event_type for record in captured.episode_records}
    assert {
        "arbitration_decision",
        "shared_safety_result",
        "reference_segment",
        "reference_segment_admission",
        "group_generation_reservation",
        "reference_segment_terminal",
        "dispatch_receipt",
        "measurement_alignment",
    }.issubset(event_types)
    activation_types = {
        "group_generation_activation",
        "reference_target_activation",
        "planner_command_frame",
    }
    if expect_activation:
        assert activation_types.issubset(event_types)
    else:
        assert not activation_types.intersection(event_types)
    decisions = materialize_actual_decision_view(captured)
    assert len(decisions) == int(action_count)
    assert tuple(item.decision_id for item in decisions) == tuple(
        item.decision.decision_id for item in steps
    )
    return captured, execution_manifest, steps[-1]


def _publish_formal_sim_seal_then_wait_for_process_termination(
    root: str,
    ready,
    stop,
) -> None:
    def wait_after_seal() -> None:
        ready.set()
        stop.wait(30.0)

    _run_scheduled_qpos_formal_capture(
        Path(root),
        scene=DeterministicSceneFacade(),
        suffix="process-kill-after-seal",
        after_seal=wait_after_seal,
    )


def test_scheduled_qpos_runtime_seals_commits_and_rebuilds_formal_capture(
    tmp_path: Path,
) -> None:
    captured, _manifest, _step = _run_scheduled_qpos_formal_capture(
        tmp_path,
        scene=DeterministicSceneFacade(),
        suffix="deterministic",
    )
    assert captured.committed


def test_real_mujoco_actuated_backend_seals_and_rebuilds_same_formal_lineage(
    tmp_path: Path,
) -> None:
    """Test-side composition only; no production sim entrypoint is implied."""

    pytest.importorskip("mujoco")
    profile = rm65_bimanual_udp_feedback_profile()
    scene = MujocoScene.from_contract(
        SimulationContract.load("sim/configs/rm65_bimanual_physics_v1.json"),
        position_actuation_profile=profile,
    )
    captured, execution_manifest, step = _run_scheduled_qpos_formal_capture(
        tmp_path,
        scene=scene,
        suffix="real-mujoco-actuated",
        actuation_profile=profile,
    )

    assert captured.committed
    assert execution_manifest.actual_reference_frame_producer == "not_applicable"
    assert step.status == "dispatched"
    event_types = {record.fact.event_type for record in captured.episode_records}
    assert "planner_command_frame" in event_types
    assert "measurement_alignment" in event_types


def test_hold_before_segment_start_seals_without_inventing_activation_or_frames(
    tmp_path: Path,
) -> None:
    captured, _manifest, _step = _run_scheduled_qpos_formal_capture(
        tmp_path,
        scene=DeterministicSceneFacade(),
        suffix="hold-before-start",
        physics_ticks_after_dispatch=0,
        hold_at=10.01,
        expect_activation=False,
    )

    assert captured.committed
    terminal = next(
        record.fact
        for record in captured.episode_records
        if record.fact.event_type == "reference_segment_terminal"
    )
    assert terminal.payload["status"] == "canceled"
    assert terminal.payload["last_reference_frame_id"] is None


def test_bounded_successor_stress_seals_without_orphaned_segment_lineage(
    tmp_path: Path,
) -> None:
    # Keep this a bounded lineage fixture.  Long-run journal throughput and
    # memory acceptance are a separate target-machine benchmark, not a unit
    # test with quadratic full-history assertions on every admission.
    action_count = 12
    captured, _manifest, _step = _run_scheduled_qpos_formal_capture(
        tmp_path,
        scene=DeterministicSceneFacade(),
        suffix="bounded-successor-stress",
        physics_ticks_after_dispatch=40,
        hold_at=None,
        action_count=action_count,
    )

    segments = tuple(
        record.fact
        for record in captured.episode_records
        if record.fact.event_type == "reference_segment"
    )
    terminals = tuple(
        record.fact
        for record in captured.episode_records
        if record.fact.event_type == "reference_segment_terminal"
    )
    planner_frames = tuple(
        record.fact
        for record in captured.episode_records
        if record.fact.event_type == "planner_command_frame"
    )
    assert captured.committed
    assert len(segments) == action_count
    assert len(terminals) == action_count
    assert len(planner_frames) > action_count
    assert len(materialize_actual_decision_view(captured)) == action_count


def test_formal_scheduled_capture_kill_after_seal_remains_uncommitted_but_auditable(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    stop = context.Event()
    process = context.Process(
        target=_publish_formal_sim_seal_then_wait_for_process_termination,
        args=(str(tmp_path), ready, stop),
    )
    process.start()
    try:
        assert ready.wait(10.0)
        process.terminate()
        process.join(10.0)
        assert not process.is_alive()

        episode_uuid = "formal-sim-episode-process-kill-after-seal"
        manifests = EpisodeCaptureManifestStore(tmp_path)
        seal = manifests.read_seal_payload(episode_uuid)
        assert seal is not None
        assert seal["trace_sealed"] is True
        assert seal["capture_commit_eligible"] is True
        assert manifests.read_commit_payload(episode_uuid) is None

        with pytest.raises(
            ControlTraceReadError,
            match="sealed but has no capture commit",
        ):
            load_episode_control_trace(tmp_path, episode_uuid)

        audit_capture = load_episode_control_trace(
            tmp_path,
            episode_uuid,
            require_commit=False,
        )
        assert not audit_capture.committed
        assert len(materialize_actual_decision_view(audit_capture)) == 1

        spool = ControlTraceSpool(
            tmp_path
            / CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1
            / "formal-sim-trace-process-kill-after-seal"
            / "control_trace.wal",
            trace_session_id="formal-sim-trace-process-kill-after-seal",
            max_bytes=16 << 20,
            physical_preallocation=False,
        )
        try:
            assert spool.recovery_report.tail_status == "zero_filled"
            assert spool.durable_high_watermark >= int(
                seal["trace_range"]["last_journal_sequence"]
            )
        finally:
            spool.close()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10.0)
