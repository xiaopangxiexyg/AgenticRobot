from __future__ import annotations

import numpy as np
import pytest

from control_station import (
    BoundedStationEventLog,
    SideEffectStationOperationCoordinator,
    StationIntentDraft,
    project_station_intent_transition,
    project_station_operation_transition,
    project_station_publication_barrier_transition,
    project_station_run_manifest,
)
from control_station.composition import (
    CS2A2B_MUJOCO_HUMAN_STATION_MANIFEST_ID_V1,
    build_scoped_cs2a2b_mujoco_human_station,
)
from control_station.sources import (
    SyntheticHumanQposSource,
    SyntheticHumanQposSourceSpec,
)
from sim.contracts import TimingSpec
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.deploy import MujocoDeployConfig, MujocoDeployRuntime
from control_station.web import (
    LocalIntentAccessPolicy,
    build_coordinated_control_station_server,
)


class ManualClock:
    def __init__(self, value: float = 20.0) -> None:
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


def _build_station(clock: ManualClock):
    scene = FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=TimingSpec(
                sample_hz=15.0,
                policy_hz=15.0,
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
            runtime_session_id="cs2a2b-runtime",
            control_domain_id="cs2a2b-mujoco-domain",
            embodiment_instance_id="cs2a2b-mujoco-rm65",
            producer_id="synthetic-human-main",
            decision_hz=30.0,
            authority_policy_id="mujoco-single-human-authority-v1",
            initial_control_mode="human",
        ),
        clock=clock,
    )
    source_spec = SyntheticHumanQposSourceSpec()
    station = build_scoped_cs2a2b_mujoco_human_station(
        runtime,
        source_spec,
        station_clock=clock,
    )
    return station, SyntheticHumanQposSource(source_spec), scene, clock


def _submit(station, *, sequence: int, kind: str, subject_id: str | None = None):
    state = station.station.state
    transition = station.station.submit_intent(
        StationIntentDraft(
            client_id="operator:test",
            client_sequence=sequence,
            intent_kind=kind,
            produced_at=station.data_plane.runtime.clock(),
            produced_clock_domain_id="operator-test-monotonic",
            expected_state_revision=state.state_revision,
            manifest_hash=station.station.manifest.semantic_content_hash,
            subject_id=subject_id,
            reason="cs2a2b fixture",
        )
    )
    if transition.receipt.status == "operation_requested":
        return transition, station.complete_pending_operation(transition)
    return transition, None


def _advance_to_authority(station) -> None:
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
    for sequence, (kind, subject_id) in enumerate(operations):
        transition, execution = _submit(
            station,
            sequence=sequence,
            kind=kind,
            subject_id=subject_id,
        )
        assert transition.receipt.accepted
        if execution is not None:
            assert execution.completion_transition.accepted


def test_cs2a2b_manifest_and_binding_are_exactly_station_scoped() -> None:
    station, _source, _scene, _clock = _build_station(ManualClock())
    manifest = station.station.manifest
    assert manifest.manifest_id == CS2A2B_MUJOCO_HUMAN_STATION_MANIFEST_ID_V1
    assert manifest.source_slots[0].supported_uses == ("authority", "candidate")
    assert "request_motion_authority" in manifest.allowed_intent_kinds
    assert "station_motion_authority_lifecycle" not in manifest.explicit_exclusions
    assert "production_entrypoint" in manifest.explicit_exclusions
    assert station.data_plane.binding_receipt.station_authority_lifecycle_bound
    station.close()


def test_cs2a2b_station_facts_have_explicit_control_trace_projection() -> None:
    station, _source, _scene, clock = _build_station(ManualClock())
    manifest = station.station.manifest
    embodiment_id = manifest.execution_binding.embodiment_instance_id
    manifest_fact = project_station_run_manifest(
        manifest,
        runtime_session_id=station.station.state.runtime_session_id,
        occurred_at=clock(),
        clock_domain_id="control-station-test-monotonic",
    )
    assert manifest_fact.event_type == "station_run_manifest"
    assert manifest_fact.payload["semantic_content_hash"] == (
        manifest.semantic_content_hash
    )
    assert manifest_fact.payload["source_slots"][0]["supported_uses"] == (
        "authority",
        "candidate",
    )

    target_id = manifest.execution_binding.binding_id
    source_id = station.source_adapter.subject_id
    intent_facts = []
    operation_facts = []
    authority_execution = None
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
        transition, execution = _submit(
            station,
            sequence=sequence,
            kind=kind,
            subject_id=subject_id,
        )
        intent_facts.append(
            project_station_intent_transition(
                transition,
                embodiment_instance_id=embodiment_id,
            )
        )
        if execution is not None:
            operation_facts.append(
                project_station_operation_transition(
                    execution.completion_transition,
                    embodiment_instance_id=embodiment_id,
                )
            )
        if kind == "request_motion_authority":
            authority_execution = execution

    assert authority_execution is not None
    authority_result = authority_execution.completion_transition.result
    barrier_fact = project_station_publication_barrier_transition(
        station.publication_barrier.snapshot(),
        runtime_session_id=station.station.state.runtime_session_id,
        control_domain_id=station.station.state.control_domain_id,
        embodiment_instance_id=embodiment_id,
        occurred_at=authority_result.available_at,
        clock_domain_id=authority_result.available_clock_domain_id,
        causal_station_fact_id=authority_result.operation_result_id,
    )

    assert len(intent_facts) == 7
    assert len(operation_facts) == 6
    assert intent_facts[-1].payload["status"] == "operation_requested"
    assert operation_facts[-1].payload["transition_status"] == (
        "operation_succeeded"
    )
    assert operation_facts[-1].payload["result_status"] == (
        "existing_single_source_authority_bound"
    )
    assert operation_facts[-1].payload["source_intent_id"] == (
        intent_facts[-1].fact_id
    )
    assert barrier_fact.payload["enabled"] is True
    assert barrier_fact.payload["authority_operation_result_id"] == (
        operation_facts[-1].fact_id
    )
    assert barrier_fact.payload["causal_station_fact_id"] == (
        operation_facts[-1].fact_id
    )
    assert barrier_fact.payload["snapshot_semantic_hash"] == (
        station.publication_barrier.snapshot().semantic_content_hash
    )
    station.close()


def test_station_completion_not_intent_opens_human_publication() -> None:
    station, source, _scene, clock = _build_station(ManualClock())
    emission = source.emit(
        {"left_arm": np.zeros(6), "right_arm": np.zeros(6)},
        emitted_at=clock(),
    )
    with pytest.raises(RuntimeError, match="authority is not active"):
        station.submit_emission(
            emission,
            effective_from=clock(),
            effective_until=clock() + 0.1,
            target_effective_at=clock(),
        )

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
        )
    ):
        _submit(station, sequence=sequence, kind=kind, subject_id=subject_id)

    request = station.station.submit_intent(
        StationIntentDraft(
            client_id="operator:test",
            client_sequence=6,
            intent_kind="request_motion_authority",
            produced_at=clock(),
            produced_clock_domain_id="operator-test-monotonic",
            expected_state_revision=station.station.state.state_revision,
            manifest_hash=station.station.manifest.semantic_content_hash,
            subject_id=source_id,
        )
    )
    assert request.receipt.status == "operation_requested"
    assert station.station.state.source_state_by_slot[source_id] == "candidate"
    with pytest.raises(RuntimeError, match="authority is not active"):
        station.submit_emission(
            emission,
            effective_from=clock(),
            effective_until=clock() + 0.1,
            target_effective_at=clock(),
        )

    completion = station.complete_pending_operation(request)
    assert completion.completion_transition.accepted
    assert station.station.state.lifecycle_state == "running"
    assert station.publication_barrier.snapshot().enabled
    station.close()


def test_cs2a2b_authority_uses_existing_core_then_release_holds() -> None:
    station, source, scene, clock = _build_station(ManualClock())
    owners = station.data_plane.runtime.core.dispatch_owners
    execution_epochs = station.data_plane.runtime.core.execution_epochs
    owner_token_before = owners.snapshot().current_token
    assert owner_token_before is not None
    epoch_before = execution_epochs.snapshot().execution_epoch

    _advance_to_authority(station)
    assert station.data_plane.runtime.core.dispatch_owners is owners
    assert station.data_plane.runtime.core.execution_epochs is execution_epochs
    assert owners.snapshot().current_token is owner_token_before

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
    step = station.step(now=clock(), physics_ticks_after_dispatch=1)
    assert step.status == "dispatched"
    assert len(scene.physics_commands) == 1

    source_id = station.source_adapter.subject_id
    release, execution = _submit(
        station,
        sequence=7,
        kind="release_motion_authority",
        subject_id=source_id,
    )
    assert release.receipt.status == "operation_requested"
    assert execution is not None and execution.completion_transition.accepted
    assert station.station.state.lifecycle_state == "hold"
    assert station.station.state.source_state_by_slot[source_id] == "candidate"
    assert not station.publication_barrier.snapshot().enabled
    assert execution_epochs.snapshot().execution_epoch == epoch_before + 1
    assert owners.snapshot().current_token is owner_token_before

    evidence = station.source_adapter.evidence_by_intent_id[release.intent.intent_id]
    assert evidence.execution_gate_status == "hold"
    assert evidence.evidence_status == "station_publication_closed_and_execution_held"
    assert evidence.lower_event_ids
    with pytest.raises(RuntimeError, match="authority is not active"):
        station.submit_emission(
            emission,
            effective_from=clock(),
            effective_until=clock() + 0.1,
            target_effective_at=clock(),
        )
    event_types = tuple(
        fact.event_type for fact in station.data_plane.runtime.trace_facts
    )
    assert "execution_gate_event" in event_types
    assert "execution_cancellation" in event_types
    assert "policy_output" not in event_types
    assert "replay_source_frame_emission" not in event_types
    station.close()


def test_cs2a2b_authority_fails_closed_when_gate_not_ready() -> None:
    station, _source, _scene, _clock = _build_station(ManualClock())
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
        )
    ):
        _submit(station, sequence=sequence, kind=kind, subject_id=subject_id)
    station.data_plane.runtime.set_execution_gate(
        "hold",
        reason="test_pre_authority_hold",
    )
    request = station.station.submit_intent(
        StationIntentDraft(
            client_id="operator:test",
            client_sequence=6,
            intent_kind="request_motion_authority",
            produced_at=station.data_plane.runtime.clock(),
            produced_clock_domain_id="operator-test-monotonic",
            expected_state_revision=station.station.state.state_revision,
            manifest_hash=station.station.manifest.semantic_content_hash,
            subject_id=source_id,
        )
    )
    execution = station.complete_pending_operation(request)
    assert execution.completion_transition.accepted
    assert execution.completion_transition.status == (
        "operation_failed:execution_gate_not_ready:hold"
    )
    assert station.station.state.source_state_by_slot[source_id] == "candidate"
    assert not station.publication_barrier.snapshot().enabled
    station.close()


def test_cs2a2b_side_effect_coordinator_closes_release_before_enqueue() -> None:
    station, source, scene, clock = _build_station(ManualClock())
    coordinator = SideEffectStationOperationCoordinator(
        coordinator_id="cs2a2b-side-effect-worker",
        runtime=station.station,
        adapters=(station.target_adapter, station.source_adapter),
        operation_prepare=station.prepare_pending_operation,
        operation_executor=station.complete_pending_operation,
        queue_capacity=2,
        clock=clock,
    )
    coordinator.start()
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
        state = station.station.state
        submission = coordinator.submit_intent(
            StationIntentDraft(
                client_id="operator:test",
                client_sequence=sequence,
                intent_kind=kind,
                produced_at=clock(),
                produced_clock_domain_id="operator-test-monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=station.station.manifest.semantic_content_hash,
                subject_id=subject_id,
            )
        )
        assert submission.transition.receipt.accepted
        assert coordinator.wait_until_idle(timeout_sec=1.0)
    assert station.station.state.lifecycle_state == "running"
    assert station.publication_barrier.snapshot().enabled

    emission = source.emit(
        {
            "left_arm": np.full(6, 0.001, dtype=float),
            "right_arm": np.full(6, -0.001, dtype=float),
        },
        emitted_at=clock(),
    )
    assert station.submit_emission(
        emission,
        effective_from=clock(),
        effective_until=clock() + 0.1,
        target_effective_at=clock(),
    ).accepted
    assert station.step(now=clock(), physics_ticks_after_dispatch=1).status == (
        "dispatched"
    )
    assert scene.physics_commands

    state = station.station.state
    release = coordinator.submit_intent(
        StationIntentDraft(
            client_id="operator:test",
            client_sequence=7,
            intent_kind="release_motion_authority",
            produced_at=clock(),
            produced_clock_domain_id="operator-test-monotonic",
            expected_state_revision=state.state_revision,
            manifest_hash=station.station.manifest.semantic_content_hash,
            subject_id=source_id,
        )
    )
    assert release.transition.receipt.status == "operation_requested"
    # The prepare hook runs before queue publication, so this assertion does
    # not depend on whether the worker has already completed lower HOLD.
    assert not station.publication_barrier.snapshot().enabled
    assert coordinator.wait_until_idle(timeout_sec=1.0)
    assert station.station.state.lifecycle_state == "hold"
    assert coordinator.stop(timeout_sec=1.0)
    station.close()


def test_side_effect_prepare_failure_resolves_pending_and_web_rejects_it() -> None:
    station, _source, _scene, clock = _build_station(ManualClock())
    event_log = BoundedStationEventLog(clock=clock)

    def fail_prepare(_transition) -> None:
        raise RuntimeError("configured prepare failure")

    coordinator = SideEffectStationOperationCoordinator(
        coordinator_id="cs2a2b-prepare-failure",
        runtime=station.station,
        adapters=(station.target_adapter, station.source_adapter),
        operation_prepare=fail_prepare,
        operation_executor=station.complete_pending_operation,
        event_log=event_log,
        queue_capacity=1,
        clock=clock,
    )
    with pytest.raises(TypeError, match="NoIoStationOperationCoordinator"):
        build_coordinated_control_station_server(
            station.station,
            access_policy=LocalIntentAccessPolicy(
                access_policy_id="cs1c-no-side-effects",
                bearer_token="a" * 32,
            ),
            event_log=event_log,
            operation_coordinator=coordinator,
        )

    coordinator.start()
    lease = coordinator.submit_intent(
        StationIntentDraft(
            client_id="operator:test",
            client_sequence=0,
            intent_kind="acquire_control_lease",
            produced_at=clock(),
            produced_clock_domain_id="operator-test-monotonic",
            expected_state_revision=station.station.state.state_revision,
            manifest_hash=station.station.manifest.semantic_content_hash,
        )
    )
    assert lease.transition.receipt.accepted
    target_id = station.station.manifest.execution_binding.binding_id
    failed = coordinator.submit_intent(
        StationIntentDraft(
            client_id="operator:test",
            client_sequence=1,
            intent_kind="connect_target",
            produced_at=clock(),
            produced_clock_domain_id="operator-test-monotonic",
            expected_state_revision=station.station.state.state_revision,
            manifest_hash=station.station.manifest.semantic_content_hash,
            subject_id=target_id,
        )
    )
    assert failed.transition.receipt.status == "operation_requested"
    assert failed.enqueue_receipt.status == "operation_prepare_failed"
    assert coordinator.reserved_work_count == 0
    assert station.station.state.target_state == "configured"
    assert not station.station.state.pending_operation_by_subject
    assert coordinator.completions[-1].completion_transition.status == (
        "operation_failed:operation_prepare_exception:RuntimeError"
    )
    assert coordinator.stop(timeout_sec=1.0)
    station.close()
