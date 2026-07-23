from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from control_station import StationIntentDraft
from control_station.composition import (
    CS2B_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1,
    ReplaySourceObserverBinding,
    build_scoped_cs2b_mujoco_replay_station,
)
from control_station.sources import ReplayGuiStationConsumer
from dataset_contracts.replay import (
    REPLAY_ACTION_STAGE_SELECTED,
    REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
    REPLAY_BINDING_VERIFIED_COMPATIBLE,
    REPLAY_BINDING_VERIFIED_INCOMPATIBLE,
    REPLAY_MODE_CANONICAL_ACTION,
    REPLAY_SCHEDULE_FIXED_RATE,
    REPLAY_SOURCE_INTERPRETATION_VERIFIED,
    ReplayScheduleSpec,
    ReplaySourceActionContract,
    ReplaySourceContract,
)
from robot_motion.action import RobotAction
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


def _runtime(clock: ManualClock) -> tuple[MujocoDeployRuntime, FakeScene]:
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
    return (
        MujocoDeployRuntime(
            rig,
            MujocoDeployConfig(
                runtime_session_id="cs2b-runtime",
                control_domain_id="cs2b-mujoco-domain",
                embodiment_instance_id="cs2b-mujoco-rm65",
                producer_id="unused-policy-source",
                replay_producer_id="replay-main",
                decision_hz=30.0,
                authority_policy_id="mujoco-single-replay-authority-v1",
                initial_control_mode="replay",
                enable_replay_authority=True,
                clock_domain_id="local_monotonic",
            ),
            clock=clock,
        ),
        scene,
    )


def _source_contracts(
    runtime: MujocoDeployRuntime,
    *,
    action_space: str = "bimanual_qpos_v1",
):
    pose = action_space == "bimanual_ee_pose_v1"
    action_contract = ReplaySourceActionContract(
        contract_id=f"cs2b-source-action:{action_space}",
        interpretation_status=REPLAY_SOURCE_INTERPRETATION_VERIFIED,
        action_stage=REPLAY_ACTION_STAGE_SELECTED,
        action_space=action_space,
        component_ids=("left_arm", "right_arm"),
        selection_group_by_component={
            "left_arm": "arms",
            "right_arm": "arms",
        },
        coupled_selection_groups=("arms",),
        pose_frame_contract_hash=("rm65-pose-frame-v1" if pose else None),
        component_schema_hash=runtime.catalog.semantic_content_hash,
        action_spec_hash_by_component={
            "left_arm": (
                "rm65-tool-tcp-pose-v1"
                if pose
                else "rm65-arm-qpos6-joint-order-v1"
            ),
            "right_arm": (
                "rm65-tool-tcp-pose-v1"
                if pose
                else "rm65-arm-qpos6-joint-order-v1"
            ),
        },
        units_convention_id=("si_m_rad_xyzw_v1" if pose else "si_rad_joint_order_v1"),
    )
    source_contract = ReplaySourceContract(
        contract_id=f"cs2b-source:{action_space}",
        replay_mode=REPLAY_MODE_CANONICAL_ACTION,
        dataset_schema="dexteleop_dataset_v2.4",
        source_dataset_id="cs2b-golden-source",
        episode_id="episode-replay-0",
        action_stage=REPLAY_ACTION_STAGE_SELECTED,
        action_field="decision.selected_action",
        source_action_space=action_space,
        source_frame_contract_hash=action_contract.pose_frame_contract_hash,
        source_component_schema_hash=runtime.catalog.semantic_content_hash,
    )
    return source_contract, action_contract


def _replay_source(runtime: MujocoDeployRuntime) -> ReplayProposalSource:
    source_contract, action_contract = _source_contracts(runtime)
    binding, assessment = build_replay_binding_assessment(
        action_contract,
        runtime.target_runtime_action_contract,
        binding_id="cs2b-qpos-source-to-mujoco",
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
                schedule_id="cs2b-replay-30hz-v1",
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
        source_slot_id="replay-main",
        source_locator_id="cs2b-golden-source",
        expected_source_path=str(path),
        source_episode_index=0,
        group_key="replay-group-0",
        source_start_index=10,
        clock_domain_id="local_monotonic",
    )


def _build_station(tmp_path: Path, clock: ManualClock):
    runtime, scene = _runtime(clock)
    source_path = tmp_path / "episode.parquet"
    source = _replay_source(runtime)
    station = build_scoped_cs2b_mujoco_replay_station(
        runtime,
        source,
        _observer_binding(source_path),
        station_clock=clock,
    )
    return station, source_path, scene


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
            reason="cs2b replay fixture",
        )
    )
    if transition.receipt.status == "operation_requested":
        return transition, station.complete_pending_operation(transition)
    return transition, None


def _advance_to_authority(station) -> None:
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
        transition, execution = _submit(
            station,
            sequence=sequence,
            kind=kind,
            subject_id=subject_id,
        )
        assert transition.receipt.accepted
        if execution is not None:
            assert execution.completion_transition.accepted


def _emission(path: Path, clock: ManualClock):
    action = RobotAction.bimanual_qpos(
        left_arm=np.full(6, 0.001, dtype=np.float32),
        right_arm=np.full(6, -0.001, dtype=np.float32),
        timestamp=clock(),
        source="existing_replay_cursor",
    )
    return create_replay_source_frame_emission(
        source_path=str(path),
        source_episode_index=0,
        group_key="replay-group-0",
        local_frame_index=0,
        source_action_index=10,
        source_relative_time=0.0,
        action=action,
        observed_at=clock(),
        target_effective_at=clock(),
        effective_until=clock() + 0.1,
        clock_domain_id="local_monotonic",
    )


def test_cs2b_qpos_manifest_preserves_existing_emission_boundary(tmp_path: Path) -> None:
    station, _path, _scene = _build_station(tmp_path, ManualClock())
    manifest = station.station.manifest
    assert manifest.manifest_id == CS2B_MUJOCO_REPLAY_STATION_MANIFEST_ID_V1
    assert manifest.source_slots[0].expected_emission_seam.endswith(
        "ReplaySourceFrameEmission"
    )
    assert "control_station_replay_cursor" in (
        manifest.source_slots[0].explicit_exclusions
    )
    assert "source_side_pose_to_qpos_conversion" in (
        manifest.source_slots[0].explicit_exclusions
    )
    assert "legacy_gui_pose_replay_authority" in manifest.explicit_exclusions
    receipt = station.data_plane.binding_receipt
    assert receipt.source_emission_identity_preserved
    assert not receipt.cursor_owned_by_control_station
    assert not receipt.resampler_owned_by_control_station
    station.close()


def test_cs2b_existing_qpos_emission_requires_station_authority_and_dispatches(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    station, path, scene = _build_station(tmp_path, clock)
    emission = _emission(path, clock)
    with pytest.raises(RuntimeError, match="authority is not active"):
        station.submit_source_frame_emission(emission)

    _advance_to_authority(station)
    status = station.authority_status_payload()
    assert status["motion_authorized"] is True
    assert status["publication_enabled"] is True
    assert status["authoritative"] is False

    submission = station.submit_source_frame_emission(emission)
    assert submission.source_emission is emission
    assert submission.proposal_admission is not None
    assert submission.proposal_admission.accepted
    assert submission.authorization_admission is not None
    assert submission.authorization_admission.accepted
    step = station.step(now=clock(), physics_ticks_after_dispatch=1)
    assert step.status == "dispatched"
    assert scene.physics_commands
    source_facts = tuple(
        fact
        for fact in station.data_plane.runtime.trace_facts
        if fact.event_type == "replay_source_frame_emission"
    )
    assert len(source_facts) == 1
    assert source_facts[0].fact_id == emission.emission_id

    release, execution = _submit(
        station,
        sequence=7,
        kind="release_motion_authority",
        subject_id=station.source_adapter.subject_id,
    )
    assert release.receipt.accepted
    assert execution is not None
    assert not station.publication_barrier.snapshot().enabled
    with pytest.raises(RuntimeError, match="authority is not active"):
        station.submit_source_frame_emission(emission)
    station.close()


def test_gui_replay_consumer_feeds_the_existing_station_publication_barrier(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    station, path, scene = _build_station(tmp_path, clock)
    _advance_to_authority(station)
    emission = _emission(path, clock)
    consumer = ReplayGuiStationConsumer(
        station.submit_source_frame_emission,
        queue_capacity=4,
    )
    consumer.start()
    try:
        consumer.emission_queue.put_nowait(emission)
        consumer.emission_queue.join()
        status = consumer.status()
        assert status.submitted_count == 1
        assert status.rejected_count == 0
        step = station.step(now=clock(), physics_ticks_after_dispatch=1)
        assert step.status == "dispatched"
        assert scene.physics_commands
        source_fact = next(
            fact
            for fact in station.data_plane.runtime.trace_facts
            if fact.event_type == "replay_source_frame_emission"
        )
        assert source_fact.fact_id == emission.emission_id
    finally:
        consumer.close()
        station.close()


def test_cs2b_observer_binding_rejects_another_cursor_identity(tmp_path: Path) -> None:
    clock = ManualClock()
    station, path, _scene = _build_station(tmp_path, clock)
    _advance_to_authority(station)
    wrong = create_replay_source_frame_emission(
        source_path=str(path),
        source_episode_index=0,
        group_key="replay-group-0",
        local_frame_index=1,
        source_action_index=10,
        source_relative_time=0.0,
        action=RobotAction.bimanual_qpos(
            left_arm=np.zeros(6),
            right_arm=np.zeros(6),
            timestamp=clock(),
            source="wrong_cursor",
        ),
        observed_at=clock(),
        target_effective_at=clock(),
        effective_until=clock() + 0.1,
        clock_domain_id="local_monotonic",
    )
    with pytest.raises(ValueError, match="source/local indices"):
        station.submit_source_frame_emission(wrong)
    assert not any(
        fact.fact_id == wrong.emission_id
        for fact in station.data_plane.runtime.trace_facts
    )
    station.close()


def test_legacy_pose_replay_is_not_claimed_compatible_with_qpos_runtime(
    tmp_path: Path,
) -> None:
    clock = ManualClock()
    runtime, _scene = _runtime(clock)
    source_contract, pose_contract = _source_contracts(
        runtime,
        action_space="bimanual_ee_pose_v1",
    )
    binding, assessment = build_replay_binding_assessment(
        pose_contract,
        runtime.target_runtime_action_contract,
        binding_id="legacy-pose-replay-to-qpos-only-mujoco",
    )
    assert assessment.verdict == REPLAY_BINDING_VERIFIED_INCOMPATIBLE
    source = ReplayProposalSource(
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
            source_action_contract=pose_contract,
            target_runtime_contract=runtime.target_runtime_action_contract,
            derivation_receipt=None,
            schedule=ReplayScheduleSpec(
                schedule_id="legacy-pose-15hz-v1",
                schedule_mode=REPLAY_SCHEDULE_FIXED_RATE,
                source_hz=15.0,
                target_hz=None,
                resampling_policy="none",
                source_start_index=10,
            ),
            target_binding_spec=binding,
            binding_assessment=assessment,
            requested_use=REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
        )
    )
    with pytest.raises(ValueError, match="verified-compatible"):
        build_scoped_cs2b_mujoco_replay_station(
            runtime,
            source,
            _observer_binding(tmp_path / "episode.parquet"),
            station_clock=clock,
        )
    assert not runtime._started
