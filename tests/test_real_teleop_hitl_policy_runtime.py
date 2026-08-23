from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_contracts.hitl import (
    ExclusiveHITLSampleLineage,
    exclusive_hitl_policy_artifact_contract_hash,
)
from robot_motion.action import BIMANUAL_QPOS_V1, RobotAction
from robot_motion.deploy import NormalizedAction, PolicyActionEmission
from robot_motion.deploy.runtime import SourceValidationResult
from teleop_runtime.real_teleop.hitl_policy_runtime import (
    ExclusiveHITLPolicyCommandSelector,
    build_exclusive_hitl_dataset_contract,
)
from teleop_runtime.schemas import EffectorCommand, TeleopCommand, TeleopObservation


def _human_command(timestamp: float = 10.0) -> TeleopCommand:
    command = TeleopCommand(timestamp=timestamp, metadata={"source": "human"})
    command.add_command(
        EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="joint_vector",
            value=np.zeros(6, dtype=np.float32),
        )
    )
    return command


def _emission(index: int, *, emitted_at: float) -> PolicyActionEmission:
    action = RobotAction(
        action_space=BIMANUAL_QPOS_V1,
        timestamp=emitted_at,
        arm_qpos_by_side={
            "right": np.full(6, float(index), dtype=np.float32),
        },
        active_effectors=("right_arm",),
        source="policy-fixture",
    )
    return PolicyActionEmission(
        emission_id=f"policy-emission:{index}",
        admission_mode="on_emission_pop_v1",
        policy_node_id="policy-fixture",
        inference_id=f"inference:{index}",
        policy_output_id=f"policy-output:{index}",
        chunk_id=index,
        chunk_keyframe_index=0,
        chunk_size=1,
        decoded_absolute_action=NormalizedAction(
            vector=np.asarray([float(index)], dtype=np.float32)
        ),
        pre_safety_robot_action=action,
        source_validation=SourceValidationResult(accepted=True),
        representation_receipt=None,
        pre_safety_faults=(),
        produced_at=emitted_at - 0.01,
        produced_clock_domain_id="policy.monotonic",
        emitted_at=emitted_at,
        target_effective_at=emitted_at + 0.01,
        effective_from=emitted_at,
        effective_until=emitted_at + 0.2,
        effective_clock_domain_id="policy.monotonic",
        schedule_spec_id="fixture-schedule",
        schedule_spec_hash="fixture-schedule-hash",
        queue_len_before=1,
        queue_len_after=0,
        queue_pop_count=index + 1,
        semantic_content_hash=f"policy-emission-hash:{index}",
    )


class _PolicySource:
    maximum_hold_sec = 0.2

    def __init__(self, emissions=()) -> None:
        self.emissions = list(emissions)
        self.started = False
        self.closed = False
        self.reset_count = 0
        self.last_error = None

    @property
    def ready(self) -> bool:
        return self.started and not self.closed

    def start(self) -> None:
        self.started = True

    def reset_for_activation(self) -> None:
        self.reset_count += 1

    def step(self, *, now=None, step_index=None):
        del now, step_index
        return None if not self.emissions else self.emissions.pop(0)

    def status(self, *, now=None):
        del now
        return {"ready": self.ready}

    def close(self) -> None:
        self.closed = True


def test_normal_dataset_contract_binds_artifact_and_scheduling_identity() -> None:
    artifact_payload = {
        "schema": "policy-artifact-fixture",
        "policy_io_layout_hash": "layout-hash:1",
    }
    artifact = SimpleNamespace(
        to_dict=lambda: artifact_payload,
        layout_hash="layout-hash:1",
        model_id="openpi-fixture",
        checkpoint_step=50176,
    )
    config = SimpleNamespace(
        policy_artifact=artifact,
        prompt="write text",
        action_hz=15.0,
        policy_hz=10.0,
        action_chunk_execution_steps=8,
        prediction_request_queue_threshold=4,
    )
    args = SimpleNamespace(
        active_sides=("right",),
        hitl_policy_mode="normal",
        hitl_initial_source="policy",
        hitl_takeover_key="u",
        hitl_policy_resume_key="i",
        _hitl_policy_emission_source=None,
    )

    contract = build_exclusive_hitl_dataset_contract(
        args,
        normal_source_config=config,
    )

    assert contract.policy_source_kind == "openpi_artifact"
    assert contract.policy_artifact_contract_hash == (
        exclusive_hitl_policy_artifact_contract_hash(artifact_payload)
    )
    assert contract.policy_io_layout_hash == "layout-hash:1"
    assert contract.checkpoint_step == 50176
    assert contract.action_chunk_execution_steps == 8


def test_normal_policy_command_is_identity_composed_before_common_command_path() -> None:
    source = _PolicySource([_emission(1, emitted_at=10.0)])
    selector = ExclusiveHITLPolicyCommandSelector(
        policy_mode="normal",
        initial_source="policy",
        normal_policy_source=source,
        monotonic_clock=lambda: 10.0,
    )
    selector.start()
    try:
        result = selector(
            TeleopObservation(timestamp=100.0),
            _human_command(timestamp=100.0),
            100.0,
        )
    finally:
        selector.close()

    assert result is not None and result.command is not None
    command = result.command
    np.testing.assert_array_equal(
        command.commands["right_arm"].value,
        np.ones(6, dtype=np.float32),
    )
    assert command.timestamp == 100.0
    assert command.metadata["control_source"] == "policy"
    assert command.metadata["policy_emission_id"] == "policy-emission:1"
    assert command.metadata["hitl_composition_evaluation_id"].startswith(
        "composition-evaluation:"
    )
    lineage = ExclusiveHITLSampleLineage.from_command_metadata(command.metadata)
    assert lineage is not None
    assert lineage.source == "policy"
    assert lineage.policy_emission_id == "policy-emission:1"
    assert lineage.policy_lineage_complete is True


def test_human_selection_preserves_human_candidate_and_adds_lineage() -> None:
    selector = ExclusiveHITLPolicyCommandSelector(
        policy_mode="normal",
        initial_source="human",
        normal_policy_source=_PolicySource(),
    )
    selector.start()
    human = _human_command()
    try:
        result = selector(TeleopObservation(timestamp=10.0), human, 10.0)
    finally:
        selector.close()

    assert result is None
    assert human.metadata["control_source"] == "human"
    assert human.metadata["hitl_exclusive_source"] == "human"


def test_policy_return_discards_pre_reset_emission_and_waits_for_fresh_one() -> None:
    source = _PolicySource(
        [
            _emission(1, emitted_at=19.9),
            _emission(2, emitted_at=20.1),
        ]
    )
    selector = ExclusiveHITLPolicyCommandSelector(
        policy_mode="normal",
        initial_source="human",
        normal_policy_source=source,
        monotonic_clock=lambda: 20.0,
    )
    selector.start()
    selector.reset_policy_for_activation(
        barrier_at=20.0,
        switch_id="switch:1",
        intent_id="intent:1",
        intent_hash="intent-hash:1",
    )
    assert source.reset_count == 1

    first = selector(TeleopObservation(timestamp=100.0), _human_command(), 100.0)
    assert first is not None and first.command is None
    assert selector.fresh_policy_emission_ready is False

    second = selector(TeleopObservation(timestamp=100.1), _human_command(), 100.1)
    assert second is not None and second.command is None
    assert selector.fresh_policy_emission_ready is True
    selector.commit_policy_selection()

    selected = selector(
        TeleopObservation(timestamp=100.2),
        _human_command(timestamp=100.2),
        100.2,
    )
    selector.close()
    assert selected is not None and selected.command is not None
    assert selected.command.metadata["policy_emission_id"] == "policy-emission:2"
    assert selected.command.metadata["hitl_switch_id"] == "switch:1"


def test_rtc_committed_emission_consumer_is_explicitly_no_action() -> None:
    inbox: queue.Queue[PolicyActionEmission] = queue.Queue(maxsize=2)
    inbox.put_nowait(_emission(3, emitted_at=30.0))
    selector = ExclusiveHITLPolicyCommandSelector(
        policy_mode="rtc",
        initial_source="policy",
        rtc_committed_emissions=inbox,
        monotonic_clock=lambda: 30.0,
    )
    selector.start()
    result = selector(TeleopObservation(timestamp=300.0), _human_command(), 300.0)

    assert result is not None and result.command is None
    assert selector.status().last_emission_id == "policy-emission:3"
    assert selector.status().rtc_no_action is True

    selector.select_human(switch_id="switch:rtc", intent_id=None, intent_hash=None)
    human = _human_command()
    assert selector(TeleopObservation(timestamp=301.0), human, 301.0) is None
    assert human.metadata["hitl_rtc_no_action"] is True


def test_rtc_policy_return_cannot_be_upgraded_to_motion() -> None:
    inbox: queue.Queue[PolicyActionEmission] = queue.Queue(maxsize=2)
    selector = ExclusiveHITLPolicyCommandSelector(
        policy_mode="rtc",
        initial_source="human",
        rtc_committed_emissions=inbox,
        monotonic_clock=lambda: 40.0,
    )
    selector.start()
    selector.reset_policy_for_activation(
        barrier_at=40.0,
        switch_id="switch:rtc",
        intent_id="intent:rtc",
        intent_hash="intent-hash:rtc",
    )
    inbox.put_nowait(_emission(4, emitted_at=40.1))
    selector(TeleopObservation(timestamp=400.0), _human_command(), 400.0)
    assert selector.fresh_policy_emission_ready is True
    with pytest.raises(RuntimeError, match="no-action"):
        selector.commit_policy_selection()


def test_normal_policy_source_failure_is_not_masked_as_an_empty_queue() -> None:
    source = _PolicySource()
    source.last_error = "model transport unavailable"
    selector = ExclusiveHITLPolicyCommandSelector(
        policy_mode="normal",
        initial_source="policy",
        normal_policy_source=source,
    )
    selector.start()

    with pytest.raises(RuntimeError, match="model transport unavailable"):
        selector(TeleopObservation(timestamp=500.0), _human_command(), 500.0)
