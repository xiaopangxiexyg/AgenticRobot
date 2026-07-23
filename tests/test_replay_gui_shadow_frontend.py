from __future__ import annotations

from pathlib import Path

import numpy as np

from dataset_contracts.replay import REPLAY_BINDING_INDETERMINATE
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from dataset_interface.control_trace import (
    EpisodeControlTrace,
    materialize_shadow_evaluation_view,
)
from robot_motion.deploy import ACTION_ARM_POSE_SLICES, DEPLOY_ACTION_DIM, nan_vector
from robot_motion.replay import ReplayTrajectoryGroup, build_replay_source_frame_emission
from teleop_runtime.action.replay_shadow_frontend import (
    ReplayGuiShadowBinding,
    ReplayGuiShadowConsumer,
    ReplayGuiShadowProfile,
)
from teleop_runtime.replay.session import ReplayGuiSession


def _group(path: Path) -> ReplayTrajectoryGroup:
    action = nan_vector(DEPLOY_ACTION_DIM)
    action[ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    return ReplayTrajectoryGroup(
        key="g",
        label="fixture",
        source_path=str(path),
        source_episode_index=1,
        start_index=0,
        end_index=1,
        frame_count=1,
        action_hz=15.0,
        actions=np.asarray([action], dtype=np.float32),
        metadata={
            "source_metadata": {
                "dataset_info": {
                    "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
                    "pose_frame_contract_hash": "source-frame",
                    "component_schema_hash": "source-components",
                }
            }
        },
    )


def _binding(path: Path, *, source_frame: str | None = "source-frame"):
    metadata = {
        "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
        "component_schema_hash": "source-components",
    }
    if source_frame is not None:
        metadata["pose_frame_contract_hash"] = source_frame
    return ReplayGuiShadowBinding(
        source_path=str(path),
        episode_id="episode_000001",
        action_field="action",
        source_hz=15.0,
        dataset_metadata=metadata,
    )


def test_gui_consumer_conditionally_evaluates_same_source_emission(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "episode.parquet"
    consumer = ReplayGuiShadowConsumer(
        profile=ReplayGuiShadowProfile(
            queue_capacity=8,
            trace_capacity=256,
            trace_root=tmp_path / "trace",
        )
    )
    try:
        consumer.bind(_binding(source_path))
        emission = build_replay_source_frame_emission(
            _group(source_path),
            active_sides=("right",),
            index=0,
            include_arms=True,
            include_end_effectors=False,
            observed_at=10.0,
            schedule_start_at=10.0,
        )
        consumer.emission_queue.put_nowait(emission)
        consumer.emission_queue.join()

        status = consumer.status()
        assert status["authoritative"] is False
        assert status["dispatch_owner_token"] is False
        assert status["binding_verdict"] == REPLAY_BINDING_INDETERMINATE
        assert status["processing_error_count"] == 0
        facts = consumer.trace_facts_for_test()
        event_types = {fact.event_type for fact in facts}
        assert "replay_source_frame_emission" in event_types
        assert "replay_binding_assessment" in event_types
        assert "candidate_binding_eligibility" in event_types
        assert "arbitration_evaluation" in event_types
        assert "arbitration_decision" not in event_types
        assert "dispatch_owner" not in event_types
        eligibility = next(
            fact for fact in facts if fact.event_type == "candidate_binding_eligibility"
        )
        assert eligibility.payload["evaluation_allowed"] is True
        assert eligibility.payload["conditional_evaluation"] is True
        assert eligibility.payload["authoritative_allowed"] is False
        trace_session_id = str(status["trace_session_id"])
        logical_trace = EpisodeControlTrace(
            dataset_root=tmp_path,
            episode_uuid="conditional-replay-shadow",
            trace_session_id=trace_session_id,
            profile_id="replay_shadow_dry_run_v1",
            committed=False,
            capabilities=(),
            seal_payload={},
            commit_payload=None,
            episode_records=consumer.trace_records_for_test(),
            dependency_records=(),
            wal_relative_path="",
            wal_tail_status="fixture_only",
        )
        rows = materialize_shadow_evaluation_view(logical_trace)
        assert len(rows) == 1
        assert rows[0].conditional_evaluation
        assert rows[0].binding_verdicts == (REPLAY_BINDING_INDETERMINATE,)
        assert "selected_action" not in rows[0].to_payload()
    finally:
        consumer.close()
    assert consumer.close_report is not None
    assert consumer.close_report.in_memory_complete
    assert consumer.close_report.durable_complete
    assert consumer.close_report.wal_directory is not None
    assert (consumer.close_report.wal_directory / "control_trace.wal").exists()


def test_unknown_source_frame_records_rejection_without_arbiter_candidate(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "episode.parquet"
    consumer = ReplayGuiShadowConsumer(
        profile=ReplayGuiShadowProfile(queue_capacity=8, trace_capacity=128)
    )
    try:
        consumer.bind(_binding(source_path, source_frame=None))
        emission = build_replay_source_frame_emission(
            _group(source_path),
            active_sides=("right",),
            index=0,
            include_arms=True,
            include_end_effectors=False,
            observed_at=10.0,
            schedule_start_at=10.0,
        )
        consumer.emission_queue.put_nowait(emission)
        consumer.emission_queue.join()
        facts = consumer.trace_facts_for_test()
        assert any(
            fact.event_type == "replay_source_adapter_receipt"
            and fact.payload["accepted"] is False
            for fact in facts
        )
        assert not any(
            fact.event_type == "action_proposal" for fact in facts
        )
        assert not any(
            fact.event_type == "arbitration_evaluation" for fact in facts
        )
    finally:
        consumer.close()


def test_replay_gui_shadow_is_default_off_and_owns_no_motion_authority() -> None:
    session = ReplayGuiSession()
    try:
        state = session.state()
        assert state["replay_action_shadow"] == {
            "enabled": False,
            "authoritative": False,
        }
        assert state["execution"]["action_observer"]["enabled"] is False
    finally:
        session.close()

    enabled = ReplayGuiSession(replay_action_shadow=True)
    try:
        state = enabled.state()
        assert state["replay_action_shadow"]["enabled"] is True
        assert state["replay_action_shadow"]["authoritative"] is False
        assert state["replay_action_shadow"]["dispatch_owner_token"] is False
        assert state["execution"]["action_observer"]["enabled"] is True
    finally:
        enabled.close()
