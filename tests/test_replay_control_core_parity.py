from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from robot_motion.action import (
    ActionArbiter,
    ActionComponentCatalog,
    ActionComponentSpec,
    build_robot_action_v1_runtime_schema,
)
from robot_motion.deploy import (
    ACTION_ARM_POSE_SLICES,
    DEPLOY_ACTION_DIM,
    DEPLOY_ACTION_VECTOR_ACTION_SPACE,
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    CanonicalDeployActionPipeline,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    build_deployment_observation,
    nan_vector,
)
from robot_motion.replay import (
    ReplayTrajectoryGroup,
    build_replay_source_frame_emission,
    replay_robot_action_at_index,
)
from robot_motion.replay.control_trace import (
    project_replay_adapter_result as project_core_replay_adapter_result,
)
from teleop_runtime.action.replay_proposal import (
    DatasetReplayProposalShadowAdapter,
    DatasetReplayProposalShadowConfig,
)
from teleop_runtime.control_trace.projection import (
    make_runtime_trace_terminal,
    project_legacy_execution,
    project_replay_binding_evidence,
    project_replay_proposal_adapter_result,
    project_replay_source_frame_emission,
    project_shadow_startup,
    project_shadow_step,
)
from teleop_runtime.control_trace import projection as teleop_trace_projection
from teleop_runtime.control_trace import ControlTraceRuntime, RuntimeTraceSession
from teleop_runtime.deploy.arbiter_shadow import (
    PassivePolicyArbiterShadow,
    PolicyArbiterShadowConfig,
)
from teleop_runtime.deploy.replay import DatasetReplayPolicy


class Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def test_replay_trace_projection_has_one_production_owner() -> None:
    replay_projection_names = (
        "project_candidate_binding_eligibility",
        "project_legacy_replay_contract_derivation",
        "project_replay_action_record",
        "project_replay_binding_assessment",
        "project_replay_binding_evidence",
        "project_replay_proposal_adapter_result",
        "project_replay_source_action_contract",
        "project_replay_source_adapter_receipt",
        "project_replay_source_frame_emission",
        "project_target_runtime_action_contract",
    )
    assert {
        getattr(teleop_trace_projection, name).__module__
        for name in replay_projection_names
    } == {"robot_motion.replay.control_trace"}


def _shadow_contract():
    catalog = ActionComponentCatalog(
        catalog_id="replay-shadow-right-arm-v1",
        action_space=DEPLOY_ACTION_VECTOR_ACTION_SPACE,
        components=(
            ActionComponentSpec(
                component_id="right_arm",
                selection_group_id="right_arm",
                effector_name="right_arm",
            ),
        ),
    )
    return catalog, build_robot_action_v1_runtime_schema(
        component_catalog=catalog,
    )


def _action(x: float) -> np.ndarray:
    value = nan_vector(DEPLOY_ACTION_DIM)
    value[ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [x, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    return value


def _observation():
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.1, 0.2, 0.3]
    return build_deployment_observation(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        step_index=0,
        arm_states={
            "right": {
                "current_qpos": np.zeros(6, dtype=np.float32),
                "current_tcp_pose": pose,
                "actual_tcp_pose": pose,
            }
        },
        hand_states={},
        timestamp=100.0,
    )


def _pipeline() -> CanonicalDeployActionPipeline:
    return CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("right",),
                arms_enabled=True,
                hands_enabled=False,
                max_arm_xyz_step_m=0.08,
            )
        ),
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
    )


def test_dataset_replay_uses_unique_pop_and_replays_same_frozen_decision(tmp_path: Path) -> None:
    clock = Clock()
    replay = DatasetReplayPolicy(
        actions=np.stack((_action(0.12), _action(0.13))),
        source_path=tmp_path / "episode.parquet",
        chunk_size=2,
        source_start_index=4,
    )
    deploy_runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
        ),
        monotonic_clock=clock,
    )
    observation = _observation()
    prediction = deploy_runtime.predict_and_queue(
        replay,
        observation,
        now=clock.value,
        queue_policy="append",
        action_keyframe_period_sec=1.0 / 15.0,
    )
    assert prediction.accepted
    action_step = deploy_runtime.prepare_next_action(
        observation,
        preparer=_pipeline(),
        policy_name="dataset-replay",
    )
    assert action_step.emission is not None
    assert action_step.raw_action is not None
    pop_count = deploy_runtime.action_queue_pop_count()

    catalog, action_schema = _shadow_contract()
    coordinator = PassivePolicyArbiterShadow(
        PolicyArbiterShadowConfig(
            runtime_session_id="replay-shadow-session",
            control_domain_id="robot-0",
            embodiment_instance_id="rm65-pair-0",
            producer_id="replay-main",
            active_effectors=("right_arm",),
            action_space=DEPLOY_ACTION_VECTOR_ACTION_SPACE,
            decision_hz=15.0,
            component_catalog=catalog,
            action_schema=action_schema,
        ),
        clock=clock,
    )
    replay_adapter = DatasetReplayProposalShadowAdapter(
        DatasetReplayProposalShadowConfig(
            runtime_session_id=coordinator.config.runtime_session_id,
            control_domain_id=coordinator.config.control_domain_id,
            embodiment_instance_id=coordinator.config.embodiment_instance_id,
            producer_id=coordinator.config.producer_id,
            producer_epoch=coordinator.config.producer_epoch,
            action_schema=action_schema,
            component_schema_hash=coordinator.component_catalog.semantic_content_hash,
            component_coverage=("right_arm",),
            source_path=str(replay.source_path),
            episode_id="episode-000001",
            action_field="action",
            source_action_space=DEPLOY_ACTION_VECTOR_ACTION_SPACE,
            source_hz=15.0,
            source_start_index=4,
            dataset_metadata={
                "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
                "pose_frame_contract_hash": "target-frame",
                "component_schema_hash": (
                    coordinator.component_catalog.semantic_content_hash
                ),
            },
            target_frame_contract_hash="target-frame",
        )
    )
    adapted = replay_adapter.adapt_emission(
        action_step.emission,
        raw_action_metadata=action_step.raw_action.metadata,
    )
    assert adapted.proposal_draft is not None
    shadow_step = coordinator.evaluate_proposal_draft(
        adapted.proposal_draft,
        source_event_id=adapted.source_adapter_receipt.representation_trace_id,
        emission=action_step.emission,
    )

    assert deploy_runtime.action_queue_pop_count() == pop_count == 1
    assert replay.cursor == 2  # prediction consumed one chunk; shadow consumed none.
    assert shadow_step.status == "evaluated"
    assert shadow_step.admission_receipt is not None
    proposal = shadow_step.admission_receipt.proposal
    assert proposal is not None
    assert proposal.producer_kind == "replay"
    assert adapted.replay_action_record is not None
    assert adapted.replay_action_record.source_action_index == 4
    assert adapted.replay_action_record.source_timestamp == 0.0
    np.testing.assert_allclose(
        proposal.action.value("right_arm").values,
        action_step.emission.pre_safety_robot_action.arm_pose_by_side["right"],
    )
    assert proposal.effective_from == action_step.emission.effective_from
    assert proposal.effective_until == action_step.emission.effective_until

    assert shadow_step.arbitration_input_snapshot is not None
    assert shadow_step.computation is not None
    replayed = ActionArbiter().arbitrate(shadow_step.arbitration_input_snapshot)
    assert replayed.semantic_content_hash == shadow_step.computation.semantic_content_hash

    replay_facts = project_replay_proposal_adapter_result(
        adapted,
        runtime_session_id=coordinator.config.runtime_session_id,
        control_domain_id=coordinator.config.control_domain_id,
        embodiment_instance_id=coordinator.config.embodiment_instance_id,
        occurred_at=clock.value,
        clock_domain_id=coordinator.config.clock_domain_id,
    )
    assert [fact.event_type for fact in replay_facts] == [
        "candidate_binding_eligibility",
        "replay_action_record",
        "replay_source_adapter_receipt",
    ]
    replay_binding_facts = project_replay_binding_evidence(
        adapted,
        runtime_session_id=coordinator.config.runtime_session_id,
        control_domain_id=coordinator.config.control_domain_id,
        embodiment_instance_id=coordinator.config.embodiment_instance_id,
        occurred_at=clock.value,
        clock_domain_id=coordinator.config.clock_domain_id,
    )
    combined_facts = project_core_replay_adapter_result(
        adapted,
        runtime_session_id=coordinator.config.runtime_session_id,
        control_domain_id=coordinator.config.control_domain_id,
        embodiment_instance_id=coordinator.config.embodiment_instance_id,
        occurred_at=clock.value,
        clock_domain_id=coordinator.config.clock_domain_id,
    )
    assert [fact.semantic_content_hash for fact in combined_facts] == [
        fact.semantic_content_hash
        for fact in (*replay_binding_facts, *replay_facts)
    ]
    trace_binding_facts = project_replay_binding_evidence(
        adapted,
        runtime_session_id=coordinator.config.runtime_session_id,
        control_domain_id="trace-domain",
        embodiment_instance_id="trace-embodiment",
        occurred_at=clock.value,
        clock_domain_id=coordinator.config.clock_domain_id,
    )
    target_fact = next(
        fact
        for fact in trace_binding_facts
        if fact.event_type == "target_runtime_action_contract"
    )
    assert target_fact.control_domain_id == adapted.target_runtime_contract.control_domain_id
    assert (
        target_fact.embodiment_instance_id
        == adapted.target_runtime_contract.embodiment_instance_id
    )
    bulk_target_fact = next(
        fact
        for fact in project_core_replay_adapter_result(
            adapted,
            runtime_session_id=coordinator.config.runtime_session_id,
            control_domain_id="trace-domain",
            embodiment_instance_id="trace-embodiment",
            occurred_at=clock.value,
            clock_domain_id=coordinator.config.clock_domain_id,
        )
        if fact.event_type == "target_runtime_action_contract"
    )
    assert bulk_target_fact.control_domain_id == "trace-domain"
    assert bulk_target_fact.embodiment_instance_id == "trace-embodiment"

    legacy = coordinator.observe_legacy_execution(
        emission=action_step.emission,
        shadow_step=shadow_step,
        canonical_action=action_step.prepared_action,
        execution=SimpleNamespace(
            accepted=True,
            mode="dry_run",
            dispatch_attempted=False,
            dispatch_completed=False,
        ),
        dispatch_error=None,
        observed_at=clock.value,
    )
    trace = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="trace-replay-shadow",
            runtime_session_id=coordinator.config.runtime_session_id,
            control_domain_id=coordinator.config.control_domain_id,
            profile_id="replay_shadow_dry_run_v1",
            started_at=clock.value,
            clock_domain_id=coordinator.config.clock_domain_id,
        ),
        capacity=64,
    )
    facts = (
        *project_shadow_startup(coordinator),
        *replay_binding_facts,
        *replay_facts,
        *project_shadow_step(coordinator, shadow_step),
        project_legacy_execution(coordinator, legacy),
        make_runtime_trace_terminal(
            coordinator,
            terminal_status="completed",
            reason="fixture_complete",
            occurred_at=clock.value,
        ),
    )
    for fact in facts:
        assert trace.issue(fact).accepted
    report = trace.completeness()
    assert report.in_memory_complete
    assert report.lineage_complete
    assert report.lineage_issues == ()
    assert "arbitration_decision" not in {
        item.fact.event_type for item in trace.records()
    }


def test_hardware_free_trajectory_action_and_proposal_adapter_have_value_time_parity(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "episode.parquet"
    group = ReplayTrajectoryGroup(
        key="g",
        label="fixture",
        source_path=str(source_path),
        source_episode_index=1,
        start_index=4,
        end_index=6,
        frame_count=2,
        action_hz=15.0,
        actions=np.stack((_action(0.12), _action(0.13))),
        metadata={
            "source_metadata": {
                "dataset_info": {
                    "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
                }
            }
        },
    )
    catalog, action_schema = _shadow_contract()
    coordinator = PassivePolicyArbiterShadow(
        PolicyArbiterShadowConfig(
            runtime_session_id="trajectory-shadow-session",
            control_domain_id="robot-0",
            embodiment_instance_id="rm65-pair-0",
            producer_id="replay-main",
            active_effectors=("right_arm",),
            action_space=DEPLOY_ACTION_VECTOR_ACTION_SPACE,
            decision_hz=15.0,
            component_catalog=catalog,
            action_schema=action_schema,
        ),
        clock=Clock(),
    )
    adapter = DatasetReplayProposalShadowAdapter(
        DatasetReplayProposalShadowConfig(
            runtime_session_id=coordinator.config.runtime_session_id,
            control_domain_id=coordinator.config.control_domain_id,
            embodiment_instance_id=coordinator.config.embodiment_instance_id,
            producer_id=coordinator.config.producer_id,
            producer_epoch=coordinator.config.producer_epoch,
            action_schema=action_schema,
            component_schema_hash=coordinator.component_catalog.semantic_content_hash,
            component_coverage=("right_arm",),
            source_path=str(source_path),
            episode_id="episode_000001",
            action_field="action",
            source_action_space=DEPLOY_ACTION_VECTOR_ACTION_SPACE,
            source_hz=15.0,
            source_start_index=4,
            dataset_metadata={
                "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
                "pose_frame_contract_hash": "target-frame",
                "component_schema_hash": (
                    coordinator.component_catalog.semantic_content_hash
                ),
            },
            target_frame_contract_hash="target-frame",
        )
    )
    target_at = 20.0
    existing = replay_robot_action_at_index(
        group,
        ("right",),
        index=0,
        include_arms=True,
        include_end_effectors=False,
        timestamp=target_at,
    )
    adapted = adapter.adapt_trajectory_frame(
        group,
        index=0,
        include_arms=True,
        include_end_effectors=False,
        produced_at=19.9,
        produced_clock_domain_id="local_monotonic",
        effective_from=20.0,
        effective_until=20.0 + 1.0 / 15.0,
        target_effective_at=target_at,
        effective_clock_domain_id="local_monotonic",
    )
    assert adapted.proposal_draft is not None
    assert adapted.replay_action_record is not None
    assert adapted.replay_action_record.source_action_index == 4
    assert adapted.replay_action_record.source_timestamp == 0.0
    assert adapted.proposal_draft.target_effective_at == target_at
    np.testing.assert_allclose(
        adapted.proposal_draft.action.value("right_arm").values,
        existing.arm_pose_by_side["right"],
    )
    source_emission = build_replay_source_frame_emission(
        group,
        active_sides=("right",),
        index=1,
        include_arms=True,
        include_end_effectors=False,
        observed_at=20.05,
        schedule_start_at=20.0,
    )
    source_fact = project_replay_source_frame_emission(
        source_emission,
        runtime_session_id=coordinator.config.runtime_session_id,
        control_domain_id=coordinator.config.control_domain_id,
        embodiment_instance_id=coordinator.config.embodiment_instance_id,
    )
    assert source_fact.event_type == "replay_source_frame_emission"
    assert source_fact.payload["source_action_index"] == 5
    direct = adapter.adapt_source_frame_emission(source_emission)
    assert direct.proposal_draft is not None
    assert direct.source_adapter_receipt.source_emission_id == source_emission.emission_id
