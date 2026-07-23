from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from robot_motion.deploy import (
    ACTION_ARM_POSE_SLICES,
    DEPLOY_ACTION_DIM,
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    CanonicalDeployActionPipeline,
    NormalizedAction,
    PolicyActionEmission,
    STATE_ARM_QPOS_SLICES,
    build_deployment_observation,
    nan_vector,
)
from teleop_runtime.deploy.safety_shadow import (
    evaluate_policy_legacy_aligned_safety,
)


def observation(*, last_action: np.ndarray | None = None):
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.2, 0.0, 0.3]
    return build_deployment_observation(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        step_index=7,
        arm_states={
            "right": {
                "current_qpos": np.full(6, 0.1),
                "current_tcp_pose": pose,
                "actual_tcp_pose": pose,
            }
        },
        hand_states={},
        timestamp=10.0,
        last_action=last_action,
    )


def pipeline() -> CanonicalDeployActionPipeline:
    return CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("right",),
                arms_enabled=True,
                hands_enabled=False,
                max_arm_xyz_step_m=0.05,
                max_arm_rot_step_deg=10.0,
                max_arm_joint_step_rad=0.2,
                arm_joint_lower_rad=(-1.0,) * 6,
                arm_joint_upper_rad=(1.0,) * 6,
            )
        ),
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
    )


def pose_action(*, x: float = 0.4, metadata=None) -> NormalizedAction:
    vector = nan_vector(DEPLOY_ACTION_DIM)
    vector[ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [x, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    return NormalizedAction(vector=vector, metadata=dict(metadata or {}))


def qpos_action(*, value: float = 0.5) -> NormalizedAction:
    return NormalizedAction(
        vector=nan_vector(DEPLOY_ACTION_DIM),
        arm_joint_targets={"right": np.full(6, value)},
    )


def emission(raw: NormalizedAction, observation_value, pipeline_value):
    pre = pipeline_value.adapt_pre_safety(raw, observation_value, policy_name="fixture")
    return PolicyActionEmission(
        emission_id="policy-emission:7",
        admission_mode="on_emission_pop_v1",
        policy_node_id="single_policy",
        inference_id="inference:1",
        policy_output_id="output:1",
        chunk_id=1,
        chunk_keyframe_index=0,
        chunk_size=1,
        decoded_absolute_action=raw,
        pre_safety_robot_action=pre.robot_action,
        source_validation=pre.source_validation,
        representation_receipt=pre.representation_receipt,
        pre_safety_faults=tuple(pre.faults),
        produced_at=9.9,
        produced_clock_domain_id="local_monotonic",
        emitted_at=10.0,
        target_effective_at=10.0,
        effective_from=10.0,
        effective_until=10.1,
        effective_clock_domain_id="local_monotonic",
        schedule_spec_id="schedule:1",
        schedule_spec_hash="schedule-hash",
        queue_len_before=1,
        queue_len_after=0,
        queue_pop_count=1,
        semantic_content_hash="emission-hash",
    )


def evaluate(raw: NormalizedAction, obs):
    pipe = pipeline()
    legacy = pipe.prepare(raw, obs, policy_name="fixture")
    result = evaluate_policy_legacy_aligned_safety(
        emission=emission(raw, obs, pipe),
        arbitration_evaluation_id="arbitration-evaluation:7",
        observation=obs,
        legacy_result=legacy,
        safety_filter=pipe.safety_filter,
    )
    return result, legacy


def test_pose_comparison_uses_the_same_previous_safe_legacy_anchor() -> None:
    # The existing filter only accepts last_action when the entire deploy
    # vector is finite, not merely the active pose slice.
    last = np.zeros(DEPLOY_ACTION_DIM, dtype=np.float32)
    last[ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )

    result, legacy = evaluate(pose_action(), observation(last_action=last))

    assert result.status == "matched"
    assert result.legacy_anchor_kind == "previous_safe"
    assert result.component_parity is not None and result.component_parity.matched
    assert result.shared_safety_result is not None
    assert result.shared_safety_result.safe_action is not None
    assert legacy.robot_action is not None
    assert legacy.robot_action.arm_pose_by_side["right"][0] == pytest.approx(0.15)


def test_pose_comparison_matches_measured_fallback_and_tightening_override() -> None:
    raw = pose_action(
        metadata={"safety_overrides": {"max_arm_xyz_step_m": 0.02}}
    )
    result, legacy = evaluate(raw, observation())

    assert result.status == "matched"
    assert result.legacy_anchor_kind == "measured_state"
    assert legacy.robot_action is not None
    assert legacy.robot_action.arm_pose_by_side["right"][0] == pytest.approx(0.22)
    assert result.details["profile_hash"]


def test_qpos_comparison_records_current_measured_anchor_semantics() -> None:
    last = nan_vector(DEPLOY_ACTION_DIM)
    last[ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [0.9, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    result, legacy = evaluate(qpos_action(), observation(last_action=last))

    assert result.status == "matched"
    assert result.legacy_anchor_kind == "measured_state"
    assert legacy.robot_action is not None
    np.testing.assert_allclose(
        legacy.robot_action.arm_qpos_by_side["right"], np.full(6, 0.3)
    )
    assert tuple(result.details["representation_mismatches"]) == (
        "right_arm:frame",
    )


def test_legacy_and_shared_constraint_faults_match_as_blocked() -> None:
    pipe = pipeline()
    obs = observation()
    state = obs.state.copy()
    state[STATE_ARM_QPOS_SLICES["right"]] = np.full(6, 1.25)
    obs = replace(obs, state=state)
    raw = qpos_action(value=0.0)
    legacy = pipe.prepare(raw, obs, policy_name="fixture")

    result = evaluate_policy_legacy_aligned_safety(
        emission=emission(raw, obs, pipe),
        arbitration_evaluation_id="arbitration-evaluation:7",
        observation=obs,
        legacy_result=legacy,
        safety_filter=pipe.safety_filter,
    )

    assert legacy.faults
    assert result.status == "matched_blocked"
    assert result.shared_safety_result is not None
    assert result.shared_safety_result.status == "fault"


def test_comparison_is_passive_and_does_not_reinvoke_legacy_filter() -> None:
    pipe = pipeline()
    obs = observation()
    raw = pose_action()
    legacy = pipe.prepare(raw, obs, policy_name="fixture")
    item = emission(raw, obs, pipe)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy safety must not be evaluated twice")

    pipe.safety_filter.filter_normalized = forbidden
    result = evaluate_policy_legacy_aligned_safety(
        emission=item,
        arbitration_evaluation_id="arbitration-evaluation:7",
        observation=obs,
        legacy_result=legacy,
        safety_filter=pipe.safety_filter,
    )

    assert result.status == "matched"
    source = inspect.getsource(evaluate_policy_legacy_aligned_safety)
    for forbidden_name in ("prepare_next_action", "pop_action", "dispatch", "send_commands"):
        assert forbidden_name not in source
