from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.deploy.actions import (
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    FilteredNormalizedAction,
    NormalizedAction,
)
from robot_motion.deploy.canonical_action import CanonicalDeployActionPipeline
from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    OBSERVATION_STATE_DIM,
    STATE_ARM_POSE_SLICES,
    STATE_ARM_QPOS_SLICES,
)
from robot_motion.position_control import (
    POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED,
    POSITION_PLAN_GAP_NONE,
    POSITION_PLAN_RELATION_REPLACEMENT,
    JointHorizonState,
    PositionSourcePlanCut,
    resolve_position_plan_splice,
)
from teleop_runtime.position_control.deploy_executor import (
    PositionDeployExecutor,
    PositionDeployExecutorConfig,
)
from teleop_runtime.position_control.official_all_ik_governor import (
    RM65OfficialAllIKGovernorPlanner,
)


def _cut(*, hold: bool = False) -> PositionSourcePlanCut:
    return PositionSourcePlanCut(
        cut_id="cut:replacement",
        current_source_plan_id="plan:new",
        current_source_step=0,
        previous_source_plan_id="plan:old",
        previous_source_step=15,
        relation=POSITION_PLAN_RELATION_REPLACEMENT,
        execution_gap=(
            POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED
            if hold
            else POSITION_PLAN_GAP_NONE
        ),
        detected_at=1.0,
    )


def _pipeline() -> CanonicalDeployActionPipeline:
    return CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("left",),
                arms_enabled=True,
                hands_enabled=False,
                max_arm_xyz_step_m=0.0,
                max_arm_rot_step_deg=0.0,
                max_arm_joint_step_rad=0.0,
            )
        ),
        active_sides=("left",),
        arms_enabled=True,
        hands_enabled=False,
    )


def _filtered() -> FilteredNormalizedAction:
    vector = np.full(ACTION_DIM, np.nan, dtype=np.float32)
    vector[ACTION_ARM_POSE_SLICES["left"]] = np.asarray(
        [0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    return FilteredNormalizedAction(
        action=NormalizedAction(vector=vector),
        metadata={"source": "unit_safe_15hz"},
    )


def _observation() -> DeploymentObservation:
    state = np.full(OBSERVATION_STATE_DIM, np.nan, dtype=np.float32)
    valid = np.zeros(OBSERVATION_STATE_DIM, dtype=bool)
    pose = np.asarray(
        [0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    qpos = np.zeros(6, dtype=np.float32)
    state[STATE_ARM_POSE_SLICES["left"]] = pose
    state[STATE_ARM_QPOS_SLICES["left"]] = qpos
    valid[STATE_ARM_POSE_SLICES["left"]] = True
    valid[STATE_ARM_QPOS_SLICES["left"]] = True
    return DeploymentObservation(
        timestamp=1.0,
        step_index=0,
        active_sides=("left",),
        arms_enabled=True,
        hands_enabled=False,
        state=state,
        state_valid=valid,
        arm_states={"left": {"measured_qpos": qpos.copy()}},
    )


def test_executor_default_off_rejects_boundary_signal_without_queue_mutation() -> None:
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(debug_events_enabled=True),
        canonical_pipeline=_pipeline(),
    )
    try:
        with pytest.raises(ValueError, match="plan_splice_governor_enabled"):
            executor.submit(
                _filtered(),
                _observation(),
                "test",
                source_plan_cut=_cut(),
            )
        assert executor.snapshot()["submitted_count"] == 0
    finally:
        executor.close()


def test_executor_enabled_path_only_attaches_robot_motion_decision() -> None:
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            debug_events_enabled=True,
            plan_splice_governor_enabled=True,
        ),
        canonical_pipeline=_pipeline(),
    )
    try:
        source_index = executor.submit(
            _filtered(),
            _observation(),
            "test",
            source_plan_cut=_cut(),
        )
        events = executor.drain_debug_events()
        submitted = next(event for event in events if event["event"] == "keyframe_submitted")
        assert source_index >= 0
        assert submitted["source_index"] == source_index
        assert submitted["position_plan_splice"]["mode"] == (
            "live_splice_preserve_committed_dynamics"
        )
    finally:
        executor.close()


class _CandidateProvider:
    def solve_candidates(self, *, seed_rad, target_endpoint):
        return (np.asarray(seed_rad, dtype=float).reshape(6).copy(),)


class _CollisionPlanner:
    collision_pair_count = 1

    def reset(self, _current_qpos_by_side):
        return {"reset": True}

    def validate_qpos_reference(self, _qpos_by_side, *, source):
        return SimpleNamespace(
            safe=True,
            reason=None,
            collision_distance_m=0.1,
            source=source,
        )

    admit_qpos_reference = validate_qpos_reference

    def observe_endpoint_poses_world(self, _qpos_by_side):
        return {"left": np.eye(4), "right": np.eye(4)}


def _official_planner() -> RM65OfficialAllIKGovernorPlanner:
    return RM65OfficialAllIKGovernorPlanner(
        _CollisionPlanner(),
        endpoint_to_tool_by_side={"left": np.eye(4), "right": np.eye(4)},
        candidate_provider=_CandidateProvider(),
    )


def _poses() -> dict[str, np.ndarray]:
    return {"left": np.eye(4), "right": np.eye(4)}


def _moving_state() -> JointHorizonState:
    return JointHorizonState(
        q_rad=np.zeros(12),
        qd_rad_s=np.full(12, 0.1),
        qdd_rad_s2=np.full(12, -0.05),
    )


def test_official_governor_delegates_live_splice_state_to_robot_motion() -> None:
    planner = _official_planner()
    decision = resolve_position_plan_splice(_cut())
    planner.stage_source_pose(
        source_index=4,
        source_timestamp=4.0 / 15.0,
        pose_by_side=_poses(),
        plan_splice_decision=decision,
    )
    moving = _moving_state()

    state, metadata = planner._initial_state_for_source_cut(
        interval_index=4,
        committed_state=moving,
    )

    assert state is moving
    assert metadata["position_plan_splice_mode"] == decision.mode
    assert metadata["position_plan_splice_preserve_committed_dynamics"] is True


def test_official_governor_rejects_hold_restart_from_moving_state() -> None:
    planner = _official_planner()
    decision = resolve_position_plan_splice(_cut(hold=True))
    planner.stage_source_pose(
        source_index=4,
        source_timestamp=4.0 / 15.0,
        pose_by_side=_poses(),
        plan_splice_decision=decision,
    )

    with pytest.raises(RuntimeError, match="acknowledged reset"):
        planner._initial_state_for_source_cut(
            interval_index=4,
            committed_state=_moving_state(),
        )
