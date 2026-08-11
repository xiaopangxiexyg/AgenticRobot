from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.deploy import (
    DEPLOY_ACTION_DIM,
    NormalizedAction,
    PolicyActionEmission,
    nan_vector,
)
from robot_motion.position_control import (
    POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED,
    POSITION_PLAN_GAP_NONE,
    POSITION_PLAN_SPLICE_DYNAMICS_CONSERVATIVE_V1,
    POSITION_PLAN_SPLICE_DYNAMICS_OFF,
    POSITION_PLAN_RELATION_CONTIGUOUS,
    POSITION_PLAN_RELATION_REPLACEMENT,
    POSITION_PLAN_RELATION_SOURCE_GAP,
    POSITION_PLAN_RELATION_STREAM_START,
    POSITION_PLAN_SPLICE_CONTINUE,
    POSITION_PLAN_SPLICE_HOLD_RESTART,
    POSITION_PLAN_SPLICE_LIVE,
    POSITION_PLAN_SPLICE_STREAM_START,
    JointHorizonState,
    PositionSourcePlanCut,
    PositionStreamStartAcquisitionConfig,
    build_position_stream_start_acquisition,
    initial_state_for_position_plan_splice,
    requires_clock_rebase_and_tail_drain_for_position_plan_splice,
    resolve_position_plan_splice,
    resolve_position_plan_splice_dynamics_config,
)
from teleop_runtime.deploy.position_plan_cut import (
    DeploymentPositionPlanCutDetector,
)
from teleop_runtime.deploy.run_policy import (
    _prepare_position_plan_cut_for_action_step,
)


def _cut(
    *,
    relation: str,
    gap: str = POSITION_PLAN_GAP_NONE,
) -> PositionSourcePlanCut:
    previous_plan = None if relation == POSITION_PLAN_RELATION_STREAM_START else "p0"
    previous_step = None if previous_plan is None else 2
    current_plan = (
        "p1" if relation == POSITION_PLAN_RELATION_REPLACEMENT else "p0"
    )
    current_step = 3
    if relation == POSITION_PLAN_RELATION_SOURCE_GAP:
        current_step = 5
    if relation == POSITION_PLAN_RELATION_STREAM_START:
        current_step = 0
    return PositionSourcePlanCut(
        cut_id=f"cut:{relation}:{gap}",
        current_source_plan_id=current_plan,
        current_source_step=current_step,
        previous_source_plan_id=previous_plan,
        previous_source_step=previous_step,
        relation=relation,
        execution_gap=gap,
        detected_at=1.0,
    )


def _state(*, moving: bool) -> JointHorizonState:
    return JointHorizonState(
        q_rad=np.linspace(0.0, 0.5, 6),
        qd_rad_s=np.full(6, 0.2 if moving else 0.0),
        qdd_rad_s2=np.full(6, -0.1 if moving else 0.0),
    )


@pytest.mark.parametrize(
    ("relation", "expected_mode", "preserve", "replace_future"),
    (
        (
            POSITION_PLAN_RELATION_STREAM_START,
            POSITION_PLAN_SPLICE_STREAM_START,
            False,
            False,
        ),
        (
            POSITION_PLAN_RELATION_CONTIGUOUS,
            POSITION_PLAN_SPLICE_CONTINUE,
            True,
            False,
        ),
        (
            POSITION_PLAN_RELATION_SOURCE_GAP,
            POSITION_PLAN_SPLICE_LIVE,
            True,
            False,
        ),
        (
            POSITION_PLAN_RELATION_REPLACEMENT,
            POSITION_PLAN_SPLICE_LIVE,
            True,
            True,
        ),
    ),
)
def test_robot_motion_resolves_source_cut_without_deployment_policy(
    relation: str,
    expected_mode: str,
    preserve: bool,
    replace_future: bool,
) -> None:
    decision = resolve_position_plan_splice(_cut(relation=relation))

    assert decision.mode == expected_mode
    assert decision.preserve_committed_dynamics is preserve
    assert decision.require_zero_dynamic_restart is (not preserve)
    assert decision.mutable_future_replacement is replace_future


def test_acknowledged_hold_overrides_replacement_and_requires_zero_state() -> None:
    decision = resolve_position_plan_splice(
        _cut(
            relation=POSITION_PLAN_RELATION_REPLACEMENT,
            gap=POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED,
        )
    )

    assert decision.mode == POSITION_PLAN_SPLICE_HOLD_RESTART
    assert decision.preserve_committed_dynamics is False
    assert decision.mutable_future_replacement is False
    with pytest.raises(RuntimeError, match="acknowledged reset"):
        initial_state_for_position_plan_splice(
            decision,
            committed_state=_state(moving=True),
        )
    stopped = _state(moving=False)
    assert (
        initial_state_for_position_plan_splice(
            decision,
            committed_state=stopped,
        )
        is stopped
    )


def test_live_replacement_preserves_exact_committed_dynamic_state() -> None:
    decision = resolve_position_plan_splice(
        _cut(relation=POSITION_PLAN_RELATION_REPLACEMENT)
    )
    moving = _state(moving=True)

    assert (
        initial_state_for_position_plan_splice(
            decision,
            committed_state=moving,
        )
        is moving
    )


def test_clock_rebase_and_tail_drain_remain_hold_restart_only() -> None:
    stream_start = resolve_position_plan_splice(
        _cut(relation=POSITION_PLAN_RELATION_STREAM_START)
    )
    hold_restart = resolve_position_plan_splice(
        _cut(
            relation=POSITION_PLAN_RELATION_REPLACEMENT,
            gap=POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED,
        )
    )
    live_replacement = resolve_position_plan_splice(
        _cut(relation=POSITION_PLAN_RELATION_REPLACEMENT)
    )
    contiguous = resolve_position_plan_splice(
        _cut(relation=POSITION_PLAN_RELATION_CONTIGUOUS)
    )

    assert not requires_clock_rebase_and_tail_drain_for_position_plan_splice(
        stream_start
    )
    assert requires_clock_rebase_and_tail_drain_for_position_plan_splice(
        hold_restart
    )
    assert not requires_clock_rebase_and_tail_drain_for_position_plan_splice(
        live_replacement
    )
    assert not requires_clock_rebase_and_tail_drain_for_position_plan_splice(
        contiguous
    )
    assert not requires_clock_rebase_and_tail_drain_for_position_plan_splice(None)


def test_plan_splice_dynamics_profile_has_exact_off_rollback() -> None:
    rollback = resolve_position_plan_splice_dynamics_config(
        POSITION_PLAN_SPLICE_DYNAMICS_OFF
    )
    conservative = resolve_position_plan_splice_dynamics_config(
        POSITION_PLAN_SPLICE_DYNAMICS_CONSERVATIVE_V1
    )

    assert rollback.hold_restart_initial_jerk_delta_regularization == 0.0
    assert rollback.hold_restart_jerk_delta_regularization == 0.0
    assert conservative.hold_restart_initial_jerk_delta_regularization == 0.0
    assert conservative.hold_restart_jerk_delta_regularization == pytest.approx(
        1e-4
    )
    with pytest.raises(ValueError, match="unsupported position plan-splice"):
        resolve_position_plan_splice_dynamics_config("unknown")


def _matrix_pose(x_m: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[0, 3] = float(x_m)
    return pose


def test_stream_start_acquisition_subdivides_from_measured_anchor() -> None:
    acquisition = build_position_stream_start_acquisition(
        {"left": _matrix_pose(0.0), "right": _matrix_pose(0.0)},
        {"left": _matrix_pose(0.043), "right": _matrix_pose(0.012)},
        config=PositionStreamStartAcquisitionConfig(
            enabled=True,
            max_translation_step_m=0.005,
            max_rotation_step_rad=np.deg2rad(0.5),
            max_source_intervals=16,
        ),
    )

    assert acquisition.active is True
    assert acquisition.source_intervals == 9
    assert len(acquisition.prelude_pose_by_side) == 9
    np.testing.assert_allclose(
        acquisition.prelude_pose_by_side[0]["left"],
        _matrix_pose(0.0),
    )
    left_x = [
        float(item["left"][0, 3])
        for item in acquisition.prelude_pose_by_side
    ] + [0.043]
    assert max(np.diff(left_x)) <= 0.005 + 1e-12
    assert left_x[-2] < left_x[-1]


def test_stream_start_acquisition_is_exactly_default_off_and_bounded() -> None:
    disabled = build_position_stream_start_acquisition(
        {"left": _matrix_pose(0.0)},
        {"left": _matrix_pose(0.043)},
        config=PositionStreamStartAcquisitionConfig(enabled=False),
    )
    assert disabled.active is False
    assert disabled.source_intervals == 0
    assert disabled.prelude_pose_by_side == ()

    invalid = _matrix_pose(0.0)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="stream-start Pose is invalid"):
        build_position_stream_start_acquisition(
            {"left": invalid},
            {"left": _matrix_pose(0.01)},
            config=PositionStreamStartAcquisitionConfig(enabled=True),
        )

    with pytest.raises(RuntimeError, match="exceeds its bounded source horizon"):
        build_position_stream_start_acquisition(
            {"left": _matrix_pose(0.0)},
            {"left": _matrix_pose(0.2)},
            config=PositionStreamStartAcquisitionConfig(
                enabled=True,
                max_translation_step_m=0.005,
                max_source_intervals=16,
            ),
        )


def _emission(
    *,
    emission_id: str,
    inference_id: str,
    chunk_id: int,
    step: int,
) -> PolicyActionEmission:
    return PolicyActionEmission(
        emission_id=emission_id,
        admission_mode="on_emission_pop_v1",
        policy_node_id="policy",
        inference_id=inference_id,
        policy_output_id=f"output:{inference_id}",
        chunk_id=chunk_id,
        chunk_keyframe_index=step,
        chunk_size=16,
        decoded_absolute_action=NormalizedAction(
            vector=nan_vector(DEPLOY_ACTION_DIM)
        ),
        pre_safety_robot_action=None,
        source_validation=None,
        representation_receipt=None,
        pre_safety_faults=(),
        produced_at=0.0,
        produced_clock_domain_id="local_monotonic",
        emitted_at=0.0,
        target_effective_at=float(step) / 15.0,
        effective_from=0.0,
        effective_until=1.0,
        effective_clock_domain_id="local_monotonic",
        schedule_spec_id="schedule",
        schedule_spec_hash="schedule-hash",
        queue_len_before=1,
        queue_len_after=0,
        queue_pop_count=1,
        semantic_content_hash=f"hash:{emission_id}",
    )


def _synchronous_action(*, chunk_id: int, step: int, chunk_size: int = 4):
    return NormalizedAction(
        vector=nan_vector(DEPLOY_ACTION_DIM),
        metadata={
            "deploy_action_chunk_id": chunk_id,
            "deploy_action_chunk_index": step,
            "deploy_action_chunk_size": chunk_size,
        },
    )


def test_deployment_detector_commits_lineage_transactionally() -> None:
    detector = DeploymentPositionPlanCutDetector()
    first = detector.prepare(
        _emission(
            emission_id="e0",
            inference_id="i0",
            chunk_id=0,
            step=0,
        ),
        detected_at=1.0,
    )
    assert first.cut.relation == POSITION_PLAN_RELATION_STREAM_START
    detector.commit(first)

    contiguous = detector.prepare(
        _emission(
            emission_id="e1",
            inference_id="i0",
            chunk_id=0,
            step=1,
        ),
        detected_at=2.0,
    )
    assert contiguous.cut.relation == POSITION_PLAN_RELATION_CONTIGUOUS
    detector.reject(contiguous)

    retried = detector.prepare(
        _emission(
            emission_id="e1-retry",
            inference_id="i0",
            chunk_id=0,
            step=1,
        ),
        detected_at=3.0,
    )
    assert retried.cut.relation == POSITION_PLAN_RELATION_CONTIGUOUS
    detector.commit(retried)

    replacement = detector.prepare(
        _emission(
            emission_id="e2",
            inference_id="i1",
            chunk_id=0,
            step=0,
        ),
        detected_at=4.0,
    )
    assert replacement.cut.relation == POSITION_PLAN_RELATION_REPLACEMENT
    assert replacement.cut.execution_gap == POSITION_PLAN_GAP_NONE


def test_pre_stream_hold_is_consumed_then_in_stream_hold_survives_rejection() -> None:
    detector = DeploymentPositionPlanCutDetector()
    detector.observe_hold_generation(4)
    first = detector.prepare(
        _emission(
            emission_id="e0",
            inference_id="i0",
            chunk_id=0,
            step=0,
        ),
        detected_at=1.0,
    )
    assert first.cut.relation == POSITION_PLAN_RELATION_STREAM_START
    assert first.cut.execution_gap == POSITION_PLAN_GAP_NONE
    detector.commit(first)

    detector.observe_hold_generation(5)
    next_step = detector.prepare(
        _emission(
            emission_id="e1",
            inference_id="i0",
            chunk_id=0,
            step=1,
        ),
        detected_at=3.0,
    )
    assert next_step.cut.execution_gap == POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED
    detector.reject(next_step)

    retry = detector.prepare(
        _emission(
            emission_id="e1-retry",
            inference_id="i0",
            chunk_id=0,
            step=1,
        ),
        detected_at=4.0,
    )
    assert retry.cut.execution_gap == POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED
    detector.commit(retry)


def test_detector_rejects_duplicate_inflight_proposals_and_old_hold_generation() -> None:
    detector = DeploymentPositionPlanCutDetector()
    proposal = detector.prepare(
        _emission(
            emission_id="e0",
            inference_id="i0",
            chunk_id=0,
            step=0,
        ),
        detected_at=1.0,
    )
    with pytest.raises(RuntimeError, match="already pending"):
        detector.prepare(
            _emission(
                emission_id="e1",
                inference_id="i0",
                chunk_id=0,
                step=1,
            ),
            detected_at=2.0,
        )
    detector.reject(proposal)
    detector.observe_hold_generation(3)
    with pytest.raises(ValueError, match="cannot move backwards"):
        detector.observe_hold_generation(2)


def test_synchronous_cut_uses_queue_lineage_when_passive_emission_is_absent() -> None:
    detector = DeploymentPositionPlanCutDetector()
    first = _prepare_position_plan_cut_for_action_step(
        detector,
        SimpleNamespace(
            raw_action=_synchronous_action(chunk_id=0, step=0),
            emission=None,
        ),
        rtc_enabled=False,
        policy_node_id="deploy-policy",
        hold_generation=1,
        detected_at=1.0,
    )

    assert first.cut.relation == POSITION_PLAN_RELATION_STREAM_START
    assert first.cut.execution_gap == POSITION_PLAN_GAP_NONE
    assert resolve_position_plan_splice(first.cut).mode == (
        POSITION_PLAN_SPLICE_STREAM_START
    )
    assert first.cut.evidence["lineage_source"] == (
        "normalized_action_chunk_metadata"
    )
    detector.commit(first)

    contiguous = _prepare_position_plan_cut_for_action_step(
        detector,
        SimpleNamespace(
            raw_action=_synchronous_action(chunk_id=0, step=1),
            emission=None,
        ),
        rtc_enabled=False,
        policy_node_id="deploy-policy",
        hold_generation=1,
        detected_at=2.0,
    )
    assert contiguous.cut.relation == POSITION_PLAN_RELATION_CONTIGUOUS
    detector.commit(contiguous)

    replacement = _prepare_position_plan_cut_for_action_step(
        detector,
        SimpleNamespace(
            raw_action=_synchronous_action(chunk_id=1, step=0),
            emission=None,
        ),
        rtc_enabled=False,
        policy_node_id="deploy-policy",
        hold_generation=1,
        detected_at=3.0,
    )
    assert replacement.cut.relation == POSITION_PLAN_RELATION_REPLACEMENT


def test_position_cut_pre_submit_gate_rejects_missing_source_lineage() -> None:
    detector = DeploymentPositionPlanCutDetector()
    with pytest.raises(RuntimeError, match="queue-owned source lineage"):
        _prepare_position_plan_cut_for_action_step(
            detector,
            SimpleNamespace(raw_action=None, emission=None),
            rtc_enabled=False,
            policy_node_id="deploy-policy",
            hold_generation=0,
            detected_at=1.0,
        )

    with pytest.raises(RuntimeError, match="committed emission lineage"):
        _prepare_position_plan_cut_for_action_step(
            detector,
            SimpleNamespace(
                raw_action=_synchronous_action(chunk_id=0, step=0),
                emission=None,
            ),
            rtc_enabled=True,
            policy_node_id="deploy-policy",
            hold_generation=0,
            detected_at=1.0,
        )
