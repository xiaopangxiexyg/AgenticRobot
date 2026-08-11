from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.position_control import (
    POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED,
    POSITION_PLAN_RELATION_REPLACEMENT,
    PositionPlanSpliceDynamicsConfig,
    PositionSourcePlanCut,
    ResampledMotionReference,
    resolve_position_plan_splice,
)
from robot_motion.rm65 import RM65PositionCollisionPlannerConfig
from teleop_runtime.position_control import official_all_ik_governor as governor_module
from teleop_runtime.position_control.official_all_ik_governor import (
    CONTINUOUS_WAYPOINT_WORKER_PROCESS,
    OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND,
    RM65ContinuousWaypointProcessSpec,
    RM65OfficialAllIKGovernorPlanner,
)


class _CandidateProvider:
    def solve_candidates(self, *, seed_rad, target_endpoint):
        seed = np.asarray(seed_rad, dtype=float).reshape(6)
        candidate = seed.copy()
        candidate[0] = float(np.asarray(target_endpoint)[0, 3])
        return (candidate,)


class _CollisionPlanner:
    collision_pair_count = 4

    def __init__(self, *, reject_continuous_admission: bool = False) -> None:
        self.reject_continuous_admission = bool(reject_continuous_admission)

    def reset(self, current_qpos_by_side):
        return {"reset": True}

    def validate_qpos_reference(self, qpos_by_side, *, source):
        rejected = bool(
            self.reject_continuous_admission
            and str(source).startswith("deploy_continuous_waypoint_admission")
        )
        return SimpleNamespace(
            safe=not rejected,
            reason="synthetic continuous-path collision" if rejected else None,
            collision_distance_m=0.0 if rejected else 0.1,
        )

    def admit_qpos_reference(self, qpos_by_side, *, source):
        return self.validate_qpos_reference(qpos_by_side, source=source)

    def observe_endpoint_poses_world(self, qpos_by_side):
        result = {}
        for side in ("left", "right"):
            pose = np.eye(4, dtype=float)
            pose[0, 3] = float(np.asarray(qpos_by_side[side])[0])
            result[side] = pose
        return result


def _poses(x: float) -> dict[str, np.ndarray]:
    result = {}
    for side in ("left", "right"):
        pose = np.eye(4, dtype=float)
        pose[0, 3] = float(x)
        result[side] = pose
    return result


def _reference(x: float = 0.0) -> ResampledMotionReference:
    return _indexed_reference(source_index=0, source_timestamp=0.0, x=x)


def _indexed_reference(
    *,
    source_index: int,
    source_timestamp: float,
    x: float = 0.0,
) -> ResampledMotionReference:
    return ResampledMotionReference(
        sequence=0,
        timestamp=float(source_timestamp),
        left_source_index=int(source_index),
        right_source_index=int(source_index),
        left_source_timestamp=float(source_timestamp),
        right_source_timestamp=float(source_timestamp),
        interpolation_alpha=0.0,
        pose_by_side=_poses(x),
        qpos_by_side={},
        metadata={},
    )


def _planner(
    *,
    shadow: bool = False,
    admission: bool = False,
    async_precompute: bool = False,
    continuous_waypoint_worker_cpu: int | None = None,
    continuous_waypoint_worker_isolation: str = "thread",
    collision_planner: _CollisionPlanner | None = None,
    plan_splice_dynamics_config: PositionPlanSpliceDynamicsConfig | None = None,
) -> RM65OfficialAllIKGovernorPlanner:
    canonical_collision_planner = collision_planner or _CollisionPlanner()
    process_isolation = bool(
        continuous_waypoint_worker_isolation
        == CONTINUOUS_WAYPOINT_WORKER_PROCESS
    )
    precompute_collision_planner = (
        _CollisionPlanner(
            reject_continuous_admission=bool(
                getattr(
                    canonical_collision_planner,
                    "reject_continuous_admission",
                    False,
                )
            )
        )
        if async_precompute and not process_isolation
        else None
    )
    process_spec = (
        RM65ContinuousWaypointProcessSpec(
            collision_planner_config=RM65PositionCollisionPlannerConfig(
                reference_hz=120.0,
                solve_min_iterations=1,
                solve_max_iterations=1,
            ),
            endpoint_to_tool_by_side={
                "left": np.eye(4),
                "right": np.eye(4),
            },
        )
        if async_precompute and process_isolation
        else None
    )
    return RM65OfficialAllIKGovernorPlanner(
        canonical_collision_planner,
        continuous_waypoint_precompute_collision_planner=(
            precompute_collision_planner
        ),
        endpoint_to_tool_by_side={
            "left": np.eye(4),
            "right": np.eye(4),
        },
        candidate_provider=_CandidateProvider(),
        continuous_waypoint_shadow_enabled=shadow,
        continuous_waypoint_admission_enabled=admission,
        continuous_waypoint_async_precompute_enabled=async_precompute,
        continuous_waypoint_worker_cpu=continuous_waypoint_worker_cpu,
        continuous_waypoint_worker_isolation=(
            continuous_waypoint_worker_isolation
        ),
        continuous_waypoint_process_spec=process_spec,
        plan_splice_dynamics_config=plan_splice_dynamics_config,
    )


def test_async_precompute_worker_affinity_is_verified_before_use(monkeypatch) -> None:
    monkeypatch.setattr(
        governor_module,
        "bind_current_thread_to_cpu",
        lambda requested_cpu, *, role: {
            "requested": True,
            "applied": True,
            "requested_cpu": requested_cpu,
            "effective_cpu_ids": [requested_cpu],
            "role": role,
        },
    )
    planner = _planner(
        admission=True,
        async_precompute=True,
        continuous_waypoint_worker_cpu=7,
    )
    try:
        receipt = planner.reset(
            {"left": np.zeros(6), "right": np.zeros(6)}
        )
        affinity = receipt["continuous_waypoint_worker_affinity_receipt"]
        assert affinity["requested_cpu"] == 7
        assert affinity["effective_cpu_ids"] == [7]
    finally:
        planner.close()


def test_async_precompute_worker_affinity_failure_is_fail_closed(monkeypatch) -> None:
    def fail_affinity(_requested_cpu, *, role):
        raise OSError(f"cannot pin {role}")

    monkeypatch.setattr(
        governor_module,
        "bind_current_thread_to_cpu",
        fail_affinity,
    )
    with pytest.raises(RuntimeError, match="CPU-affinity setup failed"):
        _planner(
            admission=True,
            async_precompute=True,
            continuous_waypoint_worker_cpu=7,
        )


def _run_remaining_source_cut(
    planner: RM65OfficialAllIKGovernorPlanner,
    *,
    interval_index: int,
    start_offset: int = 1,
    stop_offset: int = 7,
) -> None:
    interval_timestamp = interval_index / 15.0
    for offset in range(start_offset, stop_offset + 1):
        planner.plan_pose_reference_horizon(
            _indexed_reference(
                source_index=interval_index,
                source_timestamp=interval_timestamp,
            ),
            source_timestamp=interval_timestamp + offset / 120.0,
            source="test",
        )


def _wait_for_precompute_result(
    planner: RM65OfficialAllIKGovernorPlanner,
    *,
    timeout_sec: float = 1.0,
) -> None:
    deadline = time.monotonic() + timeout_sec
    if (
        planner.continuous_waypoint_worker_isolation
        == CONTINUOUS_WAYPOINT_WORKER_PROCESS
    ):
        while not planner._continuous_precompute_results:
            planner._drain_continuous_waypoint_process_results()
            if time.monotonic() >= deadline:
                raise AssertionError("continuous waypoint precompute did not finish")
            time.sleep(0.005)
        return
    with planner._continuous_precompute_condition:
        while not planner._continuous_precompute_results:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise AssertionError("continuous waypoint precompute did not finish")
            planner._continuous_precompute_condition.wait(timeout=remaining)


def test_process_isolated_precompute_consumes_a_pickled_complete_cut() -> None:
    planner = _planner(
        admission=True,
        async_precompute=True,
        continuous_waypoint_worker_isolation=(
            CONTINUOUS_WAYPOINT_WORKER_PROCESS
        ),
    )
    try:
        receipt = planner.reset(
            {"left": np.zeros(6), "right": np.zeros(6)}
        )
        assert receipt["continuous_waypoint_worker_isolation"] == "process"
        assert int(receipt["continuous_waypoint_process_pid"]) > 0
        assert receipt["continuous_waypoint_process_error"] is None
        assert receipt["continuous_waypoint_worker_affinity_receipt"][
            "parent_request_queue_feeder_warmed"
        ] is True

        for index, x in enumerate((0.0, 0.0002, 0.0004)):
            planner.stage_source_pose(
                source_index=index,
                source_timestamp=index / 15.0,
                pose_by_side=_poses(x),
            )
        first = planner.plan_pose_reference_horizon(
            _indexed_reference(source_index=0, source_timestamp=0.0),
            source_timestamp=0.0,
            source="test",
        )
        assert first.metadata["governor_execution_mode"] == (
            "formal_third_order_horizon"
        )
        for timing_key in (
            "selection_ms",
            "horizon_plan_ms",
            "collision_validation_ms",
            "endpoint_fk_ms",
            "continuous_waypoint_shadow_metadata_ms",
            "continuous_waypoint_precompute_submit_ms",
        ):
            assert float(first.metadata[timing_key]) >= 0.0
        assert float(first.metadata["frame_total_ms"]) >= float(
            first.metadata["solve_ms"]
        )

        planner.stage_source_pose(
            source_index=3,
            source_timestamp=3.0 / 15.0,
            pose_by_side=_poses(0.0006),
        )
        _run_remaining_source_cut(
            planner,
            interval_index=0,
            start_offset=1,
            stop_offset=1,
        )
        _wait_for_precompute_result(planner, timeout_sec=5.0)
        _run_remaining_source_cut(
            planner,
            interval_index=0,
            start_offset=2,
        )

        next_cut = planner.plan_pose_reference_horizon(
            _indexed_reference(
                source_index=1,
                source_timestamp=1.0 / 15.0,
            ),
            source_timestamp=1.0 / 15.0,
            source="test",
        )
        shadow = next_cut.metadata["continuous_waypoint_shadow"]
        assert next_cut.metadata["continuous_waypoint_worker_isolation"] == (
            "process"
        )
        assert next_cut.metadata["continuous_waypoint_process_error"] is None
        assert next_cut.metadata["governor_execution_mode"] == (
            "continuous_c2_waypoint"
        )
        assert shadow["precompute_consumed_async"] is True
        assert shadow["admission_status"] == (
            "admitted_for_complete_source_cut"
        )
    finally:
        planner.close(timeout_sec=2.0)


def test_process_isolated_worker_exit_falls_back_without_faulting_formal() -> None:
    planner = _planner(
        admission=True,
        async_precompute=True,
        continuous_waypoint_worker_isolation=(
            CONTINUOUS_WAYPOINT_WORKER_PROCESS
        ),
    )
    try:
        planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
        process = planner._continuous_precompute_process
        assert process is not None
        process.terminate()
        process.join(timeout=2.0)
        assert not process.is_alive()

        for index, x in enumerate((0.0, 0.0002, 0.0004, 0.0006)):
            planner.stage_source_pose(
                source_index=index,
                source_timestamp=index / 15.0,
                pose_by_side=_poses(x),
            )
        first = planner.plan_pose_reference_horizon(
            _indexed_reference(source_index=0, source_timestamp=0.0),
            source_timestamp=0.0,
            source="test",
        )
        assert first.valid is True
        assert first.metadata["governor_execution_mode"] == (
            "formal_third_order_horizon"
        )
        _run_remaining_source_cut(
            planner,
            interval_index=0,
            start_offset=1,
        )
        next_cut = planner.plan_pose_reference_horizon(
            _indexed_reference(
                source_index=1,
                source_timestamp=1.0 / 15.0,
            ),
            source_timestamp=1.0 / 15.0,
            source="test",
        )
        assert next_cut.valid is True
        assert next_cut.metadata["governor_execution_mode"] == (
            "formal_third_order_horizon"
        )
        assert "proposal process exited" in str(
            next_cut.metadata["continuous_waypoint_process_error"]
        )
        shadow = next_cut.metadata["continuous_waypoint_shadow"]
        assert shadow["precompute_miss_reason"] == "process_error"
        assert shadow["dispatch_selected"] is False
    finally:
        planner.close(timeout_sec=2.0)


def test_official_all_ik_governor_uses_staged_two_source_horizon() -> None:
    planner = _planner()
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    for index, x in enumerate((0.0, 0.08, 0.16)):
        planner.stage_source_pose(
            source_index=index,
            source_timestamp=index / 15.0,
            pose_by_side=_poses(x),
        )
    reference = ResampledMotionReference(
        sequence=0,
        timestamp=0.0,
        left_source_index=0,
        right_source_index=0,
        left_source_timestamp=0.0,
        right_source_timestamp=0.0,
        interpolation_alpha=0.0,
        pose_by_side=_poses(0.0),
        qpos_by_side={},
        metadata={},
    )

    result = planner.plan_pose_reference_horizon(
        reference,
        source_timestamp=0.0,
        source="test",
    )

    assert result.valid is True
    assert result.hold is False
    assert result.metadata["planner"] == OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND
    assert result.metadata["lookahead_source_steps_available"] == 2
    assert result.metadata["segment_collision_checked"] is True
    assert result.metadata["segment_collision_certified"] is False
    assert np.all(np.isfinite(result.q_ref_by_side["left"]))

    next_reference = ResampledMotionReference(
        sequence=1,
        timestamp=1.0 / 120.0,
        left_source_index=0,
        right_source_index=0,
        left_source_timestamp=0.0,
        right_source_timestamp=0.0,
        interpolation_alpha=0.125,
        pose_by_side=_poses(0.01),
        qpos_by_side={},
        metadata={},
    )
    next_result = planner.plan_pose_reference_horizon(
        next_reference,
        source_timestamp=1.0 / 120.0,
        source="test",
    )
    assert next_result.metadata["source_cut_replanned"] is False
    assert (
        next_result.metadata["governor_solver"]
        == "jerk_limited_joint_horizon_qp_v4_reused_batched"
    )
    assert next_result.metadata["governor_solver_tolerance"] == 1e-6


def test_hold_restart_splice_dynamics_is_applied_only_by_typed_cut() -> None:
    planner = _planner(
        plan_splice_dynamics_config=PositionPlanSpliceDynamicsConfig(
            hold_restart_jerk_delta_regularization=1e-4
        )
    )
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    decision = resolve_position_plan_splice(
        PositionSourcePlanCut(
            cut_id="hold-restart-0",
            current_source_plan_id="chunk-1",
            current_source_step=0,
            previous_source_plan_id="chunk-0",
            previous_source_step=15,
            relation=POSITION_PLAN_RELATION_REPLACEMENT,
            execution_gap=POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED,
            detected_at=0.0,
        )
    )
    for index, x in enumerate((0.0, 0.0002, 0.0004)):
        planner.stage_source_pose(
            source_index=index,
            source_timestamp=index / 15.0,
            pose_by_side=_poses(x),
            plan_splice_decision=decision if index == 0 else None,
        )

    result = planner.plan_pose_reference_horizon(
        _reference(),
        source_timestamp=0.0,
        source="test",
    )

    assert result.valid is True
    assert result.metadata[
        "plan_splice_initial_jerk_regularization_applied"
    ] is True
    assert result.metadata[
        "plan_splice_initial_jerk_regularization_weight"
    ] == 0.0
    assert result.metadata[
        "plan_splice_jerk_delta_regularization_weight"
    ] == pytest.approx(1e-4)
    assert result.metadata[
        "plan_splice_initial_jerk_regularization_fallback"
    ] is False


def test_equivalent_source_restage_preserves_exact_async_hash_anchor() -> None:
    planner = _planner(admission=True, async_precompute=True)
    try:
        original_poses = _poses(0.0006)
        planner.stage_source_pose(
            source_index=3,
            source_timestamp=3.0 / 15.0,
            pose_by_side=original_poses,
        )
        staged = planner._source_pose[3]
        equivalent_poses = {
            side: pose.copy() for side, pose in original_poses.items()
        }
        equivalent_poses["left"][0, 3] += 5e-9

        planner.stage_source_pose(
            source_index=3,
            source_timestamp=staged.timestamp + 5e-10,
            pose_by_side=equivalent_poses,
        )

        assert planner._source_pose[3] is staged
        assert planner._source_pose[3].timestamp == 3.0 / 15.0
        np.testing.assert_array_equal(
            planner._source_pose[3].pose_by_side["left"],
            original_poses["left"],
        )
    finally:
        planner.close()


def test_initial_staged_source_horizon_can_be_prepared_without_committing() -> None:
    planner = _planner(admission=True, async_precompute=True)
    try:
        planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
        for index, x in enumerate((0.0, 0.0002, 0.0004, 0.0006)):
            planner.stage_source_pose(
                source_index=index,
                source_timestamp=index / 15.0,
                pose_by_side=_poses(x),
            )

        receipt = planner.prepare_staged_source_horizon()

        assert receipt["prepared"] is True
        assert receipt["last_source_index"] == 3
        assert receipt["new_source_solutions"] == 4
        assert receipt["prepare_ms"] >= 0.0
        assert sorted(planner._source_qpos) == [0, 1, 2, 3]
        assert planner._committed_source_index is None
        assert planner._state is not None
        np.testing.assert_array_equal(planner._state["left"].qd_rad_s, np.zeros(6))
        np.testing.assert_array_equal(planner._state["right"].qd_rad_s, np.zeros(6))
    finally:
        planner.close()


def test_governor_reset_receipt_records_effective_dynamic_envelope() -> None:
    planner = RM65OfficialAllIKGovernorPlanner(
        _CollisionPlanner(),
        endpoint_to_tool_by_side={
            "left": np.eye(4, dtype=float),
            "right": np.eye(4, dtype=float),
        },
        source_hz=15.0,
        command_hz=60.0,
        lookahead_source_steps=2,
        jerk_ramp_sec=0.05,
        max_acceleration_rad_s2=(28.0, 28.0, 36.0, 60.0, 60.0, 80.0),
        candidate_provider=_CandidateProvider(),
    )

    receipt = planner.reset({"left": np.zeros(6), "right": np.zeros(6)})

    assert receipt["governor_max_acceleration_rad_s2"] == pytest.approx(
        (28.0, 28.0, 36.0, 60.0, 60.0, 80.0)
    )
    assert receipt["governor_max_jerk_rad_s3"] == pytest.approx(
        (560.0, 560.0, 720.0, 1200.0, 1200.0, 1600.0)
    )
    assert receipt["governor_jerk_ramp_sec"] == pytest.approx(0.05)


def test_initial_horizon_preparation_preserves_first_formal_command() -> None:
    cold = _planner()
    prepared = _planner()
    try:
        for planner in (cold, prepared):
            planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
            for index, x in enumerate((0.0, 0.0002, 0.0004, 0.0006)):
                planner.stage_source_pose(
                    source_index=index,
                    source_timestamp=index / 15.0,
                    pose_by_side=_poses(x),
                )
        prepared.prepare_staged_source_horizon()

        cold_result = cold.plan_pose_reference_horizon(
            _indexed_reference(source_index=0, source_timestamp=0.0),
            source_timestamp=0.0,
            source="test",
        )
        prepared_result = prepared.plan_pose_reference_horizon(
            _indexed_reference(source_index=0, source_timestamp=0.0),
            source_timestamp=0.0,
            source="test",
        )

        assert cold_result.metadata["governor_execution_mode"] == (
            "formal_third_order_horizon"
        )
        assert prepared_result.metadata["governor_execution_mode"] == (
            "formal_third_order_horizon"
        )
        for side in ("left", "right"):
            np.testing.assert_allclose(
                prepared_result.q_ref_by_side[side],
                cold_result.q_ref_by_side[side],
                rtol=0.0,
                atol=1e-9,
            )
    finally:
        cold.close()
        prepared.close()


def test_continuous_waypoint_shadow_is_non_authoritative_and_preserves_qcmd() -> None:
    baseline = _planner()
    shadow = _planner(shadow=True)
    for planner in (baseline, shadow):
        planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
        for index, x in enumerate((0.0, 0.0002, 0.0004)):
            planner.stage_source_pose(
                source_index=index,
                source_timestamp=index / 15.0,
                pose_by_side=_poses(x),
            )

    baseline_result = baseline.plan_pose_reference_horizon(
        _reference(),
        source_timestamp=0.0,
        source="test",
    )
    shadow_result = shadow.plan_pose_reference_horizon(
        _reference(),
        source_timestamp=0.0,
        source="test",
    )

    assert "continuous_waypoint_shadow" not in baseline_result.metadata
    shadow_metadata = shadow_result.metadata["continuous_waypoint_shadow"]
    assert shadow_metadata["status"] == "ready"
    assert shadow_metadata["authoritative"] is False
    assert shadow_metadata["dispatch_selected"] is False
    assert shadow_metadata["formal_qcmd_unchanged"] is True
    assert shadow_metadata["collision_checked"] is False
    assert shadow_metadata["future_source_contract"] == (
        "already_staged_same_chunk_only"
    )
    for side in ("left", "right"):
        np.testing.assert_array_equal(
            shadow_result.q_ref_by_side[side],
            baseline_result.q_ref_by_side[side],
        )


def test_continuous_waypoint_shadow_error_does_not_fault_formal_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_shadow(*_args, **_kwargs):
        raise RuntimeError("shadow-only failure")

    monkeypatch.setattr(
        governor_module,
        "build_offline_quintic_waypoint_trajectory",
        fail_shadow,
    )
    planner = _planner(shadow=True)
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    for index, x in enumerate((0.0, 0.0002, 0.0004)):
        planner.stage_source_pose(
            source_index=index,
            source_timestamp=index / 15.0,
            pose_by_side=_poses(x),
        )

    result = planner.plan_pose_reference_horizon(
        _reference(),
        source_timestamp=0.0,
        source="test",
    )

    assert result.valid is True
    assert result.hold is False
    shadow_metadata = result.metadata["continuous_waypoint_shadow"]
    assert shadow_metadata["status"] == "error"
    assert shadow_metadata["error_type"] == "RuntimeError"
    assert shadow_metadata["error"] == "shadow-only failure"
    assert shadow_metadata["dispatch_selected"] is False


def test_continuous_waypoint_admission_selects_one_complete_source_cut() -> None:
    planner = _planner(admission=True)
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    for index, x in enumerate((0.0, 0.0002, 0.0004)):
        planner.stage_source_pose(
            source_index=index,
            source_timestamp=index / 15.0,
            pose_by_side=_poses(x),
        )

    first = planner.plan_pose_reference_horizon(
        _reference(),
        source_timestamp=0.0,
        source="test",
    )
    second = planner.plan_pose_reference_horizon(
        _reference(0.000025),
        source_timestamp=1.0 / 120.0,
        source="test",
    )

    for result in (first, second):
        assert result.metadata["governor_execution_mode"] == (
            "continuous_c2_waypoint"
        )
        shadow = result.metadata["continuous_waypoint_shadow"]
        assert shadow["admission_status"] == "admitted_for_complete_source_cut"
        assert shadow["authoritative"] is True
        assert shadow["dispatch_selected"] is True
        assert shadow["collision_admitted"] is True
        assert shadow["discrete_command_certificate_passed"] is True
        assert shadow["candidate_state_semantics"] == (
            "emitted_position_finite_difference"
        )
        assert shadow["selection_scope"] == "complete_source_cut"
        assert result.metadata["governor_execution_mode_legacy"] == (
            "experimental_continuous_c2_waypoint"
        )
        assert "continuous_c2_waypoint" in result.modification_reasons
        assert "third_order_horizon_governor" not in result.modification_reasons
    assert first.metadata["source_cut_replanned"] is True
    assert second.metadata["source_cut_replanned"] is False


def test_continuous_waypoint_collision_rejection_falls_back_for_whole_cut() -> None:
    baseline = _planner()
    rejected = _planner(
        admission=True,
        collision_planner=_CollisionPlanner(reject_continuous_admission=True),
    )
    for planner in (baseline, rejected):
        planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
        for index, x in enumerate((0.0, 0.0002, 0.0004)):
            planner.stage_source_pose(
                source_index=index,
                source_timestamp=index / 15.0,
                pose_by_side=_poses(x),
            )

    baseline_result = baseline.plan_pose_reference_horizon(
        _reference(),
        source_timestamp=0.0,
        source="test",
    )
    rejected_result = rejected.plan_pose_reference_horizon(
        _reference(),
        source_timestamp=0.0,
        source="test",
    )

    assert rejected_result.metadata["governor_execution_mode"] == (
        "formal_third_order_horizon"
    )
    shadow = rejected_result.metadata["continuous_waypoint_shadow"]
    assert shadow["admission_status"] == "formal_fallback_collision_rejected"
    assert shadow["authoritative"] is False
    assert shadow["dispatch_selected"] is False
    assert shadow["collision_checked"] is True
    assert shadow["collision_admitted"] is False
    assert shadow["collision_rejected_reason"] == (
        "synthetic continuous-path collision"
    )
    for side in ("left", "right"):
        np.testing.assert_array_equal(
            rejected_result.q_ref_by_side[side],
            baseline_result.q_ref_by_side[side],
        )


def test_continuous_waypoint_discrete_dynamic_rejection_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = np.zeros((17, 12), dtype=float)
    q[1:, 0] = 0.1
    synthetic = SimpleNamespace(
        q_rad=q,
        qd_rad_s=np.zeros_like(q),
        qdd_rad_s2=np.zeros_like(q),
        jerk_rad_s3=np.zeros_like(q),
        metadata={
            "planner": "synthetic_analytic_only",
            "continuous_hard_limit_certificate": True,
        },
    )
    monkeypatch.setattr(
        governor_module,
        "build_offline_quintic_waypoint_trajectory",
        lambda *_args, **_kwargs: synthetic,
    )
    planner = _planner(admission=True)
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    for index, x in enumerate((0.0, 0.0002, 0.0004)):
        planner.stage_source_pose(
            source_index=index,
            source_timestamp=index / 15.0,
            pose_by_side=_poses(x),
        )

    result = planner.plan_pose_reference_horizon(
        _reference(),
        source_timestamp=0.0,
        source="test",
    )

    assert result.metadata["governor_execution_mode"] == (
        "formal_third_order_horizon"
    )
    shadow = result.metadata["continuous_waypoint_shadow"]
    assert shadow["admission_status"] == (
        "formal_fallback_discrete_dynamic_rejected"
    )
    assert shadow["discrete_command_certificate_passed"] is False
    assert shadow["discrete_rejected_row"] == 1
    assert shadow["discrete_rejected_joint_index"] == 1
    assert shadow["discrete_rejected_quantity"] == "velocity"
    assert shadow["dispatch_selected"] is False


def test_async_precompute_consumes_only_complete_next_cut() -> None:
    planner = _planner(admission=True, async_precompute=True)
    try:
        planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
        for index, x in enumerate((0.0, 0.0002, 0.0004)):
            planner.stage_source_pose(
                source_index=index,
                source_timestamp=index / 15.0,
                pose_by_side=_poses(x),
            )

        first = planner.plan_pose_reference_horizon(
            _indexed_reference(source_index=0, source_timestamp=0.0),
            source_timestamp=0.0,
            source="test",
        )
        first_shadow = first.metadata["continuous_waypoint_shadow"]
        assert first.metadata["governor_execution_mode"] == (
            "formal_third_order_horizon"
        )
        assert first_shadow["status"] == "precompute_not_ready"
        assert first_shadow["admission_status"] == (
            "formal_fallback_candidate_unavailable"
        )

        planner.stage_source_pose(
            source_index=3,
            source_timestamp=3.0 / 15.0,
            pose_by_side=_poses(0.0006),
        )
        _run_remaining_source_cut(
            planner,
            interval_index=0,
            start_offset=1,
            stop_offset=1,
        )
        _wait_for_precompute_result(planner)
        _run_remaining_source_cut(
            planner,
            interval_index=0,
            start_offset=2,
        )

        next_cut = planner.plan_pose_reference_horizon(
            _indexed_reference(
                source_index=1,
                source_timestamp=1.0 / 15.0,
            ),
            source_timestamp=1.0 / 15.0,
            source="test",
        )
        shadow = next_cut.metadata["continuous_waypoint_shadow"]
        assert next_cut.metadata["governor_execution_mode"] == (
            "continuous_c2_waypoint"
        )
        assert shadow["precompute_consumed_async"] is True
        assert shadow["admission_status"] == "admitted_for_complete_source_cut"
        assert shadow["selection_scope"] == "complete_source_cut"
    finally:
        planner.close()


def test_async_precompute_anchor_mismatch_never_waits_or_partially_switches() -> None:
    planner = _planner(admission=True, async_precompute=True)
    try:
        planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
        for index, x in enumerate((0.0, 0.0002, 0.0004)):
            planner.stage_source_pose(
                source_index=index,
                source_timestamp=index / 15.0,
                pose_by_side=_poses(x),
            )
        planner.plan_pose_reference_horizon(
            _indexed_reference(source_index=0, source_timestamp=0.0),
            source_timestamp=0.0,
            source="test",
        )
        planner.stage_source_pose(
            source_index=3,
            source_timestamp=3.0 / 15.0,
            pose_by_side=_poses(0.0006),
        )
        _run_remaining_source_cut(
            planner,
            interval_index=0,
            start_offset=1,
            stop_offset=1,
        )
        _wait_for_precompute_result(planner)
        # Deliberately skip the final 120 Hz emission.  The actual admitted
        # state no longer equals the worker's predicted next-cut anchor.
        _run_remaining_source_cut(
            planner,
            interval_index=0,
            start_offset=2,
            stop_offset=6,
        )

        started = time.perf_counter()
        result = planner.plan_pose_reference_horizon(
            _indexed_reference(
                source_index=1,
                source_timestamp=1.0 / 15.0,
            ),
            source_timestamp=1.0 / 15.0,
            source="test",
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        shadow = result.metadata["continuous_waypoint_shadow"]
        assert result.metadata["governor_execution_mode"] == (
            "formal_third_order_horizon"
        )
        assert shadow["status"] == "precompute_not_ready"
        assert shadow["precompute_miss_reason"] == (
            "not_submitted_or_stale_anchor"
        )
        assert shadow["dispatch_selected"] is False
        assert elapsed_ms < 100.0
    finally:
        planner.close()


def test_async_precompute_late_result_falls_back_and_reset_discards_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    original = governor_module.build_offline_quintic_waypoint_trajectory

    def delayed_build(*args, **kwargs):
        started.set()
        if not release.wait(timeout=1.0):
            raise RuntimeError("test precompute release timeout")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        governor_module,
        "build_offline_quintic_waypoint_trajectory",
        delayed_build,
    )
    planner = _planner(admission=True, async_precompute=True)
    try:
        planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
        for index, x in enumerate((0.0, 0.0002, 0.0004)):
            planner.stage_source_pose(
                source_index=index,
                source_timestamp=index / 15.0,
                pose_by_side=_poses(x),
            )
        planner.plan_pose_reference_horizon(
            _indexed_reference(source_index=0, source_timestamp=0.0),
            source_timestamp=0.0,
            source="test",
        )
        planner.stage_source_pose(
            source_index=3,
            source_timestamp=3.0 / 15.0,
            pose_by_side=_poses(0.0006),
        )
        _run_remaining_source_cut(
            planner,
            interval_index=0,
            start_offset=1,
            stop_offset=1,
        )
        assert started.wait(timeout=1.0)
        _run_remaining_source_cut(
            planner,
            interval_index=0,
            start_offset=2,
        )

        result = planner.plan_pose_reference_horizon(
            _indexed_reference(
                source_index=1,
                source_timestamp=1.0 / 15.0,
            ),
            source_timestamp=1.0 / 15.0,
            source="test",
        )
        shadow = result.metadata["continuous_waypoint_shadow"]
        assert result.metadata["governor_execution_mode"] == (
            "formal_third_order_horizon"
        )
        assert shadow["precompute_miss_reason"] == "running"

        old_epoch = planner._continuous_precompute_epoch
        planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
        release.set()
        deadline = time.monotonic() + 1.0
        with planner._continuous_precompute_condition:
            while planner._continuous_precompute_running_key is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise AssertionError("late precompute did not finish")
                planner._continuous_precompute_condition.wait(timeout=remaining)
            assert planner._continuous_precompute_epoch == old_epoch + 1
            assert not planner._continuous_precompute_results
    finally:
        release.set()
        planner.close()


def test_async_precompute_build_error_is_non_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        governor_module,
        "build_offline_quintic_waypoint_trajectory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic async proposal failure")
        ),
    )
    planner = _planner(admission=True, async_precompute=True)
    try:
        planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
        for index, x in enumerate((0.0, 0.0002, 0.0004)):
            planner.stage_source_pose(
                source_index=index,
                source_timestamp=index / 15.0,
                pose_by_side=_poses(x),
            )
        planner.plan_pose_reference_horizon(
            _indexed_reference(source_index=0, source_timestamp=0.0),
            source_timestamp=0.0,
            source="test",
        )
        planner.stage_source_pose(
            source_index=3,
            source_timestamp=3.0 / 15.0,
            pose_by_side=_poses(0.0006),
        )
        _run_remaining_source_cut(
            planner,
            interval_index=0,
            start_offset=1,
            stop_offset=1,
        )
        _wait_for_precompute_result(planner)
        _run_remaining_source_cut(
            planner,
            interval_index=0,
            start_offset=2,
        )

        result = planner.plan_pose_reference_horizon(
            _indexed_reference(
                source_index=1,
                source_timestamp=1.0 / 15.0,
            ),
            source_timestamp=1.0 / 15.0,
            source="test",
        )
        shadow = result.metadata["continuous_waypoint_shadow"]
        assert result.valid is True
        assert result.metadata["governor_execution_mode"] == (
            "formal_third_order_horizon"
        )
        assert shadow["status"] == "error"
        assert shadow["error_type"] == "RuntimeError"
        assert shadow["error"] == "synthetic async proposal failure"
        assert shadow["dispatch_selected"] is False
    finally:
        planner.close()


def test_batched_certificate_failure_uses_certified_non_batched_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _planner()
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    for index, x in enumerate((0.0, 0.0002, 0.0004)):
        planner.stage_source_pose(
            source_index=index,
            source_timestamp=index / 15.0,
            pose_by_side=_poses(x),
        )

    def fail_certificate(*_args, **_kwargs):
        raise RuntimeError(
            "reused batched joint horizon candidate failed certification "
            "for joint 9."
        )

    monkeypatch.setattr(
        planner._governor_workspace,
        "solve",
        fail_certificate,
    )
    result = planner.plan_pose_reference_horizon(
        _reference(),
        source_timestamp=0.0,
        source="test",
    )

    assert result.valid is True
    assert result.hold is False
    assert result.metadata["governor_batched_workspace_fallback"] is True
    assert "joint 9" in result.metadata["governor_batched_workspace_failure"]
    assert result.metadata["governor_solver"] == (
        "jerk_limited_joint_horizon_qp_v2_jerk_only"
    )
    assert np.all(np.isfinite(result.q_ref_by_side["left"]))
    assert np.all(np.isfinite(result.q_ref_by_side["right"]))


def test_batched_non_certificate_failure_remains_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _planner()
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    for index, x in enumerate((0.0, 0.0002, 0.0004)):
        planner.stage_source_pose(
            source_index=index,
            source_timestamp=index / 15.0,
            pose_by_side=_poses(x),
        )

    def fail_solver(*_args, **_kwargs):
        raise RuntimeError("reused batched joint horizon QP failed: infeasible")

    monkeypatch.setattr(planner._governor_workspace, "solve", fail_solver)
    with pytest.raises(RuntimeError, match="QP failed: infeasible"):
        planner.plan_pose_reference_horizon(
            _reference(),
            source_timestamp=0.0,
            source="test",
        )
