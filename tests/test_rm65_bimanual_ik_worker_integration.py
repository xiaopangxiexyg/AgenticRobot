from __future__ import annotations

import time

import numpy as np
import pytest

from robot_motion.rm65 import (
    RM65_ASSET_DIR,
    BimanualPoseTarget,
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    ThreadedBimanualIKProposalWorker,
    assess_bimanual_ik_proposal,
)

pytestmark = pytest.mark.integration


def _wait_for_completion(
    worker: ThreadedBimanualIKProposalWorker,
    *,
    timeout_sec: float,
):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        completion = worker.poll_completion()
        if completion is not None:
            return completion
        time.sleep(0.001)
    raise AssertionError(worker.status())


def test_real_coupled_solver_runs_behind_latest_only_worker() -> None:
    pytest.importorskip("pinocchio")
    pytest.importorskip("pink")
    pytest.importorskip("qpsolvers")
    if not (RM65_ASSET_DIR / "rm65_dual_arm_hand_proxy_open.urdf").is_file():
        pytest.skip("RM65 development assets are unavailable")

    solver = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            bimanual_result_contract_enabled=True,
            solve_time_budget_ms=50.0,
        )
    )
    qpos_by_side = solver.current_qpos_by_side()
    target = BimanualPoseTarget(
        timestamp=time.monotonic(),
        target_pose_by_side=solver.current_world_targets(),
    )
    worker = ThreadedBimanualIKProposalWorker(solver)
    worker.start()
    try:
        generation = worker.reset(
            control_epoch=1,
            current_qpos_by_side=qpos_by_side,
        )
        assert worker.wait_until_ready(generation, timeout_sec=2.0)
        submitted_at = time.monotonic()
        request = worker.submit(
            target,
            control_epoch=1,
            anchor_generation=1,
            reference_qpos_by_side=qpos_by_side,
            input_state_timestamp=submitted_at,
            valid_until=submitted_at + 2.0,
        )

        completion = _wait_for_completion(worker, timeout_sec=2.0)

        assert completion.error is None
        assessment = assess_bimanual_ik_proposal(
            completion,
            now=completion.completed_at,
            required_worker_generation=generation,
            required_control_epoch=1,
            required_anchor_generation=1,
            current_reference_qpos_by_side=qpos_by_side,
            max_reference_drift_rad=0.0,
            minimum_request_sequence=request.request_sequence,
        )
        assert assessment.eligible_for_admission
        assert assessment.result is not None
        for side in ("left", "right"):
            assert np.all(np.isfinite(assessment.result.side_results[side].qpos))
    finally:
        worker.close()
