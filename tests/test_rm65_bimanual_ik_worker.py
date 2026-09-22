from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from robot_motion.rm65 import (
    BimanualIKResult,
    BimanualIKSideResult,
    BimanualPoseTarget,
    ThreadedBimanualIKProposalWorker,
    assess_bimanual_ik_proposal,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _target(marker: float) -> BimanualPoseTarget:
    left = np.eye(4, dtype=float)
    right = np.eye(4, dtype=float)
    left[0, 3] = marker
    right[0, 3] = -marker
    return BimanualPoseTarget(
        timestamp=marker,
        target_pose_by_side={"left": left, "right": right},
    )


def _result(target: BimanualPoseTarget) -> BimanualIKResult:
    marker = float(target.target_pose_by_side["left"][0, 3])
    side_results = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        qpos = np.full(6, sign * marker, dtype=float)
        pose = np.asarray(target.target_pose_by_side[side], dtype=float)
        side_results[side] = BimanualIKSideResult(
            side=side,
            qpos=qpos,
            raw_qpos=qpos,
            solver_pose_world=pose,
            tool_tcp_pose_world=pose,
            command_type="joint_vector",
            command_value=qpos,
            command_frame_name="arm_base",
        )
    return BimanualIKResult(
        timestamp=target.timestamp,
        status="accepted",
        fallback_reason=None,
        side_results=side_results,
        metadata={"marker": marker},
    )


def _qpos(value: float) -> dict[str, np.ndarray]:
    return {
        "left": np.full(6, value, dtype=float),
        "right": np.full(6, -value, dtype=float),
    }


class _BlockingSolver:
    def __init__(self, *, block_first: bool = False, fail_marker: float | None = None):
        self.block_first = block_first
        self.fail_marker = fail_marker
        self.first_entered = threading.Event()
        self.release_first = threading.Event()
        self.calls: list[float] = []
        self.qpos_rebases: list[tuple[bool, float]] = []
        self.smoothing_resets = 0

    def set_current_qpos_pair(
        self,
        *,
        left_qpos=None,
        right_qpos=None,
        reset_tracking=True,
    ) -> None:
        left = np.asarray(left_qpos, dtype=float)
        right = np.asarray(right_qpos, dtype=float)
        assert left.shape == right.shape == (6,)
        self.qpos_rebases.append((bool(reset_tracking), float(left[0])))

    def reset_target_pose_smoothing(self) -> None:
        self.smoothing_resets += 1

    def solve_target_pair(self, target: BimanualPoseTarget) -> BimanualIKResult:
        marker = float(target.target_pose_by_side["left"][0, 3])
        self.calls.append(marker)
        if self.block_first and len(self.calls) == 1:
            self.first_entered.set()
            if not self.release_first.wait(timeout=2.0):
                raise TimeoutError("test did not release the first solve")
        if self.fail_marker is not None and marker == self.fail_marker:
            raise RuntimeError("synthetic coupled IK failure")
        return _result(target)


def _wait_for_completion(
    worker: ThreadedBimanualIKProposalWorker,
    *,
    timeout_sec: float = 1.0,
):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        completion = worker.poll_completion()
        if completion is not None:
            return completion
        time.sleep(0.001)
    raise AssertionError(worker.status())


def _reset_ready(
    worker: ThreadedBimanualIKProposalWorker,
    *,
    epoch: int,
    qpos_value: float = 0.0,
) -> int:
    generation = worker.reset(
        control_epoch=epoch,
        current_qpos_by_side=_qpos(qpos_value),
    )
    assert worker.wait_until_ready(generation, timeout_sec=1.0)
    return generation


def test_latest_pending_request_replaces_backlog_and_preserves_input_copy() -> None:
    solver = _BlockingSolver(block_first=True)
    worker = ThreadedBimanualIKProposalWorker(solver)
    worker.start()
    try:
        generation = _reset_ready(worker, epoch=7)
        now = time.monotonic()
        worker.submit(
            _target(1.0),
            control_epoch=7,
            anchor_generation=10,
            reference_qpos_by_side=_qpos(0.1),
            input_state_timestamp=now,
            valid_until=now + 2.0,
        )
        assert solver.first_entered.wait(timeout=1.0)

        now = time.monotonic()
        worker.submit(
            _target(2.0),
            control_epoch=7,
            anchor_generation=11,
            reference_qpos_by_side=_qpos(0.2),
            input_state_timestamp=now,
            valid_until=now + 2.0,
        )
        latest_target = _target(3.0)
        latest_qpos = _qpos(0.3)
        now = time.monotonic()
        latest_request = worker.submit(
            latest_target,
            control_epoch=7,
            anchor_generation=12,
            reference_qpos_by_side=latest_qpos,
            input_state_timestamp=now,
            valid_until=now + 2.0,
        )
        latest_target.target_pose_by_side["left"][0, 3] = 99.0
        latest_qpos["left"].fill(99.0)
        solver.release_first.set()

        deadline = time.monotonic() + 1.0
        while worker.status()["completed_count"] < 2:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        completion = _wait_for_completion(worker)
        assert solver.calls == [1.0, 3.0]
        assert completion.request.request_sequence == latest_request.request_sequence
        assert completion.request.anchor_generation == 12
        assert completion.request.reference_qpos_by_side["left"][0] == pytest.approx(0.3)
        assert completion.result is not None
        assert completion.result.metadata["marker"] == pytest.approx(3.0)

        assessment = assess_bimanual_ik_proposal(
            completion,
            now=time.monotonic(),
            required_worker_generation=generation,
            required_control_epoch=7,
            required_anchor_generation=12,
            minimum_request_sequence=latest_request.request_sequence,
        )
        assert assessment.eligible_for_admission
        assert assessment.reasons == ()
        status = worker.status()
        assert status["pending_superseded_count"] == 1
        assert status["completion_superseded_count"] == 1
    finally:
        solver.release_first.set()
        worker.close()


def test_reset_is_a_priority_barrier_and_discards_old_generation() -> None:
    solver = _BlockingSolver(block_first=True)
    worker = ThreadedBimanualIKProposalWorker(solver)
    worker.start()
    try:
        first_generation = _reset_ready(worker, epoch=1, qpos_value=0.0)
        now = time.monotonic()
        worker.submit(
            _target(1.0),
            control_epoch=1,
            anchor_generation=1,
            reference_qpos_by_side=_qpos(0.1),
            input_state_timestamp=now,
            valid_until=now + 2.0,
        )
        assert solver.first_entered.wait(timeout=1.0)

        second_generation = worker.reset(
            control_epoch=2,
            current_qpos_by_side=_qpos(0.5),
        )
        assert second_generation == first_generation + 1
        now = time.monotonic()
        worker.submit(
            _target(2.0),
            control_epoch=2,
            anchor_generation=5,
            reference_qpos_by_side=_qpos(0.6),
            input_state_timestamp=now,
            valid_until=now + 2.0,
        )
        solver.release_first.set()

        assert worker.wait_until_ready(second_generation, timeout_sec=1.0)
        completion = _wait_for_completion(worker)
        assert completion.request.worker_generation == second_generation
        assert completion.request.control_epoch == 2
        assert solver.calls == [1.0, 2.0]
        assert solver.qpos_rebases == [
            (True, 0.0),
            (False, 0.1),
            (True, 0.5),
            (False, 0.6),
        ]
        status = worker.status()
        assert status["discarded_stale_completion_count"] == 1
        assert status["reset_count"] == 2
    finally:
        solver.release_first.set()
        worker.close()


def test_transport_assessment_rejects_expired_or_wrong_anchor_result() -> None:
    solver = _BlockingSolver()
    worker = ThreadedBimanualIKProposalWorker(solver)
    worker.start()
    try:
        generation = _reset_ready(worker, epoch=4)
        now = time.monotonic()
        request = worker.submit(
            _target(4.0),
            control_epoch=4,
            anchor_generation=9,
            reference_qpos_by_side=_qpos(0.4),
            input_state_timestamp=now,
            valid_until=now + 0.2,
        )
        completion = _wait_for_completion(worker)
        wrong_anchor = assess_bimanual_ik_proposal(
            completion,
            now=completion.completed_at,
            required_worker_generation=generation,
            required_control_epoch=4,
            required_anchor_generation=10,
        )
        assert not wrong_anchor.eligible_for_admission
        assert wrong_anchor.reasons == ("anchor_generation_mismatch",)
        expired = assess_bimanual_ik_proposal(
            completion,
            now=request.valid_until + 0.001,
            required_worker_generation=generation,
            required_control_epoch=4,
            required_anchor_generation=9,
        )
        assert not expired.eligible_for_admission
        assert expired.reasons == ("result_expired",)
    finally:
        worker.close()


def test_solver_error_is_returned_as_non_admissible_completion() -> None:
    solver = _BlockingSolver(fail_marker=5.0)
    worker = ThreadedBimanualIKProposalWorker(solver)
    worker.start()
    try:
        generation = _reset_ready(worker, epoch=5)
        now = time.monotonic()
        worker.submit(
            _target(5.0),
            control_epoch=5,
            anchor_generation=2,
            reference_qpos_by_side=_qpos(0.5),
            input_state_timestamp=now,
            valid_until=now + 1.0,
        )
        completion = _wait_for_completion(worker)
        assert isinstance(completion.error, RuntimeError)
        assessment = assess_bimanual_ik_proposal(
            completion,
            now=completion.completed_at,
            required_worker_generation=generation,
            required_control_epoch=5,
            required_anchor_generation=2,
        )
        assert not assessment.eligible_for_admission
        assert assessment.reasons == ("solver_error", "result_missing")
        assert worker.status()["failed_count"] == 1
    finally:
        worker.close()


def test_submit_requires_successful_reset_and_matching_control_epoch() -> None:
    solver = _BlockingSolver()
    worker = ThreadedBimanualIKProposalWorker(solver)
    worker.start()
    try:
        now = time.monotonic()
        with pytest.raises(RuntimeError, match="successful reset"):
            worker.submit(
                _target(1.0),
                control_epoch=0,
                anchor_generation=0,
                reference_qpos_by_side=_qpos(0.0),
                input_state_timestamp=now,
                valid_until=now + 1.0,
            )
        _reset_ready(worker, epoch=3)
        now = time.monotonic()
        with pytest.raises(ValueError, match="control_epoch"):
            worker.submit(
                _target(1.0),
                control_epoch=2,
                anchor_generation=0,
                reference_qpos_by_side=_qpos(0.0),
                input_state_timestamp=now,
                valid_until=now + 1.0,
            )
    finally:
        worker.close()
