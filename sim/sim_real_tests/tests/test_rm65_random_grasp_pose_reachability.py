from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from robot_motion.rm65.grasp_reachability import (
    CollisionCandidateAssessment,
    CollisionCandidateStatus,
    ConservativeWorkspaceEnvelope,
    GraspReachabilityStatus,
    IKCandidate,
    evaluate_grasp_pose_reachability,
)
from sim.sim_real_tests.rm65 import (
    DEFAULT_REFERENCE_FRAME_ID,
    DEFAULT_REFERENCE_FRAME_SPEC_HASH,
    RM65RandomGraspPoseReachabilityConfig,
    RandomGraspPoseSample,
    build_rm65_multistart_candidate_provider,
    compute_target_pose_content_hash,
    generate_random_grasp_pose_samples,
    run_random_grasp_pose_reachability,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _pose(xyz: tuple[float, float, float] = (0.3, 0.0, 0.4)) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = xyz
    return pose


def _accepted_candidate(candidate_index: int, _target: np.ndarray) -> IKCandidate:
    return IKCandidate(
        candidate_index=candidate_index,
        qpos_rad=(0.0,) * 6,
        converged=True,
        position_residual_m=1e-5,
        orientation_residual_rad=1e-5,
        iterations=3,
    )


def _evaluate(
    target_pose: np.ndarray,
    *,
    candidate_provider=_accepted_candidate,
    collision_checker=None,
    max_candidates: int = 2,
    reference_frame_id: str = DEFAULT_REFERENCE_FRAME_ID,
    reference_frame_spec_hash: str = DEFAULT_REFERENCE_FRAME_SPEC_HASH,
    target_pose_content_hash: str | None = None,
):
    return evaluate_grasp_pose_reachability(
        target_pose,
        reference_frame_id=reference_frame_id,
        reference_frame_spec_hash=reference_frame_spec_hash,
        candidate_provider=candidate_provider,
        max_candidates=max_candidates,
        maximum_position_residual_m=0.003,
        maximum_orientation_residual_rad=np.deg2rad(1.0),
        workspace_envelope=ConservativeWorkspaceEnvelope(
            reference_frame_id=DEFAULT_REFERENCE_FRAME_ID,
            reference_frame_spec_hash=DEFAULT_REFERENCE_FRAME_SPEC_HASH,
            maximum_radius_m=1.25,
        ),
        collision_checker=collision_checker,
        target_pose_content_hash=target_pose_content_hash,
    )


def test_pcg64_shoemake_generation_is_exactly_deterministic_without_rejection() -> None:
    config = RM65RandomGraspPoseReachabilityConfig(seed=37, sample_count=8)

    first = generate_random_grasp_pose_samples(config)
    second = generate_random_grasp_pose_samples(config)

    assert len(first) == len(second) == 8
    assert [sample.sample_index for sample in first] == list(range(8))
    for lhs, rhs in zip(first, second):
        np.testing.assert_array_equal(lhs.target_pose, rhs.target_pose)
        rotation = lhs.target_pose[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)
        assert lhs.reference_frame_id == DEFAULT_REFERENCE_FRAME_ID
        assert lhs.reference_frame_spec_hash == DEFAULT_REFERENCE_FRAME_SPEC_HASH
        assert lhs.target_pose_content_hash == rhs.target_pose_content_hash
        assert lhs.target_pose_content_hash == compute_target_pose_content_hash(
            lhs.target_pose,
            sampler_id=lhs.sampler_id,
            seed=lhs.seed,
            sample_index=lhs.sample_index,
            reference_frame_id=lhs.reference_frame_id,
            reference_frame_spec_hash=lhs.reference_frame_spec_hash,
        )


def test_target_pose_content_hash_binds_seed_index_sampler_and_frame() -> None:
    sample = generate_random_grasp_pose_samples(
        RM65RandomGraspPoseReachabilityConfig(seed=37, sample_count=1)
    )[0]
    baseline = sample.target_pose_content_hash

    variants = (
        {"seed": sample.seed + 1},
        {"sample_index": sample.sample_index + 1},
        {"sampler_id": "different-sampler"},
        {"reference_frame_id": "different-frame"},
        {"reference_frame_spec_hash": "1" * 64},
    )
    for variant in variants:
        arguments = {
            "sampler_id": sample.sampler_id,
            "seed": sample.seed,
            "sample_index": sample.sample_index,
            "reference_frame_id": sample.reference_frame_id,
            "reference_frame_spec_hash": sample.reference_frame_spec_hash,
            **variant,
        }
        assert compute_target_pose_content_hash(sample.target_pose, **arguments) != baseline


def test_bounded_multistart_exhaustion_is_not_reported_as_unreachable() -> None:
    def rejected(candidate_index: int, _target: np.ndarray) -> IKCandidate:
        return IKCandidate(
            candidate_index=candidate_index,
            qpos_rad=(0.0,) * 6,
            converged=False,
            position_residual_m=0.2,
            orientation_residual_rad=0.5,
        )

    receipt = _evaluate(_pose(), candidate_provider=rejected, max_candidates=3)

    assert receipt.status is GraspReachabilityStatus.IK_SEARCH_EXHAUSTED
    assert receipt.proof_scope == "bounded_multistart_search_only"
    assert len(receipt.candidates) == 3


def test_mixed_provider_error_and_nonconvergence_is_incomplete_planner_error() -> None:
    def mixed(candidate_index: int, _target: np.ndarray) -> IKCandidate:
        if candidate_index == 0:
            raise RuntimeError("synthetic infrastructure failure")
        return IKCandidate(
            candidate_index=candidate_index,
            qpos_rad=(0.0,) * 6,
            converged=False,
            position_residual_m=0.2,
            orientation_residual_rad=0.5,
        )

    receipt = _evaluate(_pose(), candidate_provider=mixed, max_candidates=3)

    assert receipt.status is GraspReachabilityStatus.PLANNER_ERROR
    assert receipt.reason_code == "candidate_search_incomplete_due_to_provider_error"
    assert receipt.proof_scope == "incomplete_candidate_adapter_search"
    assert receipt.metadata["provider_error_count"] == 1


def test_valid_candidate_without_collision_checker_is_endpoint_reachable_only() -> None:
    receipt = _evaluate(_pose())

    assert receipt.status is GraspReachabilityStatus.REACHABLE
    assert receipt.proof_scope == "endpoint_kinematic_candidate_only"
    assert receipt.metadata["collision_checked"] is False


def test_invalid_pose_and_adapter_failure_are_distinct_terminal_statuses() -> None:
    invalid = _evaluate(np.eye(3, dtype=float))

    def failing_provider(_candidate_index: int, _target: np.ndarray) -> IKCandidate:
        raise RuntimeError("synthetic provider failure")

    failed = _evaluate(_pose(), candidate_provider=failing_provider, max_candidates=2)

    assert invalid.status is GraspReachabilityStatus.INVALID_TARGET
    assert failed.status is GraspReachabilityStatus.PLANNER_ERROR
    assert failed.metadata["provider_error_count"] == 2


def test_clearly_outside_conservative_workspace_is_proven_unreachable() -> None:
    provider_called = False

    def must_not_run(candidate_index: int, target: np.ndarray) -> IKCandidate:
        nonlocal provider_called
        provider_called = True
        return _accepted_candidate(candidate_index, target)

    receipt = _evaluate(_pose((3.0, 0.0, 0.0)), candidate_provider=must_not_run)

    assert receipt.status is GraspReachabilityStatus.KINEMATICALLY_UNREACHABLE
    assert receipt.proof_scope == "conservative_workspace_exclusion"
    assert receipt.metadata["origin_distance_m"] == pytest.approx(3.0)
    assert not provider_called


def test_workspace_exclusion_requires_exact_target_frame_identity() -> None:
    receipt = _evaluate(
        _pose((3.0, 0.0, 0.0)),
        reference_frame_id="different.arm.base",
    )

    assert receipt.status is GraspReachabilityStatus.INVALID_TARGET
    assert receipt.reason_code == "workspace_and_target_reference_frame_mismatch"
    assert receipt.proof_scope == "reference_frame_identity_validation"


def test_explicitly_blocked_candidates_project_to_collision_blocked() -> None:
    def blocked(candidate: IKCandidate, _target: np.ndarray) -> CollisionCandidateAssessment:
        return CollisionCandidateAssessment(
            status=CollisionCandidateStatus.BLOCKED,
            reason_code="cube_overlap",
            minimum_distance_m=-0.01,
            required_distance_m=0.02,
            obstacle_ids=(f"cube-{candidate.candidate_index}",),
        )

    receipt = _evaluate(_pose(), collision_checker=blocked, max_candidates=2)

    assert receipt.status is GraspReachabilityStatus.COLLISION_BLOCKED
    assert len(receipt.collision_assessments) == 2
    assert all(item["status"] == "blocked" for item in receipt.collision_assessments)


def test_mixed_blocked_and_indeterminate_projects_to_collision_search_exhausted() -> None:
    def collision(
        candidate: IKCandidate, _target: np.ndarray
    ) -> CollisionCandidateAssessment:
        if candidate.candidate_index == 0:
            return CollisionCandidateAssessment(
                status=CollisionCandidateStatus.BLOCKED,
                reason_code="cube_overlap",
            )
        return CollisionCandidateAssessment(
            status=CollisionCandidateStatus.SEARCH_EXHAUSTED,
            reason_code="collision_budget_exhausted",
        )

    receipt = _evaluate(_pose(), collision_checker=collision, max_candidates=2)

    assert receipt.status is GraspReachabilityStatus.COLLISION_SEARCH_EXHAUSTED
    assert receipt.proof_scope == "bounded_endpoint_collision_search_only"


def test_every_random_sample_yields_a_terminal_record_and_unreachable_is_not_failure() -> None:
    config = RM65RandomGraspPoseReachabilityConfig(
        seed=11,
        sample_count=5,
        position_min_m=(2.9, -0.1, -0.1),
        position_max_m=(3.1, 0.1, 0.1),
    )

    def evaluator(sample: RandomGraspPoseSample):
        return _evaluate(
            sample.target_pose,
            reference_frame_id=sample.reference_frame_id,
            reference_frame_spec_hash=sample.reference_frame_spec_hash,
            target_pose_content_hash=sample.target_pose_content_hash,
        )

    result = run_random_grasp_pose_reachability(config, evaluator=evaluator)

    assert result.passed
    assert result.metrics["terminal_records"] == 5
    assert result.metrics["status_kinematically_unreachable"] == 5
    assert len(result.records) == 5
    assert all(
        record["reference_frame_id"] == DEFAULT_REFERENCE_FRAME_ID
        and record["reference_frame_spec_hash"] == DEFAULT_REFERENCE_FRAME_SPEC_HASH
        and record["target_pose_content_hash"]
        for record in result.records
    )


def test_scenario_rejects_receipt_hash_reused_for_a_different_target_pose() -> None:
    config = RM65RandomGraspPoseReachabilityConfig(seed=11, sample_count=1)

    def substituted_target(sample: RandomGraspPoseSample):
        target = sample.target_pose.copy()
        target[:3, 3] = (3.0, 0.0, 0.0)
        return _evaluate(
            target,
            reference_frame_id=sample.reference_frame_id,
            reference_frame_spec_hash=sample.reference_frame_spec_hash,
            target_pose_content_hash=sample.target_pose_content_hash,
        )

    result = run_random_grasp_pose_reachability(config, evaluator=substituted_target)

    assert not result.passed
    assert result.metrics["status_planner_error"] == 1
    assert result.records[0]["reason_code"] == "scenario_evaluator_error"


def test_scenario_adapter_error_is_terminal_but_fails_the_scenario() -> None:
    config = RM65RandomGraspPoseReachabilityConfig(seed=11, sample_count=2)

    def broken_evaluator(_sample: RandomGraspPoseSample):
        raise RuntimeError("synthetic evaluator error")

    result = run_random_grasp_pose_reachability(config, evaluator=broken_evaluator)

    assert not result.passed
    assert result.metrics["terminal_records"] == 2
    assert result.metrics["status_planner_error"] == 2
    assert all(record["status"] == "planner_error" for record in result.records)


def test_terminal_records_do_not_contain_execution_trajectory_fields() -> None:
    receipts = (
        _evaluate(_pose()),
        _evaluate(_pose((3.0, 0.0, 0.0))),
        _evaluate(
            _pose(),
            collision_checker=lambda _candidate, _target: CollisionCandidateAssessment(
                status=CollisionCandidateStatus.BLOCKED,
                reason_code="blocked",
            ),
        ),
    )
    forbidden = {"trajectory", "path", "steps", "waypoints", "commands", "plan"}

    for receipt in receipts:
        assert not (_recursive_keys(receipt.to_dict()) & forbidden)


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for nested in value.values():
            keys.update(_recursive_keys(nested))
        return keys
    if isinstance(value, (list, tuple)):
        keys: set[str] = set()
        for nested in value:
            keys.update(_recursive_keys(nested))
        return keys
    return set()


@pytest.mark.integration
@pytest.mark.optional
def test_known_rm65_fk_pose_is_reachable_from_its_generating_qpos() -> None:
    pytest.importorskip("pinocchio")
    from robot_motion.rm65.ik import RM65SingleArmIKConfig, build_rm65_single_arm_ik_solver

    solver = build_rm65_single_arm_ik_solver(
        RM65SingleArmIKConfig(controlled_frame="tool_tcp")
    )
    generating_qpos = np.clip(
        np.zeros(solver.num_dofs, dtype=float),
        np.asarray(solver.q_min, dtype=float),
        np.asarray(solver.q_max, dtype=float),
    )
    target_pose = solver.compute_pose(generating_qpos)
    provider = build_rm65_multistart_candidate_provider(
        solver,
        qpos_seeds=(generating_qpos,),
        max_iterations=2,
        max_joint_step_rad=0.1,
        damping=0.03,
    )

    receipt = evaluate_grasp_pose_reachability(
        target_pose,
        reference_frame_id=DEFAULT_REFERENCE_FRAME_ID,
        reference_frame_spec_hash=DEFAULT_REFERENCE_FRAME_SPEC_HASH,
        candidate_provider=provider,
        max_candidates=1,
        maximum_position_residual_m=1e-8,
        maximum_orientation_residual_rad=1e-8,
        workspace_envelope=ConservativeWorkspaceEnvelope(
            reference_frame_id=DEFAULT_REFERENCE_FRAME_ID,
            reference_frame_spec_hash=DEFAULT_REFERENCE_FRAME_SPEC_HASH,
            maximum_radius_m=1.25,
        ),
    )

    assert receipt.status is GraspReachabilityStatus.REACHABLE
    assert receipt.selected_candidate_index == 0
    assert receipt.candidates[0]["position_residual_m"] <= 1e-8
    assert receipt.candidates[0]["orientation_residual_rad"] <= 1e-8


@pytest.mark.integration
@pytest.mark.optional
def test_actual_rm65_adapter_classifies_every_arbitrary_pose_once() -> None:
    pytest.importorskip("pinocchio")
    config = RM65RandomGraspPoseReachabilityConfig(
        seed=91,
        sample_count=3,
        max_ik_candidates=2,
        ik_max_iterations=40,
    )

    result = run_random_grasp_pose_reachability(config)

    assert result.passed, result.to_dict()
    assert result.metrics["terminal_records"] == 3
    assert len(result.records) == 3
    assert [record["sample_index"] for record in result.records] == [0, 1, 2]
    assert all(
        record["status"]
        in {
            "reachable",
            "kinematically_unreachable",
            "ik_search_exhausted",
            "collision_blocked",
            "collision_search_exhausted",
        }
        for record in result.records
    )
