from __future__ import annotations

import numpy as np

from hardware.rm65.ik_provider import RM65SDKIKProvider
from robot_motion.pinocchio_provider import PinocchioIKProvider


class FakeRM65Robot:
    def __init__(self, joint):
        self.joint = list(joint)

    def Algo_Inverse_Kinematics(self, q_in, pose, flag=1):
        return 0, list(self.joint)

    def Algo_IKine_Check_Joint_Position_Limit(self, joint):
        values = list(joint)
        for index, value in enumerate(values):
            if value < -180.0 or value > 180.0:
                return index + 1
        return 0


def test_rm65_ik_provider_rejects_single_solution_outside_position_limit():
    provider = RM65SDKIKProvider(
        FakeRM65Robot([0.0, 0.0, 0.0, 178.0, 32.0, -206.0]),
        use_all_solutions=False,
        use_quaternion_fallback=False,
    )

    candidates = provider.iter_ik_candidates(
        target_pose=np.eye(4),
        seeds=(np.zeros(6),),
        seed_labels=("seed",),
        options={},
    )

    assert candidates == []


def test_rm65_ik_provider_keeps_single_solution_inside_position_limit():
    provider = RM65SDKIKProvider(
        FakeRM65Robot([0.0, 0.0, 0.0, 90.0, 32.0, -120.0]),
        use_all_solutions=False,
        use_quaternion_fallback=False,
    )

    candidates = provider.iter_ik_candidates(
        target_pose=np.eye(4),
        seeds=(np.zeros(6),),
        seed_labels=("seed",),
        options={},
    )

    assert len(candidates) == 1
    assert np.allclose(np.rad2deg(candidates[0].qpos), [0.0, 0.0, 0.0, 90.0, 32.0, -120.0])


class FakePinocchioSolver:
    last_branch_selection = "original"
    last_branch_distance_score_deg = 12.5

    def solve_iterative(self, target_pose, q_init, max_iterations, max_joint_step, damping, tolerance):
        self.last_target_pose = np.asarray(target_pose, dtype=float).copy()
        return np.asarray(q_init, dtype=float) + np.deg2rad(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]))

    def compute_pose(self, qpos):
        return np.eye(4)


def test_pinocchio_provider_uses_solver_candidates():
    provider = PinocchioIKProvider(solver=FakePinocchioSolver(), source="rm65_pinocchio", qpos_size=6)

    candidates = provider.iter_ik_candidates(
        target_pose=np.eye(4),
        seeds=(np.zeros(6),),
        seed_labels=("seed",),
        options={"max_candidate_count": 4},
    )

    assert len(candidates) == 1
    assert candidates[0].source == "rm65_pinocchio"
    assert candidates[0].metadata["branch_selection"] == "original"
    assert np.allclose(np.rad2deg(candidates[0].qpos), [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])


def test_pinocchio_provider_applies_target_to_eef_transform():
    solver = FakePinocchioSolver()
    target_to_eef = np.eye(4)
    target_to_eef[2, 3] = 0.009
    provider = PinocchioIKProvider(
        solver=solver,
        source="rm65_pinocchio",
        target_to_eef_transform=target_to_eef,
        qpos_size=6,
    )

    provider.iter_ik_candidates(
        target_pose=np.eye(4),
        seeds=(np.zeros(6),),
        seed_labels=("seed",),
        options={"max_candidate_count": 4},
    )

    assert np.isclose(solver.last_target_pose[2, 3], 0.009)
