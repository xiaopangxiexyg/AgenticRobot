from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from .ik_tuning import (
    RM65IKTuningConfig,
    generate_rm65_wrist_flip_candidates,
    load_ik_tuning_config,
    weighted_joint_distance_score,
)
from robot_motion.rm65.assets import require_asset_path


class BoundedIKNumericalError(RuntimeError):
    """A query's FK/residual was non-finite; no candidate may be admitted."""


def _to_pose_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (7,):
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = R.from_quat(pose[3:7]).as_matrix()
        matrix[:3, 3] = pose[:3]
        return matrix
    raise ValueError(f"Unsupported pose shape: {pose.shape}")


class SingleArmIKSolver:
    def __init__(
        self,
        urdf_path: str | Path | None = None,
        eef_name: str = "hand_base",
        ik_tuning: RM65IKTuningConfig | None = None,
    ):
        try:
            import pinocchio as pin
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SingleArmIKSolver requires pinocchio to be importable."
            ) from exc

        if urdf_path is None:
            urdf_path = require_asset_path("rm65_arm.urdf")

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.eef_name = eef_name
        self.frame_names = tuple(frame.name for frame in self.model.frames)
        self.eef_frame_id = self.model.getFrameId(eef_name)
        if self.eef_frame_id >= len(self.model.frames):
            raise ValueError(
                f"Frame '{eef_name}' was not found in URDF '{urdf_path}'. "
                f"Available frames include: {self.frame_names[:20]}"
            )
        self.num_dofs = self.model.nv
        self.q_min = self.model.lowerPositionLimit.copy()
        self.q_max = self.model.upperPositionLimit.copy()
        self.ik_tuning = ik_tuning if ik_tuning is not None else load_ik_tuning_config()
        objective = self.ik_tuning.objective_weights()
        self._position_weight = float(objective.single_arm_position_weight)
        self._orientation_weight = float(objective.single_arm_orientation_weight)
        self._joint_distance_weights = np.asarray(
            self.ik_tuning.branch_normalization.joint_distance_weights,
            dtype=float,
        )
        self.last_branch_selection = "original"
        self.last_branch_distance_score_deg = 0.0
        self.last_solve_iterations = 0
        self.last_solve_converged = False
        self.last_solve_deadline_exceeded = False

    def compute_pose(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=float)
        self.pin.forwardKinematics(self.model, self.data, qpos)
        self.pin.updateFramePlacements(self.model, self.data)
        transform = self.data.oMf[self.eef_frame_id]
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = transform.rotation
        pose[:3, 3] = transform.translation
        return pose

    def compute_jacobian(self, qpos: np.ndarray, reference_frame: int | None = None) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=float)
        if reference_frame is None:
            reference_frame = self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        self.pin.computeJointJacobians(self.model, self.data, qpos)
        return self.pin.computeFrameJacobian(
            self.model,
            self.data,
            qpos,
            self.eef_frame_id,
            reference_frame,
        )

    def solve_step(
        self,
        target_pose: np.ndarray,
        current_qpos: np.ndarray,
        max_joint_step: float | np.ndarray = 0.05,
        damping: float = 0.1,
    ) -> np.ndarray:
        current_qpos = np.asarray(current_qpos, dtype=float)
        raw_qpos = self._solve_step_raw(
            target_pose=target_pose,
            current_qpos=current_qpos,
            max_joint_step=max_joint_step,
            damping=damping,
        )
        return self._normalize_branch(
            candidate_qpos=raw_qpos,
            reference_qpos=current_qpos,
        )

    def _solve_step_raw(
        self,
        target_pose: np.ndarray,
        current_qpos: np.ndarray,
        max_joint_step: float | np.ndarray = 0.05,
        damping: float = 0.1,
    ) -> np.ndarray:
        target_pose = _to_pose_matrix(target_pose)
        current_qpos = np.asarray(current_qpos, dtype=float)

        current_pose = self.compute_pose(current_qpos)
        pos_err = target_pose[:3, 3] - current_pose[:3, 3]
        rot_err = self.pin.log3(target_pose[:3, :3] @ current_pose[:3, :3].T)
        error = np.concatenate([pos_err, rot_err])

        jacobian = self.compute_jacobian(current_qpos)
        weighting = np.diag(
            [
                self._position_weight,
                self._position_weight,
                self._position_weight,
                self._orientation_weight,
                self._orientation_weight,
                self._orientation_weight,
            ]
        )
        weighted_error = weighting @ error
        weighted_jacobian = weighting @ jacobian
        regularizer = (damping**2) * np.eye(6, dtype=float)
        joint_update = weighted_jacobian.T @ np.linalg.solve(
            weighted_jacobian @ weighted_jacobian.T + regularizer,
            weighted_error,
        )

        max_joint_step = np.asarray(max_joint_step, dtype=float)
        if max_joint_step.shape == ():
            max_joint_step = np.full(self.num_dofs, float(max_joint_step), dtype=float)
        target_qpos = current_qpos + np.clip(joint_update, -max_joint_step, max_joint_step)
        return np.clip(target_qpos, self.q_min, self.q_max)

    def solve_iterative(
        self,
        target_pose: np.ndarray,
        q_init: np.ndarray | None = None,
        max_iterations: int = 100,
        max_joint_step: float | np.ndarray = 0.1,
        damping: float = 0.1,
        tolerance: float = 1e-4,
        deadline_monotonic: float | None = None,
        max_wall_time_sec: float | None = None,
    ) -> np.ndarray:
        target_pose = _to_pose_matrix(target_pose)
        start_time = time.perf_counter()
        deadline = _solver_deadline(
            started_at=start_time,
            deadline_monotonic=deadline_monotonic,
            max_wall_time_sec=max_wall_time_sec,
        )
        reference_q = (
            np.zeros(self.num_dofs, dtype=float)
            if q_init is None
            else np.asarray(q_init, dtype=float)
        )
        q = reference_q.copy()
        self.last_solve_iterations = 0
        self.last_solve_converged = False
        self.last_solve_deadline_exceeded = False

        for iteration in range(max_iterations):
            if _deadline_exceeded(deadline):
                self.last_solve_deadline_exceeded = True
                break
            current_pose = self.compute_pose(q)
            pos_err = target_pose[:3, 3] - current_pose[:3, 3]
            rot_err = self.pin.log3(target_pose[:3, :3] @ current_pose[:3, :3].T)
            error = np.concatenate([pos_err, rot_err])
            self.last_solve_iterations = iteration + 1
            if np.linalg.norm(error) < tolerance:
                self.last_solve_converged = True
                break
            q = self._solve_step_raw(
                target_pose=target_pose,
                current_qpos=q,
                max_joint_step=max_joint_step,
                damping=damping,
            )
            if _deadline_exceeded(deadline):
                self.last_solve_deadline_exceeded = True
                break
        return self._normalize_branch(candidate_qpos=q, reference_qpos=reference_q)

    def solve_bounded_pose(
        self,
        target_pose: np.ndarray,
        q_init: np.ndarray,
        *,
        controlled_frame_offset: np.ndarray,
        translation_tolerance_m: float,
        rotation_tolerance_rad: float,
        deadline_monotonic: float,
        cancel_requested: Callable[[], bool],
        maximum_evaluations: int = 100,
    ) -> np.ndarray:
        """Query one bounded endpoint candidate; never authorize its execution.

        Trust-region least squares handles active joint limits without clipping
        an unconstrained update. Residuals use the caller's actual TCP, not its
        flange. Callers still own FK, collision, path and dispatch admission.
        This explicit offline-query path does not change solve_step/iterative.
        """
        from scipy.optimize import least_squares

        goal = _to_pose_matrix(target_pose)
        offset = _to_pose_matrix(controlled_frame_offset)
        for pose in (goal, offset):
            if (not np.isfinite(pose).all()
                    or not np.allclose(pose[3], (0., 0., 0., 1.), atol=1e-9)
                    or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
                    or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-6)):
                raise ValueError("Bounded pose query requires rigid finite transforms.")
        seed = np.asarray(q_init, dtype=float).copy()
        if (seed.shape != self.q_min.shape or not np.isfinite(seed).all()
                or np.any(seed < self.q_min) or np.any(seed > self.q_max)):
            raise ValueError("Bounded pose seed must satisfy the original joint bounds.")
        tolerances = np.asarray((translation_tolerance_m, rotation_tolerance_rad), dtype=float)
        if not np.isfinite(tolerances).all() or np.any(tolerances <= 0):
            raise ValueError("Bounded pose tolerances must be finite and positive.")
        if (isinstance(maximum_evaluations, bool) or not isinstance(maximum_evaluations, int)
                or maximum_evaluations <= 0 or isinstance(deadline_monotonic, bool)
                or not np.isfinite(deadline_monotonic)):
            raise ValueError("Bounded pose query requires a finite deadline and evaluation budget.")
        if not callable(cancel_requested):
            raise TypeError("cancel_requested must be callable.")
        self.last_solve_iterations = 0
        self.last_solve_converged = False
        self.last_solve_deadline_exceeded = False
        self.last_solve_residual_evaluations = 0
        best_q = seed.copy()
        best_score = float("inf")

        class DeadlineExceeded(Exception):
            pass

        def residual(q):
            nonlocal best_q, best_score
            if cancel_requested():
                raise InterruptedError("Bounded endpoint query cancelled.")
            if time.perf_counter() >= deadline_monotonic:
                raise DeadlineExceeded
            tcp = self.compute_pose(q) @ offset
            if not np.isfinite(tcp).all():
                raise BoundedIKNumericalError("Bounded endpoint FK is non-finite.")
            value = np.concatenate(((goal[:3, 3] - tcp[:3, 3]) / tolerances[0],
                self.pin.log3(goal[:3, :3] @ tcp[:3, :3].T) / tolerances[1]))
            if not np.isfinite(value).all():
                raise BoundedIKNumericalError("Bounded endpoint residual is non-finite.")
            self.last_solve_residual_evaluations += 1
            score = float(max(np.linalg.norm(value[:3]), np.linalg.norm(value[3:])))
            if score < best_score:
                best_q, best_score = np.asarray(q).copy(), score
            return value

        try:
            solved = least_squares(residual, seed, bounds=(self.q_min, self.q_max),
                max_nfev=maximum_evaluations, xtol=1e-9, ftol=1e-9, gtol=1e-9)
            self.last_solve_iterations = int(solved.nfev)
        except DeadlineExceeded:
            self.last_solve_deadline_exceeded = True
        self.last_solve_converged = best_score <= 1.0
        # Do not normalize to another branch after evaluating this candidate.
        return best_q

    def _normalize_branch(
        self,
        *,
        candidate_qpos: np.ndarray,
        reference_qpos: np.ndarray,
    ) -> np.ndarray:
        candidate = np.asarray(candidate_qpos, dtype=float).reshape(-1)
        reference = np.asarray(reference_qpos, dtype=float).reshape(-1)
        if not self.ik_tuning.branch_normalization.enabled:
            self.last_branch_selection = "disabled"
            self.last_branch_distance_score_deg = weighted_joint_distance_score(
                reference_q=reference,
                candidate_q=candidate,
                joint_distance_weights=self._joint_distance_weights,
            )
            return candidate.copy()

        candidates = generate_rm65_wrist_flip_candidates(
            candidate,
            q_min=self.q_min,
            q_max=self.q_max,
            limit_tolerance_rad=float(self.ik_tuning.branch_normalization.limit_tolerance_rad),
        )
        best = min(
            candidates,
            key=lambda q: weighted_joint_distance_score(
                reference_q=reference,
                candidate_q=q,
                joint_distance_weights=self._joint_distance_weights,
            ),
        )
        self.last_branch_selection = (
            "original" if np.allclose(best, candidate, atol=1e-8) else "wrist_flip"
        )
        self.last_branch_distance_score_deg = weighted_joint_distance_score(
            reference_q=reference,
            candidate_q=best,
            joint_distance_weights=self._joint_distance_weights,
        )
        return np.asarray(best, dtype=float).copy()


def _solver_deadline(
    *,
    started_at: float,
    deadline_monotonic: float | None,
    max_wall_time_sec: float | None,
) -> float | None:
    deadlines: list[float] = []
    try:
        deadline = float(deadline_monotonic)
    except (TypeError, ValueError):
        deadline = 0.0
    if np.isfinite(deadline) and deadline > 0.0:
        deadlines.append(deadline)
    try:
        max_wall = float(max_wall_time_sec)
    except (TypeError, ValueError):
        max_wall = 0.0
    if np.isfinite(max_wall) and max_wall > 0.0:
        deadlines.append(float(started_at) + max_wall)
    return min(deadlines) if deadlines else None


def _deadline_exceeded(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= float(deadline)


__all__ = ["SingleArmIKSolver"]
