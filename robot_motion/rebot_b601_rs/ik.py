"""Pure continuous-seed inverse kinematics for the B601-RS arm.

The solver depends only on the reviewed :class:`B601PinocchioModel` contract.
It does not read MuJoCo state, import vendor code, or dispatch commands.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.transforms import validate_pose_matrix

from .model import B601PinocchioModel


@dataclass(frozen=True)
class B601IKConfig:
    """Numerical settings for one bounded damped-least-squares solve."""

    max_iterations: int = 120
    position_tolerance_m: float = 1e-5
    orientation_tolerance_rad: float = 1e-4
    damping: float = 2e-3
    # This is a numerical IK step, not a hardware command or a simulation
    # actuator velocity limit.  The previous 0.05 rad default could clip one
    # dominant wrist correction so severely that every backtracking candidate
    # increased the merit and the solver stalled at iteration zero.  Keep
    # physical joint limits enforced below, while allowing a normal reverse
    # teleoperation target to make progress in one DLS update.
    max_joint_step_rad: float = 0.10
    backtracking_steps: int = 8
    characteristic_length_m: float = 0.30
    joint_limit_margin_rad: float = 1e-6

    def __post_init__(self) -> None:
        if isinstance(self.max_iterations, bool):
            raise TypeError("max_iterations must be a positive integer")
        iterations = int(self.max_iterations)
        if iterations != self.max_iterations or iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        object.__setattr__(self, "max_iterations", iterations)
        if isinstance(self.backtracking_steps, bool):
            raise TypeError("backtracking_steps must be a positive integer")
        backtracking_steps = int(self.backtracking_steps)
        if backtracking_steps != self.backtracking_steps or backtracking_steps <= 0:
            raise ValueError("backtracking_steps must be a positive integer")
        object.__setattr__(self, "backtracking_steps", backtracking_steps)
        positive = {
            "position_tolerance_m": self.position_tolerance_m,
            "orientation_tolerance_rad": self.orientation_tolerance_rad,
            "damping": self.damping,
            "max_joint_step_rad": self.max_joint_step_rad,
            "characteristic_length_m": self.characteristic_length_m,
        }
        for name, value in positive.items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        margin = float(self.joint_limit_margin_rad)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("joint_limit_margin_rad must be finite and non-negative")


@dataclass(frozen=True)
class B601IKResult:
    """Immutable outcome of one pose solve.

    A failed result is diagnostic output, not a command. Callers must retain
    their last accepted target unless :attr:`success` is true.
    """

    success: bool
    status: str
    iterations: int
    qpos: tuple[float, ...]
    position_error_m: float
    orientation_error_rad: float
    seed_distance_rad: float
    limit_projection_applied: bool
    sigma_min: float
    condition_number: float


class B601DampedLeastSquaresIK:
    """Joint-limit-aware B601 IK with an explicit seed on every call."""

    def __init__(
        self,
        model: B601PinocchioModel,
        config: B601IKConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config if config is not None else B601IKConfig()
        lower, upper = self.model.arm_position_limits()
        margin = float(self.config.joint_limit_margin_rad)
        self._lower = lower + margin
        self._upper = upper - margin
        if np.any(self._lower >= self._upper):
            raise ValueError("joint_limit_margin_rad leaves an empty joint interval")

    def solve(
        self,
        target_pose: np.ndarray,
        *,
        seed_qpos: Sequence[float] | np.ndarray,
    ) -> B601IKResult:
        """Solve ``base_link_T_gripper_end`` from one explicit arm seed."""

        target = self._target_pose(target_pose)
        seed = self._seed(seed_qpos)
        qpos = np.clip(seed, self._lower, self._upper)
        projected = not np.array_equal(qpos, seed)

        position_error = float("inf")
        orientation_error = float("inf")
        status = "max_iterations"
        completed_iterations = 0

        for iteration in range(int(self.config.max_iterations) + 1):
            current = self.model.forward_kinematics(qpos)
            position_delta = target[:3, 3] - current[:3, 3]
            rotation_delta = Rotation.from_matrix(
                target[:3, :3] @ current[:3, :3].T
            ).as_rotvec()
            position_error = float(np.linalg.norm(position_delta))
            orientation_error = float(np.linalg.norm(rotation_delta))

            if position_error <= float(
                self.config.position_tolerance_m
            ) and orientation_error <= float(self.config.orientation_tolerance_rad):
                status = "converged"
                return self._result(
                    success=True,
                    status=status,
                    iterations=iteration,
                    qpos=qpos,
                    seed=seed,
                    position_error=position_error,
                    orientation_error=orientation_error,
                    projected=projected,
                )
            if iteration == int(self.config.max_iterations):
                completed_iterations = iteration
                break

            jacobian = self.model.arm_jacobian(qpos)
            length = float(self.config.characteristic_length_m)
            conditioned_jacobian = jacobian.copy()
            conditioned_jacobian[:3, :] /= length
            error = np.concatenate((position_delta / length, rotation_delta))
            merit = float(error @ error)
            damping = float(self.config.damping) * max(
                1.0,
                math.sqrt(merit) * 10.0,
            )
            lhs = conditioned_jacobian @ conditioned_jacobian.T
            lhs += damping * np.eye(lhs.shape[0])
            try:
                delta_q = conditioned_jacobian.T @ np.linalg.solve(lhs, error)
            except np.linalg.LinAlgError:
                status = "numerical_failure"
                completed_iterations = iteration
                break
            if not np.all(np.isfinite(delta_q)):
                status = "numerical_failure"
                completed_iterations = iteration
                break

            # The pinned upstream solver uses adaptive damping plus a short
            # backtracking search. Keep that numerical strategy here without
            # importing its package-level hardware dependencies.
            step = float(self.config.max_joint_step_rad)
            bounded_delta = np.clip(delta_q, -step, step)
            accepted = False
            alpha = 1.0
            for _ in range(int(self.config.backtracking_steps)):
                candidate = qpos + alpha * bounded_delta
                clipped = np.clip(candidate, self._lower, self._upper)
                candidate_pose = self.model.forward_kinematics(clipped)
                candidate_position_delta = target[:3, 3] - candidate_pose[:3, 3]
                candidate_rotation_delta = Rotation.from_matrix(
                    target[:3, :3] @ candidate_pose[:3, :3].T
                ).as_rotvec()
                candidate_error = np.concatenate(
                    (
                        candidate_position_delta / length,
                        candidate_rotation_delta,
                    )
                )
                if float(candidate_error @ candidate_error) < merit:
                    projected = projected or not np.array_equal(
                        candidate,
                        clipped,
                    )
                    qpos = clipped
                    accepted = True
                    completed_iterations = iteration + 1
                    break
                alpha *= 0.5
            if not accepted:
                status = "line_search_stalled"
                completed_iterations = iteration
                break

        return self._result(
            success=False,
            status=status,
            iterations=completed_iterations,
            qpos=qpos,
            seed=seed,
            position_error=position_error,
            orientation_error=orientation_error,
            projected=projected,
        )

    def _result(
        self,
        *,
        success: bool,
        status: str,
        iterations: int,
        qpos: np.ndarray,
        seed: np.ndarray,
        position_error: float,
        orientation_error: float,
        projected: bool,
    ) -> B601IKResult:
        metrics = self.model.singularity_metrics(
            qpos,
            characteristic_length_m=float(self.config.characteristic_length_m),
        )
        return B601IKResult(
            success=bool(success),
            status=str(status),
            iterations=int(iterations),
            qpos=tuple(float(value) for value in qpos),
            position_error_m=float(position_error),
            orientation_error_rad=float(orientation_error),
            seed_distance_rad=float(np.linalg.norm(qpos - seed)),
            limit_projection_applied=bool(projected),
            sigma_min=float(metrics.sigma_min),
            condition_number=float(metrics.condition_number),
        )

    def _target_pose(self, value: np.ndarray) -> np.ndarray:
        try:
            pose = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_pose must be a numeric 4x4 matrix") from exc
        if pose.shape != (4, 4):
            raise ValueError(f"target_pose must have shape (4, 4), got {pose.shape}")
        valid, reason = validate_pose_matrix(pose, atol=1e-8)
        if not valid:
            raise ValueError(f"invalid target_pose: {reason}")
        return pose.copy()

    def _seed(self, value: Sequence[float] | np.ndarray) -> np.ndarray:
        try:
            seed = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("seed_qpos must be a numeric sequence") from exc
        if seed.shape != (6,):
            raise ValueError(f"seed_qpos must have shape (6,), got {seed.shape}")
        if not np.all(np.isfinite(seed)):
            raise ValueError("seed_qpos contains non-finite values")
        physical_lower, physical_upper = self.model.arm_position_limits()
        if np.any(seed < physical_lower) or np.any(seed > physical_upper):
            raise ValueError("seed_qpos exceeds B601-RS arm position limits")
        return seed.copy()


__all__ = [
    "B601DampedLeastSquaresIK",
    "B601IKConfig",
    "B601IKResult",
]
