from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .oracle import CollisionSafetyOracle
from .policy import CollisionSafetyPolicy
from .result import CollisionCheckResult


@dataclass(frozen=True)
class CollisionProjectionResult:
    status: str
    requested_qpos: np.ndarray
    accepted_qpos: np.ndarray
    projection_alpha: float
    modified: bool
    check: CollisionCheckResult
    sweep_samples: int
    blocking_check: CollisionCheckResult | None = None

    def to_dict(self) -> dict[str, object]:
        values = {
            "status": self.status,
            "modified": bool(self.modified),
            "projection_alpha": float(self.projection_alpha),
            "sweep_samples": int(self.sweep_samples),
            **self.check.to_dict(),
        }
        if self.blocking_check is not None:
            blocking = self.blocking_check.to_dict()
            values.update(
                {
                    "blocking_minimum_distance_m": blocking["minimum_distance_m"],
                    "blocking_penetration_depth_m": blocking["penetration_depth_m"],
                    "blocking_geom1": blocking["geom1"],
                    "blocking_geom2": blocking["geom2"],
                    "blocking_body1": blocking["body1"],
                    "blocking_body2": blocking["body2"],
                    "blocking_pair_class": blocking["pair_class"],
                    "blocking_collision_category": blocking[
                        "collision_category"
                    ],
                }
            )
        return values


class KinematicCollisionProjector:
    """Project a prescribed qpos target onto the last collision-safe segment."""

    def __init__(
        self,
        mujoco,
        model,
        controlled_qpos_indices: np.ndarray,
        policy: CollisionSafetyPolicy | None = None,
    ) -> None:
        self.mujoco = mujoco
        self.model = model
        self.policy = policy or CollisionSafetyPolicy()
        self.controlled_qpos_indices = np.asarray(controlled_qpos_indices, dtype=np.int32)
        self.oracle = CollisionSafetyOracle(mujoco, model, self.policy)
        self.scratch = mujoco.MjData(model)

    def _check_candidate(
        self,
        source_data,
        controlled_qpos: np.ndarray,
        *,
        minimum_clearance_m: float | None = None,
    ) -> CollisionCheckResult:
        self.scratch.qpos[:] = source_data.qpos
        self.scratch.qpos[self.controlled_qpos_indices] = controlled_qpos
        # Distance queries only require body/geom poses. Running the full
        # dynamics/constraint pipeline here would dominate batched RL cost.
        self.mujoco.mj_kinematics(self.model, self.scratch)
        return self.oracle.check(
            self.scratch,
            minimum_clearance_m=minimum_clearance_m,
        )

    def check_current(self, data) -> CollisionCheckResult:
        return self.oracle.check(data)

    def check_qpos(
        self,
        source_data,
        controlled_qpos: np.ndarray,
        *,
        minimum_clearance_m: float | None = None,
    ) -> CollisionCheckResult:
        values = np.asarray(controlled_qpos, dtype=np.float64).reshape(-1)
        if values.shape != (self.controlled_qpos_indices.size,):
            raise ValueError("Controlled qpos shape does not match projector layout.")
        return self._check_candidate(
            source_data,
            values,
            minimum_clearance_m=minimum_clearance_m,
        )

    def project(
        self,
        source_data,
        *,
        start_qpos: np.ndarray,
        requested_qpos: np.ndarray,
        minimum_clearance_m: float | None = None,
    ) -> CollisionProjectionResult:
        start = np.asarray(start_qpos, dtype=np.float64).reshape(-1)
        requested = np.asarray(requested_qpos, dtype=np.float64).reshape(-1)
        expected = self.controlled_qpos_indices.size
        if start.shape != (expected,) or requested.shape != (expected,):
            raise ValueError(f"Collision projection expects {expected} controlled qpos values.")
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(requested)):
            raise ValueError("Collision projection qpos must be finite.")

        start_check = self._check_candidate(
            source_data,
            start,
            minimum_clearance_m=minimum_clearance_m,
        )
        if not start_check.safe:
            return CollisionProjectionResult(
                status="invalid_safe_state",
                requested_qpos=requested.copy(),
                accepted_qpos=start.copy(),
                projection_alpha=0.0,
                modified=True,
                check=start_check,
                sweep_samples=0,
                blocking_check=start_check,
            )

        delta = requested - start
        sample_count = max(
            1,
            int(np.ceil(np.max(np.abs(delta)) / self.policy.sweep_max_joint_step_rad)),
        )
        lower_alpha = 0.0
        lower_check = start_check
        upper_alpha: float | None = None
        upper_check: CollisionCheckResult | None = None
        for sample_index in range(1, sample_count + 1):
            alpha = sample_index / sample_count
            check = self._check_candidate(
                source_data,
                start + alpha * delta,
                minimum_clearance_m=minimum_clearance_m,
            )
            if not check.safe:
                upper_alpha = alpha
                upper_check = check
                break
            lower_alpha = alpha
            lower_check = check

        if upper_alpha is None:
            return CollisionProjectionResult(
                status="accepted",
                requested_qpos=requested.copy(),
                accepted_qpos=requested.copy(),
                projection_alpha=1.0,
                modified=False,
                check=lower_check,
                sweep_samples=sample_count,
            )

        for _ in range(self.policy.projection_iterations):
            alpha = 0.5 * (lower_alpha + upper_alpha)
            check = self._check_candidate(
                source_data,
                start + alpha * delta,
                minimum_clearance_m=minimum_clearance_m,
            )
            if check.safe:
                lower_alpha = alpha
                lower_check = check
            else:
                upper_alpha = alpha
                upper_check = check
        assert upper_check is not None
        accepted = start + lower_alpha * delta
        return CollisionProjectionResult(
            status="projected" if lower_alpha > 0.0 else "rejected",
            requested_qpos=requested.copy(),
            accepted_qpos=accepted,
            projection_alpha=float(lower_alpha),
            modified=True,
            check=lower_check,
            sweep_samples=sample_count,
            blocking_check=upper_check,
        )

    def recover_from_clearance_violation(
        self,
        source_data,
        *,
        start_qpos: np.ndarray,
        requested_qpos: np.ndarray,
    ) -> CollisionProjectionResult:
        """Allow only a monotonic retreat from a soft clearance violation.

        Physical actuator tracking can place the measured state inside the
        preventive clearance envelope without exceeding the hard penetration
        limit. A projector that requires a safe start would then reject even a
        command that retreats from contact. This recovery path samples the
        complete segment and accepts it only when distance never worsens, hard
        penetration is never crossed, and the requested endpoint is safe.
        """

        start = np.asarray(start_qpos, dtype=np.float64).reshape(-1)
        requested = np.asarray(requested_qpos, dtype=np.float64).reshape(-1)
        expected = self.controlled_qpos_indices.size
        if start.shape != (expected,) or requested.shape != (expected,):
            raise ValueError(f"Collision recovery expects {expected} controlled qpos values.")
        start_check = self._check_candidate(source_data, start)
        if start_check.safe:
            return self.project(
                source_data,
                start_qpos=start,
                requested_qpos=requested,
            )
        delta = requested - start
        sample_count = max(
            1,
            int(np.ceil(np.max(np.abs(delta)) / self.policy.sweep_max_joint_step_rad)),
        )
        previous_distance = float(start_check.minimum_distance_m)
        last_check = start_check
        tolerance = 1e-9
        for sample_index in range(1, sample_count + 1):
            alpha = sample_index / sample_count
            check = self._check_candidate(source_data, start + alpha * delta)
            distance = float(check.minimum_distance_m)
            if (
                check.penetration_depth_m > self.policy.hard_penetration_m
                or distance + tolerance < previous_distance
            ):
                return CollisionProjectionResult(
                    status="invalid_safe_state",
                    requested_qpos=requested.copy(),
                    accepted_qpos=start.copy(),
                    projection_alpha=0.0,
                    modified=True,
                    check=start_check,
                    sweep_samples=sample_index,
                    blocking_check=check,
                )
            previous_distance = distance
            last_check = check
        if not last_check.safe:
            return CollisionProjectionResult(
                status="invalid_safe_state",
                requested_qpos=requested.copy(),
                accepted_qpos=start.copy(),
                projection_alpha=0.0,
                modified=True,
                check=start_check,
                sweep_samples=sample_count,
                blocking_check=last_check,
            )
        return CollisionProjectionResult(
            status="recovered",
            requested_qpos=requested.copy(),
            accepted_qpos=requested.copy(),
            projection_alpha=1.0,
            modified=False,
            check=last_check,
            sweep_samples=sample_count,
            blocking_check=start_check,
        )
