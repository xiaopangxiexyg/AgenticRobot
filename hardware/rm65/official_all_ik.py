"""Device-free adapter for the bundled RM65 official all-solution IK.

The adapter owns only the vendor algorithm handle.  It does not enumerate,
connect to, or command a robot.  Runtime authority, collision certification,
branch selection, timing and dispatch remain with their respective owners.
"""

from __future__ import annotations

import threading
from typing import Any, Sequence

import numpy as np

from robot_motion.pose import matrix_to_rm_quat_pose
from robot_motion.rm65 import RM65_ABSOLUTE_Q_MAX_RAD, RM65_ABSOLUTE_Q_MIN_RAD


class RM65OfficialAllIK:
    """Enumerate normalized RM65 IK candidates near a supplied seed."""

    def __init__(self, *, limit_tolerance_deg: float = 0.002) -> None:
        tolerance = float(limit_tolerance_deg)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("limit_tolerance_deg must be finite and non-negative.")
        self._limit_tolerance_deg = tolerance
        self._algo: Any | None = None
        self._lock = threading.Lock()

    def _algorithm(self) -> Any:
        if self._algo is None:
            from hardware.rm65.Robotic_Arm.rm_ctypes_wrap import (
                rm_force_type_e,
                rm_robot_arm_model_e,
            )
            from hardware.rm65.Robotic_Arm.rm_robot_interface import Algo

            # Algo is a local kinematics handle.  It does not open a network
            # connection or enumerate hardware.
            self._algo = Algo(
                rm_robot_arm_model_e.RM_MODEL_RM_65_E,
                rm_force_type_e.RM_MODEL_RM_B_E,
            )
        return self._algo

    def _normalize_near_seed(
        self,
        candidate_deg: Sequence[float],
        seed_rad: np.ndarray,
    ) -> np.ndarray | None:
        raw = np.asarray(candidate_deg, dtype=float).reshape(-1)
        if raw.size < 6 or not np.all(np.isfinite(raw[:6])):
            return None
        seed_deg = np.degrees(seed_rad)
        lower = np.degrees(np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float))
        upper = np.degrees(np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float))
        normalized = np.empty(6, dtype=float)
        for joint_index, angle_deg in enumerate(raw[:6]):
            equivalents = np.asarray(
                [float(angle_deg) + 360.0 * offset for offset in range(-2, 3)],
                dtype=float,
            )
            valid = equivalents[
                (
                    equivalents
                    >= lower[joint_index] - self._limit_tolerance_deg
                )
                & (
                    equivalents
                    <= upper[joint_index] + self._limit_tolerance_deg
                )
            ]
            if valid.size == 0:
                return None
            selected = valid[int(np.argmin(np.abs(valid - seed_deg[joint_index])))]
            normalized[joint_index] = np.clip(
                selected,
                lower[joint_index],
                upper[joint_index],
            )
        return np.radians(normalized)

    def solve_candidates(
        self,
        *,
        seed_rad: Sequence[float],
        target_endpoint: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Return unique, in-limit solutions in deterministic vendor order."""

        from hardware.rm65.Robotic_Arm.rm_ctypes_wrap import (
            rm_inverse_kinematics_params_t,
        )

        seed = np.asarray(seed_rad, dtype=float).reshape(6)
        target = np.asarray(target_endpoint, dtype=float).reshape(4, 4)
        if not np.all(np.isfinite(seed)) or not np.all(np.isfinite(target)):
            raise ValueError("seed_rad and target_endpoint must be finite.")
        params = rm_inverse_kinematics_params_t(
            np.degrees(seed).tolist(),
            matrix_to_rm_quat_pose(target),
            0,
        )
        # The bundled algorithm library has process-global native state.
        # Serialize access even though the production planner currently has
        # one reference worker.
        with self._lock:
            result = self._algorithm().rm_algo_inverse_kinematics_all(params)
        if int(result.result) != 0 or int(result.num) <= 0:
            return ()
        candidates: list[np.ndarray] = []
        for candidate_index in range(min(8, int(result.num))):
            normalized = self._normalize_near_seed(
                result.q_solve[candidate_index],
                seed,
            )
            if normalized is not None and not any(
                np.allclose(normalized, existing, atol=1e-8, rtol=0.0)
                for existing in candidates
            ):
                candidates.append(normalized)
        return tuple(candidates)


__all__ = ["RM65OfficialAllIK"]
