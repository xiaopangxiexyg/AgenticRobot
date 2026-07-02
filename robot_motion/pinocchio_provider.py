from __future__ import annotations

import inspect
import time
from typing import Any

import numpy as np

from .types import IKCandidate, SingularityReport


def _as_float_array(values: object, *, expected_len: int | None, label: str) -> np.ndarray:
    if hasattr(values, "tolist"):
        values = values.tolist()
    try:
        array = np.asarray(list(values), dtype=float).reshape(-1)
    except Exception as exc:  # noqa: BLE001 - adapters should annotate bad solver payloads.
        raise ValueError(f"{label} is not a numeric sequence") from exc
    if expected_len is not None and array.shape != (expected_len,):
        raise ValueError(f"{label} must contain {expected_len} values, got shape={array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def _dedupe_append(candidates: list[IKCandidate], candidate: IKCandidate) -> bool:
    qpos = np.asarray(candidate.qpos, dtype=float).reshape(-1)
    for existing in candidates:
        if np.allclose(qpos, existing.qpos, atol=1e-8):
            return False
    candidates.append(
        IKCandidate(
            qpos=qpos.copy(),
            source=candidate.source,
            seed_source=candidate.seed_source,
            method=candidate.method,
            metadata=dict(candidate.metadata),
        )
    )
    return True


class PinocchioIKProvider:
    """Generic iterative IK provider wrapper for Pinocchio-style solvers.

    This class intentionally has no robot-vendor dependency. Hardware adapters
    provide the concrete solver object and any target-frame transform needed to
    map their action/control frame to the solver end-effector frame.
    """

    def __init__(
        self,
        *,
        solver: Any,
        source: str = "pinocchio",
        method: str = "pinocchio_iterative",
        target_to_eef_transform: np.ndarray | None = None,
        qpos_size: int | None = None,
    ):
        self.solver = solver
        self.source = str(source)
        self.method = str(method)
        self.qpos_size = None if qpos_size is None else int(qpos_size)
        self.target_to_eef_transform = (
            np.eye(4, dtype=float)
            if target_to_eef_transform is None
            else np.asarray(target_to_eef_transform, dtype=float).reshape(4, 4)
        )
        self.eef_to_target_transform = np.linalg.inv(self.target_to_eef_transform)

    def iter_ik_candidates(
        self,
        *,
        target_pose: np.ndarray,
        seeds: tuple[np.ndarray, ...],
        seed_labels: tuple[str, ...],
        options: dict[str, Any],
    ) -> list[IKCandidate]:
        pose_matrix = np.asarray(target_pose, dtype=float).reshape(4, 4) @ self.target_to_eef_transform
        max_count = max(1, int(options.get("max_candidate_count", 32)))
        max_iterations = max(1, int(options.get("pinocchio_max_iterations", 80)))
        max_joint_step = float(options.get("pinocchio_max_joint_step_rad", 0.08))
        damping = float(options.get("pinocchio_damping", 0.1))
        tolerance = float(options.get("pinocchio_tolerance", 1e-4))
        deadline = _deadline_from_options(options)
        remaining_time_budget_ms = _positive_float(options.get("provider_remaining_time_budget_ms"))
        candidates: list[IKCandidate] = []
        for seed, seed_label in zip(seeds, seed_labels):
            if len(candidates) >= max_count or _deadline_exceeded(deadline):
                break
            try:
                seed_rad = _as_float_array(seed, expected_len=self.qpos_size, label="seed")
                solve_kwargs = {
                    "target_pose": pose_matrix,
                    "q_init": seed_rad,
                    "max_iterations": max_iterations,
                    "max_joint_step": max_joint_step,
                    "damping": damping,
                    "tolerance": tolerance,
                }
                if deadline is not None and _call_accepts_keyword(self.solver.solve_iterative, "deadline_monotonic"):
                    solve_kwargs["deadline_monotonic"] = float(deadline)
                if (
                    remaining_time_budget_ms is not None
                    and _call_accepts_keyword(self.solver.solve_iterative, "max_wall_time_sec")
                ):
                    solve_kwargs["max_wall_time_sec"] = max(0.0, float(remaining_time_budget_ms) / 1000.0)
                qpos = np.asarray(
                    self.solver.solve_iterative(**solve_kwargs),
                    dtype=float,
                ).reshape(seed_rad.shape)
            except Exception:
                continue
            _dedupe_append(
                candidates,
                IKCandidate(
                    qpos=qpos,
                    source=self.source,
                    seed_source=str(seed_label),
                    method=self.method,
                    metadata={
                        "pinocchio_max_iterations": max_iterations,
                        "pinocchio_max_joint_step_rad": max_joint_step,
                        "pinocchio_damping": damping,
                        "pinocchio_tolerance": tolerance,
                        "pinocchio_deadline_active": bool(deadline is not None),
                        "pinocchio_remaining_time_budget_ms": remaining_time_budget_ms,
                        "pinocchio_solve_iterations": getattr(
                            self.solver,
                            "last_solve_iterations",
                            None,
                        ),
                        "pinocchio_solve_converged": getattr(
                            self.solver,
                            "last_solve_converged",
                            None,
                        ),
                        "pinocchio_solve_deadline_exceeded": getattr(
                            self.solver,
                            "last_solve_deadline_exceeded",
                            None,
                        ),
                        "target_to_eef_transform_applied": bool(
                            not np.allclose(self.target_to_eef_transform, np.eye(4), atol=1e-9)
                        ),
                        "branch_selection": getattr(
                            self.solver,
                            "last_branch_selection",
                            "unknown",
                        ),
                        "branch_distance_score_deg": getattr(
                            self.solver,
                            "last_branch_distance_score_deg",
                            None,
                        ),
                    },
                ),
            )
        return candidates[:max_count]

    def forward_kinematics(self, qpos: np.ndarray) -> np.ndarray | None:
        try:
            eef_pose = np.asarray(self.solver.compute_pose(np.asarray(qpos, dtype=float).reshape(-1)), dtype=float)
            return eef_pose.reshape(4, 4) @ self.eef_to_target_transform
        except Exception:
            return None

    def joint_position_limits(self) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            q_min = np.asarray(self.solver.q_min, dtype=float).reshape(-1)
            q_max = np.asarray(self.solver.q_max, dtype=float).reshape(-1)
        except Exception:
            return None
        if q_min.shape != q_max.shape or not np.all(np.isfinite(q_min)) or not np.all(np.isfinite(q_max)):
            return None
        return q_min.copy(), q_max.copy()

    def analyze_singularity(self, qpos: np.ndarray) -> SingularityReport:
        del qpos
        return SingularityReport(
            status_code=None,
            label="unavailable",
            metric=None,
            metadata={"provider": self.source},
        )


def _deadline_from_options(options: dict[str, Any]) -> float | None:
    raw = options.get("provider_deadline_monotonic")
    try:
        deadline = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(deadline) or deadline <= 0.0:
        return None
    return deadline


def _deadline_exceeded(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= float(deadline)


def _positive_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric <= 0.0:
        return None
    return float(numeric)


def _call_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or name == keyword
        for name, parameter in signature.parameters.items()
    )
