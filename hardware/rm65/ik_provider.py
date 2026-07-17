from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_motion.pose import matrix_to_pose6, matrix_to_rm_quat_pose, pose6_to_matrix
from robot_motion.types import IKCandidate, SingularityReport


def _as_float_array(values: object, *, expected_len: int, label: str) -> np.ndarray:
    if hasattr(values, "tolist"):
        values = values.tolist()
    try:
        array = np.asarray(list(values), dtype=float).reshape(-1)
    except Exception as exc:  # noqa: BLE001 - adapter should annotate bad SDK payloads.
        raise ValueError(f"{label} is not a numeric sequence") from exc
    if array.shape != (expected_len,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain {expected_len} finite values, got shape={array.shape}")
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


@dataclass(frozen=True)
class RM65SDKIKProvider:
    """RM65 kinematics adapter for the generic ``robot_motion`` resolver.

    The boundary is intentionally unit-normalized: public methods use radians
    and homogeneous matrices, while this adapter alone translates to the RM SDK
    convention of degree joints plus Euler/quaternion pose arrays.
    """

    robot: Any
    use_quaternion_fallback: bool = True
    use_all_solutions: bool = True
    quaternion_fallback_only_if_no_candidates: bool = True
    single_solution_only_if_no_all_solutions: bool = True
    include_wrist_flip_candidates: bool = True
    q_min: np.ndarray | None = None
    q_max: np.ndarray | None = None

    def iter_ik_candidates(
        self,
        *,
        target_pose: np.ndarray,
        seeds: tuple[np.ndarray, ...],
        seed_labels: tuple[str, ...],
        options: dict[str, Any],
    ) -> list[IKCandidate]:
        pose_matrix = np.asarray(target_pose, dtype=float).reshape(4, 4)
        euler_pose = matrix_to_pose6(pose_matrix)
        quat_pose = matrix_to_rm_quat_pose(pose_matrix)
        max_count = max(1, int(options.get("max_candidate_count", 32)))
        deadline = _deadline_from_options(options)
        candidates: list[IKCandidate] = []
        parallel_requested = bool(options.get("allow_parallel", False))

        seed_items = list(zip(seeds, seed_labels))
        for seed, seed_label in seed_items:
            if len(candidates) >= max_count or _deadline_exceeded(deadline):
                break
            seed_rad = _as_float_array(seed, expected_len=6, label="seed")
            seed_deg = np.rad2deg(seed_rad).tolist()
            self._append_sdk_candidates(
                candidates=candidates,
                max_count=max_count,
                deadline=deadline,
                seed_deg=seed_deg,
                seed_source=str(seed_label),
                pose=euler_pose,
                flag=1,
                method="euler",
                parallel_requested=parallel_requested,
            )
            if (
                self.use_quaternion_fallback
                and len(candidates) < max_count
                and (
                    not self.quaternion_fallback_only_if_no_candidates
                    or len(candidates) == 0
                )
            ):
                self._append_sdk_candidates(
                    candidates=candidates,
                    max_count=max_count,
                    deadline=deadline,
                    seed_deg=seed_deg,
                    seed_source=str(seed_label),
                    pose=quat_pose,
                    flag=0,
                    method="quaternion",
                    parallel_requested=parallel_requested,
                )

        if (
            self.include_wrist_flip_candidates
            and bool(options.get("enable_expensive_seed_expansion", False))
            and len(candidates) < max_count
            and not _deadline_exceeded(deadline)
        ):
            self._append_wrist_flip_candidates(
                candidates,
                max_count=max_count,
                deadline=deadline,
            )

        return candidates[:max_count]

    def _append_sdk_candidates(
        self,
        *,
        candidates: list[IKCandidate],
        max_count: int,
        deadline: float | None,
        seed_deg: list[float],
        seed_source: str,
        pose: list[float],
        flag: int,
        method: str,
        parallel_requested: bool,
    ) -> None:
        if _deadline_exceeded(deadline):
            return
        if self.use_all_solutions:
            before_all_count = len(candidates)
            self._append_all_solution_candidates(
                candidates=candidates,
                max_count=max_count,
                deadline=deadline,
                seed_deg=seed_deg,
                seed_source=seed_source,
                pose=pose,
                flag=flag,
                method=f"{method}_all",
                parallel_requested=parallel_requested,
            )
            if self.single_solution_only_if_no_all_solutions and len(candidates) > before_all_count:
                return
        if len(candidates) >= max_count:
            return
        if _deadline_exceeded(deadline):
            return
        try:
            code, joint = self.robot.Algo_Inverse_Kinematics(seed_deg, pose, flag=flag)
        except Exception as exc:  # noqa: BLE001 - SDK failures become candidate metadata.
            if hasattr(self.robot, "_last_ik_provider_error"):
                self.robot._last_ik_provider_error = f"{type(exc).__name__}: {exc}"
            return
        if int(code) != 0:
            return
        try:
            q_deg = _as_float_array(joint, expected_len=6, label="ik_joint")
        except ValueError:
            return
        if not self._joint_position_limit_ok(q_deg):
            return
        qpos = np.deg2rad(q_deg)
        _dedupe_append(
            candidates,
            IKCandidate(
                qpos=qpos,
                source="rm_sdk",
                seed_source=seed_source,
                method=method,
                metadata={
                    "code": int(code),
                    "sdk_flag": int(flag),
                    "parallel_requested_but_sdk_serialized": parallel_requested,
                },
            ),
        )

    def _append_all_solution_candidates(
        self,
        *,
        candidates: list[IKCandidate],
        max_count: int,
        deadline: float | None,
        seed_deg: list[float],
        seed_source: str,
        pose: list[float],
        flag: int,
        method: str,
        parallel_requested: bool,
    ) -> None:
        if _deadline_exceeded(deadline):
            return
        all_solver = getattr(self.robot, "Algo_Inverse_Kinematics_All", None)
        if all_solver is None:
            return
        try:
            all_result = all_solver(seed_deg, pose, flag=flag)
        except Exception:
            return
        result_code = int(getattr(all_result, "result", 1))
        if result_code != 0:
            return
        count = max(0, min(8, int(getattr(all_result, "num", 0))))
        rows = getattr(all_result, "q_solve", [])
        for index in range(count):
            if len(candidates) >= max_count or _deadline_exceeded(deadline):
                return
            try:
                q_deg = _as_float_array(rows[index], expected_len=8, label="all_ik_joint")[:6]
            except Exception:  # noqa: BLE001 - skip malformed SDK rows.
                continue
            if not self._joint_position_limit_ok(q_deg):
                continue
            _dedupe_append(
                candidates,
                IKCandidate(
                    qpos=np.deg2rad(q_deg),
                    source="rm_sdk_all",
                    seed_source=seed_source,
                    method=method,
                    metadata={
                        "all_solution_index": index,
                        "all_solution_count": count,
                        "sdk_flag": int(flag),
                        "parallel_requested_but_sdk_serialized": parallel_requested,
                    },
                ),
            )

    def _append_wrist_flip_candidates(
        self,
        candidates: list[IKCandidate],
        *,
        max_count: int,
        deadline: float | None,
    ) -> None:
        try:
            from robot_motion.rm65.ik_tuning import generate_rm65_wrist_flip_candidates
        except Exception:  # noqa: BLE001 - optional branch expansion.
            return

        base_candidates = list(candidates)
        for base in base_candidates:
            if len(candidates) >= max_count or _deadline_exceeded(deadline):
                return
            try:
                alternatives = generate_rm65_wrist_flip_candidates(
                    base.qpos,
                    q_min=self.q_min,
                    q_max=self.q_max,
                )
            except Exception:  # noqa: BLE001 - optional branch expansion.
                continue
            for alt_index, qpos in enumerate(alternatives):
                if len(candidates) >= max_count or _deadline_exceeded(deadline):
                    return
                if np.allclose(qpos, base.qpos, atol=1e-8):
                    continue
                _dedupe_append(
                    candidates,
                    IKCandidate(
                        qpos=np.asarray(qpos, dtype=float).reshape(6),
                        source=f"{base.source}:wrist_flip",
                        seed_source=base.seed_source,
                        method=f"{base.method}:wrist_flip",
                        metadata={**base.metadata, "wrist_flip_index": alt_index},
                    ),
                )

    def _joint_position_limit_ok(self, q_deg: np.ndarray) -> bool:
        checker = getattr(self.robot, "Algo_IKine_Check_Joint_Position_Limit", None)
        if checker is None:
            return True
        try:
            return int(checker(q_deg.tolist())) == 0
        except Exception:  # noqa: BLE001 - SDK limit checker is advisory here.
            return True

    def forward_kinematics(self, qpos: np.ndarray) -> np.ndarray | None:
        try:
            result = self.robot.Algo_Forward_Kinematics(np.rad2deg(qpos).tolist(), flag=1)
        except Exception:
            return None
        if isinstance(result, tuple) and len(result) >= 2:
            code, pose = result[:2]
            if int(code) != 0:
                return None
        else:
            pose = result
        try:
            return pose6_to_matrix(_as_float_array(pose, expected_len=6, label="fk_pose"))
        except ValueError:
            return None

    def joint_position_limits(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.q_min is None or self.q_max is None:
            return None
        try:
            q_min = np.asarray(self.q_min, dtype=float).reshape(6)
            q_max = np.asarray(self.q_max, dtype=float).reshape(6)
        except Exception:
            return None
        if not np.all(np.isfinite(q_min)) or not np.all(np.isfinite(q_max)):
            return None
        return q_min.copy(), q_max.copy()

    def analyze_singularity(self, qpos: np.ndarray) -> SingularityReport:
        labels = {
            0: "normal",
            -1: "shoulder_singularity",
            -2: "elbow_singularity",
            -3: "wrist_singularity",
        }
        try:
            result = self.robot.Algo_Kin_Singularity_Analyse(np.rad2deg(qpos).tolist())
        except Exception as exc:  # noqa: BLE001 - surface diagnostics but do not fail provider.
            return SingularityReport(
                status_code=None,
                label="unavailable",
                metric=None,
                metadata={"error": f"{type(exc).__name__}: {exc}"},
            )
        if not isinstance(result, tuple) or len(result) < 2:
            return SingularityReport(
                status_code=None,
                label="unknown",
                metric=None,
                metadata={"raw": repr(result)},
            )
        try:
            status = int(result[0])
        except (TypeError, ValueError):
            status = None
        try:
            metric = float(result[1])
        except (TypeError, ValueError):
            metric = None
        return SingularityReport(
            status_code=status,
            label=labels.get(status, f"unknown_status_{status}"),
            metric=metric,
            metadata={"raw": repr(result)},
        )
