from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class IKObjectiveWeights:
    single_arm_position_weight: float
    single_arm_orientation_weight: float
    dual_arm_position_cost: float
    dual_arm_orientation_cost: float


@dataclass(frozen=True)
class BranchNormalizationConfig:
    enabled: bool = True
    joint_distance_weights: tuple[float, ...] = (4.0, 3.0, 2.0, 1.25, 1.0, 0.75)
    limit_tolerance_rad: float = 1e-5


@dataclass(frozen=True)
class JumpGateConfig:
    enabled: bool = True
    hard_branch_joint_indices: tuple[int, ...] = (0, 1, 2)
    hard_branch_max_delta_deg: tuple[float, ...] = (90.0, 90.0, 90.0)
    small_pose_translation_m: float = 0.05
    small_pose_rotation_deg: float = 5.0
    moderate_pose_translation_m: float = 0.08
    moderate_pose_rotation_deg: float = 12.0
    moderate_pose_joint_jump_scale: float = 1.25
    per_joint_jump_deg: tuple[float, ...] = (15.0, 15.0, 18.0, 25.0, 30.0, 45.0)
    weighted_jump_threshold_deg: float = 35.0
    relaxed_per_joint_jump_deg: tuple[float, ...] = (25.0, 25.0, 30.0, 40.0, 50.0, 70.0)
    relaxed_weighted_jump_threshold_deg: float = 55.0
    release_translation_m: float = 0.08
    release_rotation_deg: float = 12.0
    seed_expansion_after_hold_frames: int = 1


@dataclass(frozen=True)
class DualArmRecoveryConfig:
    enable_seed_expansion: bool = True
    seed_modes: tuple[str, ...] = ("current", "left_flip", "right_flip", "both_flip")
    check_collision_on_branch_switch: bool = True


@dataclass(frozen=True)
class RM65IKTuningConfig:
    ik_objective_mode: str
    objective_modes: dict[str, IKObjectiveWeights]
    branch_normalization: BranchNormalizationConfig
    jump_gate: JumpGateConfig
    dual_arm_recovery: DualArmRecoveryConfig
    source_path: str | None = None

    def objective_weights(self) -> IKObjectiveWeights:
        if self.ik_objective_mode not in self.objective_modes:
            raise KeyError(
                f"Unknown ik_objective_mode '{self.ik_objective_mode}'. "
                f"Available modes: {sorted(self.objective_modes)}"
            )
        return self.objective_modes[self.ik_objective_mode]


def default_ik_tuning_json_path() -> Path:
    return Path(__file__).resolve().parents[1] / "teleop_runtime" / "configs" / "rm65_ik_tuning.json"


def _ensure_tuple_of_floats(values: Any, *, expected_len: int) -> tuple[float, ...]:
    numbers = tuple(float(value) for value in values)
    if len(numbers) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(numbers)}.")
    return numbers


def _objective_weights_from_payload(payload: dict[str, Any]) -> IKObjectiveWeights:
    return IKObjectiveWeights(
        single_arm_position_weight=float(payload["single_arm_position_weight"]),
        single_arm_orientation_weight=float(payload["single_arm_orientation_weight"]),
        dual_arm_position_cost=float(payload["dual_arm_position_cost"]),
        dual_arm_orientation_cost=float(payload["dual_arm_orientation_cost"]),
    )


def ik_tuning_from_dict(payload: dict[str, Any], *, source_path: str | None = None) -> RM65IKTuningConfig:
    objective_modes_payload = payload.get("objective_modes", {})
    objective_modes = {
        str(name): _objective_weights_from_payload(mode_payload)
        for name, mode_payload in objective_modes_payload.items()
    }
    if not objective_modes:
        raise ValueError("objective_modes must not be empty.")

    branch_payload = payload.get("branch_normalization", {})
    jump_payload = payload.get("jump_gate", {})
    recovery_payload = payload.get("dual_arm_recovery", {})
    hard_branch_joint_indices = tuple(
        int(value) for value in jump_payload.get("hard_branch_joint_indices", (0, 1, 2))
    )

    return RM65IKTuningConfig(
        ik_objective_mode=str(payload.get("ik_objective_mode", "orientation_priority")),
        objective_modes=objective_modes,
        branch_normalization=BranchNormalizationConfig(
            enabled=bool(branch_payload.get("enabled", True)),
            joint_distance_weights=_ensure_tuple_of_floats(
                branch_payload.get("joint_distance_weights", (4.0, 3.0, 2.0, 1.25, 1.0, 0.75)),
                expected_len=6,
            ),
            limit_tolerance_rad=float(branch_payload.get("limit_tolerance_rad", 1e-5)),
        ),
        jump_gate=JumpGateConfig(
            enabled=bool(jump_payload.get("enabled", True)),
            hard_branch_joint_indices=hard_branch_joint_indices,
            hard_branch_max_delta_deg=_ensure_tuple_of_floats(
                jump_payload.get("hard_branch_max_delta_deg", (90.0, 90.0, 90.0)),
                expected_len=len(hard_branch_joint_indices),
            ),
            small_pose_translation_m=float(jump_payload.get("small_pose_translation_m", 0.05)),
            small_pose_rotation_deg=float(jump_payload.get("small_pose_rotation_deg", 5.0)),
            moderate_pose_translation_m=float(jump_payload.get("moderate_pose_translation_m", 0.08)),
            moderate_pose_rotation_deg=float(jump_payload.get("moderate_pose_rotation_deg", 12.0)),
            moderate_pose_joint_jump_scale=float(
                jump_payload.get("moderate_pose_joint_jump_scale", 1.25)
            ),
            per_joint_jump_deg=_ensure_tuple_of_floats(
                jump_payload.get("per_joint_jump_deg", (15.0, 15.0, 18.0, 25.0, 30.0, 45.0)),
                expected_len=6,
            ),
            weighted_jump_threshold_deg=float(jump_payload.get("weighted_jump_threshold_deg", 35.0)),
            relaxed_per_joint_jump_deg=_ensure_tuple_of_floats(
                jump_payload.get(
                    "relaxed_per_joint_jump_deg",
                    (25.0, 25.0, 30.0, 40.0, 50.0, 70.0),
                ),
                expected_len=6,
            ),
            relaxed_weighted_jump_threshold_deg=float(
                jump_payload.get("relaxed_weighted_jump_threshold_deg", 55.0)
            ),
            release_translation_m=float(jump_payload.get("release_translation_m", 0.08)),
            release_rotation_deg=float(jump_payload.get("release_rotation_deg", 12.0)),
            seed_expansion_after_hold_frames=int(
                jump_payload.get("seed_expansion_after_hold_frames", 1)
            ),
        ),
        dual_arm_recovery=DualArmRecoveryConfig(
            enable_seed_expansion=bool(recovery_payload.get("enable_seed_expansion", True)),
            seed_modes=tuple(str(value) for value in recovery_payload.get(
                "seed_modes",
                ("current", "left_flip", "right_flip", "both_flip"),
            )),
            check_collision_on_branch_switch=bool(
                recovery_payload.get("check_collision_on_branch_switch", True)
            ),
        ),
        source_path=source_path,
    )


@lru_cache(maxsize=8)
def load_ik_tuning_config(path: str | Path | None = None) -> RM65IKTuningConfig:
    resolved = default_ik_tuning_json_path() if path in (None, "") else Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return ik_tuning_from_dict(payload, source_path=str(resolved))


def generate_rm65_wrist_flip_candidates(
    qpos: np.ndarray,
    *,
    q_min: np.ndarray | None = None,
    q_max: np.ndarray | None = None,
    limit_tolerance_rad: float = 1e-5,
) -> list[np.ndarray]:
    q = np.asarray(qpos, dtype=float).reshape(-1)
    if q.shape != (6,):
        raise ValueError(f"Expected RM65 6D joint vector, got shape {q.shape}.")

    candidates: list[np.ndarray] = [q.copy()]
    lower = None if q_min is None else np.asarray(q_min, dtype=float).reshape(-1)
    upper = None if q_max is None else np.asarray(q_max, dtype=float).reshape(-1)

    for q4_offset in (np.pi, -np.pi):
        for q6_offset in (np.pi, -np.pi):
            candidate = q.copy()
            candidate[3] += q4_offset
            candidate[4] = -candidate[4]
            candidate[5] += q6_offset
            if lower is not None and upper is not None:
                if np.any(candidate < lower - limit_tolerance_rad) or np.any(
                    candidate > upper + limit_tolerance_rad
                ):
                    continue
                candidate = np.clip(candidate, lower, upper)
            if any(np.allclose(candidate, existing, atol=1e-8) for existing in candidates):
                continue
            candidates.append(candidate)
    return candidates


def preferred_wrist_flip_candidate(
    qpos: np.ndarray,
    *,
    reference_q: np.ndarray,
    joint_distance_weights: np.ndarray | tuple[float, ...],
    q_min: np.ndarray | None = None,
    q_max: np.ndarray | None = None,
    limit_tolerance_rad: float = 1e-5,
) -> np.ndarray | None:
    candidates = generate_rm65_wrist_flip_candidates(
        qpos,
        q_min=q_min,
        q_max=q_max,
        limit_tolerance_rad=limit_tolerance_rad,
    )
    if len(candidates) <= 1:
        return None
    alternatives = [candidate for candidate in candidates if not np.allclose(candidate, qpos, atol=1e-8)]
    if not alternatives:
        return None
    return min(
        alternatives,
        key=lambda candidate: weighted_joint_distance_score(
            reference_q=reference_q,
            candidate_q=candidate,
            joint_distance_weights=joint_distance_weights,
        ),
    ).copy()


def weighted_joint_distance_score(
    *,
    reference_q: np.ndarray,
    candidate_q: np.ndarray,
    joint_distance_weights: np.ndarray | tuple[float, ...],
) -> float:
    reference = np.asarray(reference_q, dtype=float).reshape(-1)
    candidate = np.asarray(candidate_q, dtype=float).reshape(-1)
    weights = np.asarray(joint_distance_weights, dtype=float).reshape(-1)
    if reference.shape != candidate.shape or reference.shape != weights.shape:
        raise ValueError(
            "reference_q, candidate_q, and joint_distance_weights must share the same shape."
        )
    delta_deg = np.rad2deg(candidate - reference)
    return float(np.sqrt(np.sum(weights * np.square(delta_deg))))


def joint_jump_metrics(
    *,
    reference_q: np.ndarray,
    candidate_q: np.ndarray,
    joint_distance_weights: np.ndarray | tuple[float, ...],
) -> tuple[np.ndarray, float]:
    reference = np.asarray(reference_q, dtype=float).reshape(-1)
    candidate = np.asarray(candidate_q, dtype=float).reshape(-1)
    delta_deg = np.abs(np.rad2deg(candidate - reference))
    weighted_score = weighted_joint_distance_score(
        reference_q=reference,
        candidate_q=candidate,
        joint_distance_weights=joint_distance_weights,
    )
    return delta_deg, weighted_score


def joint_jump_exceeds(
    *,
    reference_q: np.ndarray,
    candidate_q: np.ndarray,
    joint_distance_weights: np.ndarray | tuple[float, ...],
    per_joint_jump_deg: np.ndarray | tuple[float, ...],
    weighted_jump_threshold_deg: float,
) -> tuple[bool, np.ndarray, float]:
    delta_deg, weighted_score = joint_jump_metrics(
        reference_q=reference_q,
        candidate_q=candidate_q,
        joint_distance_weights=joint_distance_weights,
    )
    per_joint_limit = np.asarray(per_joint_jump_deg, dtype=float).reshape(-1)
    if delta_deg.shape != per_joint_limit.shape:
        raise ValueError(
            "reference_q, candidate_q, and per_joint_jump_deg must share the same shape."
        )
    exceeds = bool(
        np.any(delta_deg > per_joint_limit)
        or weighted_score > float(weighted_jump_threshold_deg)
    )
    return exceeds, delta_deg, weighted_score


def hard_branch_jump_exceeds(
    *,
    reference_q: np.ndarray,
    candidate_q: np.ndarray,
    joint_indices: tuple[int, ...] | list[int],
    max_delta_deg: tuple[float, ...] | list[float],
) -> tuple[bool, np.ndarray]:
    reference = np.asarray(reference_q, dtype=float).reshape(-1)
    candidate = np.asarray(candidate_q, dtype=float).reshape(-1)
    indices = tuple(int(index) for index in joint_indices)
    limits = np.asarray(max_delta_deg, dtype=float).reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError("reference_q and candidate_q must share the same shape.")
    if limits.shape != (len(indices),):
        raise ValueError("max_delta_deg must match joint_indices.")
    if any(index < 0 or index >= reference.shape[0] for index in indices):
        raise ValueError("hard branch joint index out of range.")
    deltas = np.rad2deg(np.abs(candidate[list(indices)] - reference[list(indices)]))
    return bool(np.any(deltas > limits)), deltas


def pose_delta_metrics(reference_pose: np.ndarray, target_pose: np.ndarray) -> tuple[float, float]:
    reference = np.asarray(reference_pose, dtype=float)
    target = np.asarray(target_pose, dtype=float)
    if reference.shape != (4, 4) or target.shape != (4, 4):
        raise ValueError(
            f"pose_delta_metrics expects two (4, 4) poses, got {reference.shape} and {target.shape}."
        )
    translation_m = float(np.linalg.norm(target[:3, 3] - reference[:3, 3]))
    relative_rot = target[:3, :3] @ reference[:3, :3].T
    rotation_deg = float(np.rad2deg(np.linalg.norm(R.from_matrix(relative_rot).as_rotvec())))
    return translation_m, rotation_deg
