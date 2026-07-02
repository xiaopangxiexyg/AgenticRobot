from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class JointPostureEvaluation:
    score_deg: float
    labels: tuple[str, ...]
    bad: bool
    preferred: bool
    joint_deg: tuple[float, ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "score_deg": float(self.score_deg),
            "labels": list(self.labels),
            "bad": bool(self.bad),
            "preferred": bool(self.preferred),
            "joint_deg": [float(value) for value in self.joint_deg],
        }


@dataclass(frozen=True)
class JointPostureComparison:
    reference: JointPostureEvaluation
    candidate: JointPostureEvaluation

    @property
    def improvement_deg(self) -> float:
        return float(self.reference.score_deg - self.candidate.score_deg)

    @property
    def improved(self) -> bool:
        return self.improvement_deg > 0.0

    def metadata(self) -> dict[str, Any]:
        return {
            "reference": self.reference.metadata(),
            "candidate": self.candidate.metadata(),
            "improvement_deg": float(self.improvement_deg),
            "improved": bool(self.improved),
        }


@dataclass(frozen=True)
class JointPosturePreference:
    name: str
    enabled: bool = True
    j2_index: int = 1
    j3_index: int = 2
    j2_high_deg: float = 70.0
    j2_preferred_min_deg: float = -20.0
    j2_preferred_max_deg: float = 60.0
    j3_zero_margin_deg: float = 10.0
    j3_low_deg: float = 30.0
    j3_bad_max_deg: float = 60.0
    j3_preferred_min_deg: float = 80.0
    j3_preferred_max_deg: float = 130.0
    j3_upper_warning_deg: float = 125.0
    bad_branch_penalty_deg: float = 180.0
    j3_negative_penalty_deg: float = 160.0
    high_j2_low_j3_penalty_deg: float = 120.0
    j3_zero_penalty_deg: float = 70.0
    j3_low_penalty_deg: float = 40.0
    j3_upper_penalty_deg: float = 15.0
    preferred_bonus_deg: float = 20.0
    score_scale: float = 1.0
    hard_branch_escape_enabled: bool = False
    hard_branch_escape_allowed_joint_indices: tuple[int, ...] = (1, 2)
    hard_branch_escape_min_improvement_deg: float = 40.0
    singular_escape_enabled: bool = False
    singular_escape_labels: tuple[str, ...] = ("elbow_singularity",)
    singular_escape_min_improvement_deg: float = 10.0
    singular_escape_j3_abs_max_deg: float = 12.0
    local_escape_enabled: bool = False
    local_escape_max_candidates: int = 6
    local_escape_step_fraction: float = 1.0
    local_escape_max_position_worsen_m: float = 0.04
    local_escape_max_orientation_worsen_deg: float = 15.0
    local_escape_min_improvement_deg: float = 1.0
    wrist_unload_enabled: bool = False
    wrist_unload_j1_index: int = 0
    wrist_unload_j4_index: int = 3
    wrist_unload_j5_index: int = 4
    wrist_unload_j6_index: int = 5
    wrist_unload_j4_abs_activation_deg: float = 160.0
    wrist_unload_j5_abs_activation_deg: float = 75.0
    wrist_unload_j6_wrap_activation_deg: float = 165.0
    wrist_unload_step_fraction: float = 1.0
    wrist_unload_max_candidates: int = 8
    wrist_unload_max_position_worsen_m: float = 0.05
    wrist_unload_max_orientation_worsen_deg: float = 20.0
    wrist_unload_min_cost_improvement_deg: float = 0.5

    def evaluate(self, qpos: np.ndarray) -> JointPostureEvaluation:
        q = np.asarray(qpos, dtype=float).reshape(-1)
        joint_deg = np.rad2deg(q).astype(float)
        labels: list[str] = []
        score = 0.0
        j2 = _safe_joint(joint_deg, self.j2_index)
        j3 = _safe_joint(joint_deg, self.j3_index)

        if j3 < 0.0:
            labels.append("j3_negative_bad_branch")
            score += self.bad_branch_penalty_deg + self.j3_negative_penalty_deg
        if j2 > self.j2_high_deg and j3 < self.j3_bad_max_deg:
            labels.append("j2_high_j3_low_bad_branch")
            score += self.bad_branch_penalty_deg + self.high_j2_low_j3_penalty_deg
        if abs(j3) <= self.j3_zero_margin_deg:
            labels.append("j3_zero_zone")
            zero_weight = 1.0 - abs(j3) / max(1e-6, self.j3_zero_margin_deg)
            score += self.j3_zero_penalty_deg * max(0.0, zero_weight)
        elif j3 < self.j3_low_deg:
            labels.append("j3_low_zone")
            low_weight = (self.j3_low_deg - j3) / max(1e-6, self.j3_low_deg)
            score += self.j3_low_penalty_deg * max(0.0, low_weight)
        if j3 > self.j3_upper_warning_deg:
            labels.append("j3_upper_warning_zone")
            upper_width = max(1e-6, self.j3_preferred_max_deg - self.j3_upper_warning_deg)
            upper_weight = (j3 - self.j3_upper_warning_deg) / upper_width
            score += self.j3_upper_penalty_deg * max(0.0, upper_weight)

        preferred = (
            self.j2_preferred_min_deg <= j2 <= self.j2_preferred_max_deg
            and self.j3_preferred_min_deg <= j3 <= self.j3_preferred_max_deg
        )
        if preferred:
            labels.append("preferred_branch")
            score = max(0.0, score - max(0.0, self.preferred_bonus_deg))

        bad = any(label.endswith("_bad_branch") for label in labels)
        if not labels:
            labels.append("neutral")
        return JointPostureEvaluation(
            score_deg=float(max(0.0, score) * max(0.0, self.score_scale)),
            labels=tuple(labels),
            bad=bool(bad),
            preferred=bool(preferred),
            joint_deg=tuple(float(value) for value in joint_deg.tolist()),
        )

    def compare(self, reference_qpos: np.ndarray, candidate_qpos: np.ndarray) -> JointPostureComparison:
        return JointPostureComparison(
            reference=self.evaluate(reference_qpos),
            candidate=self.evaluate(candidate_qpos),
        )

    def hard_branch_escape_metadata(
        self,
        *,
        reference_qpos: np.ndarray,
        candidate_qpos: np.ndarray,
        exceeded_joint_indices: tuple[int, ...],
    ) -> dict[str, Any]:
        comparison = self.compare(reference_qpos, candidate_qpos)
        allowed_indices = {int(index) for index in self.hard_branch_escape_allowed_joint_indices}
        exceeded = {int(index) for index in exceeded_joint_indices}
        disallowed = sorted(exceeded.difference(allowed_indices))
        improvement_ok = (
            comparison.reference.bad
            and comparison.improvement_deg >= max(0.0, self.hard_branch_escape_min_improvement_deg)
        )
        allowed = bool(
            self.enabled
            and self.hard_branch_escape_enabled
            and not disallowed
            and improvement_ok
        )
        return {
            "enabled": bool(self.enabled and self.hard_branch_escape_enabled),
            "allowed": bool(allowed),
            "allowed_joint_indices": sorted(allowed_indices),
            "exceeded_joint_indices": sorted(exceeded),
            "disallowed_joint_indices": disallowed,
            "min_improvement_deg": float(self.hard_branch_escape_min_improvement_deg),
            **comparison.metadata(),
        }

    def singular_escape_metadata(
        self,
        *,
        reference_qpos: np.ndarray,
        candidate_qpos: np.ndarray,
        singularity_label: str,
    ) -> dict[str, Any]:
        comparison = self.compare(reference_qpos, candidate_qpos)
        candidate_j3 = _safe_joint(np.rad2deg(np.asarray(candidate_qpos, dtype=float).reshape(-1)), self.j3_index)
        label_allowed = str(singularity_label) in {str(label) for label in self.singular_escape_labels}
        improvement_ok = (
            comparison.reference.bad
            and comparison.improvement_deg >= max(0.0, self.singular_escape_min_improvement_deg)
        )
        j3_crossing_ok = abs(candidate_j3) <= max(0.0, self.singular_escape_j3_abs_max_deg)
        allowed = bool(
            self.enabled
            and self.singular_escape_enabled
            and label_allowed
            and improvement_ok
            and j3_crossing_ok
        )
        return {
            "enabled": bool(self.enabled and self.singular_escape_enabled),
            "allowed": bool(allowed),
            "singularity_label": str(singularity_label),
            "allowed_singularity_labels": [str(label) for label in self.singular_escape_labels],
            "candidate_j3_deg": float(candidate_j3),
            "singular_escape_j3_abs_max_deg": float(self.singular_escape_j3_abs_max_deg),
            "min_improvement_deg": float(self.singular_escape_min_improvement_deg),
            **comparison.metadata(),
        }

    def local_escape_targets(
        self,
        *,
        reference_qpos: np.ndarray,
        max_step_rad: np.ndarray,
        q_min: np.ndarray | None = None,
        q_max: np.ndarray | None = None,
    ) -> tuple[list[tuple[str, np.ndarray, dict[str, Any]]], dict[str, Any]]:
        metadata: dict[str, Any] = {
            "enabled": bool(self.enabled and self.local_escape_enabled),
            "candidate_count": 0,
        }
        if not self.enabled or not self.local_escape_enabled:
            metadata["status"] = "disabled"
            return [], metadata
        q = np.asarray(reference_qpos, dtype=float).reshape(-1)
        step = np.asarray(max_step_rad, dtype=float).reshape(q.shape) * max(
            0.0,
            min(1.0, float(self.local_escape_step_fraction)),
        )
        current = self.evaluate(q)
        metadata["reference"] = current.metadata()
        if current.score_deg <= 0.0:
            metadata["status"] = "reference_already_preferred_or_neutral"
            return [], metadata
        actions: list[tuple[str, dict[int, int]]] = []
        j2 = _safe_joint(current.joint_deg, self.j2_index)
        j3 = _safe_joint(current.joint_deg, self.j3_index)
        if j3 < self.j3_preferred_min_deg:
            actions.append(("j3_toward_positive_branch", {self.j3_index: 1}))
        elif j3 > self.j3_preferred_max_deg:
            actions.append(("j3_away_from_upper_warning", {self.j3_index: -1}))
        if j2 > self.j2_preferred_max_deg:
            actions.append(("j2_toward_preferred_window", {self.j2_index: -1}))
        elif j2 < self.j2_preferred_min_deg:
            actions.append(("j2_toward_preferred_window", {self.j2_index: 1}))
        if j3 < self.j3_preferred_min_deg and j2 > self.j2_preferred_max_deg:
            actions.append(
                (
                    "j2_j3_toward_good_branch",
                    {self.j2_index: -1, self.j3_index: 1},
                )
            )
        if not actions:
            metadata["status"] = "no_direction"
            return [], metadata

        candidates: list[tuple[str, np.ndarray, dict[str, Any]]] = []
        seen: list[np.ndarray] = []
        for label, directions in actions:
            if len(candidates) >= max(0, int(self.local_escape_max_candidates)):
                break
            candidate = q.copy()
            applied: dict[str, float] = {}
            for joint_index, direction in directions.items():
                index = int(joint_index)
                if index < 0 or index >= q.size:
                    continue
                delta = float(step[index]) * float(direction)
                if abs(delta) <= 1e-12:
                    continue
                candidate[index] += delta
                applied[f"j{index + 1}"] = float(np.rad2deg(delta))
            if not applied:
                continue
            if q_min is not None and q_max is not None:
                lower = np.asarray(q_min, dtype=float).reshape(q.shape)
                upper = np.asarray(q_max, dtype=float).reshape(q.shape)
                candidate = np.minimum(np.maximum(candidate, lower), upper)
            if any(np.allclose(candidate, existing, atol=1e-9) for existing in seen):
                continue
            comparison = self.compare(q, candidate)
            if comparison.improvement_deg < max(0.0, self.local_escape_min_improvement_deg):
                continue
            seen.append(candidate.copy())
            candidates.append(
                (
                    label,
                    candidate,
                    {
                        "local_step_posture_escape_candidate": True,
                        "local_step_posture_escape_label": label,
                        "local_step_posture_escape_applied_delta_deg": applied,
                        "local_step_posture_escape_min_improvement_deg": float(
                            self.local_escape_min_improvement_deg
                        ),
                        **comparison.metadata(),
                    },
                )
            )
        metadata.update(
            {
                "status": "active" if candidates else "no_improving_candidates",
                "candidate_count": len(candidates),
                "max_candidates": int(self.local_escape_max_candidates),
                "step_fraction": float(self.local_escape_step_fraction),
                "emitted_methods": [label for label, _q, _meta in candidates],
            }
        )
        return candidates, metadata

    def wrist_unload_targets(
        self,
        *,
        reference_qpos: np.ndarray,
        max_step_rad: np.ndarray,
        q_min: np.ndarray | None = None,
        q_max: np.ndarray | None = None,
    ) -> tuple[list[tuple[str, np.ndarray, dict[str, Any]]], dict[str, Any]]:
        metadata: dict[str, Any] = {
            "enabled": bool(self.enabled and self.wrist_unload_enabled),
            "candidate_count": 0,
        }
        if not self.enabled or not self.wrist_unload_enabled:
            metadata["status"] = "disabled"
            return [], metadata
        q = np.asarray(reference_qpos, dtype=float).reshape(-1)
        step = np.asarray(max_step_rad, dtype=float).reshape(q.shape) * max(
            0.0,
            min(1.0, float(self.wrist_unload_step_fraction)),
        )
        before = self.wrist_unload_cost(q)
        metadata["reference"] = dict(before)
        if not bool(before.get("active", False)):
            metadata["status"] = "inactive_reference_not_near_wrist_boundary"
            return [], metadata

        q_min_array = None if q_min is None else np.asarray(q_min, dtype=float).reshape(q.shape)
        q_max_array = None if q_max is None else np.asarray(q_max, dtype=float).reshape(q.shape)
        actions = self._wrist_unload_actions(q)
        candidates: list[tuple[str, np.ndarray, dict[str, Any]]] = []
        seen: list[np.ndarray] = []
        for label, directions in actions:
            if len(candidates) >= max(0, int(self.wrist_unload_max_candidates)):
                break
            candidate = q.copy()
            applied: dict[str, float] = {}
            for joint_index, direction in directions.items():
                index = int(joint_index)
                if index < 0 or index >= q.size:
                    continue
                delta = float(step[index]) * float(direction)
                if abs(delta) <= 1e-12:
                    continue
                candidate[index] += delta
                applied[f"j{index + 1}"] = float(np.rad2deg(delta))
            if not applied:
                continue
            if q_min_array is not None and q_max_array is not None:
                candidate = np.minimum(np.maximum(candidate, q_min_array), q_max_array)
            if any(np.allclose(candidate, existing, atol=1e-9) for existing in seen):
                continue
            after = self.wrist_unload_cost(candidate)
            improvement = float(before["score_deg"]) - float(after["score_deg"])
            seen.append(candidate.copy())
            candidates.append(
                (
                    label,
                    candidate,
                    {
                        "local_step_wrist_unload_candidate": True,
                        "local_step_wrist_unload_label": label,
                        "local_step_wrist_unload_applied_delta_deg": applied,
                        "local_step_wrist_unload_score_before_deg": float(before["score_deg"]),
                        "local_step_wrist_unload_score_after_deg": float(after["score_deg"]),
                        "local_step_wrist_unload_cost_improvement_deg": float(improvement),
                        "local_step_wrist_unload_min_cost_improvement_deg": float(
                            self.wrist_unload_min_cost_improvement_deg
                        ),
                        "local_step_wrist_unload_reference": dict(before),
                        "local_step_wrist_unload_candidate_state": dict(after),
                    },
                )
            )
        candidates.sort(
            key=lambda item: (
                -float(item[2]["local_step_wrist_unload_cost_improvement_deg"]),
                item[0],
            )
        )
        metadata.update(
            {
                "status": "active" if candidates else "no_candidates",
                "candidate_count": len(candidates),
                "max_candidates": int(self.wrist_unload_max_candidates),
                "step_fraction": float(self.wrist_unload_step_fraction),
                "emitted_methods": [label for label, _q, _meta in candidates],
            }
        )
        return candidates, metadata

    def wrist_unload_cost(self, qpos: np.ndarray) -> dict[str, Any]:
        q = np.asarray(qpos, dtype=float).reshape(-1)
        joint_deg = np.rad2deg(q).astype(float)
        j4 = _safe_joint(joint_deg, self.wrist_unload_j4_index)
        j5 = _safe_joint(joint_deg, self.wrist_unload_j5_index)
        j6 = _wrap_degrees(_safe_joint(joint_deg, self.wrist_unload_j6_index))
        j4_excess = max(0.0, abs(j4) - max(0.0, float(self.wrist_unload_j4_abs_activation_deg)))
        j5_excess = max(0.0, abs(j5) - max(0.0, float(self.wrist_unload_j5_abs_activation_deg)))
        j6_excess = max(0.0, abs(j6) - max(0.0, float(self.wrist_unload_j6_wrap_activation_deg)))
        score = float(j4_excess + 0.5 * j5_excess + j6_excess)
        labels: list[str] = []
        if j4_excess > 0.0:
            labels.append("j4_near_limit")
        if j5_excess > 0.0:
            labels.append("j5_high_wrist_bend")
        if j6_excess > 0.0:
            labels.append("j6_near_wrap")
        return {
            "active": bool(score > 0.0),
            "score_deg": score,
            "labels": labels,
            "j4_deg": float(j4),
            "j5_deg": float(j5),
            "j6_wrapped_deg": float(j6),
            "j4_excess_deg": float(j4_excess),
            "j5_excess_deg": float(j5_excess),
            "j6_excess_deg": float(j6_excess),
        }

    def _wrist_unload_actions(self, qpos: np.ndarray) -> list[tuple[str, dict[int, int]]]:
        q = np.asarray(qpos, dtype=float).reshape(-1)
        joint_deg = np.rad2deg(q).astype(float)
        j4 = _safe_joint(joint_deg, self.wrist_unload_j4_index)
        j6 = _wrap_degrees(_safe_joint(joint_deg, self.wrist_unload_j6_index))
        j4_center_dir = -_sign(j4)
        j6_center_dir = -_sign(j6)
        j1 = int(self.wrist_unload_j1_index)
        j4_index = int(self.wrist_unload_j4_index)
        j6_index = int(self.wrist_unload_j6_index)
        actions: list[tuple[str, dict[int, int]]] = [
            ("wrist_unload_j1_pos", {j1: 1}),
            ("wrist_unload_j1_neg", {j1: -1}),
        ]
        if j4_center_dir:
            actions.append(("wrist_unload_j4_center", {j4_index: j4_center_dir}))
            actions.append(("wrist_unload_j1_pos_j4_center", {j1: 1, j4_index: j4_center_dir}))
            actions.append(("wrist_unload_j1_neg_j4_center", {j1: -1, j4_index: j4_center_dir}))
        if j6_center_dir:
            actions.append(("wrist_unload_j6_center", {j6_index: j6_center_dir}))
            actions.append(("wrist_unload_j1_pos_j6_center", {j1: 1, j6_index: j6_center_dir}))
            actions.append(("wrist_unload_j1_neg_j6_center", {j1: -1, j6_index: j6_center_dir}))
        if j4_center_dir and j6_center_dir:
            actions.append(("wrist_unload_j4_j6_center", {j4_index: j4_center_dir, j6_index: j6_center_dir}))
            actions.append(
                (
                    "wrist_unload_j1_pos_j4_j6_center",
                    {j1: 1, j4_index: j4_center_dir, j6_index: j6_center_dir},
                )
            )
            actions.append(
                (
                    "wrist_unload_j1_neg_j4_j6_center",
                    {j1: -1, j4_index: j4_center_dir, j6_index: j6_center_dir},
                )
            )
        return actions


def posture_preference_from_metadata(metadata: dict[str, Any] | None) -> JointPosturePreference | None:
    if not isinstance(metadata, dict):
        return None
    payload = metadata.get("joint_posture_preference") or metadata.get("posture_preference")
    if not isinstance(payload, dict):
        return None
    if not bool(payload.get("enabled", True)):
        return None
    return JointPosturePreference(
        name=str(payload.get("name") or payload.get("profile") or "joint_posture"),
        enabled=True,
        j2_index=int(payload.get("j2_index", 1)),
        j3_index=int(payload.get("j3_index", 2)),
        j2_high_deg=float(payload.get("j2_high_deg", 70.0)),
        j2_preferred_min_deg=float(payload.get("j2_preferred_min_deg", -20.0)),
        j2_preferred_max_deg=float(payload.get("j2_preferred_max_deg", 60.0)),
        j3_zero_margin_deg=float(payload.get("j3_zero_margin_deg", 10.0)),
        j3_low_deg=float(payload.get("j3_low_deg", 30.0)),
        j3_bad_max_deg=float(payload.get("j3_bad_max_deg", 60.0)),
        j3_preferred_min_deg=float(payload.get("j3_preferred_min_deg", 80.0)),
        j3_preferred_max_deg=float(payload.get("j3_preferred_max_deg", 130.0)),
        j3_upper_warning_deg=float(payload.get("j3_upper_warning_deg", 125.0)),
        bad_branch_penalty_deg=float(payload.get("bad_branch_penalty_deg", 180.0)),
        j3_negative_penalty_deg=float(payload.get("j3_negative_penalty_deg", 160.0)),
        high_j2_low_j3_penalty_deg=float(payload.get("high_j2_low_j3_penalty_deg", 120.0)),
        j3_zero_penalty_deg=float(payload.get("j3_zero_penalty_deg", 70.0)),
        j3_low_penalty_deg=float(payload.get("j3_low_penalty_deg", 40.0)),
        j3_upper_penalty_deg=float(payload.get("j3_upper_penalty_deg", 15.0)),
        preferred_bonus_deg=float(payload.get("preferred_bonus_deg", 20.0)),
        score_scale=float(payload.get("score_scale", 1.0)),
        hard_branch_escape_enabled=bool(payload.get("hard_branch_escape_enabled", False)),
        hard_branch_escape_allowed_joint_indices=_tuple_int(
            payload.get("hard_branch_escape_allowed_joint_indices"),
            default=(1, 2),
        ),
        hard_branch_escape_min_improvement_deg=float(
            payload.get("hard_branch_escape_min_improvement_deg", 40.0)
        ),
        singular_escape_enabled=bool(payload.get("singular_escape_enabled", False)),
        singular_escape_labels=tuple(
            str(label) for label in payload.get("singular_escape_labels", ("elbow_singularity",))
        ),
        singular_escape_min_improvement_deg=float(
            payload.get("singular_escape_min_improvement_deg", 10.0)
        ),
        singular_escape_j3_abs_max_deg=float(payload.get("singular_escape_j3_abs_max_deg", 12.0)),
        local_escape_enabled=bool(payload.get("local_escape_enabled", False)),
        local_escape_max_candidates=int(payload.get("local_escape_max_candidates", 6)),
        local_escape_step_fraction=float(payload.get("local_escape_step_fraction", 1.0)),
        local_escape_max_position_worsen_m=float(payload.get("local_escape_max_position_worsen_m", 0.04)),
        local_escape_max_orientation_worsen_deg=float(
            payload.get("local_escape_max_orientation_worsen_deg", 15.0)
        ),
        local_escape_min_improvement_deg=float(payload.get("local_escape_min_improvement_deg", 1.0)),
        wrist_unload_enabled=bool(payload.get("wrist_unload_enabled", False)),
        wrist_unload_j1_index=int(payload.get("wrist_unload_j1_index", 0)),
        wrist_unload_j4_index=int(payload.get("wrist_unload_j4_index", 3)),
        wrist_unload_j5_index=int(payload.get("wrist_unload_j5_index", 4)),
        wrist_unload_j6_index=int(payload.get("wrist_unload_j6_index", 5)),
        wrist_unload_j4_abs_activation_deg=float(payload.get("wrist_unload_j4_abs_activation_deg", 160.0)),
        wrist_unload_j5_abs_activation_deg=float(payload.get("wrist_unload_j5_abs_activation_deg", 75.0)),
        wrist_unload_j6_wrap_activation_deg=float(payload.get("wrist_unload_j6_wrap_activation_deg", 165.0)),
        wrist_unload_step_fraction=float(payload.get("wrist_unload_step_fraction", 1.0)),
        wrist_unload_max_candidates=int(payload.get("wrist_unload_max_candidates", 8)),
        wrist_unload_max_position_worsen_m=float(payload.get("wrist_unload_max_position_worsen_m", 0.05)),
        wrist_unload_max_orientation_worsen_deg=float(payload.get("wrist_unload_max_orientation_worsen_deg", 20.0)),
        wrist_unload_min_cost_improvement_deg=float(payload.get("wrist_unload_min_cost_improvement_deg", 0.5)),
    )


def _tuple_int(value: Any, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    return tuple(int(item) for item in value)


def _safe_joint(joint_deg: Any, index: int) -> float:
    values = np.asarray(joint_deg, dtype=float).reshape(-1)
    if index < 0 or index >= values.size:
        return 0.0
    return float(values[int(index)])


def _wrap_degrees(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0
