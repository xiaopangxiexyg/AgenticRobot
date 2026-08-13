"""Observational pose-target lag metrics for PPO rollouts and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PoseTargetDiagnostics:
    sample_count: int = 0
    lag_sum: float = 0.0
    lag_max: float = 0.0
    projected_count: int = 0
    projected_lag_sum: float = 0.0
    clear_count: int = 0
    clear_lag_sum: float = 0.0
    onset_count: int = 0
    onset_lag_sum: float = 0.0
    blocked_sum: float = 0.0
    lag_square_sum: float = 0.0
    blocked_square_sum: float = 0.0
    lag_blocked_product_sum: float = 0.0

    def update(
        self,
        translation_error_m: np.ndarray,
        *,
        projection_alpha: np.ndarray,
        preventive_intervention: np.ndarray,
        preventive_onset: np.ndarray,
    ) -> None:
        lag = np.asarray(translation_error_m, dtype=np.float64)
        alpha = np.asarray(projection_alpha, dtype=np.float64)
        intervention = np.asarray(preventive_intervention, dtype=bool)
        onset = np.asarray(preventive_onset, dtype=bool)
        if not (lag.shape == alpha.shape == intervention.shape == onset.shape):
            raise ValueError("Pose-target diagnostic arrays must have identical shapes.")
        valid = np.isfinite(lag) & (lag >= 0.0) & np.isfinite(alpha)
        if np.any(valid & ((alpha < 0.0) | (alpha > 1.0))):
            raise ValueError("projection_alpha must lie in [0, 1].")
        if not np.any(valid):
            return
        lag = lag[valid]
        alpha = alpha[valid]
        intervention = intervention[valid]
        onset = onset[valid]
        blocked = np.where(intervention, 1.0 - alpha, 0.0)

        self.sample_count += int(lag.size)
        self.lag_sum += float(np.sum(lag))
        self.lag_max = max(self.lag_max, float(np.max(lag)))
        self.projected_count += int(np.count_nonzero(intervention))
        self.projected_lag_sum += float(np.sum(lag[intervention]))
        clear = ~intervention
        self.clear_count += int(np.count_nonzero(clear))
        self.clear_lag_sum += float(np.sum(lag[clear]))
        self.onset_count += int(np.count_nonzero(onset))
        self.onset_lag_sum += float(np.sum(lag[onset]))
        self.blocked_sum += float(np.sum(blocked))
        self.lag_square_sum += float(np.sum(lag * lag))
        self.blocked_square_sum += float(np.sum(blocked * blocked))
        self.lag_blocked_product_sum += float(np.sum(lag * blocked))

    def summary(self) -> dict[str, float | None]:
        return {
            "pose_target_translation_error_m_mean": self._mean(
                self.lag_sum, self.sample_count
            ),
            "pose_target_translation_error_m_max": (
                None if self.sample_count == 0 else self.lag_max
            ),
            "pose_target_translation_error_m_projected_mean": self._mean(
                self.projected_lag_sum, self.projected_count
            ),
            "pose_target_translation_error_m_clear_mean": self._mean(
                self.clear_lag_sum, self.clear_count
            ),
            "pose_target_translation_error_m_onset_mean": self._mean(
                self.onset_lag_sum, self.onset_count
            ),
            "pose_target_lag_blocked_fraction_correlation": self._correlation(),
        }

    @staticmethod
    def _mean(total: float, count: int) -> float | None:
        return None if count == 0 else total / count

    def _correlation(self) -> float | None:
        if self.sample_count < 2:
            return None
        count = float(self.sample_count)
        lag_variance_sum = self.lag_square_sum - self.lag_sum * self.lag_sum / count
        blocked_variance_sum = (
            self.blocked_square_sum - self.blocked_sum * self.blocked_sum / count
        )
        if lag_variance_sum <= 1e-18 or blocked_variance_sum <= 1e-18:
            return None
        covariance_sum = (
            self.lag_blocked_product_sum - self.lag_sum * self.blocked_sum / count
        )
        return float(
            np.clip(
                covariance_sum / np.sqrt(lag_variance_sum * blocked_variance_sum),
                -1.0,
                1.0,
            )
        )


__all__ = ["PoseTargetDiagnostics"]
