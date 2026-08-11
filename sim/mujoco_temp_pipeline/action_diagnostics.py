"""Shared action-path diagnostics for temporary PPO training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActionDiagnostics:
    """Accumulate latent, public-action, and task-decoder statistics.

    The accumulator is deliberately observational.  In particular, it never
    substitutes the public or admitted action for the latent sample used by
    PPO likelihood accounting.
    """

    saturation_threshold: float = 0.95
    policy_abs_sum: float = 0.0
    policy_component_count: int = 0
    policy_saturated_count: int = 0
    task_abs_sum: float = 0.0
    task_component_count: int = 0
    conditioning_scale_sum: float = 0.0
    conditioning_scale_count: int = 0
    conditioning_active_count: int = 0
    conditioning_scale_minimum: float = float("inf")

    def __post_init__(self) -> None:
        threshold = float(self.saturation_threshold)
        if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("saturation_threshold must lie in (0, 1].")
        self.saturation_threshold = threshold

    def update(
        self,
        policy_action: np.ndarray,
        task_action: np.ndarray,
        conditioning_scale: np.ndarray,
    ) -> None:
        """Record one scalar or batched action-path sample."""

        policy = np.asarray(policy_action, dtype=np.float32)
        task = np.asarray(task_action, dtype=np.float32)
        scale = np.asarray(conditioning_scale, dtype=np.float32)
        if policy.ndim == 0 or task.ndim == 0:
            raise ValueError("Policy and task actions must have a component axis.")
        if (
            policy.shape[:-1] != task.shape[:-1]
            or policy.shape[:-1] != scale.shape
        ):
            raise ValueError(
                "Policy action, task action, and conditioning scale batch "
                f"shapes must match: {policy.shape}, {task.shape}, {scale.shape}."
            )
        if (
            not np.all(np.isfinite(policy))
            or not np.all(np.isfinite(task))
            or not np.all(np.isfinite(scale))
        ):
            raise ValueError("Action diagnostics require finite values.")
        if np.any(scale <= 0.0) or np.any(scale > 1.0 + 1e-6):
            raise ValueError("Action conditioning scale must lie in (0, 1].")

        self.policy_abs_sum += float(np.sum(np.abs(policy)))
        self.policy_component_count += int(policy.size)
        self.policy_saturated_count += int(
            np.count_nonzero(np.abs(policy) >= self.saturation_threshold)
        )
        self.task_abs_sum += float(np.sum(np.abs(task)))
        self.task_component_count += int(task.size)
        self.conditioning_scale_sum += float(np.sum(scale))
        self.conditioning_scale_count += int(scale.size)
        self.conditioning_active_count += int(
            np.count_nonzero(scale < (1.0 - 1e-6))
        )
        if scale.size:
            self.conditioning_scale_minimum = min(
                self.conditioning_scale_minimum,
                float(np.min(scale)),
            )

    def summary(self) -> dict[str, float | None]:
        """Return stable metric names shared by train and evaluate outputs."""

        return {
            "policy_action_abs_mean": (
                self.policy_abs_sum / max(self.policy_component_count, 1)
            ),
            "policy_action_saturation_rate": (
                self.policy_saturated_count
                / max(self.policy_component_count, 1)
            ),
            "task_action_abs_mean": (
                self.task_abs_sum / max(self.task_component_count, 1)
            ),
            "action_conditioning_scale_mean": (
                self.conditioning_scale_sum
                / max(self.conditioning_scale_count, 1)
            ),
            "action_conditioning_scale_min": (
                None
                if not np.isfinite(self.conditioning_scale_minimum)
                else self.conditioning_scale_minimum
            ),
            "action_conditioning_active_rate": (
                self.conditioning_active_count
                / max(self.conditioning_scale_count, 1)
            ),
        }


__all__ = ["ActionDiagnostics"]
