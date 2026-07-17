"""Numerically stable observation normalization shared by train/eval/viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class RunningMeanStd:
    mean: np.ndarray
    var: np.ndarray
    count: float
    clip: float = 10.0

    @classmethod
    def create(cls, shape: tuple[int, ...], *, epsilon: float = 1e-4) -> "RunningMeanStd":
        return cls(
            mean=np.zeros(shape, dtype=np.float64),
            var=np.ones(shape, dtype=np.float64),
            count=float(epsilon),
        )

    def update(self, values: np.ndarray) -> None:
        batch = np.asarray(values, dtype=np.float64)
        if batch.ndim == self.mean.ndim:
            batch = batch.reshape((1, *batch.shape))
        if batch.shape[1:] != self.mean.shape or batch.shape[0] == 0:
            raise ValueError(f"Expected observation batch (*, {self.mean.shape}), got {batch.shape}.")
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = float(batch.shape[0])
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        self.mean = new_mean
        self.var = np.maximum(m2 / total, 1e-12)
        self.count = total

    def normalize(self, values: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(values, dtype=np.float32) - self.mean) / np.sqrt(
            self.var + 1e-8
        )
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def to_state(self) -> dict[str, Any]:
        return {
            "mean": self.mean.astype(np.float64).tolist(),
            "var": self.var.astype(np.float64).tolist(),
            "count": float(self.count),
            "clip": float(self.clip),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "RunningMeanStd":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float64),
            var=np.asarray(state["var"], dtype=np.float64),
            count=float(state["count"]),
            clip=float(state.get("clip", 10.0)),
        )


__all__ = ["RunningMeanStd"]
