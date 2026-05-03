from __future__ import annotations

import hashlib
from typing import Any

import torch

from .base import BaseMedProvider


def _stable_seed(*parts: Any) -> int:
    joined = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**31)


class MockMedProvider(BaseMedProvider):
    name = "Mock_Med"

    def __init__(self, config: Any) -> None:
        self.config = config

    def build_med(self, sample: Any) -> torch.Tensor:
        state = sample.observations["state"].to(dtype=torch.float32)
        if str(self.config.mode) == "gaussian":
            return self._build_gaussian_med(sample, state.device)
        return self._build_state_projected_med(sample, state)

    def infer_output_dim(self, data_summary: dict[str, Any]) -> int | None:
        del data_summary
        return int(self.config.dim)

    def _build_gaussian_med(self, sample: Any, device: torch.device) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _stable_seed(
                int(self.config.base_seed),
                sample.metadata["dataset_id"],
                sample.metadata["episode_index"],
                sample.metadata["frame_index"],
                "gaussian_med",
            )
        )
        med = torch.randn(int(self.config.dim), generator=generator, dtype=torch.float32)
        return (float(self.config.scale) * med).to(device=device)

    def _build_state_projected_med(self, sample: Any, state: torch.Tensor) -> torch.Tensor:
        med_dim = int(self.config.dim)
        base_seed = int(self.config.base_seed)
        dataset_id = sample.metadata["dataset_id"]
        state_dim = int(state.shape[0])
        projection = torch.empty(state_dim, med_dim, dtype=torch.float32)
        bias = torch.empty(med_dim, dtype=torch.float32)
        for column in range(med_dim):
            column_generator = torch.Generator()
            column_generator.manual_seed(_stable_seed(base_seed, dataset_id, "proj", column))
            projection[:, column] = torch.randn(state_dim, generator=column_generator, dtype=torch.float32) / max(1.0, float(state_dim) ** 0.5)
            bias[column] = torch.randn(1, generator=column_generator, dtype=torch.float32)[0]
        med = torch.tanh(state @ projection + 0.05 * bias)
        return float(self.config.scale) * med
