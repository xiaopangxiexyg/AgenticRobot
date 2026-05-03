from __future__ import annotations

from typing import Any

import torch

from .base import BaseMedProvider


def _product(values: list[int]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


class FutureSequenceMedProvider(BaseMedProvider):
    name = "FutureSequence_Med"

    def __init__(self, config: Any) -> None:
        self.config = config

    def _resolve_kind_from_sample(self, sample: Any) -> str:
        return str(sample.metadata.get("med_source_kind", "vector")).lower()

    def _resolve_kind_from_summary(self, data_summary: dict[str, Any]) -> str:
        return str(data_summary.get("med_source_kind", "vector")).lower()

    def _resolve_source_key(self, sample: Any) -> str:
        configured_key = getattr(self.config, "source_intermediate_key", None)
        if configured_key is not None and str(configured_key).lower() not in {"auto", "null"}:
            return str(configured_key)
        metadata_key = sample.metadata.get("med_intermediate_key")
        if metadata_key is not None and str(metadata_key).lower() not in {"auto", "null"}:
            return str(metadata_key)
        return "med"

    def _should_flatten_sequence(self, med_kind: str) -> bool:
        configured = getattr(self.config, "flatten_sequence", "auto")
        if isinstance(configured, str):
            normalized = str(configured).lower()
            if normalized == "auto":
                return med_kind == "vector"
            if normalized in {"true", "yes", "1"}:
                return True
            if normalized in {"false", "no", "0"}:
                return False
        if str(configured).lower() == "auto":
            return med_kind == "vector"
        return bool(configured)

    def _resolve_source_shape(self, data_summary: dict[str, Any], med_kind: str) -> list[int] | None:
        med_shape = data_summary.get("med_source_shape")
        if med_shape is not None:
            return [int(value) for value in med_shape]
        if med_kind == "vector":
            tactile_dim = data_summary.get("tactile_dim")
            future_steps = data_summary.get("future_tactile_steps")
            if tactile_dim is None or future_steps is None:
                return None
            return [int(future_steps), int(tactile_dim)]
        return None

    def build_med(self, sample: Any) -> torch.Tensor:
        source_key = self._resolve_source_key(sample)
        if source_key not in sample.intermediates:
            raise KeyError(f"Sample intermediates missing med source key: {source_key}")
        med_value = sample.intermediates[source_key].to(dtype=torch.float32)
        med_kind = self._resolve_kind_from_sample(sample)
        if not self._should_flatten_sequence(med_kind):
            return med_value
        if med_kind == "point_cloud" and med_value.dim() == 3:
            time_steps, point_count, feature_dim = med_value.shape
            return med_value.reshape(time_steps * point_count, feature_dim)
        return med_value.reshape(-1)

    def infer_output_dim(self, data_summary: dict[str, Any]) -> int | None:
        med_spec = self.infer_output_spec(data_summary)
        if med_spec is None:
            return None
        return int(med_spec["flat_dim"])

    def infer_output_spec(self, data_summary: dict[str, Any]) -> dict[str, Any] | None:
        med_kind = self._resolve_kind_from_summary(data_summary)
        source_shape = self._resolve_source_shape(data_summary, med_kind)
        if source_shape is None:
            return None
        output_shape = list(source_shape)
        if self._should_flatten_sequence(med_kind):
            if med_kind == "point_cloud" and len(output_shape) >= 2:
                output_shape = [_product(output_shape[:-1]), int(output_shape[-1])]
            else:
                output_shape = [_product(output_shape)]
        flat_dim = _product(output_shape)
        output_spec = {
            "kind": med_kind,
            "shape": [int(value) for value in output_shape],
            "flat_dim": int(flat_dim),
            "provider": self.name,
        }
        if med_kind == "point_cloud":
            output_spec["point_feature_dim"] = int(output_shape[-1])
            if len(output_shape) >= 3 and not self._should_flatten_sequence(med_kind):
                output_spec["sequence_length"] = _product(output_shape[:-2])
                output_spec["point_count"] = int(output_shape[-2])
                output_spec["sequence_point_count"] = _product(output_shape[:-1])
            else:
                output_spec["point_count"] = _product(output_shape[:-1]) if len(output_shape) > 1 else 1
        return output_spec
