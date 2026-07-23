"""Deterministic encoding for a :mod:`sim.mujoco.learning` binding.

The encoder is intentionally small.  It does not know about Cube-Lift,
qpos, or a policy framework; adapters supply named numeric/image fields and
the same operation can therefore be used by a real deployment adapter later.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .contracts import ModelInputBinding, ObservationContract


@dataclass(frozen=True)
class EncodedModelInput:
    """Encoded fields plus the ordered flat view consumed by simple policies."""

    fields: Mapping[str, np.ndarray]
    flat: np.ndarray
    observation_contract_hash: str
    binding_hash: str


def _as_mapping(observation: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(observation, Mapping):
        return observation
    model_input = getattr(observation, "model_input", None)
    if callable(model_input):
        value = model_input()
        if isinstance(value, Mapping):
            return value
    raise TypeError("observation must be a mapping or expose model_input()")


def _sample_shape(array: np.ndarray, shape: tuple[int | None, ...] | None) -> tuple[int, ...]:
    if shape is None:
        return tuple(int(size) for size in array.shape)
    tail_ndim = len(shape)
    if array.ndim < tail_ndim:
        raise ValueError(
            f"observation field has rank {array.ndim}, expected at least {tail_ndim}"
        )
    tail = array.shape[-tail_ndim:] if tail_ndim else ()
    for expected, actual in zip(shape, tail):
        if expected is not None and int(expected) != int(actual):
            raise ValueError(f"observation field shape mismatch: expected {shape}, got {array.shape}")
    return tuple(int(size) for size in tail)


def _normalise(array: np.ndarray, field_spec: Any) -> np.ndarray:
    normalized = field_spec.normalization
    if normalized == "identity":
        return array
    if normalized == "unit_interval":
        return np.clip(array, 0.0, 1.0)
    if normalized == "symmetric_unit":
        return np.clip(array, -1.0, 1.0)
    if normalized == "scale_offset":
        return array * field_spec.scale + field_spec.offset
    raise ValueError(f"Unsupported normalization: {normalized!r}")


def encode_model_input(
    observation: Mapping[str, Any] | Any,
    contract: ObservationContract,
    binding: ModelInputBinding,
) -> EncodedModelInput:
    """Validate and encode one or more samples according to ``binding``.

    A leading batch dimension is preserved.  Missing optional fields are
    omitted only when the binding does not request them; a requested field is
    always an error, which prevents silent sim/real input drift.
    """

    binding.validate(contract)
    source = _as_mapping(observation)
    encoded: dict[str, np.ndarray] = {}
    leading_shape: tuple[int, ...] | None = None
    flat_parts: list[np.ndarray] = []
    for name in binding.input_fields:
        field_spec = contract.field(name)
        if name not in source:
            # A binding is an explicit model-input admission list. Once a
            # field is listed, omitting it would change the model shape and
            # therefore cannot be treated as a benign optional-field case.
            raise KeyError(f"Missing required model input field: {name!r}")
        try:
            array = np.asarray(source[name], dtype=np.dtype(field_spec.dtype))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Field {name!r} cannot be converted to {field_spec.dtype}") from exc
        if array.dtype.hasobject:
            raise ValueError(f"Field {name!r} must be numeric, not object dtype")
        _sample_shape(array, field_spec.shape)
        tail_ndim = len(field_spec.shape or ())
        current_leading = array.shape[:-tail_ndim] if tail_ndim else array.shape
        if leading_shape is None:
            leading_shape = tuple(int(size) for size in current_leading)
        elif tuple(int(size) for size in current_leading) != leading_shape:
            raise ValueError(
                f"Leading batch shape mismatch for {name!r}: expected {leading_shape}, "
                f"got {current_leading}"
            )
        array = np.ascontiguousarray(_normalise(array, field_spec), dtype=np.float32)
        encoded[name] = array
        flat_parts.append(array.reshape(*leading_shape, -1) if leading_shape else array.reshape(-1))
    if not flat_parts:
        raise ValueError("ModelInputBinding produced no encoded fields")
    flat = np.concatenate(flat_parts, axis=-1)
    expected = binding.expected_flat_dim(contract)
    if flat.shape[-1] != expected:
        raise ValueError(f"Encoded model input dim mismatch: expected {expected}, got {flat.shape}")
    return EncodedModelInput(
        fields=MappingProxyType(encoded),
        flat=np.ascontiguousarray(flat, dtype=np.float32),
        observation_contract_hash=contract.semantic_content_hash,
        binding_hash=binding.semantic_content_hash,
    )


__all__ = ["EncodedModelInput", "encode_model_input"]
