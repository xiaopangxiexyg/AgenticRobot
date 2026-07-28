"""Default-off producer boundary for native provisional v2.5-p samples.

The application recorder remains the episode owner.  Producers only translate
one already-authoritative source snapshot into typed component inputs; they do
not persist data or advance any episode lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.v25p_native import V25pComponentStreamProfileLike

from .v25p_native_store import V25pComponentInput


@dataclass(frozen=True)
class V25pProducerBindings:
    """Explicit source-semantic key to native stream binding.

    A binding is intentionally supplied by the selected embodiment/profile;
    stream names and component IDs are never inferred from side or row names.
    """

    stream_id_by_value_key: Mapping[str, str]

    def __post_init__(self) -> None:
        normalized = {
            str(key).strip(): str(value).strip()
            for key, value in dict(self.stream_id_by_value_key).items()
        }
        if not normalized or any(not key or not value for key, value in normalized.items()):
            raise ValueError("Producer bindings must contain non-empty keys and stream IDs.")
        if len(set(normalized.values())) != len(normalized):
            raise ValueError("Each native stream may be bound to only one producer value key.")
        object.__setattr__(
            self,
            "stream_id_by_value_key",
            MappingProxyType(normalized),
        )


@dataclass(frozen=True)
class V25pProducedSample:
    sample_index: int
    cut_at: float
    cut_clock_domain_id: str
    values_by_stream: Mapping[str, V25pComponentInput]

    def __post_init__(self) -> None:
        if int(self.sample_index) < 0:
            raise ValueError("sample_index must be non-negative.")
        if not str(self.cut_clock_domain_id).strip():
            raise ValueError("cut_clock_domain_id must be non-empty.")
        object.__setattr__(self, "sample_index", int(self.sample_index))
        object.__setattr__(self, "cut_at", float(self.cut_at))
        object.__setattr__(
            self,
            "cut_clock_domain_id",
            str(self.cut_clock_domain_id).strip(),
        )
        object.__setattr__(
            self,
            "values_by_stream",
            MappingProxyType(dict(self.values_by_stream)),
        )


def validate_producer_bindings(
    *,
    profile: V25pComponentStreamProfileLike,
    bindings: V25pProducerBindings,
    expected_stage_by_value_key: Mapping[str, str],
) -> None:
    """Fail closed when a producer/profile pair would require guessing."""

    streams = profile.stream_by_id
    bound_stream_ids = set(bindings.stream_id_by_value_key.values())
    unknown = sorted(bound_stream_ids - set(streams))
    missing = sorted(set(profile.required_stream_ids) - bound_stream_ids)
    if unknown or missing:
        raise ValueError(
            f"Producer/profile coverage mismatch unknown={unknown} missing={missing}."
        )
    expected_stages = {
        str(key): str(value)
        for key, value in dict(expected_stage_by_value_key).items()
    }
    if set(expected_stages) != set(bindings.stream_id_by_value_key):
        raise ValueError(
            "Producer expected-stage keys must exactly match producer bindings."
        )
    for value_key, stream_id in bindings.stream_id_by_value_key.items():
        parts = value_key.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Producer value key {value_key!r} must declare role and kind."
            )
        role, kind = parts[:2]
        stream = streams[stream_id]
        expected_stage = expected_stages[value_key]
        if stream.dataset_role != role or stream.value_stage != expected_stage:
            raise ValueError(
                "Producer binding role/stage mismatch: "
                f"key={value_key!r} stream={stream_id!r} "
                f"actual={stream.dataset_role}/{stream.value_stage} "
                f"expected={role}/{expected_stage}."
            )
        if kind == "pose" and stream.representation_family not in {
            "xyz_quat_xyzw",
            "cartesian_pose",
        }:
            raise ValueError(
                f"Pose binding {value_key!r} must use a Cartesian-pose representation."
            )


def validate_produced_values(
    *,
    profile: V25pComponentStreamProfileLike,
    produced: V25pProducedSample,
) -> None:
    known = profile.stream_by_id
    actual = set(produced.values_by_stream)
    unknown = sorted(actual - set(known))
    missing = sorted(set(profile.required_stream_ids) - actual)
    if unknown or missing:
        raise ValueError(
            f"Produced native sample coverage mismatch unknown={unknown} missing={missing}."
        )
    for stream_id, value in produced.values_by_stream.items():
        if not isinstance(value, V25pComponentInput):
            raise TypeError("Produced stream values must be V25pComponentInput instances.")
        stream = known[stream_id]
        if value.valid and stream.storage_kind == "inline_tensor":
            stream.normalize_values(value.values)


__all__ = [
    "V25pProducedSample",
    "V25pProducerBindings",
    "validate_produced_values",
    "validate_producer_bindings",
]
