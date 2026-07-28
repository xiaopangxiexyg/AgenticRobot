"""Collection-owned timing attachment for projected component actions.

The dataset-interface projector emits values and semantic bindings only.  This
module is the deliberately thin boundary where an existing recorder-owned cut
attaches causal time before the values enter a native v2.5-p episode store.
It does not compose actions, generate timestamps, or own episode lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.v25p_native import V25pComponentStreamProfileV2
from dataset_interface.component_action_projection import ProjectedComponentAction

from .v25p_native_store import V25pComponentInput


@dataclass(frozen=True)
class V25pActionComponentTiming:
    effective_at: float
    available_at: float
    effective_clock_domain_id: str
    available_clock_domain_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "effective_at", float(self.effective_at))
        object.__setattr__(self, "available_at", float(self.available_at))
        for name in (
            "effective_clock_domain_id",
            "available_clock_domain_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError("Projected action clock-domain IDs must be non-empty.")
            object.__setattr__(self, name, value)


def attach_projected_component_action_timings(
    *,
    projected: ProjectedComponentAction,
    profile: V25pComponentStreamProfileV2,
    timing_by_stream: Mapping[str, V25pActionComponentTiming],
) -> Mapping[str, V25pComponentInput]:
    """Attach exact per-component source timing supplied by collection."""

    if not isinstance(projected, ProjectedComponentAction):
        raise TypeError("projected must be a ProjectedComponentAction.")
    if not isinstance(profile, V25pComponentStreamProfileV2):
        raise TypeError("profile must be a V25pComponentStreamProfileV2.")
    if projected.component_action_space_schema_hash != (
        profile.component_action_space_schema.semantic_content_hash
    ):
        raise ValueError("Projected action and native profile action spaces differ.")
    timings = dict(timing_by_stream)
    if set(timings) != set(projected.values_by_stream):
        raise ValueError(
            "Projected action timing coverage must exactly match projected streams."
        )

    inputs: dict[str, V25pComponentInput] = {}
    for stream_id, projected_value in projected.values_by_stream.items():
        timing = timings[stream_id]
        if not isinstance(timing, V25pActionComponentTiming):
            raise TypeError(
                "timing_by_stream values must be V25pActionComponentTiming."
            )
        stream = profile.stream_by_id.get(stream_id)
        if stream is None or stream.dataset_role != "action":
            raise ValueError(
                f"Projected action references a non-action or unknown stream: {stream_id!r}."
            )
        if (
            stream.component_id != projected_value.component_id
            or stream.action_component_id != projected_value.action_component_id
            or stream.payload_schema_hash != projected_value.payload_schema_hash
        ):
            raise ValueError(
                f"Projected action binding changed before collection: {stream_id!r}."
            )
        normalized = stream.normalize_values(projected_value.values)
        inputs[stream_id] = V25pComponentInput(
            values=normalized,
            effective_at=timing.effective_at,
            available_at=timing.available_at,
            effective_clock_domain_id=timing.effective_clock_domain_id,
            available_clock_domain_id=timing.available_clock_domain_id,
            source_record_id=projected.component_action_hash,
        )
    return MappingProxyType(inputs)


def attach_projected_component_action_timing(
    *,
    projected: ProjectedComponentAction,
    profile: V25pComponentStreamProfileV2,
    effective_at: float,
    available_at: float,
    effective_clock_domain_id: str,
    available_clock_domain_id: str,
) -> Mapping[str, V25pComponentInput]:
    """Attach owner-supplied causal time without changing projected values."""

    timing = V25pActionComponentTiming(
        effective_at=effective_at,
        available_at=available_at,
        effective_clock_domain_id=effective_clock_domain_id,
        available_clock_domain_id=available_clock_domain_id,
    )
    return attach_projected_component_action_timings(
        projected=projected,
        profile=profile,
        timing_by_stream={
            stream_id: timing for stream_id in projected.values_by_stream
        },
    )


__all__ = [
    "V25pActionComponentTiming",
    "attach_projected_component_action_timing",
    "attach_projected_component_action_timings",
]
