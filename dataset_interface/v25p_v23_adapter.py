"""Explicit projection of native v2.5-p component samples into v2.3 rows.

The native episode remains authoritative.  This module is a read-only adapter:
it neither owns an episode lifecycle nor writes a v2.3 dataset.  A caller may
hand the projected component fields to the existing v2.3 row encoder/writer
and publish a compatibility receipt only after that writer has finalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.embodiment import CaptureProjectionProfileLike

from .v25p_native import V25pLoadedNativeEpisode
from .v25p_projection_plan import (
    V25pV23ProjectionError,
    V25pV23ProjectionPlan,
    plan_v23_compatibility_projection,
)


@dataclass(frozen=True)
class V25pV23ProjectedRow:
    sample_index: int
    cut_at: float
    cut_clock_domain_id: str
    fields: Mapping[str, tuple[bool | int | float, ...]]
    dtype_by_field: Mapping[str, str]
    shape_by_field: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class V25pV23ProjectionResult:
    plan: V25pV23ProjectionPlan
    rows: tuple[V25pV23ProjectedRow, ...]


def project_native_episode_to_v23_rows(
    episode: V25pLoadedNativeEpisode,
    projection_profile: CaptureProjectionProfileLike,
) -> V25pV23ProjectionResult:
    """Materialize v2.3 component fields from a verified native episode."""

    if (
        projection_profile.semantic_content_hash
        not in episode.dataset_manifest.compatibility_projection_profile_hashes
    ):
        raise V25pV23ProjectionError(
            "Projection profile is not bound by the native dataset manifest."
        )
    plan = plan_v23_compatibility_projection(
        stream_profile=episode.component_stream_profile,
        projection_profile=projection_profile,
    )
    record_by_hash = {
        record.semantic_content_hash: record
        for records in episode.records_by_stream.values()
        for record in records
    }
    rows: list[V25pV23ProjectedRow] = []
    for sample in episode.samples:
        fields: dict[str, tuple[bool | int | float, ...]] = {}
        dtype_by_field: dict[str, str] = {}
        shape_by_field: dict[str, tuple[int, ...]] = {}
        for dataset_field, stream_id in plan.stream_id_by_dataset_field.items():
            record_hash = sample.record_hash_by_stream.get(stream_id)
            record = None if record_hash is None else record_by_hash.get(record_hash)
            if record is None:
                raise V25pV23ProjectionError(
                    "Selected native stream is missing from a causal sample: "
                    f"sample={sample.sample_index} stream={stream_id!r}."
                )
            if not record.valid or record.values is None:
                raise V25pV23ProjectionError(
                    "Selected native stream contains an invalid value: "
                    f"sample={sample.sample_index} stream={stream_id!r} "
                    f"reason={record.invalid_reason!r}."
                )
            fields[dataset_field] = record.values
            stream = episode.component_stream_profile.stream_by_id[stream_id]
            dtype_by_field[dataset_field] = stream.dtype
            shape_by_field[dataset_field] = stream.shape
        rows.append(
            V25pV23ProjectedRow(
                sample_index=sample.sample_index,
                cut_at=sample.cut_at,
                cut_clock_domain_id=sample.cut_clock_domain_id,
                fields=MappingProxyType(fields),
                dtype_by_field=MappingProxyType(dtype_by_field),
                shape_by_field=MappingProxyType(shape_by_field),
            )
        )
    return V25pV23ProjectionResult(plan=plan, rows=tuple(rows))


__all__ = [
    "V25pV23ProjectedRow",
    "V25pV23ProjectionError",
    "V25pV23ProjectionPlan",
    "V25pV23ProjectionResult",
    "plan_v23_compatibility_projection",
    "project_native_episode_to_v23_rows",
]
