"""Pure planning for native v2.5-p to v2.3 compatibility projection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from dataset_contracts.embodiment import (
    CaptureProjectionComponentV2,
    CaptureProjectionProfileLike,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from dataset_contracts.v25p import v25p_base_episode_manifest_hash
from dataset_contracts.v25p_native import (
    V25pComponentStreamProfileLike,
    V25pComponentStreamSpecLike,
)
from dataset_contracts.v25p_projection import V25pCompatibilityProjectionReceipt


class V25pV23ProjectionError(ValueError):
    """Raised when a native stream profile cannot project without guessing."""


@dataclass(frozen=True)
class V25pV23ProjectionPlan:
    projection_profile_hash: str
    component_stream_profile_hash: str
    stream_id_by_dataset_field: Mapping[str, str]
    omitted_stream_ids: tuple[str, ...]
    missing_optional_dataset_fields: tuple[str, ...]

    @property
    def lossy(self) -> bool:
        return bool(self.omitted_stream_ids)


def _expected_stage(
    profile: CaptureProjectionProfileLike,
    dataset_role: str,
) -> str | None:
    if dataset_role == "action":
        return profile.action_stage
    if dataset_role == "state":
        return profile.state_stage
    return None


def _candidate_streams(
    *,
    stream_profile: V25pComponentStreamProfileLike,
    component_id: str,
    dataset_role: str,
    payload_schema_hash: str,
    value_stage: str | None,
) -> tuple[V25pComponentStreamSpecLike, ...]:
    return tuple(
        stream
        for stream in stream_profile.streams
        if stream.component_id == component_id
        and stream.dataset_role == dataset_role
        and stream.payload_schema_hash == payload_schema_hash
        and (value_stage is None or stream.value_stage == value_stage)
    )


def plan_v23_compatibility_projection(
    *,
    stream_profile: V25pComponentStreamProfileLike,
    projection_profile: CaptureProjectionProfileLike,
) -> V25pV23ProjectionPlan:
    """Resolve every v2.3 field to one native stream, without heuristics."""

    if projection_profile.base_dataset_schema != DEXTELEOP_DATASET_SCHEMA_V23:
        raise V25pV23ProjectionError(
            "The compatibility projection does not target DexTeleop v2.3."
        )
    selected: dict[str, str] = {}
    missing_optional: list[str] = []
    for component in projection_profile.components:
        if isinstance(component, CaptureProjectionComponentV2):
            stream = stream_profile.stream_by_id.get(component.stream_id)
            if stream is None:
                if component.required:
                    raise V25pV23ProjectionError(
                        "Required v2.3 projection field references an unknown "
                        f"native stream: field={component.dataset_field!r} "
                        f"stream={component.stream_id!r}."
                    )
                missing_optional.append(component.dataset_field)
                continue
            if stream.storage_kind != "inline_tensor":
                raise V25pV23ProjectionError(
                    "v2.3 scalar/list fields cannot directly project an external "
                    f"artifact stream: field={component.dataset_field!r} "
                    f"stream={component.stream_id!r}."
                )
            selected[component.dataset_field] = stream.stream_id
            continue
        expected_stage = _expected_stage(
            projection_profile,
            component.dataset_role,
        )
        candidates = _candidate_streams(
            stream_profile=stream_profile,
            component_id=component.component_id,
            dataset_role=component.dataset_role,
            payload_schema_hash=component.value_schema_hash,
            value_stage=expected_stage,
        )
        if not candidates:
            if component.required:
                raise V25pV23ProjectionError(
                    "Required v2.3 projection field has no native stream: "
                    f"field={component.dataset_field!r} "
                    f"component={component.component_id!r} "
                    f"role={component.dataset_role!r} stage={expected_stage!r}."
                )
            missing_optional.append(component.dataset_field)
            continue
        if len(candidates) != 1:
            raise V25pV23ProjectionError(
                "v2.3 projection field is ambiguous: "
                f"field={component.dataset_field!r} "
                f"streams={[item.stream_id for item in candidates]}."
            )
        if candidates[0].storage_kind != "inline_tensor":
            raise V25pV23ProjectionError(
                "v2.3 scalar/list fields cannot directly project an external "
                f"artifact stream: field={component.dataset_field!r} "
                f"stream={candidates[0].stream_id!r}."
            )
        selected[component.dataset_field] = candidates[0].stream_id

    selected_stream_ids = set(selected.values())
    omitted = tuple(
        stream.stream_id
        for stream in stream_profile.streams
        if stream.stream_id not in selected_stream_ids
    )
    return V25pV23ProjectionPlan(
        projection_profile_hash=projection_profile.semantic_content_hash,
        component_stream_profile_hash=stream_profile.semantic_content_hash,
        stream_id_by_dataset_field=MappingProxyType(dict(selected)),
        omitted_stream_ids=omitted,
        missing_optional_dataset_fields=tuple(missing_optional),
    )


def verify_complete_v23_projection_receipt(
    *,
    dataset_root: str | Path,
    identity: EpisodeCaptureIdentity,
    receipt: V25pCompatibilityProjectionReceipt,
) -> None:
    """Verify that a complete receipt resolves the exact finalized v2.3 row."""

    if receipt.status != "complete":
        raise V25pV23ProjectionError("Only a complete projection can be verified.")
    expected_locator = (
        "meta/episodes.jsonl#episode_index="
        f"{identity.legacy_episode_index}"
    )
    if receipt.target_episode_locator != expected_locator:
        raise V25pV23ProjectionError(
            "Complete v2.3 compatibility receipt has an unsupported target locator."
        )
    path = Path(dataset_root).expanduser().resolve() / "meta" / "episodes.jsonl"
    if not path.is_file():
        raise V25pV23ProjectionError(
            "Complete v2.3 compatibility receipt references a missing episode index."
        )
    matches: list[Mapping[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise V25pV23ProjectionError(
                    f"{path}:{line_number} is not a JSON object."
                )
            if int(payload.get("episode_index", -1)) == identity.legacy_episode_index:
                matches.append(payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V25pV23ProjectionError(
            f"Cannot verify complete v2.3 compatibility output: {exc}"
        ) from exc
    if len(matches) != 1:
        raise V25pV23ProjectionError(
            "Complete v2.3 compatibility receipt must resolve exactly one episode row."
        )
    if v25p_base_episode_manifest_hash(matches[0]) != receipt.target_episode_record_hash:
        raise V25pV23ProjectionError(
            "Complete v2.3 compatibility episode-record hash mismatch."
        )


__all__ = [
    "V25pV23ProjectionError",
    "V25pV23ProjectionPlan",
    "plan_v23_compatibility_projection",
    "verify_complete_v23_projection_receipt",
]
