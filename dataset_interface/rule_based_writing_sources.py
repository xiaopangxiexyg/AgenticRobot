"""Filesystem resolution for rule-based writing source-pool manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset_contracts.rule_based_writing_sources import (
    RuleBasedWritingSourceCandidateV1,
    RuleBasedWritingSourcePoolV1,
)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"source-pool JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


@dataclass(frozen=True)
class ResolvedRuleBasedWritingSourceCandidate:
    candidate: RuleBasedWritingSourceCandidateV1
    dataset_root: Path
    detail_annotation_dir: Path


@dataclass(frozen=True)
class LoadedRuleBasedWritingSourcePool:
    manifest_path: Path
    contract: RuleBasedWritingSourcePoolV1
    candidates: tuple[ResolvedRuleBasedWritingSourceCandidate, ...]


def _resolve_manifest_path(value: str, *, manifest_dir: Path, field_name: str) -> Path:
    raw = Path(value).expanduser()
    resolved = (raw if raw.is_absolute() else manifest_dir / raw).resolve()
    if not resolved.is_dir():
        raise ValueError(f"{field_name} does not exist or is not a directory: {resolved}")
    return resolved


def load_rule_based_writing_source_pool(
    manifest_path: str | Path,
) -> LoadedRuleBasedWritingSourcePool:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"source-pool manifest does not exist: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read source-pool manifest {path}: {exc}") from exc
    contract = RuleBasedWritingSourcePoolV1.from_dict(payload)
    resolved = tuple(
        ResolvedRuleBasedWritingSourceCandidate(
            candidate=value,
            dataset_root=_resolve_manifest_path(
                value.dataset_root,
                manifest_dir=path.parent,
                field_name=f"candidate {value.candidate_id!r} dataset_root",
            ),
            detail_annotation_dir=_resolve_manifest_path(
                value.detail_annotation_dir,
                manifest_dir=path.parent,
                field_name=f"candidate {value.candidate_id!r} detail_annotation_dir",
            ),
        )
        for value in contract.candidates
    )
    return LoadedRuleBasedWritingSourcePool(
        manifest_path=path,
        contract=contract,
        candidates=resolved,
    )


__all__ = [
    "LoadedRuleBasedWritingSourcePool",
    "ResolvedRuleBasedWritingSourceCandidate",
    "load_rule_based_writing_source_pool",
]
