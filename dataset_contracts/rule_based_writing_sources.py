"""Pure source-pool contract for diverse rule-based writing collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .canonical_encoding import canonical_content_hash


RULE_BASED_WRITING_SOURCE_POOL_SCHEMA_V1 = (
    "dexteleop.rule_based_writing_source_pool.v1"
)
RULE_BASED_WRITING_SOURCE_POOL_HASH_DOMAIN_V1 = (
    "dexteleop.rule_based_writing_source_pool.semantic.v1"
)
RULE_BASED_WRITING_SOURCE_PHASES_V1 = ("pen_down", "write", "pen_up")


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


@dataclass(frozen=True)
class RuleBasedWritingSourceCandidateV1:
    candidate_id: str
    dataset_root: str
    episode_index: int
    detail_annotation_dir: str
    phases: tuple[str, ...] = RULE_BASED_WRITING_SOURCE_PHASES_V1
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _required_text(self.candidate_id, field_name="candidate_id"),
        )
        object.__setattr__(
            self,
            "dataset_root",
            _required_text(self.dataset_root, field_name="dataset_root"),
        )
        object.__setattr__(
            self,
            "detail_annotation_dir",
            _required_text(
                self.detail_annotation_dir,
                field_name="detail_annotation_dir",
            ),
        )
        if isinstance(self.episode_index, bool) or int(self.episode_index) < 0:
            raise ValueError("episode_index must be a non-negative integer.")
        object.__setattr__(self, "episode_index", int(self.episode_index))
        if any(not isinstance(value, str) for value in self.phases):
            raise ValueError("phases must contain strings only.")
        phases = tuple(value.strip() for value in self.phases)
        if not phases or len(phases) != len(set(phases)):
            raise ValueError("phases must be non-empty and contain no duplicates.")
        unsupported = sorted(set(phases) - set(RULE_BASED_WRITING_SOURCE_PHASES_V1))
        if unsupported:
            raise ValueError(f"unsupported rule writing phases: {unsupported}.")
        object.__setattr__(self, "phases", phases)
        if any(not isinstance(value, str) for value in self.tags):
            raise ValueError("tags must contain strings only.")
        tags = tuple(value.strip() for value in self.tags)
        if any(not value for value in tags) or len(tags) != len(set(tags)):
            raise ValueError("tags must be non-empty unique strings.")
        object.__setattr__(self, "tags", tags)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuleBasedWritingSourceCandidateV1":
        allowed = {
            "candidate_id",
            "dataset_root",
            "episode_index",
            "detail_annotation_dir",
            "phases",
            "tags",
        }
        unknown = sorted(set(payload) - allowed)
        missing = sorted(
            {"candidate_id", "dataset_root", "episode_index", "detail_annotation_dir"}
            - set(payload)
        )
        if unknown or missing:
            raise ValueError(
                f"invalid source candidate fields: missing={missing}, unknown={unknown}."
            )
        phases = payload.get("phases", RULE_BASED_WRITING_SOURCE_PHASES_V1)
        tags = payload.get("tags", ())
        if not isinstance(phases, (list, tuple)) or not isinstance(tags, (list, tuple)):
            raise ValueError("candidate phases and tags must be arrays.")
        episode_index = payload["episode_index"]
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise ValueError("episode_index must be a non-negative integer.")
        return cls(
            candidate_id=payload["candidate_id"],
            dataset_root=payload["dataset_root"],
            episode_index=episode_index,
            detail_annotation_dir=payload["detail_annotation_dir"],
            phases=tuple(phases),
            tags=tuple(tags),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "dataset_root": self.dataset_root,
            "episode_index": self.episode_index,
            "detail_annotation_dir": self.detail_annotation_dir,
            "phases": list(self.phases),
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class RuleBasedWritingSourcePoolV1:
    pool_id: str
    candidates: tuple[RuleBasedWritingSourceCandidateV1, ...]
    schema: str = RULE_BASED_WRITING_SOURCE_POOL_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != RULE_BASED_WRITING_SOURCE_POOL_SCHEMA_V1:
            raise ValueError(f"unsupported source-pool schema {self.schema!r}.")
        object.__setattr__(self, "pool_id", _required_text(self.pool_id, field_name="pool_id"))
        candidates = tuple(self.candidates)
        if not candidates or any(
            not isinstance(value, RuleBasedWritingSourceCandidateV1)
            for value in candidates
        ):
            raise ValueError("source pool requires at least one typed candidate.")
        candidate_ids = tuple(value.candidate_id for value in candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("source-pool candidate_id values must be unique.")
        object.__setattr__(self, "candidates", candidates)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuleBasedWritingSourcePoolV1":
        if not isinstance(payload, Mapping):
            raise ValueError("source-pool manifest must be a JSON object.")
        allowed = {"schema", "pool_id", "candidates"}
        unknown = sorted(set(payload) - allowed)
        missing = sorted(allowed - set(payload))
        if unknown or missing:
            raise ValueError(
                f"invalid source-pool fields: missing={missing}, unknown={unknown}."
            )
        raw_candidates = payload["candidates"]
        if not isinstance(raw_candidates, list):
            raise ValueError("source-pool candidates must be an array.")
        candidates: list[RuleBasedWritingSourceCandidateV1] = []
        for value in raw_candidates:
            if not isinstance(value, Mapping):
                raise ValueError("every source-pool candidate must be an object.")
            candidates.append(RuleBasedWritingSourceCandidateV1.from_dict(value))
        return cls(
            schema=payload["schema"],
            pool_id=payload["pool_id"],
            candidates=tuple(candidates),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "pool_id": self.pool_id,
            "candidates": [value.to_dict() for value in self.candidates],
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(),
            domain=RULE_BASED_WRITING_SOURCE_POOL_HASH_DOMAIN_V1,
        )


__all__ = [
    "RULE_BASED_WRITING_SOURCE_PHASES_V1",
    "RULE_BASED_WRITING_SOURCE_POOL_HASH_DOMAIN_V1",
    "RULE_BASED_WRITING_SOURCE_POOL_SCHEMA_V1",
    "RuleBasedWritingSourceCandidateV1",
    "RuleBasedWritingSourcePoolV1",
]
