"""Build the strict writing-stage1 cohort from read-only logical manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash


WRITING_STAGE1_COHORT_PLAN_SCHEMA_V1 = (
    "dexteleop.writing_stage1_cohort_plan.v1"
)
WRITING_STAGE1_COHORT_PLAN_HASH_DOMAIN_V1 = (
    "dataset_interface.writing_stage1_cohort_plan.semantic.v1"
)

_LOGICAL_EPISODE_REF_SCHEMA_V1 = "dexteleop.logical_episode_ref.v1"
_FRONT_SLOTS = ("base_0_rgb",)
_THREE_VIEW_SLOTS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)
_MEDIA_CAMERA_KEY_BY_SLOT = {
    "base_0_rgb": "front",
    "left_wrist_0_rgb": "left_wrist",
    "right_wrist_0_rgb": "right_wrist",
}
_REQUIRED_MEDIA_CHECKS = frozenset(
    {"declared_frame_count", "decoder_frame_count", "tail_decode"}
)


class WritingStage1CohortError(ValueError):
    """Raised when the strict 7.23/7.24 cohort cannot be proven."""


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WritingStage1CohortError(f"{name} must be a non-empty string.")
    return value.strip()


def _identity(value: object) -> str:
    result = _required(value, name="episode_identity")
    prefix = "sha256:"
    digest = result[len(prefix) :] if result.startswith(prefix) else result
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise WritingStage1CohortError(
            "episode_identity must be a lowercase SHA-256 identity."
        )
    return digest


def _non_negative_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise WritingStage1CohortError(
            f"{name} must be a non-negative integer."
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    result = _non_negative_int(value, name=name)
    if result == 0:
        raise WritingStage1CohortError(f"{name} must be positive.")
    return result


@dataclass(frozen=True)
class WritingStage1EpisodeRef:
    """One strict logical Episode retaining both current and provenance locators."""

    episode_identity_hash: str
    batch_id: str
    frame_count: int
    source_dataset_root: str
    source_episode_index: int
    provenance_dataset_root: str
    provenance_episode_index: int
    admitted_camera_slots: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "episode_identity_hash",
            _identity(self.episode_identity_hash),
        )
        for name in (
            "batch_id",
            "source_dataset_root",
            "provenance_dataset_root",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "frame_count",
            _positive_int(self.frame_count, name="frame_count"),
        )
        for name in ("source_episode_index", "provenance_episode_index"):
            object.__setattr__(
                self,
                name,
                _non_negative_int(getattr(self, name), name=name),
            )
        slots = tuple(
            _required(value, name="admitted_camera_slots")
            for value in self.admitted_camera_slots
        )
        if not slots or len(slots) != len(set(slots)):
            raise WritingStage1CohortError(
                "admitted_camera_slots must be non-empty and unique."
            )
        object.__setattr__(self, "admitted_camera_slots", slots)

    @property
    def three_view_valid(self) -> bool:
        return self.admitted_camera_slots == _THREE_VIEW_SLOTS

    def to_payload(self) -> dict[str, Any]:
        return {
            "episode_identity_hash": self.episode_identity_hash,
            "batch_id": self.batch_id,
            "frame_count": self.frame_count,
            "source_dataset_root": self.source_dataset_root,
            "source_episode_index": self.source_episode_index,
            "provenance_dataset_root": self.provenance_dataset_root,
            "provenance_episode_index": self.provenance_episode_index,
            "admitted_camera_slots": self.admitted_camera_slots,
        }


@dataclass(frozen=True)
class WritingStage1CohortPlan:
    """Immutable 479 + 240 cohort plan; no raw artifacts are materialized."""

    front_manifest_path: str
    front_manifest_sha256: str
    three_view_manifest_path: str
    three_view_manifest_sha256: str
    three_view_required_provenance_root: str
    front_0723: tuple[WritingStage1EpisodeRef, ...]
    three_view_0724: tuple[WritingStage1EpisodeRef, ...]
    excluded_three_view_other_provenance_count: int
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=WRITING_STAGE1_COHORT_PLAN_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "front_manifest_path",
            "front_manifest_sha256",
            "three_view_manifest_path",
            "three_view_manifest_sha256",
            "three_view_required_provenance_root",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        front = tuple(self.front_0723)
        three = tuple(self.three_view_0724)
        if any(not isinstance(item, WritingStage1EpisodeRef) for item in front + three):
            raise WritingStage1CohortError(
                "cohort rows must be WritingStage1EpisodeRef values."
            )
        identities = tuple(item.episode_identity_hash for item in front + three)
        if len(identities) != len(set(identities)):
            raise WritingStage1CohortError(
                "The writing-stage1 cohort contains duplicate Episode identities."
            )
        if any(item.admitted_camera_slots != _FRONT_SLOTS for item in front):
            raise WritingStage1CohortError(
                "The 7.23 cohort must contain front-only logical rows."
            )
        if any(item.admitted_camera_slots != _THREE_VIEW_SLOTS for item in three):
            raise WritingStage1CohortError(
                "The 7.24 cohort must contain strict three-view rows."
            )
        excluded = _non_negative_int(
            self.excluded_three_view_other_provenance_count,
            name="excluded_three_view_other_provenance_count",
        )
        object.__setattr__(self, "front_0723", front)
        object.__setattr__(self, "three_view_0724", three)
        object.__setattr__(
            self,
            "excluded_three_view_other_provenance_count",
            excluded,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self._semantic_payload(),
                domain=WRITING_STAGE1_COHORT_PLAN_HASH_DOMAIN_V1,
            ),
        )

    @property
    def front_training_candidates(self) -> tuple[WritingStage1EpisodeRef, ...]:
        return self.front_0723 + self.three_view_0724

    @property
    def counts(self) -> dict[str, int]:
        return {
            "front_0723": len(self.front_0723),
            "three_view_0724": len(self.three_view_0724),
            "front_training_candidates": len(self.front_training_candidates),
            "three_view_training_candidates": len(self.three_view_0724),
            "excluded_three_view_other_provenance": (
                self.excluded_three_view_other_provenance_count
            ),
        }

    def _semantic_payload(self) -> dict[str, Any]:
        """Exclude relocatable local manifest paths from cohort identity."""

        return {
            "schema": self.schema,
            "front_manifest_sha256": self.front_manifest_sha256,
            "three_view_manifest_sha256": self.three_view_manifest_sha256,
            "three_view_required_provenance_root": (
                self.three_view_required_provenance_root
            ),
            "counts": self.counts,
            "front_0723": tuple(item.to_payload() for item in self.front_0723),
            "three_view_0724": tuple(
                item.to_payload() for item in self.three_view_0724
            ),
        }

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **self._semantic_payload(),
            "front_manifest_path": self.front_manifest_path,
            "three_view_manifest_path": self.three_view_manifest_path,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


def _read_manifest(path: str | Path) -> tuple[Path, bytes, tuple[Mapping[str, Any], ...]]:
    source = Path(path).expanduser().resolve()
    try:
        encoded = source.read_bytes()
    except OSError as exc:
        raise WritingStage1CohortError(
            f"Cannot read logical manifest {source}: {exc}"
        ) from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, raw in enumerate(encoded.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise WritingStage1CohortError(
                f"{source}:{line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(row, Mapping):
            raise WritingStage1CohortError(
                f"{source}:{line_number} must contain a JSON object."
            )
        rows.append(row)
    if not rows:
        raise WritingStage1CohortError(f"Logical manifest is empty: {source}")
    return source, encoded, tuple(rows)


def _episode_from_row(
    row: Mapping[str, Any],
    *,
    batch_id: str,
    expected_view_profile: str,
    expected_slots: tuple[str, ...],
) -> WritingStage1EpisodeRef:
    if row.get("schema") != _LOGICAL_EPISODE_REF_SCHEMA_V1:
        raise WritingStage1CohortError("Unsupported logical Episode schema.")
    if row.get("task") != "writing":
        raise WritingStage1CohortError("Cohort row is not a writing Episode.")
    if row.get("view_profile") != expected_view_profile:
        raise WritingStage1CohortError(
            "Cohort row has an unexpected logical view profile."
        )
    slots_payload = row.get("camera_slots")
    if not isinstance(slots_payload, list):
        raise WritingStage1CohortError("camera_slots must be a JSON list.")
    slots = tuple(_required(item, name="camera_slots") for item in slots_payload)
    if slots != expected_slots:
        raise WritingStage1CohortError(
            f"Unexpected camera_slots: expected={expected_slots} actual={slots}."
        )
    media = row.get("media_integrity")
    if not isinstance(media, Mapping) or media.get("status") != "valid":
        raise WritingStage1CohortError(
            "Strict cohort rows must have valid media integrity."
        )
    checks = media.get("checks")
    if not isinstance(checks, list) or not _REQUIRED_MEDIA_CHECKS.issubset(
        {str(value) for value in checks}
    ):
        raise WritingStage1CohortError(
            "Strict cohort rows require declared-count, decoder-count, and "
            "tail-decode media evidence."
        )
    camera_keys = media.get("camera_keys")
    expected_camera_keys = tuple(
        _MEDIA_CAMERA_KEY_BY_SLOT[value] for value in expected_slots
    )
    if not isinstance(camera_keys, list) or tuple(camera_keys) != expected_camera_keys:
        raise WritingStage1CohortError(
            "media_integrity.camera_keys differ from admitted camera_slots."
        )
    provenance = row.get("source_provenance")
    if not isinstance(provenance, Mapping):
        raise WritingStage1CohortError(
            "Strict cohort rows require explicit source provenance."
        )
    frame_count = _positive_int(row.get("frame_count"), name="frame_count")
    if provenance.get("frame_count") != frame_count:
        raise WritingStage1CohortError(
            "Logical and provenance frame counts differ."
        )
    return WritingStage1EpisodeRef(
        episode_identity_hash=_identity(row.get("episode_identity")),
        batch_id=batch_id,
        frame_count=frame_count,
        source_dataset_root=_required(
            row.get("source_dataset_root"),
            name="source_dataset_root",
        ),
        source_episode_index=_non_negative_int(
            row.get("source_episode_index"),
            name="source_episode_index",
        ),
        provenance_dataset_root=_required(
            provenance.get("source_dataset_root"),
            name="source_provenance.source_dataset_root",
        ),
        provenance_episode_index=_non_negative_int(
            provenance.get("source_episode_index"),
            name="source_provenance.source_episode_index",
        ),
        admitted_camera_slots=slots,
    )


def build_writing_stage1_cohort_plan(
    *,
    front_manifest_path: str | Path,
    three_view_manifest_path: str | Path,
    three_view_required_provenance_root: str,
    expected_front_count: int = 479,
    expected_three_view_manifest_count: int = 323,
    expected_three_view_0724_count: int = 240,
    expected_total_count: int = 719,
) -> WritingStage1CohortPlan:
    """Read and prove the strict cohort without scanning or rewriting raw data."""

    front_path, front_bytes, front_rows = _read_manifest(front_manifest_path)
    three_path, three_bytes, three_rows = _read_manifest(three_view_manifest_path)
    required_provenance = _required(
        three_view_required_provenance_root,
        name="three_view_required_provenance_root",
    )
    if len(front_rows) != expected_front_count:
        raise WritingStage1CohortError(
            f"Expected {expected_front_count} front rows; got {len(front_rows)}."
        )
    if len(three_rows) != expected_three_view_manifest_count:
        raise WritingStage1CohortError(
            "Unexpected three-view manifest count: "
            f"expected={expected_three_view_manifest_count} actual={len(three_rows)}."
        )
    front = tuple(
        _episode_from_row(
            row,
            batch_id="2026-07-23",
            expected_view_profile="front_only",
            expected_slots=_FRONT_SLOTS,
        )
        for row in front_rows
    )
    parsed_three = tuple(
        _episode_from_row(
            row,
            batch_id=(
                "2026-07-24"
                if isinstance(row.get("source_provenance"), Mapping)
                and row["source_provenance"].get("source_dataset_root")
                == required_provenance
                else "excluded-other-provenance"
            ),
            expected_view_profile="three_view_ready",
            expected_slots=_THREE_VIEW_SLOTS,
        )
        for row in three_rows
    )
    three = tuple(
        item
        for item in parsed_three
        if item.provenance_dataset_root == required_provenance
    )
    excluded_count = len(parsed_three) - len(three)
    if len(three) != expected_three_view_0724_count:
        raise WritingStage1CohortError(
            "Unexpected strict 7.24 count after provenance filtering: "
            f"expected={expected_three_view_0724_count} actual={len(three)}."
        )
    if len(front) + len(three) != expected_total_count:
        raise WritingStage1CohortError(
            "Unexpected strict writing-stage1 total: "
            f"expected={expected_total_count} actual={len(front) + len(three)}."
        )
    return WritingStage1CohortPlan(
        front_manifest_path=str(front_path),
        front_manifest_sha256=hashlib.sha256(front_bytes).hexdigest(),
        three_view_manifest_path=str(three_path),
        three_view_manifest_sha256=hashlib.sha256(three_bytes).hexdigest(),
        three_view_required_provenance_root=required_provenance,
        front_0723=front,
        three_view_0724=three,
        excluded_three_view_other_provenance_count=excluded_count,
    )


def select_writing_stage1_pilot(
    plan: WritingStage1CohortPlan,
    *,
    per_batch: int = 6,
) -> tuple[WritingStage1EpisodeRef, ...]:
    """Deterministically cover short-to-long Episodes in both source batches."""

    if not isinstance(plan, WritingStage1CohortPlan):
        raise TypeError("plan must be WritingStage1CohortPlan.")
    count = _positive_int(per_batch, name="per_batch")

    def select(rows: tuple[WritingStage1EpisodeRef, ...]) -> tuple[WritingStage1EpisodeRef, ...]:
        if len(rows) < count:
            raise WritingStage1CohortError(
                f"Cannot select {count} pilot rows from a {len(rows)}-row batch."
            )
        ordered = tuple(
            sorted(rows, key=lambda item: (item.frame_count, item.episode_identity_hash))
        )
        positions: tuple[int, ...]
        if count == 1:
            positions = (len(ordered) // 2,)
        else:
            positions = tuple(
                round(index * (len(ordered) - 1) / (count - 1))
                for index in range(count)
            )
        return tuple(ordered[position] for position in positions)

    return select(plan.front_0723) + select(plan.three_view_0724)


__all__ = [
    "WRITING_STAGE1_COHORT_PLAN_HASH_DOMAIN_V1",
    "WRITING_STAGE1_COHORT_PLAN_SCHEMA_V1",
    "WritingStage1CohortError",
    "WritingStage1CohortPlan",
    "WritingStage1EpisodeRef",
    "build_writing_stage1_cohort_plan",
    "select_writing_stage1_pilot",
]
