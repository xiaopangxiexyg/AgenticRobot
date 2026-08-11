"""Read-only cohort and deterministic split dry-runs for arm-hold profiles.

This provisional layer discovers exact eligible Episodes and computes split
membership in memory.  It deliberately does not hash bulk media, persist a
manifest, materialize windows, or configure a model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash

from .arm_hold import project_arm_hold_training_view
from .arm_hold_profile import (
    ArmHoldDatasetProfile,
    resolve_arm_hold_source_roots,
)
from .episode_curation import source_dataset_base_fingerprint
from .prepared_training.sources import _select_annotation
from .progress_annotation_jobs import (
    ProgressAnnotationWorkItem,
    list_progress_annotation_work_items,
)


ARM_HOLD_COHORT_DRY_RUN_SCHEMA_V1 = (
    "dexteleop.arm_hold_cohort_dry_run.provisional.v1"
)
ARM_HOLD_COHORT_DRY_RUN_HASH_DOMAIN_V1 = (
    "dexteleop.arm_hold_cohort_dry_run.semantic.provisional.v1"
)


class ArmHoldCohortError(ValueError):
    """Raised when source facts cannot produce one unambiguous cohort."""


EventIdentity = tuple[str, str]


def _freeze_counts(value: Mapping[EventIdentity, int]) -> Mapping[EventIdentity, int]:
    return MappingProxyType(
        {
            (str(kind), str(key)): int(count)
            for (kind, key), count in sorted(value.items())
            if int(count) > 0
        }
    )


def _event_payload(value: Mapping[EventIdentity, int]) -> list[dict[str, Any]]:
    return [
        {"event_kind": kind, "event_key": key, "count": count}
        for (kind, key), count in sorted(value.items())
    ]


@dataclass(frozen=True)
class ArmHoldEpisodeRecord:
    source_dataset_id: str
    source_episode_index: int
    episode_uuid: str
    episode_capture_identity_hash: str
    frame_count: int
    annotation_set_id: str
    event_counts: Mapping[EventIdentity, int]
    observed_pair_states: tuple[str, ...]
    transition_directions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.source_episode_index < 0 or self.frame_count <= 0:
            raise ArmHoldCohortError("Episode index/count values are invalid.")
        for name in (
            "source_dataset_id",
            "episode_uuid",
            "episode_capture_identity_hash",
            "annotation_set_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ArmHoldCohortError(f"{name} must be non-empty.")
        object.__setattr__(self, "event_counts", _freeze_counts(self.event_counts))
        object.__setattr__(
            self,
            "observed_pair_states",
            tuple(sorted({str(value) for value in self.observed_pair_states})),
        )
        object.__setattr__(
            self,
            "transition_directions",
            tuple(sorted({str(value) for value in self.transition_directions})),
        )

    @property
    def stratum(self) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        return (
            self.source_dataset_id,
            self.observed_pair_states,
            self.transition_directions,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_dataset_id": self.source_dataset_id,
            "source_episode_index": self.source_episode_index,
            "episode_uuid": self.episode_uuid,
            "episode_capture_identity_hash": self.episode_capture_identity_hash,
            "frame_count": self.frame_count,
            "annotation_set_id": self.annotation_set_id,
            "event_counts": _event_payload(self.event_counts),
            "observed_pair_states": list(self.observed_pair_states),
            "transition_directions": list(self.transition_directions),
        }


@dataclass(frozen=True)
class ArmHoldEpisodeExclusion:
    source_dataset_id: str
    source_episode_index: int
    episode_uuid: str
    episode_capture_identity_hash: str
    reason: str

    def __post_init__(self) -> None:
        if self.source_episode_index < 0:
            raise ArmHoldCohortError("Excluded Episode index is invalid.")
        for name in (
            "source_dataset_id",
            "episode_uuid",
            "episode_capture_identity_hash",
            "reason",
        ):
            if not str(getattr(self, name)).strip():
                raise ArmHoldCohortError(f"{name} must be non-empty.")

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_dataset_id": self.source_dataset_id,
            "source_episode_index": self.source_episode_index,
            "episode_uuid": self.episode_uuid,
            "episode_capture_identity_hash": self.episode_capture_identity_hash,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ArmHoldSourceReceipt:
    source_dataset_id: str
    dataset_name: str
    source_dataset_fingerprint: str
    catalog_episode_count: int
    catalog_frame_count: int
    finalized_episode_count: int

    def __post_init__(self) -> None:
        for name in (
            "source_dataset_id",
            "dataset_name",
            "source_dataset_fingerprint",
        ):
            if not str(getattr(self, name)).strip():
                raise ArmHoldCohortError(f"{name} must be non-empty.")
        if (
            self.catalog_episode_count < 0
            or self.catalog_frame_count < 0
            or self.finalized_episode_count < 0
            or self.finalized_episode_count > self.catalog_episode_count
        ):
            raise ArmHoldCohortError("Source receipt counts are invalid.")

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_dataset_id": self.source_dataset_id,
            "dataset_name": self.dataset_name,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "catalog_episode_count": self.catalog_episode_count,
            "catalog_frame_count": self.catalog_frame_count,
            "finalized_episode_count": self.finalized_episode_count,
        }


@dataclass(frozen=True)
class ArmHoldSplitResult:
    train_episode_identities: tuple[str, ...]
    validation_episode_identities: tuple[str, ...]
    train_coverage: Mapping[EventIdentity, int]
    validation_coverage: Mapping[EventIdentity, int]

    def __post_init__(self) -> None:
        train = tuple(sorted(str(value) for value in self.train_episode_identities))
        validation = tuple(
            sorted(str(value) for value in self.validation_episode_identities)
        )
        if (
            not train
            or not validation
            or len(train) != len(set(train))
            or len(validation) != len(set(validation))
            or any(not value.strip() for value in train + validation)
        ):
            raise ArmHoldCohortError(
                "Train/validation Episode membership is empty or duplicated."
            )
        if set(train) & set(validation):
            raise ArmHoldCohortError("Train/validation Episode leakage detected.")
        object.__setattr__(self, "train_episode_identities", train)
        object.__setattr__(self, "validation_episode_identities", validation)
        object.__setattr__(self, "train_coverage", _freeze_counts(self.train_coverage))
        object.__setattr__(
            self,
            "validation_coverage",
            _freeze_counts(self.validation_coverage),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "train_episode_identities": list(self.train_episode_identities),
            "validation_episode_identities": list(
                self.validation_episode_identities
            ),
            "train_coverage": _event_payload(self.train_coverage),
            "validation_coverage": _event_payload(self.validation_coverage),
        }


@dataclass(frozen=True)
class ArmHoldCohortDryRun:
    profile_hash: str
    source_receipts: tuple[ArmHoldSourceReceipt, ...]
    included_episodes: tuple[ArmHoldEpisodeRecord, ...]
    exclusions: tuple[ArmHoldEpisodeExclusion, ...]
    split: ArmHoldSplitResult
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=ARM_HOLD_COHORT_DRY_RUN_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        included = tuple(
            sorted(
                self.included_episodes,
                key=lambda item: (
                    item.source_dataset_id,
                    item.source_episode_index,
                    item.episode_capture_identity_hash,
                ),
            )
        )
        exclusions = tuple(
            sorted(
                self.exclusions,
                key=lambda item: (
                    item.source_dataset_id,
                    item.source_episode_index,
                    item.episode_capture_identity_hash,
                ),
            )
        )
        receipts = tuple(
            sorted(self.source_receipts, key=lambda item: item.source_dataset_id)
        )
        identities = tuple(item.episode_capture_identity_hash for item in included)
        if len(identities) != len(set(identities)):
            raise ArmHoldCohortError("Included cohort has duplicate Episode identities.")
        excluded_identities = tuple(
            item.episode_capture_identity_hash for item in exclusions
        )
        if len(excluded_identities) != len(set(excluded_identities)):
            raise ArmHoldCohortError("Excluded cohort has duplicate Episode identities.")
        if set(identities) & set(excluded_identities):
            raise ArmHoldCohortError("Included/excluded Episode identities overlap.")
        receipt_ids = tuple(item.source_dataset_id for item in receipts)
        if len(receipt_ids) != len(set(receipt_ids)):
            raise ArmHoldCohortError("Source receipts contain duplicate source IDs.")
        observed_source_ids = {
            item.source_dataset_id for item in included + exclusions
        }
        if observed_source_ids != set(receipt_ids):
            raise ArmHoldCohortError(
                "Source receipts do not exactly cover cohort source IDs."
            )
        episode_counts = Counter(
            item.source_dataset_id for item in included + exclusions
        )
        if any(
            episode_counts[receipt.source_dataset_id]
            != receipt.catalog_episode_count
            for receipt in receipts
        ):
            raise ArmHoldCohortError(
                "Cohort admission does not account for every catalog Episode."
            )
        if set(identities) != (
            set(self.split.train_episode_identities)
            | set(self.split.validation_episode_identities)
        ):
            raise ArmHoldCohortError("Split membership does not cover the cohort exactly.")
        by_identity = {
            item.episode_capture_identity_hash: item for item in included
        }
        expected_train_coverage = _coverage(
            [by_identity[value] for value in self.split.train_episode_identities]
        )
        expected_validation_coverage = _coverage(
            [by_identity[value] for value in self.split.validation_episode_identities]
        )
        if dict(self.split.train_coverage) != dict(expected_train_coverage) or dict(
            self.split.validation_coverage
        ) != dict(expected_validation_coverage):
            raise ArmHoldCohortError(
                "Split coverage does not match exact Episode membership."
            )
        object.__setattr__(self, "included_episodes", included)
        object.__setattr__(self, "exclusions", exclusions)
        object.__setattr__(self, "source_receipts", receipts)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self._semantic_payload(),
                domain=ARM_HOLD_COHORT_DRY_RUN_HASH_DOMAIN_V1,
            ),
        )

    def _semantic_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "profile_hash": self.profile_hash,
            "source_receipts": [item.to_payload() for item in self.source_receipts],
            "included_episodes": [
                item.to_payload() for item in self.included_episodes
            ],
            "exclusions": [item.to_payload() for item in self.exclusions],
            "split": self.split.to_payload(),
        }

    def to_payload(self) -> dict[str, Any]:
        """Return the complete canonical payload used by frozen manifests."""

        return self._semantic_payload()

    def summary_payload(self, *, debug: bool = False) -> dict[str, Any]:
        reason_counts = Counter(item.reason for item in self.exclusions)
        payload: dict[str, Any] = {
            "schema": self.schema,
            "profile_hash": self.profile_hash,
            "dry_run_hash": self.semantic_content_hash,
            "source_receipts": [item.to_payload() for item in self.source_receipts],
            "included_episode_count": len(self.included_episodes),
            "excluded_episode_count": len(self.exclusions),
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
            "train_episode_count": len(self.split.train_episode_identities),
            "validation_episode_count": len(
                self.split.validation_episode_identities
            ),
            "train_coverage": _event_payload(self.split.train_coverage),
            "validation_coverage": _event_payload(
                self.split.validation_coverage
            ),
        }
        if debug:
            payload["included_episodes"] = [
                item.to_payload() for item in self.included_episodes
            ]
            payload["exclusions"] = [item.to_payload() for item in self.exclusions]
            payload["split"] = self.split.to_payload()
        return payload


def _constraints(split_spec: Mapping[str, Any]) -> dict[EventIdentity, tuple[int, int]]:
    return {
        (str(item["event_kind"]), str(item["event_key"])): (
            int(item["minimum_train_events"]),
            int(item["minimum_validation_events"]),
        )
        for item in split_spec["coverage_constraints"]
    }


def _coverage(episodes: Sequence[ArmHoldEpisodeRecord]) -> Counter[EventIdentity]:
    total: Counter[EventIdentity] = Counter()
    for episode in episodes:
        total.update(episode.event_counts)
    return total


def _score(seed_domain: str, identity: str) -> str:
    return hashlib.sha256(f"{seed_domain}:{identity}".encode("utf-8")).hexdigest()


def split_arm_hold_episodes(
    episodes: Sequence[ArmHoldEpisodeRecord],
    split_spec: Mapping[str, Any],
) -> ArmHoldSplitResult:
    """Build a deterministic, per-source validation split with coverage gates."""

    values = tuple(
        sorted(episodes, key=lambda item: item.episode_capture_identity_hash)
    )
    if not values:
        raise ArmHoldCohortError("Cannot split an empty cohort.")
    identities = tuple(item.episode_capture_identity_hash for item in values)
    if len(identities) != len(set(identities)):
        raise ArmHoldCohortError("Cohort contains duplicate Episode identities.")
    constraints = _constraints(split_spec)
    total_coverage = _coverage(values)
    for event, (minimum_train, minimum_validation) in constraints.items():
        if total_coverage[event] < minimum_train + minimum_validation:
            raise ArmHoldCohortError(
                f"Coverage gate is impossible for {event}: "
                f"available={total_coverage[event]} "
                f"required={minimum_train + minimum_validation}."
            )

    by_source: dict[str, list[ArmHoldEpisodeRecord]] = {}
    by_stratum: Counter[tuple[str, tuple[str, ...], tuple[str, ...]]] = Counter()
    for episode in values:
        by_source.setdefault(episode.source_dataset_id, []).append(episode)
        by_stratum[episode.stratum] += 1
    validation_fraction = float(split_spec["validation_fraction"])
    source_quota = {
        source: min(
            len(source_episodes) - 1,
            max(1, int(math.floor(len(source_episodes) * validation_fraction + 0.5))),
        )
        for source, source_episodes in by_source.items()
    }
    if any(value <= 0 for value in source_quota.values()):
        raise ArmHoldCohortError(
            "Every source requires at least two eligible Episodes for splitting."
        )
    stratum_target = {
        stratum: int(math.floor(count * validation_fraction + 0.5))
        for stratum, count in by_stratum.items()
    }
    seed_domain = str(split_spec["seed_domain"])
    selected: set[str] = set()
    validation_coverage: Counter[EventIdentity] = Counter()
    selected_by_source: Counter[str] = Counter()
    selected_by_stratum: Counter[
        tuple[str, tuple[str, ...], tuple[str, ...]]
    ] = Counter()

    def can_select(episode: ArmHoldEpisodeRecord) -> bool:
        if selected_by_source[episode.source_dataset_id] >= source_quota[
            episode.source_dataset_id
        ]:
            return False
        for event, (minimum_train, _) in constraints.items():
            remaining = (
                total_coverage[event]
                - validation_coverage[event]
                - episode.event_counts.get(event, 0)
            )
            if remaining < minimum_train:
                return False
        return True

    def select(episode: ArmHoldEpisodeRecord) -> None:
        selected.add(episode.episode_capture_identity_hash)
        validation_coverage.update(episode.event_counts)
        selected_by_source[episode.source_dataset_id] += 1
        selected_by_stratum[episode.stratum] += 1

    ordered_gates = sorted(
        constraints,
        key=lambda event: (total_coverage[event], event),
    )
    for event in ordered_gates:
        minimum_validation = constraints[event][1]
        while validation_coverage[event] < minimum_validation:
            candidates = [
                episode
                for episode in values
                if episode.episode_capture_identity_hash not in selected
                and episode.event_counts.get(event, 0) > 0
                and can_select(episode)
            ]
            if not candidates:
                raise ArmHoldCohortError(
                    f"Cannot satisfy validation coverage gate for {event}."
                )
            candidates.sort(
                key=lambda episode: (
                    -episode.event_counts.get(event, 0),
                    _score(seed_domain, episode.episode_capture_identity_hash),
                )
            )
            select(candidates[0])

    for source in sorted(by_source):
        while selected_by_source[source] < source_quota[source]:
            candidates = [
                episode
                for episode in by_source[source]
                if episode.episode_capture_identity_hash not in selected
                and can_select(episode)
            ]
            if not candidates:
                raise ArmHoldCohortError(
                    f"Cannot fill validation quota for source {source!r}."
                )
            candidates.sort(
                key=lambda episode: (
                    selected_by_stratum[episode.stratum]
                    >= stratum_target[episode.stratum],
                    _score(seed_domain, episode.episode_capture_identity_hash),
                )
            )
            select(candidates[0])

    validation = tuple(
        episode
        for episode in values
        if episode.episode_capture_identity_hash in selected
    )
    train = tuple(
        episode
        for episode in values
        if episode.episode_capture_identity_hash not in selected
    )
    train_coverage = _coverage(train)
    validation_coverage = _coverage(validation)
    for event, (minimum_train, minimum_validation) in constraints.items():
        if train_coverage[event] < minimum_train:
            raise ArmHoldCohortError(f"Train coverage gate failed for {event}.")
        if validation_coverage[event] < minimum_validation:
            raise ArmHoldCohortError(
                f"Validation coverage gate failed for {event}."
            )
    return ArmHoldSplitResult(
        train_episode_identities=tuple(
            item.episode_capture_identity_hash for item in train
        ),
        validation_episode_identities=tuple(
            item.episode_capture_identity_hash for item in validation
        ),
        train_coverage=train_coverage,
        validation_coverage=validation_coverage,
    )


def _catalog_summary(root: Path) -> tuple[int, int]:
    path = root / "meta" / "episodes.jsonl"
    count = 0
    frames = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ArmHoldCohortError(
                        f"Catalog row {path}:{line_number} is not an object."
                    )
                count += 1
                frames += int(value["frame_count"])
    except ArmHoldCohortError:
        raise
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ArmHoldCohortError(f"Cannot read source catalog {path}: {exc}") from exc
    return count, frames


def _load_info(root: Path) -> Mapping[str, Any]:
    path = root / "meta" / "info.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArmHoldCohortError(f"Cannot read dataset info {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ArmHoldCohortError(f"Dataset info is not an object: {path}.")
    return value


def _validate_source_contracts(
    *,
    profile: ArmHoldDatasetProfile,
    source: Any,
    info: Mapping[str, Any],
    items: Sequence[ProgressAnnotationWorkItem],
    catalog_episode_count: int,
    catalog_frame_count: int,
) -> None:
    required = profile.required_contracts
    observed = {
        "dataset_schema": info.get("dexteleop_dataset_schema"),
        "layout_hash": info.get("policy_io_layout_hash"),
    }
    for key, value in observed.items():
        if value != required[key]:
            raise ArmHoldCohortError(
                f"Source {source.source_dataset_id} {key} mismatch: "
                f"expected={required[key]!r} actual={value!r}."
            )
    if info.get("dataset_name") != source.expected_dataset_name:
        raise ArmHoldCohortError(
            f"Source {source.source_dataset_id} dataset_name mismatch."
        )
    if catalog_episode_count != source.expected_episode_count:
        raise ArmHoldCohortError(
            f"Source {source.source_dataset_id} Episode count drift: "
            f"expected={source.expected_episode_count} actual={catalog_episode_count}."
        )
    if catalog_frame_count != source.expected_frame_count:
        raise ArmHoldCohortError(
            f"Source {source.source_dataset_id} frame count drift: "
            f"expected={source.expected_frame_count} actual={catalog_frame_count}."
        )
    if len(items) != catalog_episode_count:
        raise ArmHoldCohortError(
            f"Source {source.source_dataset_id} contains non-finalized Episodes: "
            f"catalog={catalog_episode_count} finalized={len(items)}."
        )
    for item in items:
        if item.progress.profile.semantic_content_hash != required[
            "progress_profile_hash"
        ]:
            raise ArmHoldCohortError(
                f"Episode {item.episode_identity_hash} progress profile mismatch."
            )
        if item.progress.stage_profile is None or (
            item.progress.stage_profile.semantic_content_hash
            != required["stage_profile_hash"]
        ):
            raise ArmHoldCohortError(
                f"Episode {item.episode_identity_hash} stage profile mismatch."
            )


def _episode_events(
    item: ProgressAnnotationWorkItem,
    *,
    info: Mapping[str, Any],
    required_columns: Sequence[str],
    running_session_state: str,
) -> tuple[Mapping[EventIdentity, int], tuple[str, ...], tuple[str, ...]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ArmHoldCohortError("pyarrow is required for cohort discovery.") from exc
    try:
        parquet = pq.ParquetFile(item.parquet_path)
        schema_names = set(parquet.schema_arrow.names)
        missing = sorted(set(required_columns) - schema_names)
        if missing:
            raise ArmHoldCohortError(
                f"Episode {item.episode_identity_hash} is missing columns {missing!r}."
            )
        columns_to_read = (
            "frame_index",
            "session_state",
            "collection.continuity_segment_index",
            "teleop.arm_mode.schema",
            "teleop.arm_mode.available",
            "teleop.arm_mode.left.id",
            "teleop.arm_mode.left.label",
            "teleop.arm_mode.left.scope",
            "teleop.arm_mode.right.id",
            "teleop.arm_mode.right.label",
            "teleop.arm_mode.right.scope",
        )
        table = pq.read_table(item.parquet_path, columns=list(columns_to_read))
    except ArmHoldCohortError:
        raise
    except Exception as exc:
        raise ArmHoldCohortError(
            f"Cannot inspect Episode Parquet {item.parquet_path}: {exc}"
        ) from exc
    if table.num_rows != item.frame_count:
        raise ArmHoldCohortError(
            f"Episode {item.episode_identity_hash} frame count mismatch."
        )
    columns = {name: table[name].to_pylist() for name in columns_to_read}
    frame_indices = np.asarray(columns["frame_index"], dtype=np.int64)
    if not np.array_equal(frame_indices, np.arange(item.frame_count, dtype=np.int64)):
        raise ArmHoldCohortError(
            f"Episode {item.episode_identity_hash} frame_index is not contiguous."
        )
    running_indices = np.asarray(
        [
            index
            for index, state in enumerate(columns["session_state"])
            if str(state) == running_session_state
        ],
        dtype=np.int64,
    )
    view = project_arm_hold_training_view(
        columns,
        metadata=info,
        source_indices=running_indices,
        strict=True,
    )
    counts: Counter[EventIdentity] = Counter()
    for side_index, side in enumerate(("left", "right")):
        valid = view.valid_mask[:, side_index]
        positive = valid & (view.targets[:, side_index] == 1.0)
        negative = valid & (view.targets[:, side_index] == 0.0)
        counts[("label", f"{side}:fixed_hold")] = int(np.sum(positive))
        counts[("label", f"{side}:teleop")] = int(np.sum(negative))
        changed = view.transition_mask[:, side_index]
        counts[("transition", f"{side}:teleop->fixed_hold")] = int(
            np.sum(changed & positive)
        )
        counts[("transition", f"{side}:fixed_hold->teleop")] = int(
            np.sum(changed & negative)
        )
    both_valid = np.all(view.valid_mask, axis=1)
    for pair in view.targets[both_valid]:
        counts[("pair_state", f"{int(pair[0])}{int(pair[1])}")] += 1
    pairs = tuple(
        sorted(key for (kind, key), count in counts.items() if kind == "pair_state" and count)
    )
    transitions = tuple(
        sorted(key for (kind, key), count in counts.items() if kind == "transition" and count)
    )
    return _freeze_counts(counts), pairs, transitions


def build_arm_hold_cohort_dry_run(
    profile: ArmHoldDatasetProfile,
    workspace_root: str | Path,
) -> ArmHoldCohortDryRun:
    """Read source facts and compute exact admission/split without any writes."""

    roots = resolve_arm_hold_source_roots(profile, workspace_root)
    selectors = {
        str(key): str(value)
        for key, value in profile.episode_admission[
            "annotation_selection"
        ].items()
    }
    used_selectors: set[str] = set()
    receipts: list[ArmHoldSourceReceipt] = []
    included: list[ArmHoldEpisodeRecord] = []
    exclusions: list[ArmHoldEpisodeExclusion] = []
    seen_identities: set[str] = set()

    for source in profile.sources:
        root = roots[source.source_dataset_id]
        info = _load_info(root)
        catalog_episode_count, catalog_frame_count = _catalog_summary(root)
        items = list_progress_annotation_work_items(root)
        _validate_source_contracts(
            profile=profile,
            source=source,
            info=info,
            items=items,
            catalog_episode_count=catalog_episode_count,
            catalog_frame_count=catalog_frame_count,
        )
        receipts.append(
            ArmHoldSourceReceipt(
                source_dataset_id=source.source_dataset_id,
                dataset_name=source.expected_dataset_name,
                source_dataset_fingerprint=source_dataset_base_fingerprint(root),
                catalog_episode_count=catalog_episode_count,
                catalog_frame_count=catalog_frame_count,
                finalized_episode_count=len(items),
            )
        )
        for item in items:
            identity = item.episode_identity_hash
            if identity in seen_identities:
                raise ArmHoldCohortError(
                    f"Sources contain duplicate Episode identity {identity}."
                )
            seen_identities.add(identity)
            reason: str | None = None
            annotation = None
            if item.progress.binding.recording_scope != "FULL_TASK":
                reason = "RECORDING_SCOPE_NOT_FULL_TASK"
            elif item.progress.timeline.full_task_outcome != "SUCCESS":
                reason = "TERMINAL_OUTCOME_NOT_SUCCESS"
            else:
                selector_key = (
                    identity
                    if identity in selectors
                    else item.episode_uuid
                    if item.episode_uuid in selectors
                    else None
                )
                if selector_key is not None:
                    used_selectors.add(selector_key)
                try:
                    annotation = _select_annotation(
                        item,
                        None if selector_key is None else selectors[selector_key],
                    )
                except ValueError as exc:
                    raise ArmHoldCohortError(str(exc)) from exc
                if annotation is None:
                    reason = "NO_SEALED_STAGE_ANNOTATION"
                elif annotation.bundle.episode_abnormal is not None:
                    reason = "WHOLE_EPISODE_ABNORMAL"
                elif annotation.bundle.episode_review_request is not None:
                    reason = "WHOLE_EPISODE_REVIEW_PENDING"
                elif annotation.bundle.abnormal_intervals:
                    reason = "LOCAL_ABNORMAL_INTERVAL"

            if reason is not None:
                exclusions.append(
                    ArmHoldEpisodeExclusion(
                        source_dataset_id=source.source_dataset_id,
                        source_episode_index=item.episode_index,
                        episode_uuid=item.episode_uuid,
                        episode_capture_identity_hash=identity,
                        reason=reason,
                    )
                )
                continue
            assert annotation is not None
            counts, pairs, transitions = _episode_events(
                item,
                info=info,
                required_columns=profile.episode_admission["required_columns"],
                running_session_state=profile.sample_view[
                    "running_session_state"
                ],
            )
            included.append(
                ArmHoldEpisodeRecord(
                    source_dataset_id=source.source_dataset_id,
                    source_episode_index=item.episode_index,
                    episode_uuid=item.episode_uuid,
                    episode_capture_identity_hash=identity,
                    frame_count=item.frame_count,
                    annotation_set_id=annotation.binding.annotation_set_id,
                    event_counts=counts,
                    observed_pair_states=pairs,
                    transition_directions=transitions,
                )
            )

    unused_selectors = sorted(set(selectors) - used_selectors)
    if unused_selectors:
        raise ArmHoldCohortError(
            "annotation_selection contains selectors not used by successful "
            f"source Episodes: {unused_selectors!r}."
        )
    split = split_arm_hold_episodes(included, profile.split)
    return ArmHoldCohortDryRun(
        profile_hash=profile.semantic_content_hash,
        source_receipts=tuple(receipts),
        included_episodes=tuple(included),
        exclusions=tuple(exclusions),
        split=split,
    )


__all__ = [
    "ARM_HOLD_COHORT_DRY_RUN_HASH_DOMAIN_V1",
    "ARM_HOLD_COHORT_DRY_RUN_SCHEMA_V1",
    "ArmHoldCohortDryRun",
    "ArmHoldCohortError",
    "ArmHoldEpisodeExclusion",
    "ArmHoldEpisodeRecord",
    "ArmHoldSourceReceipt",
    "ArmHoldSplitResult",
    "build_arm_hold_cohort_dry_run",
    "split_arm_hold_episodes",
]
