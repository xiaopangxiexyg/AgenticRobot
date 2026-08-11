"""Content-addressed persistence for frozen arm-hold Episode splits.

The artifact is model- and backend-neutral.  It freezes only source receipts,
Episode admission, exact train/validation membership, and the profile snapshot
that produced them.  Window materialization, RGB caches, learner settings, and
deployment behavior remain downstream concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from dataset_contracts.canonical_encoding import (
    canonical_bytes,
    canonical_content_hash,
    canonical_loads,
)

from .arm_hold_cohort import (
    ARM_HOLD_COHORT_DRY_RUN_SCHEMA_V1,
    ArmHoldCohortDryRun,
    ArmHoldEpisodeExclusion,
    ArmHoldEpisodeRecord,
    ArmHoldSourceReceipt,
    ArmHoldSplitResult,
)
from .arm_hold_profile import ArmHoldDatasetProfile


ARM_HOLD_SPLIT_MANIFEST_SCHEMA_V1 = (
    "dexteleop.arm_hold_split_manifest.provisional.v1"
)
ARM_HOLD_SPLIT_MANIFEST_HASH_DOMAIN_V1 = (
    "dexteleop.arm_hold_split_manifest.identity.provisional.v1"
)
ARM_HOLD_SPLIT_MANIFEST_STORAGE_DOMAIN_V1 = (
    "dexteleop.arm_hold_split_manifest.storage.provisional.v1"
)
ARM_HOLD_SPLIT_RELATIVE_ROOT = Path("derived/arm_hold_splits")
ARM_HOLD_SPLIT_MANIFEST_FILE = "manifest.cjson"

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ArmHoldSplitManifestError(ValueError):
    """Raised when a frozen split artifact is invalid or cannot be published."""


def _mapping(
    value: object,
    *,
    label: str,
    keys: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArmHoldSplitManifestError(f"{label} must be an object.")
    actual = {str(key) for key in value}
    if actual != keys:
        raise ArmHoldSplitManifestError(
            f"{label} fields mismatch: missing={sorted(keys - actual)!r} "
            f"unknown={sorted(actual - keys)!r}."
        )
    return value


def _sequence(value: object, *, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ArmHoldSplitManifestError(f"{label} must be a sequence.")
    return value


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArmHoldSplitManifestError(f"{label} must be a non-empty string.")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArmHoldSplitManifestError(
            f"{label} must be an integer >= {minimum}."
        )
    return value


def _event_counts(value: object, *, label: str) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    for index, raw in enumerate(_sequence(value, label=label)):
        item = _mapping(
            raw,
            label=f"{label}[{index}]",
            keys={"event_kind", "event_key", "count"},
        )
        identity = (
            _string(item["event_kind"], label=f"{label}[{index}].event_kind"),
            _string(item["event_key"], label=f"{label}[{index}].event_key"),
        )
        count = _integer(
            item["count"], label=f"{label}[{index}].count", minimum=1
        )
        if identity in result:
            raise ArmHoldSplitManifestError(
                f"{label} contains duplicate event {identity!r}."
            )
        result[identity] = count
    return result


def _episode(value: object, *, index: int) -> ArmHoldEpisodeRecord:
    item = _mapping(
        value,
        label=f"included_episodes[{index}]",
        keys={
            "source_dataset_id",
            "source_episode_index",
            "episode_uuid",
            "episode_capture_identity_hash",
            "frame_count",
            "annotation_set_id",
            "event_counts",
            "observed_pair_states",
            "transition_directions",
        },
    )
    try:
        return ArmHoldEpisodeRecord(
            source_dataset_id=_string(
                item["source_dataset_id"],
                label=f"included_episodes[{index}].source_dataset_id",
            ),
            source_episode_index=_integer(
                item["source_episode_index"],
                label=f"included_episodes[{index}].source_episode_index",
            ),
            episode_uuid=_string(
                item["episode_uuid"],
                label=f"included_episodes[{index}].episode_uuid",
            ),
            episode_capture_identity_hash=_string(
                item["episode_capture_identity_hash"],
                label=(
                    f"included_episodes[{index}].episode_capture_identity_hash"
                ),
            ),
            frame_count=_integer(
                item["frame_count"],
                label=f"included_episodes[{index}].frame_count",
                minimum=1,
            ),
            annotation_set_id=_string(
                item["annotation_set_id"],
                label=f"included_episodes[{index}].annotation_set_id",
            ),
            event_counts=_event_counts(
                item["event_counts"],
                label=f"included_episodes[{index}].event_counts",
            ),
            observed_pair_states=tuple(
                _string(
                    part,
                    label=f"included_episodes[{index}].observed_pair_states[]",
                )
                for part in _sequence(
                    item["observed_pair_states"],
                    label=f"included_episodes[{index}].observed_pair_states",
                )
            ),
            transition_directions=tuple(
                _string(
                    part,
                    label=f"included_episodes[{index}].transition_directions[]",
                )
                for part in _sequence(
                    item["transition_directions"],
                    label=f"included_episodes[{index}].transition_directions",
                )
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ArmHoldSplitManifestError(
            f"included_episodes[{index}] is invalid: {exc}"
        ) from exc


def _exclusion(value: object, *, index: int) -> ArmHoldEpisodeExclusion:
    item = _mapping(
        value,
        label=f"exclusions[{index}]",
        keys={
            "source_dataset_id",
            "source_episode_index",
            "episode_uuid",
            "episode_capture_identity_hash",
            "reason",
        },
    )
    try:
        return ArmHoldEpisodeExclusion(
            source_dataset_id=_string(
                item["source_dataset_id"],
                label=f"exclusions[{index}].source_dataset_id",
            ),
            source_episode_index=_integer(
                item["source_episode_index"],
                label=f"exclusions[{index}].source_episode_index",
            ),
            episode_uuid=_string(
                item["episode_uuid"], label=f"exclusions[{index}].episode_uuid"
            ),
            episode_capture_identity_hash=_string(
                item["episode_capture_identity_hash"],
                label=f"exclusions[{index}].episode_capture_identity_hash",
            ),
            reason=_string(
                item["reason"], label=f"exclusions[{index}].reason"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ArmHoldSplitManifestError(
            f"exclusions[{index}] is invalid: {exc}"
        ) from exc


def _receipt(value: object, *, index: int) -> ArmHoldSourceReceipt:
    item = _mapping(
        value,
        label=f"source_receipts[{index}]",
        keys={
            "source_dataset_id",
            "dataset_name",
            "source_dataset_fingerprint",
            "catalog_episode_count",
            "catalog_frame_count",
            "finalized_episode_count",
        },
    )
    try:
        return ArmHoldSourceReceipt(
            source_dataset_id=_string(
                item["source_dataset_id"],
                label=f"source_receipts[{index}].source_dataset_id",
            ),
            dataset_name=_string(
                item["dataset_name"],
                label=f"source_receipts[{index}].dataset_name",
            ),
            source_dataset_fingerprint=_string(
                item["source_dataset_fingerprint"],
                label=f"source_receipts[{index}].source_dataset_fingerprint",
            ),
            catalog_episode_count=_integer(
                item["catalog_episode_count"],
                label=f"source_receipts[{index}].catalog_episode_count",
            ),
            catalog_frame_count=_integer(
                item["catalog_frame_count"],
                label=f"source_receipts[{index}].catalog_frame_count",
            ),
            finalized_episode_count=_integer(
                item["finalized_episode_count"],
                label=f"source_receipts[{index}].finalized_episode_count",
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ArmHoldSplitManifestError(
            f"source_receipts[{index}] is invalid: {exc}"
        ) from exc


def _split(value: object) -> ArmHoldSplitResult:
    item = _mapping(
        value,
        label="split",
        keys={
            "train_episode_identities",
            "validation_episode_identities",
            "train_coverage",
            "validation_coverage",
        },
    )
    try:
        return ArmHoldSplitResult(
            train_episode_identities=tuple(
                _string(part, label="split.train_episode_identities[]")
                for part in _sequence(
                    item["train_episode_identities"],
                    label="split.train_episode_identities",
                )
            ),
            validation_episode_identities=tuple(
                _string(part, label="split.validation_episode_identities[]")
                for part in _sequence(
                    item["validation_episode_identities"],
                    label="split.validation_episode_identities",
                )
            ),
            train_coverage=_event_counts(
                item["train_coverage"], label="split.train_coverage"
            ),
            validation_coverage=_event_counts(
                item["validation_coverage"], label="split.validation_coverage"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ArmHoldSplitManifestError(f"split is invalid: {exc}") from exc


def _cohort(value: object) -> ArmHoldCohortDryRun:
    item = _mapping(
        value,
        label="cohort",
        keys={
            "schema",
            "profile_hash",
            "source_receipts",
            "included_episodes",
            "exclusions",
            "split",
        },
    )
    if item["schema"] != ARM_HOLD_COHORT_DRY_RUN_SCHEMA_V1:
        raise ArmHoldSplitManifestError("Unsupported cohort schema.")
    try:
        return ArmHoldCohortDryRun(
            profile_hash=_string(item["profile_hash"], label="cohort.profile_hash"),
            source_receipts=tuple(
                _receipt(raw, index=index)
                for index, raw in enumerate(
                    _sequence(item["source_receipts"], label="source_receipts")
                )
            ),
            included_episodes=tuple(
                _episode(raw, index=index)
                for index, raw in enumerate(
                    _sequence(item["included_episodes"], label="included_episodes")
                )
            ),
            exclusions=tuple(
                _exclusion(raw, index=index)
                for index, raw in enumerate(
                    _sequence(item["exclusions"], label="exclusions")
                )
            ),
            split=_split(item["split"]),
        )
    except ArmHoldSplitManifestError:
        raise
    except (TypeError, ValueError) as exc:
        raise ArmHoldSplitManifestError(f"cohort is invalid: {exc}") from exc


@dataclass(frozen=True)
class ArmHoldSplitManifest:
    profile: ArmHoldDatasetProfile
    cohort: ArmHoldCohortDryRun
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=ARM_HOLD_SPLIT_MANIFEST_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        if self.cohort.profile_hash != self.profile.semantic_content_hash:
            raise ArmHoldSplitManifestError(
                "Cohort profile hash does not match the embedded profile snapshot."
            )
        expected_sources = {
            source.source_dataset_id: source for source in self.profile.sources
        }
        receipts = {
            receipt.source_dataset_id: receipt
            for receipt in self.cohort.source_receipts
        }
        if set(receipts) != set(expected_sources):
            raise ArmHoldSplitManifestError(
                "Cohort receipts do not match the embedded profile sources."
            )
        for source_id, source in expected_sources.items():
            receipt = receipts[source_id]
            if (
                receipt.dataset_name != source.expected_dataset_name
                or receipt.catalog_episode_count != source.expected_episode_count
                or receipt.catalog_frame_count != source.expected_frame_count
            ):
                raise ArmHoldSplitManifestError(
                    f"Cohort receipt does not match profile source {source_id}."
                )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(), domain=ARM_HOLD_SPLIT_MANIFEST_HASH_DOMAIN_V1
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "profile": self.profile.to_dict(),
            "cohort": self.cohort.to_payload(),
        }

    @classmethod
    def from_payload(cls, value: object) -> "ArmHoldSplitManifest":
        payload = _mapping(
            value,
            label="manifest",
            keys={"schema", "profile", "cohort"},
        )
        if payload["schema"] != ARM_HOLD_SPLIT_MANIFEST_SCHEMA_V1:
            raise ArmHoldSplitManifestError("Unsupported split manifest schema.")
        if not isinstance(payload["profile"], Mapping):
            raise ArmHoldSplitManifestError("Manifest profile must be an object.")
        try:
            profile = ArmHoldDatasetProfile.from_dict(payload["profile"])
            cohort = _cohort(payload["cohort"])
            return cls(profile=profile, cohort=cohort)
        except ArmHoldSplitManifestError:
            raise
        except (TypeError, ValueError) as exc:
            raise ArmHoldSplitManifestError(f"Manifest is invalid: {exc}") from exc


@dataclass(frozen=True)
class ArmHoldSplitManifestBuildResult:
    manifest_root: Path
    manifest_hash: str
    idempotent: bool


def build_arm_hold_split_manifest(
    profile: ArmHoldDatasetProfile,
    cohort: ArmHoldCohortDryRun,
) -> ArmHoldSplitManifest:
    """Build a typed frozen artifact in memory without filesystem writes."""

    return ArmHoldSplitManifest(profile=profile, cohort=cohort)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o664)


def load_arm_hold_split_manifest(path: str | Path) -> ArmHoldSplitManifest:
    """Load and verify one content-addressed manifest directory."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir() or not _HASH_PATTERN.fullmatch(root.name):
        raise ArmHoldSplitManifestError(
            "Split manifest path must be a 64-hex content-addressed directory."
        )
    manifest_path = root / ARM_HOLD_SPLIT_MANIFEST_FILE
    try:
        payload = canonical_loads(
            manifest_path.read_bytes(),
            expected_domain=ARM_HOLD_SPLIT_MANIFEST_STORAGE_DOMAIN_V1,
        )
        manifest = ArmHoldSplitManifest.from_payload(payload)
    except ArmHoldSplitManifestError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ArmHoldSplitManifestError(
            f"Cannot load split manifest {manifest_path}: {exc}"
        ) from exc
    if manifest.semantic_content_hash != root.name:
        raise ArmHoldSplitManifestError(
            "Split manifest content hash does not match its directory name."
        )
    return manifest


def materialize_arm_hold_split_manifest(
    manifest: ArmHoldSplitManifest,
    organized_root: str | Path,
) -> ArmHoldSplitManifestBuildResult:
    """Atomically publish a single immutable manifest under its content hash."""

    root = Path(organized_root).expanduser().resolve()
    parent = root / ARM_HOLD_SPLIT_RELATIVE_ROOT
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / manifest.semantic_content_hash
    if final.exists():
        loaded = load_arm_hold_split_manifest(final)
        if loaded.semantic_content_hash != manifest.semantic_content_hash:
            raise ArmHoldSplitManifestError(
                "Existing manifest directory has conflicting content."
            )
        return ArmHoldSplitManifestBuildResult(
            manifest_root=final,
            manifest_hash=manifest.semantic_content_hash,
            idempotent=True,
        )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".building-{manifest.semantic_content_hash}-",
            dir=parent,
        )
    )
    staging.chmod(0o775)
    try:
        _write_exclusive(
            staging / ARM_HOLD_SPLIT_MANIFEST_FILE,
            canonical_bytes(
                manifest.to_payload(),
                domain=ARM_HOLD_SPLIT_MANIFEST_STORAGE_DOMAIN_V1,
            ),
        )
        _fsync_directory(staging)
        try:
            os.rename(staging, final)
        except OSError:
            if not final.exists():
                raise
            loaded = load_arm_hold_split_manifest(final)
            if loaded.semantic_content_hash != manifest.semantic_content_hash:
                raise ArmHoldSplitManifestError(
                    "Concurrent manifest publication produced conflicting content."
                )
            return ArmHoldSplitManifestBuildResult(
                manifest_root=final,
                manifest_hash=manifest.semantic_content_hash,
                idempotent=True,
            )
        _fsync_directory(parent)
        loaded = load_arm_hold_split_manifest(final)
        return ArmHoldSplitManifestBuildResult(
            manifest_root=final,
            manifest_hash=loaded.semantic_content_hash,
            idempotent=False,
        )
    except (OSError, TypeError, ValueError) as exc:
        if isinstance(exc, ArmHoldSplitManifestError):
            raise
        raise ArmHoldSplitManifestError(
            f"Cannot publish split manifest under {parent}: {exc}"
        ) from exc
    finally:
        if staging.exists() and staging.name.startswith(".building-"):
            shutil.rmtree(staging)


__all__ = [
    "ARM_HOLD_SPLIT_MANIFEST_FILE",
    "ARM_HOLD_SPLIT_MANIFEST_HASH_DOMAIN_V1",
    "ARM_HOLD_SPLIT_MANIFEST_SCHEMA_V1",
    "ARM_HOLD_SPLIT_MANIFEST_STORAGE_DOMAIN_V1",
    "ARM_HOLD_SPLIT_RELATIVE_ROOT",
    "ArmHoldSplitManifest",
    "ArmHoldSplitManifestBuildResult",
    "ArmHoldSplitManifestError",
    "build_arm_hold_split_manifest",
    "load_arm_hold_split_manifest",
    "materialize_arm_hold_split_manifest",
]
