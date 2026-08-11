"""Compatibility-first read resolution for relocated runtime artifacts.

This module resolves already-selected artifact candidates.  Callers that use a
"latest" convention must select the latest candidate independently inside each
root before calling this resolver; candidates from different roots are never
compared by mtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Literal, Sequence
import warnings


ArtifactReadSource = Literal["explicit", "canonical", "legacy"]
ArtifactValidator = Callable[[Path], bool]
MIGRATION_ARCHIVE_SCHEMA = "dexteleop.runtime_artifact_archive.v1"
MIGRATION_COPY_PLAN_SCHEMA = "dexteleop.runtime_artifact_copy_plan.v1"
MIGRATION_COPY_JOURNAL_SCHEMA = "dexteleop.runtime_artifact_copy_journal.v1"


class LegacyArtifactFallbackWarning(UserWarning):
    """A canonical read was unavailable and a legacy artifact was selected."""


class ArtifactReadValidationError(ValueError):
    """An existing artifact candidate failed its format-specific validation."""


class RelocatedArtifactWarning(UserWarning):
    """A missing historical locator was resolved through an archive manifest."""


@dataclass(frozen=True)
class ArtifactReadResolution:
    path: Path
    source: ArtifactReadSource
    attempted_paths: tuple[Path, ...]


@dataclass(frozen=True)
class ArtifactRelocation:
    source_root: Path
    archive_root: Path
    manifest_path: Path
    migration_id: str
    source_snapshot: dict[str, object] | None = None


@dataclass(frozen=True)
class HistoricalArtifactResolution:
    requested_path: Path
    path: Path
    source: Literal["original", "relocated"]
    relocation_chain: tuple[ArtifactRelocation, ...]


def _resolved(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def _validate_candidate(
    path: Path,
    *,
    source: ArtifactReadSource,
    validate: ArtifactValidator | None,
) -> None:
    if validate is None:
        return
    try:
        valid = bool(validate(path))
    except Exception as exc:
        raise ArtifactReadValidationError(
            f"{source} artifact validation raised for {path}: {exc}"
        ) from exc
    if not valid:
        raise ArtifactReadValidationError(
            f"{source} artifact validation failed for {path}"
        )


def resolve_artifact_read_path(
    *,
    explicit_path: str | os.PathLike[str] | None = None,
    canonical_path: str | os.PathLike[str] | None = None,
    legacy_paths: Sequence[str | os.PathLike[str] | None] = (),
    validate: ArtifactValidator | None = None,
    warn_on_legacy: bool = True,
) -> ArtifactReadResolution:
    """Resolve an existing artifact with explicit/canonical/legacy precedence.

    An explicit path is exact: if it is missing or invalid, no fallback is
    attempted.  An existing canonical candidate is also fail-closed on invalid
    content.  Legacy candidates are considered only when no canonical candidate
    exists, in the order supplied by the caller.
    """

    if explicit_path is not None:
        explicit = _resolved(explicit_path)
        if not explicit.exists():
            raise FileNotFoundError(f"Explicit artifact path does not exist: {explicit}")
        _validate_candidate(explicit, source="explicit", validate=validate)
        return ArtifactReadResolution(
            path=explicit,
            source="explicit",
            attempted_paths=(explicit,),
        )

    attempted: list[Path] = []
    if canonical_path is not None:
        canonical = _resolved(canonical_path)
        attempted.append(canonical)
        if canonical.exists():
            _validate_candidate(canonical, source="canonical", validate=validate)
            return ArtifactReadResolution(
                path=canonical,
                source="canonical",
                attempted_paths=tuple(attempted),
            )

    for value in legacy_paths:
        if value is None:
            continue
        legacy = _resolved(value)
        attempted.append(legacy)
        if not legacy.exists():
            continue
        _validate_candidate(legacy, source="legacy", validate=validate)
        if warn_on_legacy:
            warnings.warn(
                f"Using legacy runtime artifact because the canonical candidate "
                f"is unavailable: {legacy}",
                LegacyArtifactFallbackWarning,
                stacklevel=2,
            )
        return ArtifactReadResolution(
            path=legacy,
            source="legacy",
            attempted_paths=tuple(attempted),
        )

    searched = ", ".join(str(path) for path in attempted) or "<no candidates>"
    raise FileNotFoundError(f"No runtime artifact candidate exists; searched: {searched}")


def _manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _copy_plan_hash(payload: dict[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("plan_hash", None)
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def _load_copy_plan_relocation(
    plan_path: Path,
    payload: dict[str, object],
) -> ArtifactRelocation:
    expected_plan_hash = str(payload.get("plan_hash") or "")
    actual_plan_hash = _copy_plan_hash(payload)
    if not expected_plan_hash or expected_plan_hash != actual_plan_hash:
        raise ArtifactReadValidationError(
            f"copy plan hash failed: {plan_path}"
        )
    recorded_plan_dir = str(payload.get("plan_dir") or "").strip()
    if not recorded_plan_dir or _resolved(recorded_plan_dir) != plan_path.parent:
        raise ArtifactReadValidationError(
            f"copy plan directory identity failed: {plan_path}"
        )
    journal_path = plan_path.with_name("journal.jsonl")
    if not journal_path.is_file() or journal_path.is_symlink():
        raise FileNotFoundError(f"copy journal is missing: {journal_path}")
    raw_journal = journal_path.read_bytes()
    if not raw_journal or not raw_journal.endswith(b"\n"):
        raise ArtifactReadValidationError(
            f"copy journal is empty or torn: {journal_path}"
        )
    previous_hash: str | None = None
    events: list[dict[str, object]] = []
    for line_number, line in enumerate(raw_journal.splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ArtifactReadValidationError(
                f"invalid copy journal JSON at {journal_path}:{line_number}"
            ) from error
        if not isinstance(event, dict):
            raise ArtifactReadValidationError(
                f"copy journal event is not an object at {journal_path}:{line_number}"
            )
        if event.get("schema") != MIGRATION_COPY_JOURNAL_SCHEMA:
            raise ArtifactReadValidationError(
                f"unsupported copy journal schema at {journal_path}:{line_number}"
            )
        if str(event.get("plan_hash") or "") != expected_plan_hash:
            raise ArtifactReadValidationError(
                f"copy journal plan hash mismatch at {journal_path}:{line_number}"
            )
        if event.get("previous_event_hash") != previous_hash:
            raise ArtifactReadValidationError(
                f"copy journal chain mismatch at {journal_path}:{line_number}"
            )
        unsigned_event = dict(event)
        expected_event_hash = str(unsigned_event.pop("event_hash", "") or "")
        actual_event_hash = hashlib.sha256(
            _canonical_json_bytes(unsigned_event)
        ).hexdigest()
        if not expected_event_hash or expected_event_hash != actual_event_hash:
            raise ArtifactReadValidationError(
                f"copy journal event hash failed at {journal_path}:{line_number}"
            )
        previous_hash = expected_event_hash
        events.append(event)
    if str(events[-1].get("state")) not in {
        "published",
        "published_recovered",
    }:
        raise ArtifactReadValidationError(
            f"copy plan has no completed publication: {plan_path}"
        )
    source_text = str(payload.get("source") or "").strip()
    target_text = str(payload.get("target") or "").strip()
    if not source_text or not Path(source_text).expanduser().is_absolute():
        raise ArtifactReadValidationError(
            f"copy plan source must be absolute: {plan_path}"
        )
    if not target_text or not Path(target_text).expanduser().is_absolute():
        raise ArtifactReadValidationError(
            f"copy plan target must be absolute: {plan_path}"
        )
    target = _resolved(target_text)
    if not target.exists() or target.is_symlink():
        raise FileNotFoundError(
            f"verified copy target is missing or became a symlink: {target}"
        )
    migration_id = str(payload.get("migration_id") or "").strip()
    if not migration_id:
        raise ArtifactReadValidationError(
            f"copy plan has no migration id: {plan_path}"
        )
    snapshot = payload.get("source_snapshot")
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("entries"), list):
        raise ArtifactReadValidationError(
            f"copy plan source snapshot is missing: {plan_path}"
        )
    return ArtifactRelocation(
        source_root=_resolved(source_text),
        archive_root=target,
        manifest_path=plan_path,
        migration_id=migration_id,
        source_snapshot=snapshot,
    )


def _validate_copy_plan_destination(
    relocation: ArtifactRelocation,
    *,
    relative_path: Path,
    destination: Path,
) -> None:
    snapshot = relocation.source_snapshot
    if snapshot is None:
        return
    relative_text = "." if relative_path == Path(".") else relative_path.as_posix()
    entries = snapshot.get("entries")
    if not isinstance(entries, list):
        raise ArtifactReadValidationError(
            f"copy plan snapshot is malformed: {relocation.manifest_path}"
        )
    matching = [
        item
        for item in entries
        if isinstance(item, dict) and item.get("relative_path") == relative_text
    ]
    if len(matching) != 1:
        raise ArtifactReadValidationError(
            f"relocated path is absent or ambiguous in copy plan: {relative_text}"
        )
    entry = matching[0]
    kind = str(entry.get("kind") or "")
    if kind == "file":
        if not destination.is_file() or destination.is_symlink():
            raise ArtifactReadValidationError(
                f"relocated regular file changed type: {destination}"
            )
        if destination.stat().st_size != int(entry.get("size", -1)):
            raise ArtifactReadValidationError(
                f"relocated regular file size changed: {destination}"
            )
        if _manifest_sha256(destination) != str(entry.get("sha256") or ""):
            raise ArtifactReadValidationError(
                f"relocated regular file content hash failed: {destination}"
            )
        return
    if kind == "directory":
        if not destination.is_dir() or destination.is_symlink():
            raise ArtifactReadValidationError(
                f"relocated directory changed type: {destination}"
            )
        return
    if kind == "symlink":
        if not destination.is_symlink() or os.readlink(destination) != str(
            entry.get("link_target")
        ):
            raise ArtifactReadValidationError(
                f"relocated symlink changed: {destination}"
            )
        return
    raise ArtifactReadValidationError(
        f"unsupported relocated entry kind {kind!r}: {destination}"
    )


def load_artifact_relocations(
    manifest_paths: Sequence[str | os.PathLike[str]],
) -> tuple[ArtifactRelocation, ...]:
    """Load only verified copy manifests and reject archive-root escapes."""

    relocations: list[ArtifactRelocation] = []
    for value in manifest_paths:
        manifest_path = _resolved(value)
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise FileNotFoundError(
                f"relocation manifest is missing or became a symlink: {manifest_path}"
            )
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ArtifactReadValidationError(
                f"invalid relocation manifest JSON: {manifest_path}"
            ) from error
        if not isinstance(payload, dict):
            raise ArtifactReadValidationError(
                f"relocation manifest root is not an object: {manifest_path}"
            )
        if payload.get("schema") == MIGRATION_COPY_PLAN_SCHEMA:
            relocations.append(
                _load_copy_plan_relocation(manifest_path, payload)
            )
            continue
        checksum_path = manifest_path.with_name("manifest.sha256")
        if not checksum_path.is_file():
            raise FileNotFoundError(
                f"relocation manifest or checksum is missing: {manifest_path}"
            )
        checksum_fields = checksum_path.read_text(encoding="ascii").split()
        if not checksum_fields or checksum_fields[0] != _manifest_sha256(manifest_path):
            raise ArtifactReadValidationError(
                f"relocation manifest checksum failed: {manifest_path}"
            )
        if payload.get("schema") != MIGRATION_ARCHIVE_SCHEMA:
            raise ArtifactReadValidationError(
                f"unsupported relocation manifest: {manifest_path}"
            )
        if not payload.get("copy_completed") or not payload.get(
            "verification_completed"
        ):
            raise ArtifactReadValidationError(
                f"relocation manifest is not a verified copy: {manifest_path}"
            )
        archive_base = manifest_path.parent.resolve()
        migration_id = str(payload.get("migration_id", "") or "").strip()
        if not migration_id:
            raise ArtifactReadValidationError(
                f"relocation manifest has no migration id: {manifest_path}"
            )
        for mapping in payload.get("relocations", []):
            if not isinstance(mapping, dict):
                raise ArtifactReadValidationError(
                    f"invalid relocation mapping in {manifest_path}"
                )
            source_text = str(mapping.get("source_root", "") or "").strip()
            if not source_text or not Path(source_text).expanduser().is_absolute():
                raise ArtifactReadValidationError(
                    f"relocation source root must be absolute in {manifest_path}: "
                    f"{source_text!r}"
                )
            source_root = _resolved(source_text)
            relative_text = str(mapping.get("archive_relative_root", "") or "")
            relative = Path(relative_text)
            if (
                not relative_text
                or relative.is_absolute()
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise ArtifactReadValidationError(
                    f"unsafe archive-relative root in {manifest_path}: {relative_text!r}"
                )
            archive_root = (archive_base / relative).resolve()
            if not archive_root.is_relative_to(archive_base):
                raise ArtifactReadValidationError(
                    f"relocation archive root escapes manifest archive: {archive_root}"
                )
            if not archive_root.is_dir():
                raise FileNotFoundError(
                    f"relocation archive root does not exist: {archive_root}"
                )
            relocations.append(
                ArtifactRelocation(
                    source_root=source_root,
                    archive_root=archive_root,
                    manifest_path=manifest_path,
                    migration_id=migration_id,
                )
            )
    return tuple(relocations)


def resolve_historical_artifact_path(
    path: str | os.PathLike[str],
    *,
    manifest_paths: Sequence[str | os.PathLike[str]],
    warn_on_relocation: bool = True,
) -> HistoricalArtifactResolution:
    """Resolve an exact historical locator through append-only copy receipts."""

    requested = _resolved(path)
    if requested.exists():
        return HistoricalArtifactResolution(
            requested_path=requested,
            path=requested,
            source="original",
            relocation_chain=(),
        )
    relocations = load_artifact_relocations(manifest_paths)
    current = requested
    chain: list[ArtifactRelocation] = []
    visited = {current}
    for _hop in range(len(relocations) + 1):
        candidates = [
            relocation
            for relocation in relocations
            if current == relocation.source_root
            or current.is_relative_to(relocation.source_root)
        ]
        if not candidates:
            searched = ", ".join(str(item.manifest_path) for item in relocations)
            raise FileNotFoundError(
                f"historical artifact is missing and has no relocation: {requested}; "
                f"manifests={searched or '<none>'}"
            )
        maximum_depth = max(len(item.source_root.parts) for item in candidates)
        most_specific = [
            item for item in candidates if len(item.source_root.parts) == maximum_depth
        ]
        destinations = {
            item.archive_root / current.relative_to(item.source_root)
            for item in most_specific
        }
        if len(destinations) != 1:
            raise ArtifactReadValidationError(
                f"ambiguous relocation for historical artifact {current}: "
                + ", ".join(str(item) for item in sorted(destinations))
            )
        selected = sorted(
            most_specific,
            key=lambda item: (str(item.manifest_path), item.migration_id),
        )[0]
        destination = next(iter(destinations))
        relative_path = current.relative_to(selected.source_root)
        resolved_destination = destination.resolve()
        if not resolved_destination.is_relative_to(selected.archive_root):
            raise ArtifactReadValidationError(
                f"relocated artifact escapes verified archive root: {destination}"
            )
        chain.append(selected)
        if resolved_destination.exists():
            _validate_copy_plan_destination(
                selected,
                relative_path=relative_path,
                destination=destination,
            )
            if warn_on_relocation:
                warnings.warn(
                    f"Resolved missing historical runtime artifact through migration "
                    f"{selected.migration_id}: {requested} -> {resolved_destination}",
                    RelocatedArtifactWarning,
                    stacklevel=2,
                )
            return HistoricalArtifactResolution(
                requested_path=requested,
                path=resolved_destination,
                source="relocated",
                relocation_chain=tuple(chain),
            )
        current = resolved_destination
        if current in visited:
            raise ArtifactReadValidationError(
                f"relocation cycle detected for historical artifact: {requested}"
            )
        visited.add(current)
    raise ArtifactReadValidationError(
        f"relocation chain exceeded manifest count for historical artifact: {requested}"
    )
