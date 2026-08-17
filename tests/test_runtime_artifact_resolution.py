from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from runtime_artifact_resolution import (
    ArtifactReadValidationError,
    LegacyArtifactFallbackWarning,
    RelocatedArtifactWarning,
    load_artifact_relocations,
    resolve_artifact_read_path,
    resolve_historical_artifact_path,
)
from scripts import runtime_artifact_migration as migration


pytestmark = pytest.mark.integration


def _write(path: Path, text: str = "valid") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_explicit_path_is_exact_and_wins_over_other_candidates(tmp_path: Path) -> None:
    explicit = _write(tmp_path / "explicit.json")
    canonical = _write(tmp_path / "canonical.json")
    legacy = _write(tmp_path / "legacy.json")

    resolution = resolve_artifact_read_path(
        explicit_path=explicit,
        canonical_path=canonical,
        legacy_paths=(legacy,),
    )

    assert resolution.path == explicit.resolve()
    assert resolution.source == "explicit"
    assert resolution.attempted_paths == (explicit.resolve(),)


def test_missing_explicit_path_does_not_fall_back(tmp_path: Path) -> None:
    canonical = _write(tmp_path / "canonical.json")

    with pytest.raises(FileNotFoundError, match="Explicit artifact path"):
        resolve_artifact_read_path(
            explicit_path=tmp_path / "missing.json",
            canonical_path=canonical,
        )


def test_canonical_candidate_wins_even_when_legacy_mtime_is_newer(
    tmp_path: Path,
) -> None:
    canonical = _write(tmp_path / "canonical.json")
    legacy = _write(tmp_path / "legacy.json")
    canonical_time = 1_700_000_000
    legacy_time = canonical_time + 10_000
    os.utime(canonical, (canonical_time, canonical_time))
    os.utime(legacy, (legacy_time, legacy_time))

    resolution = resolve_artifact_read_path(
        canonical_path=canonical,
        legacy_paths=(legacy,),
    )

    assert resolution.path == canonical.resolve()
    assert resolution.source == "canonical"


def test_missing_canonical_candidate_uses_ordered_legacy_fallback(
    tmp_path: Path,
) -> None:
    first_legacy = _write(tmp_path / "legacy-a.json")
    second_legacy = _write(tmp_path / "legacy-b.json")

    with pytest.warns(LegacyArtifactFallbackWarning, match="legacy-a.json"):
        resolution = resolve_artifact_read_path(
            canonical_path=tmp_path / "missing-canonical.json",
            legacy_paths=(first_legacy, second_legacy),
        )

    assert resolution.path == first_legacy.resolve()
    assert resolution.source == "legacy"
    assert resolution.attempted_paths == (
        (tmp_path / "missing-canonical.json").resolve(),
        first_legacy.resolve(),
    )


def test_invalid_canonical_candidate_fails_closed_without_legacy_fallback(
    tmp_path: Path,
) -> None:
    canonical = _write(tmp_path / "canonical.json", "invalid")
    legacy = _write(tmp_path / "legacy.json", "valid")

    with pytest.raises(ArtifactReadValidationError, match="canonical"):
        resolve_artifact_read_path(
            canonical_path=canonical,
            legacy_paths=(legacy,),
            validate=lambda path: path.read_text(encoding="utf-8") == "valid",
        )


def test_validator_exception_is_reported_with_candidate_source(tmp_path: Path) -> None:
    legacy = _write(tmp_path / "legacy.json")

    def fail_validation(path: Path) -> bool:
        raise RuntimeError(f"cannot parse {path.name}")

    with pytest.raises(ArtifactReadValidationError, match="legacy.*cannot parse"):
        resolve_artifact_read_path(
            canonical_path=tmp_path / "missing.json",
            legacy_paths=(legacy,),
            validate=fail_validation,
        )


def test_no_existing_candidate_reports_search_order(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.json"
    legacy = tmp_path / "legacy.json"

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_artifact_read_path(
            canonical_path=canonical,
            legacy_paths=(None, legacy),
        )

    message = str(exc_info.value)
    assert str(canonical.resolve()) in message
    assert str(legacy.resolve()) in message


def test_legacy_warning_can_be_suppressed_by_explicit_caller_policy(
    tmp_path: Path,
) -> None:
    legacy = _write(tmp_path / "legacy.json")

    resolution = resolve_artifact_read_path(
        legacy_paths=(legacy,),
        warn_on_legacy=False,
    )

    assert resolution.source == "legacy"


def _relocation_manifest(
    root: Path,
    *,
    migration_id: str,
    source_root: Path,
    archive_relative_root: str = "payload/legacy",
) -> Path:
    archive_root = root / archive_relative_root
    archive_root.mkdir(parents=True)
    payload = {
        "schema": "dexteleop.runtime_artifact_archive.v1",
        "migration_id": migration_id,
        "copy_completed": True,
        "verification_completed": True,
        "relocations": [
            {
                "source_root": str(source_root.resolve()),
                "archive_relative_root": archive_relative_root,
            }
        ],
    }
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (root / "manifest.sha256").write_text(
        f"{digest}  manifest.json\n",
        encoding="ascii",
    )
    return manifest


def test_existing_historical_path_wins_without_loading_manifest(tmp_path: Path) -> None:
    existing = _write(tmp_path / "old" / "trace.jsonl")

    resolution = resolve_historical_artifact_path(
        existing,
        manifest_paths=(tmp_path / "missing-manifest.json",),
    )

    assert resolution.source == "original"
    assert resolution.path == existing.resolve()
    assert resolution.relocation_chain == ()


def test_missing_historical_path_resolves_through_verified_archive(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old-location"
    archive = tmp_path / "archive" / "migration-001"
    manifest = _relocation_manifest(
        archive,
        migration_id="migration-001",
        source_root=old_root,
    )
    relocated = _write(archive / "payload" / "legacy" / "nested" / "trace.jsonl")

    with pytest.warns(RelocatedArtifactWarning, match="migration-001"):
        resolution = resolve_historical_artifact_path(
            old_root / "nested" / "trace.jsonl",
            manifest_paths=(manifest,),
        )

    assert resolution.source == "relocated"
    assert resolution.path == relocated.resolve()
    assert resolution.relocation_chain[0].manifest_path == manifest.resolve()


def test_relocation_manifest_checksum_failure_is_fail_closed(tmp_path: Path) -> None:
    manifest = _relocation_manifest(
        tmp_path / "archive",
        migration_id="migration-001",
        source_root=tmp_path / "old",
    )
    manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArtifactReadValidationError, match="checksum"):
        load_artifact_relocations((manifest,))


def test_conflicting_relocations_are_rejected_as_ambiguous(tmp_path: Path) -> None:
    old_root = tmp_path / "old"
    first = _relocation_manifest(
        tmp_path / "archive-a",
        migration_id="migration-a",
        source_root=old_root,
    )
    second = _relocation_manifest(
        tmp_path / "archive-b",
        migration_id="migration-b",
        source_root=old_root,
    )
    _write(first.parent / "payload" / "legacy" / "trace.jsonl", "a")
    _write(second.parent / "payload" / "legacy" / "trace.jsonl", "b")

    with pytest.raises(ArtifactReadValidationError, match="ambiguous"):
        resolve_historical_artifact_path(
            old_root / "trace.jsonl",
            manifest_paths=(first, second),
        )


def test_relocation_symlink_must_not_escape_verified_archive(tmp_path: Path) -> None:
    old_root = tmp_path / "old"
    archive = tmp_path / "archive"
    manifest = _relocation_manifest(
        archive,
        migration_id="migration-escape",
        source_root=old_root,
    )
    outside = _write(tmp_path / "outside.json")
    os.symlink(outside, archive / "payload" / "legacy" / "trace.jsonl")

    with pytest.raises(ArtifactReadValidationError, match="escapes"):
        resolve_historical_artifact_path(
            old_root / "trace.jsonl",
            manifest_paths=(manifest,),
        )


def _published_copy_plan(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "old-runtime-root"
    target_parent = tmp_path / "canonical"
    target = target_parent / "run-001"
    plan_dir = tmp_path / "migration" / "copy-001"
    source.mkdir()
    target_parent.mkdir()
    _write(source / "nested" / "trace.jsonl", "trace-content")
    migration.create_plan(
        source=source,
        target=target,
        plan_dir=plan_dir,
        migration_id="copy-001",
    )
    migration.copy_verify(plan_dir)
    return source, target, plan_dir


def test_missing_historical_path_resolves_through_published_copy_plan(
    tmp_path: Path,
) -> None:
    source, target, plan_dir = _published_copy_plan(tmp_path)
    requested = source / "nested" / "trace.jsonl"
    source.rename(tmp_path / "retained-source-outside-old-locator")

    with pytest.warns(RelocatedArtifactWarning, match="copy-001"):
        resolution = resolve_historical_artifact_path(
            requested,
            manifest_paths=(plan_dir / migration.PLAN_FILENAME,),
        )

    assert resolution.source == "relocated"
    assert resolution.path == (target / "nested" / "trace.jsonl").resolve()
    assert resolution.path.read_text(encoding="utf-8") == "trace-content"


def test_copy_plan_historical_resolution_hashes_the_selected_file(
    tmp_path: Path,
) -> None:
    source, target, plan_dir = _published_copy_plan(tmp_path)
    requested = source / "nested" / "trace.jsonl"
    source.rename(tmp_path / "retained-source-outside-old-locator")
    (target / "nested" / "trace.jsonl").write_text(
        "other-content",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactReadValidationError, match="content hash"):
        resolve_historical_artifact_path(
            requested,
            manifest_paths=(plan_dir / migration.PLAN_FILENAME,),
        )


def test_unpublished_copy_plan_is_not_a_relocation_receipt(tmp_path: Path) -> None:
    source = tmp_path / "old-runtime-root"
    target_parent = tmp_path / "canonical"
    target = target_parent / "run-001"
    plan_dir = tmp_path / "migration" / "copy-001"
    source.mkdir()
    target_parent.mkdir()
    _write(source / "trace.jsonl")
    migration.create_plan(
        source=source,
        target=target,
        plan_dir=plan_dir,
        migration_id="copy-001",
    )

    with pytest.raises(ArtifactReadValidationError, match="no completed publication"):
        load_artifact_relocations((plan_dir / migration.PLAN_FILENAME,))
