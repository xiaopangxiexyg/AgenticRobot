from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import runtime_artifact_migration as archive
from scripts import runtime_artifact_archive as archive_cli


pytestmark = pytest.mark.integration


def _source(root: Path, name: str, payload: bytes = b"artifact\n") -> Path:
    source = root / name
    source.mkdir(parents=True)
    data = source / "data.bin"
    data.write_bytes(payload)
    os.link(data, source / "data-hardlink.bin")
    sparse = source / "trace.wal"
    with sparse.open("wb") as stream:
        stream.write(b"HEAD")
        stream.seek((2 << 20) - 4)
        stream.write(b"TAIL")
    return source


def test_multi_source_archive_is_verified_copy_only(tmp_path: Path) -> None:
    logs = _source(tmp_path, "logs", b"logs\n")
    records = _source(tmp_path, "teleop_records", b"records\n")
    sources = (
        archive.MigrationSource("logs", logs),
        archive.MigrationSource("teleop_records", records),
    )
    archive_base = tmp_path / "legacy" / "runtime_artifact_archive"

    plan = archive.plan_archive_copy(
        sources,
        archive_base=archive_base,
        migration_id="pre-switch",
        quiescence_sec=0,
    )
    assert plan["copy_allowed"] is True
    assert plan["deletion_supported"] is False
    assert not Path(plan["final_archive_root"]).exists()

    manifest = archive.copy_sources_to_archive(
        sources,
        archive_base=archive_base,
        migration_id="pre-switch",
        quiescence_sec=0,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    copied_logs = manifest.parent / "payload" / "logs"

    assert logs.is_dir() and records.is_dir()
    assert payload["source_deletion_permitted"] is False
    assert payload["source_deletion_tool_available"] is False
    assert payload["archive_is_independent_device"] is False
    assert payload["source_inventory_sha256"] == payload[
        "post_copy_source_inventory_sha256"
    ]
    assert (copied_logs / "data.bin").stat().st_ino == (
        copied_logs / "data-hardlink.bin"
    ).stat().st_ino
    assert (copied_logs / "data.bin").stat().st_ino != (
        logs / "data.bin"
    ).stat().st_ino
    assert (copied_logs / "trace.wal").stat().st_blocks * 512 < (
        copied_logs / "trace.wal"
    ).stat().st_size
    assert archive.verify_archive(manifest)["passed"] is True


def test_archive_verification_detects_destination_mutation(tmp_path: Path) -> None:
    source = _source(tmp_path, "logs")
    manifest = archive.copy_sources_to_archive(
        (archive.MigrationSource("logs", source),),
        archive_base=tmp_path / "legacy",
        migration_id="mutation-check",
        quiescence_sec=0,
    )
    destination = manifest.parent / "payload" / "logs" / "data.bin"
    destination.write_bytes(b"changed\n")

    verification = archive.verify_archive(manifest)

    assert verification["passed"] is False
    assert {failure["code"] for failure in verification["failures"]} >= {
        "destination_hash_mismatch"
    }
    assert source.is_dir()


def test_archive_verification_detects_destination_metadata_mutation(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, "logs")
    manifest = archive.copy_sources_to_archive(
        (archive.MigrationSource("logs", source),),
        archive_base=tmp_path / "legacy",
        migration_id="metadata-check",
        quiescence_sec=0,
    )
    destination = manifest.parent / "payload" / "logs" / "data.bin"
    destination.chmod(0o600)

    verification = archive.verify_archive(manifest)

    assert verification["passed"] is False
    assert {failure["code"] for failure in verification["failures"]} >= {
        "destination_mode_mismatch"
    }


def test_archive_verifier_accepts_older_receipts_without_mode_field(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, "logs")
    manifest = archive.copy_sources_to_archive(
        (archive.MigrationSource("logs", source),),
        archive_base=tmp_path / "legacy",
        migration_id="old-receipt",
        quiescence_sec=0,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for receipt in payload["files"]:
        receipt.pop("source_mode", None)
        receipt.pop("source_ctime_ns", None)
    payload.pop("post_copy_source_inventory_sha256", None)
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest.with_name("manifest.sha256").write_text(
        archive._sha256_file(manifest) + "  manifest.json\n",
        encoding="ascii",
    )

    assert archive.verify_archive(manifest)["passed"] is True


def test_preflight_blocks_recent_recovery_and_special_entries(tmp_path: Path) -> None:
    source = _source(tmp_path, "logs")
    (source / "capture.recovery.json").write_text(
        '{"status":"pending"}\n',
        encoding="utf-8",
    )
    os.mkfifo(source / "writer.pipe")

    plan = archive.plan_archive_copy(
        (archive.MigrationSource("logs", source),),
        archive_base=tmp_path / "legacy",
        migration_id="blocked",
        quiescence_sec=3600,
    )

    assert plan["copy_allowed"] is False
    assert {blocker["code"] for blocker in plan["blockers"]} == {
        "unfinished_recovery_journal",
        "special_files_present",
        "source_not_quiescent",
    }


def test_archive_rejects_symlink_source_root(tmp_path: Path) -> None:
    source = _source(tmp_path, "logs")
    source_link = tmp_path / "logs-link"
    source_link.symlink_to(source, target_is_directory=True)

    with pytest.raises(archive.MigrationSafetyError, match="must not be a symlink"):
        archive.inventory_sources((archive.MigrationSource("logs", source_link),))


def test_archive_detects_tree_change_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, "logs")
    original_copy = archive._copy_regular_file
    inserted = False

    def copy_and_mutate(source_path: Path, destination: Path) -> None:
        nonlocal inserted
        original_copy(source_path, destination)
        if not inserted:
            inserted = True
            (source / "late-file.bin").write_bytes(b"late\n")

    monkeypatch.setattr(archive, "_copy_regular_file", copy_and_mutate)

    with pytest.raises(archive.MigrationSafetyError, match="source tree changed"):
        archive.copy_sources_to_archive(
            (archive.MigrationSource("logs", source),),
            archive_base=tmp_path / "legacy",
            migration_id="moving-source",
            quiescence_sec=0,
        )

    assert source.is_dir()
    assert not (tmp_path / "legacy" / "moving-source").exists()
    assert (tmp_path / "legacy" / ".moving-source.partial").is_dir()


def test_archive_detects_post_copy_rewrite_with_restored_mtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, "logs", b"good\n")
    original_copy = archive._copy_regular_file
    mutated = False

    def copy_then_mutate_earlier_file(source_path: Path, destination: Path) -> None:
        nonlocal mutated
        original_copy(source_path, destination)
        if source_path.name != "trace.wal" or mutated:
            return
        mutated = True
        earlier = source / "data.bin"
        original_mtime_ns = earlier.stat().st_mtime_ns
        earlier.write_bytes(b"evil\n")
        os.utime(earlier, ns=(original_mtime_ns, original_mtime_ns))

    monkeypatch.setattr(archive, "_copy_regular_file", copy_then_mutate_earlier_file)

    with pytest.raises(archive.MigrationSafetyError, match="source tree changed"):
        archive.copy_sources_to_archive(
            (archive.MigrationSource("logs", source),),
            archive_base=tmp_path / "legacy",
            migration_id="ctime-check",
            quiescence_sec=0,
        )

    assert source.is_dir()
    assert not (tmp_path / "legacy" / "ctime-check").exists()


def test_archive_cli_requires_explicit_copy_acknowledgement(tmp_path: Path) -> None:
    source = _source(tmp_path, "logs")

    with pytest.raises(SystemExit) as error:
        archive_cli.main(
            [
                "copy",
                "--source",
                f"logs={source}",
                "--migration-id",
                "cli-check",
                "--archive-base",
                str(tmp_path / "legacy"),
            ]
        )

    assert error.value.code == 2
    assert source.is_dir()
    assert not (tmp_path / "legacy").exists()


def test_archive_cli_returns_nonzero_when_verification_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source(tmp_path, "logs")
    manifest = archive.copy_sources_to_archive(
        (archive.MigrationSource("logs", source),),
        archive_base=tmp_path / "legacy",
        migration_id="cli-verification-failure",
        quiescence_sec=0,
    )
    destination = manifest.parent / "payload" / "logs" / "data.bin"
    destination.write_bytes(b"changed\n")

    exit_code = archive_cli.main(["verify", "--manifest", str(manifest)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 3
    assert output["passed"] is False
