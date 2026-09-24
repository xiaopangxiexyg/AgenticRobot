from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import runtime_artifact_migration as migration

pytestmark = pytest.mark.integration


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source" / "diagnostic-run"
    target_parent = tmp_path / "canonical"
    target = target_parent / "diagnostic-run"
    plan_dir = tmp_path / "migration" / "tx-1"
    source.mkdir(parents=True)
    target_parent.mkdir()
    return source, target, plan_dir


def _build_source_tree(source: Path) -> None:
    payload = source / "payload.bin"
    payload.write_bytes(b"artifact-payload\n" * 64)
    os.link(payload, source / "payload-hardlink.bin")
    (source / "nested").mkdir()
    (source / "nested" / "metadata.json").write_text(
        '{"schema":"test.v1"}\n',
        encoding="utf-8",
    )
    (source / "metadata-link").symlink_to("nested/metadata.json")
    sparse = source / "control_trace.wal"
    with sparse.open("wb") as stream:
        stream.write(b"WAL1")
        stream.seek((8 << 20) - 4)
        stream.write(b"TAIL")


def test_plan_copy_verify_preserves_sparse_links_and_source(tmp_path: Path) -> None:
    source, target, plan_dir = _paths(tmp_path)
    _build_source_tree(source)
    source_payload_inode = (source / "payload.bin").stat().st_ino

    plan = migration.create_plan(
        source=source,
        target=target,
        plan_dir=plan_dir,
        migration_id="tx-1",
    )

    assert not target.exists()
    assert plan["capabilities"]["delete_source"] is False
    assert plan["capabilities"]["switch_reader_default"] is False
    assert plan["source_snapshot"]["summary"]["sparse_file_count"] == 1
    assert plan["source_snapshot"]["summary"]["hardlinked_path_count"] == 2

    receipt = migration.copy_verify(plan_dir)

    assert receipt["source_retained"] is True
    assert source.is_dir()
    assert target.is_dir()
    assert (target / "payload.bin").read_bytes() == (
        source / "payload.bin"
    ).read_bytes()
    assert os.readlink(target / "metadata-link") == "nested/metadata.json"
    assert (target / "payload.bin").stat().st_ino == (
        target / "payload-hardlink.bin"
    ).stat().st_ino
    assert (target / "payload.bin").stat().st_ino != source_payload_inode
    target_sparse = (target / "control_trace.wal").stat()
    assert target_sparse.st_blocks * 512 < target_sparse.st_size

    events = migration._read_journal(plan_dir / migration.JOURNAL_FILENAME)
    assert [event["state"] for event in events] == [
        "planned",
        "copy_started",
        "copy_verified",
        "published",
    ]
    assert migration.verify_plan(plan_dir)["source_retained"] is True

    # Reuse the published tree to check idempotency, including its special files.
    event_count = len(events)
    second = migration.copy_verify(plan_dir)

    assert receipt["source_identity_hash"] == second["source_identity_hash"]
    assert len(migration._read_journal(plan_dir / migration.JOURNAL_FILENAME)) == (
        event_count
    )


def test_plan_records_external_hardlinks_without_recreating_them(
    tmp_path: Path,
) -> None:
    source, target, plan_dir = _paths(tmp_path)
    _build_source_tree(source)
    external = tmp_path / "external-payload.bin"
    os.link(source / "payload.bin", external)

    plan = migration.create_plan(
        source=source,
        target=target,
        plan_dir=plan_dir,
        migration_id="tx-1",
    )
    payload_records = [
        entry
        for entry in plan["source_snapshot"]["entries"]
        if entry["relative_path"] in {"payload.bin", "payload-hardlink.bin"}
    ]
    assert {entry["external_link_count"] for entry in payload_records} == {1}

    migration.copy_verify(plan_dir)

    assert (target / "payload.bin").stat().st_nlink == 2
    assert external.stat().st_ino == (source / "payload.bin").stat().st_ino
    assert external.stat().st_ino != (target / "payload.bin").stat().st_ino


def test_copy_refuses_source_change_after_frozen_plan(tmp_path: Path) -> None:
    source, target, plan_dir = _paths(tmp_path)
    _build_source_tree(source)
    migration.create_plan(
        source=source,
        target=target,
        plan_dir=plan_dir,
        migration_id="tx-1",
    )
    (source / "nested" / "metadata.json").write_text(
        '{"schema":"changed"}\n',
        encoding="utf-8",
    )

    with pytest.raises(migration.MigrationSafetyError, match="source identity changed"):
        migration.copy_verify(plan_dir)

    assert source.exists()
    assert not target.exists()
    events = migration._read_journal(plan_dir / migration.JOURNAL_FILENAME)
    assert events[-1]["state"] == "copy_failed"
    assert events[-1]["details"]["source_retained"] is True


def test_plan_and_copy_never_overwrite_an_existing_target(tmp_path: Path) -> None:
    source, target, plan_dir = _paths(tmp_path)
    _build_source_tree(source)
    target.mkdir()
    sentinel = target / "owned.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(migration.MigrationSafetyError, match="target must be absent"):
        migration.create_plan(
            source=source,
            target=target,
            plan_dir=plan_dir,
            migration_id="tx-1",
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not plan_dir.exists()


def test_plan_json_publication_never_overwrites_a_racing_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "plan.json"
    original_rename_noreplace = migration._rename_noreplace

    def publish_after_racing_writer(source: Path, target: Path) -> None:
        target.write_text("racing-writer\n", encoding="utf-8")
        original_rename_noreplace(source, target)

    monkeypatch.setattr(
        migration,
        "_rename_noreplace",
        publish_after_racing_writer,
    )
    with pytest.raises(migration.MigrationSafetyError, match="target appeared"):
        migration._write_json_exclusive(destination, {"schema": "test.v1"})

    assert destination.read_text(encoding="utf-8") == "racing-writer\n"


def test_plan_rejects_relative_overlapping_and_special_paths(tmp_path: Path) -> None:
    source, target, plan_dir = _paths(tmp_path)
    _build_source_tree(source)
    with pytest.raises(migration.MigrationSafetyError, match="absolute path"):
        migration.create_plan(
            source=Path("relative-source"),
            target=target,
            plan_dir=plan_dir,
            migration_id="tx-1",
        )

    with pytest.raises(migration.MigrationSafetyError, match="contain one another"):
        migration.create_plan(
            source=source,
            target=source / "copied",
            plan_dir=plan_dir,
            migration_id="tx-1",
        )

    fifo = source / "writer.pipe"
    os.mkfifo(fifo)
    with pytest.raises(migration.MigrationSafetyError, match="unsupported FIFO"):
        migration.create_plan(
            source=source,
            target=target,
            plan_dir=plan_dir,
            migration_id="tx-1",
        )
    assert not plan_dir.exists()


def test_plan_hash_and_journal_chain_are_fail_closed(tmp_path: Path) -> None:
    source, target, plan_dir = _paths(tmp_path)
    _build_source_tree(source)
    migration.create_plan(
        source=source,
        target=target,
        plan_dir=plan_dir,
        migration_id="tx-1",
    )
    plan_path = plan_dir / migration.PLAN_FILENAME
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["target"] = str(tmp_path / "unexpected")
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(migration.MigrationSafetyError, match="plan hash mismatch"):
        migration.copy_verify(plan_dir)

    assert not target.exists()


def test_copy_recovers_recorded_post_publish_identity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, plan_dir = _paths(tmp_path)
    _build_source_tree(source)
    migration.create_plan(
        source=source,
        target=target,
        plan_dir=plan_dir,
        migration_id="tx-1",
    )
    original_scan_tree = migration._scan_tree

    def scan_with_stale_prepublication_identity(root: Path):
        snapshot = original_scan_tree(root)
        if Path(root).name.endswith(".copying"):
            snapshot["publication_identity_hash"] = "0" * 64
        return snapshot

    monkeypatch.setattr(
        migration,
        "_scan_tree",
        scan_with_stale_prepublication_identity,
    )
    with pytest.raises(
        migration.MigrationSafetyError,
        match="target identity changed during atomic publication",
    ):
        migration.copy_verify(plan_dir)

    assert source.is_dir()
    assert target.is_dir()
    assert not migration._temporary_target(target, "tx-1").exists()

    monkeypatch.setattr(migration, "_scan_tree", original_scan_tree)
    receipt = migration.copy_verify(plan_dir)
    events = migration._read_journal(plan_dir / migration.JOURNAL_FILENAME)

    assert receipt["source_retained"] is True
    assert events[-1]["state"] == "published_recovered"
    assert events[-1]["details"]["recovery_reason"] == (
        "verified_post_fsync_publication_identity_reconciliation"
    )


def test_copy_preserves_extended_attributes_when_supported(tmp_path: Path) -> None:
    source, target, plan_dir = _paths(tmp_path)
    _build_source_tree(source)
    payload = source / "payload.bin"
    try:
        os.setxattr(payload, "user.dexteleop_test", b"traceable")
    except OSError as error:
        pytest.skip(f"test filesystem does not support user xattrs: {error}")

    plan = migration.create_plan(
        source=source,
        target=target,
        plan_dir=plan_dir,
        migration_id="tx-1",
    )
    assert plan["backup_assessment"]["independent_backup_proven"] is False
    migration.copy_verify(plan_dir)

    assert os.getxattr(target / "payload.bin", "user.dexteleop_test") == b"traceable"


def test_copy_refuses_replaced_target_parent(tmp_path: Path) -> None:
    source, target, plan_dir = _paths(tmp_path)
    _build_source_tree(source)
    migration.create_plan(
        source=source,
        target=target,
        plan_dir=plan_dir,
        migration_id="tx-1",
    )
    original_parent = target.parent.with_name("canonical-original")
    target.parent.rename(original_parent)
    target.parent.mkdir()

    with pytest.raises(
        migration.MigrationSafetyError,
        match="target parent identity changed",
    ):
        migration.copy_verify(plan_dir)

    assert source.exists()
    assert not target.exists()
