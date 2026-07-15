from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from utils.dataset_postprocess.parquet_schema_migration import (  # noqa: E402
    DEFAULT_STATE_DIR_NAME,
    MigrationError,
    audit_parquet_file,
    cleanup_migration_backups,
    migrate_parquet_file,
    run_migration,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_legacy_parquet(path: Path, *, metadata: dict[bytes, bytes] | None = None) -> None:
    """Write two legacy row groups with canonical and deliberately dynamic fields."""

    schema = pa.schema(
        [
            pa.field("index", pa.int64()),
            pa.field("action", pa.list_(pa.float64())),
            pa.field("observation.state", pa.list_(pa.float64())),
            pa.field("fault_codes", pa.list_(pa.null())),
            # This column is already correct and contains real labels.  A
            # migration must cast neither its type nor its values.
            pa.field("teleop.arm_mode.left.tags", pa.list_(pa.string())),
            # These two fields are intentionally outside schema_v21's
            # canonical sets and must remain physically unchanged.
            pa.field("diagnostics.dynamic_all_null", pa.null()),
            pa.field("observation.sim_camera.front.pose_world", pa.list_(pa.float64())),
        ],
        metadata=metadata,
    )
    rows = (
        (
            [0, 1],
            [[0.1, float("nan"), -0.0], [1.25, float("inf"), -float("inf")]],
            [[0.2, 0.3], [0.4, float("nan")]],
            [[], []],
            [["task_hold"], []],
            [None, None],
            [[np.pi, 1.0 / 3.0], [np.e, -np.pi]],
        ),
        (
            [2, 3],
            [[-0.1, 2.5, float("nan")], [4.0, 5.0, 6.0]],
            [[-0.2, -0.3], [7.0, 8.0]],
            [[], []],
            [["init_pose"], ["holding", "left"]],
            [None, None],
            [[np.sqrt(2.0), np.sqrt(3.0)], [0.123456789012345, -9.5]],
        ),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for values in rows:
            writer.write_table(pa.Table.from_arrays([pa.array(v, type=f.type) for v, f in zip(values, schema)], schema=schema))


def _assert_float_lists_equal(actual: list[list[float] | None], expected: list[list[float] | None]) -> None:
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        if actual_row is None or expected_row is None:
            assert actual_row is expected_row
            continue
        np.testing.assert_array_equal(
            np.asarray(actual_row, dtype=np.float32),
            np.asarray(expected_row, dtype=np.float32),
        )


def _write_episode_manifest(dataset_root: Path, parquet_path: Path, *, frame_count: int) -> Path:
    manifest = dataset_root / "meta/episodes.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "episode_index": 0,
        "frame_count": int(frame_count),
        "parquet_path": str(parquet_path.relative_to(dataset_root)),
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest


def test_dry_run_is_read_only_and_reports_legacy_file(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    parquet_path = root / "session" / "data/chunk-000/episode_000000.parquet"
    _write_legacy_parquet(parquet_path)
    before_hash = _sha256(parquet_path)
    before_tree = sorted(path.relative_to(root) for path in root.rglob("*"))

    audit = audit_parquet_file(parquet_path)
    report = run_migration([root], apply=False, min_age_seconds=0)

    # Avoid coupling the safety test to report presentation details while still
    # proving that both public audit paths complete successfully.
    assert audit is not None
    assert report is not None
    assert _sha256(parquet_path) == before_hash
    assert sorted(path.relative_to(root) for path in root.rglob("*")) == before_tree


def test_apply_only_casts_present_canonical_columns_and_preserves_layout(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    backup_root = tmp_path / "backups"
    parquet_path = root / "session" / "data/chunk-000/episode_000000.parquet"
    schema_metadata = {b"dataset-contract": b"dexteleop-v2.1", b"custom": b"keep-me"}
    _write_legacy_parquet(parquet_path, metadata=schema_metadata)
    os.chmod(parquet_path, 0o755)

    source = pq.ParquetFile(parquet_path)
    source_names = source.schema_arrow.names
    source_rows = source.read().to_pydict()
    source_sim_values = source_rows["observation.sim_camera.front.pose_world"]
    source_tags = source_rows["teleop.arm_mode.left.tags"]

    result = migrate_parquet_file(
        parquet_path,
        root=root,
        backup_root=backup_root,
        min_age_seconds=0,
    )
    assert result is not None

    migrated = pq.ParquetFile(parquet_path)
    migrated_schema = migrated.schema_arrow
    migrated_rows = migrated.read().to_pydict()

    assert migrated_schema.names == source_names
    assert migrated_schema.metadata == schema_metadata
    assert migrated.metadata.num_rows == source.metadata.num_rows == 4
    assert migrated.metadata.num_row_groups == source.metadata.num_row_groups == 2
    assert {
        migrated.metadata.row_group(row_group).column(column).compression
        for row_group in range(migrated.metadata.num_row_groups)
        for column in range(migrated.metadata.row_group(row_group).num_columns)
    } == {"ZSTD"}
    assert stat.S_IMODE(parquet_path.stat().st_mode) == 0o755

    assert migrated_schema.field("action").type == pa.list_(pa.float32())
    assert migrated_schema.field("observation.state").type == pa.list_(pa.float32())
    assert migrated_schema.field("fault_codes").type == pa.list_(pa.string())

    # Already-canonical, non-empty tags survive exactly.  A missing right-side
    # tag column is not synthesized.
    assert migrated_schema.field("teleop.arm_mode.left.tags").type == pa.list_(pa.string())
    assert migrated_rows["teleop.arm_mode.left.tags"] == source_tags
    assert "teleop.arm_mode.right.tags" not in migrated_schema.names

    # Dynamic null and simulator double fields are explicitly not canonical.
    assert migrated_schema.field("diagnostics.dynamic_all_null").type == pa.null()
    assert migrated_schema.field("observation.sim_camera.front.pose_world").type == pa.list_(pa.float64())
    assert migrated_rows["diagnostics.dynamic_all_null"] == [None] * 4
    assert migrated_rows["observation.sim_camera.front.pose_world"] == source_sim_values

    _assert_float_lists_equal(migrated_rows["action"], source_rows["action"])
    _assert_float_lists_equal(migrated_rows["observation.state"], source_rows["observation.state"])
    assert migrated_rows["fault_codes"] == source_rows["fault_codes"]


def test_apply_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    parquet_path = root / "session/data/chunk-000/episode_000000.parquet"
    _write_legacy_parquet(parquet_path)

    migrate_parquet_file(
        parquet_path,
        root=root,
        backup_root=tmp_path / "backups-first",
        min_age_seconds=0,
    )
    first_hash = _sha256(parquet_path)
    first_stat = parquet_path.stat()

    second = migrate_parquet_file(
        parquet_path,
        root=root,
        backup_root=tmp_path / "backups-second",
        min_age_seconds=0,
    )

    assert second is not None
    assert _sha256(parquet_path) == first_hash
    assert parquet_path.stat().st_ino == first_stat.st_ino
    assert not (tmp_path / "backups-second").exists()


def test_scalar_null_vectors_arm_scalars_and_list_child_metadata_are_normalized(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    parquet_path = root / "session/data/chunk-000/episode_000000.parquet"
    child_metadata = {b"semantic": b"robot-state", b"units": b"mixed"}
    state_type = pa.list_(
        pa.field(
            "state_value",
            pa.float64(),
            nullable=False,
            metadata=child_metadata,
        )
    )
    schema = pa.schema(
        [
            # Old all-null episodes inferred canonical vectors as scalar null.
            pa.field("action", pa.null()),
            pa.field("observation.state", state_type),
            # Old arm-mode scalars could be inferred as bool or scalar null.
            pa.field("teleop.arm_mode.available", pa.bool_()),
            pa.field("teleop.arm_mode.left.id", pa.null()),
            pa.field("teleop.arm_mode.left.label", pa.null()),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.nulls(3),
            pa.array([[0.1, -0.0], [float("nan"), 1.5], [2.0, 3.0]], type=state_type),
            pa.array([True, False, None], type=pa.bool_()),
            pa.nulls(3),
            pa.nulls(3),
        ],
        schema=schema,
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path, compression="zstd")
    source_state_field = pq.read_schema(parquet_path).field("observation.state")

    migrate_parquet_file(
        parquet_path,
        root=root,
        backup_root=tmp_path / "backups",
        min_age_seconds=0,
    )

    migrated = pq.read_table(parquet_path)
    assert migrated.schema.field("action").type == pa.list_(pa.float32())
    assert migrated.column("action").to_pylist() == [None, None, None]
    state_field = migrated.schema.field("observation.state")
    assert state_field.type.value_type == pa.float32()
    assert state_field.type.value_field.name == source_state_field.type.value_field.name
    assert (
        state_field.type.value_field.nullable
        is source_state_field.type.value_field.nullable
    )
    assert state_field.type.value_field.metadata == source_state_field.type.value_field.metadata
    assert state_field.type.value_field.metadata == child_metadata
    _assert_float_lists_equal(
        migrated.column("observation.state").to_pylist(),
        table.column("observation.state").to_pylist(),
    )
    assert migrated.schema.field("teleop.arm_mode.available").type == pa.int64()
    assert migrated.column("teleop.arm_mode.available").to_pylist() == [1, 0, None]
    assert migrated.schema.field("teleop.arm_mode.left.id").type == pa.int64()
    assert migrated.column("teleop.arm_mode.left.id").to_pylist() == [None] * 3
    assert migrated.schema.field("teleop.arm_mode.left.label").type == pa.string()
    assert migrated.column("teleop.arm_mode.left.label").to_pylist() == [None] * 3


def test_episode_manifest_frame_count_is_part_of_preflight(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    dataset_root = root / "session"
    parquet_path = dataset_root / "data/chunk-000/episode_000000.parquet"
    _write_legacy_parquet(parquet_path)
    manifest = _write_episode_manifest(dataset_root, parquet_path, frame_count=4)

    valid = run_migration([root], apply=False, min_age_seconds=0)
    assert valid.ok
    assert valid.manifest_records_checked == 1

    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["frame_count"] = 5
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    invalid = run_migration([root], apply=False, min_age_seconds=0)

    assert not invalid.ok
    assert invalid.manifest_records_checked == 1
    assert any("frame_count=5, parquet rows=4" in error["error"] for error in invalid.errors)


def test_cleanup_requires_every_file_to_be_canonical(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    first = root / "session/data/chunk-000/episode_000000.parquet"
    _write_legacy_parquet(first)

    initial = run_migration([root], apply=True, min_age_seconds=0)
    assert initial.ok
    assert initial.files_migrated == 1
    state_dir = root / DEFAULT_STATE_DIR_NAME
    assert state_dir.is_dir()

    second = root / "session/data/chunk-000/episode_000001.parquet"
    _write_legacy_parquet(second)
    second_hash = _sha256(second)
    with pytest.raises(MigrationError, match="not fully verified/canonical"):
        cleanup_migration_backups([root])
    assert state_dir.is_dir()
    assert _sha256(second) == second_hash

    remaining = run_migration([root], apply=True, min_age_seconds=0)
    assert remaining.ok
    assert remaining.files_migrated == 1
    cleanup = cleanup_migration_backups([root])
    assert cleanup["ok"] is True
    assert not state_dir.exists()
    assert audit_parquet_file(first).is_canonical
    assert audit_parquet_file(second).is_canonical


def test_run_migration_preflight_error_makes_zero_changes(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    valid = root / "a/data/chunk-000/episode_000000.parquet"
    corrupt = root / "b/data/chunk-000/episode_000000.parquet"
    _write_legacy_parquet(valid)
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"PAR1-corrupt-and-unreadable-PAR1")
    before_hashes = {path: _sha256(path) for path in (valid, corrupt)}
    before_schema = pq.read_schema(valid)

    report = run_migration([root], apply=True, min_age_seconds=0)

    assert not report.ok
    assert report.files_migrated == 0
    assert report.files_pending_after == report.files_pending_before == 1
    assert {path: _sha256(path) for path in (valid, corrupt)} == before_hashes
    assert pq.read_schema(valid).equals(before_schema)
    assert not (root / DEFAULT_STATE_DIR_NAME).exists()


def test_cleanup_rejects_canonical_current_values_that_diverge_from_backup(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    parquet_path = root / "session/data/chunk-000/episode_000000.parquet"
    _write_legacy_parquet(parquet_path)
    applied = run_migration([root], apply=True, min_age_seconds=0)
    assert applied.ok
    assert applied.files_migrated == 1
    assert len(applied.results) == 1
    backup_path = Path(applied.results[0].backup_path or "")
    assert backup_path.is_file()
    backup_hash = _sha256(backup_path)

    current = pq.read_table(parquet_path)
    action_index = current.schema.get_field_index("action")
    changed_actions = current.column(action_index).to_pylist()
    assert changed_actions[0] is not None
    changed_actions[0][0] = float(changed_actions[0][0]) + 10.0
    modified = current.set_column(
        action_index,
        current.schema.field(action_index),
        pa.array(changed_actions, type=current.schema.field(action_index).type),
    )
    pq.write_table(modified, parquet_path, compression="zstd")
    assert audit_parquet_file(parquet_path).is_canonical
    modified_hash = _sha256(parquet_path)

    with pytest.raises(
        MigrationError,
        match="backup cleanup|backup|diverg|changed since migration",
    ):
        cleanup_migration_backups([root])

    assert _sha256(parquet_path) == modified_hash
    assert backup_path.is_file()
    assert _sha256(backup_path) == backup_hash
    assert (root / DEFAULT_STATE_DIR_NAME).is_dir()


@pytest.mark.parametrize("unsafe_kind", ["dot", "parent", "absolute"])
def test_unsafe_state_dir_name_is_rejected_before_any_write(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    root = tmp_path / "datasets"
    parquet_path = root / "session/data/chunk-000/episode_000000.parquet"
    _write_legacy_parquet(parquet_path)
    outside = tmp_path / "outside-migration-state"
    state_dir_name = {
        "dot": ".",
        "parent": "..",
        "absolute": str(outside.resolve()),
    }[unsafe_kind]
    before_hash = _sha256(parquet_path)
    before_tree = sorted(path.relative_to(root) for path in root.rglob("*"))

    with pytest.raises(MigrationError, match="state_dir_name|state directory"):
        run_migration(
            [root],
            apply=True,
            min_age_seconds=0,
            state_dir_name=state_dir_name,
        )

    assert _sha256(parquet_path) == before_hash
    assert sorted(path.relative_to(root) for path in root.rglob("*")) == before_tree
    assert not outside.exists()


def test_direct_file_migration_rejects_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    target = root / "session/data/chunk-000/episode_000000.parquet"
    symlink = root / "session/data/chunk-000/episode_link.parquet"
    _write_legacy_parquet(target)
    symlink.symlink_to(target.name)
    target_hash = _sha256(target)
    symlink_target = os.readlink(symlink)

    with pytest.raises(MigrationError, match="symlink"):
        migrate_parquet_file(
            symlink,
            root=root,
            backup_root=tmp_path / "backups",
            min_age_seconds=0,
        )

    assert symlink.is_symlink()
    assert os.readlink(symlink) == symlink_target
    assert _sha256(target) == target_hash
    assert not (tmp_path / "backups").exists()


@pytest.mark.parametrize("kind", ["invalid_parquet", "float_overflow"])
def test_failed_migration_leaves_original_byte_for_byte_unchanged(
    tmp_path: Path,
    kind: str,
) -> None:
    root = tmp_path / "datasets"
    parquet_path = root / "session/data/chunk-000/episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True)
    if kind == "invalid_parquet":
        parquet_path.write_bytes(b"not a parquet file\x00\xff")
    else:
        table = pa.Table.from_arrays(
            [pa.array([[1.0e100]], type=pa.list_(pa.float64()))],
            names=["action"],
        )
        pq.write_table(table, parquet_path, compression="zstd")
    before_hash = _sha256(parquet_path)
    before_mode = stat.S_IMODE(parquet_path.stat().st_mode)

    try:
        migrate_parquet_file(
            parquet_path,
            root=root,
            backup_root=tmp_path / "backups",
            min_age_seconds=0,
        )
    except Exception:
        # The public API may either report a per-file failure or raise it; the
        # safety contract is the same in both cases.
        pass

    assert _sha256(parquet_path) == before_hash
    assert stat.S_IMODE(parquet_path.stat().st_mode) == before_mode
    assert not list(parquet_path.parent.glob("*.tmp"))
    if kind == "float_overflow":
        assert pq.read_schema(parquet_path).field("action").type == pa.list_(pa.float64())
