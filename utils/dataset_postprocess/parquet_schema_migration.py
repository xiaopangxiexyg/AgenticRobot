"""Safely normalize known DexTeleop Parquet fields to their Arrow contract.

The migration is deliberately narrower than a general Parquet rewrite:

* only fields explicitly declared by ``schema_v21`` are considered;
* absent fields are never added and unrelated inferred fields are untouched;
* dry-run is the default;
* each changed file is validated before and after an atomic replacement;
* the original inode is retained as a hard-link backup until an explicit
  cleanup command is run after a successful global verification.

This module is intended for repairing old files whose logical values are
correct but whose physical Arrow types were inferred as ``list<double>``,
``list<null>``, or scalar ``null``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import json
import os
import shutil
import stat
import sys
import tempfile
import time
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# When invoked as ``python path/to/parquet_schema_migration.py``, Python puts
# the script directory (not the repository root) on sys.path.  Keep the file
# usable both directly and through ``python -m`` without relying on PYTHONPATH.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:  # pragma: no cover - clearer CLI failure
    raise ModuleNotFoundError(
        "Parquet schema migration requires pyarrow to be installed."
    ) from exc

from dataset_contracts.parquet import (
    ARM_MODE_FLOAT64_COLUMNS,
    ARM_MODE_INT64_COLUMNS,
    ARM_MODE_STRING_COLUMNS,
    FLOAT32_LIST_COLUMNS,
    INT64_LIST_COLUMNS,
    STRING_LIST_COLUMNS,
)


MIGRATION_SCHEMA = "dexteleop_parquet_schema_v21_migration_v1"
DEFAULT_STATE_DIR_NAME = ".parquet_schema_v21_migration"
BACKUP_SUFFIX = ".before-schema-v21"
MAX_PARQUET_ROW_GROUP_ROWS = 64 * 1024 * 1024


class MigrationError(RuntimeError):
    """Raised when a file cannot be migrated without weakening guarantees."""


@dataclasses.dataclass(frozen=True)
class ColumnChange:
    name: str
    source_type: str
    target_type: str

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class FloatQuantization:
    value_count: int = 0
    rounded_value_count: int = 0
    max_absolute_error: float = 0.0
    max_relative_error: float = 0.0

    def merge(self, other: "FloatQuantization") -> "FloatQuantization":
        return FloatQuantization(
            value_count=self.value_count + other.value_count,
            rounded_value_count=self.rounded_value_count + other.rounded_value_count,
            max_absolute_error=max(self.max_absolute_error, other.max_absolute_error),
            max_relative_error=max(self.max_relative_error, other.max_relative_error),
        )

    def to_dict(self) -> dict[str, int | float]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ParquetAudit:
    path: str
    num_rows: int
    row_group_rows: tuple[int, ...]
    compression: str
    changes: tuple[ColumnChange, ...]
    quantization: dict[str, FloatQuantization]

    @property
    def is_canonical(self) -> bool:
        return not self.changes

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "num_rows": self.num_rows,
            "row_group_rows": list(self.row_group_rows),
            "compression": self.compression,
            "is_canonical": self.is_canonical,
            "changes": [change.to_dict() for change in self.changes],
            "quantization": {
                name: stats.to_dict() for name, stats in sorted(self.quantization.items())
            },
        }


@dataclasses.dataclass(frozen=True)
class FileMigrationResult:
    path: str
    status: str
    num_rows: int
    changes: tuple[ColumnChange, ...] = ()
    backup_path: str | None = None
    old_size_bytes: int | None = None
    new_size_bytes: int | None = None
    old_mtime_ns: int | None = None
    new_mtime_ns: int | None = None
    quantization: dict[str, FloatQuantization] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["changes"] = [change.to_dict() for change in self.changes]
        payload["quantization"] = {
            name: stats.to_dict() for name, stats in sorted(self.quantization.items())
        }
        return payload


@dataclasses.dataclass
class MigrationReport:
    roots: tuple[str, ...]
    apply: bool
    started_at_unix: float
    finished_at_unix: float = 0.0
    files_seen: int = 0
    files_ignored_non_episode: int = 0
    rows_seen: int = 0
    files_pending_before: int = 0
    files_selected: int = 0
    files_migrated: int = 0
    files_already_canonical: int = 0
    files_pending_after: int = 0
    manifest_records_checked: int = 0
    backup_roots: tuple[str, ...] = ()
    errors: list[dict[str, str]] = dataclasses.field(default_factory=list)
    results: list[FileMigrationResult] = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        quantization = FloatQuantization()
        column_change_counts: dict[str, int] = {}
        old_size_bytes = 0
        new_size_bytes = 0
        for result in self.results:
            for change in result.changes:
                column_change_counts[change.name] = column_change_counts.get(change.name, 0) + 1
            for stats in result.quantization.values():
                quantization = quantization.merge(stats)
            old_size_bytes += int(result.old_size_bytes or 0)
            new_size_bytes += int(result.new_size_bytes or 0)
        return {
            "migration_schema": MIGRATION_SCHEMA,
            "roots": list(self.roots),
            "apply": self.apply,
            "ok": self.ok,
            "started_at_unix": self.started_at_unix,
            "finished_at_unix": self.finished_at_unix,
            "files_seen": self.files_seen,
            "files_ignored_non_episode": self.files_ignored_non_episode,
            "rows_seen": self.rows_seen,
            "files_pending_before": self.files_pending_before,
            "files_selected": self.files_selected,
            "files_migrated": self.files_migrated,
            "files_already_canonical": self.files_already_canonical,
            "files_pending_after": self.files_pending_after,
            "manifest_records_checked": self.manifest_records_checked,
            "backup_roots": list(self.backup_roots),
            "errors": list(self.errors),
            "column_change_counts": dict(sorted(column_change_counts.items())),
            "quantization_summary": quantization.to_dict(),
            "migrated_old_size_bytes": old_size_bytes,
            "migrated_new_size_bytes": new_size_bytes,
            "results": [result.to_dict() for result in self.results],
        }


@dataclasses.dataclass(frozen=True)
class _StatSignature:
    device: int
    inode: int
    size: int
    mtime_ns: int
    mode: int
    uid: int
    gid: int
    nlink: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_StatSignature":
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            size=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            mode=int(value.st_mode),
            uid=int(value.st_uid),
            gid=int(value.st_gid),
            nlink=int(value.st_nlink),
        )

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


def _field_with_type(field: pa.Field, data_type: pa.DataType) -> pa.Field:
    return pa.field(
        field.name,
        data_type,
        nullable=field.nullable,
        metadata=field.metadata,
    )


def _list_type_preserving_child(
    source_type: pa.DataType,
    child_type: pa.DataType,
) -> pa.ListType:
    if pa.types.is_list(source_type):
        source_child = source_type.value_field
        child_field = pa.field(
            source_child.name,
            child_type,
            nullable=source_child.nullable,
            metadata=source_child.metadata,
        )
        return pa.list_(child_field)
    # PyArrow's compliant Parquet LIST encoding names the physical/logical
    # child ``element`` on read-back.  A scalar-null legacy field has no child
    # name to preserve, so choose that canonical name up front.
    return pa.list_(pa.field("element", child_type))


def _canonical_type(name: str, source_type: pa.DataType) -> pa.DataType | None:
    if name in FLOAT32_LIST_COLUMNS:
        return _list_type_preserving_child(source_type, pa.float32())
    if name in STRING_LIST_COLUMNS:
        return _list_type_preserving_child(source_type, pa.string())
    if name in INT64_LIST_COLUMNS:
        return _list_type_preserving_child(source_type, pa.int64())
    if name in ARM_MODE_INT64_COLUMNS:
        return pa.int64()
    if name in ARM_MODE_STRING_COLUMNS:
        return pa.string()
    if name in ARM_MODE_FLOAT64_COLUMNS:
        return pa.float64()
    return None


def _type_is_exact(source: pa.DataType, target: pa.DataType) -> bool:
    if pa.types.is_list(source) and pa.types.is_list(target):
        source_child = source.value_field
        target_child = target.value_field
        return (
            source_child.name == target_child.name
            and source_child.nullable == target_child.nullable
            and source_child.metadata == target_child.metadata
            and _type_is_exact(source_child.type, target_child.type)
        )
    return source == target


def _ensure_supported_cast(name: str, source: pa.DataType, target: pa.DataType) -> None:
    if pa.types.is_list(target):
        if pa.types.is_null(source):
            return
        if not pa.types.is_list(source):
            raise MigrationError(
                f"{name}: unsupported canonical vector source type {source}; expected list or null"
            )
        child = source.value_type
        target_child = target.value_type
        if pa.types.is_float32(target_child):
            valid = pa.types.is_floating(child) or pa.types.is_null(child)
        elif pa.types.is_string(target_child):
            valid = (
                pa.types.is_string(child)
                or pa.types.is_large_string(child)
                or pa.types.is_null(child)
            )
        elif pa.types.is_int64(target_child):
            valid = pa.types.is_integer(child) or pa.types.is_null(child)
        else:  # pragma: no cover - guarded by the canonical mapping
            valid = False
        if not valid:
            raise MigrationError(
                f"{name}: unsupported list child cast {child} -> {target_child}"
            )
        return

    if pa.types.is_int64(target):
        if not (
            pa.types.is_null(source)
            or pa.types.is_integer(source)
            or pa.types.is_boolean(source)
        ):
            raise MigrationError(f"{name}: unsupported scalar cast {source} -> int64")
        return

    if pa.types.is_string(target):
        if not (
            pa.types.is_null(source)
            or pa.types.is_string(source)
            or pa.types.is_large_string(source)
        ):
            raise MigrationError(f"{name}: unsupported scalar cast {source} -> string")
        return

    raise MigrationError(f"{name}: unsupported target type {target}")


def _target_schema(source: pa.Schema) -> tuple[pa.Schema, tuple[ColumnChange, ...]]:
    fields: list[pa.Field] = []
    changes: list[ColumnChange] = []
    for field in source:
        target_type = _canonical_type(field.name, field.type)
        if target_type is None or _type_is_exact(field.type, target_type):
            fields.append(field)
            continue
        _ensure_supported_cast(field.name, field.type, target_type)
        fields.append(_field_with_type(field, target_type))
        changes.append(
            ColumnChange(
                name=field.name,
                source_type=str(field.type),
                target_type=str(target_type),
            )
        )
    return pa.schema(fields, metadata=source.metadata), tuple(changes)


def _compression_for_file(parquet_file: pq.ParquetFile) -> str:
    codecs: set[str] = set()
    metadata = parquet_file.metadata
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            codecs.add(str(row_group.column(column_index).compression).upper())
    if not codecs:
        # An empty file has no physical column chunk from which to recover the
        # codec.  DexTeleop does not produce these; use the writer default only
        # for completeness.
        return "zstd"
    if len(codecs) != 1:
        raise MigrationError(
            f"mixed Parquet compression is unsupported for loss-minimizing rewrite: {sorted(codecs)}"
        )
    return next(iter(codecs)).lower()


def _flatten_list_values(array: pa.Array | pa.ChunkedArray) -> pa.Array:
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if not pa.types.is_list(array.type):
        raise MigrationError(f"expected a list array, got {array.type}")
    return pc.list_flatten(array)


def _float32_quantization(array: pa.Array | pa.ChunkedArray, *, name: str) -> FloatQuantization:
    flat = _flatten_list_values(array)
    if pa.types.is_null(flat.type):
        if len(flat):
            raise MigrationError(
                f"{name}: list<null> contains {len(flat)} child entries; refusing to guess values"
            )
        return FloatQuantization()
    if not pa.types.is_floating(flat.type):
        return FloatQuantization()

    valid = pc.drop_null(flat)
    if not len(valid):
        return FloatQuantization()
    source = np.asarray(valid.to_numpy(zero_copy_only=False), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        converted = source.astype(np.float32)
    finite_source = np.isfinite(source)
    overflow = finite_source & ~np.isfinite(converted)
    if bool(np.any(overflow)):
        example = float(source[np.flatnonzero(overflow)[0]])
        raise MigrationError(
            f"{name}: finite value {example!r} overflows float32; original left untouched"
        )

    converted64 = converted.astype(np.float64)
    error = np.abs(converted64[finite_source] - source[finite_source])
    finite_values = source[finite_source]
    if error.size:
        rounded = int(np.count_nonzero(error))
        max_absolute = float(np.max(error))
        denominators = np.abs(finite_values)
        relative = np.divide(
            error,
            denominators,
            out=np.zeros_like(error),
            where=denominators != 0.0,
        )
        max_relative = float(np.max(relative)) if relative.size else 0.0
    else:
        rounded = 0
        max_absolute = 0.0
        max_relative = 0.0
    return FloatQuantization(
        value_count=int(len(source)),
        rounded_value_count=rounded,
        max_absolute_error=max_absolute,
        max_relative_error=max_relative,
    )


def _validate_null_list_children(array: pa.Array | pa.ChunkedArray, *, name: str) -> None:
    flat = _flatten_list_values(array)
    if pa.types.is_null(flat.type) and len(flat):
        raise MigrationError(
            f"{name}: list<null> contains {len(flat)} child entries; refusing to guess values"
        )


def _audit_values(
    parquet_file: pq.ParquetFile,
    source_schema: pa.Schema,
    changes: Sequence[ColumnChange],
) -> dict[str, FloatQuantization]:
    by_name = {change.name: change for change in changes}
    quantization: dict[str, FloatQuantization] = {}
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        if not by_name:
            break
        table = parquet_file.read_row_group(row_group_index, columns=list(by_name))
        for name in by_name:
            source_type = source_schema.field(name).type
            if pa.types.is_list(source_type):
                if pa.types.is_floating(source_type.value_type):
                    stats = _float32_quantization(table.column(name), name=name)
                    quantization[name] = quantization.get(name, FloatQuantization()).merge(stats)
                elif pa.types.is_null(source_type.value_type):
                    _validate_null_list_children(table.column(name), name=name)
    return quantization


def audit_parquet_file(path: str | Path) -> ParquetAudit:
    """Deep-audit one file and return the exact canonical casts it needs."""

    parquet_path = Path(path)
    file_stat = parquet_path.lstat()
    if not stat.S_ISREG(file_stat.st_mode) or parquet_path.is_symlink():
        raise MigrationError(f"not a regular non-symlink file: {parquet_path}")

    parquet_file = pq.ParquetFile(parquet_path)
    target_schema, changes = _target_schema(parquet_file.schema_arrow)
    del target_schema
    row_group_rows = tuple(
        int(parquet_file.metadata.row_group(index).num_rows)
        for index in range(parquet_file.metadata.num_row_groups)
    )
    if any(rows < 0 or rows > MAX_PARQUET_ROW_GROUP_ROWS for rows in row_group_rows):
        raise MigrationError(
            f"{parquet_path}: row group exceeds safe one-to-one rewrite limit"
        )
    quantization = _audit_values(parquet_file, parquet_file.schema_arrow, changes)
    return ParquetAudit(
        path=str(parquet_path),
        num_rows=int(parquet_file.metadata.num_rows),
        row_group_rows=row_group_rows,
        compression=_compression_for_file(parquet_file),
        changes=changes,
        quantization=quantization,
    )


def _cast_table(table: pa.Table, target_schema: pa.Schema) -> pa.Table:
    arrays: list[pa.ChunkedArray] = []
    for index, target_field in enumerate(target_schema):
        source_field = table.schema.field(index)
        column = table.column(index)
        if _type_is_exact(source_field.type, target_field.type):
            arrays.append(column)
        else:
            if (
                pa.types.is_list(source_field.type)
                and pa.types.is_floating(source_field.type.value_type)
                and pa.types.is_list(target_field.type)
                and pa.types.is_float32(target_field.type.value_type)
            ):
                _float32_quantization(column, name=source_field.name)
            arrays.append(column.cast(target_field.type, safe=True))
    return pa.Table.from_arrays(arrays, schema=target_schema)


def _assert_field_exact(expected: pa.Field, actual: pa.Field, *, location: str) -> None:
    if (
        expected.name != actual.name
        or expected.nullable != actual.nullable
        or expected.metadata != actual.metadata
    ):
        raise MigrationError(f"field metadata changed at {location}: {expected} != {actual}")
    if pa.types.is_list(expected.type) and pa.types.is_list(actual.type):
        _assert_field_exact(
            expected.type.value_field,
            actual.type.value_field,
            location=f"{location}.list_child",
        )
    elif expected.type != actual.type:
        raise MigrationError(
            f"field type changed unexpectedly at {location}: {expected.type} != {actual.type}"
        )


def _assert_schema_exact(expected: pa.Schema, actual: pa.Schema) -> None:
    if len(expected) != len(actual) or expected.metadata != actual.metadata:
        raise MigrationError("schema field count or schema metadata changed")
    for index, (expected_field, actual_field) in enumerate(zip(expected, actual)):
        _assert_field_exact(expected_field, actual_field, location=f"field[{index}]")


def _validity_bits(array: pa.Array) -> np.ndarray:
    return np.asarray(array.is_valid().to_numpy(zero_copy_only=False), dtype=np.bool_)


def _assert_array_exact(expected: pa.Array, actual: pa.Array, *, location: str) -> None:
    if len(expected) != len(actual):
        raise MigrationError(f"array length changed at {location}")
    if expected.type != actual.type:
        raise MigrationError(f"array type changed at {location}: {expected.type} != {actual.type}")

    expected_valid = _validity_bits(expected)
    actual_valid = _validity_bits(actual)
    if not np.array_equal(expected_valid, actual_valid):
        raise MigrationError(f"null bitmap changed at {location}")
    if pa.types.is_null(expected.type) or not len(expected):
        return

    if pa.types.is_list(expected.type) or pa.types.is_large_list(expected.type):
        expected_offsets = np.asarray(expected.offsets.to_numpy(zero_copy_only=False))
        actual_offsets = np.asarray(actual.offsets.to_numpy(zero_copy_only=False))
        if not np.array_equal(expected_offsets, actual_offsets):
            raise MigrationError(f"list offsets changed at {location}")
        _assert_array_exact(
            expected.values,
            actual.values,
            location=f"{location}.values",
        )
        return

    if pa.types.is_fixed_size_list(expected.type):
        _assert_array_exact(
            expected.values,
            actual.values,
            location=f"{location}.values",
        )
        return

    if pa.types.is_struct(expected.type):
        for index in range(expected.type.num_fields):
            _assert_array_exact(
                expected.field(index),
                actual.field(index),
                location=f"{location}.{expected.type[index].name}",
            )
        return

    if pa.types.is_floating(expected.type):
        if not bool(np.any(expected_valid)):
            return
        expected_values = np.asarray(expected.to_numpy(zero_copy_only=False))[expected_valid]
        actual_values = np.asarray(actual.to_numpy(zero_copy_only=False))[actual_valid]
        unsigned_type = {
            2: np.uint16,
            4: np.uint32,
            8: np.uint64,
        }.get(expected_values.dtype.itemsize)
        if unsigned_type is None or not np.array_equal(
            expected_values.view(unsigned_type), actual_values.view(unsigned_type)
        ):
            raise MigrationError(f"floating-point bits changed at {location}")
        return

    if not expected.equals(actual):
        raise MigrationError(f"logical values changed at {location}")


def _assert_tables_exact(expected: pa.Table, actual: pa.Table, *, row_group: int) -> None:
    _assert_schema_exact(expected.schema, actual.schema)
    if expected.num_rows != actual.num_rows or expected.num_columns != actual.num_columns:
        raise MigrationError(f"table shape changed in row group {row_group}")
    for index, name in enumerate(expected.column_names):
        _assert_array_exact(
            expected.column(index).combine_chunks(),
            actual.column(index).combine_chunks(),
            location=f"row_group[{row_group}].{name}",
        )


def _assert_parquet_layout(
    source: pq.ParquetFile,
    rewritten: pq.ParquetFile,
    *,
    target_schema: pa.Schema,
    compression: str,
) -> None:
    _assert_schema_exact(target_schema, rewritten.schema_arrow)
    if source.metadata.num_rows != rewritten.metadata.num_rows:
        raise MigrationError("Parquet row count changed")
    if source.metadata.num_row_groups != rewritten.metadata.num_row_groups:
        raise MigrationError("Parquet row-group count changed")
    for index in range(source.metadata.num_row_groups):
        source_rows = int(source.metadata.row_group(index).num_rows)
        rewritten_rows = int(rewritten.metadata.row_group(index).num_rows)
        if source_rows != rewritten_rows:
            raise MigrationError(f"row-group {index} row count changed")
        row_group = rewritten.metadata.row_group(index)
        for column_index in range(row_group.num_columns):
            actual_codec = str(row_group.column(column_index).compression).lower()
            if actual_codec != compression.lower():
                raise MigrationError(
                    f"compression changed at row group {index}, column {column_index}: "
                    f"{actual_codec} != {compression}"
                )


def _validate_rewrite(source_path: Path, rewritten_path: Path) -> None:
    source = pq.ParquetFile(source_path)
    target_schema, changes = _target_schema(source.schema_arrow)
    if not changes:
        raise MigrationError(f"validation source is already canonical: {source_path}")
    compression = _compression_for_file(source)
    rewritten = pq.ParquetFile(rewritten_path)
    _assert_parquet_layout(
        source,
        rewritten,
        target_schema=target_schema,
        compression=compression,
    )
    for row_group_index in range(source.metadata.num_row_groups):
        expected = _cast_table(source.read_row_group(row_group_index), target_schema)
        actual = rewritten.read_row_group(row_group_index)
        _assert_tables_exact(expected, actual, row_group=row_group_index)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _source_signature(path: Path) -> _StatSignature:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or path.is_symlink():
        raise MigrationError(f"source stopped being a regular non-symlink file: {path}")
    return _StatSignature.from_stat(value)


def _require_unchanged(path: Path, expected: _StatSignature) -> None:
    actual = _source_signature(path)
    if actual != expected:
        raise MigrationError(
            f"source changed concurrently before replacement: {path}; "
            f"expected={expected.to_dict()}, actual={actual.to_dict()}"
        )


def _backup_path(source: Path, *, root: Path, backup_root: Path) -> Path:
    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise MigrationError(f"source {source} is outside declared root {root}") from exc
    return backup_root / relative.parent / f"{relative.name}{BACKUP_SUFFIX}"


def _append_journal(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _write_replacement(
    source_path: Path,
    temp_handle: Any,
    *,
    source_mode: int,
) -> None:
    source = pq.ParquetFile(source_path)
    target_schema, changes = _target_schema(source.schema_arrow)
    if not changes:
        raise MigrationError(f"source became canonical before rewrite: {source_path}")
    compression = _compression_for_file(source)
    version = str(source.metadata.format_version)
    with pq.ParquetWriter(
        temp_handle,
        target_schema,
        version=version,
        compression=compression,
        store_schema=True,
        use_compliant_nested_type=True,
    ) as writer:
        for row_group_index in range(source.metadata.num_row_groups):
            source_rows = int(source.metadata.row_group(row_group_index).num_rows)
            if source_rows <= 0:
                raise MigrationError(
                    f"zero-row row group {row_group_index} cannot be preserved reliably"
                )
            if source_rows > MAX_PARQUET_ROW_GROUP_ROWS:
                raise MigrationError(
                    f"row group {row_group_index} exceeds PyArrow one-to-one rewrite limit"
                )
            converted = _cast_table(source.read_row_group(row_group_index), target_schema)
            writer.write_table(converted, row_group_size=source_rows)
    temp_handle.flush()
    os.fchmod(temp_handle.fileno(), source_mode)
    os.fsync(temp_handle.fileno())


def _restore_backup(backup: Path, source: Path, *, remove_backup: bool = True) -> None:
    rollback_path: Path | None = None
    try:
        for _ in range(10):
            candidate = source.parent / (
                f".{source.name}.schema-v21-rollback-{uuid.uuid4().hex}.tmp"
            )
            try:
                os.link(backup, candidate)
                rollback_path = candidate
                break
            except FileExistsError:  # practically impossible, but never reuse a stale path
                continue
        if rollback_path is None:
            raise MigrationError(f"could not allocate a unique rollback link for {source}")
        os.replace(rollback_path, source)
        rollback_path = None
        _fsync_directory(source.parent)
        restored = _source_signature(source)
        backup_stat = _source_signature(backup)
        if restored.device != backup_stat.device or restored.inode != backup_stat.inode:
            raise MigrationError(f"rollback inode verification failed for {source}")
        if remove_backup:
            backup.unlink()
            _fsync_directory(backup.parent)
    finally:
        if rollback_path is not None:
            with contextlib.suppress(FileNotFoundError):
                rollback_path.unlink()


def _remove_redundant_backup_if_source_is_original(
    backup: Path,
    source: Path,
    original: _StatSignature,
) -> bool:
    """Remove a pre-replacement hard link without discarding the only old copy."""

    if not backup.exists() or not source.exists():
        return False
    source_stat = _source_signature(source)
    backup_stat = _source_signature(backup)
    if (
        source_stat.device != original.device
        or source_stat.inode != original.inode
        or backup_stat.device != original.device
        or backup_stat.inode != original.inode
    ):
        return False
    backup.unlink()
    _fsync_directory(backup.parent)
    return True


def _recover_original_after_failure(
    *,
    backup: Path,
    source: Path,
    original: _StatSignature,
) -> None:
    """Infer on-disk transaction state and restore the original if necessary."""

    backup_exists = backup.exists() and not backup.is_symlink()
    source_exists = source.exists() and not source.is_symlink()
    if source_exists:
        source_stat = _source_signature(source)
        source_is_original = (
            source_stat.device == original.device and source_stat.inode == original.inode
        )
    else:
        source_is_original = False

    if backup_exists:
        backup_stat = _source_signature(backup)
        backup_is_original = (
            backup_stat.device == original.device and backup_stat.inode == original.inode
        )
        if not backup_is_original:
            raise MigrationError(f"backup inode does not match the original: {backup}")
        if source_is_original:
            _remove_redundant_backup_if_source_is_original(backup, source, original)
        else:
            _restore_backup(backup, source)
        return

    if not source_is_original:
        raise MigrationError(
            f"source no longer references the original inode and no backup exists: {source}"
        )


def migrate_parquet_file(
    path: str | Path,
    *,
    root: str | Path,
    backup_root: str | Path,
    min_age_seconds: float = 300.0,
    journal_path: str | Path | None = None,
) -> FileMigrationResult:
    """Transactionally rewrite one old Parquet file in place.

    The caller is responsible for excluding concurrent writers at the dataset
    level.  This function additionally verifies the source stat immediately
    before replacement and refuses symlinks/shared hard links.
    """

    input_path = Path(path).expanduser()
    input_stat = input_path.lstat()
    if stat.S_ISLNK(input_stat.st_mode):
        raise MigrationError(f"refusing symlink source path: {input_path}")
    source_path = input_path.resolve(strict=True)
    root_path = Path(root).resolve(strict=True)
    backup_root_path = Path(backup_root).resolve(strict=False)
    journal = None if journal_path is None else Path(journal_path)
    try:
        source_path.relative_to(root_path)
    except ValueError as exc:
        raise MigrationError(f"source {source_path} is outside root {root_path}") from exc

    initial = _source_signature(source_path)
    if initial.nlink != 1:
        raise MigrationError(
            f"refusing source with {initial.nlink} hard links before migration: {source_path}"
        )
    if initial.uid != os.geteuid():
        raise MigrationError(
            f"source owner uid {initial.uid} differs from process uid {os.geteuid()}; "
            "ownership could not be preserved"
        )
    age_seconds = (time.time_ns() - initial.mtime_ns) / 1_000_000_000.0
    if age_seconds < float(min_age_seconds):
        raise MigrationError(
            f"source is only {age_seconds:.1f}s old; minimum is {float(min_age_seconds):.1f}s"
        )

    audit = audit_parquet_file(source_path)
    if audit.is_canonical:
        return FileMigrationResult(
            path=str(source_path),
            status="already_canonical",
            num_rows=audit.num_rows,
        )

    backup_path = _backup_path(
        source_path,
        root=root_path,
        backup_root=backup_root_path,
    )
    if backup_path.exists() or backup_path.is_symlink():
        raise MigrationError(
            f"backup already exists for a still-legacy source; inspect before resuming: {backup_path}"
        )

    source_mode = stat.S_IMODE(initial.mode)
    temp_path: Path | None = None
    backup_created = False
    replaced = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{source_path.name}.schema-v21-",
            suffix=".tmp",
            dir=source_path.parent,
            delete=False,
        ) as temp_handle:
            temp_path = Path(temp_handle.name)
            _write_replacement(
                source_path,
                temp_handle,
                source_mode=source_mode,
            )

        _validate_rewrite(source_path, temp_path)
        _require_unchanged(source_path, initial)

        backup_path.parent.mkdir(parents=True, exist_ok=True)
        os.link(source_path, backup_path)
        backup_created = True
        _fsync_directory(backup_path.parent)

        after_link = _source_signature(source_path)
        if (
            after_link.device != initial.device
            or after_link.inode != initial.inode
            or after_link.size != initial.size
            or after_link.mtime_ns != initial.mtime_ns
            or after_link.mode != initial.mode
            or after_link.uid != initial.uid
            or after_link.gid != initial.gid
            or after_link.nlink < initial.nlink + 1
        ):
            raise MigrationError(f"source changed while creating backup: {source_path}")

        os.replace(temp_path, source_path)
        replaced = True
        temp_path = None
        _fsync_directory(source_path.parent)

        try:
            _validate_rewrite(backup_path, source_path)
        except BaseException:
            _restore_backup(backup_path, source_path)
            replaced = False
            raise

        final = _source_signature(source_path)
        if stat.S_IMODE(final.mode) != source_mode:
            _restore_backup(backup_path, source_path)
            replaced = False
            raise MigrationError(f"file permissions changed for {source_path}")
        if final.uid != initial.uid or final.gid != initial.gid:
            _restore_backup(backup_path, source_path)
            replaced = False
            raise MigrationError(f"file ownership changed for {source_path}")

        result = FileMigrationResult(
            path=str(source_path),
            status="migrated",
            num_rows=audit.num_rows,
            changes=audit.changes,
            backup_path=str(backup_path),
            old_size_bytes=initial.size,
            new_size_bytes=final.size,
            old_mtime_ns=initial.mtime_ns,
            new_mtime_ns=final.mtime_ns,
            quantization=audit.quantization,
        )
        _append_journal(
            journal,
            {
                "migration_schema": MIGRATION_SCHEMA,
                "event": "migrated",
                "time_unix": time.time(),
                "source_before": initial.to_dict(),
                "source_after": final.to_dict(),
                "result": result.to_dict(),
            },
        )
        return result
    except BaseException as exc:
        rollback_error: str | None = None
        try:
            _recover_original_after_failure(
                backup=backup_path,
                source=source_path,
                original=initial,
            )
            replaced = False
        except BaseException as restore_exc:  # keep the backup on any ambiguity
            rollback_error = f"{type(restore_exc).__name__}: {restore_exc}"
        with contextlib.suppress(BaseException):
            _append_journal(
                journal,
                {
                    "migration_schema": MIGRATION_SCHEMA,
                    "event": "failed",
                    "time_unix": time.time(),
                    "path": str(source_path),
                    "backup_created": backup_created,
                    "replacement_active": replaced,
                    "rollback_error": rollback_error,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        if rollback_error is not None:
            raise MigrationError(
                f"migration failed ({exc}); rollback also failed ({rollback_error}); "
                f"retain and inspect {backup_path}"
            ) from exc
        raise
    finally:
        if temp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                temp_path.unlink()


def _discover_parquet(roots: Sequence[Path]) -> tuple[Path, ...]:
    discovered: list[Path] = []
    for root in roots:
        for path in root.rglob("episode_*.parquet"):
            if DEFAULT_STATE_DIR_NAME in path.parts:
                continue
            if path.parent.parent.name != "data" or not path.parent.name.startswith("chunk-"):
                continue
            discovered.append(path)
    return tuple(sorted(discovered, key=lambda item: str(item)))


def _count_ignored_parquet(roots: Sequence[Path], selected: Sequence[Path]) -> int:
    selected_text = {str(path) for path in selected}
    count = 0
    for root in roots:
        for path in root.rglob("*.parquet"):
            if DEFAULT_STATE_DIR_NAME in path.parts:
                continue
            if str(path) not in selected_text:
                count += 1
    return count


def _normalize_roots(roots: Iterable[str | Path]) -> tuple[Path, ...]:
    normalized = tuple(Path(root).expanduser().resolve(strict=True) for root in roots)
    if not normalized:
        raise MigrationError("at least one --root is required")
    if len(set(normalized)) != len(normalized):
        raise MigrationError("duplicate dataset roots are not allowed")
    for root in normalized:
        if not root.is_dir():
            raise MigrationError(f"dataset root is not a directory: {root}")
    for index, left in enumerate(normalized):
        for right in normalized[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise MigrationError(f"overlapping roots are not allowed: {left}, {right}")
    return normalized


def _validate_state_dir_name(value: str) -> str:
    name = str(value).strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise MigrationError(f"state_dir_name must be one safe path component: {value!r}")
    return name


def _preflight_apply_path(
    path: Path,
    *,
    root: Path,
    state_dir_name: str,
    min_age_seconds: float,
) -> None:
    signature = _source_signature(path)
    if signature.nlink != 1:
        raise MigrationError(f"source has {signature.nlink} hard links: {path}")
    if signature.uid != os.geteuid():
        raise MigrationError(
            f"source owner uid {signature.uid} differs from process uid {os.geteuid()}: {path}"
        )
    age_seconds = (time.time_ns() - signature.mtime_ns) / 1_000_000_000.0
    if age_seconds < float(min_age_seconds):
        raise MigrationError(
            f"source is only {age_seconds:.1f}s old; minimum is {float(min_age_seconds):.1f}s"
        )
    backup = _backup_path(
        path.resolve(strict=True),
        root=root,
        backup_root=root / state_dir_name / "backups",
    )
    if backup.exists() or backup.is_symlink():
        raise MigrationError(f"backup already exists for a legacy source: {backup}")


def _verify_episode_manifests(roots: Sequence[Path]) -> tuple[int, list[dict[str, str]]]:
    checked = 0
    errors: list[dict[str, str]] = []
    for root in roots:
        for manifest in sorted(root.rglob("meta/episodes.jsonl")):
            dataset_root = manifest.parent.parent
            try:
                lines = manifest.read_text(encoding="utf-8").splitlines()
            except Exception as exc:
                errors.append({"path": str(manifest), "error": f"cannot read manifest: {exc}"})
                continue
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                checked += 1
                try:
                    record = json.loads(line)
                    relative = Path(str(record["parquet_path"]))
                    expected_rows = int(record["frame_count"])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"unsafe parquet_path {relative}")
                    parquet_path = dataset_root / relative
                    actual_rows = int(pq.ParquetFile(parquet_path).metadata.num_rows)
                    if actual_rows != expected_rows:
                        raise ValueError(
                            f"frame_count={expected_rows}, parquet rows={actual_rows}"
                        )
                except Exception as exc:
                    errors.append(
                        {
                            "path": f"{manifest}:{line_number}",
                            "error": f"manifest integrity check failed: {exc}",
                        }
                    )
    return checked, errors


class _RootLock:
    def __init__(
        self,
        root: Path,
        state_dir_name: str,
        *,
        create_state: bool = True,
    ) -> None:
        self.root = root
        self.state_dir = root / state_dir_name
        self.create_state = bool(create_state)
        self.state_exists = False
        # Keep the lock outside the removable state directory so cleanup never
        # unlinks the inode whose flock is protecting the operation.
        self.path = root / f"{state_dir_name}.lock"
        self.handle: Any | None = None

    def __enter__(self) -> "_RootLock":
        if self.path.is_symlink():
            raise MigrationError(f"refusing symlink migration lock {self.path}")
        open_flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, open_flags, 0o600)
        self.handle = os.fdopen(descriptor, "a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise MigrationError(f"another migration holds {self.path}") from exc
        try:
            if self.state_dir.is_symlink():
                raise MigrationError(
                    f"refusing symlink migration state directory {self.state_dir}"
                )
            if self.state_dir.exists() and not self.state_dir.is_dir():
                raise MigrationError(f"migration state is not a directory: {self.state_dir}")
            state_preexisted = self.state_dir.exists()
            if not state_preexisted and not self.create_state:
                self.state_exists = False
                return self
            if not state_preexisted:
                self.state_dir.mkdir(parents=False, exist_ok=False)
                _fsync_directory(self.root)
            self.state_exists = True
            marker = self.state_dir / "marker.json"
            if marker.is_symlink():
                raise MigrationError(f"refusing symlink migration marker {marker}")
            if marker.exists():
                payload = json.loads(marker.read_text(encoding="utf-8"))
                if payload.get("migration_schema") != MIGRATION_SCHEMA:
                    raise MigrationError(f"foreign state directory: {self.state_dir}")
            else:
                if state_preexisted:
                    raise MigrationError(
                        f"pre-existing state directory lacks this migration's marker: "
                        f"{self.state_dir}"
                    )
                marker.write_text(
                    json.dumps(
                        {"migration_schema": MIGRATION_SCHEMA, "created_at_unix": time.time()},
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with marker.open("rb") as marker_handle:
                    os.fsync(marker_handle.fileno())
                _fsync_directory(self.state_dir)
        except BaseException:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def _snapshot(paths: Sequence[Path]) -> dict[str, tuple[int, int]]:
    return {
        str(path): (int(path.stat().st_size), int(pq.ParquetFile(path).metadata.num_rows))
        for path in paths
    }


def run_migration(
    roots: Iterable[str | Path],
    *,
    apply: bool = False,
    max_files: int | None = None,
    min_age_seconds: float = 300.0,
    state_dir_name: str = DEFAULT_STATE_DIR_NAME,
) -> MigrationReport:
    """Audit or migrate all Parquet files below one or more roots.

    Dry-run performs no writes, including lock/state files.  Apply first audits
    every source and every episode manifest; any preflight error aborts before
    the first replacement.
    """

    root_paths = _normalize_roots(roots)
    state_dir_name = _validate_state_dir_name(state_dir_name)
    report = MigrationReport(
        roots=tuple(str(root) for root in root_paths),
        apply=bool(apply),
        started_at_unix=time.time(),
    )
    paths = _discover_parquet(root_paths)
    report.files_seen = len(paths)
    report.files_ignored_non_episode = _count_ignored_parquet(root_paths, paths)

    audits: dict[Path, ParquetAudit] = {}
    for path in paths:
        try:
            audit = audit_parquet_file(path)
            audits[path] = audit
            report.rows_seen += audit.num_rows
        except Exception as exc:
            report.errors.append(
                {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )

    pending = [path for path in paths if path in audits and not audits[path].is_canonical]
    report.files_pending_before = len(pending)
    report.files_already_canonical = len(audits) - len(pending)
    checked, manifest_errors = _verify_episode_manifests(root_paths)
    report.manifest_records_checked = checked
    report.errors.extend(manifest_errors)

    if max_files is not None:
        if int(max_files) <= 0:
            raise MigrationError("max_files must be positive")
        selected = pending[: int(max_files)]
    else:
        selected = pending
    report.files_selected = len(selected)

    if not apply or report.errors or not selected:
        report.files_pending_after = report.files_pending_before
        report.finished_at_unix = time.time()
        return report

    for path in selected:
        owning_root = next(root for root in root_paths if path == root or root in path.parents)
        try:
            _preflight_apply_path(
                path,
                root=owning_root,
                state_dir_name=state_dir_name,
                min_age_seconds=min_age_seconds,
            )
        except Exception as exc:
            report.errors.append(
                {"path": str(path), "error": f"preflight failed: {type(exc).__name__}: {exc}"}
            )
    if report.errors:
        report.files_pending_after = report.files_pending_before
        report.finished_at_unix = time.time()
        return report

    before_snapshot = _snapshot(paths)
    backup_roots: list[str] = []
    with contextlib.ExitStack() as stack:
        locks = [stack.enter_context(_RootLock(root, state_dir_name)) for root in root_paths]
        lock_by_root = {lock.root: lock for lock in locks}
        for lock in locks:
            backup_roots.append(str(lock.state_dir / "backups"))
        report.backup_roots = tuple(backup_roots)

        for path in selected:
            owning_root = next(root for root in root_paths if path == root or root in path.parents)
            lock = lock_by_root[owning_root]
            try:
                result = migrate_parquet_file(
                    path,
                    root=owning_root,
                    backup_root=lock.state_dir / "backups",
                    min_age_seconds=min_age_seconds,
                    journal_path=lock.state_dir / "journal.jsonl",
                )
                report.results.append(result)
                if result.status == "migrated":
                    report.files_migrated += 1
            except BaseException as exc:
                report.errors.append(
                    {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
                )
                break

    after_paths = _discover_parquet(root_paths)
    if tuple(str(path) for path in after_paths) != tuple(str(path) for path in paths):
        report.errors.append(
            {"path": "<global>", "error": "Parquet file set changed during migration"}
        )
    else:
        after_snapshot = _snapshot(after_paths)
        for path_text, (_, rows_before) in before_snapshot.items():
            if path_text not in after_snapshot:
                continue
            rows_after = after_snapshot[path_text][1]
            if rows_after != rows_before:
                report.errors.append(
                    {"path": path_text, "error": f"row count changed: {rows_before} -> {rows_after}"}
                )

    pending_after = 0
    for path in after_paths:
        try:
            if not audit_parquet_file(path).is_canonical:
                pending_after += 1
        except Exception as exc:
            report.errors.append(
                {"path": str(path), "error": f"post-migration audit failed: {exc}"}
            )
    report.files_pending_after = pending_after
    if max_files is None and pending_after:
        report.errors.append(
            {"path": "<global>", "error": f"{pending_after} legacy Parquet files remain"}
        )

    checked_after, manifest_errors_after = _verify_episode_manifests(root_paths)
    report.manifest_records_checked = max(report.manifest_records_checked, checked_after)
    report.errors.extend(manifest_errors_after)
    report.finished_at_unix = time.time()
    return report


def _signature_matches_recorded(
    actual: _StatSignature,
    recorded: dict[str, Any],
    *,
    include_link_count: bool,
) -> bool:
    keys = ("device", "inode", "size", "mtime_ns", "mode", "uid", "gid")
    if include_link_count:
        keys = (*keys, "nlink")
    return all(int(getattr(actual, key)) == int(recorded[key]) for key in keys)


def _verify_state_backups_before_cleanup(root: Path, state_dir: Path) -> None:
    """Prove each retained old inode maps exactly to its migrated current file."""

    if state_dir.is_symlink() or not state_dir.is_dir():
        raise MigrationError(f"invalid migration state directory {state_dir}")
    root_device = int(root.stat().st_dev)
    if os.path.ismount(state_dir) or int(state_dir.stat().st_dev) != root_device:
        raise MigrationError(f"refusing mounted/cross-device migration state {state_dir}")
    for entry in state_dir.rglob("*"):
        if os.path.ismount(entry) or int(entry.lstat().st_dev) != root_device:
            raise MigrationError(f"refusing mounted/cross-device state entry {entry}")
    allowed_top_level = {"marker.json", "journal.jsonl", "backups"}
    unknown_top_level = sorted(
        child.name for child in state_dir.iterdir() if child.name not in allowed_top_level
    )
    if unknown_top_level:
        raise MigrationError(
            f"refusing cleanup with unknown state entries in {state_dir}: {unknown_top_level}"
        )

    marker = state_dir / "marker.json"
    if marker.is_symlink() or not marker.is_file():
        raise MigrationError(f"missing or unsafe migration marker {marker}")
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    if marker_payload.get("migration_schema") != MIGRATION_SCHEMA:
        raise MigrationError(f"foreign migration marker {marker}")

    backup_root = state_dir / "backups"
    actual_backups: dict[Path, Path] = {}
    if backup_root.exists():
        if backup_root.is_symlink() or not backup_root.is_dir():
            raise MigrationError(f"unsafe backup root {backup_root}")
        for entry in backup_root.rglob("*"):
            if entry.is_symlink():
                raise MigrationError(f"refusing symlink in backup tree: {entry}")
            if entry.is_dir():
                continue
            if not entry.name.endswith(BACKUP_SUFFIX):
                raise MigrationError(f"unknown file in backup tree: {entry}")
            backup = entry.resolve(strict=True)
            relative = backup.relative_to(backup_root.resolve(strict=True))
            source_name = relative.name[: -len(BACKUP_SUFFIX)]
            if not source_name:
                raise MigrationError(f"invalid backup filename: {entry}")
            source = (root / relative.parent / source_name).resolve(strict=True)
            try:
                source.relative_to(root)
            except ValueError as exc:
                raise MigrationError(f"backup maps outside dataset root: {entry}") from exc
            actual_backups[backup] = source

    journal = state_dir / "journal.jsonl"
    journal_backups: dict[Path, dict[str, Any]] = {}
    if journal.exists():
        if journal.is_symlink() or not journal.is_file():
            raise MigrationError(f"unsafe migration journal {journal}")
        for line_number, line in enumerate(
            journal.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except Exception as exc:
                raise MigrationError(f"invalid journal JSON at {journal}:{line_number}: {exc}") from exc
            if event.get("migration_schema") != MIGRATION_SCHEMA:
                raise MigrationError(f"foreign journal event at {journal}:{line_number}")
            if event.get("event") != "migrated":
                continue
            result = event.get("result") or {}
            backup_text = result.get("backup_path")
            if not backup_text:
                raise MigrationError(f"migrated event lacks backup_path at {journal}:{line_number}")
            backup = Path(str(backup_text)).resolve(strict=True)
            if backup in journal_backups:
                raise MigrationError(f"duplicate migrated backup in journal: {backup}")
            journal_backups[backup] = event

    if set(actual_backups) != set(journal_backups):
        missing_journal = sorted(str(path) for path in set(actual_backups) - set(journal_backups))
        missing_backup = sorted(str(path) for path in set(journal_backups) - set(actual_backups))
        raise MigrationError(
            "backup/journal set mismatch: "
            f"unrecorded_backups={missing_journal}, missing_backups={missing_backup}"
        )

    for backup, source in sorted(actual_backups.items(), key=lambda item: str(item[0])):
        event = journal_backups[backup]
        result = event["result"]
        recorded_source = Path(str(result.get("path", ""))).resolve(strict=True)
        if recorded_source != source:
            raise MigrationError(
                f"journal source does not match backup path mapping: {recorded_source} != {source}"
            )
        backup_signature = _source_signature(backup)
        source_signature = _source_signature(source)
        source_before = event.get("source_before") or {}
        source_after = event.get("source_after") or {}
        try:
            backup_matches = _signature_matches_recorded(
                backup_signature,
                source_before,
                include_link_count=False,
            )
            source_matches = _signature_matches_recorded(
                source_signature,
                source_after,
                include_link_count=True,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MigrationError(f"incomplete stat signature in journal for {source}: {exc}") from exc
        if not backup_matches:
            raise MigrationError(
                f"retained backup changed since migration; refusing backup cleanup: {backup}"
            )
        if not source_matches:
            raise MigrationError(
                f"current source changed since migration; refusing backup cleanup: {source}"
            )
        _validate_rewrite(backup, source)


def cleanup_migration_backups(
    roots: Iterable[str | Path],
    *,
    state_dir_name: str = DEFAULT_STATE_DIR_NAME,
) -> dict[str, Any]:
    """Remove migration backups only after a fresh all-canonical audit."""

    root_paths = _normalize_roots(roots)
    state_dir_name = _validate_state_dir_name(state_dir_name)
    removed: list[str] = []
    with contextlib.ExitStack() as stack:
        # Lock every declared root, including roots with no state, before the
        # global verification.  Cleanup mode never creates or adopts state.
        locks = [
            stack.enter_context(
                _RootLock(root, state_dir_name, create_state=False)
            )
            for root in root_paths
        ]

        # Repeat all verification while the migration locks are held.  The
        # file-level stat checks below still protect against non-cooperating
        # writers that do not honor these advisory locks.
        verification = run_migration(root_paths, apply=False, state_dir_name=state_dir_name)
        if not verification.ok or verification.files_pending_before:
            raise MigrationError(
                "refusing backup cleanup: current datasets are not fully verified/canonical"
            )
        for root in root_paths:
            stale_temps = sorted(root.rglob(".*.schema-v21-*.tmp"))
            if stale_temps:
                raise MigrationError(
                    f"refusing cleanup with stale migration temp files: {stale_temps[:3]}"
                )

        state_locks = [lock for lock in locks if lock.state_exists]
        for lock in state_locks:
            _verify_state_backups_before_cleanup(lock.root, lock.state_dir)

        # No deletion begins until every root and every backup has passed.
        for lock in state_locks:
            shutil.rmtree(lock.state_dir)
            _fsync_directory(lock.root)
            removed.append(str(lock.state_dir))

        # This is the final operation while all locks are still held.  Verify
        # the path still names our open lock inode before removing the now-idle
        # coordination file; the open descriptor remains locked through exit.
        for lock in locks:
            if lock.handle is None or not lock.path.exists():
                continue
            open_stat = os.fstat(lock.handle.fileno())
            path_stat = lock.path.lstat()
            if (
                int(open_stat.st_dev) != int(path_stat.st_dev)
                or int(open_stat.st_ino) != int(path_stat.st_ino)
            ):
                raise MigrationError(f"lock path changed during cleanup: {lock.path}")
            lock.path.unlink()
            _fsync_directory(lock.root)
    return {
        "migration_schema": MIGRATION_SCHEMA,
        "ok": True,
        "removed_state_directories": removed,
        "verification": verification.to_dict(),
    }


def _write_report(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, destination)
    _fsync_directory(destination.parent)


def _condense_console_payload(payload: dict[str, Any], *, result_limit: int = 1) -> dict[str, Any]:
    console = dict(payload)
    results = list(console.get("results") or [])
    if len(results) > result_limit:
        console["results"] = results[:result_limit]
        console["results_truncated"] = len(results) - result_limit
        console["results_total"] = len(results)
    verification = console.get("verification")
    if isinstance(verification, dict):
        console["verification"] = _condense_console_payload(
            verification,
            result_limit=result_limit,
        )
    return console


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or transactionally normalize only known DexTeleop Arrow field types. "
            "The default is a read-only dry-run."
        )
    )
    parser.add_argument("--root", action="append", required=True, help="Dataset root; repeatable.")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--apply", action="store_true", help="Apply in-place atomic migrations.")
    operation.add_argument(
        "--cleanup-backups",
        action="store_true",
        help="After a fresh full verification, remove this tool's retained backups/state.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Canary limit for --apply.")
    parser.add_argument("--min-age-seconds", type=float, default=300.0)
    parser.add_argument("--state-dir-name", default=DEFAULT_STATE_DIR_NAME)
    parser.add_argument("--report-json", default="", help="Optional atomic JSON report output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cleanup_backups:
            if args.max_files is not None:
                raise MigrationError("--max-files cannot be used with --cleanup-backups")
            payload = cleanup_migration_backups(
                args.root,
                state_dir_name=str(args.state_dir_name),
            )
            ok = True
        else:
            report = run_migration(
                args.root,
                apply=bool(args.apply),
                max_files=args.max_files,
                min_age_seconds=float(args.min_age_seconds),
                state_dir_name=str(args.state_dir_name),
            )
            payload = report.to_dict()
            ok = report.ok
    except BaseException as exc:
        payload = {
            "migration_schema": MIGRATION_SCHEMA,
            "ok": False,
            "fatal_error": f"{type(exc).__name__}: {exc}",
        }
        ok = False
    if args.report_json:
        _write_report(args.report_json, payload)
    print(
        json.dumps(
            _condense_console_payload(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
