"""Optional SQLite cache for rebuildable refinement coverage views.

The database is a convenience index only.  Canonical progress and annotation
artifacts remain the sole source of truth, and callers may delete this file and
rebuild it at any time.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Iterable

from .progress_coverage import (
    REFINEMENT_COVERAGE_STATUSES,
    RefinementAnnotationCoverageView,
)


_CATALOG_FORMAT_VERSION = 2


class ProgressCoverageCatalogError(ValueError):
    """Raised when a derived coverage cache is invalid or incompatible."""


@dataclass(frozen=True)
class RefinementCoverageCatalogAttemptRow:
    episode_capture_identity_hash: str
    source_progress_manifest_hash: str
    source_annotation_manifest_hash: str
    task_refinement_profile_hash: str
    annotation_set_id: str
    attempt_id: str
    progress_step_id: str
    status: str
    required_boundary_ids: tuple[str, ...]
    observed_boundary_ids: tuple[str, ...]
    missing_boundary_ids: tuple[str, ...]
    conflicting_boundary_ids: tuple[str, ...]
    annotation_ids: tuple[str, ...]
    invalid_annotation_ids: tuple[str, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class RefinementCoverageCatalogSetRow:
    episode_capture_identity_hash: str
    source_progress_manifest_hash: str
    source_annotation_manifest_hash: str
    task_refinement_profile_hash: str
    annotation_set_id: str
    clock_domain_id: str
    unassigned_annotation_ids: tuple[str, ...]
    issues: tuple[str, ...]


def _json_tuple(values: tuple[str, ...]) -> str:
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def _decode_tuple(value: str, *, column: str) -> tuple[str, ...]:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProgressCoverageCatalogError(
            f"Coverage cache column {column!r} contains invalid JSON."
        ) from exc
    if not isinstance(decoded, list) or any(
        not isinstance(item, str) for item in decoded
    ):
        raise ProgressCoverageCatalogError(
            f"Coverage cache column {column!r} must contain a string array."
        )
    return tuple(decoded)


class RefinementCoverageCatalog:
    """Small query cache with transactional replacement and full rebuild."""

    def __init__(self, database_path: str | Path) -> None:
        path = Path(database_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = path
        self._connection = sqlite3.connect(str(path), timeout=5.0)
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute("PRAGMA foreign_keys = ON")
        mode = self._connection.execute(
            "PRAGMA journal_mode = WAL"
        ).fetchone()
        if mode is None or str(mode[0]).lower() != "wal":
            self._connection.close()
            raise ProgressCoverageCatalogError(
                "Coverage cache requires SQLite WAL mode."
            )
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS catalog_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            row = self._connection.execute(
                "SELECT value FROM catalog_metadata WHERE key = "
                "'format_version'"
            ).fetchone()
            if row is not None and int(row[0]) != _CATALOG_FORMAT_VERSION:
                raise ProgressCoverageCatalogError(
                    "Coverage cache format is incompatible; delete and rebuild "
                    "the derived database."
                )
            self._connection.execute(
                "INSERT OR IGNORE INTO catalog_metadata(key, value) "
                "VALUES ('format_version', ?)",
                (str(_CATALOG_FORMAT_VERSION),),
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS annotation_set_coverage (
                    episode_capture_identity_hash TEXT NOT NULL,
                    source_progress_manifest_hash TEXT NOT NULL,
                    source_annotation_manifest_hash TEXT NOT NULL,
                    task_refinement_profile_hash TEXT NOT NULL,
                    annotation_set_id TEXT NOT NULL,
                    clock_domain_id TEXT NOT NULL,
                    unassigned_annotation_ids_json TEXT NOT NULL,
                    issues_json TEXT NOT NULL,
                    PRIMARY KEY (
                        source_progress_manifest_hash,
                        source_annotation_manifest_hash,
                        task_refinement_profile_hash,
                        annotation_set_id
                    )
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS attempt_coverage (
                    episode_capture_identity_hash TEXT NOT NULL,
                    source_progress_manifest_hash TEXT NOT NULL,
                    source_annotation_manifest_hash TEXT NOT NULL,
                    task_refinement_profile_hash TEXT NOT NULL,
                    annotation_set_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    progress_step_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    required_boundary_ids_json TEXT NOT NULL,
                    observed_boundary_ids_json TEXT NOT NULL,
                    missing_boundary_ids_json TEXT NOT NULL,
                    conflicting_boundary_ids_json TEXT NOT NULL,
                    annotation_ids_json TEXT NOT NULL,
                    invalid_annotation_ids_json TEXT NOT NULL,
                    issues_json TEXT NOT NULL,
                    PRIMARY KEY (
                        source_progress_manifest_hash,
                        source_annotation_manifest_hash,
                        task_refinement_profile_hash,
                        annotation_set_id,
                        attempt_id
                    ),
                    FOREIGN KEY (
                        source_progress_manifest_hash,
                        source_annotation_manifest_hash,
                        task_refinement_profile_hash,
                        annotation_set_id
                    ) REFERENCES annotation_set_coverage (
                        source_progress_manifest_hash,
                        source_annotation_manifest_hash,
                        task_refinement_profile_hash,
                        annotation_set_id
                    ) ON DELETE CASCADE
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS "
                "attempt_coverage_status_idx ON attempt_coverage(status)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS "
                "attempt_coverage_episode_idx ON "
                "attempt_coverage(episode_capture_identity_hash)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS "
                "attempt_coverage_profile_status_idx ON "
                "attempt_coverage(task_refinement_profile_hash, status)"
            )

    def close(self) -> None:
        self._connection.close()

    @property
    def journal_mode(self) -> str:
        row = self._connection.execute("PRAGMA journal_mode").fetchone()
        if row is None:
            raise ProgressCoverageCatalogError(
                "SQLite did not report a journal mode."
            )
        return str(row[0]).lower()

    def __enter__(self) -> "RefinementCoverageCatalog":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

    @staticmethod
    def _set_key(
        view: RefinementAnnotationCoverageView,
    ) -> tuple[str, str, str, str]:
        return (
            view.source_progress_manifest_hash,
            view.source_annotation_manifest_hash,
            view.task_refinement_profile_hash,
            view.annotation_set_id,
        )

    def _insert_view(
        self,
        view: RefinementAnnotationCoverageView,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO annotation_set_coverage(
                episode_capture_identity_hash,
                source_progress_manifest_hash,
                source_annotation_manifest_hash,
                task_refinement_profile_hash,
                annotation_set_id,
                clock_domain_id,
                unassigned_annotation_ids_json,
                issues_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                view.episode_capture_identity_hash,
                view.source_progress_manifest_hash,
                view.source_annotation_manifest_hash,
                view.task_refinement_profile_hash,
                view.annotation_set_id,
                view.clock_domain_id,
                _json_tuple(view.unassigned_annotation_ids),
                _json_tuple(view.issues),
            ),
        )
        for attempt in view.attempts:
            if attempt.status not in REFINEMENT_COVERAGE_STATUSES:
                raise ProgressCoverageCatalogError(
                    f"Unsupported coverage status {attempt.status!r}."
                )
            self._connection.execute(
                """
                INSERT INTO attempt_coverage(
                    episode_capture_identity_hash,
                    source_progress_manifest_hash,
                    source_annotation_manifest_hash,
                    task_refinement_profile_hash,
                    annotation_set_id,
                    attempt_id,
                    progress_step_id,
                    status,
                    required_boundary_ids_json,
                    observed_boundary_ids_json,
                    missing_boundary_ids_json,
                    conflicting_boundary_ids_json,
                    annotation_ids_json,
                    invalid_annotation_ids_json,
                    issues_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    view.episode_capture_identity_hash,
                    view.source_progress_manifest_hash,
                    view.source_annotation_manifest_hash,
                    view.task_refinement_profile_hash,
                    view.annotation_set_id,
                    attempt.attempt_id,
                    attempt.progress_step_id,
                    attempt.status,
                    _json_tuple(attempt.required_boundary_ids),
                    _json_tuple(attempt.observed_boundary_ids),
                    _json_tuple(attempt.missing_boundary_ids),
                    _json_tuple(attempt.conflicting_boundary_ids),
                    _json_tuple(attempt.annotation_ids),
                    _json_tuple(attempt.invalid_annotation_ids),
                    _json_tuple(attempt.issues),
                ),
            )

    def replace_view(
        self,
        view: RefinementAnnotationCoverageView,
    ) -> None:
        """Atomically replace one annotation-set cache projection."""

        key = self._set_key(view)
        with self._connection:
            self._connection.execute(
                """
                DELETE FROM annotation_set_coverage
                WHERE source_progress_manifest_hash = ?
                  AND source_annotation_manifest_hash = ?
                  AND task_refinement_profile_hash = ?
                  AND annotation_set_id = ?
                """,
                key,
            )
            self._insert_view(view)

    def rebuild(
        self,
        views: Iterable[RefinementAnnotationCoverageView],
    ) -> None:
        """Atomically rebuild the cache from caller-rederived views."""

        materialized = tuple(views)
        keys = tuple(self._set_key(item) for item in materialized)
        if len(keys) != len(set(keys)):
            raise ProgressCoverageCatalogError(
                "Coverage rebuild contains duplicate annotation-set keys."
            )
        with self._connection:
            self._connection.execute("DELETE FROM annotation_set_coverage")
            for view in materialized:
                self._insert_view(view)

    def list_attempts(
        self,
        *,
        status: str | None = None,
        episode_capture_identity_hash: str | None = None,
        source_progress_manifest_hash: str | None = None,
        source_annotation_manifest_hash: str | None = None,
        task_refinement_profile_hash: str | None = None,
        annotation_set_id: str | None = None,
        attempt_id: str | None = None,
    ) -> tuple[RefinementCoverageCatalogAttemptRow, ...]:
        conditions: list[str] = []
        parameters: list[str] = []
        if status is not None:
            normalized = status.strip().upper()
            if normalized not in REFINEMENT_COVERAGE_STATUSES:
                raise ProgressCoverageCatalogError(
                    f"Unsupported coverage status {status!r}."
                )
            conditions.append("status = ?")
            parameters.append(normalized)
        if episode_capture_identity_hash is not None:
            conditions.append("episode_capture_identity_hash = ?")
            parameters.append(episode_capture_identity_hash)
        for column, value in (
            (
                "source_progress_manifest_hash",
                source_progress_manifest_hash,
            ),
            (
                "source_annotation_manifest_hash",
                source_annotation_manifest_hash,
            ),
            (
                "task_refinement_profile_hash",
                task_refinement_profile_hash,
            ),
            ("annotation_set_id", annotation_set_id),
            ("attempt_id", attempt_id),
        ):
            if value is not None:
                conditions.append(f"{column} = ?")
                parameters.append(value)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._connection.execute(
            """
            SELECT
                episode_capture_identity_hash,
                source_progress_manifest_hash,
                source_annotation_manifest_hash,
                task_refinement_profile_hash,
                annotation_set_id,
                attempt_id,
                progress_step_id,
                status,
                required_boundary_ids_json,
                observed_boundary_ids_json,
                missing_boundary_ids_json,
                conflicting_boundary_ids_json,
                annotation_ids_json,
                invalid_annotation_ids_json,
                issues_json
            FROM attempt_coverage
            """
            + where
            + """
            ORDER BY
                episode_capture_identity_hash,
                source_progress_manifest_hash,
                source_annotation_manifest_hash,
                task_refinement_profile_hash,
                annotation_set_id,
                attempt_id
            """,
            tuple(parameters),
        ).fetchall()
        return tuple(
            RefinementCoverageCatalogAttemptRow(
                episode_capture_identity_hash=row[0],
                source_progress_manifest_hash=row[1],
                source_annotation_manifest_hash=row[2],
                task_refinement_profile_hash=row[3],
                annotation_set_id=row[4],
                attempt_id=row[5],
                progress_step_id=row[6],
                status=row[7],
                required_boundary_ids=_decode_tuple(
                    row[8], column="required_boundary_ids_json"
                ),
                observed_boundary_ids=_decode_tuple(
                    row[9], column="observed_boundary_ids_json"
                ),
                missing_boundary_ids=_decode_tuple(
                    row[10], column="missing_boundary_ids_json"
                ),
                conflicting_boundary_ids=_decode_tuple(
                    row[11], column="conflicting_boundary_ids_json"
                ),
                annotation_ids=_decode_tuple(
                    row[12], column="annotation_ids_json"
                ),
                invalid_annotation_ids=_decode_tuple(
                    row[13], column="invalid_annotation_ids_json"
                ),
                issues=_decode_tuple(row[14], column="issues_json"),
            )
            for row in rows
        )

    def list_annotation_sets(
        self,
        *,
        with_issues_only: bool = False,
    ) -> tuple[RefinementCoverageCatalogSetRow, ...]:
        where = (
            " WHERE unassigned_annotation_ids_json != '[]' "
            "OR issues_json != '[]'"
            if with_issues_only
            else ""
        )
        rows = self._connection.execute(
            """
            SELECT
                episode_capture_identity_hash,
                source_progress_manifest_hash,
                source_annotation_manifest_hash,
                task_refinement_profile_hash,
                annotation_set_id,
                clock_domain_id,
                unassigned_annotation_ids_json,
                issues_json
            FROM annotation_set_coverage
            """
            + where
            + """
            ORDER BY
                episode_capture_identity_hash,
                source_progress_manifest_hash,
                source_annotation_manifest_hash,
                task_refinement_profile_hash,
                annotation_set_id
            """
        ).fetchall()
        return tuple(
            RefinementCoverageCatalogSetRow(
                episode_capture_identity_hash=row[0],
                source_progress_manifest_hash=row[1],
                source_annotation_manifest_hash=row[2],
                task_refinement_profile_hash=row[3],
                annotation_set_id=row[4],
                clock_domain_id=row[5],
                unassigned_annotation_ids=_decode_tuple(
                    row[6], column="unassigned_annotation_ids_json"
                ),
                issues=_decode_tuple(row[7], column="issues_json"),
            )
            for row in rows
        )


__all__ = [
    "ProgressCoverageCatalogError",
    "RefinementCoverageCatalog",
    "RefinementCoverageCatalogAttemptRow",
    "RefinementCoverageCatalogSetRow",
]
