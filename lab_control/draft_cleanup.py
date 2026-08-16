from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import git
from .draft_bundle import BUNDLE_SCHEMA_ID, _repository_entry
from .drafts import DRAFT_SCHEMA_ID
from .workspace import LabError, WorkspaceManager, validate_task_id


def _timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise LabError(f"draft {label} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise LabError(f"draft {label} is invalid: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LabError(f"draft {label} must include a timezone")
    return parsed.astimezone(timezone.utc)


class DraftSnapshotPruner:
    """Inventory and explicitly retire expired lab-owned synthetic refs/state."""

    def __init__(self, workspace: WorkspaceManager) -> None:
        self.workspace = workspace

    @property
    def drafts_root(self) -> Path:
        return (self.workspace.state_root / "drafts").resolve()

    @property
    def default_exports_root(self) -> Path:
        return (self.workspace.state_root / "draft-exports").resolve()

    def inventory(
        self,
        *,
        task_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, object]:
        if task_id is not None:
            validate_task_id(task_id)
        effective_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if effective_now.tzinfo is None or effective_now.utcoffset() is None:
            raise LabError("draft prune clock must include a timezone")
        root = self.drafts_root
        if root.is_symlink():
            raise LabError(f"draft state root is a symlink: {root}")
        records: list[dict[str, object]] = []
        if root.exists():
            task_roots = (
                [root / task_id] if task_id is not None else sorted(root.iterdir())
            )
            for task_root in task_roots:
                if not task_root.exists():
                    continue
                if task_root.is_symlink() or not task_root.is_dir():
                    raise LabError(f"invalid draft Task state directory: {task_root}")
                current_task_id = validate_task_id(task_root.name)
                task = self.workspace.load_task(current_task_id)
                task_repositories = task.get("repositories")
                if not isinstance(task_repositories, dict):
                    raise LabError(f"Task has no repositories: {current_task_id}")
                for snapshot_root in sorted(task_root.iterdir()):
                    if snapshot_root.is_symlink() or not snapshot_root.is_dir():
                        raise LabError(
                            f"invalid draft snapshot directory: {snapshot_root}"
                        )
                    snapshot_id = validate_task_id(snapshot_root.name)
                    manifest_path = snapshot_root / "snapshot.json"
                    if manifest_path.is_symlink() or not manifest_path.is_file():
                        raise LabError(
                            f"draft manifest is missing or a symlink: {manifest_path}"
                        )
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        raise LabError(
                            f"cannot read draft manifest {manifest_path}: {exc}"
                        ) from exc
                    if (
                        not isinstance(manifest, dict)
                        or manifest.get("schema") != DRAFT_SCHEMA_ID
                        or manifest.get("task_id") != current_task_id
                        or manifest.get("snapshot_id") != snapshot_id
                    ):
                        raise LabError(
                            f"invalid draft manifest identity: {manifest_path}"
                        )
                    repositories = manifest.get("repositories")
                    if not isinstance(repositories, dict) or set(repositories) != set(
                        task_repositories
                    ):
                        raise LabError(f"draft/Task repository mismatch: {snapshot_id}")
                    repository_records: dict[str, object] = {}
                    for name, raw_entry in repositories.items():
                        if not isinstance(name, str):
                            raise LabError("draft repository name must be a string")
                        entry = _repository_entry(name, raw_entry)
                        task_repository = task_repositories[name]
                        if not isinstance(task_repository, dict):
                            raise LabError(f"invalid Task repository entry: {name}")
                        path = Path(
                            str(task_repository.get("worktree_path", ""))
                        ).resolve()
                        if git.repository_root(path) != path:
                            raise LabError(
                                f"Task repository path/root mismatch: {name}"
                            )
                        present = git.ref_exists(path, str(entry["ref"]))
                        if (
                            present
                            and git.resolve_commit(path, str(entry["ref"]))
                            != entry["commit"]
                        ):
                            raise LabError(
                                f"draft ref/commit mismatch: {name}/{snapshot_id}"
                            )
                        repository_records[name] = {
                            "path": str(path),
                            "ref": entry["ref"],
                            "commit": entry["commit"],
                            "ref_present": present,
                        }
                    expires_at = _timestamp(
                        manifest.get("expires_at"), label="expires_at"
                    )
                    export_root = self.default_exports_root / snapshot_id
                    default_export = False
                    if export_root.exists() or export_root.is_symlink():
                        if export_root.is_symlink() or not export_root.is_dir():
                            raise LabError(
                                f"invalid default draft export: {export_root}"
                            )
                        bundle_path = export_root / "bundle.json"
                        try:
                            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError) as exc:
                            raise LabError(
                                f"cannot read default draft export {bundle_path}: {exc}"
                            ) from exc
                        if (
                            not isinstance(bundle, dict)
                            or bundle.get("schema") != BUNDLE_SCHEMA_ID
                            or bundle.get("snapshot_id") != snapshot_id
                            or bundle.get("task_id") != current_task_id
                        ):
                            raise LabError(
                                f"default draft export identity mismatch: {export_root}"
                            )
                        default_export = True
                    records.append(
                        {
                            "task_id": current_task_id,
                            "snapshot_id": snapshot_id,
                            "expires_at": expires_at.isoformat(),
                            "expired": expires_at <= effective_now,
                            "state_path": str(snapshot_root),
                            "default_export_path": str(export_root)
                            if default_export
                            else None,
                            "repositories": repository_records,
                        }
                    )
        return {
            "schema": "lab.draft-prune-plan.v1",
            "evaluated_at": effective_now.isoformat(),
            "task_id": task_id,
            "snapshots": records,
            "expired_count": sum(bool(record["expired"]) for record in records),
            "applied": False,
        }

    def prune(
        self,
        *,
        task_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, object]:
        plan = self.inventory(task_id=task_id, now=now)
        raw_snapshots = plan.get("snapshots")
        if not isinstance(raw_snapshots, list) or not all(
            isinstance(record, dict) for record in raw_snapshots
        ):
            raise LabError("draft prune inventory has an invalid snapshot list")
        snapshots = [record for record in raw_snapshots if isinstance(record, dict)]
        expired = [record for record in snapshots if record.get("expired") is True]
        lock_keys = sorted(
            {
                f"{name}:{repository['path']}"
                for record in expired
                for name, repository in record["repositories"].items()
            }
        )
        with self.workspace.advisory_locks("repositories", lock_keys):
            refreshed = self.inventory(
                task_id=task_id,
                now=_timestamp(plan["evaluated_at"], label="evaluated_at"),
            )
            raw_refreshed = refreshed.get("snapshots")
            if not isinstance(raw_refreshed, list) or not all(
                isinstance(record, dict) for record in raw_refreshed
            ):
                raise LabError("refreshed draft inventory is invalid")
            refreshed_expired = [
                record
                for record in raw_refreshed
                if isinstance(record, dict) and record.get("expired") is True
            ]
            if refreshed_expired != expired:
                raise LabError("draft prune inventory changed before apply")
            retired_root = self.workspace.state_root / "draft-retired"
            if retired_root.is_symlink():
                raise LabError(f"draft retired root is a symlink: {retired_root}")
            retired_root.mkdir(parents=True, exist_ok=True)
            retired: list[dict[str, object]] = []
            cleanup_pending: list[str] = []
            for record in expired:
                deleted_refs: list[tuple[Path, str, str]] = []
                state_path = Path(record["state_path"])
                export_value = record["default_export_path"]
                export_path = (
                    Path(export_value) if isinstance(export_value, str) else None
                )
                snapshot_id = str(record["snapshot_id"])
                task_value = str(record["task_id"])
                retired_task_root = retired_root / task_value
                if retired_task_root.is_symlink():
                    raise LabError(
                        f"draft retired Task root is a symlink: {retired_task_root}"
                    )
                retired_task_root.mkdir(exist_ok=True)
                tombstone = retired_task_root / snapshot_id
                if tombstone.resolve().parent != retired_task_root.resolve():
                    raise LabError("draft retired tombstone escapes its Task root")
                if tombstone.exists() or tombstone.is_symlink():
                    raise LabError(
                        f"draft retired tombstone already exists: {tombstone}"
                    )
                tombstone.mkdir(parents=True)
                staged_state = tombstone / "state"
                staged_export = tombstone / "default-export"
                try:
                    state_path.rename(staged_state)
                    if export_path is not None:
                        export_path.rename(staged_export)
                    for repository in record["repositories"].values():
                        if repository["ref_present"]:
                            path = Path(repository["path"])
                            ref = str(repository["ref"])
                            commit = str(repository["commit"])
                            git.run_git(path, ["update-ref", "-d", ref, commit])
                            deleted_refs.append((path, ref, commit))
                    retired.append(record)
                except Exception as exc:
                    for path, ref, commit in reversed(deleted_refs):
                        git.run_git(path, ["update-ref", ref, commit, "0" * 40])
                    if staged_export.exists() and export_path is not None:
                        staged_export.rename(export_path)
                    if staged_state.exists():
                        staged_state.rename(state_path)
                    if tombstone.exists() and not any(tombstone.iterdir()):
                        tombstone.rmdir()
                    raise LabError(
                        f"draft prune failed for {record['snapshot_id']}: {exc}"
                    ) from exc
                try:
                    shutil.rmtree(tombstone)
                except OSError:
                    cleanup_pending.append(str(tombstone))
        return {
            **plan,
            "applied": True,
            "retired": [record["snapshot_id"] for record in retired],
            "cleanup_pending": cleanup_pending,
        }
