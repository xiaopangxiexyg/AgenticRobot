from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from . import git
from .runner_protocol import RunnerCoordinator
from .workspace import (
    LabError,
    _atomic_write_text,
    validate_repository_name,
    validate_submodule_path,
    validate_task_id,
)

SNAPSHOT_STAGE_RECEIPT_SCHEMA_ID = "lab.snapshot-stage-receipt.v1"
_RECEIPT_FILENAME = "stage-receipt.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


class ExactSnapshotStager:
    """Create a clean source slot from local Git objects without execution."""

    def __init__(
        self,
        coordinator: RunnerCoordinator,
        *,
        repository_catalog: Mapping[str, Path],
        slot_root: Path,
    ) -> None:
        self.coordinator = coordinator
        catalog: dict[str, Path] = {}
        for raw_name, raw_path in repository_catalog.items():
            name = validate_repository_name(str(raw_name))
            path = Path(raw_path).expanduser().resolve()
            if name in catalog:
                raise LabError(f"duplicate runner repository source: {name}")
            catalog[name] = path
        if not catalog:
            raise LabError("runner repository catalog cannot be empty")
        self.repository_catalog = catalog
        self.slot_root = slot_root.expanduser().resolve()

    @staticmethod
    def _repository_targets(
        source_root: Path,
        repositories: Mapping[str, object],
        links: object,
    ) -> tuple[list[str], dict[str, Path]]:
        parents: dict[str, tuple[str, str]] = {}
        if not isinstance(links, list):
            raise LabError("Run request repository links must be a list")
        for raw_link in links:
            if not isinstance(raw_link, dict):
                raise LabError("Run request repository link is invalid")
            parent_name = validate_repository_name(
                str(raw_link.get("parent_repository", ""))
            )
            child_name = validate_repository_name(
                str(raw_link.get("child_repository", ""))
            )
            submodule_path = validate_submodule_path(
                str(raw_link.get("submodule_path", ""))
            )
            if parent_name not in repositories or child_name not in repositories:
                raise LabError("Run request repository link references an absent repository")
            if child_name in parents:
                raise LabError(
                    f"repository has multiple staging parents: {child_name}"
                )
            parents[child_name] = (parent_name, submodule_path)

        order: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise LabError("Run request repository links contain a cycle")
            if name in visited:
                return
            visiting.add(name)
            parent = parents.get(name)
            if parent is not None:
                visit(parent[0])
            visiting.remove(name)
            visited.add(name)
            order.append(name)

        for name in sorted(repositories):
            visit(name)

        targets: dict[str, Path] = {}
        for name in order:
            parent = parents.get(name)
            if parent is None:
                target = source_root / name
            else:
                parent_target = targets[parent[0]]
                path = PurePosixPath(parent[1])
                target = parent_target.joinpath(*path.parts)
            resolved_parent = target.parent.resolve(strict=False)
            if not resolved_parent.is_relative_to(source_root.resolve()):
                raise LabError(f"repository staging path escapes slot: {name}")
            if target in targets.values():
                raise LabError(f"repository staging paths collide: {name}")
            targets[name] = target
        return order, targets

    @staticmethod
    def _validate_receipt(
        receipt: object,
        *,
        request: Mapping[str, object],
        claim: Mapping[str, object],
    ) -> dict[str, object]:
        expected = {
            "schema",
            "request_id",
            "idempotency_key",
            "snapshot_id",
            "claim_id",
            "runner_id",
            "runner_profile_sha256",
            "slot_id",
            "repositories",
            "repository_links",
            "network_access",
            "lfs_smudge",
            "created_at",
            "receipt_sha256",
        }
        if not isinstance(receipt, dict) or set(receipt) != expected:
            raise LabError("snapshot stage receipt has an unexpected shape")
        snapshot = request.get("snapshot")
        if not isinstance(snapshot, dict):
            raise LabError("Run request snapshot identity is unavailable")
        if (
            receipt.get("schema") != SNAPSHOT_STAGE_RECEIPT_SCHEMA_ID
            or receipt.get("request_id") != request.get("request_id")
            or receipt.get("idempotency_key") != request.get("idempotency_key")
            or receipt.get("snapshot_id") != snapshot.get("snapshot_id")
            or receipt.get("claim_id") != claim.get("claim_id")
            or receipt.get("runner_id") != claim.get("runner_id")
            or receipt.get("runner_profile_sha256")
            != claim.get("runner_profile_sha256")
            or receipt.get("slot_id") != claim.get("slot_id")
            or receipt.get("repository_links") != snapshot.get("repository_links")
            or receipt.get("network_access") is not False
            or receipt.get("lfs_smudge") is not False
        ):
            raise LabError("snapshot stage receipt identity mismatch")
        content = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        if receipt.get("receipt_sha256") != _sha256_json(content):
            raise LabError("snapshot stage receipt hash mismatch")
        return receipt

    @staticmethod
    def _validate_materialized_sources(
        receipt: Mapping[str, object],
        *,
        request_root: Path,
        request: Mapping[str, object],
    ) -> None:
        snapshot = request.get("snapshot")
        receipt_repositories = receipt.get("repositories")
        if not isinstance(snapshot, dict) or not isinstance(receipt_repositories, dict):
            raise LabError("snapshot stage receipt has no repository identities")
        expected_repositories = snapshot.get("repositories")
        if not isinstance(expected_repositories, dict) or set(receipt_repositories) != set(
            expected_repositories
        ):
            raise LabError("snapshot stage receipt repository set mismatch")
        for name, raw_receipt in receipt_repositories.items():
            raw_expected = expected_repositories[name]
            if not isinstance(raw_receipt, dict) or not isinstance(raw_expected, dict):
                raise LabError(f"invalid staged repository identity: {name}")
            relative_path = raw_receipt.get("relative_path")
            if not isinstance(relative_path, str):
                raise LabError(f"invalid staged repository path: {name}")
            path = request_root.joinpath(*PurePosixPath(relative_path).parts)
            if not path.resolve().is_relative_to(request_root.resolve()):
                raise LabError(f"staged repository escapes request slot: {name}")
            try:
                if (
                    git.current_branch(path) is not None
                    or git.current_commit(path) != raw_expected.get("commit")
                    or git.tree_oid(path) != raw_expected.get("tree_oid")
                    or git.submodule_gitlinks(path) != raw_expected.get("submodules")
                    or git.status_short(path)
                ):
                    raise LabError(f"staged repository source drift: {name}")
            except git.GitError as exc:
                raise LabError(f"cannot validate staged repository {name}: {exc}") from exc

    def stage(self, request_id: str, runner_id: str) -> dict[str, object]:
        validate_task_id(request_id)
        validate_task_id(runner_id)
        claim, state, _ = self.coordinator.validated_claim(request_id, runner_id)
        if state.get("state") == "RESERVED":
            state = self.coordinator.begin_preparing(request_id, runner_id)
        if state.get("state") != "PREPARING":
            raise LabError("snapshot staging requires a RESERVED or PREPARING request")
        request = self.coordinator.store.load_request(request_id)
        snapshot = request.get("snapshot")
        if not isinstance(snapshot, dict):
            raise LabError("Run request snapshot identity is unavailable")
        repositories = snapshot.get("repositories")
        if not isinstance(repositories, dict) or not repositories:
            raise LabError("Run request snapshot has no repositories")
        slot_id = validate_task_id(str(claim.get("slot_id", "")))
        request_root = self.slot_root / slot_id / request_id
        source_root = request_root / "source"
        receipt_path = request_root / _RECEIPT_FILENAME

        if request_root.exists():
            if request_root.is_symlink() or not receipt_path.is_file():
                raise LabError(f"snapshot slot is incomplete/quarantined: {request_root}")
            try:
                raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise LabError(f"cannot read snapshot stage receipt: {exc}") from exc
            receipt = self._validate_receipt(
                raw_receipt, request=request, claim=claim
            )
            self._validate_materialized_sources(
                receipt, request_root=request_root, request=request
            )
            persisted, _ = self.coordinator.store.put_attachment(
                request_id, "snapshot-stage-receipt", receipt
            )
            return persisted

        self.slot_root.mkdir(parents=True, exist_ok=True)
        if self.slot_root.is_symlink():
            raise LabError(f"runner slot root must not be a symlink: {self.slot_root}")
        request_root.mkdir(parents=True, exist_ok=False)
        source_root.mkdir()
        order, targets = self._repository_targets(
            source_root, repositories, snapshot.get("repository_links")
        )
        created: list[tuple[Path, Path]] = []
        try:
            receipt_repositories: dict[str, object] = {}
            for name in order:
                raw_repository = repositories[name]
                if not isinstance(raw_repository, dict):
                    raise LabError(f"invalid Run request repository identity: {name}")
                source = self.repository_catalog.get(name)
                if source is None:
                    raise LabError(f"runner has no trusted repository source: {name}")
                target = targets[name]
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.is_dir() and not any(target.iterdir()):
                        target.rmdir()
                    else:
                        raise LabError(f"repository staging target is occupied: {target}")
                try:
                    actual_commit = git.resolve_commit(
                        source, str(raw_repository.get("commit", ""))
                    )
                    actual_tree = git.tree_oid(source, actual_commit)
                    actual_submodules = git.submodule_gitlinks(source, actual_commit)
                except git.GitError as exc:
                    raise LabError(
                        f"runner lacks exact local Git objects for {name}: {exc}"
                    ) from exc
                if (
                    actual_tree != raw_repository.get("tree_oid")
                    or actual_submodules != raw_repository.get("submodules")
                ):
                    raise LabError(f"runner repository identity mismatch: {name}")
                try:
                    git.create_unfiltered_detached_worktree(
                        source, target=target, commit=actual_commit
                    )
                except git.GitError as exc:
                    raise LabError(f"cannot stage exact repository {name}: {exc}") from exc
                created.append((source, target))
                receipt_repositories[name] = {
                    "commit": actual_commit,
                    "tree_oid": actual_tree,
                    "submodules": actual_submodules,
                    "relative_path": target.relative_to(request_root).as_posix(),
                }
            content: dict[str, object] = {
                "schema": SNAPSHOT_STAGE_RECEIPT_SCHEMA_ID,
                "request_id": request_id,
                "idempotency_key": request["idempotency_key"],
                "snapshot_id": snapshot["snapshot_id"],
                "claim_id": claim["claim_id"],
                "runner_id": runner_id,
                "runner_profile_sha256": claim["runner_profile_sha256"],
                "slot_id": slot_id,
                "repositories": receipt_repositories,
                "repository_links": snapshot.get("repository_links"),
                "network_access": False,
                "lfs_smudge": False,
                "created_at": _utc_now(),
            }
            receipt = {**content, "receipt_sha256": _sha256_json(content)}
            self._validate_materialized_sources(
                receipt, request_root=request_root, request=request
            )
            _atomic_write_text(
                receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            )
            persisted, _ = self.coordinator.store.put_attachment(
                request_id, "snapshot-stage-receipt", receipt
            )
            return persisted
        except Exception:
            for source, target in reversed(created):
                try:
                    git.remove_worktree(source, target)
                except git.GitError:
                    pass
            raise

    def validate_staged(
        self, request_id: str, runner_id: str
    ) -> tuple[dict[str, object], Path]:
        """Revalidate a claimed staged source tree and return its request root."""

        validate_task_id(request_id)
        validate_task_id(runner_id)
        claim, _, _ = self.coordinator.validated_claim(request_id, runner_id)
        request = self.coordinator.store.load_request(request_id)
        slot_id = validate_task_id(str(claim.get("slot_id", "")))
        request_root = self.slot_root / slot_id / request_id
        receipt_path = request_root / _RECEIPT_FILENAME
        if (
            request_root.is_symlink()
            or not request_root.is_dir()
            or receipt_path.is_symlink()
            or not receipt_path.is_file()
        ):
            raise LabError(f"snapshot slot is incomplete/quarantined: {request_root}")
        try:
            raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LabError(f"cannot read snapshot stage receipt: {exc}") from exc
        receipt = self._validate_receipt(raw_receipt, request=request, claim=claim)
        attached = self.coordinator.store.load_attachment(
            request_id, "snapshot-stage-receipt"
        )
        if attached != receipt:
            raise LabError("staged snapshot receipt differs from durable attachment")
        self._validate_materialized_sources(
            receipt, request_root=request_root, request=request
        )
        return receipt, request_root
