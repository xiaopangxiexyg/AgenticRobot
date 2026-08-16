from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from . import git
from .draft_bundle import (
    BUNDLE_SCHEMA_ID,
    _canonical_json,
    _read_json_object,
    _repository_entry,
    _resolve_bundle_path,
    verify_draft_bundle_set,
)
from .workspace import (
    LabError,
    _atomic_write_text,
    validate_repository_name,
    validate_task_id,
)

DRAFT_SLOT_SCHEMA_ID = "lab.draft-slot-materialization.v1"


def _run_git(
    repository: Path | None,
    arguments: Sequence[str],
) -> str:
    command = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "filter.lfs.smudge=",
        "-c",
        "filter.lfs.process=",
        "-c",
        "filter.lfs.required=false",
    ]
    if repository is not None:
        command.extend(("-C", str(repository)))
    command.extend(arguments)
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        location = repository if repository is not None else "process"
        raise LabError(f"Git slot materialization failed in {location}: {detail}")
    return completed.stdout.rstrip()


def _real_repository_anchor(path: Path, *, name: str) -> Path:
    requested = path.expanduser()
    if not requested.is_absolute():
        raise LabError(f"repository anchor must be absolute: {name}")
    absolute = requested.absolute()
    resolved = requested.resolve()
    if absolute != resolved or requested.is_symlink():
        raise LabError(f"repository anchor must not use symlinks: {name}")
    try:
        root = git.repository_root(resolved)
    except git.GitError as exc:
        raise LabError(str(exc)) from exc
    if root != resolved:
        raise LabError(f"repository anchor/root mismatch: {name}")
    return resolved


def _safe_slot_root(path: Path, anchors: Mapping[str, Path]) -> Path:
    requested = path.expanduser()
    if not requested.is_absolute():
        raise LabError("slot root must be an absolute path")
    absolute = requested.absolute()
    resolved = requested.resolve()
    if absolute != resolved or requested.is_symlink():
        raise LabError(f"slot root must not use symlinks: {requested}")
    for name, anchor in anchors.items():
        if resolved.is_relative_to(anchor):
            raise LabError(f"slot root must be outside repository anchor {name}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


@contextmanager
def _slot_lock(slot_root: Path, slot_id: str) -> Iterator[None]:
    lock_root = slot_root / ".locks"
    if lock_root.is_symlink():
        raise LabError(f"slot lock root is a symlink: {lock_root}")
    lock_root.mkdir(parents=True, exist_ok=True)
    path = lock_root / f"{slot_id}.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LabError(f"slot materialization lock is held: {slot_id}") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _anchor_fingerprint(repository: Path) -> dict[str, object]:
    submodules, submodules_exact = git.submodule_status(repository)
    return {
        "branch": git.current_branch(repository),
        "commit": git.current_commit(repository),
        "status": list(git.status_short(repository)),
        "submodules": submodules,
        "submodules_exact": submodules_exact,
    }


def _configure_source_checkout(repository: Path) -> None:
    _run_git(repository, ["config", "lfs.fetchexclude", "*"])
    _run_git(repository, ["config", "lfs.fetchinclude", ""])
    _run_git(repository, ["config", "remote.origin.skipDefaultUpdate", "true"])


def _clone_detached(
    source: Path,
    target: Path,
    commit: str,
) -> None:
    if target.exists() or target.is_symlink():
        raise LabError(f"slot checkout target already exists: {target}")
    _run_git(
        None,
        [
            "clone",
            "--shared",
            "--no-checkout",
            "--no-tags",
            str(source),
            str(target),
        ],
    )
    _configure_source_checkout(target)
    _run_git(target, ["checkout", "--detach", "--force", commit])
    _run_git(target, ["remote", "remove", "origin"])


def _materialize_submodules(
    anchor: Path,
    checkout: Path,
    expected: Mapping[str, str],
) -> None:
    actual, exact = git.submodule_status(anchor)
    if not exact or actual != dict(expected):
        raise LabError(f"repository anchor has inexact submodules: {anchor}")
    materialized_paths: list[str] = []
    for relative_path, commit in sorted(
        expected.items(),
        key=lambda item: (len(Path(item[0]).parts), item[0]),
    ):
        source = (anchor / relative_path).resolve()
        if not source.is_relative_to(anchor) or git.repository_root(source) != source:
            raise LabError(f"invalid submodule anchor: {relative_path}")
        if source != (anchor / relative_path).absolute():
            raise LabError(f"submodule anchor uses a symlink: {relative_path}")
        if git.current_commit(source) != commit or git.status_short(source):
            raise LabError(f"submodule anchor is not exact and clean: {relative_path}")
        target = checkout / relative_path
        if target.is_symlink():
            raise LabError(f"slot submodule target is a symlink: {relative_path}")
        if target.is_dir():
            try:
                target.rmdir()
            except OSError as exc:
                raise LabError(
                    f"slot submodule target is unexpectedly non-empty: {relative_path}"
                ) from exc
        _clone_detached(source, target, commit)
        relative = PurePosixPath(relative_path)
        parent_path = max(
            (
                candidate
                for candidate in materialized_paths
                if relative.is_relative_to(PurePosixPath(candidate))
            ),
            key=lambda value: len(PurePosixPath(value).parts),
            default=None,
        )
        if parent_path is None:
            parent_repository = checkout
            local_path = relative_path
        else:
            parent_repository = checkout / parent_path
            local_path = str(relative.relative_to(PurePosixPath(parent_path)))
        submodule_name = _submodule_name_for_path(parent_repository, local_path)
        _run_git(
            parent_repository,
            ["config", f"submodule.{submodule_name}.active", "true"],
        )
        _run_git(
            parent_repository,
            ["config", f"submodule.{submodule_name}.url", str(source)],
        )
        _run_git(
            parent_repository,
            ["submodule", "absorbgitdirs", "--", local_path],
        )
        materialized_paths.append(relative_path)
    actual, exact = git.submodule_status(checkout)
    if not exact or actual != dict(expected):
        raise LabError(
            "materialized checkout has inexact submodules: "
            f"{checkout} (actual={actual}, exact={exact}, expected={dict(expected)})"
        )


def _materialized_ref(bundle_set_id: str, repository: str) -> str:
    suffix = hashlib.sha256(repository.encode()).hexdigest()[:16]
    return f"refs/lab/materialized/{bundle_set_id}/{suffix}"


def _submodule_name_for_path(repository: Path, path: str) -> str:
    output = _run_git(
        repository,
        ["config", "-f", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$"],
    )
    matches: list[str] = []
    for line in output.splitlines():
        key, separator, value = line.partition(" ")
        if (
            separator
            and value == path
            and key.startswith("submodule.")
            and key.endswith(".path")
        ):
            matches.append(key[len("submodule.") : -len(".path")])
    if len(matches) != 1:
        raise LabError(f"cannot resolve unique submodule name for path: {path}")
    return matches[0]


def _verify_live_materialization(
    target: Path,
    manifest: Mapping[str, object],
    receipt: Mapping[str, object],
    *,
    slot_id: str,
) -> None:
    expected_content = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if (
        receipt.get("schema") != DRAFT_SLOT_SCHEMA_ID
        or receipt.get("bundle_set_id") != manifest.get("bundle_set_id")
        or receipt.get("snapshot_id") != manifest.get("snapshot_id")
        or receipt.get("task_id") != manifest.get("task_id")
        or receipt.get("slot_id") != slot_id
        or receipt.get("materialization_root") != str(target)
        or hashlib.sha256(_canonical_json(expected_content)).hexdigest()
        != receipt.get("receipt_sha256")
        or receipt.get("source_verified") is not True
        or receipt.get("lfs_smudge_disabled") is not True
        or receipt.get("execution_eligible") is not False
        or receipt.get("remote_run_eligible") is not False
        or receipt.get("verify_eligible") is not False
        or receipt.get("device_eligible") is not False
    ):
        raise LabError(f"invalid draft slot materialization receipt: {target}")
    manifest_repositories = manifest.get("repositories")
    receipt_repositories = receipt.get("repositories")
    if (
        not isinstance(manifest_repositories, Mapping)
        or not isinstance(receipt_repositories, Mapping)
        or set(manifest_repositories) != set(receipt_repositories)
    ):
        raise LabError("draft slot repository receipt mapping mismatch")
    for name, raw_manifest_entry in manifest_repositories.items():
        if not isinstance(name, str):
            raise LabError("invalid materialized repository name")
        entry = _repository_entry(name, raw_manifest_entry)
        raw_receipt_entry = receipt_repositories.get(name)
        if not isinstance(raw_receipt_entry, Mapping):
            raise LabError(f"missing slot repository receipt: {name}")
        checkout = (target / "repositories" / name).resolve()
        if (
            raw_receipt_entry.get("path") != str(checkout)
            or raw_receipt_entry.get("commit") != entry["commit"]
            or raw_receipt_entry.get("tree_oid") != entry["tree_oid"]
            or raw_receipt_entry.get("submodules") != entry["submodules"]
            or raw_receipt_entry.get("detached") is not True
            or raw_receipt_entry.get("clean") is not True
        ):
            raise LabError(f"draft slot repository receipt mismatch: {name}")
        if git.repository_root(checkout) != checkout:
            raise LabError(f"slot repository root mismatch: {name}")
        if (
            git.current_branch(checkout) is not None
            or git.current_commit(checkout) != entry["commit"]
            or git.tree_oid(checkout) != entry["tree_oid"]
            or git.status_short(checkout)
            or _run_git(checkout, ["config", "--get", "lfs.fetchexclude"]) != "*"
        ):
            raise LabError(f"materialized slot repository drifted: {name}")
        submodules, exact = git.submodule_status(checkout)
        if not exact or submodules != entry["submodules"]:
            raise LabError(f"materialized slot submodules drifted: {name}")
        reference = _materialized_ref(str(manifest["bundle_set_id"]), name)
        if git.resolve_commit(checkout, reference) != entry["commit"]:
            raise LabError(f"materialized slot ref drifted: {name}")


def materialize_draft_bundle_set(
    manifest_path: Path,
    *,
    repository_paths: Mapping[str, Path],
    slot_root: Path,
    slot_id: str,
) -> dict[str, object]:
    """Materialize an already-verified draft into an inert, detached slot source tree."""

    validate_task_id(slot_id)
    requested_manifest = manifest_path.expanduser()
    if requested_manifest.is_symlink():
        raise LabError(f"bundle manifest is a symlink: {requested_manifest}")
    manifest_path = requested_manifest.resolve()
    manifest, _ = _read_json_object(manifest_path, label="bundle manifest")
    if manifest.get("schema") != BUNDLE_SCHEMA_ID:
        raise LabError("unsupported draft bundle schema")
    manifest_repositories = manifest.get("repositories")
    if not isinstance(manifest_repositories, Mapping):
        raise LabError("draft bundle has no repositories")
    anchors: dict[str, Path] = {}
    for name, path in repository_paths.items():
        validate_repository_name(name)
        anchors[name] = _real_repository_anchor(path, name=name)
    if set(anchors) != set(manifest_repositories):
        raise LabError("bundle repository mapping mismatch")
    slot_root = _safe_slot_root(slot_root, anchors)
    verification = verify_draft_bundle_set(
        manifest_path,
        repository_paths=anchors,
        require_synthetic_absent=True,
    )
    bundle_set_id = str(manifest.get("bundle_set_id", ""))
    validate_task_id(bundle_set_id)
    slot = slot_root / slot_id
    if slot.is_symlink():
        raise LabError(f"slot path is a symlink: {slot}")
    materializations = slot / "materialized"
    target = materializations / bundle_set_id

    with _slot_lock(slot_root, slot_id):
        if materializations.is_symlink() or target.is_symlink():
            raise LabError(f"slot materialization path is a symlink: {target}")
        materializations.mkdir(parents=True, exist_ok=True)
        if target.exists():
            receipt, _ = _read_json_object(
                target / "materialization.json",
                label="slot materialization receipt",
            )
            _verify_live_materialization(
                target,
                manifest,
                receipt,
                slot_id=slot_id,
            )
            return receipt

        anchor_fingerprints = {
            name: _anchor_fingerprint(anchor) for name, anchor in anchors.items()
        }
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{bundle_set_id}.", dir=materializations)
        )
        try:
            temporary_repositories = temporary / "repositories"
            temporary_repositories.mkdir()
            repository_receipts: dict[str, object] = {}
            for name, raw_entry in manifest_repositories.items():
                if not isinstance(name, str):
                    raise LabError("invalid materialized repository name")
                entry = _repository_entry(name, raw_entry)
                raw_submodules = entry["submodules"]
                if not isinstance(raw_submodules, Mapping):
                    raise LabError(f"invalid materialized submodules: {name}")
                submodules = {
                    str(path): str(commit) for path, commit in raw_submodules.items()
                }
                checkout = temporary_repositories / name
                _clone_detached(anchors[name], checkout, str(entry["base_commit"]))
                if entry["synthetic"]:
                    bundle_path = _resolve_bundle_path(
                        manifest_path.parent,
                        raw_entry.get("bundle_file"),
                        repository=name,
                    )
                    reference = _materialized_ref(bundle_set_id, name)
                    _run_git(
                        checkout,
                        [
                            "fetch",
                            "--no-tags",
                            "--no-write-fetch-head",
                            str(bundle_path),
                            f"{entry['ref']}:{reference}",
                        ],
                    )
                    _run_git(
                        checkout,
                        ["checkout", "--detach", "--force", str(entry["commit"])],
                    )
                else:
                    reference = _materialized_ref(bundle_set_id, name)
                    _run_git(
                        checkout,
                        ["update-ref", reference, str(entry["commit"])],
                    )
                _materialize_submodules(
                    anchors[name],
                    checkout,
                    submodules,
                )
                if (
                    git.current_branch(checkout) is not None
                    or git.current_commit(checkout) != entry["commit"]
                    or git.tree_oid(checkout) != entry["tree_oid"]
                    or git.status_short(checkout)
                ):
                    raise LabError(f"slot checkout identity mismatch: {name}")
                repository_receipts[name] = {
                    "path": str(target / "repositories" / name),
                    "commit": entry["commit"],
                    "tree_oid": entry["tree_oid"],
                    "submodules": submodules,
                    "detached": True,
                    "clean": True,
                    "large_file_smudge": False,
                    "depends_on_anchor_objects": True,
                }

            for name, anchor in anchors.items():
                if _anchor_fingerprint(anchor) != anchor_fingerprints[name]:
                    raise LabError(
                        f"repository anchor changed during materialization: {name}"
                    )
            content: dict[str, object] = {
                "schema": DRAFT_SLOT_SCHEMA_ID,
                "bundle_set_id": bundle_set_id,
                "task_id": manifest.get("task_id"),
                "snapshot_id": manifest.get("snapshot_id"),
                "slot_id": slot_id,
                "materialization_root": str(target),
                "bundle_verification_receipt_sha256": verification.get(
                    "receipt_sha256"
                ),
                "repositories": repository_receipts,
                "source_verified": True,
                "source_anchor_mutated": False,
                "lfs_smudge_disabled": True,
                "execution_eligible": False,
                "remote_run_eligible": False,
                "verify_eligible": False,
                "device_eligible": False,
            }
            receipt = {
                **content,
                "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
            }
            _atomic_write_text(
                temporary / "materialization.json",
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            )
            temporary.rename(target)
            _verify_live_materialization(
                target,
                manifest,
                receipt,
                slot_id=slot_id,
            )
            return receipt
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
