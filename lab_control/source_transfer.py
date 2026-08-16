from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from . import git
from .request_transport import (
    AUTHENTICATION_SCHEME,
    RequestEnvelopeIngress,
    build_request_envelope,
    validate_request_envelope,
)
from .run_requests import RunRequestStore
from .workspace import LabError, validate_repository_name, validate_task_id

SOURCE_PACKAGE_SCHEMA_ID = "lab.source-package.v1"
SOURCE_INGRESS_RECEIPT_SCHEMA_ID = "lab.source-ingress-receipt.v1"
SOURCE_PACKAGE_MANIFEST = "source-package.json"
REQUEST_ENVELOPE_FILE = "request-envelope.json"

_MINIMUM_SECRET_BYTES = 32
_MAXIMUM_MANIFEST_BYTES = 2 * 1024 * 1024
_DEFAULT_MAXIMUM_BLOB_BYTES = 16 * 1024 * 1024
_DEFAULT_MAXIMUM_RAW_BYTES = 128 * 1024 * 1024
_DEFAULT_MAXIMUM_PACK_BYTES = 64 * 1024 * 1024
_DEFAULT_MAXIMUM_OBJECTS = 200_000
_DEFAULT_MAXIMUM_REPOSITORIES = 32


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _secret(value: object, *, key_id: str) -> bytes:
    if not isinstance(value, bytes) or len(value) < _MINIMUM_SECRET_BYTES:
        raise LabError(
            f"transport key {key_id} must contain at least "
            f"{_MINIMUM_SECRET_BYTES} bytes"
        )
    return value


def _write_new_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_json_file(path: Path, *, maximum_bytes: int) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise LabError(f"source package file is missing or unsafe: {path}")
    size = path.stat().st_size
    if size > maximum_bytes:
        raise LabError(f"source package file exceeds {maximum_bytes} bytes: {path}")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read source package file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabError(f"source package document must be an object: {path}")
    return payload


def _safe_file(root: Path, relative_path: object, *, maximum_bytes: int) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise LabError("source package contains an invalid relative path")
    relative = PurePosixPath(relative_path)
    if (
        relative_path.startswith(("/", "~"))
        or "\\" in relative_path
        or "\x00" in relative_path
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise LabError(f"source package contains an unsafe path: {relative_path!r}")
    path = root.joinpath(*relative.parts)
    if not path.resolve(strict=False).is_relative_to(root.resolve()):
        raise LabError(f"source package path escapes its root: {relative_path!r}")
    if path.is_symlink() or not path.is_file():
        raise LabError(f"source package file is missing or unsafe: {relative_path}")
    if path.stat().st_size > maximum_bytes:
        raise LabError(
            f"source package file exceeds {maximum_bytes} bytes: {relative_path}"
        )
    return path


def _catalog(catalog: Mapping[str, Path]) -> dict[str, Path]:
    normalized: dict[str, Path] = {}
    for raw_name, raw_path in catalog.items():
        name = validate_repository_name(str(raw_name))
        path = Path(raw_path).expanduser().resolve()
        if name in normalized:
            raise LabError(f"duplicate source package repository: {name}")
        normalized[name] = path
    if not normalized:
        raise LabError("source package repository catalog cannot be empty")
    return normalized


def _git_object_inventory(
    repository: Path,
    *,
    target_commit: str,
    base_commit: str,
    maximum_blob_bytes: int,
    maximum_raw_bytes: int,
    maximum_objects: int,
) -> tuple[tuple[str, ...], int]:
    ancestry = git.run_git(
        repository,
        ["merge-base", "--is-ancestor", base_commit, target_commit],
        check=False,
    )
    if ancestry.returncode != 0:
        raise LabError(
            f"source package base is not an ancestor of target in {repository}: "
            f"{base_commit} -> {target_commit}"
        )
    listing = git.run_git(
        repository,
        ["rev-list", "--objects", target_commit, f"^{base_commit}"],
    )
    object_ids = tuple(
        line.split(" ", 1)[0] for line in listing.stdout.splitlines() if line
    )
    if len(object_ids) > maximum_objects:
        raise LabError(
            f"source package object count exceeds {maximum_objects}: {repository}"
        )
    if not object_ids:
        raise LabError(f"source package target differs but has no objects: {repository}")
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repository),
            "cat-file",
            "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        ],
        input="".join(f"{oid}\n" for oid in object_ids),
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"},
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise LabError(f"cannot inspect source package objects: {detail}")
    total_bytes = 0
    observed: list[str] = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3 or fields[1] not in {"blob", "commit", "tree", "tag"}:
            raise LabError(f"cannot parse source package object: {line!r}")
        try:
            size = int(fields[2])
        except ValueError as exc:
            raise LabError(f"cannot parse source package object size: {line!r}") from exc
        if fields[1] == "blob" and size > maximum_blob_bytes:
            raise LabError(
                f"source package blob exceeds {maximum_blob_bytes} bytes: {fields[0]}"
            )
        total_bytes += size
        if total_bytes > maximum_raw_bytes:
            raise LabError(
                f"source package objects exceed {maximum_raw_bytes} bytes: {repository}"
            )
        observed.append(fields[0])
    if tuple(observed) != object_ids:
        raise LabError("source package object inventory changed while being inspected")
    return object_ids, total_bytes


def _write_pack(repository: Path, object_ids: tuple[str, ...], path: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            completed = subprocess.run(
                [
                    "git",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-C",
                    str(repository),
                    "pack-objects",
                    "--stdout",
                ],
                input="".join(f"{oid}\n" for oid in object_ids).encode(),
                stdout=stream,
                stderr=subprocess.PIPE,
                check=False,
                env={
                    **os.environ,
                    "GIT_LFS_SKIP_SMUDGE": "1",
                    "GIT_TERMINAL_PROMPT": "0",
                },
            )
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise LabError(f"cannot create source object pack: {detail}")


class SourcePackageBuilder:
    """Build one authenticated, bounded Git-object package without networking."""

    def __init__(
        self,
        *,
        repository_catalog: Mapping[str, Path],
        maximum_blob_bytes: int = _DEFAULT_MAXIMUM_BLOB_BYTES,
        maximum_raw_bytes: int = _DEFAULT_MAXIMUM_RAW_BYTES,
        maximum_pack_bytes: int = _DEFAULT_MAXIMUM_PACK_BYTES,
        maximum_objects: int = _DEFAULT_MAXIMUM_OBJECTS,
        maximum_repositories: int = _DEFAULT_MAXIMUM_REPOSITORIES,
    ) -> None:
        self.repository_catalog = _catalog(repository_catalog)
        self.maximum_blob_bytes = maximum_blob_bytes
        self.maximum_raw_bytes = maximum_raw_bytes
        self.maximum_pack_bytes = maximum_pack_bytes
        self.maximum_objects = maximum_objects
        self.maximum_repositories = maximum_repositories

    def build(
        self,
        request: Mapping[str, object],
        destination: Path,
        *,
        base_commits: Mapping[str, str],
        sender_id: str,
        recipient_id: str,
        key_id: str,
        secret: bytes,
        ttl_seconds: int = 3600,
    ) -> dict[str, object]:
        sender_id = validate_task_id(sender_id)
        recipient_id = validate_task_id(recipient_id)
        key_id = validate_task_id(key_id)
        secret = _secret(secret, key_id=key_id)
        envelope = build_request_envelope(
            request,
            sender_id=sender_id,
            recipient_id=recipient_id,
            key_id=key_id,
            secret=secret,
            ttl_seconds=ttl_seconds,
        )
        validated_request, _ = validate_request_envelope(
            envelope,
            recipient_id=recipient_id,
            trusted_sender_keys={sender_id: {key_id: secret}},
        )
        snapshot = validated_request.get("snapshot")
        raw_repositories = snapshot.get("repositories") if isinstance(snapshot, dict) else None
        if not isinstance(raw_repositories, dict) or not raw_repositories:
            raise LabError("source package request lacks repositories")
        if len(raw_repositories) > self.maximum_repositories:
            raise LabError(
                f"source package repository count exceeds {self.maximum_repositories}"
            )
        normalized_bases = {
            validate_repository_name(str(name)): str(commit)
            for name, commit in base_commits.items()
        }
        if set(normalized_bases) != set(raw_repositories):
            raise LabError("source package requires exactly one receiver base per repository")

        destination = destination.expanduser().resolve()
        if destination.exists() or destination.is_symlink():
            raise LabError(f"source package destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.parent.is_symlink():
            raise LabError(f"source package parent must not be a symlink: {destination.parent}")
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            envelope_path = staging / REQUEST_ENVELOPE_FILE
            _write_new_json(envelope_path, envelope)
            objects_root = staging / "objects"
            objects_root.mkdir()
            records: dict[str, object] = {}
            for name, raw_repository in sorted(raw_repositories.items()):
                repository_name = validate_repository_name(str(name))
                if not isinstance(raw_repository, dict):
                    raise LabError(f"invalid source package repository: {repository_name}")
                repository = self.repository_catalog.get(repository_name)
                if repository is None:
                    raise LabError(f"source package repository is unavailable: {repository_name}")
                try:
                    canonical_identity = git.origin_identity(repository)
                    target_commit = git.resolve_commit(
                        repository, str(raw_repository.get("commit", ""))
                    )
                    target_tree = git.tree_oid(repository, target_commit)
                    target_submodules = git.submodule_gitlinks(repository, target_commit)
                    base_commit = git.resolve_commit(
                        repository, normalized_bases[repository_name]
                    )
                except git.GitError as exc:
                    raise LabError(str(exc)) from exc
                if (
                    target_commit != raw_repository.get("commit")
                    or target_tree != raw_repository.get("tree_oid")
                    or target_submodules != raw_repository.get("submodules")
                ):
                    raise LabError(
                        f"source package repository identity changed: {repository_name}"
                    )
                record: dict[str, object] = {
                    "canonical_identity": canonical_identity,
                    "target_commit": target_commit,
                    "target_tree_oid": target_tree,
                    "target_submodules": target_submodules,
                    "base_commit": base_commit,
                    "transfer_mode": "present" if base_commit == target_commit else "pack",
                    "object_count": 0,
                    "raw_object_bytes": 0,
                    "pack_path": None,
                    "pack_sha256": None,
                    "pack_bytes": 0,
                }
                if base_commit != target_commit:
                    object_ids, raw_bytes = _git_object_inventory(
                        repository,
                        target_commit=target_commit,
                        base_commit=base_commit,
                        maximum_blob_bytes=self.maximum_blob_bytes,
                        maximum_raw_bytes=self.maximum_raw_bytes,
                        maximum_objects=self.maximum_objects,
                    )
                    pack_relative = f"objects/{repository_name}.pack"
                    pack_path = staging / pack_relative
                    _write_pack(repository, object_ids, pack_path)
                    pack_bytes = pack_path.stat().st_size
                    if pack_bytes > self.maximum_pack_bytes:
                        raise LabError(
                            f"source object pack exceeds {self.maximum_pack_bytes} bytes: "
                            f"{repository_name}"
                        )
                    record.update(
                        {
                            "object_count": len(object_ids),
                            "raw_object_bytes": raw_bytes,
                            "pack_path": pack_relative,
                            "pack_sha256": _sha256_file(pack_path),
                            "pack_bytes": pack_bytes,
                        }
                    )
                records[repository_name] = record

            content: dict[str, object] = {
                "schema": SOURCE_PACKAGE_SCHEMA_ID,
                "message_type": "source-package",
                "sender_id": sender_id,
                "recipient_id": recipient_id,
                "created_at": _utc_now(),
                "request_id": validated_request["request_id"],
                "idempotency_key": validated_request["idempotency_key"],
                "envelope_path": REQUEST_ENVELOPE_FILE,
                "envelope_sha256": _sha256_file(envelope_path),
                "repositories": records,
            }
            package_sha256 = _sha256_bytes(_canonical_json(content))
            authenticated = {
                **content,
                "package_id": f"SP-{package_sha256[:24]}",
                "package_sha256": package_sha256,
                "authentication": {
                    "scheme": AUTHENTICATION_SCHEME,
                    "key_id": key_id,
                },
            }
            manifest = {
                **authenticated,
                "authentication_mac": hmac.new(
                    secret, _canonical_json(authenticated), hashlib.sha256
                ).hexdigest(),
            }
            _write_new_json(staging / SOURCE_PACKAGE_MANIFEST, manifest)
            os.replace(staging, destination)
            return manifest
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise


class SourcePackageIngress:
    """Authenticate and add exact objects to local repositories, then ingest a request."""

    def __init__(
        self,
        store: RunRequestStore,
        *,
        repository_catalog: Mapping[str, Path],
        recipient_id: str,
        trusted_sender_keys: Mapping[str, Mapping[str, bytes]],
        maximum_raw_bytes: int = _DEFAULT_MAXIMUM_RAW_BYTES,
        maximum_pack_bytes: int = _DEFAULT_MAXIMUM_PACK_BYTES,
        maximum_objects: int = _DEFAULT_MAXIMUM_OBJECTS,
        maximum_repositories: int = _DEFAULT_MAXIMUM_REPOSITORIES,
    ) -> None:
        self.store = store
        self.repository_catalog = _catalog(repository_catalog)
        self.recipient_id = validate_task_id(recipient_id)
        self.trusted_sender_keys = trusted_sender_keys
        self.maximum_raw_bytes = maximum_raw_bytes
        self.maximum_pack_bytes = maximum_pack_bytes
        self.maximum_objects = maximum_objects
        self.maximum_repositories = maximum_repositories

    def _validate_manifest(self, payload: object) -> tuple[dict[str, object], bytes]:
        expected = {
            "schema",
            "message_type",
            "sender_id",
            "recipient_id",
            "created_at",
            "request_id",
            "idempotency_key",
            "envelope_path",
            "envelope_sha256",
            "repositories",
            "package_id",
            "package_sha256",
            "authentication",
            "authentication_mac",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise LabError("source package manifest has an unexpected shape")
        sender_id = validate_task_id(str(payload.get("sender_id", "")))
        if (
            payload.get("schema") != SOURCE_PACKAGE_SCHEMA_ID
            or payload.get("message_type") != "source-package"
            or payload.get("recipient_id") != self.recipient_id
        ):
            raise LabError("source package identity mismatch")
        authentication = payload.get("authentication")
        if not isinstance(authentication, dict) or set(authentication) != {
            "scheme",
            "key_id",
        }:
            raise LabError("source package authentication has an unexpected shape")
        key_id = validate_task_id(str(authentication.get("key_id", "")))
        if authentication.get("scheme") != AUTHENTICATION_SCHEME:
            raise LabError("unsupported source package authentication scheme")
        sender_keys = self.trusted_sender_keys.get(sender_id)
        if sender_keys is None or key_id not in sender_keys:
            raise LabError(f"source package sender/key is not trusted: {sender_id}/{key_id}")
        secret = _secret(sender_keys[key_id], key_id=key_id)
        content = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "package_id",
                "package_sha256",
                "authentication",
                "authentication_mac",
            }
        }
        package_sha256 = _sha256_bytes(_canonical_json(content))
        if (
            payload.get("package_sha256") != package_sha256
            or payload.get("package_id") != f"SP-{package_sha256[:24]}"
        ):
            raise LabError("source package manifest hash mismatch")
        authenticated = {
            **content,
            "package_id": payload["package_id"],
            "package_sha256": package_sha256,
            "authentication": authentication,
        }
        mac = payload.get("authentication_mac")
        if not isinstance(mac, str) or not hmac.compare_digest(
            mac,
            hmac.new(secret, _canonical_json(authenticated), hashlib.sha256).hexdigest(),
        ):
            raise LabError("source package authentication failed")
        repositories = payload.get("repositories")
        if (
            not isinstance(repositories, dict)
            or not repositories
            or len(repositories) > self.maximum_repositories
        ):
            raise LabError("source package has an invalid repository set")
        return payload, secret

    def ingest(self, package_root: Path) -> dict[str, object]:
        package_root = package_root.expanduser().resolve()
        if package_root.is_symlink() or not package_root.is_dir():
            raise LabError(f"source package root is missing or unsafe: {package_root}")
        manifest = self._validate_manifest(
            _read_json_file(
                package_root / SOURCE_PACKAGE_MANIFEST,
                maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
            )
        )[0]
        envelope_path = _safe_file(
            package_root,
            manifest.get("envelope_path"),
            maximum_bytes=_MAXIMUM_MANIFEST_BYTES,
        )
        if _sha256_file(envelope_path) != manifest.get("envelope_sha256"):
            raise LabError("source package request envelope hash mismatch")
        envelope = _read_json_file(envelope_path, maximum_bytes=_MAXIMUM_MANIFEST_BYTES)
        request, transport = validate_request_envelope(
            envelope,
            recipient_id=self.recipient_id,
            trusted_sender_keys=self.trusted_sender_keys,
        )
        if (
            request.get("request_id") != manifest.get("request_id")
            or request.get("idempotency_key") != manifest.get("idempotency_key")
        ):
            raise LabError("source package request identity mismatch")
        snapshot = request.get("snapshot")
        request_repositories = snapshot.get("repositories") if isinstance(snapshot, dict) else None
        package_repositories = manifest.get("repositories")
        if (
            not isinstance(request_repositories, dict)
            or not isinstance(package_repositories, dict)
            or set(request_repositories) != set(package_repositories)
        ):
            raise LabError("source package repositories differ from the request")

        imported: dict[str, object] = {}
        for name, raw_record in sorted(package_repositories.items()):
            repository_name = validate_repository_name(str(name))
            expected_record_fields = {
                "canonical_identity",
                "target_commit",
                "target_tree_oid",
                "target_submodules",
                "base_commit",
                "transfer_mode",
                "object_count",
                "raw_object_bytes",
                "pack_path",
                "pack_sha256",
                "pack_bytes",
            }
            if not isinstance(raw_record, dict) or set(raw_record) != expected_record_fields:
                raise LabError(f"source package repository record is invalid: {repository_name}")
            object_count = raw_record.get("object_count")
            raw_object_bytes = raw_record.get("raw_object_bytes")
            if (
                not isinstance(object_count, int)
                or isinstance(object_count, bool)
                or not 0 <= object_count <= self.maximum_objects
                or not isinstance(raw_object_bytes, int)
                or isinstance(raw_object_bytes, bool)
                or not 0 <= raw_object_bytes <= self.maximum_raw_bytes
            ):
                raise LabError(
                    f"source package object bounds are invalid: {repository_name}"
                )
            request_identity = request_repositories[repository_name]
            if not isinstance(request_identity, dict):
                raise LabError(f"source package request repository is invalid: {repository_name}")
            if (
                raw_record.get("target_commit") != request_identity.get("commit")
                or raw_record.get("target_tree_oid") != request_identity.get("tree_oid")
                or raw_record.get("target_submodules") != request_identity.get("submodules")
            ):
                raise LabError(f"source package target differs from request: {repository_name}")
            repository = self.repository_catalog.get(repository_name)
            if repository is None:
                raise LabError(f"source package repository is unavailable: {repository_name}")
            try:
                if git.origin_identity(repository) != raw_record.get("canonical_identity"):
                    raise LabError(
                        f"source package canonical identity mismatch: {repository_name}"
                    )
                base_commit = git.resolve_commit(repository, str(raw_record.get("base_commit", "")))
            except git.GitError as exc:
                raise LabError(
                    f"source package receiver lacks its declared base for {repository_name}: {exc}"
                ) from exc
            transfer_mode = raw_record.get("transfer_mode")
            if transfer_mode == "pack":
                if base_commit == raw_record.get("target_commit"):
                    raise LabError(f"source package has a redundant pack: {repository_name}")
                pack_bytes = raw_record.get("pack_bytes")
                if (
                    not isinstance(pack_bytes, int)
                    or isinstance(pack_bytes, bool)
                    or not 0 < pack_bytes <= self.maximum_pack_bytes
                ):
                    raise LabError(f"source package pack size is invalid: {repository_name}")
                pack_path = _safe_file(
                    package_root,
                    raw_record.get("pack_path"),
                    maximum_bytes=self.maximum_pack_bytes,
                )
                if (
                    pack_path.stat().st_size != pack_bytes
                    or _sha256_file(pack_path) != raw_record.get("pack_sha256")
                ):
                    raise LabError(f"source package object pack hash mismatch: {repository_name}")
                with pack_path.open("rb") as stream:
                    completed = subprocess.run(
                        [
                            "git",
                            "-c",
                            "core.hooksPath=/dev/null",
                            "-C",
                            str(repository),
                            "index-pack",
                            "--stdin",
                            "--strict",
                        ],
                        stdin=stream,
                        check=False,
                        capture_output=True,
                        env={
                            **os.environ,
                            "GIT_LFS_SKIP_SMUDGE": "1",
                            "GIT_TERMINAL_PROMPT": "0",
                        },
                    )
                if completed.returncode != 0:
                    detail = completed.stderr.decode(errors="replace").strip()
                    raise LabError(f"cannot import source object pack: {detail}")
            elif transfer_mode == "present":
                if any(
                    raw_record.get(field) not in {None, 0}
                    for field in ("pack_path", "pack_sha256", "pack_bytes", "object_count", "raw_object_bytes")
                ):
                    raise LabError(f"present source package record has pack data: {repository_name}")
            else:
                raise LabError(f"source package transfer mode is invalid: {repository_name}")
            try:
                target_commit = git.resolve_commit(
                    repository, str(raw_record.get("target_commit", ""))
                )
                target_tree = git.tree_oid(repository, target_commit)
                target_submodules = git.submodule_gitlinks(repository, target_commit)
            except git.GitError as exc:
                raise LabError(f"imported source identity is unavailable: {exc}") from exc
            if (
                target_tree != raw_record.get("target_tree_oid")
                or target_submodules != raw_record.get("target_submodules")
            ):
                raise LabError(f"imported source identity mismatch: {repository_name}")
            ancestry = git.run_git(
                repository,
                ["merge-base", "--is-ancestor", base_commit, target_commit],
                check=False,
            )
            if ancestry.returncode != 0:
                raise LabError(
                    f"imported source does not descend from its declared base: "
                    f"{repository_name}"
                )
            imported[repository_name] = {
                "commit": target_commit,
                "tree_oid": target_tree,
                "transfer_mode": transfer_mode,
                "pack_sha256": raw_record.get("pack_sha256"),
            }

        ingress = RequestEnvelopeIngress(
            self.store,
            recipient_id=self.recipient_id,
            trusted_sender_keys=self.trusted_sender_keys,
        ).ingest(envelope)
        request_id = str(request["request_id"])
        if self.store.has_attachment(request_id, "source-package-ingress"):
            stored = self.store.load_attachment(request_id, "source-package-ingress")
            if (
                stored.get("schema") != SOURCE_INGRESS_RECEIPT_SCHEMA_ID
                or stored.get("request_id") != request_id
                or stored.get("package_id") != manifest.get("package_id")
                or stored.get("package_sha256") != manifest.get("package_sha256")
                or stored.get("repositories") != imported
            ):
                raise LabError("source package ingress receipt collision")
            return {
                "schema": "lab.source-ingress-result.v1",
                "request_id": request_id,
                "created": bool(ingress["created"]),
                "receipt_created": False,
                "state": ingress["state"],
                "receipt": stored,
            }
        receipt_content: dict[str, object] = {
            "schema": SOURCE_INGRESS_RECEIPT_SCHEMA_ID,
            "request_id": request_id,
            "idempotency_key": request["idempotency_key"],
            "package_id": manifest["package_id"],
            "package_sha256": manifest["package_sha256"],
            "envelope_id": transport["envelope_id"],
            "sender_id": transport["sender_id"],
            "recipient_id": transport["recipient_id"],
            "repositories": imported,
            "ingested_at": _utc_now(),
        }
        receipt = {
            **receipt_content,
            "receipt_sha256": _sha256_bytes(_canonical_json(receipt_content)),
        }
        stored, created = self.store.put_attachment(
            request_id, "source-package-ingress", receipt
        )
        return {
            "schema": "lab.source-ingress-result.v1",
            "request_id": request_id,
            "created": bool(ingress["created"]),
            "receipt_created": created,
            "state": ingress["state"],
            "receipt": stored,
        }
