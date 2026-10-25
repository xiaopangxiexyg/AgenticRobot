"""Persistence boundary for natively separated tactile episode bundles.

This store is subordinate to the authoritative collection writer.  It binds a
portable tactile storage root to one core dataset, publishes immutable episode
components, and mirrors one commit record to both roots.  It does not own the
episode lifecycle or append the core episode catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import uuid

from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.tactile_bundle import (
    CoreTactileEpisodeComponentV1,
    CoreTactileReferenceV1,
    TACTILE_BUNDLE_CORE_COMPONENT_FILE_DOMAIN_V1,
    TACTILE_BUNDLE_DATASET_BINDING_SCHEMA_V1,
    TACTILE_BUNDLE_DATASET_IDENTITY_FILE_DOMAIN_V1,
    TACTILE_BUNDLE_EPISODE_BINDING_SCHEMA_V1,
    TACTILE_BUNDLE_EPISODE_COMMIT_FILE_DOMAIN_V1,
    TACTILE_BUNDLE_EPISODE_IDENTITY_FILE_DOMAIN_V1,
    TACTILE_BUNDLE_LOGICAL_PREFIX_V1,
    TACTILE_BUNDLE_TACTILE_COMPONENT_FILE_DOMAIN_V1,
    TactileBundleArtifactV1,
    TactileBundleContractError,
    TactileBundleDatasetIdentityV1,
    TactileBundleDatasetLocatorV1,
    TactileEpisodeBundleCommitV1,
    TactileEpisodeComponentV1,
    TactileManifestBindingV1,
    core_dataset_identity_relative_path,
    core_episode_relative_dir,
    normalize_bundle_relative_path,
    storage_dataset_identity_relative_path,
    storage_episode_relative_dir,
    tactile_dataset_locator_relative_path,
    tactile_frame_alignment_hash,
)

from .atomic_json import atomic_write_json
from .immutable_canonical_store import (
    fsync_directory,
    publish_immutable_canonical,
)


class TactileBundleStorageError(RuntimeError):
    """Raised when separated tactile storage cannot be used safely."""


@dataclass(frozen=True)
class TactileBundleSealReport:
    dataset_uuid: str
    episode_uuid: str
    commit_hash: str
    core_component_hash: str
    tactile_component_hash: str
    core_commit_relative_path: str
    tactile_commit_relative_path: str
    artifact_count: int
    payload_bytes: int

    def episode_binding_payload(self) -> dict[str, Any]:
        return {
            "schema": TACTILE_BUNDLE_EPISODE_BINDING_SCHEMA_V1,
            "dataset_uuid": self.dataset_uuid,
            "episode_uuid": self.episode_uuid,
            "commit_hash": self.commit_hash,
            "core_commit_ref": self.core_commit_relative_path,
            "logical_prefix": TACTILE_BUNDLE_LOGICAL_PREFIX_V1,
            "tactile_required": True,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_canonical(path: Path, *, domain: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = canonical_loads(path.read_bytes(), expected_domain=domain)
    if not isinstance(payload, Mapping):
        raise TactileBundleStorageError(f"Canonical artifact is not a mapping: {path}")
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TactileBundleStorageError(
            f"Cannot read tactile manifest {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise TactileBundleStorageError(f"Tactile manifest is not an object: {path}")
    return payload


def _is_nonempty_episode_catalog(core_root: Path) -> bool:
    path = core_root / "meta" / "episodes.jsonl"
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            return any(line.strip() for line in handle)
    except OSError as exc:
        raise TactileBundleStorageError(
            f"Cannot inspect existing episode catalog: {path}: {exc}"
        ) from exc


def _assert_not_nested(left: Path, right: Path) -> None:
    if left == right or left.is_relative_to(right) or right.is_relative_to(left):
        raise TactileBundleStorageError(
            "core dataset root and tactile storage root must be disjoint"
        )


def _probe_writable_directory(path: Path) -> None:
    if not path.is_dir():
        raise TactileBundleStorageError(
            f"tactile storage root must already exist as a directory: {path}"
        )
    if path.is_symlink():
        raise TactileBundleStorageError(
            f"tactile storage root must not be a symlink: {path}"
        )
    fd, temporary_name = tempfile.mkstemp(
        prefix=".dexteleop-tactile-write-probe-",
        dir=str(path),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(b"dexteleop-tactile-bundle-probe-v1\n")
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(path)
    except OSError as exc:
        raise TactileBundleStorageError(
            f"tactile storage root is not durably writable: {path}: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _safe_resolve_under(root: Path, logical_path: str) -> Path:
    relative = normalize_bundle_relative_path(
        logical_path,
        field_name="logical_path",
        required_prefix=TACTILE_BUNDLE_LOGICAL_PREFIX_V1,
    )
    candidate = root / relative
    resolved_parent = candidate.parent.resolve()
    if not resolved_parent.is_relative_to(root.resolve()):
        raise TactileBundleStorageError(
            f"tactile artifact escapes bundle root: {logical_path}"
        )
    if candidate.is_symlink():
        raise TactileBundleStorageError(
            f"tactile artifact must not be a symlink: {logical_path}"
        )
    return candidate


class NativeTactileBundleStore:
    """Bind and publish one portable tactile bundle for a core dataset."""

    def __init__(
        self,
        *,
        core_dataset_root: str | Path,
        tactile_storage_root: str | Path,
        dataset_label: str,
        # Same-filesystem, disjoint roots are valid for live collection.  The
        # strict cross-device check remains an explicit offline/copy policy.
        require_distinct_device: bool = False,
        minimum_free_bytes: int = 0,
        uuid_factory=uuid.uuid4,
    ) -> None:
        configured_core_root = Path(core_dataset_root).expanduser()
        configured_core_root.mkdir(parents=True, exist_ok=True)
        if not configured_core_root.is_dir() or configured_core_root.is_symlink():
            raise TactileBundleStorageError(
                f"core dataset root must be a real directory: {configured_core_root}"
            )
        self.core_dataset_root = configured_core_root.resolve()
        configured_tactile_root = Path(tactile_storage_root).expanduser()
        _probe_writable_directory(configured_tactile_root)
        self.tactile_storage_root = configured_tactile_root.resolve()
        _assert_not_nested(self.core_dataset_root, self.tactile_storage_root)
        self._core_device_id = int(self.core_dataset_root.stat().st_dev)
        self._tactile_device_id = int(self.tactile_storage_root.stat().st_dev)
        if bool(require_distinct_device):
            if self._core_device_id == self._tactile_device_id:
                raise TactileBundleStorageError(
                    "native tactile separation requires a different filesystem device; "
                    "use the explicit test-only same-device override only in offline tests"
                )
        minimum = int(minimum_free_bytes)
        if minimum < 0:
            raise ValueError("minimum_free_bytes must be non-negative")
        free_bytes = int(shutil.disk_usage(self.tactile_storage_root).free)
        if free_bytes < minimum:
            raise TactileBundleStorageError(
                "tactile storage root has insufficient free space: "
                f"free={free_bytes} required={minimum}"
            )
        self.require_distinct_device = bool(require_distinct_device)
        self.minimum_free_bytes = minimum
        self.dataset_identity = self._bind_dataset_identity(
            dataset_label=str(dataset_label or "").strip(),
            uuid_factory=uuid_factory,
        )
        datasets_dir = self.tactile_storage_root / "datasets"
        if datasets_dir.is_symlink():
            raise TactileBundleStorageError(
                "tactile bundle datasets directory must not be a symlink"
            )
        datasets_dir.mkdir(exist_ok=True)
        if (
            not datasets_dir.is_dir()
            or not datasets_dir.resolve().is_relative_to(self.tactile_storage_root)
            or int(datasets_dir.stat().st_dev) != self._tactile_device_id
        ):
            raise TactileBundleStorageError(
                "tactile bundle datasets directory must remain on the configured "
                "storage filesystem"
            )
        bundle_dataset_path = datasets_dir / self.dataset_identity.dataset_uuid
        if bundle_dataset_path.is_symlink():
            raise TactileBundleStorageError(
                "tactile bundle dataset path must not be a symlink"
            )
        bundle_dataset_path.mkdir(exist_ok=True)
        if (
            not bundle_dataset_path.is_dir()
            or not bundle_dataset_path.resolve().is_relative_to(
                self.tactile_storage_root
            )
            or int(bundle_dataset_path.stat().st_dev) != self._tactile_device_id
        ):
            raise TactileBundleStorageError(
                "tactile bundle dataset path must remain directly below the storage "
                "root on the same filesystem"
            )
        self.bundle_dataset_root = bundle_dataset_path.resolve()
        self._bundle_device_id = int(self.bundle_dataset_root.stat().st_dev)
        self._publish_dataset_identity_to_storage()
        self._publish_dataset_locator()
        self._recover_abandoned_tail()

    @property
    def core_identity_path(self) -> Path:
        return self.core_dataset_root / core_dataset_identity_relative_path()

    @property
    def storage_identity_path(self) -> Path:
        return self.bundle_dataset_root / storage_dataset_identity_relative_path()

    @property
    def dataset_locator_path(self) -> Path:
        return self.tactile_storage_root / tactile_dataset_locator_relative_path(
            dataset_label=self.dataset_identity.dataset_label,
            dataset_uuid=self.dataset_identity.dataset_uuid,
        )

    @property
    def dataset_binding_payload(self) -> dict[str, Any]:
        return {
            "schema": TACTILE_BUNDLE_DATASET_BINDING_SCHEMA_V1,
            "dataset_uuid": self.dataset_identity.dataset_uuid,
            "dataset_identity_hash": self.dataset_identity.semantic_content_hash,
            "core_identity_ref": core_dataset_identity_relative_path(),
            "logical_prefix": self.dataset_identity.logical_prefix,
            "tactile_required": True,
            "portable_attachment_required": True,
        }

    def _bind_dataset_identity(
        self,
        *,
        dataset_label: str,
        uuid_factory,
    ) -> TactileBundleDatasetIdentityV1:
        if self.core_identity_path.exists():
            payload = _load_canonical(
                self.core_identity_path,
                domain=TACTILE_BUNDLE_DATASET_IDENTITY_FILE_DOMAIN_V1,
            )
            identity = TactileBundleDatasetIdentityV1.from_payload(payload)
            if dataset_label and identity.dataset_label != dataset_label:
                raise TactileBundleStorageError(
                    "configured dataset label differs from the existing tactile bundle "
                    f"identity: existing={identity.dataset_label!r} requested={dataset_label!r}"
                )
            return identity
        if _is_nonempty_episode_catalog(self.core_dataset_root):
            raise TactileBundleStorageError(
                "cannot enable native tactile separation on a populated unbound dataset; "
                "create a new dataset or run an explicit offline migration"
            )
        if not dataset_label:
            raise TactileBundleStorageError(
                "dataset_label is required when creating a tactile bundle identity"
            )
        identity = TactileBundleDatasetIdentityV1(
            dataset_uuid=str(uuid_factory()),
            dataset_label=dataset_label,
        )
        publish_immutable_canonical(
            self.core_identity_path,
            identity.to_payload(),
            domain=TACTILE_BUNDLE_DATASET_IDENTITY_FILE_DOMAIN_V1,
        )
        return identity

    def _publish_dataset_identity_to_storage(self) -> None:
        if (
            not self.storage_identity_path.is_file()
            and _is_nonempty_episode_catalog(self.core_dataset_root)
        ):
            raise TactileBundleStorageError(
                "bound tactile storage identity is missing for a populated dataset; "
                "refusing to create a replacement bundle shell"
            )
        publish_immutable_canonical(
            self.storage_identity_path,
            self.dataset_identity.to_payload(),
            domain=TACTILE_BUNDLE_DATASET_IDENTITY_FILE_DOMAIN_V1,
        )

    def _publish_dataset_locator(self) -> None:
        """Publish a readable, derived pointer without changing UUID authority."""

        locator = TactileBundleDatasetLocatorV1(
            dataset_uuid=self.dataset_identity.dataset_uuid,
            dataset_label=self.dataset_identity.dataset_label,
            dataset_identity_hash=self.dataset_identity.semantic_content_hash,
        )
        locator_root = self.tactile_storage_root / "datasets_by_name"
        if locator_root.is_symlink() or self.dataset_locator_path.parent.is_symlink():
            raise TactileBundleStorageError(
                "tactile dataset locator directories must not be symlinks"
            )
        try:
            self.dataset_locator_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise TactileBundleStorageError(
                f"cannot create tactile dataset locator directory: {exc}"
            ) from exc
        locator_parent = self.dataset_locator_path.parent
        if (
            not locator_root.is_dir()
            or not locator_parent.is_dir()
            or locator_root.is_symlink()
            or locator_parent.is_symlink()
            or not locator_parent.resolve().is_relative_to(self.tactile_storage_root)
            or int(locator_parent.stat().st_dev) != self._tactile_device_id
        ):
            raise TactileBundleStorageError(
                "tactile dataset locator must remain on the configured storage "
                "filesystem"
            )
        path = self.dataset_locator_path
        if path.is_symlink():
            raise TactileBundleStorageError(
                "tactile dataset locator must not be a symlink"
            )
        payload = locator.to_payload()
        if path.exists():
            try:
                existing_payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(existing_payload, Mapping):
                    raise TactileBundleContractError(
                        "tactile bundle dataset locator must be an object"
                    )
                existing = TactileBundleDatasetLocatorV1.from_payload(
                    existing_payload
                )
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                TactileBundleContractError,
            ) as exc:
                raise TactileBundleStorageError(
                    f"cannot validate tactile dataset locator {path}: {exc}"
                ) from exc
            if existing != locator:
                raise TactileBundleStorageError(
                    f"tactile dataset locator conflicts with bound identity: {path}"
                )
            return
        encoded = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @property
    def _recovery_journal_path(self) -> Path:
        return (
            self.core_dataset_root
            / "meta/tactile_bundle_v1/recovery/active_transaction.json"
        )

    def _read_catalog_bindings(self) -> tuple[dict[int, str], str | None]:
        catalog = self.core_dataset_root / "meta/episodes.jsonl"
        if not catalog.is_file():
            return {}, None
        try:
            raw = catalog.read_bytes()
            lines = raw.decode("utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise TactileBundleStorageError(
                f"cannot read core episode catalog during tactile recovery: {exc}"
            ) from exc
        bindings: dict[int, str] = {}
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                index = int(record["episode_index"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise TactileBundleStorageError(
                    f"invalid core episode catalog row {line_number}: {exc}"
                ) from exc
            binding = record.get("tactile_bundle")
            if not isinstance(binding, Mapping):
                raise TactileBundleStorageError(
                    "native tactile dataset catalog contains an unbound episode"
                )
            if str(binding.get("dataset_uuid") or "") != self.dataset_identity.dataset_uuid:
                raise TactileBundleStorageError(
                    "core episode catalog references another tactile dataset UUID"
                )
            episode_uuid = str(binding.get("episode_uuid") or "")
            if not episode_uuid or index in bindings:
                raise TactileBundleStorageError(
                    "core tactile episode catalog identity is missing or duplicated"
                )
            bindings[index] = episode_uuid
        return bindings, hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _episode_index_from_unit(path: Path) -> int | None:
        match = re.fullmatch(r"episode_(\d{6})", path.name)
        return None if match is None else int(match.group(1))

    def _sidecar_identity(self, path: Path) -> EpisodeCaptureIdentity:
        payload = _load_canonical(
            path / "identity.cjson",
            domain=TACTILE_BUNDLE_EPISODE_IDENTITY_FILE_DOMAIN_V1,
        )
        return EpisodeCaptureIdentity.from_payload(payload)

    def _discover_abandoned_tail_moves(
        self,
        *,
        committed: Mapping[int, str],
    ) -> tuple[int | None, list[dict[str, str]]]:
        next_index = max(committed, default=-1) + 1
        moves: list[dict[str, str]] = []
        orphan_indices: set[int] = set()
        orphan_episode_uuids: set[str] = set()
        tactile_root = self.bundle_dataset_root / "tactile"
        if tactile_root.is_dir():
            for path in sorted(tactile_root.rglob("episode_*")):
                if not path.is_dir():
                    continue
                index = self._episode_index_from_unit(path)
                if index is None:
                    continue
                if index in committed:
                    continue
                orphan_indices.add(index)
                moves.append(
                    {
                        "root": "tactile",
                        "source": path.relative_to(self.bundle_dataset_root).as_posix(),
                    }
                )
        for root_name, root in (
            (
                "core",
                self.core_dataset_root / "meta/tactile_bundle_v1/episodes",
            ),
            (
                "tactile",
                self.bundle_dataset_root / "meta/tactile_bundle_v1/episodes",
            ),
        ):
            if not root.is_dir():
                continue
            for path in sorted(item for item in root.iterdir() if item.is_dir()):
                identity = self._sidecar_identity(path)
                if path.name != identity.episode_uuid:
                    raise TactileBundleStorageError(
                        "tactile episode sidecar directory does not match its identity"
                    )
                committed_uuid = committed.get(identity.legacy_episode_index)
                if committed_uuid is not None:
                    if identity.episode_uuid != committed_uuid:
                        raise TactileBundleStorageError(
                            "committed tactile sidecar UUID differs from the catalog"
                        )
                    continue
                orphan_indices.add(identity.legacy_episode_index)
                orphan_episode_uuids.add(identity.episode_uuid)
                base = (
                    self.core_dataset_root
                    if root_name == "core"
                    else self.bundle_dataset_root
                )
                moves.append(
                    {
                        "root": root_name,
                        "source": path.relative_to(base).as_posix(),
                    }
                )
        if not moves:
            return None, []
        if orphan_indices != {next_index}:
            raise TactileBundleStorageError(
                "abandoned tactile bundle artifacts are not confined to the unique "
                f"next episode: expected={next_index} actual={sorted(orphan_indices)}"
            )
        if len(orphan_episode_uuids) > 1:
            raise TactileBundleStorageError(
                "abandoned tactile tail contains conflicting episode identities"
            )
        # A unit can be discovered once from each side but never twice at the
        # same root-relative path.
        keys = [(item["root"], item["source"]) for item in moves]
        if len(keys) != len(set(keys)):
            raise TactileBundleStorageError(
                "duplicate abandoned tactile recovery source"
            )
        return next_index, moves

    def _atomic_write_recovery_json(
        self, path: Path, payload: Mapping[str, Any]
    ) -> None:
        atomic_write_json(path, payload, sync_directory=fsync_directory)

    def _apply_recovery_transaction(self, transaction: Mapping[str, Any]) -> None:
        if transaction.get("schema") != "dexteleop.tactile_bundle.recovery.v1":
            raise TactileBundleStorageError(
                "unsupported tactile bundle recovery transaction"
            )
        committed, catalog_hash = self._read_catalog_bindings()
        if str(transaction.get("dataset_uuid") or "") != self.dataset_identity.dataset_uuid:
            raise TactileBundleStorageError(
                "tactile recovery transaction references another dataset UUID"
            )
        if transaction.get("catalog_sha256") != catalog_hash:
            raise TactileBundleStorageError(
                "core episode catalog changed during tactile bundle recovery"
            )
        operation_id = str(transaction.get("operation_id") or "")
        if re.fullmatch(r"[0-9a-f]{32}", operation_id) is None:
            raise TactileBundleStorageError("invalid tactile recovery operation ID")
        episode_index = int(transaction.get("episode_index", -1))
        expected_index = max(committed, default=-1) + 1
        if episode_index != expected_index:
            raise TactileBundleStorageError(
                "tactile recovery transaction is not for the unique next episode: "
                f"expected={expected_index} actual={episode_index}"
            )
        moves = transaction.get("moves")
        if not isinstance(moves, list) or not moves:
            raise TactileBundleStorageError("tactile recovery transaction has no moves")
        move_keys: set[tuple[str, str]] = set()
        completed: list[dict[str, str]] = []
        for raw_move in moves:
            if not isinstance(raw_move, Mapping):
                raise TactileBundleStorageError("invalid tactile recovery move")
            root_name = str(raw_move.get("root") or "")
            root = (
                self.core_dataset_root
                if root_name == "core"
                else self.bundle_dataset_root
                if root_name == "tactile"
                else None
            )
            if root is None:
                raise TactileBundleStorageError("invalid tactile recovery root")
            source_relative = Path(str(raw_move.get("source") or ""))
            if (
                source_relative.is_absolute()
                or ".." in source_relative.parts
                or not source_relative.parts
            ):
                raise TactileBundleStorageError("unsafe tactile recovery source")
            source_key = (root_name, source_relative.as_posix())
            if source_key in move_keys:
                raise TactileBundleStorageError(
                    "duplicate tactile recovery transaction source"
                )
            move_keys.add(source_key)
            is_sidecar = (
                len(source_relative.parts) == 4
                and source_relative.parts[:3]
                == ("meta", "tactile_bundle_v1", "episodes")
            )
            is_tactile_payload = (
                root_name == "tactile"
                and source_relative.parts[0] == TACTILE_BUNDLE_LOGICAL_PREFIX_V1
                and self._episode_index_from_unit(
                    Path(source_relative.parts[-1])
                )
                == episode_index
            )
            if not (is_sidecar or is_tactile_payload):
                raise TactileBundleStorageError(
                    "tactile recovery transaction source is outside the allowed tail namespaces"
                )
            target_relative = (
                Path("meta/tactile_bundle_v1/recovery/abandoned")
                / f"episode_{episode_index:06d}_{operation_id}"
                / "artifacts"
                / source_relative
            )
            source = root / source_relative
            target = root / target_relative
            for candidate, label in ((source, "source"), (target, "target")):
                if candidate.exists():
                    resolved = candidate.resolve()
                    if not resolved.is_relative_to(root.resolve()):
                        raise TactileBundleStorageError(
                            f"tactile recovery {label} escapes its storage root"
                        )
                    if candidate.is_symlink():
                        raise TactileBundleStorageError(
                            f"tactile recovery {label} must not be a symlink"
                        )
            if source.exists() and target.exists():
                raise TactileBundleStorageError(
                    "tactile recovery source and quarantine target both exist"
                )
            if not source.exists() and not target.exists():
                raise TactileBundleStorageError(
                    "tactile recovery lost both source and quarantine target"
                )
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
                fsync_directory(source.parent)
                fsync_directory(target.parent)
            completed.append(
                {
                    "root": root_name,
                    "source": source_relative.as_posix(),
                    "quarantine": target_relative.as_posix(),
                }
            )
        receipt = {
            **dict(transaction),
            "status": "complete",
            "completed_moves": completed,
        }
        receipt_name = f"episode_{episode_index:06d}_{operation_id}.json"
        self._atomic_write_recovery_json(
            self.core_dataset_root
            / "meta/tactile_bundle_v1/recovery/receipts"
            / receipt_name,
            receipt,
        )
        self._atomic_write_recovery_json(
            self.bundle_dataset_root
            / "meta/tactile_bundle_v1/recovery/receipts"
            / receipt_name,
            receipt,
        )
        self._recovery_journal_path.unlink(missing_ok=True)
        fsync_directory(self._recovery_journal_path.parent)

    def _recover_abandoned_tail(self) -> None:
        if self._recovery_journal_path.is_file():
            transaction = _load_json_object(self._recovery_journal_path)
            self._apply_recovery_transaction(transaction)
            return
        committed, catalog_hash = self._read_catalog_bindings()
        episode_index, moves = self._discover_abandoned_tail_moves(
            committed=committed
        )
        if not moves or episode_index is None:
            return
        transaction = {
            "schema": "dexteleop.tactile_bundle.recovery.v1",
            "operation_id": uuid.uuid4().hex,
            "status": "planned",
            "dataset_uuid": self.dataset_identity.dataset_uuid,
            "episode_index": episode_index,
            "catalog_sha256": catalog_hash,
            "moves": moves,
        }
        self._atomic_write_recovery_json(
            self._recovery_journal_path,
            transaction,
        )
        self._apply_recovery_transaction(transaction)

    def revalidate_runtime_storage(self) -> None:
        _assert_not_nested(self.core_dataset_root, self.tactile_storage_root)
        if (
            int(self.core_dataset_root.stat().st_dev) != self._core_device_id
            or int(self.tactile_storage_root.stat().st_dev)
            != self._tactile_device_id
            or int(self.bundle_dataset_root.stat().st_dev)
            != self._bundle_device_id
        ):
            raise TactileBundleStorageError(
                "core, tactile root, or tactile bundle filesystem device changed "
                "during writer lifetime"
            )
        _probe_writable_directory(self.tactile_storage_root)
        if self.require_distinct_device and (
            int(self.core_dataset_root.stat().st_dev)
            == int(self.tactile_storage_root.stat().st_dev)
        ):
            raise TactileBundleStorageError(
                "tactile storage root is no longer on a distinct filesystem device"
            )
        if int(shutil.disk_usage(self.tactile_storage_root).free) < self.minimum_free_bytes:
            raise TactileBundleStorageError(
                "tactile storage root no longer satisfies minimum free-space requirement"
            )
        payload = _load_canonical(
            self.storage_identity_path,
            domain=TACTILE_BUNDLE_DATASET_IDENTITY_FILE_DOMAIN_V1,
        )
        if TactileBundleDatasetIdentityV1.from_payload(payload) != self.dataset_identity:
            raise TactileBundleStorageError(
                "tactile storage dataset identity changed"
            )
        core_payload = _load_canonical(
            self.core_identity_path,
            domain=TACTILE_BUNDLE_DATASET_IDENTITY_FILE_DOMAIN_V1,
        )
        if TactileBundleDatasetIdentityV1.from_payload(core_payload) != self.dataset_identity:
            raise TactileBundleStorageError(
                "core tactile dataset identity changed"
            )

    def prepare_episode(self, identity: EpisodeCaptureIdentity) -> None:
        # Re-check at every episode boundary so a removed/remounted storage
        # device cannot silently redirect new tactile bytes to the mountpoint's
        # underlying filesystem.
        self.revalidate_runtime_storage()
        core_dir = self.core_dataset_root / core_episode_relative_dir(
            identity.episode_uuid
        )
        storage_dir = self.bundle_dataset_root / storage_episode_relative_dir(
            identity.episode_uuid
        )
        payload = identity.to_payload()
        publish_immutable_canonical(
            core_dir / "identity.cjson",
            payload,
            domain=TACTILE_BUNDLE_EPISODE_IDENTITY_FILE_DOMAIN_V1,
        )
        publish_immutable_canonical(
            storage_dir / "identity.cjson",
            payload,
            domain=TACTILE_BUNDLE_EPISODE_IDENTITY_FILE_DOMAIN_V1,
        )

    def discard_prepared_episode(self, identity: EpisodeCaptureIdentity) -> None:
        for root, relative_dir in (
            (
                self.core_dataset_root,
                core_episode_relative_dir(identity.episode_uuid),
            ),
            (
                self.bundle_dataset_root,
                storage_episode_relative_dir(identity.episode_uuid),
            ),
        ):
            path = root / relative_dir
            if path.is_dir():
                shutil.rmtree(path)
                fsync_directory(path.parent)

    def tactile_artifact_path(self, logical_path: str) -> Path:
        return _safe_resolve_under(self.bundle_dataset_root, logical_path)

    def _build_core_references(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        tactile_paths: Mapping[str, str],
    ) -> tuple[CoreTactileReferenceV1, ...]:
        references: list[CoreTactileReferenceV1] = []
        for fallback_index, row in enumerate(rows):
            sample_index = int(row.get("frame_index", fallback_index))
            for side, manifest_path in sorted(tactile_paths.items()):
                prefix = f"observation.hand.{side}.tactile"
                row_path = row.get(f"{prefix}_path")
                frame_index = row.get(f"{prefix}_frame_index")
                if row_path is None and frame_index is None:
                    continue
                if str(row_path) != str(manifest_path):
                    raise TactileBundleStorageError(
                        "core tactile reference points at an unexpected manifest: "
                        f"side={side} sample={sample_index} path={row_path!r} "
                        f"expected={manifest_path!r}"
                    )
                references.append(
                    CoreTactileReferenceV1(
                        side=side,
                        sample_frame_index=sample_index,
                        manifest_logical_path=str(row_path),
                        tactile_frame_index=int(frame_index),
                    )
                )
        return tuple(references)

    def _build_tactile_component(
        self,
        *,
        identity: EpisodeCaptureIdentity,
        sample_count: int,
        tactile_paths: Mapping[str, str],
    ) -> TactileEpisodeComponentV1:
        artifacts: list[TactileBundleArtifactV1] = []
        bindings: list[TactileManifestBindingV1] = []
        for side, manifest_logical_path in sorted(tactile_paths.items()):
            manifest_path = self.tactile_artifact_path(manifest_logical_path)
            if not manifest_path.is_file():
                raise TactileBundleStorageError(
                    f"sealed tactile manifest is missing: {manifest_logical_path}"
                )
            manifest = _load_json_object(manifest_path)
            if str(manifest.get("side")) != str(side):
                raise TactileBundleStorageError(
                    f"tactile manifest side mismatch: {manifest_logical_path}"
                )
            frames = manifest.get("frames")
            streams = manifest.get("streams")
            if not isinstance(frames, list) or not isinstance(streams, Mapping):
                raise TactileBundleStorageError(
                    f"tactile manifest is incomplete: {manifest_logical_path}"
                )
            declared_count = int(manifest.get("frame_count", -1))
            if declared_count != len(frames):
                raise TactileBundleStorageError(
                    f"tactile manifest frame count mismatch: {manifest_logical_path}"
                )
            artifacts.append(
                TactileBundleArtifactV1(
                    logical_path=manifest_logical_path,
                    role="manifest",
                    size_bytes=int(manifest_path.stat().st_size),
                    sha256=_sha256_file(manifest_path),
                )
            )
            bindings.append(
                TactileManifestBindingV1(
                    side=side,
                    manifest_logical_path=manifest_logical_path,
                    frame_count=declared_count,
                    frame_alignment_hash=tactile_frame_alignment_hash(frames),
                )
            )
            for sensor_key, raw_spec in sorted(streams.items()):
                if not isinstance(raw_spec, Mapping):
                    raise TactileBundleStorageError(
                        f"invalid tactile stream {sensor_key!r} in {manifest_logical_path}"
                    )
                logical_path = str(raw_spec.get("path") or "")
                payload_path = self.tactile_artifact_path(logical_path)
                if not payload_path.is_file():
                    raise TactileBundleStorageError(
                        f"sealed tactile payload is missing: {logical_path}"
                    )
                size_bytes = int(raw_spec.get("stored_size_bytes", -1))
                sha256 = str(raw_spec.get("stored_sha256") or "")
                if size_bytes != int(payload_path.stat().st_size):
                    raise TactileBundleStorageError(
                        f"sealed tactile payload size mismatch: {logical_path}"
                    )
                if sha256 != _sha256_file(payload_path):
                    raise TactileBundleStorageError(
                        f"sealed tactile payload hash mismatch: {logical_path}"
                    )
                artifacts.append(
                    TactileBundleArtifactV1(
                        logical_path=logical_path,
                        role="payload",
                        size_bytes=size_bytes,
                        sha256=sha256,
                    )
                )
        return TactileEpisodeComponentV1(
            dataset_uuid=self.dataset_identity.dataset_uuid,
            episode_uuid=identity.episode_uuid,
            legacy_episode_index=identity.legacy_episode_index,
            episode_identity_hash=identity.semantic_content_hash,
            sample_count=int(sample_count),
            artifacts=tuple(artifacts),
            manifests=tuple(bindings),
        )

    def _validate_cross_root_alignment(
        self,
        *,
        core_component: CoreTactileEpisodeComponentV1,
        tactile_component: TactileEpisodeComponentV1,
    ) -> None:
        refs = {
            (item.side, item.sample_frame_index): item
            for item in core_component.references
        }
        frame_by_side_index: dict[tuple[str, int], TactileManifestBindingV1] = {}
        for binding in tactile_component.manifests:
            manifest = _load_json_object(
                self.tactile_artifact_path(binding.manifest_logical_path)
            )
            frames = manifest.get("frames", ())
            if tactile_frame_alignment_hash(frames) != binding.frame_alignment_hash:
                raise TactileBundleStorageError(
                    f"tactile frame alignment changed: {binding.manifest_logical_path}"
                )
            manifest_frame_indices: set[int] = set()
            for frame in frames:
                if not isinstance(frame, Mapping):
                    raise TactileBundleStorageError(
                        f"invalid tactile frame record: {binding.manifest_logical_path}"
                    )
                tactile_frame_index = int(
                    frame.get("tactile_frame_index", -1)
                )
                frame_key = (binding.side, tactile_frame_index)
                if frame_key in frame_by_side_index:
                    raise TactileBundleStorageError(
                        "tactile manifest repeats a side/tactile frame index: "
                        f"side={frame_key[0]} tactile_frame={frame_key[1]}"
                    )
                manifest_frame_indices.add(tactile_frame_index)
                frame_by_side_index[frame_key] = binding
                sample_index = int(frame.get("sample_frame_index", -1))
                reference = refs.get((binding.side, sample_index))
                if reference is None:
                    raise TactileBundleStorageError(
                        "tactile source frame has no matching core sample reference: "
                        f"side={binding.side} sample={sample_index}"
                    )
                if (
                    reference.manifest_logical_path
                    != binding.manifest_logical_path
                    or reference.tactile_frame_index != tactile_frame_index
                ):
                    raise TactileBundleStorageError(
                        "core and tactile frame alignment disagree: "
                        f"side={binding.side} sample={sample_index}"
                    )
            if manifest_frame_indices != set(range(binding.frame_count)):
                raise TactileBundleStorageError(
                    "tactile manifest frame indices are not contiguous from zero: "
                    f"{binding.manifest_logical_path}"
                )
        for reference in core_component.references:
            binding = frame_by_side_index.get(
                (reference.side, reference.tactile_frame_index)
            )
            if (
                binding is None
                or binding.manifest_logical_path
                != reference.manifest_logical_path
            ):
                raise TactileBundleStorageError(
                    "core tactile reference points outside a committed manifest: "
                    f"side={reference.side} sample={reference.sample_frame_index}"
                )

    def seal_episode(
        self,
        *,
        identity: EpisodeCaptureIdentity,
        rows: Sequence[Mapping[str, Any]],
        tactile_paths: Mapping[str, str],
    ) -> TactileBundleSealReport:
        self.revalidate_runtime_storage()
        if not tactile_paths:
            raise TactileBundleStorageError(
                "native tactile bundle episode cannot commit without tactile manifests"
            )
        sample_count = len(rows)
        core_component = CoreTactileEpisodeComponentV1(
            dataset_uuid=self.dataset_identity.dataset_uuid,
            episode_uuid=identity.episode_uuid,
            legacy_episode_index=identity.legacy_episode_index,
            episode_identity_hash=identity.semantic_content_hash,
            sample_count=sample_count,
            references=self._build_core_references(
                rows,
                tactile_paths=tactile_paths,
            ),
        )
        tactile_component = self._build_tactile_component(
            identity=identity,
            sample_count=sample_count,
            tactile_paths=tactile_paths,
        )
        self._validate_cross_root_alignment(
            core_component=core_component,
            tactile_component=tactile_component,
        )
        core_relative_dir = core_episode_relative_dir(identity.episode_uuid)
        storage_relative_dir = storage_episode_relative_dir(identity.episode_uuid)
        core_component_ref = f"{core_relative_dir}/core_component.cjson"
        tactile_component_ref = f"{storage_relative_dir}/tactile_component.cjson"
        commit = TactileEpisodeBundleCommitV1(
            dataset_uuid=self.dataset_identity.dataset_uuid,
            episode_uuid=identity.episode_uuid,
            legacy_episode_index=identity.legacy_episode_index,
            episode_identity_hash=identity.semantic_content_hash,
            core_component_hash=core_component.semantic_content_hash,
            tactile_component_hash=tactile_component.semantic_content_hash,
            core_component_ref=core_component_ref,
            tactile_component_ref=tactile_component_ref,
        )
        publish_immutable_canonical(
            self.core_dataset_root / core_component_ref,
            core_component.to_payload(),
            domain=TACTILE_BUNDLE_CORE_COMPONENT_FILE_DOMAIN_V1,
        )
        publish_immutable_canonical(
            self.bundle_dataset_root / tactile_component_ref,
            tactile_component.to_payload(),
            domain=TACTILE_BUNDLE_TACTILE_COMPONENT_FILE_DOMAIN_V1,
        )
        tactile_commit_ref = f"{storage_relative_dir}/bundle_commit.cjson"
        core_commit_ref = f"{core_relative_dir}/bundle_commit.cjson"
        # The tactile-side commit is published first.  The core-side commit is
        # the local readiness marker; the authoritative episode catalog is
        # appended only after this method returns successfully.
        publish_immutable_canonical(
            self.bundle_dataset_root / tactile_commit_ref,
            commit.to_payload(),
            domain=TACTILE_BUNDLE_EPISODE_COMMIT_FILE_DOMAIN_V1,
        )
        publish_immutable_canonical(
            self.core_dataset_root / core_commit_ref,
            commit.to_payload(),
            domain=TACTILE_BUNDLE_EPISODE_COMMIT_FILE_DOMAIN_V1,
        )
        return TactileBundleSealReport(
            dataset_uuid=self.dataset_identity.dataset_uuid,
            episode_uuid=identity.episode_uuid,
            commit_hash=commit.semantic_content_hash,
            core_component_hash=core_component.semantic_content_hash,
            tactile_component_hash=tactile_component.semantic_content_hash,
            core_commit_relative_path=core_commit_ref,
            tactile_commit_relative_path=tactile_commit_ref,
            artifact_count=len(tactile_component.artifacts),
            payload_bytes=sum(
                item.size_bytes
                for item in tactile_component.artifacts
                if item.role == "payload"
            ),
        )


__all__ = [
    "NativeTactileBundleStore",
    "TactileBundleSealReport",
    "TactileBundleStorageError",
]
