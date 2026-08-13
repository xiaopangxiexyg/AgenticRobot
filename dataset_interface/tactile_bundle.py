"""Read-only attachment and validation for native tactile bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.tactile_bundle import (
    CoreTactileEpisodeComponentV1,
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
    TactileEpisodeBundleCommitV1,
    TactileEpisodeComponentV1,
    core_dataset_identity_relative_path,
    normalize_bundle_relative_path,
    storage_dataset_identity_relative_path,
    tactile_frame_alignment_hash,
)


class TactileBundleAttachmentError(RuntimeError):
    """Raised when core and tactile bundle identities or artifacts disagree."""


@dataclass(frozen=True)
class AttachedTactileEpisode:
    episode_uuid: str
    legacy_episode_index: int
    commit: TactileEpisodeBundleCommitV1
    core_component: CoreTactileEpisodeComponentV1
    tactile_component: TactileEpisodeComponentV1


@dataclass(frozen=True)
class TactileBundleAttachmentReport:
    dataset_uuid: str
    episode_count: int
    artifact_count: int
    payload_count: int
    payload_bytes: int
    payload_hashes_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "dexteleop.tactile_bundle.attachment_report.v1",
            "dataset_uuid": self.dataset_uuid,
            "episode_count": self.episode_count,
            "artifact_count": self.artifact_count,
            "payload_count": self.payload_count,
            "payload_bytes": self.payload_bytes,
            "payload_hashes_verified": self.payload_hashes_verified,
            "attached": True,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_canonical(path: Path, *, domain: str) -> Mapping[str, Any]:
    try:
        payload = canonical_loads(path.read_bytes(), expected_domain=domain)
    except (OSError, ValueError) as exc:
        raise TactileBundleAttachmentError(
            f"Cannot validate canonical bundle artifact {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise TactileBundleAttachmentError(
            f"Canonical bundle artifact is not a mapping: {path}"
        )
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TactileBundleAttachmentError(
            f"Cannot read JSON artifact {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise TactileBundleAttachmentError(f"JSON artifact is not an object: {path}")
    return payload


def _read_episode_catalog(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        return ()
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("row is not a JSON object")
                rows.append(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise TactileBundleAttachmentError(
            f"Cannot read core episode catalog {path}: {type(exc).__name__}: {exc}"
        ) from exc
    return tuple(rows)


def _safe_path(root: Path, relative_path: str, *, tactile_only: bool = False) -> Path:
    try:
        normalized = normalize_bundle_relative_path(
            relative_path,
            field_name="relative_path",
            required_prefix="tactile" if tactile_only else None,
        )
    except TactileBundleContractError as exc:
        raise TactileBundleAttachmentError(str(exc)) from exc
    candidate = root / normalized
    try:
        resolved_parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise TactileBundleAttachmentError(
            f"Bundle artifact parent is unavailable: {candidate}: {exc}"
        ) from exc
    if not resolved_parent.is_relative_to(root.resolve()):
        raise TactileBundleAttachmentError(
            f"Bundle artifact escapes its root: {relative_path}"
        )
    if candidate.is_symlink():
        raise TactileBundleAttachmentError(
            f"Bundle artifact must not be a symlink: {relative_path}"
        )
    return candidate


class AttachedTactileBundle:
    """Verified, movable binding between one core dataset and tactile root."""

    def __init__(
        self,
        *,
        core_dataset_root: str | Path,
        tactile_storage_root: str | Path,
        verify_payload_hashes: bool = True,
    ) -> None:
        configured_core_root = Path(core_dataset_root).expanduser()
        configured_tactile_root = Path(tactile_storage_root).expanduser()
        if configured_core_root.is_symlink():
            raise TactileBundleAttachmentError(
                "core dataset root must not be a symlink"
            )
        if configured_tactile_root.is_symlink():
            raise TactileBundleAttachmentError(
                "tactile storage root must not be a symlink"
            )
        self.core_dataset_root = configured_core_root.resolve()
        self.tactile_storage_root = configured_tactile_root.resolve()
        if not self.core_dataset_root.is_dir():
            raise TactileBundleAttachmentError(
                f"core dataset root is unavailable: {self.core_dataset_root}"
            )
        if not self.tactile_storage_root.is_dir():
            raise TactileBundleAttachmentError(
                f"tactile storage root is unavailable: {self.tactile_storage_root}"
            )
        if (
            self.core_dataset_root == self.tactile_storage_root
            or self.core_dataset_root.is_relative_to(self.tactile_storage_root)
            or self.tactile_storage_root.is_relative_to(self.core_dataset_root)
        ):
            raise TactileBundleAttachmentError(
                "core dataset and tactile storage roots must be disjoint"
            )
        core_identity_payload = _load_canonical(
            self.core_dataset_root / core_dataset_identity_relative_path(),
            domain=TACTILE_BUNDLE_DATASET_IDENTITY_FILE_DOMAIN_V1,
        )
        self.dataset_identity = TactileBundleDatasetIdentityV1.from_payload(
            core_identity_payload
        )
        bundle_dataset_path = (
            self.tactile_storage_root
            / "datasets"
            / self.dataset_identity.dataset_uuid
        )
        if (
            (self.tactile_storage_root / "datasets").is_symlink()
            or bundle_dataset_path.is_symlink()
            or not bundle_dataset_path.resolve().is_relative_to(
                self.tactile_storage_root
            )
        ):
            raise TactileBundleAttachmentError(
                "tactile bundle dataset path escapes the storage root"
            )
        self.bundle_dataset_root = bundle_dataset_path.resolve()
        storage_identity_payload = _load_canonical(
            self.bundle_dataset_root / storage_dataset_identity_relative_path(),
            domain=TACTILE_BUNDLE_DATASET_IDENTITY_FILE_DOMAIN_V1,
        )
        storage_identity = TactileBundleDatasetIdentityV1.from_payload(
            storage_identity_payload
        )
        if storage_identity != self.dataset_identity:
            raise TactileBundleAttachmentError(
                "core and tactile dataset identities disagree"
            )
        info_path = self.core_dataset_root / "meta/info.json"
        if info_path.is_file():
            info = _load_json_object(info_path)
            binding = info.get("tactile_bundle")
            if not isinstance(binding, Mapping):
                catalog_has_rows = bool(
                    _read_episode_catalog(
                        self.core_dataset_root / "meta/episodes.jsonl"
                    )
                )
                if (
                    catalog_has_rows
                    or int(info.get("episode_count", 0) or 0) > 0
                    or int(info.get("frame_count", 0) or 0) > 0
                ):
                    raise TactileBundleAttachmentError(
                        "populated core dataset info lacks its native tactile bundle binding"
                    )
            elif (
                binding.get("schema")
                != TACTILE_BUNDLE_DATASET_BINDING_SCHEMA_V1
                or str(binding.get("dataset_uuid") or "")
                != self.dataset_identity.dataset_uuid
                or str(binding.get("dataset_identity_hash") or "")
                != self.dataset_identity.semantic_content_hash
                or str(binding.get("core_identity_ref") or "")
                != core_dataset_identity_relative_path()
                or str(binding.get("logical_prefix") or "")
                != TACTILE_BUNDLE_LOGICAL_PREFIX_V1
                or binding.get("tactile_required") is not True
                or binding.get("portable_attachment_required") is not True
            ):
                raise TactileBundleAttachmentError(
                    "core dataset tactile bundle binding disagrees with its identity"
                )
        self.verify_payload_hashes = bool(verify_payload_hashes)
        self.episodes: dict[int, AttachedTactileEpisode] = {}
        self._artifact_by_logical_path: dict[str, TactileBundleArtifactV1] = {}
        self._load_and_verify_episodes()

    @property
    def report(self) -> TactileBundleAttachmentReport:
        artifacts = tuple(self._artifact_by_logical_path.values())
        payloads = tuple(item for item in artifacts if item.role == "payload")
        return TactileBundleAttachmentReport(
            dataset_uuid=self.dataset_identity.dataset_uuid,
            episode_count=len(self.episodes),
            artifact_count=len(artifacts),
            payload_count=len(payloads),
            payload_bytes=sum(item.size_bytes for item in payloads),
            payload_hashes_verified=self.verify_payload_hashes,
        )

    def iter_artifacts(self) -> tuple[TactileBundleArtifactV1, ...]:
        return tuple(
            self._artifact_by_logical_path[path]
            for path in sorted(self._artifact_by_logical_path)
        )

    def resolve(self, logical_path: str) -> Path:
        try:
            normalized = normalize_bundle_relative_path(
                logical_path,
                field_name="logical_path",
                required_prefix="tactile",
            )
        except TactileBundleContractError as exc:
            raise TactileBundleAttachmentError(str(exc)) from exc
        if normalized not in self._artifact_by_logical_path:
            raise TactileBundleAttachmentError(
                f"logical tactile artifact is not owned by a committed episode: {normalized}"
            )
        return _safe_path(self.bundle_dataset_root, normalized, tactile_only=True)

    def _load_and_verify_episodes(self) -> None:
        records = _read_episode_catalog(
            self.core_dataset_root / "meta" / "episodes.jsonl"
        )
        for record in records:
            tactile_paths = record.get("tactile_paths")
            binding = record.get("tactile_bundle")
            if tactile_paths is not None and not isinstance(tactile_paths, Mapping):
                raise TactileBundleAttachmentError(
                    "episode tactile_paths must be a mapping"
                )
            if not tactile_paths:
                raise TactileBundleAttachmentError(
                    "native tactile dataset episode lacks tactile paths"
                )
            if not isinstance(binding, Mapping):
                raise TactileBundleAttachmentError(
                    "episode with tactile paths lacks a native tactile bundle binding"
                )
            attached = self._load_episode(record=record, binding=binding)
            index = attached.legacy_episode_index
            if index in self.episodes:
                raise TactileBundleAttachmentError(
                    f"duplicate tactile bundle episode index: {index}"
                )
            self.episodes[index] = attached
            for artifact in attached.tactile_component.artifacts:
                previous = self._artifact_by_logical_path.get(artifact.logical_path)
                if previous is not None:
                    raise TactileBundleAttachmentError(
                        "logical tactile artifact is owned by more than one episode: "
                        f"{artifact.logical_path}"
                    )
                self._artifact_by_logical_path[artifact.logical_path] = artifact

    def _load_episode(
        self,
        *,
        record: Mapping[str, Any],
        binding: Mapping[str, Any],
    ) -> AttachedTactileEpisode:
        if binding.get("schema") != TACTILE_BUNDLE_EPISODE_BINDING_SCHEMA_V1:
            raise TactileBundleAttachmentError(
                "unsupported episode tactile bundle binding schema"
            )
        if binding.get("tactile_required") is not True:
            raise TactileBundleAttachmentError(
                "native tactile episode binding must require tactile"
            )
        if str(binding.get("logical_prefix") or "") != TACTILE_BUNDLE_LOGICAL_PREFIX_V1:
            raise TactileBundleAttachmentError(
                "native tactile episode binding has the wrong logical prefix"
            )
        dataset_uuid = str(binding.get("dataset_uuid") or "")
        if dataset_uuid != self.dataset_identity.dataset_uuid:
            raise TactileBundleAttachmentError(
                "episode tactile binding references a different dataset UUID"
            )
        core_commit_ref = str(binding.get("core_commit_ref") or "")
        core_commit_path = _safe_path(self.core_dataset_root, core_commit_ref)
        core_commit_bytes = core_commit_path.read_bytes()
        commit = TactileEpisodeBundleCommitV1.from_payload(
            canonical_loads(
                core_commit_bytes,
                expected_domain=TACTILE_BUNDLE_EPISODE_COMMIT_FILE_DOMAIN_V1,
            )
        )
        if str(binding.get("commit_hash")) != commit.semantic_content_hash:
            raise TactileBundleAttachmentError("episode commit hash binding mismatch")
        tactile_commit_ref = str(binding.get("tactile_commit_ref") or "")
        if not tactile_commit_ref:
            # v1 uses the same episode-relative name in the tactile namespace.
            tactile_commit_ref = str(
                Path(commit.tactile_component_ref).parent / "bundle_commit.cjson"
            )
        tactile_commit_path = _safe_path(
            self.bundle_dataset_root,
            tactile_commit_ref,
        )
        if tactile_commit_path.read_bytes() != core_commit_bytes:
            raise TactileBundleAttachmentError(
                "core and tactile bundle commit bytes disagree"
            )
        core_component = CoreTactileEpisodeComponentV1.from_payload(
            _load_canonical(
                _safe_path(self.core_dataset_root, commit.core_component_ref),
                domain=TACTILE_BUNDLE_CORE_COMPONENT_FILE_DOMAIN_V1,
            )
        )
        tactile_component = TactileEpisodeComponentV1.from_payload(
            _load_canonical(
                _safe_path(
                    self.bundle_dataset_root,
                    commit.tactile_component_ref,
                ),
                domain=TACTILE_BUNDLE_TACTILE_COMPONENT_FILE_DOMAIN_V1,
            )
        )
        if core_component.semantic_content_hash != commit.core_component_hash:
            raise TactileBundleAttachmentError("core component hash mismatch")
        if tactile_component.semantic_content_hash != commit.tactile_component_hash:
            raise TactileBundleAttachmentError("tactile component hash mismatch")
        expected_index = int(record.get("episode_index", -1))
        expected_uuid = str(binding.get("episode_uuid") or "")
        if (
            commit.dataset_uuid != self.dataset_identity.dataset_uuid
            or commit.legacy_episode_index != expected_index
            or commit.episode_uuid != expected_uuid
        ):
            raise TactileBundleAttachmentError(
                "episode commit identity does not match its core catalog row"
            )
        expected_sample_count = int(record.get("frame_count", -1))
        for component in (core_component, tactile_component):
            if (
                component.dataset_uuid != self.dataset_identity.dataset_uuid
                or component.legacy_episode_index != expected_index
                or component.episode_uuid != expected_uuid
                or component.episode_identity_hash != commit.episode_identity_hash
                or component.sample_count != expected_sample_count
            ):
                raise TactileBundleAttachmentError(
                    "episode component identity does not match its core catalog row"
                )
        record_tactile_paths = {
            str(side): str(path)
            for side, path in dict(record.get("tactile_paths") or {}).items()
        }
        component_tactile_paths = {
            item.side: item.manifest_logical_path
            for item in tactile_component.manifests
        }
        if record_tactile_paths != component_tactile_paths:
            raise TactileBundleAttachmentError(
                "core tactile_paths and committed tactile manifests disagree"
            )
        self._verify_episode_identities(commit)
        self._verify_artifacts_and_alignment(
            core_component=core_component,
            tactile_component=tactile_component,
        )
        return AttachedTactileEpisode(
            episode_uuid=commit.episode_uuid,
            legacy_episode_index=commit.legacy_episode_index,
            commit=commit,
            core_component=core_component,
            tactile_component=tactile_component,
        )

    def _verify_episode_identities(
        self, commit: TactileEpisodeBundleCommitV1
    ) -> None:
        core_identity_ref = str(
            Path(commit.core_component_ref).parent / "identity.cjson"
        )
        tactile_identity_ref = str(
            Path(commit.tactile_component_ref).parent / "identity.cjson"
        )
        core_bytes = _safe_path(
            self.core_dataset_root, core_identity_ref
        ).read_bytes()
        tactile_bytes = _safe_path(
            self.bundle_dataset_root, tactile_identity_ref
        ).read_bytes()
        if core_bytes != tactile_bytes:
            raise TactileBundleAttachmentError(
                "core and tactile episode identity bytes disagree"
            )
        identity = EpisodeCaptureIdentity.from_payload(
            canonical_loads(
                core_bytes,
                expected_domain=TACTILE_BUNDLE_EPISODE_IDENTITY_FILE_DOMAIN_V1,
            )
        )
        if (
            identity.episode_uuid != commit.episode_uuid
            or identity.legacy_episode_index != commit.legacy_episode_index
            or identity.semantic_content_hash != commit.episode_identity_hash
        ):
            raise TactileBundleAttachmentError(
                "episode identity does not match bundle commit"
            )

    def _verify_artifacts_and_alignment(
        self,
        *,
        core_component: CoreTactileEpisodeComponentV1,
        tactile_component: TactileEpisodeComponentV1,
    ) -> None:
        artifacts = {
            item.logical_path: item for item in tactile_component.artifacts
        }
        for artifact in artifacts.values():
            path = _safe_path(
                self.bundle_dataset_root,
                artifact.logical_path,
                tactile_only=True,
            )
            if not path.is_file() or int(path.stat().st_size) != artifact.size_bytes:
                raise TactileBundleAttachmentError(
                    f"tactile artifact is missing or has wrong size: {artifact.logical_path}"
                )
            if self.verify_payload_hashes or artifact.role == "manifest":
                if _sha256_file(path) != artifact.sha256:
                    raise TactileBundleAttachmentError(
                        f"tactile artifact hash mismatch: {artifact.logical_path}"
                    )
        refs = {
            (item.side, item.sample_frame_index): item
            for item in core_component.references
        }
        frame_by_side_index: dict[tuple[str, int], str] = {}
        referenced_payload_paths: set[str] = set()
        for binding in tactile_component.manifests:
            manifest = _load_json_object(
                _safe_path(
                    self.bundle_dataset_root,
                    binding.manifest_logical_path,
                    tactile_only=True,
                )
            )
            frames = manifest.get("frames")
            if (
                not isinstance(frames, list)
                or int(manifest.get("frame_count", -1)) != binding.frame_count
                or tactile_frame_alignment_hash(frames)
                != binding.frame_alignment_hash
            ):
                raise TactileBundleAttachmentError(
                    f"tactile manifest alignment changed: {binding.manifest_logical_path}"
                )
            if str(manifest.get("side") or "") != binding.side:
                raise TactileBundleAttachmentError(
                    f"tactile manifest side changed: {binding.manifest_logical_path}"
                )
            streams = manifest.get("streams")
            if not isinstance(streams, Mapping):
                raise TactileBundleAttachmentError(
                    f"tactile manifest streams are invalid: {binding.manifest_logical_path}"
                )
            for raw_spec in streams.values():
                if not isinstance(raw_spec, Mapping):
                    raise TactileBundleAttachmentError(
                        f"tactile manifest references an unowned payload: {binding.manifest_logical_path}"
                    )
                payload_path = str(raw_spec.get("path") or "")
                artifact = artifacts.get(payload_path)
                if artifact is None or artifact.role != "payload":
                    raise TactileBundleAttachmentError(
                        f"tactile manifest references an unowned payload: {binding.manifest_logical_path}"
                    )
                if payload_path in referenced_payload_paths:
                    raise TactileBundleAttachmentError(
                        f"tactile payload is referenced more than once: {payload_path}"
                    )
                referenced_payload_paths.add(payload_path)
                if (
                    int(raw_spec.get("stored_size_bytes", -1))
                    != artifact.size_bytes
                    or str(raw_spec.get("stored_sha256") or "")
                    != artifact.sha256
                ):
                    raise TactileBundleAttachmentError(
                        f"tactile manifest payload identity disagrees: {payload_path}"
                    )
            manifest_frame_indices: set[int] = set()
            for frame in frames:
                if not isinstance(frame, Mapping):
                    raise TactileBundleAttachmentError(
                        f"invalid tactile frame record: {binding.manifest_logical_path}"
                    )
                tactile_frame_index = int(
                    frame.get("tactile_frame_index", -1)
                )
                frame_key = (binding.side, tactile_frame_index)
                if frame_key in frame_by_side_index:
                    raise TactileBundleAttachmentError(
                        "tactile manifest repeats a side/tactile frame index: "
                        f"side={frame_key[0]} tactile_frame={frame_key[1]}"
                    )
                manifest_frame_indices.add(tactile_frame_index)
                frame_by_side_index[frame_key] = binding.manifest_logical_path
                sample_index = int(frame.get("sample_frame_index", -1))
                reference = refs.get((binding.side, sample_index))
                if (
                    reference is None
                    or reference.manifest_logical_path
                    != binding.manifest_logical_path
                    or reference.tactile_frame_index
                    != tactile_frame_index
                ):
                    raise TactileBundleAttachmentError(
                        "core and tactile frame alignment disagree: "
                        f"side={binding.side} sample={sample_index}"
                    )
            if manifest_frame_indices != set(range(binding.frame_count)):
                raise TactileBundleAttachmentError(
                    "tactile manifest frame indices are not contiguous from zero: "
                    f"{binding.manifest_logical_path}"
                )
        for reference in core_component.references:
            manifest_path = frame_by_side_index.get(
                (reference.side, reference.tactile_frame_index)
            )
            if manifest_path != reference.manifest_logical_path:
                raise TactileBundleAttachmentError(
                    "core tactile reference points outside a committed manifest"
                )
        declared_payload_paths = {
            item.logical_path
            for item in tactile_component.artifacts
            if item.role == "payload"
        }
        if referenced_payload_paths != declared_payload_paths:
            raise TactileBundleAttachmentError(
                "committed tactile payload set and manifest stream set disagree"
            )


def attach_tactile_bundle(
    core_dataset_root: str | Path,
    tactile_storage_root: str | Path,
    *,
    verify_payload_hashes: bool = True,
) -> AttachedTactileBundle:
    return AttachedTactileBundle(
        core_dataset_root=core_dataset_root,
        tactile_storage_root=tactile_storage_root,
        verify_payload_hashes=verify_payload_hashes,
    )


__all__ = [
    "AttachedTactileBundle",
    "AttachedTactileEpisode",
    "TactileBundleAttachmentError",
    "TactileBundleAttachmentReport",
    "attach_tactile_bundle",
]
