"""Project verified corrupt-video exclusions into training validity metadata."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_interface.camera_semantic_prepared_view import (
    load_camera_semantic_video_integrity_exclusions,
)
from dataset_interface.metadata import MODALITY_VALIDITY_SCHEMA_VERSION

from .camera_semantic_prepared_view import (
    CameraSemanticPreparedViewError,
    _episode_records,
    _selected_inventory,
    _source_dataset_identity,
)
from .io import read_json

CAMERA_VIDEO_INTEGRITY_TRAINING_POLICY = (
    "exclude_entire_episode_if_any_required_video_is_invalid"
)


@dataclass(frozen=True)
class CameraVideoIntegrityTrainingPlan:
    source_root: str
    prepared_view_root: str
    prepared_view_manifest_hash: str
    video_integrity_exclusions_hash: str
    source_dataset_identity_hash: str
    source_episode_count: int
    included_episode_count: int
    excluded_episode_count: int
    excluded_episode_indices: tuple[int, ...]
    issue_count: int
    modality_validity_path: str
    modality_validity_before_sha256: str | None
    modality_validity_after_sha256: str
    source_inventory_verified: bool
    idempotent: bool
    plan_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _read_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.is_symlink() or not path.is_file():
        raise CameraSemanticPreparedViewError(
            f"Training validity metadata is missing or unsafe: {path}"
        )
    value = read_json(path)
    if not isinstance(value, dict):
        raise CameraSemanticPreparedViewError(
            f"Training validity metadata must be a JSON object: {path}"
        )
    return value


def _desired_validity(
    current: Mapping[str, Any],
    *,
    manifest_hash: str,
    receipt: Any,
) -> dict[str, Any]:
    desired = json.loads(json.dumps(current))
    schema = desired.get("schema_version")
    if schema not in (None, MODALITY_VALIDITY_SCHEMA_VERSION):
        raise CameraSemanticPreparedViewError(
            "Unsupported modality_validity schema_version="
            f"{schema!r}; refusing to merge camera-video exclusions."
        )
    desired["schema_version"] = MODALITY_VALIDITY_SCHEMA_VERSION
    raw_training = desired.get("training")
    if raw_training is None:
        training: dict[str, Any] = {}
    elif isinstance(raw_training, Mapping):
        training = dict(raw_training)
    else:
        raise CameraSemanticPreparedViewError(
            "meta/modality_validity.json.training must be a JSON object."
        )
    raw_episodes = training.get("episodes")
    if raw_episodes is None:
        episodes: dict[str, Any] = {}
    elif isinstance(raw_episodes, Mapping):
        episodes = {str(key): value for key, value in raw_episodes.items()}
    else:
        raise CameraSemanticPreparedViewError(
            "meta/modality_validity.json.training.episodes must be a JSON object."
        )

    issues_by_episode: dict[int, list[Any]] = defaultdict(list)
    for issue in receipt.issues:
        issues_by_episode[int(issue.episode_index)].append(issue)
    for episode_index in receipt.excluded_episode_indices:
        key = str(int(episode_index))
        raw_episode = episodes.get(key)
        if raw_episode is None:
            episode: dict[str, Any] = {}
        elif isinstance(raw_episode, Mapping):
            episode = dict(raw_episode)
        else:
            raise CameraSemanticPreparedViewError(
                "Existing training validity episode entry must be an object: "
                f"episode={episode_index}."
            )
        issue_labels = sorted(
            {
                f"{item.camera_key}:{item.reason}"
                for item in issues_by_episode[int(episode_index)]
            }
        )
        marker_reason = (
            "required camera video failed full decode/frame-index integrity "
            f"validation ({', '.join(issue_labels)}); receipt="
            f"{receipt.receipt_hash}"
        )
        previous_reason = str(episode.get("reason") or "").strip()
        episode["valid"] = False
        if marker_reason not in previous_reason:
            episode["reason"] = (
                marker_reason
                if not previous_reason
                else f"{previous_reason}; {marker_reason}"
            )
        episode["camera_video_integrity"] = {
            "valid": False,
            "policy": CAMERA_VIDEO_INTEGRITY_TRAINING_POLICY,
            "receipt_hash": receipt.receipt_hash,
            "issue_count": len(issues_by_episode[int(episode_index)]),
        }
        episodes[key] = episode

    training["episodes"] = episodes
    training["camera_video_integrity_exclusions"] = {
        "policy": CAMERA_VIDEO_INTEGRITY_TRAINING_POLICY,
        "prepared_view_manifest_hash": manifest_hash,
        "receipt": receipt.to_dict(),
    }
    desired["training"] = training
    return desired


def plan_camera_video_integrity_training_exclusions(
    source_root: str | Path,
    prepared_view_root: str | Path,
) -> CameraVideoIntegrityTrainingPlan:
    """Build a no-write, hash-bound plan from a verified prepared-view receipt."""

    source = Path(source_root).expanduser().resolve()
    prepared = Path(prepared_view_root).expanduser().resolve()
    if not source.is_dir() or source.is_symlink():
        raise CameraSemanticPreparedViewError(
            f"Source dataset is missing or unsafe: {source}"
        )
    manifest, receipt = load_camera_semantic_video_integrity_exclusions(
        prepared,
        verify_payload_hashes=True,
    )
    if receipt is None:
        raise CameraSemanticPreparedViewError(
            "Prepared view has no corrupt-video exclusion receipt."
        )
    source_episodes = _episode_records(source)
    source_indices = tuple(
        sorted(int(record["episode_index"]) for record in source_episodes)
    )
    expected_indices = tuple(
        sorted(
            receipt.included_episode_indices
            + receipt.excluded_episode_indices
        )
    )
    if source.name != receipt.source_dataset_name:
        raise CameraSemanticPreparedViewError(
            "Prepared-view receipt names a different source dataset."
        )
    source_identity = _source_dataset_identity(source)
    if source_identity != receipt.source_dataset_identity_hash:
        raise CameraSemanticPreparedViewError(
            "Prepared-view receipt binds a different source dataset identity."
        )
    if source_indices != expected_indices:
        raise CameraSemanticPreparedViewError(
            "Source episode coverage differs from the verified exclusion receipt."
        )

    validity_path = source / "meta/modality_validity.json"
    current = _read_object(validity_path)
    desired = _desired_validity(
        current,
        manifest_hash=manifest.manifest_hash,
        receipt=receipt,
    )
    current_bytes = b"" if not validity_path.exists() else validity_path.read_bytes()
    desired_bytes = _json_bytes(desired)
    idempotent = current_bytes == desired_bytes
    inventory_verified = False
    if not idempotent:
        _, inventory_hash, _, _ = _selected_inventory(source, source_episodes)
        if inventory_hash != receipt.selected_source_inventory_hash:
            raise CameraSemanticPreparedViewError(
                "Source camera/numeric inventory changed after the prepared-view "
                "integrity receipt was created; rebuild the prepared view before marking."
            )
        inventory_verified = True

    payload = {
        "source_root": str(source),
        "prepared_view_root": str(prepared),
        "prepared_view_manifest_hash": manifest.manifest_hash,
        "video_integrity_exclusions_hash": receipt.receipt_hash,
        "source_dataset_identity_hash": source_identity,
        "source_episode_count": receipt.source_episode_count,
        "included_episode_count": len(receipt.included_episode_indices),
        "excluded_episode_count": len(receipt.excluded_episode_indices),
        "excluded_episode_indices": receipt.excluded_episode_indices,
        "issue_count": len(receipt.issues),
        "modality_validity_path": str(validity_path),
        "modality_validity_before_sha256": (
            None if not current_bytes else _sha256_bytes(current_bytes)
        ),
        "modality_validity_after_sha256": _sha256_bytes(desired_bytes),
        "source_inventory_verified": inventory_verified,
        "idempotent": idempotent,
    }
    return CameraVideoIntegrityTrainingPlan(
        **payload,
        plan_hash=canonical_content_hash(
            payload,
            domain="dexteleop.camera_video_integrity_training_plan.v1",
        ),
    )


def apply_camera_video_integrity_training_exclusions(
    source_root: str | Path,
    prepared_view_root: str | Path,
    *,
    expected_plan_hash: str,
) -> CameraVideoIntegrityTrainingPlan:
    """Atomically merge the exact planned exclusions into modality validity."""

    source = Path(source_root).expanduser().resolve()
    meta = source / "meta"
    lock_path = meta / ".modality_validity.lock"
    if not meta.is_dir() or meta.is_symlink():
        raise CameraSemanticPreparedViewError(
            f"Source metadata directory is missing or unsafe: {meta}"
        )
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            plan = plan_camera_video_integrity_training_exclusions(
                source,
                prepared_view_root,
            )
            if plan.plan_hash != str(expected_plan_hash).strip():
                raise CameraSemanticPreparedViewError(
                    "Camera-video training exclusion expected-plan-hash mismatch: "
                    f"expected={expected_plan_hash} actual={plan.plan_hash}"
                )
            if plan.idempotent:
                return plan
            _, receipt = load_camera_semantic_video_integrity_exclusions(
                prepared_view_root,
                verify_payload_hashes=True,
            )
            assert receipt is not None
            path = Path(plan.modality_validity_path)
            desired = _desired_validity(
                _read_object(path),
                manifest_hash=plan.prepared_view_manifest_hash,
                receipt=receipt,
            )
            payload = _json_bytes(desired)
            if _sha256_bytes(payload) != plan.modality_validity_after_sha256:
                raise CameraSemanticPreparedViewError(
                    "Training validity output changed after planning."
                )
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                with temporary.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                directory_fd = os.open(meta, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if temporary.exists():
                    temporary.unlink()
            return plan
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


__all__ = [
    "CAMERA_VIDEO_INTEGRITY_TRAINING_POLICY",
    "CameraVideoIntegrityTrainingPlan",
    "apply_camera_video_integrity_training_exclusions",
    "plan_camera_video_integrity_training_exclusions",
]
