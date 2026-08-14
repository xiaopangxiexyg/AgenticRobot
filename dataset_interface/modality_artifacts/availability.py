"""Derived availability for a hash-bound detachable modality inventory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dataset_contracts.modality_artifacts import ModalityArtifactInventoryV1
from .inventory import _sha256_file, inventory_source_binding_matches


SELF_CONTAINED = "SELF_CONTAINED"
EXTERNAL_BOUND = "EXTERNAL_BOUND"
DETACHED_VERIFIED = "DETACHED_VERIFIED"
STALE_INVENTORY = "STALE_INVENTORY"
INCOMPLETE_OR_CORRUPT = "INCOMPLETE_OR_CORRUPT"
PAYLOAD_ONLY_DETACH = "payload_only"
WHOLE_PREFIX_DETACH = "whole_prefix"
DETACHMENT_SCOPES = (PAYLOAD_ONLY_DETACH, WHOLE_PREFIX_DETACH)


@dataclass(frozen=True)
class ModalityAvailability:
    status: str
    inventory_hash: str
    present_payload_count: int
    missing_payload_count: int
    reason: str = ""

    @property
    def core_training_eligible(self) -> bool:
        return self.status in {
            SELF_CONTAINED,
            EXTERNAL_BOUND,
            DETACHED_VERIFIED,
        }

    @property
    def tactile_read_eligible(self) -> bool:
        return self.status in {SELF_CONTAINED, EXTERNAL_BOUND}


def _matches(
    path: Path,
    *,
    storage_root: Path,
    size_bytes: int,
    sha256: str,
) -> bool:
    try:
        path.relative_to(storage_root)
    except ValueError:
        return False
    cursor = path
    while cursor != storage_root:
        if cursor.is_symlink():
            return False
        if cursor.parent == cursor:
            return False
        cursor = cursor.parent
    if not path.is_file():
        return False
    stat = path.stat()
    return (
        int(stat.st_nlink) == 1
        and int(stat.st_size) == int(size_bytes)
        and _sha256_file(path) == str(sha256)
    )


def resolve_modality_availability(
    dataset_root: str | Path,
    inventory: ModalityArtifactInventoryV1,
    *,
    artifact_root: str | Path | None = None,
    detachment_scope: str = PAYLOAD_ONLY_DETACH,
) -> ModalityAvailability:
    """Resolve storage availability without changing capture validity facts."""

    root = Path(dataset_root).expanduser().resolve()
    scope = str(detachment_scope or "").strip().lower().replace("-", "_")
    if scope not in DETACHMENT_SCOPES:
        return ModalityAvailability(
            INCOMPLETE_OR_CORRUPT,
            inventory.semantic_content_hash,
            0,
            len(tuple(a for a in inventory.artifacts if a.role == "payload")),
            f"unsupported detachment scope: {detachment_scope!r}",
        )
    try:
        source_binding_matches = inventory_source_binding_matches(root, inventory)
    except (OSError, ValueError) as exc:
        return ModalityAvailability(
            INCOMPLETE_OR_CORRUPT,
            inventory.semantic_content_hash,
            0,
            len(tuple(a for a in inventory.artifacts if a.role == "payload")),
            f"cannot validate source identity: {type(exc).__name__}: {exc}",
        )
    if not source_binding_matches:
        return ModalityAvailability(
            STALE_INVENTORY,
            inventory.semantic_content_hash,
            0,
            len(tuple(a for a in inventory.artifacts if a.role == "payload")),
            "source modality reference binding changed",
        )

    manifests = tuple(a for a in inventory.artifacts if a.role == "manifest")
    payloads = tuple(a for a in inventory.artifacts if a.role == "payload")
    all_artifacts = tuple(inventory.artifacts)
    tactile_root = root / "tactile"
    discovered_hot = {
        path.relative_to(root).as_posix()
        for path in tactile_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    unknown_hot = discovered_hot - {
        artifact.logical_path for artifact in inventory.artifacts
    }
    if unknown_hot:
        return ModalityAvailability(
            INCOMPLETE_OR_CORRUPT,
            inventory.semantic_content_hash,
            0,
            len(payloads),
            f"hot modality tree contains unowned files: {sorted(unknown_hot)[:5]}",
        )
    if scope == WHOLE_PREFIX_DETACH:
        matching_hot = 0
        missing_hot = 0
        for artifact in all_artifacts:
            path = root / artifact.logical_path
            if not path.exists():
                missing_hot += 1
            elif _matches(
                path,
                storage_root=root,
                size_bytes=artifact.size_bytes,
                sha256=artifact.sha256,
            ):
                matching_hot += 1
            else:
                return ModalityAvailability(
                    INCOMPLETE_OR_CORRUPT,
                    inventory.semantic_content_hash,
                    0,
                    len(payloads),
                    f"hot module artifact is changed or unsafe: {artifact.logical_path}",
                )
        if matching_hot == len(all_artifacts):
            return ModalityAvailability(
                SELF_CONTAINED,
                inventory.semantic_content_hash,
                len(payloads),
                0,
            )
        if missing_hot != len(all_artifacts) or tactile_root.exists():
            return ModalityAvailability(
                INCOMPLETE_OR_CORRUPT,
                inventory.semantic_content_hash,
                0,
                len(payloads),
                "partial hot module availability is not a supported whole-prefix state",
            )
        if artifact_root is None:
            return ModalityAvailability(
                DETACHED_VERIFIED,
                inventory.semantic_content_hash,
                0,
                len(payloads),
            )
        external_root = Path(artifact_root).expanduser().resolve()
        for artifact in all_artifacts:
            if not _matches(
                external_root / artifact.logical_path,
                storage_root=external_root,
                size_bytes=artifact.size_bytes,
                sha256=artifact.sha256,
            ):
                return ModalityAvailability(
                    INCOMPLETE_OR_CORRUPT,
                    inventory.semantic_content_hash,
                    0,
                    len(payloads),
                    f"external module artifact is missing or changed: {artifact.logical_path}",
                )
        return ModalityAvailability(
            EXTERNAL_BOUND,
            inventory.semantic_content_hash,
            len(payloads),
            0,
        )
    for artifact in manifests:
        if not _matches(
            root / artifact.logical_path,
            storage_root=root,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
        ):
            return ModalityAvailability(
                INCOMPLETE_OR_CORRUPT,
                inventory.semantic_content_hash,
                0,
                len(payloads),
                f"hot manifest is missing or changed: {artifact.logical_path}",
            )

    hot_present = 0
    hot_missing = 0
    for artifact in payloads:
        path = root / artifact.logical_path
        if not path.exists():
            hot_missing += 1
        elif _matches(
            path,
            storage_root=root,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
        ):
            hot_present += 1
        else:
            return ModalityAvailability(
                INCOMPLETE_OR_CORRUPT,
                inventory.semantic_content_hash,
                hot_present,
                hot_missing,
                f"hot payload is changed or unsafe: {artifact.logical_path}",
            )
    if artifact_root is None:
        if hot_present == len(payloads):
            return ModalityAvailability(
                SELF_CONTAINED,
                inventory.semantic_content_hash,
                hot_present,
                0,
            )
        if hot_missing == len(payloads):
            return ModalityAvailability(
                DETACHED_VERIFIED,
                inventory.semantic_content_hash,
                0,
                hot_missing,
            )
        return ModalityAvailability(
            INCOMPLETE_OR_CORRUPT,
            inventory.semantic_content_hash,
            hot_present,
            hot_missing,
            "partial hot payload availability is not a supported M2 state",
        )

    external_root = Path(artifact_root).expanduser().resolve()
    for artifact in payloads:
        if not _matches(
            external_root / artifact.logical_path,
            storage_root=external_root,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
        ):
            return ModalityAvailability(
                INCOMPLETE_OR_CORRUPT,
                inventory.semantic_content_hash,
                hot_present,
                hot_missing,
                f"external payload is missing or changed: {artifact.logical_path}",
            )
    return ModalityAvailability(
        EXTERNAL_BOUND,
        inventory.semantic_content_hash,
        len(payloads),
        0,
    )


__all__ = [
    "DETACHMENT_SCOPES",
    "DETACHED_VERIFIED",
    "EXTERNAL_BOUND",
    "INCOMPLETE_OR_CORRUPT",
    "SELF_CONTAINED",
    "STALE_INVENTORY",
    "ModalityAvailability",
    "PAYLOAD_ONLY_DETACH",
    "WHOLE_PREFIX_DETACH",
    "resolve_modality_availability",
]
