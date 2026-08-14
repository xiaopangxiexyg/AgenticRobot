"""Read-only inventory and verification for detachable dataset modalities."""

from .inventory import (
    ModalityArtifactInventoryError,
    PAYLOAD_HASH_MODES,
    PAYLOAD_HASH_PREFER_SEALED,
    PAYLOAD_HASH_REQUIRE_SEALED,
    PAYLOAD_HASH_SCAN,
    build_modality_artifact_inventory,
    inventory_source_binding_matches,
    modality_source_binding,
    verify_modality_artifact_inventory,
)
from .availability import (
    DETACHMENT_SCOPES,
    DETACHED_VERIFIED,
    EXTERNAL_BOUND,
    INCOMPLETE_OR_CORRUPT,
    SELF_CONTAINED,
    STALE_INVENTORY,
    ModalityAvailability,
    PAYLOAD_ONLY_DETACH,
    WHOLE_PREFIX_DETACH,
    resolve_modality_availability,
)
from .admission import (
    MODALITY_ARTIFACT_SIDECAR_RELATIVE,
    DatasetCaptureWritableError,
    UnownedTactileFilesError,
    assert_dataset_capture_integrity_compatible,
    assert_dataset_capture_writable,
)
from .evidence import (
    ModalityStorageEvidenceError,
    verified_detachment_scope,
)

__all__ = [
    "ModalityArtifactInventoryError",
    "PAYLOAD_HASH_MODES",
    "PAYLOAD_HASH_PREFER_SEALED",
    "PAYLOAD_HASH_REQUIRE_SEALED",
    "PAYLOAD_HASH_SCAN",
    "build_modality_artifact_inventory",
    "inventory_source_binding_matches",
    "modality_source_binding",
    "verify_modality_artifact_inventory",
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
    "MODALITY_ARTIFACT_SIDECAR_RELATIVE",
    "DatasetCaptureWritableError",
    "UnownedTactileFilesError",
    "assert_dataset_capture_integrity_compatible",
    "assert_dataset_capture_writable",
    "ModalityStorageEvidenceError",
    "verified_detachment_scope",
]
