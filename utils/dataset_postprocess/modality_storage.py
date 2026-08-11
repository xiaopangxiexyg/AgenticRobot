"""Compatibility exports for collection-runtime-owned modality storage.

The postprocess CLI keeps its historical import path, while dataset locks,
journals, quarantine swaps, receipts, and recovery are owned by
``collection_runtime``.
"""

from collection_runtime.modality_storage import (
    DEFAULT_MIN_FREE_BYTES,
    ModalityStorageError,
    ModalityStoragePlan,
    ModalityStorageResult,
    detach_modality,
    export_modality,
    load_modality_inventory,
    modality_storage_status,
    move_modality,
    plan_modality_detach,
    plan_modality_export,
    plan_modality_rehydrate,
    recover_modality_storage_transaction,
    rehydrate_modality,
)


__all__ = [
    "DEFAULT_MIN_FREE_BYTES",
    "ModalityStorageError",
    "ModalityStoragePlan",
    "ModalityStorageResult",
    "detach_modality",
    "export_modality",
    "load_modality_inventory",
    "modality_storage_status",
    "move_modality",
    "plan_modality_detach",
    "plan_modality_export",
    "plan_modality_rehydrate",
    "recover_modality_storage_transaction",
    "rehydrate_modality",
]
