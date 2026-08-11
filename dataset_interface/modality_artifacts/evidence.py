"""Read-only verification of durable modality storage evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.modality_artifacts import ModalityStorageTransferReceiptV1

from .admission import MODALITY_ARTIFACT_SIDECAR_RELATIVE
from .availability import PAYLOAD_ONLY_DETACH, WHOLE_PREFIX_DETACH


class ModalityStorageEvidenceError(ValueError):
    """Raised when persisted storage evidence is malformed or ambiguous."""


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModalityStorageEvidenceError(
            f"cannot read modality storage receipt {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ModalityStorageEvidenceError(
            f"modality storage receipt is not an object: {path}"
        )
    return payload


def verified_detachment_scope(
    dataset_root: str | Path,
    inventory_hash: str,
) -> str | None:
    """Return the currently effective durable retirement scope.

    ``None`` means no complete, independently verified retirement fact exists.
    A later rehydrate receipt clears earlier retirement evidence. All matching
    receipt contracts are parsed and hash-validated; malformed or temporally
    ambiguous evidence fails closed instead of being silently ignored.
    """

    root = Path(dataset_root).expanduser().resolve()
    receipt_dir = (
        root / MODALITY_ARTIFACT_SIDECAR_RELATIVE / "receipts"
    )
    if not receipt_dir.is_dir():
        return None
    lifecycle: list[ModalityStorageTransferReceiptV1] = []
    for path in sorted(receipt_dir.glob("*.json")):
        try:
            receipt = ModalityStorageTransferReceiptV1.from_dict(
                _read_object(path)
            )
        except Exception as exc:
            raise ModalityStorageEvidenceError(
                f"invalid modality storage receipt {path}: {exc}"
            ) from exc
        if (
            receipt.inventory_hash == str(inventory_hash)
            and receipt.status == "complete"
            and receipt.independent_copy_verified
            and receipt.operation
            in {
                "retire_hot_payloads",
                "retire_hot_module_tree",
                "rehydrate",
            }
        ):
            lifecycle.append(receipt)
    if not lifecycle:
        return None
    latest_time = max(receipt.completed_at for receipt in lifecycle)
    latest_at_time = tuple(
        receipt for receipt in lifecycle if receipt.completed_at == latest_time
    )
    latest_operations = {receipt.operation for receipt in latest_at_time}
    if len(latest_operations) != 1:
        raise ModalityStorageEvidenceError(
            "modality storage lifecycle receipts have an ambiguous latest timestamp"
        )
    latest = max(latest_at_time, key=lambda receipt: receipt.operation_id)
    if latest.operation == "rehydrate":
        return None
    if latest.operation == "retire_hot_module_tree":
        return WHOLE_PREFIX_DETACH
    return PAYLOAD_ONLY_DETACH


__all__ = [
    "ModalityStorageEvidenceError",
    "verified_detachment_scope",
]
