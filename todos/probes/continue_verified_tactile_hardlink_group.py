"""Wait for one verified detach, then export and detach its remaining graph node."""

from __future__ import annotations

import argparse
import errno
import json
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collection_runtime.modality_storage import (
    active_collection_dataset_roots_for_date,
    detach_modality,
    export_modality,
    plan_modality_detach,
    plan_modality_export,
)
from dataset_contracts.modality_artifacts import ModalityStorageTransferReceiptV1


_TRANSIENT = frozenset(
    {
        errno.EAGAIN,
        errno.EWOULDBLOCK,
        errno.EINTR,
        errno.EBUSY,
        errno.ETIMEDOUT,
        errno.ECONNRESET,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.EHOSTDOWN,
        errno.EHOSTUNREACH,
        errno.ESTALE,
    }
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _root(plan_path: Path, index: int) -> Path:
    entries = _object(plan_path).get("entries")
    if not isinstance(entries, list) or index < 0 or index >= len(entries):
        raise ValueError(f"invalid relocation entry index: {index}")
    return Path(str(entries[index]["target"])).resolve(strict=True)


def _retired(root: Path, inventory_hash: str) -> bool:
    receipts = root / "meta/modality_artifacts/tactile/receipts"
    if not receipts.is_dir():
        return False
    for path in receipts.glob("retire_hot_payloads.*.json"):
        try:
            receipt = ModalityStorageTransferReceiptV1.from_dict(_object(path))
        except Exception:
            continue
        if (
            receipt.inventory_hash == inventory_hash
            and receipt.operation == "retire_hot_payloads"
            and receipt.status == "complete"
            and receipt.independent_copy_verified
        ):
            return True
    return False


def _retryable(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno in _TRANSIENT:
            return True
        current = current.__cause__ or current.__context__
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--cold-root", required=True, type=Path)
    parser.add_argument("--wait-entry-index", required=True, type=int)
    parser.add_argument("--wait-inventory-hash", required=True)
    parser.add_argument("--next-entry-index", required=True, type=int)
    parser.add_argument("--wait-timeout-sec", type=float, default=43200.0)
    parser.add_argument("--poll-sec", type=float, default=30.0)
    parser.add_argument("--retry-max-delay-sec", type=float, default=60.0)
    parser.add_argument("--cold-staging-namespace", default="")
    parser.add_argument("--exclude-active-collection-date", default="")
    parser.add_argument(
        "--allow-disjoint-active-collection",
        action="store_true",
        help=(
            "Allow production maintenance only when every active collection "
            "dataset root is dynamically proven disjoint from the source."
        ),
    )
    args = parser.parse_args()

    plan_path = args.plan.expanduser().resolve(strict=True)
    cold_root = args.cold_root.expanduser().resolve(strict=True)
    wait_root = _root(plan_path, args.wait_entry_index)
    next_root = _root(plan_path, args.next_entry_index)
    active_roots = (
        active_collection_dataset_roots_for_date(
            str(args.exclude_active_collection_date)
        )
        if args.exclude_active_collection_date
        else ()
    )
    if args.exclude_active_collection_date and not active_roots:
        raise RuntimeError("no exact active collection dataset leaf was resolved")
    allow_disjoint_collection = bool(active_roots) or bool(
        args.allow_disjoint_active_collection
    )
    deadline = time.monotonic() + float(args.wait_timeout_sec)
    while not _retired(wait_root, str(args.wait_inventory_hash)):
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for the preceding detach receipt")
        time.sleep(float(args.poll_sec))
    print(
        json.dumps(
            {"phase": "preceding_detach_complete", "dataset_root": str(wait_root)},
            sort_keys=True,
        ),
        flush=True,
    )

    delay = 5.0
    while True:
        try:
            export_plan = plan_modality_export(
                next_root,
                cold_root,
                min_free_bytes=1024**3,
                allow_legacy_source=True,
                cold_staging_namespace=str(args.cold_staging_namespace),
            )
            if not export_plan.ready:
                raise RuntimeError(
                    "export plan is not ready: " + "; ".join(export_plan.errors)
                )
            exported = export_modality(
                next_root,
                cold_root,
                expected_plan_hash=export_plan.plan_hash,
                confirm_dataset_fingerprint=export_plan.dataset_fingerprint,
                min_free_bytes=1024**3,
                source_storage_id="hot.realman_dmtac",
                target_storage_id="nas2.tactile_archive",
                confirm_no_active_readers=True,
                allow_legacy_source=True,
                allow_disjoint_active_collection=allow_disjoint_collection,
                active_collection_dataset_roots=active_roots,
                cold_staging_namespace=str(args.cold_staging_namespace),
            )
            break
        except Exception as exc:
            if not _retryable(exc):
                raise
            print(
                json.dumps(
                    {"phase": "export_retry", "delay_sec": delay, "error": str(exc)},
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(delay)
            delay = min(delay * 2.0, float(args.retry_max_delay_sec))
    print(json.dumps({"phase": "export_complete", "result": exported.to_dict()}, sort_keys=True), flush=True)

    delay = 5.0
    while True:
        try:
            detach_plan = plan_modality_detach(
                next_root,
                cold_root,
                min_free_bytes=1024**3,
                allow_legacy_hardlinked_source=True,
            )
            if not detach_plan.ready:
                raise RuntimeError(
                    "detach plan is not ready: " + "; ".join(detach_plan.errors)
                )
            detached = detach_modality(
                next_root,
                cold_root,
                expected_plan_hash=detach_plan.plan_hash,
                confirm_dataset_fingerprint=detach_plan.dataset_fingerprint,
                min_free_bytes=1024**3,
                source_storage_id="hot.realman_dmtac",
                target_storage_id="nas2.tactile_archive",
                confirm_no_active_readers=True,
                allow_legacy_hardlinked_source=True,
                allow_disjoint_active_collection=allow_disjoint_collection,
                active_collection_dataset_roots=active_roots,
            )
            break
        except Exception as exc:
            if not _retryable(exc):
                raise
            print(
                json.dumps(
                    {"phase": "detach_retry", "delay_sec": delay, "error": str(exc)},
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(delay)
            delay = min(delay * 2.0, float(args.retry_max_delay_sec))
    print(json.dumps({"phase": "detach_complete", "result": detached.to_dict()}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
