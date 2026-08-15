"""Finish one audited legacy hardlink group without deleting unverifiable data.

The script waits for the preceding July-25 group to retire, detaches relocation
entries 1/2/3 from their already verified independent NAS artifacts, then
exports entry 4 from actual payload bytes and detaches it.  Every unlink is
performed by the production transactional detach API after cold verification
and restore drill.  Re-running is safe: completed retire receipts are skipped.
"""

from __future__ import annotations

import argparse
import errno
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, TypeVar

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
_T = TypeVar("_T")


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _roots(plan_path: Path) -> tuple[Path, ...]:
    entries = _object(plan_path).get("entries")
    if not isinstance(entries, list):
        raise ValueError("relocation plan entries must be a list")
    return tuple(
        Path(str(entry["target"])).expanduser().resolve(strict=False)
        for entry in entries
    )


def _retired(root: Path) -> bool:
    receipts = root / "meta/modality_artifacts/tactile/receipts"
    if not receipts.is_dir():
        return False
    for path in receipts.glob("retire_hot_payloads.*.json"):
        try:
            receipt = ModalityStorageTransferReceiptV1.from_dict(_object(path))
        except Exception:
            continue
        if (
            receipt.operation == "retire_hot_payloads"
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


def _retry(label: str, operation: Callable[[], _T], max_delay: float) -> _T:
    delay = 5.0
    while True:
        try:
            return operation()
        except Exception as exc:
            if not _retryable(exc):
                raise
            print(
                json.dumps(
                    {
                        "phase": f"{label}_retry",
                        "delay_sec": delay,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(delay)
            delay = min(delay * 2.0, max_delay)


def _detach(
    root: Path,
    cold_root: Path,
    *,
    scan_payload_hashes: bool,
    max_delay: float,
    active_roots: tuple[Path, ...],
    allow_disjoint_collection: bool,
) -> None:
    if _retired(root):
        print(
            json.dumps(
                {"phase": "detach_already_complete", "dataset_root": str(root)},
                sort_keys=True,
            ),
            flush=True,
        )
        return
    def apply_once():
        proposal = plan_modality_detach(
            root,
            cold_root,
            min_free_bytes=1024**3,
            allow_legacy_hardlinked_source=True,
            legacy_scan_payload_hashes=scan_payload_hashes,
        )
        if not proposal.ready:
            raise RuntimeError(
                "detach plan is not ready: " + "; ".join(proposal.errors)
            )
        return detach_modality(
            root,
            cold_root,
            expected_plan_hash=proposal.plan_hash,
            confirm_dataset_fingerprint=proposal.dataset_fingerprint,
            min_free_bytes=1024**3,
            source_storage_id="hot.realman_dmtac",
            target_storage_id="nas2.tactile_archive",
            confirm_no_active_readers=True,
            allow_legacy_hardlinked_source=True,
            legacy_scan_payload_hashes=scan_payload_hashes,
            allow_disjoint_active_collection=allow_disjoint_collection,
            active_collection_dataset_roots=active_roots,
        )

    result = _retry(
        "detach",
        apply_once,
        max_delay,
    )
    print(
        json.dumps(
            {"phase": "detach_complete", "result": result.to_dict()},
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--cold-root", required=True, type=Path)
    parser.add_argument("--wait-entry-index", type=int, default=11)
    parser.add_argument("--verified-entry-index", action="append", type=int, default=[])
    parser.add_argument("--scan-export-entry-index", type=int, default=4)
    parser.add_argument("--wait-timeout-sec", type=float, default=86400.0)
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

    roots = _roots(args.plan.expanduser().resolve(strict=True))
    cold_root = args.cold_root.expanduser().resolve(strict=True)
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
    wait_root = roots[int(args.wait_entry_index)]
    selected_indices = {
        int(args.wait_entry_index),
        int(args.scan_export_entry_index),
        *(int(index) for index in (args.verified_entry_index or (1, 2, 3))),
    }
    missing = [str(roots[index]) for index in sorted(selected_indices) if not roots[index].is_dir()]
    if missing:
        raise FileNotFoundError(f"selected dataset roots are missing: {missing}")
    deadline = time.monotonic() + float(args.wait_timeout_sec)
    while not _retired(wait_root):
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for the preceding detach receipt")
        time.sleep(float(args.poll_sec))
    print(
        json.dumps(
            {"phase": "preceding_group_complete", "dataset_root": str(wait_root)},
            sort_keys=True,
        ),
        flush=True,
    )

    verified_indices = tuple(args.verified_entry_index or (1, 2, 3))
    for index in verified_indices:
        _detach(
            roots[int(index)],
            cold_root,
            scan_payload_hashes=False,
            max_delay=float(args.retry_max_delay_sec),
            active_roots=active_roots,
            allow_disjoint_collection=allow_disjoint_collection,
        )

    scan_root = roots[int(args.scan_export_entry_index)]
    if not _retired(scan_root):
        def export_once():
            export_plan = plan_modality_export(
                scan_root,
                cold_root,
                min_free_bytes=1024**3,
                allow_legacy_source=True,
                legacy_scan_payload_hashes=True,
                cold_staging_namespace=str(args.cold_staging_namespace),
            )
            if not export_plan.ready:
                raise RuntimeError(
                    "scan export plan is not ready: " + "; ".join(export_plan.errors)
                )
            return export_modality(
                scan_root,
                cold_root,
                expected_plan_hash=export_plan.plan_hash,
                confirm_dataset_fingerprint=export_plan.dataset_fingerprint,
                min_free_bytes=1024**3,
                source_storage_id="hot.realman_dmtac",
                target_storage_id="nas2.tactile_archive",
                confirm_no_active_readers=True,
                allow_legacy_source=True,
                legacy_scan_payload_hashes=True,
                allow_disjoint_active_collection=allow_disjoint_collection,
                active_collection_dataset_roots=active_roots,
                cold_staging_namespace=str(args.cold_staging_namespace),
            )

        exported = _retry(
            "scan_export",
            export_once,
            float(args.retry_max_delay_sec),
        )
        print(
            json.dumps(
                {"phase": "scan_export_complete", "result": exported.to_dict()},
                sort_keys=True,
            ),
            flush=True,
        )
        _detach(
            scan_root,
            cold_root,
            scan_payload_hashes=True,
            max_delay=float(args.retry_max_delay_sec),
            active_roots=active_roots,
            allow_disjoint_collection=allow_disjoint_collection,
        )
    print(json.dumps({"phase": "group_complete"}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
