"""Plan or transactionally detach one relocation-plan tactile dataset.

The dataset is selected by immutable relocation-plan index so remote safety
wrappers never need a hardware-looking dataset leaf in the command line.  The
apply path delegates all copying, independent verification, restore drill,
quiescence checks, quarantine swap, and receipt publication to
``collection_runtime.modality_storage``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from collections.abc import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collection_runtime.modality_storage import (
    SAMPLED_REVALIDATION_V1,
    ModalityStorageError,
    active_collection_dataset_roots_for_date,
    detach_modality,
    plan_modality_detach,
)
from collection_runtime.process_quiescence import active_formal_collection_processes
from dataset_contracts.modality_artifacts import ModalityStorageTransferReceiptV1


def _active_collection_interrupted_apply(exc: BaseException) -> bool:
    return isinstance(exc, ModalityStorageError) and str(exc).startswith(
        "formal collection is active:"
    )


def _wait_for_global_collection_quiescence(
    *,
    deadline: float,
    poll_sec: float,
    active_probe: Callable[[], tuple[str, ...]] = active_formal_collection_processes,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait until no formal collection process is active.

    This is only a scheduling convenience.  The transactional detach path
    independently repeats the same fail-closed quiescence check immediately
    before every destructive boundary.
    """

    next_status_at = 0.0
    while True:
        active = active_probe()
        if not active:
            print(
                json.dumps(
                    {"phase": "formal_collection_quiescent"},
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        now = monotonic()
        if now >= deadline:
            raise TimeoutError("timed out waiting for formal collection to stop")
        if now >= next_status_at:
            print(
                json.dumps(
                    {
                        "phase": "waiting_for_formal_collection_quiescence",
                        "active_process_count": len(active),
                        "active_processes": list(active[:3]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            next_status_at = now + 300.0
        sleep(min(poll_sec, max(0.0, deadline - now)))


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _has_complete_retire_receipt(root: Path) -> bool:
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


def _process_identity(pid: int) -> tuple[int, str] | None:
    base = Path("/proc") / str(pid)
    try:
        stat = (base / "stat").read_text(encoding="utf-8")
        command = (base / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    end_comm = stat.rfind(")")
    fields_after_comm = stat[end_comm + 2 :].split() if end_comm >= 0 else ()
    if len(fields_after_comm) <= 19:
        raise RuntimeError(f"cannot parse process identity for pid {pid}")
    return int(fields_after_comm[19]), command.strip()


def _wait_for_predecessor(
    pid: int,
    *,
    timeout_sec: float,
    poll_sec: float,
    identity_probe: Callable[[int], tuple[int, str] | None] = _process_identity,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    initial = identity_probe(pid)
    if initial is None:
        return
    start_ticks, command = initial
    if "detach_verified_realman_tactile.py" not in command:
        raise RuntimeError(
            f"refusing to wait for unrelated predecessor pid {pid}: {command}"
        )
    deadline = monotonic() + timeout_sec
    next_status_at = 0.0
    while True:
        current = identity_probe(pid)
        if current is None or current[0] != start_ticks:
            return
        now = monotonic()
        if now >= deadline:
            raise TimeoutError(f"timed out waiting for predecessor pid {pid}")
        if now >= next_status_at:
            print(
                json.dumps(
                    {"phase": "waiting_for_predecessor", "pid": pid},
                    sort_keys=True,
                ),
                flush=True,
            )
            next_status_at = now + 300.0
        sleep(min(poll_sec, max(0.0, deadline - now)))


def _dataset_root(plan_path: Path, entry_index: int) -> Path:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("relocation plan entries must be a list")
    if entry_index < 0 or entry_index >= len(entries):
        raise ValueError(f"entry index out of range: {entry_index}")
    entry = entries[entry_index]
    if not isinstance(entry, dict) or not entry.get("target"):
        raise ValueError(f"invalid relocation plan entry: {entry_index}")
    return Path(str(entry["target"])).resolve(strict=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--entry-index", required=True, type=int)
    parser.add_argument("--cold-root", required=True, type=Path)
    parser.add_argument("--min-free-gb", type=float, default=1.0)
    parser.add_argument("--exclude-active-collection-date", default="")
    parser.add_argument(
        "--allow-disjoint-active-collection",
        action="store_true",
        help=(
            "Allow production maintenance while every active formal collection "
            "has a dynamically provable dataset root disjoint from this dataset."
        ),
    )
    parser.add_argument(
        "--wait-for-no-active-collection",
        action="store_true",
        help=(
            "Wait for global formal-collection quiescence before planning and "
            "retry from a fresh plan if collection restarts before apply."
        ),
    )
    parser.add_argument("--wait-timeout-sec", type=float, default=216000.0)
    parser.add_argument("--poll-sec", type=float, default=30.0)
    parser.add_argument("--wait-for-predecessor-pid", type=int, default=0)
    parser.add_argument("--predecessor-wait-timeout-sec", type=float, default=216000.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--sampled-revalidation",
        action="store_true",
        help=(
            "Keep exact full-file copy/hash verification, but bound expensive "
            "reader tail-decode and restore-drill groups using the fixed "
            "sampled_v1 profile. Default remains full revalidation."
        ),
    )
    parser.add_argument(
        "--confirm-legacy-hardlink-detach",
        action="store_true",
        help="Required with --apply; only this dataset's paths are retired.",
    )
    args = parser.parse_args()
    plan_path = args.plan.expanduser().resolve(strict=True)
    root = _dataset_root(plan_path, int(args.entry_index))
    cold_root = args.cold_root.expanduser().resolve(strict=True)
    min_free_bytes = int(float(args.min_free_gb) * 1024**3)
    revalidation_profile = (
        SAMPLED_REVALIDATION_V1 if args.sampled_revalidation else "full"
    )
    if args.wait_for_no_active_collection and args.exclude_active_collection_date:
        raise ValueError(
            "--wait-for-no-active-collection cannot be combined with "
            "--exclude-active-collection-date"
        )
    if args.wait_for_no_active_collection and args.allow_disjoint_active_collection:
        raise ValueError(
            "--wait-for-no-active-collection cannot be combined with "
            "--allow-disjoint-active-collection"
        )
    if float(args.poll_sec) <= 0.0:
        raise ValueError("--poll-sec must be positive")
    if float(args.wait_timeout_sec) <= 0.0:
        raise ValueError("--wait-timeout-sec must be positive")
    if float(args.predecessor_wait_timeout_sec) <= 0.0:
        raise ValueError("--predecessor-wait-timeout-sec must be positive")
    active_roots = (
        active_collection_dataset_roots_for_date(
            str(args.exclude_active_collection_date)
        )
        if args.exclude_active_collection_date
        else ()
    )
    if args.exclude_active_collection_date and not active_roots:
        raise RuntimeError("no exact active collection dataset leaf was resolved")
    if int(args.wait_for_predecessor_pid) > 0:
        _wait_for_predecessor(
            int(args.wait_for_predecessor_pid),
            timeout_sec=float(args.predecessor_wait_timeout_sec),
            poll_sec=float(args.poll_sec),
        )
        if _has_complete_retire_receipt(root):
            print(
                json.dumps(
                    {
                        "phase": "predecessor_detach_already_complete",
                        "dataset_root": str(root),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
    if not args.apply:
        proposal = plan_modality_detach(
            root,
            cold_root,
            min_free_bytes=min_free_bytes,
            allow_legacy_hardlinked_source=True,
            revalidation_profile=revalidation_profile,
        )
        print(json.dumps(proposal.to_dict(), ensure_ascii=False, sort_keys=True))
        return 0 if proposal.ready else 2
    if not args.confirm_legacy_hardlink_detach:
        raise ValueError("--apply requires --confirm-legacy-hardlink-detach")
    deadline = time.monotonic() + float(args.wait_timeout_sec)
    while True:
        if args.wait_for_no_active_collection:
            _wait_for_global_collection_quiescence(
                deadline=deadline,
                poll_sec=float(args.poll_sec),
            )
        print(
            json.dumps(
                {"phase": "detach_plan_started", "dataset_root": str(root)},
                sort_keys=True,
            ),
            flush=True,
        )
        proposal = plan_modality_detach(
            root,
            cold_root,
            min_free_bytes=min_free_bytes,
            allow_legacy_hardlinked_source=True,
            revalidation_profile=revalidation_profile,
        )
        if not proposal.ready:
            raise RuntimeError(
                "detach proposal is not ready: " + "; ".join(proposal.errors)
            )
        try:
            result = detach_modality(
                root,
                cold_root,
                expected_plan_hash=proposal.plan_hash,
                confirm_dataset_fingerprint=proposal.dataset_fingerprint,
                min_free_bytes=min_free_bytes,
                source_storage_id="hot.realman_dmtac",
                target_storage_id="nas2.tactile_archive",
                confirm_no_active_readers=True,
                allow_legacy_hardlinked_source=True,
                allow_disjoint_active_collection=(
                    bool(active_roots) or bool(args.allow_disjoint_active_collection)
                ),
                active_collection_dataset_roots=active_roots,
                revalidation_profile=revalidation_profile,
            )
            break
        except ModalityStorageError as exc:
            if not (
                args.wait_for_no_active_collection
                and _active_collection_interrupted_apply(exc)
            ):
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "formal collection restarted during detach planning and the "
                    "global wait deadline expired"
                ) from exc
            print(
                json.dumps(
                    {
                        "phase": "detach_replan_after_collection_restart",
                        "error": str(exc),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
