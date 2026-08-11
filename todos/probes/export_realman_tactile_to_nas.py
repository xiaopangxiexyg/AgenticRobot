"""Source-preserving, resumable export of canonical Realman tactile artifacts.

This operator tool deliberately calls the production modality-export API.  It
never detaches, retires, renames, or deletes a hot artifact.  Completed exports
are content-addressed below the cold root and carry their own frozen inventory
and verified transfer receipt.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

from collection_runtime.modality_storage import (
    ModalityStorageError,
    active_collection_dataset_roots_for_date,
    export_modality,
    plan_modality_export,
)


_RETRYABLE_EXPORT_ERRNOS = frozenset(
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


class _NasMountUnavailable(RuntimeError):
    """The configured NAS path no longer resolves to the frozen mount."""


def _mount_point(path: Path) -> tuple[Path, str, str]:
    resolved = path.resolve(strict=False)
    matches: list[tuple[int, Path, str, str]] = []
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        separator = fields.index("-")
        mount_point = Path(fields[4].replace("\\040", " "))
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        matches.append(
            (
                len(mount_point.as_posix()),
                mount_point,
                fields[separator + 1],
                fields[separator + 2],
            )
        )
    if not matches:
        raise RuntimeError(f"cannot resolve mount for {path}")
    _, mount_point, fs_type, source = max(matches)
    return mount_point, fs_type, source


def _read_plan(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("relocation plan entries must be a list")
    return [dict(entry) for entry in entries]


def _tactile_size(root: Path) -> int:
    tactile = root / "tactile"
    return sum(
        path.stat().st_size
        for path in tactile.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _append_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _retryable_export_error(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _NasMountUnavailable):
            return True
        if isinstance(current, OSError) and current.errno in _RETRYABLE_EXPORT_ERRNOS:
            return True
        current = current.__cause__ or current.__context__
    return False


def _assert_same_nas_mount(
    cold_root: Path,
    *,
    expected_mount: Path,
    expected_fs: str,
    expected_source: str,
    expected_device: int,
) -> None:
    try:
        mount, fs_type, source = _mount_point(cold_root)
        device = int(cold_root.stat().st_dev)
    except OSError as exc:
        raise _NasMountUnavailable(f"NAS path is unavailable: {cold_root}") from exc
    if (
        mount != expected_mount
        or fs_type != expected_fs
        or source != expected_source
        or device != expected_device
    ):
        raise _NasMountUnavailable(
            "NAS mount identity changed or disappeared: "
            f"expected=({expected_mount},{expected_fs},{expected_source},{expected_device}) "
            f"actual=({mount},{fs_type},{source},{device})"
        )


def _completed_entry_indices(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    completed: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("status") == "complete":
            index = row.get("entry_index")
            if isinstance(index, int):
                completed.add(index)
    return completed


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _lower_maintenance_priority() -> None:
    try:
        os.nice(10)
    except OSError:
        pass
    try:
        subprocess.run(
            ("ionice", "-c", "3", "-p", str(os.getpid())),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--relocation-plan", required=True, type=Path)
    parser.add_argument("--cold-root", required=True, type=Path)
    parser.add_argument("--entry-index", action="append", type=int, default=[])
    parser.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=16.0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--state-root",
        type=Path,
        default=None,
        help="local durable run-state root; recommended when cold storage is a NAS",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="retry only transient NAS/filesystem failures and reuse verified partial files",
    )
    parser.add_argument(
        "--resume-max-attempts",
        type=int,
        default=0,
        help="maximum attempts per dataset with --auto-resume; 0 means unlimited",
    )
    parser.add_argument("--resume-initial-delay-sec", type=float, default=5.0)
    parser.add_argument("--resume-max-delay-sec", type=float, default=60.0)
    parser.add_argument(
        "--cold-staging-namespace",
        default="",
        help=(
            "explicit break-glass namespace for a fresh NAS lock and partial "
            "tree; the default lock/partial are preserved"
        ),
    )
    parser.add_argument("--exclude-root", action="append", type=Path, default=[])
    parser.add_argument(
        "--exclude-active-collection-date",
        default="",
        help=(
            "derive exact active dataset leaf/leaves for YYYY-MM-DD from live "
            "collection command lines and exclude them"
        ),
    )
    parser.add_argument("--allow-disjoint-active-collection", action="store_true")
    parser.add_argument(
        "--legacy-scan-payload-hashes",
        action="store_true",
        help=(
            "Hash actual legacy tactile payload bytes instead of trusting sealed "
            "manifest hashes. Intended only for audited recovery exports."
        ),
    )
    args = parser.parse_args()

    if args.resume_max_attempts < 0:
        raise ValueError("--resume-max-attempts must be >= 0")
    if args.resume_initial_delay_sec <= 0.0:
        raise ValueError("--resume-initial-delay-sec must be > 0")
    if args.resume_max_delay_sec < args.resume_initial_delay_sec:
        raise ValueError(
            "--resume-max-delay-sec must be >= --resume-initial-delay-sec"
        )

    source_root = args.source_root.expanduser().resolve(strict=True)
    plan_path = args.relocation_plan.expanduser().resolve(strict=True)
    cold_root = args.cold_root.expanduser().resolve(strict=True)
    explicit_excluded_roots = tuple(
        path.expanduser().resolve(strict=False) for path in args.exclude_root
    )
    discovered_active_roots: tuple[Path, ...] = ()
    if args.exclude_active_collection_date:
        discovered_active_roots = active_collection_dataset_roots_for_date(
            args.exclude_active_collection_date
        )
        if not discovered_active_roots:
            raise RuntimeError(
                "no exact active collection dataset leaf could be resolved for "
                f"date {args.exclude_active_collection_date!r}"
            )
    excluded_roots = tuple(
        dict.fromkeys((*explicit_excluded_roots, *discovered_active_roots))
    )
    source_mount, source_fs, source_device_name = _mount_point(source_root)
    cold_mount, cold_fs, cold_device_name = _mount_point(cold_root)
    cold_device = int(cold_root.stat().st_dev)
    if source_root.stat().st_dev == cold_root.stat().st_dev:
        raise RuntimeError("cold root is on the source filesystem")
    if cold_mount == Path("/") or cold_fs.lower() not in {"cifs", "nfs", "nfs4"}:
        raise RuntimeError(
            f"cold root is not on an approved NAS mount: {cold_mount} ({cold_fs})"
        )
    if not os.access(cold_root, os.W_OK):
        raise RuntimeError(f"cold root is not writable: {cold_root}")

    entries = _read_plan(plan_path)
    indexed = list(enumerate(entries))
    if args.entry_index:
        selected_indices = set(args.entry_index)
        missing = sorted(selected_indices - set(range(len(entries))))
        if missing:
            raise ValueError(f"entry indices out of range: {missing}")
        indexed = [row for row in indexed if row[0] in selected_indices]
    elif not args.all_eligible:
        raise ValueError("choose --entry-index or --all-eligible")

    candidates = []
    for index, entry in indexed:
        root = Path(str(entry["target"])).expanduser().resolve(strict=True)
        try:
            root.relative_to(source_root)
        except ValueError as exc:
            raise RuntimeError(f"dataset is outside source root: {root}") from exc
        if any(_paths_overlap(root, excluded) for excluded in excluded_roots):
            if not args.quiet:
                print(
                    json.dumps(
                        {
                            "entry_index": index,
                            "dataset_root": str(root),
                            "status": "excluded",
                            "excluded_roots": [str(path) for path in excluded_roots],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            continue
        candidates.append((index, root, _tactile_size(root)))
    candidates.sort(key=lambda row: (row[2], row[0]))

    if args.state_root is None:
        run_root = cold_root / "_export_runs"
    else:
        run_root = args.state_root.expanduser().resolve(strict=False)
        run_root.mkdir(parents=True, exist_ok=True)
        if int(run_root.stat().st_dev) == cold_device:
            raise RuntimeError("--state-root must not be on the NAS filesystem")
    run_id = str(args.run_id or f"run-{os.getpid()}").strip()
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise ValueError(f"invalid run id: {run_id!r}")
    result_path = run_root / run_id / "results.jsonl"
    completed_before_start = _completed_entry_indices(result_path)
    header = {
        "type": "tactile_export_run",
        "source_root": str(source_root),
        "cold_root": str(cold_root),
        "source_mount": str(source_mount),
        "source_fs": source_fs,
        "source_device": source_device_name,
        "cold_mount": str(cold_mount),
        "cold_fs": cold_fs,
        "cold_device": cold_device_name,
        "candidate_count": len(candidates),
        "apply": bool(args.apply),
        "excluded_roots": [str(path) for path in excluded_roots],
        "allow_disjoint_active_collection": bool(
            args.allow_disjoint_active_collection
        ),
        "auto_resume": bool(args.auto_resume),
        "resume_max_attempts": int(args.resume_max_attempts),
        "cold_staging_namespace": str(args.cold_staging_namespace or ""),
        "state_root": str(run_root),
        "completed_before_start": sorted(completed_before_start),
    }
    if args.apply:
        _append_result(result_path, header)
    if not args.quiet:
        print(json.dumps(header, ensure_ascii=False, sort_keys=True), flush=True)

    completed = 0
    skipped = 0
    if args.apply and args.allow_disjoint_active_collection:
        _lower_maintenance_priority()
    min_free_bytes = int(float(args.min_free_gb) * 1024**3)
    for sequence, (index, root, estimated_bytes) in enumerate(candidates, start=1):
        prefix = {
            "entry_index": index,
            "sequence": sequence,
            "candidate_count": len(candidates),
            "dataset_root": str(root),
            "estimated_tactile_bytes": estimated_bytes,
        }
        if args.apply and index in completed_before_start:
            row = {**prefix, "status": "already_complete_in_run_state"}
            if not args.quiet:
                print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
            skipped += 1
            continue
        if not args.quiet:
            print(json.dumps({**prefix, "phase": "planning"}, sort_keys=True), flush=True)
        try:
            plan = plan_modality_export(
                root,
                cold_root,
                min_free_bytes=min_free_bytes,
                allow_legacy_source=True,
                legacy_scan_payload_hashes=bool(args.legacy_scan_payload_hashes),
                cold_staging_namespace=args.cold_staging_namespace,
            )
        except Exception as exc:
            row = {
                **prefix,
                "status": "ineligible",
                "error": f"{type(exc).__name__}: {exc}",
            }
            if args.apply:
                _append_result(result_path, row)
            if not args.quiet:
                print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
            skipped += 1
            continue
        plan_row = {**prefix, "phase": "planned", "plan": plan.to_dict()}
        if not args.quiet:
            print(json.dumps(plan_row, ensure_ascii=False, sort_keys=True), flush=True)
        if not plan.ready:
            row = {**prefix, "status": "ineligible", "errors": list(plan.errors)}
            if args.apply:
                _append_result(result_path, row)
            if not args.quiet:
                print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
            skipped += 1
            continue
        if not args.apply:
            continue
        attempt = 0
        retry_delay = float(args.resume_initial_delay_sec)
        while True:
            attempt += 1
            try:
                _assert_same_nas_mount(
                    cold_root,
                    expected_mount=cold_mount,
                    expected_fs=cold_fs,
                    expected_source=cold_device_name,
                    expected_device=cold_device,
                )
                result = export_modality(
                    root,
                    cold_root,
                    expected_plan_hash=plan.plan_hash,
                    confirm_dataset_fingerprint=plan.dataset_fingerprint,
                    min_free_bytes=min_free_bytes,
                    source_storage_id="hot.realman_dmtac",
                    target_storage_id="nas2.tactile_archive",
                    confirm_no_active_readers=True,
                    allow_legacy_source=True,
                    legacy_scan_payload_hashes=bool(
                        args.legacy_scan_payload_hashes
                    ),
                    allow_disjoint_active_collection=(
                        args.allow_disjoint_active_collection
                    ),
                    active_collection_dataset_roots=excluded_roots,
                    cold_staging_namespace=args.cold_staging_namespace,
                )
                break
            except Exception as exc:
                retryable = bool(args.auto_resume and _retryable_export_error(exc))
                exhausted = bool(
                    args.resume_max_attempts
                    and attempt >= args.resume_max_attempts
                )
                if retryable and not exhausted:
                    row = {
                        **prefix,
                        "status": "retry_scheduled",
                        "attempt": attempt,
                        "delay_sec": retry_delay,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    _append_result(result_path, row)
                    if not args.quiet:
                        print(
                            json.dumps(row, ensure_ascii=False, sort_keys=True),
                            flush=True,
                        )
                    time.sleep(retry_delay)
                    retry_delay = min(
                        retry_delay * 2.0,
                        float(args.resume_max_delay_sec),
                    )
                    continue
                row = {
                    **prefix,
                    "status": "failed",
                    "attempt": attempt,
                    "retryable": retryable,
                    "retry_exhausted": exhausted,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                _append_result(result_path, row)
                if not args.quiet:
                    print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
                raise
        row = {
            **prefix,
            "status": "complete",
            "attempt": attempt,
            "result": result.to_dict(),
        }
        _append_result(result_path, row)
        if not args.quiet:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        completed += 1

    summary = {
        "type": "tactile_export_summary",
        "completed": completed,
        "ineligible": skipped,
        "candidate_count": len(candidates),
        "result_path": str(result_path) if args.apply else None,
        "cold_free_bytes": shutil.disk_usage(cold_root).free,
    }
    if args.apply:
        _append_result(result_path, summary)
    if not args.quiet:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if not args.apply and skipped:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (ModalityStorageError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
