"""Read-only smoke probe for adopting an existing tactile dataset.

This probe deliberately calls only ``plan_modality_detach``.  It snapshots the
hot modality sidecar and expected cold paths before and after planning, and
fails if the read-only plan creates or changes either tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collection_runtime.modality_storage import (
    ModalityStorageError,
    plan_modality_detach,
)


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, int], ...]:
    if not root.exists():
        return ()
    records: list[tuple[str, str, int]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            records.append((relative, "symlink", 0))
        elif path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append((relative, digest, int(path.stat().st_size)))
        elif path.is_dir():
            records.append((relative + "/", "directory", 0))
        else:
            records.append((relative, "special", 0))
    return tuple(records)


def _candidates(parent: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(parent.iterdir())
        if path.is_dir()
        and not path.is_symlink()
        and (path / "meta/episodes.jsonl").is_file()
        and (path / "tactile").is_dir()
    )


def _source_links(dataset_root: Path) -> dict[str, Any]:
    files = tuple(
        path
        for path in (dataset_root / "tactile").rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    linked = tuple(path for path in files if int(path.stat().st_nlink) != 1)
    return {
        "file_count": len(files),
        "hardlinked_file_count": len(linked),
        "max_link_count": max(
            (int(path.stat().st_nlink) for path in linked),
            default=1,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-parent", required=True)
    parser.add_argument("--ordinal", type=int)
    parser.add_argument("--cold-root")
    parser.add_argument("--min-free-gb", type=float, default=1.0)
    args = parser.parse_args()

    parent = Path(args.dataset_parent).expanduser().resolve()
    candidates = _candidates(parent)
    if args.ordinal is None:
        print(
            json.dumps(
                {
                    "dataset_parent": str(parent),
                    "candidates": [
                        {"ordinal": index, "name": path.name}
                        for index, path in enumerate(candidates)
                    ],
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 0
    if args.cold_root is None:
        parser.error("--cold-root is required with --ordinal")
    try:
        dataset_root = candidates[int(args.ordinal)]
    except IndexError as exc:
        raise SystemExit(f"candidate ordinal is out of range: {args.ordinal}") from exc
    cold_root = Path(args.cold_root).expanduser().resolve()
    sidecar = dataset_root / "meta/modality_artifacts"
    sidecar_before = _tree_snapshot(sidecar)
    cold_before = _tree_snapshot(cold_root)
    plan = None
    planning_error = None
    try:
        plan = plan_modality_detach(
            dataset_root,
            cold_root,
            detachment_scope="whole-prefix",
            min_free_bytes=int(float(args.min_free_gb) * 1024**3),
        )
    except ModalityStorageError as exc:
        planning_error = f"{type(exc).__name__}: {exc}"
    if _tree_snapshot(sidecar) != sidecar_before:
        raise RuntimeError("read-only plan changed the hot modality sidecar")
    if _tree_snapshot(cold_root) != cold_before:
        raise RuntimeError("read-only plan changed the cold root")
    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "source_links": _source_links(dataset_root),
                "zero_write_verified": True,
                "plan": plan.to_dict() if plan is not None else None,
                "planning_error": planning_error,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
