"""Read-only byte progress for known content-addressed tactile exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _tree_bytes(root: Path) -> tuple[int, int]:
    files = 0
    total = 0
    if root.is_dir() and not root.is_symlink():
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                files += 1
                total += path.stat().st_size
    return files, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", action="append", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for raw in args.artifact_root:
        artifact = raw.expanduser().resolve(strict=False)
        partial = artifact.parent / f".{artifact.name}.partial"
        artifact_files, artifact_bytes = _tree_bytes(artifact)
        partial_files, partial_bytes = _tree_bytes(partial)
        rows.append(
            {
                "artifact_root": str(artifact),
                "published": artifact.is_dir(),
                "published_files": artifact_files,
                "published_bytes": artifact_bytes,
                "partial_root": str(partial),
                "partial_exists": partial.is_dir(),
                "partial_files": partial_files,
                "partial_bytes": partial_bytes,
            }
        )
    print(json.dumps({"rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
