#!/usr/bin/env python3
"""Export or verify a portable refinement-annotation source projection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from check_annotation_projection import (
    AnnotationProjectionError,
    DEFAULT_MANIFEST,
    validate_annotation_projection,
)


RECEIPT_NAME = "annotation_projection_receipt.json"
RELEASE_README_NAME = "ANNOTATION_RELEASE.md"
RECEIPT_SCHEMA = "dexteleop.annotation-source-projection-receipt.v1"


class AnnotationProjectionExportError(RuntimeError):
    """Raised when a projection cannot be exported or verified safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, *, relative: Path, kind: str) -> dict[str, object]:
    return {
        "path": relative.as_posix(),
        "kind": kind,
        "byte_count": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _projection_hash(
    *,
    entrypoint: str,
    files: Sequence[Mapping[str, object]],
) -> str:
    payload = {
        "schema": RECEIPT_SCHEMA,
        "entrypoint": entrypoint,
        "files": list(files),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_source_state(
    repository_root: Path,
    included_paths: Sequence[Path],
) -> tuple[str | None, bool]:
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = (
        commit_result.stdout.strip()
        if commit_result.returncode == 0
        else None
    )
    relative_paths = [
        str(path.relative_to(repository_root)) for path in included_paths
    ]
    status_result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *relative_paths,
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = status_result.returncode != 0 or bool(status_result.stdout.strip())
    return commit, dirty


def _release_readme(projection_hash: str) -> str:
    return f"""# DexTeleop refinement annotation projection

This directory is generated from the canonical DexTeleop source projection.
It preserves the original package paths and is not an independent package or
data-protocol fork. Projection id: `{projection_hash}`.

It contains no dataset, annotation result, media cache, robot runtime, hardware
driver, simulator, deployment code, or training code.

## Verify

From this directory, run:

```bash
python scripts/export_annotation_projection.py --verify .
python scripts/check_annotation_projection.py
```

## Install and start

The host needs Linux, `uv`, Python 3.10, `ffmpeg`, `ffprobe`, and a modern
browser. Then run:

```bash
bash scripts/install_annotation_env.sh
source .venv-annotation/bin/activate
python -m control_station refinement-annotation
```

The GUI can select or change datasets. Dataset paths and retrospective source
roots are supplied at runtime and must not be copied into this code release.
See `README_ENV.md` and `docs/real/LONG_HORIZON_STAGE_ANNOTATION.md` for details.
"""


def export_projection(
    *,
    repository_root: Path,
    manifest_path: Path,
    output: Path,
) -> dict[str, object]:
    root = repository_root.expanduser().resolve()
    target = output.expanduser().resolve()
    if target == root or target in root.parents:
        raise AnnotationProjectionExportError(
            "Output must not be the repository root or one of its parents."
        )
    if target.exists():
        raise AnnotationProjectionExportError(
            f"Output already exists; choose a new release directory: {target}"
        )

    try:
        report = validate_annotation_projection(root, manifest_path)
    except AnnotationProjectionError as exc:
        raise AnnotationProjectionExportError(str(exc)) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.tmp-",
            dir=target.parent,
        )
    )
    try:
        records: list[dict[str, object]] = []
        for source in report.included_paths:
            relative = source.relative_to(root)
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            records.append(
                _record(destination, relative=relative, kind="source")
            )
        records.sort(key=lambda value: str(value["path"]))
        entrypoint = "python -m control_station refinement-annotation"
        source_projection_hash = _projection_hash(
            entrypoint=entrypoint,
            files=records,
        )
        release_readme = stage / RELEASE_README_NAME
        release_readme.write_text(
            _release_readme(source_projection_hash),
            encoding="utf-8",
        )
        records.append(
            _record(
                release_readme,
                relative=Path(RELEASE_README_NAME),
                kind="generated",
            )
        )
        records.sort(key=lambda value: str(value["path"]))
        release_hash = _projection_hash(
            entrypoint=entrypoint,
            files=records,
        )
        commit, dirty = _git_source_state(root, report.included_paths)
        receipt: dict[str, object] = {
            "schema": RECEIPT_SCHEMA,
            "entrypoint": entrypoint,
            "manifest": str(report.manifest_path.relative_to(root)),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_git_commit": commit,
            "source_projection_dirty": dirty,
            "source_projection_hash": source_projection_hash,
            "release_hash": release_hash,
            "source_file_count": len(report.included_paths),
            "file_count": len(records),
            "files": records,
        }
        (stage / RECEIPT_NAME).write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        verify_projection(stage)
        os.replace(stage, target)
        return receipt
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _receipt(root: Path) -> dict[str, Any]:
    path = root / RECEIPT_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnnotationProjectionExportError(
            f"Cannot read projection receipt {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema") != RECEIPT_SCHEMA:
        raise AnnotationProjectionExportError(
            f"Unsupported projection receipt in {path}."
        )
    return payload


def verify_projection(projection_root: Path) -> dict[str, object]:
    root = projection_root.expanduser().resolve()
    receipt = _receipt(root)
    raw_files = receipt.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise AnnotationProjectionExportError("Receipt files must be non-empty.")

    records: list[dict[str, object]] = []
    source_paths: set[str] = set()
    for raw_record in raw_files:
        if not isinstance(raw_record, dict):
            raise AnnotationProjectionExportError(
                "Receipt file records must be objects."
            )
        relative = Path(str(raw_record.get("path", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise AnnotationProjectionExportError(
                f"Unsafe receipt path: {relative}"
            )
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise AnnotationProjectionExportError(
                f"Projection file is missing or escaped the root: {relative}"
            )
        kind = str(raw_record.get("kind", ""))
        observed = _record(path, relative=relative, kind=kind)
        if observed != raw_record:
            raise AnnotationProjectionExportError(
                f"Projection file differs from its receipt: {relative}"
            )
        records.append(observed)
        if kind == "source":
            source_paths.add(relative.as_posix())

    records.sort(key=lambda value: str(value["path"]))
    entrypoint = str(receipt.get("entrypoint", ""))
    release_hash = _projection_hash(entrypoint=entrypoint, files=records)
    if release_hash != receipt.get("release_hash"):
        raise AnnotationProjectionExportError("Release hash mismatch.")
    source_records = [value for value in records if value["kind"] == "source"]
    source_hash = _projection_hash(
        entrypoint=entrypoint,
        files=source_records,
    )
    if source_hash != receipt.get("source_projection_hash"):
        raise AnnotationProjectionExportError("Source projection hash mismatch.")
    if len(source_records) != receipt.get("source_file_count"):
        raise AnnotationProjectionExportError("Source file count mismatch.")
    if len(records) != receipt.get("file_count"):
        raise AnnotationProjectionExportError("Release file count mismatch.")

    manifest = Path(str(receipt.get("manifest", DEFAULT_MANIFEST)))
    try:
        report = validate_annotation_projection(root, manifest)
    except AnnotationProjectionError as exc:
        raise AnnotationProjectionExportError(str(exc)) from exc
    projected_paths = {
        path.relative_to(root).as_posix() for path in report.included_paths
    }
    if projected_paths != source_paths:
        raise AnnotationProjectionExportError(
            "Receipt source files differ from the projection manifest."
        )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export or verify the portable DexTeleop refinement-annotation "
            "source projection."
        )
    )
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--output", type=Path)
    operation.add_argument("--verify", type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify is not None:
            receipt = verify_projection(args.verify)
            action = "verified"
            root = args.verify.expanduser().resolve()
        else:
            receipt = export_projection(
                repository_root=args.repository_root,
                manifest_path=args.manifest,
                output=args.output,
            )
            action = "exported"
            root = args.output.expanduser().resolve()
    except AnnotationProjectionExportError as exc:
        raise SystemExit(f"annotation projection export failed: {exc}") from exc
    print(
        f"annotation projection {action} root={root} "
        f"source_files={receipt['source_file_count']} "
        f"release_hash={str(receipt['release_hash'])[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
