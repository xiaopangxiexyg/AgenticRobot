"""Read-only full decode/hash verification for lossless tactile recompression."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from dataset_contracts.tactile import ChunkedTactileStream
from dataset_interface.validation import validate_v23_dataset
from collection_runtime.tactile_recompression import (
    LOSSLESS_PROFILE_V1,
    _finalized_manifest_paths,
    _read_json,
)


def _verify(root: Path) -> dict[str, Any]:
    verified: list[dict[str, Any]] = []
    for manifest_path in _finalized_manifest_paths(root):
        manifest = _read_json(manifest_path)
        recompression = manifest.get("tactile_recompression")
        if not isinstance(recompression, dict) or recompression.get("profile") != LOSSLESS_PROFILE_V1:
            continue
        streams = manifest.get("streams") or {}
        for sensor_key, raw_spec in sorted(streams.items()):
            spec = dict(raw_spec)
            source = spec.get("recompressed_from") or {}
            expected_hash = str(source.get("raw_sha256") or "")
            if len(expected_hash) != 64:
                raise ValueError(f"Missing source hash in {manifest_path}: {sensor_key}")
            reader = ChunkedTactileStream(dataset_root=root, spec=spec)
            digest = hashlib.sha256()
            for frame_index in range(reader.frame_count):
                frame = reader.frame(frame_index)
                if frame is None:
                    raise ValueError(
                        f"Missing decoded frame {frame_index} in {manifest_path}: {sensor_key}"
                    )
                digest.update(frame.tobytes(order="C"))
            actual_hash = digest.hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Decoded hash mismatch in {manifest_path}: {sensor_key}: "
                    f"{actual_hash} != {expected_hash}"
                )
            verified.append(
                {
                    "manifest": manifest_path.relative_to(root).as_posix(),
                    "sensor_key": str(sensor_key),
                    "frame_count": int(reader.frame_count),
                    "decoded_sha256": actual_hash,
                    "compressed_bytes": int(reader.path.stat().st_size),
                }
            )
    return {
        "dataset_root": str(root),
        "verified_stream_count": len(verified),
        "verified_frame_count": sum(row["frame_count"] for row in verified),
        "verified_compressed_bytes": sum(row["compressed_bytes"] for row in verified),
        "streams": verified,
    }


def _dataset_roots_from_batch_log(path: Path) -> list[Path]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("phase") != "plan":
            continue
        datasets = payload.get("datasets")
        if not isinstance(datasets, list):
            raise ValueError(f"Batch plan has no dataset list: {path}")
        return [Path(str(row["dataset_root"])).resolve() for row in datasets]
    raise ValueError(f"No tactile batch plan found in: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", nargs="?", type=Path)
    parser.add_argument(
        "--batch-log",
        type=Path,
        help="Verify every dataset in the first plan record of a batch JSONL log.",
    )
    parser.add_argument(
        "--v23-only",
        action="store_true",
        help="Run the native-v2.3 validator for the selected dataset roots.",
    )
    args = parser.parse_args()
    if bool(args.dataset_root) == bool(args.batch_log):
        parser.error("provide exactly one dataset_root or --batch-log")
    roots = (
        _dataset_roots_from_batch_log(args.batch_log.expanduser().resolve())
        if args.batch_log
        else [args.dataset_root.expanduser().resolve()]
    )
    if args.v23_only:
        results = [validate_v23_dataset(root).to_dict() for root in roots]
        failed = [row for row in results if not row["ok"]]
        payload = {
            "validation": "native_v23",
            "dataset_count": len(results),
            "failed_dataset_count": len(failed),
            "datasets": results,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        if failed:
            raise SystemExit(1)
        return
    results = [_verify(root) for root in roots]
    payload: dict[str, Any]
    if args.batch_log:
        payload = {
            "batch_log": str(args.batch_log.expanduser().resolve()),
            "dataset_count": len(results),
            "verified_stream_count": sum(
                row["verified_stream_count"] for row in results
            ),
            "verified_frame_count": sum(row["verified_frame_count"] for row in results),
            "verified_compressed_bytes": sum(
                row["verified_compressed_bytes"] for row in results
            ),
            "datasets": results,
        }
    else:
        payload = results[0]
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
