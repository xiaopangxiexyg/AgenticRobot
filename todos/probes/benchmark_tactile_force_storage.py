#!/usr/bin/env python3
"""Read-only source / temporary-output benchmark for DMTac force storage.

The probe never edits the selected dataset.  When --solve-benchmark is used it
derives a bounded number of frames into a TemporaryDirectory, benchmarks codec
candidates in memory, emits JSON, and removes the temporary output on exit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import zstandard as zstd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.dataset_postprocess.DM_force_calculate.processor import (
    FORCE_SHAPE,
    ForceProcessConfig,
    RawTactileStreamTask,
    discover_raw_tactile_tasks,
    process_raw_tactile_stream,
)


def _dataset_candidates(scan_root: Path, pattern: str) -> list[Path]:
    return sorted(
        (path.resolve() for path in scan_root.glob(pattern) if path.is_dir()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )


def _task_summary(task: RawTactileStreamTask) -> dict[str, Any]:
    spec = dict(task.source_stream_spec)
    return {
        "episode_index": int(task.episode_index),
        "side": task.side,
        "sensor_key": task.sensor_key,
        "frame_count": int(task.frame_count),
        "source_bytes": int(task.source_stream_path.stat().st_size),
        "source_dtype": task.source_dtype,
        "source_shape": list(task.source_shape),
        "source_encoding": str(spec.get("encoding") or "raw_c_order"),
        "source_compression": str(spec.get("compression") or "none"),
        "source_manifest": str(task.source_manifest_path),
    }


def _byte_shuffle(array: np.ndarray) -> bytes:
    values = np.ascontiguousarray(array)
    itemsize = int(values.dtype.itemsize)
    return values.view(np.uint8).reshape(-1, itemsize).T.copy().tobytes(order="C")


def _xor_temporal_float32(array: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    flat = values.reshape(values.shape[0], -1)
    encoded = np.empty_like(flat)
    encoded[0] = flat[0]
    if len(flat) > 1:
        encoded[1:] = np.bitwise_xor(flat[1:], flat[:-1])
    return encoded


def _delta_temporal_int16(array: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(array, dtype=np.int16).view(np.uint16)
    flat = values.reshape(values.shape[0], -1)
    encoded = np.empty_like(flat)
    encoded[0] = flat[0]
    if len(flat) > 1:
        encoded[1:] = (
            flat[1:].astype(np.uint32) - flat[:-1].astype(np.uint32)
        ).astype(np.uint16)
    return encoded


def _quantize(array: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bits not in {8, 16}:
        raise ValueError(bits)
    values = np.asarray(array, dtype=np.float32)
    limit = float((1 << (bits - 1)) - 1)
    maxima = np.max(np.abs(values), axis=(0, 1, 2)).astype(np.float64)
    scales = np.where(maxima > 0.0, maxima / limit, 1.0).astype(np.float32)
    dtype = np.int16 if bits == 16 else np.int8
    quantized = np.rint(values / scales.reshape(1, 1, 1, -1)).clip(-limit, limit).astype(dtype)
    restored = quantized.astype(np.float32) * scales.reshape(1, 1, 1, -1)
    return quantized, restored, scales


def _pool2(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    count, height, width, channels = values.shape
    if height % 2 or width % 2:
        raise ValueError(f"Force shape is not divisible by two: {values.shape}")
    return values.reshape(count, height // 2, 2, width // 2, 2, channels).mean(axis=(2, 4))


def _errors(reference: np.ndarray, restored: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    out = np.asarray(restored, dtype=np.float64)
    error = out - ref
    denominator = float(np.sum(ref * ref))
    nrmse = math.sqrt(float(np.sum(error * error)) / denominator) if denominator > 0.0 else 0.0
    per_channel_rmse = np.sqrt(np.mean(error * error, axis=(0, 1, 2)))
    per_channel_max = np.max(np.abs(error), axis=(0, 1, 2))
    ref_integral = np.sum(ref, axis=(1, 2))
    out_integral = np.sum(out, axis=(1, 2))
    integral_error = out_integral - ref_integral
    return {
        "max_abs_error": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "nrmse": float(nrmse),
        "per_channel_rmse": per_channel_rmse.tolist(),
        "per_channel_max_abs_error": per_channel_max.tolist(),
        "integrated_force_max_abs_error": np.max(np.abs(integral_error), axis=0).tolist(),
    }


def _compress_chunks(payloads: list[bytes], level: int) -> tuple[int, float]:
    compressor = zstd.ZstdCompressor(level=int(level))
    started = time.perf_counter()
    size = sum(len(compressor.compress(payload)) for payload in payloads)
    return int(size), float(time.perf_counter() - started)


def _chunks(array: np.ndarray, chunk_frames: int) -> list[np.ndarray]:
    return [
        np.ascontiguousarray(array[start : start + chunk_frames])
        for start in range(0, len(array), int(chunk_frames))
    ]


def _benchmark_force(force: np.ndarray, raw: np.ndarray, *, chunk_frames: int, level: int) -> dict[str, Any]:
    force = np.ascontiguousarray(force, dtype=np.float32)
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    raw_uncompressed = int(raw.nbytes)
    raw_zstd, raw_zstd_sec = _compress_chunks(
        [chunk.tobytes(order="C") for chunk in _chunks(raw, chunk_frames)], level
    )
    candidates: dict[str, dict[str, Any]] = {}

    def add(name: str, payloads: list[bytes], *, logical: np.ndarray, error: dict[str, Any] | None) -> None:
        stored, elapsed = _compress_chunks(payloads, level)
        candidates[name] = {
            "stored_bytes": stored,
            "ratio_vs_raw_uncompressed": stored / raw_uncompressed,
            "ratio_vs_raw_zstd": stored / raw_zstd,
            "compress_sec": elapsed,
            "logical_shape": list(logical.shape[1:]),
            "logical_dtype": str(logical.dtype),
            "error": error,
        }

    force_chunks = _chunks(force, chunk_frames)
    add(
        "float32_zstd",
        [chunk.tobytes(order="C") for chunk in force_chunks],
        logical=force,
        error=_errors(force, force),
    )
    add(
        "float32_byteshuffle_zstd",
        [_byte_shuffle(chunk) for chunk in force_chunks],
        logical=force,
        error=_errors(force, force),
    )
    add(
        "float32_xor_byteshuffle_zstd",
        [_byte_shuffle(_xor_temporal_float32(chunk)) for chunk in force_chunks],
        logical=force,
        error=_errors(force, force),
    )

    for bits in (16, 8):
        payloads: list[bytes] = []
        restored_chunks: list[np.ndarray] = []
        scale_bytes = 0
        for chunk in force_chunks:
            quantized, restored, scales = _quantize(chunk, bits)
            restored_chunks.append(restored)
            scale_bytes += int(scales.nbytes)
            if bits == 16:
                payloads.append(_byte_shuffle(_delta_temporal_int16(quantized)))
            else:
                payloads.append(quantized.tobytes(order="C"))
        restored_force = np.concatenate(restored_chunks, axis=0)
        stored, elapsed = _compress_chunks(payloads, level)
        stored += scale_bytes
        candidates[f"qint{bits}_temporal_zstd"] = {
            "stored_bytes": stored,
            "ratio_vs_raw_uncompressed": stored / raw_uncompressed,
            "ratio_vs_raw_zstd": stored / raw_zstd,
            "compress_sec": elapsed,
            "logical_shape": list(force.shape[1:]),
            "logical_dtype": "float32",
            "error": _errors(force, restored_force),
        }

    pooled = _pool2(force)
    pooled_payloads: list[bytes] = []
    pooled_restored: list[np.ndarray] = []
    pooled_scale_bytes = 0
    for chunk in _chunks(pooled, chunk_frames):
        quantized, restored, scales = _quantize(chunk, 16)
        pooled_restored.append(restored)
        pooled_scale_bytes += int(scales.nbytes)
        pooled_payloads.append(_byte_shuffle(_delta_temporal_int16(quantized)))
    pooled_stored, pooled_sec = _compress_chunks(pooled_payloads, level)
    pooled_stored += pooled_scale_bytes
    pooled_logical = np.concatenate(pooled_restored, axis=0)
    upsampled = np.repeat(np.repeat(pooled_logical, 2, axis=1), 2, axis=2)
    candidates["pool2_qint16_temporal_zstd"] = {
        "stored_bytes": pooled_stored,
        "ratio_vs_raw_uncompressed": pooled_stored / raw_uncompressed,
        "ratio_vs_raw_zstd": pooled_stored / raw_zstd,
        "compress_sec": pooled_sec,
        "logical_shape": list(pooled_logical.shape[1:]),
        "logical_dtype": "float32",
        "error_after_nearest_upsample": _errors(force, upsampled),
    }
    return {
        "frame_count": int(len(force)),
        "raw_uncompressed_bytes": raw_uncompressed,
        "raw_zstd_bytes": raw_zstd,
        "raw_zstd_ratio": raw_zstd / raw_uncompressed,
        "raw_zstd_compress_sec": raw_zstd_sec,
        "force_value_stats": {
            "min_per_channel": np.min(force, axis=(0, 1, 2)).tolist(),
            "max_per_channel": np.max(force, axis=(0, 1, 2)).tolist(),
            "mean_abs_per_channel": np.mean(np.abs(force), axis=(0, 1, 2)).tolist(),
            "exact_zero_fraction": float(np.mean(force == 0.0)),
        },
        "candidates": candidates,
    }


def _raw_frames(task: RawTactileStreamTask, count: int) -> np.ndarray:
    spec = dict(task.source_stream_spec)
    compression = str(spec.get("compression") or "none").lower()
    if compression not in {"", "none", "raw", "uncompressed"}:
        raise ValueError(f"Probe currently expects uncompressed source raw, got {compression!r}")
    values = np.memmap(
        task.source_stream_path,
        mode="r",
        dtype=np.uint8,
        shape=(int(task.frame_count), *task.source_shape),
    )
    return np.asarray(values[:count]).copy()


def _raw_predictor_encode(array: np.ndarray, predictor: str) -> np.ndarray:
    values = np.ascontiguousarray(array, dtype=np.uint8)
    encoded = np.empty_like(values)
    if predictor == "identity":
        encoded[...] = values
    elif predictor == "temporal_delta":
        encoded[0] = values[0]
        encoded[1:] = (
            values[1:].astype(np.uint16) - values[:-1].astype(np.uint16)
        ).astype(np.uint8)
    elif predictor == "temporal_xor":
        encoded[0] = values[0]
        encoded[1:] = np.bitwise_xor(values[1:], values[:-1])
    elif predictor == "horizontal_delta":
        encoded[:, :, 0] = values[:, :, 0]
        encoded[:, :, 1:] = (
            values[:, :, 1:].astype(np.uint16) - values[:, :, :-1].astype(np.uint16)
        ).astype(np.uint8)
    elif predictor == "temporal_horizontal_delta":
        temporal = _raw_predictor_encode(values, "temporal_delta")
        encoded[:, :, 0] = temporal[:, :, 0]
        encoded[:, :, 1:] = (
            temporal[:, :, 1:].astype(np.uint16)
            - temporal[:, :, :-1].astype(np.uint16)
        ).astype(np.uint8)
    else:
        raise ValueError(f"Unknown raw predictor: {predictor}")
    return encoded


def _raw_predictor_decode(array: np.ndarray, predictor: str) -> np.ndarray:
    encoded = np.ascontiguousarray(array, dtype=np.uint8)

    def modular_cumsum(*, axis: int) -> np.ndarray:
        axis_length = int(encoded.shape[axis])
        accumulator_dtype = np.uint16 if axis_length * 255 <= np.iinfo(np.uint16).max else np.uint32
        return np.cumsum(encoded, axis=axis, dtype=accumulator_dtype).astype(np.uint8)

    if predictor == "identity":
        return encoded.copy()
    if predictor == "temporal_delta":
        return modular_cumsum(axis=0)
    if predictor == "temporal_xor":
        decoded = np.empty_like(encoded)
        decoded[0] = encoded[0]
        for index in range(1, len(encoded)):
            decoded[index] = np.bitwise_xor(encoded[index], decoded[index - 1])
        return decoded
    if predictor == "horizontal_delta":
        return modular_cumsum(axis=2)
    if predictor == "temporal_horizontal_delta":
        temporal = modular_cumsum(axis=2)
        return _raw_predictor_decode(temporal, "temporal_delta")
    raise ValueError(f"Unknown raw predictor: {predictor}")


def _benchmark_raw_lossless(
    raw: np.ndarray,
    *,
    chunk_frame_counts: tuple[int, ...],
    levels: tuple[int, ...],
) -> dict[str, Any]:
    values = np.ascontiguousarray(raw, dtype=np.uint8)
    source_bytes = int(values.nbytes)
    predictors = (
        "identity",
        "temporal_delta",
        "temporal_xor",
        "horizontal_delta",
        "temporal_horizontal_delta",
    )
    candidates: list[dict[str, Any]] = []
    for chunk_frames in chunk_frame_counts:
        raw_chunks = _chunks(values, int(chunk_frames))
        for predictor in predictors:
            encode_started = time.perf_counter()
            encoded_chunks = [
                _raw_predictor_encode(chunk, predictor) for chunk in raw_chunks
            ]
            encode_sec = time.perf_counter() - encode_started
            for level in levels:
                compressor = zstd.ZstdCompressor(level=int(level))
                decompressor = zstd.ZstdDecompressor()
                compress_started = time.perf_counter()
                compressed_chunks = [
                    compressor.compress(chunk.tobytes(order="C"))
                    for chunk in encoded_chunks
                ]
                compress_sec = time.perf_counter() - compress_started
                decode_started = time.perf_counter()
                exact = True
                for source_chunk, encoded_chunk, compressed in zip(
                    raw_chunks, encoded_chunks, compressed_chunks, strict=True
                ):
                    restored_encoded = np.frombuffer(
                        decompressor.decompress(
                            compressed,
                            max_output_size=int(encoded_chunk.nbytes),
                        ),
                        dtype=np.uint8,
                    ).reshape(encoded_chunk.shape)
                    restored = _raw_predictor_decode(restored_encoded, predictor)
                    if not np.array_equal(restored, source_chunk):
                        exact = False
                        break
                decode_sec = time.perf_counter() - decode_started
                stored_bytes = sum(len(item) for item in compressed_chunks)
                candidates.append(
                    {
                        "predictor": predictor,
                        "chunk_frames": int(chunk_frames),
                        "zstd_level": int(level),
                        "stored_bytes": int(stored_bytes),
                        "ratio": stored_bytes / source_bytes,
                        "exact_roundtrip": bool(exact),
                        "encode_sec": float(encode_sec),
                        "compress_sec": float(compress_sec),
                        "decode_sec": float(decode_sec),
                    }
                )
    ordered = sorted(
        candidates,
        key=lambda item: (
            not bool(item["exact_roundtrip"]),
            float(item["ratio"]),
            float(item["decode_sec"]),
        ),
    )
    return {
        "frame_count": int(len(values)),
        "source_bytes": source_bytes,
        "source_shape": list(values.shape[1:]),
        "source_dtype": str(values.dtype),
        "best_by_size": ordered[:10],
        "all_candidates": candidates,
    }


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positive integers")
    return parsed


def _aggregate_raw_benchmarks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        for item in result["all_candidates"]:
            key = (
                str(item["predictor"]),
                int(item["chunk_frames"]),
                int(item["zstd_level"]),
            )
            grouped[key].append(item)
    aggregated: list[dict[str, Any]] = []
    for (predictor, chunk_frames, level), items in grouped.items():
        ratios = np.asarray([float(item["ratio"]) for item in items], dtype=np.float64)
        aggregated.append(
            {
                "predictor": predictor,
                "chunk_frames": chunk_frames,
                "zstd_level": level,
                "stream_count": len(items),
                "exact_roundtrip_all": all(bool(item["exact_roundtrip"]) for item in items),
                "ratio_mean": float(np.mean(ratios)),
                "ratio_min": float(np.min(ratios)),
                "ratio_max": float(np.max(ratios)),
                "compress_sec_mean": float(
                    np.mean([float(item["compress_sec"]) for item in items])
                ),
                "decode_sec_mean": float(
                    np.mean([float(item["decode_sec"]) for item in items])
                ),
            }
        )
    return sorted(
        aggregated,
        key=lambda item: (
            not bool(item["exact_roundtrip_all"]),
            float(item["ratio_mean"]),
            float(item["decode_sec_mean"]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", type=Path, default=Path("datasets"))
    parser.add_argument("--dataset-pattern", default="2026-08-02*dmtac*writing_5_stage")
    parser.add_argument(
        "--find-force-datasets",
        action="store_true",
        help="Only list matching datasets that already contain tactile_force manifests.",
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Omit the full candidate matrix after computing aggregate results.",
    )
    parser.add_argument(
        "--raw-benchmark",
        action="store_true",
        help="Benchmark exact reversible raw predictors without constructing the SDK sensor.",
    )
    parser.add_argument("--solve-benchmark", action="store_true")
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--max-streams", type=int, default=4)
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument(
        "--raw-chunk-frames",
        type=_parse_positive_ints,
        default=(1, 8, 32),
    )
    parser.add_argument(
        "--raw-zstd-levels",
        type=_parse_positive_ints,
        default=(1, 3, 7),
    )
    args = parser.parse_args()

    candidates = _dataset_candidates(args.scan_root.expanduser().resolve(), args.dataset_pattern)
    if not candidates:
        raise SystemExit(f"No dataset matched {args.dataset_pattern!r} under {args.scan_root}")
    if args.find_force_datasets:
        found = []
        for candidate in candidates:
            manifests = sorted((candidate / "tactile_force").glob("**/manifest.json"))
            if manifests:
                found.append(
                    {
                        "dataset_root": str(candidate),
                        "force_manifest_count": len(manifests),
                        "force_bytes": sum(
                            path.stat().st_size
                            for path in (candidate / "tactile_force").glob("**/*")
                            if path.is_file()
                        ),
                    }
                )
        print(json.dumps({"force_datasets": found}, indent=2, ensure_ascii=False))
        return 0
    dataset_root = candidates[0]
    inventory_config = ForceProcessConfig(
        dataset_root=dataset_root,
        output_root=Path(tempfile.gettempdir()) / "dexteleop_tactile_probe_inventory",
        jobs=1,
        dry_run=True,
        progress=False,
    )
    tasks = discover_raw_tactile_tasks(inventory_config)
    grouped: dict[int, list[RawTactileStreamTask]] = defaultdict(list)
    for task in tasks:
        grouped[int(task.episode_index)].append(task)
    payload: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "raw_stream_count": len(tasks),
        "episodes": {
            str(index): [_task_summary(task) for task in sorted(values, key=lambda item: (item.side, item.sensor_key))]
            for index, values in sorted(grouped.items())
        },
        "benchmarks": [],
    }
    if args.summary_only:
        payload["episodes"] = {
            "count": len(grouped),
            "indices": sorted(grouped),
            "total_raw_bytes": sum(task.source_stream_path.stat().st_size for task in tasks),
            "min_frames_per_stream": min((task.frame_count for task in tasks), default=0),
            "max_frames_per_stream": max((task.frame_count for task in tasks), default=0),
        }
    if not args.solve_benchmark and not args.raw_benchmark:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if not grouped:
        raise SystemExit("Selected dataset has no discoverable raw tactile streams")

    selected_episode = max(
        grouped,
        key=lambda index: (len(grouped[index]), min(task.frame_count for task in grouped[index]), index),
    )
    selected = sorted(grouped[selected_episode], key=lambda task: (task.side, task.sensor_key))[
        : max(1, int(args.max_streams))
    ]
    if args.raw_benchmark:
        raw_results: list[dict[str, Any]] = []
        for task in selected:
            count = min(int(task.frame_count), max(1, int(args.max_frames)))
            benchmark = _benchmark_raw_lossless(
                _raw_frames(task, count),
                chunk_frame_counts=tuple(args.raw_chunk_frames),
                levels=tuple(args.raw_zstd_levels),
            )
            raw_results.append(benchmark)
            output_benchmark = dict(benchmark)
            if args.compact_output:
                output_benchmark.pop("all_candidates", None)
                output_benchmark["best_by_size"] = output_benchmark["best_by_size"][:5]
            payload["benchmarks"].append(
                {
                    "episode_index": selected_episode,
                    "side": task.side,
                    "sensor_key": task.sensor_key,
                    "raw_lossless": output_benchmark,
                }
            )
        aggregate = _aggregate_raw_benchmarks(raw_results)
        payload["raw_lossless_aggregate"] = (
            aggregate[:15] if args.compact_output else aggregate
        )
        if not args.solve_benchmark:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0
    with tempfile.TemporaryDirectory(prefix="dexteleop_tactile_force_benchmark_") as temporary:
        temp_root = Path(temporary)
        solve_config = ForceProcessConfig(
            dataset_root=dataset_root,
            output_root=temp_root,
            episodes=(selected_episode,),
            jobs=1,
            calibration_policy="required",
            compression="none",
            max_frames=max(1, int(args.max_frames)),
            progress=False,
        ).normalized()
        worker_config = solve_config.to_worker_dict()
        for task in selected:
            remapped = RawTactileStreamTask(
                dataset_root=task.dataset_root,
                output_root=temp_root,
                source_manifest_path=task.source_manifest_path,
                output_manifest_path=temp_root / task.output_manifest_path.relative_to(inventory_config.output_root),
                side=task.side,
                episode_index=task.episode_index,
                sensor_key=task.sensor_key,
                source_stream_path=task.source_stream_path,
                output_stream_path=temp_root / task.output_stream_path.relative_to(inventory_config.output_root),
                frame_count=task.frame_count,
                source_shape=task.source_shape,
                source_dtype=task.source_dtype,
                source_stream_spec=task.source_stream_spec,
                source_frames=task.source_frames,
            )
            result = process_raw_tactile_stream(remapped, worker_config)
            item: dict[str, Any] = {
                "episode_index": selected_episode,
                "side": task.side,
                "sensor_key": task.sensor_key,
                "solve_status": result.status,
                "solve_elapsed_sec": result.elapsed_sec,
                "solve_error": result.error,
                "calibration": result.calibration,
            }
            if result.status == "processed":
                count = int(result.frame_count)
                force = np.memmap(
                    remapped.output_stream_path,
                    mode="r",
                    dtype=np.float32,
                    shape=(count, *FORCE_SHAPE),
                )
                item["storage_benchmark"] = _benchmark_force(
                    np.asarray(force).copy(),
                    _raw_frames(task, count),
                    chunk_frames=max(1, int(args.chunk_frames)),
                    level=int(args.zstd_level),
                )
            payload["benchmarks"].append(item)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
