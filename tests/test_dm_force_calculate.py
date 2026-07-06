from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from dataset_postprocess.DM_force_calculate.processor import (
    FORCE_SHAPE,
    ForceProcessConfig,
    RAW_SHAPE,
    StreamProcessResult,
    _merge_stream_results_into_manifests,
    discover_raw_tactile_tasks,
    process_raw_tactile_stream,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


class FakeCalculator:
    def __init__(self, *, sensor_key: str, **_: Any) -> None:
        self.sensor_key = str(sensor_key)
        self.base: np.ndarray | None = None

    def close(self) -> None:
        return None

    def set_base_frame(self, raw: np.ndarray) -> None:
        self.base = np.asarray(raw, dtype=np.uint8).copy()

    def distributed_force(self, raw: np.ndarray) -> np.ndarray:
        value = float(np.asarray(raw, dtype=np.uint8).mean())
        out = np.zeros(FORCE_SHAPE, dtype=np.float32)
        out[..., 0] = value
        out[..., 1] = 1.0 if self.base is not None else 0.0
        out[..., 2] = float(len(self.sensor_key))
        return out


class DMForceCalculateTest(unittest.TestCase):
    def test_discover_raw_tactile_tasks_filters_non_raw_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_rel = Path(
                "tactile/observation.hand.left.tactile/chunk-000/episode_000000/manifest.json"
            )
            stream_rel = manifest_rel.parent / "M_RAW.uint8.bin"
            float_rel = manifest_rel.parent / "M_FORCE.float32.bin"
            raw = np.zeros((2, 480, 640), dtype=np.uint8)
            (root / stream_rel).parent.mkdir(parents=True, exist_ok=True)
            raw.tofile(root / stream_rel)
            np.zeros((2, 288, 384, 3), dtype=np.float32).tofile(root / float_rel)
            _write_json(
                root / manifest_rel,
                {
                    "format": "binary_stream_v1",
                    "side": "left",
                    "relative_dir": str(manifest_rel.parent),
                    "frame_count": 2,
                    "sensor_keys": ["M_RAW", "M_FORCE"],
                    "streams": {
                        "M_RAW": {
                            "path": str(stream_rel),
                            "dtype": "uint8",
                            "shape": [480, 640],
                            "frame_count": 2,
                        },
                        "M_FORCE": {
                            "path": str(float_rel),
                            "dtype": "float32",
                            "shape": [288, 384, 3],
                            "frame_count": 2,
                        },
                    },
                    "frames": [
                        {"sample_frame_index": 0, "tactile_frame_index": 0},
                        {"sample_frame_index": 1, "tactile_frame_index": 1},
                    ],
                },
            )
            tasks = discover_raw_tactile_tasks(
                ForceProcessConfig(dataset_root=root, output_root=root, progress=False)
            )
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].sensor_key, "M_RAW")

    def test_process_raw_tactile_stream_writes_float32_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "raw.bin"
            output = root / "force.bin"
            raw = np.stack(
                [
                    np.zeros((480, 640), dtype=np.uint8),
                    np.full((480, 640), 7, dtype=np.uint8),
                ],
                axis=0,
            )
            raw.tofile(source)
            task = discover_raw_tactile_tasks
            from dataset_postprocess.DM_force_calculate.processor import RawTactileStreamTask

            stream_task = RawTactileStreamTask(
                dataset_root=root,
                output_root=root,
                source_manifest_path=root / "manifest.json",
                output_manifest_path=root / "out_manifest.json",
                side="left",
                episode_index=0,
                sensor_key="M_RAW",
                source_stream_path=source,
                output_stream_path=output,
                frame_count=2,
                source_shape=(480, 640),
                source_dtype="uint8",
            )
            _ = task  # keep import lint quiet in older runners
            result = process_raw_tactile_stream(
                stream_task,
                {
                    "base_frame_index": 0,
                    "max_frames": 0,
                    "overwrite": True,
                    "backend": "cpu",
                    "mode": "standard",
                },
                calculator_factory=FakeCalculator,
            )
            self.assertEqual(result.status, "processed")
            out = np.memmap(output, dtype=np.float32, mode="r", shape=(2, *FORCE_SHAPE))
            self.assertEqual(float(out[0, 0, 0, 0]), 0.0)
            self.assertEqual(float(out[1, 0, 0, 0]), 7.0)
            self.assertEqual(float(out[1, 0, 0, 1]), 1.0)

    def test_process_raw_tactile_stream_reads_lz4_chunked_source(self) -> None:
        try:
            import lz4.frame
        except ModuleNotFoundError:
            self.skipTest("lz4 is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_rel = Path(
                "tactile/observation.hand.left.tactile/chunk-000/episode_000000/manifest.json"
            )
            stream_rel = manifest_rel.parent / "M_RAW.uint8.bin.lz4"
            raw = np.stack(
                [
                    np.zeros(RAW_SHAPE, dtype=np.uint8),
                    np.full(RAW_SHAPE, 9, dtype=np.uint8),
                ],
                axis=0,
            )
            payload = raw.tobytes(order="C")
            compressed = lz4.frame.compress(payload, compression_level=0)
            (root / stream_rel).parent.mkdir(parents=True, exist_ok=True)
            (root / stream_rel).write_bytes(compressed)
            _write_json(
                root / manifest_rel,
                {
                    "format": "binary_stream_v1",
                    "side": "left",
                    "relative_dir": str(manifest_rel.parent),
                    "frame_count": 2,
                    "sensor_keys": ["M_RAW"],
                    "streams": {
                        "M_RAW": {
                            "path": str(stream_rel),
                            "dtype": "uint8",
                            "shape": list(RAW_SHAPE),
                            "frame_count": 2,
                            "encoding": "chunked_lz4",
                            "compression": "lz4",
                            "chunks": [
                                {
                                    "frame_start": 0,
                                    "frame_count": 2,
                                    "offset": 0,
                                    "compressed_bytes": len(compressed),
                                    "raw_bytes": len(payload),
                                }
                            ],
                        },
                    },
                    "frames": [
                        {"sample_frame_index": 0, "tactile_frame_index": 0},
                        {"sample_frame_index": 1, "tactile_frame_index": 1},
                    ],
                },
            )
            tasks = discover_raw_tactile_tasks(
                ForceProcessConfig(dataset_root=root, output_root=root, progress=False)
            )
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].to_dict()["source_compression"], "lz4")

            result = process_raw_tactile_stream(
                tasks[0],
                {
                    "base_frame_index": 0,
                    "max_frames": 0,
                    "overwrite": True,
                    "backend": "cpu",
                    "mode": "standard",
                },
                calculator_factory=FakeCalculator,
            )
            self.assertEqual(result.status, "processed")
            out = np.memmap(tasks[0].output_stream_path, dtype=np.float32, mode="r", shape=(2, *FORCE_SHAPE))
            self.assertEqual(float(out[0, 0, 0, 0]), 0.0)
            self.assertEqual(float(out[1, 0, 0, 0]), 9.0)

    def test_merge_results_writes_external_tactile_compatible_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks = []
            results = []
            for sensor in ("M_A", "M_B"):
                source = root / "tactile" / f"{sensor}.uint8.bin"
                output = root / "tactile_force" / "observation.hand.left.distributed_force" / "chunk-000" / "episode_000000" / f"{sensor}.float32.bin"
                source.parent.mkdir(parents=True, exist_ok=True)
                output.parent.mkdir(parents=True, exist_ok=True)
                np.zeros((2, 480, 640), dtype=np.uint8).tofile(source)
                np.zeros((2, *FORCE_SHAPE), dtype=np.float32).tofile(output)
                from dataset_postprocess.DM_force_calculate.processor import RawTactileStreamTask

                task = RawTactileStreamTask(
                    dataset_root=root,
                    output_root=root,
                    source_manifest_path=root / "tactile" / "manifest.json",
                    output_manifest_path=output.parent / "manifest.json",
                    side="left",
                    episode_index=0,
                    sensor_key=sensor,
                    source_stream_path=source,
                    output_stream_path=output,
                    frame_count=2,
                    source_shape=(480, 640),
                    source_dtype="uint8",
                    source_frames=(
                        {
                            "sample_frame_index": 0,
                            "tactile_frame_index": 0,
                            "shapes": {sensor: [480, 640]},
                            "dtypes": {sensor: "uint8"},
                            "field_by_key": {sensor: "raw"},
                        },
                    ),
                )
                tasks.append(task)
                results.append(
                    StreamProcessResult(
                        task=task.to_dict(),
                        status="processed",
                        frame_count=2,
                        output_bytes=output.stat().st_size,
                    )
                )
            manifests = _merge_stream_results_into_manifests(
                tasks,
                results,
                config=ForceProcessConfig(dataset_root=root, output_root=root, progress=False).normalized(),
            )
            self.assertEqual(len(manifests), 1)
            payload = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["format"], "binary_stream_v1")
            self.assertEqual(payload["sensor_keys"], ["M_A", "M_B"])
            self.assertEqual(payload["streams"]["M_A"]["shape"], list(FORCE_SHAPE))
            self.assertEqual(payload["frames"][0]["field_by_key"], {"M_A": "distributed_force", "M_B": "distributed_force"})


if __name__ == "__main__":
    unittest.main()
