from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dataset_contracts.tactile import ChunkedTactileStream
from utils.dataset_postprocess.tactile_compact import (
    FORCE_PROFILE_V1,
    RAW_PROFILE_V1,
    compact_tactile_dataset_in_place,
    plan_tactile_compaction,
)


HAS_ZSTD = shutil.which("zstd") is not None
try:
    import zstandard  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    HAS_ZSTD = True


def _write_manifest(
    root: Path,
    *,
    prefix: str,
    sensor: str,
    frames: np.ndarray,
    field: str,
) -> tuple[Path, Path]:
    directory = root / prefix / "observation.hand.left.tactile/chunk-000/episode_000000"
    directory.mkdir(parents=True, exist_ok=True)
    dtype = str(frames.dtype)
    stream_path = directory / f"{sensor}.{dtype}.bin"
    frames.tofile(stream_path)
    relative_stream = stream_path.relative_to(root).as_posix()
    manifest_path = directory / "manifest.json"
    payload = {
        "format": "binary_stream_v1",
        "side": "left",
        "frame_count": int(frames.shape[0]),
        "sensor_keys": [sensor],
        "frames": [
            {
                "tactile_frame_index": index,
                "sample_frame_index": index,
                "shapes": {sensor: list(frames.shape[1:])},
                "dtypes": {sensor: dtype},
                "field_by_key": {sensor: field},
            }
            for index in range(frames.shape[0])
        ],
        "streams": {
            sensor: {
                "path": relative_stream,
                "encoding": "raw_c_order",
                "compression": "none",
                "dtype": dtype,
                "shape": list(frames.shape[1:]),
                "frame_count": int(frames.shape[0]),
            }
        },
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path, stream_path


@unittest.skipUnless(HAS_ZSTD, "zstandard Python package or zstd CLI is required")
class TactileCompactTest(unittest.TestCase):
    def test_raw_profile_preserves_frequency_and_decodes_120x160_uint8(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = np.arange(5 * 480 * 640, dtype=np.uint8).reshape(5, 480, 640)
            manifest_path, stream_path = _write_manifest(
                root,
                prefix="tactile",
                sensor="raw_sensor",
                frames=source,
                field="raw",
            )

            plan = plan_tactile_compaction(root, min_free_bytes=0)
            self.assertTrue(plan.ready)
            self.assertEqual(plan.raw_stream_count, 1)
            with mock.patch(
                "utils.dataset_postprocess.tactile_compact._active_collection_processes",
                return_value=(),
            ):
                result = compact_tactile_dataset_in_place(
                    root,
                    chunk_frames=2,
                    min_free_bytes=0,
                    delete_originals=True,
                )

            self.assertTrue(result.complete)
            self.assertFalse(stream_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            spec = manifest["streams"]["raw_sensor"]
            self.assertEqual(spec["tactile_compaction"]["profile"], RAW_PROFILE_V1)
            self.assertEqual(spec["shape"], [120, 160])
            self.assertEqual(spec["frame_count"], 5)
            self.assertEqual(spec["encoding"], "chunked_delta_zstd")
            self.assertTrue(spec["tactile_compaction"]["frequency_preserved"])
            self.assertEqual(manifest["frames"][0]["shapes"]["raw_sensor"], [120, 160])
            self.assertEqual(manifest["frames"][0]["source_shapes"]["raw_sensor"], [480, 640])

            expected_sums = source.astype(np.uint32).reshape(5, 120, 4, 160, 4).sum((2, 4))
            expected_mean = ((expected_sums + 8) // 16).astype(np.uint8)
            expected = (((expected_mean >> 2).astype(np.uint16) << 2) + 2).astype(np.uint8)
            reader = ChunkedTactileStream(dataset_root=root, spec=spec)
            self.assertEqual(reader.frame_count, 5)
            for index in range(5):
                np.testing.assert_array_equal(reader.frame(index), expected[index])

    def test_force_profile_returns_logical_float32_with_small_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(7)
            source = rng.normal(0.0, 2.0e-6, size=(3, 288, 384, 3)).astype(np.float32)
            manifest_path, stream_path = _write_manifest(
                root,
                prefix="tactile_force",
                sensor="force_sensor",
                frames=source,
                field="distributed_force",
            )
            with mock.patch(
                "utils.dataset_postprocess.tactile_compact._active_collection_processes",
                return_value=(),
            ):
                result = compact_tactile_dataset_in_place(
                    root,
                    chunk_frames=2,
                    min_free_bytes=0,
                    delete_originals=True,
                )

            self.assertTrue(result.complete)
            self.assertFalse(stream_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            spec = manifest["streams"]["force_sensor"]
            self.assertEqual(spec["tactile_compaction"]["profile"], FORCE_PROFILE_V1)
            self.assertEqual(spec["dtype"], "float32")
            self.assertEqual(spec["storage_dtype"], "int16")
            reader = ChunkedTactileStream(dataset_root=root, spec=spec)
            restored = np.stack([reader.frame(index) for index in range(3)])
            self.assertEqual(restored.dtype, np.float32)
            nrmse = float(np.sqrt(np.sum((source - restored) ** 2) / np.sum(source**2)))
            self.assertLess(nrmse, 1.0e-4)

    def test_unknown_shape_blocks_uniform_apply(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_manifest(
                root,
                prefix="tactile",
                sensor="unknown",
                frames=np.zeros((2, 10, 10), dtype=np.uint8),
                field="raw",
            )
            plan = plan_tactile_compaction(root, min_free_bytes=0)
            self.assertFalse(plan.ready)
            with mock.patch(
                "utils.dataset_postprocess.tactile_compact._active_collection_processes",
                return_value=(),
            ):
                with self.assertRaisesRegex(ValueError, "Unsupported tactile streams"):
                    compact_tactile_dataset_in_place(
                        root,
                        min_free_bytes=0,
                        delete_originals=True,
                    )

    def test_reader_validation_failure_rolls_back_manifest_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = np.arange(2 * 480 * 640, dtype=np.uint8).reshape(2, 480, 640)
            manifest_path, stream_path = _write_manifest(
                root,
                prefix="tactile",
                sensor="raw_sensor",
                frames=source,
                field="raw",
            )
            original_manifest = manifest_path.read_bytes()
            with (
                mock.patch(
                    "utils.dataset_postprocess.tactile_compact._active_collection_processes",
                    return_value=(),
                ),
                mock.patch(
                    "utils.dataset_postprocess.tactile_compact._validate_committed_manifest",
                    side_effect=ValueError("injected reader failure"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "injected reader failure"):
                    compact_tactile_dataset_in_place(
                        root,
                        min_free_bytes=0,
                        delete_originals=True,
                    )

            self.assertTrue(stream_path.is_file())
            np.testing.assert_array_equal(
                np.fromfile(stream_path, dtype=np.uint8).reshape(source.shape), source
            )
            self.assertEqual(json.loads(manifest_path.read_bytes()), json.loads(original_manifest))
            self.assertFalse(list(stream_path.parent.glob("*.raw120x160-q6-delta.zst")))
            self.assertFalse(list(stream_path.parent.glob(".*.tactile-compact-rollback")))


if __name__ == "__main__":
    unittest.main()
