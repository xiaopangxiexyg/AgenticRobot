from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from utils.dataset_postprocess.wrist_cam_adjust.depth import (
    DepthRangeMaskConfig,
    mask_manifest_payload,
    pack_binary_mask,
)
from utils.dataset_postprocess.wrist_cam_adjust.visualize_depth_mask import (
    compose_preview_frame,
    find_session_mask_manifests,
    iter_mask_frames,
    load_mask_stream,
    write_mask_preview_video,
)


def _write_manifest_stream(root: Path, camera_key: str, masks: list[np.ndarray]) -> Path:
    relative_dir = (
        Path("wrist_depth")
        / "mask"
        / f"observation.images.{camera_key}"
        / "chunk-000"
        / "episode_000000"
    )
    stream_rel = relative_dir / "stream.bitpack.bin"
    manifest_rel = relative_dir / "manifest.json"
    stream_path = root / stream_rel
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    stream_path.write_bytes(b"".join(pack_binary_mask(mask) for mask in masks))
    records = [
        {
            "mask_frame_index": index,
            "sample_frame_index": index,
            "foreground_fraction": float(np.count_nonzero(mask)) / float(mask.size),
        }
        for index, mask in enumerate(masks)
    ]
    height, width = masks[0].shape
    payload = mask_manifest_payload(
        camera_key=camera_key,
        relative_dir=str(relative_dir),
        stream_path=str(stream_rel),
        frame_count=len(masks),
        height=int(height),
        width=int(width),
        records=records,
        config=DepthRangeMaskConfig(),
    )
    manifest_path = root / manifest_rel
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_rel


class WristDepthMaskVisualizationTest(unittest.TestCase):
    def test_session_resolves_two_wrist_mask_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            left_rel = _write_manifest_stream(
                root,
                "left_wrist",
                [np.asarray([[1, 0], [0, 1]], dtype=np.uint8)],
            )
            right_rel = _write_manifest_stream(
                root,
                "right_wrist",
                [np.asarray([[0, 1], [1, 0]], dtype=np.uint8)],
            )
            episodes = root / "meta" / "episodes.jsonl"
            episodes.parent.mkdir(parents=True, exist_ok=True)
            episodes.write_text(
                json.dumps(
                    {
                        "episode_index": 0,
                        "wrist_depth_mask_paths": {
                            "left_wrist": str(left_rel),
                            "right_wrist": str(right_rel),
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            manifests = find_session_mask_manifests(root)
            self.assertEqual([path.name for path in manifests], ["manifest.json", "manifest.json"])
            streams = [load_mask_stream(path, root_hint=root) for path in manifests]
            self.assertEqual([stream.label for stream in streams], ["left_wrist", "right_wrist"])
            np.testing.assert_array_equal(
                next(iter_mask_frames(streams[0])),
                np.asarray([[1, 0], [0, 1]], dtype=np.uint8),
            )
            direct_manifest_stream = load_mask_stream(root / left_rel)
            self.assertEqual(direct_manifest_stream.stream_path, root / left_rel.parent / "stream.bitpack.bin")

    def test_direct_bitpack_stream_uses_explicit_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stream_path = Path(tmpdir) / "mask.bitpack.bin"
            mask = np.asarray([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
            stream_path.write_bytes(pack_binary_mask(mask))
            stream = load_mask_stream(stream_path, shape="2x3", label="direct")
            self.assertEqual(stream.frame_count, 1)
            self.assertEqual(stream.label, "direct")
            np.testing.assert_array_equal(next(iter_mask_frames(stream)), mask)

    @pytest.mark.optional
    def test_preview_frame_and_video_are_written(self) -> None:
        try:
            import cv2  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("cv2 is not available")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            left_rel = _write_manifest_stream(
                root,
                "left_wrist",
                [
                    np.asarray([[1, 0], [0, 1]], dtype=np.uint8),
                    np.asarray([[1, 1], [0, 0]], dtype=np.uint8),
                ],
            )
            right_rel = _write_manifest_stream(
                root,
                "right_wrist",
                [
                    np.asarray([[0, 1], [1, 0]], dtype=np.uint8),
                    np.asarray([[0, 0], [1, 1]], dtype=np.uint8),
                ],
            )
            streams = [
                load_mask_stream(root / left_rel, root_hint=root),
                load_mask_stream(root / right_rel, root_hint=root),
            ]
            first_masks = [next(iter_mask_frames(stream)) for stream in streams]
            frame = compose_preview_frame(first_masks, streams=streams, frame_index=0)
            self.assertEqual(frame.shape, (36, 4, 3))

            result = write_mask_preview_video(
                streams,
                root / "preview.mp4",
                fps=15.0,
                overwrite=True,
            )
            self.assertEqual(result.frame_count, 2)
            self.assertTrue(result.output_path.exists())
            self.assertGreater(result.output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
