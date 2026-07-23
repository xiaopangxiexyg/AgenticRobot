from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from utils.dataset_postprocess.cli import build_parser
from utils.dataset_postprocess.umi_next_state_arm_action import (
    MIGRATION_ID,
    migrate_umi_arm_action_to_next_observation,
)
from utils.dataset_postprocess.tactile_recompress import (
    recompress_tactile_zstd_in_place,
)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
class UmiNextStateArmActionTest(unittest.TestCase):
    def _write_video(self, path: Path, frame_count: int) -> None:
        cv2 = pytest.importorskip("cv2")

        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            15.0,
            (16, 16),
        )
        self.assertTrue(writer.isOpened())
        for index in range(frame_count):
            writer.write(np.full((16, 16, 3), index * 40, dtype=np.uint8))
        writer.release()

    def _dataset(self, root: Path) -> None:
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")

        (root / "meta").mkdir(parents=True)
        parquet_relative = Path("data/chunk-000/episode_000000.parquet")
        parquet = root / parquet_relative
        parquet.parent.mkdir(parents=True)
        rows = []
        for index in range(3):
            left = [index + value / 10.0 for value in range(7)]
            right = [10 + index + value / 10.0 for value in range(7)]
            arm = left + right
            gripper = [0.2 + index / 10.0, 0.7 - index / 10.0]
            state_gripper = [0.1 + index / 10.0, 0.6 - index / 10.0]
            rows.append(
                {
                    "episode_index": 0,
                    "frame_index": index,
                    "timestamp": 100.0 + index * 0.1,
                    "index": index,
                    "action": arm + gripper,
                    "action.arm.pose": arm,
                    "action.left_arm.pose": left,
                    "action.right_arm.pose": right,
                    "observation.state": arm + state_gripper,
                    "observation.arm.pose": arm,
                    "observation.left_arm.pose": left,
                    "observation.right_arm.pose": right,
                    "observation.last_action": arm + gripper,
                    "observation.hand.left.tactile_storage_ref_json": json.dumps(
                        {
                            "frame_index": index,
                            "path": "tactile/observation.hand.left.tactile/chunk-000/episode_000000/manifest.json",
                        }
                    ),
                    "observation.hand.right.tactile_storage_ref_json": json.dumps(
                        {
                            "frame_index": index,
                            "path": "tactile/observation.hand.right.tactile/chunk-000/episode_000000/manifest.json",
                        }
                    ),
                }
            )
        pq.write_table(pa.Table.from_pylist(rows), parquet, compression="zstd")

        video_paths = {}
        tactile_paths = {}
        for side in ("left", "right"):
            video_relative = Path(
                f"videos/observation.images.{side}_wrist/chunk-000/episode_000000.mp4"
            )
            self._write_video(root / video_relative, 3)
            video_paths[f"{side}_wrist"] = video_relative.as_posix()

            manifest_relative = Path(
                f"tactile/observation.hand.{side}.tactile/chunk-000/"
                "episode_000000/manifest.json"
            )
            stream_relative = manifest_relative.parent / "sensor.uint8.bin"
            stream = root / stream_relative
            stream.parent.mkdir(parents=True)
            np.arange(3 * 2 * 2, dtype=np.uint8).tofile(stream)
            manifest = {
                "format": "binary_stream_v1",
                "frame_count": 3,
                "frames": [
                    {"sample_frame_index": index, "tactile_frame_index": index}
                    for index in range(3)
                ],
                "streams": {
                    "sensor": {
                        "path": stream_relative.as_posix(),
                        "encoding": "raw_c_order",
                        "compression": "none",
                        "dtype": "uint8",
                        "shape": [2, 2],
                        "frame_count": 3,
                    }
                },
            }
            (root / manifest_relative).write_text(json.dumps(manifest), encoding="utf-8")
            tactile_paths[side] = manifest_relative.as_posix()

        info = {
            "dexteleop_dataset_schema": "dexteleop_dataset_v2.3",
            "embodiment": "umi_dual_wrist",
            "episode_count": 1,
            "frame_count": 3,
            "postprocess_history": [],
        }
        (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
        episode = {
            "episode_index": 0,
            "episode_id": "episode_000000",
            "frame_count": 3,
            "start_timestamp": 100.0,
            "end_timestamp": 100.2,
            "duration_sec": 0.2,
            "parquet_path": parquet_relative.as_posix(),
            "video_paths": video_paths,
            "tactile_paths": tactile_paths,
            "wrist_depth_mask_paths": {},
            "wrist_depth_raw_paths": {},
            "metadata": {},
            "summary": {
                "frame_count": 3,
                "video_frame_stats": {
                    key: {
                        "expected": 3,
                        "written": 3,
                        "dropped": 0,
                        "missing": 0,
                        "path": value,
                    }
                    for key, value in video_paths.items()
                },
            },
        }
        (root / "meta/episodes.jsonl").write_text(
            json.dumps(episode) + "\n", encoding="utf-8"
        )
        stats = {
            "episode_index": 0,
            "frame_count": 3,
            "start_timestamp": 100.0,
            "end_timestamp": 100.2,
            "duration_sec": 0.2,
            "average_dt_sec": 0.1,
            "average_hz": 10.0,
        }
        (root / "meta/episodes_stats.jsonl").write_text(
            json.dumps(stats) + "\n", encoding="utf-8"
        )

    def test_cli_is_read_only_by_default(self) -> None:
        args = build_parser().parse_args(
            ["umi-arm-action-next-observation", "/tmp/dataset"]
        )
        self.assertFalse(args.apply_umi_arm_alignment)

    @pytest.mark.optional
    def test_apply_aligns_only_arm_and_synchronizes_media(self) -> None:
        pq = pytest.importorskip("pyarrow.parquet")

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "dataset"
            backup = parent / "backup"
            self._dataset(root)

            dry_run = migrate_umi_arm_action_to_next_observation(root)
            self.assertEqual(dry_run.source_frame_count, 3)
            self.assertEqual(dry_run.output_frame_count, 2)
            self.assertEqual(dry_run.tactile_frames_to_drop, 2)
            self.assertFalse(backup.exists())

            with mock.patch(
                "utils.dataset_postprocess.umi_next_state_arm_action._active_collection_processes",
                return_value=(),
            ):
                result = migrate_umi_arm_action_to_next_observation(
                    root,
                    backup_root=backup,
                    apply_umi_arm_alignment=True,
                )
            self.assertEqual(result.mode, "applied")
            self.assertTrue(backup.is_dir())

            rows = pq.read_table(
                root / "data/chunk-000/episode_000000.parquet"
            ).to_pylist()
            original = pq.read_table(
                backup / "data/chunk-000/episode_000000.parquet"
            ).to_pylist()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["action"][:14], original[1]["observation.state"][:14])
            self.assertEqual(rows[1]["action"][:14], original[2]["observation.state"][:14])
            self.assertEqual(rows[0]["action"][14:], original[0]["action"][14:])
            self.assertEqual(rows[1]["action"][14:], original[1]["action"][14:])
            self.assertEqual(
                rows[0]["observation.last_action"],
                original[0]["observation.state"][:14] + original[0]["action"][14:],
            )
            self.assertEqual(rows[1]["observation.last_action"], rows[0]["action"])
            self.assertEqual([row["index"] for row in rows], [0, 1])
            self.assertEqual(
                [row["action.arm.source_frame_index"] for row in rows], [1, 2]
            )

            info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["frame_count"], 2)
            self.assertEqual(
                info["umi_arm_action_alignment"]["migration_id"], MIGRATION_ID
            )
            episode = json.loads(
                (root / "meta/episodes.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(episode["frame_count"], 2)
            self.assertEqual(episode["summary"]["source_sample_count"], 3)

            for value in episode["video_paths"].values():
                self.assertEqual(
                    __import__(
                        "utils.dataset_postprocess.umi_next_state_arm_action",
                        fromlist=["_ffprobe_frame_count"],
                    )._ffprobe_frame_count(root / value),
                    2,
                )
            for value in episode["tactile_paths"].values():
                manifest = json.loads((root / value).read_text(encoding="utf-8"))
                self.assertEqual(manifest["frame_count"], 2)
                stream = root / manifest["streams"]["sensor"]["path"]
                self.assertEqual(stream.stat().st_size, 8)
            backup_episode = json.loads(
                (backup / "meta/episodes.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(backup_episode["frame_count"], 3)

            with self.assertRaisesRegex(RuntimeError, "already declares"):
                migrate_umi_arm_action_to_next_observation(root)

    @pytest.mark.optional
    def test_apply_trims_chunked_zstd_tactile(self) -> None:
        from dataset_contracts.tactile import ChunkedTactileStream

        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "dataset"
            self._dataset(root)
            with mock.patch(
                "utils.dataset_postprocess.tactile_recompress._active_collection_processes",
                return_value=(),
            ):
                recompress_tactile_zstd_in_place(
                    root,
                    level=1,
                    chunk_frames=2,
                    min_free_bytes=0,
                    delete_originals=True,
                    backend="auto",
                )
            with mock.patch(
                "utils.dataset_postprocess.umi_next_state_arm_action._active_collection_processes",
                return_value=(),
            ):
                migrate_umi_arm_action_to_next_observation(
                    root,
                    backup_root=parent / "backup",
                    apply_umi_arm_alignment=True,
                )
            episode = json.loads(
                (root / "meta/episodes.jsonl").read_text(encoding="utf-8")
            )
            for value in episode["tactile_paths"].values():
                manifest = json.loads((root / value).read_text(encoding="utf-8"))
                spec = manifest["streams"]["sensor"]
                self.assertEqual(spec["compression"], "zstd")
                self.assertEqual(spec["frame_count"], 2)
                self.assertEqual(sum(c["frame_count"] for c in spec["chunks"]), 2)
                reader = ChunkedTactileStream(dataset_root=root, spec=spec)
                np.testing.assert_array_equal(
                    reader.frame(1), np.arange(12, dtype=np.uint8).reshape(3, 2, 2)[1]
                )


if __name__ == "__main__":
    unittest.main()
