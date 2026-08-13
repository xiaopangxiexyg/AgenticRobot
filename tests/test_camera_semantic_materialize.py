from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from dataset_interface.camera_semantics import (
    CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH,
    DatasetCameraSemanticResolver,
    load_camera_role_binding,
)
from utils.dataset_postprocess.camera_semantic_materialize import (
    materialize_camera_semantics,
    plan_camera_semantic_materialization,
)


LEFT_SERIAL = "260522278055"
RIGHT_SERIAL = "261122273924"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_registry(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "cameras": {
                    "left_wrist": {"serial": LEFT_SERIAL},
                    "right_wrist": {"serial": RIGHT_SERIAL},
                }
            }
        ),
        encoding="utf-8",
    )


def _write_video(path: Path, *, frame_count: int, color: str) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=16x16:r=15",
            "-frames:v",
            str(frame_count),
            "-c:v",
            "mpeg4",
            "-y",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _write_source_dataset(root: Path) -> tuple[Path, Path]:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    meta = root / "meta"
    devices = meta / "devices"
    devices.mkdir(parents=True)
    (devices / "realsense.json").write_text(
        json.dumps(
            {
                "left_wrist": {
                    "camera_key": "left_wrist",
                    "serial": RIGHT_SERIAL,
                    "serial_number": RIGHT_SERIAL,
                },
                "right_wrist": {
                    "camera_key": "right_wrist",
                    "serial": LEFT_SERIAL,
                    "serial_number": LEFT_SERIAL,
                },
            }
        ),
        encoding="utf-8",
    )
    (meta / "info.json").write_text(
        json.dumps(
            {
                "episode_count": 1,
                "frame_count": 3,
                "features": {
                    "observation.images.left_wrist": {"shape": [16, 16, 3]},
                    "observation.images.right_wrist": {"shape": [16, 16, 3]},
                },
            }
        ),
        encoding="utf-8",
    )
    left_video = Path(
        "videos/chunk-000/observation.images.left_wrist/episode_000000.mp4"
    )
    right_video = Path(
        "videos/chunk-000/observation.images.right_wrist/episode_000000.mp4"
    )
    _write_video(root / left_video, frame_count=3, color="red")
    _write_video(root / right_video, frame_count=2, color="blue")

    left_mask = Path(
        "wrist_depth/mask/observation.images.left_wrist/"
        "episode_000000/manifest.json"
    )
    right_mask = Path(
        "wrist_depth/mask/observation.images.right_wrist/"
        "episode_000000/manifest.json"
    )
    for role, manifest, count in (
        ("left_wrist", left_mask, 3),
        ("right_wrist", right_mask, 2),
    ):
        path = root / manifest
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "frames.bin").write_bytes(bytes(range(count)))
        path.write_text(
            json.dumps(
                {
                    "camera_key": role,
                    "stream": {
                        "path": "frames.bin",
                        "frame_count": count,
                    },
                }
            ),
            encoding="utf-8",
        )

    parquet = Path("data/chunk-000/episode_000000.parquet")
    (root / parquet).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "frame_index": [0, 1, 2],
                "observation.images.left_wrist.video_frame_index": [0, 1, 2],
                "observation.images.right_wrist.video_frame_index": [0, None, 1],
                "observation.images.left_wrist.source_frame_index": [10, 11, 12],
                "observation.images.right_wrist.source_frame_index": [20, None, 21],
                "observation.images.left_wrist.path": [left_video.as_posix()] * 3,
                "observation.images.right_wrist.path": [right_video.as_posix()] * 3,
                "observation.wrist_camera.left.pose_base": [[1.0] * 7] * 3,
                "observation.wrist_camera.right.pose_base": [[2.0] * 7] * 3,
            }
        ),
        root / parquet,
    )
    episode = {
        "episode_index": 0,
        "frame_count": 3,
        "parquet_path": parquet.as_posix(),
        "video_paths": {
            "left_wrist": left_video.as_posix(),
            "right_wrist": right_video.as_posix(),
        },
        "wrist_depth_mask_paths": {
            "left_wrist": left_mask.as_posix(),
            "right_wrist": right_mask.as_posix(),
        },
        "summary": {
            "video_frame_stats": {
                "left_wrist": {"written": 3, "path": left_video.as_posix()},
                "right_wrist": {"written": 2, "path": right_video.as_posix()},
            }
        },
    }
    (meta / "episodes.jsonl").write_text(
        json.dumps(episode) + "\n", encoding="utf-8"
    )
    return root / left_video, root / right_video


@pytest.mark.optional
def test_materialization_swaps_complete_camera_families_and_preserves_gaps(
    tmp_path: Path,
) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source"
    source.mkdir()
    source_left_video, source_right_video = _write_source_dataset(source)
    source_video_hashes = (_sha256(source_left_video), _sha256(source_right_video))
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    output = tmp_path / "output"

    plan = plan_camera_semantic_materialization(source, output, registry)
    assert plan.status_counts == {"swapped": 1}
    assert plan.remapped_mismatched_row_count == 2
    result = materialize_camera_semantics(
        source,
        output,
        registry,
        expected_plan_hash=plan.plan_hash,
    )

    assert result["validated_video_count"] == 2
    assert result["validated_decoded_video_frame_count"] == 5
    table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    assert table.column(
        "observation.images.left_wrist.video_frame_index"
    ).to_pylist() == [0, None, 1]
    assert table.column(
        "observation.images.right_wrist.video_frame_index"
    ).to_pylist() == [0, 1, 2]
    assert table.column(
        "observation.images.left_wrist.source_frame_index"
    ).to_pylist() == [20, None, 21]
    # Robot/arm semantic fields are deliberately outside the camera namespace.
    assert table.column(
        "observation.wrist_camera.left.pose_base"
    ).to_pylist() == [[1.0] * 7] * 3

    episode = json.loads(
        (output / "meta/episodes.jsonl").read_text(encoding="utf-8")
    )
    output_left = output / episode["video_paths"]["left_wrist"]
    output_right = output / episode["video_paths"]["right_wrist"]
    assert _sha256(output_left) == source_video_hashes[1]
    assert _sha256(output_right) == source_video_hashes[0]
    assert output_left.stat().st_ino != source_right_video.stat().st_ino
    assert output_right.stat().st_ino != source_left_video.stat().st_ino

    left_manifest = json.loads(
        (output / episode["wrist_depth_mask_paths"]["left_wrist"]).read_text(
            encoding="utf-8"
        )
    )
    right_manifest = json.loads(
        (output / episode["wrist_depth_mask_paths"]["right_wrist"]).read_text(
            encoding="utf-8"
        )
    )
    assert left_manifest["camera_key"] == "left_wrist"
    assert left_manifest["stream"]["frame_count"] == 2
    assert right_manifest["camera_key"] == "right_wrist"
    assert right_manifest["stream"]["frame_count"] == 3

    binding = load_camera_role_binding(registry)
    resolver = DatasetCameraSemanticResolver(output, binding)
    assert resolver.resolution_for_episode(0).status == "identity"
    assert resolver.require_frame_index_alignment()["remapped_episode_count"] == 0
    assert (output / CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH).is_file()
    assert (_sha256(source_left_video), _sha256(source_right_video)) == source_video_hashes


@pytest.mark.optional
def test_apply_requires_frozen_plan_and_never_overwrites(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    source = tmp_path / "source"
    source.mkdir()
    _write_source_dataset(source)
    registry = tmp_path / "registry.json"
    _write_registry(registry)
    output = tmp_path / "output"

    with pytest.raises(Exception, match="expected-plan-hash"):
        materialize_camera_semantics(
            source,
            output,
            registry,
            expected_plan_hash="wrong",
        )

    output.mkdir()
    plan = plan_camera_semantic_materialization(source, output, registry)
    with pytest.raises(Exception, match="already exists"):
        materialize_camera_semantics(
            source,
            output,
            registry,
            expected_plan_hash=plan.plan_hash,
        )
