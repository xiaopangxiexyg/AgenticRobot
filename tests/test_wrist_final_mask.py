from __future__ import annotations

import json

import numpy as np
import pytest

from dataset_interface.random_background.mask_io import BitpackMaskReader
from utils.dataset_postprocess.wrist_cam_adjust.depth_mask.core import (
    WRIST_DEPTH_MASK_FORMAT,
    pack_binary_mask,
)
from utils.dataset_postprocess.wrist_cam_adjust.final_mask import (
    build_final_mask_tasks,
    compose_final_masks,
)
from utils.dataset_postprocess.wrist_cam_adjust.geometry_mask.dataset import (
    _base_from_stored_pose_by_side,
)


def test_desk_absolute_pose_contract_is_mapped_back_to_each_arm_base() -> None:
    left_raw_to_desk = np.asarray(
        [
            [0.0, -1.0, 0.0, -0.472],
            [1.0, 0.0, 0.0, 0.2625],
            [0.0, 0.0, 1.0, -0.01],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    right_raw_to_desk = np.asarray(
        [
            [0.0, 1.0, 0.0, -0.472],
            [-1.0, 0.0, 0.0, -0.2625],
            [0.0, 0.0, 1.0, -0.01],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transforms, storage_frame = _base_from_stored_pose_by_side(
        {
            "pose_frame_contract": {
                "pose_storage_frame": "desk_absolute",
                "source": {
                    "raw_to_canonical_desk_transform_by_side": {
                        "left": left_raw_to_desk.tolist(),
                        "right": right_raw_to_desk.tolist(),
                    }
                },
            }
        }
    )

    assert storage_frame == "desk_absolute"
    np.testing.assert_allclose(transforms["left"] @ left_raw_to_desk, np.eye(4))
    np.testing.assert_allclose(transforms["right"] @ right_raw_to_desk, np.eye(4))


def test_native_arm_base_pose_contract_does_not_add_a_transform() -> None:
    transforms, storage_frame = _base_from_stored_pose_by_side(
        {
            "pose_frame_contract": {
                "pose_storage_frame": "native_arm_base",
            }
        }
    )

    assert storage_frame == "native_arm_base"
    assert transforms == {}


def test_final_mask_tasks_accept_one_explicit_dataset_namespace(tmp_path) -> None:
    session = tmp_path / "transport-view"
    (session / "meta").mkdir(parents=True)
    (session / "meta/episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 7,
                "video_paths": {
                    "right_wrist": (
                        "videos/observation.images.right_wrist/"
                        "chunk-000/episode_000007.mp4"
                    )
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    depth_root = tmp_path / "postprocessed"
    final_root = tmp_path / "final"

    tasks = build_final_mask_tasks(
        session_root=session,
        table_mask_root=tmp_path / "table",
        depth_mask_root=depth_root,
        output_root=final_root,
        cameras=("right_wrist",),
        dataset_namespace="canonical-dataset",
    )

    assert len(tasks) == 1
    assert (
        tasks[0].depth_manifest_path
        == (
            depth_root
            / "canonical-dataset"
            / "episode_000007"
            / "right_wrist"
            / "manifest.json"
        ).resolve()
    )
    assert (
        tasks[0].output_dir
        == (
            final_root / "canonical-dataset" / "episode_000007" / "right_wrist"
        ).resolve()
    )
    with pytest.raises(ValueError, match="safe path component"):
        build_final_mask_tasks(
            session_root=session,
            table_mask_root=tmp_path / "table",
            depth_mask_root=depth_root,
            output_root=final_root,
            dataset_namespace="../escape",
        )


@pytest.mark.optional
def test_compose_final_mask_ors_postprocessed_depth_and_table_without_proxy(
    tmp_path,
) -> None:
    cv2 = pytest.importorskip("cv2")
    session = tmp_path / "session"
    depth_root = tmp_path / "depth"
    table_root = tmp_path / "table"
    output_root = tmp_path / "final"
    rgb_rel = "videos/observation.images.left_wrist/chunk-000/episode_000000.mp4"
    (session / "meta").mkdir(parents=True)
    (session / "meta/episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "frame_count": 2,
                "video_paths": {"left_wrist": rgb_rel},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    depth_masks = [
        np.asarray(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        np.asarray(
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.uint8,
        ),
    ]
    depth_dir = depth_root / session.name / "episode_000000" / "left_wrist"
    depth_dir.mkdir(parents=True)
    (depth_dir / "mask_postprocessed.bitpack.bin").write_bytes(
        b"".join(pack_binary_mask(mask) for mask in depth_masks)
    )
    (depth_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": WRIST_DEPTH_MASK_FORMAT,
                "camera_key": "left_wrist",
                "relative_dir": "episode_000000/left_wrist",
                "stream": {
                    "path": "mask_postprocessed.bitpack.bin",
                    "encoding": "np.packbits",
                    "dtype": "uint1",
                    "shape": [4, 4],
                    "packed_bytes_per_frame": 2,
                    "frame_count": 2,
                },
                "frames": [{"frame_index": 0}, {"frame_index": 1}],
            }
        ),
        encoding="utf-8",
    )

    table_masks = [
        np.asarray(
            [
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        np.asarray(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    ]
    table_video = table_root / rgb_rel
    table_video.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(
        str(table_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        15.0,
        (4, 4),
    )
    assert writer.isOpened()
    try:
        for mask in table_masks:
            frame = np.where(mask > 0, 255, 0).astype(np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()

    summary = compose_final_masks(
        session_root=session,
        table_mask_root=table_root,
        depth_mask_root=depth_root,
        output_root=output_root,
        episodes=[0],
        cameras=["left_wrist"],
        jobs=1,
        overwrite=True,
    )

    manifest = (
        output_root / session.name / "episode_000000" / "left_wrist" / "manifest.json"
    )
    assert summary["processed_count"] == 1
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["semantics"]["proxy_mask"] is False
    assert payload["semantics"]["body_mask"] is False
    with BitpackMaskReader(manifest) as reader:
        np.testing.assert_array_equal(reader.read(0), depth_masks[0] | table_masks[0])
        np.testing.assert_array_equal(reader.read(1), depth_masks[1] | table_masks[1])
