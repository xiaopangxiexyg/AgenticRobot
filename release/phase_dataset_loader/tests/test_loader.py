from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from dexteleop_dataset_loader import (  # noqa: E402
    DatasetFormatError,
    DexTeleopReleaseDataset,
    SelectionRequiredError,
)


def _tag(value: Any) -> dict[str, Any]:
    if value is None:
        return {"$type": "none"}
    if isinstance(value, bool):
        return {"$type": "bool", "value": value}
    if isinstance(value, int):
        return {"$type": "int", "value": str(value)}
    if isinstance(value, float):
        return {"$type": "float64", "bits": struct.pack("<d", value).hex()}
    if isinstance(value, str):
        return {"$type": "str", "value": value}
    if isinstance(value, Mapping):
        return {
            "$type": "mapping",
            "entries": [[str(key), _tag(value[key])] for key in sorted(value)],
        }
    if isinstance(value, tuple):
        return {"$type": "tuple", "items": [_tag(item) for item in value]}
    if isinstance(value, list):
        return {"$type": "list", "items": [_tag(item) for item in value]}
    raise TypeError(type(value).__name__)


def _canonical(
    path: Path,
    domain: str,
    payload: Mapping[str, Any],
    *,
    newline: bool = False,
) -> None:
    envelope = {
        "domain": domain,
        "payload": _tag(payload),
        "schema": "dexteleop_canonical_encoding_v1",
        "timestamp_unit": "seconds",
    }
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded + (b"\n" if newline else b""))


def _make_dataset(root: Path) -> Path:
    (root / "meta").mkdir(parents=True)
    info = {
        "dataset_name": "fixture",
        "dexteleop_dataset_schema": "dexteleop_dataset_v2.3",
        "episode_count": 1,
        "frame_count": 2,
        "nominal_fps": 15,
        "policy_io_layout_hash": "layout",
        "pose_frame_contract_hash": "pose",
        "causal_snapshot_contract_hash": "causal",
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/modality_validity.json").write_text(
        json.dumps({"schema_version": "dexteleop.modality_validity.v1"}),
        encoding="utf-8",
    )

    parquet_rel = "data/chunk-000/episode_000000.parquet"
    video_rel = "videos/observation.images.front/chunk-000/episode_000000.mp4"
    tactile_rel = "tactile/left/chunk-000/episode_000000/manifest.json"
    depth_rel = "wrist_depth/mask/left_wrist/chunk-000/episode_000000/manifest.json"
    episode = {
        "episode_index": 0,
        "episode_id": "episode_000000",
        "frame_count": 2,
        "parquet_path": parquet_rel,
        "video_paths": {"front": video_rel},
        "tactile_paths": {"left": tactile_rel},
        "wrist_depth_mask_paths": {"left_wrist": depth_rel},
        "metadata": {
            "progress_episode": {
                "episode_uuid": "fixture-uuid",
            }
        },
    }
    (root / "meta/episodes.jsonl").write_text(json.dumps(episode) + "\n", encoding="utf-8")

    parquet_path = root / parquet_rel
    parquet_path.parent.mkdir(parents=True)
    list16 = pa.list_(pa.float32(), 16)
    list12 = pa.list_(pa.float32(), 12)
    table = pa.table(
        {
            "episode_index": pa.array([0, 0], type=pa.int64()),
            "frame_index": pa.array([0, 1], type=pa.int64()),
            "timestamp": pa.array([1.0, 2.0], type=pa.float64()),
            "action": pa.array([list(range(16)), list(range(16, 32))], type=list16),
            "observation.state": pa.array(
                [list(range(16)), list(range(16))], type=list16
            ),
            "observation.last_action": pa.array(
                [list(range(16)), list(range(16))], type=list16
            ),
            "observation.arm.qpos": pa.array(
                [list(range(12)), list(range(12))], type=list12
            ),
            "observation.images.front.video_frame_index": pa.array([1, 0], type=pa.int64()),
        }
    )
    pq.write_table(table, parquet_path)
    video_path = root / video_rel
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fixture")

    tactile_manifest_path = root / tactile_rel
    tactile_stream_rel = "tactile/left/chunk-000/episode_000000/S0.uint8.bin"
    tactile_manifest_path.parent.mkdir(parents=True)
    (root / tactile_stream_rel).write_bytes(np.arange(4, dtype=np.uint8).tobytes())
    tactile_manifest_path.write_text(
        json.dumps(
            {
                "format": "binary_stream_v1",
                "sensor_keys": ["S0"],
                "streams": {
                    "S0": {
                        "path": tactile_stream_rel,
                        "dtype": "uint8",
                        "shape": [2, 2],
                        "frame_count": 1,
                        "encoding": "raw_c_order",
                        "compression": "none",
                    }
                },
                "frames": [{"sample_frame_index": 0, "tactile_frame_index": 0}],
            }
        ),
        encoding="utf-8",
    )

    depth_manifest_path = root / depth_rel
    depth_stream_rel = "wrist_depth/mask/left_wrist/chunk-000/episode_000000/stream.bin"
    depth_manifest_path.parent.mkdir(parents=True)
    mask = np.asarray([[1, 0, 1], [1, 0, 0]], dtype=np.uint8)
    (root / depth_stream_rel).write_bytes(np.packbits(mask.reshape(-1)).tobytes())
    depth_manifest_path.write_text(
        json.dumps(
            {
                "format": "dexteleop_wrist_depth_mask_bitpack_v1",
                "stream": {
                    "path": depth_stream_rel,
                    "shape": [2, 3],
                    "packed_bytes_per_frame": 1,
                    "frame_count": 1,
                },
                "frames": [{"sample_frame_index": 0, "depth_frame_index": 0}],
            }
        ),
        encoding="utf-8",
    )

    progress_root = root / "meta/progress"
    _canonical(
        progress_root / "profile.cjson",
        "dexteleop.progress_profile.file.v1",
        {"profile_id": "p"},
    )
    _canonical(
        progress_root / "stages.cjson",
        "dexteleop.task_stage_profile.file.v1",
        {"profile_id": "s"},
    )
    episode_root = progress_root / "episodes/fixture-uuid"
    _canonical(
        episode_root / "profile.cjson",
        "dexteleop.progress_profile.file.v1",
        {"profile_id": "p"},
    )
    _canonical(
        episode_root / "binding.cjson",
        "dexteleop.progress_binding.file.v1",
        {"binding": "b"},
    )
    _canonical(
        episode_root / "manifest.cjson",
        "dexteleop.progress_manifest.file.v1",
        {"semantic_content_hash": "pm"},
    )
    _canonical(
        episode_root / "facts.cjsonl",
        "dexteleop.progress_fact.file.v1",
        {"fact_id": "f"},
        newline=True,
    )
    _canonical(
        episode_root / "stages.cjson",
        "dexteleop.task_stage_profile.file.v1",
        {"profile_id": "s"},
    )
    _canonical(
        episode_root / "stage_entry.cjson",
        "dexteleop.progress_stage_entry_binding.file.v1",
        {"entry": "e"},
    )

    for set_id in ("set-a", "set-b"):
        annotation_root = progress_root / "stage_annotations/fixture-uuid" / set_id
        _canonical(
            annotation_root / "binding.cjson",
            "dexteleop.stage_annotation_binding.file.v1",
            {"annotation_set_id": set_id, "semantic_content_hash": f"binding-{set_id}"},
        )
        _canonical(
            annotation_root / "annotations.cjson",
            "dexteleop.stage_annotation_bundle.file.v1",
            {"occurrences": [], "semantic_content_hash": f"bundle-{set_id}"},
        )
        _canonical(
            annotation_root / "manifest.cjson",
            "dexteleop.stage_annotation_manifest.file.v1",
            {
                "annotation_set_binding_hash": f"binding-{set_id}",
                "annotation_bundle_hash": f"bundle-{set_id}",
            },
        )
    return root


def test_metadata_parquet_tactile_and_depth(tmp_path: Path) -> None:
    dataset = DexTeleopReleaseDataset(_make_dataset(tmp_path / "dataset"))
    assert dataset.summary()["episode_count"] == 1
    assert dataset.camera_keys == ("front",)
    table = dataset.load_parquet(0)
    assert table.num_rows == 2
    assert table.column("action")[1].as_py() == pytest.approx(list(range(16, 32)))

    tactile = dataset.load_tactile_frame(0, "left", 0)
    assert tactile is not None
    np.testing.assert_array_equal(tactile["S0"], np.arange(4, dtype=np.uint8).reshape(2, 2))
    assert dataset.load_tactile_frame(0, "left", 1) is None

    mask = dataset.load_wrist_depth_mask(0, "left_wrist", 0)
    assert mask is not None
    np.testing.assert_array_equal(mask, [[True, False, True], [True, False, False]])
    assert dataset.load_wrist_depth_mask(0, "left_wrist", 1) is None


def test_rgb_uses_parquet_video_frame_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = DexTeleopReleaseDataset(_make_dataset(tmp_path / "dataset"))
    calls: dict[str, Any] = {}

    class FakeCapture:
        def __init__(self, path: str):
            calls["path"] = path

        def isOpened(self) -> bool:
            return True

        def set(self, key: int, value: int) -> None:
            calls["set"] = (key, value)

        def read(self):
            return True, np.asarray([[[1, 2, 3]]], dtype=np.uint8)

        def release(self) -> None:
            calls["released"] = True

    fake_cv2 = SimpleNamespace(
        CAP_PROP_POS_FRAMES=7,
        COLOR_BGR2RGB=8,
        VideoCapture=FakeCapture,
        cvtColor=lambda image, _code: image[..., ::-1],
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    image = dataset.load_rgb_frame(0, "front", 0)
    np.testing.assert_array_equal(image, [[[3, 2, 1]]])
    assert calls["set"] == (7, 1)
    assert calls["released"] is True


def test_progress_and_annotation_require_explicit_multi_version_selection(tmp_path: Path) -> None:
    dataset = DexTeleopReleaseDataset(_make_dataset(tmp_path / "dataset"))
    assert dataset.load_progress_profile().payload["profile_id"] == "p"
    progress = dataset.load_progress_episode(0)
    assert progress["facts"][0]["fact_id"] == "f"
    assert dataset.stage_annotation_set_ids(0) == ("set-a", "set-b")
    with pytest.raises(SelectionRequiredError):
        dataset.load_stage_annotation(0)
    selected = dataset.load_stage_annotation(0, annotation_set_id="set-b")
    assert selected.annotation_set_id == "set-b"


def test_relative_path_cannot_escape_dataset(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path / "dataset")
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"not parquet")
    episode_path = root / "meta/episodes.jsonl"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["parquet_path"] = "../outside.parquet"
    episode_path.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    dataset = DexTeleopReleaseDataset(root)
    with pytest.raises(DatasetFormatError, match="escapes or is missing"):
        dataset.parquet_path(0)
