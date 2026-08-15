from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dataset_interface import (
    MODALITY_VALIDITY_SCHEMA_VERSION,
    DatasetInterfaceConfig,
    DexTeleopDatasetIndex,
    FrameRequest,
    InvalidTrainingDataError,
    MaskConfig,
    MaskedFramePipeline,
    MissingViewConfig,
    VideoConfig,
)
from dataset_interface.adapters.torch import TorchMaskedAugmentDataset
from dataset_interface.mask_source import FinalMaskSource
from utils.dataset_postprocess.wrist_cam_adjust.depth_mask.core import WRIST_DEPTH_MASK_FORMAT, pack_binary_mask
from dataset_interface.random_background import (
    FixedBorderMaskConfig,
    PhotometricAugmentConfig,
    RandomBackgroundAugmentConfig,
    ViewOcclusionConfig,
)


_DISABLED_PHOTOMETRIC = PhotometricAugmentConfig(enabled=False)


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    cv2 = pytest.importorskip("cv2")
    height, width = frames[0].shape[:2]
    path.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (width, height))
    assert writer.isOpened()
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _write_final_mask(root: Path, dataset_name: str, masks: list[np.ndarray]) -> Path:
    mask_dir = root / dataset_name / "episode_000000" / "left_wrist"
    mask_dir.mkdir(parents=True)
    stream = mask_dir / "final_keep.bitpack.bin"
    stream.write_bytes(b"".join(pack_binary_mask(mask) for mask in masks))
    payload = {
        "format": WRIST_DEPTH_MASK_FORMAT,
        "camera_key": "left_wrist",
        "relative_dir": str(mask_dir),
        "mask_family": "wrist_final_keep",
        "stream": {
            "path": stream.name,
            "encoding": "np.packbits",
            "dtype": "uint1",
            "shape": [int(masks[0].shape[0]), int(masks[0].shape[1])],
            "packed_bytes_per_frame": int((masks[0].size + 7) // 8),
            "frame_count": len(masks),
        },
        "semantics": {
            "value_1": "foreground/keep pixel",
            "value_0": "replaceable background pixel",
        },
    }
    manifest = mask_dir / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def _write_episode_metadata(
    dataset: Path,
    records: list[dict[str, object]],
    *,
    modality_validity: dict[str, object] | None = None,
) -> None:
    meta = dataset / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    if modality_validity is not None:
        (meta / "modality_validity.json").write_text(
            json.dumps(modality_validity, indent=2) + "\n",
            encoding="utf-8",
        )


def test_final_mask_source_does_not_fall_back_to_wrist_depth_mask(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    depth_manifest = dataset / "wrist_depth/mask/left_wrist/manifest.json"
    depth_manifest.parent.mkdir(parents=True)
    depth_manifest.write_text("{}", encoding="utf-8")
    _write_episode_metadata(
        dataset,
        [
            {
                "episode_index": 0,
                "frame_count": 1,
                "wrist_depth_mask_paths": {
                    "left_wrist": str(depth_manifest.relative_to(dataset)),
                },
            }
        ],
    )
    source = FinalMaskSource(
        dataset_index=DexTeleopDatasetIndex(dataset),
        final_mask_root=tmp_path / "missing_final_masks",
    )

    with pytest.raises(FileNotFoundError, match="not an implicit training fallback"):
        source.manifest_path(0, "left_wrist")


@pytest.mark.optional
def test_masked_frame_pipeline_reads_video_mask_and_applies_black_background(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    video_rel = "videos/observation.images.left_wrist/chunk-000/episode_000000.mp4"
    frames = [
        np.full((8, 10, 3), (20, 60, 100), dtype=np.uint8),
        np.full((8, 10, 3), (80, 120, 160), dtype=np.uint8),
    ]
    frames[1][2:6, 3:7] = (10, 200, 30)
    _write_video(dataset / video_rel, frames)
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "frame_count": 2,
                "video_paths": {"left_wrist": video_rel},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    masks = [
        np.zeros((8, 10), dtype=np.uint8),
        np.zeros((8, 10), dtype=np.uint8),
    ]
    masks[1][2:6, 3:7] = 1
    final_root = tmp_path / "final_masks"
    _write_final_mask(final_root, dataset.name, masks)

    config = DatasetInterfaceConfig(
        dataset_root=dataset,
        mask=MaskConfig(final_mask_root=final_root),
        video=VideoConfig(output_layout="bgr_hwc_uint8"),
        augmentation=RandomBackgroundAugmentConfig(
            provider="black",
            edge_blur_px=0,
            enable_epoch_variation=False,
            photometric=_DISABLED_PHOTOMETRIC,
        ),
    )
    with MaskedFramePipeline(config) as pipeline:
        sample = pipeline.read_augmented_frame(
            episode_index=0,
            camera_key="left_wrist",
            frame_index=1,
        )
        raw = pipeline.videos.read(dataset / video_rel, 1)
        clip = pipeline.read_augmented_clip(
            episode_index=0,
            camera_key="left_wrist",
            start_frame=0,
            length=2,
        )

    assert sample.shape == (8, 10, 3)
    np.testing.assert_array_equal(sample[3:5, 4:6], raw[3:5, 4:6])
    assert np.max(sample[0:2, 0:2]) == 0
    assert clip.shape == (2, 8, 10, 3)
    assert np.max(clip[0]) == 0


@pytest.mark.optional
def test_masked_frame_pipeline_can_return_rgb_chw_float(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    video_rel = "videos/observation.images.left_wrist/chunk-000/episode_000000.mp4"
    _write_video(dataset / video_rel, [np.full((4, 6, 3), 100, dtype=np.uint8)])
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "frame_count": 1, "video_paths": {"left_wrist": video_rel}}) + "\n",
        encoding="utf-8",
    )
    final_root = tmp_path / "final_masks"
    _write_final_mask(final_root, dataset.name, [np.ones((4, 6), dtype=np.uint8)])
    config = DatasetInterfaceConfig(
        dataset_root=dataset,
        mask=MaskConfig(final_mask_root=final_root),
        video=VideoConfig(output_layout="rgb_chw_float32"),
        augmentation=RandomBackgroundAugmentConfig(
            provider="black",
            edge_blur_px=0,
            photometric=_DISABLED_PHOTOMETRIC,
        ),
    )

    with MaskedFramePipeline(config) as pipeline:
        frame = pipeline.read_augmented_frame(episode_index=0, camera_key="left_wrist", frame_index=0)

    assert frame.shape == (3, 4, 6)
    assert frame.dtype == np.float32
    assert 0.0 <= float(frame.min()) <= float(frame.max()) <= 1.0


@pytest.mark.optional
def test_masked_frame_pipeline_records_photometric_plan_as_audit_metadata_only(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    video_rel = "videos/observation.images.left_wrist/chunk-000/episode_000000.mp4"
    _write_video(dataset / video_rel, [np.full((8, 12, 3), (60, 100, 140), dtype=np.uint8)])
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "frame_count": 1, "video_paths": {"left_wrist": video_rel}}) + "\n",
        encoding="utf-8",
    )
    final_root = tmp_path / "final_masks"
    _write_final_mask(final_root, dataset.name, [np.ones((8, 12), dtype=np.uint8)])
    warm_gains = (0.85, 1.00, 1.15)
    photometric = PhotometricAugmentConfig(
        exposure_probability=0.0,
        contrast_probability=0.0,
        gamma_probability=0.0,
        white_balance_probability=1.0,
        warm_bgr_gains=warm_gains,
        cool_bgr_gains=warm_gains,
        saturation_probability=0.0,
        hue_probability=0.0,
        illumination_probability=0.0,
        quality_probability=0.0,
    )
    config = DatasetInterfaceConfig(
        dataset_root=dataset,
        mask=MaskConfig(final_mask_root=final_root),
        augmentation=RandomBackgroundAugmentConfig(provider="original", photometric=photometric),
    )

    with MaskedFramePipeline(config) as pipeline:
        sample = pipeline.read_augmented(FrameRequest(episode_index=0, camera_key="left_wrist", frame_index=0))

    expected = np.clip(
        sample.image.astype(np.float32) * np.asarray(warm_gains, dtype=np.float32).reshape(1, 1, 3),
        0.0,
        255.0,
    ).astype(np.uint8)
    np.testing.assert_array_equal(sample.augmented, expected)
    audit = sample.metadata["photometric_augmentation"]
    assert audit["type"] == "full_image_photometric_v1"
    assert audit["white_balance_bgr"] == [0.85, 1.0, 1.15]
    background = sample.metadata["background_augmentation"]
    assert background["type"] == "masked_background_augmentation_v2"
    assert background["background_family"] == "wrist_pair_scene"
    assert background["provider"] == "original"
    assert background["scene_canvas_shared"] is False
    assert background["scene_canvas_source"] == "not_generated"
    assert "observation.images.left_wrist.augmented" not in sample.metadata


@pytest.mark.optional
def test_masked_frame_pipeline_applies_per_view_occlusion_before_layout_conversion(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    video_rel = "videos/observation.images.left_wrist/chunk-000/episode_000000.mp4"
    _write_video(dataset / video_rel, [np.full((18, 24, 3), (60, 100, 140), dtype=np.uint8)])
    _write_episode_metadata(
        dataset,
        [{"episode_index": 0, "frame_count": 1, "video_paths": {"left_wrist": video_rel}}],
    )
    final_root = tmp_path / "final_masks"
    _write_final_mask(final_root, dataset.name, [np.ones((18, 24), dtype=np.uint8)])
    config = DatasetInterfaceConfig(
        dataset_root=dataset,
        mask=MaskConfig(final_mask_root=final_root),
        video=VideoConfig(output_layout="rgb_chw_float32"),
        augmentation=RandomBackgroundAugmentConfig(
            provider="original",
            photometric=_DISABLED_PHOTOMETRIC,
            view_occlusion=ViewOcclusionConfig(enabled=True),
        ),
    )

    with MaskedFramePipeline(config) as pipeline:
        sample = pipeline.read_augmented(
            FrameRequest(
                episode_index=0,
                camera_key="left_wrist",
                frame_index=0,
                occlusion_ratio=0.50,
            )
        )

    assert sample.augmented.shape == (3, 18, 24)
    black_pixels = np.all(sample.augmented == 0.0, axis=0)
    assert np.count_nonzero(black_pixels) == round(0.50 * 18 * 24)
    audit = sample.metadata["view_occlusion"]
    assert audit["actual_pixels"] == round(0.50 * 18 * 24)
    assert audit["requested_ratio"] == 0.50


@pytest.mark.optional
def test_front_fixed_border_mask_does_not_require_wrist_final_mask(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    video_rel = "videos/observation.images.front/chunk-000/episode_000000.mp4"
    frames = [np.full((20, 30, 3), (20, 60, 100), dtype=np.uint8) for _ in range(2)]
    _write_video(dataset / video_rel, frames)
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "frame_count": 2,
                "video_paths": {"front": video_rel},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = DatasetInterfaceConfig(
        dataset_root=dataset,
        mask=MaskConfig(final_mask_root=tmp_path / "not_used_for_front"),
        augmentation=RandomBackgroundAugmentConfig(
            global_seed=37,
            provider="procedural",
            edge_blur_px=0,
            fixed_mask=FixedBorderMaskConfig(
                enabled=True,
                side_probability=1.0,
                thickness_fraction=0.05,
            ),
            photometric=_DISABLED_PHOTOMETRIC,
        ),
    )

    with MaskedFramePipeline(config) as pipeline:
        first = pipeline.read_augmented(FrameRequest(episode_index=0, camera_key="front", frame_index=0, epoch=1))
        second = pipeline.read_augmented(FrameRequest(episode_index=0, camera_key="front", frame_index=1, epoch=1))
        clip = pipeline.read_augmented_clip(
            episode_index=0,
            camera_key="front",
            start_frame=0,
            length=2,
            epoch=1,
        )

    assert first.mask_manifest_path is None
    assert first.metadata["mask_type"] == "front_fixed_border_replace"
    assert first.metadata["mask_manifest_path"] is None
    assert first.metadata["fixed_mask"]["selected_sides"] == ["top", "bottom", "left", "right"]
    np.testing.assert_array_equal(first.augmented[4:16, 4:26], first.image[4:16, 4:26])
    np.testing.assert_array_equal(first.augmented, second.augmented)
    np.testing.assert_array_equal(first.augmented, clip[0])
    np.testing.assert_array_equal(second.augmented, clip[1])


def test_training_validity_defaults_to_all_episodes_when_metadata_is_absent(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_episode_metadata(
        dataset,
        [
            {"episode_index": 0, "video_paths": {"front": "videos/front_0.mp4"}},
            {"episode_index": 1, "video_paths": {"left_wrist": "videos/left_1.mp4"}},
        ],
    )

    index = DexTeleopDatasetIndex(dataset)

    assert index.training_camera_keys == ("front", "left_wrist")
    assert index.training_eligibility(0, camera_keys=("front",)).eligible
    assert index.training_eligibility(1, camera_keys=("left_wrist",)).eligible


def test_training_validity_rejects_invalid_episode_or_requested_view(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_episode_metadata(
        dataset,
        [
            {
                "episode_index": 0,
                "video_paths": {"front": "videos/front_0.mp4", "left_wrist": "videos/left_0.mp4"},
            },
            {
                "episode_index": 1,
                "video_paths": {"front": "videos/front_1.mp4", "left_wrist": "videos/left_1.mp4"},
            },
        ],
        modality_validity={
            "schema_version": MODALITY_VALIDITY_SCHEMA_VERSION,
            "training": {
                "episodes": {
                    "0": {"valid": False, "reason": "tracker reset during recording"},
                    "1": {
                        "modalities": {
                            "video": {
                                "left_wrist": {"valid": False, "reason": "camera stream lost"}
                            }
                        }
                    },
                }
            },
        },
    )

    index = DexTeleopDatasetIndex(dataset)
    invalid_data = index.training_eligibility(0, camera_keys=("front",))
    invalid_view = index.training_eligibility(1, camera_keys=("front", "left_wrist"))

    assert not invalid_data.eligible
    assert "tracker reset" in invalid_data.message()
    assert not invalid_view.eligible
    assert "left_wrist" in invalid_view.message()
    assert index.training_eligibility(1, camera_keys=("front",)).eligible
    with pytest.raises(InvalidTrainingDataError, match="camera stream lost"):
        index.training_video_path(1, "left_wrist")


def test_training_validity_accepts_remote_legacy_global_allow_list(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_episode_metadata(
        dataset,
        [{"episode_index": 0, "video_paths": {"front": "videos/front_0.mp4", "left_wrist": "videos/left_0.mp4"}}],
        modality_validity={"training": {"allowed_video_camera_keys": ["front"]}},
    )

    index = DexTeleopDatasetIndex(dataset)

    assert index.training_camera_keys == ("front",)
    assert index.training_eligibility(0, camera_keys=("front",)).eligible
    assert not index.training_eligibility(0, camera_keys=("left_wrist",)).eligible


def test_pipeline_warns_and_blocks_invalid_training_data_before_video_read(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "dataset"
    _write_episode_metadata(
        dataset,
        [{"episode_index": 0, "video_paths": {"front": "videos/never_read.mp4"}}],
        modality_validity={
            "schema_version": MODALITY_VALIDITY_SCHEMA_VERSION,
            "training": {
                "episodes": {
                    "0": {"modalities": {"video": {"front": {"valid": False, "reason": "all frames stale"}}}}
                }
            },
        },
    )
    config = DatasetInterfaceConfig(
        dataset_root=dataset,
        mask=MaskConfig(final_mask_root=tmp_path / "unused_masks"),
    )

    with MaskedFramePipeline(config) as pipeline:
        with pytest.raises(InvalidTrainingDataError, match="all frames stale"):
            pipeline.read_augmented(FrameRequest(episode_index=0, camera_key="front", frame_index=0))

    assert "\x1b[31mWARNING: skipped invalid training data:" in capsys.readouterr().err


def test_torch_wrapper_skips_invalid_episode_with_red_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "dataset"
    _write_episode_metadata(
        dataset,
        [
            {"episode_index": 0, "video_paths": {"front": "videos/front_0.mp4"}},
            {"episode_index": 1, "video_paths": {"front": "videos/front_1.mp4"}},
        ],
        modality_validity={
            "schema_version": MODALITY_VALIDITY_SCHEMA_VERSION,
            "training": {"episodes": {"0": {"valid": False, "reason": "recording invalid"}}},
        },
    )
    config = DatasetInterfaceConfig(
        dataset_root=dataset,
        mask=MaskConfig(final_mask_root=tmp_path / "unused_masks"),
    )
    rows = [
        {"episode_index": 0, "frame_index": 0, "source": "invalid"},
        {"episode_index": 1, "frame_index": 0, "source": "valid"},
    ]
    wrapped = TorchMaskedAugmentDataset(
        rows,
        pipeline_factory=lambda: MaskedFramePipeline(config),
        camera_keys=(),
    )
    try:
        sample = wrapped[0]
    finally:
        wrapped.close()

    assert sample["episode_index"] == 1
    assert sample["source"] == "valid"
    assert "\x1b[31mWARNING: skipped invalid training data:" in capsys.readouterr().err


@pytest.mark.optional
def test_missing_view_uses_reference_dataset_resolution_without_video_or_mask_io(tmp_path: Path) -> None:
    reference = tmp_path / "reference_a"
    reference_video_rel = "videos/observation.images.front/chunk-000/episode_000000.mp4"
    _write_video(
        reference / reference_video_rel,
        [np.full((24, 32, 3), (20, 60, 100), dtype=np.uint8)],
    )
    _write_episode_metadata(
        reference,
        [
            {
                "episode_index": 0,
                "frame_count": 1,
                "video_paths": {"front": reference_video_rel},
            }
        ],
    )

    missing = tmp_path / "missing_b"
    _write_episode_metadata(
        missing,
        [
            {
                "episode_index": 0,
                "frame_count": 3,
                "video_paths": {"front": "videos/missing_front.mp4"},
            }
        ],
    )
    config = DatasetInterfaceConfig(
        dataset_root=missing,
        mask=MaskConfig(final_mask_root=tmp_path / "unused_masks"),
        missing_views=MissingViewConfig(
            enabled=True,
            reference_dataset_roots=(reference,),
        ),
    )

    with MaskedFramePipeline(config) as pipeline:
        sample = pipeline.read_augmented(
            FrameRequest(episode_index=0, camera_key="front", frame_index=1)
        )
        clip = pipeline.read_augmented_clip(
            episode_index=0,
            camera_key="front",
            start_frame=0,
            length=3,
        )
        assert not pipeline.videos.items
        assert not pipeline.augmenter.photometric_augmenter._plans

    assert sample.image.shape == (24, 32, 3)
    assert sample.augmented.shape == (24, 32, 3)
    assert np.count_nonzero(sample.image) == 0
    assert np.count_nonzero(sample.augmented) == 0
    assert sample.video_path is None
    assert sample.mask_manifest_path is None
    assert sample.metadata["source_view_missing"] is True
    assert sample.metadata["mask_type"] == "missing_view_full_black"
    assert sample.metadata["missing_view"]["resolution"] == [24, 32]
    assert sample.metadata["view_occlusion"]["forced_missing_view"] is True
    assert clip.shape == (3, 24, 32, 3)
    assert np.count_nonzero(clip) == 0


def test_missing_view_can_use_an_explicit_height_width_resolution(tmp_path: Path) -> None:
    dataset = tmp_path / "missing_only"
    _write_episode_metadata(
        dataset,
        [{"episode_index": 0, "frame_count": 2, "video_paths": {}}],
    )
    config = DatasetInterfaceConfig(
        dataset_root=dataset,
        mask=MaskConfig(final_mask_root=tmp_path / "unused_masks"),
        video=VideoConfig(output_layout="rgb_chw_float32"),
        missing_views=MissingViewConfig(
            enabled=True,
            explicit_resolutions={"front": (30, 40)},
        ),
    )

    rows = [{"episode_index": 0, "frame_index": 0}]
    wrapped = TorchMaskedAugmentDataset(
        rows,
        pipeline_factory=lambda: MaskedFramePipeline(config),
        camera_keys=("front",),
    )
    try:
        result = wrapped[0]
    finally:
        wrapped.close()

    image = result["observation.images.front.augmented"]
    assert image.shape == (3, 30, 40)
    assert image.dtype == np.float32
    assert np.count_nonzero(image) == 0


@pytest.mark.optional
def test_missing_view_reference_resolution_conflict_fails_at_pipeline_start(tmp_path: Path) -> None:
    references: list[Path] = []
    for name, shape in (("reference_a", (24, 32)), ("reference_c", (30, 40))):
        root = tmp_path / name
        video_rel = "videos/observation.images.front/chunk-000/episode_000000.mp4"
        _write_video(root / video_rel, [np.full((*shape, 3), 100, dtype=np.uint8)])
        _write_episode_metadata(
            root,
            [
                {
                    "episode_index": 0,
                    "frame_count": 1,
                    "video_paths": {"front": video_rel},
                }
            ],
        )
        references.append(root)
    missing = tmp_path / "missing_b"
    _write_episode_metadata(
        missing,
        [{"episode_index": 0, "frame_count": 1, "video_paths": {}}],
    )
    config = DatasetInterfaceConfig(
        dataset_root=missing,
        mask=MaskConfig(final_mask_root=tmp_path / "unused_masks"),
        missing_views=MissingViewConfig(
            enabled=True,
            reference_dataset_roots=tuple(references),
        ),
    )

    with pytest.raises(ValueError, match="inconsistent reference resolutions"):
        MaskedFramePipeline(config)


def test_missing_view_without_explicit_or_reference_resolution_remains_fail_closed(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "missing_without_resolution"
    _write_episode_metadata(
        dataset,
        [{"episode_index": 0, "frame_count": 1, "video_paths": {}}],
    )
    config = DatasetInterfaceConfig(
        dataset_root=dataset,
        mask=MaskConfig(final_mask_root=tmp_path / "unused_masks"),
        missing_views=MissingViewConfig(enabled=True),
    )

    with MaskedFramePipeline(config) as pipeline:
        with pytest.raises(InvalidTrainingDataError, match="no explicit/reference resolution"):
            pipeline.read_augmented_frame(
                episode_index=0,
                camera_key="front",
                frame_index=0,
            )
