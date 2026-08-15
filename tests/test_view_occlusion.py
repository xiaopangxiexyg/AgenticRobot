from __future__ import annotations

import numpy as np
import pytest

from dataset_interface.random_background import (
    BlobOcclusionConfig,
    GridOcclusionConfig,
    PhotometricAugmentConfig,
    RandomBackgroundAugmentConfig,
    RandomBackgroundVideoAugmenter,
    RectangleOcclusionConfig,
    ViewOcclusionAugmenter,
    ViewOcclusionConfig,
)


def _method_config(method: str, *, temporal_scope: str = "clip") -> ViewOcclusionConfig:
    return ViewOcclusionConfig(
        enabled=True,
        temporal_scope=temporal_scope,
        rectangle=RectangleOcclusionConfig(probability=1.0 if method == "random_rectangle" else 0.0),
        blob=BlobOcclusionConfig(probability=1.0 if method == "random_blob" else 0.0),
        grid=GridOcclusionConfig(probability=1.0 if method == "random_grid" else 0.0),
    )


@pytest.mark.parametrize("method", ["random_rectangle", "random_blob", "random_grid"])
def test_view_occlusion_methods_are_exact_deterministic_and_non_mutating(method: str) -> None:
    image = np.full((73, 97, 3), 173, dtype=np.uint8)
    original = image.copy()
    augmenter = ViewOcclusionAugmenter(_method_config(method))
    kwargs = dict(
        occlusion_ratio=0.37,
        global_seed=71,
        episode_index=4,
        camera_key="left_wrist",
        epoch=2,
        scope_anchor=19,
    )

    first, first_audit = augmenter.apply(image, **kwargs)
    second, second_audit = augmenter.apply(image, **kwargs)

    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(first, second)
    assert first_audit == second_audit
    assert first_audit["method"] == method
    assert first_audit["actual_pixels"] == round(0.37 * 73 * 97)
    assert np.count_nonzero(np.all(first == 0, axis=-1)) == first_audit["actual_pixels"]


def test_view_occlusion_zero_is_identity_and_one_is_strict_black() -> None:
    rows, columns = np.indices((31, 43))
    image = np.stack((rows + 1, columns + 1, rows + columns + 1), axis=-1).astype(np.uint8)
    augmenter = ViewOcclusionAugmenter(_method_config("random_blob"))
    common = dict(
        global_seed=13,
        episode_index=2,
        camera_key="front",
        epoch=0,
        scope_anchor=8,
    )

    unchanged, zero_audit = augmenter.apply(image, occlusion_ratio=0.0, **common)
    black, full_audit = augmenter.apply(image, occlusion_ratio=1.0, **common)

    np.testing.assert_array_equal(unchanged, image)
    assert zero_audit["method"] == "none"
    assert np.count_nonzero(black) == 0
    assert full_audit["method"] == "full_black"
    assert full_audit["actual_ratio"] == 1.0


@pytest.mark.parametrize("ratio", [-0.01, 1.01, float("nan"), float("inf")])
def test_view_occlusion_rejects_invalid_ratio(ratio: float) -> None:
    augmenter = ViewOcclusionAugmenter(_method_config("random_rectangle"))
    with pytest.raises(ValueError, match="occlusion_ratio"):
        augmenter.apply(
            np.ones((8, 10, 3), dtype=np.uint8),
            occlusion_ratio=ratio,
            global_seed=1,
            episode_index=0,
            camera_key="front",
            epoch=0,
            scope_anchor=0,
        )


def test_view_occlusion_clip_reuses_mask_and_frame_scope_changes_it() -> None:
    images = np.stack([np.full((48, 64, 3), 100 + index, dtype=np.uint8) for index in range(3)])
    masks = np.ones(images.shape[:3], dtype=np.uint8)
    disabled_photometric = PhotometricAugmentConfig(enabled=False)

    clip_augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            provider="original",
            photometric=disabled_photometric,
            view_occlusion=_method_config("random_grid", temporal_scope="clip"),
        )
    )
    clip = clip_augmenter.apply_batch(
        images,
        masks,
        episode_index=1,
        camera_key="right_wrist",
        frame_indices=[10, 11, 12],
        occlusion_ratio=0.42,
    )
    clip_masks = [np.all(frame == 0, axis=-1) for frame in clip]
    np.testing.assert_array_equal(clip_masks[0], clip_masks[1])
    np.testing.assert_array_equal(clip_masks[0], clip_masks[2])

    frame_augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            provider="original",
            photometric=disabled_photometric,
            view_occlusion=_method_config("random_grid", temporal_scope="frame"),
        )
    )
    per_frame = frame_augmenter.apply_batch(
        images,
        masks,
        episode_index=1,
        camera_key="right_wrist",
        frame_indices=[10, 11, 12],
        occlusion_ratio=0.42,
    )
    assert not np.array_equal(np.all(per_frame[0] == 0, axis=-1), np.all(per_frame[1] == 0, axis=-1))


def test_view_occlusion_runs_after_photometric_and_before_layout_normalization() -> None:
    image = np.full((16, 20, 3), 120, dtype=np.uint8)
    keep = np.ones(image.shape[:2], dtype=np.uint8)
    photometric = PhotometricAugmentConfig(
        exposure_probability=1.0,
        exposure_gain_range=(1.5, 1.5),
        contrast_probability=0.0,
        gamma_probability=0.0,
        white_balance_probability=0.0,
        saturation_probability=0.0,
        hue_probability=0.0,
        illumination_probability=0.0,
        quality_probability=0.0,
    )
    augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            provider="original",
            photometric=photometric,
            view_occlusion=_method_config("random_blob"),
        )
    )

    output = augmenter.apply(
        image,
        keep,
        episode_index=0,
        camera_key="front",
        frame_index=0,
        occlusion_ratio=1.0,
        forced_missing_view=True,
    )
    audit = augmenter.view_occlusion_metadata(
        width=20,
        height=16,
        episode_index=0,
        camera_key="front",
        epoch=0,
        frame_index=0,
        occlusion_ratio=1.0,
        forced_missing_view=True,
    )

    assert np.count_nonzero(output) == 0
    assert audit is not None
    assert audit["forced_missing_view"] is True


@pytest.mark.parametrize("method", ["random_rectangle", "random_blob", "random_grid"])
def test_high_resolution_occlusion_uses_bounded_generation_grid(method: str) -> None:
    height, width = 720, 1280
    image = np.full((height, width, 3), 91, dtype=np.uint8)
    augmenter = ViewOcclusionAugmenter(_method_config(method))

    output, audit = augmenter.apply(
        image,
        occlusion_ratio=0.45,
        global_seed=31,
        episode_index=2,
        camera_key="front",
        epoch=0,
        scope_anchor=7,
    )

    generation_height, generation_width = audit["method_parameters"]["generation_shape"]
    assert max(generation_height, generation_width) <= 192
    assert abs(audit["actual_ratio"] - 0.45) <= 0.0025
    assert output.shape == image.shape


def test_full_black_fast_path_skips_mask_and_photometric_work() -> None:
    image = np.full((720, 1280, 3), 177, dtype=np.uint8)
    augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            provider="procedural",
            view_occlusion=_method_config("random_blob"),
        )
    )

    # The deliberately invalid source mask would fail background composition;
    # ratio=1 must bypass it and all photometric planning.
    output = augmenter.apply(
        image,
        object(),
        episode_index=0,
        camera_key="left_wrist",
        frame_index=0,
        occlusion_ratio=1.0,
        forced_missing_view=True,
    )

    assert output.shape == image.shape
    assert np.count_nonzero(output) == 0
    assert len(augmenter.photometric_augmenter._plans) == 0
    assert len(augmenter.view_occlusion_augmenter._masks) == 0


def test_high_resolution_mask_cache_has_a_bounded_worker_memory_footprint() -> None:
    config = _method_config("random_grid")
    augmenter = ViewOcclusionAugmenter(config, cache_size=128)
    image = np.full((240, 320, 3), 120, dtype=np.uint8)

    for scope_anchor in range(config.mask_cache_size + 5):
        augmenter.apply(
            image,
            occlusion_ratio=0.3,
            global_seed=5,
            episode_index=0,
            camera_key="front",
            epoch=0,
            scope_anchor=scope_anchor,
        )

    assert augmenter.cache_size == config.mask_cache_size
    assert len(augmenter._masks) == config.mask_cache_size
