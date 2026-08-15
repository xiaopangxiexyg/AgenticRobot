from __future__ import annotations

from collections import Counter
import shutil

import numpy as np
import pytest

from utils.dataset_postprocess.wrist_cam_adjust.depth import DepthRangeMaskConfig, mask_manifest_payload, pack_binary_mask
from dataset_interface.random_background import (
    FixedBorderMaskConfig,
    LocalImageBackgroundProvider,
    MaskReaderCache,
    PhotometricAugmentConfig,
    RealImageBackgroundConfig,
    RandomBackgroundAugmentConfig,
    RandomBackgroundVideoAugmenter,
    TemporalBackgroundSpec,
    TemporalRandomBackground,
    apply_random_background,
    augment_sample_from_paths,
    generate_background,
    temporal_background_from_episode,
    write_image_manifest,
)
from utils.random_background.generate_wrist_sample_video import (
    DEFAULT_FINAL_MASK_ROOT,
    build_arg_parser as build_wrist_sample_video_arg_parser,
)


_DISABLED_PHOTOMETRIC = PhotometricAugmentConfig(enabled=False)


def test_wrist_sample_video_defaults_to_final_keep_mask() -> None:
    args = build_wrist_sample_video_arg_parser().parse_args(
        ["--session", "dataset", "--output-dir", "preview"]
    )

    assert args.final_mask_root == DEFAULT_FINAL_MASK_ROOT
    assert args.use_component_masks is False
    assert args.table_mask_root is None


def test_generate_background_is_deterministic() -> None:
    first = generate_background(64, 48, seed=123, mode="block_shuffle")
    second = generate_background(64, 48, seed=123, mode="block_shuffle")

    assert first.shape == (48, 64, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert float(first.std()) > 1.0


def test_color_block_and_shape_collage_modes_are_deterministic_and_diverse() -> None:
    for mode in ("color_blocks", "shape_collage"):
        first = generate_background(96, 72, seed=20260712, mode=mode)
        second = generate_background(96, 72, seed=20260712, mode=mode)

        assert np.array_equal(first, second)
        assert first.shape == (72, 96, 3)
        assert len(np.unique(first.reshape(-1, 3), axis=0)) > 64


def test_apply_random_background_preserves_foreground() -> None:
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    frame[3:9, 4:12] = (10, 100, 200)
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[3:9, 4:12] = 255

    out = apply_random_background(frame, mask, seed=7, edge_blur_px=0)

    assert out.shape == frame.shape
    assert np.array_equal(out[4:8, 5:11], frame[4:8, 5:11])
    assert not np.array_equal(out[0, 0], frame[0, 0])


def test_temporal_background_changes_smoothly() -> None:
    stream = TemporalRandomBackground(
        TemporalBackgroundSpec(
            width=64,
            height=48,
            seed=22,
            mode="lowfreq_noise",
            min_drift_px_per_frame=0.2,
            max_drift_px_per_frame=0.2,
            acceleration_px_per_frame2=0.0,
            velocity_noise_px_per_frame=0.0,
        )
    )
    first = stream.frame(0)
    second = stream.frame(1)
    random_first = generate_background(64, 48, seed=100, mode="lowfreq_noise")
    random_second = generate_background(64, 48, seed=101, mode="lowfreq_noise")

    assert first.shape == (48, 64, 3)
    assert not np.array_equal(first, second)
    near_delta = float(np.mean(np.abs(first.astype(np.int16) - second.astype(np.int16))))
    independent_delta = float(np.mean(np.abs(random_first.astype(np.int16) - random_second.astype(np.int16))))
    assert near_delta < independent_delta


def test_temporal_background_accepts_speed_range() -> None:
    slow = TemporalRandomBackground(
        TemporalBackgroundSpec(
            width=64,
            height=48,
            seed=5,
            min_drift_px_per_frame=0.05,
            max_drift_px_per_frame=0.05,
            acceleration_px_per_frame2=0.0,
            velocity_noise_px_per_frame=0.0,
        )
    )
    fast = TemporalRandomBackground(
        TemporalBackgroundSpec(
            width=64,
            height=48,
            seed=5,
            min_drift_px_per_frame=2.5,
            max_drift_px_per_frame=2.5,
            acceleration_px_per_frame2=0.0,
            velocity_noise_px_per_frame=0.0,
        )
    )

    slow_delta = float(np.mean(np.abs(slow.frame(0).astype(np.int16) - slow.frame(10).astype(np.int16))))
    fast_delta = float(np.mean(np.abs(fast.frame(0).astype(np.int16) - fast.frame(10).astype(np.int16))))

    assert fast_delta > slow_delta


def test_default_temporal_speed_range_is_doubled_with_zero_lower_bound() -> None:
    procedural = TemporalBackgroundSpec()
    real = RealImageBackgroundConfig()

    assert (procedural.min_drift_px_per_frame, procedural.max_drift_px_per_frame) == (0.0, 6.0)
    assert (real.min_drift_px_per_frame, real.max_drift_px_per_frame) == (0.0, 7.0)


def test_default_profile_is_paired_mixed_with_original_background_slice() -> None:
    config = RandomBackgroundAugmentConfig()

    assert config.provider == "mixed"
    assert config.mixed_original_prob == 0.1
    assert config.real_image_prob == 0.4
    assert config.mixed_black_prob == 0.0
    assert config.wrist_pair_mode == "shared_scene_family"
    assert config.wrist_pair_camera_keys == ("left_wrist", "right_wrist")
    assert config.mixed_probabilities(real_image_available=True) == {
        "original": 0.1,
        "black": 0.0,
        "real_image": 0.4,
        "procedural": 0.5,
    }
    assert config.mixed_probabilities(real_image_available=False) == {
        "original": 0.1,
        "black": 0.0,
        "real_image": 0.0,
        "procedural": 0.9,
    }


def test_default_augmenter_materializes_the_real_image_provider() -> None:
    augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(photometric=_DISABLED_PHOTOMETRIC)
    )

    assert isinstance(augmenter.real_image_provider, LocalImageBackgroundProvider)


def test_mixed_provider_selection_matches_the_default_10_40_50_profile() -> None:
    augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            global_seed=20260730,
            photometric=_DISABLED_PHOTOMETRIC,
        ),
        # Selection is tested here without decoding a donor image.
        real_image_provider=object(),  # type: ignore[arg-type]
    )

    selected = Counter(
        augmenter._provider_for_sample(
            episode_index=episode,
            camera_key="left_wrist",
            frame_index=0,
            epoch=0,
        )
        for episode in range(10000)
    )

    assert 0.08 <= selected["original"] / 10000.0 <= 0.12
    assert 0.37 <= selected["real_image"] / 10000.0 <= 0.43
    assert 0.47 <= selected["procedural"] / 10000.0 <= 0.53
    assert selected["black"] == 0


def test_mixed_real_image_branch_fails_closed_when_default_pool_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEXTELEOP_REAL_IMAGE_ROOT", "/tmp/does-not-exist-dexteleop-backgrounds")

    with pytest.raises(RuntimeError, match="can select real_image"):
        RandomBackgroundVideoAugmenter(
            RandomBackgroundAugmentConfig(photometric=_DISABLED_PHOTOMETRIC)
        )


def test_paired_temporal_background_shares_canvas_but_not_view_motion() -> None:
    left = temporal_background_from_episode(
        width=64,
        height=48,
        episode_index=3,
        mode="lowfreq_noise",
        canvas_seed=701,
        view_seed=801,
    )
    right = temporal_background_from_episode(
        width=64,
        height=48,
        episode_index=3,
        mode="lowfreq_noise",
        canvas_seed=701,
        view_seed=802,
    )

    np.testing.assert_array_equal(left.canvas, right.canvas)
    assert not np.array_equal(left.frame(0), right.frame(0))


def test_wrist_pair_provider_and_background_family_are_shared() -> None:
    rgb = np.full((48, 64, 3), 80, dtype=np.uint8)
    empty_keep = np.zeros((48, 64), dtype=np.uint8)
    augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            global_seed=101,
            provider="mixed",
            mixed_original_prob=0.0,
            mixed_black_prob=0.0,
            edge_blur_px=0,
            photometric=_DISABLED_PHOTOMETRIC,
        )
    )

    left = augmenter.apply(
        rgb,
        empty_keep,
        episode_index=7,
        camera_key="left_wrist",
        frame_index=0,
        epoch=2,
    )
    right = augmenter.apply(
        rgb,
        empty_keep,
        episode_index=7,
        camera_key="right_wrist",
        frame_index=19,
        epoch=2,
    )
    left_metadata = augmenter.background_metadata(
        episode_index=7,
        camera_key="left_wrist",
        frame_index=0,
        epoch=2,
    )
    right_metadata = augmenter.background_metadata(
        episode_index=7,
        camera_key="right_wrist",
        frame_index=19,
        epoch=2,
    )

    assert left_metadata["provider"] == right_metadata["provider"]
    assert left_metadata["background_family_id"] == right_metadata["background_family_id"]
    assert left_metadata["scene_seed"] == right_metadata["scene_seed"]
    assert left_metadata["view_seed"] != right_metadata["view_seed"]
    assert not np.array_equal(left, right)


def test_wrist_pair_mode_can_fall_back_to_camera_independent_streams() -> None:
    augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            provider="procedural",
            wrist_pair_mode="independent",
            photometric=_DISABLED_PHOTOMETRIC,
        )
    )

    left = augmenter.background_metadata(
        episode_index=1,
        camera_key="left_wrist",
        frame_index=0,
        epoch=0,
    )
    right = augmenter.background_metadata(
        episode_index=1,
        camera_key="right_wrist",
        frame_index=0,
        epoch=0,
    )

    assert left["background_family"] == "camera_independent"
    assert right["background_family"] == "camera_independent"
    assert "scene_seed" not in left
    assert "scene_seed" not in right


def test_video_augmenter_mask_roles_and_epoch_variation() -> None:
    rgb = np.full((16, 20, 3), 80, dtype=np.uint8)
    rgb[4:12, 6:14] = (20, 120, 220)
    keep = np.zeros((16, 20), dtype=np.uint8)
    keep[4:12, 6:14] = 1
    augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            global_seed=99,
            edge_blur_px=0,
            photometric=_DISABLED_PHOTOMETRIC,
        ),
    )

    first = augmenter.apply(rgb, keep, episode_index=1, camera_key="right_wrist", frame_index=3, epoch=0)
    second = augmenter.apply(rgb, keep, episode_index=1, camera_key="right_wrist", frame_index=3, epoch=0)
    changed_epoch = augmenter.apply(rgb, keep, episode_index=1, camera_key="right_wrist", frame_index=3, epoch=1)
    replace_role = augmenter.apply(rgb, keep, episode_index=1, camera_key="right_wrist", frame_index=3, epoch=0, mask_role="replace")

    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed_epoch)
    assert np.array_equal(first[5:11, 7:13], rgb[5:11, 7:13])
    assert not np.array_equal(replace_role[5:11, 7:13], rgb[5:11, 7:13])


def test_fixed_front_border_mask_defaults_match_augmentation_contract() -> None:
    config = FixedBorderMaskConfig()

    assert config.enabled is False
    assert config.side_probability == 0.25
    assert config.thickness_fraction == 0.05
    assert config.provider == "inherit"


def test_photometric_defaults_include_confirmed_white_balance_extremes() -> None:
    config = PhotometricAugmentConfig()

    assert config.enabled is True
    assert config.exposure_probability == 0.55
    assert config.exposure_gain_range == (0.55, 1.55)
    assert config.contrast_probability == 0.45
    assert config.contrast_gain_range == (0.65, 1.40)
    assert config.gamma_probability == 0.30
    assert config.gamma_range == (0.65, 1.55)
    assert config.white_balance_probability == 0.35
    assert config.warm_bgr_gains == (0.85, 1.00, 1.15)
    assert config.cool_bgr_gains == (1.15, 1.00, 0.50)
    assert config.saturation_gain_range == (0.70, 1.00)
    assert config.saturation_probability == 0.25
    assert config.hue_probability == 0.10
    assert config.hue_delta_deg == 7.0
    assert config.vignette_max_strength == 0.38
    assert config.soft_shadow_max_strength == 0.46
    assert config.low_frequency_light_max_amplitude == 0.24
    assert config.illumination_probability == 0.35
    assert config.gaussian_noise_sigma_max == 8.0
    assert config.poisson_peak_range == (45.0, 160.0)
    assert config.jpeg_quality_range == (25, 90)
    assert config.defocus_blur_sigma_max == 1.8
    assert config.motion_blur_length_range == (3, 11)
    assert config.downsample_scale_range == (0.45, 0.80)
    assert config.combo_noise_sigma == 4.0
    assert config.combo_blur_sigma == 0.70
    assert config.combo_jpeg_quality == 35
    assert config.quality_probability == 0.45


def _white_balance_only(gains: tuple[float, float, float]) -> PhotometricAugmentConfig:
    return PhotometricAugmentConfig(
        exposure_probability=0.0,
        contrast_probability=0.0,
        gamma_probability=0.0,
        white_balance_probability=1.0,
        warm_bgr_gains=gains,
        cool_bgr_gains=gains,
        saturation_probability=0.0,
        hue_probability=0.0,
        illumination_probability=0.0,
        quality_probability=0.0,
    )


def test_photometric_full_image_white_balance_uses_confirmed_warm_and_cool_gains() -> None:
    rgb = np.full((8, 10, 3), 100, dtype=np.uint8)
    keep = np.ones((8, 10), dtype=np.uint8)
    warm = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            provider="original",
            photometric=_white_balance_only((0.85, 1.00, 1.15)),
        )
    ).apply(rgb, keep, episode_index=1, camera_key="front", frame_index=0)
    cool = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            provider="original",
            photometric=_white_balance_only((1.15, 1.00, 0.50)),
        )
    ).apply(rgb, keep, episode_index=1, camera_key="front", frame_index=0)

    np.testing.assert_array_equal(warm, np.full_like(rgb, (85, 100, 115)))
    np.testing.assert_array_equal(cool, np.full_like(rgb, (115, 100, 50)))


def test_photometric_plan_is_reproducible_per_episode_camera_epoch_and_auditable() -> None:
    rgb = np.full((12, 16, 3), 120, dtype=np.uint8)
    keep = np.ones((12, 16), dtype=np.uint8)
    photometric = PhotometricAugmentConfig(
        exposure_probability=1.0,
        exposure_gain_range=(0.55, 1.55),
        contrast_probability=0.0,
        gamma_probability=0.0,
        white_balance_probability=0.0,
        saturation_probability=0.0,
        hue_probability=0.0,
        illumination_probability=0.0,
        quality_probability=0.0,
    )
    augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(global_seed=91, provider="original", photometric=photometric)
    )

    first = augmenter.apply(rgb, keep, episode_index=5, camera_key="front", frame_index=0, epoch=2)
    later_frame = augmenter.apply(rgb, keep, episode_index=5, camera_key="front", frame_index=11, epoch=2)
    changed_epoch = augmenter.apply(rgb, keep, episode_index=5, camera_key="front", frame_index=0, epoch=3)
    metadata = augmenter.photometric_metadata(
        width=16,
        height=12,
        episode_index=5,
        camera_key="front",
        epoch=2,
    )

    np.testing.assert_array_equal(first, later_frame)
    assert not np.array_equal(first, changed_epoch)
    assert metadata is not None
    assert metadata["type"] == "full_image_photometric_v1"
    assert metadata["static_scope"] == "episode_camera_epoch"
    assert metadata["quality"]["kind"] == "none"


def test_photometric_quality_noise_is_deterministic_per_frame_and_batch_reuses_plan() -> None:
    images = np.stack(
        [
            np.full((10, 12, 3), (40 + index * 20, 80, 120), dtype=np.uint8)
            for index in range(3)
        ],
        axis=0,
    )
    masks = np.ones(images.shape[:3], dtype=np.uint8)
    noise_config = PhotometricAugmentConfig(
        exposure_probability=0.0,
        contrast_probability=0.0,
        gamma_probability=0.0,
        white_balance_probability=0.0,
        saturation_probability=0.0,
        hue_probability=0.0,
        illumination_probability=0.0,
        quality_probability=1.0,
        quality_kinds=("gaussian_noise",),
        gaussian_noise_sigma_max=8.0,
    )
    noise_augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(global_seed=17, provider="original", photometric=noise_config)
    )
    same_first = noise_augmenter.apply(images[0], masks[0], episode_index=2, camera_key="front", frame_index=4)
    same_second = noise_augmenter.apply(images[0], masks[0], episode_index=2, camera_key="front", frame_index=4)
    next_frame = noise_augmenter.apply(images[0], masks[0], episode_index=2, camera_key="front", frame_index=5)

    np.testing.assert_array_equal(same_first, same_second)
    assert not np.array_equal(same_first, next_frame)

    jpeg_config = PhotometricAugmentConfig(
        exposure_probability=0.0,
        contrast_probability=0.0,
        gamma_probability=0.0,
        white_balance_probability=0.0,
        saturation_probability=0.0,
        hue_probability=0.0,
        illumination_probability=0.0,
        quality_probability=1.0,
        quality_kinds=("jpeg",),
        jpeg_quality_range=(35, 35),
    )
    batch_augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(global_seed=23, provider="original", photometric=jpeg_config)
    )
    frame_indices = [3, 4, 5]
    expected = np.stack(
        [
            batch_augmenter.apply(
                images[index],
                masks[index],
                episode_index=2,
                camera_key="front",
                frame_index=frame_index,
            )
            for index, frame_index in enumerate(frame_indices)
        ],
        axis=0,
    )
    actual = batch_augmenter.apply_batch(
        images,
        masks,
        episode_index=2,
        camera_key="front",
        frame_indices=frame_indices,
    )

    np.testing.assert_array_equal(actual, expected)
    assert len(batch_augmenter.photometric_augmenter._plans) == 1
    assert len(batch_augmenter.photometric_augmenter._prepared) == 1


def test_photometric_runs_after_mask_composition_on_the_complete_image() -> None:
    rgb = np.full((14, 18, 3), (40, 100, 160), dtype=np.uint8)
    keep = np.zeros((14, 18), dtype=np.uint8)
    keep[4:10, 5:13] = 1
    warm_gains = (0.85, 1.00, 1.15)
    common = dict(global_seed=47, provider="procedural", edge_blur_px=0)
    baseline = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(**common, photometric=_DISABLED_PHOTOMETRIC)
    ).apply(rgb, keep, episode_index=3, camera_key="left_wrist", frame_index=7, epoch=1)
    augmented = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(**common, photometric=_white_balance_only(warm_gains))
    ).apply(rgb, keep, episode_index=3, camera_key="left_wrist", frame_index=7, epoch=1)

    expected = np.clip(
        baseline.astype(np.float32)
        / 255.0
        * np.asarray(warm_gains, dtype=np.float32).reshape(1, 1, 3)
        * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)
    np.testing.assert_array_equal(augmented, expected)


def test_photometric_illumination_and_quality_candidates_are_deterministic_and_shape_preserving() -> None:
    rows, columns = np.indices((23, 31))
    image = np.stack(
        [
            (rows * 19 + columns * 7) % 256,
            (rows * 3 + columns * 23) % 256,
            (rows * 29 + columns * 11) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    keep = np.ones(image.shape[:2], dtype=np.uint8)

    def run(config: PhotometricAugmentConfig) -> np.ndarray:
        augmenter = RandomBackgroundVideoAugmenter(
            RandomBackgroundAugmentConfig(global_seed=59, provider="original", photometric=config)
        )
        first = augmenter.apply(image, keep, episode_index=4, camera_key="front", frame_index=8, epoch=2)
        repeated = augmenter.apply(image, keep, episode_index=4, camera_key="front", frame_index=8, epoch=2)
        np.testing.assert_array_equal(first, repeated)
        assert first.shape == image.shape
        assert first.dtype == np.uint8
        assert not np.array_equal(first, image)
        return first

    neutral = {
        "exposure_probability": 0.0,
        "contrast_probability": 0.0,
        "gamma_probability": 0.0,
        "white_balance_probability": 0.0,
        "saturation_probability": 0.0,
        "hue_probability": 0.0,
    }
    for illumination_kind in ("vignette", "soft_shadow", "low_frequency"):
        run(
            PhotometricAugmentConfig(
                **neutral,
                illumination_probability=1.0,
                illumination_kinds=(illumination_kind,),
                quality_probability=0.0,
            )
        )
    for quality_kind in (
        "gaussian_noise",
        "poisson_noise",
        "jpeg",
        "defocus_blur",
        "motion_blur",
        "downsample",
        "combo",
    ):
        run(
            PhotometricAugmentConfig(
                **neutral,
                illumination_probability=0.0,
                quality_probability=1.0,
                quality_kinds=(quality_kind,),
                jpeg_quality_range=(25, 25),
            )
        )


def test_fixed_front_border_mask_is_static_per_episode_and_independent_per_side() -> None:
    rgb = np.full((40, 60, 3), 80, dtype=np.uint8)
    keep = np.ones((40, 60), dtype=np.uint8)
    augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            global_seed=83,
            provider="procedural",
            edge_blur_px=0,
            fixed_mask=FixedBorderMaskConfig(
                enabled=True,
                side_probability=1.0,
                thickness_fraction=0.05,
            ),
            photometric=_DISABLED_PHOTOMETRIC,
        )
    )

    first = augmenter.apply(rgb, keep, episode_index=4, camera_key="front", frame_index=0, epoch=2)
    later_frame = augmenter.apply(rgb, keep, episode_index=4, camera_key="front", frame_index=19, epoch=2)
    next_epoch = augmenter.apply(rgb, keep, episode_index=4, camera_key="front", frame_index=0, epoch=3)
    metadata = augmenter.fixed_mask_metadata(
        width=60,
        height=40,
        episode_index=4,
        camera_key="front",
        epoch=2,
    )

    assert metadata is not None
    assert metadata["static_scope"] == "episode_epoch"
    assert metadata["selected_sides"] == ["top", "bottom", "left", "right"]
    assert metadata["thickness_px"] == {"top": 2, "bottom": 2, "left": 3, "right": 3}
    assert len({item["camera_key"] for item in metadata["backgrounds"].values()}) == 4
    np.testing.assert_array_equal(first, later_frame)
    assert not np.array_equal(first, next_epoch)
    np.testing.assert_array_equal(first[8:32, 10:50], rgb[8:32, 10:50])
    assert not np.array_equal(first[0:4, 20:24], first[12:16, 0:4])


def test_fixed_front_border_mask_can_leave_an_episode_unmasked() -> None:
    rgb = np.full((12, 16, 3), 80, dtype=np.uint8)
    keep = np.ones((12, 16), dtype=np.uint8)
    augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            provider="procedural",
            edge_blur_px=0,
            fixed_mask=FixedBorderMaskConfig(enabled=True, side_probability=0.0),
            photometric=_DISABLED_PHOTOMETRIC,
        )
    )

    out = augmenter.apply(rgb, keep, episode_index=7, camera_key="front", frame_index=5, epoch=0)
    metadata = augmenter.fixed_mask_metadata(
        width=16,
        height=12,
        episode_index=7,
        camera_key="front",
        epoch=0,
    )

    np.testing.assert_array_equal(out, rgb)
    assert metadata is not None
    assert metadata["selected_sides"] == []


def test_fixed_front_border_mask_mixed_provider_never_falls_back_to_original_or_black() -> None:
    rgb = np.full((20, 30, 3), 80, dtype=np.uint8)
    keep = np.ones((20, 30), dtype=np.uint8)
    augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            global_seed=13,
            provider="mixed",
            mixed_original_prob=1.0,
            mixed_black_prob=0.0,
            real_image_prob=0.0,
            edge_blur_px=0,
            fixed_mask=FixedBorderMaskConfig(enabled=True, side_probability=1.0),
            photometric=_DISABLED_PHOTOMETRIC,
        )
    )

    out = augmenter.apply(rgb, keep, episode_index=2, camera_key="front", frame_index=8, epoch=0)
    metadata = augmenter.fixed_mask_metadata(
        width=30,
        height=20,
        episode_index=2,
        camera_key="front",
        epoch=0,
    )

    assert metadata is not None
    assert {item["provider"] for item in metadata["backgrounds"].values()} == {"procedural"}
    assert not np.array_equal(out[0:2, 10:14], rgb[0:2, 10:14])


@pytest.mark.integration
def test_augment_sample_from_bitpack_manifest(tmp_path) -> None:
    masks = [
        np.asarray([[1, 0], [0, 1]], dtype=np.uint8),
        np.asarray([[0, 1], [1, 0]], dtype=np.uint8),
    ]
    relative_dir = "wrist_depth/mask/observation.images.right_wrist/chunk-000/episode_000000"
    stream_rel = f"{relative_dir}/stream.bitpack.bin"
    manifest_rel = f"{relative_dir}/manifest.json"
    stream_path = tmp_path / stream_rel
    stream_path.parent.mkdir(parents=True)
    stream_path.write_bytes(b"".join(pack_binary_mask(mask) for mask in masks))
    payload = mask_manifest_payload(
        camera_key="right_wrist",
        relative_dir=relative_dir,
        stream_path=stream_rel,
        frame_count=2,
        height=2,
        width=2,
        records=[],
        config=DepthRangeMaskConfig(),
    )
    manifest_path = tmp_path / manifest_rel
    manifest_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    rgb = np.full((2, 2, 3), 50, dtype=np.uint8)
    rgb[0, 1] = (20, 120, 220)
    augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            global_seed=7,
            edge_blur_px=0,
            enable_epoch_variation=False,
            photometric=_DISABLED_PHOTOMETRIC,
        ),
    )
    cache = MaskReaderCache(max_size=2)
    try:
        out = augment_sample_from_paths(
            rgb,
            mask_manifest_path=manifest_path,
            episode_index=0,
            camera_key="right_wrist",
            frame_index=1,
            augmenter=augmenter,
            mask_cache=cache,
        )
    finally:
        cache.close()

    assert np.array_equal(out[0, 1], rgb[0, 1])
    assert not np.array_equal(out[0, 0], rgb[0, 0])


@pytest.mark.optional
def test_real_image_background_provider(tmp_path) -> None:
    cv2 = __import__("cv2")
    image_root = tmp_path / "real_bg"
    image_root.mkdir()
    for index, color in enumerate(((30, 80, 120), (120, 80, 30))):
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        image[:, :] = color
        image[:, index * 5 : index * 5 + 12] = (200, 180, 60)
        assert cv2.imwrite(str(image_root / f"bg_{index}.png"), image)
    manifest_path = image_root / "manifest.json"
    manifest = write_image_manifest(image_root, manifest_path, source="unit")
    provider = LocalImageBackgroundProvider(manifest_path=manifest_path)

    assert manifest["count"] == 2
    frame = provider.stream(width=20, height=16, episode_index=3, camera_key="left_wrist", epoch=0).frame(2)
    assert frame.shape == (16, 20, 3)

    rgb = np.full((16, 20, 3), 90, dtype=np.uint8)
    keep = np.zeros((16, 20), dtype=np.uint8)
    keep[4:12, 5:15] = 1
    augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            provider="real_image",
            edge_blur_px=0,
            photometric=_DISABLED_PHOTOMETRIC,
        ),
        real_image_provider=provider,
    )
    out = augmenter.apply(rgb, keep, episode_index=3, camera_key="left_wrist", frame_index=2)
    assert np.array_equal(out[6:10, 7:13], rgb[6:10, 7:13])
    assert not np.array_equal(out[0, 0], rgb[0, 0])

    fixed = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            provider="real_image",
            edge_blur_px=0,
            fixed_mask=FixedBorderMaskConfig(enabled=True, side_probability=1.0),
            photometric=_DISABLED_PHOTOMETRIC,
        ),
        real_image_provider=provider,
    )
    fixed_first = fixed.apply(rgb, np.ones(keep.shape, dtype=np.uint8), episode_index=3, camera_key="front", frame_index=0)
    fixed_later = fixed.apply(rgb, np.ones(keep.shape, dtype=np.uint8), episode_index=3, camera_key="front", frame_index=12)
    fixed_metadata = fixed.fixed_mask_metadata(
        width=20,
        height=16,
        episode_index=3,
        camera_key="front",
        epoch=0,
    )

    np.testing.assert_array_equal(fixed_first, fixed_later)
    assert fixed_metadata is not None
    assert {item["provider"] for item in fixed_metadata["backgrounds"].values()} == {"real_image"}
    assert not np.array_equal(fixed_first[0, 10], rgb[0, 10])


@pytest.mark.optional
def test_real_image_manifest_is_portable_with_its_image_directory(tmp_path) -> None:
    cv2 = __import__("cv2")
    source_root = tmp_path / "source_pool"
    image_root = source_root / "images"
    image_root.mkdir(parents=True)
    image = np.full((32, 48, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_root / "background.png"), image)
    manifest_path = source_root / "real_background_manifest.json"

    payload = write_image_manifest(image_root, manifest_path, source="unit")
    assert payload["image_root"] == "images"
    assert payload["image_root_kind"] == "manifest_relative"

    copied_root = tmp_path / "copied_pool"
    shutil.copytree(source_root, copied_root)
    provider = LocalImageBackgroundProvider(
        manifest_path=copied_root / "real_background_manifest.json"
    )
    assert provider.image_root == (copied_root / "images").resolve()
    assert provider.paths == [(copied_root / "images/background.png").resolve()]


@pytest.mark.optional
def test_manifest_image_root_can_be_overridden_for_legacy_absolute_manifest(
    tmp_path,
) -> None:
    cv2 = __import__("cv2")
    image_root = tmp_path / "remote_pool" / "images"
    image_root.mkdir(parents=True)
    image = np.full((24, 24, 3), 64, dtype=np.uint8)
    assert cv2.imwrite(str(image_root / "background.png"), image)
    manifest_path = tmp_path / "legacy_manifest.json"
    manifest_path.write_text(
        __import__("json").dumps(
            {
                "format": "dexteleop_real_background_manifest_v1",
                "source": "unit",
                "image_root": "/stale/local/path/images",
                "count": 1,
                "images": ["background.png"],
            }
        ),
        encoding="utf-8",
    )

    provider = LocalImageBackgroundProvider(
        manifest_path=manifest_path,
        image_root=image_root,
    )
    assert provider.image_root == image_root.resolve()
    assert provider.paths == [(image_root / "background.png").resolve()]


@pytest.mark.optional
def test_paired_real_image_background_shares_source_but_not_view(tmp_path) -> None:
    cv2 = __import__("cv2")
    image_root = tmp_path / "real_pool"
    image_root.mkdir()
    for index in range(3):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[:, :] = (20 + index * 60, 80, 180 - index * 40)
        image[:, 8 + index * 5 : 24 + index * 5] = (220, 180, 40)
        assert cv2.imwrite(str(image_root / f"background_{index}.png"), image)
    provider = LocalImageBackgroundProvider(image_root=image_root)
    augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            provider="real_image",
            edge_blur_px=0,
            photometric=_DISABLED_PHOTOMETRIC,
        ),
        real_image_provider=provider,
    )
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    keep = np.zeros((48, 64), dtype=np.uint8)

    left = augmenter.apply(
        rgb,
        keep,
        episode_index=4,
        camera_key="left_wrist",
        frame_index=0,
        epoch=3,
    )
    right = augmenter.apply(
        rgb,
        keep,
        episode_index=4,
        camera_key="right_wrist",
        frame_index=0,
        epoch=3,
    )
    streams = list(augmenter.real_streams.items.values())
    left_metadata = augmenter.background_metadata(
        episode_index=4,
        camera_key="left_wrist",
        frame_index=0,
        epoch=3,
    )
    right_metadata = augmenter.background_metadata(
        episode_index=4,
        camera_key="right_wrist",
        frame_index=0,
        epoch=3,
    )

    assert len(streams) == 2
    assert streams[0].path == streams[1].path
    assert not np.array_equal(left, right)
    assert left_metadata["background_family_id"] == right_metadata["background_family_id"]
    assert left_metadata["scene_source_shared"] is True
    assert left_metadata["scene_canvas_source"] == "shared_real_image_independent_views"


@pytest.mark.optional
def test_real_image_provider_applies_seeded_scale_and_aspect_stretch(tmp_path) -> None:
    cv2 = __import__("cv2")
    image_root = tmp_path / "real_bg"
    image_root.mkdir()
    source = np.zeros((32, 48, 3), dtype=np.uint8)
    source[:, :24] = (20, 90, 210)
    source[:, 24:] = (210, 90, 20)
    assert cv2.imwrite(str(image_root / "source.png"), source)

    config = RealImageBackgroundConfig(
        seed=9,
        min_scale=0.55,
        max_scale=0.55,
        min_aspect_stretch=1.8,
        max_aspect_stretch=1.8,
        color_jitter=0.0,
        blur_px=0,
    )
    provider = LocalImageBackgroundProvider(image_root=image_root, config=config)
    first = provider.stream(width=40, height=24, episode_index=2, camera_key="left", epoch=1)
    second = provider.stream(width=40, height=24, episode_index=2, camera_key="left", epoch=1)

    assert first.source_scale == 0.55
    assert first.source_stretch_x == 1.8
    assert first.source_stretch_y == 1.8
    np.testing.assert_array_equal(first.frame(3), second.frame(3))
