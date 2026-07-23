from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from utils.portable_vis.DMTAC_vis.dmtac_viewer import (
    VISUAL_STYLES,
    SHOWCASE_30,
    SHOWCASE_3D_10,
    TactileChannel,
    TactileEpisode,
    _ArrayStream,
    _distributed_force_image,
    _distributed_force_visual,
    _force_metrics,
    _orient_force_grid,
    _surface_compression_direction,
    _surface_outward_direction,
    _is_distributed_force,
    _parse_args,
    _temporally_smoothed_value,
    export_episode_videos,
    load_tactile_episode,
    render_episode_frame,
)


def _write_dataset(root: Path, *, side_count: int = 2) -> None:
    (root / "meta").mkdir(parents=True)
    tactile_paths: dict[str, str] = {}
    for side_index, side in enumerate(("left", "right")[:side_count]):
        relative = Path("tactile") / f"observation.hand.{side}.tactile" / "chunk-000" / "episode_000003"
        directory = root / relative
        directory.mkdir(parents=True)
        keys = [f"{side[0].upper()}_IN", f"{side[0].upper()}_OUT"]
        streams: dict[str, object] = {}
        for key_index, key in enumerate(keys):
            values = np.stack(
                [
                    np.full((3, 4), side_index * 50 + key_index * 10 + frame, dtype=np.uint8)
                    for frame in range(3)
                ]
            )
            stream_path = directory / f"{key}.uint8.bin"
            values.tofile(stream_path)
            streams[key] = {
                "path": stream_path.relative_to(root).as_posix(),
                "dtype": "uint8",
                "shape": [3, 4],
                "frame_count": 3,
                "encoding": "raw_c_order",
                "compression": "none",
                "chunks": [],
            }
        manifest = {
            "format": "binary_stream_v1",
            "side": side,
            "frame_count": 3,
            "sensor_keys": keys,
            "frames": [
                {
                    "sample_frame_index": sample,
                    "tactile_frame_index": tactile,
                    "field_by_key": {key: "raw" for key in keys},
                }
                for sample, tactile in ((0, 0), (2, 1), (4, 2))
            ],
            "streams": streams,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        tactile_paths[side] = (relative / "manifest.json").as_posix()
    (root / "meta" / "info.json").write_text(
        json.dumps({"dexteleop_dataset_schema": "dexteleop_dataset_v2.3", "chunk_size": 1000}),
        encoding="utf-8",
    )
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 3,
                "episode_id": "episode_000003",
                "frame_count": 5,
                "duration_sec": 0.4,
                "parquet_path": "data/chunk-000/episode_000003.parquet",
                "tactile_paths": tactile_paths,
                "metadata": {"dexteleop_dataset_schema": "dexteleop_dataset_v2.3"},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_force_streams(root: Path) -> None:
    for side_index, side in enumerate(("left", "right")):
        relative = (
            Path("tactile_force")
            / f"observation.hand.{side}.distributed_force"
            / "chunk-000"
            / "episode_000003"
        )
        directory = root / relative
        directory.mkdir(parents=True)
        keys = [f"{side[0].upper()}_IN", f"{side[0].upper()}_OUT"]
        streams: dict[str, object] = {}
        for key_index, key in enumerate(keys):
            values = np.zeros((3, 6, 8, 3), dtype=np.float32)
            values[..., 0] = side_index + 0.1 * key_index
            values[..., 1] = np.arange(3, dtype=np.float32)[:, None, None]
            values[..., 2] = -0.2
            stream_path = directory / f"{key}.float32.bin"
            values.tofile(stream_path)
            streams[key] = {
                "path": stream_path.relative_to(root).as_posix(),
                "dtype": "float32",
                "shape": [6, 8, 3],
                "frame_count": 3,
                "encoding": "raw_c_order",
                "compression": "none",
            }
        manifest = {
            "format": "binary_stream_v1",
            "side": side,
            "frame_count": 3,
            "sensor_keys": keys,
            "frames": [
                {
                    "sample_frame_index": sample,
                    "tactile_frame_index": tactile,
                    "field_by_key": {key: "distributed_force" for key in keys},
                }
                for sample, tactile in ((0, 0), (2, 1), (4, 2))
            ],
            "streams": streams,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_loads_four_raw_image_channels_and_sparse_sample_mapping(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    episode = load_tactile_episode(tmp_path, episode=3)

    assert episode.schema == "dexteleop_dataset_v2.3"
    assert episode.frame_count == 5
    assert [channel.sensor_key for channel in episode.channels] == ["L_OUT", "L_IN", "R_IN", "R_OUT"]
    assert [channel.tactile_slot for channel in episode.channels] == ["outer", "inner", "inner", "outer"]
    inner = next(channel for channel in episode.channels if channel.sensor_key == "L_IN")
    assert inner.dtype == "uint8"
    assert inner.shape == (3, 4)
    # Sparse external frames use the newest tactile snapshot at-or-before the
    # dataset sample, matching replay semantics.
    np.testing.assert_array_equal(inner.frame(1), np.zeros((3, 4), dtype=np.uint8))
    np.testing.assert_array_equal(inner.frame(2), np.ones((3, 4), dtype=np.uint8))
    np.testing.assert_allclose(inner.change_series, [0, 0, 1, 0, 1])


def test_renders_current_delta_and_baseline_dashboards(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    episode = load_tactile_episode(tmp_path, episode=3)

    for mode in ("current", "delta", "baseline"):
        image = render_episode_frame(episode, 4, mode=mode, width=800)
        assert image.dtype == np.uint8
        assert image.shape[1] == 800
        assert image.shape[0] > 600
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        assert encoded.size > 1000


def test_direct_single_side_manifest_keeps_empty_panels(tmp_path: Path) -> None:
    _write_dataset(tmp_path, side_count=1)
    manifest = (
        tmp_path
        / "tactile/observation.hand.left.tactile/chunk-000/episode_000003/manifest.json"
    )

    episode = load_tactile_episode(manifest)

    assert len(episode.channels) == 2
    image = render_episode_frame(episode, 0, width=640)
    assert image.shape[1] == 640


def test_auto_prefers_derived_force_and_raw_can_be_selected(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    _write_force_streams(tmp_path)

    automatic = load_tactile_episode(tmp_path, episode=3)
    raw = load_tactile_episode(tmp_path, episode=3, tactile_source="raw")
    force = load_tactile_episode(tmp_path, episode=3, tactile_source="force")

    assert automatic.tactile_source == "force"
    assert len(automatic.channels) == 4
    assert all(channel.field_name == "distributed_force" for channel in automatic.channels)
    assert automatic.force_scales["current"][0] > 1.0
    assert automatic.force_scales["current"][0] == pytest.approx(
        automatic.force_scales["current"][1]
    )
    assert all(channel.shape == (6, 8, 3) for channel in force.channels)
    assert raw.tactile_source == "raw"
    assert all(channel.field_name == "raw" for channel in raw.channels)


def test_force_source_reports_how_to_derive_when_missing(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    try:
        load_tactile_episode(tmp_path, episode=3, tactile_source="force")
    except ValueError as exc:
        message = str(exc)
        assert "--derive-force" in message
        assert "--episode 3" in message
        assert str(tmp_path) in message
        assert "--force-sdk-root assets/DM_Tac_SDK_Publish_1.2.10" in message
    else:
        raise AssertionError("force source should fail when no derived stream exists")


def test_cli_force_derivation_defaults_to_original_sdk_result() -> None:
    args = _parse_args(["/tmp/example", "--episode", "3", "--derive-force"])

    assert args.tactile_source == "auto"
    assert args.force_base_frame_index is None
    assert not args.force_overwrite
    assert args.force_compression == "zstd"
    assert args.force_compression_level == 3
    assert args.force_chunk_frames == 16


def test_distributed_force_uses_negative_fz_heat_and_fx_fy_arrows(tmp_path: Path) -> None:
    force = np.zeros((24, 32, 3), dtype=np.float32)
    force[6:18, 8:24, 0] = 0.4
    force[6:18, 8:24, 1] = -0.3
    force[6:18, 8:24, 2] = -1.2
    stream = _ArrayStream([np.zeros_like(force), force])
    channel = TactileChannel(
        side="left",
        sensor_key="L_FORCE",
        field_name="distributed_force",
        dtype="float32",
        shape=force.shape,
        stream=stream,
        sample_indices=(0, 1),
        tactile_indices=(0, 1),
        change_series=np.asarray([0.0, np.mean(np.abs(force))], dtype=np.float32),
    )
    episode = TactileEpisode(
        dataset_root=tmp_path,
        episode_index=0,
        schema="dexteleop_dataset_v2.3",
        frame_count=2,
        channels=[channel],
        source_paths=(),
    )

    assert _is_distributed_force(channel, force)
    force_image, normal_ref, stats = _distributed_force_image(
        force,
        signed_normal=False,
    )
    assert force_image.shape == (400, 600, 3)
    assert normal_ref > 1.0
    assert "Fx=" in stats and "Fy=" in stats and "Fz=" in stats

    dashboard = render_episode_frame(episode, 1, mode="current", width=800)
    delta_dashboard = render_episode_frame(episode, 1, mode="delta", width=800)
    assert dashboard.shape == delta_dashboard.shape
    assert not np.array_equal(dashboard, delta_dashboard)


def test_all_presentation_styles_render_from_same_force_episode(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    _write_force_streams(tmp_path)
    episode = load_tactile_episode(tmp_path, episode=3, tactile_source="force")

    images = {
        style: render_episode_frame(episode, 4, mode="baseline", style=style, width=800)
        for style in VISUAL_STYLES
    }

    assert {image.shape for image in images.values()} == {
        (703, 800, 3),
        (450, 800, 3),
    }
    assert images["surface_turbo_hero"].shape == (450, 800, 3)
    assert all(image.dtype == np.uint8 for image in images.values())
    assert not np.array_equal(images["classic"], images["surface"])
    assert not np.array_equal(images["vectors"], images["split"])


def test_all_presentation_styles_also_support_raw_numeric_frames(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    episode = load_tactile_episode(tmp_path, episode=3, tactile_source="raw")

    for style in VISUAL_STYLES:
        image = render_episode_frame(episode, 4, mode="delta", style=style, width=640)
        expected_height = 360 if "surface" in style else 703
        assert image.shape == (expected_height, 640, 3)
        assert image.dtype == np.uint8


def test_exports_one_or_multiple_video_variants(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    _write_force_streams(dataset)
    episode = load_tactile_episode(dataset, episode=3, tactile_source="force")

    single = export_episode_videos(
        episode,
        tmp_path / "single.avi",
        styles="surface",
        modes="current",
        width=640,
        fps=12.0,
        codec="MJPG",
        start_frame=1,
        end_frame=4,
        frame_step=2,
    )
    multiple = export_episode_videos(
        episode,
        tmp_path / "variants.avi",
        styles="classic,contour",
        modes="current,delta",
        width=640,
        codec="MJPG",
        end_frame=1,
    )

    assert [item.path.name for item in single] == ["single.avi"]
    assert single[0].frame_count == 2
    capture = cv2.VideoCapture(str(single[0].path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(12.0, abs=0.1)
    finally:
        capture.release()
    assert len(multiple) == 4
    assert {item.path.name for item in multiple} == {
        "variants--classic--current.avi",
        "variants--classic--delta.avi",
        "variants--contour--current.avi",
        "variants--contour--delta.avi",
    }
    assert all(item.path.stat().st_size > 1000 for item in multiple)


def test_cli_video_export_defaults_and_selections() -> None:
    args = _parse_args(
        [
            "/tmp/example",
            "--export-video",
            "/tmp/show.mp4",
            "--style",
            "neon",
            "--export-styles",
            "neon,surface",
            "--export-modes",
            "current,delta",
            "--export-fps",
            "30",
        ]
    )

    assert args.export_video == Path("/tmp/show.mp4")
    assert args.style == "neon"
    assert args.export_styles == "neon,surface"
    assert args.export_modes == "current,delta"
    assert args.export_fps == 30.0


def test_standard_profile_locks_approved_presentation_settings() -> None:
    args = _parse_args(
        [
            "/tmp/example",
            "--profile",
            "standard",
            "--tactile-source",
            "raw",
            "--style",
            "neon",
            "--mode",
            "delta",
            "--width",
            "640",
            "--temporal-smoothing",
            "0.0",
            "--no-text",
        ]
    )

    assert args.profile == "standard"
    assert args.tactile_source == "force"
    assert args.style == args.export_styles == "surface_turbo_top"
    assert args.mode == args.export_modes == "current"
    assert args.width == 2560
    assert args.temporal_smoothing == pytest.approx(0.18)
    assert args.no_text


def test_surface_render_supports_text_free_presentation(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    _write_force_streams(tmp_path)
    episode = load_tactile_episode(tmp_path, episode=3, tactile_source="force")

    labeled = render_episode_frame(
        episode,
        4,
        mode="current",
        style="surface_turbo_top",
        width=800,
        temporal_smoothing=0.18,
    )
    clean = render_episode_frame(
        episode,
        4,
        mode="current",
        style="surface_turbo_top",
        width=800,
        temporal_smoothing=0.18,
        show_text=False,
    )

    assert clean.shape == labeled.shape == (450, 800, 3)
    assert not np.array_equal(clean, labeled)


def test_center_of_pressure_is_hidden_below_contact_gate() -> None:
    noise = np.zeros((20, 30, 3), dtype=np.float32)
    noise[4:8, 10:14, 2] = -0.08
    contact = noise.copy()
    contact[4:8, 10:14, 2] = -0.45

    noise_metrics = _force_metrics(noise, normal_ref=1.0)
    contact_metrics = _force_metrics(contact, normal_ref=1.0)

    assert noise_metrics.center_of_pressure is None
    assert noise_metrics.contact_strength == pytest.approx(0.08)
    assert contact_metrics.center_of_pressure is not None
    assert contact_metrics.center_of_pressure[0] == pytest.approx(11.5 / 29.0)
    assert contact_metrics.center_of_pressure[1] == pytest.approx(5.5 / 19.0)


def test_outer_portrait_transform_adds_vertical_mirror() -> None:
    outer = np.zeros((20, 30, 3), dtype=np.float32)
    inner = np.zeros_like(outer)
    outer[4, 5] = (1.0, 2.0, -3.0)
    inner[4, 24] = (1.0, 2.0, -3.0)

    outer_portrait = _orient_force_grid(outer, "outer")
    inner_portrait = _orient_force_grid(inner, "inner")
    outer_point = np.unravel_index(np.argmax(-outer_portrait[..., 2]), outer_portrait.shape[:2])
    inner_point = np.unravel_index(np.argmax(-inner_portrait[..., 2]), inner_portrait.shape[:2])

    assert outer_portrait.shape == inner_portrait.shape == (30, 20, 3)
    assert outer_point == (24, 15)
    assert inner_point == (5, 4)
    np.testing.assert_allclose(outer_portrait[outer_point][:2], (2.0, 1.0))
    np.testing.assert_allclose(inner_portrait[inner_point][:2], (-2.0, 1.0))


def test_each_physical_pad_uses_its_gripper_outward_normal() -> None:
    np.testing.assert_allclose(_surface_outward_direction("left", "outer"), (-1.0, 0.0))
    np.testing.assert_allclose(_surface_outward_direction("left", "inner"), (1.0, 0.0))
    np.testing.assert_allclose(_surface_outward_direction("right", "inner"), (-1.0, 0.0))
    np.testing.assert_allclose(_surface_outward_direction("right", "outer"), (1.0, 0.0))
    np.testing.assert_allclose(_surface_compression_direction("left", "outer"), (1.0, 0.0))
    np.testing.assert_allclose(_surface_compression_direction("left", "inner"), (-1.0, 0.0))
    np.testing.assert_allclose(_surface_compression_direction("right", "inner"), (1.0, 0.0))
    np.testing.assert_allclose(_surface_compression_direction("right", "outer"), (-1.0, 0.0))


def test_3d_surface_projects_tangential_force_as_arrows() -> None:
    normal_only = np.zeros((20, 30, 3), dtype=np.float32)
    normal_only[5:15, 8:22, 2] = -0.7
    with_shear = normal_only.copy()
    with_shear[5:15, 8:22, 0] = 0.5
    with_shear[5:15, 8:22, 1] = -0.35

    normal_image, _, _ = _distributed_force_visual(
        normal_only,
        signed_normal=False,
        style="surface",
        normal_ref=1.0,
        vector_ref=1.0,
    )
    shear_image, _, _ = _distributed_force_visual(
        with_shear,
        signed_normal=False,
        style="surface",
        normal_ref=1.0,
        vector_ref=1.0,
    )

    assert normal_image.shape == shear_image.shape
    assert not np.array_equal(normal_image, shear_image)


def test_showcase_30_is_curated_unique_and_cli_selectable() -> None:
    assert len(SHOWCASE_30) == 30
    assert len(set(SHOWCASE_30)) == 30
    assert all(style in VISUAL_STYLES for style, _ in SHOWCASE_30)
    assert all(mode in {"current", "delta", "baseline"} for _, mode in SHOWCASE_30)

    args = _parse_args(
        ["/tmp/example", "--export-video", "/tmp/show.mp4", "--export-showcase-30"]
    )
    assert args.export_showcase_30


def test_showcase_3d_10_and_display_only_temporal_smoothing() -> None:
    assert len(SHOWCASE_3D_10) == 10
    assert len({style for style, _, _ in SHOWCASE_3D_10}) == 10
    assert all(style in VISUAL_STYLES for style, _, _ in SHOWCASE_3D_10)
    assert all(0.0 < smoothing <= 0.26 for _, _, smoothing in SHOWCASE_3D_10)

    values = [
        np.zeros((2, 2, 3), dtype=np.float32),
        np.full((2, 2, 3), 10.0, dtype=np.float32),
        np.zeros((2, 2, 3), dtype=np.float32),
    ]
    channel = TactileChannel(
        side="left",
        sensor_key="smooth",
        field_name="distributed_force",
        dtype="float32",
        shape=(2, 2, 3),
        stream=_ArrayStream(values),
        sample_indices=(0, 1, 2),
        tactile_indices=(0, 1, 2),
        change_series=np.zeros(3, dtype=np.float32),
    )
    smoothed = _temporally_smoothed_value(
        channel,
        1,
        mode="current",
        current=values[1],
        amount=0.18,
    )

    np.testing.assert_allclose(smoothed, 8.2)
    np.testing.assert_array_equal(values[1], np.full((2, 2, 3), 10.0, dtype=np.float32))
    args = _parse_args(
        ["/tmp/example", "--export-video", "/tmp/3d.mp4", "--export-showcase-3d-10"]
    )
    assert args.export_showcase_3d_10
