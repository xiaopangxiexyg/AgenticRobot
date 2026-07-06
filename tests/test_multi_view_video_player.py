from __future__ import annotations

import importlib

import pytest


cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

viewer = importlib.import_module("utils.portable_vis.multi_view.multi_view_video_player")


def test_parse_size_accepts_common_separators() -> None:
    assert viewer._parse_size("1280x720") == (1280, 720)
    assert viewer._parse_size("1280,720") == (1280, 720)


def test_parse_layout_validates_capacity() -> None:
    assert viewer._parse_layout("auto", 5) is None
    assert viewer._parse_layout("2x3", 5) == (2, 3)
    with pytest.raises(ValueError, match="only 4 cells"):
        viewer._parse_layout("2x2", 5)


def test_parse_offsets_accepts_csv_and_fills_missing() -> None:
    assert viewer._parse_offsets(["0,-0.25", "0.5"], 4) == [0.0, -0.25, 0.5, 0.0]


def test_labels_fall_back_to_file_stems(tmp_path) -> None:
    paths = [tmp_path / "left_camera.mp4", tmp_path / "right_camera.mp4"]
    assert viewer._labels_for_paths(["left"], paths) == ["left", "right_camera"]


def test_parse_path_input_text_accepts_newlines_quotes_and_globs(tmp_path) -> None:
    video_with_space = tmp_path / "left camera.mp4"
    folder = tmp_path / "clips"
    video_with_space.touch()
    folder.mkdir()

    text = f' "{video_with_space}"\n{folder}\n{tmp_path / "*.mp4"}\n'

    assert viewer._parse_path_input_text(text) == [
        str(video_with_space),
        str(folder),
        str(tmp_path / "*.mp4"),
    ]


def test_parse_path_input_text_supports_shell_style_multiple_paths(tmp_path) -> None:
    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    left.touch()
    right.touch()

    assert viewer._parse_path_input_text(f"{left} {right}") == [str(left), str(right)]


def test_parse_tk_drop_data_uses_tk_splitlist() -> None:
    def fake_splitlist(_value: str) -> tuple[str, str]:
        return ("/tmp/a path.mp4", "/tmp/b.mp4")

    assert viewer._parse_tk_drop_data("{/tmp/a path.mp4} /tmp/b.mp4", fake_splitlist) == [
        "/tmp/a path.mp4",
        "/tmp/b.mp4",
    ]


def test_parse_http_range_supports_open_ended_and_suffix_ranges() -> None:
    assert viewer._parse_http_range("bytes=10-19", 100) == (10, 19)
    assert viewer._parse_http_range("bytes=10-", 100) == (10, 99)
    assert viewer._parse_http_range("bytes=-20", 100) == (80, 99)
    assert viewer._parse_http_range("items=0-10", 100) is None


def test_web_video_registry_deduplicates_paths(tmp_path) -> None:
    video = tmp_path / "camera.mp4"
    video.touch()
    registry = viewer.WebVideoRegistry()

    first = registry.register(video, label="front")
    second = registry.register(video, label="front2", offset_s=0.25)

    assert first["id"] == second["id"]
    assert second["url"] == f"/media/{second['id']}"
    assert second["mode"] == "frame"
    assert second["frameUrl"] == f"/frame/{second['id']}"
    assert isinstance(second["cacheKey"], str)
    assert second["cacheKey"]
    assert second["label"] == "front2"
    assert second["offset"] == 0.25


def test_web_video_registry_unregister_removes_temporary_file(tmp_path) -> None:
    video = tmp_path / "uploaded.avi"
    video.touch()
    registry = viewer.WebVideoRegistry()

    payload = registry.register(video, temporary=True)

    assert video.exists()
    assert registry.unregister(payload["id"]) is True
    assert registry.get(payload["id"]) is None
    assert not video.exists()
    assert registry.unregister(payload["id"]) is False


def test_web_video_registry_unregister_keeps_non_temporary_file(tmp_path) -> None:
    video = tmp_path / "source.avi"
    video.touch()
    registry = viewer.WebVideoRegistry()

    payload = registry.register(video, temporary=False)

    assert registry.unregister(payload["id"]) is True
    assert video.exists()


def test_jpeg_frame_for_web_item_reads_opencv_frame(tmp_path) -> None:
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (32, 24),
    )
    assert writer.isOpened()
    try:
        for index in range(3):
            frame = np.full((24, 32, 3), index * 50, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    registry = viewer.WebVideoRegistry()
    payload = registry.register(video)
    item = registry.get(payload["id"])
    assert item is not None
    try:
        jpeg = viewer._jpeg_frame_for_web_item(item, 1, None)
    finally:
        registry.close()

    assert jpeg.startswith(b"\xff\xd8")
    assert jpeg.endswith(b"\xff\xd9")


def test_resolve_web_paths_filters_non_video_files(tmp_path) -> None:
    (tmp_path / "clip.mp4").touch()
    (tmp_path / "notes.txt").touch()

    paths = viewer._resolve_web_paths_from_text(
        str(tmp_path),
        recursive=False,
        max_videos=10,
        sort_mode="natural",
    )

    assert [path.name for path in paths] == ["clip.mp4"]


def test_safe_upload_name_strips_directories_and_unsafe_characters() -> None:
    assert viewer._safe_upload_name("../../left camera!.avi") == "left_camera_.avi"
    assert viewer._safe_upload_name("...") == "uploaded_video"


def test_unique_upload_path_does_not_overwrite_existing_file(tmp_path) -> None:
    first = viewer._unique_upload_path(tmp_path, "same.avi")
    first.touch()
    second = viewer._unique_upload_path(tmp_path, "same.avi")

    assert first.name == "same.avi"
    assert second.name == "same_0001.avi"


def test_resolve_input_paths_discovers_directory_videos_in_natural_order(tmp_path) -> None:
    (tmp_path / "cam10.mp4").touch()
    (tmp_path / "cam2.mp4").touch()
    (tmp_path / "notes.txt").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "cam1.mp4").touch()

    paths = viewer._resolve_input_paths(
        [str(tmp_path)],
        recursive=False,
        max_videos=10,
        sort_mode="natural",
        no_picker=True,
    )

    assert [path.name for path in paths] == ["cam2.mp4", "cam10.mp4"]


def test_resolve_input_paths_supports_recursive_discovery(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "cam2.mp4").touch()
    (nested / "cam1.mp4").touch()

    paths = viewer._resolve_input_paths(
        [str(tmp_path)],
        recursive=True,
        max_videos=10,
        sort_mode="natural",
        no_picker=True,
    )

    assert [path.name for path in paths] == ["cam1.mp4", "cam2.mp4"]
