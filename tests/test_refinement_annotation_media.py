from __future__ import annotations

from http.client import HTTPConnection
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from control_station.refinement_annotation import RefinementAnnotationGui
from control_station.refinement_annotation.media import BrowserMediaCache
from control_station.refinement_annotation.server import _AnnotationHandler


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _make_mp4v(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for the browser media compatibility test")
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=32x24:r=5:d=0.4",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _codec(path: Path) -> str:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        pytest.skip("ffprobe is required for the browser media compatibility test")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_mp4v_is_exposed_as_h264_without_touching_source(tmp_path: Path) -> None:
    source = tmp_path / "episode.mp4"
    _make_mp4v(source)
    source_before = source.read_bytes()

    cache = BrowserMediaCache(tmp_path / "cache")
    try:
        browser_path = cache.resolve(source)
        assert browser_path != source
        assert _codec(browser_path) == "h264"
        assert source.read_bytes() == source_before
        assert cache.resolve(source) == browser_path
    finally:
        cache.close()


def test_media_cache_reuses_derived_video_after_reopen(tmp_path: Path) -> None:
    source = tmp_path / "episode.mp4"
    _make_mp4v(source)
    cache_root = tmp_path / "cache"

    first_cache = BrowserMediaCache(cache_root)
    try:
        first_path = first_cache.resolve(source)
    finally:
        first_cache.close()

    second_cache = BrowserMediaCache(cache_root)
    try:
        assert second_cache.resolve(source) == first_path
    finally:
        second_cache.close()


def test_browser_compatible_h264_is_served_without_transcode(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for the browser media compatibility test")
    source = tmp_path / "episode.mp4"
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=32x24:r=5:d=0.4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    cache = BrowserMediaCache(tmp_path / "cache")
    try:
        assert cache.resolve(source) == source
    finally:
        cache.close()


def test_media_route_returns_browser_compatible_range_response(
    tmp_path: Path,
) -> None:
    source = tmp_path / "episode.mp4"
    _make_mp4v(source)

    class _MediaApplication:
        def media_path(self, episode_uuid: str, camera_key: str) -> Path:
            assert episode_uuid == "episode"
            assert camera_key == "front"
            return source

        def close(self) -> None:
            return None

    gui = RefinementAnnotationGui(open_browser=False)
    gui.server._annotation_application = _MediaApplication()  # type: ignore[attr-defined]
    try:
        gui.start()
        address = urlsplit(gui.url)
        connection = HTTPConnection(address.hostname, address.port, timeout=5.0)
        try:
            connection.request(
                "GET",
                "/media/episode/front",
                headers={"Range": "bytes=0-31"},
            )
            response = connection.getresponse()
            payload = response.read()
        finally:
            connection.close()
        assert response.status == 206
        assert response.getheader("Content-Type") == "video/mp4"
        assert response.getheader("Content-Range", "").startswith("bytes 0-31/")
        assert len(payload) == 32
    finally:
        gui.close()


def test_media_route_treats_browser_disconnect_as_cancelled_request(
    tmp_path: Path,
) -> None:
    source = tmp_path / "episode.mp4"
    source.write_bytes(b"media" * 1024)

    class _BrokenPipeWriter:
        def write(self, _payload: bytes) -> int:
            raise BrokenPipeError(32, "Broken pipe")

    class _MediaApplication:
        def media_path(self, episode_uuid: str, camera_key: str) -> Path:
            assert episode_uuid == "episode"
            assert camera_key == "front"
            return source

    application = _MediaApplication()
    handler = object.__new__(_AnnotationHandler)
    handler.server = SimpleNamespace(
        require_application=lambda: application,
        resolve_browser_media=lambda path: path,
    )
    handler.command = "GET"
    handler.headers = {}
    handler.wfile = _BrokenPipeWriter()
    handler.send_response = lambda *_args: None
    handler.send_header = lambda *_args: None
    handler._security_headers = lambda **_kwargs: None
    handler.end_headers = lambda: None

    assert handler._serve_media("/media/episode/front") is True
