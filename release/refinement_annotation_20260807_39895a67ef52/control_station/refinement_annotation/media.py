"""Browser-compatible read-only views for finalized Episode videos.

The collection writer historically used OpenCV's ``mp4v`` codec.  That is a
valid MP4 stream for OpenCV/ffmpeg, but it is not a browser-playable source in
the Chromium builds used by the annotation workstation.  Keep the dataset
media immutable and materialize a temporary H.264 view only when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import uuid


class BrowserMediaError(RuntimeError):
    """Raised when a dataset video cannot be exposed to the browser."""


class BrowserMediaToolUnavailable(BrowserMediaError):
    """Raised when the system media compatibility tools are unavailable."""


class BrowserMediaTranscodeError(BrowserMediaError):
    """Raised when creating a browser-compatible media view fails."""


@dataclass(frozen=True)
class _VideoProbe:
    codec_name: str
    pixel_format: str
    format_names: tuple[str, ...]

    @property
    def browser_compatible(self) -> bool:
        return (
            self.codec_name == "h264"
            and "mp4" in self.format_names
            and self.pixel_format in {"yuv420p", "yuvj420p"}
        )


def _tool_path(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise BrowserMediaToolUnavailable(
            "视频需要浏览器兼容转码，但系统未找到 "
            f"{name}。请安装 ffmpeg（其中包含 ffprobe）后重启标注服务。"
        )
    return path


def _probe_video(path: Path) -> _VideoProbe:
    ffprobe = _tool_path("ffprobe")
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt",
                "-show_entries",
                "format=format_name",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BrowserMediaError(f"无法检查视频编码：{path.name}") from exc
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()[-1:]
        suffix = f" ({detail[0]})" if detail else ""
        raise BrowserMediaError(f"无法读取视频编码：{path.name}{suffix}")
    try:
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [])[0]
        format_names = tuple(
            value.strip()
            for value in str(
                (payload.get("format") or {}).get("format_name", "")
            ).split(",")
            if value.strip()
        )
        return _VideoProbe(
            codec_name=str(stream.get("codec_name", "")).strip().lower(),
            pixel_format=str(stream.get("pix_fmt", "")).strip().lower(),
            format_names=format_names,
        )
    except (IndexError, AttributeError, TypeError, ValueError) as exc:
        raise BrowserMediaError(
            f"ffprobe 返回了无法识别的视频信息：{path.name}"
        ) from exc


class BrowserMediaCache:
    """Resolve one immutable source video to a browser-playable local path."""

    _TRANSCODE_TIMEOUT_SECONDS = 30 * 60
    _CACHE_SCHEMA_VERSION = "browser-media-v1"

    @staticmethod
    def _default_cache_root() -> Path:
        configured = os.environ.get("DEXTELEOP_ANNOTATION_MEDIA_CACHE", "")
        if configured.strip():
            return Path(configured).expanduser().resolve()
        cache_base = os.environ.get("XDG_CACHE_HOME", "").strip()
        base = (
            Path(cache_base).expanduser().resolve()
            if cache_base
            else Path.home() / ".cache"
        )
        return base / "dexteleop" / "refinement-annotation-media"

    def __init__(self, cache_root: str | Path | None = None) -> None:
        self._owns_cache_root = False
        if cache_root is None:
            self._cache_root = self._default_cache_root()
            try:
                self._cache_root.mkdir(parents=True, exist_ok=True)
            except OSError:
                # A persistent cache is an optimization only.  If the user
                # cache is unavailable, retain the previous temporary-cache
                # behavior rather than preventing the annotation GUI from
                # starting.
                self._cache_root = Path(
                    tempfile.mkdtemp(prefix="dexteleop-annotation-media-")
                )
                self._owns_cache_root = True
        else:
            self._cache_root = Path(cache_root).expanduser().resolve()
        self._cache_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._source_locks: dict[str, threading.Lock] = {}
        self._resolved_paths: dict[str, Path] = {}
        self._closed = False

    @property
    def cache_root(self) -> Path:
        return self._cache_root

    def _source_key(self, source: Path) -> str:
        try:
            stat = source.stat()
        except OSError as exc:
            raise BrowserMediaError(f"视频文件不可读：{source}") from exc
        if not source.is_file() or stat.st_size <= 0:
            raise BrowserMediaError(f"视频文件不存在或为空：{source}")
        identity = (
            f"{self._CACHE_SCHEMA_VERSION}\0{source}\0"
            f"{stat.st_dev}:{stat.st_ino}:"
            f"{stat.st_size}:{stat.st_mtime_ns}"
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _lock_for(self, key: str) -> threading.Lock:
        with self._lock:
            return self._source_locks.setdefault(key, threading.Lock())

    def resolve(self, source: str | Path) -> Path:
        """Return ``source`` or a temporary H.264 view without changing it."""

        with self._lock:
            if self._closed:
                raise BrowserMediaError("视频兼容缓存已经关闭。")
        source_path = Path(source).expanduser().resolve()
        key = self._source_key(source_path)
        source_lock = self._lock_for(key)
        with source_lock:
            with self._lock:
                resolved = self._resolved_paths.get(key)
            if resolved is not None and resolved.is_file():
                return resolved
            probe = _probe_video(source_path)
            if probe.browser_compatible:
                with self._lock:
                    self._resolved_paths[key] = source_path
                return source_path

            target = self._cache_root / f"{key}.mp4"
            if target.is_file() and target.stat().st_size > 0:
                with self._lock:
                    self._resolved_paths[key] = target
                return target
            ffmpeg = _tool_path("ffmpeg")
            temporary = self._cache_root / f".{key}.{uuid.uuid4().hex}.tmp.mp4"
            try:
                result = subprocess.run(
                    [
                        ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        "-y",
                        "-i",
                        str(source_path),
                        "-map",
                        "0:v:0",
                        "-an",
                        "-vsync",
                        "0",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "18",
                        # Keep random timeline seeks bounded by a short GOP
                        # instead of making the browser decode a long
                        # interval from the previous keyframe.
                        "-g",
                        "30",
                        "-keyint_min",
                        "15",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(temporary),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self._TRANSCODE_TIMEOUT_SECONDS,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise BrowserMediaTranscodeError(
                    f"视频转为浏览器兼容格式失败：{source_path.name}"
                ) from exc
            try:
                if result.returncode != 0 or not temporary.is_file():
                    detail = (result.stderr or "").strip().splitlines()[-1:]
                    suffix = f" ({detail[0]})" if detail else ""
                    raise BrowserMediaTranscodeError(
                        f"视频转为 H.264 失败：{source_path.name}{suffix}"
                    )
                output_probe = _probe_video(temporary)
                if not output_probe.browser_compatible:
                    raise BrowserMediaTranscodeError(
                        f"转码结果仍不是浏览器兼容的 H.264 MP4：{source_path.name}"
                    )
                os.replace(temporary, target)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            with self._lock:
                self._resolved_paths[key] = target
            return target

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._owns_cache_root:
            shutil.rmtree(self._cache_root, ignore_errors=True)


__all__ = [
    "BrowserMediaCache",
    "BrowserMediaError",
    "BrowserMediaToolUnavailable",
    "BrowserMediaTranscodeError",
]
