#!/usr/bin/env python3
"""Synchronized multi-video viewer for quick dataset inspection."""

from __future__ import annotations

import argparse
import glob
import json
import math
import mimetypes
import re
import shutil
import shlex
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence
from urllib.parse import parse_qs, unquote, urlparse
import webbrowser

try:
    import cv2
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("OpenCV is required to run this script. Install python-opencv/cv2 first.") from exc

import numpy as np


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
    ".gif",
}

DEFAULT_CANVAS_SIZE = (1600, 900)
DEFAULT_MAX_VIDEOS = 16
TRACKBAR_MAX = 1000

LEFT_KEYS = {81, 2424832, 65361}
RIGHT_KEYS = {83, 2555904, 65363}
UP_KEYS = {82, 2490368, 65362}
DOWN_KEYS = {84, 2621440, 65364}
HOME_KEYS = {2359296, 65360}
END_KEYS = {2293760, 65367}


@dataclass
class VideoSource:
    path: Path
    capture: cv2.VideoCapture
    frame_count: int
    fps: float
    width: int
    height: int
    label: str
    offset_s: float = 0.0
    cached_index: int = field(default=-1, init=False)
    cached_frame: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def duration_s(self) -> float:
        if self.frame_count <= 0 or self.fps <= 0.0:
            return 0.0
        return max(0.0, (self.frame_count - 1) / self.fps)

    def frame_index_for_time(self, global_time_s: float) -> int | None:
        local_time_s = global_time_s + self.offset_s
        if local_time_s < 0.0:
            return None
        index = int(math.floor(local_time_s * self.fps))
        return max(0, min(index, self.frame_count - 1))

    def local_time_for_global(self, global_time_s: float) -> float:
        return global_time_s + self.offset_s

    def read_frame(self, frame_index: int) -> np.ndarray:
        index = max(0, min(int(frame_index), self.frame_count - 1))
        if index == self.cached_index and self.cached_frame is not None:
            return self.cached_frame

        if index != self.cached_index + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)

        ok, frame = self.capture.read()
        if not ok or frame is None:
            frame = _message_frame(
                max(self.width, 320),
                max(self.height, 240),
                "frame unavailable",
                color=(32, 46, 72),
            )
        else:
            frame = _ensure_bgr(frame)

        self.cached_index = index
        self.cached_frame = frame
        return frame

    def close(self) -> None:
        self.capture.release()


@dataclass
class PlayerConfig:
    window_name: str
    canvas_size: tuple[int, int]
    layout: str
    seek_step_s: float
    snapshot_dir: Path
    loop: bool
    show_help: bool


@dataclass
class PlayerRuntime:
    playing: bool
    speed: float
    elapsed_s: float
    duration_s: float
    notice: str = ""
    notice_until: float = 0.0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect multiple videos in one synchronized OpenCV window. "
            "Inputs may be files, directories, or glob patterns."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Video files, directories, or glob patterns. With the default web UI these preload into the page.",
    )
    parser.add_argument(
        "--ui",
        choices=("web", "opencv"),
        default="web",
        help="User interface. The web UI supports browser-native drag-and-drop.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directories and ** glob patterns recursively.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=DEFAULT_MAX_VIDEOS,
        help="Maximum number of videos to load after discovery.",
    )
    parser.add_argument(
        "--sort",
        choices=("natural", "name", "path", "mtime", "none"),
        default="natural",
        help="Ordering used after expanding directories/globs.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help="Optional labels. Accepts space-separated values or comma-separated values.",
    )
    parser.add_argument(
        "--offsets",
        nargs="*",
        default=[],
        help=(
            "Per-video time offsets in seconds. The sampled local time is "
            "global_time + offset. Accepts space-separated or comma-separated values."
        ),
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Initial playback speed multiplier.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Initial global playback time in seconds.",
    )
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="Open the player paused.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop when the longest synchronized timeline reaches the end.",
    )
    parser.add_argument(
        "--canvas-size",
        default=f"{DEFAULT_CANVAS_SIZE[0]}x{DEFAULT_CANVAS_SIZE[1]}",
        help="Rendered canvas size as WIDTHxHEIGHT.",
    )
    parser.add_argument(
        "--layout",
        default="auto",
        help="Grid layout: auto or ROWSxCOLS, for example 2x3.",
    )
    parser.add_argument(
        "--seek-step",
        type=float,
        default=5.0,
        help="Seconds jumped by left/right arrows.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("multi_view_snapshots"),
        help="Directory used when pressing s to save the current canvas.",
    )
    parser.add_argument(
        "--window-name",
        default="Multi View Video Player",
        help="OpenCV window title.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host used by the web UI.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port used by the web UI. If occupied, nearby ports are tried.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Print the web UI URL without opening a browser.",
    )
    parser.add_argument(
        "--no-picker",
        action="store_true",
        help="With --ui opencv, fail instead of opening a Tk file picker when no inputs are provided.",
    )
    parser.add_argument(
        "--no-help",
        action="store_true",
        help="Start with the expanded help overlay hidden.",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        help="Render one synchronized canvas to this image path and exit.",
    )
    parser.add_argument(
        "--at",
        type=float,
        help="Timestamp used with --contact-sheet. Defaults to --start-time.",
    )
    return parser.parse_args(argv)


def _pick_videos_from_dialog(max_videos: int) -> list[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Tkinter file dialog is unavailable: {exc}") from exc

    dnd_files = None
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore[import-not-found]

        root = TkinterDnD.Tk()
        dnd_files = DND_FILES
    except Exception:  # noqa: BLE001
        root = tk.Tk()

    root.title("Multi View Video Picker")
    root.geometry("1080x620")
    root.minsize(900, 500)

    selected_paths: list[Path] = []
    recursive_var = tk.BooleanVar(value=True)
    header_text = tk.StringVar()
    path_text = tk.StringVar()

    def refresh_header() -> None:
        drop_hint = "Drag files/folders here, paste paths below, or use Add files." if dnd_files else (
            "Paste/type paths below, or use Add files. Install tkinterdnd2 to enable drag-and-drop."
        )
        header_text.set(
            f"Selected {len(selected_paths)} / {max_videos} videos. "
            f"{drop_hint}"
        )

    header = tk.Label(root, textvariable=header_text, anchor="w", padx=12, pady=10)
    header.pack(fill="x")

    body = tk.Frame(root)
    body.pack(fill="both", expand=True, padx=12, pady=8)

    list_frame = tk.Frame(body)
    list_frame.pack(side="left", fill="both", expand=True)

    hint = tk.Label(
        list_frame,
        text="Drop files/folders into the list. Typed paths may be files, folders, or glob patterns.",
        anchor="w",
        padx=2,
        pady=4,
    )
    hint.pack(fill="x")

    listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
    scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
    listbox.configure(yscrollcommand=scrollbar.set)
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    input_frame = tk.Frame(root)
    input_frame.pack(fill="x", padx=12, pady=8)
    input_label = tk.Label(input_frame, text="Path / folder / glob:", anchor="w")
    input_label.pack(side="left", padx=8)
    path_entry = tk.Entry(input_frame, textvariable=path_text)
    path_entry.pack(side="left", fill="x", expand=True)

    controls = tk.Frame(body)
    controls.pack(side="right", fill="y", padx=12)

    def refresh_listbox() -> None:
        listbox.delete(0, tk.END)
        for item in selected_paths:
            listbox.insert(tk.END, str(item))
        refresh_header()

    def add_paths(paths: Sequence[Path]) -> None:
        added = 0
        for candidate in paths:
            resolved = candidate.expanduser()
            if resolved in selected_paths:
                continue
            if len(selected_paths) >= max_videos:
                break
            selected_paths.append(resolved)
            added += 1
        refresh_listbox()
        if paths and added == 0 and len(selected_paths) >= max_videos:
            messagebox.showinfo("Multi View Video Picker", f"Only {max_videos} videos can be loaded at once.")

    def discover_and_add(values: Sequence[str]) -> None:
        if not values:
            return
        try:
            paths = _resolve_input_paths(
                values,
                recursive=bool(recursive_var.get()),
                max_videos=max_videos,
                sort_mode="natural",
                no_picker=True,
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showwarning("Multi View Video Picker", str(exc))
            return
        add_paths(paths)

    def add_files() -> None:
        picked = filedialog.askopenfilenames(
            parent=root,
            title="Add videos",
            initialdir=str(Path.cwd()),
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.gif"),
                ("All files", "*.*"),
            ],
        )
        add_paths([Path(item) for item in picked if item])

    def add_typed_path() -> None:
        values = _parse_path_input_text(path_text.get())
        if not values:
            messagebox.showwarning("Multi View Video Picker", "Please type or paste a path first.")
            return
        discover_and_add(values)
        path_text.set("")
        path_entry.focus_set()

    def add_folder() -> None:
        folder = filedialog.askdirectory(parent=root, title="Add all videos from a folder", initialdir=str(Path.cwd()))
        if not folder:
            return
        paths = _videos_in_directory(Path(folder), recursive=bool(recursive_var.get()))
        if not paths:
            messagebox.showwarning("Multi View Video Picker", "No supported videos were found in that folder.")
            return
        add_paths(paths)

    def remove_selected() -> None:
        indices = list(listbox.curselection())
        for index in reversed(indices):
            del selected_paths[index]
        refresh_listbox()

    def move_selected(delta: int) -> None:
        indices = list(listbox.curselection())
        if not indices:
            return
        if delta < 0:
            ordered = indices
        else:
            ordered = list(reversed(indices))
        for index in ordered:
            new_index = index + delta
            if new_index < 0 or new_index >= len(selected_paths):
                continue
            selected_paths[index], selected_paths[new_index] = selected_paths[new_index], selected_paths[index]
        refresh_listbox()
        for index in indices:
            new_index = max(0, min(index + delta, len(selected_paths) - 1))
            listbox.selection_set(new_index)

    def clear_all() -> None:
        selected_paths.clear()
        refresh_listbox()

    def start() -> None:
        if not selected_paths:
            messagebox.showwarning("Multi View Video Picker", "Please add at least one video first.")
            return
        root.quit()

    def cancel() -> None:
        selected_paths.clear()
        root.quit()

    def on_drop(event: object) -> str:
        data = str(getattr(event, "data", ""))
        values = _parse_tk_drop_data(data, root.tk.splitlist)
        discover_and_add(values)
        return "break"

    def register_drop_target(widget: object) -> None:
        if dnd_files is None:
            return
        try:
            widget.drop_target_register(dnd_files)  # type: ignore[attr-defined]
            widget.dnd_bind("<<Drop>>", on_drop)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return

    tk.Button(controls, text="Add files", command=add_files, width=20).pack(fill="x", pady=4)
    tk.Button(controls, text="Add folder", command=add_folder, width=20).pack(fill="x", pady=4)
    tk.Button(controls, text="Add typed path", command=add_typed_path, width=20).pack(fill="x", pady=4)
    tk.Checkbutton(controls, text="Folder recursive", variable=recursive_var, anchor="w").pack(fill="x", pady=4)
    tk.Button(controls, text="Move up", command=lambda: move_selected(-1), width=20).pack(fill="x", pady=6)
    tk.Button(controls, text="Move down", command=lambda: move_selected(1), width=20).pack(fill="x", pady=4)
    tk.Button(controls, text="Remove selected", command=remove_selected, width=20).pack(fill="x", pady=6)
    tk.Button(controls, text="Clear all", command=clear_all, width=20).pack(fill="x", pady=4)
    tk.Button(controls, text="Start", command=start, width=20).pack(fill="x", pady=10)
    tk.Button(controls, text="Cancel", command=cancel, width=20).pack(fill="x")

    path_entry.bind("<Return>", lambda _event: add_typed_path())
    register_drop_target(root)
    register_drop_target(listbox)
    register_drop_target(path_entry)

    root.protocol("WM_DELETE_WINDOW", cancel)
    refresh_listbox()
    root.mainloop()
    root.destroy()

    return [path for path in selected_paths[:max_videos] if path.exists()]


def _parse_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX,]\s*(\d+)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError(f"Expected WIDTHxHEIGHT, got: {value}")
    width = int(match.group(1))
    height = int(match.group(2))
    if width < 320 or height < 240:
        raise argparse.ArgumentTypeError("Canvas size must be at least 320x240.")
    return width, height


def _parse_layout(value: str, count: int) -> tuple[int, int] | None:
    normalized = value.strip().lower()
    if normalized == "auto":
        return None
    match = re.fullmatch(r"(\d+)\s*x\s*(\d+)", normalized)
    if not match:
        raise ValueError(f"Layout must be 'auto' or ROWSxCOLS, got: {value}")
    rows = int(match.group(1))
    cols = int(match.group(2))
    if rows <= 0 or cols <= 0:
        raise ValueError("Layout rows and columns must be positive.")
    if rows * cols < count:
        raise ValueError(f"Layout {rows}x{cols} has only {rows * cols} cells for {count} videos.")
    return rows, cols


def _split_csv_like(values: Sequence[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in str(value).split(",") if part.strip())
    return items


def _strip_wrapping_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def _parse_path_input_text(text: str) -> list[str]:
    values: list[str] = []
    for raw_line in text.replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        unquoted = _strip_wrapping_quotes(line)
        if Path(unquoted).expanduser().exists() or _is_glob_pattern(unquoted):
            values.append(unquoted)
            continue

        try:
            parts = shlex.split(line)
        except ValueError:
            parts = [unquoted]
        values.extend(_strip_wrapping_quotes(part) for part in parts if part.strip())
    return values


def _parse_tk_drop_data(data: str, splitlist: object) -> list[str]:
    try:
        items = list(splitlist(data))  # type: ignore[operator]
    except Exception:  # noqa: BLE001
        return _parse_path_input_text(data)
    return [_strip_wrapping_quotes(str(item)) for item in items if str(item).strip()]


def _parse_offsets(values: Sequence[str], count: int) -> list[float]:
    raw_items = _split_csv_like(values)
    if len(raw_items) > count:
        raise ValueError(f"Received {len(raw_items)} offsets for {count} videos.")
    offsets: list[float] = []
    for item in raw_items:
        try:
            offsets.append(float(item))
        except ValueError as exc:
            raise ValueError(f"Invalid offset value: {item}") from exc
    offsets.extend([0.0] * (count - len(offsets)))
    return offsets


def _labels_for_paths(values: Sequence[str], paths: Sequence[Path]) -> list[str]:
    raw_items = _split_csv_like(values)
    if len(raw_items) > len(paths):
        raise ValueError(f"Received {len(raw_items)} labels for {len(paths)} videos.")
    labels = raw_items[:]
    labels.extend(path.stem for path in paths[len(labels) :])
    return labels


def _natural_key(path: Path) -> tuple[object, ...]:
    pieces = re.split(r"(\d+)", path.name.lower())
    return tuple(int(piece) if piece.isdigit() else piece for piece in pieces)


def _is_glob_pattern(value: str) -> bool:
    return any(char in value for char in "*?[")


def _videos_in_directory(directory: Path, *, recursive: bool) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return [path for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]


def _sort_paths(paths: list[Path], mode: str) -> list[Path]:
    if mode == "none":
        return paths
    if mode == "mtime":
        return sorted(paths, key=lambda path: (path.stat().st_mtime, str(path)))
    if mode == "name":
        return sorted(paths, key=lambda path: (path.name.lower(), str(path)))
    if mode == "path":
        return sorted(paths, key=lambda path: str(path).lower())
    return sorted(paths, key=lambda path: (_natural_key(path), str(path).lower()))


def _resolve_input_paths(
    values: Sequence[str],
    *,
    recursive: bool,
    max_videos: int,
    sort_mode: str,
    no_picker: bool,
) -> list[Path]:
    if max_videos <= 0:
        raise ValueError("--max-videos must be greater than 0.")

    if not values:
        if no_picker:
            raise ValueError("No inputs were provided.")
        paths = _pick_videos_from_dialog(max_videos)
    else:
        paths = []
        for value in values:
            expanded = Path(value).expanduser()
            if _is_glob_pattern(value):
                matches = [Path(item) for item in glob.glob(str(expanded), recursive=recursive)]
                if not matches:
                    print(f"Warning: no files matched glob pattern: {value}", file=sys.stderr)
                paths.extend(path for path in matches if path.is_file())
            elif expanded.is_dir():
                discovered = _videos_in_directory(expanded, recursive=recursive)
                if not discovered:
                    print(f"Warning: no supported videos found in directory: {expanded}", file=sys.stderr)
                paths.extend(discovered)
            else:
                paths.append(expanded)

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.expanduser().resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path.expanduser())

    deduped = _sort_paths(deduped, sort_mode)
    if len(deduped) > max_videos:
        print(
            f"Warning: found {len(deduped)} videos; loading the first {max_videos}. "
            "Use --max-videos to change this.",
            file=sys.stderr,
        )
        deduped = deduped[:max_videos]
    if not deduped:
        raise ValueError("No videos were selected or discovered.")
    return deduped


def _open_video(path: Path, *, label: str, offset_s: float) -> VideoSource:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Video does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Video path is not a file: {resolved}")

    capture = cv2.VideoCapture(str(resolved))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {resolved}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Could not read frame count from video: {resolved}")
    if fps <= 0.0:
        fps = 30.0

    return VideoSource(
        path=resolved,
        capture=capture,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        label=label,
        offset_s=offset_s,
    )


def _open_videos(paths: Sequence[Path], labels: Sequence[str], offsets: Sequence[float]) -> list[VideoSource]:
    sources: list[VideoSource] = []
    try:
        for path, label, offset_s in zip(paths, labels, offsets):
            sources.append(_open_video(path, label=label, offset_s=offset_s))
    except Exception:
        _close_sources(sources)
        raise
    return sources


def _close_sources(sources: Sequence[VideoSource]) -> None:
    for source in sources:
        source.close()


def _timeline_duration(sources: Sequence[VideoSource]) -> float:
    if not sources:
        return 0.0
    return max(max(0.0, source.duration_s - source.offset_s) for source in sources)


def _reference_frame_step(sources: Sequence[VideoSource]) -> float:
    fps_values = [source.fps for source in sources if source.fps > 0.0]
    if not fps_values:
        return 1.0 / 30.0
    return 1.0 / max(fps_values)


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _message_frame(width: int, height: int, message: str, *, color: tuple[int, int, int]) -> np.ndarray:
    canvas = np.full((max(1, height), max(1, width), 3), color, dtype=np.uint8)
    _draw_text(
        canvas,
        message,
        (24, max(34, height // 2)),
        scale=0.72,
        color=(232, 236, 242),
        thickness=2,
        max_width=max(1, width - 48),
    )
    return canvas


def _resize_to_fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    source_h, source_w = image.shape[:2]
    if source_h == 0 or source_w == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / source_w, height / source_h)
    resized = cv2.resize(
        image,
        (max(1, int(source_w * scale)), max(1, int(source_h * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.full((height, width, 3), (18, 20, 24), dtype=np.uint8)
    offset_x = (width - resized.shape[1]) // 2
    offset_y = (height - resized.shape[0]) // 2
    canvas[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized
    return canvas


def _auto_grid_shape(count: int, content_size: tuple[int, int]) -> tuple[int, int]:
    if count <= 1:
        return 1, 1

    content_w, content_h = content_size
    target_aspect = 16.0 / 9.0
    best: tuple[float, int, int] | None = None
    for rows in range(1, count + 1):
        cols = math.ceil(count / rows)
        tile_w = max(1, content_w / cols)
        tile_h = max(1, content_h / rows)
        aspect_error = abs((tile_w / tile_h) - target_aspect)
        empty_cells = rows * cols - count
        score = empty_cells * 20.0 + aspect_error + rows * 0.02
        if best is None or score < best[0]:
            best = (score, rows, cols)
    assert best is not None
    return best[1], best[2]


def _tile_rects(
    count: int,
    *,
    canvas_size: tuple[int, int],
    layout: str,
) -> list[tuple[int, int, int, int]]:
    canvas_w, canvas_h = canvas_size
    top_h = 104
    bottom_h = 58
    gap = 10
    margin_x = 12
    content_x = margin_x
    content_y = top_h
    content_w = max(1, canvas_w - 2 * margin_x)
    content_h = max(1, canvas_h - top_h - bottom_h)

    parsed_layout = _parse_layout(layout, count)
    rows, cols = parsed_layout if parsed_layout is not None else _auto_grid_shape(count, (content_w, content_h))
    tile_w = max(1, int((content_w - gap * (cols - 1)) / cols))
    tile_h = max(1, int((content_h - gap * (rows - 1)) / rows))

    rects: list[tuple[int, int, int, int]] = []
    for index in range(count):
        row = index // cols
        col = index % cols
        x = content_x + col * (tile_w + gap)
        y = content_y + row * (tile_h + gap)
        rects.append((x, y, tile_w, tile_h))
    return rects


def _trim_text_to_width(text: str, max_width: int, scale: float, thickness: int) -> str:
    if max_width <= 0:
        return ""
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] <= max_width:
        return text
    ellipsis = "..."
    low = 0
    high = len(text)
    while low < high:
        mid = (low + high + 1) // 2
        candidate = text[:mid] + ellipsis
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if width <= max_width:
            low = mid
        else:
            high = mid - 1
    return text[:low] + ellipsis


def _draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    max_width: int | None = None,
) -> None:
    text_to_draw = text
    if max_width is not None:
        text_to_draw = _trim_text_to_width(text, max_width, scale, thickness)
    cv2.putText(canvas, text_to_draw, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:04.1f}"
    return f"{minutes:02d}:{secs:04.1f}"


def _draw_header(canvas: np.ndarray, sources: Sequence[VideoSource], runtime: PlayerRuntime) -> None:
    canvas_w = canvas.shape[1]
    status = "PLAY" if runtime.playing else "PAUSE"
    cv2.rectangle(canvas, (0, 0), (canvas_w, 96), (14, 18, 26), -1)
    _draw_text(
        canvas,
        f"Multi View  |  {len(sources)} videos",
        (18, 34),
        scale=0.82,
        color=(246, 248, 252),
        thickness=2,
        max_width=canvas_w - 36,
    )
    status_color = (78, 204, 122) if runtime.playing else (78, 154, 230)
    cv2.rectangle(canvas, (18, 48), (92, 76), status_color, -1)
    _draw_text(canvas, status, (28, 68), scale=0.55, color=(7, 12, 18), thickness=2)

    summary = (
        f"t {_format_time(runtime.elapsed_s)} / {_format_time(runtime.duration_s)}  |  "
        f"{runtime.speed:.2f}x  |  space play/pause  |  arrows seek  |  ,/. frame  |  h help  |  q quit"
    )
    _draw_text(canvas, summary, (106, 68), scale=0.52, color=(205, 213, 224), max_width=canvas_w - 124)

    now = time.perf_counter()
    if runtime.notice and now < runtime.notice_until:
        _draw_text(canvas, runtime.notice, (18, 92), scale=0.45, color=(112, 218, 181), max_width=canvas_w - 36)


def _draw_tile(
    canvas: np.ndarray,
    source: VideoSource,
    global_time_s: float,
    rect: tuple[int, int, int, int],
    index: int,
) -> None:
    x, y, width, height = rect
    header_h = 36
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (69, 76, 88), 1)
    cv2.rectangle(canvas, (x, y), (x + width, y + header_h), (26, 32, 43), -1)

    local_time = source.local_time_for_global(global_time_s)
    frame_index = source.frame_index_for_time(global_time_s)
    title = f"{index + 1}. {source.label}"
    meta = (
        f"{_format_time(max(0.0, local_time))}/{_format_time(source.duration_s)}  "
        f"{source.fps:.1f}fps  offset {source.offset_s:+.3f}s"
    )
    _draw_text(canvas, title, (x + 10, y + 22), scale=0.55, color=(244, 247, 252), max_width=width - 20)
    _draw_text(canvas, meta, (x + max(width // 2, 120), y + 22), scale=0.43, color=(176, 186, 198), max_width=width // 2 - 14)

    inner_x = x + 4
    inner_y = y + header_h + 4
    inner_w = max(1, width - 8)
    inner_h = max(1, height - header_h - 8)

    if frame_index is None:
        frame = _message_frame(
            max(source.width, 320),
            max(source.height, 240),
            f"starts in {_format_time(-local_time)}",
            color=(32, 38, 48),
        )
    else:
        frame = source.read_frame(frame_index)
    fitted = _resize_to_fit(frame, inner_w, inner_h)
    canvas[inner_y : inner_y + inner_h, inner_x : inner_x + inner_w] = fitted

    if frame_index is not None:
        frame_text = f"frame {frame_index + 1}/{source.frame_count}"
        cv2.rectangle(canvas, (inner_x + 8, inner_y + inner_h - 30), (inner_x + 168, inner_y + inner_h - 8), (0, 0, 0), -1)
        _draw_text(canvas, frame_text, (inner_x + 14, inner_y + inner_h - 14), scale=0.42, color=(236, 240, 246), max_width=148)


def _draw_progress(canvas: np.ndarray, runtime: PlayerRuntime) -> None:
    canvas_w, canvas_h = canvas.shape[1], canvas.shape[0]
    x0 = 18
    y0 = canvas_h - 40
    width = canvas_w - 36
    height = 12
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (45, 52, 64), -1)
    progress = 0.0 if runtime.duration_s <= 0.0 else max(0.0, min(1.0, runtime.elapsed_s / runtime.duration_s))
    cv2.rectangle(canvas, (x0, y0), (x0 + int(width * progress), y0 + height), (88, 166, 255), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (95, 104, 118), 1)
    marker_x = x0 + int(width * progress)
    cv2.line(canvas, (marker_x, y0 - 5), (marker_x, y0 + height + 5), (245, 248, 252), 2)
    _draw_text(
        canvas,
        "Drag the OpenCV trackbar to scrub. Press s for snapshot.",
        (18, canvas_h - 14),
        scale=0.43,
        color=(166, 176, 190),
        max_width=canvas_w - 36,
    )


def _draw_help_overlay(canvas: np.ndarray) -> None:
    lines = [
        "Keyboard",
        "space play/pause    left/right seek    up/down speed    home/end jump",
        ", or b previous frame    . or n next frame    r restart    l loop",
        "+/- speed    0 reset speed    s snapshot    h hide help    q or esc quit",
    ]
    overlay = canvas.copy()
    x0, y0 = 24, 112
    x1 = min(canvas.shape[1] - 24, x0 + 880)
    y1 = min(canvas.shape[0] - 78, y0 + 146)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (18, 23, 32), -1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (99, 113, 132), 1)
    cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, dst=canvas)
    for row, line in enumerate(lines):
        scale = 0.58 if row else 0.68
        color = (245, 248, 252) if row == 0 else (208, 216, 228)
        thickness = 2 if row == 0 else 1
        _draw_text(canvas, line, (x0 + 18, y0 + 32 + row * 28), scale=scale, color=color, thickness=thickness, max_width=x1 - x0 - 36)


def _build_canvas(
    sources: Sequence[VideoSource],
    runtime: PlayerRuntime,
    config: PlayerConfig,
) -> np.ndarray:
    width, height = config.canvas_size
    canvas = np.full((height, width, 3), (12, 15, 21), dtype=np.uint8)
    _draw_header(canvas, sources, runtime)

    for index, (source, rect) in enumerate(
        zip(sources, _tile_rects(len(sources), canvas_size=config.canvas_size, layout=config.layout))
    ):
        _draw_tile(canvas, source, runtime.elapsed_s, rect, index)

    _draw_progress(canvas, runtime)
    if config.show_help:
        _draw_help_overlay(canvas)
    return canvas


def _save_snapshot(canvas: np.ndarray, snapshot_dir: Path, elapsed_s: float) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = snapshot_dir / f"multi_view_{stamp}_t{elapsed_s:010.3f}.png"
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Failed to write snapshot: {path}")
    return path


def _clamp_time(value: float, duration_s: float) -> float:
    if duration_s <= 0.0:
        return max(0.0, value)
    return max(0.0, min(value, duration_s))


def _set_notice(runtime: PlayerRuntime, message: str, *, seconds: float = 2.0) -> None:
    runtime.notice = message
    runtime.notice_until = time.perf_counter() + seconds


def run_player(
    sources: Sequence[VideoSource],
    *,
    window_name: str,
    speed: float,
    start_time_s: float,
    start_paused: bool,
    config: PlayerConfig,
) -> int:
    if speed <= 0.0:
        raise ValueError("Playback speed must be greater than 0.")
    if config.seek_step_s <= 0.0:
        raise ValueError("--seek-step must be greater than 0.")

    duration_s = _timeline_duration(sources)
    runtime = PlayerRuntime(
        playing=not start_paused,
        speed=float(speed),
        elapsed_s=_clamp_time(start_time_s, duration_s),
        duration_s=duration_s,
    )
    clock_anchor = time.perf_counter() - (runtime.elapsed_s / runtime.speed)

    trackbar_state = {"ignore": False, "seek_s": None}

    def set_elapsed(value: float) -> None:
        nonlocal clock_anchor
        runtime.elapsed_s = _clamp_time(value, runtime.duration_s)
        clock_anchor = time.perf_counter() - (runtime.elapsed_s / runtime.speed)

    def set_speed(value: float) -> None:
        nonlocal clock_anchor
        runtime.speed = max(0.05, min(8.0, value))
        clock_anchor = time.perf_counter() - (runtime.elapsed_s / runtime.speed)
        _set_notice(runtime, f"speed {runtime.speed:.2f}x", seconds=1.2)

    def on_trackbar(value: int) -> None:
        if trackbar_state["ignore"] or runtime.duration_s <= 0.0:
            return
        trackbar_state["seek_s"] = (value / TRACKBAR_MAX) * runtime.duration_s

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, *config.canvas_size)
        cv2.createTrackbar("position", window_name, 0, TRACKBAR_MAX, on_trackbar)

        last_trackbar_pos = -1
        while True:
            now = time.perf_counter()
            if runtime.playing:
                runtime.elapsed_s = (now - clock_anchor) * runtime.speed
                if runtime.duration_s > 0.0 and runtime.elapsed_s >= runtime.duration_s:
                    if config.loop:
                        runtime.elapsed_s = runtime.elapsed_s % runtime.duration_s
                        clock_anchor = now - (runtime.elapsed_s / runtime.speed)
                    else:
                        runtime.elapsed_s = runtime.duration_s
                        runtime.playing = False
                        clock_anchor = now - (runtime.elapsed_s / runtime.speed)

            if trackbar_state["seek_s"] is not None:
                set_elapsed(float(trackbar_state["seek_s"]))
                trackbar_state["seek_s"] = None

            canvas = _build_canvas(sources, runtime, config)
            cv2.imshow(window_name, canvas)

            if runtime.duration_s > 0.0:
                pos = int(round((runtime.elapsed_s / runtime.duration_s) * TRACKBAR_MAX))
                pos = max(0, min(TRACKBAR_MAX, pos))
                if pos != last_trackbar_pos:
                    trackbar_state["ignore"] = True
                    cv2.setTrackbarPos("position", window_name, pos)
                    trackbar_state["ignore"] = False
                    last_trackbar_pos = pos

            key = cv2.waitKeyEx(10 if runtime.playing else 30)
            if key == -1:
                continue

            key_low = key & 0xFF
            if key_low in (ord("q"), 27):
                return 0
            if key_low == ord(" "):
                runtime.playing = not runtime.playing
                clock_anchor = time.perf_counter() - (runtime.elapsed_s / runtime.speed)
            elif key in LEFT_KEYS or key_low == ord("a"):
                set_elapsed(runtime.elapsed_s - config.seek_step_s)
            elif key in RIGHT_KEYS or key_low == ord("d"):
                set_elapsed(runtime.elapsed_s + config.seek_step_s)
            elif key in UP_KEYS:
                set_speed(runtime.speed * 1.25)
            elif key in DOWN_KEYS:
                set_speed(runtime.speed / 1.25)
            elif key in HOME_KEYS:
                set_elapsed(0.0)
            elif key in END_KEYS:
                set_elapsed(runtime.duration_s)
            elif key_low in (ord(","), ord("b")):
                runtime.playing = False
                set_elapsed(runtime.elapsed_s - _reference_frame_step(sources))
            elif key_low in (ord("."), ord("n")):
                runtime.playing = False
                set_elapsed(runtime.elapsed_s + _reference_frame_step(sources))
            elif key_low in (ord("+"), ord("=")):
                set_speed(runtime.speed * 1.25)
            elif key_low in (ord("-"), ord("_")):
                set_speed(runtime.speed / 1.25)
            elif key_low == ord("0"):
                set_speed(1.0)
            elif key_low == ord("r"):
                set_elapsed(0.0)
            elif key_low == ord("l"):
                config.loop = not config.loop
                _set_notice(runtime, f"loop {'on' if config.loop else 'off'}", seconds=1.2)
            elif key_low == ord("h"):
                config.show_help = not config.show_help
            elif key_low == ord("s"):
                path = _save_snapshot(canvas, config.snapshot_dir, runtime.elapsed_s)
                _set_notice(runtime, f"saved {path}", seconds=2.5)
    finally:
        _close_sources(sources)
        cv2.destroyAllWindows()


def render_contact_sheet(
    sources: Sequence[VideoSource],
    *,
    output_path: Path,
    timestamp_s: float,
    config: PlayerConfig,
) -> Path:
    duration_s = _timeline_duration(sources)
    runtime = PlayerRuntime(
        playing=False,
        speed=1.0,
        elapsed_s=_clamp_time(timestamp_s, duration_s),
        duration_s=duration_s,
    )
    old_help = config.show_help
    config.show_help = False
    try:
        canvas = _build_canvas(sources, runtime, config)
    finally:
        config.show_help = old_help

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to write contact sheet: {output_path}")
    return output_path


@dataclass
class WebVideoItem:
    item_id: str
    path: Path
    label: str
    offset_s: float = 0.0
    frame_count: int = 0
    fps: float = 30.0
    width: int = 0
    height: int = 0
    temporary: bool = False
    cache_key: str = field(default_factory=lambda: uuid.uuid4().hex)
    capture: cv2.VideoCapture | None = field(default=None, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def duration_s(self) -> float:
        if self.frame_count <= 0 or self.fps <= 0.0:
            return 0.0
        return max(0.0, (self.frame_count - 1) / self.fps)

    def close(self) -> None:
        with self.lock:
            if self.capture is not None:
                self.capture.release()
                self.capture = None


class WebVideoRegistry:
    def __init__(self) -> None:
        self._items: dict[str, WebVideoItem] = {}
        self._ids_by_path: dict[str, str] = {}
        self._next_id = 1

    def register(
        self,
        path: Path,
        *,
        label: str | None = None,
        offset_s: float = 0.0,
        temporary: bool = False,
    ) -> dict[str, object]:
        resolved = path.expanduser().resolve()
        key = str(resolved)
        item_id = self._ids_by_path.get(key)
        if item_id is None:
            item_id = f"v{self._next_id}"
            self._next_id += 1
            self._ids_by_path[key] = item_id
            frame_count, fps, width, height = _read_video_metadata(resolved)
            self._items[item_id] = WebVideoItem(
                item_id=item_id,
                path=resolved,
                label=label or resolved.stem,
                offset_s=float(offset_s),
                frame_count=frame_count,
                fps=fps,
                width=width,
                height=height,
                temporary=temporary,
            )
        else:
            item = self._items[item_id]
            if label:
                item.label = label
            item.offset_s = float(offset_s)
            item.temporary = item.temporary or temporary
        return self.payload_for(item_id)

    def register_many(
        self,
        paths: Sequence[Path],
        *,
        labels: Sequence[str] | None = None,
        offsets: Sequence[float] | None = None,
    ) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        for index, path in enumerate(paths):
            label = labels[index] if labels and index < len(labels) else None
            offset_s = offsets[index] if offsets and index < len(offsets) else 0.0
            payloads.append(self.register(path, label=label, offset_s=offset_s))
        return payloads

    def get(self, item_id: str) -> WebVideoItem | None:
        return self._items.get(item_id)

    def unregister(self, item_id: str) -> bool:
        item = self._items.pop(item_id, None)
        if item is None:
            return False
        self._ids_by_path.pop(str(item.path), None)
        item.close()
        if item.temporary:
            try:
                item.path.unlink(missing_ok=True)
            except OSError:
                pass
        return True

    def payload_for(self, item_id: str) -> dict[str, object]:
        item = self._items[item_id]
        return {
            "id": item.item_id,
            "kind": "server",
            "mode": "frame",
            "label": item.label,
            "path": str(item.path),
            "url": f"/media/{item.item_id}",
            "frameUrl": f"/frame/{item.item_id}",
            "cacheKey": item.cache_key,
            "offset": item.offset_s,
            "duration": item.duration_s,
            "fps": item.fps,
            "frameCount": item.frame_count,
            "width": item.width,
            "height": item.height,
        }

    def close(self) -> None:
        for item in self._items.values():
            item.close()


WEB_APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multi View Video Player</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101318;
      --panel: #181d25;
      --panel-2: #202734;
      --line: #384253;
      --text: #eef2f8;
      --muted: #9da9ba;
      --blue: #61a8ff;
      --green: #62d394;
      --red: #ff716d;
      --yellow: #e8c65a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input, select, textarea {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #242c3a;
      color: var(--text);
      min-height: 34px;
      padding: 7px 10px;
      cursor: pointer;
    }
    button:hover { border-color: var(--blue); }
    button.primary { background: var(--blue); color: #08111e; border-color: var(--blue); font-weight: 700; }
    button.danger { color: #ffd9d7; border-color: #73403f; background: #352123; }
    input, select, textarea {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0e1218;
      color: var(--text);
      padding: 8px 9px;
    }
    textarea { min-height: 76px; resize: vertical; width: 100%; }
    label { color: var(--muted); font-size: 12px; }
    .app {
      display: grid;
      grid-template-columns: 330px minmax(0, 1fr);
      min-height: 100vh;
    }
    .side {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .brand {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .brand h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
    }
    .status {
      color: var(--muted);
      font-size: 12px;
      min-height: 18px;
    }
    .drop {
      border: 1px dashed #637086;
      border-radius: 8px;
      background: #111720;
      padding: 18px;
      text-align: center;
      color: var(--muted);
      min-height: 118px;
      display: grid;
      place-items: center;
      transition: border-color 120ms ease, background 120ms ease;
    }
    .drop.dragging {
      border-color: var(--green);
      background: #122018;
      color: var(--text);
    }
    .stack { display: grid; gap: 8px; }
    .row { display: flex; gap: 8px; align-items: center; }
    .row > * { min-width: 0; }
    .row .grow { flex: 1; }
    .main {
      min-width: 0;
      display: flex;
      flex-direction: column;
    }
    .toolbar {
      position: sticky;
      top: 0;
      z-index: 5;
      display: grid;
      gap: 10px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: rgba(16, 19, 24, 0.96);
    }
    .controls {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .controls input[type="range"] {
      flex: 1 1 420px;
      min-width: 180px;
      padding: 0;
      height: 28px;
      accent-color: var(--blue);
    }
    .controls input[type="range"]::-webkit-slider-runnable-track {
      height: 10px;
      border-radius: 999px;
      background: #364153;
    }
    .controls input[type="range"]::-webkit-slider-thumb {
      width: 22px;
      height: 22px;
      margin-top: -6px;
    }
    .controls input[type="range"]::-moz-range-track {
      height: 10px;
      border-radius: 999px;
      background: #364153;
    }
    .controls input[type="range"]::-moz-range-thumb {
      width: 22px;
      height: 22px;
    }
    .time {
      min-width: 152px;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      font-size: 13px;
    }
    .grid {
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 10px;
      align-items: start;
    }
    .tile {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #0b0f15;
      min-width: 0;
    }
    .tile-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      padding: 8px;
      background: var(--panel-2);
      border-bottom: 1px solid var(--line);
    }
    .title {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
      font-weight: 650;
    }
    .tile-actions { display: flex; gap: 5px; }
    .tile-actions button {
      min-height: 28px;
      min-width: 28px;
      padding: 4px 7px;
    }
    .video-wrap {
      position: relative;
      aspect-ratio: 16 / 9;
      background: #05070a;
    }
    video, .frame-img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      background: #05070a;
    }
    .cover {
      position: absolute;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      color: #d8e1ee;
      background: rgba(5, 7, 10, 0.72);
      font-size: 14px;
    }
    .tile-foot {
      display: grid;
      grid-template-columns: 1fr 92px;
      gap: 8px;
      align-items: center;
      padding: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .tile-foot input { width: 100%; padding: 5px 6px; }
    .empty {
      min-height: 360px;
      display: grid;
      place-items: center;
      color: var(--muted);
      text-align: center;
      padding: 28px;
    }
    .list {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      min-height: 80px;
      background: #0e1218;
    }
    .list-item {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 6px;
      align-items: center;
      padding: 8px;
      border-bottom: 1px solid #242c38;
      font-size: 12px;
    }
    .list-item:last-child { border-bottom: 0; }
    .list-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      color: var(--muted);
      font-size: 11px;
    }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      .side { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <script id="initial-data" type="application/json">__INITIAL_DATA__</script>
  <div class="app">
    <aside class="side">
      <div class="brand">
        <h1>Multi View</h1>
        <span class="pill" id="countPill">0 videos</span>
      </div>
      <div class="status" id="status">Ready</div>
      <div class="drop" id="dropZone">
        <div>
          <strong>Drop videos or folders</strong><br>
          <span>Browser files play locally. Path input is served by Python.</span>
        </div>
      </div>
      <div class="stack">
        <input id="fileInput" type="file" multiple accept="video/*,.avi,.mkv,.mov,.mp4,.webm,.m4v,.gif">
        <input id="folderInput" type="file" multiple webkitdirectory>
      </div>
      <div class="stack">
        <label for="pathBox">Path, folder, or glob</label>
        <textarea id="pathBox" placeholder="/path/to/session/videos&#10;/path/to/**/*.mp4"></textarea>
        <div class="row">
          <button class="primary grow" id="addPathBtn">Add path</button>
          <button id="clearPathBtn">Clear</button>
        </div>
      </div>
      <div class="row">
        <label class="grow"><input id="recursiveBox" type="checkbox" checked> Recursive</label>
        <select id="sortSelect">
          <option value="natural">Natural</option>
          <option value="name">Name</option>
          <option value="path">Path</option>
          <option value="mtime">Modified</option>
          <option value="none">As found</option>
        </select>
      </div>
      <div class="row">
        <label class="grow">Max videos</label>
        <input id="maxVideosInput" type="number" min="1" max="128" value="16">
      </div>
      <div class="list" id="videoList"></div>
      <button class="danger" id="clearVideosBtn">Clear videos</button>
    </aside>
    <main class="main">
      <div class="toolbar">
        <div class="controls">
          <button class="primary" id="playBtn">Play</button>
          <button id="backBtn">-5s</button>
          <button id="forwardBtn">+5s</button>
          <button id="prevFrameBtn">Prev</button>
          <button id="nextFrameBtn">Next</button>
          <div class="time" id="timeLabel">00:00.0 / 00:00.0</div>
          <input id="timeline" type="range" min="0" max="100000" value="0">
        </div>
        <div class="controls">
          <label>Speed <select id="speedSelect">
            <option value="0.25">0.25x</option>
            <option value="0.5">0.5x</option>
            <option value="1" selected>1x</option>
            <option value="1.5">1.5x</option>
            <option value="2">2x</option>
            <option value="4">4x</option>
          </select></label>
          <label>Layout <select id="layoutSelect">
            <option value="auto">Auto</option>
            <option value="1">1 col</option>
            <option value="2">2 cols</option>
            <option value="3">3 cols</option>
            <option value="4">4 cols</option>
          </select></label>
          <label><input id="loopBox" type="checkbox"> Loop</label>
          <label><input id="mutedBox" type="checkbox" checked> Muted</label>
          <button id="restartBtn">Restart</button>
          <button id="snapshotBtn">Snapshot</button>
        </div>
      </div>
      <section class="grid" id="grid">
        <div class="empty">Add videos from the left panel.</div>
      </section>
    </main>
  </div>
  <script>
    const initialData = JSON.parse(document.getElementById("initial-data").textContent);
    const TIMELINE_MAX = 100000;
    const state = {
      videos: [],
      playing: false,
      globalTime: 0,
      duration: 0,
      speed: 1,
      anchorTime: 0,
      anchorClock: 0,
      lastSyncClock: 0,
      scrubbing: false,
      localCounter: 1,
      syncTimer: null
    };

    const els = {
      dropZone: document.getElementById("dropZone"),
      fileInput: document.getElementById("fileInput"),
      folderInput: document.getElementById("folderInput"),
      pathBox: document.getElementById("pathBox"),
      addPathBtn: document.getElementById("addPathBtn"),
      clearPathBtn: document.getElementById("clearPathBtn"),
      recursiveBox: document.getElementById("recursiveBox"),
      sortSelect: document.getElementById("sortSelect"),
      maxVideosInput: document.getElementById("maxVideosInput"),
      videoList: document.getElementById("videoList"),
      clearVideosBtn: document.getElementById("clearVideosBtn"),
      grid: document.getElementById("grid"),
      playBtn: document.getElementById("playBtn"),
      backBtn: document.getElementById("backBtn"),
      forwardBtn: document.getElementById("forwardBtn"),
      prevFrameBtn: document.getElementById("prevFrameBtn"),
      nextFrameBtn: document.getElementById("nextFrameBtn"),
      timeLabel: document.getElementById("timeLabel"),
      timeline: document.getElementById("timeline"),
      speedSelect: document.getElementById("speedSelect"),
      layoutSelect: document.getElementById("layoutSelect"),
      loopBox: document.getElementById("loopBox"),
      mutedBox: document.getElementById("mutedBox"),
      restartBtn: document.getElementById("restartBtn"),
      snapshotBtn: document.getElementById("snapshotBtn"),
      countPill: document.getElementById("countPill"),
      status: document.getElementById("status")
    };

    function setStatus(message) {
      els.status.textContent = message;
    }

    function formatTime(value) {
      const seconds = Math.max(0, Number(value) || 0);
      const h = Math.floor(seconds / 3600);
      const m = Math.floor((seconds % 3600) / 60);
      const s = seconds % 60;
      const ss = s.toFixed(1).padStart(4, "0");
      return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${String(m).padStart(2, "0")}:${ss}`;
    }

    function isSupportedVideoName(name) {
      return /\.(mp4|avi|mov|mkv|webm|m4v|gif)$/i.test(name || "");
    }

    function videoDuration(v) {
      return Number.isFinite(v.duration) ? Math.max(0, v.duration - v.offset) : 0;
    }

    function recomputeDuration() {
      state.duration = state.videos.reduce((best, v) => Math.max(best, videoDuration(v)), 0);
      state.globalTime = clampTime(state.globalTime);
      updateChrome();
    }

    function clampTime(value) {
      if (state.duration <= 0) return Math.max(0, value);
      return Math.max(0, Math.min(value, state.duration));
    }

    function uniqueLabel(preferred, path) {
      const base = String(preferred || "video");
      if (!state.videos.some((v) => v.label === base)) return base;
      const parts = String(path || "").split(/[\\/]/).filter(Boolean);
      if (parts.length >= 2) {
        const withParent = `${parts[parts.length - 2]}/${base}`;
        if (!state.videos.some((v) => v.label === withParent)) return withParent;
      }
      let index = 2;
      while (state.videos.some((v) => v.label === `${base} (${index})`)) index += 1;
      return `${base} (${index})`;
    }

    function addVideo(item) {
      const id = item.id || `local-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      const kind = item.kind || "local";
      if (kind !== "local" && state.videos.some((v) => v.id === id || v.serverId === id || (item.path && v.path === item.path))) return;
      const label = uniqueLabel(item.label || "video", item.path || "");
      state.videos.push({
        id,
        clientId: item.clientId || `client-${state.localCounter++}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        serverId: kind === "server" ? id : "",
        kind,
        mode: item.mode || (kind === "server" ? "frame" : "media"),
        label,
        path: item.path || "",
        url: item.url,
        frameUrl: item.frameUrl || "",
        cacheKey: item.cacheKey || id,
        objectUrl: item.objectUrl || "",
        offset: Number(item.offset || 0),
        duration: Number(item.duration || 0),
        fps: Number(item.fps || 30),
        frameCount: Number(item.frameCount || 0),
        lastFrameIndex: -1,
        file: item.file || null,
        uploading: false,
        uploadController: null,
        alive: true,
        el: null,
        cover: null,
        error: false
      });
    }

    function addLocalFiles(files) {
      let added = 0;
      for (const file of files) {
        if (!isSupportedVideoName(file.name) && !String(file.type || "").startsWith("video/")) continue;
        const displayPath = file._multiViewRelativePath || file.webkitRelativePath || file.name;
        const localId = `local-${state.localCounter++}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        const objectUrl = URL.createObjectURL(file);
        addVideo({
          id: localId,
          clientId: localId,
          kind: "local",
          label: displayPath,
          path: `${displayPath} | ${file.size} bytes | ${file.lastModified || 0} | ${localId}`,
          url: objectUrl,
          objectUrl: objectUrl,
          file: file,
          offset: 0
        });
        added += 1;
      }
      renderAll();
      setStatus(added ? `Added ${added} local video(s)` : "No supported videos found");
    }

    function addServerItems(items) {
      for (const item of items) addVideo(item);
      renderAll();
      setStatus(items.length ? `Added ${items.length} path video(s)` : "No videos added");
    }

    async function uploadForOpenCvFallback(v) {
      if (!v.alive || !v.file || v.uploading || v.mode === "frame") return;
      v.uploading = true;
      v.uploadController = new AbortController();
      if (v.cover) {
        v.cover.style.display = "flex";
        v.cover.textContent = "browser cannot decode; uploading to local OpenCV fallback...";
      }
      setStatus(`Uploading fallback for ${v.label}`);
      try {
        const response = await fetch("/api/upload", {
          method: "POST",
          headers: {
            "Content-Type": "application/octet-stream",
            "X-Filename": encodeURIComponent(v.file.name || "uploaded_video"),
            "X-Display-Path": encodeURIComponent(v.label || v.file.name || "uploaded_video")
          },
          body: v.file,
          signal: v.uploadController.signal
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Fallback upload failed");
        const replacement = data.video;
        if (!v.alive || !state.videos.includes(v)) {
          if (replacement && replacement.id) unregisterServerVideoId(replacement.id);
          return;
        }
        if (v.objectUrl) URL.revokeObjectURL(v.objectUrl);
        v.serverId = replacement.id;
        v.kind = "server";
        v.mode = "frame";
        v.path = replacement.path || v.path;
        v.url = replacement.url;
        v.frameUrl = replacement.frameUrl;
        v.cacheKey = replacement.cacheKey || replacement.id;
        v.objectUrl = "";
        v.duration = Number(replacement.duration || 0);
        v.fps = Number(replacement.fps || 30);
        v.frameCount = Number(replacement.frameCount || 0);
        v.lastFrameIndex = -1;
        v.file = null;
        v.error = false;
        v.uploading = false;
        v.uploadController = null;
        renderAll();
        setStatus(`Using OpenCV fallback: ${v.label}`);
      } catch (err) {
        if (!v.alive) return;
        v.uploading = false;
        v.uploadController = null;
        if (err && err.name === "AbortError") {
          setStatus(`Cancelled fallback for ${v.label}`);
          return;
        }
        v.error = true;
        if (v.cover) {
          v.cover.style.display = "flex";
          v.cover.textContent = String(err);
        }
        setStatus(String(err));
      }
    }

    async function addPathText() {
      const text = els.pathBox.value.trim();
      if (!text) {
        setStatus("Type a path, folder, or glob first");
        return;
      }
      setStatus("Resolving paths...");
      const response = await fetch("/api/resolve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          recursive: els.recursiveBox.checked,
          maxVideos: Number(els.maxVideosInput.value || initialData.maxVideos || 16),
          sort: els.sortSelect.value
        })
      });
      const data = await response.json();
      if (!response.ok) {
        setStatus(data.error || "Failed to resolve paths");
        return;
      }
      addServerItems(data.videos || []);
      els.pathBox.value = "";
    }

    function detachDom(v) {
      if (!v || !v.el) return;
      try {
        if (v.mode === "media") {
          v.el.pause();
          v.el.removeAttribute("src");
          v.el.load();
        } else {
          v.el.removeAttribute("src");
        }
      } catch (_err) {}
      v.el = null;
      v.cover = null;
    }

    function unregisterServerVideoId(id) {
      if (!id) return;
      fetch("/api/unregister", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id }),
        keepalive: true
      }).catch(() => {});
    }

    function unregisterServerVideo(v) {
      if (!v) return;
      const id = v.serverId || (v.kind === "server" ? v.id : "");
      unregisterServerVideoId(id);
      v.serverId = "";
    }

    function cleanupVideo(v, unregisterServer) {
      if (!v) return;
      v.alive = false;
      if (v.uploadController) {
        try { v.uploadController.abort(); } catch (_err) {}
        v.uploadController = null;
      }
      detachDom(v);
      if (v.objectUrl) {
        URL.revokeObjectURL(v.objectUrl);
        v.objectUrl = "";
      }
      if (unregisterServer) unregisterServerVideo(v);
      v.file = null;
    }

    function renderAll() {
      renderGrid();
      renderList();
      recomputeDuration();
      syncVideos(true);
    }

    function renderGrid() {
      for (const v of state.videos) detachDom(v);
      els.grid.innerHTML = "";
      if (!state.videos.length) {
        els.grid.innerHTML = '<div class="empty">Add videos from the left panel.</div>';
        return;
      }
      const cols = els.layoutSelect.value;
      els.grid.style.gridTemplateColumns = cols === "auto" ? "repeat(auto-fit, minmax(320px, 1fr))" : `repeat(${cols}, minmax(0, 1fr))`;

      state.videos.forEach((v, index) => {
        const tile = document.createElement("article");
        tile.className = "tile";
        const mediaMarkup = v.mode === "frame"
          ? '<img class="frame-img" alt="">'
          : '<video playsinline preload="metadata"></video>';
        tile.innerHTML = `
          <div class="tile-head">
            <div class="title" title="${escapeHtml(v.path || v.label)}">${index + 1}. ${escapeHtml(v.label)}</div>
            <div class="tile-actions">
              <button data-action="up" title="Move up">Up</button>
              <button data-action="down" title="Move down">Down</button>
              <button data-action="remove" title="Remove">X</button>
            </div>
          </div>
          <div class="video-wrap">
            ${mediaMarkup}
            <div class="cover"></div>
          </div>
          <div class="tile-foot">
            <div>${escapeHtml(v.kind)} ${v.path ? escapeHtml(v.path) : ""}</div>
            <label>Offset <input data-action="offset" type="number" step="0.001" value="${v.offset}"></label>
          </div>`;
        const media = v.mode === "frame" ? tile.querySelector(".frame-img") : tile.querySelector("video");
        const cover = tile.querySelector(".cover");
        if (v.mode === "media") {
          media.src = v.url;
          media.muted = els.mutedBox.checked;
          media.playbackRate = state.speed;
          media.addEventListener("loadedmetadata", () => {
            if (!v.alive || v.el !== media) return;
            v.duration = media.duration || 0;
            recomputeDuration();
            syncVideos(true);
          });
          media.addEventListener("durationchange", () => {
            if (!v.alive || v.el !== media) return;
            v.duration = media.duration || v.duration || 0;
            recomputeDuration();
          });
          media.addEventListener("error", () => {
            if (!v.alive || v.el !== media) return;
            v.error = true;
            if (cover) {
              cover.style.display = "flex";
              cover.textContent = v.file
                ? "browser cannot decode; preparing OpenCV fallback..."
                : "browser cannot decode this video; use path input or --ui opencv";
            }
            setStatus(`Cannot play in browser: ${v.label}`);
            if (v.file) uploadForOpenCvFallback(v);
          });
        } else {
          media.addEventListener("error", () => {
            if (!v.alive || v.el !== media) return;
            v.error = true;
            if (cover) {
              cover.style.display = "flex";
              cover.textContent = "failed to load decoded frame";
            }
            setStatus(`Cannot load frame: ${v.label}`);
          });
        }
        tile.addEventListener("click", (event) => {
          const action = event.target && event.target.dataset ? event.target.dataset.action : "";
          if (!action) return;
          if (action === "remove") removeVideo(v.clientId);
          if (action === "up") moveVideo(index, -1);
          if (action === "down") moveVideo(index, 1);
        });
        tile.querySelector('[data-action="offset"]').addEventListener("change", (event) => {
          if (!v.alive) return;
          v.offset = Number(event.target.value || 0);
          recomputeDuration();
          syncVideos(true);
        });
        v.el = media;
        v.cover = cover;
        els.grid.appendChild(tile);
        if (v.mode === "frame") updateFrameImage(v, state.globalTime + v.offset, true);
      });
    }

    function renderList() {
      els.countPill.textContent = `${state.videos.length} video${state.videos.length === 1 ? "" : "s"}`;
      els.videoList.innerHTML = "";
      if (!state.videos.length) {
        els.videoList.innerHTML = '<div class="list-item"><div class="list-name">No videos loaded</div></div>';
        return;
      }
      state.videos.forEach((v, index) => {
        const item = document.createElement("div");
        item.className = "list-item";
        item.innerHTML = `<div class="list-name" title="${escapeHtml(v.path || v.label)}">${index + 1}. ${escapeHtml(v.label)}</div><span class="pill">${escapeHtml(v.kind)}</span>`;
        els.videoList.appendChild(item);
      });
    }

    function removeVideo(id) {
      const index = state.videos.findIndex((v) => v.clientId === id || v.id === id || v.serverId === id);
      if (index < 0) return;
      const [v] = state.videos.splice(index, 1);
      cleanupVideo(v, true);
      renderAll();
    }

    function moveVideo(index, delta) {
      const next = index + delta;
      if (next < 0 || next >= state.videos.length) return;
      const temp = state.videos[index];
      state.videos[index] = state.videos[next];
      state.videos[next] = temp;
      renderAll();
    }

    function clearVideos() {
      pause();
      for (const v of state.videos) cleanupVideo(v, true);
      state.videos = [];
      state.globalTime = 0;
      renderAll();
      setStatus("Cleared");
    }

    function currentClockTime() {
      if (!state.playing) return state.globalTime;
      return state.anchorTime + ((performance.now() - state.anchorClock) / 1000) * state.speed;
    }

    function seekTo(value, force = true) {
      state.globalTime = clampTime(value);
      if (state.playing) {
        state.anchorTime = state.globalTime;
        state.anchorClock = performance.now();
      }
      syncVideos(force);
      updateChrome();
    }

    function play() {
      if (!state.videos.length) return;
      state.playing = true;
      state.anchorTime = state.globalTime;
      state.anchorClock = performance.now();
      syncVideos(true);
      updateChrome();
    }

    function pause() {
      state.globalTime = clampTime(currentClockTime());
      state.playing = false;
      for (const v of state.videos) {
        if (v.mode === "media" && v.el) v.el.pause();
      }
      updateChrome();
    }

    function togglePlay() {
      state.playing ? pause() : play();
    }

    function updateCover(v, local) {
      if (!v.cover) return;
      if (v.error) {
        v.cover.style.display = "flex";
        v.cover.textContent = v.mode === "frame" ? "failed to load decoded frame" : "browser cannot decode this video; use path input or --ui opencv";
      } else if (local < 0) {
        v.cover.style.display = "flex";
        v.cover.textContent = `starts in ${formatTime(-local)}`;
      } else if (v.duration && local > v.duration) {
        v.cover.style.display = "flex";
        v.cover.textContent = "ended";
      } else {
        v.cover.style.display = "none";
        v.cover.textContent = "";
      }
    }

    function updateFrameImage(v, local, force) {
      if (!v.el || !v.frameUrl || v.error) return;
      updateCover(v, local);
      if (local < 0 || (v.duration && local > v.duration)) return;
      const fps = Number(v.fps || 30);
      let frameIndex = Math.max(0, Math.floor(local * fps));
      if (v.frameCount > 0) frameIndex = Math.min(frameIndex, v.frameCount - 1);
      if (!force && frameIndex === v.lastFrameIndex) return;
      v.lastFrameIndex = frameIndex;
      v.el.src = `${v.frameUrl}?i=${frameIndex}&ck=${encodeURIComponent(v.cacheKey || v.id || "")}`;
    }

    function syncVideos(force) {
      const now = performance.now();
      const softSync = force || now - state.lastSyncClock > 800;
      if (softSync) state.lastSyncClock = now;
      for (const v of state.videos) {
        const el = v.el;
        if (!el) continue;
        if (v.error) {
          updateCover(v, state.globalTime + v.offset);
          if (v.mode === "media" && !el.paused) el.pause();
          continue;
        }
        const local = state.globalTime + v.offset;
        if (v.mode === "frame") {
          updateFrameImage(v, local, force || softSync);
          continue;
        }
        el.playbackRate = state.speed;
        el.muted = els.mutedBox.checked;
        const active = local >= 0 && (!Number.isFinite(v.duration) || !v.duration || local <= v.duration);
        const target = Math.max(0, Math.min(local, v.duration || local || 0));
        updateCover(v, local);
        const drift = Math.abs((el.currentTime || 0) - target);
        const driftLimit = state.playing ? 0.35 : 0.03;
        if (Number.isFinite(target) && (force || (softSync && drift > driftLimit))) {
          try { el.currentTime = target; } catch (_err) {}
        }
        if (state.playing && active) {
          if (el.paused) {
            const promise = el.play();
            if (promise && promise.catch) promise.catch(() => {});
          }
        } else {
          if (!el.paused) el.pause();
        }
      }
    }

    function tick() {
      if (state.playing) {
        let next = currentClockTime();
        if (state.duration > 0 && next >= state.duration) {
          if (els.loopBox.checked) {
            next = next % state.duration;
            state.anchorTime = next;
            state.anchorClock = performance.now();
          } else {
            state.globalTime = state.duration;
            pause();
            requestAnimationFrame(tick);
            return;
          }
        }
        state.globalTime = clampTime(next);
        syncVideos(false);
        updateChrome();
      }
      requestAnimationFrame(tick);
    }

    function updateChrome() {
      els.playBtn.textContent = state.playing ? "Pause" : "Play";
      els.timeLabel.textContent = `${formatTime(state.globalTime)} / ${formatTime(state.duration)}`;
      const max = Number(els.timeline.max || TIMELINE_MAX);
      const value = state.duration > 0 ? Math.round((state.globalTime / state.duration) * max) : 0;
      if (!state.scrubbing && Number(els.timeline.value) !== value) els.timeline.value = String(value);
    }

    function step(seconds) {
      seekTo(state.globalTime + seconds, true);
    }

    function frameStep(sign) {
      pause();
      seekTo(state.globalTime + sign / 30, true);
    }

    function restart() {
      seekTo(0, true);
    }

    async function filesFromDrop(event) {
      const items = Array.from(event.dataTransfer.items || []);
      if (!items.length) return Array.from(event.dataTransfer.files || []);
      const entries = items.map((item) => item.webkitGetAsEntry ? item.webkitGetAsEntry() : null).filter(Boolean);
      if (!entries.length) return Array.from(event.dataTransfer.files || []);
      const batches = await Promise.all(entries.map((entry) => readEntry(entry, "")));
      return batches.flat();
    }

    function readEntry(entry, prefix) {
      if (entry.isFile) {
        return new Promise((resolve) => entry.file((file) => {
          file._multiViewRelativePath = prefix ? `${prefix}/${file.name}` : file.name;
          resolve([file]);
        }, () => resolve([])));
      }
      if (!entry.isDirectory) return Promise.resolve([]);
      const nextPrefix = prefix ? `${prefix}/${entry.name}` : entry.name;
      const reader = entry.createReader();
      const all = [];
      return new Promise((resolve) => {
        const readBatch = () => {
          reader.readEntries(async (entries) => {
            if (!entries.length) {
              resolve(all);
              return;
            }
            const nested = await Promise.all(entries.map((child) => readEntry(child, nextPrefix)));
            all.push(...nested.flat());
            readBatch();
          }, () => resolve(all));
        };
        readBatch();
      });
    }

    function snapshot() {
      if (!state.videos.length) return;
      const cols = Math.ceil(Math.sqrt(state.videos.length));
      const rows = Math.ceil(state.videos.length / cols);
      const tileW = 480;
      const tileH = 300;
      const canvas = document.createElement("canvas");
      canvas.width = cols * tileW;
      canvas.height = rows * tileH;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#101318";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      state.videos.forEach((v, index) => {
        const x = (index % cols) * tileW;
        const y = Math.floor(index / cols) * tileH;
        ctx.fillStyle = "#181d25";
        ctx.fillRect(x, y, tileW, tileH);
        ctx.fillStyle = "#eef2f8";
        ctx.font = "16px sans-serif";
        ctx.fillText(`${index + 1}. ${v.label}`, x + 12, y + 24);
        if (v.el && ((v.mode === "media" && v.el.readyState >= 2) || (v.mode === "frame" && v.el.complete && v.el.naturalWidth > 0))) {
          const videoW = tileW;
          const videoH = tileH - 42;
          ctx.drawImage(v.el, x, y + 38, videoW, videoH);
        }
      });
      const a = document.createElement("a");
      a.download = `multi_view_${Date.now()}_t${state.globalTime.toFixed(3)}.png`;
      a.href = canvas.toDataURL("image/png");
      a.click();
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      })[ch]);
    }

    els.dropZone.addEventListener("dragover", (event) => {
      event.preventDefault();
      els.dropZone.classList.add("dragging");
    });
    els.dropZone.addEventListener("dragleave", () => els.dropZone.classList.remove("dragging"));
    els.dropZone.addEventListener("drop", async (event) => {
      event.preventDefault();
      els.dropZone.classList.remove("dragging");
      addLocalFiles(await filesFromDrop(event));
    });
    els.fileInput.addEventListener("change", () => addLocalFiles(Array.from(els.fileInput.files || [])));
    els.folderInput.addEventListener("change", () => addLocalFiles(Array.from(els.folderInput.files || [])));
    els.addPathBtn.addEventListener("click", () => addPathText().catch((err) => setStatus(String(err))));
    els.clearPathBtn.addEventListener("click", () => { els.pathBox.value = ""; });
    els.clearVideosBtn.addEventListener("click", clearVideos);
    els.playBtn.addEventListener("click", togglePlay);
    els.backBtn.addEventListener("click", () => step(-5));
    els.forwardBtn.addEventListener("click", () => step(5));
    els.prevFrameBtn.addEventListener("click", () => frameStep(-1));
    els.nextFrameBtn.addEventListener("click", () => frameStep(1));
    els.restartBtn.addEventListener("click", restart);
    els.snapshotBtn.addEventListener("click", snapshot);
    els.speedSelect.addEventListener("change", () => {
      state.speed = Number(els.speedSelect.value || 1);
      if (state.playing) {
        state.anchorTime = state.globalTime;
        state.anchorClock = performance.now();
      }
      syncVideos(true);
    });
    els.layoutSelect.addEventListener("change", renderGrid);
    els.mutedBox.addEventListener("change", () => syncVideos(false));
    els.timeline.addEventListener("pointerdown", () => { state.scrubbing = true; });
    els.timeline.addEventListener("pointerup", () => { state.scrubbing = false; updateChrome(); });
    els.timeline.addEventListener("pointercancel", () => { state.scrubbing = false; updateChrome(); });
    els.timeline.addEventListener("keydown", () => { state.scrubbing = true; });
    els.timeline.addEventListener("keyup", () => { state.scrubbing = false; updateChrome(); });
    els.timeline.addEventListener("input", () => {
      state.scrubbing = true;
      const value = (Number(els.timeline.value) / Number(els.timeline.max || TIMELINE_MAX)) * state.duration;
      seekTo(value, true);
    });
    els.timeline.addEventListener("change", () => {
      state.scrubbing = false;
      const value = (Number(els.timeline.value) / Number(els.timeline.max || TIMELINE_MAX)) * state.duration;
      seekTo(value, true);
    });
    document.addEventListener("keydown", (event) => {
      const tag = event.target && event.target.tagName ? event.target.tagName.toLowerCase() : "";
      if (["input", "textarea", "select"].includes(tag)) return;
      if (event.key === " ") { event.preventDefault(); togglePlay(); }
      if (event.key === "ArrowLeft") step(-5);
      if (event.key === "ArrowRight") step(5);
      if (event.key === ",") frameStep(-1);
      if (event.key === ".") frameStep(1);
      if (event.key === "r") restart();
      if (event.key === "l") els.loopBox.checked = !els.loopBox.checked;
    });

    els.maxVideosInput.value = String(initialData.maxVideos || 16);
    els.recursiveBox.checked = Boolean(initialData.recursive);
    els.sortSelect.value = initialData.sort || "natural";
    for (const item of initialData.videos || []) addVideo(item);
    renderAll();
    tick();
  </script>
</body>
</html>
"""


def _web_index_html(initial_data: dict[str, object]) -> bytes:
    payload = json.dumps(initial_data, ensure_ascii=True).replace("</", "<\\/")
    return WEB_APP_HTML.replace("__INITIAL_DATA__", payload).encode("utf-8")


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def _media_type_for_path(path: Path) -> str:
    media_type, _encoding = mimetypes.guess_type(str(path))
    return media_type or "application/octet-stream"


def _read_video_metadata(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0, 30.0, 0, 0
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if fps <= 0.0:
        fps = 30.0
    return max(0, frame_count), fps, max(0, width), max(0, height)


def _jpeg_frame_for_web_item(item: WebVideoItem, frame_index: int | None, timestamp_s: float | None) -> bytes:
    with item.lock:
        if item.capture is None or not item.capture.isOpened():
            item.capture = cv2.VideoCapture(str(item.path))

        if item.capture is None or not item.capture.isOpened():
            frame = _message_frame(640, 360, "video unavailable", color=(34, 38, 48))
        else:
            if frame_index is None:
                fps = item.fps if item.fps > 0.0 else 30.0
                frame_index = int(max(0.0, timestamp_s or 0.0) * fps)
            if item.frame_count > 0:
                frame_index = max(0, min(int(frame_index), item.frame_count - 1))
            else:
                frame_index = max(0, int(frame_index))
            item.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = item.capture.read()
            if not ok or frame is None:
                frame = _message_frame(
                    max(item.width, 640),
                    max(item.height, 360),
                    "frame unavailable",
                    color=(34, 38, 48),
                )
            else:
                frame = _ensure_bgr(frame)

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError("Failed to encode frame.")
    return encoded.tobytes()


def _parse_http_range(value: str | None, size: int) -> tuple[int, int] | None:
    if not value or not value.startswith("bytes=") or size <= 0:
        return None
    first_range = value[6:].split(",", 1)[0].strip()
    if "-" not in first_range:
        return None
    start_text, end_text = first_range.split("-", 1)
    try:
        if start_text == "":
            suffix = int(end_text)
            if suffix <= 0:
                return None
            start = max(0, size - suffix)
            end = size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
    except ValueError:
        return None
    if start >= size:
        return None
    end = min(end, size - 1)
    if end < start:
        return None
    return start, end


def _filter_video_paths(paths: Sequence[Path]) -> list[Path]:
    return [path for path in paths if path.suffix.lower() in VIDEO_EXTENSIONS]


def _resolve_web_paths_from_text(
    text: str,
    *,
    recursive: bool,
    max_videos: int,
    sort_mode: str,
) -> list[Path]:
    values = _parse_path_input_text(text)
    if not values:
        raise ValueError("No path text was provided.")
    paths = _resolve_input_paths(
        values,
        recursive=recursive,
        max_videos=max_videos,
        sort_mode=sort_mode,
        no_picker=True,
    )
    video_paths = _filter_video_paths(paths)
    if not video_paths:
        raise ValueError("No supported video files were found.")
    return video_paths


def _safe_upload_name(value: str) -> str:
    name = Path(value or "uploaded_video").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "uploaded_video"


def _unique_upload_path(upload_root: Path, filename: str) -> Path:
    upload_root.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_upload_name(filename)
    stem = Path(safe_name).stem or "uploaded_video"
    suffix = Path(safe_name).suffix
    for index in range(10000):
        candidate = upload_root / (safe_name if index == 0 else f"{stem}_{index:04d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not allocate an upload filename.")


def _make_web_handler(
    registry: WebVideoRegistry,
    initial_data: dict[str, object],
    *,
    max_videos: int,
    upload_root: Path,
) -> type[BaseHTTPRequestHandler]:
    class MultiViewWebHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format: str, *args: object) -> None:
            return

        def _send_bytes(self, status: int, body: bytes, media_type: str) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", media_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _send_json(self, status: int, payload: object) -> None:
            self._send_bytes(status, _json_bytes(payload), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_bytes(HTTPStatus.OK, _web_index_html(initial_data), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/ping":
                self._send_json(HTTPStatus.OK, {"ok": True})
                return
            if parsed.path.startswith("/media/"):
                self._serve_media(parsed.path.rsplit("/", 1)[-1])
                return
            if parsed.path.startswith("/frame/"):
                self._serve_frame(parsed.path.rsplit("/", 1)[-1], parsed.query)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/upload":
                self._handle_upload()
                return
            if parsed.path == "/api/unregister":
                self._handle_unregister()
                return
            if parsed.path != "/api/resolve":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                data = json.loads(body.decode("utf-8") or "{}")
                requested_max = int(data.get("maxVideos") or max_videos)
                requested_max = max(1, min(requested_max, 128))
                paths = _resolve_web_paths_from_text(
                    str(data.get("text") or ""),
                    recursive=bool(data.get("recursive")),
                    max_videos=requested_max,
                    sort_mode=str(data.get("sort") or "natural"),
                )
                payloads = registry.register_many(paths[:requested_max])
            except Exception as exc:  # noqa: BLE001
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, {"videos": payloads})

        def _handle_upload(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("Empty upload.")
                filename = unquote(self.headers.get("X-Filename") or "uploaded_video")
                label = unquote(self.headers.get("X-Display-Path") or filename)
                output_path = _unique_upload_path(upload_root, filename)
                remaining = length
                with output_path.open("wb") as handle:
                    while remaining > 0:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        handle.write(chunk)
                        remaining -= len(chunk)
                if remaining:
                    output_path.unlink(missing_ok=True)
                    raise RuntimeError("Upload ended before all bytes were received.")
                if output_path.suffix.lower() not in VIDEO_EXTENSIONS:
                    output_path.unlink(missing_ok=True)
                    raise ValueError(f"Unsupported video extension: {output_path.suffix}")
                payload = registry.register(output_path, label=Path(label).stem or output_path.stem, temporary=True)
            except Exception as exc:  # noqa: BLE001
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, {"video": payload})

        def _handle_unregister(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(body.decode("utf-8") or "{}")
                item_id = str(data.get("id") or "")
                if not item_id:
                    raise ValueError("Missing video id.")
                removed = registry.unregister(item_id)
            except Exception as exc:  # noqa: BLE001
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, {"removed": removed})

        def _serve_media(self, item_id: str) -> None:
            item = registry.get(item_id)
            if item is None or not item.path.exists():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Video not found"})
                return

            size = item.path.stat().st_size
            byte_range = _parse_http_range(self.headers.get("Range"), size)
            if byte_range is None:
                start, end = 0, max(0, size - 1)
                status = HTTPStatus.OK
            else:
                start, end = byte_range
                status = HTTPStatus.PARTIAL_CONTENT

            length = max(0, end - start + 1)
            self.send_response(status)
            self.send_header("Content-Type", _media_type_for_path(item.path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()

            with item.path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    remaining -= len(chunk)

        def _serve_frame(self, item_id: str, query: str) -> None:
            item = registry.get(item_id)
            if item is None or not item.path.exists():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Video not found"})
                return
            params = parse_qs(query)
            frame_index: int | None = None
            timestamp_s: float | None = None
            try:
                if "i" in params:
                    frame_index = int(float(params["i"][0]))
                if "t" in params:
                    timestamp_s = float(params["t"][0])
                body = _jpeg_frame_for_web_item(item, frame_index, timestamp_s)
            except Exception as exc:  # noqa: BLE001
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return
            self._send_bytes(HTTPStatus.OK, body, "image/jpeg")

    return MultiViewWebHandler


def _bind_web_server(
    host: str,
    port: int,
    handler_class: type[BaseHTTPRequestHandler],
) -> ThreadingHTTPServer:
    ports = [port] if port == 0 else list(range(port, port + 20))
    last_error: OSError | None = None
    for candidate in ports:
        try:
            return ThreadingHTTPServer((host, candidate), handler_class)
        except OSError as exc:
            last_error = exc
    raise RuntimeError(f"Could not bind web UI on {host}:{port}: {last_error}")


def run_web_gui(
    *,
    initial_paths: Sequence[Path],
    labels: Sequence[str],
    offsets: Sequence[float],
    host: str,
    port: int,
    max_videos: int,
    recursive: bool,
    sort_mode: str,
    layout: str,
    no_open: bool,
) -> int:
    registry = WebVideoRegistry()
    initial_payloads = registry.register_many(initial_paths, labels=labels, offsets=offsets)
    initial_data: dict[str, object] = {
        "videos": initial_payloads,
        "maxVideos": max_videos,
        "recursive": recursive,
        "sort": sort_mode,
        "layout": layout,
    }
    upload_root = Path(tempfile.mkdtemp(prefix="multi_view_uploads_"))
    handler_class = _make_web_handler(registry, initial_data, max_videos=max_videos, upload_root=upload_root)
    server = _bind_web_server(host, port, handler_class)
    actual_host, actual_port = server.server_address[:2]
    display_host = "127.0.0.1" if actual_host in ("0.0.0.0", "") else actual_host
    url = f"http://{display_host}:{actual_port}/"
    print(f"Multi View web UI: {url}")
    print("Press Ctrl+C in this terminal to stop the server.")
    if not no_open:
        try:
            webbrowser.open_new_tab(url)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not open browser automatically: {exc}", file=sys.stderr)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        registry.close()
        shutil.rmtree(upload_root, ignore_errors=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    sources: list[VideoSource] = []
    try:
        max_videos = int(args.max_videos)
        if args.ui == "web" and not args.contact_sheet:
            initial_paths: list[Path] = []
            if args.inputs:
                initial_paths = _filter_video_paths(
                    _resolve_input_paths(
                        args.inputs,
                        recursive=bool(args.recursive),
                        max_videos=max_videos,
                        sort_mode=str(args.sort),
                        no_picker=True,
                    )
                )
                if not initial_paths:
                    raise ValueError("No supported video files were found in the provided inputs.")
            labels = _labels_for_paths(args.labels, initial_paths)
            offsets = _parse_offsets(args.offsets, len(initial_paths))
            return run_web_gui(
                initial_paths=initial_paths,
                labels=labels,
                offsets=offsets,
                host=str(args.host),
                port=int(args.port),
                max_videos=max_videos,
                recursive=bool(args.recursive),
                sort_mode=str(args.sort),
                layout=str(args.layout),
                no_open=bool(args.no_open),
            )

        canvas_size = _parse_size(str(args.canvas_size))
        input_paths = _resolve_input_paths(
            args.inputs,
            recursive=bool(args.recursive),
            max_videos=max_videos,
            sort_mode=str(args.sort),
            no_picker=bool(args.no_picker),
        )
        _parse_layout(str(args.layout), len(input_paths))
        labels = _labels_for_paths(args.labels, input_paths)
        offsets = _parse_offsets(args.offsets, len(input_paths))
        sources = _open_videos(input_paths, labels, offsets)

        config = PlayerConfig(
            window_name=str(args.window_name),
            canvas_size=canvas_size,
            layout=str(args.layout),
            seek_step_s=float(args.seek_step),
            snapshot_dir=args.snapshot_dir.expanduser(),
            loop=bool(args.loop),
            show_help=not bool(args.no_help),
        )
        if args.contact_sheet:
            timestamp_s = float(args.at if args.at is not None else args.start_time)
            try:
                output_path = render_contact_sheet(
                    sources,
                    output_path=args.contact_sheet,
                    timestamp_s=timestamp_s,
                    config=config,
                )
            finally:
                _close_sources(sources)
            print(f"Wrote contact sheet: {output_path}")
            return 0

        return run_player(
            sources,
            window_name=config.window_name,
            speed=float(args.speed),
            start_time_s=float(args.start_time),
            start_paused=bool(args.start_paused),
            config=config,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        _close_sources(sources)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
