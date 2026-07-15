from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.m4v")
DEFAULT_WINDOW_NAME = "Wrist camera robot-body mask painter"
MODE_PAINT = "paint"
MODE_ERASE = "erase"


@dataclass
class SaveResult:
    mask_path: Path
    metadata_path: Path
    frame_path: Path | None = None


@dataclass
class LoadRequest:
    mask_path: Path | None = None
    video_path: Path | None = None
    frame_index: int | None = None
    output_name: str = ""
    metadata_path: Path | None = None


@dataclass
class PaintState:
    cv2: Any
    video_path: Path
    capture: Any
    frame_count: int
    fps: float
    output_dir: Path
    metadata_dir: Path
    window_name: str
    max_display_width: int
    max_display_height: int
    output_name: str = ""
    save_frame: bool = False
    frame_index: int = 0
    frame_bgr: np.ndarray | None = None
    mask: np.ndarray | None = None
    drawing: bool = False
    draw_value: int = 255
    last_point: tuple[int, int] | None = None
    mouse_point: tuple[int, int] | None = None
    display_scale: float = 1.0
    undo_stack: list[np.ndarray] = field(default_factory=list)
    last_saved: SaveResult | None = None

    def ensure_mask(self) -> np.ndarray:
        if self.frame_bgr is None:
            raise RuntimeError("No frame is loaded.")
        height, width = self.frame_bgr.shape[:2]
        if self.mask is None or self.mask.shape != (height, width):
            self.mask = np.zeros((height, width), dtype=np.uint8)
        return self.mask

    def push_undo(self) -> None:
        mask = self.ensure_mask()
        self.undo_stack.append(mask.copy())
        if len(self.undo_stack) > 40:
            del self.undo_stack[0]

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        self.mask = self.undo_stack.pop()
        return True


def repo_default_dir() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def metadata_path_string(path: Path | None) -> str | None:
    if path is None:
        return None
    relative = os.path.relpath(Path(path).expanduser().resolve(), repo_root())
    return Path(relative).as_posix()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open a saved wrist-camera video, pick one frame, paint the robot "
            "body area, and save a binary PNG mask."
        )
    )
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        help="Path to the saved real-robot wrist-camera video. If omitted, a file dialog is opened.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_default_dir(),
        help="Directory for painted mask PNG files.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=repo_default_dir().parent / "configs",
        help="Directory for JSON metadata files that describe the saved mask.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Initial frame index shown when the GUI opens.",
    )
    parser.add_argument(
        "--brush",
        type=int,
        default=24,
        help="Initial brush radius in pixels.",
    )
    parser.add_argument(
        "--max-display-width",
        type=int,
        default=3200,
        help=(
            "Maximum total display width for the side-by-side original/painted view; "
            "mask is still saved at original video resolution."
        ),
    )
    parser.add_argument(
        "--max-display-height",
        type=int,
        default=1800,
        help=(
            "Maximum display height for the side-by-side original/painted view; "
            "mask is still saved at original video resolution."
        ),
    )
    parser.add_argument(
        "--save-frame",
        action="store_true",
        help="Also save the selected source frame next to the mask for visual reference.",
    )
    parser.add_argument(
        "--output-name",
        default="",
        help=(
            "Custom output file stem. For example, 'right_dmtac_m_mask' saves "
            "painting/right_dmtac_m_mask.png and configs/right_dmtac_m_mask.json. "
            "Leave empty to use the video/frame-based name."
        ),
    )
    parser.add_argument(
        "--load",
        type=Path,
        default=None,
        help=(
            "Load an existing mask PNG or saved metadata JSON to continue editing. "
            "A JSON file can restore its video path and frame index."
        ),
    )
    parser.add_argument(
        "--window-name",
        default=DEFAULT_WINDOW_NAME,
        help="Tk window title.",
    )
    parser.add_argument(
        "--no-highgui-preflight",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def choose_video_with_dialog(initial_dir: Path | None = None) -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # noqa: BLE001
        print(f"tkinter file dialog is unavailable: {exc}", file=sys.stderr)
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        root.update()
        selected = filedialog.askopenfilename(
            title="Choose a saved wrist-camera video",
            initialdir=str(initial_dir or Path.cwd()),
            filetypes=[
                ("Video files", " ".join(VIDEO_EXTENSIONS)),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
    except Exception as exc:  # noqa: BLE001
        print(f"failed to open file dialog: {exc}", file=sys.stderr)
        return None
    return Path(selected) if selected else None


def open_video(cv2: Any, video_path: Path) -> tuple[Any, int, float]:
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Video path is not a file: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(
            f"Could not read frame count from video: {path}. "
            "Please use a seekable saved video file such as mp4/avi."
        )
    return capture, frame_count, fps


def read_frame(state: PaintState, frame_index: int) -> np.ndarray:
    index = int(max(0, min(int(frame_index), state.frame_count - 1)))
    state.capture.set(state.cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = state.capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to decode frame {index} from {state.video_path}")
    state.frame_index = index
    state.frame_bgr = np.asarray(frame)
    state.ensure_mask()
    return state.frame_bgr


def _sanitize_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    stem = stem.strip("._-")
    return stem or "wrist_camera_video"


def _custom_output_stem(output_name: str | None) -> str:
    raw = str(output_name or "").strip()
    if not raw:
        return ""
    name = Path(raw).name.strip()
    for suffix in (".png", ".json", ".jpg", ".jpeg"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return _sanitize_stem(name)


def _path_from_metadata(value: object, *, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [
        repo_root() / path,
        base_dir / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_request_from_path(path: Path, *, metadata_dir: Path) -> LoadRequest:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Load target does not exist: {target}")

    if target.suffix.lower() == ".json":
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Metadata JSON must contain an object: {target}")
        base_dir = target.parent
        mask_path = _path_from_metadata(data.get("mask_path"), base_dir=base_dir)
        video_path = _path_from_metadata(data.get("video_path"), base_dir=base_dir)
        frame_index = data.get("frame_index")
        output_name = str(data.get("output_name") or "")
        if not output_name and mask_path is not None:
            output_name = mask_path.stem
        return LoadRequest(
            mask_path=mask_path,
            video_path=video_path,
            frame_index=None if frame_index is None else int(frame_index),
            output_name=output_name,
            metadata_path=target,
        )

    if target.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        metadata_path = Path(metadata_dir).expanduser().resolve() / f"{target.stem}.json"
        if metadata_path.exists():
            request = load_request_from_path(metadata_path, metadata_dir=metadata_dir)
            request.mask_path = target
            if not request.output_name:
                request.output_name = target.stem
            return request
        return LoadRequest(mask_path=target, output_name=target.stem)

    raise ValueError(f"Expected a .json metadata file or image mask, got: {target}")


def load_mask_image(cv2: Any, mask_path: Path, *, expected_shape: tuple[int, int]) -> np.ndarray:
    path = Path(mask_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Mask file does not exist: {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask image: {path}")
    expected_height, expected_width = expected_shape
    if tuple(mask.shape[:2]) != (int(expected_height), int(expected_width)):
        raise ValueError(
            "Loaded mask shape does not match the selected frame: "
            f"mask={tuple(mask.shape[:2])} frame={(int(expected_height), int(expected_width))}. "
            "Load the matching video/frame metadata JSON or select the correct frame first."
        )
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def default_output_stem(video_path: Path, frame_index: int) -> str:
    stem = _sanitize_stem(Path(video_path).stem)
    return f"{stem}_frame_{int(frame_index):06d}_robot_body_mask"


def unique_artifact_paths(
    *,
    video_path: Path,
    frame_index: int,
    output_dir: Path,
    metadata_dir: Path,
    save_frame: bool,
    output_name: str | None = None,
) -> tuple[Path, Path, Path | None]:
    base = _custom_output_stem(output_name)
    if not base:
        base = default_output_stem(video_path, frame_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for suffix in ("", *(f"_{index:02d}" for index in range(1, 1000))):
        mask_path = output_dir / f"{base}{suffix}.png"
        metadata_path = metadata_dir / f"{base}{suffix}.json"
        frame_path = output_dir / f"{base}{suffix}_source.jpg" if save_frame else None
        paths = [mask_path, metadata_path]
        if frame_path is not None:
            paths.append(frame_path)
        if not any(path.exists() for path in paths):
            return mask_path, metadata_path, frame_path
    raise RuntimeError(f"Could not find an unused output name for {base}")


def save_mask_artifacts(state: PaintState) -> SaveResult:
    if state.frame_bgr is None:
        raise RuntimeError("No frame is loaded.")
    mask = state.ensure_mask()
    mask_binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    mask_path, metadata_path, frame_path = unique_artifact_paths(
        video_path=state.video_path,
        frame_index=state.frame_index,
        output_dir=state.output_dir,
        metadata_dir=state.metadata_dir,
        save_frame=state.save_frame,
        output_name=state.output_name,
    )
    if not bool(state.cv2.imwrite(str(mask_path), mask_binary)):
        raise RuntimeError(f"Failed to write mask: {mask_path}")
    if frame_path is not None:
        if not bool(state.cv2.imwrite(str(frame_path), state.frame_bgr)):
            raise RuntimeError(f"Failed to write source frame: {frame_path}")

    height, width = mask_binary.shape
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "path_base": "repo_root",
        "video_path": metadata_path_string(state.video_path),
        "video_name": state.video_path.name,
        "frame_index": int(state.frame_index),
        "frame_count": int(state.frame_count),
        "fps": float(state.fps),
        "frame_shape_hw": [int(height), int(width)],
        "mask_path": metadata_path_string(mask_path),
        "source_frame_path": metadata_path_string(frame_path),
        "output_name": _custom_output_stem(state.output_name),
        "mask_semantics": "uint8 PNG; 255 means painted robot body/self area, 0 means background",
        "painted_pixels": int(np.count_nonzero(mask_binary)),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result = SaveResult(mask_path=mask_path, metadata_path=metadata_path, frame_path=frame_path)
    state.last_saved = result
    return result


def draw_mask_stroke(
    *,
    cv2: Any,
    mask: np.ndarray,
    start: tuple[int, int] | None,
    end: tuple[int, int],
    radius: int,
    value: int,
) -> None:
    radius = max(1, int(radius))
    color = int(value)
    if start is None:
        cv2.circle(mask, end, radius, color, thickness=-1, lineType=cv2.LINE_AA)
        return
    cv2.line(mask, start, end, color, thickness=radius * 2, lineType=cv2.LINE_AA)
    cv2.circle(mask, end, radius, color, thickness=-1, lineType=cv2.LINE_AA)


def preview_images_rgb(
    cv2: Any,
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float,
    max_total_width: int,
    max_height: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    frame = np.asarray(frame_bgr, dtype=np.uint8)
    mask_array = np.asarray(mask)
    height, width = frame.shape[:2]
    scale = min(
        float(max_total_width) / max(1, width * 2),
        float(max_height) / max(1, height),
    )
    scale = max(0.01, scale)
    opacity = max(0.0, min(1.0, float(alpha)))

    overlay = frame.copy()
    overlay[mask_array > 0] = (0, 0, 255)
    painted = cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0.0)

    original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    painted_rgb = cv2.cvtColor(painted, cv2.COLOR_BGR2RGB)
    if abs(scale - 1.0) > 1e-6:
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        original_rgb = cv2.resize(original_rgb, size, interpolation=cv2.INTER_AREA)
        painted_rgb = cv2.resize(painted_rgb, size, interpolation=cv2.INTER_AREA)
    return original_rgb, painted_rgb, scale


class MaskPainterApp:
    def __init__(
        self,
        *,
        state: PaintState,
        start_frame: int,
        brush: int,
        initial_load: LoadRequest | None,
        tk: Any,
        ttk: Any,
        filedialog: Any,
        messagebox: Any,
        image_module: Any,
        image_tk_module: Any,
    ) -> None:
        self.state = state
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.Image = image_module
        self.ImageTk = image_tk_module
        self.root = tk.Tk()
        self.root.title(state.window_name)

        self.mode_var = tk.StringVar(value=MODE_PAINT)
        self.brush_var = tk.IntVar(value=max(1, min(200, int(brush))))
        self.brush_text_var = tk.StringVar()
        self.alpha_var = tk.IntVar(value=45)
        self.frame_var = tk.IntVar(value=0)
        self.save_frame_var = tk.BooleanVar(value=bool(state.save_frame))
        self.output_name_var = tk.StringVar(value=str(state.output_name or ""))
        self.status_var = tk.StringVar(value="")
        self.video_var = tk.StringVar(value="")
        self.original_photo: Any | None = None
        self.painted_photo: Any | None = None
        self.canvas_width = 1
        self.canvas_height = 1
        self.canvas_offsets: dict[Any, tuple[int, int]] = {}
        self.render_after_id: str | None = None

        read_frame(state, int(max(0, min(int(start_frame), state.frame_count - 1))))
        self.frame_var.set(state.frame_index)
        self._sync_brush_label()
        self._build_widgets()
        self._install_bindings()
        self._sync_video_label()
        if initial_load is not None and initial_load.mask_path is not None:
            self.apply_load_request(initial_load)
        self._render()

    def _build_widgets(self) -> None:
        tk = self.tk
        ttk = self.ttk

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=(10, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(2, weight=1)

        ttk.Button(top, text="选择视频", command=self.choose_video).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(top, text="加载 Mask/配置", command=self.load_mask_or_config).grid(row=0, column=1, padx=(0, 8))
        ttk.Label(top, textvariable=self.video_var).grid(row=0, column=2, sticky="w")
        ttk.Button(top, text="保存 Mask", command=self.save).grid(row=0, column=3, padx=(8, 0))
        ttk.Button(top, text="退出", command=self.root.destroy).grid(row=0, column=4, padx=(8, 0))
        ttk.Label(top, text="保存文件名(留空自动)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        output_entry = ttk.Entry(top, textvariable=self.output_name_var)
        output_entry.grid(row=1, column=1, columnspan=4, sticky="ew", pady=(8, 0))

        self.image_area = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        self.image_area.grid(row=1, column=0, sticky="nsew")
        self.image_area.columnconfigure(0, weight=1)
        self.image_area.columnconfigure(1, weight=1)
        self.image_area.rowconfigure(1, weight=1)

        ttk.Label(self.image_area, text="原图").grid(row=0, column=0, sticky="w")
        ttk.Label(self.image_area, text="涂抹预览").grid(row=0, column=1, sticky="w")
        initial_panel_width = max(320, min(800, int(self.state.max_display_width) // 2))
        initial_panel_height = max(240, min(520, int(self.state.max_display_height)))
        self.original_canvas = tk.Canvas(
            self.image_area,
            bg="#151515",
            width=initial_panel_width,
            height=initial_panel_height,
            highlightthickness=1,
            highlightbackground="#777",
        )
        self.painted_canvas = tk.Canvas(
            self.image_area,
            bg="#151515",
            width=initial_panel_width,
            height=initial_panel_height,
            highlightthickness=1,
            highlightbackground="#777",
        )
        self.original_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.painted_canvas.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        controls = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)
        controls.columnconfigure(5, weight=1)

        ttk.Label(controls, text="帧").grid(row=0, column=0, sticky="w")
        self.frame_scale = tk.Scale(
            controls,
            from_=0,
            to=max(0, self.state.frame_count - 1),
            orient=tk.HORIZONTAL,
            variable=self.frame_var,
            showvalue=True,
            resolution=1,
            command=self.on_frame_change,
        )
        self.frame_scale.grid(row=0, column=1, columnspan=5, sticky="ew", padx=(8, 0))

        mode_box = ttk.LabelFrame(controls, text="模式")
        mode_box.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Radiobutton(mode_box, text="涂抹", value=MODE_PAINT, variable=self.mode_var, command=self._render).grid(
            row=0, column=0, padx=(8, 6), pady=4
        )
        ttk.Radiobutton(mode_box, text="擦除", value=MODE_ERASE, variable=self.mode_var, command=self._render).grid(
            row=0, column=1, padx=(0, 8), pady=4
        )

        ttk.Button(controls, text="撤回", command=self.undo).grid(row=1, column=2, padx=(8, 0), pady=(8, 0), sticky="ew")
        ttk.Button(controls, text="清空", command=self.clear).grid(row=1, column=3, padx=(8, 0), pady=(8, 0), sticky="ew")
        ttk.Button(controls, text="反选", command=self.invert).grid(row=1, column=4, padx=(8, 0), pady=(8, 0), sticky="ew")

        ttk.Label(controls, text="笔刷").grid(row=2, column=0, sticky="w", pady=(8, 0))
        brush_box = ttk.Frame(controls)
        brush_box.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(8, 0))
        brush_box.columnconfigure(2, weight=1)
        ttk.Button(brush_box, text="-", width=3, command=lambda: self.adjust_brush(-2)).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(brush_box, textvariable=self.brush_text_var, width=8).grid(
            row=0, column=1, sticky="w", padx=(8, 8)
        )
        brush_scale = tk.Scale(
            brush_box,
            from_=1,
            to=200,
            orient=tk.HORIZONTAL,
            variable=self.brush_var,
            showvalue=False,
            resolution=1,
            command=self.on_brush_change,
        )
        brush_scale.grid(row=0, column=2, sticky="ew")
        ttk.Button(brush_box, text="+", width=3, command=lambda: self.adjust_brush(2)).grid(
            row=0, column=3, sticky="e", padx=(8, 0)
        )

        ttk.Label(controls, text="透明度").grid(row=2, column=3, sticky="w", padx=(8, 0), pady=(8, 0))
        alpha_scale = tk.Scale(
            controls,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.alpha_var,
            showvalue=True,
            resolution=1,
            command=lambda _value: self._render(),
        )
        alpha_scale.grid(row=2, column=4, columnspan=2, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Checkbutton(controls, text="保存源帧", variable=self.save_frame_var).grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Label(controls, textvariable=self.status_var).grid(
            row=3, column=1, columnspan=5, sticky="w", padx=(8, 0), pady=(8, 0)
        )

    def _install_bindings(self) -> None:
        for canvas in (self.original_canvas, self.painted_canvas):
            canvas.bind("<ButtonPress-1>", lambda event: self.begin_stroke(event, force_erase=False))
            canvas.bind("<B1-Motion>", self.continue_stroke)
            canvas.bind("<ButtonRelease-1>", self.end_stroke)
            canvas.bind("<ButtonPress-3>", lambda event: self.begin_stroke(event, force_erase=True))
            canvas.bind("<B3-Motion>", self.continue_stroke)
            canvas.bind("<ButtonRelease-3>", self.end_stroke)
            canvas.bind("<Motion>", self.update_mouse_preview)
            canvas.bind("<Leave>", self.clear_mouse_preview)
            canvas.bind("<Configure>", self._schedule_render)

        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("z", lambda _event: self.undo())
        self.root.bind("u", lambda _event: self.undo())
        self.root.bind("p", lambda _event: self.set_mode(MODE_PAINT))
        self.root.bind("e", lambda _event: self.set_mode(MODE_ERASE))
        self.root.bind("s", lambda _event: self.save())
        self.root.bind("c", lambda _event: self.clear())
        self.root.bind("i", lambda _event: self.invert())
        self.root.bind("q", lambda _event: self.root.destroy())
        self.root.bind("<Escape>", lambda _event: self.root.destroy())
        self.root.bind("[", lambda _event: self.adjust_brush(-2))
        self.root.bind("]", lambda _event: self.adjust_brush(2))

    def run(self) -> None:
        self.root.mainloop()

    def _schedule_render(self, _event: Any | None = None) -> None:
        if self.render_after_id is not None:
            self.root.after_cancel(self.render_after_id)
        self.render_after_id = self.root.after(40, self._render)

    def choose_video(self) -> None:
        selected = self.filedialog.askopenfilename(
            title="Choose a saved wrist-camera video",
            initialdir=str(self.state.video_path.parent),
            filetypes=[
                ("Video files", " ".join(VIDEO_EXTENSIONS)),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        try:
            capture, frame_count, fps = open_video(self.state.cv2, Path(selected))
            self.state.capture.release()
            self.state.video_path = Path(selected).expanduser().resolve()
            self.state.capture = capture
            self.state.frame_count = int(frame_count)
            self.state.fps = float(fps)
            self.state.frame_index = 0
            self.state.frame_bgr = None
            self.state.mask = None
            self.state.undo_stack.clear()
            read_frame(self.state, 0)
            self.frame_scale.configure(to=max(0, self.state.frame_count - 1))
            self.frame_var.set(0)
            self._sync_video_label()
            self.set_status("loaded video")
            self._render()
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Failed to open video", str(exc), parent=self.root)

    def load_mask_or_config(self) -> None:
        selected = self.filedialog.askopenfilename(
            title="Load an existing mask PNG or metadata JSON",
            initialdir=str(self.state.metadata_dir),
            filetypes=[
                ("Mask metadata or image", "*.json *.png *.jpg *.jpeg"),
                ("Metadata JSON", "*.json"),
                ("Mask images", "*.png *.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        try:
            request = load_request_from_path(Path(selected), metadata_dir=self.state.metadata_dir)
            self.apply_load_request(request)
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Failed to load mask", str(exc), parent=self.root)

    def apply_load_request(self, request: LoadRequest) -> None:
        if request.video_path is not None:
            video_path = Path(request.video_path).expanduser().resolve()
            if video_path != self.state.video_path:
                capture, frame_count, fps = open_video(self.state.cv2, video_path)
                self.state.capture.release()
                self.state.video_path = video_path
                self.state.capture = capture
                self.state.frame_count = int(frame_count)
                self.state.fps = float(fps)
                self.frame_scale.configure(to=max(0, self.state.frame_count - 1))

        frame_index = self.state.frame_index
        if request.frame_index is not None:
            frame_index = int(max(0, min(int(request.frame_index), self.state.frame_count - 1)))
        read_frame(self.state, frame_index)
        self.frame_var.set(self.state.frame_index)

        if request.mask_path is None:
            raise ValueError("Loaded metadata does not include mask_path.")
        self.state.mask = load_mask_image(
            self.state.cv2,
            request.mask_path,
            expected_shape=self.state.frame_bgr.shape[:2],
        )
        self.state.undo_stack.clear()
        self.state.mouse_point = None
        self.output_name_var.set(request.output_name or Path(request.mask_path).stem)
        self.state.output_name = str(self.output_name_var.get()).strip()
        self._sync_video_label()
        self.set_status(f"loaded mask {Path(request.mask_path).name}")
        self._render()

    def on_frame_change(self, value: str) -> None:
        if self.state.drawing:
            return
        index = int(round(float(value)))
        if index == self.state.frame_index:
            return
        try:
            read_frame(self.state, index)
            self.frame_var.set(self.state.frame_index)
            self.set_status(f"frame {self.state.frame_index}/{self.state.frame_count - 1}")
            self._render()
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Failed to read frame", str(exc), parent=self.root)

    def begin_stroke(self, event: Any, *, force_erase: bool) -> None:
        point = self.event_to_image_point(event)
        if point is None:
            return
        self.state.push_undo()
        self.state.drawing = True
        self.state.draw_value = 0 if force_erase or self.mode_var.get() == MODE_ERASE else 255
        self.state.last_point = point
        self.state.mouse_point = point
        self.draw_to(point)

    def continue_stroke(self, event: Any) -> None:
        point = self.event_to_image_point(event)
        if point is None:
            return
        if self.state.drawing:
            self.draw_to(point)
        else:
            self.state.mouse_point = point
            self._render()

    def end_stroke(self, event: Any) -> None:
        if self.state.drawing:
            point = self.event_to_image_point(event)
            if point is not None:
                self.draw_to(point)
        self.state.drawing = False
        self.state.last_point = None

    def draw_to(self, point: tuple[int, int]) -> None:
        draw_mask_stroke(
            cv2=self.state.cv2,
            mask=self.state.ensure_mask(),
            start=self.state.last_point,
            end=point,
            radius=int(self.brush_var.get()),
            value=int(self.state.draw_value),
        )
        self.state.last_point = point
        self.state.mouse_point = point
        self._render()

    def update_mouse_preview(self, event: Any) -> None:
        point = self.event_to_image_point(event)
        if point == self.state.mouse_point:
            return
        self.state.mouse_point = point
        self._render()

    def clear_mouse_preview(self, _event: Any) -> None:
        if self.state.mouse_point is None:
            return
        self.state.mouse_point = None
        self._render()

    def event_to_image_point(self, event: Any) -> tuple[int, int] | None:
        if self.state.frame_bgr is None:
            return None
        height, width = self.state.frame_bgr.shape[:2]
        scale = max(1e-6, float(self.state.display_scale))
        offset_x, offset_y = self.canvas_offsets.get(event.widget, (0, 0))
        local_x = int(event.x) - int(offset_x)
        local_y = int(event.y) - int(offset_y)
        if local_x < 0 or local_y < 0:
            return None
        image_x = int(round(float(local_x) / scale))
        image_y = int(round(float(local_y) / scale))
        if image_x < 0 or image_y < 0 or image_x >= width or image_y >= height:
            return None
        return image_x, image_y

    def undo(self) -> None:
        if self.state.undo():
            self.set_status("undo")
            self._render()
        else:
            self.set_status("nothing to undo")

    def clear(self) -> None:
        self.state.push_undo()
        self.state.ensure_mask()[:, :] = 0
        self.set_status("cleared mask")
        self._render()

    def invert(self) -> None:
        self.state.push_undo()
        mask = self.state.ensure_mask()
        mask[:, :] = np.where(mask > 0, 0, 255).astype(np.uint8)
        self.set_status("inverted mask")
        self._render()

    def save(self) -> None:
        try:
            self.state.save_frame = bool(self.save_frame_var.get())
            self.state.output_name = str(self.output_name_var.get()).strip()
            result = save_mask_artifacts(self.state)
            self.set_status(f"saved {result.mask_path.name}")
            self.messagebox.showinfo(
                "Saved",
                f"Mask:\n{result.mask_path}\n\nMetadata:\n{result.metadata_path}",
                parent=self.root,
            )
        except Exception as exc:  # noqa: BLE001
            self.messagebox.showerror("Failed to save mask", str(exc), parent=self.root)

    def set_mode(self, mode: str) -> None:
        self.mode_var.set(mode)
        self._render()

    def on_brush_change(self, _value: str | None = None) -> None:
        self._sync_brush_label()
        self._render()

    def adjust_brush(self, delta: int) -> None:
        current = int(self.brush_var.get())
        self.brush_var.set(max(1, min(200, current + int(delta))))
        self._sync_brush_label()
        self._render()

    def set_status(self, text: str) -> None:
        self.status_var.set(str(text))

    def _sync_brush_label(self) -> None:
        self.brush_text_var.set(f"{int(self.brush_var.get())} px")

    def _sync_video_label(self) -> None:
        self.video_var.set(
            f"{self.state.video_path.name}  "
            f"frames={self.state.frame_count} fps={self.state.fps:.2f}"
        )

    def _preview_limits(self) -> tuple[int, int]:
        left_width = int(self.original_canvas.winfo_width())
        right_width = int(self.painted_canvas.winfo_width())
        left_height = int(self.original_canvas.winfo_height())
        right_height = int(self.painted_canvas.winfo_height())

        if left_width <= 1 or right_width <= 1:
            total_width = int(self.state.max_display_width)
        else:
            total_width = min(
                int(self.state.max_display_width),
                max(2, min(left_width, right_width) * 2),
            )

        if left_height <= 1 or right_height <= 1:
            max_height = int(self.state.max_display_height)
        else:
            max_height = min(
                int(self.state.max_display_height),
                max(1, min(left_height, right_height)),
            )
        return max(2, total_width), max(1, max_height)

    def _render(self) -> None:
        self.render_after_id = None
        if self.state.frame_bgr is None:
            return
        alpha = float(self.alpha_var.get()) / 100.0
        max_total_width, max_height = self._preview_limits()
        original_rgb, painted_rgb, scale = preview_images_rgb(
            self.state.cv2,
            self.state.frame_bgr,
            self.state.ensure_mask(),
            alpha=alpha,
            max_total_width=max_total_width,
            max_height=max_height,
        )
        self.state.display_scale = scale
        self.canvas_width = int(original_rgb.shape[1])
        self.canvas_height = int(original_rgb.shape[0])

        self.original_photo = self.ImageTk.PhotoImage(self.Image.fromarray(original_rgb))
        self.painted_photo = self.ImageTk.PhotoImage(self.Image.fromarray(painted_rgb))
        self._draw_canvas(self.original_canvas, self.original_photo)
        self._draw_canvas(self.painted_canvas, self.painted_photo)

    def _draw_canvas(self, canvas: Any, photo: Any) -> None:
        canvas.delete("all")
        available_width = int(canvas.winfo_width())
        available_height = int(canvas.winfo_height())
        if available_width <= 1:
            available_width = self.canvas_width
        if available_height <= 1:
            available_height = self.canvas_height
        image_width = int(photo.width())
        image_height = int(photo.height())
        offset_x = max(0, (available_width - image_width) // 2)
        offset_y = max(0, (available_height - image_height) // 2)
        self.canvas_offsets[canvas] = (offset_x, offset_y)
        canvas.create_image(offset_x, offset_y, anchor="nw", image=photo)
        if self.state.mouse_point is None:
            return
        radius = max(1, int(round(int(self.brush_var.get()) * self.state.display_scale)))
        x = int(round(self.state.mouse_point[0] * self.state.display_scale)) + offset_x
        y = int(round(self.state.mouse_point[1] * self.state.display_scale)) + offset_y
        color = "#00b050" if self.mode_var.get() == MODE_PAINT else "#ff9900"
        canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=color, width=2)


def run_gui(
    state: PaintState,
    start_frame: int,
    brush: int,
    initial_load: LoadRequest | None = None,
) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from PIL import Image, ImageTk
    except ImportError as exc:
        raise RuntimeError("Tkinter and Pillow are required for the GUI.") from exc

    app = MaskPainterApp(
        state=state,
        start_frame=int(start_frame),
        brush=int(brush),
        initial_load=initial_load,
        tk=tk,
        ttk=ttk,
        filedialog=filedialog,
        messagebox=messagebox,
        image_module=Image,
        image_tk_module=ImageTk,
    )
    app.run()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata_dir = Path(args.metadata_dir).expanduser().resolve()
    initial_load = None
    if args.load is not None:
        try:
            initial_load = load_request_from_path(Path(args.load), metadata_dir=metadata_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load --load target: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2

    video_path = args.video
    if video_path is None and initial_load is not None and initial_load.video_path is not None:
        video_path = initial_load.video_path
    if video_path is None:
        video_path = choose_video_with_dialog(Path.cwd())
    if video_path is None:
        print("No video selected.", file=sys.stderr)
        return 2

    try:
        import cv2
    except ImportError as exc:
        print(f"OpenCV (cv2) is required: {exc}", file=sys.stderr)
        return 2

    capture = None
    try:
        capture, frame_count, fps = open_video(cv2, Path(video_path))
        output_name = str(args.output_name)
        if not output_name and initial_load is not None:
            output_name = str(initial_load.output_name)
        start_frame = int(args.start_frame)
        if initial_load is not None and initial_load.frame_index is not None:
            start_frame = int(initial_load.frame_index)
        state = PaintState(
            cv2=cv2,
            video_path=Path(video_path).expanduser().resolve(),
            capture=capture,
            frame_count=int(frame_count),
            fps=float(fps),
            output_dir=Path(args.output_dir).expanduser().resolve(),
            metadata_dir=metadata_dir,
            window_name=str(args.window_name),
            max_display_width=int(args.max_display_width),
            max_display_height=int(args.max_display_height),
            output_name=output_name,
            save_frame=bool(args.save_frame),
        )
        run_gui(
            state,
            start_frame=start_frame,
            brush=int(args.brush),
            initial_load=initial_load,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Mask painter failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if capture is not None:
            capture.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
