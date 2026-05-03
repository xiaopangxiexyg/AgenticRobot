#!/usr/bin/env python3
"""Interactive LeRobot episode player with video and tail-dimension panels."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from TestScripts.tools.lerobot_dataset_tool import EpisodeData, LeRobotDataset


@dataclass
class Button:
    label: str
    rect: tuple[int, int, int, int]
    action: str


def resize_to_fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = image.shape[:2]
    if source_h == 0 or source_w == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / source_w, height / source_h)
    resized = cv2.resize(
        image,
        (max(1, int(source_w * scale)), max(1, int(source_h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), (12, 14, 18), dtype=np.uint8)
    offset_x = (width - resized.shape[1]) // 2
    offset_y = (height - resized.shape[0]) // 2
    canvas[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized
    return canvas


def draw_panel(canvas: np.ndarray, x: int, y: int, width: int, height: int, title: str) -> tuple[int, int, int, int]:
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (78, 84, 92), 1)
    cv2.putText(
        canvas,
        title,
        (x + 16, y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (238, 242, 247),
        2,
        cv2.LINE_AA,
    )
    return x + 16, y + 42, width - 32, height - 58


def initial_rotation_steps(name: str) -> int:
    mapping = {"none": 0, "cw": 1, "ccw": -1}
    return mapping[name]


class EpisodePlayer:
    def __init__(
        self,
        dataset_root: str,
        episode_index: int,
        video_key: str,
        initial_rotation: str,
        start_paused: bool,
    ) -> None:
        self.dataset = LeRobotDataset.open(dataset_root)
        self.episode = self.dataset.load_episode(episode_index)
        self.video_key = video_key or (self.dataset.video_keys[0] if self.dataset.video_keys else "")
        if not self.video_key:
            raise ValueError(f"No video feature found in {self.dataset.root}")

        self.window_name = "LeRobot Episode Player"
        self.trackbar_name = "row"
        self.capture = cv2.VideoCapture(str(self.dataset.video_path_for_episode(self.episode.episode_index, self.video_key)))
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open video for {self.video_key}")
        self.capture_next_frame_index: int | None = None

        self.rotation_steps = initial_rotation_steps(initial_rotation)
        self.is_playing = not start_paused
        self.current_row = 0
        self.max_row = max(self.episode.length - 1, 0)
        self.syncing_trackbar = False
        self.last_tick = time.perf_counter()
        self.cached_row = -1
        self.cached_frame: np.ndarray | None = None
        self.buttons: list[Button] = []
        self.window_ready = False
        self.header_height = 110
        self.canvas_size = (980, 1860)

        self.display_indices = self._select_display_indices()
        self.display_mode = "gripper-tail-2" if len(self.display_indices) == 2 else "tail-6"
        self.display_labels = self._build_display_labels()
        self.action_values = self.episode.action[:, self.display_indices]
        self.state_values = self.episode.state[:, self.display_indices]
        self.diff_values = self.state_values - self.action_values
        self.action_scales = self._compute_scales(self.action_values)
        self.state_scales = self._compute_scales(self.state_values)
        self.diff_scales = self._compute_scales(self.diff_values)

    def close(self) -> None:
        self.capture.release()
        cv2.destroyAllWindows()

    def _select_display_indices(self) -> list[int]:
        total_dim = int(self.episode.action.shape[1])
        if total_dim <= 6:
            return list(range(total_dim))
        if total_dim <= 8:
            return list(range(6, total_dim))
        return list(range(total_dim - 6, total_dim))

    def _build_display_labels(self) -> list[str]:
        action_names = self.dataset.action_names or [f"action_{index}" for index in range(self.episode.action.shape[1])]
        labels = []
        for index in self.display_indices:
            labels.append(f"d{index}: {action_names[index]}")
        return labels

    def _compute_scales(self, values: np.ndarray) -> np.ndarray:
        scales = np.max(np.abs(values), axis=0)
        return np.maximum(scales, 1e-6)

    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        steps = self.rotation_steps % 4
        if steps == 1:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if steps == 2:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if steps == 3:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _read_frame_for_row(self, row: int) -> np.ndarray:
        row = int(np.clip(row, 0, self.max_row))
        if self.cached_row == row and self.cached_frame is not None:
            return self.cached_frame.copy()

        video_frame = int(self.episode.frame_index[row])
        if self.capture_next_frame_index != video_frame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
            self.capture_next_frame_index = video_frame
        ok, frame = self.capture.read()
        if not ok or frame is None:
            frame = np.zeros((256, 256, 3), dtype=np.uint8)
            cv2.putText(frame, "frame missing", (24, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            self.capture_next_frame_index = None
        else:
            self.capture_next_frame_index = video_frame + 1
        frame = self._rotate_frame(frame)
        self.cached_row = row
        self.cached_frame = frame.copy()
        return frame

    def _sync_trackbar(self) -> None:
        if not self.window_ready:
            return
        self.syncing_trackbar = True
        cv2.setTrackbarPos(self.trackbar_name, self.window_name, int(self.current_row))
        self.syncing_trackbar = False

    def _set_row(self, row: int, from_trackbar: bool = False) -> None:
        self.current_row = int(np.clip(row, 0, self.max_row))
        self.cached_row = -1
        self.cached_frame = None
        if not from_trackbar:
            self._sync_trackbar()

    def _toggle_play(self) -> None:
        self.is_playing = not self.is_playing
        self.last_tick = time.perf_counter()

    def _step(self, delta: int) -> None:
        self._set_row(self.current_row + delta)
        self.is_playing = False

    def _rotate(self, direction: int) -> None:
        self.rotation_steps = (self.rotation_steps + direction) % 4
        self.cached_row = -1
        self.cached_frame = None

    def _rotation_label(self) -> str:
        steps = self.rotation_steps % 4
        mapping = {0: "0deg", 1: "90deg cw", 2: "180deg", 3: "90deg ccw"}
        return mapping[steps]

    def _on_trackbar(self, value: int) -> None:
        if self.syncing_trackbar:
            return
        self._set_row(value, from_trackbar=True)
        self.is_playing = False

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for button in self.buttons:
            bx, by, bw, bh = button.rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                if button.action == "toggle":
                    self._toggle_play()
                elif button.action == "prev":
                    self._step(-1)
                elif button.action == "next":
                    self._step(1)
                elif button.action == "ccw":
                    self._rotate(-1)
                elif button.action == "cw":
                    self._rotate(1)
                break

    def _draw_button(self, canvas: np.ndarray, button: Button) -> None:
        x, y, width, height = button.rect
        fill = (52, 58, 66)
        if button.action == "toggle":
            fill = (44, 90, 64) if self.is_playing else (90, 64, 44)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), fill, -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (112, 118, 126), 1)
        text_size, _ = cv2.getTextSize(button.label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2
        cv2.putText(canvas, button.label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 244, 248), 1, cv2.LINE_AA)

    def _build_header(self, canvas: np.ndarray) -> None:
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], self.header_height), (18, 20, 25), -1)
        cv2.line(canvas, (0, self.header_height), (canvas.shape[1], self.header_height), (70, 76, 84), 1)

        title = f"{self.dataset.dataset_name} | episode {self.episode.episode_index:06d}"
        subtitle = (
            f"row {self.current_row}/{self.max_row}  "
            f"video_frame {int(self.episode.frame_index[self.current_row]) if self.episode.length else 0}  "
            f"time {float(self.episode.timestamp[self.current_row]) if self.episode.length else 0.0:.2f}s  "
            f"rotation {self._rotation_label()}  "
            f"mode {self.display_mode}"
        )
        task_text = ", ".join(self.episode.tasks) if self.episode.tasks else "n/a"
        help_text = "space: play/pause | a/d: prev/next | z/x: ccw/cw | q: quit"

        cv2.putText(canvas, title, (28, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (244, 248, 252), 2, cv2.LINE_AA)
        cv2.putText(canvas, subtitle, (28, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (192, 198, 206), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"task: {task_text}", (28, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (192, 198, 206), 1, cv2.LINE_AA)
        cv2.putText(canvas, help_text, (28, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (156, 164, 172), 1, cv2.LINE_AA)

        button_y = 28
        button_w = 110
        button_h = 34
        gap = 10
        labels = [
            ("Play" if not self.is_playing else "Pause", "toggle"),
            ("Prev", "prev"),
            ("Next", "next"),
            ("CCW", "ccw"),
            ("CW", "cw"),
        ]
        self.buttons = []
        start_x = canvas.shape[1] - 28 - len(labels) * button_w - (len(labels) - 1) * gap
        for idx, (label, action) in enumerate(labels):
            rect = (start_x + idx * (button_w + gap), button_y, button_w, button_h)
            button = Button(label=label, rect=rect, action=action)
            self.buttons.append(button)
            self._draw_button(canvas, button)

    def _draw_video_panel(self, canvas: np.ndarray, x: int, y: int, width: int, height: int) -> None:
        inner_x, inner_y, inner_w, inner_h = draw_panel(canvas, x, y, width, height, "observation.camera_0.rgb")
        frame = self._read_frame_for_row(self.current_row)
        fitted = resize_to_fit(frame, inner_w, inner_h)
        canvas[inner_y : inner_y + inner_h, inner_x : inner_x + inner_w] = fitted
        cv2.rectangle(canvas, (inner_x, inner_y), (inner_x + inner_w, inner_y + inner_h), (84, 90, 98), 1)

    def _draw_value_panel(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        title: str,
        labels: Sequence[str],
        all_values: np.ndarray,
        scales: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        inner_x, inner_y, inner_w, inner_h = draw_panel(canvas, x, y, width, height, title)
        card_gap = 10
        count = max(len(labels), 1)
        card_h = (inner_h - (count - 1) * card_gap) // count
        current_values = all_values[self.current_row]

        for idx, label in enumerate(labels):
            card_y = inner_y + idx * (card_h + card_gap)
            cv2.rectangle(canvas, (inner_x, card_y), (inner_x + inner_w, card_y + card_h), (30, 34, 40), -1)
            cv2.rectangle(canvas, (inner_x, card_y), (inner_x + inner_w, card_y + card_h), (70, 76, 84), 1)

            value = float(current_values[idx])
            scale = float(scales[idx])
            scale = max(scale, 1e-6)
            norm = float(np.clip(value / scale, -1.0, 1.0))

            cv2.putText(canvas, label, (inner_x + 12, card_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (228, 232, 238), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"{value:+.4f}",
                (inner_x + 12, card_y + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"span +/-{scale:.3f}",
                (inner_x + 12, card_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (156, 164, 172),
                1,
                cv2.LINE_AA,
            )

            bar_x = inner_x + 12
            bar_y = card_y + card_h - 30
            bar_w = inner_w - 24
            bar_h = 14
            center_x = bar_x + bar_w // 2
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (18, 20, 24), -1)
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (86, 92, 98), 1)
            cv2.line(canvas, (center_x, bar_y - 3), (center_x, bar_y + bar_h + 3), (120, 126, 134), 1)

            if norm >= 0:
                fill_x0 = center_x
                fill_x1 = center_x + int((bar_w // 2 - 2) * norm)
            else:
                fill_x0 = center_x + int((bar_w // 2 - 2) * norm)
                fill_x1 = center_x
            cv2.rectangle(canvas, (fill_x0, bar_y + 2), (fill_x1, bar_y + bar_h - 2), color, -1)

    def render(self, row: int | None = None) -> np.ndarray:
        if row is not None:
            self._set_row(row)

        height, width = self.canvas_size
        canvas = np.full((height, width, 3), (14, 16, 20), dtype=np.uint8)
        self._build_header(canvas)

        outer_margin = 18
        body_y = self.header_height + outer_margin
        body_h = height - body_y - outer_margin
        body_w = width - 2 * outer_margin
        video_w = int(body_w * 0.50)
        gap = 16
        right_w = body_w - video_w - gap
        column_gap = 10
        column_w = (right_w - 2 * column_gap) // 3

        self._draw_video_panel(canvas, outer_margin, body_y, video_w, body_h)

        base_x = outer_margin + video_w + gap
        self._draw_value_panel(
            canvas,
            base_x,
            body_y,
            column_w,
            body_h,
            "action",
            self.display_labels,
            self.action_values,
            self.action_scales,
            (78, 168, 255),
        )
        self._draw_value_panel(
            canvas,
            base_x + column_w + column_gap,
            body_y,
            column_w,
            body_h,
            "state",
            self.display_labels,
            self.state_values,
            self.state_scales,
            (111, 224, 155),
        )
        self._draw_value_panel(
            canvas,
            base_x + 2 * (column_w + column_gap),
            body_y,
            column_w,
            body_h,
            "state-action",
            self.display_labels,
            self.diff_values,
            self.diff_scales,
            (255, 185, 83),
        )
        return canvas

    def save_snapshot(self, output_path: str, row: int | None = None) -> Path:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        image = self.render(row=row)
        cv2.imwrite(str(output), image)
        return output

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.canvas_size[1], self.canvas_size[0])
        cv2.createTrackbar(self.trackbar_name, self.window_name, 0, self.max_row, self._on_trackbar)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        self.window_ready = True
        self._sync_trackbar()

        frame_interval = 1.0 / max(self.dataset.fps, 1)
        while True:
            now = time.perf_counter()
            if self.is_playing and self.current_row < self.max_row and (now - self.last_tick) >= frame_interval:
                self._set_row(self.current_row + 1)
                self.last_tick = now

            canvas = self.render()
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(10) & 0xFF

            if key == ord("q") or key == 27:
                break
            if key == ord(" "):
                self._toggle_play()
            elif key == ord("a"):
                self._step(-1)
            elif key == ord("d"):
                self._step(1)
            elif key == ord("z"):
                self._rotate(-1)
            elif key == ord("x"):
                self._rotate(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive LeRobot episode player.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root that contains meta/, data/, and videos/.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index to play.")
    parser.add_argument("--video-key", default="", help="Video feature key, e.g. observation.camera_0.rgb.")
    parser.add_argument("--rotate", choices=["none", "cw", "ccw"], default="none", help="Initial video rotation.")
    parser.add_argument("--start-paused", action="store_true", help="Open the player in paused mode.")
    parser.add_argument("--snapshot-output", default="", help="Optional PNG output path for headless rendering.")
    parser.add_argument("--snapshot-row", type=int, default=-1, help="Row index used when --snapshot-output is set.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    player = EpisodePlayer(
        dataset_root=args.dataset_root,
        episode_index=int(args.episode_index),
        video_key=args.video_key,
        initial_rotation=args.rotate,
        start_paused=bool(args.start_paused),
    )
    try:
        if args.snapshot_output:
            snapshot_row = None if args.snapshot_row < 0 else int(args.snapshot_row)
            output = player.save_snapshot(args.snapshot_output, row=snapshot_row)
            print(output)
            return 0
        player.run()
        return 0
    finally:
        player.close()


if __name__ == "__main__":
    raise SystemExit(main())
