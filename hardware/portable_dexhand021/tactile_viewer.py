"""Tkinter tactile force viewer for :mod:`portable_dexhand021`."""

from __future__ import annotations

import argparse
import math
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Dict, Optional

from .controller import DexHandController, FINGER_ORDER


FINGER_LABELS = {
    "th": "thumb th",
    "ff": "index ff",
    "mf": "middle mf",
    "rf": "ring rf",
    "lf": "little lf",
}


@dataclass(frozen=True)
class FingerLayout:
    x: int
    y: int
    label: str


FINGER_LAYOUTS = {
    "th": FingerLayout(135, 255, FINGER_LABELS["th"]),
    "ff": FingerLayout(285, 125, FINGER_LABELS["ff"]),
    "mf": FingerLayout(425, 85, FINGER_LABELS["mf"]),
    "rf": FingerLayout(565, 125, FINGER_LABELS["rf"]),
    "lf": FingerLayout(700, 195, FINGER_LABELS["lf"]),
}


def force_color(force_z: float, max_force: float) -> str:
    ratio = 0.0 if max_force <= 0 else min(max(force_z / max_force, 0.0), 1.0)
    red = int(45 + 210 * ratio)
    green = int(125 * (1.0 - ratio) + 35 * ratio)
    blue = int(220 * (1.0 - ratio) + 45 * ratio)
    return f"#{red:02x}{green:02x}{blue:02x}"


class TactileViewer:
    """Small tactile viewer that reads from an existing controller."""

    def __init__(
        self,
        root: tk.Tk,
        controller: DexHandController,
        *,
        rate_hz: float = 20.0,
        max_normal_force: float = 5.0,
        max_tangential_force: float = 2.0,
    ) -> None:
        self.root = root
        self.controller = controller
        self.running = True
        self.rate_var = tk.DoubleVar(value=rate_hz)
        self.max_normal_force_var = tk.DoubleVar(value=max_normal_force)
        self.max_tangential_force_var = tk.DoubleVar(value=max_tangential_force)
        self.status_var = tk.StringVar(value="not started")
        self.canvas_items = {}

        self._build_ui()
        self._schedule_update()

    def _build_ui(self) -> None:
        self.root.title(f"DexHand tactile viewer - {self.controller.hand_name}")
        self.root.geometry("980x670")
        self.root.minsize(880, 580)

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(8, weight=1)

        ttk.Label(controls, text=f"hand: {self.controller.hand_name}").grid(row=0, column=0, padx=(0, 14))
        ttk.Label(controls, text="Hz").grid(row=0, column=1, padx=(0, 4))
        ttk.Spinbox(controls, from_=1.0, to=100.0, increment=1.0, textvariable=self.rate_var, width=6).grid(
            row=0, column=2, padx=(0, 10)
        )
        ttk.Label(controls, text="Fz max N").grid(row=0, column=3, padx=(0, 4))
        ttk.Spinbox(
            controls,
            from_=0.1,
            to=50.0,
            increment=0.1,
            textvariable=self.max_normal_force_var,
            width=7,
        ).grid(row=0, column=4, padx=(0, 10))
        ttk.Label(controls, text="Fxy max N").grid(row=0, column=5, padx=(0, 4))
        ttk.Spinbox(
            controls,
            from_=0.1,
            to=50.0,
            increment=0.1,
            textvariable=self.max_tangential_force_var,
            width=7,
        ).grid(row=0, column=6, padx=(0, 10))
        ttk.Button(controls, text="pause", command=self.toggle_running).grid(row=0, column=7, padx=3)

        body = ttk.Frame(main)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(body, bg="#f7f9fb", highlightthickness=1, highlightbackground="#c7d0d9")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._draw_static_canvas()

        self.detail_text = tk.Text(body, height=24, width=42, wrap=tk.NONE)
        self.detail_text.grid(row=0, column=1, sticky="nsew")

        ttk.Label(
            main,
            text="circle size/color: local +Z force Fz; arrow: local XY force vector",
            anchor="w",
        ).grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(main, textvariable=self.status_var, anchor="w").grid(row=3, column=0, sticky="ew", pady=(6, 0))

    def _draw_static_canvas(self) -> None:
        self.canvas.create_text(28, 28, anchor="w", text="DexHand fingertips", fill="#354052", font=("", 13, "bold"))
        self.canvas.create_line(135, 405, 285, 490, 435, 510, 590, 490, 725, 410, fill="#ccd6df", width=10, smooth=True)
        for finger, layout in FINGER_LAYOUTS.items():
            self.canvas.create_text(layout.x, layout.y - 72, text=layout.label, fill="#1f2933", font=("", 10, "bold"))
            self.canvas.create_oval(
                layout.x - 48,
                layout.y - 48,
                layout.x + 48,
                layout.y + 48,
                outline="#95a3b3",
                width=2,
                fill="#e8eef5",
            )
            force = self.canvas.create_oval(
                layout.x - 10,
                layout.y - 10,
                layout.x + 10,
                layout.y + 10,
                outline="",
                fill="#2d7dd2",
            )
            arrow = self.canvas.create_line(
                layout.x,
                layout.y,
                layout.x,
                layout.y,
                fill="#172b4d",
                width=4,
                arrow=tk.LAST,
            )
            value = self.canvas.create_text(layout.x, layout.y + 70, text="Fz --\nFxy --\nprox --", fill="#354052")
            self.canvas_items[finger] = {"force": force, "arrow": arrow, "value": value}

    def toggle_running(self) -> None:
        self.running = not self.running
        self._status("running" if self.running else "paused")

    def _schedule_update(self) -> None:
        self.update_once()
        interval_ms = max(10, int(1000.0 / max(float(self.rate_var.get()), 1.0)))
        self.root.after(interval_ms, self._schedule_update)

    def update_once(self) -> None:
        if not self.running:
            return
        try:
            self._render(self.controller.read_tactile())
            self._status("updated")
        except Exception as exc:
            self._status(f"read failed: {exc}")

    def _render(self, touches: Dict[str, Dict[str, object]]) -> None:
        max_fn = max(float(self.max_normal_force_var.get()), 0.1)
        max_ft = max(float(self.max_tangential_force_var.get()), 0.1)
        for finger in FINGER_ORDER:
            self._render_finger(finger, touches.get(finger), max_fn, max_ft)
        self._render_detail_text(touches)

    def _render_finger(
        self,
        finger: str,
        touch: Optional[Dict[str, object]],
        max_fn: float,
        max_ft: float,
    ) -> None:
        layout = FINGER_LAYOUTS[finger]
        items = self.canvas_items[finger]
        if touch is None:
            self.canvas.itemconfigure(items["force"], fill="#b7c4d1")
            self.canvas.coords(items["force"], layout.x - 8, layout.y - 8, layout.x + 8, layout.y + 8)
            self.canvas.coords(items["arrow"], layout.x, layout.y, layout.x, layout.y)
            self.canvas.itemconfigure(items["value"], text="Fz --\nFxy --\nprox --")
            return

        force_x = float(touch["force_x"])
        force_y = float(touch["force_y"])
        force_z = float(touch["force_z"])
        tangential_force = math.hypot(force_x, force_y)
        proximity = int(touch["proximity"])

        ratio = min(max(force_z / max_fn, 0.0), 1.0)
        radius = 10 + 38 * math.sqrt(ratio)
        self.canvas.coords(items["force"], layout.x - radius, layout.y - radius, layout.x + radius, layout.y + radius)
        self.canvas.itemconfigure(items["force"], fill=force_color(force_z, max_fn))

        if not bool(touch.get("force_direction_valid")) or tangential_force <= 0:
            self.canvas.coords(items["arrow"], layout.x, layout.y, layout.x, layout.y)
        else:
            length = 18 + 54 * min(max(tangential_force / max_ft, 0.0), 1.0)
            end_x = layout.x + (force_x / tangential_force) * length
            end_y = layout.y - (force_y / tangential_force) * length
            self.canvas.coords(items["arrow"], layout.x, layout.y, end_x, end_y)

        self.canvas.itemconfigure(items["value"], text=f"Fz {force_z:.2f}N\nFxy {tangential_force:.2f}N\nprox {proximity}")

    def _render_detail_text(self, touches: Dict[str, Dict[str, object]]) -> None:
        lines = []
        for finger in FINGER_ORDER:
            lines.append(FINGER_LABELS[finger])
            touch = touches.get(finger)
            if touch is None:
                lines.extend(["  no data", ""])
                continue
            timestamp = touch.get("timestamp")
            age = "--" if timestamp is None else f"{max(0.0, time.time() - float(timestamp)):.3f} s"
            force_x = float(touch["force_x"])
            force_y = float(touch["force_y"])
            force_z = float(touch["force_z"])
            force_xy = math.hypot(force_x, force_y)
            lines.extend(
                [
                    f"  force_x:                 {force_x:.4f} N",
                    f"  force_y:                 {force_y:.4f} N",
                    f"  force_z:                 {force_z:.4f} N",
                    f"  force_xy:                {force_xy:.4f} N",
                    f"  force_direction_valid:   {touch['force_direction_valid']}",
                    f"  force_z_delta_raw:       {touch['force_z_delta_raw']}",
                    f"  force_xy_delta_raw:      {touch['force_xy_delta_raw']}",
                    f"  proximity:               {touch['proximity']}",
                    f"  temperature:             {touch['temperature']} C",
                    f"  age:                     {age}",
                    "",
                ]
            )
        self.detail_text.configure(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert("1.0", "\n".join(lines))
        self.detail_text.configure(state=tk.DISABLED)

    def _status(self, text: str) -> None:
        self.status_var.set(f"{time.strftime('%H:%M:%S')}  {text}")


def run_viewer(
    controller: DexHandController,
    *,
    rate_hz: float = 20.0,
    max_normal_force: float = 5.0,
    max_tangential_force: float = 2.0,
) -> None:
    root = tk.Tk()
    TactileViewer(
        root,
        controller,
        rate_hz=rate_hz,
        max_normal_force=max_normal_force,
        max_tangential_force=max_tangential_force,
    )
    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable DexHand tactile force viewer.")
    parser.add_argument("--hand", choices=["left", "right"], default="left")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--device-type", choices=["auto", "config", "100U", "200U"], default="auto")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--period-ms", type=int, default=20)
    parser.add_argument("--max-normal-force", type=float, default=5.0)
    parser.add_argument("--max-tangential-force", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with DexHandController(
            hand=args.hand,
            device_index=args.device_index,
            device_type=args.device_type,
            mock=args.mock,
            tactile_period_ms=args.period_ms,
        ) as controller:
            run_viewer(
                controller,
                rate_hz=args.rate,
                max_normal_force=args.max_normal_force,
                max_tangential_force=args.max_tangential_force,
            )
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Portable DexHand tactile viewer", str(exc))
        root.destroy()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
