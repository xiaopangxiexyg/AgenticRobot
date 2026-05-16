#!/usr/bin/env python3
"""Visualize five fingertip tactile force readings."""

import argparse
import math
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Dict, Optional

from pyzlg_dexhand import LeftDexHand, MockZCANWrapper, RightDexHand, ZCANWrapper
from pyzlg_dexhand.dexhand_interface import FeedbackMode, StampedTouchFeedback
from pyzlg_dexhand.zcan import ZCANDeviceType


FINGER_ORDER = ("th", "ff", "mf", "rf", "lf")
FINGER_LABELS = {
    "th": "拇指 th",
    "ff": "食指 ff",
    "mf": "中指 mf",
    "rf": "无名指 rf",
    "lf": "小指 lf",
}
DEVICE_TYPE_CHOICES = {
    "100U": ZCANDeviceType.ZCAN_USBCANFD_100U,
    "200U": ZCANDeviceType.ZCAN_USBCANFD_200U,
}


@dataclass
class FingerLayout:
    """Canvas layout for one fingertip tactile point."""

    x: int
    y: int
    label: str


FINGER_LAYOUTS = {
    "th": FingerLayout(155, 275, FINGER_LABELS["th"]),
    "ff": FingerLayout(310, 135, FINGER_LABELS["ff"]),
    "mf": FingerLayout(455, 95, FINGER_LABELS["mf"]),
    "rf": FingerLayout(600, 135, FINGER_LABELS["rf"]),
    "lf": FingerLayout(745, 205, FINGER_LABELS["lf"]),
}


def open_zcan(zcan: ZCANWrapper, device_index: int, device_type: str) -> str:
    """Open the ZCAN adapter and return the matched device type label."""
    if device_type in DEVICE_TYPE_CHOICES:
        selected_type = DEVICE_TYPE_CHOICES[device_type]
        if zcan.open(device_type=selected_type, device_index=device_index):
            return device_type
        raise RuntimeError(f"Failed to open USBCANFD-{device_type}")

    if device_type == "config":
        if zcan.open(device_index=device_index):
            return zcan.zcan_device_type.replace("ZCAN_USBCANFD_", "")
        raise RuntimeError(f"Failed to open configured adapter type {zcan.zcan_device_type}")

    configured_type = zcan.device_type
    candidates = [configured_type]
    for candidate in DEVICE_TYPE_CHOICES.values():
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if zcan.open(device_type=candidate, device_index=device_index):
            for label, enum_value in DEVICE_TYPE_CHOICES.items():
                if enum_value == candidate:
                    return label

    raise RuntimeError(
        "Failed to open USBCANFD-100U or USBCANFD-200U. "
        "Check USB connection, permissions, config.yaml, and adapter occupancy."
    )


def normalize_direction(direction: int) -> Optional[int]:
    """Return None for invalid force direction readings."""
    if direction == 65535 or direction > 359 or direction < 0:
        return None
    return direction


def force_color(normal_force: float, max_force: float) -> str:
    """Map normal force to a blue-red color."""
    ratio = 0.0 if max_force <= 0 else min(max(normal_force / max_force, 0.0), 1.0)
    red = int(45 + 210 * ratio)
    green = int(125 * (1.0 - ratio) + 35 * ratio)
    blue = int(220 * (1.0 - ratio) + 45 * ratio)
    return f"#{red:02x}{green:02x}{blue:02x}"


class TactileForceVisualizer:
    """Tkinter tactile force visualization app."""

    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.zcan = None
        self.hands = {}
        self.running = True
        self.current_hand_var = tk.StringVar(value=args.hands[0])
        self.rate_var = tk.DoubleVar(value=args.rate)
        self.max_normal_force_var = tk.DoubleVar(value=args.max_normal_force)
        self.max_tangential_force_var = tk.DoubleVar(value=args.max_tangential_force)
        self.status_var = tk.StringVar(value="未连接")
        self.canvas_items = {}
        self.latest_touch: Dict[str, StampedTouchFeedback] = {}
        self.demo_t0 = time.time()

        if not args.demo:
            self._connect_hardware()

        self._build_ui()
        if not args.demo and not args.no_configure_feedback:
            self._configure_feedback()
        self.status("demo 模式" if args.demo else f"已连接: {', '.join(self.hands.keys())}")
        self._schedule_update()

    def _connect_hardware(self) -> None:
        if self.args.mock:
            self.zcan = MockZCANWrapper()
            if not self.zcan.open(device_index=self.args.device_index):
                raise RuntimeError("Failed to open mock ZCAN")
        else:
            self.zcan = ZCANWrapper()
            open_zcan(self.zcan, self.args.device_index, self.args.device_type)

        for hand_name in self.args.hands:
            hand_class = LeftDexHand if hand_name == "left" else RightDexHand
            hand = hand_class(zcan=self.zcan, auto_init=False)
            if not hand.init(device_index=self.args.device_index):
                raise RuntimeError(f"Failed to initialize {hand_name} hand")
            self.hands[hand_name] = hand

    def _configure_feedback(self) -> None:
        for hand in self.hands.values():
            hand.set_feedback_mode(
                FeedbackMode.PERIODIC,
                period_ms=self.args.period_ms,
                enable=True,
            )
        time.sleep(max(self.args.period_ms / 1000.0 * 2.0, 0.05))

    def _build_ui(self) -> None:
        self.root.title("DexHand Fingertip Tactile Force Visualizer")
        self.root.geometry("1040x720")
        self.root.minsize(920, 620)

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(12, weight=1)

        ttk.Label(controls, text="手:").grid(row=0, column=0, padx=(0, 4))
        hand_values = self.args.hands if not self.args.demo else ["demo"]
        if self.args.demo:
            self.current_hand_var.set("demo")
        hand_combo = ttk.Combobox(
            controls,
            textvariable=self.current_hand_var,
            values=hand_values,
            state="readonly",
            width=8,
        )
        hand_combo.grid(row=0, column=1, padx=(0, 10))

        ttk.Label(controls, text="刷新 Hz:").grid(row=0, column=2, padx=(0, 4))
        ttk.Spinbox(controls, from_=1.0, to=100.0, increment=1.0, textvariable=self.rate_var, width=6).grid(
            row=0, column=3, padx=(0, 10)
        )

        ttk.Label(controls, text="Fn满量程 N:").grid(row=0, column=4, padx=(0, 4))
        ttk.Spinbox(
            controls,
            from_=0.1,
            to=50.0,
            increment=0.1,
            textvariable=self.max_normal_force_var,
            width=7,
        ).grid(row=0, column=5, padx=(0, 10))

        ttk.Label(controls, text="Ft满量程 N:").grid(row=0, column=6, padx=(0, 4))
        ttk.Spinbox(
            controls,
            from_=0.1,
            to=50.0,
            increment=0.1,
            textvariable=self.max_tangential_force_var,
            width=7,
        ).grid(row=0, column=7, padx=(0, 10))

        ttk.Button(controls, text="暂停/继续", command=self.toggle_running).grid(row=0, column=8, padx=3)
        ttk.Button(controls, text="单次刷新", command=lambda: self.update_once(force=True)).grid(row=0, column=9, padx=3)
        ttk.Button(controls, text="重新配置反馈", command=self._configure_feedback_safe).grid(row=0, column=10, padx=3)
        ttk.Button(controls, text="清空显示", command=self.clear_display).grid(row=0, column=11, padx=3)

        body = ttk.Frame(main)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        canvas_frame = ttk.LabelFrame(body, text="5 指尖触觉点", padding=8)
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="#f7f9fb", highlightthickness=1, highlightbackground="#c7d0d9")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._draw_static_canvas()

        detail_frame = ttk.LabelFrame(body, text="详细力觉数据", padding=8)
        detail_frame.grid(row=0, column=1, sticky="nsew")
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        self.detail_text = tk.Text(detail_frame, height=24, width=42, wrap=tk.NONE)
        self.detail_text.grid(row=0, column=0, sticky="nsew")

        legend = ttk.Label(
            main,
            text="圆面积/颜色表示法向力 Fn，箭头表示切向力 Ft 和方向；prox 为接近觉原始值，temp 为温度。",
            anchor="w",
        )
        legend.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        status = ttk.Label(main, textvariable=self.status_var, anchor="w")
        status.grid(row=3, column=0, sticky="ew", pady=(6, 0))

    def status(self, text: str) -> None:
        """Update the status bar with a timestamp."""
        self.status_var.set(f"{time.strftime('%H:%M:%S')}  {text}")

    def _draw_static_canvas(self) -> None:
        self.canvas.create_text(30, 30, anchor="w", text="DexHand fingertips", fill="#354052", font=("", 13, "bold"))
        self.canvas.create_line(160, 430, 310, 515, 470, 530, 650, 510, 790, 430, fill="#ccd6df", width=10, smooth=True)
        for finger, layout in FINGER_LAYOUTS.items():
            self.canvas.create_text(layout.x, layout.y - 74, text=layout.label, fill="#1f2933", font=("", 11, "bold"))
            base = self.canvas.create_oval(
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
            value = self.canvas.create_text(layout.x, layout.y + 70, text="Fn --\nFt --\nprox --", fill="#354052")
            self.canvas_items[finger] = {
                "base": base,
                "force": force,
                "arrow": arrow,
                "value": value,
            }

    def toggle_running(self) -> None:
        self.running = not self.running
        self.status("继续刷新" if self.running else "已暂停")

    def clear_display(self) -> None:
        self.latest_touch = {}
        self._render({})
        self.status("已清空显示")

    def _configure_feedback_safe(self) -> None:
        if self.args.demo:
            self.status("demo 模式无需配置反馈")
            return
        try:
            self._configure_feedback()
            self.status("已重新配置周期反馈")
        except Exception as exc:
            self.status(f"配置反馈失败: {exc}")

    def _schedule_update(self) -> None:
        self.update_once()
        interval_ms = max(10, int(1000.0 / max(float(self.rate_var.get()), 1.0)))
        self.root.after(interval_ms, self._schedule_update)

    def update_once(self, force: bool = False) -> None:
        if not self.running and not force:
            return
        try:
            touches = self._read_touches()
            self.latest_touch = touches
            self._render(touches)
            self.status(f"已刷新 {self.current_hand_var.get()} 触觉")
        except Exception as exc:
            self.status(f"刷新失败: {exc}")

    def _read_touches(self) -> Dict[str, StampedTouchFeedback]:
        if self.args.demo:
            return self._demo_touches()
        hand = self.hands[self.current_hand_var.get()]
        return hand.get_tactile_feedback()

    def _demo_touches(self) -> Dict[str, StampedTouchFeedback]:
        now = time.time()
        t = now - self.demo_t0
        touches = {}
        for idx, finger in enumerate(FINGER_ORDER):
            phase = t * 1.7 + idx * 0.9
            normal_force = max(0.0, 2.0 + 1.8 * math.sin(phase))
            tangential_force = max(0.0, 0.7 + 0.6 * math.cos(phase * 0.8))
            touches[finger] = StampedTouchFeedback(
                timestamp=now * 1e9,
                normal_force=normal_force,
                normal_force_delta=int(80 * math.sin(phase * 1.5)),
                tangential_force=tangential_force,
                tangential_force_delta=int(50 * math.cos(phase * 1.2)),
                direction=int((math.degrees(phase) * 2.0) % 360),
                proximity=int(300 + 180 * max(0.0, math.sin(phase + 0.4))),
                temperature=26 + int(2 * max(0.0, math.sin(phase * 0.4))),
            )
        return touches

    def _render(self, touches: Dict[str, StampedTouchFeedback]) -> None:
        max_fn = max(float(self.max_normal_force_var.get()), 0.1)
        max_ft = max(float(self.max_tangential_force_var.get()), 0.1)

        for finger in FINGER_ORDER:
            touch = touches.get(finger)
            self._render_finger(finger, touch, max_fn, max_ft)
        self._render_detail_text(touches)

    def _render_finger(
        self,
        finger: str,
        touch: Optional[StampedTouchFeedback],
        max_fn: float,
        max_ft: float,
    ) -> None:
        layout = FINGER_LAYOUTS[finger]
        items = self.canvas_items[finger]
        if touch is None:
            self.canvas.itemconfigure(items["force"], fill="#b7c4d1")
            self.canvas.coords(items["force"], layout.x - 8, layout.y - 8, layout.x + 8, layout.y + 8)
            self.canvas.coords(items["arrow"], layout.x, layout.y, layout.x, layout.y)
            self.canvas.itemconfigure(items["value"], text="Fn --\nFt --\nprox --")
            return

        ratio = min(max(touch.normal_force / max_fn, 0.0), 1.0)
        radius = 10 + 38 * math.sqrt(ratio)
        self.canvas.coords(
            items["force"],
            layout.x - radius,
            layout.y - radius,
            layout.x + radius,
            layout.y + radius,
        )
        self.canvas.itemconfigure(items["force"], fill=force_color(touch.normal_force, max_fn))

        direction = normalize_direction(touch.direction)
        if direction is None or touch.tangential_force <= 0:
            self.canvas.coords(items["arrow"], layout.x, layout.y, layout.x, layout.y)
        else:
            length = 18 + 54 * min(max(touch.tangential_force / max_ft, 0.0), 1.0)
            theta = math.radians(direction)
            end_x = layout.x + math.sin(theta) * length
            end_y = layout.y - math.cos(theta) * length
            self.canvas.coords(items["arrow"], layout.x, layout.y, end_x, end_y)

        self.canvas.itemconfigure(
            items["value"],
            text=f"Fn {touch.normal_force:.2f}N\nFt {touch.tangential_force:.2f}N\nprox {touch.proximity}",
        )

    def _render_detail_text(self, touches: Dict[str, StampedTouchFeedback]) -> None:
        lines = []
        for finger in FINGER_ORDER:
            label = FINGER_LABELS[finger]
            touch = touches.get(finger)
            lines.append(f"{label}")
            if touch is None:
                lines.append("  no data")
                lines.append("")
                continue

            direction = normalize_direction(touch.direction)
            direction_text = "--" if direction is None else f"{direction} deg"
            lines.extend(
                [
                    f"  normal_force:            {touch.normal_force:.4f} N",
                    f"  normal_force_delta:      {touch.normal_force_delta}",
                    f"  tangential_force:        {touch.tangential_force:.4f} N",
                    f"  tangential_force_delta:  {touch.tangential_force_delta}",
                    f"  direction:               {direction_text}",
                    f"  proximity:               {touch.proximity}",
                    f"  temperature:             {touch.temperature} C",
                    f"  age:                     {self._age_seconds(touch):.3f} s",
                    "",
                ]
            )
        self.detail_text.configure(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert("1.0", "\n".join(lines))
        self.detail_text.configure(state=tk.DISABLED)

    def _age_seconds(self, touch: StampedTouchFeedback) -> float:
        timestamp_s = touch.timestamp / 1e9
        return max(0.0, time.time() - timestamp_s)

    def close(self) -> None:
        if self.zcan is not None:
            self.zcan.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize five fingertip tactile force readings.")
    parser.add_argument("--hands", nargs="+", choices=["left", "right"], default=["left"])
    parser.add_argument("--mock", action="store_true", help="Use mock hardware.")
    parser.add_argument("--demo", action="store_true", help="Use synthetic tactile data and do not open hardware.")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--device-type", choices=["auto", "config", "100U", "200U"], default="auto")
    parser.add_argument("--rate", type=float, default=20.0, help="Refresh rate in Hz.")
    parser.add_argument("--period-ms", type=int, default=20, help="Periodic feedback interval to request.")
    parser.add_argument("--no-configure-feedback", action="store_true")
    parser.add_argument("--max-normal-force", type=float, default=5.0, help="Normal-force visualization scale in N.")
    parser.add_argument("--max-tangential-force", type=float, default=2.0, help="Tangential-force arrow scale in N.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    app = None
    try:
        app = TactileForceVisualizer(root, args)
    except Exception as exc:
        messagebox.showerror("DexHand tactile visualizer", str(exc))
        root.destroy()
        return 1

    def on_close() -> None:
        app.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
