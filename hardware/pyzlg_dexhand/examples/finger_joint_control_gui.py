#!/usr/bin/env python3
"""Finger-oriented GUI for fine DexHand joint control."""

import argparse
import logging
import math
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Dict, Iterable, List, Optional

from pyzlg_dexhand import (
    ControlMode,
    LeftDexHand,
    MockZCANWrapper,
    RightDexHand,
    ZCANWrapper,
)
from pyzlg_dexhand.dexhand_interface import FeedbackMode, StampedTouchFeedback
from pyzlg_dexhand.zcan import ZCANDeviceType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JointConfig:
    """Display and range configuration for one DexHand joint."""

    name: str
    label: str
    min_angle: float
    max_angle: float
    default: float = 0.0

    @property
    def half_angle(self) -> float:
        return (self.min_angle + self.max_angle) / 2.0


JOINT_CONFIGS: Dict[str, JointConfig] = {
    "th_rot": JointConfig("th_rot", "拇指旋转 th_rot", 0, 150),
    "th_mcp": JointConfig("th_mcp", "拇指 MCP th_mcp", 0, 90),
    "th_dip": JointConfig("th_dip", "拇指 DIP th_dip", 0, 90),
    "ff_spr": JointConfig("ff_spr", "四指侧展 ff_spr", 0, 30),
    "ff_mcp": JointConfig("ff_mcp", "食指 MCP ff_mcp", 0, 90),
    "ff_dip": JointConfig("ff_dip", "食指 DIP ff_dip", 0, 90),
    "mf_mcp": JointConfig("mf_mcp", "中指 MCP mf_mcp", 0, 90),
    "mf_dip": JointConfig("mf_dip", "中指 DIP mf_dip", 0, 90),
    "rf_mcp": JointConfig("rf_mcp", "无名指 MCP rf_mcp", 0, 90),
    "rf_dip": JointConfig("rf_dip", "无名指 DIP rf_dip", 0, 90),
    "lf_mcp": JointConfig("lf_mcp", "小指 MCP lf_mcp", 0, 90),
    "lf_dip": JointConfig("lf_dip", "小指 DIP lf_dip", 0, 90),
}

FINGER_GROUPS = [
    ("thumb", "拇指", ["th_rot", "th_mcp", "th_dip"]),
    ("index", "食指", ["ff_mcp", "ff_dip"]),
    ("middle", "中指", ["mf_mcp", "mf_dip"]),
    ("ring", "无名指", ["rf_mcp", "rf_dip"]),
    ("little", "小指", ["lf_mcp", "lf_dip"]),
    ("spread", "四指侧展", ["ff_spr"]),
]

FINGER_TOUCH_KEYS = {
    "thumb": "th",
    "index": "ff",
    "middle": "mf",
    "ring": "rf",
    "little": "lf",
}

FINGER_ORDER = ("th", "ff", "mf", "rf", "lf")
FINGER_LABELS = {
    "th": "拇指 th",
    "ff": "食指 ff",
    "mf": "中指 mf",
    "rf": "无名指 rf",
    "lf": "小指 lf",
}


@dataclass(frozen=True)
class FingerLayout:
    """Canvas layout for one fingertip tactile point."""

    x: int
    y: int
    label: str


FINGER_LAYOUTS = {
    "th": FingerLayout(85, 185, FINGER_LABELS["th"]),
    "ff": FingerLayout(175, 85, FINGER_LABELS["ff"]),
    "mf": FingerLayout(275, 60, FINGER_LABELS["mf"]),
    "rf": FingerLayout(375, 85, FINGER_LABELS["rf"]),
    "lf": FingerLayout(465, 140, FINGER_LABELS["lf"]),
}

CONTROL_MODES = {
    "impedance_grasp": ControlMode.IMPEDANCE_GRASP,
    "cascaded_pid": ControlMode.CASCADED_PID,
    "protect_hall_position": ControlMode.PROTECT_HALL_POSITION,
    "hall_position": ControlMode.HALL_POSITION,
}

DEVICE_TYPE_CHOICES = {
    "100U": ZCANDeviceType.ZCAN_USBCANFD_100U,
    "200U": ZCANDeviceType.ZCAN_USBCANFD_200U,
}


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a float value to a closed range."""
    return min(max(value, min_value), max_value)


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


def open_zcan(zcan: ZCANWrapper, device_index: int, device_type: str) -> str:
    """Open a ZCAN adapter and return the matched device type label."""
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
        "Check USB connection, permissions, config.yaml, and whether another process is using the adapter."
    )


class JointRow:
    """One row of joint controls: slider, numeric input, and open/close buttons."""

    def __init__(self, parent: ttk.Frame, app: "FingerJointControlApp", config: JointConfig):
        self.app = app
        self.config = config
        self.var = tk.DoubleVar(value=config.default)

        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(1, weight=1)

        ttk.Label(self.frame, text=config.label, width=20).grid(row=0, column=0, sticky="w", padx=(0, 8))

        self.scale = tk.Scale(
            self.frame,
            from_=config.min_angle,
            to=config.max_angle,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            showvalue=False,
            variable=self.var,
            command=self._scale_changed,
            length=220,
        )
        self.scale.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        self.spin = ttk.Spinbox(
            self.frame,
            from_=config.min_angle,
            to=config.max_angle,
            increment=0.1,
            textvariable=self.var,
            width=7,
            command=self._spin_changed,
            format="%.1f",
        )
        self.spin.grid(row=0, column=2, sticky="w")
        ttk.Label(self.frame, text="deg").grid(row=0, column=3, sticky="w", padx=(3, 8))

        ttk.Button(self.frame, text="开", width=4, command=self.open_joint).grid(row=0, column=4, padx=2)
        ttk.Button(self.frame, text="闭", width=4, command=self.close_joint).grid(row=0, column=5, padx=2)
        ttk.Button(self.frame, text="-", width=3, command=self.decrease).grid(row=0, column=6, padx=2)
        ttk.Button(self.frame, text="+", width=3, command=self.increase).grid(row=0, column=7, padx=2)

        self.feedback_label = ttk.Label(self.frame, text="fb: --", width=10)
        self.feedback_label.grid(row=0, column=8, sticky="e", padx=(8, 0))

        self.var.trace_add("write", self._var_changed)

    def _scale_changed(self, _value: str) -> None:
        self.app.on_joint_value_changed(self.config.name)

    def _spin_changed(self) -> None:
        self.app.on_joint_value_changed(self.config.name)

    def _var_changed(self, *_args) -> None:
        self.app.on_joint_value_changed(self.config.name)

    def value(self) -> float:
        return clamp(float(self.var.get()), self.config.min_angle, self.config.max_angle)

    def set_value(self, value: float) -> None:
        self.var.set(round(clamp(value, self.config.min_angle, self.config.max_angle), 1))

    def open_joint(self) -> None:
        self.app.set_joint_value(self.config.name, self.config.min_angle, send=True)

    def close_joint(self) -> None:
        self.app.set_joint_value(self.config.name, self.config.max_angle, send=True)

    def decrease(self) -> None:
        self.app.set_joint_value(self.config.name, self.value() - self.app.step_degrees(), send=True)

    def increase(self) -> None:
        self.app.set_joint_value(self.config.name, self.value() + self.app.step_degrees(), send=True)

    def set_feedback(self, angle: Optional[float]) -> None:
        if angle is None:
            self.feedback_label.configure(text="fb: --")
        else:
            self.feedback_label.configure(text=f"fb: {angle:.1f}")


class FingerJointControlApp:
    """Tkinter application for hand and per-finger DexHand control."""

    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.zcan = None
        self.hands = {}
        self.joint_rows: Dict[str, JointRow] = {}
        self.touch_labels: Dict[str, ttk.Label] = {}
        self.tactile_canvas_items = {}
        self.latest_touch: Dict[str, StampedTouchFeedback] = {}
        self.hand_positions = {
            hand_name: {name: config.default for name, config in JOINT_CONFIGS.items()}
            for hand_name in args.hands
        }
        self._bulk_update = False
        self._send_job = None
        self._pending_joint_names = set()
        self.active_hand_name = args.hands[0]

        self.current_hand_var = tk.StringVar(value=args.hands[0])
        self.control_mode_var = tk.StringVar(value=args.mode)
        self.auto_send_var = tk.BooleanVar(value=not args.no_auto_send)
        self.use_broadcast_var = tk.BooleanVar(value=False)
        self.clear_before_send_var = tk.BooleanVar(value=not args.no_pre_clear)
        self.clear_after_send_var = tk.BooleanVar(value=False)
        self.tactile_enabled_var = tk.BooleanVar(value=not args.no_tactile)
        self.tactile_rate_var = tk.DoubleVar(value=args.tactile_rate)
        self.max_normal_force_var = tk.DoubleVar(value=args.max_normal_force)
        self.max_tangential_force_var = tk.DoubleVar(value=args.max_tangential_force)
        self.step_var = tk.DoubleVar(value=args.step)
        self.status_var = tk.StringVar(value="未连接")

        self._connect_hardware()
        self._build_ui()
        initial_status = None
        if not args.no_read_initial:
            initial_status = self.initialize_positions_from_feedback()
        self._load_positions_for_current_hand()
        if not args.no_tactile and args.no_read_initial:
            self.configure_tactile_feedback(show_status=False)
        if initial_status:
            self.status(f"已连接: {', '.join(self.hands.keys())}; {initial_status}")
        else:
            self.status(f"已连接: {', '.join(self.hands.keys())}")
        if not args.no_tactile:
            self.schedule_tactile_update()

    def _connect_hardware(self) -> None:
        if self.args.mock:
            self.zcan = MockZCANWrapper()
            if not self.zcan.open(device_index=self.args.device_index):
                raise RuntimeError("Failed to open mock ZCAN")
            opened_type = "mock"
        else:
            self.zcan = ZCANWrapper()
            opened_type = open_zcan(self.zcan, self.args.device_index, self.args.device_type)

        for hand_name in self.args.hands:
            hand_class = LeftDexHand if hand_name == "left" else RightDexHand
            hand = hand_class(zcan=self.zcan, auto_init=False)
            if not hand.init(device_index=self.args.device_index):
                raise RuntimeError(f"Failed to initialize {hand_name} hand")
            self.hands[hand_name] = hand

        logger.info("Opened ZCAN adapter: %s", opened_type)

    def _build_ui(self) -> None:
        self.root.title("DexHand Finger Control + Tactile Force")
        self.root.geometry("1480x820")
        self.root.minsize(1220, 680)

        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TLabelframe.Label", font=("", 11, "bold"))
        style.configure("Status.TLabel", foreground="#1f2933")

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(1, weight=1)

        self._build_top_bar(main)
        self._build_finger_grid(main)
        self._build_tactile_panel(main)
        self._build_status_bar(main)

    def _build_top_bar(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        top.columnconfigure(14, weight=1)

        ttk.Label(top, text="手:").grid(row=0, column=0, padx=(0, 4))
        hand_combo = ttk.Combobox(
            top,
            textvariable=self.current_hand_var,
            values=list(self.hands.keys()),
            width=8,
            state="readonly",
        )
        hand_combo.grid(row=0, column=1, padx=(0, 10))
        hand_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_hand_changed())

        ttk.Label(top, text="模式:").grid(row=0, column=2, padx=(0, 4))
        mode_combo = ttk.Combobox(
            top,
            textvariable=self.control_mode_var,
            values=list(CONTROL_MODES.keys()),
            width=20,
            state="readonly",
        )
        mode_combo.grid(row=0, column=3, padx=(0, 10))

        ttk.Label(top, text="步进:").grid(row=0, column=4, padx=(0, 4))
        ttk.Spinbox(top, from_=0.1, to=20.0, increment=0.1, textvariable=self.step_var, width=6, format="%.1f").grid(
            row=0, column=5, padx=(0, 10)
        )
        ttk.Label(top, text="deg").grid(row=0, column=6, padx=(0, 10))

        ttk.Checkbutton(top, text="实时发送", variable=self.auto_send_var).grid(row=0, column=7, padx=(0, 8))
        ttk.Checkbutton(top, text="广播控制", variable=self.use_broadcast_var).grid(row=0, column=8, padx=(0, 8))
        ttk.Checkbutton(top, text="发送前清错", variable=self.clear_before_send_var).grid(row=0, column=9, padx=(0, 8))
        ttk.Checkbutton(top, text="发送后清错", variable=self.clear_after_send_var).grid(row=0, column=10, padx=(0, 10))

        ttk.Button(top, text="发送当前手势", command=self.send_current_pose).grid(row=0, column=11, padx=3)
        ttk.Button(top, text="全部张开", command=lambda: self.set_all("open")).grid(row=0, column=12, padx=3)
        ttk.Button(top, text="全部半闭", command=lambda: self.set_all("half")).grid(row=0, column=13, padx=3)
        ttk.Button(top, text="全部闭合", command=lambda: self.set_all("close")).grid(row=0, column=14, padx=3)
        ttk.Button(top, text="清除错误", command=self.clear_errors).grid(row=0, column=15, padx=3)
        ttk.Button(top, text="读取反馈", command=self.refresh_feedback).grid(row=0, column=16, padx=3)
        ttk.Button(top, text="反馈设为当前", command=self.set_current_from_feedback).grid(row=0, column=17, padx=3)

    def _build_finger_grid(self, parent: ttk.Frame) -> None:
        container = ttk.Frame(parent)
        container.grid(row=1, column=0, sticky="nsew")
        for col in range(2):
            container.columnconfigure(col, weight=1)
        for row in range(3):
            container.rowconfigure(row, weight=1)

        for idx, (finger_key, title, joint_names) in enumerate(FINGER_GROUPS):
            row = idx // 2
            col = idx % 2
            frame = ttk.LabelFrame(container, text=title, padding=10)
            frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
            frame.columnconfigure(0, weight=1)

            buttons = ttk.Frame(frame)
            buttons.grid(row=0, column=0, sticky="ew", pady=(0, 8))
            ttk.Button(buttons, text="本指张开", command=lambda joints=joint_names: self.set_joints(joints, "open")).pack(
                side=tk.LEFT, padx=(0, 4)
            )
            ttk.Button(buttons, text="本指半闭", command=lambda joints=joint_names: self.set_joints(joints, "half")).pack(
                side=tk.LEFT, padx=4
            )
            ttk.Button(buttons, text="本指闭合", command=lambda joints=joint_names: self.set_joints(joints, "close")).pack(
                side=tk.LEFT, padx=4
            )
            ttk.Button(buttons, text="发送本指", command=lambda joints=joint_names: self.send_current_pose(joints)).pack(
                side=tk.LEFT, padx=4
            )

            touch_key = FINGER_TOUCH_KEYS.get(finger_key)
            if touch_key is not None:
                label = ttk.Label(buttons, text="prox: -- / force: --", anchor="e")
                label.pack(side=tk.RIGHT)
                self.touch_labels[touch_key] = label

            for joint_row, joint_name in enumerate(joint_names, start=1):
                row_widget = JointRow(frame, self, JOINT_CONFIGS[joint_name])
                row_widget.frame.grid(row=joint_row, column=0, sticky="ew", pady=3)
                self.joint_rows[joint_name] = row_widget

    def _build_tactile_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="指尖触觉力觉", padding=8)
        panel.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=3)
        panel.rowconfigure(2, weight=2)

        controls = ttk.Frame(panel)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(10, weight=1)

        ttk.Checkbutton(controls, text="触觉刷新", variable=self.tactile_enabled_var).grid(row=0, column=0, padx=(0, 8))
        ttk.Label(controls, text="Hz").grid(row=0, column=1, padx=(0, 3))
        ttk.Spinbox(
            controls,
            from_=1.0,
            to=100.0,
            increment=1.0,
            textvariable=self.tactile_rate_var,
            width=5,
        ).grid(row=0, column=2, padx=(0, 8))

        ttk.Label(controls, text="Fn满量程").grid(row=0, column=3, padx=(0, 3))
        ttk.Spinbox(
            controls,
            from_=0.1,
            to=50.0,
            increment=0.1,
            textvariable=self.max_normal_force_var,
            width=6,
        ).grid(row=0, column=4, padx=(0, 8))

        ttk.Label(controls, text="Ft满量程").grid(row=0, column=5, padx=(0, 3))
        ttk.Spinbox(
            controls,
            from_=0.1,
            to=50.0,
            increment=0.1,
            textvariable=self.max_tangential_force_var,
            width=6,
        ).grid(row=0, column=6, padx=(0, 8))

        ttk.Button(controls, text="单次刷新", command=self.update_tactile_once).grid(row=0, column=7, padx=3)
        ttk.Button(controls, text="配置反馈", command=self.configure_tactile_feedback).grid(row=0, column=8, padx=3)
        ttk.Button(controls, text="清空", command=self.clear_tactile_display).grid(row=0, column=9, padx=3)

        self.tactile_canvas = tk.Canvas(
            panel,
            bg="#f7f9fb",
            height=330,
            highlightthickness=1,
            highlightbackground="#c7d0d9",
        )
        self.tactile_canvas.grid(row=1, column=0, sticky="nsew")
        self._draw_tactile_static_canvas()

        self.tactile_detail_text = tk.Text(panel, height=13, width=48, wrap=tk.NONE)
        self.tactile_detail_text.grid(row=2, column=0, sticky="nsew", pady=(8, 0))

        legend = ttk.Label(
            panel,
            text="圆面积/颜色=法向力 Fn，箭头=切向力 Ft/方向；prox=接近觉。",
            anchor="w",
        )
        legend.grid(row=3, column=0, sticky="ew", pady=(6, 0))

    def _draw_tactile_static_canvas(self) -> None:
        self.tactile_canvas.create_text(
            18,
            22,
            anchor="w",
            text="5 Fingertip Tactile Points",
            fill="#354052",
            font=("", 12, "bold"),
        )
        self.tactile_canvas.create_line(
            90,
            275,
            175,
            315,
            280,
            320,
            390,
            305,
            470,
            260,
            fill="#ccd6df",
            width=9,
            smooth=True,
        )
        for finger, layout in FINGER_LAYOUTS.items():
            self.tactile_canvas.create_text(
                layout.x,
                layout.y - 54,
                text=layout.label,
                fill="#1f2933",
                font=("", 10, "bold"),
            )
            base = self.tactile_canvas.create_oval(
                layout.x - 34,
                layout.y - 34,
                layout.x + 34,
                layout.y + 34,
                outline="#95a3b3",
                width=2,
                fill="#e8eef5",
            )
            force = self.tactile_canvas.create_oval(
                layout.x - 8,
                layout.y - 8,
                layout.x + 8,
                layout.y + 8,
                outline="",
                fill="#2d7dd2",
            )
            arrow = self.tactile_canvas.create_line(
                layout.x,
                layout.y,
                layout.x,
                layout.y,
                fill="#172b4d",
                width=3,
                arrow=tk.LAST,
            )
            value = self.tactile_canvas.create_text(
                layout.x,
                layout.y + 52,
                text="Fn --\nFt --\nprox --",
                fill="#354052",
                font=("", 9),
            )
            self.tactile_canvas_items[finger] = {
                "base": base,
                "force": force,
                "arrow": arrow,
                "value": value,
            }

    def _build_status_bar(self, parent: ttk.Frame) -> None:
        status = ttk.Label(parent, textvariable=self.status_var, style="Status.TLabel", anchor="w")
        status.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def current_hand_name(self) -> str:
        return self.current_hand_var.get()

    def current_hand(self):
        return self.hands[self.current_hand_name()]

    def step_degrees(self) -> float:
        return max(0.1, float(self.step_var.get()))

    def control_mode(self) -> ControlMode:
        return CONTROL_MODES[self.control_mode_var.get()]

    def collect_positions(self) -> Dict[str, float]:
        return {name: row.value() for name, row in self.joint_rows.items()}

    def status(self, text: str) -> None:
        self.status_var.set(f"{time.strftime('%H:%M:%S')}  {text}")

    def initialize_positions_from_feedback(self) -> str:
        """Use current hardware feedback as the initial GUI slider values."""
        if self.args.mock:
            return "mock 模式跳过初始反馈读取"

        loaded_summary = []
        for hand_name, hand in self.hands.items():
            try:
                hand.set_feedback_mode(FeedbackMode.PERIODIC, period_ms=self.args.tactile_period_ms, enable=True)
                time.sleep(0.08)
                loaded = self._read_hand_positions_into_cache(hand_name, attempts=6, wait_s=0.03)
                loaded_summary.append(f"{hand_name}: {loaded}/{len(JOINT_CONFIGS)}")
            except Exception as exc:
                logger.warning("Failed to initialize %s positions from feedback: %s", hand_name, exc)
                loaded_summary.append(f"{hand_name}: failed")

        return "初始反馈读取 " + ", ".join(loaded_summary)

    def _read_hand_positions_into_cache(self, hand_name: str, attempts: int = 3, wait_s: float = 0.03) -> int:
        """Read joint feedback and store finite joint angles in hand_positions."""
        hand = self.hands[hand_name]
        positions = dict(self.hand_positions[hand_name])
        loaded_joints = set()

        for _ in range(max(1, attempts)):
            feedback = hand.get_feedback()
            for joint_name, joint_feedback in feedback.joints.items():
                if joint_name not in JOINT_CONFIGS:
                    continue
                angle = joint_feedback.angle
                if angle is None or not math.isfinite(angle):
                    continue
                config = JOINT_CONFIGS[joint_name]
                positions[joint_name] = clamp(angle, config.min_angle, config.max_angle)
                loaded_joints.add(joint_name)

            if len(loaded_joints) == len(JOINT_CONFIGS):
                break
            time.sleep(wait_s)

        self.hand_positions[hand_name] = positions
        return len(loaded_joints)

    def on_hand_changed(self) -> None:
        self.hand_positions[self.active_hand_name] = self.collect_positions()
        self.active_hand_name = self.current_hand_name()
        self._load_positions_for_current_hand()
        self.latest_touch = {}
        self.render_tactile({})
        self.status(f"切换到 {self.current_hand_name()} hand")

    def _load_positions_for_current_hand(self) -> None:
        positions = self.hand_positions[self.current_hand_name()]
        self._bulk_update = True
        try:
            for joint_name, value in positions.items():
                self.joint_rows[joint_name].set_value(value)
        finally:
            self._bulk_update = False

    def on_joint_value_changed(self, joint_name: str) -> None:
        if self._bulk_update or joint_name not in self.joint_rows:
            return
        self.hand_positions[self.current_hand_name()][joint_name] = self.joint_rows[joint_name].value()
        if self.auto_send_var.get():
            self.schedule_send([joint_name])

    def schedule_send(self, joint_names: Optional[Iterable[str]] = None) -> None:
        if joint_names is None:
            self._pending_joint_names = None
        elif self._pending_joint_names is not None:
            self._pending_joint_names.update(joint_names)
        if self._send_job is not None:
            self.root.after_cancel(self._send_job)
        self._send_job = self.root.after(80, self._send_pending_pose)

    def _send_pending_pose(self) -> None:
        joint_names = None if self._pending_joint_names is None else list(self._pending_joint_names)
        self._pending_joint_names = set()
        self.send_current_pose(joint_names)

    def set_joint_value(self, joint_name: str, value: float, send: bool = False) -> None:
        self._bulk_update = True
        try:
            self.joint_rows[joint_name].set_value(value)
            self.hand_positions[self.current_hand_name()][joint_name] = self.joint_rows[joint_name].value()
        finally:
            self._bulk_update = False
        if send:
            self.send_current_pose([joint_name])

    def set_joints(self, joint_names: Iterable[str], target: str) -> None:
        self._bulk_update = True
        try:
            for joint_name in joint_names:
                config = JOINT_CONFIGS[joint_name]
                if target == "open":
                    value = config.min_angle
                elif target == "half":
                    value = config.half_angle
                elif target == "close":
                    value = config.max_angle
                else:
                    raise ValueError(f"Unknown target: {target}")
                self.joint_rows[joint_name].set_value(value)
                self.hand_positions[self.current_hand_name()][joint_name] = self.joint_rows[joint_name].value()
        finally:
            self._bulk_update = False
        self.send_current_pose(joint_names)

    def set_all(self, target: str) -> None:
        self.set_joints(JOINT_CONFIGS.keys(), target)

    def send_current_pose(self, joint_names: Optional[Iterable[str]] = None) -> None:
        if self._send_job is not None:
            self.root.after_cancel(self._send_job)
            self._send_job = None
        try:
            positions = self.collect_positions()
            self.hand_positions[self.current_hand_name()] = positions
            command_positions = positions
            if joint_names is not None:
                joint_name_set = set(joint_names)
                command_positions = {
                    name: positions[name]
                    for name in joint_name_set
                    if name in positions
                }
            if not command_positions:
                self.status("没有可发送的关节")
                return
            hand = self.current_hand()
            if self.clear_before_send_var.get():
                hand.clear_errors(clear_all=True, use_broadcast=False)
            ok = hand.move_joints(
                **command_positions,
                control_mode=self.control_mode(),
                use_broadcast=self.use_broadcast_var.get(),
                clear_error=False,
                request_feedback=True,
            )
            if self.clear_after_send_var.get():
                hand.clear_errors(clear_all=True, use_broadcast=False)
            if ok:
                if joint_names is None:
                    self.status(f"已发送 {self.current_hand_name()} 全手手势")
                else:
                    self.status(f"已发送 {self.current_hand_name()} {', '.join(sorted(command_positions))}")
            else:
                self.status("发送失败")
        except Exception as exc:
            logger.exception("Failed to send joint command")
            self.status(f"发送错误: {exc}")

    def clear_errors(self) -> None:
        try:
            self.current_hand().clear_errors(clear_all=True, use_broadcast=False)
            self.status(f"已清除 {self.current_hand_name()} 错误状态")
        except Exception as exc:
            logger.exception("Failed to clear errors")
            self.status(f"清错失败: {exc}")

    def refresh_feedback(self) -> None:
        try:
            feedback = self.current_hand().get_feedback()
            for joint_name, row in self.joint_rows.items():
                joint_feedback = feedback.joints.get(joint_name)
                row.set_feedback(None if joint_feedback is None else joint_feedback.angle)
            for touch_key, label in self.touch_labels.items():
                touch = feedback.touch.get(touch_key)
                if touch is None:
                    label.configure(text="prox: -- / force: --")
                else:
                    label.configure(text=f"prox: {touch.proximity} / force: {touch.normal_force:.2f}N")
            self.latest_touch = feedback.touch
            self.render_tactile(feedback.touch)
            self.status(f"已读取 {self.current_hand_name()} 反馈")
        except Exception as exc:
            logger.exception("Failed to refresh feedback")
            self.status(f"读取反馈失败: {exc}")

    def configure_tactile_feedback(self, show_status: bool = True) -> None:
        """Enable periodic tactile feedback without opening another ZCAN connection."""
        try:
            for hand in self.hands.values():
                hand.set_feedback_mode(
                    FeedbackMode.PERIODIC,
                    period_ms=self.args.tactile_period_ms,
                    enable=True,
                )
            time.sleep(max(self.args.tactile_period_ms / 1000.0 * 2.0, 0.05))
            if show_status:
                self.status("已配置触觉周期反馈")
        except Exception as exc:
            logger.exception("Failed to configure tactile feedback")
            if show_status:
                self.status(f"配置触觉反馈失败: {exc}")

    def schedule_tactile_update(self) -> None:
        """Schedule continuous tactile refresh in the same Tk event loop."""
        self.update_tactile_once(force=False, update_status=False)
        interval_ms = max(20, int(1000.0 / max(float(self.tactile_rate_var.get()), 1.0)))
        self.root.after(interval_ms, self.schedule_tactile_update)

    def update_tactile_once(self, force: bool = True, update_status: bool = True) -> None:
        """Read and render tactile feedback for the currently selected hand."""
        if not self.tactile_enabled_var.get() and not force:
            return
        try:
            touches = self.current_hand().get_tactile_feedback()
            self.latest_touch = touches
            self.render_tactile(touches)
            if update_status:
                self.status(f"已刷新 {self.current_hand_name()} 触觉")
        except Exception as exc:
            logger.exception("Failed to update tactile feedback")
            if update_status:
                self.status(f"触觉刷新失败: {exc}")

    def clear_tactile_display(self) -> None:
        """Clear tactile visualization values."""
        self.latest_touch = {}
        self.render_tactile({})
        self.status("已清空触觉显示")

    def render_tactile(self, touches: Dict[str, StampedTouchFeedback]) -> None:
        """Render tactile force map and detail table."""
        max_fn = max(float(self.max_normal_force_var.get()), 0.1)
        max_ft = max(float(self.max_tangential_force_var.get()), 0.1)
        for finger in FINGER_ORDER:
            self.render_tactile_finger(finger, touches.get(finger), max_fn, max_ft)
        self.render_tactile_detail(touches)

    def render_tactile_finger(
        self,
        finger: str,
        touch: Optional[StampedTouchFeedback],
        max_fn: float,
        max_ft: float,
    ) -> None:
        """Render one fingertip on the tactile canvas."""
        layout = FINGER_LAYOUTS[finger]
        items = self.tactile_canvas_items[finger]
        if touch is None:
            self.tactile_canvas.itemconfigure(items["force"], fill="#b7c4d1")
            self.tactile_canvas.coords(items["force"], layout.x - 7, layout.y - 7, layout.x + 7, layout.y + 7)
            self.tactile_canvas.coords(items["arrow"], layout.x, layout.y, layout.x, layout.y)
            self.tactile_canvas.itemconfigure(items["value"], text="Fn --\nFt --\nprox --")
            if finger in self.touch_labels:
                self.touch_labels[finger].configure(text="prox: -- / force: --")
            return

        ratio = min(max(touch.normal_force / max_fn, 0.0), 1.0)
        radius = 8 + 28 * math.sqrt(ratio)
        self.tactile_canvas.coords(
            items["force"],
            layout.x - radius,
            layout.y - radius,
            layout.x + radius,
            layout.y + radius,
        )
        self.tactile_canvas.itemconfigure(items["force"], fill=force_color(touch.normal_force, max_fn))

        direction = normalize_direction(touch.direction)
        if direction is None or touch.tangential_force <= 0:
            self.tactile_canvas.coords(items["arrow"], layout.x, layout.y, layout.x, layout.y)
        else:
            length = 14 + 42 * min(max(touch.tangential_force / max_ft, 0.0), 1.0)
            theta = math.radians(direction)
            end_x = layout.x + math.sin(theta) * length
            end_y = layout.y - math.cos(theta) * length
            self.tactile_canvas.coords(items["arrow"], layout.x, layout.y, end_x, end_y)

        self.tactile_canvas.itemconfigure(
            items["value"],
            text=f"Fn {touch.normal_force:.2f}N\nFt {touch.tangential_force:.2f}N\nprox {touch.proximity}",
        )
        if finger in self.touch_labels:
            self.touch_labels[finger].configure(text=f"prox: {touch.proximity} / force: {touch.normal_force:.2f}N")

    def render_tactile_detail(self, touches: Dict[str, StampedTouchFeedback]) -> None:
        """Render detailed tactile values in text form."""
        lines = []
        for finger in FINGER_ORDER:
            touch = touches.get(finger)
            lines.append(FINGER_LABELS[finger])
            if touch is None:
                lines.append("  no data")
                lines.append("")
                continue

            direction = normalize_direction(touch.direction)
            direction_text = "--" if direction is None else f"{direction} deg"
            timestamp_s = touch.timestamp / 1e9
            age = max(0.0, time.time() - timestamp_s)
            lines.extend(
                [
                    f"  normal_force:            {touch.normal_force:.4f} N",
                    f"  normal_force_delta:      {touch.normal_force_delta}",
                    f"  tangential_force:        {touch.tangential_force:.4f} N",
                    f"  tangential_force_delta:  {touch.tangential_force_delta}",
                    f"  direction:               {direction_text}",
                    f"  proximity:               {touch.proximity}",
                    f"  temperature:             {touch.temperature} C",
                    f"  age:                     {age:.3f} s",
                    "",
                ]
            )

        self.tactile_detail_text.configure(state=tk.NORMAL)
        self.tactile_detail_text.delete("1.0", tk.END)
        self.tactile_detail_text.insert("1.0", "\n".join(lines))
        self.tactile_detail_text.configure(state=tk.DISABLED)

    def set_current_from_feedback(self) -> None:
        """Read current feedback and use it as the editable GUI pose."""
        try:
            loaded = self._read_hand_positions_into_cache(self.current_hand_name(), attempts=6, wait_s=0.03)
            self._load_positions_for_current_hand()
            self.refresh_feedback()
            self.status(f"已用反馈更新当前界面: {loaded}/{len(JOINT_CONFIGS)} 个关节")
        except Exception as exc:
            logger.exception("Failed to set current pose from feedback")
            self.status(f"反馈设为当前失败: {exc}")

    def close(self) -> None:
        try:
            if self.zcan is not None:
                self.zcan.close()
        except Exception as exc:
            logger.warning("Failed to close ZCAN cleanly: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine finger-oriented DexHand joint control GUI.")
    parser.add_argument(
        "--hands",
        nargs="+",
        choices=["left", "right"],
        default=["left"],
        help="Hands to initialize. Default: left.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock hardware for UI testing.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="ZCAN device index. Default: 0.",
    )
    parser.add_argument(
        "--device-type",
        choices=["auto", "config", "100U", "200U"],
        default="auto",
        help="ZCAN adapter type. Default: auto.",
    )
    parser.add_argument(
        "--mode",
        choices=list(CONTROL_MODES.keys()),
        default="impedance_grasp",
        help="Initial control mode. Default: impedance_grasp.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=1.0,
        help="Initial button step in degrees. Default: 1.0.",
    )
    parser.add_argument(
        "--no-auto-send",
        action="store_true",
        help="Disable sending while sliders/numeric fields are edited.",
    )
    parser.add_argument(
        "--no-pre-clear",
        action="store_true",
        help="Disable clearing latched board errors immediately before each motion command.",
    )
    parser.add_argument(
        "--no-read-initial",
        action="store_true",
        help="Do not initialize GUI joint values from current hardware feedback.",
    )
    parser.add_argument(
        "--no-tactile",
        action="store_true",
        help="Disable automatic embedded tactile force refresh at startup.",
    )
    parser.add_argument(
        "--tactile-rate",
        type=float,
        default=20.0,
        help="Embedded tactile visualization refresh rate in Hz. Default: 20.",
    )
    parser.add_argument(
        "--tactile-period-ms",
        type=int,
        default=20,
        help="Hardware periodic feedback period used for joint/tactile feedback in ms. Default: 20.",
    )
    parser.add_argument(
        "--max-normal-force",
        type=float,
        default=5.0,
        help="Normal force full-scale value for tactile visualization in N. Default: 5.0.",
    )
    parser.add_argument(
        "--max-tangential-force",
        type=float,
        default=2.0,
        help="Tangential force full-scale value for tactile visualization in N. Default: 2.0.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    app = None
    try:
        app = FingerJointControlApp(root, args)
    except Exception as exc:
        logger.exception("Failed to start GUI")
        messagebox.showerror("DexHand GUI", str(exc))
        root.destroy()
        return 1

    def on_close() -> None:
        app.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
