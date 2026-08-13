from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from .core import Endpoint, GripperPair, URDFDocument, URDFValidationError, get_origin, set_origin
from .mesh import MeshCache
from .viewer import GripperCanvas, build_gripper_scene, geometry_key


def _millimeters(values: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(value * 1000.0 for value in values)  # type: ignore[return-value]


def _degrees(values: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(math.degrees(value) for value in values)  # type: ignore[return-value]


@dataclass
class ParameterControl:
    scale: ttk.Scale
    spinbox: ttk.Spinbox
    variable: tk.DoubleVar
    step_kind: str
    hard_minimum: float
    hard_maximum: float


class GripperURDFEditorApp:
    def __init__(self, root: tk.Tk, input_path: str | Path | None = None):
        self.root = root
        self.root.title("Gripper URDF Editor · 夹爪末端编辑器")
        self.root.geometry("1420x900")
        self.root.minsize(1120, 720)
        self.document: URDFDocument | None = None
        self.mesh_cache = MeshCache(max_triangles=1600)
        self.pairs: list[GripperPair] = []
        self.endpoint_options: dict[str, list[Endpoint]] = {"a": [], "b": []}
        self.endpoints: dict[str, Endpoint | None] = {"a": None, "b": None}
        self.undo_stack: list[tuple[bytes, str]] = []
        self.redo_stack: list[tuple[bytes, str]] = []
        self.baseline_snapshot: bytes | None = None
        self.dirty = False
        self._last_pair_key: tuple[str, str] | None = None
        self._suspend_live_controls = False
        self._live_before: bytes | None = None
        self._live_description = ""
        self._live_after_id: str | None = None
        self.pose_defaults: dict[tuple[str, str, str], tuple[tuple[float, float, float], tuple[float, float, float]]] = {}

        self.path_var = tk.StringVar(value="")
        self.pair_var = tk.StringVar(value="")
        self.endpoint_var = {"a": tk.StringVar(), "b": tk.StringVar()}
        self.size_vars = {side: [tk.DoubleVar(value=1.0) for _ in range(3)] for side in ("a", "b")}
        self.pose_vars = {side: [tk.DoubleVar(value=0.0) for _ in range(3)] for side in ("a", "b")}
        self.rpy_vars = {side: [tk.DoubleVar(value=0.0) for _ in range(3)] for side in ("a", "b")}
        self.pose_owner_vars = {side: tk.StringVar(value="—") for side in ("a", "b")}
        self.kind_vars = {side: tk.StringVar(value="—") for side in ("a", "b")}
        self.closed_q_var = tk.DoubleVar(value=-35.0)
        self.open_q_var = tk.DoubleVar(value=0.0)
        self.linear_step_var = tk.StringVar(value="0.01")
        self.angular_step_var = tk.StringVar(value="0.01")
        self.preview_var = tk.DoubleVar(value=100.0)
        self.preview_label_var = tk.StringVar(value="Preview: —")
        self.gap_label_var = tk.StringVar(value="开口: —")
        self.link_size_var = tk.BooleanVar(value=True)
        self.link_pose_var = tk.BooleanVar(value=True)
        self.sync_geometry_var = tk.BooleanVar(value=True)
        self.snap_contact_var = tk.BooleanVar(value=True)
        self.show_visuals_var = tk.BooleanVar(value=True)
        self.show_collisions_var = tk.BooleanVar(value=True)
        self.show_contacts_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="请选择一个 URDF。输入文件永远不会被原位覆盖。")
        self.parameter_controls: dict[tuple[str, str, int], ParameterControl] = {}

        self._build_ui()
        self._precision_changed()
        self.status_var.set("请选择一个 URDF。输入文件永远不会被原位覆盖。")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.bind_all("<Control-z>", lambda _event: self.undo())
        self.root.bind_all("<Control-y>", lambda _event: self.redo())
        self.root.bind_all("<Control-s>", lambda _event: self.export_as())
        if input_path is not None:
            self.load(input_path)

    @property
    def pair(self) -> GripperPair | None:
        selection = self.pair_combo.current()
        return self.pairs[selection] if 0 <= selection < len(self.pairs) else None

    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(1, weight=1)
        ttk.Label(toolbar, text="URDF").grid(row=0, column=0, padx=(0, 6))
        ttk.Entry(toolbar, textvariable=self.path_var, state="readonly").grid(row=0, column=1, sticky="ew")
        ttk.Button(toolbar, text="打开…", command=self.open_dialog).grid(row=0, column=2, padx=5)
        ttk.Button(toolbar, text="导出新 URDF…", command=self.export_as).grid(row=0, column=3, padx=5)
        ttk.Button(toolbar, text="撤销", command=self.undo).grid(row=0, column=4, padx=(18, 3))
        ttk.Button(toolbar, text="重做", command=self.redo).grid(row=0, column=5, padx=3)
        ttk.Button(toolbar, text="回到输入版本", command=self.revert).grid(row=0, column=6, padx=3)

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew")
        controls_outer = ttk.Frame(main, width=440)
        viewer_outer = ttk.Frame(main)
        main.add(controls_outer, weight=0)
        main.add(viewer_outer, weight=1)

        controls_canvas = tk.Canvas(controls_outer, highlightthickness=0, width=430)
        scrollbar = ttk.Scrollbar(controls_outer, orient=tk.VERTICAL, command=controls_canvas.yview)
        controls = ttk.Frame(controls_canvas, padding=10)
        controls_window = controls_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls_canvas.configure(yscrollcommand=scrollbar.set)
        controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        controls.bind("<Configure>", lambda _event: controls_canvas.configure(scrollregion=controls_canvas.bbox("all")))
        controls_canvas.bind("<Configure>", lambda event: controls_canvas.itemconfigure(controls_window, width=event.width))

        pair_frame = ttk.LabelFrame(controls, text="1. 自动识别平行夹爪", padding=9)
        pair_frame.pack(fill=tk.X, pady=(0, 8))
        self.pair_combo = ttk.Combobox(pair_frame, textvariable=self.pair_var, state="readonly")
        self.pair_combo.pack(fill=tk.X)
        self.pair_combo.bind("<<ComboboxSelected>>", lambda _event: self._pair_changed())
        ttk.Label(
            pair_frame,
            text="按 prismatic 同父级、平行运动轴和 mimic 关系排序；整机双臂 URDF 可在此切换左右夹爪。",
            wraplength=380,
            foreground="#596273",
        ).pack(anchor="w", pady=(6, 0))

        motion = ttk.LabelFrame(controls, text="2. 开合范围与实时预览", padding=9)
        motion.pack(fill=tk.X, pady=8)
        motion.columnconfigure(1, weight=1)
        self._add_parameter_row(
            motion, 0, "闭合 q", self.closed_q_var, -100.0, 100.0, 0.5,
            self._live_motion_changed, unit="mm", control_key=("motion", "q", 0), step_kind="linear",
        )
        self._add_parameter_row(
            motion, 1, "张开 q", self.open_q_var, -100.0, 100.0, 0.5,
            self._live_motion_changed, unit="mm", control_key=("motion", "q", 1), step_kind="linear",
        )
        ttk.Label(motion, text="拖动实时更新；微调步长可在末端面板选择。", foreground="#596273").grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(3, 5)
        )
        preview_buttons = ttk.Frame(motion)
        preview_buttons.grid(row=3, column=0, columnspan=4, sticky="ew")
        ttk.Button(preview_buttons, text="闭合预览", command=lambda: self._set_preview(0.0)).pack(side=tk.LEFT)
        ttk.Button(preview_buttons, text="半开", command=lambda: self._set_preview(50.0)).pack(side=tk.LEFT, padx=4)
        ttk.Button(preview_buttons, text="张开预览", command=lambda: self._set_preview(100.0)).pack(side=tk.LEFT)
        ttk.Scale(motion, from_=0, to=100, variable=self.preview_var, command=lambda _value: self.refresh_scene()).grid(
            row=4, column=0, columnspan=4, sticky="ew", pady=(4, 0)
        )
        labels = ttk.Frame(motion)
        labels.grid(row=5, column=0, columnspan=4, sticky="ew")
        ttk.Label(labels, textvariable=self.preview_label_var).pack(side=tk.LEFT)
        ttk.Label(labels, textvariable=self.gap_label_var, foreground="#276f56").pack(side=tk.RIGHT)

        notebook = ttk.Notebook(controls)
        notebook.pack(fill=tk.X, pady=8)
        self.endpoint_combos: dict[str, ttk.Combobox] = {}
        for side, title, color in (("a", "末端 A · 蓝", "#167fc4"), ("b", "末端 B · 橙", "#bd6418")):
            frame = ttk.Frame(notebook, padding=9)
            notebook.add(frame, text=title)
            self._build_endpoint_tab(frame, side, color)

        linked = ttk.LabelFrame(controls, text="3. 对称夹爪联动与语义保护", padding=9)
        linked.pack(fill=tk.X, pady=8)
        ttk.Checkbutton(linked, text="A/B 尺寸同值联动", variable=self.link_size_var).pack(anchor="w")
        ttk.Checkbutton(linked, text="A/B 安装位置与旋转同值联动", variable=self.link_pose_var).pack(anchor="w")
        ttk.Checkbutton(linked, text="同类 visual/collision 联动", variable=self.sync_geometry_var).pack(anchor="w")
        ttk.Checkbutton(linked, text="自动把 inner_contact frame 贴到新的内侧表面", variable=self.snap_contact_var).pack(anchor="w")
        ttk.Label(
            linked,
            text="若两侧局部坐标系并不相同，请关闭位置联动；尺寸联动仍可单独保留。",
            wraplength=380,
            foreground="#7b5529",
        ).pack(anchor="w", pady=(5, 0))

        export_note = ttk.LabelFrame(controls, text="导出保证", padding=9)
        export_note.pack(fill=tk.X, pady=8)
        ttk.Label(
            export_note,
            text="仅 Save As；原输入文件不可覆盖。相对 mesh 路径会按新 URDF 目录自动重写，导出前后均执行 XML、关节、link、mimic 与运动树校验。",
            wraplength=380,
        ).pack(anchor="w")
        ttk.Label(
            export_note,
            text="注意：几何改变不会自动猜测材料密度或改写 mass/inertia；用于动力学仿真前需另行复核 inertial。",
            wraplength=380,
            foreground="#a04c28",
        ).pack(anchor="w", pady=(5, 0))

        viewer_outer.rowconfigure(1, weight=1)
        viewer_outer.columnconfigure(0, weight=1)
        viewbar = ttk.Frame(viewer_outer, padding=(8, 6))
        viewbar.grid(row=0, column=0, sticky="ew")
        for text, name in (("等轴", "iso"), ("正视", "front"), ("侧视", "side"), ("俯视", "top")):
            ttk.Button(viewbar, text=text, command=lambda selected=name: self.viewer.set_view(selected)).pack(side=tk.LEFT, padx=2)
        ttk.Button(viewbar, text="适配视图", command=self.viewer_fit).pack(side=tk.LEFT, padx=(10, 2))
        ttk.Checkbutton(viewbar, text="Visual", variable=self.show_visuals_var, command=self.refresh_scene).pack(side=tk.RIGHT, padx=5)
        ttk.Checkbutton(viewbar, text="Collision", variable=self.show_collisions_var, command=self.refresh_scene).pack(side=tk.RIGHT, padx=5)
        ttk.Checkbutton(viewbar, text="Contact frames", variable=self.show_contacts_var, command=self.refresh_scene).pack(side=tk.RIGHT, padx=5)
        self.viewer = GripperCanvas(viewer_outer)
        self.viewer.grid(row=1, column=0, sticky="nsew")

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(10, 6), relief=tk.SUNKEN)
        status.grid(row=2, column=0, sticky="ew")

    def _build_endpoint_tab(self, parent: ttk.Frame, side: str, color: str) -> None:
        ttk.Label(parent, text="自动选择的末端几何", foreground=color).grid(row=0, column=0, columnspan=4, sticky="w")
        combo = ttk.Combobox(parent, textvariable=self.endpoint_var[side], state="readonly")
        combo.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(2, 7))
        combo.bind("<<ComboboxSelected>>", lambda _event, selected=side: self._endpoint_changed(selected))
        self.endpoint_combos[side] = combo
        parent.columnconfigure(1, weight=1)

        precision = ttk.Frame(parent)
        precision.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(0, 5))
        ttk.Label(precision, text="线性步长").pack(side=tk.LEFT)
        linear = ttk.Combobox(
            precision,
            textvariable=self.linear_step_var,
            values=("0.01", "0.05", "0.10", "0.50", "1.00"),
            state="readonly",
            width=6,
        )
        linear.pack(side=tk.LEFT, padx=(4, 3))
        ttk.Label(precision, text="mm").pack(side=tk.LEFT)
        ttk.Label(precision, text="旋转步长").pack(side=tk.LEFT, padx=(12, 0))
        angular = ttk.Combobox(
            precision,
            textvariable=self.angular_step_var,
            values=("0.01", "0.05", "0.10", "0.50", "1.00"),
            state="readonly",
            width=6,
        )
        angular.pack(side=tk.LEFT, padx=(4, 3))
        ttk.Label(precision, text="°").pack(side=tk.LEFT)
        linear.bind("<<ComboboxSelected>>", lambda _event: self._precision_changed())
        angular.bind("<<ComboboxSelected>>", lambda _event: self._precision_changed())

        ttk.Label(parent, text="几何类型").grid(row=3, column=0, sticky="w")
        ttk.Label(parent, textvariable=self.kind_vars[side]).grid(row=3, column=1, columnspan=3, sticky="w")
        ttk.Label(parent, text="尺寸 · mm", foreground=color).grid(row=4, column=0, columnspan=4, sticky="w", pady=(7, 1))
        for index, (axis, variable) in enumerate(zip("XYZ", self.size_vars[side]), start=5):
            self._add_parameter_row(
                parent, index, axis, variable, 0.001, 500.0, 0.01,
                lambda selected=side: self._live_endpoint_changed(selected), unit="mm",
                control_key=(side, "size", index - 5), step_kind="linear",
            )

        ttk.Separator(parent).grid(row=8, column=0, columnspan=4, sticky="ew", pady=7)
        ttk.Label(parent, textvariable=self.pose_owner_vars[side], foreground="#596273").grid(row=9, column=0, columnspan=4, sticky="w")
        position_header = ttk.Frame(parent)
        position_header.grid(row=10, column=0, columnspan=4, sticky="ew", pady=(5, 1))
        ttk.Label(position_header, text="位置 · mm", foreground=color).pack(side=tk.LEFT)
        ttk.Button(position_header, text="恢复输入位置", command=lambda selected=side: self.reset_pose(selected, "position", False)).pack(side=tk.RIGHT)
        ttk.Button(position_header, text="位置归零", command=lambda selected=side: self.reset_pose(selected, "position", True)).pack(side=tk.RIGHT, padx=3)
        for index, (axis, variable) in enumerate(zip("XYZ", self.pose_vars[side]), start=11):
            self._add_parameter_row(
                parent, index, axis, variable, -1000.0, 1000.0, 0.01,
                lambda selected=side: self._live_endpoint_changed(selected), unit="mm",
                control_key=(side, "pose", index - 11), step_kind="linear",
            )
        rotation_header = ttk.Frame(parent)
        rotation_header.grid(row=14, column=0, columnspan=4, sticky="ew", pady=(5, 1))
        ttk.Label(rotation_header, text="旋转 · deg", foreground=color).pack(side=tk.LEFT)
        ttk.Button(rotation_header, text="恢复输入旋转", command=lambda selected=side: self.reset_pose(selected, "rotation", False)).pack(side=tk.RIGHT)
        ttk.Button(rotation_header, text="旋转归零", command=lambda selected=side: self.reset_pose(selected, "rotation", True)).pack(side=tk.RIGHT, padx=3)
        for index, (axis, variable) in enumerate(zip("RPY", self.rpy_vars[side]), start=15):
            self._add_parameter_row(
                parent, index, axis, variable, -360.0, 360.0, 0.01,
                lambda selected=side: self._live_endpoint_changed(selected), unit="°",
                control_key=(side, "rpy", index - 15), step_kind="angular",
            )
        ttk.Button(parent, text="应用键入值", command=lambda selected=side: self.apply_endpoint(selected)).grid(
            row=18, column=0, columnspan=4, sticky="e", pady=(8, 0)
        )

    def _add_parameter_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.DoubleVar,
        minimum: float,
        maximum: float,
        increment: float,
        callback: Callable[[], None],
        *,
        unit: str,
        control_key: tuple[str, str, int],
        step_kind: str,
    ) -> tuple[ttk.Scale, ttk.Spinbox]:
        ttk.Label(parent, text=label, width=7).grid(row=row, column=0, sticky="w")
        scale = ttk.Scale(
            parent,
            from_=minimum,
            to=maximum,
            variable=variable,
            command=lambda value, key=control_key: self._scale_changed(key, value, callback),
        )
        scale.grid(row=row, column=1, sticky="ew", padx=(3, 7), pady=1)
        spinbox = ttk.Spinbox(
            parent,
            from_=minimum,
            to=maximum,
            increment=increment,
            textvariable=variable,
            width=9,
            format="%.3f",
            command=lambda key=control_key: self._spinbox_committed(key, callback),
        )
        spinbox.grid(row=row, column=2, sticky="e")
        spinbox.bind("<Return>", lambda _event, key=control_key: self._spinbox_committed(key, callback))
        spinbox.bind("<FocusOut>", lambda _event, key=control_key: self._spinbox_committed(key, callback))
        ttk.Label(parent, text=unit, width=3).grid(row=row, column=3, sticky="w", padx=(3, 0))
        self.parameter_controls[control_key] = ParameterControl(
            scale=scale,
            spinbox=spinbox,
            variable=variable,
            step_kind=step_kind,
            hard_minimum=minimum,
            hard_maximum=maximum,
        )
        return scale, spinbox

    def _step_value(self, step_kind: str) -> float:
        variable = self.angular_step_var if step_kind == "angular" else self.linear_step_var
        try:
            step = float(variable.get())
        except ValueError:
            step = 0.01
        return max(step, 1e-6)

    def _precision_changed(self) -> None:
        self._suspend_live_controls = True
        try:
            for key, control in self.parameter_controls.items():
                step = self._step_value(control.step_kind)
                decimals = max(0, min(6, int(math.ceil(-math.log10(step))) + 1))
                control.spinbox.configure(increment=step, format=f"%.{decimals}f")
                self._center_parameter(key)
        finally:
            self._suspend_live_controls = False
        linear = self._step_value("linear")
        angular = self._step_value("angular")
        self.status_var.set(f"精调步长：线性 {linear:g} mm，旋转 {angular:g}°")

    def _center_parameter(self, key: tuple[str, str, int]) -> None:
        control = self.parameter_controls.get(key)
        if control is None:
            return
        try:
            center = float(control.variable.get())
        except (tk.TclError, ValueError):
            return
        step = self._step_value(control.step_kind)
        half_span = max(step * 100.0, step)
        minimum = max(control.hard_minimum, center - half_span)
        maximum = min(control.hard_maximum, center + half_span)
        if maximum <= minimum:
            minimum, maximum = control.hard_minimum, control.hard_maximum
        control.scale.configure(from_=minimum, to=maximum)

    def _center_controls(self, *, side: str | None = None) -> None:
        for key in self.parameter_controls:
            if side is None or key[0] in {side, "motion"}:
                self._center_parameter(key)

    def _scale_changed(
        self,
        key: tuple[str, str, int],
        raw_value: str,
        callback: Callable[[], None],
    ) -> None:
        if self._suspend_live_controls:
            return
        control = self.parameter_controls[key]
        step = self._step_value(control.step_kind)
        value = min(control.hard_maximum, max(control.hard_minimum, float(raw_value)))
        snapped = round(value / step) * step
        self._suspend_live_controls = True
        try:
            control.variable.set(round(snapped, 9))
        finally:
            self._suspend_live_controls = False
        callback()

    def _spinbox_committed(self, key: tuple[str, str, int], callback: Callable[[], None]) -> None:
        if self._suspend_live_controls:
            return
        callback()
        self._center_parameter(key)

    def _pose_default_key(self, endpoint: Endpoint) -> tuple[str, str, str]:
        if self.document is None:
            raise URDFValidationError("No URDF is loaded")
        target = self.document.pose_target(endpoint)
        return endpoint.driving_joint, target.label, target.owner_name

    def _capture_pose_defaults(self) -> None:
        if self.document is None:
            return
        self.pose_defaults.clear()
        for pair in self.pairs:
            for joint_name in (pair.joint_a, pair.joint_b):
                for endpoint in self.document.endpoint_candidates(joint_name):
                    key = self._pose_default_key(endpoint)
                    self.pose_defaults.setdefault(key, get_origin(self.document.pose_target(endpoint).element))

    def reset_pose(self, side: str, component: str, zero: bool) -> None:
        if self.document is None or self.pair is None:
            return
        self._commit_live_edit()
        endpoint = self.endpoints[side]
        other = self.endpoints["b" if side == "a" else "a"]
        if endpoint is None or other is None:
            return
        targets = [endpoint]
        if self.link_pose_var.get():
            targets.append(other)

        def operation() -> None:
            for target_endpoint in targets:
                pose_target = self.document.pose_target(target_endpoint)
                current_xyz, current_rpy = get_origin(pose_target.element)
                if zero:
                    default_xyz, default_rpy = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                else:
                    key = self._pose_default_key(target_endpoint)
                    if key not in self.pose_defaults:
                        raise URDFValidationError(f"没有找到 {pose_target.owner_name} 的输入位姿")
                    default_xyz, default_rpy = self.pose_defaults[key]
                xyz = default_xyz if component == "position" else current_xyz
                rpy = default_rpy if component == "rotation" else current_rpy
                set_origin(pose_target.element, xyz, rpy)
            if self.snap_contact_var.get():
                preview_q = self._preview_q()
                self.document.snap_contact_frame_to_inner_face(
                    self.pair, endpoint, other, preview_q, self.mesh_cache.dimensions
                )
                self.document.snap_contact_frame_to_inner_face(
                    self.pair, other, endpoint, preview_q, self.mesh_cache.dimensions
                )

        action = "归零" if zero else "恢复输入值"
        noun = "位置" if component == "position" else "旋转"
        self._transaction(f"末端 {side.upper()} {noun}{action}", operation, fit=False)

    def _set_preview(self, percent: float) -> None:
        self.preview_var.set(percent)
        self.refresh_scene()

    def _start_live_edit(self, description: str) -> None:
        if self.document is None or self._suspend_live_controls:
            return
        if self._live_before is None:
            self._live_before = self.document.snapshot()
        self._live_description = description
        if self._live_after_id is not None:
            self.root.after_cancel(self._live_after_id)
        self._live_after_id = self.root.after(450, self._commit_live_edit)

    def _commit_live_edit(self) -> None:
        if self.document is None or self._live_before is None:
            self._live_after_id = None
            return
        before = self._live_before
        description = self._live_description
        self._live_before = None
        self._live_after_id = None
        try:
            self.document.validate()
        except URDFValidationError as exc:
            self.document.restore(before)
            self._rebind_after_restore(fit=True)
            self.status_var.set(f"已回退无效调整：{exc}")
            return
        if before != self.document.snapshot():
            self.undo_stack.append((before, description))
            self.redo_stack.clear()
            self.dirty = True
            self.status_var.set(f"已调整：{description}")

    def _cancel_live_edit(self) -> None:
        if self._live_after_id is not None:
            self.root.after_cancel(self._live_after_id)
        self._live_after_id = None
        self._live_before = None

    def _rollback_live_edit(self, reason: Exception) -> None:
        if self.document is not None and self._live_before is not None:
            before = self._live_before
            self._cancel_live_edit()
            self.document.restore(before)
            self._rebind_after_restore(fit=False)
        self.status_var.set(f"该调整无效，已回退：{reason}")

    def _live_motion_changed(self) -> None:
        if self._suspend_live_controls or self.document is None or self.pair is None:
            return
        try:
            closed = float(self.closed_q_var.get()) / 1000.0
            opened = float(self.open_q_var.get()) / 1000.0
            self._start_live_edit("拖动夹爪开合范围")
            self.document.set_pair_limits(self.pair, closed, opened)
        except (tk.TclError, ValueError, URDFValidationError) as exc:
            self._rollback_live_edit(exc)
            return
        self.refresh_scene()

    def _live_endpoint_changed(self, side: str) -> None:
        if self._suspend_live_controls or self.document is None:
            return
        try:
            self._start_live_edit(f"拖动末端 {side.upper()} 参数")
            self._apply_endpoint_values(side)
        except (tk.TclError, ValueError, URDFValidationError) as exc:
            self._rollback_live_edit(exc)
            return
        self._sync_linked_field_values(side)
        self.refresh_scene()

    def _sync_linked_field_values(self, source_side: str) -> None:
        other_side = "b" if source_side == "a" else "a"
        self._suspend_live_controls = True
        try:
            if self.link_size_var.get():
                for source, target in zip(self.size_vars[source_side], self.size_vars[other_side]):
                    target.set(source.get())
            if self.link_pose_var.get():
                for source_group, target_group in (
                    (self.pose_vars[source_side], self.pose_vars[other_side]),
                    (self.rpy_vars[source_side], self.rpy_vars[other_side]),
                ):
                    for source, target in zip(source_group, target_group):
                        target.set(source.get())
        finally:
            self._suspend_live_controls = False

    def open_dialog(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择夹爪或整机 URDF",
            filetypes=(("URDF", "*.urdf"), ("XML", "*.xml"), ("All files", "*")),
        )
        if selected:
            self.load(selected)

    def load(self, path: str | Path) -> None:
        self._commit_live_edit()
        if self.dirty and not messagebox.askyesno("未导出修改", "当前修改尚未导出。仍要打开另一个 URDF 吗？"):
            return
        try:
            document = URDFDocument.load(path)
            pairs = document.detect_gripper_pairs()
            if not pairs:
                raise URDFValidationError("未找到两个平行 prismatic jaw joints；该工具当前面向双指平行夹爪")
        except URDFValidationError as exc:
            messagebox.showerror("无法打开 URDF", str(exc))
            return
        self.document, self.pairs = document, pairs
        self.path_var.set(str(document.source_path))
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.baseline_snapshot = document.snapshot()
        self.dirty = False
        self._capture_pose_defaults()
        self.pair_combo.configure(values=[pair.label for pair in pairs])
        self.pair_combo.current(0)
        self._pair_changed(fit=True)
        self.status_var.set(f"已载入 {document.source_path.name}；识别到 {len(pairs)} 对平行夹爪。")

    def _pair_changed(self, *, fit: bool = True) -> None:
        self._commit_live_edit()
        if self.document is None or self.pair is None:
            return
        pair = self.pair
        self._last_pair_key = (pair.joint_a, pair.joint_b)
        for side, joint_name in (("a", pair.joint_a), ("b", pair.joint_b)):
            options = self.document.endpoint_candidates(joint_name)
            default = self.document.default_endpoint(joint_name)
            self.endpoint_options[side] = options
            labels = [f"{option.link_name} · {option.geometry.label}" for option in options]
            self.endpoint_combos[side].configure(values=labels)
            default_index = next((index for index, option in enumerate(options) if geometry_key(option.geometry) == geometry_key(default.geometry)), 0)
            self.endpoint_combos[side].current(default_index)
            self.endpoints[side] = options[default_index]
        self._refresh_motion_fields()
        self._refresh_endpoint_fields()
        self.refresh_scene(fit=fit)

    def _endpoint_changed(self, side: str) -> None:
        self._commit_live_edit()
        selection = self.endpoint_combos[side].current()
        if 0 <= selection < len(self.endpoint_options[side]):
            self.endpoints[side] = self.endpoint_options[side][selection]
            self._refresh_endpoint_fields()
            self._refresh_motion_fields()
            self.refresh_scene()

    def _refresh_motion_fields(self) -> None:
        if self.document is None or self.pair is None or None in self.endpoints.values():
            return
        endpoint_a, endpoint_b = self.endpoints["a"], self.endpoints["b"]
        assert endpoint_a is not None and endpoint_b is not None
        closed, opened = self.document.inferred_open_closed(
            self.pair, endpoint_a, endpoint_b, self.mesh_cache.dimensions
        )
        self._suspend_live_controls = True
        try:
            self.closed_q_var.set(closed * 1000.0)
            self.open_q_var.set(opened * 1000.0)
        finally:
            self._suspend_live_controls = False
        self._center_parameter(("motion", "q", 0))
        self._center_parameter(("motion", "q", 1))

    def _refresh_endpoint_fields(self) -> None:
        if self.document is None:
            return
        self._suspend_live_controls = True
        try:
            for side in ("a", "b"):
                endpoint = self.endpoints[side]
                if endpoint is None:
                    continue
                try:
                    dimensions = self.document.geometry_dimensions(endpoint.geometry, self.mesh_cache.dimensions)
                except URDFValidationError:
                    dimensions = None
                if dimensions is not None:
                    size_values = _millimeters(dimensions)
                    for variable, value in zip(self.size_vars[side], size_values):
                        variable.set(value)
                pose_target = self.document.pose_target(endpoint)
                xyz, rpy = get_origin(pose_target.element)
                for variable, value in zip(self.pose_vars[side], _millimeters(xyz)):
                    variable.set(value)
                for variable, value in zip(self.rpy_vars[side], _degrees(rpy)):
                    variable.set(value)
                self.pose_owner_vars[side].set(f"位置来源: {pose_target.label} · {pose_target.owner_name}")
                self.kind_vars[side].set(endpoint.geometry.label)
        finally:
            self._suspend_live_controls = False
        self._center_controls()

    def _parse_fields(self, variables: list[tk.DoubleVar], scale: float, label: str) -> tuple[float, float, float]:
        try:
            values = tuple(float(variable.get()) * scale for variable in variables)
        except (tk.TclError, ValueError) as exc:
            raise URDFValidationError(f"{label} 必须是数值") from exc
        if not all(math.isfinite(value) for value in values):
            raise URDFValidationError(f"{label} 必须是有限数值")
        return values  # type: ignore[return-value]

    def _transaction(self, description: str, operation: Callable[[], None], *, fit: bool = False) -> None:
        if self.document is None:
            return
        self._commit_live_edit()
        before = self.document.snapshot()
        try:
            operation()
            self.document.validate()
        except (URDFValidationError, ValueError) as exc:
            self.document.restore(before)
            self._rebind_after_restore(fit=False)
            messagebox.showerror("修改未应用", str(exc))
            return
        self.undo_stack.append((before, description))
        self.redo_stack.clear()
        self.dirty = True
        self._rebind_after_restore(fit=fit)
        self.status_var.set(f"已应用：{description}")

    def apply_motion_range(self) -> None:
        if self.document is None or self.pair is None:
            return
        try:
            closed = float(self.closed_q_var.get()) / 1000.0
            opened = float(self.open_q_var.get()) / 1000.0
        except (tk.TclError, ValueError):
            messagebox.showerror("开合范围无效", "闭合 q 和张开 q 必须是毫米数值。")
            return
        pair = self.pair
        self._transaction("更新夹爪开合 joint limits", lambda: self.document.set_pair_limits(pair, closed, opened))

    def apply_endpoint(self, side: str) -> None:
        if self.document is None or self.pair is None:
            return
        self._commit_live_edit()
        self._transaction(f"更新末端 {side.upper()} 尺寸/位姿", lambda: self._apply_endpoint_values(side), fit=True)

    def _apply_endpoint_values(self, side: str) -> None:
        if self.document is None or self.pair is None:
            return
        endpoint = self.endpoints[side]
        other_side = "b" if side == "a" else "a"
        other = self.endpoints[other_side]
        if endpoint is None or other is None:
            return
        dimensions = self._parse_fields(self.size_vars[side], 0.001, "尺寸")
        xyz = self._parse_fields(self.pose_vars[side], 0.001, "位置")
        rpy = self._parse_fields(self.rpy_vars[side], math.pi / 180.0, "旋转")
        pair = self.pair
        preview_q = self._preview_q()
        targets = [endpoint]
        if self.link_size_var.get():
            targets.append(other)
        for target in targets:
            matching = (
                self.document.matching_geometry(target.geometry, self.mesh_cache.dimensions)
                if self.sync_geometry_var.get()
                else None
            )
            self.document.set_geometry_dimensions(
                target.geometry,
                dimensions,
                self.mesh_cache.dimensions,
                self.mesh_cache.center,
            )
            if matching is not None:
                self.document.set_geometry_dimensions(
                    matching,
                    dimensions,
                    self.mesh_cache.dimensions,
                    self.mesh_cache.center,
                )
        pose_targets = [endpoint]
        if self.link_pose_var.get():
            pose_targets.append(other)
        for target in pose_targets:
            set_origin(self.document.pose_target(target).element, xyz, rpy)
        if self.snap_contact_var.get():
            self.document.snap_contact_frame_to_inner_face(pair, endpoint, other, preview_q, self.mesh_cache.dimensions)
            self.document.snap_contact_frame_to_inner_face(pair, other, endpoint, preview_q, self.mesh_cache.dimensions)

    def _preview_q(self) -> float:
        try:
            closed = float(self.closed_q_var.get()) / 1000.0
            opened = float(self.open_q_var.get()) / 1000.0
        except (tk.TclError, ValueError):
            return 0.0
        fraction = min(1.0, max(0.0, self.preview_var.get() / 100.0))
        return closed + (opened - closed) * fraction

    def refresh_scene(self, _unused: object = None, *, fit: bool = False) -> None:
        if self.document is None or self.pair is None:
            self.viewer.set_scene([], [], fit=False)
            return
        endpoint_a, endpoint_b = self.endpoints["a"], self.endpoints["b"]
        if endpoint_a is None or endpoint_b is None:
            return
        q = self._preview_q()
        try:
            gap = self.document.opening_gap(self.pair, endpoint_a, endpoint_b, q, self.mesh_cache.dimensions)
            shapes, markers, warnings = build_gripper_scene(
                self.document,
                self.pair,
                endpoint_a,
                endpoint_b,
                q,
                self.mesh_cache,
                show_visuals=self.show_visuals_var.get(),
                show_collisions=self.show_collisions_var.get(),
                show_contact_frames=self.show_contacts_var.get(),
            )
        except (URDFValidationError, OSError) as exc:
            self.status_var.set(f"预览警告：{exc}")
            return
        self.preview_label_var.set(f"Preview {self.preview_var.get():.0f}% · q={q * 1000.0:.3f} mm")
        if gap is None:
            self.gap_label_var.set("开口: 无法计算")
        else:
            self.gap_label_var.set(f"内侧开口: {gap * 1000.0:.3f} mm" + (" ⚠ 穿透" if gap < -1e-6 else ""))
        if warnings:
            self.status_var.set("预览略过部分网格：" + "；".join(dict.fromkeys(warnings)))
        self.viewer.set_scene(shapes, markers, fit=fit)
        if fit:
            # Initial load can happen before Tk has assigned the canvas its real
            # size. Fit once more after layout so the gripper fills the viewport.
            self.root.after_idle(self.viewer.fit)

    def viewer_fit(self) -> None:
        self.refresh_scene(fit=True)

    def _rebind_after_restore(self, *, fit: bool = False) -> None:
        if self.document is None:
            return
        previous_pair_key = self._last_pair_key
        previous_geometry_keys = {
            side: None if self.endpoints[side] is None else geometry_key(self.endpoints[side].geometry)
            for side in ("a", "b")
        }
        self.pairs = self.document.detect_gripper_pairs()
        self.pair_combo.configure(values=[pair.label for pair in self.pairs])
        pair_index = next(
            (index for index, pair in enumerate(self.pairs) if (pair.joint_a, pair.joint_b) == previous_pair_key),
            0,
        )
        self.pair_combo.current(pair_index)
        pair = self.pair
        if pair is None:
            return
        self._last_pair_key = (pair.joint_a, pair.joint_b)
        for side, joint_name in (("a", pair.joint_a), ("b", pair.joint_b)):
            options = self.document.endpoint_candidates(joint_name)
            self.endpoint_options[side] = options
            self.endpoint_combos[side].configure(values=[f"{option.link_name} · {option.geometry.label}" for option in options])
            selected_key = previous_geometry_keys[side]
            selected_index = next((index for index, option in enumerate(options) if geometry_key(option.geometry) == selected_key), 0)
            self.endpoint_combos[side].current(selected_index)
            self.endpoints[side] = options[selected_index]
        self._refresh_motion_fields()
        self._refresh_endpoint_fields()
        self.refresh_scene(fit=fit)

    def undo(self) -> None:
        self._commit_live_edit()
        if self.document is None or not self.undo_stack:
            return
        current = self.document.snapshot()
        snapshot, description = self.undo_stack.pop()
        self.redo_stack.append((current, description))
        self.document.restore(snapshot)
        self.dirty = self.baseline_snapshot != self.document.snapshot()
        self._rebind_after_restore(fit=True)
        self.status_var.set(f"已撤销：{description}")

    def redo(self) -> None:
        self._commit_live_edit()
        if self.document is None or not self.redo_stack:
            return
        current = self.document.snapshot()
        snapshot, description = self.redo_stack.pop()
        self.undo_stack.append((current, description))
        self.document.restore(snapshot)
        self.dirty = True
        self._rebind_after_restore(fit=True)
        self.status_var.set(f"已重做：{description}")

    def revert(self) -> None:
        self._commit_live_edit()
        if self.document is None or self.baseline_snapshot is None or not self.dirty:
            return
        if not messagebox.askyesno("回到输入版本", "放弃当前尚未导出的编辑并回到输入 URDF 吗？"):
            return
        self.document.restore(self.baseline_snapshot)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.dirty = False
        self._rebind_after_restore(fit=True)
        self.status_var.set("已回到输入 URDF；磁盘上的输入文件始终未被修改。")

    def export_as(self) -> None:
        self._commit_live_edit()
        if self.document is None:
            return
        source = self.document.source_path
        default_name = f"{source.stem}_gripper_edited.urdf"
        selected = filedialog.asksaveasfilename(
            title="导出新的 URDF",
            initialdir=source.parent,
            initialfile=default_name,
            defaultextension=".urdf",
            filetypes=(("URDF", "*.urdf"), ("XML", "*.xml"), ("All files", "*")),
        )
        if not selected:
            return
        target = Path(selected).expanduser().resolve()
        overwrite = False
        if target.exists():
            overwrite = messagebox.askyesno("替换已有导出文件", f"目标已存在：\n{target}\n\n确认替换这个导出文件吗？")
            if not overwrite:
                return
        try:
            output = self.document.export(target, overwrite=overwrite, rebase_mesh_paths=True)
        except URDFValidationError as exc:
            messagebox.showerror("导出失败", str(exc))
            return
        self.dirty = False
        self.status_var.set(f"已原子导出并复验：{output}")
        messagebox.showinfo("导出完成", f"新 URDF 已导出并通过结构校验：\n{output}\n\n输入文件未修改。")

    def _close(self) -> None:
        self._commit_live_edit()
        if self.dirty and not messagebox.askyesno("未导出修改", "当前修改尚未导出。确认退出吗？"):
            return
        self.root.destroy()


def launch(input_path: str | Path | None = None) -> None:
    root = tk.Tk()
    GripperURDFEditorApp(root, input_path=input_path)
    root.mainloop()
