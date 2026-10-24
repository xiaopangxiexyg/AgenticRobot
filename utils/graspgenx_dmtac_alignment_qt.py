"""Modern native desktop GUI for the GraspGenX/DMTAC alignment preview.

This is the preferred local GUI. It shows the installed DMTAC assembly and
current grasp model only. Legacy 4B2 state remains in the shared JSON reader
for compatibility, but has no controls, proxy, markers or metrics here.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import QPoint, QPointF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .graspgenx_dmtac_alignment_gui import (
    DEFAULT_GRASP_CENTER_FROM_TACTILE_Z_M,
    DEFAULT_MODEL_CORRECTION_M,
    DEFAULT_PALM_Z_M,
    AlignmentModel,
    AlignmentState,
    current_dmtac_state,
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID,
    _frame_angle,
    _mat4_translation,
    _parallel_jaw_triangles,
    _state_from_payload,
    _transform_triangles,
    load_tool_tcp_pose,
)
from .gripper_urdf_editor.core import (
    Mat4,
    URDFValidationError,
    transform_point,
)
from .gripper_urdf_editor.mesh import MeshCache

DEFAULT_URDF = os.environ.get("DMTAC_ALIGNMENT_URDF", "")
AXIS_COLORS = ("#ef5350", "#51c878", "#5b8ff9")
FRAME_COLORS = {
    "rm65_arm_endpoint": "#f1f5f9",
    "real_flange": "#f5a623",
    "tool_tcp": "#ef4b9b",
    "dmtac_grasp_center": "#9be15d",
    "graspgenx_model": "#c97b30",
    "graspgenx_sweep_center": "#f1c75b",
    "dmtac_tactile": "#16c6d8",
    "dmtac_sensing_0": "#72e8ed",
    "dmtac_sensing_1": "#72e8ed",
}
FRAME_LABELS = {
    "rm65_arm_endpoint": "RM65 solver endpoint",
    "real_flange": "真实安装法兰",
    "tool_tcp": "控制 TCP",
    "dmtac_grasp_center": "已确认抓取中心",
    "graspgenx_model": "DMTAC 对齐 GraspGenX",
    "graspgenx_sweep_center": "GraspGenX 扫掠中心",
    "dmtac_tactile": "DMTAC tactile",
    "dmtac_sensing_0": "DMTAC sensing 1",
    "dmtac_sensing_1": "DMTAC sensing 2",
}




def _cli_value(name: str, default: str = "") -> str:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return default
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else default


def _parse_qpos(value: str) -> list[float]:
    qpos = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(qpos) != 6:
        raise ValueError("--arm-qpos requires six comma-separated values")
    return qpos


@dataclass(frozen=True)
class RenderTriangle:
    points: tuple[tuple[float, float, float], ...]
    color: str
    alpha: int
    wire: bool = False


@dataclass(frozen=True)
class RenderLine:
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    color: str
    label: str


@dataclass(frozen=True)
class RenderMarker:
    position: tuple[float, float, float]
    color: str
    label: str


class ComponentSceneBuilder:
    """Caches only the small set of end-tool geometry used by the alignment GUI."""

    def __init__(self, model: AlignmentModel) -> None:
        self.model = model
        self.cached_geometry: list[tuple[object, tuple, str, bool]] = []
        self.warnings: list[str] = []
        self.rebuild_geometry_cache()

    def _links(self, include_cad: bool, include_dmtac: bool) -> set[str]:
        side = self.model.state.side
        prefix = f"{side}_"
        root = f"{prefix}arm_base_link"
        links = (
            {name for name, _depth in self.model.document.subtree_links(root)}
            if root in self.model.document.links
            else set(self.model.document.links)
        )
        flange = self.model.frame_links().get("real_flange")
        selected: set[str] = set()
        for name in links:
            low = name.lower()
            if name == flange or "realman_to" in low or "flange" in low:
                selected.add(name)
                continue
            if any(token in low for token in ("camera", "realsense", "arm_link", "arm_base", "arm_joint")):
                continue
            is_dmtac = any(
                token in low
                for token in ("dm_tac", "dmtac", "tactile", "w2m", "sensor", "sensing", "contact")
            )
            is_cad = any(
                token in low
                for token in ("eg2_4b2_cad", "gripper", "legacy_gripper", "hand", "finger", "palm", "pad")
            )
            if (is_dmtac and include_dmtac) or (is_cad and include_cad):
                selected.add(name)
        return selected

    @staticmethod
    def _color(link: str, role: str) -> tuple[str, bool]:
        low = link.lower()
        if "flange" in low or "realman_to" in low:
            return "#f5a623", False
        if any(token in low for token in ("dm_tac", "dmtac", "tactile", "w2m", "sensor", "sensing", "contact")):
            return "#079ab0", False
        if role == "collision":
            return "#718096", True
        return "#94a3b8", False

    def rebuild_geometry_cache(self) -> None:
        self.cached_geometry.clear()
        self.warnings.clear()
        selected = self._links(include_cad=True, include_dmtac=True)
        for link in selected:
            for ref in self.model.document.geometry_refs(link):
                try:
                    triangles = self.model.mesh_cache.geometry_triangles(self.model.document, ref)
                except URDFValidationError as exc:
                    self.warnings.append(str(exc))
                    continue
                if triangles:
                    color, wire = self._color(link, ref.role)
                    self.cached_geometry.append((ref, triangles, color, wire))

    def build(
        self,
        *,
        show_cad: bool,
        show_dmtac: bool,
        show_collision: bool,
        show_graspgenx: bool,
    ) -> tuple[list[RenderTriangle], list[RenderLine], list[RenderMarker], dict[str, Mat4], dict[str, object]]:
        frames, details = self.model.frames()
        transforms = details["transforms"]  # type: ignore[assignment]
        active_links = self._links(show_cad, show_dmtac)
        triangles: list[RenderTriangle] = []
        for ref, local_triangles, color, wire in self.cached_geometry:
            if ref.link_name not in active_links:
                continue
            if ref.role == "collision" and not show_collision:
                continue
            geometry = self.model.document.geometry_transform(ref, transforms)
            triangles.extend(
                RenderTriangle(transformed, color, 52 if wire else 185, wire)
                for transformed in _transform_triangles(geometry, local_triangles)
            )

        if show_graspgenx:
            state = self.model.state
            triangles.extend(
                RenderTriangle(transformed, "#c97b30", 110, True)
                for transformed in _parallel_jaw_triangles(
                    frames["graspgenx_model"],
                    state,
                    palm_z_m=state.model_palm_z_m,
                )
            )
        lines: list[RenderLine] = []
        markers: list[RenderMarker] = []
        visible_frames = [
            "rm65_arm_endpoint",
            "real_flange",
            "tool_tcp",
            "dmtac_tactile",
            "dmtac_sensing_0",
            "dmtac_sensing_1",
            "dmtac_grasp_center",
        ]
        if show_graspgenx:
            visible_frames.extend(("graspgenx_model", "graspgenx_sweep_center"))
        for name in visible_frames:
            frame = frames.get(name)
            if frame is None:
                continue
            origin = _mat4_translation(frame)
            markers.append(RenderMarker(origin, FRAME_COLORS[name], FRAME_LABELS[name]))
            for axis, color in enumerate(AXIS_COLORS):
                tip = transform_point(frame, tuple(0.035 if index == axis else 0.0 for index in range(3)))
                lines.append(RenderLine(origin, tip, color, f"{FRAME_LABELS[name]} {'XYZ'[axis]}"))
        return triangles, lines, markers, frames, details


class SceneView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(720, 620)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.yaw = math.radians(-48.0)
        self.pitch = math.radians(24.0)
        self.zoom = 4500.0
        self.center = (0.0, 0.0, 0.08)
        self.triangles: list[RenderTriangle] = []
        self.lines: list[RenderLine] = []
        self.markers: list[RenderMarker] = []
        self._last_mouse: QPoint | None = None
        self._panning = False

    def set_scene(
        self,
        triangles: list[RenderTriangle],
        lines: list[RenderLine],
        markers: list[RenderMarker],
        *,
        fit: bool = False,
    ) -> None:
        self.triangles, self.lines, self.markers = triangles, lines, markers
        if fit:
            self.fit_scene()
        self.update()

    def _basis(self) -> tuple[tuple[float, float, float], ...]:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        return (
            (-sy, cy, 0.0),
            (-sp * cy, -sp * sy, cp),
            (cp * cy, cp * sy, sp),
        )

    def _project(self, point: Sequence[float]) -> tuple[float, float, float]:
        right, up, forward = self._basis()
        delta = tuple(float(point[index]) - self.center[index] for index in range(3))
        return (
            self.width() / 2.0 + sum(delta[i] * right[i] for i in range(3)) * self.zoom,
            self.height() / 2.0 - sum(delta[i] * up[i] for i in range(3)) * self.zoom,
            sum(delta[i] * forward[i] for i in range(3)),
        )

    def fit_scene(self) -> None:
        points = [point for item in self.triangles for point in item.points]
        points.extend(item.start for item in self.lines)
        points.extend(item.end for item in self.lines)
        points.extend(item.position for item in self.markers)
        if not points:
            return
        minimum = [min(point[index] for point in points) for index in range(3)]
        maximum = [max(point[index] for point in points) for index in range(3)]
        self.center = tuple((minimum[index] + maximum[index]) / 2.0 for index in range(3))
        extent = max(maximum[index] - minimum[index] for index in range(3))
        self.zoom = min(self.width(), self.height()) * 0.72 / max(extent, 0.01)
        self.update()

    def mousePressEvent(self, event) -> None:
        self._last_mouse = event.pos()
        self._panning = event.button() in (Qt.MiddleButton, Qt.RightButton)

    def mouseMoveEvent(self, event) -> None:
        if self._last_mouse is None:
            return
        delta = event.pos() - self._last_mouse
        self._last_mouse = event.pos()
        if self._panning:
            right, up, _forward = self._basis()
            scale = 1.0 / max(self.zoom, 1.0)
            self.center = tuple(
                self.center[index] - right[index] * delta.x() * scale + up[index] * delta.y() * scale
                for index in range(3)
            )
        else:
            self.yaw += delta.x() * 0.008
            self.pitch = max(math.radians(-89.0), min(math.radians(89.0), self.pitch - delta.y() * 0.008))
        self.update()

    def wheelEvent(self, event) -> None:
        self.zoom = max(100.0, min(100000.0, self.zoom * (1.12 if event.angleDelta().y() > 0 else 1.0 / 1.12)))
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#101722"))
        projected: list[tuple[float, RenderTriangle, list[QPointF]]] = []
        for triangle in self.triangles:
            values = [self._project(point) for point in triangle.points]
            projected.append((sum(value[2] for value in values) / 3.0, triangle, [QPointF(value[0], value[1]) for value in values]))
        for _depth, triangle, polygon in sorted(projected, key=lambda item: item[0], reverse=True):
            color = QColor(triangle.color)
            color.setAlpha(triangle.alpha)
            painter.setBrush(color)
            pen = QPen(QColor("#d8e2f0") if triangle.wire else QColor(triangle.color))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawPolygon(QPolygonF(polygon))
        for line in self.lines:
            start = self._project(line.start)
            end = self._project(line.end)
            pen = QPen(QColor(line.color))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(QPointF(start[0], start[1]), QPointF(end[0], end[1]))
        painter.setFont(QFont("Sans", 9, QFont.Bold))
        for marker in self.markers:
            x, y, _depth = self._project(marker.position)
            painter.setBrush(QColor(marker.color))
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawEllipse(QPointF(x, y), 4.5, 4.5)
            painter.setPen(QPen(QColor(marker.color)))
            painter.drawText(QPointF(x + 8, y - 8), marker.label)
        painter.setPen(QPen(QColor("#aeb9cb")))
        painter.setFont(QFont("Sans", 9))
        painter.drawText(14, 22, "左键旋转 · 右键/中键平移 · 滚轮缩放")


class AlignmentWindow(QMainWindow):
    def __init__(self, model: AlignmentModel, *,
                 alignment_profile_id: str = DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID) -> None:
        super().__init__()
        self.model = model
        self.alignment_profile_id = alignment_profile_id
        self.builder = ComponentSceneBuilder(model)
        self.controls: dict[str, QDoubleSpinBox] = {}
        self.updating = False
        self.setWindowTitle("GraspGenX ↔ DMTAC 坐标对齐 · native")
        self.resize(1540, 960)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #111827; color: #e5e7eb; }
            QGroupBox { border: 1px solid #334155; border-radius: 6px; margin-top: 10px; padding: 10px; font-size: 13px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #93c5fd; }
            QLabel { font-size: 13px; }
            QDoubleSpinBox, QComboBox, QLineEdit { background: #1f2937; border: 1px solid #475569; padding: 5px; min-height: 30px; color: #f8fafc; font-size: 13px; }
            QPushButton { background: #26364d; border: 1px solid #4b6585; border-radius: 4px; padding: 7px 10px; min-height: 34px; font-size: 13px; }
            QPushButton:hover { background: #365476; }
            QCheckBox { padding: 5px; min-height: 28px; font-size: 13px; }
            QScrollArea { border: none; }
            """
        )
        self._build()
        self._apply_state_to_controls()
        self.refresh(fit=True)

    def _spin(
        self,
        form: QFormLayout,
        label: str,
        name: str,
        value: float,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int = 3,
    ) -> None:
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setSingleStep(step)
        box.setDecimals(decimals)
        box.setValue(value)
        box.valueChanged.connect(self.refresh)
        form.addRow(label, box)
        self.controls[name] = box

    def _build(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        panel = QWidget()
        panel.setMinimumWidth(420)
        panel.setMaximumWidth(500)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(420)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        scroll.setWidget(panel)
        splitter.addWidget(scroll)
        self.view = SceneView()
        splitter.addWidget(self.view)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(1, 1)

        toolbar = QGridLayout()
        toolbar.setHorizontalSpacing(8)
        toolbar.setVerticalSpacing(8)
        save_button = QPushButton("保存 alignment JSON")
        save_button.clicked.connect(self.save_alignment)
        load_button = QPushButton("读取 alignment JSON")
        load_button.clicked.connect(self.load_alignment)
        reset_button = QPushButton("重置名义值")
        reset_button.clicked.connect(self.reset_defaults)
        fit_button = QPushButton("适配视图")
        fit_button.clicked.connect(self.view.fit_scene)
        for button in (save_button, load_button, reset_button, fit_button):
            button.setMinimumWidth(170)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar.addWidget(save_button, 0, 0)
        toolbar.addWidget(load_button, 0, 1)
        toolbar.addWidget(reset_button, 1, 0)
        toolbar.addWidget(fit_button, 1, 1)
        panel_layout.addLayout(toolbar)

        state_group = QGroupBox("离线状态")
        state_form = QFormLayout(state_group)
        side = QComboBox()
        side.addItems(["right", "left"])
        side.currentTextChanged.connect(self.side_changed)
        self.side_combo = side
        state_form.addRow("机械臂侧", side)
        self._spin(state_form, "DMTAC master q", "gripper_q", 0.0, -10.0, 10.0, 0.001, 4)
        for index in range(1, 7):
            self._spin(state_form, f"RM65 joint {index} (rad)", f"q{index}", 0.0, -7.0, 7.0, 0.001, 5)
        self._spin(state_form, "TCP ← RM65 endpoint Z (mm)", "tcp_z_mm", 140.0, -200.0, 300.0, 0.1, 2)
        panel_layout.addWidget(state_group)

        model_group = QGroupBox("当前 DMTAC 抓姿模型")
        model_form = QFormLayout(model_group)
        self._spin(model_form, "最大开口 (mm)", "opening_mm", 70.0, 1.0, 200.0, 0.1, 3)
        self._spin(model_form, "model palm Z (mm)", "palm_z_mm", DEFAULT_PALM_Z_M * 1000.0, -200.0, 200.0, 0.1, 3)
        self._spin(model_form, "finger length (mm)", "finger_length_mm", 50.0, 1.0, 200.0, 0.1, 3)
        self._spin(model_form, "finger thickness (mm)", "finger_thickness_mm", 10.0, 1.0, 100.0, 0.1, 3)
        self._spin(model_form, "pad height (mm)", "pad_height_mm", 25.5, 1.0, 100.0, 0.1, 3)
        self._spin(model_form, "抓取中心 ← tactile X (mm)", "grasp_center_x_mm", 0.0, -100.0, 100.0, 0.1, 3)
        self._spin(model_form, "抓取中心 ← tactile Y (mm)", "grasp_center_y_mm", 0.0, -100.0, 100.0, 0.1, 3)
        self._spin(model_form, "抓取中心 ← tactile Z (mm)", "grasp_center_z_mm", DEFAULT_GRASP_CENTER_FROM_TACTILE_Z_M * 1000.0, -100.0, 100.0, 0.1, 3)
        panel_layout.addWidget(model_group)

        correction_group = QGroupBox("GraspGenX base 局部微调")
        correction_form = QFormLayout(correction_group)
        for index, (label, name) in enumerate((("X (mm)", "corr_x_mm"), ("Y (mm)", "corr_y_mm"), ("Z (mm)", "corr_z_mm"))):
            self._spin(correction_form, label, name, DEFAULT_MODEL_CORRECTION_M[index] * 1000.0, -100.0, 100.0, 0.1, 3)
        for label, name in (("roll (deg)", "corr_roll_deg"), ("pitch (deg)", "corr_pitch_deg"), ("yaw (deg)", "corr_yaw_deg")):
            self._spin(correction_form, label, name, 0.0, -180.0, 180.0, 0.1, 2)
        panel_layout.addWidget(correction_group)

        display_group = QGroupBox("显示组件")
        display_layout = QVBoxLayout(display_group)
        self.show_cad = QCheckBox("当前 DMTAC 夹爪机构")
        self.show_dmtac = QCheckBox("DMTAC 触觉传感器")
        self.show_collision = QCheckBox("URDF collision")
        self.show_graspgenx = QCheckBox("当前 DMTAC 抓姿立方体")
        self.show_cad.setChecked(True)
        self.show_dmtac.setChecked(True)
        self.show_graspgenx.setChecked(True)
        for checkbox in (self.show_cad, self.show_dmtac, self.show_collision, self.show_graspgenx):
            checkbox.stateChanged.connect(self.refresh)
            display_layout.addWidget(checkbox)
        panel_layout.addWidget(display_group)

        self.metric_label = QLabel()
        self.metric_label.setWordWrap(True)
        self.metric_label.setStyleSheet("color: #8bd5ff; padding: 6px;")
        panel_layout.addWidget(self.metric_label)
        self.status_label = QLabel("离线预览；不连接真机。")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #f4b56a; padding: 6px;")
        panel_layout.addWidget(self.status_label)
        panel_layout.addStretch(1)

        legend = QLabel("白色=RM65 endpoint · 橙色=安装法兰 · 粉色=控制 TCP · 青色=DMTAC · 绿色=抓取中心 · 棕色=DMTAC 抓姿模型 · 黄色小框=扫掠中心 · RGB=局部 XYZ")
        legend.setStyleSheet("color: #9caec4; padding: 6px;")
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(splitter)
        root_layout.addWidget(legend)
        self.setCentralWidget(root)
        splitter.setSizes([450, 1090])

    def _apply_state_to_controls(self) -> None:
        state = self.model.state
        values = {
            "gripper_q": state.gripper_q,
            "tcp_z_mm": state.tcp_z_m * 1000.0,
            "opening_mm": state.model_opening_m * 1000.0,
            "palm_z_mm": state.model_palm_z_m * 1000.0,
            "finger_length_mm": state.model_finger_length_m * 1000.0,
            "finger_thickness_mm": state.model_finger_thickness_m * 1000.0,
            "pad_height_mm": state.model_pad_height_m * 1000.0,
            "grasp_center_x_mm": state.grasp_center_from_tactile_x_m * 1000.0,
            "grasp_center_y_mm": state.grasp_center_from_tactile_y_m * 1000.0,
            "grasp_center_z_mm": state.grasp_center_from_tactile_z_m * 1000.0,
            "corr_x_mm": (state.model_correction_xyz_m or [0.0] * 3)[0] * 1000.0,
            "corr_y_mm": (state.model_correction_xyz_m or [0.0] * 3)[1] * 1000.0,
            "corr_z_mm": (state.model_correction_xyz_m or [0.0] * 3)[2] * 1000.0,
            "corr_roll_deg": math.degrees((state.model_correction_rpy_rad or [0.0] * 3)[0]),
            "corr_pitch_deg": math.degrees((state.model_correction_rpy_rad or [0.0] * 3)[1]),
            "corr_yaw_deg": math.degrees((state.model_correction_rpy_rad or [0.0] * 3)[2]),
        }
        for index, value in enumerate(state.arm_qpos or [0.0] * 6, start=1):
            values[f"q{index}"] = value
        self.updating = True
        self.side_combo.setCurrentText(state.side)
        for name, value in values.items():
            self.controls[name].setValue(value)
        self.updating = False

    def _read_controls(self) -> None:
        state = self.model.state
        state.side = self.side_combo.currentText()
        state.gripper_q = self.controls["gripper_q"].value()
        state.arm_qpos = [self.controls[f"q{index}"].value() for index in range(1, 7)]
        state.tcp_z_m = self.controls["tcp_z_mm"].value() / 1000.0
        state.model_opening_m = self.controls["opening_mm"].value() / 1000.0
        state.model_palm_z_m = self.controls["palm_z_mm"].value() / 1000.0
        state.model_finger_length_m = self.controls["finger_length_mm"].value() / 1000.0
        state.model_finger_thickness_m = self.controls["finger_thickness_mm"].value() / 1000.0
        state.model_pad_height_m = self.controls["pad_height_mm"].value() / 1000.0
        state.grasp_center_from_tactile_x_m = self.controls["grasp_center_x_mm"].value() / 1000.0
        state.grasp_center_from_tactile_y_m = self.controls["grasp_center_y_mm"].value() / 1000.0
        state.grasp_center_from_tactile_z_m = self.controls["grasp_center_z_mm"].value() / 1000.0
        state.model_correction_xyz_m = [self.controls[name].value() / 1000.0 for name in ("corr_x_mm", "corr_y_mm", "corr_z_mm")]
        state.model_correction_rpy_rad = [math.radians(self.controls[name].value()) for name in ("corr_roll_deg", "corr_pitch_deg", "corr_yaw_deg")]

    def side_changed(self, _side: str) -> None:
        if self.updating:
            return
        self._read_controls()
        self.model._refresh_pair()
        self.builder.rebuild_geometry_cache()
        self.refresh(fit=True)

    def refresh(self, *_args, fit: bool = False) -> None:
        if self.updating or not hasattr(self, "view"):
            return
        self._read_controls()
        self.model._refresh_pair()
        triangles, lines, markers, frames, details = self.builder.build(
            show_cad=self.show_cad.isChecked(),
            show_dmtac=self.show_dmtac.isChecked(),
            show_collision=self.show_collision.isChecked(),
            show_graspgenx=self.show_graspgenx.isChecked(),
        )
        self.view.set_scene(triangles, lines, markers, fit=fit)
        width = details.get("sensor_width_m")
        distance = math.dist(
            _mat4_translation(frames["graspgenx_sweep_center"]),
            _mat4_translation(frames["dmtac_grasp_center"]),
        )
        angle = _frame_angle(frames["graspgenx_sweep_center"], frames["dmtac_grasp_center"])
        tcp_distance = math.dist(
            _mat4_translation(frames["tool_tcp"]),
            _mat4_translation(frames["dmtac_tactile"]),
        )
        width_text = "—" if width is None else f"{width * 1000.0:.2f}"
        self.metric_label.setText(
            f"DMTAC sensing-origin separation: {width_text} mm\n"
            f"控制 TCP ↔ DMTAC: {tcp_distance * 1000.0:.2f} mm\n"
            f"GraspGenX 扫掠中心 ↔ 已确认抓取中心: {distance * 1000.0:.2f} mm / {angle:.2f}°"
        )
        warnings = self.builder.warnings + list(details.get("warnings", []))
        self.status_label.setText("；".join(warnings) if warnings else "离线预览正常；未连接真机。")

    def reset_defaults(self) -> None:
        side = self.side_combo.currentText()
        tcp_pose = list(self.model.state.tool_tcp_pose_endpoint_m_rad or ())
        self.model.state = current_dmtac_state(side,
            tcp_pose if len(tcp_pose) == 6 else [0., 0., .140, 0., 0., 0.], [0.] * 6,
            alignment_profile_id=self.alignment_profile_id)
        self._apply_state_to_controls()
        self.model._refresh_pair()
        self.builder = ComponentSceneBuilder(self.model)
        self.refresh(fit=True)

    def save_alignment(self) -> None:
        self._read_controls()
        target, _filter = QFileDialog.getSaveFileName(self, "保存 alignment JSON", f"graspgenx-dmtac-alignment-{self.model.state.side}.json", "JSON (*.json)")
        if not target:
            return
        try:
            Path(target).write_text(json.dumps(self.model.payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.status_label.setText(f"已保存：{target}")
        except OSError as exc:
            QMessageBox.critical(self, "保存失败", str(exc))

    def load_alignment(self) -> None:
        source, _filter = QFileDialog.getOpenFileName(self, "读取 alignment JSON", "", "JSON (*.json)")
        if not source:
            return
        try:
            payload = json.loads(Path(source).read_text(encoding="utf-8"))
            state = _state_from_payload(payload)
            self.model.state = state
            self._apply_state_to_controls()
            self.model._refresh_pair()
            self.builder = ComponentSceneBuilder(self.model)
            self.refresh(fit=True)
            self.status_label.setText(f"已读取：{source}")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "读取失败", f"{type(exc).__name__}: {exc}")


def main() -> int:
    try:
        urdf_value = _cli_value("--urdf", DEFAULT_URDF)
        qpos_value = _cli_value("--arm-qpos", "0,0,0,0,0,0")
        side = _cli_value("--side", "right")
        if not urdf_value:
            raise ValueError("set DMTAC_ALIGNMENT_URDF or pass --urdf")
        urdf = Path(urdf_value).expanduser()
        if not urdf.is_file():
            raise FileNotFoundError(f"URDF does not exist: {urdf}")
        profile_value = _cli_value("--tool-profile", "")
        if profile_value:
            tool_profile = Path(profile_value).expanduser()
        else:
            tool_profile = (
                Path(__file__).resolve().parents[1]
                / "hardware/rm65/utils/configs/calibrations"
                / f"rm65_{side}_DMTAC_M.json"
            )
        tcp_pose = load_tool_tcp_pose(tool_profile)
        alignment_profile_id = _cli_value("--alignment-profile", DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID)
        state = current_dmtac_state(side, tcp_pose, _parse_qpos(qpos_value),
            alignment_profile_id=alignment_profile_id)
        app = QApplication(sys.argv)
        app.setApplicationName("GraspGenX DMTAC Alignment")
        app.setStyle("Fusion")
        app_font = QFont("Sans")
        app_font.setPointSize(11)
        app.setFont(app_font)
        model = AlignmentModel(urdf, state)
        # The native diagnostic view is intentionally decimated.  This affects
        # only display triangles, never the URDF or production collision model.
        model.mesh_cache = MeshCache(max_triangles=320)
        window = AlignmentWindow(model, alignment_profile_id=alignment_profile_id)
        window.show()
        return app.exec_()
    except (OSError, ValueError, URDFValidationError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
