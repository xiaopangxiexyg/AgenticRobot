from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from typing import Sequence

from .core import (
    Endpoint,
    GeometryRef,
    GripperPair,
    URDFDocument,
    URDFValidationError,
    Vec3,
    identity,
    invert_rigid,
    iter_geometry,
    matmul,
    transform_point,
)
from .mesh import MeshCache, Triangle


@dataclass(frozen=True)
class SceneShape:
    triangles: tuple[Triangle, ...]
    color: str
    outline: str
    wireframe: bool


@dataclass(frozen=True)
class SceneMarker:
    position: Vec3
    color: str
    label: str


def geometry_key(ref: GeometryRef) -> tuple[str, str, int]:
    return ref.link_name, ref.role, ref.index


def _hex_shade(color: str, factor: float) -> str:
    color = color.lstrip("#")
    if len(color) != 6:
        return "#808080"
    values = [int(color[index : index + 2], 16) for index in (0, 2, 4)]
    values = [max(0, min(255, int(value * factor))) for value in values]
    return "#" + "".join(f"{value:02x}" for value in values)


def _material_color(ref: GeometryRef) -> str:
    material = ref.element.find("material")
    color = None if material is None else material.find("color")
    if color is None or not color.get("rgba"):
        return "#8a94a6"
    try:
        rgb = [float(value) for value in color.get("rgba", "").split()[:3]]
    except ValueError:
        return "#8a94a6"
    if len(rgb) != 3:
        return "#8a94a6"
    return "#" + "".join(f"{max(0, min(255, round(value * 255))):02x}" for value in rgb)


class GripperCanvas(tk.Canvas):
    """Small dependency-free orthographic 3-D viewer optimized for jaw editing."""

    def __init__(self, master: tk.Misc, **kwargs):
        super().__init__(master, background="#111722", highlightthickness=0, **kwargs)
        self.yaw = math.radians(-48.0)
        self.pitch = math.radians(24.0)
        self.zoom = 3800.0
        self.center: Vec3 = (0.0, 0.0, 0.08)
        self.shapes: list[SceneShape] = []
        self.markers: list[SceneMarker] = []
        self._last_mouse: tuple[int, int] | None = None
        self._panning = False
        self._fit_on_resize = False
        self.bind("<Configure>", self._configure)
        self.bind("<ButtonPress-1>", self._press_rotate)
        self.bind("<B1-Motion>", self._drag_rotate)
        self.bind("<ButtonPress-3>", self._press_pan)
        self.bind("<B3-Motion>", self._drag_pan)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda _event: self._zoom(1.12))
        self.bind("<Button-5>", lambda _event: self._zoom(1 / 1.12))

    def _configure(self, event: tk.Event) -> None:
        if self._fit_on_resize and event.width > 200 and event.height > 200:
            self._fit_on_resize = False
            self.after_idle(self.fit)
        else:
            self.redraw()

    def _press_rotate(self, event: tk.Event) -> None:
        self._last_mouse = (event.x, event.y)
        self._panning = False

    def _drag_rotate(self, event: tk.Event) -> None:
        if self._last_mouse is None:
            return
        dx, dy = event.x - self._last_mouse[0], event.y - self._last_mouse[1]
        self._last_mouse = (event.x, event.y)
        self.yaw += dx * 0.008
        self.pitch = max(math.radians(-89), min(math.radians(89), self.pitch - dy * 0.008))
        self.redraw()

    def _press_pan(self, event: tk.Event) -> None:
        self._last_mouse = (event.x, event.y)
        self._panning = True

    def _drag_pan(self, event: tk.Event) -> None:
        if self._last_mouse is None:
            return
        dx, dy = event.x - self._last_mouse[0], event.y - self._last_mouse[1]
        self._last_mouse = (event.x, event.y)
        right, up, _forward = self._camera_basis()
        scale = 1.0 / max(self.zoom, 1.0)
        self.center = tuple(
            self.center[index] - right[index] * dx * scale + up[index] * dy * scale
            for index in range(3)
        )  # type: ignore[assignment]
        self.redraw()

    def _wheel(self, event: tk.Event) -> None:
        self._zoom(1.12 if event.delta > 0 else 1 / 1.12)

    def _zoom(self, factor: float) -> None:
        self.zoom = max(100.0, min(100000.0, self.zoom * factor))
        self.redraw()

    def set_view(self, name: str) -> None:
        views = {
            "iso": (math.radians(-48), math.radians(24)),
            "front": (0.0, 0.0),
            "side": (math.radians(-90), 0.0),
            "top": (0.0, math.radians(89.0)),
        }
        self.yaw, self.pitch = views[name]
        self.redraw()

    def _camera_basis(self) -> tuple[Vec3, Vec3, Vec3]:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        forward = (cp * cy, cp * sy, sp)
        right = (-sy, cy, 0.0)
        up = (-sp * cy, -sp * sy, cp)
        return right, up, forward

    def _project(self, point: Sequence[float]) -> tuple[float, float, float]:
        delta = tuple(float(point[index]) - self.center[index] for index in range(3))
        right, up, forward = self._camera_basis()
        width, height = max(1, self.winfo_width()), max(1, self.winfo_height())
        return (
            width / 2 + sum(delta[index] * right[index] for index in range(3)) * self.zoom,
            height / 2 - sum(delta[index] * up[index] for index in range(3)) * self.zoom,
            sum(delta[index] * forward[index] for index in range(3)),
        )

    def fit(self) -> None:
        points = [vertex for shape in self.shapes for triangle in shape.triangles for vertex in triangle]
        points.extend(marker.position for marker in self.markers)
        if not points:
            return
        minimum = [min(point[index] for point in points) for index in range(3)]
        maximum = [max(point[index] for point in points) for index in range(3)]
        self.center = tuple((minimum[index] + maximum[index]) / 2.0 for index in range(3))  # type: ignore[assignment]
        extent = max(maximum[index] - minimum[index] for index in range(3))
        viewport = max(100, min(self.winfo_width(), self.winfo_height()))
        self.zoom = viewport * 0.72 / max(extent, 0.01)
        self.redraw()

    def set_scene(self, shapes: list[SceneShape], markers: list[SceneMarker], *, fit: bool = False) -> None:
        self.shapes, self.markers = shapes, markers
        if fit:
            self._fit_on_resize = True
            self.fit()
        else:
            self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        self._draw_grid_and_axes()
        _right, _up, forward = self._camera_basis()
        projected: list[tuple[float, SceneShape, tuple[tuple[float, float, float], ...], float]] = []
        for shape in self.shapes:
            for triangle in shape.triangles:
                points = tuple(self._project(vertex) for vertex in triangle)
                depth = sum(point[2] for point in points) / 3.0
                a, b, c = triangle
                normal = (
                    (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
                    (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
                    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
                )
                normal_length = math.sqrt(sum(value * value for value in normal)) or 1.0
                light = 0.48 + 0.52 * abs(sum(normal[index] * forward[index] for index in range(3)) / normal_length)
                projected.append((depth, shape, points, light))
        for _depth, shape, points, light in sorted(projected, key=lambda item: item[0], reverse=True):
            coords = [coordinate for point in points for coordinate in point[:2]]
            fill = "" if shape.wireframe else _hex_shade(shape.color, light)
            self.create_polygon(*coords, fill=fill, outline=shape.outline, width=1)
        for marker in self.markers:
            x, y, _ = self._project(marker.position)
            self.create_oval(x - 4, y - 4, x + 4, y + 4, fill=marker.color, outline="#ffffff")
            self.create_text(x + 7, y - 7, text=marker.label, fill=marker.color, anchor="sw", font=("TkDefaultFont", 9, "bold"))
        self.create_text(12, 12, anchor="nw", text="左键旋转  ·  右键平移  ·  滚轮缩放", fill="#aeb9cb")

    def _draw_grid_and_axes(self) -> None:
        grid_half = 0.08
        step = 0.01
        for index in range(-8, 9):
            value = index * step
            for start, end in (((-grid_half, value, 0.0), (grid_half, value, 0.0)), ((value, -grid_half, 0.0), (value, grid_half, 0.0))):
                a, b = self._project(start), self._project(end)
                self.create_line(a[0], a[1], b[0], b[1], fill="#253044")
        for axis, color, label in (((0.03, 0, 0), "#ff6677", "X"), ((0, 0.03, 0), "#66dd99", "Y"), ((0, 0, 0.03), "#66aaff", "Z")):
            origin, tip = self._project((0, 0, 0)), self._project(axis)
            self.create_line(origin[0], origin[1], tip[0], tip[1], fill=color, width=3, arrow=tk.LAST)
            self.create_text(tip[0], tip[1], text=label, fill=color, anchor="sw")


def build_gripper_scene(
    document: URDFDocument,
    pair: GripperPair,
    endpoint_a: Endpoint,
    endpoint_b: Endpoint,
    master_q: float,
    mesh_cache: MeshCache,
    *,
    show_visuals: bool,
    show_collisions: bool,
    show_contact_frames: bool,
) -> tuple[list[SceneShape], list[SceneMarker], list[str]]:
    positions = document.pair_joint_positions(pair, master_q)
    transforms = document.link_transforms(positions)
    base_transform = transforms.get(pair.parent_link, identity())
    localizer = invert_rigid(base_transform)
    links = {link for link, _depth in document.subtree_links(pair.parent_link)}
    selected_a, selected_b = geometry_key(endpoint_a.geometry), geometry_key(endpoint_b.geometry)
    shapes: list[SceneShape] = []
    warnings: list[str] = []
    for ref in iter_geometry(document, links):
        if ref.role == "visual" and not show_visuals:
            continue
        if ref.role == "collision" and not show_collisions:
            continue
        try:
            triangles = mesh_cache.geometry_triangles(document, ref)
        except URDFValidationError as exc:
            warnings.append(str(exc))
            continue
        if not triangles:
            continue
        geometry_transform = matmul(localizer, document.geometry_transform(ref, transforms))
        transformed = tuple(
            tuple(transform_point(geometry_transform, vertex) for vertex in triangle)
            for triangle in triangles
        )
        key = geometry_key(ref)
        if key == selected_a:
            color, outline = "#37a8ff", "#bde7ff"
        elif key == selected_b:
            color, outline = "#ff9a3d", "#ffe0b8"
        elif ref.role == "collision":
            color, outline = "#40d58a", "#64efaa"
        else:
            color, outline = _material_color(ref), "#30394a"
        shapes.append(SceneShape(transformed, color, outline, ref.role == "collision"))

    markers: list[SceneMarker] = []
    if show_contact_frames:
        for endpoint, color, label in ((endpoint_a, "#37a8ff", "A contact"), (endpoint_b, "#ff9a3d", "B contact")):
            contact = document.contact_frame(endpoint)
            if contact and contact in transforms:
                markers.append(SceneMarker(transform_point(matmul(localizer, transforms[contact]), (0, 0, 0)), color, label))
    return shapes, markers, warnings
