from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .core import GeometryRef, URDFDocument, URDFValidationError, Vec3, _floats


Triangle = tuple[Vec3, Vec3, Vec3]


@dataclass(frozen=True)
class MeshData:
    triangles: tuple[Triangle, ...]
    minimum: Vec3
    maximum: Vec3

    @property
    def dimensions(self) -> Vec3:
        return tuple(self.maximum[index] - self.minimum[index] for index in range(3))  # type: ignore[return-value]


def _bounds(points: Iterable[Sequence[float]]) -> tuple[Vec3, Vec3]:
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    count = 0
    for point in points:
        count += 1
        for index in range(3):
            value = float(point[index])
            minimum[index] = min(minimum[index], value)
            maximum[index] = max(maximum[index], value)
    if count == 0:
        raise URDFValidationError("Mesh has no vertices")
    return tuple(minimum), tuple(maximum)  # type: ignore[return-value]


def _load_binary_stl(data: bytes, max_triangles: int) -> MeshData:
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + triangle_count * 50
    if len(data) < expected:
        raise URDFValidationError("Truncated binary STL")
    stride = max(1, math.ceil(triangle_count / max_triangles))
    triangles: list[Triangle] = []
    all_points: list[Vec3] = []
    for index in range(triangle_count):
        offset = 84 + index * 50 + 12
        values = struct.unpack_from("<9f", data, offset)
        triangle: Triangle = (
            (values[0], values[1], values[2]),
            (values[3], values[4], values[5]),
            (values[6], values[7], values[8]),
        )
        all_points.extend(triangle)
        if index % stride == 0:
            triangles.append(triangle)
    minimum, maximum = _bounds(all_points)
    return MeshData(tuple(triangles), minimum, maximum)


def _load_ascii_stl(data: bytes, max_triangles: int) -> MeshData:
    vertices: list[Vec3] = []
    for raw_line in data.decode("utf-8", errors="replace").splitlines():
        parts = raw_line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            try:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError as exc:
                raise URDFValidationError("Invalid vertex in ASCII STL") from exc
    if len(vertices) < 3:
        raise URDFValidationError("ASCII STL contains no triangles")
    triangles_all = [tuple(vertices[index : index + 3]) for index in range(0, len(vertices) - 2, 3)]
    stride = max(1, math.ceil(len(triangles_all) / max_triangles))
    minimum, maximum = _bounds(vertices)
    return MeshData(tuple(triangles_all[::stride]), minimum, maximum)  # type: ignore[arg-type]


def _load_obj(data: bytes, max_triangles: int) -> MeshData:
    vertices: list[Vec3] = []
    faces: list[tuple[int, ...]] = []
    for raw_line in data.decode("utf-8", errors="replace").splitlines():
        parts = raw_line.strip().split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif parts[0] == "f" and len(parts) >= 4:
            indices = []
            for value in parts[1:]:
                index = int(value.split("/", 1)[0])
                indices.append(index - 1 if index > 0 else len(vertices) + index)
            faces.append(tuple(indices))
    triangles: list[Triangle] = []
    for face in faces:
        for index in range(1, len(face) - 1):
            triangles.append((vertices[face[0]], vertices[face[index]], vertices[face[index + 1]]))
    if not triangles:
        raise URDFValidationError("OBJ contains no triangular faces")
    stride = max(1, math.ceil(len(triangles) / max_triangles))
    minimum, maximum = _bounds(vertices)
    return MeshData(tuple(triangles[::stride]), minimum, maximum)


def load_mesh(path: Path, *, max_triangles: int = 1800) -> MeshData:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise URDFValidationError(f"Cannot read mesh {path}: {exc}") from exc
    suffix = path.suffix.lower()
    if suffix == ".stl":
        if len(data) >= 84:
            triangle_count = struct.unpack_from("<I", data, 80)[0]
            looks_ascii = data.lstrip().startswith(b"solid") and b"facet" in data[:1024].lower()
            if 84 + triangle_count * 50 <= len(data) and not looks_ascii:
                return _load_binary_stl(data, max_triangles)
        return _load_ascii_stl(data, max_triangles)
    if suffix == ".obj":
        return _load_obj(data, max_triangles)
    raise URDFValidationError(f"Preview supports STL/OBJ meshes, not {suffix or 'extensionless meshes'}")


def box_triangles(dimensions: Sequence[float]) -> tuple[Triangle, ...]:
    x, y, z = (float(value) / 2.0 for value in dimensions)
    vertices: tuple[Vec3, ...] = (
        (-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
        (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z),
    )
    faces = (
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    )
    return tuple((vertices[a], vertices[b], vertices[c]) for a, b, c in faces)


def cylinder_triangles(radius: float, length: float, segments: int = 24) -> tuple[Triangle, ...]:
    triangles: list[Triangle] = []
    half = length / 2.0
    bottom, top = (0.0, 0.0, -half), (0.0, 0.0, half)
    for index in range(segments):
        angle_a, angle_b = 2 * math.pi * index / segments, 2 * math.pi * (index + 1) / segments
        a0, b0 = (radius * math.cos(angle_a), radius * math.sin(angle_a), -half), (radius * math.cos(angle_b), radius * math.sin(angle_b), -half)
        a1, b1 = (a0[0], a0[1], half), (b0[0], b0[1], half)
        triangles.extend(((a0, b0, b1), (a0, b1, a1), (bottom, b0, a0), (top, a1, b1)))
    return tuple(triangles)


def sphere_triangles(radius: float, rings: int = 10, segments: int = 20) -> tuple[Triangle, ...]:
    triangles: list[Triangle] = []
    for ring in range(rings):
        latitude_a = -math.pi / 2 + math.pi * ring / rings
        latitude_b = -math.pi / 2 + math.pi * (ring + 1) / rings
        for segment in range(segments):
            longitude_a = 2 * math.pi * segment / segments
            longitude_b = 2 * math.pi * (segment + 1) / segments
            def point(latitude: float, longitude: float) -> Vec3:
                return (
                    radius * math.cos(latitude) * math.cos(longitude),
                    radius * math.cos(latitude) * math.sin(longitude),
                    radius * math.sin(latitude),
                )
            a, b, c, d = point(latitude_a, longitude_a), point(latitude_a, longitude_b), point(latitude_b, longitude_b), point(latitude_b, longitude_a)
            triangles.extend(((a, b, c), (a, c, d)))
    return tuple(triangles)


class MeshCache:
    def __init__(self, max_triangles: int = 1800):
        self.max_triangles = max_triangles
        self._cache: dict[Path, MeshData] = {}

    def get(self, path: Path) -> MeshData:
        resolved = path.resolve()
        if resolved not in self._cache:
            self._cache[resolved] = load_mesh(resolved, max_triangles=self.max_triangles)
        return self._cache[resolved]

    def dimensions(self, path: Path) -> Vec3:
        return self.get(path).dimensions

    def center(self, path: Path) -> Vec3:
        mesh = self.get(path)
        return tuple((mesh.minimum[index] + mesh.maximum[index]) / 2.0 for index in range(3))  # type: ignore[return-value]

    def geometry_triangles(self, document: URDFDocument, ref: GeometryRef) -> tuple[Triangle, ...]:
        geometry = ref.element.find("geometry")
        if geometry is None or len(geometry) != 1:
            return ()
        shape = geometry[0]
        if shape.tag == "box":
            return box_triangles(_floats(shape.get("size"), 3, (1.0, 1.0, 1.0)))
        if shape.tag == "cylinder":
            return cylinder_triangles(float(shape.get("radius", "0")), float(shape.get("length", "0")))
        if shape.tag == "sphere":
            return sphere_triangles(float(shape.get("radius", "0")))
        if shape.tag == "mesh":
            resolved = document.resolve_mesh_filename(shape.get("filename", ""))
            if resolved is None:
                return ()
            mesh = self.get(resolved)
            scale = _floats(shape.get("scale"), 3, (1.0, 1.0, 1.0))
            return tuple(
                tuple(tuple(vertex[index] * scale[index] for index in range(3)) for vertex in triangle)
                for triangle in mesh.triangles
            )  # type: ignore[return-value]
        return ()
