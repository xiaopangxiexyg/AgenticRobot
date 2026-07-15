"""Visual-only OBJ/MTL material partitioning for MuJoCo mesh assets.

MuJoCo loads OBJ geometry but colors meshes through the referencing geom.  OBJ
``usemtl`` sections therefore have to become separate mesh/geom pairs before
they can retain CAD colors in an MJCF model.  This module is deliberately kept
out of the collision path: collision meshes remain the source URDF meshes.
"""

from __future__ import annotations

import hashlib
import math
import re
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_CACHE_NAMESPACE = "dexteleop_mujoco_obj_materials_v1"


@dataclass(frozen=True)
class ObjMaterialMeshPart:
    """One visual mesh partition and its resolved diffuse RGBA color."""

    material_name: str
    mesh_path: Path
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class _ObjFaceVertex:
    position: int
    texcoord: int | None
    normal: int | None


@dataclass(frozen=True)
class _ObjFace:
    vertices: tuple[_ObjFaceVertex, _ObjFaceVertex, _ObjFaceVertex]


def _safe_component(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text)).strip("._")
    return normalized or "material"


def _resolve_index(raw: str, *, size: int) -> int:
    value = int(raw)
    if value == 0:
        raise ValueError("OBJ indices are one-based and cannot be zero.")
    resolved = value if value > 0 else size + value + 1
    if resolved < 1 or resolved > size:
        raise ValueError(f"OBJ index {value} is outside the current range 1..{size}.")
    return resolved


def _parse_face_vertex(
    token: str,
    *,
    vertex_count: int,
    texcoord_count: int,
    normal_count: int,
) -> _ObjFaceVertex:
    fields = token.split("/")
    if not fields or not fields[0]:
        raise ValueError(f"Invalid OBJ face token {token!r}.")
    position = _resolve_index(fields[0], size=vertex_count)
    texcoord = (
        None
        if len(fields) < 2 or not fields[1]
        else _resolve_index(fields[1], size=texcoord_count)
    )
    normal = (
        None
        if len(fields) < 3 or not fields[2]
        else _resolve_index(fields[2], size=normal_count)
    )
    if len(fields) > 3:
        raise ValueError(f"Unsupported OBJ face token {token!r}.")
    return _ObjFaceVertex(position=position, texcoord=texcoord, normal=normal)


def _parse_mtl(
    path: Path,
) -> tuple[dict[str, tuple[float, float, float, float]], set[str]]:
    materials: dict[str, list[float | None]] = {}
    texture_mapped_materials: set[str] = set()
    current_name: str | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        keyword, _, payload = line.partition(" ")
        values = payload.strip()
        normalized = keyword.lower()
        if normalized == "newmtl":
            current_name = values
            if current_name:
                materials.setdefault(current_name, [None, None, None, None])
            continue
        if current_name is None or current_name not in materials:
            continue
        if normalized.startswith("map_") or normalized in {"bump", "disp", "decal", "refl"}:
            texture_mapped_materials.add(current_name)
            continue
        if normalized == "kd":
            try:
                rgb = [float(value) for value in values.split()[:3]]
            except ValueError:
                continue
            if len(rgb) == 3 and all(math.isfinite(value) for value in rgb):
                materials[current_name][:3] = rgb
        elif normalized == "d":
            try:
                alpha = float(values.split()[0])
            except (IndexError, ValueError):
                continue
            if math.isfinite(alpha):
                materials[current_name][3] = alpha
        elif normalized == "tr" and materials[current_name][3] is None:
            try:
                transparency = float(values.split()[0])
            except (IndexError, ValueError):
                continue
            if math.isfinite(transparency):
                materials[current_name][3] = 1.0 - transparency

    out: dict[str, tuple[float, float, float, float]] = {}
    for name, values in materials.items():
        if any(value is None for value in values[:3]):
            continue
        rgb = [min(1.0, max(0.0, float(value))) for value in values[:3]]
        alpha = 1.0 if values[3] is None else min(1.0, max(0.0, float(values[3])))
        out[name] = (rgb[0], rgb[1], rgb[2], alpha)
    return out, texture_mapped_materials


def _mtl_paths(source_path: Path, names: Iterable[str]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for name in names:
        candidate = (source_path.parent / name).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"OBJ material library not found: {candidate}")
        if candidate not in paths:
            paths.append(candidate)
    return tuple(paths)


def _cache_dir(source_path: Path, material_paths: Iterable[Path]) -> Path:
    digest = hashlib.sha256()
    for path in (source_path, *material_paths):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    root = Path(tempfile.gettempdir()) / _CACHE_NAMESPACE / digest.hexdigest()[:20]
    root.mkdir(parents=True, exist_ok=True)
    return root


def _format_face_vertex(
    vertex: _ObjFaceVertex,
    *,
    position_map: dict[int, int],
    texcoord_map: dict[int, int],
    normal_map: dict[int, int],
) -> str:
    position = position_map[vertex.position]
    if vertex.texcoord is None and vertex.normal is None:
        return str(position)
    if vertex.texcoord is None:
        return f"{position}//{normal_map[vertex.normal]}"
    if vertex.normal is None:
        return f"{position}/{texcoord_map[vertex.texcoord]}"
    return f"{position}/{texcoord_map[vertex.texcoord]}/{normal_map[vertex.normal]}"


def _write_partition(
    output_path: Path,
    *,
    source_path: Path,
    material_name: str,
    vertices: list[str],
    texcoords: list[str],
    normals: list[str],
    faces: Iterable[_ObjFace],
) -> None:
    position_map: dict[int, int] = {}
    texcoord_map: dict[int, int] = {}
    normal_map: dict[int, int] = {}
    selected_faces = tuple(faces)
    for face in selected_faces:
        for vertex in face.vertices:
            position_map.setdefault(vertex.position, len(position_map) + 1)
            if vertex.texcoord is not None:
                texcoord_map.setdefault(vertex.texcoord, len(texcoord_map) + 1)
            if vertex.normal is not None:
                normal_map.setdefault(vertex.normal, len(normal_map) + 1)

    lines = [
        f"# Derived visual-only partition of {source_path.name}",
        f"# OBJ material: {material_name}",
    ]
    for old_index in position_map:
        lines.append(f"v {vertices[old_index - 1]}")
    for old_index in texcoord_map:
        lines.append(f"vt {texcoords[old_index - 1]}")
    for old_index in normal_map:
        lines.append(f"vn {normals[old_index - 1]}")
    for face in selected_faces:
        face_tokens = [
            _format_face_vertex(
                vertex,
                position_map=position_map,
                texcoord_map=texcoord_map,
                normal_map=normal_map,
            )
            for vertex in face.vertices
        ]
        lines.append(f"f {' '.join(face_tokens)}")
    content = "\n".join(lines) + "\n"
    if not output_path.exists() or output_path.read_text(encoding="utf-8") != content:
        output_path.write_text(content, encoding="utf-8")


def obj_material_mesh_parts(source_path: Path) -> tuple[ObjMaterialMeshPart, ...]:
    """Return visual mesh partitions for a complete OBJ/MTL diffuse-color asset.

    An empty return value means the source cannot be represented completely by
    this lightweight Kd/d converter.  Callers should retain their existing
    one-geom visual fallback in that case.
    """

    source_path = Path(source_path).resolve()
    if source_path.suffix.lower() != ".obj":
        return ()
    try:
        vertices: list[str] = []
        texcoords: list[str] = []
        normals: list[str] = []
        material_libraries: list[str] = []
        faces_by_material: OrderedDict[str, list[_ObjFace]] = OrderedDict()
        active_material: str | None = None
        for raw_line in source_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            keyword, _, payload = line.partition(" ")
            values = payload.strip()
            if keyword == "v":
                vertices.append(values)
            elif keyword == "vt":
                texcoords.append(values)
            elif keyword == "vn":
                normals.append(values)
            elif keyword == "mtllib":
                material_libraries.extend(values.split())
            elif keyword == "usemtl":
                active_material = values or None
            elif keyword == "f":
                if active_material is None:
                    return ()
                raw_vertices = values.split()
                if len(raw_vertices) < 3:
                    return ()
                parsed = tuple(
                    _parse_face_vertex(
                        token,
                        vertex_count=len(vertices),
                        texcoord_count=len(texcoords),
                        normal_count=len(normals),
                    )
                    for token in raw_vertices
                )
                material_faces = faces_by_material.setdefault(active_material, [])
                for index in range(1, len(parsed) - 1):
                    material_faces.append(
                        _ObjFace(vertices=(parsed[0], parsed[index], parsed[index + 1]))
                    )
        if not vertices or not faces_by_material or not material_libraries:
            return ()
        material_paths = _mtl_paths(source_path, material_libraries)
        materials: dict[str, tuple[float, float, float, float]] = {}
        texture_mapped_materials: set[str] = set()
        for material_path in material_paths:
            parsed_materials, parsed_texture_mapped_materials = _parse_mtl(material_path)
            materials.update(parsed_materials)
            texture_mapped_materials.update(parsed_texture_mapped_materials)
        if set(faces_by_material) - set(materials):
            return ()
        if set(faces_by_material) & texture_mapped_materials:
            # Mapping a texture without preserving UV/material assets would be
            # an incomplete visual conversion. Fall back rather than silently
            # pretending Kd is the original textured appearance.
            return ()
    except (OSError, UnicodeError, ValueError):
        return ()

    if len(faces_by_material) == 1:
        material_name = next(iter(faces_by_material))
        return (
            ObjMaterialMeshPart(
                material_name=material_name,
                mesh_path=source_path,
                rgba=materials[material_name],
            ),
        )

    cache_dir = _cache_dir(source_path, material_paths)
    parts: list[ObjMaterialMeshPart] = []
    for material_name, faces in faces_by_material.items():
        output_path = cache_dir / f"{source_path.stem}__{_safe_component(material_name)}.obj"
        _write_partition(
            output_path,
            source_path=source_path,
            material_name=material_name,
            vertices=vertices,
            texcoords=texcoords,
            normals=normals,
            faces=faces,
        )
        parts.append(
            ObjMaterialMeshPart(
                material_name=material_name,
                mesh_path=output_path,
                rgba=materials[material_name],
            )
        )
    return tuple(parts)


__all__ = ["ObjMaterialMeshPart", "obj_material_mesh_parts"]
