from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


ImageSize = Tuple[int, int]
RGBColor = Tuple[int, int, int]


@dataclass(frozen=True)
class UVMesh:
    """UV-aware mesh data loaded from an OBJ file."""

    uv_coords: np.ndarray
    uv_faces: np.ndarray
    vertices: np.ndarray
    mesh: object


def _as_image_size(image_size: ImageSize) -> ImageSize:
    if len(image_size) != 2:
        raise ValueError("image_size must be a tuple: (width, height)")
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size width and height must be positive")
    return width, height


def _validate_rgb(color: RGBColor, name: str) -> RGBColor:
    if len(color) != 3:
        raise ValueError(f"{name} must contain three RGB values")
    rgb = tuple(int(c) for c in color)
    if any(c < 0 or c > 255 for c in rgb):
        raise ValueError(f"{name} RGB values must be in [0, 255]")
    return rgb


def _validate_uv_layout_inputs(
    uv_coords: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    uv_coords = np.asarray(uv_coords, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)

    if uv_coords.ndim != 2 or uv_coords.shape[1] != 2:
        raise ValueError("uv_coords must have shape (num_vertices, 2)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (num_faces, 3)")
    if faces.size and (faces.min() < 0 or faces.max() >= len(uv_coords)):
        raise ValueError("faces contains an index outside uv_coords")

    return uv_coords, faces


def generate_uv_layout_image(
    uv_coords: np.ndarray,
    faces: np.ndarray,
    image_size: ImageSize = (1024, 1024),
    line_color: RGBColor = (0, 0, 0),
    bg_color: RGBColor = (255, 255, 255),
    line_width: int = 1,
) -> np.ndarray:
    """
    Generate a UV wireframe texture image from UV coordinates and triangle faces.

    This is the portable version of render_static_texture_hand.py:

        uv_texture_image = generate_uv_layout_image(
            self.uv_coords[ref_hand],
            self.uv_faces[ref_hand],
            image_size=img_size,
        )

    Args:
        uv_coords: Array with shape (num_vertices, 2), values normally in [0, 1].
        faces: Triangle index array with shape (num_faces, 3).
        image_size: Output size as (width, height).
        line_color: RGB line color.
        bg_color: RGB background color.
        line_width: Wireframe line width in pixels.

    Returns:
        RGB image as a uint8 numpy array with shape (height, width, 3).
    """
    uv_coords, faces = _validate_uv_layout_inputs(uv_coords, faces)
    width, height = _as_image_size(image_size)
    line_color = _validate_rgb(line_color, "line_color")
    bg_color = _validate_rgb(bg_color, "bg_color")

    if line_width <= 0:
        raise ValueError("line_width must be positive")

    pixel_coords = np.copy(uv_coords)
    pixel_coords[:, 0] *= width - 1
    pixel_coords[:, 1] = (1.0 - pixel_coords[:, 1]) * (height - 1)
    pixel_coords = np.rint(pixel_coords).astype(np.int64)

    image = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(image)

    for face in faces:
        p0 = tuple(pixel_coords[face[0]])
        p1 = tuple(pixel_coords[face[1]])
        p2 = tuple(pixel_coords[face[2]])
        draw.line([p0, p1], fill=line_color, width=line_width)
        draw.line([p1, p2], fill=line_color, width=line_width)
        draw.line([p2, p0], fill=line_color, width=line_width)

    return np.asarray(image, dtype=np.uint8)


def generate_uv_checkerboard(
    image_size: ImageSize = (1024, 1024),
    num_squares: int = 16,
    color1: RGBColor = (50, 50, 50),
    color2: RGBColor = (100, 100, 100),
) -> np.ndarray:
    """
    Generate a checkerboard texture for quickly checking UV stretching.

    Returns an RGB uint8 numpy array with shape (height, width, 3).
    """
    width, height = _as_image_size(image_size)
    if num_squares <= 0:
        raise ValueError("num_squares must be positive")

    color1_np = np.asarray(_validate_rgb(color1, "color1"), dtype=np.uint8)
    color2_np = np.asarray(_validate_rgb(color2, "color2"), dtype=np.uint8)
    image = np.zeros((height, width, 3), dtype=np.uint8)

    row_edges = np.linspace(0, height, num_squares + 1, dtype=np.int64)
    col_edges = np.linspace(0, width, num_squares + 1, dtype=np.int64)
    for row in range(num_squares):
        for col in range(num_squares):
            color = color1_np if (row + col) % 2 == 0 else color2_np
            image[row_edges[row] : row_edges[row + 1], col_edges[col] : col_edges[col + 1]] = color

    return image


def set_texture_coords(mesh: object, uv_coords: np.ndarray) -> object:
    """
    Set VTK/PyVista texture coordinates on a mesh and return the same mesh.

    vtk is imported lazily so the image-generation helpers can be used without
    installing PyVista/VTK.
    """
    uv_coords = np.asarray(uv_coords, dtype=np.float32)
    if uv_coords.ndim != 2 or uv_coords.shape[1] != 2:
        raise ValueError("uv_coords must have shape (num_points, 2)")

    n_points = getattr(mesh, "n_points", None)
    if n_points is None:
        raise TypeError("mesh must expose a PyVista/VTK-style n_points attribute")
    if len(uv_coords) != n_points:
        raise ValueError(
            f"Texture coordinate count ({len(uv_coords)}) does not match mesh.n_points ({n_points})"
        )

    try:
        import vtk
    except ImportError as exc:
        raise ImportError("set_texture_coords requires vtk or pyvista to be installed") from exc

    vtk_array = vtk.vtkFloatArray()
    vtk_array.SetNumberOfComponents(2)
    vtk_array.SetNumberOfTuples(len(uv_coords))
    for idx, (u, v) in enumerate(uv_coords):
        vtk_array.SetTuple2(idx, float(u), float(v))

    mesh.GetPointData().SetTCoords(vtk_array)
    return mesh


def get_default_mano_uv_obj_path(hand: str = "right") -> Path:
    """Return the bundled MANO_UV OBJ path for 'right' or 'left'."""
    hand = hand.lower()
    if hand not in {"right", "left"}:
        raise ValueError("hand must be 'right' or 'left'")
    return Path(__file__).resolve().parent / "assets" / f"MANO_UV_{hand}.obj"


def load_uv_obj(obj_path: str | Path) -> UVMesh:
    """
    Load an OBJ file containing vt UV coordinates.

    For the bundled MANO_UV OBJ files this returns:
        uv_coords: (891, 2)
        uv_faces:  (1538, 3)
        vertices:  (891, 3)
    """
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("load_uv_obj requires trimesh to be installed") from exc

    obj_path = Path(obj_path)
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    mesh = trimesh.load(obj_path, process=True)
    if not hasattr(mesh.visual, "uv") or mesh.visual.uv is None:
        raise ValueError(f"OBJ does not contain UV coordinates: {obj_path}")

    return UVMesh(
        uv_coords=np.asarray(mesh.visual.uv, dtype=np.float32),
        uv_faces=np.asarray(mesh.faces, dtype=np.int64),
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        mesh=mesh,
    )


def generate_uv_texture_image(
    uv_coords: np.ndarray,
    uv_faces: Optional[np.ndarray] = None,
    pattern: str = "wireframe",
    image_size: ImageSize = (1024, 1024),
    checker_squares: int = 16,
) -> np.ndarray:
    """
    Generate either the wireframe UV layout or checkerboard texture.
    """
    pattern = pattern.lower()
    if pattern == "wireframe":
        if uv_faces is None:
            raise ValueError("uv_faces is required when pattern='wireframe'")
        return generate_uv_layout_image(uv_coords, uv_faces, image_size=image_size)
    if pattern == "checkerboard":
        return generate_uv_checkerboard(image_size=image_size, num_squares=checker_squares)
    raise ValueError("pattern must be 'wireframe' or 'checkerboard'")


def generate_uv_layout_from_hand_data(
    uv_coords_by_hand: Mapping[str, np.ndarray],
    uv_faces_by_hand: Mapping[str, np.ndarray],
    ref_hand: str,
    image_size: ImageSize = (1024, 1024),
) -> np.ndarray:
    """
    Convenience wrapper for code shaped like renderer.uv_coords/ref_hand.
    """
    if ref_hand not in uv_coords_by_hand:
        raise KeyError(f"Missing uv_coords for hand: {ref_hand}")
    if ref_hand not in uv_faces_by_hand:
        raise KeyError(f"Missing uv_faces for hand: {ref_hand}")
    return generate_uv_layout_image(
        uv_coords_by_hand[ref_hand],
        uv_faces_by_hand[ref_hand],
        image_size=image_size,
    )


def save_rgb_image(image: np.ndarray, output_path: str | Path) -> Path:
    """Save an RGB numpy image without requiring OpenCV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(output_path)
    return output_path
