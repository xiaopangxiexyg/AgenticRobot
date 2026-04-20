# portable_init_UV

Portable UV initialization/debug-texture helpers extracted from:

- `tactask/tactile_representation/render_static_texture_hand.py`
- `tactask/tactile_representation/mesh_utils.py`

This package contains the three requested functions:

- `generate_uv_layout_image(...)`
- `generate_uv_checkerboard(...)`
- `set_texture_coords(...)`

It also includes a small loader and CLI so the same UV wireframe texture can be
generated without importing the full renderer, MANO layer, `PathConfig`, or
project-specific modules.

## Files

- `uv_texture.py`: core portable functions.
- `generate_uv_texture.py`: command-line entry point.
- `assets/MANO_UV_right.obj`, `assets/MANO_UV_left.obj`: bundled MANO UV OBJ
  files used by the CLI.
- `requirements.txt`: minimal dependencies plus optional PyVista/VTK rendering
  dependencies.

## Install Dependencies

From the repository root:

```bash
pip install -r portable_init_UV/requirements.txt
```

If you only need to generate PNG textures, `numpy`, `pillow`, and `trimesh` are
enough. `pyvista` and `vtk` are only needed for `set_texture_coords(...)`.

## Generate a UV Wireframe PNG

From the repository root:

```bash
python -m portable_init_UV.generate_uv_texture \
  --hand right \
  --pattern wireframe \
  --image-size 1024x1024 \
  --output portable_init_UV/uv_texture_RIGHT_wireframe.png
```

For the left hand:

```bash
python -m portable_init_UV.generate_uv_texture \
  --hand left \
  --pattern wireframe \
  --output portable_init_UV/uv_texture_LEFT_wireframe.png
```

## Generate a Checkerboard PNG

```bash
python -m portable_init_UV.generate_uv_texture \
  --pattern checkerboard \
  --checker-squares 16 \
  --output portable_init_UV/uv_checkerboard.png
```

## Use in Python

Equivalent to the original renderer code:

```python
from portable_init_UV import generate_uv_layout_image

uv_texture_image = generate_uv_layout_image(
    self.uv_coords[ref_hand],
    self.uv_faces[ref_hand],
    image_size=img_size,
)
```

If you want to load the bundled MANO UV OBJ directly:

```python
from portable_init_UV import (
    generate_uv_layout_image,
    get_default_mano_uv_obj_path,
    load_uv_obj,
    save_rgb_image,
)

uv_mesh = load_uv_obj(get_default_mano_uv_obj_path("right"))
uv_texture_image = generate_uv_layout_image(
    uv_mesh.uv_coords,
    uv_mesh.uv_faces,
    image_size=(1024, 1024),
)
save_rgb_image(uv_texture_image, "uv_texture_RIGHT.png")
```

For renderer-style dictionaries:

```python
from portable_init_UV import generate_uv_layout_from_hand_data

uv_texture_image = generate_uv_layout_from_hand_data(
    uv_coords_by_hand=self.uv_coords,
    uv_faces_by_hand=self.uv_faces,
    ref_hand=ref_hand,
    image_size=img_size,
)
```

## Notes

- `uv_faces` in `render_static_texture_hand.py` comes from
  `transfer_uv_layout_robust(...)`, but for the static UV layout image it is the
  same triangle topology as the MANO UV OBJ faces.
- The package intentionally avoids importing `manotorch`, `PathConfig`, or the
  full `mesh_utils.py` so it can be copied into another project.
- `set_texture_coords(mesh, uv_coords)` imports `vtk` lazily. PNG generation
  works even when `vtk` is not installed.
