from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from .uv_texture import (
    generate_uv_checkerboard,
    generate_uv_layout_image,
    get_default_mano_uv_obj_path,
    load_uv_obj,
    save_rgb_image,
)


def _parse_image_size(value: str) -> Tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use WIDTHxHEIGHT, for example 1024x1024")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("WIDTH and HEIGHT must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("WIDTH and HEIGHT must be positive")
    return width, height


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate portable MANO UV debug textures.")
    parser.add_argument(
        "--pattern",
        choices=["wireframe", "checkerboard"],
        default="wireframe",
        help="Texture pattern to generate.",
    )
    parser.add_argument(
        "--hand",
        choices=["right", "left"],
        default="right",
        help="Bundled MANO_UV OBJ to use when --obj is not provided.",
    )
    parser.add_argument(
        "--obj",
        type=Path,
        default=None,
        help="Optional OBJ containing vt UV coordinates. Used by wireframe mode.",
    )
    parser.add_argument(
        "--image-size",
        type=_parse_image_size,
        default=(1024, 1024),
        help="Output image size, for example 1024x1024.",
    )
    parser.add_argument(
        "--checker-squares",
        type=int,
        default=16,
        help="Number of checkerboard squares per side.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to uv_texture_<HAND>_<PATTERN>.png.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.pattern == "wireframe":
        obj_path = args.obj or get_default_mano_uv_obj_path(args.hand)
        uv_mesh = load_uv_obj(obj_path)
        image = generate_uv_layout_image(
            uv_mesh.uv_coords,
            uv_mesh.uv_faces,
            image_size=args.image_size,
        )
    else:
        image = generate_uv_checkerboard(
            image_size=args.image_size,
            num_squares=args.checker_squares,
        )

    output = args.output or Path(f"uv_texture_{args.hand.upper()}_{args.pattern}.png")
    save_rgb_image(image, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
