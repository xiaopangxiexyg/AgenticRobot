from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .background import BackgroundMode, generate_background
from .real_image_provider import LocalImageBackgroundProvider, RealImageBackgroundConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REAL_IMAGE_ROOT = REPO_ROOT / "assets/random_real_image/openimages_v7_validation/images"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "samples/color_geometry_real_v2"
MODES: tuple[BackgroundMode, ...] = (
    "block_shuffle",
    "patch_mosaic",
    "lowfreq_noise",
    "pixel_shuffle",
    "color_blocks",
    "shape_collage",
)


def _require_cv2():
    import cv2

    return cv2


def _label(image: np.ndarray, text: str) -> np.ndarray:
    cv2 = _require_cv2()
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    return out


def _grid(images: list[np.ndarray], *, columns: int) -> np.ndarray:
    rows = (len(images) + columns - 1) // columns
    height, width = images[0].shape[:2]
    blank = np.zeros_like(images[0])
    padded = images + [blank] * (rows * columns - len(images))
    return np.vstack(
        [np.hstack(padded[row * columns : (row + 1) * columns]) for row in range(rows)]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate inspectable random-background augmentation galleries.")
    parser.add_argument("--real-image-root", type=Path, default=DEFAULT_REAL_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cv2 = _require_cv2()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = int(args.width), int(args.height)

    real_config = RealImageBackgroundConfig(
        seed=int(args.seed),
        min_scale=0.55,
        max_scale=2.4,
        min_aspect_stretch=0.50,
        max_aspect_stretch=2.0,
        color_jitter=0.10,
        blur_px=1,
    )
    provider = LocalImageBackgroundProvider(image_root=args.real_image_root, config=real_config)

    tiles = [
        _label(generate_background(width, height, seed=int(args.seed) + index, mode=mode), f"procedural: {mode}")
        for index, mode in enumerate(MODES)
    ]
    for index in range(6):
        stream = provider.stream(
            width=width,
            height=height,
            episode_index=100 + index,
            camera_key="gallery",
            epoch=0,
        )
        tiles.append(
            _label(
                stream.frame(index * 7),
                f"real s={stream.source_scale:.2f} x={stream.source_stretch_x:.2f} y={stream.source_stretch_y:.2f}",
            )
        )
    gallery_path = output_dir / "procedural_and_real_gallery.png"
    if not cv2.imwrite(str(gallery_path), _grid(tiles, columns=3)):
        raise RuntimeError(f"failed to write {gallery_path}")

    stream = provider.stream(width=width, height=height, episode_index=777, camera_key="motion", epoch=0)
    motion_tiles = [
        _label(
            stream.frame(frame_index),
            f"real f={frame_index} s={stream.source_scale:.2f} x={stream.source_stretch_x:.2f} y={stream.source_stretch_y:.2f}",
        )
        for frame_index in (0, 5, 15, 30, 60, 100)
    ]
    motion_path = output_dir / "real_image_scale_stretch_motion_strip.png"
    if not cv2.imwrite(str(motion_path), _grid(motion_tiles, columns=3)):
        raise RuntimeError(f"failed to write {motion_path}")

    print(
        json.dumps(
            {
                "real_image_count": len(provider.paths),
                "gallery": str(gallery_path),
                "motion_strip": str(motion_path),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
