from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np


BackgroundMode = Literal[
    "embodied_mix",
    "block_shuffle",
    "patch_mosaic",
    "lowfreq_noise",
    "pixel_shuffle",
    "color_blocks",
    "shape_collage",
]


@dataclass(frozen=True)
class BackgroundSpec:
    width: int = 640
    height: int = 480
    seed: int = 0
    mode: BackgroundMode = "embodied_mix"
    blur_px: int = 3
    noise_std: float = 5.0


@dataclass(frozen=True)
class TemporalBackgroundSpec:
    width: int = 640
    height: int = 480
    seed: int = 0
    mode: BackgroundMode = "embodied_mix"
    canvas_scale: float = 2.5
    min_drift_px_per_frame: float = 0.0
    max_drift_px_per_frame: float = 6.0
    drift_px_per_frame: float | None = None
    acceleration_px_per_frame2: float = 0.012
    rotation_deg_per_frame: float = 0.03
    zoom_per_frame: float = 0.0008
    jitter_px: float = 0.25
    velocity_noise_px_per_frame: float = 0.035


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("utils.random_background requires opencv-python for image IO/resizing.") from exc
    return cv2


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _resize(image: np.ndarray, width: int, height: int, interpolation: int | None = None) -> np.ndarray:
    cv2 = _require_cv2()
    interp = cv2.INTER_LINEAR if interpolation is None else interpolation
    return cv2.resize(image, (int(width), int(height)), interpolation=interp)


def _blur(image: np.ndarray, radius: int) -> np.ndarray:
    cv2 = _require_cv2()
    r = max(0, int(radius))
    if r <= 0:
        return image
    return cv2.GaussianBlur(image, (r * 2 + 1, r * 2 + 1), 0.0, borderType=cv2.BORDER_REFLECT)


def _palette(random: np.random.Generator) -> np.ndarray:
    presets = np.asarray(
        [
            [[36, 40, 43], [82, 88, 91], [143, 135, 118], [192, 184, 166]],
            [[52, 56, 62], [95, 105, 110], [132, 145, 150], [178, 188, 188]],
            [[45, 48, 40], [91, 96, 78], [132, 122, 95], [170, 155, 126]],
            [[50, 44, 48], [95, 82, 88], [135, 118, 126], [180, 170, 166]],
            [[35, 45, 52], [70, 93, 104], [124, 132, 126], [172, 169, 152]],
        ],
        dtype=np.float32,
    )
    base = presets[int(random.integers(0, len(presets)))].copy()
    jitter = random.normal(0.0, 12.0, size=base.shape).astype(np.float32)
    return np.clip(base + jitter, 0, 255)


def _lowfreq_noise(height: int, width: int, random: np.random.Generator) -> np.ndarray:
    cv2 = _require_cv2()
    small_h = int(random.integers(8, 25))
    small_w = max(8, int(round(small_h * width / max(1, height))))
    colors = _palette(random)
    weights = random.random((small_h, small_w, len(colors))).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=2, keepdims=True), 1e-6)
    small = weights @ colors
    image = cv2.resize(small.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)

    y = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1, 1)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32).reshape(1, width, 1)
    lighting = 0.72 + 0.36 * (
        float(random.uniform(-0.8, 0.8)) * (x - 0.5)
        + float(random.uniform(-0.8, 0.8)) * (y - 0.5)
        + float(random.uniform(0.1, 0.8))
    )
    return np.clip(image * lighting, 0, 255).astype(np.uint8)


def _patch_mosaic(height: int, width: int, random: np.random.Generator) -> np.ndarray:
    cv2 = _require_cv2()
    out = np.zeros((height, width, 3), dtype=np.float32)
    colors = _palette(random)
    y0 = 0
    while y0 < height:
        row_h = int(random.integers(20, 120))
        h = min(row_h, height - y0)
        x0 = 0
        while x0 < width:
            tile_w = int(random.integers(20, 140))
            w = min(tile_w, width - x0)
            c0, c1 = colors[random.choice(len(colors), size=2, replace=True)]
            yy = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1, 1)
            xx = np.linspace(0.0, 1.0, w, dtype=np.float32).reshape(1, w, 1)
            mix = np.clip(0.5 * xx + 0.5 * yy + random.normal(0.0, 0.08, (h, w, 1)), 0.0, 1.0)
            out[y0 : y0 + h, x0 : x0 + w] = c0 * (1.0 - mix) + c1 * mix
            x0 += w
        y0 += h
    for _ in range(int(random.integers(8, 22))):
        patch_h = int(random.integers(max(12, height // 20), max(24, height // 5)))
        patch_w = int(random.integers(max(12, width // 24), max(24, width // 4)))
        y = int(random.integers(0, max(1, height - patch_h + 1)))
        x = int(random.integers(0, max(1, width - patch_w + 1)))
        color = colors[int(random.integers(0, len(colors)))]
        alpha = float(random.uniform(0.15, 0.45))
        out[y : y + patch_h, x : x + patch_w] = out[y : y + patch_h, x : x + patch_w] * (1.0 - alpha) + color * alpha
    if random.random() < 0.5:
        out = cv2.rotate(out, cv2.ROTATE_180)
    return np.clip(_blur(out.astype(np.uint8), int(random.integers(1, 5))), 0, 255).astype(np.uint8)


def _color_block_palette(random: np.random.Generator) -> np.ndarray:
    """Return muted and vivid BGR colors for non-semantic shape augmentation."""
    cv2 = _require_cv2()
    count = int(random.integers(7, 16))
    hsv = np.empty((count, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = random.integers(0, 180, size=count, dtype=np.uint8)
    hsv[:, 0, 1] = random.integers(75, 256, size=count, dtype=np.uint8)
    hsv[:, 0, 2] = random.integers(80, 246, size=count, dtype=np.uint8)
    vivid = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(count, 3).astype(np.float32)
    palette = np.concatenate((vivid, _palette(random)), axis=0)
    if random.random() < 0.4:
        palette = np.concatenate(
            (palette, np.asarray([[18, 18, 18], [238, 238, 238]], dtype=np.float32)),
            axis=0,
        )
    return palette


def _blend_shape(canvas: np.ndarray, overlay: np.ndarray, *, alpha: float) -> np.ndarray:
    return np.clip(canvas * (1.0 - float(alpha)) + overlay * float(alpha), 0, 255)


def _color_blocks(height: int, width: int, random: np.random.Generator) -> np.ndarray:
    """Generate overlapping color rectangles at arbitrary positions and scales."""
    cv2 = _require_cv2()
    palette = _color_block_palette(random)
    base = palette[int(random.integers(0, len(palette)))].astype(np.float32)
    out = np.broadcast_to(base, (height, width, 3)).copy()
    min_side = max(4, min(width, height) // 80)
    for _ in range(int(random.integers(18, 96))):
        if random.random() < 0.18:
            block_w = int(random.uniform(0.32, 1.15) * width)
            block_h = int(random.uniform(0.20, 0.95) * height)
        else:
            block_w = int(random.uniform(min_side, max(min_side + 1, width * 0.36)))
            block_h = int(random.uniform(min_side, max(min_side + 1, height * 0.36)))
        x0 = int(random.integers(-block_w // 2, width))
        y0 = int(random.integers(-block_h // 2, height))
        overlay = out.copy()
        color = tuple(float(value) for value in palette[int(random.integers(0, len(palette)))])
        cv2.rectangle(overlay, (x0, y0), (x0 + block_w, y0 + block_h), color, thickness=-1)
        out = _blend_shape(out, overlay, alpha=float(random.uniform(0.36, 1.0)))
        if random.random() < 0.16:
            border = tuple(float(value) for value in palette[int(random.integers(0, len(palette)))])
            cv2.rectangle(out, (x0, y0), (x0 + block_w, y0 + block_h), border, thickness=int(random.integers(1, 5)))
    return np.clip(out, 0, 255).astype(np.uint8)


def _shape_collage(height: int, width: int, random: np.random.Generator) -> np.ndarray:
    """Generate a non-semantic collage of blocks, disks, triangles, and lines."""
    cv2 = _require_cv2()
    palette = _color_block_palette(random)
    out = _color_blocks(height, width, random).astype(np.float32)
    for _ in range(int(random.integers(16, 70))):
        overlay = out.copy()
        color = tuple(float(value) for value in palette[int(random.integers(0, len(palette)))])
        kind = str(random.choice(("rectangle", "ellipse", "triangle", "line")))
        center_x = int(random.integers(-width // 8, width + width // 8))
        center_y = int(random.integers(-height // 8, height + height // 8))
        if kind == "rectangle":
            half_w = int(random.uniform(max(4, width * 0.02), max(5, width * 0.24)))
            half_h = int(random.uniform(max(4, height * 0.02), max(5, height * 0.24)))
            cv2.rectangle(overlay, (center_x - half_w, center_y - half_h), (center_x + half_w, center_y + half_h), color, thickness=-1)
        elif kind == "ellipse":
            axes = (
                int(random.uniform(max(4, width * 0.02), max(5, width * 0.20))),
                int(random.uniform(max(4, height * 0.02), max(5, height * 0.20))),
            )
            cv2.ellipse(overlay, (center_x, center_y), axes, float(random.uniform(0.0, 180.0)), 0.0, 360.0, color, thickness=-1)
        elif kind == "triangle":
            scale = float(random.uniform(0.04, 0.32)) * min(width, height)
            angles = random.uniform(0.0, 2.0 * np.pi, size=3)
            points = np.asarray(
                [
                    (
                        center_x + np.cos(angle) * scale * random.uniform(0.35, 1.0),
                        center_y + np.sin(angle) * scale * random.uniform(0.35, 1.0),
                    )
                    for angle in angles
                ],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(overlay, points, color)
        else:
            x1 = int(random.integers(-width // 4, width + width // 4))
            y1 = int(random.integers(-height // 4, height + height // 4))
            cv2.line(overlay, (center_x, center_y), (x1, y1), color, thickness=int(random.integers(2, max(3, min(width, height) // 18))))
        out = _blend_shape(out, overlay, alpha=float(random.uniform(0.24, 0.86)))
    return np.clip(out, 0, 255).astype(np.uint8)


def _block_shuffle(base: np.ndarray, random: np.random.Generator) -> np.ndarray:
    h, w = base.shape[:2]
    block = int(random.choice([8, 12, 16, 24, 32, 48, 64]))
    blocks: list[np.ndarray] = []
    targets: list[tuple[int, int, int, int]] = []
    for y0 in range(0, h, block):
        for x0 in range(0, w, block):
            patch = base[y0 : min(y0 + block, h), x0 : min(x0 + block, w)].copy()
            blocks.append(patch)
            ph, pw = patch.shape[:2]
            targets.append((y0, x0, ph, pw))
    order = random.permutation(len(blocks))
    out = np.empty_like(base)
    for target_index, source_index in enumerate(order):
        y0, x0, th, tw = targets[target_index]
        patch = blocks[int(source_index)]
        if patch.shape[:2] != (th, tw):
            patch = _resize(patch, tw, th)
        out[y0 : y0 + th, x0 : x0 + tw] = patch[:th, :tw]
    return out


def _pixel_shuffle(base: np.ndarray, random: np.random.Generator) -> np.ndarray:
    flat = base.reshape(-1, 3).copy()
    random.shuffle(flat, axis=0)
    out = flat.reshape(base.shape)
    return _blur(out, int(random.integers(1, 3)))


def _camera_noise(image: np.ndarray, random: np.random.Generator, noise_std: float) -> np.ndarray:
    noisy = image.astype(np.float32)
    if float(noise_std) > 0.0:
        noisy += random.normal(0.0, float(noise_std), size=noisy.shape).astype(np.float32)
    if random.random() < 0.45:
        scale = float(random.uniform(0.88, 1.12))
        bias = float(random.uniform(-12.0, 12.0))
        noisy = noisy * scale + bias
    return np.clip(noisy, 0, 255).astype(np.uint8)


def generate_background(
    width: int,
    height: int,
    *,
    seed: int | np.random.Generator | None = None,
    mode: BackgroundMode = "embodied_mix",
    blur_px: int = 3,
    noise_std: float = 5.0,
) -> np.ndarray:
    """Generate a BGR uint8 background for replacing masked-out image regions.

    The default mode favors spatially coherent disorder over pure white noise,
    which is usually a better augmentation target for wrist-camera embodied-AI
    datasets: the policy should ignore scene identity, but the image statistics
    should still look camera-like.
    """

    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    random = _rng(seed)
    chosen = mode
    if mode == "embodied_mix":
        chosen = str(random.choice(("block_shuffle", "patch_mosaic", "lowfreq_noise", "pixel_shuffle", "color_blocks", "color_blocks", "shape_collage", "shape_collage")))  # type: ignore[assignment]

    base = _lowfreq_noise(height, width, random)
    if chosen == "lowfreq_noise":
        out = base
    elif chosen == "patch_mosaic":
        out = _patch_mosaic(height, width, random)
    elif chosen == "block_shuffle":
        out = _block_shuffle(base, random)
    elif chosen == "pixel_shuffle":
        out = _pixel_shuffle(base, random)
    elif chosen == "color_blocks":
        out = _color_blocks(height, width, random)
    elif chosen == "shape_collage":
        out = _shape_collage(height, width, random)
    else:
        raise ValueError(f"unsupported background mode: {mode!r}")

    return _camera_noise(_blur(out, int(blur_px)), random, float(noise_std))


def generate_background_from_spec(spec: BackgroundSpec) -> np.ndarray:
    return generate_background(
        spec.width,
        spec.height,
        seed=spec.seed,
        mode=spec.mode,
        blur_px=spec.blur_px,
        noise_std=spec.noise_std,
    )


def load_background(path: str | Path, *, width: int | None = None, height: int | None = None) -> np.ndarray:
    cv2 = _require_cv2()
    image = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read background image: {path}")
    if width is not None or height is not None:
        target_w = int(width if width is not None else image.shape[1])
        target_h = int(height if height is not None else image.shape[0])
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return image


def apply_random_background(
    frame: np.ndarray,
    keep_mask: np.ndarray,
    background: np.ndarray | None = None,
    *,
    seed: int | np.random.Generator | None = None,
    mode: BackgroundMode = "embodied_mix",
    blur_px: int = 2,
    edge_blur_px: int = 1,
) -> np.ndarray:
    """Composite frame foreground over a generated background.

    `keep_mask > 0` is foreground. The input and output use OpenCV BGR order.
    """

    cv2 = _require_cv2()
    image = np.asarray(frame, dtype=np.uint8)
    height, width = image.shape[:2]
    if background is None:
        bg = generate_background(width, height, seed=seed, mode=mode, blur_px=blur_px)
    else:
        bg = np.asarray(background, dtype=np.uint8)
        if bg.shape[:2] != (height, width):
            bg = cv2.resize(bg, (width, height), interpolation=cv2.INTER_AREA)
    mask = np.where(np.asarray(keep_mask) > 0, 255, 0).astype(np.uint8)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    if int(edge_blur_px) <= 0:
        out = bg.copy()
        cv2.copyTo(image, mask, out)
        return out
    alpha = cv2.GaussianBlur(mask, (int(edge_blur_px) * 2 + 1, int(edge_blur_px) * 2 + 1), 0.0)
    alpha_f = (alpha.astype(np.float32) / 255.0)[..., None]
    return np.clip(image.astype(np.float32) * alpha_f + bg.astype(np.float32) * (1.0 - alpha_f), 0, 255).astype(np.uint8)


class TemporalRandomBackground:
    """World-like random background stream with smooth camera-relative motion.

    This is the preferred model for wrist-camera video augmentation when no
    real background frame is available. It avoids frame-independent flicker and
    also avoids a static image stuck to the camera plane.
    """

    def __init__(self, spec: TemporalBackgroundSpec):
        self.spec = spec
        self.random = _rng(int(spec.seed))
        self.width = int(spec.width)
        self.height = int(spec.height)
        scale = max(1.2, float(spec.canvas_scale))
        self.canvas_width = max(self.width + 8, int(round(self.width * scale)))
        self.canvas_height = max(self.height + 8, int(round(self.height * scale)))
        self.canvas = generate_background(
            self.canvas_width,
            self.canvas_height,
            seed=int(spec.seed) + 104729,
            mode=spec.mode,
            blur_px=2,
            noise_std=3.0,
        )
        angle = float(self.random.uniform(0.0, 2.0 * np.pi))
        if spec.drift_px_per_frame is None:
            min_speed = max(0.0, float(spec.min_drift_px_per_frame))
            max_speed = max(min_speed, float(spec.max_drift_px_per_frame))
            speed = float(self.random.uniform(min_speed, max_speed))
        else:
            speed = max(0.0, float(spec.drift_px_per_frame))
        self.velocity = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32) * speed
        accel_angle = angle + float(self.random.uniform(-0.75 * np.pi, 0.75 * np.pi))
        accel = float(spec.acceleration_px_per_frame2) * float(self.random.uniform(-1.0, 1.0))
        self.acceleration = np.asarray([np.cos(accel_angle), np.sin(accel_angle)], dtype=np.float32) * accel
        self.origin = np.asarray(
            [
                float(self.random.uniform(0.0, self.canvas_width - self.width)),
                float(self.random.uniform(0.0, self.canvas_height - self.height)),
            ],
            dtype=np.float32,
        )
        self.rotation0 = float(self.random.uniform(-2.0, 2.0))
        self.rotation_velocity = float(spec.rotation_deg_per_frame) * float(self.random.uniform(-1.0, 1.0))
        self.zoom0 = float(self.random.uniform(0.98, 1.04))
        self.zoom_velocity = float(spec.zoom_per_frame) * float(self.random.uniform(-1.0, 1.0))
        self.motion_phase = float(self.random.uniform(0.0, 2.0 * np.pi))
        self.motion_period = float(self.random.uniform(45.0, 160.0))
        self.jitter_phase = self.random.uniform(0.0, 2.0 * np.pi, size=2).astype(np.float32)
        self.jitter_period = self.random.uniform(11.0, 29.0, size=2).astype(np.float32)
        self.velocity_noise_phase = self.random.uniform(0.0, 2.0 * np.pi, size=2).astype(np.float32)
        self.velocity_noise_period = self.random.uniform(70.0, 190.0, size=2).astype(np.float32)

    def frame(self, frame_index: int) -> np.ndarray:
        cv2 = _require_cv2()
        t = float(max(0, int(frame_index)))
        max_x = max(1.0, float(self.canvas_width - self.width))
        max_y = max(1.0, float(self.canvas_height - self.height))
        jitter = (
            np.sin((t / self.jitter_period) * 2.0 * np.pi + self.jitter_phase)
            * float(self.spec.jitter_px)
        ).astype(np.float32)
        speed_mod = 1.0 + 0.35 * np.sin((t / self.motion_period) * 2.0 * np.pi + self.motion_phase)
        velocity_noise = (
            np.sin((t / self.velocity_noise_period) * 2.0 * np.pi + self.velocity_noise_phase)
            * float(self.spec.velocity_noise_px_per_frame)
            * t
        ).astype(np.float32)
        offset = self.origin + self.velocity * (t * speed_mod) + 0.5 * self.acceleration * (t * t) + velocity_noise + jitter
        x0 = float(offset[0] % max_x)
        y0 = float(offset[1] % max_y)

        # Wrap-pad the canvas so crop motion can loop without hard edges.
        x = int(np.floor(x0))
        y = int(np.floor(y0))
        padded = cv2.copyMakeBorder(
            self.canvas,
            self.height,
            self.height,
            self.width,
            self.width,
            borderType=cv2.BORDER_WRAP,
        )
        crop = padded[y + self.height : y + self.height + self.height, x + self.width : x + self.width + self.width].copy()

        angle = self.rotation0 + self.rotation_velocity * t
        zoom = np.clip(self.zoom0 + self.zoom_velocity * t, 0.92, 1.10)
        center = (self.width * 0.5, self.height * 0.5)
        matrix = cv2.getRotationMatrix2D(center, angle, zoom)
        return cv2.warpAffine(
            crop,
            matrix,
            (self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def apply(self, frame: np.ndarray, keep_mask: np.ndarray, frame_index: int, *, edge_blur_px: int = 1) -> np.ndarray:
        return apply_random_background(
            frame,
            keep_mask,
            background=self.frame(frame_index),
            edge_blur_px=edge_blur_px,
        )


def temporal_background_from_episode(
    *,
    width: int,
    height: int,
    episode_index: int,
    camera_key: str = "",
    seed: int = 20260709,
    mode: BackgroundMode = "embodied_mix",
) -> TemporalRandomBackground:
    camera_hash = sum((index + 1) * ord(ch) for index, ch in enumerate(str(camera_key)))
    spec = TemporalBackgroundSpec(
        width=int(width),
        height=int(height),
        seed=int(seed) + int(episode_index) * 100003 + int(camera_hash),
        mode=mode,
    )
    return TemporalRandomBackground(spec)


def save_background_set(
    output_dir: str | Path,
    *,
    width: int = 640,
    height: int = 480,
    count: int = 192,
    seed: int = 20260709,
    modes: Sequence[BackgroundMode] = (
        "block_shuffle",
        "patch_mosaic",
        "lowfreq_noise",
        "pixel_shuffle",
        "color_blocks",
        "shape_collage",
    ),
) -> dict[str, Any]:
    cv2 = _require_cv2()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "format": "dexteleop_random_background_manifest_v1",
        "width": int(width),
        "height": int(height),
        "seed": int(seed),
        "count": int(count),
        "files": [],
        "note": (
            "Procedural BGR backgrounds for masked wrist-camera augmentation. "
            "They mix spatially coherent texture, arbitrary color blocks, and non-semantic shapes instead of pure white noise."
        ),
    }
    for index in range(int(count)):
        mode = modes[index % len(modes)]
        item_seed = int(seed) + index * 9973
        spec = BackgroundSpec(width=width, height=height, seed=item_seed, mode=mode)
        image = generate_background_from_spec(spec)
        filename = f"random_bg_{index:04d}_{mode}.png"
        path = output / filename
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"failed to write background image: {path}")
        item = asdict(spec)
        item["path"] = filename
        manifest["files"].append(item)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest


def iter_manifest_paths(manifest_path: str | Path) -> Iterable[Path]:
    manifest_file = Path(manifest_path)
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    for item in payload.get("files", []):
        if isinstance(item, dict) and item.get("path"):
            yield (manifest_file.parent / str(item["path"])).resolve()
