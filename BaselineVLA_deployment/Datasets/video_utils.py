from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def load_video_frames_ffmpeg(
    video_path: str | Path,
    frame_indices: Sequence[int],
    resize_height: int,
    resize_width: int,
) -> torch.Tensor:
    frames = load_video_frames_ffmpeg_uint8(
        video_path=video_path,
        frame_indices=frame_indices,
        resize_height=resize_height,
        resize_width=resize_width,
        color_order="rgb",
    )
    if int(frames.shape[0]) == 0:
        return torch.zeros(0, 3, int(resize_height), int(resize_width), dtype=torch.float32)
    return torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0


def load_video_frames_ffmpeg_uint8(
    video_path: str | Path,
    frame_indices: Sequence[int],
    resize_height: int,
    resize_width: int,
    color_order: str = "rgb",
) -> np.ndarray:
    resolved_indices = [int(frame_index) for frame_index in frame_indices]
    if not resolved_indices:
        return np.zeros((0, int(resize_height), int(resize_width), 3), dtype=np.uint8)
    if min(resolved_indices) < 0:
        raise ValueError(f"frame_indices must be non-negative, got {resolved_indices}")

    return _decode_selected_frames_ffmpeg(
        video_path=video_path,
        frame_indices=resolved_indices,
        resize_height=int(resize_height),
        resize_width=int(resize_width),
        pix_fmt=_resolve_pixel_format(color_order),
    )


def build_rgb_mean_video_tokens(video_frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if video_frames.dim() != 4:
        raise ValueError(f"video_frames must have shape [T, C, H, W], got {tuple(video_frames.shape)}")
    if int(video_frames.shape[0]) == 0:
        return torch.zeros(0, 4, dtype=torch.float32), torch.zeros(0, dtype=torch.float32)

    frame_means = video_frames.mean(dim=(2, 3))
    temporal_positions = torch.linspace(
        0.0,
        1.0,
        steps=int(video_frames.shape[0]),
        dtype=video_frames.dtype,
        device=video_frames.device,
    ).unsqueeze(-1)
    tokens = torch.cat([frame_means, temporal_positions], dim=-1)
    mask = torch.ones(int(video_frames.shape[0]), dtype=torch.float32, device=video_frames.device)
    return tokens, mask


def apply_video_augmentations(video_frames: torch.Tensor, augmentation_config: object) -> torch.Tensor:
    if int(video_frames.shape[0]) == 0 or not _config_bool(augmentation_config, "enabled", default=False):
        return video_frames
    if video_frames.dim() != 4:
        raise ValueError(f"video_frames must have shape [T, C, H, W], got {tuple(video_frames.shape)}")

    augmented = video_frames
    augmented = _random_resized_crop(augmented, augmentation_config)
    augmented = _color_jitter(augmented, augmentation_config)
    augmented = _brightness_boost(augmented, augmentation_config)
    augmented = _gaussian_noise(augmented, augmentation_config)
    augmented = _gaussian_blur(augmented, augmentation_config)
    augmented = _random_erasing(augmented, augmentation_config)
    return augmented.clamp_(0.0, 1.0)


def _config_value(config: object, name: str, default: object) -> object:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _config_bool(config: object, name: str, default: bool = False) -> bool:
    value = _config_value(config, name, default)
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "none", "null"}:
            return False
    return bool(value)


def _config_float(config: object, name: str, default: float) -> float:
    value = _config_value(config, name, default)
    if value is None:
        return float(default)
    return float(value)


def _config_int(config: object, name: str, default: int) -> int:
    value = _config_value(config, name, default)
    if value is None:
        return int(default)
    return int(value)


def _rand_uniform(low: float, high: float, *, device: torch.device) -> float:
    return float(torch.empty((), device=device).uniform_(float(low), float(high)).item())


def _rand_prob(probability: float, *, device: torch.device) -> bool:
    return float(probability) > 0.0 and float(torch.rand((), device=device).item()) < float(probability)


def _random_resized_crop(video_frames: torch.Tensor, config: object) -> torch.Tensor:
    probability = _config_float(config, "random_resized_crop_prob", 0.0)
    if not _rand_prob(probability, device=video_frames.device):
        return video_frames

    _, _, height, width = video_frames.shape
    if height <= 1 or width <= 1:
        return video_frames

    scale_min = max(0.01, _config_float(config, "crop_scale_min", 0.85))
    scale_max = min(1.0, max(scale_min, _config_float(config, "crop_scale_max", 1.0)))
    ratio_min = max(0.1, _config_float(config, "crop_ratio_min", 0.9))
    ratio_max = max(ratio_min, _config_float(config, "crop_ratio_max", 1.1))
    area = float(height * width)

    crop_h = height
    crop_w = width
    for _ in range(10):
        target_area = area * _rand_uniform(scale_min, scale_max, device=video_frames.device)
        aspect_ratio = _rand_uniform(ratio_min, ratio_max, device=video_frames.device)
        candidate_w = int(round((target_area * aspect_ratio) ** 0.5))
        candidate_h = int(round((target_area / aspect_ratio) ** 0.5))
        if 1 <= candidate_h <= height and 1 <= candidate_w <= width:
            crop_h = candidate_h
            crop_w = candidate_w
            break

    if crop_h == height and crop_w == width:
        return video_frames

    top = 0 if crop_h == height else int(torch.randint(0, height - crop_h + 1, (), device=video_frames.device).item())
    left = 0 if crop_w == width else int(torch.randint(0, width - crop_w + 1, (), device=video_frames.device).item())
    cropped = video_frames[..., top : top + crop_h, left : left + crop_w]
    return F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)


def _color_jitter(video_frames: torch.Tensor, config: object) -> torch.Tensor:
    probability = _config_float(config, "color_jitter_prob", 0.0)
    if not _rand_prob(probability, device=video_frames.device):
        return video_frames

    jittered = video_frames
    brightness = max(0.0, _config_float(config, "brightness", 0.0))
    contrast = max(0.0, _config_float(config, "contrast", 0.0))
    saturation = max(0.0, _config_float(config, "saturation", 0.0))

    if brightness > 0.0:
        factor = _rand_uniform(1.0 - brightness, 1.0 + brightness, device=video_frames.device)
        jittered = jittered * factor
    if contrast > 0.0:
        factor = _rand_uniform(1.0 - contrast, 1.0 + contrast, device=video_frames.device)
        mean = jittered.mean(dim=(2, 3), keepdim=True)
        jittered = (jittered - mean) * factor + mean
    if saturation > 0.0 and int(jittered.shape[1]) >= 3:
        factor = _rand_uniform(1.0 - saturation, 1.0 + saturation, device=video_frames.device)
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=jittered.dtype, device=jittered.device).view(1, 3, 1, 1)
        gray = (jittered[:, :3] * weights).sum(dim=1, keepdim=True)
        jittered_rgb = (jittered[:, :3] - gray) * factor + gray
        jittered = torch.cat([jittered_rgb, jittered[:, 3:]], dim=1) if int(jittered.shape[1]) > 3 else jittered_rgb
    return jittered


def _brightness_boost(video_frames: torch.Tensor, config: object) -> torch.Tensor:
    probability = _config_float(config, "brightness_boost_prob", 0.0)
    max_delta = max(0.0, _config_float(config, "brightness_boost_max_delta", 0.0))
    if max_delta <= 0.0 or not _rand_prob(probability, device=video_frames.device):
        return video_frames
    delta = _rand_uniform(0.0, max_delta, device=video_frames.device)
    return video_frames + delta


def _gaussian_noise(video_frames: torch.Tensor, config: object) -> torch.Tensor:
    probability = _config_float(config, "gaussian_noise_prob", 0.0)
    noise_std_max = max(0.0, _config_float(config, "gaussian_noise_std", 0.0))
    if noise_std_max <= 0.0 or not _rand_prob(probability, device=video_frames.device):
        return video_frames
    noise_std = _rand_uniform(0.0, noise_std_max, device=video_frames.device)
    return video_frames + torch.randn_like(video_frames) * noise_std


def _gaussian_blur(video_frames: torch.Tensor, config: object) -> torch.Tensor:
    probability = _config_float(config, "gaussian_blur_prob", 0.0)
    if not _rand_prob(probability, device=video_frames.device):
        return video_frames

    kernel_size = _config_int(config, "gaussian_blur_kernel", 3)
    if kernel_size <= 1:
        return video_frames
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma_max = max(0.0, _config_float(config, "gaussian_blur_sigma", 0.8))
    if sigma_max <= 0.0:
        return video_frames
    sigma = max(1e-6, _rand_uniform(0.0, sigma_max, device=video_frames.device))
    radius = kernel_size // 2
    if int(video_frames.shape[-2]) <= radius or int(video_frames.shape[-1]) <= radius:
        return video_frames
    coords = torch.arange(-radius, radius + 1, dtype=video_frames.dtype, device=video_frames.device)
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    channels = int(video_frames.shape[1])
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    padded = F.pad(video_frames, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel, groups=channels)


def _random_erasing(video_frames: torch.Tensor, config: object) -> torch.Tensor:
    probability = _config_float(config, "random_erasing_prob", 0.0)
    if not _rand_prob(probability, device=video_frames.device):
        return video_frames

    _, _, height, width = video_frames.shape
    if height <= 1 or width <= 1:
        return video_frames
    scale_min = max(0.0, _config_float(config, "random_erasing_scale_min", 0.02))
    scale_max = max(scale_min, _config_float(config, "random_erasing_scale_max", 0.08))
    ratio_min = max(0.1, _config_float(config, "random_erasing_ratio_min", 0.3))
    ratio_max = max(ratio_min, _config_float(config, "random_erasing_ratio_max", 3.3))
    center_protect_fraction = min(1.0, max(0.0, _config_float(config, "random_erasing_center_protect_fraction", 0.5)))
    protect_h = int(round(height * center_protect_fraction))
    protect_w = int(round(width * center_protect_fraction))
    protect_top = (height - protect_h) // 2
    protect_left = (width - protect_w) // 2
    protect_bottom = protect_top + protect_h
    protect_right = protect_left + protect_w
    area = float(height * width)

    erase_h = 0
    erase_w = 0
    top = 0
    left = 0
    for _ in range(50):
        target_area = area * _rand_uniform(scale_min, scale_max, device=video_frames.device)
        aspect_ratio = _rand_uniform(ratio_min, ratio_max, device=video_frames.device)
        candidate_w = int(round((target_area * aspect_ratio) ** 0.5))
        candidate_h = int(round((target_area / aspect_ratio) ** 0.5))
        if not (1 <= candidate_h < height and 1 <= candidate_w < width):
            continue
        candidate_top = int(torch.randint(0, height - candidate_h + 1, (), device=video_frames.device).item())
        candidate_left = int(torch.randint(0, width - candidate_w + 1, (), device=video_frames.device).item())
        candidate_bottom = candidate_top + candidate_h
        candidate_right = candidate_left + candidate_w
        intersects_protected_center = (
            candidate_left < protect_right
            and candidate_right > protect_left
            and candidate_top < protect_bottom
            and candidate_bottom > protect_top
        )
        if intersects_protected_center:
            continue
        erase_h = candidate_h
        erase_w = candidate_w
        top = candidate_top
        left = candidate_left
        break
    if erase_h <= 0 or erase_w <= 0:
        return video_frames

    erased = video_frames.clone()
    fill_value = _config_float(config, "random_erasing_value", 0.0)
    erased[..., top : top + erase_h, left : left + erase_w] = float(fill_value)
    return erased


def _decode_selected_frames_ffmpeg(
    video_path: str | Path,
    frame_indices: Sequence[int],
    resize_height: int,
    resize_width: int,
    pix_fmt: str,
) -> np.ndarray:
    resolved_path = Path(video_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Video file not found: {resolved_path}")
    if int(resize_height) <= 0 or int(resize_width) <= 0:
        raise ValueError(f"resize_height and resize_width must be positive, got {(resize_height, resize_width)}")

    unique_frame_indices = sorted(set(int(frame_index) for frame_index in frame_indices))
    # Decode the whole clip with one ffmpeg process so DataLoader workers do not fork once per frame.
    filter_expression = _build_frame_select_filter(
        frame_indices=unique_frame_indices,
        resize_height=int(resize_height),
        resize_width=int(resize_width),
    )
    expected_frame_count = len(unique_frame_indices)
    expected_frame_bytes = int(resize_height) * int(resize_width) * 3

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-threads",
        "1",
        "-i",
        str(resolved_path),
        "-map",
        "0:v:0",
        "-vf",
        filter_expression,
        "-vsync",
        "0",
        "-frames:v",
        str(expected_frame_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg executable not found in PATH.") from error

    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(
            f"ffmpeg failed for {resolved_path} frames {unique_frame_indices}: {stderr_text}"
        )
    expected_bytes = expected_frame_count * expected_frame_bytes
    if len(result.stdout) != expected_bytes:
        actual_frame_count = 0 if expected_frame_bytes == 0 else len(result.stdout) // expected_frame_bytes
        raise RuntimeError(
            f"Decoded frame byte size mismatch for {resolved_path}: "
            f"expected {expected_frame_count} frames ({expected_bytes} bytes), "
            f"got {actual_frame_count} frames ({len(result.stdout)} bytes). "
            f"Requested frames: {unique_frame_indices}"
        )

    decoded_frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        expected_frame_count,
        int(resize_height),
        int(resize_width),
        3,
    )
    frame_lookup = {frame_index: decoded_frames[position] for position, frame_index in enumerate(unique_frame_indices)}
    return np.stack([frame_lookup[int(frame_index)] for frame_index in frame_indices], axis=0)


def _build_frame_select_filter(frame_indices: Sequence[int], resize_height: int, resize_width: int) -> str:
    select_terms = "+".join(f"eq(n\\,{int(frame_index)})" for frame_index in frame_indices)
    return f"select={select_terms},scale={int(resize_width)}:{int(resize_height)}"


def _resolve_pixel_format(color_order: str) -> str:
    normalized = str(color_order).lower()
    if normalized == "rgb":
        return "rgb24"
    if normalized == "bgr":
        return "bgr24"
    raise ValueError(f"Unsupported color_order={color_order}. Expected 'rgb' or 'bgr'.")
