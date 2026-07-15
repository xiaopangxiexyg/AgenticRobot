from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from math import exp, log
from typing import Any, Literal

import numpy as np


TemporalScope = Literal["frame", "clip"]
OcclusionMethod = Literal["random_rectangle", "random_blob", "random_grid", "full_black", "none"]


def _validate_probability(value: float, *, name: str) -> None:
    if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")


def _validate_range(values: tuple[float, float], *, name: str, lower_exclusive: bool = True) -> None:
    low, high = (float(values[0]), float(values[1]))
    lower_ok = low > 0.0 if lower_exclusive else low >= 0.0
    if not lower_ok or not np.isfinite(low) or not np.isfinite(high) or high < low:
        raise ValueError(f"{name} must be a finite increasing positive range")


@dataclass(frozen=True)
class RectangleOcclusionConfig:
    probability: float = 0.40
    block_count_range: tuple[int, int] = (1, 5)
    area_fraction_range: tuple[float, float] = (0.05, 0.50)
    aspect_ratio_range: tuple[float, float] = (0.10, 10.0)

    def __post_init__(self) -> None:
        _validate_probability(self.probability, name="RectangleOcclusionConfig.probability")
        low, high = (int(self.block_count_range[0]), int(self.block_count_range[1]))
        if low < 1 or high < low:
            raise ValueError("RectangleOcclusionConfig.block_count_range must be increasing and >= 1")
        _validate_range(self.area_fraction_range, name="RectangleOcclusionConfig.area_fraction_range")
        if float(self.area_fraction_range[1]) > 1.0:
            raise ValueError("RectangleOcclusionConfig.area_fraction_range cannot exceed 1")
        _validate_range(self.aspect_ratio_range, name="RectangleOcclusionConfig.aspect_ratio_range")


@dataclass(frozen=True)
class BlobOcclusionConfig:
    probability: float = 0.40
    blob_count_range: tuple[int, int] = (1, 8)
    sigma_fraction_range: tuple[float, float] = (0.03, 0.35)
    low_frequency_noise_weight: float = 0.25

    def __post_init__(self) -> None:
        _validate_probability(self.probability, name="BlobOcclusionConfig.probability")
        low, high = (int(self.blob_count_range[0]), int(self.blob_count_range[1]))
        if low < 1 or high < low:
            raise ValueError("BlobOcclusionConfig.blob_count_range must be increasing and >= 1")
        _validate_range(self.sigma_fraction_range, name="BlobOcclusionConfig.sigma_fraction_range")
        if not np.isfinite(self.low_frequency_noise_weight) or float(self.low_frequency_noise_weight) < 0.0:
            raise ValueError("BlobOcclusionConfig.low_frequency_noise_weight must be finite and >= 0")


@dataclass(frozen=True)
class GridOcclusionConfig:
    probability: float = 0.20
    patch_fractions: tuple[float, ...] = (1.0 / 28.0, 1.0 / 14.0, 1.0 / 7.0)
    independent_probability: float = 0.60
    checkerboard_probability: float = 0.20
    stripe_probability: float = 0.20
    random_offset: bool = True

    def __post_init__(self) -> None:
        _validate_probability(self.probability, name="GridOcclusionConfig.probability")
        if not self.patch_fractions or any(
            not np.isfinite(value) or not 0.0 < float(value) <= 1.0 for value in self.patch_fractions
        ):
            raise ValueError("GridOcclusionConfig.patch_fractions must contain values within (0, 1]")
        subprobabilities = (
            float(self.independent_probability),
            float(self.checkerboard_probability),
            float(self.stripe_probability),
        )
        for index, value in enumerate(subprobabilities):
            _validate_probability(value, name=f"GridOcclusionConfig.subprobability[{index}]")
        if not np.isclose(sum(subprobabilities), 1.0, atol=1e-6):
            raise ValueError("GridOcclusionConfig submode probabilities must sum to 1")


@dataclass(frozen=True)
class ViewOcclusionConfig:
    enabled: bool = False
    enable_epoch_variation: bool | None = None
    temporal_scope: TemporalScope = "clip"
    generation_max_side: int = 192
    mask_cache_size: int = 16
    rectangle: RectangleOcclusionConfig = field(default_factory=RectangleOcclusionConfig)
    blob: BlobOcclusionConfig = field(default_factory=BlobOcclusionConfig)
    grid: GridOcclusionConfig = field(default_factory=GridOcclusionConfig)

    def __post_init__(self) -> None:
        if self.temporal_scope not in ("frame", "clip"):
            raise ValueError("ViewOcclusionConfig.temporal_scope must be 'frame' or 'clip'")
        if int(self.generation_max_side) < 32:
            raise ValueError("ViewOcclusionConfig.generation_max_side must be >= 32")
        if int(self.mask_cache_size) < 1:
            raise ValueError("ViewOcclusionConfig.mask_cache_size must be >= 1")
        probabilities = (
            float(self.rectangle.probability),
            float(self.blob.probability),
            float(self.grid.probability),
        )
        if not np.isclose(sum(probabilities), 1.0, atol=1e-6):
            raise ValueError("view-occlusion method probabilities must sum to 1")


def validate_occlusion_ratio(value: float) -> float:
    ratio = float(value)
    if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("occlusion_ratio must be finite and within [0, 1]")
    return ratio


def _camera_hash(camera_key: str) -> int:
    return sum((index + 1) * ord(ch) for index, ch in enumerate(str(camera_key)))


def _seed(
    *,
    global_seed: int,
    episode_index: int,
    camera_key: str,
    epoch: int,
    scope_anchor: int,
) -> int:
    return (
        int(global_seed)
        + int(episode_index) * 100003
        + _camera_hash(f"{camera_key}:view_occlusion") * 1009
        + int(epoch) * 10000019
        + int(scope_anchor) * 9176
    )


def _top_k_mask(score: np.ndarray, target_pixels: int) -> np.ndarray:
    flat = np.asarray(score, dtype=np.float64).reshape(-1)
    target = int(target_pixels)
    if target <= 0:
        return np.zeros(score.shape, dtype=bool)
    if target >= flat.size:
        return np.ones(score.shape, dtype=bool)
    selected = np.argpartition(flat, flat.size - target)[flat.size - target :]
    mask = np.zeros(flat.size, dtype=bool)
    mask[selected] = True
    return mask.reshape(score.shape)


def _resize_bilinear_2d(source: np.ndarray, height: int, width: int) -> np.ndarray:
    """Small dependency-free bilinear resize for low-frequency score fields."""

    array = np.asarray(source, dtype=np.float64)
    source_y = np.linspace(0.0, 1.0, array.shape[0])
    source_x = np.linspace(0.0, 1.0, array.shape[1])
    target_y = np.linspace(0.0, 1.0, int(height))
    target_x = np.linspace(0.0, 1.0, int(width))
    horizontal = np.stack([np.interp(target_x, source_x, row) for row in array], axis=0)
    return np.stack(
        [np.interp(target_y, source_y, horizontal[:, column]) for column in range(horizontal.shape[1])],
        axis=1,
    )


class ViewOcclusionAugmenter:
    """Deterministic high-resolution view occlusion applied before normalization."""

    def __init__(self, config: ViewOcclusionConfig | None = None, *, cache_size: int = 128):
        self.config = ViewOcclusionConfig() if config is None else config
        self.cache_size = max(1, min(int(cache_size), int(self.config.mask_cache_size)))
        self._masks: OrderedDict[tuple[Any, ...], tuple[np.ndarray, dict[str, Any]]] = OrderedDict()

    def _method(self, random: np.random.Generator) -> OcclusionMethod:
        value = float(random.random())
        rectangle_end = float(self.config.rectangle.probability)
        blob_end = rectangle_end + float(self.config.blob.probability)
        if value < rectangle_end:
            return "random_rectangle"
        if value < blob_end:
            return "random_blob"
        return "random_grid"

    @staticmethod
    def _log_uniform(random: np.random.Generator, values: tuple[float, float]) -> float:
        low, high = float(values[0]), float(values[1])
        return exp(float(random.uniform(log(low), log(high))))

    def _rectangle_score(
        self, random: np.random.Generator, height: int, width: int, target_ratio: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        config = self.config.rectangle
        low, high = int(config.block_count_range[0]), int(config.block_count_range[1])
        block_count = int(random.integers(low, high + 1))
        yy, xx = np.mgrid[0:height, 0:width]
        base_x = float(random.uniform(0.0, max(1.0, width - 1.0)))
        base_y = float(random.uniform(0.0, max(1.0, height - 1.0)))
        base_sigma_x = max(1.0, float(width) * float(random.uniform(0.35, 0.80)))
        base_sigma_y = max(1.0, float(height) * float(random.uniform(0.35, 0.80)))
        # A very weak broad field makes any required area outside the sampled
        # rectangles contiguous instead of degenerating into pixel dropout.
        score = 1e-6 * np.exp(
            -0.5 * (((xx - base_x) / base_sigma_x) ** 2 + ((yy - base_y) / base_sigma_y) ** 2)
        )
        score += random.random((height, width)) * 1e-12
        blocks: list[dict[str, Any]] = []
        # Stratified centers spread blocks across the image while retaining random placement.
        center_slots = random.permutation(max(block_count, 1))
        target_per_block = max(float(target_ratio) / float(block_count), 1.0 / float(height * width))
        for index in range(block_count):
            sampled_area = self._log_uniform(random, config.area_fraction_range)
            area_fraction = min(
                float(config.area_fraction_range[1]),
                max(min(float(config.area_fraction_range[0]), target_per_block), target_per_block * 1.8, sampled_area),
            )
            aspect = self._log_uniform(random, config.aspect_ratio_range)
            block_width = max(1, min(width, int(round((area_fraction * height * width * aspect) ** 0.5))))
            block_height = max(1, min(height, int(round(area_fraction * height * width / block_width))))
            slot = int(center_slots[index])
            center_x = ((slot + float(random.random())) / block_count) * width
            center_y = float(random.random()) * height
            x0 = int(np.clip(round(center_x - block_width / 2), 0, max(0, width - block_width)))
            y0 = int(np.clip(round(center_y - block_height / 2), 0, max(0, height - block_height)))
            weight = 1.0 + float(random.random())
            score[y0 : y0 + block_height, x0 : x0 + block_width] += weight
            blocks.append(
                {
                    "x": x0,
                    "y": y0,
                    "width": block_width,
                    "height": block_height,
                    "aspect_ratio": float(block_width) / float(block_height),
                }
            )
        return score, {"block_count": block_count, "blocks": blocks}

    def _blob_score(
        self, random: np.random.Generator, height: int, width: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        config = self.config.blob
        low, high = int(config.blob_count_range[0]), int(config.blob_count_range[1])
        blob_count = int(random.integers(low, high + 1))
        yy, xx = np.mgrid[0:height, 0:width]
        score = np.zeros((height, width), dtype=np.float64)
        blobs: list[dict[str, Any]] = []
        for _ in range(blob_count):
            center_x = float(random.uniform(0.0, max(1.0, width - 1.0)))
            center_y = float(random.uniform(0.0, max(1.0, height - 1.0)))
            sigma_x = max(1.0, width * self._log_uniform(random, config.sigma_fraction_range))
            sigma_y = max(1.0, height * self._log_uniform(random, config.sigma_fraction_range))
            angle = float(random.uniform(0.0, np.pi))
            cosine, sine = float(np.cos(angle)), float(np.sin(angle))
            dx, dy = xx - center_x, yy - center_y
            rotated_x = cosine * dx + sine * dy
            rotated_y = -sine * dx + cosine * dy
            strength = float(random.uniform(0.5, 1.5))
            score += strength * np.exp(
                -0.5 * ((rotated_x / sigma_x) ** 2 + (rotated_y / sigma_y) ** 2)
            )
            blobs.append(
                {
                    "center_x": center_x,
                    "center_y": center_y,
                    "sigma_x": sigma_x,
                    "sigma_y": sigma_y,
                    "angle_deg": float(np.degrees(angle)),
                }
            )
        # A smooth multi-scale field breaks overly elliptical boundaries without pixel dropout.
        coarse_h, coarse_w = max(2, min(16, height)), max(2, min(16, width))
        coarse = random.random((coarse_h, coarse_w))
        low_frequency = _resize_bilinear_2d(coarse, height, width)
        score += float(config.low_frequency_noise_weight) * low_frequency
        score += random.random((height, width)) * 1e-12
        return score, {"blob_count": blob_count, "blobs": blobs, "low_frequency_noise": True}

    def _grid_score(
        self, random: np.random.Generator, height: int, width: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        config = self.config.grid
        patch_fraction = float(random.choice(np.asarray(config.patch_fractions, dtype=np.float64)))
        patch_height = max(1, int(round(height * patch_fraction)))
        patch_width = max(1, int(round(width * patch_fraction)))
        offset_y = int(random.integers(0, patch_height)) if config.random_offset else 0
        offset_x = int(random.integers(0, patch_width)) if config.random_offset else 0
        value = float(random.random())
        independent_end = float(config.independent_probability)
        checkerboard_end = independent_end + float(config.checkerboard_probability)
        mode = "independent" if value < independent_end else "checkerboard" if value < checkerboard_end else "stripe"
        yy, xx = np.mgrid[0:height, 0:width]
        fine_direction_x = -1.0 if bool(random.integers(0, 2)) else 1.0
        fine_direction_y = -1.0 if bool(random.integers(0, 2)) else 1.0
        fine_score = (
            fine_direction_x * xx / max(1, width - 1)
            + fine_direction_y * yy / max(1, height - 1)
        )
        score = np.asarray(fine_score, dtype=np.float64) * 1e-6
        row_index = 0
        patch_count = 0
        stripe_by_row = bool(random.integers(0, 2))
        for y0 in range(-offset_y, height, patch_height):
            column_index = 0
            for x0 in range(-offset_x, width, patch_width):
                ys, ye = max(0, y0), min(height, y0 + patch_height)
                xs, xe = max(0, x0), min(width, x0 + patch_width)
                if ys >= ye or xs >= xe:
                    column_index += 1
                    continue
                if mode == "independent":
                    cell_score = float(random.random())
                elif mode == "checkerboard":
                    phase = (row_index + column_index) % 2
                    cell_score = float(phase) + float(random.random()) * 0.25
                else:
                    stripe = row_index if stripe_by_row else column_index
                    cell_score = float(random.random()) * 0.25 + float(stripe % 2)
                score[ys:ye, xs:xe] = cell_score
                patch_count += 1
                column_index += 1
            row_index += 1
        score += random.random((height, width)) * 1e-12
        return score, {
            "grid_mode": mode,
            "patch_fraction": patch_fraction,
            "patch_height": patch_height,
            "patch_width": patch_width,
            "patch_count": patch_count,
            "offset_x": offset_x,
            "offset_y": offset_y,
        }

    def generate(
        self,
        *,
        height: int,
        width: int,
        occlusion_ratio: float,
        global_seed: int,
        episode_index: int,
        camera_key: str,
        epoch: int,
        scope_anchor: int,
        forced_missing_view: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        ratio = validate_occlusion_ratio(occlusion_ratio)
        if bool(forced_missing_view) and ratio != 1.0:
            raise ValueError("forced_missing_view requires occlusion_ratio=1")
        height, width = int(height), int(width)
        if height < 1 or width < 1:
            raise ValueError("view-occlusion image dimensions must be positive")
        if not self.config.enabled and ratio != 0.0 and not bool(forced_missing_view):
            raise ValueError("non-zero occlusion_ratio requires ViewOcclusionConfig(enabled=True)")
        target_pixels = int(round(ratio * height * width))
        effective_anchor = int(scope_anchor) if self.config.temporal_scope == "frame" else int(scope_anchor)
        seed = _seed(
            global_seed=int(global_seed),
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            epoch=int(epoch),
            scope_anchor=effective_anchor,
        )
        cache_key = (
            height,
            width,
            target_pixels,
            ratio,
            int(global_seed),
            int(episode_index),
            str(camera_key),
            int(epoch),
            effective_anchor,
        )
        cached = self._masks.get(cache_key)
        if cached is not None:
            self._masks.move_to_end(cache_key)
            mask, base_metadata = cached
        else:
            random = np.random.default_rng(seed)
            if target_pixels == 0:
                method: OcclusionMethod = "none"
                mask = np.zeros((height, width), dtype=bool)
                method_parameters: dict[str, Any] = {}
            elif target_pixels == height * width:
                method = "full_black"
                mask = np.ones((height, width), dtype=bool)
                method_parameters = {}
            else:
                scale = min(1.0, float(self.config.generation_max_side) / float(max(height, width)))
                generation_height = max(1, int(round(height * scale)))
                generation_width = max(1, int(round(width * scale)))
                generation_target = int(round(ratio * generation_height * generation_width))
                method = self._method(random)
                if method == "random_rectangle":
                    score, method_parameters = self._rectangle_score(
                        random, generation_height, generation_width, ratio
                    )
                elif method == "random_blob":
                    score, method_parameters = self._blob_score(
                        random, generation_height, generation_width
                    )
                else:
                    score, method_parameters = self._grid_score(
                        random, generation_height, generation_width
                    )
                generation_mask = _top_k_mask(score, generation_target)
                if (generation_height, generation_width) == (height, width):
                    mask = generation_mask
                else:
                    source_y = np.minimum(
                        generation_height - 1,
                        np.arange(height, dtype=np.int64) * generation_height // height,
                    )
                    source_x = np.minimum(
                        generation_width - 1,
                        np.arange(width, dtype=np.int64) * generation_width // width,
                    )
                    mask = generation_mask[source_y[:, None], source_x[None, :]]
                method_parameters = dict(method_parameters)
                method_parameters["generation_shape"] = [generation_height, generation_width]
            actual_pixels = int(np.count_nonzero(mask))
            base_metadata = {
                "type": "view_occlusion_v1",
                "method": method,
                "requested_ratio": ratio,
                "actual_ratio": float(actual_pixels) / float(height * width),
                "target_pixels": target_pixels,
                "actual_pixels": actual_pixels,
                "temporal_scope": self.config.temporal_scope,
                "effective_epoch": int(epoch),
                "scope_anchor": effective_anchor,
                "seed": seed,
                "method_parameters": method_parameters,
            }
            if target_pixels not in (0, height * width):
                self._masks[cache_key] = (mask, base_metadata)
                if len(self._masks) > self.cache_size:
                    self._masks.popitem(last=False)
        metadata = dict(base_metadata)
        metadata["forced_missing_view"] = bool(forced_missing_view)
        return mask, metadata

    def apply(
        self,
        image: np.ndarray,
        *,
        occlusion_ratio: float,
        global_seed: int,
        episode_index: int,
        camera_key: str,
        epoch: int,
        scope_anchor: int,
        forced_missing_view: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"view-occlusion image must be HxWx3, got shape={array.shape!r}")
        ratio = validate_occlusion_ratio(occlusion_ratio)
        if bool(forced_missing_view) and ratio != 1.0:
            raise ValueError("forced_missing_view requires occlusion_ratio=1")
        if not self.config.enabled and ratio != 0.0 and not bool(forced_missing_view):
            raise ValueError("non-zero occlusion_ratio requires ViewOcclusionConfig(enabled=True)")
        if ratio == 0.0:
            metadata = self.generate(
                height=int(array.shape[0]),
                width=int(array.shape[1]),
                occlusion_ratio=ratio,
                global_seed=global_seed,
                episode_index=episode_index,
                camera_key=camera_key,
                epoch=epoch,
                scope_anchor=scope_anchor,
                forced_missing_view=forced_missing_view,
            )[1]
            return array.copy(), metadata
        if ratio == 1.0:
            metadata = self.generate(
                height=int(array.shape[0]),
                width=int(array.shape[1]),
                occlusion_ratio=ratio,
                global_seed=global_seed,
                episode_index=episode_index,
                camera_key=camera_key,
                epoch=epoch,
                scope_anchor=scope_anchor,
                forced_missing_view=forced_missing_view,
            )[1]
            return np.zeros_like(array), metadata
        mask, metadata = self.generate(
            height=int(array.shape[0]),
            width=int(array.shape[1]),
            occlusion_ratio=occlusion_ratio,
            global_seed=global_seed,
            episode_index=episode_index,
            camera_key=camera_key,
            epoch=epoch,
            scope_anchor=scope_anchor,
            forced_missing_view=forced_missing_view,
        )
        output = np.empty_like(array)
        np.multiply(array, np.logical_not(mask)[..., None], out=output, casting="unsafe")
        return output, metadata


__all__ = [
    "BlobOcclusionConfig",
    "GridOcclusionConfig",
    "OcclusionMethod",
    "RectangleOcclusionConfig",
    "TemporalScope",
    "ViewOcclusionAugmenter",
    "ViewOcclusionConfig",
    "validate_occlusion_ratio",
]
