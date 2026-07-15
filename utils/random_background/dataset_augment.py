from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from .background import BackgroundMode, TemporalRandomBackground, apply_random_background, temporal_background_from_episode
from .mask_io import BitpackMaskReader, MaskRole, normalize_keep_mask
from .photometric import PhotometricAugmentConfig, PhotometricImageAugmenter
from .real_image_provider import LocalImageBackgroundProvider, RealImageTemporalBackground
from .view_occlusion import ViewOcclusionAugmenter, ViewOcclusionConfig, validate_occlusion_ratio


ProviderKind = Literal["procedural", "real_image", "original", "black", "mixed"]
FixedMaskProviderKind = Literal["inherit", "procedural", "real_image"]

_FIXED_BORDER_SIDES = ("top", "bottom", "left", "right")


def _camera_hash(camera_key: str) -> int:
    return sum((index + 1) * ord(ch) for index, ch in enumerate(str(camera_key)))


@dataclass(frozen=True)
class FixedBorderMaskConfig:
    """Static, per-episode border replacement for the front camera.

    The mask itself and every selected edge's background are deterministic for
    one ``episode + epoch``.  Unlike wrist-camera backgrounds, the selected
    background is always sampled at frame zero and therefore never moves
    during an episode.
    """

    enabled: bool = False
    side_probability: float = 0.25
    thickness_fraction: float = 0.05
    provider: FixedMaskProviderKind = "inherit"
    # ``None`` intentionally follows RandomBackgroundAugmentConfig so callers
    # can switch epoch variation for all augmentation in one place.
    enable_epoch_variation: bool | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.side_probability) <= 1.0:
            raise ValueError("FixedBorderMaskConfig.side_probability must be within [0, 1]")
        if not 0.0 < float(self.thickness_fraction) <= 0.5:
            raise ValueError("FixedBorderMaskConfig.thickness_fraction must be within (0, 0.5]")


@dataclass(frozen=True)
class RandomBackgroundAugmentConfig:
    global_seed: int = 20260709
    provider: ProviderKind = "procedural"
    mode: BackgroundMode = "embodied_mix"
    real_image_prob: float = 0.4
    mask_role: MaskRole = "keep"
    edge_blur_px: int = 1
    enable_epoch_variation: bool = True
    mixed_original_prob: float = 0.1
    mixed_black_prob: float = 0.1
    fixed_mask: FixedBorderMaskConfig = field(default_factory=FixedBorderMaskConfig)
    photometric: PhotometricAugmentConfig = field(default_factory=PhotometricAugmentConfig)
    view_occlusion: ViewOcclusionConfig = field(default_factory=ViewOcclusionConfig)


@dataclass(frozen=True)
class _FixedBorderMaskPlan:
    selected_sides: tuple[str, ...]
    replace_masks: dict[str, np.ndarray]
    thickness_px: dict[str, int]


def _fixed_mask_seed(*, seed: int, episode_index: int, camera_key: str, epoch: int) -> int:
    return (
        int(seed)
        + int(episode_index) * 100003
        + _camera_hash(f"{camera_key}:fixed_border") * 1009
        + int(epoch) * 10000019
    )


class _LruFixedBorderMaskPlans:
    def __init__(self, max_size: int = 128):
        self.max_size = max(1, int(max_size))
        self.items: OrderedDict[tuple[int, int, int, str, int], _FixedBorderMaskPlan] = OrderedDict()

    def get(
        self,
        *,
        width: int,
        height: int,
        episode_index: int,
        camera_key: str,
        epoch: int,
        seed: int,
        side_probability: float,
        thickness_fraction: float,
    ) -> _FixedBorderMaskPlan:
        key = (int(width), int(height), int(episode_index), str(camera_key), int(epoch))
        plan = self.items.get(key)
        if plan is not None:
            self.items.move_to_end(key)
            return plan

        random = np.random.default_rng(
            _fixed_mask_seed(
                seed=int(seed),
                episode_index=int(episode_index),
                camera_key=str(camera_key),
                epoch=int(epoch),
            )
        )
        selected_sides = tuple(
            side for side in _FIXED_BORDER_SIDES if float(random.random()) < float(side_probability)
        )
        vertical_thickness = max(1, int(round(int(height) * float(thickness_fraction))))
        horizontal_thickness = max(1, int(round(int(width) * float(thickness_fraction))))
        thickness_px = {
            "top": vertical_thickness,
            "bottom": vertical_thickness,
            "left": horizontal_thickness,
            "right": horizontal_thickness,
        }
        regions = {
            "top": (slice(0, vertical_thickness), slice(0, int(width))),
            "bottom": (slice(int(height) - vertical_thickness, int(height)), slice(0, int(width))),
            "left": (slice(0, int(height)), slice(0, horizontal_thickness)),
            "right": (slice(0, int(height)), slice(int(width) - horizontal_thickness, int(width))),
        }
        assigned = np.zeros((int(height), int(width)), dtype=bool)
        replace_masks: dict[str, np.ndarray] = {}
        for side in selected_sides:
            candidate = np.zeros_like(assigned)
            rows, columns = regions[side]
            candidate[rows, columns] = True
            # Border strips overlap at corners.  Canonical side order makes
            # ownership deterministic while keeping each pixel sourced from
            # exactly one independently sampled static background.
            replace_mask = candidate & ~assigned
            assigned |= replace_mask
            replace_masks[side] = replace_mask
        plan = _FixedBorderMaskPlan(
            selected_sides=selected_sides,
            replace_masks=replace_masks,
            thickness_px=thickness_px,
        )
        self.items[key] = plan
        if len(self.items) > self.max_size:
            self.items.popitem(last=False)
        return plan


class _LruStaticBackgroundFrames:
    def __init__(self, max_size: int = 128):
        self.max_size = max(1, int(max_size))
        self.items: OrderedDict[tuple[int, int, int, str, int, str], np.ndarray] = OrderedDict()

    def get(self, key: tuple[int, int, int, str, int, str], factory: Callable[[], np.ndarray]) -> np.ndarray:
        frame = self.items.get(key)
        if frame is not None:
            self.items.move_to_end(key)
            return frame
        frame = np.asarray(factory(), dtype=np.uint8)
        self.items[key] = frame
        if len(self.items) > self.max_size:
            self.items.popitem(last=False)
        return frame


class _LruStreams:
    def __init__(self, max_size: int = 128):
        self.max_size = max(1, int(max_size))
        self.items: OrderedDict[tuple[int, str, int, int, int, str], TemporalRandomBackground] = OrderedDict()

    def get(
        self,
        *,
        width: int,
        height: int,
        episode_index: int,
        camera_key: str,
        epoch: int,
        seed: int,
        mode: BackgroundMode,
    ) -> TemporalRandomBackground:
        key = (int(width), int(height), int(episode_index), _camera_hash(camera_key), int(epoch), str(mode))
        stream = self.items.get(key)
        if stream is not None:
            self.items.move_to_end(key)
            return stream
        stream = temporal_background_from_episode(
            width=int(width),
            height=int(height),
            episode_index=int(episode_index) + int(epoch) * 10000019,
            camera_key=str(camera_key),
            seed=int(seed),
            mode=mode,
        )
        self.items[key] = stream
        if len(self.items) > self.max_size:
            self.items.popitem(last=False)
        return stream


class _LruRealStreams:
    def __init__(self, max_size: int = 128):
        self.max_size = max(1, int(max_size))
        self.items: OrderedDict[tuple[int, int, int, str], RealImageTemporalBackground] = OrderedDict()

    def get(
        self,
        provider: LocalImageBackgroundProvider,
        *,
        width: int,
        height: int,
        episode_index: int,
        camera_key: str,
        epoch: int,
    ) -> RealImageTemporalBackground:
        key = (int(width), int(height), int(episode_index), f"{camera_key}:{int(epoch)}")
        stream = self.items.get(key)
        if stream is not None:
            self.items.move_to_end(key)
            return stream
        stream = provider.stream(
            width=int(width),
            height=int(height),
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            epoch=int(epoch),
        )
        self.items[key] = stream
        if len(self.items) > self.max_size:
            self.items.popitem(last=False)
        return stream


class RandomBackgroundVideoAugmenter:
    """Training-time random background compositor for RGB frames and masks.

    The class is intentionally stateless with respect to dataset storage. It can
    be used inside a DataLoader worker after the worker has decoded an RGB frame
    and the corresponding mask.
    """

    def __init__(
        self,
        config: RandomBackgroundAugmentConfig | None = None,
        *,
        stream_cache_size: int = 128,
        real_image_provider: LocalImageBackgroundProvider | None = None,
    ):
        self.config = config if config is not None else RandomBackgroundAugmentConfig()
        self.streams = _LruStreams(max_size=stream_cache_size)
        self.real_streams = _LruRealStreams(max_size=stream_cache_size)
        self.fixed_mask_plans = _LruFixedBorderMaskPlans(max_size=stream_cache_size)
        self.fixed_backgrounds = _LruStaticBackgroundFrames(max_size=stream_cache_size)
        self.photometric_augmenter = PhotometricImageAugmenter(
            self.config.photometric,
            cache_size=stream_cache_size,
        )
        self.view_occlusion_augmenter = ViewOcclusionAugmenter(
            self.config.view_occlusion,
            cache_size=stream_cache_size,
        )
        self.real_image_provider = real_image_provider

    def _provider_for_sample(self, *, episode_index: int, camera_key: str, frame_index: int, epoch: int) -> ProviderKind:
        provider = self.config.provider
        if provider != "mixed":
            return provider
        seed = (
            int(self.config.global_seed)
            + int(episode_index) * 100003
            + _camera_hash(camera_key)
            + int(frame_index) * 9176
            + int(epoch) * 10000019
        )
        value = float(np.random.default_rng(seed).random())
        if value < float(self.config.mixed_original_prob):
            return "original"
        if value < float(self.config.mixed_original_prob) + float(self.config.mixed_black_prob):
            return "black"
        if (
            self.real_image_provider is not None
            and value
            < float(self.config.mixed_original_prob)
            + float(self.config.mixed_black_prob)
            + float(self.config.real_image_prob)
        ):
            return "real_image"
        return "procedural"

    def uses_fixed_border_mask(self, camera_key: str) -> bool:
        """Whether the configured fixed replacement mask applies to this camera."""

        return bool(self.config.fixed_mask.enabled and str(camera_key) == "front")

    def _fixed_epoch_variation_enabled(self) -> bool:
        enabled = self.config.fixed_mask.enable_epoch_variation
        if enabled is None:
            enabled = self.config.enable_epoch_variation
        return bool(enabled)

    def _fixed_effective_epoch(self, epoch: int) -> int:
        return int(epoch) if self._fixed_epoch_variation_enabled() else 0

    def _photometric_epoch_variation_enabled(self) -> bool:
        enabled = self.config.photometric.enable_epoch_variation
        if enabled is None:
            enabled = self.config.enable_epoch_variation
        return bool(enabled)

    def _photometric_effective_epoch(self, epoch: int) -> int:
        return int(epoch) if self._photometric_epoch_variation_enabled() else 0

    def _view_occlusion_epoch_variation_enabled(self) -> bool:
        enabled = self.config.view_occlusion.enable_epoch_variation
        if enabled is None:
            enabled = self.config.enable_epoch_variation
        return bool(enabled)

    def _view_occlusion_effective_epoch(self, epoch: int) -> int:
        return int(epoch) if self._view_occlusion_epoch_variation_enabled() else 0

    def _validate_view_occlusion_request(
        self,
        occlusion_ratio: float,
        *,
        forced_missing_view: bool,
    ) -> float:
        ratio = validate_occlusion_ratio(occlusion_ratio)
        if bool(forced_missing_view) and ratio != 1.0:
            raise ValueError("forced_missing_view requires occlusion_ratio=1")
        if (
            not self.config.view_occlusion.enabled
            and ratio != 0.0
            and not bool(forced_missing_view)
        ):
            raise ValueError("non-zero occlusion_ratio requires ViewOcclusionConfig(enabled=True)")
        return ratio

    @staticmethod
    def _fixed_side_camera_key(camera_key: str, side: str) -> str:
        return f"{camera_key}:fixed_border:{side}"

    def _fixed_border_plan(
        self,
        *,
        width: int,
        height: int,
        episode_index: int,
        camera_key: str,
        epoch: int,
    ) -> _FixedBorderMaskPlan:
        fixed_mask = self.config.fixed_mask
        return self.fixed_mask_plans.get(
            width=int(width),
            height=int(height),
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            epoch=self._fixed_effective_epoch(epoch),
            seed=int(self.config.global_seed),
            side_probability=float(fixed_mask.side_probability),
            thickness_fraction=float(fixed_mask.thickness_fraction),
        )

    def _fixed_provider_for_side(
        self,
        *,
        episode_index: int,
        camera_key: str,
        side: str,
        epoch: int,
    ) -> Literal["procedural", "real_image"]:
        requested = self.config.fixed_mask.provider
        if requested != "inherit":
            return requested

        provider = self.config.provider
        if provider in ("procedural", "real_image"):
            return provider
        if provider == "mixed":
            side_key = self._fixed_side_camera_key(camera_key, side)
            seed = _fixed_mask_seed(
                seed=int(self.config.global_seed),
                episode_index=int(episode_index),
                camera_key=side_key,
                epoch=self._fixed_effective_epoch(epoch),
            )
            use_real_image = (
                self.real_image_provider is not None
                and float(np.random.default_rng(seed).random()) < float(self.config.real_image_prob)
            )
            return "real_image" if use_real_image else "procedural"
        raise ValueError(
            "fixed front-border masks require a random background provider; "
            "set FixedBorderMaskConfig(provider='procedural' or 'real_image') "
            "when the base provider is 'original' or 'black'"
        )

    def _fixed_background_for_side(
        self,
        *,
        width: int,
        height: int,
        episode_index: int,
        camera_key: str,
        side: str,
        epoch: int,
        provider: Literal["procedural", "real_image"],
    ) -> np.ndarray:
        effective_epoch = self._fixed_effective_epoch(epoch)
        side_key = self._fixed_side_camera_key(camera_key, side)
        cache_key = (
            int(width),
            int(height),
            int(episode_index),
            side_key,
            effective_epoch,
            str(provider),
        )

        def build() -> np.ndarray:
            if provider == "real_image":
                if self.real_image_provider is None:
                    raise ValueError("fixed border provider='real_image' requires real_image_provider")
                stream = self.real_streams.get(
                    self.real_image_provider,
                    width=int(width),
                    height=int(height),
                    episode_index=int(episode_index),
                    camera_key=side_key,
                    epoch=effective_epoch,
                )
                return stream.frame(0)
            stream = self.streams.get(
                width=int(width),
                height=int(height),
                episode_index=int(episode_index),
                camera_key=side_key,
                epoch=effective_epoch,
                seed=int(self.config.global_seed),
                mode=self.config.mode,
            )
            return stream.frame(0)

        return self.fixed_backgrounds.get(cache_key, build)

    def fixed_mask_metadata(
        self,
        *,
        width: int,
        height: int,
        episode_index: int,
        camera_key: str,
        epoch: int,
    ) -> dict[str, Any] | None:
        """Return audit metadata for a fixed mask without adding model fields."""

        if not self.uses_fixed_border_mask(camera_key):
            return None
        plan = self._fixed_border_plan(
            width=int(width),
            height=int(height),
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            epoch=int(epoch),
        )
        effective_epoch = self._fixed_effective_epoch(epoch)
        return {
            "type": "front_fixed_border_replace",
            "static_scope": "episode_epoch" if self._fixed_epoch_variation_enabled() else "episode",
            "effective_epoch": effective_epoch,
            "selected_sides": list(plan.selected_sides),
            "side_probability": float(self.config.fixed_mask.side_probability),
            "thickness_fraction": float(self.config.fixed_mask.thickness_fraction),
            "thickness_px": dict(plan.thickness_px),
            "backgrounds": {
                side: {
                    "provider": self._fixed_provider_for_side(
                        episode_index=int(episode_index),
                        camera_key=str(camera_key),
                        side=side,
                        epoch=int(epoch),
                    ),
                    "camera_key": self._fixed_side_camera_key(str(camera_key), side),
                    "frame_index": 0,
                }
                for side in plan.selected_sides
            },
        }

    def photometric_metadata(
        self,
        *,
        width: int,
        height: int,
        episode_index: int,
        camera_key: str,
        epoch: int,
    ) -> dict[str, Any] | None:
        effective_epoch = self._photometric_effective_epoch(epoch)
        return self.photometric_augmenter.metadata(
            width=int(width),
            height=int(height),
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            epoch=effective_epoch,
            global_seed=int(self.config.global_seed),
            static_scope="episode_camera_epoch" if self._photometric_epoch_variation_enabled() else "episode_camera",
        )

    def _apply_fixed_border_mask(
        self,
        image: np.ndarray,
        *,
        episode_index: int,
        camera_key: str,
        epoch: int,
    ) -> np.ndarray:
        if not self.uses_fixed_border_mask(camera_key):
            return image
        height, width = image.shape[:2]
        plan = self._fixed_border_plan(
            width=width,
            height=height,
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            epoch=int(epoch),
        )
        if not plan.selected_sides:
            return image.copy()

        out = image.copy()
        for side in plan.selected_sides:
            keep_mask = np.ones((height, width), dtype=np.uint8)
            keep_mask[plan.replace_masks[side]] = 0
            provider = self._fixed_provider_for_side(
                episode_index=int(episode_index),
                camera_key=str(camera_key),
                side=side,
                epoch=int(epoch),
            )
            background = self._fixed_background_for_side(
                width=width,
                height=height,
                episode_index=int(episode_index),
                camera_key=str(camera_key),
                side=side,
                epoch=int(epoch),
                provider=provider,
            )
            out = apply_random_background(
                out,
                keep_mask,
                background=background,
                edge_blur_px=int(self.config.edge_blur_px),
            )
        return out

    def _apply_composited(
        self,
        rgb: np.ndarray,
        mask: Any,
        *,
        episode_index: int,
        camera_key: str,
        frame_index: int,
        epoch: int = 0,
        mask_role: MaskRole | None = None,
    ) -> np.ndarray:
        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"rgb must be HxWx3 uint8/BGR array, got shape={image.shape!r}")
        role = self.config.mask_role if mask_role is None else mask_role
        keep_mask = normalize_keep_mask(mask, mask_role=role)
        fixed_mask_applies = self.uses_fixed_border_mask(camera_key)
        if fixed_mask_applies and bool(np.all(keep_mask > 0)):
            augmented = image.copy()
        else:
            provider = self._provider_for_sample(
                episode_index=episode_index,
                camera_key=camera_key,
                frame_index=frame_index,
                epoch=epoch if self.config.enable_epoch_variation else 0,
            )
            if provider == "original":
                augmented = image.copy()
            elif provider == "black":
                augmented = apply_random_background(
                    image,
                    keep_mask,
                    background=np.zeros_like(image),
                    edge_blur_px=int(self.config.edge_blur_px),
                )
            else:
                height, width = image.shape[:2]
                effective_epoch = int(epoch) if self.config.enable_epoch_variation else 0
                if provider == "real_image":
                    if self.real_image_provider is None:
                        raise ValueError("provider='real_image' requires real_image_provider")
                    stream = self.real_streams.get(
                        self.real_image_provider,
                        width=width,
                        height=height,
                        episode_index=int(episode_index),
                        camera_key=str(camera_key),
                        epoch=effective_epoch,
                    )
                    augmented = apply_random_background(
                        image,
                        keep_mask,
                        background=stream.frame(int(frame_index)),
                        edge_blur_px=int(self.config.edge_blur_px),
                    )
                elif provider == "procedural":
                    stream = self.streams.get(
                        width=width,
                        height=height,
                        episode_index=int(episode_index),
                        camera_key=str(camera_key),
                        epoch=effective_epoch,
                        seed=int(self.config.global_seed),
                        mode=self.config.mode,
                    )
                    augmented = stream.apply(
                        image,
                        keep_mask,
                        int(frame_index),
                        edge_blur_px=int(self.config.edge_blur_px),
                    )
                else:
                    raise ValueError(f"unsupported background provider: {provider!r}")
        return self._apply_fixed_border_mask(
            augmented,
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            epoch=int(epoch),
        )

    def apply(
        self,
        rgb: np.ndarray,
        mask: Any,
        *,
        episode_index: int,
        camera_key: str,
        frame_index: int,
        epoch: int = 0,
        mask_role: MaskRole | None = None,
        occlusion_ratio: float = 0.0,
        occlusion_scope_anchor: int | None = None,
        forced_missing_view: bool = False,
    ) -> np.ndarray:
        ratio = self._validate_view_occlusion_request(
            occlusion_ratio,
            forced_missing_view=forced_missing_view,
        )
        source = np.asarray(rgb)
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError(f"rgb must be HxWx3 uint8/BGR array, got shape={source.shape!r}")
        # A fully missing view is independent of background, mask and
        # photometric plans.  Skip all of them, including mask normalization.
        if ratio == 1.0:
            return np.zeros(source.shape, dtype=np.uint8)
        composited = self._apply_composited(
            rgb,
            mask,
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            frame_index=int(frame_index),
            epoch=int(epoch),
            mask_role=mask_role,
        )
        photometric = self.photometric_augmenter.apply(
            composited,
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            frame_index=int(frame_index),
            epoch=self._photometric_effective_epoch(epoch),
            global_seed=int(self.config.global_seed),
        )
        if ratio == 0.0:
            return photometric
        occluded, _metadata = self.view_occlusion_augmenter.apply(
            photometric,
            occlusion_ratio=ratio,
            global_seed=int(self.config.global_seed),
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            epoch=self._view_occlusion_effective_epoch(epoch),
            scope_anchor=int(frame_index if occlusion_scope_anchor is None else occlusion_scope_anchor),
            forced_missing_view=bool(forced_missing_view),
        )
        return occluded

    def apply_batch(
        self,
        rgbs: np.ndarray,
        masks: np.ndarray,
        *,
        episode_index: int,
        camera_key: str,
        frame_indices: np.ndarray | list[int],
        epoch: int = 0,
        mask_role: MaskRole | None = None,
        occlusion_ratio: float = 0.0,
        occlusion_scope_anchor: int | None = None,
        forced_missing_view: bool = False,
    ) -> np.ndarray:
        images = np.asarray(rgbs)
        mask_array = np.asarray(masks)
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"rgbs must be NxHxWx3, got shape={images.shape!r}")
        if len(images) != len(mask_array):
            raise ValueError("rgbs and masks must have the same first dimension")
        indices = [int(value) for value in frame_indices]
        if len(indices) != len(images):
            raise ValueError("frame_indices length must match batch size")
        ratio = self._validate_view_occlusion_request(
            occlusion_ratio,
            forced_missing_view=forced_missing_view,
        )
        if ratio == 1.0:
            return np.zeros(images.shape, dtype=np.uint8)
        composited = np.empty_like(images, dtype=np.uint8)
        for index, frame_index in enumerate(indices):
            composited[index] = self._apply_composited(
                images[index],
                mask_array[index],
                episode_index=episode_index,
                camera_key=camera_key,
                frame_index=frame_index,
                epoch=epoch,
                mask_role=mask_role,
            )
        photometric = self.photometric_augmenter.apply_batch(
            composited,
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            frame_indices=indices,
            epoch=self._photometric_effective_epoch(epoch),
            global_seed=int(self.config.global_seed),
        )
        if ratio == 0.0:
            return photometric
        if not indices:
            return photometric
        anchor = int(indices[0] if occlusion_scope_anchor is None else occlusion_scope_anchor)
        occluded = np.empty_like(photometric)
        for index, frame_index in enumerate(indices):
            frame_anchor = int(frame_index) if self.config.view_occlusion.temporal_scope == "frame" else anchor
            occluded[index], _metadata = self.view_occlusion_augmenter.apply(
                photometric[index],
                occlusion_ratio=ratio,
                global_seed=int(self.config.global_seed),
                episode_index=int(episode_index),
                camera_key=str(camera_key),
                epoch=self._view_occlusion_effective_epoch(epoch),
                scope_anchor=frame_anchor,
                forced_missing_view=bool(forced_missing_view),
            )
        return occluded

    def view_occlusion_metadata(
        self,
        *,
        width: int,
        height: int,
        episode_index: int,
        camera_key: str,
        epoch: int,
        frame_index: int,
        occlusion_ratio: float,
        occlusion_scope_anchor: int | None = None,
        forced_missing_view: bool = False,
    ) -> dict[str, Any] | None:
        if not self.config.view_occlusion.enabled and float(occlusion_ratio) == 0.0:
            return None
        _mask, metadata = self.view_occlusion_augmenter.generate(
            height=int(height),
            width=int(width),
            occlusion_ratio=float(occlusion_ratio),
            global_seed=int(self.config.global_seed),
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            epoch=self._view_occlusion_effective_epoch(epoch),
            scope_anchor=int(frame_index if occlusion_scope_anchor is None else occlusion_scope_anchor),
            forced_missing_view=bool(forced_missing_view),
        )
        return metadata


class MaskReaderCache:
    """Small LRU cache for bit-packed mask readers inside DataLoader workers."""

    def __init__(self, max_size: int = 32):
        self.max_size = max(1, int(max_size))
        self.items: OrderedDict[Path, BitpackMaskReader] = OrderedDict()

    def close(self) -> None:
        for reader in self.items.values():
            reader.close()
        self.items.clear()

    def read(self, manifest_path: str | Path, frame_index: int) -> np.ndarray:
        path = Path(manifest_path).expanduser().resolve()
        reader = self.items.get(path)
        if reader is None:
            reader = BitpackMaskReader(path)
            self.items[path] = reader
            if len(self.items) > self.max_size:
                _, old = self.items.popitem(last=False)
                old.close()
        else:
            self.items.move_to_end(path)
        return reader.read(frame_index)


def augment_sample_from_paths(
    rgb: np.ndarray,
    *,
    mask_manifest_path: str | Path,
    episode_index: int,
    camera_key: str,
    frame_index: int,
    augmenter: RandomBackgroundVideoAugmenter | None = None,
    mask_cache: MaskReaderCache | None = None,
    epoch: int = 0,
    occlusion_ratio: float = 0.0,
    occlusion_scope_anchor: int | None = None,
    forced_missing_view: bool = False,
) -> np.ndarray:
    """Convenience helper when the training dataset stores masks as manifests."""

    aug = augmenter if augmenter is not None else RandomBackgroundVideoAugmenter()
    cache = mask_cache if mask_cache is not None else MaskReaderCache(max_size=1)
    mask = cache.read(mask_manifest_path, frame_index)
    return aug.apply(
        rgb,
        mask,
        episode_index=episode_index,
        camera_key=camera_key,
        frame_index=frame_index,
        epoch=epoch,
        occlusion_ratio=occlusion_ratio,
        occlusion_scope_anchor=occlusion_scope_anchor,
        forced_missing_view=forced_missing_view,
    )
