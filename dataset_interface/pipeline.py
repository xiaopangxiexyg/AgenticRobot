from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

from .config import DatasetInterfaceConfig, ImageLayout
from .mask_source import FinalMaskSource
from .metadata import DexTeleopDatasetIndex, InvalidTrainingDataError, TrainingEligibility
from .missing_views import MissingViewResolver
from .sample import FrameRequest, MaskedFrameSample
from .video_source import VideoReaderCache
from utils.random_background import LocalImageBackgroundProvider, RandomBackgroundVideoAugmenter


def convert_layout(image_bgr: np.ndarray, layout: ImageLayout) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8)
    if layout == "bgr_hwc_uint8":
        return image.copy()
    if layout == "rgb_hwc_uint8":
        return image[..., ::-1].copy()
    if layout == "rgb_chw_float32":
        return np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1)).astype(np.float32) / 255.0
    raise ValueError(f"unsupported image layout: {layout!r}")


class MaskedFramePipeline:
    """Model-agnostic RGB + final-mask augmentation interface.

    The pipeline is intended to live inside each DataLoader worker. It caches
    video readers, bitpack mask readers, and background streams for speed.
    """

    def __init__(
        self,
        config: DatasetInterfaceConfig,
        *,
        real_image_provider: LocalImageBackgroundProvider | None = None,
    ):
        self.config = config
        self.index = DexTeleopDatasetIndex(config.dataset_root)
        self.missing_views = (
            MissingViewResolver(self.index, config.missing_views)
            if config.missing_views.enabled
            else None
        )
        self.videos = VideoReaderCache(max_size=config.video.reader_cache_size)
        self.masks = FinalMaskSource(
            dataset_index=self.index,
            final_mask_root=config.mask.final_mask_root,
            cache_size=config.video.mask_cache_size,
            manifest_name=config.mask.manifest_name,
        )
        self.augmenter = RandomBackgroundVideoAugmenter(
            config.augmentation,
            real_image_provider=real_image_provider,
        )
        self._warned_ineligible_training: set[tuple[int, tuple[str, ...], tuple[str, ...]]] = set()

    def close(self) -> None:
        self.videos.close()
        self.masks.close()

    def __enter__(self) -> "MaskedFramePipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _has_readable_video(self, episode_index: int, camera_key: str) -> bool:
        if not self.index.has_video(episode_index, camera_key):
            return False
        return self.index.video_path(episode_index, camera_key).is_file()

    def read_raw(self, request: FrameRequest) -> tuple[np.ndarray, np.ndarray, Path | None, Path | None]:
        self.require_training_eligibility(request.episode_index, camera_keys=(request.camera_key,))
        if not self._has_readable_video(request.episode_index, request.camera_key):
            if self.missing_views is None:
                # Preserve the legacy fail-closed error when missing-view
                # synthesis was not explicitly enabled by the model loader.
                self.index.video_path(request.episode_index, request.camera_key)
                raise AssertionError("unreachable")
            resolution = self.missing_views.require(request.camera_key)
            image = np.zeros((resolution.height, resolution.width, 3), dtype=np.uint8)
            return image, np.ones(image.shape[:2], dtype=np.uint8), None, None
        video_path = self.index.video_path(request.episode_index, request.camera_key)
        image = self.videos.read(video_path, request.frame_index)
        if self.missing_views is not None:
            expected = self.missing_views.get(request.camera_key)
            if expected is not None and image.shape[:2] != expected.shape:
                raise ValueError(
                    f"camera {request.camera_key!r} frame resolution {image.shape[1]}x{image.shape[0]} "
                    f"does not match configured/reference resolution {expected.width}x{expected.height}"
                )
        if self.augmenter.uses_fixed_border_mask(request.camera_key):
            # Fixed front-border augmentation is an online mask type.  It does
            # not need (and must not accidentally reuse) a wrist final-mask.
            return image, np.ones(image.shape[:2], dtype=np.uint8), video_path, None
        mask, manifest = self.masks.read(
            request.episode_index,
            request.camera_key,
            request.frame_index,
        )
        return image, mask, video_path, manifest

    def read_augmented(self, request: FrameRequest) -> MaskedFrameSample:
        image, mask, video_path, manifest = self.read_raw(request)
        source_view_missing = video_path is None
        occlusion_ratio = 1.0 if source_view_missing else request.occlusion_ratio
        forced_missing_view = source_view_missing or request.forced_missing_view
        fixed_mask = (
            not source_view_missing
            and self.augmenter.uses_fixed_border_mask(request.camera_key)
        )
        augmented_bgr = self.augmenter.apply(
            image,
            mask,
            episode_index=request.episode_index,
            camera_key=request.camera_key,
            frame_index=request.frame_index,
            epoch=request.epoch,
            mask_role="keep" if fixed_mask else self.config.mask.mask_role,
            occlusion_ratio=occlusion_ratio,
            occlusion_scope_anchor=request.occlusion_scope_anchor,
            forced_missing_view=forced_missing_view,
        )
        fixed_mask_metadata = (
            None
            if source_view_missing
            else self.augmenter.fixed_mask_metadata(
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                episode_index=request.episode_index,
                camera_key=request.camera_key,
                epoch=request.epoch,
            )
        )
        photometric_metadata = (
            None
            if source_view_missing
            else self.augmenter.photometric_metadata(
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                episode_index=request.episode_index,
                camera_key=request.camera_key,
                epoch=request.epoch,
            )
        )
        view_occlusion_metadata = self.augmenter.view_occlusion_metadata(
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            episode_index=request.episode_index,
            camera_key=request.camera_key,
            epoch=request.epoch,
            frame_index=request.frame_index,
            occlusion_ratio=occlusion_ratio,
            occlusion_scope_anchor=request.occlusion_scope_anchor,
            forced_missing_view=forced_missing_view,
        )
        metadata: dict[str, Any] = {
            "dataset_root": str(self.index.dataset_root),
            "video_path": None if video_path is None else str(video_path),
            "mask_manifest_path": None if manifest is None else str(manifest),
            "mask_role": "keep" if fixed_mask else self.config.mask.mask_role,
            "mask_type": (
                "missing_view_full_black"
                if source_view_missing
                else "front_fixed_border_replace"
                if fixed_mask
                else "final_keep"
            ),
            "source_view_missing": source_view_missing,
        }
        if source_view_missing and self.missing_views is not None:
            resolution = self.missing_views.require(request.camera_key)
            metadata["missing_view"] = {
                "camera_key": request.camera_key,
                "resolution": [resolution.height, resolution.width],
                "resolution_sources": list(resolution.sources),
                "replacement": "full_black",
            }
        if fixed_mask_metadata is not None:
            metadata["fixed_mask"] = fixed_mask_metadata
        if photometric_metadata is not None:
            metadata["photometric_augmentation"] = photometric_metadata
        if view_occlusion_metadata is not None:
            metadata["view_occlusion"] = view_occlusion_metadata
        return MaskedFrameSample(
            request=request,
            image=convert_layout(image, self.config.video.output_layout),
            mask=mask,
            augmented=convert_layout(augmented_bgr, self.config.video.output_layout),
            video_path=video_path,
            mask_manifest_path=manifest,
            metadata=metadata,
        )

    def read_augmented_frame(
        self,
        *,
        episode_index: int,
        camera_key: str,
        frame_index: int,
        epoch: int = 0,
        occlusion_ratio: float = 0.0,
        occlusion_scope_anchor: int | None = None,
        forced_missing_view: bool = False,
    ) -> np.ndarray:
        request = FrameRequest(
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            frame_index=int(frame_index),
            epoch=int(epoch),
            occlusion_ratio=float(occlusion_ratio),
            occlusion_scope_anchor=(
                None if occlusion_scope_anchor is None else int(occlusion_scope_anchor)
            ),
            forced_missing_view=bool(forced_missing_view),
        )
        return self.read_augmented(request).augmented

    def read_augmented_clip(
        self,
        *,
        episode_index: int,
        camera_key: str,
        start_frame: int,
        length: int,
        stride: int = 1,
        epoch: int = 0,
        occlusion_ratio: float = 0.0,
        forced_missing_view: bool = False,
    ) -> np.ndarray:
        self.require_training_eligibility(episode_index, camera_keys=(camera_key,))
        source_view_missing = not self._has_readable_video(episode_index, camera_key)
        if source_view_missing:
            if self.missing_views is None:
                self.index.video_path(episode_index, camera_key)
                raise AssertionError("unreachable")
            resolution = self.missing_views.require(camera_key)
            images = np.zeros(
                (int(length), resolution.height, resolution.width, 3),
                dtype=np.uint8,
            )
        else:
            video_path = self.index.video_path(episode_index, camera_key)
            images = self.videos.read_clip(video_path, start_frame, length, stride)
            if self.missing_views is not None:
                expected = self.missing_views.get(camera_key)
                if expected is not None and images.shape[1:3] != expected.shape:
                    raise ValueError(
                        f"camera {camera_key!r} clip resolution {images.shape[2]}x{images.shape[1]} "
                        f"does not match configured/reference resolution {expected.width}x{expected.height}"
                    )
        fixed_mask = self.augmenter.uses_fixed_border_mask(camera_key)
        if fixed_mask or source_view_missing:
            masks = np.ones(images.shape[:3], dtype=np.uint8)
        else:
            masks, _manifest = self.masks.read_clip(episode_index, camera_key, start_frame, length, stride)
        indices = [int(start_frame) + offset * int(stride) for offset in range(int(length))]
        augmented = self.augmenter.apply_batch(
            images,
            masks,
            episode_index=int(episode_index),
            camera_key=str(camera_key),
            frame_indices=indices,
            epoch=int(epoch),
            mask_role="keep" if fixed_mask else self.config.mask.mask_role,
            occlusion_ratio=1.0 if source_view_missing else float(occlusion_ratio),
            occlusion_scope_anchor=int(start_frame),
            forced_missing_view=source_view_missing or bool(forced_missing_view),
        )
        if self.config.video.output_layout == "bgr_hwc_uint8":
            return augmented
        return np.stack([convert_layout(frame, self.config.video.output_layout) for frame in augmented], axis=0)

    def transform_sample_dict(
        self,
        sample: dict[str, Any],
        *,
        episode_key: str = "episode_index",
        frame_key: str = "frame_index",
        epoch: int = 0,
        camera_keys: tuple[str, ...] = ("left_wrist", "right_wrist"),
        output_prefix: str = "observation.images",
        occlusion_ratios: Mapping[str, float] | None = None,
        forced_missing_views: Mapping[str, bool] | None = None,
    ) -> dict[str, Any]:
        """Add augmented images to a generic sample dict.

        This adapter keeps model-specific datasets thin: as long as a sample
        exposes episode_index and frame_index, augmented camera arrays are added
        under ``<output_prefix>.<camera_key>.augmented``.
        """

        episode_index = int(sample[episode_key])
        frame_index = int(sample[frame_key])
        self.require_training_eligibility(episode_index, camera_keys=camera_keys)
        out = dict(sample)
        ratios = {} if occlusion_ratios is None else dict(occlusion_ratios)
        missing_views = {} if forced_missing_views is None else dict(forced_missing_views)
        for camera_key in camera_keys:
            key = f"{output_prefix}.{camera_key}.augmented"
            out[key] = self.read_augmented_frame(
                episode_index=episode_index,
                camera_key=camera_key,
                frame_index=frame_index,
                epoch=epoch,
                occlusion_ratio=float(ratios.get(camera_key, 0.0)),
                forced_missing_view=bool(missing_views.get(camera_key, False)),
            )
        return out

    def training_eligibility(
        self,
        episode_index: int,
        *,
        camera_keys: tuple[str, ...] | list[str] = (),
    ) -> TrainingEligibility:
        base = self.index.training_eligibility(episode_index, camera_keys=())
        if not base.eligible:
            return TrainingEligibility(
                episode_index=base.episode_index,
                camera_keys=tuple(str(key) for key in camera_keys),
                eligible=False,
                reasons=base.reasons,
            )
        reasons: list[str] = []
        requested = tuple(dict.fromkeys(str(key) for key in camera_keys))
        for camera_key in requested:
            if self.index.has_video(episode_index, camera_key):
                eligibility = self.index.training_eligibility(
                    episode_index,
                    camera_keys=(camera_key,),
                )
                reasons.extend(eligibility.reasons)
                if eligibility.reasons or self._has_readable_video(episode_index, camera_key):
                    continue
            if self.missing_views is None or self.missing_views.get(camera_key) is None:
                reasons.append(
                    f"video view {camera_key!r} is missing and has no explicit/reference resolution"
                )
        return TrainingEligibility(
            episode_index=int(episode_index),
            camera_keys=requested,
            eligible=not reasons,
            reasons=tuple(reasons),
        )

    def warn_ineligible_training(self, eligibility: TrainingEligibility) -> None:
        """Emit one visible warning per rejected episode/view combination."""

        key = (eligibility.episode_index, eligibility.camera_keys, eligibility.reasons)
        if key in self._warned_ineligible_training:
            return
        self._warned_ineligible_training.add(key)
        print(
            f"\033[31mWARNING: skipped invalid training data: {eligibility.message()}\033[0m",
            file=sys.stderr,
            flush=True,
        )

    def require_training_eligibility(
        self,
        episode_index: int,
        *,
        camera_keys: tuple[str, ...] | list[str] = (),
    ) -> TrainingEligibility:
        eligibility = self.training_eligibility(episode_index, camera_keys=camera_keys)
        if not eligibility.eligible:
            self.warn_ineligible_training(eligibility)
            raise InvalidTrainingDataError(eligibility)
        return eligibility
