from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from dataset_interface.pipeline import MaskedFramePipeline


class TorchMaskedAugmentDataset:
    """Thin wrapper for PyTorch-style datasets.

    The wrapped dataset must return a mutable mapping or mapping-like sample
    with ``episode_index`` and ``frame_index`` fields by default. Torch is not
    imported here; this class follows the Dataset protocol without forcing a
    hard dependency on a specific training stack.
    """

    def __init__(
        self,
        base_dataset: Any,
        pipeline_factory: Callable[[], MaskedFramePipeline],
        *,
        camera_keys: Sequence[str] = ("left_wrist", "right_wrist"),
        episode_key: str = "episode_index",
        frame_key: str = "frame_index",
        epoch_getter: Callable[[], int] | None = None,
        output_prefix: str = "observation.images",
        occlusion_ratio_getter: Callable[[dict[str, Any], str, int], float] | None = None,
        forced_missing_view_getter: Callable[[dict[str, Any], str], bool] | None = None,
    ):
        self.base_dataset = base_dataset
        self.pipeline_factory = pipeline_factory
        self.camera_keys = tuple(str(key) for key in camera_keys)
        self.episode_key = episode_key
        self.frame_key = frame_key
        self.epoch_getter = epoch_getter
        self.output_prefix = output_prefix
        self.occlusion_ratio_getter = occlusion_ratio_getter
        self.forced_missing_view_getter = forced_missing_view_getter
        self._pipeline: MaskedFramePipeline | None = None

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _get_pipeline(self) -> MaskedFramePipeline:
        if self._pipeline is None:
            self._pipeline = self.pipeline_factory()
        return self._pipeline

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None

    def __getitem__(self, index: int) -> dict[str, Any]:
        item_count = len(self.base_dataset)
        if item_count < 1:
            raise IndexError("cannot read from an empty base_dataset")
        pipeline = self._get_pipeline()
        start_index = int(index) % item_count
        for offset in range(item_count):
            source_index = (start_index + offset) % item_count
            sample = dict(self.base_dataset[source_index])
            episode_index = int(sample[self.episode_key])
            eligibility = pipeline.training_eligibility(
                episode_index,
                camera_keys=self.camera_keys,
            )
            if not eligibility.eligible:
                pipeline.warn_ineligible_training(eligibility)
                continue
            epoch = 0 if self.epoch_getter is None else int(self.epoch_getter())
            occlusion_ratios = {
                camera_key: (
                    0.0
                    if self.occlusion_ratio_getter is None
                    else float(self.occlusion_ratio_getter(sample, camera_key, epoch))
                )
                for camera_key in self.camera_keys
            }
            forced_missing_views = {
                camera_key: (
                    False
                    if self.forced_missing_view_getter is None
                    else bool(self.forced_missing_view_getter(sample, camera_key))
                )
                for camera_key in self.camera_keys
            }
            return pipeline.transform_sample_dict(
                sample,
                episode_key=self.episode_key,
                frame_key=self.frame_key,
                epoch=epoch,
                camera_keys=self.camera_keys,
                output_prefix=self.output_prefix,
                occlusion_ratios=occlusion_ratios,
                forced_missing_views=forced_missing_views,
            )
        raise RuntimeError("all base_dataset samples are invalid for the configured training camera_keys")
