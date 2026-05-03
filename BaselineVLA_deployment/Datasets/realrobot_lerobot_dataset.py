from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import random
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from Datasets.base import CanonicalSample, DatasetManifest, SampleAddress
from Datasets.config_utils import get_positive_limit
from Datasets.desktasktac_dataset import _merge_address_groups, _product, _sequence_offsets, _stable_seed
from Datasets.tactile_uv import InspireTactileUVConverter
from Datasets.task_utils import build_task_name_to_id
from Datasets.video_utils import apply_video_augmentations, build_rgb_mean_video_tokens, load_video_frames_ffmpeg


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("RealRobot LeRobot v2.1 loading requires pyarrow in the active environment.") from error
    return pq


class _EpisodeRowCache:
    def __init__(self, max_size: int, required_columns: List[str]) -> None:
        self.max_size = max(1, int(max_size))
        self.required_columns = list(required_columns)
        self.cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()

    def get_rows(self, parquet_path: str) -> List[Dict[str, Any]]:
        if parquet_path in self.cache:
            self.cache.move_to_end(parquet_path)
            return self.cache[parquet_path]

        pq = _require_pyarrow()
        rows = pq.read_table(parquet_path, columns=self.required_columns).to_pylist()
        self.cache[parquet_path] = rows
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        return rows


class RealRobotLeRobotSequenceDataset(Dataset):
    dataset_type_name = "RealRobotLeRobot_Sequence"

    def __init__(self, manifests: List[DatasetManifest], split: str, config: Any) -> None:
        self.manifests = manifests
        self.split = str(split)
        self.config = config
        self.sequence_config = config.sequence
        self.video_config = config.video
        self.manifest_by_id = {manifest.dataset_id: manifest for manifest in self.manifests}
        self.task_name_to_id = build_task_name_to_id(self.manifests)
        self.tactile_normalization_stats: Dict[str, Any] | None = None
        self.tactile_converter = (
            self._build_tactile_converter()
            if self._tactile_observation_enabled() and self._tactile_representation() == "uv"
            else None
        )
        self.required_columns = self._build_required_columns()
        self.episode_cache = _EpisodeRowCache(
            max_size=int(config.sampling.max_episode_cache_size),
            required_columns=self.required_columns,
        )
        self._validate_configuration()
        self.addresses = self._build_addresses()
        self.sample_count_by_dataset = self._count_addresses_by_dataset(self.addresses)
        if not self.addresses:
            raise RuntimeError(f"No samples discovered for split={split}. Check filters or sampling windows.")
        if self._tactile_observation_enabled() and self._tactile_normalize_enabled():
            self.tactile_normalization_stats = self._resolve_tactile_normalization_stats()
            self._write_tactile_normalization_stats_to_config(self.tactile_normalization_stats)

    def __len__(self) -> int:
        return len(self.addresses)

    def __getitem__(self, index: int) -> CanonicalSample:
        address = self.addresses[int(index)]
        manifest = self.manifest_by_id[address.dataset_id]
        episode_parquet_path = self._episode_parquet_path(manifest, address.episode_index)
        rows = self.episode_cache.get_rows(episode_parquet_path)
        current_row = rows[address.frame_index]

        state_indices = self._history_indices(address.frame_index)
        state_sequence = torch.stack(
            [self._concat_fields(rows[row_index], self.sequence_config.state_fields) for row_index in state_indices],
            dim=0,
        )
        state = state_sequence.reshape(-1)

        action_indices = self._future_action_indices(address.frame_index)
        action_sequence = torch.stack(
            [self._concat_fields(rows[row_index], self.sequence_config.action_fields) for row_index in action_indices],
            dim=0,
        )
        action = self._format_temporal_target(action_sequence)

        tactile_observation = None
        tactile_observation_mask = None
        if self._tactile_observation_enabled():
            tactile_indices = self._tactile_history_indices(address.frame_index)
            tactile_observation = torch.stack(
                [self._build_tactile_observation_frame(rows[row_index]) for row_index in tactile_indices],
                dim=0,
            )
            tactile_observation_mask = torch.ones(int(tactile_observation.shape[0]), dtype=torch.float32)

        video_path, video_frame_index = self._resolve_video_location(manifest, current_row, address.episode_index)
        video_frames = self._load_video_frames(video_path=video_path, current_video_frame_index=video_frame_index)
        video_frames = self._augment_video_frames(video_frames)
        vision_tokens, vision_mask = build_rgb_mean_video_tokens(video_frames)

        task_text = str(current_row.get("task") or manifest.episode_task_lookup.get(address.episode_index, "unknown_task"))
        task_id = int(self.task_name_to_id.get(task_text, self.task_name_to_id["unknown_task"]))

        observations: Dict[str, Any] = {
            "state": state,
            "state_sequence": state_sequence,
            "task_text": task_text,
            "task_id": torch.tensor(task_id, dtype=torch.long),
            "vision_tokens": vision_tokens,
            "vision_mask": vision_mask,
        }
        if bool(self.video_config.emit_frames):
            observations["video_frames"] = video_frames
            observations["video_frame_mask"] = vision_mask.clone()
        if tactile_observation is not None:
            observations["tactile"] = tactile_observation
            observations["tactile_mask"] = tactile_observation_mask

        return CanonicalSample(
            observations=observations,
            intermediates={},
            targets={"action": action},
            availability={
                "state": torch.tensor(1.0, dtype=torch.float32),
                "action": torch.tensor(1.0, dtype=torch.float32),
                "vision": torch.tensor(float(int(video_frames.shape[0]) > 0), dtype=torch.float32),
                "task": torch.tensor(1.0, dtype=torch.float32),
                "tactile": torch.tensor(float(tactile_observation is not None), dtype=torch.float32),
            },
            metadata={
                "dataset_id": manifest.dataset_id,
                "dataset_root": manifest.root,
                "raw_data_type": self.dataset_type_name,
                "source_format": manifest.source_format,
                "source_data_type": manifest.raw_data_type,
                "robot_type": manifest.robot_type,
                "split": address.split,
                "episode_index": address.episode_index,
                "frame_index": address.frame_index,
                "row_frame_index": int(current_row.get("frame_index", address.frame_index)),
                "episode_length": address.episode_length,
                "episode_parquet_path": episode_parquet_path,
                "state_dim": int(state.shape[0]),
                "state_step_dim": int(state_sequence.shape[-1]),
                "action_dim": int(action_sequence.shape[-1]),
                "action_chunk": int(action_sequence.shape[0]),
                "task_text": task_text,
                "task_id": task_id,
                "video_key": str(self.video_config.key),
                "video_path": "" if video_path is None else str(video_path),
                "video_frame_index": int(video_frame_index),
                "tactile_observation_enabled": self._tactile_observation_enabled(),
                "tactile_observation_field": self._tactile_observation_field() if self._tactile_observation_enabled() else None,
                "tactile_observation_raw_dim": self._tactile_raw_dim(manifest) if self._tactile_observation_enabled() else None,
                "tactile_observation_dim": self._tactile_output_dim(manifest) if self._tactile_observation_enabled() else None,
                "tactile_observation_representation": self._tactile_representation()
                if self._tactile_observation_enabled()
                else None,
                "tactile_observation_normalized": self._tactile_normalize_enabled()
                if self._tactile_observation_enabled()
                else None,
                "tactile_observation_normalization": self._tactile_normalization_summary()
                if self._tactile_observation_enabled()
                else None,
                "tactile_observation_actual_uv_dim": self.tactile_converter.actual_valid_dim
                if self._tactile_observation_enabled() and self.tactile_converter is not None
                else None,
                "tactile_observation_union_uv_dim": self.tactile_converter.union_valid_dim
                if self._tactile_observation_enabled() and self.tactile_converter is not None
                else None,
                "tactile_observation_history": self._tactile_observation_history(),
                "tactile_observation_stride": self._tactile_observation_stride(),
                "tactile_observation_shape": [] if tactile_observation is None else [int(value) for value in tactile_observation.shape],
                "tactile_uv_conversion_root": self._tactile_uv_conversion_root(),
                "tactile_uv_hand": self._tactile_hand(),
            },
        )

    @property
    def summary(self) -> Dict[str, Any]:
        first_manifest = self.manifests[0]
        state_step_dim = sum(self._feature_dim(first_manifest, field_name) for field_name in self.sequence_config.state_fields)
        action_step_dim = sum(self._feature_dim(first_manifest, field_name) for field_name in self.sequence_config.action_fields)
        return {
            "dataset_type": self.dataset_type_name,
            "manifest_loader": str(self.config.manifest_loader),
            "split": self.split,
            "sample_count": len(self.addresses),
            "dataset_count": len(self.manifests),
            "sample_count_by_dataset": dict(self.sample_count_by_dataset),
            "state_dim": state_step_dim * int(self.sequence_config.state_history),
            "state_step_dim": state_step_dim,
            "state_history": int(self.sequence_config.state_history),
            "state_stride": int(self.sequence_config.state_stride),
            "action_dim": action_step_dim,
            "action_chunk": self._action_chunk(),
            "action_stride": int(self.sequence_config.action_stride),
            "video_key": str(self.video_config.key),
            "video_token_slots": int(self.video_config.clip_length) if bool(self.video_config.enabled) else 0,
            "video_clip_length": int(self.video_config.clip_length) if bool(self.video_config.enabled) else 0,
            "video_augmentation_enabled": self._video_augmentation_enabled(),
            "video_augmentation": self._video_augmentation_summary(),
            "tactile_observation_enabled": self._tactile_observation_enabled(),
            "tactile_observation_source_dim": None if not self._tactile_observation_enabled() else self._tactile_raw_dim(first_manifest),
            "tactile_observation_representation": None
            if not self._tactile_observation_enabled()
            else self._tactile_representation(),
            "tactile_observation_normalized": None
            if not self._tactile_observation_enabled()
            else self._tactile_normalize_enabled(),
            "tactile_observation_normalization": None
            if not self._tactile_observation_enabled()
            else self._tactile_normalization_summary(),
            "tactile_observation_actual_uv_dim": None
            if not self._tactile_observation_enabled() or self.tactile_converter is None
            else self.tactile_converter.actual_valid_dim,
            "tactile_observation_union_uv_dim": None
            if not self._tactile_observation_enabled() or self.tactile_converter is None
            else self.tactile_converter.union_valid_dim,
            "tactile_observation_dim": None if not self._tactile_observation_enabled() else self._tactile_output_dim(first_manifest),
            "tactile_observation_history": self._tactile_observation_history(),
            "tactile_observation_stride": self._tactile_observation_stride(),
            "tactile_observation_shape": []
            if not self._tactile_observation_enabled()
            else [self._tactile_observation_history(), self._tactile_output_dim(first_manifest)],
            "tactile_observation_fields": {}
            if not self._tactile_observation_enabled()
            else {
                "field": self._tactile_observation_field(),
                "raw_dim": self._tactile_raw_dim(first_manifest),
                "representation": self._tactile_representation(),
                "hand_uv_dim": None if self.tactile_converter is None else self.tactile_converter.actual_valid_dim,
                "union_uv_dim": None if self.tactile_converter is None else self.tactile_converter.union_valid_dim,
                "converted_dim": self._tactile_output_dim(first_manifest),
                "uv_conversion_root": self._tactile_uv_conversion_root(),
                "hand": self._tactile_hand(),
                "correct_order": self._tactile_correct_order(),
                "normalize": self._tactile_normalize_enabled(),
                "normalization_mode": self._tactile_normalization_mode(),
                "normalization_min": self._tactile_normalization_min(),
                "normalization_max": self._tactile_normalization_max(),
                "normalization_range": self._tactile_normalization_range(),
                "normalization_eps": self._tactile_normalization_eps(),
                "normalization_clip": self._tactile_normalization_clip_enabled(),
                "normalization_source": self._tactile_normalization_source(),
                "normalization_offset": self._tactile_normalization_min(),
                "normalization_scale": self._tactile_normalization_range(),
            },
            "task_count": len(self.task_name_to_id),
            "task_names": [task_name for task_name, _ in sorted(self.task_name_to_id.items(), key=lambda item: item[1])],
            "dataset_ids": [manifest.dataset_id for manifest in self.manifests],
        }

    def _build_required_columns(self) -> List[str]:
        columns = {"episode_index", "frame_index"}
        columns.update(str(field_name) for field_name in self.sequence_config.state_fields)
        columns.update(str(field_name) for field_name in self.sequence_config.action_fields)
        if self._tactile_observation_enabled():
            columns.add(self._tactile_observation_field())
        if bool(self.video_config.enabled):
            video_key = str(self.video_config.key)
            columns.add(f"{video_key}.path")
            columns.add(f"{video_key}.video_frame_index")
        return sorted(columns)

    def _build_addresses(self) -> List[SampleAddress]:
        address_groups: List[List[SampleAddress]] = []
        max_frames = get_positive_limit(self.config, "max_frames_per_episode")
        max_total = get_positive_limit(self.config, "max_total_samples")

        for manifest in self.manifests:
            manifest_addresses: List[SampleAddress] = []
            for episode_index in self._select_episode_indices(manifest):
                episode_length = int(manifest.episode_lengths.get(int(episode_index), 0))
                for frame_index in self._select_valid_frame_indices(episode_length, max_frames):
                    manifest_addresses.append(
                        SampleAddress(
                            dataset_id=manifest.dataset_id,
                            split=self.split,
                            episode_index=int(episode_index),
                            frame_index=int(frame_index),
                            episode_length=int(episode_length),
                        )
                    )
            if manifest_addresses:
                address_groups.append(manifest_addresses)

        return _merge_address_groups(
            groups=address_groups,
            max_total_samples=None if max_total is None else int(max_total),
            strategy=str(getattr(self.config.sampling, "mix_strategy", "round_robin")),
        )

    def _select_episode_indices(self, manifest: DatasetManifest) -> List[int]:
        episode_indices = list(manifest.split_to_episode_indices.get(self.split, []))
        max_episodes = get_positive_limit(self.config, "max_episodes_per_dataset")
        if max_episodes is None or len(episode_indices) <= int(max_episodes):
            return episode_indices

        rng = random.Random(
            _stable_seed(
                int(getattr(self.config, "split_seed", 0)),
                manifest.dataset_id,
                self.split,
                "episode_subset",
            )
        )
        sampled = list(episode_indices)
        rng.shuffle(sampled)
        return sampled[: int(max_episodes)]

    def _select_valid_frame_indices(self, episode_length: int, max_frames: int | None) -> List[int]:
        history_requirement = max(
            (int(self.sequence_config.state_history) - 1) * int(self.sequence_config.state_stride),
            (int(self.video_config.clip_length) - 1) * int(self.video_config.clip_stride) if bool(self.video_config.enabled) else 0,
            (self._tactile_observation_history() - 1) * self._tactile_observation_stride()
            if self._tactile_observation_enabled()
            else 0,
        )
        future_requirement = (self._action_chunk() - 1) * int(self.sequence_config.action_stride)
        start_index = int(history_requirement)
        end_index = int(episode_length) - int(future_requirement)
        if end_index <= start_index:
            return []

        frame_stride = max(1, int(self.config.sampling.frame_stride))
        candidate_indices = list(range(start_index, end_index, frame_stride))
        if max_frames is None or len(candidate_indices) <= int(max_frames):
            return candidate_indices
        if int(max_frames) <= 1:
            return [candidate_indices[0]]
        step = (len(candidate_indices) - 1) / float(max_frames - 1)
        selected = [candidate_indices[round(position * step)] for position in range(max_frames)]
        return sorted(set(selected))

    def _count_addresses_by_dataset(self, addresses: List[SampleAddress]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for address in addresses:
            counts[address.dataset_id] = counts.get(address.dataset_id, 0) + 1
        return counts

    def _history_indices(self, current_index: int) -> List[int]:
        return [
            current_index - offset
            for offset in _sequence_offsets(
                length=int(self.sequence_config.state_history),
                stride=int(self.sequence_config.state_stride),
                reverse=True,
            )
        ]

    def _tactile_history_indices(self, current_index: int) -> List[int]:
        return [
            current_index - offset
            for offset in _sequence_offsets(
                length=self._tactile_observation_history(),
                stride=self._tactile_observation_stride(),
                reverse=True,
            )
        ]

    def _future_action_indices(self, current_index: int) -> List[int]:
        return [
            current_index + offset
            for offset in _sequence_offsets(
                length=self._action_chunk(),
                stride=int(self.sequence_config.action_stride),
                reverse=False,
            )
        ]

    def _concat_fields(self, row: Dict[str, Any], field_names: List[str]) -> torch.Tensor:
        chunks = [self._tensor_from_value(row[str(field_name)]) for field_name in field_names]
        return torch.cat(chunks, dim=0).to(dtype=torch.float32)

    def _tensor_from_value(self, value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value.to(dtype=torch.float32).reshape(-1)
            return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if isinstance(value, list):
            tensor = torch.tensor(value, dtype=torch.float32).reshape(-1)
            return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        tensor = torch.tensor([float(value)], dtype=torch.float32)
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_tactile_observation_frame(self, row: Dict[str, Any]) -> torch.Tensor:
        raw_values = self._tensor_from_value(row[self._tactile_observation_field()]).numpy()
        converted = self._convert_tactile_values(raw_values)
        converted = self._normalize_tactile_values(converted)
        return torch.from_numpy(converted.reshape(-1)).to(dtype=torch.float32)

    def _convert_tactile_values(self, raw_values: Any) -> np.ndarray:
        raw_array = np.asarray(raw_values, dtype=np.float32).reshape(-1)
        if self._tactile_representation() == "raw":
            return raw_array.astype("float32", copy=False)
        if self.tactile_converter is None:
            raise RuntimeError("Tactile UV observation requested, but the tactile converter is disabled.")
        return self.tactile_converter.convert(raw_array).astype("float32", copy=False)

    def _load_video_frames(self, video_path: Path | None, current_video_frame_index: int) -> torch.Tensor:
        if not bool(self.video_config.enabled):
            return torch.zeros(0, 3, int(self.video_config.resize_height), int(self.video_config.resize_width), dtype=torch.float32)
        if video_path is None:
            raise FileNotFoundError("RealRobot video is enabled, but no episode video path was resolved.")

        frame_indices = [
            int(current_video_frame_index) - offset
            for offset in _sequence_offsets(
                length=int(self.video_config.clip_length),
                stride=int(self.video_config.clip_stride),
                reverse=True,
            )
        ]
        return load_video_frames_ffmpeg(
            video_path=video_path,
            frame_indices=frame_indices,
            resize_height=int(self.video_config.resize_height),
            resize_width=int(self.video_config.resize_width),
        )

    def _augment_video_frames(self, video_frames: torch.Tensor) -> torch.Tensor:
        if not self._video_augmentation_enabled():
            return video_frames
        return apply_video_augmentations(video_frames, self._video_augmentation_config())

    def _resolve_video_location(
        self,
        manifest: DatasetManifest,
        current_row: Dict[str, Any],
        episode_index: int,
    ) -> tuple[Path | None, int]:
        video_key = str(self.video_config.key)
        current_video_frame_index = int(current_row.get(f"{video_key}.video_frame_index", current_row.get("frame_index", 0)))
        if not bool(self.video_config.enabled):
            return None, current_video_frame_index

        video_value = current_row.get(f"{video_key}.path")
        if video_value is None:
            video_value = manifest.metadata.get("episode_video_lookup", {}).get(int(episode_index))
        if video_value is None:
            raise KeyError(f"Missing RealRobot video path for episode_index={episode_index}, video_key={video_key}.")
        video_path = Path(str(video_value))
        if not video_path.is_absolute():
            video_path = Path(manifest.root) / video_path
        return video_path.resolve(), current_video_frame_index

    def _episode_parquet_path(self, manifest: DatasetManifest, episode_index: int) -> str:
        episode_lookup = manifest.metadata["episode_parquet_lookup"]
        episode_value = episode_lookup[int(episode_index)]
        episode_path = Path(str(episode_value))
        if not episode_path.is_absolute():
            episode_path = Path(manifest.root) / episode_path
        return str(episode_path.resolve())

    def _feature_dim(self, manifest: DatasetManifest, field_name: str) -> int:
        spec = manifest.feature_specs[str(field_name)]
        shape = list(spec.get("shape", []))
        return _product(shape) if shape else 1

    def _action_chunk(self) -> int:
        if hasattr(self.sequence_config, "action_chunk"):
            return int(self.sequence_config.action_chunk)
        return int(self.sequence_config.action_horizon)

    def _video_augmentation_config(self) -> Any:
        return getattr(self.video_config, "augmentation", None)

    def _video_augmentation_config_value(self, key: str, default: Any) -> Any:
        augmentation_config = self._video_augmentation_config()
        if augmentation_config is None:
            return default
        if isinstance(augmentation_config, dict):
            return augmentation_config.get(key, default)
        return getattr(augmentation_config, key, default)

    def _video_augmentation_config_bool(self, key: str, default: bool) -> bool:
        value = self._video_augmentation_config_value(key, default)
        if value is None:
            return bool(default)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off", "none", "null"}:
                return False
        return bool(value)

    def _video_augmentation_enabled(self) -> bool:
        if not bool(self.video_config.enabled):
            return False
        if not self._video_augmentation_config_bool("enabled", False):
            return False
        train_only = self._video_augmentation_config_bool("train_only", True)
        return (not train_only) or self.split.lower() == "train"

    def _video_augmentation_summary(self) -> Dict[str, Any]:
        augmentation_config = self._video_augmentation_config()
        if augmentation_config is None:
            return {"enabled": False}
        keys = (
            "enabled",
            "train_only",
            "color_jitter_prob",
            "brightness",
            "contrast",
            "saturation",
            "brightness_boost_prob",
            "brightness_boost_max_delta",
            "random_resized_crop_prob",
            "crop_scale_min",
            "crop_scale_max",
            "crop_ratio_min",
            "crop_ratio_max",
            "gaussian_noise_prob",
            "gaussian_noise_std",
            "gaussian_blur_prob",
            "gaussian_blur_kernel",
            "gaussian_blur_sigma",
            "random_erasing_prob",
            "random_erasing_scale_min",
            "random_erasing_scale_max",
            "random_erasing_ratio_min",
            "random_erasing_ratio_max",
            "random_erasing_center_protect_fraction",
            "random_erasing_value",
        )
        summary = {key: self._video_augmentation_config_value(key, None) for key in keys}
        summary["active_for_split"] = self._video_augmentation_enabled()
        summary["split"] = self.split
        summary["flip_enabled"] = False
        summary["rotation_enabled"] = False
        return summary

    def _tactile_observation_config(self) -> Any:
        return getattr(self.sequence_config, "tactile", None)

    def _tactile_observation_enabled(self) -> bool:
        tactile_config = self._tactile_observation_config()
        if tactile_config is None:
            return False
        return bool(getattr(tactile_config, "enabled", False))

    def _tactile_observation_field(self) -> str:
        tactile_config = self._tactile_observation_config()
        explicit_field = None if tactile_config is None else getattr(tactile_config, "field", None)
        if explicit_field not in (None, "null"):
            return str(explicit_field)
        return f"observation.hand.{self._tactile_hand()}.tactile"

    def _tactile_observation_history(self) -> int:
        tactile_config = self._tactile_observation_config()
        if tactile_config is None:
            return 1
        history_value = getattr(tactile_config, "history", None)
        if history_value is None:
            history_value = getattr(tactile_config, "history_size", 1)
        return max(1, int(history_value))

    def _tactile_observation_stride(self) -> int:
        tactile_config = self._tactile_observation_config()
        if tactile_config is None:
            return int(self.sequence_config.state_stride)
        stride_value = getattr(tactile_config, "stride", None)
        if stride_value is None:
            stride_value = int(self.sequence_config.state_stride)
        return max(1, int(stride_value))

    def _tactile_hand(self) -> str:
        tactile_config = self._tactile_observation_config()
        return str(getattr(tactile_config, "hand", "right")).lower() if tactile_config is not None else "right"

    def _tactile_representation(self) -> str:
        tactile_config = self._tactile_observation_config()
        value = None if tactile_config is None else getattr(tactile_config, "representation", None)
        if value in (None, "null"):
            value = "uv"
        normalized = str(value).strip().lower()
        aliases = {
            "uv": "uv",
            "uv351": "uv",
            "converted": "uv",
            "compact": "uv",
            "raw": "raw",
            "raw1062": "raw",
            "1062": "raw",
        }
        if normalized not in aliases:
            raise ValueError(
                f"Unsupported tactile representation={value!r}; expected one of {sorted(aliases)}."
            )
        return aliases[normalized]

    def _tactile_uv_conversion_root(self) -> str:
        tactile_config = self._tactile_observation_config()
        value = None if tactile_config is None else getattr(tactile_config, "uv_conversion_root", None)
        return str(value) if value not in (None, "null") else "PolicyDeploy/UV_conversion"

    def _tactile_expected_dim(self) -> int:
        tactile_config = self._tactile_observation_config()
        value = None if tactile_config is None else getattr(tactile_config, "converted_dim", None)
        return int(value) if value not in (None, "null") else 351

    def _tactile_correct_order(self) -> bool:
        tactile_config = self._tactile_observation_config()
        value = None if tactile_config is None else getattr(tactile_config, "correct_order", None)
        return True if value is None else bool(value)

    def _tactile_normalization_config(self) -> Any:
        tactile_config = self._tactile_observation_config()
        return None if tactile_config is None else getattr(tactile_config, "normalization", None)

    def _tactile_normalize_enabled(self) -> bool:
        tactile_config = self._tactile_observation_config()
        value = None if tactile_config is None else getattr(tactile_config, "normalize", None)
        normalization_config = self._tactile_normalization_config()
        nested_value = None if normalization_config is None else getattr(normalization_config, "enabled", None)
        return bool(value) or bool(nested_value)

    def _config_value_is_null(self, value: Any) -> bool:
        return value is None or str(value).strip().lower() in {"", "null", "none"}

    def _tactile_normalization_config_float(self, *names: str) -> float | None:
        normalization_config = self._tactile_normalization_config()
        if normalization_config is None:
            return None
        for name in names:
            value = getattr(normalization_config, str(name), None)
            if not self._config_value_is_null(value):
                return float(value)
        return None

    def _tactile_normalization_mode(self) -> str:
        normalization_config = self._tactile_normalization_config()
        value = None if normalization_config is None else getattr(normalization_config, "mode", None)
        if self._config_value_is_null(value):
            return "minmax"
        normalized = str(value).strip().lower()
        aliases = {
            "minmax": "minmax",
            "min_max": "minmax",
            "dataset_minmax": "minmax",
            "offset_scale": "offset_scale",
            "scale_offset": "offset_scale",
            "legacy": "offset_scale",
        }
        if normalized not in aliases:
            raise ValueError(
                f"Unsupported tactile normalization mode={value!r}; expected one of {sorted(aliases)}."
            )
        return aliases[normalized]

    def _tactile_normalization_eps(self) -> float:
        value = self._tactile_normalization_config_float("eps", "epsilon")
        return 1e-6 if value is None else float(value)

    def _tactile_normalization_clip_enabled(self) -> bool:
        normalization_config = self._tactile_normalization_config()
        value = None if normalization_config is None else getattr(normalization_config, "clip", None)
        return False if value is None else bool(value)

    def _tactile_normalization_min(self) -> float | None:
        if self.tactile_normalization_stats is not None:
            value = self.tactile_normalization_stats.get("min")
            return None if value is None else float(value)
        return self._tactile_normalization_config_float("min", "minimum")

    def _tactile_normalization_max(self) -> float | None:
        if self.tactile_normalization_stats is not None:
            value = self.tactile_normalization_stats.get("max")
            return None if value is None else float(value)
        return self._tactile_normalization_config_float("max", "maximum")

    def _tactile_normalization_range(self) -> float | None:
        if self.tactile_normalization_stats is not None:
            value = self.tactile_normalization_stats.get("range")
            return None if value is None else float(value)
        min_value = self._tactile_normalization_min()
        max_value = self._tactile_normalization_max()
        if min_value is None or max_value is None:
            return None
        return float(max_value - min_value)

    def _tactile_normalization_source(self) -> str | None:
        if self.tactile_normalization_stats is not None:
            return str(self.tactile_normalization_stats.get("source", "unknown"))
        return None if self._tactile_normalize_enabled() else "disabled"

    def _normalize_tactile_values(self, values: Any) -> Any:
        array = np.asarray(values, dtype=np.float32)
        if not self._tactile_normalize_enabled():
            return array
        stats = self.tactile_normalization_stats
        if stats is None:
            stats = self._resolve_tactile_normalization_stats()
            self.tactile_normalization_stats = stats

        mode = str(stats.get("mode", self._tactile_normalization_mode()))
        if mode == "minmax":
            normalized = (array - float(stats["min"])) / float(stats["range"])
            if bool(stats.get("clip", False)):
                normalized = np.clip(normalized, 0.0, 1.0)
            return normalized.astype("float32", copy=False)
        if mode == "offset_scale":
            normalized = (array - float(stats["offset"])) / float(stats["scale"])
            return normalized.astype("float32", copy=False)
        raise ValueError(f"Unsupported tactile normalization mode={mode!r}.")

    def _tactile_normalization_summary(self) -> Dict[str, Any]:
        enabled = self._tactile_normalize_enabled()
        return {
            "enabled": bool(enabled),
            "mode": self._tactile_normalization_mode(),
            "min": self._tactile_normalization_min(),
            "max": self._tactile_normalization_max(),
            "range": self._tactile_normalization_range(),
            "eps": self._tactile_normalization_eps(),
            "clip": self._tactile_normalization_clip_enabled(),
            "source": self._tactile_normalization_source(),
            "representation": self._tactile_representation() if self._tactile_observation_enabled() else None,
        }

    def _resolve_tactile_normalization_stats(self) -> Dict[str, Any]:
        mode = self._tactile_normalization_mode()
        if mode == "offset_scale":
            offset = self._tactile_normalization_config_float("offset")
            scale = self._tactile_normalization_config_float("scale")
            offset = 0.0 if offset is None else float(offset)
            scale = 1.0 if scale is None else float(scale)
            if scale == 0.0:
                raise ValueError("tactile normalization scale must be non-zero.")
            return {
                "enabled": True,
                "mode": "offset_scale",
                "offset": offset,
                "scale": scale,
                "min": offset,
                "max": None,
                "range": scale,
                "eps": self._tactile_normalization_eps(),
                "clip": self._tactile_normalization_clip_enabled(),
                "source": "config",
                "representation": self._tactile_representation(),
            }

        configured_min = self._tactile_normalization_config_float("min", "minimum")
        configured_max = self._tactile_normalization_config_float("max", "maximum")
        if (configured_min is None) != (configured_max is None):
            raise ValueError("tactile min-max normalization requires both normalization.min and normalization.max.")
        if configured_min is None:
            min_value, max_value = self._compute_tactile_dataset_minmax()
            source = "dataset_scan"
        else:
            min_value = float(configured_min)
            max_value = float(configured_max)
            source = "config"

        value_range = float(max_value - min_value)
        eps = self._tactile_normalization_eps()
        if value_range <= eps:
            raise ValueError(
                f"tactile min-max normalization requires max > min; got min={min_value}, max={max_value}."
            )
        return {
            "enabled": True,
            "mode": "minmax",
            "min": float(min_value),
            "max": float(max_value),
            "range": value_range,
            "offset": float(min_value),
            "scale": value_range,
            "eps": eps,
            "clip": self._tactile_normalization_clip_enabled(),
            "source": source,
            "representation": self._tactile_representation(),
        }

    def _compute_tactile_dataset_minmax(self) -> tuple[float, float]:
        episode_rows: Dict[tuple[str, int], set[int]] = {}
        for address in self.addresses:
            key = (str(address.dataset_id), int(address.episode_index))
            indices = episode_rows.setdefault(key, set())
            for row_index in self._tactile_history_indices(address.frame_index):
                indices.add(int(row_index))

        min_value = float("inf")
        max_value = float("-inf")
        for (dataset_id, episode_index), row_indices in episode_rows.items():
            manifest = self.manifest_by_id[dataset_id]
            rows = self.episode_cache.get_rows(self._episode_parquet_path(manifest, episode_index))
            for row_index in sorted(row_indices):
                raw_values = self._tensor_from_value(rows[row_index][self._tactile_observation_field()]).numpy()
                tactile_values = self._convert_tactile_values(raw_values)
                finite_values = tactile_values[np.isfinite(tactile_values)]
                if finite_values.size == 0:
                    continue
                min_value = min(min_value, float(finite_values.min()))
                max_value = max(max_value, float(finite_values.max()))

        if not np.isfinite(min_value) or not np.isfinite(max_value):
            raise ValueError("Unable to compute tactile normalization min/max because no finite values were found.")
        return min_value, max_value

    def _set_config_value(self, config: Any, key: str, value: Any) -> None:
        if config is None:
            return
        try:
            if isinstance(config, dict):
                config[str(key)] = value
            else:
                setattr(config, str(key), value)
        except Exception:
            return

    def _write_tactile_normalization_stats_to_config(self, stats: Dict[str, Any]) -> None:
        tactile_config = self._tactile_observation_config()
        normalization_config = self._tactile_normalization_config()
        self._set_config_value(tactile_config, "normalize", True)
        self._set_config_value(normalization_config, "enabled", True)
        for key in ("mode", "min", "max", "range", "eps", "clip", "source"):
            if key in stats:
                self._set_config_value(normalization_config, key, stats[key])

    def _tactile_raw_dim(self, manifest: DatasetManifest) -> int:
        return self._feature_dim(manifest, self._tactile_observation_field())

    def _tactile_output_dim(self, manifest: DatasetManifest) -> int:
        if self._tactile_representation() == "raw":
            return self._tactile_raw_dim(manifest)
        if self.tactile_converter is None:
            return self._tactile_expected_dim()
        return self.tactile_converter.output_dim

    def _build_tactile_converter(self) -> InspireTactileUVConverter:
        return InspireTactileUVConverter(
            uv_conversion_root=self._tactile_uv_conversion_root(),
            hand=self._tactile_hand(),
            expected_dim=self._tactile_expected_dim(),
            correct_order=self._tactile_correct_order(),
        )

    def _validate_configuration(self) -> None:
        required_fields = [str(field_name) for field_name in self.sequence_config.state_fields]
        required_fields.extend(str(field_name) for field_name in self.sequence_config.action_fields)
        if self._tactile_observation_enabled():
            required_fields.append(self._tactile_observation_field())
        if bool(self.video_config.enabled):
            video_key = str(self.video_config.key)
            required_fields.append(f"{video_key}.path")
            required_fields.append(f"{video_key}.video_frame_index")

        for manifest in self.manifests:
            missing_fields = [field_name for field_name in required_fields if field_name not in manifest.feature_specs]
            if missing_fields:
                raise ValueError(f"RealRobot manifest {manifest.dataset_id} is missing required fields: {missing_fields}")
            if self._tactile_observation_enabled():
                raw_dim = self._tactile_raw_dim(manifest)
                if int(raw_dim) != 1062:
                    raise ValueError(
                        f"RealRobot tactile field {self._tactile_observation_field()} must be 1062-dim, "
                        f"got {raw_dim} in {manifest.dataset_id}."
                    )

    def _format_temporal_target(self, sequence: torch.Tensor) -> torch.Tensor:
        if int(sequence.shape[0]) == 1:
            return sequence[0]
        return sequence
