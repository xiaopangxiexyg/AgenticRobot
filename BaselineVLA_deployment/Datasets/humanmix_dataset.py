from __future__ import annotations

from pathlib import Path
import random
import warnings
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from Datasets.base import CanonicalSample, DatasetManifest, SampleAddress
from Datasets.config_utils import get_positive_limit, safe_config_get
from Datasets.desktasktac_dataset import (
    _EpisodeRowCache,
    _merge_address_groups,
    _product,
    _sequence_offsets,
    _stable_seed,
    infer_manifest_med_capabilities,
)
from Datasets.task_utils import build_task_name_to_id
from Datasets.video_utils import build_rgb_mean_video_tokens, load_video_frames_ffmpeg


class HumanMixVideoLoadError(RuntimeError):
    def __init__(
        self,
        *,
        video_path: Path | None,
        frame_indices: List[int],
        original_error: BaseException,
    ) -> None:
        self.video_path = None if video_path is None else str(video_path)
        self.frame_indices = [int(frame_index) for frame_index in frame_indices]
        self.original_error = original_error
        super().__init__(
            f"HumanMix video load failed for {self.video_path} frames {self.frame_indices}: {original_error}"
        )


class HumanMixSequenceDataset(Dataset):
    dataset_type_name = "HumanMix_Sequence"

    def __init__(self, manifests: List[DatasetManifest], split: str, config: Any) -> None:
        self.manifests = manifests
        self.split = str(split)
        self.config = config
        self.sequence_config = config.sequence
        self.video_config = config.video
        self.med_intermediate_key = self._med_intermediate_key()
        self.emit_med_intermediate = self._should_emit_med_intermediate()
        self.manifest_by_id = {manifest.dataset_id: manifest for manifest in self.manifests}
        self.task_name_to_id = build_task_name_to_id(self.manifests)
        self.required_columns = self._build_required_columns()
        self.episode_cache = _EpisodeRowCache(
            max_size=int(config.sampling.max_episode_cache_size),
            required_columns=self.required_columns,
        )
        self._validate_configuration()
        self.addresses = self._build_addresses()
        self.sample_count_by_dataset = self._count_addresses_by_dataset(self.addresses)
        self.sample_count_by_source_dataset = self._count_addresses_by_source_dataset(self.addresses)
        self._bad_video_paths: set[str] = set()
        self._warned_bad_video_paths: set[str] = set()
        if not self.addresses:
            raise RuntimeError(f"No samples discovered for split={split}. Check filters or sampling windows.")

    def __len__(self) -> int:
        return len(self.addresses)

    def __getitem__(self, index: int) -> CanonicalSample:
        requested_index = int(index)
        policy = self._video_decode_error_policy()
        if policy == "raise" or not bool(self.video_config.enabled):
            return self._build_sample(self.addresses[requested_index])

        max_attempts = min(len(self.addresses), self._video_decode_error_max_replacements() + 1)
        last_error: HumanMixVideoLoadError | None = None
        for attempt_index in range(max_attempts):
            candidate_index = (requested_index + attempt_index) % len(self.addresses)
            candidate_address = self.addresses[candidate_index]
            try:
                sample = self._build_sample(candidate_address)
            except HumanMixVideoLoadError as error:
                if not self._is_recoverable_video_error(error):
                    raise
                last_error = error
                self._remember_bad_video(error)
                if policy != "replace_sample":
                    raise
                continue

            if attempt_index > 0:
                requested_address = self.addresses[requested_index]
                sample.metadata["video_replaced_sample"] = True
                sample.metadata["requested_dataset_id"] = requested_address.dataset_id
                sample.metadata["requested_episode_index"] = requested_address.episode_index
                sample.metadata["requested_frame_index"] = requested_address.frame_index
                sample.metadata["replacement_attempts"] = attempt_index
                if last_error is not None:
                    sample.metadata["replacement_reason"] = str(last_error)
            return sample

        if last_error is not None:
            raise RuntimeError(
                "Unable to find a replacement HumanMix sample after "
                f"{max_attempts} attempts for requested index={requested_index}. Last error: {last_error}"
            ) from last_error
        return self._build_sample(self.addresses[requested_index])

    def _build_sample(self, address: SampleAddress) -> CanonicalSample:
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

        tactile_indices = self._future_tactile_indices(address.frame_index)
        med_shape = self._infer_med_source_shape(manifest, len(tactile_indices))
        med_intermediate_value = None
        if self.emit_med_intermediate:
            med_intermediate_value = self._build_med_intermediate(rows=rows, indices=tactile_indices)
            med_shape = [int(value) for value in med_intermediate_value.shape]

        tactile_observation = None
        tactile_observation_mask = None
        if self._tactile_observation_enabled():
            tactile_history_indices = self._tactile_history_indices(address.frame_index)
            tactile_observation = torch.stack(
                [self._build_tactile_observation_frame(rows[row_index]) for row_index in tactile_history_indices],
                dim=0,
            )
            tactile_observation_mask = torch.ones(int(tactile_observation.shape[0]), dtype=torch.float32)

        video_path: Path | None = None
        video_frame_index = int(current_row.get(f"{str(self.video_config.key)}_frame_index", current_row.get("frame_index", 0)))
        try:
            video_path, video_frame_index = self._resolve_video_location(manifest, current_row, address.episode_index)
            video_frames = self._load_video_frames(video_path=video_path, current_video_frame_index=video_frame_index)
        except (FileNotFoundError, KeyError, RuntimeError) as error:
            if not bool(self.video_config.enabled):
                raise
            raise HumanMixVideoLoadError(
                video_path=video_path,
                frame_indices=self._video_frame_indices(video_frame_index),
                original_error=error,
            ) from error
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
            intermediates={} if med_intermediate_value is None else {self.med_intermediate_key: med_intermediate_value},
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
                "original_episode_index": int(manifest.metadata.get("episode_original_index", {}).get(address.episode_index, address.episode_index)),
                "source_dataset": str(manifest.metadata.get("episode_source_dataset", {}).get(address.episode_index, "")),
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
                "future_tactile_steps": int(len(tactile_indices)),
                "med_future_start": self._med_future_start(),
                "med_future_end": self._med_future_end(),
                "med_stride": self._med_stride(),
                "med_intermediate_key": self.med_intermediate_key,
                "med_intermediate_emitted": bool(self.emit_med_intermediate),
                "med_side": self._med_side(),
                "med_source_kind": self._med_kind(),
                "med_selected_fields": self._selected_med_fields(),
                "med_shape": [int(value) for value in med_shape],
                "tactile_observation_enabled": self._tactile_observation_enabled(),
                "tactile_observation_field": self._tactile_observation_field() if self._tactile_observation_enabled() else None,
                "tactile_observation_mask_field": self._tactile_observation_mask_field() if self._tactile_observation_enabled() and self._tactile_observation_zero_invalid() else None,
                "tactile_observation_history": self._tactile_observation_history(),
                "tactile_observation_stride": self._tactile_observation_stride(),
                "tactile_observation_shape": [] if tactile_observation is None else [int(value) for value in tactile_observation.shape],
                "video_replaced_sample": False,
            },
        )

    @property
    def summary(self) -> Dict[str, Any]:
        first_manifest = self.manifests[0]
        state_step_dim = sum(self._feature_dim(first_manifest, field_name) for field_name in self.sequence_config.state_fields)
        action_step_dim = sum(self._feature_dim(first_manifest, field_name) for field_name in self.sequence_config.action_fields)
        future_tactile_steps = len(self._future_tactile_offsets())
        source_dataset_summary = dict(first_manifest.metadata.get("source_dataset_summary", {}))
        shared_capabilities = set(infer_manifest_med_capabilities(first_manifest))
        for manifest in self.manifests[1:]:
            shared_capabilities &= set(infer_manifest_med_capabilities(manifest))
        return {
            "dataset_type": self.dataset_type_name,
            "manifest_loader": str(self.config.manifest_loader),
            "split": self.split,
            "sample_count": len(self.addresses),
            "dataset_count": len(self.manifests),
            "source_dataset_count": len(source_dataset_summary),
            "sample_count_by_dataset": dict(self.sample_count_by_dataset),
            "sample_count_by_source_dataset": dict(self.sample_count_by_source_dataset),
            "state_dim": state_step_dim * int(self.sequence_config.state_history),
            "state_step_dim": state_step_dim,
            "state_history": int(self.sequence_config.state_history),
            "action_dim": action_step_dim,
            "action_chunk": self._action_chunk(),
            "video_key": str(self.video_config.key),
            "video_token_slots": int(self.video_config.clip_length) if bool(self.video_config.enabled) else 0,
            "video_clip_length": int(self.video_config.clip_length) if bool(self.video_config.enabled) else 0,
            "tactile_dim": self._feature_dim(first_manifest, self._med_source_field()),
            "tactile_observation_enabled": self._tactile_observation_enabled(),
            "tactile_observation_dim": None if not self._tactile_observation_enabled() else self._feature_dim(first_manifest, self._tactile_observation_field()),
            "tactile_observation_history": self._tactile_observation_history(),
            "tactile_observation_stride": self._tactile_observation_stride(),
            "tactile_observation_shape": []
            if not self._tactile_observation_enabled()
            else [self._tactile_observation_history(), self._feature_dim(first_manifest, self._tactile_observation_field())],
            "tactile_observation_fields": {}
            if not self._tactile_observation_enabled()
            else {
                "field": self._tactile_observation_field(),
                "mask_field": self._tactile_observation_mask_field() if self._tactile_observation_zero_invalid() else None,
                "zero_invalid": self._tactile_observation_zero_invalid(),
            },
            "future_tactile_steps": future_tactile_steps,
            "med_steps": future_tactile_steps,
            "med_future_start": self._med_future_start(),
            "med_future_end": self._med_future_end(),
            "med_stride": self._med_stride(),
            "med_intermediate_key": self.med_intermediate_key,
            "med_intermediate_emitted": bool(self.emit_med_intermediate),
            "med_side": self._med_side(),
            "med_source_kind": self._med_kind(),
            "med_source_shape": self._infer_med_source_shape(first_manifest, future_tactile_steps),
            "med_selected_fields": self._selected_med_fields(),
            "available_med_capabilities": sorted(shared_capabilities),
            "video_decode_error_policy": self._video_decode_error_policy(),
            "video_decode_error_max_replacements": self._video_decode_error_max_replacements(),
            "task_count": len(self.task_name_to_id),
            "task_names": [task_name for task_name, _ in sorted(self.task_name_to_id.items(), key=lambda item: item[1])],
            "dataset_ids": [manifest.dataset_id for manifest in self.manifests],
            "source_dataset_summary": source_dataset_summary,
            "med_point_count": None,
            "med_point_feature_dim": None,
        }

    def _build_required_columns(self) -> List[str]:
        columns = {"episode_index", "frame_index", "task_index"}
        columns.update(str(field_name) for field_name in self.sequence_config.state_fields)
        columns.update(str(field_name) for field_name in self.sequence_config.action_fields)
        if self.emit_med_intermediate:
            columns.add(self._med_source_field())
            if self._med_vector_zero_invalid():
                columns.add(self._med_vector_mask_field())
        if self._tactile_observation_enabled():
            columns.add(self._tactile_observation_field())
            if self._tactile_observation_zero_invalid():
                columns.add(self._tactile_observation_mask_field())
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
        future_requirement = max(
            (self._action_chunk() - 1) * int(self.sequence_config.action_stride),
            max(self._future_tactile_offsets()),
        )
        sampling_config = safe_config_get(self.config, "sampling", None)
        history_requirement = max(
            int(history_requirement),
            int(safe_config_get(sampling_config, "min_history_frames", 0) or 0),
        )
        future_requirement = max(
            int(future_requirement),
            int(safe_config_get(sampling_config, "min_future_frames", 0) or 0),
        )
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

    def _count_addresses_by_source_dataset(self, addresses: List[SampleAddress]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for address in addresses:
            manifest = self.manifest_by_id[address.dataset_id]
            source_name = str(manifest.metadata.get("episode_source_dataset", {}).get(address.episode_index, "unknown"))
            counts[source_name] = counts.get(source_name, 0) + 1
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

    def _future_tactile_offsets(self) -> List[int]:
        return list(range(self._med_future_start(), self._med_future_end() + 1, self._med_stride()))

    def _future_tactile_indices(self, current_index: int) -> List[int]:
        return [current_index + offset for offset in self._future_tactile_offsets()]

    def _concat_fields(self, row: Dict[str, Any], field_names: List[str]) -> torch.Tensor:
        chunks = [self._tensor_from_value(row[str(field_name)]) for field_name in field_names]
        return torch.cat(chunks, dim=0).to(dtype=torch.float32)

    def _tensor_from_value(self, value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(dtype=torch.float32).reshape(-1)
        if isinstance(value, list):
            return torch.tensor(value, dtype=torch.float32).reshape(-1)
        return torch.tensor([float(value)], dtype=torch.float32)

    def _build_med_intermediate(self, rows: List[Dict[str, Any]], indices: List[int]) -> torch.Tensor:
        if self._med_kind() != "vector":
            raise ValueError("HumanMix only supports tactile vector intermediates.")
        return torch.stack([self._build_vector_frame(rows[row_index]) for row_index in indices], dim=0)

    def _build_vector_frame(self, row: Dict[str, Any]) -> torch.Tensor:
        values = self._tensor_from_value(row[self._med_source_field()])
        if not self._med_vector_zero_invalid():
            return values
        mask = self._tensor_from_value(row[self._med_vector_mask_field()]).to(dtype=torch.float32)
        if int(mask.numel()) != int(values.numel()):
            raise ValueError(
                f"Vector mask dim mismatch for {self._med_vector_mask_field()}: "
                f"expected {values.numel()}, got {mask.numel()}."
            )
        return values * mask

    def _build_tactile_observation_frame(self, row: Dict[str, Any]) -> torch.Tensor:
        values = self._tensor_from_value(row[self._tactile_observation_field()])
        if not self._tactile_observation_zero_invalid():
            return values
        mask = self._tensor_from_value(row[self._tactile_observation_mask_field()]).to(dtype=torch.float32)
        if int(mask.numel()) != int(values.numel()):
            raise ValueError(
                f"Tactile observation mask dim mismatch for {self._tactile_observation_mask_field()}: "
                f"expected {values.numel()}, got {mask.numel()}."
            )
        return values * mask

    def _load_video_frames(self, video_path: Path | None, current_video_frame_index: int) -> torch.Tensor:
        if not bool(self.video_config.enabled):
            return torch.zeros(0, 3, int(self.video_config.resize_height), int(self.video_config.resize_width), dtype=torch.float32)
        if video_path is None:
            raise FileNotFoundError("HumanMix video is enabled, but no episode video path was resolved.")
        video_path_key = str(Path(video_path).expanduser().resolve())
        if video_path_key in self._bad_video_paths:
            raise RuntimeError(f"Video path was previously marked invalid in this worker: {video_path_key}")

        return load_video_frames_ffmpeg(
            video_path=video_path,
            frame_indices=self._video_frame_indices(current_video_frame_index),
            resize_height=int(self.video_config.resize_height),
            resize_width=int(self.video_config.resize_width),
        )

    def _video_frame_indices(self, current_video_frame_index: int) -> List[int]:
        return [
            int(current_video_frame_index) - offset
            for offset in _sequence_offsets(
                length=int(self.video_config.clip_length),
                stride=int(self.video_config.clip_stride),
                reverse=True,
            )
        ]

    def _video_decode_error_policy(self) -> str:
        value = safe_config_get(self.video_config, "decode_error_policy", "raise")
        normalized = str(value).strip().lower()
        aliases = {
            "strict": "raise",
            "error": "raise",
            "fail": "raise",
            "raise": "raise",
            "replace": "replace_sample",
            "replacement": "replace_sample",
            "replace_sample": "replace_sample",
            "skip": "replace_sample",
            "drop": "replace_sample",
        }
        if normalized not in aliases:
            raise ValueError(
                "Unsupported HumanMix video.decode_error_policy="
                f"{value!r}. Expected one of: raise, replace_sample."
            )
        return aliases[normalized]

    def _video_decode_error_max_replacements(self) -> int:
        value = safe_config_get(self.video_config, "decode_error_max_replacements", 128)
        return max(0, int(value))

    def _video_warn_on_decode_error(self) -> bool:
        return bool(safe_config_get(self.video_config, "warn_on_decode_error", True))

    def _remember_bad_video(self, error: HumanMixVideoLoadError) -> None:
        if error.video_path is None:
            return
        self._bad_video_paths.add(str(error.video_path))
        if not self._video_warn_on_decode_error() or str(error.video_path) in self._warned_bad_video_paths:
            return
        self._warned_bad_video_paths.add(str(error.video_path))
        warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    def _is_recoverable_video_error(self, error: HumanMixVideoLoadError) -> bool:
        original_text = str(error.original_error).lower()
        if "ffmpeg executable not found" in original_text:
            return False
        return True

    def _resolve_video_location(
        self,
        manifest: DatasetManifest,
        current_row: Dict[str, Any],
        episode_index: int,
    ) -> tuple[Path | None, int]:
        current_video_frame_index = int(current_row.get(f"{str(self.video_config.key)}_frame_index", current_row.get("frame_index", 0)))
        if not bool(self.video_config.enabled):
            return None, current_video_frame_index

        video_lookup = manifest.metadata.get("episode_video_lookup", {})
        video_value = video_lookup.get(int(episode_index))
        if video_value is None:
            raise KeyError(f"Missing HumanMix video lookup for episode_index={episode_index}.")
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

    def _med_config(self) -> Any:
        med_config = getattr(self.sequence_config, "med", None)
        return self.sequence_config if med_config is None else med_config

    def _med_vector_config(self) -> Any:
        med_config = getattr(self.sequence_config, "med", None)
        return None if med_config is None else getattr(med_config, "vector", None)

    def _med_side(self) -> str:
        med_config = self._med_config()
        return str(getattr(med_config, "side", "right")).lower()

    def _med_kind(self) -> str:
        med_config = self._med_config()
        return str(getattr(med_config, "kind", getattr(self.sequence_config, "med_representation", "vector"))).lower()

    def _med_source_field(self) -> str:
        vector_config = self._med_vector_config()
        explicit_field = None if vector_config is None else getattr(vector_config, "field", None)
        if explicit_field not in (None, "null"):
            return str(explicit_field)
        legacy_field = getattr(self.sequence_config, "med_source_field", None)
        if legacy_field not in (None, "null"):
            return str(legacy_field)
        return f"tactile.{self._med_side()}_tacglove"

    def _med_vector_mask_field(self) -> str:
        vector_config = self._med_vector_config()
        explicit_field = None if vector_config is None else getattr(vector_config, "mask_field", None)
        if explicit_field not in (None, "null"):
            return str(explicit_field)
        legacy_field = getattr(self.sequence_config, "med_vector_mask_field", None)
        if legacy_field not in (None, "null"):
            return str(legacy_field)
        return f"tactile.{self._med_side()}_tacglove_mask"

    def _med_vector_zero_invalid(self) -> bool:
        vector_config = self._med_vector_config()
        explicit_value = None if vector_config is None else getattr(vector_config, "zero_invalid", None)
        if explicit_value is not None:
            return bool(explicit_value)
        legacy_value = getattr(self.sequence_config, "med_vector_zero_invalid", None)
        if legacy_value is not None:
            return bool(legacy_value)
        return False

    def _med_intermediate_key(self) -> str:
        med_config = self._med_config()
        explicit_key = getattr(med_config, "intermediate_key", None)
        if explicit_key not in (None, "null"):
            return str(explicit_key)
        legacy_key = getattr(self.sequence_config, "med_intermediate_key", None)
        if legacy_key not in (None, "null"):
            return str(legacy_key)
        return f"future_{self._med_side()}_tactile"

    def _should_emit_med_intermediate(self) -> bool:
        med_config = self._med_config()
        explicit_value = getattr(med_config, "emit_intermediate", None)
        if explicit_value is not None:
            return bool(explicit_value)
        legacy_value = getattr(self.sequence_config, "med_emit_intermediate", None)
        if legacy_value is not None:
            return bool(legacy_value)
        return True

    def _med_future_start(self) -> int:
        med_config = self._med_config()
        return int(getattr(med_config, "future_start", getattr(self.sequence_config, "med_future_start", 1)))

    def _med_future_end(self) -> int:
        med_config = self._med_config()
        return int(getattr(med_config, "future_end", getattr(self.sequence_config, "med_future_end", 1)))

    def _med_stride(self) -> int:
        med_config = self._med_config()
        return int(getattr(med_config, "stride", getattr(self.sequence_config, "med_stride", 1)))

    def _selected_med_fields(self) -> Dict[str, Any]:
        return {
            "kind": self._med_kind(),
            "side": self._med_side(),
            "vector_field": self._med_source_field(),
            "mask_field": self._med_vector_mask_field() if self._med_vector_zero_invalid() else None,
            "zero_invalid": self._med_vector_zero_invalid(),
        }

    def _infer_med_source_shape(self, manifest: DatasetManifest, future_steps: int) -> List[int]:
        return [int(future_steps), int(self._feature_dim(manifest, self._med_source_field()))]

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
        return self._med_source_field()

    def _tactile_observation_mask_field(self) -> str:
        tactile_config = self._tactile_observation_config()
        explicit_field = None if tactile_config is None else getattr(tactile_config, "mask_field", None)
        if explicit_field not in (None, "null"):
            return str(explicit_field)
        side = self._med_side()
        return f"tactile.{side}_tacglove_mask"

    def _tactile_observation_zero_invalid(self) -> bool:
        tactile_config = self._tactile_observation_config()
        if tactile_config is None:
            return False
        explicit_value = getattr(tactile_config, "zero_invalid", None)
        if explicit_value is not None:
            return bool(explicit_value)
        return False

    def _tactile_observation_history(self) -> int:
        tactile_config = self._tactile_observation_config()
        if tactile_config is None:
            return 1
        if bool(getattr(tactile_config, "use_state_history", False)):
            return int(self.sequence_config.state_history)
        history_value = getattr(tactile_config, "history", None)
        if history_value is None:
            history_value = getattr(tactile_config, "history_size", 1)
        return max(1, int(history_value))

    def _tactile_observation_stride(self) -> int:
        tactile_config = self._tactile_observation_config()
        if tactile_config is None:
            return int(self.sequence_config.state_stride)
        if bool(getattr(tactile_config, "use_state_history", False)):
            return int(self.sequence_config.state_stride)
        stride_value = getattr(tactile_config, "stride", None)
        if stride_value is None:
            stride_value = int(self.sequence_config.state_stride)
        return max(1, int(stride_value))

    def _validate_configuration(self) -> None:
        if self._med_kind() != "vector":
            raise ValueError("HumanMix is configured as tactile-vector only; set raw.sequence.med.kind=vector.")

        required_fields = [str(field_name) for field_name in self.sequence_config.state_fields]
        required_fields.extend(str(field_name) for field_name in self.sequence_config.action_fields)
        if self.emit_med_intermediate:
            required_fields.append(self._med_source_field())
            if self._med_vector_zero_invalid():
                required_fields.append(self._med_vector_mask_field())
        if self._tactile_observation_enabled():
            required_fields.append(self._tactile_observation_field())
            if self._tactile_observation_zero_invalid():
                required_fields.append(self._tactile_observation_mask_field())

        expected_vector_dim = None
        for manifest in self.manifests:
            missing_fields = [field_name for field_name in required_fields if field_name not in manifest.feature_specs]
            if missing_fields:
                raise ValueError(f"HumanMix manifest {manifest.dataset_id} is missing required fields: {missing_fields}")

            vector_dim = self._feature_dim(manifest, self._med_source_field())
            if expected_vector_dim is None:
                expected_vector_dim = vector_dim
            elif int(vector_dim) != int(expected_vector_dim):
                raise ValueError(
                    f"HumanMix manifests have inconsistent tactile vector dims: "
                    f"expected {expected_vector_dim}, got {vector_dim} in {manifest.dataset_id}."
                )

            if self._med_vector_zero_invalid():
                mask_dim = self._feature_dim(manifest, self._med_vector_mask_field())
                if int(mask_dim) != int(vector_dim):
                    raise ValueError(
                        f"HumanMix tactile mask dim mismatch in {manifest.dataset_id}: "
                        f"vector={vector_dim}, mask={mask_dim}."
                    )

            if self._tactile_observation_enabled():
                tactile_dim = self._feature_dim(manifest, self._tactile_observation_field())
                if int(tactile_dim) <= 0:
                    raise ValueError(
                        f"HumanMix tactile observation field={self._tactile_observation_field()} "
                        f"has invalid dim={tactile_dim} in {manifest.dataset_id}."
                    )
                if self._tactile_observation_zero_invalid():
                    tactile_mask_dim = self._feature_dim(manifest, self._tactile_observation_mask_field())
                    if int(tactile_mask_dim) != int(tactile_dim):
                        raise ValueError(
                            f"HumanMix tactile observation mask dim mismatch in {manifest.dataset_id}: "
                            f"vector={tactile_dim}, mask={tactile_mask_dim}."
                        )

            if bool(self.video_config.enabled):
                configured_video_key = str(self.video_config.key)
                if configured_video_key not in manifest.video_keys:
                    raise ValueError(
                        f"Configured HumanMix video key={configured_video_key!r} is missing. "
                        f"Available video keys: {manifest.video_keys}."
                    )

    def _format_temporal_target(self, sequence: torch.Tensor) -> torch.Tensor:
        if int(sequence.shape[0]) == 1:
            return sequence[0]
        return sequence
