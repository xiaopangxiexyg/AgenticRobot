from __future__ import annotations
from collections import OrderedDict
import hashlib
from pathlib import Path
import random
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from Datasets.base import CanonicalSample, DatasetManifest, SampleAddress
from Datasets.config_utils import get_positive_limit
from Datasets.task_utils import build_task_name_to_id
from Datasets.video_utils import build_rgb_mean_video_tokens, load_video_frames_ffmpeg


MED_CAPABILITY_FIELDS = {
    "right_tactile_vector": ("tactile.right_tacglove",),
    "left_tactile_vector": ("tactile.left_tacglove",),
    "right_tactile_point_cloud": ("tactile_cloud.right_xyz", "tactile_cloud.right_valid_mask"),
    "left_tactile_point_cloud": ("tactile_cloud.left_xyz", "tactile_cloud.left_valid_mask"),
}


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "DeskTaskTac real-data loading requires 'pyarrow' in the ManipRep_310 environment."
        ) from error
    return pq


def _product(shape: List[int]) -> int:
    total = 1
    for value in shape:
        total *= int(value)
    return total


def _stable_seed(*parts: Any) -> int:
    joined = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**31)


def _sequence_offsets(length: int, stride: int, reverse: bool = False) -> List[int]:
    offsets = [index * int(stride) for index in range(int(length))]
    if reverse:
        offsets = list(reversed(offsets))
    return offsets


def _axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    axis_angle = axis_angle.to(dtype=torch.float32).reshape(3)
    angle = torch.linalg.vector_norm(axis_angle)
    if not bool(torch.isfinite(angle).item()) or float(angle.item()) < 1e-8:
        return torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)

    axis = axis_angle / angle
    x, y, z = axis.unbind()
    zero = axis.new_tensor(0.0)
    skew = torch.stack(
        [
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ]
    ).view(3, 3)
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return identity + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)


def _normalize_point_cloud_coordinate_frame(value: Any) -> str:
    normalized = str(value or "camera").strip().lower()
    aliases = {
        "camera": "camera",
        "camera_frame": "camera",
        "first_person": "camera",
        "first_person_camera": "camera",
        "first_person_camera_opencv": "camera",
        "egocentric": "camera",
        "wrist": "wrist_local",
        "wrist_local": "wrist_local",
        "eef": "wrist_local",
        "eef_local": "wrist_local",
        "hand": "wrist_local",
        "hand_local": "wrist_local",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported point-cloud coordinate_frame="
            f"{value!r}. Supported values: camera, first_person_camera_opencv, wrist_local."
        )
    return aliases[normalized]


def _merge_address_groups(groups: List[List[SampleAddress]], max_total_samples: int | None, strategy: str) -> List[SampleAddress]:
    if not groups:
        return []
    normalized_strategy = str(strategy).lower()
    if normalized_strategy not in {"concat", "round_robin"}:
        raise ValueError(f"Unsupported dataset mix strategy: {strategy}")
    if normalized_strategy == "concat" or len(groups) <= 1:
        merged = [address for group in groups for address in group]
        if max_total_samples is not None:
            return merged[: int(max_total_samples)]
        return merged

    cursors = [0 for _ in groups]
    merged: List[SampleAddress] = []
    while True:
        advanced = False
        for group_index, group in enumerate(groups):
            if cursors[group_index] >= len(group):
                continue
            merged.append(group[cursors[group_index]])
            cursors[group_index] += 1
            advanced = True
            if max_total_samples is not None and len(merged) >= int(max_total_samples):
                return merged
        if not advanced:
            break
    return merged


def infer_manifest_med_capabilities(manifest: DatasetManifest) -> List[str]:
    feature_specs = getattr(manifest, "feature_specs", {})
    capabilities = []
    for capability_name, required_fields in MED_CAPABILITY_FIELDS.items():
        if all(str(field_name) in feature_specs for field_name in required_fields):
            capabilities.append(capability_name)
    return sorted(capabilities)


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


class DeskTaskTacSequenceDataset(Dataset):
    dataset_type_name = "DeskTaskTac_Sequence"

    def __init__(self, manifests: List[DatasetManifest], split: str, config: Any) -> None:
        self.manifests = manifests
        self.split = split
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
        self._validate_med_configuration()
        self.addresses = self._build_addresses()
        self.sample_count_by_dataset = self._count_addresses_by_dataset(self.addresses)
        if not self.addresses:
            raise RuntimeError(f"No samples discovered for split={split}. Check filters or sampling windows.")

    def __len__(self) -> int:
        return len(self.addresses)

    def __getitem__(self, index: int) -> CanonicalSample:
        address = self.addresses[index]
        manifest = self.manifest_by_id[address.dataset_id]
        rows = self.episode_cache.get_rows(self._episode_parquet_path(manifest, address.episode_index))

        current_row = rows[address.frame_index]
        episode_parquet_path = self._episode_parquet_path(manifest, address.episode_index)
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

        video_frames = self._load_video_frames(manifest, current_row)
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
            targets={
                "action": action,
            },
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
                "episode_length": address.episode_length,
                "episode_parquet_path": episode_parquet_path,
                "state_dim": int(state.shape[0]),
                "state_step_dim": int(state_sequence.shape[-1]),
                "action_dim": int(action_sequence.shape[-1]),
                "action_chunk": int(action_sequence.shape[0]),
                "task_text": task_text,
                "task_id": task_id,
                "video_key": str(self.video_config.key),
                "video_path": str((Path(manifest.root) / str(current_row[str(self.video_config.key)])).resolve()),
                "video_frame_index": int(current_row[f"{str(self.video_config.key)}_frame_index"]),
                "session_dir": str(current_row.get("session_dir", "")),
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
            },
        )

    @property
    def summary(self) -> Dict[str, Any]:
        first_manifest = self.manifests[0]
        state_step_dim = sum(self._feature_dim(first_manifest, field_name) for field_name in self.sequence_config.state_fields)
        action_step_dim = sum(self._feature_dim(first_manifest, field_name) for field_name in self.sequence_config.action_fields)
        med_source_kind = self._med_kind()
        tactile_dim = None
        if med_source_kind == "vector":
            tactile_dim = self._feature_dim(first_manifest, self._med_source_field())
        future_tactile_steps = len(self._future_tactile_offsets())
        med_source_shape = self._infer_med_source_shape(first_manifest, future_tactile_steps)
        shared_capabilities = set(infer_manifest_med_capabilities(first_manifest))
        for manifest in self.manifests[1:]:
            shared_capabilities &= set(infer_manifest_med_capabilities(manifest))
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
            "action_dim": action_step_dim,
            "action_chunk": self._action_chunk(),
            "video_token_slots": int(self.video_config.clip_length) if bool(self.video_config.enabled) else 0,
            "video_clip_length": int(self.video_config.clip_length) if bool(self.video_config.enabled) else 0,
            "tactile_dim": tactile_dim,
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
            "med_source_kind": med_source_kind,
            "med_source_shape": med_source_shape,
            "med_selected_fields": self._selected_med_fields(),
            "available_med_capabilities": sorted(shared_capabilities),
            "task_count": len(self.task_name_to_id),
            "task_names": [task_name for task_name, _ in sorted(self.task_name_to_id.items(), key=lambda item: item[1])],
            "dataset_ids": [manifest.dataset_id for manifest in self.manifests],
            "med_point_count": None if med_source_kind != "point_cloud" else int(med_source_shape[-2]),
            "med_point_feature_dim": None if med_source_kind != "point_cloud" else int(med_source_shape[-1]),
        }

    def _build_required_columns(self) -> List[str]:
        columns = {
            "frame_index",
            "task_index",
            "task",
            str(self.video_config.key),
            f"{str(self.video_config.key)}_frame_index",
        }
        columns.update(str(field_name) for field_name in self.sequence_config.state_fields)
        columns.update(str(field_name) for field_name in self.sequence_config.action_fields)
        if not self.emit_med_intermediate:
            med_representation = None
        else:
            med_representation = self._med_kind()
            if med_representation == "vector":
                columns.add(self._med_source_field())
            elif med_representation == "point_cloud":
                columns.add(self._med_point_xyz_field())
                columns.add(self._med_point_valid_mask_field())
                if self._med_point_include_value():
                    columns.add(self._med_point_value_field())
                if self._med_point_coordinate_frame() == "wrist_local":
                    columns.add(self._med_point_reference_position_field())
                    columns.add(self._med_point_reference_rotation_field())
            else:
                raise ValueError(f"Unsupported med_representation={med_representation}")
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
            episode_indices = self._select_episode_indices(manifest)
            for episode_index in episode_indices:
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

    def _count_addresses_by_dataset(self, addresses: List[SampleAddress]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for address in addresses:
            counts[address.dataset_id] = counts.get(address.dataset_id, 0) + 1
        return counts

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

    def _med_point_cloud_config(self) -> Any:
        med_config = getattr(self.sequence_config, "med", None)
        return None if med_config is None else getattr(med_config, "point_cloud", None)

    def _med_side(self) -> str:
        med_config = self._med_config()
        return str(getattr(med_config, "side", "right")).lower()

    def _med_kind(self) -> str:
        med_config = self._med_config()
        return str(getattr(med_config, "kind", getattr(self.sequence_config, "med_representation", "vector"))).lower()

    def _med_representation(self) -> str:
        return self._med_kind()

    def _med_source_field(self) -> str:
        vector_config = self._med_vector_config()
        explicit_field = None if vector_config is None else getattr(vector_config, "field", None)
        if explicit_field not in (None, "null"):
            return str(explicit_field)
        legacy_field = getattr(self.sequence_config, "med_source_field", None)
        if legacy_field not in (None, "null"):
            return str(legacy_field)
        return f"tactile.{self._med_side()}_tacglove"

    def _med_point_xyz_field(self) -> str:
        point_cloud_config = self._med_point_cloud_config()
        explicit_field = None if point_cloud_config is None else getattr(point_cloud_config, "xyz_field", None)
        if explicit_field not in (None, "null"):
            return str(explicit_field)
        legacy_field = getattr(self.sequence_config, "med_point_xyz_field", None)
        if legacy_field not in (None, "null"):
            return str(legacy_field)
        return f"tactile_cloud.{self._med_side()}_xyz"

    def _med_point_valid_mask_field(self) -> str:
        point_cloud_config = self._med_point_cloud_config()
        explicit_field = None if point_cloud_config is None else getattr(point_cloud_config, "valid_mask_field", None)
        if explicit_field not in (None, "null"):
            return str(explicit_field)
        legacy_field = getattr(self.sequence_config, "med_point_valid_mask_field", None)
        if legacy_field not in (None, "null"):
            return str(legacy_field)
        return f"tactile_cloud.{self._med_side()}_valid_mask"

    def _med_point_value_field(self) -> str:
        point_cloud_config = self._med_point_cloud_config()
        explicit_field = None if point_cloud_config is None else getattr(point_cloud_config, "value_field", None)
        if explicit_field not in (None, "null"):
            return str(explicit_field)
        legacy_field = getattr(self.sequence_config, "med_point_value_field", None)
        if legacy_field not in (None, "null"):
            return str(legacy_field)
        return self._med_source_field()

    def _med_point_include_value(self) -> bool:
        point_cloud_config = self._med_point_cloud_config()
        if point_cloud_config is not None and getattr(point_cloud_config, "include_value", None) is not None:
            return bool(point_cloud_config.include_value)
        return bool(getattr(self.sequence_config, "med_point_include_value", True))

    def _med_point_include_valid_feature(self) -> bool:
        point_cloud_config = self._med_point_cloud_config()
        if point_cloud_config is not None and getattr(point_cloud_config, "include_valid_feature", None) is not None:
            return bool(point_cloud_config.include_valid_feature)
        return bool(getattr(self.sequence_config, "med_point_include_valid_feature", False))

    def _med_point_coordinate_frame(self) -> str:
        point_cloud_config = self._med_point_cloud_config()
        explicit_frame = None if point_cloud_config is None else getattr(point_cloud_config, "coordinate_frame", None)
        if explicit_frame not in (None, "null"):
            return _normalize_point_cloud_coordinate_frame(explicit_frame)
        legacy_frame = getattr(self.sequence_config, "med_point_coordinate_frame", None)
        if legacy_frame not in (None, "null"):
            return _normalize_point_cloud_coordinate_frame(legacy_frame)
        return "camera"

    def _med_point_reference_position_field(self) -> str:
        point_cloud_config = self._med_point_cloud_config()
        explicit_field = None if point_cloud_config is None else getattr(point_cloud_config, "reference_position_field", None)
        if explicit_field not in (None, "null"):
            return str(explicit_field)
        legacy_field = getattr(self.sequence_config, "med_point_reference_position_field", None)
        if legacy_field not in (None, "null"):
            return str(legacy_field)
        return "state.eef_position" if self._med_side() == "right" else "state.left_eef_position"

    def _med_point_reference_rotation_field(self) -> str:
        point_cloud_config = self._med_point_cloud_config()
        explicit_field = None if point_cloud_config is None else getattr(point_cloud_config, "reference_rotation_field", None)
        if explicit_field not in (None, "null"):
            return str(explicit_field)
        legacy_field = getattr(self.sequence_config, "med_point_reference_rotation_field", None)
        if legacy_field not in (None, "null"):
            return str(legacy_field)
        return "state.eef_rotation" if self._med_side() == "right" else "state.left_eef_rotation"

    def _med_point_zero_invalid(self) -> bool:
        point_cloud_config = self._med_point_cloud_config()
        if point_cloud_config is not None and getattr(point_cloud_config, "zero_invalid", None) is not None:
            return bool(point_cloud_config.zero_invalid)
        return bool(getattr(self.sequence_config, "med_point_zero_invalid", True))

    def _med_intermediate_key(self) -> str:
        med_config = self._med_config()
        explicit_key = getattr(med_config, "intermediate_key", None)
        if explicit_key not in (None, "null"):
            return str(explicit_key)
        legacy_key = getattr(self.sequence_config, "med_intermediate_key", None)
        if legacy_key not in (None, "null"):
            return str(legacy_key)
        med_suffix = "point_cloud" if self._med_kind() == "point_cloud" else "tactile"
        return f"future_{self._med_side()}_{med_suffix}"

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
        selected = {
            "kind": self._med_kind(),
            "side": self._med_side(),
        }
        if self._med_kind() == "vector":
            selected["vector_field"] = self._med_source_field()
            return selected
        selected["xyz_field"] = self._med_point_xyz_field()
        selected["valid_mask_field"] = self._med_point_valid_mask_field()
        selected["include_value"] = self._med_point_include_value()
        if self._med_point_include_value():
            selected["value_field"] = self._med_point_value_field()
        selected["include_valid_feature"] = self._med_point_include_valid_feature()
        selected["zero_invalid"] = self._med_point_zero_invalid()
        selected["coordinate_frame"] = self._med_point_coordinate_frame()
        if self._med_point_coordinate_frame() == "wrist_local":
            selected["reference_position_field"] = self._med_point_reference_position_field()
            selected["reference_rotation_field"] = self._med_point_reference_rotation_field()
        return selected

    def _future_tactile_offsets(self) -> List[int]:
        return list(
            range(
                self._med_future_start(),
                self._med_future_end() + 1,
                self._med_stride(),
            )
        )

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
        med_representation = self._med_kind()
        if med_representation == "vector":
            return torch.stack(
                [self._tensor_from_value(rows[row_index][self._med_source_field()]) for row_index in indices],
                dim=0,
            )
        if med_representation == "point_cloud":
            return torch.stack([self._build_point_cloud_frame(rows[row_index]) for row_index in indices], dim=0)
        raise ValueError(f"Unsupported med_representation={med_representation}")

    def _build_point_cloud_frame(self, row: Dict[str, Any]) -> torch.Tensor:
        xyz = self._tensor_from_value(row[self._med_point_xyz_field()])
        point_count = self._point_count(self.manifests[0], self._med_point_xyz_field())
        xyz = xyz.view(point_count, 3)
        xyz = self._transform_point_xyz(row, xyz)
        valid_mask = self._tensor_from_value(row[self._med_point_valid_mask_field()]).view(point_count).to(dtype=torch.float32)
        feature_chunks = [xyz]
        if self._med_point_include_value():
            point_values = self._tensor_from_value(row[self._med_point_value_field()]).view(point_count, -1)
            feature_chunks.append(point_values)
        if self._med_point_include_valid_feature():
            feature_chunks.append(valid_mask.unsqueeze(-1))
        point_features = torch.cat(feature_chunks, dim=-1).to(dtype=torch.float32)
        if self._med_point_zero_invalid():
            point_features = point_features * valid_mask.unsqueeze(-1)
        return point_features

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

    def _transform_point_xyz(self, row: Dict[str, Any], xyz: torch.Tensor) -> torch.Tensor:
        coordinate_frame = self._med_point_coordinate_frame()
        if coordinate_frame == "camera":
            return xyz
        if coordinate_frame != "wrist_local":
            raise ValueError(f"Unsupported point-cloud coordinate_frame={coordinate_frame!r}")

        position = self._tensor_from_value(row[self._med_point_reference_position_field()]).view(1, 3).to(dtype=xyz.dtype)
        rotation = self._tensor_from_value(row[self._med_point_reference_rotation_field()]).view(3).to(dtype=xyz.dtype)
        rotation_matrix = _axis_angle_to_rotation_matrix(rotation).to(dtype=xyz.dtype)
        return (xyz - position) @ rotation_matrix

    def _load_video_frames(self, manifest: DatasetManifest, current_row: Dict[str, Any]) -> torch.Tensor:
        if not bool(self.video_config.enabled):
            return torch.zeros(0, 3, int(self.video_config.resize_height), int(self.video_config.resize_width), dtype=torch.float32)

        video_relpath = str(current_row[str(self.video_config.key)])
        video_path = Path(manifest.root) / video_relpath
        current_video_frame_index = int(current_row[f"{str(self.video_config.key)}_frame_index"])
        frame_indices = [
            current_video_frame_index - offset
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

    def _episode_parquet_path(self, manifest: DatasetManifest, episode_index: int) -> str:
        episode_relpath = manifest.metadata["episode_parquet_lookup"][int(episode_index)]
        return str((Path(manifest.root) / episode_relpath).resolve())

    def _feature_dim(self, manifest: DatasetManifest, field_name: str) -> int:
        spec = manifest.feature_specs[str(field_name)]
        shape = list(spec.get("shape", []))
        return _product(shape) if shape else 1

    def _point_count(self, manifest: DatasetManifest, field_name: str) -> int:
        spec = manifest.feature_specs[str(field_name)]
        shape = list(spec.get("shape", []))
        if len(shape) != 2 or int(shape[-1]) != 3:
            raise ValueError(
                f"Point-cloud xyz field {field_name} is expected to have shape [N, 3], got {shape}."
            )
        return int(shape[0])

    def _point_feature_dim(self, manifest: DatasetManifest) -> int:
        feature_dim = 3
        if self._med_point_include_value():
            value_field = self._med_point_value_field()
            point_count = self._point_count(manifest, self._med_point_xyz_field())
            value_dim = self._feature_dim(manifest, value_field)
            if value_dim % point_count != 0:
                raise ValueError(
                    f"Point-cloud value field {value_field} has flattened dim {value_dim}, "
                    f"which is not divisible by point_count={point_count}."
                )
            feature_dim += value_dim // point_count
        if self._med_point_include_valid_feature():
            feature_dim += 1
        return int(feature_dim)

    def _infer_med_source_shape(self, manifest: DatasetManifest, future_steps: int) -> List[int]:
        if self._med_kind() == "vector":
            tactile_dim = self._feature_dim(manifest, self._med_source_field())
            return [int(future_steps), int(tactile_dim)]
        point_count = self._point_count(manifest, self._med_point_xyz_field())
        feature_dim = self._point_feature_dim(manifest)
        return [int(future_steps), int(point_count), int(feature_dim)]

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
        return f"tactile.{self._med_side()}_tacglove_mask"

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

    def _validate_med_configuration(self) -> None:
        self._validate_tactile_observation_configuration()
        med_representation = self._med_kind()
        if med_representation == "vector":
            field_name = self._med_source_field()
            expected_dim = None
            for manifest in self.manifests:
                if field_name not in manifest.feature_specs:
                    capabilities = infer_manifest_med_capabilities(manifest)
                    raise ValueError(
                        f"Configured vector med field={field_name} is missing from dataset features for {manifest.dataset_id}. "
                        f"Available med capabilities: {capabilities if capabilities else ['none']}."
                    )
                field_dim = self._feature_dim(manifest, field_name)
                if expected_dim is None:
                    expected_dim = field_dim
                elif int(field_dim) != int(expected_dim):
                    raise ValueError(
                        f"Mixed datasets have inconsistent med_source_field dims for {field_name}: "
                        f"expected {expected_dim}, got {field_dim} in {manifest.dataset_id}."
                    )
            return
        if med_representation != "point_cloud":
            raise ValueError(f"Unsupported med_representation={med_representation}")
        xyz_field = self._med_point_xyz_field()
        valid_mask_field = self._med_point_valid_mask_field()
        value_field = self._med_point_value_field()
        coordinate_frame = self._med_point_coordinate_frame()
        reference_position_field = self._med_point_reference_position_field()
        reference_rotation_field = self._med_point_reference_rotation_field()
        expected_point_count = None
        expected_feature_dim = None
        for manifest in self.manifests:
            if xyz_field not in manifest.feature_specs:
                capabilities = infer_manifest_med_capabilities(manifest)
                raise ValueError(
                    f"Configured point-cloud xyz field={xyz_field} is missing from dataset features for {manifest.dataset_id}. "
                    f"Available med capabilities: {capabilities if capabilities else ['none']}."
                )
            if valid_mask_field not in manifest.feature_specs:
                raise ValueError(
                    f"Configured point-cloud valid_mask field={valid_mask_field} is missing from dataset features for {manifest.dataset_id}."
                )
            if self._med_point_include_value() and value_field not in manifest.feature_specs:
                raise ValueError(
                    f"Configured point-cloud value field={value_field} is missing from dataset features for {manifest.dataset_id}."
                )
            if coordinate_frame == "wrist_local":
                for reference_field in (reference_position_field, reference_rotation_field):
                    if reference_field not in manifest.feature_specs:
                        raise ValueError(
                            f"Configured wrist-local point-cloud reference field={reference_field} "
                            f"is missing from dataset features for {manifest.dataset_id}."
                        )
                    reference_dim = self._feature_dim(manifest, reference_field)
                    if int(reference_dim) != 3:
                        raise ValueError(
                            f"Wrist-local point-cloud reference field={reference_field} must be 3D, "
                            f"got flattened dim {reference_dim} in {manifest.dataset_id}."
                        )
            point_count = self._point_count(manifest, xyz_field)
            feature_dim = self._point_feature_dim(manifest)
            if expected_point_count is None:
                expected_point_count = point_count
                expected_feature_dim = feature_dim
                continue
            if int(point_count) != int(expected_point_count):
                raise ValueError(
                    f"Mixed datasets have inconsistent point counts: expected {expected_point_count}, "
                    f"got {point_count} in {manifest.dataset_id}."
                )
            if int(feature_dim) != int(expected_feature_dim):
                raise ValueError(
                    f"Mixed datasets have inconsistent point feature dims: expected {expected_feature_dim}, "
                    f"got {feature_dim} in {manifest.dataset_id}."
                )

    def _validate_tactile_observation_configuration(self) -> None:
        if not self._tactile_observation_enabled():
            return
        field_name = self._tactile_observation_field()
        mask_field = self._tactile_observation_mask_field()
        expected_dim = None
        for manifest in self.manifests:
            if field_name not in manifest.feature_specs:
                raise ValueError(f"Configured tactile observation field={field_name} is missing from {manifest.dataset_id}.")
            field_dim = self._feature_dim(manifest, field_name)
            if expected_dim is None:
                expected_dim = field_dim
            elif int(field_dim) != int(expected_dim):
                raise ValueError(
                    f"Mixed datasets have inconsistent tactile observation dims for {field_name}: "
                    f"expected {expected_dim}, got {field_dim} in {manifest.dataset_id}."
                )
            if self._tactile_observation_zero_invalid():
                if mask_field not in manifest.feature_specs:
                    raise ValueError(
                        f"Configured tactile observation mask field={mask_field} is missing from {manifest.dataset_id}."
                    )
                mask_dim = self._feature_dim(manifest, mask_field)
                if int(mask_dim) != int(field_dim):
                    raise ValueError(
                        f"Tactile observation mask dim mismatch in {manifest.dataset_id}: "
                        f"vector={field_dim}, mask={mask_dim}."
                    )

    def _format_temporal_target(self, sequence: torch.Tensor) -> torch.Tensor:
        if int(sequence.shape[0]) == 1:
            return sequence[0]
        return sequence


DeskTaskTacRawHandSequenceDataset = DeskTaskTacSequenceDataset
