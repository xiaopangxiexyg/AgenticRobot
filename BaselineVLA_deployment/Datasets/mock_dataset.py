from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from .base import CanonicalSample, DatasetManifest, SampleAddress
from .config_utils import get_positive_limit
from .task_utils import build_task_name_to_id


def _stable_seed(*parts: Any) -> int:
    joined = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**31)


def _safe_vector(values: List[float], dim: int, fallback_std: float = 1.0) -> torch.Tensor:
    if not values:
        if fallback_std == 1.0:
            return torch.zeros(dim, dtype=torch.float32)
        return torch.full((dim,), float(fallback_std), dtype=torch.float32)
    vector = torch.tensor(values[:dim], dtype=torch.float32)
    if vector.numel() < dim:
        pad_value = 0.0 if fallback_std == 1.0 else float(fallback_std)
        vector = torch.nn.functional.pad(vector, (0, dim - vector.numel()), value=pad_value)
    return vector


def _align_tensor_dim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    if tensor.numel() == target_dim:
        return tensor
    if tensor.numel() > target_dim:
        return tensor[:target_dim]
    return torch.nn.functional.pad(tensor, (0, target_dim - tensor.numel()))


class MockRawSequenceDataset(Dataset):
    def __init__(self, manifests: List[DatasetManifest], split: str, config: Any) -> None:
        self.manifests = manifests
        self.split = split
        self.config = config
        self.manifest_by_id = {manifest.dataset_id: manifest for manifest in self.manifests}
        self.task_name_to_id = build_task_name_to_id(self.manifests)
        self.addresses = self._build_addresses()
        if not self.addresses:
            raise RuntimeError(f"No samples discovered for split={split}. Check filters or data roots.")

    def __len__(self) -> int:
        return len(self.addresses)

    def __getitem__(self, index: int) -> CanonicalSample:
        address = self.addresses[index]
        manifest = self.manifest_by_id[address.dataset_id]
        state = self._build_state(manifest, address)
        action = self._build_action(manifest, address, state)
        vision_tokens, vision_mask = self._build_vision_tokens(manifest)
        task_text = manifest.episode_task_lookup.get(
            address.episode_index,
            manifest.task_lookup.get(0, "unknown_task"),
        )
        task_id = int(self.task_name_to_id.get(task_text, self.task_name_to_id["unknown_task"]))
        return CanonicalSample(
            observations={
                "state": state,
                "vision_tokens": vision_tokens,
                "vision_mask": vision_mask,
                "task_text": task_text,
                "task_id": torch.tensor(task_id, dtype=torch.long),
            },
            intermediates={},
            targets={
                "action": action,
            },
            availability={
                "state": torch.tensor(1.0, dtype=torch.float32),
                "action": torch.tensor(1.0, dtype=torch.float32),
                "vision": torch.tensor(float(len(manifest.video_keys) > 0), dtype=torch.float32),
                "task": torch.tensor(1.0, dtype=torch.float32),
            },
            metadata={
                "dataset_id": manifest.dataset_id,
                "dataset_root": manifest.root,
                "raw_data_type": "Mock_RawSequence",
                "source_format": manifest.source_format,
                "source_data_type": manifest.raw_data_type,
                "robot_type": manifest.robot_type,
                "split": address.split,
                "episode_index": address.episode_index,
                "frame_index": address.frame_index,
                "episode_length": address.episode_length,
                "state_dim": manifest.state_dim,
                "action_dim": manifest.action_dim,
                "task_text": task_text,
                "task_id": task_id,
                "video_keys": list(manifest.video_keys),
            },
        )

    @property
    def summary(self) -> Dict[str, Any]:
        first_manifest = self.manifests[0]
        return {
            "dataset_type": "Mock_RawSequence",
            "manifest_loader": str(self.config.manifest_loader),
            "split": self.split,
            "sample_count": len(self.addresses),
            "dataset_count": len(self.manifests),
            "state_dim": first_manifest.state_dim,
            "action_dim": first_manifest.action_dim,
            "action_chunk": 1,
            "video_token_slots": int(self.config.sampling.max_video_tokens),
            "task_count": len(self.task_name_to_id),
            "task_names": [
                task_name
                for task_name, _ in sorted(self.task_name_to_id.items(), key=lambda item: item[1])
            ],
            "dataset_ids": [manifest.dataset_id for manifest in self.manifests],
        }

    def _build_addresses(self) -> List[SampleAddress]:
        addresses: List[SampleAddress] = []
        max_frames = get_positive_limit(self.config, "max_frames_per_episode")
        max_total = get_positive_limit(self.config, "max_total_samples")

        for manifest in self.manifests:
            episode_indices = self._select_episode_indices(manifest)
            for episode_index in episode_indices:
                episode_length = int(manifest.episode_lengths.get(int(episode_index), 0))
                if episode_length <= 0:
                    continue
                for frame_index in self._select_frame_indices(episode_length, max_frames):
                    addresses.append(
                        SampleAddress(
                            dataset_id=manifest.dataset_id,
                            split=self.split,
                            episode_index=int(episode_index),
                            frame_index=int(frame_index),
                            episode_length=int(episode_length),
                        )
                    )
                    if max_total is not None and len(addresses) >= int(max_total):
                        return addresses
        return addresses

    def _select_episode_indices(self, manifest: DatasetManifest) -> List[int]:
        episode_indices = list(manifest.split_to_episode_indices.get(self.split, []))
        max_episodes = get_positive_limit(self.config, "max_episodes_per_dataset")
        if max_episodes is None or len(episode_indices) <= int(max_episodes):
            return episode_indices

        generator_seed = _stable_seed(
            int(getattr(self.config.sampling, "base_seed", 0)),
            manifest.dataset_id,
            self.split,
            "episode_subset",
        )
        rng = random.Random(generator_seed)
        sampled = list(episode_indices)
        rng.shuffle(sampled)
        return sampled[: int(max_episodes)]

    def _select_frame_indices(self, episode_length: int, max_frames: int | None) -> List[int]:
        frame_stride = max(1, int(self.config.sampling.frame_stride))
        candidate_indices = list(range(0, int(episode_length), frame_stride))
        if max_frames is None or len(candidate_indices) <= int(max_frames):
            return candidate_indices
        if int(max_frames) <= 1:
            return [candidate_indices[0]]
        step = (len(candidate_indices) - 1) / float(max_frames - 1)
        selected = [candidate_indices[round(position * step)] for position in range(max_frames)]
        return sorted(set(selected))

    def _make_generator(self, manifest: DatasetManifest, address: SampleAddress, suffix: str) -> torch.Generator:
        seed = _stable_seed(
            int(self.config.sampling.base_seed),
            manifest.dataset_id,
            address.episode_index,
            address.frame_index,
            suffix,
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def _build_state(self, manifest: DatasetManifest, address: SampleAddress) -> torch.Tensor:
        generator = self._make_generator(manifest, address, "state")
        mean = _safe_vector(list(manifest.metadata.get("state_mean", [])), manifest.state_dim)
        std = _safe_vector(list(manifest.metadata.get("state_std", [])), manifest.state_dim, fallback_std=0.25).clamp_min(1e-3)
        state = mean + float(self.config.sampling.state_scale) * std * torch.randn(manifest.state_dim, generator=generator)
        phase = (2.0 * math.pi * float(address.frame_index)) / max(1.0, float(address.episode_length))
        phase_vector = torch.linspace(0.0, 1.0, steps=manifest.state_dim, dtype=torch.float32)
        state = state + float(self.config.sampling.temporal_signal_scale) * torch.sin(phase + phase_vector)
        return state.to(dtype=torch.float32)

    def _build_action(self, manifest: DatasetManifest, address: SampleAddress, state: torch.Tensor) -> torch.Tensor:
        generator = self._make_generator(manifest, address, "action")
        mean = _safe_vector(list(manifest.metadata.get("action_mean", [])), manifest.action_dim)
        std = _safe_vector(list(manifest.metadata.get("action_std", [])), manifest.action_dim, fallback_std=0.25).clamp_min(1e-3)
        residual = float(self.config.sampling.action_scale) * std * torch.randn(manifest.action_dim, generator=generator)
        state_aligned = _align_tensor_dim(state, manifest.action_dim)
        mixed = float(self.config.sampling.action_state_mix) * state_aligned + (1.0 - float(self.config.sampling.action_state_mix)) * mean
        return (mixed + 0.1 * residual).to(dtype=torch.float32)

    def _build_vision_tokens(self, manifest: DatasetManifest) -> tuple[torch.Tensor, torch.Tensor]:
        token_slots = int(self.config.sampling.max_video_tokens)
        token_dim = 4
        tokens = torch.zeros(token_slots, token_dim, dtype=torch.float32)
        mask = torch.zeros(token_slots, dtype=torch.float32)
        for index, video_key in enumerate(manifest.video_keys[:token_slots]):
            mask[index] = 1.0
            stats = manifest.feature_stats.get(video_key, {})
            mean = _safe_vector(list(_flatten_video_stats(stats.get("mean", []))), 3)
            tokens[index, :3] = mean[:3]
            tokens[index, 3] = float(index) / max(1.0, float(token_slots - 1))
        return tokens, mask


def _flatten_video_stats(value: Any) -> List[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        flattened: List[float] = []
        for item in value:
            flattened.extend(_flatten_video_stats(item))
        return flattened
    return []
