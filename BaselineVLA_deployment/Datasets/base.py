from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

import torch


@dataclass
class DatasetManifest:
    dataset_id: str
    root: str
    source_format: str
    raw_data_type: str
    robot_type: str
    state_key: str
    action_key: str
    state_dim: int
    action_dim: int
    video_keys: List[str]
    episode_lengths: Dict[int, int]
    split_to_episode_indices: Dict[str, List[int]]
    feature_specs: Dict[str, Dict[str, Any]]
    feature_stats: Dict[str, Dict[str, Any]]
    task_lookup: Dict[int, str]
    episode_task_lookup: Dict[int, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleAddress:
    dataset_id: str
    split: str
    episode_index: int
    frame_index: int
    episode_length: int


@dataclass
class CanonicalSample:
    observations: Dict[str, Any]
    intermediates: Dict[str, Any]
    targets: Dict[str, Any]
    availability: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class PreparedBatch:
    model_inputs: Dict[str, torch.Tensor]
    targets: Dict[str, torch.Tensor]
    metadata: List[Dict[str, Any]]
    availability: Dict[str, Any]


def recursive_collate(values: Sequence[Any]) -> Any:
    first = values[0]
    if torch.is_tensor(first):
        return torch.stack(list(values), dim=0)
    if isinstance(first, Mapping):
        return {key: recursive_collate([value[key] for value in values]) for key in first.keys()}
    return list(values)


def collate_canonical_samples(samples: Sequence[CanonicalSample]) -> Dict[str, Any]:
    return {
        "observations": recursive_collate([sample.observations for sample in samples]),
        "intermediates": recursive_collate([sample.intermediates for sample in samples]),
        "targets": recursive_collate([sample.targets for sample in samples]),
        "availability": recursive_collate([sample.availability for sample in samples]),
        "metadata": [sample.metadata for sample in samples],
    }
