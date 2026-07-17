"""Deterministic v2.3 training samples over one episode's absolute rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.pose_chunk import (
    POSE_CHUNK_TAIL_DROP,
    PoseActionChunkContract,
)
from dataset_contracts.pose_frames import (
    POSE_FRAME_CONTRACT_METADATA_KEY,
    PoseFrameContract,
    PoseFrameContractError,
)
from dataset_contracts.v22 import PolicyIOLayout
from dataset_contracts.v23 import resolve_v23_dataset_contract

from .pose_action import PosePolicyViewError, pose_policy_window_from_rows


@dataclass(frozen=True)
class PoseActionChunkLocation:
    """One legal chunk start within a single episode."""

    episode_index: int
    start_index: int
    valid_action_count: int


def pose_action_chunk_start_indices(
    frame_count: int,
    contract: PoseActionChunkContract,
) -> tuple[int, ...]:
    """Return legal starts without ever crossing an episode boundary."""

    count = int(frame_count)
    if count < 0:
        raise PosePolicyViewError("frame_count must be non-negative.")
    if contract.tail_policy == POSE_CHUNK_TAIL_DROP:
        stop = max(0, count - contract.action_horizon + 1)
        return tuple(range(stop))
    return tuple(range(count))


@dataclass(frozen=True)
class PoseActionChunkSample:
    """One observation and its fixed-horizon model action target."""

    action: np.ndarray
    action_discrete: np.ndarray
    action_is_pad: np.ndarray
    observation_state: np.ndarray
    observation_state_discrete: np.ndarray
    last_action: np.ndarray
    last_action_discrete: np.ndarray
    actual_tcp_pose_by_side: Mapping[str, np.ndarray]
    layout_hash: str
    chunk_contract_hash: str
    pose_frame_contract_hash: str
    valid_action_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action = np.asarray(self.action, dtype=np.float32)
        action_discrete = np.asarray(self.action_discrete, dtype=np.int64)
        action_is_pad = np.asarray(self.action_is_pad, dtype=bool)
        observation_state = np.asarray(self.observation_state, dtype=np.float32)
        observation_state_discrete = np.asarray(
            self.observation_state_discrete, dtype=np.int64
        )
        last_action = np.asarray(self.last_action, dtype=np.float32)
        last_action_discrete = np.asarray(self.last_action_discrete, dtype=np.int64)
        if action.ndim != 2 or action_discrete.ndim != 2:
            raise PosePolicyViewError("Chunk action arrays must be 2D [horizon, dim].")
        horizon = action.shape[0]
        if action_discrete.shape[0] != horizon or action_is_pad.shape != (horizon,):
            raise PosePolicyViewError("Chunk action arrays and action_is_pad must share a horizon.")
        if observation_state.ndim != 1 or observation_state_discrete.ndim != 1:
            raise PosePolicyViewError("Chunk observation arrays must represent one frame.")
        if last_action.ndim != 1 or last_action_discrete.ndim != 1:
            raise PosePolicyViewError("Chunk last_action arrays must represent one frame.")
        valid_count = int(self.valid_action_count)
        if not 1 <= valid_count <= horizon:
            raise PosePolicyViewError("valid_action_count must be within the action horizon.")
        expected_mask = np.arange(horizon) >= valid_count
        if not np.array_equal(action_is_pad, expected_mask):
            raise PosePolicyViewError(
                "action_is_pad must be a contiguous false-prefix/true-tail mask."
            )
        anchors = {
            str(side): np.asarray(value, dtype=np.float32).copy()
            for side, value in dict(self.actual_tcp_pose_by_side).items()
        }
        for side, pose in anchors.items():
            if pose.shape != (7,) or not np.all(np.isfinite(pose)):
                raise PosePolicyViewError(
                    f"actual_tcp_pose_by_side[{side!r}] must be one finite pose7."
                )
        object.__setattr__(self, "action", action.copy())
        object.__setattr__(self, "action_discrete", action_discrete.copy())
        object.__setattr__(self, "action_is_pad", action_is_pad.copy())
        object.__setattr__(self, "observation_state", observation_state.copy())
        object.__setattr__(
            self, "observation_state_discrete", observation_state_discrete.copy()
        )
        object.__setattr__(self, "last_action", last_action.copy())
        object.__setattr__(self, "last_action_discrete", last_action_discrete.copy())
        object.__setattr__(self, "actual_tcp_pose_by_side", anchors)
        object.__setattr__(self, "layout_hash", str(self.layout_hash))
        object.__setattr__(self, "chunk_contract_hash", str(self.chunk_contract_hash))
        object.__setattr__(
            self, "pose_frame_contract_hash", str(self.pose_frame_contract_hash)
        )
        object.__setattr__(self, "valid_action_count", valid_count)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def actions_for_normalization(self) -> np.ndarray:
        """Return only real actions; padding must never affect statistics."""

        return self.action[~self.action_is_pad].copy()

    def model_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.copy(),
            "action.discrete": self.action_discrete.copy(),
            "action_is_pad": self.action_is_pad.copy(),
            "observation.state": self.observation_state.copy(),
            "observation.state.discrete": self.observation_state_discrete.copy(),
            "observation.last_action": self.last_action.copy(),
            "observation.last_action.discrete": self.last_action_discrete.copy(),
            "policy_io_layout_hash": self.layout_hash,
            "pose_action_chunk_contract_hash": self.chunk_contract_hash,
            "pose_frame_contract_hash": self.pose_frame_contract_hash,
            "metadata": dict(self.metadata),
        }


def _pad_last(array: np.ndarray, *, horizon: int) -> np.ndarray:
    if array.shape[0] == horizon:
        return array.copy()
    padding = np.repeat(array[-1:], horizon - array.shape[0], axis=0)
    return np.concatenate((array, padding), axis=0)


def pose_action_chunk_from_episode_rows(
    episode_rows: Sequence[Mapping[str, Any]],
    layout: PolicyIOLayout,
    *,
    pose_frame_contract: PoseFrameContract,
    chunk_contract: PoseActionChunkContract,
    start_index: int,
    active_sides: Sequence[str] = ("left", "right"),
) -> PoseActionChunkSample:
    """Build one sample without crossing the supplied episode's boundary."""

    if not episode_rows:
        raise PosePolicyViewError("Cannot build an action chunk from an empty episode.")
    start = int(start_index)
    if start < 0 or start >= len(episode_rows):
        raise PosePolicyViewError(
            f"start_index={start} is outside episode frame_count={len(episode_rows)}."
        )
    stop = min(len(episode_rows), start + chunk_contract.action_horizon)
    real_rows = episode_rows[start:stop]
    valid_count = len(real_rows)
    if (
        chunk_contract.tail_policy == POSE_CHUNK_TAIL_DROP
        and valid_count < chunk_contract.action_horizon
    ):
        raise PosePolicyViewError(
            "Incomplete episode tail is not legal for tail_policy='drop': "
            f"start_index={start}, valid={valid_count}, horizon={chunk_contract.action_horizon}."
        )

    window = pose_policy_window_from_rows(
        real_rows,
        layout,
        pose_frame_contract=pose_frame_contract,
        active_sides=active_sides,
        action_pose_mode=chunk_contract.action_pose_mode,
        last_action_pose_mode=chunk_contract.last_action_pose_mode,
    )
    horizon = chunk_contract.action_horizon
    action = _pad_last(window.action, horizon=horizon)
    action_discrete = _pad_last(window.action_discrete, horizon=horizon)
    action_is_pad = np.arange(horizon) >= valid_count
    return PoseActionChunkSample(
        action=action,
        action_discrete=action_discrete,
        action_is_pad=action_is_pad,
        observation_state=window.observation_state[0],
        observation_state_discrete=window.observation_state_discrete[0],
        last_action=window.last_action[0],
        last_action_discrete=window.last_action_discrete[0],
        actual_tcp_pose_by_side={
            side: values[0] for side, values in window.actual_tcp_pose_by_side.items()
        },
        layout_hash=window.layout_hash,
        chunk_contract_hash=chunk_contract.contract_hash,
        pose_frame_contract_hash=pose_frame_contract.contract_hash,
        valid_action_count=valid_count,
        metadata={
            **dict(window.metadata),
            "start_index": start,
            "action_horizon": horizon,
            "valid_action_count": valid_count,
            "tail_policy": chunk_contract.tail_policy,
            "normalization_excludes_padding": True,
        },
    )


def pose_action_chunk_from_dataset_metadata(
    episode_rows: Sequence[Mapping[str, Any]],
    dataset_metadata: Mapping[str, Any],
    *,
    chunk_contract: PoseActionChunkContract,
    start_index: int,
    episode_metadata: Mapping[str, Any] | None = None,
    active_sides: Sequence[str] | None = None,
) -> PoseActionChunkSample:
    """Resolve v2.2/v2.3 metadata and build one deterministic sample."""

    resolved = resolve_v23_dataset_contract(dataset_metadata, active_sides=active_sides)
    episode = {} if episode_metadata is None else dict(episode_metadata)
    episode_contract_payload = episode.get(POSE_FRAME_CONTRACT_METADATA_KEY)
    if episode_contract_payload is not None:
        if not isinstance(episode_contract_payload, Mapping):
            raise PosePolicyViewError("episode pose_frame_contract must be a mapping.")
        try:
            episode_contract = PoseFrameContract.from_dict(episode_contract_payload)
        except PoseFrameContractError as exc:
            raise PosePolicyViewError(f"episode pose_frame_contract is invalid: {exc}") from exc
        if episode_contract.to_dict() != resolved.pose_frames.to_dict():
            raise PosePolicyViewError(
                "episode pose_frame_contract differs from the dataset-wide contract."
            )
    sides = active_sides or resolved.pose_frames.source.active_sides
    return pose_action_chunk_from_episode_rows(
        episode_rows,
        resolved.policy_io_layout,
        pose_frame_contract=resolved.pose_frames,
        chunk_contract=chunk_contract,
        start_index=start_index,
        active_sides=sides,
    )


__all__ = [
    "PoseActionChunkLocation",
    "PoseActionChunkSample",
    "pose_action_chunk_from_dataset_metadata",
    "pose_action_chunk_from_episode_rows",
    "pose_action_chunk_start_indices",
]
