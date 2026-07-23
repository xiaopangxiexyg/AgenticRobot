"""Strict canonical desk/tool-TCP action adapter for MuJoCo replay.

The historical public name remains ``umi_dataset`` for compatibility.  The
adapter also accepts canonical real-teleop 26D actions; qpos is intentionally
not a fallback because Cartesian actions are the production replay contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from dataset_contracts.pose_frames import PoseFrameContract
from robot_motion.position_control import TimestampedMotionTarget
from robot_motion.transforms import parse_pose_matrix

from .dataset import ResolvedEpisode, read_parquet_rows, resolve_episode


@dataclass(frozen=True)
class UmiReplayFrame:
    source_index: int
    frame_index: int
    timestamp: float
    relative_timestamp: float
    pose_desk_by_side: Mapping[str, np.ndarray]
    gripper_close_by_side: Mapping[str, float]
    tactile_manifest_by_side: Mapping[str, Path | None]
    tactile_frame_index_by_side: Mapping[str, int | None]


@dataclass(frozen=True)
class UmiReplayEpisode:
    episode: ResolvedEpisode
    pose_frame_contract: PoseFrameContract
    pose_frame_contract_hash: str
    policy_io_layout_hash: str
    nominal_hz: float
    action_dim: int
    frames: tuple[UmiReplayFrame, ...]
    tactile_storage_by_side: Mapping[str, str | None]

    def motion_targets(self) -> list[TimestampedMotionTarget]:
        return [
            TimestampedMotionTarget(
                timestamp=frame.relative_timestamp,
                available_timestamp=frame.relative_timestamp,
                source_index=frame.source_index,
                pose_by_side=frame.pose_desk_by_side,
                metadata={"frame_index": frame.frame_index, "pose_frame": "desk"},
            )
            for frame in self.frames
        ]

    def interpolated_gripper_targets(
        self,
        *,
        left_source_index: int,
        right_source_index: int,
        alpha: float,
    ) -> dict[str, float]:
        left = self.frames[left_source_index].gripper_close_by_side
        right = self.frames[right_source_index].gripper_close_by_side
        blend = float(alpha)
        return {
            side: float((1.0 - blend) * left[side] + blend * right[side])
            for side in ("left", "right")
        }


def _read_info(root: Path) -> dict[str, object]:
    path = root / "meta/info.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _validate_info(
    info: Mapping[str, object],
) -> tuple[PoseFrameContract, str, str, float]:
    raw_contract = info.get("pose_frame_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("UMI dataset is missing pose_frame_contract")
    pose_contract = PoseFrameContract.from_dict(raw_contract)
    if pose_contract.pose_storage_frame != "desk_absolute":
        raise ValueError(
            "UMI replay requires pose_storage_frame=desk_absolute; refusing to apply "
            "raw_to_desk_transform implicitly"
        )
    if pose_contract.canonical_absolute_frame != "desk":
        raise ValueError("UMI replay requires canonical_absolute_frame=desk")
    if tuple(pose_contract.source.active_sides) != ("left", "right"):
        raise ValueError("UMI replay requires a left/right pose-frame contract")
    if str(info.get("arm_pose_action_mode", "")) != "absolute":
        raise ValueError("UMI replay requires absolute arm pose actions")
    arm_specs = info.get("arm_action_specs_by_side")
    if not isinstance(arm_specs, Mapping):
        raise ValueError("UMI dataset is missing arm_action_specs_by_side")
    for side in ("left", "right"):
        spec = arm_specs.get(side)
        if not isinstance(spec, Mapping):
            raise ValueError(f"UMI dataset is missing {side} arm action spec")
        expected = {
            "controlled_frame": "tool_tcp",
            "reference_frame": "desk",
            "representation": "xyz_quat",
            "dof": 7,
        }
        for key, value in expected.items():
            if spec.get(key) != value:
                raise ValueError(
                    f"{side} arm action spec {key}={spec.get(key)!r}, expected {value!r}"
                )
    layout_hash = str(info.get("policy_io_layout_hash", "")).strip()
    if not layout_hash:
        raise ValueError("UMI dataset is missing policy_io_layout_hash")
    nominal_hz = float(info.get("nominal_fps", 0.0))
    if not np.isfinite(nominal_hz) or nominal_hz <= 0.0:
        raise ValueError("UMI nominal_fps must be finite and positive")
    pose_hash = str(info.get("pose_frame_contract_hash", "")).strip()
    if not pose_hash:
        raise ValueError("UMI dataset is missing pose_frame_contract_hash")
    return pose_contract, pose_hash, layout_hash, nominal_hz


def _manifest_storage(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"tactile manifest {path} must contain an object")
    storage = payload.get("storage")
    compression = payload.get("compression")
    if isinstance(storage, Mapping):
        compression = storage.get("compression", compression)
        storage = storage.get("format") or storage.get("storage")
    streams = payload.get("streams")
    if isinstance(streams, Mapping) and streams:
        stream_values = [value for value in streams.values() if isinstance(value, Mapping)]
        encodings = {str(value.get("encoding", "")) for value in stream_values}
        compressions = {str(value.get("compression", "")) for value in stream_values}
        encodings.discard("")
        compressions.discard("")
        if len(encodings) == 1:
            storage = next(iter(encodings))
        if len(compressions) == 1:
            compression = next(iter(compressions))
    values = [str(value) for value in (storage, compression) if value]
    return "/".join(values) or "unknown"


def _gripper_close_from_action(
    action: np.ndarray,
    row: Mapping[str, object],
    *,
    side: str,
    previous: float | None,
) -> float:
    """Resolve the DMTAC scalar without treating arm qpos as an action source."""

    if action.shape == (16,):
        value = action[14 if side == "left" else 15]
    elif action.shape == (26,):
        # Canonical 26D layout: two 7D arm poses followed by left/right q6.
        value = action[14 if side == "left" else 20]
    else:  # guarded by the caller; keep this helper fail-closed in isolation.
        raise ValueError(f"unsupported canonical action width {action.size}")
    if np.isfinite(value):
        return float(value)

    # A held/frozen teleop frame may leave the main hand slot NaN while the
    # dataset preserves the backend's last accepted target in this explicit
    # replay column.  This is still an action target, not observation/qpos.
    for column in (
        f"action.replay.{side}_hand_joint_target",
        f"action.{side}_hand.joint_target",
    ):
        raw = row.get(column)
        if raw is None:
            continue
        values = np.asarray(raw, dtype=float).reshape(-1)
        if values.size and np.isfinite(values[0]):
            return float(values[0])
    if previous is not None:
        return float(previous)
    raise ValueError(f"first canonical action has no finite {side} gripper target")


def load_umi_replay_episode(
    dataset_root: str | Path,
    episode: str | int,
    *,
    start_frame: int = 0,
    max_frames: int | None = None,
) -> UmiReplayEpisode:
    resolved = resolve_episode(dataset_root, episode)
    info = _read_info(resolved.dataset_root)
    pose_contract, pose_hash, layout_hash, nominal_hz = _validate_info(info)
    rows = read_parquet_rows(resolved.parquet_path)
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    selected = rows[start_frame:]
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive when set")
        selected = selected[:max_frames]
    if not selected:
        raise ValueError("selected UMI replay range contains no frames")

    first_timestamp = float(selected[0]["timestamp"])
    previous_timestamp = -np.inf
    frames: list[UmiReplayFrame] = []
    action_dim: int | None = None
    previous_gripper: dict[str, float | None] = {"left": None, "right": None}
    tactile_storage: dict[str, str | None] = {"left": None, "right": None}
    manifest_storage_cache: dict[Path, str] = {}
    for source_index, row in enumerate(selected):
        action = np.asarray(row.get("action"), dtype=float).reshape(-1)
        if action.shape not in {(16,), (26,)}:
            raise ValueError(
                f"frame {row.get('frame_index')} canonical action must contain 16 or 26 values"
            )
        if action_dim is None:
            action_dim = int(action.size)
        elif action.size != action_dim:
            raise ValueError("canonical action width changes within episode")
        # Only Cartesian arm poses are mandatory here.  Hand padding may be NaN
        # in real teleop datasets and is resolved separately below.
        if not np.all(np.isfinite(action[:14])):
            raise ValueError(
                f"frame {row.get('frame_index')} canonical arm poses must be finite"
            )
        row_storage = str(row.get("dexteleop.pose_storage_frame", ""))
        if row_storage and row_storage != "desk_absolute":
            raise ValueError(f"frame {row.get('frame_index')} is not desk_absolute")
        row_hash = str(row.get("policy_io_layout_hash", ""))
        if row_hash and row_hash != layout_hash:
            raise ValueError(f"frame {row.get('frame_index')} layout hash mismatch")
        row_pose_hash = str(row.get("dexteleop.pose_frame_contract_hash", ""))
        if row_pose_hash and row_pose_hash != pose_hash:
            raise ValueError(f"frame {row.get('frame_index')} pose-frame hash mismatch")
        timestamp = float(row["timestamp"])
        if not np.isfinite(timestamp) or timestamp <= previous_timestamp:
            raise ValueError("UMI timestamps must be finite and strictly increasing")
        previous_timestamp = timestamp
        poses = {
            "left": parse_pose_matrix(action[0:7]),
            "right": parse_pose_matrix(action[7:14]),
        }
        manifests: dict[str, Path | None] = {}
        tactile_indices: dict[str, int | None] = {}
        for side in ("left", "right"):
            raw_path = str(row.get(f"observation.hand.{side}.tactile_path", "")).strip()
            manifest = None if not raw_path else resolved.dataset_root / raw_path
            if manifest is not None:
                if not manifest.is_file():
                    raise FileNotFoundError(f"missing tactile manifest: {manifest}")
                current_storage = manifest_storage_cache.get(manifest)
                if current_storage is None:
                    current_storage = _manifest_storage(manifest)
                    manifest_storage_cache[manifest] = current_storage
                if tactile_storage[side] not in (None, current_storage):
                    raise ValueError(f"{side} tactile manifest storage changes within episode")
                tactile_storage[side] = current_storage
            manifests[side] = manifest
            value = row.get(f"observation.hand.{side}.tactile_frame_index")
            tactile_indices[side] = None if value is None else int(value)
        gripper_close = {
            side: _gripper_close_from_action(
                action,
                row,
                side=side,
                previous=previous_gripper[side],
            )
            for side in ("left", "right")
        }
        previous_gripper.update(gripper_close)
        frames.append(
            UmiReplayFrame(
                source_index=source_index,
                frame_index=int(row.get("frame_index", source_index)),
                timestamp=timestamp,
                relative_timestamp=timestamp - first_timestamp,
                pose_desk_by_side=poses,
                gripper_close_by_side=gripper_close,
                tactile_manifest_by_side=manifests,
                tactile_frame_index_by_side=tactile_indices,
            )
        )
    assert action_dim is not None
    return UmiReplayEpisode(
        episode=resolved,
        pose_frame_contract=pose_contract,
        pose_frame_contract_hash=pose_hash,
        policy_io_layout_hash=layout_hash,
        nominal_hz=nominal_hz,
        action_dim=action_dim,
        frames=tuple(frames),
        tactile_storage_by_side=tactile_storage,
    )


__all__ = ["UmiReplayEpisode", "UmiReplayFrame", "load_umi_replay_episode"]
