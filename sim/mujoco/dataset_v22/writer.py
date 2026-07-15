"""Native v2.2 MuJoCo dataset writer.

The writer owns its metadata, media, and parquet output instead of adapting
the legacy 26D/38D replay writer.  The only model-facing low-dimensional
representation it writes is ``PolicyIOLayout`` v2.2.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.parquet import write_rows_to_parquet
from dataset_contracts.v22 import DEXTELEOP_DATASET_SCHEMA_V22, PolicyIOLayout, bind_policy_io_layout

from ..scene import MujocoScene
from ..transforms import pose_to_xyz_quat_xyzw
from ..replay.trajectory import GripperReplaySample, QposSample
from .layout import load_rm65_bimanual_dmtac_layout


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=True), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), ensure_ascii=True) + "\n")


def _episode_id(index: int) -> str:
    return f"episode_{int(index):06d}"


def _next_episode_index(root_dir: Path) -> int:
    indices = [
        int(record["episode_index"])
        for record in _read_jsonl(root_dir / "meta" / "episodes.jsonl")
        if str(record.get("episode_index", "")).strip()
    ]
    return max(indices, default=-1) + 1


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("MuJoCo v2.2 RGB dataset export requires opencv-python.") from exc
    return cv2


@dataclass(frozen=True)
class MujocoV22DatasetWriterConfig:
    root_dir: Path
    policy_io_layout_path: Path | None = None
    dataset_name: str = ""
    task_id: str = "mujoco_real2sim_replay_v22"
    task_description: str = "MuJoCo real2sim replay (native v2.2)"
    nominal_fps: float = 30.0
    episode_index: int | None = None
    video_codec: str = "mp4v"
    parquet_compression: str = "zstd"
    source_dataset_root: Path | None = None
    source_episode_id: str = ""
    source_parquet_path: Path | None = None
    urdf_path: Path | None = None
    mode: str = "dual"
    qpos_source: str = "auto"
    gripper_source: str = "auto"
    control_hz: float = 150.0
    reference_hz: float = 45.0
    scene_contract_path: Path | None = None
    gripper_name_by_side: Mapping[str, str] = field(default_factory=dict)


class MujocoV22DatasetWriter:
    """Write a single MuJoCo episode with real-compatible v2.2 policy fields."""

    def __init__(self, scene: MujocoScene, config: MujocoV22DatasetWriterConfig, *, camera_names: Sequence[str]):
        self.scene = scene
        self.config = config
        self.layout: PolicyIOLayout = load_rm65_bimanual_dmtac_layout(config.policy_io_layout_path)
        self.root_dir = Path(config.root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.camera_names = tuple(str(name) for name in camera_names)
        if not self.camera_names:
            raise ValueError("A v2.2 MuJoCo dataset requires at least one rendered RGB camera.")
        if set(self.scene.arm_specs) != {"left", "right"}:
            raise ValueError("RM65 bimanual v2.2 export requires left and right MuJoCo arm specs.")
        if set(self.config.gripper_name_by_side) != {"left", "right"}:
            raise ValueError(
                "RM65 bimanual v2.2 export requires an explicit left/right MuJoCo gripper-name mapping."
            )
        self.episode_index = int(config.episode_index) if config.episode_index is not None else _next_episode_index(self.root_dir)
        self.episode_id = _episode_id(self.episode_index)
        self.chunk_id = f"chunk-{self.episode_index // 1000:03d}"
        self.parquet_relative_path = str(Path("data") / self.chunk_id / f"{self.episode_id}.parquet")
        self.video_relative_paths = {
            name: str(Path("videos") / f"observation.images.{name}" / self.chunk_id / f"{self.episode_id}.mp4")
            for name in self.camera_names
        }
        self.rows: list[dict[str, Any]] = []
        self._video_writers: dict[str, Any] = {}
        self._video_sizes: dict[str, tuple[int, int]] = {}
        self._previous_action: np.ndarray | None = None
        self._created_at = datetime.now().isoformat(timespec="seconds")
        self._fallback_timestamp_origin = time.time()
        self._closed = False
        self._write_initial_metadata()

    def _provenance(self) -> dict[str, Any]:
        return {
            "schema": "dexteleop.sim_provenance.v1",
            "producer": "sim.mujoco.dataset_v22",
            "created_at": self._created_at,
            "simulator": "mujoco",
            "source_dataset_root": self.config.source_dataset_root,
            "source_episode_id": self.config.source_episode_id,
            "source_parquet_path": self.config.source_parquet_path,
            "urdf_path": self.config.urdf_path,
            "mjcf_path": self.scene.conversion.mjcf_path,
            "scene_contract_path": self.config.scene_contract_path,
            "mode": self.config.mode,
            "qpos_source": self.config.qpos_source,
            "gripper_source": self.config.gripper_source,
            "control_hz": float(self.config.control_hz),
            "reference_hz": float(self.config.reference_hz),
            "camera_metadata": self.scene.camera_metadata(),
            "gripper_semantics": {"units": "normalized_0_1", "open": 0.0, "close": 1.0},
            "gripper_name_by_side": dict(self.config.gripper_name_by_side),
        }

    def _write_initial_metadata(self) -> None:
        meta = self.root_dir / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        tasks_path = meta / "tasks.jsonl"
        if not any(str(row.get("task_id")) == self.config.task_id for row in _read_jsonl(tasks_path)):
            _append_jsonl(tasks_path, {"task_index": 0, "task_id": self.config.task_id, "task": self.config.task_description})
        _write_json(meta / "sim_provenance.json", self._provenance())
        _write_json(
            meta / "modality_validity.json",
            {
                "schema_version": "dexteleop.modality_validity.v1",
                "modalities": {
                    "force_torque": {"available": False, "reason": "MuJoCo replay export has no calibrated real force/torque sensor stream."},
                    "tactile": {"available": False, "reason": "MuJoCo replay export has no tactile sensor model."},
                    "rgb": {"available": True, "camera_keys": list(self.camera_names)},
                },
                "training": {"valid": True, "allowed_video_camera_keys": list(self.camera_names)},
            },
        )
        self._write_info()

    def _write_info(self) -> None:
        records = _read_jsonl(self.root_dir / "meta" / "episodes.jsonl")
        info: dict[str, Any] = {
            "format": "lerobot_v2.1_style",
            "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V22,
            "dataset_name": self.config.dataset_name or self.root_dir.name,
            "nominal_fps": float(self.config.nominal_fps),
            "chunk_size": 1000,
            "episode_count": len(records),
            "frame_count": sum(int(record.get("frame_count") or 0) for record in records),
            "write_parquet": True,
            "write_video": True,
            "task_id": self.config.task_id,
            "task_description": self.config.task_description,
            "features": self.layout.feature_metadata(),
            "sim_provenance": self._provenance(),
        }
        info.update(bind_policy_io_layout(None, self.layout))
        _write_json(self.root_dir / "meta" / "info.json", info)

    def _open_video_writer(self, camera_name: str, image: np.ndarray) -> None:
        cv2 = _require_cv2()
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"RGB frame for {camera_name!r} must have shape (H, W, 3), got {rgb.shape}.")
        height, width = rgb.shape[:2]
        path = self.root_dir / self.video_relative_paths[camera_name]
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*self.config.video_codec), float(self.config.nominal_fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open v2.2 sim video writer: {path}")
        self._video_writers[camera_name] = writer
        self._video_sizes[camera_name] = (width, height)

    def _write_video_frame(self, camera_name: str, image: np.ndarray) -> None:
        rgb = np.asarray(image, dtype=np.uint8)
        if camera_name not in self._video_writers:
            self._open_video_writer(camera_name, rgb)
        height, width = rgb.shape[:2]
        if self._video_sizes[camera_name] != (width, height):
            raise ValueError(f"Camera {camera_name!r} frame size changed during an episode.")
        self._video_writers[camera_name].write(rgb[..., ::-1])

    def _arm_pose_by_side(self) -> dict[str, np.ndarray]:
        poses: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            controlled_frame = self.layout.arm_controlled_frame_by_side[side]
            if controlled_frame != "tool_tcp":
                raise ValueError(
                    "RM65 bimanual v2.2 export only supports the canonical tool_tcp frame; "
                    f"{side} declares {controlled_frame!r}."
                )
            # The canonical RM65 URDF maps tool_tcp to the fixed hand-base link.
            # Do not silently substitute arm_link6: that would change the policy
            # coordinate frame relative to the real collector and planner.
            body_name = f"{side}_hand_base"
            base_body_name = f"{side}_arm_base_link"
            try:
                world_from_tool = self.scene.body_pose(body_name)
                world_from_base = self.scene.body_pose(base_body_name)
            except KeyError as exc:
                raise KeyError(
                    f"MuJoCo scene is missing {body_name!r} or {base_body_name!r}, required to export the "
                    f"real-aligned {side} tool_tcp pose in the {side}_arm_base frame."
                ) from exc
            poses[side] = pose_to_xyz_quat_xyzw(np.linalg.inv(world_from_base) @ world_from_tool)
        return poses

    @staticmethod
    def _component_fields(prefix: str, packed, layout: PolicyIOLayout, *, role: str) -> dict[str, list[float] | list[int]]:
        suffix = "target" if role == "action" else "state"
        fields: dict[str, list[float] | list[int]] = {}
        for value_type, vector in (("continuous", packed.continuous), ("discrete", packed.discrete)):
            for component in layout.components(role, value_type):
                if component.name.endswith(".arm.pose"):
                    continue
                key = f"{prefix}.end_effector.{component.side}.{suffix}"
                if value_type == "discrete":
                    key += "_token"
                fields[key] = vector[component.offset : component.stop].tolist()
        return fields

    def record_frame(
        self,
        *,
        sample: QposSample,
        qpos_result: Mapping[str, Sequence[float]],
        gripper_sample: GripperReplaySample,
        gripper_result: Mapping[str, float],
        images: Mapping[str, np.ndarray],
        playback_time_sec: float,
        active_sides: Sequence[str],
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot record to a closed v2.2 sim dataset writer.")
        missing = sorted(set(self.camera_names) - set(images))
        if missing:
            raise ValueError(f"Missing rendered camera images: {missing}")
        frame_index = len(self.rows)
        timestamp = float(sample.timestamp) if sample.timestamp is not None else self._fallback_timestamp_origin + float(playback_time_sec)
        for camera_name in self.camera_names:
            self._write_video_frame(camera_name, images[camera_name])

        arm_pose = self._arm_pose_by_side()
        qpos = {side: np.asarray(qpos_result[side], dtype=np.float32).reshape(6) for side in ("left", "right")}
        action_close = {side: [float(gripper_sample.close_by_side[side])] for side in ("left", "right")}
        measured_close = {
            side: [float(gripper_result.get(name, gripper_sample.close_by_side[side]))]
            for side, name in self.config.gripper_name_by_side.items()
        }
        action = self.layout.pack_action(arm_pose_by_side=arm_pose, end_effector_by_side=action_close, strict=True)
        state = self.layout.pack_observation_state(arm_pose_by_side=arm_pose, end_effector_by_side=measured_close, strict=True)
        last_action = action.continuous if self._previous_action is None else self._previous_action
        row: dict[str, Any] = {
            "index": frame_index,
            "episode_index": self.episode_index,
            "frame_index": frame_index,
            "source_frame_index": int(sample.frame_index),
            "timestamp": timestamp,
            "relative_time_sec": float(sample.relative_time_sec),
            "playback_time_sec": float(playback_time_sec),
            "dexteleop.dataset_schema": DEXTELEOP_DATASET_SCHEMA_V22,
            "policy_io_layout_hash": self.layout.layout_hash,
            "active_sides": list(active_sides),
            "camera_keys": list(self.camera_names),
            "action": action.continuous.tolist(),
            "observation.state": state.continuous.tolist(),
            "observation.last_action": last_action.tolist(),
            "action.arm.pose": np.concatenate((arm_pose["left"], arm_pose["right"])).tolist(),
            "observation.arm.pose": np.concatenate((arm_pose["left"], arm_pose["right"])).tolist(),
            "observation.arm.qpos": self.layout.pack_arm_qpos_audit(qpos).tolist(),
            "observation.arm.qpos_valid": [True] * 12,
            "sim.provenance_json": json.dumps({"qpos_source_by_side": sample.source_by_side, "gripper_source_by_side": gripper_sample.source_by_side}, ensure_ascii=True, sort_keys=True),
            **self._component_fields("action", action, self.layout, role="action"),
            **self._component_fields("observation", state, self.layout, role="state"),
        }
        if self.layout.action_discrete_dim:
            row["action.discrete"] = action.discrete.tolist()
            row["observation.last_action.discrete"] = action.discrete.tolist()
        if self.layout.state_discrete_dim:
            row["observation.state.discrete"] = state.discrete.tolist()
        for camera_name in self.camera_names:
            prefix = f"observation.images.{camera_name}"
            row.update({
                f"{prefix}.source_timestamp": timestamp,
                f"{prefix}.receive_timestamp": timestamp,
                f"{prefix}.video_frame_index": frame_index,
                f"{prefix}.path": self.video_relative_paths[camera_name],
                f"observation.sim_camera.{camera_name}.pose_world": list(
                    pose_to_xyz_quat_xyzw(self.scene.camera_pose(camera_name))
                ),
            })
        self.rows.append(row)
        self._previous_action = action.continuous.copy()

    def close(self, *, summary: Mapping[str, Any] | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        for writer in self._video_writers.values():
            writer.release()
        if self.rows:
            write_rows_to_parquet(
                self.rows,
                self.root_dir / self.parquet_relative_path,
                compression=self.config.parquet_compression,
                fixed_size_list_columns=self.layout.parquet_fixed_list_columns(),
            )
            _append_jsonl(
                self.root_dir / "meta" / "episodes.jsonl",
                {
                    "episode_index": self.episode_index,
                    "episode_id": self.episode_id,
                    "frame_count": len(self.rows),
                    "start_timestamp": self.rows[0]["timestamp"],
                    "end_timestamp": self.rows[-1]["timestamp"],
                    "parquet_path": self.parquet_relative_path,
                    "video_paths": self.video_relative_paths,
                    "tactile_paths": {},
                    "metadata": {
                        "source_entrypoint": "sim.mujoco.replay.runner",
                        "policy_io_layout_hash": self.layout.layout_hash,
                        "sim_provenance": self._provenance(),
                        "modality_availability": {"force_torque": "unavailable", "tactile": "unavailable"},
                    },
                    "summary": {} if summary is None else dict(summary),
                },
            )
        self._write_info()
