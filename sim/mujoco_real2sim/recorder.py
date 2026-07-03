from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np

from .scene import MujocoScene
from .transforms import pose_to_xyz_quat_xyzw


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


@dataclass(frozen=True)
class RecorderConfig:
    root_dir: Path
    save_images: bool = True


class Real2SimRecorder:
    def __init__(self, scene: MujocoScene, config: RecorderConfig):
        self.scene = scene
        self.config = config
        self.root_dir = Path(config.root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.frames_path = self.root_dir / "frames.jsonl"
        self._frames_file = self.frames_path.open("a", encoding="utf-8")
        self._frame_index = 0
        self._write_metadata()

    @classmethod
    def create_timestamped(
        cls,
        scene: MujocoScene,
        *,
        root_dir: str | Path | None = None,
        save_images: bool = True,
    ) -> "Real2SimRecorder":
        base = Path(root_dir) if root_dir is not None else Path("sim") / "outputs"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls(scene, RecorderConfig(root_dir=base / f"mujoco_real2sim_{stamp}", save_images=save_images))

    def _write_metadata(self) -> None:
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "urdf_path": str(self.scene.conversion.urdf_path),
            "mjcf_path": str(self.scene.conversion.mjcf_path),
            "arms": {
                side: {
                    "joint_names": list(spec.joint_names),
                    "controlled_body": spec.controlled_body,
                    "effector_name": spec.effector_name,
                }
                for side, spec in sorted(self.scene.arm_specs.items())
            },
            "cameras": self.scene.camera_metadata(),
        }
        (self.root_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )

    def record_step(
        self,
        *,
        images: Mapping[str, np.ndarray] | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        frame_index = self._frame_index
        self._frame_index += 1
        image_paths: dict[str, str] = {}
        if images and self.config.save_images:
            from PIL import Image

            for camera_name, image in images.items():
                camera_dir = self.root_dir / "images" / str(camera_name)
                camera_dir.mkdir(parents=True, exist_ok=True)
                path = camera_dir / f"{frame_index:06d}.png"
                Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
                image_paths[str(camera_name)] = str(path.relative_to(self.root_dir))

        body_poses = {
            side: pose_to_xyz_quat_xyzw(self.scene.body_pose(spec.controlled_body))
            for side, spec in sorted(self.scene.arm_specs.items())
        }
        record = {
            "frame_index": frame_index,
            "timestamp": time.time(),
            "qpos_by_side": self.scene.arm_qpos_by_side(),
            "controlled_body_pose_world_xyz_quat_xyzw": body_poses,
            "camera_metadata": self.scene.camera_metadata(),
            "image_paths": image_paths,
            "extra": {} if extra is None else dict(extra),
        }
        self._frames_file.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        self._frames_file.flush()
        return record

    def close(self) -> None:
        self._frames_file.close()

    def __enter__(self) -> "Real2SimRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

