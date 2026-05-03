from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from profiles import DeploymentProfile, FRANKA_PROFILE, REALMAN20_PROFILE, REALMAN21_PROFILE, REALMAN22_PROFILE
from tactile import tactile_to_351


DEFAULT_IMAGE_SHAPE = (480, 640, 3)


def warning(message: str) -> None:
    print(f"WARNING: {message}")


def safe_array(value: Any, width: int, name: str, default: np.ndarray | None = None) -> np.ndarray:
    if value is None:
        if default is not None:
            return np.asarray(default, dtype=np.float32).reshape(width)
        warning(f"{name} is missing; using zeros.")
        return np.zeros(width, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.zeros(width, dtype=np.float32)
    copy = min(width, arr.shape[0])
    out[:copy] = arr[:copy]
    if not np.isfinite(out).all():
        warning(f"{name} contains non-finite values; replacing with zeros.")
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def first_parquet(dataset_root: Path) -> Path:
    candidates = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet episodes found under {dataset_root / 'data'}")
    return candidates[0]


def load_episode_dataframe(dataset_path: str, parquet_path: str = "", episode_index: int = -1):
    import pandas as pd

    dataset_root = Path(dataset_path)
    path = Path(parquet_path) if parquet_path else first_parquet(dataset_root)
    df = pd.read_parquet(path)
    if episode_index >= 0 and "episode_index" in df.columns:
        selected = df[df["episode_index"] == episode_index]
        if len(selected):
            df = selected
        else:
            warning(f"episode_index={episode_index} not found in {path}; using all rows.")
    if len(df) == 0:
        raise ValueError(f"{path} is empty.")
    df = df.reset_index(drop=True)
    df.attrs["dataset_root"] = str(dataset_root)
    df.attrs["parquet_path"] = str(path)
    return df


def zero_image(height: int = DEFAULT_IMAGE_SHAPE[0], width: int = DEFAULT_IMAGE_SHAPE[1]) -> np.ndarray:
    return np.zeros((int(height), int(width), 3), dtype=np.uint8)


def load_video_frame(df: Any, row: Any, fallback_height: int, fallback_width: int) -> np.ndarray:
    rel_path = row.get("observation.images.front.path") if hasattr(row, "get") else None
    if not rel_path:
        return zero_image(fallback_height, fallback_width)
    video_path = Path(df.attrs["dataset_root"]) / str(rel_path)
    if not video_path.exists():
        warning(f"Video path missing: {video_path}")
        return zero_image(fallback_height, fallback_width)
    try:
        import cv2
    except Exception as exc:
        warning(f"cv2 unavailable for video decoding: {exc}")
        return zero_image(fallback_height, fallback_width)
    frame_index = int(row.get("observation.images.front.video_frame_index", 0) or 0)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warning(f"Could not open video: {video_path}")
        return zero_image(fallback_height, fallback_width)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index, 0))
        ok, frame_bgr = cap.read()
    finally:
        cap.release()
    if not ok or frame_bgr is None:
        warning(f"Could not decode frame {frame_index} from {video_path}")
        return zero_image(fallback_height, fallback_width)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)


def tactile_history_from_df(df: Any, row_index: int, column: str, side: str, history_size: int, history_stride: int) -> np.ndarray:
    if column not in df.columns:
        warning(f"{column} missing; using zero tactile history.")
        return np.zeros((history_size, 351), dtype=np.float32)
    base = row_index % len(df)
    samples = []
    for offset in range(history_size - 1, -1, -1):
        idx = max(0, base - offset * history_stride)
        samples.append(tactile_to_351(df.iloc[idx].get(column), side))
    return np.stack(samples, axis=0).astype(np.float32)


def franka_pose_from_row(row: Any) -> np.ndarray:
    raw = row.get("observation.arm.raw_state_json")
    if raw:
        try:
            data = json.loads(raw)
            return np.concatenate(
                [
                    safe_array(data.get("ee_pos"), 3, "ee_pos"),
                    safe_array(data.get("ee_quat"), 4, "ee_quat", default=np.array([0, 0, 0, 1], dtype=np.float32)),
                ],
                axis=0,
            ).astype(np.float32)
        except Exception as exc:
            warning(f"Could not parse FR3 raw_state_json: {exc}")
    state = safe_array(row.get("observation.state"), 13, "observation.state")
    return np.concatenate([state[:3], state[3:7]], axis=0).astype(np.float32)


class DatasetObservationSource:
    def __init__(
        self,
        profile: DeploymentProfile,
        dataset_path: str,
        parquet_path: str,
        episode_index: int,
        start_frame: int,
        history_size: int,
        history_stride: int,
        fallback_image_height: int,
        fallback_image_width: int,
        frame_stride: int = 1,
        loop: bool = False,
    ):
        self.profile = profile
        self.df = load_episode_dataframe(dataset_path, parquet_path, episode_index)
        self.row_index = max(0, int(start_frame))
        self.next_index = self.row_index
        self.history_size = max(1, int(history_size))
        self.history_stride = max(1, int(history_stride))
        self.fallback_image_height = int(fallback_image_height)
        self.fallback_image_width = int(fallback_image_width)
        self.frame_stride = max(1, int(frame_stride))
        self.loop = bool(loop)
        print(f"[Dataset] loaded {len(self.df)} rows from {self.df.attrs['parquet_path']}")
        print(f"[Dataset] start_frame={self.row_index} frame_stride={self.frame_stride} loop={self.loop}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def next_payload_base(self) -> dict[str, Any]:
        if self.next_index >= len(self.df):
            if not self.loop:
                raise StopIteration(
                    f"Dataset source reached end of episode at row {self.next_index}; "
                    "increase --dataset-loop to wrap around."
                )
            self.next_index = self.next_index % len(self.df)
        row_index = int(self.next_index)
        row = self.df.iloc[row_index]
        profile = self.profile
        if profile is FRANKA_PROFILE:
            state = {
                "eef_pose": franka_pose_from_row(row),
                "right_hand_qpos": safe_array(row.get("observation.right_hand.qpos"), 6, "observation.right_hand.qpos"),
            }
        elif profile is REALMAN20_PROFILE:
            state = {
                "right_arm_pose": safe_array(row.get("observation.right_arm.pose"), 7, "observation.right_arm.pose"),
                "right_hand_qpos": safe_array(row.get("observation.right_hand.qpos"), 6, "observation.right_hand.qpos"),
            }
        elif profile is REALMAN21_PROFILE:
            state = {
                "right_arm_pose": safe_array(row.get("observation.right_arm.pose"), 7, "observation.right_arm.pose"),
                "right_hand_qpos": safe_array(row.get("observation.right_hand.qpos"), 6, "observation.right_hand.qpos"),
                "left_hand_qpos": safe_array(row.get("observation.left_hand.qpos"), 6, "observation.left_hand.qpos"),
            }
        elif profile is REALMAN22_PROFILE:
            state = {
                "left_arm_pose": safe_array(row.get("observation.left_arm.pose"), 7, "observation.left_arm.pose"),
                "right_arm_pose": safe_array(row.get("observation.right_arm.pose"), 7, "observation.right_arm.pose"),
                "left_hand_qpos": safe_array(row.get("observation.left_hand.qpos"), 6, "observation.left_hand.qpos"),
                "right_hand_qpos": safe_array(row.get("observation.right_hand.qpos"), 6, "observation.right_hand.qpos"),
            }
        else:
            raise ValueError(f"Unsupported profile: {profile}")

        payload = {
            "image": load_video_frame(self.df, row, self.fallback_image_height, self.fallback_image_width),
            "state": state,
            "tactile": {
                "right": tactile_history_from_df(
                    self.df,
                    row_index,
                    "observation.hand.right.tactile",
                    "right",
                    self.history_size,
                    self.history_stride,
                ),
                "left": tactile_history_from_df(
                    self.df,
                    row_index,
                    "observation.hand.left.tactile",
                    "left",
                    self.history_size,
                    self.history_stride,
                ),
            },
        }
        payload["__dataset_row_index"] = row_index
        self.row_index = row_index
        self.next_index = row_index + self.frame_stride
        return payload
