"""Stable LeRobot/DexTeleop dataset layout helpers."""

from __future__ import annotations


def episode_id_from_index(index: int) -> str:
    return f"episode_{int(index):06d}"


def chunk_id_from_episode_index(index: int, *, chunk_size: int) -> str:
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    return f"chunk-{int(index) // int(chunk_size):03d}"


def camera_video_channel(camera_key: str) -> str:
    return f"observation.images.{str(camera_key)}"


__all__ = [
    "camera_video_channel",
    "chunk_id_from_episode_index",
    "episode_id_from_index",
]
