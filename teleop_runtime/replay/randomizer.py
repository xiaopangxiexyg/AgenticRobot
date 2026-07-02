from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM


def _random_unit_rotvec(rng: np.random.Generator, max_angle_rad: float) -> np.ndarray:
    if float(max_angle_rad) <= 0.0:
        return np.zeros(3, dtype=np.float32)
    direction = rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        direction = direction / norm
    angle = rng.uniform(-float(max_angle_rad), float(max_angle_rad))
    return (direction * angle).astype(np.float32)


def _random_xyz_offset(rng: np.random.Generator, max_xyz_m: float) -> np.ndarray:
    if float(max_xyz_m) <= 0.0:
        return np.zeros(3, dtype=np.float32)
    return rng.uniform(
        -float(max_xyz_m),
        float(max_xyz_m),
        size=3,
    ).astype(np.float32)


def _smoothstep_unit(alpha: float) -> float:
    x = float(np.clip(alpha, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


@dataclass
class ReplayPoseOffset:
    xyz_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    rotvec_rad: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def copy(self) -> "ReplayPoseOffset":
        return ReplayPoseOffset(
            xyz_m=np.asarray(self.xyz_m, dtype=np.float32).reshape(3).copy(),
            rotvec_rad=np.asarray(self.rotvec_rad, dtype=np.float32).reshape(3).copy(),
        )

    def to_metadata(self) -> dict[str, Any]:
        rotvec = np.asarray(self.rotvec_rad, dtype=np.float32).reshape(3)
        return {
            "xyz_m": np.round(np.asarray(self.xyz_m, dtype=np.float32).reshape(3), 6).tolist(),
            "rotvec_rad": np.round(rotvec, 6).tolist(),
            "rot_angle_deg": float(np.rad2deg(np.linalg.norm(rotvec))),
        }

    @staticmethod
    def interpolate(
        start: "ReplayPoseOffset",
        target: "ReplayPoseOffset",
        alpha: float,
    ) -> "ReplayPoseOffset":
        t = float(alpha)
        return ReplayPoseOffset(
            xyz_m=(
                np.asarray(start.xyz_m, dtype=np.float32)
                + (np.asarray(target.xyz_m, dtype=np.float32) - np.asarray(start.xyz_m, dtype=np.float32)) * t
            ).astype(np.float32),
            rotvec_rad=(
                np.asarray(start.rotvec_rad, dtype=np.float32)
                + (
                    np.asarray(target.rotvec_rad, dtype=np.float32)
                    - np.asarray(start.rotvec_rad, dtype=np.float32)
                )
                * t
            ).astype(np.float32),
        )


class ReplayTrajectoryRandomizer:
    def __init__(
        self,
        *,
        enabled: bool,
        mode: str,
        seed: int | None,
        active_sides: tuple[str, ...],
        max_xyz_m: float,
        max_rot_deg: float,
        noise_xyz_m: float,
        noise_rot_deg: float,
        refresh_sec: float,
        smoothing_sec: float,
        independent_sides: bool,
    ):
        self.enabled = bool(enabled)
        self.mode = str(mode)
        self.seed = seed
        self.active_sides = tuple(side for side in ("left", "right") if side in set(active_sides))
        self.max_xyz_m = max(0.0, float(max_xyz_m))
        self.max_rot_rad = np.deg2rad(max(0.0, float(max_rot_deg)))
        self.noise_xyz_m = max(0.0, float(noise_xyz_m))
        self.noise_rot_rad = np.deg2rad(max(0.0, float(noise_rot_deg)))
        self.refresh_sec = max(1e-6, float(refresh_sec))
        self.smoothing_sec = max(0.0, float(smoothing_sec))
        self.independent_sides = bool(independent_sides)
        self._rng = np.random.default_rng(seed)
        self._episode_offsets = self._sample_offsets(
            max_xyz_m=self.max_xyz_m,
            max_rot_rad=self.max_rot_rad,
        )
        self._current_noise = self._zero_offsets()
        self._start_noise = self._zero_offsets()
        self._target_noise = self._zero_offsets()
        self._target_started_at: float | None = None
        self._next_target_at: float | None = None

    @classmethod
    def from_args(
        cls,
        args: Any,
        *,
        active_sides: tuple[str, ...],
        episode_index: int,
        pool_started_count: int,
        source_episode_index: int,
    ) -> "ReplayTrajectoryRandomizer | None":
        if not bool(getattr(args, "replay_randomize", False)):
            return None
        base_seed = int(getattr(args, "replay_random_seed", -1))
        seed = None
        if base_seed >= 0:
            seed = (
                int(base_seed)
                + int(episode_index) * 1000003
                + int(source_episode_index) * 9176
                + int(pool_started_count) * 131
            ) % (2**32)
        max_xyz = float(getattr(args, "replay_random_xyz_m", 0.0))
        max_rot = float(getattr(args, "replay_random_rot_deg", 0.0))
        noise_xyz = float(getattr(args, "replay_random_noise_xyz_m", -1.0))
        noise_rot = float(getattr(args, "replay_random_noise_rot_deg", -1.0))
        if noise_xyz < 0.0:
            noise_xyz = max_xyz * 0.5
        if noise_rot < 0.0:
            noise_rot = max_rot * 0.5
        return cls(
            enabled=True,
            mode=str(getattr(args, "replay_random_mode", "both")),
            seed=seed,
            active_sides=active_sides,
            max_xyz_m=max_xyz,
            max_rot_deg=max_rot,
            noise_xyz_m=noise_xyz,
            noise_rot_deg=noise_rot,
            refresh_sec=float(getattr(args, "replay_random_refresh_sec", 0.8)),
            smoothing_sec=float(getattr(args, "replay_random_smoothing_sec", 0.3)),
            independent_sides=bool(getattr(args, "replay_random_independent_sides", False)),
        )

    def _zero_offsets(self) -> dict[str, ReplayPoseOffset]:
        return {side: ReplayPoseOffset() for side in self.active_sides}

    def _sample_one_offset(self, *, max_xyz_m: float, max_rot_rad: float) -> ReplayPoseOffset:
        return ReplayPoseOffset(
            xyz_m=_random_xyz_offset(self._rng, max_xyz_m),
            rotvec_rad=_random_unit_rotvec(self._rng, max_rot_rad),
        )

    def _sample_offsets(self, *, max_xyz_m: float, max_rot_rad: float) -> dict[str, ReplayPoseOffset]:
        if not self.independent_sides:
            shared = self._sample_one_offset(max_xyz_m=max_xyz_m, max_rot_rad=max_rot_rad)
            return {side: shared.copy() for side in self.active_sides}
        return {
            side: self._sample_one_offset(max_xyz_m=max_xyz_m, max_rot_rad=max_rot_rad)
            for side in self.active_sides
        }

    def _episode_offset_for(self, side: str) -> ReplayPoseOffset:
        if self.mode not in {"episode_offset", "both"}:
            return ReplayPoseOffset()
        return self._episode_offsets.get(side, ReplayPoseOffset())

    def _noise_offset_for(self, side: str, now: float | None) -> ReplayPoseOffset:
        if self.mode not in {"smooth_noise", "both"}:
            return ReplayPoseOffset()
        if now is None:
            return self._current_noise.get(side, ReplayPoseOffset())
        self._advance_noise(float(now))
        return self._current_noise.get(side, ReplayPoseOffset())

    def _advance_noise(self, now: float) -> None:
        if self._target_started_at is None:
            self._target_started_at = float(now)
            self._next_target_at = float(now)
        if self._next_target_at is not None and now >= self._next_target_at:
            self._start_noise = {side: offset.copy() for side, offset in self._current_noise.items()}
            self._target_noise = self._sample_offsets(
                max_xyz_m=self.noise_xyz_m,
                max_rot_rad=self.noise_rot_rad,
            )
            self._target_started_at = float(now)
            self._next_target_at = float(now) + self.refresh_sec

        assert self._target_started_at is not None
        if self.smoothing_sec <= 1e-9:
            alpha = 1.0
        else:
            alpha = _smoothstep_unit((float(now) - self._target_started_at) / self.smoothing_sec)
        self._current_noise = {
            side: ReplayPoseOffset.interpolate(
                self._start_noise.get(side, ReplayPoseOffset()),
                self._target_noise.get(side, ReplayPoseOffset()),
                alpha,
            )
            for side in self.active_sides
        }

    @staticmethod
    def _compose_offsets(first: ReplayPoseOffset, second: ReplayPoseOffset) -> ReplayPoseOffset:
        return ReplayPoseOffset(
            xyz_m=(
                np.asarray(first.xyz_m, dtype=np.float32)
                + np.asarray(second.xyz_m, dtype=np.float32)
            ).astype(np.float32),
            rotvec_rad=(
                np.asarray(first.rotvec_rad, dtype=np.float32)
                + np.asarray(second.rotvec_rad, dtype=np.float32)
            ).astype(np.float32),
        )

    def offset_for(self, side: str, now: float | None) -> ReplayPoseOffset:
        return self._compose_offsets(
            self._episode_offset_for(side),
            self._noise_offset_for(side, now),
        )

    def apply(self, action: np.ndarray, *, now: float | None = None) -> np.ndarray:
        vector = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM).copy()
        if not self.enabled:
            return vector
        for side in self.active_sides:
            target_slice = ACTION_ARM_POSE_SLICES[side]
            pose = vector[target_slice].astype(np.float32, copy=True)
            if pose.shape != (7,) or not np.all(np.isfinite(pose)):
                continue
            offset = self.offset_for(side, now)
            pose[:3] = pose[:3] + np.asarray(offset.xyz_m, dtype=np.float32).reshape(3)
            quat = np.asarray(pose[3:7], dtype=float).reshape(4)
            quat_norm = float(np.linalg.norm(quat))
            if quat_norm <= 1e-12:
                continue
            base_rot = R.from_quat(quat / quat_norm)
            delta_rot = R.from_rotvec(np.asarray(offset.rotvec_rad, dtype=float).reshape(3))
            pose[3:7] = (delta_rot * base_rot).as_quat().astype(np.float32)
            vector[target_slice] = pose
        return vector

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "seed": self.seed,
            "active_sides": list(self.active_sides),
            "independent_sides": bool(self.independent_sides),
            "max_xyz_m": float(self.max_xyz_m),
            "max_rot_deg": float(np.rad2deg(self.max_rot_rad)),
            "noise_xyz_m": float(self.noise_xyz_m),
            "noise_rot_deg": float(np.rad2deg(self.noise_rot_rad)),
            "refresh_sec": float(self.refresh_sec),
            "smoothing_sec": float(self.smoothing_sec),
            "episode_offsets": {
                side: offset.to_metadata()
                for side, offset in sorted(self._episode_offsets.items())
            },
        }

    def current_offsets_metadata(self) -> dict[str, Any]:
        return {
            side: self.offset_for(side, None).to_metadata()
            for side in self.active_sides
        }
