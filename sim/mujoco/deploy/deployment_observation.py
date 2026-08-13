"""MuJoCo-to-DeploymentObservation bridge.

The bridge samples only immutable scene state and optional rendered images. It
does not step physics, mutate qpos, invoke IK, or submit actions. This keeps a
simulator observation source interchangeable with the real deployment source
while leaving model-specific field names to the model adapter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dataset_contracts.pose_frames import PoseFrameContract, PoseFrameContractError
from robot_motion.deploy import DeploymentObservation, build_deployment_observation
from robot_motion.deploy.schema import normalize_active_sides


CameraRenderer = Callable[[str], Any]
TcpPoseBySideProvider = Callable[[], Mapping[str, Any]]


def _normalize_mapping(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return {str(key).strip().lower(): value[key] for key in value}


def _normalize_pose_contract(
    value: PoseFrameContract | Mapping[str, Any] | None,
) -> PoseFrameContract | None:
    if value is None:
        return None
    if isinstance(value, PoseFrameContract):
        return value
    if isinstance(value, Mapping):
        try:
            return PoseFrameContract.from_dict(value)
        except PoseFrameContractError as exc:
            raise ValueError(f"invalid pose_frame_contract: {exc}") from exc
    raise TypeError("pose_frame_contract must be a PoseFrameContract, mapping, or None.")


@dataclass(frozen=True)
class MujocoDeploymentObservationConfig:
    """Explicit scene-to-policy observation bindings."""

    active_sides: tuple[str, ...]
    tcp_body_by_side: Mapping[str, str]
    hands_enabled: bool = True
    end_effector_sides: tuple[str, ...] | None = None
    gripper_name_by_side: Mapping[str, str] = field(default_factory=dict)
    camera_names: tuple[str, ...] = ()
    require_images: bool = False
    camera_frame_max_age_sec: float | None = 0.1
    pose_frame_contract: PoseFrameContract | Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sides = normalize_active_sides(self.active_sides)
        if not sides:
            raise ValueError("active_sides must not be empty.")
        tcp = _normalize_mapping(self.tcp_body_by_side, name="tcp_body_by_side")
        if set(tcp) != set(sides):
            raise ValueError(
                "tcp_body_by_side must contain exactly the active sides: "
                f"expected={sides!r}, got={tuple(sorted(tcp))!r}."
            )
        for side, body in tcp.items():
            if not str(body).strip():
                raise ValueError(f"tcp_body_by_side[{side!r}] must be non-empty.")
        requested_ee = self.end_effector_sides
        if not self.hands_enabled:
            if requested_ee:
                raise ValueError("end_effector_sides requires hands_enabled=True.")
            requested_ee = ()
        else:
            requested_ee = sides if requested_ee is None else normalize_active_sides(requested_ee)
            if not set(requested_ee).issubset(set(sides)):
                raise ValueError("end_effector_sides must be a subset of active_sides.")
        grippers = _normalize_mapping(self.gripper_name_by_side, name="gripper_name_by_side")
        if not set(grippers).issubset(set(requested_ee)):
            raise ValueError("gripper_name_by_side contains a side outside end_effector_sides.")
        cameras = tuple(str(name).strip() for name in self.camera_names)
        if any(not name for name in cameras):
            raise ValueError("camera_names must contain non-empty names.")
        if len(set(cameras)) != len(cameras):
            raise ValueError("camera_names must not contain duplicates.")
        if self.require_images and not cameras:
            raise ValueError("require_images=True requires at least one camera_name.")
        max_age = self.camera_frame_max_age_sec
        if max_age is not None:
            max_age = float(max_age)
            if max_age <= 0.0:
                raise ValueError("camera_frame_max_age_sec must be positive or None.")
        pose_contract = _normalize_pose_contract(self.pose_frame_contract)
        object.__setattr__(self, "active_sides", sides)
        object.__setattr__(self, "tcp_body_by_side", tcp)
        object.__setattr__(self, "hands_enabled", bool(self.hands_enabled))
        object.__setattr__(self, "end_effector_sides", tuple(requested_ee))
        object.__setattr__(self, "gripper_name_by_side", grippers)
        object.__setattr__(self, "camera_names", cameras)
        object.__setattr__(self, "require_images", bool(self.require_images))
        object.__setattr__(self, "camera_frame_max_age_sec", max_age)
        object.__setattr__(self, "pose_frame_contract", pose_contract)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class _StaticFrame:
    image: np.ndarray
    source_timestamp: float
    available_timestamp: float


class _StaticFrameBuffer:
    def __init__(self, frame: _StaticFrame):
        self._frame = frame

    def get_latest_before_or_at(self, cutoff_timestamp: float) -> _StaticFrame | None:
        if self._frame.source_timestamp <= float(cutoff_timestamp):
            return self._frame
        return None


class _StaticCameraSource:
    def __init__(self, frame: _StaticFrame):
        self.frame_buffer = lambda: _StaticFrameBuffer(frame)

    def get_latest_frame(self) -> _StaticFrame:
        return self.frame_buffer().get_latest_before_or_at(float("inf"))  # type: ignore[return-value]


class MujocoDeploymentObservationAdapter:
    """Read a :class:`MujocoScene` into the shared observation contract."""

    def __init__(
        self,
        scene: Any,
        config: MujocoDeploymentObservationConfig,
        *,
        renderer: Any | None = None,
        tcp_pose_by_side_provider: TcpPoseBySideProvider | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, MujocoDeploymentObservationConfig):
            raise TypeError("config must be a MujocoDeploymentObservationConfig.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        if tcp_pose_by_side_provider is not None and not callable(
            tcp_pose_by_side_provider
        ):
            raise TypeError(
                "tcp_pose_by_side_provider must be callable or None."
            )
        self.scene = scene
        self.config = config
        self.renderer = renderer
        self.tcp_pose_by_side_provider = tcp_pose_by_side_provider
        self.clock = clock
        self._step_index = 0
        self._last_action: np.ndarray | None = None

    def set_last_action(self, action: Sequence[float] | np.ndarray | None) -> None:
        if action is None:
            self._last_action = None
            return
        vector = np.asarray(action, dtype=np.float32).reshape(-1)
        if vector.shape != (26,) or not np.all(np.isfinite(vector)):
            raise ValueError("last_action must be one finite 26D runtime action vector.")
        self._last_action = vector.copy()

    def reset(self) -> None:
        """Release policy-local history without mutating simulator state."""

        self._step_index = 0
        self._last_action = None

    def _find_gripper_name(self, side: str) -> str:
        configured = self.config.gripper_name_by_side.get(side)
        specs = getattr(self.scene, "gripper_specs", {})
        if configured is not None:
            if configured not in specs:
                raise KeyError(f"Scene has no configured {side} gripper {configured!r}.")
            return configured
        candidates = [
            str(name)
            for name, spec in specs.items()
            if getattr(spec, "side", None) == side
        ]
        if not candidates:
            candidates = [
                str(name)
                for name in specs
                if str(name).strip().lower() in {f"{side}_gripper", f"{side}_hand"}
            ]
        if len(candidates) != 1:
            raise KeyError(
                f"Cannot resolve one {side} gripper from scene specs: {sorted(candidates)!r}."
            )
        return candidates[0]

    def _render_images(self, timestamp: float) -> Mapping[str, Any]:
        if not self.config.camera_names:
            return {}
        if self.renderer is None:
            raise RuntimeError(
                "camera_names are configured but no MuJoCo renderer was supplied."
            )
        render = self.renderer if callable(self.renderer) else getattr(self.renderer, "render", None)
        if not callable(render):
            raise TypeError("renderer must be callable or expose render(camera_name).")
        sources: dict[str, _StaticCameraSource] = {}
        for camera_name in self.config.camera_names:
            image = np.asarray(render(camera_name))
            if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
                raise ValueError(
                    f"MuJoCo camera {camera_name!r} must return HWC image, got {image.shape}."
                )
            if image.shape[-1] != 3:
                raise ValueError(
                    f"MuJoCo camera {camera_name!r} must return 3-channel RGB, got {image.shape}."
                )
            frame = _StaticFrame(
                image=np.ascontiguousarray(image.copy()),
                source_timestamp=float(timestamp),
                available_timestamp=float(timestamp),
            )
            sources[camera_name] = _StaticCameraSource(frame)
        return sources

    def capture(
        self,
        *,
        step_index: int | None = None,
        last_action: Sequence[float] | np.ndarray | None = None,
    ) -> DeploymentObservation:
        timestamp = float(self.clock())
        if not np.isfinite(timestamp):
            raise ValueError("clock returned a non-finite timestamp.")
        index = self._step_index if step_index is None else int(step_index)
        if index < 0:
            raise ValueError("step_index must be non-negative.")

        arm_states: dict[str, dict[str, Any]] = {}
        provided_pose_by_side = (
            None
            if self.tcp_pose_by_side_provider is None
            else dict(self.tcp_pose_by_side_provider())
        )
        if provided_pose_by_side is not None and set(provided_pose_by_side) != set(
            self.config.active_sides
        ):
            raise ValueError(
                "tcp_pose_by_side_provider must return exactly the active sides."
            )
        for side in self.config.active_sides:
            qpos = np.asarray(self.scene.get_arm_qpos(side), dtype=np.float32).reshape(-1)
            if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
                raise ValueError(f"MuJoCo {side} arm qpos is not a finite 6D vector.")
            pose = np.asarray(
                self.scene.body_pose(self.config.tcp_body_by_side[side])
                if provided_pose_by_side is None
                else provided_pose_by_side[side],
                dtype=np.float64,
            )
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                raise ValueError(f"MuJoCo {side} TCP body pose is invalid.")
            arm_states[side] = {
                "current_qpos": qpos.copy(),
                "actual_qpos": qpos.copy(),
                "current_tcp_pose": pose.copy(),
                "actual_tcp_pose": pose.copy(),
                "measured_at": timestamp,
            }

        hand_states: dict[str, dict[str, Any]] = {}
        for side in self.config.end_effector_sides or ():
            name = self._find_gripper_name(side)
            qpos = np.asarray(self.scene.get_gripper_qpos(name), dtype=np.float32).reshape(-1)
            if qpos.size == 0 or not np.all(np.isfinite(qpos)):
                raise ValueError(f"MuJoCo {side} gripper qpos is invalid.")
            close = float(self.scene.get_gripper_close(name))
            if not np.isfinite(close):
                raise ValueError(f"MuJoCo {side} gripper close state is invalid.")
            if qpos.size > 6:
                raise ValueError(
                    f"MuJoCo {side} gripper qpos has {qpos.size} values; expected at most 6."
                )
            hand_states[side] = {
                "joint_pos": qpos.copy()
                if qpos.size == 6
                else np.pad(qpos, (0, 6 - qpos.size)),
                "gripper_state": np.asarray([close], dtype=np.float32),
                "gripper_state_scalar": close,
                "gripper_measured_scalar": close,
                "measured_gripper_close": close,
                "measured_close": close,
                "state": np.asarray([close], dtype=np.float32),
                "current_close": close,
                "joint_timestamp": timestamp,
                "source_gripper_name": name,
            }

        metadata = dict(self.config.metadata)
        metadata.update(
            {
                "source": "mujoco",
                "tcp_body_by_side": dict(self.config.tcp_body_by_side),
                "camera_names": list(self.config.camera_names),
                "mujoco_step_index": index,
            }
        )
        if self.config.pose_frame_contract is not None:
            metadata["pose_frame_contract"] = self.config.pose_frame_contract.to_dict()
            metadata["pose_frame_contract_hash"] = self.config.pose_frame_contract.contract_hash

        selected_last_action = self._last_action if last_action is None else last_action
        camera_sources = self._render_images(timestamp)
        observation = build_deployment_observation(
            active_sides=self.config.active_sides,
            arms_enabled=True,
            hands_enabled=self.config.hands_enabled,
            end_effector_sides=self.config.end_effector_sides,
            step_index=index,
            arm_states=arm_states,
            hand_states=hand_states,
            camera_sources=camera_sources,
            last_action=selected_last_action,
            timestamp=timestamp,
            require_images=self.config.require_images,
            camera_frame_max_age_sec=self.config.camera_frame_max_age_sec,
            metadata=metadata,
        )
        self._step_index = index + 1
        return observation


__all__ = [
    "MujocoDeploymentObservationAdapter",
    "MujocoDeploymentObservationConfig",
]
