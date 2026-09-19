"""Pure RM65 free-drive to B601 pose retargeting.

The module is deliberately below the control-station application layer.  It
accepts an already admitted, observation-only RM65 pose sample and a reviewed
coordinate-frame graph, then reuses the existing B601 continuous-seed IK
solver.  It does not open a device, read hardware, command a motor, or step a
MuJoCo scene.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.coordinate_frames import CoordinateFrameGraph
from robot_motion.transforms import parse_pose_matrix, validate_pose_matrix

from .ik import B601IKResult

RM65_FREE_DRIVE_SAMPLE_SCHEMA_V1 = "robot_motion.rm65_free_drive_sample.v1"
RM65_TO_B601_RETARGET_SCHEMA_V1 = "robot_motion.rm65_to_b601_retarget.v1"
RM65_FREE_DRIVE_SAMPLE_HASH_DOMAIN_V1 = (
    "dexteleop.robot_motion.rm65_free_drive_sample.v1"
)
RM65_TO_B601_RETARGET_HASH_DOMAIN_V1 = (
    "dexteleop.robot_motion.rm65_to_b601_retarget.v1"
)


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite_scalar(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite, not bool.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _finite_qpos(value: object, *, name: str) -> tuple[float, ...]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain six finite joint values.") from exc
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain six finite joint values.")
    return tuple(float(item) for item in array)


def _pose_tuple(value: object, *, name: str) -> tuple[tuple[float, ...], ...]:
    try:
        pose = parse_pose_matrix(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a supported rigid 4x4 pose.") from exc
    valid, reason = validate_pose_matrix(pose, atol=1e-8)
    if not valid:
        raise ValueError(f"{name} is invalid: {reason}")
    return tuple(tuple(float(item) for item in row) for row in pose)


def _pose_array(value: object, *, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    valid, reason = validate_pose_matrix(pose, atol=1e-8)
    if not valid:
        raise ValueError(f"{name} is invalid: {reason}")
    return pose.copy()


@dataclass(frozen=True)
class RM65FreeDriveSample:
    """One fresh RM65 observation admitted by the read-only source adapter."""

    sequence: int
    source_timestamp_sec: float
    receive_timestamp_sec: float
    receive_timestamp_monotonic_sec: float
    qpos_rad: tuple[float, ...]
    pose_source_reference_source_controlled: tuple[tuple[float, ...], ...]
    source_id: str
    source_clock_domain_id: str
    source_reference_frame_id: str
    source_controlled_frame_id: str
    observation_only: bool = True
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool):
            raise TypeError("sequence must be a positive integer.")
        sequence = int(self.sequence)
        if sequence != self.sequence or sequence <= 0:
            raise ValueError("sequence must be a positive integer.")
        source_timestamp = _finite_scalar(
            self.source_timestamp_sec,
            name="source_timestamp_sec",
        )
        receive_timestamp = _finite_scalar(
            self.receive_timestamp_sec,
            name="receive_timestamp_sec",
        )
        receive_monotonic = _finite_scalar(
            self.receive_timestamp_monotonic_sec,
            name="receive_timestamp_monotonic_sec",
        )
        if source_timestamp > receive_timestamp + 1e-6:
            raise ValueError("source_timestamp_sec cannot be newer than receipt.")
        qpos = _finite_qpos(self.qpos_rad, name="qpos_rad")
        pose = _pose_tuple(
            self.pose_source_reference_source_controlled,
            name="pose_source_reference_source_controlled",
        )
        source_id = _required_text(self.source_id, name="source_id")
        source_clock = _required_text(
            self.source_clock_domain_id,
            name="source_clock_domain_id",
        )
        source_reference = _required_text(
            self.source_reference_frame_id,
            name="source_reference_frame_id",
        )
        source_controlled = _required_text(
            self.source_controlled_frame_id,
            name="source_controlled_frame_id",
        )
        if self.observation_only is not True:
            raise ValueError("RM65 free-drive samples must be observation-only.")
        payload = {
            "schema": RM65_FREE_DRIVE_SAMPLE_SCHEMA_V1,
            "sequence": sequence,
            "source_timestamp_sec": source_timestamp,
            "receive_timestamp_sec": receive_timestamp,
            "receive_timestamp_monotonic_sec": receive_monotonic,
            "qpos_rad": qpos,
            "pose_source_reference_source_controlled": pose,
            "source_id": source_id,
            "source_clock_domain_id": source_clock,
            "source_reference_frame_id": source_reference,
            "source_controlled_frame_id": source_controlled,
            "observation_only": True,
        }
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "source_timestamp_sec", source_timestamp)
        object.__setattr__(self, "receive_timestamp_sec", receive_timestamp)
        object.__setattr__(
            self,
            "receive_timestamp_monotonic_sec",
            receive_monotonic,
        )
        object.__setattr__(self, "qpos_rad", qpos)
        object.__setattr__(
            self,
            "pose_source_reference_source_controlled",
            pose,
        )
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "source_clock_domain_id", source_clock)
        object.__setattr__(self, "source_reference_frame_id", source_reference)
        object.__setattr__(self, "source_controlled_frame_id", source_controlled)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=RM65_FREE_DRIVE_SAMPLE_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class RM65ToB601RetargetConfig:
    """Reverse use of one existing B601/RM65 frame calibration graph."""

    frame_graph: CoordinateFrameGraph
    source_reference_frame_id: str
    source_controlled_frame_id: str
    target_reference_frame_id: str
    target_controlled_frame_id: str
    source_clock_domain_id: str = "local_wall_time"
    translation_gain_target_reference_xyz: tuple[float, float, float] = (
        1.0,
        1.0,
        1.0,
    )
    rotation_gain: float = 1.0
    maximum_source_age_sec: float = 0.25
    maximum_receive_age_sec: float = 0.25
    maximum_future_skew_sec: float = 0.05
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.frame_graph, CoordinateFrameGraph):
            raise TypeError("frame_graph must be a CoordinateFrameGraph.")
        names = {
            name: _required_text(getattr(self, name), name=name)
            for name in (
                "source_reference_frame_id",
                "source_controlled_frame_id",
                "source_clock_domain_id",
                "target_reference_frame_id",
                "target_controlled_frame_id",
            )
        }
        if not self.frame_graph.connected(
            names["source_reference_frame_id"],
            names["target_reference_frame_id"],
        ) or not self.frame_graph.connected(
            names["source_controlled_frame_id"],
            names["target_controlled_frame_id"],
        ):
            raise ValueError("RM65/B601 frame graph does not connect both pose pairs.")
        try:
            gains = tuple(float(item) for item in self.translation_gain_target_reference_xyz)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "translation_gain_target_reference_xyz must contain three finite values."
            ) from exc
        if len(gains) != 3 or any(not math.isfinite(item) or item <= 0.0 for item in gains):
            raise ValueError(
                "translation_gain_target_reference_xyz must contain three positive finite values."
            )
        rotation_gain = _finite_scalar(self.rotation_gain, name="rotation_gain")
        if rotation_gain <= 0.0:
            raise ValueError("rotation_gain must be positive.")
        ages = {
            "maximum_source_age_sec": self.maximum_source_age_sec,
            "maximum_receive_age_sec": self.maximum_receive_age_sec,
            "maximum_future_skew_sec": self.maximum_future_skew_sec,
        }
        normalized_ages: dict[str, float] = {}
        for name, value in ages.items():
            age = _finite_scalar(value, name=name)
            if age < 0.0 or (name != "maximum_future_skew_sec" and age <= 0.0):
                raise ValueError(f"{name} must be non-negative and finite.")
            normalized_ages[name] = age
        payload = {
            "schema": RM65_TO_B601_RETARGET_SCHEMA_V1,
            **names,
            "frame_graph_hash": self.frame_graph.semantic_content_hash,
            "translation_gain_target_reference_xyz": gains,
            "rotation_gain": rotation_gain,
            **normalized_ages,
        }
        for name, value in names.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "translation_gain_target_reference_xyz",
            gains,
        )
        object.__setattr__(self, "rotation_gain", rotation_gain)
        for name, value in normalized_ages.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=RM65_TO_B601_RETARGET_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class RM65ToB601Anchor:
    """Source and target pose cuts captured for one explicit recenter."""

    source_sample_hash: str
    source_sequence: int
    source_pose_target_reference_target_controlled: tuple[tuple[float, ...], ...]
    target_anchor_qpos_rad: tuple[float, ...]
    target_anchor_pose: tuple[tuple[float, ...], ...]
    frame_graph_hash: str


@dataclass(frozen=True)
class RM65ToB601RetargetResult:
    """One accepted target or a current-tick fail-closed rejection."""

    sample: RM65FreeDriveSample
    accepted: bool
    qpos_rad: tuple[float, ...] | None
    target_pose: tuple[tuple[float, ...], ...] | None
    ik_result: B601IKResult | None
    rejection_reason: str | None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample, RM65FreeDriveSample):
            raise TypeError("sample must be RM65FreeDriveSample.")
        if self.accepted:
            if self.qpos_rad is None or self.target_pose is None:
                raise ValueError("Accepted retarget results require qpos and pose.")
            if self.ik_result is None or self.ik_result.success is not True:
                raise ValueError("Accepted retarget results require successful IK.")
            if self.rejection_reason is not None:
                raise ValueError("Accepted retarget results cannot have a rejection.")
        else:
            if self.rejection_reason is None or not str(self.rejection_reason).strip():
                raise ValueError("Rejected retarget results require a reason.")
            if self.qpos_rad is not None or self.target_pose is not None:
                raise ValueError("Rejected retarget results cannot carry a target.")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping.")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


class _B601ForwardKinematics(Protocol):
    def forward_kinematics(self, q_arm: Sequence[float]) -> object: ...


class _B601IKSolver(Protocol):
    def solve(
        self,
        target_pose: object,
        *,
        seed_qpos: Sequence[float],
    ) -> B601IKResult: ...


class RM65ToB601Retargeter:
    """Map a fresh RM65 free-drive pose to a bounded B601 IK target."""

    def __init__(
        self,
        config: RM65ToB601RetargetConfig,
        model: _B601ForwardKinematics,
        ik_solver: _B601IKSolver,
        *,
        wall_clock: Any,
        monotonic_clock: Any,
    ) -> None:
        if not isinstance(config, RM65ToB601RetargetConfig):
            raise TypeError("config must be RM65ToB601RetargetConfig.")
        if not callable(getattr(model, "forward_kinematics", None)):
            raise TypeError("model must provide forward_kinematics.")
        if not callable(getattr(ik_solver, "solve", None)):
            raise TypeError("ik_solver must provide solve.")
        if not callable(wall_clock) or not callable(monotonic_clock):
            raise TypeError("wall_clock and monotonic_clock must be callable.")
        self.config = config
        self.model = model
        self.ik_solver = ik_solver
        self.wall_clock = wall_clock
        self.monotonic_clock = monotonic_clock
        self._anchor: RM65ToB601Anchor | None = None
        self._last_sequence: int | None = None

    @property
    def anchor(self) -> RM65ToB601Anchor | None:
        return self._anchor

    @property
    def last_sequence(self) -> int | None:
        return self._last_sequence

    def clear_anchor(self) -> None:
        self._anchor = None
        self._last_sequence = None

    def capture_anchor(
        self,
        sample: RM65FreeDriveSample,
        *,
        target_qpos_rad: Sequence[float],
    ) -> RM65ToB601Anchor:
        """Capture one fresh source pose and the current B601 target qpos."""

        self._validate_sample_identity(sample)
        self._validate_freshness(sample)
        target_qpos = _finite_qpos(target_qpos_rad, name="target_qpos_rad")
        try:
            mapped_source_pose, _ = self.config.frame_graph.transform_pose_between_frames(
                sample.pose_source_reference_source_controlled,
                source_reference_frame_id=self.config.source_reference_frame_id,
                source_controlled_frame_id=self.config.source_controlled_frame_id,
                target_reference_frame_id=self.config.target_reference_frame_id,
                target_controlled_frame_id=self.config.target_controlled_frame_id,
            )
            target_pose = _pose_array(
                self.model.forward_kinematics(target_qpos),
                name="target_anchor_pose",
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Unable to capture RM65/B601 anchor: {exc}") from exc
        anchor = RM65ToB601Anchor(
            source_sample_hash=sample.semantic_content_hash,
            source_sequence=sample.sequence,
            source_pose_target_reference_target_controlled=tuple(
                tuple(float(item) for item in row) for row in mapped_source_pose
            ),
            target_anchor_qpos_rad=target_qpos,
            target_anchor_pose=tuple(
                tuple(float(item) for item in row) for row in target_pose
            ),
            frame_graph_hash=self.config.frame_graph.semantic_content_hash,
        )
        self._anchor = anchor
        self._last_sequence = sample.sequence
        return anchor

    def retarget(
        self,
        sample: RM65FreeDriveSample,
        *,
        seed_qpos_rad: Sequence[float],
    ) -> RM65ToB601RetargetResult:
        """Return one target qpos, never an older target on rejection."""

        if self._anchor is None:
            return self._rejected(sample, "anchor_not_captured")
        try:
            self._validate_sample_identity(sample)
            if self._last_sequence is not None and sample.sequence <= self._last_sequence:
                raise ValueError(
                    f"sample sequence {sample.sequence} is not newer than "
                    f"{self._last_sequence}"
                )
            # Once a structurally valid sequence is observed, consume it even
            # if freshness or IK rejects it.  This prevents replaying a stale
            # frame after a transient source fault.
            self._last_sequence = sample.sequence
            self._validate_freshness(sample)
            seed = _finite_qpos(seed_qpos_rad, name="seed_qpos_rad")
            mapped_current, _ = self.config.frame_graph.transform_pose_between_frames(
                sample.pose_source_reference_source_controlled,
                source_reference_frame_id=self.config.source_reference_frame_id,
                source_controlled_frame_id=self.config.source_controlled_frame_id,
                target_reference_frame_id=self.config.target_reference_frame_id,
                target_controlled_frame_id=self.config.target_controlled_frame_id,
            )
            mapped_anchor = np.asarray(
                self._anchor.source_pose_target_reference_target_controlled,
                dtype=np.float64,
            )
            target_anchor = np.asarray(self._anchor.target_anchor_pose, dtype=np.float64)
            current = np.asarray(mapped_current, dtype=np.float64)
            delta_translation = (
                np.asarray(self.config.translation_gain_target_reference_xyz)
                * (current[:3, 3] - mapped_anchor[:3, 3])
            )
            delta_rotation = mapped_anchor[:3, :3].T @ current[:3, :3]
            delta_rotvec = (
                Rotation.from_matrix(delta_rotation).as_rotvec()
                * self.config.rotation_gain
            )
            target_pose = target_anchor.copy()
            target_pose[:3, 3] = target_anchor[:3, 3] + delta_translation
            target_pose[:3, :3] = target_anchor[:3, :3] @ Rotation.from_rotvec(
                delta_rotvec
            ).as_matrix()
            target_pose = _pose_array(target_pose, name="target_pose")
            seed_pose = _pose_array(
                self.model.forward_kinematics(seed),
                name="seed_pose",
            )
            seed_rotation_delta = Rotation.from_matrix(
                target_pose[:3, :3] @ seed_pose[:3, :3].T
            ).as_rotvec()
            diagnostics = {
                "target_seed_translation_delta_m": float(
                    np.linalg.norm(target_pose[:3, 3] - seed_pose[:3, 3])
                ),
                "target_seed_orientation_delta_rad": float(
                    np.linalg.norm(seed_rotation_delta)
                ),
            }
            ik_result = self.ik_solver.solve(target_pose, seed_qpos=seed)
            if not isinstance(ik_result, B601IKResult):
                raise TypeError("B601 IK solver returned an invalid result type.")
            if ik_result.success is not True:
                diagnostics["target_pose_candidate"] = tuple(
                    tuple(float(item) for item in row) for row in target_pose
                )
                return self._rejected(
                    sample,
                    f"ik_rejected:{ik_result.status}",
                    ik_result=ik_result,
                    diagnostics=diagnostics,
                )
            qpos = _finite_qpos(ik_result.qpos, name="ik_result.qpos")
        except (TypeError, ValueError, RuntimeError) as exc:
            return self._rejected(sample, f"retarget_rejected:{type(exc).__name__}:{exc}")
        return RM65ToB601RetargetResult(
            sample=sample,
            accepted=True,
            qpos_rad=qpos,
            target_pose=tuple(
                tuple(float(item) for item in row) for row in target_pose
            ),
            ik_result=ik_result,
            rejection_reason=None,
            diagnostics=diagnostics,
        )

    def _validate_sample_identity(self, sample: RM65FreeDriveSample) -> None:
        if not isinstance(sample, RM65FreeDriveSample):
            raise TypeError("sample must be RM65FreeDriveSample.")
        if sample.observation_only is not True:
            raise ValueError("RM65 sample is not observation-only.")
        if sample.source_reference_frame_id != self.config.source_reference_frame_id:
            raise ValueError("RM65 sample reference frame does not match retarget config.")
        if sample.source_controlled_frame_id != self.config.source_controlled_frame_id:
            raise ValueError("RM65 sample controlled frame does not match retarget config.")
        if sample.source_clock_domain_id != self.config.source_clock_domain_id:
            raise ValueError("RM65 sample clock domain does not match retarget config.")

    def _validate_freshness(self, sample: RM65FreeDriveSample) -> None:
        now_wall = _finite_scalar(self.wall_clock(), name="wall_clock")
        now_monotonic = _finite_scalar(
            self.monotonic_clock(),
            name="monotonic_clock",
        )
        source_age = now_wall - sample.source_timestamp_sec
        receive_age = now_monotonic - sample.receive_timestamp_monotonic_sec
        if source_age < -self.config.maximum_future_skew_sec:
            raise ValueError(f"source timestamp is too far in the future: {source_age:.6f}s")
        if receive_age < -self.config.maximum_future_skew_sec:
            raise ValueError(f"receive timestamp is too far in the future: {receive_age:.6f}s")
        if source_age > self.config.maximum_source_age_sec:
            raise ValueError(f"source sample is stale: age={source_age:.6f}s")
        if receive_age > self.config.maximum_receive_age_sec:
            raise ValueError(f"source receipt is stale: age={receive_age:.6f}s")

    @staticmethod
    def _rejected(
        sample: RM65FreeDriveSample,
        reason: str,
        *,
        ik_result: B601IKResult | None = None,
        diagnostics: Mapping[str, object] | None = None,
    ) -> RM65ToB601RetargetResult:
        return RM65ToB601RetargetResult(
            sample=sample,
            accepted=False,
            qpos_rad=None,
            target_pose=None,
            ik_result=ik_result,
            rejection_reason=reason,
            diagnostics={} if diagnostics is None else diagnostics,
        )


__all__ = [
    "RM65_FREE_DRIVE_SAMPLE_HASH_DOMAIN_V1",
    "RM65_FREE_DRIVE_SAMPLE_SCHEMA_V1",
    "RM65_TO_B601_RETARGET_HASH_DOMAIN_V1",
    "RM65_TO_B601_RETARGET_SCHEMA_V1",
    "RM65FreeDriveSample",
    "RM65ToB601Anchor",
    "RM65ToB601RetargetConfig",
    "RM65ToB601RetargetResult",
    "RM65ToB601Retargeter",
]
