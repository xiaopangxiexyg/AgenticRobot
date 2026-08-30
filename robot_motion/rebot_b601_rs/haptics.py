"""Backend-neutral B601-RS spatial-wrench and virtual-fixture math.

The functions in this module perform no I/O and own no actuator.  Wrenches use
SI units and the order ``(fx, fy, fz, tx, ty, tz)``.  The B601 Jacobian-facing
frame is world/base-axis aligned at the ``gripper_end`` origin, matching the
``LOCAL_WORLD_ALIGNED`` Jacobian exposed by :mod:`.model`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .model import B601PinocchioModel

B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME = "base_link/world_aligned@gripper_end"
B601_RS_WRENCH_ORDER = ("fx", "fy", "fz", "tx", "ty", "tz")


def _finite_vector(
    values: Sequence[float],
    *,
    size: int,
    name: str,
) -> tuple[float, ...]:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric sequence") from exc
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return tuple(float(value) for value in array)


def _non_negative_vector(
    values: Sequence[float],
    *,
    size: int,
    name: str,
    strictly_positive: bool = False,
) -> tuple[float, ...]:
    normalized = _finite_vector(values, size=size, name=name)
    array = np.asarray(normalized, dtype=float)
    invalid = array <= 0.0 if strictly_positive else array < 0.0
    if np.any(invalid):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must contain {size} {qualifier} values")
    return normalized


def _finite_scalar(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _non_negative_scalar(
    value: float,
    *,
    name: str,
    strictly_positive: bool = False,
) -> float:
    normalized = _finite_scalar(value, name=name)
    invalid = normalized <= 0.0 if strictly_positive else normalized < 0.0
    if invalid:
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return normalized


def _frame_id(value: str, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _proper_transform(
    target_T_source: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    try:
        transform = np.asarray(target_T_source, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_T_source must be a numeric 4-by-4 matrix") from exc
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("target_T_source must be a finite 4-by-4 matrix")
    rotation = transform[:3, :3]
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12):
        raise ValueError("target_T_source must have a homogeneous final row")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
        raise ValueError("target_T_source rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9):
        raise ValueError("target_T_source rotation must be right handed")
    return transform.copy()


@dataclass(frozen=True)
class B601SpatialWrench:
    """Force and torque about one named frame origin."""

    force_n: tuple[float, ...]
    torque_nm: tuple[float, ...]
    frame_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "force_n",
            _finite_vector(self.force_n, size=3, name="force_n"),
        )
        object.__setattr__(
            self,
            "torque_nm",
            _finite_vector(self.torque_nm, size=3, name="torque_nm"),
        )
        object.__setattr__(
            self,
            "frame_id",
            _frame_id(self.frame_id, name="frame_id"),
        )

    def as_vector(self) -> np.ndarray:
        """Return a new ``(force, torque)`` vector."""

        return np.asarray((*self.force_n, *self.torque_nm), dtype=float)


@dataclass(frozen=True)
class B601WrenchJointTorque:
    """Auditable result of the B601 ``J.T @ wrench`` mapping."""

    joint_torque_nm: tuple[float, ...]
    cartesian_twist: tuple[float, ...] | None
    cartesian_power_w: float | None
    joint_power_w: float | None


@dataclass(frozen=True)
class B601VirtualSpringResult:
    wrench: B601SpatialWrench
    displacement_m: tuple[float, ...]
    raw_force_n: tuple[float, ...]
    force_limited: bool


@dataclass(frozen=True)
class B601PoseVirtualSpringResult:
    """Auditable six-axis restoring wrench about one fixed SE(3) anchor."""

    wrench: B601SpatialWrench
    displacement_m: tuple[float, ...]
    rotation_error_rad: tuple[float, ...]
    raw_force_n: tuple[float, ...]
    raw_torque_nm: tuple[float, ...]
    force_limited: bool
    torque_limited: bool


@dataclass(frozen=True)
class B601VirtualWallResult:
    wrench: B601SpatialWrench
    penetration_m: float
    normal_velocity_m_s: float
    raw_restoring_force_n: float
    force_limited: bool


def transform_spatial_wrench(
    wrench: B601SpatialWrench,
    *,
    target_frame_id: str,
    target_T_source: Sequence[Sequence[float]] | np.ndarray,
) -> B601SpatialWrench:
    """Apply ``Ad(target_T_source)^-T`` to a spatial wrench.

    ``target_T_source`` expresses the source frame in the target frame.  The
    output torque is about the target origin, so translation contributes
    ``p x f`` in addition to the rotated source torque.
    """

    if not isinstance(wrench, B601SpatialWrench):
        raise TypeError("wrench must be a B601SpatialWrench")
    target = _frame_id(target_frame_id, name="target_frame_id")
    transform = _proper_transform(target_T_source)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    force = rotation @ np.asarray(wrench.force_n, dtype=float)
    torque = np.cross(translation, force) + rotation @ np.asarray(
        wrench.torque_nm, dtype=float
    )
    return B601SpatialWrench(
        force_n=tuple(float(value) for value in force),
        torque_nm=tuple(float(value) for value in torque),
        frame_id=target,
    )


def transform_spatial_twist(
    twist: Sequence[float],
    *,
    target_T_source: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Apply ``Ad(target_T_source)`` to a ``(linear, angular)`` twist."""

    values = np.asarray(
        _finite_vector(twist, size=6, name="twist"),
        dtype=float,
    )
    transform = _proper_transform(target_T_source)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    angular = rotation @ values[3:]
    linear = rotation @ values[:3] + np.cross(translation, angular)
    return np.concatenate((linear, angular))


def map_world_aligned_ee_wrench_to_joint_torque(
    model: B601PinocchioModel,
    qpos_rad: Sequence[float],
    wrench: B601SpatialWrench,
    *,
    qvel_rad_s: Sequence[float] | None = None,
) -> B601WrenchJointTorque:
    """Map one world-aligned gripper-origin wrench to six joint torques."""

    if not isinstance(model, B601PinocchioModel):
        raise TypeError("model must be a B601PinocchioModel")
    if not isinstance(wrench, B601SpatialWrench):
        raise TypeError("wrench must be a B601SpatialWrench")
    if wrench.frame_id != B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME:
        raise ValueError("wrench frame must match the B601 world-aligned gripper frame")
    qpos = _finite_vector(qpos_rad, size=6, name="qpos_rad")
    jacobian = model.arm_jacobian(qpos)
    wrench_vector = wrench.as_vector()
    joint_torque = jacobian.T @ wrench_vector
    if not np.all(np.isfinite(joint_torque)):
        raise ValueError("computed joint torque contains non-finite values")

    cartesian_twist: tuple[float, ...] | None = None
    cartesian_power: float | None = None
    joint_power: float | None = None
    if qvel_rad_s is not None:
        qvel = np.asarray(
            _finite_vector(qvel_rad_s, size=6, name="qvel_rad_s"),
            dtype=float,
        )
        twist = jacobian @ qvel
        cartesian_power = float(wrench_vector @ twist)
        joint_power = float(joint_torque @ qvel)
        if not math.isfinite(cartesian_power) or not math.isfinite(joint_power):
            raise ValueError("computed wrench power contains non-finite values")
        cartesian_twist = tuple(float(value) for value in twist)

    return B601WrenchJointTorque(
        joint_torque_nm=tuple(float(value) for value in joint_torque),
        cartesian_twist=cartesian_twist,
        cartesian_power_w=cartesian_power,
        joint_power_w=joint_power,
    )


@dataclass(frozen=True)
class B601TranslationalVirtualSpring:
    """Axis-wise Cartesian spring and damper at the gripper origin."""

    anchor_position_m: tuple[float, ...]
    stiffness_n_per_m: tuple[float, ...]
    damping_n_s_per_m: tuple[float, ...]
    max_abs_force_n: tuple[float, ...]
    frame_id: str = B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "anchor_position_m",
            _finite_vector(
                self.anchor_position_m,
                size=3,
                name="anchor_position_m",
            ),
        )
        object.__setattr__(
            self,
            "stiffness_n_per_m",
            _non_negative_vector(
                self.stiffness_n_per_m,
                size=3,
                name="stiffness_n_per_m",
            ),
        )
        object.__setattr__(
            self,
            "damping_n_s_per_m",
            _non_negative_vector(
                self.damping_n_s_per_m,
                size=3,
                name="damping_n_s_per_m",
            ),
        )
        object.__setattr__(
            self,
            "max_abs_force_n",
            _non_negative_vector(
                self.max_abs_force_n,
                size=3,
                name="max_abs_force_n",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "frame_id",
            _frame_id(self.frame_id, name="frame_id"),
        )

    def evaluate(
        self,
        position_m: Sequence[float],
        linear_velocity_m_s: Sequence[float],
    ) -> B601VirtualSpringResult:
        position = np.asarray(
            _finite_vector(position_m, size=3, name="position_m"),
            dtype=float,
        )
        velocity = np.asarray(
            _finite_vector(
                linear_velocity_m_s,
                size=3,
                name="linear_velocity_m_s",
            ),
            dtype=float,
        )
        displacement = position - np.asarray(self.anchor_position_m, dtype=float)
        raw_force = -(
            np.asarray(self.stiffness_n_per_m, dtype=float) * displacement
            + np.asarray(self.damping_n_s_per_m, dtype=float) * velocity
        )
        limit = np.asarray(self.max_abs_force_n, dtype=float)
        force = np.clip(raw_force, -limit, limit)
        return B601VirtualSpringResult(
            wrench=B601SpatialWrench(
                force_n=tuple(float(value) for value in force),
                torque_nm=(0.0, 0.0, 0.0),
                frame_id=self.frame_id,
            ),
            displacement_m=tuple(float(value) for value in displacement),
            raw_force_n=tuple(float(value) for value in raw_force),
            force_limited=not np.array_equal(raw_force, force),
        )


@dataclass(frozen=True)
class B601PoseVirtualSpring:
    """Axis-wise world-aligned SE(3) spring and damper at the gripper origin.

    ``anchor_pose`` and the pose passed to :meth:`evaluate` are
    ``base_link_T_gripper_end``.  The rotational error is the shortest
    world-aligned rotation vector of ``R_current @ R_anchor.T``.  Therefore a
    positive pose displacement produces a negative, restoring wrench in the
    same ``LOCAL_WORLD_ALIGNED`` convention as the B601 Jacobian.
    """

    anchor_pose: Sequence[Sequence[float]] | np.ndarray
    translational_stiffness_n_per_m: tuple[float, ...]
    translational_damping_n_s_per_m: tuple[float, ...]
    rotational_stiffness_nm_per_rad: tuple[float, ...]
    rotational_damping_nm_s_per_rad: tuple[float, ...]
    max_abs_force_n: tuple[float, ...]
    max_abs_torque_nm: tuple[float, ...]
    frame_id: str = B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME

    def __post_init__(self) -> None:
        anchor = _proper_transform(self.anchor_pose)
        object.__setattr__(
            self,
            "anchor_pose",
            tuple(tuple(float(value) for value in row) for row in anchor),
        )
        for name in (
            "translational_stiffness_n_per_m",
            "translational_damping_n_s_per_m",
            "rotational_stiffness_nm_per_rad",
            "rotational_damping_nm_s_per_rad",
        ):
            object.__setattr__(
                self,
                name,
                _non_negative_vector(
                    getattr(self, name),
                    size=3,
                    name=name,
                ),
            )
        object.__setattr__(
            self,
            "max_abs_force_n",
            _non_negative_vector(
                self.max_abs_force_n,
                size=3,
                name="max_abs_force_n",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "max_abs_torque_nm",
            _non_negative_vector(
                self.max_abs_torque_nm,
                size=3,
                name="max_abs_torque_nm",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "frame_id",
            _frame_id(self.frame_id, name="frame_id"),
        )

    def evaluate(
        self,
        pose: Sequence[Sequence[float]] | np.ndarray,
        spatial_velocity: Sequence[float],
    ) -> B601PoseVirtualSpringResult:
        current = _proper_transform(pose)
        anchor = np.asarray(self.anchor_pose, dtype=float)
        velocity = np.asarray(
            _finite_vector(spatial_velocity, size=6, name="spatial_velocity"),
            dtype=float,
        )
        displacement = current[:3, 3] - anchor[:3, 3]
        rotation_error = Rotation.from_matrix(
            current[:3, :3] @ anchor[:3, :3].T
        ).as_rotvec()
        raw_force = -(
            np.asarray(self.translational_stiffness_n_per_m, dtype=float)
            * displacement
            + np.asarray(self.translational_damping_n_s_per_m, dtype=float)
            * velocity[:3]
        )
        raw_torque = -(
            np.asarray(self.rotational_stiffness_nm_per_rad, dtype=float)
            * rotation_error
            + np.asarray(self.rotational_damping_nm_s_per_rad, dtype=float)
            * velocity[3:]
        )
        force_limit = np.asarray(self.max_abs_force_n, dtype=float)
        torque_limit = np.asarray(self.max_abs_torque_nm, dtype=float)
        force = np.clip(raw_force, -force_limit, force_limit)
        torque = np.clip(raw_torque, -torque_limit, torque_limit)
        return B601PoseVirtualSpringResult(
            wrench=B601SpatialWrench(
                force_n=tuple(float(value) for value in force),
                torque_nm=tuple(float(value) for value in torque),
                frame_id=self.frame_id,
            ),
            displacement_m=tuple(float(value) for value in displacement),
            rotation_error_rad=tuple(float(value) for value in rotation_error),
            raw_force_n=tuple(float(value) for value in raw_force),
            raw_torque_nm=tuple(float(value) for value in raw_torque),
            force_limited=not np.array_equal(raw_force, force),
            torque_limited=not np.array_equal(raw_torque, torque),
        )


@dataclass(frozen=True)
class B601VirtualWall:
    """One-sided Cartesian wall with forbidden half-space ``n.p > offset``."""

    normal: tuple[float, ...]
    offset_m: float
    stiffness_n_per_m: float
    damping_n_s_per_m: float
    max_force_n: float
    frame_id: str = B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME

    def __post_init__(self) -> None:
        normal = np.asarray(
            _finite_vector(self.normal, size=3, name="normal"),
            dtype=float,
        )
        norm = float(np.linalg.norm(normal))
        if norm <= 0.0:
            raise ValueError("normal must be non-zero")
        normal /= norm
        stiffness = _non_negative_scalar(
            self.stiffness_n_per_m,
            name="stiffness_n_per_m",
        )
        damping = _non_negative_scalar(
            self.damping_n_s_per_m,
            name="damping_n_s_per_m",
        )
        if stiffness == 0.0 and damping == 0.0:
            raise ValueError("wall stiffness or damping must be positive")
        object.__setattr__(self, "normal", tuple(float(value) for value in normal))
        object.__setattr__(
            self,
            "offset_m",
            _finite_scalar(self.offset_m, name="offset_m"),
        )
        object.__setattr__(self, "stiffness_n_per_m", stiffness)
        object.__setattr__(self, "damping_n_s_per_m", damping)
        object.__setattr__(
            self,
            "max_force_n",
            _non_negative_scalar(
                self.max_force_n,
                name="max_force_n",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "frame_id",
            _frame_id(self.frame_id, name="frame_id"),
        )

    def evaluate(
        self,
        position_m: Sequence[float],
        linear_velocity_m_s: Sequence[float],
    ) -> B601VirtualWallResult:
        position = np.asarray(
            _finite_vector(position_m, size=3, name="position_m"),
            dtype=float,
        )
        velocity = np.asarray(
            _finite_vector(
                linear_velocity_m_s,
                size=3,
                name="linear_velocity_m_s",
            ),
            dtype=float,
        )
        normal = np.asarray(self.normal, dtype=float)
        penetration = max(float(normal @ position) - self.offset_m, 0.0)
        normal_velocity = float(normal @ velocity)
        raw_restoring = (
            self.stiffness_n_per_m * penetration
            + self.damping_n_s_per_m * normal_velocity
        )
        # The unilateral wall can repel but never pull the handle into itself.
        raw_restoring = max(raw_restoring, 0.0) if penetration > 0.0 else 0.0
        restoring = min(raw_restoring, self.max_force_n)
        force = -restoring * normal
        return B601VirtualWallResult(
            wrench=B601SpatialWrench(
                force_n=tuple(float(value) for value in force),
                torque_nm=(0.0, 0.0, 0.0),
                frame_id=self.frame_id,
            ),
            penetration_m=penetration,
            normal_velocity_m_s=normal_velocity,
            raw_restoring_force_n=raw_restoring,
            force_limited=raw_restoring > restoring,
        )


__all__ = [
    "B601_RS_WORLD_ALIGNED_EE_WRENCH_FRAME",
    "B601_RS_WRENCH_ORDER",
    "B601PoseVirtualSpring",
    "B601PoseVirtualSpringResult",
    "B601SpatialWrench",
    "B601TranslationalVirtualSpring",
    "B601VirtualSpringResult",
    "B601VirtualWall",
    "B601VirtualWallResult",
    "B601WrenchJointTorque",
    "map_world_aligned_ee_wrench_to_joint_torque",
    "transform_spatial_twist",
    "transform_spatial_wrench",
]
