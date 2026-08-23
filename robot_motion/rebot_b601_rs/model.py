"""Pure Pinocchio model access for the reBotArm B601-RS.

This module parses an explicitly supplied URDF.  It does not import the vendor
Python package, discover devices, open CAN, or register a runtime backend.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType

import numpy as np

B601_RS_ARM_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)
B601_RS_AUXILIARY_JOINT_NAMES = (
    "joint_left",
    "joint_right",
)
B601_RS_BASE_FRAME = "base_link"
B601_RS_END_EFFECTOR_FRAME = "gripper_end"
B601_RS_URDF_RELATIVE_PATH = Path(
    "urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf"
)
B601_RS_JACOBIAN_REFERENCE = "LOCAL_WORLD_ALIGNED"
B601_RS_TWIST_ORDER = ("vx", "vy", "vz", "wx", "wy", "wz")


class B601ModelError(RuntimeError):
    """The supplied model does not satisfy the B601-RS adapter contract."""


class B601ModelDependencyError(RuntimeError):
    """A required model-only dependency is unavailable."""


@dataclass(frozen=True)
class B601SingularityMetrics:
    """SVD diagnostics for the 6-by-6 geometric arm Jacobian."""

    singular_values: tuple[float, ...]
    sigma_min: float
    sigma_max: float
    condition_number: float
    rank: int
    rank_tolerance: float
    characteristic_length_m: float | None


def _load_pinocchio() -> ModuleType:
    try:
        return import_module("pinocchio")
    except ModuleNotFoundError as exc:
        raise B601ModelDependencyError(
            "Pinocchio is required to construct a B601PinocchioModel"
        ) from exc


def _resolve_existing_urdf(urdf_path: str | Path) -> Path:
    try:
        path = Path(urdf_path).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise FileNotFoundError(f"B601-RS URDF not found: {urdf_path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"B601-RS URDF is not a regular file: {path}")
    if path.suffix.lower() != ".urdf":
        raise B601ModelError(f"B601-RS model path must end in .urdf: {path}")
    return path


def _resolve_rebot_source_urdf(source_root: str | Path) -> Path:
    try:
        root = Path(source_root).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise FileNotFoundError(
            f"reBotArm source root not found: {source_root}"
        ) from exc
    if not root.is_dir():
        raise FileNotFoundError(f"reBotArm source root is not a directory: {root}")

    candidate = _resolve_existing_urdf(root / B601_RS_URDF_RELATIVE_PATH)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise B601ModelError(
            "B601-RS URDF resolves outside the supplied reBotArm source root: "
            f"{candidate}"
        ) from exc
    return candidate


class B601PinocchioModel:
    """Fixed-base B601-RS arm kinematics with explicit frame semantics.

    Inputs are the six arm joints in :data:`B601_RS_ARM_JOINT_NAMES` order.
    Gripper joints in the upstream URDF remain at their Pinocchio neutral
    configuration for kinematics.  Gravity evaluation accepts their positions
    explicitly so a simulation backend cannot silently mix a neutral Pinocchio
    model with an open MuJoCo gripper.  Poses are
    ``base_link_T_gripper_end`` in SI units.  The Jacobian maps arm joint
    velocity to a world/base-axis-aligned twist at the ``gripper_end`` origin,
    ordered linear XYZ then angular XYZ.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        end_effector_frame: str = B601_RS_END_EFFECTOR_FRAME,
        arm_joint_names: Sequence[str] = B601_RS_ARM_JOINT_NAMES,
    ) -> None:
        self._urdf_path = _resolve_existing_urdf(urdf_path)
        self._pin = _load_pinocchio()

        joint_names = tuple(str(name) for name in arm_joint_names)
        if len(joint_names) != 6 or len(set(joint_names)) != 6:
            raise B601ModelError(
                "B601-RS arm_joint_names must contain six unique joint names"
            )
        if not end_effector_frame:
            raise B601ModelError("B601-RS end_effector_frame must not be empty")

        try:
            self._model = self._pin.buildModelFromUrdf(str(self._urdf_path))
        except Exception as exc:
            raise B601ModelError(
                f"failed to build Pinocchio model from {self._urdf_path}"
            ) from exc

        base_frame_id = int(self._model.getFrameId(B601_RS_BASE_FRAME))
        if base_frame_id >= len(self._model.frames):
            raise B601ModelError(
                f"B601-RS base frame is absent: {B601_RS_BASE_FRAME}"
            )
        base_frame = self._model.frames[base_frame_id]
        base_placement = np.asarray(base_frame.placement.homogeneous, dtype=float)
        if int(base_frame.parentJoint) != 0 or not np.allclose(
            base_placement,
            np.eye(4),
            atol=1e-12,
        ):
            raise B601ModelError(
                "B601-RS base_link must be fixed to the Pinocchio world frame"
            )

        frame_id = int(self._model.getFrameId(str(end_effector_frame)))
        if frame_id >= len(self._model.frames):
            raise B601ModelError(
                f"B601-RS end-effector frame is absent: {end_effector_frame}"
            )

        q_indices: list[int] = []
        v_indices: list[int] = []
        for name in joint_names:
            joint_id = int(self._model.getJointId(name))
            if joint_id <= 0 or joint_id >= self._model.njoints:
                raise B601ModelError(f"B601-RS arm joint is absent: {name}")
            joint = self._model.joints[joint_id]
            if int(joint.nq) != 1 or int(joint.nv) != 1:
                raise B601ModelError(
                    f"B601-RS arm joint must have nq=nv=1: {name} "
                    f"has nq={joint.nq}, nv={joint.nv}"
                )
            q_indices.append(int(joint.idx_q))
            v_indices.append(int(joint.idx_v))

        if len(set(q_indices)) != 6 or len(set(v_indices)) != 6:
            raise B601ModelError("B601-RS arm joints do not map to unique coordinates")
        if any(index < 0 or index >= self._model.nq for index in q_indices):
            raise B601ModelError("B601-RS arm joint q index is outside the model")
        if any(index < 0 or index >= self._model.nv for index in v_indices):
            raise B601ModelError("B601-RS arm joint v index is outside the model")

        auxiliary_q_indices: list[int] = []
        auxiliary_v_indices: list[int] = []
        auxiliary_limits: dict[str, tuple[float, float]] = {}
        for name in B601_RS_AUXILIARY_JOINT_NAMES:
            joint_id = int(self._model.getJointId(name))
            if joint_id <= 0 or joint_id >= self._model.njoints:
                raise B601ModelError(f"B601-RS auxiliary joint is absent: {name}")
            joint = self._model.joints[joint_id]
            if int(joint.nq) != 1 or int(joint.nv) != 1:
                raise B601ModelError(
                    f"B601-RS auxiliary joint must have nq=nv=1: {name} "
                    f"has nq={joint.nq}, nv={joint.nv}"
                )
            q_index = int(joint.idx_q)
            v_index = int(joint.idx_v)
            if (
                q_index < 0
                or q_index >= self._model.nq
                or v_index < 0
                or v_index >= self._model.nv
            ):
                raise B601ModelError(
                    "B601-RS auxiliary joint coordinate index is outside the model"
                )
            lower_limit = float(self._model.lowerPositionLimit[q_index])
            upper_limit = float(self._model.upperPositionLimit[q_index])
            if (
                not math.isfinite(lower_limit)
                or not math.isfinite(upper_limit)
                or lower_limit >= upper_limit
            ):
                raise B601ModelError(
                    f"B601-RS auxiliary joint has invalid position limits: {name}"
                )
            auxiliary_q_indices.append(q_index)
            auxiliary_v_indices.append(v_index)
            auxiliary_limits[name] = (lower_limit, upper_limit)

        if (
            len(set(auxiliary_q_indices)) != len(auxiliary_q_indices)
            or len(set(auxiliary_v_indices)) != len(auxiliary_v_indices)
            or set(auxiliary_q_indices).intersection(q_indices)
            or set(auxiliary_v_indices).intersection(v_indices)
        ):
            raise B601ModelError(
                "B601-RS auxiliary joints must use unique coordinates disjoint "
                "from the arm"
            )
        lower = np.asarray(self._model.lowerPositionLimit, dtype=float)[q_indices]
        upper = np.asarray(self._model.upperPositionLimit, dtype=float)[q_indices]
        if (
            lower.shape != (6,)
            or upper.shape != (6,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or not np.all(lower < upper)
        ):
            raise B601ModelError(
                "B601-RS arm joints must have finite increasing position limits"
            )

        self._arm_joint_names = joint_names
        self._arm_q_indices = tuple(q_indices)
        self._arm_v_indices = tuple(v_indices)
        self._arm_lower_position_limits = lower.copy()
        self._arm_upper_position_limits = upper.copy()
        self._auxiliary_joint_names = B601_RS_AUXILIARY_JOINT_NAMES
        self._auxiliary_q_indices = tuple(auxiliary_q_indices)
        self._auxiliary_position_limits = auxiliary_limits
        self._end_effector_frame = str(end_effector_frame)
        self._end_effector_frame_id = frame_id

    @classmethod
    def from_rebot_source_root(
        cls,
        source_root: str | Path,
        *,
        end_effector_frame: str = B601_RS_END_EFFECTOR_FRAME,
        arm_joint_names: Sequence[str] = B601_RS_ARM_JOINT_NAMES,
    ) -> B601PinocchioModel:
        """Construct from the root of an explicit reBotArm source snapshot."""

        return cls(
            _resolve_rebot_source_urdf(source_root),
            end_effector_frame=end_effector_frame,
            arm_joint_names=arm_joint_names,
        )

    @property
    def urdf_path(self) -> Path:
        return self._urdf_path

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        return self._arm_joint_names

    @property
    def arm_q_indices(self) -> tuple[int, ...]:
        return self._arm_q_indices

    @property
    def arm_v_indices(self) -> tuple[int, ...]:
        return self._arm_v_indices

    @property
    def arm_dof(self) -> int:
        return len(self._arm_joint_names)

    @property
    def auxiliary_joint_names(self) -> tuple[str, ...]:
        """Return the non-arm coordinates required for gravity evaluation."""

        return self._auxiliary_joint_names

    @property
    def gravity_mps2(self) -> tuple[float, float, float]:
        """Return the Pinocchio model gravity vector in base/world axes."""

        gravity = np.asarray(self._model.gravity.linear, dtype=float)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise B601ModelError("Pinocchio model has an invalid gravity vector")
        return tuple(float(value) for value in gravity)

    @property
    def full_model_nq(self) -> int:
        return int(self._model.nq)

    @property
    def full_model_nv(self) -> int:
        return int(self._model.nv)

    @property
    def full_model_joint_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self._model.names[1:])

    @property
    def end_effector_frame(self) -> str:
        return self._end_effector_frame

    def arm_position_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Return lower and upper limits in the six-arm-joint order."""

        return (
            self._arm_lower_position_limits.copy(),
            self._arm_upper_position_limits.copy(),
        )

    def auxiliary_position_limits(self) -> dict[str, tuple[float, float]]:
        """Return a copy of the auxiliary joint position-limit mapping."""

        return dict(self._auxiliary_position_limits)

    def arm_generalized_gravity(
        self,
        q_arm: Sequence[float] | np.ndarray,
        *,
        auxiliary_joint_positions: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Return six compensating gravity torques in arm-joint order.

        ``auxiliary_joint_positions=None`` deliberately means the Pinocchio
        neutral configuration.  A backend whose gripper is not neutral must
        pass all names in :attr:`auxiliary_joint_names` explicitly.
        """

        q_full = self._full_configuration(
            q_arm,
            auxiliary_joint_positions=auxiliary_joint_positions,
        )
        data = self._model.createData()
        gravity = np.asarray(
            self._pin.computeGeneralizedGravity(self._model, data, q_full),
            dtype=float,
        )[list(self._arm_v_indices)]
        if gravity.shape != (6,) or not np.all(np.isfinite(gravity)):
            raise B601ModelError(
                "Pinocchio returned invalid B601-RS generalized gravity"
            )
        return gravity.copy()

    def forward_kinematics(self, q_arm: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return ``base_link_T_gripper_end`` as a 4-by-4 matrix."""

        q_full = self._full_configuration(q_arm)
        data = self._model.createData()
        self._pin.forwardKinematics(self._model, data, q_full)
        self._pin.updateFramePlacements(self._model, data)
        placement = data.oMf[self._end_effector_frame_id]
        return np.asarray(placement.homogeneous, dtype=float).copy()

    def arm_jacobian(
        self,
        q_arm: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Return the 6-by-6 LOCAL_WORLD_ALIGNED arm Jacobian.

        Rows are ``(vx, vy, vz, wx, wy, wz)`` and columns follow
        :attr:`arm_joint_names`.
        """

        q_full = self._full_configuration(q_arm)
        data = self._model.createData()
        jacobian = self._pin.computeFrameJacobian(
            self._model,
            data,
            q_full,
            self._end_effector_frame_id,
            self._pin.LOCAL_WORLD_ALIGNED,
        )
        arm_jacobian = np.asarray(jacobian, dtype=float)[
            :, list(self._arm_v_indices)
        ]
        if arm_jacobian.shape != (6, 6) or not np.all(np.isfinite(arm_jacobian)):
            raise B601ModelError(
                "Pinocchio returned an invalid B601-RS arm Jacobian: "
                f"shape={arm_jacobian.shape}"
            )
        return arm_jacobian.copy()

    def singularity_metrics(
        self,
        q_arm: Sequence[float] | np.ndarray,
        *,
        characteristic_length_m: float | None = None,
    ) -> B601SingularityMetrics:
        """Return SVD diagnostics for the geometric arm Jacobian.

        With no characteristic length, linear and angular rows retain their
        native SI units.  When a positive finite length is supplied, the first
        three rows are divided by that length before SVD.  Neither form assigns
        a calibrated safety threshold.
        """

        jacobian = self.arm_jacobian(q_arm)
        length = self._validate_characteristic_length(characteristic_length_m)
        conditioned_jacobian = jacobian.copy()
        if length is not None:
            conditioned_jacobian[:3, :] /= length
        singular_values = np.linalg.svd(conditioned_jacobian, compute_uv=False)
        sigma_max = float(singular_values[0])
        sigma_min = float(singular_values[-1])
        rank_tolerance = float(
            max(jacobian.shape) * np.finfo(float).eps * sigma_max
        )
        rank = int(np.count_nonzero(singular_values > rank_tolerance))
        condition_number = (
            float("inf")
            if sigma_min <= rank_tolerance
            else float(sigma_max / sigma_min)
        )
        return B601SingularityMetrics(
            singular_values=tuple(float(value) for value in singular_values),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            condition_number=condition_number,
            rank=rank,
            rank_tolerance=rank_tolerance,
            characteristic_length_m=length,
        )

    @staticmethod
    def _validate_characteristic_length(
        characteristic_length_m: float | None,
    ) -> float | None:
        if characteristic_length_m is None:
            return None
        try:
            length = float(characteristic_length_m)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "characteristic_length_m must be a positive finite number"
            ) from exc
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError(
                "characteristic_length_m must be a positive finite number"
            )
        return length

    def _full_configuration(
        self,
        q_arm: Sequence[float] | np.ndarray,
        *,
        auxiliary_joint_positions: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        try:
            arm = np.asarray(q_arm, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("q_arm must be a numeric sequence") from exc
        if arm.shape != (6,):
            raise ValueError(f"q_arm must have shape (6,), got {arm.shape}")
        if not np.all(np.isfinite(arm)):
            raise ValueError("q_arm contains non-finite values")

        q_full = np.asarray(self._pin.neutral(self._model), dtype=float).copy()
        q_full[list(self._arm_q_indices)] = arm
        if auxiliary_joint_positions is not None:
            if not isinstance(auxiliary_joint_positions, Mapping):
                raise TypeError("auxiliary_joint_positions must be a mapping")
            supplied_names = set(auxiliary_joint_positions)
            expected_names = set(self._auxiliary_joint_names)
            if supplied_names != expected_names:
                missing = sorted(expected_names - supplied_names)
                unexpected = sorted(supplied_names - expected_names)
                raise ValueError(
                    "auxiliary_joint_positions must contain exactly "
                    f"{self._auxiliary_joint_names}; missing={missing}, "
                    f"unexpected={unexpected}"
                )
            for name, q_index in zip(
                self._auxiliary_joint_names,
                self._auxiliary_q_indices,
                strict=True,
            ):
                try:
                    value = float(auxiliary_joint_positions[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"auxiliary joint {name} position must be numeric"
                    ) from exc
                lower, upper = self._auxiliary_position_limits[name]
                if not math.isfinite(value):
                    raise ValueError(f"auxiliary joint {name} position must be finite")
                if not lower <= value <= upper:
                    raise ValueError(
                        f"auxiliary joint {name} position exceeds its limits"
                    )
                q_full[q_index] = value
        return q_full


__all__ = [
    "B601_RS_ARM_JOINT_NAMES",
    "B601_RS_AUXILIARY_JOINT_NAMES",
    "B601_RS_BASE_FRAME",
    "B601_RS_END_EFFECTOR_FRAME",
    "B601_RS_JACOBIAN_REFERENCE",
    "B601_RS_TWIST_ORDER",
    "B601_RS_URDF_RELATIVE_PATH",
    "B601ModelDependencyError",
    "B601ModelError",
    "B601PinocchioModel",
    "B601SingularityMetrics",
]
