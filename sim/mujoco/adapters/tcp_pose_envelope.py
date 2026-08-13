"""MuJoCo binding for the backend-neutral TCP-pose safety envelope.

This adapter owns only the simulator-side frame conversion and the final
pre-IK projection seam.  Profile selection, Home/anchor policy, scenario
lifecycle, statistics, and execution authority remain with the caller.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.target_safety.tcp_pose_envelope import (
    TCP_CONTROLLED_FRAME_ID,
    TCPPoseEnvelopeAnchorReceipt,
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeProjectionFeedback,
    TCPPoseEnvelopeProjectionResult,
    project_tcp_pose_envelope,
)


MUJOCO_TCP_POSE_TARGET_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.sim_real_tcp_pose_envelope_target_binding.development"
)
MUJOCO_TCP_POSE_TOOL_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.sim_real_tcp_pose_envelope_tool_binding.development"
)


class MujocoTCPPoseEnvelopeError(RuntimeError):
    """Fail-closed error raised by an Active MuJoCo target filter."""


def _required_text(value: object, *, name: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _pose(value: object, *, name: str) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 4x4 pose") from exc
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite 4x4 pose")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} must be homogeneous")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} rotation determinant must be +1")
    return pose.copy()


def mujoco_target_frame_binding_hash(
    *,
    reference_frame: str,
    world_from_desk: np.ndarray,
    world_to_base_by_side: Mapping[str, np.ndarray],
) -> str:
    """Hash the transforms defining one MuJoCo target-world binding."""

    frame = _required_text(reference_frame, name="reference_frame")
    transforms = {
        _required_text(side, name="world_to_base_by_side.side"): _pose(
            transform,
            name=f"world_to_base_by_side.{side}",
        ).tolist()
        for side, transform in world_to_base_by_side.items()
    }
    if not transforms:
        raise ValueError("world_to_base_by_side must not be empty")
    return canonical_content_hash(
        {
            "reference_frame": frame,
            "world_from_desk": _pose(
                world_from_desk,
                name="world_from_desk",
            ).tolist(),
            "world_to_base_by_side": transforms,
        },
        domain=MUJOCO_TCP_POSE_TARGET_BINDING_HASH_DOMAIN_DEVELOPMENT,
    )


def mujoco_tool_frame_binding_hashes(
    retargeter: Any,
    *,
    sides: Sequence[str],
) -> dict[str, str]:
    """Hash the simulator retargeter's fixed solver-to-tool transforms."""

    convert = getattr(
        retargeter,
        "tool_tcp_world_target_from_solver_world_target",
        None,
    )
    if not callable(convert):
        raise TypeError("retargeter lacks solver-to-tool_tcp conversion")
    result: dict[str, str] = {}
    identity = np.eye(4, dtype=float)
    for raw_side in sides:
        side = _required_text(raw_side, name="side")
        if side in result:
            raise ValueError(f"duplicate side {side!r}")
        transform = _pose(
            convert(side=side, solver_world_target=identity),
            name=f"retargeter.{side}.solver_to_tool_tcp",
        )
        result[side] = canonical_content_hash(
            {
                "side": side,
                "controlled_frame_id": TCP_CONTROLLED_FRAME_ID,
                "solver_to_tool_tcp": transform.tolist(),
            },
            domain=MUJOCO_TCP_POSE_TOOL_BINDING_HASH_DOMAIN_DEVELOPMENT,
        )
    if not result:
        raise ValueError("sides must not be empty")
    return result


class MujocoTCPPoseEnvelopeTargetFilter:
    """Project solver targets at MuJoCo's existing final pre-IK seam."""

    def __init__(
        self,
        *,
        retargeter: Any,
        profile: TCPPoseEnvelopeProfile,
        target_frame_binding_hash: str,
        execution_epoch: int,
        anchor_receipt: TCPPoseEnvelopeAnchorReceipt | None,
        apply_to_target: bool = False,
    ) -> None:
        if not isinstance(profile, TCPPoseEnvelopeProfile):
            raise TypeError("profile must be TCPPoseEnvelopeProfile")
        target_hash = _required_text(
            target_frame_binding_hash,
            name="target_frame_binding_hash",
        )
        if profile.target_frame_binding_hash != target_hash:
            raise MujocoTCPPoseEnvelopeError(
                "profile target-frame binding does not match MuJoCo binding"
            )
        tool_hashes = mujoco_tool_frame_binding_hashes(
            retargeter,
            sides=profile.enabled_sides,
        )
        for side in profile.enabled_sides:
            if tool_hashes[side] != profile.tool_frame_binding_hash_by_side[side]:
                raise MujocoTCPPoseEnvelopeError(
                    f"profile tool-frame binding does not match MuJoCo for {side!r}"
                )
        epoch = int(execution_epoch)
        if isinstance(execution_epoch, bool) or epoch < 0:
            raise ValueError("execution_epoch must be a non-negative integer")

        self.retargeter = retargeter
        self.profile = profile
        self.apply_to_target = bool(apply_to_target)
        self._target_frame_binding_hash = target_hash
        self._tool_frame_binding_hash_by_side = tool_hashes
        self._execution_epoch = epoch
        self._anchor_receipt = anchor_receipt
        self._lock = threading.Lock()
        self._frame_count = 0
        self._last_projection: TCPPoseEnvelopeProjectionResult | None = None
        self._last_error: str | None = None

    @property
    def projection_count(self) -> int:
        with self._lock:
            return int(self._frame_count)

    @property
    def last_projection(self) -> TCPPoseEnvelopeProjectionResult | None:
        with self._lock:
            return self._last_projection

    def rebind(
        self,
        *,
        execution_epoch: int,
        anchor_receipt: TCPPoseEnvelopeAnchorReceipt | None,
    ) -> None:
        """Replace a relative profile's anchor while simulated motion is held."""

        epoch = int(execution_epoch)
        if isinstance(execution_epoch, bool) or epoch < 0:
            raise ValueError("execution_epoch must be a non-negative integer")
        required = set(self.profile.anchor_required_sides)
        if required:
            if anchor_receipt is None:
                raise MujocoTCPPoseEnvelopeError(
                    f"TCP envelope anchor is required for sides {sorted(required)}"
                )
            if anchor_receipt.execution_epoch != epoch:
                raise MujocoTCPPoseEnvelopeError(
                    "TCP envelope anchor execution epoch does not match rebind epoch"
                )
            if not required.issubset(anchor_receipt.tcp_pose_by_side):
                raise MujocoTCPPoseEnvelopeError(
                    "TCP envelope anchor is missing one or more required sides"
                )
        with self._lock:
            self._execution_epoch = epoch
            self._anchor_receipt = anchor_receipt
            self._last_projection = None
            self._last_error = None

    def __call__(
        self,
        solver_targets_by_side: Mapping[str, np.ndarray],
        *_unused: Any,
    ) -> Mapping[str, np.ndarray]:
        with self._lock:
            epoch = self._execution_epoch
            anchor = self._anchor_receipt
        try:
            tool_targets = {
                side: _pose(
                    self.retargeter.tool_tcp_world_target_from_solver_world_target(
                        side=side,
                        solver_world_target=target,
                    ),
                    name=f"tool_tcp_target.{side}",
                )
                for side, target in solver_targets_by_side.items()
            }
            projection = project_tcp_pose_envelope(
                tool_targets,
                profile=self.profile,
                target_frame_binding_hash=self._target_frame_binding_hash,
                tool_frame_binding_hash_by_side=(
                    self._tool_frame_binding_hash_by_side
                ),
                controlled_frame_id=TCP_CONTROLLED_FRAME_ID,
                reference_frame_by_side=self.profile.reference_frame_by_side,
                execution_epoch=epoch,
                anchor_receipt=anchor,
            )
            output = {
                side: np.asarray(target, dtype=float).copy()
                for side, target in solver_targets_by_side.items()
            }
            for side in projection.modified_sides:
                output[side] = _pose(
                    self.retargeter.solver_world_target_from_tool_tcp_world_target(
                        side=side,
                        tool_tcp_world_target=(
                            projection.effective_tcp_pose_by_side[side]
                        ),
                    ),
                    name=f"effective_solver_target.{side}",
                )
        except Exception as exc:  # noqa: BLE001 - Shadow observes; Active stops.
            error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._frame_count += 1
                self._last_projection = None
                self._last_error = error
            if self.apply_to_target:
                if isinstance(exc, MujocoTCPPoseEnvelopeError):
                    raise
                raise MujocoTCPPoseEnvelopeError(error) from exc
            return solver_targets_by_side

        with self._lock:
            self._frame_count += 1
            self._last_projection = projection
            self._last_error = None
        if not self.apply_to_target:
            return solver_targets_by_side
        return output

    def feedback_snapshot(self) -> TCPPoseEnvelopeProjectionFeedback:
        """Return compact passive feedback without transferring authority."""

        with self._lock:
            anchor_hash = (
                None
                if self._anchor_receipt is None
                else self._anchor_receipt.semantic_content_hash
            )
            bound = bool(
                not self.profile.anchor_required_sides
                or self._anchor_receipt is not None
            )
            return TCPPoseEnvelopeProjectionFeedback.from_evaluation(
                apply_to_target=self.apply_to_target,
                bound=bound,
                profile_id=self.profile.profile_id,
                profile_hash=self.profile.semantic_content_hash,
                execution_epoch=self._execution_epoch,
                anchor_hash=anchor_hash,
                frame_count=self._frame_count,
                target_role="final_pre_ik",
                projection=self._last_projection,
                error=self._last_error,
            )

    def command_metadata(self) -> dict[str, object]:
        return self.feedback_snapshot().to_payload()


def install_mujoco_tcp_pose_envelope_filter(
    *,
    retargeter: Any,
    profile: TCPPoseEnvelopeProfile,
    target_frame_binding_hash: str,
    execution_epoch: int,
    anchor_receipt: TCPPoseEnvelopeAnchorReceipt | None,
    apply_to_target: bool = False,
) -> MujocoTCPPoseEnvelopeTargetFilter:
    """Install one filter without replacing an existing safety owner."""

    if getattr(retargeter, "pre_ik_target_safety_filter", None) is not None:
        raise MujocoTCPPoseEnvelopeError(
            "retargeter already has a final pre-IK target safety owner"
        )
    target_filter = MujocoTCPPoseEnvelopeTargetFilter(
        retargeter=retargeter,
        profile=profile,
        target_frame_binding_hash=target_frame_binding_hash,
        execution_epoch=execution_epoch,
        anchor_receipt=anchor_receipt,
        apply_to_target=apply_to_target,
    )
    retargeter.pre_ik_target_safety_filter = target_filter
    return target_filter


__all__ = [
    "MUJOCO_TCP_POSE_TARGET_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "MUJOCO_TCP_POSE_TOOL_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "MujocoTCPPoseEnvelopeError",
    "MujocoTCPPoseEnvelopeTargetFilter",
    "install_mujoco_tcp_pose_envelope_filter",
    "mujoco_target_frame_binding_hash",
    "mujoco_tool_frame_binding_hashes",
]
