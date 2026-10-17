"""Final TCP-envelope admission for explicitly stepped MuJoCo physics.

The position executor remains the only command/actuator owner. This adapter
queries the existing envelope and a copied held-control response; it never
advances an IK cache, live physics, a reference cursor, or a transport receipt.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np

from ..control import DirectQposCommand
from .tcp_pose_envelope import (
    MujocoTCPPoseEnvelopeError,
    MujocoTCPPoseEnvelopeTargetFilter,
)


class MujocoTCPPoseCommandGuard:
    """Shorten one admitted joint segment using the exact bound TCP profile."""

    def __init__(
        self,
        *,
        target_filter: MujocoTCPPoseEnvelopeTargetFilter,
        fk_by_side: Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]],
        max_preview_time_sec: float = 1.0,
    ) -> None:
        if not isinstance(target_filter, MujocoTCPPoseEnvelopeTargetFilter):
            raise TypeError("target_filter must be a MuJoCo TCP envelope filter")
        if not target_filter.apply_to_target:
            raise MujocoTCPPoseEnvelopeError("final command admission requires Active projection")
        duration = float(max_preview_time_sec)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("max_preview_time_sec must be finite and positive")
        if not callable(fk_by_side):
            raise TypeError("fk_by_side must be callable")
        self._target_filter = target_filter
        self._fk = fk_by_side
        self._max_preview_time_sec = duration
        self._binding = self._current_binding()
        self._admission_count = 0
        self._modified_count = 0
        self._preview_count = 0

    def _current_binding(self) -> dict[str, object]:
        feedback = self._target_filter.feedback_snapshot()
        return {
            "profile_hash": feedback.profile_hash,
            "anchor_hash": feedback.anchor_hash,
            "execution_epoch": feedback.execution_epoch,
            "max_preview_time_sec": self._max_preview_time_sec,
        }

    def _assert_binding(self) -> None:
        if (
            self._target_filter.retargeter.pre_ik_target_safety_filter is not self._target_filter
            or not self._target_filter.apply_to_target
            or self._current_binding() != self._binding
        ):
            raise MujocoTCPPoseEnvelopeError("final command guard binding changed; rearm is required")

    def _contains(self, qpos_by_side: Mapping[str, np.ndarray]) -> bool:
        self._assert_binding()
        return self._target_filter.contains_tcp_pose_pair(self._fk(qpos_by_side))

    def check_qpos_pair(self, qpos_by_side: Mapping[str, np.ndarray]) -> None:
        if not self._contains(qpos_by_side):
            raise MujocoTCPPoseEnvelopeError("measured TCP pose exceeds its final command envelope")

    def tcp_pose_pair(self, qpos_by_side: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Read FK using the guard's isolated data, including for diagnostics."""

        self._assert_binding()
        return self._fk(qpos_by_side)

    def admission_fraction(
        self,
        command: DirectQposCommand,
        *,
        previous_qpos_by_side: Mapping[str, np.ndarray],
        preview_held_command: Callable[..., bool],
    ) -> float:
        """Choose only a tested prefix, including the delayed stopping response."""

        self._assert_binding()
        self._admission_count += 1

        def safe(alpha: float) -> bool:
            candidate = replace(command, qpos_by_side={
                s: np.asarray(q) + alpha * (command.qpos_by_side[s] - q)
                for s, q in previous_qpos_by_side.items()
            })
            if not self._contains(candidate.qpos_by_side):
                return False
            self._preview_count += 1
            return preview_held_command(
                candidate,
                qpos_check=self._contains,
                max_preview_time_sec=self._max_preview_time_sec,
            )

        if safe(1.0):
            return 1.0
        if not safe(0.0):
            raise MujocoTCPPoseEnvelopeError("no safe held-control response within the preview bound")
        low, high = 0.0, 1.0
        # The safe set in joint space need not be convex. Retain only an
        # explicitly tested candidate; do not infer that an untested endpoint
        # or every point between two successful queries is safe.
        for _ in range(16):
            middle = (low + high) / 2.0
            if safe(middle):
                low = middle
            else:
                high = middle
        self._modified_count += 1
        return low

    def snapshot(self) -> dict[str, object]:
        return {
            "binding": dict(self._binding),
            "mode": "discrete_native_held_control_preview",
            "admission_count": self._admission_count,
            "modified_count": self._modified_count,
            "preview_count": self._preview_count,
            "swept_tcp_certified": False,
        }


def install_mujoco_tcp_pose_command_guard(
    *,
    rig,
    target_filter: MujocoTCPPoseEnvelopeTargetFilter,
    max_preview_time_sec: float = 1.0,
) -> MujocoTCPPoseCommandGuard:
    """Bind a private FK data buffer and the existing physical executor port."""

    retargeter = target_filter.retargeter
    solver = retargeter.ik_solver
    model = solver.robot.model
    data = model.createData()
    base_q = solver.current_q.copy()
    frame_ids = {
        s: model.getFrameId(getattr(retargeter.profile, f"{s}_frame_name"))
        for s in ("left", "right")
    }
    if any(i >= len(model.frames) for i in frame_ids.values()):
        raise ValueError("TCP envelope FK frame is absent from the active solver model")

    def fk(qpos_by_side: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        q = retargeter.solver_q_from_qpos_by_side(qpos_by_side, base_solver_q=base_q)
        solver.pin.framesForwardKinematics(model, data, q)
        return {
            s: retargeter.tool_tcp_world_target_from_solver_world_target(
                side=s, solver_world_target=data.oMf[i].homogeneous,
            )
            for s, i in frame_ids.items()
        }

    guard = MujocoTCPPoseCommandGuard(
        target_filter=target_filter,
        fk_by_side=fk,
        max_preview_time_sec=max_preview_time_sec,
    )
    rig.install_final_command_guard(guard)
    return guard


__all__ = ["MujocoTCPPoseCommandGuard", "install_mujoco_tcp_pose_command_guard"]
