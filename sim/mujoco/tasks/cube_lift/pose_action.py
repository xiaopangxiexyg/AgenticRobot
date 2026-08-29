"""Pose-action adapter for the fitted RM65 Cube-Lift policy.

The policy-facing arm command is a bounded TCP pose increment.  This module
converts that increment to the canonical sim/real-neutral RM65 pose planner's
qref, after which the existing MuJoCo fitted position and gripper actuators
remain unchanged.  It intentionally owns no MuJoCo actuator dynamics and does
not import ``teleop_runtime`` or initialize a vendor device.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from project_paths import resolve_path
from robot_motion.pose import pose6_to_matrix
from robot_motion.rm65 import (
    RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
    RM65PositionCollisionPlanner,
    RM65PositionCollisionPlannerConfig,
)

from ...ik import IKResult, PoseTarget, RM65CanonicalIK
from .config import CubeLiftTaskConfig


# MuJoCo enforces revolute joint ranges through a soft constraint.  A position
# servo can therefore leave a tiny residual (observed here at roughly
# 1e-5--5e-5 rad) outside the canonical RM65 range even though the requested
# qref is valid.  This is a simulator numerical-recovery tolerance only; it
# does not alter the shared planner or any real-robot limit.
SIM_RM65_JOINT_LIMIT_RECOVERY_TOLERANCE_RAD = 1e-4
_FALLBACK_POSE_POSITION_TOLERANCE_M = 2e-3
_FALLBACK_POSE_ORIENTATION_TOLERANCE_RAD = float(np.deg2rad(1.0))
# Keep pose-IK position references deliberately inside the unchanged hard
# limits.  MuJoCo models a revolute range as a soft constraint and the fitted
# position servo can overshoot a reference that lies exactly on the bound.
# One milliradian (0.057 deg) is large compared with the observed numerical
# residual, but negligible compared with the RM65 working range.
SIM_RM65_POSE_QREF_LIMIT_MARGIN_RAD = 1e-3


def inset_joint_position_target(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    margin_rad: float = SIM_RM65_POSE_QREF_LIMIT_MARGIN_RAD,
) -> np.ndarray:
    """Clip a pose-IK qref to an inset of the existing hard joint bounds."""

    values = np.asarray(target, dtype=float)
    minimum = np.asarray(lower, dtype=float)
    maximum = np.asarray(upper, dtype=float)
    if values.shape != minimum.shape or values.shape != maximum.shape:
        raise ValueError("Pose qref and joint-limit arrays must have matching shapes.")
    margin = float(margin_rad)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("Pose qref joint-limit margin must be finite and non-negative.")
    inset_lower = minimum + margin
    inset_upper = maximum - margin
    if np.any(inset_lower > inset_upper):
        raise ValueError("Pose qref joint-limit margin leaves an empty joint range.")
    return np.clip(values, inset_lower, inset_upper)


def _load_endpoint_to_tool_transform(path: str | Path, *, side: str) -> np.ndarray:
    """Load only the TCP transform from a calibration JSON.

    The task config records the profile path, but this simulation adapter reads
    the small JSON payload directly so the ``sim`` package does not need to
    import a vendor hardware backend merely to build its pose contract.
    """

    resolved = resolve_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"RM65 tool profile {resolved} must contain a JSON object.")
    arm = str(payload.get("arm", "")).strip().lower()
    if arm and arm != str(side).strip().lower():
        raise ValueError(
            f"RM65 tool profile {resolved} is for arm {arm!r}, expected {side!r}."
        )
    tcp = payload.get("tcp")
    if not isinstance(tcp, dict):
        raise ValueError(f"RM65 tool profile {resolved} is missing tcp metadata.")
    values = np.asarray(tcp.get("pose_flange_m_rad"), dtype=float).reshape(-1)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"RM65 tool profile {resolved} tcp.pose_flange_m_rad must contain six finite values."
        )
    return pose6_to_matrix(values)


def _arm_base_body_name(scene, side: str) -> str:
    candidates = (
        f"{side}_arm_base_link",
        "arm_base_link",
        "base_link",
    )
    for name in candidates:
        if scene.has_body(name):
            return name
    raise KeyError(f"Could not resolve {side} arm base body; tried={candidates}.")


@dataclass(frozen=True)
class CubeLiftPoseActionStep:
    """One pose-action resolution result before MuJoCo safety projection."""

    target_pose_arm_base: np.ndarray
    ik_result: IKResult


@dataclass(frozen=True)
class CubeLiftPosePlannerSyncStatus:
    """Result of re-anchoring the canonical planner to measured MuJoCo qpos."""

    safe: bool
    recovered: bool
    max_violation_rad: float = 0.0
    violating_side: str | None = None
    violating_joint_index: int | None = None
    failure_reason: str | None = None
    failure_message: str | None = None


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=float)


class _MujocoDampedLeastSquaresIK:
    """Lightweight MuJoCo-Jacobian fallback for isolated worker processes.

    Some conda worker processes cannot load the optional Pinocchio wheel's C++
    ABI even though the parent process can.  Pose semantics must remain
    available in that case, so this fallback performs one bounded DLS update
    against the same RM65 link6 + calibrated TCP transform.  The downstream
    fitted position actuator and full MuJoCo collision projector are unchanged.
    """

    def __init__(self, scene, *, side: str, endpoint_to_tool: np.ndarray) -> None:
        self.scene = scene
        self.side = str(side)
        self.endpoint_to_tool = np.asarray(endpoint_to_tool, dtype=float).reshape(4, 4)
        self.body_name = scene.arm_specs[self.side].controlled_body
        self.body_id = scene.body_id(self.body_name)
        self.qpos_idx = np.asarray(
            [scene.joint_qpos_addr(name) for name in scene.arm_specs[self.side].joint_names],
            dtype=np.int32,
        )
        self.dof_idx = np.asarray(
            [scene.joint_dof_addr(name) for name in scene.arm_specs[self.side].joint_names],
            dtype=np.int32,
        )
        self.lower = np.asarray(
            (-3.1, -1.57079632679, -2.355, -2.443460952792061, -2.233, -2.0943951023931953),
            dtype=float,
        )
        self.upper = -self.lower

    def current_pose_arm_base(self) -> np.ndarray:
        base = self.scene.body_pose(_arm_base_body_name(self.scene, self.side))
        endpoint = self.scene.body_pose(self.body_name)
        return np.linalg.inv(base) @ endpoint @ self.endpoint_to_tool

    def solve(
        self,
        target_pose_arm_base: np.ndarray,
        *,
        current_qpos: np.ndarray,
        max_iterations: int = 3,
        max_joint_step_rad: float = 0.12,
        damping: float = 0.04,
    ) -> tuple[np.ndarray, int, float, float]:
        target = np.asarray(target_pose_arm_base, dtype=float).reshape(4, 4)
        q_reference = np.asarray(current_qpos, dtype=float).reshape(6).copy()
        q = q_reference.copy()
        data = self.scene.data
        mujoco = self.scene.mujoco
        saved_qpos = data.qpos.copy()
        saved_qvel = data.qvel.copy()
        saved_act = data.act.copy()
        saved_ctrl = data.ctrl.copy()
        saved_qacc_warmstart = data.qacc_warmstart.copy()
        iterations = 0
        position_error_m = float("inf")
        orientation_error_rad = float("inf")
        try:
            base = self.scene.body_pose(_arm_base_body_name(self.scene, self.side))
            base_rotation_world = base[:3, :3]
            world_from_base_rotation = base_rotation_world
            base_from_world_rotation = world_from_base_rotation.T
            for iterations in range(1, int(max_iterations) + 1):
                data.qpos[self.qpos_idx] = q
                mujoco.mj_forward(self.scene.model, data)
                current = self.current_pose_arm_base()
                position_error = target[:3, 3] - current[:3, 3]
                rotation_error = Rotation.from_matrix(
                    target[:3, :3] @ current[:3, :3].T
                ).as_rotvec()
                position_error_m = float(np.linalg.norm(position_error))
                orientation_error_rad = float(np.linalg.norm(rotation_error))
                if position_error_m <= 2e-3 and orientation_error_rad <= 0.03:
                    break

                jacobian_position_world = np.zeros((3, self.scene.model.nv), dtype=float)
                jacobian_rotation_world = np.zeros((3, self.scene.model.nv), dtype=float)
                mujoco.mj_jacBody(
                    self.scene.model,
                    data,
                    jacobian_position_world,
                    jacobian_rotation_world,
                    self.body_id,
                )
                endpoint_world = self.scene.body_pose(self.body_name)
                tcp_offset_world = endpoint_world[:3, :3] @ self.endpoint_to_tool[:3, 3]
                tcp_position_world = jacobian_position_world - _skew(tcp_offset_world) @ jacobian_rotation_world
                jacobian = np.vstack(
                    (
                        base_from_world_rotation @ tcp_position_world[:, self.dof_idx],
                        base_from_world_rotation @ jacobian_rotation_world[:, self.dof_idx],
                    )
                )
                weighting = np.diag((1.0, 1.0, 1.0, 0.35, 0.35, 0.35))
                weighted_jacobian = weighting @ jacobian
                weighted_error = weighting @ np.concatenate((position_error, rotation_error))
                normal = weighted_jacobian @ weighted_jacobian.T
                dq = weighted_jacobian.T @ np.linalg.solve(
                    normal + (float(damping) ** 2) * np.eye(6),
                    weighted_error,
                )
                q = np.clip(
                    q + np.clip(dq, -float(max_joint_step_rad), float(max_joint_step_rad)),
                    self.lower,
                    self.upper,
                )
        finally:
            data.qpos[:] = saved_qpos
            data.qvel[:] = saved_qvel
            data.act[:] = saved_act
            data.ctrl[:] = saved_ctrl
            data.qacc_warmstart[:] = saved_qacc_warmstart
            mujoco.mj_forward(self.scene.model, data)
        return q, iterations, position_error_m, orientation_error_rad


class CubeLiftPoseActionController:
    """Resolve normalized TCP pose increments into right-arm qref targets."""

    def __init__(self, scene, config: CubeLiftTaskConfig) -> None:
        if not config.uses_pose_action:
            raise ValueError("CubeLiftPoseActionController requires an actuated task config.")
        self.scene = scene
        self.config = config
        self.side = str(config.controlled_side)
        transforms = {
            "left": _load_endpoint_to_tool_transform(
                config.left_tool_profile_path,
                side="left",
            ),
            "right": _load_endpoint_to_tool_transform(
                config.right_tool_profile_path,
                side="right",
            ),
        }
        # The MuJoCo collision projector is the authoritative sim-side safety
        # check.  Use its configured clearance for this sim planner instead of
        # the real-runtime default (3 cm); otherwise a physically valid sim
        # state (for example 2.7 cm from a proxy pair while the sim policy
        # allows 2 mm) cannot be re-anchored and aborts a worker.  This changes
        # only this sim planner instance; the shared real planner keeps its
        # certified default.
        scene_spec = getattr(scene, "scene_spec", None)
        scene_collision = getattr(scene_spec, "collision", None)
        sim_collision_min_distance = getattr(
            scene_collision,
            "safety_minimum_clearance_m",
            None,
        )
        if sim_collision_min_distance is None:
            sim_collision_min_distance = 0.03
        sim_collision_min_distance = float(sim_collision_min_distance)
        if not np.isfinite(sim_collision_min_distance) or sim_collision_min_distance < 0.0:
            raise ValueError("MuJoCo pose planner collision clearance must be finite and non-negative.")
        planner_config = RM65PositionCollisionPlannerConfig(
            # Policy actions arrive at 15 Hz, while the shared position rig
            # consumes qref at 60 Hz and qcmd at 120 Hz.  The planner remains
            # configured at the certified 60 Hz reference rate; the downstream
            # fitted trajectory executor performs the temporal interpolation.
            reference_hz=float(config.reference_hz),
            max_joint_velocity_rad_s=RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
            solve_max_iterations=2,
            solve_min_iterations=1,
            solve_adaptive_iterations_enabled=False,
            solve_translation_tolerance_m=2e-3,
            solve_orientation_tolerance_deg=1.0,
            collision_min_distance_m=sim_collision_min_distance,
        )
        self.planner_collision_min_distance_m = sim_collision_min_distance
        self._ik: RM65CanonicalIK | None = None
        self._fallback_ik: _MujocoDampedLeastSquaresIK | None = None
        try:
            planner = RM65PositionCollisionPlanner(
                config=planner_config,
                endpoint_to_tool_by_side=transforms,
            )
            self._ik = RM65CanonicalIK(scene, planner)
            self.solver_name = "robot_motion.rm65.position_collision"
        except (ImportError, ModuleNotFoundError, OSError):
            # Keep normal sim workers usable when the optional Pinocchio binary
            # cannot be loaded in a spawned process.  This branch is explicit in
            # metadata and never changes the fitted actuator profile.
            self._fallback_ik = _MujocoDampedLeastSquaresIK(
                scene,
                side=self.side,
                endpoint_to_tool=transforms[self.side],
            )
            self.solver_name = "mujoco.damped_least_squares_pose"
        self._endpoint_to_tool_by_side = transforms
        self._target_pose_arm_base: np.ndarray | None = None
        # Keep the last resolved qref separate from the measured scene qpos.
        # A zero incremental pose action means "hold the current target"; it
        # must not invoke a numerically under-converged IK solve again and
        # again, because equivalent TCP poses can otherwise produce a small
        # different joint branch on every policy tick.
        self._last_resolved_qpos_by_side: dict[str, np.ndarray] | None = None
        # Keep the exact, limit-sanitized qpos used to re-anchor the canonical
        # planner.  MuJoCo's soft joint-range constraint can leave a tiny
        # residual outside the planner limits; retaining the raw measured
        # value here would feed that residual back into the next zero-action
        # solve and make branch selection dependent on solver history.
        self._last_planner_qpos_by_side: dict[str, np.ndarray] | None = None
        self._pose_motion_started = False
        self.last_step: CubeLiftPoseActionStep | None = None
        self._last_planner_sync = CubeLiftPosePlannerSyncStatus(safe=True, recovered=False)

    @property
    def last_planner_sync(self) -> CubeLiftPosePlannerSyncStatus:
        return self._last_planner_sync

    @property
    def target_pose_arm_base(self) -> np.ndarray:
        if self._target_pose_arm_base is None:
            raise RuntimeError("CubeLiftPoseActionController.reset() is required first.")
        return self._target_pose_arm_base.copy()

    def _current_tool_pose_arm_base(self) -> np.ndarray:
        base = self.scene.body_pose(_arm_base_body_name(self.scene, self.side))
        endpoint = self.scene.body_pose(self.scene.arm_specs[self.side].controlled_body)
        return (
            np.linalg.inv(base)
            @ endpoint
            @ self._endpoint_to_tool_by_side[self.side]
        )

    def _effective_joint_limits_by_side(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Read the limits already resolved by the canonical planner.

        The planner intersects its configured RM65 limits with the Pinocchio
        model limits.  Reading that resolved pair here keeps the sim-only
        numerical recovery aligned with the exact planner instance instead of
        duplicating or widening a real-robot limit.
        """

        if self._ik is not None:
            planner = self._ik.planner
            lower = getattr(planner, "_joint_lower_by_side", None)
            upper = getattr(planner, "_joint_upper_by_side", None)
            if isinstance(lower, Mapping) and isinstance(upper, Mapping):
                return (
                    {
                        side: np.asarray(lower[side], dtype=float).reshape(6).copy()
                        for side in ("left", "right")
                    },
                    {
                        side: np.asarray(upper[side], dtype=float).reshape(6).copy()
                        for side in ("left", "right")
                    },
                )

        # The fallback DLS solver is only used when the optional canonical
        # planner cannot load.  It still exposes the same local RM65 bounds for
        # a consistent diagnostic, while no shared planner is re-anchored.
        if self._fallback_ik is not None:
            lower = np.asarray(self._fallback_ik.lower, dtype=float).reshape(6).copy()
            upper = np.asarray(self._fallback_ik.upper, dtype=float).reshape(6).copy()
            return (
                {"left": lower.copy(), "right": lower.copy()},
                {"left": upper.copy(), "right": upper.copy()},
            )
        raise RuntimeError("Pose action controller has no IK solver.")

    def _sanitized_scene_qpos(
        self,
    ) -> tuple[dict[str, np.ndarray], CubeLiftPosePlannerSyncStatus]:
        lower_by_side, upper_by_side = self._effective_joint_limits_by_side()
        observed = {
            side: np.asarray(self.scene.get_arm_qpos(side), dtype=float).reshape(6).copy()
            for side in ("left", "right")
        }
        max_violation = 0.0
        violating_side: str | None = None
        violating_joint_index: int | None = None
        sanitized: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            values = observed[side]
            lower = lower_by_side[side]
            upper = upper_by_side[side]
            if not np.all(np.isfinite(values)):
                return observed, CubeLiftPosePlannerSyncStatus(
                    safe=False,
                    recovered=False,
                    max_violation_rad=float("inf"),
                    violating_side=side,
                    violating_joint_index=None,
                )
            violation = np.maximum(np.maximum(lower - values, values - upper), 0.0)
            index = int(np.argmax(violation))
            amount = float(violation[index])
            if amount > max_violation:
                max_violation = amount
                violating_side = side
                violating_joint_index = index
            sanitized[side] = np.clip(values, lower, upper)

        recovered = max_violation > 1e-10
        safe = max_violation <= SIM_RM65_JOINT_LIMIT_RECOVERY_TOLERANCE_RAD
        return sanitized, CubeLiftPosePlannerSyncStatus(
            safe=safe,
            recovered=recovered and safe,
            max_violation_rad=max_violation,
            violating_side=violating_side,
            violating_joint_index=violating_joint_index,
        )

    def _reset_planner_from_scene(self) -> CubeLiftPosePlannerSyncStatus:
        sanitized, status = self._sanitized_scene_qpos()
        self._last_planner_sync = status
        if not status.safe:
            return status
        if self._ik is not None:
            try:
                self._ik.reset_from_qpos(sanitized)
            except ValueError as exc:
                message = str(exc)
                if "collision planner initial qpos is not certifiably safe" not in message:
                    raise
                reason = "collision_margin_low"
                if "reason=" in message:
                    reason = message.split("reason=", 1)[1].split(",", 1)[0].strip()
                status = CubeLiftPosePlannerSyncStatus(
                    safe=False,
                    recovered=False,
                    max_violation_rad=status.max_violation_rad,
                    violating_side=status.violating_side,
                    violating_joint_index=status.violating_joint_index,
                    failure_reason=reason,
                    failure_message=message,
                )
                self._last_planner_sync = status
                return status
        self._last_planner_qpos_by_side = {
            side: values.copy() for side, values in sanitized.items()
        }
        return status

    def reset(self) -> np.ndarray:
        """Anchor the pose target and canonical planner to the measured scene."""

        status = self._reset_planner_from_scene()
        if not status.safe:
            raise RuntimeError(
                "MuJoCo pose controller reset observed a hard RM65 joint-limit "
                "violation: "
                f"side={status.violating_side} "
                f"joint={status.violating_joint_index} "
                f"excess_rad={status.max_violation_rad:.9g} "
                f"reason={status.failure_reason or 'joint_limit'}."
            )
        self._target_pose_arm_base = self._current_tool_pose_arm_base()
        if self._last_planner_qpos_by_side is None:
            raise RuntimeError(
                "MuJoCo pose controller did not establish a planner qpos anchor."
            )
        self._last_resolved_qpos_by_side = {
            side: values.astype(np.float64, copy=True)
            for side, values in self._last_planner_qpos_by_side.items()
        }
        self._pose_motion_started = False
        self.last_step = None
        return self.target_pose_arm_base

    def sync_planner_from_scene(self) -> CubeLiftPosePlannerSyncStatus:
        """Refresh the planner's measured qpos anchor without discarding target pose."""

        return self._reset_planner_from_scene()

    def reanchor_target_from_scene(self) -> np.ndarray:
        """Drop an unexecuted pose target after IK/safety intervention.

        The planner qpos is synchronized separately.  This method only resets
        the next relative-action anchor to the measured tool pose so rejected
        increments cannot accumulate into an unreachable target.
        """

        sanitized, status = self._sanitized_scene_qpos()
        if not status.safe:
            raise RuntimeError(
                "Cannot re-anchor MuJoCo pose target from an unsafe joint state: "
                f"side={status.violating_side} "
                f"joint={status.violating_joint_index} "
                f"excess_rad={status.max_violation_rad:.9g}."
            )
        self._target_pose_arm_base = self._current_tool_pose_arm_base()
        self._last_resolved_qpos_by_side = {
            side: values.astype(np.float64, copy=True)
            for side, values in sanitized.items()
        }
        return self.target_pose_arm_base

    def begin_policy_episode(self) -> None:
        """Start policy accounting after an optional scripted preparation.

        Scripted pre-grasp/contact preparation uses the same pose-action
        controller as policy actions so it benefits from the canonical IK and
        collision planner.  That preparation is deliberately not part of the
        policy trajectory, however: a zero policy action immediately after the
        preparation must be treated as a stationary hold, not as a request to
        re-solve the preparation target from a different IK branch.  Keep the
        resolved qref/held TCP target intact for servo convergence while
        clearing only the policy-motion phase marker.
        """

        if self._target_pose_arm_base is None:
            raise RuntimeError(
                "CubeLiftPoseActionController.reset() is required first."
            )
        self._pose_motion_started = False
        self.last_step = None

    def _zero_increment_result(self) -> IKResult:
        """Track the held TCP target from a continuity-preserving IK seed.

        A zero arm increment means that the *TCP target* is unchanged; it does
        not mean that a lagging physical servo should be frozen at the last
        qref.  Re-solving from the measured qpos on every gripper-only frame
        can select a nearby equivalent wrist branch and produces visible
        jitter.  Re-solving from the previous resolved qref keeps the branch
        continuous while still allowing the physical arm to converge to the
        held target.
        """

        if self._last_resolved_qpos_by_side is None:
            raise RuntimeError("CubeLiftPoseActionController.reset() is required first.")
        seed = {
            side: values.astype(np.float64, copy=True)
            for side, values in self._last_resolved_qpos_by_side.items()
        }
        if self._ik is not None:
            # RM65CanonicalIK owns the coupled planner state.  Explicitly
            # rebasing it to the last resolved result prevents a solver branch
            # change when only the gripper dimension is active.
            self._ik.reset_from_qpos(seed)
            result = self._ik.solve(
                {
                    self.side: PoseTarget(
                        side=self.side,
                        pose=self._target_pose_arm_base,
                        controlled_frame=self.config.pose_control_frame,
                    )
                }
            )
        else:
            assert self._fallback_ik is not None
            qpos, iterations, position_error_m, orientation_error_rad = (
                self._fallback_ik.solve(
                    self._target_pose_arm_base,
                    current_qpos=seed[self.side],
                )
            )
            qpos_by_side = {
                side: values.tolist() for side, values in seed.items()
            }
            qpos_by_side[self.side] = qpos.tolist()
            result = IKResult(
                success=bool(
                    position_error_m <= _FALLBACK_POSE_POSITION_TOLERANCE_M
                    and orientation_error_rad
                    <= _FALLBACK_POSE_ORIENTATION_TOLERANCE_RAD
                ),
                iterations=iterations,
                qpos_by_side=qpos_by_side,
                position_error_m={self.side: position_error_m},
                orientation_error_rad={self.side: orientation_error_rad},
                controlled_frame_by_side={
                    self.side: self.config.pose_control_frame,
                },
                status=(
                    "tracking"
                    if (
                        position_error_m <= _FALLBACK_POSE_POSITION_TOLERANCE_M
                        and orientation_error_rad
                        <= _FALLBACK_POSE_ORIENTATION_TOLERANCE_RAD
                    )
                    else "limited"
                ),
                metadata={"solver_fallback": True},
                solver=self.solver_name,
            )
        return IKResult(
            success=result.success,
            iterations=result.iterations,
            qpos_by_side=result.qpos_by_side,
            position_error_m=result.position_error_m,
            orientation_error_rad=result.orientation_error_rad,
            controlled_frame_by_side=result.controlled_frame_by_side,
            message=result.message,
            status=result.status,
            hold=result.hold,
            metadata={**result.metadata, "zero_pose_action_hold": True},
            solver=result.solver,
        )

    def apply(self, action: Any) -> CubeLiftPoseActionStep:
        values = np.asarray(action, dtype=float).reshape(-1)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("CubeLift pose action must be a finite 7-D vector.")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("CubeLift pose action values must lie in [-1, 1].")
        if self._target_pose_arm_base is None:
            raise RuntimeError("CubeLiftPoseActionController.reset() is required first.")

        # Keep an unchanged TCP target on a continuity-preserving IK seed.
        # Re-solving from measured qpos can choose a nearby equivalent joint
        # branch, while returning the old qref forever prevents a lagging
        # physical servo from converging.  The helper tracks the held target
        # without introducing a new relative pose increment.
        if np.all(np.abs(values[:6]) <= 1e-12):
            if not self._pose_motion_started:
                # A zero action immediately after reset is an unconditional
                # stationary hold.  No physical target is lagging yet, so a
                # fresh nonlinear solve would only introduce an arbitrary
                # equivalent wrist branch (and can move a gripper into the
                # table while the policy is intentionally doing nothing).
                qpos_by_side = {
                    side: values.astype(float, copy=True).tolist()
                    for side, values in self._last_resolved_qpos_by_side.items()
                }
                result = IKResult(
                    success=True,
                    iterations=0,
                    qpos_by_side=qpos_by_side,
                    controlled_frame_by_side={
                        self.side: self.config.pose_control_frame,
                    },
                    message="zero incremental pose action; holding reset qref",
                    status="held",
                    hold=False,
                    metadata={"zero_pose_action_hold": True},
                    solver=self.solver_name,
                )
            else:
                result = self._zero_increment_result()
            step = CubeLiftPoseActionStep(
                target_pose_arm_base=self._target_pose_arm_base.copy(),
                ik_result=result,
            )
            if step.ik_result.success and not step.ik_result.hold:
                self._last_resolved_qpos_by_side = {
                    side: np.asarray(qpos, dtype=np.float64).reshape(6).copy()
                    for side, qpos in step.ik_result.qpos_by_side.items()
                    if side in {"left", "right"}
                }
            self.last_step = step
            return step

        self._pose_motion_started = True
        target = self._target_pose_arm_base.copy()
        target[:3, 3] += values[:3] * np.asarray(self.config.right_tcp_delta_m, dtype=float)
        rotation_delta = Rotation.from_rotvec(
            values[3:6] * np.asarray(self.config.right_tcp_rotation_delta_rad, dtype=float)
        ).as_matrix()
        # The policy action is expressed in the arm-base axes.  Left-multiply
        # so translation and rotation semantics are independent of the current
        # wrist orientation and are easy to reproduce on the real position
        # chain.
        target[:3, :3] = rotation_delta @ target[:3, :3]
        self._target_pose_arm_base = target
        if self._ik is not None:
            result = self._ik.solve(
                {
                    self.side: PoseTarget(
                        side=self.side,
                        pose=target,
                        controlled_frame=self.config.pose_control_frame,
                    )
                }
            )
        else:
            assert self._fallback_ik is not None
            fallback_seed = (
                self._last_resolved_qpos_by_side[self.side]
                if self._last_resolved_qpos_by_side is not None
                else self.scene.get_arm_qpos(self.side)
            )
            qpos, iterations, position_error_m, orientation_error_rad = self._fallback_ik.solve(
                target,
                current_qpos=fallback_seed,
            )
            fallback_success = bool(
                position_error_m <= _FALLBACK_POSE_POSITION_TOLERANCE_M
                and orientation_error_rad <= _FALLBACK_POSE_ORIENTATION_TOLERANCE_RAD
            )
            qpos_by_side = {
                side: (
                    self._last_resolved_qpos_by_side[side].tolist()
                    if self._last_resolved_qpos_by_side is not None
                    else self.scene.get_arm_qpos(side).tolist()
                )
                for side in ("left", "right")
            }
            qpos_by_side[self.side] = qpos.tolist()
            result = IKResult(
                success=fallback_success,
                iterations=iterations,
                qpos_by_side=qpos_by_side,
                position_error_m={self.side: position_error_m},
                orientation_error_rad={self.side: orientation_error_rad},
                controlled_frame_by_side={self.side: self.config.pose_control_frame},
                status="tracking" if fallback_success else "limited",
                metadata={"solver_fallback": True},
                solver=self.solver_name,
            )
        step = CubeLiftPoseActionStep(
            target_pose_arm_base=target.copy(),
            ik_result=result,
        )
        if result.success and not result.hold:
            self._last_resolved_qpos_by_side = {
                side: np.asarray(values, dtype=np.float64).reshape(6).copy()
                for side, values in result.qpos_by_side.items()
                if side in {"left", "right"}
            }
        self.last_step = step
        return step


__all__ = [
    "CubeLiftPoseActionController",
    "CubeLiftPoseActionStep",
    "CubeLiftPosePlannerSyncStatus",
    "SIM_RM65_POSE_QREF_LIMIT_MARGIN_RAD",
    "SIM_RM65_JOINT_LIMIT_RECOVERY_TOLERANCE_RAD",
    "inset_joint_position_target",
]
