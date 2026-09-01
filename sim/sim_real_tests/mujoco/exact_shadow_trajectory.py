"""Execute one registered joint trajectory through the MuJoCo qpos owner.

The executor consumes only immutable shadow contracts.  It collision-checks
every registered point with the measured gripper geometry, publishes the exact
dual-arm qpos references on their relative-time lattice, waits for measured
terminal convergence, and then confirms a lower-owned terminal HOLD.  No
result from this module grants real-hardware motion authority.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime import EffectorCommand, TeleopCommand
from robot_motion.shadow_execution import (
    RegisteredJointTrajectoryPoint,
    RegisteredShadowJointTrajectory,
    ShadowExecutionIdentity,
    ShadowSimulationReceipt,
    ShadowSimulationStatus,
)

_SIDES = ("left", "right")
_EXECUTION_EVIDENCE_HASH_DOMAIN = (
    "agenticrobot.shadow_execution.mujoco_exact_qpos_evidence.v1"
)


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _refs(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    refs = tuple(sorted({str(value).strip() for value in values}))
    if not refs or any(not value for value in refs):
        raise ValueError(f"{name} requires non-empty evidence refs.")
    return refs


@dataclass(frozen=True)
class MujocoShadowModelBinding:
    """Observed MuJoCo model identity, independently bound to one scene."""

    observed_identity: ShadowExecutionIdentity
    scene_hash: str
    evidence_refs: tuple[str, ...]
    verified: bool = False
    verification_method: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observed_identity, ShadowExecutionIdentity):
            raise TypeError("observed_identity must be ShadowExecutionIdentity.")
        scene_hash = str(self.scene_hash).strip()
        if not scene_hash:
            raise ValueError("scene_hash must be non-empty.")
        object.__setattr__(self, "scene_hash", scene_hash)
        object.__setattr__(
            self,
            "evidence_refs",
            _refs(self.evidence_refs, name="model binding"),
        )
        if not isinstance(self.verified, bool):
            raise TypeError("verified must be boolean.")
        method = (
            None
            if self.verification_method is None
            else str(self.verification_method).strip()
        )
        if self.verified and not method:
            raise ValueError("a verified model binding requires a verification method.")
        if not self.verified and method is not None:
            raise ValueError("an unverified model binding cannot name a verification method.")
        object.__setattr__(self, "verification_method", method)


@dataclass(frozen=True)
class MujocoExactQposExecutionConfig:
    """Offline execution limits; none are accepted from an Agent request."""

    start_tolerance_rad: float = 1e-9
    tracking_tolerance_rad: float = 2e-3
    gripper_hold_tolerance: float = 2e-3
    final_settle_timeout_sec: float = 4.0
    terminal_hold_timeout_sec: float = 2.0
    terminal_hold_dwell_sec: float = 0.08
    terminal_velocity_tolerance_rad_s: float = 2e-3
    collision_sample_max_joint_step_rad: float = math.radians(0.1)

    def __post_init__(self) -> None:
        for name in (
            "start_tolerance_rad",
            "tracking_tolerance_rad",
            "gripper_hold_tolerance",
            "final_settle_timeout_sec",
            "terminal_hold_timeout_sec",
            "terminal_hold_dwell_sec",
            "terminal_velocity_tolerance_rad_s",
            "collision_sample_max_joint_step_rad",
        ):
            value = _finite(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)
        if self.tracking_tolerance_rad <= 0.0:
            raise ValueError("tracking_tolerance_rad must be positive.")
        if self.final_settle_timeout_sec <= 0.0:
            raise ValueError("final_settle_timeout_sec must be positive.")
        if self.terminal_hold_timeout_sec <= 0.0:
            raise ValueError("terminal_hold_timeout_sec must be positive.")
        if self.collision_sample_max_joint_step_rad <= 0.0:
            raise ValueError("collision_sample_max_joint_step_rad must be positive.")


def _gripper_close_by_name(control, trajectory: RegisteredShadowJointTrajectory):
    profile = getattr(control.rig.config, "gripper_actuation_profile", None)
    if profile is None:
        raise RuntimeError("exact shadow execution requires a gripper model.")
    result = {
        name: float(trajectory.initial_snapshot.gripper_close_by_side[side])
        for name, side in profile.side_by_gripper.items()
    }
    if set(profile.side_by_gripper.values()) != set(_SIDES):
        raise RuntimeError("gripper model must map exactly left and right.")
    return result


def _gripper_close_by_side(control) -> dict[str, float]:
    profile = getattr(control.rig.config, "gripper_actuation_profile", None)
    if profile is None:
        raise RuntimeError("exact shadow execution requires a gripper model.")
    return {
        side: float(control.scene.get_gripper_close(name))
        for name, side in profile.side_by_gripper.items()
    }


def _maximum_qpos_error(
    observed: Mapping[str, Sequence[float]],
    expected: Mapping[str, Sequence[float]],
) -> float:
    return max(
        float(
            np.max(
                np.abs(
                    np.asarray(observed[side], dtype=float)
                    - np.asarray(expected[side], dtype=float)
                )
            )
        )
        for side in _SIDES
    )


def _maximum_gripper_error(control, trajectory) -> float:
    observed = _gripper_close_by_side(control)
    return max(
        abs(
            observed[side]
            - float(trajectory.initial_snapshot.gripper_close_by_side[side])
        )
        for side in _SIDES
    )


def _point_command(
    point: RegisteredJointTrajectoryPoint,
    trajectory: RegisteredShadowJointTrajectory,
    *,
    timestamp: float,
) -> TeleopCommand:
    command = TeleopCommand(timestamp=float(timestamp))
    for side in _SIDES:
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=point.qpos_by_side[side],
                metadata={
                    "source": "registered_shadow_exact_qpos_v1",
                    "trajectory_hash": trajectory.semantic_content_hash,
                    "point_hash": point.semantic_content_hash,
                    "time_from_start_sec": point.time_from_start_sec,
                },
            )
        )
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_hand",
                side=side,
                command_type="gripper_scalar",
                value=np.asarray(
                    [trajectory.initial_snapshot.gripper_close_by_side[side]],
                    dtype=float,
                ),
                metadata={"source": "registered_shadow_hold_measured_v1"},
            )
        )
    return command


def _preflight_points(control, trajectory, gripper_close_by_name, config):
    projector = control.scene.collision_projector()
    if projector is None:
        raise RuntimeError("exact shadow execution requires MuJoCo collision safety.")
    base = control.scene.kinematic_controlled_qpos()
    minimum_clearance: float | None = None
    def check(qpos_by_side, *, point_index, segment_index, segment_fraction):
        nonlocal minimum_clearance
        target = control.scene.kinematic_target_vector(
            base,
            qpos_by_side,
            gripper_close_by_name=gripper_close_by_name,
        )
        collision = projector.check_qpos(control.scene.data, target)
        payload = collision.to_dict()
        raw_clearance = payload.get("minimum_distance_m")
        if raw_clearance is not None:
            clearance = float(raw_clearance)
            minimum_clearance = (
                clearance
                if minimum_clearance is None
                else min(minimum_clearance, clearance)
            )
        if not collision.safe:
            if point_index == 0:
                status = ShadowSimulationStatus.PATH_START_COLLISION_BLOCKED
            elif point_index == len(trajectory.points) - 1:
                status = ShadowSimulationStatus.PATH_TARGET_COLLISION_BLOCKED
            else:
                status = ShadowSimulationStatus.PATH_COLLISION_BLOCKED
            return (
                status,
                {
                    "point_index": point_index,
                    "segment_index": segment_index,
                    "segment_fraction": segment_fraction,
                },
                minimum_clearance,
                payload,
            )
        return None

    start = check(
        trajectory.points[0].qpos_by_side,
        point_index=0,
        segment_index=None,
        segment_fraction=0.0,
    )
    if start is not None:
        return start
    for segment_index, (left, right) in enumerate(pairwise(trajectory.points)):
        maximum_motion = max(
            float(
                np.max(
                    np.abs(right.qpos_by_side[side] - left.qpos_by_side[side])
                )
            )
            for side in _SIDES
        )
        sample_count = max(
            1,
            math.ceil(maximum_motion / config.collision_sample_max_joint_step_rad),
        )
        for sample_index in range(1, sample_count + 1):
            fraction = sample_index / sample_count
            qpos = {
                side: left.qpos_by_side[side]
                + fraction
                * (right.qpos_by_side[side] - left.qpos_by_side[side])
                for side in _SIDES
            }
            is_registered_endpoint = sample_index == sample_count
            point_index = segment_index + 1 if is_registered_endpoint else None
            result = check(
                qpos,
                point_index=point_index,
                segment_index=segment_index,
                segment_fraction=fraction,
            )
            if result is not None:
                # Only the exact final registered point is a target collision;
                # every between-point sample remains a path collision.
                if not is_registered_endpoint and result[0] is not ShadowSimulationStatus.PATH_COLLISION_BLOCKED:
                    result = (
                        ShadowSimulationStatus.PATH_COLLISION_BLOCKED,
                        result[1],
                        result[2],
                        result[3],
                    )
                return result
    return None, None, minimum_clearance, None


def _evidence_ref(kind: str, payload: Mapping[str, object]) -> str:
    digest = canonical_content_hash(
        dict(payload),
        domain=_EXECUTION_EVIDENCE_HASH_DOMAIN,
    )
    return f"mujoco-shadow-{kind}:{digest}"


def _terminal_hold(control, trajectory, config):
    control.rig.hold_motion("registered shadow trajectory terminal HOLD")
    timing = control.rig.config.timing
    dt = float(timing.physics_timestep_sec)
    max_ticks = max(1, math.ceil(config.terminal_hold_timeout_sec / dt))
    dwell_ticks = max(1, math.ceil(config.terminal_hold_dwell_sec / dt))
    consecutive = 0
    previous = control.rig.measured_qpos_by_side()
    last_observation: dict[str, object] = {}
    for tick in range(max_ticks):
        control.step_physics()
        measured = control.rig.measured_qpos_by_side()
        accepted = control.rig.accepted_qcmd_by_side()
        velocity = max(
            float(
                np.max(
                    np.abs(
                        (np.asarray(measured[side]) - np.asarray(previous[side])) / dt
                    )
                )
            )
            for side in _SIDES
        )
        tracking_error = _maximum_qpos_error(measured, accepted)
        gripper_error = _maximum_gripper_error(control, trajectory)
        ready = (
            velocity <= config.terminal_velocity_tolerance_rad_s
            and tracking_error <= config.tracking_tolerance_rad
            and gripper_error <= config.gripper_hold_tolerance
        )
        consecutive = consecutive + 1 if ready else 0
        last_observation = {
            "tick": tick + 1,
            "maximum_measured_velocity_rad_s": velocity,
            "maximum_accepted_tracking_error_rad": tracking_error,
            "maximum_gripper_hold_error": gripper_error,
            "required_dwell_ticks": dwell_ticks,
            "observed_dwell_ticks": consecutive,
        }
        previous = measured
        if consecutive >= dwell_ticks:
            return True, last_observation
    return False, last_observation


def execute_registered_shadow_trajectory(
    *,
    control,
    trajectory: RegisteredShadowJointTrajectory,
    model_binding: MujocoShadowModelBinding,
    config: MujocoExactQposExecutionConfig | None = None,
) -> ShadowSimulationReceipt:
    """Run exact qpos references in MuJoCo and return typed terminal evidence."""

    if not isinstance(trajectory, RegisteredShadowJointTrajectory):
        raise TypeError("trajectory must be RegisteredShadowJointTrajectory.")
    if not isinstance(model_binding, MujocoShadowModelBinding):
        raise TypeError("model_binding must be MujocoShadowModelBinding.")
    cfg = config or MujocoExactQposExecutionConfig()
    if not isinstance(cfg, MujocoExactQposExecutionConfig):
        raise TypeError("config must be MujocoExactQposExecutionConfig.")
    for name in ("scene", "rig", "step_physics"):
        if not hasattr(control, name):
            raise TypeError(f"control must expose {name}.")

    evidence = list(model_binding.evidence_refs)
    identity_matches = (
        model_binding.verified
        and model_binding.verification_method == "canonical_model_identity_v1"
        and model_binding.observed_identity.semantic_content_hash
        == trajectory.initial_snapshot.identity.semantic_content_hash
        and model_binding.scene_hash == trajectory.scene_hash
    )
    if not identity_matches:
        held, hold_observation = _terminal_hold(control, trajectory, cfg)
        evidence.append(_evidence_ref("terminal-hold", hold_observation))
        return ShadowSimulationReceipt(
            trajectory_hash=trajectory.semantic_content_hash,
            initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
            identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
            scene_hash=trajectory.scene_hash,
            status=ShadowSimulationStatus.MODEL_IDENTITY_MISMATCH,
            executed_point_count=0,
            terminal_hold_confirmed=held,
            exact_planned_qpos_execution_confirmed=False,
            model_identity_confirmed=False,
            evidence_refs=tuple(evidence),
        )

    measured_start = control.rig.measured_qpos_by_side()
    accepted_start = control.rig.accepted_qcmd_by_side()
    start_arm_error = max(
        _maximum_qpos_error(measured_start, trajectory.initial_snapshot.qpos_by_side),
        _maximum_qpos_error(accepted_start, trajectory.initial_snapshot.qpos_by_side),
    )
    start_gripper_error = _maximum_gripper_error(control, trajectory)
    if (
        start_arm_error > cfg.start_tolerance_rad
        or start_gripper_error > cfg.gripper_hold_tolerance
    ):
        held, hold_observation = _terminal_hold(control, trajectory, cfg)
        evidence.extend(
            (
                _evidence_ref(
                    "start-mismatch",
                    {
                        "maximum_arm_error_rad": start_arm_error,
                        "maximum_gripper_error": start_gripper_error,
                    },
                ),
                _evidence_ref("terminal-hold", hold_observation),
            )
        )
        return ShadowSimulationReceipt(
            trajectory_hash=trajectory.semantic_content_hash,
            initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
            identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
            scene_hash=trajectory.scene_hash,
            status=ShadowSimulationStatus.TRAJECTORY_START_MISMATCH,
            executed_point_count=0,
            terminal_hold_confirmed=held,
            exact_planned_qpos_execution_confirmed=False,
            model_identity_confirmed=True,
            evidence_refs=tuple(evidence),
        )

    gripper_close_by_name = _gripper_close_by_name(control, trajectory)
    blocked, blocked_location, minimum_clearance, collision = _preflight_points(
        control,
        trajectory,
        gripper_close_by_name,
        cfg,
    )
    if blocked is not None:
        held, hold_observation = _terminal_hold(control, trajectory, cfg)
        evidence.extend(
            (
                _evidence_ref(
                    "collision-blocked",
                    {"location": blocked_location, "collision": collision},
                ),
                _evidence_ref("terminal-hold", hold_observation),
            )
        )
        return ShadowSimulationReceipt(
            trajectory_hash=trajectory.semantic_content_hash,
            initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
            identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
            scene_hash=trajectory.scene_hash,
            status=blocked,
            executed_point_count=0,
            terminal_hold_confirmed=held,
            exact_planned_qpos_execution_confirmed=False,
            model_identity_confirmed=True,
            evidence_refs=tuple(evidence),
            minimum_clearance_m=minimum_clearance,
        )

    timing = control.rig.config.timing
    dt = float(timing.physics_timestep_sec)
    started_at = float(control.scene.data.time)
    executed_count = 0
    for index, point in enumerate(trajectory.points):
        due = started_at + point.time_from_start_sec
        while float(control.scene.data.time) + 1e-12 < due:
            control.step_physics()
            if _maximum_gripper_error(control, trajectory) > cfg.gripper_hold_tolerance:
                held, hold_observation = _terminal_hold(control, trajectory, cfg)
                evidence.extend(
                    (
                        _evidence_ref("gripper-drift", {"point_index": index}),
                        _evidence_ref("terminal-hold", hold_observation),
                    )
                )
                return ShadowSimulationReceipt(
                    trajectory_hash=trajectory.semantic_content_hash,
                    initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
                    identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
                    scene_hash=trajectory.scene_hash,
                    status=ShadowSimulationStatus.GRIPPER_HOLD_DRIFT,
                    executed_point_count=executed_count,
                    terminal_hold_confirmed=held,
                    exact_planned_qpos_execution_confirmed=False,
                    model_identity_confirmed=True,
                    evidence_refs=tuple(evidence),
                    minimum_clearance_m=minimum_clearance,
                )
        receipts = control.rig.send_commands(
            _point_command(
                point,
                trajectory,
                timestamp=float(control.scene.data.time),
            )
        )
        required = {"left_arm", "right_arm", "left_hand", "right_hand"}
        if set(receipts) != required or any(
            not bool(receipts[name].accepted) for name in required
        ):
            raise RuntimeError("MuJoCo exact-qpos dispatch was not atomically accepted.")
        executed_count += 1
        evidence.append(
            _evidence_ref(
                "point",
                {
                    "point_index": index,
                    "point_hash": point.semantic_content_hash,
                    "published_sim_time": float(control.scene.data.time) - started_at,
                },
            )
        )
        # Consume the mailbox before another registered point can replace it.
        control.step_physics()

    final_qpos = trajectory.points[-1].qpos_by_side
    max_ticks = max(1, math.ceil(cfg.final_settle_timeout_sec / dt))
    final_observation: dict[str, object] = {}
    final_settled = False
    for tick in range(max_ticks):
        measured = control.rig.measured_qpos_by_side()
        accepted = control.rig.accepted_qcmd_by_side()
        measured_error = _maximum_qpos_error(measured, final_qpos)
        accepted_error = _maximum_qpos_error(accepted, final_qpos)
        gripper_error = _maximum_gripper_error(control, trajectory)
        final_observation = {
            "tick": tick,
            "maximum_measured_error_rad": measured_error,
            "maximum_accepted_error_rad": accepted_error,
            "maximum_gripper_hold_error": gripper_error,
        }
        if gripper_error > cfg.gripper_hold_tolerance:
            break
        if (
            measured_error <= cfg.tracking_tolerance_rad
            and accepted_error <= cfg.tracking_tolerance_rad
        ):
            final_settled = True
            break
        control.step_physics()

    if not final_settled:
        status = (
            ShadowSimulationStatus.GRIPPER_HOLD_DRIFT
            if final_observation.get("maximum_gripper_hold_error", 0.0)
            > cfg.gripper_hold_tolerance
            else ShadowSimulationStatus.POINT_TRACKING_TIMEOUT
        )
        held, hold_observation = _terminal_hold(control, trajectory, cfg)
        evidence.extend(
            (
                _evidence_ref("final-unsettled", final_observation),
                _evidence_ref("terminal-hold", hold_observation),
            )
        )
        return ShadowSimulationReceipt(
            trajectory_hash=trajectory.semantic_content_hash,
            initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
            identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
            scene_hash=trajectory.scene_hash,
            status=status,
            executed_point_count=executed_count,
            terminal_hold_confirmed=held,
            exact_planned_qpos_execution_confirmed=False,
            model_identity_confirmed=True,
            evidence_refs=tuple(evidence),
            minimum_clearance_m=minimum_clearance,
        )

    held, hold_observation = _terminal_hold(control, trajectory, cfg)
    evidence.extend(
        (
            _evidence_ref("final-settled", final_observation),
            _evidence_ref("terminal-hold", hold_observation),
        )
    )
    status = (
        ShadowSimulationStatus.COMPLETED
        if held
        else ShadowSimulationStatus.TERMINAL_HOLD_FAILED
    )
    return ShadowSimulationReceipt(
        trajectory_hash=trajectory.semantic_content_hash,
        initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
        identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
        scene_hash=trajectory.scene_hash,
        status=status,
        executed_point_count=executed_count,
        terminal_hold_confirmed=held,
        exact_planned_qpos_execution_confirmed=True,
        model_identity_confirmed=True,
        evidence_refs=tuple(evidence),
        minimum_clearance_m=minimum_clearance,
    )


__all__ = [
    "MujocoExactQposExecutionConfig",
    "MujocoShadowModelBinding",
    "execute_registered_shadow_trajectory",
]
