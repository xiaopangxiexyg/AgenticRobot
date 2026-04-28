from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Literal, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from rm65_inspire_tac.ik_tuning import (
    RM65IKTuningConfig,
    joint_jump_exceeds,
    load_ik_tuning_config,
    pose_delta_metrics,
    preferred_wrist_flip_candidate,
    weighted_joint_distance_score,
)
from rm65_inspire_tac.multitask_ik import MultiTaskIKSolver
from rm65_inspire_tac.pipeline import (
    DEFAULT_LEFT_ARM_QPOS,
    DEFAULT_RIGHT_ARM_QPOS,
    DualArmSceneProfile,
    build_dual_arm_scene_profile,
)

from ..calibration import TeleopCalibration
from ..interfaces import Retargeter
from ..registry import register_retargeter
from ..schemas import EffectorCommand, TeleopCommand, TeleopObservation
from ..tcp import HAND_BASE_TCP_FRAME, canonical_tcp_frame
from ..transforms import as_pose_matrix, interpolate_pose
from .pose_to_arm import PoseToArmRetargeter, PoseToArmRetargeterConfig


_HAND_BASE_FROM_ARM_LINK6 = np.eye(4, dtype=float)
_HAND_BASE_FROM_ARM_LINK6[2, 3] = 0.009
_ARM_LINK6_FROM_HAND_BASE = np.linalg.inv(_HAND_BASE_FROM_ARM_LINK6)
_CURRENT_TARGET_HOLD_TRANSLATION_M = 1e-3
_CURRENT_TARGET_HOLD_ROTATION_DEG = 0.5


def _optional_qpos(value: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    qpos = np.asarray(value, dtype=float).reshape(-1)
    if qpos.shape != (6,):
        raise ValueError(f"Expected a 6D joint vector, got shape {qpos.shape}.")
    return qpos.copy()


def _optional_joint_step(value: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    joint_step = np.asarray(value, dtype=float).reshape(-1)
    if joint_step.shape != (6,):
        raise ValueError(f"Expected a 6D joint-step vector, got shape {joint_step.shape}.")
    if np.any(joint_step <= 0.0):
        raise ValueError("Execution joint-step limits must be strictly positive.")
    return joint_step.copy()


@dataclass(frozen=True)
class DualArmTargetContext:
    timestamp: float | None
    left_target: np.ndarray
    right_target: np.ndarray
    left_cartesian: EffectorCommand | None
    right_cartesian: EffectorCommand | None
    left_input_skipped: str | None
    right_input_skipped: str | None


@dataclass
class DualArmCollisionAwareRetargeterConfig:
    left_input_channel_name: str = "left_wrist_pose"
    right_input_channel_name: str = "right_wrist_pose"
    left_effector_name: str = "left_arm"
    right_effector_name: str = "right_arm"
    control_mode: str = "delta_clutch"
    orientation_mode: str = "full"
    translation_scale: float = 1.0
    workspace_clip: bool = True
    scene_profile: str = "dual_arm_hand_proxy_open"
    contact_mode: Literal["boundary_slide", "freeze_on_contact"] = "boundary_slide"
    solver: str | None = None
    solver_frequency: float = 30.0
    posture_cost: float = 1e-1
    collision_min_distance_m: float = 0.03
    collision_distance_epsilon_m: float = 1e-4
    collision_barrier_gain: float = 10.0
    collision_safe_displacement_gain: float = 0.001
    hold_last_safe_on_failure: bool = True
    debug_viz: bool = False
    debug_profile: bool = False
    debug_profile_print_every: int = 0
    solve_max_iterations: int = 1
    solve_translation_tolerance_m: float = 2e-3
    solve_orientation_tolerance_deg: float = 0.5
    enforce_realistic_joint_step: bool = False
    left_execution_max_joint_step_rad: np.ndarray | Sequence[float] | None = None
    right_execution_max_joint_step_rad: np.ndarray | Sequence[float] | None = None
    left_initial_qpos: np.ndarray | Sequence[float] | None = None
    right_initial_qpos: np.ndarray | Sequence[float] | None = None
    ik_tuning_json_path: str | None = None
    ik_tuning: RM65IKTuningConfig | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.contact_mode not in {"boundary_slide", "freeze_on_contact"}:
            raise ValueError(
                f"Unsupported contact_mode: {self.contact_mode}. "
                "Expected 'boundary_slide' or 'freeze_on_contact'."
            )
        if self.solver_frequency <= 0.0:
            raise ValueError("solver_frequency must be positive.")
        if self.collision_min_distance_m < 0.0:
            raise ValueError("collision_min_distance_m must be >= 0.")
        if self.collision_distance_epsilon_m < 0.0:
            raise ValueError("collision_distance_epsilon_m must be >= 0.")
        if self.solve_max_iterations <= 0:
            raise ValueError("solve_max_iterations must be positive.")
        if self.solve_translation_tolerance_m < 0.0:
            raise ValueError("solve_translation_tolerance_m must be >= 0.")
        if self.solve_orientation_tolerance_deg < 0.0:
            raise ValueError("solve_orientation_tolerance_deg must be >= 0.")
        if self.debug_profile_print_every < 0:
            raise ValueError("debug_profile_print_every must be >= 0.")
        self.left_execution_max_joint_step_rad = _optional_joint_step(
            self.left_execution_max_joint_step_rad
        )
        self.right_execution_max_joint_step_rad = _optional_joint_step(
            self.right_execution_max_joint_step_rad
        )
        if self.enforce_realistic_joint_step:
            if self.left_execution_max_joint_step_rad is None:
                raise ValueError(
                    "enforce_realistic_joint_step=True requires left_execution_max_joint_step_rad."
                )
            if self.right_execution_max_joint_step_rad is None:
                raise ValueError(
                    "enforce_realistic_joint_step=True requires right_execution_max_joint_step_rad."
                )
        self.left_initial_qpos = _optional_qpos(self.left_initial_qpos)
        self.right_initial_qpos = _optional_qpos(self.right_initial_qpos)
        if self.ik_tuning is None:
            self.ik_tuning = load_ik_tuning_config(self.ik_tuning_json_path)


@register_retargeter("dual_arm_collision_aware")
class DualArmCollisionAwareRetargeter(Retargeter):
    def __init__(
        self,
        config: DualArmCollisionAwareRetargeterConfig,
        *,
        left_pose_retargeter: PoseToArmRetargeter | None = None,
        right_pose_retargeter: PoseToArmRetargeter | None = None,
        ik_solver: MultiTaskIKSolver | None = None,
    ):
        self.config = config
        self.ik_tuning = config.ik_tuning
        assert self.ik_tuning is not None
        self.profile: DualArmSceneProfile = build_dual_arm_scene_profile(config.scene_profile)
        self.left_pose_retargeter = (
            left_pose_retargeter
            if left_pose_retargeter is not None
            else PoseToArmRetargeter(
                PoseToArmRetargeterConfig(
                    side="left",
                    input_channel_name=config.left_input_channel_name,
                    effector_name=config.left_effector_name,
                    control_mode=config.control_mode,
                    orientation_mode=config.orientation_mode,
                    translation_scale=config.translation_scale,
                    workspace_clip=config.workspace_clip,
                )
            )
        )
        self.right_pose_retargeter = (
            right_pose_retargeter
            if right_pose_retargeter is not None
            else PoseToArmRetargeter(
                PoseToArmRetargeterConfig(
                    side="right",
                    input_channel_name=config.right_input_channel_name,
                    effector_name=config.right_effector_name,
                    control_mode=config.control_mode,
                    orientation_mode=config.orientation_mode,
                    translation_scale=config.translation_scale,
                    workspace_clip=config.workspace_clip,
                )
            )
        )
        if ik_solver is not None:
            self.ik_solver = ik_solver
        else:
            try:
                self.ik_solver = MultiTaskIKSolver(
                    robot_urdf_path=self.profile.urdf_path,
                    robot_package_dirs=list(self.profile.package_dirs),
                    robot_root_joint=None,
                    robot_list_of_joints_to_lock=None,
                    robot_collision_pair_names=list(self.profile.collision_pairs),
                    solver=config.solver,
                    frequency=float(config.solver_frequency),
                    debug_viz=bool(config.debug_viz),
                    debug_profile=bool(config.debug_profile),
                    sleep=False,
                    collision_barrier_gain=float(config.collision_barrier_gain),
                    collision_safe_displacement_gain=float(
                        config.collision_safe_displacement_gain
                    ),
                    collision_min_distance_m=float(config.collision_min_distance_m),
                )
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "DualArmCollisionAwareRetargeter requires pinocchio, pink, and qpsolvers "
                    "for joint-space collision avoidance."
                ) from exc

        objective = self.ik_tuning.objective_weights()
        self.ik_solver.add_frame_task(
            self.profile.left_frame_name,
            position_cost=float(objective.dual_arm_position_cost),
            orientation_cost=float(objective.dual_arm_orientation_cost),
        )
        self.ik_solver.add_frame_task(
            self.profile.right_frame_name,
            position_cost=float(objective.dual_arm_position_cost),
            orientation_cost=float(objective.dual_arm_orientation_cost),
        )
        self.ik_solver.add_posture_task(cost=float(config.posture_cost))

        self._arm_joint_weights = np.asarray(
            self.ik_tuning.branch_normalization.joint_distance_weights,
            dtype=float,
        )
        self._full_joint_weights = self._make_full_joint_weights(self.ik_solver.current_q)
        self._full_execution_joint_step = self._make_full_execution_joint_step(
            self.ik_solver.current_q
        )
        self._left_arm_limits = self._split_solution(
            np.asarray(self.ik_solver.robot.model.lowerPositionLimit, dtype=float)
        )[0], self._split_solution(
            np.asarray(self.ik_solver.robot.model.upperPositionLimit, dtype=float)
        )[0]
        self._right_arm_limits = self._split_solution(
            np.asarray(self.ik_solver.robot.model.lowerPositionLimit, dtype=float)
        )[1], self._split_solution(
            np.asarray(self.ik_solver.robot.model.upperPositionLimit, dtype=float)
        )[1]
        self._last_target_poses: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._jump_hold_active = False
        self._jump_hold_full_q: np.ndarray | None = None
        self._jump_hold_anchor_targets: dict[str, np.ndarray] | None = None
        self._jump_hold_frames = 0
        self._profile_frame_index = 0

        self.set_current_qpos_pair(
            left_qpos=config.left_initial_qpos,
            right_qpos=config.right_initial_qpos,
        )
        self._last_safe_full_q = self.ik_solver.current_q.copy()

    @property
    def name(self) -> str:
        return "dual_arm_collision_aware"

    def set_current_qpos_pair(
        self,
        *,
        left_qpos: Sequence[float] | np.ndarray | None = None,
        right_qpos: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        left = (
            DEFAULT_LEFT_ARM_QPOS.copy()
            if left_qpos is None
            else np.asarray(left_qpos, dtype=float).reshape(-1)
        )
        right = (
            DEFAULT_RIGHT_ARM_QPOS.copy()
            if right_qpos is None
            else np.asarray(right_qpos, dtype=float).reshape(-1)
        )
        if left.shape != (6,) or right.shape != (6,):
            raise ValueError(
                f"DualArmCollisionAwareRetargeter expects 6D arm qpos per side, got {left.shape} and {right.shape}."
            )
        full_q = self._embed_arm_qpos(
            current_full_q=self.ik_solver.current_q,
            left_qpos=left,
            right_qpos=right,
        )
        self.ik_solver.set_configuration_q(full_q)
        self._last_safe_full_q = full_q.copy()
        self._clear_hold()

    def current_world_targets(self) -> dict[str, np.ndarray]:
        return {
            "left": self.ik_solver.compute_frame_pose(self.profile.left_frame_name),
            "right": self.ik_solver.compute_frame_pose(self.profile.right_frame_name),
        }

    def _profile_template(self) -> dict[str, object] | None:
        if not self.config.debug_profile:
            return None
        return {
            "solve_calls": 0,
            "solve_total_ms": 0.0,
            "solve_target_setup_ms": 0.0,
            "solve_qp_ms": 0.0,
            "solve_integrate_ms": 0.0,
            "solve_set_configuration_ms": 0.0,
            "min_distance_calls": 0,
            "min_distance_total_ms": 0.0,
            "min_distance_phase_ms": {},
            "branch_ms": 0.0,
            "jump_gate_ms": 0.0,
            "hold_recovery_ms": 0.0,
            "target_already_reached": False,
            "frame_total_ms": 0.0,
        }

    def _record_solver_profile(self, profile: dict[str, object] | None) -> None:
        if profile is None:
            return
        solver_profile = dict(self.ik_solver.last_solve_profile)
        if not solver_profile:
            return
        profile["solve_calls"] = int(profile.get("solve_calls", 0)) + 1
        for source_key, target_key in (
            ("total_ms", "solve_total_ms"),
            ("target_setup_ms", "solve_target_setup_ms"),
            ("qp_ms", "solve_qp_ms"),
            ("integrate_ms", "solve_integrate_ms"),
            ("set_configuration_ms", "solve_set_configuration_ms"),
        ):
            profile[target_key] = float(profile.get(target_key, 0.0)) + float(
                solver_profile.get(source_key, 0.0)
            )

    def _profiled_min_collision_distance(
        self,
        q: np.ndarray,
        profile: dict[str, object] | None,
        *,
        phase: str,
    ) -> float | None:
        start_time = time.perf_counter()
        min_distance = self.ik_solver.compute_min_collision_distance(q)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if profile is not None:
            profile["min_distance_calls"] = int(profile.get("min_distance_calls", 0)) + 1
            profile["min_distance_total_ms"] = float(
                profile.get("min_distance_total_ms", 0.0)
            ) + elapsed_ms
            phase_totals = profile.get("min_distance_phase_ms")
            if not isinstance(phase_totals, dict):
                phase_totals = {}
                profile["min_distance_phase_ms"] = phase_totals
            phase_totals[phase] = float(phase_totals.get(phase, 0.0)) + elapsed_ms
        return min_distance

    def _emit_profile(
        self,
        *,
        profile: dict[str, object] | None,
        command: TeleopCommand,
        status: str,
        iterations: int,
        branch_candidate: str,
    ) -> None:
        if profile is None:
            return
        self._profile_frame_index += 1
        command.metadata["ik_debug_profile"] = dict(profile)
        print_every = int(self.config.debug_profile_print_every)
        if print_every <= 0 or self._profile_frame_index % print_every != 0:
            return
        print(
            "[dual_arm_collision_aware profile] "
            f"frame={self._profile_frame_index} "
            f"total_ms={float(profile.get('frame_total_ms', 0.0)):.3f} "
            f"solve_calls={int(profile.get('solve_calls', 0))} "
            f"solve_ms={float(profile.get('solve_total_ms', 0.0)):.3f} "
            f"qp_ms={float(profile.get('solve_qp_ms', 0.0)):.3f} "
            f"integrate_ms={float(profile.get('solve_integrate_ms', 0.0)):.3f} "
            f"min_dist_ms={float(profile.get('min_distance_total_ms', 0.0)):.3f} "
            f"iters={iterations} status={status} branch={branch_candidate}"
        )

    def interpolate_world_targets(
        self,
        start_targets: dict[str, np.ndarray],
        end_targets: dict[str, np.ndarray],
        alpha: float,
    ) -> dict[str, np.ndarray]:
        return {
            "left": interpolate_pose(start_targets["left"], end_targets["left"], alpha),
            "right": interpolate_pose(start_targets["right"], end_targets["right"], alpha),
        }

    def build_target_context(
        self,
        observation: TeleopObservation,
        calibration: TeleopCalibration | None = None,
    ) -> DualArmTargetContext:
        if calibration is None:
            raise RuntimeError(f"{self.name} requires a TeleopCalibration instance.")

        left_command = self.left_pose_retargeter.retarget(observation, calibration)
        right_command = self.right_pose_retargeter.retarget(observation, calibration)
        left_cartesian = left_command.commands.get(self.config.left_effector_name)
        right_cartesian = right_command.commands.get(self.config.right_effector_name)

        left_target = self._target_pose_world(
            command=left_cartesian,
            fallback_frame_name=self.profile.left_frame_name,
            side="left",
            calibration=calibration,
        )
        right_target = self._target_pose_world(
            command=right_cartesian,
            fallback_frame_name=self.profile.right_frame_name,
            side="right",
            calibration=calibration,
        )
        return DualArmTargetContext(
            timestamp=observation.timestamp,
            left_target=left_target.copy(),
            right_target=right_target.copy(),
            left_cartesian=left_cartesian.copy() if left_cartesian is not None else None,
            right_cartesian=right_cartesian.copy() if right_cartesian is not None else None,
            left_input_skipped=left_command.metadata.get("skipped"),
            right_input_skipped=right_command.metadata.get("skipped"),
        )

    def build_command_for_context(
        self,
        context: DualArmTargetContext,
        *,
        targets: dict[str, np.ndarray] | None = None,
    ) -> TeleopCommand:
        frame_start = time.perf_counter()
        profile = self._profile_template()
        left_target = as_pose_matrix(
            context.left_target if targets is None else targets["left"]
        )
        right_target = as_pose_matrix(
            context.right_target if targets is None else targets["right"]
        )
        current_targets = {"left": left_target.copy(), "right": right_target.copy()}
        solver_targets_before_solve = self.current_world_targets()
        left_source_target_pose_base = (
            None
            if context.left_cartesian is None or context.left_cartesian.command_type != "cartesian_pose"
            else as_pose_matrix(context.left_cartesian.value)
        )
        right_source_target_pose_base = (
            None
            if context.right_cartesian is None or context.right_cartesian.command_type != "cartesian_pose"
            else as_pose_matrix(context.right_cartesian.value)
        )
        left_source_target_pose_desk = (
            None
            if context.left_cartesian is None
            else context.left_cartesian.metadata.get("target_pose_desk")
        )
        right_source_target_pose_desk = (
            None
            if context.right_cartesian is None
            else context.right_cartesian.metadata.get("target_pose_desk")
        )
        left_source_target_pose_operator = (
            None
            if context.left_cartesian is None
            else context.left_cartesian.metadata.get("target_pose_operator")
        )
        right_source_target_pose_operator = (
            None
            if context.right_cartesian is None
            else context.right_cartesian.metadata.get("target_pose_operator")
        )

        command = TeleopCommand(timestamp=context.timestamp)
        command.metadata.update(
            {
                "controller": self.name,
                "scene_profile": self.profile.name,
                "collision_pair_count": len(self.profile.collision_pairs),
                "collision_avoidance_enabled": True,
                "collision_avoidance_mode": self.config.contact_mode,
                "collision_min_distance_m": float(self.config.collision_min_distance_m),
                "left_input_skipped": context.left_input_skipped,
                "right_input_skipped": context.right_input_skipped,
                "ik_objective_mode": self.ik_tuning.ik_objective_mode,
                "ik_tuning_source": self.ik_tuning.source_path,
                "source_target_pose_base": {
                    "left": left_source_target_pose_base,
                    "right": right_source_target_pose_base,
                },
                "source_target_pose_desk": {
                    "left": left_source_target_pose_desk,
                    "right": right_source_target_pose_desk,
                },
                "source_target_pose_operator": {
                    "left": left_source_target_pose_operator,
                    "right": right_source_target_pose_operator,
                },
            }
        )

        status = "tracking"
        fallback_reason: str | None = None
        min_distance_m: float | None = None
        branch_candidate = "original"
        branch_score_deg: float | None = None
        jump_gate_status = "pass"
        jump_gate_reason: str | None = None
        jump_gate_release_ready = False
        jump_gate_hold_frames = self._jump_hold_frames
        jump_gate_metrics: dict[str, object] = {}
        execution_step_metadata: dict[str, object] = {
            "enabled": bool(self.config.enforce_realistic_joint_step),
            "step_limited": False,
            "weighted_delta_deg": 0.0,
            "left_max_abs_delta_deg": 0.0,
            "right_max_abs_delta_deg": 0.0,
        }
        solve_iteration_count = 0
        solve_pose_error: dict[str, dict[str, float]] = {}

        pre_solve_q = self.ik_solver.current_q.copy()
        active_q = pre_solve_q.copy()
        target_already_reached = True
        for side in ("left", "right"):
            translation_m, rotation_deg = pose_delta_metrics(
                solver_targets_before_solve[side],
                current_targets[side],
            )
            solve_pose_error[side] = {
                "translation_m": float(translation_m),
                "rotation_deg": float(rotation_deg),
            }
            if (
                translation_m > _CURRENT_TARGET_HOLD_TRANSLATION_M
                or rotation_deg > _CURRENT_TARGET_HOLD_ROTATION_DEG
            ):
                target_already_reached = False
        if profile is not None:
            profile["target_already_reached"] = bool(target_already_reached)

        try:
            if target_already_reached:
                candidate_q = pre_solve_q.copy()
                branch_candidate = "current_pose_hold"
                branch_score_deg = 0.0
                branch_min_distance = self._profiled_min_collision_distance(
                    candidate_q,
                    profile,
                    phase="current_pose_hold",
                )
            else:
                solved_q, solve_iteration_count, solve_pose_error = self._solve_toward_targets(
                    left_target=left_target,
                    right_target=right_target,
                    profile=profile,
                )
                branch_start = time.perf_counter()
                candidate_q, branch_candidate, branch_score_deg, branch_min_distance = (
                    self._select_normalized_candidate(
                        raw_full_q=solved_q,
                        reference_full_q=pre_solve_q,
                        profile=profile,
                    )
                )
                if profile is not None:
                    profile["branch_ms"] = float(profile.get("branch_ms", 0.0)) + (
                        time.perf_counter() - branch_start
                    ) * 1000.0
                gate_start = time.perf_counter()
                gate_result = self._evaluate_jump_gate(
                    reference_full_q=pre_solve_q,
                    candidate_full_q=candidate_q,
                    current_targets=current_targets,
                )
                if profile is not None:
                    profile["jump_gate_ms"] = float(profile.get("jump_gate_ms", 0.0)) + (
                        time.perf_counter() - gate_start
                    ) * 1000.0
                jump_gate_status = str(gate_result["status"])
                jump_gate_reason = (
                    None if gate_result["reason"] is None else str(gate_result["reason"])
                )
                jump_gate_release_ready = bool(gate_result["release_ready"])
                jump_gate_metrics = dict(gate_result["metrics"])

                if bool(gate_result["hold"]) and not jump_gate_release_ready:
                    self._enter_hold(hold_full_q=pre_solve_q, current_targets=current_targets)
                    jump_gate_hold_frames = self._jump_hold_frames
                    active_q = (
                        self._jump_hold_full_q.copy()
                        if self._jump_hold_full_q is not None
                        else pre_solve_q.copy()
                    )
                    self.ik_solver.set_configuration_q(active_q)
                    status = "jump_gate_hold"
                    fallback_reason = jump_gate_reason
                    recovery_start = time.perf_counter()
                    recovery_result = self._attempt_hold_recovery(
                        current_targets=current_targets,
                        profile=profile,
                    )
                    if profile is not None:
                        profile["hold_recovery_ms"] = float(
                            profile.get("hold_recovery_ms", 0.0)
                        ) + (time.perf_counter() - recovery_start) * 1000.0
                    if recovery_result is not None:
                        active_q = recovery_result["q"].copy()
                        min_distance_m = recovery_result["min_distance_m"]
                        branch_candidate = str(recovery_result["candidate_name"])
                        branch_score_deg = float(recovery_result["score_deg"])
                        jump_gate_status = str(recovery_result["jump_gate_status"])
                        jump_gate_reason = (
                            None
                            if recovery_result["jump_gate_reason"] is None
                            else str(recovery_result["jump_gate_reason"])
                        )
                        jump_gate_release_ready = bool(recovery_result["release_ready"])
                        jump_gate_metrics = dict(recovery_result["metrics"])
                        near_boundary = (
                            min_distance_m is not None
                            and min_distance_m
                            <= float(self.config.collision_min_distance_m)
                            + float(self.config.collision_distance_epsilon_m)
                        )
                        status = "boundary_active" if near_boundary else "tracking"
                        fallback_reason = None
                        self.ik_solver.set_configuration_q(active_q)
                        self._last_safe_full_q = active_q.copy()
                        self._clear_hold()
                else:
                    min_distance_m = (
                        branch_min_distance
                        if branch_min_distance is not None
                        else self._profiled_min_collision_distance(
                            candidate_q,
                            profile,
                            phase="final_candidate",
                        )
                    )
                    fallback_reason = self._fallback_reason_for_distance(min_distance_m)
                    if fallback_reason is not None and self.config.hold_last_safe_on_failure:
                        self.ik_solver.set_configuration_q(self._last_safe_full_q)
                        active_q = self._last_safe_full_q.copy()
                        status = "hold_last_safe"
                    else:
                        active_q = candidate_q.copy()
                        self.ik_solver.set_configuration_q(active_q)
                        near_boundary = (
                            min_distance_m is not None
                            and min_distance_m
                            <= float(self.config.collision_min_distance_m)
                            + float(self.config.collision_distance_epsilon_m)
                        )
                        status = "boundary_active" if near_boundary else "tracking"
                        self._last_safe_full_q = active_q.copy()
                    self._clear_hold()
            if target_already_reached:
                min_distance_m = (
                    branch_min_distance
                    if branch_min_distance is not None
                    else self._profiled_min_collision_distance(
                        candidate_q,
                        profile,
                        phase="target_hold_finalize",
                    )
                )
                fallback_reason = self._fallback_reason_for_distance(min_distance_m)
                active_q = candidate_q.copy()
                self.ik_solver.set_configuration_q(active_q)
                if fallback_reason is not None and self.config.hold_last_safe_on_failure:
                    active_q = self._last_safe_full_q.copy()
                    self.ik_solver.set_configuration_q(active_q)
                    status = "hold_last_safe"
                else:
                    near_boundary = (
                        min_distance_m is not None
                        and min_distance_m
                        <= float(self.config.collision_min_distance_m)
                        + float(self.config.collision_distance_epsilon_m)
                    )
                    status = "boundary_active" if near_boundary else "tracking"
                    self._last_safe_full_q = active_q.copy()
                self._clear_hold()
        except Exception as exc:
            fallback_reason = f"ik_exception:{type(exc).__name__}"
            if self.config.hold_last_safe_on_failure:
                self.ik_solver.set_configuration_q(self._last_safe_full_q)
                active_q = self._last_safe_full_q.copy()
                status = "hold_last_safe"
            else:
                self.ik_solver.set_configuration_q(pre_solve_q)
                active_q = pre_solve_q.copy()
                status = "unsafe_candidate"

        if status not in {"jump_gate_hold", "hold_last_safe", "unsafe_candidate"}:
            active_q, execution_step_metadata = self._limit_to_execution_step(
                reference_full_q=pre_solve_q,
                candidate_full_q=active_q,
            )
            self.ik_solver.set_configuration_q(active_q)
            if bool(execution_step_metadata.get("enabled")):
                min_distance_m = self._profiled_min_collision_distance(
                    active_q,
                    profile,
                    phase="execution_step",
                )
                fallback_reason = self._fallback_reason_for_distance(min_distance_m)
                if fallback_reason is not None and self.config.hold_last_safe_on_failure:
                    self.ik_solver.set_configuration_q(self._last_safe_full_q)
                    active_q = self._last_safe_full_q.copy()
                    status = "hold_last_safe"
                else:
                    near_boundary = (
                        min_distance_m is not None
                        and min_distance_m
                        <= float(self.config.collision_min_distance_m)
                        + float(self.config.collision_distance_epsilon_m)
                    )
                    status = "boundary_active" if near_boundary else "tracking"
                    self._last_safe_full_q = active_q.copy()

        left_qpos, right_qpos = self._split_solution(active_q)
        left_inactive_jitter_hold = self._inactive_jitter_hold_qpos(context.left_cartesian)
        if left_inactive_jitter_hold is not None:
            left_qpos = left_inactive_jitter_hold
            active_q = self._embed_arm_qpos(
                current_full_q=active_q,
                left_qpos=left_qpos,
                right_qpos=right_qpos,
            )
            self.ik_solver.set_configuration_q(active_q)
            self._last_safe_full_q = active_q.copy()
        if profile is not None:
            profile["frame_total_ms"] = (time.perf_counter() - frame_start) * 1000.0
        command.metadata.update(
            {
                "collision_avoidance_status": status,
                "collision_avoidance_fallback_reason": fallback_reason,
                "collision_distance_m": min_distance_m,
                "ik_branch_candidate": branch_candidate,
                "ik_branch_distance_score_deg": branch_score_deg,
                "ik_solve_iterations": int(solve_iteration_count),
                "ik_solve_pose_error": {
                    side: dict(metrics) for side, metrics in solve_pose_error.items()
                },
                "ik_jump_gate_status": jump_gate_status,
                "ik_jump_gate_reason": jump_gate_reason,
                "ik_jump_gate_release_ready": jump_gate_release_ready,
                "ik_jump_gate_hold_frames": jump_gate_hold_frames,
                "ik_jump_gate_metrics": jump_gate_metrics,
                "execution_step_metadata": dict(execution_step_metadata),
                "left_inactive_jitter_hold": left_inactive_jitter_hold is not None,
            }
        )
        command.add_command(
            EffectorCommand(
                effector_name=self.config.left_effector_name,
                side="left",
                command_type="joint_vector",
                value=left_qpos,
                metadata={
                    "controller": self.name,
                    "scene_profile": self.profile.name,
                    "collision_pair_count": len(self.profile.collision_pairs),
                    "collision_avoidance_mode": self.config.contact_mode,
                    "collision_avoidance_status": status,
                    "collision_avoidance_fallback_reason": fallback_reason,
                    "collision_distance_m": min_distance_m,
                    "source_command_type": (
                        None
                        if context.left_cartesian is None
                        else context.left_cartesian.command_type
                    ),
                    "source_target_pose_base": left_source_target_pose_base,
                    "source_target_pose_desk": left_source_target_pose_desk,
                    "source_target_pose_operator": left_source_target_pose_operator,
                    "ik_objective_mode": self.ik_tuning.ik_objective_mode,
                    "ik_branch_candidate": branch_candidate,
                    "ik_branch_distance_score_deg": branch_score_deg,
                    "ik_solve_iterations": int(solve_iteration_count),
                    "ik_solve_pose_error": dict(solve_pose_error.get("left", {})),
                    "ik_jump_gate_status": jump_gate_status,
                    "ik_jump_gate_reason": jump_gate_reason,
                    "ik_jump_gate_release_ready": jump_gate_release_ready,
                    "ik_jump_gate_hold_frames": jump_gate_hold_frames,
                    "ik_jump_gate_metrics": dict(jump_gate_metrics.get("left", {})),
                    "execution_step_metadata": dict(execution_step_metadata),
                    "inactive_jitter_hold": left_inactive_jitter_hold is not None,
                },
            )
        )
        command.add_command(
            EffectorCommand(
                effector_name=self.config.right_effector_name,
                side="right",
                command_type="joint_vector",
                value=right_qpos,
                metadata={
                    "controller": self.name,
                    "scene_profile": self.profile.name,
                    "collision_pair_count": len(self.profile.collision_pairs),
                    "collision_avoidance_mode": self.config.contact_mode,
                    "collision_avoidance_status": status,
                    "collision_avoidance_fallback_reason": fallback_reason,
                    "collision_distance_m": min_distance_m,
                    "source_command_type": (
                        None
                        if context.right_cartesian is None
                        else context.right_cartesian.command_type
                    ),
                    "source_target_pose_base": right_source_target_pose_base,
                    "source_target_pose_desk": right_source_target_pose_desk,
                    "source_target_pose_operator": right_source_target_pose_operator,
                    "ik_objective_mode": self.ik_tuning.ik_objective_mode,
                    "ik_branch_candidate": branch_candidate,
                    "ik_branch_distance_score_deg": branch_score_deg,
                    "ik_solve_iterations": int(solve_iteration_count),
                    "ik_solve_pose_error": dict(solve_pose_error.get("right", {})),
                    "ik_jump_gate_status": jump_gate_status,
                    "ik_jump_gate_reason": jump_gate_reason,
                    "ik_jump_gate_release_ready": jump_gate_release_ready,
                    "ik_jump_gate_hold_frames": jump_gate_hold_frames,
                    "ik_jump_gate_metrics": dict(jump_gate_metrics.get("right", {})),
                    "execution_step_metadata": dict(execution_step_metadata),
                },
            )
        )
        self._last_target_poses = {
            "left": left_target.copy(),
            "right": right_target.copy(),
        }
        self._emit_profile(
            profile=profile,
            command=command,
            status=status,
            iterations=solve_iteration_count,
            branch_candidate=branch_candidate,
        )
        return command

    def retarget(
        self, observation: TeleopObservation, calibration: TeleopCalibration | None = None
    ) -> TeleopCommand:
        context = self.build_target_context(observation, calibration)
        return self.build_command_for_context(context)

    def _target_pose_world(
        self,
        *,
        command: EffectorCommand | None,
        fallback_frame_name: str,
        side: Literal["left", "right"],
        calibration: TeleopCalibration | None = None,
    ) -> np.ndarray:
        if command is None:
            return self.ik_solver.compute_frame_pose(fallback_frame_name)
        if command.command_type != "cartesian_pose":
            raise ValueError(
                f"{self.name} expected cartesian_pose from the composed arm retargeter, got {command.command_type}."
            )
        target_pose_desk = command.metadata.get("target_pose_desk")
        if target_pose_desk is not None:
            return self._solver_frame_target_pose(
                pose_world=as_pose_matrix(target_pose_desk),
                command=command,
                fallback_frame_name=fallback_frame_name,
            )
        pose_base = as_pose_matrix(command.value)
        if calibration is not None:
            arm = calibration.get_arm_for_side(side)
            if arm.T_D_B is not None:
                return self._solver_frame_target_pose(
                    pose_world=arm.T_D_B @ pose_base,
                    command=command,
                    fallback_frame_name=fallback_frame_name,
                )
        pose_transform = (
            self.profile.left_pose_transform
            if side == "left"
            else self.profile.right_pose_transform
        )
        return self._solver_frame_target_pose(
            pose_world=pose_transform @ pose_base,
            command=command,
            fallback_frame_name=fallback_frame_name,
        )

    def _inactive_jitter_hold_qpos(self, command: EffectorCommand | None) -> np.ndarray | None:
        if command is None:
            return None
        metadata = command.metadata or {}
        if not bool(metadata.get("jitter_arm")):
            return None
        if bool(metadata.get("jitter_active", True)):
            return None
        hold_qpos = metadata.get("hold_qpos")
        if hold_qpos is None:
            return None
        qpos = np.asarray(hold_qpos, dtype=float).reshape(-1)
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            return None
        return qpos.copy()

    def _solver_frame_target_pose(
        self,
        *,
        pose_world: np.ndarray,
        command: EffectorCommand,
        fallback_frame_name: str,
    ) -> np.ndarray:
        pose_world = as_pose_matrix(pose_world)
        tcp_frame = canonical_tcp_frame(command.metadata.get("tcp_frame", HAND_BASE_TCP_FRAME))
        if tcp_frame != HAND_BASE_TCP_FRAME:
            return pose_world
        if fallback_frame_name not in {"arm_link6", "left_arm_link6", "right_arm_link6"}:
            return pose_world
        return pose_world @ _ARM_LINK6_FROM_HAND_BASE

    def _solve_toward_targets(
        self,
        *,
        left_target: np.ndarray,
        right_target: np.ndarray,
        profile: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, int, dict[str, dict[str, float]]]:
        targets = [
            self._pose_to_target(left_target),
            self._pose_to_target(right_target),
        ]
        last_q = self.ik_solver.current_q.copy()
        pose_error: dict[str, dict[str, float]] = {}
        iterations = 0
        for iteration in range(1, int(self.config.solve_max_iterations) + 1):
            last_q = self.ik_solver.solve(targets=targets)
            self._record_solver_profile(profile)
            iterations = iteration
            current_left = self.ik_solver.compute_frame_pose(
                self.profile.left_frame_name,
                last_q,
            )
            current_right = self.ik_solver.compute_frame_pose(
                self.profile.right_frame_name,
                last_q,
            )
            left_translation_m, left_rotation_deg = pose_delta_metrics(current_left, left_target)
            right_translation_m, right_rotation_deg = pose_delta_metrics(current_right, right_target)
            pose_error = {
                "left": {
                    "translation_m": float(left_translation_m),
                    "rotation_deg": float(left_rotation_deg),
                },
                "right": {
                    "translation_m": float(right_translation_m),
                    "rotation_deg": float(right_rotation_deg),
                },
            }
            if (
                left_translation_m <= float(self.config.solve_translation_tolerance_m)
                and left_rotation_deg <= float(self.config.solve_orientation_tolerance_deg)
                and right_translation_m <= float(self.config.solve_translation_tolerance_m)
                and right_rotation_deg <= float(self.config.solve_orientation_tolerance_deg)
            ):
                break
        return last_q.copy(), iterations, pose_error

    def _fallback_reason_for_distance(self, min_distance_m: float | None) -> str | None:
        if min_distance_m is None:
            return None
        safe_distance = float(self.config.collision_min_distance_m)
        epsilon = float(self.config.collision_distance_epsilon_m)
        if self.config.contact_mode == "freeze_on_contact":
            if min_distance_m <= safe_distance + epsilon:
                return "freeze_on_contact"
            return None
        if min_distance_m < safe_distance - epsilon:
            return "collision_margin_low"
        return None

    def _embed_arm_qpos(
        self,
        *,
        current_full_q: np.ndarray,
        left_qpos: np.ndarray,
        right_qpos: np.ndarray,
    ) -> np.ndarray:
        full_q = np.asarray(current_full_q, dtype=float).copy()
        if self.profile.solution_layout == "dual_arm_hand" and full_q.shape[0] >= 24:
            full_q[0:6] = left_qpos
            full_q[18:24] = right_qpos
            return full_q
        if full_q.shape[0] == 14:
            full_q[1:7] = left_qpos
            full_q[8:14] = right_qpos
            return full_q
        if full_q.shape[0] == 12:
            full_q[0:6] = left_qpos
            full_q[6:12] = right_qpos
            return full_q
        raise ValueError(
            f"{self.name} cannot embed arm qpos into solver configuration shape {full_q.shape}."
        )

    def _split_solution(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray(qpos, dtype=float)
        if self.profile.solution_layout == "dual_arm_hand" and qpos.shape[0] >= 24:
            return qpos[0:6].copy(), qpos[18:24].copy()
        if qpos.shape[0] == 14:
            return qpos[1:7].copy(), qpos[8:14].copy()
        if qpos.shape[0] == 12:
            return qpos[0:6].copy(), qpos[6:12].copy()
        raise ValueError(f"{self.name} received unsupported solver output shape: {qpos.shape}")

    def _pose_to_target(self, pose: np.ndarray) -> np.ndarray:
        quat = R.from_matrix(as_pose_matrix(pose)[:3, :3]).as_quat()
        pose_matrix = as_pose_matrix(pose)
        return np.array(
            [
                pose_matrix[0, 3],
                pose_matrix[1, 3],
                pose_matrix[2, 3],
                quat[0],
                quat[1],
                quat[2],
                quat[3],
            ],
            dtype=float,
        )

    def _make_full_joint_weights(self, qpos: np.ndarray) -> np.ndarray:
        q = np.asarray(qpos, dtype=float).reshape(-1)
        weights = np.zeros_like(q)
        if self.profile.solution_layout == "dual_arm_hand" and q.shape[0] >= 24:
            weights[0:6] = self._arm_joint_weights
            weights[18:24] = self._arm_joint_weights
            return weights
        if q.shape[0] == 14:
            weights[1:7] = self._arm_joint_weights
            weights[8:14] = self._arm_joint_weights
            return weights
        if q.shape[0] == 12:
            weights[0:6] = self._arm_joint_weights
            weights[6:12] = self._arm_joint_weights
            return weights
        raise ValueError(f"{self.name} cannot build weights for solver configuration shape {q.shape}.")

    def _make_full_execution_joint_step(self, qpos: np.ndarray) -> np.ndarray | None:
        if not self.config.enforce_realistic_joint_step:
            return None
        left_step = self.config.left_execution_max_joint_step_rad
        right_step = self.config.right_execution_max_joint_step_rad
        if left_step is None or right_step is None:
            return None
        q = np.asarray(qpos, dtype=float).reshape(-1)
        max_step = np.zeros_like(q)
        if self.profile.solution_layout == "dual_arm_hand" and q.shape[0] >= 24:
            max_step[0:6] = left_step
            max_step[18:24] = right_step
            return max_step
        if q.shape[0] == 14:
            max_step[1:7] = left_step
            max_step[8:14] = right_step
            return max_step
        if q.shape[0] == 12:
            max_step[0:6] = left_step
            max_step[6:12] = right_step
            return max_step
        raise ValueError(
            f"{self.name} cannot build execution joint-step limits for solver shape {q.shape}."
        )

    def set_execution_max_joint_step(
        self,
        *,
        left_step_rad: Sequence[float] | np.ndarray,
        right_step_rad: Sequence[float] | np.ndarray,
    ) -> None:
        self.config.left_execution_max_joint_step_rad = _optional_joint_step(left_step_rad)
        self.config.right_execution_max_joint_step_rad = _optional_joint_step(right_step_rad)
        self._full_execution_joint_step = self._make_full_execution_joint_step(
            self.ik_solver.current_q
        )

    def _limit_to_execution_step(
        self,
        *,
        reference_full_q: np.ndarray,
        candidate_full_q: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object]]:
        reference_q = np.asarray(reference_full_q, dtype=float).reshape(-1)
        candidate_q = np.asarray(candidate_full_q, dtype=float).reshape(-1)
        metadata = {
            "enabled": bool(self.config.enforce_realistic_joint_step),
            "step_limited": False,
            "weighted_delta_deg": 0.0,
            "left_max_abs_delta_deg": 0.0,
            "right_max_abs_delta_deg": 0.0,
        }
        if not self.config.enforce_realistic_joint_step or self._full_execution_joint_step is None:
            return candidate_q.copy(), metadata
        limited_q = reference_q + np.clip(
            candidate_q - reference_q,
            -self._full_execution_joint_step,
            self._full_execution_joint_step,
        )
        step_limited = not np.allclose(limited_q, candidate_q, atol=1e-10)
        left_delta, right_delta = self._split_solution(limited_q - reference_q)
        metadata.update(
            {
                "step_limited": bool(step_limited),
                "weighted_delta_deg": float(
                    weighted_joint_distance_score(
                        reference_q=np.zeros_like(limited_q),
                        candidate_q=limited_q - reference_q,
                        joint_distance_weights=self._full_joint_weights,
                    )
                ),
                "left_max_abs_delta_deg": float(np.rad2deg(np.max(np.abs(left_delta)))),
                "right_max_abs_delta_deg": float(np.rad2deg(np.max(np.abs(right_delta)))),
            }
        )
        return limited_q.copy(), metadata

    def _full_joint_distance_score(self, reference_full_q: np.ndarray, candidate_full_q: np.ndarray) -> float:
        return weighted_joint_distance_score(
            reference_q=reference_full_q,
            candidate_q=candidate_full_q,
            joint_distance_weights=self._full_joint_weights,
        )

    def _branch_candidates(
        self,
        *,
        raw_full_q: np.ndarray,
        reference_full_q: np.ndarray,
    ) -> list[tuple[str, np.ndarray, float]]:
        raw_full = np.asarray(raw_full_q, dtype=float).reshape(-1)
        reference_full = np.asarray(reference_full_q, dtype=float).reshape(-1)
        raw_left, raw_right = self._split_solution(raw_full)
        ref_left, ref_right = self._split_solution(reference_full)
        left_min, left_max = self._left_arm_limits
        right_min, right_max = self._right_arm_limits
        left_alt = preferred_wrist_flip_candidate(
            raw_left,
            reference_q=ref_left,
            joint_distance_weights=self._arm_joint_weights,
            q_min=left_min,
            q_max=left_max,
            limit_tolerance_rad=float(self.ik_tuning.branch_normalization.limit_tolerance_rad),
        )
        right_alt = preferred_wrist_flip_candidate(
            raw_right,
            reference_q=ref_right,
            joint_distance_weights=self._arm_joint_weights,
            q_min=right_min,
            q_max=right_max,
            limit_tolerance_rad=float(self.ik_tuning.branch_normalization.limit_tolerance_rad),
        )

        candidates: list[tuple[str, np.ndarray]] = [("original", raw_full.copy())]
        if left_alt is not None:
            candidates.append(
                (
                    "left_flip",
                    self._embed_arm_qpos(
                        current_full_q=raw_full,
                        left_qpos=left_alt,
                        right_qpos=raw_right,
                    ),
                )
            )
        if right_alt is not None:
            candidates.append(
                (
                    "right_flip",
                    self._embed_arm_qpos(
                        current_full_q=raw_full,
                        left_qpos=raw_left,
                        right_qpos=right_alt,
                    ),
                )
            )
        if left_alt is not None and right_alt is not None:
            candidates.append(
                (
                    "both_flip",
                    self._embed_arm_qpos(
                        current_full_q=raw_full,
                        left_qpos=left_alt,
                        right_qpos=right_alt,
                    ),
                )
            )

        ranked: list[tuple[str, np.ndarray, float]] = []
        seen: list[np.ndarray] = []
        for name, candidate_q in candidates:
            if any(np.allclose(candidate_q, existing, atol=1e-8) for existing in seen):
                continue
            seen.append(candidate_q.copy())
            ranked.append(
                (
                    name,
                    candidate_q.copy(),
                    self._full_joint_distance_score(reference_full, candidate_q),
                )
            )
        ranked.sort(key=lambda item: item[2])
        return ranked

    def _select_normalized_candidate(
        self,
        *,
        raw_full_q: np.ndarray,
        reference_full_q: np.ndarray,
        profile: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, str, float, float | None]:
        ranked = self._branch_candidates(
            raw_full_q=raw_full_q,
            reference_full_q=reference_full_q,
        )
        candidate_name, candidate_q, score_deg = ranked[0]
        candidate_min_distance: float | None = None
        if (
            candidate_name != "original"
            and self.ik_tuning.dual_arm_recovery.check_collision_on_branch_switch
        ):
            candidate_min_distance = self._profiled_min_collision_distance(
                candidate_q,
                profile,
                phase="branch_switch",
            )
            if self._fallback_reason_for_distance(candidate_min_distance) is not None:
                for fallback_name, fallback_q, fallback_score in ranked:
                    if fallback_name == "original":
                        return fallback_q.copy(), fallback_name, float(fallback_score), None
        return candidate_q.copy(), candidate_name, float(score_deg), candidate_min_distance

    def _evaluate_jump_gate(
        self,
        *,
        reference_full_q: np.ndarray,
        candidate_full_q: np.ndarray,
        current_targets: dict[str, np.ndarray],
    ) -> dict[str, object]:
        jump_gate = self.ik_tuning.jump_gate
        relaxed = self._jump_hold_active
        per_joint_limit = (
            jump_gate.relaxed_per_joint_jump_deg
            if relaxed
            else jump_gate.per_joint_jump_deg
        )
        weighted_limit = (
            jump_gate.relaxed_weighted_jump_threshold_deg
            if relaxed
            else jump_gate.weighted_jump_threshold_deg
        )
        ref_left_q, ref_right_q = self._split_solution(reference_full_q)
        cand_left_q, cand_right_q = self._split_solution(candidate_full_q)

        metrics: dict[str, object] = {}
        triggered_sides: list[str] = []
        for side, ref_q, cand_q in (
            ("left", ref_left_q, cand_left_q),
            ("right", ref_right_q, cand_right_q),
        ):
            pose_translation_m = float("inf")
            pose_rotation_deg = float("inf")
            previous_target = self._last_target_poses.get(side)
            if previous_target is not None:
                pose_translation_m, pose_rotation_deg = pose_delta_metrics(
                    previous_target,
                    current_targets[side],
                )
            pose_is_small = bool(
                previous_target is not None
                and pose_translation_m <= float(jump_gate.small_pose_translation_m)
                and pose_rotation_deg <= float(jump_gate.small_pose_rotation_deg)
            )
            exceeded, per_joint_delta_deg, weighted_jump_deg = joint_jump_exceeds(
                reference_q=ref_q,
                candidate_q=cand_q,
                joint_distance_weights=self._arm_joint_weights,
                per_joint_jump_deg=per_joint_limit,
                weighted_jump_threshold_deg=weighted_limit,
            )
            if pose_is_small and exceeded:
                triggered_sides.append(side)
            metrics[side] = {
                "pose_delta_m": None if not np.isfinite(pose_translation_m) else pose_translation_m,
                "pose_delta_deg": None if not np.isfinite(pose_rotation_deg) else pose_rotation_deg,
                "pose_is_small": pose_is_small,
                "joint_jump_max_deg": float(np.max(per_joint_delta_deg)),
                "joint_jump_weighted_deg": float(weighted_jump_deg),
                "joint_jump_exceeded": bool(exceeded),
            }

        release_ready = False
        release_metrics: dict[str, object] = {}
        if self._jump_hold_active and self._jump_hold_anchor_targets is not None:
            for side in ("left", "right"):
                release_translation_m, release_rotation_deg = pose_delta_metrics(
                    self._jump_hold_anchor_targets[side],
                    current_targets[side],
                )
                side_ready = bool(
                    release_translation_m >= float(jump_gate.release_translation_m)
                    or release_rotation_deg >= float(jump_gate.release_rotation_deg)
                )
                release_ready = release_ready or side_ready
                release_metrics[side] = {
                    "translation_m": release_translation_m,
                    "rotation_deg": release_rotation_deg,
                    "ready": side_ready,
                }
        metrics["release"] = release_metrics

        hold = bool(triggered_sides)
        if hold:
            status = "holding" if self._jump_hold_active else "hold_entered"
            reason = f"small_pose_large_q_jump:{','.join(triggered_sides)}"
        elif self._jump_hold_active and release_ready:
            status = "released"
            reason = "hold_anchor_released"
        elif self._jump_hold_active:
            status = "recovering"
            reason = None
        else:
            status = "pass"
            reason = None
        return {
            "hold": hold,
            "status": status,
            "reason": reason,
            "release_ready": release_ready,
            "metrics": metrics,
        }

    def _enter_hold(
        self,
        *,
        hold_full_q: np.ndarray,
        current_targets: dict[str, np.ndarray],
    ) -> None:
        if not self._jump_hold_active:
            self._jump_hold_active = True
            self._jump_hold_full_q = np.asarray(hold_full_q, dtype=float).copy()
            self._jump_hold_anchor_targets = {
                side: pose.copy()
                for side, pose in current_targets.items()
            }
            self._jump_hold_frames = 0
        self._jump_hold_frames += 1

    def _clear_hold(self) -> None:
        self._jump_hold_active = False
        self._jump_hold_full_q = None
        self._jump_hold_anchor_targets = None
        self._jump_hold_frames = 0

    def _hold_seed_candidates(self) -> list[tuple[str, np.ndarray]]:
        if self._jump_hold_full_q is None:
            return []
        hold_q = self._jump_hold_full_q.copy()
        hold_left, hold_right = self._split_solution(hold_q)
        left_min, left_max = self._left_arm_limits
        right_min, right_max = self._right_arm_limits
        left_alt = preferred_wrist_flip_candidate(
            hold_left,
            reference_q=hold_left,
            joint_distance_weights=self._arm_joint_weights,
            q_min=left_min,
            q_max=left_max,
            limit_tolerance_rad=float(self.ik_tuning.branch_normalization.limit_tolerance_rad),
        )
        right_alt = preferred_wrist_flip_candidate(
            hold_right,
            reference_q=hold_right,
            joint_distance_weights=self._arm_joint_weights,
            q_min=right_min,
            q_max=right_max,
            limit_tolerance_rad=float(self.ik_tuning.branch_normalization.limit_tolerance_rad),
        )
        seed_map = {"current": hold_q.copy()}
        if left_alt is not None:
            seed_map["left_flip"] = self._embed_arm_qpos(
                current_full_q=hold_q,
                left_qpos=left_alt,
                right_qpos=hold_right,
            )
        if right_alt is not None:
            seed_map["right_flip"] = self._embed_arm_qpos(
                current_full_q=hold_q,
                left_qpos=hold_left,
                right_qpos=right_alt,
            )
        if left_alt is not None and right_alt is not None:
            seed_map["both_flip"] = self._embed_arm_qpos(
                current_full_q=hold_q,
                left_qpos=left_alt,
                right_qpos=right_alt,
            )

        ordered: list[tuple[str, np.ndarray]] = []
        for seed_name in self.ik_tuning.dual_arm_recovery.seed_modes:
            candidate = seed_map.get(seed_name)
            if candidate is None:
                continue
            if any(np.allclose(candidate, existing, atol=1e-8) for _, existing in ordered):
                continue
            ordered.append((seed_name, candidate.copy()))
        if not ordered:
            ordered.append(("current", hold_q.copy()))
        return ordered

    def _attempt_hold_recovery(
        self,
        *,
        current_targets: dict[str, np.ndarray],
        profile: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        if not self._jump_hold_active or self._jump_hold_full_q is None:
            return None
        if not self.ik_tuning.dual_arm_recovery.enable_seed_expansion:
            return None
        if self._jump_hold_frames < int(self.ik_tuning.jump_gate.seed_expansion_after_hold_frames):
            return None

        reference_full_q = self._jump_hold_full_q.copy()
        best_result: dict[str, object] | None = None
        seen_candidates: list[np.ndarray] = []
        try:
            for seed_name, seed_q in self._hold_seed_candidates():
                self.ik_solver.set_configuration_q(seed_q)
                try:
                    raw_q = self.ik_solver.solve(
                        targets=[
                            self._pose_to_target(current_targets["left"]),
                            self._pose_to_target(current_targets["right"]),
                        ]
                    )
                    self._record_solver_profile(profile)
                except Exception:
                    continue
                for branch_name, candidate_q, score_deg in self._branch_candidates(
                    raw_full_q=raw_q,
                    reference_full_q=reference_full_q,
                ):
                    if any(np.allclose(candidate_q, existing, atol=1e-8) for existing in seen_candidates):
                        continue
                    seen_candidates.append(candidate_q.copy())
                    gate_result = self._evaluate_jump_gate(
                        reference_full_q=reference_full_q,
                        candidate_full_q=candidate_q,
                        current_targets=current_targets,
                    )
                    if bool(gate_result["hold"]) and not bool(gate_result["release_ready"]):
                        continue
                    min_distance_m = self._profiled_min_collision_distance(
                        candidate_q,
                        profile,
                        phase="hold_recovery_candidate",
                    )
                    if self._fallback_reason_for_distance(min_distance_m) is not None:
                        continue
                    candidate_name = (
                        seed_name if branch_name == "original" else f"{seed_name}+{branch_name}"
                    )
                    best_result = {
                        "q": candidate_q.copy(),
                        "min_distance_m": min_distance_m,
                        "candidate_name": candidate_name,
                        "score_deg": float(score_deg),
                        "jump_gate_status": gate_result["status"],
                        "jump_gate_reason": gate_result["reason"],
                        "release_ready": gate_result["release_ready"],
                        "metrics": gate_result["metrics"],
                    }
                    return best_result
        finally:
            self.ik_solver.set_configuration_q(reference_full_q)
        return best_result
