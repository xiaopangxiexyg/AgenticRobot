"""Canonical desk TCP targets through the shared RM65 position-control chain.

This module is the simulation composition layer for the same control stages used
by the real position runtime: timestamped Cartesian references, coupled 60 Hz
collision-aware IK, 120 Hz joint commands, physical actuators, and MuJoCo
physics.  Dataset parsing and viewer policy deliberately live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dataset_contracts.pose_frames import PoseFrameContract, load_pose_frame_contract
from project_paths import resolve_path
from robot_motion.position_control import (
    PoseActionMotionIntentStateMachine,
    ResampledMotionReference,
)
from robot_motion.rm65 import (
    DEFAULT_COLLISION_SCENE,
    DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS,
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    IKObjectiveWeights,
    LEGACY_POSITION_CONTROL_PROFILE_NAME,
    RM65_BIMANUAL_POSITION_V1,
    RM65_BIMANUAL_POSITION_V1_NAME,
    load_rm65_tool_profile,
    load_ik_tuning_config,
    tool_tcp_pose_flange_m_rad,
)
from robot_motion.rm65.dual_arm_target_solver import (
    BimanualIKResult,
    BimanualPoseTarget,
    BimanualPoseTargetSource,
)
from robot_motion.rm65.position_collision import load_world_from_arm_base_transforms
from robot_motion.runtime import EffectorCommand, TeleopCommand
from sim.contracts import SimulationContract, TimingSpec
from ..actuation import (
    cad_bimanual_dmtac_gripper_profile,
    rm65_bimanual_udp_feedback_profile,
)
from ..adapters import MujocoPositionExecutionConfig, MujocoPositionExecutionRig
from ..scene import MujocoScene


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SIMULATION_CONTRACT_PATH = _REPO_ROOT / "sim/configs/rm65_bimanual_physics_v1.json"
DEFAULT_STATION_POSE_CONTRACT_PATH = (
    _REPO_ROOT
    / "robot_motion/configs/pose_frames/"
    "rm65_dual_arm_workspace_near_boundary_v1_collection_desk_absolute.json"
)
DEFAULT_TOOL_PROFILE_BY_SIDE = {
    side: _REPO_ROOT
    / f"hardware/rm65/utils/configs/calibrations/rm65_{side}_DMTAC_M.json"
    for side in ("left", "right")
}
DEFAULT_INSPIRE_V3_DUAL_ARM_URDF = (
    _REPO_ROOT
    / "assets/rm65_inspire_gripper_v3/"
    "rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf"
)
# MuJoCo-safe dual-arm reset used by the collision-enabled task scenes.  The
# historical visualizer fallback has right J4=3.10 rad, outside this URDF's
# physical joint limit, and must not seed a physical replay.
DEFAULT_SIM_HOME_QPOS_BY_SIDE = {
    "left": np.array(
        [
            1.5853514942,
            0.1381550277,
            2.1352611562,
            0.0699109098,
            -0.7493894092,
            -0.0572276017,
        ]
    ),
    "right": np.array(
        [
            -1.4983346097,
            0.0517420315,
            2.1069143897,
            0.0504173256,
            -0.6155243686,
            -0.0529943045,
        ]
    ),
}


def _plain_metadata(value: Any) -> Any:
    """Copy immutable shared-reference metadata into owned plain containers."""

    if isinstance(value, Mapping):
        return {str(key): _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_metadata(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.asarray(value).copy()
    return value


def position_15_60_120_timing(
    base: TimingSpec,
    *,
    parameter_source: str = RM65_BIMANUAL_POSITION_V1_NAME,
) -> TimingSpec:
    """Bind a scene's physics/camera clocks to the certified position rates."""

    return TimingSpec(
        sample_hz=15.0,
        policy_hz=15.0,
        reference_hz=60.0,
        ik_hz=60.0,
        command_hz=120.0,
        physics_hz=base.physics_hz,
        camera_hz=base.camera_hz,
        interpolation=base.interpolation,
        command_hold=base.command_hold,
        parameter_source=str(parameter_source),
    )


def _station_world_from_desk(
    *,
    station_contract: PoseFrameContract,
    world_from_base_by_side: Mapping[str, np.ndarray],
) -> np.ndarray:
    if station_contract.pose_storage_frame != "desk_absolute":
        raise ValueError("station pose contract must store canonical desk_absolute poses")
    if station_contract.canonical_absolute_frame != "desk":
        raise ValueError("station pose contract canonical frame must be 'desk'")
    candidates = []
    for side in ("left", "right"):
        desk_from_base = (
            station_contract.source.effective_raw_to_canonical_desk_transform(side)
        )
        candidates.append(
            np.asarray(world_from_base_by_side[side], dtype=float)
            @ np.linalg.inv(desk_from_base)
        )
    if not np.allclose(candidates[0], candidates[1], atol=1e-8, rtol=0.0):
        raise ValueError("left/right station contracts disagree on world_from_desk")
    return candidates[0]


@dataclass(frozen=True)
class DeskTcpPositionControlConfig:
    simulation_contract_path: Path = DEFAULT_SIMULATION_CONTRACT_PATH
    station_pose_contract_path: Path = DEFAULT_STATION_POSE_CONTRACT_PATH
    left_tool_profile: Path = DEFAULT_TOOL_PROFILE_BY_SIDE["left"]
    right_tool_profile: Path = DEFAULT_TOOL_PROFILE_BY_SIDE["right"]
    realtime_worker: bool = False
    position_control_profile_name: str = RM65_BIMANUAL_POSITION_V1_NAME
    solve_max_iterations: int = RM65_BIMANUAL_POSITION_V1.ik_execution.solve_max_iterations
    solve_min_iterations: int = RM65_BIMANUAL_POSITION_V1.ik_execution.solve_min_iterations
    solve_adaptive_iterations_enabled: bool = (
        RM65_BIMANUAL_POSITION_V1.ik_execution.adaptive_iterations
    )
    solve_iteration_reserve_ms: float = (
        RM65_BIMANUAL_POSITION_V1.ik_execution.iteration_reserve_ms
    )
    damping: float = RM65_BIMANUAL_POSITION_V1.ik_execution.damping
    orientation_weight: float = RM65_BIMANUAL_POSITION_V1.ik_execution.orientation_weight
    joint_velocity_limit_profile: str = (
        RM65_BIMANUAL_POSITION_V1.ik_execution.joint_velocity_limit_profile
    )
    max_command_step_deg: float = (
        RM65_BIMANUAL_POSITION_V1.command_execution.max_command_step_deg
    )
    target_velocity_filter_tau_sec: float = (
        RM65_BIMANUAL_POSITION_V1.command_execution.target_velocity_filter_tau_sec
    )
    follow_qpos_alpha: float = (
        RM65_BIMANUAL_POSITION_V1.command_execution.follow_qpos_alpha
    )
    post_ik_qpos_smoothing_alpha: float = 1.0
    post_ik_qpos_smoothing_adaptive_enabled: bool = False
    post_ik_qpos_smoothing_fast_alpha: float = 1.0
    left_home_qpos: tuple[float, ...] = tuple(DEFAULT_SIM_HOME_QPOS_BY_SIDE["left"])
    right_home_qpos: tuple[float, ...] = tuple(DEFAULT_SIM_HOME_QPOS_BY_SIDE["right"])

    def __post_init__(self) -> None:
        profile_name = str(self.position_control_profile_name).strip()
        if profile_name not in {
            RM65_BIMANUAL_POSITION_V1_NAME,
            LEGACY_POSITION_CONTROL_PROFILE_NAME,
        }:
            raise ValueError(f"unsupported position_control_profile_name: {profile_name!r}")
        object.__setattr__(self, "position_control_profile_name", profile_name)
        for name in ("left_home_qpos", "right_home_qpos"):
            value = np.asarray(getattr(self, name), dtype=float).reshape(-1)
            if value.shape != (6,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain six finite joint values")
        if not np.isfinite(float(self.orientation_weight)) or float(
            self.orientation_weight
        ) < 0.0:
            raise ValueError("orientation_weight must be finite and >= 0")


def _replay_ik_tuning(orientation_weight: float):
    """Return a replay-local objective without changing shared/teleop defaults."""

    tuning = load_ik_tuning_config()
    weight = float(orientation_weight)
    mode_name = "mujoco_replay_position_2j"
    objective_modes = dict(tuning.objective_modes)
    objective_modes[mode_name] = IKObjectiveWeights(
        single_arm_position_weight=1.0,
        single_arm_orientation_weight=weight,
        dual_arm_position_cost=1.0,
        dual_arm_orientation_cost=weight,
    )
    return replace(
        tuning,
        ik_objective_mode=mode_name,
        objective_modes=objective_modes,
    )


class DeskTcpPositionControl:
    """Own one deterministic desk-TCP -> MuJoCo position execution chain."""

    def __init__(
        self,
        *,
        config: DeskTcpPositionControlConfig,
        contract: SimulationContract,
        station_pose_contract: PoseFrameContract,
        scene: MujocoScene,
        retargeter: DualArmCollisionAwareRetargeter,
        rig: MujocoPositionExecutionRig,
        world_from_desk: np.ndarray,
        world_to_base_by_side: Mapping[str, np.ndarray],
    ) -> None:
        self.config = config
        self.contract = contract
        self.station_pose_contract = station_pose_contract
        self.scene = scene
        self.retargeter = retargeter
        self.rig = rig
        self.world_from_desk = np.asarray(world_from_desk, dtype=float).copy()
        self.world_to_base_by_side = {
            side: np.asarray(value, dtype=float).copy()
            for side, value in world_to_base_by_side.items()
        }
        self.reference_count = 0
        self.status_counts: dict[str, int] = {}
        self._pose_action_motion_intent = {
            side: PoseActionMotionIntentStateMachine()
            for side in ("left", "right")
        }
        self._pose_action_motion_state_counts = {
            side: {} for side in ("left", "right")
        }
        self._output_qpos_smoothing_reason_counts = {
            side: {} for side in ("left", "right")
        }
        self._output_qpos_smoothing_alpha_stats = {
            side: {"count": 0, "sum": 0.0, "min": 1.0, "max": 0.0}
            for side in ("left", "right")
        }
        self._height_error = {
            source: {
                side: {"count": 0, "sum_signed_m": 0.0, "sum_abs_m": 0.0, "max_abs_m": 0.0}
                for side in ("left", "right")
            }
            for source in ("ik_reference", "mujoco_actual")
        }

    def _record_height_error(self, source: str, side: str, error_m: float) -> None:
        stats = self._height_error[source][side]
        stats["count"] += 1
        stats["sum_signed_m"] += float(error_m)
        stats["sum_abs_m"] += abs(float(error_m))
        stats["max_abs_m"] = max(float(stats["max_abs_m"]), abs(float(error_m)))

    @classmethod
    def build(
        cls,
        config: DeskTcpPositionControlConfig | None = None,
    ) -> "DeskTcpPositionControl":
        cfg = config or DeskTcpPositionControlConfig()
        base_contract = SimulationContract.load(cfg.simulation_contract_path)
        timing = position_15_60_120_timing(
            base_contract.timing,
            parameter_source=cfg.position_control_profile_name,
        )
        contract = replace(base_contract, timing=timing)
        arm_profile = rm65_bimanual_udp_feedback_profile()
        # The canonical desk scene now embeds the official six-joint CAD
        # mechanism.  Keep the identified two-joint proxy profile available
        # for explicit legacy URDF callers, but bind this default chain to the
        # matching CAD actuator layout.
        gripper_profile = cad_bimanual_dmtac_gripper_profile()
        scene = MujocoScene.from_contract(
            contract,
            position_actuation_profile=arm_profile,
            gripper_actuation_profile=gripper_profile,
        )
        try:
            scene.set_qpos_by_side(
                {
                    "left": np.asarray(cfg.left_home_qpos, dtype=float),
                    "right": np.asarray(cfg.right_home_qpos, dtype=float),
                }
            )
            projector = scene.collision_projector()
            if projector is not None:
                check = projector.check_qpos(
                    scene.data, scene.kinematic_controlled_qpos()
                )
                if not check.safe:
                    raise RuntimeError(
                        "simulation home qpos is collision-unsafe: "
                        f"{check.to_dict()}"
                    )

            urdf_path = resolve_path(contract.scene.urdf_path)
            if urdf_path.resolve() != DEFAULT_INSPIRE_V3_DUAL_ARM_URDF.resolve():
                raise ValueError(
                    "desk TCP replay requires the calibrated Inspire-v3 dual-arm URDF: "
                    f"expected={DEFAULT_INSPIRE_V3_DUAL_ARM_URDF} actual={urdf_path}"
                )
            world_from_base = load_world_from_arm_base_transforms(urdf_path)
            world_to_base = {
                side: np.linalg.inv(world_from_base[side]) for side in ("left", "right")
            }
            station_contract = load_pose_frame_contract(cfg.station_pose_contract_path)
            world_from_desk = _station_world_from_desk(
                station_contract=station_contract,
                world_from_base_by_side=world_from_base,
            )
            initial = {
                side: scene.get_arm_qpos(side).copy() for side in ("left", "right")
            }

            def tool_tcp(side: str, path: Path) -> np.ndarray:
                return tool_tcp_pose_flange_m_rad(
                    load_rm65_tool_profile(path, side=side)
                )

            retargeter = DualArmCollisionAwareRetargeter(
                DualArmCollisionAwareRetargeterConfig(
                    # Reuse the exact collision profile used by real position
                    # teleoperation. The MuJoCo scene above remains the full
                    # Inspire-v3 model; this shared proxy URDF is only the
                    # coupled IK collision model.
                    scene_profile=DEFAULT_COLLISION_SCENE,
                    solver_frequency=timing.ik_hz,
                    damping=float(cfg.damping),
                    solve_max_iterations=int(cfg.solve_max_iterations),
                    solve_min_iterations=int(cfg.solve_min_iterations),
                    solve_adaptive_iterations_enabled=bool(
                        cfg.solve_adaptive_iterations_enabled
                    ),
                    solve_iteration_reserve_ms=float(
                        cfg.solve_iteration_reserve_ms
                    ),
                    ik_tuning=_replay_ik_tuning(cfg.orientation_weight),
                    bimanual_result_contract_enabled=True,
                    enforce_realistic_joint_step=True,
                    left_initial_qpos=initial["left"],
                    right_initial_qpos=initial["right"],
                    left_tool_tcp_pose_flange_m_rad=tool_tcp(
                        "left", cfg.left_tool_profile
                    ),
                    right_tool_tcp_pose_flange_m_rad=tool_tcp(
                        "right", cfg.right_tool_profile
                    ),
                    joint_velocity_limit_profile=cfg.joint_velocity_limit_profile,
                    output_qpos_smoothing_alpha=(
                        0.14
                        if cfg.position_control_profile_name
                        == LEGACY_POSITION_CONTROL_PROFILE_NAME
                        else float(cfg.post_ik_qpos_smoothing_alpha)
                    ),
                    output_qpos_smoothing_adaptive_enabled=(
                        True
                        if cfg.position_control_profile_name
                        == LEGACY_POSITION_CONTROL_PROFILE_NAME
                        else bool(cfg.post_ik_qpos_smoothing_adaptive_enabled)
                    ),
                    output_qpos_smoothing_fast_alpha=(
                        0.45
                        if cfg.position_control_profile_name
                        == LEGACY_POSITION_CONTROL_PROFILE_NAME
                        else float(cfg.post_ik_qpos_smoothing_fast_alpha)
                    ),
                    inactive_side_hold_enabled=(
                        cfg.position_control_profile_name
                        == LEGACY_POSITION_CONTROL_PROFILE_NAME
                    ),
                    inactive_side_hold_execution_cap_enabled=(
                        cfg.position_control_profile_name
                        == LEGACY_POSITION_CONTROL_PROFILE_NAME
                    ),
                    inactive_side_hold_execution_release_enabled=(
                        cfg.position_control_profile_name
                        == LEGACY_POSITION_CONTROL_PROFILE_NAME
                    ),
                    prefer_non_wrist_singular_branch=True,
                    prefer_non_elbow_singular_branch=True,
                    local_step_candidate_clip_scale=8.0,
                )
            )
            if retargeter.profile.name != DEFAULT_COLLISION_SCENE:
                raise RuntimeError(
                    "replay collision profile diverged from real teleoperation: "
                    f"expected={DEFAULT_COLLISION_SCENE!r} "
                    f"actual={retargeter.profile.name!r}"
                )
            if tuple(retargeter.profile.collision_pairs) != (
                DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS
            ):
                raise RuntimeError(
                    "replay must use the exact 12 collision pairs shared with "
                    "real teleoperation"
                )
            rig = MujocoPositionExecutionRig(
                scene,
                MujocoPositionExecutionConfig(
                    timing=timing,
                    max_command_step_deg=float(cfg.max_command_step_deg),
                    target_velocity_filter_tau_sec=float(
                        cfg.target_velocity_filter_tau_sec
                    ),
                    follow_qpos_alpha=float(cfg.follow_qpos_alpha),
                    actuation_profile=arm_profile,
                    gripper_actuation_profile=gripper_profile,
                    realtime_worker=bool(cfg.realtime_worker),
                ),
            )
            rig.start()
            return cls(
                config=cfg,
                contract=contract,
                station_pose_contract=station_contract,
                scene=scene,
                retargeter=retargeter,
                rig=rig,
                world_from_desk=world_from_desk,
                world_to_base_by_side=world_to_base,
            )
        except BaseException:
            close = getattr(scene, "close", None)
            if callable(close):
                close()
            raise

    def solve_and_dispatch(
        self,
        reference: ResampledMotionReference,
        *,
        gripper_close_by_side: Mapping[str, float],
    ) -> BimanualIKResult:
        if set(reference.pose_by_side) != {"left", "right"}:
            raise ValueError("desk TCP reference must contain left and right poses")
        if set(gripper_close_by_side) != {"left", "right"}:
            raise ValueError("gripper targets must contain left and right")

        accepted = self.rig.accepted_qcmd_by_side()
        self.retargeter.set_current_qpos_pair(
            left_qpos=accepted["left"],
            right_qpos=accepted["right"],
            reset_tracking=False,
        )
        tool_world = self.tool_tcp_world_targets(reference)
        solver_targets = {
            side: self.retargeter.solver_world_target_from_tool_tcp_world_target(
                side=side,
                tool_tcp_world_target=tool_world[side],
            )
            for side in ("left", "right")
        }
        interval_velocity = dict(
            reference.metadata.get("pose_action_interval_velocity", {}) or {}
        )
        velocity_by_side = interval_velocity.get("by_side", {})
        velocity_by_side = (
            dict(velocity_by_side) if isinstance(velocity_by_side, Mapping) else {}
        )
        source_by_side = {}
        for side in ("left", "right"):
            side_velocity = velocity_by_side.get(side)
            side_velocity = (
                dict(side_velocity) if isinstance(side_velocity, Mapping) else {}
            )
            motion_intent = self._pose_action_motion_intent[side].update(
                timestamp=float(reference.timestamp),
                pose=np.asarray(reference.pose_by_side[side], dtype=float),
                interval_velocity=side_velocity,
            )
            state = str(motion_intent.get("state", "UNKNOWN"))
            counts = self._pose_action_motion_state_counts[side]
            counts[state] = int(counts.get(state, 0)) + 1
            source_by_side[side] = BimanualPoseTargetSource(
                command_type="cartesian_pose",
                value=solver_targets[side],
                frame_name="world",
                metadata={
                    "pose_action_motion_intent": motion_intent,
                    "pose_action_interval_velocity": side_velocity,
                },
            )
        result = self.retargeter.solve_target_pair(
            BimanualPoseTarget(
                timestamp=float(reference.timestamp),
                target_pose_by_side=solver_targets,
                world_to_base_by_side=self.world_to_base_by_side,
                source_by_side=source_by_side,
                target_smoothing_metadata={
                    "enabled": True,
                    "source": "robot_motion.position_control.resampling",
                    **_plain_metadata(reference.metadata),
                },
            )
        )
        actual_qpos = {
            side: self.scene.get_arm_qpos(side).copy() for side in ("left", "right")
        }
        actual_tool_world = self.retargeter.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=actual_qpos["left"], right_qpos=actual_qpos["right"]
        )
        for side in ("left", "right"):
            target_z = float(tool_world[side][2, 3])
            self._record_height_error(
                "ik_reference",
                side,
                float(result.side_results[side].tool_tcp_pose_world[2, 3]) - target_z,
            )
            self._record_height_error(
                "mujoco_actual",
                side,
                float(actual_tool_world[side][2, 3]) - target_z,
            )
        command = TeleopCommand(
            timestamp=result.timestamp, metadata=dict(result.metadata)
        )
        for side in ("left", "right"):
            side_result = result.side_results[side]
            command.add_command(
                EffectorCommand(
                    effector_name=f"{side}_arm",
                    side=side,
                    command_type="joint_vector",
                    value=side_result.qpos.copy(),
                    metadata=dict(side_result.metadata),
                )
            )
            close = float(gripper_close_by_side[side])
            if not np.isfinite(close):
                raise ValueError(f"{side} gripper target must be finite")
            command.add_command(
                EffectorCommand(
                    effector_name=f"{side}_hand",
                    side=side,
                    command_type="gripper_scalar",
                    value=np.array([np.clip(close, 0.0, 1.0)], dtype=float),
                    metadata={"gripper_slot": 0, "source": "desk_tcp_position"},
                )
            )
        self.rig.send_commands(command)
        self.reference_count += 1
        self.status_counts[result.status] = (
            self.status_counts.get(result.status, 0) + 1
        )
        smoothing = dict(result.metadata.get("output_qpos_smoothing", {}) or {})
        adaptive = dict(smoothing.get("adaptive", {}) or {})
        alpha_by_side = dict(smoothing.get("alpha_by_side", {}) or {})
        for side in ("left", "right"):
            side_adaptive = dict(adaptive.get(side, {}) or {})
            reason = str(side_adaptive.get("reason", "missing"))
            reason_counts = self._output_qpos_smoothing_reason_counts[side]
            reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
            alpha = float(alpha_by_side.get(side, self.config.follow_qpos_alpha))
            stats = self._output_qpos_smoothing_alpha_stats[side]
            stats["count"] = int(stats["count"]) + 1
            stats["sum"] = float(stats["sum"]) + alpha
            stats["min"] = min(float(stats["min"]), alpha)
            stats["max"] = max(float(stats["max"]), alpha)
        return result

    def tool_tcp_world_targets(
        self, reference: ResampledMotionReference
    ) -> dict[str, np.ndarray]:
        """Return the exact public TCP targets supplied to IK, without mutation."""

        if set(reference.pose_by_side) != {"left", "right"}:
            raise ValueError("desk TCP reference must contain left and right poses")
        return {
            side: self.world_from_desk @ np.asarray(reference.pose_by_side[side])
            for side in ("left", "right")
        }

    def step_physics(self) -> None:
        self.rig.step_physics()

    def snapshot(self) -> dict[str, object]:
        height_error = {}
        for source, by_side in self._height_error.items():
            height_error[source] = {}
            for side, values in by_side.items():
                count = int(values["count"])
                height_error[source][side] = {
                    "count": count,
                    "mean_signed_m": (
                        None if count == 0 else float(values["sum_signed_m"]) / count
                    ),
                    "mean_abs_m": (
                        None if count == 0 else float(values["sum_abs_m"]) / count
                    ),
                    "max_abs_m": float(values["max_abs_m"]),
                }
        return {
            "position_control_profile": (
                RM65_BIMANUAL_POSITION_V1.to_metadata()
                if self.config.position_control_profile_name
                == RM65_BIMANUAL_POSITION_V1_NAME
                else {
                    "name": LEGACY_POSITION_CONTROL_PROFILE_NAME,
                    "compatibility_mode": True,
                }
            ),
            "main_action_contract": "accepted_qref_fk_pose",
            "source_shaping_reapplied": (
                self.config.position_control_profile_name
                == LEGACY_POSITION_CONTROL_PROFILE_NAME
            ),
            "rates_hz": {
                "sample": self.contract.timing.sample_hz,
                "ik_reference": self.contract.timing.ik_hz,
                "command": self.contract.timing.command_hz,
                "physics": self.contract.timing.physics_hz,
            },
            "reference_count": self.reference_count,
            "ik_status_counts": dict(self.status_counts),
            "pose_action_motion_intent": {
                "speed_source": "adjacent_pose_action_interval",
                "state_counts_by_side": {
                    side: dict(counts)
                    for side, counts in self._pose_action_motion_state_counts.items()
                },
                "config": self._pose_action_motion_intent[
                    "left"
                ].config.to_metadata(),
            },
            "output_qpos_smoothing_speed_evidence": {
                side: {
                    "reason_counts": dict(
                        self._output_qpos_smoothing_reason_counts[side]
                    ),
                    "alpha": {
                        "count": int(
                            self._output_qpos_smoothing_alpha_stats[side]["count"]
                        ),
                        "mean": (
                            None
                            if int(
                                self._output_qpos_smoothing_alpha_stats[side][
                                    "count"
                                ]
                            )
                            == 0
                            else float(
                                self._output_qpos_smoothing_alpha_stats[side]["sum"]
                            )
                            / int(
                                self._output_qpos_smoothing_alpha_stats[side][
                                    "count"
                                ]
                            )
                        ),
                        "min": (
                            None
                            if int(
                                self._output_qpos_smoothing_alpha_stats[side][
                                    "count"
                                ]
                            )
                            == 0
                            else float(
                                self._output_qpos_smoothing_alpha_stats[side]["min"]
                            )
                        ),
                        "max": (
                            None
                            if int(
                                self._output_qpos_smoothing_alpha_stats[side][
                                    "count"
                                ]
                            )
                            == 0
                            else float(
                                self._output_qpos_smoothing_alpha_stats[side]["max"]
                            )
                        ),
                    },
                }
                for side in ("left", "right")
            },
            "retargeter": type(self.retargeter).__name__,
            "retargeter_scene_profile": self.retargeter.profile.name,
            "retargeter_urdf_path": str(self.retargeter.profile.urdf_path),
            "retargeter_collision_pair_count": len(
                self.retargeter.profile.collision_pairs
            ),
            "replay_position_ik_profile": {
                "profile": self.config.position_control_profile_name,
                "solve_min_iterations": self.config.solve_min_iterations,
                "solve_max_iterations": self.config.solve_max_iterations,
                "solve_iteration_reserve_ms": (
                    self.config.solve_iteration_reserve_ms
                ),
                "orientation_weight": self.config.orientation_weight,
                "damping": self.config.damping,
                "joint_velocity_limit_profile": (
                    self.config.joint_velocity_limit_profile
                ),
                "post_ik_qpos_smoothing_alpha": (
                    0.14
                    if self.config.position_control_profile_name
                    == LEGACY_POSITION_CONTROL_PROFILE_NAME
                    else self.config.post_ik_qpos_smoothing_alpha
                ),
                "post_ik_qpos_smoothing_adaptive_enabled": (
                    True
                    if self.config.position_control_profile_name
                    == LEGACY_POSITION_CONTROL_PROFILE_NAME
                    else self.config.post_ik_qpos_smoothing_adaptive_enabled
                ),
                "inactive_side_hold_reapplied": (
                    self.config.position_control_profile_name
                    == LEGACY_POSITION_CONTROL_PROFILE_NAME
                ),
            },
            "tcp_source": "DMTAC_M_tool_profile",
            "orientation_policy": "track_dataset_orientation",
            "dmtac_tcp_height_error": height_error,
            "world_from_desk": self.world_from_desk.tolist(),
            "rig": self.rig.snapshot(),
        }

    def close(self) -> None:
        self.rig.close()
        close = getattr(self.scene, "close", None)
        if callable(close):
            close()


__all__ = [
    "DEFAULT_SIMULATION_CONTRACT_PATH",
    "DEFAULT_STATION_POSE_CONTRACT_PATH",
    "DEFAULT_TOOL_PROFILE_BY_SIDE",
    "DeskTcpPositionControl",
    "DeskTcpPositionControlConfig",
    "position_15_60_120_timing",
]
