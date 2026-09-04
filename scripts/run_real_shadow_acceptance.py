"""Capture RM65/DMTac state and run one simulation-first grasp acceptance.

The default mode opens the assigned RM65 arms and DMTac grippers in
observation-only mode, captures one causal state cut, initializes MuJoCo from
that cut, and executes the registered joint trajectory only in MuJoCo.  The
explicit ``--execute-real`` mode closes that passive context after a completed
MuJoCo receipt, starts the active RM65 lower owner, and replays the exact
registered qpos trajectory through the existing checked dispatch path.  DMTac
is deliberately measured/passive in both modes: the replay keeps the measured
opening fixed and never sends a gripper command.

The assignment files are explicit command-line inputs so this probe cannot
silently fall back to a stale checkout-local hardware assignment.  The normal
remote command points them at the deployment checkout while all source code is
read from the isolated Skill worktree.

The optional MuJoCo window and post-simulation terminal confirmation belong to
this hardware acceptance probe only.  They are not part of the deployed Skill
runtime protocol.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SIDES = ("left", "right")
OBSTACLE_FRAME_ID = "rm65_scene_world"
OBSTACLE_FRAME_SPEC_HASH = "mujoco-desk-collision-world-v1"
REAL_REPLAY_AUTHORITY_TTL_SEC = 30.0
REAL_EXECUTION_CONFIRMATION_TEXT = "EXECUTE"


class OperatorConfirmationDeclined(RuntimeError):
    """Typed no-motion outcome for the acceptance-only terminal gate."""

    reason_code = "operator_confirmation_declined"


@dataclasses.dataclass
class _PostSimulationConfirmation:
    reference: str | None = None
    declined: bool = False


def _jsonable(value: Any) -> Any:
    """Convert typed receipts and numpy values without changing semantics."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.name != "semantic_content_hash"
        } | (
            {"semantic_content_hash": value.semantic_content_hash}
            if hasattr(value, "semantic_content_hash")
            else {}
        )
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _load_json(path: str | Path) -> Mapping[str, Any] | list[Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, (Mapping, list)):
        raise TypeError(f"JSON root must be an object or array: {path}")
    return value


def _pose_from_json(value: object) -> np.ndarray:
    if isinstance(value, Mapping):
        for key in (
            "target_pose_base_from_dmtac_tcp",
            "pose",
            "matrix",
        ):
            if key in value:
                value = value[key]
                break
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("target pose JSON must contain a finite 4x4 matrix")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError("target pose JSON has an invalid homogeneous row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("target pose JSON rotation must be proper and orthonormal")
    return pose.copy()


def _pose_from_cli(args: argparse.Namespace) -> tuple[np.ndarray | None, str]:
    from sim.sim_real_tests.mujoco.arbitrary_grasp_pose_motion import (
        pose_from_xyz_quat_xyzw,
        pose_from_xyz_rpy_deg,
    )

    offset_rpy = np.asarray(args.target_offset_rpy_deg, dtype=float).reshape(-1)
    if offset_rpy.shape != (3,) or not np.all(np.isfinite(offset_rpy)):
        raise ValueError("--target-offset-rpy-deg must be a finite 3-vector")
    if args.target_offset_xyz_m is None and float(np.linalg.norm(offset_rpy)) > 1e-9:
        raise ValueError("--target-offset-rpy-deg requires --target-offset-xyz-m")

    modes = int(args.target_pose_json is not None) + int(args.target_xyz_m is not None)
    modes += int(bool(args.target_from_current))
    modes += int(args.target_offset_xyz_m is not None)
    if modes != 1:
        raise ValueError(
            "choose exactly one of --target-pose-json, --target-xyz-m, "
            "--target-offset-xyz-m, or --target-from-current"
        )
    if args.target_pose_json is not None:
        if args.target_quat_xyzw is not None or args.target_rpy_deg is not None:
            raise ValueError("orientation options cannot accompany --target-pose-json")
        return _pose_from_json(_load_json(args.target_pose_json)), "json_matrix"
    if args.target_from_current:
        if args.target_quat_xyzw is not None or args.target_rpy_deg is not None:
            raise ValueError(
                "orientation options cannot accompany --target-from-current"
            )
        return None, "measured_current_tcp"
    if args.target_offset_xyz_m is not None:
        if args.target_quat_xyzw is not None or args.target_rpy_deg is not None:
            raise ValueError(
                "absolute orientation options cannot accompany --target-offset-xyz-m"
            )
        offset = np.asarray(args.target_offset_xyz_m, dtype=float).reshape(-1)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("--target-offset-xyz-m must be a finite 3-vector")
        if (
            float(np.linalg.norm(offset)) <= 1e-9
            and float(np.linalg.norm(offset_rpy)) <= 1e-9
        ):
            raise ValueError(
                "relative target must request non-zero translation or rotation"
            )
        return None, "measured_current_tcp_plus_base_translation_and_tool_rotation"
    if args.target_quat_xyzw is not None:
        return (
            pose_from_xyz_quat_xyzw(args.target_xyz_m, args.target_quat_xyzw),
            "quaternion_xyzw",
        )
    if args.target_rpy_deg is not None:
        return (
            pose_from_xyz_rpy_deg(args.target_xyz_m, args.target_rpy_deg),
            "rpy_xyz_degrees",
        )
    raise ValueError(
        "--target-xyz-m requires a target orientation via --target-quat-xyzw "
        "or --target-rpy-deg"
    )


def _scene_from_cli(args: argparse.Namespace):
    from robot_motion.scene_obstacles import (
        AxisAlignedBoxObstacle,
        ObstacleSceneSnapshot,
    )

    if args.obstacle_json is not None:
        value = _load_json(args.obstacle_json)
        if not isinstance(value, Mapping):
            raise ValueError("--obstacle-json must contain an object")
        scene = ObstacleSceneSnapshot.from_mapping(value)
        if (
            scene.reference_frame_id != OBSTACLE_FRAME_ID
            or scene.reference_frame_spec_hash != OBSTACLE_FRAME_SPEC_HASH
        ):
            raise ValueError(
                "obstacle scene must use the RM65 MuJoCo planning frame "
                f"{OBSTACLE_FRAME_ID!r}/{OBSTACLE_FRAME_SPEC_HASH!r}"
            )
        return scene
    obstacles = tuple(
        AxisAlignedBoxObstacle(
            obstacle_id=f"cube-{index}",
            center_xyz_m=tuple(float(item) for item in cube[:3]),
            size_xyz_m=(float(cube[3]),) * 3,
        )
        for index, cube in enumerate(args.obstacle_cube_world_m or ())
    )
    return ObstacleSceneSnapshot(
        reference_frame_id=OBSTACLE_FRAME_ID,
        reference_frame_spec_hash=OBSTACLE_FRAME_SPEC_HASH,
        revision=int(args.obstacle_revision),
        obstacles=obstacles,
        minimum_clearance_m=float(args.obstacle_clearance_m),
    )


def _simulation_config(profile: Any, snapshot: Any):
    from sim.mujoco.control.desk_tcp_position import (
        DEFAULT_SIMULATION_CONTRACT_PATH,
        DEFAULT_STATION_POSE_CONTRACT_PATH,
        DeskTcpPositionControlConfig,
    )

    ik = profile.position_profile.ik_execution
    command = profile.position_profile.command_execution
    return DeskTcpPositionControlConfig(
        simulation_contract_path=DEFAULT_SIMULATION_CONTRACT_PATH,
        station_pose_contract_path=DEFAULT_STATION_POSE_CONTRACT_PATH,
        left_tool_profile=profile.tool_profile_path_by_side["left"],
        right_tool_profile=profile.tool_profile_path_by_side["right"],
        position_control_profile_name=profile.position_profile.name,
        solve_max_iterations=int(ik.solve_max_iterations),
        solve_min_iterations=int(ik.solve_min_iterations),
        solve_adaptive_iterations_enabled=bool(ik.adaptive_iterations),
        solve_iteration_reserve_ms=float(ik.iteration_reserve_ms),
        damping=float(ik.damping),
        orientation_weight=float(ik.orientation_weight),
        joint_velocity_limit_profile=ik.joint_velocity_limit_profile,
        max_command_step_deg=float(command.max_command_step_deg),
        target_velocity_filter_tau_sec=float(command.target_velocity_filter_tau_sec),
        follow_qpos_alpha=float(command.follow_qpos_alpha),
        left_home_qpos=tuple(float(item) for item in snapshot.qpos_by_side["left"]),
        right_home_qpos=tuple(float(item) for item in snapshot.qpos_by_side["right"]),
    )


def _target_from_current(
    snapshot: Any,
    profile: Any,
    *,
    side: str,
    translation_offset_m: Sequence[float] | None = None,
    tool_rotation_offset_rpy_deg: Sequence[float] | None = None,
):
    from sim.sim_real_tests.mujoco.arbitrary_grasp_pose_motion import (
        GraspMotionTarget,
        build_shadow_initialized_control,
    )

    control = build_shadow_initialized_control(
        snapshot,
        config=_simulation_config(profile, snapshot),
    )
    try:
        qpos = {side: control.scene.get_arm_qpos(side).copy() for side in SIDES}
        world = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=qpos["left"],
            right_qpos=qpos["right"],
        )
        base_from_dmtac = control.world_to_base_by_side[side] @ world[side]
        known_qpos = qpos[side]
    finally:
        control.close()
    if translation_offset_m is not None:
        from scipy.spatial.transform import Rotation

        offset = np.asarray(translation_offset_m, dtype=float).reshape(3)
        base_from_dmtac = base_from_dmtac.copy()
        base_from_dmtac[:3, 3] += offset
        rotation_offset = np.asarray(
            (
                (0.0, 0.0, 0.0)
                if tool_rotation_offset_rpy_deg is None
                else tool_rotation_offset_rpy_deg
            ),
            dtype=float,
        ).reshape(3)
        base_from_dmtac[:3, :3] = (
            base_from_dmtac[:3, :3]
            @ Rotation.from_euler("xyz", rotation_offset, degrees=True).as_matrix()
        )
        known_qpos = None
    return GraspMotionTarget(
        side=side,
        target_pose_base_from_dmtac_tcp=base_from_dmtac,
        source=(
            "measured_current_dmtac_tcp"
            if translation_offset_m is None
            else "measured_current_dmtac_tcp_plus_base_translation_and_tool_rotation"
        ),
        known_qpos_rad=known_qpos,
        metadata={
            "target_frame": "selected_rm65_arm_base",
            "translation_offset_m": (
                None
                if translation_offset_m is None
                else tuple(float(value) for value in translation_offset_m)
            ),
            "tool_rotation_offset_rpy_deg": (
                None
                if tool_rotation_offset_rpy_deg is None
                else tuple(float(value) for value in tool_rotation_offset_rpy_deg)
            ),
        },
    )


def _build_shadow_control(profile: Any, snapshot: Any, scene: Any):
    """Build one configured MuJoCo control and bind the requested cube scene."""

    from sim.sim_real_tests.mujoco.arbitrary_grasp_pose_motion import (
        build_obstacle_collision_oracle,
        build_shadow_initialized_control,
    )

    control = build_shadow_initialized_control(
        snapshot,
        config=_simulation_config(profile, snapshot),
    )
    try:
        control.shadow_obstacle_oracle = build_obstacle_collision_oracle(
            control,
            scene,
        )
        return control
    except BaseException:
        control.close()
        raise


def _build_identity(profile: Any):
    from dataset_contracts.canonical_encoding import canonical_content_hash
    from dataset_contracts.pose_frames import load_pose_frame_contract
    from robot_motion.shadow_execution import ShadowExecutionIdentity
    from sim.contracts import SimulationContract
    from sim.mujoco.actuation import (
        cad_bimanual_dmtac_gripper_profile,
        rm65_bimanual_udp_feedback_profile,
    )
    from sim.mujoco.control.desk_tcp_position import (
        DEFAULT_SIMULATION_CONTRACT_PATH,
        DEFAULT_STATION_POSE_CONTRACT_PATH,
    )

    station = load_pose_frame_contract(DEFAULT_STATION_POSE_CONTRACT_PATH)
    simulation = SimulationContract.load(DEFAULT_SIMULATION_CONTRACT_PATH)
    arm_profile = rm65_bimanual_udp_feedback_profile()
    gripper_profile = cad_bimanual_dmtac_gripper_profile()
    tool_payloads = {
        side: {
            key: value
            for key, value in profile.tool_profile_by_side[side].items()
            if not str(key).startswith("_")
        }
        for side in SIDES
    }
    desk_from_base = {
        side: station.source.effective_raw_to_canonical_desk_transform(side).tolist()
        for side in SIDES
    }
    return ShadowExecutionIdentity(
        station_revision=station.source_id,
        reference_frame_id=station.canonical_absolute_frame,
        reference_frame_spec_hash=station.contract_hash,
        frame_graph_hash=canonical_content_hash(
            {
                "station_contract_hash": station.contract_hash,
                "desk_from_base": desk_from_base,
            },
            domain="agenticrobot.real_shadow_acceptance.frame_graph.v1",
        ),
        tool_profile_hash_by_side={
            side: canonical_content_hash(
                tool_payloads[side],
                domain="agenticrobot.real_shadow_acceptance.tool_profile.v1",
            )
            for side in SIDES
        },
        robot_model_hash=canonical_content_hash(
            {
                "scene_profile": simulation.scene.profile_name,
                "urdf_path": simulation.scene.urdf_path,
            },
            domain="agenticrobot.real_shadow_acceptance.robot_model.v1",
        ),
        collision_model_hash=canonical_content_hash(
            {"profile": profile.collision_scene_profile},
            domain="agenticrobot.real_shadow_acceptance.collision_model.v1",
        ),
        joint_layout_hash=canonical_content_hash(
            {"joint_names_by_side": arm_profile.joint_names_by_side},
            domain="agenticrobot.real_shadow_acceptance.joint_layout.v1",
        ),
        gripper_model_hash=canonical_content_hash(
            gripper_profile.to_metadata(),
            domain="agenticrobot.real_shadow_acceptance.gripper_model.v1",
        ),
    )


def _build_binding(profile: Any, trajectory: Any):
    from dataset_contracts.pose_frames import load_pose_frame_contract
    from sim.contracts import SimulationContract
    from sim.mujoco.actuation import (
        cad_bimanual_dmtac_gripper_profile,
        rm65_bimanual_udp_feedback_profile,
    )
    from sim.mujoco.control.desk_tcp_position import (
        DEFAULT_SIMULATION_CONTRACT_PATH,
        DEFAULT_STATION_POSE_CONTRACT_PATH,
    )
    from sim.sim_real_tests.mujoco.exact_shadow_trajectory import (
        MujocoShadowCompatibilityExpectation,
    )

    station = load_pose_frame_contract(DEFAULT_STATION_POSE_CONTRACT_PATH)
    simulation = SimulationContract.load(DEFAULT_SIMULATION_CONTRACT_PATH)
    arm_profile = rm65_bimanual_udp_feedback_profile()
    gripper_profile = cad_bimanual_dmtac_gripper_profile()
    tool_ids = {}
    for side in SIDES:
        tool = profile.tool_profile_by_side[side]
        name = str(tool.get("name", tool.get("tool_name", ""))).strip()
        if not name:
            raise ValueError(f"{side} tool profile has no name")
        tool_ids[side] = f"{name}:{side}"
    return MujocoShadowCompatibilityExpectation(
        station_revision=station.source_id,
        reference_frame_id=station.canonical_absolute_frame,
        desk_from_arm_base_by_side={
            side: tuple(
                station.source.effective_raw_to_canonical_desk_transform(side).reshape(
                    -1
                )
            )
            for side in SIDES
        },
        joint_names_by_side={
            side: arm_profile.joint_names_by_side[side] for side in SIDES
        },
        tool_profile_id_by_side=tool_ids,
        tool_tcp_pose_flange_m_rad_by_side={
            side: profile.tool_tcp_pose_flange_m_rad_by_side[side] for side in SIDES
        },
        gripper_profile_name=gripper_profile.name,
        gripper_side_by_name=gripper_profile.side_by_gripper,
        scene_revision=trajectory.scene_hash,
        simulation_scene_profile_name=simulation.scene.profile_name,
        collision_profile_name=profile.collision_scene_profile,
        evidence_refs=(
            f"real-owner-profile:{profile.semantic_content_hash}",
            f"real-owner-station:{station.contract_hash}",
        ),
    )


def _passive_arm_backend_factory(config: Any):
    from teleop_runtime.backends.arms import RM65ArmBackend

    return RM65ArmBackend(
        replace(
            config,
            observation_only=True,
            background_streaming_enabled=False,
            connect_hardware=True,
        )
    )


def _real_arm_backend_factory(config: Any):
    """Construct the active RM65 backend only after simulation acceptance."""

    from teleop_runtime.backends.arms import RM65ArmBackend

    return RM65ArmBackend(
        replace(
            config,
            observation_only=False,
            background_streaming_enabled=True,
            connect_hardware=True,
        )
    )


def _passive_dmtac_backends(assignment: Mapping[str, Any]):
    from teleop_runtime.backends.grippers import (
        DMTacGripperBackend,
    )
    from teleop_runtime.configs.dmtac_hardware import (
        dmtac_backend_configs_from_assignment,
    )

    configs = dmtac_backend_configs_from_assignment(
        assignment,
        active_sides=SIDES,
        prefer_existing_by_path_port=True,
    )
    if set(configs) != set(SIDES):
        raise ValueError("DMTac assignment must bind both left and right grippers")
    return {
        side: DMTacGripperBackend(
            replace(
                configs[side],
                observation_only=True,
                send_commands=False,
                clear_fault_on_start=False,
                open_on_start=False,
                connect_tactile=False,
                enable_telemetry=False,
                tactile_reset_on_start=False,
            )
        )
        for side in SIDES
    }


def _git_metadata() -> dict[str, str | None]:
    result: dict[str, str | None] = {"hostname": socket.gethostname()}
    for key, command in (
        ("branch", ("git", "branch", "--show-current")),
        ("commit", ("git", "rev-parse", "HEAD")),
        ("tree", ("git", "rev-parse", "HEAD^{tree}")),
    ):
        try:
            result[key] = subprocess.check_output(
                command,
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            result[key] = None
    return result


class _SnapshotContext:
    def __init__(self, *, profile: Any, dmtac_assignment: str):
        from teleop_runtime.configs.dmtac_hardware import load_dmtac_hardware_assignment
        from teleop_runtime.skill_deploy import (
            assemble_exclusive_rm65_skill_lower,
        )

        self.profile = profile
        self.dmtac_assignment = str(dmtac_assignment)
        self._load_assignment = load_dmtac_hardware_assignment
        self._assemble_lower = assemble_exclusive_rm65_skill_lower
        self.identity = _build_identity(profile)
        self._started = False
        self._active_hardware = False
        self._build_context(backend_factory=_passive_arm_backend_factory)

    def _build_context(self, *, backend_factory: Any) -> None:
        assignment = self._load_assignment(
            self.dmtac_assignment,
            validate_current_usb=True,
            allow_missing=False,
            assume_yes_for_extra=False,
        )
        self.dmtac_backends = _passive_dmtac_backends(assignment)
        self.lower = self._assemble_lower(
            profile=self.profile,
            backend_factory=backend_factory,
        )

    def start(self) -> None:
        for backend in self.dmtac_backends.values():
            backend.start()
            if not backend.is_ready():
                raise RuntimeError(f"{backend.name} DMTac backend is not ready")
        self.lower.ports.start()
        self.lower.startup_admission()
        self._started = True

    @property
    def active_hardware(self) -> bool:
        return self._active_hardware

    def promote_to_active_hardware(self) -> None:
        """Replace passive readers with the active lower owner after Sim passes."""

        if self._active_hardware:
            return
        cleanup_errors = self.close()
        if cleanup_errors:
            raise RuntimeError(
                "cannot promote to active RM65 lower after passive cleanup: "
                + "; ".join(cleanup_errors)
            )
        self._build_context(backend_factory=_real_arm_backend_factory)
        try:
            self.start()
        except Exception:
            # ``start`` performs its own rollback, while this close releases
            # any DMTac reader that did start before the lower failed.
            self.close()
            raise
        self._active_hardware = True

    def snapshot(self):
        from teleop_runtime.skill_deploy.registered_trajectory import (
            capture_authoritative_shadow_state_from_backends,
        )

        if not self._started:
            raise RuntimeError("live observation context is not started")
        arm_cut = self.lower.ports.observe()
        return capture_authoritative_shadow_state_from_backends(
            arm_observation_cut=arm_cut,
            gripper_backends_by_side=self.dmtac_backends,
            identity=self.identity,
        )

    def close(self) -> list[str]:
        errors: list[str] = []
        try:
            if (
                self.lower.ports.resources_started
                or not self.lower.ports.resources_closed
            ):
                self.lower.ports.close()
        except Exception as exc:  # noqa: BLE001 - report all cleanup failures.
            errors.append(f"rm65:{type(exc).__name__}:{exc}")
        for side in reversed(SIDES):
            try:
                self.dmtac_backends[side].close()
            except Exception as exc:  # noqa: BLE001 - report the other side too.
                errors.append(f"dmtac-{side}:{type(exc).__name__}:{exc}")
        self._started = False
        return errors


def _target_tool_tcp_world(
    profile: Any,
    target: Any,
) -> np.ndarray:
    """Convert the requested RM65-base TCP pose into the solver world frame."""

    side = str(target.side).strip().lower()
    world_from_base = np.asarray(
        profile.scene_world_from_arm_base_by_side[side],
        dtype=float,
    ).reshape(4, 4)
    pose = np.asarray(target.target_pose_base_from_dmtac_tcp, dtype=float).reshape(4, 4)
    result = world_from_base @ pose
    result.setflags(write=False)
    return result


def _build_real_ports_factory(
    *,
    context: _SnapshotContext,
    profile: Any,
    target: Any,
):
    """Promote the lower owner lazily and bind exact DMTac/RM65 replay ports."""

    def factory(trajectory: Any):
        from teleop_runtime.skill_deploy import (
            RM65_SKILL_SCENE_WORLD_FRAME_ID_V1,
            DMTacRM65RegisteredTrajectoryPorts,
            DMTacShadowReplayPortProfile,
        )

        if set(trajectory.moving_sides) != {str(target.side).strip().lower()}:
            raise ValueError("real replay target side differs from trajectory.")
        context.promote_to_active_hardware()
        world_from_base_by_side = {
            side: np.asarray(
                profile.scene_world_from_arm_base_by_side[side], dtype=float
            ).reshape(4, 4)
            for side in SIDES
        }

        def project_tool_tcp_world(side: str, tool_tcp_world: np.ndarray) -> np.ndarray:
            world_from_base = world_from_base_by_side[side]
            base_from_tool = np.linalg.inv(world_from_base) @ tool_tcp_world
            projected_base_from_tool = (
                context.lower.target_safety.envelope_filter.project_arm_base_tool_tcp(
                    side,
                    base_from_tool,
                )
            )
            return world_from_base @ projected_base_from_tool

        return DMTacRM65RegisteredTrajectoryPorts(
            trajectory=trajectory,
            identity=context.identity,
            rm65_ports=context.lower.ports,
            dmtac_backends_by_side=context.dmtac_backends,
            tool_tcp_world_fk=context.lower.target_builder,
            target_tool_tcp_world_by_side={
                str(target.side).strip().lower(): _target_tool_tcp_world(
                    profile,
                    target,
                )
            },
            target_reference_frame_id=RM65_SKILL_SCENE_WORLD_FRAME_ID_V1,
            tool_tcp_world_envelope_projector=project_tool_tcp_world,
            profile=DMTacShadowReplayPortProfile(
                require_trajectory_tcp_envelope=True,
            ),
        )

    return factory


def _build_real_authority_factory(
    *,
    confirmation: _PostSimulationConfirmation,
    ttl_sec: float = REAL_REPLAY_AUTHORITY_TTL_SEC,
    clock=time.time,
):
    """Bind the accepted trajectory after the acceptance-only terminal gate."""

    if not isinstance(confirmation, _PostSimulationConfirmation):
        raise TypeError("confirmation must be _PostSimulationConfirmation.")
    ttl = float(ttl_sec)
    if not np.isfinite(ttl) or ttl <= 0.0:
        raise ValueError("real authority TTL must be finite and positive.")
    if not callable(clock):
        raise TypeError("authority clock must be callable.")

    def factory(trajectory: Any, simulation: Any):
        from teleop_runtime.skill_deploy import (
            RegisteredTrajectoryExecutionAuthority,
            SingleUseRegisteredTrajectoryAuthorityConsumer,
        )

        if confirmation.declined or not str(confirmation.reference or "").strip():
            raise OperatorConfirmationDeclined(
                "operator did not confirm real replay after MuJoCo completion"
            )
        issued_at = float(clock())
        run_id = (
            "real-grasp-skill:"
            + trajectory.semantic_content_hash[:16]
            + f":{int(issued_at * 1_000_000)}"
        )
        authority = RegisteredTrajectoryExecutionAuthority(
            authority_id="authority:" + run_id,
            operator_confirmation_ref=str(confirmation.reference),
            trajectory_hash=trajectory.semantic_content_hash,
            simulation_receipt_hash=simulation.semantic_content_hash,
            run_id=run_id,
            owner_epoch=1,
            issued_at=issued_at,
            expires_at=issued_at + ttl,
        )
        consumer = SingleUseRegisteredTrajectoryAuthorityConsumer(
            authority=authority,
            clock=clock,
        )
        return authority, consumer.consume

    return factory


def _target_pose_world_by_side(control: Any, target: Any) -> dict[str, np.ndarray]:
    """Build display-only target poses without modifying the registered path."""

    qpos = control.rig.measured_qpos_by_side()
    poses = {
        side: np.asarray(value, dtype=float).reshape(4, 4).copy()
        for side, value in control.retargeter.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=qpos["left"],
            right_qpos=qpos["right"],
        ).items()
    }
    side = str(target.side).strip().lower()
    world_from_base = np.linalg.inv(
        np.asarray(control.world_to_base_by_side[side], dtype=float).reshape(4, 4)
    )
    poses[side] = world_from_base @ np.asarray(
        target.target_pose_base_from_dmtac_tcp,
        dtype=float,
    ).reshape(4, 4)
    return poses


def _confirm_real_replay_after_simulation(
    *,
    trajectory: Any,
    simulation: Any,
    viewer_handle: Any,
    confirmation: _PostSimulationConfirmation,
    input_fn=None,
) -> bool:
    """Ask once, in the terminal, while the accepted terminal pose is visible."""

    if not bool(viewer_handle.is_running()):
        confirmation.declined = True
        return False
    if input_fn is None:
        input_fn = input
    print(
        "\n[REAL ACCEPTANCE] MuJoCo 已从真机状态完成整条轨迹并在终点保持。\n"
        "请现场核对机械臂路径、目标和障碍物；此时尚未打开真机运动端口。",
        flush=True,
    )
    try:
        answer = input_fn(
            f"确认真机环境可执行同一条轨迹后输入 {REAL_EXECUTION_CONFIRMATION_TEXT}；"
            "其他输入将终止且不运动真机: "
        )
    except (EOFError, KeyboardInterrupt):
        answer = ""
    accepted = str(answer).strip() == REAL_EXECUTION_CONFIRMATION_TEXT
    confirmation.declined = not accepted
    if accepted:
        confirmation.reference = (
            "interactive-post-mujoco:"
            + trajectory.semantic_content_hash[:16]
            + ":"
            + simulation.semantic_content_hash[:16]
        )
        print(
            "[REAL ACCEPTANCE] 已确认；MuJoCo 窗口关闭后才会连接真机运动端口。",
            flush=True,
        )
    else:
        print("[REAL ACCEPTANCE] 未确认；真机不会运动。", flush=True)
    return accepted


def _execute_registered_shadow_with_viewer(
    *,
    control: Any,
    trajectory: Any,
    model_binding: Any,
    target: Any,
    obstacle_scene: Any,
    confirmation: _PostSimulationConfirmation,
    require_confirmation: bool,
    playback_hz: float,
    terminal_display_sec: float,
    camera_distance: float,
    camera_azimuth_deg: float,
    camera_elevation_deg: float,
    input_fn=None,
):
    """Run the exact simulator in wall-clock time inside a passive MuJoCo GUI."""

    from robot_motion.shadow_execution import ShadowSimulationStatus
    from sim.mujoco.replay.tcp_target_visualization import TcpTargetVisualizer
    from sim.mujoco.viewer import passive_viewer_handle
    from sim.sim_real_tests.mujoco.exact_shadow_trajectory import (
        execute_registered_shadow_trajectory,
    )
    from sim.sim_real_tests.mujoco.view_arbitrary_grasp_pose_motion import (
        _ObstacleSceneVisualizer,
    )

    if not np.isfinite(float(playback_hz)) or float(playback_hz) <= 0.0:
        raise ValueError("playback_hz must be finite and positive")
    target_world = _target_pose_world_by_side(control, target)
    target_visualizer = TcpTargetVisualizer()
    obstacle_visualizer = _ObstacleSceneVisualizer(obstacle_scene)
    frame_period = 1.0 / float(playback_hz)

    with passive_viewer_handle(
        control.scene,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer_handle:
        moving_target = target_world[str(target.side).strip().lower()]
        viewer_handle.cam.lookat[:] = moving_target[:3, 3]
        viewer_handle.cam.distance = float(camera_distance)
        viewer_handle.cam.azimuth = float(camera_azimuth_deg)
        viewer_handle.cam.elevation = float(camera_elevation_deg)

        def sync_overlay() -> None:
            if not bool(viewer_handle.is_running()):
                raise RuntimeError("mujoco_viewer_closed_before_simulation_completed")
            target_visualizer.update(viewer_handle, target_world)
            obstacle_visualizer.update(viewer_handle)
            viewer_handle.sync()

        sync_overlay()
        original_step = control.step_physics
        simulation_started_at = float(control.scene.data.time)
        wall_started_at = time.monotonic()
        next_frame_at = simulation_started_at

        def paced_step():
            nonlocal next_frame_at
            result = original_step()
            simulation_now = float(control.scene.data.time)
            if simulation_now + 1e-12 < next_frame_at:
                return result
            target_wall = wall_started_at + (simulation_now - simulation_started_at)
            while True:
                remaining = target_wall - time.monotonic()
                if remaining <= 0.0:
                    break
                sync_overlay()
                time.sleep(min(0.01, remaining))
            sync_overlay()
            next_frame_at = simulation_now + frame_period
            return result

        control.step_physics = paced_step
        try:
            simulation = execute_registered_shadow_trajectory(
                control=control,
                trajectory=trajectory,
                model_binding=model_binding,
            )
        finally:
            delattr(control, "step_physics")
        sync_overlay()
        if simulation.status is ShadowSimulationStatus.COMPLETED:
            if require_confirmation:
                _confirm_real_replay_after_simulation(
                    trajectory=trajectory,
                    simulation=simulation,
                    viewer_handle=viewer_handle,
                    confirmation=confirmation,
                    input_fn=input_fn,
                )
            else:
                deadline = time.monotonic() + max(0.0, float(terminal_display_sec))
                while time.monotonic() < deadline:
                    sync_overlay()
                    time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
        return simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-ip-assignment", required=True)
    parser.add_argument("--dmtac-assignment", required=True)
    parser.add_argument("--side", choices=SIDES, default="left")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-pose-json")
    target.add_argument("--target-from-current", action="store_true")
    target.add_argument(
        "--target-offset-xyz-m",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        help=(
            "Translate the measured DMTac TCP by DX/DY/DZ in the selected "
            "RM65 base frame; its orientation is preserved unless "
            "--target-offset-rpy-deg is supplied."
        ),
    )
    parser.add_argument(
        "--target-offset-rpy-deg",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DROLL", "DPITCH", "DYAW"),
        help=(
            "Apply a tool-local XYZ Euler rotation after the measured TCP "
            "orientation; valid only with --target-offset-xyz-m."
        ),
    )
    target.add_argument("--target-xyz-m", type=float, nargs=3, metavar=("X", "Y", "Z"))
    orientation = parser.add_mutually_exclusive_group()
    orientation.add_argument(
        "--target-quat-xyzw", type=float, nargs=4, metavar=("QX", "QY", "QZ", "QW")
    )
    orientation.add_argument(
        "--target-rpy-deg", type=float, nargs=3, metavar=("ROLL", "PITCH", "YAW")
    )
    parser.add_argument(
        "--left-tool-profile",
        default=str(
            REPO_ROOT
            / "hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json"
        ),
    )
    parser.add_argument(
        "--right-tool-profile",
        default=str(
            REPO_ROOT
            / "hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json"
        ),
    )
    parser.add_argument(
        "--envelope-profile",
        help="Strict TCP envelope JSON; defaults to the selected side's tight profile.",
    )
    parser.add_argument("--obstacle-json")
    parser.add_argument(
        "--obstacle-cube-world-m",
        type=float,
        nargs=4,
        action="append",
        metavar=("X", "Y", "Z", "SIDE"),
    )
    parser.add_argument("--obstacle-revision", type=int, default=0)
    parser.add_argument("--obstacle-clearance-m", type=float, default=0.02)
    parser.add_argument(
        "--require-obstacle-detour",
        action="store_true",
        help=(
            "Fail before simulation/real motion unless configured obstacles "
            "block the direct joint path and the planner finds a detour."
        ),
    )
    parser.add_argument("--linear-speed-m-s", type=float, default=0.12)
    parser.add_argument("--angular-speed-deg-s", type=float, default=45.0)
    parser.add_argument("--minimum-duration-sec", type=float, default=2.0)
    parser.add_argument("--output-json")
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="Explicitly select the simulation-only path (the safe default).",
    )
    parser.add_argument(
        "--execute-real",
        action="store_true",
        help=(
            "Open MuJoCo, replay and hold the exact trajectory, then ask for "
            "terminal confirmation before connecting real RM65 motion ports."
        ),
    )
    parser.add_argument(
        "--mujoco-viewer",
        action="store_true",
        help=(
            "Open a realtime MuJoCo window for simulation-only acceptance. "
            "--execute-real always enables this window."
        ),
    )
    parser.add_argument("--viewer-playback-hz", type=float, default=60.0)
    parser.add_argument("--viewer-terminal-display-sec", type=float, default=2.0)
    parser.add_argument("--viewer-camera-distance", type=float, default=1.55)
    parser.add_argument("--viewer-camera-azimuth-deg", type=float, default=135.0)
    parser.add_argument("--viewer-camera-elevation-deg", type=float, default=-24.0)
    return parser


def _write_output(payload: Mapping[str, Any], path: str | None) -> None:
    text = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2)
    print(text)
    if path is not None:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.no_motion and args.execute_real:
        print(
            "ERROR: choose exactly one of --no-motion or --execute-real",
            file=sys.stderr,
        )
        return 2
    if not args.no_motion and not args.execute_real:
        print(
            "ERROR: pass --no-motion for simulation-only or --execute-real for "
            "explicit real replay",
            file=sys.stderr,
        )
        return 2
    if args.execute_real and args.target_from_current:
        print(
            "ERROR: --target-from-current is a zero-motion diagnostic and cannot "
            "be used with --execute-real; provide a real goal",
            file=sys.stderr,
        )
        return 2
    for name in (
        "viewer_playback_hz",
        "viewer_camera_distance",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            print(
                f"ERROR: --{name.replace('_', '-')} must be finite and positive",
                file=sys.stderr,
            )
            return 2
    if (
        not np.isfinite(float(args.viewer_terminal_display_sec))
        or float(args.viewer_terminal_display_sec) < 0.0
    ):
        print(
            "ERROR: --viewer-terminal-display-sec must be finite and non-negative",
            file=sys.stderr,
        )
        return 2
    for name in (
        "viewer_camera_azimuth_deg",
        "viewer_camera_elevation_deg",
    ):
        if not np.isfinite(float(getattr(args, name))):
            print(
                f"ERROR: --{name.replace('_', '-')} must be finite",
                file=sys.stderr,
            )
            return 2
    if args.obstacle_json is not None and args.obstacle_cube_world_m:
        print(
            "ERROR: choose --obstacle-json or --obstacle-cube-world-m", file=sys.stderr
        )
        return 2
    if args.require_obstacle_detour and not (
        args.obstacle_json is not None or args.obstacle_cube_world_m
    ):
        print(
            "ERROR: --require-obstacle-detour needs at least one obstacle",
            file=sys.stderr,
        )
        return 2
    try:
        pose, orientation_format = _pose_from_cli(args)
        scene = _scene_from_cli(args)
        from teleop_runtime.skill_deploy.rm65_profile import (
            ExclusiveRM65SkillDeploymentProfile,
            resolve_exclusive_rm65_skill_deployment_profile,
        )

        profile = resolve_exclusive_rm65_skill_deployment_profile(
            ExclusiveRM65SkillDeploymentProfile(
                arm_ip_assignment_path=Path(args.arm_ip_assignment),
                tool_profile_path_by_side={
                    "left": Path(args.left_tool_profile),
                    "right": Path(args.right_tool_profile),
                },
                envelope_profile_path=Path(
                    args.envelope_profile
                    or REPO_ROOT
                    / "teleop_runtime/configs/skill_deploy/"
                    / f"agent_primitive_{args.side}_relative_tight_v1.json"
                ),
            )
        )
        context = _SnapshotContext(
            profile=profile,
            dmtac_assignment=args.dmtac_assignment,
        )
    except Exception as exc:  # noqa: BLE001 - typed operator failure below.
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        details = getattr(exc, "details", None)
        if details is not None:
            error["details"] = _jsonable(details)
        _write_output(
            {
                "schema": "agenticrobot.real_shadow_acceptance.v1",
                "mode": "real_replay" if args.execute_real else "observation_only",
                "status": "setup_failed",
                "git": _git_metadata(),
                "error": error,
            },
            args.output_json,
        )
        return 1

    try:
        context.start()
        snapshot = context.snapshot()
        if pose is None:
            target = _target_from_current(
                snapshot,
                profile,
                side=args.side,
                translation_offset_m=args.target_offset_xyz_m,
                tool_rotation_offset_rpy_deg=args.target_offset_rpy_deg,
            )
        else:
            from sim.sim_real_tests.mujoco.arbitrary_grasp_pose_motion import (
                GraspMotionTarget,
            )

            target = GraspMotionTarget(
                side=args.side,
                target_pose_base_from_dmtac_tcp=pose,
                source="real_shadow_cli_target",
                metadata={
                    "orientation_format": orientation_format,
                    "reference_frame_id": f"{args.side}_rm65_base",
                },
            )
        from control_station.real_shadow_acceptance import run_real_shadow_acceptance
        from sim.sim_real_tests.mujoco.exact_shadow_trajectory import (
            RealTcpObstacleTrajectoryRequest,
            build_registered_tcp_obstacle_trajectory,
            execute_registered_shadow_trajectory,
        )

        request = RealTcpObstacleTrajectoryRequest(
            side=args.side,
            target_pose_base_from_dmtac_tcp=target.target_pose_base_from_dmtac_tcp,
            obstacle_scene=scene,
            source=target.source,
            known_qpos_rad=target.known_qpos_rad,
            linear_speed_m_s=float(args.linear_speed_m_s),
            angular_speed_deg_s=float(args.angular_speed_deg_s),
            minimum_duration_sec=float(args.minimum_duration_sec),
            require_obstacle_detour=bool(args.require_obstacle_detour),
        )
        sim_config = _simulation_config(profile, snapshot)
        real_ports_factory = (
            _build_real_ports_factory(
                context=context,
                profile=profile,
                target=target,
            )
            if args.execute_real
            else None
        )
        post_simulation_confirmation = _PostSimulationConfirmation()
        real_authority_factory = (
            _build_real_authority_factory(
                confirmation=post_simulation_confirmation,
            )
            if args.execute_real
            else None
        )
        viewer_enabled = bool(args.mujoco_viewer or args.execute_real)
        receipt = run_real_shadow_acceptance(
            # The target and all downstream planning must share the exact
            # causal state cut captured above.  Reading the hardware again
            # here can introduce a small qpos drift between
            # ``target-from-current`` and the trajectory start, making an
            # otherwise stationary acceptance request appear to require a
            # collision-prone move.
            snapshot_provider=lambda: snapshot,
            trajectory_builder=lambda snap: build_registered_tcp_obstacle_trajectory(
                snap,
                request,
                control_factory=lambda state, obstacle: _build_shadow_control(
                    profile,
                    state,
                    obstacle,
                ),
            ),
            simulation_control_factory=lambda snap: _build_shadow_control(
                profile,
                snap,
                scene,
            ),
            model_binding_provider=lambda _snap, trajectory: _build_binding(
                profile,
                trajectory,
            ),
            simulation_executor=lambda *, control, trajectory, model_binding: (
                _execute_registered_shadow_with_viewer(
                    control=control,
                    trajectory=trajectory,
                    model_binding=model_binding,
                    target=target,
                    obstacle_scene=scene,
                    confirmation=post_simulation_confirmation,
                    require_confirmation=bool(args.execute_real),
                    playback_hz=float(args.viewer_playback_hz),
                    terminal_display_sec=float(args.viewer_terminal_display_sec),
                    camera_distance=float(args.viewer_camera_distance),
                    camera_azimuth_deg=float(args.viewer_camera_azimuth_deg),
                    camera_elevation_deg=float(args.viewer_camera_elevation_deg),
                )
                if viewer_enabled
                else execute_registered_shadow_trajectory(
                    control=control,
                    trajectory=trajectory,
                    model_binding=model_binding,
                )
            ),
            real_replay_enabled=bool(args.execute_real),
            real_ports_factory=real_ports_factory,
            current_scene_hash_provider=lambda: scene.semantic_content_hash,
            real_authority_factory=real_authority_factory,
        )
        payload = {
            "schema": "agenticrobot.real_shadow_acceptance.v1",
            "mode": "real_replay" if args.execute_real else "observation_only",
            "status": receipt.status.value,
            "real_motion_disposition": receipt.real_motion_disposition.value,
            "terminal_hold_confirmed": receipt.terminal_hold_confirmed,
            "snapshot_hash": receipt.snapshot_hash,
            "trajectory_hash": receipt.trajectory_hash,
            "reason_codes": receipt.reason_codes,
            "evidence_refs": receipt.evidence_refs,
            "receipt_hash": receipt.semantic_content_hash,
            "simulation_receipt": receipt.simulation_receipt,
            "real_replay_receipt": receipt.real_replay_receipt,
            "git": _git_metadata(),
            "target": target,
            "obstacle_scene": scene,
            "sim_config": sim_config,
            "trajectory_summary": {
                "configured_obstacle_count": len(request.obstacle_scene.obstacles),
                "obstacle_detour_required": bool(args.require_obstacle_detour),
                "obstacle_detour_verified": bool(
                    args.require_obstacle_detour
                    and receipt.status.value
                    in {"simulation_accepted_no_motion", "real_replay_completed"}
                ),
            },
            "acceptance_ui": {
                "mujoco_viewer_enabled": viewer_enabled,
                "post_simulation_terminal_confirmation_required": bool(
                    args.execute_real
                ),
                "post_simulation_terminal_confirmation_obtained": bool(
                    post_simulation_confirmation.reference
                ),
            },
        }
        _write_output(payload, args.output_json)
        return (
            0
            if receipt.status.value
            in {
                "simulation_accepted_no_motion",
                "real_replay_completed",
            }
            else 1
        )
    except Exception as exc:  # noqa: BLE001 - always close both lower sources.
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        reason_code = getattr(exc, "reason_code", None)
        if reason_code is not None:
            error["reason_code"] = str(reason_code)
        details = getattr(exc, "details", None)
        if details is not None:
            error["details"] = _jsonable(details)
        _write_output(
            {
                "schema": "agenticrobot.real_shadow_acceptance.v1",
                "mode": "real_replay" if args.execute_real else "observation_only",
                "status": "runtime_failed",
                "git": _git_metadata(),
                "error": error,
            },
            args.output_json,
        )
        return 1
    finally:
        cleanup_errors = context.close()
        if cleanup_errors:
            print(
                "cleanup errors: " + "; ".join(cleanup_errors),
                file=sys.stderr,
            )


if __name__ == "__main__":
    raise SystemExit(main())
