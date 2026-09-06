"""RGB-D point-grasp Skill to physical MuJoCo cube-lift evaluation.

This experiment composes existing owners instead of creating another motion
runtime: the read-only Gateway provider owns SAM2 + GraspNet inference, while
``CubeLiftKernel`` retains IK, collision projection, actuator, contact, and
episode-success authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from agent_gateway.profiles.real.point_grasp_deployment import (
    build_real_point_grasp_registration,
)
from agent_gateway.profiles.real.point_grasp_gateway import PointGraspProvider
from agent_gateway.skills.perception.point_grasp_schemas import (
    validate_point_grasp_parameters_v1,
    validate_point_grasp_result_for_parameters_v1,
    validate_point_grasp_result_v1,
)
from deployments.perception.sam2_graspnet_point_grasp import (
    GraspNetPointGraspModelSpec,
    Sam2GraspNetPointGraspPlanner,
)
from deployments.perception.sam2_grounding_dino import (
    Sam2GroundingDinoLocalizer,
    Sam2GroundingDinoModelSpec,
)
from deployments.perception.sam2_vgn_point_grasp import (
    DMTAC_M_MAX_VISIBLE_COLLISION_POINTS,
    PARALLEL_JAW_TCP_UPWARD_SELECTION_V1,
)
from project_paths import resolve_path
from robot_motion.rm65 import endpoint_to_tool_transform, load_rm65_tool_profile
from sim.mujoco.pointcloud import RenderCameraIntrinsics, render_intrinsics_from_fovy
from sim.mujoco.scene import MujocoCameraRenderer
from sim.mujoco.tasks.cube_lift.config import (
    BILATERAL_CONTACT_LIFT_SUCCESS_REASON,
    CubeLiftTaskConfig,
)
from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv

DEFAULT_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v2.json"
)
EVALUATION_LIGHT_SCENE_CONTRACT = Path(
    "experiments/sim_point_grasp/configs/rm65_bimanual_tall_box_v1.json"
)
EVALUATION_LIGHT_SCENE_CONTRACT_SHA256 = (
    "7aba17d52d7e0def19620945222df9325b29fa089f604ea75378e6a2777f0cbd"
)
EVALUATION_HEAVY_SCENE_CONTRACT = Path(
    "experiments/sim_point_grasp/configs/rm65_bimanual_tall_box_heavy_v1.json"
)
EVALUATION_HEAVY_SCENE_CONTRACT_SHA256 = (
    "43a9c7b2590a4fd8980feb9e1162640d50504dd9ba5778cf7d87e7a700831789"
)
EVALUATION_RIGHT_TOOL_PROFILE = Path(
    "experiments/sim_point_grasp/configs/rm65_right_DMTAC_M_4b2_cad_tool_tcp.json"
)
EVALUATION_RIGHT_TOOL_PROFILE_SHA256 = (
    "c9746da4094a36c252dd96c901ccca9bc0f4f419154b8fb9ea786b321a14dc5a"
)
EVALUATION_OBJECT_SIZE_M = (0.04, 0.04, 0.10)
EVALUATION_OBJECT_CENTER_XY_M = (-0.272, 0.0)
EVALUATION_GRIPPER_PROFILE_NAME = "dmtac_bimanual_position_eg2_4b2_cad_20260829.v2"
EVALUATION_GRIPPER_PROFILE_SHA256 = (
    "171c6456fffd3fc9abcd02911c389048eb6cdf5a8732bc3704303008c8c12a8c"
)
EVALUATION_STRONG_GRIPPER_PROFILE_NAME = (
    "dmtac_bimanual_position_eg2_4b2_cad_20260831.v3_strong"
)
EVALUATION_STRONG_GRIPPER_PROFILE_SHA256 = (
    "192230c71e240b9ab296a3a83d90c7ea5f7d41a0b322890f33aedb6b2c36f3c8"
)
MULTIPOINT_MAX_COMBINED_CANDIDATES = 10
RESULT_SCHEMA_V1 = "dexteleop.experiments.sim_point_grasp_evaluation.v1"
TRIAL_SCHEMA_V1 = "dexteleop.experiments.sim_point_grasp_trial.v1"
CAMERA_NAME = "front"


@dataclass(frozen=True)
class SimPointGraspEvaluationConfig:
    output_dir: Path
    sam2_asset_bundle: Path
    graspnet_asset_bundle: Path
    trials: int = 10
    seed_start: int = 0
    device: str = "cuda"
    max_candidates: int = 5
    target_point_count: int = 1
    object_mass_kg: float = 0.08
    gripper_force_profile: str = "standard"
    cube_center_xy_m: tuple[float, float] = EVALUATION_OBJECT_CENTER_XY_M
    cube_spawn_xy_jitter_m: tuple[float, float] = (0.025, 0.080)
    cube_spawn_yaw_jitter_rad: float = math.pi / 4.0
    pregrasp_offset_m: float = 0.070
    adaptive_pregrasp_offsets_m: tuple[float, ...] = (0.050, 0.035)
    candidate_selection_policy: str = "reachability_aware"
    lift_offset_m: float = 0.100
    pose_position_tolerance_m: float = 0.008
    pose_orientation_tolerance_rad: float = math.radians(6.0)
    phase_step_limit: int = 90
    lift_step_limit: int = 120
    close_step_limit: int = 15
    close_action_scale: float = 1.0
    contact_stability_steps: int = 1
    lift_translation_action_scale: float = 0.25
    save_video: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_dir", Path(self.output_dir).expanduser().resolve()
        )
        for field_name in ("sam2_asset_bundle", "graspnet_asset_bundle"):
            path = Path(getattr(self, field_name)).expanduser().resolve(strict=True)
            if not path.is_dir():
                raise ValueError(f"{field_name} must be an existing directory.")
            object.__setattr__(self, field_name, path)
        if not 1 <= int(self.trials) <= 100:
            raise ValueError("trials must be in [1, 100].")
        if not 1 <= int(self.max_candidates) <= 5:
            raise ValueError("max_candidates must be in [1, 5].")
        if not 1 <= int(self.target_point_count) <= 8:
            raise ValueError("target_point_count must be in [1, 8].")
        object.__setattr__(self, "target_point_count", int(self.target_point_count))
        mass = float(self.object_mass_kg)
        if mass not in {0.08, 0.20}:
            raise ValueError("object_mass_kg must be one of {0.08, 0.20}.")
        object.__setattr__(self, "object_mass_kg", mass)
        if self.gripper_force_profile not in {"standard", "strong"}:
            raise ValueError("gripper_force_profile must be standard or strong.")
        if not str(self.device).startswith("cuda"):
            raise ValueError("The reviewed GraspNet profile requires a CUDA device.")
        if len(self.cube_center_xy_m) != 2 or any(
            not math.isfinite(float(value)) for value in self.cube_center_xy_m
        ):
            raise ValueError("cube_center_xy_m must contain two finite values.")
        object.__setattr__(
            self,
            "cube_center_xy_m",
            tuple(float(value) for value in self.cube_center_xy_m),
        )
        if len(self.cube_spawn_xy_jitter_m) != 2 or any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in self.cube_spawn_xy_jitter_m
        ):
            raise ValueError(
                "cube_spawn_xy_jitter_m must contain two non-negative values."
            )
        for field_name in (
            "pregrasp_offset_m",
            "lift_offset_m",
            "pose_position_tolerance_m",
            "pose_orientation_tolerance_rad",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")
        adaptive_offsets = tuple(
            float(value) for value in self.adaptive_pregrasp_offsets_m
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in adaptive_offsets):
            raise ValueError(
                "adaptive_pregrasp_offsets_m must contain positive finite values."
            )
        object.__setattr__(self, "adaptive_pregrasp_offsets_m", adaptive_offsets)
        if self.candidate_selection_policy not in {
            "provider_selected_ordinal",
            "reachability_aware",
        }:
            raise ValueError(
                "candidate_selection_policy must be provider_selected_ordinal or "
                "reachability_aware."
            )
        if not 1 <= int(self.phase_step_limit) <= 300:
            raise ValueError("phase_step_limit must be in [1, 300].")
        if not 1 <= int(self.lift_step_limit) <= 300:
            raise ValueError("lift_step_limit must be in [1, 300].")
        if not 1 <= int(self.close_step_limit) <= 100:
            raise ValueError("close_step_limit must be in [1, 100].")
        if not math.isfinite(float(self.close_action_scale)) or not (
            0.0 < float(self.close_action_scale) <= 1.0
        ):
            raise ValueError("close_action_scale must be finite and in (0, 1].")
        if not 1 <= int(self.contact_stability_steps) <= int(self.close_step_limit):
            raise ValueError(
                "contact_stability_steps must be in [1, close_step_limit]."
            )
        if not math.isfinite(float(self.lift_translation_action_scale)) or not (
            0.0 < float(self.lift_translation_action_scale) <= 1.0
        ):
            raise ValueError(
                "lift_translation_action_scale must be finite and in (0, 1]."
            )


@dataclass(frozen=True)
class SimPointGraspTrialResult:
    seed: int
    success: bool
    failure_stage: str | None
    reason: str
    termination_reason: str | None
    selected_score: float | None
    selected_model_score: float | None
    selected_width_m: float | None
    candidate_count: int
    click_uv: tuple[int, int] | None
    mask_bbox_xyxy: tuple[int, int, int, int] | None
    max_lift_m: float
    bilateral_contact_observed: bool
    motion_steps: int
    safety_intervention_steps: int
    ik_rejection_steps: int
    trial_dir: str
    timing_ms: dict[str, Any] = field(default_factory=dict)
    ik_solver: str | None = None
    max_pose_ik_limit_violation_rad: float = 0.0
    pose_ik_limit_violation_side: str | None = None
    pose_ik_limit_violation_joint: int | None = None
    pose_ik_sync_failure_reason: str | None = None
    selected_ordinal: int | None = None
    selected_pregrasp_offset_m: float | None = None
    selected_wrist_symmetry_deg: int = 0
    candidate_selection_policy: str = "provider_selected_ordinal"
    candidate_preflight: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": TRIAL_SCHEMA_V1, **asdict(self)}


@dataclass
class _ExecutionTrace:
    info: dict[str, Any]
    step_count: int = 0
    safety_intervention_steps: int = 0
    ik_rejection_steps: int = 0
    bilateral_contact_observed: bool = False
    max_cube_z_m: float = -math.inf
    termination_reason: str | None = None
    ik_solve_samples_ms: list[float] = field(default_factory=list)
    ik_frame_total_samples_ms: list[float] = field(default_factory=list)
    ik_solver: str | None = None
    max_pose_ik_limit_violation_rad: float = 0.0
    pose_ik_limit_violation_side: str | None = None
    pose_ik_limit_violation_joint: int | None = None
    pose_ik_sync_failure_reason: str | None = None

    def observe(
        self,
        info: Mapping[str, Any],
        *,
        cube_z_m: float | None = None,
        ik_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.info = dict(info)
        self.step_count += 1
        self.safety_intervention_steps += int(
            bool(info.get("safety_intervention", False))
        )
        self.ik_rejection_steps += int(
            info.get("pose_ik_success") is False
            or bool(info.get("pose_ik_hold", False))
        )
        self.bilateral_contact_observed = bool(
            self.bilateral_contact_observed
            or info.get("bilateral_finger_cube_contact", False)
        )
        cube_z = info.get("cube_z_m") if cube_z_m is None else cube_z_m
        if cube_z is not None and math.isfinite(float(cube_z)):
            self.max_cube_z_m = max(self.max_cube_z_m, float(cube_z))
        self.termination_reason = info.get("termination_reason")
        solver = info.get("pose_ik_solver")
        if isinstance(solver, str) and solver:
            self.ik_solver = solver
        violation = info.get("pose_ik_limit_violation_rad")
        if isinstance(violation, (int, float)) and math.isfinite(float(violation)):
            value = float(violation)
            if value >= self.max_pose_ik_limit_violation_rad:
                self.max_pose_ik_limit_violation_rad = value
                side = info.get("pose_ik_limit_violation_side")
                joint = info.get("pose_ik_limit_violation_joint")
                self.pose_ik_limit_violation_side = (
                    side if isinstance(side, str) and side else None
                )
                self.pose_ik_limit_violation_joint = (
                    int(joint)
                    if isinstance(joint, (int, np.integer))
                    and not isinstance(joint, (bool, np.bool_))
                    else None
                )
        sync_failure = info.get("pose_ik_sync_failure_reason")
        if isinstance(sync_failure, str) and sync_failure:
            self.pose_ik_sync_failure_reason = sync_failure
        metadata = dict(ik_metadata or {})
        for name, samples in (
            ("solve_ms", self.ik_solve_samples_ms),
            ("frame_total_ms", self.ik_frame_total_samples_ms),
        ):
            value = metadata.get(name)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                samples.append(float(value))


@dataclass(frozen=True)
class _CandidatePreflight:
    ordinal: int
    pregrasp_offset_m: float
    wrist_symmetry_deg: int
    pregrasp_reached: bool
    grasp_reached: bool
    reason: str
    termination_reason: str | None
    motion_steps: int
    safety_intervention_steps: int
    ik_rejection_steps: int
    final_joint_margin_rad: float | None
    model_score: float
    combined_score: float

    @property
    def executable(self) -> bool:
        return self.pregrasp_reached and self.grasp_reached

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"executable": self.executable}


def _timing_summary_ms(samples: list[float]) -> dict[str, float | int | None]:
    values = np.asarray(
        [float(value) for value in samples if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if not len(values):
        return {
            "count": 0,
            "total": 0.0,
            "mean": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "total": float(values.sum()),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _last_ik_metadata(env: FastCubeLiftEnv) -> Mapping[str, Any]:
    controller = env.kernel._pose_action_controller
    if controller is None or controller.last_step is None:
        return {}
    metadata = controller.last_step.ik_result.metadata
    return metadata if isinstance(metadata, Mapping) else {}


def _aggregate_trial_timings(
    results: list[SimPointGraspTrialResult],
) -> dict[str, Any]:
    scalar_samples: dict[str, list[float]] = {}
    ik_solve_samples: list[float] = []
    ik_frame_samples: list[float] = []
    for result in results:
        for name, value in result.timing_ms.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                scalar_samples.setdefault(name, []).append(float(value))
        ik_solve_samples.extend(result.timing_ms.get("ik_pose_solve_samples", ()))
        ik_frame_samples.extend(result.timing_ms.get("ik_pose_frame_total_samples", ()))
    return {
        "trial_phase_wall_clock": {
            name: _timing_summary_ms(samples)
            for name, samples in sorted(scalar_samples.items())
        },
        "ik_pose_solve_per_call": _timing_summary_ms(ik_solve_samples),
        "ik_pose_frame_total_per_call": _timing_summary_ms(ik_frame_samples),
    }


class _TrialVideo:
    def __init__(self, path: Path, *, width: int, height: int, enabled: bool) -> None:
        self._writer = None
        if enabled:
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                15.0,
                (int(width), int(height)),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open evaluation video writer: {path}")
            self._writer = writer

    def write(self, rgb: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(
                cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            )

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def build_sim_point_grasp_task_config(
    *,
    object_mass_kg: float = 0.08,
    gripper_force_profile: str = "standard",
    cube_center_xy_m: tuple[float, float] = EVALUATION_OBJECT_CENTER_XY_M,
    cube_spawn_xy_jitter_m: tuple[float, float],
    cube_spawn_yaw_jitter_rad: float,
    max_episode_steps: int | None = None,
) -> CubeLiftTaskConfig:
    """Return the physical Cube-Lift profile with observable random spawn."""

    base = CubeLiftTaskConfig.load(DEFAULT_TASK_CONFIG)
    if float(object_mass_kg) == 0.08:
        contract_relative = EVALUATION_LIGHT_SCENE_CONTRACT
        expected_contract_sha256 = EVALUATION_LIGHT_SCENE_CONTRACT_SHA256
    elif float(object_mass_kg) == 0.20:
        contract_relative = EVALUATION_HEAVY_SCENE_CONTRACT
        expected_contract_sha256 = EVALUATION_HEAVY_SCENE_CONTRACT_SHA256
    else:
        raise ValueError("object_mass_kg must be one of {0.08, 0.20}.")
    if gripper_force_profile == "standard":
        gripper_profile_name = EVALUATION_GRIPPER_PROFILE_NAME
        gripper_profile_sha256 = EVALUATION_GRIPPER_PROFILE_SHA256
    elif gripper_force_profile == "strong":
        gripper_profile_name = EVALUATION_STRONG_GRIPPER_PROFILE_NAME
        gripper_profile_sha256 = EVALUATION_STRONG_GRIPPER_PROFILE_SHA256
    else:
        raise ValueError("gripper_force_profile must be standard or strong.")
    contract_path = resolve_path(contract_relative)
    contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    if contract_sha256 != expected_contract_sha256:
        raise ValueError(
            "Simulation point-grasp scene contract hash changed: "
            f"expected={expected_contract_sha256}, actual={contract_sha256}."
        )
    tool_profile_path = resolve_path(EVALUATION_RIGHT_TOOL_PROFILE)
    tool_profile_sha256 = hashlib.sha256(tool_profile_path.read_bytes()).hexdigest()
    if tool_profile_sha256 != EVALUATION_RIGHT_TOOL_PROFILE_SHA256:
        raise ValueError(
            "Simulation point-grasp CAD TCP profile hash changed: "
            f"expected={EVALUATION_RIGHT_TOOL_PROFILE_SHA256}, "
            f"actual={tool_profile_sha256}."
        )
    episode_steps = (
        base.max_episode_steps if max_episode_steps is None else int(max_episode_steps)
    )
    if episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive.")
    return replace(
        base,
        name="rm65_right_cube_lift_sam2_graspnet_evaluation_v1",
        base_contract_path=str(contract_relative),
        base_contract_sha256=contract_sha256,
        object_size_m=EVALUATION_OBJECT_SIZE_M,
        wrist_rgb_camera_name=CAMERA_NAME,
        cube_center_xy_m=tuple(float(value) for value in cube_center_xy_m),
        cube_spawn_xy_jitter_m=tuple(float(value) for value in cube_spawn_xy_jitter_m),
        cube_spawn_yaw_jitter_rad=float(cube_spawn_yaw_jitter_rad),
        success_requires_bilateral_contact=True,
        gripper_actuation_profile_name=gripper_profile_name,
        gripper_actuation_profile_sha256=gripper_profile_sha256,
        left_gripper_name="left_eg2_4b2_cad",
        right_gripper_name="right_eg2_4b2_cad",
        initial_gripper_close_by_name={
            "left_eg2_4b2_cad": 0.0,
            "right_eg2_4b2_cad": 0.0,
        },
        right_tool_profile_path=str(EVALUATION_RIGHT_TOOL_PROFILE),
        right_grasp_body_name="right_eg2_4b2_cad_ee_link_4",
        grasp_latch_left_finger_body_name="right_eg2_4b2_cad_ee_link_2",
        grasp_latch_right_finger_body_name="right_eg2_4b2_cad_ee_link_3",
        pose_target_anti_windup=True,
        max_episode_steps=episode_steps,
        # Once real bilateral contact is observed, keep the accepted physical
        # closure target separate from arm collision projection.  Otherwise a
        # lagging measured finger state can overwrite the force target on the
        # first lift step and silently release the object.
        post_contact_phase_enabled=True,
        post_contact_phase_activation_steps=1,
        post_contact_phase_min_measured_close=0.0,
        post_contact_phase_loss_grace_steps=3,
    )


def optical_camera_pose_world(scene: Any, camera_name: str = CAMERA_NAME) -> np.ndarray:
    """Return URDF optical-frame pose, not MuJoCo's Rx(pi) render frame."""

    pose = np.asarray(scene.camera_link_pose(camera_name), dtype=float).reshape(4, 4)
    if not np.all(np.isfinite(pose)):
        raise RuntimeError("Optical camera pose is non-finite.")
    return pose


def project_world_point(
    point_world_m: np.ndarray,
    *,
    world_from_camera: np.ndarray,
    intrinsics: RenderCameraIntrinsics,
) -> tuple[int, int, float]:
    camera_from_world = np.linalg.inv(
        np.asarray(world_from_camera, dtype=float).reshape(4, 4)
    )
    point = (
        camera_from_world
        @ np.r_[np.asarray(point_world_m, dtype=float).reshape(3), 1.0]
    )
    if not np.all(np.isfinite(point)) or point[2] <= 0.0:
        raise RuntimeError("Target center is behind the fixed optical camera.")
    u = round(intrinsics.fx * point[0] / point[2] + intrinsics.cx)
    v = round(intrinsics.fy * point[1] / point[2] + intrinsics.cy)
    if not (0 <= u < intrinsics.width and 0 <= v < intrinsics.height):
        raise RuntimeError(
            "Projected target center lies outside the fixed camera image."
        )
    return u, v, float(point[2])


def sample_mask_prompt_pixels(
    mask: np.ndarray,
    depth_m: np.ndarray,
    *,
    initial_uv: tuple[int, int],
    count: int,
) -> list[tuple[int, int]]:
    """Choose deterministic, spatially dispersed valid points inside a SAM mask."""

    target = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float32)
    if target.ndim != 2 or depth.shape != target.shape:
        raise ValueError("mask and depth_m must be aligned HxW arrays.")
    if not 1 <= int(count) <= 8:
        raise ValueError("count must be in [1, 8].")
    u0, v0 = (int(initial_uv[0]), int(initial_uv[1]))
    if not (0 <= v0 < target.shape[0] and 0 <= u0 < target.shape[1]):
        raise ValueError("initial_uv lies outside the mask image.")
    valid = target & np.isfinite(depth) & (depth > 0.0)
    if not valid[v0, u0]:
        raise ValueError("initial_uv must have valid depth inside the target mask.")
    eroded = cv2.erode(
        valid.astype(np.uint8),
        np.ones((7, 7), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    if np.count_nonzero(eroded) < int(count):
        eroded = valid
    rows, columns = np.nonzero(eroded)
    coordinates = np.column_stack((columns, rows)).astype(np.float64)
    selected: list[tuple[int, int]] = [(u0, v0)]
    minimum_distance_sq = np.sum(
        (coordinates - np.asarray((u0, v0), dtype=np.float64)) ** 2,
        axis=1,
    )
    while len(selected) < int(count):
        index = int(np.argmax(minimum_distance_sq))
        point = (int(columns[index]), int(rows[index]))
        if point in selected or float(minimum_distance_sq[index]) <= 0.0:
            break
        selected.append(point)
        distance_sq = np.sum(
            (coordinates - np.asarray(point, dtype=np.float64)) ** 2,
            axis=1,
        )
        minimum_distance_sq = np.minimum(minimum_distance_sq, distance_sq)
    if len(selected) != int(count):
        raise RuntimeError(
            "Target mask does not contain enough distinct prompt pixels."
        )
    return selected


def merge_point_prompt_candidates(
    candidates_by_prompt: list[list[Mapping[str, Any]]],
    prompt_pixels: list[tuple[int, int]],
    *,
    maximum: int = MULTIPOINT_MAX_COMBINED_CANDIDATES,
) -> list[dict[str, Any]]:
    """Merge prompt candidates while removing near-identical grasp poses."""

    if len(candidates_by_prompt) != len(prompt_pixels):
        raise ValueError("Candidate batches and prompt pixels must have equal length.")
    flattened: list[dict[str, Any]] = []
    for prompt_index, (batch, prompt_uv) in enumerate(
        zip(candidates_by_prompt, prompt_pixels, strict=True)
    ):
        for provider_ordinal, candidate in enumerate(batch):
            item = dict(candidate)
            item["_prompt_index"] = int(prompt_index)
            item["_prompt_uv"] = [int(prompt_uv[0]), int(prompt_uv[1])]
            item["_provider_ordinal"] = int(provider_ordinal)
            flattened.append(item)
    flattened.sort(
        key=lambda item: (
            -float(item["score"]),
            -float(item["model_score"]),
            int(item["_prompt_index"]),
            int(item["_provider_ordinal"]),
        )
    )
    merged: list[dict[str, Any]] = []
    for item in flattened:
        pose = np.asarray(item["T_base_from_tcp"], dtype=float).reshape(4, 4)
        duplicate = False
        for existing in merged:
            other = np.asarray(existing["T_base_from_tcp"], dtype=float).reshape(4, 4)
            translation = float(np.linalg.norm(pose[:3, 3] - other[:3, 3]))
            rotation = float(
                Rotation.from_matrix(other[:3, :3].T @ pose[:3, :3]).magnitude()
            )
            if translation <= 0.004 and rotation <= math.radians(5.0):
                duplicate = True
                break
        if not duplicate:
            merged.append(item)
        if len(merged) >= int(maximum):
            break
    if not merged:
        raise RuntimeError("Point prompts returned no distinct grasp candidates.")
    return merged


def bounded_pose_action(
    current_target: np.ndarray,
    desired_target: np.ndarray,
    *,
    translation_step_m: tuple[float, float, float],
    rotation_step_rad: tuple[float, float, float],
) -> np.ndarray:
    """Create one normalized arm-base pose increment toward an absolute goal."""

    current = np.asarray(current_target, dtype=float).reshape(4, 4)
    desired = np.asarray(desired_target, dtype=float).reshape(4, 4)
    action = np.zeros(7, dtype=np.float32)
    action[:3] = np.clip(
        (desired[:3, 3] - current[:3, 3]) / np.asarray(translation_step_m, dtype=float),
        -1.0,
        1.0,
    )
    rotation_error = Rotation.from_matrix(
        desired[:3, :3] @ current[:3, :3].T
    ).as_rotvec()
    action[3:6] = np.clip(
        rotation_error / np.asarray(rotation_step_rad, dtype=float),
        -1.0,
        1.0,
    )
    return action


def _pose_error(current: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    position = float(np.linalg.norm(target[:3, 3] - current[:3, 3]))
    rotation = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T)
    return position, float(np.linalg.norm(rotation.as_rotvec()))


def _current_tool_pose_arm_base(scene: Any, config: CubeLiftTaskConfig) -> np.ndarray:
    side = config.controlled_side
    base = scene.body_pose(f"{side}_arm_base_link")
    endpoint = scene.body_pose(scene.arm_specs[side].controlled_body)
    profile = load_rm65_tool_profile(
        resolve_path(config.right_tool_profile_path), side=side
    )
    return np.linalg.inv(base) @ endpoint @ endpoint_to_tool_transform(profile)


def build_pregrasp_pose(grasp_pose: np.ndarray, offset_m: float) -> np.ndarray:
    """Back away from a grasp along its declared approach axis."""

    pose = np.asarray(grasp_pose, dtype=float).reshape(4, 4).copy()
    offset = float(offset_m)
    if not math.isfinite(offset) or offset <= 0.0:
        raise ValueError("pregrasp offset must be positive and finite.")
    pose[:3, 3] -= pose[:3, 2] * offset
    return pose


def apply_parallel_jaw_wrist_symmetry(
    grasp_pose: np.ndarray,
    wrist_symmetry_deg: int,
) -> np.ndarray:
    """Return an object-equivalent pose that swaps the two symmetric fingers."""

    degrees = int(wrist_symmetry_deg)
    if degrees not in (0, 180):
        raise ValueError("parallel-jaw wrist symmetry must be 0 or 180 degrees.")
    pose = np.asarray(grasp_pose, dtype=float).reshape(4, 4).copy()
    if degrees == 180:
        pose[:3, :3] = pose[:3, :3] @ np.diag((-1.0, -1.0, 1.0))
    return pose


def _minimum_controlled_joint_margin_rad(env: FastCubeLiftEnv) -> float | None:
    controller = env.kernel._pose_action_controller
    if controller is None:
        return None
    lower_by_side, upper_by_side = controller._effective_joint_limits_by_side()
    side = env.kernel.config.controlled_side
    qpos = np.asarray(env.kernel.scene.get_arm_qpos(side), dtype=float).reshape(6)
    margins = np.minimum(qpos - lower_by_side[side], upper_by_side[side] - qpos)
    if not np.all(np.isfinite(margins)):
        return None
    return float(np.min(margins))


def _project_camera_point(
    point: np.ndarray,
    intrinsics: Mapping[str, float | int],
) -> tuple[int, int] | None:
    xyz = np.asarray(point, dtype=float).reshape(3)
    if xyz[2] <= 1e-6:
        return None
    u = round(float(intrinsics["fx"]) * xyz[0] / xyz[2] + float(intrinsics["cx"]))
    v = round(float(intrinsics["fy"]) * xyz[1] / xyz[2] + float(intrinsics["cy"]))
    if not (0 <= u < int(intrinsics["width"]) and 0 <= v < int(intrinsics["height"])):
        return None
    return u, v


def draw_grasp_overlay(
    rgb: np.ndarray,
    *,
    camera_from_tcp: np.ndarray,
    gripper_width_m: float,
    intrinsics: Mapping[str, float | int],
    click_uv: tuple[int, int],
    bbox_xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    """Draw TCP axes and an approximate two-finger footprint in image space."""

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (click_uv[0] - 5, click_uv[1] - 5, click_uv[0] + 5, click_uv[1] + 5),
        outline=(255, 255, 0),
        width=3,
    )
    x0, y0, x1, y1 = bbox_xyxy
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(40, 220, 255), width=3)
    transform = np.asarray(camera_from_tcp, dtype=float).reshape(4, 4)

    def pixel(local: tuple[float, float, float]) -> tuple[int, int] | None:
        camera = transform @ np.r_[np.asarray(local, dtype=float), 1.0]
        return _project_camera_point(camera[:3], intrinsics)

    origin = pixel((0.0, 0.0, 0.0))
    if origin is not None:
        for local, color, label in (
            ((0.035, 0.0, 0.0), (255, 40, 40), "x/open"),
            ((0.0, 0.035, 0.0), (40, 220, 40), "y"),
            ((0.0, 0.0, 0.050), (40, 100, 255), "z/approach"),
        ):
            endpoint = pixel(local)
            if endpoint is not None:
                draw.line((*origin, *endpoint), fill=color, width=4)
                draw.text((endpoint[0] + 3, endpoint[1] + 3), label, fill=color)
        half_width = float(gripper_width_m) / 2.0
        for sign in (-1.0, 1.0):
            jaw = pixel((sign * half_width, 0.0, 0.0))
            finger = pixel((sign * half_width, 0.0, 0.050))
            if jaw is not None and finger is not None:
                draw.line((*jaw, *finger), fill=(255, 128, 20), width=5)
        left = pixel((-half_width, 0.0, 0.0))
        right = pixel((half_width, 0.0, 0.0))
        if left is not None and right is not None:
            draw.line((*left, *right), fill=(255, 128, 20), width=3)
    return np.asarray(image)


def _label_frame(frame: np.ndarray, label: str) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 28), fill=(0, 0, 0))
    draw.text((8, 7), label, fill=(255, 255, 255))
    return image


def save_montage(path: Path, frames: list[tuple[str, np.ndarray]]) -> None:
    if not frames:
        return
    labeled = [_label_frame(frame, label) for label, frame in frames]
    width = max(item.width for item in labeled)
    height = max(item.height for item in labeled)
    columns = min(3, len(labeled))
    rows = math.ceil(len(labeled) / columns)
    canvas = Image.new("RGB", (columns * width, rows * height), color=(28, 28, 28))
    for index, image in enumerate(labeled):
        canvas.paste(image, ((index % columns) * width, (index // columns) * height))
    canvas.save(path)


def save_summary_sheet(
    path: Path,
    results: list[SimPointGraspTrialResult],
) -> None:
    """Save a compact success/failure contact sheet across all trials."""

    if not results:
        return
    tile_width, tile_height, label_height = 480, 240, 34
    columns = 2
    rows = math.ceil(len(results) / columns)
    canvas = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + label_height)),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    for index, result in enumerate(results):
        montage_path = Path(result.trial_dir) / "execution_montage.png"
        montage = Image.open(montage_path).convert("RGB")
        montage.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + label_height)
        color = (35, 150, 70) if result.success else (180, 55, 45)
        draw.rectangle((x, y, x + tile_width, y + label_height), fill=color)
        status = "SUCCESS" if result.success else f"FAIL {result.failure_stage}"
        draw.text((x + 8, y + 9), f"seed={result.seed}  {status}", fill=(255, 255, 255))
        canvas.paste(
            montage,
            (x + (tile_width - montage.width) // 2, y + label_height),
        )
    canvas.save(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _drive_pose(
    *,
    env: FastCubeLiftEnv,
    desired_pose: np.ndarray,
    trace: _ExecutionTrace,
    renderer: MujocoCameraRenderer | None,
    video: _TrialVideo | None,
    maximum_steps: int,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    translation_action_scale: float = 1.0,
    on_frame: Callable[[np.ndarray], None] | None = None,
) -> tuple[bool, str]:
    kernel = env.kernel
    controller = kernel._pose_action_controller
    if controller is None:
        raise RuntimeError(
            "Physical Cube-Lift profile has no pose-action motion owner."
        )
    for _ in range(maximum_steps):
        measured = _current_tool_pose_arm_base(kernel.scene, kernel.config)
        position_error, orientation_error = _pose_error(measured, desired_pose)
        if (
            position_error <= position_tolerance_m
            and orientation_error <= orientation_tolerance_rad
        ):
            return True, "pose_reached"
        action = bounded_pose_action(
            controller.target_pose_arm_base,
            desired_pose,
            translation_step_m=kernel.config.right_tcp_delta_m,
            rotation_step_rad=kernel.config.right_tcp_rotation_delta_rad,
        )
        action[:3] *= float(translation_action_scale)
        step = env.step(action)
        trace.observe(
            step[4],
            cube_z_m=float(
                kernel.scene.body_pose(kernel.config.object_body_name)[2, 3]
            ),
            ik_metadata=_last_ik_metadata(env),
        )
        if renderer is not None:
            frame = renderer.render(CAMERA_NAME)
            if video is not None:
                video.write(frame)
            if on_frame is not None:
                on_frame(frame)
        if step[2] or step[3]:
            return False, str(step[4].get("termination_reason") or "episode_terminated")
    measured = _current_tool_pose_arm_base(kernel.scene, kernel.config)
    position_error, orientation_error = _pose_error(measured, desired_pose)
    return (
        False,
        f"pose_timeout:position={position_error:.6f},rotation={orientation_error:.6f}",
    )


def _preflight_candidate(
    *,
    env: FastCubeLiftEnv,
    candidate: Mapping[str, Any],
    ordinal: int,
    pregrasp_offset_m: float,
    wrist_symmetry_deg: int,
    seed: int,
    config: SimPointGraspEvaluationConfig,
) -> _CandidatePreflight:
    """Check IK and the authoritative MuJoCo path without closing or lifting."""

    _observation, reset_info = env.reset(seed=seed)
    trace = _ExecutionTrace(info=dict(reset_info))
    grasp = apply_parallel_jaw_wrist_symmetry(
        np.asarray(candidate["T_base_from_tcp"], dtype=float).reshape(4, 4),
        wrist_symmetry_deg,
    )
    pregrasp = build_pregrasp_pose(grasp, pregrasp_offset_m)
    pregrasp_reached, reason = _drive_pose(
        env=env,
        desired_pose=pregrasp,
        trace=trace,
        renderer=None,
        video=None,
        maximum_steps=config.phase_step_limit,
        position_tolerance_m=config.pose_position_tolerance_m,
        orientation_tolerance_rad=config.pose_orientation_tolerance_rad,
    )
    grasp_reached = False
    if pregrasp_reached:
        grasp_reached, reason = _drive_pose(
            env=env,
            desired_pose=grasp,
            trace=trace,
            renderer=None,
            video=None,
            maximum_steps=config.phase_step_limit,
            position_tolerance_m=config.pose_position_tolerance_m,
            orientation_tolerance_rad=config.pose_orientation_tolerance_rad,
        )
    return _CandidatePreflight(
        ordinal=int(ordinal),
        pregrasp_offset_m=float(pregrasp_offset_m),
        wrist_symmetry_deg=int(wrist_symmetry_deg),
        pregrasp_reached=bool(pregrasp_reached),
        grasp_reached=bool(grasp_reached),
        reason=str(reason),
        termination_reason=trace.termination_reason,
        motion_steps=trace.step_count,
        safety_intervention_steps=trace.safety_intervention_steps,
        ik_rejection_steps=trace.ik_rejection_steps,
        final_joint_margin_rad=_minimum_controlled_joint_margin_rad(env),
        model_score=float(candidate["model_score"]),
        combined_score=float(candidate["score"]),
    )


def _candidate_preflight_rank(value: _CandidatePreflight) -> tuple[Any, ...]:
    stage_rank = 0 if value.executable else (1 if value.pregrasp_reached else 2)
    margin = value.final_joint_margin_rad
    safe_margin_rank = (
        -float(margin) if margin is not None and math.isfinite(margin) else math.inf
    )
    return (
        stage_rank,
        int(value.termination_reason is not None),
        value.safety_intervention_steps,
        value.ik_rejection_steps,
        safe_margin_rank,
        value.motion_steps,
        -value.combined_score,
        value.ordinal,
        value.wrist_symmetry_deg,
    )


def select_reachable_candidate(
    *,
    env: FastCubeLiftEnv,
    candidates: list[Mapping[str, Any]],
    provider_selected_ordinal: int,
    seed: int,
    config: SimPointGraspEvaluationConfig,
) -> tuple[int, float, int, list[_CandidatePreflight]]:
    """Select on execution feasibility, never on close/contact/lift outcome."""

    if config.candidate_selection_policy == "provider_selected_ordinal":
        return int(provider_selected_ordinal), float(config.pregrasp_offset_m), 0, []

    offsets: list[float] = []
    for value in (config.pregrasp_offset_m, *config.adaptive_pregrasp_offsets_m):
        if not any(abs(float(value) - previous) <= 1e-12 for previous in offsets):
            offsets.append(float(value))
    attempts: list[_CandidatePreflight] = []
    for offset_m in offsets:
        # The point-grasp generator already resolves the parallel-jaw wrist
        # symmetry to one upward-facing tool TCP.  Do not recreate the rejected
        # upside-down TCP during execution feasibility selection.
        for wrist_symmetry_deg in (0,):
            batch = [
                _preflight_candidate(
                    env=env,
                    candidate=candidate,
                    ordinal=ordinal,
                    pregrasp_offset_m=offset_m,
                    wrist_symmetry_deg=wrist_symmetry_deg,
                    seed=seed,
                    config=config,
                )
                for ordinal, candidate in enumerate(candidates)
            ]
            attempts.extend(batch)
            executable = [value for value in batch if value.executable]
            if executable:
                selected = min(executable, key=_candidate_preflight_rank)
                return (
                    selected.ordinal,
                    selected.pregrasp_offset_m,
                    selected.wrist_symmetry_deg,
                    attempts,
                )
    selected = min(attempts, key=_candidate_preflight_rank)
    return (
        selected.ordinal,
        selected.pregrasp_offset_m,
        selected.wrist_symmetry_deg,
        attempts,
    )


def _step_action(
    *,
    env: FastCubeLiftEnv,
    action: np.ndarray,
    trace: _ExecutionTrace,
    renderer: MujocoCameraRenderer,
    video: _TrialVideo,
) -> tuple[bool, bool, dict[str, Any], np.ndarray]:
    observation, reward, terminated, truncated, info = env.step(action)
    del observation, reward
    trace.observe(
        info,
        cube_z_m=float(
            env.kernel.scene.body_pose(env.kernel.config.object_body_name)[2, 3]
        ),
        ik_metadata=_last_ik_metadata(env),
    )
    frame = renderer.render(CAMERA_NAME)
    video.write(frame)
    return terminated, truncated, info, frame


def _make_provider(
    config: SimPointGraspEvaluationConfig,
) -> tuple[Any, PointGraspProvider]:
    sam_spec = Sam2GroundingDinoModelSpec.from_asset_bundle(config.sam2_asset_bundle)
    localizer = Sam2GroundingDinoLocalizer(model_spec=sam_spec, device=config.device)
    grasp_spec = GraspNetPointGraspModelSpec.from_asset_bundle(
        config.graspnet_asset_bundle
    )
    planner = Sam2GraspNetPointGraspPlanner(
        model_spec=grasp_spec,
        point_segmenter=localizer,
        device=config.device,
    )
    registration = build_real_point_grasp_registration(
        backend_id=planner.backend_id,
        model_profile_id=grasp_spec.model_profile_id,
        model_profile_hash=grasp_spec.semantic_content_hash,
        inference_implementation_hash=planner.semantic_content_hash,
        local_path_enabled=True,
        runtime_session_id="sim-point-grasp-evaluation",
        control_domain_id="offline-mujoco-evaluation",
        target_binding_id="simulated-fixed-head-rgbd",
        station_manifest_hash="not-applicable-offline-simulation",
        execution_manifest_hash="sim-point-grasp-evaluation-v1",
    )
    provider = PointGraspProvider(
        planner=planner,
        registration=registration,
        allowed_local_image_roots=(config.output_dir,),
    )
    return localizer, provider


def _trial(
    *,
    env: FastCubeLiftEnv,
    renderer: MujocoCameraRenderer,
    provider: PointGraspProvider,
    config: SimPointGraspEvaluationConfig,
    seed: int,
) -> SimPointGraspTrialResult:
    trial_started = time.perf_counter()
    timing_ms: dict[str, Any] = {}
    trial_dir = config.output_dir / f"trial_{seed:04d}"
    trial_dir.mkdir(parents=False, exist_ok=False)
    video = _TrialVideo(
        trial_dir / "execution.mp4",
        width=renderer.width,
        height=renderer.height,
        enabled=config.save_video,
    )
    keyframes: list[tuple[str, np.ndarray]] = []
    trace = _ExecutionTrace(info={})
    selected_score = selected_model_score = selected_width = None
    selected_ordinal: int | None = None
    selected_pregrasp_offset_m: float | None = None
    selected_wrist_symmetry_deg = 0
    candidate_preflight: list[dict[str, Any]] = []
    click_uv = None
    bbox = None
    candidate_count = 0
    resting_z = math.nan
    failure_stage: str | None = "setup"
    reason = "trial_not_started"
    try:
        phase_started = time.perf_counter()
        try:
            _observation, reset_info = env.reset(seed=seed)
        finally:
            timing_ms["scene_reset"] = (time.perf_counter() - phase_started) * 1000.0
        trace.info = dict(reset_info)
        resting_z = float(reset_info["resting_center_z_m"])
        trace.max_cube_z_m = float(reset_info["cube_z_m"])
        phase_started = time.perf_counter()
        try:
            initial_rgb = renderer.render(CAMERA_NAME)
            depth = np.ascontiguousarray(
                renderer.render_depth(CAMERA_NAME), dtype=np.float32
            )
        finally:
            timing_ms["rgbd_render"] = (time.perf_counter() - phase_started) * 1000.0
        phase_started = time.perf_counter()
        video.write(initial_rgb)
        keyframes.append(("initial RGB", initial_rgb.copy()))
        Image.fromarray(initial_rgb).save(trial_dir / "initial_rgb.png")
        np.save(trial_dir / "aligned_depth_m.npy", depth, allow_pickle=False)
        finite_depth = depth[np.isfinite(depth) & (depth > 0.0)]
        if not len(finite_depth):
            raise RuntimeError("Simulated depth image has no positive metric samples.")
        low, high = np.percentile(finite_depth, (2.0, 98.0))
        scaled = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
        Image.fromarray(np.asarray(255.0 * (1.0 - scaled), dtype=np.uint8)).save(
            trial_dir / "depth_visualization.png"
        )
        timing_ms["input_artifact_write"] = (
            time.perf_counter() - phase_started
        ) * 1000.0

        scene = env.kernel.scene
        metadata = scene.camera_metadata()[CAMERA_NAME]
        intrinsics = render_intrinsics_from_fovy(
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            fovy_deg=float(metadata["fovy_deg"]),
        )
        world_from_camera = optical_camera_pose_world(scene)
        cube_center = scene.body_pose(env.kernel.config.object_body_name)[:3, 3]
        u, v, projected_depth = project_world_point(
            cube_center,
            world_from_camera=world_from_camera,
            intrinsics=intrinsics,
        )
        click_uv = (u, v)
        if depth[v, u] <= 0.0 or abs(float(depth[v, u]) - projected_depth) > 0.080:
            raise RuntimeError(
                "Projected cube-center click is not on the visible cube surface: "
                f"rendered_depth={float(depth[v, u]):.6f}, center_depth={projected_depth:.6f}."
            )
        world_from_base = scene.body_pose(
            f"{env.kernel.config.controlled_side}_arm_base_link"
        )
        base_from_camera = np.linalg.inv(world_from_base) @ world_from_camera
        parameters = {
            "input_source": "local_path",
            "local_rgb_path": str(trial_dir / "initial_rgb.png"),
            "local_depth_path": str(trial_dir / "aligned_depth_m.npy"),
            "target_pixel_uv": [u, v],
            "camera_intrinsics": {
                "width": intrinsics.width,
                "height": intrinsics.height,
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy,
            },
            "camera_frame_id": "head_camera_frame_optical",
            "base_frame_id": f"{env.kernel.config.controlled_side}_arm_base_link",
            "T_base_from_camera": base_from_camera.reshape(-1).tolist(),
            "max_candidates": config.max_candidates,
        }
        failure_stage = "perception"
        profile = provider.registration.profile
        prompt_pixels = [click_uv]
        successful_prompt_pixels: list[tuple[int, int]] = []
        prompt_parameters: list[dict[str, Any]] = []
        prompt_results: list[dict[str, Any]] = []
        prompt_records: list[dict[str, Any]] = []
        candidate_batches: list[list[Mapping[str, Any]]] = []
        provider_started = time.perf_counter()
        prompt_index = 0
        while prompt_index < len(prompt_pixels):
            prompt_uv = prompt_pixels[prompt_index]
            current_parameters = {
                **parameters,
                "target_pixel_uv": [int(prompt_uv[0]), int(prompt_uv[1])],
            }
            validate_point_grasp_parameters_v1(
                current_parameters,
                local_path_enabled=True,
                max_candidates=5,
            )
            try:
                prepared = provider.plan_request(
                    SimpleNamespace(
                        parameters=current_parameters,
                        input_artifact_bindings=(),
                    ),
                    artifact_store=SimpleNamespace(),
                )
                result = prepared.execution_result
                validate_point_grasp_result_v1(
                    result,
                    backend_id=profile.backend_id,
                    model_profile_id=profile.model_profile_id,
                    model_profile_hash=profile.model_profile_hash,
                    max_candidates=profile.max_candidates,
                )
                validate_point_grasp_result_for_parameters_v1(
                    current_parameters,
                    result,
                )
                annotated_name = (
                    "skill_annotated.jpg"
                    if prompt_index == 0
                    else f"skill_annotated_point_{prompt_index:02d}.jpg"
                )
                (trial_dir / annotated_name).write_bytes(prepared.annotated_jpeg)
                successful_prompt_pixels.append(prompt_uv)
                prompt_parameters.append(current_parameters)
                prompt_results.append(result)
                candidate_batches.append(list(result["candidates"]))
                prompt_records.append(
                    {
                        "prompt_index": prompt_index,
                        "pixel_uv": list(prompt_uv),
                        "status": "ok",
                        "candidate_count": int(result["candidate_count"]),
                    }
                )
                if prompt_index == 0 and config.target_point_count > 1:
                    target_mask = getattr(provider.planner, "last_target_mask", None)
                    if target_mask is None:
                        raise RuntimeError(
                            "Point-grasp planner did not expose the first SAM2 mask."
                        )
                    prompt_pixels = sample_mask_prompt_pixels(
                        target_mask,
                        depth,
                        initial_uv=click_uv,
                        count=config.target_point_count,
                    )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                prompt_records.append(
                    {
                        "prompt_index": prompt_index,
                        "pixel_uv": list(prompt_uv),
                        "status": "failed",
                        "reason": str(exc),
                    }
                )
                if prompt_index == 0:
                    raise
            finally:
                planner_timing = getattr(
                    provider.planner,
                    "last_plan_timing_ms",
                    {},
                )
                if isinstance(planner_timing, Mapping):
                    for name, value in planner_timing.items():
                        if isinstance(value, (int, float)) and math.isfinite(
                            float(value)
                        ):
                            timing_ms[str(name)] = float(
                                timing_ms.get(str(name), 0.0)
                            ) + float(value)
            prompt_index += 1
        timing_ms["point_grasp_provider_total"] = (
            time.perf_counter() - provider_started
        ) * 1000.0
        if not prompt_results:
            raise RuntimeError("No point prompt returned a valid grasp result.")
        result = prompt_results[0]
        combined_candidates = merge_point_prompt_candidates(
            candidate_batches,
            successful_prompt_pixels,
            maximum=(
                config.max_candidates
                if config.target_point_count == 1
                else MULTIPOINT_MAX_COMBINED_CANDIDATES
            ),
        )
        candidate_count = len(combined_candidates)
        bbox = tuple(int(value) for value in result["target_mask_bbox_xyxy"])
        phase_started = time.perf_counter()
        try:
            (
                selected_ordinal,
                selected_pregrasp_offset_m,
                selected_wrist_symmetry_deg,
                preflight,
            ) = select_reachable_candidate(
                env=env,
                candidates=combined_candidates,
                provider_selected_ordinal=0,
                seed=seed,
                config=config,
            )
            candidate_preflight = [value.to_dict() for value in preflight]
        finally:
            timing_ms["candidate_motion_preflight"] = (
                time.perf_counter() - phase_started
            ) * 1000.0
        if config.candidate_selection_policy == "reachability_aware" and not any(
            value.executable for value in preflight
        ):
            failure_stage = "motion_planning_preflight"
            best = min(preflight, key=_candidate_preflight_rank)
            raise RuntimeError(f"no_reachable_candidate:{best.reason}")
        candidate = combined_candidates[selected_ordinal]
        selected_score = float(candidate["score"])
        selected_model_score = float(candidate["model_score"])
        selected_width = float(candidate["gripper_width_m"])
        base_from_tcp = apply_parallel_jaw_wrist_symmetry(
            np.asarray(candidate["T_base_from_tcp"], dtype=float).reshape(4, 4),
            selected_wrist_symmetry_deg,
        )
        camera_from_tcp = apply_parallel_jaw_wrist_symmetry(
            np.asarray(candidate["T_camera_from_tcp"], dtype=float).reshape(4, 4),
            selected_wrist_symmetry_deg,
        )
        overlay = draw_grasp_overlay(
            initial_rgb,
            camera_from_tcp=camera_from_tcp,
            gripper_width_m=selected_width,
            intrinsics=parameters["camera_intrinsics"],
            click_uv=tuple(int(value) for value in candidate["_prompt_uv"]),
            bbox_xyxy=bbox,
        )
        Image.fromarray(overlay).save(trial_dir / "grasp_pose_overlay.png")
        keyframes.append(("Skill selected grasp", overlay))
        _write_json(
            trial_dir / "skill_result.json",
            {
                "schema_version": RESULT_SCHEMA_V1 + ".skill_input_output",
                "parameters": parameters,
                "result": result,
                "point_prompt_policy": {
                    "requested_count": config.target_point_count,
                    "sampled_pixels_uv": [list(value) for value in prompt_pixels],
                    "records": prompt_records,
                    "successful_parameters": prompt_parameters,
                    "combined_candidate_limit": (
                        config.max_candidates
                        if config.target_point_count == 1
                        else MULTIPOINT_MAX_COMBINED_CANDIDATES
                    ),
                    "combined_candidates": combined_candidates,
                },
                "execution_selection": {
                    "policy": config.candidate_selection_policy,
                    "provider_selected_ordinal": 0,
                    "executed_ordinal": selected_ordinal,
                    "executed_prompt_index": int(candidate["_prompt_index"]),
                    "executed_prompt_uv": list(candidate["_prompt_uv"]),
                    "pregrasp_offset_m": selected_pregrasp_offset_m,
                    "wrist_symmetry_deg": selected_wrist_symmetry_deg,
                    "preflight": candidate_preflight,
                },
            },
        )
        timing_ms["skill_validation_and_visualization"] = (
            time.perf_counter() - phase_started
        ) * 1000.0

        # Candidate preflight advances the simulator only as a read-only motion
        # feasibility check.  Recreate the deterministic initial state before
        # the one execution that counts toward the trial outcome.
        phase_started = time.perf_counter()
        try:
            _observation, execution_reset_info = env.reset(seed=seed)
        finally:
            timing_ms["post_preflight_execution_reset"] = (
                time.perf_counter() - phase_started
            ) * 1000.0
        trace = _ExecutionTrace(info=dict(execution_reset_info))
        resting_z = float(execution_reset_info["resting_center_z_m"])
        trace.max_cube_z_m = float(execution_reset_info["cube_z_m"])

        failure_stage = "motion_planning_pregrasp"
        assert selected_pregrasp_offset_m is not None
        pregrasp = build_pregrasp_pose(base_from_tcp, selected_pregrasp_offset_m)
        phase_started = time.perf_counter()
        try:
            reached, reason = _drive_pose(
                env=env,
                desired_pose=pregrasp,
                trace=trace,
                renderer=renderer,
                video=video,
                maximum_steps=config.phase_step_limit,
                position_tolerance_m=config.pose_position_tolerance_m,
                orientation_tolerance_rad=config.pose_orientation_tolerance_rad,
            )
        finally:
            timing_ms["pregrasp_motion"] = (
                time.perf_counter() - phase_started
            ) * 1000.0
        if not reached:
            raise RuntimeError(reason)
        pregrasp_frame = renderer.render(CAMERA_NAME)
        keyframes.append(("pregrasp reached", pregrasp_frame))

        failure_stage = "motion_planning_grasp"
        phase_started = time.perf_counter()
        try:
            reached, reason = _drive_pose(
                env=env,
                desired_pose=base_from_tcp,
                trace=trace,
                renderer=renderer,
                video=video,
                maximum_steps=config.phase_step_limit,
                position_tolerance_m=config.pose_position_tolerance_m,
                orientation_tolerance_rad=config.pose_orientation_tolerance_rad,
            )
        finally:
            timing_ms["grasp_motion"] = (time.perf_counter() - phase_started) * 1000.0
        if not reached:
            raise RuntimeError(reason)
        grasp_frame = renderer.render(CAMERA_NAME)
        keyframes.append(("grasp goal reached", grasp_frame))

        failure_stage = "gripper_contact"
        close_frame = grasp_frame
        phase_started = time.perf_counter()
        try:
            consecutive_bilateral_steps = 0
            for _ in range(config.close_step_limit):
                action = np.zeros(7, dtype=np.float32)
                if not bool(
                    trace.info.get("bilateral_finger_cube_contact", False)
                ) and float(trace.info.get("right_gripper_target_close", 0.0)) < (
                    env.kernel.config.right_gripper_max_close - 1e-6
                ):
                    action[6] = float(config.close_action_scale)
                terminated, truncated, info, close_frame = _step_action(
                    env=env,
                    action=action,
                    trace=trace,
                    renderer=renderer,
                    video=video,
                )
                if terminated or truncated:
                    reason = str(
                        info.get("termination_reason")
                        or "episode_terminated_during_close"
                    )
                    raise RuntimeError(reason)
                if bool(info.get("bilateral_finger_cube_contact", False)):
                    consecutive_bilateral_steps += 1
                else:
                    consecutive_bilateral_steps = 0
                if consecutive_bilateral_steps >= config.contact_stability_steps:
                    break
        finally:
            timing_ms["gripper_close"] = (time.perf_counter() - phase_started) * 1000.0
        if consecutive_bilateral_steps < config.contact_stability_steps:
            reason = "stable_bilateral_finger_contact_not_acquired"
            raise RuntimeError(reason)
        keyframes.append(("bilateral contact", close_frame.copy()))

        failure_stage = "lift_execution"
        lift = base_from_tcp.copy()
        world_lift = np.asarray((0.0, 0.0, config.lift_offset_m), dtype=float)
        lift[:3, 3] += world_from_base[:3, :3].T @ world_lift
        phase_started = time.perf_counter()
        try:
            reached, reason = _drive_pose(
                env=env,
                desired_pose=lift,
                trace=trace,
                renderer=renderer,
                video=video,
                maximum_steps=config.lift_step_limit,
                position_tolerance_m=config.pose_position_tolerance_m,
                orientation_tolerance_rad=config.pose_orientation_tolerance_rad,
                translation_action_scale=config.lift_translation_action_scale,
            )
        finally:
            timing_ms["lift_motion"] = (time.perf_counter() - phase_started) * 1000.0
        success = trace.termination_reason == BILATERAL_CONTACT_LIFT_SUCCESS_REASON
        if not success and reached:
            for _ in range(5):
                terminated, truncated, info, _frame = _step_action(
                    env=env,
                    action=np.zeros(7, dtype=np.float32),
                    trace=trace,
                    renderer=renderer,
                    video=video,
                )
                if terminated or truncated:
                    break
            success = trace.termination_reason == BILATERAL_CONTACT_LIFT_SUCCESS_REASON
        if not success:
            raise RuntimeError(
                reason if not reached else "lift_success_predicate_not_satisfied"
            )
        failure_stage = None
        reason = BILATERAL_CONTACT_LIFT_SUCCESS_REASON
        keyframes.append(("successful lift", renderer.render(CAMERA_NAME)))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        success = False
        reason = str(exc)
        try:
            keyframes.append(
                (f"failure: {failure_stage}", renderer.render(CAMERA_NAME))
            )
        except Exception:  # noqa: BLE001, S110 - failure rendering is best effort
            pass
    finally:
        video.close()
        save_montage(trial_dir / "execution_montage.png", keyframes)
        timing_ms["trial_total"] = (time.perf_counter() - trial_started) * 1000.0
        timing_ms["ik_pose_solve_samples"] = list(trace.ik_solve_samples_ms)
        timing_ms["ik_pose_frame_total_samples"] = list(trace.ik_frame_total_samples_ms)
        timing_ms["ik_pose_solve_summary"] = _timing_summary_ms(
            trace.ik_solve_samples_ms
        )
        timing_ms["ik_pose_frame_total_summary"] = _timing_summary_ms(
            trace.ik_frame_total_samples_ms
        )

    max_lift = 0.0
    if math.isfinite(resting_z) and math.isfinite(trace.max_cube_z_m):
        max_lift = max(0.0, trace.max_cube_z_m - resting_z)
    trial_result = SimPointGraspTrialResult(
        seed=seed,
        success=success,
        failure_stage=failure_stage,
        reason=reason,
        termination_reason=trace.termination_reason,
        selected_score=selected_score,
        selected_model_score=selected_model_score,
        selected_width_m=selected_width,
        candidate_count=candidate_count,
        click_uv=click_uv,
        mask_bbox_xyxy=bbox,
        max_lift_m=max_lift,
        bilateral_contact_observed=trace.bilateral_contact_observed,
        motion_steps=trace.step_count,
        safety_intervention_steps=trace.safety_intervention_steps,
        ik_rejection_steps=trace.ik_rejection_steps,
        trial_dir=str(trial_dir),
        timing_ms=timing_ms,
        ik_solver=trace.ik_solver,
        max_pose_ik_limit_violation_rad=trace.max_pose_ik_limit_violation_rad,
        pose_ik_limit_violation_side=trace.pose_ik_limit_violation_side,
        pose_ik_limit_violation_joint=trace.pose_ik_limit_violation_joint,
        pose_ik_sync_failure_reason=trace.pose_ik_sync_failure_reason,
        selected_ordinal=selected_ordinal,
        selected_pregrasp_offset_m=selected_pregrasp_offset_m,
        selected_wrist_symmetry_deg=selected_wrist_symmetry_deg,
        candidate_selection_policy=config.candidate_selection_policy,
        candidate_preflight=candidate_preflight,
    )
    _write_json(trial_dir / "trial_result.json", trial_result.to_dict())
    return trial_result


def evaluate_sim_point_grasps(config: SimPointGraspEvaluationConfig) -> dict[str, Any]:
    """Run deterministic-seed physical trials and return the aggregate report."""

    if config.output_dir.exists():
        raise ValueError(f"Evaluation output already exists: {config.output_dir}")
    config.output_dir.mkdir(parents=True)
    # Pregrasp, grasp, close, lift, and the final predicate hold all consume
    # the same episode clock.  The historical fixed 150-step task budget was
    # smaller than the evaluator's own legal worst-case path, causing valid
    # lift attempts to truncate before their phase limit.
    execution_step_budget = (
        2 * int(config.phase_step_limit)
        + int(config.close_step_limit)
        + int(config.lift_step_limit)
        + 5
    )
    task_config = build_sim_point_grasp_task_config(
        object_mass_kg=config.object_mass_kg,
        gripper_force_profile=config.gripper_force_profile,
        cube_center_xy_m=config.cube_center_xy_m,
        cube_spawn_xy_jitter_m=config.cube_spawn_xy_jitter_m,
        cube_spawn_yaw_jitter_rad=config.cube_spawn_yaw_jitter_rad,
        max_episode_steps=execution_step_budget,
    )
    scene_contract = task_config.base_contract()
    scene_urdf_path = resolve_path(scene_contract.scene.urdf_path)
    _write_json(
        config.output_dir / "run_config.json",
        {
            "schema_version": RESULT_SCHEMA_V1 + ".config",
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
            "task_config": task_config.to_metadata(),
        },
    )
    setup_timing_ms: dict[str, float] = {}
    setup_started = time.perf_counter()
    env = FastCubeLiftEnv(
        task_config,
        include_training_reward_state=True,
        use_model_cache=True,
    )
    renderer = MujocoCameraRenderer(env.kernel.scene, width=640, height=480)
    setup_timing_ms["mujoco_environment_and_renderer_startup"] = (
        time.perf_counter() - setup_started
    ) * 1000.0
    localizer = provider = None
    started_at = time.time()
    results: list[SimPointGraspTrialResult] = []
    try:
        provider_build_started = time.perf_counter()
        localizer, provider = _make_provider(config)
        setup_timing_ms["model_provider_construction"] = (
            time.perf_counter() - provider_build_started
        ) * 1000.0
        sam2_startup_started = time.perf_counter()
        localizer.start()
        setup_timing_ms["sam2_startup"] = (
            time.perf_counter() - sam2_startup_started
        ) * 1000.0
        graspnet_startup_started = time.perf_counter()
        provider.start()
        setup_timing_ms["graspnet_startup"] = (
            time.perf_counter() - graspnet_startup_started
        ) * 1000.0
        for offset in range(config.trials):
            result = _trial(
                env=env,
                renderer=renderer,
                provider=provider,
                config=config,
                seed=config.seed_start + offset,
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "trial_completed": len(results),
                        "trial_total": config.trials,
                        "seed": result.seed,
                        "success": result.success,
                        "failure_stage": result.failure_stage,
                        "reason": result.reason,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        try:
            if provider is not None:
                provider.close()
        finally:
            try:
                if localizer is not None:
                    localizer.close()
            finally:
                renderer.close()
                env.close()

    success_count = sum(int(item.success) for item in results)
    save_summary_sheet(config.output_dir / "summary_montage.jpg", results)
    failure_counts: dict[str, int] = {}
    for result in results:
        if result.failure_stage is not None:
            failure_counts[result.failure_stage] = (
                failure_counts.get(result.failure_stage, 0) + 1
            )
    report = {
        "schema_version": RESULT_SCHEMA_V1,
        "attempted_trials": len(results),
        "success_count": success_count,
        "success_rate": success_count / max(len(results), 1),
        "failure_count_by_stage": dict(sorted(failure_counts.items())),
        "elapsed_sec": time.time() - started_at,
        "startup_timing_ms": setup_timing_ms,
        "timing_summary_ms": _aggregate_trial_timings(results),
        "warm_model_inference_excluding_first_trial_ms": {
            name: _timing_summary_ms(
                [
                    float(item.timing_ms[name])
                    for item in results[1:]
                    if isinstance(item.timing_ms.get(name), (int, float))
                    and math.isfinite(float(item.timing_ms[name]))
                ]
            )
            for name in ("sam2_inference", "graspnet_inference")
        },
        "success_definition": {
            "termination_reason": BILATERAL_CONTACT_LIFT_SUCCESS_REASON,
            "lift_delta_m": task_config.lift_delta_m,
            "success_hold_physics_ticks": task_config.success_hold_physics_ticks,
            "bilateral_contact_required": True,
            "table_contact_forbidden_at_success": True,
        },
        "camera": {
            "name": CAMERA_NAME,
            "frame": "head_camera_frame optical: +x right, +y down, +z forward",
            "rgb_depth_aligned": True,
            "depth_unit": "meter",
        },
        "scene": {
            "contract_path": str(task_config.base_contract_path),
            "contract_sha256": task_config.base_contract_sha256,
            "urdf_path": str(scene_contract.scene.urdf_path),
            "urdf_sha256": hashlib.sha256(scene_urdf_path.read_bytes()).hexdigest(),
            "object_body_name": task_config.object_body_name,
            "object_shape": "box",
            "object_size_m": list(task_config.object_size_m),
            "object_mass_kg": scene_contract.scene.objects[0].mass_kg,
            "object_inertia_diagonal_kg_m2": list(
                scene_contract.scene.objects[0].inertia_diagonal_kg_m2
            ),
            "object_center_xy_m": list(task_config.cube_center_xy_m),
            "spawn_xy_jitter_m": list(task_config.cube_spawn_xy_jitter_m),
            "spawn_yaw_jitter_rad": task_config.cube_spawn_yaw_jitter_rad,
            "collision_safety": {
                "minimum_clearance_m": (
                    scene_contract.scene.collision.safety_minimum_clearance_m
                ),
                "hard_penetration_m": (
                    scene_contract.scene.collision.safety_hard_penetration_m
                ),
                "collision_checking_enabled": (scene_contract.scene.collision.enabled),
            },
            "gripper_actuation_profile_name": (
                task_config.gripper_actuation_profile_name
            ),
            "gripper_actuation_profile_sha256": (
                task_config.gripper_actuation_profile_sha256
            ),
            "right_gripper_name": task_config.right_gripper_name,
            "right_tool_profile_path": task_config.right_tool_profile_path,
            "right_tool_profile_sha256": EVALUATION_RIGHT_TOOL_PROFILE_SHA256,
        },
        "skill": {
            "skill_id": "perception.plan_point_grasp.v1",
            "backend": "graspnet",
            "selected_candidate_policy": config.candidate_selection_policy,
            "target_point_count": config.target_point_count,
            "target_point_sampling": (
                "projected_object_center_only"
                if config.target_point_count == 1
                else "projected_object_center_plus_sam2_mask_eroded_farthest_points"
            ),
            "combined_candidate_limit": (
                config.max_candidates
                if config.target_point_count == 1
                else MULTIPOINT_MAX_COMBINED_CANDIDATES
            ),
            "candidate_preflight_scope": (
                "pregrasp_and_grasp_ik_full_scene_collision_only;"
                "no_close_contact_or_lift_outcome"
            ),
            "parallel_jaw_wrist_symmetry_deg": [0],
            "tcp_orientation_selection": PARALLEL_JAW_TCP_UPWARD_SELECTION_V1,
            "adaptive_pregrasp_offsets_m": [
                config.pregrasp_offset_m,
                *config.adaptive_pregrasp_offsets_m,
            ],
            "maximum_visible_collision_points": DMTAC_M_MAX_VISIBLE_COLLISION_POINTS,
            "contact_stability_policy_steps": config.contact_stability_steps,
            "close_action_scale": config.close_action_scale,
            "close_step_limit": config.close_step_limit,
            "gripper_force_profile": config.gripper_force_profile,
            "phase_step_limit": config.phase_step_limit,
            "lift_translation_action_scale": config.lift_translation_action_scale,
        },
        "successful_trajectory_videos": [
            str(Path(item.trial_dir) / "execution.mp4")
            for item in results
            if item.success and config.save_video
        ],
        "trials": [item.to_dict() for item in results],
    }
    _write_json(config.output_dir / "summary.json", report)
    return report


__all__ = [
    "CAMERA_NAME",
    "SimPointGraspEvaluationConfig",
    "SimPointGraspTrialResult",
    "apply_parallel_jaw_wrist_symmetry",
    "bounded_pose_action",
    "build_pregrasp_pose",
    "build_sim_point_grasp_task_config",
    "draw_grasp_overlay",
    "evaluate_sim_point_grasps",
    "merge_point_prompt_candidates",
    "optical_camera_pose_world",
    "project_world_point",
    "sample_mask_prompt_pixels",
    "save_montage",
    "save_summary_sheet",
    "select_reachable_candidate",
]
