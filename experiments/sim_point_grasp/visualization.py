"""Visualization-only SAM2 + GraspNet grasp projection in MuJoCo cameras."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from agent_gateway.profiles.real.point_grasp_deployment import (
    build_real_point_grasp_registration,
)
from agent_gateway.profiles.real.point_grasp_gateway import PointGraspProvider
from agent_gateway.skills.perception.point_grasp_schemas import (
    validate_point_grasp_parameters_v1,
    validate_point_grasp_result_for_parameters_v1,
    validate_point_grasp_result_v1,
)
from deployments.perception.dmtac_grasp_overlay import (
    DmtacGraspOverlayCandidate,
    render_dmtac_grasp_overlays,
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
    PARALLEL_JAW_TCP_UPWARD_SELECTION_V1,
)
from experiments.sim_point_grasp.evaluation import (
    build_sim_point_grasp_task_config,
    optical_camera_pose_world,
    project_world_point,
    save_montage,
)
from project_paths import resolve_path
from sim.mujoco.pointcloud import render_intrinsics_from_fovy
from sim.mujoco.scene import MujocoCameraRenderer
from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv

VISUALIZATION_CAMERA_NAMES = ("front", "left_wrist", "right_wrist")
VISUALIZATION_SCHEMA_V1 = "dexteleop.experiments.sim_point_grasp_visualization.v1"


@dataclass(frozen=True)
class SimPointGraspVisualizationConfig:
    output_dir: Path
    sam2_asset_bundle: Path
    graspnet_asset_bundle: Path
    seed: int = 0
    device: str = "cuda"
    max_candidates: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_dir", Path(self.output_dir).expanduser().resolve()
        )
        for field_name in ("sam2_asset_bundle", "graspnet_asset_bundle"):
            path = Path(getattr(self, field_name)).expanduser().resolve(strict=True)
            if not path.is_dir():
                raise ValueError(f"{field_name} must be an existing directory.")
            object.__setattr__(self, field_name, path)
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise TypeError("seed must be an integer.")
        object.__setattr__(self, "seed", int(self.seed))
        if not str(self.device).startswith("cuda"):
            raise ValueError("The reviewed GraspNet profile requires a CUDA device.")
        if not 1 <= int(self.max_candidates) <= 5:
            raise ValueError("max_candidates must be in [1, 5].")
        object.__setattr__(self, "max_candidates", int(self.max_candidates))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _make_provider(
    config: SimPointGraspVisualizationConfig,
) -> tuple[Sam2GroundingDinoLocalizer, PointGraspProvider]:
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
        runtime_session_id="sim-point-grasp-visualization",
        control_domain_id="offline-mujoco-visualization-only",
        target_binding_id="simulated-front-rgbd-three-camera-overlay",
        station_manifest_hash="not-applicable-offline-simulation",
        execution_manifest_hash="sim-point-grasp-visualization-v1",
    )
    return localizer, PointGraspProvider(
        planner=planner,
        registration=registration,
        allowed_local_image_roots=(config.output_dir,),
    )


def _state_digest(scene: Any) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray([scene.data.time], dtype=np.float64).tobytes())
    for name in ("qpos", "qvel", "ctrl"):
        digest.update(np.asarray(getattr(scene.data, name), dtype=np.float64).tobytes())
    return digest.hexdigest()


def render_candidates_in_mujoco_views(
    *,
    scene: Any,
    renderer: MujocoCameraRenderer,
    base_from_tcp_candidates: Sequence[Mapping[str, Any]],
    world_from_base: Any,
    selected_ordinal: int = 0,
    camera_names: Sequence[str] = VISUALIZATION_CAMERA_NAMES,
    front_target_pixel_uv: tuple[int, int] | None = None,
    front_target_bbox_xyxy: tuple[int, int, int, int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Render grasp overlays without stepping or otherwise mutating the scene."""

    requested = tuple(str(name) for name in camera_names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("camera_names must be non-empty and unique.")
    missing = sorted(set(requested) - set(scene.camera_names))
    if missing:
        raise ValueError(f"MuJoCo scene is missing cameras: {missing!r}.")
    base_pose = np.asarray(world_from_base, dtype=np.float64).reshape(4, 4)
    candidates = tuple(
        DmtacGraspOverlayCandidate.from_pose(
            ordinal=int(candidate["ordinal"]),
            score=float(candidate["score"]),
            gripper_width_m=float(candidate["gripper_width_m"]),
            reference_from_tcp=candidate["T_base_from_tcp"],
            selected=int(candidate["ordinal"]) == int(selected_ordinal),
        )
        for candidate in base_from_tcp_candidates
    )
    metadata_by_name = scene.camera_metadata()
    images: dict[str, np.ndarray] = {}
    records: dict[str, dict[str, Any]] = {}
    for camera_name in requested:
        metadata = metadata_by_name[camera_name]
        intrinsics = render_intrinsics_from_fovy(
            width=renderer.width,
            height=renderer.height,
            fovy_deg=float(metadata["fovy_deg"]),
        )
        world_from_camera = optical_camera_pose_world(scene, camera_name)
        base_from_camera = np.linalg.inv(base_pose) @ world_from_camera
        raw = renderer.render(camera_name)
        overlay, projections = render_dmtac_grasp_overlays(
            raw,
            candidates=candidates,
            reference_from_camera=base_from_camera,
            intrinsics=intrinsics.to_dict(),
            target_pixel_uv=(front_target_pixel_uv if camera_name == "front" else None),
            target_bbox_xyxy=(
                front_target_bbox_xyxy if camera_name == "front" else None
            ),
        )
        images[camera_name] = np.asarray(overlay)
        records[camera_name] = {
            "camera_name": camera_name,
            "camera_frame_semantics": "urdf_optical_frame_x_right_y_down_z_forward",
            "T_base_from_camera": [
                float(item) for item in base_from_camera.reshape(-1)
            ],
            "camera_intrinsics": intrinsics.to_dict(),
            "depth_occlusion_tested": False,
            "candidates": [projection.to_dict() for projection in projections],
        }
    return images, records


def visualize_sim_point_grasps(
    config: SimPointGraspVisualizationConfig,
) -> dict[str, Any]:
    """Run RGB-D inference once, then overlay proposals in three cameras."""

    if config.output_dir.exists():
        raise ValueError(f"Visualization output already exists: {config.output_dir}")
    config.output_dir.mkdir(parents=True)
    task_config = build_sim_point_grasp_task_config(
        cube_spawn_xy_jitter_m=(0.025, 0.080),
        cube_spawn_yaw_jitter_rad=math.pi / 4.0,
    )
    scene_contract = task_config.base_contract()
    # Resolve now so a missing shared asset fails before any model lifecycle starts.
    resolve_path(scene_contract.scene.urdf_path)
    env = FastCubeLiftEnv(
        task_config,
        include_training_reward_state=False,
        use_model_cache=True,
    )
    renderer = MujocoCameraRenderer(env.kernel.scene, width=640, height=480)
    localizer = provider = None
    try:
        _observation, reset_info = env.reset(seed=config.seed)
        del _observation
        scene = env.kernel.scene
        if not set(VISUALIZATION_CAMERA_NAMES) <= set(scene.camera_names):
            raise RuntimeError(
                "Visualization scene does not expose front and both wrist cameras."
            )
        state_before = _state_digest(scene)
        raw_images = renderer.render_all(VISUALIZATION_CAMERA_NAMES)
        for camera_name, pixels in raw_images.items():
            Image.fromarray(pixels).save(config.output_dir / f"{camera_name}_rgb.png")
        depth = np.ascontiguousarray(renderer.render_depth("front"), dtype=np.float32)
        np.save(config.output_dir / "front_depth_m.npy", depth, allow_pickle=False)

        metadata = scene.camera_metadata()["front"]
        front_intrinsics = render_intrinsics_from_fovy(
            width=renderer.width,
            height=renderer.height,
            fovy_deg=float(metadata["fovy_deg"]),
        )
        world_from_front = optical_camera_pose_world(scene, "front")
        cube_center = scene.body_pose(task_config.object_body_name)[:3, 3]
        u, v, projected_depth = project_world_point(
            cube_center,
            world_from_camera=world_from_front,
            intrinsics=front_intrinsics,
        )
        if depth[v, u] <= 0.0 or abs(float(depth[v, u]) - projected_depth) > 0.080:
            raise RuntimeError(
                "Projected cube-center click is not on the rendered object surface."
            )
        side = task_config.controlled_side
        world_from_base = scene.body_pose(f"{side}_arm_base_link")
        base_from_front = np.linalg.inv(world_from_base) @ world_from_front
        parameters = {
            "input_source": "local_path",
            "local_rgb_path": str(config.output_dir / "front_rgb.png"),
            "local_depth_path": str(config.output_dir / "front_depth_m.npy"),
            "target_pixel_uv": [u, v],
            "camera_intrinsics": {
                key: value
                for key, value in front_intrinsics.to_dict().items()
                if key in {"width", "height", "fx", "fy", "cx", "cy"}
            },
            "camera_frame_id": "head_camera_frame_optical",
            "base_frame_id": f"{side}_arm_base_link",
            "T_base_from_camera": base_from_front.reshape(-1).tolist(),
            "max_candidates": config.max_candidates,
        }
        validate_point_grasp_parameters_v1(
            parameters,
            local_path_enabled=True,
            max_candidates=5,
        )
        localizer, provider = _make_provider(config)
        localizer.start()
        provider.start()
        prepared = provider.plan_request(
            SimpleNamespace(parameters=parameters, input_artifact_bindings=()),
            artifact_store=SimpleNamespace(),
        )
        result = prepared.execution_result
        profile = provider.registration.profile
        validate_point_grasp_result_v1(
            result,
            backend_id=profile.backend_id,
            model_profile_id=profile.model_profile_id,
            model_profile_hash=profile.model_profile_hash,
            max_candidates=profile.max_candidates,
        )
        validate_point_grasp_result_for_parameters_v1(parameters, result)
        (config.output_dir / "front_skill_annotation.jpg").write_bytes(
            prepared.annotated_jpeg
        )
        _write_json(config.output_dir / "skill_result.json", result)

        overlay_images, camera_records = render_candidates_in_mujoco_views(
            scene=scene,
            renderer=renderer,
            base_from_tcp_candidates=result["candidates"],
            world_from_base=world_from_base,
            selected_ordinal=int(result["selected_ordinal"]),
            front_target_pixel_uv=(u, v),
            front_target_bbox_xyxy=tuple(result["target_mask_bbox_xyxy"]),
        )
        montage_frames = []
        for camera_name in VISUALIZATION_CAMERA_NAMES:
            pixels = overlay_images[camera_name]
            Image.fromarray(pixels).save(
                config.output_dir / f"{camera_name}_grasp_overlay.png"
            )
            montage_frames.append((f"{camera_name} grasp overlay", pixels))
        save_montage(config.output_dir / "grasp_multiview_montage.png", montage_frames)

        state_after = _state_digest(scene)
        state_unchanged = state_before == state_after
        if not state_unchanged:
            raise RuntimeError("Visualization unexpectedly changed MuJoCo state.")
        report = {
            "schema_version": VISUALIZATION_SCHEMA_V1,
            "seed": config.seed,
            "camera_names": list(VISUALIZATION_CAMERA_NAMES),
            "generation_camera": "front",
            "candidate_count": int(result["candidate_count"]),
            "selected_ordinal": int(result["selected_ordinal"]),
            "upward_tcp_selection_policy": PARALLEL_JAW_TCP_UPWARD_SELECTION_V1,
            "selected_pose_policy": "provider_rank_after_upward_tcp_symmetry_selection",
            "overlay_semantics": "simple_3d_wireframe_projection_without_depth_occlusion",
            "motion_execution_attempted": False,
            "ik_attempted": False,
            "gripper_actuation_attempted": False,
            "simulation_state_unchanged": state_unchanged,
            "simulation_state_digest_before": state_before,
            "simulation_state_digest_after": state_after,
            "simulation_time_after_reset_s": float(scene.data.time),
            "reset_info": {
                "cube_z_m": float(reset_info["cube_z_m"]),
                "resting_center_z_m": float(reset_info["resting_center_z_m"]),
            },
            "cameras": camera_records,
        }
        _write_json(config.output_dir / "visualization_report.json", report)
        return report
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


__all__ = [
    "VISUALIZATION_CAMERA_NAMES",
    "VISUALIZATION_SCHEMA_V1",
    "SimPointGraspVisualizationConfig",
    "render_candidates_in_mujoco_views",
    "visualize_sim_point_grasps",
]
