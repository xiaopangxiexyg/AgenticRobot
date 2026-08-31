from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agent_gateway.profiles.real.perception_application import (
    assemble_camera_only_real_perception_composition,
)
from agent_gateway.profiles.real.realsense_perception import CameraOnlyRealPerceptionResources
from agent_gateway.profiles.real.realsense_perception import RealPerceptionSourceBinding
from agent_gateway.profiles.real.point_grasp_deployment import (
    REAL_POINT_GRASP_DEPTH_SLOT_ID_V1,
    REAL_POINT_GRASP_RGB_SLOT_ID_V1,
    build_real_point_grasp_registration,
)
from agent_gateway.profiles.real.point_grasp_gateway import PointGraspProvider
from agent_gateway.transport.client_bindings import (
    compile_agent_gateway_unix_client_bindings,
)
from dataset_contracts.perception_artifacts import (
    LOCALIZATION_INPUT_IMAGE_COMPRESSION_V1,
    LOCALIZATION_INPUT_PNG_ENCODING_V1,
    LOCALIZATION_INPUT_PNG_MEDIA_TYPE_V1,
    POINT_GRASP_DEPTH_COMPRESSION_V1,
    POINT_GRASP_DEPTH_ENCODING_V1,
    POINT_GRASP_DEPTH_MEDIA_TYPE_V1,
    POINT_GRASP_RGB_SCHEMA_ID_V1,
)
from deployments.perception.sam2_vgn_point_grasp import (
    DEFAULT_DMTAC_M_URDF,
    DEFAULT_DMTAC_M_URDF_SHA256,
    DMTAC_M_MAX_VISIBLE_COLLISION_POINTS,
    PointGraspCandidate,
    PointGraspResult,
    Sam2VgnPointGraspPlanner,
    VGN_GIT_URL,
    VGN_REVIEWED_REVISION,
    VgnPointGraspModelSpec,
)
from deployments.perception.sam2_graspnet_point_grasp import (
    GRASPNET_GIT_URL,
    GRASPNET_LICENSE_ACKNOWLEDGEMENT_V1,
    GRASPNET_REVIEWED_REVISION,
    GraspNetPointGraspModelSpec,
    Sam2GraspNetPointGraspPlanner,
)
from sim.contracts.scene.spec import DEFAULT_RM65_PHYSICS_URDF


class _FakeModelSpec:
    model_profile_id = "vgn:test"
    semantic_content_hash = "vgn-model:test"


class _FakePlanner:
    model_spec = _FakeModelSpec()
    semantic_content_hash = "point-grasp-planner:test"

    def start(self):
        return None

    def close(self):
        return None

    def plan(self, image, depth, *, pixel_uv, intrinsics, base_from_camera, max_candidates):
        assert image.size == (12, 10)
        assert depth.shape == (10, 12)
        assert pixel_uv == (6, 5)
        assert max_candidates == 3
        camera = np.eye(4)
        camera[:3, 3] = [0.01, 0.02, 0.5]
        base = np.asarray(base_from_camera) @ camera
        return PointGraspResult(
            target_point_camera_m=(0.0, 0.0, 0.5),
            target_point_base_m=tuple(base_from_camera[:3, 3] + [0, 0, 0.5]),
            target_mask_bbox_xyxy=(2, 2, 10, 9),
            candidates=(
                PointGraspCandidate(
                    score=0.9,
                    model_score=0.95,
                    click_distance_m=0.02,
                    gripper_width_m=0.05,
                    camera_from_tcp=tuple(camera.reshape(-1)),
                    base_from_tcp=tuple(base.reshape(-1)),
                    visible_collision_free=True,
                ),
            ),
        )


def _registration(local_path_enabled=True):
    return build_real_point_grasp_registration(
        backend_id="vgn",
        model_profile_id=_FakeModelSpec.model_profile_id,
        model_profile_hash=_FakeModelSpec.semantic_content_hash,
        inference_implementation_hash=_FakePlanner.semantic_content_hash,
        local_path_enabled=local_path_enabled,
        runtime_session_id="session:test",
        control_domain_id="domain:test",
        target_binding_id="target:test",
        station_manifest_hash="station:test",
        execution_manifest_hash="execution:test",
    )


def _parameters(rgb: Path | None, depth: Path | None):
    return {
        "input_source": "local_path" if rgb else "artifact",
        "local_rgb_path": None if rgb is None else str(rgb),
        "local_depth_path": None if depth is None else str(depth),
        "target_pixel_uv": [6, 5],
        "camera_intrinsics": {
            "width": 12, "height": 10, "fx": 10.0, "fy": 10.0, "cx": 6.0, "cy": 5.0,
        },
        "camera_frame_id": "fixed_camera_optical_frame",
        "base_frame_id": "right_arm_base_link",
        "T_base_from_camera": [1, 0, 0, 0.1, 0, 1, 0, 0.2, 0, 0, 1, 0.3, 0, 0, 0, 1],
        "max_candidates": 3,
    }


def test_point_grasp_registration_is_passive_and_has_two_inputs() -> None:
    registration = _registration(local_path_enabled=False)
    route = registration.skill.routes[0]
    assert registration.skill.definition.skill_id == "perception.plan_point_grasp"
    assert route.control_effect_class == "read_only"
    assert route.may_move_robot is False
    assert {slot.slot_id for slot in registration.skill.io_contract.input_artifact_slots} == {
        REAL_POINT_GRASP_RGB_SLOT_ID_V1,
        REAL_POINT_GRASP_DEPTH_SLOT_ID_V1,
    }
    registration.payload_validator.validate_parameters(_parameters(None, None))


def test_point_grasp_local_rgbd_produces_camera_and_base_tcp_poses(tmp_path: Path) -> None:
    rgb = tmp_path / "frame.png"
    Image.new("RGB", (12, 10), color=(32, 64, 96)).save(rgb)
    depth = tmp_path / "depth.npy"
    np.save(depth, np.full((10, 12), 0.5, dtype=np.float32), allow_pickle=False)
    registration = _registration()
    provider = PointGraspProvider(
        planner=_FakePlanner(), registration=registration, allowed_local_image_roots=(tmp_path,)
    )
    request = SimpleNamespace(parameters=_parameters(rgb, depth), input_artifact_bindings=())
    prepared = provider.plan_request(request, artifact_store=SimpleNamespace())
    result = prepared.execution_result
    assert result["candidate_count"] == 1
    assert result["selected_ordinal"] == 0
    assert result["candidates"][0]["tcp_xyz_camera_m"] == [0.01, 0.02, 0.5]
    assert result["candidates"][0]["tcp_xyz_base_m"] == [0.11, 0.22, 0.8]
    registration.payload_validator.validate_execution_result(result)
    registration.payload_validator.validate_execution_result_for_parameters(request.parameters, result)
    assert Image.open(io.BytesIO(prepared.annotated_jpeg)).size == (12, 10)


def test_local_rgb_and_depth_may_use_different_allowlisted_roots(tmp_path: Path) -> None:
    rgb_root = tmp_path / "rgb"
    depth_root = tmp_path / "depth"
    rgb_root.mkdir()
    depth_root.mkdir()
    rgb = rgb_root / "frame.png"
    Image.new("RGB", (12, 10), color=(32, 64, 96)).save(rgb)
    depth = depth_root / "depth.npy"
    np.save(depth, np.full((10, 12), 0.5, dtype=np.float32), allow_pickle=False)
    provider = PointGraspProvider(
        planner=_FakePlanner(),
        registration=_registration(),
        allowed_local_image_roots=(rgb_root, depth_root),
    )
    prepared = provider.plan_request(
        SimpleNamespace(parameters=_parameters(rgb, depth), input_artifact_bindings=()),
        artifact_store=SimpleNamespace(),
    )
    assert prepared.execution_result["candidate_count"] == 1


def test_artifact_depth_identity_is_checked_before_store_read() -> None:
    provider = PointGraspProvider(
        planner=_FakePlanner(), registration=_registration(local_path_enabled=False)
    )
    rgb = SimpleNamespace(
        media_type=LOCALIZATION_INPUT_PNG_MEDIA_TYPE_V1,
        schema_id=POINT_GRASP_RGB_SCHEMA_ID_V1,
        encoding=LOCALIZATION_INPUT_PNG_ENCODING_V1,
        compression=LOCALIZATION_INPUT_IMAGE_COMPRESSION_V1,
    )
    depth = SimpleNamespace(
        media_type=POINT_GRASP_DEPTH_MEDIA_TYPE_V1,
        schema_id="wrong-depth-schema",
        encoding=POINT_GRASP_DEPTH_ENCODING_V1,
        compression=POINT_GRASP_DEPTH_COMPRESSION_V1,
    )
    request = SimpleNamespace(
        parameters=_parameters(None, None),
        input_artifact_bindings=(
            SimpleNamespace(slot_id=REAL_POINT_GRASP_RGB_SLOT_ID_V1, artifact=rgb),
            SimpleNamespace(slot_id=REAL_POINT_GRASP_DEPTH_SLOT_ID_V1, artifact=depth),
        ),
    )
    with pytest.raises(ValueError, match="depth Artifact identity"):
        provider.plan_request(request, artifact_store=SimpleNamespace())


def test_model_resources_start_sam_before_vgn_and_close_in_reverse() -> None:
    events: list[str] = []

    class _Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def start(self) -> None:
            events.append(self.name + ".start")

        def close(self) -> None:
            events.append(self.name + ".close")

    class _Camera:
        def close(self) -> None:
            events.append("camera.close")

    resources = CameraOnlyRealPerceptionResources(
        camera_sources={"main": _Camera()},
        artifact_store=_Resource("store"),
        additional_resources=(_Resource("sam2"), _Resource("vgn")),
        start_cameras=lambda _sources: events.append("camera.start"),
        admit_cameras=lambda _sources: events.append("camera.admit"),
    )
    resources.start()
    resources.close()
    assert events == [
        "camera.start",
        "camera.admit",
        "sam2.start",
        "vgn.start",
        "vgn.close",
        "sam2.close",
        "camera.close",
        "store.close",
    ]


def test_camera_only_composition_registers_shared_sam2_and_point_grasp(
    tmp_path: Path,
) -> None:
    class _Camera:
        config = SimpleNamespace(
            camera_key="front",
            width=12,
            height=10,
            depth_width=12,
            depth_height=10,
            align_depth_to_color=True,
            enable_depth=True,
            color_space="rgb",
            serial="front-test",
        )

        def device_info(self):
            return {"running": False, "generation": 0, "serial_number": "front-test"}

        def close(self):
            return None

    class _SamSpec:
        model_profile_id = "sam2:test"
        semantic_content_hash = "sam2-model:test"

    class _Sam:
        model_spec = _SamSpec()
        semantic_content_hash = "sam2-localizer:test"

        def start(self):
            return None

        def close(self):
            return None

        def localize(self, *_args, **_kwargs):
            return ()

        def segment_point(self, *_args, **_kwargs):
            return np.ones((10, 12), dtype=bool)

    class _VgnSpec:
        model_profile_id = "vgn:test"
        semantic_content_hash = "vgn-model:test"

    class _Planner:
        backend_id = "vgn"
        model_spec = _VgnSpec()
        semantic_content_hash = "sam2-vgn-planner:test"

        def __init__(self, segmenter):
            self.point_segmenter = segmenter

        def start(self):
            return None

        def close(self):
            return None

        def plan(self, *_args, **_kwargs):
            raise AssertionError("composition test must not run inference")

    sam2 = _Sam()
    source_binding = RealPerceptionSourceBinding(
        camera_sources={"front": _Camera()},
        world_from_depth_camera_by_source={"front": np.eye(4)},
        semantic_content_hash="fixed-camera-binding:test",
    )
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    composition = assemble_camera_only_real_perception_composition(
        source_binding=source_binding,
        socket_path=tmp_path / "gateway.sock",
        artifact_root=artifact_root,
        client_bindings=compile_agent_gateway_unix_client_bindings(
            primary_client_id="test-client", primary_client_uid=1234
        ),
        runtime_session_id="point-grasp-composition:test",
        control_domain_id="read-only:test",
        scene_geometry_hash="scene:test",
        object_localizer=sam2,
        point_grasp_planner=_Planner(sam2),
    )
    try:
        assert composition.localization_registration is not None
        assert composition.point_grasp_registration is not None
        assert composition.session.localization_executor is not None
        assert composition.session.point_grasp_executor is not None
        assert len(composition.contract_bundle.route_manifest.skill_routes) == 4
        assert composition.resources.additional_resources == (
            composition.localization_provider,
            composition.point_grasp_provider,
        )
    finally:
        composition.session.close()
        composition.resources.close()


def test_vgn_manifest_and_dmtac_m_default_urdf_are_pinned(tmp_path: Path) -> None:
    checkpoint = tmp_path / "vgn_conv.pth"
    checkpoint.write_bytes(b"vgn")
    manifest = {
        "schema_version": "dexteleop.perception_point_grasp_assets.v1",
        "model_profile_id": "vgn-dmtac-m:test",
        "source": {"git_url": VGN_GIT_URL, "revision": VGN_REVIEWED_REVISION},
        "checkpoint": {
            "relative_path": checkpoint.name,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "model_type": "conv",
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    spec = VgnPointGraspModelSpec.from_asset_bundle(tmp_path)
    assert spec.checkpoint_path == checkpoint
    assert DEFAULT_DMTAC_M_URDF == DEFAULT_RM65_PHYSICS_URDF
    urdf = Path(__file__).resolve().parents[1] / DEFAULT_DMTAC_M_URDF
    assert hashlib.sha256(urdf.read_bytes()).hexdigest() == DEFAULT_DMTAC_M_URDF_SHA256


def test_projective_tsdf_uses_clicked_sam2_mask(tmp_path: Path) -> None:
    checkpoint = tmp_path / "vgn_conv.pth"
    checkpoint.write_bytes(b"vgn")
    spec = VgnPointGraspModelSpec(
        model_profile_id="test", checkpoint_path=checkpoint,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_revision=VGN_REVIEWED_REVISION, model_type="conv",
    )
    segmenter = SimpleNamespace(segment_point=lambda *_args, **_kwargs: None, semantic_content_hash="sam:test")
    planner = Sam2VgnPointGraspPlanner(model_spec=spec, point_segmenter=segmenter)
    depth = np.ones((10, 12), dtype=np.float32)
    mask = np.zeros_like(depth, dtype=bool)
    mask[:, 4:8] = True
    grid = planner._projective_tsdf(
        depth, mask, {"fx": 10.0, "fy": 10.0, "cx": 6.0, "cy": 5.0}, np.eye(4)
    )
    assert grid.shape == (1, 40, 40, 40)
    assert grid.dtype == np.float32
    assert np.count_nonzero(grid) > 0
    assert np.count_nonzero(grid) < grid.size


@pytest.mark.parametrize(
    "planner_class",
    (Sam2VgnPointGraspPlanner, Sam2GraspNetPointGraspPlanner),
)
def test_dmtac_visible_collision_gate_accepts_two_hundred_points_but_rejects_two_hundred_one(
    planner_class: type,
) -> None:
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[5:15, :] = 1.0
    camera_from_tcp = np.eye(4, dtype=np.float64)
    camera_from_tcp[2, 3] = 1.0
    intrinsics = {"fx": 1000.0, "fy": 1000.0, "cx": 10.0, "cy": 10.0}

    assert DMTAC_M_MAX_VISIBLE_COLLISION_POINTS == 200
    assert planner_class._visible_collision_free(
        camera_from_tcp, 0.05, depth, intrinsics
    )

    depth[15, 10] = 1.0
    assert not planner_class._visible_collision_free(
        camera_from_tcp, 0.05, depth, intrinsics
    )


def test_graspnet_manifest_requires_noncommercial_local_download_acknowledgement(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-rs.tar"
    checkpoint.write_bytes(b"graspnet")
    manifest = {
        "schema_version": "dexteleop.perception_graspnet_point_grasp_assets.v1",
        "model_profile_id": "graspnet-realsense-dmtac-m:test",
        "source": {
            "git_url": GRASPNET_GIT_URL,
            "revision": GRASPNET_REVIEWED_REVISION,
        },
        "checkpoint": {
            "relative_path": checkpoint.name,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "model_type": "realsense",
        },
        "license_acknowledgement": GRASPNET_LICENSE_ACKNOWLEDGEMENT_V1,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    spec = GraspNetPointGraspModelSpec.from_asset_bundle(tmp_path)
    assert spec.checkpoint_path == checkpoint

    manifest["license_acknowledgement"] = "missing"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="license acknowledgement"):
        GraspNetPointGraspModelSpec.from_asset_bundle(tmp_path)


def test_graspnet_candidate_axes_are_mapped_to_dmtac_tool_tcp(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-rs.tar"
    checkpoint.write_bytes(b"graspnet")
    spec = GraspNetPointGraspModelSpec(
        model_profile_id="graspnet:test",
        checkpoint_path=checkpoint,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_revision=GRASPNET_REVIEWED_REVISION,
        model_type="realsense",
        license_acknowledgement=GRASPNET_LICENSE_ACKNOWLEDGEMENT_V1,
    )
    segmenter = SimpleNamespace(
        segment_point=lambda *_args, **_kwargs: None,
        semantic_content_hash="sam:test",
    )
    planner = Sam2GraspNetPointGraspPlanner(
        model_spec=spec, point_segmenter=segmenter, device="cuda"
    )
    prediction = np.zeros((2, 17), dtype=np.float64)
    prediction[0, 0] = 0.9
    prediction[0, 1] = 0.05
    prediction[0, 2] = 0.02
    prediction[0, 3] = 0.03
    prediction[0, 4:13] = np.eye(3).reshape(-1)
    prediction[0, 13:16] = [0.0, 0.0, 0.5]
    prediction[0, 16] = -1
    prediction[1] = prediction[0]
    prediction[1, 0] = 0.8
    prediction[1, 4:13] = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    ).reshape(-1)
    candidates = planner._rank(
        prediction,
        target_points=np.asarray(
            [[0.0, -0.01, 0.51], [0.0, 0.02, 0.53]],
            dtype=np.float32,
        ),
        target_camera=np.asarray([0.0, 0.005, 0.5]),
        base_from_camera=np.asarray(
            [[1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]],
            dtype=float,
        ),
        depth=np.zeros((10, 12), dtype=np.float32),
        intrinsics={"fx": 10.0, "fy": 10.0, "cx": 6.0, "cy": 5.0},
        maximum=5,
    )
    # A distinct roll about the same approach axis is a different grasp pose.
    assert len(candidates) == 2
    camera = np.asarray(candidates[0].camera_from_tcp).reshape(4, 4)
    base = np.asarray(candidates[0].base_from_tcp).reshape(4, 4)
    np.testing.assert_allclose(camera[:3, 0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(camera[:3, 2], [1.0, 0.0, 0.0])
    # depth=0.03 and the official 0.06 m finger length place tool_tcp at
    # the model palm/finger boundary, 0.03 m behind the grasp seed.
    # Robust target-cloud extents also center the grasp on both axes
    # perpendicular to approach (opening y and finger-height z).
    np.testing.assert_allclose(camera[:3, 3], [-0.03, 0.005, 0.52])
    np.testing.assert_allclose(base[:3, 3], [0.07, 0.205, 0.82])


def test_graspnet_point_conditioned_workspace_crops_unrelated_scene_points() -> None:
    planner = object.__new__(Sam2GraspNetPointGraspPlanner)
    planner.point_conditioned_workspace_margin_m = 0.120
    target = np.column_stack(
        (
            np.linspace(-0.01, 0.01, 32),
            np.zeros(32),
            np.full(32, 0.50),
        )
    ).astype(np.float32)
    nearby = np.column_stack(
        (
            np.linspace(-0.10, 0.10, 160),
            np.zeros(160),
            np.full(160, 0.52),
        )
    ).astype(np.float32)
    unrelated = np.column_stack(
        (
            np.linspace(0.5, 1.0, 200),
            np.zeros(200),
            np.full(200, 1.5),
        )
    ).astype(np.float32)
    scene = np.concatenate((target, nearby, unrelated), axis=0)
    workspace = planner._point_conditioned_workspace(scene, target)
    assert workspace.flags.c_contiguous
    assert workspace.dtype == np.float32
    assert len(workspace) == len(target) + len(nearby)
    assert np.max(workspace[:, 2]) < 1.0


def test_graspnet_point_conditioned_workspace_falls_back_when_crop_is_too_sparse() -> None:
    planner = object.__new__(Sam2GraspNetPointGraspPlanner)
    planner.point_conditioned_workspace_margin_m = 0.120
    target = np.zeros((16, 3), dtype=np.float32)
    scene = np.concatenate((target, np.ones((200, 3), dtype=np.float32)), axis=0)
    workspace = planner._point_conditioned_workspace(scene, target)
    np.testing.assert_array_equal(workspace, scene)
