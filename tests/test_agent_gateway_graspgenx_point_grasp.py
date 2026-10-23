from __future__ import annotations

import json
import math
from pathlib import Path

import msgpack
import numpy as np
import pytest
from PIL import Image

from agent_gateway.profiles.real.point_grasp_deployment import (
    DEFAULT_POINT_GRASP_BACKEND_ID,
)
from deployments.perception.graspgenx_client import (
    GRASPGENX_DEFAULT_MODE,
    GRASPGENX_POINT_GRASP_MOE_NUM_YAWS,
    GRASPGENX_POINT_GRASP_MOE_SIDE_HEIGHT_OFFSETS_CM,
    GRASPGENX_POINT_GRASP_MOE_Z_OFFSETS_CM,
    GRASPGENX_POINT_GRASP_MOE_OBB_DENSITY,
    GRASPGENX_POINT_GRASP_MOE_OBB_POSITION_SPACING_CM,
    GraspGenXClient,
    GraspGenXInferenceBackend,
    GraspGenXInProcessBackend,
    pack_graspgenx_message,
    unpack_graspgenx_message,
)
from deployments.perception.graspgenx_runtime_contract import (
    GRASPGENX_INPROCESS_RUNTIME_CONTRACT_ID,
    GraspGenXRuntimeCheck,
    GraspGenXRuntimeReceipt,
    verify_graspgenx_inprocess_runtime,
)
from deployments.perception.sam2_graspgenx_point_grasp import (
    DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
    GRASPGENX_CHECKPOINT_REVIEWED_REVISION,
    GRASPGENX_REVIEWED_REVISION,
    GraspGenXCandidateSelection,
    GraspGenXPointGraspModelSpec,
    Sam2GraspGenXPointGraspPlanner,
    _RankedGraspCandidate,
    _deduplicate_ranked_poses,
    _pose_is_duplicate,
)
from deployments.perception.sam2_vgn_point_grasp import Sam2VgnPointGraspPlanner
from deployments.perception.dmtac_grasp_overlay import (
    reviewed_4b2_cad_solid_geometry,
    reviewed_graspgenx_dmtac_solid_geometry,
)
from robot_motion.grasp_visible_scene import VisibleGraspScene
from deployments.perception.grasp_planning_priority import (
    GRASP_PLANNING_ORDER_FIELD, GRASP_PLANNING_ORDER_SCOPE, grasp_planning_candidate_ordinals,
)


class _FakeSocket:
    def __init__(self, responses: list[bytes]) -> None:
        self.responses = list(responses)
        self.sent: list[bytes] = []
        self.connected_to = None
        self.closed = False

    def setsockopt(self, *_args) -> None:
        return None

    def connect(self, address: str) -> None:
        self.connected_to = address

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    def recv(self) -> bytes:
        return self.responses.pop(0)

    def close(self, *, linger: int) -> None:
        assert linger == 0
        self.closed = True


class _FakeContext:
    def __init__(self, socket: _FakeSocket) -> None:
        self._socket = socket

    def socket(self, socket_type: int) -> _FakeSocket:
        assert socket_type
        return self._socket


class _Segmenter:
    semantic_content_hash = "sam2:test"

    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask

    def segment_point(self, image: Image.Image, *, pixel_uv: tuple[int, int]):
        assert image.mode == "RGB"
        assert self.mask[pixel_uv[1], pixel_uv[0]]
        return self.mask.copy()


class _PlannerClient(GraspGenXClient):
    def __init__(self, grasps: np.ndarray, scores: np.ndarray) -> None:
        super().__init__()
        self.grasps = grasps
        self.scores = scores
        self.started = False
        self.closed_for_test = False
        self.calls = []

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed_for_test = True

    def infer_object(self, point_cloud, **kwargs):
        self.calls.append((np.asarray(point_cloud).copy(), kwargs))
        return self.grasps.copy(), self.scores.copy()


class _DuckBackend:
    backend_id = "duck-backend-v1"
    semantic_config = {"backend": backend_id}

    def __init__(self, grasps: np.ndarray, scores: np.ndarray) -> None:
        self.grasps = grasps
        self.scores = scores
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def infer_object(self, point_cloud, **kwargs):
        return self.grasps.copy(), self.scores.copy()


def test_embodied_half_turn_covers_tilted_candidate_and_keeps_pool_lineage_internal(tmp_path):
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[8:13, 8:13] = 1.
    mask = depth > 0
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 1.025
    planner = Sam2GraspGenXPointGraspPlanner(model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(mask), client=_PlannerClient(pose[None], np.array([.9])),
        orientation_variants_local_axis_degrees=(("x", -10.),), enable_finger_exchange=True)
    chosen = []

    def select(pool, _target, _maximum):
        for index, candidate in enumerate(pool):
            parent_index = candidate.symmetry_parent_index
            if parent_index is None:
                continue
            parent = pool[parent_index]
            if parent.orientation_variant_axis != "x":
                continue
            first = planner.solid_geometry.solid_pose(np.asarray(parent.camera_from_tcp).reshape(4, 4))
            second = planner.solid_geometry.solid_pose(np.asarray(candidate.camera_from_tcp).reshape(4, 4))
            np.testing.assert_allclose(first[:3, 2], second[:3, 2], atol=1e-12)
            np.testing.assert_allclose(first[:3, 0], -second[:3, 0], atol=1e-12)
            local = (np.array([0., 0., 1.]) - first[:3, 3]) @ first[:3, :3]
            np.testing.assert_allclose(second[:3, :3] @ local + second[:3, 3], [0., 0., 1.], atol=1e-12)
            chosen.append(candidate)
            return GraspGenXCandidateSelection((index,), {})
        pytest.fail("The readmitted tilted proposal must have a checked half-turn")

    planner.start()
    try:
        result = planner.plan(Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)), depth,
            pixel_uv=(10, 10), intrinsics={"width":20,"height":20,"fx":100.,"fy":100.,"cx":10.,"cy":10.},
            base_from_camera=np.eye(4), max_candidates=1, candidate_selector=select)
    finally:
        planner.close()
    assert len(chosen) == 1
    assert not hasattr(result.candidates[0], "symmetry_parent_index")
    assert result.candidates[0].orientation_variant_angle_deg == 180.


def test_support_clearance_retains_arm_side_and_top_without_ignoring_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth = np.zeros((80, 80), dtype=np.float32)
    mask = np.zeros(depth.shape, dtype=bool)
    mask[38:43, 38:43] = True
    depth[mask] = 1.0
    top = np.diag((1.0, -1.0, -1.0, 1.0))
    top[2, 3] = 1.025
    top_poses = []
    for index, yaw in enumerate(np.linspace(0.0, 2 * np.pi, 8, endpoint=False)):
        cosine, sine = np.cos(yaw), np.sin(yaw)
        yaw_rotation = np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
        )
        pose = top.copy()
        pose[:3, :3] = yaw_rotation @ top[:3, :3]
        pose[0, 3] = 0.0015 * ((index % 3) - 1)
        pose[1, 3] = 0.0015 * ((index // 3) - 1)
        top_poses.append(pose)
    side = np.eye(4)
    side[:3, :3] = np.column_stack(
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    )
    side[:3, 3] = (-0.025, 0.0, 1.0)
    planner = Sam2GraspGenXPointGraspPlanner(
        model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(mask),
        client=_PlannerClient(
            np.stack((*top_poses, side)),
            np.asarray((*np.linspace(0.9, 0.83, 8), 0.8)),
        ),
        solid_geometry=reviewed_graspgenx_dmtac_solid_geometry(),
        preferred_approach_origin_base_m=(-0.2, 0.0, 0.0),
    )
    blocked_table_heights = []

    def _terminal(_scene, pose, *, opening_m, contact_depth_m):
        del opening_m, contact_depth_m
        direction = np.asarray(pose)[:3, 2]
        z = float(np.asarray(pose)[2, 3])
        # The observed table remains solid: side grasps require a larger
        # height correction than the top grasp to clear it.
        blocked = z < (1.01 if abs(direction[2]) < 0.7 else 1.03)
        if blocked:
            blocked_table_heights.append(z)
        return 0, 0, (3 if abs(direction[2]) < 0.7 else 1) if blocked else 0

    monkeypatch.setattr(
        VisibleGraspScene,
        "collision_counts_with_terminal_target_contact",
        _terminal,
    )
    monkeypatch.setattr(
        VisibleGraspScene, "approach_collision_counts",
        lambda *_args, **_kwargs: (0, 0),
    )
    pool = []

    def _select(candidates, _target, maximum):
        pool.extend(candidates)
        return GraspGenXCandidateSelection(
            selected_indices=tuple(range(min(maximum, len(candidates)))),
            diagnostics={},
        )

    planner.start()
    try:
        planner.plan(
            Image.new("RGB", (80, 80)), depth, pixel_uv=(40, 40),
            intrinsics={
                "width": 80, "height": 80, "fx": 1000.0,
                "fy": 1000.0, "cx": 40.0, "cy": 40.0,
            },
            base_from_camera=np.eye(4), max_candidates=5,
            visible_scene_factory=lambda **kwargs: VisibleGraspScene(
                points=np.asarray(((0.0, 0.0, 1.0), (1.0, 1.0, 1.0))),
                target_mask=np.asarray((True, False)),
                geometry=kwargs["geometry"],
            ),
            candidate_selector=_select,
        )
        lifted = [
            candidate for candidate in pool
            if candidate.pose_source == "dmtac_support_clearance_variant_v1"
        ]
        assert any(
            abs(np.asarray(candidate.base_from_tcp).reshape(4, 4)[2, 2]) < 0.7
            for candidate in lifted
        )
        assert any(
            -np.asarray(candidate.base_from_tcp).reshape(4, 4)[2, 2] >= 0.7
            for candidate in lifted
        )
        assert blocked_table_heights
        assert planner.last_filter_summary["support_clearance_side_admitted"] == 4
        assert planner.last_filter_summary["support_clearance_top_admitted"] > 0
        assert all(
            not _terminal(
                None,
                planner.solid_geometry.solid_pose(
                    np.asarray(candidate.base_from_tcp).reshape(4, 4)
                ),
                opening_m=0.0705,
                contact_depth_m=0.006,
            )[2]
            for candidate in lifted
        )
    finally:
        planner.close()


def test_graspgenx_default_mode_is_inprocess() -> None:
    assert DEFAULT_POINT_GRASP_BACKEND_ID == "graspgenx"
    assert GRASPGENX_DEFAULT_MODE == "in-process"


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "graspgenx"
    paths = {
        "generator_config": "checkpoints/release/gen/config.yaml",
        "generator_checkpoint": "checkpoints/release/gen/epoch_736.pth",
        "discriminator_config": "checkpoints/release/dis/config.yaml",
        "discriminator_checkpoint": "checkpoints/release/dis/epoch_1056.pth",
    }
    files = {}
    for index, (name, relative) in enumerate(paths.items()):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"test-{index}".encode())
        files[name] = {"relative_path": relative, "sha256": _sha256(path)}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "dexteleop.perception_graspgenx_point_grasp_assets.v1",
                "model_profile_id": "graspgenx-test",
                "source": {
                    "git_url": "https://github.com/NVlabs/GraspGenX.git",
                    "revision": GRASPGENX_REVIEWED_REVISION,
                    "license": "Apache-2.0",
                },
                "checkpoint_repository": {
                    "git_url": "https://huggingface.co/adithyamurali/GraspGenXModel",
                    "revision": GRASPGENX_CHECKPOINT_REVIEWED_REVISION,
                    "license": "NVIDIA-Open-Model-License",
                },
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return root


def _model_spec(tmp_path: Path) -> GraspGenXPointGraspModelSpec:
    return GraspGenXPointGraspModelSpec.from_asset_bundle(_asset_bundle(tmp_path))


def _receipt(*, passed: bool, failed_name: str | None = None) -> GraspGenXRuntimeReceipt:
    return GraspGenXRuntimeReceipt(
        contract_id=GRASPGENX_INPROCESS_RUNTIME_CONTRACT_ID,
        device="cuda",
        source_root="/tmp/graspgenx-source",
        model_spec_hash="0" * 64,
        checks=(
            GraspGenXRuntimeCheck(
                name=failed_name or "python_version",
                status=passed,
                expected="expected",
                observed="observed",
                detail="" if passed else "stubbed failure",
            ),
        ),
    )


def test_inprocess_backend_is_a_planner_backend_without_loading_on_init(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = GraspGenXInProcessBackend(
        model_spec=_model_spec(tmp_path),
        source_root=tmp_path / "missing-source",
    )

    assert isinstance(backend, GraspGenXInferenceBackend)
    assert backend.semantic_config["backend"] == "graspgenx-inprocess-v1"
    assert (
        backend.semantic_config["runtime_contract"]
        == GRASPGENX_INPROCESS_RUNTIME_CONTRACT_ID
    )
    assert backend._runtime_verifier is verify_graspgenx_inprocess_runtime
    monkeypatch.setattr(
        backend,
        "_runtime_verifier",
        lambda **_kwargs: _receipt(passed=False, failed_name="source_root"),
    )

    with pytest.raises(RuntimeError, match="runtime contract .*source_root"):
        backend.start()
    assert backend.runtime_receipt is not None
    assert not backend.runtime_receipt.passed


def test_inprocess_backend_start_fails_closed_on_contract_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = GraspGenXInProcessBackend(model_spec=_model_spec(tmp_path))
    monkeypatch.setattr(
        backend,
        "_runtime_verifier",
        lambda **_kwargs: _receipt(passed=False, failed_name="torch_version"),
    )

    with pytest.raises(RuntimeError, match="torch_version"):
        backend.start()
    assert backend.runtime_receipt is not None
    assert backend.runtime_receipt.failed_checks[0].name == "torch_version"
    with pytest.raises(RuntimeError, match="not running"):
        backend.infer_object(
            np.zeros((16, 3), dtype=np.float32),
            sweep_volume_params=DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
        )


def test_inprocess_backend_rejects_cpu_before_touching_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        GraspGenXInProcessBackend(model_spec=_model_spec(tmp_path), device="cpu")


def test_inprocess_backend_reuses_client_filtering_and_sampler_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _Cuda:
        class _Context:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        def device(self, _device):
            return self._Context()

    class _Torch:
        cuda = _Cuda()

    class _Params:
        @classmethod
        def coerce(cls, value):
            assert value["gripper_type"] == 0
            return cls()

        def cache_key(self):
            return "test"

    class _Sampler:
        calls = 0

        @classmethod
        def from_sweep_volume(cls, cfg, params, *, model):
            assert cfg == "cfg"
            assert isinstance(params, _Params)
            assert model == "model"
            cls.calls += 1
            return "sampler"

    runner_calls = []
    first = np.eye(4, dtype=np.float64)
    second = np.eye(4, dtype=np.float64)
    second[0, 3] = 0.01

    def _runner(points, sampler, **kwargs):
        runner_calls.append((points.copy(), sampler, kwargs))
        return (
            np.concatenate((np.repeat(first[None], 1001, axis=0), second[None])),
            np.asarray([0.2] * 1001 + [0.9]), [], None,
        )

    backend = GraspGenXInProcessBackend(model_spec=_model_spec(tmp_path))
    monkeypatch.setattr(
        backend, "_runtime_verifier", lambda **_kwargs: _receipt(passed=True)
    )
    monkeypatch.setattr(
        backend,
        "_load_runtime",
        lambda: (
            _Torch(),
            "cuda",
            "cfg",
            "model",
            _Sampler,
            _Params,
            _runner,
            False,
        ),
    )
    backend.start()
    assert backend.runtime_receipt is not None and backend.runtime_receipt.passed

    points = np.zeros((16, 3), dtype=np.float32)
    grasps, scores = backend.infer_object(
        points,
        sweep_volume_params=DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
        score_threshold=0.3,
        top_k=1,
    )

    assert _Sampler.calls == 1
    assert len(runner_calls) == 1
    assert runner_calls[0][2]["topk_num_grasps"] == -1
    assert (
        runner_calls[0][2]["moe_num_yaws"]
        == GRASPGENX_POINT_GRASP_MOE_NUM_YAWS
    )
    assert runner_calls[0][2]["moe_z_offsets_cm"] == GRASPGENX_POINT_GRASP_MOE_Z_OFFSETS_CM
    assert runner_calls[0][2]["moe_side_height_offsets_cm"] == (
        GRASPGENX_POINT_GRASP_MOE_SIDE_HEIGHT_OFFSETS_CM
    )
    assert -3.0 in GRASPGENX_POINT_GRASP_MOE_Z_OFFSETS_CM
    assert set((-8.0, -6.0, -4.0, -2.0, -1.0, 0.0)) <= set(GRASPGENX_POINT_GRASP_MOE_Z_OFFSETS_CM)
    assert backend.semantic_config["graspmoe_sampling"]["z_offsets_cm"] == GRASPGENX_POINT_GRASP_MOE_Z_OFFSETS_CM
    assert backend.semantic_config["graspmoe_sampling"]["side_height_offsets_cm"] == (
        GRASPGENX_POINT_GRASP_MOE_SIDE_HEIGHT_OFFSETS_CM
    )
    assert (
        runner_calls[0][2]["moe_obb_density"]
        == GRASPGENX_POINT_GRASP_MOE_OBB_DENSITY
    )
    assert runner_calls[0][2]["moe_obb_position_spacing_cm"] == pytest.approx(
        GRASPGENX_POINT_GRASP_MOE_OBB_POSITION_SPACING_CM
    )
    assert scores.tolist() == pytest.approx([0.9])
    np.testing.assert_allclose(grasps[0], second)
    all_grasps, all_scores = backend.infer_object(
        points, sweep_volume_params=DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
        top_k=-1,
    )
    assert _Sampler.calls == 1
    assert len(runner_calls) == 2
    assert len(all_grasps) == len(all_scores) == 1002
    np.testing.assert_allclose(all_grasps[0], second)
    assert all_scores[0] == pytest.approx(.9)
    assert all_scores[-1] == pytest.approx(.2)
    backend.close()


def test_planner_accepts_any_backend_implementing_the_inference_contract(
    tmp_path: Path,
) -> None:
    backend = _DuckBackend(
        np.eye(4, dtype=np.float64)[None],
        np.asarray((0.9,), dtype=np.float64),
    )

    planner = Sam2GraspGenXPointGraspPlanner(
        model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(np.ones((4, 4), dtype=bool)),
        client=backend,
    )

    assert isinstance(planner.client, GraspGenXInferenceBackend)
    assert planner.semantic_content_hash


def test_numpy_codec_matches_upstream_msgpack_numpy_shape() -> None:
    points = np.arange(12, dtype=np.float32).reshape(4, 3)
    packed = pack_graspgenx_message({"point_cloud": points})
    raw = msgpack.unpackb(packed, raw=False)

    encoded_array = raw["point_cloud"]
    assert encoded_array[b"nd"] is True
    assert encoded_array[b"type"] == "<f4"
    assert tuple(encoded_array[b"shape"]) == (4, 3)
    decoded = unpack_graspgenx_message(packed)
    np.testing.assert_array_equal(decoded["point_cloud"], points)


def test_client_sends_dmtac_sweep_volume_and_sorts_top_k() -> None:
    first = np.eye(4, dtype=np.float32)
    second = np.eye(4, dtype=np.float32)
    second[0, 3] = 0.01
    socket = _FakeSocket(
        [
            pack_graspgenx_message(
                {
                    "grasps": np.stack((first, second)),
                    "confidences": np.asarray((0.2, 0.8), dtype=np.float32),
                }
            )
        ]
    )
    client = GraspGenXClient(context=_FakeContext(socket))

    grasps, scores = client.infer_object(
        np.zeros((16, 3), dtype=np.float32),
        sweep_volume_params=DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
        top_k=1,
    )

    assert socket.connected_to == "tcp://127.0.0.1:5556"
    request = unpack_graspgenx_message(socket.sent[0])
    assert request["action"] == "infer_object"
    assert request["moe_num_yaws"] == GRASPGENX_POINT_GRASP_MOE_NUM_YAWS
    assert request["moe_z_offsets_cm"] == list(GRASPGENX_POINT_GRASP_MOE_Z_OFFSETS_CM)
    assert request["moe_side_height_offsets_cm"] == list(
        GRASPGENX_POINT_GRASP_MOE_SIDE_HEIGHT_OFFSETS_CM
    )
    assert request["moe_obb_density"] == GRASPGENX_POINT_GRASP_MOE_OBB_DENSITY
    assert request["moe_obb_position_spacing_cm"] == pytest.approx(
        GRASPGENX_POINT_GRASP_MOE_OBB_POSITION_SPACING_CM
    )
    assert request["sweep_volume_params"]["gripper_type"] == 0
    np.testing.assert_allclose(
        request["sweep_volume_params"]["extents_open"],
        (0.0705, 0.030, 0.100),
    )
    assert scores.tolist() == pytest.approx([0.8])
    np.testing.assert_allclose(grasps[0], second)


def test_client_start_requires_health_and_infer_object_metadata() -> None:
    socket = _FakeSocket(
        [
            pack_graspgenx_message({"status": "ok"}),
            pack_graspgenx_message(
                {"actions": ["health", "metadata", "infer_object"]}
            ),
        ]
    )
    client = GraspGenXClient(context=_FakeContext(socket))

    client.start()

    assert [unpack_graspgenx_message(payload)["action"] for payload in socket.sent] == [
        "health",
        "metadata",
    ]


def test_client_can_retain_scored_union_larger_than_1000() -> None:
    grasps = np.repeat(np.eye(4)[None], 1002, axis=0)
    grasps[:, 0, 3] = np.arange(1002) * .001
    scores = np.linspace(.3, .9, 1002)
    socket = _FakeSocket([pack_graspgenx_message(
        {"grasps": grasps, "confidences": scores},
    )])
    client = GraspGenXClient(context=_FakeContext(socket))

    actual, confidence = client.infer_object(
        np.zeros((16, 3), dtype=np.float32),
        sweep_volume_params=DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
        top_k=-1,
    )

    assert len(actual) == len(confidence) == 1002
    np.testing.assert_allclose(actual, grasps[::-1])
    np.testing.assert_allclose(confidence, scores[::-1])
    # The reviewed service returns all candidates and rejects a wire top-k.
    assert "topk_num_grasps" not in unpack_graspgenx_message(socket.sent[0])


@pytest.mark.parametrize("top_k", [-2, 0, True, 1001])
@pytest.mark.parametrize("inprocess", [False, True])
def test_inference_rejects_invalid_top_k_before_model_or_transport(
    tmp_path: Path, top_k, inprocess: bool,
) -> None:
    backend = (GraspGenXInProcessBackend(model_spec=_model_spec(tmp_path))
               if inprocess else GraspGenXClient(context=_FakeContext(_FakeSocket([]))))
    with pytest.raises(ValueError, match="top_k"):
        backend.infer_object(
            np.zeros((16, 3), dtype=np.float32),
            sweep_volume_params=DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
            top_k=top_k,
        )


def test_moe_samples_five_mm_insertion_intervals_before_scoring() -> None:
    offsets = np.asarray(GRASPGENX_POINT_GRASP_MOE_Z_OFFSETS_CM)
    assert set((-8., -6., -4., -2., -1., 0.)) <= set(offsets)
    np.testing.assert_allclose(np.diff(offsets[offsets >= -4.]), .5)


def test_moe_samples_support_aware_side_heights_before_scoring() -> None:
    assert GRASPGENX_POINT_GRASP_MOE_SIDE_HEIGHT_OFFSETS_CM == (0.0, 1.0, 2.0)


def test_client_rejects_score_threshold_outside_confidence_range() -> None:
    client = GraspGenXClient(context=_FakeContext(_FakeSocket([])))

    with pytest.raises(ValueError, match="score_threshold"):
        client.infer_object(
            np.zeros((16, 3), dtype=np.float32),
            sweep_volume_params=DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
            score_threshold=1.01,
        )


def test_client_rejects_non_rigid_server_pose() -> None:
    malformed = np.eye(4, dtype=np.float32)
    malformed[0, 0] = 2.0
    socket = _FakeSocket(
        [
            pack_graspgenx_message(
                {
                    "grasps": malformed[None],
                    "confidences": np.asarray((0.5,), dtype=np.float32),
                }
            )
        ]
    )
    client = GraspGenXClient(context=_FakeContext(socket))

    with pytest.raises(RuntimeError, match="non-rigid"):
        client.infer_object(
            np.zeros((16, 3), dtype=np.float32),
            sweep_volume_params=DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
        )


def test_model_spec_verifies_all_four_checkpoint_files(tmp_path: Path) -> None:
    bundle = _asset_bundle(tmp_path)
    spec = GraspGenXPointGraspModelSpec.from_asset_bundle(bundle)

    assert spec.source_revision == GRASPGENX_REVIEWED_REVISION
    assert spec.checkpoint_revision == GRASPGENX_CHECKPOINT_REVIEWED_REVISION
    assert spec.checkpoint_root == bundle.resolve() / "checkpoints/release"
    assert len(spec.semantic_content_hash) == 64

    checkpoint = bundle / "checkpoints/release/gen/epoch_736.pth"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="generator_checkpoint hash mismatch"):
        GraspGenXPointGraspModelSpec.from_asset_bundle(bundle)


def test_planner_rejects_non_finite_score_threshold(tmp_path: Path) -> None:
    mask = np.ones((4, 4), dtype=bool)

    with pytest.raises(ValueError, match="score_threshold"):
        Sam2GraspGenXPointGraspPlanner(
            model_spec=_model_spec(tmp_path),
            point_segmenter=_Segmenter(mask),
            client=_PlannerClient(
                np.eye(4, dtype=np.float64)[None], np.asarray((0.9,))
            ),
            score_threshold=float("nan"),
        )


@pytest.mark.parametrize("cad_bound", [False, True])
@pytest.mark.parametrize("base_tilt_degrees", [0, -30])
def test_planner_infers_in_base_frame_and_converts_pose_to_calibrated_dmtac_tcp(
    tmp_path: Path,
    cad_bound: bool,
    base_tilt_degrees: float,
) -> None:
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[8:13, 8:13] = 1.0
    mask = depth > 0.0
    camera_from_model_base = np.eye(4, dtype=np.float64)
    camera_from_model_base[2, 3] = 0.975
    base_from_camera = np.eye(4, dtype=np.float64)
    base_from_camera[:3, :3] = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    tilt = math.radians(base_tilt_degrees)
    base_from_camera[:3, :3] = base_from_camera[:3, :3] @ np.asarray(
        ((1, 0, 0), (0, math.cos(tilt), -math.sin(tilt)),
         (0, math.sin(tilt), math.cos(tilt)))
    )
    base_from_camera[:3, 3] = (0.20, -0.10, 0.30)
    base_from_model_base = base_from_camera @ camera_from_model_base
    client = _PlannerClient(
        base_from_model_base[None], np.asarray((0.9,), dtype=np.float64)
    )
    planner = Sam2GraspGenXPointGraspPlanner(
        model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(mask),
        client=client,
        solid_geometry=reviewed_graspgenx_dmtac_solid_geometry() if cad_bound else None,
    )
    planner.start()

    result = planner.plan(
        Image.new("RGB", (20, 20)),
        depth,
        pixel_uv=(10, 10),
        intrinsics={
            "width": 20,
            "height": 20,
            "fx": 100.0,
            "fy": 100.0,
            "cx": 10.0,
            "cy": 10.0,
        },
        base_from_camera=base_from_camera,
        max_candidates=1,
    )

    assert client.started
    assert len(client.calls) == 1
    assert client.calls[0][0].shape == (25, 3)
    expected_camera_points = np.asarray(
        [
            ((u - 10.0) / 100.0, (v - 10.0) / 100.0, 1.0)
            for v in range(8, 13)
            for u in range(8, 13)
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        client.calls[0][0],
        expected_camera_points @ base_from_camera[:3, :3].T
        + base_from_camera[:3, 3],
        atol=1e-6,
    )
    assert client.calls[0][1]["planner"] == "graspmoe"
    assert client.calls[0][1]["top_k"] == -1
    candidate = result.candidates[0]
    camera_from_tcp = np.asarray(candidate.camera_from_tcp).reshape(4, 4)
    camera_from_geometry = planner.solid_geometry.solid_pose(camera_from_tcp)
    expected_geometry = camera_from_model_base.copy()
    np.testing.assert_allclose(camera_from_geometry, expected_geometry, atol=1e-9)
    expected_tcp = expected_geometry @ np.linalg.inv(
        planner.solid_geometry.solid_pose(np.eye(4))
    )
    np.testing.assert_allclose(camera_from_tcp, expected_tcp, atol=1e-9)
    np.testing.assert_allclose(np.asarray(candidate.base_from_tcp).reshape(4, 4),
                               base_from_camera @ camera_from_tcp, atol=1e-9)
    assert candidate.model_score == pytest.approx(.9)
    assert 0.005 <= candidate.gripper_width_m <= 0.0705
    assert candidate.visible_collision_free is True
    assert planner.last_target_mask is not None
    assert "graspgenx_inference" in planner.last_plan_timing_ms
    assert planner.last_filter_summary["detected"] == 1
    assert planner.last_filter_summary["width_valid"] == 1
    assert planner.last_filter_summary["target_capture_valid"] == 1

    planner.close()
    assert client.closed_for_test


@pytest.mark.parametrize("cad_bound", [False, True])
def test_planner_preserves_scored_finger_exchange_for_independent_robot_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cad_bound: bool,
) -> None:
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[8:13, 8:13] = 1.0
    mask = depth > 0.0
    grasps = []
    for degrees in (0.0, 30.0, 60.0, 180.0):
        radians = math.radians(degrees)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.asarray(
            (
                (math.cos(radians), -math.sin(radians), 0.0),
                (math.sin(radians), math.cos(radians), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        pose[2, 3] = 0.975
        grasps.append(pose)
    tilted = np.eye(4, dtype=np.float64)
    tilted_angle = math.radians(30.0)
    tilted[:3, :3] = np.asarray(
        (
            (math.cos(tilted_angle), 0.0, math.sin(tilted_angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(tilted_angle), 0.0, math.cos(tilted_angle)),
        )
    )
    tilted[2, 3] = 0.975
    # A scored pose must already put the object in its opening. Do not rely
    # on production code silently translating this tilted fixture afterwards.
    tilted[:3, 3] = np.asarray((0, 0, 1.0)) - 0.025 * tilted[:3, 2]
    grasps.append(tilted)
    client = _PlannerClient(
        np.stack(grasps), np.asarray((0.9, 0.8, 0.7, 0.65, 0.6), dtype=np.float64)
    )
    planner = Sam2GraspGenXPointGraspPlanner(
        model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(mask),
        client=client,
        solid_geometry=reviewed_graspgenx_dmtac_solid_geometry() if cad_bound else None,
    )
    monkeypatch.setattr(
        Sam2VgnPointGraspPlanner,
        "_visible_collision_free",
        staticmethod(lambda *_args, **_kwargs: True),
    )
    planner.start()

    result = planner.plan(
        Image.new("RGB", (20, 20)),
        depth,
        pixel_uv=(10, 10),
        intrinsics={
            "width": 20,
            "height": 20,
            "fx": 200.0,
            "fy": 200.0,
            "cx": 10.0,
            "cy": 10.0,
        },
        base_from_camera=np.eye(4),
        max_candidates=5,
    )

    assert [candidate.model_score for candidate in result.candidates] == pytest.approx(
        [0.9, 0.8, 0.7, 0.65, 0.6]
    )
    for candidate, scored_pose in zip(result.candidates, grasps, strict=True):
        expected = scored_pose.copy()
        actual_tcp = np.asarray(candidate.camera_from_tcp).reshape(4, 4)
        np.testing.assert_allclose(planner.solid_geometry.solid_pose(actual_tcp), expected, atol=1e-9)
    exchanged = np.asarray(result.candidates[3].camera_from_tcp).reshape(4, 4)
    first = np.asarray(result.candidates[0].camera_from_tcp).reshape(4, 4)
    first_model = planner.solid_geometry.solid_pose(first)
    exchanged_model = planner.solid_geometry.solid_pose(exchanged)
    assert np.dot(first_model[:3, 0], exchanged_model[:3, 0]) == pytest.approx(-1)
    expected_first_tcp = grasps[0] @ np.linalg.inv(planner.solid_geometry.solid_pose(np.eye(4)))
    expected_exchanged_tcp = grasps[3] @ np.linalg.inv(planner.solid_geometry.solid_pose(np.eye(4)))
    np.testing.assert_allclose(first, expected_first_tcp, atol=1e-9)
    np.testing.assert_allclose(exchanged, expected_exchanged_tcp, atol=1e-9)
    tilted_result = np.asarray(result.candidates[-1].camera_from_tcp).reshape(4, 4)
    tilted_model = planner.solid_geometry.solid_pose(tilted_result)
    assert np.dot(first_model[:3, 2], tilted_model[:3, 2]) == pytest.approx(
        math.cos(tilted_angle)
    )
    planner.close()


def test_candidate_selection_balances_approaches_and_closing_axis_variants() -> None:
    poses = []
    for yaw_degrees in (0.0, 30.0, 60.0, 90.0, 120.0):
        yaw = math.radians(yaw_degrees)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.asarray(
            (
                (math.cos(yaw), -math.sin(yaw), 0.0),
                (math.sin(yaw), math.cos(yaw), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        poses.append(pose)
    for tilt_degrees in (30.0, -30.0, 60.0):
        tilt = math.radians(tilt_degrees)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.asarray(
            (
                (math.cos(tilt), 0.0, math.sin(tilt)),
                (0.0, 1.0, 0.0),
                (-math.sin(tilt), 0.0, math.cos(tilt)),
            )
        )
        poses.append(pose)

    representatives, selected = (
        Sam2GraspGenXPointGraspPlanner._distinct_approach_first_indices(
            tuple(poses),
            maximum=4,
        )
    )

    assert representatives == [0, 5, 6, 7]
    assert selected == [0, 5, 6, 7]


@pytest.mark.parametrize("maximum", [2, 4, 5])
@pytest.mark.parametrize("cad_bound", [False, True])
def test_planner_retains_scored_depth_variants_after_family_priority(
    tmp_path: Path, maximum: int, cad_bound: bool,
) -> None:
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[8:13, 8:13] = 1.
    poses = []
    for yaw, z in ((0, .975), (0, .975), (0, .980),
                   (math.pi, .975), (math.pi, .980)):
        pose = np.eye(4)
        pose[:3, :3] = ((math.cos(yaw), -math.sin(yaw), 0),
                       (math.sin(yaw), math.cos(yaw), 0), (0, 0, 1))
        pose[2, 3] = z
        poses.append(pose)
    planner = Sam2GraspGenXPointGraspPlanner(
        model_spec=_model_spec(tmp_path), point_segmenter=_Segmenter(depth > 0),
        client=_PlannerClient(np.stack(poses), np.asarray((.9, .89, .8, .7, .6))),
        solid_geometry=reviewed_graspgenx_dmtac_solid_geometry() if cad_bound else None,
    )
    planner.start()
    result = planner.plan(
        Image.new("RGB", (20, 20)), depth, pixel_uv=(10, 10),
        intrinsics=dict(width=20, height=20, fx=200., fy=200., cx=10., cy=10.),
        base_from_camera=np.eye(4), max_candidates=maximum,
    )
    expected_indices = (
        [0, 3]
        if maximum == 2
        else [0, 2, 3, 4]
        if maximum == 4
        else [0, 2, 2, 3, 4]
    )
    assert len(result.candidates) == len(expected_indices)
    for ordinal, (candidate, index) in enumerate(
        zip(result.candidates, expected_indices, strict=True)
    ):
        expected = poses[index].copy()
        if maximum == 5 and ordinal == 1:
            # The fifth slot intentionally keeps the smallest support-clearance
            # alternative before the remaining scored depth variants.
            expected[2, 3] += 0.005
            assert candidate.pose_source == "dmtac_support_clearance_variant_v1"
        actual = planner.solid_geometry.solid_pose(
            np.asarray(candidate.camera_from_tcp).reshape(4, 4),
        )
        np.testing.assert_allclose(actual, expected, atol=1e-9)
        assert candidate.model_score == pytest.approx((.9, .89, .8, .7, .6)[index])
        assert candidate.visible_collision_free
    assert planner.last_filter_summary["candidate_family_count"] == 2
    assert planner.last_filter_summary["candidate_variant_count"] == (
        5 if maximum == 5 else 4
    )
    planner.close()


@pytest.mark.parametrize("yaw_degrees,same_family", [(0, True), (10, True), (30, False), (180, False)])
def test_candidate_family_deduplicates_nearby_orientations_not_finger_exchange(yaw_degrees, same_family):
    first = np.eye(4)
    second = np.eye(4)
    angle = math.radians(yaw_degrees)
    second[:3, :3] = np.asarray(
        ((math.cos(angle), -math.sin(angle), 0),
         (math.sin(angle), math.cos(angle), 0), (0, 0, 1))
    )
    assert Sam2GraspGenXPointGraspPlanner._same_candidate_family(first, second) is same_family


def test_planner_does_not_translate_scored_offset_grasp_to_hide_solid_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[8:13, 8:13] = 1.0
    mask = depth > 0.0
    grasp = np.eye(4, dtype=np.float64)
    grasp[0, 3] = -0.020
    grasp[1, 3] = -0.012
    grasp[2, 3] = 0.975
    client = _PlannerClient(grasp[None], np.asarray((0.9,), dtype=np.float64))
    planner = Sam2GraspGenXPointGraspPlanner(
        model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(mask),
        client=client,
    )
    monkeypatch.setattr(
        Sam2VgnPointGraspPlanner,
        "_visible_collision_free",
        staticmethod(lambda *_args, **_kwargs: True),
    )
    planner.start()

    with pytest.raises(PointGraspPlanningError, match="physical_filtering"):
        planner.plan(
            Image.new("RGB", (20, 20)),
            depth,
            pixel_uv=(10, 10),
            intrinsics={
                "width": 20, "height": 20,
                "fx": 100.0, "fy": 100.0, "cx": 10.0, "cy": 10.0,
            },
            base_from_camera=np.eye(4),
            max_candidates=1,
        )

    # The candidate is rejected before collision because the complete target
    # span no longer fits between its proposed jaw faces.  This is the same
    # no-translation invariant at the earlier, more informative boundary.
    assert planner.last_filter_summary["target_capture_valid"] == 0
    assert planner.last_filter_summary["target_solid_collision_rejected"] == 0
    assert planner.last_filter_summary["min_target_capture_violation_m"] > 0
    assert grasp[:3, 3] == pytest.approx((-0.020, -0.012, 0.975))
    planner.close()


def test_planner_normalizes_float32_model_rotation_for_strict_frame_contract(
    tmp_path: Path,
) -> None:
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[8:13, 8:13] = 1.0
    mask = depth > 0.0
    angle = np.float32(np.deg2rad(10.0))
    cosine = np.float32(np.cos(angle))
    sine = np.float32(np.sin(angle))
    base_from_model_base = np.eye(4, dtype=np.float64)
    base_from_model_base[:3, :3] = np.asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    base_from_model_base[2, 3] = 0.975
    original_rotation = base_from_model_base[:3, :3].copy()
    assert np.max(
        np.abs(original_rotation.T @ original_rotation - np.eye(3))
    ) == pytest.approx(4.555837662678641e-08)
    planner = Sam2GraspGenXPointGraspPlanner(
        model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(mask),
        client=_PlannerClient(
            base_from_model_base[None], np.asarray((0.9,), dtype=np.float64)
        ),
    )
    planner.start()

    result = planner.plan(
        Image.new("RGB", (20, 20)),
        depth,
        pixel_uv=(10, 10),
        intrinsics={
            "width": 20,
            "height": 20,
            "fx": 100.0,
            "fy": 100.0,
            "cx": 10.0,
            "cy": 10.0,
        },
        base_from_camera=np.eye(4),
        max_candidates=1,
    )

    candidate = result.candidates[0]
    for values in (candidate.camera_from_tcp, candidate.base_from_tcp):
        pose = np.asarray(values, dtype=np.float64).reshape(4, 4)
        rotation = pose[:3, :3]
        np.testing.assert_allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-9,
            rtol=0.0,
        )
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)
        model_pose = planner.solid_geometry.solid_pose(pose)
        np.testing.assert_allclose(
            model_pose[:3, :3], original_rotation, atol=3e-8, rtol=0.0
        )
    planner.close()


def test_visible_collision_proxy_uses_open_gripper_and_keeps_ingress_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth = np.zeros((80, 80), dtype=np.float32)
    mask = np.zeros(depth.shape, dtype=bool)
    mask[38:43, 28:53] = True
    mask[40, 62] = True
    depth[mask] = 1.0
    model_pose = np.eye(4)
    model_pose[2, 3] = .975
    planner = Sam2GraspGenXPointGraspPlanner(
        model_spec=_model_spec(tmp_path), point_segmenter=_Segmenter(mask),
        client=_PlannerClient(model_pose[None], np.asarray([.9])),
    )
    endpoint_openings = []
    approach_openings = []

    def _collision_with_terminal_contact(
        _scene, _pose, *, opening_m, contact_depth_m,
    ):
        endpoint_openings.append(float(opening_m))
        assert contact_depth_m == pytest.approx(.006)
        return 0, 0, 0

    def _approach(
        _scene, _pose, *, opening_m, distance_m,
        allow_terminal_target_contact, terminal_target_contact_depth_m,
    ):
        approach_openings.append((float(opening_m), float(distance_m)))
        assert allow_terminal_target_contact is True
        assert terminal_target_contact_depth_m == pytest.approx(.006)
        return 1, 1

    monkeypatch.setattr(
        VisibleGraspScene,
        "collision_counts_with_terminal_target_contact",
        _collision_with_terminal_contact,
    )
    monkeypatch.setattr(
        VisibleGraspScene, "approach_collision_counts", _approach
    )
    planner.start()
    try:
        result = planner.plan(Image.new("RGB", (80, 80)), depth, pixel_uv=(40, 40),
            intrinsics={"width": 80, "height": 80, "fx": 1000., "fy": 1000., "cx": 40., "cy": 40.},
            base_from_camera=np.eye(4), max_candidates=1)
        candidate = result.candidates[0]
        expected_opening = planner.max_gripper_width_m
        assert endpoint_openings == pytest.approx([expected_opening])
        assert approach_openings == pytest.approx([(expected_opening, .06)])
        assert candidate.visible_collision_free
        assert candidate.model_score == pytest.approx(.9)
        pose = np.asarray(candidate.camera_from_tcp).reshape(4, 4)
        np.testing.assert_allclose(
            planner.solid_geometry.solid_pose(pose), model_pose, atol=1e-9
        )
    finally:
        planner.close()


def test_capture_filtered_visible_scene_prefilters_terminal_background_before_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth = np.zeros((40, 40), dtype=np.float32)
    mask = np.zeros(depth.shape, dtype=bool)
    mask[18:23, 18:23] = True
    depth[mask] = 1.0
    poses = []
    for x in (0.0, 0.002):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = x
        pose[2, 3] = 0.975
        poses.append(pose)
    planner = Sam2GraspGenXPointGraspPlanner(
        model_spec=_model_spec(tmp_path), point_segmenter=_Segmenter(mask),
        client=_PlannerClient(
            np.stack(poses), np.asarray((.9, .8), dtype=np.float64)
        ),
    )
    endpoint_scene_sizes = []

    def _terminal(scene, pose, *, opening_m, contact_depth_m):
        del opening_m, contact_depth_m
        endpoint_scene_sizes.append(len(scene.points))
        if len(scene.points) == 2 and float(pose[0, 3]) < 0.001:
            return 0, 0, 1
        return 0, 0, 0

    monkeypatch.setattr(
        VisibleGraspScene,
        "collision_counts_with_terminal_target_contact",
        _terminal,
    )
    monkeypatch.setattr(
        VisibleGraspScene,
        "approach_collision_counts",
        lambda *_args, **_kwargs: (0, 0),
    )

    factory_calls = []

    def _factory(*, depth_m, target_mask, intrinsics, geometry):
        del depth_m, intrinsics
        factory_calls.append(target_mask.copy())
        return VisibleGraspScene(
            points=np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
            target_mask=np.asarray((True, False), dtype=np.bool_),
            geometry=geometry,
        )

    planner.start()
    try:
        result = planner.plan(
            Image.new("RGB", (40, 40)), depth, pixel_uv=(20, 20),
            intrinsics={
                "width": 40, "height": 40,
                "fx": 100.0, "fy": 100.0, "cx": 20.0, "cy": 20.0,
            },
            base_from_camera=np.eye(4), max_candidates=1,
            visible_scene_factory=_factory,
        )
        assert len(factory_calls) == 1
        np.testing.assert_array_equal(factory_calls[0], mask)
        assert endpoint_scene_sizes == [2, 2]
        assert len(result.candidates) == 1
        assert result.candidates[0].model_score == pytest.approx(.8)
        assert planner.last_filter_summary["terminal_background_prefilter_enabled"]
        assert planner.last_filter_summary["terminal_background_collision_rejected"] == 1
        assert planner.last_filter_summary["visible_scene_point_count"] == 2
        assert planner.last_filter_summary["visible_scene_background_point_count"] == 1
    finally:
        planner.close()


def test_multiview_target_points_are_the_actual_single_inference_input(tmp_path, monkeypatch):
    depth = np.zeros((40, 40), np.float32)
    mask = np.zeros((40, 40), bool)
    mask[18:23, 18:23] = True
    depth[mask] = 1.
    pose = np.eye(4)
    pose[2, 3] = .975
    client = _PlannerClient(pose[None], np.array([.9]))
    planner = Sam2GraspGenXPointGraspPlanner(model_spec=_model_spec(tmp_path), point_segmenter=_Segmenter(mask), client=client)
    # Include real metric samples absent from the anchor depth image.
    fused = np.array([(x*.01, y*.01, 1.) for x in range(-2, 3) for y in range(-2, 3)] + [(.007, .013, 1.008)])
    monkeypatch.setattr(VisibleGraspScene, "collision_counts_with_terminal_target_contact", lambda *a, **kw: (0, 0, 0))
    monkeypatch.setattr(VisibleGraspScene, "approach_collision_counts", lambda *a, **kw: (0, 0))
    planner.start()
    try:
        planner.plan(Image.new("RGB", (40, 40)), depth, pixel_uv=(20, 20),
            intrinsics={"width": 40, "height": 40, "fx": 100., "fy": 100., "cx": 20., "cy": 20.},
            base_from_camera=np.eye(4), max_candidates=1, use_visible_target_points=True,
            visible_scene_factory=lambda **kw: VisibleGraspScene(points=fused, target_mask=np.ones(len(fused), bool), geometry=kw["geometry"]))
        assert len(client.calls) == 1
        np.testing.assert_allclose(client.calls[0][0], fused)
    finally:
        planner.close()


def test_planner_fails_closed_when_model_pose_misses_clicked_object(
    tmp_path: Path,
) -> None:
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[8:13, 8:13] = 1.0
    mask = depth > 0.0
    missed = np.eye(4, dtype=np.float64)
    missed[:3, 3] = (0.20, 0.0, 0.975)
    planner = Sam2GraspGenXPointGraspPlanner(
        model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(mask),
        client=_PlannerClient(missed[None], np.asarray((0.9,))),
    )
    planner.start()

    with pytest.raises(PointGraspPlanningError) as raised:
        planner.plan(
            Image.new("RGB", (20, 20)),
            depth,
            pixel_uv=(10, 10),
            intrinsics={
                "width": 20,
                "height": 20,
                "fx": 100.0,
                "fy": 100.0,
                "cx": 10.0,
                "cy": 10.0,
            },
            base_from_camera=np.eye(4),
            max_candidates=1,
        )
    assert raised.value.diagnostic["stage"] == "physical_filtering"
    assert raised.value.diagnostic["counts"]["target_capture_valid"] == 0


def test_batched_pose_dedup_preserves_greedy_order_and_original_objects():
    rng = np.random.default_rng(19)
    poses = []
    for _ in range(100):
        pose = np.eye(4)
        pose[:3, 3] = rng.uniform(-.2, .2, 3)
        poses.extend((pose, pose.copy()))
        shifted = pose.copy()
        shifted[:3, 3] += rng.uniform(-1.5e-6, 1.5e-6, 3)
        poses.append(shifted)
    # Near-duplicate matching is intentionally not transitive.
    for x in (0., .75e-6, 1.5e-6):
        pose = np.eye(4)
        pose[0, 3] = x
        poses.append(pose)
    candidates = [_RankedGraspCandidate(score=1.-index*.001, model_score=.9,
        click_distance_m=.01, gripper_width_m=.05,
        camera_from_grasp_geometry=pose, base_from_grasp_geometry=pose)
        for index, pose in enumerate(poses)]
    expected = []
    for candidate in candidates:
        if not any(np.allclose(candidate.camera_from_grasp_geometry,
            previous.camera_from_grasp_geometry, atol=1e-6, rtol=0.) for previous in expected):
            expected.append(candidate)
    actual = _deduplicate_ranked_poses(candidates)
    assert [id(item) for item in actual] == [id(item) for item in expected]
    assert actual[-2] is candidates[-3] and actual[-1] is candidates[-1]
    assert _deduplicate_ranked_poses([]) == []


@pytest.mark.parametrize("offset", [0., np.nextafter(1e-6, 0.), 1e-6, np.nextafter(1e-6, np.inf)])
@pytest.mark.parametrize("component", ["translation", "rotation"])
def test_pose_duplicate_comparison_matches_componentwise_threshold(offset, component):
    first, second = np.eye(4), np.eye(4)
    if component == "translation":
        second[:3, 3] = offset
    else:
        second[:2, :2] = ((math.cos(offset), -math.sin(offset)),
            (math.sin(offset), math.cos(offset)))
    expected = bool(np.allclose(first, second, rtol=0., atol=1e-6))
    assert _pose_is_duplicate(first, [second]) is expected
    assert _pose_is_duplicate(first, np.asarray([second])) is expected
    assert _pose_is_duplicate(first, []) is False


def test_selector_planning_priority_survives_public_score_order_without_changing_poses(tmp_path):
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[8:13, 8:13] = 1.
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 2, 3] = .975
    poses[:, 0, 3] = (-.004, 0., .004)
    planner = Sam2GraspGenXPointGraspPlanner(model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(depth > 0), client=_PlannerClient(poses, np.array([.9, .8, .7])))
    selected = []

    def select(pool, _target, _maximum):
        indices = tuple(next(index for index, candidate in enumerate(pool)
            if candidate.pose_source == "model_scored" and candidate.model_score == score)
            for score in (.9, .7))
        selected.extend(pool[index] for index in indices)
        return GraspGenXCandidateSelection(indices[::-1], {})

    planner.start()
    options = {"pixel_uv": (10, 10), "intrinsics": {"width": 20, "height": 20,
        "fx": 100., "fy": 100., "cx": 10., "cy": 10.}, "base_from_camera": np.eye(4), "max_candidates": 2}
    image = Image.new("RGB", (20, 20))
    try:
        result = planner.plan(image, depth, candidate_selector=select, **options)
        assert [c.model_score for c in result.candidates] == [.9, .7]
        assert result.candidates[0].score > result.candidates[1].score
        for actual, expected in zip(result.candidates, selected, strict=True):
            assert actual.base_from_tcp == expected.base_from_tcp
            assert actual.camera_from_tcp == expected.camera_from_tcp
        assert grasp_planning_candidate_ordinals(planner.last_filter_summary, candidate_count=2) == (1, 0)
        # A later unembodied request cannot inherit the previous recommendation.
        planner.plan(image, depth, **options)
        assert grasp_planning_candidate_ordinals(planner.last_filter_summary, candidate_count=2) is None
    finally:
        planner.close()


@pytest.mark.parametrize("encoded", ('[0,0]', '[0]', '[0,2]', '[false,1]', 'null', '{}', 'not-json'))
def test_present_corrupt_planning_priority_is_not_silently_ignored(encoded):
    with pytest.raises(ValueError):
        grasp_planning_candidate_ordinals({GRASP_PLANNING_ORDER_FIELD: encoded,
            "arm_shortlist_planning_order_scope": GRASP_PLANNING_ORDER_SCOPE}, candidate_count=2)


def test_planning_priority_is_advisory_and_optional_for_existing_producers():
    assert grasp_planning_candidate_ordinals({}, candidate_count=2) is None
    with pytest.raises(ValueError, match="scope"):
        grasp_planning_candidate_ordinals({GRASP_PLANNING_ORDER_FIELD: '[1,0]',
            "arm_shortlist_planning_order_scope": "full_path_validated"}, candidate_count=2)
    with pytest.raises(TypeError, match="JSON text"):
        grasp_planning_candidate_ordinals({GRASP_PLANNING_ORDER_FIELD: [1, 0],
            "arm_shortlist_planning_order_scope": GRASP_PLANNING_ORDER_SCOPE}, candidate_count=2)
    for count in (-1, True, 1.5):
        with pytest.raises(ValueError, match="candidate_count"):
            grasp_planning_candidate_ordinals({}, candidate_count=count)


def test_grasp_timing_separates_arm_selection_and_preserves_failed_stage(tmp_path, monkeypatch):
    import time

    from deployments.perception.sam2_graspgenx_point_grasp import (
        GraspGenXCandidateSelection,
    )

    depth = np.zeros((20, 20), dtype=np.float32)
    depth[8:13, 8:13] = 1.
    pose = np.eye(4)
    pose[2, 3] = .975
    planner = Sam2GraspGenXPointGraspPlanner(model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(depth > 0), client=_PlannerClient(pose[None], np.array([.9])))
    planner.start()
    options = {"pixel_uv": (10, 10), "intrinsics": {"width": 20, "height": 20,
        "fx": 100., "fy": 100., "cx": 10., "cy": 10.}, "base_from_camera": np.eye(4), "max_candidates": 1}

    def select(*_):
        time.sleep(.02)
        return GraspGenXCandidateSelection((0,), {})

    image = Image.new("RGB", (20, 20))
    try:
        planner.plan(image, depth, candidate_selector=select, **options)
        timings = planner.last_plan_timing_ms
        assert timings["arm_candidate_selection"] >= 15.
        assert timings["candidate_ranking_and_visible_collision"] == pytest.approx(
            timings["arm_candidate_selection"] + timings["candidate_filtering_excluding_arm_selection"], abs=1.)

        def fail(*_, **__):
            raise RuntimeError("selection_failed")

        with pytest.raises(RuntimeError, match="selection_failed"):
            planner.plan(image, depth, candidate_selector=fail, **options)
        assert planner.last_plan_timing_ms["arm_candidate_selection"] < timings["arm_candidate_selection"]
        assert "graspgenx_inference" in planner.last_plan_timing_ms
        monkeypatch.setattr(planner.point_segmenter, "segment_point", fail)
        with pytest.raises(RuntimeError, match="selection_failed"):
            planner.plan(image, depth, **options)
        assert set(planner.last_plan_timing_ms) == {"sam2_inference", "plan_total"}
    finally:
        planner.close()

    with pytest.raises(RuntimeError, match="not running"):
        planner.plan(image, depth, **options)
    assert planner.last_plan_timing_ms == {}


def _shared_proposal_fixture(tmp_path, **planner_options):
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[8:13, 8:13] = 1.
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 2, 3] = .975
    poses[:, 0, 3] = (-.004, 0., .004)
    client = _PlannerClient(poses, np.array([.9, .8, .7]))
    segmenter = _Segmenter(depth > 0)
    planner = Sam2GraspGenXPointGraspPlanner(model_spec=_model_spec(tmp_path),
        point_segmenter=segmenter, client=client, **planner_options)
    options = {"pixel_uv": (10, 10), "intrinsics": {"width": 20, "height": 20,
        "fx": 100., "fy": 100., "cx": 10., "cy": 10.},
        "base_from_camera": np.eye(4), "max_candidates": 2}
    return planner, Image.new("RGB", (20, 20)), depth, options


def test_components_share_exact_pool_and_perception_once(tmp_path, monkeypatch):
    planner, image, depth, options = _shared_proposal_fixture(tmp_path, enable_finger_exchange=True)
    calls, pools = [], []
    segment = planner.point_segmenter.segment_point
    rank = planner._rank_pool

    def record_segment(*args, **kwargs):
        calls.append("segment")
        return segment(*args, **kwargs)

    def record_rank(*args, **kwargs):
        calls.append("rank")
        return rank(*args, **kwargs)

    def select(pool, target, maximum):
        pools.append(pool)
        assert target == (0., 0., 1.) and maximum == 2
        return GraspGenXCandidateSelection((2, 0), {"arm_shortlist_attempts": 2})

    def empty(pool, target, maximum):
        pools.append(pool)
        return GraspGenXCandidateSelection((), {"arm_shortlist_attempts": 3})

    monkeypatch.setattr(planner.point_segmenter, "segment_point", record_segment)
    monkeypatch.setattr(planner, "_rank_pool", record_rank)
    planner.start()
    try:
        result = planner.plan_for_components(image, depth,
            candidate_selectors={"left_arm": select, "right_arm": empty}, **options)
        assert calls == ["segment", "rank"] and len(planner.client.calls) == 1
        assert pools[0] is pools[1] is result.candidate_pool
        assert len(result.results_by_component["left_arm"].candidates) == 2
        assert result.results_by_component["right_arm"].candidates == ()
        assert grasp_planning_candidate_ordinals(result.filter_summary_by_component["left_arm"], candidate_count=2) == (1, 0)
        assert grasp_planning_candidate_ordinals(result.filter_summary_by_component["right_arm"], candidate_count=0) == ()
        assert GRASP_PLANNING_ORDER_FIELD not in planner.last_filter_summary
        assert "selected" not in planner.last_filter_summary
        assert set(result.selection_timing_ms_by_component) == {"left_arm", "right_arm"}
        with pytest.raises(TypeError):
            result.results_by_component["right_arm"] = result.results_by_component["left_arm"]
        with pytest.raises(TypeError):
            result.filter_summary_by_component["left_arm"]["selected"] = 7
        for component in result.results_by_component.values():
            assert component.target_point_camera_m == (0., 0., 1.)
            assert component.target_mask_bbox_xyxy == (8, 8, 13, 13)
            assert all(not hasattr(candidate, "symmetry_parent_index") for candidate in component.candidates)
    finally:
        planner.close()


def test_single_component_matches_existing_explicit_arm_request(tmp_path):
    planner, image, depth, options = _shared_proposal_fixture(tmp_path,
        preferred_approach_origin_base_m=(-.2, 0., 0.))
    def select(pool, target, maximum):
        return GraspGenXCandidateSelection((1, 0), {"arm_shortlist_attempts": 2})
    planner.start()
    try:
        single = planner.plan(image, depth, candidate_selector=select, **options)
        summary = planner.last_filter_summary
        shared = planner.plan_for_components(image, depth, candidate_selectors={"right_arm": select}, **options)
        assert shared.results_by_component["right_arm"] == single
        assert dict(shared.filter_summary_by_component["right_arm"]) == summary
    finally:
        planner.close()


@pytest.mark.parametrize("selectors", [{}, {"": lambda *_: None}, {"left": None},
    {"left": lambda *_: None, "right": lambda *_: None, "third": lambda *_: None}])
def test_invalid_components_fail_before_inference(tmp_path, selectors):
    planner, image, depth, options = _shared_proposal_fixture(tmp_path)
    planner.start()
    try:
        with pytest.raises((ValueError, TypeError)):
            planner.plan_for_components(image, depth, candidate_selectors=selectors, **options)
        assert not planner.client.calls
    finally:
        planner.close()


@pytest.mark.parametrize("indices", [(0, 0), (True,), (-1,), (999,), (0, 1, 2), ([0],)])
def test_bad_component_admission_does_not_publish_partial_batch(tmp_path, indices):
    planner, image, depth, options = _shared_proposal_fixture(tmp_path)
    planner.start()
    try:
        with pytest.raises(ValueError, match="invalid indices"):
            planner.plan_for_components(image, depth, candidate_selectors={
                "left_arm": lambda *_: GraspGenXCandidateSelection((0,), {}),
                "right_arm": lambda *_: GraspGenXCandidateSelection(indices, {})}, **options)
        assert "arm_candidate_selection" in planner.last_plan_timing_ms
    finally:
        planner.close()


def test_no_arm_admitted_is_distinct_from_no_physical_proposals(tmp_path):
    planner, image, depth, options = _shared_proposal_fixture(tmp_path)
    calls = []
    def select(*_):
        calls.append(1)
        return GraspGenXCandidateSelection((), {})
    planner.start()
    try:
        result = planner.plan_for_components(image, depth,
            candidate_selectors={"left_arm": select, "right_arm": select}, **options)
        assert result.candidate_pool and all(not view.candidates for view in result.results_by_component.values())
        planner.client.grasps = np.empty((0, 4, 4))
        planner.client.scores = np.empty((0,))
        with pytest.raises(PointGraspPlanningError, match="physical_filtering"):
            planner.plan_for_components(image, depth, candidate_selectors={"left_arm": select}, **options)
        assert len(calls) == 2
    finally:
        planner.close()


def test_multiple_origins_preserve_single_origin_identity_and_validate_inputs(tmp_path):
    single, *_ = _shared_proposal_fixture(tmp_path, preferred_approach_origin_base_m=(-.2, 0., 0.))
    equivalent, *_ = _shared_proposal_fixture(tmp_path,
        preferred_approach_origins_base_m=((- .2, 0., 0.),))
    assert single.semantic_content_hash == equivalent.semantic_content_hash
    dual, *_ = _shared_proposal_fixture(tmp_path,
        preferred_approach_origins_base_m=((- .2, 0., 0.), (.2, 0., 0.)))
    assert dual.semantic_content_hash != single.semantic_content_hash
    for origins in (((0., 0.),), ((float("nan"), 0., 0.),), ((0., 0., 0.),)*3):
        with pytest.raises(ValueError):
            _shared_proposal_fixture(tmp_path, preferred_approach_origins_base_m=origins)
    with pytest.raises(ValueError, match="not both"):
        _shared_proposal_fixture(tmp_path, preferred_approach_origin_base_m=(0., 0., 0.),
            preferred_approach_origins_base_m=((1., 0., 0.),))


from deployments.perception.sam2_graspgenx_point_grasp import (
    DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
    GRASPGENX_CHECKPOINT_REVIEWED_REVISION,
    GRASPGENX_REVIEWED_REVISION,
    GraspGenXCandidateSelection,
    GraspGenXPointGraspModelSpec,
    Sam2GraspGenXPointGraspPlanner,
    _RankedGraspCandidate,
    _deduplicate_ranked_poses,
    _pad_height_shifts,
    _pose_is_duplicate,
)
from deployments.perception.point_grasp_failure import PointGraspPlanningError
from robot_motion.grasp_visible_scene import VisibleGraspScene, parallel_jaw_pad_height_support

@pytest.mark.parametrize('refine', [False, True])
def test_capture_support_is_available_to_selector_but_not_public_candidates(tmp_path, refine):
    planner, image, depth, options = _shared_proposal_fixture(tmp_path)
    observed = []
    surfaces = []

    def select(pool, target, maximum):
        observed.extend(candidate.capture_support_score for candidate in pool)
        surfaces.extend(candidate.surface_alignment_score for candidate in pool)
        return GraspGenXCandidateSelection((0,), {})

    try:
        planner.start()
        result = planner.plan(image, depth, candidate_selector=select, refine_grasp_centers=refine, **options)
        assert observed and all(0 <= value <= 1 for value in observed)
        assert planner.last_filter_summary["min_selected_capture_support_score"] == observed[0]
        assert all(not hasattr(candidate, "capture_support_score") for candidate in result.candidates)
        assert all(not hasattr(candidate, 'surface_alignment_score') for candidate in result.candidates)
        selected = json.loads(planner.last_filter_summary['selected_capture_geometry_json'])[0]
        assert selected['surface_alignment_score'] == surfaces[0]
        assert 0 <= selected['surface_alignment_observed_sides'] <= 2
        assert any(value < 1 for value in surfaces) if refine else all(value == 1 for value in surfaces)
    finally:
        planner.close()


def _axial_pool(tmp_path, *, enabled=True, background=False, count=1, target_z=.570,
                deep_prefix=0, priority=None):
    planner, _, depth, options = _shared_proposal_fixture(tmp_path)
    camera_from_model = np.repeat(np.eye(4)[None], count, axis=0)
    camera_from_model[:, 2, 3] = .5
    camera_from_model[:deep_prefix, 2, 3] = .515
    camera_from_model[:, 0, 3] = np.linspace(-.003, .003, count) if count > 1 else 0.
    base_from_camera = np.array([[0., 0., 1., .1], [0., 1., 0., -.2], [-1., 0., 0., .3], [0., 0., 0., 1.]])
    target = np.array([[-.01, 0., target_z], [0., 0., target_z], [.01, 0., target_z]])
    geometry = planner.solid_geometry
    from dataclasses import replace
    model_geometry = replace(geometry, tool_from_solid_xyz_m=(0., 0., 0.),
        solid_yaw_rad=0., tool_from_solid_rotation=(1., 0., 0., 0., 1., 0., 0., 0., 1.))
    points = np.concatenate((target, [[.039, 0., .580]])) if background else target
    scene = VisibleGraspScene(points=points, target_mask=np.arange(len(points)) < len(target), geometry=model_geometry)
    counts = {}
    pool = planner._rank_pool(base_from_camera @ camera_from_model, np.full(count, .9),
        target_points_camera=target, target_camera=target[1], base_from_camera=base_from_camera,
        depth=depth, intrinsics=options["intrinsics"], maximum=1, visible_scene=scene,
        for_embodiment=True, filter_summary=counts, refine_grasp_axial_depth=enabled,
        proposal_priority=priority)
    return planner, pool, counts, base_from_camera


def test_axial_refinement_is_a_new_calibrated_local_z_pose_and_preserves_parent(tmp_path):
    planner, pool, counts, extrinsic = _axial_pool(tmp_path)
    children = [c for c in pool.candidates if c.axial_parent_index is not None
        and pool.candidates[c.axial_parent_index].pose_source == "model_scored"]
    assert len(children) == 1
    child, = children
    parent = pool.candidates[child.axial_parent_index]
    assert parent.pose_source == "model_scored" and parent.axial_parent_index is None
    parent_model = planner.solid_geometry.solid_pose(np.asarray(parent.base_from_tcp).reshape(4, 4))
    child_model = planner.solid_geometry.solid_pose(np.asarray(child.base_from_tcp).reshape(4, 4))
    expected = parent_model.copy()
    expected[:3, 3] += parent_model[:3, 2] * .020
    np.testing.assert_allclose(child_model, expected, atol=1e-12)
    assert child.axial_shift_m == .020 and child.capture_support_score > parent.capture_support_score
    assert child.model_score == parent.model_score == .9
    assert child.pose_source == "dmtac_axial_insertion_variant_v1"
    original_model = extrinsic.copy()
    original_model[:3, 3] += extrinsic[:3, 2] * .5
    np.testing.assert_allclose(parent_model, original_model, atol=1e-12)
    original = planner._publish_pool(pool, target=(0., 0., 0.), maximum=1,
        selector=lambda *_: GraspGenXCandidateSelection((pool.candidates.index(child),), {}))[0][0]
    assert original.base_from_tcp == child.base_from_tcp
    assert not hasattr(original, "axial_parent_index")
    from deployments.perception.dmtac_grasp_overlay import (
        DmtacGraspOverlayCandidate,
        project_dmtac_grasp,
    )
    projected = project_dmtac_grasp(DmtacGraspOverlayCandidate.from_pose(ordinal=0,
        score=original.score, gripper_width_m=original.gripper_width_m,
        reference_from_tcp=np.asarray(original.base_from_tcp).reshape(4, 4), selected=True),
        reference_from_camera=extrinsic,
        intrinsics={"width": 640, "height": 480, "fx": 20., "fy": 20., "cx": 320., "cy": 240.},
        solid_geometry=planner.solid_geometry)
    camera_tcp = np.asarray(original.camera_from_tcp).reshape(4, 4)
    point = camera_tcp[:3, 3]
    assert projected.tcp_pixel_uv == (round(20. * point[0] / point[2] + 320.),
        round(20. * point[1] / point[2] + 240.))
    assert counts["axial_variant_attempted"] == 4 * counts["axial_variant_source_count"]


def test_axial_depth_still_rejects_real_background_and_keeps_only_safe_improvement(tmp_path):
    _, pool, counts, _ = _axial_pool(tmp_path, background=True)
    child, = [c for c in pool.candidates if c.axial_parent_index is not None
        and pool.candidates[c.axial_parent_index].pose_source == "model_scored"]
    assert child.axial_shift_m == .005
    assert counts["axial_variant_background_solid_collision_rejected"] >= 3


def test_axial_is_default_off_and_its_generation_is_bounded(tmp_path):
    _, pool, counts, _ = _axial_pool(tmp_path, enabled=False)
    assert all(c.axial_parent_index is None for c in pool.candidates)
    assert "axial_variant_attempted" not in counts
    _, pool, counts, _ = _axial_pool(tmp_path, count=20)
    assert counts["axial_variant_source_count"] == 16
    assert counts["axial_variant_attempted"] == 64
    assert counts["axial_variant_pool_count"] <= 16
    assert len(counts["axial_variant_sources_json"]) < 8192
    for child in pool.candidates:
        if child.axial_parent_index is not None:
            parent = pool.candidates[child.axial_parent_index]
            assert parent.axial_parent_index is None
            assert child.score == pytest.approx(parent.score * math.exp(
                (parent.click_distance_m - child.click_distance_m) / .060))


def test_axial_does_not_spend_proposals_on_already_deep_captures(tmp_path):
    _, pool, counts, _ = _axial_pool(tmp_path, target_z=.560)
    assert counts["axial_variant_attempted"] == counts["axial_variant_source_count"] == 0
    assert all(c.axial_parent_index is None for c in pool.candidates)


def test_deep_sources_do_not_exhaust_the_shallow_refinement_allowance(tmp_path, monkeypatch):
    import deployments.perception.sam2_graspgenx_point_grasp as module
    monkeypatch.setattr(module, "GRASPGENX_DMTAC_AXIAL_SOURCE_LIMIT", 1)
    _, pool, counts, _ = _axial_pool(tmp_path, count=3, deep_prefix=2,
        priority=lambda *_: 0.)
    assert counts["axial_variant_source_count"] == 1
    assert counts["axial_variant_attempted"] == 4
    assert any(c.axial_parent_index is not None for c in pool.candidates)
    assert all(pool.candidates[c.axial_parent_index].capture_support_score < .25
        for c in pool.candidates if c.axial_parent_index is not None)


@pytest.mark.parametrize("enabled", [1, "true", None])
def test_axial_flag_is_strict_before_model_inference(tmp_path, enabled):
    planner, image, depth, options = _shared_proposal_fixture(tmp_path)
    planner.start()
    try:
        with pytest.raises(TypeError, match="refine_grasp_axial_depth"):
            planner.plan(image, depth, refine_grasp_axial_depth=enabled, **options)
        assert not planner.client.calls
    finally:
        planner.close()


@pytest.mark.parametrize("name,value", [
    ("GRASPGENX_DMTAC_AXIAL_SOURCE_LIMIT", 8),
    ("GRASPGENX_DMTAC_AXIAL_SOURCE_ORDER", "test:changed-source-order"),
    ("GRASPGENX_DMTAC_AXIAL_SHIFTS_M", (.005, .010)),
    ("GRASPGENX_DMTAC_AXIAL_SHALLOW_SUPPORT", .3),
])
def test_axial_constants_are_bound_to_model_deployment_identity(tmp_path, monkeypatch, name, value):
    import deployments.perception.sam2_graspgenx_point_grasp as module
    planner, *_ = _shared_proposal_fixture(tmp_path)
    before = planner.semantic_content_hash
    monkeypatch.setattr(module, name, value)
    assert planner.semantic_content_hash != before


def test_centering_source_limit_preserves_a_later_approach_cone(tmp_path):
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[8:13, 8:13] = 1.
    poses = np.repeat(np.eye(4)[None], 41, axis=0)
    poses[:40, 0, 3] = -.009
    poses[:40, 2, 3] = np.linspace(.973, .977, 40)
    poses[-1] = np.diag([1., -1., -1., 1.])
    poses[-1, :3, 3] = [-.009, 0, 1.025]
    planner = Sam2GraspGenXPointGraspPlanner(model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(depth > 0), client=_PlannerClient(poses, np.linspace(.9, .5, 41)))

    def select(pool, target, maximum):
        indices = [i for i, candidate in enumerate(pool)
            if candidate.pose_source == "dmtac_centering_variant_v1"
            and planner.solid_geometry.solid_pose(np.array(candidate.base_from_tcp).reshape(4, 4))[2, 2] < -.9]
        assert indices, "A lower-priority approach must survive source allocation before rechecking"
        return GraspGenXCandidateSelection((indices[0],), {})

    try:
        planner.start()
        planner.plan(Image.new("RGB", (20, 20)), depth, pixel_uv=(10, 10),
            intrinsics={"width":20, "height":20, "fx":100., "fy":100., "cx":10., "cy":10.},
            base_from_camera=np.eye(4), max_candidates=1,
            candidate_selector=select, refine_grasp_centers=True,
            proposal_priority=lambda pose, score, target: int(pose[2, 2] < 0))
        counts = planner.last_filter_summary
        assert counts["centering_variant_source_count"] == 32
        assert counts["centering_variant_source_limit_rejected"] == 9
        assert counts["centering_variant_selected"] == 1
    finally:
        planner.close()


def test_empty_local_chain_reports_aggregate_stage_and_retains_real_request_arrays(tmp_path):
    planner, image, depth, options = _shared_proposal_fixture(tmp_path)
    def select(pool, _target, _maximum):
        return GraspGenXCandidateSelection((), {
            "arm_shortlist_pool_count": len(pool),
            "arm_shortlist_local_chain_assessment_count": 31,
            "arm_shortlist_local_chain_feasible_count": 0,
            "arm_shortlist_visible_scene_check_count": 0,
            "arm_shortlist_budget_exhausted": False,
            "arm_shortlist_status": "no_production_local_grasp_chain_candidate",
            "arm_shortlist_attempts_json": "x" * 10000,
            "arm_shortlist_rejections_json": json.dumps({
                "ik_search_exhausted": 15, "approach_ik_search_exhausted": 15,
                "grasp_path_motion_quality_exceeded": 1}),
        })
    planner.start()
    try:
        with pytest.raises(PointGraspPlanningError) as raised:
            planner.plan(image, depth, candidate_selector=select, **options)
        error = raised.value
        assert error.reason_code == "point_grasp_no_local_grasp_chain_candidate"
        assert error.diagnostic["counts"]["local_chain_assessed"] == 31
        assert error.diagnostic["counts"]["observed_chain_checked"] == 0
        assert error.diagnostic["rejections_by_stage"] == {
            "endpoint_ik": 15, "local_grasp_chain": 15, "motion_quality": 1}
        assert error.diagnostic["search_budget_exhausted"]["local_chain"] is False
        assert len(str(error)) < 512
        assert "clicked region" not in str(error)
        assert error.filter_summary["arm_shortlist_attempts_json"] == "x" * 10000
        np.testing.assert_array_equal(error.replay_arrays["target_mask"], depth > 0)
        np.testing.assert_array_equal(error.replay_arrays["raw_grasps_reference"], planner.client.grasps)
        assert len(error.replay_arrays["pool_tool_poses_reference"]) > 0
    finally:
        planner.close()


@pytest.mark.parametrize("extra,stage", [
    ({"arm_shortlist_status": "grasp_fast_search_budget_exhausted",
      "arm_shortlist_full_path_query_count": 5}, "full_path_search"),
    ({"arm_shortlist_status": "static_scene_blocked"}, "static_scene"),
    ({"arm_shortlist_geometric_retained_count": 0,
      "arm_shortlist_geometric_rejected_count": 4}, "geometric_reach"),
    ({"arm_shortlist_status": "no_production_local_grasp_chain_candidate",
      "arm_shortlist_wrist_orientation_rejected_count": 4,
      "arm_shortlist_geometric_checked_count": 0,
      "arm_shortlist_geometric_retained_count": 0}, "arm_candidate_selection"),
    ({"arm_shortlist_local_chain_assessment_count": 1,
      "arm_shortlist_rejections_json": '{"grasp_path_motion_quality_exceeded":1}'}, "motion_quality"),
    ({"arm_shortlist_local_chain_assessment_count": 1,
      "arm_shortlist_rejections_json": '{"endpoint_geometric_reach_exceeded":4,"grasp_path_motion_quality_exceeded":1}'}, "motion_quality"),
    ({}, "physical_filtering"),
])
def test_point_grasp_failure_does_not_conflate_stages(extra, stage):
    error = PointGraspPlanningError(extra)
    assert error.diagnostic["stage"] == stage
    assert error.diagnostic["same_frozen_pixel_retry_recommended"] is False
    assert error.diagnostic["counts"]["endpoint_probed"] is None


def test_fast_failure_reports_complete_rejection_counts_and_bounded_native_examples():
    import json
    from agent_gateway.runtime.passive_read_executor import PassiveReadError

    attempts = [{"index": i, "reason_code": "ik_search_exhausted", "elapsed_sec": 1.2}
                for i in range(63)]
    attempts.append({"index": 63, "reason_code": "path_planning_time_budget_exhausted", "elapsed_sec": .3})
    native = [{"index": row["index"], "allocated_budget_sec": 8.,
               "reason_code": row["reason_code"], "planning_phase": "terminal_branch_selection",
               "cycles": 2360, "endpoint_seed_count": 47,
               "best_position_residual_m": .039864, "best_orientation_residual_deg": 19.56}
              for row in attempts]
    error = PointGraspPlanningError({"arm_shortlist_full_path_query_count": 64,
        "arm_shortlist_full_path_attempts_json": json.dumps(attempts),
        "arm_shortlist_native_path_attempts_json": json.dumps(native)})
    diagnostic = error.diagnostic
    assert diagnostic["full_path_rejections_by_reason"] == {
        "ik_search_exhausted": 63, "path_planning_time_budget_exhausted": 1}
    assert diagnostic["rejections_by_stage"] == {"endpoint_ik": 63, "search_budget": 1}
    assert len(diagnostic["native_path_attempts"]) == 4
    assert diagnostic["native_path_attempts_omitted"] == 60
    assert diagnostic["counts"]["endpoint_probed"] is None
    # Include the actual provider's budget fields within the neutral 4 KiB limit.
    PassiveReadError(str(error), reason_code=error.reason_code, diagnostic={
        **diagnostic, "elapsed_sec": 20.1, "budget_sec": 20.,
        "budget_scope": "complete_request_including_preparation",
        "debug_evidence": {"artifact_id": "artifact:"+"a"*64, "sha256": "b"*64,
                           "scope": "operator_debug_only_not_success_output_or_motion_authority"}})
    from deployments.perception.point_grasp_failure import bounded_point_grasp_failure_diagnostic
    oversized = {**diagnostic, "debug_evidence_failure": {
        "exception_type": "OSError", "message": "😀"*256},
        "elapsed_sec": 20.1, "budget_sec": 20.,
        "budget_scope": "complete_request_including_preparation"}
    assert len(json.dumps(oversized, separators=(",", ":")).encode()) > 4096
    bounded = bounded_point_grasp_failure_diagnostic(oversized)
    projected = PassiveReadError(str(error), reason_code=error.reason_code, diagnostic=bounded)
    assert projected.reason_code == error.reason_code
    assert bounded["counts"] == diagnostic["counts"]
    assert bounded["full_path_rejections_by_reason"] == diagnostic["full_path_rejections_by_reason"]
    assert len(bounded["native_path_attempts"]) + bounded["native_path_attempts_omitted"] == 64
    assert bounded["debug_evidence_failure"]["message_truncated"]
    # Required oversized facts still fail rather than being silently discarded.
    with pytest.raises(ValueError, match="Required point-grasp failure"):
        bounded_point_grasp_failure_diagnostic({"reason_code": "required"*1000})


def test_dmtac_model_conditioning_and_solid_share_axial_origin():
    # The reviewed model origin is not its palm. An origin-free length would
    # tell OBB generation to place the fingertips 29 mm beyond the real solid.
    geometry = reviewed_graspgenx_dmtac_solid_geometry()
    params = DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1
    for state in ("open", "mid"):
        half = np.array(params[f"extents_{state}"]) / 2
        offset = np.array(params[f"offset_{state}"])
        low, high = offset - half, offset + half
        assert low[2] == pytest.approx(geometry.palm_z_m)
        assert high[2] == pytest.approx(params["fingertip_depth"])
        # Collision alone includes the reviewed 1 mm CAD tip extension. It is
        # not extra physical reach for the neural or OBB grasp proposals.
        assert high[2] < geometry.palm_z_m + geometry.finger_length_m
        assert geometry.palm_z_m + geometry.finger_length_m - high[2] == pytest.approx(.001)


@pytest.mark.parametrize("padding", [-.001, float("nan"), float("inf"), True])
def test_planner_rejects_invalid_background_padding(tmp_path, padding):
    with pytest.raises(ValueError, match="background_proxy_padding_m"):
        Sam2GraspGenXPointGraspPlanner(
            model_spec=_model_spec(tmp_path), point_segmenter=_Segmenter(np.ones((4, 4), bool)),
            client=_PlannerClient(np.eye(4)[None], np.array([.9])),
            background_proxy_padding_m=padding,
        )


@pytest.mark.parametrize("padding", [0., .003])
def test_terminal_prefilter_preserves_real_background_padding_geometry(tmp_path, padding):
    # This external point misses the nominal finger's height by 1 mm, but
    # overlaps its required padded volume. Do not relabel or remove the point.
    pose = np.eye(4)
    pose[2, 3] = .975
    geometry = reviewed_graspgenx_dmtac_solid_geometry()
    from dataclasses import replace
    geometry = replace(geometry, tool_from_solid_xyz_m=(0, 0, 0),
        tool_from_solid_rotation=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    point = np.array([[geometry.maximum_opening_m / 2 + .002, .016, 1.015]])
    scene = VisibleGraspScene(points=point, target_mask=np.array([False]), geometry=geometry)
    planner = Sam2GraspGenXPointGraspPlanner(
        model_spec=_model_spec(tmp_path), point_segmenter=_Segmenter(np.ones((4, 4), bool)),
        client=_PlannerClient(pose[None], np.array([.9])), background_proxy_padding_m=padding,
    )
    summary = {}
    planner._rank_pool(pose[None], np.array([.9]),
        target_points_camera=np.array([[0., 0., 1.]]), target_camera=np.array([0., 0., 1.]),
        base_from_camera=np.eye(4), depth=np.ones((4, 4)),
        intrinsics={"fx": 100., "fy": 100., "cx": 2., "cy": 2., "width": 4, "height": 4},
        maximum=1, filter_summary=summary, visible_scene=scene)
    assert summary['terminal_background_collision_rejected'] == int(padding > 0)
    np.testing.assert_array_equal(scene.points, point)
    np.testing.assert_array_equal(scene.target_mask, [False])


def _centering_fixture(tmp_path, *, shift_m=.010, yaw_degrees=0.0):
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[8:13, 8:13] = 1.0
    angle = math.radians(yaw_degrees)
    grasp = np.eye(4)
    grasp[:3, :3] = (
        (math.cos(angle), -math.sin(angle), 0),
        (math.sin(angle), math.cos(angle), 0),
        (0, 0, 1),
    )
    grasp[:3, 3] = np.asarray((0., 0., .975)) - grasp[:3, 0] * shift_m
    planner = Sam2GraspGenXPointGraspPlanner(
        model_spec=_model_spec(tmp_path), point_segmenter=_Segmenter(depth > 0),
        client=_PlannerClient(grasp[None], np.array([.9])),
    )
    options = {
        "pixel_uv": (10, 10),
        "intrinsics": {"width": 20, "height": 20, "fx": 100., "fy": 100., "cx": 10., "cy": 10.},
        "base_from_camera": np.eye(4), "max_candidates": 1,
    }
    return planner, Image.new("RGB", (20, 20)), depth, options


@pytest.mark.parametrize("shift_m", [-.010, .010])
@pytest.mark.parametrize("yaw_degrees", [0., 90.])
def test_centering_opt_in_generates_separate_model_x_pose(tmp_path, shift_m, yaw_degrees):
    planner, image, depth, options = _centering_fixture(
        tmp_path, shift_m=shift_m, yaw_degrees=yaw_degrees)
    camera_parent = planner.client.grasps[0].copy()
    angle = math.radians(30)
    base_from_camera = np.eye(4)
    base_from_camera[:3, :3] = (
        (math.cos(angle), -math.sin(angle), 0),
        (math.sin(angle), math.cos(angle), 0), (0, 0, 1),
    )
    base_from_camera[:3, 3] = (.1, -.2, .05)
    options["base_from_camera"] = base_from_camera
    planner.client.grasps = base_from_camera @ planner.client.grasps
    raw = planner.client.grasps.copy()
    planner.start()
    try:
        with pytest.raises(PointGraspPlanningError, match="physical_filtering"):
            planner.plan(image, depth, **options)
        result = planner.plan(image, depth, refine_grasp_centers=True, **options)
        candidate, = result.candidates
        expected = camera_parent.copy()
        expected[:3, 3] += expected[:3, 0] * shift_m
        np.testing.assert_allclose(
            planner.solid_geometry.solid_pose(np.asarray(candidate.camera_from_tcp).reshape(4, 4)),
            expected, atol=1e-9)
        np.testing.assert_allclose(
            np.asarray(candidate.base_from_tcp).reshape(4, 4),
            base_from_camera @ np.asarray(candidate.camera_from_tcp).reshape(4, 4), atol=1e-9)
        np.testing.assert_array_equal(planner.client.grasps, raw)
        np.testing.assert_array_equal(planner.client.scores, [.9])
        assert candidate.pose_source == "dmtac_centering_variant_v1"
        assert candidate.orientation_variant_axis is None
        assert candidate.orientation_variant_angle_deg == 0
        assert candidate.model_score == .9
        assert candidate.score == pytest.approx(.9 * math.exp(-.025/.060) * math.exp(-abs(shift_m)/.060))
        counts = planner.last_filter_summary
        for stage in ("source_eligible", "source_count", "attempted", "width_valid", "target_capture_valid",
                      "click_distance_valid", "visible_collision_valid", "visible_approach_valid", "pool_count", "selected"):
            assert counts[f"centering_variant_{stage}"] == 1
        assert counts["target_capture_valid"] == 0
        assert counts["orientation_variant_attempted"] == 0
        parents = json.loads(counts["centering_variant_sources_json"])
        assert parents == [{"raw_index": 0, "shift_m": pytest.approx(shift_m), "pool_index": 0}]
        with pytest.raises(PointGraspPlanningError, match="physical_filtering"):
            planner.plan(image, depth, refine_grasp_centers=False, **options)
        assert "centering_variant_attempted" not in planner.last_filter_summary
    finally:
        planner.close()


@pytest.mark.parametrize("shift_m,accepted", [(.015, True), (-.015, True), (.015001, False), (-.015001, False), (.020, False)])
def test_centering_shift_limit_does_not_clamp_excessive_offset(tmp_path, shift_m, accepted):
    planner, image, depth, options = _centering_fixture(tmp_path, shift_m=shift_m)
    raw = planner.client.grasps.copy()
    planner.start()
    try:
        if accepted:
            assert planner.plan(image, depth, refine_grasp_centers=True, **options).candidates
        else:
            with pytest.raises(PointGraspPlanningError, match="physical_filtering"):
                planner.plan(image, depth, refine_grasp_centers=True, **options)
        counts = planner.last_filter_summary
        assert counts["centering_variant_attempted"] == int(accepted)
        assert counts["centering_variant_shift_limit_rejected"] == int(not accepted)
        np.testing.assert_array_equal(planner.client.grasps, raw)
    finally:
        planner.close()


@pytest.mark.parametrize("failed_gate", ["width", "y", "z"])
def test_centering_requires_raw_width_and_yz_capture_overlap(tmp_path, failed_gate):
    planner, image, depth, options = _centering_fixture(tmp_path)
    if failed_gate == "width":
        options["intrinsics"]["fx"] = 50.
    elif failed_gate == "y":
        planner.client.grasps[0, 1, 3] = -.060
    else:
        planner.client.grasps[0, 2, 3] = .900
    planner.start()
    try:
        with pytest.raises(PointGraspPlanningError, match="physical_filtering"):
            planner.plan(image, depth, refine_grasp_centers=True, **options)
        counts = planner.last_filter_summary
        assert counts["centering_variant_source_count"] == counts["centering_variant_attempted"] == 0
        assert counts["centering_variant_yz_capture_rejected"] == int(failed_gate != "width")
    finally:
        planner.close()


def test_centering_still_rejects_real_visible_background_collision(tmp_path):
    planner, image, depth, options = _centering_fixture(tmp_path)

    def scene_factory(*, depth_m, target_mask, intrinsics, geometry):
        points = Sam2VgnPointGraspPlanner._masked_points_camera(depth_m, target_mask, intrinsics)
        return VisibleGraspScene(
            points=np.vstack((points, (.040, 0., 1.))),
            target_mask=np.r_[np.ones(len(points), dtype=bool), False], geometry=geometry)

    planner.start()
    try:
        with pytest.raises(PointGraspPlanningError, match="physical_filtering"):
            planner.plan(image, depth, refine_grasp_centers=True, visible_scene_factory=scene_factory, **options)
        counts = planner.last_filter_summary
        assert counts["centering_variant_target_capture_valid"] == 1
        assert counts["centering_variant_background_solid_collision_rejected"] == 1
        assert counts["centering_variant_visible_collision_valid"] == counts["centering_variant_pool_count"] == 0
        assert counts["centering_variant_selected"] == 0
        assert json.loads(counts["centering_variant_sources_json"])[0]["pool_index"] is None
    finally:
        planner.close()


@pytest.mark.parametrize("use_priority", [False, True])
def test_centering_source_cap_keeps_best_32_and_bounds_lineage(tmp_path, use_priority):
    planner, image, depth, options = _centering_fixture(tmp_path)
    planner.client.grasps = np.repeat(planner.client.grasps, 40, axis=0)
    planner.client.scores = np.linspace(.1, .9, 40)
    priority = (lambda _pose, score, _target: score) if use_priority else None
    planner.start()
    try:
        planner.plan(image, depth, refine_grasp_centers=True, proposal_priority=priority, **options)
        counts = planner.last_filter_summary
        assert counts["centering_variant_source_eligible"] == 40
        assert counts["centering_variant_source_count"] == counts["centering_variant_attempted"] == 32
        assert counts["centering_variant_source_limit_rejected"] == 8
        assert counts["centering_variant_pool_count"] == 1
        parents = json.loads(counts["centering_variant_sources_json"])
        assert [row["raw_index"] for row in parents] == (list(range(32)) if use_priority else list(range(39, 7, -1)))
        assert len(counts["centering_variant_sources_json"]) < 4096
    finally:
        planner.close()


def test_centering_keeps_admitted_scored_candidates_and_their_scores(tmp_path):
    planner, image, depth, options = _centering_fixture(tmp_path)
    admitted = np.eye(4)
    admitted[2, 3] = .965
    planner.client.grasps = np.concatenate((admitted[None], planner.client.grasps))
    planner.client.scores = np.array([.8, .9])
    options["max_candidates"] = 2
    planner.start()
    try:
        original = planner.plan(image, depth, **options)
        refined = planner.plan(image, depth, refine_grasp_centers=True, **options)
        preserved = tuple(c for c in refined.candidates if c.pose_source == "model_scored")
        assert preserved == tuple(c for c in original.candidates if c.pose_source == "model_scored")
        assert any(c.pose_source == "dmtac_centering_variant_v1" for c in refined.candidates)
        assert planner.plan(image, depth, refine_grasp_centers=False, **options) == original
    finally:
        planner.close()


def test_centering_does_not_override_embodiment_selector_rejection(tmp_path):
    planner, image, depth, options = _centering_fixture(tmp_path)
    checked = []

    def reject(pool, _target, _maximum):
        checked.extend(pool)
        return GraspGenXCandidateSelection((), {})

    planner.start()
    try:
        with pytest.raises(PointGraspPlanningError):
            planner.plan(image, depth, refine_grasp_centers=True, candidate_selector=reject, **options)
        assert len(checked) == 1 and checked[0].pose_source == "dmtac_centering_variant_v1"
        assert planner.last_filter_summary["centering_variant_selected"] == 0
    finally:
        planner.close()


def test_centering_rechecks_real_visible_approach_before_owner_admission(tmp_path):
    planner, image, depth, options = _centering_fixture(tmp_path)
    checked = []

    def scene_factory(*, depth_m, target_mask, intrinsics, geometry):
        points = Sam2VgnPointGraspPlanner._masked_points_camera(depth_m, target_mask, intrinsics)
        return VisibleGraspScene(
            points=np.vstack((points, (.040, 0., .910))),
            target_mask=np.r_[np.ones(len(points), dtype=bool), False], geometry=geometry)

    def reject(pool, _target, _maximum):
        # The existing owner still admits/rejects full paths, including ingress.
        checked.extend(pool)
        return GraspGenXCandidateSelection((), {})

    planner.start()
    try:
        with pytest.raises(PointGraspPlanningError):
            planner.plan(image, depth, refine_grasp_centers=True, visible_scene_factory=scene_factory,
                         candidate_selector=reject, **options)
        assert len(checked) == 1 and checked[0].pose_source == "dmtac_centering_variant_v1"
        counts = planner.last_filter_summary
        assert counts["centering_variant_visible_collision_valid"] == 1
        assert counts["centering_variant_visible_approach_collision_candidates"] == 1
        assert counts["centering_variant_visible_approach_background_collision_candidates"] == 1
        assert counts["centering_variant_visible_approach_valid"] == counts["centering_variant_selected"] == 0
    finally:
        planner.close()


def test_centering_fixed_limits_are_bound_to_planner_identity(tmp_path, monkeypatch):
    from deployments.perception import sam2_graspgenx_point_grasp as module

    planner, *_ = _centering_fixture(tmp_path)
    baseline = planner.semantic_content_hash
    for name, value in (("GRASPGENX_DMTAC_CENTERING_MAX_SHIFT_M", .016),
                        ("GRASPGENX_DMTAC_CENTERING_SOURCE_LIMIT", 33),
                        ("GRASPGENX_DMTAC_CENTERING_SCORE_DECAY_M", .061)):
        with monkeypatch.context() as patch:
            patch.setattr(module, name, value)
            assert planner.semantic_content_hash != baseline


@pytest.mark.parametrize("enabled", [1, None, "true"])
def test_centering_rejects_non_boolean_opt_in_before_inference(tmp_path, enabled):
    planner, image, depth, options = _centering_fixture(tmp_path)
    planner.start()
    try:
        with pytest.raises(TypeError, match="refine_grasp_centers must be bool"):
            planner.plan(image, depth, refine_grasp_centers=enabled, **options)
        assert not planner.client.calls
    finally:
        planner.close()


@pytest.mark.parametrize("angle", (0., 90.))
@pytest.mark.parametrize("blocked", (False, True))
def test_pad_height_refinement_is_local_and_rechecks_background(tmp_path, angle, blocked):
    from dataclasses import replace

    from scipy.spatial.transform import Rotation
    planner = Sam2GraspGenXPointGraspPlanner(model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(np.ones((20,20),dtype=bool)),
        client=_PlannerClient(np.eye(4)[None],np.array([.8])))
    pose = np.eye(4)
    pose[:3,:3] = Rotation.from_euler("x",angle,degrees=True).as_matrix()
    pose[2,3] = 1.
    local = np.array([[x,y,z] for x in (-.025,0.,.025)
                      for y in np.linspace(.010,.045,40) for z in (.020,.030,.040)])
    points = local@pose[:3,:3].T+pose[:3,3]
    geometry = replace(planner.solid_geometry, tool_from_solid_xyz_m=(0,0,0),
        tool_from_solid_rotation=(1,0,0,0,1,0,0,0,1))
    background = np.array([[.042,.027,.03]])@pose[:3,:3].T+pose[:3,3]
    scene = VisibleGraspScene(points=np.concatenate((points,background)) if blocked else points,
        target_mask=np.r_[np.ones(len(points),dtype=bool),False] if blocked else np.ones(len(points),dtype=bool),
        geometry=geometry)
    counts = {}
    pool = planner._rank_pool(pose[None],np.array([.8]),target_points_camera=points,
        target_camera=points.mean(0),base_from_camera=np.eye(4),
        depth=np.ones((20,20)),intrinsics={"fx":100.,"fy":100.,"cx":10.,"cy":10.},
        maximum=1,filter_summary=counts,visible_scene=scene,refine_grasp_centers=True)
    parents = [c for c in pool.candidates if c.pose_source == "model_scored"]
    children = [c for c in pool.candidates if c.pose_source == "dmtac_pad_height_centering_variant_v1"]
    assert len(parents) == 1 and parents[0].capture_height_support_score < .5
    if blocked:
        assert not children and counts["height_variant_background_solid_collision_rejected"] > 0
    else:
        assert children
        for child in children:
            actual = planner.solid_geometry.solid_pose(np.array(child.base_from_tcp).reshape(4,4))
            np.testing.assert_allclose(actual[:3,3]-pose[:3,3],pose[:3,1]*child.height_shift_m,atol=1e-12)
            assert abs(child.height_shift_m) <= .040
            assert child.capture_height_support_score >= .5 and child.raw_index == 0


@pytest.mark.parametrize("width,height,angle", ((.02, .03, 0.), (.05, .05, 0.), (.06, .09, 90.)))
@pytest.mark.parametrize("blocked", (False, True))
@pytest.mark.parametrize("offset", (0., .010, .020))
def test_contact_band_refinement_survives_dense_top_and_density_miss(
    tmp_path, width, height, angle, blocked, offset,
):
    from dataclasses import replace
    from scipy.spatial.transform import Rotation

    planner = Sam2GraspGenXPointGraspPlanner(model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(np.ones((20, 20), dtype=bool)),
        client=_PlannerClient(np.eye(4)[None], np.array([.8])))
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("x", angle, degrees=True).as_matrix()
    pose[2, 3] = 1.
    sides = np.array([(x, y, z) for x in (-width/2, width/2)
                      for y in np.linspace(-.001, height-.001, 41)
                      for z in np.linspace(.02, .04, 13)])
    top = np.array([(x, -.001, z) for x in np.linspace(-width/2, width/2, 101)
                    for z in np.linspace(.02, .04, 61)])
    local = np.unique(np.vstack((sides, top)), axis=0)
    local[:, 1] += offset
    points = local @ pose[:3, :3].T + pose[:3, 3]
    geometry = replace(planner.solid_geometry, tool_from_solid_xyz_m=(0., 0., 0.),
        solid_yaw_rad=0., tool_from_solid_rotation=(1., 0., 0., 0., 1., 0., 0., 0., 1.))
    # A strip alongside the hand intersects both proposed translations while
    # staying outside the original pad band; one point need not block both.
    background = np.array([[.042, y, .03] for y in np.linspace(.019, .09, 72)]) @ pose[:3, :3].T + pose[:3, 3]
    scene = VisibleGraspScene(points=np.vstack((points, background)) if blocked else points,
        target_mask=np.r_[np.ones(len(points), dtype=bool), np.zeros(len(background), dtype=bool)] if blocked else np.ones(len(points), dtype=bool),
        geometry=geometry)
    summary = {}
    pool = planner._rank_pool(pose[None], np.array([.8]), target_points_camera=points,
        target_camera=points.mean(0), base_from_camera=np.eye(4), depth=np.ones((20, 20)),
        intrinsics={"fx": 100., "fy": 100., "cx": 10., "cy": 10.}, maximum=1,
        filter_summary=summary, visible_scene=scene, refine_grasp_centers=True)
    parents = [c for c in pool.candidates if c.pose_source == "model_scored"]
    children = [c for c in pool.candidates if c.pose_source == "dmtac_pad_height_centering_variant_v1"]
    assert len(parents) == (0 if offset == .020 else 1)
    if parents:
        parent = parents[0]
        assert (parent.capture_height_support_score > .9 if offset == 0.
                else parent.capture_height_support_score < .5)
        assert parent.contact_height_evidence_weight >= 3.
        assert parent.contact_height_support_score < .85
    assert summary["height_variant_contact_source_count"] == 1
    assert summary["height_variant_attempted"] <= 2
    if blocked:
        assert not children and summary["height_variant_background_solid_collision_rejected"] > 0
    else:
        assert children
        for child in children:
            assert child.height_shift_basis == "observed_contact_band"
            assert child.contact_height_support_score >= .5
            if parents:
                assert child.contact_height_support_score > parent.contact_height_support_score
            actual = planner.solid_geometry.solid_pose(np.array(child.base_from_tcp).reshape(4, 4))
            np.testing.assert_allclose(actual[:3, 3]-pose[:3, 3], pose[:3, 1]*child.height_shift_m, atol=1e-12)
            assert 0. < abs(child.height_shift_m) <= .040
            from deployments.perception.grasp_first_path import FirstGraspPathPolicy, grasp_capture_height_acceptable
            assert grasp_capture_height_acceptable(child, policy=FirstGraspPathPolicy())


@pytest.mark.parametrize(("shift", "expected"), (
    (.060, (.030,)), (-.060, (-.030,)), (.040, (.020, .040)),
    (.081, ()), (0., ()),
))
def test_pad_height_budget_uses_permitted_actual_shifts(shift, expected):
    assert _pad_height_shifts(shift) == expected


@pytest.mark.parametrize("source_limit", (1, 32))
def test_contact_height_children_receive_bounded_rechecked_orientation_neighbours(tmp_path, source_limit):
    planner = Sam2GraspGenXPointGraspPlanner(model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(np.ones((20,20),dtype=bool)),
        client=_PlannerClient(np.eye(4)[None],np.array([.8])),
        orientation_variants_local_axis_degrees=(("x", -10.), ("x", 10.)),
        orientation_variant_source_limit=source_limit)
    pose = np.eye(4); pose[2,3] = 1.
    local = np.array([(x,y,z) for x in (-.025,.025)
        for y in np.linspace(.009,.055,41) for z in np.linspace(.02,.04,13)]
        + [(x,.009,z) for x in np.linspace(-.025,.025,101) for z in np.linspace(.02,.04,61)])
    points = local + pose[:3,3]
    counts = {}
    pool = planner._rank_pool(pose[None],np.array([.8]),target_points_camera=points,
        target_camera=points.mean(0),base_from_camera=np.eye(4),depth=np.ones((20,20)),
        intrinsics={"fx":100.,"fy":100.,"cx":10.,"cy":10.},maximum=1,filter_summary=counts,
        refine_grasp_centers=True,for_embodiment=True,proposal_priority=lambda *_:(0,0.))
    links = json.loads(counts["contact_orientation_sources_json"])
    assert 0 < len(links) <= min(source_limit,8)*2
    assert counts["contact_orientation_pool_count"] == len(links)
    assert len({link["parent_index"] for link in links}) <= min(source_limit,8)
    for link in links:
        parent,child = (pool.candidates[link[k]] for k in ("parent_index","pool_index"))
        assert parent.pose_source == "dmtac_pad_height_centering_variant_v1"
        assert child.pose_source == "dmtac_local_orientation_variant_v1"
        assert child.visible_collision_free and child.contact_height_support_score >= .5
        pp,cp = (planner.solid_geometry.solid_pose(np.array(c.base_from_tcp).reshape(4,4)) for c in (parent,child))
        local_points = (points-pp[:3,3])@pp[:3,:3]
        centre = np.percentile(local_points,(5,95),axis=0).mean(0)
        world_centre = pp[:3,:3]@centre+pp[:3,3]
        np.testing.assert_allclose((world_centre-cp[:3,3])@cp[:3,:3],centre,atol=1e-10)


def test_overlap_miss_is_not_discarded_by_distant_density_before_contact_enrichment(tmp_path):
    planner = Sam2GraspGenXPointGraspPlanner(model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(np.ones((20,20),dtype=bool)),
        client=_PlannerClient(np.eye(4)[None],np.array([.8])))
    pose = np.eye(4); pose[2,3] = 1.
    local = np.array([(x,y,z) for x in (-.025,.025)
        for y in np.linspace(.020,.055,41) for z in np.linspace(.02,.04,13)]
        + [(x,.120,z) for x in np.linspace(-.025,.025,61) for z in np.linspace(.02,.04,61)])
    density = parallel_jaw_pad_height_support(local,geometry=planner.solid_geometry,opening_m=.0705)
    assert density.center_shift_m == pytest.approx(.120)
    assert not _pad_height_shifts(density.center_shift_m)
    points = local + pose[:3,3]
    counts = {}
    pool = planner._rank_pool(pose[None],np.array([.8]),target_points_camera=points,
        target_camera=np.array([0.,.035,1.03]),base_from_camera=np.eye(4),depth=np.ones((20,20)),
        intrinsics={"fx":100.,"fy":100.,"cx":10.,"cy":10.},maximum=1,filter_summary=counts,
        refine_grasp_centers=True)
    assert counts["height_variant_source_count"] == 1 and counts["height_variant_attempted"] <= 2
    assert pool.candidates
    assert all(c.pose_source == "dmtac_pad_height_centering_variant_v1" and
        c.height_shift_basis == "observed_contact_band" and c.contact_height_support_score >= .5
        for c in pool.candidates)


@pytest.mark.parametrize("rejection", ("capture_bounds", "joint_empty", "click_distance"))
def test_rejected_pose_does_not_compute_unused_capture_preferences(tmp_path, monkeypatch, rejection):
    from deployments.perception import sam2_graspgenx_point_grasp as module

    planner, _, depth, options = _shared_proposal_fixture(
        tmp_path, max_click_distance_m=.01 if rejection == "click_distance" else .1)
    tip = module.DMTAC_M_GRASPGENX_PALM_Z_M + module.DMTAC_M_GRASPGENX_FINGER_LENGTH_M
    if rejection == "capture_bounds":
        local = np.array([[-.01, 0., tip+.1], [0., 0., tip+.1], [.01, 0., tip+.1]])
    elif rejection == "joint_empty":
        # Every marginal interval overlaps, but no point is inside all axes.
        local = np.array([[-.01, .03, .02], [.01, -.03, .02], [0., 0., tip+.02]])
    else:
        local = np.array([[x, y, .025] for x in (-.01, .01) for y in (-.002, 0., .002)])
    pose = np.eye(4)
    pose[2, 3] = 1.
    points = local + pose[:3, 3]

    def unused(*_args, **_kwargs):
        raise AssertionError("Rejected pose computed an unused capture preference")

    for name in ("parallel_jaw_pad_height_support", "parallel_jaw_opposed_capture_support",
                 "parallel_jaw_surface_alignment"):
        monkeypatch.setattr(module, name, unused)
    summary = {}
    pool = planner._rank_pool(pose[None], np.array([.8]), target_points_camera=points,
        target_camera=points.mean(0), base_from_camera=np.eye(4), depth=depth,
        intrinsics=options["intrinsics"], maximum=1, filter_summary=summary,
        refine_grasp_centers=True)
    assert not pool.candidates
    assert summary["width_valid"] == 1
    assert summary["target_capture_joint_empty_rejected"] == int(rejection == "joint_empty")
    assert summary["target_capture_valid"] == int(rejection == "click_distance")
    assert summary["click_distance_valid"] == 0


@pytest.mark.parametrize("callback", ("parallel_jaw_pad_height_support",
    "parallel_jaw_opposed_capture_support", "parallel_jaw_surface_alignment"))
def test_surviving_pose_does_not_hide_capture_preference_errors(tmp_path, monkeypatch, callback):
    from deployments.perception import sam2_graspgenx_point_grasp as module

    planner, image, depth, options = _shared_proposal_fixture(tmp_path)

    def fail(*_args, **_kwargs):
        raise RuntimeError("capture preference failed")

    monkeypatch.setattr(module, callback, fail)
    planner.start()
    try:
        with pytest.raises(RuntimeError, match="capture preference failed"):
            planner.plan(image, depth, refine_grasp_centers=True, **options)
    finally:
        planner.close()


def test_terminal_collision_does_not_compute_an_unused_surface_preference(tmp_path, monkeypatch):
    from dataclasses import replace
    from deployments.perception import sam2_graspgenx_point_grasp as module

    planner, _, depth, options = _shared_proposal_fixture(tmp_path)
    pose = np.eye(4)
    pose[2, 3] = .5
    points = np.array([[-.01, 0., .525], [0., 0., .525], [.01, 0., .525]])
    background = np.array([[.042, 0., z] for z in np.linspace(.43, .71, 281)])
    geometry = replace(planner.solid_geometry, tool_from_solid_xyz_m=(0., 0., 0.),
        solid_yaw_rad=0., tool_from_solid_rotation=(1., 0., 0., 0., 1., 0., 0., 0., 1.))
    scene = VisibleGraspScene(points=np.vstack((points, background)),
        target_mask=np.r_[np.ones(len(points), dtype=bool), np.zeros(len(background), dtype=bool)],
        geometry=geometry)

    def unused(*_args, **_kwargs):
        raise AssertionError("Terminal-collision rejection computed an unused surface preference")

    monkeypatch.setattr(module, "parallel_jaw_surface_alignment", unused)
    summary = {}
    pool = planner._rank_pool(pose[None], np.array([.8]), target_points_camera=points,
        target_camera=points.mean(0), base_from_camera=np.eye(4), depth=depth,
        intrinsics=options["intrinsics"], maximum=1, visible_scene=scene,
        filter_summary=summary, refine_grasp_centers=True)
    assert not pool.candidates
    assert summary["background_solid_collision_rejected"] > 0


@pytest.mark.parametrize("blocked", (False, True))
def test_rejected_height_band_still_produces_rechecked_recovery_candidates(tmp_path, blocked):
    from dataclasses import replace

    planner, _, depth, options = _shared_proposal_fixture(tmp_path)
    pose = np.eye(4)
    pose[2, 3] = 1.
    local = np.array([[x, y, .025] for x in (-.01, .01)
        for y in np.linspace(.025, .035, 20)])
    points = local + pose[:3, 3]
    geometry = replace(planner.solid_geometry, tool_from_solid_xyz_m=(0, 0, 0),
        solid_yaw_rad=0., tool_from_solid_rotation=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    background = np.array([[.042, .025, .025]]) + pose[:3, 3]
    scene = VisibleGraspScene(points=np.vstack((points, background)) if blocked else points,
        target_mask=np.r_[np.ones(len(points), dtype=bool), False] if blocked else np.ones(len(points), dtype=bool),
        geometry=geometry)
    summary = {}
    pool = planner._rank_pool(pose[None], np.array([.8]), target_points_camera=points,
        target_camera=points.mean(0), base_from_camera=np.eye(4), depth=depth,
        intrinsics=options["intrinsics"], maximum=1, filter_summary=summary,
        refine_grasp_centers=True, visible_scene=scene)
    assert not [candidate for candidate in pool.candidates if candidate.pose_source == "model_scored"]
    assert summary["height_variant_source_count"] == 1
    children = [candidate for candidate in pool.candidates
        if candidate.pose_source == "dmtac_pad_height_centering_variant_v1"]
    if blocked:
        assert not children and summary["height_variant_background_solid_collision_rejected"] > 0
        return
    assert children and all(candidate.visible_collision_free for candidate in children)
    for candidate in children:
        assert candidate.raw_index == 0
        assert 0 < abs(candidate.height_shift_m) <= .040
        assert candidate.capture_height_support_score > .5
