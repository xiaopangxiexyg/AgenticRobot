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
    GraspGenXPointGraspModelSpec,
    Sam2GraspGenXPointGraspPlanner,
)
from deployments.perception.sam2_vgn_point_grasp import Sam2VgnPointGraspPlanner
from deployments.perception.dmtac_grasp_overlay import (
    reviewed_graspgenx_dmtac_solid_geometry,
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
    assert -3.0 in GRASPGENX_POINT_GRASP_MOE_Z_OFFSETS_CM
    assert set((-8.0, -6.0, -4.0, -2.0, -1.0, 0.0)) <= set(GRASPGENX_POINT_GRASP_MOE_Z_OFFSETS_CM)
    assert backend.semantic_config["graspmoe_sampling"]["z_offsets_cm"] == GRASPGENX_POINT_GRASP_MOE_Z_OFFSETS_CM
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
    assert selected == [0, 5, 1, 2]


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
    expected_indices = [0, 3] if maximum == 2 else [0, 2, 3, 4]
    assert len(result.candidates) == len(expected_indices)
    for candidate, index in zip(result.candidates, expected_indices, strict=True):
        expected = poses[index].copy()
        actual = planner.solid_geometry.solid_pose(
            np.asarray(candidate.camera_from_tcp).reshape(4, 4),
        )
        np.testing.assert_allclose(actual, expected, atol=1e-9)
        assert candidate.model_score == pytest.approx((.9, .89, .8, .7, .6)[index])
        assert candidate.visible_collision_free
    assert planner.last_filter_summary["candidate_family_count"] == 2
    assert planner.last_filter_summary["candidate_variant_count"] == 4
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

    with pytest.raises(RuntimeError, match="No visible collision-free"):
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

    assert planner.last_filter_summary["target_solid_collision_rejected"] == 1
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


def test_noncontact_endpoint_uses_open_jaws_not_estimated_closure_width(tmp_path: Path) -> None:
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
    planner.start()
    try:
        result = planner.plan(Image.new("RGB", (80, 80)), depth, pixel_uv=(40, 40),
            intrinsics={"width": 80, "height": 80, "fx": 1000., "fy": 1000., "cx": 40., "cy": 40.},
            base_from_camera=np.eye(4), max_candidates=1)
        candidate = result.candidates[0]
        pose = np.asarray(candidate.camera_from_tcp).reshape(4, 4)
        from robot_motion.grasp_visible_scene import VisibleGraspScene
        scene = VisibleGraspScene.from_depth(depth_m=depth, target_mask=mask,
            intrinsics={"fx": 1000., "fy": 1000., "cx": 40., "cy": 40.},
            reference_from_camera=np.eye(4), geometry=planner.solid_geometry)
        assert scene.collision_counts(pose, opening_m=candidate.gripper_width_m)[0] > 0
        assert scene.approach_collision_counts(pose, opening_m=.07, distance_m=.06) == (0, 0)
        assert candidate.visible_collision_free
        assert candidate.model_score == pytest.approx(.9)
        np.testing.assert_allclose(
            planner.solid_geometry.solid_pose(pose), model_pose, atol=1e-9
        )
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

    with pytest.raises(RuntimeError, match='"target_capture_valid":0'):
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
