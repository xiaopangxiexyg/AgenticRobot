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
        return np.stack((first, second)), np.asarray((0.2, 0.9)), [], None

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
    assert scores.tolist() == pytest.approx([0.9])
    np.testing.assert_allclose(grasps[0], second)
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
    assert request["sweep_volume_params"]["gripper_type"] == 0
    np.testing.assert_allclose(
        request["sweep_volume_params"]["extents_open"],
        (0.070, 0.0255, 0.050),
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


def test_planner_infers_in_base_frame_and_converts_pose_to_calibrated_dmtac_tcp(
    tmp_path: Path,
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
    base_from_camera[:3, 3] = (0.20, -0.10, 0.30)
    base_from_model_base = base_from_camera @ camera_from_model_base
    client = _PlannerClient(
        base_from_model_base[None], np.asarray((0.9,), dtype=np.float64)
    )
    planner = Sam2GraspGenXPointGraspPlanner(
        model_spec=_model_spec(tmp_path),
        point_segmenter=_Segmenter(mask),
        client=client,
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
    assert client.calls[0][1]["top_k"] == 1_000
    candidate = result.candidates[0]
    camera_from_tcp = np.asarray(candidate.camera_from_tcp).reshape(4, 4)
    assert camera_from_tcp[2, 3] == pytest.approx(1.017999, abs=1e-6)
    assert 0.005 <= candidate.gripper_width_m <= 0.070
    assert candidate.visible_collision_free is True
    assert planner.last_target_mask is not None
    assert "graspgenx_inference" in planner.last_plan_timing_ms
    assert planner.last_filter_summary["detected"] == 1
    assert planner.last_filter_summary["width_valid"] == 1
    assert planner.last_filter_summary["target_capture_valid"] == 1

    planner.close()
    assert client.closed_for_test


def test_planner_preserves_closing_axis_variants_and_groups_finger_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    grasps.append(tilted)
    client = _PlannerClient(
        np.stack(grasps), np.asarray((0.9, 0.8, 0.7, 0.65, 0.6), dtype=np.float64)
    )
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
        max_candidates=5,
    )

    assert [candidate.model_score for candidate in result.candidates] == pytest.approx(
        [0.9, 0.8, 0.7, 0.6]
    )
    first = np.asarray(result.candidates[0].camera_from_tcp).reshape(4, 4)
    tilted_result = np.asarray(result.candidates[-1].camera_from_tcp).reshape(4, 4)
    assert np.dot(first[:3, 2], tilted_result[:3, 2]) == pytest.approx(
        math.cos(tilted_angle)
    )
    planner.close()


def test_candidate_selection_covers_distinct_approaches_before_yaw_variants() -> None:
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


def test_planner_bounded_centering_recovers_offset_grasp_without_changing_depth(
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

    candidate = np.asarray(result.candidates[0].camera_from_tcp).reshape(4, 4)
    # The target centre is moved onto the physical closing/height centre, but
    # the model's approach depth is not altered by this admission correction.
    assert candidate[0, 3] == pytest.approx(0.0, abs=1e-9)
    assert candidate[1, 3] == pytest.approx(0.0, abs=1e-9)
    assert candidate[2, 3] == pytest.approx(1.017999, abs=1e-9)
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
        np.testing.assert_allclose(rotation, original_rotation, atol=3e-8, rtol=0.0)
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
