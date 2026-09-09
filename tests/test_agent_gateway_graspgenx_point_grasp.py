from __future__ import annotations

import json
from pathlib import Path

import msgpack
import numpy as np
import pytest
from PIL import Image

from deployments.perception.graspgenx_client import (
    GraspGenXClient,
    pack_graspgenx_message,
    unpack_graspgenx_message,
)
from deployments.perception.sam2_graspgenx_point_grasp import (
    DMTAC_M_GRASPGENX_SWEEP_VOLUME_V1,
    GRASPGENX_CHECKPOINT_REVIEWED_REVISION,
    GRASPGENX_REVIEWED_REVISION,
    GraspGenXPointGraspModelSpec,
    Sam2GraspGenXPointGraspPlanner,
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


def test_planner_converts_gripper_base_pose_to_calibrated_dmtac_tcp(
    tmp_path: Path,
) -> None:
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[8:13, 8:13] = 1.0
    mask = depth > 0.0
    camera_from_model_base = np.eye(4, dtype=np.float64)
    camera_from_model_base[2, 3] = 0.975
    client = _PlannerClient(
        camera_from_model_base[None], np.asarray((0.9,), dtype=np.float64)
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
        base_from_camera=np.eye(4),
        max_candidates=1,
    )

    assert client.started
    assert len(client.calls) == 1
    assert client.calls[0][0].shape == (25, 3)
    assert client.calls[0][1]["planner"] == "graspmoe"
    candidate = result.candidates[0]
    camera_from_tcp = np.asarray(candidate.camera_from_tcp).reshape(4, 4)
    assert camera_from_tcp[2, 3] == pytest.approx(1.017999, abs=1e-6)
    assert 0.005 <= candidate.gripper_width_m <= 0.070
    assert candidate.visible_collision_free is True
    assert planner.last_target_mask is not None
    assert "graspgenx_inference" in planner.last_plan_timing_ms

    planner.close()
    assert client.closed_for_test


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

    with pytest.raises(RuntimeError, match="No visible collision-free"):
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
