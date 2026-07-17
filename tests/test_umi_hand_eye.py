from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dataset_contracts.pose_frames import PoseFrameContract
from utils.hand_eye.transforms import invert_transform, make_transform, transform_distance
from utils.umi_hand_eye.calibration import (
    UmiCalibrationResult,
    UmiCalibrationSample,
    load_result,
    load_samples,
    save_result,
    save_samples,
    solve_umi_hand_eye,
)
from utils.umi_hand_eye.config import load_config


def _transform(rpy_deg, translation):
    return make_transform(Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix(), translation)


def _synthetic_samples(count: int = 30):
    rng = np.random.default_rng(42)
    T_camera_to_tracker = _transform([13.0, -21.0, 8.0], [0.035, -0.018, 0.062])
    T_board_to_vive = _transform([-7.0, 16.0, 31.0], [0.45, 0.12, 1.08])
    samples = []
    for index in range(count):
        T_tracker_to_vive = make_transform(
            Rotation.from_rotvec(rng.uniform(-1.1, 1.1, size=3)).as_matrix(),
            rng.uniform([-0.35, -0.25, 0.55], [0.35, 0.25, 1.25]),
        )
        T_board_to_camera = (
            invert_transform(T_camera_to_tracker)
            @ invert_transform(T_tracker_to_vive)
            @ T_board_to_vive
        )
        samples.append(
            UmiCalibrationSample(
                T_tracker_to_vive_world=T_tracker_to_vive,
                T_board_to_camera=T_board_to_camera,
                metadata={"sample_index": index},
            )
        )
    return samples, T_camera_to_tracker, T_board_to_vive


def test_solve_recovers_explicit_umi_transforms():
    samples, expected_camera_to_tracker, expected_board_to_vive = _synthetic_samples()
    created = datetime(2026, 7, 17, 1, 2, 3, tzinfo=timezone.utc)
    result = solve_umi_hand_eye(
        samples,
        side="left",
        camera_serial="camera-left",
        tracker_name="tracker_1",
        tracker_role="left_wrist",
        validity_seconds=3600.0,
        created_at=created,
        refine=True,
    )
    translation, rotation = transform_distance(
        result.T_camera_to_tracker, expected_camera_to_tracker
    )
    assert translation < 1e-6
    assert rotation < 1e-4
    translation, rotation = transform_distance(
        result.T_board_to_vive_world, expected_board_to_vive
    )
    assert translation < 1e-6
    assert rotation < 1e-4
    assert np.allclose(result.T_vive_world_to_desk, invert_transform(expected_board_to_vive))
    assert result.expires_at_utc == "2026-07-17T02:02:03Z"
    json.dumps(result.to_dict())


def test_result_expiry_binding_and_pose_contract():
    created = datetime(2026, 7, 17, tzinfo=timezone.utc)
    result = UmiCalibrationResult(
        T_camera_to_tracker=np.eye(4),
        T_board_to_vive_world=_transform([0, 0, 90], [1, 2, 3]),
        created_at_utc="2026-07-17T00:00:00Z",
        expires_at_utc="2026-07-17T01:00:00Z",
        side="left",
        camera_serial="cam",
        tracker_name="tracker_1",
        tracker_role="left_wrist",
        method="test",
        metrics={},
        sample_count=25,
    )
    result.require_valid(
        when=created + timedelta(minutes=30),
        camera_serial="cam",
        tracker_name="tracker_1",
    )
    with pytest.raises(RuntimeError, match="expired"):
        result.require_valid(when=created + timedelta(hours=2))
    with pytest.raises(RuntimeError, match="camera serial"):
        result.require_valid(when=created, camera_serial="other")

    contract = PoseFrameContract.from_dict(result.pose_frame_contract())
    assert contract.pose_storage_frame == "desk_absolute"
    assert np.allclose(
        contract.source.effective_raw_to_desk_transform("right"),
        result.T_vive_world_to_desk,
    )


def test_sample_result_and_default_config_round_trip(tmp_path):
    samples, _, _ = _synthetic_samples(3)
    samples_path = tmp_path / "samples.json"
    save_samples(samples_path, samples, metadata={"side": "left"})
    restored, metadata = load_samples(samples_path)
    assert len(restored) == 3
    assert metadata == {"side": "left"}
    assert np.allclose(
        restored[0].T_tracker_to_vive_world, samples[0].T_tracker_to_vive_world
    )

    result = UmiCalibrationResult(
        T_camera_to_tracker=np.eye(4),
        T_board_to_vive_world=np.eye(4),
        created_at_utc="2026-07-17T00:00:00Z",
        expires_at_utc="2026-07-17T01:00:00Z",
        side="right",
        camera_serial="cam",
        tracker_name="tracker_2",
        tracker_role="right_wrist",
        method="test",
        metrics={"value": 1.0},
        sample_count=25,
    )
    result_path = tmp_path / "result.json"
    save_result(result_path, result)
    assert load_result(result_path).tracker_name == "tracker_2"
    json.loads(result_path.read_text())

    config_path = Path(__file__).parents[1] / "utils/umi_hand_eye/configs/rm65_dual_wrist_umi_v1.json"
    config = load_config(config_path)
    assert config.validity_seconds == 3600.0
    assert config.side("left").camera_serial == "261122273924"
    assert config.side("right").camera_serial == "260522278055"
