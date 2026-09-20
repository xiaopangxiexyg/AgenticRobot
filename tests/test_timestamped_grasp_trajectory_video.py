from __future__ import annotations

import numpy as np
import pytest

from experiments.sim_point_grasp.commands import (
    record_timestamped_trajectory as recorder_module,
)
from experiments.sim_point_grasp.commands.record_timestamped_trajectory import (
    _annotate_frame,
    build_parser,
    infer_uniform_sample_hz,
)
from teleop_runtime.skill_deploy.grasp_trajectory_record import (
    write_grasp_trajectory_record,
)
from teleop_runtime.skill_deploy.motion_port import RealAgentJointTrajectory


def test_infer_uniform_sample_hz_preserves_production_60hz_lattice() -> None:
    timestamps = np.arange(2025, dtype=float) / 60.0
    assert infer_uniform_sample_hz(timestamps) == pytest.approx(60.0)
    assert timestamps[-1] == pytest.approx(33.733333333333334)


def test_infer_uniform_sample_hz_rejects_variable_rate_video_claim() -> None:
    with pytest.raises(ValueError, match="uniform trajectory timestamp lattice"):
        infer_uniform_sample_hz(np.asarray((0.0, 1.0 / 60.0, 0.04)))


def test_annotation_preserves_frame_shape_and_marks_timing() -> None:
    rgb = np.full((48, 64, 3), 127, dtype=np.uint8)
    annotated = _annotate_frame(
        rgb,
        elapsed_sec=1.0,
        duration_sec=2.0,
        sample_index=60,
        sample_count=121,
    )
    assert annotated.shape == rgb.shape
    assert annotated.dtype == np.uint8
    assert not np.array_equal(annotated, rgb)


def test_cli_defaults_to_blue_cube_front_camera() -> None:
    args = build_parser().parse_args(
        ["--trajectory-record", "plan.npz", "--output-dir", "output"]
    )
    assert args.camera == "front"
    assert args.width == 640
    assert args.height == 480
    assert args.scene_contract.name == "rm65_bimanual_blue_cube_v1.json"


def test_recording_finishes_without_a_scene_close_contract(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = RealAgentJointTrajectory(
        time_from_start_sec=np.asarray((0.0, 1.0 / 60.0, 2.0 / 60.0)),
        qpos_by_side={
            "left": np.zeros((3, 6)),
            "right": np.ones((3, 6)) * 0.1,
        },
        planner_evidence_refs=("test:timestamped-video",),
    )
    trajectory_dir = tmp_path / "trajectory"
    trajectory_dir.mkdir()
    record = write_grasp_trajectory_record(
        trajectory_dir,
        trajectory,
        moving_side="right",
        scene_hash="scene:test",
        collision_scope="test",
        collision_scene_profile="test",
        target_tool_tcp_world=np.eye(4),
        speed_scale=0.2,
        stage_boundaries=(),
    )
    scene_contract = tmp_path / "scene.json"
    scene_contract.write_text("{}", encoding="utf-8")

    class _Data:
        qvel = np.zeros(4)

    class _Scene:
        data = _Data()

        def set_gripper_close(self, _name, _value):
            return None

        def set_qpos_by_side(self, _value):
            return None

        def forward(self):
            return None

    class _Renderer:
        def __init__(self, _scene, *, width, height):
            self.frame = np.zeros((height, width, 3), dtype=np.uint8)

        def render(self, _camera):
            return self.frame

        def close(self):
            return None

    class _Writer:
        def __init__(self):
            self.frames = 0
            self.released = False

        def isOpened(self):
            return True

        def write(self, _frame):
            self.frames += 1

        def release(self):
            self.released = True

    writer = _Writer()
    monkeypatch.setattr(recorder_module, "resolve_path", lambda path: path)
    monkeypatch.setattr(
        recorder_module.SimulationContract, "load", lambda _path: object()
    )
    monkeypatch.setattr(
        recorder_module.MujocoScene, "from_contract", lambda *_args, **_kwargs: _Scene()
    )
    monkeypatch.setattr(recorder_module, "MujocoCameraRenderer", _Renderer)
    monkeypatch.setattr(recorder_module.cv2, "VideoWriter", lambda *_args: writer)

    report = recorder_module.record_timestamped_trajectory(
        trajectory_record=record,
        output_dir=tmp_path / "output",
        scene_contract_path=scene_contract,
        camera_name="front",
        width=64,
        height=48,
        left_gripper_close=0.2,
        right_gripper_close=0.3,
    )
    assert report["status"] == "recorded"
    assert writer.frames == 3
    assert writer.released is True
