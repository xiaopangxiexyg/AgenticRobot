from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim.mujoco.dataset_v22 import (
    MujocoV22DatasetWriter,
    MujocoV22DatasetWriterConfig,
    load_rm65_bimanual_dmtac_layout,
)
from sim.mujoco.replay.trajectory import GripperReplaySample, QposSample


class _FakeScene:
    arm_specs = {"left": SimpleNamespace(), "right": SimpleNamespace()}
    conversion = SimpleNamespace(mjcf_path=Path("/tmp/rm65.xml"))

    @staticmethod
    def body_pose(name: str) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        if name == "left_hand_base":
            pose[0, 3] = 0.1
        elif name == "right_hand_base":
            pose[0, 3] = 0.2
        elif name not in {"left_arm_base_link", "right_arm_base_link"}:
            raise KeyError(name)
        return pose

    @staticmethod
    def camera_pose(name: str) -> np.ndarray:
        if name != "front":
            raise KeyError(name)
        return np.eye(4, dtype=np.float64)

    @staticmethod
    def camera_metadata() -> dict[str, object]:
        return {"front": {"width": 6, "height": 4}}


def test_rm65_bimanual_dmtac_layout_is_the_real_collector_contract() -> None:
    layout = load_rm65_bimanual_dmtac_layout()

    assert layout.name == "dual_arm_bimanual_dmtac_gripper"
    assert layout.action_dim == 16
    assert layout.state_dim == 16
    for side in ("left", "right"):
        channel = layout.channel("action", side)
        assert layout.end_effector(side).adapter == "dmtac_gripper_v1"
        assert channel.dim == 1
        assert channel.units == "normalized_0_1"


@pytest.mark.optional
def test_native_mujoco_v22_writer_binds_layout_and_marks_force_unavailable(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    pq = pytest.importorskip("pyarrow.parquet")
    writer = MujocoV22DatasetWriter(
        _FakeScene(),
        MujocoV22DatasetWriterConfig(
            root_dir=tmp_path,
            nominal_fps=15.0,
            gripper_name_by_side={"left": "left_gripper", "right": "right_gripper"},
        ),
        camera_names=("front",),
    )
    sample = QposSample(
        frame_index=7,
        timestamp=10.0,
        relative_time_sec=0.0,
        qpos_by_side={"left": np.zeros(6), "right": np.ones(6)},
        source_by_side={"left": "source.left", "right": "source.right"},
    )
    gripper = GripperReplaySample(
        frame_index=7,
        timestamp=10.0,
        relative_time_sec=0.0,
        close_by_name={"left_gripper": 0.0, "right_gripper": 1.0},
        close_by_side={"left": 0.0, "right": 1.0},
        source_by_name={"left_gripper": "source.left", "right_gripper": "source.right"},
        source_by_side={"left": "source.left", "right": "source.right"},
    )
    writer.record_frame(
        sample=sample,
        qpos_result=sample.qpos_by_side,
        gripper_sample=gripper,
        gripper_result={"left_gripper": 0.0, "right_gripper": 1.0},
        images={"front": np.zeros((48, 64, 3), dtype=np.uint8)},
        playback_time_sec=0.0,
        active_sides=("left", "right"),
    )
    writer.close()

    info = json.loads((tmp_path / "meta" / "info.json").read_text(encoding="utf-8"))
    validity = json.loads((tmp_path / "meta" / "modality_validity.json").read_text(encoding="utf-8"))
    row = pq.read_table(tmp_path / "data/chunk-000/episode_000000.parquet").to_pylist()[0]

    assert info["dexteleop_dataset_schema"] == "dexteleop_dataset_v2.2"
    assert info["policy_io_layout_hash"] == writer.layout.layout_hash
    assert validity["modalities"]["force_torque"]["available"] is False
    assert row["action"][-2:] == pytest.approx([0.0, 1.0])
    assert row["observation.state"][-2:] == pytest.approx([0.0, 1.0])
    assert len(row["observation.arm.qpos"]) == 12
