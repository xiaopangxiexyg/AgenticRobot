from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from utils.portable_vis import shared_episode_pose as pose


def test_episode_selection_preserves_id_and_index_matching() -> None:
    path = Path("data/chunk-000/episode_000012.parquet")
    assert pose._episode_index_from_path(path) == 12
    assert pose._episode_index_from_path(Path("episode_12_extra.parquet")) is None
    assert pose._episode_id_from_index(12, "unused") == "episode_000012"
    assert pose._episode_id_from_index(None, "fallback") == "fallback"
    assert pose._episode_matches({}, "12", path)
    assert pose._episode_matches({"episode_id": "custom"}, "custom", path)
    assert pose._episode_matches({"episode_index": "7"}, "7", path)
    assert not pose._episode_matches({"episode_index": "bad"}, "12", path)
    assert not pose._episode_matches({}, "", path)


def test_jsonl_reader_keeps_object_records_and_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    assert pose._read_jsonl(path) == []
    path.write_text('\n{"episode_id": "a"}\n[1, 2]\n{"episode_id": "b"}\n')
    assert pose._read_jsonl(path) == [{"episode_id": "a"}, {"episode_id": "b"}]
    path.write_text("not json\n")
    with pytest.raises(json.JSONDecodeError):
        pose._read_jsonl(path)


def test_numeric_helpers_preserve_finite_and_shape_guards() -> None:
    np.testing.assert_array_equal(pose._vector_or_none([1, 2, 3], length=3), [1, 2, 3])
    assert pose._vector_or_none([1, float("nan"), 3], length=3) is None
    assert pose._vector_or_none([1, 2], length=3) is None
    assert pose._int_or_none("5") == 5
    assert pose._int_or_none(None) is None
    assert pose._float_or_none("2.5") == 2.5
    assert pose._float_or_none(float("inf")) is None
    assert pose._downsample_indices(10, 3) == [0, 4, 9]
    assert pose._downsample_indices(3, 10) == [0, 1, 2]


@pytest.mark.parametrize(
    "quaternion",
    [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.25, -0.5, 0.75, 0.125],
    ],
)
def test_rotation_roundtrip_across_trace_branches(quaternion: list[float]) -> None:
    matrix = pose._quat_xyzw_to_matrix(quaternion)
    assert matrix is not None
    recovered = pose._rotation_matrix_to_quat_xyzw(matrix)
    assert recovered is not None
    np.testing.assert_allclose(pose._quat_xyzw_to_matrix(recovered), matrix, atol=1e-12)


def test_invalid_quaternions_and_matrix_are_rejected() -> None:
    assert pose._quat_xyzw_to_matrix([0.0, 0.0, 0.0, 0.0]) is None
    assert pose._quat_xyzw_to_matrix([0.0, 0.0, float("nan"), 1.0]) is None
    assert pose._quat_xyzw_to_matrix([1.0, 2.0, 3.0]) is None
    assert pose._rotation_matrix_to_quat_xyzw(np.full((3, 3), np.nan)) is None


@pytest.mark.parametrize(
    "script",
    [
        "eef_vis/visualize_eef_trajectory.py",
        "eef_vis/visualize_pose_action_alignment.py",
        "real_projection_view/project_real_trajectory.py",
    ],
)
def test_direct_script_help_from_outside_repo(tmp_path: Path, script: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "utils" / "portable_vis" / script
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
