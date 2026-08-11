from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from robot_motion.arm_init import ArmQposTransitionConfig
from robot_motion.joint_limits import JointVelocityLimitProfile
from robot_motion.rm65.arm_init.qpos_sequence import (
    QposSequenceEntry,
    QposTestSequence,
    build_qpos_test_sequence_from_dataset,
    dry_run_qpos_test_sequence,
    load_qpos_test_sequence,
    sample_episode_qpos_records,
    save_qpos_test_sequence,
)


def _qpos(left0: float = 0.0, right0: float = 0.0) -> dict[str, np.ndarray]:
    left = np.zeros(6, dtype=float)
    right = np.zeros(6, dtype=float)
    left[0] = float(left0)
    right[0] = float(right0)
    return {"left": left, "right": right}


def _test_transition_config() -> ArmQposTransitionConfig:
    return ArmQposTransitionConfig(
        dt_sec=0.1,
        joint_velocity_profile=JointVelocityLimitProfile(
            name="test_rm65_sequence",
            max_velocity_deg_s=(10.0,) * 6,
            max_dt_sec=1.0,
        ),
    )


def test_qpos_test_sequence_json_round_trip(tmp_path: Path) -> None:
    sequence = QposTestSequence(
        name="unit_sequence",
        entries=(
            QposSequenceEntry(index=0, name="a", qpos_by_side=_qpos(left0=0.1, right0=-0.1)),
        ),
        metadata={"dataset_root": "dummy"},
    )
    path = tmp_path / "sequence.json"

    save_qpos_test_sequence(sequence, path)
    loaded = load_qpos_test_sequence(path)

    assert loaded.name == "unit_sequence"
    assert loaded.metadata["dataset_root"] == "dummy"
    np.testing.assert_allclose(loaded.entries[0].qpos_by_side["left"], _qpos(left0=0.1)["left"])


def test_dry_run_qpos_test_sequence_chains_targets() -> None:
    sequence = QposTestSequence(
        name="chain",
        entries=(
            QposSequenceEntry(index=0, name="first", qpos_by_side=_qpos(left0=np.deg2rad(5.0))),
            QposSequenceEntry(index=1, name="second", qpos_by_side=_qpos(left0=np.deg2rad(8.0))),
        ),
    )

    report = dry_run_qpos_test_sequence(
        sequence=sequence,
        current_qpos_by_side=_qpos(),
        config=_test_transition_config(),
    )

    assert report["target_count"] == 2
    assert report["completed_count"] == 2
    assert report["results"][0]["summary"]["step_count"] == 5
    assert report["results"][1]["summary"]["step_count"] == 3
    np.testing.assert_allclose(report["final_qpos_by_side"]["left"], _qpos(left0=np.deg2rad(8.0))["left"])


def test_dry_run_qpos_test_sequence_rejects_j6_outside_120_deg() -> None:
    target = _qpos()
    target["right"][5] = np.deg2rad(120.1)
    sequence = QposTestSequence(
        name="invalid_j6",
        entries=(QposSequenceEntry(index=7, name="invalid", qpos_by_side=target),),
    )

    with pytest.raises(ValueError, match=r"sequence_entry_7\.right\.joint6=.*outside"):
        dry_run_qpos_test_sequence(
            sequence=sequence,
            current_qpos_by_side=_qpos(),
            config=_test_transition_config(),
        )


@pytest.mark.optional
def test_sample_episode_qpos_records_from_parquet(tmp_path: Path) -> None:
    pq, pa = _require_pyarrow()
    episode_path = tmp_path / "episode_000000.parquet"
    _write_episode_parquet(pq, pa, episode_path, episode_index=0, row_count=5)

    records, summary = sample_episode_qpos_records(
        path=episode_path,
        samples_per_episode=3,
        rng=random.Random(0),
        active_sides=("left", "right"),
    )

    assert len(records) == 3
    assert summary["row_count"] == 5
    assert summary["finite_qpos_row_count"] == 5
    assert summary["absolute_limit_rejected_row_count"] == 1
    assert summary["valid_qpos_row_count"] == 4
    assert summary["sampled_row_indices"] == sorted(summary["sampled_row_indices"])
    for record in records:
        assert record["qpos_by_side"]["left"].shape == (6,)
        assert record["qpos_by_side"]["right"].shape == (6,)


@pytest.mark.optional
def test_build_qpos_test_sequence_from_dataset_samples_each_episode(tmp_path: Path) -> None:
    pq, pa = _require_pyarrow()
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    _write_episode_parquet(pq, pa, data_dir / "episode_000000.parquet", episode_index=0, row_count=4)
    _write_episode_parquet(pq, pa, data_dir / "episode_000001.parquet", episode_index=1, row_count=4)

    sequence = build_qpos_test_sequence_from_dataset(
        dataset_root=tmp_path,
        samples_per_episode=2,
        seed=1,
        active_sides=("left", "right"),
    )

    assert sequence.name == f"{tmp_path.name}_2_per_episode_seed1"
    assert len(sequence.entries) == 4
    assert sequence.metadata["episode_count"] == 2
    assert [entry.index for entry in sequence.entries] == [0, 1, 2, 3]
    assert all(entry.source["episode_path"].startswith("data/chunk-000/") for entry in sequence.entries)


def _write_episode_parquet(pq, pa, path: Path, *, episode_index: int, row_count: int) -> None:
    rows = list(range(row_count))
    left = [[float(row), 0.0, 0.0, 0.0, 0.0, 0.0] for row in rows]
    right = [[-float(row), 0.0, 0.0, 0.0, 0.0, 0.0] for row in rows]
    table = pa.table(
        {
            "episode_index": [episode_index] * row_count,
            "frame_index": rows,
            "timestamp": [float(row) * 0.1 for row in rows],
            "observation.arm.left.current_qpos": left,
            "observation.arm.right.current_qpos": right,
        }
    )
    pq.write_table(table, path)


def _require_pyarrow():
    pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pq, pa
