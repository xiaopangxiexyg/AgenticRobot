from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def _load_probe_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "probes"
        / "rm65_vive_velocity_execution_probe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "rm65_vive_velocity_execution_probe_test_target",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_causal_pose_twist_uses_only_past_60hz_rows() -> None:
    probe = _load_probe_module()
    count = 20
    dt = 1.0 / 60.0
    times = np.arange(count, dtype=float) * dt
    poses = np.repeat(np.eye(4, dtype=float)[None, :, :], count, axis=0)
    poses[:, 0, 3] = 0.12 * times
    poses[:, :3, :3] = Rotation.from_rotvec(
        np.column_stack(
            (
                np.zeros(count),
                np.zeros(count),
                0.60 * times,
            )
        )
    ).as_matrix()
    valid = np.ones(count, dtype=bool)

    twist, estimated, metadata = probe._causal_pose_twist_60hz(
        poses,
        times,
        valid,
        window_rows=5,
        efficiency_power=1.0,
    )

    np.testing.assert_allclose(twist[4:, 0], 0.12, atol=1e-10)
    np.testing.assert_allclose(twist[4:, 5], 0.60, atol=1e-10)
    np.testing.assert_allclose(twist[4:, 1:3], 0.0, atol=1e-12)
    np.testing.assert_allclose(twist[4:, 3:5], 0.0, atol=1e-12)
    assert metadata["estimated_ratio"] > 0.8
    assert np.all(estimated[2:])

    changed_future = poses.copy()
    changed_future[12:, 0, 3] += 10.0
    changed_twist, _changed_estimated, _ = probe._causal_pose_twist_60hz(
        changed_future,
        times,
        valid,
        window_rows=5,
        efficiency_power=1.0,
    )
    # Modifying future samples cannot alter an earlier estimate.
    np.testing.assert_array_equal(changed_twist[:12], twist[:12])


def test_causal_pose_twist_rejects_invalid_or_stalled_intervals() -> None:
    probe = _load_probe_module()
    poses = np.repeat(np.eye(4, dtype=float)[None, :, :], 8, axis=0)
    poses[:, 1, 3] = np.arange(8, dtype=float) * 0.001
    times = np.arange(8, dtype=float) / 60.0
    valid = np.ones(8, dtype=bool)
    valid[3:6] = False

    twist, estimated, metadata = probe._causal_pose_twist_60hz(
        poses,
        times,
        valid,
        window_rows=5,
        efficiency_power=1.0,
    )

    np.testing.assert_array_equal(twist[3:7], np.zeros((4, 6), dtype=float))
    assert not np.any(estimated[3:7])
    assert metadata["estimated_ratio"] < 0.5


def test_midcut_q_qdd_candidate_preserves_formal_action_contract() -> None:
    probe = _load_probe_module()
    count = 48
    phase = np.arange(count, dtype=float) / 60.0
    qref = np.zeros((count, 6), dtype=float)
    qref[:, 0] = 0.08 * phase
    qref[:, 1] = -0.05 * phase
    qref[:, 3] = 0.03 * phase
    velocity = np.zeros_like(qref)
    velocity[:, 0] = 0.08
    velocity[:, 1] = -0.05
    velocity[:, 3] = 0.03
    valid = np.ones(count, dtype=bool)
    candidate = next(
        item
        for item in probe._candidates()
        if item.name == "30_vive_dls_midcut_q_qdd_anchor_a50_c35"
    )

    result = probe._simulate(
        qref,
        candidate,
        external_velocity_60hz=velocity,
        external_velocity_valid_60hz=valid,
    )
    metrics = probe._evaluate_simulation(result)

    assert metrics["action_next_state_error_deg"]["max"] == 0.0
    assert metrics["position_violation_count"] == 0
    assert metrics["velocity_violation_count"] == 0
    assert metrics["acceleration_violation_count"] == 0
    assert metrics["jerk_violation_count"] == 0
