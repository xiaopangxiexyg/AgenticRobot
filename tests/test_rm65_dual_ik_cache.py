from __future__ import annotations

import numpy as np
import pytest

from robot_motion.rm65.assets import RM65_ASSET_DIR
from robot_motion.rm65.ik import (
    RM65DualArmCollisionIKConfig,
    build_rm65_dual_arm_collision_ik_solver,
)


def _build_kinematic_solver():
    """Build the production dual-arm solver without optional collision pairs."""

    if not RM65_ASSET_DIR.exists():
        pytest.skip("reviewed local RM65 motion-model assets are unavailable")
    pytest.importorskip("pinocchio")
    pytest.importorskip("pink")
    pytest.importorskip("qpsolvers")
    return build_rm65_dual_arm_collision_ik_solver(
        RM65DualArmCollisionIKConfig(collision_constraints_enabled=False)
    )


def test_set_configuration_q_refreshes_pink_kinematic_cache() -> None:
    """A rebase must update frame placements, not only the public q vector."""

    bundle = _build_kinematic_solver()
    solver = bundle.solver
    q = solver.current_q
    q[:6] += np.asarray([0.015, -0.012, 0.010, -0.008, 0.006, -0.004])

    # Compute the reference pose with an independent Pinocchio data object so
    # the assertion cannot accidentally refresh Pink's cached configuration.
    expected_data = solver.pin.Data(solver.robot.model)
    solver.pin.forwardKinematics(solver.robot.model, expected_data, q)
    solver.pin.updateFramePlacements(solver.robot.model, expected_data)
    frame_name = bundle.left_frame_name
    frame_id = solver.robot.model.getFrameId(frame_name)
    expected_transform = expected_data.oMf[frame_id]

    solver.set_configuration_q(q)
    cached_transform = solver.configuration.get_transform_frame_to_world(frame_name)

    np.testing.assert_allclose(
        cached_transform.translation,
        expected_transform.translation,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        cached_transform.rotation,
        expected_transform.rotation,
        atol=1e-12,
    )


def test_set_configuration_q_rebase_is_safe_for_both_arm_frames() -> None:
    """The shared setter refreshes both coupled arm branches after a rebase."""

    bundle = _build_kinematic_solver()
    solver = bundle.solver
    q = solver.current_q
    q += np.asarray(
        [0.010, -0.008, 0.006, -0.004, 0.002, -0.001,
         -0.009, 0.007, -0.005, 0.003, -0.002, 0.001],
        dtype=float,
    )
    expected_data = solver.pin.Data(solver.robot.model)
    solver.pin.forwardKinematics(solver.robot.model, expected_data, q)
    solver.pin.updateFramePlacements(solver.robot.model, expected_data)

    solver.set_configuration_q(q)
    for frame_name in (bundle.left_frame_name, bundle.right_frame_name):
        frame_id = solver.robot.model.getFrameId(frame_name)
        expected_transform = expected_data.oMf[frame_id]
        cached_transform = solver.configuration.get_transform_frame_to_world(frame_name)
        np.testing.assert_allclose(
            cached_transform.translation,
            expected_transform.translation,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            cached_transform.rotation,
            expected_transform.rotation,
            atol=1e-12,
        )
