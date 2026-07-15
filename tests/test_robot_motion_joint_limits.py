from __future__ import annotations

import numpy as np

from robot_motion.joint_limits import (
    JointRateLimiter,
    JointVelocityLimitProfile,
    limit_joint_step,
    load_joint_velocity_limit_profiles,
    plan_limited_joint_path,
)


def test_limit_joint_step_clips_each_joint_by_velocity_profile():
    profile = JointVelocityLimitProfile(name="test", max_velocity_deg_s=(10.0, 20.0))

    result = limit_joint_step(
        reference_qpos=np.deg2rad([0.0, 0.0]),
        target_qpos=np.deg2rad([30.0, -30.0]),
        profile=profile,
        dt_sec=0.1,
    )

    assert result.limited
    assert np.allclose(np.rad2deg(result.qpos), [1.0, -2.0])
    assert np.allclose(result.max_step_deg, [1.0, 2.0])


def test_joint_velocity_profile_caps_effective_dt_for_long_runtime_gaps():
    profile = JointVelocityLimitProfile(name="test", max_velocity_deg_s=(10.0, 20.0), max_dt_sec=1.0)

    assert profile.effective_dt_sec(0.2) == 0.2
    assert profile.effective_dt_sec(2.0) == 1.0
    assert np.allclose(profile.max_step_deg(2.0), [10.0, 20.0])


def test_rate_limiter_chases_latest_target_without_replaying_old_target():
    profile = JointVelocityLimitProfile(name="test", max_velocity_deg_s=(10.0,))
    limiter = JointRateLimiter(profile)

    first = limiter.step(
        reference_qpos=np.deg2rad([0.0]),
        target_qpos=np.deg2rad([30.0]),
        dt_sec=0.1,
    )
    second = limiter.step(target_qpos=np.deg2rad([-30.0]), dt_sec=0.1)

    assert np.allclose(np.rad2deg(first.qpos), [1.0])
    assert np.allclose(np.rad2deg(second.qpos), [0.0])


def test_plan_limited_joint_path_reaches_target_with_bounded_steps():
    profile = JointVelocityLimitProfile(name="test", max_velocity_deg_s=(10.0,))

    steps = plan_limited_joint_path(
        start_qpos=np.deg2rad([0.0]),
        target_qpos=np.deg2rad([5.0]),
        profile=profile,
        dt_sec=0.2,
    )

    assert len(steps) == 3
    assert all(abs(step.delta_deg[0]) <= 2.0 + 1e-9 for step in steps)
    assert np.allclose(np.rad2deg(steps[-1].qpos), [5.0])


def test_default_rm65_joint_velocity_profile_loads():
    profiles = load_joint_velocity_limit_profiles()

    assert "rm65" in profiles
    rm65 = profiles["rm65"]
    assert rm65.dof == 6
    assert rm65.max_dt_sec == 1.0
    assert np.allclose(rm65.max_velocity_deg_s, [85.0, 85.0, 95.0, 105.0, 120.0, 120.0])
    assert rm65.max_step_deg(0.05).shape == (6,)
    assert np.isclose(rm65.max_step_deg(0.2)[3], 21.0)
    assert np.isclose(rm65.max_step_deg(1.0)[3], 105.0)

    assert "rm65_init_slow" in profiles
    rm65_init_slow = profiles["rm65_init_slow"]
    assert rm65_init_slow.dof == 6
    assert rm65_init_slow.max_dt_sec == 0.25
    assert np.allclose(rm65_init_slow.max_velocity_deg_s, [6.0, 6.0, 7.0, 8.0, 10.0, 10.0])
    assert np.isclose(rm65_init_slow.max_step_deg(0.05)[0], 0.3)
    assert np.isclose(rm65_init_slow.max_step_deg(1.0)[5], 2.5)
