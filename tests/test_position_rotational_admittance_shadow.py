from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.position_control.cartesian_admittance import (
    CartesianAdmittanceConfig,
    discrete_lowpass_alpha_to_tau,
)
from robot_motion.position_control.rotational_admittance import (
    ROTATIONAL_ADMITTANCE_PROFILE_BALANCED_5DEG,
    ROTATIONAL_ADMITTANCE_PROFILE_CAUTIOUS_3DEG,
    ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG,
    RotationalAdmittanceConfig,
    rotational_admittance_config_for_profile,
)
from teleop_runtime.real_teleop.position_admittance_shadow import (
    POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY,
    POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP,
    POSITION_ADMITTANCE_ROTATION_SCHEMA_VERSION,
    POSITION_ADMITTANCE_ROTATION_ACTIVE_QPOS_FAST_PATH_ALPHA,
    POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_FAST_PATH_ALPHA,
    POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_RAMP_SEC,
    POSITION_ADMITTANCE_TRANSLATION_RELEASE_QPOS_FAST_PATH_ALPHA,
    PositionAdmittanceActiveFault,
    PositionAdmittanceShadowTargetFilter,
    install_position_admittance_active,
)


class _Monitor:
    def __init__(self) -> None:
        self.source_time = 1.0
        self.values = {"left": [0.0] * 6, "right": [0.0] * 6}

    def snapshot(self) -> dict[str, object]:
        return {
            "fault": None,
            "last_reading_by_side": {
                side: {
                    "source": "tool",
                    "values": list(values),
                    "monotonic_time_sec": self.source_time,
                }
                for side, values in self.values.items()
            },
        }


def _translation_disabled() -> CartesianAdmittanceConfig:
    return CartesianAdmittanceConfig(
        enabled_axes=(False, False, False),
        control_frame="solver_world",
        response_model="profile_compliance",
        max_offset_m=(0.08, 0.08, 0.08),
        max_velocity_m_s=(0.06, 0.06, 0.06),
        max_return_velocity_m_s=(0.03, 0.03, 0.03),
        max_output_slew_m_s=(0.06, 0.06, 0.06),
        stale_after_sec=0.15,
        max_force_dt_sec=0.15,
    )


def _rotation_config() -> RotationalAdmittanceConfig:
    return RotationalAdmittanceConfig(
        enabled_axes=(True, True, True),
        control_frame="solver_world",
        stiffness_nm_rad=(10.0, 10.0, 10.0),
        torque_deadband_nm=(0.04, 0.04, 0.04),
        torque_filter_tau_sec=discrete_lowpass_alpha_to_tau(
            0.9,
            sample_hz=40.0,
        ),
        max_offset_rad=(0.4, 0.4, 0.4),
        max_velocity_rad_s=(0.35, 0.35, 0.35),
        max_return_velocity_rad_s=(0.1, 0.1, 0.1),
        max_output_slew_rad_s=(0.35, 0.35, 0.35),
        return_torque_deadband_nm=(0.04, 0.04, 0.04),
        stale_after_sec=0.15,
        max_torque_dt_sec=0.15,
    )


def _targets() -> dict[str, np.ndarray]:
    left = np.eye(4)
    right = np.eye(4)
    left[:3, 3] = (-0.4, -0.2, 0.2)
    right[:3, 3] = (-0.4, 0.2, 0.2)
    return {"left": left, "right": right}


def _observer(
    monitor: _Monitor,
    *,
    apply: bool,
    side_activity=None,
) -> PositionAdmittanceShadowTargetFilter:
    observer = PositionAdmittanceShadowTargetFilter(
        monitor=monitor,
        active_sides=("left", "right"),
        config=_translation_disabled(),
        rotational_config=_rotation_config(),
        previous_filter=None,
        side_activity=(
            side_activity
            if side_activity is not None
            else lambda side: (
                (True, "running") if side == "left" else (False, "fixed")
            )
        ),
        tool_rotation_from_solver_by_side={
            "left": np.eye(3),
            "right": np.eye(3),
        },
        apply_to_target=apply,
        apply_sides=("left",) if apply else (),
    )
    observer.reset_from_monitor(now_sec=1.0)
    return observer


def test_rotational_shadow_observes_but_preserves_target_identity() -> None:
    monitor = _Monitor()
    observer = _observer(monitor, apply=False)
    targets = _targets()
    originals = {side: pose.copy() for side, pose in targets.items()}

    monitor.source_time = 1.05
    observer(targets, object(), object(), now_sec=1.05)
    monitor.source_time = 1.10
    monitor.values["left"][5] = 1.0
    result = observer(targets, object(), object(), now_sec=1.10)

    assert result is None
    for side in targets:
        np.testing.assert_array_equal(targets[side], originals[side])
    snapshot = observer.snapshot(now_sec=1.10)
    assert snapshot["schema_version"] == POSITION_ADMITTANCE_ROTATION_SCHEMA_VERSION
    assert snapshot["rotation_authority"].startswith("shadow_or_device_free")
    left = snapshot["sides"]["left"]
    assert np.linalg.norm(left["rotation_output_rotvec_rad"]) > 0.0
    assert left["applied_rotation_rotvec_rad"] == pytest.approx((0.0, 0.0, 0.0))
    np.testing.assert_array_equal(
        left["applied_target_pose_solver_world"],
        originals["left"],
    )


def test_production_active_installer_admits_only_isolated_cautious_rotation() -> None:
    monitor = _Monitor()
    retargeter = SimpleNamespace(
        target_override_filter=None,
        _tool_tcp_from_solver={"left": np.eye(4), "right": np.eye(4)},
    )
    translation_disabled_cautious = replace(
        _translation_disabled(),
        max_offset_m=(0.002,) * 3,
        max_velocity_m_s=(0.005,) * 3,
        max_return_velocity_m_s=(0.005,) * 3,
        max_output_slew_m_s=(0.005,) * 3,
    )
    cautious = rotational_admittance_config_for_profile(
        ROTATIONAL_ADMITTANCE_PROFILE_CAUTIOUS_3DEG
    )
    observer = install_position_admittance_active(
        retargeter=retargeter,
        monitor=monitor,
        active_sides=("left", "right"),
        config=translation_disabled_cautious,
        rotational_config=cautious,
        side_activity=lambda side: (
            (True, "running") if side == "left" else (False, "fixed")
        ),
        apply_sides=("left",),
        now_sec=1.0,
    )
    snapshot = observer.snapshot(now_sec=1.0)
    assert snapshot["rotation_authority"] == (
        "production_active_cautious_3deg_single_side"
    )
    assert snapshot["active_qpos_fast_path_alpha"] == pytest.approx(
        POSITION_ADMITTANCE_ROTATION_ACTIVE_QPOS_FAST_PATH_ALPHA
    )
    targets = _targets()
    monitor.source_time = 1.01
    initial = observer(targets, object(), object(), now_sec=1.01)
    assert isinstance(initial, dict)
    observer.observe_admitted_poses(initial, now_sec=1.01)
    for tick in range(1, 4):
        now = 1.0 + 0.05 * tick
        monitor.source_time = now
        monitor.values["left"][5] = 1.0
        result = observer(targets, object(), object(), now_sec=now)
        assert isinstance(result, dict)
        observer.observe_admitted_poses(result, now_sec=now)
    contact_activity = observer.target_activity_override_by_side()["left"]
    assert contact_activity["modifier_phase"] == "contact"
    assert contact_activity["modifier_qpos_fast_path_alpha"] == pytest.approx(
        POSITION_ADMITTANCE_ROTATION_ACTIVE_QPOS_FAST_PATH_ALPHA
    )

    return_activities = []
    for tick in range(4, 30):
        now = 1.0 + 0.05 * tick
        monitor.source_time = now
        monitor.values["left"][5] = 0.0
        result = observer(targets, object(), object(), now_sec=now)
        assert isinstance(result, dict)
        observer.observe_admitted_poses(result, now_sec=now)
        activity = observer.target_activity_override_by_side().get("left")
        if activity is not None and activity.get("modifier_phase") == "return":
            return_activities.append(activity)
            if activity["modifier_qpos_fast_path_alpha"] == pytest.approx(
                POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_FAST_PATH_ALPHA
            ):
                break
    assert return_activities
    return_alphas = [
        float(activity["modifier_qpos_fast_path_alpha"])
        for activity in return_activities
    ]
    assert return_alphas[0] == pytest.approx(
        POSITION_ADMITTANCE_ROTATION_ACTIVE_QPOS_FAST_PATH_ALPHA
    )
    assert return_alphas == sorted(return_alphas)
    assert return_alphas[-1] == pytest.approx(
        POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_FAST_PATH_ALPHA
    )
    assert observer.snapshot(now_sec=now)[
        "rotation_release_qpos_ramp_sec"
    ] == pytest.approx(POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_RAMP_SEC)
    assert retargeter.target_override_filter is observer

    retargeter.target_override_filter = None
    with pytest.raises(ValueError, match="requires explicit combined"):
        install_position_admittance_active(
            retargeter=retargeter,
            monitor=monitor,
            active_sides=("left", "right"),
            config=translation_disabled_cautious,
            rotational_config=rotational_admittance_config_for_profile(
                ROTATIONAL_ADMITTANCE_PROFILE_BALANCED_5DEG
            ),
            side_activity=lambda _side: (True, "running"),
            apply_sides=("left",),
            now_sec=1.0,
        )
    assert retargeter.target_override_filter is None


def test_combined_active_uses_translation_release_fast_path_without_changing_rotation_phase() -> None:
    monitor = _Monitor()
    observer = PositionAdmittanceShadowTargetFilter(
        monitor=monitor,
        active_sides=("left", "right"),
        config=replace(
            _translation_disabled(),
            enabled_axes=(True, False, False),
        ),
        rotational_config=_rotation_config(),
        previous_filter=None,
        side_activity=lambda side: (
            (True, "running") if side == "left" else (False, "fixed")
        ),
        tool_rotation_from_solver_by_side={
            "left": np.eye(3),
            "right": np.eye(3),
        },
        apply_to_target=True,
        apply_sides=("left",),
        qpos_fast_path_alpha=0.80,
    )
    observer.reset_from_monitor(now_sec=1.0)
    targets = _targets()

    for tick in range(1, 7):
        now = 1.0 + 0.05 * tick
        monitor.source_time = now
        monitor.values["left"][0] = 4.0
        monitor.values["left"][5] = 1.0
        result = observer(targets, object(), object(), now_sec=now)
        assert isinstance(result, dict)
        observer.observe_admitted_poses(result, now_sec=now)

    release_activity = None
    release_side = None
    for tick in range(7, 25):
        now = 1.0 + 0.05 * tick
        monitor.source_time = now
        monitor.values["left"][0] = 0.0
        monitor.values["left"][5] = 1.0
        result = observer(targets, object(), object(), now_sec=now)
        assert isinstance(result, dict)
        observer.observe_admitted_poses(result, now_sec=now)
        release_side = observer.snapshot(now_sec=now)["sides"]["left"]
        activity = observer.target_activity_override_by_side().get("left")
        if (
            activity is not None
            and not release_side["contact_active"]
            and np.linalg.norm(release_side["applied_total_offset_m"]) > 1e-12
        ):
            release_activity = activity
            break

    assert release_activity is not None
    assert release_side is not None
    assert release_side["rotation_contact_active"] is True
    assert release_activity["modifier_phase"] == "contact"
    assert release_activity["modifier_qpos_fast_path_alpha"] == pytest.approx(
        POSITION_ADMITTANCE_TRANSLATION_RELEASE_QPOS_FAST_PATH_ALPHA
    )


@pytest.mark.parametrize(
    ("rotation_profile", "authority", "maximum_slew_deg_s"),
    (
        (
            ROTATIONAL_ADMITTANCE_PROFILE_BALANCED_5DEG,
            "production_active_normal_parity_balanced_5deg_"
            "translation_rotation_single_side",
            8.0,
        ),
        (
            ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG,
            "production_active_normal_parity_rotation_emphasis_5deg_"
            "translation_rotation_single_side",
            9.0,
        ),
    ),
)
def test_production_active_installer_admits_normal_parity_combined_5deg(
    rotation_profile: str,
    authority: str,
    maximum_slew_deg_s: float,
) -> None:
    monitor = _Monitor()
    retargeter = SimpleNamespace(
        target_override_filter=None,
        _tool_tcp_from_solver={"left": np.eye(4), "right": np.eye(4)},
    )
    translation_config = replace(
        _translation_disabled(),
        enabled_axes=(True, True, True),
    )
    rotation_config = rotational_admittance_config_for_profile(
        rotation_profile
    )
    observer = install_position_admittance_active(
        retargeter=retargeter,
        monitor=monitor,
        active_sides=("left", "right"),
        config=translation_config,
        rotational_config=rotation_config,
        side_activity=lambda side: (
            (True, "running") if side == "left" else (False, "fixed")
        ),
        apply_sides=("left",),
        active_envelope=POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY,
        combined_translation_rotation_active=True,
        now_sec=1.0,
    )
    startup_snapshot = observer.snapshot(now_sec=1.0)
    assert startup_snapshot["rotation_authority"] == authority
    assert startup_snapshot["active_endpoint_policy"] == "runtime_home_hold"
    assert startup_snapshot["active_nominal_alignment_range_guard_enabled"] is True
    assert startup_snapshot["active_nominal_translation_limit_m"] == pytest.approx(
        0.08
    )
    assert startup_snapshot["active_nominal_rotation_limit_deg"] == pytest.approx(
        3.0
    )

    targets = _targets()
    previous_translation = np.zeros(3)
    previous_rotation = np.zeros(3)
    max_translation = 0.0
    max_rotation_deg = 0.0
    for tick in range(1, 181):
        now = 1.0 + 0.05 * tick
        monitor.source_time = now
        monitor.values["left"] = (
            [25.0, 0.0, 0.0, 0.0, 0.0, 3.0]
            if tick <= 25
            else [0.0] * 6
        )
        result = observer(targets, object(), object(), now_sec=now)
        assert result is not None
        observer.observe_admitted_poses(result, now_sec=now)
        side = observer.snapshot(now_sec=now)["sides"]["left"]
        translation = np.asarray(side["applied_total_offset_m"], dtype=float)
        rotation = np.asarray(side["applied_rotation_rotvec_rad"], dtype=float)
        assert np.linalg.norm(translation - previous_translation) <= (
            0.06 * 0.05 + 1e-9
        )
        assert np.linalg.norm(rotation - previous_rotation) <= (
            np.deg2rad(maximum_slew_deg_s) * 0.05 + 1e-9
        )
        max_translation = max(max_translation, float(np.linalg.norm(translation)))
        max_rotation_deg = max(
            max_rotation_deg,
            float(np.degrees(np.linalg.norm(rotation))),
        )
        previous_translation = translation
        previous_rotation = rotation

    assert 0.04 < max_translation <= 0.08 + 1e-9
    assert 3.0 < max_rotation_deg <= 5.0 + 1e-9
    assert np.linalg.norm(previous_translation) <= 0.0001
    assert np.degrees(np.linalg.norm(previous_rotation)) <= 0.05


def test_formal_teleop_dual_side_authority_composes_both_bounded_modifiers() -> None:
    monitor = _Monitor()
    retargeter = SimpleNamespace(
        target_override_filter=None,
        _tool_tcp_from_solver={"left": np.eye(4), "right": np.eye(4)},
    )
    translation_config = replace(
        _translation_disabled(),
        enabled_axes=(True, True, True),
    )
    rotation_config = rotational_admittance_config_for_profile(
        ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG
    )

    with pytest.raises(ValueError, match="dual-side Active requires formal_teleop"):
        install_position_admittance_active(
            retargeter=retargeter,
            monitor=monitor,
            active_sides=("left", "right"),
            config=translation_config,
            rotational_config=rotation_config,
            side_activity=lambda _side: (True, "running"),
            apply_sides=("left", "right"),
            active_envelope=POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY,
            combined_translation_rotation_active=True,
            now_sec=1.0,
        )
    assert retargeter.target_override_filter is None

    observer = install_position_admittance_active(
        retargeter=retargeter,
        monitor=monitor,
        active_sides=("left", "right"),
        config=translation_config,
        rotational_config=rotation_config,
        side_activity=lambda _side: (True, "running_live_side"),
        apply_sides=("left", "right"),
        active_envelope=POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY,
        combined_translation_rotation_active=True,
        endpoint_policy=POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP,
        now_sec=1.0,
    )
    targets = _targets()
    monitor.source_time = 1.01
    initial = observer(targets, object(), object(), now_sec=1.01)
    assert initial is not None
    observer.observe_admitted_poses(initial, now_sec=1.01)

    previous_translation = {
        side: np.zeros(3, dtype=float) for side in ("left", "right")
    }
    previous_rotation = {
        side: np.zeros(3, dtype=float) for side in ("left", "right")
    }
    result = None
    for tick in range(1, 11):
        now = 1.01 + 0.05 * tick
        monitor.source_time = now
        monitor.values["left"] = [25.0, 0.0, 0.0, 0.0, 0.0, 3.0]
        monitor.values["right"] = [-20.0, 0.0, 0.0, 0.0, 0.0, -2.0]
        result = observer(targets, object(), object(), now_sec=now)
        assert result is not None
        observer.observe_admitted_poses(result, now_sec=now)
        snapshot = observer.snapshot(now_sec=now)
        for side in ("left", "right"):
            side_snapshot = snapshot["sides"][side]
            translation = np.asarray(
                side_snapshot["applied_total_offset_m"], dtype=float
            )
            rotation = np.asarray(
                side_snapshot["applied_rotation_rotvec_rad"], dtype=float
            )
            assert np.linalg.norm(
                translation - previous_translation[side]
            ) <= 0.06 * 0.05 + 1e-9
            assert np.linalg.norm(
                rotation - previous_rotation[side]
            ) <= np.deg2rad(9.0) * 0.05 + 1e-9
            assert np.linalg.norm(translation) <= 0.08 + 1e-9
            assert np.degrees(np.linalg.norm(rotation)) <= 5.0 + 1e-9
            previous_translation[side] = translation
            previous_rotation[side] = rotation

    assert result is not None
    final_snapshot = observer.snapshot(now_sec=1.51)
    assert final_snapshot["apply_sides"] == ["left", "right"]
    assert final_snapshot["rotation_authority"] == (
        "production_active_normal_parity_rotation_emphasis_5deg_"
        "translation_rotation_dual_side"
    )
    assert final_snapshot["active_nominal_alignment_range_guard_enabled"] is False
    assert previous_translation["left"][0] > 0.0
    assert previous_translation["right"][0] < 0.0
    assert previous_rotation["left"][2] > 0.0
    assert previous_rotation["right"][2] < 0.0


def test_formal_teleop_preserves_large_nominal_target_and_bounded_modifier() -> None:
    monitor = _Monitor()
    retargeter = SimpleNamespace(
        target_override_filter=None,
        _tool_tcp_from_solver={"left": np.eye(4), "right": np.eye(4)},
    )
    observer = install_position_admittance_active(
        retargeter=retargeter,
        monitor=monitor,
        active_sides=("left", "right"),
        config=replace(_translation_disabled(), enabled_axes=(True, True, True)),
        rotational_config=rotational_admittance_config_for_profile(
            ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG
        ),
        side_activity=lambda side: (
            (True, "running") if side == "left" else (False, "fixed")
        ),
        apply_sides=("left",),
        active_envelope=POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY,
        combined_translation_rotation_active=True,
        endpoint_policy=POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP,
        now_sec=1.0,
    )
    targets = _targets()
    nominal_left = targets["left"].copy()
    nominal_left[0, 3] += 0.15
    targets["left"] = nominal_left
    previous_translation = np.zeros(3)
    result = None
    for tick in range(1, 6):
        now = 1.0 + 0.05 * tick
        monitor.source_time = now
        monitor.values["left"] = [25.0, 0.0, 0.0, 0.0, 0.0, 3.0]
        result = observer(targets, object(), object(), now_sec=now)
        assert result is not None
        observer.observe_admitted_poses(result, now_sec=now)
        applied_translation = result["left"][:3, 3] - nominal_left[:3, 3]
        assert np.linalg.norm(
            applied_translation - previous_translation
        ) <= 0.06 * 0.05 + 1e-9
        previous_translation = applied_translation

    assert result is not None
    applied_translation = result["left"][:3, 3] - nominal_left[:3, 3]
    assert np.linalg.norm(applied_translation) > 0.0
    assert np.linalg.norm(applied_translation) <= 0.08 + 1e-9
    snapshot = observer.snapshot(now_sec=1.25)
    assert snapshot["active_endpoint_policy"] == "formal_teleop"
    assert snapshot["active_nominal_alignment_range_guard_enabled"] is False
    assert snapshot["active_nominal_translation_limit_m"] is None
    assert snapshot["active_nominal_rotation_limit_deg"] is None
    assert np.linalg.norm(
        result["left"][:3, 3] - _targets()["left"][:3, 3]
    ) > 0.08

    # Reproduce ordinary formal-teleop tracking lag while the bounded
    # modifier is nonzero.  The previous admitted qref is fresh, but the next
    # nominal VIVE target is deliberately 15 cm / 10 degrees away.  These
    # ranges belong to the unchanged formal IK/safety chain, not to the
    # admittance modifier.
    lagged_targets = {side: pose.copy() for side, pose in targets.items()}
    lagged_targets["left"][0, 3] += 0.15
    angle = np.deg2rad(10.0)
    lagged_targets["left"][:3, :3] = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    monitor.source_time = 1.30
    lagged_result = observer(
        lagged_targets,
        object(),
        object(),
        now_sec=1.30,
    )
    assert lagged_result is not None
    lagged_snapshot = observer.snapshot(now_sec=1.30)
    assert np.linalg.norm(
        lagged_result["left"][:3, 3] - lagged_targets["left"][:3, 3]
    ) <= 0.08 + 1e-9
    assert np.degrees(
        np.linalg.norm(
            lagged_snapshot["sides"]["left"][
                "applied_rotation_rotvec_rad"
            ]
        )
    ) <= 5.0 + 1e-9

    # Removing the temporary range gates must not remove admitted-qref
    # freshness: a stale command receipt still fails closed before overlay.
    monitor.source_time = 1.40
    with pytest.raises(
        PositionAdmittanceActiveFault,
        match="admitted-qref baseline is stale",
    ):
        observer(
            lagged_targets,
            object(),
            object(),
            now_sec=1.46,
        )


def test_production_active_installer_requires_explicit_combined_authority() -> None:
    monitor = _Monitor()
    retargeter = SimpleNamespace(
        target_override_filter=None,
        _tool_tcp_from_solver={"left": np.eye(4), "right": np.eye(4)},
    )
    translation_cautious = replace(
        _translation_disabled(),
        enabled_axes=(True, True, True),
        max_offset_m=(0.002,) * 3,
        max_velocity_m_s=(0.005,) * 3,
        max_return_velocity_m_s=(0.005,) * 3,
        max_output_slew_m_s=(0.005,) * 3,
    )
    rotation_cautious = rotational_admittance_config_for_profile(
        ROTATIONAL_ADMITTANCE_PROFILE_CAUTIOUS_3DEG
    )

    with pytest.raises(ValueError, match="translation axes disabled"):
        install_position_admittance_active(
            retargeter=retargeter,
            monitor=monitor,
            active_sides=("left", "right"),
            config=translation_cautious,
            rotational_config=rotation_cautious,
            side_activity=lambda side: (
                (True, "running") if side == "left" else (False, "fixed")
            ),
            apply_sides=("left",),
            now_sec=1.0,
        )
    assert retargeter.target_override_filter is None

    observer = install_position_admittance_active(
        retargeter=retargeter,
        monitor=monitor,
        active_sides=("left", "right"),
        config=translation_cautious,
        rotational_config=rotation_cautious,
        side_activity=lambda side: (
            (True, "running") if side == "left" else (False, "fixed")
        ),
        apply_sides=("left",),
        combined_translation_rotation_active=True,
        now_sec=1.0,
    )
    assert observer.snapshot(now_sec=1.0)["rotation_authority"] == (
        "production_active_cautious_translation_rotation_single_side"
    )

    targets = _targets()
    monitor.source_time = 1.01
    initial = observer(targets, object(), object(), now_sec=1.01)
    assert initial is not None
    observer.observe_admitted_poses(initial, now_sec=1.01)

    monitor.source_time = 1.06
    monitor.values["left"][0] = 2.0
    monitor.values["left"][5] = 1.0
    combined = observer(targets, object(), object(), now_sec=1.06)
    assert combined is not None
    assert np.linalg.norm(
        combined["left"][:3, 3] - targets["left"][:3, 3]
    ) > 0.0
    assert not np.allclose(
        combined["left"][:3, :3],
        targets["left"][:3, :3],
    )
    np.testing.assert_array_equal(combined["right"], targets["right"])
    snapshot = observer.snapshot(now_sec=1.06)["sides"]["left"]
    assert np.linalg.norm(snapshot["applied_total_offset_m"]) <= 0.002 + 1e-12
    assert np.degrees(
        np.linalg.norm(snapshot["applied_rotation_rotvec_rad"])
    ) <= 3.0 + 1e-9


def test_combined_active_sudden_wrench_release_and_hold_remain_bounded() -> None:
    monitor = _Monitor()
    running = {"value": True}
    retargeter = SimpleNamespace(
        target_override_filter=None,
        _tool_tcp_from_solver={"left": np.eye(4), "right": np.eye(4)},
    )
    translation_config = replace(
        _translation_disabled(),
        enabled_axes=(True, True, True),
        max_offset_m=(0.002,) * 3,
        max_velocity_m_s=(0.005,) * 3,
        max_return_velocity_m_s=(0.005,) * 3,
        max_output_slew_m_s=(0.005,) * 3,
    )
    rotation_config = rotational_admittance_config_for_profile(
        ROTATIONAL_ADMITTANCE_PROFILE_CAUTIOUS_3DEG
    )
    observer = install_position_admittance_active(
        retargeter=retargeter,
        monitor=monitor,
        active_sides=("left", "right"),
        config=translation_config,
        rotational_config=rotation_config,
        side_activity=lambda side: (
            (running["value"], "running" if running["value"] else "operator_hold")
            if side == "left"
            else (False, "fixed")
        ),
        apply_sides=("left",),
        combined_translation_rotation_active=True,
        now_sec=1.0,
    )
    targets = _targets()
    previous_translation = np.zeros(3)
    previous_rotation = np.zeros(3)
    max_translation = 0.0
    max_rotation = 0.0

    for tick in range(1, 101):
        now = 1.0 + 0.05 * tick
        monitor.source_time = now
        monitor.values["left"] = (
            [2.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            if tick <= 10
            else [0.0] * 6
        )
        result = observer(targets, object(), object(), now_sec=now)
        assert result is not None
        observer.observe_admitted_poses(result, now_sec=now)
        side = observer.snapshot(now_sec=now)["sides"]["left"]
        translation = np.asarray(side["applied_total_offset_m"], dtype=float)
        rotation = np.asarray(side["applied_rotation_rotvec_rad"], dtype=float)
        assert np.linalg.norm(translation - previous_translation) <= (
            0.005 * 0.05 + 1e-9
        )
        assert np.linalg.norm(rotation - previous_rotation) <= (
            max(rotation_config.max_output_slew_rad_s) * 0.05 + 1e-9
        )
        max_translation = max(max_translation, float(np.linalg.norm(translation)))
        max_rotation = max(max_rotation, float(np.linalg.norm(rotation)))
        previous_translation = translation
        previous_rotation = rotation

    assert 0.0 < max_translation <= 0.002 + 1e-9
    assert 0.0 < np.degrees(max_rotation) <= 3.0 + 1e-9
    assert np.linalg.norm(previous_translation) <= 0.0001
    assert np.degrees(np.linalg.norm(previous_rotation)) <= 0.05

    # Hold removes both overlays and re-entry starts from zero rather than
    # replaying either pre-Hold state.
    running["value"] = False
    monitor.source_time = 6.05
    held = observer(targets, object(), object(), now_sec=6.05)
    assert held is not None
    np.testing.assert_allclose(held["left"], targets["left"], atol=1e-12)
    running["value"] = True
    monitor.source_time = 6.10
    resumed = observer(targets, object(), object(), now_sec=6.10)
    assert resumed is not None
    resumed_side = observer.snapshot(now_sec=6.10)["sides"]["left"]
    assert resumed_side["applied_total_offset_m"] == pytest.approx((0.0, 0.0, 0.0))
    assert resumed_side["applied_rotation_rotvec_rad"] == pytest.approx(
        (0.0, 0.0, 0.0)
    )


def test_device_free_active_counterfactual_composes_bounded_so3_rotation() -> None:
    monitor = _Monitor()
    observer = _observer(monitor, apply=True)
    targets = _targets()

    monitor.source_time = 1.05
    first = observer(targets, object(), object(), now_sec=1.05)
    assert isinstance(first, dict)
    observer.observe_admitted_poses(first, now_sec=1.05)

    monitor.source_time = 1.10
    monitor.values["left"][5] = 1.0
    result = observer(targets, object(), object(), now_sec=1.10)
    assert isinstance(result, dict)
    np.testing.assert_allclose(result["left"][:3, 3], targets["left"][:3, 3])
    assert not np.allclose(result["left"][:3, :3], targets["left"][:3, :3])
    np.testing.assert_allclose(
        result["left"][:3, :3].T @ result["left"][:3, :3],
        np.eye(3),
        atol=1e-12,
    )
    assert np.linalg.det(result["left"][:3, :3]) == pytest.approx(1.0)
    np.testing.assert_array_equal(result["right"], targets["right"])

    activity = observer.target_activity_override_by_side()
    assert activity["left"]["modifier_offset_norm_m"] == 0.0
    assert activity["left"]["modifier_rotation_offset_norm_rad"] > 0.0


def test_hold_freezes_and_reentry_resets_rotation_state() -> None:
    monitor = _Monitor()
    running = {"value": True}
    observer = _observer(
        monitor,
        apply=False,
        side_activity=lambda side: (
            (running["value"], "running" if running["value"] else "operator_hold")
            if side == "left"
            else (False, "fixed")
        ),
    )
    targets = _targets()

    monitor.source_time = 1.05
    observer(targets, object(), object(), now_sec=1.05)
    monitor.source_time = 1.10
    monitor.values["left"][3] = 1.0
    observer(targets, object(), object(), now_sec=1.10)
    assert np.linalg.norm(
        observer.snapshot(now_sec=1.10)["sides"]["left"][
            "rotation_output_rotvec_rad"
        ]
    ) > 0.0

    running["value"] = False
    monitor.source_time = 1.15
    observer(targets, object(), object(), now_sec=1.15)
    assert observer.snapshot(now_sec=1.15)["sides"]["left"]["rotation_frozen"]

    running["value"] = True
    monitor.values["left"][3] = 0.0
    monitor.source_time = 1.20
    observer(targets, object(), object(), now_sec=1.20)
    left = observer.snapshot(now_sec=1.20)["sides"]["left"]
    assert left["rotation_output_rotvec_rad"] == pytest.approx((0.0, 0.0, 0.0))
    assert left["rotation_frozen"] is False


def test_nominal_alignment_removes_previous_bounded_rotation_modifier() -> None:
    monitor = _Monitor()
    observer = _observer(monitor, apply=True)
    targets = _targets()

    admitted = observer(targets, object(), object(), now_sec=1.0)
    assert admitted is not None
    observer.observe_admitted_poses(admitted, now_sec=1.0)
    for index in range(1, 8):
        monitor.source_time = 1.0 + 0.05 * index
        monitor.values["left"][5] = 4.0
        admitted = observer(
            targets,
            object(),
            object(),
            now_sec=monitor.source_time,
        )
        assert admitted is not None
        observer.observe_admitted_poses(admitted, now_sec=monitor.source_time)

    rotation = observer.snapshot(now_sec=monitor.source_time)["sides"]["left"]
    assert np.degrees(
        np.linalg.norm(rotation["applied_rotation_rotvec_rad"])
    ) > 3.0


def test_nominal_alignment_still_rejects_distant_raw_rotation_target() -> None:
    monitor = _Monitor()
    observer = _observer(monitor, apply=True)
    targets = _targets()

    admitted = observer(targets, object(), object(), now_sec=1.0)
    assert admitted is not None
    observer.observe_admitted_poses(admitted, now_sec=1.0)
    monitor.source_time = 1.05
    monitor.values["left"][5] = 1.0
    moved_targets = {side: pose.copy() for side, pose in targets.items()}
    angle = np.deg2rad(10.0)
    moved_targets["left"][:3, :3] = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    with pytest.raises(PositionAdmittanceActiveFault, match="nominal target"):
        observer(
            moved_targets,
            object(),
            object(),
            now_sec=monitor.source_time,
        )


@pytest.mark.parametrize(
    ("translation_residual_m", "accepted"),
    ((0.0049, True), (0.0051, False)),
)
def test_rotational_active_uses_explicit_five_mm_nominal_translation_gate(
    translation_residual_m: float,
    accepted: bool,
) -> None:
    monitor = _Monitor()
    observer = _observer(monitor, apply=True)
    targets = _targets()
    initial = observer(targets, object(), object(), now_sec=1.0)
    assert initial is not None
    admitted = {side: pose.copy() for side, pose in targets.items()}
    admitted["left"][0, 3] += translation_residual_m
    observer.observe_admitted_poses(admitted, now_sec=1.0)
    monitor.source_time = 1.05
    monitor.values["left"][5] = 1.0

    if accepted:
        assert observer(targets, object(), object(), now_sec=1.05) is not None
    else:
        with pytest.raises(
            PositionAdmittanceActiveFault,
            match="translation residual",
        ):
            observer(targets, object(), object(), now_sec=1.05)
