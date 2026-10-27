"""Reused frozen-relation and release tests, independent of active motion."""

from types import SimpleNamespace

import numpy as np
import pytest

from sim.mujoco.deploy.grasped_object import (
    MujocoGraspedObjectObserver, ObjectMotionPreconditionError, ObjectRelationInvalidated,
)

pytestmark = pytest.mark.unit


def _observer(*, track_close_contacts=False, contact_verified_close=False):
    plant = SimpleNamespace(time=0., z=0., support=True, other=False, slip=0.,
                            sides=(True, True), command=(1, 1.0),
                            measured=.2, settled=True)
    plant.data = object()
    plant.scene = SimpleNamespace(data=plant.data)
    observer = MujocoGraspedObjectObserver(
        rig=SimpleNamespace(capture_scene=lambda callback: callback(plant.scene)), object_body_name="cube",
        sensor_geom_names=("left_sensor", "right_sensor"), support_body_names=("table",),
        gripper_name="right", command_state=lambda: plant.command,
        world_tcp_from_scene=lambda _: np.eye(4), track_close_contacts=track_close_contacts,
        contact_verified_close=contact_verified_close,
        gripper_settled=(lambda: plant.settled) if contact_verified_close else None)

    def capture(_, source_data=None):
        obj, tcp = np.eye(4), np.eye(4)
        obj[:3, 3] = (plant.slip, 0, .05 + plant.z)
        tcp[2, 3] = .1 + plant.z
        return plant.time, obj, tcp, plant.sides, plant.support, plant.other, plant.measured, tuple(obj[:3, 3])

    observer._capture = capture
    return plant, observer


@pytest.mark.parametrize("measured", [.045, .2])
def test_contact_verified_close_waits_for_settled_hand_before_freezing_relation(measured):
    plant, observer = _observer(track_close_contacts=True, contact_verified_close=True)
    plant.measured, plant.settled = measured, False
    _physics_observation(plant, observer)
    assert observer.latest is None
    plant.time, plant.sides = .005, (False, True)
    _physics_observation(plant, observer)
    assert observer._anchor is None and observer._failure is None
    plant.sides = (True, True)
    plant.time, plant.settled = .01, True
    _physics_observation(plant, observer)
    assert observer.latest is not None
    assert observer.latest.status == "contact_secured"
    frozen = observer.latest.tool_from_object.copy()
    plant.time, plant.slip = .02, .004
    _physics_observation(plant, observer)
    assert observer._failure is not None
    np.testing.assert_array_equal(observer.latest.tool_from_object, frozen)


@pytest.mark.parametrize("measured,sides", [
    (.045, (True, True)), (.05, (True, True)), (.2, (True, False)), (.2, (True, True)),
])
def test_target_bound_explicit_lift_cannot_freeze_a_moving_close(measured, sides):
    plant, observer = _observer(track_close_contacts=True, contact_verified_close=True)
    plant.measured, plant.settled, plant.sides = measured, False, sides
    with pytest.raises(ObjectMotionPreconditionError, match="must settle"):
        observer.observe(phase="lift", establish=True)
    assert observer.latest is None and observer._anchor is None and observer._failure is None
    plant.settled, plant.sides = True, (True, True)
    assert observer.observe(phase="lift", establish=True).status == "contact_secured"


def test_established_target_bound_relation_still_latches_contact_loss_when_hand_unsettles():
    plant, observer = _observer(track_close_contacts=True, contact_verified_close=True)
    observer.observe(phase="lift", establish=True)
    plant.time, plant.settled, plant.sides = .01, False, (True, False)
    with pytest.raises(ObjectRelationInvalidated, match="Both target sensor contacts"):
        observer.observe(phase="lift")
    assert observer._failure is not None
    plant.settled, plant.sides = True, (True, True)
    with pytest.raises(ObjectRelationInvalidated, match="invalidated"):
        observer.observe(phase="lift")


def test_unbound_close_retains_legacy_aperture_establishment():
    plant, observer = _observer(track_close_contacts=True, contact_verified_close=False)
    plant.measured, plant.settled = .2, False
    _physics_observation(plant, observer)
    assert observer.latest is not None and observer.latest.status == "contact_secured"


def test_selected_object_support_does_not_count_a_table_fall_as_release():
    plant, observer = _observer(track_close_contacts=True)
    observer.support_body_names = ("table", "box")
    plant.touching_support = "table"
    original_capture = observer._capture

    def capture(scene, source_data=None, *, support_body_name=None):
        stamp, obj, tcp, sides, _support, _other, measured, com = original_capture(
            scene, source_data)
        selected = support_body_name or observer.active_support_body_name
        return (stamp, obj, tcp, sides,
                bool(plant.support and plant.touching_support == selected),
                bool(plant.support and plant.touching_support != selected),
                measured, com)

    observer._capture = capture
    _lift(plant, observer)
    frozen = observer.latest.tool_from_object.copy()
    with pytest.raises(ObjectMotionPreconditionError, match="admitted object identity"):
        observer.observe(phase="place", place_support_body_name="foreign")
    assert observer._failure is None
    assert observer.active_support_body_name == "table"
    observer.observe(phase="place", establish=True, place_support_body_name="box")
    assert observer.active_support_body_name == "table"
    observer.observe(phase="place", source_data=plant.scene.data,
                     place_support_body_name="box")
    assert observer.active_support_body_name == "box"
    plant.support, plant.touching_support = True, "box"
    plant.time += .01
    _physics_observation(plant, observer)
    assert observer.latest.support_contact and not observer.latest.other_contact
    np.testing.assert_array_equal(observer.latest.tool_from_object, frozen)

    plant.command, plant.sides, plant.touching_support = (2, 0.), (False, False), "table"
    for _ in range(62):
        plant.time += .01
        _physics_observation(plant, observer)
    release = observer.diagnostic_snapshot()["release_observation"]
    assert release is not None and not release["stable_supported_release"]
    assert observer.active_support_body_name == "box"
    plant.command = (3, 1.)
    _physics_observation(plant, observer)
    assert observer.active_support_body_name == "table"


def _lift(plant, observer):
    first = observer.observe(phase="lift", establish=True)
    assert first.status == "contact_secured"
    plant.z, plant.support = .04, False
    for index in range(1, 54):
        plant.time = index * .01
        last = observer.observe(phase="lift")
    assert last.status == "held_verified"
    return last


def test_opt_in_close_tracking_requires_measured_lift_before_release_can_start():
    plant, observer = _observer(track_close_contacts=True)
    plant.command = (0, None)
    _physics_observation(plant, observer)
    assert observer.latest is None
    plant.command = (1, 1.)
    _physics_observation(plant, observer)
    anchor = observer.latest.tool_from_object.copy()
    for _ in range(70):
        plant.time += .01
        _physics_observation(plant, observer)
    assert observer.latest.status == "contact_secured"
    np.testing.assert_array_equal(observer.latest.tool_from_object, anchor)
    plant.command = (2, 0.)
    _physics_observation(plant, observer)
    assert observer.diagnostic_snapshot()["release_observation"] is None


@pytest.mark.parametrize("condition", ["one_contact", "unsupported", "other_contact", "not_closed"])
def test_opt_in_tracking_does_not_anchor_unqualified_contact(condition):
    plant, observer = _observer(track_close_contacts=True)
    if condition == "one_contact":
        plant.sides = (True, False)
    elif condition == "unsupported":
        plant.support = False
    elif condition == "other_contact":
        plant.other = True
    else:
        plant.command = (1, .5)
    _physics_observation(plant, observer)
    assert observer.latest is None and observer._anchor is None


def test_opt_in_tracking_preserves_frozen_relation_and_measured_release_policy():
    plant, observer = _observer(track_close_contacts=True)
    _physics_observation(plant, observer)
    plant.z, plant.support = .02, False
    for _ in range(62):
        plant.time += .01
        _physics_observation(plant, observer)
    assert observer.latest.status == "contact_secured"
    plant.z = .04
    for _ in range(62):
        plant.time += .01
        _physics_observation(plant, observer)
    assert observer.latest.status == "held_verified"
    plant.command, plant.support, plant.sides = (2, 0.), True, (False, False)
    for _ in range(62):
        plant.time += .01
        _physics_observation(plant, observer)
    release = observer.diagnostic_snapshot()["release_observation"]
    assert release["stable_supported_release"]
    assert release["window_reason"] == "complete"
    assert release["max_observed_sample_gap_s"] == pytest.approx(.01)
    from test_agent_gateway_object_state import _state, _validate
    measured = {**_state(), "measured_at": plant.time, "cutoff_at": plant.time,
                "release_observation": release}
    _validate(measured, release_supported=True)
    plant.scene.data = SimpleNamespace(time=plant.time)
    before = observer.release_snapshot()
    observer._capture = lambda *_: pytest.fail("History reads must not resample the scene")
    assert observer.release_snapshot() == before


def test_lost_grasp_followed_by_open_cannot_arm_release_from_historical_verification():
    plant, observer = _observer(track_close_contacts=True)
    _lift(plant, observer)
    plant.slip = .004
    plant.time += .01
    _physics_observation(plant, observer)
    assert observer._failure is not None
    plant.command, plant.slip, plant.support, plant.sides = (2, 0.), 0., True, (False, False)
    for _ in range(62):
        plant.time += .01
        _physics_observation(plant, observer)
    assert observer.diagnostic_snapshot()["release_observation"] is None


def test_actual_following_required_and_initial_anchor_not_refreshed():
    plant, observer = _observer()
    with pytest.raises(ObjectMotionPreconditionError):
        observer.observe(phase="carry", establish=True)
    last = _lift(plant, observer)
    assert last.relative_translation_error_m == pytest.approx(0.)
    anchor = last.tool_from_object.copy()
    for slip in (.001, .002):
        plant.time += .01
        plant.slip = slip
        cut = observer.observe(phase="carry")
        np.testing.assert_array_equal(cut.tool_from_object, anchor)
    plant.time += .01
    plant.slip = .0031
    with pytest.raises(ValueError, match="slipped"):
        observer.observe(phase="carry")
    plant.slip = 0.
    with pytest.raises(ValueError, match="invalidated"):
        observer.observe(phase="carry")


def test_object_observer_reuses_last_bilateral_cut_for_one_physics_tick_only():
    plant, observer = _observer()
    timestep = .0005
    observer._model = SimpleNamespace(opt=SimpleNamespace(timestep=timestep))
    first = observer.observe(phase="lift", establish=True)
    plant.time = timestep
    plant.sides = (False, True)
    for _ in range(3):
        assert observer.observe(phase="lift", source_data=object()) is first
    assert observer.latest is first

    plant.time = 2 * timestep
    plant.sides = (True, True)
    recovered = observer.observe(phase="lift", source_data=object())
    assert recovered.sim_time_s == pytest.approx(2 * timestep)
    assert recovered.sensor_contacts == (True, True)
    assert recovered.tool_from_object is first.tool_from_object


def test_object_observer_rejects_persistent_or_out_of_relation_contact_dropout():
    plant, observer = _observer()
    timestep = .0005
    observer._model = SimpleNamespace(opt=SimpleNamespace(timestep=timestep))
    first = observer.observe(phase="lift", establish=True)
    plant.sides = (False, True)
    plant.time = timestep
    assert observer.observe(phase="lift", source_data=object()) is first
    plant.time = 2 * timestep
    with pytest.raises(ValueError, match="one physics-step recovery"):
        observer.observe(phase="lift", source_data=object())
    plant.sides = (True, True)
    plant.time = 3 * timestep
    with pytest.raises(ValueError, match="invalidated"):
        observer.observe(phase="lift", source_data=object())

    plant, observer = _observer()
    observer._model = SimpleNamespace(opt=SimpleNamespace(timestep=timestep))
    observer.observe(phase="lift", establish=True)
    plant.time = timestep
    plant.sides = (False, True)
    plant.slip = .0031
    with pytest.raises(ValueError, match="one physics-step recovery"):
        observer.observe(phase="lift", source_data=object())
    plant.sides = (True, True)
    plant.slip = 0.
    plant.time = 2 * timestep
    with pytest.raises(ValueError, match="invalidated"):
        observer.observe(phase="lift", source_data=object())


def test_new_gripper_command_revokes_old_object_relation():
    plant, observer = _observer()
    _lift(plant, observer)
    plant.command = (2, 0.)
    with pytest.raises(ValueError, match="closed command"):
        observer.observe(phase="carry", establish=True)
    assert observer.latest is None
    plant.command = (3, 1.)
    with pytest.raises(ObjectMotionPreconditionError, match="lift relation"):
        observer.observe(phase="place", establish=True)
    assert observer.observe(phase="lift", establish=True).status == "contact_secured"


def test_place_support_is_explicit_and_foreign_contact_invalidates_relation():
    plant, observer = _observer()
    _lift(plant, observer)
    plant.time += .01
    plant.support = True
    assert observer.observe(phase="place").support_contact
    plant.time += .01
    plant.other = True
    with pytest.raises(ValueError, match="outside"):
        observer.observe(phase="place")


def test_unobserved_simulation_gap_revokes_verified_window_without_reanchoring():
    plant, observer = _observer()
    first = _lift(plant, observer)
    plant.time += .1
    with pytest.raises(ObjectMotionPreconditionError, match="requires observed"):
        observer.observe(phase="carry", establish=True)
    cut = observer.observe(phase="lift", establish=True)
    assert cut.status == "contact_secured"
    np.testing.assert_array_equal(cut.tool_from_object, first.tool_from_object)
    for _ in range(53):
        plant.time += .01
        cut = observer.observe(phase="lift")
    assert cut.status == "held_verified"


def test_preparation_gap_cannot_bridge_two_sample_follow_window():
    plant, observer = _observer()
    first = observer.observe(phase="lift", establish=True)
    plant.z, plant.support = .04, False
    for stamp in (.01, .02):
        plant.time = stamp
        assert observer.observe(phase="lift").status == "contact_secured"
    plant.time = .62
    cut = observer.observe(phase="lift", establish=True)
    assert cut.status == "contact_secured"
    np.testing.assert_array_equal(cut.tool_from_object, first.tool_from_object)
    assert observer.observe(phase="lift", establish=True).status == "contact_secured"


def test_simulation_idle_does_not_revoke_a_verified_relation():
    plant, observer = _observer()
    first = _lift(plant, observer)
    cut = observer.observe(phase="carry", establish=True)
    assert cut.status == "held_verified" and cut.sim_time_s == first.sim_time_s


def _physics_observation(plant, observer):
    observer.observe_physics_step(SimpleNamespace(data=SimpleNamespace(time=plant.time)))


def test_idle_physics_ticks_preserve_fresh_following_without_reanchoring():
    plant, observer = _observer()
    first = _lift(plant, observer)
    anchor = first.tool_from_object.copy()
    for _ in range(100):
        plant.time += .01
        plant.slip += .00001
        _physics_observation(plant, observer)
    cut = observer.observe(phase="carry", establish=True)
    assert cut.status == "held_verified"
    assert cut.sim_time_s == plant.time
    assert cut.relative_translation_error_m == pytest.approx(.001)
    np.testing.assert_array_equal(cut.tool_from_object, anchor)


def test_passive_ticks_cannot_establish_a_grasp_and_new_command_revokes_it():
    plant, observer = _observer()
    for _ in range(60):
        plant.time += .01
        _physics_observation(plant, observer)
    assert observer.latest is None and observer._anchor is None
    plant.time = 0.
    _lift(plant, observer)
    for command in ((2, 0.), (3, 1.)):
        plant.command = command
        plant.time += .01
        _physics_observation(plant, observer)
        assert observer.latest is None and observer._anchor is None
        assert observer._tracking_phase is None
    with pytest.raises(ObjectMotionPreconditionError, match="lift relation"):
        observer.observe(phase="carry", establish=True)


@pytest.mark.parametrize("fault", ["slip", "gap", "contact"])
def test_idle_relation_failure_stays_latched_without_faulting_empty_hand_motion(fault):
    plant, observer = _observer()
    first = _lift(plant, observer)
    plant.time += .1 if fault == "gap" else .01
    plant.slip = .004 if fault == "slip" else 0.
    plant.sides = (True, False) if fault == "contact" else (True, True)
    _physics_observation(plant, observer)
    assert observer._failure is not None
    plant.slip, plant.sides = 0., (True, True)
    for _ in range(60):
        plant.time += .01
        _physics_observation(plant, observer)
    assert observer.latest is first
    with pytest.raises(ObjectRelationInvalidated, match="invalidated"):
        observer.observe(phase="carry", establish=True)


def test_idle_tick_preserves_the_single_native_contact_dropout_recovery():
    plant, observer = _observer()
    observer._model = SimpleNamespace(opt=SimpleNamespace(timestep=.0005))
    first = _lift(plant, observer)
    plant.time += .0005
    plant.sides = (True, False)
    _physics_observation(plant, observer)
    assert observer.latest is first and observer._failure is None
    plant.time += .0005
    plant.sides = (True, True)
    _physics_observation(plant, observer)
    assert observer.latest.status == "held_verified"
    assert observer.latest.sim_time_s == plant.time


def test_active_post_step_cut_is_not_copied_or_counted_again():
    plant, observer = _observer()
    _lift(plant, observer)
    observer.observe(phase="lift", source_data=plant.data)
    count, first = observer._window_count, observer.latest

    def unexpected_capture(*_args):
        raise AssertionError("The active post-step cut was already measured.")

    observer._capture = unexpected_capture
    _physics_observation(plant, observer)
    assert observer.latest is first and observer._window_count == count


def test_active_guard_dropout_is_not_measured_again_by_the_passive_callback():
    plant, observer = _observer()
    observer._model = SimpleNamespace(opt=SimpleNamespace(timestep=.0005))
    first = _lift(plant, observer)
    plant.time += .0005
    plant.sides = (True, False)
    assert observer.observe(phase="carry", source_data=plant.data) is first
    count = observer._window_count

    def unexpected_capture(*_args):
        raise AssertionError("The guarded dropout cut was already measured.")

    observer._capture = unexpected_capture
    _physics_observation(plant, observer)
    assert observer.latest is first and observer._window_count == count
    assert observer._failure is None


def test_place_preflight_does_not_change_idle_carry_support_policy():
    plant, observer = _observer()
    _lift(plant, observer)
    observer.observe(phase="carry", source_data=plant.data)
    observer.observe(phase="place", establish=True)
    observer.observe(phase="place", source_data=object())
    assert observer._tracking_phase == "carry"
    plant.support = True
    plant.time += .01
    _physics_observation(plant, observer)
    with pytest.raises(ObjectRelationInvalidated, match="outside"):
        observer.observe(phase="place", establish=True)


def test_active_place_guard_may_change_idle_support_policy():
    plant, observer = _observer()
    _lift(plant, observer)
    observer.observe(phase="carry", source_data=plant.data)
    observer.observe(phase="place", source_data=plant.data)
    plant.support = True
    plant.time += .01
    _physics_observation(plant, observer)
    assert observer._tracking_phase == "place" and observer._failure is None
    assert observer.latest.support_contact


def test_unexpected_passive_capture_error_is_not_swallowed():
    plant, observer = _observer()
    _lift(plant, observer)
    plant.time += .01

    def broken_capture(*_args):
        raise ValueError("native model changed")

    observer._capture = broken_capture
    with pytest.raises(ValueError, match="native model changed"):
        _physics_observation(plant, observer)
    assert observer._failure == "native model changed"


def test_capture_and_relation_updates_share_the_physics_lock():
    plant, observer = _observer()
    inside = False
    original_capture = observer._capture

    def locked(callback):
        nonlocal inside
        assert not inside
        inside = True
        try:
            return callback(None)
        finally:
            inside = False

    def command():
        assert inside
        return plant.command

    def capture(*args):
        assert inside
        return original_capture(*args)

    observer.rig.capture_scene = locked
    observer.command_state, observer._capture = command, capture
    assert observer.observe(phase="lift", establish=True).status == "contact_secured"


def test_diagnostic_read_does_not_renew_stale_holding_evidence_or_reanchor():
    plant, observer = _observer()
    first = _lift(plant, observer)
    assert observer.diagnostic_snapshot()["holding_verified"] is True
    plant.time += .1
    plant.slip = .001
    snapshot = observer.diagnostic_snapshot()
    assert snapshot["sim_time_s"] == plant.time
    assert snapshot["historical_status"] == "held_verified"
    assert snapshot["current_relation_within_bounds"] is True
    assert snapshot["evidence_age_sec"] == pytest.approx(.1)
    assert snapshot["evidence_fresh"] is False
    assert snapshot["holding_verified"] is False
    assert observer.latest is first
    np.testing.assert_array_equal(observer._anchor, first.tool_from_object)
    with pytest.raises(ObjectMotionPreconditionError, match="requires observed"):
        observer.observe(phase="carry", establish=True)


def test_diagnostic_read_reports_current_slip_without_repairing_failed_relation():
    plant, observer = _observer()
    first = _lift(plant, observer)
    plant.time += .01
    plant.slip = .004
    snapshot = observer.diagnostic_snapshot()
    assert snapshot["relative_translation_error_m"] == pytest.approx(.004)
    assert snapshot["current_relation_within_bounds"] is False
    assert snapshot["holding_verified"] is False
    assert observer.latest is first
    with pytest.raises(ValueError, match="slipped"):
        observer.observe(phase="carry")
    plant.slip = 0.
    snapshot = observer.diagnostic_snapshot()
    assert "slipped" in snapshot["failure_reason"]
    assert snapshot["current_relation_within_bounds"] is False
    assert snapshot["holding_verified"] is False


def test_diagnostic_read_reports_unestablished_and_new_command_without_proving_hold():
    plant, observer = _observer()
    snapshot = observer.diagnostic_snapshot()
    assert snapshot["last_evidence_sim_time_s"] is None
    assert snapshot["relative_rotation_error_deg"] is None
    assert snapshot["holding_verified"] is False
    assert observer.latest is None
    first = _lift(plant, observer)
    plant.command = (2, 0.)
    snapshot = observer.diagnostic_snapshot()
    assert snapshot["command_matches_relation"] is False
    assert snapshot["holding_verified"] is False
    assert observer.latest is first


def test_release_diagnostics_continue_on_existing_physics_ticks_without_grasp_reanchoring():
    plant, observer = _observer()
    first = _lift(plant, observer)
    plant.command, plant.support, plant.sides = (2, 0.), True, (False, False)
    for _ in range(62):
        plant.time += .01
        _physics_observation(plant, observer)
    snapshot = observer.diagnostic_snapshot()
    assert snapshot["release_observation"]["stable_supported_release"]
    assert snapshot["holding_verified"] is False
    assert observer.latest is None and observer._anchor is None
    assert first.status == "held_verified"
    before = snapshot["release_observation"]
    for _ in range(3):
        _physics_observation(plant, observer)
        assert observer.diagnostic_snapshot()["release_observation"] == before
    plant.time += .01
    plant.slip = .003
    unobserved = observer.diagnostic_snapshot()["release_observation"]
    assert unobserved["status"] == "current_cut_unobserved"
    assert not unobserved["stable_supported_release"]
    assert unobserved["measurement_count"] == before["measurement_count"]
    _physics_observation(plant, observer)
    snapshot = observer.diagnostic_snapshot()
    assert snapshot["release_observation"]["status"] == "moving"
    assert not snapshot["release_observation"]["stable_supported_release"]
    assert observer.latest is None and observer._anchor is None


def test_release_monitor_does_not_infer_release_from_startup_open_or_stopped_sampling():
    plant, observer = _observer()
    plant.command = (0, 0.)
    _physics_observation(plant, observer)
    assert observer.diagnostic_snapshot()["release_observation"] is None
    plant.command = (1, 1.)
    _lift(plant, observer)
    plant.command, plant.support, plant.sides = (2, 0.), True, (False, False)
    for _ in range(62):
        plant.time += .01
        _physics_observation(plant, observer)
    plant.time += .1
    snapshot = observer.diagnostic_snapshot()
    assert snapshot["release_observation"]["status"] == "stale"
    assert not snapshot["release_observation"]["stable_supported_release"]
    plant.command = (3, 1.)
    assert observer.diagnostic_snapshot()["release_observation"] is None
    _physics_observation(plant, observer)
    assert observer._release_window is None and observer.latest is None


def test_release_contact_is_measured_and_original_carry_precondition_remains_required():
    plant, observer = _observer()
    _lift(plant, observer)
    plant.command, plant.support = (2, 0.), True
    plant.time += .01
    _physics_observation(plant, observer)
    assert observer.diagnostic_snapshot()["release_observation"]["status"] == "not_released"
    with pytest.raises(ObjectRelationInvalidated, match="closed command"):
        observer.observe(phase="carry", establish=True)
    plant.sides = (False, False)
    for _ in range(62):
        plant.time += .01
        _physics_observation(plant, observer)
    assert observer.diagnostic_snapshot()["release_observation"]["stable_supported_release"]
    assert observer.latest is None


def test_diagnostic_read_rejects_reversed_time_without_altering_evidence():
    plant, observer = _observer()
    first = _lift(plant, observer)
    plant.time -= .01
    with pytest.raises(ValueError, match="went backwards"):
        observer.diagnostic_snapshot()
    assert observer.latest is first
