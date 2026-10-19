"""Synthetic command/position evidence, not hardware or measured-force proof."""

import pytest
from test_b601_calibration_session import Replay
from test_b601_calibration_session import settings as shared_settings

from robot_motion.rebot_b601_rs.holding_calibration import (
    HoldingForceSearch,
    HoldingSearchConfig,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


@pytest.fixture
def settings(tmp_path):
    return shared_settings.__wrapped__(tmp_path)


@pytest.mark.parametrize("sign", [-1, 1])
def test_search_finds_repeatable_nonzero_hold_boundary_without_claiming_unique_gravity(
    sign,
):
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=sign * 0.4, cap_nm=1)
    while search.state == "searching":
        u = search.candidate_nm
        # Known synthetic static interval, not a B601 calibration.
        aligned = sign * u
        outcome = (
            "stationary"
            if 0.26 <= aligned <= 0.6
            else ("positive_motion" if u > sign * 0.4 else "negative_motion")
        )
        search.record(command_nm=u, outcome=outcome, evidence={"same_pose": True})
    result = search.status(detailed=True)
    assert result["state"] == "complete", result
    assert 0.26 <= abs(result["usable_hold_nm"]) <= 0.285
    assert (
        sum(
            r["outcome"] == "stationary" and r["command_nm"] == result["usable_hold_nm"]
            for r in result["observations"]
        )
        >= 2
    )
    assert result["identifies_unique_gravity"] is result["auto_apply"] is False


def test_zero_is_a_valid_hold_and_requires_repetition():
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=0, cap_nm=1)
    search.record(command_nm=0, outcome="stationary", evidence={})
    assert search.state == "searching"
    search.record(command_nm=0, outcome="stationary", evidence={})
    assert search.state == "complete"
    assert search.status()["usable_hold_nm"] == 0


def test_seed_and_zero_need_not_form_a_valid_bracket_and_expansion_is_bounded():
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=-0.1, cap_nm=0.4)
    while search.state == "searching":
        search.record(
            command_nm=search.candidate_nm, outcome="positive_motion", evidence={}
        )
    assert search.reason == "holding_authority_insufficient"
    assert all(abs(r["command_nm"]) <= 0.4 for r in search.observations)


def test_inconsistent_same_level_is_not_forced_into_a_binary_result():
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=-0.3, cap_nm=1)
    search.record(command_nm=-0.3, outcome="stationary", evidence={})
    search.record(command_nm=-0.3, outcome="positive_motion", evidence={})
    assert search.state == "incomplete"
    assert search.status()["usable_hold_nm"] is None


@pytest.mark.parametrize("sign", [-1, 1])
def test_censored_withdrawal_refines_next_probe_without_fabricating_motion_bound(sign):
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=sign * 0.304858, cap_nm=1)
    search.record(command_nm=search.candidate_nm, outcome="stationary", evidence={})
    assert search.candidate_nm == 0
    search.record_transition(
        command_nm=sign * 0.070971,
        requested_command_nm=0,
        evidence={"outcome": "censored_plateau_outside_point"},
    )
    first_retry = search.candidate_nm
    assert first_retry == pytest.approx(sign * (0.304858 + 0.070971) / 2)
    result = search.status(detailed=True)
    assert result["trial_count"] == 1
    assert result["transition_count"] == 1
    assert result["negative_motion_below_nm"] is None
    assert result["positive_motion_above_nm"] is None
    assert result["usable_hold_nm"] is None
    # Even a successful smaller probe must not restart the failed zero sweep.
    search.record(command_nm=first_retry, outcome="stationary", evidence={})
    assert abs(sign * 0.070971) < abs(search.candidate_nm) < abs(first_retry)


def test_resolved_nonzero_motion_bound_is_used_before_retrying_zero():
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=-0.3, cap_nm=1)
    search.record(command_nm=-0.3, outcome="stationary", evidence={})
    search.record(command_nm=-0.1, outcome="positive_motion", evidence={})
    assert search.candidate_nm == pytest.approx(-0.2)


def test_transition_refinement_requires_a_repeated_hold_not_a_fabricated_root():
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=-0.3, cap_nm=1)
    search.record(command_nm=-0.3, outcome="stationary", evidence={})
    search.record_transition(command_nm=-0.28, requested_command_nm=0, evidence={})
    assert search.candidate_nm == -0.3
    assert search.state == "searching"
    search.record(command_nm=-0.3, outcome="stationary", evidence={})
    assert search.state == "complete"
    assert search.status()["usable_hold_nm"] == -0.3
    assert search.status()["positive_motion_above_nm"] is None


@pytest.mark.parametrize("command", [True, float("nan"), 1.001])
def test_transition_rejects_invalid_command_without_mutating_search(command):
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=-0.3, cap_nm=1)
    search.record(command_nm=-0.3, outcome="stationary", evidence={})
    with pytest.raises(ValueError, match="envelope"):
        search.record_transition(command_nm=command, requested_command_nm=0, evidence={})
    assert search.transitions == []
    assert search.candidate_nm == 0


def test_censored_attempts_remain_in_original_finite_trial_budget():
    search = HoldingForceSearch(HoldingSearchConfig(max_trials=2), seed_nm=-0.3, cap_nm=1)
    search.record(command_nm=-0.3, outcome="stationary", evidence={})
    search.record_transition(command_nm=-0.1, requested_command_nm=0, evidence={})
    assert search.state == "incomplete"
    assert search.reason == "holding_trial_budget_exhausted"
    assert search.status()["usable_hold_nm"] is None


def test_stale_transition_cannot_change_a_later_probe():
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=-0.3, cap_nm=1)
    search.record(command_nm=-0.3, outcome="stationary", evidence={})
    search.record_transition(command_nm=-0.1, requested_command_nm=0, evidence={})
    with pytest.raises(ValueError, match="current probe"):
        search.record_transition(command_nm=-0.1, requested_command_nm=0, evidence={})
    assert len(search.transitions) == 1


def test_transition_after_repeated_old_holds_still_needs_a_fresh_return_check():
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=-0.3, cap_nm=1)
    for _ in range(2):
        search.record(command_nm=-0.3, outcome="stationary", evidence={})
    search.record_transition(command_nm=-0.28, requested_command_nm=0, evidence={})
    assert search.state == "searching"
    assert search.candidate_nm == -0.3
    assert search.status()["usable_hold_nm"] is None
    search.record(command_nm=-0.3, outcome="stationary", evidence={})
    assert search.state == "complete"


@pytest.mark.parametrize(
    "values",
    [
        {"observation_sec": True},
        {"max_trials": 1},
        {"verification_repeats": 2.0},
        {"movement_rad": float("nan")},
        {"observation_sec": 4, "observation_timeout_sec": 3},
    ],
)
def test_invalid_search_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        HoldingSearchConfig(**values)


def holding_replay(settings, *, control_period_sec=0.01, joint_name="joint3"):
    from dataclasses import replace

    from control_station.b601_torque_console.automatic_calibration import (
        AutomaticCalibrationConfig,
        B601AutomaticCalibrationController,
    )

    config = AutomaticCalibrationConfig(
        motion_plan="local_identification",
        motion_acquisition_arrival_tolerance_rad=0.0039,
        motion_position_resolution_rad=0.0004,
        motion_responsive_tracking_brake=True,
        motion_reference_pacing=True,
        motion_speed_governor=True,
        motion_arrival_stabilization=True,
        motion_continuous_speed_regulation=True,
        motion_tracking_recovery_cap_nm=settings.limits.max_abs_assist_nm,
        motion_breakaway_probe=True,
        motion_onset_handoff="track",
        motion_tracking_recovery_velocity_feedback_nm_s_per_rad=1,
    )
    resolved = B601AutomaticCalibrationController(settings, config).settings
    # Workspace resolves the established profile first, then binds each axis.
    resolved = replace(resolved, limits=replace(resolved.limits, joint_name=joint_name),
        staged_assist=replace(resolved.staged_assist, joint_name=joint_name))
    # One existing local-sweep endpoint; budgets are not reset by reacquisition.
    return Replay(
        resolved,
        plan_change=lambda p: p.update(steps=p["steps"][:1]),
        acquisition_arrival_tolerance_rad=0.0039,
        holding_search=HoldingSearchConfig(
            observation_sec=0.15, observation_timeout_sec=2
        ),
        control_period_sec=control_period_sec,
    )


def zero_point_replay(settings, *, period=0.01, joint_name="joint3", limits=None, centre_first=True):
    resolved = holding_replay(settings, joint_name=joint_name).controller.settings
    def centre_then_move(plan):
        first, second = plan["steps"][:2]
        centre = {**second, "target_qpos_rad": plan["anchor_qpos_rad"]}
        plan["steps"] = [centre, first] if centre_first else [first, centre]
    # The local plan revisits its centre; recoil can arrive there before the
    # next leg starts, as in k3b28lv8. Keep a nonzero next leg in the same plan.
    return Replay(resolved,
        plan_change=centre_then_move,
        acquisition_arrival_tolerance_rad=0.0039,
        holding_search=HoldingSearchConfig(observation_sec=0.15, observation_timeout_sec=2),
        control_period_sec=period, limits=limits)


@pytest.mark.parametrize("period", [0.01, 0.025])
@pytest.mark.parametrize("joint_name", ["joint2", "joint3"])
def test_already_at_point_observes_zero_twice_without_cue_then_runs_next_leg(settings, period, joint_name):
    replay = zero_point_replay(settings, period=period, joint_name=joint_name)
    replay.begin(breakaway=True)
    replay.tick()
    assert replay.session.status()["state"] == "observing"
    replay.run_until(lambda: bool(replay.session.status()["completed_legs"]), max_ticks=1000)
    point = replay.session.status()["completed_legs"][0]["holding_calibration"]
    assert point["usable_hold_nm"] == 0
    assert point["observation_mode"] == "passive_zero_assist"
    assert len(point["observations"]) == 2
    assert all(r["outcome"] == "stationary" and r["command_nm"] == 0 for r in point["observations"])
    assert all(r["started_at_sec"] > r["first_causal_receipt_at_sec"] for r in point["observations"])
    assert all(row["commands"]["total_nm"] == (0.0,) * 6 for row in replay.history)
    assert all(row["commands"]["driver_kd_nm_s_per_rad"] == (0.0,) * 6 for row in replay.history)
    assert not any(row["audit"]["phase"] in {"cue", "breakaway", "track"} for row in replay.history)
    replay.run_until(lambda: replay.session.status()["state"] in {"complete", "failed"},
                     move=True, max_ticks=3000)
    assert replay.session.result_record()["state"] == "complete"
    assert len(replay.session.result_record()["completed_legs"]) == 2


@pytest.mark.parametrize("kind", ["human", "permit", "command_tail", "hard_speed"])
def test_zero_observation_stops_on_lost_authority_or_invalid_command(settings, kind):
    replay = zero_point_replay(settings)
    replay.begin(breakaway=True)
    replay.tick()
    assert replay.session.status()["state"] == "observing"
    kwargs = {"human": True} if kind == "human" else {"permit": False} if kind == "permit" else {}
    if kind == "command_tail":
        kwargs["command_delta"] = (0, 0, 0.001, 0, 0, 0)
    elif kind == "hard_speed":
        q = list(replay.q)
        q[2] += 1.1 * replay.controller.limits.max_joint_velocity_rad_s * replay.control_period_sec
        kwargs["q"] = q
    replay.tick(**kwargs)
    assert replay.session.status()["state"] in {"releasing", "failed", "cancelled"}
    assert replay.session.status()["completed_legs"] == []
    assert replay.session._holding.status()["usable_hold_nm"] is None
    assert not any(row["audit"]["phase"] in {"cue", "breakaway", "track"} for row in replay.history)


def test_zero_observation_keeps_original_leg_deadline(settings):
    from dataclasses import replace
    template = zero_point_replay(settings)
    replay = zero_point_replay(settings, limits=replace(template.session.limits, max_leg_sec=0.12))
    replay.begin(breakaway=True)
    replay.run_until(lambda: replay.session.status()["state"] in {"failed", "cancelled"}, max_ticks=100)
    assert replay.session.result_record()["reason"] == "leg_timeout"
    assert replay.session.result_record()["completed_legs"] == []


def test_near_point_waits_for_quiet_window_instead_of_publishing_tiny_cue(settings):
    replay = zero_point_replay(settings)
    for _ in range(17):
        q = list(replay.q)
        q[2] += 0.0001
        replay.tick(q=q)
    replay.begin(breakaway=True)
    replay.tick()
    assert replay.session.status()["state"] == "preparing"
    assert replay.controller.owner_snapshot()["operator_phase"] == "inactive"
    replay.run_until(lambda: replay.session.status()["state"] == "observing", max_ticks=50)
    assert not any(row["audit"]["phase"] in {"cue", "breakaway", "track"} for row in replay.history)


def test_zero_observation_drift_is_not_completed_or_retried_with_force(settings):
    replay = zero_point_replay(settings)
    replay.begin(breakaway=True)
    replay.run_until(lambda: replay.session.status()["state"] == "observing")
    for _ in range(25):
        q = list(replay.q)
        q[2] += 0.0001
        replay.tick(q=q)
    replay.run_until(lambda: replay.session.status()["state"] == "failed", max_ticks=300)
    assert replay.session.status()["reason"] == "holding_search_incomplete"
    assert replay.session.status()["completed_legs"] == []
    assert all(row["commands"]["total_nm"] == (0.0,) * 6 for row in replay.history)


@pytest.mark.parametrize("period", [0.01, 0.025])
def test_nonzero_hold_recoil_can_land_at_next_zero_hold_point(settings, period):
    replay = zero_point_replay(settings, period=period, centre_first=False)
    sample = prime_offset_arrival(replay)
    for _ in range(4000):
        status = replay.session.status()
        if status["state"] in {"complete", "failed"}:
            break
        if status["step_index"] == 0 and status["state"] == "releasing":
            q = list(replay.q)
            q[2] += max(-0.005 * period, min(0.005 * period, -q[2]))
            replay.tick(q=q)
        elif status["step_index"] == 0:
            q = list(replay.q)
            if replay.session._holding_status()["stage"] == "fixed_command_observation":
                sent = replay.commands["total_nm"][2]
                if sent > -0.12:
                    q[2] += 0.015 * period
                elif sent < -0.24:
                    q[2] -= 0.015 * period
            else:
                q[2] += max(-0.01 * period, min(0.01 * period, sample - q[2]))
            replay.tick(q=q)
        else:
            replay.tick()
    result = replay.session.result_record()
    assert result["state"] == "complete", result
    assert result["completed_legs"][0]["holding_calibration"]["usable_hold_nm"] != 0
    assert result["completed_legs"][1]["holding_calibration"]["usable_hold_nm"] == 0
    assert not any(e.get("step_index") == 1 and e["state"] in {"arming", "moving"} for e in result["events"])


@pytest.mark.parametrize("period", [0.01, 0.025])
def test_native_session_reaches_fixed_hold_before_zero_and_retains_target(
    settings, period
):
    replay = holding_replay(settings, control_period_sec=period)
    replay.begin(breakaway=True)
    replay.run_until(
        lambda: (
            replay.session.status()["holding_calibration"]["probe_command_nm"]
            is not None
        ),
        max_ticks=3000,
        move=True,
    )
    target = replay.plan["steps"][0]["target_qpos_rad"]
    replay.run_until(
        lambda: replay.session.status()["state"] in {"complete", "failed"},
        max_ticks=3000,
        move=True,
    )
    result = replay.session.status()
    assert result["state"] == "complete", result
    point = result["completed_legs"][0]["holding_calibration"]
    assert point["state"] == "complete"
    assert all(r["nominal_target_rad"] == target for r in point["observations"])
    assert all(
        r["started_at_sec"] > r["first_causal_receipt_at_sec"]
        for r in point["observations"]
    )
    assert any(
        row["audit"].get("motion_control", {}).get("reason")
        == "calibration_fixed_command_observation"
        for row in replay.history
        if row["audit"].get("motion_control")
    )


def test_human_override_withdraws_during_probe_without_recording_false_hold(settings):
    replay = holding_replay(settings)
    replay.begin(breakaway=True)
    replay.run_until(
        lambda: replay.session._holding_probe_nm is not None, max_ticks=3000, move=True
    )
    replay.tick(human=True)
    assert replay.session.status()["state"] == "releasing"
    assert (
        replay.audit["motion_control"]["reason"]
        != "calibration_fixed_command_observation"
    )
    assert replay.session.status()["completed_legs"] == []


def scripted_tick(replay, *, drift_on_release=False):
    """Prescribed monotone signed response, not an identified physical plant."""
    q = list(replay.q)
    j = 2
    target = replay.plan["steps"][0]["target_qpos_rad"][j]
    phase = replay.session.status()["state"]
    motion = replay.audit.get("motion_control") or {}
    if phase == "moving":
        if motion.get("reason") == "calibration_fixed_command_observation":
            sent = replay.commands["total_nm"][j]
            if sent > -0.04:
                q[j] += 0.00008
            elif sent < -0.08:
                q[j] -= 0.00008
        elif motion.get("phase") == "track":
            q[j] = motion["q_reference_rad"]
        elif motion.get("arrival_stabilization_active"):
            q[j] += max(-0.00008, min(0.00008, target - q[j]))
        elif replay.audit.get("phase") == "breakaway" or (
            replay.audit.get("phase") == "releasing"
            and not replay.audit.get("release_terminal")
        ):
            q[j] += max(-0.0001, min(0.0001, target - q[j]))
    elif phase == "releasing" and drift_on_release:
        q[j] += max(-0.00005, min(0.00005, target + 0.006 - q[j]))
    replay.tick(q=q)


def prime_offset_arrival(replay):
    """Prescribe the measured offset and bounded arrival command from the fault."""
    replay.begin(breakaway=True)
    replay.run_until(
        lambda: (replay.audit.get("motion_control") or {}).get("phase") == "track",
        move=True,
    )
    j = int(replay.controller.limits.joint_name[-1]) - 1
    target = replay.plan["steps"][0]["target_qpos_rad"][j]
    sample = target + 0.0019466
    for _ in range(1000):
        q = list(replay.q)
        step = 0.01 * replay.control_period_sec
        q[j] += max(-step, min(step, sample - q[j]))
        delta = [0.0] * 6
        delta[j] = -0.2
        replay.tick(q=q, command_delta=tuple(delta) if replay.session._holding is None else None)
        if replay.session._holding is not None and replay.session._holding.observations:
            return sample
    pytest.fail(f"offset arrival was not observed: {replay.session.status()}")


@pytest.mark.parametrize("period", [0.01, 0.025])
@pytest.mark.parametrize("joint_name", ["joint2", "joint3"])
def test_native_censored_recoil_changes_candidate_and_retains_stable_return_command(settings, period, joint_name):
    replay = holding_replay(settings, control_period_sec=period, joint_name=joint_name)
    j = int(joint_name[-1]) - 1
    sample = prime_offset_arrival(replay)
    for _ in range(500):
        q = list(replay.q)
        holding = replay.session._holding_status()
        if holding["stage"] == "fixed_command_observation":
            # Release during the ramp; then keep moving briefly at its plateau.
            sent = replay.commands["total_nm"][j]
            if sent > -0.15:
                q[j] += 0.015 * period
        replay.tick(q=q)
        if replay.session._holding.transitions:
            break
    result = replay.session._holding.status(detailed=True)
    assert result["transition_count"] == 1, replay.session.status()
    assert -0.2 < result["candidate_nm"] < 0
    assert result["trial_count"] == 1
    assert result["positive_motion_above_nm"] is None
    assert result["negative_motion_below_nm"] is None
    assert replay.session._holding_reacquire_bias_nm == -0.2
    assert replay.session._holding_pose[j] == pytest.approx(sample)
    transition = result["transitions"][0]
    assert transition["requested_command_nm"] == 0
    assert transition["actual_plateau_command_nm"] == transition["command_nm"]
    assert transition["completed_at_sec"] > transition["first_causal_receipt_at_sec"]
    assert transition["final_qpos_rad"][j] > sample + 0.002
    assert replay.session._leg_started < transition["reacquired_at_sec"]
    assert replay.session.status()["batch_command_exposure_nm_sec"] > 0


def test_native_point_comparison_uses_actual_sample_not_nominal_arrival_edge(settings):
    replay = holding_replay(settings)
    sample = prime_offset_arrival(replay)
    # The nominal target is almost 0.002 rad below the actual sample. A single
    # additional count must not consume the entire same-point observation band.
    target = replay.plan["steps"][0]["target_qpos_rad"][2]
    for _ in range(100):
        q = list(replay.q)
        q[2] += min(0.00004, max(0, sample + 0.0004 - q[2]))
        replay.tick(q=q, command_delta=(0, 0, 0, 0, 0, 0))
        if len(replay.session._holding.observations) >= 2:
            break
    observations = replay.session._holding.observations
    assert len(observations) == 2, replay.session.status()
    assert observations[1]["outcome"] == "stationary"
    assert observations[1]["final_qpos_rad"][2] - target > 0.002
    assert observations[1]["final_qpos_rad"][2] - sample < 0.002


def test_short_plateau_recoil_then_stop_refines_without_claiming_sustained_motion(settings):
    replay = holding_replay(settings)
    prime_offset_arrival(replay)
    replay.run_until(lambda: replay.session._holding_observation is not None
        and replay.session._holding_observation["requested_command_nm"] == 0
        and replay.session._holding_observation["started_at_sec"] is not None)
    # Complete a short rebound before the 80 ms sustained-motion window.
    for _ in range(7):
        q = list(replay.q)
        q[2] += 0.029 * replay.control_period_sec
        replay.tick(q=q)
    replay.run_until(lambda: bool(replay.session._holding.transitions), max_ticks=100)
    result = replay.session._holding.status(detailed=True)
    assert result["transitions"][0]["outcome"] == "censored_settled_outside_point"
    assert result["candidate_nm"] == pytest.approx(-0.1)
    assert result["trial_count"] == 1
    assert result["positive_motion_above_nm"] is None


@pytest.mark.parametrize("period", [0.01, 0.025])
def test_native_transition_refinement_reacquires_verifies_then_withdraws(settings, period):
    replay = holding_replay(settings, control_period_sec=period)
    sample = prime_offset_arrival(replay)
    for _ in range(2000):
        state = replay.session.status()["state"]
        if state in {"complete", "failed"}:
            break
        q = list(replay.q)
        if state == "moving":
            if replay.session._holding_status()["stage"] == "fixed_command_observation":
                # Synthetic static interval; not an identified gearbox model.
                sent = replay.commands["total_nm"][2]
                if sent > -0.12:
                    q[2] += 0.015 * period
                elif sent < -0.24:
                    q[2] -= 0.015 * period
            else:
                q[2] += max(-0.01 * period, min(0.01 * period, sample - q[2]))
        elif state == "releasing":
            q[2] += max(-0.005 * period, min(0.005 * period, sample + 0.006 - q[2]))
        replay.tick(q=q)
    result = replay.session.result_record()
    assert result["state"] == "complete", result
    leg = result["completed_legs"][0]
    point = leg["holding_calibration"]
    assert point["transition_count"] >= 1
    assert point["sample_pose_rad"][2] == pytest.approx(sample)
    assert point["nominal_target_rad"] == replay.plan["steps"][0]["target_qpos_rad"]
    assert -0.24 <= point["usable_hold_nm"] <= -0.12
    assert sum(r["outcome"] == "stationary" and r["command_nm"] == point["usable_hold_nm"]
               for r in point["observations"]) >= 2
    assert leg["final_zero_and_still_confirmed"]
    assert abs(leg["target_error_rad"]) > result["acquisition_arrival_tolerance_rad"]
    assert all(abs(row["commands"]["total_nm"][2]) <= 0.3 + 1e-9 for row in replay.history)


def test_nonzero_hold_survives_subsequent_recoil_and_withdrawal_still_requires_receipts(
    settings,
):
    replay = holding_replay(settings)
    replay.begin(breakaway=True)
    for _ in range(3000):
        if replay.session.status()["state"] in {"complete", "failed"}:
            break
        scripted_tick(replay, drift_on_release=True)
    result = replay.session.status()
    assert result["state"] == "complete", result
    leg = result["completed_legs"][0]
    point = leg["holding_calibration"]
    assert -0.08 <= point["usable_hold_nm"] <= -0.04
    assert abs(leg["target_error_rad"]) > result["acquisition_arrival_tolerance_rad"]
    assert leg["final_zero_and_still_confirmed"]
    assert (
        leg["arrival_semantics"]
        == "verified_nonzero_hold_then_separate_safe_withdrawal"
    )
    assert any(r["outcome"] == "positive_motion" for r in point["observations"])
    assert any(
        (row["audit"].get("tracking_recovery") or {}).get(
            "calibration_approach_bias_nm", 0
        )
        != 0
        for row in replay.history
    )
    assert all(
        abs(row["commands"]["total_nm"][2])
        <= replay.controller.limits.max_abs_assist_nm + 1e-9
        for row in replay.history
    )
    assert all(
        row["commands"]["total_nm"][i] == 0
        for row in replay.history
        for i in (0, 1, 3, 4, 5)
    )
    assert (
        result["leg_command_exposure_nm_sec"] == 0
    )  # Reset only after the completed leg.
    assert result["batch_command_exposure_nm_sec"] > 0


@pytest.mark.parametrize("tail_joint", [1, 2])
def test_conditioned_tail_cannot_start_observation_clock(settings, tail_joint):
    replay = holding_replay(settings)
    replay.begin(breakaway=True)
    replay.run_until(
        lambda: replay.session._holding_probe_nm is not None, max_ticks=3000, move=True
    )
    goal = replay.session._holding_probe_nm
    for _ in range(25):
        delta = [0.0] * 6
        delta[2] = goal
        delta[tail_joint] += 0.001
        replay.tick(command_delta=tuple(delta))
    assert replay.session._holding_observation["started_at_sec"] is None
    assert replay.session._holding.observations == []


def test_holding_probe_does_not_disable_raw_or_soft_speed_stops(settings):
    replay = holding_replay(settings)
    replay.begin(breakaway=True)
    replay.run_until(
        lambda: replay.session._holding_probe_nm is not None, max_ticks=3000, move=True
    )
    q = list(replay.q)
    q[2] += replay.controller.limits.max_joint_velocity_rad_s * 0.011
    replay.tick(q=q)
    assert replay.audit["phase"] == "fault"
    assert "velocity_exceeded" in replay.audit["fault"]
    assert replay.session.status()["completed_legs"] == []


@pytest.mark.parametrize("joint", [1, 2])
def test_soft_motion_stops_override_fixed_command_observation(settings, joint):
    replay = holding_replay(settings)
    replay.begin(breakaway=True)
    replay.run_until(
        lambda: replay.session._holding_probe_nm is not None, max_ticks=3000, move=True
    )
    soft = replay.controller.settings.staged_assist.motion_control.soft_velocity_rad_s
    assert soft * 1.2 < replay.controller.limits.max_joint_velocity_rad_s
    for _ in range(12):
        q = list(replay.q)
        q[joint] += soft * 1.2 * 0.01
        replay.tick(q=q)
        if (replay.audit.get("motion_control") or {}).get(
            "reason"
        ) != "calibration_fixed_command_observation":
            break
    assert replay.audit["motion_control"]["phase"] == "brake"
    assert replay.session._holding_probe_nm is None
    assert replay.session.status()["completed_legs"] == []


def test_binary_search_has_a_finite_trial_budget():
    search = HoldingForceSearch(
        HoldingSearchConfig(max_trials=2), seed_nm=-0.4, cap_nm=1
    )
    search.record(command_nm=-0.4, outcome="stationary", evidence={})
    search.record(command_nm=0, outcome="positive_motion", evidence={})
    assert search.state == "incomplete"
    assert search.reason == "holding_trial_budget_exhausted"
    assert search.status()["usable_hold_nm"] is None


def test_computed_receipt_roundoff_at_the_cap_does_not_widen_the_command_limit():
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=0.1 + 0.2, cap_nm=0.3)
    assert search.candidate_nm == 0.3
    with pytest.raises(ValueError, match="envelope"):
        HoldingForceSearch(HoldingSearchConfig(), seed_nm=0.30001, cap_nm=0.3)


def test_explicit_profile_samples_nearby_points_without_changing_legacy_stop_band(
    tmp_path,
):
    import json
    import math
    from dataclasses import replace
    from pathlib import Path

    from test_b601_workspace_calibration import LIMITS, production_controller

    from control_station.b601_torque_console.workspace_calibration import (
        B601WorkspaceCalibrationController,
        WorkspaceCalibrationConfig,
    )
    from scripts.probes.b601_workspace_collision_audit import nominal_segments

    legacy = production_controller.__wrapped__(tmp_path)._workspace
    profile = (
        Path(__file__).resolve().parents[1]
        / "docs/hardware/profiles/b601_j2_j3_holding_search_v1.json"
    )
    config = WorkspaceCalibrationConfig(**json.loads(profile.read_text())["workspace"])
    controller = B601WorkspaceCalibrationController(
        legacy.settings, legacy.config, config
    )
    controller.configure_context(Replay(legacy.settings).condition, LIMITS)
    anchor = (0.0, math.radians(3), math.radians(3), 0.0, 0.0, 0.0)
    controller._q = anchor
    controller._tasks = [{"kind": "sweep", "joint": "joint2", "pose": 0}]
    controller._begin_task(1.0)
    session = controller._session
    assert (
        max(abs(s["target_qpos_rad"][1] - anchor[1]) for s in session._steps)
        <= 0.01 + 1e-12
    )
    assert session.acquisition_arrival_tolerance_rad == 0.002
    assert (
        controller.settings.staged_assist.settle_error_rad
        == legacy.settings.staged_assist.settle_error_rad
    )
    segments = list(
        nominal_segments(config, legacy.config, (0.0,) * 6, controller._condition)
    )
    assert all(
        abs(end[int(task["joint"][-1]) - 1] - math.radians(3)) <= 0.01 + 1e-12
        for task, _, end in segments
        if task["kind"] == "sweep"
    )
    with pytest.raises(ValueError, match="existing local sweep"):
        B601WorkspaceCalibrationController(
            legacy.settings, legacy.config, replace(config, holding_excursion_rad=0.04)
        )


def test_slow_approach_to_plateau_does_not_consume_constant_observation_timeout(
    settings,
):
    replay = holding_replay(settings)
    replay.begin(breakaway=True)
    replay.run_until(
        lambda: replay.session._holding_probe_nm is not None, max_ticks=3000, move=True
    )
    goal = replay.session._holding_probe_nm
    for _ in range(210):
        delta = [0.0] * 6
        delta[2] = goal + 0.001
        replay.tick(command_delta=tuple(delta))
    assert replay.session.status()["state"] == "moving"
    assert replay.session._holding.observations == []
    assert replay.session._holding_observation["started_at_sec"] is None


def test_holding_windows_are_not_silently_fitted_as_running_resistance():
    from control_station.b601_torque_console.calibration_report import _admission_reason

    assert (
        _admission_reason(
            {"torque_console": {"holding_calibration": {"enabled": True}}}, {}
        )
        == "holding_search_not_running_resistance"
    )


def test_workspace_gui_renders_the_search_and_clears_it_in_legacy_mode():
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable for the existing GUI JavaScript")
    script = r"""
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const src = fs.readFileSync("control_station/b601_torque_console/static/app.js", "utf8");
const fn = src.slice(src.indexOf("function renderWorkspace("), src.indexOf("function preparationReasonLabel("));
const nodes = {};
const context = {state: {owner_ready: true}, number: v => Number.isFinite(v) ? String(v) : "—",
  el: id => nodes[id] ||= {}, batch: {workspace: {phase: "sweep", setup_label: "candidate",
    holding_search_enabled: true, acquisition_ranges_deg: {joint2:[0,10], joint3:[0,10]},
    holding_calibration: {enabled:true, stage:"fixed_command_observation", attempt_count:3,
      max_attempts:12, probe_command_nm:-.3, search:{trial_count:2, transition_count:1,
        candidate_nm:-.2, reason:"next_bounded_probe",
        usable_hold_nm:null, observed_stationary_range_nm:[-.4,-.3]}}}}};
vm.createContext(context);
vm.runInContext(fn+"; renderWorkspace(batch);", context);
assert(nodes["phase"].textContent.includes("保持命令搜索"));
assert(nodes["automatic-progress"].textContent.includes("定命令观察"));
assert(nodes["automatic-progress"].textContent.includes("3/12"));
assert(nodes["automatic-progress"].textContent.includes("变力退回／缩步 1 次"));
assert(nodes["automatic-progress"].textContent.includes("下一试探 -0.2"));
assert(nodes["automatic-trials"].textContent.includes("不是实测保持力"));
context.batch.workspace.holding_calibration.stage="observing";
vm.runInContext("renderWorkspace(batch);", context);
assert(nodes["automatic-progress"].textContent.includes("零附加命令观察（不驱动）"));
context.batch.workspace.holding_calibration.stage="withdrawal";
vm.runInContext("renderWorkspace(batch);", context);
assert(nodes["automatic-progress"].textContent.includes("正在撤力停稳"));
context.batch.workspace.holding_search_enabled=false;
context.batch.workspace.holding_calibration=null;
vm.runInContext("renderWorkspace(batch);", context);
assert.equal(nodes["phase"].textContent, "双向双速采集");
assert.equal(nodes["automatic-trials"].textContent, "");
"""
    result = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("withdrawal_ok", [False, True])
def test_report_keeps_verified_hold_even_if_later_withdrawal_failed(
    tmp_path, withdrawal_ok
):
    import json

    from control_station.b601_torque_console.automatic_report import (
        build_automatic_report,
    )

    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=0, cap_nm=1)
    for _ in range(2):
        search.record(
            command_nm=0, outcome="stationary", evidence={"qpos_rad": [0.0] * 6}
        )
    point = search.status(detailed=True)
    leg = {"trial_id": "p1", "holding_calibration": point}
    child = {
        "state": "complete" if withdrawal_ok else "failed",
        "reason": "done"
        if withdrawal_ok
        else "withdrawal_not_confirmed_owner_protection_required",
        "completed_legs": [leg] if withdrawal_ok else [],
        "holding_calibration": None
        if withdrawal_ok
        else {
            "search": point,
            "sample_pose_rad": [0.0] * 6,
            "nominal_target_rad": [0.0] * 6,
        },
    }
    result = {
        "kind": "automatic_calibration_result",
        "schema": "b601.automatic-joint-calibration.v1",
        "state": child["state"],
        "reason": child["reason"],
        "onset_trials": [],
        "workspace": {
            "setup_label": "test",
            "acquisitions": [
                {
                    "task": {"kind": "sweep", "joint": "joint2", "pose": 0},
                    "result": child,
                }
            ],
        },
    }
    (tmp_path / "b601_owner.debug.jsonl").write_text(json.dumps(result) + "\n")
    (tmp_path / "b601_owner.transparency.jsonl").write_text("")
    report = build_automatic_report(tmp_path)
    assert report["holding_calibration"]["verified_points"] == 1
    assert (
        report["holding_calibration"]["points"][0]["withdrawal_confirmed"]
        == withdrawal_ok
    )
    assert report["holding_calibration"]["auto_apply"] is False
    assert report["holding_calibration"]["identifies_unique_gravity"] is False
    assert report["acquisition_complete"] == withdrawal_ok
