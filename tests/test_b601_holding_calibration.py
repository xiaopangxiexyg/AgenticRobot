"""Pure holding-search contracts; application/session/GUI cases remain in the source branch."""

import pytest
from robot_motion.rebot_b601_rs.holding_calibration import (
    HoldingForceSearch,
    HoldingSearchConfig,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


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
    assert search.return_command_nm is None


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


@pytest.mark.parametrize("sign", [-1, 1])
def test_pk81t8jv_failed_recheck_selects_another_observed_hold_not_same_retry(sign):
    # First six trial outcomes from pk81t8jv, mirrored to cover both signs.
    # Positions are handled by the session; censored ramps are not motion bounds.
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=sign * 0.3544235726612437, cap_nm=1)
    search.record(command_nm=search.candidate_nm, outcome="stationary", evidence={})
    search.record_transition(command_nm=sign * 0.07012636713020526,
                             requested_command_nm=0, evidence={})
    for _ in range(3):
        search.record(command_nm=search.candidate_nm, outcome="stationary", evidence={})
    weak = search.candidate_nm
    assert abs(weak) == pytest.approx(0.08789494247589516)
    search.record_transition(command_nm=weak, requested_command_nm=weak, evidence={})
    failed = search.candidate_nm
    assert abs(failed) == pytest.approx(0.10566351782158506)
    search.record_transition(command_nm=failed, requested_command_nm=failed, evidence={})
    fallback = search.candidate_nm
    assert abs(fallback) == pytest.approx(0.14120066851296487)
    assert search.return_command_nm == fallback
    assert search.state == "searching"
    assert search.status()["usable_hold_nm"] is None
    assert search.transitions[-1]["verification_command_nm"] == failed
    search.record(command_nm=fallback, outcome="stationary", evidence={"fresh_return": True})
    result = search.status(detailed=True)
    assert result["state"] == "complete"
    assert result["usable_hold_nm"] == fallback
    assert result["trial_count"] == 5
    assert result["transition_count"] == 3
    assert result["negative_motion_below_nm"] is result["positive_motion_above_nm"] is None
    assert failed in [r["command_nm"] for r in result["observations"]]


def test_censored_recheck_without_another_anchor_stops_without_fabricating_hold():
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=-0.3, cap_nm=1)
    search.record(command_nm=-0.3, outcome="stationary", evidence={})
    search.record_transition(command_nm=-0.28, requested_command_nm=0, evidence={})
    assert search.candidate_nm == -0.3
    search.record_transition(command_nm=-0.3, requested_command_nm=-0.3, evidence={})
    assert search.state == "incomplete"
    assert search.reason == "holding_no_repeatable_anchor"
    assert search.status()["usable_hold_nm"] is None
    assert len(search.observations) == 1


def test_each_failed_return_check_removes_only_that_candidate_from_further_retries():
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=-0.4, cap_nm=1)
    search.record(command_nm=-0.4, outcome="stationary", evidence={})
    search.record(command_nm=-0.3, outcome="stationary", evidence={})
    search.record_transition(command_nm=-0.29, requested_command_nm=0, evidence={})
    search.record_transition(command_nm=-0.3, requested_command_nm=-0.3, evidence={})
    assert search.candidate_nm == search.return_command_nm == -0.4
    search.record_transition(command_nm=-0.4, requested_command_nm=-0.4, evidence={})
    assert search.state == "incomplete"
    assert search.reason == "holding_no_repeatable_anchor"
    assert search.status()["usable_hold_nm"] is None
    assert len(search.observations) + len(search.transitions) < search.config.max_trials


@pytest.mark.parametrize("sign", [-1, 1])
def test_4j8unjzh_causal_frozen_plateau_is_new_hold_to_recheck_not_failed_request(sign):
    seed = sign * 0.3423907938939641
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=seed, cap_nm=1)
    search.record(command_nm=seed, outcome="stationary", evidence={})
    search.record_transition(command_nm=sign * 0.032948165695660725,
                             requested_command_nm=0, evidence={})
    search.record(command_nm=search.candidate_nm, outcome="stationary", evidence={})
    for _ in range(4):
        search.record_transition(command_nm=search.candidate_nm,
                                 requested_command_nm=search.candidate_nm, evidence={})
    assert search.candidate_nm == seed
    actual = sign * 0.32665337778768744
    search.record(command_nm=actual, requested_command_nm=seed,
                  outcome="stationary", evidence={"ramp_frozen": True})
    assert search.state == "searching"
    assert search.candidate_nm == search.return_command_nm == actual
    assert search.status()["usable_hold_nm"] is None
    search.record(command_nm=actual, requested_command_nm=actual,
                  outcome="stationary", evidence={"ramp_frozen": False})
    result = search.status(detailed=True)
    assert result["state"] == "complete"
    assert result["usable_hold_nm"] == actual
    assert [r["command_nm"] for r in result["observations"][-2:]] == [actual, actual]
    assert result["observations"][-2]["requested_command_nm"] == seed
    assert result["negative_motion_below_nm"] is result["positive_motion_above_nm"] is None


@pytest.mark.parametrize("requested", [True, float("nan"), 1.01, 0.1])
def test_changed_plateau_rejects_invalid_or_stale_request_before_recording(requested):
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=-0.3, cap_nm=1)
    with pytest.raises(ValueError, match="current probe"):
        search.record(command_nm=-0.28, requested_command_nm=requested,
                      outcome="stationary", evidence={"ramp_frozen": True})
    assert search.observations == []


def test_changed_plateau_requires_explicit_freeze_and_a_new_exact_repeat():
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=-0.3, cap_nm=1)
    search.record(command_nm=-0.3, outcome="stationary", evidence={})
    search.record_transition(command_nm=-0.28, requested_command_nm=0, evidence={})
    with pytest.raises(ValueError, match="frozen ramp"):
        search.record(command_nm=-0.29, requested_command_nm=-0.3,
                      outcome="stationary", evidence={"ramp_frozen": False})
    assert len(search.observations) == 1
    search.record(command_nm=-0.29, requested_command_nm=-0.3,
                  outcome="stationary", evidence={"ramp_frozen": True})
    assert search.state == "searching"
    search.record(command_nm=-0.29, requested_command_nm=-0.29,
                  outcome="positive_motion", evidence={})
    assert search.state == "incomplete"
    assert search.status()["usable_hold_nm"] is None


def test_new_plateau_does_not_renew_trial_budget_or_relax_unexplained_mismatch():
    search = HoldingForceSearch(HoldingSearchConfig(max_trials=3), seed_nm=-0.3, cap_nm=1)
    search.record(command_nm=-0.3, outcome="stationary", evidence={})
    search.record_transition(command_nm=-0.28, requested_command_nm=0, evidence={})
    search.record(command_nm=-0.29, requested_command_nm=-0.3,
                  outcome="stationary", evidence={"ramp_frozen": True})
    assert search.reason == "holding_trial_budget_exhausted"
    assert search.status()["usable_hold_nm"] is None
    unexplained = HoldingForceSearch(HoldingSearchConfig(), seed_nm=-0.3, cap_nm=1)
    unexplained.record(command_nm=-0.3, outcome="stationary", evidence={})
    unexplained.record_transition(command_nm=-0.28, requested_command_nm=0, evidence={})
    unexplained.record(command_nm=-0.29, outcome="stationary", evidence={})
    assert unexplained.reason == "holding_verification_changed"
    assert unexplained.status()["usable_hold_nm"] is None


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


@pytest.mark.parametrize("sign", [-1, 1])
def test_f0x4xla6_inconsistent_boundary_rechecks_an_observed_interior_hold(sign):
    search = HoldingForceSearch(HoldingSearchConfig(max_trials=27),
        seed_nm=sign*.3357710250542558, cap_nm=1, recheck_interior_on_conflict=True)
    search.record(command_nm=search.candidate_nm, outcome="stationary", evidence={})
    search.record_transition(command_nm=sign*.07752669457079908, requested_command_nm=0,
                             evidence={"outcome": "censored_plateau_outside_point"})
    for _ in range(2):
        search.record(command_nm=search.candidate_nm, outcome="stationary", evidence={})
    weak = search.observations[-1]["command_nm"]
    for _ in range(3):
        search.record(command_nm=search.candidate_nm,
                      outcome="negative_motion" if sign > 0 else "positive_motion", evidence={})
    assert search.state == "searching"
    fallback = search.candidate_nm
    assert fallback == pytest.approx(sign*.20664885981252745)
    assert search.return_command_nm == fallback
    assert weak in search.status()["rejected_stationary_commands_nm"]
    assert search.status()["usable_hold_nm"] is None
    retained = list(search.observations)
    for n in range(2):
        search.record(command_nm=fallback, outcome="stationary", evidence={"fresh_return": n})
        assert search.state == ("searching" if n == 0 else "complete")
    assert search.observations[:len(retained)] == retained
    assert search.status()["usable_hold_nm"] == fallback
    assert search.status()["auto_apply"] is search.status()["identifies_unique_gravity"] is False

@pytest.mark.parametrize("budget", [2, 3, 4])
def test_interior_recheck_never_renews_the_trial_budget(budget):
    search = HoldingForceSearch(HoldingSearchConfig(max_trials=budget), seed_nm=-.4,
        cap_nm=1, recheck_interior_on_conflict=True)
    search.record(command_nm=-.4, outcome="stationary", evidence={})
    search.record(command_nm=-.2, outcome="stationary", evidence={})
    if search.state == "searching":
        search.record(command_nm=-.2, outcome="positive_motion", evidence={})
    while search.state == "searching":
        search.record(command_nm=search.candidate_nm, outcome="stationary", evidence={})
    assert search.reason == "holding_trial_budget_exhausted"
    assert search.status()["usable_hold_nm"] is None
    assert len(search.observations) == budget

def test_interior_recheck_exhausts_candidates_without_reusing_a_rejected_hold():
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=-.4,
        cap_nm=1, recheck_interior_on_conflict=True)
    for command in [-.4, -.3, -.2]:
        search.record(command_nm=command, outcome="stationary", evidence={})
    for command in [-.2, -.3, -.4]:
        search.record(command_nm=command, outcome="positive_motion", evidence={})
        if command != -.4:
            assert search.state == "searching"
            assert search.candidate_nm < command
    assert search.state == "incomplete"
    assert search.status()["usable_hold_nm"] is None
    assert len(search.observations) == 6

def test_interior_recheck_never_treats_crossed_motion_bounds_as_a_hold():
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=-.4,
        cap_nm=1, recheck_interior_on_conflict=True)
    search.record(command_nm=-.4, outcome="stationary", evidence={})
    search.record(command_nm=-.2, outcome="positive_motion", evidence={})
    search.record(command_nm=-.1, outcome="negative_motion", evidence={})
    assert search.state == "incomplete"
    assert search.reason == "holding_response_not_repeatable_or_monotone"

@pytest.mark.parametrize("frozen_plateau", [False, True])
def test_interior_recheck_censored_or_changed_plateau_requires_fresh_exact_repeats(frozen_plateau):
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=-.4,
        cap_nm=1, recheck_interior_on_conflict=True)
    for command in [-.4, -.3, -.2]:
        search.record(command_nm=command, outcome="stationary", evidence={})
    search.record(command_nm=-.2, outcome="positive_motion", evidence={})
    assert search.candidate_nm == -.3
    if frozen_plateau:
        fallback = -.35
        search.record(command_nm=fallback, requested_command_nm=-.3, outcome="stationary",
                      evidence={"ramp_frozen": True})
    else:
        search.record(command_nm=-.3, outcome="stationary", evidence={})
        search.record_transition(command_nm=-.3, requested_command_nm=-.3, evidence={})
        fallback = search.candidate_nm
        assert fallback == -.4
        search.record(command_nm=fallback, outcome="stationary", evidence={})
    assert search.state == "searching"
    assert search.status()["usable_hold_nm"] is None
    search.record(command_nm=fallback, outcome="stationary", evidence={})
    assert search.state == "complete"
    assert search.status()["usable_hold_nm"] == fallback

@pytest.mark.parametrize("value", [1, None, "true"])
def test_interior_recheck_option_is_explicit_boolean(value):
    with pytest.raises(TypeError, match="boolean"):
        HoldingForceSearch(HoldingSearchConfig(), seed_nm=0, cap_nm=1,
                          recheck_interior_on_conflict=value)

@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("aligned", [False, True])
def test_unverified_arrival_uses_censored_direction_only_for_a_bounded_new_proposal(sign, aligned):
    search = HoldingForceSearch(HoldingSearchConfig(movement_rad=.0045), seed_nm=sign*.14,
        cap_nm=.3, recheck_interior_on_conflict=True)
    progress = sign*.0044*(1 if aligned else -1)
    search.record_transition(command_nm=sign*.14, requested_command_nm=sign*.14,
                             evidence={"net_displacement_rad": progress})
    assert search.state == "searching"
    assert search.candidate_nm == pytest.approx(sign*(.07 if aligned else .28))
    assert search.return_command_nm is None
    assert search.observations == []
    assert search.status()["positive_motion_above_nm"] is search.status()["negative_motion_below_nm"] is None
    search.record(command_nm=search.candidate_nm, outcome="stationary", evidence={})
    assert search.status()["usable_hold_nm"] is None

def test_unverified_arrival_proposals_share_the_original_transition_budget():
    search = HoldingForceSearch(HoldingSearchConfig(max_trials=2), seed_nm=.14,
        cap_nm=.3, recheck_interior_on_conflict=True)
    for _ in range(2):
        value = search.candidate_nm
        search.record_transition(command_nm=value, requested_command_nm=value,
                                 evidence={"net_displacement_rad": .003})
    assert search.state == "incomplete"
    assert search.reason == "holding_trial_budget_exhausted"
    assert len(search.transitions) == 2 and search.observations == []

@pytest.mark.parametrize("progress", [True, float("nan"), float("inf")])
def test_invalid_censored_progress_cannot_mutate_a_proposal(progress):
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=.14,
        cap_nm=.3, recheck_interior_on_conflict=True)
    with pytest.raises(ValueError, match="progress"):
        search.record_transition(command_nm=.14, requested_command_nm=.14,
                                 evidence={"net_displacement_rad": progress})
    assert search.transitions == [] and search.candidate_nm == .14


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("excluded_by", ["resolved_boundary", "failed_return"])
def test_excluded_frozen_plateau_never_becomes_a_return_or_usable_hold(sign, excluded_by):
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=sign * .4,
                               cap_nm=1, recheck_interior_on_conflict=True)
    for value in ([.4, .3, .2] if excluded_by == "resolved_boundary" else [.4, .3]):
        search.record(command_nm=sign * value, outcome="stationary", evidence={})
    if excluded_by == "resolved_boundary":
        search.record(command_nm=sign * .2,
                      outcome="negative_motion" if sign > 0 else "positive_motion", evidence={})
        actual, fallback = sign * .2, sign * .3
    else:
        search.record_transition(command_nm=sign * .29, requested_command_nm=0, evidence={})
        search.record_transition(command_nm=sign * .3, requested_command_nm=sign * .3, evidence={})
        actual, fallback = sign * .3, sign * .4
    assert search.candidate_nm == fallback
    before = list(search.observations)
    search.record(command_nm=actual, requested_command_nm=fallback,
                  outcome="stationary", evidence={"ramp_frozen": True})
    result = search.status(detailed=True)
    assert result["state"] == "searching"
    assert result["usable_hold_nm"] is None
    assert result["candidate_nm"] == result["return_command_nm"] == fallback
    assert actual in result["rejected_stationary_commands_nm"]
    assert result["observations"][:len(before)] == before
    assert result["observations"][-1]["command_nm"] == actual
    assert result["observations"][-1]["requested_command_nm"] == fallback
    assert result["interior_recheck_start_observation"] == len(search.observations)
    assert result["trial_count"] + result["transition_count"] < result["max_trials"]
    for repeat in range(2):
        search.record(command_nm=fallback, requested_command_nm=fallback,
                      outcome="stationary", evidence={"fresh_return": repeat})
        assert search.state == ("searching" if repeat == 0 else "complete")
    assert search.status()["usable_hold_nm"] == fallback
    assert search.status()["auto_apply"] is search.status()["identifies_unique_gravity"] is False


@pytest.mark.parametrize("sign", [-1, 1])
def test_exhausted_interior_holds_publish_no_rejected_return_command(sign):
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=sign * .4,
                               cap_nm=1, recheck_interior_on_conflict=True)
    for value in [.4, .3, .2]:
        search.record(command_nm=sign * value, outcome="stationary", evidence={})
    for value in [.2, .3, .4]:
        search.record(command_nm=sign * value,
                      outcome="negative_motion" if sign > 0 else "positive_motion", evidence={})
    result = search.status(detailed=True)
    assert result["state"] == "incomplete"
    assert result["usable_hold_nm"] is None
    assert result["return_command_nm"] is None
    assert search.return_command_nm is None
    assert len(result["observations"]) == 6
    assert set(result["rejected_stationary_commands_nm"]) == {sign * value for value in [.2, .3, .4]}


@pytest.mark.parametrize("sign", [-1, 1])
def test_default_rejected_frozen_plateau_never_completes(sign):
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=sign * .4, cap_nm=1)
    for value in [.4, .3]:
        search.record(command_nm=sign * value, outcome="stationary", evidence={})
    search.record_transition(command_nm=sign * .29, requested_command_nm=0, evidence={})
    search.record_transition(command_nm=sign * .3, requested_command_nm=sign * .3, evidence={})
    assert search.candidate_nm == sign * .4
    before = list(search.observations)
    search.record(command_nm=sign * .3, requested_command_nm=sign * .4,
                  outcome="stationary", evidence={"ramp_frozen": True})
    result = search.status(detailed=True)
    assert result["state"] == "incomplete"
    assert result["reason"] == "holding_response_not_repeatable_or_monotone"
    assert result["usable_hold_nm"] is None
    assert result["candidate_nm"] == result["return_command_nm"] == sign * .4
    assert result["observations"][:len(before)] == before
    assert result["observations"][-1]["command_nm"] == sign * .3
    assert result["trial_count"] + result["transition_count"] < result["max_trials"]


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("excluded_by", ["censored_check", "resolved_boundary"])
def test_default_terminal_recheck_publishes_no_rejected_return(sign, excluded_by):
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=sign * .3, cap_nm=1)
    search.record(command_nm=sign * .3, outcome="stationary", evidence={})
    search.record_transition(command_nm=sign * .28, requested_command_nm=0, evidence={})
    assert search.return_command_nm == sign * .3
    if excluded_by == "censored_check":
        search.record_transition(command_nm=sign * .3, requested_command_nm=sign * .3, evidence={})
    else:
        search.record(command_nm=sign * .3,
                      outcome="negative_motion" if sign > 0 else "positive_motion", evidence={})
    result = search.status(detailed=True)
    assert result["state"] == "incomplete"
    assert result["usable_hold_nm"] is None
    assert result["return_command_nm"] is None
    assert search.return_command_nm is None
