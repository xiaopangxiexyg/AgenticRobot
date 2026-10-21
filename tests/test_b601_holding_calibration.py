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


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("explicit", [False, True])
def test_motion_then_censored_zero_can_propose_midpoint_without_a_stable_anchor(sign, explicit):
    seed = sign*.8855391221587121
    search = HoldingForceSearch(HoldingSearchConfig(max_trials=27, movement_rad=.0045),
        seed_nm=seed, cap_nm=1, recheck_interior_on_conflict=explicit)
    search.record(command_nm=seed, requested_command_nm=seed,
        outcome="negative_motion" if sign < 0 else "positive_motion", evidence={})
    assert search.candidate_nm == 0
    search.record_transition(command_nm=0, requested_command_nm=0,
        evidence={"outcome":"censored_settled_outside_point", "net_displacement_rad":-sign*.0015339814})
    if not explicit:
        assert search.state == "incomplete"
        return
    assert search.state == "searching"
    assert search.candidate_nm == pytest.approx(seed/2)
    view = search.status(detailed=True)
    assert view["trial_count"] == view["transition_count"] == 1
    assert view["observed_stationary_range_nm"] is None
    assert view["return_command_nm"] is view["usable_hold_nm"] is None
    assert view["negative_motion_below_nm"] == (seed if sign < 0 else None)
    assert view["positive_motion_above_nm"] == (seed if sign > 0 else None)
    # An intermediate proposal is not a successful observation or completion.
    for _ in range(12):
        if search.state != "searching":
            break
        u = search.candidate_nm
        if abs(u) < .4:
            search.record_transition(command_nm=u, requested_command_nm=u,
                evidence={"outcome":"censored_settled_outside_point"})
        else:
            search.record(command_nm=u, requested_command_nm=u, outcome="stationary", evidence={})
    assert search.state == "complete"
    assert search.status()["auto_apply"] is search.status()["identifies_unique_gravity"] is False
    hold = search.status()["usable_hold_nm"]
    assert sum(r["outcome"] == "stationary" and r["command_nm"] == hold for r in search.observations) >= 2


@pytest.mark.parametrize("budget", [2, 3, 5])
def test_unanchored_midpoint_transitions_keep_original_budget_and_never_complete(budget):
    search = HoldingForceSearch(HoldingSearchConfig(max_trials=budget), seed_nm=-.8, cap_nm=1,
        recheck_interior_on_conflict=True)
    search.record(command_nm=-.8, outcome="negative_motion", evidence={})
    while search.state == "searching":
        u = search.candidate_nm
        search.record_transition(command_nm=u, requested_command_nm=u,
            evidence={"outcome":"censored_settled_outside_point"})
        assert abs(search.candidate_nm) <= 1
    assert search.reason == "holding_trial_budget_exhausted"
    assert len(search.observations)+len(search.transitions) == budget
    assert search.status()["usable_hold_nm"] is None


@pytest.mark.parametrize("sign", [-1, 1])
def test_unanchored_transition_keeps_both_resolved_bounds_and_original_evidence(sign):
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=sign*.8, cap_nm=1,
                               recheck_interior_on_conflict=True)
    search.record(command_nm=sign*.8,
                  outcome="positive_motion" if sign > 0 else "negative_motion", evidence={})
    search.record(command_nm=0, requested_command_nm=0,
                  outcome="negative_motion" if sign > 0 else "positive_motion", evidence={})
    old_rows = [dict(row) for row in search.observations]
    requested = search.candidate_nm
    search.record_transition(command_nm=sign*.2, requested_command_nm=requested,
                             evidence={"outcome": "censored_settled_outside_point"})
    result = search.status(detailed=True)
    assert result["state"] == "searching"
    assert result["candidate_nm"] == pytest.approx(sign*.4)
    assert result["observations"] == old_rows
    assert result["trial_count"] == 2 and result["transition_count"] == 1
    assert result["negative_motion_below_nm"] == (0 if sign > 0 else -.8)
    assert result["positive_motion_above_nm"] == (.8 if sign > 0 else 0)
    assert result["return_command_nm"] is result["usable_hold_nm"] is None
    assert result["observed_stationary_range_nm"] is None
    assert result["transitions"][0]["command_nm"] == sign*.2
    assert result["transitions"][0]["requested_command_nm"] == requested


@pytest.mark.parametrize("sign", [-1, 1])
def test_unanchored_transition_on_the_wrong_side_cannot_invent_an_interval(sign):
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=sign*.8, cap_nm=1,
                               recheck_interior_on_conflict=True)
    search.record(command_nm=sign*.8,
                  outcome="positive_motion" if sign > 0 else "negative_motion", evidence={})
    search.record_transition(command_nm=sign*.9, requested_command_nm=0, evidence={})
    result = search.status(detailed=True)
    assert result["state"] == "incomplete"
    assert result["reason"] == "holding_transition_without_stable_anchor"
    assert result["candidate_nm"] == 0
    assert result["trial_count"] == result["transition_count"] == 1
    assert result["observed_stationary_range_nm"] is None
    assert result["return_command_nm"] is result["usable_hold_nm"] is None


@pytest.mark.parametrize("sign", [-1, 1])
def test_unanchored_censored_narrow_interval_never_certifies_a_hold(sign):
    search = HoldingForceSearch(HoldingSearchConfig(torque_resolution_nm=.025),
                               seed_nm=sign*.01, cap_nm=1,
                               recheck_interior_on_conflict=True)
    search.record(command_nm=sign*.01,
                  outcome="positive_motion" if sign > 0 else "negative_motion", evidence={})
    search.record_transition(command_nm=0, requested_command_nm=search.candidate_nm,
                             evidence={"outcome": "censored_settled_outside_point"})
    result = search.status(detailed=True)
    assert result["state"] == "searching"
    assert result["reason"] == "refine_unverified_hold_between_motion_and_transition"
    assert result["candidate_nm"] == pytest.approx(sign*.005)
    assert result["trial_count"] == result["transition_count"] == 1
    assert result["observed_stationary_range_nm"] is None
    assert result["return_command_nm"] is result["usable_hold_nm"] is None
    assert result["auto_apply"] is result["identifies_unique_gravity"] is False
    while search.state == "searching":
        command = search.candidate_nm
        search.record_transition(command_nm=command, requested_command_nm=command, evidence={})
    exhausted = search.status(detailed=True)
    assert exhausted["reason"] == "holding_trial_budget_exhausted"
    assert exhausted["trial_count"]+exhausted["transition_count"] == search.config.max_trials
    assert exhausted["return_command_nm"] is exhausted["usable_hold_nm"] is None


def test_unanchored_exhausted_budget_cannot_be_revived_by_a_later_transition():
    search = HoldingForceSearch(HoldingSearchConfig(max_trials=2), seed_nm=-.8, cap_nm=1,
                               recheck_interior_on_conflict=True)
    search.record(command_nm=-.8, outcome="negative_motion", evidence={})
    search.record_transition(command_nm=0, requested_command_nm=0, evidence={})
    before = search.status(detailed=True)
    assert before["state"] == "incomplete"
    assert before["reason"] == "holding_trial_budget_exhausted"
    with pytest.raises(ValueError, match="terminal"):
        search.record_transition(command_nm=search.candidate_nm,
                                 requested_command_nm=search.candidate_nm, evidence={})
    assert search.status(detailed=True) == before


@pytest.mark.parametrize("sign", [-1, 1])
def test_unanchored_large_finite_interval_keeps_a_finite_proposal(sign):
    import math

    cap = 1.5e308
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=sign*cap, cap_nm=cap,
                               recheck_interior_on_conflict=True)
    search.record(command_nm=sign*cap,
                  outcome="positive_motion" if sign > 0 else "negative_motion", evidence={})
    search.record_transition(command_nm=sign*1e308, requested_command_nm=0, evidence={})
    result = search.status(detailed=True)
    assert math.isfinite(result["candidate_nm"])
    assert result["state"] == "searching"
    assert result["candidate_nm"] == pytest.approx(sign*1.25e308)
    assert -cap <= result["candidate_nm"] <= cap
    assert result["trial_count"] == result["transition_count"] == 1
    assert result["return_command_nm"] is result["usable_hold_nm"] is None


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("explicit", [False, True])
def test_resolved_large_finite_motion_bounds_keep_a_finite_proposal(sign, explicit):
    import math

    cap = 1.5e308
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=sign*cap, cap_nm=cap,
                               recheck_interior_on_conflict=explicit)
    search.record(command_nm=sign*cap,
                  outcome="positive_motion" if sign > 0 else "negative_motion", evidence={})
    search.record(command_nm=sign*1e308,
                  outcome="negative_motion" if sign > 0 else "positive_motion", evidence={})
    result = search.status(detailed=True)
    assert math.isfinite(result["candidate_nm"])
    assert result["state"] == "searching"
    assert result["candidate_nm"] == pytest.approx(sign*1.25e308)
    assert result["trial_count"] == 2 and result["transition_count"] == 0
    assert result["return_command_nm"] is result["usable_hold_nm"] is None


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("explicit", [False, True])
def test_large_finite_stationary_and_censored_endpoint_keep_a_finite_proposal(sign, explicit):
    import math

    cap = 1.5e308
    search = HoldingForceSearch(HoldingSearchConfig(), seed_nm=sign*cap, cap_nm=cap,
                               recheck_interior_on_conflict=explicit)
    search.record(command_nm=sign*cap, outcome="stationary", evidence={})
    search.record_transition(command_nm=sign*1e308, requested_command_nm=0, evidence={})
    result = search.status(detailed=True)
    assert math.isfinite(result["candidate_nm"])
    assert result["state"] == "searching"
    assert result["candidate_nm"] == pytest.approx(sign*1.25e308)
    assert result["return_command_nm"] == sign*cap
    assert result["usable_hold_nm"] is None


@pytest.mark.parametrize("sign", [-1, 1])
def test_rdsb_m51_same_direction_censored_zero_explores_across_zero(sign):
    seed = sign*.12737109345797037
    search = HoldingForceSearch(HoldingSearchConfig(max_trials=27, movement_rad=.0045),
        seed_nm=seed, cap_nm=1, recheck_interior_on_conflict=True)
    search.record(command_nm=seed, requested_command_nm=seed,
        outcome="positive_motion" if sign > 0 else "negative_motion",
        evidence={"net_displacement_rad":sign*.0046019405126571655})
    search.record_transition(command_nm=0, requested_command_nm=0,
        evidence={"outcome":"censored_plateau_outside_point",
                  "net_displacement_rad":sign*.003067959100008011})
    assert search.state == "searching"
    assert search.candidate_nm == pytest.approx(-seed)
    assert search.status()["observed_stationary_range_nm"] is None
    assert search.return_command_nm is None
    assert len(search.observations) == len(search.transitions) == 1
    # Only the factual prefix above is replayed. This hypothetical static
    # interval lies outside [zero, arrival]; it is not a measured B601 force.
    for _ in range(27):
        if search.state != "searching":
            break
        u = search.candidate_nm
        aligned = sign*u
        outcome = ("stationary" if -.24 <= aligned <= -.18 else
                   "positive_motion" if aligned > -.18 else "negative_motion")
        if sign < 0 and outcome != "stationary":
            outcome = "negative_motion" if outcome == "positive_motion" else "positive_motion"
        search.record(command_nm=u, requested_command_nm=u, outcome=outcome, evidence={})
    assert search.state == "complete", search.status(detailed=True)
    hold = search.status()["usable_hold_nm"]
    assert -.24 <= sign*hold <= -.18
    assert sum(r["outcome"] == "stationary" and r["command_nm"] == hold
               for r in search.observations) >= 2
    assert search.status()["auto_apply"] is search.status()["identifies_unique_gravity"] is False


@pytest.mark.parametrize("sign", [-1, 1])
def test_unanchored_directed_censors_expand_finitely_without_fabricating_bounds(sign):
    search = HoldingForceSearch(HoldingSearchConfig(max_trials=27, movement_rad=.0045),
        seed_nm=sign*.1, cap_nm=.3, recheck_interior_on_conflict=True)
    search.record(command_nm=sign*.1,
        outcome="positive_motion" if sign > 0 else "negative_motion", evidence={})
    while search.state == "searching":
        u = search.candidate_nm
        search.record_transition(command_nm=u, requested_command_nm=u,
            evidence={"net_displacement_rad":sign*.003})
        assert abs(search.candidate_nm) <= .3
    assert search.reason == "holding_authority_insufficient"
    assert [r["command_nm"] for r in search.transitions] == pytest.approx([0, -sign*.1, -sign*.3])
    assert len(search.observations) == 1
    assert search.status()["usable_hold_nm"] is None
    assert search.status()["negative_motion_below_nm" if sign > 0 else "positive_motion_above_nm"] is None


def test_censored_narrow_interval_is_not_a_resolved_force_bracket():
    search = HoldingForceSearch(HoldingSearchConfig(max_trials=12), seed_nm=.03,
        cap_nm=1, recheck_interior_on_conflict=True)
    search.record(command_nm=.03, outcome="positive_motion", evidence={})
    search.record_transition(command_nm=0, requested_command_nm=0, evidence={})
    u = search.candidate_nm
    search.record_transition(command_nm=u, requested_command_nm=u, evidence={})
    assert .03-u < search.config.torque_resolution_nm
    assert search.state == "searching"
    assert search.status()["negative_motion_below_nm"] is None
    assert search.status()["usable_hold_nm"] is None


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("explicit", [False, True])
def test_directed_unanchored_exploration_requires_explicit_opt_in(sign, explicit):
    search = HoldingForceSearch(HoldingSearchConfig(movement_rad=.0045), seed_nm=sign*.2,
                               cap_nm=1, recheck_interior_on_conflict=explicit)
    search.record(command_nm=sign*.2,
                  outcome="positive_motion" if sign > 0 else "negative_motion", evidence={})
    search.record_transition(command_nm=0, requested_command_nm=0,
                             evidence={"net_displacement_rad":sign*.003})
    result = search.status(detailed=True)
    assert result["state"] == ("searching" if explicit else "incomplete")
    assert result["candidate_nm"] == pytest.approx(-sign*.2 if explicit else 0)
    assert result["trial_count"] == result["transition_count"] == 1
    assert result["return_command_nm"] is result["usable_hold_nm"] is None
    assert result["observed_stationary_range_nm"] is None


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("progress_kind", ["below_threshold", "at_threshold", "opposing"])
def test_directed_unanchored_exploration_respects_progress_threshold_and_direction(sign, progress_kind):
    config = HoldingSearchConfig(movement_rad=.0045)
    search = HoldingForceSearch(config, seed_nm=sign*.2, cap_nm=1,
                               recheck_interior_on_conflict=True)
    search.record(command_nm=sign*.2,
                  outcome="positive_motion" if sign > 0 else "negative_motion", evidence={})
    progress = sign*config.movement_rad*({"below_threshold":.49, "at_threshold":.5, "opposing":-1}[progress_kind])
    search.record_transition(command_nm=0, requested_command_nm=0,
                             evidence={"net_displacement_rad":progress})
    result = search.status(detailed=True)
    expected = -sign*.2 if progress_kind == "at_threshold" else sign*.1
    assert result["state"] == "searching"
    assert result["candidate_nm"] == pytest.approx(expected)
    assert len(result["observations"]) == len(result["transitions"]) == 1
    assert result["negative_motion_below_nm" if sign > 0 else "positive_motion_above_nm"] is None
    assert result["return_command_nm"] is result["usable_hold_nm"] is None


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("progress_sign", [-1, 1])
def test_directed_unanchored_progress_cannot_escape_two_resolved_motion_bounds(sign, progress_sign):
    search = HoldingForceSearch(HoldingSearchConfig(movement_rad=.0045), seed_nm=sign*.8,
                               cap_nm=1, recheck_interior_on_conflict=True)
    search.record(command_nm=sign*.8,
                  outcome="positive_motion" if sign > 0 else "negative_motion", evidence={})
    search.record(command_nm=0, requested_command_nm=0,
                  outcome="negative_motion" if sign > 0 else "positive_motion", evidence={})
    old = [dict(row) for row in search.observations]
    search.record_transition(command_nm=sign*.2, requested_command_nm=search.candidate_nm,
                             evidence={"net_displacement_rad":progress_sign*.003})
    result = search.status(detailed=True)
    assert result["state"] == "searching"
    assert result["candidate_nm"] == pytest.approx(sign*.4)
    assert result["observations"] == old
    assert result["trial_count"] == 2 and result["transition_count"] == 1
    assert result["return_command_nm"] is result["usable_hold_nm"] is None


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("max_trials", [2, 3])
def test_directed_unanchored_expansion_cannot_renew_the_original_trial_budget(sign, max_trials):
    search = HoldingForceSearch(HoldingSearchConfig(max_trials=max_trials, movement_rad=.0045),
                               seed_nm=sign*.1, cap_nm=1, recheck_interior_on_conflict=True)
    search.record(command_nm=sign*.1,
                  outcome="positive_motion" if sign > 0 else "negative_motion", evidence={})
    while search.state == "searching":
        command = search.candidate_nm
        search.record_transition(command_nm=command, requested_command_nm=command,
                                 evidence={"net_displacement_rad":sign*.003})
    result = search.status(detailed=True)
    assert result["state"] == "incomplete" and result["reason"] == "holding_trial_budget_exhausted"
    assert result["trial_count"]+result["transition_count"] == max_trials
    assert result["trial_count"] == 1
    assert result["return_command_nm"] is result["usable_hold_nm"] is None
    with pytest.raises(ValueError, match="terminal"):
        search.record_transition(command_nm=search.candidate_nm,
                                 requested_command_nm=search.candidate_nm, evidence={"net_displacement_rad":sign*.003})
    assert search.status(detailed=True) == result


@pytest.mark.parametrize("sign", [-1, 1])
def test_directed_unanchored_large_finite_commands_are_clamped_before_publication(sign):
    import math

    cap = 1.5e308
    search = HoldingForceSearch(HoldingSearchConfig(max_trials=6, movement_rad=.0045),
                               seed_nm=sign*cap, cap_nm=cap, recheck_interior_on_conflict=True)
    search.record(command_nm=sign*cap,
                  outcome="positive_motion" if sign > 0 else "negative_motion", evidence={})
    search.record_transition(command_nm=-sign*1e308, requested_command_nm=0,
                             evidence={"net_displacement_rad":sign*.003})
    assert search.state == "searching"
    assert math.isfinite(search.candidate_nm) and search.candidate_nm == -sign*cap
    search.record_transition(command_nm=search.candidate_nm, requested_command_nm=search.candidate_nm,
                             evidence={"net_displacement_rad":sign*.003})
    result = search.status(detailed=True)
    assert result["state"] == "incomplete" and result["reason"] == "holding_authority_insufficient"
    assert math.isfinite(result["candidate_nm"]) and result["candidate_nm"] == -sign*cap
    assert result["trial_count"] == 1 and result["transition_count"] == 2
    assert result["observed_stationary_range_nm"] is None
    assert result["return_command_nm"] is result["usable_hold_nm"] is None
