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
