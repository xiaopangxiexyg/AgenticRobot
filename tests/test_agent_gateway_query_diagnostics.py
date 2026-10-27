"""The inline projection must retain decisions without copying solver internals."""
from copy import deepcopy
import json

import pytest

from agent_gateway.profiles.mujoco.object_goal_query import _candidate_diagnostic

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_dense_endpoint_summary_keeps_residuals_coverage_and_original_diagnostics():
    attempts = [dict(position_error_m=.1000012345678901 + i * .001,
        rotation_error_rad=.2000012345678901 + i * .01,
        seed_index=i, iterations=100, search_method="iterative", joint_vector=[1.] * 7)
        for i in range(9)]
    details = dict(endpoint_attempts=[], endpoint_check_reasons=[],
        endpoint_search_diagnostics=dict(attempts=attempts, seed_count=8, attempted_seed_count=9,
            deadline_exceeded=False, translation_tolerance_m=.003, rotation_tolerance_rad=.017453292519943295),
        motion_quality_limits=dict(maximum_proximal_joint_travel_deg=180.))
    original = deepcopy(details)
    rows = [_candidate_diagnostic(dict(candidate_index=i, reason_code="safe_tcp_endpoint_search_exhausted",
        wall_sec=.1, details=details), [0., 0., .3, 0., 0., 0., 1.]) for i in range(64)]
    assert details == original
    assert len(json.dumps(rows, separators=(",", ":")).encode()) < 60_000
    for row in rows:
        assert row["reason_code"] == "safe_tcp_endpoint_search_exhausted"
        assert row["details"]["attempted_seed_count"] == 9
        assert row["details"]["endpoint_error_pairs_m_rad"] == [
            [a["position_error_m"], a["rotation_error_rad"]] for a in attempts]
        assert "ranking_cost" not in row and "trajectory_hash" not in row
    quality = _candidate_diagnostic(dict(reason_code="safe_tcp_motion_quality_exceeded",
        details={**details, "endpoint_check_reasons": ["tcp_envelope_blocked", "tcp_envelope_blocked"]}),
        [0., 0., .3, 0., 0., 0., 1.])
    assert quality["details"]["motion_quality_limits"] == details["motion_quality_limits"]
    assert quality["details"]["endpoint_check_reasons"] == ["tcp_envelope_blocked"]


def test_success_and_other_failure_evidence_remain_exact():
    for item in [dict(candidate_index=0, reason_code="path_found", ranking_cost=[0., 1., .5],
            trajectory_hash="a" * 64, phase_timing_sec={"ik": .1}),
            dict(candidate_index=1, reason_code="safe_tcp_path_search_exhausted",
                details={"endpoint_check_reasons": ["collision"], "path_search_skipped": False})]:
        before = deepcopy(item)
        row = _candidate_diagnostic(item, [0., 0., .3, 0., 0., 0., 1.])
        assert item == before
        assert {key: row[key] for key in item} == item


@pytest.mark.parametrize("normalized", [False, True])
def test_motion_profile_payload_preserves_legacy_values_and_explicit_mode(normalized):
    from agent_gateway.profiles.mujoco.motion_deployment import MujocoAgentTcpPoseMotionProfile

    profile = MujocoAgentTcpPoseMotionProfile("ref", "ref-spec", "graph", "tool", "calibration",
        normalize_ik_objective=normalized)
    payload = profile.to_payload()
    assert payload["contact_policy_hash"] is None
    assert isinstance(payload["workspace_bounds"], tuple)
    assert payload["workspace_bounds"] == profile.workspace_bounds
    if normalized:
        assert payload["normalize_ik_objective"] is True
    else:
        assert "normalize_ik_objective" not in payload
    restored = MujocoAgentTcpPoseMotionProfile(**json.loads(json.dumps(payload)))
    assert restored.normalize_ik_objective is normalized
    assert restored.semantic_content_hash == profile.semantic_content_hash
