from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from robot_motion.ik_candidate_policy import (
    local_step_scale_factors,
    local_step_worsening_limits,
    prefer_tracking_candidate_pool,
)


def test_local_step_scale_factors_preserve_normalization_order() -> None:
    profile = SimpleNamespace(
        local_step_scale_factors=(
            float("nan"),
            -1.0,
            0.5,
            1.5,
            0.5000000001,
            0.25,
        )
    )

    assert local_step_scale_factors(profile) == (1.0, 0.5, 0.25)
    assert local_step_scale_factors(SimpleNamespace(local_step_scale_factors=())) == (
        1.0,
    )


def test_local_step_worsening_limits_preserve_dynamic_policy() -> None:
    profile = SimpleNamespace(
        local_step_max_position_worsen_m=0.001,
        local_step_max_orientation_worsen_deg=1.0,
        local_step_dynamic_worsen_enabled=True,
        local_step_dynamic_worsen_error_start_m=0.05,
        local_step_dynamic_worsen_error_start_deg=15.0,
        local_step_dynamic_worsen_position_fraction=0.10,
        local_step_dynamic_worsen_orientation_fraction=0.10,
        local_step_dynamic_worsen_max_position_m=0.02,
        local_step_dynamic_worsen_max_orientation_deg=8.0,
    )

    limits = local_step_worsening_limits(
        profile=profile,
        before_position_m=0.10,
        before_orientation_deg=20.0,
    )

    assert limits["position_m"] == pytest.approx(0.01)
    assert limits["orientation_deg"] == pytest.approx(2.0)
    assert limits["base_position_m"] == pytest.approx(0.001)
    assert limits["base_orientation_deg"] == pytest.approx(1.0)
    assert limits["enabled"] is True
    assert limits["active"] is True
    assert limits["position_fraction"] == pytest.approx(0.10)
    assert limits["orientation_fraction"] == pytest.approx(0.10)


@dataclass(frozen=True)
class _Candidate:
    metadata: dict[str, object]


def _is_auxiliary_without_progress(metadata: dict[str, object]) -> bool:
    return bool(metadata.get("auxiliary_without_progress", False))


def test_prefer_tracking_candidate_pool_preserves_selection_receipt() -> None:
    auxiliary = _Candidate({"auxiliary_without_progress": True})
    tracking = _Candidate({"auxiliary_without_progress": False})
    evaluated = [auxiliary, tracking]

    pool, receipt = prefer_tracking_candidate_pool(
        evaluated,
        is_auxiliary_without_pose_progress=_is_auxiliary_without_progress,
    )

    assert pool == [tracking]
    assert receipt == {
        "name": "prefer_tracking_over_auxiliary_posture_recovery",
        "accepted_count": 2,
        "selection_count": 1,
        "posture_wrist_auxiliary_deprioritized_count": 1,
        "fallback_to_auxiliary_recovery": False,
    }


def test_prefer_tracking_candidate_pool_falls_back_to_original_list() -> None:
    evaluated = [_Candidate({"auxiliary_without_progress": True})]

    pool, receipt = prefer_tracking_candidate_pool(
        evaluated,
        is_auxiliary_without_pose_progress=_is_auxiliary_without_progress,
    )

    assert pool is evaluated
    assert receipt["selection_count"] == 1
    assert receipt["fallback_to_auxiliary_recovery"] is True
