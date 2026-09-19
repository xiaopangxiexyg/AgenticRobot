from __future__ import annotations

import math

import pytest

from control_station.sim_hil_profile.rebot_b601_rm65_real_connectivity import (
    _sample_acceptance_counts,
    map_source_pose_to_target,
)


def test_map_source_pose_to_target_composes_profile_transforms() -> None:
    reference = (
        (1.0, 0.0, 0.0, 0.25),
        (0.0, 1.0, 0.0, -0.1),
        (0.0, 0.0, 1.0, 0.05),
        (0.0, 0.0, 0.0, 1.0),
    )
    controlled = (
        (0.0, -1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    source_pose = (
        (1.0, 0.0, 0.0, 0.1),
        (0.0, 1.0, 0.0, 0.2),
        (0.0, 0.0, 1.0, 0.3),
        (0.0, 0.0, 0.0, 1.0),
    )

    mapped = map_source_pose_to_target(
        source_pose,
        matrix_target_reference_source_reference=reference,
        matrix_source_controlled_target_controlled=controlled,
    )

    assert mapped[0][3] == pytest.approx(0.35)
    assert mapped[1][3] == pytest.approx(0.1)
    assert mapped[2][3] == pytest.approx(0.35)
    assert mapped[0][0] == pytest.approx(0.0)
    assert mapped[0][1] == pytest.approx(1.0)
    assert mapped[1][0] == pytest.approx(-1.0)
    assert mapped[1][1] == pytest.approx(0.0)


def test_map_source_pose_to_target_rejects_non_rigid_shape() -> None:
    with pytest.raises(ValueError, match="numeric 4x4 matrix"):
        map_source_pose_to_target(
            ((1.0, 0.0, 0.0),),
            matrix_target_reference_source_reference=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            matrix_source_controlled_target_controlled=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )


def test_mapping_is_deterministic_for_identity_profile() -> None:
    identity = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    pose = map_source_pose_to_target(
        identity,
        matrix_target_reference_source_reference=identity,
        matrix_source_controlled_target_controlled=identity,
    )
    assert pose == identity
    assert all(math.isfinite(item) for row in pose for item in row)


def test_connectivity_pass_gate_requires_a_paired_valid_sample() -> None:
    counts = _sample_acceptance_counts(
        [
            {
                "b601": {"accepted": True},
                "rm65": {"hardware_read_success": False},
            },
            {
                "b601": {"accepted": False},
                "rm65": {"hardware_read_success": True},
            },
            {
                "b601": {"accepted": True},
                "rm65": {"hardware_read_success": True},
                "mapped_target_pose": [[1.0] * 4] * 4,
                "mapped_vs_observed_target": {"translation_error_m": 0.0},
            },
        ]
    )
    assert counts == {
        "b601_accepted_sample_count": 2,
        "rm65_successful_sample_count": 2,
        "valid_pose_pair_count": 1,
    }
