from __future__ import annotations

import numpy as np
import pytest

from robot_motion.rm65 import (
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    RM65_QREF_ACCELERATION_COMFORT_35_V1,
    RM65_QREF_ACCELERATION_COMFORT_50_V1,
    RM65_QREF_ACCELERATION_COMFORT_75_V1,
    RM65_QREF_ACCELERATION_HARD_V1,
    resolve_rm65_qref_acceleration_profile,
)


@pytest.mark.parametrize(
    ("profile", "expected_scale"),
    (
        (RM65_QREF_ACCELERATION_HARD_V1, 1.0),
        (RM65_QREF_ACCELERATION_COMFORT_75_V1, 0.75),
        (RM65_QREF_ACCELERATION_COMFORT_50_V1, 0.50),
        (RM65_QREF_ACCELERATION_COMFORT_35_V1, 0.35),
    ),
)
def test_qref_acceleration_profiles_scale_without_mutating_hard_envelope(
    profile: str,
    expected_scale: float,
) -> None:
    hard = tuple(RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2)

    scale, resolved = resolve_rm65_qref_acceleration_profile(
        profile,
        hard_max_acceleration_rad_s2=hard,
    )

    assert scale == pytest.approx(expected_scale)
    np.testing.assert_allclose(resolved, np.asarray(hard) * expected_scale)
    assert hard == RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2


def test_qref_acceleration_profile_rejects_unknown_or_invalid_hard_envelope() -> None:
    with pytest.raises(ValueError, match="unsupported RM65 qref acceleration"):
        resolve_rm65_qref_acceleration_profile(
            "unknown",
            hard_max_acceleration_rad_s2=np.ones(6),
        )
    with pytest.raises(ValueError, match="six finite positive"):
        resolve_rm65_qref_acceleration_profile(
            RM65_QREF_ACCELERATION_HARD_V1,
            hard_max_acceleration_rad_s2=(1.0, 2.0),
        )
