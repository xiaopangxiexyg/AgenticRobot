from __future__ import annotations

import numpy as np
import pytest

from robot_motion.action import RobotAction
from robot_motion.action.safety import (
    BackendGuardResult,
    SharedCanonicalSafetyResult,
    SourceValidationResult,
)


def test_source_shared_and_backend_safety_results_are_distinct_types() -> None:
    action = RobotAction.bimanual_ee_pose(
        left_arm=np.asarray([0.3, 0.0, 0.4, 0, 0, 0, 1], dtype=np.float32),
        source="policy",
    )
    source = SourceValidationResult(True)
    shared = SharedCanonicalSafetyResult("selected-action-0", action)
    backend = BackendGuardResult(False, "rejected", ("sdk_not_ready",))

    assert source.accepted
    assert shared.status == "safe"
    assert backend.status == "rejected"
    with pytest.raises(ValueError):
        shared.safe_action.arm_pose_by_side["left"][0] = 1.0


def test_shared_safety_status_is_typed_and_cannot_invent_an_action() -> None:
    hold = SharedCanonicalSafetyResult("selected-action-0", None, status="hold")
    assert hold.safe_action is None
    with pytest.raises(ValueError):
        SharedCanonicalSafetyResult("selected-action-0", None, status="backend_retry")
