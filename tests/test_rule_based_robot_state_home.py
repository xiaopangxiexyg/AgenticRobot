from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from teleop_runtime.real_teleop.arm_state_sync import (
    capture_runtime_arm_qpos_snapshot,
)


class _StateBackend:
    def __init__(self, qpos) -> None:
        self._qpos = qpos
        self.reads = 0

    def refresh_state_from_hardware(self):
        self.reads += 1
        return {"current_qpos": self._qpos}


def test_rule_runtime_home_reads_only_current_arm_qpos_without_tracker_channels() -> None:
    left = _StateBackend(np.arange(6, dtype=float))
    right = _StateBackend(np.arange(6, dtype=float) + 10.0)
    with ThreadPoolExecutor(max_workers=2) as executor:
        components = SimpleNamespace(
            active_sides=("left", "right"),
            left_state_backend=left,
            right_state_backend=right,
            arm_state_read_executor=executor,
        )
        snapshot = capture_runtime_arm_qpos_snapshot(components)

    np.testing.assert_allclose(snapshot["left"], np.arange(6, dtype=float))
    np.testing.assert_allclose(snapshot["right"], np.arange(6, dtype=float) + 10.0)
    assert left.reads == 1
    assert right.reads == 1




def test_rule_runtime_home_rejects_missing_or_nonfinite_arm_state() -> None:
    components = SimpleNamespace(
        active_sides=("left", "right"),
        left_state_backend=_StateBackend(np.zeros(6)),
        right_state_backend=_StateBackend(np.asarray([0, 0, 0, 0, 0, np.nan])),
        arm_state_read_executor=None,
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        capture_runtime_arm_qpos_snapshot(components)
