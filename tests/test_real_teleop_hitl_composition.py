from __future__ import annotations

import numpy as np

from robot_motion.action import BIMANUAL_QPOS_V1, RobotAction
from teleop_runtime.real_teleop.single_source_composition import (
    ExclusiveSingleSourceComposer,
)


def test_exclusive_single_source_composition_is_identity_and_records_lineage() -> None:
    action = RobotAction(
        action_space=BIMANUAL_QPOS_V1,
        timestamp=10.0,
        arm_qpos_by_side={"right": np.arange(6, dtype=np.float32)},
        active_effectors=("right_arm",),
        source="policy",
    )
    composer = ExclusiveSingleSourceComposer()
    result = composer.compose(
        action,
        runtime_session_id="session:hitl",
        control_domain_id="robot:hitl",
        producer_id="policy:fixture",
        producer_epoch=1,
        policy_output_id="policy-output:1",
        produced_at=9.9,
        produced_clock_domain_id="policy.monotonic",
        available_at=10.0,
        available_clock_domain_id="control.monotonic",
        effective_from=10.0,
        effective_until=10.2,
        effective_clock_domain_id="control.monotonic",
        target_effective_at=10.1,
        context_ids=("switch:intent:1",),
    )

    np.testing.assert_array_equal(
        result.action.arm_qpos_by_side["right"],
        action.arm_qpos_by_side["right"],
    )
    assert result.composition_input_id.startswith("exclusive-hitl-composition-input:")
    assert result.composition_evaluation_id.startswith("composition-evaluation:")
    assert result.composed_candidate_hash


def test_exclusive_single_source_composition_binds_schema_for_activation() -> None:
    composer = ExclusiveSingleSourceComposer()
    qpos = RobotAction(
        action_space=BIMANUAL_QPOS_V1,
        arm_qpos_by_side={"right": np.zeros(6, dtype=np.float32)},
        active_effectors=("right_arm",),
        source="policy",
    )
    kwargs = dict(
        runtime_session_id="session:hitl",
        control_domain_id="robot:hitl",
        producer_id="policy:fixture",
        producer_epoch=1,
        policy_output_id="policy-output:1",
        produced_at=9.9,
        produced_clock_domain_id="policy.monotonic",
        available_at=10.0,
        available_clock_domain_id="control.monotonic",
        effective_from=10.0,
        effective_until=10.2,
        effective_clock_domain_id="control.monotonic",
        target_effective_at=10.1,
    )
    composer.compose(qpos, **kwargs)

    incompatible = RobotAction(
        action_space=BIMANUAL_QPOS_V1,
        arm_qpos_by_side={"left": np.zeros(6, dtype=np.float32)},
        active_effectors=("left_arm",),
        source="policy",
    )
    try:
        composer.compose(incompatible, **kwargs)
    except ValueError as exc:
        assert "outside the runtime action schema" in str(exc)
    else:
        raise AssertionError("activation schema change must fail closed")
