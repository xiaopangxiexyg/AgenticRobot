from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

from robot_motion.runtime import EffectorCommand, TeleopCommand
from teleop_runtime.deploy import action_execution
from teleop_runtime.deploy.action_execution import PreparedActionExecutor
from teleop_runtime.position_control import PositionDeployExecutorFault


class _PositionExecutor:
    def __init__(
        self,
        *,
        source_index: int = 17,
        error: Exception | None = None,
        fault_stage: str = "",
    ) -> None:
        self.source_index = source_index
        self.error = error
        self.fault_stage = fault_stage
        self.submissions: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.clear_count = 0

    def submit(self, *args: object, **kwargs: object) -> int:
        self.submissions.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.source_index

    def snapshot(self) -> dict[str, object]:
        return {"fault_stage": self.fault_stage}

    def clear(self) -> None:
        self.clear_count += 1


def _command(*kinds: str) -> TeleopCommand:
    command = TeleopCommand(timestamp=12.5, metadata={"source": "test"})
    if "arm" in kinds:
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=[0.0] * 6,
            )
        )
    if "hand" in kinds:
        command.add_command(
            EffectorCommand(
                effector_name="left_hand",
                side="left",
                command_type="joint_vector",
                value=[0.5],
            )
        )
    return command


def _canonical(command: TeleopCommand | None, *, filtered: object = None):
    return SimpleNamespace(command=command, filtered=filtered)


def _executor(
    *,
    position_executor: _PositionExecutor | None = None,
    send_actions: bool = False,
    position_worker_binding: object | None = None,
    owner_submit: Mock | None = None,
) -> PreparedActionExecutor:
    return PreparedActionExecutor(
        position_deploy_executor=position_executor,
        send_actions=send_actions,
        arm_backends={"left": object()},
        hand_backends={"left": object()},
        active_sides=("left",),
        position_worker_binding=position_worker_binding,
        submit_legacy_owner_state=owner_submit or Mock(),
    )


def test_no_command_preserves_rejected_no_dispatch_result() -> None:
    execution = _executor(send_actions=True).execute(
        _canonical(None),
        object(),
        policy_name="test",
    )

    assert execution.accepted is False
    assert execution.mode == "no_command"
    assert execution.dispatch_attempted is False
    assert execution.dispatch_completed is False
    assert execution.asynchronous is False
    assert execution.position_source_index is None


def test_direct_dry_run_preserves_timing_without_dispatch() -> None:
    phase_timing: dict[str, object] = {}
    command = _command("arm", "hand")

    with patch.object(action_execution, "dispatch_teleop_command") as dispatch:
        execution = _executor(send_actions=False).execute(
            _canonical(command),
            object(),
            policy_name="test",
            phase_timing=phase_timing,
        )

    dispatch.assert_not_called()
    assert execution.mode == "dry_run"
    assert execution.accepted is True
    assert execution.dispatch_attempted is False
    assert execution.dispatch_completed is False
    assert float(phase_timing["direct_dispatch_ms"]) >= 0.0


def test_direct_active_path_delegates_exact_command_and_backends() -> None:
    command = _command("arm", "hand")
    canonical = _canonical(command)
    executor = _executor(send_actions=True)

    with patch.object(action_execution, "dispatch_teleop_command") as dispatch:
        execution = executor.execute(canonical, object(), policy_name="test")

    dispatch.assert_called_once_with(
        command,
        arm_backends=executor._arm_backends,
        hand_backends=executor._hand_backends,
    )
    assert execution.mode == "direct"
    assert execution.dispatch_attempted is True
    assert execution.dispatch_completed is True


@pytest.mark.parametrize(
    ("send_actions", "expected_mode"),
    ((False, "position_dry_run_queue"), (True, "position_async_queue")),
)
def test_position_arm_path_forwards_existing_submit_contract(
    send_actions: bool,
    expected_mode: str,
) -> None:
    position = _PositionExecutor(source_index=23)
    observation = object()
    lookahead_observation = object()
    second_observation = object()
    third_observation = object()
    canonical = _canonical(_command("arm", "hand"), filtered="primary-filtered")
    lookahead = _canonical(_command("arm", "hand"), filtered="lookahead-filtered")
    second = _canonical(_command("arm"), filtered="second-filtered")
    third = _canonical(_command("arm"), filtered="third-filtered")
    phase_timing: dict[str, object] = {}

    execution = _executor(
        position_executor=position,
        send_actions=send_actions,
    ).execute(
        canonical,
        observation,
        policy_name="test-policy",
        lookahead_canonical_action=lookahead,
        lookahead_observation=lookahead_observation,
        second_lookahead_canonical_action=second,
        second_lookahead_observation=second_observation,
        third_lookahead_canonical_action=third,
        third_lookahead_observation=third_observation,
        source_plan_cut=None,
        phase_timing=phase_timing,
    )

    assert execution.mode == expected_mode
    assert execution.accepted is True
    assert execution.asynchronous is True
    assert execution.position_source_index == 23
    assert execution.dispatch_attempted is False
    assert execution.dispatch_completed is False
    assert len(position.submissions) == 1
    args, kwargs = position.submissions[0]
    assert args == ("primary-filtered", observation)
    assert kwargs["policy_name"] == "test-policy"
    assert set(kwargs["discrete_command"].commands) == {"left_hand"}
    assert kwargs["lookahead_filtered"] == "lookahead-filtered"
    assert kwargs["lookahead_observation"] is lookahead_observation
    assert set(kwargs["lookahead_discrete_command"].commands) == {"left_hand"}
    assert kwargs["second_lookahead_filtered"] == "second-filtered"
    assert kwargs["second_lookahead_observation"] is second_observation
    assert kwargs["third_lookahead_filtered"] == "third-filtered"
    assert kwargs["third_lookahead_observation"] is third_observation
    assert kwargs["execution_work_item"] is None
    assert kwargs["lookahead_execution_work_item"] is None
    assert kwargs["source_plan_cut"] is None
    assert float(phase_timing["position_submit_ms"]) >= 0.0


def test_position_safety_fault_is_the_only_executor_fault_converted_to_result() -> None:
    position = _PositionExecutor(
        error=PositionDeployExecutorFault("blocked"),
        fault_stage="15hz_keyframe_safety",
    )

    execution = _executor(position_executor=position).execute(
        _canonical(_command("arm"), filtered="filtered"),
        object(),
        policy_name="test",
    )

    assert execution.accepted is False
    assert execution.mode == "position_safety_fault_latched"

    position.fault_stage = "reference_worker"
    with pytest.raises(PositionDeployExecutorFault, match="blocked"):
        _executor(position_executor=position).execute(
            _canonical(_command("arm"), filtered="filtered"),
            object(),
            policy_name="test",
        )


def test_position_hand_only_active_path_rechecks_arms_and_clears_on_failure() -> None:
    position = _PositionExecutor()
    executor = _executor(position_executor=position, send_actions=True)
    canonical = _canonical(_command("hand"), filtered="filtered")

    with (
        patch.object(
            action_execution,
            "require_arm_backends_ready_for_position_dispatch",
        ) as require_ready,
        patch.object(
            action_execution,
            "dispatch_teleop_command",
            side_effect=RuntimeError("hand dispatch failed"),
        ) as dispatch,
    ):
        with pytest.raises(RuntimeError, match="hand dispatch failed"):
            executor.execute(canonical, object(), policy_name="test")

    require_ready.assert_called_once_with(
        executor._arm_backends,
        require_background_target=False,
        stage="pre_direct_hand_only_dispatch",
        required_sides=("left",),
    )
    assert set(dispatch.call_args.args[0].commands) == {"left_hand"}
    assert dispatch.call_args.kwargs == {
        "arm_backends": {},
        "hand_backends": executor._hand_backends,
    }
    assert position.clear_count == 1


def test_legacy_position_submission_preserves_owner_and_lineage_handoff() -> None:
    position = _PositionExecutor()
    context = Mock()
    context.create_reference_segment_work_item.side_effect = (
        lambda **kwargs: (
            kwargs["source_decision_id"],
            kwargs["source_reference_segment_id"],
        )
    )
    binding = SimpleNamespace(context=context)
    owner_submit = Mock()
    executor = _executor(
        position_executor=position,
        position_worker_binding=binding,
        owner_submit=owner_submit,
    )
    primary = _canonical(_command("arm"), filtered="primary")
    lookahead = _canonical(_command("arm"), filtered="lookahead")

    executor.execute(
        primary,
        object(),
        policy_name="test",
        source_decision_id="decision-7",
        source_reference_segment_id="segment-9",
        lookahead_canonical_action=lookahead,
        lookahead_observation=object(),
    )

    owner_submit.assert_called_once_with(
        lifecycle_state="running",
        execution_gate="ready",
        reason="legacy_reference_work_admission",
        source_event_ids=("decision-7", "segment-9"),
    )
    assert context.create_reference_segment_work_item.call_args_list == [
        call(
            source_decision_id="decision-7",
            source_reference_segment_id="segment-9",
        ),
        call(
            source_decision_id="decision-7",
            source_reference_segment_id="segment-9:lookahead",
        ),
    ]
    _, kwargs = position.submissions[0]
    assert kwargs["execution_work_item"] == ("decision-7", "segment-9")
    assert kwargs["lookahead_execution_work_item"] == (
        "decision-7",
        "segment-9:lookahead",
    )


def test_legacy_position_submission_still_requires_exact_source_lineage() -> None:
    binding = SimpleNamespace(context=Mock())
    executor = _executor(
        position_executor=_PositionExecutor(),
        position_worker_binding=binding,
    )

    with pytest.raises(RuntimeError, match="requires exact source lineage"):
        executor.execute(
            _canonical(_command("arm"), filtered="filtered"),
            object(),
            policy_name="test",
        )
