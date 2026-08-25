from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.action import BIMANUAL_EE_POSE_V1, RobotAction
from robot_motion.action.sources import CallableActionSource
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM
from robot_motion.pose_frame_profiles import (
    default_rm65_workspace_near_boundary_collection_pose_frame_contract,
)
from robot_motion.replay import (
    ReplayTrajectoryGroup,
    RuleBasedActionSegment,
    RuleBasedCaptureSelection,
    RuleBasedCollectionPlan,
    RuleBasedCustomAction,
    RuleBasedInitialize,
    RuleBasedReplay,
    RuleBasedSegmentStopCondition,
)
from teleop_runtime.command_override import TeleopCommandOverride
from teleop_runtime.real_teleop.rule_based_runtime import (
    RuleBasedRealTeleopConfig,
    RuleBasedRealTeleopDeployment,
    RuleBasedRealTeleopExtension,
    preflight_rule_based_real_teleop_deployment,
)
from teleop_runtime.real_teleop.runtime_extension import (
    RealTeleopDatasetSampleAdmission,
    runtime_extension_dataset_sample_admission,
)


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.5) -> np.ndarray:
    return np.asarray([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _matrix(pose: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = pose[:3]
    return matrix


def _group() -> ReplayTrajectoryGroup:
    actions = np.zeros((2, ACTION_DIM), dtype=np.float32)
    for side in ("left", "right"):
        actions[0, ACTION_ARM_POSE_SLICES[side]] = _pose(0.0)
        actions[1, ACTION_ARM_POSE_SLICES[side]] = _pose(0.02)
    qpos = np.stack((np.zeros(6), np.full(6, 0.1)), axis=0)
    return ReplayTrajectoryGroup(
        key="source",
        label="source",
        source_path="fixture",
        source_episode_index=0,
        start_index=0,
        end_index=2,
        frame_count=2,
        action_hz=10.0,
        actions=actions,
        arm_joint_targets={"left": qpos, "right": qpos},
        arm_observation_actual_tcp_pose={
            "left": np.stack((_pose(0.0), _pose(0.01))),
            "right": np.stack((_pose(0.0), _pose(0.01))),
        },
    )


def _deployment() -> RuleBasedRealTeleopDeployment:
    contract = default_rm65_workspace_near_boundary_collection_pose_frame_contract(
        ("left", "right")
    )
    assert contract is not None
    return RuleBasedRealTeleopDeployment(
        active_sides=("left", "right"),
        collection_pose_frame_contract=contract,
        tcp_pose_envelope_profile="fixture-envelope.json",
    )


class _Poller:
    def __init__(self) -> None:
        self.qpos = {"left": np.zeros(6), "right": np.zeros(6)}
        self.tcp = {"left": np.eye(4), "right": np.eye(4)}

    def readiness_by_side(self, *, required_sides, sample_timestamp):
        del sample_timestamp
        return {side: None for side in required_sides}

    def latest_at_or_before(self, now):
        return {
            side: {
                "actual_qpos": self.qpos[side].copy(),
                "actual_tcp_pose": self.tcp[side].copy(),
            }
            for side in ("left", "right")
        }


class _Recorder:
    def __init__(self, poller: _Poller) -> None:
        self.arm_state_poller = poller
        self.is_recording = False
        self.is_paused = False

    @property
    def can_record_samples(self) -> bool:
        return self.is_recording and not self.is_paused


class _Session:
    def __init__(self) -> None:
        self.command_override_provider = None
        self.human_command_selection_provider = None
        self.hold_reasons: list[str] = []

    def hold_hardware(self, reason: str) -> bool:
        self.hold_reasons.append(reason)
        return True


class _Clutch:
    def __init__(self) -> None:
        self.reasons: list[str] = []

    def request_hold(self, *, reason: str) -> None:
        self.reasons.append(reason)


def _components(session: _Session):
    profile = SimpleNamespace(enabled_sides=("left", "right"))
    target_filter = SimpleNamespace(apply_to_target=True, profile=profile)
    return SimpleNamespace(
        session=session,
        active_sides=("left", "right"),
        collision_guard_requested=True,
        dual_arm_retargeter=SimpleNamespace(
            pre_ik_target_safety_filter=target_filter
        ),
    )


def _bind(
    extension: RuleBasedRealTeleopExtension,
    recorder: _Recorder,
) -> tuple[_Session, _Clutch]:
    session = _Session()
    components = _components(session)
    clutch = _Clutch()
    extension.attach_components(components)
    extension.bind_runtime(
        components=components,
        clutch_input=clutch,
        camera_sources={},
        dataset_recorder=recorder,
        dataset_recorder_provider=lambda: recorder,
        episode_recorder=SimpleNamespace(),
        control_ui=None,
        recording_active=lambda: recorder.is_recording,
        pause_recording=lambda *_args, **_kwargs: True,
    )
    extension.start(components=components)
    return session, clutch


def test_deployment_preflight_requires_existing_safety_and_collection() -> None:
    contract = _deployment().collection_pose_frame_contract
    args = SimpleNamespace(
        active_sides=("left", "right"),
        arm_control_mode="position",
        disable_dual_arm_collision_avoidance=False,
        tcp_pose_envelope_active=True,
        tcp_pose_envelope_profile="envelope.json",
        position_force_guard=True,
        position_force_guard_source="tool",
        no_collection=False,
        position_admittance_active=False,
        position_admittance_rotation_active=False,
        position_admittance_combined_active=False,
        pose_frame_contract=contract,
    )

    deployment = preflight_rule_based_real_teleop_deployment(args)
    assert deployment.collection_pose_frame_contract is contract

    args.position_force_guard = False
    with pytest.raises(ValueError, match="force guard"):
        preflight_rule_based_real_teleop_deployment(args)


def test_real_plan_rejects_partial_stream_selection() -> None:
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="partial",
                action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                capture=RuleBasedCaptureSelection(stream_ids=("action.left",)),
            ),
        )
    )

    with pytest.raises(ValueError, match="formal samples are atomic"):
        RuleBasedRealTeleopExtension(
            deployment=_deployment(),
            plan=plan,
            trajectory_groups={},
        )


def test_runtime_waits_for_formal_episode_and_holds_after_completion() -> None:
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedInitialize(step_id="init", group_key="source"),
            RuleBasedCustomAction(
                step_id="collect",
                action=RobotAction.bimanual_qpos(
                    left_arm=np.full(6, 0.2),
                    right_arm=np.full(6, 0.2),
                    source="rule",
                ),
                capture=RuleBasedCaptureSelection(enabled=True),
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(
            action_hz=10.0,
            settle_consecutive_samples=2,
        ),
    )
    poller = _Poller()
    recorder = _Recorder(poller)
    session, clutch = _bind(extension, recorder)

    waiting = session.command_override_provider(SimpleNamespace(), 10.0)
    assert isinstance(waiting, TeleopCommandOverride)
    assert waiting.command is None
    assert extension.status().execution_begun is False

    recorder.is_recording = True
    init = session.command_override_provider(SimpleNamespace(), 10.1)
    assert init.command.commands["left_arm"].command_type == "joint_vector"
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=10.1,
    )
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=10.2,
    )

    collect = session.command_override_provider(SimpleNamespace(), 10.21)
    assert collect.command.metadata["rule_based_step_id"] == "collect"
    admission = extension.dataset_sample_admission(
        step_result=SimpleNamespace(command_dispatched=True),
        now=10.21,
    )
    assert admission.allowed is True
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=10.21,
    )
    completed = session.command_override_provider(SimpleNamespace(), 10.32)
    assert completed.command is None
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=False),
        now=10.32,
    )

    assert extension.status().completed is True
    assert session.hold_reasons == ["rule-based plan completed"]
    assert clutch.reasons == ["rule_based_runtime_hold"]


def test_runtime_fails_closed_if_writer_pauses_after_motion_begins() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
    )
    recorder = _Recorder(_Poller())
    recorder.is_recording = True
    session, _clutch = _bind(extension, recorder)
    assert session.command_override_provider(SimpleNamespace(), 15.0).command is not None

    recorder.is_paused = True
    with pytest.raises(RuntimeError, match="lost formal collection readiness"):
        session.command_override_provider(SimpleNamespace(), 15.01)


def test_attachment_refuses_inactive_final_tcp_envelope() -> None:
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(
                RuleBasedCustomAction(
                    step_id="collect",
                    action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
                ),
            )
        ),
        trajectory_groups={},
    )
    components = _components(_Session())
    components.dual_arm_retargeter.pre_ik_target_safety_filter.apply_to_target = False

    with pytest.raises(RuntimeError, match="Active final TCP envelope"):
        extension.attach_components(components)


def test_single_arm_canonical_pose_uses_checked_runtime_contract() -> None:
    action = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={"right": _pose(0.4, -0.1, 0.3)},
        active_effectors=("right_arm",),
        frame_by_effector={"right_arm": "canonical_desk"},
        controlled_frame_by_effector={"right_arm": "tool_tcp"},
        source="canonical_rule",
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=RuleBasedCollectionPlan(
            steps=(RuleBasedCustomAction(step_id="pose", action=action),)
        ),
        trajectory_groups={},
    )
    poller = _Poller()
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch = _bind(extension, recorder)

    override = session.command_override_provider(SimpleNamespace(), 20.0)
    expected = extension.runtime_pose_frame_contract.canonical_desk_pose_vector_to_storage(
        action.arm_pose_by_side["right"],
        side="right",
    )

    assert override.command.commands["right_arm"].frame_name == "right_arm_base"
    np.testing.assert_allclose(
        override.command.commands["right_arm"].metadata["target_xyzquat"],
        expected,
    )


def test_terminal_runtime_segment_publishes_measured_relative_boundary() -> None:
    source = CallableActionSource(
        lambda _observation: RobotAction.bimanual_qpos(
            left_arm=np.zeros(6),
            right_arm=np.zeros(6),
            source="bounded_rule",
        )
    )
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedActionSegment(
                step_id="collect",
                action_source=source,
                stop_condition=RuleBasedSegmentStopCondition(max_steps=1),
            ),
            RuleBasedReplay(
                step_id="relative",
                group_key="source",
                start_index=1,
                end_index=2,
                relative_to="collect",
                source_start_index=0,
            ),
        )
    )
    extension = RuleBasedRealTeleopExtension(
        deployment=_deployment(),
        plan=plan,
        trajectory_groups={"source": _group()},
        config=RuleBasedRealTeleopConfig(settle_consecutive_samples=1),
    )
    poller = _Poller()
    poller.tcp["left"][:3, 3] = [0.5, 0.0, 0.5]
    poller.tcp["right"][:3, 3] = [0.7, 0.0, 0.5]
    recorder = _Recorder(poller)
    recorder.is_recording = True
    session, _clutch = _bind(extension, recorder)

    terminal = session.command_override_provider(SimpleNamespace(), 30.0)
    assert terminal.command is not None
    extension.after_session_step(
        step_result=SimpleNamespace(command_dispatched=True),
        now=30.0,
    )
    relative = session.command_override_provider(SimpleNamespace(), 30.01)

    assert extension.status().published_boundaries == ("collect",)
    assert relative.command.metadata["rule_based_step_id"] == "relative"
    left_target = relative.command.commands["left_arm"].metadata["target_xyzquat"]
    assert left_target[0] == pytest.approx(0.52)


def test_formal_sample_gate_defaults_open_and_rejects_bad_hook_type() -> None:
    default = runtime_extension_dataset_sample_admission(
        None,
        step_result=SimpleNamespace(),
        now=1.0,
    )
    assert default == RealTeleopDatasetSampleAdmission()

    bad = SimpleNamespace(dataset_sample_admission=lambda **_kwargs: True)
    with pytest.raises(TypeError, match="RealTeleopDatasetSampleAdmission"):
        runtime_extension_dataset_sample_admission(
            bad,
            step_result=SimpleNamespace(),
            now=1.0,
        )
