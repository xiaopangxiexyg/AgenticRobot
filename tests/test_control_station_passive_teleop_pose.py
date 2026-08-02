from __future__ import annotations

import numpy as np
import pytest

from control_station.sources import (
    AttachablePassiveTeleopComponentSource,
    HumanPoseSourceContract,
    PassiveTeleopComponentClockBridge,
    PassiveTeleopComponentSessionSource,
    PassiveTeleopComponentSourceSpec,
    PassiveTeleopComponentSpec,
    PassiveTeleopPoseSessionSource,
    PassiveTeleopPoseSourceSpec,
    SyntheticHumanPoseSource,
    SyntheticHumanPoseSourceSpec,
    build_passive_teleop_component_session_source,
    build_passive_teleop_pose_session_source,
)
from robot_motion.action import ArmActionSpec, HandActionSpec
from robot_motion.runtime import EffectorCommand, ObservationChannel, TeleopCommand
from teleop_runtime.session import TeleopSession, TeleopSessionConfig


def _pose(x: float) -> np.ndarray:
    value = np.eye(4, dtype=float)
    value[0, 3] = x
    return value


class _BimanualPoseSource:
    name = "virtual-live-tracker"

    def __init__(self, sides: tuple[str, ...] = ("left", "right")) -> None:
        self.sides = tuple(sides)
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def expected_channel_names(self) -> tuple[str, ...]:
        return tuple(f"{side}_wrist_pose" for side in self.sides)

    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        return {
            f"{side}_wrist_pose": ObservationChannel(
                channel_name=f"{side}_wrist_pose",
                capability="pose6d",
                side=side,
                value=_pose(-0.1 if side == "left" else 0.1),
                frame_name="desk",
                timestamp=1.0,
                source_name=self.name,
            )
            for side in self.sides
        }


class _PoseRetargeter:
    def __init__(
        self,
        side: str,
        *,
        command_type: str = "cartesian_pose",
        frame_name: str | None = None,
    ) -> None:
        self.side = side
        self.command_type = command_type
        self.frame_name = frame_name or f"{side}_arm_base"
        self.name = f"virtual-{side}-pose-retargeter"

    def retarget(self, observation, calibration) -> TeleopCommand:
        del observation, calibration
        command = TeleopCommand(timestamp=1.0)
        value = _pose(-0.2 if self.side == "left" else 0.2)
        if self.command_type == "joint_vector":
            value = np.zeros(6, dtype=float)
        command.add_command(
            EffectorCommand(
                effector_name=f"{self.side}_arm",
                side=self.side,
                command_type=self.command_type,
                value=value,
                frame_name=self.frame_name,
            )
        )
        return command


class _HandQposRetargeter:
    name = "virtual-left-hand-qpos-retargeter"

    def __init__(
        self,
        *,
        value: np.ndarray | None = None,
        hand_action_spec: HandActionSpec | None = None,
    ) -> None:
        self.value = (
            np.asarray([0.25] * 6, dtype=float)
            if value is None
            else np.asarray(value, dtype=float)
        )
        self.hand_action_spec = hand_action_spec

    def retarget(self, observation, calibration) -> TeleopCommand:
        del observation, calibration
        command = TeleopCommand(timestamp=1.0)
        metadata = {}
        if self.hand_action_spec is not None:
            metadata["hand_action_spec"] = self.hand_action_spec.to_metadata()
        command.add_command(
            EffectorCommand(
                effector_name="left_hand",
                side="left",
                command_type="joint_vector",
                value=self.value,
                metadata=metadata,
            )
        )
        return command


class _Rig:
    name = "forbidden-rig"


def _session(
    *,
    right_command_type: str = "cartesian_pose",
    right_frame_name: str | None = None,
    include_hand: bool = False,
    rig=None,
    sides: tuple[str, ...] = ("left", "right"),
):
    source = _BimanualPoseSource(sides)
    retargeters = []
    for side in sides:
        retargeters.append(
            _PoseRetargeter(
                side,
                command_type=(
                    right_command_type if side == "right" else "cartesian_pose"
                ),
                frame_name=(
                    right_frame_name
                    if side == "right" and right_frame_name is not None
                    else None
                ),
            )
        )
    if include_hand:
        retargeters.append(_HandQposRetargeter())
    session = TeleopSession(
        TeleopSessionConfig(
            mode="observe-only-pose-test",
            record_enabled=False,
            allow_hardware_after_debug_pass=False,
        ),
        sources=(source,),
        retargeters=tuple(retargeters),
        rig=rig,
    )
    return session, source


def test_observe_only_session_emits_exact_pose_without_dispatch() -> None:
    session, lower_source = _session()
    observed = []
    spec = PassiveTeleopPoseSourceSpec(source_id="live-human-slot")
    assert isinstance(spec, HumanPoseSourceContract)
    source = PassiveTeleopPoseSessionSource(
        session,
        spec,
        passive_subscribers={"audit": lambda emission: observed.append(emission)},
    )
    try:
        receipt = source.start()
        source.set_calibration_ready()
        step = source.step(now=1.0)

        assert receipt.rig_absent and not receipt.authoritative
        assert step.emission_valid
        assert step.emission is not None
        assert not step.command_dispatched
        assert step.session_step.command is not None
        assert not step.session_step.command_dispatched
        assert set(step.session_step.input_sample_refs) == {
            "left_wrist_pose",
            "right_wrist_pose",
        }
        assert set(step.emission.observation_sample_refs) == {
            "left_wrist_pose",
            "right_wrist_pose",
        }
        assert step.emission.observation_sample_set_hash
        assert observed == [step.emission]
        queued = source.ingress.wait_for_post_boundary_emission(
            boundary_at=0.0,
            timeout_sec=0.01,
        )
        assert queued is step.emission
        assert source.session_state.value == "DEBUG_ONLY"
    finally:
        source.close()
    assert lower_source.started and lower_source.closed


def test_observe_only_session_accepts_single_arm_declared_contract() -> None:
    session, lower_source = _session(sides=("right",))
    spec = PassiveTeleopPoseSourceSpec.for_arm(
        "right_arm",
        reference_frame="right_arm_base",
        action_spec_hash="rm65-tool-tcp-pose-v1",
        source_id="single-arm-human-slot",
        source_profile_id="single-arm-virtual-tracker-v1",
    )
    assert spec.effector_names == ("right_arm",)
    assert spec.contract_id == "control_station_passive_teleop_pose_v1"
    source = PassiveTeleopPoseSessionSource(session, spec)
    try:
        source.start()
        source.set_calibration_ready()
        step = source.step(now=1.0)
        assert step.emission_valid
        assert step.emission is not None
        assert tuple(item.effector_name for item in step.emission.merged_commands) == (
            "right_arm",
        )
        assert not step.command_dispatched
    finally:
        source.close()
    assert lower_source.started and lower_source.closed


def test_synthetic_pose_source_keeps_single_arm_identity_typed() -> None:
    spec = SyntheticHumanPoseSourceSpec(
        reference_frame_by_effector=(
            ("right_arm", "right_arm_base"),
        ),
        action_spec_hash_by_effector=(
            ("right_arm", "rm65-tool-tcp-pose-v1"),
        ),
    )
    source = SyntheticHumanPoseSource(spec)
    emission = source.emit({"right_arm": _pose(0.2)}, emitted_at=1.0)
    assert spec.effector_names == ("right_arm",)
    assert spec.contract_id == "control_station_synthetic_human_pose_v1"
    assert tuple(item.effector_name for item in emission.merged_commands) == (
        "right_arm",
    )


def test_non_pose_arm_command_is_rejected_before_passive_fanout() -> None:
    session, _ = _session(right_command_type="joint_vector")
    observed = []
    source = PassiveTeleopPoseSessionSource(
        session,
        PassiveTeleopPoseSourceSpec(),
        passive_subscribers={"audit": lambda emission: observed.append(emission)},
    )
    try:
        source.start()
        source.set_calibration_ready()
        step = source.step(now=1.0)

        assert not step.emission_valid
        assert step.emission is None
        assert "right_arm is not a cartesian_pose" in step.rejection_reason
        assert observed == []
        assert source.ingress.status().observed_count == 0
        assert not step.command_dispatched
    finally:
        source.close()


def test_wrong_pose_reference_frame_is_rejected_before_passive_fanout() -> None:
    session, _ = _session(right_frame_name="desk")
    source = PassiveTeleopPoseSessionSource(
        session,
        PassiveTeleopPoseSourceSpec(),
    )
    try:
        source.start()
        source.set_calibration_ready()
        step = source.step(now=1.0)

        assert not step.emission_valid
        assert step.emission is None
        assert "right_arm frame 'desk'" in step.rejection_reason
        assert source.ingress.status().observed_count == 0
        assert not step.command_dispatched
    finally:
        source.close()


def test_pose_source_rejects_hand_component_instead_of_silently_dropping_it() -> None:
    session, _ = _session(include_hand=True)
    source = PassiveTeleopPoseSessionSource(
        session,
        PassiveTeleopPoseSourceSpec(),
    )
    try:
        source.start()
        source.set_calibration_ready()
        step = source.step(now=1.0)

        assert not step.emission_valid
        assert step.emission is None
        assert "Hand/qpos components require a separate typed component source" in step.rejection_reason
        assert source.ingress.status().observed_count == 0
        assert not step.command_dispatched
    finally:
        source.close()


def test_wrapper_rejects_any_session_with_dispatch_rig_or_existing_observer() -> None:
    session, _ = _session(rig=_Rig())
    with pytest.raises(ValueError, match="rig=None"):
        PassiveTeleopPoseSessionSource(session, PassiveTeleopPoseSourceSpec())

    observed_session, _ = _session()
    observed_session.human_command_observer = lambda emission: None
    with pytest.raises(ValueError, match="exclusive observer"):
        PassiveTeleopPoseSessionSource(
            observed_session,
            PassiveTeleopPoseSourceSpec(),
        )


def test_source_mapping_generation_resets_pose_baseline_before_safety_check() -> None:
    session = TeleopSession(
        TeleopSessionConfig(
            mode="mapping-transition-test",
            record_enabled=False,
            allow_hardware_after_debug_pass=False,
        ),
        sources=(),
        retargeters=(),
    )
    channel = ObservationChannel(
        channel_name="right_wrist_pose",
        capability="pose6d",
        side="right",
        value=_pose(0.1),
        frame_name="right_arm_base",
        timestamp=1.0,
        source_name="vive_pose",
        metadata={"source_mapping_generation": 0},
    )

    session._observe_source_mapping_transitions({channel.channel_name: channel})
    session.safety_monitor.evaluate_channels(
        {channel.channel_name: channel},
        now=1.0,
    )
    assert channel.channel_name not in session.safety_monitor._pose_reset_pending_by_channel

    remapped = channel.copy()
    remapped.metadata["source_mapping_generation"] = 1
    remapped.value = _pose(0.8)
    session._observe_source_mapping_transitions(
        {remapped.channel_name: remapped}
    )

    assert channel.channel_name in session.safety_monitor._pose_reset_pending_by_channel


def test_external_arm_transition_is_detected_before_another_source_poll() -> None:
    session, _ = _session()
    source = PassiveTeleopPoseSessionSource(
        session,
        PassiveTeleopPoseSourceSpec(),
    )
    try:
        source.start()
        source.set_calibration_ready()
        session.arm_hardware()
        with pytest.raises(RuntimeError, match="forbidden state ARMED"):
            source.step(now=1.0)
        assert source.ingress.status().observed_count == 0
    finally:
        source.close()


def test_passive_subscriber_failure_cannot_invalidate_or_block_pose_emission() -> None:
    session, _ = _session()

    def fail(_emission) -> None:
        raise RuntimeError("audit sink unavailable")

    source = PassiveTeleopPoseSessionSource(
        session,
        PassiveTeleopPoseSourceSpec(),
        passive_subscribers={"failing_audit": fail},
    )
    try:
        source.start()
        source.set_calibration_ready()
        step = source.step(now=1.0)

        assert step.emission_valid
        assert step.emission is not None
        assert source.ingress.status().accepted_count == 1
        status = source.fanout.status()
        assert status["subscribers"]["failing_audit"]["error_count"] == 1
        assert not step.command_dispatched
    finally:
        source.close()


def test_device_neutral_builder_preserves_observe_only_boundary() -> None:
    lower_source = _BimanualPoseSource()
    source = build_passive_teleop_pose_session_source(
        PassiveTeleopPoseSourceSpec(source_profile_id="virtual-tracker-test-v1"),
        sources=(lower_source,),
        retargeters=(_PoseRetargeter("left"), _PoseRetargeter("right")),
    )
    try:
        receipt = source.start()
        source.set_calibration_ready()
        step = source.step(now=1.0)

        assert receipt.session_mode == "control-station-passive-pose"
        assert receipt.rig_absent and not receipt.authoritative
        assert step.emission_valid and not step.command_dispatched
        assert source.session_state.value == "DEBUG_ONLY"
    finally:
        source.close()
    assert lower_source.started and lower_source.closed


@pytest.mark.parametrize(
    ("config", "message"),
    (
        (
            TeleopSessionConfig(record_enabled=True, allow_hardware_after_debug_pass=False),
            "forbids session recording",
        ),
        (
            TeleopSessionConfig(record_enabled=False, allow_hardware_after_debug_pass=True),
            "forbids hardware-arm eligibility",
        ),
    ),
)
def test_device_neutral_builder_rejects_active_session_capabilities(
    config: TeleopSessionConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_passive_teleop_pose_session_source(
            PassiveTeleopPoseSourceSpec(),
            sources=(_BimanualPoseSource(),),
            retargeters=(_PoseRetargeter("left"), _PoseRetargeter("right")),
            session_config=config,
        )


def _left_pose_component() -> PassiveTeleopComponentSpec:
    return PassiveTeleopComponentSpec.arm_pose(
        "left_arm_pose",
        "left_arm",
        action_spec_hash="generic-tool-tcp-pose-v1",
        arm_action_spec=ArmActionSpec.tool_tcp(
            "left",
            reference_frame="left_arm_base",
        ),
    )


def _left_hand_component(
    hand_spec: HandActionSpec | None = None,
) -> PassiveTeleopComponentSpec:
    return PassiveTeleopComponentSpec.hand_qpos(
        "left_hand_qpos",
        "left_hand",
        action_spec_hash="generic-left-hand-qpos-v1",
        hand_action_spec=hand_spec or HandActionSpec.inspire_tac("left"),
    )


def test_component_source_accepts_pose_and_hand_qpos_without_dispatch() -> None:
    lower_source = _BimanualPoseSource(sides=("left",))
    spec = PassiveTeleopComponentSourceSpec(
        components=(_left_pose_component(), _left_hand_component()),
        source_profile_id="virtual-pose-hand-device-v1",
    )
    source = build_passive_teleop_component_session_source(
        spec,
        sources=(lower_source,),
        retargeters=(_PoseRetargeter("left"), _HandQposRetargeter()),
    )
    assert isinstance(source, PassiveTeleopComponentSessionSource)
    try:
        receipt = source.start()
        source.set_calibration_ready()
        step = source.step(now=1.0)

        assert receipt.session_mode == "control-station-passive-components"
        assert receipt.rig_absent and not receipt.authoritative
        assert step.emission_valid and step.emission is not None
        assert tuple(
            item.effector_name for item in step.emission.merged_commands
        ) == ("left_arm", "left_hand")
        assert not step.command_dispatched
        assert spec.arm_action_specs_by_side["left"].reference_frame == "left_arm_base"
        assert spec.hand_action_specs_by_side["left"].slot_count == 6
    finally:
        source.close()
    assert lower_source.started and lower_source.closed


def test_attachable_component_source_stays_passive_until_exact_delegate_attaches() -> None:
    lower_source = _BimanualPoseSource(sides=("left",))
    spec = PassiveTeleopComponentSourceSpec(
        components=(_left_pose_component(),),
        source_profile_id="deferred-human-source-v1",
    )
    delegate = build_passive_teleop_component_session_source(
        spec,
        sources=(lower_source,),
        retargeters=(_PoseRetargeter("left"),),
    )
    source = AttachablePassiveTeleopComponentSource(spec)
    try:
        receipt = source.start()
        source.set_calibration_ready()
        waiting = source.step(now=1.0)

        assert receipt.rig_absent and not receipt.authoritative
        assert source.attached is False
        assert waiting.emission is None
        assert waiting.rejection_reason == "human_source_not_attached"
        assert lower_source.started is False

        attached = source.attach(delegate)
        assert attached["status"] == "attached"
        assert attached["command_dispatched"] is False
        assert attached["authoritative"] is False
        assert source.attached is True
        assert lower_source.started is True

        emitted = source.step(now=1.1)
        assert emitted.emission_valid is True
        assert emitted.command_dispatched is False
    finally:
        source.close()
    assert lower_source.closed is True


def test_component_clock_bridge_keeps_session_freshness_on_wall_clock() -> None:
    lower_source = _BimanualPoseSource(sides=("left",))
    spec = PassiveTeleopComponentSourceSpec(
        components=(_left_pose_component(),),
        source_profile_id="clock-bridge-test-v1",
    )
    source = build_passive_teleop_component_session_source(
        spec,
        sources=(lower_source,),
        retargeters=(_PoseRetargeter("left"),),
    )
    bridge = PassiveTeleopComponentClockBridge(
        source,
        wall_clock=lambda: 1.01,
    )
    try:
        bridge.start()
        bridge.set_calibration_ready()

        # The outer tick is deliberately from an unrelated monotonic domain.
        # TeleopSession must receive the source-compatible wall time.
        step = bridge.step(now=9_000.0)

        assert step.session_step.timestamp == pytest.approx(1.01)
        assert step.emission_valid is True
        assert step.session_step.tracker_loss_channels == ()
    finally:
        bridge.close()


def test_component_source_accepts_hand_only_qpos_from_virtual_device() -> None:
    lower_source = _BimanualPoseSource(sides=("left",))
    source = build_passive_teleop_component_session_source(
        PassiveTeleopComponentSourceSpec(
            components=(_left_hand_component(),),
            source_profile_id="virtual-hand-only-device-v1",
        ),
        sources=(lower_source,),
        retargeters=(_HandQposRetargeter(),),
    )
    try:
        source.start()
        source.set_calibration_ready()
        step = source.step(now=1.0)
        assert step.emission_valid and step.emission is not None
        assert tuple(
            item.effector_name for item in step.emission.merged_commands
        ) == ("left_hand",)
        assert not step.command_dispatched
    finally:
        source.close()


def test_component_source_rejects_mismatched_hand_spec_metadata() -> None:
    lower_source = _BimanualPoseSource(sides=("left",))
    source = build_passive_teleop_component_session_source(
        PassiveTeleopComponentSourceSpec(
            components=(_left_hand_component(HandActionSpec.inspire_tac("left")),),
        ),
        sources=(lower_source,),
        retargeters=(
            _HandQposRetargeter(
                hand_action_spec=HandActionSpec.dmtac_gripper("left")
            ),
        ),
    )
    try:
        source.start()
        source.set_calibration_ready()
        step = source.step(now=1.0)
        assert not step.emission_valid
        assert "hand_action_spec does not match" in step.rejection_reason
        assert source.ingress.status().observed_count == 0
    finally:
        source.close()


def test_component_source_enforces_q16_inactive_nan_slots() -> None:
    q16_spec = HandActionSpec.q16("left", active_dof=2)
    invalid = np.zeros(16, dtype=float)
    source = build_passive_teleop_component_session_source(
        PassiveTeleopComponentSourceSpec(
            components=(_left_hand_component(q16_spec),),
        ),
        sources=(_BimanualPoseSource(sides=("left",)),),
        retargeters=(_HandQposRetargeter(value=invalid),),
    )
    try:
        source.start()
        source.set_calibration_ready()
        step = source.step(now=1.0)
        assert not step.emission_valid
        assert "inactive slots must be NaN" in step.rejection_reason
    finally:
        source.close()


def test_component_contract_keeps_arm_qpos_outside_pose_mainline() -> None:
    with pytest.raises(ValueError, match="arm qpos mixing is not a mainline"):
        PassiveTeleopComponentSpec(
            component_id="left_arm_qpos",
            effector_name="left_arm",
            command_type="joint_vector",
            action_spec_hash="future-homogeneous-arm-v1",
            hand_action_spec=HandActionSpec.inspire_tac("left"),
        )
