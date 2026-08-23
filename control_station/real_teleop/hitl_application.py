"""Pure Station coordination for the concrete RealTeleop HITL binding."""

from __future__ import annotations

import argparse
import queue
import time
from typing import Any, Callable, Mapping

from control_station.hitl import (
    HITL_SWITCH_FAILED,
    HITL_SWITCH_HOLD_REQUESTED,
    HITL_SWITCH_WAITING_HUMAN_HOME,
    HITL_SWITCH_WAITING_POLICY_EMISSION,
    HITL_SWITCH_WAITING_POLICY_RESET,
    HITL_SWITCH_WAITING_TARGET_DISPATCH,
    ExclusiveHITLSourceIntent,
    ExclusiveHITLTransitionState,
    begin_exclusive_hitl_transition,
    complete_exclusive_hitl_transition,
    fail_exclusive_hitl_transition,
    observe_exclusive_hitl_hold,
    observe_exclusive_hitl_human_home,
    observe_exclusive_hitl_policy_emission,
    observe_exclusive_hitl_policy_reset,
)
from dataset_contracts.canonical_encoding import canonical_content_hash


HITL_HOME_CAPTURE_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.real_teleop_hitl_home_capture.v1"
)
HITL_POLICY_MODE_NORMAL = "normal"
HITL_POLICY_MODE_RTC = "rtc"
HITL_POLICY_MODES = (HITL_POLICY_MODE_NORMAL, HITL_POLICY_MODE_RTC)


class ExclusiveHITLRealTeleopExtension:
    """Coordinate lower HOLD/Home/reset effects without owning dispatch."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        selector_factory: Callable[..., Any],
        normal_policy_source_builder: Callable[..., tuple[object, object | None]],
        normal_source_config: object | None,
    ) -> None:
        self.args = args
        self.policy_mode = str(
            getattr(args, "hitl_policy_mode", HITL_POLICY_MODE_NORMAL)
        ).strip().lower()
        self.initial_source = str(
            getattr(args, "hitl_initial_source", "policy")
        ).strip().lower()
        if self.policy_mode not in HITL_POLICY_MODES:
            raise ValueError("HITL policy mode must be 'normal' or 'rtc'.")
        if self.initial_source not in {"human", "policy"}:
            raise ValueError("HITL initial source must be 'human' or 'policy'.")
        if not callable(selector_factory):
            raise TypeError("selector_factory must be callable.")
        if not callable(normal_policy_source_builder):
            raise TypeError("normal_policy_source_builder must be callable.")
        self._selector_factory = selector_factory
        self._normal_policy_source_builder = normal_policy_source_builder
        self._normal_source_config = normal_source_config
        artifact_path = getattr(args, "hitl_policy_artifact_json", None)
        if (
            self.policy_mode == HITL_POLICY_MODE_RTC
            and artifact_path is not None
            and str(artifact_path).strip()
        ):
            raise ValueError(
                "RTC HITL consumes its committed emission port; "
                "--hitl-policy-artifact-json is only for normal mode."
            )
        self.state = ExclusiveHITLTransitionState(
            active_source=self.initial_source
        )
        self._components: object | None = None
        self._clutch_input: Any | None = None
        self._dataset_recorder: object | None = None
        self._episode_recorder: object | None = None
        self._control_ui: object | None = None
        self._recording_active: Callable[[], bool] | None = None
        self._pause_recording: Callable[..., bool] | None = None
        self._selector: Any | None = None
        self._observation_capture: object | None = None
        self._intent_sequence = 0
        self._home_requested = False
        self._started = False
        self._closed = False
        injected_rtc_queue = getattr(
            args,
            "_hitl_rtc_committed_emissions",
            None,
        )
        if injected_rtc_queue is not None and not isinstance(
            injected_rtc_queue,
            queue.Queue,
        ):
            raise TypeError("_hitl_rtc_committed_emissions must be queue.Queue.")
        self.rtc_committed_emissions: queue.Queue[Any] | None = (
            injected_rtc_queue
            if injected_rtc_queue is not None
            else queue.Queue(maxsize=8)
            if self.policy_mode == HITL_POLICY_MODE_RTC
            else None
        )

    @property
    def selector(self) -> Any:
        if self._selector is None:
            raise RuntimeError("HITL runtime is not bound.")
        return self._selector

    def attach_components(
        self,
        components: Any,
    ) -> Any:
        session = components.session
        if session.command_override_provider is not None:
            raise RuntimeError(
                "Exclusive HITL cannot share RealTeleop with another command override."
            )
        if session.human_command_selection_provider is not None:
            raise RuntimeError(
                "Exclusive HITL cannot share RealTeleop with another source selector."
            )
        session.human_command_selection_provider = self._select_command
        self._components = components
        return components

    def _build_normal_policy_source(
        self,
        *,
        components: Any,
        dataset_recorder: object | None,
        dataset_recorder_provider: Callable[[], object | None],
        camera_sources: Mapping[str, Any],
    ) -> object:
        source, capture = self._normal_policy_source_builder(
            args=self.args,
            config=self._normal_source_config,
            components=components,
            dataset_recorder=dataset_recorder,
            dataset_recorder_provider=dataset_recorder_provider,
            camera_sources=camera_sources,
        )
        self._observation_capture = capture
        return source

    def bind_runtime(
        self,
        *,
        components: Any,
        clutch_input: Any,
        camera_sources: Mapping[str, Any],
        dataset_recorder: object | None,
        dataset_recorder_provider: Callable[[], object | None],
        episode_recorder: object,
        control_ui: object | None,
        recording_active: Callable[[], bool],
        pause_recording: Callable[..., bool],
    ) -> None:
        if components is not self._components:
            raise RuntimeError("RealTeleop did not retain HITL-attached components.")
        if self._selector is not None:
            raise RuntimeError("HITL runtime is already bound.")
        normal_source = (
            self._build_normal_policy_source(
                components=components,
                dataset_recorder=dataset_recorder,
                dataset_recorder_provider=dataset_recorder_provider,
                camera_sources=camera_sources,
            )
            if self.policy_mode == HITL_POLICY_MODE_NORMAL
            else None
        )
        self._selector = self._selector_factory(
            policy_mode=self.policy_mode,
            initial_source=self.initial_source,
            normal_policy_source=normal_source,
            rtc_committed_emissions=self.rtc_committed_emissions,
            runtime_session_id=f"real-teleop-hitl:{time.time_ns()}",
            control_domain_id="real-teleop",
        )
        self._clutch_input = clutch_input
        self._dataset_recorder = dataset_recorder
        self._episode_recorder = episode_recorder
        self._control_ui = control_ui
        self._recording_active = recording_active
        self._pause_recording = pause_recording

    def start(self, *, components: Any) -> None:
        if components is not self._components:
            raise RuntimeError("RealTeleop did not retain HITL-attached components.")
        self.selector.start()
        self._started = True
        if (
            self.policy_mode == HITL_POLICY_MODE_RTC
            and self.initial_source == "policy"
        ):
            if not components.session.hold_hardware(
                "RTC HITL is committed-emission no-action; initial Policy is held"
            ):
                raise RuntimeError("lower Rig HOLD failed for RTC no-action startup.")
        self._publish_message(
            "已启用 exclusive HITL："
            f"initial={self.initial_source}, mode={self.policy_mode}."
        )

    def _select_command(self, observation, human_candidate, now):
        return self.selector(observation, human_candidate, now)

    def _publish_message(self, message: str) -> None:
        append = getattr(self._control_ui, "append_message", None)
        if callable(append):
            append("HITL", str(message))

    def _record_event(
        self,
        kind: str,
        *,
        now: float,
        payload: Mapping[str, Any],
    ) -> None:
        recorder = self._episode_recorder
        if recorder is None or not bool(getattr(recorder, "is_recording", False)):
            return
        record = getattr(recorder, "record_event", None)
        if callable(record):
            record(kind, payload=dict(payload), timestamp=float(now))

    def _pause_capture(self, now: float, *, reason: str) -> None:
        if (
            self._recording_active is not None
            and self._pause_recording is not None
            and self._recording_active()
        ):
            self._pause_recording(float(now), reason=reason)

    def _new_intent(self, source: str) -> ExclusiveHITLSourceIntent:
        intent = ExclusiveHITLSourceIntent(
            requested_source=source,
            requested_at=time.monotonic(),
            clock_domain_id="local_monotonic",
            input_source="real_teleop.operator_input",
            input_event=f"select_{source}",
            sequence=self._intent_sequence,
        )
        self._intent_sequence += 1
        return intent

    def _begin_request(self, source: str, *, now: float) -> None:
        if self.state.phase == HITL_SWITCH_FAILED:
            self._publish_message(
                "上一次切换失败，保持停臂；请处理错误后重启当前 HITL runtime。"
            )
            return
        if self.state.phase != "idle":
            self._publish_message("已有 HITL 切换正在进行，本次按键已忽略。")
            return
        intent = self._new_intent(source)
        next_state = begin_exclusive_hitl_transition(self.state, intent)
        if next_state is self.state:
            return
        self.state = next_state
        self._home_requested = False
        assert self.state.switch_id is not None
        self._pause_capture(
            now,
            reason=f"hitl_switch:{self.state.switch_id}:{source}",
        )
        self._record_event(
            "hitl_switch_requested",
            now=now,
            payload={
                "switch_id": self.state.switch_id,
                "intent_id": intent.intent_id,
                "intent_hash": intent.semantic_content_hash,
                "previous_source": self.state.active_source,
                "target_source": source,
                "policy_mode": self.policy_mode,
            },
        )
        if source == "human":
            self._clutch_input.request_hold(reason="hitl_policy_to_human")
            self._publish_message(
                "已请求人工接管：先进入既有 Hold，再执行 guarded runtime Home。"
            )
        else:
            self._begin_policy_return()

    def _begin_policy_return(self) -> None:
        assert self._components is not None
        assert self.state.pending_intent is not None
        hold_ok = self._components.session.hold_hardware(
            "exclusive HITL human-to-policy activation"
        )
        if not hold_ok:
            raise RuntimeError("lower Rig HOLD failed before policy activation.")
        barrier = time.monotonic()
        self.state = observe_exclusive_hitl_hold(
            self.state,
            established_at=barrier,
            evidence_refs=(
                f"real-teleop-session-hold:{self.state.switch_id}",
            ),
        )
        assert self.state.phase == HITL_SWITCH_WAITING_POLICY_RESET
        self.selector.reset_policy_for_activation(
            barrier_at=barrier,
            switch_id=self.state.switch_id,
            intent_id=self.state.pending_intent.intent_id,
            intent_hash=self.state.pending_intent.semantic_content_hash,
        )
        self.state = observe_exclusive_hitl_policy_reset(
            self.state,
            reset_at=barrier,
        )
        self._publish_message("已进入 Policy reset；等待 reset 后首个新鲜 emission。")

    @staticmethod
    def _session_state_name(components: object) -> str:
        state = components.session.state_machine.state
        return str(getattr(state, "value", state))

    def _advance_human_takeover(self) -> None:
        assert self._components is not None
        assert self._clutch_input is not None
        if self.state.phase == HITL_SWITCH_HOLD_REQUESTED:
            if self._session_state_name(self._components) != "DEBUG_ONLY":
                return
            barrier = time.monotonic()
            self.state = observe_exclusive_hitl_hold(
                self.state,
                established_at=barrier,
                evidence_refs=(
                    f"real-teleop-session-hold:{self.state.switch_id}",
                ),
            )
        if self.state.phase != HITL_SWITCH_WAITING_HUMAN_HOME:
            return
        if not self._home_requested:
            request_home = getattr(
                self._clutch_input,
                "request_runtime_home_for_hitl",
                None,
            )
            if not callable(request_home):
                raise RuntimeError("RealTeleop input lacks guarded HITL Home request.")
            self._home_requested = bool(request_home())
            return
        if self._session_state_name(self._components) not in {"ARMED", "RUNNING"}:
            return
        calibration = self._components.session.calibration
        calibration_payload = (
            {} if calibration is None else dict(calibration.to_dict())
        )
        captured_at = time.monotonic()
        home_hash = canonical_content_hash(
            {
                "switch_id": self.state.switch_id,
                "captured_at": captured_at,
                "calibration": calibration_payload,
            },
            domain=HITL_HOME_CAPTURE_HASH_DOMAIN_V1,
        )
        self.state = observe_exclusive_hitl_human_home(
            self.state,
            home_capture_id=f"real-teleop-hitl-home:{home_hash}",
            home_capture_hash=home_hash,
            captured_at=captured_at,
        )
        assert self.state.pending_intent is not None
        self.selector.select_human(
            switch_id=self.state.switch_id,
            intent_id=self.state.pending_intent.intent_id,
            intent_hash=self.state.pending_intent.semantic_content_hash,
        )
        self._publish_message("runtime Home 已完成；等待首个人工命令通过安全链。")

    def _advance_policy_return(self) -> None:
        if self.state.phase != HITL_SWITCH_WAITING_POLICY_EMISSION:
            return
        emission = self.selector.cached_emission
        if emission is None:
            return
        self.state = observe_exclusive_hitl_policy_emission(
            self.state,
            emission_id=emission.emission_id,
            emission_hash=emission.semantic_content_hash,
            emitted_at=emission.emitted_at,
        )
        if self.policy_mode == HITL_POLICY_MODE_RTC:
            raise RuntimeError(
                "RTC committed emission observed, but the approved RTC profile "
                "is no-action and cannot be returned to motion."
            )
        self.selector.commit_policy_selection()
        assert self._components is not None
        self._components.session.arm_hardware()
        self._publish_message("新鲜 Policy emission 已就绪；等待首个策略命令安全下发。")

    def _fail_transition(self, error: BaseException, *, now: float) -> None:
        if self.state.phase not in {"idle", HITL_SWITCH_FAILED}:
            self.state = fail_exclusive_hitl_transition(self.state, error)
        self._pause_capture(now, reason="hitl_switch_failed")
        if self._components is not None:
            self._components.session.hold_hardware(
                f"exclusive HITL switch failed: {type(error).__name__}: {error}"
            )
        self._publish_message(
            "HITL 切换失败，保持停臂/暂停数采："
            f"{type(error).__name__}: {error}"
        )

    def after_control_input_poll(self, *, now: float) -> None:
        if not self._started or self._clutch_input is None:
            return
        try:
            requests = tuple(self._clutch_input.consume_hitl_source_requests())
            if requests:
                self._begin_request(str(requests[0]), now=now)
                if len(requests) > 1:
                    self._publish_message("同一 control cut 只接收首个 HITL 按键。")
            if self.state.target_source == "human":
                self._advance_human_takeover()
            elif self.state.target_source == "policy":
                self._advance_policy_return()
        except Exception as exc:  # noqa: BLE001 - transition must remain held.
            self._fail_transition(exc, now=now)

    def after_session_step(self, *, step_result: object, now: float) -> None:
        if self.state.phase != HITL_SWITCH_WAITING_TARGET_DISPATCH:
            return
        command = getattr(step_result, "command", None)
        dispatched = bool(getattr(step_result, "command_dispatched", False))
        source = (
            None
            if command is None
            else command.metadata.get("hitl_exclusive_source")
        )
        if not dispatched or source != self.state.target_source:
            return
        completed_switch_id = self.state.switch_id
        target_source = self.state.target_source
        self._record_event(
            "hitl_switch_completed",
            now=now,
            payload={
                "switch_id": completed_switch_id,
                "target_source": target_source,
                "command_metadata": {
                    key: value
                    for key, value in command.metadata.items()
                    if key.startswith("hitl_") or key.startswith("policy_")
                },
            },
        )
        self.state = complete_exclusive_hitl_transition(self.state)
        self._home_requested = False
        self._publish_message(
            f"HITL 切换完成：{target_source}；数采按既有稳定帧规则恢复。"
        )

    def close_before_session(
        self,
        *,
        components: Any,
        terminal_error: BaseException | None,
    ) -> None:
        del terminal_error
        if self._closed:
            return
        try:
            if self._started:
                components.session.hold_hardware("exclusive HITL runtime shutdown")
        finally:
            if self._selector is not None:
                self._selector.close()
            self._started = False
            self._closed = True


def build_exclusive_hitl_real_teleop_extension(
    args: argparse.Namespace,
    *,
    selector_factory: Callable[..., Any],
    normal_policy_source_builder: Callable[..., tuple[object, object | None]],
    normal_source_config: object | None,
) -> ExclusiveHITLRealTeleopExtension | None:
    if not bool(getattr(args, "hitl_policy_control", False)):
        return None
    return ExclusiveHITLRealTeleopExtension(
        args,
        selector_factory=selector_factory,
        normal_policy_source_builder=normal_policy_source_builder,
        normal_source_config=normal_source_config,
    )


__all__ = [
    "ExclusiveHITLRealTeleopExtension",
    "build_exclusive_hitl_real_teleop_extension",
]
