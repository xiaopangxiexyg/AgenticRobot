"""Small step-gated orchestrator for one rule-writing task cycle.

The chain owns only operator-visible sequencing.  Robot motion, gripper
feedback, height calibration, rule execution, and Episode commitment remain
owned by their existing runtime components.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum


class RuleWritingTaskChainPhase(str, Enum):
    IDLE = "idle"
    INITIAL_RELEASE_PENDING = "initial_release_pending"
    READY_CLOSE = "ready_close"
    CLOSE_PENDING = "close_pending"
    READY_HEIGHT = "ready_height"
    HEIGHT_PENDING = "height_pending"
    READY_RULE = "ready_rule"
    CONFIGURATION_PENDING = "configuration_pending"
    RULE_PENDING = "rule_pending"
    READY_FINAL_RELEASE = "ready_final_release"
    FINAL_RELEASE_PENDING = "final_release_pending"
    COMPLETE = "complete"


_PHASE_PRESENTATION: dict[RuleWritingTaskChainPhase, tuple[str, str, str]] = {
    RuleWritingTaskChainPhase.IDLE: (
        "0 / 5",
        "任务链未启动",
        "等待右夹爪、安全门禁和轨迹配置就绪。",
    ),
    RuleWritingTaskChainPhase.INITIAL_RELEASE_PENDING: (
        "1 / 5",
        "正在松开右夹爪",
        "等待现有夹爪反馈链确认右夹爪实测松开；左侧保持当前状态。",
    ),
    RuleWritingTaskChainPhase.READY_CLOSE: (
        "2 / 5",
        "等待闭紧右夹爪",
        "右夹爪用于抓笔和书写；左侧保持当前状态。确认工件和工作区安全后继续。",
    ),
    RuleWritingTaskChainPhase.CLOSE_PENDING: (
        "2 / 5",
        "正在闭紧右夹爪",
        "等待现有夹爪反馈链确认右夹爪实测闭紧；左侧不发送闭合命令。",
    ),
    RuleWritingTaskChainPhase.READY_HEIGHT: (
        "3 / 5",
        "等待标定书写高度",
        "将右臂 TCP 手动放到书写高度，确认后继续。",
    ),
    RuleWritingTaskChainPhase.HEIGHT_PENDING: (
        "3 / 5",
        "正在标定书写高度",
        "等待现有高度标定接口读取新鲜 TCP 实测值。",
    ),
    RuleWritingTaskChainPhase.READY_RULE: (
        "4 / 5",
        "等待执行一条规则采集",
        "继续后将从已预验证配置池随机选择一条并自动执行完整 Episode。",
    ),
    RuleWritingTaskChainPhase.CONFIGURATION_PENDING: (
        "4 / 5",
        "正在激活随机配置",
        "等待现有配置接口完成来源读取和验证；此阶段不输出运动。",
    ),
    RuleWritingTaskChainPhase.RULE_PENDING: (
        "4 / 5",
        "规则采集执行中",
        "落笔 replay、rule-based writing 和 relative 提笔 replay 由现有运行时自动执行。",
    ),
    RuleWritingTaskChainPhase.READY_FINAL_RELEASE: (
        "5 / 5",
        "等待最终松开右夹爪",
        "正式 Episode 已完整提交；确认安全后仅松开右夹爪完成闭环。",
    ),
    RuleWritingTaskChainPhase.FINAL_RELEASE_PENDING: (
        "5 / 5",
        "正在最终松开右夹爪",
        "等待现有夹爪反馈链确认右夹爪实测松开；左侧保持当前状态。",
    ),
    RuleWritingTaskChainPhase.COMPLETE: (
        "5 / 5",
        "本条规则采集已完成",
        "再次继续将按调整抓笔周期决定直接采集下一条，或先重新抓笔并标定高度。",
    ),
}


@dataclass
class RuleWritingTaskChain:
    """Track one chain without owning any lower-layer action."""

    phase: RuleWritingTaskChainPhase = RuleWritingTaskChainPhase.IDLE
    cycle_count: int = 0
    grasp_adjustment_interval: int = 10
    episodes_since_grasp_adjustment: int = 0
    grasp_adjustment_initialized: bool = False
    selected_configuration: dict[str, object] = field(default_factory=dict)
    selected_configuration_index: int | None = None
    configuration_prepared: bool = False
    last_error: str = ""
    rule_execution_seen: bool = False
    gripper_action_baseline_id: int | None = None
    gripper_action_id: int | None = None
    rule_start_waiting: bool = False

    @property
    def active(self) -> bool:
        return self.phase not in {
            RuleWritingTaskChainPhase.IDLE,
            RuleWritingTaskChainPhase.COMPLETE,
        }

    @property
    def grasp_adjustment_due(self) -> bool:
        return bool(
            not self.grasp_adjustment_initialized
            or self.episodes_since_grasp_adjustment
            >= self.grasp_adjustment_interval
        )

    def configure_grasp_adjustment_interval(self, value: int) -> None:
        normalized = int(value)
        if normalized < 1:
            raise ValueError("grasp adjustment interval must be at least 1.")
        self.grasp_adjustment_interval = normalized

    def begin_initial_release(self, *, action_id_baseline: int | None) -> None:
        self._require(
            RuleWritingTaskChainPhase.IDLE,
            RuleWritingTaskChainPhase.COMPLETE,
        )
        self.phase = RuleWritingTaskChainPhase.INITIAL_RELEASE_PENDING
        self.grasp_adjustment_initialized = False
        self.rule_execution_seen = False
        self.gripper_action_baseline_id = action_id_baseline
        self.gripper_action_id = None
        self.last_error = ""
        self.rule_start_waiting = False

    def finish_initial_release(self, *, error: str | None = None) -> None:
        self._require(RuleWritingTaskChainPhase.INITIAL_RELEASE_PENDING)
        self.phase = (
            RuleWritingTaskChainPhase.READY_CLOSE
            if not error
            else RuleWritingTaskChainPhase.IDLE
        )
        self.last_error = str(error or "")

    def begin_close(self, *, action_id_baseline: int | None) -> None:
        self._require(RuleWritingTaskChainPhase.READY_CLOSE)
        self.phase = RuleWritingTaskChainPhase.CLOSE_PENDING
        self.gripper_action_baseline_id = action_id_baseline
        self.gripper_action_id = None
        self.last_error = ""
        self.rule_start_waiting = False

    def finish_close(self, *, error: str | None = None) -> None:
        self._require(RuleWritingTaskChainPhase.CLOSE_PENDING)
        self.phase = (
            RuleWritingTaskChainPhase.READY_HEIGHT
            if not error
            else RuleWritingTaskChainPhase.READY_CLOSE
        )
        self.last_error = str(error or "")
        self.rule_start_waiting = False

    def begin_height(self) -> None:
        self._require(RuleWritingTaskChainPhase.READY_HEIGHT)
        self.phase = RuleWritingTaskChainPhase.HEIGHT_PENDING
        self.last_error = ""

    def finish_height(self, *, error: str | None = None) -> None:
        self._require(RuleWritingTaskChainPhase.HEIGHT_PENDING)
        self.phase = (
            RuleWritingTaskChainPhase.READY_RULE
            if not error
            else RuleWritingTaskChainPhase.READY_HEIGHT
        )
        self.last_error = str(error or "")
        if not error:
            self.grasp_adjustment_initialized = True
            self.episodes_since_grasp_adjustment = 0

    def begin_configuration(
        self,
        *,
        index: int,
        selected: Mapping[str, object],
    ) -> None:
        self._require(RuleWritingTaskChainPhase.READY_RULE)
        self.phase = RuleWritingTaskChainPhase.CONFIGURATION_PENDING
        self.selected_configuration_index = int(index)
        self.selected_configuration = dict(selected)
        self.configuration_prepared = False
        self.rule_execution_seen = False
        self.last_error = ""
        self.rule_start_waiting = False

    def begin_rule(self) -> None:
        self._require(RuleWritingTaskChainPhase.CONFIGURATION_PENDING)
        self.phase = RuleWritingTaskChainPhase.RULE_PENDING
        self.configuration_prepared = True
        self.rule_execution_seen = False
        self.last_error = ""
        self.rule_start_waiting = False

    def begin_rule_after_wait(self) -> None:
        """Start the validated rule only after a later explicit gate retry."""

        self._require(RuleWritingTaskChainPhase.READY_RULE)
        if not self.rule_start_waiting:
            raise RuntimeError("rule start is not waiting for a gate retry.")
        self.phase = RuleWritingTaskChainPhase.RULE_PENDING
        self.configuration_prepared = True
        self.rule_execution_seen = False
        self.last_error = ""
        self.rule_start_waiting = False

    def set_prepared_configuration(
        self,
        *,
        index: int,
        selected: Mapping[str, object],
    ) -> None:
        """Mirror a command-inert configuration already activated by runtime."""

        self._require(
            RuleWritingTaskChainPhase.IDLE,
            RuleWritingTaskChainPhase.COMPLETE,
            RuleWritingTaskChainPhase.READY_RULE,
        )
        self.selected_configuration_index = int(index)
        self.selected_configuration = dict(selected)
        self.configuration_prepared = True

    def begin_prepared_rule(
        self,
        *,
        index: int,
        selected: Mapping[str, object],
    ) -> None:
        """Start the runtime-prepared next attempt without rebuilding it.

        A successful rule attempt already advances random source selection and
        prepares the next immutable plan.  Re-submitting the same GUI
        configuration here would reset that selection ordinal to zero.
        """

        self._require(
            RuleWritingTaskChainPhase.COMPLETE,
            RuleWritingTaskChainPhase.READY_RULE,
        )
        self.phase = RuleWritingTaskChainPhase.RULE_PENDING
        self.selected_configuration_index = int(index)
        self.selected_configuration = dict(selected)
        self.configuration_prepared = True
        self.rule_execution_seen = False
        self.last_error = ""
        self.rule_start_waiting = False

    def wait_for_rule_start(self) -> None:
        """Keep validated configuration while the lower start gate is closed."""

        self._require(RuleWritingTaskChainPhase.CONFIGURATION_PENDING)
        self.phase = RuleWritingTaskChainPhase.READY_RULE
        self.configuration_prepared = True
        self.last_error = ""
        self.rule_start_waiting = True

    def wait_for_recovered_rule_start(self) -> None:
        """Resume the same validated rule after device/Home recovery.

        Unlike configuration completion, recovery already owns a validated
        selection.  It may only reopen the existing START gate; it never
        chooses or rebuilds a different trajectory.
        """

        self._require(RuleWritingTaskChainPhase.READY_RULE)
        if not self.selected_configuration:
            raise RuntimeError(
                "recovered rule start requires the previously selected configuration."
            )
        self.last_error = ""
        self.rule_start_waiting = True

    def prepare_device_recovery(self) -> str | None:
        """Freeze an interrupted step at its nearest retryable boundary.

        The returned token describes only a step that had already been
        accepted before the device fault.  Merely waiting at a manual gate
        does not grant automatic continuation.
        """

        phase = self.phase
        operation: str | None = None
        if phase is RuleWritingTaskChainPhase.INITIAL_RELEASE_PENDING:
            self.phase = RuleWritingTaskChainPhase.IDLE
            operation = "initial_release"
        elif phase is RuleWritingTaskChainPhase.CLOSE_PENDING:
            self.phase = RuleWritingTaskChainPhase.READY_CLOSE
            operation = "close"
        elif phase is RuleWritingTaskChainPhase.CONFIGURATION_PENDING:
            # The command-inert configuration worker keeps its ownership.  Its
            # normal callback will move the chain to READY_RULE.
            operation = "rule"
        elif phase is RuleWritingTaskChainPhase.RULE_PENDING:
            self.phase = RuleWritingTaskChainPhase.READY_RULE
            operation = "rule"
        elif phase is RuleWritingTaskChainPhase.FINAL_RELEASE_PENDING:
            self.phase = RuleWritingTaskChainPhase.READY_FINAL_RELEASE
            operation = "final_release"
        elif phase is RuleWritingTaskChainPhase.IDLE and self.last_error:
            operation = "initial_release"
        elif phase is RuleWritingTaskChainPhase.READY_CLOSE and self.last_error:
            operation = "close"
        elif (
            phase is RuleWritingTaskChainPhase.READY_RULE
            and self.selected_configuration
            and (self.last_error or self.rule_start_waiting)
        ):
            operation = "rule"
        elif (
            phase is RuleWritingTaskChainPhase.READY_FINAL_RELEASE
            and self.last_error
        ):
            operation = "final_release"

        if operation is None:
            return None
        self.gripper_action_baseline_id = None
        self.gripper_action_id = None
        self.rule_execution_seen = False
        self.rule_start_waiting = False
        self.last_error = ""
        return operation

    def fail_rule_preparation(self, error: str) -> None:
        self._require(RuleWritingTaskChainPhase.CONFIGURATION_PENDING)
        self.phase = RuleWritingTaskChainPhase.READY_RULE
        self.configuration_prepared = False
        self.last_error = str(error)
        self.rule_start_waiting = False

    def observe_rule_execution(self) -> None:
        self._require(RuleWritingTaskChainPhase.RULE_PENDING)
        self.rule_execution_seen = True

    def bind_gripper_action(self, action_id: int) -> bool:
        """Bind one runtime-owned action receipt to the current gripper phase."""

        self._require(
            RuleWritingTaskChainPhase.INITIAL_RELEASE_PENDING,
            RuleWritingTaskChainPhase.CLOSE_PENDING,
            RuleWritingTaskChainPhase.FINAL_RELEASE_PENDING,
        )
        normalized = int(action_id)
        if normalized < 1:
            return False
        if (
            self.gripper_action_baseline_id is not None
            and normalized <= self.gripper_action_baseline_id
        ):
            return False
        if self.gripper_action_id is None:
            self.gripper_action_id = normalized
        return self.gripper_action_id == normalized

    def finish_rule(self) -> None:
        self._require(RuleWritingTaskChainPhase.RULE_PENDING)
        self.phase = RuleWritingTaskChainPhase.COMPLETE
        self.cycle_count += 1
        self.episodes_since_grasp_adjustment += 1
        self.last_error = ""
        self.rule_start_waiting = False

    def fail_rule(self, error: str) -> None:
        self._require(RuleWritingTaskChainPhase.RULE_PENDING)
        self.phase = RuleWritingTaskChainPhase.READY_RULE
        self.last_error = str(error)
        self.rule_start_waiting = False

    def begin_final_release(self, *, action_id_baseline: int | None) -> None:
        self._require(RuleWritingTaskChainPhase.READY_FINAL_RELEASE)
        self.phase = RuleWritingTaskChainPhase.FINAL_RELEASE_PENDING
        self.gripper_action_baseline_id = action_id_baseline
        self.gripper_action_id = None
        self.last_error = ""
        self.rule_start_waiting = False

    def cancel(self, reason: str) -> None:
        """Return operator sequencing to idle after lower-owned Hold confirms."""

        if not self.active:
            return
        self.phase = RuleWritingTaskChainPhase.IDLE
        self.selected_configuration = {}
        self.selected_configuration_index = None
        self.configuration_prepared = False
        self.rule_execution_seen = False
        self.gripper_action_baseline_id = None
        self.gripper_action_id = None
        self.rule_start_waiting = False
        self.grasp_adjustment_initialized = False
        self.last_error = str(reason)

    def finish_final_release(self, *, error: str | None = None) -> None:
        self._require(RuleWritingTaskChainPhase.FINAL_RELEASE_PENDING)
        if error:
            self.phase = RuleWritingTaskChainPhase.READY_FINAL_RELEASE
            self.last_error = str(error)
            return
        self.phase = RuleWritingTaskChainPhase.COMPLETE
        self.last_error = ""
        self.rule_start_waiting = False

    def presentation(self) -> dict[str, object]:
        step, title, detail = _PHASE_PRESENTATION[self.phase]
        if self.last_error:
            detail = f"上一步未完成：{self.last_error} 请确认安全后重试当前步骤。"
        elif self.rule_start_waiting:
            detail = (
                "配置已验证，等待现有安全门禁放行；门禁就绪后将自动开始规则采集，"
                "不会自动绕过安全检查。"
            )
        elif self.phase is RuleWritingTaskChainPhase.COMPLETE:
            if self.grasp_adjustment_due:
                title = "本条已完成，下一条前调整抓笔"
                detail = (
                    f"已连续完成 {self.episodes_since_grasp_adjustment} 条；"
                    "下一步将依次松开右夹爪、闭紧右夹爪并重新标定高度。"
                )
            else:
                title = "本条已完成，可直接继续"
                detail = (
                    "本周期继续复用当前抓笔和高度；"
                    f"已完成 {self.episodes_since_grasp_adjustment} / "
                    f"{self.grasp_adjustment_interval} 条。"
                )
        return {
            "active": self.active,
            "phase": self.phase.value,
            "step": step,
            "title": title,
            "detail": detail,
            "cycle_count": int(self.cycle_count),
            "grasp_adjustment_interval": int(self.grasp_adjustment_interval),
            "episodes_since_grasp_adjustment": int(
                self.episodes_since_grasp_adjustment
            ),
            "grasp_adjustment_due": bool(self.grasp_adjustment_due),
            "configuration_prepared": bool(self.configuration_prepared),
            "selected_configuration": dict(self.selected_configuration),
            "selected_configuration_index": self.selected_configuration_index,
        }

    def _require(self, *phases: RuleWritingTaskChainPhase) -> None:
        if self.phase not in phases:
            expected = ", ".join(phase.value for phase in phases)
            raise RuntimeError(
                f"task-chain phase {self.phase.value!r} does not allow this "
                f"transition; expected {expected}."
            )


__all__ = ["RuleWritingTaskChain", "RuleWritingTaskChainPhase"]
