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
        "按“继续任务链”后先松开夹爪。",
    ),
    RuleWritingTaskChainPhase.INITIAL_RELEASE_PENDING: (
        "1 / 5",
        "正在松开夹爪",
        "等待现有夹爪反馈链确认实测松开。",
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
        "等待最终松开夹爪",
        "正式 Episode 已完整提交；确认安全后继续完成闭环。",
    ),
    RuleWritingTaskChainPhase.FINAL_RELEASE_PENDING: (
        "5 / 5",
        "正在最终松开夹爪",
        "等待现有夹爪反馈链确认实测松开。",
    ),
    RuleWritingTaskChainPhase.COMPLETE: (
        "5 / 5",
        "任务链已完成",
        "本轮闭环完成；再次继续将开始下一轮。",
    ),
}


@dataclass
class RuleWritingTaskChain:
    """Track one chain without owning any lower-layer action."""

    phase: RuleWritingTaskChainPhase = RuleWritingTaskChainPhase.IDLE
    cycle_count: int = 0
    selected_configuration: dict[str, object] = field(default_factory=dict)
    selected_configuration_index: int | None = None
    last_error: str = ""
    rule_execution_seen: bool = False
    gripper_pending_seen: bool = False
    gripper_status_baseline: str = ""
    rule_start_waiting: bool = False

    @property
    def active(self) -> bool:
        return self.phase not in {
            RuleWritingTaskChainPhase.IDLE,
            RuleWritingTaskChainPhase.COMPLETE,
        }

    def begin_initial_release(self, *, status_baseline: str) -> None:
        self._require(
            RuleWritingTaskChainPhase.IDLE,
            RuleWritingTaskChainPhase.COMPLETE,
        )
        self.phase = RuleWritingTaskChainPhase.INITIAL_RELEASE_PENDING
        self.selected_configuration = {}
        self.selected_configuration_index = None
        self.rule_execution_seen = False
        self.gripper_pending_seen = False
        self.gripper_status_baseline = str(status_baseline)
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

    def begin_close(self, *, status_baseline: str) -> None:
        self._require(RuleWritingTaskChainPhase.READY_CLOSE)
        self.phase = RuleWritingTaskChainPhase.CLOSE_PENDING
        self.gripper_pending_seen = False
        self.gripper_status_baseline = str(status_baseline)
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
        self.rule_execution_seen = False
        self.last_error = ""
        self.rule_start_waiting = False

    def begin_rule(self) -> None:
        self._require(RuleWritingTaskChainPhase.CONFIGURATION_PENDING)
        self.phase = RuleWritingTaskChainPhase.RULE_PENDING
        self.rule_execution_seen = False
        self.last_error = ""
        self.rule_start_waiting = False

    def begin_rule_after_wait(self) -> None:
        """Start the validated rule only after a later explicit gate retry."""

        self._require(RuleWritingTaskChainPhase.READY_RULE)
        if not self.rule_start_waiting:
            raise RuntimeError("rule start is not waiting for a gate retry.")
        self.phase = RuleWritingTaskChainPhase.RULE_PENDING
        self.rule_execution_seen = False
        self.last_error = ""
        self.rule_start_waiting = False

    def wait_for_rule_start(self) -> None:
        """Keep validated configuration while the lower start gate is closed."""

        self._require(RuleWritingTaskChainPhase.CONFIGURATION_PENDING)
        self.phase = RuleWritingTaskChainPhase.READY_RULE
        self.last_error = ""
        self.rule_start_waiting = True

    def fail_rule_preparation(self, error: str) -> None:
        self._require(RuleWritingTaskChainPhase.CONFIGURATION_PENDING)
        self.phase = RuleWritingTaskChainPhase.READY_RULE
        self.last_error = str(error)
        self.rule_start_waiting = False

    def observe_rule_execution(self) -> None:
        self._require(RuleWritingTaskChainPhase.RULE_PENDING)
        self.rule_execution_seen = True

    def observe_gripper_pending(self) -> None:
        self._require(
            RuleWritingTaskChainPhase.INITIAL_RELEASE_PENDING,
            RuleWritingTaskChainPhase.CLOSE_PENDING,
            RuleWritingTaskChainPhase.FINAL_RELEASE_PENDING,
        )
        self.gripper_pending_seen = True

    def finish_rule(self) -> None:
        self._require(RuleWritingTaskChainPhase.RULE_PENDING)
        self.phase = RuleWritingTaskChainPhase.READY_FINAL_RELEASE
        self.last_error = ""
        self.rule_start_waiting = False

    def fail_rule(self, error: str) -> None:
        self._require(RuleWritingTaskChainPhase.RULE_PENDING)
        self.phase = RuleWritingTaskChainPhase.READY_RULE
        self.last_error = str(error)
        self.rule_start_waiting = False

    def begin_final_release(self, *, status_baseline: str) -> None:
        self._require(RuleWritingTaskChainPhase.READY_FINAL_RELEASE)
        self.phase = RuleWritingTaskChainPhase.FINAL_RELEASE_PENDING
        self.gripper_pending_seen = False
        self.gripper_status_baseline = str(status_baseline)
        self.last_error = ""
        self.rule_start_waiting = False

    def finish_final_release(self, *, error: str | None = None) -> None:
        self._require(RuleWritingTaskChainPhase.FINAL_RELEASE_PENDING)
        if error:
            self.phase = RuleWritingTaskChainPhase.READY_FINAL_RELEASE
            self.last_error = str(error)
            return
        self.phase = RuleWritingTaskChainPhase.COMPLETE
        self.cycle_count += 1
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
        return {
            "active": self.active,
            "phase": self.phase.value,
            "step": step,
            "title": title,
            "detail": detail,
            "cycle_count": int(self.cycle_count),
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
