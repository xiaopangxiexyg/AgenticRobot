from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class ActionPreparer(Protocol):
    """Converts a policy action into a resolved command boundary."""

    def prepare(self, action: Any, observation: Any, *, policy_name: str = "") -> Any:
        ...


class PolicyLike(Protocol):
    def predict(self, observation: Any) -> Any:
        ...


@dataclass(frozen=True)
class DeployRuntimeStepResult:
    observation: Any
    raw_action: Any = None
    prepared_action: Any = None
    command: Any = None
    sent: bool = False
    faults: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "faults", tuple(str(fault) for fault in self.faults))
        object.__setattr__(self, "metadata", dict(self.metadata))


class DeployRuntime:
    """Minimal common single-step runtime glue for sim/real deployment.

    Migration note:
    The full real-time loop currently remains in ``teleop_runtime.deploy``.
    This common class is intentionally small so future sim and real launchers
    can share the same source -> policy -> canonical pipeline -> dispatcher
    shape without moving the existing production loop in one large edit.
    """

    def __init__(
        self,
        *,
        source: Any,
        policy: PolicyLike,
        preparer: ActionPreparer,
        dispatcher: Any,
        policy_name: str = "",
        send_actions: bool = False,
    ) -> None:
        self.source = source
        self.policy = policy
        self.preparer = preparer
        self.dispatcher = dispatcher
        self.policy_name = str(policy_name or getattr(policy, "name", "") or "policy")
        self.send_actions = bool(send_actions)

    def step(self, *, last_action: Any = None, now: float | None = None) -> DeployRuntimeStepResult:
        observation = self.source.get_observation(last_action=last_action, now=now)
        raw_action = self.policy.predict(observation)
        prepared = self.preparer.prepare(raw_action, observation, policy_name=self.policy_name)
        faults = tuple(getattr(prepared, "faults", ()) or ())
        command = getattr(prepared, "command", None)
        sent = False
        if command is not None and not faults and self.send_actions:
            dispatch = getattr(self.dispatcher, "dispatch", None)
            if dispatch is None:
                dispatch = self.dispatcher
            dispatch(command)
            sent = True
        return DeployRuntimeStepResult(
            observation=observation,
            raw_action=raw_action,
            prepared_action=prepared,
            command=command,
            sent=sent,
            faults=faults,
            metadata={"policy_name": self.policy_name},
        )
