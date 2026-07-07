from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class DeploymentObservationSource(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def session_contract(self) -> dict[str, Any]:
        ...

    def get_observation(
        self,
        *,
        last_action: Any = None,
        now: float | None = None,
    ) -> Any:
        ...


class ActionPreparer(Protocol):
    def prepare(self, action: Any, observation: Any, *, policy_name: str = "") -> Any:
        ...


class CommandDispatcher(Protocol):
    def dispatch(self, command: Any) -> None:
        ...


@dataclass(frozen=True)
class ClientRuntimeStepResult:
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


class DeploymentClientRuntime:
    """Generic source -> remote policy -> action preparer -> backend dispatcher."""

    def __init__(
        self,
        *,
        source: DeploymentObservationSource,
        policy: Any,
        preparer: ActionPreparer,
        dispatcher: CommandDispatcher,
        policy_name: str = "",
        send_actions: bool = False,
    ) -> None:
        self.source = source
        self.policy = policy
        self.preparer = preparer
        self.dispatcher = dispatcher
        self.policy_name = str(policy_name or getattr(policy, "name", "") or "policy")
        self.send_actions = bool(send_actions)

    def step(self, *, last_action: Any = None, now: float | None = None) -> ClientRuntimeStepResult:
        observation = self.source.get_observation(last_action=last_action, now=now)
        predict = getattr(self.policy, "predict", self.policy)
        raw_action = predict(observation)
        prepared = self.preparer.prepare(raw_action, observation, policy_name=self.policy_name)
        faults = tuple(getattr(prepared, "faults", ()) or ())
        command = getattr(prepared, "command", None)
        sent = False
        if command is not None and not faults and self.send_actions:
            self.dispatcher.dispatch(command)
            sent = True
        return ClientRuntimeStepResult(
            observation=observation,
            raw_action=raw_action,
            prepared_action=prepared,
            command=command,
            sent=sent,
            faults=faults,
            metadata={"policy_name": self.policy_name},
        )

