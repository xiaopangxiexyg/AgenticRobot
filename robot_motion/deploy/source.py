from __future__ import annotations

from typing import Any, Protocol


class DeploymentObservationSource(Protocol):
    """Sim/real-neutral source of deployment observations."""

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
