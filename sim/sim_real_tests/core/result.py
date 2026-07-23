from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SimRealScenarioResult:
    """Serializable result shared by model-only and physics-backed scenarios."""

    scenario: str
    passed: bool
    metrics: dict[str, float | int | bool | str | None] = field(default_factory=dict)
    records: tuple[dict[str, Any], ...] = ()
    failure: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": str(self.scenario),
            "passed": bool(self.passed),
            "metrics": dict(self.metrics),
            "records": [dict(record) for record in self.records],
            "failure": str(self.failure),
        }


__all__ = ["SimRealScenarioResult"]
