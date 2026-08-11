"""Immutable indexes shared by ControlTrace lineage validators."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ..control_trace import ControlTraceFact


@dataclass(frozen=True, slots=True)
class ControlTraceValidationContext:
    """One immutable view of the assessed fact sequence.

    Fact order remains caller supplied. Duplicate ``fact_id`` values preserve
    the historical last-value-wins lookup behavior while event-type buckets
    retain every fact in original order.
    """

    ordered: tuple[ControlTraceFact, ...]
    by_id: Mapping[str, ControlTraceFact]
    by_event_type: Mapping[str, tuple[ControlTraceFact, ...]]

    @classmethod
    def build(
        cls,
        facts: Sequence[ControlTraceFact],
    ) -> "ControlTraceValidationContext":
        ordered = tuple(facts)
        grouped: dict[str, list[ControlTraceFact]] = {}
        for fact in ordered:
            grouped.setdefault(fact.event_type, []).append(fact)
        return cls(
            ordered=ordered,
            by_id=MappingProxyType({fact.fact_id: fact for fact in ordered}),
            by_event_type=MappingProxyType(
                {
                    event_type: tuple(event_facts)
                    for event_type, event_facts in grouped.items()
                }
            ),
        )

    def facts_of_type(self, event_type: str) -> tuple[ControlTraceFact, ...]:
        return self.by_event_type.get(str(event_type), ())

    def payload_index(
        self,
        *,
        event_type: str,
        payload_key: str,
    ) -> dict[str, list[ControlTraceFact]]:
        """Index one event family without changing historical list semantics."""

        result: dict[str, list[ControlTraceFact]] = {}
        for fact in self.facts_of_type(event_type):
            value = fact.payload.get(payload_key)
            if value is not None:
                result.setdefault(str(value), []).append(fact)
        return result


__all__ = ["ControlTraceValidationContext"]
