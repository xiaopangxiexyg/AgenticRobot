"""Issue construction and deterministic result ordering."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Protocol, TypeVar


class ControlTraceLineageIssueView(Protocol):
    code: str
    root_fact_id: str
    detail: str
    related_fact_ids: tuple[str, ...]


class ControlTraceLineageIssueCollector(Protocol):
    def add(
        self,
        code: str,
        root_fact_id: str,
        detail: str,
        *related_fact_ids: str,
    ) -> None: ...


IssueT = TypeVar("IssueT", bound=ControlTraceLineageIssueView)


class ControlTraceLineageIssueSink(Generic[IssueT]):
    """Append-only issue collector shared by explicit validator families."""

    __slots__ = ("_issue_factory", "_issues")

    def __init__(self, issue_factory: Callable[..., IssueT]) -> None:
        self._issue_factory = issue_factory
        self._issues: list[IssueT] = []

    def add(
        self,
        code: str,
        root_fact_id: str,
        detail: str,
        *related_fact_ids: str,
    ) -> None:
        self._issues.append(
            self._issue_factory(
                code=code,
                root_fact_id=root_fact_id,
                detail=detail,
                related_fact_ids=tuple(
                    str(value) for value in related_fact_ids if value
                ),
            )
        )

    def sorted_issues(self) -> tuple[IssueT, ...]:
        return tuple(
            sorted(
                self._issues,
                key=lambda item: (
                    item.root_fact_id,
                    item.code,
                    item.related_fact_ids,
                ),
            )
        )


__all__ = [
    "ControlTraceLineageIssueCollector",
    "ControlTraceLineageIssueSink",
    "ControlTraceLineageIssueView",
]
