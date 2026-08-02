"""Optional cross-task authoring vocabulary for coarse progress semantics.

These identifiers improve cross-profile search. They are not external
templates: every task profile still embeds its own step identity, display
name, ordering, and transitions, so historical profiles remain self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommonProgressSemanticType:
    semantic_type: str
    description: str


COMMON_PROGRESS_SEMANTIC_TYPES = (
    CommonProgressSemanticType(
        "APPROACH_TARGET",
        "Move into a task-specific pre-interaction or alignment state.",
    ),
    CommonProgressSemanticType(
        "ACQUIRE_OBJECT",
        "Establish and verify control of a task object.",
    ),
    CommonProgressSemanticType(
        "TRANSFER_OBJECT",
        "Move a controlled object toward a task-specific destination.",
    ),
    CommonProgressSemanticType(
        "PLACE_OBJECT",
        "Place or release an object at a task-specific destination.",
    ),
    CommonProgressSemanticType(
        "VERIFY_OUTCOME",
        "Verify a task-specific terminal or intermediate outcome.",
    ),
    CommonProgressSemanticType(
        "RECOVER_STATE",
        "Return from an off-nominal state to a task-valid state.",
    ),
)

_BY_ID = {
    item.semantic_type: item
    for item in COMMON_PROGRESS_SEMANTIC_TYPES
}


def common_progress_semantic_type(
    semantic_type: str,
) -> CommonProgressSemanticType | None:
    """Return optional shared vocabulary metadata without enforcing it."""

    return _BY_ID.get(str(semantic_type).strip().upper())


__all__ = [
    "COMMON_PROGRESS_SEMANTIC_TYPES",
    "CommonProgressSemanticType",
    "common_progress_semantic_type",
]
