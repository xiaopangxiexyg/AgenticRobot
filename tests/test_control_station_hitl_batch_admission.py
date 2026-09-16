"""Fail-closed HITL admission diagnostics without device execution."""

from types import SimpleNamespace

import pytest

from control_station.composition.mujoco_policy_human_allocation_source import (
    _rejection_detail,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_rejected_human_batch_reports_underlying_admission_receipt() -> None:
    submission = SimpleNamespace(
        error="human_allocation_input_batch_rejected",
        input_cycle=SimpleNamespace(
            status="batch_rejected",
            publication=SimpleNamespace(
                receipt=SimpleNamespace(
                    rejection_code="retention_window_violation",
                    message="Input validity ended before admission.",
                )
            ),
        ),
    )

    assert _rejection_detail(submission) == (
        "human_allocation_input_batch_rejected:retention_window_violation:"
        "Input validity ended before admission."
    )


def test_rejected_human_batch_without_receipt_keeps_existing_detail() -> None:
    submission = SimpleNamespace(
        error="human_allocation_input_batch_rejected",
        input_cycle=SimpleNamespace(status="batch_rejected", publication=None),
    )

    assert _rejection_detail(submission) == "human_allocation_input_batch_rejected"
