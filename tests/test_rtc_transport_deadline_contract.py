from __future__ import annotations

import ast
import inspect
import math

import pytest

import deployments.client.rtc_transport as rtc_transport_module
from deployments.client.rtc_transport import (
    RTC_TRANSPORT_PHASE_COMPLETED,
    RTC_TRANSPORT_PHASE_INFERENCE,
    RTC_TRANSPORT_PHASE_TIMED_OUT,
    RtcTransportDeadlineExceeded,
    RtcTransportDeadlineSpec,
    RtcTransportPhaseReceipt,
)


def test_deadline_spec_is_model_neutral_immutable_and_content_addressed() -> None:
    disabled = RtcTransportDeadlineSpec()
    first = RtcTransportDeadlineSpec(
        connect_timeout_sec=1,
        metadata_timeout_sec=2,
        inference_timeout_sec=3,
        close_timeout_sec=4,
    )
    second = RtcTransportDeadlineSpec(
        connect_timeout_sec=1.0,
        metadata_timeout_sec=2.0,
        inference_timeout_sec=3.0,
        close_timeout_sec=4.0,
    )

    assert not disabled.enabled
    assert first.enabled
    assert first.semantic_content_hash == second.semantic_content_hash
    assert "model" not in first.__dataclass_fields__
    assert "checkpoint" not in first.__dataclass_fields__
    assert "horizon" not in first.__dataclass_fields__


def test_deadline_contract_import_firewall_has_no_model_or_transport_family() -> None:
    tree = ast.parse(inspect.getsource(rtc_transport_module))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            assert node.module is not None
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots <= {"math", "dataclasses", "dataset_contracts"}


@pytest.mark.parametrize("value", (0.0, -1.0, math.inf, -math.inf, math.nan))
def test_deadline_spec_rejects_nonpositive_or_nonfinite_budget(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        RtcTransportDeadlineSpec(inference_timeout_sec=value)


def test_phase_receipt_distinguishes_completion_from_timeout() -> None:
    completed = RtcTransportPhaseReceipt(
        sequence=0,
        phase=RTC_TRANSPORT_PHASE_INFERENCE,
        status=RTC_TRANSPORT_PHASE_COMPLETED,
        started_at=10.0,
        completed_at=10.1,
        timeout_sec=0.5,
        session_generation=2,
        connection_generation=3,
    )
    timed_out = RtcTransportPhaseReceipt(
        sequence=1,
        phase=RTC_TRANSPORT_PHASE_INFERENCE,
        status=RTC_TRANSPORT_PHASE_TIMED_OUT,
        started_at=11.0,
        completed_at=11.5,
        timeout_sec=0.5,
        session_generation=2,
        connection_generation=3,
        failure_type="TimeoutError",
        failure_detail="synthetic deadline",
    )

    assert completed.duration_sec == pytest.approx(0.1)
    assert completed.semantic_content_hash != timed_out.semantic_content_hash
    error = RtcTransportDeadlineExceeded(timed_out)
    assert error.receipt is timed_out
    assert "inference" in str(error)


def test_phase_receipt_failure_shape_fails_closed() -> None:
    with pytest.raises(ValueError, match="requires failure details"):
        RtcTransportPhaseReceipt(
            sequence=0,
            phase=RTC_TRANSPORT_PHASE_INFERENCE,
            status=RTC_TRANSPORT_PHASE_TIMED_OUT,
            started_at=1.0,
            completed_at=2.0,
            timeout_sec=1.0,
            session_generation=0,
            connection_generation=0,
        )
    with pytest.raises(ValueError, match="cannot carry a failure"):
        RtcTransportPhaseReceipt(
            sequence=0,
            phase=RTC_TRANSPORT_PHASE_INFERENCE,
            status=RTC_TRANSPORT_PHASE_COMPLETED,
            started_at=1.0,
            completed_at=1.5,
            timeout_sec=1.0,
            session_generation=0,
            connection_generation=0,
            failure_type="RuntimeError",
            failure_detail="not allowed",
        )
    with pytest.raises(ValueError, match="requires timeout_sec"):
        RtcTransportPhaseReceipt(
            sequence=0,
            phase=RTC_TRANSPORT_PHASE_INFERENCE,
            status=RTC_TRANSPORT_PHASE_TIMED_OUT,
            started_at=1.0,
            completed_at=2.0,
            timeout_sec=None,
            session_generation=0,
            connection_generation=0,
            failure_type="TimeoutError",
            failure_detail="native timeout without an explicit RTC budget",
        )
