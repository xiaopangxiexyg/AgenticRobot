"""Composition-only compatibility projections for ControlTrace.

This module is intentionally loaded only by explicit multi-source callers.
Single-source deploy, teleop, and replay paths use the general projection
module without importing ``control_composition``.
"""

from __future__ import annotations

from control_composition.composition import (
    CompositionEvaluation,
    CompositionInputAdmissionReceipt,
    CompositionInputRecord,
    CompositionInputSnapshot,
    CompositionSpec,
)
from dataset_contracts.control_trace import ControlTraceFact
from dataset_interface.composition_control_trace import (
    project_composition_evaluation as _project_composition_evaluation,
    project_composition_input as _project_composition_input,
    project_composition_input_admission as _project_composition_input_admission,
    project_composition_input_snapshot as _project_composition_input_snapshot,
    project_composition_spec as _project_composition_spec,
)
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext


def project_composition_input_admission(
    receipt: CompositionInputAdmissionReceipt,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    return _project_composition_input_admission(
        AuthoritativeTraceContext(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=clock_domain_id,
        ),
        receipt,
        occurred_at=occurred_at,
    )


def project_composition_spec(
    spec: CompositionSpec,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    """Compatibility projection; implementation lives in the dataset adapter."""

    return _project_composition_spec(
        AuthoritativeTraceContext(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=clock_domain_id,
        ),
        spec,
        occurred_at=occurred_at,
    )


def project_composition_input(
    record: CompositionInputRecord,
    *,
    embodiment_instance_id: str,
) -> ControlTraceFact:
    return _project_composition_input(
        AuthoritativeTraceContext(
            runtime_session_id=record.runtime_session_id,
            control_domain_id=record.control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=record.available_clock_domain_id,
        ),
        record,
    )


def project_composition_input_snapshot(
    snapshot: CompositionInputSnapshot,
    *,
    runtime_session_id: str,
    embodiment_instance_id: str,
    clock_domain_id: str,
) -> ControlTraceFact:
    return _project_composition_input_snapshot(
        AuthoritativeTraceContext(
            runtime_session_id=runtime_session_id,
            control_domain_id=snapshot.control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=clock_domain_id,
        ),
        snapshot,
    )


def project_composition_evaluation(
    result: CompositionEvaluation,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
) -> ControlTraceFact:
    return _project_composition_evaluation(
        AuthoritativeTraceContext(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=clock_domain_id,
        ),
        result,
        occurred_at=occurred_at,
    )


__all__ = [
    "project_composition_input",
    "project_composition_input_admission",
    "project_composition_input_snapshot",
    "project_composition_evaluation",
    "project_composition_spec",
]
