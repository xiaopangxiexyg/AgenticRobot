"""Compatibility exports for RM65 target-backend execution evidence.

The canonical evidence contract lives in
``teleop_runtime.backends.arms.rm65_execution_evidence``. This deployment path
remains a stateless re-export for existing callers and persisted import usage.
"""

from teleop_runtime.backends.arms.rm65_execution_evidence import (
    RM65_M5_8B0_STOP_ONLY_FIELD_EVIDENCE_V1,
    RM65_M5_8B1_IDLE_WORKER_FIELD_EVIDENCE_V1,
    RM65ExecutionEvidenceBundle,
    RM65HistoricalExecutionEvidenceContract,
    build_rm65_execution_evidence_bundle,
    resolve_rm65_historical_execution_evidence_contract,
)

__all__ = [
    "RM65_M5_8B0_STOP_ONLY_FIELD_EVIDENCE_V1",
    "RM65_M5_8B1_IDLE_WORKER_FIELD_EVIDENCE_V1",
    "RM65ExecutionEvidenceBundle",
    "RM65HistoricalExecutionEvidenceContract",
    "build_rm65_execution_evidence_bundle",
    "resolve_rm65_historical_execution_evidence_contract",
]
