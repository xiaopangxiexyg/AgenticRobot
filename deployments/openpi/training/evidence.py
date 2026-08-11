"""Fail-closed checkpoint evidence for formally trained OpenPI model-RTC."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.model_continuation import (
    require_trained_model_rtc_v1_contract,
)
from dataset_contracts.model_continuation_training import (
    ModelContinuationCheckpointReceipt,
    ModelContinuationTrainingRunManifest,
)
from dataset_contracts.policy_artifact import PolicyArtifactContract
from deployments.policy_artifact import load_policy_artifact
from deployments.openpi.artifact_identity import openpi_rtc_artifact_hash


MODEL_CONTINUATION_TRAINING_RUN_FILENAME_V1 = (
    "model_continuation_training_run.json"
)
MODEL_CONTINUATION_CHECKPOINT_RECEIPT_FILENAME_V1 = (
    "model_continuation_checkpoint_receipt.json"
)


def _json_object(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Model-RTC evidence is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Model-RTC evidence must be an object: {path}")
    return payload


def _sha256(value: str) -> str:
    normalized = str(value)
    return normalized if normalized.startswith("sha256:") else f"sha256:{normalized}"


@dataclass(frozen=True)
class OpenPIModelRtcCheckpointEvidence:
    checkpoint_dir: Path
    artifact: PolicyArtifactContract
    training_run: ModelContinuationTrainingRunManifest
    checkpoint_receipt: ModelContinuationCheckpointReceipt


def verify_openpi_model_rtc_checkpoint(
    checkpoint_dir: str | Path,
) -> OpenPIModelRtcCheckpointEvidence:
    """Verify the actual artifact, template identity and training receipts."""

    root = Path(checkpoint_dir).expanduser().resolve()
    artifact = load_policy_artifact(root / "policy_artifact.json")
    require_trained_model_rtc_v1_contract(artifact.model_continuation)
    run = ModelContinuationTrainingRunManifest.from_dict(
        _json_object(root / MODEL_CONTINUATION_TRAINING_RUN_FILENAME_V1)
    )
    receipt = ModelContinuationCheckpointReceipt.from_dict(
        _json_object(
            root / MODEL_CONTINUATION_CHECKPOINT_RECEIPT_FILENAME_V1
        )
    )
    if _sha256(run.semantic_content_hash) != (
        receipt.training_run_manifest_hash
    ):
        raise ValueError(
            "Checkpoint receipt does not bind the training-run manifest."
        )
    if _sha256(openpi_rtc_artifact_hash(artifact)) != (
        receipt.policy_artifact_hash
    ):
        raise ValueError(
            "Checkpoint receipt does not bind the actual policy artifact."
        )
    if artifact.model_id != run.output_model_id:
        raise ValueError(
            "Policy artifact model_id differs from the training-run manifest."
        )
    if receipt.model_id != run.output_model_id:
        raise ValueError(
            "Checkpoint receipt model_id differs from the training run."
        )
    try:
        artifact_step = int(artifact.checkpoint_step)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Trained model-RTC artifact must bind an integer checkpoint step."
        ) from exc
    if (
        artifact_step != receipt.checkpoint_step
        or root.name != str(receipt.checkpoint_step)
    ):
        raise ValueError(
            "Checkpoint directory, artifact and receipt step identities differ."
        )
    template = replace(artifact, checkpoint_step=None)
    if _sha256(openpi_rtc_artifact_hash(template)) != (
        run.policy_artifact_template_hash
    ):
        raise ValueError(
            "Actual policy artifact differs from the frozen training template "
            "outside checkpoint_step."
        )
    return OpenPIModelRtcCheckpointEvidence(
        checkpoint_dir=root,
        artifact=artifact,
        training_run=run,
        checkpoint_receipt=receipt,
    )


__all__ = [
    "MODEL_CONTINUATION_CHECKPOINT_RECEIPT_FILENAME_V1",
    "MODEL_CONTINUATION_TRAINING_RUN_FILENAME_V1",
    "OpenPIModelRtcCheckpointEvidence",
    "verify_openpi_model_rtc_checkpoint",
]
