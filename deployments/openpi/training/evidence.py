"""Fail-closed checkpoint evidence for formally trained OpenPI model-RTC."""

from __future__ import annotations

import gc
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
    orbax_commit_timestamp_nsecs: int
    orbax_item_handlers: tuple[str, ...]
    params_leaf_count: int
    params_restore_verified: bool


def _verify_orbax_checkpoint(
    root: Path,
    *,
    expected_step: int,
    restore_params: bool,
) -> tuple[int, tuple[str, ...], int, bool]:
    """Validate committed Orbax structure and optionally perform a full restore."""

    checkpoint_metadata = _json_object(root / "_CHECKPOINT_METADATA")
    try:
        initialized_at = int(checkpoint_metadata["init_timestamp_nsecs"])
        committed_at = int(checkpoint_metadata["commit_timestamp_nsecs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Orbax checkpoint has invalid commit timestamps.") from exc
    if initialized_at <= 0 or committed_at < initialized_at:
        raise ValueError("Orbax checkpoint is not committed after initialization.")
    handlers = checkpoint_metadata.get("item_handlers")
    if not isinstance(handlers, Mapping):
        raise ValueError("Orbax checkpoint item_handlers metadata is missing.")
    handler_names = tuple(sorted(str(name) for name in handlers))
    if "params" not in handlers or "assets" not in handlers:
        raise ValueError("Orbax checkpoint must contain params and assets items.")
    state_item = "training_step" if "training_step" in handlers else "train_state"
    if state_item not in handlers:
        raise ValueError("Orbax checkpoint has neither training_step nor train_state.")

    params_dir = root / "params"
    params_metadata = _json_object(params_dir / "_METADATA")
    tree_metadata = params_metadata.get("tree_metadata")
    if not isinstance(tree_metadata, Mapping) or not tree_metadata:
        raise ValueError("Orbax params metadata has no parameter leaves.")
    params_leaf_count = len(tree_metadata)
    for required in (
        params_dir / "manifest.ocdbt",
        params_dir / "_sharding",
        params_dir / "array_metadatas" / "process_0",
    ):
        if not required.is_file() or required.stat().st_size <= 0:
            raise ValueError(f"Orbax params item is incomplete: {required}")
    if not (params_dir / "d").is_dir():
        raise ValueError("Orbax params data directory is missing.")
    if any("tmp" in path.name.lower() for path in root.iterdir()):
        raise ValueError("Orbax checkpoint contains a temporary top-level item.")

    if state_item == "training_step":
        training_step = _json_object(root / "training_step" / "metadata")
        try:
            recorded_step = int(training_step["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Orbax lightweight training_step is invalid.") from exc
        if recorded_step != expected_step:
            raise ValueError(
                "Orbax training_step differs from the checkpoint directory."
            )
    else:
        train_state_metadata = root / "train_state" / "_METADATA"
        if not train_state_metadata.is_file():
            raise ValueError("Orbax train_state metadata is missing.")

    restored = False
    if restore_params:
        try:
            import orbax.checkpoint as ocp
        except ImportError as exc:
            raise RuntimeError(
                "Full params restore verification requires the OpenPI Orbax environment."
            ) from exc
        with ocp.PyTreeCheckpointer() as checkpointer:
            restored_tree = checkpointer.restore(params_dir)
        if restored_tree is None:
            raise ValueError("Orbax params full restore returned no tree.")
        restored = True
        del restored_tree
        gc.collect()
    return committed_at, handler_names, params_leaf_count, restored


def verify_openpi_model_rtc_checkpoint(
    checkpoint_dir: str | Path,
    *,
    restore_params: bool = False,
) -> OpenPIModelRtcCheckpointEvidence:
    """Verify sidecars plus committed Orbax state; optionally restore params."""

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
    (
        orbax_commit_timestamp_nsecs,
        orbax_item_handlers,
        params_leaf_count,
        params_restore_verified,
    ) = _verify_orbax_checkpoint(
        root,
        expected_step=artifact_step,
        restore_params=bool(restore_params),
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
        orbax_commit_timestamp_nsecs=orbax_commit_timestamp_nsecs,
        orbax_item_handlers=orbax_item_handlers,
        params_leaf_count=params_leaf_count,
        params_restore_verified=params_restore_verified,
    )


__all__ = [
    "MODEL_CONTINUATION_CHECKPOINT_RECEIPT_FILENAME_V1",
    "MODEL_CONTINUATION_TRAINING_RUN_FILENAME_V1",
    "OpenPIModelRtcCheckpointEvidence",
    "verify_openpi_model_rtc_checkpoint",
]
