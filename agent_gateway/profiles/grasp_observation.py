"""Backend-neutral bounded reader for retained, unplanned observations."""

from __future__ import annotations

from collections.abc import Mapping

from dataset_contracts.grasp_multiview_snapshot import (
    GRASP_MULTIVIEW_MAX_BYTES,
    decode_grasp_multiview_snapshot,
    decode_grasp_multiview_snapshot_arrays,
)

OBSERVATION_SLOT = "grasp_observation"


def read_retained_grasp_observation(*, executor, operation_id, artifact_store, runtime_session_id,
                                    expected_station_revision, frame_binding_hash,
                                    observation_slot=OBSERVATION_SLOT, decode_arrays=False):
    completion = executor.completion_for(operation_id)
    if completion is None or not completion.accepted:
        raise ValueError("Grasp observation is not retained by this session")
    operation = completion.operation
    if (operation.runtime_session_id != runtime_session_id or operation.status != "completed"
            or operation.execution_outcome != "completed"
            or not isinstance(operation.execution_result, Mapping)
            or "station_revision" not in operation.execution_result
            or operation.execution_result["station_revision"] != expected_station_revision):
        raise ValueError("Grasp observation session, revision, or completion differs")
    bindings = [b for b in operation.output_artifact_bindings if b.slot_id == observation_slot]
    if len(bindings) != 1:
        raise ValueError("Grasp observation lacks a unique retained Artifact")
    artifact = bindings[0].artifact
    data = artifact_store.read_bytes(artifact, maximum=GRASP_MULTIVIEW_MAX_BYTES)
    decoder = decode_grasp_multiview_snapshot_arrays if decode_arrays else decode_grasp_multiview_snapshot
    views, metadata, points, _ = decoder(data)
    if (metadata.get("runtime_session_id") != runtime_session_id
            or metadata.get("station_revision") != expected_station_revision
            or metadata.get("frame_binding_hash") != frame_binding_hash or points is not None):
        raise ValueError("Grasp observation provenance differs")
    return views, metadata
