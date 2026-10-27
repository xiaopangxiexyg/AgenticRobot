import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.grasp_observation import (
    OBSERVATION_SLOT,
    read_retained_grasp_observation,
)
from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.artifact_store import (
    ArtifactStoreIntegrityError, RecoverableAgentArtifactStore,
)
from dataset_contracts.grasp_multiview_snapshot import (
    GRASP_MULTIVIEW_ENCODING,
    GRASP_MULTIVIEW_SCHEMA_ID,
    encode_grasp_multiview_snapshot,
)
from dataset_contracts.grasp_rgbd_snapshot import (
    GRASP_RGBD_SNAPSHOT_SCHEMA_ID,
    encode_grasp_rgbd_snapshot,
)


@pytest.fixture
def retained_observation(tmp_path):
    store = RecoverableAgentArtifactStore(
        root=tmp_path / "artifacts",
        quota_bytes=1024 * 1024,
        max_artifact_bytes=512 * 1024,
        max_chunk_bytes=64 * 1024,
    )

    def build(*, roles=("main",), metadata_changes=None, with_scene=False):
        views = [
            encode_grasp_rgbd_snapshot(
                rgb=np.full((4, 5, 3), index, dtype=np.uint8),
                depth_m=np.full((4, 5), 0.5, dtype=np.float32),
                target_mask=np.zeros((4, 5), dtype=bool),
                metadata={"schema_id": GRASP_RGBD_SNAPSHOT_SCHEMA_ID, "role": role},
            )
            for index, role in enumerate(roles)
        ]
        metadata = {
            "schema_id": GRASP_MULTIVIEW_SCHEMA_ID,
            "runtime_session_id": "session:test",
            "station_revision": 7,
            "frame_binding_hash": "frame:test",
            **(metadata_changes or {}),
        }
        data = encode_grasp_multiview_snapshot(
            views=views,
            metadata=metadata,
            scene_points=np.array([[0.0, 0.0, 0.5]]) if with_scene else None,
            scene_target_mask=np.array([True]) if with_scene else None,
        )
        artifact = artifact_ref_for_bytes(
            data,
            media_type="application/zip",
            schema_id=GRASP_MULTIVIEW_SCHEMA_ID,
            encoding=GRASP_MULTIVIEW_ENCODING,
            compression="zip",
        )
        commit_artifact_bytes(
            store=store,
            data=data,
            artifact=artifact,
            producer_id="producer:test",
            runtime_session_id="session:test",
            idempotency_key="observation:test",
            now=10.0,
        )
        binding = AgentArtifactBinding(
            slot_id=OBSERVATION_SLOT,
            ordinal=0,
            artifact=artifact,
            production_receipt_ref="grasp-observation:observation:test",
        )
        operation = SimpleNamespace(
            runtime_session_id="session:test",
            status="completed",
            execution_outcome="completed",
            execution_result=MappingProxyType({"station_revision": 7}),
            output_artifact_bindings=(binding,),
        )
        completion = SimpleNamespace(accepted=True, operation=operation)
        lookups = []

        def completion_for(identifier):
            lookups.append(identifier)
            return completion

        return SimpleNamespace(
            arguments=dict(
                executor=SimpleNamespace(completion_for=completion_for),
                operation_id="observation:test",
                artifact_store=store,
                runtime_session_id="session:test",
                expected_station_revision=7,
                frame_binding_hash="frame:test",
            ),
            operation=operation,
            completion=completion,
            lookups=lookups,
            data=data,
            path=store.managed_read_path(artifact.artifact_id),
            views=views,
            metadata=metadata,
        )

    return build


def test_shared_reader_import_does_not_load_real_or_teleop_profiles():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from agent_gateway.profiles.grasp_observation import read_retained_grasp_observation
for prefix in ("agent_gateway.profiles.real", "teleop_runtime", "control_station", "hardware"):
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules), prefix
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_real_reader_export_is_the_shared_reader():
    from agent_gateway.profiles.real import grasp_observation

    assert (
        grasp_observation.read_retained_grasp_observation
        is read_retained_grasp_observation
    )
    assert grasp_observation.OBSERVATION_SLOT == OBSERVATION_SLOT


@pytest.mark.parametrize(
    "roles", [("main",), ("wrist_left", "main"), ("main", "wrist_left", "wrist_right")]
)
def test_shared_reader_returns_exact_retained_views_and_metadata(
    retained_observation, roles
):
    retained = retained_observation(roles=roles)
    views, metadata = read_retained_grasp_observation(**retained.arguments)

    assert views == retained.views
    assert metadata == retained.metadata
    assert retained.lookups == ["observation:test"]
    assert retained.path.read_bytes() == retained.data


@pytest.mark.parametrize("missing", [True, False])
def test_shared_reader_rejects_missing_or_unaccepted_completion(
    retained_observation, missing
):
    retained = retained_observation()
    if missing:
        retained.arguments["executor"].completion_for = lambda _: None
    else:
        retained.completion.accepted = False
    with pytest.raises(ValueError, match="not retained by this session"):
        read_retained_grasp_observation(**retained.arguments)


@pytest.mark.parametrize(
    "changes",
    [
        {"runtime_session_id": "session:other"},
        {"status": "running"},
        {"status": "failed"},
        {"execution_outcome": "failed"},
        {"execution_outcome": "pending"},
        {"execution_result": {"station_revision": 8}},
        {"execution_result": None},
        {"execution_result": {}},
    ],
)
def test_shared_reader_rejects_wrong_producer_session_revision_or_completion(
    retained_observation, changes
):
    retained = retained_observation()
    for name, value in changes.items():
        setattr(retained.operation, name, value)
    retained.arguments["artifact_store"] = None
    with pytest.raises(ValueError, match="session, revision, or completion differs"):
        read_retained_grasp_observation(**retained.arguments)


@pytest.mark.parametrize("count", [0, 2])
def test_shared_reader_requires_one_observation_artifact(retained_observation, count):
    retained = retained_observation()
    retained.operation.output_artifact_bindings *= count
    with pytest.raises(ValueError, match="unique retained Artifact"):
        read_retained_grasp_observation(**retained.arguments)


@pytest.mark.parametrize("tamper", ["length", "sha256"])
def test_shared_reader_rejects_changed_retained_bytes(
    retained_observation, tamper, tmp_path
):
    retained = retained_observation()
    changed = (
        retained.data + b"extra"
        if tamper == "length"
        else retained.data[:-1] + bytes([retained.data[-1] ^ 1])
    )
    # Corrupt a separate task-owned store; preserve the original retained bytes.
    store = RecoverableAgentArtifactStore(
        root=tmp_path / "corrupted-store", quota_bytes=1024 * 1024,
        max_artifact_bytes=512 * 1024, max_chunk_bytes=64 * 1024,
    )
    artifact = retained.operation.output_artifact_bindings[0].artifact
    try:
        commit_artifact_bytes(
            store=store, data=retained.data, artifact=artifact,
            producer_id="producer:test", runtime_session_id="session:test",
            idempotency_key="observation:corrupted", now=10.0,
        )
        corrupted = store.managed_read_path(artifact.artifact_id)
        corrupted.chmod(0o600)
        corrupted.write_bytes(changed)
        retained.arguments["artifact_store"] = store
        with pytest.raises(ArtifactStoreIntegrityError, match="failed verification"):
            read_retained_grasp_observation(**retained.arguments)
        assert store.artifact(artifact.artifact_id).state == "quarantined"
    finally:
        store.close()
    assert retained.path.read_bytes() == retained.data


@pytest.mark.parametrize(
    "metadata_changes",
    [
        {"runtime_session_id": "session:other"},
        {"station_revision": 8},
        {"frame_binding_hash": "frame:other"},
    ],
)
def test_shared_reader_rejects_archive_provenance_mismatch(
    retained_observation, metadata_changes
):
    retained = retained_observation(metadata_changes=metadata_changes)
    with pytest.raises(ValueError, match="provenance differs"):
        read_retained_grasp_observation(**retained.arguments)


def test_shared_reader_rejects_precomputed_scene(retained_observation):
    retained = retained_observation(with_scene=True)
    with pytest.raises(ValueError, match="provenance differs"):
        read_retained_grasp_observation(**retained.arguments)
