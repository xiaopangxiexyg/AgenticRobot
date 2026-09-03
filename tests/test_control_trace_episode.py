from __future__ import annotations

import hashlib
import multiprocessing
from pathlib import Path

import numpy as np
import pytest

from dataset_contracts.completeness import TraceRequirementProfile, trace_requirement_profile
from dataset_contracts.canonical_encoding import canonical_bytes, canonical_loads
from dataset_contracts.control_trace import ControlTraceFact
from dataset_contracts.control_trace_episode import V24EpisodeCaptureCommit
from dataset_contracts.stream_catalog import (
    NATIVE_STREAM_EPISODE_MANIFEST_FILE_DOMAIN_V1,
    NATIVE_STREAM_EPISODE_RELATIVE_ROOT_V1,
    NativeStreamContract,
    NativeStreamExternalArtifactReceipt,
    NativeStreamExternalSampleIntent,
)
from dataset_interface.streams import load_native_stream_episode
from collection_runtime.stream_sink import NativeStreamEpisodeWriter
from teleop_runtime.control_trace import (
    ControlTraceRuntime,
    ControlTraceSpool,
    ControlTraceSpoolSink,
    ControlTraceWriterHealthLedger,
    EpisodeCaptureManifestStore,
    EpisodeTraceRecorderSink,
    RuntimeTraceSession,
    SubordinateEpisodeTraceLifecycle,
    assess_episode_trace_seal,
)


def _profile(
    *,
    formal: bool = False,
    source_validation_required: bool = False,
) -> TraceRequirementProfile:
    base = trace_requirement_profile("plain_dry_run_v1")
    required = set(base.required_event_types)
    optional = set(base.optional_event_types)
    if source_validation_required:
        required.add("source_validation")
        optional.discard("source_validation")
    return TraceRequirementProfile(
        profile_id=("formal-test-v1" if formal else "diagnostic-test-v1"),
        runtime_mode=("formal_test" if formal else "diagnostic_test"),
        required_event_types=frozenset(required),
        optional_event_types=frozenset(optional),
        not_applicable_event_types=base.not_applicable_event_types,
        receipt_policy=base.receipt_policy,
        measurement_policy=base.measurement_policy,
        writer_failure_behavior=base.writer_failure_behavior,
        lineage_policy_id="generic_reference_integrity_v1",
        profile_scope=("formal_capture" if formal else "runtime_diagnostic"),
        implementation_status=("runtime_integrated" if formal else "fixture_only"),
    )


def _runtime(profile: TraceRequirementProfile) -> ControlTraceRuntime:
    return ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="trace-session-1",
            runtime_session_id="runtime-session-1",
            control_domain_id="robot-0",
            profile_id=profile.profile_id,
            started_at=0.0,
            clock_domain_id="local_monotonic",
        ),
        profile=profile,
        capacity=128,
    )


def _source_fact(index: int) -> ControlTraceFact:
    return ControlTraceFact.create(
        fact_id=f"source-validation-{index}",
        event_type="source_validation",
        runtime_session_id="runtime-session-1",
        control_domain_id="robot-0",
        occurred_at=float(index),
        clock_domain_id="local_monotonic",
        payload={
            "validation_id": f"validation-{index}",
            "source_event_id": f"source-{index}",
            "accepted": True,
            "reasons": (),
        },
    )


def _terminal(index: int) -> ControlTraceFact:
    return ControlTraceFact.create(
        fact_id="runtime-terminal",
        event_type="runtime_trace_terminal",
        runtime_session_id="runtime-session-1",
        control_domain_id="robot-0",
        occurred_at=float(index),
        clock_domain_id="local_monotonic",
        payload={
            "terminal_status": "completed",
            "reason": "fixture_complete",
            "occurred_at": float(index),
        },
    )


def _sink(runtime: ControlTraceRuntime, root: Path):
    spool = ControlTraceSpool(
        root / "control_trace.wal",
        trace_session_id=runtime.session.trace_session_id,
        max_bytes=1 << 20,
        physical_preallocation=False,
    )
    health = ControlTraceWriterHealthLedger(
        root / "writer_health.ledger",
        trace_session_id=runtime.session.trace_session_id,
    )
    sink = ControlTraceSpoolSink(
        sink_id="episode-test-sink",
        runtime=runtime,
        spool=spool,
        health_ledger=health,
    )
    return spool, health, sink


def _native_q6_contract() -> NativeStreamContract:
    return NativeStreamContract(
        stream_id="human.left_hand_q6",
        modality="human_joint_input",
        producer_id="virtual-q6",
        value_schema="float32[6]",
        shape=(6,),
        dtype="<f4",
        units="normalized_joint_position",
        frame=None,
        measured_clock_domain_id="local_monotonic",
        available_clock_domain_id="local_monotonic",
        nominal_rate_hz=30.0,
        min_expected_rate_hz=10.0,
        max_expected_rate_hz=60.0,
        access_role="actor_input",
    )


def _publish_seal_then_wait_for_process_termination(
    root: str,
    ready,
    stop,
) -> None:
    dataset_root = Path(root)
    spool_root = dataset_root / "spool"
    spool_root.mkdir(parents=True, exist_ok=True)
    runtime = _runtime(_profile(formal=True))
    spool, health, durable_sink = _sink(runtime, spool_root)
    recorder_sink = EpisodeTraceRecorderSink(
        runtime,
        lifecycle_owner_id="canonical-gui-recorder",
        flush_to_durable=durable_sink.flush_pending,
        manifest_store=EpisodeCaptureManifestStore(dataset_root),
        uuid_factory=lambda: "process-kill-episode",
    )
    prepared = recorder_sink.prepare_episode(legacy_episode_index=0, prepared_at=1.0)
    recorder_sink.bind_prepared_episode(prepared, bound_at=1.0)
    released = recorder_sink.release_episode(
        prepared,
        unbound_at=2.0,
        reason="legacy_finalize_requested",
    )
    assert released.seal.trace_sealed
    assert released.seal_publish_receipt is not None
    ready.set()
    stop.wait(30.0)
    health.close()
    spool.close()


def test_multiple_episode_seals_do_not_terminate_runtime_session(tmp_path: Path) -> None:
    runtime = _runtime(_profile())
    lifecycle = SubordinateEpisodeTraceLifecycle(
        runtime,
        lifecycle_owner_id="canonical-gui-recorder",
    )
    spool, health, sink = _sink(runtime, tmp_path)
    try:
        lifecycle.bind_external_episode(
            episode_uuid="episode-0",
            binding_event_id="episode-0-bind",
            bound_at=1.0,
        )
        assert runtime.issue(_source_fact(2)).accepted
        episode_0 = lifecycle.release_external_episode(
            unbinding_event_id="episode-0-unbind",
            unbound_at=3.0,
            reason="legacy_finalize_requested",
        )
        assert sink.flush_pending().failure_code is None
        seal_0 = assess_episode_trace_seal(
            runtime,
            episode_0.trace_range,
            assessment_id="episode-0-seal",
            assessed_at=4.0,
        )

        lifecycle.bind_external_episode(
            episode_uuid="episode-1",
            binding_event_id="episode-1-bind",
            bound_at=5.0,
        )
        assert runtime.issue(_source_fact(6)).accepted
        episode_1 = lifecycle.release_external_episode(
            unbinding_event_id="episode-1-unbind",
            unbound_at=7.0,
            reason="legacy_finalize_requested",
        )
        assert sink.flush_pending().failure_code is None
        seal_1 = assess_episode_trace_seal(
            runtime,
            episode_1.trace_range,
            assessment_id="episode-1-seal",
            assessed_at=8.0,
        )

        assert seal_0.trace_sealed and seal_1.trace_sealed
        assert episode_0.trace_range.clock_domain_id == "local_monotonic"
        assert not seal_0.capture_commit_eligible
        assert episode_0.trace_range.last_journal_sequence < episode_1.trace_range.first_journal_sequence
        assert runtime.issue(_terminal(9)).accepted
        assert seal_0.semantic_content_hash == assess_episode_trace_seal(
            runtime,
            episode_0.trace_range,
            assessment_id="episode-0-seal",
            assessed_at=4.0,
        ).semantic_content_hash
    finally:
        health.close()
        spool.close()


def test_formal_seal_requires_durable_dependency_closure(tmp_path: Path) -> None:
    runtime = _runtime(_profile(formal=True, source_validation_required=True))
    startup = runtime.issue(_source_fact(1))
    assert startup.accepted
    lifecycle = SubordinateEpisodeTraceLifecycle(
        runtime,
        lifecycle_owner_id="canonical-gui-recorder",
    )
    lifecycle.bind_external_episode(
        episode_uuid="episode-0",
        binding_event_id="episode-0-bind",
        bound_at=2.0,
    )
    released = lifecycle.release_external_episode(
        unbinding_event_id="episode-0-unbind",
        unbound_at=3.0,
        reason="legacy_finalize_requested",
    )
    spool, health, sink = _sink(runtime, tmp_path)
    try:
        assert sink.flush_pending().failure_code is None
        without_dependency = assess_episode_trace_seal(
            runtime,
            released.trace_range,
            assessment_id="seal-without-dependency",
            assessed_at=4.0,
        )
        with_dependency = assess_episode_trace_seal(
            runtime,
            released.trace_range,
            assessment_id="seal-with-dependency",
            assessed_at=4.0,
            dependency_sequences=(startup.journal_issued_sequence,),
        )

        assert not without_dependency.trace_sealed
        assert "required_event_missing" in without_dependency.reason_codes
        assert with_dependency.trace_sealed
        assert with_dependency.capture_commit_eligible
        assert len(with_dependency.dependency_refs) == 1

        commit = V24EpisodeCaptureCommit.create(
            commit_id="capture-commit-0",
            legacy_episode_index=0,
            legacy_schema="lerobot_v2.3_compatible",
            seal=with_dependency,
            capabilities=("control_trace_core",),
            committed_at=5.0,
        )
        assert commit.episode_uuid == "episode-0"
        assert commit.trace_seal_hash == with_dependency.semantic_content_hash
        assert commit.committed_clock_domain_id == "local_wall_clock"
    finally:
        health.close()
        spool.close()


def test_unflushed_episode_cannot_seal_or_commit() -> None:
    runtime = _runtime(_profile(formal=True))
    lifecycle = SubordinateEpisodeTraceLifecycle(
        runtime,
        lifecycle_owner_id="canonical-gui-recorder",
    )
    lifecycle.bind_external_episode(
        episode_uuid="episode-0",
        binding_event_id="episode-0-bind",
        bound_at=1.0,
    )
    released = lifecycle.release_external_episode(
        unbinding_event_id="episode-0-unbind",
        unbound_at=2.0,
        reason="legacy_finalize_requested",
    )

    seal = assess_episode_trace_seal(
        runtime,
        released.trace_range,
        assessment_id="unflushed-seal",
        assessed_at=3.0,
    )

    assert not seal.trace_sealed
    assert seal.durability_gap_ranges == ((1, 2),)
    with pytest.raises(ValueError, match="not eligible"):
        V24EpisodeCaptureCommit.create(
            commit_id="invalid-commit",
            legacy_episode_index=0,
            legacy_schema="lerobot_v2.3_compatible",
            seal=seal,
            capabilities=("control_trace_core",),
            committed_at=4.0,
        )


def test_subordinate_lifecycle_exposes_no_dataset_finalize_api() -> None:
    lifecycle = SubordinateEpisodeTraceLifecycle(
        _runtime(_profile()),
        lifecycle_owner_id="canonical-gui-recorder",
    )

    assert not hasattr(lifecycle, "start_episode")
    assert not hasattr(lifecycle, "finalize_episode")
    assert not hasattr(lifecycle, "commit_episode")


def test_recorder_sink_only_creates_commit_candidate_after_legacy_finalize(
    tmp_path: Path,
) -> None:
    runtime = _runtime(_profile(formal=True))
    spool, health, durable_sink = _sink(runtime, tmp_path)
    recorder_sink = EpisodeTraceRecorderSink(
        runtime,
        lifecycle_owner_id="canonical-gui-recorder",
        flush_to_durable=durable_sink.flush_pending,
        uuid_factory=lambda: "episode-uuid-0",
    )
    try:
        prepared = recorder_sink.prepare_episode(
            legacy_episode_index=0,
            prepared_at=1.0,
        )
        recorder_sink.bind_prepared_episode(prepared, bound_at=2.0)
        released = recorder_sink.release_episode(
            prepared,
            unbound_at=3.0,
            reason="legacy_finalize_requested",
        )

        assert released.seal.capture_commit_eligible
        report = recorder_sink.legacy_finalize_completed(
            released,
            completed_at=4.0,
        )
        assert report.legacy_finalize_status == "finalized"
        assert report.capture_commit_candidate is not None
        assert report.capture_commit_candidate.episode_uuid == "episode-uuid-0"
    finally:
        health.close()
        spool.close()


def test_recorder_sink_never_commits_failed_or_discarded_legacy_episode(
    tmp_path: Path,
) -> None:
    runtime = _runtime(_profile(formal=True))
    spool, health, durable_sink = _sink(runtime, tmp_path)
    recorder_sink = EpisodeTraceRecorderSink(
        runtime,
        lifecycle_owner_id="canonical-gui-recorder",
        flush_to_durable=durable_sink.flush_pending,
        uuid_factory=lambda: "episode-uuid-0",
    )
    try:
        prepared = recorder_sink.prepare_episode(
            legacy_episode_index=0,
            prepared_at=1.0,
        )
        recorder_sink.bind_prepared_episode(prepared, bound_at=2.0)
        released = recorder_sink.release_episode(
            prepared,
            unbound_at=3.0,
            reason="legacy_finalize_requested",
        )
        failed = recorder_sink.legacy_finalize_failed(
            released,
            completed_at=4.0,
            error="legacy writer failed",
        )

        assert failed is not None
        assert failed.legacy_finalize_status == "failed"
        assert failed.capture_commit_candidate is None
    finally:
        health.close()
        spool.close()


def test_episode_manifest_store_publishes_seal_then_commit_atomically(
    tmp_path: Path,
) -> None:
    runtime = _runtime(_profile(formal=True))
    spool_root = tmp_path / "spool"
    spool_root.mkdir()
    spool, health, durable_sink = _sink(runtime, spool_root)
    store = EpisodeCaptureManifestStore(tmp_path / "dataset")
    recorder_sink = EpisodeTraceRecorderSink(
        runtime,
        lifecycle_owner_id="canonical-gui-recorder",
        flush_to_durable=durable_sink.flush_pending,
        uuid_factory=lambda: "episode-uuid-0",
        manifest_store=store,
    )
    try:
        prepared = recorder_sink.prepare_episode(
            legacy_episode_index=0,
            prepared_at=1.0,
        )
        recorder_sink.bind_prepared_episode(prepared, bound_at=2.0)
        released = recorder_sink.release_episode(
            prepared,
            unbound_at=3.0,
            reason="legacy_finalize_requested",
        )
        finalized = recorder_sink.legacy_finalize_completed(
            released,
            completed_at=4.0,
        )
        assert finalized.capture_commit_candidate is not None
        assert released.seal_publish_receipt is not None
        assert finalized.capture_commit_status == "committed"
        assert finalized.capture_commit_publish_receipt is not None
        seal_receipt = released.seal_publish_receipt
        commit_receipt = finalized.capture_commit_publish_receipt
        idempotent = store.publish_capture_commit(finalized.capture_commit_candidate)

        assert seal_receipt.durable and commit_receipt.durable
        assert not commit_receipt.idempotent and idempotent.idempotent
        assert store.read_seal_payload("episode-uuid-0")[
            "semantic_content_hash"
        ] == released.seal.semantic_content_hash
        assert store.read_commit_payload("episode-uuid-0")[
            "legacy_finalize_status"
        ] == "finalized"
        assert not list((store.root / "episode-uuid-0").glob("*.tmp"))
    finally:
        health.close()
        spool.close()


def test_native_stream_manifest_is_bound_to_same_episode_commit(
    tmp_path: Path,
) -> None:
    runtime = _runtime(_profile(formal=True))
    dataset_root = tmp_path / "dataset"
    spool_root = tmp_path / "spool"
    spool_root.mkdir()
    spool, health, durable_sink = _sink(runtime, spool_root)
    contract = _native_q6_contract()
    store = EpisodeCaptureManifestStore(dataset_root)
    recorder_sink = EpisodeTraceRecorderSink(
        runtime,
        lifecycle_owner_id="canonical-gui-recorder",
        flush_to_durable=durable_sink.flush_pending,
        uuid_factory=lambda: "episode-native-bound",
        manifest_store=store,
        committed_capabilities=("control_trace_core", "native_streams"),
        native_stream_writer_factory=lambda prepared: NativeStreamEpisodeWriter(
            dataset_root,
            episode_uuid=prepared.episode_uuid,
            legacy_episode_index=prepared.legacy_episode_index,
            contracts={contract.stream_id: contract},
        ),
    )
    try:
        prepared = recorder_sink.prepare_episode(
            legacy_episode_index=7,
            prepared_at=1.0,
        )
        native_writer = recorder_sink.native_stream_capture(prepared)
        assert native_writer.episode_uuid == prepared.episode_uuid
        assert native_writer.legacy_episode_index == prepared.legacy_episode_index
        native_writer.append(
            stream_id=contract.stream_id,
            sample_id="q6-0",
            sequence=0,
            measured_at=1.1,
            measured_clock_domain_id="local_monotonic",
            available_at=1.2,
            available_clock_domain_id="local_monotonic",
            source_payload_object_id="virtual-q6-0",
            payload=np.arange(6, dtype=np.float32),
            written_at=1.3,
            written_clock_domain_id="local_monotonic",
        )
        recorder_sink.bind_prepared_episode(prepared, bound_at=2.0)
        released = recorder_sink.release_episode(
            prepared,
            unbound_at=3.0,
            reason="legacy_finalize_requested",
        )
        assert released.native_stream_manifest is not None
        assert released.native_stream_manifest.episode_uuid == prepared.episode_uuid
        assert (
            released.native_stream_manifest.legacy_episode_index
            == prepared.legacy_episode_index
        )

        finalized = recorder_sink.legacy_finalize_completed(
            released,
            completed_at=4.0,
        )
        assert finalized.capture_commit_status == "committed"
        assert finalized.capture_commit_candidate is not None
        native_hash = released.native_stream_manifest.semantic_content_hash
        assert finalized.native_stream_manifest_hash == native_hash
        assert (
            finalized.capture_commit_candidate.capability_artifact_hash_by_name
            == {"native_streams": native_hash}
        )
        assert store.read_commit_payload(prepared.episode_uuid)[
            "capability_artifact_hash_by_name"
        ] == {"native_streams": native_hash}

        captured = load_native_stream_episode(dataset_root, prepared.episode_uuid)
        assert captured.manifest.semantic_content_hash == native_hash
        assert tuple(captured.records_by_stream) == (contract.stream_id,)
        assert captured.records_by_stream[contract.stream_id][0].sample_id == "q6-0"

        manifest_path = (
            dataset_root
            / NATIVE_STREAM_EPISODE_RELATIVE_ROOT_V1
            / prepared.episode_uuid
            / "manifest.cjson"
        )
        tampered = canonical_loads(
            manifest_path.read_bytes(),
            expected_domain=NATIVE_STREAM_EPISODE_MANIFEST_FILE_DOMAIN_V1,
        )
        tampered["sealed_at"] = float(tampered["sealed_at"]) + 1.0
        manifest_path.write_bytes(
            canonical_bytes(
                tampered,
                domain=NATIVE_STREAM_EPISODE_MANIFEST_FILE_DOMAIN_V1,
            )
        )
        with pytest.raises(ValueError, match="does not match"):
            store.publish_capture_commit(finalized.capture_commit_candidate)
    finally:
        health.close()
        spool.close()


def test_native_stream_seal_failure_does_not_commit_or_break_legacy_finalize(
    tmp_path: Path,
) -> None:
    class FailingNativeCapture:
        def __init__(self, episode_uuid: str, legacy_episode_index: int) -> None:
            self.episode_uuid = episode_uuid
            self.legacy_episode_index = legacy_episode_index
            self.closed = False

        def seal(self, *, sealed_at: float, sealed_clock_domain_id: str):
            del sealed_at, sealed_clock_domain_id
            raise OSError("fixture native seal failure")

        def close_unsealed(self) -> None:
            self.closed = True

    runtime = _runtime(_profile(formal=True))
    spool_root = tmp_path / "spool"
    spool_root.mkdir()
    spool, health, durable_sink = _sink(runtime, spool_root)
    captures: list[FailingNativeCapture] = []

    def create_capture(prepared):
        capture = FailingNativeCapture(
            prepared.episode_uuid,
            prepared.legacy_episode_index,
        )
        captures.append(capture)
        return capture

    recorder_sink = EpisodeTraceRecorderSink(
        runtime,
        lifecycle_owner_id="canonical-gui-recorder",
        flush_to_durable=durable_sink.flush_pending,
        uuid_factory=lambda: "episode-native-failed",
        manifest_store=EpisodeCaptureManifestStore(tmp_path / "dataset"),
        committed_capabilities=("control_trace_core", "native_streams"),
        native_stream_writer_factory=create_capture,
    )
    try:
        prepared = recorder_sink.prepare_episode(
            legacy_episode_index=0,
            prepared_at=1.0,
        )
        recorder_sink.bind_prepared_episode(prepared, bound_at=2.0)
        released = recorder_sink.release_episode(
            prepared,
            unbound_at=3.0,
            reason="legacy_finalize_requested",
        )
        assert released.seal.capture_commit_eligible
        assert released.native_stream_manifest is None
        assert "fixture native seal failure" in released.native_stream_seal_error
        assert captures[0].closed

        finalized = recorder_sink.legacy_finalize_completed(
            released,
            completed_at=4.0,
        )
        assert finalized.legacy_finalize_status == "finalized"
        assert finalized.capture_commit_candidate is None
        assert finalized.capture_commit_status == "not_eligible"
        assert finalized.message.startswith("native_streams_not_sealed:")
        assert (
            EpisodeCaptureManifestStore(tmp_path / "dataset").read_commit_payload(
                prepared.episode_uuid
            )
            is None
        )
    finally:
        health.close()
        spool.close()


def test_external_native_stream_seals_only_after_legacy_artifact_receipt(
    tmp_path: Path,
) -> None:
    runtime = _runtime(_profile(formal=True))
    dataset_root = tmp_path / "dataset"
    spool_root = tmp_path / "spool"
    spool_root.mkdir()
    spool, health, durable_sink = _sink(runtime, spool_root)
    contract = _native_q6_contract()
    store = EpisodeCaptureManifestStore(dataset_root)
    recorder_sink = EpisodeTraceRecorderSink(
        runtime,
        lifecycle_owner_id="canonical-gui-recorder",
        flush_to_durable=durable_sink.flush_pending,
        uuid_factory=lambda: "episode-external-bound",
        manifest_store=store,
        committed_capabilities=("control_trace_core", "native_streams"),
        native_stream_writer_factory=lambda prepared: NativeStreamEpisodeWriter(
            dataset_root,
            episode_uuid=prepared.episode_uuid,
            legacy_episode_index=prepared.legacy_episode_index,
            contracts={contract.stream_id: contract},
            external_payload_stream_ids=(contract.stream_id,),
        ),
    )
    try:
        prepared = recorder_sink.prepare_episode(
            legacy_episode_index=11,
            prepared_at=1.0,
        )
        writer = recorder_sink.native_stream_capture(prepared)
        payload = np.arange(6, dtype=np.float32)
        intent = NativeStreamExternalSampleIntent.create(
            contract=contract,
            sample_id="q6-external-0",
            sequence=0,
            measured_at=1.1,
            measured_clock_domain_id="local_monotonic",
            available_at=1.2,
            available_clock_domain_id="local_monotonic",
            source_payload_object_id="shared-slot:q6-external-0",
            payload=payload,
        )
        writer.stage_external_sample(intent)
        recorder_sink.bind_prepared_episode(prepared, bound_at=2.0)
        released = recorder_sink.release_episode(
            prepared,
            unbound_at=3.0,
            reason="legacy_finalize_requested",
        )
        assert released.native_stream_pending_finalize
        assert released.native_stream_manifest is None
        with pytest.raises(RuntimeError, match="frozen"):
            writer.stage_external_sample(intent)

        artifact_bytes = b"fake-specialized-writer-artifact"
        relative_path = "videos/chunk-000/episode_000011.fake"
        artifact_path = dataset_root / relative_path
        artifact_path.parent.mkdir(parents=True)
        artifact_path.write_bytes(artifact_bytes)
        receipt = NativeStreamExternalArtifactReceipt(
            episode_uuid=prepared.episode_uuid,
            legacy_episode_index=prepared.legacy_episode_index,
            stream_id=contract.stream_id,
            sample_id=intent.sample_id,
            payload_content_hash=intent.payload_content_hash,
            stored_payload_object_id=(
                f"dataset:{relative_path}#frame=0"
            ),
            artifact_relative_path=relative_path,
            artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
            artifact_byte_count=len(artifact_bytes),
            encoding=contract.encoding,
            compression=contract.compression,
            durable_at=3.5,
            durable_clock_domain_id="local_wall_clock",
        )
        finalized = recorder_sink.legacy_finalize_completed(
            released,
            completed_at=4.0,
            native_stream_artifact_receipts=(receipt,),
        )
        assert finalized.legacy_finalize_status == "finalized"
        assert finalized.capture_commit_status == "committed"
        assert finalized.native_stream_manifest_hash is not None
        assert finalized.capture_commit_candidate is not None
        assert (
            finalized.capture_commit_candidate
            .capability_artifact_hash_by_name["native_streams"]
            == finalized.native_stream_manifest_hash
        )
        captured = load_native_stream_episode(
            dataset_root,
            prepared.episode_uuid,
        )
        assert captured.records_by_stream[contract.stream_id][
            0
        ].stored_payload_object_id.endswith("#frame=0")
    finally:
        health.close()
        spool.close()


def test_missing_external_receipt_preserves_capture_for_post_finalize_retry(
    tmp_path: Path,
) -> None:
    runtime = _runtime(_profile(formal=True))
    dataset_root = tmp_path / "dataset"
    spool_root = tmp_path / "spool"
    spool_root.mkdir()
    spool, health, durable_sink = _sink(runtime, spool_root)
    contract = _native_q6_contract()
    recorder_sink = EpisodeTraceRecorderSink(
        runtime,
        lifecycle_owner_id="canonical-gui-recorder",
        flush_to_durable=durable_sink.flush_pending,
        uuid_factory=lambda: "episode-external-missing",
        manifest_store=EpisodeCaptureManifestStore(dataset_root),
        committed_capabilities=("control_trace_core", "native_streams"),
        native_stream_writer_factory=lambda prepared: NativeStreamEpisodeWriter(
            dataset_root,
            episode_uuid=prepared.episode_uuid,
            legacy_episode_index=prepared.legacy_episode_index,
            contracts={contract.stream_id: contract},
            external_payload_stream_ids=(contract.stream_id,),
        ),
    )
    try:
        prepared = recorder_sink.prepare_episode(
            legacy_episode_index=12,
            prepared_at=1.0,
        )
        writer = recorder_sink.native_stream_capture(prepared)
        intent = NativeStreamExternalSampleIntent.create(
            contract=contract,
            sample_id="q6-missing-0",
            sequence=0,
            measured_at=1.1,
            measured_clock_domain_id="local_monotonic",
            available_at=1.2,
            available_clock_domain_id="local_monotonic",
            source_payload_object_id="shared-slot:q6-missing-0",
            payload=np.zeros(6, dtype=np.float32),
        )
        writer.stage_external_sample(intent)
        recorder_sink.bind_prepared_episode(prepared, bound_at=2.0)
        released = recorder_sink.release_episode(
            prepared,
            unbound_at=3.0,
            reason="legacy_finalize_requested",
        )
        finalized = recorder_sink.legacy_finalize_completed(
            released,
            completed_at=4.0,
        )
        assert finalized.legacy_finalize_status == "finalized"
        assert finalized.capture_commit_candidate is None
        assert finalized.capture_commit_status == "not_eligible"
        assert finalized.message.startswith("native_streams_not_sealed:")
        assert "coverage mismatch" in finalized.native_stream_seal_error
        assert finalized.native_stream_retry_pending is True
        assert (
            EpisodeCaptureManifestStore(dataset_root).read_commit_payload(
                prepared.episode_uuid
            )
            is None
        )

        artifact_bytes = b"late-specialized-writer-artifact"
        relative_path = "videos/chunk-000/episode_000012.fake"
        artifact_path = dataset_root / relative_path
        artifact_path.parent.mkdir(parents=True)
        artifact_path.write_bytes(artifact_bytes)
        receipt = NativeStreamExternalArtifactReceipt(
            episode_uuid=prepared.episode_uuid,
            legacy_episode_index=prepared.legacy_episode_index,
            stream_id=contract.stream_id,
            sample_id=intent.sample_id,
            payload_content_hash=intent.payload_content_hash,
            stored_payload_object_id=f"dataset:{relative_path}#frame=0",
            artifact_relative_path=relative_path,
            artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
            artifact_byte_count=len(artifact_bytes),
            encoding=contract.encoding,
            compression=contract.compression,
            durable_at=4.5,
            durable_clock_domain_id="local_wall_clock",
        )
        retried = recorder_sink.legacy_finalize_completed(
            released,
            completed_at=5.0,
            native_stream_artifact_receipts=(receipt,),
        )
        assert retried.capture_commit_status == "committed"
        assert retried.native_stream_manifest_hash is not None
        assert retried.native_stream_seal_error == ""
        assert retried.native_stream_retry_pending is False
        assert (
            EpisodeCaptureManifestStore(dataset_root).read_commit_payload(
                prepared.episode_uuid
            )
            is not None
        )
    finally:
        health.close()
        spool.close()


@pytest.mark.parametrize(
    ("terminal_kind", "expected_status"),
    (
        ("failed", "failed"),
        ("discarded", "discarded"),
    ),
)
def test_external_native_capture_is_closed_when_legacy_episode_does_not_finalize(
    tmp_path: Path,
    terminal_kind: str,
    expected_status: str,
) -> None:
    runtime = _runtime(_profile(formal=True))
    dataset_root = tmp_path / terminal_kind
    spool_root = dataset_root / "spool"
    spool_root.mkdir(parents=True)
    spool, health, durable_sink = _sink(runtime, spool_root)
    contract = _native_q6_contract()
    recorder_sink = EpisodeTraceRecorderSink(
        runtime,
        lifecycle_owner_id="canonical-gui-recorder",
        flush_to_durable=durable_sink.flush_pending,
        uuid_factory=lambda: f"episode-external-{terminal_kind}",
        manifest_store=EpisodeCaptureManifestStore(dataset_root),
        committed_capabilities=("control_trace_core", "native_streams"),
        native_stream_writer_factory=lambda prepared: NativeStreamEpisodeWriter(
            dataset_root,
            episode_uuid=prepared.episode_uuid,
            legacy_episode_index=prepared.legacy_episode_index,
            contracts={contract.stream_id: contract},
            external_payload_stream_ids=(contract.stream_id,),
        ),
    )
    try:
        prepared = recorder_sink.prepare_episode(
            legacy_episode_index=13,
            prepared_at=1.0,
        )
        writer = recorder_sink.native_stream_capture(prepared)
        recorder_sink.bind_prepared_episode(prepared, bound_at=2.0)
        released = recorder_sink.release_episode(
            prepared,
            unbound_at=3.0,
            reason="legacy_finalize_requested",
        )
        assert released.native_stream_pending_finalize
        if terminal_kind == "failed":
            report = recorder_sink.legacy_finalize_failed(
                released,
                completed_at=4.0,
                error="fixture finalize failure",
            )
        else:
            report = recorder_sink.legacy_episode_discarded(
                released,
                completed_at=4.0,
            )
        assert report is not None
        assert report.legacy_finalize_status == expected_status
        assert report.capture_commit_candidate is None
        with pytest.raises(RuntimeError, match="closed"):
            writer.seal_after_legacy_finalize(
                (),
                sealed_at=5.0,
                sealed_clock_domain_id="local_monotonic",
            )
        manifest_path = (
            dataset_root
            / NATIVE_STREAM_EPISODE_RELATIVE_ROOT_V1
            / prepared.episode_uuid
            / "manifest.cjson"
        )
        assert not manifest_path.exists()
    finally:
        health.close()
        spool.close()


def test_native_stream_capability_and_writer_must_be_enabled_together() -> None:
    runtime = _runtime(_profile(formal=True))

    with pytest.raises(ValueError, match="must be enabled together"):
        EpisodeTraceRecorderSink(
            runtime,
            lifecycle_owner_id="canonical-gui-recorder",
            committed_capabilities=("control_trace_core", "native_streams"),
        )
    with pytest.raises(ValueError, match="must be enabled together"):
        EpisodeTraceRecorderSink(
            runtime,
            lifecycle_owner_id="canonical-gui-recorder",
            native_stream_writer_factory=lambda _prepared: None,
        )


def test_process_kill_after_seal_leaves_episode_explicitly_uncommitted(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    stop = context.Event()
    process = context.Process(
        target=_publish_seal_then_wait_for_process_termination,
        args=(str(tmp_path), ready, stop),
    )
    process.start()
    try:
        assert ready.wait(5.0)
        process.terminate()
        process.join(5.0)
        assert not process.is_alive()

        manifests = EpisodeCaptureManifestStore(tmp_path)
        seal = manifests.read_seal_payload("process-kill-episode")
        assert seal is not None
        assert seal["trace_sealed"] is True
        assert seal["capture_commit_eligible"] is True
        assert manifests.read_commit_payload("process-kill-episode") is None

        recovered = ControlTraceSpool(
            tmp_path / "spool/control_trace.wal",
            trace_session_id="trace-session-1",
            max_bytes=1 << 20,
            physical_preallocation=False,
        )
        try:
            assert recovered.recovery_report.recovered_sequences == (1, 2)
            assert recovered.durable_high_watermark == 2
        finally:
            recovered.close()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5.0)


def test_episode_manifest_commit_requires_matching_persisted_seal(tmp_path: Path) -> None:
    runtime = _runtime(_profile(formal=True))
    spool_root = tmp_path / "spool"
    spool_root.mkdir()
    spool, health, durable_sink = _sink(runtime, spool_root)
    recorder_sink = EpisodeTraceRecorderSink(
        runtime,
        lifecycle_owner_id="canonical-gui-recorder",
        flush_to_durable=durable_sink.flush_pending,
        uuid_factory=lambda: "episode-uuid-0",
    )
    try:
        prepared = recorder_sink.prepare_episode(legacy_episode_index=0, prepared_at=1.0)
        recorder_sink.bind_prepared_episode(prepared, bound_at=2.0)
        released = recorder_sink.release_episode(
            prepared,
            unbound_at=3.0,
            reason="legacy_finalize_requested",
        )
        finalized = recorder_sink.legacy_finalize_completed(released, completed_at=4.0)
        assert finalized.capture_commit_candidate is not None
        store = EpisodeCaptureManifestStore(tmp_path / "dataset")

        with pytest.raises(ValueError, match="before its seal"):
            store.publish_capture_commit(finalized.capture_commit_candidate)
    finally:
        health.close()
        spool.close()
