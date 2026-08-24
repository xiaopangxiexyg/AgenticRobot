from __future__ import annotations

import copy
import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import teleop_runtime.deploy.quest_paired_pose_evidence_store as evidence_store
from dataset_contracts.canonical_encoding import (
    canonical_bytes,
    canonical_content_hash,
    canonical_loads,
)
from dataset_contracts.quest_paired_pose_artifact import (
    QUEST_PAIRED_POSE_CAPTURE_ARTIFACT_KIND_V1,
    QUEST_PAIRED_POSE_CAPTURE_FILE_DOMAIN_V1,
    QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_ARTIFACT_KIND_V1,
    QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_DOMAIN_V1,
    QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_NAME_V1,
    QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_HASH_DOMAIN_V1,
)
from hardware.quest_dev.features.visualization.paired_pose_calibration import (
    PairedPoseFit,
)
from hardware.quest_dev.features.visualization.paired_pose_capture import (
    ManualStaticPairedPoseCapture,
    PairedPoseCaptureEvidence,
)
from hardware.quest_dev.features.visualization.transforms import (
    matrix_to_quaternion_xyzw,
)
from hardware.quest_dev.tests.test_paired_pose_calibration import synthetic_payload
from hardware.quest_dev.tests.test_paired_pose_capture import (
    _accepted_capture,
    _binding,
    _healthy_quest_cut,
    _healthy_robot_cut,
    _profile,
    _projection,
    _quest_event,
    _robot_event,
)
from runtime_artifact_paths import RuntimeArtifactLayout
from teleop_runtime.deploy.quest_paired_pose_evidence_store import (
    QuestPairedPoseEvidenceStoreError,
    fit_quest_paired_pose_evidence,
    load_quest_paired_pose_diagnostic_fit,
    load_sealed_quest_paired_pose_evidence,
    publish_quest_paired_pose_diagnostic_fit,
    publish_sealed_quest_paired_pose_evidence,
    quest_paired_pose_diagnostic_fit_path,
    quest_paired_pose_evidence_path,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _evidence() -> PairedPoseCaptureEvidence:
    return _accepted_capture()[2]


def _robot_event_at_pose(
    *,
    event_sequence: int,
    source_sequence: int,
    received_monotonic_ns: int,
    projection,
    pose: np.ndarray,
):
    event = _robot_event(
        event_sequence=event_sequence,
        source_sequence=source_sequence,
        received_monotonic_ns=received_monotonic_ns,
        projection=projection,
        x_m=float(pose[0, 3]),
    )
    component_id = projection.coordinate_binding.component_id
    pose_values = (
        *pose[:3, 3].tolist(),
        *matrix_to_quaternion_xyzw(pose[:3, :3]),
    )
    measurement = replace(
        event.publication.measurement,
        measurement_id=f"robot-diverse-measurement-{source_sequence}",
        source_measurement_hash=f"robot-diverse-source-{source_sequence}",
        measured_at=(
            -0.0 if source_sequence == 1 else event.publication.measurement.measured_at
        ),
        value_by_component={component_id: pose_values},
    )
    sample_ref = replace(
        event.publication.sample_ref,
        sample_id=measurement.measurement_id,
        measured_at=measurement.measured_at,
        payload_object_id=measurement.semantic_content_hash,
    )
    publication = replace(
        event.publication,
        measurement=measurement,
        sample_ref=sample_ref,
    )
    return replace(event, publication=publication)


def _quest_event_at_pose(
    *,
    event_sequence: int,
    sample_sequence: int,
    received_monotonic_ns: int,
    pose: np.ndarray,
):
    event = _quest_event(
        event_sequence=event_sequence,
        sample_sequence=sample_sequence,
        received_monotonic_ns=received_monotonic_ns,
        x_m=float(pose[0, 3]),
    )
    controller = replace(
        event.sample.controllers[0],
        matrix_reference_from_grip=tuple(pose.reshape(16, order="F")),
    )
    return replace(
        event,
        sample=replace(event.sample, controllers=(controller,)),
    )


def _diverse_four_window_evidence():
    payload, expected_reference, expected_mount = synthetic_payload()
    first_poses = tuple(
        np.asarray(sample["first_observation"]["matrix_reference_from_moving"])
        for sample in payload["samples"][:4]
    )
    second_poses = tuple(
        np.asarray(sample["second_observation"]["matrix_reference_from_moving"])
        for sample in payload["samples"][:4]
    )
    projection = _projection()
    binding = _binding(projection=projection)
    capture = ManualStaticPairedPoseCapture(
        profile=_profile(max_completed_windows=4),
        binding=binding,
    )
    latest_robot = None
    for index, (first_pose, second_pose) in enumerate(
        zip(first_poses, second_poses, strict=True)
    ):
        start_ns = 1_000_000_000 + index * 1_000_000_000
        robot_start = 1 + index * 2
        quest_start = index * 2
        capture.begin_window(
            window_id=f"diverse-window-{index}",
            start_received_monotonic_ns=start_ns,
            robot_start_next_event_sequence=robot_start,
            quest_start_next_event_sequence=quest_start,
        )
        for offset, receipt_offset_ns in enumerate((100_000_000, 300_000_000)):
            latest_robot = _robot_event_at_pose(
                event_sequence=robot_start + offset,
                source_sequence=robot_start + offset,
                received_monotonic_ns=start_ns + receipt_offset_ns,
                projection=projection,
                pose=first_pose,
            )
            quest = _quest_event_at_pose(
                event_sequence=quest_start + offset,
                sample_sequence=quest_start + offset + 1,
                received_monotonic_ns=start_ns + receipt_offset_ns + 5_000_000,
                pose=second_pose,
            )
            capture.ingest_robot_event(latest_robot)
            capture.ingest_quest_event(quest)
        receipt = capture.complete_window(
            end_received_monotonic_ns=start_ns + 500_000_000,
            robot_end_next_event_sequence=robot_start + 2,
            quest_end_next_event_sequence=quest_start + 2,
        )
        assert receipt is not None
    assert latest_robot is not None
    evidence = capture.seal_capture(
        robot_cut=_healthy_robot_cut(
            binding=binding,
            latest_event=latest_robot,
        ),
        quest_cut=_healthy_quest_cut(
            binding=binding,
            next_event_sequence=8,
            last_accepted_sample_sequence=8,
        ),
    )
    assert evidence is not None
    return evidence, expected_reference, expected_mount


def _rehash_outer_report(payload: dict) -> None:
    semantic_payload = copy.deepcopy(payload)
    semantic_payload.pop("semantic_content_hash", None)
    payload["semantic_content_hash"] = canonical_content_hash(
        semantic_payload,
        domain=QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_HASH_DOMAIN_V1,
    )


def test_publish_load_round_trip_binds_outer_and_dataset_hashes(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    layout = RuntimeArtifactLayout(root=tmp_path / "runtime")

    assert not layout.root.exists()
    commit = publish_sealed_quest_paired_pose_evidence(
        layout=layout,
        evidence=evidence,
    )

    assert commit.path == (
        layout.evidence_dir(kind=QUEST_PAIRED_POSE_CAPTURE_ARTIFACT_KIND_V1)
        / f"{evidence.semantic_content_hash}.cjson"
    )
    assert commit.evidence_semantic_content_hash == evidence.semantic_content_hash
    assert commit.dataset_semantic_sha256 == evidence.dataset.semantic_sha256
    assert commit.file_sha256 == hashlib.sha256(commit.path.read_bytes()).hexdigest()
    assert commit.byte_length == len(commit.path.read_bytes())
    assert commit.idempotent is False
    loaded = load_sealed_quest_paired_pose_evidence(
        layout=layout,
        evidence_semantic_content_hash=commit.evidence_semantic_content_hash,
    )
    assert loaded.as_dict() == evidence.as_dict()
    assert loaded.semantic_content_hash == evidence.semantic_content_hash
    assert loaded.dataset.semantic_sha256 == evidence.dataset.semantic_sha256
    assert (
        canonical_loads(
            commit.path.read_bytes(),
            expected_domain=QUEST_PAIRED_POSE_CAPTURE_FILE_DOMAIN_V1,
        )
        == evidence.as_dict()
    )


def test_publish_is_byte_stable_and_idempotent(tmp_path: Path) -> None:
    evidence = _evidence()
    layout = RuntimeArtifactLayout(root=tmp_path / "runtime")
    first = publish_sealed_quest_paired_pose_evidence(
        layout=layout,
        evidence=evidence,
    )
    original = first.path.read_bytes()

    second = publish_sealed_quest_paired_pose_evidence(
        layout=layout,
        evidence=evidence,
    )

    assert second.idempotent is True
    assert second.path == first.path
    assert second.file_sha256 == first.file_sha256
    assert second.path.read_bytes() == original
    assert tuple(second.path.parent.iterdir()) == (second.path,)
    assert second.path.stat().st_nlink == 1


def test_two_racing_publishers_never_overwrite_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    layout = RuntimeArtifactLayout(root=tmp_path / "runtime")
    link_barrier = threading.Barrier(2)
    real_link = evidence_store.os.link

    def racing_link(*args, **kwargs):
        link_barrier.wait(timeout=5.0)
        return real_link(*args, **kwargs)

    monkeypatch.setattr(evidence_store.os, "link", racing_link)
    with ThreadPoolExecutor(max_workers=2) as executor:
        commits = tuple(
            executor.map(
                lambda _: publish_sealed_quest_paired_pose_evidence(
                    layout=layout,
                    evidence=evidence,
                ),
                range(2),
            )
        )

    assert sorted(commit.idempotent for commit in commits) == [False, True]
    assert commits[0].path == commits[1].path
    assert commits[0].file_sha256 == commits[1].file_sha256
    assert tuple(commits[0].path.parent.iterdir()) == (commits[0].path,)
    assert commits[0].path.stat().st_nlink == 1


def test_publish_refuses_to_overwrite_truncated_artifact(tmp_path: Path) -> None:
    evidence = _evidence()
    layout = RuntimeArtifactLayout(root=tmp_path / "runtime")
    commit = publish_sealed_quest_paired_pose_evidence(
        layout=layout,
        evidence=evidence,
    )
    truncated = b"{"
    commit.path.write_bytes(truncated)

    with pytest.raises(
        QuestPairedPoseEvidenceStoreError,
        match="immutable Quest evidence differs",
    ):
        publish_sealed_quest_paired_pose_evidence(
            layout=layout,
            evidence=evidence,
        )

    assert commit.path.read_bytes() == truncated


def test_publish_rejects_symlinked_directory_or_target(tmp_path: Path) -> None:
    evidence = _evidence()
    outside = tmp_path / "outside"
    outside.mkdir()
    symlinked_layout = RuntimeArtifactLayout(root=tmp_path / "symlinked-runtime")
    symlinked_layout.root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        QuestPairedPoseEvidenceStoreError,
        match="directory path is invalid",
    ):
        publish_sealed_quest_paired_pose_evidence(
            layout=symlinked_layout,
            evidence=evidence,
        )
    assert tuple(outside.iterdir()) == ()

    target_layout = RuntimeArtifactLayout(root=tmp_path / "target-runtime")
    path = quest_paired_pose_evidence_path(
        layout=target_layout,
        evidence_semantic_content_hash=evidence.semantic_content_hash,
    )
    path.parent.mkdir(parents=True)
    outside_file = tmp_path / "outside.cjson"
    outside_file.write_bytes(b"do-not-follow")
    path.symlink_to(outside_file)

    with pytest.raises(QuestPairedPoseEvidenceStoreError):
        publish_sealed_quest_paired_pose_evidence(
            layout=target_layout,
            evidence=evidence,
        )
    assert path.is_symlink()
    assert outside_file.read_bytes() == b"do-not-follow"


def test_load_rejects_wrong_canonical_domain(tmp_path: Path) -> None:
    evidence = _evidence()
    layout = RuntimeArtifactLayout(root=tmp_path / "runtime")
    path = quest_paired_pose_evidence_path(
        layout=layout,
        evidence_semantic_content_hash=evidence.semantic_content_hash,
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical_bytes(evidence.as_dict(), domain="test.wrong"))

    with pytest.raises(
        QuestPairedPoseEvidenceStoreError,
        match="Canonical encoding domain mismatch",
    ):
        load_sealed_quest_paired_pose_evidence(
            layout=layout,
            evidence_semantic_content_hash=evidence.semantic_content_hash,
        )


def test_load_rejects_noncanonical_bytes_and_symlink(tmp_path: Path) -> None:
    evidence = _evidence()
    layout = RuntimeArtifactLayout(root=tmp_path / "noncanonical-runtime")
    path = quest_paired_pose_evidence_path(
        layout=layout,
        evidence_semantic_content_hash=evidence.semantic_content_hash,
    )
    path.parent.mkdir(parents=True)
    encoded = canonical_bytes(
        evidence.as_dict(),
        domain=QUEST_PAIRED_POSE_CAPTURE_FILE_DOMAIN_V1,
    )
    path.write_bytes(encoded.replace(b'{"domain"', b'{ "domain"', 1))

    with pytest.raises(
        QuestPairedPoseEvidenceStoreError,
        match="bytes are not canonical",
    ):
        load_sealed_quest_paired_pose_evidence(
            layout=layout,
            evidence_semantic_content_hash=evidence.semantic_content_hash,
        )

    symlink_layout = RuntimeArtifactLayout(root=tmp_path / "symlink-runtime")
    symlink_path = quest_paired_pose_evidence_path(
        layout=symlink_layout,
        evidence_semantic_content_hash=evidence.semantic_content_hash,
    )
    symlink_path.parent.mkdir(parents=True)
    target = tmp_path / "outside.cjson"
    target.write_bytes(encoded)
    symlink_path.symlink_to(target)

    with pytest.raises(QuestPairedPoseEvidenceStoreError):
        load_sealed_quest_paired_pose_evidence(
            layout=symlink_layout,
            evidence_semantic_content_hash=evidence.semantic_content_hash,
        )


def test_capture_load_rejects_symlinked_layout_component(tmp_path: Path) -> None:
    evidence = _evidence()
    real_layout = RuntimeArtifactLayout(root=tmp_path / "real-runtime")
    commit = publish_sealed_quest_paired_pose_evidence(
        layout=real_layout,
        evidence=evidence,
    )
    symlink_layout = RuntimeArtifactLayout(root=tmp_path / "symlink-runtime")
    symlink_layout.root.symlink_to(real_layout.root, target_is_directory=True)

    with pytest.raises(
        QuestPairedPoseEvidenceStoreError,
        match="directory path is invalid",
    ):
        load_sealed_quest_paired_pose_evidence(
            layout=symlink_layout,
            evidence_semantic_content_hash=commit.evidence_semantic_content_hash,
        )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFO support")
def test_load_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    evidence = _evidence()
    layout = RuntimeArtifactLayout(root=tmp_path / "fifo-runtime")
    path = quest_paired_pose_evidence_path(
        layout=layout,
        evidence_semantic_content_hash=evidence.semantic_content_hash,
    )
    path.parent.mkdir(parents=True)
    os.mkfifo(path)

    with pytest.raises(
        QuestPairedPoseEvidenceStoreError,
        match="not a regular file",
    ):
        load_sealed_quest_paired_pose_evidence(
            layout=layout,
            evidence_semantic_content_hash=evidence.semantic_content_hash,
        )


def test_load_rejects_content_under_wrong_hash_path(tmp_path: Path) -> None:
    evidence = _evidence()
    layout = RuntimeArtifactLayout(root=tmp_path / "runtime")
    commit = publish_sealed_quest_paired_pose_evidence(
        layout=layout,
        evidence=evidence,
    )
    wrong_hash = "0" * 64
    assert wrong_hash != evidence.semantic_content_hash
    wrong_path = quest_paired_pose_evidence_path(
        layout=layout,
        evidence_semantic_content_hash=wrong_hash,
    )
    wrong_path.write_bytes(commit.path.read_bytes())

    with pytest.raises(
        QuestPairedPoseEvidenceStoreError,
        match="path does not bind",
    ):
        load_sealed_quest_paired_pose_evidence(
            layout=layout,
            evidence_semantic_content_hash=wrong_hash,
        )


@pytest.mark.parametrize(
    "digest",
    ("a" * 63, "A" * 64, "sha256:" + "a" * 64, "../" + "a" * 64),
)
def test_path_rejects_noncanonical_sha256_digest(
    tmp_path: Path,
    digest: str,
) -> None:
    layout = RuntimeArtifactLayout(root=tmp_path / "runtime")

    with pytest.raises(ValueError, match="64 lowercase SHA-256 hex chars"):
        quest_paired_pose_evidence_path(
            layout=layout,
            evidence_semantic_content_hash=digest,
        )

    assert not layout.root.exists()


def test_diagnostic_fit_preserves_existing_minimum_sample_gate() -> None:
    evidence = _evidence()

    with pytest.raises(ValueError, match="at least 3 paired poses"):
        fit_quest_paired_pose_evidence(evidence)


def test_diagnostic_fit_runs_real_solver_on_diverse_capture_evidence() -> None:
    evidence, expected_reference, expected_mount = _diverse_four_window_evidence()

    report = fit_quest_paired_pose_evidence(evidence)

    assert report.evidence_semantic_content_hash == evidence.semantic_content_hash
    assert report.dataset_semantic_sha256 == evidence.dataset.semantic_sha256
    assert report.fit.sample_count == 4
    np.testing.assert_allclose(
        report.fit.second_reference_from_first_reference,
        expected_reference,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        report.fit.first_moving_from_second_moving,
        expected_mount,
        atol=1e-9,
    )
    assert report.fit.translation_max_m < 1e-9
    assert report.fit.rotation_max_deg < 1e-6


def test_offline_synthetic_capture_fit_report_publish_load_chain(
    tmp_path: Path,
) -> None:
    evidence, expected_reference, expected_mount = _diverse_four_window_evidence()
    layout = RuntimeArtifactLayout(root=tmp_path / "runtime")
    capture_commit = publish_sealed_quest_paired_pose_evidence(
        layout=layout,
        evidence=evidence,
    )
    sealed = load_sealed_quest_paired_pose_evidence(
        layout=layout,
        evidence_semantic_content_hash=capture_commit.evidence_semantic_content_hash,
    )

    diagnostic = fit_quest_paired_pose_evidence(sealed)

    assert not layout.derived_root.exists()
    commit = publish_quest_paired_pose_diagnostic_fit(
        layout=layout,
        diagnostic_fit=diagnostic,
    )
    assert commit.path == (
        layout.derived_dir(
            kind=QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_ARTIFACT_KIND_V1,
            artifact_id=diagnostic.semantic_content_hash,
        )
        / QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_NAME_V1
    )
    assert (
        commit.diagnostic_fit_semantic_content_hash == diagnostic.semantic_content_hash
    )
    assert commit.evidence_semantic_content_hash == evidence.semantic_content_hash
    assert commit.dataset_semantic_sha256 == evidence.dataset.semantic_sha256
    assert commit.fit_semantic_sha256 == diagnostic.fit.as_dict()["fit_semantic_sha256"]
    assert commit.file_sha256 == hashlib.sha256(commit.path.read_bytes()).hexdigest()
    assert commit.byte_length == len(commit.path.read_bytes())
    assert commit.idempotent is False

    loaded = load_quest_paired_pose_diagnostic_fit(
        layout=layout,
        diagnostic_fit_semantic_content_hash=(
            commit.diagnostic_fit_semantic_content_hash
        ),
    )
    assert loaded.as_dict() == diagnostic.as_dict()
    assert loaded.as_dict()["candidate_only"] is True
    assert loaded.as_dict()["calibration_accepted"] is None
    assert loaded.fit.as_dict()["accepted"] is None
    np.testing.assert_allclose(
        loaded.fit.second_reference_from_first_reference,
        expected_reference,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        loaded.fit.first_moving_from_second_moving,
        expected_mount,
        atol=1e-9,
    )
    assert (
        canonical_loads(
            commit.path.read_bytes(),
            expected_domain=QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_DOMAIN_V1,
        )
        == diagnostic.as_dict()
    )


def test_diagnostic_report_publish_is_explicit_and_load_is_standalone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, _, _ = _diverse_four_window_evidence()
    layout = RuntimeArtifactLayout(root=tmp_path / "runtime")
    diagnostic = fit_quest_paired_pose_evidence(evidence)

    assert not layout.root.exists()
    commit = publish_quest_paired_pose_diagnostic_fit(
        layout=layout,
        diagnostic_fit=diagnostic,
    )
    assert not layout.evidence_root.exists()

    def fail_if_solver_runs(_dataset):
        raise AssertionError("strict report load must not rerun the solver")

    monkeypatch.setattr(
        evidence_store,
        "fit_paired_pose_calibration",
        fail_if_solver_runs,
    )
    loaded = load_quest_paired_pose_diagnostic_fit(
        layout=layout,
        diagnostic_fit_semantic_content_hash=(
            commit.diagnostic_fit_semantic_content_hash
        ),
    )
    assert loaded.as_dict() == diagnostic.as_dict()


def test_diagnostic_report_publish_is_byte_stable_and_idempotent(
    tmp_path: Path,
) -> None:
    evidence, _, _ = _diverse_four_window_evidence()
    diagnostic = fit_quest_paired_pose_evidence(evidence)
    layout = RuntimeArtifactLayout(root=tmp_path / "runtime")
    first = publish_quest_paired_pose_diagnostic_fit(
        layout=layout,
        diagnostic_fit=diagnostic,
    )
    original = first.path.read_bytes()

    second = publish_quest_paired_pose_diagnostic_fit(
        layout=layout,
        diagnostic_fit=diagnostic,
    )

    assert second.idempotent is True
    assert second.path == first.path
    assert second.file_sha256 == first.file_sha256
    assert second.path.read_bytes() == original
    assert tuple(second.path.parent.iterdir()) == (second.path,)
    assert second.path.stat().st_nlink == 1


def test_two_racing_diagnostic_report_publishers_never_overwrite_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, _, _ = _diverse_four_window_evidence()
    diagnostic = fit_quest_paired_pose_evidence(evidence)
    layout = RuntimeArtifactLayout(root=tmp_path / "runtime")
    link_barrier = threading.Barrier(2)
    real_link = evidence_store.os.link

    def racing_link(*args, **kwargs):
        link_barrier.wait(timeout=5.0)
        return real_link(*args, **kwargs)

    monkeypatch.setattr(evidence_store.os, "link", racing_link)
    with ThreadPoolExecutor(max_workers=2) as executor:
        commits = tuple(
            executor.map(
                lambda _: publish_quest_paired_pose_diagnostic_fit(
                    layout=layout,
                    diagnostic_fit=diagnostic,
                ),
                range(2),
            )
        )

    assert sorted(commit.idempotent for commit in commits) == [False, True]
    assert commits[0].path == commits[1].path
    assert commits[0].file_sha256 == commits[1].file_sha256
    assert tuple(commits[0].path.parent.iterdir()) == (commits[0].path,)
    assert commits[0].path.stat().st_nlink == 1


def test_diagnostic_report_publish_refuses_existing_different_content(
    tmp_path: Path,
) -> None:
    evidence, _, _ = _diverse_four_window_evidence()
    diagnostic = fit_quest_paired_pose_evidence(evidence)
    layout = RuntimeArtifactLayout(root=tmp_path / "runtime")
    commit = publish_quest_paired_pose_diagnostic_fit(
        layout=layout,
        diagnostic_fit=diagnostic,
    )
    truncated = b"{"
    commit.path.write_bytes(truncated)

    with pytest.raises(
        QuestPairedPoseEvidenceStoreError,
        match="immutable Quest diagnostic fit report differs",
    ):
        publish_quest_paired_pose_diagnostic_fit(
            layout=layout,
            diagnostic_fit=diagnostic,
        )

    assert commit.path.read_bytes() == truncated


@pytest.mark.parametrize(
    "digest",
    ("a" * 63, "A" * 64, "sha256:" + "a" * 64, "../" + "a" * 64),
)
def test_diagnostic_report_path_rejects_noncanonical_digest_without_io(
    tmp_path: Path,
    digest: str,
) -> None:
    layout = RuntimeArtifactLayout(root=tmp_path / "runtime")

    with pytest.raises(ValueError, match="64 lowercase SHA-256 hex chars"):
        quest_paired_pose_diagnostic_fit_path(
            layout=layout,
            diagnostic_fit_semantic_content_hash=digest,
        )

    assert not layout.root.exists()


def test_diagnostic_report_load_rejects_wrong_domain_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    evidence, _, _ = _diverse_four_window_evidence()
    diagnostic = fit_quest_paired_pose_evidence(evidence)
    layout = RuntimeArtifactLayout(root=tmp_path / "wrong-domain")
    path = quest_paired_pose_diagnostic_fit_path(
        layout=layout,
        diagnostic_fit_semantic_content_hash=diagnostic.semantic_content_hash,
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(
        canonical_bytes(
            diagnostic.as_dict(),
            domain=QUEST_PAIRED_POSE_CAPTURE_FILE_DOMAIN_V1,
        )
    )
    with pytest.raises(
        QuestPairedPoseEvidenceStoreError,
        match="Canonical encoding domain mismatch",
    ):
        load_quest_paired_pose_diagnostic_fit(
            layout=layout,
            diagnostic_fit_semantic_content_hash=diagnostic.semantic_content_hash,
        )

    layout = RuntimeArtifactLayout(root=tmp_path / "noncanonical")
    path = quest_paired_pose_diagnostic_fit_path(
        layout=layout,
        diagnostic_fit_semantic_content_hash=diagnostic.semantic_content_hash,
    )
    path.parent.mkdir(parents=True)
    encoded = canonical_bytes(
        diagnostic.as_dict(),
        domain=QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_DOMAIN_V1,
    )
    path.write_bytes(encoded.replace(b'{"domain"', b'{ "domain"', 1))
    with pytest.raises(
        QuestPairedPoseEvidenceStoreError,
        match="bytes are not canonical",
    ):
        load_quest_paired_pose_diagnostic_fit(
            layout=layout,
            diagnostic_fit_semantic_content_hash=diagnostic.semantic_content_hash,
        )


def test_diagnostic_report_load_rejects_path_and_payload_tampering(
    tmp_path: Path,
) -> None:
    evidence, _, _ = _diverse_four_window_evidence()
    diagnostic = fit_quest_paired_pose_evidence(evidence)

    wrong_path_layout = RuntimeArtifactLayout(root=tmp_path / "wrong-path")
    wrong_hash = "0" * 64
    assert wrong_hash != diagnostic.semantic_content_hash
    wrong_path = quest_paired_pose_diagnostic_fit_path(
        layout=wrong_path_layout,
        diagnostic_fit_semantic_content_hash=wrong_hash,
    )
    wrong_path.parent.mkdir(parents=True)
    wrong_path.write_bytes(
        canonical_bytes(
            diagnostic.as_dict(),
            domain=QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_DOMAIN_V1,
        )
    )
    with pytest.raises(
        QuestPairedPoseEvidenceStoreError,
        match="path does not bind",
    ):
        load_quest_paired_pose_diagnostic_fit(
            layout=wrong_path_layout,
            diagnostic_fit_semantic_content_hash=wrong_hash,
        )

    relation_layout = RuntimeArtifactLayout(root=tmp_path / "relation")
    relation_payload = diagnostic.as_dict()
    relation_payload["dataset_semantic_sha256"] = "f" * 64
    _rehash_outer_report(relation_payload)
    relation_path = quest_paired_pose_diagnostic_fit_path(
        layout=relation_layout,
        diagnostic_fit_semantic_content_hash=relation_payload["semantic_content_hash"],
    )
    relation_path.parent.mkdir(parents=True)
    relation_path.write_bytes(
        canonical_bytes(
            relation_payload,
            domain=QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_DOMAIN_V1,
        )
    )
    with pytest.raises(
        QuestPairedPoseEvidenceStoreError,
        match="dataset hash differs",
    ):
        load_quest_paired_pose_diagnostic_fit(
            layout=relation_layout,
            diagnostic_fit_semantic_content_hash=relation_payload[
                "semantic_content_hash"
            ],
        )

    candidate_layout = RuntimeArtifactLayout(root=tmp_path / "candidate")
    candidate_payload = diagnostic.as_dict()
    candidate_payload["candidate_only"] = False
    _rehash_outer_report(candidate_payload)
    candidate_path = quest_paired_pose_diagnostic_fit_path(
        layout=candidate_layout,
        diagnostic_fit_semantic_content_hash=candidate_payload["semantic_content_hash"],
    )
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_bytes(
        canonical_bytes(
            candidate_payload,
            domain=QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_DOMAIN_V1,
        )
    )
    with pytest.raises(
        QuestPairedPoseEvidenceStoreError,
        match="candidate_only",
    ):
        load_quest_paired_pose_diagnostic_fit(
            layout=candidate_layout,
            diagnostic_fit_semantic_content_hash=candidate_payload[
                "semantic_content_hash"
            ],
        )


def test_diagnostic_report_load_rejects_outer_contract_drift(tmp_path: Path) -> None:
    evidence, _, _ = _diverse_four_window_evidence()
    diagnostic = fit_quest_paired_pose_evidence(evidence)
    payloads: list[dict] = []

    changed = diagnostic.as_dict()
    changed["schema"] = "wrong.schema"
    _rehash_outer_report(changed)
    payloads.append(changed)

    changed = diagnostic.as_dict()
    changed["calibration_accepted"] = True
    _rehash_outer_report(changed)
    payloads.append(changed)

    changed = diagnostic.as_dict()
    changed.pop("calibration_accepted")
    _rehash_outer_report(changed)
    payloads.append(changed)

    changed = diagnostic.as_dict()
    changed["annotation"] = "not in the contract"
    _rehash_outer_report(changed)
    payloads.append(changed)

    changed = diagnostic.as_dict()
    changed["semantic_content_hash"] = "0" * 64
    assert changed["semantic_content_hash"] != diagnostic.semantic_content_hash
    payloads.append(changed)

    for index, payload in enumerate(payloads):
        layout = RuntimeArtifactLayout(root=tmp_path / f"outer-{index}")
        path = quest_paired_pose_diagnostic_fit_path(
            layout=layout,
            diagnostic_fit_semantic_content_hash=payload["semantic_content_hash"],
        )
        path.parent.mkdir(parents=True)
        path.write_bytes(
            canonical_bytes(
                payload,
                domain=QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_FILE_DOMAIN_V1,
            )
        )
        with pytest.raises(QuestPairedPoseEvidenceStoreError):
            load_quest_paired_pose_diagnostic_fit(
                layout=layout,
                diagnostic_fit_semantic_content_hash=payload["semantic_content_hash"],
            )


def test_diagnostic_report_rejects_symlink_and_fifo(tmp_path: Path) -> None:
    evidence, _, _ = _diverse_four_window_evidence()
    diagnostic = fit_quest_paired_pose_evidence(evidence)
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_layout = RuntimeArtifactLayout(root=tmp_path / "symlink-runtime")
    symlink_layout.root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(
        QuestPairedPoseEvidenceStoreError,
        match="directory path is invalid",
    ):
        publish_quest_paired_pose_diagnostic_fit(
            layout=symlink_layout,
            diagnostic_fit=diagnostic,
        )
    assert tuple(outside.iterdir()) == ()

    real_layout = RuntimeArtifactLayout(root=tmp_path / "real-runtime")
    real_commit = publish_quest_paired_pose_diagnostic_fit(
        layout=real_layout,
        diagnostic_fit=diagnostic,
    )
    load_symlink_layout = RuntimeArtifactLayout(root=tmp_path / "load-symlink")
    load_symlink_layout.root.symlink_to(real_layout.root, target_is_directory=True)
    with pytest.raises(
        QuestPairedPoseEvidenceStoreError,
        match="directory path is invalid",
    ):
        load_quest_paired_pose_diagnostic_fit(
            layout=load_symlink_layout,
            diagnostic_fit_semantic_content_hash=(
                real_commit.diagnostic_fit_semantic_content_hash
            ),
        )

    target_layout = RuntimeArtifactLayout(root=tmp_path / "target-runtime")
    target_path = quest_paired_pose_diagnostic_fit_path(
        layout=target_layout,
        diagnostic_fit_semantic_content_hash=diagnostic.semantic_content_hash,
    )
    target_path.parent.mkdir(parents=True)
    outside_file = tmp_path / "outside-report.cjson"
    outside_file.write_bytes(real_commit.path.read_bytes())
    target_path.symlink_to(outside_file)
    with pytest.raises(QuestPairedPoseEvidenceStoreError):
        load_quest_paired_pose_diagnostic_fit(
            layout=target_layout,
            diagnostic_fit_semantic_content_hash=diagnostic.semantic_content_hash,
        )

    if not hasattr(os, "mkfifo"):
        pytest.skip("requires POSIX FIFO support")
    fifo_layout = RuntimeArtifactLayout(root=tmp_path / "fifo-runtime")
    fifo_path = quest_paired_pose_diagnostic_fit_path(
        layout=fifo_layout,
        diagnostic_fit_semantic_content_hash=diagnostic.semantic_content_hash,
    )
    fifo_path.parent.mkdir(parents=True)
    os.mkfifo(fifo_path)
    with pytest.raises(
        QuestPairedPoseEvidenceStoreError,
        match="not a regular file",
    ):
        load_quest_paired_pose_diagnostic_fit(
            layout=fifo_layout,
            diagnostic_fit_semantic_content_hash=diagnostic.semantic_content_hash,
        )


def test_diagnostic_fit_binds_capture_and_dataset_without_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    observed_datasets = []

    def fake_fit(dataset):
        observed_datasets.append(dataset)
        return PairedPoseFit(
            dataset_sha256=dataset.semantic_sha256,
            frames=dataset.frames,
            tool_profile_hash=dataset.tool_profile_hash,
            second_reference_from_first_reference=np.eye(4),
            first_moving_from_second_moving=np.eye(4),
            translation_residuals_m=np.zeros(1),
            rotation_residuals_deg=np.zeros(1),
            translation_rms_m=0.0,
            translation_max_m=0.0,
            rotation_rms_deg=0.0,
            rotation_max_deg=0.0,
            sample_count=1,
            observability={"relative_pair_count": 0},
        )

    monkeypatch.setattr(evidence_store, "fit_paired_pose_calibration", fake_fit)
    report = fit_quest_paired_pose_evidence(evidence)
    payload = report.as_dict()

    assert len(observed_datasets) == 1
    assert observed_datasets[0].as_dict() == evidence.dataset.as_dict()
    assert observed_datasets[0] is not evidence.dataset
    assert report.evidence_semantic_content_hash == evidence.semantic_content_hash
    assert report.dataset_semantic_sha256 == evidence.dataset.semantic_sha256
    assert payload["fit"] == report.fit.as_dict()
    assert payload["candidate_only"] is True
    assert payload["calibration_accepted"] is None
    semantic_hash = payload.pop("semantic_content_hash")
    assert semantic_hash == canonical_content_hash(
        payload,
        domain=QUEST_PAIRED_POSE_DIAGNOSTIC_FIT_HASH_DOMAIN_V1,
    )

    layout = RuntimeArtifactLayout(root=tmp_path / "runtime")
    with pytest.raises(ValueError, match="sample_count must be at least 3"):
        publish_quest_paired_pose_diagnostic_fit(
            layout=layout,
            diagnostic_fit=report,
        )
    assert not layout.root.exists()
