# Copyright (c) 2026 DexTeleop Contributors
from __future__ import annotations

from pathlib import Path

import pytest

from collection_runtime.planning_history import (
    PlanningHistoryRecord,
    PlanningHistoryRecorder,
    PlanningHistorySourceBinding,
    PlanningHistoryStoreError,
    PlanningHistoryStoreLimits,
    SQLitePlanningHistoryStore,
)


def _binding() -> PlanningHistorySourceBinding:
    return PlanningHistorySourceBinding(
        source_id="right-arm-tcp-low-rate-v1",
        runtime_session_id="runtime-session-history-1",
        clock_domain_id="runtime-monotonic-v1",
        channels=("tcp_pose",),
        capture_profile_hash="capture-profile-hash-1",
        source_contract={
            "representation": "cartesian_pose",
            "controlled_frame_id": "tool_tcp",
            "component_ids": ["right_arm"],
        },
    )


def _limits(*, max_records: int = 16) -> PlanningHistoryStoreLimits:
    return PlanningHistoryStoreLimits(
        retention_sec=60.0,
        max_records=max_records,
        max_record_payload_bytes=8_192,
        memory_buffer_max_records=3,
    )


def _record(
    binding: PlanningHistorySourceBinding,
    index: int,
    *,
    record_id: str | None = None,
    boundary_event: str | None = None,
    payload_extra: dict[str, object] | None = None,
) -> PlanningHistoryRecord:
    measured_at = 10.0 + index
    payload = {
        "pose_by_component": {"right_arm": [0.1 * index, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]},
        "measurement_id": f"measurement-{index}",
    }
    payload.update(payload_extra or {})
    return PlanningHistoryRecord(
        record_id=record_id or f"history-record-{index}",
        source_binding_hash=binding.semantic_content_hash,
        runtime_session_id=binding.runtime_session_id,
        channel="tcp_pose",
        measured_at=measured_at,
        available_at=measured_at + 0.01,
        cutoff_at=measured_at + 0.02,
        payload=payload,
        evidence_refs=(f"measurement-evidence-{index}",),
        boundary_event=boundary_event,
    )


def test_store_requires_explicit_absolute_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="root must be absolute"):
        SQLitePlanningHistoryStore(
            root=Path("relative-history"),
            binding=_binding(),
            limits=_limits(),
        )
    assert not (tmp_path / "relative-history").exists()


def test_store_is_durable_idempotent_and_binding_closed(tmp_path: Path) -> None:
    binding = _binding()
    history_root = (tmp_path / "history").resolve()
    with SQLitePlanningHistoryStore(
        root=history_root,
        binding=binding,
        limits=_limits(),
    ) as store:
        first = store.append(_record(binding, 0))
        replay = store.append(_record(binding, 0))
        assert first.accepted and first.durable and first.sequence == 1
        assert replay.idempotent and replay.sequence == first.sequence
        assert store.database_path.is_file()

    with SQLitePlanningHistoryStore(
        root=history_root,
        binding=binding,
        limits=_limits(),
    ) as reopened:
        result = reopened.query(
            slice_id="history-slice-restart",
            channels=("tcp_pose",),
            cutoff_at=10.02,
            history_window_sec=1.0,
            max_samples=4,
            max_payload_bytes=8_192,
            sampling_policy="uniform",
        )
        assert [item.sequence for item in result.records] == [1]
        assert result.records[0].record.record_id == "history-record-0"

    other_binding = PlanningHistorySourceBinding(
        source_id="another-source",
        runtime_session_id=binding.runtime_session_id,
        clock_domain_id=binding.clock_domain_id,
        channels=("tcp_pose",),
        capture_profile_hash="capture-profile-hash-2",
        source_contract={"representation": "cartesian_pose"},
    )
    with pytest.raises(PlanningHistoryStoreError, match="another source binding"):
        SQLitePlanningHistoryStore(
            root=history_root,
            binding=other_binding,
            limits=_limits(),
        )


def test_boundary_plus_uniform_is_bounded_and_content_addressed(
    tmp_path: Path,
) -> None:
    binding = _binding()
    with SQLitePlanningHistoryStore(
        root=(tmp_path / "history").resolve(),
        binding=binding,
        limits=_limits(),
    ) as store:
        for index in range(5):
            store.append(
                _record(
                    binding,
                    index,
                    boundary_event=("skill_terminal" if index in {1, 4} else None),
                )
            )
        result = store.query(
            slice_id="history-slice-bounded",
            channels=("tcp_pose",),
            cutoff_at=14.02,
            history_window_sec=5.0,
            max_samples=3,
            max_payload_bytes=8_192,
            sampling_policy="boundary_plus_uniform",
        )
        replay = store.query(
            slice_id="history-slice-bounded",
            channels=("tcp_pose",),
            cutoff_at=14.02,
            history_window_sec=5.0,
            max_samples=3,
            max_payload_bytes=8_192,
            sampling_policy="boundary_plus_uniform",
        )

    assert [item.sequence for item in result.records] == [2, 4, 5]
    assert result.truncated
    assert result.semantic_content_hash == replay.semantic_content_hash
    assert [item.record.boundary_event for item in result.records] == [
        "skill_terminal",
        None,
        "skill_terminal",
    ]


def test_store_rejects_clock_regression_and_applies_retention(tmp_path: Path) -> None:
    binding = _binding()
    with SQLitePlanningHistoryStore(
        root=(tmp_path / "history").resolve(),
        binding=binding,
        limits=_limits(max_records=2),
    ) as store:
        for index in range(3):
            store.append(_record(binding, index))
        retained = store.query(
            slice_id="history-slice-retained",
            channels=("tcp_pose",),
            cutoff_at=12.02,
            history_window_sec=5.0,
            max_samples=8,
            max_payload_bytes=8_192,
            sampling_policy="uniform",
        )
        with pytest.raises(PlanningHistoryStoreError, match="clocks must be monotonic"):
            store.append(
                _record(
                    binding,
                    1,
                    record_id="history-record-regressed-clock",
                    payload_extra={"retry": True},
                )
            )

    assert [item.sequence for item in retained.records] == [2, 3]


def test_recorder_is_default_off_and_manual_capture_is_explicit(tmp_path: Path) -> None:
    binding = _binding()
    with SQLitePlanningHistoryStore(
        root=(tmp_path / "history").resolve(),
        binding=binding,
        limits=_limits(),
    ) as store:
        calls: list[int] = []

        def capture() -> PlanningHistoryRecord:
            calls.append(len(calls))
            return _record(binding, calls[-1])

        recorder = PlanningHistoryRecorder(
            store=store,
            capture=capture,
            interval_sec=1.0,
        )
        assert not recorder.started
        assert calls == []
        receipt = recorder.capture_once()
        assert receipt is not None and receipt.accepted
        assert calls == [0]
        assert recorder.close()
