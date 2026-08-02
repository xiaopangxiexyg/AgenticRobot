from __future__ import annotations

from types import SimpleNamespace

import pytest

from collection_runtime.sample_acceptance import (
    AcceptedSampleReceipt,
    SampleAcceptanceReceiptBuffer,
)
from teleop_runtime.real_teleop.dataset_recorder import RealTeleopDatasetRecorder


def test_accepted_sample_receipt_validates_exact_coordinate() -> None:
    receipt = AcceptedSampleReceipt(
        episode_index=2,
        sample_index=19,
        episode_frame_index=3,
        sample_timestamp=12.5,
    )

    assert receipt.episode_index == 2
    assert receipt.sample_index == 19
    assert receipt.episode_frame_index == 3
    assert receipt.sample_timestamp == 12.5

    with pytest.raises(ValueError, match="sample_index"):
        AcceptedSampleReceipt(
            episode_index=2,
            sample_index=-1,
            episode_frame_index=3,
            sample_timestamp=12.5,
        )
    with pytest.raises(ValueError, match="sample_timestamp"):
        AcceptedSampleReceipt(
            episode_index=2,
            sample_index=19,
            episode_frame_index=3,
            sample_timestamp=float("nan"),
        )


def _thread_recorder(*, capacity: int) -> RealTeleopDatasetRecorder:
    recorder = RealTeleopDatasetRecorder.__new__(RealTeleopDatasetRecorder)
    recorder.collection_worker_mode = "thread"
    recorder._sample_acceptance_receipt_buffer = SampleAcceptanceReceiptBuffer(
        capacity=capacity
    )
    return recorder


def test_thread_writer_receipt_matches_returned_sample() -> None:
    recorder = _thread_recorder(capacity=4)

    recorder._record_thread_sample_acceptance(
        SimpleNamespace(
            episode_index=7,
            index=101,
            frame_index=11,
            timestamp=42.25,
        )
    )

    assert recorder.drain_accepted_sample_receipts() == (
        AcceptedSampleReceipt(
            episode_index=7,
            sample_index=101,
            episode_frame_index=11,
            sample_timestamp=42.25,
        ),
    )
    assert recorder.drain_accepted_sample_receipts() == ()


def test_sample_acceptance_overflow_fails_closed() -> None:
    recorder = _thread_recorder(capacity=1)
    for sample_index in (3, 4):
        recorder._record_thread_sample_acceptance(
            SimpleNamespace(
                episode_index=0,
                index=sample_index,
                frame_index=sample_index,
                timestamp=10.0 + sample_index,
            )
        )

    with pytest.raises(RuntimeError, match="overflowed"):
        recorder.drain_accepted_sample_receipts()
    assert recorder.drain_accepted_sample_receipts() == ()


def test_disabled_thread_receipts_do_not_inspect_sample() -> None:
    recorder = _thread_recorder(capacity=0)

    class _MustNotBeRead:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"disabled receipt path read {name}")

    recorder._record_thread_sample_acceptance(_MustNotBeRead())
    assert recorder.drain_accepted_sample_receipts() == ()
