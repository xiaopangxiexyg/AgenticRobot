from __future__ import annotations

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dataset_contracts.v22 import PolicyIOLayout
from teleop_runtime.collection import v25p_dual_write_shadow as shadow_module
from teleop_runtime.collection.operator_labels import OperatorLabelStateMachine
from teleop_runtime.collection.umi.recorder import UmiDatasetRecorder
from teleop_runtime.collection.umi.types import UmiSample


class _Writer:
    next_episode_index = 3

    def __init__(self) -> None:
        self.metadata = None
        self.finalized = []
        self.discarded = []

    def start_episode(self, *, metadata):
        self.metadata = dict(metadata)
        return self.next_episode_index

    def finalize_episode(self, episode_index, *, summary):
        self.finalized.append((int(episode_index), dict(summary)))

    def discard_episode(self, episode_index):
        self.discarded.append(int(episode_index))


class _Pool:
    def __init__(self) -> None:
        self.started = False
        self.joined = False

    def start(self) -> None:
        self.started = True

    def join(self) -> None:
        self.joined = True


class _Store:
    def __init__(self) -> None:
        self.prepared = None
        self.bound = None
        self.finalized = None

    def prepare_episode(self, identity) -> None:
        self.prepared = identity

    def bind_prepared_episode(self, identity, **_kwargs) -> None:
        assert identity is self.prepared
        self.bound = identity

    def finalize_after_legacy_episode(self, identity, **_kwargs) -> None:
        assert identity is self.bound
        self.finalized = identity

    def discard_episode(self, identity) -> None:
        assert identity is self.prepared

    def close_unsealed(self, identity) -> None:
        assert identity is self.prepared


def _new_recorder() -> UmiDatasetRecorder:
    recorder = UmiDatasetRecorder.__new__(UmiDatasetRecorder)
    writer = _Writer()
    pool = _Pool()
    store = _Store()
    recorder.writer = writer
    recorder.pool = pool
    recorder.encoder = type(
        "Encoder",
        (),
        {"reset_episode": lambda _self, _episode_index: None},
    )()
    recorder.v25p_sidecar_store = store
    recorder._episode_uuid_factory = lambda: "umi-owner-uuid"
    recorder._active_episode_identity = None
    recorder.episode_index = None
    recorder.reload_finalized_dataset_state = lambda: None
    recorder.action_alignment_max_gap_sec = 0.2
    recorder.action_continuity_break_reasons = Counter()
    recorder.rejection_counts = Counter()
    return recorder


def test_umi_owner_passes_one_identity_through_sidecar_lifecycle() -> None:
    recorder = _new_recorder()
    writer, pool, store = recorder.writer, recorder.pool, recorder.v25p_sidecar_store

    episode_index = recorder.start_episode({"source": "test"})

    assert episode_index == 3
    assert store.prepared is store.bound
    assert recorder.active_episode_identity is store.prepared
    assert writer.metadata["source"] == "test"
    assert writer.metadata["v25p_episode"]["episode_uuid"] == "umi-owner-uuid"

    recorder.frame_count = 1
    recorder.accepted_sample_count = 1
    recorder.stop_episode(reason="test")

    assert pool.started is True
    assert pool.joined is True
    assert store.finalized is store.prepared
    assert recorder.active_episode_identity is None
    assert writer.finalized[0][0] == 3


@pytest.mark.parametrize("append_fails", [False, True])
def test_umi_native_shadow_observes_aligned_sample_without_replanning(append_fails) -> None:
    recorder = _new_recorder()
    recorder.policy_io_layout = PolicyIOLayout(name="test_no_end_effector")
    recorder.v25p_native_shadow = shadow = Mock()
    episode_index = recorder.start_episode()
    row = {"frame_index": 0}
    jobs = (SimpleNamespace(kind="lerobot_v21.row", payload={"row": row}),)
    recorder.writer.plan_sample_jobs = Mock(return_value=jobs)
    recorder.pool.submit = Mock()
    if append_fails:
        shadow.append.side_effect = RuntimeError("shadow unavailable")
    sample = UmiSample(
        episode_index=episode_index, frame_index=0, timestamp=10.0,
        poses={"left": object(), "right": object()},
        gripper_commands={"left": 0.1, "right": 0.9}, hand_states={},
        operator_labels=OperatorLabelStateMachine().snapshot(),
    )
    next_sample = replace(
        sample, frame_index=1, timestamp=10.1,
        poses={"left": object(), "right": object()},
    )

    assert recorder.submit(sample)
    shadow.append.assert_not_called()
    assert recorder.submit(next_sample)

    recorder.writer.plan_sample_jobs.assert_called_once()
    aligned = recorder.writer.plan_sample_jobs.call_args.args[0]
    assert aligned.poses is sample.poses
    assert aligned.arm_action_poses is next_sample.poses
    assert aligned.arm_action_source_frame_index == 1
    shadow.append.assert_called_once_with(
        recorder.active_episode_identity, source_sample=aligned, legacy_row=row,
    )
    assert shadow.append.call_args.kwargs["source_sample"] is aligned
    recorder.pool.submit.assert_called_once_with(jobs[0])
    assert recorder.frame_count == 1
    assert recorder._v25p_native_shadow_active is not append_fails
    if append_fails:
        assert recorder._v25p_native_shadow_error == "append:RuntimeError: shadow unavailable"
        shadow.close_unsealed.assert_called_once_with(recorder.active_episode_identity)
        assert recorder.submit(replace(next_sample, frame_index=2, timestamp=10.2))
        assert recorder.frame_count == 2
        assert shadow.append.call_count == 1


@pytest.mark.parametrize("outcome", ["complete", "failed", "raises", "discard"])
def test_umi_native_finalize_keeps_saved_outcome_and_clears_active_state(
    tmp_path, monkeypatch, outcome,
) -> None:
    recorder = _new_recorder()
    recorder.writer.root_dir = tmp_path
    recorder.v25p_native_shadow = shadow = Mock()
    episode_index = recorder.start_episode()
    identity = recorder.active_episode_identity
    shadow.active_identity = identity
    recorder.frame_count = recorder.accepted_sample_count = 1
    report = SimpleNamespace(receipt=SimpleNamespace(status=outcome, message=""))

    def finalize(active_identity, **_kwargs):
        assert active_identity is identity
        assert recorder.writer.finalized[0][0] == episode_index
        if outcome == "raises":
            raise RuntimeError("seal failed")
        return report

    shadow.finalize_after_legacy.side_effect = finalize
    monkeypatch.setattr(
        shadow_module, "load_finalized_legacy_episode_record",
        Mock(return_value={"episode_index": episode_index}),
    )

    assert recorder.stop_episode(reason="test", discard=outcome == "discard") == episode_index

    assert recorder.last_outcome == ("discarded" if outcome == "discard" else "saved")
    assert recorder._v25p_native_shadow_active is False
    assert recorder.active_episode_identity is None
    assert shadow.finalize_after_legacy.call_count == int(outcome != "discard")
    assert shadow.close_unsealed.call_count == int(outcome == "raises")
    assert shadow.discard_episode.call_count == int(outcome == "discard")
    if outcome == "raises":
        assert recorder._v25p_native_shadow_error == "finalize:RuntimeError: seal failed"
        assert recorder.last_message.endswith(
            "; v2.5-p native shadow incomplete: finalize:RuntimeError: seal failed"
        )
    else:
        assert recorder._v25p_native_shadow_error == ""
