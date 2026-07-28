from __future__ import annotations

from collections import Counter

from teleop_runtime.collection.umi.recorder import UmiDatasetRecorder


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


def test_umi_owner_passes_one_identity_through_sidecar_lifecycle() -> None:
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
