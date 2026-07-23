from __future__ import annotations

import queue

import numpy as np

from robot_motion.action import RobotAction
from robot_motion.replay.proposal_source import create_replay_source_frame_emission
from teleop_runtime.replay.emission_fanout import ReplaySourceFrameEmissionFanout
from teleop_runtime.replay.session import ReplayGuiSession


def _emission():
    action = RobotAction.bimanual_ee_pose(
        right_arm=np.asarray(
            [0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        timestamp=10.0,
        source="replay",
    )
    return create_replay_source_frame_emission(
        source_path="/tmp/replay-source/episode.parquet",
        source_episode_index=0,
        group_key="episode-0",
        local_frame_index=0,
        source_action_index=0,
        source_relative_time=0.0,
        action=action,
        observed_at=9.9,
        target_effective_at=10.0,
        effective_until=10.1,
        clock_domain_id="local_monotonic",
    )


def test_fanout_delivers_the_same_immutable_emission_to_all_subscribers() -> None:
    first: queue.Queue = queue.Queue(maxsize=2)
    second: queue.Queue = queue.Queue(maxsize=2)
    fanout = ReplaySourceFrameEmissionFanout(
        {"gui_shadow": first, "control_station": second}
    )
    emission = _emission()

    fanout.put_nowait(emission)

    assert first.get_nowait() is emission
    assert second.get_nowait() is emission
    status = fanout.status()
    assert status["input_offer_count"] == 1
    assert status["subscribers"]["gui_shadow"]["offer_count"] == 1
    assert status["subscribers"]["control_station"]["offer_count"] == 1


def test_full_or_failing_branch_cannot_affect_another_subscriber() -> None:
    class FailingQueue(queue.Queue):
        def put_nowait(self, _item: object) -> None:
            raise RuntimeError("observer failed")

    full: queue.Queue = queue.Queue(maxsize=1)
    full.put_nowait(object())
    failing: queue.Queue = FailingQueue(maxsize=1)
    healthy: queue.Queue = queue.Queue(maxsize=1)
    fanout = ReplaySourceFrameEmissionFanout(
        {"full": full, "failing": failing, "healthy": healthy}
    )
    emission = _emission()

    fanout.put_nowait(emission)

    assert healthy.get_nowait() is emission
    status = fanout.status()["subscribers"]
    assert status["full"]["drop_count"] == 1
    assert status["failing"]["error_count"] == 1
    assert "observer failed" in status["failing"]["last_error"]
    assert status["healthy"]["offer_count"] == 1


def test_gui_session_can_bind_shadow_and_external_passive_observer() -> None:
    external: queue.Queue = queue.Queue(maxsize=4)
    session = ReplayGuiSession(
        replay_action_shadow=True,
        action_observer_queue=external,
    )
    try:
        state = session.state()
        assert state["replay_action_shadow"]["enabled"] is True
        assert state["replay_action_fanout"]["enabled"] is True
        assert state["replay_action_fanout"]["subscriber_count"] == 2
        assert state["execution"]["action_observer"]["enabled"] is True
        assert state["execution"]["action_observer"]["authoritative"] is False
    finally:
        session.close()


def test_fanout_rejects_unbounded_subscriber_queue() -> None:
    with np.testing.assert_raises_regex(ValueError, "bounded queue"):
        ReplaySourceFrameEmissionFanout({"unbounded": queue.Queue()})
