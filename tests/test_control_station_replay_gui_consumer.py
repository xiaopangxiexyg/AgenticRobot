from __future__ import annotations

import queue

import numpy as np

from control_station.sources import ReplayGuiStationConsumer
from robot_motion.action import RobotAction
from robot_motion.replay.proposal_source import create_replay_source_frame_emission
from teleop_runtime.replay.emission_fanout import ReplaySourceFrameEmissionFanout


def _emission(index: int = 0):
    action = RobotAction.bimanual_ee_pose(
        right_arm=np.asarray(
            [0.2 + 0.01 * index, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        timestamp=10.0 + index,
        source="replay",
    )
    return create_replay_source_frame_emission(
        source_path="/tmp/station-replay/episode.parquet",
        source_episode_index=0,
        group_key="episode-0",
        local_frame_index=index,
        source_action_index=index,
        source_relative_time=0.1 * index,
        action=action,
        observed_at=9.9 + index,
        target_effective_at=10.0 + index,
        effective_until=10.1 + index,
        clock_domain_id="local_monotonic",
    )


def test_station_consumer_is_explicit_start_and_preserves_emission_identity() -> None:
    observed = []

    def submit(emission):
        observed.append(emission)
        return type("Submission", (), {"source_emission": emission})()

    consumer = ReplayGuiStationConsumer(submit, queue_capacity=2)
    emission = _emission()
    consumer.emission_queue.put_nowait(emission)
    assert consumer.status().started is False

    consumer.start()
    consumer.emission_queue.join()

    assert observed == [emission]
    assert observed[0] is emission
    status = consumer.status()
    assert status.received_count == 1
    assert status.submitted_count == 1
    assert status.rejected_count == 0
    consumer.close()


def test_station_rejection_is_isolated_and_later_emission_still_runs() -> None:
    accepted = []

    def submit(emission):
        if emission.local_frame_index == 0:
            raise RuntimeError("publication barrier disabled")
        accepted.append(emission)
        return None

    consumer = ReplayGuiStationConsumer(submit, queue_capacity=2)
    consumer.start()
    consumer.emission_queue.put_nowait(_emission(0))
    second = _emission(1)
    consumer.emission_queue.put_nowait(second)
    consumer.emission_queue.join()

    status = consumer.status()
    assert accepted == [second]
    assert status.rejected_count == 1
    assert status.submitted_count == 1
    assert "publication barrier disabled" in status.last_error
    consumer.close()


def test_gui_fanout_and_station_consumer_share_one_emission_object() -> None:
    gui_shadow_queue: queue.Queue = queue.Queue(maxsize=2)
    submitted = []
    consumer = ReplayGuiStationConsumer(
        lambda emission: submitted.append(emission),
        queue_capacity=2,
    )
    consumer.start()
    fanout = ReplaySourceFrameEmissionFanout(
        {
            "gui_shadow": gui_shadow_queue,
            "control_station": consumer.emission_queue,
        }
    )
    emission = _emission()

    fanout.put_nowait(emission)
    consumer.emission_queue.join()

    assert gui_shadow_queue.get_nowait() is emission
    assert submitted == [emission]
    assert submitted[0] is emission
    consumer.close()
