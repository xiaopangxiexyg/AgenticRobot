from __future__ import annotations

import numpy as np
import pytest

from control_station.sources import (
    PassiveHumanCommandEmissionIngress,
    PassiveHumanCommandObserverFanout,
    SyntheticHumanQposSource,
    SyntheticHumanQposSourceSpec,
)


def _source() -> SyntheticHumanQposSource:
    return SyntheticHumanQposSource(
        SyntheticHumanQposSourceSpec(source_id="human-ingress-test")
    )


def _emission(source: SyntheticHumanQposSource, emitted_at: float):
    return source.emit(
        {
            "left_arm": np.full(6, emitted_at / 1000.0, dtype=float),
            "right_arm": np.full(6, -emitted_at / 1000.0, dtype=float),
        },
        emitted_at=emitted_at,
    )


def test_human_ingress_is_bounded_latest_preserving_and_identity_exact() -> None:
    source = _source()
    ingress = PassiveHumanCommandEmissionIngress(
        "canonical-gui-human-emission-v1",
        queue_capacity=1,
    )
    stale = _emission(source, 10.0)
    fresh = _emission(source, 11.0)

    assert ingress.observe_emission(stale)
    assert ingress.observe_emission(fresh)
    acquired = ingress.wait_for_post_boundary_emission(
        boundary_at=10.5,
        timeout_sec=0.1,
    )

    assert acquired is fresh
    status = ingress.status()
    assert status.observed_count == 2
    assert status.accepted_count == 2
    assert status.evicted_count == 1
    assert status.acquired_count == 1
    assert status.queue_size == 0


def test_human_ingress_discards_pre_boundary_emission_and_times_out() -> None:
    source = _source()
    ingress = PassiveHumanCommandEmissionIngress("human-emission-time-gate")
    ingress.observe_emission(_emission(source, 9.0))

    with pytest.raises(TimeoutError, match="post-boundary"):
        ingress.wait_for_post_boundary_emission(
            boundary_at=10.0,
            timeout_sec=0.01,
        )

    status = ingress.status()
    assert status.stale_count == 1
    assert status.acquired_count == 0


def test_human_observer_fanout_preserves_identity_and_isolates_failure() -> None:
    source = _source()
    emission = _emission(source, 12.0)
    ingress = PassiveHumanCommandEmissionIngress("station-human-ingress")
    observed = []

    def failing_subscriber(candidate) -> None:
        assert candidate is emission
        raise RuntimeError("passive trace unavailable")

    fanout = PassiveHumanCommandObserverFanout(
        {
            "existing_shadow": failing_subscriber,
            "control_station": ingress.observe_emission,
            "diagnostic": lambda candidate: observed.append(candidate),
        }
    )

    assert fanout.observe_emission(emission) is False
    assert observed == [emission]
    assert ingress.wait_for_post_boundary_emission(
        boundary_at=12.0,
        timeout_sec=0.1,
    ) is emission
    status = fanout.status()
    assert status["input_offer_count"] == 1
    assert status["subscribers"]["existing_shadow"]["error_count"] == 1
    assert status["subscribers"]["control_station"]["offer_count"] == 1


def test_human_emission_ingress_is_not_a_readiness_provider() -> None:
    ingress = PassiveHumanCommandEmissionIngress("emission-only")

    assert not hasattr(ingress, "acquire")
