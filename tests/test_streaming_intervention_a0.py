from __future__ import annotations

import sys

import pytest

from control_composition.allocation import (
    COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
)
from experiments.sim_learning.streaming_intervention import (
    StreamingInterventionFrame,
    StreamingInterventionMockRunner,
)
from experiments.sim_learning.streaming_intervention.mock_runner import (
    MOCK_OBSERVATION_KEYS_A0,
)
from sim.robosuite import RobosuiteMixedControlAdapter, RobosuiteResetManifest


def _manifest(*, seed: int = 11, max_episode_steps: int = 3) -> RobosuiteResetManifest:
    return RobosuiteResetManifest(
        environment_id="Lift",
        environment_contract_hash="env-contract-a0",
        task_id="streaming-intervention-a0",
        episode_id="episode-0001",
        reset_sequence=0,
        seed=seed,
        action_dimension=2,
        observation_keys=MOCK_OBSERVATION_KEYS_A0,
        max_episode_steps=max_episode_steps,
        options={"controller": "osc"},
    )


def _frames() -> tuple[StreamingInterventionFrame, ...]:
    return (
        StreamingInterventionFrame(
            frame_id="frame-0",
            mode=COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
            base_action=(0.25, -0.25),
        ),
        StreamingInterventionFrame(
            frame_id="frame-1",
            mode=COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
            base_action=(0.75, -0.75),
            intervention_action=(0.5, -0.5),
        ),
        StreamingInterventionFrame(
            frame_id="frame-2",
            mode=COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
            base_action=(-0.9, -0.9),
            intervention_action=(0.4, 1.5),
        ),
    )


def _runner() -> StreamingInterventionMockRunner:
    return StreamingInterventionMockRunner(
        RobosuiteMixedControlAdapter(
            action_low=(-1.0, -1.0),
            action_high=(1.0, 1.0),
        )
    )


def test_mock_runner_consumes_all_three_modes_and_stops_at_manifest_limit() -> None:
    result = _runner().run(manifest=_manifest(), frames=iter(_frames()))

    assert [step.selection.applied_action for step in result.steps] == [
        (0.25, -0.25),
        (1.0, -1.0),
        (0.4, 1.0),
    ]
    assert result.stop_reason == "manifest_step_limit"
    assert result.steps[-1].backend_step.truncated
    assert all(not step.selection.authoritative for step in result.steps)


def test_mock_replay_is_deterministic_for_exact_manifest_and_frames() -> None:
    first = _runner().run(manifest=_manifest(), frames=_frames())
    second = _runner().run(manifest=_manifest(), frames=_frames())

    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.run_id == second.run_id
    assert first.steps[-1].backend_step.observation == (
        second.steps[-1].backend_step.observation
    )


def test_seed_change_changes_reset_and_run_identity() -> None:
    first = _runner().run(manifest=_manifest(seed=11), frames=_frames())
    second = _runner().run(manifest=_manifest(seed=12), frames=_frames())

    assert first.reset_receipt.initial_observation_hash != (
        second.reset_receipt.initial_observation_hash
    )
    assert first.semantic_content_hash != second.semantic_content_hash


def test_mock_runner_rejects_duplicate_frame_identity() -> None:
    frame = _frames()[0]
    with pytest.raises(ValueError, match="unique"):
        _runner().run(
            manifest=_manifest(max_episode_steps=4),
            frames=(frame, frame),
        )


def test_mock_frame_rejects_nonfinite_input_before_replay_identity() -> None:
    with pytest.raises(ValueError, match="finite"):
        StreamingInterventionFrame(
            frame_id="frame-invalid",
            mode=COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
            base_action=(float("nan"), 0.0),
        )


def test_mock_reset_supports_action_dimensions_larger_than_one_digest() -> None:
    dimension = 40
    manifest = RobosuiteResetManifest(
        environment_id="LargeActionMock",
        environment_contract_hash="env-contract-large-a0",
        task_id="streaming-intervention-a0",
        episode_id="episode-large",
        reset_sequence=0,
        seed=3,
        action_dimension=dimension,
        observation_keys=MOCK_OBSERVATION_KEYS_A0,
        max_episode_steps=1,
    )
    runner = StreamingInterventionMockRunner(
        RobosuiteMixedControlAdapter(
            action_low=(-1.0,) * dimension,
            action_high=(1.0,) * dimension,
        )
    )

    result = runner.run(
        manifest=manifest,
        frames=(
            StreamingInterventionFrame(
                frame_id="frame-large",
                mode=COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
                base_action=(0.0,) * dimension,
            ),
        ),
    )

    assert len(result.steps[0].backend_step.observation["state"]) == dimension


def test_a0_path_does_not_import_real_robosuite_runtime() -> None:
    _runner().run(manifest=_manifest(max_episode_steps=1), frames=_frames())

    assert "robosuite" not in sys.modules
