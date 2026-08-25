from __future__ import annotations

import numpy as np

from experiments.sim_learning.state_bc.contracts import (
    StateBCArtifactMetadata,
)
from experiments.sim_learning.state_bc.model import StateBCModel, fit_linear_state_bc


def _metadata() -> StateBCArtifactMetadata:
    return StateBCArtifactMetadata(
        selector_id="fixture",
        train_episode_count=2,
        test_episode_count=1,
        train_frame_count=12,
        test_frame_count=4,
    )


def test_linear_state_bc_fit_round_trips_a_self_contained_artifact(tmp_path):
    rng = np.random.default_rng(7)
    states = rng.normal(size=(24, 16)).astype(np.float32)
    true_weights = rng.normal(scale=0.05, size=(16, 16)).astype(np.float32)
    true_bias = rng.normal(scale=0.01, size=(16,)).astype(np.float32)
    actions = (states @ true_weights + true_bias).astype(np.float32)

    model = fit_linear_state_bc(states, actions, artifact_metadata=_metadata(), ridge=1e-8)
    np.testing.assert_allclose(model.predict(states[0]), actions[0], atol=2e-3)
    artifact = model.save(tmp_path / "state_policy")
    loaded = StateBCModel.load(artifact)

    assert loaded.artifact_hash == model.artifact_hash
    np.testing.assert_allclose(
        [loaded.predict(row) for row in states],
        [model.predict(row) for row in states],
        atol=1e-6,
    )
