from __future__ import annotations

import math

import pytest

from control_composition.allocation import (
    COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
)
from experiments.sim_learning.streaming_intervention import (
    DeterministicRobosuiteMockBackend,
)
from experiments.sim_learning.streaming_intervention.mock_runner import (
    MOCK_OBSERVATION_KEYS_A0,
)
from sim.robosuite import RobosuiteMixedControlAdapter, RobosuiteResetManifest


def _manifest(**overrides: object) -> RobosuiteResetManifest:
    values: dict[str, object] = {
        "environment_id": "Lift",
        "environment_contract_hash": "env-contract-a0",
        "task_id": "mixed-control-a0",
        "episode_id": "episode-0001",
        "reset_sequence": 0,
        "seed": 7,
        "action_dimension": 2,
        "observation_keys": MOCK_OBSERVATION_KEYS_A0,
        "max_episode_steps": 3,
        "options": {"controller": "osc", "initialization": {"noise": 0.0}},
    }
    values.update(overrides)
    return RobosuiteResetManifest(**values)  # type: ignore[arg-type]


def test_reset_manifest_is_content_addressed_and_mapping_order_independent() -> None:
    first = _manifest(options={"b": [1, 2], "a": {"enabled": True}})
    second = _manifest(options={"a": {"enabled": True}, "b": [1, 2]})

    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.manifest_id == second.manifest_id
    with pytest.raises(TypeError):
        first.options["new"] = "mutation"  # type: ignore[index]


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"seed": -1}, "seed"),
        ({"observation_keys": ("state", "state")}, "unique"),
        ({"options": {"opaque": object()}}, "primitive"),
    ],
)
def test_reset_manifest_fails_closed(
    overrides: dict[str, object],
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        _manifest(**overrides)


def test_adapter_composes_three_modes_then_enforces_final_envelope() -> None:
    adapter = RobosuiteMixedControlAdapter(
        action_low=(-1.0, -1.0),
        action_high=(1.0, 1.0),
    )

    base_only = adapter.compose(
        mode=COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
        base_action=(0.25, -0.5),
    )
    relative = adapter.compose(
        mode=COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
        base_action=(0.75, -0.75),
        intervention_action=(0.5, -0.5),
    )
    absolute = adapter.compose(
        mode=COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
        base_action=(-0.8, -0.8),
        intervention_action=(0.4, 1.5),
    )

    assert base_only.applied_action == (0.25, -0.5)
    assert relative.preclip_action == (1.25, -1.25)
    assert relative.applied_action == (1.0, -1.0)
    assert relative.clipped_dimensions == (0, 1)
    assert absolute.applied_action == (0.4, 1.0)
    assert absolute.clipped_dimensions == (1,)
    assert not base_only.authoritative
    assert not relative.authoritative
    assert not absolute.authoritative


def test_adapter_rejects_invalid_actions_and_modes() -> None:
    adapter = RobosuiteMixedControlAdapter(
        action_low=(-1.0, -1.0),
        action_high=(1.0, 1.0),
    )

    with pytest.raises(ValueError, match="Unsupported"):
        adapter.compose(mode="unknown", base_action=(0.0, 0.0))
    with pytest.raises(ValueError, match="required"):
        adapter.compose(
            mode=COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
            base_action=(0.0, 0.0),
        )
    with pytest.raises(ValueError, match="dimension"):
        adapter.compose(
            mode=COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
            base_action=(0.0,),
        )
    with pytest.raises(ValueError, match="finite"):
        adapter.compose(
            mode=COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
            base_action=(math.nan, 0.0),
        )


def test_reset_receipt_binds_backend_manifest_and_initial_observation() -> None:
    adapter = RobosuiteMixedControlAdapter(
        action_low=(-1.0, -1.0),
        action_high=(1.0, 1.0),
    )
    manifest = _manifest()

    observation, receipt = adapter.reset_backend(
        DeterministicRobosuiteMockBackend(),
        manifest,
    )

    assert tuple(observation) == manifest.observation_keys
    assert receipt.backend_id == "robosuite-a0-deterministic-mock"
    assert receipt.manifest_id == manifest.manifest_id
    assert receipt.manifest_hash == manifest.semantic_content_hash
    assert receipt.initial_observation_hash
