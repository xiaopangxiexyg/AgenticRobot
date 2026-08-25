from __future__ import annotations

import argparse

from teleop_runtime.collection import hand_action_specs, run_collect


def _args(**overrides: str) -> argparse.Namespace:
    values = {"hand_backend": "inspire"}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_collect_preserves_hand_action_spec_alias() -> None:
    assert (
        run_collect._collection_hand_action_specs
        is hand_action_specs.resolve_collection_hand_action_specs
    )


def test_no_dmtac_backend_preserves_none_result() -> None:
    assert hand_action_specs.resolve_collection_hand_action_specs(
        _args(), ("left", "right")
    ) is None


def test_dmtac_specs_follow_active_sides_and_overrides() -> None:
    specs = hand_action_specs.resolve_collection_hand_action_specs(
        _args(right_hand_backend="dmtac-gripper"), ("left", "right")
    )

    assert specs is not None
    assert tuple(specs) == ("right",)
    assert specs["right"].side == "right"


def test_global_dmtac_mode_selects_only_active_sides() -> None:
    specs = hand_action_specs.resolve_collection_hand_action_specs(
        _args(hand_backend="dmtac-gripper"), ("left",)
    )

    assert specs is not None
    assert tuple(specs) == ("left",)
