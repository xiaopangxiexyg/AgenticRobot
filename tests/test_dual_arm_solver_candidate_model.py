from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from robot_motion.rm65 import dual_arm_target_solver as facade
from robot_motion.rm65.dual_arm_solver import candidate_model


def test_facade_reexports_candidate_value_objects() -> None:
    assert facade._DualArmCandidate is candidate_model._DualArmCandidate
    assert (
        facade._DualArmEvaluatedCandidate is candidate_model._DualArmEvaluatedCandidate
    )


def test_candidate_defaults_and_evaluated_record_are_value_objects() -> None:
    candidate = candidate_model._DualArmCandidate(
        name="tracking",
        qpos=np.zeros(12, dtype=float),
        source="test",
        method="base",
    )
    evaluated = candidate_model._DualArmEvaluatedCandidate(
        candidate=candidate,
        output_qpos=np.ones(12, dtype=float),
        score=1.25,
        metadata={"safe": True},
    )

    assert candidate.raw_score_deg == 0.0
    assert candidate.metadata == {}
    assert evaluated.candidate is candidate
    assert evaluated.score == pytest.approx(1.25)
    assert evaluated.metadata == {"safe": True}

    with pytest.raises(FrozenInstanceError):
        candidate.name = "changed"  # type: ignore[misc]


def test_candidate_model_has_no_facade_import() -> None:
    source = candidate_model.__loader__.get_source(candidate_model.__name__)
    assert source is not None
    assert "dual_arm_target_solver" not in source
