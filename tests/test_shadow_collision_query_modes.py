"""Keep grasp admission independent of inconsistent broad mesh diagnostics."""

import numpy as np
import pytest

from sim.mujoco.collision_safety import CollisionSafetyPolicy
from sim.sim_real_tests.mujoco.arbitrary_grasp_pose_motion import (
    _ArtifactAwareKinematicCollisionProjector,
)


@pytest.mark.parametrize(
    "separation_m,expected_safe", [(0.01, True), (0.001, False), (-0.002, False)]
)
def test_shadow_broad_query_artifact_cannot_override_admission(
    monkeypatch, separation_m, expected_safe
):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody>'
        '<body><geom type="box" size=".02 .02 .02" contype="2" conaffinity="2"/></body>'
        f'<body pos="{.04 + separation_m} 0 0"><joint type="slide" axis="1 0 0"/>'
        '<geom type="box" size=".02 .02 .02" contype="2" conaffinity="2"/>'
        '</body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    projector = _ArtifactAwareKinematicCollisionProjector(
        mujoco, model, np.array([0]), CollisionSafetyPolicy(minimum_clearance_m=0.002)
    )
    original = mujoco.mj_geomDistance
    radii = []

    def inconsistent_broad_query(model, data, first, second, limit, witness):
        radii.append(limit)
        measured = original(model, data, first, second, limit, witness)
        return -0.026 if limit > 0.002 else measured

    monkeypatch.setattr(mujoco, "mj_geomDistance", inconsistent_broad_query)
    result = projector.check_qpos(data, data.qpos)
    assert result.safe is expected_safe
    assert result.minimum_distance_m == pytest.approx(min(0.002, separation_m))
    if expected_safe:
        assert radii == [0.05]  # Narrow admission prunes this separated AABB.
    else:
        assert radii == [0.002]  # Unsafe admission never requests a diagnostic.
