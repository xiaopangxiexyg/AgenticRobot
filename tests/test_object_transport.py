from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.object_transport import GraspedObjectObservation, verify_rigid_follow


def pose(xyz=(0, 0, 0), angles=(0, 0, 0)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    result[:3, 3] = xyz
    return result


def observation(**overrides):
    fields = dict(object_id="cube", side="right", reference_frame_id="world", scene_revision=1,
        measured_at=10., reference_from_object=pose((.3, -.2, .4), (10, 20, 30)),
        reference_from_tool=pose((.27, -.25, .48), (170, 0, 90)),
        qpos_by_side={side: np.zeros(6) for side in ("left", "right")}, box_size_m=(.04, .05, .06),
        object_from_box=pose(), geometry_complete=True, pose_evidence_ref="pose:post-grasp",
        grasp_evidence_ref="contact-and-follow:trial-lift")
    fields.update(overrides)
    return GraspedObjectObservation(**fields)


def test_tool_goal_preserves_rotated_off_center_object_relation():
    held = observation()
    goal = pose((.4, .1, .6), (30, -20, 80))
    np.testing.assert_allclose(held.tool_goal(goal) @ held.tool_from_object, goal, atol=1e-12)
    assert np.linalg.norm(held.tool_goal(goal)[:3, 3]-goal[:3, 3]) > .05
    assert not held.reference_from_object.flags.writeable
    assert not held.qpos_by_side["right"].flags.writeable


@pytest.mark.parametrize("change", [dict(geometry_complete=False), dict(grasp_evidence_ref=""),
    dict(reference_from_object=np.zeros((4, 4))), dict(object_from_box=np.diag([1, 1, -1, 1])),
    dict(box_size_m=(0, .1, .1)), dict(measured_at=float("nan"))])
def test_rejects_unknown_geometry_unverified_grasp_or_invalid_frames(change):
    with pytest.raises(ValueError):
        observation(**change)


@pytest.mark.parametrize("change", [dict(now=10.3), dict(now=9.99), dict(scene_revision=2),
    dict(side="left"), dict(object_id="other"), dict(reference_frame_id="camera")])
def test_freshness_identity_and_frame_must_match(change):
    fields = dict(now=10.1, scene_revision=1, reference_frame_id="world", object_id="cube", side="right")
    fields.update(change)
    with pytest.raises(ValueError):
        observation().require_current(**fields)


def test_rigid_follow_requires_actual_motion_and_detects_slip():
    before = observation()
    delta = pose((.01, 0, .04), (0, 3, 0))
    after = replace(before, measured_at=10.5, reference_from_tool=delta @ before.reference_from_tool,
                    reference_from_object=delta @ before.reference_from_object)
    assert verify_rigid_follow(before, after)["relative_translation_change_m"] < 1e-10
    with pytest.raises(ValueError, match="Static"):
        verify_rigid_follow(before, replace(before, measured_at=10.5))
    with pytest.raises(ValueError, match="slipped"):
        verify_rigid_follow(before, replace(after, reference_from_object=pose((.02, 0, 0)) @ after.reference_from_object))


def test_fcl_carried_box_orientation_and_explicit_other_arm_pairs():
    pin = pytest.importorskip("pinocchio")
    fcl = pytest.importorskip("hppfcl")
    from robot_motion.rm65.obstacle_collision import AttachedBoxCollisionProxy, RM65ObstacleCollisionOracle
    from robot_motion.scene_obstacles import ObstacleSceneSnapshot
    model = pin.Model()
    joint = model.addJoint(0, pin.JointModelPX(), pin.SE3.Identity(), "x")
    frame = model.addFrame(pin.Frame("tool", joint, 0, pin.SE3.Identity(), pin.FrameType.OP_FRAME))
    geometries = pin.GeometryModel()
    geometries.addGeometryObject(pin.GeometryObject("other_arm_proxy", 0, 0,
        pin.SE3(np.eye(3), np.array([0., .2, 0.])), fcl.Sphere(.025)))
    def build(rotation, targets):
        return RM65ObstacleCollisionOracle(model=model, base_collision_model=geometries,
            reference_full_q=np.zeros(1), qpos_indices_by_component={"right": (0,)},
            scene=ObstacleSceneSnapshot("world", "frame", 1),
            attached_box_proxies=(AttachedBoxCollisionProxy("held", "tool", (0, 0, 0), (.4, .04, .04),
                tuple(rotation.reshape(-1)), targets),))
    unrotated = build(np.eye(3), ("other_arm_proxy",))
    assert unrotated.check({"right": np.zeros(1)}).safe
    rotated = build(Rotation.from_euler("z", 90, degrees=True).as_matrix(), ("other_arm_proxy",))
    assert not rotated.check({"right": np.zeros(1)}).safe
    assert rotated.check({"right": np.array([.2])}).safe
    assert len(geometries.geometryObjects) == 1 and len(geometries.collisionPairs) == 0
    with pytest.raises(ValueError, match="Invalid attached"):
        build(np.eye(3), ("missing",))
