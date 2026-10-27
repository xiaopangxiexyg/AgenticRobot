"""Native attached-geometry queries and measured guards; no physics stepping."""

import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.action import BIMANUAL_EE_POSE_V1, RobotAction
from robot_motion.path_planning import JointConfigurationCheck
from robot_motion.relative_action import pose_matrix_to_xyzquat
from sim.mujoco.collision_safety import KinematicCollisionProjector
from sim.mujoco.collision_safety.policy import CollisionSafetyPolicy
from sim.mujoco.deploy import planned_tcp_motion as motion
from sim.mujoco.deploy.grasp_contact import MujocoGraspContactBinding
from sim.mujoco.deploy.payload_motion import MujocoPayloadBinding, MujocoPayloadSupport


def _pose(data, model, name):
    body = model.body(name).id
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = data.xmat[body].reshape(3, 3), data.xpos[body]
    return result


def _scene(*, shape="box", extra="", table=False, table_shape="plane", second_geom=False,
           sensor_y=.0245, table_mocap=False, payload_side="right"):
    import mujoco

    def joints_for(side):
        return "".join(f'<joint name="{side}_{name}" type="{kind}" axis="{axis}"/>'
            for name, kind, axis in (("x", "slide", "1 0 0"), ("y", "slide", "0 1 0"),
                ("z", "slide", "0 0 1"), ("roll", "hinge", "1 0 0"),
                ("pitch", "hinge", "0 1 0"), ("yaw", "hinge", "0 0 1")))

    vertices = " ".join(f"{x} {y} {z}" for x in (-.05, .05) for y in (-.015, .015) for z in (-.015, .015))
    asset = f'<asset><mesh name="native_mesh" vertex="{vertices}"/></asset>' if shape == "mesh" else ""
    geom = 'type="mesh" mesh="native_mesh"' if shape == "mesh" else 'type="box" size=".05 .015 .015"'
    table_xml = '''<body name="table" pos="0 0 .385"><geom name="table_geom" type="plane"
      size="1 1 .01" contype="1" conaffinity="15"/></body>''' if table else ""
    if table and table_shape == "box":
        table_xml = '''<body name="table" pos="0 0 .355"><geom name="table_geom" type="box"
          size=".6 .6 .03" contype="1" conaffinity="15"/></body>'''
    if table_mocap:
        table_xml = table_xml.replace('<body name="table"', '<body name="table" mocap="true"')
    tail = '''<geom name="payload_tail" pos=".07 0 0" type="box" size=".01 .01 .01"
      contype="8" conaffinity="15"/>''' if second_geom else ""
    passive_side = "left" if payload_side == "right" else "right"
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>{asset}<option gravity="0 0 0"/>
      <worldbody>
        <body name="passive_tool" pos="-2 0 .4">{joints_for(passive_side)}
          <geom name="left_arm_geom" type="sphere" size=".01" contype="2" conaffinity="15"/>
        </body>
        <body name="tool" pos="0 0 .4">{joints_for(payload_side)}
          <geom name="right_arm_geom" pos="-.2 0 0" type="sphere" size=".01" contype="2" conaffinity="15"/>
          <body name="sensor_left_body" pos=".08 {-sensor_y:.9f} 0"><geom name="sensor_left"
            type="box" size=".01 .01 .01" contype="4" conaffinity="15"/></body>
          <body name="sensor_right_body" pos=".08 {sensor_y:.9f} 0"><geom name="sensor_right"
            type="box" size=".01 .01 .01" contype="4" conaffinity="15"/></body>
        </body>
        <body name="payload" pos=".08 0 .4"><freejoint/>
          <geom name="payload_geom" {geom} contype="8" conaffinity="15"/>{tail}
        </body>{table_xml}{extra}
      </worldbody>
    </mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    scene = SimpleNamespace(mujoco=mujoco, model=model, data=data)
    scene.arm_specs = {side: SimpleNamespace(joint_names=tuple(
        f"{side}_{axis}" for axis in ("x", "y", "z", "roll", "pitch", "yaw"))) for side in ("left", "right")}
    scene.joint_qpos_addr = lambda name: int(model.joint(name).qposadr[0])
    addresses = np.array([scene.joint_qpos_addr(name) for side in ("left", "right")
                          for name in scene.arm_specs[side].joint_names])
    original = KinematicCollisionProjector(mujoco, model, addresses, CollisionSafetyPolicy())
    scene.collision_projector = lambda: original
    scene.arm_qpos_by_side = lambda: {side: np.array([data.qpos[scene.joint_qpos_addr(name)]
        for name in scene.arm_specs[side].joint_names]) for side in ("left", "right")}
    return scene


def _binding(scene, **overrides):
    options = {"target_body_name": "payload", "sensor_geom_names": ("sensor_left", "sensor_right"),
        "tcp_pose_from_data": lambda d: _pose(d, scene.model, "tool"), "gripper_closed_from_data": lambda d: True}
    options.update(overrides)
    return MujocoPayloadBinding.capture(scene, scene.data, **options)


def _support(**changes):
    values = {"geom_names": ("table_geom",), "normal_world": (0., 0., 1.),
        "region_min_world": (-.5, -.5, .384), "region_max_world": (.5, .5, .386),
        "phase": "lift", "max_penetration_m": .001}
    values.update(changes)
    return MujocoPayloadSupport(**values)


def _oracle(scene, binding, **kwargs):
    return motion.MujocoJointSceneOracle(scene, np.asarray(binding.source_model_qpos), payload_binding=binding, **kwargs)


def _pair(oracle, *, x=0., y=0., z=0., yaw=0.):
    result = {side: oracle.source.qpos[indices].copy() for side, indices in oracle.addresses.items()}
    result["right"][:] = (x, y, z, 0., 0., yaw)
    return result


def _planner(scene, monkeypatch, *, side="right"):
    """Native query/resolver fixture; analytic IK is not an RM65 acceptance."""
    class Projector:
        def project(self, *, model_qpos, qpos_by_source, component_ids):
            data = scene.mujoco.MjData(scene.model)
            data.qpos[:] = model_qpos
            for name, q in qpos_by_source.items():
                for joint, value in zip(scene.arm_specs[name].joint_names, q):
                    data.qpos[scene.joint_qpos_addr(joint)] = value
            scene.mujoco.mj_kinematics(scene.model, data)
            return {side + "_arm": tuple(pose_matrix_to_xyzquat(_pose(data, scene.model, "tool")))}

    guard = {"oracle": None, "lock_depth": 0}
    lock = threading.RLock()

    def capture(callback):
        with lock:
            guard["lock_depth"] += 1
            try:
                return callback(scene)
            finally:
                guard["lock_depth"] -= 1

    def install(value):
        assert guard["lock_depth"] > 0 or value is None
        guard["oracle"] = value

    scene.body_pose = lambda name: np.eye(4)
    rig = SimpleNamespace(scene=scene, capture_scene=capture,
        measured_qpos_by_side=scene.arm_qpos_by_side, set_additional_contact_oracle=install,
        config=SimpleNamespace(max_velocity_rad_s=(1.,) * 6, max_acceleration_rad_s2=(2.,) * 6,
            max_command_step_deg=5., velocity_ramp_sec=.2, timing=SimpleNamespace(command_hz=60.)))
    planner = motion.MujocoPlannedTcpMotion(rig=rig,
        task_config=SimpleNamespace(left_tool_profile_path="unused", right_tool_profile_path="unused"),
        pose_projector=Projector(), reference_hz=60., motion_side=side)

    def fk(q):
        return motion.xyzquat_to_matrix(planner.pose_for_configuration(scene.data.qpos, {side: q}))

    def solve(**kwargs):
        return np.array([*kwargs["target_pose"][:3, 3] - (0, 0, .4), 0., 0., 0.])

    planner._solver = SimpleNamespace(q_min=np.full(6, -2.), q_max=np.full(6, 2.),
        compute_pose=fk, solve_iterative=solve)
    monkeypatch.setattr(motion, "load_rm65_tool_profile", lambda *args, **kwargs: None)
    monkeypatch.setattr(motion, "endpoint_to_tool_transform", lambda _: np.eye(4))
    return planner, guard


def _plan(planner, binding, *, x=0., z=.03, **kwargs):
    return planner.plan((x, 0., .4 + z, 0., 0., 0., 1.),
        translation_tolerance_m=.001, rotation_tolerance_rad=.01,
        cancel_requested=lambda: False, payload_binding=binding, **kwargs)


def _action(planner, pose):
    return RobotAction(action_space=BIMANUAL_EE_POSE_V1,
                       arm_pose_by_side={planner.motion_side: pose})


@pytest.mark.unit
@pytest.mark.parametrize("changes", [{"phase": "carry"}, {"normal_world": (0, 0, 2)},
    {"geom_names": ("table_geom", "table_geom")}, {"max_penetration_m": True},
    {"region_min_world": (1., 1., 1.)}, {"lateral_tolerance_m": .004}])
def test_support_requires_explicit_bounded_identity_direction_region(changes):
    with pytest.raises(ValueError):
        _support(**changes)


def test_fixed_support_keeps_its_original_serialized_identity():
    assert set(_support().to_payload()) == {
        "geom_names", "normal_world", "region_min_world", "region_max_world",
        "phase", "max_penetration_m", "lateral_tolerance_m",
    }
    with pytest.raises(ValueError, match="both body identity and frozen pose"):
        _support(dynamic_body_name="support")


@pytest.mark.integration
def test_payload_query_requires_exact_binding_cut_and_rejects_contact_policy_mix():
    scene = _scene()
    binding = _binding(scene)
    changed = np.asarray(binding.source_model_qpos).copy()
    changed[scene.joint_qpos_addr("right_x")] += .001
    with pytest.raises(ValueError, match="same-cut source qpos"):
        motion.MujocoJointSceneOracle(scene, changed, payload_binding=binding)
    with pytest.raises(TypeError, match="MujocoPayloadBinding"):
        motion.MujocoJointSceneOracle(scene, scene.data.qpos, payload_binding=True)
    contact = MujocoGraspContactBinding("payload", ("sensor_left", "sensor_right"))
    with pytest.raises(ValueError, match="distinct operation bindings"):
        _oracle(scene, binding, contact_binding=contact)
    oracle = _oracle(scene, binding)
    assert oracle.check(_pair(oracle)).safe


@pytest.mark.integration
@pytest.mark.parametrize("changes", [
    {"translation_tolerance_m": .003001},
    {"rotation_tolerance_rad": math.radians(2.001)},
    {"translation_tolerance_m": True},
    {"rotation_tolerance_rad": float("nan")},
])
def test_payload_capture_cannot_expand_frozen_relation_bounds(changes):
    scene = _scene()
    with pytest.raises(ValueError, match="3mm/2deg"):
        _binding(scene, **changes)


@pytest.mark.unit
def test_shared_noncontact_policy_retains_default_fields_and_pair_contract():
    from sim.mujoco.collision_safety.policy import NonContactObjectMovePolicy

    assert motion._NonContactMovePolicy is NonContactObjectMovePolicy
    ordinary, shared = CollisionSafetyPolicy(), NonContactObjectMovePolicy()
    assert vars(shared) == vars(ordinary)
    assert shared.distance_query_limit_m == ordinary.minimum_clearance_m
    for first in (1, 2, 4, 8):
        for second in (1, 2, 4, 8):
            expected = ordinary.pair_is_forbidden(first, second) or {first, second} == {4, 8}
            assert shared.pair_is_forbidden(first, second) is expected


@pytest.mark.integration
def test_changed_timestep_invalidates_boundary_and_existing_actual_oracle():
    scene = _scene(sensor_y=.0249)
    binding = _binding(scene)
    guard = binding.contact_oracle(scene)
    original_step = float(scene.model.opt.timestep)
    assert guard.check_contacts(scene.data).safe
    scene.model.opt.timestep = original_step / 2
    for check in (lambda: binding.validate_scene(scene),
                  lambda: binding.check_measured(scene),
                  lambda: guard.check_contacts(scene.data),
                  lambda: _oracle(scene, binding)):
        with pytest.raises(ValueError, match="timestep"):
            check()
    assert guard._last_bilateral_contact_time == 0.


@pytest.mark.integration
@pytest.mark.parametrize("entry", ["pull", "raw_oracle"])
def test_noncurrent_native_copies_cannot_publish_actual_contact_history(entry):
    scene = _scene(sensor_y=.0249)
    binding = _binding(scene)
    guard = binding.contact_oracle(scene)
    assert guard.check_contacts(scene.data).safe
    previous = guard._last_bilateral_contact_time
    future = scene.mujoco.MjData(scene.model)
    scene.mujoco.mj_copyData(future, scene.model, scene.data)
    future.time = 100 * float(scene.model.opt.timestep)
    with pytest.raises(ValueError, match="current live native cut"):
        if entry == "pull":
            binding.check_measured(scene, source_data=future)
        else:
            guard.check_contacts(future)
    assert guard._last_bilateral_contact_time == previous
    with pytest.raises(ValueError, match="exact binding/model"):
        binding.check_measured(scene, _oracle=_oracle(scene, binding).projector.oracle)


@pytest.mark.integration
def test_direct_projector_rejects_another_cut_and_unsafe_actual_sweep_source():
    scene = _scene()
    binding = _binding(scene)
    oracle = _oracle(scene, binding)
    q = np.concatenate(tuple(_pair(oracle).values()))
    another = scene.mujoco.MjData(scene.model)
    scene.mujoco.mj_copyData(another, scene.model, scene.data)
    another.qpos[scene.joint_qpos_addr("right_x")] += .001
    with pytest.raises(ValueError, match="frozen same-cut geometry source"):
        oracle.projector.check_qpos(another, q)
    with pytest.raises(ValueError, match="current live native cut"):
        oracle.projector.project(another, start_qpos=q, requested_qpos=q)
    address = int(scene.model.jnt_qposadr[scene.model.body_jntadr[scene.model.body("payload").id]])
    scene.data.qpos[address] += .004
    with pytest.raises(ValueError, match="actual source is unsafe"):
        oracle.projector.project(scene.data, start_qpos=q, requested_qpos=q)


@pytest.mark.integration
def test_actual_sweep_source_validation_does_not_publish_history_or_observer_results():
    scene = _scene()
    cuts = []

    def observer(data):
        cuts.append(float(data.time))
        return JointConfigurationCheck(safe=True)

    binding = _binding(scene, measured_guard=observer, guard_identity="test.no-query-publication")
    oracle = _oracle(scene, binding)
    actual = binding.contact_oracle(scene)
    previous_time = actual._last_bilateral_contact_time
    previous_details = actual.last_measured_details
    count = len(cuts)
    start = np.concatenate(tuple(_pair(oracle).values()))
    goal = np.concatenate(tuple(_pair(oracle, z=.01).values()))
    assert oracle.projector.project(scene.data, start_qpos=start, requested_qpos=goal).check.safe
    assert len(cuts) == count
    assert actual._last_bilateral_contact_time == previous_time
    assert actual.last_measured_details is previous_details
    with pytest.raises(ValueError, match="current measured arm cut"):
        oracle.projector.project(scene.data, start_qpos=goal, requested_qpos=start)


@pytest.mark.integration
def test_frozen_auxiliary_geometry_is_used_and_foreign_mocap_cuts_are_rejected():
    scene = _scene(extra='''<body name="moving_obstacle" mocap="true" pos="1 0 .4">
      <geom name="moving_geom" type="box" size=".004 .004 .004" contype="1" conaffinity="15"/></body>''')
    scene.data.mocap_pos[0, 0] = .8
    binding = _binding(scene)
    oracle = _oracle(scene, binding)
    check = oracle.check(_pair(oracle, x=.7))
    assert not check.safe
    assert "moving_geom" in (check.metadata["geom1"], check.metadata["geom2"])
    altered = scene.mujoco.MjData(scene.model)
    scene.mujoco.mj_copyData(altered, scene.model, scene.data)
    altered.mocap_pos[0, 0] = 1.2
    q = np.concatenate(tuple(_pair(oracle).values()))
    with pytest.raises(ValueError, match="frozen same-cut geometry source"):
        oracle.projector.check_qpos(altered, q)
    with pytest.raises(ValueError, match="current live native cut"):
        binding.check_measured(scene, source_data=altered)


@pytest.mark.integration
def test_mocap_driven_body_cannot_be_declared_a_fixed_support():
    scene = _scene(table=True, table_shape="box", table_mocap=True)
    with pytest.raises(ValueError, match="fixed, not mocap-driven"):
        _binding(scene, support=_support())


@pytest.mark.integration
def test_other_free_box_support_is_frozen_and_movement_revokes_placement():
    scene = _scene(extra='''<body name="support" pos=".08 0 .355"><freejoint/>
      <geom name="support_geom" type="box" size=".1 .08 .03" mass=".4"
        contype="8" conaffinity="15"/></body>''')
    support = _support(
        geom_names=("support_geom",), phase="place",
        region_min_world=(-.02, -.08, .385),
        region_max_world=(.18, .08, .385),
        dynamic_body_name="support",
        frozen_pose_xyz_quat_xyzw=tuple(pose_matrix_to_xyzquat(
            _pose(scene.data, scene.model, "support"))),
    )
    binding = _binding(scene, support=support)
    oracle = _oracle(scene, binding)
    assert oracle.check(_pair(oracle, z=.02)).safe
    assert binding.check_measured(scene).safe
    address = int(scene.model.jnt_qposadr[scene.model.body_jntadr[scene.model.body("support").id]])
    scene.data.qpos[address] += .004
    scene.mujoco.mj_kinematics(scene.model, scene.data)
    scene.mujoco.mj_collision(scene.model, scene.data)
    # The historical planning cut remains immutable; the live measured guard
    # must reject the support's actual movement before another dispatch.
    assert oracle.check(_pair(oracle, z=.02)).safe
    rejected = binding.check_measured(scene)
    assert not rejected.safe and rejected.metadata["pair_class"] == "payload_support_moved"


@pytest.mark.integration
def test_free_support_place_plan_stops_if_support_moves_before_dispatch(monkeypatch):
    scene = _scene(extra='''<body name="support" pos=".08 0 .355"><freejoint/>
      <geom name="support_geom" type="box" size=".1 .08 .03" mass=".4"
        contype="8" conaffinity="15"/></body>''')
    model = scene.model
    scene.data.qpos[scene.joint_qpos_addr("right_z")] += .06
    payload_address = int(model.jnt_qposadr[model.body_jntadr[model.body("payload").id]])
    scene.data.qpos[payload_address + 2] += .06
    scene.mujoco.mj_kinematics(model, scene.data)
    scene.mujoco.mj_collision(model, scene.data)
    support = _support(
        geom_names=("support_geom",), phase="place",
        region_min_world=(-.02, -.08, .385),
        region_max_world=(.18, .08, .385),
        dynamic_body_name="support",
        frozen_pose_xyz_quat_xyzw=tuple(pose_matrix_to_xyzquat(
            _pose(scene.data, model, "support"))),
    )
    binding = _binding(scene, support=support)
    planner, installed = _planner(scene, monkeypatch)
    plan = _plan(planner, binding, z=0.)
    assert plan.reference_poses and installed["oracle"] is None
    planner.activate(plan)
    assert installed["oracle"] is binding.contact_oracle(scene)
    planner._reference_index = len(plan.reference_poses) - 1
    assert planner.resolve(_action(planner, plan.reference_poses[-1])) is not None
    support_address = int(model.jnt_qposadr[model.body_jntadr[model.body("support").id]])
    scene.data.qpos[support_address] += .004
    scene.mujoco.mj_kinematics(model, scene.data)
    scene.mujoco.mj_collision(model, scene.data)
    with pytest.raises(motion.MujocoTcpPlanRejected, match="payload_measured_blocked"):
        planner.resolve(_action(planner, plan.reference_poses[-1]))
    planner.clear()
    assert installed["oracle"] is None


@pytest.mark.integration
def test_free_support_cannot_be_used_as_a_fixed_table_exception():
    scene = _scene(extra='''<body name="support" pos=".08 0 .355"><freejoint/>
      <geom name="support_geom" type="box" size=".1 .08 .03" mass=".4"
        contype="8" conaffinity="15"/></body>''')
    with pytest.raises(ValueError, match="fixed native environment"):
        _binding(scene, support=_support(geom_names=("support_geom",)))


@pytest.mark.integration
@pytest.mark.parametrize("shape", ["box", "mesh"])
def test_native_binding_freezes_complete_relation_and_only_scratch_follows(shape, monkeypatch):
    scene = _scene(shape=shape, second_geom=True)
    before = {n: getattr(scene.data, n).copy() for n in ("qpos", "qvel", "ctrl", "xpos", "xmat", "geom_xpos", "eq_active")}
    geometry = scene.model.geom_pos.copy(), scene.model.geom_quat.copy(), scene.model.mesh_vert.copy()
    binding = _binding(scene)
    frozen = np.asarray(binding.tool_from_object).copy()
    oracle = _oracle(scene, binding)

    def no_step(*args):
        raise AssertionError("No physics stepping or dynamics forwarding is permitted")

    monkeypatch.setattr(scene.mujoco, "mj_step", no_step)
    monkeypatch.setattr(scene.mujoco, "mj_forward", no_step)
    pair = _pair(oracle, x=.2, y=-.1, z=.1, yaw=.7)
    assert oracle.check(pair).safe
    scratch = oracle.projector.scratch
    expected = _pose(scratch, scene.model, "tool") @ frozen
    np.testing.assert_allclose(_pose(scratch, scene.model, "payload"), expected, atol=1e-12)
    for name, saved in before.items():
        np.testing.assert_array_equal(getattr(scene.data, name), saved)
    for value, saved in zip((scene.model.geom_pos, scene.model.geom_quat, scene.model.mesh_vert), geometry):
        np.testing.assert_array_equal(value, saved)
    assert oracle.check(_pair(oracle)).safe
    assert binding.check_measured(scene).safe
    assert scene.data.time == 0.


@pytest.mark.integration
@pytest.mark.parametrize("shape", ["box", "mesh"])
def test_native_rotating_payload_corner_blocks_with_stationary_tcp(shape):
    scene = _scene(shape=shape, extra='''<body name="corner" pos=".09 .09 .4">
      <geom name="corner_geom" type="box" size=".004 .004 .004" contype="1" conaffinity="15"/></body>''')
    oracle = _oracle(scene, _binding(scene))
    assert oracle.check(_pair(oracle)).safe
    check = oracle.check(_pair(oracle, yaw=math.pi / 4))
    assert not check.safe
    assert {check.metadata["geom1"], check.metadata["geom2"]} == {"payload_geom", "corner_geom"}
    np.testing.assert_allclose(_pose(oracle.projector.scratch, scene.model, "tool")[:3, 3], (0, 0, .4))


@pytest.mark.integration
@pytest.mark.parametrize("group", [1, 2, 4, 8])
def test_native_payload_hits_foreign_geometry_while_robot_and_sensors_clear(group):
    scene = _scene(extra=f'''<body name="foreign" pos=".425 0 .4"><freejoint/>
      <geom name="foreign_geom" type="box" size=".004 .004 .004" contype="{group}" conaffinity="15"/></body>''')
    binding = _binding(scene)
    oracle = _oracle(scene, binding)
    check = oracle.check(_pair(oracle, x=.3))
    assert not check.safe
    assert {check.metadata["geom1"], check.metadata["geom2"]} == {"payload_geom", "foreign_geom"}
    assert scene.collision_projector().oracle.check(oracle.projector.scratch).safe is (group != 2)
    canonical_pairs = {(p.geom1, p.geom2) for p in scene.collision_projector().oracle._pairs}
    assert canonical_pairs <= {(p.geom1, p.geom2) for p in oracle.projector.oracle._pairs}


@pytest.mark.integration
def test_complete_native_target_includes_secondary_offset_collision_geom():
    scene = _scene(second_geom=True, extra='''<body pos=".465 0 .4"><geom name="tail_obstacle"
      type="box" size=".009 .009 .009" contype="1" conaffinity="15"/></body>''')
    oracle = _oracle(scene, _binding(scene))
    check = oracle.check(_pair(oracle, x=.3))
    assert not check.safe
    assert {check.metadata["geom1"], check.metadata["geom2"]} == {"payload_tail", "tail_obstacle"}


@pytest.mark.integration
def test_only_measured_closed_bilateral_contact_can_construct_binding():
    scene = _scene()
    with pytest.raises(ValueError, match="capture rejected"):
        _binding(scene, gripper_closed_from_data=lambda d: False)
    with pytest.raises(ValueError, match="exactly two"):
        _binding(scene, sensor_geom_names=("sensor_left",))
    with pytest.raises(ValueError, match="guard_identity"):
        _binding(scene, measured_guard=lambda d: JointConfigurationCheck(safe=True))
    target = int(scene.model.jnt_qposadr[scene.model.body_jntadr[scene.model.body("payload").id]])
    scene.data.qpos[target + 1] += .002
    with pytest.raises(ValueError, match="capture rejected"):
        _binding(scene)


@pytest.mark.integration
def test_frozen_original_anchor_is_validated_instead_of_reanchored_on_later_capture():
    scene = _scene()
    binding = _binding(scene)
    q = int(scene.model.jnt_qposadr[scene.model.body_jntadr[scene.model.body("payload").id]])
    scene.data.qpos[q] += .002
    newer = _binding(scene, tool_from_object=binding.tool_from_object)
    np.testing.assert_array_equal(newer.tool_from_object, binding.tool_from_object)
    scene.data.qpos[q] += .002
    assert not binding.check_measured(scene).safe
    with pytest.raises(ValueError, match="capture rejected"):
        _binding(scene, tool_from_object=binding.tool_from_object)


@pytest.mark.integration
@pytest.mark.parametrize("shape", ["box", "mesh"])
def test_lift_starts_on_only_named_table_pair_and_cannot_descend_or_slide(shape):
    scene = _scene(shape=shape, table=True)
    with pytest.raises(ValueError, match="capture rejected"):
        _binding(scene)
    binding = _binding(scene, support=_support())
    oracle = _oracle(scene, binding, clearance_padding_m=.002)
    for z in (0., .0005, .002, .01, .04):
        assert oracle.check(_pair(oracle, z=z)).safe
    assert not oracle.check(_pair(oracle, z=-.002)).safe
    assert not oracle.check(_pair(oracle, x=.01)).safe
    assert binding.check_measured(scene).safe
    with pytest.raises(ValueError, match="hard penetration"):
        _binding(scene, support=_support(max_penetration_m=.002))


@pytest.mark.integration
def test_place_allows_only_outward_face_inside_region_at_hard_bound():
    scene = _scene(table=True)
    binding = _binding(scene, support=_support(phase="place"))
    oracle = _oracle(scene, binding)
    assert oracle.check(_pair(oracle, z=.1)).safe
    assert oracle.check(_pair(oracle, z=-.0005)).safe
    assert not oracle.check(_pair(oracle, z=-.0015)).safe
    assert not oracle.check(_pair(oracle, x=.004)).safe
    assert not oracle.check(_pair(oracle, x=.6)).safe
    with pytest.raises(ValueError, match="outward native support face"):
        _binding(scene, support=_support(normal_world=(0., 0., -1.)))


@pytest.mark.integration
def test_native_box_table_retains_hard_bound_for_initial_094mm_penetration_without_padding():
    scene = _scene(table=True, table_shape="box")
    address = int(scene.model.jnt_qposadr[scene.model.body_jntadr[scene.model.body("payload").id]])
    scene.data.qpos[address + 2] -= .00094
    binding = _binding(scene, support=_support())
    oracle = _oracle(scene, binding, clearance_padding_m=.002)
    assert oracle.check(_pair(oracle)).safe
    assert oracle.check(_pair(oracle, z=.01)).safe
    assert not oracle.check(_pair(oracle, z=-.0001)).safe
    assert binding.check_measured(scene).safe


@pytest.mark.integration
@pytest.mark.parametrize("tracking_drop_m", [7.5792195e-9, 5e-8, 2e-6, 5e-5])
def test_realized_lift_start_tracking_noise_keeps_hard_safe_support_but_down_intent_fails(tracking_drop_m):
    scene = _scene(table=True, table_shape="box")
    address = int(scene.model.jnt_qposadr[scene.model.body_jntadr[scene.model.body("payload").id]])
    scene.data.qpos[address + 2] -= .00094
    binding = _binding(scene, support=_support())
    oracle = _oracle(scene, binding)
    # Identical candidate displacement is downward intent and remains blocked.
    assert not oracle.check(_pair(oracle, z=-tracking_drop_m)).safe
    actual_arm_z = scene.joint_qpos_addr("right_z")
    scene.data.qpos[actual_arm_z] -= tracking_drop_m
    scene.data.qpos[address + 2] -= tracking_drop_m
    # Keep live xpos/contact stale, as at a post-integration check site.
    assert binding.check_measured(scene).safe
    assert binding.contact_oracle(scene).check_contacts(scene.data).safe
    # Dispatch still checks its actual starting cut before sweeping *up* to
    # the unchanged, strict planned reference. This is not a new command path.
    oracle.source.qpos[:] = scene.data.qpos
    actual = np.concatenate([scene.data.qpos[i] for i in oracle.addresses.values()])
    commanded = np.concatenate([np.asarray(binding.source_model_qpos)[i] for i in oracle.addresses.values()])
    sweep = oracle.projector.project(oracle.source, start_qpos=actual, requested_qpos=commanded)
    assert not sweep.modified and sweep.check.safe
    down = commanded.copy()
    down[-4] = -.0002  # right_z; do not convert an actual tolerance into intent permission.
    assert oracle.projector.project(oracle.source, start_qpos=actual, requested_qpos=down).modified


@pytest.mark.integration
def test_realized_tracking_tolerance_never_permits_support_hard_penetration_or_foreign_contact():
    scene = _scene(table=True, table_shape="box", extra='''<body name="foreign" pos="1 0 .4"><freejoint/>
      <geom name="foreign_geom" type="box" size=".004 .004 .004" contype="8" conaffinity="15"/></body>''')
    address = int(scene.model.jnt_qposadr[scene.model.body_jntadr[scene.model.body("payload").id]])
    scene.data.qpos[address + 2] -= .00094
    binding = _binding(scene, support=_support())
    scene.data.qpos[scene.joint_qpos_addr("right_z")] -= .0001
    scene.data.qpos[address + 2] -= .0001
    assert not binding.check_measured(scene).safe
    scene.data.qpos[:] = binding.source_model_qpos
    foreign = int(scene.model.jnt_qposadr[scene.model.body_jntadr[scene.model.body("foreign").id]])
    scene.data.qpos[foreign] = .125
    assert not binding.check_measured(scene).safe


@pytest.mark.integration
def test_support_permission_never_exempts_other_objects_or_robot_table_contacts():
    scene = _scene(table=True, extra='''<body name="distractor" pos=".35 0 .4"><freejoint/>
      <geom name="distractor_geom" type="box" size=".01 .01 .015" contype="8" conaffinity="15"/></body>''')
    oracle = _oracle(scene, _binding(scene, support=_support(phase="place")))
    check = oracle.check(_pair(oracle, x=.22))
    assert not check.safe
    assert "distractor_geom" in (check.metadata["geom1"], check.metadata["geom2"])
    check = oracle.check(_pair(oracle, z=-.01))
    assert not check.safe
    assert scene.collision_projector().oracle.check(oracle.projector.scratch).safe is False


@pytest.mark.integration
def test_native_geometry_changes_including_mesh_vertices_invalidate_binding():
    scene = _scene(shape="mesh")
    binding = _binding(scene)
    scene.model.mesh_vert[0, 0] += .01
    with pytest.raises(ValueError, match="geometry/model/policy changed"):
        _oracle(scene, binding)
    with pytest.raises(ValueError, match="geometry/model/policy changed"):
        binding.validate_scene(scene)


@pytest.mark.integration
def test_optional_actual_guard_runs_on_native_contacts_but_never_candidate_queries():
    scene = _scene()
    cuts = []
    state = {"safe": True}

    def guard(data):
        cuts.append(data.qpos.copy())
        return JointConfigurationCheck(safe=state["safe"], reason=None if state["safe"] else "test_observer_gap")

    binding = _binding(scene, measured_guard=guard, guard_identity="trusted.native-follow.v1")
    oracle = _oracle(scene, binding)
    count = len(cuts)
    assert oracle.check(_pair(oracle, z=.1)).safe
    assert len(cuts) == count
    actual_guard = binding.contact_oracle(scene)
    assert actual_guard.check_contacts(scene.data).safe
    state["safe"] = False
    assert not actual_guard.check_contacts(scene.data).safe
    assert actual_guard.last_measured_details["measured_guard"]["reason"] == "test_observer_gap"
    assert not binding.check_measured(scene).safe


@pytest.mark.integration
def test_actual_tick_guard_recomputes_contacts_from_current_qpos_without_live_writes():
    scene = _scene()
    binding = _binding(scene)
    guard = binding.contact_oracle(scene)
    stale = scene.data.xpos.copy()
    address = int(scene.model.jnt_qposadr[scene.model.body_jntadr[scene.model.body("payload").id]])
    scene.data.qpos[address] += .004
    assert not guard.check_contacts(scene.data).safe
    np.testing.assert_array_equal(scene.data.xpos, stale)
    assert guard.last_measured_details["relative_translation_error_m"] == pytest.approx(.004)


@pytest.mark.integration
def test_actual_tick_guard_allows_only_one_physics_step_sensor_dropout_then_requires_recovery():
    scene = _scene(sensor_y=.0249)
    binding = _binding(scene)
    guard = binding.contact_oracle(scene)
    timestep = float(scene.model.opt.timestep)
    payload_joint = int(scene.model.body_jntadr[scene.model.body("payload").id])
    payload_qpos = int(scene.model.jnt_qposadr[payload_joint])

    assert guard.check_contacts(scene.data).safe
    scene.data.qpos[payload_qpos + 1] += .0002  # Lose only the left sensor by 0.1 mm.
    same_cut = guard.check_contacts(scene.data)
    assert not same_cut.safe  # State changes without one elapsed physics tick do not get grace.
    assert guard.last_measured_details["sensor_contact_dropout_grace"]["accepted"] is False
    scene.data.qpos[payload_qpos + 1] -= .0002
    assert guard.check_contacts(scene.data).safe
    scene.data.qpos[payload_qpos + 1] += .0002
    scene.data.time = timestep
    one_step = guard.check_contacts(scene.data)
    assert one_step.safe
    assert guard.last_measured_details["sensor_contacts"] == (False, True)
    grace = guard.last_measured_details["sensor_contact_dropout_grace"]
    assert grace["accepted"] is True
    assert grace["age_sec"] == pytest.approx(timestep)
    # Pre/post checks at the same native cut do not consume another time tick.
    assert guard.check_contacts(scene.data).safe

    scene.data.time = 2 * timestep
    persistent = guard.check_contacts(scene.data)
    assert not persistent.safe
    assert persistent.pair_class == "payload_relation"
    assert guard.last_measured_details["sensor_contact_dropout_grace"]["accepted"] is False

    # A recovered bilateral cut resets the anchor time without changing the
    # frozen object pose or any physical geometry.
    scene.data.qpos[payload_qpos + 1] -= .0002
    assert guard.check_contacts(scene.data).safe
    assert guard.last_measured_details["sensor_contacts"] == (True, True)


@pytest.mark.integration
@pytest.mark.parametrize("physics_hz", [1000., 2000.])
def test_pull_measurement_uses_the_same_one_step_contact_history_as_physics(physics_hz):
    scene = _scene(sensor_y=.0249)
    scene.model.opt.timestep = 1. / physics_hz
    binding = _binding(scene)
    guard = binding.contact_oracle(scene)
    timestep = float(scene.model.opt.timestep)
    payload_joint = int(scene.model.body_jntadr[scene.model.body("payload").id])
    address = int(scene.model.jnt_qposadr[payload_joint])

    # The physics owner observes bilateral contact between sparse pull calls.
    # No extra physics is stepped by either observer in this geometry fixture.
    for tick in range(1, 41):
        scene.data.time = tick * timestep
        assert guard.check_contacts(scene.data).safe
    scene.data.qpos[address + 1] += .0002
    scene.data.time = 41 * timestep
    assert guard.check_contacts(scene.data).safe
    pull = binding.check_measured(scene)
    assert pull.safe, pull.to_dict()
    assert pull.metadata["sensor_contact_dropout_grace"]["age_sec"] == pytest.approx(timestep)
    assert binding.check_measured(scene).safe  # Duplicate read cannot consume or extend grace.

    scene.data.time = 42 * timestep
    assert not guard.check_contacts(scene.data).safe
    assert not binding.check_measured(scene).safe


@pytest.mark.integration
def test_query_contact_history_cannot_advance_the_actual_binding_history():
    scene = _scene(sensor_y=.0249)
    binding = _binding(scene)
    actual = binding.contact_oracle(scene)
    assert actual.check_contacts(scene.data).safe
    query = _oracle(scene, binding)
    assert query.projector.oracle is not actual
    future = scene.mujoco.MjData(scene.model)
    scene.mujoco.mj_copyData(future, scene.model, scene.data)
    timestep = float(scene.model.opt.timestep)
    future.time = 100 * timestep
    assert query.projector.oracle.check_contacts(future).safe

    payload_joint = int(scene.model.body_jntadr[scene.model.body("payload").id])
    address = int(scene.model.jnt_qposadr[payload_joint])
    scene.data.qpos[address + 1] += .0002
    scene.data.time = timestep
    pull = binding.check_measured(scene)
    assert pull.safe
    assert pull.metadata["sensor_contact_dropout_grace"]["age_sec"] == pytest.approx(timestep)
    scene.data.time = 2 * timestep
    assert not binding.check_measured(scene).safe


@pytest.mark.integration
def test_sensor_dropout_grace_never_exempts_frozen_hand_object_relation_bound():
    scene = _scene(sensor_y=.0249)
    binding = _binding(scene)
    guard = binding.contact_oracle(scene)
    timestep = float(scene.model.opt.timestep)
    payload_joint = int(scene.model.body_jntadr[scene.model.body("payload").id])
    payload_qpos = int(scene.model.jnt_qposadr[payload_joint])
    assert guard.check_contacts(scene.data).safe
    scene.data.qpos[payload_qpos] += .004  # Along the jaw, preserving sensor contact geometry.
    scene.data.qpos[payload_qpos + 1] += .0002
    scene.data.time = timestep
    check = guard.check_contacts(scene.data)
    assert not check.safe
    assert check.pair_class == "payload_relation"
    assert guard.last_measured_details["sensor_contact_dropout_grace"]["accepted"] is False


@pytest.mark.integration
def test_measured_guard_preserves_invalid_return_and_original_callback_failure():
    scene = _scene()
    state = {"mode": "valid"}

    def callback(data):
        assert data is not scene.data
        if state["mode"] == "raise":
            raise RuntimeError("observer failure stays visible")
        if state["mode"] == "invalid":
            return True
        return JointConfigurationCheck(safe=True)

    binding = _binding(scene, measured_guard=callback, guard_identity="trusted.follow.v1")
    oracle = binding.contact_oracle(scene)
    state["mode"] = "invalid"
    with pytest.raises(TypeError, match="JointConfigurationCheck"):
        oracle.check_contacts(scene.data)
    state["mode"] = "raise"
    with pytest.raises(RuntimeError, match="observer failure stays visible"):
        oracle.check_contacts(scene.data)


@pytest.mark.integration
@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), 1e308])
def test_invalid_or_overflowing_candidate_never_becomes_safe(coordinate):
    scene = _scene()
    oracle = _oracle(scene, _binding(scene))
    with pytest.raises((ValueError, FloatingPointError)):
        oracle.check(_pair(oracle, x=coordinate))


@pytest.mark.integration
def test_full_signature_is_boundary_only_and_record_narrow_native_cost(monkeypatch, record_property):
    from sim.mujoco.deploy import payload_motion

    scene = _scene(shape="mesh")
    binding = _binding(scene)
    oracle = _oracle(scene, binding)
    guard = binding.contact_oracle(scene)
    started = time.perf_counter()
    for _ in range(100):
        payload_motion._model_signature(scene.model)
    signature_s = (time.perf_counter() - started) / 100

    def forbidden(*args):
        raise AssertionError("Full CAD hashes must not run on candidate/physics ticks")

    monkeypatch.setattr(payload_motion, "_model_signature", forbidden)
    started = time.perf_counter()
    for _ in range(100):
        assert oracle.check(_pair(oracle)).safe
    candidate_s = (time.perf_counter() - started) / 100
    started = time.perf_counter()
    for _ in range(100):
        assert guard.check_contacts(scene.data).safe
    actual_s = (time.perf_counter() - started) / 100
    started = time.perf_counter()
    for _ in range(100):
        assert binding.check_measured(scene).safe
    measured_api_s = (time.perf_counter() - started) / 100
    record_property("narrow_native_signature_us", signature_s * 1e6)
    record_property("narrow_native_candidate_us", candidate_s * 1e6)
    record_property("narrow_native_actual_guard_us", actual_s * 1e6)
    record_property("narrow_native_measured_api_us", measured_api_s * 1e6)


@pytest.mark.integration
@pytest.mark.parametrize("side", ["left", "right"])
def test_payload_owned_plan_resolves_exact_references_and_clear_releases_only_guard(monkeypatch, side):
    scene = _scene(table=True, payload_side=side)
    binding = _binding(scene, support=_support())
    planner, installed = _planner(scene, monkeypatch, side=side)
    before = scene.data.qpos.copy()
    plan = _plan(planner, binding)
    assert plan.payload_binding is binding
    assert len(plan.reference_poses) == len(plan.trajectory.q_rad) > 2
    assert installed["oracle"] is None and not planner.active
    planner.activate(plan)
    assert installed["oracle"] is binding.contact_oracle(scene)
    reference = planner.reference(0)
    from sim.mujoco.deploy import payload_motion

    def forbidden_hash(*args):
        raise AssertionError("A live reference must not rehash immutable CAD")

    with monkeypatch.context() as context:
        context.setattr(payload_motion, "_model_signature", forbidden_hash)
        answer = planner.resolve(_action(planner, reference))
    assert answer.success and answer.qpos_by_side[side] == plan.trajectory.q_rad[0].tolist()
    np.testing.assert_array_equal(scene.data.qpos, before)
    planner.clear()
    planner.clear()
    assert installed["oracle"] is None and not planner.active
    assert binding.check_measured(scene).safe
    with pytest.raises(motion.MujocoTcpPlanRejected, match="reference_not_selected"):
        planner.resolve(_action(planner, reference))
    with pytest.raises(motion.MujocoTcpPlanRejected, match="not_owned"):
        planner.activate(plan)


@pytest.mark.integration
def test_payload_preview_cannot_activate_or_replace_owned_plan(monkeypatch):
    scene = _scene()
    binding = _binding(scene)
    planner, installed = _planner(scene, monkeypatch)
    owned = _plan(planner, binding)
    preview = _plan(planner, binding, preview_model_qpos=scene.data.qpos)
    with pytest.raises(motion.MujocoTcpPlanRejected, match="not_owned"):
        planner.activate(preview)
    assert planner._pending_plan is owned and installed["oracle"] is None
    planner.activate(owned)
    planner.clear()


@pytest.mark.integration
@pytest.mark.parametrize("changed_part", ["arm", "object"])
def test_payload_preview_must_retain_frozen_cut_and_keeps_original_owned_plan(monkeypatch, changed_part):
    scene = _scene()
    binding = _binding(scene)
    planner, installed = _planner(scene, monkeypatch)
    owned = _plan(planner, binding)
    preview_source = scene.data.qpos.copy()
    address = (scene.joint_qpos_addr("right_z") if changed_part == "arm" else
        int(scene.model.jnt_qposadr[scene.model.body_jntadr[scene.model.body("payload").id]]))
    preview_source[address] += .001
    before = scene.data.qpos.copy()
    with pytest.raises(ValueError, match="binding's exact same-cut source qpos"):
        _plan(planner, binding, preview_model_qpos=preview_source)
    assert planner._pending_plan is owned and installed["oracle"] is None and not planner.active
    np.testing.assert_array_equal(scene.data.qpos, before)
    assert scene.data.time == 0.
    planner.clear()


@pytest.mark.integration
@pytest.mark.parametrize("change", ["slip", "geometry", "mocap"])
def test_payload_activation_checks_actual_state_before_installing_guard(monkeypatch, change):
    scene = _scene(extra='''<body name="foreign" mocap="true" pos="1 0 .4">
      <geom type="box" size=".004 .004 .004" contype="8" conaffinity="15"/></body>''')
    binding = _binding(scene)
    planner, installed = _planner(scene, monkeypatch)
    plan = _plan(planner, binding)
    if change == "slip":
        address = int(scene.model.jnt_qposadr[scene.model.body_jntadr[scene.model.body("payload").id]])
        scene.data.qpos[address] += .004
    elif change == "geometry":
        scene.model.geom_size[scene.model.geom("payload_geom").id, 0] += .01
    else:
        scene.data.mocap_pos[0] = (.08, 0., .4)
    with pytest.raises((ValueError, motion.MujocoTcpPlanRejected)):
        planner.activate(plan)
    assert installed["oracle"] is None and not planner.active


@pytest.mark.integration
def test_payload_live_sweep_uses_full_current_cut_under_rig_lock_without_reanchoring(monkeypatch):
    scene = _scene(extra='''<body name="foreign" mocap="true" pos="1 0 .4">
      <geom type="box" size=".004 .004 .004" contype="8" conaffinity="15"/></body>''')
    planner, installed = _planner(scene, monkeypatch)
    guard_calls = []

    def measured_guard(data):
        assert installed["lock_depth"] > 0
        guard_calls.append(float(data.time))
        return JointConfigurationCheck(safe=True)

    binding = planner.rig.capture_scene(lambda s: _binding(
        s, measured_guard=measured_guard, guard_identity="test:one-current-observer"))
    frozen = binding.tool_from_object
    plan = _plan(planner, binding)
    planner.activate(plan)
    # Deterministic native cut perturbation, not a physical transport claim.
    scene.data.time = .002
    scene.data.qvel[:] = .1
    scene.data.qpos[scene.joint_qpos_addr("right_z")] += .004
    address = int(scene.model.body_jntadr[scene.model.body("payload").id])
    scene.data.qpos[int(scene.model.jnt_qposadr[address]) + 2] += .004
    scene.data.mocap_pos[0, 2] += .03
    planner._reference_index = len(plan.reference_poses) - 1
    before = scene.data.qpos.copy()
    count = len(guard_calls)
    assert planner.resolve(_action(planner, plan.reference_poses[-1])).success
    assert len(guard_calls) == count + 1
    source = planner._live_oracle.source
    for name in ("qpos", "qvel", "mocap_pos", "mocap_quat", "eq_active"):
        np.testing.assert_array_equal(getattr(source, name), getattr(scene.data, name))
    assert source.time == scene.data.time and binding.tool_from_object == frozen
    np.testing.assert_array_equal(scene.data.qpos, before)
    planner.clear()


@pytest.mark.integration
def test_payload_dispatch_rejects_new_swept_obstacle_then_actual_slip(monkeypatch):
    scene = _scene(extra='''<body name="foreign" mocap="true" pos="1 0 .4">
      <geom type="box" size=".004 .004 .004" contype="8" conaffinity="15"/></body>''')
    binding = _binding(scene)
    planner, installed = _planner(scene, monkeypatch)
    plan = _plan(planner, binding, x=.3, z=0.)
    planner.activate(plan)
    planner._reference_index = len(plan.reference_poses) - 1
    action = _action(planner, plan.reference_poses[-1])
    scene.data.mocap_pos[0] = (.425, 0., .4)
    with pytest.raises(motion.MujocoTcpPlanRejected, match="live_reference_sweep_blocked"):
        planner.resolve(action)
    address = int(scene.model.jnt_qposadr[scene.model.body_jntadr[scene.model.body("payload").id]])
    scene.data.qpos[address] += .004
    with pytest.raises(motion.MujocoTcpPlanRejected, match="payload_measured_blocked"):
        planner.resolve(action)
    assert installed["oracle"] is binding.contact_oracle(scene)
    planner.clear()


@pytest.mark.integration
def test_payload_shared_measured_guard_rejects_without_publishing_query_history(monkeypatch):
    scene = _scene()
    allowed = True
    binding = _binding(scene, measured_guard=lambda data: JointConfigurationCheck(safe=allowed),
                       guard_identity="test:existing-observer")
    planner, installed = _planner(scene, monkeypatch)
    plan = _plan(planner, binding)
    planner.activate(plan)
    allowed = False
    with pytest.raises(motion.MujocoTcpPlanRejected, match="payload_measured_blocked"):
        planner.resolve(_action(planner, planner.reference(0)))
    assert installed["oracle"] is binding.contact_oracle(scene)
    planner.clear()


@pytest.mark.integration
def test_payload_resolve_retains_passive_arm_and_configuration_constraints(monkeypatch):
    scene = _scene()
    binding = _binding(scene)
    planner, _ = _planner(scene, monkeypatch)
    allowed = True
    constraint = lambda source, pair: JointConfigurationCheck(safe=allowed, reason="test:constraint")
    plan = _plan(planner, binding, configuration_constraint=constraint, constraint_identity="test:fixed")
    planner.activate(plan)
    action = _action(planner, planner.reference(0))
    allowed = False
    with pytest.raises(motion.MujocoTcpPlanRejected, match="live_reference_constraint_blocked"):
        planner.resolve(action)
    allowed = True
    scene.data.qpos[scene.joint_qpos_addr("left_x")] += .002
    with pytest.raises(motion.MujocoTcpPlanRejected, match="passive_arm_changed"):
        planner.resolve(action)
    planner.clear()


@pytest.mark.integration
def test_payload_invalid_binding_mix_and_tool_side_fail_closed(monkeypatch):
    scene = _scene()
    binding = _binding(scene)
    planner, installed = _planner(scene, monkeypatch)
    with pytest.raises(TypeError, match="MujocoPayloadBinding"):
        _plan(planner, True)
    with pytest.raises(ValueError, match="distinct operation bindings"):
        _plan(planner, binding, contact_binding=MujocoGraspContactBinding(
            "payload", ("sensor_left", "sensor_right")))
    # A wrong calibration callback cannot attach a payload to another TCP.
    wrong_base = np.eye(4)
    wrong_base[0, 3] = .1
    planner.scene.body_pose = lambda name: wrong_base
    with pytest.raises(motion.MujocoTcpPlanRejected, match="payload_tool_frame_mismatch"):
        _plan(planner, binding)
    assert installed["oracle"] is None and not planner.active


@pytest.mark.integration
def test_payload_cancelled_plan_and_failed_guard_install_do_not_become_active(monkeypatch):
    scene = _scene()
    binding = _binding(scene)
    planner, installed = _planner(scene, monkeypatch)
    with pytest.raises(motion.MujocoTcpPlanRejected, match="planning_cancelled"):
        planner.plan((0., 0., .43, 0., 0., 0., 1.),
            translation_tolerance_m=.001, rotation_tolerance_rad=.01,
            cancel_requested=lambda: True, payload_binding=binding)
    assert planner._pending_plan is None and not planner.active and installed["oracle"] is None
    plan = _plan(planner, binding)

    def reject_install(oracle):
        raise RuntimeError("test:scoped guard installation unavailable")

    monkeypatch.setattr(planner.rig, "set_additional_contact_oracle", reject_install)
    with pytest.raises(RuntimeError, match="installation unavailable"):
        planner.activate(plan)
    assert planner._pending_plan is plan and not planner.active and installed["oracle"] is None
    assert not planner._contact_guard_bound
    planner.clear()


@pytest.mark.integration
def test_payload_timestep_change_invalidates_active_guard_without_dispatch(monkeypatch):
    scene = _scene()
    binding = _binding(scene)
    planner, installed = _planner(scene, monkeypatch)
    plan = _plan(planner, binding)
    planner.activate(plan)
    action = _action(planner, planner.reference(0))
    before = scene.data.qpos.copy()
    scene.model.opt.timestep *= .5
    with pytest.raises(ValueError, match="timestep changed"):
        planner.resolve(action)
    np.testing.assert_array_equal(scene.data.qpos, before)
    assert installed["oracle"] is binding._measured_oracle
    planner.clear()


@pytest.mark.integration
def test_payload_candidate_selection_keeps_one_guarded_plan_without_motion(monkeypatch):
    scene = _scene(table=True)
    binding = _binding(scene, support=_support())
    planner, installed = _planner(scene, monkeypatch)
    before = (scene.data.qpos.copy(), scene.data.qvel.copy(), float(scene.data.time))
    goals = [(0., 0., .43, 0., 0., 0., 1.)] * 2
    selection = planner.plan_candidates(
        goals, translation_tolerance_m=.001, rotation_tolerance_rad=.01,
        cancel_requested=lambda: False, payload_binding=binding,
        time_budget_sec=8., path_search_time_budget_sec=3.)
    assert selection.candidate_index == 0 and selection.all_candidates_checked
    assert selection.plan.payload_binding is binding
    assert planner.owns_pending_plan(selection.plan)
    assert installed["oracle"] is None and not planner.active
    np.testing.assert_array_equal(scene.data.qpos, before[0])
    np.testing.assert_array_equal(scene.data.qvel, before[1])
    assert float(scene.data.time) == before[2]
    planner.activate(selection.plan)
    assert installed["oracle"] is binding.contact_oracle(scene)
    planner.clear()
    assert installed["oracle"] is None
    with pytest.raises(TypeError, match="MujocoPayloadBinding"):
        planner.plan_candidates(goals, translation_tolerance_m=.001,
            rotation_tolerance_rad=.01, cancel_requested=lambda: False,
            payload_binding=True)
    with pytest.raises(ValueError, match="full guarded plans"):
        planner.plan_candidates(goals, translation_tolerance_m=.001,
            rotation_tolerance_rad=.01, cancel_requested=lambda: False,
            payload_binding=binding, prioritize_current_seed=True)
    frozen = motion.MujocoPlannedTcpMotion(
        rig=planner.rig, task_config=planner.task_config,
        pose_projector=planner.pose_projector, reference_hz=planner.reference_hz,
        motion_side="right", source_model_qpos=scene.data.qpos.copy())
    with pytest.raises(ValueError, match="live owned planner"):
        frozen.plan_candidates(goals, translation_tolerance_m=.001,
            rotation_tolerance_rad=.01, cancel_requested=lambda: False,
            payload_binding=binding)
