"""Native feedback and actual Gateway wiring, not pick/place success."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.mujoco.motion_factory import (
    build_mujoco_agent_tcp_pose_gateway_session, mujoco_agent_tcp_pose_execution_observation_plan_v1,
)
from agent_gateway.profiles.mujoco.object_state import MujocoObjectStateSource
from agent_gateway.profiles.object_state import ObjectStateProfile
from project_paths import resolve_path
from sim.mujoco.deploy import MujocoSimHILTargetConfig
from sim.mujoco.transforms import pose_to_xyz_quat_xyzw

pytestmark = pytest.mark.integration
TASK = Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")
CLIENT = "test:mujoco-tcp-comparison"


def test_object_state_is_absent_from_the_default_native_catalog():
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(task_config_path=resolve_path(TASK), realtime_worker=False,
            runtime_session_id=CLIENT, control_domain_id=CLIENT,
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",), allow_scripted_task_reset=False),
        allowed_client_ids=(CLIENT,))
    try:
        session.application.start_observation_ready()
        assert session.contract_bundle.object_state_registration is None
        assert session.object_state_executor is None
        assert all(entry.route.skill_id != "state.read_object_state"
                   for entry in session.contract_bundle.route_manifest.skill_routes)
    finally:
        assert session.close().accepted


def test_object_state_requires_the_explicit_contact_target_before_resource_creation():
    with pytest.raises(ValueError, match="contact target"):
        build_mujoco_agent_tcp_pose_gateway_session(
            target_config=MujocoSimHILTargetConfig(task_config_path=resolve_path(TASK), realtime_worker=False,
                runtime_session_id=CLIENT, control_domain_id=CLIENT,
                execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
                exposed_pose_sides=("right",), allow_scripted_task_reset=False),
            allowed_client_ids=(CLIENT,), enable_object_state=True)


def test_follow_observation_requires_explicit_state_before_resource_creation():
    with pytest.raises(ValueError, match="object-state"):
        build_mujoco_agent_tcp_pose_gateway_session(
            target_config=MujocoSimHILTargetConfig(task_config_path=resolve_path(TASK), realtime_worker=False,
                runtime_session_id=CLIENT, control_domain_id=CLIENT,
                execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
                exposed_pose_sides=("right",), allow_scripted_task_reset=False),
            allowed_client_ids=(CLIENT,), enable_object_follow_observation=True)


def test_existing_single_cut_deployment_does_not_bind_a_follow_observer():
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(task_config_path=resolve_path(TASK), realtime_worker=False,
            runtime_session_id=CLIENT, control_domain_id=CLIENT,
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",), allow_scripted_task_reset=False),
        allowed_client_ids=(CLIENT,), enable_grippers=True, enable_grasp_approach=True,
        grasp_contact_target_body="debug_cube", enable_object_state=True)
    try:
        session.application.start_observation_ready()
        assert not session.contract_bundle.object_state_registration.profile.release_supported
        assert session.application.runtime.rig._physics_step_observer is None
    finally:
        assert session.close().accepted


def test_current_native_feedback_uses_the_public_frame_and_never_advances_physics():
    from test_agent_gateway_mujoco_tcp_pose_comparison import _request

    facts = []
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(task_config_path=resolve_path(TASK), realtime_worker=False,
            runtime_session_id=CLIENT, control_domain_id=CLIENT,
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",), allow_scripted_task_reset=False),
        allowed_client_ids=(CLIENT,), enable_grippers=True, enable_grasp_approach=True,
        grasp_contact_target_body="debug_cube", enable_object_state=True,
        enable_object_follow_observation=True, trace_sink=facts.append)
    try:
        session.application.start_observation_ready()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT, read_only_access=True)
        reg = session.contract_bundle.object_state_registration
        assert reg is not None
        assert reg.profile.release_supported
        rig = session.application.runtime.rig
        before = rig.capture_scene(lambda s: (s.data.qpos.copy(), s.data.qvel.copy(), float(s.data.time)))
        params = {"object_id": "debug_cube", "reference_frame_id": reg.profile.reference_frame_id,
                  "max_measurement_age_ms": 50}
        request = _request(session, adapter, skill_id="state.read_object_state", version="1", parameters=params, sequence=0)
        admitted = adapter.submit_read_only_skill_request(request.to_wire())
        assert admitted["accepted"], admitted
        assert session.object_state_executor.wait_all(timeout_sec=10)
        op = session.service.get_operation(admitted["operation"]["operation_id"])
        assert op.execution_outcome == "completed", op.to_wire()
        state = op.execution_result
        assert state["observation_source"] == "mujoco_native_physics"
        assert state["release_observation"] is None
        assert rig._physics_step_observer is not None
        assert state["clock_domain_id"] == "mujoco-native-physics:" + CLIENT
        after = rig.capture_scene(lambda s: (s.data.qpos.copy(), s.data.qvel.copy(), float(s.data.time)))
        assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
        expected = session.application.transform_tcp_pose_xyzquat(
            tuple(pose_to_xyz_quat_xyzw(session.application.scene.body_pose("debug_cube"))),
            source_frame_id="mujoco_sim_hil/world", target_frame_id=reg.profile.reference_frame_id)
        np.testing.assert_allclose(state["pose_xyz_quat_xyzw"], expected, atol=1e-10)
        assert not any(f.event_type == "component_execution_target" for f in facts)
    finally:
        assert session.close().accepted
    assert rig._physics_step_observer is None


def test_native_close_command_generation_is_owner_bound_and_rejected_commands_do_not_renew_it():
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(task_config_path=resolve_path(TASK), realtime_worker=False,
            runtime_session_id=CLIENT, control_domain_id=CLIENT,
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",), allow_scripted_task_reset=False),
        allowed_client_ids=(CLIENT,), enable_grippers=True, enable_grasp_approach=True,
        grasp_contact_target_body="debug_cube", enable_object_state=True, enable_object_follow_observation=True)
    try:
        session.application.start_observation_ready()
        rig = session.application.runtime.rig
        right, left = (session.gripper_ports[side + "_gripper"].gripper_name for side in ("right", "left"))
        assert rig.latest_gripper_command(right) == (0, None)
        sequence = rig.queue_gripper_close_targets({right: 1.})
        assert rig.latest_gripper_command(right) == (sequence, 1.)
        assert rig.latest_gripper_command(left) == (0, None)
        with pytest.raises(ValueError):
            rig.queue_gripper_close_targets({right: float("nan")})
        assert rig.latest_gripper_command(right) == (sequence, 1.)
        rig.step_physics()
        rig.hold_gripper_motion("test: original intent is not a HOLD reference")
        assert rig.latest_gripper_command(right) == (sequence, 1.)
        assert rig._physics_step_observer.__self__.latest is None
    finally:
        assert session.close().accepted


def _small_source(*, gap=0., rotate=False):
    import mujoco

    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="table"><geom type="plane" size="1 1 .1"/></body>
      <body name="target" pos="0 0 {.025 + gap}"><freejoint/>
        <inertial pos=".005 0 0" mass=".1" diaginertia=".001 .001 .001"/>
        <geom type="sphere" size=".025"/>
      </body>
      <body name="sensor" pos="1 0 0"><geom name="pad" type="box" size=".01 .01 .01"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qvel[:3] = [.1, .2, .3]
    data.qvel[3:] = [0., 0., .4]
    # Deliberately leave derived transforms stale: capture must derive them privately.
    pose = np.eye(4)
    if rotate:
        pose[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    scene = SimpleNamespace(mujoco=mujoco, model=model, data=data)
    graph = SimpleNamespace(resolve=lambda *_: SimpleNamespace(matrix_target_source=pose),
        semantic_content_hash="graph", frame_by_id={"station": SimpleNamespace(semantic_content_hash="spec")})
    app = SimpleNamespace(scene=scene, task_config=SimpleNamespace(table_body_name="table"),
        target_frame_profile=SimpleNamespace(frame_graph=graph),
        runtime=SimpleNamespace(config=SimpleNamespace(runtime_session_id="unit")),
        capture_passive_application_cut=lambda **kw: SimpleNamespace(lifecycle_state="held", scene_projection=kw['scene_projector'](scene)))
    profile = ObjectStateProfile(object_id="target", reference_frame_id="station",
        reference_frame_spec_hash="spec", frame_graph_hash="graph", clock_domain_id="mujoco-native-physics:unit",
        observation_source="mujoco_native_physics", source_binding_hash="binding")
    source = MujocoObjectStateSource(application=app, profile=profile,
        contact_binding=SimpleNamespace(target_body_name="target", sensor_geom_names=("pad",)))
    return scene, source


@pytest.mark.parametrize("rotate", [False, True])
def test_private_native_jacobian_velocity_and_contacts_use_the_same_cut(rotate):
    scene, source = _small_source(rotate=rotate)
    before = (scene.data.qpos.copy(), scene.data.qvel.copy(), scene.data.xpos.copy(), float(scene.data.time))
    state = source.capture()
    after = (scene.data.qpos.copy(), scene.data.qvel.copy(), scene.data.xpos.copy(), float(scene.data.time))
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
    assert state["support_contact"] and not state["gripper_contact"] and not state["other_contact"]
    np.testing.assert_allclose(state["linear_velocity_xyz_m_s"], [-.2, .1, .3] if rotate else [.1, .2, .3])
    np.testing.assert_allclose(state["angular_velocity_xyz_rad_s"], [0., 0., .4])
    assert state["release_observation"] is None


def test_positive_contact_margin_does_not_become_measured_support():
    scene, source = _small_source(gap=.001)
    assert not source.capture()["support_contact"]
    scene.data.qvel[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        source.capture()


def test_native_release_capability_requires_its_bound_producer():
    _, source = _small_source()
    with pytest.raises(ValueError, match="release-window"):
        MujocoObjectStateSource(application=source.application,
            profile=replace(source.profile, release_supported=True), contact_binding=source.contact_binding)


@pytest.mark.parametrize("change", [{"reference_frame_spec_hash": "foreign"},
    {"frame_graph_hash": "foreign"}, {"clock_domain_id": "wall-clock"}])
def test_native_object_source_rejects_mismatched_frame_and_clock_at_construction(change):
    _, source = _small_source()
    with pytest.raises(ValueError, match="frame/clock"):
        MujocoObjectStateSource(application=source.application,
            profile=replace(source.profile, **change), contact_binding=source.contact_binding)
