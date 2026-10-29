"""Carried-object contracts and native lifecycle wiring, not grasp success."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from agent_gateway.profiles.mujoco.motion_deployment import MujocoAgentTcpPoseMotionProfile
from agent_gateway.profiles.mujoco.motion_factory import (
    build_mujoco_agent_tcp_pose_gateway_session,
    mujoco_agent_tcp_pose_execution_observation_plan_v1,
)
from agent_gateway.profiles.mujoco.motion_skill import MujocoAgentSafeTcpPoseSkillLifecycle
from agent_gateway.skills.motion.grasped_object import (
    GRASPED_OBJECT_MOVE_SKILL_ID, grasped_object_move_definition, grasped_object_move_schema,
    grasped_object_place_definition_v2, grasped_object_place_schema_v2,
    validate_grasped_object_move, validate_grasped_object_place_v2,
    validate_object_motion_state, validate_object_motion_state_v2,
)
from project_paths import resolve_path
from sim.mujoco.deploy import MujocoSimHILTargetConfig
from sim.mujoco.actuation.executor import MujocoPhysicalContactFault

CLIENT = "test:mujoco-tcp-comparison"
TASK = Path("sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json")


def _profile():
    return MujocoAgentTcpPoseMotionProfile(reference_frame_id="station", reference_frame_spec_hash="spec",
        frame_graph_hash="graph", tool_config_hash="tool", calibration_hash="calibration")


def _parameters(profile, **changes):
    return {**{name: getattr(profile, name) for name in (
        "reference_frame_id", "reference_frame_spec_hash", "frame_graph_hash", "tool_config_hash", "calibration_hash")},
        "controlled_frame_id": "object_body", "component_id": profile.component_id,
        "object_id": "debug_cube", "phase": "lift", "translation_delta_xyz_m": [0., 0., .04], **changes}


def _session(*, enabled=True, **changes):
    options = dict(enable_grippers=True, enable_grasp_approach=True, grasp_contact_target_body="debug_cube",
                   enable_object_state=True, enable_object_follow_observation=True, enable_object_motion=enabled)
    options.update(changes)
    return build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(task_config_path=resolve_path(TASK), realtime_worker=False,
            runtime_session_id=CLIENT, control_domain_id=CLIENT,
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",), allow_scripted_task_reset=False),
        allowed_client_ids=(CLIENT,), **options)


@pytest.mark.unit
def test_object_goal_schema_and_validation_keep_exact_object_body_identity():
    profile = _profile()
    definition = grasped_object_move_definition(controlled_component_ids=("right_arm",))
    schema = grasped_object_move_schema(definition=definition, profile=profile, object_id="debug_cube",
        observation_source="mujoco_native_physics", clock_domain_id="native:test")
    assert schema.parameter_schema["properties"]["controlled_frame_id"]["const"] == "object_body"
    parameters = _parameters(profile)
    validate_grasped_object_move(parameters, profile=profile, object_id="debug_cube")
    parameters.pop("translation_delta_xyz_m")
    parameters["object_pose_xyz_quat_xyzw"] = [0., 0., .3, 0., 0., 0., 1.]
    validate_grasped_object_move(parameters, profile=profile, object_id="debug_cube")
    parameters["translation_delta_xyz_m"] = [0., 0., .04]
    with pytest.raises(ValueError, match="exact"):
        validate_grasped_object_move(parameters, profile=profile, object_id="debug_cube")


@pytest.mark.unit
def test_object_support_v2_requires_named_place_and_exact_terminal_support():
    profile = _profile()
    definition = grasped_object_place_definition_v2(controlled_component_ids=("right_arm",))
    schema = grasped_object_place_schema_v2(
        definition=definition, profile=profile, object_id="debug_cube",
        support_object_ids=("large_blue_cube", "red_box"),
        observation_source="mujoco_native_physics", clock_domain_id="mujoco-native-physics:test")
    assert definition.skill_id == GRASPED_OBJECT_MOVE_SKILL_ID and schema.skill_version == "2"
    assert schema.parameter_schema["properties"]["phase"] == {"const": "place"}
    parameters = _parameters(profile, phase="place", support_object_id="large_blue_cube")
    validate_grasped_object_place_v2(parameters, profile=profile, object_id="debug_cube",
                                      support_object_ids=("large_blue_cube", "red_box"))
    for change in ({"support_object_id": "foreign"}, {"phase": "lift"}):
        with pytest.raises(ValueError):
            validate_grasped_object_place_v2({**parameters, **change}, profile=profile,
                                              object_id="debug_cube",
                                              support_object_ids=("large_blue_cube", "red_box"))
    state = dict(object_id="debug_cube", phase="place", reference_frame_id="station",
                 pose_xyz_quat_xyzw=[0., 0., .3, 0., 0., 0., 1.], measured_at=0.,
                 clock_domain_id="mujoco-native-physics:test", holding_verified=True,
                 support_contact=True, source="mujoco_native_physics",
                 support_object_id="large_blue_cube")
    validate_object_motion_state_v2(state, parameters=parameters, profile=profile,
        object_id="debug_cube", observation_source="mujoco_native_physics",
        clock_domain_id="mujoco-native-physics:test")
    with pytest.raises(ValueError, match="terminal support differs"):
        validate_object_motion_state_v2({**state, "support_object_id": "red_box"},
            parameters=parameters, profile=profile, object_id="debug_cube",
            observation_source="mujoco_native_physics", clock_domain_id="mujoco-native-physics:test")


@pytest.mark.unit
@pytest.mark.parametrize("changes", [
    {"object_id": "neighbor"}, {"controlled_frame_id": "tool_tcp"}, {"component_id": "left_arm"},
    {"calibration_hash": "old"}, {"phase": "teleport"}, {"translation_delta_xyz_m": [True, 0., 0.]},
    {"translation_delta_xyz_m": [float("nan"), 0., 0.]}, {"translation_delta_xyz_m": [0., 0.]},
    {"contact_exception": True},
])
def test_object_goal_rejects_foreign_or_invalid_parameters(changes):
    with pytest.raises((TypeError, ValueError)):
        validate_grasped_object_move(_parameters(_profile(), **changes), profile=_profile(), object_id="debug_cube")


@pytest.mark.unit
@pytest.mark.parametrize("change", [{}, {"holding_verified": False}, {"phase": "place"},
    {"support_contact": True}, {"measured_at": True}, {"clock_domain_id": "wall-clock"}])
def test_object_terminal_state_keeps_native_identity_and_physical_phase(change):
    state = dict(object_id="debug_cube", phase="carry", reference_frame_id="station",
        pose_xyz_quat_xyzw=[0., 0., .3, 0., 0., 0., 1.], measured_at=1., clock_domain_id="native:test",
        holding_verified=True, support_contact=False, source="mujoco_native_physics")
    state.update(change)
    if change:
        with pytest.raises(ValueError):
            validate_object_motion_state(state, profile=_profile(), object_id="debug_cube",
                observation_source="mujoco_native_physics", clock_domain_id="native:test")
    else:
        validate_object_motion_state(state, profile=_profile(), object_id="debug_cube",
            observation_source="mujoco_native_physics", clock_domain_id="native:test")


@pytest.mark.unit
def test_composed_target_cancellation_precedes_measurement_or_resolution():
    lifecycle = object.__new__(MujocoAgentSafeTcpPoseSkillLifecycle)
    lifecycle._validate_call = lambda **kwargs: None
    lifecycle._capture_pose = lambda: pytest.fail("Cancelled activation must not capture/resolve")
    evidence = lifecycle.activate(request=None, operation=None, cancel_requested=lambda: True)
    assert not evidence.succeeded and evidence.lower_operation_id is None
    assert evidence.status == "tcp_pose_activation_cancelled_before_lower_start"


@pytest.mark.unit
def test_default_terminal_hash_input_keeps_original_tcp_contract():
    lifecycle = object.__new__(MujocoAgentSafeTcpPoseSkillLifecycle)
    lifecycle.request = SimpleNamespace(runtime_session_id="session")
    lifecycle.operation = SimpleNamespace(operation_id="operation")
    lifecycle.component_id = "right_arm"
    lifecycle.deployment_binding = SimpleNamespace(control_domain_id="domain")
    lifecycle.motion_profile = _profile()
    pose = (0., 0., .3, 0., 0., 0., 1.)
    assert lifecycle._terminal_state_evidence(pose, 1.) == dict(
        runtime_session_id="session", control_domain_id="domain", operation_id="operation",
        component_id="right_arm", pose_xyz_quat_xyzw=pose, measured_at=1.,
        reference_frame_id="station", controlled_frame_id="tool_tcp")


@pytest.mark.unit
def test_object_terminal_hash_input_contains_measured_body_goal_and_native_clock():
    from agent_gateway.profiles.mujoco.object_motion import MujocoAgentObjectMotionLifecycle

    lifecycle = object.__new__(MujocoAgentObjectMotionLifecycle)
    lifecycle.request = SimpleNamespace(runtime_session_id="session")
    lifecycle.operation = SimpleNamespace(operation_id="operation")
    lifecycle.component_id = "right_arm"
    lifecycle.deployment_binding = SimpleNamespace(control_domain_id="domain")
    lifecycle.motion_profile = _profile()
    lifecycle._object_id, lifecycle._phase = "debug_cube", "carry"
    lifecycle._support_object_id = None
    obj = np.eye(4)
    obj[:3, 3] = (.1, .2, .3)
    cut = SimpleNamespace(world_from_object=obj, sim_time_s=2., status="held_verified", support_contact=False)
    lifecycle._object_observer = SimpleNamespace(observe=lambda **kwargs: cut)
    lifecycle._public_from_world = lambda: np.eye(4)
    evidence = lifecycle._terminal_state_evidence((0., 0., .4, 0., 0., 0., 1.), 2.)
    assert evidence["object_state"] is lifecycle._object_terminal
    assert evidence["object_state"]["pose_xyz_quat_xyzw"] == [.1, .2, .3, 0., 0., 0., 1.]
    assert evidence["object_state"]["clock_domain_id"] == "mujoco-native-physics:session"
    validate_object_motion_state(evidence["object_state"], profile=_profile(), object_id="debug_cube",
        observation_source="mujoco_native_physics", clock_domain_id="mujoco-native-physics:session")


@pytest.mark.unit
@pytest.mark.parametrize("activated", [False, True])
def test_object_stage_commits_only_after_plan_activation_on_the_live_cut(monkeypatch, activated):
    from agent_gateway.profiles.mujoco.object_motion import MujocoAgentObjectMotionLifecycle

    calls = []
    scene = SimpleNamespace(data=object())
    lifecycle = object.__new__(MujocoAgentObjectMotionLifecycle)
    lifecycle._phase = "place"
    lifecycle._support_object_id = None
    lifecycle.application = SimpleNamespace(runtime=SimpleNamespace(
        rig=SimpleNamespace(capture_scene=lambda callback: callback(scene))))
    lifecycle._object_observer = SimpleNamespace(observe=lambda **kwargs: calls.append(kwargs))
    evidence = SimpleNamespace(succeeded=activated)
    monkeypatch.setattr(MujocoAgentSafeTcpPoseSkillLifecycle, "activate", lambda *args, **kwargs: evidence)
    assert lifecycle.activate() is evidence
    assert calls == ([{"phase": "place", "source_data": scene.data}] if activated else [])


@pytest.mark.unit
@pytest.mark.parametrize("pair_class,expected", [
    ("payload_relation", "object_follow_relation_lost_after_hold"),
    ("payload_support_moved", "object_support_moved_after_hold"),
    ("8:1", None),
])
def test_object_motion_only_classifies_typed_payload_fault_after_hold(monkeypatch, pair_class, expected):
    from agent_gateway.profiles.mujoco.object_motion import MujocoAgentObjectMotionLifecycle
    lifecycle = object.__new__(MujocoAgentObjectMotionLifecycle)
    fault = MujocoPhysicalContactFault("native guard", SimpleNamespace(pair_class=pair_class))
    monkeypatch.setattr(MujocoAgentSafeTcpPoseSkillLifecycle, "run",
                        lambda *args, **kwargs: (_ for _ in ()).throw(fault))
    holds = []
    lifecycle._hold = lambda *, reason: holds.append(reason) or object()
    lifecycle._failure = lambda *, reason_code, hold: reason_code
    if expected is None:
        with pytest.raises(MujocoPhysicalContactFault):
            lifecycle.run()
        assert not holds
    else:
        assert lifecycle.run() == expected
        assert holds == [expected]


@pytest.mark.unit
@pytest.mark.parametrize("changes", [{"enable_object_motion": "yes"},
    {"enable_object_follow_observation": False}, {"enable_grippers": False},
    {"enable_grasp_approach": False}, {"grasp_contact_target_body": None}])
def test_object_motion_invalid_combinations_reject_before_scene_creation(monkeypatch, changes):
    from agent_gateway.profiles.mujoco import motion_factory
    monkeypatch.setattr(motion_factory, "prepare_mujoco_sim_hil_target",
                        lambda *args, **kwargs: pytest.fail("Invalid composition must not construct the plant"))
    with pytest.raises((TypeError, ValueError)):
        _session(**changes)


@pytest.mark.integration
@pytest.mark.parametrize("enabled", [False, True])
def test_native_object_route_is_explicit_and_reuses_exact_tick_observer(enabled):
    session = _session(enabled=enabled)
    try:
        app = session.application
        app.start_observation_ready()
        assert (session.contract_bundle.object_motion_contract is not None) is enabled
        assert (session.object_motion_executor is not None) is enabled
        if enabled:
            assert app.runtime.rig._physics_step_observer.__self__ is session.object_observer
            assert session.object_motion_executor in session._motion_executors()
            contract = session.contract_bundle.object_motion_contract
            entry = session.contract_bundle.route_manifest.route(GRASPED_OBJECT_MOVE_SKILL_ID, "1")
            assert len(entry.deployment_binding.safety_envelope_refs) == 1
            assert entry.deployment_binding.deployment_binding_hash == contract.deployment_binding.semantic_content_hash
            validator = session.contract_bundle.payload_validators.validator_for(entry.route)
            validator.validate_parameters(_parameters(contract.motion_profile))
            parameters = _parameters(contract.motion_profile, phase="carry")
            result = dict(execution_receipt_refs=["test:actual-hold"], final_measured_state_hash="test:state",
                object_state=dict(object_id="debug_cube", phase="carry", reference_frame_id=contract.motion_profile.reference_frame_id,
                    pose_xyz_quat_xyzw=[0., 0., .3, 0., 0., 0., 1.], measured_at=0.,
                    clock_domain_id="mujoco-native-physics:" + CLIENT, holding_verified=True,
                    support_contact=False, source="mujoco_native_physics"))
            validator.validate_execution_result_for_parameters(parameters, result)
            result["object_state"].update(phase="place", support_contact=True)
            validator.validate_execution_result(result)
            with pytest.raises(ValueError, match="admitted request phase"):
                validator.validate_execution_result_for_parameters(parameters, result)
        else:
            assert all(e.route.skill_id != GRASPED_OBJECT_MOVE_SKILL_ID
                       for e in session.contract_bundle.route_manifest.skill_routes)
        assert app.scene.data.time == 0.
    finally:
        assert session.close().accepted
    assert app.runtime.rig._physics_step_observer is None


@pytest.mark.integration
def test_dynamic_support_binding_uses_measured_tilted_top_without_stepping(tmp_path):
    from agent_gateway.profiles.mujoco.object_motion import _native_support_for_object_motion
    from sim.mujoco.scene_suite import materialize_case

    task_path = materialize_case("same_color_sizes", tmp_path / "case")
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=task_path, realtime_worker=False,
            runtime_session_id=CLIENT, control_domain_id=CLIENT,
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",), allow_scripted_task_reset=False),
        allowed_client_ids=(CLIENT,), enable_grippers=True,
        enable_grasp_approach=True, grasp_contact_target_body="debug_cube",
        enable_object_state=True, enable_object_follow_observation=True,
        enable_object_motion=True, enable_object_support_motion_v2=True,
        object_support_body_names=("large_blue_cube",))
    try:
        assert session.application.runtime.config.prune_expired_proposals is True
        session.application.start_observation_ready()
        scene = session.application.scene
        model, data = scene.model, scene.data
        body = int(model.body("large_blue_cube").id)
        address = int(model.jnt_qposadr[model.body_jntadr[body]])
        quaternion = Rotation.from_euler("y", .2, degrees=True).as_quat()
        data.qpos[address + 3:address + 7] = [quaternion[3], *quaternion[:3]]
        before = (data.qpos.copy(), float(data.time))
        support = session.application.runtime.rig.capture_scene(
            lambda current: _native_support_for_object_motion(
                current, body_name="large_blue_cube", phase="place",
                max_penetration_m=.001, dynamic=True))
        assert support.dynamic_body_name == "large_blue_cube"
        assert support.region_min_world[2] < support.region_max_world[2]
        expected = Rotation.from_euler("y", .2, degrees=True).apply((0., 0., 1.))
        np.testing.assert_allclose(support.normal_world, expected, atol=1e-8)
        np.testing.assert_array_equal(data.qpos, before[0])
        assert float(data.time) == before[1]
        tilted = Rotation.from_euler("y", 2., degrees=True).as_quat()
        data.qpos[address + 3:address + 7] = [tilted[3], *tilted[:3]]
        with pytest.raises(ValueError, match="near-horizontal"):
            session.application.runtime.rig.capture_scene(
                lambda current: _native_support_for_object_motion(
                    current, body_name="large_blue_cube", phase="place",
                    max_penetration_m=.001, dynamic=True))
        for angle in (90., 180.):
            sideways = Rotation.from_euler("x", angle, degrees=True).as_quat()
            data.qpos[address + 3:address + 7] = [sideways[3], *sideways[:3]]
            support = session.application.runtime.rig.capture_scene(
                lambda current: _native_support_for_object_motion(
                    current, body_name="large_blue_cube", phase="place",
                    max_penetration_m=.001, dynamic=True))
            np.testing.assert_allclose(support.normal_world, (0., 0., 1.), atol=1e-8)
            assert support.region_max_world[2] - support.region_min_world[2] < 1e-8
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_object_support_v2_is_opt_in_and_reuses_the_same_physics_owner(tmp_path, monkeypatch):
    from sim.mujoco.scene_suite import materialize_case

    task_path = materialize_case("same_color_sizes", tmp_path / "case")
    options = dict(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=task_path, realtime_worker=False,
            runtime_session_id=CLIENT, control_domain_id=CLIENT,
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",), allow_scripted_task_reset=False),
        allowed_client_ids=(CLIENT,), enable_grippers=True,
        enable_grasp_approach=True, grasp_contact_target_body="debug_cube",
        enable_object_state=True, enable_object_follow_observation=True,
        enable_object_motion=True)
    session = build_mujoco_agent_tcp_pose_gateway_session(
        **options, enable_object_support_motion_v2=True,
        object_support_body_names=("large_blue_cube", "medium_blue_cube"))
    try:
        session.application.start_observation_ready()
        bundle = session.contract_bundle
        assert bundle.object_motion_contract is not None
        assert bundle.object_support_motion_contract_v2 is not None
        v1_route_hash = bundle.route_manifest.route(GRASPED_OBJECT_MOVE_SKILL_ID, "1").route.semantic_content_hash
        assert bundle.object_support_body_names == ("large_blue_cube", "medium_blue_cube")
        assert session.object_observer.support_body_names == (
            "table", "large_blue_cube", "medium_blue_cube")
        assert session.application.runtime.rig._physics_step_observer.__self__ is session.object_observer
        route = bundle.route_manifest.route(GRASPED_OBJECT_MOVE_SKILL_ID, "2")
        assert route.route.route_version == "2"
        validator = bundle.payload_validators.validator_for(route.route)
        profile = bundle.object_support_motion_contract_v2.motion_profile
        parameters = {**_parameters(profile, phase="place",
                                   translation_delta_xyz_m=[0., 0., 0.]),
                      "support_object_id": "large_blue_cube"}
        validator.validate_parameters(parameters)
        with pytest.raises(ValueError):
            validator.validate_parameters({**parameters, "support_object_id": "unbound"})
        state = dict(object_id="debug_cube", phase="place",
            reference_frame_id=profile.reference_frame_id,
            pose_xyz_quat_xyzw=[0., 0., .3, 0., 0., 0., 1.], measured_at=0.,
            clock_domain_id="mujoco-native-physics:" + CLIENT, holding_verified=True,
            support_contact=True, source="mujoco_native_physics",
            support_object_id="large_blue_cube")
        validator.validate_execution_result_for_parameters(parameters,
            dict(execution_receipt_refs=["test:hold"], final_measured_state_hash="test:state",
                 object_state=state))
        with pytest.raises(ValueError, match="terminal support differs"):
            validator.validate_execution_result_for_parameters(parameters,
                dict(execution_receipt_refs=["test:hold"], final_measured_state_hash="test:state",
                     object_state={**state, "support_object_id": "medium_blue_cube"}))
        from agent_gateway.profiles.mujoco.object_motion import MujocoAgentObjectMotionLifecycle
        from sim.mujoco.deploy.payload_motion import MujocoPayloadBinding
        native = session.application.scene
        before_qpos = native.data.qpos.copy()
        before_time = float(native.data.time)
        captured = {}
        with monkeypatch.context() as patcher:
            def capture(_cls, _scene, _data, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(semantic_content_hash="test:frozen-payload")
            patcher.setattr(MujocoPayloadBinding, "capture", classmethod(capture))
            lifecycle = object.__new__(MujocoAgentObjectMotionLifecycle)
            lifecycle.application = session.application
            lifecycle.motion_profile = profile
            lifecycle.request = SimpleNamespace(parameters=parameters)
            lifecycle._phase = "place"
            lifecycle._object_id = "debug_cube"
            lifecycle._support_object_id = "large_blue_cube"
            lifecycle._object_observer = SimpleNamespace(
                observe=lambda **kwargs: SimpleNamespace(
                    world_from_object=native.body_pose("debug_cube"),
                    world_from_tcp=native.body_pose("debug_cube"),
                    tool_from_object=np.eye(4), command_generation=1),
                sensor_geom_names=session.object_observer.sensor_geom_names,
                gripper_name=session.object_observer.gripper_name,
                command_state=lambda: (1, 1.),
                world_tcp_from_scene=lambda view: np.eye(4),
                measured_closed=lambda value: value > 0.)
            lifecycle._resolve_object_goal(None)
        assert captured["support"].dynamic_body_name == "large_blue_cube"
        assert captured["support"].phase == "place"
        assert set(captured["support"].geom_names) == {
            native.model.geom(i).name for i in range(native.model.ngeom)
            if native.model.geom_bodyid[i] == native.model.body("large_blue_cube").id
            and (native.model.geom_contype[i] or native.model.geom_conaffinity[i])}
        np.testing.assert_array_equal(native.data.qpos, before_qpos)
        assert float(native.data.time) == before_time
        assert session.application.scene.data.time == 0.
        from test_agent_gateway_mujoco_tcp_pose_comparison import _request
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        before = session.application.scene.data.qpos.copy()
        request = _request(session, adapter, skill_id=GRASPED_OBJECT_MOVE_SKILL_ID,
                           version="2", parameters=parameters, sequence=0)
        admitted = adapter.submit_skill_request(request.to_wire())
        assert admitted["accepted"]
        assert session.wait(timeout_sec=30.)
        operation = session.service.get_operation(admitted["operation"]["operation_id"])
        assert operation.execution_outcome == "failed"
        assert operation.reason_code == "object_motion_measured_relation_unavailable"
        assert operation.progress_summary["lower_hold_confirmed"] is True
        np.testing.assert_array_equal(session.application.scene.data.qpos, before)
    finally:
        assert session.close().accepted
    baseline = build_mujoco_agent_tcp_pose_gateway_session(**options)
    try:
        assert baseline.contract_bundle.object_support_motion_contract_v2 is None
        assert baseline.contract_bundle.route_manifest.route(
            GRASPED_OBJECT_MOVE_SKILL_ID, "1").route.semantic_content_hash == v1_route_hash
        with pytest.raises(KeyError):
            baseline.contract_bundle.route_manifest.route(GRASPED_OBJECT_MOVE_SKILL_ID, "2")
        baseline.application.start_observation_ready()
    finally:
        assert baseline.close().accepted


@pytest.mark.integration
def test_named_release_v3_reuses_one_gripper_skill_and_the_existing_observer(tmp_path, monkeypatch):
    from sim.mujoco.scene_suite import materialize_case

    task_path = materialize_case("same_color_sizes", tmp_path / "case")
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=task_path, realtime_worker=False,
            runtime_session_id=CLIENT, control_domain_id=CLIENT,
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",), allow_scripted_task_reset=False),
        allowed_client_ids=(CLIENT,), enable_grippers=True,
        enable_grasp_approach=True, grasp_contact_target_body="debug_cube",
        enable_object_state=True, enable_object_follow_observation=True,
        enable_object_motion=True, enable_object_support_motion_v2=True,
        enable_named_release_v3=True,
        object_support_body_names=("large_blue_cube", "medium_blue_cube"))
    try:
        session.application.start_observation_ready()
        hand = session.contract_bundle.gripper_contract
        assert hand.deployment_binding.skill_version == "3"
        assert hand.deployment_binding.release_support_object_ids == (
            "large_blue_cube", "medium_blue_cube")
        route = session.contract_bundle.route_manifest.route("manipulation.set_gripper_state", "3")
        with pytest.raises(KeyError):
            session.contract_bundle.route_manifest.route("manipulation.set_gripper_state", "2")
        validator = session.contract_bundle.payload_validators.validator_for(route.route)
        validator.validate_parameters({"component_id": "right_gripper", "target_state": "closed"})
        validator.validate_parameters({"component_id": "left_gripper", "target_state": "open"})
        named = {"component_id": "right_gripper", "target_state": "open",
                 "release_support_object_id": "large_blue_cube"}
        validator.validate_parameters(named)
        for invalid in (
            {**named, "component_id": "left_gripper"},
            {**named, "target_state": "closed"},
            {**named, "release_support_object_id": "unbound"},
        ):
            with pytest.raises(ValueError):
                validator.validate_parameters(invalid)
        port = session.gripper_ports["right_gripper"]
        assert port._release_observer is session.object_observer
        before = session.application.runtime.rig.gripper_execution_snapshot()
        assert not port.command_open_to_support(
            "large_blue_cube", cancel_requested=lambda: True).accepted
        assert session.application.runtime.rig.gripper_execution_snapshot() == before
        with pytest.raises(ValueError):
            port.command_open_to_support("large_blue_cube")
        assert session.application.runtime.rig.gripper_execution_snapshot() == before

        def require_order(_scene, _support):
            assert session.application._lock._is_owned()
            assert session.application.runtime.rig._lock._is_owned()
            raise ValueError("read-only precondition rejected")

        monkeypatch.setattr(session.object_observer,
                            "check_named_release_precondition", require_order)
        with pytest.raises(ValueError, match="read-only precondition rejected"):
            port.command_open_to_support("large_blue_cube")
        assert session.application.runtime.rig.gripper_execution_snapshot() == before

        observer = session.object_observer
        original_commit = observer.commit_named_release_support
        attempts = []

        def fail_first_commit(support):
            attempts.append(support)
            original_commit(support)
            if len(attempts) == 1:
                raise RuntimeError("observer commit failed")

        monkeypatch.setattr(observer, "check_named_release_precondition",
                            lambda scene, _support: (float(scene.data.time), 1))
        monkeypatch.setattr(observer, "commit_named_release_support", fail_first_commit)
        prior_support = observer.active_support_body_name
        prior_command = observer.command_state()
        with pytest.raises(RuntimeError, match="observer commit failed"):
            port.command_open_to_support("large_blue_cube")
        after = session.application.runtime.rig.gripper_execution_snapshot()
        assert after["pending_close_by_gripper"] == {}
        assert observer.active_support_body_name == prior_support
        assert observer.command_state() == prior_command
        assert attempts == ["large_blue_cube", prior_support]
    finally:
        assert session.close().accepted


@pytest.mark.integration
def test_ungrasped_object_request_is_rejected_with_original_domain_hold_and_cleanup():
    from test_agent_gateway_mujoco_tcp_pose_comparison import _request
    session = _session()
    try:
        app = session.application
        app.start_observation_ready()
        before = app.scene.data.qpos.copy()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        profile = session.contract_bundle.object_motion_contract.motion_profile
        request = _request(session, adapter, skill_id=GRASPED_OBJECT_MOVE_SKILL_ID, version="1",
            parameters=_parameters(profile), sequence=0)
        admitted = adapter.submit_skill_request(request.to_wire())
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=30.)
        operation = session.service.get_operation(admitted["operation"]["operation_id"])
        assert operation.execution_outcome == "failed", operation.to_wire()
        assert operation.reason_code == "object_motion_measured_relation_unavailable", operation.to_wire()
        assert operation.progress_summary["lower_hold_confirmed"] is True
        assert operation.execution_result is None and not app.target.planned_tcp_motion.active
        np.testing.assert_array_equal(app.scene.data.qpos, before)
    finally:
        assert session.close().accepted
