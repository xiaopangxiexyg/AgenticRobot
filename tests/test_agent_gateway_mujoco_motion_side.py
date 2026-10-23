"""Pure side-binding tests: fake scene/ports, no simulator or dispatch proof."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.mujoco import motion_application, motion_skill, observation
from agent_gateway.profiles.mujoco.motion_application import (
    MujocoAgentSkillMotionApplication,
)
from agent_gateway.profiles.mujoco.observation import MujocoAgentObservationProducer
from agent_gateway.profiles.mujoco.observation_binding import (
    MujocoAgentObservationSourceBinding,
)
from agent_gateway.profiles.mujoco.state_deployment import (
    MujocoPassiveTcpPoseStateDeploymentBinding,
)
from agent_gateway.runtime.station_operation_worker import (
    StationLifecycleActivationEvidence,
    StationLifecycleControlEvidence,
)
from agent_gateway.skills.state.tcp_pose_comparison import (
    validate_tcp_pose_comparison_result_v2,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.deploy.schema import xyzquat_to_pose_matrix
from robot_motion.runtime.control_admission import ControlCut
from sim.mujoco.deploy.agent_skill_target import MujocoAgentSkillMotionTarget
from sim.mujoco.deploy.component_action_profile import (
    build_mujoco_pose_component_target_profile,
)
from sim.mujoco.deploy.planned_tcp_motion import (
    MujocoPlannedTcpMotion,
    MujocoTcpPlanRejected,
)


def _forbidden(*_args, **_kwargs):
    raise AssertionError("This pure fixture cannot start, step, render or dispatch.")


@pytest.fixture(autouse=True)
def _no_scene_construction(monkeypatch):
    from sim.mujoco import scene

    monkeypatch.setattr(scene.MujocoScene, "__init__", _forbidden)
    monkeypatch.setattr(scene.MujocoCameraRenderer, "__init__", _forbidden)


def _schema(side, *, declared_side=None, components=None, controlled_frame="tool_tcp"):
    component = side + "_arm"
    arms = {component: side if declared_side is None else declared_side}
    if components is not None:
        arms = {value: value.removesuffix("_arm") for value in components}
    return build_mujoco_pose_component_target_profile(
        SimpleNamespace(arm_specs={"left": object(), "right": object()}),
        profile_id="fixture.motion-side.v1",
        arm_side_by_component=arms,
        arm_controlled_frame_by_component={key: controlled_frame for key in arms},
        selection_group_by_component={key: "arms" for key in arms},
    ).action_schema


def _fixture(monkeypatch, side="right", *, explicit=True, station_from_base=None):
    """Construct the real application around passive, in-memory fake ports."""
    component = side + "_arm"
    state = {"time": 10.0, "finite": True, "captures": 0, "cuts": 0}
    events = []
    poses = {
        "left": xyzquat_to_pose_matrix((0.2, 0.3, 0.4, 0, 0, 0, 1)),
        "right": xyzquat_to_pose_matrix((0.6, -0.3, 0.5, 0, 0, 0, 1)),
    }
    fake_scene = SimpleNamespace(state_is_finite=lambda: state["finite"])

    def capture_scene(projector):
        state["captures"] += 1
        return projector(fake_scene)

    def capture_control_cut(*, cutoff_at):
        state["cuts"] += 1
        return ControlCut("runtime:side-fixture", "domain:side-fixture",
                          "clock:side-fixture", 0, cutoff_at, 7)

    plan = SimpleNamespace(
        enabled=True, measurement_mode="full_frames", component_ids=(component,),
        semantic_content_hash=f"plan:{component}",
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            runtime_session_id="runtime:side-fixture",
            control_domain_id="domain:side-fixture",
            clock_domain_id="clock:side-fixture",
            initial_control_mode="agent_skill",
            direct_component_proposal_source=SimpleNamespace(producer_kind="agent_skill"),
            execution_observation_plan=plan,
        ),
        action_schema=_schema(side),
        execution_observation_binding=SimpleNamespace(
            binding_id="observation:side-fixture", plan_hash=plan.semantic_content_hash,
        ),
        execution_observation_capability=SimpleNamespace(
            representation_by_component={component: "cartesian_pose"},
            schema_hash_by_component={component: f"pose-schema:{component}"},
        ),
        authoritative_port=SimpleNamespace(capture_receiver_state=lambda: SimpleNamespace(
            current_owner_generation=2, current_authority_epoch=3, current_execution_epoch=4,
        )),
        rig=SimpleNamespace(capture_scene=capture_scene),
        capture_control_cut=capture_control_cut,
        clock=lambda: state["time"], start=_forbidden, step=_forbidden,
        submit_component_action=_forbidden,
        close=lambda: events.append("runtime.close"),
    )
    station = CoordinateFrameSpec(
        frame_id="station", semantic_parameters={"axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1},
    )
    matrix = np.eye(4)
    matrix[0, 3] = 1.0 if side == "left" else 2.0
    if station_from_base is not None:
        matrix = np.asarray(station_from_base, dtype=float).copy()
    graph = CoordinateFrameGraph(
        graph_id=f"frames:{side}",
        frames=(CoordinateFrameSpec(frame_id=f"{side}_arm_base"), station),
        transforms=(RigidFrameTransformSpec(
            transform_id=f"base-to-station:{side}", source_frame_id=f"{side}_arm_base",
            target_frame_id="station", matrix_target_source=matrix,
        ),),
    )
    target = MujocoAgentSkillMotionTarget(
        task_config=SimpleNamespace(right_gripper_name="right_gripper"),
        scene=fake_scene, runtime=runtime,
        target_frame_profile=SimpleNamespace(
            frame_graph=graph, canonical_reference_frame_id="station",
            reference_frame_by_component={component: f"{side}_arm_base"},
        ),
        tool_frame_profile=SimpleNamespace(
            semantic_content_hash=f"tools:{side}", binding_by_component={
                component: SimpleNamespace(controlled_frame_id="tool_tcp"),
            },
        ),
        task_initialization_info={}, motion_side=side,
    )
    for module in (motion_application, observation):
        monkeypatch.setattr(module, "capture_mujoco_sim_hil_tool_tcp_pose_by_side",
                            lambda _scene, _config: poses)
    for name in ("MujocoRgbSnapshotSource", "MujocoDepthSnapshotSource"):
        monkeypatch.setattr(motion_application, name, lambda **_kwargs: SimpleNamespace(
            close=lambda: events.append("snapshot.close"),
        ))
    kwargs = {"motion_side": side} if explicit else {}
    app = MujocoAgentSkillMotionApplication(
        target=target, calibration_hash=f"calibration:{side}", **kwargs,
    )
    binding = MujocoPassiveTcpPoseStateDeploymentBinding(
        deployment_binding_id="deployment:state-fixture", deployment_binding_version="1",
        route_hash="route:state-fixture", runtime_session_id=runtime.config.runtime_session_id,
        control_domain_id=runtime.config.control_domain_id, target_binding_id="target:side-fixture",
        station_manifest_hash="station:side-fixture", execution_manifest_hash="execution:side-fixture",
        observation_binding_id=runtime.execution_observation_binding.binding_id,
        observation_binding_hash="observation-hash:side-fixture",
        observation_plan_hash=plan.semantic_content_hash,
        observation_capability_hash=f"capability:{component}", pose_schema_hash=f"pose-schema:{component}",
        reference_frame_id="station", reference_frame_spec_hash=station.semantic_content_hash,
        frame_graph_hash=graph.semantic_content_hash, tool_config_hash=f"tools:{side}",
        calibration_hash=app.calibration_hash, executor_id="reader:side-fixture",
        explicit_exclusions=("no_motion",), component_id=component,
    )
    bundle = SimpleNamespace(
        deployment_binding=binding, state_deployment_binding=binding,
        route_manifest=SimpleNamespace(semantic_content_hash="routes:side-fixture"),
    )
    source = MujocoAgentObservationSourceBinding(
        runtime_session_id=runtime.config.runtime_session_id,
        control_domain_id=runtime.config.control_domain_id,
        source_binding_id="source:side-fixture", source_binding_hash="source-hash:side-fixture",
        producer_epoch=1, source_kind="agent_skill",
    )
    return SimpleNamespace(app=app, target=target, runtime=runtime, bundle=bundle,
                           binding=binding, source=source, state=state, events=events, poses=poses)


def _producer(fixture):
    return MujocoAgentObservationProducer(
        application=fixture.app, contract_bundle=fixture.bundle, source_binding=fixture.source,
    )


def _parameters(side):
    return {"component_ids": [side + "_arm"], "reference_frame_id": "station",
            "controlled_frame_id": "tool_tcp", "max_measurement_age_ms": 50}


@pytest.mark.parametrize("side,explicit", [("right", False), ("right", True), ("left", True)])
def test_application_fixed_side_and_readonly_properties(monkeypatch, side, explicit):
    fixture = _fixture(monkeypatch, side, explicit=explicit)
    app = fixture.app
    assert (app.motion_side, app.component_id) == (side, side + "_arm")
    for name, value in (("motion_side", "left"), ("component_id", "left_arm")):
        with pytest.raises(AttributeError):
            setattr(app, name, value)
    measured = app.capture_tcp_pose_by_component()
    assert set(measured) == {app.component_id}
    assert measured[app.component_id][:3] == tuple(fixture.poses[side][:3, 3])
    assert (fixture.state["captures"], fixture.state["cuts"], fixture.events) == (1, 0, [])
    assert not app.started and not app.held and not app.closed


@pytest.mark.parametrize("side", [None, True, 0, "", "both", "LEFT", " left", "left "])
def test_invalid_side_cannot_construct_application(monkeypatch, side):
    fixture = _fixture(monkeypatch)
    with pytest.raises(ValueError, match="motion_side"):
        MujocoAgentSkillMotionApplication(target=fixture.target, calibration_hash="cal", motion_side=side)


@pytest.mark.parametrize("mismatch", ["target_side", "schema_side", "two_components", "other_component", "tool_frame", "mode", "producer"])
def test_application_rejects_inconsistent_runtime_before_capture(monkeypatch, mismatch):
    fixture = _fixture(monkeypatch, "left")
    runtime, target = fixture.runtime, fixture.target
    if mismatch == "target_side":
        target = replace(target, motion_side="right")
    elif mismatch == "schema_side":
        runtime.action_schema = _schema("left", declared_side="right")
    elif mismatch == "two_components":
        runtime.action_schema = _schema("left", components=("left_arm", "right_arm"))
    elif mismatch == "other_component":
        runtime.action_schema = _schema("right")
    elif mismatch == "tool_frame":
        runtime.action_schema = _schema("left", controlled_frame="flange")
    elif mismatch == "mode":
        runtime.config.initial_control_mode = "policy"
    else:
        runtime.config.direct_component_proposal_source.producer_kind = "policy"
    with pytest.raises(ValueError, match="Motion application"):
        MujocoAgentSkillMotionApplication(target=target, calibration_hash="cal", motion_side="left")
    assert fixture.state["captures"] == 0 and fixture.events == []


def _planned_adapter(fixture, side):
    # The real adapter constructor only binds ports/locks; no Scene, FK,
    # planning, model allocation or runtime start is involved here.
    fixture.runtime.rig.scene = fixture.target.scene
    return MujocoPlannedTcpMotion(
        rig=fixture.runtime.rig,
        task_config=fixture.target.task_config,
        pose_projector=SimpleNamespace(project=_forbidden),
        reference_hz=15.0,
        motion_side=side,
    )


@pytest.mark.parametrize("side,explicit", [("right", False), ("right", True), ("left", True)])
def test_application_accepts_matching_actual_planned_adapter(monkeypatch, side, explicit):
    fixture = _fixture(monkeypatch, side)
    planner = _planned_adapter(fixture, side)
    target = replace(fixture.target, planned_tcp_motion=planner)
    kwargs = {"motion_side": side} if explicit else {}
    app = MujocoAgentSkillMotionApplication(target=target, calibration_hash="cal", **kwargs)
    assert app.target.planned_tcp_motion is planner
    assert (planner.motion_side, planner.component_id) == (app.motion_side, app.component_id)
    assert not app.started and not app.held and not app.closed
    assert fixture.state["captures"] == fixture.state["cuts"] == 0
    assert fixture.events == [] and not planner.active


@pytest.mark.parametrize("side", ["right", "left"])
def test_application_rejects_opposite_actual_planned_adapter_before_resources(monkeypatch, side):
    fixture = _fixture(monkeypatch, side)
    planner = _planned_adapter(fixture, "right" if side == "left" else "left")
    target = replace(fixture.target, planned_tcp_motion=planner)
    for name in ("MujocoRgbSnapshotSource", "MujocoDepthSnapshotSource"):
        monkeypatch.setattr(motion_application, name, _forbidden)
    with pytest.raises(ValueError, match="planned TCP adapter"):
        MujocoAgentSkillMotionApplication(target=target, calibration_hash="cal", motion_side=side)
    assert fixture.state["captures"] == fixture.state["cuts"] == 0
    assert fixture.events == [] and not planner.active


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("identity", ["missing_both", "missing_side", "missing_component", "wrong_side", "wrong_component"])
def test_application_nonnull_planner_needs_explicit_matching_identity(monkeypatch, side, identity):
    fixture = _fixture(monkeypatch, side)
    other = "right" if side == "left" else "left"
    attributes = {"motion_side": side, "component_id": side + "_arm"}
    if identity == "missing_both":
        attributes.clear()
    elif identity == "missing_side":
        attributes.pop("motion_side")
    elif identity == "missing_component":
        attributes.pop("component_id")
    elif identity == "wrong_side":
        attributes["motion_side"] = other
    else:
        attributes["component_id"] = other + "_arm"
    target = replace(fixture.target, planned_tcp_motion=SimpleNamespace(**attributes))
    for name in ("MujocoRgbSnapshotSource", "MujocoDepthSnapshotSource"):
        monkeypatch.setattr(motion_application, name, _forbidden)
    with pytest.raises(ValueError, match="planned TCP adapter"):
        MujocoAgentSkillMotionApplication(target=target, calibration_hash="cal", motion_side=side)
    assert fixture.state["captures"] == fixture.state["cuts"] == 0 and fixture.events == []


@pytest.mark.parametrize("side", ["right", "left"])
def test_state_uses_one_passive_cut_selected_frame_tool_and_stream(monkeypatch, side):
    fixture = _fixture(monkeypatch, side)
    producer = _producer(fixture)
    capture = producer._capture(active_operation_id="read:fixture")
    cut, measured = capture.observation_cut, capture.measurement
    component = side + "_arm"
    assert cut is not None and measured is not None
    assert set(cut.selected_sample_by_stream) == {f"robot_state.{component}.tool_tcp_pose"}
    assert cut.observation_contract_hash == canonical_content_hash(
        {"contract_id": "dexteleop.agent_state.mujoco_tcp_pose_observation.v1",
         "stream_id": f"robot_state.{component}.tool_tcp_pose", "payload": "xyz+quat_xyzw",
         "measurement_owner": "mujoco_scene_ground_truth"},
        domain="dexteleop.agent_state.observation_contract.v1",
    )
    assert cut.policy_io_layout_hash == canonical_content_hash(
        {"layout_id": "dexteleop.agent_state.tcp_pose_read.v1",
         "input_streams": (f"robot_state.{component}.tool_tcp_pose",),
         "output": "measured_tcp_pose_projection"}, domain="dexteleop.agent_state.io_layout.v1",
    )
    assert set(measured.value_by_component) == {component}
    assert measured.value_by_component[component][0] == pytest.approx(1.2 if side == "left" else 2.6)
    assert measured.source_measurement_hash == capture.snapshot.measurement_summary["measurement_hash"]
    assert measured.measured_at == capture.snapshot.captured_at == cut.cutoff_at == 10.0
    assert measured.observation_plan_hash == fixture.binding.observation_plan_hash
    assert capture.snapshot.tool_config_hash == f"tools:{side}"
    assert capture.snapshot.calibration_hash == f"calibration:{side}"
    assert (fixture.state["captures"], fixture.state["cuts"], fixture.events) == (1, 1, [])
    read = producer.capture_tcp_pose_state(active_operation_id="read:next", parameters=_parameters(side))
    result = read.execution_result
    assert result["component_ids"] == [component]
    assert result["pose_by_component"] == {component: list(measured.value_by_component[component])}
    assert result["frame_graph_hash"] == fixture.binding.frame_graph_hash
    assert result["pose_schema_hash_by_component"] == {component: f"pose-schema:{component}"}
    assert (fixture.state["captures"], fixture.state["cuts"], fixture.events) == (2, 2, [])
    # V2's null-target path is the unchanged V1 measurement, not new sampling.
    validate_tcp_pose_comparison_result_v2(
        {**_parameters(side), "target_operation_id": None},
        {**result, "target_comparison": None}, allowed_component_ids=(component,),
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_station_full_pose_rotation_translation_roundtrip(monkeypatch, side):
    station_from_base = np.array(
        [
            [0.0, 0.0, 1.0, 0.13],
            [1.0, 0.0, 0.0, -0.17],
            [0.0, 1.0, 0.0, 0.29],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    rotation = station_from_base[:3, :3]
    translation = station_from_base[:3, 3]
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    fixture = _fixture(monkeypatch, side, station_from_base=station_from_base)
    quarter_turn = np.sqrt(0.5)
    measured_in_base = xyzquat_to_pose_matrix(
        (0.21, -0.13, 0.37, quarter_turn, 0.0, 0.0, quarter_turn)
    )
    fixture.poses[side] = measured_in_base
    component = side + "_arm"
    execution_frame = side + "_arm_base"

    # The real producer and application transform must compose in Station,
    # including the noncommuting orientation, without mirroring either side.
    read = _producer(fixture).capture_tcp_pose_state(
        active_operation_id="read:rotated-station", parameters=_parameters(side)
    )
    public_pose = read.execution_result["pose_by_component"][component]
    np.testing.assert_allclose(
        xyzquat_to_pose_matrix(public_pose),
        station_from_base @ measured_in_base,
        rtol=0.0,
        atol=1e-6,
    )
    assert read.execution_result["reference_frame_id"] == "station"
    assert read.execution_result["component_ids"] == [component]
    measured_roundtrip = fixture.app.transform_tcp_pose_xyzquat(
        public_pose, source_frame_id="station", target_frame_id=execution_frame
    )
    np.testing.assert_allclose(
        xyzquat_to_pose_matrix(measured_roundtrip),
        measured_in_base,
        rtol=0.0,
        atol=1e-6,
    )

    # Absolute targets travel in the opposite direction at execution. Check
    # the inverse explicitly, not merely a round-trip that could cancel errors.
    public_target = (0.41, 0.08, 0.62, 0.0, quarter_turn, 0.0, quarter_turn)
    target_in_station = xyzquat_to_pose_matrix(public_target)
    expected_target_in_base = np.eye(4)
    expected_target_in_base[:3, :3] = rotation.T @ target_in_station[:3, :3]
    expected_target_in_base[:3, 3] = rotation.T @ (
        target_in_station[:3, 3] - translation
    )
    execution_target = fixture.app.transform_tcp_pose_xyzquat(
        public_target, source_frame_id="station", target_frame_id=execution_frame
    )
    np.testing.assert_allclose(
        xyzquat_to_pose_matrix(execution_target),
        expected_target_in_base,
        rtol=0.0,
        atol=1e-6,
    )
    target_roundtrip = fixture.app.transform_tcp_pose_xyzquat(
        execution_target, source_frame_id=execution_frame, target_frame_id="station"
    )
    np.testing.assert_allclose(
        xyzquat_to_pose_matrix(target_roundtrip),
        target_in_station,
        rtol=0.0,
        atol=1e-6,
    )
    assert (fixture.state["captures"], fixture.state["cuts"], fixture.events) == (1, 1, [])
    assert not fixture.app.started and not fixture.app.held and not fixture.app.closed


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("requested", [["left_arm", "right_arm"], ["other"], []])
def test_state_rejects_nonselected_or_both_components(monkeypatch, side, requested):
    producer = _producer(_fixture(monkeypatch, side))
    with pytest.raises(ValueError, match="exposes only"):
        producer.capture_tcp_pose_state(
            active_operation_id="read:fixture", parameters={**_parameters(side), "component_ids": requested},
        )


@pytest.mark.parametrize("side", ["right", "left"])
def test_state_rejects_other_arm_request(monkeypatch, side):
    producer = _producer(_fixture(monkeypatch, side))
    with pytest.raises(ValueError, match="exposes only"):
        producer.capture_tcp_pose_state(active_operation_id="read:fixture", parameters=_parameters("left" if side == "right" else "right"))


@pytest.mark.parametrize("mismatch", ["app_side", "state_binding", "motion_binding", "plan", "schema", "frame", "tool", "source"])
def test_producer_rejects_cross_side_binding_before_any_cut(monkeypatch, mismatch):
    fixture = _fixture(monkeypatch, "left")
    if mismatch == "app_side":
        fixture.app._motion_side = "right"
    elif mismatch == "state_binding":
        fixture.bundle.state_deployment_binding = replace(fixture.binding, component_id="right_arm")
    elif mismatch == "motion_binding":
        fixture.bundle.deployment_binding = replace(fixture.binding, component_id="right_arm")
    elif mismatch == "plan":
        fixture.runtime.config.execution_observation_plan.component_ids = ("right_arm",)
    elif mismatch == "schema":
        fixture.runtime.action_schema = _schema("right")
    elif mismatch == "frame":
        fixture.app.target_frame_profile.reference_frame_by_component = {"right_arm": "left_arm_base"}
    elif mismatch == "tool":
        fixture.app.tool_frame_profile.binding_by_component = {"right_arm": SimpleNamespace(controlled_frame_id="tool_tcp")}
    else:
        fixture.source = replace(fixture.source, control_domain_id="domain:wrong")
    with pytest.raises(ValueError):
        _producer(fixture)
    assert fixture.state["captures"] == fixture.state["cuts"] == 0


def test_left_rejects_unreviewed_history_and_gripper(monkeypatch):
    fixture = _fixture(monkeypatch, "left")
    fixture.bundle.history_source_binding = object()
    with pytest.raises(ValueError, match="history.*right"):
        _producer(fixture)
    with pytest.raises(ValueError, match="gripper.*right-only"):
        fixture.app.capture_arm_gripper_state()
    assert fixture.state["captures"] == 0


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("invalid", ["nonfinite", "missing_cut", "wrong_clock", "stale", "bad_quaternion", "clock_regression"])
def test_selected_reader_keeps_validity_freshness_and_clock_gates(monkeypatch, side, invalid):
    fixture = _fixture(monkeypatch, side)
    producer = _producer(fixture)
    original = fixture.app.capture_passive_application_cut

    def capture(*, scene_projector):
        cut = original(scene_projector=scene_projector)
        if invalid == "missing_cut":
            return replace(cut, control_cut=None)
        if invalid == "wrong_clock":
            return replace(cut, control_cut=replace(cut.control_cut, clock_domain_id="clock:wrong"))
        if invalid == "stale":
            return replace(cut, captured_at=cut.captured_at - 1.0)
        if invalid == "bad_quaternion":
            return replace(cut, scene_projection={"finite_state": True,
                "pose_xyz_quat_xyzw_by_component": {side + "_arm": [0, 0, 0, 0, 0, 0, 0]}})
        return cut

    fixture.app.capture_passive_application_cut = capture
    if invalid == "nonfinite":
        fixture.state["finite"] = False
    if invalid == "clock_regression":
        times = iter((10.0, 10.0, 9.0))
        fixture.runtime.clock = lambda: next(times)
        producer.clock = fixture.runtime.clock
    with pytest.raises((ValueError, RuntimeError)):
        producer.capture_tcp_pose_state(active_operation_id="read:invalid", parameters=_parameters(side))
    assert fixture.events == []


def test_default_state_binding_preserves_old_semantic_preimage_and_ref_shape(monkeypatch):
    fixture = _fixture(monkeypatch)
    values = asdict(fixture.binding)
    values.pop("component_id")
    default = MujocoPassiveTcpPoseStateDeploymentBinding(**values)
    legacy_payload = {
        **values, "controlled_component_ids": ("right_arm",), "controlled_frame_id": "tool_tcp",
        "target_kind": "sim", "read_semantics": "application_locked_passive_cut",
    }
    expected = canonical_content_hash(legacy_payload, domain="dexteleop.agent_state.mujoco_tcp_pose_deployment_binding.v1")
    assert default.component_id == "right_arm"
    assert default.semantic_content_hash == fixture.binding.semantic_content_hash == expected
    assert default.to_ref().to_wire(include_reference_hash=False) == {
        "deployment_binding_id": default.deployment_binding_id,
        "deployment_binding_hash": expected,
        "target_binding_id": default.target_binding_id,
        "target_kind": "sim", "execution_manifest_hash": default.execution_manifest_hash,
        "cross_target_deployment_review": None,
    }
    left = replace(default, component_id="left_arm")
    assert left.semantic_content_hash == canonical_content_hash(
        {**legacy_payload, "controlled_component_ids": ("left_arm",)},
        domain="dexteleop.agent_state.mujoco_tcp_pose_deployment_binding.v1",
    )
    assert left.semantic_content_hash != expected


@pytest.mark.parametrize("component", [None, True, [], "", "left", "both", " left_arm", "left_arm "])
def test_state_binding_component_id_is_exact(monkeypatch, component):
    fixture = _fixture(monkeypatch)
    with pytest.raises(ValueError, match="component_id"):
        replace(fixture.binding, component_id=component)


def test_default_observation_stream_contract_hashes_are_unchanged():
    assert observation.MUJOCO_AGENT_TCP_POSE_OBSERVATION_CONTRACT_HASH_V1 == canonical_content_hash(
        {"contract_id": "dexteleop.agent_state.mujoco_tcp_pose_observation.v1",
         "stream_id": "robot_state.right_arm.tool_tcp_pose", "payload": "xyz+quat_xyzw",
         "measurement_owner": "mujoco_scene_ground_truth"},
        domain="dexteleop.agent_state.observation_contract.v1",
    )
    assert observation.MUJOCO_AGENT_TCP_POSE_IO_LAYOUT_HASH_V1 == canonical_content_hash(
        {"layout_id": "dexteleop.agent_state.tcp_pose_read.v1",
         "input_streams": ("robot_state.right_arm.tool_tcp_pose",),
         "output": "measured_tcp_pose_projection"}, domain="dexteleop.agent_state.io_layout.v1",
    )


def test_default_right_full_state_wire_matches_frozen_79b6(monkeypatch):
    producer = _producer(_fixture(monkeypatch, explicit=False))
    # Captured by running the original 79b6 producer against this same pure
    # fixture. These include the snapshot, state result and evidence refs.
    expected_hashes = (
        "0e6a1243f2532356b27032e80169fbe5af9fc6df826d706d83792e7c8af8d258",
        "ac6cbd6943e7eb3c346d20fff135e97d5bb68882b6f2dfdcb21015f31d4a17b5",
    )
    for expected in expected_hashes:
        read = producer.capture_tcp_pose_state(
            active_operation_id="read:fixture", parameters=_parameters("right"),
        )
        encoded = json.dumps({
            "snapshot": read.snapshot.to_wire(), "result": read.execution_result,
            "evidence_refs": read.evidence_refs,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        assert hashlib.sha256(encoded).hexdigest() == expected


def _safe_fixture(monkeypatch, side, *, reject_at=None, basis_change=False, planning_cancel=False):
    """Exercise the real safe adapter; lower activation/search/HOLD are fake ports."""
    events, projects = [], []
    measured = (0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0)
    target = (0.21, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0)
    activation = StationLifecycleActivationEvidence(True, "ready", "lower:fixture", ("activation:fixture",))

    def parent_activate(self, *, request, operation, cancel_requested):
        events.append("lower.activate")
        if cancel_requested():
            return StationLifecycleActivationEvidence(False, "cancelled", "lower:fixture", ("hold:fixture",), True)
        self._activation_measured_pose = self._capture_pose()
        return activation

    monkeypatch.setattr(motion_skill.MujocoAgentTcpPoseSkillLifecycle, "activate", parent_activate)

    def project(poses):
        projects.append(poses)
        modified = (side,) if len(projects) == reject_at else ()
        return SimpleNamespace(
            envelope_profile_hash="envelope:fixture",
            anchor_hash="anchor:changed" if basis_change and len(projects) > 1 else "anchor:fixture",
            modified_sides=modified, to_payload=lambda: {"modified_sides": modified},
        )

    plan = SimpleNamespace(reference_poses=(measured, target), trajectory_hash="trajectory:fixture")

    def search(_target, **kwargs):
        events.append("planner.plan")
        if planning_cancel:
            raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
        check = kwargs["configuration_constraint"](np.zeros(1), {})
        if not check.safe:
            raise MujocoTcpPlanRejected(check.reason)
        return plan

    lifecycle = object.__new__(motion_skill.MujocoAgentSafeTcpPoseSkillLifecycle)
    lifecycle.request = SimpleNamespace(parameters={})
    lifecycle.component_id = side + "_arm"
    lifecycle.application = SimpleNamespace(tcp_pose_envelope_runtime=SimpleNamespace(project=project))
    lifecycle.motion_profile = SimpleNamespace(max_translation_delta_m=0.1, max_rotation_delta_rad=0.2,
        translation_tolerance_m=0.005, rotation_tolerance_rad=0.02, stable_measurement_count=2, max_control_cycles=20)
    lifecycle.target_pose = target
    lifecycle._validate_call = lambda **_kwargs: None
    lifecycle._capture_pose = lambda: measured
    lifecycle._lower_operation_id = "lower:fixture"
    lifecycle._execution_refs = set()
    lifecycle._path_search_time_budget_sec = 3.0
    lifecycle._contact_binding = None
    lifecycle._planner = SimpleNamespace(
        pose_for_configuration=lambda _qpos, _pair: target, plan=search,
        activate=lambda _plan: events.append("planner.activate"),
    )

    def hold(*, reason):
        events.append(("lower.hold", reason))
        return StationLifecycleControlEvidence(True, "hold", "lower:fixture", ("hold:fixture",), True)

    lifecycle._hold = hold
    return lifecycle, events, projects


@pytest.mark.parametrize("side", ["right", "left"])
def test_safe_initial_constraint_and_reference_projections_use_only_selected_side(monkeypatch, side):
    lifecycle, events, projects = _safe_fixture(monkeypatch, side)
    evidence = lifecycle.activate(request=object(), operation=object(), cancel_requested=lambda: False)
    assert evidence.succeeded
    assert [set(value) for value in projects] == [{side}] * 4
    assert events == ["lower.activate", "planner.plan", "planner.activate"]
    assert lifecycle._execution_refs == {"mujoco-safe-joint-trajectory:trajectory:fixture"}


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("rejection", ["constraint", "reference", "basis", "cancel"])
def test_safe_projection_planning_failure_preserves_hold_and_no_activation(monkeypatch, side, rejection):
    lifecycle, events, projects = _safe_fixture(
        monkeypatch, side, reject_at={"constraint": 2, "reference": 3}.get(rejection),
        basis_change=rejection == "basis", planning_cancel=rejection == "cancel",
    )
    evidence = lifecycle.activate(request=object(), operation=object(), cancel_requested=lambda: False)
    assert not evidence.succeeded and evidence.lower_hold_confirmed
    assert evidence.evidence_refs == ("hold:fixture",)
    assert "planner.activate" not in events and lifecycle._execution_refs == set()
    assert events[-1][0] == "lower.hold"
    assert all(set(value) == {side} for value in projects)


@pytest.mark.parametrize("side", ["right", "left"])
def test_safe_precancel_keeps_original_lower_gate_without_search(monkeypatch, side):
    lifecycle, events, projects = _safe_fixture(monkeypatch, side)
    evidence = lifecycle.activate(request=object(), operation=object(), cancel_requested=lambda: True)
    assert not evidence.succeeded and evidence.lower_hold_confirmed
    assert events == ["lower.activate"] and projects == []


@pytest.mark.parametrize("side", ["right", "left"])
def test_application_close_still_requires_hold_and_is_idempotent(monkeypatch, side):
    fixture = _fixture(monkeypatch, side)
    app = fixture.app
    app._started = True
    app._active_operation_id = "operation:fixture"
    with pytest.raises(RuntimeError, match="requires HOLD"):
        app.close_after_hold()
    assert not fixture.events and not app.closed

    def transition(status, **kwargs):
        fixture.events.append((status, kwargs["force_cancellation"], kwargs["source_event_ids"]))
        return SimpleNamespace(lower_event_ids=("lower-hold:fixture",))

    fixture.runtime.transition_execution_gate = transition
    app.request_hold(reason="test-cancel", operation_id="operation:fixture")
    assert app.held
    app.close_after_hold()
    app.close_after_hold()
    assert app.closed and app.active_operation_id is None
    assert fixture.events == [("hold", True, ("operation:fixture",)), "runtime.close", "snapshot.close", "snapshot.close"]
    with pytest.raises(RuntimeError, match="closed"):
        app.capture_tcp_pose_by_component()


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("with_receipt", [False, True])
def test_envelope_deactivation_still_requires_lower_hold_receipt(monkeypatch, side, with_receipt):
    fixture = _fixture(monkeypatch, side)
    app = fixture.app
    app._started = True
    app._active_operation_id = "operation:fixture"
    refs = ("lower-hold:fixture",) if with_receipt else ()
    fixture.runtime.transition_execution_gate = lambda *_args, **_kwargs: SimpleNamespace(lower_event_ids=refs)
    app.tcp_pose_envelope_runtime = SimpleNamespace(
        deactivate_after_hold=lambda **kwargs: fixture.events.append(kwargs),
    )
    app.target = replace(fixture.target, planned_tcp_motion=SimpleNamespace(
        clear=lambda: fixture.events.append("planner.clear"),
    ))
    if not with_receipt:
        with pytest.raises(motion_application.TCPPoseEnvelopeLifecycleError, match="deactivation evidence"):
            app.request_hold(reason="test-cancel", operation_id="operation:fixture")
        assert not app.held and fixture.events == []
        with pytest.raises(RuntimeError, match="requires HOLD"):
            app.close_after_hold()
    else:
        app.request_hold(reason="test-cancel", operation_id="operation:fixture")
        assert app.held
        assert fixture.events == [{
            "operation_id": "operation:fixture", "next_execution_epoch": 4,
            "hold_evidence_ids": refs,
        }, "planner.clear"]


def test_tcp_producer_rejects_missing_motion_component_without_a_legacy_fallback(monkeypatch):
    fixture = _fixture(monkeypatch, "right")
    fixture.bundle.deployment_binding = SimpleNamespace()
    with pytest.raises(ValueError, match="one bound arm component"):
        _producer(fixture)
    assert fixture.state["captures"] == fixture.state["cuts"] == 0
