import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_agent_gateway_real_camera_only import _frame_binding
from test_agent_gateway_real_realsense_perception import _provider
from test_agent_gateway_service import _request

from agent_gateway.profiles.real.back_projection import (
    build_rgbd_geometry_registrations,
    project_retained_pixels,
)
from agent_gateway.profiles.real.perception_application import (
    assemble_camera_only_real_perception_composition,
)
from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.skills.perception.back_projection import (
    BACK_PROJECT_SKILL_ID,
    BACK_PROJECT_SKILL_VERSION,
    CAPTURE_RGBD_SKILL_ID,
    MAX_BACK_PROJECT_PIXELS,
    validate_back_project_parameters,
)
from agent_gateway.transport.client_bindings import (
    compile_agent_gateway_unix_client_bindings,
)


def geometry_composition(tmp_path, *, max_retained_operations=16, source_binding=None, object_localizer=None):
    provider, sources, _, _ = _provider()
    composition = assemble_camera_only_real_perception_composition(
        source_binding=provider.source_binding if source_binding is None else source_binding,
        socket_path=tmp_path / "geometry.sock",
        artifact_root=tmp_path, runtime_session_id="rgbd-geometry:test",
        control_domain_id="read-only-geometry:test", frame_binding=_frame_binding(),
        client_bindings=compile_agent_gateway_unix_client_bindings(primary_client_id="rpent:test", primary_client_uid=1000),
        enable_back_projection=True, max_retained_operations=max_retained_operations,
        object_localizer=object_localizer, clock=lambda: 10.)
    return composition, sources


@pytest.fixture
def geometry(tmp_path):
    composition, sources = geometry_composition(tmp_path)
    yield composition, sources
    composition.application.close()


def submit(composition, skill_id, parameters, sequence, *, input_artifact_bindings=(), skill_version=None):
    session = composition.session
    snapshot = session.observation_producer.capture_and_publish(session.service).snapshot
    manifest = composition.contract_bundle.route_manifest
    version = (BACK_PROJECT_SKILL_VERSION if skill_id == BACK_PROJECT_SKILL_ID else "1") if skill_version is None else skill_version
    route = next(entry.route for entry in manifest.skill_routes
                 if (entry.route.skill_id, entry.route.skill_version) == (skill_id, version))
    request = replace(_request(manifest, route, sequence=sequence, key=f"key:{sequence}",
        request_id=f"request:{sequence}", timeout_ms=min(5000, route.max_operation_duration_ms),
        snapshot_id=snapshot.snapshot_id, client_id="rpent:test"),
        expected_station_revision=snapshot.station_revision, parameters=parameters,
        input_artifact_bindings=input_artifact_bindings)
    facade = session.wire_adapter(authenticated_client_id="rpent:test")
    admitted = facade.submit_read_only_skill_request(request.to_wire())
    assert admitted["accepted"], admitted
    return admitted["operation"]["operation_id"]


def invoke(composition, skill_id, parameters, sequence, *, input_artifact_bindings=(), skill_version=None):
    operation_id = submit(composition, skill_id, parameters, sequence,
        input_artifact_bindings=input_artifact_bindings, skill_version=skill_version)
    session = composition.session
    for executor in session.rgbd_geometry_executors:
        assert executor.wait_all(timeout_sec=2.)
    if session.back_projection_executor is not None:
        assert session.back_projection_executor.wait_all(timeout_sec=2.)
    if session.localization_executor is not None:
        assert session.localization_executor.wait_all(timeout_sec=2.)
    operation = session.service.operation_store.get(operation_id)
    assert operation.terminal
    return operation


def capture(composition, sequence=1):
    operation = invoke(composition, CAPTURE_RGBD_SKILL_ID, {"selected_roles": ["main"]}, sequence)
    assert operation.status == "completed", operation.reason_code
    return operation


def test_model_free_geometry_discovery_is_closed_and_default_off(geometry):
    composition, sources = geometry
    assert composition.localization_provider is None
    assert composition.point_grasp_planner is None
    assert not composition.resources.started
    assert all(source.starts == 0 for source in sources.values())
    registrations = composition.contract_bundle.rgbd_geometry_registrations
    assert [item.skill.definition.skill_id for item in registrations] == [CAPTURE_RGBD_SKILL_ID, BACK_PROJECT_SKILL_ID]
    for item in registrations:
        route = item.skill.routes[0]
        assert route.control_effect_class == "read_only"
        assert route.may_move_robot is False
        assert route.requires_operator_confirmation is False
        assert route.skill_io_contract_hash == item.skill.io_contract.semantic_content_hash
        assert item.skill.schema_entry.parameter_schema["properties"][
            "selected_roles" if route.skill_id == CAPTURE_RGBD_SKILL_ID else "role"][
            "items" if route.skill_id == CAPTURE_RGBD_SKILL_ID else "enum"] == (
                {"type": "string", "enum": ("main",)} if route.skill_id == CAPTURE_RGBD_SKILL_ID else ("main",))
    from test_agent_gateway_real_camera_only import _composition
    default_root = composition.socket_path.parent / "default"
    default_root.mkdir()
    default = _composition(default_root)
    try:
        assert default.contract_bundle.rgbd_geometry_registrations == ()
    finally:
        default.application.close()


def test_live_v1_and_retained_v2_keep_distinct_closed_contracts(geometry):
    composition, _ = geometry
    entries = {entry.route.skill_version: entry for entry in composition.contract_bundle.route_manifest.skill_routes
               if entry.route.skill_id == BACK_PROJECT_SKILL_ID}
    assert set(entries) == {"1", "2"}
    schemas = composition.contract_bundle.schema_catalog
    for version, expected in (("1", {"role", "target_pixel_uv", "pixel_region_xyxy", "world_z_min_m", "world_z_max_m"}),
                              ("2", {"observation_operation_id", "role", "pixels_uv"})):
        entry = entries[version]
        schema = next(item for item in schemas.schema_entries
                      if (item.skill_id, item.skill_version) == (BACK_PROJECT_SKILL_ID, version))
        assert set(schema.parameter_schema["properties"]) == expected
        assert entry.route.parameter_schema_hash == schema.parameter_schema_hash
        assert not entry.route.may_move_robot and not entry.route.may_open_publication
        assert entry.route.control_effect_class == "read_only"
    assert entries["1"].route.semantic_content_hash != entries["2"].route.semantic_content_hash
    live = invoke(composition, BACK_PROJECT_SKILL_ID, {
        "role": "main", "target_pixel_uv": [2, 1], "pixel_region_xyxy": None,
        "world_z_min_m": None, "world_z_max_m": None}, 1, skill_version="1")
    assert live.status == "completed", live.reason_code
    assert "world_xyz_m" in live.execution_result and "points" not in live.execution_result
    observed = capture(composition, 2)
    retained = invoke(composition, BACK_PROJECT_SKILL_ID, {
        "observation_operation_id": observed.operation_id, "role": "main", "pixels_uv": [[2, 1]]},
        3, skill_version="2")
    assert retained.status == "completed", retained.reason_code
    assert "points" in retained.execution_result and "world_xyz_m" not in retained.execution_result


def test_back_projection_reuses_same_rgbd_and_returns_surface_not_pose(geometry):
    composition, sources = geometry
    observed = capture(composition)
    frame = observed.execution_result["frames"][0]
    assert frame["role"] == "main"
    reads_before = sources["front"].latest_frame_reads
    # A later camera frame is irrelevant: the request names the retained cut.
    sources["front"].frame = replace(sources["front"].frame,
        depth_image=np.full((3, 4), 900, np.uint16), source_timestamp=9.99)
    parameters = {"observation_operation_id": observed.operation_id, "role": "main", "pixels_uv": [[2, 1], [3, 2]]}
    operation = invoke(composition, BACK_PROJECT_SKILL_ID, parameters, 2)
    assert operation.status == "completed", operation.reason_code
    result = operation.execution_result
    assert result["point_semantics"] == "observed_surface_not_object_center_or_pose"
    assert result["acquired_at"] == frame["acquired_at"] == 9.95
    assert sources["front"].latest_frame_reads == reads_before
    assert result["points"][0]["depth_m"] == pytest.approx(.2)
    np.testing.assert_allclose(result["points"][0]["point_camera_m"], [0, -.2/6, .2], atol=1e-8)
    np.testing.assert_allclose(result["points"][0]["point_reference_m"], [.3, -.4+.2/6, .6], atol=1e-8)
    validator = composition.contract_bundle.rgbd_geometry_registrations[1].payload_validator
    validator.validate_execution_result(result)
    validator.validate_execution_result_for_parameters(parameters, result)


def test_public_capture_preview_localization_and_projection_use_the_same_observation(tmp_path):
    from test_agent_gateway_real_object_localization import _FakeLocalizer

    from deployments.perception.sam2_grounding_dino import GroundedObjectDetection

    class Localizer(_FakeLocalizer):
        def segment(self, *_args, **_kwargs):
            raise AssertionError("This test must only use the retained preview for localization")

        def encode_segment_artifacts(self, *_args, **_kwargs):
            raise AssertionError("This test must not create segmentation artifacts")

        def localize(self, image, *, language_prompt, max_detections):
            self.calls.append((np.array(image), language_prompt, max_detections))
            return (GroundedObjectDetection(box_xyxy=(1, 0, 3, 2), score=.9, label=language_prompt),)

    localizer = Localizer()
    composition, sources = geometry_composition(tmp_path, object_localizer=localizer)
    try:
        observed = capture(composition)
        preview = next(binding for binding in observed.output_artifact_bindings if binding.slot_id == "rgb_frames")
        reads = sources["front"].latest_frame_reads
        sources["front"].frame = replace(sources["front"].frame,
            image=np.zeros((3, 4, 3), np.uint8), depth_image=np.full((3, 4), 900, np.uint16))
        localized = invoke(composition, "perception.localize_objects", {
            "input_source": "artifact", "local_image_path": None,
            "language_prompt": "blue cube", "max_detections": 1}, 2,
            input_artifact_bindings=(AgentArtifactBinding(slot_id="input_image", ordinal=0, artifact=preview.artifact),))
        assert localized.status == "completed", localized.reason_code
        assert localized.execution_result["input_media_content_hash"] == observed.execution_result["frames"][0]["media_content_hash"]
        assert localizer.calls[0][0][0, 0, 0] > 250  # Retained red preview, not the new black frame.
        x0, y0, x1, y1 = localized.execution_result["detections"][0]["box_xyxy"]
        projected = invoke(composition, BACK_PROJECT_SKILL_ID, {
            "observation_operation_id": observed.operation_id, "role": "main",
            "pixels_uv": [[(x0+x1)//2, (y0+y1)//2]]}, 3)
        assert projected.status == "completed", projected.reason_code
        assert projected.execution_result["points"][0]["depth_m"] == pytest.approx(.2)
        assert projected.execution_result["acquired_at"] == observed.execution_result["frames"][0]["acquired_at"]
        assert sources["front"].latest_frame_reads == reads
        assert all(source.starts == 0 for source in sources.values())
    finally:
        composition.application.close()


@pytest.mark.parametrize("pixels,match", [([[0, 1]], "missing depth"), ([[4, 1]], "outside"), ([[1, 3]], "outside")])
def test_missing_depth_and_out_of_bounds_fail_without_filling_or_recapture(geometry, pixels, match):
    composition, sources = geometry
    observed = capture(composition)
    reads_before = sources["front"].latest_frame_reads
    executor = composition.session.rgbd_geometry_executors[1]
    with pytest.raises(ValueError, match=match):
        executor._read(SimpleNamespace(client_id="rpent:test", parameters={
            "observation_operation_id": observed.operation_id, "role": "main", "pixels_uv": pixels}),
            SimpleNamespace(operation_id="projection:invalid"))
    assert sources["front"].latest_frame_reads == reads_before


def test_source_client_session_and_retention_are_enforced(geometry, monkeypatch):
    composition, _ = geometry
    observed = capture(composition)
    executor = composition.session.rgbd_geometry_executors[1]
    request = SimpleNamespace(client_id="another-client", parameters={
        "observation_operation_id": observed.operation_id, "role": "main", "pixels_uv": [[2, 1]]})
    with pytest.raises(ValueError, match="this client"):
        executor._read(request, SimpleNamespace(operation_id="projection:foreign"))
    request.client_id = "rpent:test"
    capture_executor = composition.session.rgbd_geometry_executors[0]
    completion = capture_executor.completion_for(observed.operation_id)
    monkeypatch.setattr(capture_executor, "completion_for", lambda _: replace(
        completion, operation=replace(completion.operation, runtime_session_id="foreign:session")))
    with pytest.raises(ValueError, match="session"):
        executor._read(request, SimpleNamespace(operation_id="projection:foreign-session"))
    monkeypatch.setattr(capture_executor, "completion_for", lambda _: None)
    with pytest.raises(ValueError, match="no longer retained"):
        executor._read(request, SimpleNamespace(operation_id="projection:expired"))


def test_historical_projection_preserves_capture_revision_without_claiming_current_pose(geometry):
    composition, sources = geometry
    observed = capture(composition)
    sources["front"].info["generation"] += 1
    operation = invoke(composition, BACK_PROJECT_SKILL_ID, {
        "observation_operation_id": observed.operation_id, "role": "main", "pixels_uv": [[2, 1]]}, 2)
    assert operation.status == "completed", operation.reason_code
    assert operation.execution_result["station_revision"] == observed.execution_result["station_revision"]
    assert operation.current_station_revision > operation.execution_result["station_revision"]


@pytest.mark.parametrize("pixels", [[], [[True, 1]], [[1.5, 1]], [[-1, 1]], [[1]], [[1, 2, 3]], [[1, 1]]*(MAX_BACK_PROJECT_PIXELS+1)])
def test_projection_input_bounds_are_strict(pixels):
    with pytest.raises(ValueError):
        validate_back_project_parameters({"observation_operation_id": "observation:1", "role": "main", "pixels_uv": pixels}, roles=("main",))


@pytest.mark.parametrize("yaw", [0., .7, -1.2])
def test_projection_uses_captured_camera_rotation_not_current_wrist_pose(yaw):
    frame = _frame_binding()
    pose = np.eye(4)
    pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    pose[:3, 3] = [.12, -.3, .5]
    reference, _ = frame.frame_graph.transform_pose_matrix(pose,
        source_reference_frame_id=frame.source_frame_id, target_reference_frame_id=frame.reference_frame_id)
    metadata = {"camera_intrinsics": {"width": 4, "height": 3, "fx": 4., "fy": 3., "cx": 2., "cy": 1.5},
        "frame_binding_hash": frame.semantic_content_hash, "reference_frame_id": frame.reference_frame_id,
        "T_source_from_camera": pose, "T_reference_from_camera": reference}
    points = project_retained_pixels(arrays={"depth_m": np.full((3, 4), .3)}, metadata=metadata,
                                    pixels_uv=[[3, 1]], frame_binding=frame)
    expected = reference @ np.array([.075, -.05, .3, 1.])
    np.testing.assert_allclose(points[0]["point_reference_m"], expected[:3])
    metadata["T_reference_from_camera"] = np.eye(4)
    with pytest.raises(ValueError, match="disagrees"):
        project_retained_pixels(arrays={"depth_m": np.full((3, 4), .3)}, metadata=metadata,
                                pixels_uv=[[3, 1]], frame_binding=frame)


def test_geometry_registration_rejects_unframed_depth(geometry):
    composition, _ = geometry
    with pytest.raises(ValueError, match="typed public frame"):
        build_rgbd_geometry_registrations(replace(composition.registration, frame_binding=None,
                                                  back_projection=None,
                                                  depth_binding=composition.registration.rgb_binding))


def test_geometry_requires_room_for_both_source_and_projection(tmp_path):
    with pytest.raises(ValueError, match="max_retained_operations >= 2"):
        geometry_composition(tmp_path, max_retained_operations=1)


def test_retention_eviction_is_not_misreported_as_another_client(tmp_path):
    composition, sources = geometry_composition(tmp_path, max_retained_operations=2)
    try:
        observed = capture(composition)
        parameters = {"observation_operation_id": observed.operation_id, "role": "main", "pixels_uv": [[2, 1]]}
        assert invoke(composition, BACK_PROJECT_SKILL_ID, parameters, 2).status == "completed"
        capture(composition, 3)
        reads = sources["front"].latest_frame_reads
        expired = invoke(composition, BACK_PROJECT_SKILL_ID, parameters, 4)
        assert expired.status == "failed"
        assert "no longer retained" in expired.progress_summary["failure_diagnostic"]["message"]
        assert sources["front"].latest_frame_reads == reads
    finally:
        composition.application.close()


def test_geometry_roles_follow_aligned_catalog_not_plain_depth(tmp_path):
    provider, sources, _, _ = _provider(wrist_depth=True)
    sources["front"].config.align_depth_to_color = False
    binding = replace(provider.source_binding, world_from_depth_camera_by_source={
        **provider.source_binding.world_from_depth_camera_by_source, "left_wrist": np.eye(4)})
    composition, _ = geometry_composition(tmp_path, source_binding=binding)
    try:
        assert composition.registration.profile.depth_roles == ("main", "wrist_left")
        schema = composition.contract_bundle.rgbd_geometry_registrations[0].skill.schema_entry.parameter_schema
        assert schema["properties"]["selected_roles"]["items"]["enum"] == ("wrist_left",)
        operation = invoke(composition, CAPTURE_RGBD_SKILL_ID, {"selected_roles": ["wrist_left"]}, 1)
        assert operation.status == "completed", operation.reason_code
        assert operation.execution_result["frames"][0]["role"] == "wrist_left"
        assert all(source.starts == 0 for source in sources.values())
    finally:
        composition.application.close()


def test_no_aligned_depth_fails_before_device_start(tmp_path):
    provider, sources, _, _ = _provider()
    sources["front"].config.align_depth_to_color = False
    with pytest.raises(ValueError, match="RGB-D roles"):
        geometry_composition(tmp_path, source_binding=provider.source_binding)
    assert all(source.starts == source.closes == 0 for source in sources.values())


def test_public_capture_completion_can_be_projected_before_worker_returns(geometry, monkeypatch):
    composition, _ = geometry
    session = composition.session
    published = threading.Event()
    release = threading.Event()
    projection_returned = threading.Event()
    complete = session.service.complete_operation
    project_executor = session.rgbd_geometry_executors[1]
    project = project_executor._read

    def pause_after_publication(**kwargs):
        outcome = complete(**kwargs)
        if kwargs["reason_code"] == "passive_rgbd_observation_retained":
            published.set()
            assert release.wait(2.)
        return outcome

    def record_projection(*args):
        try:
            return project(*args)
        finally:
            projection_returned.set()

    monkeypatch.setattr(session.service, "complete_operation", pause_after_publication)
    monkeypatch.setattr(project_executor, "_read", record_projection)
    source_id = submit(composition, CAPTURE_RGBD_SKILL_ID, {"selected_roles": ["main"]}, 1)
    try:
        assert published.wait(1.)
        assert session.service.operation_store.get(source_id).status == "completed"
        projection_id = submit(composition, BACK_PROJECT_SKILL_ID, {
            "observation_operation_id": source_id, "role": "main", "pixels_uv": [[2, 1]]}, 2)
        assert not projection_returned.wait(.05)
    finally:
        release.set()
        assert all(executor.wait_all(timeout_sec=2.) for executor in session.rgbd_geometry_executors)
    assert session.service.operation_store.get(projection_id).status == "completed"


@pytest.mark.parametrize("phase", ["capture", "project"])
@pytest.mark.parametrize("control", ["cancel", "disconnect", "close"])
def test_geometry_read_control_waits_for_worker_before_closing_resources(geometry, monkeypatch, phase, control):
    composition, sources = geometry
    session = composition.session
    if phase == "project":
        observed = capture(composition)
        executor = session.rgbd_geometry_executors[1]
        skill_id = BACK_PROJECT_SKILL_ID
        parameters = {"observation_operation_id": observed.operation_id, "role": "main", "pixels_uv": [[2, 1]]}
        sequence = 2
    else:
        executor = session.rgbd_geometry_executors[0]
        skill_id = CAPTURE_RGBD_SKILL_ID
        parameters = {"selected_roles": ["main"]}
        sequence = 1
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    read = executor._read
    close_errors = []

    def blocked_read(*args):
        entered.set()
        assert release.wait(2.)
        return read(*args)

    def close():
        try:
            composition.application.close()
        except Exception as error:  # noqa: BLE001 - relay a close-thread failure to the test.
            close_errors.append(error)
        finally:
            closed.set()

    monkeypatch.setattr(executor, "_read", blocked_read)
    operation_id = submit(composition, skill_id, parameters, sequence)
    close_thread = None
    try:
        assert entered.wait(1.)
        if control == "close":
            close_thread = threading.Thread(target=close)
            close_thread.start()
            assert not closed.wait(.05)
            assert all(source.closes == 0 for source in sources.values())
        elif control == "cancel":
            assert session.service.cancel_operation(operation_id=operation_id, reason="test cancellation",
                current_station_revision=0, now=10.).accepted
        else:
            assert session.service.handle_client_disconnect(client_id="rpent:test", reason="test disconnect",
                current_station_revision=0, now=10.).accepted
    finally:
        release.set()
        assert executor.wait_all(timeout_sec=2.)
        if close_thread is not None:
            close_thread.join(2.)
            assert closed.is_set() and not close_errors
    operation = session.service.operation_store.get(operation_id)
    assert operation.terminal and operation.status != "completed"
    assert all(source.starts == 0 for source in sources.values())


def test_public_capture_validates_its_serialized_view_only_once(geometry, monkeypatch):
    import dataset_contracts.grasp_rgbd_snapshot as single
    import dataset_contracts.grasp_multiview_snapshot as multi

    decode = single.decode_grasp_rgbd_snapshot
    calls = []
    def counted(data):
        calls.append(data)
        return decode(data)
    monkeypatch.setattr(single, "decode_grasp_rgbd_snapshot", counted)
    monkeypatch.setattr(multi, "decode_grasp_rgbd_snapshot", counted)
    composition, _ = geometry
    operation = capture(composition)
    assert operation.execution_outcome == "completed"
    assert len(operation.execution_result["frames"]) == len(calls) == 1
