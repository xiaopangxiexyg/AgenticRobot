"""RGB v3 contract/executor checks without cameras, rendering, models or devices."""

import hashlib
import io
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agent_gateway.profiles.back_projection import (
    BackProjectionProfile,
    CorrelatedPlacementCapture,
    LiveAlignedDepthCapture,
    LiveBackProjectionProvider,
    build_back_projection_registration,
)
from agent_gateway.profiles.mujoco.rgb_deployment import (
    build_mujoco_rgb_capture_registration,
    build_mujoco_rgb_capture_registration_v2,
)
from agent_gateway.profiles.rgbd_capture import (
    build_rgbd_capture_executor,
    build_rgbd_capture_registration,
)
from agent_gateway.profiles.segmentation import LiveRgbSegmentationCapture
from agent_gateway.profiles.spatial_goal import SpatialGoalRegistration
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.perception_artifacts import PAIRED_RGBD_SCHEMA_ID_V1

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _registrations(*, deployment_identity=None, catalog=None, **depth_changes):
    identity = {
        "runtime_session_id": "session:test",
        "control_domain_id": "domain:test",
        "target_binding_id": "target:test",
        "station_manifest_hash": "station:test",
        "execution_manifest_hash": "execution:test",
    }
    if deployment_identity is not None:
        identity.update(deployment_identity)
    if catalog is None:
        catalog = tuple(
            {"source_id": source, "role": role, "native_width": 4, "native_height": 3}
            for source, role in (("front", "main"), ("wrist", "wrist_right"))
        )
    rgb = build_mujoco_rgb_capture_registration_v2(source_catalog=catalog, **identity)
    depth = build_back_projection_registration(
        **{
            **identity,
            "target_kind": "sim",
            "current_mode": "sim_active",
            "source_catalog": catalog[:1],
            "reference_frame_id": "canonical_station",
            "reference_frame_spec_hash": "frame:station:test",
            "frame_graph_hash": "graph:test",
            **depth_changes,
        }
    )
    return rgb, depth


_DEPLOYMENT_IDENTITIES = (
    {"runtime_session_id": "session:second"},
    {"station_manifest_hash": "station:second"},
    {"execution_manifest_hash": "execution:second"},
    {
        "runtime_session_id": "session:second",
        "station_manifest_hash": "station:second",
        "execution_manifest_hash": "execution:second",
    },
)


@pytest.mark.parametrize("deployment_identity", _DEPLOYMENT_IDENTITIES)
def test_v3_configuration_identity_is_stable_across_deployments(deployment_identity):
    rgb, depth = _registrations()
    other_rgb, other_depth = _registrations(deployment_identity=deployment_identity)
    assert rgb.profile == other_rgb.profile
    assert depth.profile == other_depth.profile
    assert rgb.registration.routes == other_rgb.registration.routes
    assert depth.skill.routes == other_depth.skill.routes
    original = build_rgbd_capture_registration(
        rgb_registration=rgb, back_projection=depth
    )
    other = build_rgbd_capture_registration(
        rgb_registration=other_rgb, back_projection=other_depth
    )
    assert (
        original.entry.route.allowed_parameter_profiles
        == other.entry.route.allowed_parameter_profiles
    )
    assert original.skill.schema_entry == other.skill.schema_entry
    assert original.skill.io_contract == other.skill.io_contract
    assert (
        original.entry.route.semantic_content_hash
        == other.entry.route.semantic_content_hash
    )
    assert (
        original.entry.deployment_binding.deployment_binding_hash
        != other.entry.deployment_binding.deployment_binding_hash
    )
    assert original.entry.availability_hash != other.entry.availability_hash


@pytest.mark.parametrize("deployment_identity", _DEPLOYMENT_IDENTITIES)
def test_v3_stable_route_still_rejects_old_deployment_provider(
    case, deployment_identity
):
    rgb, depth = _registrations(deployment_identity=deployment_identity)
    registration = build_rgbd_capture_registration(
        rgb_registration=rgb, back_projection=depth
    )
    assert (
        registration.entry.route.semantic_content_hash
        == case.registration.entry.route.semantic_content_hash
    )
    with pytest.raises(ValueError, match="exact back-projection binding"):
        build_rgbd_capture_executor(**{**case.arguments, "registration": registration})
    assert not case.rgb_provider.calls and not case.port.calls


def test_v3_stable_route_still_rejects_old_session_request(case):
    rgb, depth = _registrations(deployment_identity=_DEPLOYMENT_IDENTITIES[-1])
    registration = build_rgbd_capture_registration(
        rgb_registration=rgb, back_projection=depth
    )
    provider = LiveBackProjectionProvider(capture_port=case.port, registration=depth)
    executor = build_rgbd_capture_executor(
        **{
            **case.arguments,
            "registration": registration,
            "back_projection_provider": provider,
        }
    )
    with pytest.raises(ValueError, match="another runtime session"):
        executor._read(case.request, case.operation)
    assert not case.rgb_provider.calls and not case.port.calls


def test_v3_native_grid_configuration_still_changes_route_and_profile():
    rgb, depth = _registrations()
    other_rgb, other_depth = _registrations(
        catalog=tuple(
            {"source_id": source, "role": role, "native_width": 8, "native_height": 6}
            for source, role in (("front", "main"), ("wrist", "wrist_right"))
        )
    )
    original = build_rgbd_capture_registration(
        rgb_registration=rgb, back_projection=depth
    )
    other = build_rgbd_capture_registration(
        rgb_registration=other_rgb, back_projection=other_depth
    )
    assert (
        original.entry.route.allowed_parameter_profiles
        != other.entry.route.allowed_parameter_profiles
    )
    assert (
        original.entry.route.semantic_content_hash
        != other.entry.route.semantic_content_hash
    )


def _parameters(**changes):
    return {
        "selected_roles": ["main"],
        "width": 4,
        "height": 3,
        "max_total_bytes": 16 * 1024 * 1024,
        "include_depth": False,
        **changes,
    }


def _jpeg(color=73):
    stream = io.BytesIO()
    Image.new("RGB", (4, 3), (color, 0, 0)).save(stream, format="JPEG")
    return stream.getvalue()


class _RgbProvider:
    def __init__(self):
        self.calls = []
        self.on_capture = lambda: None

    def capture(self, parameters, *, expected_station_revision):
        assert "include_depth" not in parameters
        self.calls.append((dict(parameters), expected_station_revision))
        self.on_capture()
        frames = tuple(
            (
                "front" if role == "main" else "wrist",
                role,
                SimpleNamespace(
                    data=_jpeg(73 + index),
                    media_type="image/jpeg",
                    width=4,
                    height=3,
                    acquired_at=9.5 + index * 0.1,
                ),
            )
            for index, role in enumerate(parameters["selected_roles"])
        )
        return SimpleNamespace(
            frames=frames,
            captured_at=9.7,
            station_revision=7,
            capture_window_started_at=9.4,
            capture_window_ended_at=9.8,
        )


@pytest.fixture
def case(tmp_path):
    rgb, depth = _registrations()
    registration = build_rgbd_capture_registration(
        rgb_registration=rgb, back_projection=depth
    )
    store = RecoverableAgentArtifactStore(
        root=tmp_path / "artifacts",
        quota_bytes=64 * 1024 * 1024,
        max_artifact_bytes=16 * 1024 * 1024,
    )
    port = SimpleNamespace(calls=[])

    def capture(**kwargs):
        port.calls.append(kwargs)
        raise AssertionError("RGB-only branch must not capture aligned depth")

    port.capture_placement_rgbd = capture
    port.capture_aligned_depth = lambda **kwargs: pytest.fail(
        "No separate depth capture is permitted"
    )
    provider = LiveBackProjectionProvider(capture_port=port, registration=depth)
    rgb_provider = _RgbProvider()
    state = SimpleNamespace(now=10.0, revision=7)
    arguments = {
        "registration": registration,
        "rgb_provider": rgb_provider,
        "back_projection_provider": provider,
        "store": store,
        "current_station_revision": lambda: state.revision,
        "clock": lambda: state.now,
        "clock_domain_id": "test.monotonic",
        "max_retained_operations": 3,
    }
    executor = build_rgbd_capture_executor(**arguments)
    request = SimpleNamespace(
        parameters=_parameters(),
        expected_station_revision=7,
        runtime_session_id="session:test",
        semantic_content_hash="a" * 64,
    )
    operation = SimpleNamespace(operation_id="operation:test", deadline_at=13.0)
    executor._cancel_events[operation.operation_id] = threading.Event()
    yield SimpleNamespace(
        rgb=rgb,
        depth=depth,
        registration=registration,
        store=store,
        port=port,
        provider=provider,
        rgb_provider=rgb_provider,
        state=state,
        arguments=arguments,
        executor=executor,
        request=request,
        operation=operation,
    )
    store.close()


def _read(case, **parameters):
    case.request.parameters = {**case.request.parameters, **parameters}
    return case.executor._read(case.request, case.operation)


def test_missing_cancellation_owner_rejects_before_capture(case):
    case.executor._cancel_events.pop(case.operation.operation_id)
    with pytest.raises(RuntimeError, match="cancelled or expired"):
        _read(case)
    assert not case.rgb_provider.calls and not case.port.calls


def test_direct_left_factory_keeps_the_explicit_v3_feature_fence(tmp_path, monkeypatch):
    from agent_gateway.profiles.mujoco import motion_factory as factory
    from sim.mujoco.deploy import MujocoSimHILTargetConfig

    monkeypatch.setattr(factory, "prepare_mujoco_sim_hil_target",
                        lambda *a, **kw: pytest.fail("Unsupported side reached Scene preparation"))
    config = MujocoSimHILTargetConfig(
        task_config_path=tmp_path / "never-opened.json", realtime_worker=False,
        exposed_pose_sides=("left",), allow_scripted_task_reset=False)
    with pytest.raises(ValueError, match="Left motion mode"):
        factory.build_mujoco_agent_tcp_pose_gateway_session(
            target_config=config, motion_side="left", allowed_client_ids=("test:left",),
            artifact_storage=factory.MujocoAgentArtifactStorageConfig(root=tmp_path),
            enable_rgbd_capture_v3=True)


def test_v3_registration_is_neutral_and_opt_in_without_mutating_rgb_identities():
    rgb, depth = _registrations()
    assert type(depth.profile) is BackProjectionProfile
    original_depth_hash = depth.profile.semantic_content_hash
    original = (
        rgb.registration,
        rgb.profile,
        rgb.deployment_binding,
        rgb.payload_validator,
        rgb.availability_hash,
    )
    v3 = build_rgbd_capture_registration(rgb_registration=rgb, back_projection=depth)
    assert isinstance(v3, SpatialGoalRegistration)
    assert v3.skill.definition.skill_id == "perception.capture_rgb"
    assert v3.skill.definition.skill_version == "3"
    assert depth.profile.semantic_content_hash == original_depth_hash
    assert original == (
        rgb.registration,
        rgb.profile,
        rgb.deployment_binding,
        rgb.payload_validator,
        rgb.availability_hash,
    )
    rebuilt = _registrations()[0]
    assert rebuilt.registration == rgb.registration
    assert rebuilt.profile == rgb.profile
    assert rebuilt.deployment_binding == rgb.deployment_binding
    assert rebuilt.availability_hash == rgb.availability_hash
    slots = {slot.slot_id: slot for slot in v3.skill.io_contract.output_artifact_slots}
    assert set(slots) == {"rgb_frames", "depth_frames", "paired_observations"}
    assert (slots["rgb_frames"].min_count, slots["rgb_frames"].max_count) == (1, 2)
    for name in ("depth_frames", "paired_observations"):
        assert (slots[name].min_count, slots[name].max_count) == (0, 1)
    assert slots["paired_observations"].allowed_schema_ids == (
        PAIRED_RGBD_SCHEMA_ID_V1,
    )
    assert slots["paired_observations"].max_artifact_bytes == 65536
    assert not v3.skill.io_contract.input_artifact_slots
    route = v3.entry.route
    assert (
        route.control_effect_class,
        route.privilege_tier,
        route.cancellation_policy,
        route.may_move_robot,
        route.may_open_publication,
    ) == ("read_only", "t0_read_only", "cancel_only", False, False)
    assert (
        route.semantic_content_hash != rgb.registration.routes[0].semantic_content_hash
    )
    v3.payload_validator.validate_route(route)


@pytest.mark.parametrize(
    "changes",
    [
        {"include_depth": 1},
        {"include_depth": None},
        {"selected_roles": ["missing"]},
        {"include_depth": True, "selected_roles": ["main", "wrist_right"]},
        {"include_depth": True, "selected_roles": ["wrist_right"]},
        {"include_depth": True, "width": 3},
        {"include_depth": True, "height": 2},
        {"width": True},
        {"max_total_bytes": True},
        {"extra": 1},
    ],
)
def test_invalid_parameters_fail_before_capture_or_commit(case, changes):
    with pytest.raises((ValueError, TypeError)):
        _read(case, **changes)
    assert not case.rgb_provider.calls and not case.port.calls


def test_include_depth_is_required_and_schema_retains_exact_v2_properties(case):
    from agent_gateway.skills.perception.definition import (
        capture_rgb_skill_definition_v2,
    )
    from agent_gateway.skills.perception.rgbd_schemas import (
        capture_rgb_skill_definition_v3,
    )

    parameters = dict(case.request.parameters)
    del parameters["include_depth"]
    with pytest.raises(ValueError, match="fields"):
        case.registration.payload_validator.validate_parameters(parameters)
    old = case.rgb.registration.schema_entry.parameter_schema
    new = case.registration.skill.schema_entry.parameter_schema
    assert set(new["properties"]) == set(old["properties"]) | {"include_depth"}
    assert set(new["required"]) == set(old["required"]) | {"include_depth"}
    assert all(
        new["properties"][key] == value for key, value in old["properties"].items()
    )
    description = new["properties"]["include_depth"]["description"]
    for role, (width, height) in case.depth.profile.dimensions_by_role.items():
        assert f"{role}: width={width}, height={height}" in description
    assert "wrist_right:" not in description
    assert capture_rgb_skill_definition_v2().skill_version == "2"
    assert capture_rgb_skill_definition_v3().skill_version == "3"


@pytest.mark.parametrize(
    "changes",
    [
        {"runtime_session_id": "other"},
        {"station_manifest_hash": "other"},
        {"control_domain_id": "other"},
        {"target_binding_id": "other"},
        {"execution_manifest_hash": "other"},
        {
            "source_catalog": (
                {
                    "source_id": "other",
                    "role": "main",
                    "native_width": 4,
                    "native_height": 3,
                },
            )
        },
        {
            "source_catalog": (
                {
                    "source_id": "front",
                    "role": "main",
                    "native_width": 8,
                    "native_height": 3,
                },
            )
        },
        {
            "source_catalog": (
                {
                    "source_id": "other",
                    "role": "other",
                    "native_width": 4,
                    "native_height": 3,
                },
            )
        },
    ],
)
def test_registration_rejects_inconsistent_deployment_source_or_grid(changes):
    rgb, depth = _registrations(**changes)
    with pytest.raises(ValueError):
        build_rgbd_capture_registration(rgb_registration=rgb, back_projection=depth)


def test_registration_rejects_legacy_rgb_v1():
    rgb, depth = _registrations()
    legacy = build_mujoco_rgb_capture_registration(
        source_catalog=tuple(
            {
                "source_id": source,
                "role": role,
                "native_width": width,
                "native_height": height,
            }
            for source, role, width, height in rgb.profile.sources
        ),
        **{
            key: getattr(rgb.deployment_binding, key)
            for key in (
                "runtime_session_id",
                "control_domain_id",
                "target_binding_id",
                "station_manifest_hash",
                "execution_manifest_hash",
            )
        },
    )
    with pytest.raises(ValueError, match="RGB v2"):
        build_rgbd_capture_registration(rgb_registration=legacy, back_projection=depth)


def test_rgb_only_uses_legacy_capture_order_actual_grid_and_no_depth(case):
    read = _read(case, selected_roles=["wrist_right", "main"])
    result = read.execution_result
    assert len(case.rgb_provider.calls) == 1 and not case.port.calls
    assert case.rgb_provider.calls[0] == (
        {
            "selected_roles": ["wrist_right", "main"],
            "width": 4,
            "height": 3,
            "max_total_bytes": 16 * 1024 * 1024,
        },
        7,
    )
    assert result["include_depth"] is False and result["paired_observations"] == []
    assert (
        result["capture_window_started_at"] == 9.4
        and result["capture_window_ended_at"] == 9.8
    )
    assert [frame["role"] for frame in result["frames"]] == ["wrist_right", "main"]
    assert [frame["acquired_at"] for frame in result["frames"]] == [9.5, 9.6]
    assert {(frame["width"], frame["height"]) for frame in result["frames"]} == {(4, 3)}
    assert result["total_bytes"] == sum(
        item.artifact.byte_length for item in read.output_artifact_bindings
    )
    for index, binding in enumerate(read.output_artifact_bindings):
        assert (binding.slot_id, binding.ordinal) == ("rgb_frames", index)
        assert case.store.managed_read_path(
            binding.artifact.artifact_id
        ).read_bytes() == _jpeg(73 + index)
    case.registration.payload_validator.validate_execution_result(result)


@pytest.mark.parametrize("phase", ["before", "after_capture"])
@pytest.mark.parametrize("failure", ["cancel", "timeout", "revision"])
def test_rgb_only_checks_cancel_timeout_and_revision_before_any_commit(
    case, monkeypatch, phase, failure
):
    from agent_gateway.profiles import rgbd_capture

    monkeypatch.setattr(
        rgbd_capture,
        "commit_artifact_bytes",
        lambda **kwargs: pytest.fail("No commit expected"),
    )

    def interrupt():
        if failure == "cancel":
            monkeypatch.setattr(
                case.executor, "cancel_requested", lambda operation: True
            )
        elif failure == "timeout":
            case.state.now = 13.0
        else:
            case.state.revision = 8

    if phase == "before":
        interrupt()
    else:
        case.rgb_provider.on_capture = interrupt
    with pytest.raises(RuntimeError, match="cancelled or expired|Station revision"):
        _read(case)
    assert len(case.rgb_provider.calls) == (phase == "after_capture")


def test_total_rgb_byte_bound_is_checked_before_first_commit(case, monkeypatch):
    from agent_gateway.profiles import rgbd_capture

    monkeypatch.setattr(
        rgbd_capture,
        "commit_artifact_bytes",
        lambda **kwargs: pytest.fail("No commit expected"),
    )
    with pytest.raises(ValueError, match="byte bound"):
        _read(case, max_total_bytes=1)


@pytest.mark.parametrize(
    "field", ["reference_frame_id", "reference_frame_spec_hash", "frame_graph_hash"]
)
def test_executor_rejects_changed_depth_binding_before_capture(case, field):
    _, other = _registrations(**{field: "other:" + field})
    provider = LiveBackProjectionProvider(capture_port=case.port, registration=other)
    with pytest.raises(ValueError, match="exact back-projection"):
        build_rgbd_capture_executor(
            **{**case.arguments, "back_projection_provider": provider}
        )
    assert not case.rgb_provider.calls and not case.port.calls


def test_executor_rejects_another_session_before_capture(case):
    case.request.runtime_session_id = "other"
    with pytest.raises(ValueError, match="runtime session"):
        _read(case)
    assert not case.rgb_provider.calls and not case.port.calls


@pytest.mark.parametrize(
    "field", ["reference_frame_id", "reference_frame_spec_hash", "frame_graph_hash"]
)
def test_v3_binds_actual_projection_frame_authority(field):
    rgb, depth = _registrations()
    _, other = _registrations(**{field: "other:" + field})
    assert type(depth.profile) is type(other.profile) is BackProjectionProfile
    original = build_rgbd_capture_registration(
        rgb_registration=rgb, back_projection=depth
    )
    changed = build_rgbd_capture_registration(
        rgb_registration=rgb, back_projection=other
    )
    assert (
        original.entry.route.semantic_content_hash
        != changed.entry.route.semantic_content_hash
    )
    assert original.entry.availability_hash != changed.entry.availability_hash


def _paired_cut():
    rgb_jpeg = _jpeg()
    depth = np.full((3, 4), 0.5, dtype=np.float32)
    stream = io.BytesIO()
    np.save(stream, depth, allow_pickle=False)
    depth_npy = stream.getvalue()
    with Image.open(io.BytesIO(rgb_jpeg)) as image:
        rgb = LiveRgbSegmentationCapture(
            source_id="front",
            role="main",
            captured_at=9.5,
            station_revision=7,
            image=image,
            rgb_content_hash=hashlib.sha256(rgb_jpeg).hexdigest(),
        )
    content_hash = canonical_content_hash(
        {
            "dtype": "float32",
            "unit": "meter",
            "invalid_value": 0.0,
            "shape": list(depth.shape),
            "byte_sha256": hashlib.sha256(depth.tobytes(order="C")).hexdigest(),
        },
        domain="dexteleop.mujoco_back_projection.depth_content.v1",
    )
    aligned = LiveAlignedDepthCapture(
        source_id="front",
        role="main",
        captured_at=9.5,
        station_revision=7,
        capture_group_id="cut:test",
        depth_content_hash=content_hash,
        intrinsics={
            "width": 4,
            "height": 3,
            "fx": 2.0,
            "fy": 2.0,
            "cx": 2.0,
            "cy": 1.0,
        },
        reference_from_camera=np.eye(4),
        depth_m=depth,
    )
    return CorrelatedPlacementCapture(rgb, aligned, rgb_jpeg, depth_npy)


def _use_cut(case, cut, *, after_capture=lambda: None):
    def capture(**kwargs):
        case.port.calls.append(kwargs)
        after_capture()
        return cut

    case.port.capture_placement_rgbd = capture
    case.request.parameters["include_depth"] = True


def test_rgbd_preserves_original_media_full_refs_and_exact_profile_frame_identifiers(
    case, monkeypatch
):
    cut = _paired_cut()
    _use_cut(case, cut)
    original_pixels = cut.rgb.image.tobytes()

    def forbidden(*args, **kwargs):
        pytest.fail("RGBD capture must not resize or re-encode the original media")

    monkeypatch.setattr(Image.Image, "save", forbidden)
    monkeypatch.setattr(Image.Image, "resize", forbidden)
    monkeypatch.setattr(np, "save", forbidden)
    read = _read(case)
    assert not case.rgb_provider.calls
    assert case.port.calls == [{"role": "main", "expected_station_revision": 7}]
    assert cut.rgb.image.tobytes() == original_pixels
    bindings = {binding.slot_id: binding for binding in read.output_artifact_bindings}
    assert set(bindings) == {"rgb_frames", "depth_frames", "paired_observations"}
    assert all(item.ordinal == 0 for item in bindings.values())
    media = {
        name: case.store.managed_read_path(binding.artifact.artifact_id).read_bytes()
        for name, binding in bindings.items()
    }
    assert (
        media["rgb_frames"] == cut.rgb_jpeg and media["depth_frames"] == cut.depth_npy
    )
    assert len(media["paired_observations"]) <= 65536
    metadata = json.loads(media["paired_observations"])
    profile = case.depth.profile
    assert type(profile) is BackProjectionProfile
    assert set(metadata) == {
        "schema_id",
        "reference_frame_id",
        "reference_frame_spec_hash",
        "frame_graph_hash",
        "clock_domain_id",
        "source_operation_id",
        "source_request_hash",
        "runtime_session_id",
        "width",
        "height",
        "role",
        "source_id",
        "captured_at",
        "station_revision",
        "capture_group_id",
        "rgb_artifact",
        "depth_artifact",
        "intrinsics",
        "reference_from_camera",
        "depth_content_hash",
    }
    assert metadata["reference_frame_id"] == profile.reference_frame_id
    assert metadata["reference_frame_spec_hash"] == profile.reference_frame_spec_hash
    assert metadata["frame_graph_hash"] == profile.frame_graph_hash
    assert metadata["rgb_artifact"] == bindings["rgb_frames"].artifact.to_wire()
    assert metadata["depth_artifact"] == bindings["depth_frames"].artifact.to_wire()
    assert metadata["source_operation_id"] == case.operation.operation_id
    assert metadata["source_request_hash"] == case.request.semantic_content_hash
    assert metadata["runtime_session_id"] == case.request.runtime_session_id
    assert metadata["clock_domain_id"] == "test.monotonic"
    assert metadata["station_revision"] == 7
    assert metadata["width"] == 4 and metadata["height"] == 3
    assert metadata["captured_at"] == 9.5
    assert metadata["role"] == "main" and metadata["source_id"] == "front"
    assert metadata["reference_from_camera"] == cut.depth.reference_from_camera.tolist()
    result = read.execution_result
    assert result["include_depth"] is True
    assert (
        result["capture_window_started_at"] == result["capture_window_ended_at"] == 9.5
    )
    assert result["total_bytes"] == sum(map(len, media.values()))
    assert result["frames"] == [
        {
            "ordinal": 0,
            "source_id": "front",
            "role": "main",
            "acquired_at": 9.5,
            "media_type": "image/jpeg",
            "width": 4,
            "height": 3,
            "media_content_hash": bindings["rgb_frames"].artifact.sha256,
        }
    ]
    assert result["paired_observations"] == [
        {
            "role": "main",
            "source_id": "front",
            "rgb_frame_ordinal": 0,
            "depth_artifact_ordinal": 0,
            "observation_artifact_ordinal": 0,
            "observation_sha256": hashlib.sha256(
                media["paired_observations"]
            ).hexdigest(),
        }
    ]


def test_rgbd_combined_byte_bound_includes_depth_and_observation_before_commit(
    case, monkeypatch
):
    from agent_gateway.profiles import rgbd_capture

    cut = _paired_cut()
    _use_cut(case, cut)
    monkeypatch.setattr(
        rgbd_capture,
        "commit_artifact_bytes",
        lambda **kwargs: pytest.fail("No commit expected"),
    )
    with pytest.raises(ValueError, match="byte bound"):
        _read(case, max_total_bytes=len(cut.rgb_jpeg) + len(cut.depth_npy))
    assert len(case.port.calls) == 1 and not case.rgb_provider.calls


@pytest.mark.parametrize(
    "fault",
    [
        "rgb_missing",
        "depth_missing",
        "rgb_hash",
        "depth_dtype",
        "rgb_pixels",
        "rgb_time",
        "revision",
        "role",
        "grid",
        "exif",
    ],
)
def test_rgbd_rejects_wrong_or_missing_original_cut_before_commit(
    case, monkeypatch, fault
):
    from agent_gateway.profiles import rgbd_capture

    cut = _paired_cut()
    if fault == "rgb_missing":
        cut = replace(cut, rgb_jpeg=None)
    elif fault == "depth_missing":
        cut = replace(cut, depth_npy=None)
    elif fault == "rgb_hash":
        cut = replace(cut, rgb=replace(cut.rgb, rgb_content_hash="b" * 64))
    elif fault == "depth_dtype":
        stream = io.BytesIO()
        np.save(stream, cut.depth.depth_m.astype(np.float64), allow_pickle=False)
        cut = replace(cut, depth_npy=stream.getvalue())
    elif fault == "rgb_pixels":
        cut = replace(cut, rgb=replace(cut.rgb, image=Image.new("RGB", (4, 3))))
    elif fault == "rgb_time":
        cut = replace(cut, rgb=replace(cut.rgb, captured_at=9.6))
    elif fault in {"revision", "role"}:
        change = {"station_revision": 8} if fault == "revision" else {"role": "other"}
        cut = replace(
            cut, rgb=replace(cut.rgb, **change), depth=replace(cut.depth, **change)
        )
    elif fault == "grid":
        cut = SimpleNamespace(
            rgb=replace(cut.rgb, image=Image.new("RGB", (8, 3))), depth=cut.depth
        )
    elif fault == "exif":
        stream = io.BytesIO()
        exif = Image.Exif()
        exif[274] = 6
        cut.rgb.image.save(stream, format="JPEG", exif=exif)
        data = stream.getvalue()
        cut = replace(
            cut,
            rgb_jpeg=data,
            rgb=replace(cut.rgb, rgb_content_hash=hashlib.sha256(data).hexdigest()),
        )
    _use_cut(case, cut)
    monkeypatch.setattr(
        rgbd_capture,
        "commit_artifact_bytes",
        lambda **kwargs: pytest.fail("No commit expected"),
    )
    with pytest.raises((ValueError, RuntimeError)):
        _read(case)
    assert len(case.port.calls) == 1 and not case.rgb_provider.calls


@pytest.mark.parametrize("failure", ["cancel", "timeout", "revision"])
@pytest.mark.parametrize(
    "phase", ["after_capture", "after_encode", "after_first_commit"]
)
def test_rgbd_cancel_timeout_revision_checks_preserve_no_success(
    case, monkeypatch, failure, phase
):
    from agent_gateway.profiles import paired_rgbd, rgbd_capture

    def interrupt():
        if failure == "cancel":
            monkeypatch.setattr(
                case.executor, "cancel_requested", lambda operation: True
            )
        elif failure == "timeout":
            case.state.now = 13.0
        else:
            case.state.revision = 8

    _use_cut(
        case,
        _paired_cut(),
        after_capture=interrupt if phase == "after_capture" else lambda: None,
    )
    original_encode = paired_rgbd.encode_paired_observation
    original_commit = rgbd_capture.commit_artifact_bytes
    commits = []

    def encode(*args, **kwargs):
        data = original_encode(*args, **kwargs)
        if phase == "after_encode":
            interrupt()
        return data

    def commit(**kwargs):
        result = original_commit(**kwargs)
        commits.append(result)
        if phase == "after_first_commit":
            interrupt()
        return result

    monkeypatch.setattr(paired_rgbd, "encode_paired_observation", encode)
    monkeypatch.setattr(rgbd_capture, "commit_artifact_bytes", commit)
    with pytest.raises(RuntimeError, match="cancelled or expired|Station revision"):
        _read(case)
    assert len(commits) == (phase == "after_first_commit")


@pytest.mark.parametrize(
    "field,value",
    [
        ("rgb_frame_ordinal", True),
        ("depth_artifact_ordinal", 1),
        ("observation_artifact_ordinal", False),
        ("observation_sha256", "wrong"),
        ("source_id", "other"),
        ("role", "wrist_right"),
    ],
)
def test_paired_result_rejects_false_ordinal_hash_and_role_claims(case, field, value):
    _use_cut(case, _paired_cut())
    result = _read(case).execution_result
    pair = {**result["paired_observations"][0], field: value}
    with pytest.raises(ValueError):
        case.registration.payload_validator.validate_execution_result(
            {**result, "paired_observations": [pair]}
        )


def test_rgb_only_result_must_not_claim_pair_or_change_include_depth(case):
    result = _read(case).execution_result
    with pytest.raises(ValueError, match="must not advertise"):
        case.registration.payload_validator.validate_execution_result(
            {**result, "paired_observations": [{}]}
        )
    with pytest.raises(ValueError, match="differs from the request"):
        case.registration.payload_validator.validate_execution_result_for_parameters(
            case.request.parameters, {**result, "include_depth": True}
        )


@pytest.mark.parametrize(
    "boundary,field,value",
    [
        ("parameters", "include_depth", 1),
        ("result", "include_depth", None),
        ("result", "paired_observations", None),
    ],
)
def test_v3_rejects_wrong_json_member_types_explicitly(case, boundary, field, value):
    if boundary == "parameters":
        payload = {**case.request.parameters, field: value}
        validate = case.registration.payload_validator.validate_parameters
    else:
        payload = {**_read(case).execution_result, field: value}
        validate = case.registration.payload_validator.validate_execution_result
    with pytest.raises(TypeError):
        validate(payload)
