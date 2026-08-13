from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest

from control_station.agent_gateway.protocol.contracts import (
    AgentObservationSnapshot,
    AgentSkillRequest,
)
from control_station.real_teleop.agent_observation_gateway import (
    REAL_TCP_POSE_STATE_MAX_MEASUREMENT_AGE_MS_V1,
    RealControlOwnerState,
    RealTcpPoseMeasurementCut,
    RealTcpPoseObserveAttachmentSpec,
    RealTcpPoseOwnerPublication,
    RealTcpPoseOwnerPublicationBuffer,
    RealTcpPoseOwnerPublicationPort,
    RealTcpPosePublishedRead,
    assemble_real_tcp_pose_observe_gateway_session,
    build_real_tcp_pose_observe_contract_bundle,
)
from teleop_runtime.action.legacy_execution_context import (
    LegacyControlOwnerStateTransitionRequest,
)
from teleop_runtime.deploy.rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_manifest,
)
from teleop_runtime.deploy.rm65_execution_evidence import (
    build_rm65_execution_evidence_bundle,
)
from teleop_runtime.deploy.rm65_legacy_worker_binding import (
    build_fake_only_rm65_legacy_worker_binding,
)

COMPONENT_ID = "tool_component"


class _ManualClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _attachment(
    *,
    embodiment_instance_id: str = "embodiment:fixture-alpha",
    component_ids: tuple[str, ...] = (COMPONENT_ID,),
) -> RealTcpPoseObserveAttachmentSpec:
    return RealTcpPoseObserveAttachmentSpec(
        runtime_session_id="real-session:observe-test",
        control_domain_id="real-control-domain:fixture",
        embodiment_instance_id=embodiment_instance_id,
        target_binding_id=f"real-target:{embodiment_instance_id}",
        component_ids=component_ids,
        station_manifest_hash="station-manifest:real:v1",
        execution_manifest_hash="execution-manifest:real:v1",
        backend_inventory_hash="backend-inventory:real:v1",
        source_binding_id="real-owner:cached-measured-state:v1",
        source_binding_hash="source-binding-hash:real:v1",
        observation_binding_id="real-owner:tcp-fk:v1",
        observation_binding_hash="observation-binding-hash:real:v1",
        observation_plan_hash="observation-plan-hash:real:v1",
        observation_capability_hash="observation-capability-hash:real:v1",
        pose_schema_hash="pose-schema-hash:xyz-quat-xyzw:v1",
        reference_frame_id="embodiment_base",
        reference_frame_spec_hash="frame-spec-hash:embodiment-base:v1",
        frame_graph_hash="frame-graph-hash:real:v1",
        tool_config_hash="tool-config-hash:real:v1",
        calibration_hash="calibration-hash:real:v1",
        clock_domain_id="real-owner-monotonic",
        producer_epoch=3,
    )


class _OwnerPublishedSource:
    """Fake existing owner; forbidden calls explode if the Gateway touches them."""

    def __init__(
        self,
        *,
        bundle,
        clock: _ManualClock,
        freshness_ms: int = 10,
        measurement_valid: bool = True,
        summary_overrides: Mapping[str, object] | None = None,
    ) -> None:
        self.bundle = bundle
        self.clock = clock
        self.freshness_ms = freshness_ms
        self.measurement_valid = measurement_valid
        self.summary_overrides = dict(summary_overrides or {})
        self.capture_calls = 0
        self.forbidden_calls = 0

    def start(self) -> None:
        self.forbidden_calls += 1
        raise AssertionError("The observation Gateway must not start a device.")

    def refresh_state_from_hardware(self) -> None:
        self.forbidden_calls += 1
        raise AssertionError("The observation Gateway must not refresh hardware.")

    def command(self) -> None:
        self.forbidden_calls += 1
        raise AssertionError("The observation Gateway must not dispatch commands.")

    def request_hold(self) -> None:
        self.forbidden_calls += 1
        raise AssertionError("The observation Gateway must not request owner HOLD.")

    def capture_published_tcp_pose(
        self,
        *,
        active_operation_id: str | None,
        parameters: Mapping[str, object],
    ) -> RealTcpPosePublishedRead:
        self.capture_calls += 1
        attachment = self.bundle.attachment
        sequence = self.capture_calls
        now = self.clock()
        measured_at = now - self.freshness_ms / 1_000.0
        pose_by_component = {
            component_id: [
                0.41 + index * 0.01,
                -0.12,
                0.38,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
            for index, component_id in enumerate(attachment.component_ids)
        }
        result = {
            "observation_cut_id": f"real-owner-cut:{sequence}",
            "observation_cut_hash": f"real-owner-cut-hash:{sequence}",
            "cutoff_at": now,
            "cutoff_clock_domain_id": attachment.clock_domain_id,
            "component_ids": list(attachment.component_ids),
            "reference_frame_id": attachment.reference_frame_id,
            "reference_frame_spec_hash": attachment.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": attachment.frame_graph_hash,
            "tool_config_hash": attachment.tool_config_hash,
            "calibration_hash": attachment.calibration_hash,
            "measurement_id": f"real-owner-measurement:{sequence}",
            "measurement_observation_hash": f"measurement-hash:{sequence}",
            "source_measurement_hash": f"owner-state-hash:{sequence}",
            "measured_at": measured_at,
            "available_at": now,
            "clock_domain_id": attachment.clock_domain_id,
            "pose_by_component": pose_by_component,
            "pose_schema_hash_by_component": {
                component_id: attachment.pose_schema_hash
                for component_id in attachment.component_ids
            },
            "representation_by_component": {
                component_id: "cartesian_pose"
                for component_id in attachment.component_ids
            },
            "observation_binding_id": attachment.observation_binding_id,
            "observation_plan_hash": attachment.observation_plan_hash,
        }
        summary = {
            "pose_xyz_quat_xyzw_by_component": pose_by_component,
            "measured_at": measured_at,
            "attachment_hash": attachment.semantic_content_hash,
            "embodiment_instance_id": attachment.embodiment_instance_id,
            "source_binding_hash": attachment.source_binding_hash,
            "observation_binding_id": attachment.observation_binding_id,
            "observation_binding_hash": attachment.observation_binding_hash,
            "observation_plan_hash": attachment.observation_plan_hash,
            "observation_capability_hash": (attachment.observation_capability_hash),
        }
        summary.update(self.summary_overrides)
        snapshot = AgentObservationSnapshot(
            snapshot_id=f"real-owner-snapshot:{sequence}",
            runtime_session_id=attachment.runtime_session_id,
            skill_route_manifest_hash=(
                self.bundle.route_manifest.semantic_content_hash
            ),
            station_revision=sequence,
            captured_at=measured_at,
            available_at=now,
            clock_domain_id=attachment.clock_domain_id,
            authority_epoch=5,
            execution_epoch=7,
            dispatch_owner_generation=11,
            lifecycle_state="external_real_runtime_attached",
            execution_gate="not_observed_by_passive_gateway",
            active_operation_id=active_operation_id,
            source_binding_id=attachment.source_binding_id,
            producer_epoch=attachment.producer_epoch,
            target_binding_id=attachment.target_binding_id,
            execution_manifest_hash=attachment.execution_manifest_hash,
            frame_graph_hash=attachment.frame_graph_hash,
            tool_config_hash=attachment.tool_config_hash,
            calibration_hash=attachment.calibration_hash,
            measurement_summary=summary,
            measurement_freshness_ms=self.freshness_ms,
            measurement_valid=self.measurement_valid,
        )
        return RealTcpPosePublishedRead(
            snapshot=snapshot,
            execution_result=result,
            evidence_refs=(f"real-owner-evidence:{sequence}",),
        )


def _request(bundle, snapshot: AgentObservationSnapshot) -> AgentSkillRequest:
    entry = bundle.route_manifest.route("state.read_tcp_pose", "1")
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    return AgentSkillRequest(
        client_id="rpent:real-observe-test",
        request_id="request:real-tcp-pose:1",
        idempotency_key="idem:real-tcp-pose:1",
        client_sequence=1,
        runtime_session_id=bundle.route_manifest.runtime_session_id,
        skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
        expected_station_revision=snapshot.station_revision,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=(entry.deployment_binding.deployment_binding_hash),
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters={
            "component_ids": list(bundle.attachment.component_ids),
            "reference_frame_id": bundle.attachment.reference_frame_id,
            "controlled_frame_id": "tool_tcp",
            "max_measurement_age_ms": (bundle.attachment.max_measurement_age_ms),
        },
        observation_snapshot_id=snapshot.snapshot_id,
        requested_mode="real_observe",
        requested_timeout_ms=500,
    )


def _owner_state(
    bundle,
    *,
    captured_at: float = 9.995,
    authority_epoch: int = 5,
    execution_epoch: int = 7,
    owner_generation: int = 11,
    owner_state_revision: int = 12,
    owner_token_id: str = "legacy-owner-token:fixture",
) -> RealControlOwnerState:
    attachment = bundle.attachment
    return RealControlOwnerState(
        runtime_session_id=attachment.runtime_session_id,
        control_domain_id=attachment.control_domain_id,
        embodiment_instance_id=attachment.embodiment_instance_id,
        target_binding_id=attachment.target_binding_id,
        execution_manifest_hash=attachment.execution_manifest_hash,
        backend_inventory_hash=attachment.backend_inventory_hash,
        clock_domain_id=attachment.clock_domain_id,
        owner_token_id=owner_token_id,
        owner_kind="legacy_control_path",
        owner_generation=owner_generation,
        authority_epoch=authority_epoch,
        execution_epoch=execution_epoch,
        lifecycle_state="lower_owner_running",
        execution_gate="owner_controlled",
        owner_state_revision=owner_state_revision,
        captured_at=captured_at,
        source_owner_state_hash=f"owner-state:{captured_at}",
        evidence_refs=(f"owner-registry-evidence:{captured_at}",),
    )


def _measurement(
    bundle,
    *,
    sequence: int = 1,
    measured_at: float = 9.99,
    available_at: float = 10.0,
) -> RealTcpPoseMeasurementCut:
    attachment = bundle.attachment
    return RealTcpPoseMeasurementCut(
        embodiment_instance_id=attachment.embodiment_instance_id,
        target_binding_id=attachment.target_binding_id,
        clock_domain_id=attachment.clock_domain_id,
        source_binding_hash=attachment.source_binding_hash,
        observation_binding_hash=attachment.observation_binding_hash,
        observation_plan_hash=attachment.observation_plan_hash,
        pose_schema_hash=attachment.pose_schema_hash,
        measurement_id=f"owner-measurement:{sequence}",
        source_measurement_hash=f"measured-state:{sequence}",
        measured_at=measured_at,
        available_at=available_at,
        pose_xyz_quat_xyzw_by_component={
            component_id: (
                0.4 + index * 0.01,
                -0.1,
                0.3,
                0.0,
                0.0,
                0.0,
                1.0,
            )
            for index, component_id in enumerate(attachment.component_ids)
        },
        evidence_refs=(f"measurement-evidence:{sequence}",),
    )


def test_real_tcp_pose_route_is_attach_only_and_read_only() -> None:
    bundle = build_real_tcp_pose_observe_contract_bundle(_attachment())
    manifest = bundle.route_manifest
    route = manifest.route("state.read_tcp_pose", "1").route

    assert manifest.target_kind == "real"
    assert manifest.current_mode == "real_observe"
    assert route.supported_modes == ("real_observe",)
    assert route.control_effect_class == "read_only"
    assert route.privilege_tier == "t0_read_only"
    assert route.may_move_robot is False
    assert "own_domain_hold_v1" not in manifest.supported_transport_features
    assert {
        "device_connection",
        "device_start",
        "direct_hardware_refresh",
        "domain_hold",
        "motion",
        "second_dispatch_owner",
    } <= set(manifest.explicit_exclusions)


def test_contract_factory_is_embodiment_neutral_and_multi_component() -> None:
    attachment = _attachment(
        embodiment_instance_id="embodiment:fixture-beta",
        component_ids=("camera_gimbal", "tool_component"),
    )
    bundle = build_real_tcp_pose_observe_contract_bundle(attachment)
    wire = bundle.route_manifest.to_wire()
    schema = bundle.schema_catalog.entry("state.read_tcp_pose", "1")

    assert bundle.route_manifest.route("state.read_tcp_pose", "1").route.route_id == (
        "real.passive_tcp_pose"
    )
    assert bundle.definition_catalog.definition(
        "state.read_tcp_pose",
        "1",
    ).controlled_component_ids == ("camera_gimbal", "tool_component")
    assert tuple(
        schema.parameter_schema["properties"]["component_ids"]["items"]["enum"]
    ) == ("camera_gimbal", "tool_component")
    assert "rm65" not in str(wire).lower()
    assert "right_arm" not in str(wire).lower()

    same_shape_other_embodiment = build_real_tcp_pose_observe_contract_bundle(
        _attachment(embodiment_instance_id="embodiment:fixture-gamma")
    )
    baseline = build_real_tcp_pose_observe_contract_bundle(_attachment())
    assert (
        same_shape_other_embodiment.definition_catalog.definition(
            "state.read_tcp_pose",
            "1",
        ).semantic_content_hash
        == baseline.definition_catalog.definition(
            "state.read_tcp_pose",
            "1",
        ).semantic_content_hash
    )
    assert (
        same_shape_other_embodiment.deployment_binding.semantic_content_hash
        != baseline.deployment_binding.semantic_content_hash
    )


def test_owner_publication_buffer_never_pulls_backend_state() -> None:
    clock = _ManualClock()
    bundle = build_real_tcp_pose_observe_contract_bundle(_attachment())
    source = RealTcpPoseOwnerPublicationBuffer(
        contract_bundle=bundle,
        clock=clock,
    )
    session = assemble_real_tcp_pose_observe_gateway_session(
        contract_bundle=bundle,
        source=source,
        allowed_client_ids=("rpent:real-observe-test",),
        clock=clock,
    )

    with pytest.raises(RuntimeError, match="No real-owner"):
        session.observation_producer.capture_and_publish(session.service)
    publication = RealTcpPoseOwnerPublication(
        publication_id="real-owner-publication:1",
        attachment_hash=bundle.attachment.semantic_content_hash,
        clock_domain_id=bundle.attachment.clock_domain_id,
        station_revision=1,
        owner_token_id="legacy-owner-token:1",
        owner_kind="legacy_control_path",
        source_owner_state_hash="owner-registry-snapshot:1",
        authority_epoch=5,
        execution_epoch=7,
        dispatch_owner_generation=11,
        source_owner_state_revision=12,
        lifecycle_state="real_owner_running",
        execution_gate="owner_controlled",
        measurement_id="owner-measurement:1",
        source_measurement_hash="owner-state-hash:1",
        measured_at=9.99,
        available_at=10.0,
        pose_xyz_quat_xyzw_by_component={
            COMPONENT_ID: (0.41, -0.12, 0.38, 0.0, 0.0, 0.0, 1.0),
        },
        evidence_refs=("real-owner-state-record:1",),
    )
    source.publish_from_owner(publication)

    outcome = session.observation_producer.capture_and_publish(session.service)
    assert outcome.accepted
    snapshot = session.service.get_observation()
    assert snapshot is not None
    assert snapshot.measurement_summary["owner_publication_hash"] == (
        publication.semantic_content_hash
    )
    assert snapshot.measurement_freshness_ms == 10
    with pytest.raises(ValueError, match="revision did not advance"):
        source.publish_from_owner(publication)
    with pytest.raises(ValueError, match="clock domain mismatch"):
        source.publish_from_owner(
            replace(
                publication,
                station_revision=2,
                clock_domain_id="wall-clock",
            )
        )

    clock.value = 10.3
    with pytest.raises(RuntimeError, match="stale"):
        session.observation_producer.capture_and_publish(session.service)


def test_owner_publication_port_owns_revision_and_binds_exact_lower_facts() -> None:
    clock = _ManualClock()
    bundle = build_real_tcp_pose_observe_contract_bundle(
        _attachment(component_ids=("sensor_head", "tool_component"))
    )
    buffer = RealTcpPoseOwnerPublicationBuffer(
        contract_bundle=bundle,
        clock=clock,
    )
    port = RealTcpPoseOwnerPublicationPort(
        contract_bundle=bundle,
        publication_buffer=buffer,
        clock=clock,
    )

    first_owner = _owner_state(bundle)
    first_measurement = _measurement(bundle)
    first = port.publish(
        owner_state=first_owner,
        measurement=first_measurement,
    )

    assert first.station_revision == 1
    assert first.owner_token_id == first_owner.owner_token_id
    assert first.source_owner_state_hash == first_owner.source_owner_state_hash
    assert first.authority_epoch == first_owner.authority_epoch
    assert first.execution_epoch == first_owner.execution_epoch
    assert first.dispatch_owner_generation == first_owner.owner_generation
    assert first.source_owner_state_revision == first_owner.owner_state_revision
    assert buffer.latest_publication is first
    assert port.no_io is True

    clock.value = 10.02
    second = port.publish(
        owner_state=_owner_state(
            bundle,
            captured_at=10.01,
            authority_epoch=6,
            execution_epoch=8,
            owner_state_revision=13,
        ),
        measurement=_measurement(
            bundle,
            sequence=2,
            measured_at=10.005,
            available_at=10.01,
        ),
    )
    assert second.station_revision == 2

    clock.value = 10.025
    with pytest.raises(ValueError, match="hash changed without a new owner-state"):
        port.publish(
            owner_state=replace(
                _owner_state(
                    bundle,
                    captured_at=10.02,
                    authority_epoch=6,
                    execution_epoch=8,
                    owner_state_revision=13,
                ),
                source_owner_state_hash="different-state-at-same-revision",
            ),
            measurement=_measurement(
                bundle,
                sequence=20,
                measured_at=10.015,
                available_at=10.02,
            ),
        )

    wrong_owner = replace(
        _owner_state(bundle, captured_at=10.02),
        embodiment_instance_id="another-embodiment",
    )
    with pytest.raises(ValueError, match="embodiment_instance_id mismatch"):
        port.publish(
            owner_state=wrong_owner,
            measurement=_measurement(
                bundle,
                sequence=3,
                measured_at=10.01,
                available_at=10.02,
            ),
        )

    clock.value = 10.03
    with pytest.raises(ValueError, match="authority_epoch regressed"):
        port.publish(
            owner_state=_owner_state(
                bundle,
                captured_at=10.02,
                authority_epoch=5,
                execution_epoch=8,
            ),
            measurement=_measurement(
                bundle,
                sequence=4,
                measured_at=10.015,
                available_at=10.02,
            ),
        )


def test_owner_publication_projects_the_same_lower_execution_context() -> None:
    clock = _ManualClock()
    manifest = build_rm65_policy_arm_qpos_candidate_manifest()
    evidence = build_rm65_execution_evidence_bundle(manifest)
    binding = build_fake_only_rm65_legacy_worker_binding(
        evidence_bundle=evidence,
        runtime_session_id="real-session:observe-test",
        control_domain_id="real-control-domain:fixture",
        embodiment_instance_id="embodiment:fixture-alpha",
        clock_domain_id="real-owner-monotonic",
        clock=clock,
    )
    binding.context.transition_owner_state(
        LegacyControlOwnerStateTransitionRequest(
            target_lifecycle_state="shadow_ready",
            target_execution_gate="hold",
            reason="owner_projection_fixture_ready",
        )
    )
    binding.context.transition_owner_state(
        LegacyControlOwnerStateTransitionRequest(
            target_lifecycle_state="running",
            target_execution_gate="ready",
            reason="owner_projection_fixture_running",
        )
    )
    lower_snapshot = binding.context.snapshot()
    attachment = replace(
        _attachment(),
        execution_manifest_hash=lower_snapshot.execution_manifest_hash,
        backend_inventory_hash=lower_snapshot.dispatch_inventory_hash,
    )
    bundle = build_real_tcp_pose_observe_contract_bundle(attachment)
    buffer = RealTcpPoseOwnerPublicationBuffer(
        contract_bundle=bundle,
        clock=clock,
    )
    port = RealTcpPoseOwnerPublicationPort(
        contract_bundle=bundle,
        publication_buffer=buffer,
        clock=clock,
    )

    projected = port.project_lower_owner_state(lower_snapshot)
    publication = port.publish(
        owner_state=projected,
        measurement=_measurement(
            bundle,
            measured_at=10.0,
            available_at=10.0,
        ),
    )

    assert projected.source_owner_state_hash == lower_snapshot.semantic_content_hash
    assert projected.owner_state_revision == lower_snapshot.owner_state_revision
    assert projected.lifecycle_state == "running"
    assert projected.execution_gate == "ready"
    assert publication.source_owner_state_revision == (
        lower_snapshot.owner_state_revision
    )
    assert publication.source_owner_state_hash == lower_snapshot.semantic_content_hash


def test_owner_publication_projects_a_requested_component_subset() -> None:
    clock = _ManualClock()
    bundle = build_real_tcp_pose_observe_contract_bundle(
        _attachment(component_ids=("camera_gimbal", "tool_component"))
    )
    source = RealTcpPoseOwnerPublicationBuffer(
        contract_bundle=bundle,
        clock=clock,
    )
    source.publish_from_owner(
        RealTcpPoseOwnerPublication(
            publication_id="real-owner-publication:multi:1",
            attachment_hash=bundle.attachment.semantic_content_hash,
            clock_domain_id=bundle.attachment.clock_domain_id,
            station_revision=1,
            owner_token_id="legacy-owner-token:multi:1",
            owner_kind="legacy_control_path",
            source_owner_state_hash="owner-registry-snapshot:multi:1",
            authority_epoch=5,
            execution_epoch=7,
            dispatch_owner_generation=11,
            source_owner_state_revision=12,
            lifecycle_state="real_owner_running",
            execution_gate="owner_controlled",
            measurement_id="owner-measurement:multi:1",
            source_measurement_hash="owner-state-hash:multi:1",
            measured_at=9.99,
            available_at=10.0,
            pose_xyz_quat_xyzw_by_component={
                "camera_gimbal": (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0),
                "tool_component": (0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0),
            },
            evidence_refs=("real-owner-state-record:multi:1",),
        )
    )

    read = source.capture_published_tcp_pose(
        active_operation_id=None,
        parameters={
            "component_ids": ["camera_gimbal"],
            "reference_frame_id": bundle.attachment.reference_frame_id,
            "controlled_frame_id": "tool_tcp",
            "max_measurement_age_ms": 200,
        },
    )

    assert read.execution_result["component_ids"] == ["camera_gimbal"]
    assert set(read.execution_result["pose_by_component"]) == {"camera_gimbal"}
    assert set(
        read.snapshot.measurement_summary["pose_xyz_quat_xyzw_by_component"]
    ) == {"camera_gimbal"}


def test_assembly_is_passive_and_one_read_completes_without_control_io() -> None:
    clock = _ManualClock()
    bundle = build_real_tcp_pose_observe_contract_bundle(_attachment())
    source = _OwnerPublishedSource(bundle=bundle, clock=clock)
    session = assemble_real_tcp_pose_observe_gateway_session(
        contract_bundle=bundle,
        source=source,
        allowed_client_ids=("rpent:real-observe-test",),
        clock=clock,
    )
    assert source.capture_calls == 0
    assert source.forbidden_calls == 0

    published = session.observation_producer.capture_and_publish(session.service)
    assert published.accepted
    snapshot = session.service.get_observation()
    assert snapshot is not None
    submitted = session.service.submit_request(
        request=_request(bundle, snapshot),
        current_station_revision=snapshot.station_revision,
        now=clock(),
    )
    assert submitted.accepted
    assert submitted.operation is not None
    operation_id = submitted.operation.operation_id

    launched = session.service.launch_operation(operation_id)
    assert launched.accepted
    assert session.state_executor.wait(operation_id, timeout_sec=1.0)
    completed = session.service.get_operation(operation_id)
    assert completed is not None
    assert completed.status == "completed"
    assert completed.execution_result is not None
    assert list(completed.execution_result["pose_by_component"][COMPONENT_ID]) == [
        0.41,
        -0.12,
        0.38,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert source.capture_calls == 2
    assert source.forbidden_calls == 0

    closed = session.close()
    assert closed.accepted
    assert closed.port_receipt is not None
    assert closed.port_receipt.authoritative_lower_receipt is False
    assert closed.port_receipt.proves_motion_or_hold is False
    assert source.forbidden_calls == 0


@pytest.mark.parametrize(
    ("freshness_ms", "measurement_valid", "match"),
    [
        (
            REAL_TCP_POSE_STATE_MAX_MEASUREMENT_AGE_MS_V1 + 1,
            True,
            "stale",
        ),
        (10, False, "invalid"),
    ],
)
def test_stale_or_invalid_owner_measurement_fails_closed(
    freshness_ms: int,
    measurement_valid: bool,
    match: str,
) -> None:
    clock = _ManualClock()
    bundle = build_real_tcp_pose_observe_contract_bundle(_attachment())
    source = _OwnerPublishedSource(
        bundle=bundle,
        clock=clock,
        freshness_ms=freshness_ms,
        measurement_valid=measurement_valid,
    )
    session = assemble_real_tcp_pose_observe_gateway_session(
        contract_bundle=bundle,
        source=source,
        allowed_client_ids=("rpent:real-observe-test",),
        clock=clock,
    )

    with pytest.raises(RuntimeError, match=match):
        session.observation_producer.capture_and_publish(session.service)
    assert session.service.get_observation() is None
    assert source.forbidden_calls == 0


@pytest.mark.parametrize(
    ("summary_overrides", "match"),
    [
        ({"source_binding_hash": "wrong-source-hash"}, "source_binding_hash"),
        ({"attachment_hash": "wrong-attachment-hash"}, "attachment_hash"),
        (
            {"embodiment_instance_id": "embodiment:wrong-instance"},
            "embodiment_instance_id",
        ),
    ],
)
def test_owner_attachment_identities_are_checked_per_snapshot(
    summary_overrides: Mapping[str, object],
    match: str,
) -> None:
    clock = _ManualClock()
    bundle = build_real_tcp_pose_observe_contract_bundle(_attachment())
    source = _OwnerPublishedSource(
        bundle=bundle,
        clock=clock,
        summary_overrides=summary_overrides,
    )
    session = assemble_real_tcp_pose_observe_gateway_session(
        contract_bundle=bundle,
        source=source,
        allowed_client_ids=("rpent:real-observe-test",),
        clock=clock,
    )

    with pytest.raises(ValueError, match=match + " mismatch"):
        session.observation_producer.capture_and_publish(session.service)
    assert session.service.get_observation() is None
    assert source.forbidden_calls == 0
