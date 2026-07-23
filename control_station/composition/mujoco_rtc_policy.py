"""Scoped RTC-3c emission → immediate-qref MuJoCo data-plane fixture.

The caller drives ``RtcPolicyShadowFrontend.step()`` outside Control Station.
This module accepts only the immutable returned step whose due-pop and
emission-commit evidence identify the exact ``PolicyActionEmission``.  It does
not run inference, merge a timed plan, pop a keyframe, or own the RTC worker.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    BIMANUAL_QPOS_V1,
    ControlExecutionManifest,
    ExecutionProfileExclusion,
    build_control_execution_manifest,
)
from robot_motion.deploy import (
    CanonicalDeployActionPipeline,
    PolicyActionEmission,
    PolicyDeployRuntime,
)
from sim.mujoco.deploy import MujocoDeployRuntime, MujocoPolicyEmissionSubmission
from teleop_runtime.deploy.rtc_frontend import (
    RtcFrontendStep,
    RtcPolicyShadowConfig,
)

from ..contracts import ControlStationRunManifest
from ..publication_barrier import StationPublicationBarrier
from ..runtime import ControlStationRuntime
from ..sources.rtc_policy import (
    RTC_POLICY_BINDING_HASH_DOMAIN_V1,
    RtcPolicyEmissionBinding,
)
from ..targets.authoritative import StationAuthoritativeTargetAdapterBase
from .mujoco_policy import (
    CS2C_POLICY_PRODUCER_EPOCH_HASH_DOMAIN_V1,
    MujocoPolicyOnlyBindingReceipt,
    MujocoPolicyOnlyComposition,
    MujocoPolicyStationSourceAdapter,
    MujocoPolicyStationTargetAdapter,
    ScopedMujocoPolicyStationComposition,
    SyncPolicyEmissionBinding,
    validate_policy_data_plane_binding,
    build_cs2c_sync_policy_mujoco_manifest,
    build_sync_policy_emission_binding,
)


CS2D_RTC_POLICY_MUJOCO_MANIFEST_ID_V1 = (
    "control_station_cs2d_rtc_policy_mujoco_v1"
)
CS2D_RTC_POLICY_MUJOCO_STATION_MANIFEST_ID_V1 = (
    "control_station_cs2d_rtc_policy_mujoco_station_v1"
)
CS2D_RTC_POLICY_EXECUTION_MANIFEST_ID_V1 = (
    "cs2d_rtc_policy_station_immediate_qref_sim_v1"
)
CS2D_RTC_POLICY_BINDING_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs2d_rtc_policy_binding_receipt.v1"
)


def build_rtc_policy_emission_binding(
    *,
    rtc_config: RtcPolicyShadowConfig,
    policy_runtime: PolicyDeployRuntime,
    pipeline: CanonicalDeployActionPipeline,
    target_runtime: MujocoDeployRuntime,
) -> RtcPolicyEmissionBinding:
    """Bind a generic RTC source to this exact MuJoCo target profile."""

    if not isinstance(rtc_config, RtcPolicyShadowConfig):
        raise TypeError("rtc_config must be a RtcPolicyShadowConfig.")
    if rtc_config.profile.send_actions:
        raise ValueError("CS-2d starts from the RTC no-action shadow profile.")
    if rtc_config.profile.position_lookahead_enabled:
        raise ValueError("CS-2d forbids hidden RTC position lookahead.")
    if target_runtime.reference_activation_spec.activation_mode != "immediate_target":
        raise ValueError(
            "CS-2d v1 is immediate-qref only; scheduled target lead is unresolved."
        )
    base = build_sync_policy_emission_binding(
        policy_runtime=policy_runtime,
        pipeline=pipeline,
        target_runtime=target_runtime,
    )
    if (
        rtc_config.control_domain_id != target_runtime.config.control_domain_id
        or rtc_config.producer_id != target_runtime.config.producer_id
        or rtc_config.policy_node_id != base.policy_node_id
        or rtc_config.clock_domain_id != base.policy_clock_domain_id
    ):
        raise ValueError("RTC frontend config targets another control runtime.")
    payload = {
        "source_slot_id": f"rtc-policy-source:{base.producer_id}",
        "producer_id": base.producer_id,
        "producer_epoch": base.producer_epoch,
        "policy_node_id": base.policy_node_id,
        "policy_clock_domain_id": base.policy_clock_domain_id,
        "policy_runtime_config_hash": base.policy_runtime_config_hash,
        "action_adapter_contract_hash": base.action_adapter_contract_hash,
        "rtc_frontend_config_hash": rtc_config.semantic_content_hash,
        "rtc_profile_hash": rtc_config.profile.semantic_content_hash,
        "artifact_hash": rtc_config.artifact_hash,
        "action_codec_hash": rtc_config.action_codec_hash,
        "frame_contract_hash": rtc_config.frame_contract_hash,
        "layout_hash": rtc_config.layout_hash,
        "normalization_hash": rtc_config.normalization_hash,
    }
    return RtcPolicyEmissionBinding(
        **payload,
        semantic_content_hash=canonical_content_hash(
            payload,
            domain=RTC_POLICY_BINDING_HASH_DOMAIN_V1,
        ),
    )


def build_cs2d_rtc_policy_mujoco_manifest(
    runtime: MujocoDeployRuntime,
    binding: RtcPolicyEmissionBinding,
) -> ControlStationRunManifest:
    """Build an exact immediate-qref RTC source manifest without lifecycle."""

    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if not isinstance(binding, RtcPolicyEmissionBinding):
        raise TypeError("binding must be an RtcPolicyEmissionBinding.")
    if runtime.reference_activation_spec.activation_mode != "immediate_target":
        raise ValueError("CS-2d v1 supports immediate-qref MuJoCo only.")
    sync_binding = SyncPolicyEmissionBinding(
        source_slot_id=binding.source_slot_id,
        producer_id=binding.producer_id,
        producer_epoch=binding.producer_epoch,
        policy_node_id=binding.policy_node_id,
        policy_clock_domain_id=binding.policy_clock_domain_id,
        policy_runtime_config_hash=binding.policy_runtime_config_hash,
        action_adapter_contract_hash=binding.action_adapter_contract_hash,
        semantic_content_hash=binding.semantic_content_hash,
    )
    base = build_cs2c_sync_policy_mujoco_manifest(runtime, sync_binding)
    source_slot = replace(
        base.source_slots[0],
        expected_emission_seam=(
            "teleop_runtime.deploy.rtc_frontend.RtcFrontendStep."
            "action_preparation.emission"
        ),
        required_binding_refs=(
            binding.semantic_content_hash,
            binding.policy_runtime_config_hash,
            binding.rtc_frontend_config_hash,
            binding.rtc_profile_hash,
            runtime.catalog.semantic_content_hash,
        ),
        explicit_exclusions=(
            "control_station_policy_merge",
            "control_station_policy_pop",
            "control_station_policy_predict",
            "control_station_rtc_frontend_step",
            "sync_policy_queue_source",
        ),
    )
    producer_epoch_policy_hash = canonical_content_hash(
        {
            "producer_id": binding.producer_id,
            "producer_epoch": binding.producer_epoch,
            "activation": "owned_by_rtc_async_producer_snapshot",
        },
        domain=CS2C_POLICY_PRODUCER_EPOCH_HASH_DOMAIN_V1,
    )
    source_slot = replace(
        source_slot,
        producer_epoch_policy_hash=producer_epoch_policy_hash,
    )
    return replace(
        base,
        manifest_id=CS2D_RTC_POLICY_MUJOCO_MANIFEST_ID_V1,
        source_slots=(source_slot,),
        capture_policy_id="capture_not_owned_by_cs2d_v1",
        explicit_exclusions=tuple(
            sorted(
                {
                    *(
                        value
                        for value in base.explicit_exclusions
                        if value != "rtc_async_policy"
                    ),
                    "formal_episode_lifecycle",
                    "production_entrypoint",
                    "real_motion",
                    "station_motion_authority_lifecycle",
                    "sync_policy_queue_source",
                }
            )
        ),
    )


def build_cs2d_rtc_policy_mujoco_station_manifest(
    runtime: MujocoDeployRuntime,
    binding: RtcPolicyEmissionBinding,
) -> ControlStationRunManifest:
    """Add only a downstream publication lifecycle to the RTC data plane.

    Connect/preflight/candidate operations bind the already-existing RTC
    source contract.  They do not start, step, hold, rearm, or stop its worker.
    """

    base = build_cs2d_rtc_policy_mujoco_manifest(runtime, binding)
    source_slot = replace(
        base.source_slots[0],
        supported_uses=("candidate", "authority"),
        trace_requirement_refs=tuple(
            sorted(
                {
                    *base.source_slots[0].trace_requirement_refs,
                    "station_run_manifest",
                    "station_intent_transition",
                    "station_operation_transition",
                    "station_publication_barrier_transition",
                }
            )
        ),
        explicit_exclusions=tuple(
            sorted(
                {
                    *base.source_slots[0].explicit_exclusions,
                    "control_station_rtc_producer_lifecycle",
                }
            )
        ),
    )
    exclusions = tuple(
        value
        for value in base.explicit_exclusions
        if value != "station_motion_authority_lifecycle"
    )
    return replace(
        base,
        manifest_id=CS2D_RTC_POLICY_MUJOCO_STATION_MANIFEST_ID_V1,
        execution_binding=replace(
            base.execution_binding,
            trace_requirement_profile_id=(
                "deployable_sim_immediate_qref_rtc_policy_station_v1"
            ),
        ),
        source_slots=(source_slot,),
        trace_requirement_profile_id=(
            "deployable_sim_immediate_qref_rtc_policy_station_v1"
        ),
        capture_policy_id="capture_not_owned_by_cs2d_station_v1",
        allowed_intent_kinds=(
            "acquire_control_lease",
            "release_control_lease",
            "connect_source",
            "disconnect_source",
            "connect_target",
            "disconnect_target",
            "run_preflight",
            "request_candidate",
            "release_candidate",
            "request_motion_authority",
            "release_motion_authority",
            "hold",
            "stop_runtime",
        ),
        explicit_exclusions=tuple(
            sorted({*exclusions, "control_station_rtc_producer_lifecycle"})
        ),
    )


def build_cs2d_rtc_policy_execution_manifest(
    runtime: MujocoDeployRuntime,
    station_manifest: ControlStationRunManifest,
) -> ControlExecutionManifest:
    """Freeze the RTC source used by the immediate-qref formal fixture."""

    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if not isinstance(station_manifest, ControlStationRunManifest):
        raise TypeError("station_manifest must be a ControlStationRunManifest.")
    if (
        station_manifest.manifest_id
        != CS2D_RTC_POLICY_MUJOCO_STATION_MANIFEST_ID_V1
    ):
        raise ValueError("RTC execution manifest requires the exact CS2d manifest.")
    if runtime.catalog.action_space != BIMANUAL_QPOS_V1:
        raise ValueError("Formal CS2d capture requires bimanual qpos.")
    if runtime.reference_activation_spec.activation_mode != "immediate_target":
        raise ValueError("Formal CS2d v1 requires immediate-qref activation.")
    if station_manifest.trace_requirement_profile_id != (
        "deployable_sim_immediate_qref_rtc_policy_station_v1"
    ):
        raise ValueError("CS2d Station manifest lacks the RTC formal profile.")

    evidence = (station_manifest.semantic_content_hash,)
    exclusions = tuple(
        ExecutionProfileExclusion.create(
            capability_id=capability_id,
            reason="station_rtc_policy_sim_fixture_only",
            evidence_ids=evidence,
        )
        for capability_id in (
            "control_station_rtc_producer_lifecycle",
            "dataset_replay",
            "hardware_device_initialization",
            "human_motion_proposal",
            "policy_artifact_preflight",
            "production_station_entrypoint",
            "real_backend",
            "scheduled_reference_segment",
        )
    )
    return build_control_execution_manifest(
        manifest_id=CS2D_RTC_POLICY_EXECUTION_MANIFEST_ID_V1,
        manifest_version="v1",
        action_space=runtime.catalog.action_space,
        catalog=runtime.catalog,
        motion_authority_sources=(
            "policy:lower_owned_rtc_emission_via_control_station",
        ),
        human_motion_proposal_status="not_applicable",
        safety_control_sources=("runtime_safety_supervisor",),
        safety_control_requests=("hold",),
        lifecycle_control_sources=("control_station_fixture",),
        lifecycle_control_requests=("start", "stop"),
        capability=runtime.execution_capability,
        activation=runtime.reference_activation_spec,
        actual_reference_frame_producer="not_applicable",
        exclusions=exclusions,
    )


class MujocoRtcPolicyOnlyComposition(MujocoPolicyOnlyComposition):
    """Accept exact lower RTC steps while leaving all producer work outside."""

    binding: RtcPolicyEmissionBinding

    def _validate_exact_binding(self) -> MujocoPolicyOnlyBindingReceipt:
        if self.manifest.manifest_id == CS2D_RTC_POLICY_MUJOCO_MANIFEST_ID_V1:
            expected = build_cs2d_rtc_policy_mujoco_manifest(
                self.runtime,
                self.binding,
            )
            station_authority_lifecycle_bound = False
        elif (
            self.manifest.manifest_id
            == CS2D_RTC_POLICY_MUJOCO_STATION_MANIFEST_ID_V1
        ):
            expected = build_cs2d_rtc_policy_mujoco_station_manifest(
                self.runtime,
                self.binding,
            )
            station_authority_lifecycle_bound = True
        else:
            raise ValueError("Unsupported RTC-policy MuJoCo manifest.")
        return validate_policy_data_plane_binding(
            manifest=self.manifest,
            expected_manifest=expected,
            runtime=self.runtime,
            binding=self.binding,
            station_authority_lifecycle_bound=station_authority_lifecycle_bound,
            receipt_hash_domain=CS2D_RTC_POLICY_BINDING_RECEIPT_HASH_DOMAIN_V1,
            scope_label="CS-2d RTC-policy",
        )

    def submit_emission(self, emission: PolicyActionEmission):
        del emission
        raise RuntimeError("CS-2d requires the complete immutable RtcFrontendStep.")

    def submit_frontend_step(
        self,
        step: RtcFrontendStep,
    ) -> MujocoPolicyEmissionSubmission:
        if not self.started:
            raise RuntimeError("MujocoRtcPolicyOnlyComposition.start() is required.")
        emission = self.binding.validate_frontend_step(step)
        return self.runtime.submit_policy_action_emission(emission)


class MujocoRtcPolicyStationTargetAdapter(MujocoPolicyStationTargetAdapter):
    """RTC-scoped target adapter; it never starts or drives the RTC source."""

    def __init__(self, *, composition, clock, clock_domain_id) -> None:
        StationAuthoritativeTargetAdapterBase.__init__(
            self,
            adapter_id="cs2d-mujoco-target-adapter",
            subject_id=composition.manifest.execution_binding.binding_id,
            supported_operations=(
                "connect_target",
                "disconnect_target",
                "run_preflight",
            ),
            composition=composition,
            clock=clock,
            clock_domain_id=clock_domain_id,
            expected_control_mode="policy",
        )


class MujocoRtcPolicyStationSourceAdapter(MujocoPolicyStationSourceAdapter):
    """Bind an external RTC source without owning its producer lifecycle."""

    def __init__(self, *, composition, clock, clock_domain_id) -> None:
        StationAuthoritativeTargetAdapterBase.__init__(
            self,
            adapter_id="cs2d-existing-rtc-emission-source-adapter",
            subject_id=composition.binding.source_slot_id,
            supported_operations=(
                "connect_source",
                "disconnect_source",
                "run_preflight",
                "request_candidate",
                "release_candidate",
                "request_motion_authority",
                "release_motion_authority",
            ),
            composition=composition,
            clock=clock,
            clock_domain_id=clock_domain_id,
            expected_control_mode="policy",
        )


class ScopedMujocoRtcPolicyStationComposition(
    ScopedMujocoPolicyStationComposition
):
    """Gate exact RTC frontend steps without driving the lower producer."""

    def __init__(self, *, station, data_plane, clock, clock_domain_id) -> None:
        super().__init__(
            station=station,
            data_plane=data_plane,
            clock=clock,
            clock_domain_id=clock_domain_id,
        )
        self.target_adapter = MujocoRtcPolicyStationTargetAdapter(
            composition=data_plane,
            clock=clock,
            clock_domain_id=clock_domain_id,
        )
        self.source_adapter = MujocoRtcPolicyStationSourceAdapter(
            composition=data_plane,
            clock=clock,
            clock_domain_id=clock_domain_id,
        )
        self.publication_barrier = StationPublicationBarrier(
            source_slot_id=self.source_adapter.subject_id
        )

    @property
    def data_plane(self) -> MujocoRtcPolicyOnlyComposition:
        return self._data_plane

    def submit_emission(self, emission: PolicyActionEmission):
        del emission
        raise RuntimeError("CS-2d requires the complete immutable RtcFrontendStep.")

    def submit_frontend_step(
        self,
        step: RtcFrontendStep,
    ) -> MujocoPolicyEmissionSubmission:
        initial = self._station.state
        if (
            initial.lifecycle_state != "running"
            or initial.source_state_by_slot[self.source_adapter.subject_id]
            != "active"
        ):
            raise RuntimeError("Station RTC-policy authority is not active.")

        def submit() -> MujocoPolicyEmissionSubmission:
            state = self._station.state
            if (
                state.lifecycle_state != "running"
                or state.source_state_by_slot[self.source_adapter.subject_id]
                != "active"
            ):
                raise RuntimeError("Station RTC-policy authority is not active.")
            return self._data_plane.submit_frontend_step(step)

        return self.publication_barrier.execute_if_enabled(submit)


def build_scoped_cs2d_rtc_policy_mujoco(
    runtime: MujocoDeployRuntime,
    *,
    rtc_config: RtcPolicyShadowConfig,
    policy_runtime: PolicyDeployRuntime,
    pipeline: CanonicalDeployActionPipeline,
) -> MujocoRtcPolicyOnlyComposition:
    """Build CS-2d without creating or driving an RTC frontend."""

    binding = build_rtc_policy_emission_binding(
        rtc_config=rtc_config,
        policy_runtime=policy_runtime,
        pipeline=pipeline,
        target_runtime=runtime,
    )
    manifest = build_cs2d_rtc_policy_mujoco_manifest(runtime, binding)
    return MujocoRtcPolicyOnlyComposition(
        manifest=manifest,
        runtime=runtime,
        binding=binding,  # type: ignore[arg-type]
    )


def build_scoped_cs2d_rtc_policy_mujoco_station(
    runtime: MujocoDeployRuntime,
    *,
    rtc_config: RtcPolicyShadowConfig,
    policy_runtime: PolicyDeployRuntime,
    pipeline: CanonicalDeployActionPipeline,
    station_clock: Callable[[], float] | None = None,
    station_clock_domain_id: str = "control-station-mujoco-monotonic",
) -> ScopedMujocoRtcPolicyStationComposition:
    """Build the scoped downstream gate without receiving an RTC frontend."""

    binding = build_rtc_policy_emission_binding(
        rtc_config=rtc_config,
        policy_runtime=policy_runtime,
        pipeline=pipeline,
        target_runtime=runtime,
    )
    manifest = build_cs2d_rtc_policy_mujoco_station_manifest(runtime, binding)
    data_plane = MujocoRtcPolicyOnlyComposition(
        manifest=manifest,
        runtime=runtime,
        binding=binding,  # type: ignore[arg-type]
    )
    clock = runtime.clock if station_clock is None else station_clock
    station = ControlStationRuntime(
        manifest=manifest,
        runtime_session_id=runtime.config.runtime_session_id,
        clock_domain_id=station_clock_domain_id,
        clock=clock,
    )
    return ScopedMujocoRtcPolicyStationComposition(
        station=station,
        data_plane=data_plane,
        clock=clock,
        clock_domain_id=station_clock_domain_id,
    )


__all__ = [
    "CS2D_RTC_POLICY_MUJOCO_MANIFEST_ID_V1",
    "CS2D_RTC_POLICY_MUJOCO_STATION_MANIFEST_ID_V1",
    "CS2D_RTC_POLICY_EXECUTION_MANIFEST_ID_V1",
    "MujocoRtcPolicyOnlyComposition",
    "MujocoRtcPolicyStationSourceAdapter",
    "MujocoRtcPolicyStationTargetAdapter",
    "RtcPolicyEmissionBinding",
    "ScopedMujocoRtcPolicyStationComposition",
    "build_cs2d_rtc_policy_mujoco_manifest",
    "build_cs2d_rtc_policy_mujoco_station_manifest",
    "build_cs2d_rtc_policy_execution_manifest",
    "build_rtc_policy_emission_binding",
    "build_scoped_cs2d_rtc_policy_mujoco",
    "build_scoped_cs2d_rtc_policy_mujoco_station",
]
