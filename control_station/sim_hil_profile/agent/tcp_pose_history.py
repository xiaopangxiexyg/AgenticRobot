# Copyright (c) 2026 DexTeleop Contributors
"""MuJoCo passive TCP-pose history adapter over the generic disk substrate.

This adapter is read-only with respect to the robot domain.  It asks the sole
observation producer for one application-locked cut, appends the resulting
portable record to an explicitly injected store, and projects a bounded query
result into the public Agent state schema.  It never starts periodic capture.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from collection_runtime.planning_history import (
    PlanningHistoryRecord,
    PlanningHistorySlice,
    PlanningHistoryStoreError,
    SQLitePlanningHistoryStore,
)
from control_station.agent_gateway import (
    agent_state_history_slice_content_hash_v1,
    validate_agent_state_parameters_v1,
    validate_agent_state_result_v1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

from .motion_deployment import (
    MujocoAgentTcpPoseContractBundle,
    MujocoAgentTcpPoseHistoryProfile,
)
from .observation import (
    MujocoAgentObservationProducer,
    MujocoAgentTcpPoseHistoryCapture,
)

MUJOCO_AGENT_TCP_POSE_HISTORY_SLICE_ID_HASH_DOMAIN_V1 = (
    "dexteleop.agent_state.mujoco_tcp_pose_history_slice_id.v1"
)
_SAMPLE_FIELDS = {
    "measurement_id",
    "measurement_observation_hash",
    "source_measurement_hash",
    "measured_at",
    "available_at",
    "clock_domain_id",
    "pose_by_component",
    "pose_schema_hash_by_component",
    "representation_by_component",
    "observation_binding_id",
    "observation_plan_hash",
}


def _wire_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _wire_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_wire_json(item) for item in value]
    return value


def _result_bytes(result: Mapping[str, Any]) -> int:
    return len(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


@dataclass(frozen=True)
class MujocoAgentTcpPoseHistoryRead:
    """One public history result and its lower-owned evidence references."""

    execution_result: Mapping[str, Any]
    evidence_refs: tuple[str, ...]


class MujocoAgentTcpPoseHistoryProvider:
    """Exact store/source/schema binding for one optional history Route."""

    def __init__(
        self,
        *,
        observation_producer: MujocoAgentObservationProducer,
        contract_bundle: MujocoAgentTcpPoseContractBundle,
        store: SQLitePlanningHistoryStore,
    ) -> None:
        if not isinstance(observation_producer, MujocoAgentObservationProducer):
            raise TypeError("observation_producer must be typed.")
        if not isinstance(contract_bundle, MujocoAgentTcpPoseContractBundle):
            raise TypeError("contract_bundle must be typed.")
        if observation_producer.contract_bundle is not contract_bundle:
            raise ValueError("Observation producer uses another contract bundle.")
        if not isinstance(store, SQLitePlanningHistoryStore):
            raise TypeError("store must be SQLitePlanningHistoryStore.")
        profile = contract_bundle.history_profile
        source_binding = contract_bundle.history_source_binding
        deployment_binding = contract_bundle.history_deployment_binding
        if profile is None:
            raise ValueError("Contract bundle does not enable TCP pose history.")
        if not isinstance(profile, MujocoAgentTcpPoseHistoryProfile):
            raise TypeError("history_profile must be typed.")
        if source_binding is None or deployment_binding is None:
            raise ValueError("History contract bundle is incomplete.")
        if store.binding.semantic_content_hash != source_binding.semantic_content_hash:
            raise ValueError("History store uses another immutable source binding.")
        if store.limits != profile.store_limits:
            raise ValueError("History store limits do not match the bound profile.")
        if deployment_binding.history_source_contract_hash != (
            source_binding.semantic_content_hash
        ):
            raise ValueError("History deployment uses another source contract.")
        self.observation_producer = observation_producer
        self.contract_bundle = contract_bundle
        self.store = store
        self.profile = profile
        self.source_binding = source_binding

    def capture_record(
        self,
        *,
        active_operation_id: str | None,
        boundary_event: str | None = None,
    ) -> PlanningHistoryRecord | None:
        """Return one record for an explicitly owned recorder; do not append it."""

        return self.observation_producer.capture_tcp_pose_history_record(
            active_operation_id=active_operation_id,
            boundary_event=boundary_event,
        )

    def read(
        self,
        *,
        active_operation_id: str,
        parameters: Mapping[str, Any],
    ) -> MujocoAgentTcpPoseHistoryRead:
        """Capture the query cut, durably append it, then read one causal slice."""

        validate_agent_state_parameters_v1(
            schema_kind="tcp_pose_history",
            parameters=parameters,
            allowed_component_ids=("right_arm",),
            limits=self.profile.schema_limits,
        )
        capture = self.observation_producer.capture_tcp_pose_history_sample(
            active_operation_id=active_operation_id,
            boundary_event="history_query",
        )
        if not isinstance(capture, MujocoAgentTcpPoseHistoryCapture):
            raise PlanningHistoryStoreError(
                "A typed passive TCP pose history sample is unavailable."
            )
        self.store.append(capture.history_record)
        observation_cut = capture.observation_cut
        slice_identity = canonical_content_hash(
            {
                "active_operation_id": active_operation_id,
                "parameters": dict(parameters),
                "observation_cut_hash": observation_cut.semantic_content_hash,
                "history_source_contract_hash": (
                    self.source_binding.semantic_content_hash
                ),
            },
            domain=MUJOCO_AGENT_TCP_POSE_HISTORY_SLICE_ID_HASH_DOMAIN_V1,
        )
        history_slice = self.store.query(
            slice_id="agent-tcp-pose-history:" + slice_identity,
            channels=("tcp_pose",),
            cutoff_at=observation_cut.cutoff_at,
            history_window_sec=float(parameters["history_window_ms"]) / 1_000.0,
            max_samples=int(parameters["max_samples"]),
            max_payload_bytes=self.profile.max_result_payload_bytes,
            sampling_policy=self.profile.sampling_policy,
        )
        result = self._project_result(
            capture=capture,
            history_slice=history_slice,
            parameters=parameters,
        )
        return MujocoAgentTcpPoseHistoryRead(
            execution_result=result,
            evidence_refs=(
                observation_cut.observation_cut_id,
                "planning-history-record:"
                + capture.history_record.semantic_content_hash,
                "planning-history-slice:" + history_slice.semantic_content_hash,
            ),
        )

    def _project_result(
        self,
        *,
        capture: MujocoAgentTcpPoseHistoryCapture,
        history_slice: PlanningHistorySlice,
        parameters: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        samples: list[dict[str, Any]] = []
        for stored in history_slice.records:
            sample = _wire_json(stored.record.payload)
            if not isinstance(sample, dict) or set(sample) != _SAMPLE_FIELDS:
                raise PlanningHistoryStoreError(
                    "Stored TCP pose history sample has another projection schema."
                )
            samples.append({"history_sequence": stored.sequence, **sample})
        if not samples:
            raise PlanningHistoryStoreError(
                "TCP pose history query produced no causally available samples."
            )
        truncated = history_slice.truncated
        result = self._result_payload(
            capture=capture,
            history_slice=history_slice,
            samples=samples,
            truncated=truncated,
        )
        while _result_bytes(result) > self.profile.max_result_payload_bytes:
            if len(samples) == 1:
                raise PlanningHistoryStoreError(
                    "One TCP pose history sample exceeds the result payload bound."
                )
            samples.pop(0)
            truncated = True
            result = self._result_payload(
                capture=capture,
                history_slice=history_slice,
                samples=samples,
                truncated=truncated,
            )
        validate_agent_state_result_v1(
            schema_kind="tcp_pose_history",
            parameters=parameters,
            result=result,
            allowed_component_ids=("right_arm",),
            limits=self.profile.schema_limits,
        )
        return result

    def _result_payload(
        self,
        *,
        capture: MujocoAgentTcpPoseHistoryCapture,
        history_slice: PlanningHistorySlice,
        samples: list[dict[str, Any]],
        truncated: bool,
    ) -> dict[str, Any]:
        observation_cut = capture.observation_cut
        binding = self.contract_bundle.deployment_binding
        result: dict[str, Any] = {
            "observation_cut_id": observation_cut.observation_cut_id,
            "observation_cut_hash": observation_cut.semantic_content_hash,
            "cutoff_at": observation_cut.cutoff_at,
            "cutoff_clock_domain_id": observation_cut.cutoff_clock_domain_id,
            "history_source_id": self.source_binding.source_id,
            "history_source_contract_hash": self.source_binding.semantic_content_hash,
            "window_begin_at": history_slice.window_begin_at,
            "window_end_at": history_slice.window_end_at,
            "sample_count": len(samples),
            "component_ids": ["right_arm"],
            "reference_frame_id": self.observation_producer.reference_frame_id,
            "reference_frame_spec_hash": (
                self.observation_producer.reference_frame_spec_hash
            ),
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": binding.frame_graph_hash,
            "tool_config_hash": binding.tool_config_hash,
            "calibration_hash": binding.calibration_hash,
            "samples": samples,
            "truncated": truncated,
        }
        result["history_slice_hash"] = agent_state_history_slice_content_hash_v1(result)
        return result


__all__ = [
    "MUJOCO_AGENT_TCP_POSE_HISTORY_SLICE_ID_HASH_DOMAIN_V1",
    "MujocoAgentTcpPoseHistoryProvider",
    "MujocoAgentTcpPoseHistoryRead",
]
